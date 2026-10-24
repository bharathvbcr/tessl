//go:build darwin && arm64

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/bharathvbcr/tessl/go/embedgemma2"
)

// embedder is what the server needs from a model: *embedgemma2.Model in the
// binary, a fake in the unit tests.
type embedder interface {
	Tokenize(ctx context.Context, texts []string, prompt string) ([][]uint32, error)
	Encode(ctx context.Context, ids [][]uint32, truncateDim int) ([][]float32, error)
	EmbeddingDim() int
	MaxTokens() int
}

// baseName is the model name the server answers to. It is not Ollama's
// "embeddinggemma-2": a vector store records the model that made its vectors,
// and Ollama's model of that name is a different build (nvfp4 on MLX), so the
// two must never pass for each other.
const baseName = "embeddinggemma-2-tessl"

// listedTags are the tags /api/tags shows, in this order: a client that picks
// the first embedding model it sees (DevPrism's first_embedding_model) gets the
// untagged name. Every other prompt of the snapshot is served but not listed.
var listedTags = []string{"latest", "document", "query"}

// variant is one served model name: the snapshot's prompt it applies.
type variant struct {
	tag    string
	prompt string // a prompt name of the snapshot; "" for the snapshot's default
}

// limits bounds what one request may ask for.
type limits struct {
	MaxBodyBytes     int64         // the request body
	MaxBatch         int           // inputs per request, at most embedgemma2.MaxBatch
	MaxRequestTokens int           // tokens per request, after truncation
	Timeout          time.Duration // per request, queueing included
	MaxConcurrent    int           // requests working on the model at once
	MaxQueue         int           // requests waiting for a working slot; more are refused
}

// modelInfo is what /api/tags and /api/show report about the snapshot.
type modelInfo struct {
	Size       int64
	Digest     string
	ModifiedAt time.Time
}

type server struct {
	m        embedder
	lim      limits
	info     modelInfo
	variants map[string]variant // by tag
	eosID    uint32
	hasEOS   bool

	work    chan struct{} // MaxConcurrent slots
	waiting atomic.Int64

	// fatal is called once when the model becomes unusable (a tessl panic or a
	// poisoned runtime): the process should exit rather than answer 500 forever.
	fatal     func(error)
	fatalOnce sync.Once
	dead      atomic.Bool
}

// newServer resolves the served names against the snapshot's prompts and reads
// the end-of-sequence id the tokenizer writes, which truncation keeps.
func newServer(ctx context.Context, m embedder, prompts embedgemma2.Prompts, lim limits, info modelInfo, fatal func(error)) (*server, error) {
	if lim.MaxBatch < 1 || lim.MaxBatch > embedgemma2.MaxBatch {
		return nil, fmt.Errorf("max batch %d is outside 1..=%d", lim.MaxBatch, embedgemma2.MaxBatch)
	}
	if lim.MaxConcurrent < 1 || lim.MaxQueue < 0 || lim.MaxBodyBytes < 1 || lim.MaxRequestTokens < 1 || lim.Timeout <= 0 {
		return nil, fmt.Errorf("limits must be positive: %+v", lim)
	}
	s := &server{
		m:        m,
		lim:      lim,
		info:     info,
		variants: map[string]variant{"latest": {tag: "latest"}},
		work:     make(chan struct{}, lim.MaxConcurrent),
		fatal:    fatal,
	}
	for _, name := range prompts.Names {
		if name == "latest" {
			return nil, fmt.Errorf("the snapshot has a prompt named %q, which would shadow the untagged name", name)
		}
		s.variants[name] = variant{tag: name, prompt: name}
	}
	for _, tag := range listedTags {
		if _, ok := s.variants[tag]; !ok {
			return nil, fmt.Errorf("the snapshot has no %q prompt (config_sentence_transformers.json): %s:%s cannot be served", tag, baseName, tag)
		}
	}
	// "" is the two-token sequence <bos> <eos>: its last id is the one a
	// truncated sequence must still end with.
	ids, err := m.Tokenize(ctx, []string{""}, "")
	if err != nil {
		return nil, fmt.Errorf("tokenizing the empty text: %w", err)
	}
	if len(ids) == 1 && len(ids[0]) == 2 {
		s.eosID, s.hasEOS = ids[0][1], true
	}
	return s, nil
}

func (s *server) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/embed", s.handleEmbed)
	mux.HandleFunc("/api/tags", s.handleTags)
	mux.HandleFunc("/api/show", s.handleShow)
	mux.HandleFunc("/api/version", s.handleVersion)
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" && (r.Method == http.MethodGet || r.Method == http.MethodHead) {
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			_, _ = w.Write([]byte("tessl-embed is running"))
			return
		}
		writeError(w, http.StatusNotFound, "404 page not found")
	})
	return mux
}

// lookup maps a requested name to a variant: "embeddinggemma-2-tessl" and
// "embeddinggemma-2-tessl:latest" are the snapshot's default prompt, and
// "embeddinggemma-2-tessl:<prompt>" each named prompt of the snapshot.
func (s *server) lookup(name string) (variant, bool) {
	base, tag, found := strings.Cut(name, ":")
	if base != baseName {
		return variant{}, false
	}
	if !found {
		tag = "latest"
	}
	v, ok := s.variants[tag]
	return v, ok
}

type embedRequest struct {
	Model      string          `json:"model"`
	Input      json.RawMessage `json:"input"`
	Truncate   *bool           `json:"truncate"`
	Dimensions *int            `json:"dimensions"`
	// Accepted and ignored: the model stays loaded and has no sampling options.
	KeepAlive json.RawMessage `json:"keep_alive"`
	Options   json.RawMessage `json:"options"`
}

type embedResponse struct {
	Model           string      `json:"model"`
	Embeddings      [][]float32 `json:"embeddings"`
	TotalDuration   int64       `json:"total_duration"`
	LoadDuration    int64       `json:"load_duration"`
	PromptEvalCount int         `json:"prompt_eval_count"`
}

// httpError is a refusal with the status it is answered with.
type httpError struct {
	status int
	msg    string
}

func (e *httpError) Error() string { return e.msg }

func refuse(status int, format string, args ...any) *httpError {
	return &httpError{status: status, msg: fmt.Sprintf(format, args...)}
}

func (s *server) handleEmbed(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	resp, herr := s.embed(w, r, start)
	if herr != nil {
		if herr.status == 0 {
			// The client went away: nobody reads an answer.
			return
		}
		writeError(w, herr.status, herr.msg)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *server) embed(w http.ResponseWriter, r *http.Request, start time.Time) (*embedResponse, *httpError) {
	if s.dead.Load() {
		return nil, refuse(http.StatusServiceUnavailable, "the model is unusable and the server is shutting down")
	}
	req, herr := s.decode(w, r)
	if herr != nil {
		return nil, herr
	}
	v, ok := s.lookup(req.Model)
	if !ok {
		return nil, refuse(http.StatusNotFound, "model '%s' not found", req.Model)
	}
	texts, herr := parseInput(req.Input, s.lim.MaxBatch)
	if herr != nil {
		return nil, herr
	}
	dim := 0
	if req.Dimensions != nil {
		dim = *req.Dimensions
		switch {
		case dim < 0:
			return nil, refuse(http.StatusBadRequest, "dimensions %d must not be negative", dim)
		case dim > s.m.EmbeddingDim():
			return nil, refuse(http.StatusBadRequest, "dimensions %d exceeds embedding length %d", dim, s.m.EmbeddingDim())
		}
	}
	truncate := req.Truncate == nil || *req.Truncate

	ctx, cancel := context.WithTimeout(r.Context(), s.lim.Timeout)
	defer cancel()
	release, herr := s.acquire(ctx, r.Context())
	if herr != nil {
		return nil, herr
	}
	defer release()

	ids, err := s.m.Tokenize(ctx, texts, v.prompt)
	if err != nil {
		return nil, s.modelError(r.Context(), "tokenize", err, http.StatusBadRequest)
	}
	if len(ids) != len(texts) {
		return nil, s.modelError(r.Context(), "tokenize", fmt.Errorf("%d sequences for %d texts", len(ids), len(texts)), http.StatusInternalServerError)
	}
	ids, tokens, herr := s.fit(ids, truncate)
	if herr != nil {
		return nil, herr
	}
	rows, err := s.m.Encode(ctx, ids, dim)
	if err != nil {
		return nil, s.modelError(r.Context(), "encode", err, http.StatusInternalServerError)
	}
	if len(rows) != len(texts) {
		return nil, refuse(http.StatusInternalServerError, "the model returned %d embeddings for %d inputs", len(rows), len(texts))
	}
	for i, row := range rows {
		for j, x := range row {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				return nil, refuse(http.StatusInternalServerError, "embedding %d holds a non-finite value at %d", i, j)
			}
		}
	}
	return &embedResponse{
		Model:           req.Model,
		Embeddings:      rows,
		TotalDuration:   time.Since(start).Nanoseconds(),
		LoadDuration:    0, // the model is loaded once, at start
		PromptEvalCount: tokens,
	}, nil
}

func (s *server) decode(w http.ResponseWriter, r *http.Request) (*embedRequest, *httpError) {
	body := http.MaxBytesReader(w, r.Body, s.lim.MaxBodyBytes)
	dec := json.NewDecoder(body)
	var req embedRequest
	if err := dec.Decode(&req); err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			return nil, refuse(http.StatusRequestEntityTooLarge, "the request body is larger than %d bytes", s.lim.MaxBodyBytes)
		}
		return nil, refuse(http.StatusBadRequest, "%v", err)
	}
	// One JSON value and nothing after it.
	if dec.More() {
		return nil, refuse(http.StatusBadRequest, "the request body holds more than one JSON value")
	}
	if req.Model == "" {
		return nil, refuse(http.StatusBadRequest, "model is required")
	}
	return &req, nil
}

// parseInput reads Ollama's input: one string or an array of strings. An
// absent, null, empty-string or empty-array input is refused (Ollama answers it
// with no embeddings, as a way to load a model; this server has nothing to
// load). An empty string inside a non-empty array is a text like any other and
// gets its row.
func parseInput(raw json.RawMessage, maxBatch int) ([]string, *httpError) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return nil, refuse(http.StatusBadRequest, "input is required")
	}
	var texts []string
	if strings.HasPrefix(trimmed, "[") {
		if err := json.Unmarshal(raw, &texts); err != nil {
			return nil, refuse(http.StatusBadRequest, "input must be a string or an array of strings: %v", err)
		}
	} else {
		var one string
		if err := json.Unmarshal(raw, &one); err != nil {
			return nil, refuse(http.StatusBadRequest, "input must be a string or an array of strings: %v", err)
		}
		if one == "" {
			return nil, refuse(http.StatusBadRequest, "input is empty")
		}
		texts = []string{one}
	}
	switch {
	case len(texts) == 0:
		return nil, refuse(http.StatusBadRequest, "input is empty")
	case len(texts) > maxBatch:
		return nil, refuse(http.StatusBadRequest, "input has %d texts, the limit is %d per request", len(texts), maxBatch)
	}
	return texts, nil
}

// fit applies the per-sequence limit: a sequence longer than MaxTokens is
// refused, or, with truncate, cut to its first MaxTokens ids, the last of them
// the end-of-sequence id when the sequence ended with it. It also bounds the
// request's total.
func (s *server) fit(ids [][]uint32, truncate bool) ([][]uint32, int, *httpError) {
	limit := s.m.MaxTokens()
	total := 0
	for i, seq := range ids {
		if len(seq) > limit {
			if !truncate {
				return nil, 0, refuse(http.StatusBadRequest, "the input length exceeds the context length")
			}
			cut := make([]uint32, limit)
			copy(cut, seq[:limit])
			if s.hasEOS && seq[len(seq)-1] == s.eosID {
				cut[limit-1] = s.eosID
			}
			ids[i] = cut
		}
		total += len(ids[i])
	}
	if total > s.lim.MaxRequestTokens {
		return nil, 0, refuse(http.StatusBadRequest, "the request holds %d tokens, the limit is %d per request", total, s.lim.MaxRequestTokens)
	}
	return ids, total, nil
}

// acquire takes a working slot, waiting behind at most MaxQueue others.
func (s *server) acquire(ctx, client context.Context) (func(), *httpError) {
	select {
	case s.work <- struct{}{}:
		return func() { <-s.work }, nil
	default:
	}
	if s.waiting.Add(1) > int64(s.lim.MaxQueue) {
		s.waiting.Add(-1)
		return nil, refuse(http.StatusServiceUnavailable, "server busy: %d requests are already waiting", s.lim.MaxQueue)
	}
	defer s.waiting.Add(-1)
	select {
	case s.work <- struct{}{}:
		return func() { <-s.work }, nil
	case <-ctx.Done():
		return nil, s.ctxError(client)
	}
}

// ctxError answers a request whose context ended: nothing when the client
// left, a timeout otherwise.
func (s *server) ctxError(client context.Context) *httpError {
	if client.Err() != nil {
		return &httpError{status: 0, msg: "client went away"}
	}
	return refuse(http.StatusGatewayTimeout, "the request took longer than %v", s.lim.Timeout)
}

// modelError maps a model call's error. inputStatus is what a tessl refusal is
// answered with: the caller's fault when tokenizing, the server's when encoding
// ids the server itself checked.
func (s *server) modelError(client context.Context, op string, err error, inputStatus int) *httpError {
	switch {
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return s.ctxError(client)
	case errors.Is(err, embedgemma2.ErrPanic), errors.Is(err, embedgemma2.ErrPoisoned),
		errors.Is(err, embedgemma2.ErrNoModel), errors.Is(err, embedgemma2.ErrClosed):
		s.dead.Store(true)
		s.fatalOnce.Do(func() {
			if s.fatal != nil {
				s.fatal(err)
			}
		})
		return refuse(http.StatusInternalServerError, "%s: %v", op, err)
	case errors.Is(err, embedgemma2.ErrInvalidArgument):
		return refuse(http.StatusBadRequest, "%s: %v", op, err)
	}
	var te *embedgemma2.Error
	if errors.As(err, &te) && te.Status == embedgemma2.StatusError {
		return refuse(inputStatus, "%s: %s", op, te.Message)
	}
	return refuse(http.StatusInternalServerError, "%s: %v", op, err)
}

type tagModel struct {
	Name         string     `json:"name"`
	Model        string     `json:"model"`
	ModifiedAt   time.Time  `json:"modified_at"`
	Size         int64      `json:"size"`
	Digest       string     `json:"digest"`
	Details      tagDetails `json:"details"`
	Capabilities []string   `json:"capabilities"`
}

type tagDetails struct {
	ParentModel       string   `json:"parent_model"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
	ContextLength     int      `json:"context_length"`
	EmbeddingLength   int      `json:"embedding_length"`
	Runner            string   `json:"runner"`
}

func (s *server) details() tagDetails {
	return tagDetails{
		Format:            "safetensors",
		Family:            "embeddinggemma2",
		Families:          []string{"embeddinggemma2"},
		QuantizationLevel: "F32",
		ContextLength:     s.m.MaxTokens(),
		EmbeddingLength:   s.m.EmbeddingDim(),
		Runner:            "tessl",
	}
}

func (s *server) tag(t string) tagModel {
	name := baseName + ":" + t
	return tagModel{
		Name: name, Model: name,
		ModifiedAt:   s.info.ModifiedAt,
		Size:         s.info.Size,
		Digest:       s.info.Digest,
		Details:      s.details(),
		Capabilities: []string{"embedding"},
	}
}

func (s *server) handleTags(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	models := make([]tagModel, 0, len(listedTags))
	for _, t := range listedTags {
		models = append(models, s.tag(t))
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": models})
}

func (s *server) handleShow(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	var req struct {
		Model string `json:"model"`
		Name  string `json:"name"`
	}
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10))
	if err := dec.Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	name := req.Model
	if name == "" {
		name = req.Name
	}
	v, ok := s.lookup(name)
	if !ok {
		writeError(w, http.StatusNotFound, fmt.Sprintf("model '%s' not found", name))
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"details":      s.details(),
		"capabilities": []string{"embedding"},
		"modified_at":  s.info.ModifiedAt,
		"model_info": map[string]any{
			"general.architecture":             "embeddinggemma2",
			"embeddinggemma2.context_length":   s.m.MaxTokens(),
			"embeddinggemma2.embedding_length": s.m.EmbeddingDim(),
		},
		"prompts": s.promptsOf(v),
	})
}

// promptsOf reports which snapshot prompt a name applies, and every tag served.
func (s *server) promptsOf(v variant) map[string]any {
	tags := make([]string, 0, len(s.variants))
	for t := range s.variants {
		tags = append(tags, t)
	}
	sort.Strings(tags)
	return map[string]any{"applied": v.prompt, "served_tags": tags}
}

func (s *server) handleVersion(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"version": version})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_, _ = w.Write(b)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	b, _ := json.Marshal(map[string]string{"error": msg})
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_, _ = w.Write(b)
}
