//go:build darwin && arm64

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/bharathvbcr/tessl/go/embedgemma2"
)

// The TestUnit tests run the HTTP contract against a fake model: no device, no
// checkpoint (make -C go unit runs them). The device and real tests in
// model_test.go run the same handler on tessl.

const fakeDim = 8

// fakeModel tokenizes a text to <bos>, one id per byte of prompt+text, <eos>,
// and embeds ids to a deterministic unit vector.
type fakeModel struct {
	maxTokens int
	prompts   map[string]string

	mu          sync.Mutex
	lastPrompt  string
	lastIDs     [][]uint32
	encodeCalls int

	tokenizeErr error
	encodeErr   error
	// block, if set, makes Encode wait for it or for its context.
	block   chan struct{}
	entered chan struct{}
	ctxErr  atomic.Pointer[error]
}

func newFake() *fakeModel {
	return &fakeModel{
		maxTokens: 64,
		prompts:   map[string]string{"document": "title: none | text: ", "query": "task: search result | query: ", "Retrieval-query": "task: search result | query: "},
	}
}

func (f *fakeModel) promptSet() embedgemma2.Prompts {
	p := embedgemma2.Prompts{Text: map[string]string{}}
	for _, n := range []string{"document", "query", "Retrieval-query"} {
		p.Names = append(p.Names, n)
		p.Text[n] = f.prompts[n]
	}
	return p
}

func (f *fakeModel) Tokenize(ctx context.Context, texts []string, prompt string) ([][]uint32, error) {
	if f.tokenizeErr != nil {
		return nil, f.tokenizeErr
	}
	f.mu.Lock()
	f.lastPrompt = prompt
	f.mu.Unlock()
	out := make([][]uint32, len(texts))
	for i, t := range texts {
		ids := []uint32{2}
		for _, b := range []byte(f.prompts[prompt] + t) {
			ids = append(ids, uint32(b)+10)
		}
		out[i] = append(ids, 1)
	}
	return out, nil
}

func (f *fakeModel) Encode(ctx context.Context, ids [][]uint32, truncateDim int) ([][]float32, error) {
	f.mu.Lock()
	f.encodeCalls++
	f.lastIDs = ids
	f.mu.Unlock()
	if f.entered != nil {
		f.entered <- struct{}{}
	}
	if f.block != nil {
		select {
		case <-f.block:
		case <-ctx.Done():
			err := ctx.Err()
			f.ctxErr.Store(&err)
			return nil, fmt.Errorf("embedgemma2: Encode: %w", err)
		}
	}
	if f.encodeErr != nil {
		return nil, f.encodeErr
	}
	width := fakeDim
	if truncateDim != 0 {
		width = truncateDim
	}
	rows := make([][]float32, len(ids))
	for i, s := range ids {
		row := make([]float32, width)
		var norm float64
		for j := range row {
			row[j] = float32(len(s)+j+1) / 7
			norm += float64(row[j]) * float64(row[j])
		}
		for j := range row {
			row[j] /= float32(math.Sqrt(norm))
		}
		rows[i] = row
	}
	return rows, nil
}

func (f *fakeModel) EmbeddingDim() int { return fakeDim }
func (f *fakeModel) MaxTokens() int    { return f.maxTokens }

func testLimits() limits {
	return limits{MaxBodyBytes: 1 << 16, MaxBatch: 16, MaxRequestTokens: 4096, Timeout: 5 * time.Second, MaxConcurrent: 2, MaxQueue: 4}
}

type harness struct {
	t     *testing.T
	fake  *fakeModel
	srv   *server
	ts    *httptest.Server
	fatal atomic.Int32
}

func newHarness(t *testing.T, fake *fakeModel, lim limits) *harness {
	t.Helper()
	h := &harness{t: t, fake: fake}
	s, err := newServer(context.Background(), fake, fake.promptSet(), lim, modelInfo{Size: 123, Digest: "rev", ModifiedAt: time.Unix(0, 0).UTC()},
		func(error) { h.fatal.Add(1) })
	if err != nil {
		t.Fatal(err)
	}
	h.srv = s
	h.ts = httptest.NewServer(s.handler())
	t.Cleanup(h.ts.Close)
	return h
}

// post sends body to path and returns the status and the decoded JSON object.
func (h *harness) post(path, body string) (int, map[string]any) {
	h.t.Helper()
	resp, err := http.Post(h.ts.URL+path, "application/json", strings.NewReader(body))
	if err != nil {
		h.t.Fatal(err)
	}
	defer resp.Body.Close()
	return resp.StatusCode, decodeObject(h.t, resp)
}

func decodeObject(t *testing.T, resp *http.Response) map[string]any {
	t.Helper()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Fatalf("Content-Type %q, want JSON", ct)
	}
	var v map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&v); err != nil {
		t.Fatalf("decoding the reply: %v", err)
	}
	return v
}

func embeddings(t *testing.T, v map[string]any) [][]float64 {
	t.Helper()
	raw, ok := v["embeddings"].([]any)
	if !ok {
		t.Fatalf("no embeddings array in %v", v)
	}
	out := make([][]float64, len(raw))
	for i, r := range raw {
		row, ok := r.([]any)
		if !ok {
			t.Fatalf("embedding %d is %T", i, r)
		}
		for _, x := range row {
			out[i] = append(out[i], x.(float64))
		}
	}
	return out
}

func wantError(t *testing.T, status int, v map[string]any, wantStatus int, wantMsg string) {
	t.Helper()
	msg, _ := v["error"].(string)
	if status != wantStatus || !strings.Contains(msg, wantMsg) {
		t.Fatalf("got %d %q, want %d containing %q", status, msg, wantStatus, wantMsg)
	}
}

func TestUnitEmbedAnswersOllamasShape(t *testing.T) {
	h := newHarness(t, newFake(), testLimits())
	for _, tc := range []struct {
		body string
		rows int
	}{
		{`{"model":"embeddinggemma-2-tessl","input":"hello"}`, 1},
		{`{"model":"embeddinggemma-2-tessl","input":["hello","world","x"]}`, 3},
	} {
		status, v := h.post("/api/embed", tc.body)
		if status != http.StatusOK {
			t.Fatalf("%s: %d %v", tc.body, status, v)
		}
		if v["model"] != "embeddinggemma-2-tessl" {
			t.Fatalf("model %v", v["model"])
		}
		rows := embeddings(t, v)
		if len(rows) != tc.rows {
			t.Fatalf("%d rows, want %d", len(rows), tc.rows)
		}
		for _, r := range rows {
			if len(r) != fakeDim {
				t.Fatalf("width %d", len(r))
			}
		}
		for _, k := range []string{"total_duration", "load_duration", "prompt_eval_count"} {
			if _, ok := v[k].(float64); !ok {
				t.Fatalf("%s missing or not a number: %v", k, v[k])
			}
		}
	}
	// prompt_eval_count is the tokens the model was given: <bos> + 5 + <eos>, twice.
	_, v := h.post("/api/embed", `{"model":"embeddinggemma-2-tessl","input":["hello","world"]}`)
	if v["prompt_eval_count"].(float64) != 14 {
		t.Fatalf("prompt_eval_count %v, want 14", v["prompt_eval_count"])
	}
	if v["total_duration"].(float64) <= 0 {
		t.Fatalf("total_duration %v", v["total_duration"])
	}
}

func TestUnitNamesSelectTheSnapshotsPrompts(t *testing.T) {
	fake := newFake()
	h := newHarness(t, fake, testLimits())
	for name, prompt := range map[string]string{
		"embeddinggemma-2-tessl":                 "",
		"embeddinggemma-2-tessl:latest":          "",
		"embeddinggemma-2-tessl:document":        "document",
		"embeddinggemma-2-tessl:query":           "query",
		"embeddinggemma-2-tessl:Retrieval-query": "Retrieval-query",
	} {
		status, v := h.post("/api/embed", fmt.Sprintf(`{"model":%q,"input":"x"}`, name))
		if status != http.StatusOK {
			t.Fatalf("%s: %d %v", name, status, v)
		}
		if v["model"] != name {
			t.Fatalf("model echoed as %v, want %s", v["model"], name)
		}
		fake.mu.Lock()
		got := fake.lastPrompt
		fake.mu.Unlock()
		if got != prompt {
			t.Fatalf("%s applied prompt %q, want %q", name, got, prompt)
		}
	}
	for _, name := range []string{"nomic-embed-text", "embeddinggemma-2", "embeddinggemma-2:270m", "embeddinggemma-2-tessl:nope", "EMBEDDINGGEMMA-2-TESSL", "embeddinggemma-2-tessl:"} {
		status, v := h.post("/api/embed", fmt.Sprintf(`{"model":%q,"input":"x"}`, name))
		wantError(t, status, v, http.StatusNotFound, fmt.Sprintf("model '%s' not found", name))
	}
}

func TestUnitMalformedRequestsAreRefusedWithJSON(t *testing.T) {
	fake := newFake()
	h := newHarness(t, fake, testLimits())
	for body, want := range map[string]string{
		`{"model":"embeddinggemma-2-tessl"}`:                    "input is required",
		`{"model":"embeddinggemma-2-tessl","input":null}`:       "input is required",
		`{"model":"embeddinggemma-2-tessl","input":""}`:         "input is empty",
		`{"model":"embeddinggemma-2-tessl","input":[]}`:         "input is empty",
		`{"model":"embeddinggemma-2-tessl","input":[1,2]}`:      "input must be a string or an array of strings",
		`{"model":"embeddinggemma-2-tessl","input":{"a":1}}`:    "input must be a string or an array of strings",
		`{"model":"embeddinggemma-2-tessl","input":["a",null]}`: "",
		`{"input":"x"}`: "model is required",
		`garbage`:       "invalid character",
		`{"model":"embeddinggemma-2-tessl","input":"x"} {}`:               "more than one JSON value",
		`{"model":"embeddinggemma-2-tessl","input":"x","dimensions":"8"}`: "cannot unmarshal",
	} {
		status, v := h.post("/api/embed", body)
		if body == `{"model":"embeddinggemma-2-tessl","input":["a",null]}` {
			// encoding/json reads a null element of []string as "": a row like any other.
			if status != http.StatusOK || len(embeddings(t, v)) != 2 {
				t.Fatalf("%s: %d %v", body, status, v)
			}
			continue
		}
		wantError(t, status, v, http.StatusBadRequest, want)
	}
	if fake.encodeCalls != 1 {
		t.Fatalf("a refused request reached the model: %d encode calls", fake.encodeCalls)
	}
}

func TestUnitAnEmptyTextInABatchKeepsItsRow(t *testing.T) {
	fake := newFake()
	h := newHarness(t, fake, testLimits())
	status, v := h.post("/api/embed", `{"model":"embeddinggemma-2-tessl","input":["a","","b"]}`)
	if status != http.StatusOK || len(embeddings(t, v)) != 3 {
		t.Fatalf("%d %v", status, v)
	}
	if got := fake.lastIDs[1]; len(got) != 2 || got[0] != 2 || got[1] != 1 {
		t.Fatalf("the empty text is %v, want <bos> <eos>", got)
	}
}

func TestUnitBatchBodyDimensionsAndTokensAreBounded(t *testing.T) {
	lim := testLimits()
	lim.MaxRequestTokens = 40
	h := newHarness(t, newFake(), lim)

	many := make([]string, lim.MaxBatch+1)
	for i := range many {
		many[i] = "x"
	}
	b, _ := json.Marshal(map[string]any{"model": baseName, "input": many})
	status, v := h.post("/api/embed", string(b))
	wantError(t, status, v, http.StatusBadRequest, "input has 17 texts, the limit is 16")

	big := strings.Repeat("y", int(lim.MaxBodyBytes))
	status, v = h.post("/api/embed", fmt.Sprintf(`{"model":%q,"input":%q}`, baseName, big))
	wantError(t, status, v, http.StatusRequestEntityTooLarge, "larger than 65536 bytes")

	status, v = h.post("/api/embed", `{"model":"embeddinggemma-2-tessl","input":"x","dimensions":9}`)
	wantError(t, status, v, http.StatusBadRequest, "dimensions 9 exceeds embedding length 8")
	status, v = h.post("/api/embed", `{"model":"embeddinggemma-2-tessl","input":"x","dimensions":-1}`)
	wantError(t, status, v, http.StatusBadRequest, "must not be negative")
	for dims, width := range map[string]int{`3`: 3, `8`: 8, `0`: 8} {
		status, v = h.post("/api/embed", `{"model":"embeddinggemma-2-tessl","input":"x","dimensions":`+dims+`}`)
		if status != http.StatusOK || len(embeddings(t, v)[0]) != width {
			t.Fatalf("dimensions %s: %d %v", dims, status, v)
		}
	}

	// 3 texts of 15 tokens are 45, past the 40 the request may hold.
	status, v = h.post("/api/embed", `{"model":"embeddinggemma-2-tessl","input":["0123456789abc","0123456789abc","0123456789abc"]}`)
	wantError(t, status, v, http.StatusBadRequest, "the request holds 45 tokens, the limit is 40")
}

func TestUnitLongTextsAreTruncatedKeepingEOSOrRefused(t *testing.T) {
	fake := newFake()
	fake.maxTokens = 10
	h := newHarness(t, fake, testLimits())
	long := strings.Repeat("z", 30) // 32 tokens
	for _, body := range []string{
		fmt.Sprintf(`{"model":%q,"input":[%q,"ok"]}`, baseName, long),
		fmt.Sprintf(`{"model":%q,"input":[%q,"ok"],"truncate":true}`, baseName, long),
	} {
		status, v := h.post("/api/embed", body)
		if status != http.StatusOK {
			t.Fatalf("%d %v", status, v)
		}
		cut := fake.lastIDs[0]
		if len(cut) != 10 || cut[0] != 2 || cut[9] != 1 || cut[8] != 'z'+10 {
			t.Fatalf("truncated to %v, want <bos>, 8 ids of the text, <eos>", cut)
		}
		if len(fake.lastIDs[1]) != 4 {
			t.Fatalf("the short text was changed: %v", fake.lastIDs[1])
		}
		if v["prompt_eval_count"].(float64) != 14 {
			t.Fatalf("prompt_eval_count %v, want the truncated 10 + 4", v["prompt_eval_count"])
		}
	}
	// The prompt counts against the limit too.
	status, v := h.post("/api/embed", fmt.Sprintf(`{"model":"%s:query","input":"short","truncate":false}`, baseName))
	wantError(t, status, v, http.StatusBadRequest, "the input length exceeds the context length")
	status, v = h.post("/api/embed", fmt.Sprintf(`{"model":%q,"input":%q,"truncate":false}`, baseName, long))
	wantError(t, status, v, http.StatusBadRequest, "the input length exceeds the context length")
}

func TestUnitAFullQueueIsRefusedAndATimeoutAnswered(t *testing.T) {
	fake := newFake()
	fake.block = make(chan struct{})
	fake.entered = make(chan struct{}, 4)
	lim := testLimits()
	lim.MaxConcurrent, lim.MaxQueue, lim.Timeout = 1, 0, 300*time.Millisecond
	h := newHarness(t, fake, lim)

	first := make(chan int, 1)
	go func() {
		status, _ := h.post("/api/embed", `{"model":"embeddinggemma-2-tessl","input":"a"}`)
		first <- status
	}()
	<-fake.entered
	status, v := h.post("/api/embed", `{"model":"embeddinggemma-2-tessl","input":"b"}`)
	wantError(t, status, v, http.StatusServiceUnavailable, "server busy")
	// The first never unblocks: its deadline answers it.
	if got := <-first; got != http.StatusGatewayTimeout {
		t.Fatalf("the blocked request got %d, want 504", got)
	}
	if p := fake.ctxErr.Load(); p == nil || !errors.Is(*p, context.DeadlineExceeded) {
		t.Fatalf("the model call did not see the deadline: %v", p)
	}
}

func TestUnitAQueuedRequestWaitsForASlot(t *testing.T) {
	fake := newFake()
	fake.block = make(chan struct{})
	fake.entered = make(chan struct{}, 4)
	lim := testLimits()
	lim.MaxConcurrent, lim.MaxQueue = 1, 1
	h := newHarness(t, fake, lim)
	results := make(chan int, 2)
	for _, in := range []string{"a", "b"} {
		go func() {
			status, _ := h.post("/api/embed", fmt.Sprintf(`{"model":%q,"input":%q}`, baseName, in))
			results <- status
		}()
		if in == "a" {
			<-fake.entered
		}
	}
	// "b" is queued behind "a": it has not reached the model.
	time.Sleep(50 * time.Millisecond)
	select {
	case <-fake.entered:
		t.Fatal("two requests worked at once with MaxConcurrent 1")
	default:
	}
	close(fake.block)
	<-fake.entered
	for range 2 {
		if s := <-results; s != http.StatusOK {
			t.Fatalf("status %d", s)
		}
	}
}

func TestUnitAClientThatLeavesCancelsItsCall(t *testing.T) {
	fake := newFake()
	fake.block = make(chan struct{})
	fake.entered = make(chan struct{}, 1)
	h := newHarness(t, fake, testLimits())
	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, h.ts.URL+"/api/embed", strings.NewReader(`{"model":"embeddinggemma-2-tessl","input":"a"}`))
	done := make(chan error, 1)
	go func() {
		resp, err := http.DefaultClient.Do(req)
		if err == nil {
			resp.Body.Close()
		}
		done <- err
	}()
	<-fake.entered
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("client error %v", err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for fake.ctxErr.Load() == nil && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if p := fake.ctxErr.Load(); p == nil || !errors.Is(*p, context.Canceled) {
		t.Fatalf("the model call did not see the client leave: %v", p)
	}
	// The slot came back.
	close(fake.block)
	if status, v := h.post("/api/embed", `{"model":"embeddinggemma-2-tessl","input":"a"}`); status != http.StatusOK {
		t.Fatalf("%d %v", status, v)
	}
}

func TestUnitModelErrorsMapToStatuses(t *testing.T) {
	fake := newFake()
	h := newHarness(t, fake, testLimits())
	fake.tokenizeErr = &embedgemma2.Error{Op: "Tokenize", Status: embedgemma2.StatusError, Message: "text 0 holds <|image|>"}
	status, v := h.post("/api/embed", `{"model":"embeddinggemma-2-tessl","input":"<|image|>"}`)
	wantError(t, status, v, http.StatusBadRequest, "tokenize: text 0 holds <|image|>")
	fake.tokenizeErr = nil

	fake.encodeErr = &embedgemma2.Error{Op: "Encode", Status: embedgemma2.StatusError, Message: "out of memory"}
	status, v = h.post("/api/embed", `{"model":"embeddinggemma-2-tessl","input":"x"}`)
	wantError(t, status, v, http.StatusInternalServerError, "encode: out of memory")
	if h.fatal.Load() != 0 {
		t.Fatal("a refusal was treated as fatal")
	}

	fake.encodeErr = &embedgemma2.Error{Op: "Encode", Status: embedgemma2.StatusPanic, Message: "boom"}
	status, v = h.post("/api/embed", `{"model":"embeddinggemma-2-tessl","input":"x"}`)
	wantError(t, status, v, http.StatusInternalServerError, "boom")
	if h.fatal.Load() != 1 {
		t.Fatalf("a panic called fatal %d times, want 1", h.fatal.Load())
	}
	fake.encodeErr = nil
	status, v = h.post("/api/embed", `{"model":"embeddinggemma-2-tessl","input":"x"}`)
	wantError(t, status, v, http.StatusServiceUnavailable, "unusable")
	if h.fatal.Load() != 1 {
		t.Fatal("fatal was called again")
	}
}

func TestUnitTagsAndShowDescribeTheServedNames(t *testing.T) {
	h := newHarness(t, newFake(), testLimits())
	resp, err := http.Get(h.ts.URL + "/api/tags")
	if err != nil {
		t.Fatal(err)
	}
	var tags struct {
		Models []tagModel `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	var names []string
	for _, m := range tags.Models {
		names = append(names, m.Name)
		if m.Model != m.Name || m.Details.EmbeddingLength != fakeDim || m.Details.ContextLength != 64 ||
			len(m.Capabilities) != 1 || m.Capabilities[0] != "embedding" || m.Size != 123 || m.Digest != "rev" {
			t.Fatalf("tag %+v", m)
		}
		// DevPrism's looks_like_embedding and first_embedding_model match on "embed".
		if !strings.Contains(strings.ToLower(m.Name), "embed") {
			t.Fatalf("%s would not be picked as an embedding model", m.Name)
		}
	}
	if strings.Join(names, ",") != "embeddinggemma-2-tessl:latest,embeddinggemma-2-tessl:document,embeddinggemma-2-tessl:query" {
		t.Fatalf("listed %v", names)
	}

	status, v := h.post("/api/show", `{"model":"embeddinggemma-2-tessl:query"}`)
	if status != http.StatusOK {
		t.Fatalf("%d %v", status, v)
	}
	caps := v["capabilities"].([]any)
	if len(caps) != 1 || caps[0] != "embedding" {
		t.Fatalf("capabilities %v", caps)
	}
	if v["model_info"].(map[string]any)["embeddinggemma2.context_length"].(float64) != 64 {
		t.Fatalf("model_info %v", v["model_info"])
	}
	if v["prompts"].(map[string]any)["applied"] != "query" {
		t.Fatalf("prompts %v", v["prompts"])
	}
	status, v = h.post("/api/show", `{"name":"nope"}`)
	wantError(t, status, v, http.StatusNotFound, "model 'nope' not found")
}

func TestUnitWrongMethodsAndPathsAnswerJSON(t *testing.T) {
	h := newHarness(t, newFake(), testLimits())
	for _, tc := range []struct {
		method, path string
		status       int
	}{
		{http.MethodGet, "/api/embed", http.StatusMethodNotAllowed},
		{http.MethodPost, "/api/tags", http.StatusMethodNotAllowed},
		{http.MethodGet, "/api/show", http.StatusMethodNotAllowed},
		{http.MethodPost, "/api/embeddings", http.StatusNotFound},
		{http.MethodPost, "/api/chat", http.StatusNotFound},
	} {
		req, _ := http.NewRequest(tc.method, h.ts.URL+tc.path, nil)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		v := decodeObject(t, resp)
		resp.Body.Close()
		if resp.StatusCode != tc.status || v["error"] == nil {
			t.Fatalf("%s %s: %d %v", tc.method, tc.path, resp.StatusCode, v)
		}
	}
}

// TestUnitKbIngestEmbedCallShape replays what kb-ingest-devprism's
// apply.py embed_texts() sends and checks what it reads: POST <base>/api/embed
// with {"model", "input": [...]} and content-type application/json; it reads
// "embeddings", wants one row per text and every value finite. A replica of its
// shape, not a run of apply.py (see model_test.go and go/README.md for that).
func TestUnitKbIngestEmbedCallShape(t *testing.T) {
	h := newHarness(t, newFake(), testLimits())
	texts := []string{"Built a Metal GEMM.", "Led the ingest rewrite.", "日本語 テキスト", "emoji 🧪", "", "x", "y", "z"}
	body, _ := json.Marshal(map[string]any{"model": baseName + ":document", "input": texts})
	req, _ := http.NewRequest(http.MethodPost, h.ts.URL+"/api/embed", bytes.NewReader(body))
	req.Header.Set("content-type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("%d %s", resp.StatusCode, raw)
	}
	var payload struct {
		Embeddings [][]float64 `json:"embeddings"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	if len(payload.Embeddings) != len(texts) {
		t.Fatalf("%d vectors for %d texts", len(payload.Embeddings), len(texts))
	}
	for i, v := range payload.Embeddings {
		for j, x := range v {
			if math.IsNaN(x) || math.IsInf(x, 0) {
				t.Fatalf("embedding %d holds a non-finite value at %d", i, j)
			}
		}
	}
}

func TestUnitNewServerRefusesASnapshotWithoutTheRolePrompts(t *testing.T) {
	fake := newFake()
	p := embedgemma2.Prompts{Names: []string{"query"}, Text: map[string]string{"query": "q: "}}
	if _, err := newServer(context.Background(), fake, p, testLimits(), modelInfo{}, nil); err == nil || !strings.Contains(err.Error(), `no "document" prompt`) {
		t.Fatalf("got %v", err)
	}
	lim := testLimits()
	lim.MaxBatch = embedgemma2.MaxBatch + 1
	if _, err := newServer(context.Background(), fake, fake.promptSet(), lim, modelInfo{}, nil); err == nil {
		t.Fatal("a batch past tessl's MaxBatch was accepted")
	}
}
