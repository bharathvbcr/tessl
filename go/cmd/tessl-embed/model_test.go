//go:build darwin && arm64

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/bharathvbcr/tessl/go/embedgemma2"
)

// The server on tessl itself. TestDevice runs on the tiny checkpoint that
// tests/go_fixtures.rs writes (make -C go device; TESSL_GO_REQUIRE_DEVICE turns
// a missing one from a skip into a failure). TestReal runs on the real snapshot
// in EMBEDGEMMA2_SNAPSHOT (make -C go serve-real requires it) against the
// Hugging Face oracle in testdata/hf_reference.json.

func tinySnapshot(t *testing.T) string {
	t.Helper()
	dir := os.Getenv("TESSL_GO_FIXTURES")
	if dir == "" {
		dir = filepath.Join("..", "..", "..", "target", "go_fixtures")
	}
	snap := filepath.Join(dir, "snapshot")
	if _, err := os.Stat(filepath.Join(snap, "config.json")); err != nil {
		if os.Getenv("TESSL_GO_REQUIRE_DEVICE") != "" {
			t.Fatalf("TESSL_GO_REQUIRE_DEVICE is set and the tiny checkpoint is missing: %v", err)
		}
		t.Skipf("no tiny checkpoint (%v); make -C go device writes it", err)
	}
	return snap
}

func realSnapshot(t *testing.T) string {
	t.Helper()
	snap := os.Getenv("EMBEDGEMMA2_SNAPSHOT")
	if snap == "" {
		if os.Getenv("TESSL_EMBED_REQUIRE_REAL") != "" {
			t.Fatal("TESSL_EMBED_REQUIRE_REAL is set and EMBEDGEMMA2_SNAPSHOT is not")
		}
		t.Skip("EMBEDGEMMA2_SNAPSHOT is not set")
	}
	return snap
}

type liveServer struct {
	m   *embedgemma2.Model
	url string
}

func startLive(t *testing.T, snapshot string, lim limits) *liveServer {
	t.Helper()
	m, err := embedgemma2.Load(snapshot, embedgemma2.Options{TextOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := m.Close(); err != nil {
			t.Error(err)
		}
	})
	prompts, err := m.Prompts(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	info, err := snapshotInfo(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	s, err := newServer(context.Background(), m, prompts, lim, info, func(err error) { t.Errorf("fatal: %v", err) })
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(s.handler())
	t.Cleanup(ts.Close)
	return &liveServer{m: m, url: ts.URL}
}

func defaultLimits() limits {
	return limits{MaxBodyBytes: 32 << 20, MaxBatch: embedgemma2.MaxBatch, MaxRequestTokens: 1 << 18, Timeout: 2 * time.Minute, MaxConcurrent: 2, MaxQueue: 64}
}

type embedReply struct {
	Model           string      `json:"model"`
	Embeddings      [][]float32 `json:"embeddings"`
	PromptEvalCount int         `json:"prompt_eval_count"`
	Error           string      `json:"error"`
}

func (l *liveServer) embed(t *testing.T, body any) (int, embedReply) {
	t.Helper()
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(l.url+"/api/embed", "application/json", strings.NewReader(string(b)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var r embedReply
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, r
}

func cosine(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	return ab / math.Sqrt(aa*bb)
}

// sameBits reports the first value where a and b differ bit for bit, or -1.
func sameBits(a, b []float32) int {
	if len(a) != len(b) {
		return 0
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return i
		}
	}
	return -1
}

var parityTexts = []string{
	"Built a Metal GEMM kernel that runs EmbeddingGemma 2 on Apple silicon.",
	"how do I speed up attention?",
	"",
	"日本語のテキストと中文文本",
	"family 👨‍👩‍👧‍👦, flag 🇮🇳",
	"  leading, double  spaces,\ttab\nnewline  ",
}

// checkServerMatchesThePackage asserts that what the server answers over HTTP
// is, bit for bit, what embedgemma2.Model.EmbedTexts returns for the same batch
// under the same prompt: JSON carries float32 exactly, and Tokenize followed by
// Encode is the same computation as EmbedTexts.
func checkServerMatchesThePackage(t *testing.T, l *liveServer) {
	ctx := context.Background()
	for _, tc := range []struct{ name, prompt string }{
		{baseName, ""}, {baseName + ":latest", ""}, {baseName + ":document", "document"}, {baseName + ":query", "query"},
	} {
		for _, dims := range []int{0, l.m.EmbeddingDim() / 2} {
			status, r := l.embed(t, map[string]any{"model": tc.name, "input": parityTexts, "dimensions": dims})
			if status != http.StatusOK {
				t.Fatalf("%s: %d %s", tc.name, status, r.Error)
			}
			want, err := l.m.EmbedTexts(ctx, parityTexts, tc.prompt, dims)
			if err != nil {
				t.Fatal(err)
			}
			if len(r.Embeddings) != len(want) {
				t.Fatalf("%s: %d rows, want %d", tc.name, len(r.Embeddings), len(want))
			}
			for i := range want {
				if j := sameBits(r.Embeddings[i], want[i]); j >= 0 {
					t.Fatalf("%s dims %d row %d differs from EmbedTexts at %d (cosine %.9f)", tc.name, dims, i, j, cosine(r.Embeddings[i], want[i]))
				}
			}
			ids, err := l.m.Tokenize(ctx, parityTexts, tc.prompt)
			if err != nil {
				t.Fatal(err)
			}
			n := 0
			for _, s := range ids {
				n += len(s)
			}
			if r.PromptEvalCount != n {
				t.Fatalf("prompt_eval_count %d, want %d", r.PromptEvalCount, n)
			}
		}
	}
	// One text alone is the same row as in a batch, within the GEMM tiling the
	// model test bounds (docs/embedgemma2.md, Batches).
	status, r := l.embed(t, map[string]any{"model": baseName, "input": parityTexts[0]})
	if status != http.StatusOK {
		t.Fatalf("%d %s", status, r.Error)
	}
	want, _ := l.m.EmbedTexts(ctx, parityTexts, "", 0)
	if c := cosine(r.Embeddings[0], want[0]); c < 0.9999 {
		t.Fatalf("a text alone is at cosine %.9f to the same text in a batch", c)
	}
}

// checkTruncation holds the server's cut to the model's: a long text, truncated
// to MaxTokens, is the row Encode gives for its first MaxTokens-1 ids and <eos>.
func checkTruncation(t *testing.T, l *liveServer, long string) {
	ctx := context.Background()
	ids, err := l.m.Tokenize(ctx, []string{long}, "document")
	if err != nil {
		t.Fatal(err)
	}
	limit := l.m.MaxTokens()
	if len(ids[0]) <= limit {
		t.Fatalf("the long text is %d tokens, not past %d", len(ids[0]), limit)
	}
	eos := ids[0][len(ids[0])-1]
	cut := append(append([]uint32(nil), ids[0][:limit-1]...), eos)
	want, err := l.m.Encode(ctx, [][]uint32{cut}, 0)
	if err != nil {
		t.Fatal(err)
	}
	status, r := l.embed(t, map[string]any{"model": baseName + ":document", "input": []string{long}})
	if status != http.StatusOK {
		t.Fatalf("%d %s", status, r.Error)
	}
	if j := sameBits(r.Embeddings[0], want[0]); j >= 0 {
		t.Fatalf("the truncated row differs from Encode of the cut ids at %d", j)
	}
	if r.PromptEvalCount != limit {
		t.Fatalf("prompt_eval_count %d, want %d", r.PromptEvalCount, limit)
	}
	status, r = l.embed(t, map[string]any{"model": baseName + ":document", "input": []string{long}, "truncate": false})
	if status != http.StatusBadRequest || r.Error != "the input length exceeds the context length" {
		t.Fatalf("truncate:false gave %d %q", status, r.Error)
	}
}

func TestDeviceServerMatchesThePackageBitForBit(t *testing.T) {
	l := startLive(t, tinySnapshot(t), defaultLimits())
	checkServerMatchesThePackage(t, l)
	if err := l.m.SetMaxTokens(context.Background(), 24); err != nil {
		t.Fatal(err)
	}
	checkTruncation(t, l, strings.Repeat("attention is all you need ", 20))
}

type hfCase struct {
	Prompt    string    `json:"prompt"`
	Text      string    `json:"text"`
	IDs       []uint32  `json:"ids"`
	Embedding []float32 `json:"embedding"`
}

func loadHF(t *testing.T) (string, []hfCase) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "hf_reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Revision string   `json:"snapshot_revision"`
		Cases    []hfCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	if len(ref.Cases) == 0 {
		t.Fatal("the reference holds no cases")
	}
	return ref.Revision, ref.Cases
}

// TestRealServerMatchesHuggingFace holds the server, over HTTP, to
// sentence-transformers on the real checkpoint: the token count of every text
// under each prompt equals the ids sentence-transformers feeds the model (and
// Tokenize equals those ids exactly), and every row is within 1e-4 max abs and
// 0.99999 cosine of SentenceTransformer.encode (the bounds of tessl's other
// real-checkpoint tests).
func TestRealServerMatchesHuggingFace(t *testing.T) {
	snap := realSnapshot(t)
	rev, cases := loadHF(t)
	info, err := snapshotInfo(snap)
	if err != nil {
		t.Fatal(err)
	}
	if info.Digest != rev {
		t.Fatalf("the reference is for snapshot %s, EMBEDGEMMA2_SNAPSHOT is %s", rev, info.Digest)
	}
	l := startLive(t, snap, defaultLimits())
	ctx := context.Background()
	byPrompt := map[string][]hfCase{}
	var order []string
	for _, c := range cases {
		if _, ok := byPrompt[c.Prompt]; !ok {
			order = append(order, c.Prompt)
		}
		byPrompt[c.Prompt] = append(byPrompt[c.Prompt], c)
	}
	worstCos, worstAbs, nIDs := 1.0, 0.0, 0
	for _, prompt := range order {
		cs := byPrompt[prompt]
		texts := make([]string, len(cs))
		for i, c := range cs {
			texts[i] = c.Text
		}
		ids, err := l.m.Tokenize(ctx, texts, prompt)
		if err != nil {
			t.Fatal(err)
		}
		name := baseName
		if prompt != "" {
			name += ":" + prompt
		}
		status, r := l.embed(t, map[string]any{"model": name, "input": texts})
		if status != http.StatusOK {
			t.Fatalf("%s: %d %s", name, status, r.Error)
		}
		total := 0
		for i, c := range cs {
			if fmt.Sprint(ids[i]) != fmt.Sprint(c.IDs) {
				t.Fatalf("%s %q: tessl ids %v, Hugging Face %v", name, c.Text, ids[i], c.IDs)
			}
			nIDs += len(c.IDs)
			total += len(c.IDs)
			cos := cosine(r.Embeddings[i], c.Embedding)
			var maxAbs float64
			for j := range c.Embedding {
				maxAbs = math.Max(maxAbs, math.Abs(float64(r.Embeddings[i][j]-c.Embedding[j])))
			}
			worstCos, worstAbs = math.Min(worstCos, cos), math.Max(worstAbs, maxAbs)
			if cos < 0.99999 || maxAbs > 1e-4 {
				t.Fatalf("%s %q: cosine %.9f, max abs %.3g against sentence-transformers", name, c.Text, cos, maxAbs)
			}
		}
		if r.PromptEvalCount != total {
			t.Fatalf("%s: prompt_eval_count %d, Hugging Face's ids are %d", name, r.PromptEvalCount, total)
		}
	}
	t.Logf("%d cases, %d ids equal to Hugging Face's; worst cosine %.9f, worst max abs %.3g", len(cases), nIDs, worstCos, worstAbs)
}

func TestRealServerMatchesThePackageBitForBit(t *testing.T) {
	l := startLive(t, realSnapshot(t), defaultLimits())
	checkServerMatchesThePackage(t, l)
	if err := l.m.SetMaxTokens(context.Background(), 128); err != nil {
		t.Fatal(err)
	}
	checkTruncation(t, l, strings.Repeat("Sliding-window attention bounds the cost of long documents. ", 40))
}
