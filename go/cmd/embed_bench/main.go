//go:build darwin && arm64

// Command embed_bench times Ollama-protocol embedding servers on the same
// texts: tessl-embed (EmbeddingGemma 2 in f32 on tessl's Metal runtime) against
// Ollama's nomic-embed-text and Ollama's embeddinggemma-2:270m.
//
// Every arm is driven the same way: one HTTP client, POST /api/embed with
// {"model", "input": [...]}, the wall time around the whole request, the reply
// decoded and its row count checked. Rounds rotate which arm runs first; within
// a round each arm runs every workload iters times, one request at a time, after
// a warm-up. The result is the minimum over every call of every round, and the
// median of the rounds' medians (min-of-N, as bench_overhead reads). The host's
// load average is recorded before and after, and the run waits for it to fall
// below -max-load first: a timing from a busy machine measures the machine.
//
// It also reports, as information and not as a gate, the cosine between
// tessl-embed's rows and Ollama's embeddinggemma-2:270m rows for the same raw
// texts: the same model card, different weights' precision (nvfp4 on MLX
// against f32), so near but not equal.
//
//	go run ./cmd/embed_bench -out ../bench/results/tessl_embed_m5pro.json
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strings"
	"time"
)

type arm struct {
	Label string `json:"label"`
	Base  string `json:"base_url"`
	Model string `json:"model"`
	// What the arm runs, for the record.
	Engine string `json:"engine"`
}

type workload struct {
	Name  string
	Batch int
	Words int // words per text
	Iters int
	texts []string
}

// vocabulary for the synthetic career-knowledge-base chunks: deterministic,
// English, technical, and free of anyone's real data.
var vocabulary = strings.Fields(`built led designed shipped measured reduced improved migrated wrote tested
profiled benchmarked refactored deployed owned reviewed scaled automated debugged documented
kernel pipeline service index embedding retrieval latency throughput memory cache queue scheduler
compiler parser tokenizer runtime GPU Metal Rust Go Python SQLite vector search ranking model
training inference evaluation dataset batch request response contract schema migration release
team customers users engineers platform infrastructure observability reliability security cost
by from into across within under over against with for the a an of to and on in at per while
percent faster smaller safer two three four ten hundred thousand million weekly daily quarterly`)

func makeTexts(r *rand.Rand, n, words int) []string {
	out := make([]string, n)
	for i := range out {
		w := make([]string, words)
		for j := range w {
			w[j] = vocabulary[r.IntN(len(vocabulary))]
		}
		w[0] = strings.ToUpper(w[0][:1]) + w[0][1:]
		out[i] = strings.Join(w, " ") + "."
	}
	return out
}

type reply struct {
	Embeddings      [][]float32 `json:"embeddings"`
	TotalDuration   int64       `json:"total_duration"`
	LoadDuration    int64       `json:"load_duration"`
	PromptEvalCount int         `json:"prompt_eval_count"`
	Error           string      `json:"error"`
}

var client = &http.Client{Timeout: 5 * time.Minute}

func call(a arm, texts []string) (time.Duration, reply, error) {
	body, _ := json.Marshal(map[string]any{"model": a.Model, "input": texts, "keep_alive": "60m"})
	t := time.Now()
	resp, err := client.Post(a.Base+"/api/embed", "application/json", bytes.NewReader(body))
	if err != nil {
		return 0, reply{}, err
	}
	raw, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	var r reply
	if err == nil {
		err = json.Unmarshal(raw, &r)
	}
	wall := time.Since(t)
	if err != nil {
		return 0, r, err
	}
	if resp.StatusCode != http.StatusOK {
		return 0, r, fmt.Errorf("%s: HTTP %d: %s", a.Label, resp.StatusCode, r.Error)
	}
	if len(r.Embeddings) != len(texts) {
		return 0, r, fmt.Errorf("%s: %d rows for %d texts", a.Label, len(r.Embeddings), len(texts))
	}
	return wall, r, nil
}

func load1() float64 {
	f := strings.Fields(strings.Trim(sysctl("vm.loadavg"), "{} "))
	var v float64
	if len(f) > 0 {
		fmt.Sscanf(f[0], "%g", &v)
	}
	return v
}

func sysctl(name string) string {
	out, err := exec.Command("sysctl", "-n", name).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func version(base string) string {
	resp, err := client.Get(base + "/api/version")
	if err != nil {
		return "unreachable: " + err.Error()
	}
	defer resp.Body.Close()
	var v struct {
		Version string `json:"version"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&v)
	return v.Version
}

func median(v []float64) float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	n := len(s)
	if n == 0 {
		return math.NaN()
	}
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

func minOf(v []float64) float64 {
	m := math.Inf(1)
	for _, x := range v {
		m = math.Min(m, x)
	}
	return m
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

func r3(x float64) float64 { return math.Round(x*1000) / 1000 }

func fail(err error) {
	fmt.Fprintln(os.Stderr, "embed_bench:", err)
	os.Exit(1)
}

func main() {
	var (
		tesslURL  = flag.String("tessl", "http://127.0.0.1:11435", "tessl-embed base URL")
		ollamaURL = flag.String("ollama", "http://127.0.0.1:11434", "Ollama base URL")
		rounds    = flag.Int("rounds", 9, "rounds; each runs every arm, the order rotating")
		warmup    = flag.Int("warmup", 3, "unmeasured calls per arm and workload before the first round")
		maxLoad   = flag.Float64("max-load", 12, "wait for the 1-minute load average to fall to this before starting, and refuse to write if it is above at the end; 0 turns it off")
		wait      = flag.Duration("wait", 20*time.Minute, "how long to wait for -max-load")
		seed      = flag.Uint64("seed", 20261024, "the texts' seed")
		out       = flag.String("out", "", "write the JSON result here (and the table beside it as .txt)")
	)
	flag.Parse()

	arms := []arm{
		{Label: "tessl-embed", Base: *tesslURL, Model: "embeddinggemma-2-tessl", Engine: "tessl Metal, EmbeddingGemma 2, f32, text encoder only"},
		{Label: "ollama-eg2", Base: *ollamaURL, Model: "embeddinggemma-2:270m", Engine: "Ollama MLX runner, EmbeddingGemma 2, nvfp4"},
		{Label: "ollama-nomic", Base: *ollamaURL, Model: "nomic-embed-text", Engine: "Ollama ggml runner, nomic-embed-text v1.5 (137M), F16"},
	}
	r := rand.New(rand.NewPCG(*seed, 1))
	works := []workload{
		{Name: "query", Batch: 1, Words: 10, Iters: 20},
		{Name: "chunks8", Batch: 8, Words: 120, Iters: 8},
		{Name: "chunks64", Batch: 64, Words: 120, Iters: 2},
	}
	for i := range works {
		works[i].texts = makeTexts(r, works[i].Batch, works[i].Words)
	}

	if *maxLoad > 0 {
		deadline := time.Now().Add(*wait)
		for load1() > *maxLoad {
			if time.Now().After(deadline) {
				fail(fmt.Errorf("the 1-minute load average is %.1f, above -max-load %.1f, after waiting %v", load1(), *maxLoad, *wait))
			}
			time.Sleep(15 * time.Second)
		}
	}
	loadBefore := sysctl("vm.loadavg")

	// Warm-up: loads Ollama's models and tessl's pipelines; checks every arm answers.
	for _, a := range arms {
		for _, w := range works {
			for range *warmup {
				if _, _, err := call(a, w.texts); err != nil {
					fail(err)
				}
			}
		}
	}

	type key struct{ arm, work string }
	walls := map[key][]float64{}   // ms, every call
	medians := map[key][]float64{} // ms, per round
	tokens := map[key]int{}
	maxLoadNS := map[key]int64{}
	serverMin := map[key]float64{}
	for round := range *rounds {
		for k := range arms {
			a := arms[(round+k)%len(arms)]
			for _, w := range works {
				kk := key{a.Label, w.Name}
				var mine []float64
				for range w.Iters {
					wall, rep, err := call(a, w.texts)
					if err != nil {
						fail(err)
					}
					ms := float64(wall.Nanoseconds()) / 1e6
					mine = append(mine, ms)
					tokens[kk] = rep.PromptEvalCount
					maxLoadNS[kk] = max(maxLoadNS[kk], rep.LoadDuration)
					if s := float64(rep.TotalDuration) / 1e6; serverMin[kk] == 0 || s < serverMin[kk] {
						serverMin[kk] = s
					}
				}
				walls[kk] = append(walls[kk], mine...)
				medians[kk] = append(medians[kk], median(mine))
			}
		}
	}
	loadAfter := sysctl("vm.loadavg")

	// Agreement with Ollama's EmbeddingGemma 2 on the same raw texts.
	var agree []map[string]any
	for _, w := range works[:2] {
		_, t, err := call(arms[0], w.texts)
		if err != nil {
			fail(err)
		}
		_, o, err := call(arms[1], w.texts)
		if err != nil {
			fail(err)
		}
		lo, sum := 1.0, 0.0
		for i := range t.Embeddings {
			c := cosine(t.Embeddings[i], o.Embeddings[i])
			lo, sum = math.Min(lo, c), sum+c
		}
		agree = append(agree, map[string]any{
			"workload": w.Name, "rows": len(t.Embeddings),
			"min_cosine": math.Round(lo*1e6) / 1e6, "mean_cosine": math.Round(sum/float64(len(t.Embeddings))*1e6) / 1e6,
			"tessl_tokens": t.PromptEvalCount, "ollama_tokens": o.PromptEvalCount,
		})
	}

	type row struct {
		Arm          string  `json:"arm"`
		Workload     string  `json:"workload"`
		Batch        int     `json:"batch"`
		Tokens       int     `json:"tokens"`
		MinMS        float64 `json:"min_ms"`
		MedianMS     float64 `json:"median_ms"`
		TextsPerSec  float64 `json:"texts_per_s_at_min"`
		TokensPerSec float64 `json:"tokens_per_s_at_min"`
		ServerMinMS  float64 `json:"server_total_duration_min_ms"`
		MaxLoadMS    float64 `json:"max_load_duration_ms"`
		VsTesslMin   float64 `json:"min_ms_over_tessl_min_ms"`
		Samples      int     `json:"samples"`
	}
	var rows []row
	for _, w := range works {
		tmin := minOf(walls[key{arms[0].Label, w.Name}])
		for _, a := range arms {
			kk := key{a.Label, w.Name}
			m := minOf(walls[kk])
			rows = append(rows, row{
				Arm: a.Label, Workload: w.Name, Batch: w.Batch, Tokens: tokens[kk],
				MinMS: r3(m), MedianMS: r3(median(medians[kk])),
				TextsPerSec: math.Round(float64(w.Batch) / (m / 1e3)), TokensPerSec: math.Round(float64(tokens[kk]) / (m / 1e3)),
				ServerMinMS: r3(serverMin[kk]), MaxLoadMS: r3(float64(maxLoadNS[kk]) / 1e6),
				VsTesslMin: r3(m / tmin), Samples: len(walls[kk]),
			})
		}
	}

	var tbl strings.Builder
	fmt.Fprintf(&tbl, "%-13s %-9s %5s %6s %10s %10s %9s %10s %8s\n", "arm", "workload", "batch", "tokens", "min ms", "median ms", "texts/s", "tokens/s", "vs tessl")
	for _, x := range rows {
		fmt.Fprintf(&tbl, "%-13s %-9s %5d %6d %10.2f %10.2f %9.0f %10.0f %8.2f\n", x.Arm, x.Workload, x.Batch, x.Tokens, x.MinMS, x.MedianMS, x.TextsPerSec, x.TokensPerSec, x.VsTesslMin)
	}
	for _, a := range agree {
		fmt.Fprintf(&tbl, "tessl vs ollama-eg2 (%s): min cosine %v, mean %v over %v rows\n", a["workload"], a["min_cosine"], a["mean_cosine"], a["rows"])
	}
	fmt.Fprintf(&tbl, "loadavg before %s, after %s\n", loadBefore, loadAfter)
	fmt.Print(tbl.String())

	if *maxLoad > 0 && load1() > *maxLoad {
		fail(fmt.Errorf("the 1-minute load average rose to %.1f, above -max-load %.1f, during the run: not written", load1(), *maxLoad))
	}
	if *out == "" {
		return
	}
	doc := map[string]any{
		"benchmark": "tessl_embed_vs_ollama",
		"date":      time.Now().UTC().Format(time.RFC3339),
		"method": "Three Ollama-protocol servers embed the same seeded synthetic texts (a vocabulary of career-KB words; no real data): " +
			"one HTTP client, POST /api/embed {model, input}, wall time around the whole request including JSON both ways. " +
			"Rounds rotate the arm order; every arm runs every workload iters times per round, one request at a time, after a warm-up. " +
			"min_ms is the minimum over all calls; median_ms the median of the rounds' medians. tokens is each server's own prompt_eval_count " +
			"(different tokenizers count differently; nomic and EmbeddingGemma are different models). Ollama's arms are sent keep_alive 60m; " +
			"max_load_duration_ms > 0 would mean a model was reloaded during measurement. The cosine rows compare tessl-embed with Ollama's " +
			"embeddinggemma-2:270m on the same raw texts (no prompt on either): information about precision (f32 against nvfp4), not a gate.",
		"host": map[string]any{
			"model": sysctl("hw.model"), "chip": sysctl("machdep.cpu.brand_string"), "memsize": sysctl("hw.memsize"),
			"os": runtime.GOOS, "arch": runtime.GOARCH, "go": runtime.Version(),
			"loadavg_before": loadBefore, "loadavg_after": loadAfter, "max_load_gate": *maxLoad,
		},
		"servers":   map[string]string{"tessl-embed": version(*tesslURL), "ollama": version(*ollamaURL)},
		"arms":      arms,
		"workloads": works,
		"rounds":    *rounds,
		"warmup":    *warmup,
		"seed":      *seed,
		"rows":      rows,
		"agreement": agree,
	}
	raw, err := json.MarshalIndent(doc, "", " ")
	if err != nil {
		fail(err)
	}
	if err := os.WriteFile(*out, append(raw, '\n'), 0o644); err != nil {
		fail(err)
	}
	txt := strings.TrimSuffix(*out, ".json") + ".txt"
	if err := os.WriteFile(txt, []byte(tbl.String()), 0o644); err != nil {
		fail(err)
	}
	fmt.Println("wrote", *out, "and", txt)
}
