//go:build darwin && arm64

// Command bench_overhead measures what the Go binding adds to a tessl call.
//
// Two arms do the same work on the same checkpoint: the Go package, through
// gusset and the umbrella archive, and go/engine's bench_direct, which calls
// tessl's C ABI straight from Rust and times the call alone. A round runs one
// arm and then the other (the order alternates), in separate processes that
// never overlap on the GPU; every workload is run iters times after a warm-up
// in each. The result is the minimum over all rounds and iterations, the
// median of the rounds' medians, and the difference of the minima: min-of-N, the
// way gusset's crossover benchmark reads. Both arms also print a checksum of
// their output, and a workload whose two checksums differ is refused: the arms
// must be doing the same work.
//
// The Go arm also reports the time the engine measured around tessl's call
// (Options.Observer), so the cost can be read two ways: Go wall minus the direct
// arm's, and Go wall minus the same call's own tessl time (paired, per call).
//
//	go run ./cmd/bench_overhead -direct engine/target/release/bench_direct \
//	    -snapshot ../target/go_fixtures/snapshot -out ../bench/results/embedgemma2_go_m5pro.json
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"os/exec"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/bharathvbcr/tessl/go/embedgemma2"
)

const side = 224

type workload struct {
	name  string
	kind  string
	batch int
	run   func(ctx context.Context, m *embedgemma2.Model) ([][]float32, error)
}

func text(i int) string {
	return fmt.Sprintf("bench text %d: %s", i, strings.Repeat("lorem ipsum dolor sit amet ", 3))
}

func image(j int) embedgemma2.Image {
	pix := make([]byte, side*side*3)
	for k := range pix {
		pix[k] = byte((131*k + 17*j) % 251)
	}
	return embedgemma2.Image{Pix: pix, Height: side, Width: side, Channels: 3}
}

func workloads() []workload {
	var ws []workload
	for _, n := range []int{1, 4, 16, 64, 256} {
		texts := make([]string, n)
		for i := range texts {
			texts[i] = text(i)
		}
		ws = append(ws, workload{fmt.Sprintf("texts x%d", n), "texts", n,
			func(ctx context.Context, m *embedgemma2.Model) ([][]float32, error) {
				return m.EmbedTexts(ctx, texts, "", 0)
			}})
	}
	for _, n := range []int{1, 8} {
		images := make([]embedgemma2.Image, n)
		for j := range images {
			images[j] = image(j)
		}
		ws = append(ws, workload{fmt.Sprintf("images x%d", n), "images", n,
			func(ctx context.Context, m *embedgemma2.Model) ([][]float32, error) {
				return m.EmbedImages(ctx, images, 0)
			}})
	}
	return ws
}

func checksum(rows [][]float32) string {
	h := fnv.New64a()
	var b [4]byte
	for _, r := range rows {
		for _, v := range r {
			u := math.Float32bits(v)
			b[0], b[1], b[2], b[3] = byte(u), byte(u>>8), byte(u>>16), byte(u>>24)
			h.Write(b[:])
		}
	}
	return fmt.Sprintf("%016x", h.Sum64())
}

func median(v []float64) float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	return s[len(s)/2]
}

func minOf(v []float64) float64 {
	m := math.Inf(1)
	for _, x := range v {
		m = math.Min(m, x)
	}
	return m
}

func round3(x float64) float64 { return math.Round(x*1000) / 1000 }

type directRow struct {
	Workload string `json:"workload"`
	MinNS    uint64 `json:"min_ns"`
	MedianNS uint64 `json:"median_ns"`
	Iters    int    `json:"iters"`
	Checksum string `json:"checksum"`
}

// directOut holds the direct arm twice: with the model on the process's main
// thread, and on a thread of its own like the engine's worker (the comparison
// that differs from Go in the boundary alone).
type directOut struct {
	Dim    int         `json:"dim"`
	Main   []directRow `json:"main"`
	Worker []directRow `json:"worker"`
}

func runDirect(bin, snapshot string, iters, warmup int) (directOut, error) {
	cmd := exec.Command(bin, "--snapshot", snapshot, "--iters", fmt.Sprint(iters), "--warmup", fmt.Sprint(warmup))
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return directOut{}, fmt.Errorf("%s: %v\n%s", bin, err, stderr.String())
	}
	var out directOut
	if err := json.Unmarshal(stdout.Bytes(), &out); err != nil {
		return directOut{}, fmt.Errorf("%s printed unreadable output: %v", bin, err)
	}
	return out, nil
}

type samples struct {
	goWall, goWork, paired []float64 // microseconds, every iteration of every round
	goMedians              []float64 // one per round
	dirMin                 []float64 // direct arm on a worker-like thread
	dirMedians             []float64
	dirMainMin             []float64 // direct arm on the main thread
	dirMainMedians         []float64
	dirChecksum            string
	goChecksum             string
}

// load1 is the machine's 1-minute load average, from `sysctl vm.loadavg`
// ("{ 1.23 4.56 7.89 }").
func load1() (float64, error) {
	f := strings.Fields(strings.Trim(sysctl("vm.loadavg"), "{} "))
	if len(f) < 1 {
		return 0, fmt.Errorf("vm.loadavg is unreadable")
	}
	var v float64
	if _, err := fmt.Sscanf(f[0], "%g", &v); err != nil {
		return 0, fmt.Errorf("vm.loadavg %q: %v", f[0], err)
	}
	return v, nil
}

// waitIdle blocks until the 1-minute load average is at most max, or fails once
// budget has passed: a timing taken on a busy machine measures the machine.
func waitIdle(max float64, budget time.Duration) error {
	deadline := time.Now().Add(budget)
	for {
		v, err := load1()
		if err != nil {
			return err
		}
		if v <= max {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("the 1-minute load average is %.1f, above the %.1f this benchmark requires, after waiting %v", v, max, budget)
		}
		time.Sleep(15 * time.Second)
	}
}

func sysctl(name string) string {
	out, err := exec.Command("sysctl", "-n", name).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func gussetVersion() string {
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, d := range bi.Deps {
			if d.Path == "github.com/bharathvbcr/gusset" {
				return d.Version
			}
		}
	}
	return "unknown"
}

func main() {
	var (
		snapshot   = flag.String("snapshot", "../target/go_fixtures/snapshot", "the tiny checkpoint go_fixtures writes")
		direct     = flag.String("direct", "engine/target/release/bench_direct", "the direct-arm binary")
		out        = flag.String("out", "../bench/results/embedgemma2_go_m5pro.json", "result file")
		rounds     = flag.Int("rounds", 7, "rounds, each running both arms")
		iters      = flag.Int("iters", 100, "timed calls per workload per round")
		warmup     = flag.Int("warmup", 5, "unmeasured calls per workload per round")
		maxLoad    = flag.Float64("max-load", 0, "wait for the 1-minute load average to be at most this before starting, and refuse to write a result if it is above it at the end (0: do not check)")
		waitBudget = flag.Duration("wait", 20*time.Minute, "how long -max-load may wait for the machine to quiet")
	)
	flag.Parse()
	if *rounds < 1 || *iters < 1 || *warmup < 0 {
		fmt.Fprintln(os.Stderr, "bench_overhead: -rounds and -iters must be positive")
		os.Exit(2)
	}

	var lastWork atomic.Int64
	m, err := embedgemma2.Load(*snapshot, embedgemma2.Options{
		Observer: func(ci embedgemma2.CallInfo) { lastWork.Store(int64(ci.Work)) },
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "bench_overhead:", err)
		os.Exit(1)
	}
	defer m.Close()
	ctx := context.Background()
	ws := workloads()
	if *maxLoad > 0 {
		if err := waitIdle(*maxLoad, *waitBudget); err != nil {
			fmt.Fprintln(os.Stderr, "bench_overhead:", err)
			os.Exit(3)
		}
	}
	loadBefore := sysctl("vm.loadavg")

	acc := make([]samples, len(ws))
	var pingUS, pingMedians []float64

	goArm := func() {
		for i, w := range ws {
			for k := 0; k < *warmup; k++ {
				if _, err := w.run(ctx, m); err != nil {
					fmt.Fprintln(os.Stderr, "bench_overhead:", w.name, err)
					os.Exit(1)
				}
			}
			round := make([]float64, 0, *iters)
			var rows [][]float32
			for k := 0; k < *iters; k++ {
				t0 := time.Now()
				r, err := w.run(ctx, m)
				wall := time.Since(t0)
				if err != nil {
					fmt.Fprintln(os.Stderr, "bench_overhead:", w.name, err)
					os.Exit(1)
				}
				rows = r
				wallUS := float64(wall.Nanoseconds()) / 1e3
				workUS := float64(lastWork.Load()) / 1e3
				acc[i].goWall = append(acc[i].goWall, wallUS)
				acc[i].goWork = append(acc[i].goWork, workUS)
				acc[i].paired = append(acc[i].paired, wallUS-workUS)
				round = append(round, wallUS)
			}
			acc[i].goMedians = append(acc[i].goMedians, median(round))
			acc[i].goChecksum = checksum(rows)
		}
		// The boundary with no work in it.
		for k := 0; k < *warmup; k++ {
			_ = m.Ping(ctx)
		}
		round := make([]float64, 0, 10**iters)
		for k := 0; k < 10**iters; k++ {
			t0 := time.Now()
			if err := m.Ping(ctx); err != nil {
				fmt.Fprintln(os.Stderr, "bench_overhead: ping:", err)
				os.Exit(1)
			}
			round = append(round, float64(time.Since(t0).Nanoseconds())/1e3)
		}
		pingUS = append(pingUS, round...)
		pingMedians = append(pingMedians, median(round))
	}
	directArm := func() {
		d, err := runDirect(*direct, *snapshot, *iters, *warmup)
		if err != nil {
			fmt.Fprintln(os.Stderr, "bench_overhead:", err)
			os.Exit(1)
		}
		for _, set := range []struct {
			name string
			rows []directRow
		}{{"main", d.Main}, {"worker", d.Worker}} {
			if len(set.rows) != len(ws) {
				fmt.Fprintf(os.Stderr, "bench_overhead: the direct arm (%s thread) ran %d workloads, this program %d\n", set.name, len(set.rows), len(ws))
				os.Exit(1)
			}
			for i, r := range set.rows {
				if r.Workload != ws[i].name {
					fmt.Fprintf(os.Stderr, "bench_overhead: workload %d is %q in the direct arm and %q here\n", i, r.Workload, ws[i].name)
					os.Exit(1)
				}
				if set.name == "main" {
					acc[i].dirMainMin = append(acc[i].dirMainMin, float64(r.MinNS)/1e3)
					acc[i].dirMainMedians = append(acc[i].dirMainMedians, float64(r.MedianNS)/1e3)
					if acc[i].dirChecksum != "" && acc[i].dirChecksum != r.Checksum {
						fmt.Fprintf(os.Stderr, "bench_overhead: %s: the direct arm's two threads disagree\n", r.Workload)
						os.Exit(1)
					}
					acc[i].dirChecksum = r.Checksum
					continue
				}
				acc[i].dirMin = append(acc[i].dirMin, float64(r.MinNS)/1e3)
				acc[i].dirMedians = append(acc[i].dirMedians, float64(r.MedianNS)/1e3)
				if acc[i].dirChecksum != r.Checksum {
					fmt.Fprintf(os.Stderr, "bench_overhead: %s: the direct arm's two threads disagree\n", r.Workload)
					os.Exit(1)
				}
			}
		}
	}
	for r := 0; r < *rounds; r++ {
		if r%2 == 0 {
			directArm()
			goArm()
		} else {
			goArm()
			directArm()
		}
		fmt.Fprintf(os.Stderr, "round %d/%d done\n", r+1, *rounds)
	}
	loadAfter := sysctl("vm.loadavg")
	if *maxLoad > 0 {
		// The run itself adds a little load; the bound is the same one it started under.
		if v, err := load1(); err != nil || v > *maxLoad+2 {
			fmt.Fprintf(os.Stderr, "bench_overhead: the machine was not idle at the end of the run (load %s, bound %.1f): nothing written\n", loadAfter, *maxLoad)
			os.Exit(3)
		}
	}

	type row struct {
		Workload      string  `json:"workload"`
		Kind          string  `json:"kind"`
		Batch         int     `json:"batch"`
		DirectMainMin float64 `json:"direct_main_min_us"`
		DirectMainMed float64 `json:"direct_main_median_us"`
		DirectMinUS   float64 `json:"direct_min_us"`
		DirectMedUS   float64 `json:"direct_median_us"`
		GoMinUS       float64 `json:"go_min_us"`
		GoMedUS       float64 `json:"go_median_us"`
		EngineWorkMin float64 `json:"engine_work_min_us"`
		VsDirectMin   float64 `json:"vs_direct_min_us"`
		PairedMin     float64 `json:"paired_overhead_min_us"`
		PairedMed     float64 `json:"paired_overhead_median_us"`
		RatioMin      float64 `json:"ratio_min"`
		Checksum      string  `json:"checksum"`
		SamplesPerArm int     `json:"samples_per_arm"`
	}
	var rows []row
	for i, w := range ws {
		a := acc[i]
		if a.dirChecksum != a.goChecksum {
			fmt.Fprintf(os.Stderr, "bench_overhead: %s: the arms disagree (direct %s, Go %s): they did not do the same work\n", w.name, a.dirChecksum, a.goChecksum)
			os.Exit(1)
		}
		dMin, gMin := minOf(a.dirMin), minOf(a.goWall)
		dMed, gMed := median(a.dirMedians), median(a.goMedians)
		rows = append(rows, row{
			Workload: w.name, Kind: w.kind, Batch: w.batch,
			DirectMainMin: round3(minOf(a.dirMainMin)), DirectMainMed: round3(median(a.dirMainMedians)),
			DirectMinUS: round3(dMin), DirectMedUS: round3(dMed),
			GoMinUS: round3(gMin), GoMedUS: round3(gMed),
			EngineWorkMin: round3(minOf(a.goWork)),
			VsDirectMin:   round3(gMin - dMin),
			PairedMin:     round3(minOf(a.paired)),
			PairedMed:     round3(median(a.paired)),
			RatioMin:      round3(gMin / dMin),
			Checksum:      a.goChecksum, SamplesPerArm: *rounds * *iters,
		})
	}
	doc := map[string]any{
		"benchmark": "embedgemma2_go_overhead",
		"method": "Two arms do the same tessl calls on the tiny multimodal checkpoint: the Go package through gusset " +
			"(wall time around the Go method, input already built) and tessl's C ABI called straight from Rust (time around the call alone), " +
			"with the model on a thread of its own like the engine's worker (direct_*) and on the process's main thread (direct_main_*). " +
			"Rounds alternate which arm runs first; each arm is its own process. Minima are over every call of every round; " +
			"medians are of the rounds' medians. paired_overhead is Go wall minus the engine's own timing of the same call, per call: " +
			"the boundary's cost with the GPU time held identical. vs_direct_min is Go minus direct, which also carries the " +
			"difference between two processes' GPU times and can be negative. Checksums of all arms' outputs are equal.",
		"host": map[string]any{
			"chip": sysctl("machdep.cpu.brand_string"), "os": runtime.GOOS, "arch": runtime.GOARCH,
			"gomaxprocs": runtime.GOMAXPROCS(0), "go": runtime.Version(), "gusset": gussetVersion(),
			"loadavg_before": loadBefore, "loadavg_after": loadAfter,
		},
		"rounds":          *rounds,
		"iters_per_round": *iters,
		"warmup":          *warmup,
		"ping": map[string]any{
			"samples": len(pingUS), "min_us": round3(minOf(pingUS)), "median_us": round3(median(pingMedians)),
		},
		"rows": rows,
	}
	raw, err := json.MarshalIndent(doc, "", " ")
	if err != nil {
		fmt.Fprintln(os.Stderr, "bench_overhead:", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*out, append(raw, '\n'), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "bench_overhead:", err)
		os.Exit(1)
	}
	fmt.Printf("%-12s %12s %12s %12s %12s %10s %8s\n", "workload", "main us", "direct us", "go us", "vs direct us", "paired us", "go/direct")
	for _, r := range rows {
		fmt.Printf("%-12s %12.1f %12.1f %12.1f %12.1f %10.1f %8.3f\n", r.Workload, r.DirectMainMin, r.DirectMinUS, r.GoMinUS, r.VsDirectMin, r.PairedMin, r.RatioMin)
	}
	fmt.Printf("%-12s min %.1f us, median %.1f us (%d calls)\n", "ping (noop)", minOf(pingUS), median(pingMedians), len(pingUS))
	fmt.Println("wrote", *out)
}
