//go:build darwin && arm64

package embedgemma2_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/bharathvbcr/gusset"

	"github.com/bharathvbcr/tessl/go/embedgemma2"
)

// call is one kind of request the concurrency tests issue, with its serial
// answer to compare against.
type call struct {
	name string
	run  func(ctx context.Context, m *embedgemma2.Model) ([][]float32, error)
}

func calls(t testing.TB, f *fixtures) []call {
	imgs, clips, vids := f.images(t), f.clips(t), f.videos(t)
	return []call{
		{"texts", func(ctx context.Context, m *embedgemma2.Model) ([][]float32, error) {
			return m.EmbedTexts(ctx, f.m.Texts, "query", 0)
		}},
		{"texts-trunc", func(ctx context.Context, m *embedgemma2.Model) ([][]float32, error) {
			return m.EmbedTexts(ctx, f.m.Texts, "document", 40)
		}},
		{"images", func(ctx context.Context, m *embedgemma2.Model) ([][]float32, error) {
			return m.EmbedImages(ctx, imgs, 0)
		}},
		{"audio", func(ctx context.Context, m *embedgemma2.Model) ([][]float32, error) {
			return m.EmbedAudio(ctx, clips, 0)
		}},
		{"video-ts", func(ctx context.Context, m *embedgemma2.Model) ([][]float32, error) {
			return m.EmbedVideos(ctx, vids, true, 0)
		}},
		{"mixed", func(ctx context.Context, m *embedgemma2.Model) ([][]float32, error) {
			return m.EmbedMixed(ctx, [][]embedgemma2.Segment{
				{embedgemma2.Text("look"), embedgemma2.ImageSegment(imgs[0])},
				{embedgemma2.AudioSegment(clips[0]), embedgemma2.Text("listen")},
			}, 0)
		}},
	}
}

func equalBits(a, b [][]float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if math.Float32bits(a[i][j]) != math.Float32bits(b[i][j]) {
				return false
			}
		}
	}
	return true
}

// peakThreads samples gusset.Threads (the Go scheduler's OS thread count,
// which is how gusset's docs measure the bound) until stop is closed.
func peakThreads(stop <-chan struct{}) <-chan int64 {
	out := make(chan int64, 1)
	go func() {
		peak := gusset.Threads()
		for {
			select {
			case <-stop:
				out <- peak
				return
			default:
				if n := gusset.Threads(); n > peak {
					peak = n
				}
				time.Sleep(500 * time.Microsecond)
			}
		}
	}()
	return out
}

func TestDeviceManyGoroutinesGiveTheSerialResultsOnABoundedThreadCount(t *testing.T) {
	f := loadFixtures(t)
	m := sharedModel(t)
	ctx := context.Background()
	cs := calls(t, f)

	// The serial answer for each kind of call.
	want := make([][][]float32, len(cs))
	for i, c := range cs {
		rows, err := c.run(ctx, m)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		want[i] = rows
	}

	// The same total work from 8, 64 and 256 goroutines. Threads are measured as
	// gusset's docs measure them: the Go scheduler's OS thread count while the
	// load runs. The model has one worker, so more goroutines only queue; the
	// count must not follow them (a blocking cgo call per goroutine would add
	// one thread each).
	const totalCalls = 384
	start := gusset.Threads()
	peaks := map[int]int64{}
	for _, goroutines := range []int{8, 64, 256} {
		before := gusset.Threads()
		stop := make(chan struct{})
		peakc := peakThreads(stop)
		var wg sync.WaitGroup
		failures := make(chan string, totalCalls)
		started := time.Now()
		for g := 0; g < goroutines; g++ {
			wg.Add(1)
			go func(g int) {
				defer wg.Done()
				for r := 0; r < totalCalls/goroutines; r++ {
					i := (g + r) % len(cs)
					got, err := cs[i].run(ctx, m)
					switch {
					case err != nil:
						failures <- fmt.Sprintf("goroutine %d %s: %v", g, cs[i].name, err)
					case !equalBits(got, want[i]):
						failures <- fmt.Sprintf("goroutine %d %s: result differs from the serial one", g, cs[i].name)
					}
				}
			}(g)
		}
		wg.Wait()
		elapsed := time.Since(started)
		close(stop)
		peaks[goroutines] = <-peakc
		close(failures)
		for msg := range failures {
			t.Error(msg)
		}
		t.Logf("%3d goroutines, %d calls in %v: OS threads %d before, %d at the peak", goroutines, totalCalls, elapsed.Round(time.Millisecond), before, peaks[goroutines])
	}
	// 256 goroutines cost no more threads than 8 (within runtime noise), and
	// stay far below one thread per goroutine.
	if peaks[256] > peaks[8]+6 {
		t.Errorf("thread peak grew from %d (8 goroutines) to %d (256 goroutines)", peaks[8], peaks[256])
	}
	if peaks[256] > start+32 {
		t.Errorf("thread peak %d under 256 goroutines is too far above the idle %d", peaks[256], start)
	}
}

func TestDeviceTwoModelsRunAtOnceAndEachGivesItsSerialResults(t *testing.T) {
	f := loadFixtures(t)
	ctx := context.Background()
	a, b := loadOwn(t, embedgemma2.Options{}), loadOwn(t, embedgemma2.Options{})
	cs := calls(t, f)
	want := make([][][]float32, len(cs))
	for i, c := range cs {
		rows, err := c.run(ctx, a)
		if err != nil {
			t.Fatal(err)
		}
		want[i] = rows
	}
	var wg sync.WaitGroup
	errs := make(chan error, 64)
	for g := 0; g < 16; g++ {
		m := a
		if g%2 == 1 {
			m = b
		}
		wg.Add(1)
		go func(g int, m *embedgemma2.Model) {
			defer wg.Done()
			for r := 0; r < 4; r++ {
				i := (g + r) % len(cs)
				got, err := cs[i].run(ctx, m)
				if err != nil {
					errs <- fmt.Errorf("%s: %w", cs[i].name, err)
				} else if !equalBits(got, want[i]) {
					errs <- fmt.Errorf("%s: model %d differs from the serial result", cs[i].name, g%2)
				}
			}
		}(g, m)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	// Closing one leaves the other serving.
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if rows, err := cs[0].run(ctx, b); err != nil || !equalBits(rows, want[0]) {
		t.Fatalf("the second model after closing the first: %v", err)
	}
	if _, err := cs[0].run(ctx, a); !errors.Is(err, embedgemma2.ErrClosed) {
		t.Fatalf("the closed model: %v", err)
	}
}

func TestDeviceCancelledAndExpiredCallsNeverCorruptOthers(t *testing.T) {
	f := loadFixtures(t)
	m := sharedModel(t)
	cs := calls(t, f)
	want := make([][][]float32, len(cs))
	for i, c := range cs {
		rows, err := c.run(context.Background(), m)
		if err != nil {
			t.Fatal(err)
		}
		want[i] = rows
	}
	var wg sync.WaitGroup
	var mu sync.Mutex
	var done, left int
	for g := 0; g < 48; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			i := g % len(cs)
			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(g%8)*time.Millisecond)
			defer cancel()
			got, err := cs[i].run(ctx, m)
			mu.Lock()
			defer mu.Unlock()
			switch {
			case err == nil:
				done++
				if !equalBits(got, want[i]) {
					t.Errorf("goroutine %d %s: a call that finished returned a wrong result", g, cs[i].name)
				}
			case errors.Is(err, context.DeadlineExceeded):
				left++
			default:
				t.Errorf("goroutine %d %s: %v", g, cs[i].name, err)
			}
		}(g)
	}
	wg.Wait()
	t.Logf("%d calls finished, %d left at their deadline", done, left)
	// The abandoned work drains, and the model serves normally afterwards.
	for i, c := range cs {
		rows, err := c.run(context.Background(), m)
		if err != nil || !equalBits(rows, want[i]) {
			t.Fatalf("%s after the cancellations: %v", c.name, err)
		}
	}
}

func TestDeviceALargeRequestTravelsInARustBufferAndEmbedsEveryRow(t *testing.T) {
	f := loadFixtures(t)
	var seen embedgemma2.CallInfo
	m := loadOwn(t, embedgemma2.Options{TextOnly: true, Observer: func(ci embedgemma2.CallInfo) { seen = ci }})
	texts := make([]string, embedgemma2.MaxBatch)
	for i := range texts {
		texts[i] = fmt.Sprintf("%s -- item %d %s", f.m.Texts[i%len(f.m.Texts)], i, strings.Repeat("filler ", 12))
	}
	rows, err := m.EmbedTexts(context.Background(), texts, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	if seen.RequestBytes < 8192 {
		t.Fatalf("the request was %d bytes: not past the inline limit, so the buffer path was not taken", seen.RequestBytes)
	}
	if len(rows) != len(texts) {
		t.Fatalf("%d rows for %d texts", len(rows), len(texts))
	}
	for i, r := range rows {
		var sum float64
		for _, v := range r {
			sum += float64(v) * float64(v)
		}
		if math.Abs(sum-1) > 1e-4 {
			t.Fatalf("row %d has squared norm %v, want 1", i, sum)
		}
	}
	again, err := m.EmbedTexts(context.Background(), texts, "", 0)
	if err != nil || !equalBits(rows, again) {
		t.Fatalf("a repeated large call differs: %v", err)
	}
	// 257 texts is past MaxBatch and is refused in Go.
	if _, err := m.EmbedTexts(context.Background(), append(texts, "one more"), "", 0); !errors.Is(err, embedgemma2.ErrInvalidArgument) {
		t.Fatalf("a batch past MaxBatch: %v", err)
	}
}
