//go:build darwin && arm64

package embedgemma2

import (
	"context"
	"errors"
	"math/rand/v2"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/bharathvbcr/tessl/go/embedgemma2/internal/wire"
)

// Inline path (no Done channel): a panic releases the turn and the model works.
func TestUnitAdversarialSetMaxTokensInlinePanicReleasesTurn(t *testing.T) {
	m := bare()
	m.maxTokens.Store(8192)
	var boom atomic.Bool
	boom.Store(true)
	m.send = func(context.Context, string, uint32, *wire.Frame) (wire.Reply, error) {
		if boom.Load() {
			panic("observer exploded")
		}
		return wire.Reply{Status: wire.StatusOK}, nil
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Error("inline panic was swallowed")
			}
		}()
		_ = m.SetMaxTokens(context.Background(), 6)
	}()
	boom.Store(false)
	done := make(chan error, 1)
	go func() { done <- m.SetMaxTokens(context.Background(), 9) }()
	select {
	case err := <-done:
		if err != nil || m.MaxTokens() != 9 {
			t.Fatalf("after inline panic: err=%v MaxTokens=%d", err, m.MaxTokens())
		}
	case <-time.After(3 * time.Second):
		t.Fatal("turn leaked by the inline panic")
	}
}

// The caller's context ends from inside the engine call, the instant before the
// reply: the helper delivering and the caller abandoning race on every
// iteration. Either outcome is legal; hanging, and a cache that does not follow
// the engine, are not. A nil return must mean the cache held n at that point
// or was overwritten only by a later call (none here: one caller).
func TestUnitAdversarialSetMaxTokensHandoffRaceNeverHangsOrDesyncs(t *testing.T) {
	iters := 10000
	if testing.Short() {
		iters = 500
	}
	e := &fakeEngine{}
	m := modelOver(e)
	var nils, ctxErrs atomic.Int64
	var cancelNow atomic.Pointer[context.CancelFunc]
	m.send = func(ctx context.Context, op string, c uint32, f *wire.Frame) (wire.Reply, error) {
		r, err := e.send(ctx, op, c, f)
		if cf := cancelNow.Load(); cf != nil && rand.IntN(2) == 0 {
			(*cf)()
			runtime.Gosched()
		}
		return r, err
	}
	base := runtime.NumGoroutine()
	for i := 0; i < iters; i++ {
		n := 1 + i%32768
		ctx, cancel := context.WithCancel(context.Background())
		cancelNow.Store(&cancel)
		done := make(chan error, 1)
		go func() { done <- m.SetMaxTokens(ctx, n) }()
		var err error
		select {
		case err = <-done:
		case <-time.After(5 * time.Second):
			t.Fatalf("iteration %d: caller blocked forever", i)
		}
		cancelNow.Store(nil)
		cancel()
		switch {
		case err == nil:
			nils.Add(1)
			if got := m.MaxTokens(); got != n {
				t.Fatalf("iteration %d: nil return but MaxTokens() = %d, want %d", i, got, n)
			}
		case errors.Is(err, context.Canceled):
			ctxErrs.Add(1)
		default:
			t.Fatalf("iteration %d: unexpected error %v", i, err)
		}
		settle(m)
		if got, want := m.MaxTokens(), e.enforced(); got != want {
			t.Fatalf("iteration %d: MaxTokens() = %d, engine %d", i, got, want)
		}
	}
	t.Logf("nil=%d ctxErr=%d", nils.Load(), ctxErrs.Load())
	if !waitFor(2*time.Second, func() bool { return runtime.NumGoroutine() <= base }) {
		t.Errorf("goroutines leaked: %d vs %d", runtime.NumGoroutine(), base)
	}
}

// Randomised deadlines around the engine's latency, many callers at once.
func TestUnitAdversarialSetMaxTokensRandomDeadlinesStress(t *testing.T) {
	rounds := 200
	if testing.Short() {
		rounds = 20
	}
	e := &fakeEngine{jitter: true}
	m := modelOver(e)
	base := runtime.NumGoroutine()
	for r := 0; r < rounds; r++ {
		var wg sync.WaitGroup
		for i := 0; i < 50; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				ctx, cancel := context.WithTimeout(context.Background(), time.Duration(rand.IntN(400))*time.Microsecond)
				defer cancel()
				err := m.SetMaxTokens(ctx, 1+r*50+i)
				if err != nil && !errors.Is(err, context.DeadlineExceeded) {
					t.Errorf("unexpected error %v", err)
				}
			}(i)
		}
		wg.Wait()
		settle(m)
		if got, want := m.MaxTokens(), e.enforced(); got != want {
			t.Fatalf("round %d: MaxTokens() = %d, engine %d", r, got, want)
		}
	}
	if !waitFor(2*time.Second, func() bool { return runtime.NumGoroutine() <= base }) {
		t.Errorf("goroutines leaked: %d vs %d", runtime.NumGoroutine(), base)
	}
}

// With N live-context callers behind one running call, goroutines above the
// baseline are N callers and one helper; no helper per waiter.
func TestUnitAdversarialSetMaxTokensGoroutinesAreCallersPlusOneHelper(t *testing.T) {
	const N = 40
	e := &fakeEngine{started: make(chan struct{}), release: make(chan struct{})}
	m := modelOver(e)
	base := runtime.NumGoroutine()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var wg sync.WaitGroup
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(i int) { defer wg.Done(); _ = m.SetMaxTokens(ctx, 10+i) }(i)
	}
	<-e.started
	time.Sleep(50 * time.Millisecond)
	if got := runtime.NumGoroutine() - base; got > N+1 {
		t.Errorf("%d goroutines above baseline for %d callers, want at most %d", got, N, N+1)
	}
	close(e.release)
	wg.Wait()
	settle(m)
	if got, want := m.MaxTokens(), e.enforced(); got != want {
		t.Errorf("MaxTokens() = %d, engine %d", got, want)
	}
	if !waitFor(2*time.Second, func() bool { return runtime.NumGoroutine() <= base }) {
		t.Errorf("goroutines leaked: %d vs %d", runtime.NumGoroutine(), base)
	}
}

// A caller that abandons loses a non-panic engine error silently (the doc says
// "may or may not have changed"); pin that the cache stays on the old limit.
func TestUnitAdversarialSetMaxTokensAbandonedEngineErrorKeepsOldLimit(t *testing.T) {
	m := bare()
	m.maxTokens.Store(8192)
	gate := make(chan struct{})
	boom := errors.New("engine refused")
	m.send = func(context.Context, string, uint32, *wire.Frame) (wire.Reply, error) {
		<-gate
		return wire.Reply{}, boom
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if err := m.SetMaxTokens(ctx, 6); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v", err)
	}
	close(gate)
	settle(m)
	if m.MaxTokens() != 8192 {
		t.Errorf("MaxTokens() = %d after a failed engine call", m.MaxTokens())
	}
}
