//go:build darwin && arm64

package embedgemma2

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/bharathvbcr/tessl/go/embedgemma2/internal/wire"
)

// waitFor polls cond for up to d.
func waitFor(d time.Duration, cond func() bool) bool {
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if cond() {
			return true
		}
		time.Sleep(time.Millisecond)
	}
	return cond()
}

// Attack 1: a Close that lands while helpers are queued on setMu. The helpers
// must leave with ErrClosed, never touch the engine, and not leak.
func TestUnitAdversarialCloseWhileSetMaxTokensQueuedLeavesNoGoroutinesAndKeepsTheLimit(t *testing.T) {
	e := &fakeEngine{started: make(chan struct{}), release: make(chan struct{})}
	m := modelOver(e)
	base := runtime.NumGoroutine()
	errs := make(chan error, 17)
	ctxs := make([]context.CancelFunc, 0, 17)
	for i := 0; i < 17; i++ {
		ctx, cancel := context.WithCancel(context.Background())
		ctxs = append(ctxs, cancel)
		go func(i int) { errs <- m.SetMaxTokens(ctx, 100+i) }(i)
		if i == 0 {
			<-e.started
		}
	}
	time.Sleep(20 * time.Millisecond) // 16 helpers now wait on setMu
	m.closed.Store(true)              // what Close does first
	close(e.release)                  // the running job finishes
	var closedErrs int
	for i := 0; i < 17; i++ {
		select {
		case err := <-errs:
			if errors.Is(err, ErrClosed) {
				closedErrs++
			} else if err != nil {
				t.Errorf("unexpected error %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("SetMaxTokens still blocked after the model closed (deadlock)")
		}
	}
	for _, c := range ctxs {
		c()
	}
	if closedErrs != 16 {
		t.Errorf("%d queued calls got ErrClosed, want 16", closedErrs)
	}
	if !waitFor(2*time.Second, func() bool { return runtime.NumGoroutine() <= base }) {
		t.Errorf("goroutines leaked: %d now, %d before", runtime.NumGoroutine(), base)
	}
	if got, want := m.MaxTokens(), e.enforced(); got != want {
		t.Errorf("MaxTokens() = %d, engine enforces %d", got, want)
	}
}

// Attack 2: cancelled calls pile up behind one long call. Records the peak and
// requires that all of them drain once the long call ends, without touching
// the engine.
func TestUnitAdversarialCancelledSetMaxTokensBehindALongCallDrainAndNeverReachTheEngine(t *testing.T) {
	e := &fakeEngine{started: make(chan struct{}), release: make(chan struct{})}
	m := modelOver(e)
	base := runtime.NumGoroutine()
	go func() { _ = m.SetMaxTokens(context.Background(), 7) }()
	<-e.started
	const n = 300
	for i := 0; i < n; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
		_ = m.SetMaxTokens(ctx, 1000+i)
		cancel()
	}
	peak := runtime.NumGoroutine() - base
	t.Logf("goroutines parked behind a single running call after %d cancelled calls: %d", n, peak)
	close(e.release)
	if !waitFor(5*time.Second, func() bool { return runtime.NumGoroutine() <= base+1 }) {
		t.Fatalf("helpers did not drain: %d goroutines over base", runtime.NumGoroutine()-base)
	}
	settle(m)
	if m.MaxTokens() != 7 || e.enforced() != 7 {
		t.Fatalf("cache %d engine %d: a cancelled, never-started call changed the limit", m.MaxTokens(), e.enforced())
	}
}

// Attack 3: the window between the engine applying a limit and the cache
// following. Pins that the window exists (the doc must not say "always").
func TestUnitAdversarialMaxTokensLagsTheEngineWhileTheReplyIsInFlight(t *testing.T) {
	e := &fakeEngine{started: make(chan struct{}), release: make(chan struct{})}
	m := modelOver(e)
	done := make(chan error, 1)
	go func() { done <- m.SetMaxTokens(context.Background(), 6) }()
	<-e.started
	if m.MaxTokens() == e.enforced() {
		t.Fatal("expected the cache to lag the engine while the reply is in flight; the doc's \"always\" would then be true")
	}
	close(e.release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if m.MaxTokens() != 6 {
		t.Fatalf("MaxTokens() = %d after nil return, want 6", m.MaxTokens())
	}
}

// Attack 5: the engine no longer sees the caller's deadline or cancellation.
func TestUnitAdversarialSetMaxTokensHandsTheEngineNoDeadlineOrCancellation(t *testing.T) {
	m := bare()
	m.maxTokens.Store(8192)
	var sawDeadline, sawDone atomic.Bool
	m.send = func(ctx context.Context, _ string, _ uint32, _ *wire.Frame) (wire.Reply, error) {
		_, d := ctx.Deadline()
		sawDeadline.Store(d)
		sawDone.Store(ctx.Done() != nil)
		return wire.Reply{Status: wire.StatusOK}, nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Hour)
	defer cancel()
	if err := m.SetMaxTokens(ctx, 6); err != nil {
		t.Fatal(err)
	}
	if sawDeadline.Load() || sawDone.Load() {
		t.Fatalf("engine ctx has deadline=%v done=%v", sawDeadline.Load(), sawDone.Load())
	}
}

// Attack 6: error identity on both paths, and the message shape other ops use.
func TestUnitAdversarialSetMaxTokensContextErrorIdentityOnBothPaths(t *testing.T) {
	// pre-turn: already-expired deadline
	e := &fakeEngine{}
	m := modelOver(e)
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	err := m.SetMaxTokens(ctx, 6)
	if !errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
		t.Fatalf("pre-turn: %v", err)
	}
	if got, want := err.Error(), "embedgemma2: SetMaxTokens: context deadline exceeded"; got != want {
		t.Fatalf("message %q, want %q", got, want)
	}
	// post-turn: deadline expires while the worker is busy
	e2 := &fakeEngine{started: make(chan struct{}), release: make(chan struct{})}
	m2 := modelOver(e2)
	ctx2, cancel2 := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel2()
	err = m2.SetMaxTokens(ctx2, 6)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("post-turn: %v", err)
	}
	if got, want := err.Error(), "embedgemma2: SetMaxTokens: context deadline exceeded"; got != want {
		t.Fatalf("message %q, want %q", got, want)
	}
	close(e2.release)
	settle(m2)
}

// Attack 7: nil ctx is an *ArgError, for a valid n, and n is validated first.
func TestUnitAdversarialSetMaxTokensNilContext(t *testing.T) {
	m := modelOver(&fakeEngine{})
	//lint:ignore SA1012 deliberate
	wantArg(t, "nil ctx", m.SetMaxTokens(nil, 6), "ctx")      //nolint:staticcheck
	wantArg(t, "bad n, nil ctx", m.SetMaxTokens(nil, 0), "n") //nolint:staticcheck
	if m.MaxTokens() != 8192 {
		t.Fatal("limit changed")
	}
}

// Attack 8: a failing engine, a dead model and a closed model leave the cache alone;
// a panic with the lock held (inline path) does not wedge later calls.
func TestUnitAdversarialFailedDeadAndPanickingEngineLeaveTheCacheAndTheLock(t *testing.T) {
	m := bare()
	m.maxTokens.Store(8192)
	boom := errors.New("boom")
	m.send = func(context.Context, string, uint32, *wire.Frame) (wire.Reply, error) { return wire.Reply{}, boom }
	if err := m.SetMaxTokens(context.Background(), 6); !errors.Is(err, boom) || m.MaxTokens() != 8192 {
		t.Fatalf("err %v limit %d", err, m.MaxTokens())
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := m.SetMaxTokens(ctx, 6); !errors.Is(err, boom) || m.MaxTokens() != 8192 {
		t.Fatalf("helper path: err %v limit %d", err, m.MaxTokens())
	}
	m.send = func(context.Context, string, uint32, *wire.Frame) (wire.Reply, error) { panic("engine panic") }
	func() {
		defer func() { _ = recover() }()
		_ = m.SetMaxTokens(context.Background(), 6)
	}()
	if !m.setMu.TryLock() {
		t.Fatal("setMu still held after a panic on the inline path")
	}
	m.setMu.Unlock()
	dead := errors.New("dead")
	m.dead.Store(&dead)
	m.send = func(context.Context, string, uint32, *wire.Frame) (wire.Reply, error) {
		t.Error("engine reached on a dead model")
		return wire.Reply{Status: wire.StatusOK}, nil
	}
	if err := m.SetMaxTokens(ctx, 6); !errors.Is(err, dead) || m.MaxTokens() != 8192 {
		t.Fatalf("dead: %v limit %d", err, m.MaxTokens())
	}
}

// A panic in the engine call (in practice: a user's Observer, which roundTrip
// runs in a defer) on the helper path kills the process: the caller cannot
// recover it, as it could when the call ran on its own goroutine.
func TestUnitAdversarialPanicOnTheHelperPathIsNotRecoverableByTheCaller(t *testing.T) {
	if os.Getenv("TESSL_ADV_PANIC") == "1" {
		m := bare()
		m.maxTokens.Store(8192)
		m.send = func(context.Context, string, uint32, *wire.Frame) (wire.Reply, error) { panic("observer panic") }
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		defer func() { _ = recover(); os.Exit(0) }() // survived: the caller recovered it
		_ = m.SetMaxTokens(ctx, 6)
		os.Exit(0)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestUnitAdversarialPanicOnTheHelperPathIsNotRecoverableByTheCaller$")
	cmd.Env = append(os.Environ(), "TESSL_ADV_PANIC=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Skip("panic was recoverable by the caller")
	}
	t.Logf("a caller-side recover does not catch a panic on the helper path (child exited: %v)", err)
	_ = out
}

// Ordering: sequential calls from one goroutine, some cancelled while the
// worker is busy, end on the engine's last applied limit.
func TestUnitAdversarialSequentialCancelledThenLiveCallsAgree(t *testing.T) {
	for round := 0; round < 50; round++ {
		e := &fakeEngine{jitter: true}
		m := modelOver(e)
		for i := 1; i <= 20; i++ {
			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(i%3)*50*time.Microsecond)
			_ = m.SetMaxTokens(ctx, 100+i)
			cancel()
		}
		last := m.SetMaxTokens(context.Background(), 999)
		if last != nil {
			t.Fatal(last)
		}
		settle(m)
		if m.MaxTokens() != 999 || e.enforced() != 999 {
			t.Fatalf("round %d: cache %d engine %d, the last (live) call set 999", round, m.MaxTokens(), e.enforced())
		}
	}
}
