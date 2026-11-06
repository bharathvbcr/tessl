//go:build darwin && arm64

package embedgemma2

import (
	"bytes"
	"context"
	"errors"
	"log"
	"math/rand/v2"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/bharathvbcr/tessl/go/embedgemma2/internal/wire"
)

// fakeEngine stands in for the worker: it applies a limit the way the engine
// does (when the job runs), then holds the reply for `hold` the way a worker
// busy with a non-cooperative job does. Like gusset, it drops the reply and
// returns the context's error when the context handed to it ends first.
type fakeEngine struct {
	mu      sync.Mutex
	limit   uint32
	started chan struct{} // closed when the first job has applied its limit
	once    sync.Once
	release chan struct{} // nil: reply at once; else reply after it closes
	jitter  bool
}

func (e *fakeEngine) send(ctx context.Context, _ string, _ uint32, f *wire.Frame) (wire.Reply, error) {
	if err := ctx.Err(); err != nil { // roundTrip refuses a finished context before it submits
		return wire.Reply{}, err
	}
	buf := make([]byte, f.Size())
	if err := f.Fill(buf); err != nil {
		return wire.Reply{}, err
	}
	n := uint32(buf[0]) | uint32(buf[1])<<8 | uint32(buf[2])<<16 | uint32(buf[3])<<24
	e.mu.Lock()
	e.limit = n
	e.mu.Unlock()
	if e.started != nil {
		e.once.Do(func() { close(e.started) })
	}
	if e.jitter {
		time.Sleep(time.Duration(rand.IntN(200)) * time.Microsecond)
	}
	if e.release != nil {
		select {
		case <-e.release:
		case <-ctx.Done():
			return wire.Reply{}, ctx.Err()
		}
	}
	return wire.Reply{Status: wire.StatusOK}, nil
}

func (e *fakeEngine) enforced() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return int(e.limit)
}

func modelOver(e *fakeEngine) *Model {
	m := bare()
	m.maxTokens.Store(8192)
	e.limit = 8192
	m.send = e.send
	return m
}

// settle waits for every SetMaxTokens already issued to finish its cache update.
func settle(m *Model) { m.setTurn <- struct{}{}; <-m.setTurn }

func TestUnitSetMaxTokensCancelledAfterTheWorkerStartedKeepsTheCacheOnTheEngine(t *testing.T) {
	e := &fakeEngine{started: make(chan struct{}), release: make(chan struct{})}
	m := modelOver(e)
	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() { errc <- m.SetMaxTokens(ctx, 6) }()
	<-e.started // the worker has applied 6 and is still busy
	cancel()
	if err := <-errc; !errors.Is(err, context.Canceled) {
		t.Fatalf("SetMaxTokens after cancel = %v, want context.Canceled", err)
	}
	close(e.release)
	settle(m)
	if got, want := m.MaxTokens(), e.enforced(); got != want {
		t.Fatalf("MaxTokens() = %d, the engine enforces %d", got, want)
	}
}

func TestUnitSetMaxTokensCancelledBeforeItIsQueuedNeverReachesTheEngine(t *testing.T) {
	e := &fakeEngine{}
	m := modelOver(e)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := m.SetMaxTokens(ctx, 6); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
	settle(m)
	if m.MaxTokens() != 8192 || e.enforced() != 8192 {
		t.Fatalf("cache %d engine %d, both must stay 8192", m.MaxTokens(), e.enforced())
	}
}

func TestUnitConcurrentSetMaxTokensEndsOnTheEnginesLastLimit(t *testing.T) {
	for round := 0; round < 20; round++ {
		e := &fakeEngine{jitter: true}
		m := modelOver(e)
		var wg sync.WaitGroup
		for i := 1; i <= 48; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if err := m.SetMaxTokens(context.Background(), 100+i); err != nil {
					t.Error(err)
				}
			}()
		}
		wg.Wait()
		if got, want := m.MaxTokens(), e.enforced(); got != want {
			t.Fatalf("round %d: MaxTokens() = %d, the engine enforces %d", round, got, want)
		}
	}
}

func TestUnitConcurrentCancelledSetMaxTokensEndsOnTheEnginesLastLimit(t *testing.T) {
	e := &fakeEngine{jitter: true, release: make(chan struct{})}
	m := modelOver(e)
	var wg sync.WaitGroup
	for i := 1; i <= 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(i)*100*time.Microsecond)
			defer cancel()
			_ = m.SetMaxTokens(ctx, 100+i)
		}()
	}
	wg.Wait()
	close(e.release)
	time.Sleep(20 * time.Millisecond)
	settle(m)
	if got, want := m.MaxTokens(), e.enforced(); got != want {
		t.Fatalf("MaxTokens() = %d, the engine enforces %d", got, want)
	}
}

func TestUnitSetMaxTokensCancelledBeforeTheirTurnLeaveNoGoroutinesAndNeverReachTheEngine(t *testing.T) {
	e := &fakeEngine{started: make(chan struct{}), release: make(chan struct{})}
	m := modelOver(e)
	base := runtime.NumGoroutine()
	go func() { _ = m.SetMaxTokens(context.Background(), 7) }()
	<-e.started // the turn is held by a long call
	var wg sync.WaitGroup
	for i := 0; i < 2000; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
			defer cancel()
			if err := m.SetMaxTokens(ctx, 1000+i); !errors.Is(err, context.DeadlineExceeded) {
				t.Errorf("got %v", err)
			}
		}()
	}
	wg.Wait()
	// Only the long call's goroutine is left, and no helper was ever started.
	if !waitFor(5*time.Second, func() bool { return runtime.NumGoroutine() <= base+1 }) {
		t.Fatalf("%d goroutines over the baseline with the turn still held", runtime.NumGoroutine()-base)
	}
	close(e.release)
	if !waitFor(5*time.Second, func() bool { return runtime.NumGoroutine() <= base }) {
		t.Fatalf("%d goroutines over the baseline after the long call", runtime.NumGoroutine()-base)
	}
	if m.MaxTokens() != 7 || e.enforced() != 7 {
		t.Fatalf("cache %d engine %d: a call cancelled before its turn changed the limit", m.MaxTokens(), e.enforced())
	}
}

// panickingModel is over an engine whose first call panics after applying its
// limit (as an Observer does, after the reply) and whose later calls work.
func panickingModel() (*Model, *fakeEngine) {
	e := &fakeEngine{}
	m := modelOver(e)
	var first atomic.Bool
	m.send = func(ctx context.Context, op string, code uint32, f *wire.Frame) (wire.Reply, error) {
		reply, err := e.send(ctx, op, code, f)
		if first.CompareAndSwap(false, true) {
			panic("observer panic")
		}
		return reply, err
	}
	return m, e
}

func wantUsable(t *testing.T, m *Model, e *fakeEngine) {
	t.Helper()
	if len(m.setTurn) != 0 {
		t.Fatal("the turn is still held")
	}
	if err := m.SetMaxTokens(context.Background(), 77); err != nil {
		t.Fatalf("after a panic the model refuses SetMaxTokens: %v", err)
	}
	if m.MaxTokens() != 77 || e.enforced() != 77 {
		t.Fatalf("cache %d engine %d, want 77", m.MaxTokens(), e.enforced())
	}
}

func TestUnitSetMaxTokensPanicOnTheHelperPathReachesTheWaitingCaller(t *testing.T) {
	m, e := panickingModel()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var got any
	func() {
		defer func() { got = recover() }()
		_ = m.SetMaxTokens(ctx, 6)
	}()
	if got != "observer panic" {
		t.Fatalf("recovered %v, want the original panic value", got)
	}
	wantUsable(t, m, e)
}

func TestUnitSetMaxTokensPanicAfterTheCallerLeftIsLoggedAndReleasesTheTurn(t *testing.T) {
	var logs bytes.Buffer
	var lmu sync.Mutex
	log.SetOutput(writerFunc(func(b []byte) (int, error) { lmu.Lock(); defer lmu.Unlock(); return logs.Write(b) }))
	defer log.SetOutput(os.Stderr)
	m, e := panickingModel()
	started, release := make(chan struct{}), make(chan struct{})
	inner := m.send
	var held atomic.Bool
	m.send = func(ctx context.Context, op string, code uint32, f *wire.Frame) (wire.Reply, error) {
		if held.CompareAndSwap(false, true) {
			close(started)
			<-release
		}
		return inner(ctx, op, code, f)
	}
	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() { errc <- m.SetMaxTokens(ctx, 6) }()
	<-started
	cancel()
	if err := <-errc; !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
	close(release)
	if !waitFor(5*time.Second, func() bool { return len(m.setTurn) == 0 }) {
		t.Fatal("the turn was not released")
	}
	if !waitFor(5*time.Second, func() bool { lmu.Lock(); defer lmu.Unlock(); return strings.Contains(logs.String(), "observer panic") }) {
		t.Fatal("the panic was not logged")
	}
	wantUsable(t, m, e)
}

type writerFunc func([]byte) (int, error)

func (w writerFunc) Write(b []byte) (int, error) { return w(b) }
