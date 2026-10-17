//go:build darwin && arm64

package embedgemma2

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/bharathvbcr/gusset"

	"github.com/bharathvbcr/tessl/go/embedgemma2/internal/wire"
)

// These tests drive the engine archive through a Model that holds no tessl
// model: the engine's no-work operations (a ping, a deliberate panic and a
// sleep that stands in for a tessl call waiting on the GPU) make the
// containment and cancellation behaviour testable without a device and without
// timing luck. The sleep has no cancellation point, as a tessl call has none.

func rawModel(t *testing.T, observer func(CallInfo)) *Model {
	t.Helper()
	if err := registerEngine(); err != nil {
		t.Fatal(err)
	}
	h, err := gusset.Open(gusset.WithPoolSize(1))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = h.Close() })
	return &Model{h: h, dim: 8, observer: observer}
}

func sleepFrame(ms uint64) *wire.Frame { return new(wire.Frame).U64(ms) }

func TestUnitPingCrossesAndTheObserverSeesIt(t *testing.T) {
	var seen []CallInfo
	m := rawModel(t, func(ci CallInfo) { seen = append(seen, ci) })
	reply, err := m.roundTrip(context.Background(), "Ping", wire.OpPing, new(wire.Frame))
	if err != nil || reply.Status != wire.StatusOK || len(reply.Payload) != 0 {
		t.Fatalf("%+v %v", reply, err)
	}
	if len(seen) != 1 || seen[0].Op != "Ping" || seen[0].Err != nil || seen[0].RequestBytes != 0 || seen[0].ReplyBytes != wire.ReplyHeader {
		t.Fatalf("observed %+v", seen)
	}
	// A request the operation does not take is refused in Rust, by its message.
	_, err = m.roundTrip(context.Background(), "Ping", wire.OpPing, new(wire.Frame).U32(1))
	if err == nil {
		t.Fatal("a ping with a body was accepted")
	}
}

func TestUnitAnOversizedRequestTravelsInARustBuffer(t *testing.T) {
	m := rawModel(t, nil)
	// A request past the inline limit takes the buffer path; the engine rejects
	// the body of a ping, but only after decoding it whole, so the refusal is
	// the engine's, not the transport's.
	big := make([]byte, 2*inlineLimit)
	_, err := m.roundTrip(context.Background(), "Ping", wire.OpPing, new(wire.Frame).Bytes(big))
	if err == nil || errors.Is(err, ErrClosed) || errors.Is(err, ErrPanic) {
		t.Fatalf("got %v", err)
	}
	if reply, err := m.roundTrip(context.Background(), "Ping", wire.OpPing, new(wire.Frame)); err != nil || reply.Status != wire.StatusOK {
		t.Fatalf("the handle is unusable after a large refused request: %v", err)
	}
}

func TestUnitAPanicInTheEngineIsContainedAndKillsOnlyThatModel(t *testing.T) {
	m := rawModel(t, nil)
	_, err := m.roundTrip(context.Background(), "SelfTest", wire.OpSelfTestPanic, new(wire.Frame))
	if !errors.Is(err, ErrPanic) {
		t.Fatalf("got %v, want ErrPanic", err)
	}
	if !errors.Is(err, gusset.ErrPanic) {
		t.Errorf("gusset's own error is not in the chain: %v", err)
	}
	if m.dead.Load() == nil {
		t.Fatal("the model was not marked unusable")
	}
	// Every later call on that model says so, without crossing.
	_, err = m.call(context.Background(), "EmbedTexts", wire.OpPing, new(wire.Frame))
	if !errors.Is(err, ErrPanic) || !strings.Contains(err.Error(), "Close it and Load again") {
		t.Fatalf("a call on the dead model: %v", err)
	}
	// The process and other models are unharmed.
	other := rawModel(t, nil)
	if _, err := other.roundTrip(context.Background(), "Ping", wire.OpPing, new(wire.Frame)); err != nil {
		t.Fatalf("another model after a panic: %v", err)
	}
}

func TestUnitQueuedCallLeavesAtItsDeadlineAndNeverBlocksOnTheWorker(t *testing.T) {
	m := rawModel(t, nil)
	started := time.Now()
	var wg sync.WaitGroup
	wg.Add(1)
	var first error
	go func() {
		defer wg.Done()
		_, first = m.roundTrip(context.Background(), "Sleep", wire.OpSelfTestSleep, sleepFrame(600))
	}()
	time.Sleep(100 * time.Millisecond) // the worker is inside the sleep

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	waited := time.Now()
	_, err := m.roundTrip(ctx, "Ping", wire.OpPing, new(wire.Frame))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v, want the deadline", err)
	}
	if d := time.Since(waited); d > 400*time.Millisecond {
		t.Errorf("the queued call stayed %v: it waited for the worker", d)
	}
	if time.Since(started) > 550*time.Millisecond {
		t.Fatal("the queued call outlasted the sleep it was meant to leave early")
	}

	// A cancel works the same way.
	cctx, ccancel := context.WithCancel(context.Background())
	time.AfterFunc(30*time.Millisecond, ccancel)
	if _, err := m.roundTrip(cctx, "Ping", wire.OpPing, new(wire.Frame)); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want a cancel", err)
	}

	wg.Wait()
	if first != nil {
		t.Fatalf("the running call was disturbed by the queued ones leaving: %v", first)
	}
	if _, err := m.roundTrip(context.Background(), "Ping", wire.OpPing, new(wire.Frame)); err != nil {
		t.Fatalf("the worker is unusable after cancellations: %v", err)
	}
}

func TestUnitRunningCallReleasesItsCallerButTheWorkerFinishesFirst(t *testing.T) {
	m := rawModel(t, nil)
	start := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, err := m.roundTrip(ctx, "Sleep", wire.OpSelfTestSleep, sleepFrame(500))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v, want the deadline", err)
	}
	if d := time.Since(start); d > 400*time.Millisecond {
		t.Fatalf("the caller was held %v by work it had abandoned", d)
	}
	// The abandoned work still holds the worker: the next call starts only when
	// it ends, so abandoned work never exceeds one call in flight.
	if _, err := m.roundTrip(context.Background(), "Ping", wire.OpPing, new(wire.Frame)); err != nil {
		t.Fatal(err)
	}
	if d := time.Since(start); d < 450*time.Millisecond {
		t.Errorf("the next call ran after %v, before the abandoned 500 ms of work ended", d)
	}
}

func TestUnitSelfTestSleepRefusesPastItsLimit(t *testing.T) {
	m := rawModel(t, nil)
	_, err := m.roundTrip(context.Background(), "Sleep", wire.OpSelfTestSleep, sleepFrame(10_000))
	if err == nil || !strings.Contains(err.Error(), "limit") {
		t.Fatalf("got %v", err)
	}
}

func TestUnitGoroutinesShareOneWorkerThread(t *testing.T) {
	m := rawModel(t, nil)
	before := gusset.Threads()
	const callers = 64
	var wg sync.WaitGroup
	errs := make(chan error, callers)
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := m.roundTrip(context.Background(), "Sleep", wire.OpSelfTestSleep, sleepFrame(2))
			errs <- err
		}()
	}
	peak := int64(0)
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
loop:
	for {
		select {
		case <-done:
			break loop
		default:
			if n := gusset.Threads(); n > peak {
				peak = n
			}
			time.Sleep(time.Millisecond)
		}
	}
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	// Gusset's figure for 64 concurrent callers is 8-10 OS threads in all (its
	// docs/adoption.md): callers park on the Go semaphore. A blocking cgo call per
	// goroutine would add 64. The bound is the baseline plus a few runtime
	// threads, far below one per caller.
	t.Logf("%d goroutines on one worker: OS threads %d before, %d at the peak", callers, before, peak)
	if peak > before+6 {
		t.Errorf("%d goroutines raised the thread count from %d to %d", callers, before, peak)
	}
}
