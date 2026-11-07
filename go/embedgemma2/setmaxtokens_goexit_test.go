//go:build darwin && arm64

package embedgemma2

import (
	"context"
	"runtime"
	"testing"
	"time"

	"github.com/bharathvbcr/tessl/go/embedgemma2/internal/wire"
)

// A helper whose engine call ends its goroutine with runtime.Goexit (a
// testing.T.FailNow in an Observer does this) never reached the cache update.
// The caller must not be told nil: only a nil return guarantees the new limit.
func TestUnitAdversarialSetMaxTokensHelperGoexitIsNotReportedAsSuccess(t *testing.T) {
	m := bare()
	m.maxTokens.Store(8192)
	m.send = func(context.Context, string, uint32, *wire.Frame) (wire.Reply, error) {
		runtime.Goexit()
		return wire.Reply{}, nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := make(chan error, 1)
	go func() { errc <- m.SetMaxTokens(ctx, 6) }()
	select {
	case err := <-errc:
		if err == nil && m.MaxTokens() != 6 {
			t.Fatalf("SetMaxTokens returned nil but MaxTokens() = %d, want 6", m.MaxTokens())
		}
	case <-time.After(3 * time.Second):
		t.Fatal("caller blocked forever after the helper's Goexit")
	}
	if len(m.setTurn) != 0 {
		t.Error("turn not released after Goexit")
	}
}
