//go:build darwin && arm64

package embedgemma2

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/bharathvbcr/gusset"

	"github.com/bharathvbcr/tessl/go/embedgemma2/internal/wire"
)

// Limits of tessl's C ABI that the package checks before a call crosses into
// Rust. tessl checks all of them again (and more: patches, frames, tokens);
// these are the ones a caller can be told about without a round trip.
const (
	// MaxBatch is the most items one call embeds: one output row each.
	MaxBatch = 256
	// MaxPathBytes is the longest file path or snapshot directory.
	MaxPathBytes = 4096
	// MaxPromptNameBytes is the longest prompt name.
	MaxPromptNameBytes = 256
	// MaxTextBytes bounds the texts of one call: they hold fewer bytes.
	MaxTextBytes = 1 << 28
	// MaxClipSamples is the most samples one raw clip may hold (the extractor
	// then cuts it to 30 s).
	MaxClipSamples = 1 << 28
	// MaxExtensionBytes is the longest video type hint.
	MaxExtensionBytes = 8
	// MaxSegments is the most segments one EmbedMixed call takes.
	MaxSegments = 1 << 16
	// MaxRequestBytes is the largest request: gusset's 1 GiB buffer ceiling.
	// An encoded video or a raw clip bigger than this goes by path, not by
	// value, although tessl itself accepts up to 4 GiB of encoded video.
	MaxRequestBytes = wire.MaxFrame
)

// maxTokensLimit is the largest token limit tessl accepts
// (embedgemma2::MAX_BATCH_TOKENS).
const maxTokensLimit = 32768

// inlineLimit is the largest request gusset copies inline (its R16 limit);
// anything bigger travels in a Rust-owned buffer.
const inlineLimit = 4096

// Operands is the GEMM operand precision of a multimodal model's text encoder
// and both towers.
type Operands int

const (
	// ExactF32 runs every GEMM in f32 (the default).
	ExactF32 Operands = iota
	// BF16 rounds the operands to bf16 with f32 accumulation.
	BF16
)

// Options configures Load.
type Options struct {
	// Operands is the GEMM precision. BF16 applies to the text encoder and
	// both towers, so it needs a multimodal model (TextOnly false).
	Operands Operands
	// TextOnly loads the text encoder alone (about 0.8 GB on the device
	// against 1.7 GB), which refuses every image, audio and video call.
	TextOnly bool
	// Observer, if set, is called once after every call that reaches the
	// engine, from the calling goroutine, with what the call cost. It must not
	// call back into the Model.
	Observer func(CallInfo)
}

// CallInfo is what one call to the engine cost, for metrics.
type CallInfo struct {
	// Op is the method that made the call.
	Op string
	// Work is the time spent inside tessl's C ABI, measured in Rust around the
	// call itself: the same work a Rust caller would time directly.
	Work time.Duration
	// RequestBytes and ReplyBytes are the sizes of the frames that crossed.
	RequestBytes, ReplyBytes int
	// Err is the call's error, nil on success.
	Err error
}

// Model is a loaded EmbeddingGemma 2 on the system's Metal device.
//
// # Concurrency
//
// A Model is safe for concurrent use. tessl's handles belong to the thread
// that made them, so each Model owns one gusset handle with exactly one worker
// thread, which makes the model and runs every call against it. Calls from many
// goroutines queue on gusset's semaphore in Go, not on an OS thread each:
// 64 goroutines embedding at once cost one worker thread, and a goroutine whose
// context ends while it waits leaves without ever reaching the device. Models
// are independent: two Models run on two workers and two Metal runtimes.
//
// # Cancellation
//
// What a context can and cannot stop:
//
//   - A call still waiting for the worker is cancelled at once, and never
//     reaches Rust. Its context's deadline is also handed to the engine, which
//     refuses a job that expired in the queue before it starts.
//   - A call the worker has begun runs to completion. tessl's entries are one
//     synchronous GPU wait with no cancellation point, so the work (and the
//     worker, and Close) cannot be interrupted. The caller is released at its
//     deadline all the same, returning ctx.Err(); the result is discarded and
//     the worker's permit returns only when the work stops, so abandoned work
//     never exceeds one call in flight per Model.
//   - Request data stays valid for the abandoned work: gusset holds the buffer
//     it travelled in until the job ends.
type Model struct {
	h        *gusset.Handle
	observer func(CallInfo)

	dim        int
	multimodal bool

	closed  atomic.Bool
	dead    atomic.Pointer[error]
	closeMu sync.Mutex

	promptsMu sync.Mutex
	prompts   *Prompts

	maxTokens atomic.Int64
}

// Load reads the snapshot directory snapshotDir (a Hugging Face snapshot of
// google/embeddinggemma-2: config.json, model.safetensors and the tokenizer and
// processor files) onto the system's Metal device.
//
// The tokenizer and the preprocessing settings are read from the directory on
// the first call that needs them. Loading the full model takes seconds; use
// LoadContext to bound the wait (the load itself, once begun, is not
// interruptible).
func Load(snapshotDir string, opts Options) (*Model, error) {
	return LoadContext(context.Background(), snapshotDir, opts)
}

// LoadContext is Load under a context: see Model for what it can cancel.
func LoadContext(ctx context.Context, snapshotDir string, opts Options) (*Model, error) {
	const op = "Load"
	if err := checkLoad(snapshotDir, opts); err != nil {
		return nil, err
	}
	if err := registerEngine(); err != nil {
		return nil, err
	}
	h, err := gusset.Open(gusset.WithPoolSize(1))
	if err != nil {
		return nil, fmt.Errorf("embedgemma2: %s: opening the engine handle: %w", op, err)
	}
	m := &Model{h: h, observer: opts.Observer}
	multimodal := uint32(1)
	if opts.TextOnly {
		multimodal = 0
	}
	f := new(wire.Frame).U32(uint32(opts.Operands)).U32(multimodal).Text(snapshotDir)
	reply, err := m.roundTrip(ctx, op, wire.OpLoad, f)
	if err != nil {
		_ = h.Close()
		return nil, err
	}
	r := wire.NewReader(reply.Payload)
	dim, maxTokens := r.U32("embedding_dim"), r.U32("max_tokens")
	if err := r.Done(); err != nil {
		_ = h.Close()
		return nil, protocolErr(op, "%v", err)
	}
	m.dim = int(dim)
	m.multimodal = !opts.TextOnly
	m.maxTokens.Store(int64(maxTokens))
	return m, nil
}

func checkLoad(snapshotDir string, opts Options) error {
	const op = "Load"
	switch {
	case snapshotDir == "":
		return argErr(op, "snapshotDir", "is empty")
	case len(snapshotDir) > MaxPathBytes:
		return argErr(op, "snapshotDir", "is %d bytes, the limit is %d", len(snapshotDir), MaxPathBytes)
	case !wire.ValidUTF8(snapshotDir):
		return argErr(op, "snapshotDir", "is not UTF-8 or contains a NUL byte")
	}
	switch opts.Operands {
	case ExactF32:
	case BF16:
		if opts.TextOnly {
			return argErr(op, "Options.Operands", "BF16 applies to a multimodal load (the text encoder and both towers); TextOnly loads the text encoder in exact f32")
		}
	default:
		return argErr(op, "Options.Operands", "%d is neither ExactF32 nor BF16", int(opts.Operands))
	}
	return nil
}

// Close frees the model and its Metal runtime and stops the worker. It waits
// for a call the worker has begun (see Model for why that cannot be cut short;
// gusset gives up after 30 seconds and Close then reports it). Calls made after
// Close return ErrClosed. A second Close does nothing.
func (m *Model) Close() error {
	m.closeMu.Lock()
	defer m.closeMu.Unlock()
	if !m.closed.CompareAndSwap(false, true) {
		return nil
	}
	// The free runs on the worker that made the model: tessl leaks a handle
	// freed anywhere else. It queues behind calls already in flight.
	_, freeErr := m.roundTrip(context.Background(), "Close", wire.OpFree, new(wire.Frame))
	var te *Error
	if errors.As(freeErr, &te) && te.Status == StatusNoModel {
		// A model that panicked has no live handle left to free.
		freeErr = nil
	}
	return errors.Join(freeErr, m.h.Close())
}

// EmbeddingDim is the width of an untruncated embedding, and the largest
// truncateDim.
func (m *Model) EmbeddingDim() int { return m.dim }

// Multimodal reports whether the vision and audio towers are loaded.
func (m *Model) Multimodal() bool { return m.multimodal }

// MaxTokens is the per-sequence token limit (8192 until SetMaxTokens).
func (m *Model) MaxTokens() int { return int(m.maxTokens.Load()) }

// SetMaxTokens raises or lowers the per-sequence token limit, from 1 to 32768.
// It bounds the image and audio sequences of the multimodal calls too.
func (m *Model) SetMaxTokens(ctx context.Context, n int) error {
	const op = "SetMaxTokens"
	if n < 1 || n > maxTokensLimit {
		return argErr(op, "n", "%d is outside 1..=%d", n, maxTokensLimit)
	}
	f := new(wire.Frame).U32(uint32(n))
	if _, err := m.call(ctx, op, wire.OpSetMaxTokens, f); err != nil {
		return err
	}
	m.maxTokens.Store(int64(n))
	return nil
}

// roundTrip sends one request to the worker and parses the reply, mapping a
// non-OK status to *Error. It does not look at the closed flag.
func (m *Model) roundTrip(ctx context.Context, op string, code uint32, f *wire.Frame) (reply wire.Reply, err error) {
	var requestBytes, replyBytes int
	if m.observer != nil {
		defer func() {
			m.observer(CallInfo{
				Op:           op,
				Work:         time.Duration(reply.WorkNS),
				RequestBytes: requestBytes,
				ReplyBytes:   replyBytes,
				Err:          err,
			})
		}()
	}
	if ctx == nil {
		return reply, argErr(op, "ctx", "is nil")
	}
	if err := f.Err(); err != nil {
		return reply, argErr(op, "request", "%v", err)
	}
	if err := ctx.Err(); err != nil {
		return reply, fmt.Errorf("embedgemma2: %s: %w", op, err)
	}
	requestBytes = f.Size()
	cctx := gusset.ContextWithOpcode(ctx, code)
	var raw []byte
	if requestBytes <= inlineLimit {
		in := make([]byte, requestBytes)
		if err := f.Fill(in); err != nil {
			return reply, protocolErr(op, "%v", err)
		}
		raw, err = m.h.Call(cctx, in)
	} else {
		raw, err = m.callLarge(cctx, f)
	}
	if err != nil {
		return reply, m.mapGussetErr(op, err)
	}
	replyBytes = len(raw)
	reply, err = wire.ParseReply(raw)
	if err != nil {
		return reply, protocolErr(op, "%v", err)
	}
	if reply.Status != wire.StatusOK {
		e := &Error{
			Op:       op,
			Status:   Status(reply.Status),
			Message:  string(reply.Payload),
			Poisoned: reply.Flags&wire.FlagPoisoned != 0,
		}
		if e.Status == StatusPanic || e.Status == StatusNoModel || e.Poisoned {
			m.markDead(e)
		}
		return reply, e
	}
	return reply, nil
}

// callLarge sends a request too big to travel inline, in a Rust-owned buffer.
// The reply is copied into Go memory before its buffer is freed.
func (m *Model) callLarge(ctx context.Context, f *wire.Frame) ([]byte, error) {
	in, err := m.h.NewBuffer(f.Size())
	if err != nil {
		return nil, err
	}
	// Free is safe while an abandoned job still reads the buffer: gusset's
	// registry holds the job's own reference until the job ends.
	defer func() { _ = in.Free() }()
	dst := in.Bytes()
	if dst == nil {
		return nil, gusset.ErrClosed
	}
	if err := f.Fill(dst); err != nil {
		return nil, err
	}
	out, err := m.h.CallBuffer(ctx, in)
	if err != nil {
		return nil, err
	}
	defer func() { _ = out.Free() }()
	b := out.Bytes()
	raw := make([]byte, len(b))
	copy(raw, b)
	return raw, nil
}

// mapGussetErr turns what gusset reports into this package's errors, keeping
// gusset's own in the chain so errors.Is reaches both.
func (m *Model) mapGussetErr(op string, err error) error {
	switch {
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return fmt.Errorf("embedgemma2: %s: %w", op, err)
	case errors.Is(err, gusset.ErrClosed):
		return fmt.Errorf("embedgemma2: %s: %w: %w", op, ErrClosed, err)
	case errors.Is(err, gusset.ErrPanic), errors.Is(err, gusset.ErrPoisoned):
		e := fmt.Errorf("embedgemma2: %s: %w: %w", op, ErrPanic, err)
		m.markDead(e)
		return e
	}
	var ge *gusset.Error
	if errors.As(err, &ge) {
		return fmt.Errorf("embedgemma2: %s: %w: %w", op, ErrProtocol, err)
	}
	return fmt.Errorf("embedgemma2: %s: %w", op, err)
}

func (m *Model) markDead(err error) {
	m.dead.CompareAndSwap(nil, &err)
}

// call is roundTrip for a public method: it refuses a closed or dead model.
func (m *Model) call(ctx context.Context, op string, code uint32, f *wire.Frame) (wire.Reply, error) {
	if m.closed.Load() {
		return wire.Reply{}, fmt.Errorf("embedgemma2: %s: %w", op, ErrClosed)
	}
	if dead := m.dead.Load(); dead != nil {
		return wire.Reply{}, fmt.Errorf("embedgemma2: %s: the model is unusable, Close it and Load again: %w", op, *dead)
	}
	return m.roundTrip(ctx, op, code, f)
}

// checkTruncate validates a Matryoshka width: 0 for the full width, else
// 1..=EmbeddingDim.
func (m *Model) checkTruncate(op string, truncateDim int) error {
	if truncateDim < 0 || truncateDim > m.dim {
		return argErr(op, "truncateDim", "%d must be 0 (the full width) or in 1..=%d", truncateDim, m.dim)
	}
	return nil
}

// checkCount validates the number of items of a batch.
func checkCount(op, field string, n int) error {
	if n < 1 || n > MaxBatch {
		return argErr(op, field, "has %d items, expected 1..=%d", n, MaxBatch)
	}
	return nil
}

// embed sends an embedding request and splits the reply into n rows.
func (m *Model) embed(ctx context.Context, op string, code uint32, f *wire.Frame, n, truncateDim int) ([][]float32, error) {
	reply, err := m.call(ctx, op, code, f)
	if err != nil {
		return nil, err
	}
	width := m.dim
	if truncateDim != 0 {
		width = truncateDim
	}
	r := wire.NewReader(reply.Payload)
	flat := r.F32s("embeddings")
	if err := r.Done(); err != nil {
		return nil, protocolErr(op, "%v", err)
	}
	if len(flat) != n*width {
		return nil, protocolErr(op, "%d values for %d rows of %d", len(flat), n, width)
	}
	rows := make([][]float32, n)
	for i := range rows {
		rows[i] = flat[i*width : (i+1)*width : (i+1)*width]
	}
	return rows, nil
}
