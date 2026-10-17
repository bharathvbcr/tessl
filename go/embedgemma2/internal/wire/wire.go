// Package wire is the frame format between the Go package and the Rust engine
// (go/engine/src/wire.rs is the other half).
//
// It has no cgo, so its tests run without the engine archive or a device.
//
// Gusset moves bytes and an opcode; the schema is the adopter's. A request is a
// sequence of fields, all little-endian:
//
//   - a scalar is one 64-bit word;
//   - a blob is a u64 length, that many bytes, then zero bytes up to the next
//     multiple of 8, so every field starts 8-aligned whatever the lengths were.
//
// Typed arrays (u32, u64, f32, f64) are blobs of their elements. A reply is a
// 16-byte header (status u32, flags u32, work_ns u64) followed by the
// operation's payload, or by tessl's error message when the status is not
// StatusOK.
package wire

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"unicode/utf8"
)

// Version is the frame version. The engine archive reports its own and the
// package refuses a mismatch.
const Version = 1

// Opcodes, as registered by the engine (go/engine/src/lib.rs).
const (
	OpLoad              uint32 = 1
	OpFree              uint32 = 2
	OpInfo              uint32 = 3
	OpSetMaxTokens      uint32 = 4
	OpPrompts           uint32 = 5
	OpTokenize          uint32 = 6
	OpDecode            uint32 = 7
	OpEncode            uint32 = 8
	OpEmbedTexts        uint32 = 9
	OpEmbedImages       uint32 = 10
	OpEmbedAudio        uint32 = 11
	OpEmbedVideos       uint32 = 12
	OpEmbedImageSources uint32 = 13
	OpEmbedAudioSources uint32 = 14
	OpEmbedVideoSources uint32 = 15
	OpImageSoftTokens   uint32 = 16
	OpAudioSoftTokens   uint32 = 17
	OpVideoSoftTokens   uint32 = 18
	OpEmbedSegments     uint32 = 19
	OpPing              uint32 = 20
	OpSelfTestPanic     uint32 = 0x7fff_0001
)

// Reply statuses.
const (
	StatusOK      uint32 = 0
	StatusErr     uint32 = 1
	StatusPanic   uint32 = 2
	StatusNoModel uint32 = 3
)

// FlagPoisoned is the reply flag that says the model's runtime is poisoned.
const FlagPoisoned uint32 = 1

// ReplyHeader is the size of a reply header in bytes.
const ReplyHeader = 16

// MaxFrame is the largest request frame: gusset's buffer ceiling (1 GiB,
// gusset.MaxBufferBytes), which is also all one request can ever hold.
const MaxFrame = 1 << 30

type kind uint8

const (
	kScalar  kind = iota
	kBytes        // one blob of byte chunks laid end to end
	kStrings      // one blob of strings laid end to end
	kU32s
	kU64s
	kF32s // a blob of f32 chunks laid end to end
	kF64s
)

type part struct {
	kind   kind
	v      uint64
	chunks [][]byte
	strs   []string
	u32s   []uint32
	u64s   []uint64
	f32s   [][]float32
	f64s   []float64
	// n is the byte length of the payload of a blob part (before padding).
	n uint64
}

func pad8(n uint64) uint64 { return (8 - n%8) % 8 }

// Frame is a request under construction. It holds references to the caller's
// slices instead of copying them, so a large image or clip is copied once, into
// the buffer that carries it.
type Frame struct {
	parts []part
	size  uint64
	err   error
}

func (f *Frame) add(p part, payload uint64) {
	if payload > MaxFrame {
		f.setErr(fmt.Errorf("a single field of %d bytes exceeds the %d-byte frame limit", payload, uint64(MaxFrame)))
		return
	}
	p.n = payload
	f.parts = append(f.parts, p)
	f.size += 8
	if p.kind != kScalar {
		f.size += payload + pad8(payload)
	}
	if f.size > MaxFrame {
		f.setErr(fmt.Errorf("the request grew to %d bytes, past the %d-byte frame limit (gusset's buffer ceiling); pass files by path instead of bytes", f.size, uint64(MaxFrame)))
	}
}

func (f *Frame) setErr(err error) {
	if f.err == nil {
		f.err = err
	}
}

// U64 appends a scalar.
func (f *Frame) U64(v uint64) *Frame {
	f.add(part{kind: kScalar, v: v}, 0)
	return f
}

// U32 appends a scalar.
func (f *Frame) U32(v uint32) *Frame { return f.U64(uint64(v)) }

// Bytes appends one blob made of the chunks laid end to end.
func (f *Frame) Bytes(chunks ...[]byte) *Frame {
	var n uint64
	for _, c := range chunks {
		n += uint64(len(c))
	}
	f.add(part{kind: kBytes, chunks: chunks}, n)
	return f
}

// Strings appends one blob made of the strings laid end to end.
func (f *Frame) Strings(s ...string) *Frame {
	var n uint64
	for _, x := range s {
		n += uint64(len(x))
	}
	f.add(part{kind: kStrings, strs: s}, n)
	return f
}

// Text appends a string as a blob.
func (f *Frame) Text(s string) *Frame { return f.Strings(s) }

// U32s appends a u32 array.
func (f *Frame) U32s(v []uint32) *Frame {
	f.add(part{kind: kU32s, u32s: v}, 4*uint64(len(v)))
	return f
}

// U64s appends a u64 array.
func (f *Frame) U64s(v []uint64) *Frame {
	f.add(part{kind: kU64s, u64s: v}, 8*uint64(len(v)))
	return f
}

// F32s appends one f32 array made of the chunks laid end to end.
func (f *Frame) F32s(chunks ...[]float32) *Frame {
	var n uint64
	for _, c := range chunks {
		n += 4 * uint64(len(c))
	}
	f.add(part{kind: kF32s, f32s: chunks}, n)
	return f
}

// F64s appends an f64 array.
func (f *Frame) F64s(v []float64) *Frame {
	f.add(part{kind: kF64s, f64s: v}, 8*uint64(len(v)))
	return f
}

// Size is the frame's length in bytes.
func (f *Frame) Size() int { return int(f.size) }

// Err reports why the frame could not be built (a field or the whole past the
// frame limit).
func (f *Frame) Err() error { return f.err }

// Fill writes the frame into dst, which must be exactly Size() bytes.
func (f *Frame) Fill(dst []byte) error {
	if f.err != nil {
		return f.err
	}
	if uint64(len(dst)) != f.size {
		return fmt.Errorf("wire: destination is %d bytes, the frame is %d", len(dst), f.size)
	}
	le := binary.LittleEndian
	at := uint64(0)
	for i := range f.parts {
		p := &f.parts[i]
		if p.kind == kScalar {
			le.PutUint64(dst[at:], p.v)
			at += 8
			continue
		}
		le.PutUint64(dst[at:], p.n)
		at += 8
		body := dst[at : at+p.n]
		switch p.kind {
		case kBytes:
			o := 0
			for _, c := range p.chunks {
				o += copy(body[o:], c)
			}
		case kStrings:
			o := 0
			for _, s := range p.strs {
				o += copy(body[o:], s)
			}
		case kU32s:
			for j, x := range p.u32s {
				le.PutUint32(body[4*j:], x)
			}
		case kU64s:
			for j, x := range p.u64s {
				le.PutUint64(body[8*j:], x)
			}
		case kF32s:
			o := 0
			for _, c := range p.f32s {
				for _, x := range c {
					le.PutUint32(body[o:], math.Float32bits(x))
					o += 4
				}
			}
		case kF64s:
			for j, x := range p.f64s {
				le.PutUint64(body[8*j:], math.Float64bits(x))
			}
		}
		at += p.n
		// The padding bytes are zero: dst may be a reused buffer.
		for z := pad8(p.n); z > 0; z-- {
			dst[at] = 0
			at++
		}
	}
	return nil
}

// ValidUTF8 reports whether s is valid UTF-8 and has no NUL byte, the shape of
// a path or a name tessl accepts.
func ValidUTF8(s string) bool {
	if !utf8.ValidString(s) {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] == 0 {
			return false
		}
	}
	return true
}

// Reply is a parsed reply.
type Reply struct {
	Status  uint32
	Flags   uint32
	WorkNS  uint64
	Payload []byte
}

// ErrShortReply is returned for a reply shorter than its header.
var ErrShortReply = errors.New("wire: reply shorter than its 16-byte header")

// ParseReply splits a reply into its header and payload. The payload aliases b.
func ParseReply(b []byte) (Reply, error) {
	if len(b) < ReplyHeader {
		return Reply{}, fmt.Errorf("%w (%d bytes)", ErrShortReply, len(b))
	}
	le := binary.LittleEndian
	return Reply{
		Status:  le.Uint32(b[0:4]),
		Flags:   le.Uint32(b[4:8]),
		WorkNS:  le.Uint64(b[8:16]),
		Payload: b[ReplyHeader:],
	}, nil
}

// Reader reads a reply payload. After the first failure every read returns the
// zero value and Err reports the failure, so a decoder checks once at the end.
type Reader struct {
	b   []byte
	err error
}

// NewReader reads b.
func NewReader(b []byte) *Reader { return &Reader{b: b} }

func (r *Reader) take(n uint64, what string) []byte {
	if r.err != nil {
		return nil
	}
	if n > uint64(len(r.b)) {
		r.err = fmt.Errorf("truncated reply: %s needs %d bytes, %d remain", what, n, len(r.b))
		return nil
	}
	out := r.b[:n]
	r.b = r.b[n:]
	return out
}

// U64 reads a scalar.
func (r *Reader) U64(what string) uint64 {
	b := r.take(8, what)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint64(b)
}

// U32 reads a scalar that must fit 32 bits.
func (r *Reader) U32(what string) uint32 {
	v := r.U64(what)
	if v > math.MaxUint32 {
		if r.err == nil {
			r.err = fmt.Errorf("%s: %d does not fit 32 bits", what, v)
		}
		return 0
	}
	return uint32(v)
}

// I32 reads a sign-extended scalar.
func (r *Reader) I32(what string) int32 {
	v := int64(r.U64(what))
	if v < math.MinInt32 || v > math.MaxInt32 {
		if r.err == nil {
			r.err = fmt.Errorf("%s: %d does not fit a signed 32-bit integer", what, v)
		}
		return 0
	}
	return int32(v)
}

// Blob reads a blob, aliasing the reply.
func (r *Reader) Blob(what string) []byte {
	n := r.U64(what)
	if r.err != nil {
		return nil
	}
	body := r.take(n, what)
	r.take(pad8(n), what)
	if r.err != nil {
		return nil
	}
	return body
}

func (r *Reader) array(width uint64, what string) []byte {
	b := r.Blob(what)
	if r.err == nil && uint64(len(b))%width != 0 {
		r.err = fmt.Errorf("%s: %d bytes is not a whole number of %d-byte elements", what, len(b), width)
		return nil
	}
	return b
}

// U32s reads a u32 array.
func (r *Reader) U32s(what string) []uint32 {
	b := r.array(4, what)
	out := make([]uint32, len(b)/4)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(b[4*i:])
	}
	return out
}

// U64s reads a u64 array.
func (r *Reader) U64s(what string) []uint64 {
	b := r.array(8, what)
	out := make([]uint64, len(b)/8)
	for i := range out {
		out[i] = binary.LittleEndian.Uint64(b[8*i:])
	}
	return out
}

// F32s reads an f32 array.
func (r *Reader) F32s(what string) []float32 {
	b := r.array(4, what)
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

// Err is the first failure, if any.
func (r *Reader) Err() error { return r.err }

// Done is Err, or a failure for bytes left unread: a reply the decoder did not
// consume is a decoder and an engine that disagree.
func (r *Reader) Done() error {
	if r.err != nil {
		return r.err
	}
	if len(r.b) != 0 {
		return fmt.Errorf("%d unread bytes after the last field", len(r.b))
	}
	return nil
}
