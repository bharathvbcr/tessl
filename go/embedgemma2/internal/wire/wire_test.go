package wire

import (
	"bytes"
	"encoding/binary"
	"flag"
	"math"
	"os"
	"strings"
	"testing"
)

var update = flag.Bool("update", false, "rewrite testdata/golden_request.bin")

// goldenRequest is the frame testdata/golden_request.bin holds. The Rust
// engine's wire.rs decodes the same file (golden_request_decodes_as_the_go_side_wrote_it),
// so the two halves are checked against one byte string, not against each
// other's idea of it.
func goldenRequest() *Frame {
	return new(Frame).
		U32(7).
		Strings("héllo", "wörld"). // 6 + 6 bytes: pads by 4
		U64s([]uint64{0, 6, 12}).
		F32s([]float32{1.5, -2}, []float32{0.25}).
		F64s([]float64{3.5, math.Inf(-1)}).
		Bytes([]byte{1, 2, 3}, []byte{4}). // 4 bytes: pads by 4
		Text("").
		U32s([]uint32{1, 2, 3}). // 12 bytes: pads by 4
		U64(math.MaxUint64)
}

func fill(t *testing.T, f *Frame) []byte {
	t.Helper()
	b := make([]byte, f.Size())
	if err := f.Fill(b); err != nil {
		t.Fatal(err)
	}
	return b
}

func TestUnitGoldenRequestIsByteForByteWhatTheRustSideDecodes(t *testing.T) {
	got := fill(t, goldenRequest())
	const path = "testdata/golden_request.bin"
	if *update {
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("the frame is %d bytes and differs from %s (%d bytes): if the format changed on purpose, bump Version, update wire.rs, and rerun with -update", len(got), path, len(want))
	}
}

func TestUnitGoldenReplyFromRustParses(t *testing.T) {
	raw, err := os.ReadFile("testdata/golden_reply.bin")
	if err != nil {
		t.Fatal(err)
	}
	rep, err := ParseReply(raw)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Status != StatusOK || rep.Flags != FlagPoisoned || rep.WorkNS != 123456789 {
		t.Fatalf("header %+v", rep)
	}
	r := NewReader(rep.Payload)
	if v := r.U32("a"); v != 42 {
		t.Errorf("u32 %d", v)
	}
	if v := r.I32("b"); v != -1 {
		t.Errorf("i32 %d", v)
	}
	if v := r.U64("c"); v != 1<<40 {
		t.Errorf("u64 %d", v)
	}
	if v := string(r.Blob("d")); v != "hello" {
		t.Errorf("blob %q", v)
	}
	if v := r.U32s("e"); len(v) != 3 || v[0] != 9 || v[2] != 11 {
		t.Errorf("u32s %v", v)
	}
	if v := r.U64s("f"); len(v) != 2 || v[1] != 1<<33 {
		t.Errorf("u64s %v", v)
	}
	if v := r.F32s("g"); len(v) != 3 || v[0] != 0.5 || v[1] != -1.25 || v[2] != 3 {
		t.Errorf("f32s %v", v)
	}
	if err := r.Done(); err != nil {
		t.Fatal(err)
	}
}

func TestUnitFramePaddingKeepsEveryFieldEightAlignedAndZero(t *testing.T) {
	f := new(Frame).Bytes([]byte{1, 2, 3}).U32(5).Text("abcdefghi") // 3 -> 5 pad, 9 -> 7 pad
	if f.Size() != (8+8)+8+(8+16) {
		t.Fatalf("size %d", f.Size())
	}
	dirty := bytes.Repeat([]byte{0xff}, f.Size())
	if err := f.Fill(dirty); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(dirty, fill(t, f)) {
		t.Fatal("padding depends on the buffer's old contents")
	}
	le := binary.LittleEndian
	if le.Uint64(dirty[0:]) != 3 || !bytes.Equal(dirty[8:11], []byte{1, 2, 3}) || !bytes.Equal(dirty[11:16], make([]byte, 5)) {
		t.Fatalf("first blob %x", dirty[:16])
	}
	if le.Uint64(dirty[16:]) != 5 {
		t.Fatalf("scalar %x", dirty[16:24])
	}
	if le.Uint64(dirty[24:]) != 9 || string(dirty[32:41]) != "abcdefghi" || !bytes.Equal(dirty[41:48], make([]byte, 7)) {
		t.Fatalf("second blob %x", dirty[24:])
	}
}

func TestUnitFillRefusesAWrongSizedDestination(t *testing.T) {
	f := new(Frame).U32(1)
	if err := f.Fill(make([]byte, 7)); err == nil || !strings.Contains(err.Error(), "destination") {
		t.Errorf("short: %v", err)
	}
	if err := f.Fill(make([]byte, 9)); err == nil {
		t.Error("long destination accepted")
	}
}

func TestUnitFrameLimitIsReportedNotTruncated(t *testing.T) {
	f := new(Frame)
	f.add(part{kind: kBytes}, MaxFrame+1)
	if f.Err() == nil || !strings.Contains(f.Err().Error(), "frame limit") {
		t.Fatalf("one field past the limit: %v", f.Err())
	}
	if err := f.Fill(nil); err == nil {
		t.Error("Fill of a failed frame succeeded")
	}
	// Two fields that together pass it; the first error is kept.
	g := new(Frame)
	g.add(part{kind: kBytes}, MaxFrame/2)
	g.add(part{kind: kBytes}, MaxFrame/2)
	if g.Err() == nil || !strings.Contains(g.Err().Error(), "pass files by path") {
		t.Fatalf("a frame past the limit: %v", g.Err())
	}
	first := g.Err()
	g.U32(1)
	if g.Err() != first {
		t.Error("a later field replaced the first error")
	}
	// Exactly at the limit is fine.
	h := new(Frame)
	h.add(part{kind: kBytes}, MaxFrame-8)
	if h.Err() != nil {
		t.Fatalf("a frame of exactly the limit: %v", h.Err())
	}
}

func TestUnitReaderRefusesTruncationOverflowAndLeftovers(t *testing.T) {
	le := binary.LittleEndian
	word := func(v uint64) []byte { return le.AppendUint64(nil, v) }

	r := NewReader(word(8)) // a blob of 8 bytes with none behind it
	if b := r.Blob("x"); b != nil || r.Err() == nil || !strings.Contains(r.Err().Error(), "truncated") {
		t.Errorf("blob past the end: %v %v", b, r.Err())
	}
	r = NewReader(word(1 << 40))
	r.Blob("x")
	if r.Err() == nil {
		t.Error("an absurd length was accepted")
	}
	r = NewReader(word(math.MaxUint64))
	r.Blob("x")
	if r.Err() == nil {
		t.Error("a length of 2^64-1 was accepted")
	}
	r = NewReader(word(1 << 32))
	if r.U32("x"); r.Err() == nil || !strings.Contains(r.Err().Error(), "32 bits") {
		t.Errorf("u32 overflow: %v", r.Err())
	}
	r = NewReader(word(1 << 40))
	if r.I32("x"); r.Err() == nil {
		t.Error("i32 overflow accepted")
	}
	r = NewReader(word(^uint64(0))) // -1
	if v := r.I32("x"); v != -1 || r.Err() != nil {
		t.Errorf("i32 -1: %d %v", v, r.Err())
	}
	// A blob of 3 bytes is not a u32 array.
	r = NewReader(append(word(3), 1, 2, 3, 0, 0, 0, 0, 0))
	if r.U32s("x"); r.Err() == nil || !strings.Contains(r.Err().Error(), "whole number") {
		t.Errorf("ragged array: %v", r.Err())
	}
	// A blob whose padding is cut off.
	r = NewReader(append(word(3), 1, 2, 3))
	if r.Blob("x"); r.Err() == nil {
		t.Error("a missing pad was accepted")
	}
	// Bytes left over.
	r = NewReader(append(word(7), 0, 0, 0, 0, 0, 0, 0, 0xee))
	r.U64("x")
	if err := r.Done(); err == nil || !strings.Contains(err.Error(), "unread") {
		t.Errorf("leftovers: %v", err)
	}
	// The first error is sticky and later reads return zero values.
	r = NewReader(nil)
	r.U64("first")
	first := r.Err()
	if r.U32("second") != 0 || r.Blob("third") != nil || r.Err() != first {
		t.Error("a failed reader kept reading or replaced its error")
	}
}

func TestUnitRequestsRoundTripThroughTheReader(t *testing.T) {
	// The reply reader reads the same blob layout the request writer lays down.
	b := fill(t, new(Frame).U32(9).U64s([]uint64{3, 1 << 33}).F32s([]float32{1, 2}, nil, []float32{3}).Text("xyz"))
	r := NewReader(b)
	if r.U32("scalar") != 9 {
		t.Error("scalar")
	}
	if v := r.U64s("u64s"); len(v) != 2 || v[0] != 3 || v[1] != 1<<33 {
		t.Errorf("u64s %v", v)
	}
	if v := r.F32s("f32s"); len(v) != 3 || v[2] != 3 {
		t.Errorf("f32s %v", v)
	}
	if string(r.Blob("text")) != "xyz" {
		t.Error("text")
	}
	if err := r.Done(); err != nil {
		t.Fatal(err)
	}
}

func TestUnitParseReplyRefusesAShortHeader(t *testing.T) {
	for n := 0; n < ReplyHeader; n++ {
		if _, err := ParseReply(make([]byte, n)); err == nil {
			t.Errorf("a %d-byte reply parsed", n)
		}
	}
	rep, err := ParseReply(make([]byte, ReplyHeader))
	if err != nil || rep.Status != StatusOK || len(rep.Payload) != 0 {
		t.Errorf("an empty OK reply: %+v %v", rep, err)
	}
	raw := make([]byte, ReplyHeader, ReplyHeader+5)
	binary.LittleEndian.PutUint32(raw[0:], StatusErr)
	raw = append(raw, "oops!"...)
	rep, err = ParseReply(raw)
	if err != nil || rep.Status != StatusErr || string(rep.Payload) != "oops!" {
		t.Errorf("a failure reply: %+v %v", rep, err)
	}
}

func TestUnitValidUTF8RefusesNULAndBadSequences(t *testing.T) {
	for s, want := range map[string]bool{"": true, "plain": true, "hé": true, "a\x00b": false, "\xff": false, "\xc3": false} {
		if got := ValidUTF8(s); got != want {
			t.Errorf("ValidUTF8(%q) = %v", s, got)
		}
	}
}
