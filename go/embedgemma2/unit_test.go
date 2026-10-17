//go:build darwin && arm64

package embedgemma2

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/bharathvbcr/tessl/go/embedgemma2/internal/wire"
)

// These tests need no GPU and no checkpoint: they build a Model that holds no
// engine handle, and call methods whose arguments are refused in Go before a
// request is framed. A refusal that reached the (absent) handle would panic, so
// each case also proves the check runs first.

func bare() *Model { return &Model{dim: 8, multimodal: true} }

func wantArg(t *testing.T, what string, err error, field string) {
	t.Helper()
	var ae *ArgError
	if !errors.As(err, &ae) {
		t.Errorf("%s: got %v, want an *ArgError", what, err)
		return
	}
	if !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("%s: does not match ErrInvalidArgument", what)
	}
	if !strings.HasPrefix(ae.Field, field) {
		t.Errorf("%s: field %q, want %q", what, ae.Field, field)
	}
}

func rgb(h, w int) Image { return Image{Pix: make([]byte, h*w*3), Height: h, Width: w, Channels: 3} }

func clip(n, rate int) Audio { return Audio{Samples: make([]float32, n), SampleRate: rate} }

func vid(frames, h, w int, fps float64) Video {
	return Video{Frames: make([]byte, frames*h*w*3), NumFrames: frames, Height: h, Width: w, FPS: fps}
}

func TestUnitBatchSizeIsBounded(t *testing.T) {
	ctx, m := context.Background(), bare()
	wantArg(t, "no texts", func() error { _, err := m.EmbedTexts(ctx, nil, "", 0); return err }(), "texts")
	wantArg(t, "too many texts", func() error {
		_, err := m.EmbedTexts(ctx, make([]string, MaxBatch+1), "", 0)
		return err
	}(), "texts")
	wantArg(t, "no ids", func() error { _, err := m.Encode(ctx, nil, 0); return err }(), "ids")
	wantArg(t, "no images", func() error { _, err := m.EmbedImages(ctx, nil, 0); return err }(), "images")
	wantArg(t, "no clips", func() error { _, err := m.EmbedAudio(ctx, nil, 0); return err }(), "clips")
	wantArg(t, "no videos", func() error { _, err := m.EmbedVideos(ctx, nil, false, 0); return err }(), "videos")
	wantArg(t, "no paths", func() error { _, err := m.EmbedImageFiles(ctx, nil, 0); return err }(), "paths")
	wantArg(t, "no data", func() error { _, err := m.EmbedAudioBytes(ctx, nil, 0); return err }(), "data")
	wantArg(t, "no items", func() error { _, err := m.EmbedMixed(ctx, nil, 0); return err }(), "items")
	wantArg(t, "too many tokenize texts", func() error {
		_, err := m.Tokenize(ctx, make([]string, MaxTokenizeTexts+1), "")
		return err
	}(), "texts")
	wantArg(t, "too many ids", func() error {
		_, err := m.Decode(ctx, make([]uint32, MaxDecodeIDs+1), true)
		return err
	}(), "ids")
}

func TestUnitTruncateDimIsBoundedByTheModelWidth(t *testing.T) {
	ctx, m := context.Background(), bare()
	for _, bad := range []int{-1, 9, math.MaxInt32} {
		wantArg(t, "texts", func() error { _, err := m.EmbedTexts(ctx, []string{"a"}, "", bad); return err }(), "truncateDim")
		wantArg(t, "images", func() error { _, err := m.EmbedImages(ctx, []Image{rgb(2, 2)}, bad); return err }(), "truncateDim")
		wantArg(t, "mixed", func() error {
			_, err := m.EmbedMixed(ctx, [][]Segment{{Text("a")}}, bad)
			return err
		}(), "truncateDim")
	}
	wantArg(t, "SetMaxTokens low", m.SetMaxTokens(ctx, 0), "n")
	wantArg(t, "SetMaxTokens high", m.SetMaxTokens(ctx, maxTokensLimit+1), "n")
}

func TestUnitTextsAndPromptsAreCheckedForEncodingAndSize(t *testing.T) {
	ctx, m := context.Background(), bare()
	wantArg(t, "invalid UTF-8 text", func() error {
		_, err := m.EmbedTexts(ctx, []string{"fine", "bad \xff"}, "", 0)
		return err
	}(), "texts")
	wantArg(t, "invalid UTF-8 prompt", func() error { _, err := m.EmbedTexts(ctx, []string{"a"}, "\xc3", 0); return err }(), "prompt")
	wantArg(t, "long prompt", func() error {
		_, err := m.EmbedTexts(ctx, []string{"a"}, strings.Repeat("p", MaxPromptNameBytes+1), 0)
		return err
	}(), "prompt")
	wantArg(t, "tokenize prompt", func() error { _, err := m.Tokenize(ctx, []string{"a"}, "\xc3"); return err }(), "prompt")
	wantArg(t, "empty sequence", func() error { _, err := m.Encode(ctx, [][]uint32{{1, 2}, {}}, 0); return err }(), "ids")
}

func TestUnitRawImagesAreCheckedAgainstTheirDimensions(t *testing.T) {
	ctx, m := context.Background(), bare()
	short := rgb(4, 4)
	short.Pix = short.Pix[:len(short.Pix)-1]
	for what, im := range map[string]Image{
		"short pixels":      short,
		"long pixels":       {Pix: make([]byte, 4*4*3+1), Height: 4, Width: 4, Channels: 3},
		"zero channels":     {Pix: make([]byte, 16), Height: 4, Width: 4, Channels: 0},
		"five channels":     {Pix: make([]byte, 80), Height: 4, Width: 4, Channels: 5},
		"zero height":       {Pix: nil, Height: 0, Width: 4, Channels: 3},
		"negative width":    {Pix: nil, Height: 4, Width: -1, Channels: 3},
		"product overflows": {Pix: nil, Height: math.MaxInt32, Width: math.MaxInt32, Channels: 3},
	} {
		wantArg(t, what, func() error { _, err := m.EmbedImages(ctx, []Image{rgb(2, 2), im}, 0); return err }(), "images[1]")
	}
}

func TestUnitRawClipsAreCheckedForLengthAndRate(t *testing.T) {
	ctx, m := context.Background(), bare()
	wantArg(t, "empty clip", func() error { _, err := m.EmbedAudio(ctx, []Audio{clip(0, 16000)}, 0); return err }(), "clips[0]")
	wantArg(t, "zero rate", func() error { _, err := m.EmbedAudio(ctx, []Audio{clip(10, 0)}, 0); return err }(), "clips[0]")
	wantArg(t, "negative rate", func() error { _, err := m.EmbedAudio(ctx, []Audio{clip(10, -16000)}, 0); return err }(), "clips[0]")
	wantArg(t, "mixed rates", func() error {
		_, err := m.EmbedAudio(ctx, []Audio{clip(10, 16000), clip(10, 8000)}, 0)
		return err
	}(), "clips[1]")
}

func TestUnitRawVideosAreCheckedForFramesAndRate(t *testing.T) {
	ctx, m := context.Background(), bare()
	short := vid(2, 4, 4, 2)
	short.Frames = short.Frames[:len(short.Frames)-3]
	for what, v := range map[string]Video{
		"short frames": short,
		"no frames":    {NumFrames: 0, Height: 4, Width: 4, FPS: 2},
		"zero fps":     vid(1, 2, 2, 0),
		"negative fps": vid(1, 2, 2, -1),
		"nan fps":      vid(1, 2, 2, math.NaN()),
		"infinite fps": vid(1, 2, 2, math.Inf(1)),
		"zero width":   {Frames: nil, NumFrames: 1, Height: 4, Width: 0, FPS: 2},
		"too many":     {NumFrames: MaxVideoFrames + 1, Height: 1, Width: 1, FPS: 2},
	} {
		wantArg(t, what, func() error { _, err := m.EmbedVideos(ctx, []Video{vid(1, 2, 2, 1), v}, true, 0); return err }(), "videos[1]")
	}
	wantArg(t, "EmbedVideo names its own op", func() error { _, err := m.EmbedVideo(ctx, short, false, 0); return err }(), "videos[0]")
}

func TestUnitFilesAndBytesAreCheckedBeforeTheyCross(t *testing.T) {
	ctx, m := context.Background(), bare()
	wantArg(t, "empty path", func() error { _, err := m.EmbedImageFiles(ctx, []string{"a.png", ""}, 0); return err }(), "paths[1]")
	wantArg(t, "NUL in path", func() error { _, err := m.EmbedAudioFiles(ctx, []string{"a\x00b.wav"}, 0); return err }(), "paths[0]")
	wantArg(t, "long path", func() error {
		_, err := m.EmbedVideoFiles(ctx, []string{strings.Repeat("p", MaxPathBytes+1)}, false, 0)
		return err
	}(), "paths[0]")
	wantArg(t, "empty bytes", func() error { _, err := m.EmbedImageBytes(ctx, [][]byte{{1}, {}}, 0); return err }(), "data[1]")
	wantArg(t, "bad extension", func() error { _, err := m.EmbedVideoBytes(ctx, [][]byte{{1}}, "mp 4", false, 0); return err }(), "extension")
	wantArg(t, "long extension", func() error {
		_, err := m.EmbedVideoBytes(ctx, [][]byte{{1}}, strings.Repeat("a", MaxExtensionBytes+1), false, 0)
		return err
	}(), "extension")
	wantArg(t, "extension on paths", func() error {
		_, err := m.embedSources(ctx, "EmbedVideoFiles", wire.OpEmbedVideoSources, "paths", []string{"a.mp4"}, nil, false, "mp4", false, 0)
		return err
	}(), "extension")
}

func TestUnitSoftTokenCountArgumentsAreChecked(t *testing.T) {
	ctx, m := context.Background(), bare()
	wantArg(t, "no sizes", func() error { _, err := m.ImageSoftTokens(ctx, nil); return err }(), "sizes")
	wantArg(t, "empty size", func() error { _, err := m.ImageSoftTokens(ctx, []ImageSize{{4, 4}, {0, 4}}); return err }(), "sizes[1]")
	wantArg(t, "no samples", func() error { _, err := m.AudioSoftTokens(ctx, nil, 16000); return err }(), "sampleCounts")
	wantArg(t, "zero samples", func() error { _, err := m.AudioSoftTokens(ctx, []int{10, 0}, 16000); return err }(), "sampleCounts[1]")
	wantArg(t, "bad rate", func() error { _, err := m.AudioSoftTokens(ctx, []int{10}, 0); return err }(), "sampleRate")
	wantArg(t, "no videos", func() error { _, err := m.VideoSoftTokens(ctx, nil); return err }(), "videos")
	wantArg(t, "bad fps", func() error {
		_, err := m.VideoSoftTokens(ctx, []VideoInfo{{Height: 4, Width: 4, Frames: 2, FPS: math.NaN()}})
		return err
	}(), "videos[0]")
	wantArg(t, "no frames", func() error {
		_, err := m.VideoSoftTokens(ctx, []VideoInfo{{Height: 4, Width: 4, Frames: 0, FPS: 2}})
		return err
	}(), "videos[0]")
}

func TestUnitMixedItemsAreCheckedAndOneSourcePerModality(t *testing.T) {
	ctx, m := context.Background(), bare()
	img := rgb(2, 2)
	wantArg(t, "empty item", func() error { _, err := m.EmbedMixed(ctx, [][]Segment{{Text("a")}, {}}, 0); return err }(), "items[1]")
	wantArg(t, "zero segment", func() error { _, err := m.EmbedMixed(ctx, [][]Segment{{{}}}, 0); return err }(), "items[0][0]")
	wantArg(t, "bad text", func() error { _, err := m.EmbedMixed(ctx, [][]Segment{{Text("\xff")}}, 0); return err }(), "items[0][0]")
	wantArg(t, "bad raw image", func() error {
		_, err := m.EmbedMixed(ctx, [][]Segment{{ImageSegment(Image{Pix: []byte{1}, Height: 2, Width: 2, Channels: 3})}}, 0)
		return err
	}(), "items[0][0]")

	// Source mixing is named by position, both ends.
	err := func() error {
		_, err := m.EmbedMixed(ctx, [][]Segment{{ImageSegment(img), Text("x")}, {ImageFileSegment("a.png")}}, 0)
		return err
	}()
	wantArg(t, "raw then path", err, "items")
	if err == nil || !strings.Contains(err.Error(), "items[1][0]") || !strings.Contains(err.Error(), "items[0][0]") {
		t.Errorf("the message should name both positions: %v", err)
	}
	wantArg(t, "path then bytes", func() error {
		_, err := m.EmbedMixed(ctx, [][]Segment{{AudioFileSegment("a.wav"), AudioBytesSegment([]byte{1})}}, 0)
		return err
	}(), "items")
	wantArg(t, "two rates", func() error {
		_, err := m.EmbedMixed(ctx, [][]Segment{{AudioSegment(clip(5, 16000)), AudioSegment(clip(5, 8000))}}, 0)
		return err
	}(), "items[0][1]")
	wantArg(t, "two extensions", func() error {
		_, err := m.EmbedMixed(ctx, [][]Segment{{VideoBytesSegment([]byte{1}, "mp4", false), VideoBytesSegment([]byte{1}, "mov", false)}}, 0)
		return err
	}(), "items[0][1]")
	wantArg(t, "extension on a path", func() error {
		s := VideoFileSegment("a.mp4", false)
		s.ext = "mp4"
		_, err := m.EmbedMixed(ctx, [][]Segment{{s}}, 0)
		return err
	}(), "items[0][0]")

	// Different modalities may use different sources: the plan accepts it.
	p, err := planMixed("EmbedMixed", [][]Segment{{
		Text("a"), ImageFileSegment("a.png"), AudioBytesSegment([]byte{1, 2}), VideoSegment(vid(1, 2, 2, 1), true),
	}})
	if err != nil {
		t.Fatal(err)
	}
	if p.image.source != sourcePaths || p.audio.source != sourceBytes || p.video.source != sourceRaw {
		t.Errorf("sources %d %d %d", p.image.source, p.audio.source, p.video.source)
	}
	if got := p.kinds; len(got) != 4 || got[3] != segVideoTimestamp {
		t.Errorf("kinds %v", got)
	}
}

func TestUnitPlanNumbersMediaInOrderOfMention(t *testing.T) {
	p, err := planMixed("EmbedMixed", [][]Segment{
		{ImageFileSegment("one.png"), Text("a"), ImageFileSegment("two.png")},
		{Text("b"), ImageFileSegment("three.png")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []uint64{0, 0, 1, 1, 2}; len(p.refs) != 5 || p.refs[0] != 0 || p.refs[2] != 1 || p.refs[4] != 2 {
		t.Errorf("refs %v, want %v", p.refs, want)
	}
	if got := p.itemCuts; len(got) != 3 || got[1] != 3 || got[2] != 5 {
		t.Errorf("item cuts %v", got)
	}
	if got := p.image.cuts; len(got) != 4 || got[3] != uint64(len("one.png")+len("two.png")+len("three.png")) {
		t.Errorf("image path cuts %v", got)
	}
}

func TestUnitClosedAndDeadModelsRefuseCallsWithoutTouchingTheEngine(t *testing.T) {
	ctx := context.Background()
	m := bare()
	m.closed.Store(true)
	for name, call := range map[string]func() error{
		"EmbedTexts": func() error { _, err := m.EmbedTexts(ctx, []string{"a"}, "", 0); return err },
		"Tokenize":   func() error { _, err := m.Tokenize(ctx, []string{"a"}, ""); return err },
		"Decode":     func() error { _, err := m.Decode(ctx, []uint32{1}, true); return err },
		"Prompts":    func() error { _, err := m.Prompts(ctx); return err },
		"Ping":       func() error { return m.Ping(ctx) },
		"SetMax":     func() error { return m.SetMaxTokens(ctx, 10) },
		"Images":     func() error { _, err := m.EmbedImages(ctx, []Image{rgb(2, 2)}, 0); return err },
		"Mixed":      func() error { _, err := m.EmbedMixed(ctx, [][]Segment{{Text("a")}}, 0); return err },
	} {
		if err := call(); !errors.Is(err, ErrClosed) {
			t.Errorf("%s on a closed model: %v", name, err)
		}
	}
	if err := m.Close(); err != nil {
		t.Errorf("Close of a closed model: %v", err)
	}

	d := bare()
	d.markDead(&Error{Op: "EmbedTexts", Status: StatusPanic, Message: "boom"})
	err := func() error { _, err := d.EmbedTexts(ctx, []string{"a"}, "", 0); return err }()
	if !errors.Is(err, ErrPanic) || !strings.Contains(err.Error(), "boom") || !strings.Contains(err.Error(), "Load again") {
		t.Errorf("a dead model: %v", err)
	}
}

func TestUnitContextIsCheckedBeforeAnythingCrosses(t *testing.T) {
	m := bare()
	done, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := m.EmbedTexts(done, []string{"a"}, "", 0)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("a cancelled context: %v", err)
	}
	expired, cancel2 := context.WithTimeout(context.Background(), 0)
	defer cancel2()
	if _, err := m.EmbedTexts(expired, []string{"a"}, "", 0); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("an expired context: %v", err)
	}
	var nilCtx context.Context // the case under test: a caller that passes none
	if _, err := m.EmbedTexts(nilCtx, []string{"a"}, "", 0); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("a nil context: %v", err)
	}
}

func TestUnitLoadRefusesBadArgumentsBeforeOpeningAnything(t *testing.T) {
	for what, tc := range map[string]struct {
		dir  string
		opts Options
		want string
	}{
		"empty dir":         {"", Options{}, "snapshotDir"},
		"long dir":          {strings.Repeat("d", MaxPathBytes+1), Options{}, "snapshotDir"},
		"NUL dir":           {"a\x00b", Options{}, "snapshotDir"},
		"bf16 text only":    {"x", Options{Operands: BF16, TextOnly: true}, "Options.Operands"},
		"unknown operands":  {"x", Options{Operands: 7}, "Options.Operands"},
		"negative operands": {"x", Options{Operands: -1}, "Options.Operands"},
	} {
		_, err := Load(tc.dir, tc.opts)
		wantArg(t, what, err, tc.want)
	}
}
