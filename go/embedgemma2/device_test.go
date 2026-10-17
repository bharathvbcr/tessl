//go:build darwin && arm64

package embedgemma2_test

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/bharathvbcr/tessl/go/embedgemma2"
)

func TestMain(m *testing.M) {
	code := m.Run()
	if shared.model != nil {
		if err := shared.model.Close(); err != nil {
			println("closing the shared model:", err.Error())
			code = 1
		}
	}
	os.Exit(code)
}

func truncName(trunc int) string { return fmt.Sprint(trunc) }

func TestDeviceTextsMatchRustBitForBit(t *testing.T) {
	f := loadFixtures(t)
	m := sharedModel(t)
	ctx := context.Background()
	for _, prompt := range []string{"none", "query", "document"} {
		for _, trunc := range []int{0, 40} {
			name := "texts_" + prompt + "_" + truncName(trunc)
			p := prompt
			if p == "none" {
				p = ""
			}
			got, err := m.EmbedTexts(ctx, f.m.Texts, p, trunc)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			requireBits(t, name, got, f.expected(t, name))
		}
	}
}

func TestDeviceInfoAndPrompts(t *testing.T) {
	f := loadFixtures(t)
	m := sharedModel(t)
	if m.EmbeddingDim() != f.m.Dim || !m.Multimodal() || m.MaxTokens() != 8192 {
		t.Fatalf("dim %d multimodal %v max tokens %d", m.EmbeddingDim(), m.Multimodal(), m.MaxTokens())
	}
	p, err := m.Prompts(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var file struct {
		Prompts       map[string]string `json:"prompts"`
		DefaultPrompt *string           `json:"default_prompt_name"`
	}
	raw, err := os.ReadFile(f.snapshot() + "/config_sentence_transformers.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		t.Fatal(err)
	}
	if len(p.Names) != len(file.Prompts) || len(p.Names) == 0 {
		t.Fatalf("prompts %v, the file holds %v", p.Names, file.Prompts)
	}
	for name, text := range file.Prompts {
		if p.Text[name] != text {
			t.Errorf("prompt %q is %q, the file says %q", name, p.Text[name], text)
		}
	}
	if p.HasDefault != (file.DefaultPrompt != nil) {
		t.Errorf("default %q/%v, the file says %v", p.Default, p.HasDefault, file.DefaultPrompt)
	}
	// The result is a copy: editing it does not change the next answer.
	p.Text["query"] = "edited"
	again, err := m.Prompts(context.Background())
	if err != nil || again.Text["query"] == "edited" {
		t.Fatalf("the cached prompts were edited through a returned copy: %v %v", again.Text, err)
	}
}

func TestDeviceTokenizeDecodeAndEncodeMatchRust(t *testing.T) {
	f := loadFixtures(t)
	m := sharedModel(t)
	ctx := context.Background()
	ids, err := m.Tokenize(ctx, f.m.Texts, "document")
	if err != nil {
		t.Fatal(err)
	}
	flat := f.read(t, "tokenize_document_ids.u32")
	cuts := f.read(t, "tokenize_document_cuts.u64")
	if len(ids) != len(cuts)/8-1 {
		t.Fatalf("%d sequences, want %d", len(ids), len(cuts)/8-1)
	}
	for i, seq := range ids {
		lo := binary.LittleEndian.Uint64(cuts[8*i:])
		hi := binary.LittleEndian.Uint64(cuts[8*(i+1):])
		if uint64(len(seq)) != hi-lo {
			t.Fatalf("sequence %d has %d ids, want %d", i, len(seq), hi-lo)
		}
		for j, id := range seq {
			if want := binary.LittleEndian.Uint32(flat[4*(lo+uint64(j)):]); id != want {
				t.Fatalf("sequence %d id %d is %d, want %d", i, j, id, want)
			}
		}
	}
	for name, skip := range map[string]bool{"skip": true, "keep": false} {
		got, err := m.Decode(ctx, ids[0], skip)
		if err != nil {
			t.Fatal(err)
		}
		if want := string(f.read(t, "decode_"+name+".txt")); got != want {
			t.Errorf("decode(skip=%v) = %q, want %q", skip, got, want)
		}
	}
	if s, err := m.Decode(ctx, nil, true); err != nil || s != "" {
		t.Errorf("decode of no ids = %q, %v", s, err)
	}
	// Encoding the ids is embedding the texts, bit for bit.
	viaIDs, err := m.Encode(ctx, ids, 0)
	if err != nil {
		t.Fatal(err)
	}
	requireBits(t, "Encode(Tokenize)", viaIDs, f.expected(t, "texts_document_0"))
}

func TestDeviceRawMediaMatchRustBitForBit(t *testing.T) {
	f := loadFixtures(t)
	m := sharedModel(t)
	ctx := context.Background()
	for _, trunc := range []int{0, 40} {
		got, err := m.EmbedImages(ctx, f.images(t), trunc)
		if err != nil {
			t.Fatal(err)
		}
		requireBits(t, "images", got, f.expected(t, "images_"+truncName(trunc)))
		got, err = m.EmbedAudio(ctx, f.clips(t), trunc)
		if err != nil {
			t.Fatal(err)
		}
		requireBits(t, "audio", got, f.expected(t, "audio_"+truncName(trunc)))
		for name, stamps := range map[string]bool{"plain": false, "ts": true} {
			got, err = m.EmbedVideos(ctx, f.videos(t), stamps, trunc)
			if err != nil {
				t.Fatal(err)
			}
			requireBits(t, "video "+name, got, f.expected(t, "video_"+name+"_"+truncName(trunc)))
		}
	}
	// One video alone, with and without timestamps, is a row of the batch to
	// the batch tolerance; the stamped one differs from the plain one.
	one, err := m.EmbedVideo(ctx, f.videos(t)[0], true, 0)
	if err != nil {
		t.Fatal(err)
	}
	plain, err := m.EmbedVideo(ctx, f.videos(t)[0], false, 0)
	if err != nil {
		t.Fatal(err)
	}
	if maxAbs(one, plain) < 1e-4 {
		t.Error("timestamps changed nothing")
	}
}

func maxAbs(a, b []float32) float64 {
	var m float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		if d < 0 {
			d = -d
		}
		if d > m {
			m = d
		}
	}
	return m
}

func TestDeviceFilesAndBytesMatchRustBitForBit(t *testing.T) {
	f := loadFixtures(t)
	m := sharedModel(t)
	ctx := context.Background()
	images := []string{"rgb.png", "gray.png"}
	clips := []string{"pcm16_16k_mono.wav", "tone_44k_stereo.flac"}

	var paths []string
	var data [][]byte
	for _, n := range images {
		paths = append(paths, mediaPath(n))
		data = append(data, mediaBytes(t, n))
	}
	got, err := m.EmbedImageFiles(ctx, paths, 0)
	if err != nil {
		t.Fatal(err)
	}
	requireBits(t, "image files", got, f.expected(t, "image_files"))
	got, err = m.EmbedImageBytes(ctx, data, 0)
	if err != nil {
		t.Fatal(err)
	}
	requireBits(t, "image bytes", got, f.expected(t, "image_bytes"))

	paths, data = nil, nil
	for _, n := range clips {
		paths = append(paths, mediaPath(n))
		data = append(data, mediaBytes(t, n))
	}
	got, err = m.EmbedAudioFiles(ctx, paths, 0)
	if err != nil {
		t.Fatal(err)
	}
	requireBits(t, "audio files", got, f.expected(t, "audio_files"))
	got, err = m.EmbedAudioBytes(ctx, data, 0)
	if err != nil {
		t.Fatal(err)
	}
	requireBits(t, "audio bytes", got, f.expected(t, "audio_bytes"))

	for name, stamps := range map[string]bool{"plain": false, "ts": true} {
		got, err = m.EmbedVideoFiles(ctx, []string{mediaPath("colors.mp4")}, stamps, 0)
		if err != nil {
			t.Fatal(err)
		}
		requireBits(t, "video files "+name, got, f.expected(t, "video_files_"+name))
		got, err = m.EmbedVideoBytes(ctx, [][]byte{mediaBytes(t, "colors.mp4")}, "mp4", stamps, 0)
		if err != nil {
			t.Fatal(err)
		}
		requireBits(t, "video bytes "+name, got, f.expected(t, "video_bytes_"+name))
	}
	// The default extension is mp4.
	got, err = m.EmbedVideoBytes(ctx, [][]byte{mediaBytes(t, "colors.mp4")}, "", false, 0)
	if err != nil {
		t.Fatal(err)
	}
	requireBits(t, "video bytes, default extension", got, f.expected(t, "video_bytes_plain"))
}

func TestDeviceMixedItemsMatchRustBitForBit(t *testing.T) {
	f := loadFixtures(t)
	m := sharedModel(t)
	ctx := context.Background()
	imgs, clips, vids := f.images(t), f.clips(t), f.videos(t)
	texts := []string{"a 0", "0 b", "1: 2", "00"}
	items := [][]embedgemma2.Segment{
		{embedgemma2.Text(texts[0]), embedgemma2.ImageSegment(imgs[0]), embedgemma2.Text(texts[1])},
		{embedgemma2.Text(texts[2]), embedgemma2.AudioSegment(clips[1]), embedgemma2.ImageSegment(imgs[1])},
		{embedgemma2.ImageSegment(imgs[2]), embedgemma2.VideoSegment(vids[0], false), embedgemma2.Text(texts[3]), embedgemma2.AudioSegment(clips[0])},
		{embedgemma2.Text(texts[0]), embedgemma2.VideoSegment(vids[1], true), embedgemma2.Text(texts[1])},
	}
	// Every medium is named once, and the fixtures' Rust call named them the same
	// way, so the rows are bit-for-bit the same.
	for _, trunc := range []int{0, 40} {
		got, err := m.EmbedMixed(ctx, items, trunc)
		if err != nil {
			t.Fatalf("truncate %d: %v", trunc, err)
		}
		requireBits(t, "segments_raw_"+truncName(trunc), got, f.expected(t, "segments_raw_"+truncName(trunc)))
	}

	// Images by path, clips as bytes, a video raw: each modality its own source.
	mixed := [][]embedgemma2.Segment{
		{
			embedgemma2.Text("see"),
			embedgemma2.ImageFileSegment(mediaPath("rgb.png")),
			embedgemma2.Text("and hear"),
			embedgemma2.AudioBytesSegment(mediaBytes(t, "tone_44k_stereo.flac")),
		},
		{
			embedgemma2.ImageFileSegment(mediaPath("gray.png")),
			embedgemma2.Text("0"),
			embedgemma2.VideoSegment(vids[0], false),
			embedgemma2.AudioBytesSegment(mediaBytes(t, "pcm16_16k_mono.wav")),
		},
	}
	// The Go plan numbers the clips in order of mention; the Rust call's table
	// order differs, but each item reads the same media, so the rows agree.
	got, err := m.EmbedMixed(ctx, mixed, 0)
	if err != nil {
		t.Fatal(err)
	}
	requireBits(t, "segments_mixed_0", got, f.expected(t, "segments_mixed_0"))
}

func TestDeviceSoftTokenCountsMatchRust(t *testing.T) {
	f := loadFixtures(t)
	ctx := context.Background()
	var want struct {
		ImageSizes   [][2]int     `json:"image_sizes"`
		Image        []int        `json:"image"`
		AudioSamples []int        `json:"audio_samples"`
		Audio        []int        `json:"audio"`
		Videos       [][4]float64 `json:"videos"`
		Video        []int        `json:"video"`
	}
	if err := json.Unmarshal(f.read(t, "soft_tokens.json"), &want); err != nil {
		t.Fatal(err)
	}
	// They work on a text-only model too: they read processor_config.json, not a tower.
	m := loadOwn(t, embedgemma2.Options{TextOnly: true})
	sizes := make([]embedgemma2.ImageSize, len(want.ImageSizes))
	for i, s := range want.ImageSizes {
		sizes[i] = embedgemma2.ImageSize{Height: s[0], Width: s[1]}
	}
	img, err := m.ImageSoftTokens(ctx, sizes)
	if err != nil || fmt.Sprint(img) != fmt.Sprint(want.Image) {
		t.Fatalf("image soft tokens %v %v, want %v", img, err, want.Image)
	}
	aud, err := m.AudioSoftTokens(ctx, want.AudioSamples, 16000)
	if err != nil || fmt.Sprint(aud) != fmt.Sprint(want.Audio) {
		t.Fatalf("audio soft tokens %v %v, want %v", aud, err, want.Audio)
	}
	infos := make([]embedgemma2.VideoInfo, len(want.Videos))
	for i, v := range want.Videos {
		infos[i] = embedgemma2.VideoInfo{Height: int(v[0]), Width: int(v[1]), Frames: int(v[2]), FPS: v[3]}
	}
	vid, err := m.VideoSoftTokens(ctx, infos)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range vid {
		if v.Frames != want.Video[2*i] || v.PerFrame != want.Video[2*i+1] {
			t.Errorf("video %d soft tokens %+v, want %d x %d", i, v, want.Video[2*i], want.Video[2*i+1])
		}
	}
}

func TestDeviceBF16AndTextOnlyModels(t *testing.T) {
	f := loadFixtures(t)
	ctx := context.Background()
	bf := loadOwn(t, embedgemma2.Options{Operands: embedgemma2.BF16})
	got, err := bf.EmbedTexts(ctx, f.m.Texts, "query", 0)
	if err != nil {
		t.Fatal(err)
	}
	requireBits(t, "bf16 texts", got, f.expected(t, "bf16_texts_query_0"))
	got, err = bf.EmbedImages(ctx, f.images(t), 40)
	if err != nil {
		t.Fatal(err)
	}
	requireBits(t, "bf16 images", got, f.expected(t, "bf16_images_40"))
	got, err = bf.EmbedAudio(ctx, f.clips(t), 0)
	if err != nil {
		t.Fatal(err)
	}
	requireBits(t, "bf16 audio", got, f.expected(t, "bf16_audio_0"))
	got, err = bf.EmbedVideos(ctx, f.videos(t), true, 0)
	if err != nil {
		t.Fatal(err)
	}
	requireBits(t, "bf16 timestamped video", got, f.expected(t, "bf16_video_ts_0"))

	// A text-only model embeds texts as the multimodal one does, and refuses
	// media by naming the load that would allow it.
	text := loadOwn(t, embedgemma2.Options{TextOnly: true})
	if text.Multimodal() {
		t.Fatal("a TextOnly model reports towers")
	}
	got, err = text.EmbedTexts(ctx, f.m.Texts, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	requireBits(t, "text-only texts", got, f.expected(t, "texts_none_0"))
	_, err = text.EmbedImages(ctx, f.images(t)[:1], 0)
	var te *embedgemma2.Error
	if !errors.As(err, &te) || te.Status != embedgemma2.StatusError || !strings.Contains(te.Message, "load_multimodal") {
		t.Fatalf("an image on a text-only model: %v", err)
	}
	// The refusal leaves the model usable.
	if _, err := text.EmbedTexts(ctx, f.m.Texts[:2], "", 0); err != nil {
		t.Fatalf("the model is unusable after a refusal: %v", err)
	}
}

func TestDeviceTesslRefusalsAreTypedErrorsWithItsMessage(t *testing.T) {
	f := loadFixtures(t)
	m := sharedModel(t)
	ctx := context.Background()
	for name, tc := range map[string]struct {
		call func() error
		want string
	}{
		"unknown prompt": {func() error { _, err := m.EmbedTexts(ctx, f.m.Texts, "nope", 0); return err }, "no prompt named"},
		"rate that is not the extractor's": {func() error {
			a := f.clips(t)
			for i := range a {
				a[i].SampleRate = 8000
			}
			_, err := m.EmbedAudio(ctx, a, 0)
			return err
		}, "does not resample"},
		"missing file": {func() error {
			_, err := m.EmbedImageFiles(ctx, []string{mediaPath("no-such-image.png")}, 0)
			return err
		}, "file 0"},
		"garbage bytes": {func() error {
			_, err := m.EmbedImageBytes(ctx, [][]byte{[]byte("this is not an image")}, 0)
			return err
		}, "bytes 0"},
	} {
		err := tc.call()
		var te *embedgemma2.Error
		if !errors.As(err, &te) {
			t.Errorf("%s: got %T %v, want *embedgemma2.Error", name, err, err)
			continue
		}
		if te.Status != embedgemma2.StatusError || te.Poisoned || !strings.Contains(te.Message, tc.want) {
			t.Errorf("%s: %+v, want a StatusError mentioning %q", name, te, tc.want)
		}
		if errors.Is(err, embedgemma2.ErrPanic) || errors.Is(err, embedgemma2.ErrPoisoned) || errors.Is(err, embedgemma2.ErrInvalidArgument) {
			t.Errorf("%s: %v matches a sentinel it should not", name, err)
		}
	}
	// None of them poisoned or killed the model.
	got, err := m.EmbedTexts(ctx, f.m.Texts[:3], "", 0)
	if err != nil {
		t.Fatalf("the shared model is unusable after refusals: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("%d rows", len(got))
	}
}

func TestDeviceMaxTokensLimitsTextsAndIsReadBack(t *testing.T) {
	f := loadFixtures(t)
	m := loadOwn(t, embedgemma2.Options{})
	ctx := context.Background()
	if err := m.SetMaxTokens(ctx, 6); err != nil {
		t.Fatal(err)
	}
	if m.MaxTokens() != 6 {
		t.Fatalf("MaxTokens() = %d after SetMaxTokens(6)", m.MaxTokens())
	}
	long := f.m.Texts[len(f.m.Texts)-1]
	_, err := m.EmbedTexts(ctx, []string{long}, "", 0)
	var te *embedgemma2.Error
	if !errors.As(err, &te) || !strings.Contains(te.Message, "tokens") {
		t.Fatalf("a text past the limit: %v", err)
	}
	if err := m.SetMaxTokens(ctx, 8192); err != nil {
		t.Fatal(err)
	}
	if _, err := m.EmbedTexts(ctx, []string{long}, "", 0); err != nil {
		t.Fatalf("after raising the limit: %v", err)
	}
	for _, bad := range []int{0, -1, 32769} {
		if err := m.SetMaxTokens(ctx, bad); !errors.Is(err, embedgemma2.ErrInvalidArgument) {
			t.Errorf("SetMaxTokens(%d) = %v, want ErrInvalidArgument", bad, err)
		}
	}
}

func TestDeviceCloseIsIdempotentAndRefusesLaterCalls(t *testing.T) {
	f := loadFixtures(t)
	m, err := embedgemma2.Load(f.snapshot(), embedgemma2.Options{TextOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.EmbedTexts(context.Background(), f.m.Texts[:1], "", 0); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	_, err = m.EmbedTexts(context.Background(), f.m.Texts[:1], "", 0)
	if !errors.Is(err, embedgemma2.ErrClosed) {
		t.Fatalf("a call after Close: %v", err)
	}
	if _, err := m.Prompts(context.Background()); !errors.Is(err, embedgemma2.ErrClosed) {
		t.Fatalf("Prompts after Close: %v", err)
	}
}

func TestDeviceLoadFailuresAreTypedAndLeaveNothingBehind(t *testing.T) {
	f := loadFixtures(t)
	dir := t.TempDir()
	_, err := embedgemma2.Load(dir, embedgemma2.Options{})
	var te *embedgemma2.Error
	if !errors.As(err, &te) || te.Status != embedgemma2.StatusError || !strings.Contains(te.Message, "config.json") {
		t.Fatalf("loading an empty directory: %v", err)
	}
	// A failed load does not stop the next.
	m, err := embedgemma2.Load(f.snapshot(), embedgemma2.Options{TextOnly: true})
	if err != nil {
		t.Fatalf("a load after a failed one: %v", err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestDeviceObserverSeesWorkInsideTessl(t *testing.T) {
	f := loadFixtures(t)
	var seen []embedgemma2.CallInfo
	m := loadOwn(t, embedgemma2.Options{TextOnly: true, Observer: func(ci embedgemma2.CallInfo) { seen = append(seen, ci) }})
	if _, err := m.EmbedTexts(context.Background(), f.m.Texts, "", 0); err != nil {
		t.Fatal(err)
	}
	last := seen[len(seen)-1]
	if last.Op != "EmbedTexts" || last.Err != nil || last.Work <= 0 || last.RequestBytes <= 0 || last.ReplyBytes <= 16 {
		t.Fatalf("observed %+v", last)
	}
	if _, err := m.EmbedTexts(context.Background(), f.m.Texts, "nope", 0); err == nil {
		t.Fatal("an unknown prompt was accepted")
	}
	last = seen[len(seen)-1]
	if last.Err == nil || last.Work <= 0 {
		t.Fatalf("a refusal is observed with its error and the time tessl took: %+v", last)
	}
}
