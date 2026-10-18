//go:build darwin && arm64

package embedgemma2_test

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/bharathvbcr/tessl/go/embedgemma2"
	"github.com/bharathvbcr/tessl/go/embedgemma2/internal/npy"
)

// The real-checkpoint tests compare the Go path with sentence-transformers'
// own embeddings of google/embeddinggemma-2, as tessl's Rust and Python tests
// do. They are opt-in:
//
//	EMBEDGEMMA2_SNAPSHOT     the Hugging Face snapshot directory
//	EMBEDGEMMA2_REF_DIR      target/embedgemma2_ref (text references)
//	EMBEDGEMMA2_MM_REF_DIR   target/embedgemma2_mm_ref (image, audio, video)
//
// With EMBEDGEMMA2_SNAPSHOT unset they skip. With it set they have opted in,
// and a missing reference directory or file is a failure, never a skip: a check
// that could not run must not read as one that passed.
//
// Tolerances are the Rust tests': max abs error 1e-4 and cosine 0.99999 against
// the f32 reference.
const (
	embAbs = 1e-4
	embCos = 0.99999
	// The bf16 floors of tests/embedgemma2_mm_model.rs: cosine against the fp32
	// reference, and against the reference's own bf16 run (the triangle bound
	// over two forwards each within 1-cos <= 1e-4 of fp32).
	bf16CosF32  = 0.9999
	bf16CosBF16 = 0.9996
)

var realModel struct {
	once  sync.Once
	model *embedgemma2.Model
	err   error
}

func realSnapshot(t testing.TB) string {
	t.Helper()
	s := os.Getenv("EMBEDGEMMA2_SNAPSHOT")
	if s == "" {
		t.Skip("EMBEDGEMMA2_SNAPSHOT is not set: the real-checkpoint tests are opt-in")
	}
	return s
}

func refDir(t testing.TB, env string) string {
	t.Helper()
	realSnapshot(t) // skips unless the tests have opted in
	d := os.Getenv(env)
	if d == "" {
		t.Fatalf("EMBEDGEMMA2_SNAPSHOT is set, so the real-checkpoint tests have opted in, but %s is not", env)
	}
	if st, err := os.Stat(d); err != nil || !st.IsDir() {
		t.Fatalf("%s=%s is not a directory (%v)", env, d, err)
	}
	return d
}

// realF32 is the multimodal exact-f32 model on the real checkpoint, loaded once.
func realF32(t testing.TB) *embedgemma2.Model {
	t.Helper()
	snap := realSnapshot(t)
	realModel.once.Do(func() {
		realModel.model, realModel.err = embedgemma2.Load(snap, embedgemma2.Options{})
	})
	if realModel.err != nil {
		t.Fatalf("loading the real checkpoint: %v", realModel.err)
	}
	return realModel.model
}

func closeRealModel() error {
	if realModel.model == nil {
		return nil
	}
	return realModel.model.Close()
}

func readNpy(t testing.TB, dir, name string) *npy.Array {
	t.Helper()
	a, err := npy.Read(filepath.Join(dir, name))
	if err != nil {
		t.Fatalf("the reference %s is missing or unreadable: %v", name, err)
	}
	return a
}

func floats(t testing.TB, dir, name string) []float32 {
	t.Helper()
	v, err := readNpy(t, dir, name).Float32s()
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return v
}

func maxAbsDiff(a, b []float32) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > m {
			m = d
		}
	}
	return m
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / math.Sqrt(na*nb)
}

// parity compares each row of got with the reference and quotes the worst
// numbers; it fails past the tolerance.
func parity(t *testing.T, what string, got [][]float32, want []float32, width int) {
	t.Helper()
	if len(want) != len(got)*width {
		t.Fatalf("%s: the reference holds %d values, %d rows of %d need %d", what, len(want), len(got), width, len(got)*width)
	}
	worstAbs, worstCos := 0.0, 1.0
	for i, row := range got {
		if len(row) != width {
			t.Fatalf("%s: row %d has %d values, want %d", what, i, len(row), width)
		}
		ref := want[i*width : (i+1)*width]
		var norm float64
		for _, v := range row {
			norm += float64(v) * float64(v)
		}
		if math.Abs(norm-1) > 1e-4 {
			t.Errorf("%s: row %d has squared norm %v", what, i, norm)
		}
		if e := maxAbsDiff(row, ref); e > worstAbs {
			worstAbs = e
		}
		if c := cosine(row, ref); c < worstCos {
			worstCos = c
		}
	}
	t.Logf("parity %-34s rows %d: max abs %.3e, min cosine %.9f", what, len(got), worstAbs, worstCos)
	if worstAbs > embAbs {
		t.Errorf("%s: max abs error %.3e exceeds %.0e", what, worstAbs, embAbs)
	}
	if worstCos < embCos {
		t.Errorf("%s: cosine %.9f is below %v", what, worstCos, embCos)
	}
}

type textCase struct {
	PromptName string `json:"prompt_name"`
	Text       string `json:"text"`
}

func loadTexts(t testing.TB, dir string) []textCase {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, "texts.json"))
	if err != nil {
		t.Fatalf("texts.json is missing: %v", err)
	}
	var file struct {
		Texts []textCase `json:"texts"`
	}
	if err := json.Unmarshal(raw, &file); err != nil || len(file.Texts) == 0 {
		t.Fatalf("texts.json: %v (%d texts)", err, len(file.Texts))
	}
	return file.Texts
}

// embedByPrompt embeds the texts a prompt name at a time and returns the rows in
// the texts' own order.
func embedByPrompt(t testing.TB, m *embedgemma2.Model, cases []textCase, trunc int) [][]float32 {
	t.Helper()
	groups := map[string][]int{}
	var order []string
	for i, c := range cases {
		if _, seen := groups[c.PromptName]; !seen {
			order = append(order, c.PromptName)
		}
		groups[c.PromptName] = append(groups[c.PromptName], i)
	}
	rows := make([][]float32, len(cases))
	for _, name := range order {
		idx := groups[name]
		texts := make([]string, len(idx))
		for j, i := range idx {
			texts[j] = cases[i].Text
		}
		got, err := m.EmbedTexts(context.Background(), texts, name, trunc)
		if err != nil {
			t.Fatalf("prompt %q: %v", name, err)
		}
		for j, i := range idx {
			rows[i] = got[j]
		}
	}
	return rows
}

func TestRealTextsMatchSentenceTransformers(t *testing.T) {
	dir := refDir(t, "EMBEDGEMMA2_REF_DIR")
	m := realF32(t)
	cases := loadTexts(t, dir)
	if m.EmbeddingDim() != 768 {
		t.Fatalf("embedding dim %d", m.EmbeddingDim())
	}
	want := floats(t, dir, "emb_f32.npy")
	parity(t, fmt.Sprintf("texts (%d, by prompt_name)", len(cases)), embedByPrompt(t, m, cases, 0), want, m.EmbeddingDim())
	for _, d := range []int{512, 256, 128} {
		ref := floats(t, dir, fmt.Sprintf("emb_f32_trunc%d.npy", d))
		parity(t, fmt.Sprintf("texts, Matryoshka %d", d), embedByPrompt(t, m, cases, d), ref, d)
	}
}

func TestRealTokenizerIdsMatchTheReference(t *testing.T) {
	dir := refDir(t, "EMBEDGEMMA2_REF_DIR")
	m := realF32(t)
	cases := loadTexts(t, dir)
	flat, err := readNpy(t, dir, "ids.npy").Int64s()
	if err != nil {
		t.Fatal(err)
	}
	cuts, err := readNpy(t, dir, "offsets.npy").Int64s()
	if err != nil {
		t.Fatal(err)
	}
	if len(cuts) != len(cases)+1 {
		t.Fatalf("%d offsets for %d texts", len(cuts), len(cases))
	}
	ctx := context.Background()
	for i, c := range cases {
		got, err := m.Tokenize(ctx, []string{c.Text}, c.PromptName)
		if err != nil {
			t.Fatalf("text %d: %v", i, err)
		}
		want := flat[cuts[i]:cuts[i+1]]
		if len(got[0]) != len(want) {
			t.Fatalf("text %d: %d ids, the reference has %d", i, len(got[0]), len(want))
		}
		for j, id := range got[0] {
			if int64(id) != want[j] {
				t.Fatalf("text %d id %d is %d, the reference has %d", i, j, id, want[j])
			}
		}
	}
	t.Logf("token ids equal the reference's for all %d texts (%d ids)", len(cases), len(flat))
}

type mmInput struct {
	Kind        string  `json:"kind"`
	Index       int     `json:"index"`
	Prefix      string  `json:"prefix"`
	Width       int     `json:"width"`
	Height      int     `json:"height"`
	SoftTokens  int     `json:"soft_tokens"`
	SoftPer     int     `json:"soft_per_frame"`
	Sampled     int     `json:"frames_sampled"`
	Timestamps  bool    `json:"add_timestamps"`
	SampleRate  int     `json:"sampling_rate"`
	FPS         float64 `json:"fps"`
	FramesTotal int     `json:"frames"`
}

func loadMM(t testing.TB, dir string) []mmInput {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, "inputs.json"))
	if err != nil {
		t.Fatalf("inputs.json is missing: %v", err)
	}
	var file struct {
		Inputs []mmInput `json:"inputs"`
	}
	if err := json.Unmarshal(raw, &file); err != nil || len(file.Inputs) == 0 {
		t.Fatalf("inputs.json: %v (%d inputs)", err, len(file.Inputs))
	}
	return file.Inputs
}

func (in mmInput) image(t testing.TB, dir string) embedgemma2.Image {
	a := readNpy(t, dir, in.Prefix+"raw.npy")
	pix, err := a.Uint8s()
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Shape) != 3 {
		t.Fatalf("%sraw: shape %v", in.Prefix, a.Shape)
	}
	return embedgemma2.Image{Pix: pix, Height: a.Shape[0], Width: a.Shape[1], Channels: a.Shape[2]}
}

func (in mmInput) audio(t testing.TB, dir string) embedgemma2.Audio {
	return embedgemma2.Audio{Samples: floats(t, dir, in.Prefix+"raw.npy"), SampleRate: in.SampleRate}
}

func (in mmInput) video(t testing.TB, dir string) embedgemma2.Video {
	a := readNpy(t, dir, in.Prefix+"raw.npy")
	pix, err := a.Uint8s()
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Shape) != 4 || a.Shape[3] != 3 {
		t.Fatalf("%sraw: shape %v", in.Prefix, a.Shape)
	}
	fps, err := readNpy(t, dir, in.Prefix+"fps.npy").Float64()
	if err != nil {
		t.Fatal(err)
	}
	return embedgemma2.Video{Frames: pix, NumFrames: a.Shape[0], Height: a.Shape[1], Width: a.Shape[2], FPS: fps}
}

func TestRealImagesMatchSentenceTransformers(t *testing.T) {
	dir := refDir(t, "EMBEDGEMMA2_MM_REF_DIR")
	m := realF32(t)
	var images []embedgemma2.Image
	var want []float32
	for _, in := range loadMM(t, dir) {
		if in.Kind == "image" {
			images = append(images, in.image(t, dir))
			want = append(want, floats(t, dir, in.Prefix+"emb_f32.npy")...)
		}
	}
	if len(images) != 2 {
		t.Fatalf("%d image references, expected 2", len(images))
	}
	got, err := m.EmbedImages(context.Background(), images, 0)
	if err != nil {
		t.Fatal(err)
	}
	parity(t, "images (2, one call)", got, want, m.EmbeddingDim())
}

func TestRealAudioMatchesSentenceTransformers(t *testing.T) {
	dir := refDir(t, "EMBEDGEMMA2_MM_REF_DIR")
	m := realF32(t)
	var clips []embedgemma2.Audio
	var want []float32
	for _, in := range loadMM(t, dir) {
		if in.Kind == "audio" {
			clips = append(clips, in.audio(t, dir))
			want = append(want, floats(t, dir, in.Prefix+"emb_f32.npy")...)
		}
	}
	if len(clips) != 2 {
		t.Fatalf("%d audio references, expected 2", len(clips))
	}
	got, err := m.EmbedAudio(context.Background(), clips, 0)
	if err != nil {
		t.Fatal(err)
	}
	parity(t, "audio (2, one call)", got, want, m.EmbeddingDim())
}

func TestRealVideoAndTimestampedVideoMatchSentenceTransformers(t *testing.T) {
	dir := refDir(t, "EMBEDGEMMA2_MM_REF_DIR")
	m := realF32(t)
	n := 0
	for _, in := range loadMM(t, dir) {
		if in.Kind != "video" {
			continue
		}
		n++
		want := floats(t, dir, in.Prefix+"emb_f32.npy")
		got, err := m.EmbedVideo(context.Background(), in.video(t, dir), in.Timestamps, 0)
		if err != nil {
			t.Fatalf("%s: %v", in.Prefix, err)
		}
		name := fmt.Sprintf("video %s (timestamps=%v)", in.Prefix, in.Timestamps)
		parity(t, name, [][]float32{got}, want, m.EmbeddingDim())
	}
	if n != 3 {
		t.Fatalf("%d video references, expected video0, video0t and video1", n)
	}
}

func TestRealSoftTokenCountsMatchTheProcessor(t *testing.T) {
	dir := refDir(t, "EMBEDGEMMA2_MM_REF_DIR")
	m := realF32(t)
	ctx := context.Background()
	for _, in := range loadMM(t, dir) {
		switch in.Kind {
		case "image":
			got, err := m.ImageSoftTokens(ctx, []embedgemma2.ImageSize{{Height: in.Height, Width: in.Width}})
			if err != nil || got[0] != in.SoftTokens {
				t.Errorf("%s: %v soft tokens (%v), the processor made %d", in.Prefix, got, err, in.SoftTokens)
			}
		case "audio":
			a := in.audio(t, dir)
			got, err := m.AudioSoftTokens(ctx, []int{len(a.Samples)}, a.SampleRate)
			if err != nil || got[0] != in.SoftTokens {
				t.Errorf("%s: %v soft tokens (%v), the processor made %d", in.Prefix, got, err, in.SoftTokens)
			}
		case "video":
			v := in.video(t, dir)
			got, err := m.VideoSoftTokens(ctx, []embedgemma2.VideoInfo{{Height: v.Height, Width: v.Width, Frames: v.NumFrames, FPS: v.FPS}})
			if err != nil || got[0].Frames != in.Sampled || got[0].PerFrame != in.SoftPer || got[0].Frames*got[0].PerFrame != in.SoftTokens {
				t.Errorf("%s: %+v (%v), the processor kept %d frames of %d (%d soft tokens)", in.Prefix, got, err, in.Sampled, in.SoftPer, in.SoftTokens)
			}
		}
	}
}

func TestRealBF16MatchesTheReferenceWithinItsFloors(t *testing.T) {
	snap := realSnapshot(t)
	textDir := refDir(t, "EMBEDGEMMA2_REF_DIR")
	mmDir := refDir(t, "EMBEDGEMMA2_MM_REF_DIR")
	m, err := embedgemma2.Load(snap, embedgemma2.Options{Operands: embedgemma2.BF16})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ctx := context.Background()
	dim := m.EmbeddingDim()

	check := func(what string, got [][]float32, f32, bf16 []float32) {
		t.Helper()
		var worst32, worst16 = 1.0, 1.0
		for i, row := range got {
			a, b := f32[i*dim:(i+1)*dim], bf16[i*dim:(i+1)*dim]
			if c := cosine(row, a); c < worst32 {
				worst32 = c
			}
			if c := cosine(row, b); c < worst16 {
				worst16 = c
			}
		}
		t.Logf("bf16 %-26s rows %d: min cosine vs f32 %.6f, vs the reference's bf16 %.6f", what, len(got), worst32, worst16)
		if worst32 < bf16CosF32 {
			t.Errorf("%s: cosine vs f32 %.6f below %v", what, worst32, bf16CosF32)
		}
		if worst16 < bf16CosBF16 {
			t.Errorf("%s: cosine vs the reference's bf16 %.6f below %v", what, worst16, bf16CosBF16)
		}
	}

	cases := loadTexts(t, textDir)
	check("texts", embedByPrompt(t, m, cases, 0), floats(t, textDir, "emb_f32.npy"), floats(t, textDir, "emb_bf16.npy"))

	var images []embedgemma2.Image
	var f32, bf16 []float32
	for _, in := range loadMM(t, mmDir) {
		if in.Kind == "image" {
			images = append(images, in.image(t, mmDir))
			f32 = append(f32, floats(t, mmDir, in.Prefix+"emb_f32.npy")...)
			bf16 = append(bf16, floats(t, mmDir, in.Prefix+"emb_bf16.npy")...)
		}
	}
	got, err := m.EmbedImages(ctx, images, 0)
	if err != nil {
		t.Fatal(err)
	}
	check("images", got, f32, bf16)

	var clips []embedgemma2.Audio
	f32, bf16 = nil, nil
	for _, in := range loadMM(t, mmDir) {
		if in.Kind == "audio" {
			clips = append(clips, in.audio(t, mmDir))
			f32 = append(f32, floats(t, mmDir, in.Prefix+"emb_f32.npy")...)
			bf16 = append(bf16, floats(t, mmDir, in.Prefix+"emb_bf16.npy")...)
		}
	}
	got, err = m.EmbedAudio(ctx, clips, 0)
	if err != nil {
		t.Fatal(err)
	}
	check("audio", got, f32, bf16)

	for _, in := range loadMM(t, mmDir) {
		if in.Kind != "video" {
			continue
		}
		row, err := m.EmbedVideo(ctx, in.video(t, mmDir), in.Timestamps, 0)
		if err != nil {
			t.Fatal(err)
		}
		check("video "+in.Prefix, [][]float32{row}, floats(t, mmDir, in.Prefix+"emb_f32.npy"), floats(t, mmDir, in.Prefix+"emb_bf16.npy"))
	}
}

func TestRealAFailedCallLeavesTheRealModelServing(t *testing.T) {
	m := realF32(t)
	ctx := context.Background()
	if _, err := m.EmbedTexts(ctx, []string{"a"}, "no-such-prompt", 0); err == nil {
		t.Fatal("an unknown prompt was accepted")
	}
	if _, err := m.EmbedImageFiles(ctx, []string{"/no/such/file.png"}, 0); err == nil {
		t.Fatal("a missing file was accepted")
	}
	rows, err := m.EmbedTexts(ctx, []string{"still serving"}, "query", 0)
	if err != nil || len(rows) != 1 || len(rows[0]) != m.EmbeddingDim() {
		t.Fatalf("after refusals: %v", err)
	}
}
