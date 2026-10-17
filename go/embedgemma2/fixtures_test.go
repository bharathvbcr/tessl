//go:build darwin && arm64

package embedgemma2_test

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/bharathvbcr/tessl/go/embedgemma2"
)

// The device tests read what tests/go_fixtures.rs wrote: the tiny multimodal
// checkpoint as a snapshot directory, the raw media, and what tessl's C ABI
// answered for them when called from Rust. `make -C go test` generates the
// directory first and sets TESSL_GO_REQUIRE_DEVICE, which turns a missing
// directory from a skip into a failure: a check that could not run must never
// read as one that passed.

// fixtureDir is where tests/go_fixtures.rs writes (TESSL_GO_FIXTURES overrides).
func fixtureDir() string {
	if d := os.Getenv("TESSL_GO_FIXTURES"); d != "" {
		return d
	}
	return filepath.Join("..", "..", "target", "go_fixtures")
}

type expectedRef struct {
	File string `json:"file"`
	Rows int    `json:"rows"`
	Dim  int    `json:"dim"`
}

type manifest struct {
	Dim    int      `json:"dim"`
	Texts  []string `json:"texts"`
	Images []struct {
		File    string `json:"file"`
		H, W, C int
	} `json:"images"`
	Audio []struct {
		File    string `json:"file"`
		Samples int    `json:"samples"`
	} `json:"audio"`
	Videos []struct {
		File   string `json:"file"`
		Frames int    `json:"frames"`
		H, W   int
		FPS    float64 `json:"fps"`
	} `json:"videos"`
	Expected map[string]expectedRef `json:"expected"`
}

type fixtures struct {
	dir string
	m   manifest
}

var (
	fixtureOnce sync.Once
	fixtureSet  *fixtures
	fixtureErr  error
)

// loadFixtures reads the manifest once. A missing directory skips unless
// TESSL_GO_REQUIRE_DEVICE is set, when it fails.
func loadFixtures(t testing.TB) *fixtures {
	t.Helper()
	fixtureOnce.Do(func() {
		dir := fixtureDir()
		raw, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
		if err != nil {
			fixtureErr = err
			return
		}
		f := &fixtures{dir: dir}
		if err := json.Unmarshal(raw, &f.m); err != nil {
			fixtureErr = err
			return
		}
		fixtureSet = f
	})
	if fixtureErr != nil {
		if os.Getenv("TESSL_GO_REQUIRE_DEVICE") != "" {
			t.Fatalf("the device fixtures are missing (%v): run `cargo test --release --test go_fixtures` or `make -C go test`", fixtureErr)
		}
		t.Skipf("no device fixtures (%v): `make -C go test` generates them", fixtureErr)
	}
	return fixtureSet
}

func (f *fixtures) snapshot() string { return filepath.Join(f.dir, "snapshot") }

func (f *fixtures) read(t testing.TB, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(f.dir, name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func (f *fixtures) f32s(t testing.TB, name string) []float32 {
	t.Helper()
	b := f.read(t, name)
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

// expected returns the rows tessl's C ABI answered for the case key.
func (f *fixtures) expected(t testing.TB, key string) [][]float32 {
	t.Helper()
	ref, ok := f.m.Expected[key]
	if !ok {
		t.Fatalf("the fixtures hold no expected rows %q", key)
	}
	flat := f.f32s(t, ref.File)
	if len(flat) != ref.Rows*ref.Dim {
		t.Fatalf("%s: %d values for %d rows of %d", key, len(flat), ref.Rows, ref.Dim)
	}
	rows := make([][]float32, ref.Rows)
	for i := range rows {
		rows[i] = flat[i*ref.Dim : (i+1)*ref.Dim]
	}
	return rows
}

func (f *fixtures) images(t testing.TB) []embedgemma2.Image {
	t.Helper()
	out := make([]embedgemma2.Image, len(f.m.Images))
	for i, im := range f.m.Images {
		out[i] = embedgemma2.Image{Pix: f.read(t, im.File), Height: im.H, Width: im.W, Channels: im.C}
	}
	return out
}

func (f *fixtures) clips(t testing.TB) []embedgemma2.Audio {
	t.Helper()
	out := make([]embedgemma2.Audio, len(f.m.Audio))
	for i, a := range f.m.Audio {
		out[i] = embedgemma2.Audio{Samples: f.f32s(t, a.File), SampleRate: 16000}
	}
	return out
}

func (f *fixtures) videos(t testing.TB) []embedgemma2.Video {
	t.Helper()
	out := make([]embedgemma2.Video, len(f.m.Videos))
	for i, v := range f.m.Videos {
		out[i] = embedgemma2.Video{Frames: f.read(t, v.File), NumFrames: v.Frames, Height: v.H, Width: v.W, FPS: v.FPS}
	}
	return out
}

// media returns the path of a media fixture of the tessl checkout.
func mediaPath(name string) string {
	p, err := filepath.Abs(filepath.Join("..", "..", "tests", "fixtures", "media", name))
	if err != nil {
		panic(err)
	}
	return p
}

func mediaBytes(t testing.TB, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(mediaPath(name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// requireBits fails unless got is want bit for bit.
func requireBits(t testing.TB, what string, got, want [][]float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d rows, want %d", what, len(got), len(want))
	}
	for i := range got {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("%s: row %d has %d values, want %d", what, i, len(got[i]), len(want[i]))
		}
		for j := range got[i] {
			if math.Float32bits(got[i][j]) != math.Float32bits(want[i][j]) {
				t.Fatalf("%s: row %d column %d is %v (%#x), want %v (%#x)", what, i, j,
					got[i][j], math.Float32bits(got[i][j]), want[i][j], math.Float32bits(want[i][j]))
			}
		}
	}
}

// sharedModel is the multimodal exact-f32 tiny model the read-only device tests
// share, loaded once.
var shared struct {
	once  sync.Once
	model *embedgemma2.Model
	err   error
}

func sharedModel(t testing.TB) *embedgemma2.Model {
	t.Helper()
	f := loadFixtures(t)
	shared.once.Do(func() {
		shared.model, shared.err = embedgemma2.Load(f.snapshot(), embedgemma2.Options{})
	})
	if shared.err != nil {
		t.Fatalf("loading the tiny checkpoint: %v", shared.err)
	}
	return shared.model
}

// loadOwn loads a model for a test that closes it, changes it or needs other
// options.
func loadOwn(t testing.TB, opts embedgemma2.Options) *embedgemma2.Model {
	t.Helper()
	f := loadFixtures(t)
	m, err := embedgemma2.Load(f.snapshot(), opts)
	if err != nil {
		t.Fatalf("loading the tiny checkpoint: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m
}
