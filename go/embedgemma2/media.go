//go:build darwin && arm64

package embedgemma2

import (
	"context"
	"math"
	"strconv"

	"github.com/bharathvbcr/tessl/go/embedgemma2/internal/wire"
)

// Limits on media that the package checks before a call crosses into Rust.
const (
	// MaxImagePixels is the most pixels one raw image may hold.
	MaxImagePixels = 1 << 28
	// MaxVideoFrames is the most decoded frames one call's raw videos hold.
	MaxVideoFrames = 1 << 20
	// MaxImageFileBytes, MaxAudioFileBytes and MaxVideoFileBytes bound one
	// encoded medium passed by value. A request as a whole is bounded by
	// MaxRequestBytes, which is below the audio and video figures.
	MaxImageFileBytes = 1 << 30
	MaxAudioFileBytes = 1 << 32
	MaxVideoFileBytes = 1 << 32
)

// source codes of the C ABI (TESSL_EMBEDGEMMA2_SOURCE_*).
const (
	sourceRaw   uint32 = 0
	sourcePaths uint32 = 1
	sourceBytes uint32 = 2
)

// Image is a decoded picture: Height*Width*Channels bytes, row-major HWC.
// Channels is 1 (gray), 2 (gray and alpha), 3 (RGB) or 4 (RGBA); alpha is
// dropped as PIL's convert("RGB") does.
type Image struct {
	Pix                     []byte
	Height, Width, Channels int
}

// Audio is a mono clip of f32 PCM in [-1, 1]. tessl takes the extractor's rate
// (16000 for the checkpoint) and refuses any other: it never resamples raw
// samples. (The file and bytes entries decode and resample to 16 kHz.) A clip
// is cut to 30 s.
type Audio struct {
	Samples    []float32
	SampleRate int
}

// Video is a decoded clip: NumFrames RGB frames of Height*Width*3 bytes each,
// laid end to end, row-major HWC, at the source rate FPS. tessl samples frames
// at the snapshot's own rate (its fps and max_frames).
type Video struct {
	Frames                   []byte
	NumFrames, Height, Width int
	FPS                      float64
}

func fitsU32(n int) bool { return n >= 0 && uint64(n) <= math.MaxUint32 }

// checkImage validates one raw image and returns its pixel count.
func checkImage(op, field string, im Image) error {
	switch {
	case im.Channels < 1 || im.Channels > 4:
		return argErr(op, field, "has %d channels, expected 1 (gray), 2, 3 (RGB) or 4 (RGBA)", im.Channels)
	case im.Height < 1 || im.Width < 1 || !fitsU32(im.Height) || !fitsU32(im.Width):
		return argErr(op, field, "is %dx%d, expected both dimensions in 1..=%d", im.Height, im.Width, uint64(math.MaxUint32))
	case uint64(im.Height)*uint64(im.Width) > MaxImagePixels:
		return argErr(op, field, "is %dx%d, above the %d-pixel limit", im.Height, im.Width, MaxImagePixels)
	}
	want := uint64(im.Height) * uint64(im.Width) * uint64(im.Channels)
	if uint64(len(im.Pix)) != want {
		return argErr(op, field, "holds %d bytes, but %dx%dx%d needs %d", len(im.Pix), im.Height, im.Width, im.Channels, want)
	}
	return nil
}

func checkAudio(op, field string, a Audio) error {
	switch {
	case len(a.Samples) < 1:
		return argErr(op, field, "has no samples")
	case len(a.Samples) > MaxClipSamples:
		return argErr(op, field, "has %d samples, the limit is %d", len(a.Samples), MaxClipSamples)
	case a.SampleRate < 1 || !fitsU32(a.SampleRate):
		return argErr(op, field, "sample rate %d is not a positive 32-bit integer", a.SampleRate)
	}
	return nil
}

func checkVideo(op, field string, v Video) error {
	switch {
	case v.NumFrames < 1 || v.NumFrames > MaxVideoFrames:
		return argErr(op, field, "has %d frames, expected 1..=%d", v.NumFrames, MaxVideoFrames)
	case v.Height < 1 || v.Width < 1 || !fitsU32(v.Height) || !fitsU32(v.Width):
		return argErr(op, field, "frames are %dx%d, expected both dimensions in 1..=%d", v.Height, v.Width, uint64(math.MaxUint32))
	case uint64(v.Height)*uint64(v.Width) > MaxImagePixels:
		return argErr(op, field, "frames are %dx%d, above the %d-pixel limit", v.Height, v.Width, MaxImagePixels)
	case math.IsNaN(v.FPS) || math.IsInf(v.FPS, 0) || v.FPS <= 0:
		return argErr(op, field, "fps %v is not finite and positive", v.FPS)
	}
	frame := uint64(v.Height) * uint64(v.Width) * 3
	if want := frame * uint64(v.NumFrames); uint64(len(v.Frames)) != want {
		return argErr(op, field, "holds %d bytes, but %d frames of %dx%dx3 need %d", len(v.Frames), v.NumFrames, v.Height, v.Width, want)
	}
	return nil
}

// EmbedImages embeds raw images into len(images) rows, each the embedding of
// the processor's sequence for that image: tessl resizes, rescales and
// patchifies it as the snapshot's processor_config.json says, then runs the
// vision tower. Needs a multimodal model. Up to MaxBatch images.
func (m *Model) EmbedImages(ctx context.Context, images []Image, truncateDim int) ([][]float32, error) {
	const op = "EmbedImages"
	if err := checkCount(op, "images", len(images)); err != nil {
		return nil, err
	}
	if err := m.checkTruncate(op, truncateDim); err != nil {
		return nil, err
	}
	pix := make([][]byte, len(images))
	dims := make([]uint32, 0, 3*len(images))
	for i, im := range images {
		if err := checkImage(op, indexed("images", i), im); err != nil {
			return nil, err
		}
		pix[i] = im.Pix
		dims = append(dims, uint32(im.Height), uint32(im.Width), uint32(im.Channels))
	}
	f := new(wire.Frame).U32(uint32(truncateDim)).Bytes(pix...).U32s(dims)
	return m.embed(ctx, op, wire.OpEmbedImages, f, len(images), truncateDim)
}

// EmbedAudio embeds raw mono clips into len(clips) rows through the audio
// tower. Every clip must carry the same SampleRate, and tessl refuses any but
// the extractor's. Needs a multimodal model. Up to MaxBatch clips.
func (m *Model) EmbedAudio(ctx context.Context, clips []Audio, truncateDim int) ([][]float32, error) {
	const op = "EmbedAudio"
	if err := checkCount(op, "clips", len(clips)); err != nil {
		return nil, err
	}
	if err := m.checkTruncate(op, truncateDim); err != nil {
		return nil, err
	}
	pcm := make([][]float32, len(clips))
	offsets := make([]uint64, 1, len(clips)+1)
	var total uint64
	for i, c := range clips {
		if err := checkAudio(op, indexed("clips", i), c); err != nil {
			return nil, err
		}
		if c.SampleRate != clips[0].SampleRate {
			return nil, argErr(op, indexed("clips", i), "sample rate %d differs from clip 0's %d: a call takes one rate", c.SampleRate, clips[0].SampleRate)
		}
		pcm[i] = c.Samples
		total += uint64(len(c.Samples))
		offsets = append(offsets, total)
	}
	f := new(wire.Frame).U32(uint32(truncateDim)).U32(uint32(clips[0].SampleRate)).F32s(pcm...).U64s(offsets)
	return m.embed(ctx, op, wire.OpEmbedAudio, f, len(clips), truncateDim)
}

// EmbedVideo embeds one raw video: see EmbedVideos.
func (m *Model) EmbedVideo(ctx context.Context, video Video, timestamps bool, truncateDim int) ([]float32, error) {
	rows, err := m.embedVideos(ctx, "EmbedVideo", []Video{video}, timestamps, truncateDim)
	if err != nil {
		return nil, err
	}
	return rows[0], nil
}

// EmbedVideos embeds raw videos into len(videos) rows. tessl samples each
// video's frames at the snapshot's rate, resizes them, runs every sampled frame
// through the vision tower and lays the sequence out as the processor does.
// With timestamps each frame's block is preceded by its mm:ss time (the
// processor's add_timestamps = true), which needs tokenizer.json. Needs a
// multimodal model. Up to MaxBatch videos, MaxVideoFrames frames in all.
func (m *Model) EmbedVideos(ctx context.Context, videos []Video, timestamps bool, truncateDim int) ([][]float32, error) {
	return m.embedVideos(ctx, "EmbedVideos", videos, timestamps, truncateDim)
}

func (m *Model) embedVideos(ctx context.Context, op string, videos []Video, timestamps bool, truncateDim int) ([][]float32, error) {
	if err := checkCount(op, "videos", len(videos)); err != nil {
		return nil, err
	}
	if err := m.checkTruncate(op, truncateDim); err != nil {
		return nil, err
	}
	frames := make([][]byte, len(videos))
	dims := make([]uint32, 0, 2*len(videos))
	cuts := make([]uint64, 1, len(videos)+1)
	fps := make([]float64, len(videos))
	var total uint64
	for i, v := range videos {
		if err := checkVideo(op, indexed("videos", i), v); err != nil {
			return nil, err
		}
		total += uint64(v.NumFrames)
		if total > MaxVideoFrames {
			return nil, argErr(op, "videos", "hold %d frames by video %d; a call takes at most %d", total, i, MaxVideoFrames)
		}
		frames[i] = v.Frames
		dims = append(dims, uint32(v.Height), uint32(v.Width))
		cuts = append(cuts, total)
		fps[i] = v.FPS
	}
	f := new(wire.Frame).U32(uint32(truncateDim)).U32(flag(timestamps)).Bytes(frames...).U32s(dims).U64s(cuts).F64s(fps)
	return m.embed(ctx, op, wire.OpEmbedVideos, f, len(videos), truncateDim)
}

func flag(b bool) uint32 {
	if b {
		return 1
	}
	return 0
}

// encoded is one modality's files or encoded bytes, ready for the frame.
type encoded struct {
	source  uint32
	paths   []string
	data    [][]byte
	offsets []uint64
}

func (e encoded) blob(f *wire.Frame) {
	if e.source == sourcePaths {
		f.Strings(e.paths...)
	} else {
		f.Bytes(e.data...)
	}
}

// pathTable validates paths and cuts them end to end.
func pathTable(op, field string, paths []string) (encoded, error) {
	e := encoded{source: sourcePaths, paths: paths, offsets: make([]uint64, 1, len(paths)+1)}
	var total uint64
	for i, p := range paths {
		switch {
		case len(p) == 0:
			return e, argErr(op, indexed(field, i), "is an empty path")
		case len(p) > MaxPathBytes:
			return e, argErr(op, indexed(field, i), "is %d bytes, the limit is %d", len(p), MaxPathBytes)
		case !wire.ValidUTF8(p):
			return e, argErr(op, indexed(field, i), "is not UTF-8 or contains a NUL byte")
		}
		total += uint64(len(p))
		e.offsets = append(e.offsets, total)
	}
	return e, nil
}

// bytesTable validates encoded media and cuts them end to end. max is the
// tessl limit on one medium of this kind.
func bytesTable(op, field string, data [][]byte, max uint64) (encoded, error) {
	e := encoded{source: sourceBytes, data: data, offsets: make([]uint64, 1, len(data)+1)}
	var total uint64
	for i, d := range data {
		switch {
		case len(d) == 0:
			return e, argErr(op, indexed(field, i), "is empty")
		case uint64(len(d)) > max:
			return e, argErr(op, indexed(field, i), "is %d bytes, the limit for one is %d", len(d), max)
		}
		total += uint64(len(d))
		if total > MaxRequestBytes {
			return e, argErr(op, field, "hold %d bytes by item %d, past the %d-byte request limit: pass files by path", total, i, MaxRequestBytes)
		}
		e.offsets = append(e.offsets, total)
	}
	return e, nil
}

// EmbedImageFiles embeds image files into len(paths) rows: ImageIO decodes
// them (PNG, JPEG, GIF, HEIC, TIFF, BMP, ... the first image of the file; RGB
// samples as stored, EXIF orientation not applied, as torchvision reads a
// path), then the path is EmbedImages'. A file that cannot be decoded is an
// *Error naming its index and the decoder's reason. Needs a multimodal model.
func (m *Model) EmbedImageFiles(ctx context.Context, paths []string, truncateDim int) ([][]float32, error) {
	const op = "EmbedImageFiles"
	return m.embedSources(ctx, op, wire.OpEmbedImageSources, "paths", paths, nil, false, "", false, truncateDim)
}

// EmbedImageBytes is EmbedImageFiles for whole encoded files held in memory.
func (m *Model) EmbedImageBytes(ctx context.Context, data [][]byte, truncateDim int) ([][]float32, error) {
	const op = "EmbedImageBytes"
	return m.embedSources(ctx, op, wire.OpEmbedImageSources, "data", nil, data, true, "", false, truncateDim)
}

// EmbedAudioFiles embeds audio files into len(paths) rows: AudioToolbox
// decodes them (WAV, AIFF, CAF, MP3, M4A/AAC, FLAC, ...) at their own rate, and
// tessl mixes to mono and resamples to the extractor's 16 kHz as torchcodec
// does (libswresample's channel matrix and resampler), then the path
// is EmbedAudio's. A clip longer than 30 minutes is refused by the decoder.
func (m *Model) EmbedAudioFiles(ctx context.Context, paths []string, truncateDim int) ([][]float32, error) {
	const op = "EmbedAudioFiles"
	return m.embedSources(ctx, op, wire.OpEmbedAudioSources, "paths", paths, nil, false, "", false, truncateDim)
}

// EmbedAudioBytes is EmbedAudioFiles for whole encoded files held in memory.
func (m *Model) EmbedAudioBytes(ctx context.Context, data [][]byte, truncateDim int) ([][]float32, error) {
	const op = "EmbedAudioBytes"
	return m.embedSources(ctx, op, wire.OpEmbedAudioSources, "data", nil, data, true, "", false, truncateDim)
}

// EmbedVideoFiles embeds video files into len(paths) rows: AVFoundation decodes
// the frames the processor's sampler keeps (MP4, MOV, ... H.264, HEVC; upright
// frames), then the path is EmbedVideos'. With timestamps each frame's block
// carries its mm:ss time.
func (m *Model) EmbedVideoFiles(ctx context.Context, paths []string, timestamps bool, truncateDim int) ([][]float32, error) {
	const op = "EmbedVideoFiles"
	return m.embedSources(ctx, op, wire.OpEmbedVideoSources, "paths", paths, nil, false, "", timestamps, truncateDim)
}

// EmbedVideoBytes is EmbedVideoFiles for whole encoded videos in memory.
// extension ("mp4", "mov", ...: 1 to 8 letters or digits; "" means mp4) is the
// type AVFoundation sniffs the bytes as, since it reads files and the bytes are
// written to a private temporary file for the call. A request is at most
// MaxRequestBytes; a larger video goes by path.
func (m *Model) EmbedVideoBytes(ctx context.Context, data [][]byte, extension string, timestamps bool, truncateDim int) ([][]float32, error) {
	const op = "EmbedVideoBytes"
	return m.embedSources(ctx, op, wire.OpEmbedVideoSources, "data", nil, data, true, extension, timestamps, truncateDim)
}

func checkExtension(op, ext string) error {
	if ext == "" {
		return nil
	}
	if len(ext) > MaxExtensionBytes {
		return argErr(op, "extension", "is %d bytes, the limit is %d", len(ext), MaxExtensionBytes)
	}
	for i := 0; i < len(ext); i++ {
		c := ext[i]
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z') {
			return argErr(op, "extension", "%q must be letters and digits", ext)
		}
	}
	return nil
}

func (m *Model) embedSources(ctx context.Context, op string, code uint32, field string, paths []string, data [][]byte, byValue bool, ext string, timestamps bool, truncateDim int) ([][]float32, error) {
	n := len(paths)
	if byValue {
		n = len(data)
	}
	if err := checkCount(op, field, n); err != nil {
		return nil, err
	}
	if err := m.checkTruncate(op, truncateDim); err != nil {
		return nil, err
	}
	var e encoded
	var err error
	if byValue {
		max := uint64(MaxImageFileBytes)
		switch code {
		case wire.OpEmbedAudioSources:
			max = MaxAudioFileBytes
		case wire.OpEmbedVideoSources:
			max = MaxVideoFileBytes
		}
		e, err = bytesTable(op, field, data, max)
	} else {
		e, err = pathTable(op, field, paths)
	}
	if err != nil {
		return nil, err
	}
	f := new(wire.Frame).U32(uint32(truncateDim)).U32(e.source)
	if code == wire.OpEmbedVideoSources {
		if err := checkExtension(op, ext); err != nil {
			return nil, err
		}
		if !byValue && ext != "" {
			return nil, argErr(op, "extension", "goes with encoded bytes, not file paths (a file's own name says what it is)")
		}
		f.U32(flag(timestamps)).Text(ext)
	}
	e.blob(f)
	f.U64s(e.offsets)
	return m.embed(ctx, op, code, f, n, truncateDim)
}

// ImageSize is the size of an image in pixels.
type ImageSize struct{ Height, Width int }

// ImageSoftTokens returns how many soft tokens (placeholders in the processor's
// sequence) each image of the given size makes, from its size alone: no pixel
// is touched. Up to 4096 sizes; works on a TextOnly model (it reads
// processor_config.json, not a tower), so a prompt can be budgeted before any
// media is read.
func (m *Model) ImageSoftTokens(ctx context.Context, sizes []ImageSize) ([]int, error) {
	const op = "ImageSoftTokens"
	if len(sizes) < 1 || len(sizes) > maxCallPictures {
		return nil, argErr(op, "sizes", "has %d sizes, expected 1..=%d", len(sizes), maxCallPictures)
	}
	dims := make([]uint32, 0, 2*len(sizes))
	for i, s := range sizes {
		if s.Height < 1 || s.Width < 1 || !fitsU32(s.Height) || !fitsU32(s.Width) {
			return nil, argErr(op, indexed("sizes", i), "%dx%d has no pixels", s.Height, s.Width)
		}
		dims = append(dims, uint32(s.Height), uint32(s.Width))
	}
	reply, err := m.call(ctx, op, wire.OpImageSoftTokens, new(wire.Frame).U32s(dims))
	if err != nil {
		return nil, err
	}
	return m.counts(op, reply, len(sizes))
}

// AudioSoftTokens returns how many soft tokens each clip of sampleCounts
// samples makes at sampleRate (the extractor's: 16000): the true entries of the
// audio tower's mask for the frames the extractor would build after its 30 s
// cut, 750 at most, with no feature computed. A clip too short for one frame is
// refused by index. Up to 4096 clips; works on a TextOnly model.
func (m *Model) AudioSoftTokens(ctx context.Context, sampleCounts []int, sampleRate int) ([]int, error) {
	const op = "AudioSoftTokens"
	if len(sampleCounts) < 1 || len(sampleCounts) > maxCallPictures {
		return nil, argErr(op, "sampleCounts", "has %d clips, expected 1..=%d", len(sampleCounts), maxCallPictures)
	}
	if sampleRate < 1 || !fitsU32(sampleRate) {
		return nil, argErr(op, "sampleRate", "%d is not a positive 32-bit integer", sampleRate)
	}
	counts := make([]uint64, len(sampleCounts))
	for i, c := range sampleCounts {
		if c < 1 || c > MaxClipSamples {
			return nil, argErr(op, indexed("sampleCounts", i), "%d samples, expected 1..=%d", c, MaxClipSamples)
		}
		counts[i] = uint64(c)
	}
	reply, err := m.call(ctx, op, wire.OpAudioSoftTokens, new(wire.Frame).U32(uint32(sampleRate)).U64s(counts))
	if err != nil {
		return nil, err
	}
	return m.counts(op, reply, len(sampleCounts))
}

// VideoInfo describes a video for VideoSoftTokens: the size of its frames
// (after any rotation), how many it has, and its source rate.
type VideoInfo struct {
	Height, Width, Frames int
	FPS                   float64
}

// VideoTokens is what a video becomes: the frames the processor's sampler
// keeps, and the soft tokens each makes, so its placeholders number their
// product.
type VideoTokens struct{ Frames, PerFrame int }

// VideoSoftTokens returns what each video becomes; nothing is decoded. Up to
// 4096 videos; works on a TextOnly model.
func (m *Model) VideoSoftTokens(ctx context.Context, videos []VideoInfo) ([]VideoTokens, error) {
	const op = "VideoSoftTokens"
	if len(videos) < 1 || len(videos) > maxCallPictures {
		return nil, argErr(op, "videos", "has %d videos, expected 1..=%d", len(videos), maxCallPictures)
	}
	dims := make([]uint32, 0, 2*len(videos))
	frames := make([]uint64, len(videos))
	fps := make([]float64, len(videos))
	for i, v := range videos {
		field := indexed("videos", i)
		switch {
		case v.Height < 1 || v.Width < 1 || !fitsU32(v.Height) || !fitsU32(v.Width):
			return nil, argErr(op, field, "frames are %dx%d, which has no pixels", v.Height, v.Width)
		case v.Frames < 1 || v.Frames > MaxVideoFrames:
			return nil, argErr(op, field, "has %d frames, expected 1..=%d", v.Frames, MaxVideoFrames)
		case math.IsNaN(v.FPS) || math.IsInf(v.FPS, 0) || v.FPS <= 0:
			return nil, argErr(op, field, "fps %v is not finite and positive", v.FPS)
		}
		dims = append(dims, uint32(v.Height), uint32(v.Width))
		frames[i] = uint64(v.Frames)
		fps[i] = v.FPS
	}
	f := new(wire.Frame).U32s(dims).U64s(frames).F64s(fps)
	reply, err := m.call(ctx, op, wire.OpVideoSoftTokens, f)
	if err != nil {
		return nil, err
	}
	pairs, err := m.counts(op, reply, 2*len(videos))
	if err != nil {
		return nil, err
	}
	out := make([]VideoTokens, len(videos))
	for i := range out {
		out[i] = VideoTokens{Frames: pairs[2*i], PerFrame: pairs[2*i+1]}
	}
	return out, nil
}

// maxCallPictures is tessl's MAX_CALL_IMAGES and MAX_CALL_CLIPS (both 4096):
// the most items a count call takes.
const maxCallPictures = 4096

func (m *Model) counts(op string, reply wire.Reply, n int) ([]int, error) {
	r := wire.NewReader(reply.Payload)
	raw := r.U64s("counts")
	if err := r.Done(); err != nil {
		return nil, protocolErr(op, "%v", err)
	}
	if len(raw) != n {
		return nil, protocolErr(op, "%d counts for %d items", len(raw), n)
	}
	out := make([]int, n)
	for i, c := range raw {
		if c > math.MaxInt32 {
			return nil, protocolErr(op, "count %d is implausibly large", c)
		}
		out[i] = int(c)
	}
	return out, nil
}

func indexed(field string, i int) string { return field + "[" + strconv.Itoa(i) + "]" }
