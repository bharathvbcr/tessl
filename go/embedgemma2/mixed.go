//go:build darwin && arm64

package embedgemma2

import (
	"context"
	"fmt"
	"unicode/utf8"

	"github.com/bharathvbcr/tessl/go/embedgemma2/internal/wire"
)

// segment kinds of the C ABI (TESSL_EMBEDGEMMA2_SEGMENT_*), as the request
// carries them.
const (
	segText           uint32 = 0
	segImage          uint32 = 1
	segAudio          uint32 = 2
	segVideo          uint32 = 3
	segVideoTimestamp uint32 = 4
)

// What a Segment is. These start at 1 so that the zero Segment, which no
// constructor makes, is not silently an empty text.
const (
	kindText = iota + 1
	kindImage
	kindAudio
	kindVideo
)

// Segment is one piece of a mixed item: a text, or a medium given as decoded
// arrays, a file path, or encoded bytes. Build one with Text, ImageSegment,
// ImageFileSegment, ImageBytesSegment and their audio and video siblings.
type Segment struct {
	kind       int    // kindText, kindImage, kindAudio or kindVideo; 0 is no Segment
	source     uint32 // for media: sourceRaw, sourcePaths or sourceBytes
	timestamps bool

	text  string
	image Image
	audio Audio
	video Video
	path  string
	data  []byte
	ext   string
}

// Text is a text segment.
func Text(s string) Segment { return Segment{kind: kindText, text: s} }

// ImageSegment is a raw image.
func ImageSegment(im Image) Segment { return Segment{kind: kindImage, source: sourceRaw, image: im} }

// ImageFileSegment is an image read from a file.
func ImageFileSegment(path string) Segment {
	return Segment{kind: kindImage, source: sourcePaths, path: path}
}

// ImageBytesSegment is an encoded image held in memory.
func ImageBytesSegment(data []byte) Segment {
	return Segment{kind: kindImage, source: sourceBytes, data: data}
}

// AudioSegment is a raw mono clip.
func AudioSegment(a Audio) Segment { return Segment{kind: kindAudio, source: sourceRaw, audio: a} }

// AudioFileSegment is a clip read from a file.
func AudioFileSegment(path string) Segment {
	return Segment{kind: kindAudio, source: sourcePaths, path: path}
}

// AudioBytesSegment is an encoded clip held in memory.
func AudioBytesSegment(data []byte) Segment {
	return Segment{kind: kindAudio, source: sourceBytes, data: data}
}

// VideoSegment is a raw video; with timestamps each frame's block carries its
// mm:ss time.
func VideoSegment(v Video, timestamps bool) Segment {
	return Segment{kind: kindVideo, source: sourceRaw, video: v, timestamps: timestamps}
}

// VideoFileSegment is a video read from a file.
func VideoFileSegment(path string, timestamps bool) Segment {
	return Segment{kind: kindVideo, source: sourcePaths, path: path, timestamps: timestamps}
}

// VideoBytesSegment is an encoded video held in memory; extension is the type
// hint EmbedVideoBytes describes ("" is mp4).
func VideoBytesSegment(data []byte, extension string, timestamps bool) Segment {
	return Segment{kind: kindVideo, source: sourceBytes, data: data, ext: extension, timestamps: timestamps}
}

// mixedPlan is the call's tables, built from the items.
type mixedPlan struct {
	texts     []string
	textCuts  []uint64
	kinds     []uint32
	refs      []uint64
	itemCuts  []uint64
	image     modality
	audio     modality
	video     modality
	rate      uint32
	extension string

	// raw tables
	pix       [][]byte
	imageDims []uint32
	pcm       [][]float32
	frames    [][]byte
	videoDims []uint32
	videoFPS  []float64
	paths     [3][]string
	data      [3][][]byte
}

// modality tracks one kind of medium across the call: how many, and the one
// source they all use.
type modality struct {
	n      int
	source uint32
	first  string // where the source was fixed, for the message
	cuts   []uint64
	total  uint64
}

// claim registers one medium of source s found at where, refusing a mix.
func (md *modality) claim(op string, s uint32, at string) error {
	if md.n == 0 {
		md.source, md.first = s, at
		md.cuts = []uint64{0}
	} else if md.source != s {
		return argErr(op, "items", "%s is %s but %s is %s: a call takes one source per modality (decoded arrays, files, or encoded bytes)",
			at, sourceName(s), md.first, sourceName(md.source))
	}
	md.n++
	return nil
}

func (md *modality) cut(size uint64) {
	md.total += size
	md.cuts = append(md.cuts, md.total)
}

func sourceName(s uint32) string {
	switch s {
	case sourceRaw:
		return "a decoded array"
	case sourcePaths:
		return "a file path"
	}
	return "encoded bytes"
}

func where(i, j int) string { return fmt.Sprintf("items[%d][%d]", i, j) }

// EmbedMixed embeds items that interleave text with media, into len(items) rows,
// without the caller counting a placeholder: items[i] is a list of segments, and
// tessl lays out the processor's sequence for it, writing each medium as its
// <boi> <image> x n <eoi> block (or audio, or one block per sampled video frame)
// with n from the tower, the text between media as one stretch, the whole wrapped
// in <bos> and <eos>. This is what sentence-transformers' structured messages
// produce through the processor. No prompt is added: write it in the first text.
//
// Within one call every modality uses one source: decoded arrays, file paths or
// encoded bytes (raw clips must share a SampleRate; encoded videos share an
// extension). Needs a multimodal model and the snapshot's tokenizer.json. Up to
// MaxBatch items and MaxSegments segments.
func (m *Model) EmbedMixed(ctx context.Context, items [][]Segment, truncateDim int) ([][]float32, error) {
	const op = "EmbedMixed"
	if err := checkCount(op, "items", len(items)); err != nil {
		return nil, err
	}
	if err := m.checkTruncate(op, truncateDim); err != nil {
		return nil, err
	}
	p, err := planMixed(op, items)
	if err != nil {
		return nil, err
	}
	f := new(wire.Frame).U32(uint32(truncateDim)).
		Strings(p.texts...).U64s(p.textCuts).U32s(p.kinds).U64s(p.refs).U64s(p.itemCuts)
	// Images.
	f.U32(p.image.source)
	switch p.image.source {
	case sourceRaw:
		f.Bytes(p.pix...)
	case sourcePaths:
		f.Strings(p.paths[0]...)
	default:
		f.Bytes(p.data[0]...)
	}
	f.U64s(p.image.cuts).U32s(p.imageDims)
	// Audio.
	f.U32(p.audio.source)
	switch p.audio.source {
	case sourceRaw:
		f.F32s(p.pcm...)
	case sourcePaths:
		f.Strings(p.paths[1]...)
	default:
		f.Bytes(p.data[1]...)
	}
	f.U64s(p.audio.cuts).U32(p.rate)
	// Videos.
	f.U32(p.video.source)
	switch p.video.source {
	case sourceRaw:
		f.Bytes(p.frames...)
	case sourcePaths:
		f.Strings(p.paths[2]...)
	default:
		f.Bytes(p.data[2]...)
	}
	f.U64s(p.video.cuts).U32s(p.videoDims).F64s(p.videoFPS).Text(p.extension)
	return m.embed(ctx, op, wire.OpEmbedSegments, f, len(items), truncateDim)
}

func planMixed(op string, items [][]Segment) (*mixedPlan, error) {
	p := &mixedPlan{textCuts: []uint64{0}, itemCuts: []uint64{0}}
	var textBytes uint64
	var videoExtSet bool
	var rateSet bool
	for i, item := range items {
		if len(item) == 0 {
			return nil, argErr(op, indexed("items", i), "has no segments")
		}
		for j, s := range item {
			at := where(i, j)
			switch s.kind {
			case kindText:
				if !utf8.ValidString(s.text) {
					return nil, argErr(op, at, "text is not valid UTF-8")
				}
				textBytes += uint64(len(s.text))
				if textBytes >= MaxTextBytes {
					return nil, argErr(op, at, "the segments hold %d bytes of text; a call takes fewer than %d", textBytes, MaxTextBytes)
				}
				p.kinds = append(p.kinds, segText)
				p.refs = append(p.refs, uint64(len(p.texts)))
				p.texts = append(p.texts, s.text)
				p.textCuts = append(p.textCuts, textBytes)
			case kindImage:
				if err := p.image.claim(op, s.source, at); err != nil {
					return nil, err
				}
				p.kinds = append(p.kinds, segImage)
				p.refs = append(p.refs, uint64(p.image.n-1))
				switch s.source {
				case sourceRaw:
					if err := checkImage(op, at, s.image); err != nil {
						return nil, err
					}
					p.pix = append(p.pix, s.image.Pix)
					p.imageDims = append(p.imageDims, uint32(s.image.Height), uint32(s.image.Width), uint32(s.image.Channels))
					p.image.cut(uint64(len(s.image.Pix)))
				default:
					if err := p.encodedMedium(op, at, 0, s, MaxImageFileBytes, &p.image); err != nil {
						return nil, err
					}
				}
			case kindAudio:
				if err := p.audio.claim(op, s.source, at); err != nil {
					return nil, err
				}
				p.kinds = append(p.kinds, segAudio)
				p.refs = append(p.refs, uint64(p.audio.n-1))
				switch s.source {
				case sourceRaw:
					if err := checkAudio(op, at, s.audio); err != nil {
						return nil, err
					}
					if rateSet && uint32(s.audio.SampleRate) != p.rate {
						return nil, argErr(op, at, "sample rate %d differs from the call's %d", s.audio.SampleRate, p.rate)
					}
					p.rate, rateSet = uint32(s.audio.SampleRate), true
					p.pcm = append(p.pcm, s.audio.Samples)
					p.audio.cut(4 * uint64(len(s.audio.Samples)))
				default:
					if err := p.encodedMedium(op, at, 1, s, MaxAudioFileBytes, &p.audio); err != nil {
						return nil, err
					}
				}
			case kindVideo:
				if err := p.video.claim(op, s.source, at); err != nil {
					return nil, err
				}
				kind := segVideo
				if s.timestamps {
					kind = segVideoTimestamp
				}
				p.kinds = append(p.kinds, kind)
				p.refs = append(p.refs, uint64(p.video.n-1))
				switch s.source {
				case sourceRaw:
					if err := checkVideo(op, at, s.video); err != nil {
						return nil, err
					}
					p.frames = append(p.frames, s.video.Frames)
					p.videoDims = append(p.videoDims, uint32(s.video.Height), uint32(s.video.Width))
					p.videoFPS = append(p.videoFPS, s.video.FPS)
					p.video.cut(uint64(len(s.video.Frames)))
				default:
					if err := p.encodedMedium(op, at, 2, s, MaxVideoFileBytes, &p.video); err != nil {
						return nil, err
					}
					if s.source == sourceBytes {
						if err := checkExtension(op, s.ext); err != nil {
							return nil, err
						}
						if videoExtSet && s.ext != p.extension {
							return nil, argErr(op, at, "extension %q differs from the call's %q", s.ext, p.extension)
						}
						p.extension, videoExtSet = s.ext, true
					} else if s.ext != "" {
						return nil, argErr(op, at, "an extension goes with encoded bytes, not a path")
					}
				}
			default:
				return nil, argErr(op, at, "is the zero Segment: build segments with Text, ImageSegment and the others")
			}
		}
		p.itemCuts = append(p.itemCuts, uint64(len(p.kinds)))
		if len(p.kinds) > MaxSegments {
			return nil, argErr(op, "items", "hold %d segments by item %d; a call takes at most %d", len(p.kinds), i, MaxSegments)
		}
	}
	return p, nil
}

// encodedMedium validates a path or encoded bytes segment and appends it to the
// plan's table number slot (0 images, 1 audio, 2 videos).
func (p *mixedPlan) encodedMedium(op, at string, slot int, s Segment, max uint64, md *modality) error {
	if s.source == sourcePaths {
		e, err := pathTable(op, at, []string{s.path})
		if err != nil {
			return err
		}
		p.paths[slot] = append(p.paths[slot], s.path)
		md.cut(e.offsets[1])
		return nil
	}
	e, err := bytesTable(op, at, [][]byte{s.data}, max)
	if err != nil {
		return err
	}
	p.data[slot] = append(p.data[slot], s.data)
	md.cut(e.offsets[1])
	return nil
}
