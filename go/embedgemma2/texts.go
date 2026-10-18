//go:build darwin && arm64

package embedgemma2

import (
	"context"
	"unicode/utf8"

	"github.com/bharathvbcr/tessl/go/embedgemma2/internal/wire"
)

// MaxTokenizeTexts is the most texts one Tokenize call takes.
const MaxTokenizeTexts = 1 << 16

// MaxDecodeIDs is the most ids one Decode call takes.
const MaxDecodeIDs = 1 << 24

// packTexts lays texts end to end for the frame: the strings, and the
// len(texts)+1 byte offsets that cut them. Every text must be UTF-8, and the
// texts of a call hold fewer than MaxTextBytes in all.
func packTexts(op string, texts []string, limit int) ([]string, []uint64, error) {
	if len(texts) < 1 || len(texts) > limit {
		return nil, nil, argErr(op, "texts", "has %d texts, expected 1..=%d", len(texts), limit)
	}
	offsets := make([]uint64, len(texts)+1)
	var total uint64
	for i, t := range texts {
		if !utf8.ValidString(t) {
			return nil, nil, argErr(op, "texts", "text %d is not valid UTF-8", i)
		}
		total += uint64(len(t))
		if total >= MaxTextBytes {
			return nil, nil, argErr(op, "texts", "the texts hold %d bytes by text %d; a call takes fewer than %d", total, i, MaxTextBytes)
		}
		offsets[i+1] = total
	}
	return texts, offsets, nil
}

func checkPrompt(op, prompt string) error {
	if len(prompt) > MaxPromptNameBytes {
		return argErr(op, "prompt", "is %d bytes, the limit is %d", len(prompt), MaxPromptNameBytes)
	}
	if !utf8.ValidString(prompt) {
		return argErr(op, "prompt", "is not valid UTF-8")
	}
	return nil
}

// EmbedTexts embeds texts into len(texts) L2-normalized rows of EmbeddingDim
// values (truncateDim values when it is not 0).
//
// prompt names one of the snapshot's task prompts (see Prompts), written in
// front of every text as sentence-transformers' encode(prompt_name=...) does;
// "" applies the snapshot's default prompt, and the checkpoint has none. Each
// text is wrapped in <bos> and <eos>, and may not pass MaxTokens tokens. An
// empty text is the two-token sequence <bos> <eos>.
//
// truncateDim 0 returns the full width; otherwise it must be in
// 1..=EmbeddingDim and returns that Matryoshka prefix, renormalized.
//
// Up to MaxBatch texts per call. Works on a TextOnly model.
func (m *Model) EmbedTexts(ctx context.Context, texts []string, prompt string, truncateDim int) ([][]float32, error) {
	const op = "EmbedTexts"
	strs, offsets, err := packTexts(op, texts, MaxBatch)
	if err != nil {
		return nil, err
	}
	if err := checkPrompt(op, prompt); err != nil {
		return nil, err
	}
	if err := m.checkTruncate(op, truncateDim); err != nil {
		return nil, err
	}
	f := new(wire.Frame).U32(uint32(truncateDim)).Strings(strs...).U64s(offsets).Text(prompt)
	return m.embed(ctx, op, wire.OpEmbedTexts, f, len(texts), truncateDim)
}

// Encode embeds token ids you tokenized yourself (the ids must carry <bos>,
// <eos> and any task prompt already): ids[i] is sequence i, which may not be
// empty or pass MaxTokens. Up to MaxBatch sequences.
func (m *Model) Encode(ctx context.Context, ids [][]uint32, truncateDim int) ([][]float32, error) {
	const op = "Encode"
	if err := checkCount(op, "ids", len(ids)); err != nil {
		return nil, err
	}
	if err := m.checkTruncate(op, truncateDim); err != nil {
		return nil, err
	}
	flat, offsets, err := flattenIDs(op, ids)
	if err != nil {
		return nil, err
	}
	f := new(wire.Frame).U32(uint32(truncateDim)).U32s(flat).U64s(offsets)
	return m.embed(ctx, op, wire.OpEncode, f, len(ids), truncateDim)
}

func flattenIDs(op string, ids [][]uint32) ([]uint32, []uint64, error) {
	var total int
	for i, s := range ids {
		if len(s) == 0 {
			return nil, nil, argErr(op, "ids", "sequence %d is empty", i)
		}
		total += len(s)
	}
	flat := make([]uint32, 0, total)
	offsets := make([]uint64, 1, len(ids)+1)
	for _, s := range ids {
		flat = append(flat, s...)
		offsets = append(offsets, uint64(len(flat)))
	}
	return flat, offsets, nil
}

// Tokenize returns the ids the model is given for each of texts with the
// prompt prompt ("" for the snapshot's default): <bos>, the prompt, the text,
// <eos>. No limit of the encoder applies, so a text may be longer than
// MaxTokens. Up to MaxTokenizeTexts texts.
func (m *Model) Tokenize(ctx context.Context, texts []string, prompt string) ([][]uint32, error) {
	const op = "Tokenize"
	strs, offsets, err := packTexts(op, texts, MaxTokenizeTexts)
	if err != nil {
		return nil, err
	}
	if err := checkPrompt(op, prompt); err != nil {
		return nil, err
	}
	f := new(wire.Frame).Strings(strs...).U64s(offsets).Text(prompt)
	reply, err := m.call(ctx, op, wire.OpTokenize, f)
	if err != nil {
		return nil, err
	}
	r := wire.NewReader(reply.Payload)
	flat, cuts := r.U32s("ids"), r.U64s("id_offsets")
	if err := r.Done(); err != nil {
		return nil, protocolErr(op, "%v", err)
	}
	if len(cuts) != len(texts)+1 || cuts[0] != 0 || cuts[len(cuts)-1] != uint64(len(flat)) {
		return nil, protocolErr(op, "%d offsets for %d texts and %d ids", len(cuts), len(texts), len(flat))
	}
	out := make([][]uint32, len(texts))
	for i := range out {
		lo, hi := cuts[i], cuts[i+1]
		if hi < lo || hi > uint64(len(flat)) {
			return nil, protocolErr(op, "id offsets %d..%d out of order", lo, hi)
		}
		out[i] = flat[lo:hi:hi]
	}
	return out, nil
}

// Decode turns ids back into text as the tokenizer library's decode does (an id
// with no token is skipped). With skipSpecialTokens the library's default,
// <bos>, <eos> and the other special tokens are left out. Up to MaxDecodeIDs.
func (m *Model) Decode(ctx context.Context, ids []uint32, skipSpecialTokens bool) (string, error) {
	const op = "Decode"
	if len(ids) > MaxDecodeIDs {
		return "", argErr(op, "ids", "has %d ids, the limit is %d", len(ids), MaxDecodeIDs)
	}
	skip := uint32(0)
	if skipSpecialTokens {
		skip = 1
	}
	f := new(wire.Frame).U32(skip).U32s(ids)
	reply, err := m.call(ctx, op, wire.OpDecode, f)
	if err != nil {
		return "", err
	}
	r := wire.NewReader(reply.Payload)
	text := r.Blob("text")
	if err := r.Done(); err != nil {
		return "", protocolErr(op, "%v", err)
	}
	return string(text), nil
}

// Prompts are the snapshot's named task prompts
// (config_sentence_transformers.json).
type Prompts struct {
	// Names lists the prompts in the file's order.
	Names []string
	// Text maps each name to the text written in front of a text.
	Text map[string]string
	// Default names the prompt applied when none is given; HasDefault is false
	// when the snapshot names none (the checkpoint's is null).
	Default    string
	HasDefault bool
}

// Prompts returns the snapshot's named prompts. They are read once and cached;
// a snapshot with no config_sentence_transformers.json has none.
func (m *Model) Prompts(ctx context.Context) (Prompts, error) {
	const op = "Prompts"
	m.promptsMu.Lock()
	defer m.promptsMu.Unlock()
	if m.prompts != nil {
		return m.prompts.clone(), nil
	}
	reply, err := m.call(ctx, op, wire.OpPrompts, new(wire.Frame))
	if err != nil {
		return Prompts{}, err
	}
	r := wire.NewReader(reply.Payload)
	def := r.I32("default_prompt")
	count := r.U32("prompt_count")
	p := Prompts{Text: map[string]string{}}
	for i := uint32(0); i < count && r.Err() == nil; i++ {
		name, text := string(r.Blob("name")), string(r.Blob("text"))
		if r.Err() != nil {
			break
		}
		p.Names = append(p.Names, name)
		p.Text[name] = text
	}
	if err := r.Done(); err != nil {
		return Prompts{}, protocolErr(op, "%v", err)
	}
	if def >= 0 {
		if int(def) >= len(p.Names) {
			return Prompts{}, protocolErr(op, "default prompt %d of %d", def, len(p.Names))
		}
		p.Default, p.HasDefault = p.Names[def], true
	}
	m.prompts = &p
	return p.clone(), nil
}

func (p Prompts) clone() Prompts {
	c := Prompts{Names: append([]string(nil), p.Names...), Text: make(map[string]string, len(p.Text)), Default: p.Default, HasDefault: p.HasDefault}
	for k, v := range p.Text {
		c.Text[k] = v
	}
	return c
}
