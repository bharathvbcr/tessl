# EmbeddingGemma 2 text encoder

`tessl::embedgemma2` runs `google/embeddinggemma-2`'s text path (the 271M
encoder) on tessl's kernels, from the checkpoint's own `model.safetensors`.
The vision tower and its embedder are `tessl::embedgemma2_vision`
([below](#vision-tower)); the audio tower and its embedder are
`tessl::embedgemma2_audio` ([Audio tower](#audio-tower)). All three behind one
snapshot and one call, from Rust, C and Python, are `tessl::embedgemma2_mm`
([One call: text, images and audio](#one-call-text-images-and-audio)).

> **Status: run on the GPU against sentence-transformers** (Apple M5 Pro,
> 2026-10-08, branch `fix/embedgemma2-kv-and-e2e` on base `163b28d`, snapshot
> `914f7f89142e33e77833254d9c9b90c3cef7303b`; torch 2.12.1, transformers
> 5.19.0, sentence-transformers 6.1.0). Every bound below holds, the
> per-layer traces included, with at least two orders of magnitude to spare
> on the embeddings.

## What it computes

sentence-transformers' `Transformer -> Pooling(mean, include_prompt) ->
Normalize` for this checkpoint, as transformers 5.19's
`EmbeddingGemma2TextModel` defines it. The block and the five places it is
*not* Gemma 3 are written out in the module docs of `src/embedgemma2.rs`:
attention scale 1.0, weighted q/k norms and a weightless v norm, norms that
multiply by `w` (not `1 + w`), projection-only per-layer inputs, and a
symmetric inclusive sliding window `|i - j| <= 512`.

`encode` takes token ids (the tokenizer adds `<bos>` and `<eos>`; prompts such
as `"task: search result | query: "` are part of the text). The ids can come
from `tessl::embedgemma2_tok` ([Tokenizer](#tokenizer)), which reads the
snapshot's `tokenizer.json` in Rust, or from any tokenizer the caller holds.
sentence-transformers applies no length limit to this checkpoint; tessl
refuses sequences past `max_tokens` (default 8192, the model card's context)
rather than truncating, so a caller that wants truncation does it visibly.

## Matryoshka prefixes

`encode(batch, Some(d), trace)` returns the first `d` columns of each
embedding, normalized on the GPU over those columns. That is
sentence-transformers' `encode(truncate_dim=d, normalize_embeddings=True)`:
it slices after its `Normalize` module, then normalizes again. The model
card's sizes are 768, 512, 256 and 128; any `d` in `1..=768` runs. `None` is
the full 768.

## Batches

`encode` takes up to 256 sequences of any mix of lengths. It sorts them
longest first and cuts that order into forwards, minimizing padded rows plus
a fixed per-forward cost (each forward re-reads every weight), with each
forward at most 32,768 padded rows. A 6,000-token document therefore runs on
its own instead of padding a batch of queries to its length. Embeddings come
back in the caller's order, and `EncodeOutput::forwards` says how many
forwards ran. The per-forward cost (67 padded rows, see Speed) is an estimate
that only steers the split: a sequence's embedding does not depend on its
batch beyond GEMM tiling, which the model test bounds at `1e-5`. Because the
cost differs between a queued run (67) and a serialized one (1046), the same
batch can be cut into different forwards in the two modes, and the embeddings
then agree to that bound rather than bit for bit.

`encode` queues every dispatch of the batch and waits once, on a runtime in
any mode: it opens a `GpuRuntime::async_scope` and leaves the runtime's
`set_async_encode` flag as it found it. The runtime is `!Send`/`!Sync`, so
nothing else dispatches on it during the call, and anything the caller queued
before it is waited for by `encode`'s own wait. With `trace` the forward still
waits at every layer, to read the residual back.

The activations are allocated once per `encode`, for its largest forward,
and every forward works on a prefix of them. None is zeroed on the host:
every element a forward reads, an earlier kernel of that forward wrote, which
`tests/embedgemma2_tiny.rs` checks by poisoning every unzeroed allocation with
NaN (and by asserting an encode zeroes no host bytes and allocates the same
whether it runs one forward or two).

The set is 400 MiB at 8192 rows (`tests/embedgemma2_memory.rs` itemizes it to
the byte on the tiny model), and consecutive `encode`s hold one set, not two:
`encode` hands its activations and each forward's uploaded inputs back to the
runtime before its one wait, and the runtime keeps them alive until that wait
has finished and then returns them to the pool, where the next call's
allocations find them. (The pool used to recycle a dropped buffer only at the
first commit after the drop; dropped after the wait, the buffers were recycled
at the next call's wait, which comes after that call has allocated. It now
also recycles at allocation time, see "Pool recycle at allocation" below, so
this hand-back is no longer needed for the memory bound; it still lets the
wait itself return the set.) The `q|k|v`
GEMM's output and the MLP's `gate | up` are one buffer: the q/k-norm-RoPE kernel
is the last reader of the first, and the second is written after the attention,
so their lifetimes within a layer never overlap, in either `GemmOperands`
mode.

One K/V pair, sized for the widest layer, serves every layer and forward. The
K/V writer and the attention both derive the per-sequence capacity from the
buffers' size, so they agree on the stride whatever the layer's width or the
forward's shape. When the attention read at stride `seq` instead, a buffer
holding more positions than `seq` (a narrower layer's, or now a smaller
forward's) misplaced every sequence after the first: sequence 1 of a batch
came out at cosine `0.27`. `tests/embedgemma2_tiny.rs` mixes K/V widths 512
and 256 and runs a 30-token forward in buffers sized for a 4000-token one.

## From Python / C

The C ABI in `src/capi_embedgemma2.rs` (a module of its own beside
`src/capi.rs`, whose handle, thread and status conventions it reuses) and
`python/tessl_torch/embedgemma2.py` put the encoder behind Python. The
tokenizer, the task prompts and the image, audio and video preprocessing are
Rust's, read from the snapshot ([Raw inputs](#raw-inputs-texts-images-audio-video-and-media)),
so a caller needs numpy and the built library and nothing else. Build the
library once (`cargo build --release`, or point `TESSL_LIB` at a
`libtessl.dylib`):

```python
from tessl_torch.embedgemma2 import EmbedGemma2

model = EmbedGemma2.from_pretrained(snapshot_dir)   # a local snapshot directory
q = model.encode_texts(["how do I speed up attention?"], prompt_name="query")
d = model.encode_texts(["Sliding-window attention ..."], prompt_name="document")
scores = q @ d.T                                    # rows are L2-normalized float32
ids = model.tokenize(["a text"])                    # the ids the model is given
small = model.encode(ids, truncate_dim=256)         # or ids you tokenized yourself
```

`encode_texts` writes the snapshot's own prompt (the `prompts` of
`config_sentence_transformers.json`) in front of each text and wraps it in
`<bos>`/`<eos>`, which is what sentence-transformers' `encode(prompt_name=...)`
does; the tokenizer is [`Tokenizer`](#tokenizer), checked against the reference
ids. A tokenizer of your own is still accepted, as the optional keyword
`tokenizer=` (a `tokenizers.Tokenizer` or a transformers tokenizer, duck-typed
and never imported): it is asked for the ids and the prompt text still comes
from the snapshot. It stays because `encode_texts(texts, tokenizer, ...)` was
this method's documented form.

The C entries, each returning a status with its message in a caller buffer, as
the training entries do:

| Entry | Does |
|---|---|
| `tessl_embedgemma2_load(runtime, dir, dir_len, &out, err, err_len)` | reads `config.json` and `model.safetensors` from the snapshot directory (UTF-8, not NUL-terminated, at most 4096 bytes) |
| `tessl_embedgemma2_free(model)` | frees it once; null is a no-op |
| `tessl_embedgemma2_embedding_dim` / `_max_tokens` / `_set_max_tokens` | the width, and the per-sequence token limit (1 to 32768, default 8192) |
| `tessl_embedgemma2_encode(model, ids, n_ids, offsets, n_seqs, truncate_dim, out, out_len, err, err_len)` | `ids` flat `u32`, `offsets[n_seqs + 1]` CSR-style `u64` (starting at 0, strictly increasing, ending at `n_ids`), `truncate_dim` 0 for the full width, `out` `f32 [n_seqs, dim]` with `out_len` exactly `n_seqs * dim` |

Everything about an `encode` call is checked before anything is allocated or
sent to the GPU, and the output is written only when the whole batch has
succeeded: null pointers, `n_seqs` past 256, an offset table that is not
monotone or does not end at `n_ids`, a sequence past `max_tokens`, a token id
past the vocabulary, a `truncate_dim` past the width, an `out_len` that is not
the output's size. No caller length sizes a slice before it is bounded by those
limits. The model takes its own reference to the runtime, so it stays valid if
the runtime handle is freed first. As everywhere in the C ABI, a handle is used
from the thread that made it (a foreign-thread call is refused) and a freed
handle is the caller's to drop; the runtime must have relaxed f32 precision off,
as the C ABI leaves it. `tests/capi_embedgemma2.rs` pins each refusal and that
a successful call equals `EmbedGemma2Model::encode` bit for bit on the same
weights; `python/tests/test_embedgemma2.py` holds the real checkpoint to the
bounds below (set `EMBEDGEMMA2_SNAPSHOT`).

### Images and audio

Three further entries, in the same module, take what a transformers processor
returns. They are additive: no existing signature or `#[repr(C)]` layout
changed, so they bumped no ABI version of their own (they ship at ABI 12, whose
bump is the C ABI 12 entry points), and a library built before them is named as
such by the Python binding.

| Entry | Does |
|---|---|
| `tessl_embedgemma2_load_multimodal(runtime, dir, dir_len, operands, &out, err, err_len)` | `tessl_embedgemma2_load` with the vision and audio towers as well: one read of `config.json`, one open of `model.safetensors`. `operands` is `TESSL_OPERANDS_EXACT_F32` (0) or `TESSL_OPERANDS_BF16` (1) and applies to the **text encoder and both towers** (the audio tower since [Audio tower bf16 and speed](#audio-tower-bf16-and-speed)). The handle takes every text entry too, and `_free` |
| `tessl_embedgemma2_encode_images(model, pixel_values, pixel_values_len, position_ids, position_ids_len, offsets, n_images, truncate_dim, out, out_len, err, err_len)` | `pixel_values` f32 `[offsets[n_images], patch_dim]` and `position_ids` i32 `[offsets[n_images], 2]`, the processor's `pixel_values` and `image_position_ids` with the images laid end to end (padding patches `(-1, -1)` included), `offsets` CSR-style `u64` over patch rows; `out` is f32 `[n_images, dim]`, one row per image |
| `tessl_embedgemma2_encode_audio(model, input_features, input_features_len, input_features_mask, mask_len, offsets, n_clips, truncate_dim, out, out_len, err, err_len)` | `input_features` f32 `[offsets[n_clips], mel_bins]` and the mask one byte per frame (0 or 1; any other byte is refused), `offsets` over frames; one row per clip |

Each row is the embedding of the processor's sequence for that item
(`image_sequence` / `audio_sequence` of the tower's soft-token count) with the
tower's soft tokens in place of the placeholders, so it equals sentence-transformers'
`encode(image_or_audio)`. Every pointer and length is checked before any
allocation or GPU work and the output is written only on success, as for
`encode`, and the lengths are bounded before they size a slice: `n_images` by
256 and an image's patch rows by 16,384 (`MAX_BATCH_PATCHES`; the processor's
`P` is 2520), `n_clips` by 256 and a clip's frames by 16,384
(`MAX_FRAMES_PER_FORWARD`), each array length must be exactly what the offsets
imply, `out_len` exactly `n * dim`. A clip's features are copied into the owned
form the audio tower takes (`AudioInput`) after its length was bounded. A
text-only handle refuses both entries, naming `load_multimodal`.
`tests/capi_embedgemma2_multimodal.rs` pins each refusal (the over-limit cases
declare lengths far past tiny buffers, so an entry that read first would
fault) and that a successful call equals `EmbedGemma2Multimodal::encode` bit
for bit; `tests/capi_contract.rs` holds the new entries and
`python/tessl_torch/embedgemma2.py` to the same signature rules, injected faults
included (it learned `c_uint8` for the mask). Each entry is one
`EmbedGemma2Multimodal::encode` call, so it makes one host wait (two until the
GPU-resident hand-off; no signature or layout changed).

From Python (needs only numpy and the built library; `transformers` is not
imported and not a dependency):

```python
model = EmbedGemma2.from_pretrained(snapshot_dir, multimodal=True)    # operands="bf16" for the text encoder and the vision tower
out = processor(images=images, return_tensors="np")                   # your transformers processor
emb = model.encode_images(out["pixel_values"], out["image_position_ids"])   # float32 [n, 768]
emb = model.encode_audio(feats["input_features"], feats["input_features_mask"])   # likewise for audio
```

(With no processor of your own, `embed_images` / `embed_audio` take the raw
arrays: [Raw inputs](#raw-inputs-texts-images-audio-video-and-media).)
`encode_images` / `encode_audio` take the processor's arrays or tensors
(`[N, P, 768]` / `[N, F, 128]`, one item without the leading axis, or a list of
per-item arrays of different lengths; numpy or torch, recognised by their
methods and never imported; bf16 widened first), check dtype, shape, counts and
the int32 and 0/1 ranges, and hand flat arrays to the library.
(`embed_images` / `embed_audio` used to call a processor passed in; they take
raw arrays now and the processor path is `encode_images` / `encode_audio` on its
output, which `test_embedgemma2_mm.py` still holds to the saved tensors.)
`python/tests/test_embedgemma2_mm_host.py` holds the argument handling to
a recording stand-in library; `python/tests/test_embedgemma2_mm.py` runs the
tiny checkpoint, and with `EMBEDGEMMA2_SNAPSHOT` and `EMBEDGEMMA2_MM_REF_DIR`
the real one (see Observed below).

### Raw inputs: texts, images, audio, video and media

Eleven further entries run the snapshot's own front ends in Rust, so the
caller hands over UTF-8, pixels, PCM and frames and needs no tokenizer, no
processor and no `transformers`. They are additive like the entries above (ABI
12 is unreleased: `CHANGELOG.md` lists it under `[Unreleased]` and no tag
contains the C ABI at all, so nothing bumps), they live at the end of
`src/capi_embedgemma2.rs`, and every one of them goes through the same
`guarded` boundary: a null or foreign-thread handle is refused before a
reference is formed, a panic is caught, the output is written only on
success.

| Entry | Takes | Runs |
|---|---|---|
| `tessl_embedgemma2_embed_texts(model, text_bytes, text_bytes_len, text_offsets, n_texts, prompt_name, prompt_name_len, truncate_dim, out, out_len, err, err_len)` | one UTF-8 byte array and `n_texts + 1` `u64` offsets (starting at 0, never decreasing, ending at `text_bytes_len`; an empty text is `<bos> <eos>`), a prompt name or null for the snapshot's default one | `Tokenizer::encode_with_prompt` per text, then `encode`; works on either kind of handle |
| `tessl_embedgemma2_tokenize(model, text_bytes, .., prompt_name, prompt_name_len, ids_out, ids_cap, id_offsets_out, n_ids_out, err, err_len)` | the texts as above | the ids end to end plus `n_texts + 1` offsets; called first with a null `ids_out` and capacity 0 it writes only the total |
| `tessl_embedgemma2_decode(model, ids, n_ids, skip_special_tokens, buf, buf_len, out_len, err, err_len)` | up to 2^24 ids, skip flag 0 or 1 | the library's `decode`, UTF-8 into `buf`, its length in `*out_len` (a null `buf` and 0 asks for the length; a short buffer is refused with the length set) |
| `tessl_embedgemma2_prompt_count` / `_prompt_name(index, buf, ..)` / `_prompt_get(name, name_len, buf, ..)` / `_default_prompt(&index)` | the snapshot's `config_sentence_transformers.json` | the count, a name by index, a prompt's text, and the default's index (-1 for none) |
| `tessl_embedgemma2_embed_images(model, pixels, pixels_len, image_dims, n_images, truncate_dim, out, out_len, err, err_len)` | `u8` HWC images laid end to end, `image_dims` `[n, 3]` `u32` `(height, width, channels)`, channels 1 to 4 (alpha dropped as PIL's `convert("RGB")` does) | `preprocess_image`, the vision tower |
| `tessl_embedgemma2_embed_audio(model, samples, samples_len, sample_offsets, n_clips, sampling_rate, truncate_dim, out, out_len, err, err_len)` | mono `f32` PCM, clips end to end with `n_clips + 1` strictly increasing offsets; any rate but the extractor's is refused | `AudioFrontEnd::extract`, the audio tower |
| `tessl_embedgemma2_embed_videos(model, frames, frames_len, video_dims, frame_offsets, video_fps, n_videos, timestamps, truncate_dim, out, out_len, err, err_len)` | decoded RGB `u8` frames end to end, `[n, 2]` `(height, width)` per video, frame offsets, one source fps each, `timestamps` 0 or 1 | `preprocess_video`, then `MmItem::Video` or `MmItem::TimestampedVideo` |
| `tessl_embedgemma2_embed_media(model, ids, n_ids, id_offsets, n_items, <images>, <clips>, sampling_rate, <videos>, truncate_dim, out, out_len, err, err_len)` | caller ids per item with the raw media their placeholders stand for, each modality laid out as above plus `n_items + 1` per-item cuts into it | `MmItem::Media` |
| `tessl_clip_coef(total_norm, max_norm) -> f32` | two `f64` | `torch.nn.utils.clip_grad_norm_`'s coefficient; see below |

The tokenizer, `processor_config.json` and the log-mel front end are read from
the snapshot directory the handle was loaded from, on the first call that needs
them, and kept on the handle. A file that is missing is refused naming it and is
read again by the next call (nothing failed is cached); a snapshot without any of
them still loads and `encode`s ids. `config_sentence_transformers.json` is
optional (no prompts, no default); a `processor_config.json` whose patch size,
pooling cell or mel bins do not fit the loaded towers is refused naming both.

**Bounds.** Every count is bounded before it sizes a slice, and every raw entry
bounds what the preprocessing *would* build before it builds anything. The
output has one row per item, so `n_texts`, `n_images`, `n_clips`, `n_videos` and
`n_items` are at most `MAX_BATCH` (256); `MAX_IMAGES` and `MAX_CLIPS` are the size
of one tower forward and bound nothing here any more (the image and clip entries
of the previous section had them as their count caps; they are `MAX_BATCH` now
and the call is bounded by the totals). What a call may hold is the per-call
totals of [Pictures and clips past a forward](#pictures-and-clips-past-a-forward):
`MAX_CALL_IMAGES` pictures (images and video frames) and `MAX_CALL_PATCHES`
padded patches, `MAX_CALL_CLIPS` clips and `MAX_CALL_FRAMES` frames. A raw image
becomes `max_soft_tokens * pooling_kernel²` patches of 3 KiB whatever its size,
so `n_images * max_patches` is compared with `MAX_CALL_PATCHES` before the first
image is resized, and a video counts the frames it samples
(`sample_frame_indices`). A text is below `MAX_TEXT_BYTES` (2^28) and a
tokenized one is refused as soon as it passes `max_tokens`; a clip is at most
`TESSL_EMBEDGEMMA2_MAX_CLIP_SAMPLES` (2^28) samples before the extractor cuts it
to 30 s; a call takes at most `MAX_VIDEO_FRAMES` (2^20) decoded frames. The
over-limit tests pass buffers four elements long with lengths in the millions.

**Precision.** An `operands` bf16 handle accepts a runtime with relaxed f32
GEMMs switched on, for every call, clips and videos included: none of its GEMMs
is an f32 one, so the switch changes nothing (the test shows the embeddings of
texts, images, clips, videos and timestamped videos equal bit for bit with it on
and off). An exact-f32 handle refuses it by name on every call that runs a
model. Nothing in the C ABI turns relaxed precision on; `tests/capi_embedgemma2_raw.rs`
reaches the runtime through a test hook to pin both.

**Safety contract.** Each entry's `# Safety` section says what a pointer must
hold; the common rule is that a pointer is null (refused, or accepted where its
length is 0) or addresses exactly its stated number of elements, aligned for its
type, and that nothing writes the input arrays during the call. The string
queries (`prompt_name`, `prompt_get`, `decode`, `tokenize`) write their length
cell before anything else and the bytes only to a buffer that holds them.

```python
model = EmbedGemma2.from_pretrained(snapshot_dir, multimodal=True)
emb = model.embed_images([rgb_uint8_hwc, gray_uint8_hw])        # [H, W, 1..4] uint8 arrays, a PIL image works
emb = model.embed_audio([pcm_float32_16khz])                     # mono clips, PCM in [-1, 1]
emb = model.embed_video(frames_uint8_thwc, fps=24.0, timestamps=True)    # embed_videos([..], fps=[..]) for several
emb = model.media([{"ids": ids, "images": [rgb], "audio": [pcm], "videos": [(frames, fps)]}])
ids = model.tokenize(["a text"], prompt_name="document"); text = model.decode(ids[0])
```

The Python side is marshalling: it flattens the arrays and cuts, and checks only
what ctypes would wrap silently, the dtype first: uint8 for pixels (a float or
wider integer cast to `uint8` wraps), floating point for PCM (an `int16` clip
would need a scale the binding cannot guess), dimensions that fit a `u32`, and
integer ranges (`truncate_dim`, `sampling_rate`, ids). (Files, bytes and
segment-built items are [the next section](#files-bytes-soft-token-counts-and-text-interleaved-with-media).) Everything else (the
channel count, a clip too short for a frame, a text over the token limit, an
unknown prompt, a placeholder count that is not the media's soft-token count,
fps not positive) is the library's refusal and arrives as its message. The
prompts-file reader of the previous binding (`_prompt_config`, JSON parsing,
stat and size checks) and the processor-calling `embed_images` / `embed_audio`
are gone; the tokenizer path of `encode_texts` fetches the prompt text with
`tessl_embedgemma2_prompt_get`. The ids form of `media` needs ids of your own
because the placeholders and the number of soft tokens each medium makes are the
checkpoint's (`config.json`, `processor_config.json`); the `*_soft_token_count`
entries tell the counts, and the segments form of the next section lays the ids
out; `embed_images` and friends build the processor's own sequence.

`tessl_clip_coef` is `clip_grad_norm_`'s coefficient in torch's f32 arithmetic.
The Rust side had no such function to reuse (`src/capi.rs` only forms the norm,
`tessl_qwen35_grad_sq_norm`, and the benchmark's own `1 / sqrt(sq).max(1)` is a
different formula), and the arithmetic is not `max_norm / (norm + 1e-6)` as an f32
division: torch's `max_norm / tensor` multiplies the reciprocal, so a division
differs in the last bit for about one input in eight. The function is
`(1 / (norm + 1e-6)) * max_norm` in f32 after rounding both inputs to f32,
clamped to at most 1, NaN kept. `python/tests/test_clip_coef.py` checks it
against torch over 200,000 pairs bit for bit, and `tests/capi_contract.rs` holds
golden values (eight of which a division gets wrong). `tessl_torch.clip_coef`
calls it.

**Tests.** `tests/capi_embedgemma2_raw.rs` (tiny checkpoint, tokenizer, prompts
and a processor config for the tiny towers) pins every entry bit for bit against
the Rust facade on the same weights, both `operands`, and every refusal: the
output untouched, the handle usable, a declared length bounded before it is read,
a missing file named and read again. `python/tests/test_embedgemma2_host.py` and
`..._mm_host.py` hold the marshalling to a recording stand-in;
`python/tests/test_embedgemma2_raw.py` runs the tiny snapshot and, opt-in, the real
checkpoint against sentence-transformers.

### Files, bytes, soft-token counts and text interleaved with media

Seven further entries finish the C ABI for callers that hold media as files or
encoded bytes, or that interleave text with media. They are additive like the
rest (nothing bumps), sit at the end of `src/capi_embedgemma2.rs`, and go through
the same `guarded` boundary. The decoders are [`tessl::media`](#media-decoding)'s
(ImageIO, AudioToolbox, AVFoundation) through
`embedgemma2_media::{prepare_image, prepare_audio, OpenedVideo}`, which is the
Rust owner of "file or bytes to tower input": decode with the defaults the
reference does, preprocess at once, drop the decoded pixels.

| Entry | Takes | Runs |
|---|---|---|
| `tessl_embedgemma2_embed_image_sources(model, source, blob, blob_len, offsets, n_images, truncate_dim, out, out_len, err, err_len)` | `source` `TESSL_EMBEDGEMMA2_SOURCE_PATHS` (1: UTF-8 paths) or `_BYTES` (2: whole encoded files), laid end to end in `blob` and cut by `n_images + 1` strictly increasing `u64` offsets from 0 to `blob_len`; `n_images` 1 to 256 | ImageIO, `preprocess_image`, the vision tower; one wait |
| `tessl_embedgemma2_embed_audio_sources(.. same, n_clips ..)` | the same | AudioToolbox to 16 kHz mono (the channel mean), `AudioFrontEnd::extract`, the audio tower; one wait |
| `tessl_embedgemma2_embed_video_sources(model, source, blob, blob_len, offsets, n_videos, extension, extension_len, timestamps, truncate_dim, out, out_len, err, err_len)` | as above; `extension` (1 to 8 letters or digits; null with 0 is `mp4`, paths take null with 0) is the type video bytes are sniffed as; `timestamps` 0 or 1 | the sampler's frames decoded, `MmItem::Video` or `MmItem::TimestampedVideo`; one wait |
| `tessl_embedgemma2_image_soft_token_count(model, image_dims, n_images, out, out_len, err, err_len)` | `[n, 2]` `u32` `(height, width)`; `out` `n` `u64`, `out_len` exactly `n` | the count of placeholders the image makes, from its size alone (`embedgemma2_pre::image_soft_tokens`) |
| `tessl_embedgemma2_audio_soft_token_count(model, sample_counts, n_clips, sampling_rate, out, out_len, err, err_len)` | `n` sample counts (1 to 2^28) at the extractor's rate | the tower's mask count for the frames the extractor would build, after the 30 s cut (`audio_soft_tokens`, 750 at most); a clip too short for a frame is refused by index |
| `tessl_embedgemma2_video_soft_token_count(model, video_dims, frame_counts, video_fps, n_videos, out, out_len, err, err_len)` | `[n, 2]` frame sizes, frames per video (1 to 2^20), a finite positive rate each; `out` `[n, 2]` `u64` | `(frames the sampler keeps, soft tokens per frame)` (`video_soft_tokens`); nothing is decoded |
| `tessl_embedgemma2_embed_segments(model, <texts>, <segments>, <images>, <clips>, <videos>, video_extension, video_extension_len, truncate_dim, out, out_len, err, err_len)` | see below | `MmItem::Segments`; one wait |

The counts work on a text-only handle as well (they read `processor_config.json`,
not a tower), so a prompt can be budgeted before any media is read. They are what
`tessl_embedgemma2_embed_media` needed and the ABI never said: the number of
placeholders per medium. A caller who still lays out ids of their own gets them
from here; one who does not uses `embed_segments`.

**Segments.** An item is a list of segments, each a text or a medium, and the
library lays out the processor's sequence for it: the chat template joins the
pieces with no separator, each medium becomes `<boi> <image> x n <eoi>` (or
`<boa> <audio> x n <eoa>`, or one `<boi> <video> x n <eoi>` block per sampled
frame with a `mm:ss` label before it under timestamps), `n` comes from the tower,
and the tokenizer wraps the whole in `<bos>`/`<eos>`. Added tokens split the
text, so the text between two blocks is one stretch tokenized on its own;
adjacent texts merge, and a timestamp label joins the text before it. The Rust
owner is `EmbedGemma2Config::mixed_sequence` over `LayoutSegment` (the facade's
`MmItem::Segments` calls it with the towers' counts); `image_sequence` and the
other sequence builders are unchanged. The layout is verified against the
processor's own output (below).

The C entry takes the texts (`text_bytes`, `text_offsets`, `n_texts`, as for
`embed_texts`, no prompt is added: write it in the first text), the segments
(`segment_kinds` `TESSL_EMBEDGEMMA2_SEGMENT_{TEXT,IMAGE,AUDIO,VIDEO,VIDEO_TIMESTAMPED}`
= 0 to 4, `segment_refs` indexing the call's texts or media by that kind,
`item_offsets` cutting at most 65,536 segments into items) and, for each modality,
a `*_source` (`TESSL_EMBEDGEMMA2_SOURCE_RAW` 0, `_PATHS`, `_BYTES`), a blob with
its byte cuts and, for raw media, the tables `embed_images`, `embed_audio` and
`embed_videos` take (`image_dims` `[n, 3]`, native-endian `f32` PCM at
`sampling_rate` with cuts a multiple of 4, `video_dims` `[n, 2]` and `video_fps`).
A text or a medium may be named by several segments (decoded and preprocessed
once, through its tower once per mention); one no segment names, and a reference
past a table, are refused by index. A modality with no media passes `n = 0`,
null pointers and 0.

**Bounds and errors.** Every count and cut is checked before anything is read: a
path is at most 4,096 bytes with no NUL, an image file at most 2^30 bytes, an
audio file 2^32, a video 2^32; the embedding entries take at most `MAX_BATCH` items
and the count entries at most `MAX_CALL_IMAGES` / `MAX_CALL_CLIPS`; the call's
pictures and patches, clips and frames are held to the per-call totals of
[Pictures and clips past a forward](#pictures-and-clips-past-a-forward) *before any
file is opened or frame decoded* (a video's first pass reports its sampled
frames). A failure names the file or bytes by index and the decoder's reason
(`file 2: cannot read ...`, `bytes 1: image: ...`, `segment 3 names image 5, but
the call has 2`), and the output is written only on success, in the three count
entries as in the embedding ones (a refused count call leaves its `out` alone).
Bytes of video are written to a private temporary file, removed before the call
returns on every path (the tests count the temporary files).

**From Python.**

```python
model = EmbedGemma2.from_pretrained(snapshot_dir, multimodal=True)
emb = model.embed_image_files(["cat.png", "dog.jpg"])         # or embed_image_bytes([data, ...])
emb = model.embed_audio_files("clip.wav")                      # or embed_audio_bytes
emb = model.embed_video_files("clip.mp4", timestamps=True)     # or embed_video_bytes(data, extension="mov")
model.image_soft_token_count([(480, 640)])                     # [n_placeholders]
model.audio_soft_token_count([16000]); model.video_soft_token_count((48, 64), 12, 10.0)   # [(frames, per_frame)]
emb = model.media([{"segments": ["a photo of", {"image": "cat.png"}, "on a table"]}])
```

A segment of `media()` is a `str` or a mapping with exactly one of `image`,
`audio` and `video` (and `timestamps` with a video); the medium is an array
(`embed_images`' and `embed_audio`'s; a video is `(frames, fps)`), a path or encoded
bytes. Within one call each modality uses one of the three, a mix is refused
naming the two segments that disagree, and the same object named twice is one
medium. `video_extension` is the type video bytes are sniffed as. The ids form
(`{"ids": ..., "images": ...}`) is unchanged; one call takes one form. The
Python side marshals and checks only what ctypes would wrap (types, ranges,
UTF-8); decoding failures arrive as the library's message.

**Parity.** On the real checkpoint (`tools/embedgemma2_ref/make_segments_reference.py`
renders each case through the snapshot's chat template and `EmbeddingGemma2Processor`,
mean-pools and normalizes, and checks its own route against sentence-transformers'
`encode` on the cases that route can take, to 1e-6):

| Case (message) | Ids vs the processor's | Embedding, max abs | Cosine |
|---|---|---|---|
| text, image, text | equal | 1.9e-7 | 1.00000000 |
| text, clip, text | equal | 1.6e-7 | 1.00000000 |
| text, video, text | equal | 2.7e-7 | 1.00000000 |
| image, text | equal | 1.5e-7 | 1.00000000 |
| text, image, text, clip, text, video | equal | 2.1e-7 | 1.00000000 |
| video, video | equal | 1.2e-7 | 1.00000000 |
| text, timestamped video, text | equal | 1.5e-7 | 1.00000000 |

(`tests/embedgemma2_segments_model.rs`, bounds max abs 1e-4 and cosine 0.99999; the
Python run of the same cases, with the image and clip once as files and once as
bytes, reports the same figures. Each case alone and the seven in one call agree
to the last digit shown.) The counts from sizes alone equal the processor's
placeholder counts (image 224x224, a 1 s clip, the 24-frame video).

**Tests.** `tests/embedgemma2_segments.rs` holds the layout to the tokenizer's
ids of the processor's whole string for 17 messages on a tiny tokenizer (a
mutation that flushes the text before a timestamp label fails two of them),
segments equal the same ids passed as a `Media` item bit for bit with one wait,
and the soft-count helpers equal preprocessing and the tower's mask across
boundary lengths. `tests/capi_embedgemma2_media.rs` pins the seven entries the
way `capi_embedgemma2_raw.rs` pins the raw ones: a null handle, a text-only handle,
every table refusal (over-limit lengths against tiny buffers, which would fault
if anything read first), a missing file, garbage bytes and a cut-short image named
by index, no temporary video file left, a foreign thread, relaxed precision, and
bit-for-bit equality with the Rust facade for both `operands`. The two Python
files are `python/tests/test_embedgemma2_media_host.py` (marshalling against a
stand-in) and `python/tests/test_embedgemma2_media.py` (tiny snapshot: a file, its
bytes and its array give the same row; opt-in, the real checkpoint against the
references above).

Regenerating: `EMBEDGEMMA2_SNAPSHOT=... ~/.venvs/ml-2026-10/bin/python
tools/embedgemma2_ref/make_segments_reference.py` writes
`target/embedgemma2_seg_ref/` (`EMBEDGEMMA2_SEG_REF_DIR` overrides it).

## Multimodal input

The vision and audio towers are other modules (`embedgemma2_vision`,
`embedgemma2_audio`); this is the merge they feed. Their outputs are 512-wide
"soft tokens" that replace the placeholder rows of the text sequence before the
text encoder runs, as transformers' `EmbeddingGemma2Model.forward` does: the
ids with every `image_token_id` (258880), `video_token_id` (258884) and
`audio_token_id` (258881) replaced by `pad_token_id` (0) are embedded through the
*scaled* table (`sqrt(hidden)`), then `masked_scatter` writes the soft tokens
over those rows, in order and **unscaled**; the text encoder, the per-layer
inputs included, runs on the result, followed by the same mean pool over every
token (soft ones too, `include_prompt`) and normalize.

```rust
let ids = model.config().image_sequence(soft.len() / hidden)?; // or audio_sequence
let out = model.encode_multimodal(&[MmSequence { ids: &ids, soft: &soft }], None, false)?;
```

* `MmSequence { ids, soft }`: `soft` is `[n_placeholders, hidden]` row-major f32
  and replaces, in order, the rows of `ids` whose id is any of the three
  placeholder ids. A batch may mix sequences with and without soft rows, and
  placeholders of any kind in any positions; the result is one
  `EncodeOutput` as `encode` returns it, with the same Matryoshka `truncate_dim`
  and `trace`.
* `image_sequence(n)` and `audio_sequence(n)` (on `EmbedGemma2Config`) build the
  processor's layout from the config's ids: `[<bos>, <boi>, <image> x n, <eoi>,
  <eos>]` and `[<bos>, <boa>, <audio> x n, <eoa>, <eos>]`. They equal the
  generator's `ids.npy` for all four reference inputs (checked in the real test).
* The config keeps `special_tokens`: `text_config.pad_token_id`, `bos_token_id`,
  `eos_token_id`, and the top-level `image_token_id`, `video_token_id`,
  `audio_token_id`, `boi_token_id`, `eoi_token_id`, `boa_token_id`,
  `eoa_token_index` (spelled so in the checkpoint). A text-only config still
  loads and `encode` still runs; `encode_multimodal` refuses one that lacks any,
  naming each missing key.
* Refused before any GPU work (no allocation, no wait): a missing id, a `soft`
  whose length is not a multiple of `hidden`, a non-finite soft value (with its
  row and column), a count of soft rows different from the count of placeholder
  ids, a `pad_token_id` outside the vocabulary, and everything `encode`
  refuses. A placeholder id is exempt from the vocabulary bound (the real ones
  are inside it; the tiny test puts them outside), since it is never looked up.
* `encode` is the special case with no soft rows and no placeholder ids: a text
  id that equals a placeholder id stays an ordinary token there, as before, and
  `encode_multimodal` with no soft rows is bit for bit `encode`. Batching,
  packing (longest first), the activation reuse, the single host wait and the
  async queueing are shared code, not a copy.

The soft rows have two sources and one path. A host caller's rows
(`MmSequence::soft`) are uploaded with the forward's ids (`alloc_buffer_from_u32`
on the f32 bits: no wait, and a forward with no soft rows allocates nothing
extra) as one device buffer; the multimodal facade's rows are the towers' own
output buffers, described as `SoftSegment { buffer, first row, rows }` and read
where the towers wrote them (next section). Either way each sequence's soft rows
are a list of segments in placeholder order, `soft_runs` pairs them with the
placeholder rows of the padded forward, and the rows land in `embeds` after the
scaled gather and before anything reads it: the residual stream, the bf16 copy
and the per-layer-input GEMM all see the merged rows, so the rest of the forward
is unchanged. The write is `qwen35_bwd::copy_cols` (one flat copy per maximal run
of consecutive placeholder rows whose soft rows are consecutive in one buffer, a
single row of the run's whole width): no new kernel. One image or clip is one
run; an adversarial layout of alternating placeholder and text tokens is one
dispatch per placeholder, at most `(rows + 1) / 2` of a forward's (32,768 padded
rows at most). The host path's finiteness check (a non-finite soft value is
refused with its row and column) applies to host rows only: device rows are not
read back before the text encoder runs, so the facade checks the embeddings it
returns instead (below).

Checked with the towers out of the loop (their golden soft tokens,
`tools/embedgemma2_ref/make_mm_reference.py`), `tests/embedgemma2_mm_model.rs`
(opt-in, `EMBEDGEMMA2_MM_REF_DIR`) and the f64 forward with an `inputs_embeds`
override on the tiny checkpoint, `tests/embedgemma2_mm_tiny.rs`. Bounds, fixed
before the first run, the text tests'; observed on an Apple M5 Pro, 2026-10-15, branch `worktree-agent-a403f81d87fc8f2cd` at `744d3b9` on base `ac5364a`, the status line's snapshot:

| Check | Bound | Observed |
|---|---|---|
| `lm_final`, ExactF32, 4 inputs (29 to 268 tokens) | within `1e-4` of the largest magnitude | 3.6e-7 to 1.3e-6 of it |
| embedding vs `emb_f32`, ExactF32 | max abs `1e-4`, cosine `>= 0.99999` | max abs 1.2e-7 to 2.1e-7, cosine 1.00000000 |
| the four in one call vs alone, both modes | `1e-5` | 0 (bit for bit; 2 forwards) |
| Bf16 vs fp32, per input | cosine `>= 0.9999` | 0.999995 to 0.999999 |
| Bf16 vs transformers bf16, per input | cosine `>= 0.9996` | 0.99975 to 0.99997 |
| Bf16 mean `1 - cos` vs fp32 | no worse than transformers' bf16 | 2.2e-6 against 8.6e-5 |
| tiny checkpoint vs f64 forward, ExactF32 | layers `1e-4`, embedding `1e-4` / `0.99999` | layers 2.2e-6 of scale at worst, embedding max abs 5.4e-7 |
| tiny checkpoint vs f64 forward, Bf16 | layers `5e-2`, embedding `1e-2` / `0.9999` | layers 1.2e-2 of scale at worst, embedding max abs 2.1e-3, cosine 0.999977 |
| tiny: one-forward batch vs alone | `1e-5` first, made bit-exact after the first run measured 0 | 0 |

Video is not in this table (the entry handles its placeholder id like the other
two; the video layout and its reference are in [Video](#video)). The towers'
output reaches this entry as host rows; [One call](#one-call-text-images-and-audio) does that composition.

The table above was observed against goldens written under eager attention;
the same test passes against the sdpa goldens `make_mm_reference.py` now
writes by default (see [Audio tower](#audio-tower) for why).

**End to end** (`tests/embedgemma2_e2e_model.rs`, opt-in; Apple M5 Pro,
2026-10-15, `eg2/integrate` after the audio merge `06465a9`, sdpa goldens):
the processor's pixels and log-mel features go through tessl's vision and audio
towers, and their soft tokens through `encode_multimodal`, all four inputs in
one call. The towers' soft-token counts equal the placeholders of `ids.npy`,
and the builders lay out exactly `ids.npy`. Against sentence-transformers'
`emb_f32` (bounds `1e-4` max abs and cosine `0.99999`, fixed before the run):
max abs 2.2e-7 to 3.8e-7, cosine 1.000000000 for both images and both clips.

## One call: text, images and audio

`tessl::embedgemma2_mm::EmbedGemma2Multimodal` owns the three models of one
snapshot and embeds a batch of mixed items in one call. It is a composition of
the existing owners, not a fourth implementation: the towers do the towers'
work, `EmbedGemma2Model::encode_multimodal` does the text encoder's, and the
facade does only what neither can see, the layout (how many placeholders an
item has, which soft row goes under which placeholder id, in what order across
modalities) and the waits. It sits beside `embedgemma2` rather than inside it
because that module stays text-only (the C ABI's text entries should not load
1.2 GB of towers) and is named as the tests are (`embedgemma2_mm_*`).

```rust
let m = EmbedGemma2Multimodal::load(&rt, &snapshot_dir, GemmOperands::ExactF32)?; // one config read, one open
let items = [
    MmItem::Text(&ids),                       // tokenizer ids, no placeholders
    MmItem::Image(ImageInput { pixel_values, position_ids }),   // the processor's tensors
    MmItem::Audio(&clip),                     // AudioInput { input_features, mask }
    MmItem::Media { ids: &my_ids, images: &[img], audio: &[clip] },  // caller ids with placeholders
];
let out = m.encode(&items, Some(256))?;       // EncodeOutput: one row per item, in order
```

* `load(rt, dir, operands)` / `load_from(rt, &SafeTensors, config_json, operands)`
  parse the text, vision and audio configs from the same `config.json` text,
  read `language_model.*`, the towers' tensors and refuse a config without the
  multimodal ids before reading weights; `from_parts(text, vision, audio)`
  composes loaded models and refuses a width mismatch between a tower's soft
  tokens and the text hidden size, or models of two runtimes.
* `MmItem::Image` / `Audio` lay out the processor's sequence
  (`image_sequence` / `audio_sequence` of the tower's soft-token count).
  `MmItem::Media` takes caller ids: the `j`-th image placeholder takes the
  `j`-th soft row of the item's images laid end to end, the audio placeholders
  likewise from its clips, which is `masked_scatter` per modality, so image and
  audio placeholders may interleave and a run need not line up with an image's
  end. The counts must match exactly (the video placeholders likewise take the
  frames of its `videos`, [Video](#video)); a `MmItem::Text` holding a placeholder id is refused (in a
  multimodal call it would be replaced by a soft token it does not have).
* `queue_images` / `queue_audio` on the towers (new) return a `PendingImages` /
  `PendingAudio` that holds each forward's output and waits for the queued work
  on drop; `collect()` reads the soft tokens. The activations and each forward's
  uploaded inputs are not held: they are handed to the runtime's deferred
  release before `queue_*` returns, so the wait recycles them, and a tower call
  followed by another holds one activation set
  (`consecutive_{vision,audio}_calls_hold_one_activation_set`; before, a call
  dropped them at `collect`, after its wait, and the pool recycled them at the
  next call's first commit, after that call had allocated). `encode_images` /
  `encode_audio` are `queue` + `synchronize` + `collect` and now queue under a
  `GpuRuntime::async_scope`, so they wait once whatever the runtime's
  `async_encode` flag says (before, a runtime in the default mode committed and
  waited per dispatch: with the vision scope removed, the by-hand composition of
  `tests/embedgemma2_multimodal_tiny.rs` made 29 waits instead of 3). `check_images` / `check_clips` are their pre-GPU
  checks as a soft-token count per item, and `EmbedGemma2Model::check_sequences`
  is `encode`'s id checks, so the facade refuses before any tower is queued.
* **Waits.** Every call waits for the GPU **once**, with or without media. The
  towers' forwards and then the text encoder's (however many) are queued under
  one scope behind one `synchronize`, and the towers' soft tokens never reach
  the host: the text encoder's scatter reads them from the towers' output
  buffers. (Until the GPU-resident hand-off a call with media waited twice: the
  towers' soft tokens were read back, concatenated per item on the host and
  uploaded again as host rows for the text encoder.) The by-hand composition
  waits three times (vision, audio, text). The sequences' layout is planned
  before the towers run, since `check_images` / `check_clips` return the
  placeholder counts; the soft rows themselves are described after the towers
  are queued (`PendingImages::into_device`, `PendingAudio::into_device`: the
  buffers, which the runtime keeps until the wait) and assigned to placeholders
  in id order, so image and audio placeholders interleave as `masked_scatter`
  per modality has them.
* **The towers' valid rows are not contiguous; the hand-off keeps them in place.**
  A vision image's soft tokens are one run of rows of its forward's output (the
  pooler writes only real cells, end to end); an audio clip's are the rows its
  tower mask keeps (`mask[::4]`), one run per maximal stretch of kept rows, which
  is one run for a processor's prefix mask and several for a mask with a hole.
  Compacting them into a dense buffer would add a kernel and a buffer to handle
  what `copy_cols` already takes, runs of rows, so the text encoder's scatter is
  given runs (`SoftSegment`) and nothing is compacted: a clip with a hole costs one
  more copy dispatch, not a pass over its rows. Consecutive rows of one buffer
  are one segment however the item's placeholders interleave with text.
* **Finiteness.** A host caller's soft rows are checked finite before any GPU
  work; the towers' are on the device and are not read back first. A media item's
  returned embedding is checked finite after the wait instead, and a non-finite
  one is an error naming the item (`a_tower_overflow_is_refused_after_the_wait_naming_the_item`:
  finite pixels that overflow the patch projection). The check cannot name the
  tower row or column.
* **Precision.** One `GemmOperands` (`ExactF32` or `Bf16`) configures the text
  encoder and both towers alike (`EmbedGemma2VisionModel::load_with`, [Vision
  tower](#vision-tower); `EmbedGemma2Audio::load_with`, [Audio tower bf16 and
  speed](#audio-tower-bf16-and-speed)): it is one flag because they make one
  precision claim to the caller, and a caller composing the models by hand
  (`from_parts`) can still mix them. The call refuses a runtime with relaxed f32
  GEMMs on when the mode is `ExactF32`; a `Bf16` call runs no f32 GEMM in any of
  the three models and accepts it.
* **Refusals** come before any allocation or wait: the batch size, each
  tower's input checks, the text checks, the placeholder counts. A refused call
  leaves the facade usable.

Tests and bounds, fixed before the first run. `tests/embedgemma2_multimodal_tiny.rs`
(tiny random checkpoint of `tests/common/embedgemma2_mm.rs`; the oracle is the
composition done by hand, since the pieces are held to f64 references in their
own tests): every item bit for bit the by-hand composition in both text modes
(observed: 0), a text item bit for bit `EmbedGemma2Model::encode` (0), a mixed
batch of text, images and clips bit for bit each item alone when the text side
packs one forward (0, one forward) and within `1e-5` across forwards (0
observed, 2 text forwards, both text modes), waits exactly 1 / 1 / 3 for text,
media and by hand (media was 2 until the GPU-resident hand-off, and the bound
changed because the behaviour did; the by-hand composition's bits are the
two-wait path's, so equality with it is the one-wait path equalling the two-wait
one, in both modes, for a clip whose mask has a hole and for caller-built ids
that interleave images and clips), every refusal `(allocations, waits) == (0, 0)`,
every unzeroed activation poisoned with NaN changing no bit, and consecutive
calls holding one activation set (`tests/embedgemma2_memory.rs`: calls 2 to 4 of
a tower, or of the facade, reach at most one fresh output above the first).
`tests/embedgemma2_multimodal_model.rs`
(real checkpoint, opt-in) and `python/tests/test_embedgemma2_mm.py`, Apple M5
Pro, 2026-10-15, sdpa goldens, snapshot `914f7f89`:

| Check | Bound | Observed |
|---|---|---|
| 8 texts (10 to 6147 tokens), 2 images, 2 clips in one call, vs `emb_f32` | max abs `1e-4`, cosine `>= 0.99999` | texts `1.2e-7` to `3.0e-7`, images `1.3e-7` / `2.1e-7`, clips `2.2e-7` / `2.5e-7`; cosine 1.000000000; 5 text forwards, 1 wait (27a0407: the same errors digit for digit, 2 waits) |
| each image and clip alone vs inside that call | `1e-5` | 0 (bit for bit) |
| media alone, mixed call, texts alone: waits | exact | 1; 1; 1 (media alone and mixed were 2) |
| text encoder and both towers `Bf16`, 4 media rows vs fp32 | cosine `>= 0.9999` | 0.999986 to 0.999993 (images 0.999988, 0.999988; clips 0.999993, 0.999986; the clips were 0.999995, 0.999994 with the audio tower exact) |
| Python, `encode_images` / `encode_audio`, alone and in one call | `1e-4`, `0.99999`; alone vs batch `1e-5` | `2.2e-7` to `3.8e-7`, cosine 1.000000000; alone vs batch 0 |
| Python, `embed_images` / `embed_audio` with transformers 5.19's processor on the generator's inputs | processor tensors vs saved `1e-6`, then the same embedding bounds | tensors identical (0), `3.8e-7` and `3.0e-7`, cosine 1.000000000 |

Follow-ups: (1) done: the GPU-resident soft-token hand-off (the towers' output
buffers read by the text encoder's `copy_cols`, no host round trip, one wait for
a mixed call; speed: `bench/results/embedgemma2_mm_onewait_m5pro.txt`). (2) A
`Media` entry in the C ABI and
Python for caller ids with placeholders; the C entries take pure images and
clips. (3) done: the tower tiny-checkpoint generators were in four test files
(`tests/common/embedgemma2_mm.rs`, `embedgemma2_vision_tiny.rs`,
`embedgemma2_audio_tiny.rs`, and the text one in `common`); the vision and audio
tower tests now draw their weights from `common::embedgemma2_mm`
(`vision_weights`, `audio_weights`: the stored values as f64 for the f64
references, and their safetensors entries), with the stored dtype a parameter
(`VisionStorage`). The four suites' output, `--nocapture`, is byte-identical to
before (same seeds, same values, same bounds).

## Kernels

| Kernel | File | New? |
|---|---|---|
| `encoder_attn_tiled_h{256_q32_k32_sg4, 512_q32_k32_sg8, 64_q32_k32_sg2, 32_q16_k32_sg1}` | `kernels/encoder_attn.metal` over `kernels/encoder_attn_tiled.h` and `kernels/attn_tiled.h` | new: bidirectional attention with a symmetric window and per-row lengths on the TensorOps matrix units, the FlashAttention-2 body Qwen3.5's prefill uses (`ENCODER` mode). What `encoder_attn` runs on a device with TensorOps, at every head dim (BERT's 32 and 64 and the vision tower's 64 included; the geometry per head dim is the sweep's winner, `bench/results/embedgemma2_attention_m5pro.txt`) |
| `encoder_attn_tiled_bf16_h{64_q32_k32_sg2, 256_q32_k64_sg4, 512_q32_k128_sg8}` | same | new: the same body with bf16 Q, K, V and probability tiles (scores, softmax, accumulator and output f32). `encoder_attn_bf16`, what the forward runs under `GemmOperands::Bf16` |
| `encoder_attn_rows_h{256,512,32,64}_*` | `kernels/encoder_attn.metal` over `kernels/attn_rows.h` | new: the same contract, one simdgroup per query row, scalar f32. Every head dim on a device without TensorOps (and the baseline the tiled kernels are timed against). `encoder_attn_with` names a kernel; every attention test runs on the rows and the f32 tiled kernels |
| `segment_mean_rows_f32`, `l2_normalize_rows_f32` | `kernels/embed_pool.metal` | new: row-range means (the pool here, the option pool in rsi-jev) and the (Matryoshka-prefix) normalize |
| q/k norm + full RoPE + K/V store, with the weightless V norm | `qwen35_attn_qk_norm_rope` via `qwen35::attn_qk_norm_rope_columns` (`QkvColumns::v_norm`) | reused; the host entry is new (separate K and V column blocks), and so is the V norm inside it. The frequencies are the host table `nn::rope_inv_freq` (torch's formula), not a device `pow` |
| embedding gather times `sqrt(hidden)` (bf16 table) | `qwen35_embed_rows_bf16_scaled` via `qwen35::embed_rows_scaled` | new: the gather and the scale in one pass |
| GELU-tanh gate over column windows (`gate \| up` from one GEMM; each layer's slice of the per-layer inputs) | `qwen35_gelu_tanh_glu_f32` via `qwen35::gated_act` | new: `qwen35_swiglu`'s kernel with the activation as a parameter |
| soft tokens over the placeholder rows (`encode_multimodal`) | `qwen35_copy_cols_f32` via `qwen35_bwd::copy_cols` | reused: a flat copy per run of consecutive rows |
| RMSNorm (layer 0's input norm), the exact-f32 feed-forward residual add | `nn::rms_norm_f32`, `nn::rms_norm_bf16`, `nn::rms_norm_residual_add_f32` | reused |
| a post-norm residual add with the layer scalar and the norm after it (pre-feed-forward, next layer's input, final); the feed-forward add with its bf16 copy; the per-layer inputs' scale and norm | `eg2t_resid_norm_f32`, `eg2t_resid_norm_bf16`, `eg2t_resid_add_bf16_copy`, `eg2t_scale_rms_norm_f32` (`kernels/embedgemma2_text.metal`) | new: "Text-encoder fusion (G7)" below |
| bf16-output attention | `encoder_attn_tiled_bf16o_h{64_q32_k32_sg2, 256_q32_k64_sg4, 512_q32_k128_sg8}` (`kernels/encoder_attn.metal`) | new: the bf16-operand kernel storing O as bf16 (`encoder_attn_bf16_out`) |
| GEMM | exact f32, or bf16 operands with f32 accumulation (`load_with`), TensorOps (NN, and NT for the per-layer-input projection) | reused |

The rows encoder attention is its own kernel rather than a mode of
`flash_attn_rows`: there the causal rule is the loop bound (keys above the
query are never visited) and the live length is one device scalar for every
row, and that body is the tuned prefill path of gemma-metal and Qwen3.5. The
per-key work (the dot, the online-softmax step, the store) is shared code,
`kernels/attn_rows.h`.

The per-layer inputs do not depend on the layer, so one NT GEMM forms a run
of layers' at once, straight from the checkpoint's `[24 * 512, 512]`
projection, followed by one scale and one in-place norm; each layer's gate
reads its column window. A run is as many layers as fit in `PLE_RUN_BYTES`
(16 MiB; it was 256): all 24 up to 341 rows, 4 per run at 2048 rows, one per
run from 8192 rows (never fewer than one layer's inputs, 64 MiB at 32,768 rows).

## Bounds and observed errors

The bounds were fixed before the first run. The observed errors come from the
run named in the status line.

Kernel tests, `tests/embedgemma2_kernels.rs` (fixtures from
`scripts/gen_embedgemma2_fixtures.py`; versions and generator commit in
`tests/fixtures/embedgemma2/provenance.json`). Regenerating with the current
generator rewrote all 58 earlier goldens byte-identical. It added the
Matryoshka prefixes `eg2_pool_out_trunc{512,256,128}`, the model's own RoPE
tables `eg2_inv_freq_{256,512}`, and a position-8000 case for the 512-wide
global layers (`eg2_qkv512_far_*`):

* the f64 references against transformers: within `1e-5` (attention) and
  `5e-6` (norm/RoPE near position 0, PLE, pool) of the largest magnitude;
* the kernels against the f64 references and the fixtures: within `2e-5` of
  the largest magnitude, at the fixture shapes and at the real ones (window
  512, head dims 256 and 512, sequences longer than the window, ragged
  batches); padding rows exactly `+0`;
* `nn::rope_inv_freq` against the model's rotary buffers: every entry within
  one ulp (they differ only where torch's own f32 `pow` is not correctly
  rounded: 1 of 128 pairs at d=256, 3 of 256 at d=512);
* q/k RoPE at position 8000, d=256 and d=512, against transformers: within
  `2e-5` plus `(pos0 + T) * max |ours - torch's|` over the table, derived
  from the fixtures, not tuned (~2e-7 at d=256, ~1.2e-4 at d=512). The
  device-`pow` kernel this replaced failed it at 3.6e-4;
* a batch row bit-identical to the same sequence run alone; two runs
  bit-identical.

Tiny-checkpoint test, `tests/embedgemma2_tiny.rs`. It runs in the plain GPU
suite with no checkpoint: random weights for a 3-layer config with K/V widths
512, 256 and 512, window 3, and a batch of four sequences of lengths 9, 4, 1
and 7. It is checked against an f64 host forward composed from the
references above (`tests/common/embedgemma2.rs`):

| Check | Bound | Observed |
|---|---|---|
| per-layer residual + final norm (9 tokens) | `1e-4` × max | ≤ `1.6e-6` × max |
| embeddings, alone and batched | `1e-4` max abs, cos ≥ `0.99999` | ≤ `3.8e-7`, cos 1.000000000 |
| batched vs alone | `1e-5` | 0 (bit-identical) |
| Matryoshka prefixes 64, 17, 1 | as embeddings | pass |
| device memory after the model drops | ≤ before loading | ≤ before (weights are `BufferKind::Hot`) |

Before the K/V fix, sequence 1 of the batch came out at cosine `0.27`
against the same sequence run alone. The loader also refuses an unread
tensor under the prefix (a `q_proj.bias`, a `norm.bias`); without that check,
the test fails with "loaded". With the weights allocated `Cold`, 2.2 MB stayed
allocated after the drop. The checkpoint is built in memory and opened with
`SafeTensors::from_bytes`.

Model test, `tests/embedgemma2_model.rs` (opt-in; needs the checkpoint and
`tools/embedgemma2_ref/make_reference.py`'s outputs, which record their own
`provenance.json`). The reference is the fp32 eager model:

| Check | Bound | Observed |
|---|---|---|
| text 0 (20 tokens): 24 layers + final norm | `1e-4` × max | ≤ `1.7e-6` × max |
| text 6 (1658 tokens): 24 layers + final norm | `1e-4` × max | `1.1e-6` to `4.9e-5` × max |
| all 8 embeddings (10 to 6147 tokens) | `1e-4` max abs, cos ≥ `0.99999` | ≤ `2.5e-7`, cos 1.00000000 |
| alone vs one ragged batch (≥ 2 forwards) | `1e-5` | 0 (bit-identical) |
| Matryoshka 512 / 256 / 128 vs sentence-transformers | as embeddings | ≤ `3.9e-7`, cos 1.00000000 |

For scale: the checkpoint's own bf16 forward moves the embeddings to cosine
`0.99990`–`0.99998` against the fp32 reference.

**Long positions and RoPE.** The first run had the RoPE kernel computing
`inv_freq` on the device (`precise::pow`). That is about an ulp off torch's
host-computed value at most pairs, and the angle `pos * inv_freq` multiplies
it by the position. The 1658-token trace came out `3.6e-5` relative at layer
0 and `1.8e-4` at layer 15, past the bound. Attention at that length was
within `1e-5` of f64, and transformers' own fp32 run was `2e-6` from its fp64
run, so the excess was tessl's. The RoPE kernels (this one, its backward and
`rms_qkv_rope`) now read the host table `nn::rope_inv_freq`, which matches
torch's formula. Layer 0 is now `1.9e-6`, at the reference's own fp32 noise,
and the 6147-token embedding moved from `9.2e-7` to `2.1e-7`.

## Speed

Measured on an M5 Pro against the real checkpoint, paired and interleaved
(`bench_paired` over `bench_embedgemma2`; `bench/results/embedgemma2_*_m5pro.*`).

* **The 2026-10-15 work as a whole** (60a6240 in its best mode, `BENCH_ASYNC=1`,
  against `eg2/integrate` at 7459e16, with a control lane at 0.995x to 1.002x;
  `embedgemma2_final_m5pro.{txt,json,png}`). ExactF32, the default: 1x16 0.60x
  of the time (lane G's opt-in split-K), every other workload unchanged within
  2 % (its f32 GEMMs are compute-bound on the exact-f32 matrix rate). Bf16:
  0.53x at 1x16 and 0.335x to 0.363x from 64x32 up (2.75x to 3.0x faster).
  Device peak above the model 3.0x to 3.9x lower in both modes (1652 to 466 MiB
  ExactF32 and 424 MiB Bf16 at 32x256 / 8x1024 / 2x4096). 60a6240's default
  mode, which waited on every dispatch, is not a lane: its 1x16 series failed
  the spread gate in a first attempt, and its cost is the G1 figure below.

* **Whole encode, before and after the nine landed changes** (2a7cb8b against
  0055612, combined, not per commit): an encode takes 0.35x to 0.84x of the
  time (1.2x to 2.9x faster) in the default runtime, uniform and ragged
  workloads alike. The loader's peak physical footprint fell from 1156 MB to
  871 MB (77d8732) and 845 MB (0055612). The device peak of the single-forward
  workloads rose 29 % (1283 to 1652 MiB at 8x1024; "The device-peak rise,
  isolated" below says why, and "fixed" that it is now 400 MiB). Numbers:
  `embedgemma2_before_after_m5pro.txt`, `embedgemma2_load_footprint_m5pro.txt`.
* **One host wait per `encode`.** The forwards of one `encode` are queued back
  to back and the host waits once; each keeps its own small ids, lengths,
  segments and output buffers (written with `alloc_buffer_from_u32`, which does
  not wait), and the large activations stay shared because the GPU orders
  them. `encode` opens a `GpuRuntime::async_scope`, so this holds on a
  default runtime: before it, nothing overlapped unless the caller had set
  `set_async_encode(true)`, and every other caller's dispatches each
  committed and waited. The scope is a counter beside the flag, so `encode`
  never stores or restores the caller's setting; the scope's other effects are
  on its API doc. With the flag on and `bench_embedgemma2` (`BENCH_ASYNC=1`),
  the figures measured before the scope were 1x16 in 7.4 ms against 84.9 ms
  synchronous, and the 128x8-512 ragged batch in 1903 ms against 2366 ms; the
  default runtime now takes the queued path (indicative, not re-measured here:
  the paired benchmark is the coordinator's).
  `tests/embedgemma2_tiny.rs` asserts one host wait and one command buffer for
  a default-runtime encode, and `tests/async_scope.rs` the scope itself.
* **What a forward costs, for the split.** `pack_forwards` trades padded rows
  against a fixed cost per forward, which the sweep
  (`embedgemma2_forward_cost_m5pro.json`) puts at `54.9 ms + 0.0525 ms/row`
  synchronous (1046 rows) and `3.42 ms + 0.0515 ms/row` queued (67 rows). It
  uses the queued one unless the runtime is serialized
  (`set_serialized_dispatch`, a reference mode in which every dispatch
  commits and waits; the tests use it as the oracle that queuing changes no
  bit).
* **Where a forward's time goes** (`profile_embedgemma2`,
  `embedgemma2_stage_profile_m5pro.txt`; a replica of `forward` that is bit-equal
  to `encode`, stages timed alone in async mode, min over three passes, about
  +-10 %). Exact-f32 GEMMs are 52-88 % of a forward (gate|up alone 14-32 %), and
  run at 75-98 % of the 7.4 TFLOP/s a 4096^3 f32 GEMM reaches, so they are
  compute-bound; the same shapes with bf16 operands run 3.8-4.6x faster at
  >= 2048 rows (1.9-2.2x at 16 rows). Attention is 3.5 % of a forward at 32
  tokens per sequence and 24 % / 43 % at 8x1024 / 2x4096 (the four global layers
  alone are 27 % at 2x4096), at the f32 matrix rate. The elementwise stages are
  10-18 % and already at the bandwidth roofline (190-285 GB/s), so only fusing
  them helps. At 16 rows the down, o and per-layer-input GEMMs reach 17-22 % of
  the copy bandwidth where gate|up reaches 84 %.
* **Attention on the matrix units at every head dim, and in bf16**
  (`bench_encoder_attn`, `embedgemma2_attention_m5pro.txt`: min of 24 rounds of
  interleaved A/B, a shared GPU). The vision tower's head dim 64 (12 heads, 2520
  patches) was the scalar rows kernel; the tiled kernel runs it 2.8x and 2.9x
  faster at 1 and 4 images (BERT's 32: about 2.3x). With bf16 tiles the kernel
  runs the text shapes 3.0 to 3.8x faster than the f32 tiled one (4.4 to 56 ms
  down to 1.5 to 15 ms, 17 to 19 TFLOP/s against 5 to 6) and the vision shapes
  2.6x. End to end, the bf16 forward (`bench_paired`, 6 rounds, base and new
  binaries interleaved) takes 171.6 ms against 198.0 at 32x256, 197.6 against
  301.2 at 8x1024 and 244.2 against 452.8 at 2x4096 (0.87x, 0.66x, 0.53x of the
  time); the exact-f32 forward is unchanged (0.995x, 0.991x, 0.997x, every
  range straddling 1).
* **Small-M GEMMs: an opt-in parallel K-partition path** (`gemm_small_m`,
  exact f32; lane G, `embedgemma2_small_m_m5pro.txt`). The four GEMMs that ran
  at 17-22 % of the copy bandwidth at 16 rows (down, o, the two per-layer-input
  ones) have a C of 16 tiles of 32x32: one dispatch gave them 16
  single-simdgroup threadgroups, each walking all of K, so their time was that
  walk's latency. A call that opts in, at N up to 512, K of 512 or more and
  under 64 tiles (M up to 96 at N 512), cuts K into about 128 threadgroups'
  worth of partitions, `matmul2d_tensorops_nn_splitk_par_f32`, into a scratch
  with one slice per partition and `reduce_partitions_f32` adding the slices in
  partition order: the TN lane's design, for the NN layout. **The opt-in is the
  call's**: `gemm`, `ExactF32.nn`, the vision and audio towers (through the
  shared `project`), Qwen3.5 and external consumers keep the route and the bits
  they had, and `Bf16` is untouched. Only the text encoder's own `project`
  opts in, with the forward's idle `qkv | gate_up` buffer as the scratch, so a
  forward adds no device memory (`tests/embedgemma2_small_m.rs` holds one
  forward's bytes to the itemized set exactly; the real checkpoint's device peak
  is unchanged, 2 MiB at 1x16). Per GEMM at 16 rows, interleaved against the
  single dispatch: down 3.0x (89.9 to 29.7 us, 64 % of the copy bandwidth), o
  2.4-3.0x, ple 1.8x. Whole `encode` (`bench_paired`, before = `27a0407`,
  a quiet machine, control 0.997): 1x16 0.60x of the time (1.66x faster), 1x64
  0.70x; 64x32 and 8x1024 take the same kernels as before (the gate returns
  none at 2048 and 8192 rows) and read 1.000x. The arithmetic stays exact f32:
  only the summation order changes, K in partitions summed in order, and the
  lane is held to the f32 dot-product bound `gamma_(K+8) * sum|a.b|` that every
  exact-f32 GEMM meets (worst result `0.0039` of it; `tests/gemm_small_m.rs`).
  This is less than the 1.8x the stage profile's bandwidth estimate allowed:
  the lane reaches about 64 % of the copy ceiling, not 70 %, and pays a reduce
  dispatch per GEMM. **The text encoder's bits now depend on M across the
  gate**: a text alone (up to 96 padded rows, the lane) and the same text in a
  larger batch (the single dispatch) sum K in different orders. Observed, all
  inside the existing bounds: alone against batched 0 to 1.7e-7 on the real
  checkpoint (`BATCH_ABS` 1e-5; bf16 exactly 0), 8.9e-8 to 1.3e-7 at the real
  GEMM shapes with synthetic weights (`tests/embedgemma2_small_m.rs`, bound
  fixed before the first run), a mixed multimodal call against alone 1.2e-7 to
  3.0e-7, the vision and audio towers bit-identical alone against batched. A
  bf16 form of the lane (1x16 0.79x) was built and removed: it moved the bf16
  text check by 2.4e-4 to 5.7e-4 (commit `b85f5e1`).
* **The device-peak rise, isolated.** One activation set is 736 MiB at 8192 rows
  against 576 MiB at 2a7cb8b (+160: the per-layer-input run buffer is +240, the
  packed gate|up and the dropped normed-V buffer save 80). The table above
  doubles it: `encode` drops its activations after its last wait, the pool
  recycles them at the next commit, and the next `encode` allocates first, so
  two sets are live (1472 MiB; 736 with an `rt.synchronize()` between calls). The
  bench's three rows at 32x256 / 8x1024 / 2x4096 are one number: the peak
  restarts from the freelist cache of the workloads before it
  (1652.4 = 436.3 cached + 2 x 608.1). `set_ple_run_bytes(16 MiB)` takes 240 MiB
  off each set at no measurable time (encode 32x256 +5 %, 8x1024 -2 %, inside the
  noise).
* **The device-peak rise, fixed** (`embedgemma2_memory_m5pro.{txt,json,png}`,
  against 16f5190, fresh process per cell, identical on a second run). Four
  `encode`s back to back peak at 400.6 / 400.4 / 400.3 MiB above the model at
  32x256 / 8x1024 / 2x4096 (1472.7 / 1472.4 / 1472.3 before; Bf16 1456.4 to
  392.4 at 8x1024), 112.9 MiB at 64x32 (433.0) and 2.0 MiB at 1x16 (4.25). One
  encode peaks at the same figures: consecutive calls hold one set. The three
  changes: `encode` hands its activations and uploads back to the runtime
  before its wait (the wait then recycles them; they were recycled one call
  late), `PLE_RUN_BYTES` is 16 MiB (-240 MiB), and the `q|k|v` GEMM output
  shares a buffer with `gate | up` (-96 MiB, as inferred). Every embedding hash
  is unchanged on the real checkpoint and the tiny model, in both modes. Time:
  the min-of-N ratio against 16f5190 is 0.86 to 1.05 over three paired series
  (the machine was shared and its round-to-round spread 1.2 to 28, so a 3 %
  bound is not verifiable there; the file says so), and the 16 MiB
  per-layer-input bound alone, measured as one binary at 16 against 256 MiB,
  is 0.96 to 1.03 (median of rounds) and 0.90 to 1.00 (min-of-N).
* **The towers, the same fix.** `queue_images` / `queue_audio` kept their
  activations and uploaded inputs until `collect`, after their wait, so a tower
  call followed by another had two sets live. They now hand them to the runtime
  before returning, keeping each forward's output only
  (`tests/embedgemma2_memory.rs`, tiny checkpoint, calls 2 to 4 above the
  first: vision 1,769,472 bytes against a bound of one 8,192-byte output before,
  within the bound after; audio 540,672 against 16,384, within the bound after).
  A facade call holds a tower's set and the text encoder's at once, as before
  (the towers' buffers cannot be recycled until the wait, which now follows the
  text encoder's queuing): `bench/results/embedgemma2_mm_onewait_m5pro.txt` has
  the device peak of the mixed call before and after.
* **`segment_mean_rows`** stays serial: it is under 0.2 % of an encode on
  every workload (`embedgemma2_segment_mean_share_m5pro.txt`).
* **No bf16 attention output** (until "Text-encoder fusion (G7)" below).
  `encoder_attn` writes f32 only, at bf16 operands too; the `out_bf16`
  argument had no caller and is gone. The bf16 forward's per-layer cast of the
  attention output is gone too: `encoder_attn_bf16_out` stores bf16 through
  the shared body's existing flag.

## Precision

`EmbedGemma2Model::load(..)` is the exact-f32 forward: bf16 weights widened
exactly, f32 activations, exact-f32 GEMMs (`GemmOperands::ExactF32`). Its
results are unchanged by the bf16 path below: the embeddings and the layer
trace of the tiny model and of the real checkpoint (8 texts, and text 0's
24-layer trace) hash identically before and after.

`EmbedGemma2Model::load_with(.., GemmOperands::Bf16)` is the bf16-operand
forward, chosen when the model is loaded (`GemmOperands` is the operand
precision `src/gemm.rs` already defines for callers that choose per call):

* **Weights.** Every projection stays as the checkpoint stores it, bf16, on the
  device, with no f32 copy; a checkpoint tensor stored in another dtype is
  refused by name rather than rounded. The embedding table is bf16 under both
  modes, so it does not change; the norms and the scalar stay f32. Device
  weight bytes (`EmbedGemma2Model::weight_bytes`, the real checkpoint):
  `ExactF32` 815,575,040 (777.8 MiB), `Bf16` 542,158,848 (517.0 MiB), 66 %.
  Each forward re-reads the projections at half the width.
* **GEMMs.** All eight are bf16 x bf16 -> f32 accumulation on the TensorOps
  lane (`GemmOperands::Bf16.nn` / `.nt`, the per-layer-input projection being
  the NT one). A GEMM taking an f32 A and a bf16 B does not exist (`gemm`
  requires matching operand dtypes), so A is staged in bf16, in buffers
  allocated once per `encode` like the other activations: the norms
  (`rms_norm_bf16`, `eg2t_resid_norm_bf16`), the feed-forward residual add
  (`eg2t_resid_add_bf16_copy`), the attention (`encoder_attn_bf16_out`) and the
  GELU gates (`qwen35_gelu_tanh_glu_bf16`) write bf16 directly, at no extra
  dispatch, and two cast passes (`cast_f32_to_bf16_into`) cover the rest: the
  embeddings and the pooled vectors once per forward. (Before the fusion of
  "Text-encoder fusion (G7)" the attention output and the residual stream into
  the per-layer-input gate were cast every layer too: four cast passes.)
* **Attention.** Q, K and V are stored bf16 by the q/k-norm-RoPE kernel
  (`attn_qk_norm_rope_columns(.., DType::BF16, ..)`, rounded to nearest even
  from its f32 result: no cast pass, and the three activation buffers are half
  the bytes), and `encoder_attn_bf16` runs the matrix units on bf16 tiles.
  The probability tile that multiplies V is rounded to bf16 too (kept f32, with
  bf16 Q, K and V, the kernel gained 1.1 to 1.8x on f32 where bf16 P gains 2.5
  to 3.8x: `embedgemma2_attention_m5pro.txt`), with the row's sum taken over
  the rounded weights, so the output is a convex combination of V's rows. Scores, the online softmax and its scale, the accumulator and the
  output are f32. The error budget is derived in `tests/embedgemma2_kernels.rs`
  (four first-order sources, 6 sigma per element).
* **What stays f32.** The residual stream, the norms' arithmetic, the q/k/v
  norms and RoPE (before their bf16 store), the attention's scores, softmax,
  accumulation and output, the pooling, the normalization, and every GEMM
  output.
* **Relaxed precision.** No f32 GEMM runs, so the runtime's relaxed-f32 switch
  does not touch the forward and `encode` does not refuse it (the exact-f32
  model still does); the embeddings are bit-identical with the switch on.
* **Needs** TensorOps (refused at load otherwise).

Bounds, fixed before the first run, derived in the header of
`tests/embedgemma2_tiny.rs` (bf16 rounding is a relative error of at most
`2^-8`, about `2^-8 / sqrt(3)` rms; a GEMM's error is that of its inputs, not
of K; three layers of three branches add in quadrature) and, for the real
checkpoint, in `tests/embedgemma2_model.rs` (held to transformers' bf16 forward
the way `tests/qwen35_model.rs` holds Qwen3.5's):

| Check | Bound | Observed |
|---|---|---|
| tiny: per-layer residual + final norm vs the f64 forward over the same bf16 weights | `5e-2` x max | `5.7e-3` to `1.3e-2` x max (`4.9e-3` to `8.4e-3` before the bf16 attention) |
| tiny: embeddings, alone and batched | `1e-2` max abs, cos >= `0.9999` | <= `2.0e-3`, cos >= `0.999976` (`1.4e-3`, `0.999987` before) |
| tiny: batched vs alone, ragged runs of per-layer inputs, sync vs async | `1e-5` | 0 (bit-identical) |
| real: cos vs fp32, each of 8 texts | >= `0.9999` | `0.99999337` to `0.99999808` (max abs <= `4.8e-4`; `0.99999324` to `0.99999848`, `4.3e-4` before) |
| real: mean `1 - cos` vs fp32, against transformers' bf16 forward | <= its | `4.9e-6` against `4.7e-5` (`4.1e-6` before) |
| real: cos vs transformers' bf16 (`emb_bf16.npy`) | >= `0.9996` (triangle inequality over two `1e-4` deficits) | `0.99989566` to `0.99997993` |
| real: alone vs one ragged batch (>= 2 forwards) | `1e-5` | 0 (bit-identical) |
| bf16 attention kernel vs the f64 reference of the exact inputs, 7 cases (D = 64, 256, 512; windows 1 to 512; padding, length 0, device lengths past T) | `6 sigma_x + 2e-5 max` per element, `sigma_x` derived from bf16's roundoff (header of `tests/embedgemma2_kernels.rs`) | worst element at `0.04` to `0.40` of its bound; max abs `6.2e-4` to `1.7e-2` (`5e-4` to `5.8e-3` of max) |
| bf16 attention kernel vs the f64 reference of the bf16-rounded inputs (P's rounding alone) | `6 sigma_p + 2e-5 max` | worst element at `0.29` to `0.47` of its bound |

transformers' bf16 forward rounds every activation, the residual stream
included, which is why it sits at cosine `0.99990` to `0.99998` from fp32
while this forward, which rounds only what feeds a GEMM, sits an order of
magnitude closer (`1 - cos` of `1.5e-6` to `6.8e-6`). Against `emb_bf16.npy` the two
are therefore about as far apart as `emb_bf16.npy` is from fp32.

## Vision tower

> **Status: run on the GPU against sentence-transformers** (Apple M5 Pro,
> 2026-10-15, worktree branch `worktree-agent-a00e8fdf52347d85d` on base
> `60a6240`, snapshot `914f7f89142e33e77833254d9c9b90c3cef7303b`; goldens from
> `tools/embedgemma2_ref/make_mm_reference.py`: fp32, eager attention, CPU).
> The real-checkpoint test passed again on `eg2/integrate` at the merge
> (`6eedd5a`).
>
> **Update (lane J, Apple M5 Pro, 2026-10-15, branch
> `worktree-agent-ad119ed6f8584407f` on `433e659`):** a `GemmOperands::Bf16` mode
> ([Precision](#vision-tower-precision)), the tower profiled stage by stage and
> sped up ([Speed](#vision-tower-speed)). `load` and `GemmOperands::ExactF32`
> are unchanged bit for bit (hashes below).

`tessl::embedgemma2_vision` (`src/embedgemma2_vision.rs`) is the 167M-parameter
vision tower (`vision_tower.*`, transformers 5.19's `Gemma4VisionModel`) and
`EmbeddingGemma2MultimodalEmbedder` (`embed_vision.*`). It maps the image
processor's `pixel_values [P, 768]` and `position_ids [P, 2]` to the 512-wide
soft tokens the text encoder substitutes for an image's placeholder tokens
(`image_token_id` 258880; the soft tokens are not scaled by `sqrt(hidden)`).
Merging them into a text sequence, and the processor (resize, patchify,
`P = 280 * 9 = 2520` with `(-1, -1)` padding), stay the caller's, as
tokenization does.

```text
let model = EmbedGemma2VisionModel::load(rt, &st, VisionConfig::from_config_file(..)?)?;
// or: EmbedGemma2VisionModel::load_with(rt, &st, cfg, GemmOperands::Bf16)?  (see Precision)
let out = model.encode_images(&[ImageInput { pixel_values, position_ids }, ..], trace)?;
// out.soft[i]: [S_i, 512] f32; out.tokens[i] = S_i; out.trace (per layer) when asked
```

**What it computes.** `x = 2 * (pixels - 0.5)` (the host, f32); `h = x @ Wproj +
tab[0][pos_x] + tab[1][pos_y]` (zero on padding; the table is `[2, 10240, 768]`
bf16); 16 layers with the text encoder's four `* w` norms, q/k RMSNorm with
weights, a weightless v norm, **2-D axial RoPE**, bidirectional attention at
scale 1.0 over 12 heads of 64 (MHA) and a GELU-tanh gated MLP; then a `3 x 3`
average pool by patch position times `sqrt(768)`; then a weightless RMSNorm and
`embedding_projection` (768 to 512). The axial RoPE splits each head's 64
channels into halves of 32: the first rotates by the patch's x, the second by
its y, each with `rotate_half` over its own 32 channels and `inv_freq[p] =
100^(-2p/32)`, `p < 16`; the table is `nn::rope_inv_freq(16, 32, 100)` on the
host (the text path's lesson: a device `pow` is an ulp off at most pairs).
`use_clipped_linears` and `standardize` are false in this checkpoint; the
config parser refuses a config that sets them (and `attention_bias`, a nonzero
dropout, a kv-head count that is not the head count, a pooling kernel under 2,
and any `rope_type` but `axial`).

**Padding and batching.** Padding patches are a suffix of `(-1, -1)` and never
reach a real one (keys are masked and the pooler gives them zero hidden state),
so the host strips them: only a suffix is accepted (anything else is refused),
and so is a real patch with one negative coordinate, which transformers would
treat as real and clamp to 0 (deliberately stricter than the reference). `P`
only sizes the pool's output (`P / k²` cells, `k² * (P / k²) == P`, as
`_avg_pool_by_positions` checks). Images of different real-patch counts run in
one forward padded to the longest, with a per-image length for attention
(`encoder_attn`, window 0, scale 1.0; its K/V layout `[batch, capacity, heads,
64]` is exactly what the new rope kernel writes). A call is cut into forwards
of at most 16,384 padded patches (`MAX_BATCH_PATCHES`) and 256 images,
longest first, queued under a `GpuRuntime::async_scope` behind one
`synchronize` (`queue_images` is the call up to the wait, for a caller that
queues more behind it: [One call](#one-call-text-images-and-audio)). A call may
hold more than one forward's worth of images: it is bounded by `MAX_CALL_IMAGES`
(4096) and by `MAX_CALL_PATCHES` (2^20 padded patches, the memory the forwards'
uploaded inputs and soft tokens hold until the wait: about 3.4 GiB at the bound under
`ExactF32`, 1.9 GiB under `Bf16`, beside one activation set), each refused by name
before any allocation.

**Pooling.** A patch at `(x, y)` goes to cell `x/k + (max_x/k) * (y/k)`,
`max_x = max(x) + 1` of its own image, which collides cells when the grid width
is not a multiple of `k` (transformers' behaviour, kept: the tiny test uses an
odd width on purpose). The output has one row per occupied cell in cell order,
plus cell 0 when the image has padding (transformers clamps a padding position
to `(0, 0)`, so cell 0 always counts it; an empty cell 0 yields exactly `+0`
and a soft token of `+0`). A cell at or past `P / k²` is an error in
transformers (`one_hot`) and here. `plan_pooling` is that mapping on the host;
the kernel sums each cell's rows in patch order.

| Kernel | File | New? |
|---|---|---|
| `vision_qkv_norm_rope_f32`, `vision_qkv_norm_rope_bf16` | `kernels/embedgemma2_vision.metal` | new: q/k RMSNorm (`* w`), the two-axis RoPE, the weightless v norm and the K/V store. One simdgroup takes up to six heads of one kind of one patch (`ROPE_HEADS_PER_UNIT`) and forms the patch's `cos`/`sin` table once for all of them (the angle depends on the patch and the pair, not the head; `precise::cos`/`sin` dominated the old per-head kernel). The bf16 instantiation is the same body storing bf16, rounded to nearest even from the f32 result (what a cast pass would do, with no pass) for `encoder_attn_bf16`; head dims up to 512 (`MAX_ROPE_HEAD_DIM`). Its own kernel, not a mode of `qwen35_attn_qk_norm_rope`: that one rotates a head by one position per token and its row routine is mirrored by the training backward, whose inverse rotation must stay its exact transpose |
| `vision_pos_embed_add_f32` | same | new: `proj + (x_emb + y_emb)` from the bf16 table, exactly `+0` on padding. The `+0` guarantee lives here and at the pool's output; the rope kernel does not special-case padding (a zero row rotated by -1 may come out `-0.0`), and its padded q/k/v slots are masked or overwritten by `encoder_attn` |
| `vision_pool_members_f32` | same | new: the pooler's cell sums, weighted `1/k²`, times `sqrt(hidden)` |
| weightless RMSNorm | `nn::rms_norm_f32` with a hot `[768]` buffer of ones | reused (the multiply by `1.0` is exact) |
| attention | `embedgemma2::encoder_attn`, tiled kernel `encoder_attn_tiled_h64_q32_k32_sg2` on TensorOps devices (rows kernel `encoder_attn_rows_h64_r4_g16` otherwise); under `Bf16`, `encoder_attn_bf16` (head dim 64, TensorOps only) | reused |
| RMSNorm, residual add, GELU-tanh gate, GEMM | `nn::rms_norm_f32` / `rms_norm_bf16`, `nn::rms_norm_residual_add_f32`, `qwen35::gated_act` (dtype of the stored mid), exact-f32 TensorOps GEMM, or under `Bf16` the bf16 x bf16 GEMM (`embedgemma2::project`, shared with the text encoder) | reused |

**Bounds and observed errors.** Fixed before the first run; the tiny and
kernel tests run in the plain GPU suite, the model test is opt-in.

Kernel tests, `tests/embedgemma2_vision_kernels.rs`, against f64 references
(`tests/common/embedgemma2_vision.rs`), at fixture shapes and the real ones
(2376 patches in a 66 x 36 grid plus 144 padding, 12 heads of 64, a 10240-row
table, 768 columns):

| Check | Bound | Observed |
|---|---|---|
| axial RoPE + norms + K/V store (`q`, `k`) | `2e-5` × max, plus `max_pos * max_p \|inv_freq_f32 - f64\|` | ≤ `2.2e-6` × max |
| weightless v norm | `2e-5` × max | ≤ `1.3e-7` × max |
| position-embedding add, 2520 x 768 | `1e-6` × max | ≤ `3e-8` × max; padding rows exactly `+0` |
| pool, 2520 rows to 256 / 264 cells x 768 | `2e-5` × max | ≤ `1.5e-7` × max |
| `plan_pooling` vs the dense one-hot reference | equal cells and sums | equal (and equal where it raises) |

Tiny-checkpoint test, `tests/embedgemma2_vision_tiny.rs` (random weights, 2
layers of 3 heads of 32, hidden 96, patch 4, 2x2 pooling; four images of 48, 20,
1 and 4 real patches, one with an odd grid width, against the f64 forward):

| Check | Bound | Observed |
|---|---|---|
| patch embedder, each layer, pooled output | `1e-4` × max | ≤ `7.3e-7` × max |
| soft tokens | `1e-4` max abs | ≤ `7.9e-7` |
| batched vs alone, reversed order | bit-identical | bit-identical |
| every unzeroed activation poisoned with NaN | bit-identical | bit-identical |
| a call over `MAX_BATCH_PATCHES` (2 forwards), async, poisoned | each image bit-identical to alone | bit-identical |

Model test, `tests/embedgemma2_vision_model.rs` (opt-in: `EMBEDGEMMA2_SNAPSHOT`
and `EMBEDGEMMA2_MM_REF_DIR`). Image 0 is 224 x 224 (2304 real patches, 256 soft
tokens), image 1 is 640 x 360 (2376, 264); each fp32 golden is run alone:

| Check | Bound | Observed |
|---|---|---|
| patch embedder (real rows) | `1e-4` × max | `0` |
| layers 0 to 15 (real rows) | `1e-4` × max | `4e-7` to `1.4e-5` × max (image 0), `4e-7` to `2.9e-5` (image 1); grows with depth, largest at layer 15 |
| pooled (after `* sqrt(768)`) | `1e-4` × max | `3.9e-6` (image 0), `6.9e-6` (image 1) |
| soft tokens | `1e-4` × max | `1.2e-5` (image 0), `3.3e-5` (image 1) × max |
| both in one forward vs each alone | bit-identical | bit-identical |

The residual stream reaches magnitudes near 1000 by layer 15 (the checkpoint's
norms do not bound it), so the layer-15 absolute error of `3e-2` is the same
`3e-5` relative. Nothing needed tuning: the bounds held on the first run.

### Vision tower precision

`EmbedGemma2VisionModel::load_with(rt, st, cfg, operands)` takes the text
encoder's `GemmOperands`; `load` is `load_with(.., ExactF32)`. One flag covers
the text encoder and both towers of the facade
(`EmbedGemma2Multimodal::load(.., operands)`, `tessl_embedgemma2_load_multimodal`'s
`operands`, `EmbedGemma2.from_pretrained(.., multimodal=True, operands="bf16")`).
The audio tower follows it too since its bf16 mode ([Audio tower bf16 and
speed](#audio-tower-bf16-and-speed)).

**`ExactF32` is bit for bit what it was.** Checked by hashing the output bits
(FNV-1a over the soft tokens) before and after on the same inputs: the tiny
fixture (`tests/embedgemma2_vision_tiny.rs`, hash `a1e38036581d2780` before and
after) and the real checkpoint (`bench_embedgemma2_vision`, `HASH`: image 0 alone
`d9a0d85dd2f8f279`, the 8-image call `4f48fd6b310f86a5`, the same from the build
at `433e659` and from this branch). The rope kernel changed (one `cos`/`sin`
table per patch for up to six heads); it writes the two products of each output
as `fma(x0, c, -(x1 * s))` and `fma(x0, s, x1 * c)`, the contraction the
compiler had chosen for the plain expression, so the bits did not move. (The
form matters: with the plain expression the compiler fused the chunked kernel
differently, `fma(x1, c, x0 * s)` for the second product, and the tiny hash
moved to `c033b6dfa23990f9`; `fma(-x1, s, x0 * c)` with `fma(x0, s, x1 * c)`
gave a third hash, `addc2d9b50726a23`. All three pass the tolerance tests: only
the hash sees it.)

**What `Bf16` does.**

| | `ExactF32` | `Bf16` |
|---|---|---|
| the 16 layers' q/k/v/o/gate/up/down weights | f32 on the device (widened from the checkpoint's bf16 at load) | bf16, the checkpoint's own bytes (one copy) |
| patch embedding, `embedding_projection` | f32 GEMM | bf16 x bf16 GEMM, f32 accumulate |
| pixels `2 * (p - 0.5)` | f32 | f32 on the host, rounded to bf16 on the host with the GPU's rounding (the GEMM input) |
| residual stream, norms (`rms_norm`), the pos-embed add, the pool | f32 | f32; the norms read f32 and write the bf16 GEMM operand |
| q/k RMSNorm, axial RoPE, v norm | f32, `vision_qkv_norm_rope_f32` | f32 arithmetic, stored bf16 (`vision_qkv_norm_rope_bf16`) |
| attention | `encoder_attn` (f32 tiles) | `encoder_attn_bf16` at head dim 64: bf16 q/k/v tiles, f32 scores, softmax and output |
| attention output into the `o` GEMM | f32 | cast to bf16 (one pass) |
| MLP gate and up GEMM, GELU-tanh product | f32 | bf16 GEMM (f32 out); the gated product is stored bf16 (the `down` GEMM operand) |
| soft tokens | f32 | f32 |

Resident weights: 335,621,120 B (320.1 MiB) against 639,577,088 B (609.9 MiB),
52.5% (`weight_bytes()`, measured by `bench_embedgemma2_vision`'s `weight_mib`).
The peaks fall by less (the activations are bf16 only where they feed a GEMM or
the attention): process footprint 1029 to 685 MiB at one image, 2996 to 2329 MiB
at eight; device peak 302 to 254 and 2192 to 1820 MiB (`bench_embedgemma2_vision`).

`load_with(.., Bf16)` refuses what it cannot run, naming it: a device without
TensorOps, a head dim `encoder_attn_bf16` is not compiled for (only 64), and a checkpoint with any projection
stored in another dtype than bf16 (an f32 checkpoint is not rounded at load).
The rope kernel entry points (`axial_qkv_norm_rope`, `_bf16`) now refuse a head dim over 512 in either mode (its table holds 8 pairs per lane; before they took any multiple of 4). No config loads differently: the config parser only admits head dims 32, 64, 256 and 512, so `load` accepts what it accepted, and "bit for bit" holds for every config that loads.

`ExactF32` keeps refusing relaxed-f32 GEMM selections (`TESSL_*`); `Bf16`'s
GEMM is the one the text encoder runs, so the exactness check does not apply.

**Bounds, fixed before the first bf16 run.**

Tiny test (`tests/embedgemma2_vision_tiny.rs`, 2 layers, 2 heads of 64, every
projection stored bf16, four images of 48, 20, 1 and 4 real patches), against
the f64 forward over exactly those weights, so that only the activations round.
Derived in the test's module docs: one bf16 rounding is a relative error of at
most `2^-8` (3.9e-3) and the stream passes a few dozen of them.

| Check | Bound | Observed |
|---|---|---|
| patch embedder, max abs over max | `2e-2` | `9.5e-4` |
| each layer, pooled output, max abs over max | `5e-2` | layers up to `6.65e-3`, pooled `4.73e-3` |
| soft tokens, max abs over max | `5e-2` | `5.7e-3` |
| soft tokens, worst cosine over all tokens | `>= 0.999` | `0.99998285` |
| bf16 soft tokens vs the `ExactF32` ones (same weights) | `5e-2` | `3.94e-3` |
| batched vs alone, reversed order, a call split into two forwards, every unzeroed activation poisoned with NaN | bit-identical | bit-identical |
| `ExactF32` on the same fixture | the exact bounds above (layers `1e-4`, soft `1e-4` abs) | unchanged |

Real checkpoint (opt-in, `EMBEDGEMMA2_SNAPSHOT` + `EMBEDGEMMA2_MM_REF_DIR`),
`tests/embedgemma2_vision_model.rs`; the goldens are fp32 sentence-transformers:

| Check | Bound | Observed |
|---|---|---|
| image embedding end to end (bf16 vision tower + bf16 text encoder, through the facade) vs `image{0,1}_emb_f32.npy`, cosine | `>= 0.9999` | `0.99998769`, `0.99998850` (max abs `6.4e-4`, `6.6e-4`) |
| the same embeddings vs transformers' bf16 run (`image{0,1}_emb_bf16.npy`), cosine | `>= 0.9996` | `0.99997923`, `0.99997317` |
| mean `1 - cos` vs fp32, against transformers' own bf16 run (`image{0,1}_emb_bf16.npy`) | `<=` transformers' | `1.191e-5` against `2.612e-5` (transformers bf16 vs fp32: `0.99997775`, `0.99997001`) |
| the tower: each image alone vs both in one forward, vs reversed order | bit-identical | bit-identical |
| the embeddings: each image alone vs inside one call of both | `1e-5` max abs (the text encoder's batch bound) | bit-identical |
| the tower alone, bf16 vs the fp32 goldens, max abs over max (reported, not gated) | - | patch embedder `1.4e-3`, `1.6e-3`; layers `1.3e-3` to `8.8e-3` up to layer 10, growing to `7.9e-2` and `7.4e-2` at layer 15; pooled `3.2e-2`, `2.8e-2`; soft tokens `5.9e-2`, `8.0e-2` |

The tower's own error against the fp32 goldens is large in max-abs terms (the
residual stream's few outlier channels reach 1000 and one bf16 rounding there is
4) and small where it counts: the embedding that comes out of the text encoder
is closer to the fp32 reference than transformers' own bf16 run is. The tower
bounds are not gated on those max-abs figures for that reason; the gate is the
embedding.

### Vision tower speed

Measured on an Apple M5 Pro, 2026-10-15, on the real checkpoint, on a GPU other
sessions shared (so every figure carries its spread). Tools:
`profile_embedgemma2_vision` (a bit-checked replica of the forward, each stage
alone on the real shapes and weights, `PROFILE_OPERANDS=f32|bf16`) and
`bench_embedgemma2_vision` under `bench_paired` (interleaved fresh processes,
order reversed on alternate rounds, min-of-20 per process, 8 rounds). Data,
plot and method: `bench/results/embedgemma2_vision_{profile,speed}_m5pro.*`.

**Stage ranking** (one image, 2304 real patches padded to 2520, each stage's
share of the sum of the stages run alone, the 16 layers together; one pass per
label):

| Stage | before, `ExactF32` | after, `ExactF32` | after, `Bf16` |
|---|---|---|---|
| gate and up GEMM | 59.7 ms, 31% | 58.2 ms, 33% | 13.4 ms, 23% |
| attention (head dim 64) | 49.9 ms, 26% | 47.4 ms, 27% | 17.6 ms, 30% |
| down GEMM | 28.0 ms, 15% | 27.1 ms, 15% | 7.1 ms, 12% |
| qkv GEMM | 22.2 ms, 12% | 21.3 ms, 12% | 5.0 ms, 9% |
| qk-norm, axial rope, K/V store | 8.4 ms, 4.4% | 3.1 ms, 1.8% | 2.6 ms, 4.4% |
| o GEMM | 8.1 ms | 7.6 ms | 1.9 ms |
| gelu * up | 6.3 ms | 6.0 ms | 4.6 ms, 8.0% |
| norm + residual, rms_norm | 5.2 ms | 4.4 ms | 4.1 ms |
| attention output cast f32 to bf16 | - | - | 0.8 ms |
| patch embed, pos-embed, pool, embedder | 3.2 ms | 1.8 ms | 0.9 ms |
| full forward (the replica, one wait) | 183.1 ms | 166.6 ms | 57.5 ms |

The f32 GEMMs run at 5.7 to 6.5 TFLOP/s, the bf16 GEMMs at 22 to 26 (the
text encoder's own bf16 GEMM, shared through `embedgemma2::project`). At 8
images (two forwards, 18,864 padded rows) the ranking is the same: attention and
the GEMMs are 91% of the exact forward's stage time and attention is the largest stage under
`Bf16` (161 of 502 ms).

**What was fixed, and why only that.** The rope kernel was the one stage where
the ranking and the arithmetic agreed that work was owed: it formed `precise::cos`
and `precise::sin` once per (patch, head), 24 times for the same patch and pair
(12 q, 12 k heads; v takes none), 4.4% of the exact forward and, in a first bf16
profile taken while the GPU was contended, 11% of the bf16 one. A simdgroup now
takes up to six heads of one kind and forms the table once (`ROPE_HEADS_PER_UNIT`;
head dims up to `MAX_ROPE_HEAD_DIM = 512`, at most 8 pairs per lane): 8.4 to 3.1 ms at one image
and 85.9 to 24.8 ms at 8 (single passes at different moments; the paired
end-to-end ratio below agrees but is at the limit of what that run resolves).
Output bits are unchanged (above). The GEMMs, the attention and the residual
passes were not touched: the GEMMs are lane G's (`src/gemm*.rs`), the attention
kernel is shared with the text encoder, and the bf16 operand mode is itself the
fix for the top four stages.

**End to end** (`encode_images`, host pixels to host soft tokens; median over 8
paired rounds of the per-process min-of-20, `vs` is the median per-round ratio
with the lowest and highest round):

| Workload | before (`433e659`) | `ExactF32`, after | `Bf16` | `Bf16` vs `ExactF32` |
|---|---|---|---|---|
| 1 image | 261 ms | 187 ms | 63.5 ms | 0.353x [0.184, 0.783] |
| 8 images | 1693 ms | 1683 ms | 547 ms | 0.325x [0.260, 0.366] |

Several of the 8 rounds were contended (the 1-image before lane reads 340 to
407 ms in four rounds and 163 to 182 in the other four), which is what the wide
1-image spread is. On the four clean 1-image rounds: 173.6 ms before, 169.3 ms
`ExactF32` after (0.974x, a 3% gain that is the rope's), 58.2 ms `Bf16` (0.349x
of `ExactF32`, 2.9x faster, range 0.342 to 0.355); the 8-image `Bf16` ratio is
0.354 over the same rounds (0.348 to 0.366). The `ExactF32` before and after
8-image spread [0.849, 1.286] crosses 1.0: that comparison alone is not a
result. Measured by the same bench: process peak footprint 1029 to 685 MiB at
one image and 2996 to 2329 MiB at eight, device peak 302 to 254 MiB and 2192 to
1820 MiB (`ExactF32` to `Bf16`); resident weights 609.9 to 320.1 MiB.

**Where the rest of the time is** (not done, with the evidence): under `Bf16`
the bf16 attention (`encoder_attn_bf16`, 13.5 to 14.9 TFLOP/s against the GEMMs'
25 to 26) is the largest stage at 30%, and it is the text encoder's kernel; the
per-layer attention-output cast (1.4%) goes away if `encoder_attn` stores bf16
(an `out_bf16` mode); the three f32 elementwise passes around each GEMM (norms,
the residual add, `gelu * up`) and the cast are 17% of the bf16 forward at 177 to
309 GB/s.
The pos-embed add, the pool and the embedder are under 1% each. Images are still
padded to the longest of a forward and the greedy longest-first cut does not
weigh a per-forward fixed cost as the text packer does.


## Audio tower

`tessl::embedgemma2_audio` runs the checkpoint's audio tower (`audio_tower.*`,
305M parameters: a conformer of 12 layers, hidden 1024, 8 heads of 128) and
its multimodal embedder (`embed_audio.*`) on the same kernels' conventions:
bf16 weights widened exactly, f32 activations, exact-f32 GEMMs, a strict
loader that refuses an unread tensor under either prefix. Preprocessing (the
128-bin log-mel front end) is the caller's: the input is the processor's
`input_features [frames, 128]` and `input_features_mask [frames]`.

```rust
let cfg = AudioConfig::from_config_file(&snapshot.join("config.json"))?;
let model = EmbedGemma2Audio::load(&rt, &safetensors, "", cfg)?;
let out = model.encode_audio(&[AudioInput { input_features, mask }, ...], false)?;
// out.soft_tokens[i]: [S_i, 512], the rows the text encoder takes in place of
// clip i's audio placeholders; out.out_masks[i]: which tower rows were kept.
```

> **Status** (Apple M5 Pro, 2026-10-15, branch
> `worktree-agent-a975c42e7f8ed0908` on base `60a6240`, then lane I's kernels
> ("Speed"; `worktree-agent-a52c15ece0674d300` on `df5d329`), snapshot
> `914f7f89`; torch 2.14.1, transformers 5.19.0): every bound below holds.
> `encode_audio` queues its forwards under a `GpuRuntime::async_scope` and ends
> with one `synchronize()` (`queue_audio` is the call up to the wait).

### What it computes

`Gemma4AudioModel.forward` then `EmbeddingGemma2MultimodalEmbedder.forward`
(transformers 5.19), written out in the module docs of
`src/embedgemma2_audio.rs`:

* **Subsampling.** The frames are multiplied by the frame mask, then two
  `Conv2d(3x3, stride 2, pad 1, no bias)` (1 -> 128 -> 32 channels), each
  followed by a channel `LayerNorm` (weight, no bias, eps 1e-6) and ReLU; the
  second conv's input is masked by `mask[::2]`. The rows are flattened
  `(frequency, channel)` and projected 1024 -> 1024. `T = ceil(ceil(F/2)/2)`
  rows, and the tower's mask is `mask[::4]` (99 frames: 25 tokens; 330 frames
  with 329 valid: 83).
* **Layer.** Half-weighted feed-forward, chunked local attention, the light
  conv (pre-norm, linear 1024 -> 2048, GLU, depthwise causal conv with 5 taps,
  norm, SiLU, linear), a second feed-forward, a final norm. Every linear is a
  `Gemma4ClippableLinear`: its input and output are clamped to four scalar bf16
  bounds from the checkpoint, and the residual stream is clamped to
  `+-gradient_clipping` (1e10) around the norms.
* **Attention.** Queries come in blocks of 12; each block sees a window of 24
  keys; a query attends to itself and the 11 frames before it (the mask is
  `0 <= q - k < attention_context_left - 1 = 12`; a query sees 12 keys, not 13
  and none ahead), restricted to valid frames. Logits are Transformer-XL:
  `(q * q_scale * softplus(per_dim_scale)) . (k * k_scale)` plus the same query
  against `relative_k_proj(sinusoidal table)` shifted by `_rel_shift`, a tanh
  softcap of 50, masked logits at -1e9, softmax in f32. `q_scale = head_dim^-0.5
  / ln 2`, `k_scale = ln(1 + e) / ln 2`. The relative-position table and the
  query scale are computed once per layer at load (f64, rounded once).
  `rel_shift_source` is the shift's index math in one function: output `(i, j)`
  of a block reads column `j - i` of row `i` when `0 <= j - i < 13`, else zero,
  so a visible key at distance `d` reads table row `12 - d` (position `d`) and
  row 0 is never visible.
* **Output.** `output_proj` 1024 -> 1536 with bias is the tower output; the
  embedder is a weightless RMSNorm over 1536 and `embedding_projection`
  1536 -> 512. Soft tokens are kept where the tower's mask is true.

### The attention mask: transformers' eager path inverts it

`Gemma4AudioAttention.forward` fills masked logits with
`attention_mask.logical_not()`, which expects a bool mask. Under
`attn_implementation="sdpa"` it gets one. Under `"eager"` (what
`tools/embedgemma2_ref/make_mm_reference.py` ran before it defaulted to sdpa)
`create_bidirectional_mask` returns a float additive mask (0.0 visible, `finfo.min` not), and
`logical_not()` of a float is true exactly where it is 0.0: the visible keys are
filled with -1e9 and the keys outside the window, and the zero padding the
blocked layout adds, are attended to. The attention is hand-written matmuls in
both, so the setting selects only the mask's dtype, and the sdpa behaviour is
the specified one. tessl implements that.

Consequence for references: under eager, `make_mm_reference.py`'s audio files
(`layer*.npy`, `tower_out.npy`, `soft.npy`, and the audio `emb_*`) describe the
inverted mask; its `subsample.npy` does not touch attention and is right. The
generator therefore runs sdpa by default, which is also sentence-transformers'
default; `EMBEDGEMMA2_MM_ATTN=eager` writes the eager set (the text and vision
goldens change only by rounding between the two). Measured, tessl against the eager references of the first clip:
subsample `3.1e-7` of max (agrees), layer 0 `2.8e-2` of max, layer 3 `1.2e-1`,
layer 10 `4.8e-1`, tower output `6.8e-1`
(`eager_references_encode_the_inverted_mask`, opt-in). The fixture
`eg2a_audio_c1_eager_tower` pins the same difference device-free (`5.4e-1`).

### Kernels

New, `kernels/embedgemma2_audio.metal`; the matmuls are the exact-f32 `gemm`.
`gemm_epilogue` has no exact-f32 path (its contract needs bf16 or relaxed f32),
so no clamp rides on a GEMM; each is folded into the row pass beside it. A layer
is 17 dispatches, 19 when q, k and v need different input clamps (it was 28; see
"Speed").

| Kernel | Does |
|---|---|
| `eg2a_sub_conv0_ln_relu` | the strided conv with the frame mask, the channel LayerNorm and ReLU in one pass; threadgroup per output position (channels across threads). Output channels-last, so the second layer's flatten order is its store order |
| `eg2a_sub_conv1_ln_relu` | the second strided conv, LayerNorm and ReLU: one simdgroup per (row, 8 frequency positions); each of the nine taps stages its 8-position input patch in threadgroup memory (zero where the tap leaves the frequency range, the rows or the mask) and multiplies it by that tap's `[C0, C1]` weights in 8x8 `simdgroup_float8x8` tiles, ragged edge tiles through a zero-padded stage; the LayerNorm statistics are `simd_sum`s. The per-position kernel it replaced re-read all 147 KB of weights for every output position |
| `eg2a_rms_norm` | one or two row stages per threadgroup, one row each. A stage is `[resid + \| bias +] scale * clamp(act(rms_norm(clamp(clamp(x))) [* w]))` (a linear's output clamp and the gradient clamp before, SiLU and the next linear's input clamp after; the norm and the weight are each optional: without the norm it is the output clamp plus the residual, or plus the output bias). With two stages the second reads what the first wrote and writes one to three outputs, the same normed value under their own clamps: the layer's residual add and the norm after it are one launch, the attention norm writes the clamped input q, k and v read, `norm_out` runs in place on the stream. In place allowed |
| `eg2a_ew` | `clamp(act(clamp(x)))`, between the feed-forward's linears |
| `eg2a_lconv` | the light conv's middle in one pass per row: the GLU (after the linear's output clamp), the depthwise causal conv over clips (rows before a clip's start read 0), the conv norm with its gradient clamp and SiLU, and the next linear's input clamp. The GLU is recomputed for the 5 rows a row's conv reads (the neighbours' rows, from cache), which beats writing the GLU's and the conv's output |
| `eg2a_attn` | the chunked local attention, one simdgroup per (query, head); the q/k/v output clamps are applied as it loads, the output projection's input clamp as it stores. q, k, v are read at a row stride and column offset, so a fused q\|k\|v projection output serves all three. Only the `past` candidate keys (`j = i + 1 .. i + past`) are computed: each lane holds its share of `ac + bd` for each, one butterfly reduce-scatter (`ega_scatter_sum16`) sums all 16 candidate slots across the lanes, and the softcap, maximum and exponentials are one value per lane pair |

The attention drops the masked slots' `exp` instead of computing it: a masked
logit is `-1e9`, whose weight is exactly 0 beside any unmasked logit (the softcap
keeps those within `+-cap`), so the host requires `attention_invalid_logits_value
<= -(cap + 104)` (`AudioConfig::validate` and `attention` refuse otherwise) and
requires at most 16 past frames (the real config has 12). When nothing in the
window is visible the logits are all equal and the row takes the uniform `1/ctx`
over the window, values included, as before; that is the kernel's second loop.

q, k and v are one `[hidden, 3 * hidden]` GEMM on one clamped copy of the
attention norm whenever their three input clamps are equal. They are in every
layer of the checkpoint (read from the safetensors header for all 12 layers: for
each, q, k and v have identical `input_min/max` and `output_min/max`, e.g. layer
0 `[-20.375, 20.25]` in and `[-34.5, 34.25]` out, layer 11 `[-9.375, 9.25]` and
`[-17.125, 17.0]`). The first version of this module and its notes said they
differ, which is not so. A checkpoint whose clamps differ takes the three-GEMM
path with three clamped copies; `tests/embedgemma2_audio_tiny.rs` runs both
(random per-linear clamps: all layers split; q's copied to k and v: all layers
fused, and one fused beside one split layer) against the f64 forward.

SiLU and the sigmoid are `kernels/act.h`'s overflow-free forms. A clip bound of
infinity (the module's untrained default) is held as `+-f32::MAX`, since the
kernels are compiled with fast math.

Sums reordered against transformers' (each held to the bounds below, with the
observed errors in the table): the attention logit is one per-lane partial of
`ac + bd` finished by the butterfly (it was `simd_sum(ac) + simd_sum(bd)`), the
second conv accumulates its `9 * C0` terms in 8x8 tiles, and the light conv
sums its five taps as before but from the GLU recomputed in registers.

### Batches and limits

`encode_audio` takes up to 4096 clips (`MAX_CALL_CLIPS`) of 2^21 frames in all (`MAX_CALL_FRAMES`) of any lengths. They are laid end to end
in each row space and run as one forward up to 16,384 frames (`MAX_FRAMES_PER_FORWARD`)
and 256 clips (`MAX_CLIPS`, the clip table's size); a clip above the frame limit is
refused, more clips are more forwards, all queued before the one wait. What bounds a
call is the memory its forwards' inputs and soft tokens hold until that wait (about
1.25 KiB a frame, 2.5 GiB at `MAX_CALL_FRAMES`): a call over either bound is refused
by name before any allocation. Clips never read each other's rows: the convolutions and the attention
work in clip-local indices through a clip table, and everything else is row-wise,
so a clip is bit for bit the same alone or in any batch. Padding (masked)
frames are defined everywhere: the only place a masked frame influences a kept
one is the subsampling's zeroing before each conv. Rows of masked frames are
computed exactly as transformers computes them (a query with no visible key
softmaxes to the uniform `1/24` over its window), and then dropped. The attention
kernel takes at most 16 past frames (`attention_context_left - 1`; the real config
has 12) and an `attention_invalid_logits_value` of at most `-(cap + 104)`;
`AudioConfig::validate` refuses a config outside either.
`trace` returns each clip's subsample output, per-layer residuals and tower
output. The activations are unzeroed, as the text encoder's are; the tests
poison every one with NaN.

### Bounds and observed errors

Fixed before the first run (the real-checkpoint bounds are the ones
`tests/embedgemma2_audio_model.rs` states):

| Check | Bound | Observed |
|---|---|---|
| the f64 reference vs transformers (`tests/embedgemma2_audio_fixtures.rs`: tiny random tower, 3 clips, trailing and interior padding, bounds that bite) | `1e-5` × max | ≤ `5.7e-7` |
| each kernel vs the f64 reference, small and real shapes (`tests/embedgemma2_audio_kernels.rs`) | `2e-5` × max | ≤ `1.4e-6` (attention, 8 heads of 128), second conv `1.3e-6` (128 -> 32 channels), light conv `5.9e-7`; the new kernels have cases that fail under a one-tap, one-pixel or one-lane-mask mutation (see below) |
| tiny tower, 2 layers (`tests/embedgemma2_audio_tiny.rs`): subsample, each layer, tower output; fused and split q/k/v | `1e-4` × max | ≤ `1.4e-6` |
| tiny tower, soft tokens | `1e-4` max abs | ≤ `3.0e-6` |
| same at the checkpoint's attention geometry (hidden 256, head dim 128, chunk 12) | as above | within the two rows above |
| clip alone vs inside any batch, reordered, traced, with every unzeroed activation poisoned, across a split into two forwards | bit-identical | bit-identical |
| real checkpoint, clip 0 (99 frames, 25 tokens), subsample, 12 layers, tower output | `1e-4` × max | ≤ `1.9e-6` (layer 11), tower `2.7e-6` |
| real checkpoint, clip 1 (330 frames, 329 valid, 83 tokens) | `1e-4` × max | ≤ `4.7e-6` (layer 11), tower `1.1e-5` |
| real checkpoint, soft tokens | `1e-4` × max | `3.8e-6` and `2.3e-5` (max abs `1.9e-5`, `1.2e-4` on values up to 5.5); cosine 1.000000000 |
| both clips in one batch vs alone | bit-identical | bit-identical |

The tiny test's bounds bite: with the clip bounds, the gradient clip or the
softcap relaxed in the reference, the tower output moves by `74 %`, `30 %`
and `3.7 %` of its maximum. Two mutations of the attention kernel (the window
one frame wider, the relative index off by one) each fail it at `1e-1`. The
kernels added for speed were each mutated once in the shader and run against
`tests/embedgemma2_audio_kernels.rs`, then restored: the light conv's tap window one
row off fails `lconv_matches_the_f64_reference`, the second conv's frequency padding
dropped fails `sub_conv1_matches_the_f64_reference`, the attention's last butterfly
mask 1 changed to 2 fails `attention_matches_the_f64_reference`.

The first run of the real-checkpoint test failed its soft-token check at
`1.14e-4` max abs because the check had been written as an absolute `1e-4`
where the other stages (and the task) bound by `1e-4` of the largest
magnitude; the stated bound is the relative one, and the soft tokens are within
`2.1e-5` of it. Nothing else was changed. The reference there is the sdpa one.

### Speed

`profile_embedgemma2_audio` times each stage of `encode_audio` on the real
checkpoint (stage-mask variants on zeroed activations, interleaved rounds, min
over rounds; `bench/results/embedgemma2_audio_profile_m5pro.{txt,json,png}`), for
the 1.0 s and 3.3 s golden clips and a synthetic batch of 8 clips of 30 s (24,000
frames, two forwards). `bench_embedgemma2_audio` times the whole call and is what
`bench_paired` runs (`bench/results/embedgemma2_audio_speed_m5pro.{txt,json,png}`).

**Where the time was** (`df5d329`'s kernels, sum of the stages run alone): the
exact-f32 GEMMs are 70-78 % of it (the feed-forward pair is 44 % on the batch and
the 1.0 s clip and 42 % on the 3.3 s clip; the batch GEMMs reach 4.6-7.2 TFLOP/s,
the one-clip GEMMs 0.2-4.5: with 25-83 rows the 4096-wide feed-forward matrices
are streamed for almost no reuse). Next on the batch was the second subsampling conv, 8.0 % (64 ms: one
threadgroup per output position re-read all 147 KB of its weights), then the
q/k/v clamps and pre-norms, the depthwise conv, the GLU and the other row passes
at 1-3 % each, 28 dispatches a layer. On the 1.0 s clip the forward is
launch-bound as much as GEMM-bound. The GEMMs are `gemm`'s (not this module's);
the stage ranking after the work below is the same, led by the feed-forward
GEMMs at 54 % of the batch and 50 % of the 1.0 s clip.

**What changed** (kernels in "Kernels"; each is held to the f64 reference by
`tests/embedgemma2_audio_kernels.rs` and the tiny-tower tests, bounds in the table
above):

* the second conv is an 8x8 `simdgroup_float8x8` tiled kernel: 64.1 to 13.9 ms on
  the batch, 1.07 to 0.51 ms on the 3.3 s clip (stage alone, in two profile runs
  that were not equally loaded; see below);
* the row passes are fused: the residual add and the next norm are one launch,
  the attention norm writes the clamped input q, k and v read (the three input
  clamps are equal in all 12 layers of the checkpoint, so q|k|v is one GEMM
  instead of three), the GLU, depthwise conv, conv norm and SiLU are one launch:
  28 to 17 dispatches a layer, 343 to 210 per one-clip forward;
* the attention computes only the `past` candidate keys, finishes all their logits
  with one butterfly reduce-scatter instead of two `simd_sum`s per key, and reads
  q, k, v at a row stride so the fused projection output is read in place.

`eg2a_glu` and `eg2a_dwconv1d_causal` are gone (`eg2a_lconv` is in their place).
The results stay inside the bounds above, and a clip is still bit for bit the same
alone or in a batch.

**Measured, with its spread.** The GPU was shared with other jobs (1-minute load
average 45 to 220; the profiler printed 90 to 100 % utilisation), and this is the
limit of the result. `bench_paired` with three lanes (`df5d329`, the head, and the
`df5d329` binary again as a control), 12 rounds, min of 8 per process: after/before
median 0.84x (1.0 s clip), 0.84x (3.3 s clip), 0.95x (batch), but the control,
which is the same code as before, reads 1.08x, 1.03x, 1.04x, and single-round
ratios range from 0.2x to 2.2x. Within the noise: no end-to-end speedup is claimed
from the benchmark. With `--evidence --max-spread 1.25` the drift gate refused all
nine series of an 8-round run (spreads 1.9x to 3.7x), and no quiet window came in
nine minutes of waiting. What the profile does show inside a run, where load
touches the GEMM kernels (unchanged; q, k and v are one wider GEMM, which makes
the ratio below slightly conservative) and the changed stages alike: the time of everything
but the GEMMs, per millisecond of GEMM time, went from 0.28, 0.42 and 0.41 (1.0 s,
3.3 s, batch) to 0.25, 0.25 and 0.21, so on the batch, where the GEMMs are 71-83 %
of the forward, the passes outside the GEMMs went from about 29 % to 17 % of the
forward. The end-to-end gain that implies, if the stage sum is the forward and the
GEMMs are unchanged (inferred, not measured): about 14 % on the batch, 12 % on the
3.3 s clip, 2 % on the 1.0 s clip. `bench_paired --evidence
--rounds 8 --max-spread 1.25` on an idle host is the run that replaces this one.

**Still open** (each needs its own benchmark): the feed-forward GEMMs are most of
what is left; at 25-83 rows the 4096 -> 1024 GEMM is under-parallelised (a split-K
or small-M route belongs in `gemm`), and a bf16-weight, f32-activation exact GEMM
would halve the weight traffic (the weights are bf16 in the checkpoint; the
kernels widen them exactly to f32 today); the trailing masked rows of a padded
clip are computed and dropped, and could be skipped (the only place a masked frame
influences a kept one is the subsampling's zeroing, so trailing masked rows feed
no kept row; not implemented); the attention could use
`simdgroup_float8x8` tiles for the 12-query blocks; the first conv and the
second could share one launch.

## Preprocessing

`tessl::embedgemma2_pre` makes the towers' inputs from raw data on the CPU, with
no Python processor: an image (RGB, RGBA, gray or gray+alpha bytes), mono f32 PCM
at 16 kHz, or decoded video frames become `ImageInput` / `AudioInput` (what
`EmbedGemma2Multimodal::encode` takes) and, for video, the per-frame patch tensors
with the sampled indices and timestamps. Its parameters are read from the
snapshot's `processor_config.json` through the crate's strict JSON parser
(`PreConfig::from_snapshot`); a setting it does not implement (normalisation,
another resample filter, pre-emphasis, dither, `per_bin_mean`, left padding, a
frame or FFT length that disagrees with the one transformers derives) is refused by
name, not ignored.

```rust
let cfg = PreConfig::from_snapshot(&snapshot)?;
let img = preprocess_image(&cfg.image, &RawImage::rgb(w, h, &bytes))?;     // PreparedImage
let clip = preprocess_audio(&cfg.audio, &pcm, 16_000)?;                    // AudioInput
let out = model.encode(&[MmItem::Image(img.input()), MmItem::Audio(&clip)], None)?;
let video = preprocess_video(&cfg.video, &frames, source_fps)?;            // PreparedVideo
```

### Which transformers path

`AutoProcessor.from_pretrained(snapshot)` builds `Gemma4ImageProcessor`, the
torchvision ("fast") processor, and sentence-transformers calls that processor;
`Gemma4ImageProcessorPil` is not what either uses (checked: the object's type, and
`image0_pixel_values.npy` of `tools/embedgemma2_ref/make_mm_reference.py`, which is
sentence-transformers' own input, equals this module's output bit for bit). The two
differ in the resize, so the choice matters: on a uint8 CPU tensor
`tvF.resize(..., bicubic, antialias=True)` runs ATen's fixed-point kernel, not PIL's.
Measured against `Image.resize(BICUBIC)` on the 17 fixture inputs at the checkpoint
geometry: at most 2 grey levels apart, 0.1 % to 0.8 % of the bytes differ. A
float32 resize without ATen's rounding between the two passes is much further off
(up to 18 grey levels on noise), which is why the kernel is reproduced rather than
approximated.

**Image**, in order: `do_convert_rgb` (PIL's `convert("RGB")`: gray replicated,
alpha dropped, no compositing; transformers applies it to PIL input only, a numpy
RGBA array is passed on with four channels, which the tower cannot take, so this
module converts whatever it is given); the target size, the largest multiple of
`patch_size * pooling_kernel` per side within `max_soft_tokens * pooling_kernel² *
patch_size²` pixels (an extreme aspect ratio clamps the long side: 1x2000 becomes
48x13440); the resize; `f32(u8) * f32(rescale_factor)`; patches of `patch_size² * 3`
values in (row, column, channel) order; `(x, y)` positions row-major; zero padding
and `(-1, -1)` positions to `max_soft_tokens * pooling_kernel²` patches. The resize
(`resize_rgb_u8`): per axis the antialiased bicubic taps (`a = -0.5`, support
`2 * max(scale, 1)`) are computed in f64 and normalised, converted to integers at
the largest precision `p` (at most 22) for which `2^(p+1) * max weight < 2^15`, and
applied as `(sum + 2^(p-1)) >> p` clamped to a byte, the width pass first and
rounded to uint8, then the height pass. This was recovered by experiment against
`torch.nn.functional.interpolate` and then matched bit for bit; it is arm64's generic
ATen kernel. **Unverified:** ATen's x86 AVX2 uint8 kernel implements the same scheme
separately and was not available here.

**Audio**: described in the module docs; the frames, mask and frame count equal the
extractor's (`(padded + 160 - 321) / 160 + 1` frames, where the clip is cut to
`max_length` rounded up to 128, padded to a multiple of 128, and frame `i` is real
when sample `i * 160 + 160` exists), a clip too short for a frame has none, and
the features are `ln(mel + 1e-3)` of the HTK filterbank applied to the magnitudes of
an f64 radix-2 real FFT of the f32-windowed frame. transformers' FFT is numpy's
float32 one, so the features agree to f32 rounding rather than bit for bit. The
lowest mel filter falls between FFT bins and is all zero (transformers warns about
it): its feature is exactly `ln(1e-3)`. A different sampling rate is refused
(transformers ignores `sampling_rate` and would silently compute on the wrong
rate), as is a non-finite sample (transformers propagates NaN into every frame).
Soft tokens per clip are the true entries of the tower's mask
(`embedgemma2_audio::out_mask`): 750 for a 30 s clip; `audio_seq_length` (280 in
`processor_config.json`) caps only `_get_num_multimodal_tokens`, not the
placeholders `replace_audio_token` writes, so the processor's `input_ids` of a 31 s
clip hold 750 audio tokens.

**Video**: `sample_frame_indices` is `EmbeddingGemma2VideoProcessor.sample_frames`:
with `fps` (1) it takes `max(1, int(duration * fps))` frames at `min(total - 1,
int(i * source_fps / fps))`, then, above `max_frames` (32), `np.linspace(0, n - 1,
32, dtype=int)` of them (`uniform`, numpy's own `i * step`, last element forced) or
the first 32 (`truncate`). `duration` is `frames / source_fps`, as transformers'
loaders set it; without a duration transformers skips fps sampling and keeps every
frame (that is the config `fps: null`, or `do_sample_frames: false` for no sampling
at all). Each sampled frame goes through the image path at `max_soft_tokens` 140
(1260 patches). `PreparedVideo` carries the frames, their source indices and the
timestamps `index / source_fps`; `timestamp_label` is the processor's `mm:ss`.
`add_timestamps` (`VideoConfig::add_timestamps`, `false` in the snapshot) is read
like any other setting and changes nothing here: the frames are preprocessed the
same either way, and the flag is the sequence layout's (see [Video](#video)).
Placing the video in a sequence, and the frames through the tower, are the
multimodal model's.

### Parity with transformers

`scripts/gen_embedgemma2_pre_fixtures.py` runs the snapshot's own processors and
saves inputs and outputs under `tests/fixtures/embedgemma2/` (`eg2p_*`, 3.9 MB,
deterministic); `tests/embedgemma2_pre.rs` holds this module to them, without a
device. Measured:

| What | Cases | Result |
|---|---|---|
| Image pixels and positions, small geometry (patch 4, pooling 2, 70 tokens), stored whole | 13 inputs: 1x1, 1x2000, 2000x1, an exact-target size, gray, gray+alpha, RGBA, a photo, noise, tall, down- and up-scales | bit for bit |
| Image pixels, the checkpoint's geometry (16, 3, 280), CRC-32 of 7.7 MB | the same 13 | equal |
| Same, full arrays (`EMBEDGEMMA2_PRE_BIG_DIR`, `#[ignore]`) | those 13 and a 2000x1500 photo, the 768x768 exact target, 3000x200, noise 1024², RGBA 640x360 | bit for bit |
| Image pixels of `make_mm_reference.py`'s two images (224², 640x360), from the raw bytes | 2 | bit for bit (0 of 1,935,360 values differ) |
| Log-mel features, mask and frame count | 15 clips: silence, sweep, noise, shorter than a frame (0 frames), 150 samples (1 frame, not real), exactly 161, 320, odd lengths, DC, Nyquist, loud, near silent, cut at a limit; plus a 30 s and a 35 s clip | mask equal; max abs 4.77e-7 (one f32 ulp at magnitude 4 to 8), max relative 2.35e-7; bound 2e-6 |
| Sampled frame indices | 36 frame counts (1 to 5000) x 21 source rates (NTSC, 1/10, 7/3, 120 among them), uniform cap; truncate; no fps; max_frames 1 | 778 lists equal |
| Videos: indices, timestamps (exact f64), pixels, positions | 7 (3 to 70 frames; 33 and 40 frames capped to 32; 1 frame; both geometries), and 100- and 600-frame ones in the big set | bit for bit |
| Raw image / clip to embedding, real checkpoint (`tests/embedgemma2_pre_model.rs`, ignored) | 2 images, 2 clips vs sentence-transformers fp32 | max abs 1.3e-7 to 2.3e-7, cosine 1.000000000 |

Mutation check: capping the resize's weight precision at 12 bits instead of 22
fails the image and video tests at the first pixel.

### Bounds and speed

Bounds, each fixed by what was measured above: pixels, positions, sampled indices
and timestamps exact; log-mel features 2e-6 absolute (measured 4.77e-7); the
embedding of a raw input within `1e-4` max abs and cosine `0.99999` of
sentence-transformers' (the bound of `tests/embedgemma2_e2e_model.rs`; measured
2.3e-7 and 1.000000000).

Speed, CPU, Apple M5 Pro, `bench_embedgemma2_pre` against
`bench/embedgemma2_pre_hf.py` (transformers' processors on the same inputs,
CRC-equal) under `bench_paired`, 8 interleaved rounds, min of 15 per process
(`bench/results/embedgemma2_pre_m5pro.{txt,json,png}`). The host was shared (1-minute
load average 72 to 146 on 18 cores), so only the ratios and their spread are
claimed: tessl is faster by the median per-round ratio, 2.4x to 4.6x on images, 10 s
and 30 s of audio and video (single-round ranges 1.4x to 7.4x, never below 1), and
a 1 s clip, where both take about 0.2 ms, is not distinguished (0.14x to 2.34x).

| workload | tessl ms | transformers ms | median ratio [lowest, highest round] |
|---|---|---|---|
| image 224x224 | 0.75 | 2.26 | 3.32x [2.02, 4.52] |
| image 640x480 | 1.15 | 3.10 | 2.72x [1.72, 4.94] |
| image 1920x1080 | 2.64 | 6.53 | 2.68x [1.43, 3.63] |
| image 4000x3000 | 4.77 | 21.72 | 4.62x [2.70, 7.19] |
| audio 1 s | 0.17 | 0.23 | 1.38x [0.14, 2.34], inconclusive |
| audio 10 s | 0.85 | 2.59 | 3.02x [2.09, 3.56] |
| audio 30 s | 2.37 | 7.98 | 3.16x [2.28, 4.97] |
| video 40 frames 640x360 (32 sampled, 140 tokens) | 29.2 | 80.9 | 2.38x [1.64, 6.35] |
| video 40 frames 1280x720 | 55.0 | 193.3 | 3.97x [1.37, 7.38] |

Both lanes use their own default threads (tessl at most 8, transformers torch's
pool); single-thread speed, x86 and speed beside other work on the same cores were
not measured. Before the front end split short clips across threads, a 1 s clip took
0.38 ms (one process of an earlier, abandoned run, not part of the run above).

Reproduce: `scripts/gen_embedgemma2_pre_fixtures.py` (fixtures; with
`EMBEDGEMMA2_PRE_BIG_DIR=target/embedgemma2_pre_big` the big cases),
`cargo test --release --test embedgemma2_pre`, and with the checkpoint
`tools/embedgemma2_ref/make_mm_reference.py` (now also saves `*_raw.npy`) then
`cargo test --release --test embedgemma2_pre_model -- --ignored --test-threads=1`.

## Tokenizer

`tessl::embedgemma2_tok` turns a text into the ids the encoder takes, from the
snapshot's own `tokenizer.json`, in Rust: no Python tokenizer, no new
dependency. `Tokenizer::from_snapshot(dir)` reads `tokenizer.json`,
`tokenizer_config.json` and `config_sentence_transformers.json`;
`Tokenizer::from_file(path)` reads a `tokenizer.json` alone.

```rust
let tok = Tokenizer::from_snapshot(&snapshot_dir)?;           // once: ~0.4 s, see Speed
let ids = tok.encode_with_prompt(Some("query"), "how do I speed up attention?")?;
let out = model.encode_texts(&tok, Some("document"), &["a text", "another"], None)?;
let text = tok.decode(&ids);                                    // tokenizers' decode
```

### What the format is (read from the snapshot, `tokenizers` 0.23.2)

`tokenizer.json` (version 1.0, no truncation, no padding):

| piece | what it is | effect |
| --- | --- | --- |
| model | `BPE`, 262144 tokens (ids dense, `0..262144`), 514906 merges as `[left, right]` pairs with no repeat, `byte_fallback: true` with all 256 `<0xHH>` tokens, `unk_token: <unk>`, `fuse_unk: true`, `ignore_merges: false`, `dropout: null` | a character with no vocabulary entry becomes its UTF-8 bytes |
| normalizer | `Replace` the literal `" "` with `U+2581` | no other normalization: no lower-casing, no Unicode normalization, no strip |
| pre-tokenizer | `Split` on the literal `" "`, `MergedWithPrevious` | **inert**: after the normalizer no space is left (`normalize_str("a b  c")` is `a▁b▁▁c` and it pre-tokenizes to one piece), so the BPE merges run over each stretch of text between added tokens as one word, 100k characters included |
| added tokens | 24, all `special: true`, `normalized: false`, no `lstrip`/`rstrip`/`single_word`: `<pad>` 0, `<eos>` 1, `<bos>` 2, `<unk>` 3, `<mask>` 4, `<\|tool>` 46, `<tool\|>` 47, `<\|tool_call>` 48, `<tool_call\|>` 49, `<\|tool_response>` 50, `<tool_response\|>` 51, `<\|"\|>` 52, `<\|think\|>` 98, `<\|channel>` 100, `<channel\|>` 101, `<\|turn>` 105, `<turn\|>` 106, `<\|image>` 255999, `<\|audio>` 256000, `<\|image\|>` 258880, `<\|audio\|>` 258881, `<image\|>` 258882, `<audio\|>` 258883, `<\|video\|>` 258884 | matched in the **raw** text before the normalizer, leftmost then longest; `<bo`, `<BOS>`, `< bos>` and the 6227 `<unusedN>` tokens (in the vocabulary but not added tokens) are ordinary text |
| post-processor | `TemplateProcessing`, single: `<bos> $A <eos>` | `[2, ..., 1]` |
| decoder | `Replace` `U+2581` with `" "`, `ByteFallback`, `Fuse` | a run of `<0xHH>` tokens is one UTF-8 string, or one `U+FFFD` per byte if the run is not valid UTF-8 |

`tokenizer_config.json`: `bos_token` `<bos>`, `image_token` `<|image|>`,
`audio_token` `<|audio|>`, `video_token` `<|video|>`, `model_max_length` 1e30 (so
nothing truncates). `config_sentence_transformers.json`: 20 named prompts
(`query` is `"task: search result | query: "`, `document` is `"title: none |
text: "`, ...), `default_prompt_name: null`.

### The ids sentence-transformers feeds the model

They are not `Tokenizer.encode(text)`. Read from sentence-transformers 6.1 and
transformers 5.19 (`base/modules/transformer.py` `preprocess`,
`processing_utils.py` `apply_chat_template`) and confirmed by probing:

* the model's `modules.json` declares a `message` modality, so a text is routed
  through the chat template (`chat_template.jinja`), and a prompt is a **system
  message** that the template writes in front of the text with nothing between:
  the tokenizer sees `prompt + text`, exactly;
* `<bos>`/`<eos>` come from the post-processor, **except** that
  `apply_chat_template` passes `add_special_tokens=False` when the rendered
  string starts with `tokenizer.bos_token`. So `"<bos>hi"` is `[2, 2202]`: the text's
  own `<bos>` is the only one and there is **no `<eos>`**. `"a<bos>b"` is wrapped
  as usual, and a prompt never starts with `<bos>`, so a prompted text is always wrapped;
* `validate_inputs` raises `ValueError` for a text containing `<|image|>`,
  `<|video|>` or `<|audio|>` (no media to fill the placeholder). The other added
  tokens in a text are encoded as their ids (`<|image>` is 255999);
* nothing is stripped, normalized or truncated (`max_seq_length` is 1e30);
* **batch quirk**: that `startswith` test is made on the **first** text of the
  batch (`single_prompt = prompt[0]`) and applied to the whole batch.
  `["<bos>hi", "plain"]` gives `plain` the ids `[30448]` (neither `<bos>` nor
  `<eos>`), `["plain", "<bos>hi"]` gives `<bos>hi` `[2, 2, 2202, 1]`. A text's ids
  depending on its neighbours is a bug, so `Tokenizer::encode_batch` applies the
  rule to each text alone: it returns what `encode` returns for each, which is
  what sentence-transformers gives for a batch of one. **This is a deliberate
  difference** from `SentenceTransformer.encode` for a batch whose first text starts
  with `<bos>` (and only then). `scripts/gen_embedgemma2_tok_fixtures.py real`
  asserts the quirk against the installed transformers, so a fix upstream is
  noticed.

Not reproduced: `SentenceTransformer.encode` classifies a string as an image,
audio or video (not text) when it is an `http(s)` URL without spaces ending in
a media extension, a `data:image/` URI, or the path of an existing media file
(`base/modality.py` `infer_modality`). `Tokenizer` treats every `&str` as text.

### API

| call | returns |
| --- | --- |
| `Tokenizer::from_snapshot(dir)`, `from_file(path)`, `from_json(text)` | a tokenizer; `with_tokenizer_config(text)` and `with_prompts(text)` add the sidecars |
| `encode(text)`, `encode_with_prompt(Some(name), text)` | the ids the model receives (one text, as above); `Err` for a refused placeholder or an unknown prompt name |
| `encode_batch(texts)`, `encode_batch_with_prompt(name, texts)` | each text as `encode` has it; an `Err` names the text's position |
| `encode_raw(text, add_special_tokens)` | `Tokenizer.encode(text, add_special_tokens).ids` of the library, no sentence-transformers rule. A text (with its prompt) of `MAX_TEXT_BYTES` (256 MiB) or more is an `Err` from `encode`/`encode_with_prompt` and a panic here: the merge list's positions are 32-bit |
| `decode(ids)`, `decode_with(ids, skip_special_tokens)` | `Tokenizer.decode`: special tokens skipped by default, as that method does (transformers' `decode` defaults to keeping them); an id with no token is skipped, as the library does |
| `prompt(name)`, `prompt_names()`, `token_to_id`, `id_to_token`, `vocab_size` | lookups |
| `EmbedGemma2Model::encode_texts(&tok, prompt, texts, truncate_dim)` | tokenize, then `encode` (one prompt for the call) |

A format feature this tokenizer does not implement is refused by name when the
file loads, never approximated: a dropout, a continuing-subword prefix or
end-of-word suffix, a regex pattern, a `Split` that is not `MergedWithPrevious`,
`ByteLevel`/`Metaspace`/`Strip` steps, any added token with `lstrip`, `rstrip`,
`single_word` or `normalized`, an added token that is not the vocabulary's
entry for its text, a non-`TemplateProcessing` post-processor, a null decoder,
a truncation or padding block, sparse or repeated vocabulary ids, a merge
naming a token the vocabulary lacks. `tests/embedgemma2_tok_tiny.rs` makes each
edit to a real file and checks the error.

Two behaviours of the library are reproduced because the fixtures showed them,
not because they are sensible: a repeated merge keeps the **later** rank, and
an unknown-character run is written after the byte-fallback tokens that follow
it (`merge_word` only flushes it at the next known character).

### Load and the JSON parser

`tokenizer.json` is 32 MB. It goes through `src/json.rs` (the crate's strict
parser: unique keys, valid UTF-8, RFC escapes, no trailing bytes) with `Syntax {
uints_only: true, literals: true, max_depth: 6 }`: that set already fits the
format (non-negative integers, `null` and booleans, containers four deep), so
`Syntax` is **unchanged** and no other caller's check was touched. The tree is
then reduced to the token text in one arena, an open-addressing index from text
to id, a table of the one-character tokens, and an open-addressing table of the
514906 merge ranks (key and value in one 16-byte slot, behind a 2 MiB bit
filter that answers most absent pairs from cache), and dropped. Encoding never
reads the JSON again, allocates only its working buffers, and the value is
`Send + Sync`. The files are read through the crate's regular-file open with a
byte cap (256 MiB for `tokenizer.json`, 1 MiB for the sidecars).

The BPE is the library's `merge_word` and `Word::merge_all`: a min-heap on
(rank, position) over a linked list of symbols, so a 100k-character word is
O(n log n). I could not fetch the library's Rust source (the GitHub URLs tried
returned 404), so those two routines are written from the algorithm as I know it
and held to the library by the corpora below, not by reading its code.

### Checked against

Exact id equality, no tolerance. Tiny fixtures (committed, device-free,
`tests/embedgemma2_tok_tiny.rs`): three hand-written tokenizers
(`tests/fixtures/embedgemma2_tok/tiny_*.json`: the snapshot's shape with an
added-token prefix pair `<a>`/`<a><b>`; a real space split with `"a b"` merges, a
repeated merge, `ignore_merges`, `fuse_unk` and a plain added token; a byte table
with gaps and a two-step normalizer), each with 663 texts (with and without
specials) and 800 decode cases from the library, all equal. The real snapshot
(`tests/embedgemma2_tok_real.rs`, `#[ignore]`, corpus from
`scripts/gen_embedgemma2_tok_fixtures.py real` into `target/`, 11 MB, not
committed): 4671 texts through sentence-transformers' own `preprocess`, one at a
time, and through the library alone:

| corpus | texts | encode vs sentence-transformers | encode_raw vs `tokenizers` (with and without specials) |
| --- | --- | --- | --- |
| every prompt x 6 samples, empty, `<bos>` after a prompt | 121 | 121/121 | 242/242 |
| plain (no prompt) | 7 | 7/7 | 14/14 |
| whitespace (runs to 200, tabs, newlines, CRLF, NBSP, ideographic space, leading/trailing) | 36 | 36/36 | 72/72 |
| CJK, Devanagari, Arabic, Thai, other scripts | 10, 6, 6, 5, 7 | all | all |
| combining marks | 15 | 15/15 | 30/30 |
| emoji (ZWJ sequences, flags, keycaps, tag sequences) | 19 | 19/19 | 38/38 |
| control characters (all C0, DEL, all C1) | 69 | 69/69 | 138/138 |
| private use | 9 | 9/9 | 18/18 |
| byte fallback (characters with no vocabulary entry) | 124 | 124/124 | 248/248 |
| special-token strings (every added token, doubled, adjacent to text, partial) | 190 | 190/190 | 380/380 |
| `<\|image\|>`, `<\|video\|>`, `<\|audio\|>` (refused by sentence-transformers) | 9 | refused, and the library's ids equal | 18/18 |
| digits, code | 20, 14 | all | all |
| long (100000 characters, `word ` x 20000, `x` x 50000, `日本語` x 10000) | 4 | 4/4 | 8/8 |
| 4000 seeded random strings over all of the above | 4000 | 4000/4000 | 8000/8000 |
| **total** | **4671** | **4680/4680** (4671 texts, the 9 refused counted twice) | **9342/9342** |

15255 of the fuzz tokens and 642 of the byte-fallback group's are `<0xHH>`
bytes, so the fallback is exercised. Decode: 3533 cases, 3533 equal to the
library's string: 1000 round trips of the corpus's own ids with
`skip_special_tokens` on and off, 2500 random id sequences (ordinary ids, byte
tokens, special tokens, ids past the vocabulary, `u32::MAX - 1`) and 33 byte
sequences (valid 2-, 3- and 4-byte, truncated, stray continuation, overlong,
surrogate, too large, `0xFF`, all 256).

I checked that the corpora bite: reversing the merge heap's position tie-break
fails 7 of the 11 tiny tests, and ordering the added tokens shortest first fails
the tiny parity test (26 of 2126 `tiny_gemma` records differ).

The text path from a string: `tests/embedgemma2_e2e_model.rs`
(`#[ignore]`, `EMBEDGEMMA2_REF_DIR` the text references) holds the Rust
tokenizer to `ids.npy` for the eight reference texts (exact) and
`EmbedGemma2Model::encode_texts` to `emb_f32.npy` with the bounds of that file
(`1e-4` max abs, cosine `0.99999`): observed max abs `1.3e-7` to `2.4e-7`,
cosine `1.000000000`, on all eight (10 to 6147 tokens), and bit-identical to
`encode` on the reference ids.

### Speed

`bench_embedgemma2_tok` (load time, tokens per second on a seeded corpus)
against `bench/embedgemma2_tok_hf.py` (the same corpus through `tokenizers`),
alternated by `bench_paired`; the results and the plot are
`bench/results/embedgemma2_tok_m5pro.{txt,json,png}`. On a heavily loaded
machine (load average 110 to 135), best of 12 interleaved rounds: one tessl
thread is 2.2x to 3.7x one `tokenizers` thread (3.8, 2.1, 4.0 and 5.1 M tokens/s
on short queries, documents, multilingual text and a 100k-character text, against
1.5, 1.0, 1.1 and 1.6 M), ahead of `encode_batch` on all cores on three of the four
workloads and behind on the documents (0.3x), and loads in 0.48 s against 0.65
to 0.68 s. The spread between rounds is a factor of two to five, so these are
ratios of minima, not predictions. The hash of all ids is equal in both lanes
on every workload.

### Regenerating, and what is not covered

```text
~/.venvs/ml-2026-10/bin/python scripts/gen_embedgemma2_tok_fixtures.py tiny   # rewrites tests/fixtures/embedgemma2_tok/tiny_*/
~/.venvs/ml-2026-10/bin/python scripts/gen_embedgemma2_tok_fixtures.py real   # target/embedgemma2_tok_ref/
EMBEDGEMMA2_SNAPSHOT=... cargo test --release --test embedgemma2_tok_real -- --ignored --test-threads=1
```

Verified only for snapshot `914f7f89...` with tokenizers 0.23.2, transformers
5.19.0 and sentence-transformers 6.1.0; another revision with a different
`tokenizer.json` is loaded by the same code and checked by the tiny fixtures'
features, not by this corpus. The Python binding's `encode_texts`
(`python/tessl_torch/embedgemma2.py`) and the C ABI are not changed by this
module: the Python one asks its tokenizer for special tokens unconditionally,
which differs from sentence-transformers for a text starting with `<bos>`.

## Text-encoder fusion (G7)

The text encoder's elementwise passes between its GEMMs are fused where the arithmetic
can be kept bit for bit (`docs/embedgemma2_audit.md`, G7; the audio tower's version of the
same work is "Kernels" and "Speed" under "Audio tower"). `ExactF32` embeddings are
unchanged to the bit on the tiny and the real checkpoint (hashes below), `Bf16` ones too,
so every bound of "Bounds and observed errors" and "Precision" holds as it was. What
changed is the dispatch count (15 to 13 per layer at `ExactF32`, 17 to 13 at `Bf16`, and
369 to 320 per forward at 1x16 `ExactF32`, 419 to 322 at `Bf16`), the bytes the row passes
move, and the bf16 casts, which are gone but for two per forward. How much of that is
time is "Speed (text-encoder fusion)" below, and it is small at `ExactF32`.

### What a layer dispatched, and what the elementwise passes cost

One layer before (`5229a1c`), in dispatch order. "GEMM" is one launch at the exact-f32
GEMM's default route, plus a zero fill that route starts from, and a K-partition reduction
where `gemm_small_m` takes the few-tile shapes at 16 rows (those two are counted in the
measured totals, not in the list):

```text
rms_norm_f32              input norm               *
GEMM                      q|k|v
qwen35_attn_qk_norm_rope  q/k norm, RoPE, K/V store
encoder_attn_tiled        attention                               (Bf16: f32 output)
cast_f32_to_bf16          attention output         Bf16 only      *
GEMM                      o
rms_norm_residual_add_f32 post-attention add       *
rms_norm_f32              pre-feed-forward norm    *
GEMM                      gate|up
qwen35_gelu_tanh_glu      gelu(gate) * up
GEMM                      down
rms_norm_residual_add_f32 post-feed-forward add    *
cast_f32_to_bf16          residual stream          Bf16 only      *
GEMM                      per-layer-input gate
qwen35_gelu_tanh_glu      gelu(gate) * per-layer input
GEMM                      per-layer-input output
rms_norm_residual_add_f32 post-per-layer-block add, times the layer scalar   *
```

15 launches at `ExactF32` (6 GEMMs), 17 at `Bf16`. Per run of per-layer inputs (all 24
layers up to 341 rows, one layer per run from 8192 rows): the NT GEMM, `scale_f32_inplace`
and `rms_norm_f32`. Per forward: the gather, a copy, the final norm, the pooling, the
projection GEMM and the normalize, and at `Bf16` the casts of the embeddings and the pooled
vectors. The starred passes are what the fusion touches.

One layer after, in dispatch order:

```text
(layer 0 only) rms_norm_f32     input norm
GEMM                            q|k|v
qwen35_attn_qk_norm_rope        q/k norm, RoPE, K/V store
encoder_attn_tiled              attention        (Bf16: encoder_attn_tiled_bf16o, bf16 output)
GEMM                            o
eg2t_resid_norm_{f32,bf16}      post-attention add + pre-feed-forward norm
GEMM                            gate|up
qwen35_gelu_tanh_glu            gelu(gate) * up
GEMM                            down
rms_norm_residual_add_f32       post-feed-forward add         (Bf16: eg2t_resid_add_bf16_copy,
                                                                add and the residual's bf16 copy)
GEMM                            per-layer-input gate
qwen35_gelu_tanh_glu            gelu(gate) * per-layer input
GEMM                            per-layer-input output
eg2t_resid_norm_{f32,bf16}      post-per-layer-block add, times the layer scalar, + the NEXT
                                layer's input norm (after the last layer: the final norm)
```

13 launches at both precisions (6 GEMMs, 7 others), where it was 15 and 17: the pre-feed-forward
norm rides on the post-attention add, the input norm of layer N+1 rides on layer N's last
add and the final norm on the last layer's, and at `Bf16` the two casts are gone (the
attention writes bf16 straight into the buffer the o GEMM reads, and the feed-forward add
writes the bf16 copy of the stream as it adds). Per run of per-layer inputs the scale and
the norm are one `eg2t_scale_rms_norm_f32`. That is exactly 2 launches per layer plus 1 per
run of per-layer inputs at `ExactF32`, and 4 per layer plus 1 per run at `Bf16`.

Per forward on the real checkpoint (`profile_embedgemma2`'s replica, `PROFILE_MODE=stages`,
which prints `dispatches_full`; the replica is checked against `encode` below):

| workload (batch x tokens) | `ExactF32` before | `ExactF32` after | `Bf16` before | `Bf16` after |
|---|---|---|---|---|
| 1x16 | 369 | 320 | 419 | 322 |
| 64x32 | 384 | 330 | 434 | 332 |
| 8x1024 | 438 | 366 | 488 | 368 |

`tests/embedgemma2_tiny.rs` pins the layer: `a_layer_runs_the_pinned_kernels` and
`bf16_a_layer_runs_the_pinned_kernels` encode tiny models of 1, 2 and 3 layers under the
kernel trace (`TESSL_KERNEL_TRACE`) and require that each added layer adds exactly seven
kernels outside the GEMMs, and these by name: two `eg2t_resid_norm_*`, one feed-forward
add (`rms_norm_residual_add_f32`, or `eg2t_resid_add_bf16_copy` at `Bf16`), two
`qwen35_gelu_tanh_glu_*`, one `qwen35_attn_qk_norm_rope*` and one attention kernel
(`encoder_attn_tiled_bf16o_h256_q32_k64_sg4` at `Bf16`). The whole forward has one standalone
`rms_norm_*` (layer 0's input norm), no `scale_f32_inplace`, one `eg2t_scale_rms_norm_f32`,
and no `cast_f32_to_bf16` at `ExactF32` and two at `Bf16` (the embeddings and the pooled
vectors). The exact-f32 GEMMs launch a zero fill and, at small M, a partition reduction of
their own, so the all-in count per layer at `ExactF32` is not pinned, only the non-GEMM
part, which is where the fusion acts. Both tests fail at `5229a1c`, where a layer has 9
(`ExactF32`) and 11 (`Bf16`) non-GEMM kernels against 7.

### What is fused, and how it stays bit for bit

`kernels/embedgemma2_text.metal` holds four kernels (the rows of "Kernels" above name them):

- `eg2t_resid_norm_f32` / `eg2t_resid_norm_bf16`: `resid = s * (resid + rms_norm(y) * w_post)`
  (a plain `+=` when the layer scalar `s` is 1), then `out = rms_norm(resid) * w_next`, one
  threadgroup per row. The second norm reads the row it has just written, from the same
  threadgroup. The `_bf16` one writes `out` as bf16 (the next GEMM's operand).
- `eg2t_resid_add_bf16_copy`: the feed-forward add, which also writes `bfloat(resid)`.
- `eg2t_scale_rms_norm_f32`: the per-layer inputs' scale and norm, in place.

Both reductions are `rms_norm.metal`'s, copied, not re-derived: each lane sums
`d = lid, lid + tptg, ...` and the lanes fold with `reduce_row_add`, and the host picks the
threadgroup width with the same `reduce_tptg(maxTotalThreadsPerThreadgroup, dim)`, so a
fused kernel's pipeline limit must equal that of the norm it replaces (a test pins the
equality, since the limits depend on each kernel's register use and could differ).
The per-element expressions are the unfused ones in the same order. The Metal compiler
runs with fast-math, which may reassociate, and `(y * inv) * w` and `y * (inv * w)`
compile to the same code, so the bit identity is shown by test and by hash, not argued.

Tests in `tests/embedgemma2_kernels.rs` (section 7), each comparing against the unfused
kernels' bits (outputs poisoned first) at `(rows, dim)` of (1,1), (1,8), (9,33), (3,64),
(5,512), (4,1000), (7,768), (2,2048) and (3,4096) with layer scalars 1.0, 0.37 and 1.75:
`resid_add_norm_f32_is_the_unfused_pair_bit_for_bit`,
`resid_add_norm_bf16_is_the_unfused_pair_bit_for_bit`,
`resid_add_bf16_copy_is_the_unfused_add_and_cast_bit_for_bit`,
`scale_rms_norm_inplace_is_the_unfused_pair_bit_for_bit`,
`the_fused_row_passes_launch_as_wide_as_the_kernels_they_replace`,
`the_fused_row_passes_refuse_what_they_cannot_run` (dim 0, eps 0, a NaN or infinite
scale, an output aliasing the residual, short weights or outputs, bf16 storage handed
to an f32 output; a refused call writes nothing) and `the_bf16_output_attention_is_the_bf16_kernel_rounded_bit_for_bit` (head dims
64, 256 and 512). The four row-pass tests fail if an element of the fused kernel is scaled
by one ulp (`* 1.0000001f`, or `* 1.01f` on the bf16 copy), which was checked by mutating
the kernel and restoring it.

Kept unfused, deliberately:

- `qwen35_gelu_tanh_glu` (the two gated activations per layer). It is 5 to 13 % of the
  stage sum (5 to 6 % at `ExactF32`, 12 to 15 % at `Bf16` on the two larger workloads,
  below), so it is not nothing. The way to fuse it is into the
  GEMM that produces `gate|up`: but `gemm_epilogue` refuses the exact-f32 GEMM, and an
  epilogue applies one activation to each output on its own, whereas this one forms
  `gelu(gate) * up` across the two halves of the packed columns. That is a new GEMM
  kernel (a column-pair epilogue), not a fusion of existing launches, and it has its own
  bit-identity question (the exact-f32 GEMM's accumulation order). Left open.
- `rms_norm_f32` for layer 0's input norm, `rms_norm_residual_add_f32` for the f32 feed-forward
  add (it already is an add and a norm in one pass; the per-layer-input gate GEMM
  follows it, and a GEMM's input is not a row pass), the GEMMs, the qk norm with RoPE,
  the attention.

### The bf16 attention output

The follow-up "No bf16 attention output" is done: the shared attention body
(`attn_tiled_core` in `kernels/attn_tiled.h`) already had a runtime flag that makes it
store bf16 (`out_bf16`); `kernels/encoder_attn_tiled.h` passed it as 0. The three `encoder_attn_tiled_bf16o_*`
entries (`kernels/encoder_attn.metal`) pass the literal flag `1u` and a `bfloat` output
type, so the stored value is the f32 result rounded exactly as `cast_f32_to_bf16` rounded
it (tested bit for bit). At `Bf16` the attention writes straight into the buffer the o GEMM
reads, and the f32 `a.attn` buffer is no longer written (it is still allocated, as the
split-K scratch the exact-f32 GEMM wants and `Bf16` ignores; removing it would save
`rows * q_width * 4` bytes and rewrite the itemised-bytes assertions of
`tests/embedgemma2_memory.rs` and `tests/embedgemma2_small_m.rs`, which is its own change).

### Embeddings are unchanged: the hashes

An FNV-1a hash over the bits of every embedding, `5229a1c` against this change, same
inputs, same process settings. The tools are `PROFILE_MODE=hash cargo run --release --bin
profile_embedgemma2` (the real checkpoint, six workloads at both precisions, and with
`PROFILE_PLE_BYTES=1`, which forces one layer per run of per-layer inputs) and the ignored
test `tiny_embedding_hashes` in `tests/embedgemma2_memory.rs` (the tiny checkpoint, three
cases at both precisions). Every digest is identical, 12 + 8 + 6 of 26:

| real checkpoint | `ExactF32` | `Bf16` |
|---|---|---|
| 1x16 | `e509a29307bb3de2` | `ce9fe1d156897056` |
| 64x32 | `ab2f8973d25ec584` | `899de415b7e967ca` |
| 32x256 | `a37be16381616af5` | `7e0e2b608c46942f` |
| 8x1024 | `bfa1432d66775a24` | `274f02bf9af7dcdc` |
| 2x4096 | `87452f717e79b313` | `542773be4bc27c88` |
| 1x4096 + 63x32 | `64b78e28e233bbc5` | `b5c029be1653df84` |

| tiny checkpoint | `ExactF32` | `Bf16` |
|---|---|---|
| three | `144c75402e5d10cc` | `82a22b39f0392f8e` |
| ragged | `559743a0812c82ea` | `6e9c7c3abf39976b` |
| three, truncated to 48 | `ea88c028d86ad065` | `6b8f6505f013e97b` |

The `PLE_BYTES=1` run repeats the 1x16, 64x32, 8x1024 and mixed rows at both precisions, all
equal to the rows above. Each digest was also stable across a repeat run
(`repeat_stable=true`). Because the bits are the same, the bf16 bounds of "Precision" and
the exact-f32 bounds of "Bounds and observed errors" are not re-derived; the ignored
real-checkpoint tests (`embedgemma2_model`, `embedgemma2_e2e_model`, `embedgemma2_mm_model`)
check them again at this head (results in the report of the change, not here).

### The profiler's replica

`src/bin/profile_embedgemma2.rs` runs a hand-written replica of the layer to time its stages
one at a time, and it was stale: against `encode` the `5229a1c` replica reproduced 8 of the
768 embedding values at 1x16, because lane G's small-M split-K GEMM had not been mirrored.
It now mirrors both dispatch sequences: `PROFILE_FUSED=0` is the old sequence, `1` the fused
one, and `PROFILE_OPERANDS=f32|bf16` picks the precision (including the bf16 weights, the
casts and the bf16 attention output). The replica equals `encode` bit for bit, 768 of 768
values at 1x16 and 49152 of 49152 at 64x32, in all four modes, and the stage classes are
renamed for the fused passes ("residual add post-attn (+ pre-ffn norm)" and so on).
`results_report text-fusion` (`src/bin/results_report/`) tabulates its output and
`plot_results text-fusion bench/results/embedgemma2_text_fusion_profile_m5pro.json` (`src/bin/plot_results/`) plots it.

### Profile (text-encoder fusion)

`profile_embedgemma2` with `PROFILE_MODE=stages` times each class of dispatch alone (every
other dispatch of the forward skipped) in the replica above, on the real checkpoint, at
1x16, 64x32 and 8x1024, in the four modes, three passes each (pass 3 in reverse order of
the modes), and a class's time is its minimum over all of them. The tables are in
`bench/results/embedgemma2_text_fusion_profile_tables.txt`, the data in
`bench/results/embedgemma2_text_fusion_profile_m5pro.json`, the plot in
`bench/results/embedgemma2_text_fusion_profile_m5pro.svg`.

**These are not clean timings.** The GPU was shared with other jobs throughout (1-minute
load average 77 to 97, `ioreg` utilisation 82 to 99 % at every workload's start and end),
and the harness flagged 130 of 130 samples of every `ExactF32` run, and of the unfused
`Bf16` ones, as taken with another tessl GPU process present (the pass-1 fused `Bf16`
runs read 0 of 130, which is the luck of the draw, not a quiet machine). The per-pass sums
of the row-pass classes differ several-fold within one mode (for example `ExactF32` unfused
at 1x16 reads 0.54 to 2.44 ms, `ExactF32` fused 1.04 to 6.51 ms), and the 1x16
`ExactF32` rows are unusable: the fused one reads slower than the unfused one in every
class, which is contention, not the kernels. Dispatch counts do not depend on any of this.

What the counts and the byte model say first. Counting traffic in units of one rows x dim
f32 tensor, the row passes of one layer at `ExactF32` moved 13 before (input norm 2, post-
attention add 3, pre-feed-forward norm 2, post-feed-forward add 3, post-per-layer-block add
3) and 11 after (the add with its norm: y and the residual in, the residual and the norm
output out, 4, twice, and the feed-forward add 3). That is a 15 % cut of that traffic
(inferred from the kernels' loads and stores, not measured), and fewer launches
(dispatch table above).

Row passes (every norm and residual add, the per-layer inputs' scale and norm, the `Bf16`
casts), ms per forward, minimum over the passes, with their share of the sum of all
stages. The classes are not like for like (a fused add now carries a norm), so only the
total of the group is comparable:

| workload | mode | row passes before | row passes after | share before | share after |
|---|---|---|---|---|---|
| 64x32 | `ExactF32` | 8.29 ms | 6.65 ms | 7.0 % | 5.6 % |
| 8x1024 | `ExactF32` | 30.38 ms | 24.76 ms | 5.5 % | 4.4 % |
| 64x32 | `Bf16` | 8.56 ms | 4.54 ms | 19.5 % | 12.1 % |
| 8x1024 | `Bf16` | 38.43 ms | 22.80 ms | 19.1 % | 12.7 % |
| 1x16 | `Bf16` | 1.38 ms | 1.01 ms | 19.1 % | 14.0 % |

At 8x1024 `Bf16` the two bf16 casts per layer were 9.43 ms of the 38.43 and are 0.44 ms
(the casts that remain), and `rms_norm` standing alone fell from 6.39 ms to 0.45 ms
(layer 0's input norm). At `ExactF32` the group fell by 18 % at 8x1024, close to the 15 %
the traffic model gives, and the rest of the stages did not move (the GEMMs, the
attention and `gelu*up` read the same in both modes to within the noise: 342 and 347 ms,
138 and 141 ms, 28 and 30 ms at 8x1024). Whole-forward figures of the replica
(`full ms`, min of 3 passes): `ExactF32` 5.14 and 5.11 ms at 1x16, 104.8 and 105.3 ms at
64x32, 549.6 and 546.1 ms at 8x1024; `Bf16` 5.27 and 4.36 ms, 46.5 and 35.1 ms, 191.0 and
173.5 ms. What the fusion can be worth in the whole forward, from the group alone (inferred):
about 1 % of the forward at
`ExactF32` (5.6 ms of 550 ms at 8x1024), where the GEMMs are 62 % and the attention 25 %
of the stage sum, and about 8 % at `Bf16` (15.6 ms of 201 ms), where the casts and the
row passes together were a fifth of it. The `Bf16` forward rests more on elementwise
passes because its GEMMs are 4.2x faster at 8x1024 (82 ms against 342 ms).

### Speed (text-encoder fusion)

No end-to-end speedup is claimed. `bench_paired --evidence --rounds 6 --max-spread 1.25
--clean-env` (lanes: `5229a1c` at `ExactF32` and `Bf16`, a byte copy of the `5229a1c`
binary as the control, and this change at both precisions; the eight workloads of
`bench/results/embedgemma2_final_m5pro.txt`) was refused by its own drift gate twice, on a
host whose 1-minute load average stayed between 66 and 108 (other jobs on the same GPU):

| attempt | finished | series over the 1.25x limit | spread range |
|---|---|---|---|
| 1 | 10:12 | 51 | 1.316x to 6.541x |
| 2 | 11:24 | 55 | 1.292x to 19.254x |

Nothing from either attempt is published (the control lane's own ratio against the base
moved 1.6x to 5.3x). The 10-minute-apart third attempt was not made in this session. What
stands instead is the dispatch count (exact, pinned by test) and the profile above, which
is itself taken on the shared GPU: about 1 % of the `ExactF32` forward and about 8 % of the
`Bf16` forward at 8x1024 are the most the row passes can give (inferred from the stage
sums, not measured end to end). `bench_paired --evidence --rounds 6 --max-spread 1.25`
on an idle host is the run that settles it.

### What is left (text-encoder fusion)

- `gelu(gate) * up` (5 to 6 % of the stage sum at `ExactF32`, 12 to 15 % at `Bf16`) into the
  `gate|up` GEMM: a new GEMM kernel with its own bit-identity question, see above.
- The f32 `a.attn` buffer is dead at `Bf16` apart from its role as split-K scratch; dropping
  it changes the itemised bytes `tests/embedgemma2_memory.rs` and
  `tests/embedgemma2_small_m.rs` assert.
- GEMMs that write bf16 directly would drop the remaining casts but change the bits the
  `Bf16` hashes pin, so they need their own bound.

## Audio tower bf16 and speed

`EmbedGemma2Audio::load_with(rt, st, prefix, cfg, operands)` takes the same
`GemmOperands` as the text encoder and the vision tower; `load` is `load_with(..,
ExactF32)`. The facade passes its one flag to all three models
(`EmbedGemma2Multimodal::load(.., operands)`, so `tessl_embedgemma2_load_multimodal`
and `EmbedGemma2.from_pretrained(.., multimodal=True, operands="bf16")` too):
a `Bf16` facade call runs no f32 GEMM anywhere, and accepts a runtime whose
relaxed f32 GEMMs are on.

**What `Bf16` is.** Every projection of the tower (the input projection; per
layer the two feed-forwards, q|k|v, the attention output, the light conv's
`linear_start` and `linear_end`; the output projection and the embedder's) stays
bf16 on the device as the checkpoint stores it, with no f32 copy
(`weight_bytes`: 587,121,792 bytes, 559.9 MiB, against 1,172,751,488, 1118.4 MiB,
widened to f32), and every GEMM is `gemm_bf16` (bf16 x bf16, f32 accumulation,
f32 output). The operand of each GEMM is stored bf16 by the pass that makes it,
in the pass's existing kernel entry, by a flag the host sets (`ActStore`,
`ega_store` in the shader), so the mode adds no kernel entry and no cast per
GEMM:

| GEMM | its operand is written by | as |
|---|---|---|
| feed-forward up (x2 per layer) | `rms_norm_store` (the pre-norm, with the first linear's input clamp) | bf16 |
| feed-forward down (x2) | `ew_store` (the first linear's output clamp, SiLU, the second's input clamp), into its own bf16 buffer | bf16 |
| q\|k\|v (one GEMM, or three when the input clamps differ) | `rms_norm_then_store` (the attention norm's one to three clamped outputs) | bf16 |
| attention output | `attention_store` | bf16 |
| light conv `linear_start`, `linear_end` | `rms_norm_then_store` (the attention post-norm and the conv pre-norm); `lconv_store` | bf16 |
| input projection | one `cast_f32_to_bf16_into` of the second subsampling conv's output | per forward |
| output projection | one cast of the last layer's residual stream | per forward |
| embedding projection | `rms_norm_then_store` (the output bias, then the embedder's weightless norm) | bf16 |

An operand is rounded to nearest even as the GPU's own cast rounds it. What is
**not** bf16: the residual stream, the norms' and clamps' arithmetic, the two
subsampling convolutions, the light conv's depthwise conv, every GEMM's output
(so q, k and v), the relative-key table and the query scale (host constants),
and the soft tokens. The attention is the f32 `eg2a_attn` kernel and stays so:
the text and vision towers' bf16 tile kernel (`encoder_attn_bf16`) computes
`softmax(q.k)` over a full or windowed row, and this attention has
Transformer-XL relative-position logits, a softcap of 50, a block-shifted window
mask and per-key clamps, so it is not that kernel's shape; it reads the
projection's f32 output and stores its result bf16 (3 % of the f32 forward, the
GEMMs are the cost). A bf16 operand is never stored in place (two bf16 elements
share a word, so a thread storing one races with its neighbour's read of the f32
it overwrites): every `*_store` entry refuses a bf16 output that is also an
input, and `ew` writes its bf16 output to its own buffer (the f32 pass stays in
place). `EmbedGemma2Audio::forward` refuses to hand `gemm_bf16` an f32 operand
(it would cast it into a temporary, a dispatch and an allocation the forward has
none of).

**`ExactF32` is bit for bit what it was.** The shader's f32 arithmetic is the same
code (one branch on the store type per element), and the host's f32 entries are
the old ones (`rms_norm`, `rms_norm_then`, `ew`, `lconv`, `attention` keep their
signatures and call the `_store` entries with `ActStore::F32`). Checked on the
real checkpoint against the tree this work started from (`5229a1c`, built in a
detached worktree): the facade's `media_only` and `mixed_short` embeddings
(two images and two clips; the twelve-item batch) hash the same (FNV-1a over the
bits, `bench_embedgemma2_mm`'s `HASH` line: `c37bcf8a3961f5e5`,
`5d800d44bfda33e1`, f32 mode, before and after), the tower's soft-token error
against transformers on both golden clips is the same to all 17 digits
(`3.8057482766826946e-6`, `2.2770603099052907e-5`), and every number the
existing tiny, kernel, fixture and real-checkpoint tests print is unchanged
(the `--nocapture` output of the fixture, kernel, tiny, `mm_tiny` and
`multimodal_tiny` tests, compared line by line with the same run on `5229a1c`:
every line of the old run is in the new one except the test counts, which grew
with the `Bf16` tests).

**Accuracy.** Tiny towers against the f64 forward (`tests/embedgemma2_audio_tiny.rs`,
both configurations, split and fused q|k|v, every mask pattern), bound fixed from
the first measurement and set at about twice the worst: each stage within `2e-2`
of its largest magnitude (observed worst `1.05e-2`, tiny clip 0 layer 1; wide
`4.7e-3`) and the soft tokens within `4e-2` max abs (observed `1.9e-2`); the exact
path's bounds are `1e-4`, observed `5e-7`, so a `Bf16` run that was secretly exact
would pass these, and `bf16_mode_is_not_the_exact_forward` is the test that shows
it is not (soft tokens `1.9e-2` from the exact ones, `2.7e-6` from the reference
when exact). A clip alone, in a batch, reordered, traced, across a split into two
forwards and with every unzeroed activation poisoned with NaN: bit-identical under
`Bf16` too (the bf16 GEMM sums K in the same order whatever M is). Each store pass
is held to its f32 pass exactly, element by element
(`tests/embedgemma2_audio_kernels.rs`): a bf16 output is the f32 output rounded,
and an output left f32 beside it is the f32 pass's bits.

Real checkpoint (`tests/embedgemma2_audio_model.rs`, opt-in, Apple M5 Pro, sdpa
goldens, snapshot `914f7f89`):

| Check | Bound | Observed |
|---|---|---|
| tower soft tokens, `Bf16`, vs transformers fp32, clips 0 and 1 | cosine `>= 0.9997` | `0.999951`, `0.999922` (max abs over max `\|ref\|` `4.4e-2`, `8.9e-2`) |
| tower stages vs the fp32 goldens | reported, not gated | subsample `2.2e-4` / `3.1e-4`; layers `1.7e-4` to `1.0e-3` through layer 9, then `5.6e-3` / `7.8e-3` (layer 10), `9.2e-3` / `1.7e-2` (layer 11), tower output `2.3e-2` / `4.5e-2` (the relative error grows over the last layers) |
| two 90 s clips (2250 tokens each, 9000 frames, two forwards), `Bf16` vs `ExactF32`, token by token | every token cosine `>= 0.99` | whole clips `0.999904`, `0.999911`; worst single token `0.994434`, `0.996582` |
| clip embeddings end to end (facade, text encoder and both towers `Bf16`) vs fp32 | cosine `>= 0.9999` | `0.99999343`, `0.99998571` |
| the same vs transformers' own bf16 forward | cosine `>= 0.9996` | `0.99980337`, `0.99968055` (transformers bf16 vs fp32: `0.99982002`, `0.99968966`) |
| mean `1 - cos` vs fp32, over the two clips | at most transformers' bf16 (`2.45e-4`) | `1.04e-5` |
| each clip alone vs both in one call (tower, and embedding) | bit-identical (embedding `1e-5`) | bit-identical |
| facade `Bf16` (text, vision and audio), 4 media rows vs fp32 | cosine `>= 0.9999` | images `0.999988`, `0.999988`; clips `0.999993`, `0.999986` (clips were `0.999995`, `0.999994` with the audio tower exact) |
| weight bytes, `Bf16` against `ExactF32` | less | `587,121,792` (559.9 MiB) against `1,172,751,488` (1118.4 MiB): 50.1 % |

The error grows over the last layers and the long-clip worst token is about
`5e-3` from 1, so a consumer that compares individual soft tokens (not pooled
embeddings) sees a bf16 error of that size at the tail of a 90 s clip; the
pooled clip embedding is at `0.99998` or better from fp32 on both golden clips.

### Speed: measured, and refused by the drift gate

The speed of `Bf16` against `ExactF32` on the tower is **not published here**:
`bench_paired` refused every attempt. The benchmark (`bench_embedgemma2_audio`
under `bench_paired --evidence --max-spread 1.25 --clean-env`, 8 rounds, the
lane order reversed on alternate rounds, min over 20 iterations per round, 3
warm-up iterations) has four lanes: the commit before lane I's merge (`433e659`,
the audio tower as it was: exact f32, no store flag), built in a detached
worktree with the benchmark's one `load_with(.., operands)` call replaced by
`load(..)`; a control, a copy of that `433e659` binary run as the second lane
(so against the first lane it should read 1.0, the machine's drift between lanes
shows up as its ratio, and a control ratio over the gate refuses the series);
the head at `ExactF32`; the head at `Bf16`. Workloads: one 1 s clip, 4 clips of 5 s, 8 clips
of 10 s, one clip of `MAX_FRAMES_PER_FORWARD` frames (163.84 s). A ratio series
whose per-round ratio moves by more than `1.25x` between its best and worst
round is not reported; any refused series means the run exits with status 3 and
writes no result.

| Attempt (start, CDT, Oct 16) | Ratio series refused | Spread of the refused series (max / min per-round ratio) | 1-minute load average at start |
|---|---|---|---|
| tower 1, 08:58 | 12 | `1.27x` to `1.85x` | 83 |
| tower 2, 09:20 | 17 | `1.27x` to `18.1x` | 77 |
| tower 3, 09:45 | 20 | `1.71x` to `12.3x` | 113 |
| facade (audio items only) 1, 10:04 | 9 | `2.45x` to `7.16x` | 105 |
| facade 2, 10:27 | 4 (all `Bf16`) | `1.30x` to `1.44x` | 119 |
| facade 3, 10:45 | 9 | `2.5x` to `10.7x` | 73 |

The attempts were at least ten minutes apart, on an Apple M5 Pro that other
sessions' builds, tests and benchmarks were using at the same time (the
1-minute load average read between 41 and 119 at the start and end of the
attempts). The control lane is a copy of the first lane's binary (the tower benchmark: the
`433e659` build; the facade benchmark: the head at `ExactF32`, which is its first
lane; the facade control was copied before the head binary was rebuilt for the
last time; the two files differ in bytes, and that the code is the same, the
source changes in between being comments, is inferred, not checked), so it should read 1.0 against it; its series are among the refused ones in five of the
six attempts (spreads up to `8.3x`, tower 3, 1 s clip), which is how the gate
shows that the machine, not the code, moved. In facade attempt 2 only the `Bf16`
series moved (up to `1.44x`, control and `ExactF32` within the gate): a repeat
on a quiet machine is the way to tell whether that is the machine or the mode.
Nothing about the speed of `Bf16` is claimed. What is measured and does not
depend on timing: the weights are 50.1 % of the bytes (above), and
`tests/embedgemma2_memory.rs` holds the activation allocation of both modes to
one set across consecutive calls.

To reproduce when the machine is quiet: build `bench_embedgemma2_audio`,
`bench_embedgemma2_mm` and `bench_paired` (`cargo build --release --bins`), and
run `bench_paired --rounds 8 --evidence --max-spread 1.25 --clean-env` with one
lane per binary and `BENCH_PRECISION=f32` or `bf16`, `BENCH_ITERS`,
`BENCH_WARMUP` and `BENCH_WORKLOADS=clip_1.0s,batch_4x5s,batch_8x10s,clip_max`
(the facade: `BENCH_WORKLOADS=audio_only,audio_4x5s,audio_8x10s`) in each lane's
environment; the layout of a published result is
`bench/results/embedgemma2_final_m5pro.txt`'s. The per-stage profile is
`profile_embedgemma2_audio` with `PROFILE_OPERANDS=bf16`.

## Video

`MmItem::Video` embeds a video: its sampled frames (what
`embedgemma2_pre::preprocess_video` makes, or the processor's tensors) go through
the vision tower in the call's one batch of pictures, and their soft tokens fill
the video placeholders of the processor's sequence. Raw frames to embedding:

```rust
let cfg = PreConfig::from_snapshot(&snapshot)?;
let video = preprocess_video(&cfg.video, &frames, source_fps)?;            // PreparedVideo
let frames: Vec<ImageInput<'_>> = video.frames.iter().map(PreparedImage::input).collect();
let out = model.encode(&[MmItem::Video(&frames)], None)?;                  // one row, one wait
```

From a file, `tessl::media::video::decode_video_file` samples and decodes the frames and
`preprocess_decoded_video` makes the same `PreparedVideo`; for the H.264 and HEVC streams
VideoToolbox decodes (8 and 10 bit, 4:2:0, 4:2:2, 4:4:4) the frames are byte-equal to the `ffmpeg`
CLI's, which the reference loader's converter is (torchcodec's own flags are unverified), and the embedding of an MP4
equals sentence-transformers' to 8.5e-7 at worst (see "Media decoding", "Video" and "End to end").

### The token layout

Transformers 5.19 (`embedding_gemma2/processing_embedding_gemma2.py`,
`modeling_embedding_gemma2.py`, `video_processing_embedding_gemma2.py`;
verified by reading them and against the `ids.npy` of the reference run):

* The processor expands each `<|video|>` of the text to one block per sampled frame,
  `<boi> <|video|> x S <eoi>` (`replace_video_token`, processing:244-253), with `S`
  the single `num_soft_tokens_per_video` of the video (video processor:286, the same
  for every frame) and no separator between blocks (`"".join`, processing:253), then
  the tokenizer wraps the text in `<bos>` and `<eos>`. With `add_timestamps` on
  (`False` in the class, video processor:194, and in the snapshot's
  `processor_config.json`) every block is preceded by its `mm:ss` timestamp and the
  blocks are joined by spaces (processing:265-268). So

  ```text
  [<bos>, (<boi>, <video> x S, <eoi>) x F, <eos>]        2 + F * (S + 2) ids
  ```

  `EmbedGemma2Config::video_sequence(&[S; F])` builds it from `config.json`'s ids
  (`boi_token_id` 255999, `eoi_token_id` 258882, `video_token_id` 258884); `S` may
  differ per frame there, `MmItem::Video` requires the processor's one `S`.

  Under `add_timestamps` the processor's string is `"00:00 <boi><video>*S<eoi>
  00:01 <boi>..."`: `f"{int(s // 60):02d}:{int(s % 60):02d}"` per frame (the
  existing `timestamp_label`; truncation, not rounding, so 59.9 s is `00:59`, and a
  minutes count past 99 has three digits), then the blocks joined by one space.
  The added tokens split that text, so the stretches between them are tokenized on
  their own: `"mm:ss "` before the first block and `" mm:ss "` before each later
  one, which the snapshot's tokenizer makes `00:00` as five single-digit and colon
  tokens and a `▁` (six ids), and `▁`, five, `▁` (seven) for the others.
  `EmbedGemma2Config::video_sequence_with_timestamps(&tokenizer, &timestamps, &[S; F])`
  builds `[<bos>, label ids, block, label ids, block, ..., <eos>]` that way, through
  `Tokenizer::encode_raw(stretch, false)` and not by building the string, and
  refuses (naming the frame) a timestamp count other than the frame count and a time
  that is not finite, is negative or is past `MAX_TIMESTAMP_SECONDS` (10^9, about 31
  years). The tiny test holds it to `encode_raw` of the whole processor string on a
  tokenizer with merges, so the equivalence of "tokenize the stretches" with
  "tokenize the string" is checked on that tokenizer and, for the real one, by the
  ids below. The
  placeholder is its own id, not the image's. (The processor adds `<|video|>` to the
  tokenizer itself, processing:88-89, with a FIXME to put it in the config; the
  generator asserts that its id is the config's `video_token_id`, which is the one
  the model masks on, modeling:717.)
* The model embeds the ids with every placeholder as `pad_token_id`, then
  `masked_scatter`s the tower's rows over the video placeholders in order and
  unscaled (modeling:802-817), as for images and clips; the tower's rows for all
  videos are split back per video by `num_frames_per_video` (modeling:884-929) and
  `torch.cat`ed, so the `j`-th video placeholder takes the `j`-th soft row of the
  frames laid end to end. Pooling is the mean over every token, soft and boundary
  tokens included, then normalize.
* Frames reach the tower as images do: `pixel_values_videos [F_total, P, 768]`,
  `video_position_ids [F_total, P, 2]` with `P = max_soft_tokens * 9 = 1260`
  (140 tokens a frame at most; a 640x360 frame is resized to 384x720 and makes
  120). The frames of all the videos in a batch are concatenated on one axis. The
  tower runs on all of them at once under padding; tessl cuts them into forwards
  of at most 16,384 padded patches (13 to 15 frames at the sizes below), which
  changes nothing but the GEMM tiling (`video1`'s 32 frames are 3 forwards).
* Which frames: the sampler is `embedgemma2_pre::sample_frame_indices` (see
  [Preprocessing](#preprocessing)). A video of `T` frames at `fps` is
  `int(T / fps)` frames at 1 fps, cut to 32 (uniform) above that. A bare frame array
  with no metadata keeps every frame (sentence-transformers fills in `fps: None`
  for it; the reference generator passes `fps`, `duration` and `total_num_frames`
  and asserts the sampled count).

### The facade

* `MmItem::Video(&[ImageInput])`: the processor's sequence of the frames' soft-token
  counts, which must all be equal (refused naming the first frame that differs).
  No frames is refused. The frames join the call's images in the vision tower's one
  batch, so a call queues one tower call and still waits once, and a video's soft
  rows stay on the device, read by the text encoder's scatter where the tower
  wrote them (by the scatter's rule of one copy per maximal run of placeholder rows,
  one per frame; inferred, the dispatches were not counted).
* `MmItem::Media { ids, images, audio, videos }`: `videos` is one frame list per
  video; the video placeholders of `ids` take the frames of all the videos laid end
  to end, as image placeholders take the images'. This is the entry for ids of your
  own (a prompt, other text between the blocks, several videos in a row). A video
  placeholder with no frames, or frames with no placeholder, is refused as a count
  mismatch (before, any video placeholder was refused: there was no tower).
* `MmItem::TimestampedVideo { frames, timestamps, tokenizer }`: the same video under
  `add_timestamps = true`. The item carries each frame's time in seconds
  (`PreparedVideo::timestamps`) and the snapshot's `&Tokenizer`, and the facade builds
  the ids with `video_sequence_with_timestamps`; the frames are numbered, counted and
  limited as `Video`'s are, and the one-count rule holds. A wrong timestamp count or
  time is refused with 0 allocations and 0 waits, as every refusal is. `PreConfig`
  now reads `video.add_timestamps` (it refused `true` by name while this crate had
  no tokenizer), so a caller picks the item by the flag: `TimestampedVideo` when it
  is on, `Video` when it is off. `MmItem::Video(&[ImageInput])` is unchanged.
* Limits: the frames of all the call's videos count with its images toward the
  vision tower's per-call bounds, `MAX_CALL_IMAGES` (4096) and `MAX_CALL_PATCHES`
  (2^20 padded patches: 800 full 1260-patch frames), not toward the 256 of one tower
  forward: a call past 256 pictures is cut into several tower forwards inside the call
  (see [Pictures and clips past a forward](#facade-follow-ups)). Frames are numbered
  with the images in the tower's messages. The text limit
  (`max_tokens`, 8192) refuses a sequence past it: 32 frames of 140 tokens is 4546 ids.
* Both operand modes, as images: `GemmOperands::Bf16` runs the vision tower and the
  text encoder in bf16, so the video's frames do too.

### The oracle

`tools/embedgemma2_ref/make_mm_reference.py` now also runs two synthetic videos alone
through sentence-transformers (fp32, sdpa, CPU) with the metadata that makes the
processor sample at 1 fps, and saves (prefix `video{n}_`) the raw frames, source fps,
sampled indices and timestamps, `pixel_values_videos`, `video_position_ids`, ids, the
soft tokens, `lm_final`, and the fp32 and bf16 embeddings:

| | frames | size | source fps | sampled | soft tokens a frame | ids |
|---|---|---|---|---|---|---|
| `video0` | 24 | 96x176 | 4 | 6 (indices 0, 4, ..., 20) | 128 | 782 |
| `video1` | 100 | 72x104 | 2.5 | 40, cut to 32 by the cap (0, 2, 5, 7, 12, ..., 97) | 126 | 4098 |
| `video0t` | as `video0`, `add_timestamps` on | | | | 128 | 823 |

`video0t`'s ids are `<bos>`, then per frame the tokenizer's ids of `mm:ss`
(`00:00` ... `00:05`, the digits and `:` as single tokens) and `▁` before the block,
then `<eos>`: 6 ids before the first block and 7 before each other one (the space
between blocks is a `▁` in front of the next label). The 102 image and audio `.npy`
files the generator already wrote are array-for-array identical on regeneration
(`np.array_equal` against the shared reference directory).

### Parity with transformers

On the real checkpoint (`tests/embedgemma2_video_model.rs`, opt-in, same environment
as `embedgemma2_e2e_model`; Apple M5 Pro, 2026-10-16, sdpa goldens, snapshot
`914f7f89`). Bounds are the existing tests' (embedding 1e-4 and cosine 0.99999,
the tower's soft tokens `1e-4` of the largest magnitude) and bf16's, fixed from the
first measurement:

| Check | Bound | Observed |
|---|---|---|
| Native sampling: indices, timestamps (f64) | exact | equal, both videos |
| Native preprocessing: every frame's pixels and positions vs the processor's | bit for bit | 0 of 5,806,080 and 0 of 30,965,760 values differ |
| `video_sequence` vs `ids.npy` | equal | equal (782 and 4098 ids) |
| `video0t`: ids = the golden's labels + the plain blocks | equal | equal |
| `video0t`: `video_sequence_with_timestamps` (snapshot tokenizer; this crate's timestamps, and the golden's) vs `ids.npy` | equal | equal (823 ids, both) |
| Tower soft tokens per frame vs captured (`video1`: 32 frames, 3 tower forwards) | 1e-4 of the largest magnitude | 1.25e-5 and 1.51e-5 |
| Raw frames to embedding, both videos in one call, ExactF32 | max abs `1e-4`, cosine `>= 0.99999` | max abs 3.7e-7 and 1.5e-7, cosine 1.000000000 |
| `video0t` through `Media` with the golden's ids | same | max abs 2.1e-7, cosine 1.000000000 |
| `video0t` through `MmItem::TimestampedVideo` (a config with `add_timestamps` on, native frames and timestamps, the snapshot's tokenizer) | same | max abs 2.1e-7, cosine 1.000000000; vs the `Media` item 0 (bit for bit) |
| Each video alone vs in the call | `1e-5` | 0 (bit for bit) |
| Mixed call: 2 videos, an image, a clip, ExactF32 | same | 1.5e-7 to 3.7e-7, cosine 1.000000000, alone vs in the call 0 |
| Bf16 video vs fp32 | cosine `>= 0.99997` | 0.9999930 and 0.9999968 |
| Bf16 video vs transformers bf16 | cosine `>= 0.9996` | 0.999968 and 0.999986 (transformers' own bf16 vs its fp32: 0.999970 and 0.999986) |
| Bf16 mixed call vs fp32 | cosine `>= 0.99997` | 0.9999885 to 0.9999968; alone vs in the call 0 |

The bf16 bound is 2.6x the worst measured `1 - cos` over the videos, image and clip
(1.2e-5), as the earlier bf16 bounds are derived from the measured values.

Tiny checkpoint (`tests/embedgemma2_video_tiny.rs`, device, no checkpoint): the
facade's video item equals the by-hand composition (`encode_images` of the frames,
the ids written out in the test, `encode_multimodal`) bit for bit in both modes, for
a 5-frame video and a 25-frame one the tower cuts into 2 forwards; reversed frames
and swapped videos change the embedding; a `Media` item with text between the
frames' blocks, an image and two videos of different frame sizes equals the by-hand
composition; a mixed batch of a text, a video, an image, a clip, caller-built ids, a
second video and a text equals each item alone (bit for bit, one forward, poisoned
activations change nothing) and agrees within 1e-5 across several text forwards; one
wait per call; every refusal (no frames, frames of different sizes, a placeholder
count that is not the frames' soft tokens, 4097 frames, a bad frame after good ones, a
sequence past the limit) is raised with 0 allocations and 0 waits and leaves the
facade usable. Timestamps: `PreConfig` reads `add_timestamps` (on, off, absent, not
a boolean); on a tiny tokenizer with merges (`0 0`, then a space glued to a following
`0`, built by `tests/common/embedgemma2_mm.rs::timestamp_tokenizer_json`)
`video_sequence_with_timestamps` equals the tokenizer's ids of the processor's whole
string for stamps crossing every `mm:ss` carry (59.9, 60, 3725.5, three-digit minutes)
and a 32-frame video, and is written out for two cases; `MmItem::TimestampedVideo`
equals the by-hand composition and the `Media` item with those ids bit for bit in both
modes, in one wait, and its labels, times and frame order are observable; its
refusals (wrong timestamp count, NaN, negative, no frames, frames of different sizes,
a bad item after a good one, a sequence past the limit) make 0 allocations and 0
waits. Mutation checks, each failing the tests that should: frames taken in
reverse in the segment walk (4 of 8 tests), no `<eoi>` in the builder (5 of 8), and
no leading space before the later frames' labels (3 of 12 tests of the file).

### Speed

`bench_embedgemma2_video`: 32 synthetic 640x360 frames (resized to 384x720, 120
soft tokens a frame, 3906 ids), min of 15 calls, three consecutive runs
(`bench/results/embedgemma2_video_m5pro.{txt,json,png}`). **Indicative only:** the
machine was shared with other work, the one-minute load average at the workloads'
starts was 30 to 110 on 18 cores (and the GPU is shared too), and the same workload
moved by up to 2.7x between runs. Best of the three runs, ms:

| | preprocess (CPU) | vision tower | whole call (tower + text encoder, 1 wait) |
|---|---|---|---|
| f32 | 50 | 2423 | 3732 |
| bf16 | (same) | 791 | 1488 |

The bf16 tower and call minima were taken in unequally loaded periods (load 80 and
98), so the gap between the two is not the text encoder's cost; the runs' spread is in
the figure. The wait count is 1 in both modes. The tower is the larger part in f32 (best
runs 2423 of 3732 ms; it runs 32 x 1080 patches against the text encoder's 3906
tokens); in bf16 the runs are too disturbed to say.

### Not done

* No `Video` entry in the C ABI or Python (`capi*.rs` and `python/` are another
  piece of work); `MmItem::Video` is Rust-only for now.
* x86 / AVX2 behaviour of the native resize is as unverified as [Preprocessing](#preprocessing) says.

## Pool recycle at allocation

The pool used to return a dropped buffer to its freelist only at a commit. A
buffer dropped after a wait therefore stayed out of the pool until the next
call's wait, and the next call's allocations (which come first) missed it and
made fresh ones: two activation sets live across consecutive calls. `encode`
and the towers worked around it by handing their buffers back before their
wait. The class fix is in the runtime, so it covers every caller (Qwen3.5, user
code, a model that drops after its wait): `alloc_buffer_kind` first runs
`recycle_if_idle`, which releases what has been dropped when

* no command buffer is open (`active_m4` holds none; if the lock is held, this
  call is inside a commit or an encode step on this thread and it skips, as
  `flush_residency` does), and
* the private lifetime event, read through `checked_lifetime_value`, has
  reached the value of the last commit submitted.

Neither is a wait: an allocation while work is queued or running allocates
fresh memory, as before, and the at-commit drains are unchanged. A buffer
dropped while a command buffer is open, or while one it was bound in is still
running, is not reused until its lifetime event passes. The hand-back in
`encode` and the towers stays: it lets the wait itself return the set, so the
buffers are in the freelist before the call returns, whatever the caller does
next.

Measured on the real checkpoint (`profile_embedgemma2`, `PROFILE_MODE=memory-table`,
the same harness binary source on the base `7c7a61f` and with the change, fresh
process per cell, deterministic): the device peak above the model of one
encode, four back to back and three with `synchronize` between is unchanged in
the single and sync-between columns (400.47 / 400.33 / 400.28 MiB f32, 392.50 /
392.33 / 392.28 bf16 at 32x256 / 8x1024 / 2x4096). The back-to-back column
falls to the single figure in every cell: the previous call's small output
buffer, which was recycled one call late, is reused (400.59 to 400.47 at
32x256, 112.88 to 112.62 at 64x32 f32; 110.94 to 110.69 bf16). Every embedding
hash of `PROFILE_MODE=hash` (6 workloads x 2 modes x full and truncated) is
equal between the two builds. `tests/pool_recycle_at_alloc.rs` is the check
that does not go through the hand-back: a set dropped after the wait is served
to the next allocation with no commit between, with the same buffers and the
same device bytes, and the in-flight cases above hold. The decision logic
under a hand-set lifetime event and under a held `active_m4` lock is in
`runtime::alloc_recycle_tests`.

## Facade follow-ups

Three loose ends of the multimodal facade, closed together. Navigation was by `rg`
(the DevMap index is not built for this checkout).

### Timestamped video

`add_timestamps` is native: `PreConfig` reads `video.add_timestamps`,
`EmbedGemma2Config::video_sequence_with_timestamps(&tokenizer, &timestamps, &soft)`
builds the processor's ids with the snapshot's tokenizer, and
`MmItem::TimestampedVideo { frames, timestamps, tokenizer }` embeds the video in one
item. The layout, the exact ids of `video0t` and the embedding against
sentence-transformers are in [Video](#video) (ids equal to `ids.npy`, 823 of 823;
embedding max abs 2.1e-7, cosine 1.000000000, bit for bit the `Media` item with the
golden's ids).

### Pictures and clips past a forward

A tower call used to be refused above 256 images or clips, though both towers already
cut a call into forwards (images by padded patches and `MAX_IMAGES`, clips by frames)
queued behind its one wait. The cap was `check_call`'s, the one place both the
facade and `encode_images` / `encode_audio` pass, so it is lifted there and not
worked around in the facade: `queue_images` / `queue_audio` are still one call, one
`async_scope` and one wait, the soft rows are still the forwards' output buffers read by
the text encoder's scatter on the device, and `PendingImages::into_device` and
`PendingAudio::into_device` already handled many forwards. The audio plan also cuts a
forward at `MAX_CLIPS` clips now (the clip table's size, which a call of 300 short
clips would otherwise exceed in one forward).

What a longer call costs is memory: every forward's uploaded inputs and its soft tokens
are held until the call's wait (the one-wait rule), so a call is bounded by the bytes
they take, refused by name before any allocation:

| Bound | Value | Memory at the bound |
|---|---|---|
| `MAX_CALL_IMAGES` (pictures: images and video frames) | 4096 | a sanity bound beside the next one |
| `MAX_CALL_PATCHES` (padded patches, summed over the forwards) | 2^20 | about 3.4 GiB (`ExactF32`; 3 KiB a patch of pixels, 8 bytes of positions, soft tokens a ninth of the rows) or 1.9 GiB (`Bf16`), beside one activation set |
| `MAX_CALL_CLIPS` | 4096 | a sanity bound beside the next one |
| `MAX_CALL_FRAMES` (summed over the clips) | 2^21 (5.8 h) | about 2.5 GiB (1.25 KiB a frame: 512 bytes of features, 768 of soft tokens) beside one activation set |

`MAX_IMAGES` (256) and `MAX_CLIPS` (256) now name what they always bounded, the size
of one forward; the C ABI (`capi_embedgemma2.rs`) still applies them to `n_images` and
`n_clips` of its own entries, which is the binding lane's to lift. The bounds are above what
the old cap admitted in practice (256 images of 2520 patches are 645,120 padded
patches; 256 clips of 30 s are 768,000 frames), but not above its worst case (256
images of 16,384 patches, 256 clips of 16,384 frames): those calls are now refused
naming the bound, and a caller splits them (each call waits once).

Measured, tiny checkpoint (`tests/embedgemma2_vision_tiny.rs`,
`embedgemma2_audio_tiny.rs`, `embedgemma2_video_tiny.rs`, both operand modes where
stated), all bit for bit unless said:

| Check | Result |
|---|---|
| 300 images in one `encode_images` call (2 forwards: 256 + 44; asynchronous runtime, poisoned activations) vs two calls of 150 | equal, `ExactF32` and `Bf16`; 5 of them vs alone equal |
| 300 clips (ragged lengths, prefix and gapped masks; 2 forwards cut by the clip table) vs two calls of 150 | equal; 5 vs alone equal |
| Facade: 150 items of 2 images and 2 clips (300 pictures, 300 clips) in one call vs two calls of 75 | equal, both modes; 1 wait; 5 items vs alone within the text side's batch bound 1e-5 (the towers' rows are the same bits; the text GEMMs tile 14 rows differently from 2100) |
| Facade: one video of 300 frames vs the by-hand composition with the frames in two tower calls of 150 | equal, both modes; 1 wait |
| Refusals: 4097 images (`MAX_CALL_IMAGES`), 65 images of 16,384 patches (`MAX_CALL_PATCHES`), 4097 clips, 129 clips of 16,384 frames; the boundary itself passes `check_images` / `check_clips` | refused naming the bound; through the facade 0 allocations, 0 waits |

Mutation check: dropping the clip-table cut from the audio plan fails the 300-clip test
(`300 clips; need 1..=256`).

### Memory of consecutive calls, and the hand-back

After lane R the pool recycles a dropped buffer at the next allocation when the GPU is
idle, so consecutive `encode`, vision, audio and facade calls hold exactly one call's
set, the previous call's output buffer included. `tests/embedgemma2_memory.rs` allowed
one output buffer above the first call's peak (`extra <= out_bucket`); it now asserts
`extra == 0` for calls 2 to 4 in both modes. The tiny outputs were under 16 KiB, below
which the device does not charge a buffer's own length, so the tests would not have
seen a missed output at all; they now use batches whose output is 16 KiB or more, and
assert it.

With `recycle_if_idle` stubbed to a no-op (temporarily, in `src/runtime.rs`, reverted),
the tightened tests fail, each by exactly one output buffer (`ExactF32`; the first
mode's failure ends a test):

| Test | Peak above the first call, stubbed | Now | One call's set |
|---|---|---|---|
| `consecutive_encodes_hold_one_activation_set` | 16,384 B | 0 | 36,732,928 B |
| `consecutive_vision_calls_hold_one_activation_set` | 32,768 B | 0 | 6,520,832 B |
| `consecutive_audio_calls_hold_one_activation_set` | 16,384 B | 0 | 802,816 B |
| `consecutive_facade_calls_hold_one_activation_set` | 32,768 B | 0 | 26,181,632 B |

(`Bf16` is 0 now, measured; its stubbed figure was not taken.) The old bound
`extra <= out_bucket` passes the stubbed build, which is why it was loose.

**The hand-back stays.** `encode`, the towers and the facade hand their activations and
uploaded inputs to the runtime before their wait. With the pool fix that is redundant
for back-to-back calls: measured by holding the text encoder's set until after the wait
(the pre-R behaviour of `encode`), `consecutive_encodes_hold_one_activation_set` passes
at `extra == 0`. It is not redundant for a call that starts while its caller has work
open: `recycle_if_idle` does nothing while a command buffer is open or the GPU has
not caught up, so a set that was only dropped after the previous wait is not reused
then. `an_encode_beside_open_work_reuses_the_previous_calls_set` encodes a copy in an
open `async_scope`, then calls `encode`: with the hand-back the call reaches exactly
one fresh output (16,384 B) above the first call's peak; with the set held until the
wait instead, the same test reached 36,470,784 B, a whole second set. So the
hand-back is defence the pool fix does not replace, it costs nothing, and it is kept
(`src/embedgemma2.rs` `queue_impl`, `embedgemma2_vision.rs` and
`embedgemma2_audio.rs` `queue_impl`, the facade's `drop((image_soft, clip_soft))`). The
open-work case was measured for the text encoder only; the towers and the facade use
the same pattern and were not measured held-back.

## Media decoding

`tessl::media` turns an image, audio or video file (or its bytes) into the raw inputs
`tessl::embedgemma2_pre` takes, with no Python and no new crates, through the system
frameworks: ImageIO and CoreGraphics for images, AudioToolbox for audio, AVFoundation for
video. macOS only. The C declarations are this crate's own, written from the SDK headers;
AVFoundation goes through `objc2`'s `msg_send!` and `block2`, which the crate already
depends on.

```rust
let cfg = PreConfig::from_snapshot(&snapshot)?;
let img = media::image::decode_image_file(path, &ImageOptions::default())?;
let prepared = preprocess_image(&cfg.image, &img.raw())?;
let clip = media::audio::decode_audio_file(path, &AudioOptions::default())?;   // mono f32, 16 kHz
let features = preprocess_audio(&cfg.audio, &clip.samples, clip.sample_rate)?;
let video = media::video::decode_video_file(path, &cfg.video, &VideoOptions::default())?;
let out = media::video::preprocess_decoded_video(&cfg.video, &video)?;         // PreparedVideo
```

Every function returns a `MediaError`. A file that cannot be decoded is an error, never an
empty result, and a file that is cut short is an error rather than the part that survived
(below).

| Framework function (declared by hand) | Header | Used for |
|---|---|---|
| `CGImageSourceCreateWithData`, `CGImageSourceCopyPropertiesAtIndex`, `CGImageSourceCreateImageAtIndex`, `CGImageSourceGetStatusAtIndex` | ImageIO `CGImageSource.h` | decode, size and orientation before the decode, completeness after |
| `CGImageGet*`, `CGDataProviderCopyData`, `CGColorSpace*`, `CGBitmapContextCreate`, `CGContextDrawImage` | CoreGraphics `CGImage.h`, `CGDataProvider.h`, `CGColorSpace.h`, `CGBitmapContext.h` | the direct sample copy; the sRGB fallback |
| `ExtAudioFileOpenURL`, `ExtAudioFileWrapAudioFileID`, `ExtAudioFileRead`, `ExtAudioFileGetProperty`/`SetProperty`, `ExtAudioFileDispose` | AudioToolbox `ExtendedAudioFile.h` | decode to `f32` at the file's rate |
| `AudioFileOpenWithCallbacks`, `AudioFileClose` | `AudioFile.h` | decode from bytes |
| `AudioConverterNew`, `AudioConverterSetProperty`, `AudioConverterFillComplexBuffer`, `AudioConverterDispose` | `AudioConverter.h` | the sample-rate conversion of `Resampler::AudioToolbox` (the default converts in `media::resample`, no framework) |
| `AVURLAsset`, `AVAssetReader`, `AVAssetReaderTrackOutput`, `AVAssetTrack` (`loadTracksWithMediaType:completionHandler:`, `preferredTransform`, `nominalFrameRate`), `CMSampleBuffer*`, `CVPixelBuffer*` | AVFoundation, CoreMedia, CoreVideo | video |
| `VTDecompressionSessionCreate`, `VTDecompressionSessionDecodeFrameWithOutputHandler`, `VTDecompressionSessionWaitForAsynchronousFrames`, `VTDecompressionSessionInvalidate`, `CVBufferCopyAttachment` (clean aperture) | VideoToolbox `VTDecompressionSession.h`, CoreVideo | the 8-bit 4:2:0 video decode (`src/media/vt.rs`) |
| `CFData*`, `CFURLCreateFromFileSystemRepresentation`, `CFDictionaryGetValue`, `CFNumberGetValue`, `CFRelease` | CoreFoundation | held by `cf::CfRef`, released on every path |

### What the reference decodes, and what this follows

Verified against the installed sentence-transformers 6.1 and transformers 5.19 (the
`feature-extraction` task of `sentence_bert_config.json`):

- **Image path string.** It reaches the image processor's `fetch_images`, which calls
  `torchvision.io.decode_image(path, mode=ImageReadMode.RGB)` with
  `apply_exif_orientation=False`: libpng and libjpeg-turbo, no colour management, no EXIF
  transpose (traced by wrapping that function while encoding a PNG path). The PIL route with
  `ImageOps.exif_transpose` and `convert("RGB")` is taken for a PIL image and by
  `transformers.image_utils.load_image`. The defaults here follow the first;
  `ExifOrientation::Apply` is the second. torchvision and PIL gave the same bytes on every
  fixture tried, JPEG and CMYK included.
- **Audio path string.** Read from the installed sources: the processor's
  `prepare_inputs_layout` (`transformers/processing_utils.py:739-740`) calls
  `feature_extractor.fetch_audio(path, sampling_rate=16000)`
  (`feature_extraction_sequence_utils.py:385`), which is `load_audio(path, sampling_rate=16000)`
  with `backend="auto"` (`audio_utils.py:198`). `auto` resolves to **torchcodec** when
  torchcodec >= 0.3 is installed and to **librosa** otherwise (`:230-233`); the
  `"torchaudio"` backend (`:269-276`: `waveform.mean(dim=0)`, then
  `torchaudio.functional.resample`) is reached only when asked for. The Gemma4 audio feature
  extractor never resamples or checks the rate: its `__call__` takes the array as it is
  (`models/gemma4/feature_extraction_gemma4.py`). With torchcodec the call is
  `AudioDecoder(source, sample_rate=16000, num_channels=1).get_all_samples()` (`:267`), and
  torchcodec builds its `SwrContext` with `swr_alloc_set_opts2` and **no options**
  (`FFMPEGCommon.cpp`, `create_swr_context_from_layouts`; torchcodec's `main`, read online):
  libswresample at its defaults, its resampler and its mix to mono. With librosa it is
  `librosa.load(path, sr=16000)` (`:279`): soundfile, channel mean, `soxr_hq`. **Which one a
  user gets depends on what is installed**: `sentence-transformers[audio]` is
  `transformers[audio]`, which lists librosa and torchaudio but not torchcodec, so that
  install resolves to librosa; installing torchcodec as well switches it to libswresample.
  **Neither is installed here, nor soundfile, PyAV or soxr** (librosa 0.11, soundfile and
  soxr 1.1 exist in `~/.venvs/voice-mac`, used only to measure, below). So a `.wav` path
  cannot be given to sentence-transformers in this environment, and the audio oracles are the
  `ffmpeg` binary (FFmpeg 9.0.2, the library torchcodec calls; labelled "ffmpeg standing in
  for torchcodec"), `torchaudio.functional.resample`, and the analytic signal the fixtures
  are built from. The end-to-end audio test feeds sentence-transformers the array FFmpeg
  decodes (`-ac 1 -ar 16000`).
  **This module follows the torchcodec branch**: `auto` prefers it when present; its
  algorithm has an oracle on disk that can be matched to the bit; the librosa branch's
  `soxr_hq` cannot be reproduced here (no libsoxr source on disk). That torchcodec adds
  nothing else for a WAV, FLAC or MP3 file (frame chunking, seeking, flushing) is read from
  its source, **not run**: torchcodec is not installed.
- **Video.** `fetch_videos` prefers torchcodec (`VideoDecoder(seek_mode="exact")`); torchvision
  0.29 no longer decodes video (removed in 0.26) and torchcodec is not installed here, so the
  oracle is the `ffmpeg` CLI, which wraps the library torchcodec calls. torchcodec is **not
  installed, so its behaviour below is from its documented design and was not run**; what
  makes the CLI an equivalent oracle is that for even-sized 8-bit 4:2:0 frames both convert
  with libswscale using the frame's own matrix and range, with chroma replicated 2x2 (the
  scaler flags do not matter when the frame is not resized; the CLI half is verified, the
  torchcodec half is inferred). Whether torchcodec applies a rotation flag, and in what order
  with the conversion, was **not checked**; the default here applies it. For even sizes the
  order does not change the bytes (a 90 degree turn or flip maps 2x2 chroma blocks to 2x2
  blocks), except through the width-dependent choice between the two converters (below),
  which the rotated fixtures do not separate. The reference
  loader's metadata is taken in exact seek mode: `num_frames` is the
  packet count, `duration` is `max(pts + packet duration) - min(pts)` in seconds, and `fps`
  is `num_frames / duration`, which is not the header's rate (the end-to-end fixture:
  23.976023976023978 against a nominal 23.976024); the sampler
  (`EmbeddingGemma2VideoProcessor.sample_frames`) runs on those. An untagged stream is
  converted as BT.601 limited, as libswscale does. This follows exactly (below). The
  rotation default (`Rotation::Apply`) is what the CLI and OpenCV do.

### Images

`ImageOptions::default()`: RGB, EXIF ignored, colour left as stored (`ColorHandling::Native`),
16-bit gray as PIL turns it into RGB (`Gray16::PilClip`), at most `MAX_IMAGE_PIXELS`
(checked from the header before anything is decoded).

- **Direct path** (every lossless case measured): the decoded `CGImage` already holds the
  file's samples, which are copied out of its data provider honouring stride, byte order and
  alpha placement, with no colour conversion and no resampling. **Result: byte for byte equal
  to the oracle, 0 differing bytes in all 18 lossless files, as RGB and as RGBA** (PNG 8 and
  16 bit, palette, gray, gray+alpha, RGBA with alpha 0 keeping its colour, TIFF, BMP, the
  first GIF frame, a Display-P3 PNG's raw samples).
- **sRGB context** (CMYK, float, packed or unusual formats, or `ColorHandling::Srgb` on an
  image that is not sRGB): the image is drawn into an sRGB bitmap context and un-premultiplied.
  `ImageInfo::path` says which ran. A CMYK JPEG goes this way and is close to PIL's
  conversion, not equal to it.
- **16-bit**: PIL takes the high byte (`v >> 8`) of 16-bit RGB, RGBA and gray+alpha; pure
  16-bit gray (`I;16`) goes through `convert("RGB")`, which clips to `min(v, 255)`. Measured
  on PIL, reproduced by default; `Gray16::HighByte` takes the high byte of gray too.
- **JPEG** is the only lossy case with an oracle, and is *not* bit-exact: ImageIO's decoder is
  not libjpeg-turbo. Against libjpeg-turbo: 4:2:0 `rgb.jpg` max 43, mean 0.745, 3083 of
  9216 bytes differ; 4:4:4 max 3, mean 0.169; gray max 1, mean 0.057 (bounds fixed in
  `jpeg_differs_from_the_oracle_by_a_measured_bound`: max 48, mean 1.0). The difference is the
  chroma upsampling of 4:2:0. End to end it is 0.9986 cosine (below).
- **HEIC** has no libheif/torchvision oracle here; it is compared with the picture it was
  made from (`sips`): mean 5.10, max 155 (bound: mean below 6), which is the codec's loss.
- **EXIF**: all eight orientations are applied by `orient` when asked, checked against PIL's
  `exif_transpose` on `exif1.jpg`..`exif8.jpg` and a PNG with an orientation tag.

**Cut-short files.** ImageIO decodes cut TIFF and BMP files as complete pictures with
zeros where the data is missing, and a PNG cut inside its last CRC as complete; an image
source's status is "complete" for them. `integrity::check_container` compares the sizes a PNG
(chunks must be whole), BMP (uncompressed pixel array) and TIFF (strip and tile offsets and
counts of the first IFD, classic and BigTIFF) declare with the bytes present. Sweep over 13
formats cut at about every 1/97 of their length (`every_truncation_is_an_error_or_the_whole_image`,
plus HEIC in `media_codecs`): every cut fails, except cuts after a GIF's first frame, which
decode that frame, and cuts that drop only a trailer. Without that check the sweep found
zero-filled TIFF and BMP pictures (measured before it was added).

### Audio

`AudioOptions::default()`: 16 kHz, torchcodec's mix and resampler (libswresample's), at most
1800 s (an error, or the first `max_seconds` with `truncate: true`).

1. **Decode** with `ExtAudioFile` to packed `f32` at the file's own rate and channel count,
   with no rate conversion by Apple. Integer PCM becomes `v / 2^(bits-1)`, so a PCM8, PCM16,
   PCM24, PCM32 or float file already at 16 kHz mono is **bit for bit** the reference's floats
   (WAV, AIFF, CAF and FLAC, from a path and from bytes; 0 differing samples).
2. **Mix and resample** in `media::resample`, a Rust re-implementation of libswresample's
   default conversion (`Resampler::Swresample`, `Downmix::Swresample`; the defaults). Per
   channel, then the mix: `swr_init` sets `resample_first` from `out_ch / in_ch - 1 <
   out_rate / in_rate - 1` with an *integer* channel ratio, which is true for every downmix, so
   a stereo or 5.1 file is resampled channel by channel and mixed at 16 kHz.
   `Downmix::Mean` (librosa, the torchaudio backend) takes the mean in `f32` first and
   resamples the one channel. The two are not the same signal: FFmpeg's default mix for float
   output is `0.7071 * L + 0.7071 * R` (no normalisation), not the mean, so a stereo clip is
   `sqrt(2)` louder (7.7 dB for uncorrelated channels). The matrix is `swr_build_matrix2` with
   `clev = slev = 0.70710678f` and `lfe_mix_level = 0` for FFmpeg's default layout of 1 to 8
   channels (3 channels is FL FR LFE; 4 is 4.0; 5, 6, 7, 8 are 5.0, 5.1, 6.1, 7.1); more
   channels, and layouts the file declares itself, are not modelled (an error for more than 8;
   a file's own channel mask is not read).
   What it reproduces, each checked against the `ffmpeg` binary: a Kaiser (beta 9) windowed
   sinc of `ceil(32 / factor)` taps rounded up to even, `factor = min(out * 0.97 / in, 1)`; the
   reduced ratio's numerator as the phase count when it is at most 1024 (44.1 -> 16 kHz: 92
   taps, 160 phases; 48 -> 16 kHz: 100 taps, 1 phase), else 1024 phases with linear
   interpolation (16000/44101, tested); `index`/`frac` integer position arithmetic; the first
   window centred on sample 0 with the start mirrored about it, and at the end a mirror of
   `(min(left, taps) + 1) / 2` samples from the last one; and the float sum in the order of
   FFmpeg's AArch64 NEON routine (four fused-multiply-add lanes, `faddp` twice, a fused scalar
   tail). A clip too short to fill one window after both mirrors (61 samples at 44.1 kHz) gives
   **no samples, as FFmpeg's does**, and is an error here ("decoded to no samples").
3. **Gapless**: AAC, MPEG-1 MP3, FLAC and the rest are trimmed by AudioToolbox. MPEG-2 and
   2.5 Layer III MP3 (16, 22.05, 24 kHz) are **not**: a 1 s, 16 kHz MP3 decoded to 17280
   frames starting 576 frames early. `mp3.rs` reads the Xing/Info/LAME tag (after an optional
   ID3v2 tag) and the delay and padding are trimmed here (16000 frames, as FFmpeg).
4. `Resampler::AudioToolbox` keeps the earlier design as a comparison: an `AudioConverter` at
   the maximum quality and the `Mastering` complexity, as a second stage. (A first design let
   `ExtAudioFile` resample as its client rate; AAC and Ogg Opus then came out 0.26 samples
   late against FFmpeg, 15.6 dB SNR. Cause not found; decoding natively and converting
   separately removed it.)

**Parity with libswresample** (`tests/media_resample.rs`, fixtures from
`scripts/gen_resample_fixtures.py`, goldens from `ffmpeg -i x.wav -ac 1 -ar 16000 -f f32le`,
FFmpeg 9.0.2 on Apple silicon): **44 inputs, 163,513 output samples, 0 differing samples, max
0 ULP**, for every case below, through `convert_interleaved` and through `decode_audio_file`
(an AudioToolbox decode at the native rate, from path and from bytes):

| axis | cases (verified, all bit-equal) |
|---|---|
| input rate (-> 16 kHz) | 8, 11.025, 22.05, 32, 44.1, 48, 96 kHz mono; 8, 22.05, 44.1, 48, 96 kHz stereo; 44.1 and 48 kHz 5.1; 44.101 kHz (1024 phases, linear interpolation) and 12.345 kHz; 16 kHz (mix only); 8 kHz up to 16 kHz and 44.1 up to 48 kHz |
| channels | mono, stereo, 3, 4, 5, 5.1, 7, 8 (the mix); `Downmix::Mean` on stereo against FFmpeg's float `pan` mean, and 6 and 8 channels against an f32 sum / n |
| length | 1, 2, 31, 32, 33, 61, 62, 63, 64, 93, 200, 1001 samples, odd lengths, 0.25 to 3 s |
| signal | log sine sweeps up to 0.45 of the input Nyquist rate (through the transition band), an impulse, a step, white noise |
| chunking | the output is identical for any split of the input (random chunk sizes, including empty ones) |

Bit equality is reached, so there is no ULP bound to state beyond 0. It depends on the
operation order, and that is measured, not assumed: swapping the NEON order for plain
C-order sums (what an x86 build of FFmpeg does for its scalar path; **its SIMD order was not
read or run**) leaves 39,610 of the 48,000 samples of the 3 s noise clip different, up to
49,427 ULP (at near-zero samples), and the same with fused adds still differs; the NEON order
with the fused tail is the one that matches. So on x86 FFmpeg's own output differs from this
module's, and from the machine it ran on, at the last bits; **the torchcodec reference's
last bits depend on its CPU**. **Unverified**: that torchcodec's own call sequence (frame
chunking, its output buffers) changes nothing; the output was shown independent of chunking
here, and `swr_get_out_samples` bounds are loose enough not to truncate the flush.

The earlier tone fixtures (0.4 s of the signal `tone`, all below 5.4 kHz) against the analytic
truth, FFmpeg's decode with a float channel mean, and `torchaudio.functional.resample`
(`sinc_interp_hann`), SNR in dB of the interior (10 ms edges removed; with the edges in
brackets). "ffmpeg" for the stereo files is the earlier golden, whose `pan` stage quantises the
mix to 16 bits (87.8 dB is that floor; `tests/media_resample.rs` has float goldens); the mono
ones are equal to the bit. "was" is the AudioToolbox converter, the default before this change:

| input | this vs truth | this vs ffmpeg | this vs torchaudio | was vs truth | was vs ffmpeg | was vs torchaudio |
|---|---|---|---|---|---|---|
| 44.1 kHz stereo | 92.0 (48.3) | 87.8 | 66.1 (42.8) | 86.1 (46.7) | 83.8 | 66.1 |
| 48 kHz stereo | 92.5 (48.2) | 87.8 | 66.1 (42.5) | 86.1 (46.3) | 83.8 | 66.1 |
| 22.05 kHz mono | 90.3 (52.9) | equal | 66.9 (52.2) | 91.8 (58.9) | 93.0 | 66.8 |
| 8 kHz mono | 90.8 (42.1) | equal | 51.3 (36.0) | 69.7 (41.1) | 69.8 | 51.3 |

The output length equals FFmpeg's (6400 samples for 0.4 s) and torchaudio's, except that
torchaudio and `librosa` give one more sample at some rates. The edges are FFmpeg's (mirrors),
which are farther from the analytic truth than Apple's were at 22.05 kHz (52.9 against 58.9
dB with the edges), and closer to FFmpeg's, being the same.

**The other loaders, measured** (`tools/embedgemma2_ref/audio_loader_gap.py`; librosa 0.11.0 and
soxr 1.1.0 from `~/.venvs/voice-mac`, torchaudio 2.11.0 from `ml-2026-10`; not part of any
test): against this module's output, band-limited tones (stereo against the float-mean
golden, so 87 dB is the golden's floor), SNR over the interior:

| input | librosa `soxr_hq` | torchaudio |
|---|---|---|
| 8 kHz mono | 101.4 (36.1 whole) | 51.3 (36.0) |
| 22.05 kHz mono | 93.7 (51.1) | 66.9 (52.2) |
| 44.1 kHz stereo | 86.9 (42.2) | 66.1 (42.8) |
| 48 kHz stereo | 87.0 (42.3) | 66.1 (42.5) |

On the 10-sweep set of `media_resample/` (which run to 0.45 of the input Nyquist rate, so
they carry energy above 8 kHz that each converter attenuates or folds differently) the
agreement is 25 to 38 dB whole-signal for both. So on band-limited material the librosa branch
(`soxr_hq`) is within about 1e-5 of the swresample output away from the first and last 10 ms
(different edge conventions), and the torchaudio branch is 51 to 67 dB from it; for
material with energy near or above 8 kHz, and for the first and last 10 ms, the three differ.
A user whose install resolves to librosa gets the first row of that table's distance, not
zero: **this module does not reproduce `soxr_hq`**.

Lossy and compressed inputs, 1 s fixtures decoded with the channel mean (the goldens are
`pan`-then-resample, hence the 16-bit floor on stereo), SNR against FFmpeg's decode and
against the signal the encoder was given (the codec's own error against the signal is the
larger part: FFmpeg's own MP3 decode is 26.0 dB from it). With the resampling now the same as
FFmpeg's, what remains is the codec decoders' difference:

| file | frames | vs ffmpeg | vs the signal |
|---|---|---|---|
| 44.1 kHz stereo AAC (m4a) | 16000 | 138.8 | 33.2 |
| 44.1 kHz stereo MP3 | 16000 | 92.6 | 26.0 |
| 44.1 kHz stereo FLAC | 16000 | 87.7 (the golden's floor) | 52.2 |
| 16 kHz mono MP3 (LAME tag trim) | 16000 | 89.5 | 26.0 |
| 16 kHz mono AAC | 16000 | 135.9 | 30.9 |
| 48 kHz Ogg Opus (16 kHz content) | 16000 | 14.6 (see below) | 14.5 |

**MP3 and AAC stay on Apple's decoders** (the owner's decision), and their bounds against
FFmpeg's `mp3float` and `aac` decoders, default mix, on the files of
`media_resample/` (0.5 s of a 44.1 kHz stereo sweep encoded by ffmpeg at 128 kb/s), measured
after the change and fixed in `aac_and_mp3_meet_their_measured_bounds_against_ffmpegs_decode`:

| codec | whole-signal SNR | interior SNR (1024 samples off each end) | max abs difference |
|---|---|---|---|
| MP3 (libmp3lame) | 95.3 dB | 95.4 dB | 3.7e-5 |
| AAC (ffmpeg `aac`, mp4) | 56.7 dB | 101.8 dB | 1.1e-2 |

The AAC difference is in the first and last frames: at the native 44.1 kHz, before any
resampling, the first 4410 samples are 59 dB and the last 4410 are 42 dB from FFmpeg's while the
rest is 96 to 98 dB, so it is the decoders' treatment of the priming and the final frame
(inferred: the cause inside the two decoders was not isolated), not the resampler. End to
end, MP3 is 9.7e-6 max abs, cosine 0.999999996; AAC is 7.1e-4, cosine 0.999988 (below).

**Ogg Opus: AudioToolbox does not lag libopus; FFmpeg's native Opus decoder does.** The
earlier claim in this section (AudioToolbox "two 48 kHz frames behind libopus") compared
AudioToolbox with FFmpeg's *default* Opus decoder, which is not libopus: `ffmpeg` picks its
own `opus` decoder ("opus (native)", from the stream mapping line), so that is what torchcodec
runs too. Measured against `ffmpeg -c:a libopus`, at 48 kHz so that no conversion is involved
(`tests/media_resample.rs`):

| clip (0.6 s, 48 kHz mono, libopus encode) | AudioToolbox vs libopus | native vs libopus | AudioToolbox vs native |
|---|---|---|---|
| SILK, narrow-band (`-application voip -cutoff 4000`) | lag 0, **identical** (3025 dB is the float floor) | 4 samples behind | 15.1 dB |
| SILK, wide-band (`-cutoff 8000`) | lag 0, **identical** | 2 samples behind | 19.8 dB |
| CELT (`-application audio`) | lag 0, 82.3 dB | lag 0 | 138.7 dB |

(The committed tone file of `fixtures/media/` is the same: AudioToolbox against libopus is
lag 0 at 92.8 dB over all 48000 samples; ffmpeg's native decoder is 1 to 2 samples behind it in its
first 18000 samples (1 to 2 behind) and identical to AudioToolbox's after that.) The length is libopus's (the
pre-skip is removed: 312 samples in these files, `OpusHead`), the alignment is libopus's, and
AudioToolbox's SILK is libopus's to the bit while its CELT is FFmpeg's native decoder's
(138.7 dB) and 82 dB from libopus's. So there is **no pre-skip or priming bug to fix**: the
2-sample lag the earlier measurement saw is FFmpeg's native decoder's own (its SILK
resampler out of SILK's internal rate delays the output; 4 samples at narrow-band, 2 at
wide-band, 0 in CELT; the cause inside FFmpeg is inferred from that pattern, not read). Against the
reference this means: if the reference is libopus (soundfile via librosa), AudioToolbox is
exact for SILK and 82 dB for CELT; if it is torchcodec (FFmpeg's native decoder), AudioToolbox
is exact (138 dB) in CELT frames and 15 to 20 dB from it in SILK frames, because the native
decoder is late. Reproducing FFmpeg's SILK output would need its decoder, not a shift: the best
shift gives only 28 dB. An Ogg Opus file in SILK or hybrid mode (speech at low bit rates) is
therefore the one input of this module that does not match a torchcodec reference closely.

WavPack is rejected by AudioToolbox (`'typ?'`) and is an error here.

**Cut-short files.** AudioToolbox returns what a cut WAV, AIFF or CAF still holds as the
whole (a WAV cut in half decoded 3189 of 6400 samples with `noErr`). `containers::check_audio`
compares the declared size of the `data`/`SSND` chunk, the top-level MP4 boxes, and an MP3's
Xing/Info stream size with the bytes present (streamed WAV `0xFFFFFFFF` and CAF `-1` sizes are
accepted); FLAC's STREAMINFO sample count is compared with what was decoded (a cut FLAC fails
with OSStatus -39 anyway). Every cut tried is an error (10 shapes, bytes and path).

### Video

`VideoSource::open` reads the **compressed** samples (no decoder) and returns the exact frame
count and each frame's presentation time, in presentation order, then derives the figures the
reference loader would (see "What the reference decodes"): `info()` gives `total_frames`, `fps`
(torchcodec's exact-mode `average_fps`, `total_frames / duration`), `duration`
(`max(pts + packet duration) - min(pts)`), `nominal_fps` (the header's, kept for reference),
`timestamps`, `orientation` and the stream's colour tags. `decode_frames(&indices)` returns
frames in the order asked (duplicates allowed); `decode_sampled(&VideoConfig)` samples as
`EmbeddingGemma2VideoProcessor.sample_frames` does (`max(1, int(duration * fps))` frames at
`min(total - 1, int(i * average_fps / fps))`, then the `max_frames` cap with its overflow
strategy). Bounds: `VideoOptions::max_frames` (4096), `max_bytes` (4 GiB), the track-load
timeout (60 s). Autorelease pools wrap the per-frame loops and a `Reader` guard cancels a
reader still reading when it is dropped.

**Colour: why VideoToolbox is driven directly.** `ColorPath::Swscale` (the default) gives the
reference's RGB byte for byte for 8-bit 4:2:0 H.264 and HEVC. AVFoundation's own decode cannot:

- `AVAssetReader` asked for 420v/420f (the decoder's native planes) returns the decoder's
  luma, but **different chroma**: the share of Cb/Cr samples that differ from FFmpeg's decode
  depends on the content (52 to 86 percent in the clips measured in this lane, for example
  3843 of 4608 Cb samples of `h264_709_tv_64x48`, by up to 45 levels; an earlier lane
  measured 8 to 15 percent on other content, which was not re-measured). **The cause is a chroma re-siting**, found by regressing the reader's chroma on
  FFmpeg's neighbouring chroma samples (see "Why AVAssetReader's chroma differs" below):
  the reader moves the chroma to the `BottomLeft` location with a smooth three-tap kernel
  (about 0.06, 0.62, 0.31 for a quarter-sample shift), and a stream already tagged
  `BottomLeft` comes out with 0 differing samples. Asked for BGRA it
  differs from the oracle in 39941 of 55296 bytes (largest 151, mean 3.67; `h264_709_tv_64x48`).
- VideoToolbox itself is exact: FFmpeg's `-hwaccel videotoolbox` equals its software decode, so
  the difference is in what `AVAssetReader` does after it. So `AVAssetReader` with no output
  settings supplies the compressed samples in decode order, and `src/media/vt.rs` decodes them
  with a `VTDecompressionSession` (`VTDecompressionSessionDecodeFrameWithOutputHandler`, one
  handler block per sample, destination format 420v/420f, the clean aperture applied for
  cropped streams such as 66x38 coded in 80x48). **The Y, Cb and Cr planes are byte-equal to
  FFmpeg's on every fixture** (`the_decoders_planes_are_the_planes_ffmpeg_decodes`: H.264 and
  HEVC, BT.601, BT.709 and untagged, limited and full range, odd sizes, cropped, rotated).
- The planes are converted by `media::yuv::nv12_to_rgb24` (below), then rotated.
- The stream's layout (chroma format and bit depths) is read from its own `avcC` (the first
  SPS) or `hvcC`, because a requested pixel format says nothing about the stream
  (`src/media/codec_config.rs`). The decoder is asked for the matching one of the twelve
  bi-planar formats (8-bit `420v/420f/422v/422f/444v/444f`, 10-bit `x420/xf20/x422/xf22/x444/xf44`).
- **There is no silent fallback any more.** A stream that cannot be converted exactly is
  refused by name with `MediaError::Unsupported`, which says why and points to
  `ColorPath::AvFoundation`; only that explicit request decodes with AVFoundation's own
  conversion (`Conversion::AvFoundation` in the result says so). Refused: a codec other than H.264
  and HEVC (MPEG-4 Part 2), 12-bit and monochrome video, a matrix other than BT.601, BT.709,
  BT.2020 non-constant, FCC and SMPTE 240M, and shapes libswscale does not define (below).
  Before, those decoded to BGRA by AVFoundation: a Main 10 HEVC fixture was 4.06 mean abs away from
  the oracle (still measured, `ten_bit_through_avfoundation_is_not_the_oracle_and_is_never_the_default`).

**The conversion (`src/media/yuv.rs`).** libswscale reaches `rgb24` through different routines
depending on the sample format, and each was found by black-box comparison with the `ffmpeg`
binary (the module's provenance note lists the experiments in order; no libswscale source was
read):

| input | what libswscale computes | notes |
|---|---|---|
| 8-bit 4:2:0 and 4:2:2 | chroma replicated over 2 luma samples (and 2 rows for 4:2:0), then a table converter, or on aarch64 a NEON one when the width is a multiple of 16 and the height is even | `Flavor::Aarch64` (default) chooses as FFmpeg does on Apple silicon, `Flavor::Portable` is the table converter always; the two differ by up to 3 levels |
| 8-bit 4:4:4 | a direct sum of 13-bit coefficients in 32-bit integers at 22 fractional bits, which **wraps** (a blue of 512 or more comes out 0) | no table, no NEON, no flavor |
| 10-bit 4:4:4 | the same at ten bits | |
| 10-bit 4:2:2 | luma `(v+2)>>2` (not clipped, so it can be 256) and chroma `min((v+2)>>2, 255)`, then the 8-bit table converter | |
| 10-bit 4:2:0 | the chroma rows are first resampled vertically to luma rows by the scaler's filter, then as 4:2:2 | the only class that depends on the scaler's flags: `ChromaFilter` |

All start from the 16.16 coefficients `round(k * 65536 * 255 / 224)` of the matrix's public
`Kr`, `Kb` (BT.709 117489, 138438, 13975, 34925; BT.601 104597, 132201, 25675, 53279; BT.2020nc
110014, 140363, 12277, 42626; FCC 104448, 132798, 24759, 53109; SMPTE 240M 117579, 136230,
16907, 35559) and use their 13-bit rounding. FCC and SMPTE 240M, which lane V had to refuse, are
exact once the NEON converter's two negative green coefficients are rounded as signed values
(half up) and not as the negation of the rounded magnitude: that tie rule was the full-range G
mismatch (81106 and 97789 bytes of 2^24).

**Verified against the `ffmpeg` binary (9.0.2, aarch64)**, all with 0 differing bytes:

- every one of the 2^24 `(Y, Cb, Cr)` triples, both ranges, five matrices, 8-bit 4:2:0 on both
  converters, 4:2:2 and 4:4:4 (`exhaustive_against_a_live_ffmpeg`,
  `exhaustive_8_bit_422_and_444_against_a_live_ffmpeg`, `--ignored`, need `ffmpeg`);
- **every one of the 2^30 ten-bit `(Y, Cb, Cr)` triples** for 4:4:4 (3,221,225,472 bytes per
  matrix and range, 12 combinations: `exhaustive_10_bit_444_against_a_live_ffmpeg`), and for
  10-bit 4:2:2 every distinct combination of the rounded 8-bit values the converter sees
  (`exhaustive_10_bit_422_against_a_live_ffmpeg`, 12 combinations);
- 10-bit 4:2:0 under both filters at every even height from 8 (bilinear) or 14 (bicubic) to 120 at width 34, and at
  32x64, 640x480, 1280x720, 1920x1080, 1918x1080 and 3840x2160
  (`ten_bit_420_filters_against_a_live_ffmpeg_at_many_heights`);
- the committed goldens: random planes of 34 format x matrix x range x size classes,
  including odd sizes (`tests/media_yuv_formats.rs`, hosted).

Run on 2026-10-17: the five `--ignored` tests (`exhaustive_against_a_live_ffmpeg` and the four
of `media_yuv_formats`) passed, 0 differing bytes.

**The 10-bit 4:2:0 filter.** The scaler resamples the half-height chroma to luma rows with a
filter that depends on its flags: bilinear `(3072, 1024)/4096`, or the `ffmpeg` CLI's default,
bicubic `(-115, 985, 3572, -346)/4096` and its mirror, both with edge rows measured. Both are
exact against the CLI run with the matching `-sws_flags`, for even heights from 14 (bicubic)
or 8 (bilinear); below that the scaler's filter shrinks and the picture is refused by name
(`YuvError::TooShort`). **Which flags torchcodec passes is not established**: its source was not
read (the brief forbids it for FFmpeg and it was not downloaded, and it is not installed), and
the only thing written down here is a recollection that it asks for bilinear scaling, which is
unverified. `VideoOptions::chroma_filter` defaults to `Bilinear` on that recollection and
`Bicubic` selects the CLI's own; the reference decode in the end-to-end script passes
`-sws_flags bilinear` to agree with that default, so the end-to-end case shows tessl equals
the CLI under the chosen flag, not that torchcodec uses it. If torchcodec's flag is bicubic,
only 10-bit 4:2:0 changes (about 65 percent of its bytes differ between the two filters), and
`ChromaFilter::Bicubic` is already exact.

**Odd sizes.** H.264 and HEVC cannot code an odd width or height in 4:2:0, or an odd width in
4:2:2 (the conformance window crops in chroma units), so odd shapes occur only in 4:4:4 (any
size) and 4:2:2 (odd height), and both are exact. libswscale leaves the last output column of
an odd-width 8-bit 4:2:0 or 4:2:2 picture **unwritten** (a destination pre-filled with 0xAB keeps
it) and uses a non-periodic vertical filter for odd-height 4:2:0; neither is a defined output,
so the converter refuses those shapes by name (`YuvError::Size`).

**Rotation order.** The CLI's autorotation turns a 4:2:0 picture's planes before the
conversion (the converter's width rule, and the 10-bit filter, see the upright picture) and a
4:2:2 or 4:4:4 picture's RGB after it. On the 48x40 rotated fixture the two orders differ in
11185 of 11520 bytes of 8-bit 4:2:0, so `media::video` follows the CLI. torchcodec's order is
unknown.

**Why AVAssetReader's chroma differs.** Found in this lane. With the reader asked for
`420v`, luma is the decoder's (0 differing samples) and chroma differs in 52 to 86 percent of
samples of the clips measured, with the pixel buffer tagged `CVImageBufferChromaLocation = Left`
(the stream's own). Regressing the reader's Cb on FFmpeg's Cb in a 3x3 window gives vertical
weights `(0.062, 0.625, 0.312)` and zero horizontal weights, residual rms 0.31 to 0.39 (about
the rounding noise of an 8-bit result); `(1, 10, 5)/16` with truncation reproduces 96 percent
of the samples exactly (243 of 6144 differ; the remaining arithmetic was not found). The
centroid is 0.25 chroma rows, half a luma row: the move from `Left` (vertically between two
luma rows) to `BottomLeft` (vertically on the lower one). The same experiment on x264 streams
tagged each chroma location:

| stream tagged | reader's chroma against FFmpeg's |
|---|---|
| `bottomleft` | 0 differing samples |
| `left` | vertical shift of a quarter chroma row, kernel above, no horizontal change |
| `center`, `topleft` (the reader reports `Center` for both) | the same kernel on both axes |
| `top`, `bottom` | horizontal shift only |

So the reader (inferred: in its pixel-transfer step behind `AVAssetReader`) resamples chroma to
the `BottomLeft` siting; the VideoToolbox session driven directly does not. Verified: the
relation above, on H.264 and HEVC (`target/v2` scratch scripts, not committed). Inferred: that
this is a siting conversion rather than another filter; not reproduced: the reader's exact
arithmetic, and why `top`/`bottom` shift only horizontally. Earlier lanes ruled out whole-sample
shifts, deblocking, intra-only streams and `AVVideoColorPropertiesKey`; this is the fractional
shift they could not see. It does not change the design: decoding through VideoToolbox is
exact without needing the reader's resampling modelled.

**Parity per fixture class** (`tests/media_video_exact.rs`, 13 tests, local-only because
VideoToolbox is not available to the hosted sandbox; frames are compared to the `ffmpeg` CLI's
decode, `-pix_fmt rgb24`, autorotate; goldens from `scripts/gen_media_video_exact_fixtures.py`
in `tests/fixtures/media_video/`, 0 differing bytes in all; 10-bit 4:2:0 against the
`-sws_flags bilinear` golden, and again against the CLI's default under
`ChromaFilter::Bicubic`):

| class | fixtures |
|---|---|
| H.264 8-bit 4:2:0: BT.709/601 limited, full range, untagged, NEON and table widths, cropped | `h264_709_tv_64x48`, `h264_601_tv_66x38`, `h264_709_pc_80x48`, `h264_601_pc_50x34`, `h264_untagged_64x48` |
| HEVC 8-bit 4:2:0 | `hevc_709_tv_64x48`, `hevc_709_pc_96x64`, `hevc_601_pc_34x26` |
| HEVC 10-bit 4:2:0 (Main 10): 709, 601 full, 2020, FCC | `hevc_main10_64x48`, `hevc_p10_601_pc_66x38`, `hevc_p10_2020_tv_64x32`, `hevc_p10_fcc_pc_50x34` |
| HEVC 4:2:2, 8 and 10 bit, odd heights (33, 17) | `hevc_422_*`, `hevc_p210_*` |
| HEVC 4:4:4, 8 and 10 bit, odd 33x17 | `hevc_444_*`, `hevc_p410_*` |
| FCC and SMPTE 240M, both ranges | `hevc_fcc_*`, `hevc_240m_*`, `hevc_444_fcc_pc_33x17`, `hevc_p410_240m_tv_33x17` |
| rotation 90, 180, 270 and horizontal and vertical flips of every layout | `*_rot90`, `*_rot180`, `*_rot270`, `*_hflip`, `*_vflip` |
| `colors.mp4`, `quad_rot90.mp4` (`media_codecs`) | zero differing bytes |

The Y, Cb and Cr planes VideoToolbox returns equal FFmpeg's for all six HEVC layouts
(`the_decoders_planes_are_the_planes_ffmpeg_decodes`: 8-bit `nv12`, `nv16`, `nv24` and 10-bit
`p010`, `p210`, `p410` against the `ffmpeg` CLI, 0 differing samples; the 10-bit planes carry
their value in the high ten bits, and the low six are checked to be zero).
**VideoToolbox on this machine (M5 Pro) does not decode H.264 High 10, 4:2:2 or 4:4:4** (it
refuses at open), so those H.264 shapes are not decodable here by this path and are not in the
tests; the HEVC ones are. 12-bit and monochrome HEVC and Motion JPEG (a codec AVFoundation
decodes) are refused by name; MPEG-4 Part 2 is refused by AVFoundation itself, which cannot
decode it under either colour path.

Sampling and timing agree with the reference too: `total_frames`, `duration`, `fps` and every
timestamp are bit-equal to the `ffprobe`-derived figures, and the indices for four sampling
settings (fps 1 with the 32 cap, fps 5 with a 4 cap under `uniform` and under `truncate`, no
fps with a 3 cap) equal the processor's `sample_frames` run on those figures, on every
fixture (12 files, 48 settings). The end-to-end file has a nominal 24000/1001 rate whose exact-mode
`fps` is not the header's to the last bits; sampling at 1 fps picks frames 0, 23, 47 of 80.

- **Timestamps** are track time with the edit list applied, as FFmpeg and OpenCV report: an
  H.264 file with B-frames has media time starting at +2 frames, which the edit list shifts
  to 0. The first pass's media times are shifted by the first decoded frame's time, and the
  decode checks every frame against it to 1e-6 s, so a track whose edit list drops or reorders
  frames is an error, not a misaligned clip.
- **Rotation**: `preferredTransform` becomes an EXIF orientation (1 to 8; a 90 degrees
  counter-clockwise display matrix is 8). `Rotation::Apply` (default) returns upright frames,
  turning 4:2:0 planes before the conversion and 4:2:2 and 4:4:4 frames after it, as the
  `ffmpeg` CLI's autorotate does (0 differing bytes on `quad_rot90.mp4` and the rotated and
  flipped fixtures of every layout); `Rotation::Ignore` returns frames as stored,
  equal to `-noautorotate`. A transform that is not a right-angle rotation or flip is refused
  when applying. OpenCV's own colour conversion is 9.7 mean from FFmpeg's on the quadrant
  file, so it serves as a rotation oracle only.
- **Preprocessing**: `preprocess_decoded_video` runs `preprocess_video` with its own
  sampling off, so frames sampled while decoding are not sampled twice, and restores the true
  source indices and `index / fps` timestamps. Equal to preprocessing every decoded frame in
  one call (`frames`, `indices`, `timestamps`: tested, with fewer frames decoded than the
  video holds).
- **Cut-short files**: top-level boxes are checked as for M4A (a cut `mdat` of a file with
  `moov` first would otherwise decode fewer frames without an error). 1287 cuts of three
  fixtures, every one an error.
- **Release of CoreVideo / CoreMedia objects**: a `Lock` guard unlocks every pixel buffer, the
  session is invalidated in `Drop`, and `leaks --atExit` reported 0 leaks for 0 bytes on
  `media_video_exact` (13 tests, 19411 nodes malloced) and `media_codecs_leaks` (19750 nodes),
  both run with `--test-threads=1`.

### Hosted job: what runs and what cannot

The hosted job's sandbox denies `iokit-open`. ImageIO's HEIC decoder and AVFoundation's H.264
and HEVC decoders go through VideoToolbox, which opens IOKit: under the sandbox they fail, loud,
never silently ("pixel data is shorter than its size", "no frame could be decoded"; verified by
running the tests under the job's profile). So `tests/media_decode.rs` (21 tests: every image
and audio format, the cut-file and bad-input refusals for all three kinds, no codec hardware)
and `tests/media_leaks.rs`, `tests/media_resample.rs` (17 tests: the converter against the `ffmpeg`
binary, AAC, MP3 and Opus; no device) and `tests/plot_results_resample.rs` are on
`scripts/hosted_tests.txt` together with the 14 unit tests of `media::`; `tests/media_codecs.rs`
(HEIC, H.264, HEVC: 8 tests), `tests/media_codecs_leaks.rs` and `tests/media_video_exact.rs`
(13 tests) are local-only (`cargo test --release --test media_codecs`) and fail in the sandbox by
construction, which is what `hosted_completeness` needs (a test that passes there must be
listed). The converter on its own, with no codec, is hosted: `tests/media_yuv_exact.rs` (5
tests, 8-bit 4:2:0) and `tests/media_yuv_formats.rs` (5 tests, every other format), plus their
`--ignored` exhaustive ones that run `ffmpeg`.

### Leaks

Every Core Foundation object is held by `cf::CfRef` and AudioToolbox objects by `ExtFile` and
`Converter` guards. `tests/media_leaks.rs` decodes in loops (images from path and bytes, the
sRGB context, audio resampled, MP3, AAC, and the refusing paths: a cut TIFF, a cut WAV, a
length-limit error) and bounds the growth of the resident set (`task_info`, since `ps` is
not allowed in the sandbox) to 256 KiB for the least-growing of three batches of 100 calls,
after warm-up. First-call growth of 2.5 MiB in a resampling decode is allocator and framework
cache and is flat afterwards (a plateau, which the minimum over batches tolerates and a leak
does not). A mutant that skips `CFRelease` fails it at once (2.9 MiB over 100 calls).
`tests/media_codecs_leaks.rs` does the same for HEIC and video (open, decode 12 frames,
bytes with a temporary file, a refused file), and asserts that no temporary file is left.
**`leaks --atExit` reported 0 leaks for 0 bytes** on all four test binaries: `media_decode`
(21 tests, 12385 nodes malloced), `media_codecs` (8 tests, 13335 nodes), `media_leaks`
(5135 nodes) and `media_codecs_leaks` (16416 nodes), run with `--test-threads=1`.

### Debug builds and the Objective-C encodings

`objc2` checks every `msg_send!` argument and return type against the method's runtime
encoding in debug builds only. A debug run of `media_codecs` (not part of the release gates)
found that `copyNextSampleBuffer` was declared as returning `*const c_void` (`^v`) while the
runtime reports `^{opaqueCMSampleBuffer=}`: harmless in release (same ABI), a panic on every
video decode in a debug build. Fixed with an `OpaqueCmSampleBuffer` pointee, and
`media::video::tests::the_encodings_used_for_avfoundation_returns_are_the_runtimes` holds
that return, `preferredTransform` and `nominalFrameRate` to the runtime's encodings in every
profile (it fails against the old type, with `^v` against `^{opaqueCMSampleBuffer=}`). All 31
integration tests and the unit tests pass in the debug profile too.

### End to end

`tests/media_e2e_model.rs` (`--ignored`, real checkpoint, GPU; reference from
`tools/embedgemma2_ref/make_media_reference.py`): a file path through `tessl::media`,
`embedgemma2_pre` and the towers, against sentence-transformers' embedding of the same
content (fp32, sdpa, CPU):

| file | what the reference was given | max abs | cosine |
|---|---|---|---|
| `photo.png` (224x224) | the path | 1.34e-7 | 1.000000000 |
| `clip16.wav` (16 kHz mono PCM16) | scipy's array | 1.57e-7 | 1.000000000 |
| `photo.jpg` (4:2:0, q90) | the path | 7.25e-3 | 0.998566 |
| `clip44.wav` (44.1 kHz stereo) | FFmpeg's 16 kHz decode, `-ac 1` (torchcodec's mix and resampler) | 1.74e-7 | 1.000000000 |
| `clip44.m4a` (AAC) | the same | 7.07e-4 | 0.999987957 |
| `clip44.mp3` (MP3, libmp3lame 128 kb/s) | the same | 9.69e-6 | 0.999999996 |

(Before the Rust resampler `clip44.wav` was 1.29e-2 / 0.993142 and `clip44.m4a` 1.09e-2 /
0.995248, against FFmpeg's channel *mean*; the reference script now gives sentence-transformers
FFmpeg's default decode, which is what torchcodec returns.) Three decodes are exact and hold the
bounds of `tests/embedgemma2_e2e_model.rs` (1e-4, 0.99999) with three orders of magnitude to
spare, `clip44.wav` among them: the 44.1 kHz stereo file is mixed and resampled to the bit as
FFmpeg does. The JPEG differs by the IDCT and chroma upsampler (bounds 1e-2 / 0.998). The
AAC clip differs by AudioToolbox's decoder against FFmpeg's (first and last frame, above);
its bounds are 1e-3 / 0.99998 and the MP3's 2e-5 / 0.9999999, from the measurements. Natural
speech has little energy near 8 kHz, where the earlier resamplers differed.
The e2e reference mixes like torchcodec and
resamples like torchcodec; a librosa-branch reference would differ in the last bits of the
resampler and in the 3 dB mix gain for stereo (not measured end to end).

**Video.** `tests/media_e2e_video.rs` (`--ignored`, real checkpoint, GPU; reference from
`tools/embedgemma2_ref/make_video_reference.py`, written to `target/media_e2e_ref/`): an MP4
of `tests/fixtures/media_video/` through `decode_video_file`, `preprocess_decoded_video` and the
facade (`MmItem::Video`, or `MmItem::TimestampedVideo` with the snapshot's tokenizer),
against sentence-transformers' embedding of the frames and metadata the reference loader would
produce from the same file. torchcodec is not installed, so the array it is given is the
`ffmpeg` CLI's decode of the whole file to rgb24 with autorotation and the metadata is
torchcodec's exact-mode figures (`make_video_reference.py` docstring; the sampler's indices it
keeps are asserted equal to the fixture's `facts.txt`, and tessl's equal them too). Snapshot
`914f7f89`, Apple M5 Pro, 2026-10-17, fp32 sdpa goldens, `GemmOperands::ExactF32`; every case
holds the 1e-4 / 0.99999 bounds of `tests/embedgemma2_e2e_model.rs`:

| case | file | sampled | max abs | cosine |
|---|---|---|---|---|
| `e2e_video` | 96x64, 80 frames, 24000/1001 fps, H.264 BT.709 limited | 0, 23, 47 | 1.79e-7 | 1.000000000 |
| `e2e_video_ts` | the same with `add_timestamps` on (`00:00`, `00:00`, `00:01` labels) | 0, 23, 47 | 1.79e-7 | 1.000000000 |
| `rot90` | 64x48 stored, display matrix 90 (the model sees 48x64) | 0 | 1.79e-7 | 1.000000000 |
| `hevc_pc` | HEVC 34x26, BT.601 full range, a width off the NEON widths | 0 | 8.53e-7 | 1.000000000 |
| `hevc_main10` | HEVC Main 10 (`yuv420p10le`) 66x38, BT.601 full range; the reference decode is given `-sws_flags bilinear` (see "The 10-bit 4:2:0 filter") | 0 | 1.79e-7 | 1.000000000 |
| `odd_444_10bit` | HEVC 4:4:4 10-bit at an odd 33x17, SMPTE 240M | 0 | 1.38e-7 | 1.000000000 |
| `odd_422` | HEVC 4:2:2 at an odd height of 33 | 0 | 2.98e-7 | 1.000000000 |

The timestamped case is not redundant: the reference's embedding of the two sequences differs
by 6.9e-2 max abs (cosine 0.827). A control in the same test decodes the first file again with
AVFoundation's own conversion (`ColorPath::AvFoundation`): **2.0e-2 max abs, cosine 0.9864**,
four orders of magnitude from the exact path and outside the bounds, which the test asserts,
so the bound can tell the two colour paths apart. What this does not show is torchcodec itself:
it shows that tessl equals the frames and figures its documented behaviour produces.

### Speed

`bench_media_decode` (inputs from `bench/make_media_bench_inputs.py`; one call per workload,
minimum of 20 after 3 warm-ups; **the machine was running other agents (load average 27 to
31 when the run started), so these are
indicative**; `bench/results/media_decode_m5pro.{txt,json,svg}`, plotted by
`plot_results media-decode`):

| workload | ms (min) | per unit |
|---|---|---|
| 1920x1080 PNG | 19.8 | 9.6 ms per Mpixel |
| 1920x1080 JPEG | 8.9 | 4.3 |
| 1920x1080 TIFF | 5.0 | 2.4 |
| 30 s 16 kHz mono WAV | 0.22 | 0.007 ms per audio s |
| 30 s 44.1 kHz stereo WAV (resampled) | 158 | 5.3 |
| 30 s 44.1 kHz stereo FLAC | 207 | 6.9 |
| 30 s 44.1 kHz stereo AAC | 175 | 5.8 |
| 30 s 44.1 kHz stereo MP3 | 180 | 6.0 |
| 10 s 720p H.264, `VideoSource::open` (pass one, 300 frames) | 25.3 | 0.084 ms per frame |
| the same, open plus 9 sampled frames decoded with AVFoundation's BGRA conversion (the decode before the exact path; decodes all 300) | 221 | |

**Video to RGB, exact path against the BGRA path** (`BENCH_MEDIA_DIR=target/media_bench_video`,
clips from `bench/make_video_bench_inputs.py`, minimum of 30 calls after 5 warm-ups, a fresh
source each; **load average 21 when the run started and the machine ran other agents, so
indicative**; `bench/results/media_video_color_m5pro.{txt,json,svg}`, plotted by
`plot_results video-color`; the bench refuses a run in which a path fell back to the other).
Wall time of one call, ms, with ms per frame in brackets for the first-30-frames rows:

| clip | call | exact (VideoToolbox planes + Rust conversion) | bgra (AVFoundation) |
|---|---|---|---|
| 720p H.264 (1280 wide, NEON) | 10 sampled frames | 248 | 222 |
| | first 30 frames | 235 (7.8) | 229 (7.6) |
| 720p HEVC | 10 sampled | 152 | 216 |
| | first 30 | 181 (6.0) | 197 (6.6) |
| 1080p H.264 | 5 sampled | 168 | 233 |
| | first 30 | 228 (7.6) | 139 (4.6) |
| 1270x720 H.264 (table converter) | 5 sampled | 125 | 119 |
| | first 30 | 127 (4.2) | 117 (3.9) |

Both paths decode every frame up to the last one asked for, so the rows are dominated by the
decode. At 720p the two paths are within 0.7 ms per frame of each other in the first-30 rows,
in either direction (HEVC: exact faster), and the sampled rows swap order from run to run (an
earlier, noisier run had them the other way). At 1080p the first-30 rows are 89 ms apart
(228 against 139), but the same clip's sampled rows are 65 ms apart in the opposite direction
(168 against 233), so this run does not isolate the conversion's cost; a single-threaded
conversion of 2 Mpixel is a plausible reason for the first-30 gap (inferred, not profiled).
**The conversion is now threaded** (`Params::threads`, `Threads::Auto` by default: row bands
over `std::thread::scope`, one band per available core up to eight, pictures under 65,536
pixels single). It
cannot change the bytes, since each output pixel is a function of its own samples (and, for
10-bit 4:2:0, of fixed neighbouring chroma rows), and the tests hold that:
`threading_never_changes_a_byte` (every format, many sizes and thread counts) and
`the_thread_count_never_changes_the_frames` (through the whole decode). **Its speed was not
measured**: the benchmark (`bench_media_decode` with `BENCH_YUV=1`, which now also times the
converter alone, `yuv:LAYOUT_WxH:single|threads`, and the whole call with one thread,
`exact1`) was not run, because the machine's load average was 130 to 190 for the whole lane
and a timing taken then would measure the other agents, so `bench/results/media_video_color_m5pro.*`
above are still the pre-threading numbers. `plot_results video-color` draws the third bar once a run
contains `exact1` rows.

The audio rows above were measured with the `AudioConverter` as the default (before the Rust
resampler); the sample-rate conversion at the `Mastering` setting was most of the audio time.
`bench_media_resample` (same inputs; `bench/results/media_resample_m5pro.{txt,json,svg}`,
plotted by `plot_results media-resample`; minimum of 30 after 2 warm-ups; **again a shared
machine, other agents running, so indicative**; the repeated runs moved the codec rows by up to
50 %, the WAV rows by under 10 %) times the same call both ways, `decode_audio_file` of 30 s of
44.1 kHz stereo to 16 kHz mono:

| file | decode at the file's rate | libswresample in Rust (default) | AudioToolbox converter | Rust converter alone, one channel |
|---|---|---|---|---|
| WAV | 1.3 ms | 17.1 ms | 144.5 ms | 10.5 ms |
| FLAC | 29.9 | 60.2 | 272.2 | 10.5 |
| AAC (m4a) | 10.8 | 41.8 | 234.5 | 10.8 |
| MP3 | 33.4 | 56.0 | 170.4 | 8.3 |

The Rust path is 3.0 to 8.4 times faster end to end (WAV 8.4x, MP3 3.0x) and takes 16 to 31 ms more than the
decode alone for two channels. An earlier, quieter run of the same program measured 7.5 ms for the
converter alone on 30 s (4000 times real time) and 20.7 ms against 141.6 ms for the WAV. No
comparison lane (torchvision, librosa) is timed: librosa and torchcodec are not installed, and a
timing against subprocess FFmpeg would measure process start. Not compared: the `ffmpeg` binary's
own speed (it is the same algorithm in C with NEON).

### What is left

- torchcodec itself is not in the environment: the video reference is the `ffmpeg` CLI and
  torchcodec's documented exact-mode figures, so a run with torchcodec installed would close
  the last inferred step. What it does with a rotation flag is unverified (the default here
  follows FFmpeg's CLI and OpenCV).
- `AVAssetReader`'s chroma difference is explained to within a three-tap kernel (a
  re-siting to `BottomLeft`, above), not reproduced to the byte (96 percent of samples with
  `(1, 10, 5)/16`); VideoToolbox is driven directly, so nothing depends on it.
- torchcodec's scaler flags and its rotation order are not established. Only 10-bit 4:2:0 and
  rotated 4:2:0/10-bit streams depend on them (above), and `ChromaFilter` and the rotation
  order are the two places to change if a run with torchcodec shows otherwise.
- VideoToolbox does not decode H.264 High 10, 4:2:2 or 4:4:4 on this machine, so those H.264
  shapes cannot be decoded by this path (VideoToolbox answered "Cannot Decode" when a session was
  created for them, checked in this lane with a scratch program and not kept as a test), and
  MPEG-4 Part 2 fails to open under both colour paths (a test). 12-bit and monochrome HEVC are refused by name. Odd-width 4:2:0 and 4:2:2 and
  odd-height 4:2:0 cannot be coded by either codec and are refused by the converter.
- Ogg Opus in SILK or hybrid mode does not match a torchcodec reference closely (15 to 20 dB
  from FFmpeg's native decoder, which is itself 2 to 4 samples behind libopus there); matching
  it needs FFmpeg's SILK decoding, not a shift. AudioToolbox is exact against libopus for
  SILK. MP3 and AAC stay on Apple's decoders by decision; their bounds are above.
- The librosa branch's `soxr_hq` is not reproduced (no libsoxr source on disk; measured 87 to
  101 dB from this module in the interior on band-limited tones, 25 to 40 dB whole-signal where
  the content reaches 8 kHz). The `torchaudio` branch (`sinc_interp_hann`) is not reproduced
  either, and is 51 to 67 dB from it on those tones. Each would be its own converter with its
  own oracle.
- HEIC, H.264 and HEVC decoding cannot run in the hosted sandbox, so their tests run locally only.
- Bit equality is with the AArch64 NEON build of FFmpeg 9.0.2. An x86 torchcodec sums in
  another order (unverified); and torchcodec itself was never run.
- A file's own channel mask (WAVE_FORMAT_EXTENSIBLE) is not read: FFmpeg's default layout for
  the channel count is assumed (1 to 8 channels). The `libswresample` port is LGPL-2.1-or-later
  code's behaviour in a crate that is MIT OR Apache-2.0: for the owner to settle.
