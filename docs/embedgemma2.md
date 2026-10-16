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
`python/tessl_torch/embedgemma2.py` put the encoder behind the tokenizer the
caller already holds. Build the library once (`cargo build --release`, or point
`TESSL_LIB` at a `libtessl.dylib`):

```python
from tokenizers import Tokenizer            # or transformers.AutoTokenizer
from tessl_torch.embedgemma2 import EmbedGemma2

model = EmbedGemma2.from_pretrained(snapshot_dir)   # a local snapshot directory
tok = Tokenizer.from_file(f"{snapshot_dir}/tokenizer.json")
q = model.encode_texts(["how do I speed up attention?"], tok, prompt_name="query")
d = model.encode_texts(["Sliding-window attention ..."], tok, prompt_name="document")
scores = q @ d.T                                    # rows are L2-normalized float32
small = model.encode(ids, truncate_dim=256)         # token ids you tokenized yourself
```

`encode_texts` prepends the snapshot's own prompt (the `prompts` of
`config_sentence_transformers.json`) to each text and asks the tokenizer for
special tokens, which is what sentence-transformers' `encode(prompt_name=...)`
does. The Python test checks that the ids it produces equal the reference's
for both a `tokenizers.Tokenizer` and a transformers tokenizer. It duck-types the
tokenizer and imports neither package, so neither is a dependency.

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
changed, so the ABI version (11) did not, and a library built before them is
named as such by the Python binding.

| Entry | Does |
|---|---|
| `tessl_embedgemma2_load_multimodal(runtime, dir, dir_len, operands, &out, err, err_len)` | `tessl_embedgemma2_load` with the vision and audio towers as well: one read of `config.json`, one open of `model.safetensors`. `operands` is `TESSL_OPERANDS_EXACT_F32` (0) or `TESSL_OPERANDS_BF16` (1) and applies to the **text encoder and the vision tower**; the audio tower is exact f32 under either (it has no bf16 mode). The handle takes every text entry too, and `_free` |
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
emb = model.embed_images(images, processor)                           # the same two lines in one call
emb = model.embed_audio(waveforms, processor)                         # or encode_audio(input_features, input_features_mask)
```

`encode_images` / `encode_audio` take the processor's arrays or tensors
(`[N, P, 768]` / `[N, F, 128]`, one item without the leading axis, or a list of
per-item arrays of different lengths; numpy or torch, recognised by their
methods and never imported; bf16 widened first), check dtype, shape, counts and
the int32 and 0/1 ranges, and hand flat arrays to the library.
`embed_images` / `embed_audio` call a processor you pass in with
`processor(images=list(images), return_tensors="np")` /
`processor(audio=list(clips), return_tensors="np")` (checked against
transformers 5.19: a list of images is one sample of several images, so only
`pixel_values` are used, one embedding per image) and require the two output
keys. `python/tests/test_embedgemma2_mm_host.py` holds the argument handling to
a recording stand-in library; `python/tests/test_embedgemma2_mm.py` runs the
tiny checkpoint, and with `EMBEDGEMMA2_SNAPSHOT` and `EMBEDGEMMA2_MM_REF_DIR`
the real one (see Observed below).

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

Not covered: video (the placeholder id is handled like the other two, but no
video tower or reference exists here). The towers' output reaches this entry
as host rows; [One call](#one-call-text-images-and-audio) does that composition.

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
  end. The counts must match exactly; a video placeholder is refused (no video
  tower); a `MmItem::Text` holding a placeholder id is refused (in a
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
  encoder and the vision tower alike (`EmbedGemma2VisionModel::load_with`, [Vision
  tower](#vision-tower)): it is one flag because both make one precision claim to
  the caller, and a caller composing the models by hand (`from_parts`) can still
  mix them. The audio tower is exact f32 under both: it has no bf16 mode (lane I
  owns it). The call refuses a runtime with relaxed f32 GEMMs on when it holds
  clips or the mode is `ExactF32`; a `Bf16` call of text and images runs no f32
  GEMM and accepts it.
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
| text encoder and vision tower `Bf16`, audio tower exact, 4 media rows vs fp32 | cosine `>= 0.9999` | 0.999987 to 0.999995 (images 0.999988, 0.999988; clips 0.999995, 0.999994) |
| Python, `encode_images` / `encode_audio`, alone and in one call | `1e-4`, `0.99999`; alone vs batch `1e-5` | `2.2e-7` to `3.8e-7`, cosine 1.000000000; alone vs batch 0 |
| Python, `embed_images` / `embed_audio` with transformers 5.19's processor on the generator's inputs | processor tensors vs saved `1e-6`, then the same embedding bounds | tensors identical (0), `3.8e-7` and `3.0e-7`, cosine 1.000000000 |

Follow-ups: (1) done: the GPU-resident soft-token hand-off (the towers' output
buffers read by the text encoder's `copy_cols`, no host round trip, one wait for
a mixed call; speed: `bench/results/embedgemma2_mm_onewait_m5pro.txt`). (2) A
`Media` entry in the C ABI and
Python for caller ids with placeholders; the C entries take pure images and
clips. (3) The tower tiny-checkpoint generators exist in four test files
(`tests/common/embedgemma2_mm.rs`, `embedgemma2_vision_tiny.rs`,
`embedgemma2_audio_tiny.rs`, and the text one in `common`); the last two carry
f64 references and were left alone, but converge on `common`.

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
| RMSNorm, post-norm residual add with layer scale, scale | `nn::rms_norm_f32`, `nn::rms_norm_residual_add_f32`, `nn::scale_f32_inplace` | reused |
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
* **No bf16 attention output.** `encoder_attn` writes f32 only, at bf16
  operands too; the `out_bf16` argument had no caller and is gone. (The
  shared body still has the flag, so the bf16 forward's per-layer cast of the
  attention output could be dropped by passing it; not done here.)

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
  (`rms_norm_bf16`) and the GELU gates (`qwen35_gelu_tanh_glu_bf16`) write
  bf16 directly, at no extra dispatch, and four cast passes
  (`cast_f32_to_bf16_into`) cover the rest: the attention output and the
  residual stream into the per-layer-input gate once per layer, the embeddings
  and the pooled vectors once per forward. `encoder_attn` writes f32 only, so
  the attention output's cast stays until it gains a bf16 output (it could
  write bf16 through the body's existing flag, which would drop that pass; not
  done here).
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
queues more behind it: [One call](#one-call-text-images-and-audio)).

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
the text encoder and the vision tower of the facade
(`EmbedGemma2Multimodal::load(.., operands)`, `tessl_embedgemma2_load_multimodal`'s
`operands`, `EmbedGemma2.from_pretrained(.., multimodal=True, operands="bf16")`).
The audio tower stays exact f32 under either flag: its bf16 mode belongs to its
own lane (the audio tower's section below), and until it lands a `Bf16` load
runs the audio tower exactly as `ExactF32` does.

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

`encode_audio` takes up to 256 clips of any lengths. They are laid end to end
in each row space and run as one forward up to 16,384 frames (`MAX_FRAMES_PER_FORWARD`;
a clip above that is refused, more clips are more forwards, all queued before the
one wait). Clips never read each other's rows: the convolutions and the attention
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
