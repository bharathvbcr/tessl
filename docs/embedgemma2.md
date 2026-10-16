# EmbeddingGemma 2 text encoder

`tessl::embedgemma2` runs `google/embeddinggemma-2`'s text path (the 271M
encoder) on tessl's kernels, from the checkpoint's own `model.safetensors`.
The vision tower and its embedder are `tessl::embedgemma2_vision`
([below](#vision-tower)); the audio tower and its embedder are
`tessl::embedgemma2_audio` ([Audio tower](#audio-tower)).

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

Tokenization is the caller's (the tokenizer adds `<bos>` and `<eos>`;
prompts such as `"task: search result | query: "` are part of the text).
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

The soft rows are uploaded with the forward's ids (`alloc_buffer_from_u32` on the
f32 bits: no wait, and a forward with no soft rows allocates nothing extra), and
land in `embeds` after the scaled gather and before anything reads it: the
residual stream, the bf16 copy and the per-layer-input GEMM all see the merged
rows, so the rest of the forward is unchanged. The write is `qwen35_bwd::copy_cols`
(one flat copy per maximal run of consecutive placeholder rows of the padded
forward, a single row of the run's whole width): no new kernel. One image or clip
is one run; an adversarial layout of alternating placeholder and text tokens is
one dispatch per placeholder, at most `(rows + 1) / 2` of a forward's (32,768
padded rows at most).

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
video tower or reference exists here), and the C and Python entries, which
embed text only.

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

## Kernels

| Kernel | File | New? |
|---|---|---|
| `encoder_attn_tiled_h256_q32_k32_sg4`, `encoder_attn_tiled_h512_q32_k32_sg8` | `kernels/encoder_attn.metal` over `kernels/attn_tiled.h` | new: bidirectional attention with a symmetric window and per-row lengths on the TensorOps matrix units, the FlashAttention-2 body Qwen3.5's prefill uses (`ENCODER` mode). What `encoder_attn` runs on a device with TensorOps |
| `encoder_attn_rows_h{256,512,32,64}_*` | `kernels/encoder_attn.metal` over `kernels/attn_rows.h` | new: the same contract, one simdgroup per query row, scalar f32. Head dims 32/64 (BERT), and 256/512 on a device without TensorOps. `encoder_attn_with` names either kernel; every attention test runs on both |
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
(256 MiB): all 24 up to 5461 rows, 6 runs of 4 at 32,768 rows.

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

* **Whole encode, before and after the nine landed changes** (2a7cb8b against
  0055612, combined, not per commit): an encode takes 0.35x to 0.84x of the
  time (1.2x to 2.9x faster) in the default runtime, uniform and ragged
  workloads alike. The loader's peak physical footprint fell from 1156 MB to
  871 MB (77d8732) and 845 MB (0055612). The device peak of the single-forward
  workloads rose 29 % (1283 to 1652 MiB at 8x1024; "The device-peak rise,
  isolated" below says why). Numbers:
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
* **Small-M GEMMs: parallel K-partitions** (`gemm`, `gemm_bf16`; lane G,
  `embedgemma2_small_m_m5pro.txt`). The four GEMMs that ran at 17-22 % of the copy
  bandwidth at 16 rows (down, o, the two per-layer-input ones) have a C of 16
  tiles of 32x32: one dispatch gave them 16 single-simdgroup threadgroups, each
  walking all of K, so their time was that walk's latency. Below 97 rows at
  N = 512 (a C of under 64 tiles of 32x32, or under 16 of 64x64 for bf16) the
  K is cut into about 128 (64 for bf16) partitions' worth of threadgroups,
  `matmul2d_tensorops_nn_splitk_par_f32` (bf16:
  `..._nn_splitk_par_bf16_f32`) into a scratch with one slice per partition and
  `reduce_partitions_f32` adding the slices in partition order: the TN lane's
  design, for the NN layout. No change in `src/embedgemma2.rs`: its `project`
  already goes through `gemm` / `GemmOperands::Bf16.nn`. Per GEMM at 16 rows,
  interleaved against the single dispatch: down 2.0x, o 1.8-2.2x, ple 1.35x
  (exact f32); down 1.1x, o 1.4-1.7x, ple 1.16x (bf16; medians of per-round
  ratios, GPU shared). Whole `encode`
  (`bench_paired`, before = `df5d329`): 1x16 0.63x of the time in clean rounds
  (f32, 1.6x faster) and 0.79x (bf16); 1x64 0.81x and 0.85x; 64x32 and 8x1024
  take the same kernels as before (the gate returns none at 2048 and 8192 rows)
  and moved inside the machine's noise. The arithmetic stays exact f32 (bf16
  operands, f32 accumulate, under `Bf16`): only the summation order changes, K
  in partitions summed in order, and the lane is held to the f32 dot-product
  bound `gamma_(K+8) * sum|a.b|` that every exact-f32 GEMM meets (worst result
  `0.0039` of it; `tests/gemm_small_m.rs`, `tests/gemm_small_m_bf16.rs`), and
  the forward to the same `1e-4` of fp32 as before. A row's bits now depend on M across
  the gate (M up to 96 and M of 97 or more at N 512 sum K in different orders). This is less than the 1.8x
  the stage profile's bandwidth estimate allowed: the lane reaches about 55 %
  of the copy ceiling, not 70 %, and pays a reduce dispatch per GEMM.
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
* **`segment_mean_rows`** stays serial: it is under 0.2 % of an encode on
  every workload (`embedgemma2_segment_mean_share_m5pro.txt`).
* **No bf16 attention output.** `encoder_attn` writes f32 only; the `out_bf16`
  argument had no caller and is gone.

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
  the attention output's cast stays until it gains a bf16 output.
* **What stays f32.** The residual stream, the norms' arithmetic, the q/k/v
  norms and RoPE, the attention, the pooling, the normalization, and every
  GEMM output.
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
| tiny: per-layer residual + final norm vs the f64 forward over the same bf16 weights | `5e-2` x max | `4.9e-3` to `8.4e-3` x max |
| tiny: embeddings, alone and batched | `1e-2` max abs, cos >= `0.9999` | <= `1.4e-3`, cos >= `0.999987` |
| tiny: batched vs alone, ragged runs of per-layer inputs, sync vs async | `1e-5` | 0 (bit-identical) |
| real: cos vs fp32, each of 8 texts | >= `0.9999` | `0.99999324` to `0.99999848` (max abs <= `4.3e-4`) |
| real: mean `1 - cos` vs fp32, against transformers' bf16 forward | <= its | `4.1e-6` against `4.7e-5` |
| real: cos vs transformers' bf16 (`emb_bf16.npy`) | >= `0.9996` (triangle inequality over two `1e-4` deficits) | `0.99989959` to `0.99997965` |
| real: alone vs one ragged batch (>= 2 forwards) | `1e-5` | 0 (bit-identical) |

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
longest first, queued behind one `synchronize`. `GpuRuntime::async_scope()` is
not on this base; when it lands, `encode_images` is where it replaces that
call.

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
| `vision_qkv_norm_rope_f32` | `kernels/embedgemma2_vision.metal` | new: q/k RMSNorm (`* w`), the two-axis RoPE, the weightless v norm and the K/V store, one simdgroup per (patch, head). Its own kernel, not a mode of `qwen35_attn_qk_norm_rope`: that one rotates a head by one position per token and its row routine is mirrored by the training backward, whose inverse rotation must stay its exact transpose |
| `vision_pos_embed_add_f32` | same | new: `proj + (x_emb + y_emb)` from the bf16 table, exactly `+0` on padding. The `+0` guarantee lives here and at the pool's output; the rope kernel does not special-case padding (a zero row rotated by -1 may come out `-0.0`), and its padded q/k/v slots are masked or overwritten by `encoder_attn` |
| `vision_pool_members_f32` | same | new: the pooler's cell sums, weighted `1/k²`, times `sqrt(hidden)` |
| weightless RMSNorm | `nn::rms_norm_f32` with a hot `[768]` buffer of ones | reused (the multiply by `1.0` is exact) |
| attention | `embedgemma2::encoder_attn`, rows kernel `encoder_attn_rows_h64_r4_g16` | reused |
| RMSNorm, residual add, GELU-tanh gate, GEMM | `nn::rms_norm_f32`, `nn::rms_norm_residual_add_f32`, `qwen35::gated_act`, exact-f32 TensorOps GEMM | reused |

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

**Speed follow-ups, not done.** (1) Attention at 2520 patches runs the scalar
rows kernel (`encoder_attn_rows_h64_r4_g16`), one simdgroup per query row: about
17 GFLOP of QK and PV per layer per image at 12 heads (2376 x 2376 x 64 x 4
flops per head: arithmetic, inferred, not measured), all through scalar f32.
A tiled TensorOps kernel for head dim 64 (the h256/h512 tiled kernels'
FlashAttention-2 body with a 64-wide tile) is the next step; no timing was
taken here. (2) The 16 layers' f32 weights are 600 MiB resident; a bf16
operand path (a later step) halves that. (3) The pos-embed add, the pool and
the rope kernel each take one pass over the hidden state that the neighbouring
norm or GEMM epilogue could absorb. (4) Images are padded to the longest of a
forward; the greedy longest-first cut does not weigh a per-forward fixed cost
as the text packer does.

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
> `worktree-agent-a975c42e7f8ed0908` on base `60a6240`, snapshot
> `914f7f89`; torch 2.14.1, transformers 5.19.0): every bound below holds. The
> runtime has no `async_scope` on this base, so `encode_audio` queues its
> forwards and ends with one `synchronize()`.

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
so no clamp rides on a GEMM; each is folded into the elementwise pass beside it.

| Kernel | Does |
|---|---|
| `eg2a_sub_conv0_ln_relu`, `eg2a_sub_conv1_ln_relu` | the strided conv with the frame mask, the channel LayerNorm and ReLU in one pass; threadgroup per output position (channels across threads), simdgroup per position for the 32-channel layer. Output channels-last, so the second layer's flatten order is its store order |
| `eg2a_rms_norm` | `[resid +] scale * clamp(act(rms_norm(clamp(clamp(x))) [* w]))`: a linear's output clamp and the gradient clamp before, SiLU and the next linear's input clamp after, the residual add; in place allowed. The weightless embedder norm is `weight = None` |
| `eg2a_ew` | `clamp(act(clamp(x))) + residual \| bias`: between the feed-forward's linears, a linear's output clamp plus the residual, an input clamp, the output bias |
| `eg2a_glu` | the GLU with the linear's output clamp |
| `eg2a_dwconv1d_causal` | the depthwise causal conv over clips (rows before a clip's start read 0) |
| `eg2a_attn` | the chunked local attention, one simdgroup per (query, head); the q/k/v output clamps are applied as it loads, the output projection's input clamp as it stores |

SiLU and the sigmoid are `kernels/act.h`'s overflow-free forms. A clip bound of
infinity (the module's untrained default) is held as `+-f32::MAX`, since the
kernels are compiled with fast math.

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
softmaxes to the uniform `1/24` over its window), and then dropped.
`trace` returns each clip's subsample output, per-layer residuals and tower
output. The activations are unzeroed, as the text encoder's are; the tests
poison every one with NaN.

### Bounds and observed errors

Fixed before the first run (the real-checkpoint bounds are the ones
`tests/embedgemma2_audio_model.rs` states):

| Check | Bound | Observed |
|---|---|---|
| the f64 reference vs transformers (`tests/embedgemma2_audio_fixtures.rs`: tiny random tower, 3 clips, trailing and interior padding, bounds that bite) | `1e-5` × max | ≤ `5.7e-7` |
| each kernel vs the f64 reference, small and real shapes (`tests/embedgemma2_audio_kernels.rs`) | `2e-5` × max | ≤ `1.4e-6` (second conv, 128 -> 32 channels) |
| tiny tower, 2 layers (`tests/embedgemma2_audio_tiny.rs`): subsample, each layer, tower output | `1e-4` × max | ≤ `1.7e-6` |
| tiny tower, soft tokens | `1e-4` max abs | ≤ `2.4e-6` |
| same at the checkpoint's attention geometry (hidden 256, head dim 128, chunk 12) | as above | ≤ `1.4e-6`, soft ≤ `2.2e-6` |
| clip alone vs inside any batch, reordered, traced, with every unzeroed activation poisoned, across a split into two forwards | bit-identical | bit-identical |
| real checkpoint, clip 0 (99 frames, 25 tokens), subsample, 12 layers, tower output | `1e-4` × max | ≤ `2.9e-6` (layer 11), tower `3.9e-6` |
| real checkpoint, clip 1 (330 frames, 329 valid, 83 tokens) | `1e-4` × max | ≤ `5.1e-6` (layer 11), tower `1.0e-5` |
| real checkpoint, soft tokens | `1e-4` × max | `5.7e-6` and `2.1e-5` (max abs `2.9e-5`, `1.1e-4` on values up to 5.5); cosine 1.000000000 |
| both clips in one batch vs alone | bit-identical | bit-identical |

The tiny test's bounds bite: with the clip bounds, the gradient clip or the
softcap relaxed in the reference, the tower output moves by `74 %`, `30 %`
and `3.7 %` of its maximum. Two mutations of the attention kernel (the window
one frame wider, the relative index off by one) each fail it at `1e-1`.

The first run of the real-checkpoint test failed its soft-token check at
`1.14e-4` max abs because the check had been written as an absolute `1e-4`
where the other stages (and the task) bound by `1e-4` of the largest
magnitude; the stated bound is the relative one, and the soft tokens are within
`2.1e-5` of it. Nothing else was changed. The reference there is the sdpa one.

### Speed follow-ups (not measured; no timing was run)

About 28 dispatches per layer, most of them elementwise, and 48 MFLOP per row
per layer in the GEMMs (two feed-forwards 33.6, attention projections 8.4,
light-conv linears 6.3; inferred from the shapes). Candidates, each needing a
benchmark before it is believed: one packed q|k|v GEMM (the three input bounds
differ in the checkpoint, so it needs a clamping prologue or one clamped copy per
distinct bound); clamps on the exact-f32 GEMM path; fusing the GLU into the
depthwise conv and the conv-norm; sharing K/V loads across the 12 queries of a
block in the attention (each (query, head) reloads its 12 keys and values); the
bf16 mode.
