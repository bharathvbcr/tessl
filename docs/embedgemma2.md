# EmbeddingGemma 2 text encoder

`tessl::embedgemma2` runs `google/embeddinggemma-2`'s text path (the 271M
encoder) on tessl's kernels, from the checkpoint's own `model.safetensors`.
The vision tower and its embedder are `tessl::embedgemma2_vision`
([below](#vision-tower)); the audio tower is not implemented.

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
forwards ran. The per-forward cost (256 rows) is an estimate that only steers
the split: a sequence's embedding does not depend on its batch beyond GEMM
tiling, which the model test bounds at `1e-5`.

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

## Kernels

| Kernel | File | New? |
|---|---|---|
| `encoder_attn_tiled_h256_q32_k32_sg4`, `encoder_attn_tiled_h512_q32_k32_sg8` | `kernels/encoder_attn.metal` over `kernels/attn_tiled.h` | new: bidirectional attention with a symmetric window and per-row lengths on the TensorOps matrix units, the FlashAttention-2 body Qwen3.5's prefill uses (`ENCODER` mode). What `encoder_attn` runs on a device with TensorOps |
| `encoder_attn_rows_h{256,512,32,64}_*` | `kernels/encoder_attn.metal` over `kernels/attn_rows.h` | new: the same contract, one simdgroup per query row, scalar f32. Head dims 32/64 (BERT), and 256/512 on a device without TensorOps. `encoder_attn_with` names either kernel; every attention test runs on both |
| `segment_mean_rows_f32`, `l2_normalize_rows_f32` | `kernels/embed_pool.metal` | new: row-range means (the pool here, the option pool in rsi-jev) and the (Matryoshka-prefix) normalize |
| q/k norm + full RoPE + K/V store, with the weightless V norm | `qwen35_attn_qk_norm_rope` via `qwen35::attn_qk_norm_rope_columns` (`QkvColumns::v_norm`) | reused; the host entry is new (separate K and V column blocks), and so is the V norm inside it. The frequencies are the host table `nn::rope_inv_freq` (torch's formula), not a device `pow` |
| embedding gather times `sqrt(hidden)` (bf16 table) | `qwen35_embed_rows_bf16_scaled` via `qwen35::embed_rows_scaled` | new: the gather and the scale in one pass |
| GELU-tanh gate over column windows (`gate \| up` from one GEMM; each layer's slice of the per-layer inputs) | `qwen35_gelu_tanh_glu_f32` via `qwen35::gated_act` | new: `qwen35_swiglu`'s kernel with the activation as a parameter |
| RMSNorm, post-norm residual add with layer scale, scale | `nn::rms_norm_f32`, `nn::rms_norm_residual_add_f32`, `nn::scale_f32_inplace` | reused |
| GEMM | exact f32, TensorOps (NN, and NT for the per-layer-input projection) | reused |

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
  workloads rose 29 % (1283 to 1652 MiB at 8x1024). Numbers:
  `embedgemma2_before_after_m5pro.txt`, `embedgemma2_load_footprint_m5pro.txt`.
* **One host wait per `encode`.** The forwards of one `encode` are queued back
  to back and the host waits once; each keeps its own small ids, lengths,
  segments and output buffers (written with `alloc_buffer_from_u32`, which does
  not wait), and the large activations stay shared because the GPU orders
  them. This only overlaps anything when the caller runs the runtime with
  `set_async_encode(true)`: with it off (the default) every dispatch commits
  and waits, so nothing changes. `encode` does not flip the flag, since the
  runtime is shared. With it on, `bench_embedgemma2` (`BENCH_ASYNC=1`) runs
  1x16 in 7.4 ms against 84.9 ms, and the 128x8-512 ragged batch in 1903 ms
  against 2366 ms.
* **What a forward costs, for the split.** `pack_forwards` trades padded rows
  against a fixed cost per forward, which the sweep
  (`embedgemma2_forward_cost_m5pro.json`) puts at `54.9 ms + 0.0525 ms/row`
  synchronous (1046 rows) and `3.42 ms + 0.0515 ms/row` queued (67 rows); it
  uses the one for the runtime's mode.
* **`segment_mean_rows`** stays serial: it is under 0.2 % of an encode on
  every workload (`embedgemma2_segment_mean_share_m5pro.txt`).
* **No bf16 attention output.** `encoder_attn` writes f32 only; the `out_bf16`
  argument had no caller and is gone.

## Precision

F32 only: bf16 weights widened exactly, f32 activations, exact-f32 GEMMs.
A bf16-operand path is not written; it would be held to transformers' bf16
forward the way `tests/qwen35_model.rs` holds Qwen3.5's.

## Vision tower

> **Status: run on the GPU against sentence-transformers** (Apple M5 Pro,
> 2026-10-15, worktree branch `worktree-agent-a00e8fdf52347d85d` on base
> `60a6240`, snapshot `914f7f89142e33e77833254d9c9b90c3cef7303b`; goldens from
> `make_mm_reference.py`, the multimodal sibling of `make_reference.py` that
> lives on branch `eg2/integrate` and not on this base: fp32, eager attention,
> CPU).

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
