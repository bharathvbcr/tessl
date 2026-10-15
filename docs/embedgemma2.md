# EmbeddingGemma 2 text encoder

`tessl::embedgemma2` runs `google/embeddinggemma-2`'s text path (the 271M
encoder; the audio tower is `tessl::embedgemma2_audio`, see
[Audio tower](#audio-tower), and the vision tower is not loaded here)
on tessl's kernels, from the checkpoint's own `model.safetensors`.

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
> `914f7f89`; torch 2.12.1, transformers 5.19.0): every bound below holds. The
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
`attn_implementation="sdpa"` it gets one. Under `"eager"` (what the
multimodal reference generator `make_mm_reference.py`, in `tools/embedgemma2_ref/`
on the multimodal branch, runs) `create_bidirectional_mask` returns a float additive mask (0.0 visible, `finfo.min` not), and
`logical_not()` of a float is true exactly where it is 0.0: the visible keys are
filled with -1e9 and the keys outside the window, and the zero padding the
blocked layout adds, are attended to. The attention is hand-written matmuls in
both, so the setting selects only the mask's dtype, and the sdpa behaviour is
the specified one. tessl implements that.

Consequence for references: `make_mm_reference.py`'s audio files
(`layer*.npy`, `tower_out.npy`, `soft.npy`, and the audio `emb_*`) describe the
inverted mask; its `subsample.npy` does not touch attention and is right.
`tools/embedgemma2_ref/make_audio_sdpa_reference.py` rewrites the audio
references under sdpa from the same `input_features`; the fix to the generator
is `attn_implementation="sdpa"` (the text and vision towers only change
rounding). Measured, tessl against the eager references of the first clip:
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
