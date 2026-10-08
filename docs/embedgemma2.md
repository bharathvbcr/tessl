# EmbeddingGemma 2 text encoder

`tessl::embedgemma2` runs `google/embeddinggemma-2`'s text path (the 271M
encoder; the vision and audio towers in the same checkpoint are not loaded)
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

Layers whose `kv_heads * head_dim` differ get separate, exactly sized K/V
buffers. The K/V writer derives its per-sequence stride from the buffer's
size, while the attention and the value norm read at stride `seq`, so a
shared buffer sized for the widest layer misplaces every sequence after the
first in the narrower layers. The checkpoint's layers all have width 512, so
it never showed this; `tests/embedgemma2_tiny.rs` mixes 512 and 256.

## Kernels

| Kernel | File | New? |
|---|---|---|
| `encoder_attn_rows_h256_r16_g32`, `encoder_attn_rows_h512_r32_g32` | `kernels/encoder_attn.metal` | new: bidirectional rows attention, symmetric window, per-row lengths, scale as a parameter |
| `segment_mean_rows_f32`, `l2_normalize_rows_f32` | `kernels/embed_pool.metal` | new: row-range means (the pool here, the option pool in rsi-jev) and the (Matryoshka-prefix) normalize |
| q/k norm + full RoPE + K/V store | `qwen35_attn_qk_norm_rope` via `qwen35::attn_qk_norm_rope_columns` | reused; the host entry is new (separate K and V column blocks). The frequencies are the host table `nn::rope_inv_freq` (torch's formula), not a device `pow` |
| RMSNorm, post-norm residual add with layer scale, GELU-tanh gate, scale | `nn::rms_norm_f32`, `nn::rms_norm_residual_add_f32`, `nn::mlp_gelu_tanh`, `nn::scale_f32_inplace` | reused |
| embedding gather (bf16 table) | `qwen35_embed_rows_bf16` | reused |
| GEMM | exact f32, TensorOps | reused |

The encoder attention is its own kernel rather than a mode of
`flash_attn_rows`: there the causal rule is the loop bound (keys above the
query are never visited) and the live length is one device scalar for every
row, and that body is the tuned prefill path of gemma-metal and Qwen3.5.

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

## Precision

F32 only: bf16 weights widened exactly, f32 activations, exact-f32 GEMMs.
A bf16-operand path is not written; it would be held to transformers' bf16
forward the way `tests/qwen35_model.rs` holds Qwen3.5's.
