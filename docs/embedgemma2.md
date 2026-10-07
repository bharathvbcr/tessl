# EmbeddingGemma 2 text encoder

`tessl::embedgemma2` runs `google/embeddinggemma-2`'s text path (the 271M
encoder; the vision and audio towers in the same checkpoint are not loaded)
on tessl's kernels, from the checkpoint's own `model.safetensors`.

> **Status: written, not yet run on the GPU.** The numbers below are the
> bounds the tests are held to, fixed before any run. This line is replaced
> by the measured results and the commit they were measured at.

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

## Kernels

| Kernel | File | New? |
|---|---|---|
| `encoder_attn_rows_h256_r16_g32`, `encoder_attn_rows_h512_r32_g32` | `kernels/encoder_attn.metal` | new: bidirectional rows attention, symmetric window, per-row lengths, scale as a parameter |
| `segment_mean_rows_f32`, `l2_normalize_rows_f32` | `kernels/embed_pool.metal` | new: row-range means (the pool here, the option pool in rsi-jev) and the (Matryoshka-prefix) normalize |
| q/k norm + full RoPE + K/V store | `qwen35_attn_qk_norm_rope` via `qwen35::attn_qk_norm_rope_columns` | reused; the host entry is new (separate K and V column blocks) |
| RMSNorm, post-norm residual add with layer scale, GELU-tanh gate, scale | `nn::rms_norm_f32`, `nn::rms_norm_residual_add_f32`, `nn::mlp_gelu_tanh`, `nn::scale_f32_inplace` | reused |
| embedding gather (bf16 table) | `qwen35_embed_rows_bf16` | reused |
| GEMM | exact f32, TensorOps | reused |

The encoder attention is its own kernel rather than a mode of
`flash_attn_rows`: there the causal rule is the loop bound (keys above the
query are never visited) and the live length is one device scalar for every
row, and that body is the tuned prefill path of gemma-metal and Qwen3.5.

## Bounds (pre-registered)

Kernel tests, `tests/embedgemma2_kernels.rs` (fixtures from
`scripts/gen_embedgemma2_fixtures.py`, transformers 5.19):

* the f64 references against transformers: within `1e-5` (attention) and
  `5e-6` (norm/RoPE near position 0, PLE, pool) of the largest magnitude;
* the kernels against the f64 references and the fixtures: within `2e-5` of
  the largest magnitude, at the fixture shapes and at the real ones (window
  512, head dims 256 and 512, sequences longer than the window, ragged
  batches); padding rows exactly `+0`;
* q/k RoPE at position 8000: within `2e-3`, because transformers forms the
  angle `pos * inv_freq` in f32 (~`pos * 2^-24` rad of rounding on each side);
* a batch row bit-identical to the same sequence run alone; two runs
  bit-identical.

Model test, `tests/embedgemma2_model.rs` (opt-in, needs the checkpoint and
`tools/embedgemma2_ref/make_reference.py`'s outputs, fp32 eager reference):

* every layer's residual stream within `1e-4` of its largest magnitude;
* every embedding within `1e-4` max abs and cosine `>= 0.99999`;
* alone vs in a ragged batch within `1e-5`.

## Precision

F32 only: bf16 weights widened exactly, f32 activations, exact-f32 GEMMs.
A bf16-operand path is not written; it would be held to transformers' bf16
forward the way `tests/qwen35_model.rs` holds Qwen3.5's.
