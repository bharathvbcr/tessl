# EmbeddingGemma 2 on Apple silicon: gap audit (2026-10-15)

Scope: `tessl::embedgemma2` at `main` 60a6240, the checkpoint
`google/embeddinggemma-2` snapshot `914f7f89…`, on the M5 Pro. This audit
lists what is still missing after the correctness work (2026-10-08) and the
nine landed speed changes (2026-10-14). Each gap is labelled **verified**
(read in code or measured in a committed result), **inferred**, or
**unverified**.

## Where it stands

* Correctness: f32-exact forward, held to the fp32 eager model within
  `1e-4` × max per layer and cosine 1.00000000 on all eight reference
  embeddings (`docs/embedgemma2.md`). **verified** (committed result).
* Speed: 1.2x to 2.9x faster than 2a7cb8b in the default runtime
  (`bench/results/embedgemma2_before_after_m5pro.txt`). **verified**.
* Text model shape (config.json): hidden 512, intermediate 2048, 24
  layers (20 sliding at window 512, head_dim 256, 2 K/V heads; 4 global,
  head_dim 512, 1 K/V head), 4 query heads, per-layer inputs 512, vocab
  262144, output 768. About 126M non-embedding parameters (inferred from the
  shapes). **verified** (config).

## Gaps

### G1. Short batches pay a commit-and-wait per dispatch (largest latency gap)

`encode` leaves the runtime's global `async_encode` flag alone
(`src/embedgemma2.rs:1006-1013`), and with it off every dispatch commits and
waits (`src/runtime.rs:1951-1963`, `with_binder_sync`). One forward is about
15 dispatches per layer times 24 layers. Measured: 1x16 tokens takes
**84.9 ms** synchronous against **7.4 ms** queued (11x); 64x32 163 → 103 ms;
large workloads 0.80-0.90x. **verified** (committed result). Every library
caller (and any future C API or Python caller) gets the slow path by default.

Fix shape: a scoped batch the runtime owns. `encode` opens it, queues every
forward, and waits once. The runtime precedent is `with_binder_sync`, which
stores and restores the flag around one closure (`src/runtime.rs:2168-2177`).
Hard question: the flag is one atomic shared by every thread holding the
`Arc<GpuRuntime>`, separate from the per-dispatch `acquire_access` lease. A
store/restore pair would (a) batch another thread's dispatches into this
encode's command buffer, and (b) clobber a flag another thread changed while
the scope was open. A scope *count* that is OR-ed with the caller's flag
avoids (b). (a) is readback-safe, because `host_access` commits and waits,
but it is a behavior change that must be stated and tested.
The const-arena exhaustion that made unwaited async poison the runtime
(ft-a3b7e231) is fixed on main by 23ca02c (`reclaim_const_arena_if_low`).
**verified** (git log on main).

Also: `pack_forwards` must use the queued per-forward cost (67 rows) once
`encode` always queues, not the synchronous one (1046). **inferred**.

### G2. No bf16 compute path; weights held as f32 (0.82 GB on device)

Every GEMM is exact f32 (`BACKEND`, `encode` refuses a relaxed runtime at
`src/embedgemma2.rs:964-968`). The f32 copy of every bf16 weight doubles the
weight bytes each forward reads. A ragged 128x8-512 encode runs 5 to 16
forwards, each re-reading the weights. `docs/embedgemma2.md` §Precision:
"A bf16-operand path is not written". The checkpoint's own bf16 forward is
at cosine 0.99990-0.99998 against fp32, which is the natural bound.
**verified** (code and doc). The size of the speedup is **unverified**.

Fix shape: a load-time precision choice (`ExactF32` default, `Bf16`). Under
`Bf16`, weights stay bf16 on device and the GEMMs take bf16 operands with
f32 accumulation. Hold it to transformers' bf16 forward and to cosine
≥ 0.9999 against fp32, as `tests/qwen35_model.rs` holds Qwen3.5.
Related, do not re-open: ft-0693aaa7 (ExactF32.nn silently runs bf16),
ft-b4d2a884 / ft-ea33f477 (GEMM routing).

### G3. No C ABI or Python entry for EmbedGemma2

`src/capi.rs` exports Qwen3.5, GDN and cross-entropy, and nothing for
EmbedGemma2 or BERT (`rg embedgemma src/capi.rs`: no match). Python users,
who hold the tokenizer, cannot reach the encoder. **verified**.
In flight beside it: `gp/w9-capi` (ABI 12, async entry, ft-ba1413a0)
changes `capi.rs`, so new exports belong in their own module.

### G4. Tokenization is the caller's

There is no tokenizer in tessl (`rg -l tokeniz src` finds only doc comments).
A Rust Gemma tokenizer would need either the `tokenizers` crate (a new
dependency, which needs the owner's decision) or a hand-written BPE over
`tokenizer.json` with a Python reference oracle. The Python binding (G3) can
take a caller's `tokenizers.Tokenizer` and apply the checkpoint's prompts
(`config_sentence_transformers.json`) without a new Rust dependency.
**verified** (no tokenizer); the decision is the owner's.

### G5. Where an encode's time goes is not measured per stage

The committed results time whole encodes. There is no per-stage split
(projections, attention sliding vs global, the per-layer-input GEMMs, norms
and gates, pooling), so the next kernel to fuse or rewrite is unranked.
`segment_mean_rows` alone was measured (< 0.2 %). **verified** (absence).

### G6. Device peak rose 29 % on single-forward workloads, cause not isolated

1283 → 1652 MiB above the model at 32x256 / 8x1024 / 2x4096; the cause is
"inferred … not isolated" (packed gate|up, per-layer-input run buffer).
**verified** (committed result says so).

### G7. Many small elementwise dispatches per layer

Per layer: three RMSNorm-residual adds, two RMSNorms, two gated activations,
a scale and a norm for the per-layer inputs at each run start, around six
GEMMs. Fusing norm-into-GEMM prologue or gate-into-GEMM epilogue
(`gemm_epilogue` exists) would cut dispatches and activation traffic.
Value **unverified** until G5 ranks it.

### G8. Two thirds of the checkpoint are not implemented: the vision and audio towers

`google/embeddinggemma-2` is a 744.4M-parameter multimodal embedder, all
bf16 (safetensors header census, `tools`-free: summed shapes per prefix).
tessl loads only `language_model.*`, the text encoder:

| Part | Parameters | In tessl |
|---|---|---|
| `language_model` (text encoder; 134.2M of it the token table) | ~271M | yes |
| `vision_tower` (16-layer ViT, hidden 768, 12 heads of 64, patch 16) | ~167M | no |
| `audio_tower` (12-layer conformer, hidden 1024, 8 heads of 128) | ~305M | no |
| `embed_vision`, `embed_audio` (weightless RMSNorm + projection to 512) | 1.2M | no |

**verified** (header census, config). How transformers 5.19 composes them
(`EmbeddingGemma2Model.forward`): each tower's output goes through its
embedder to 512 columns, and those soft tokens replace the rows of the
`image_token_id` (258880) / `audio_token_id` (258881) placeholders in the
text sequence. The placeholders are gathered as `pad_token_id` and then
overwritten, and the soft tokens are not scaled by `sqrt(hidden)`. The text
encoder then runs unchanged on the merged embeddings, the per-layer inputs
included, followed by sentence-transformers' mean pool and normalize.

Vision tower: `pixel_values` (patches of `3*16*16`) mapped by `2x - 1`, a
projection, plus a 2-D position embedding (x and y tables summed, zero on
padding); 16 layers of RMSNorm (`x * w`), q/k RMSNorm with weights and a
weightless v norm, 2-D axial RoPE (theta 100; the head's halves rotate by x
and by y), bidirectional attention at scale 1.0, gated GELU-tanh MLP, the
same four norms as the text layers; then a 3x3 average pool by patch
position to at most 280 soft tokens, times `sqrt(768)`.

Audio tower: 128-bin log-mel `input_features` through two
`Conv2d(3x3, stride 2)` + LayerNorm (no bias) + ReLU (1 → 128 → 32
channels), a projection to 1024; 12 conformer layers (half-weighted FFN,
chunked local attention with chunk 12, 12 frames of left context, no right
context, Transformer-XL relative position logits, a per-dimension softplus
query scale, a 50 logit softcap, a GLU + depthwise causal conv(5) block,
a second FFN), every linear clamped to scalar input/output bounds stored in
the checkpoint; then `output_proj` (1024 → 1536, with bias).

Preprocessing (image resize and patchify, the audio log-mel front end) stays
the caller's, as tokenization does; tessl starts from the processor's
tensors. The goldens are `tools/embedgemma2_ref/make_mm_reference.py`.

### Already on the board (referenced, not re-filed)

* ft-d0a49f5d: re-measure `attn_tiled.h` for the EG2 encoder after 4e5faac.
* ft-c8e7726b: shared config parsing and batch validation across BERT,
  EmbedGemma2 and Qwen3.5.
* ft-f5d974317: panicking `write_u32` uploads on library paths (EmbedGemma2
  included).
* ft-4efcd42f: EG2 speed card (forward cost, overlap, segment mean), in review.

### Checked and not a gap

* Sliding-window attention skips key tiles outside the window
  (`kernels/attn_tiled.h:51-54`), so sliding layers are linear in length.
  **verified** (code).
* Pooling cost: < 0.2 % of an encode. **verified** (committed result).

## Work plan (this session)

Four lanes, each in its own worktree and target directory, with files
partitioned so no two lanes edit the same region:

| Lane | Gap | Files |
|---|---|---|
| A | G1 scoped batch | `src/runtime.rs` (scope API), `src/embedgemma2.rs` `encode()` only, tests |
| B | G2 bf16 path | `src/embedgemma2.rs` load/Layer/forward, tests, bench flag |
| C | G3 C ABI + Python | new `src/capi_embedgemma2.rs`, `python/tessl_torch/embedgemma2.py`, tests |
| D | G5 + G6 measurement | `bench/results/`, a profiling bench binary |

Then, as lanes free up (at most four agents at once): M1 the vision tower
and M2 the audio tower, each held layer by layer to the multimodal goldens,
and M3 the merge (placeholder rows overwritten by soft tokens before the
text encoder), held to sentence-transformers' image and audio embeddings.

G4 waits for the owner's decision on a tokenizer dependency. G7 waits for
G5's ranking. GPU benchmarks run one at a time, paired and interleaved
(`bench_paired`), because run-to-run noise on this machine is about ±30 %.
