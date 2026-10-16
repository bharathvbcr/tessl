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

**Resolved** (lane C, text; lane F, images and audio): `src/capi_embedgemma2.rs`
and `python/tessl_torch/embedgemma2.py` carry the text entries
(`tessl_embedgemma2_{load,free,embedding_dim,max_tokens,set_max_tokens,encode}`)
and the multimodal ones (`tessl_embedgemma2_{load_multimodal,encode_images,encode_audio}`),
additive at ABI 11; see `docs/embedgemma2.md`, From Python / C. The towers
(G8) are in `tessl::embedgemma2_vision` / `embedgemma2_audio` and compose with
the text encoder in `tessl::embedgemma2_mm`. **One wait (lane K):** a call with
media waited twice (the towers' soft tokens were read back and uploaded again
as host rows); the text encoder's scatter now reads the towers' output buffers
directly, so every `EmbedGemma2Multimodal::encode` call, the C entries and the
Python methods included, waits once. See `docs/embedgemma2.md`, One call.

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

**Resolved by measurement** (lane D, `embedgemma2_stage_profile_m5pro.txt`,
reconciled to 0.01 MiB): the packed gate|up *saves* 64 MiB; one activation
set at 8192 rows grew 576 → 736 MiB because the per-layer-input run buffer
is 256 MiB instead of 16. The table's figure doubles it: `encode` drops its
activations after its last wait, the pool recycles them only at the next
commit, and the next `encode` allocates first, so two sets are live (1472
MiB, 736 with a commit between calls), and the three rows are one number
carried by the freelist cache. Fixes: release before returning, a 16 MiB
per-layer-input run (no measurable time), and aliasing qkv with gate|up
(inferred 96 MiB).

**Fixed (lane H)**; numbers in `embedgemma2_memory_m5pro.txt`, fresh process
per cell, identical on a second run, **verified**:

| Peak above the model, ExactF32 | 16f5190 | now |
|---|---|---|
| one encode, 32x256 / 8x1024 / 2x4096 | 736.5 / 736.3 / 736.3 MiB | 400.5 / 400.3 / 400.3 MiB |
| four encodes back to back, same | 1472.7 / 1472.4 / 1472.3 MiB | 400.6 / 400.4 / 400.3 MiB |
| four encodes back to back, 64x32 | 433.0 MiB | 112.9 MiB |
| four encodes back to back, 1x16 | 4.25 MiB | 2.00 MiB |

Bf16 goes 1456.4 → 392.4 MiB at 8x1024 (its set is 8 MiB smaller than
ExactF32's: the half-width Q, K, V and per-layer gate save more than the four
bf16 operand copies add). The three changes, each with the test that fails
without it:

* **Release before the wait** (the owner is `encode_impl`, not the pool). A
  dropped pooled buffer waits in the runtime's deferred-release queue for the
  next waiting commit; `encode` dropped its activations after its own wait,
  so the next `encode` found the freelist empty and allocated a second set.
  Dropping them (and each forward's uploaded inputs) before the wait lets that
  same wait recycle them, and the deferred queue's strong reference keeps them
  alive until it has finished. The pool's deferral is right (a buffer dropped
  mid-batch may be bound by the open command buffer) and every other caller
  relies on it; changing it would touch Qwen3.5 training and inference for a
  fix `encode` can make alone. One host wait per `encode` is unchanged
  (`tests/async_scope.rs`, `embedgemma2_tiny`'s wait and commit counts).
  `consecutive_encodes_hold_one_activation_set` (`tests/embedgemma2_memory.rs`)
  fails on 16f5190 (three more calls reach 2,981,888 bytes above the first, the
  activations being 3,244,032) and passes now.
* **`PLE_RUN_BYTES` 256 → 16 MiB.** The embeddings do not move one bit (real
  checkpoint, both modes, six workloads; tiny model); time against the 256 MiB
  bound, one binary, is 0.96 to 1.03 (median of rounds) and 0.90 to 1.00
  (min-of-N) at 1024 to 8192 rows, inside the limits fixed in advance (one cell
  read 1.0303 on a loaded series and 1.005 on its 12-round repeat). The whole
  change against 16f5190 is min-of-N 0.86 to 1.05 over three paired series, but
  the machine's round spread (1.2 to 28) is too wide to verify a 3 % bound
  there: **unverified** at that precision, as the results file records. `the_per_layer_input_run_is_bounded_at_the_default` (a
  device-free unit test in `embedgemma2::tests`) fails on 16f5190.
* **qkv aliased with gate|up** (one buffer, `Acts::qkv` a clone of
  `Acts::gate_up`). Safe because within a layer the q/k-norm-RoPE kernel is the
  last reader of the first before the MLP's GEMM writes the second, the next
  layer's q|k|v GEMM writes it again only after `gated_act` has read gate|up,
  every dispatch is barriered against the one before, and both GEMM outputs are
  f32 in both `GemmOperands` modes (the bf16 change moved only Q, K and V and
  the per-layer gate, not these). Measured saving 96 MiB at 8192 rows, as
  inferred. `one_encode_allocates_the_itemized_set` (to the byte, both modes)
  fails on 16f5190.

**Fixed for the towers (lane K).** The vision and audio towers
(`embedgemma2_vision.rs`, `embedgemma2_audio.rs`) held their activations across
their own `synchronize` in the same shape, so a tower call followed by another
left the same second set. `queue_images` / `queue_audio` now build the
pending call from the forwards' outputs and `drop(acts)` (and each forward's
uploaded inputs) before returning, so the wait recycles them.
`consecutive_vision_calls_hold_one_activation_set` (both modes) and
`consecutive_audio_calls_hold_one_activation_set`
(`tests/embedgemma2_memory.rs`, tiny checkpoint) fail on 27a0407 (three more
calls reach 1,769,472 and 540,672 bytes above the first against bounds of one
8,192- and one 16,384-byte output) and pass now; a facade call is covered by
`consecutive_facade_calls_hold_one_activation_set` (passes on both: the old
facade's text wait recycled the towers' set before the next call, so its fix
was the peak inside the call, which is unchanged: a tower's set and the text
encoder's are live together until the one wait, **verified** by the device peak
in `bench/results/embedgemma2_mm_onewait_m5pro.txt`). The class fix in the pool
(recycle at allocation time when no command buffer is open and the lifetime
event has caught up) would cover every caller but changes when Qwen3.5's buffers
return to the freelist, and needs the re-entrancy care `flush_residency`
already shows (`active_m4` may be held); it is still a follow-up, not bundled.

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
