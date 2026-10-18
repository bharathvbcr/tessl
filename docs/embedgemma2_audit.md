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
additive (no version bump of their own; they ship at ABI 12); see `docs/embedgemma2.md`, From Python / C. The towers
(G8) are in `tessl::embedgemma2_vision` / `embedgemma2_audio` and compose with
the text encoder in `tessl::embedgemma2_mm`. **One wait (lane K):** a call with
media waited twice (the towers' soft tokens were read back and uploaded again
as host rows); the text encoder's scatter now reads the towers' output buffers
directly, so every `EmbedGemma2Multimodal::encode` call, the C entries and the
Python methods included, waits once. See `docs/embedgemma2.md`, One call.

**Closed further (2026-10-17).** The C and Python entries no longer need a
tokenizer or a processor of the caller's: eleven raw front-end entries
(`tessl_embedgemma2_{embed_texts,tokenize,decode,prompt_*,embed_images,embed_audio,embed_videos,embed_media}`,
merge 641dba8) and seven for files, bytes, soft-token counts and text
interleaved with media (`tessl_embedgemma2_{embed_*_sources,*_soft_token_count,embed_segments}`,
merge 879ead2), all in `src/capi_embedgemma2.rs` (`rg -o 'extern "C" fn
tessl_embedgemma2_\w+'` lists 27 entries); `tests/capi_embedgemma2_raw.rs` and
`tests/capi_embedgemma2_media.rs`. A Go binding over the same ABI is
`go/embedgemma2` (merge ec4c3ec; `make -C go device` compares every entry bit
for bit with the Rust call). Python's `embed_images` / `embed_audio` now take
raw images and PCM (`CHANGELOG.md`, Changed (breaking)).

### G4. Tokenization is the caller's

There is no tokenizer in tessl (`rg -l tokeniz src` finds only doc comments).
A Rust Gemma tokenizer would need either the `tokenizers` crate (a new
dependency, which needs the owner's decision) or a hand-written BPE over
`tokenizer.json` with a Python reference oracle. The Python binding (G3) can
take a caller's `tokenizers.Tokenizer` and apply the checkpoint's prompts
(`config_sentence_transformers.json`) without a new Rust dependency.
**verified** (no tokenizer); the decision is the owner's.

**Resolved (lane T, merge 5a2e9c6).** `tessl::embedgemma2_tok` is a hand-written
BPE over the snapshot's `tokenizer.json`: no new dependency, no Python.
`tests/embedgemma2_tok_tiny.rs` (device-free, three hand-written tokenizers) and
the opt-in `tests/embedgemma2_tok_real.rs` (4671 texts: ids equal
sentence-transformers' in 4680 of 4680 comparisons, the nine refused strings
counted twice) hold it to exact id equality (`docs/embedgemma2.md`,
Tokenizer). The Python and C text entries tokenize through it.

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

**Partly resolved (lane X, merge 201abfc).** The row passes are fused where
the arithmetic stays bit for bit (`kernels/embedgemma2_text.metal`): 15 to 13
dispatches per layer at `ExactF32`, 17 to 13 at `Bf16`. Embeddings are
unchanged to the bit (tiny: `tiny_embedding_hashes` in
`tests/embedgemma2_memory.rs`; real checkpoint: 12 hashes at both precisions,
`docs/embedgemma2.md`, Text-encoder fusion) and
`a_layer_runs_the_pinned_kernels` / `bf16_a_layer_runs_the_pinned_kernels`
(`tests/embedgemma2_tiny.rs`) pin the layer. **No end-to-end speed-up is
claimed**: `bench_paired` refused both runs on its own drift gate (spreads up to
19.3x on a host at load 66 to 108). Still open: `gelu(gate) * up` into the
`gate|up` GEMM, dropping the dead f32 `a.attn` buffer at `Bf16`, GEMMs that
write bf16 directly (`docs/embedgemma2.md`, What is left (text-encoder
fusion)). The audio tower's version of this work is lane I (ae11582).

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

**Resolved** (towers M1, M2; merge M3; facade F; one wait K, as in the table
below). **Preprocessing is no longer the caller's** (lane P, merge f8ce932):
`tessl::embedgemma2_pre` makes the towers' inputs from raw bytes, PCM and
frames, bit-exact for images and about one ulp for the log-mel
(`tests/embedgemma2_pre.rs`, device-free; `docs/embedgemma2.md`,
Preprocessing). Decoding the files into those bytes is `tessl::media`
(`docs/embedgemma2.md`, Media decoding).

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

## Status (closed 2026-10-16, updated 2026-10-17)

The first wave (lanes A to K) was merged into `main` as 8c80af7 (`git
merge-base --is-ancestor 8c80af7 main` holds); `main` is no longer 60a6240. The
later lanes P, T, X, AB, R, F2 and V (video input, merge 6f62a50, not the
decode lane V below) are ancestors of 8c80af7 too (`git merge-base
--is-ancestor` on each merge commit, **verified**). The lanes after it (PY3, D,
BIND, M, CR, J, V (decode, 4d06651), G, V2) are on `eg2/integrate` (e2c345f at
this update). Every merge of the first wave was
verified on the merged tree on 2026-10-16: fmt, clippy and rustdoc with `-D
warnings`, the device test binaries, the six real-checkpoint suites
(`--ignored`), `scripts/hosted_tests.sh` and `--outside`, and the Python
binding tests. **This update reads the table below from the code and the
docs; it did not rerun any of that.** Whole-encode result:
`bench/results/embedgemma2_final_m5pro.*`.

| Gap | Status | Merge | Evidence |
|---|---|---|---|
| G1 per-dispatch waits | resolved: `GpuRuntime::async_scope`; `encode` waits once | d13d1d3 (A) | `tests/async_scope.rs`; 1x16 84.9 -> 7.4 ms (`embedgemma2_before_after_m5pro.txt`) |
| G2 no bf16 path | resolved: `GemmOperands::Bf16` (B), bf16 attention tiles (E), vision tower bf16 (J), audio tower bf16 (AB: `EmbedGemma2Audio::load_with`, weights 587,121,792 B against 1,172,751,488 B; its speed is not published, the drift gate refused every run) | e5be45e, 16f5190, 224be3f, 005063c | `tests/embedgemma2_model.rs`, `tests/embedgemma2_audio_tiny.rs` (`bf16_mode_is_not_the_exact_forward`); 2.75x to 3.0x faster than 60a6240 from 64x32 up (`embedgemma2_final_m5pro.txt`) |
| G3 no C ABI / Python | resolved: text (C), multimodal (F); raw front ends (BIND), files, bytes, counts and segments (M), Go (G) | ac5364a, 433e659, 641dba8, 879ead2, ec4c3ec | `tests/capi_embedgemma2*.rs`, `tests/capi_contract.rs`, `python/tests/test_embedgemma2*.py`, `go/embedgemma2` (`make -C go device`) |
| G4 tokenizer | resolved: `tessl::embedgemma2_tok`, a hand-written BPE, no new dependency (T) | 5a2e9c6 | `tests/embedgemma2_tok_tiny.rs`; `tests/embedgemma2_tok_real.rs` (opt-in): 4680 of 4680 comparisons equal sentence-transformers' |
| G5 stage profile | resolved (D); towers profiled (I, J) | 2e7ba35 | `embedgemma2_stage_profile_m5pro.txt`, `embedgemma2_{audio,vision}_profile_m5pro.txt` |
| G6 device peak | resolved: 1652 -> 466 MiB above the model (H); towers release before their wait (K); the pool recycles a dropped buffer at the next allocation when the GPU is idle (R: `recycle_if_idle`, consecutive calls hold `extra == 0`) | 27a0407, 03dae0f, 1f447c9 | `tests/embedgemma2_memory.rs`, `tests/pool_recycle_at_alloc.rs`, `embedgemma2_memory_m5pro.txt` |
| G7 elementwise dispatches | partly: text encoder fused 15 -> 13 dispatches per layer at `ExactF32`, 17 -> 13 at `Bf16`, bits unchanged (X); audio tower 28 -> 17 (I); vision RoPE table (J). No end-to-end speed-up shown for X or I: the drift gate refused X's runs, I's was inside the noise | 201abfc, ae11582, 224be3f | `tests/embedgemma2_tiny.rs` (`a_layer_runs_the_pinned_kernels`), `tiny_embedding_hashes`, `embedgemma2_audio_profile_m5pro.txt` |
| G8 vision and audio towers | resolved: towers (M1, M2), merge (M3), facade (F), one host wait per media call (K); preprocessing in Rust (P); video input in Rust (V of 6f62a50, F2) and decoded natively (D bd993d4, the decode lane V 4d06651, V2 e2c345f) | 6eedd5a, 06465a9, 908c3a2, 433e659, 03dae0f, f8ce932, 6f62a50, 8967e31, bd993d4, 4d06651, e2c345f | `tests/embedgemma2_e2e_model.rs`: max abs 3.8e-7 against sentence-transformers; `tests/embedgemma2_pre.rs`; `tests/embedgemma2_video_tiny.rs`; `tests/media_e2e_video.rs`: 1.38e-7 to 8.53e-7 |
| 1x16 latency (from G5) | resolved for ExactF32: opt-in split-K for the text encoder only (G) | f1a8677 | 1x16 0.60x (`embedgemma2_small_m_m5pro.txt`, `embedgemma2_final_m5pro.txt`) |

Found on the way:

* transformers 5.19's eager attention inverts the audio tower's mask
  (`logical_not()` of a float mask); the multimodal goldens are now sdpa, and
  `eager_references_encode_the_inverted_mask` pins the disagreement (06465a9).
* `embedgemma2_memory` read a process-wide device figure, so its result
  depended on test order; device-byte tests now run in a child process (7459e16).

Closed since 2026-10-16, each from the code: G4 (T, above); an audio tower bf16
mode (AB, `src/embedgemma2_audio.rs` `load_with`); video (`MmItem::Video` and
`MmItem::TimestampedVideo` in `src/embedgemma2_mm.rs`, `tessl_embedgemma2_embed_videos`
in the C ABI, frames decoded by `tessl::media`); a C/Python entry for
caller-built ids with media (`tessl_embedgemma2_embed_media`, Python
`media([{"ids": ...}])`); the pool-level recycle at allocation time (R,
`src/runtime.rs` `recycle_if_idle`).

Still open:

* G7: `gelu(gate) * up` into the `gate|up` GEMM, dropping the dead f32 `a.attn`
  buffer at `Bf16`, and GEMMs that write bf16 directly (each changes bits or the
  itemised bytes some test asserts), and the end-to-end speed of the fusion that
  landed, which needs a quiet host (`bench_paired --evidence --rounds 6
  --max-spread 1.25`).
* The audio tower's end-to-end speed-up, fused and `Bf16`, on a quiet machine
  (`docs/embedgemma2.md`, Audio tower bf16 and speed: refused by the drift gate
  in all six attempts).
* The media list in `docs/embedgemma2.md`, What is left: torchcodec itself was
  never run (the video reference is the `ffmpeg` CLI arranged as its source
  arranges libswscale); one lossless-JPEG shape refused by name and block
  smoothing / lossless rules measured against the oracle, not read from the
  library source; `AVAssetReader`'s chroma resampling characterised, not
  reproduced; VideoToolbox here does not decode H.264 High 10, 4:2:2 or 4:4:4;
  the threaded colour conversion's cost is unmeasured; Ogg Opus in SILK or
  hybrid mode is 15 to 20 dB from a torchcodec reference; librosa's `soxr_hq`
  and torchaudio's resampler are not reproduced; HEIC, H.264 and HEVC tests
  cannot run in the hosted sandbox; bit equality is with the AArch64 NEON
  FFmpeg (x86 unverified); a file's own channel mask is not read.
