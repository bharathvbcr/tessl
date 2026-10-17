# Changelog

All notable changes to `tessl` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **EmbeddingGemma 2 multimodal call: one host wait, towers' soft tokens stay on
  the device** (audit gaps G6, G8; `EmbedGemma2Multimodal::encode`, so the C
  `tessl_embedgemma2_{encode_images, encode_audio}` entries and Python's
  `encode_images` / `encode_audio` / `embed_*` too; no signature or layout
  changed). A call with media waited twice: the towers' soft tokens were read back,
  concatenated per item on the host and uploaded again as host rows for the text
  encoder. The facade now queues the towers and then the text encoder behind one
  `synchronize`, and the text encoder's scatter reads the towers' output buffers
  where they were written: the soft source is one shape, per sequence a list of
  `(buffer, first row, rows)` segments in placeholder order, for a host caller's
  `MmSequence::soft` (uploaded as one buffer) and for the towers alike, so there
  is one `soft_io` / `soft_runs` / `scatter_soft` path, not two. A vision image's
  rows are one segment; an audio clip's are one per maximal run of rows its output
  mask keeps; nothing is compacted (`copy_cols` already takes runs). The text
  encoder's `encode_impl` is split into `queue_impl` (a `PendingEncode` holding
  only each forward's output) and `finish`; `encode` and `encode_multimodal` are
  queue + wait + finish. The host path's finiteness check of soft values cannot run
  on device rows, so a media item's embedding is checked finite after the wait
  (an overflowing tower is an error naming the item). Every refusal still
  happens before any GPU work. Host waits: 2 -> 1 for a call with media (text
  alone and the by-hand composition's 1 and 3 unchanged); embeddings bit for bit
  the two-wait path's on the tiny checkpoint in both operand modes and on the real
  one (hashes `922bb84e7eb7eef9` / `3f25a99974a7687c` / `c37bcf8a3961f5e5` for
  the three benchmark workloads, equal at 27a0407). The multimodal tests' wait
  bounds changed from 2 to 1 because the behaviour did. **No speedup is
  claimed**: interleaved against 27a0407 (8 rounds, control lane) the calls are
  1089 / 414 / 401 ms against 1092 / 413 / 391, medians 1.003x / 0.995x / 1.023x
  with ranges straddling 1 and the control's own spread as wide
  (`bench/results/embedgemma2_mm_onewait_m5pro.{txt,json,png}`); the removed wait
  is about a millisecond of a 380 to 1070 ms call.
- **EmbeddingGemma 2 towers release their activations before their wait**
  (`queue_images` / `queue_audio`; audit G6, the part lane H left): a tower call
  followed by another held two activation sets because the pending call kept
  `acts` and the uploaded inputs until `collect`, after its wait; they are now
  handed to the runtime's deferred release before the call returns, so the wait
  recycles them, and only each forward's output is kept. Tiny checkpoint, calls
  2 to 4 above the first: 1,769,472 B (vision) and 540,672 B (audio) against
  bounds of one output (8,192 and 16,384 B) before, within them after;
  `tests/embedgemma2_memory.rs` fails on 27a0407 for both towers. The facade's
  own device peak is unchanged (one tower's set and the text encoder's are live
  together until the wait): 628.6 -> 628.5, 653.8 -> 653.6 MiB.

### Added

- **C ABI 12** (additive; `TESSL_ABI_VERSION` 11 → 12):
  `tessl_runtime_set_async_encode`, and `tessl_qwen35_read_parameter_host` /
  `tessl_qwen35_write_parameter_host` for one parameter-table entry. A per-entry
  write is refused when the entry carries AdamW aux state (an `F32Master` or
  `Bf16Kahan` buffer), which the next optimizer step would copy back over it;
  the message points to `TESSL_WRITE_ADAMW_AUX` or to writing before the
  optimizer exists. `TESSL_WRITE_PARAMS` is unchanged. `tessl_torch` turns async
  encode on, finishes torch's queue before handing buffers over, and gains
  `read_parameter` / `write_parameter`. `bench_capi` and
  `bench/capi_torch_overhead.py` measure the waits (2B `train_step` 1660 →
  1507 ms, `bench/results/capi_*`).
- **Untied Qwen3.5 LM head** (`tie_word_embeddings: false`): `lm_head.weight` is
  its own parameter, read by the same loader as the embedding. A tied config
  whose checkpoint carries `lm_head.weight`, disagreeing root and `text_config`
  flags, and a config silent on the key are refused.
- **Checkpoint reader bounds.** `SafeTensors::open_with_limit`,
  `CheckpointView::open_with_limit` and `npy::read_npy_with_limit` refuse,
  before allocating, a declared payload above the caller's limit (default:
  physical memory); each read's own allocation (an f32 widening of a 16-bit
  tensor) is budgeted too. `CheckpointView::tensor_info` gives a tensor's dtype
  and shape across shards.
- **Fixture manifests.** `tests/fixtures/qwen35/`, `qwen35_train/`,
  `qwen35_train_grouped/` and `qwen35_train_untied/` carry `MANIFEST.json`
  (torch / transformers / numpy versions, generator and commit, input source,
  per-file `published`); `tests/fixture_manifests.rs` fails for a fixture
  directory without one. The Qwen3.5 gate kernel has a golden captured from
  transformers' own module (`qwen35_gates_*`).
- **Bench lanes.** `bench_qwen35_layers --attn-tiles`, `bench_qwen35_train
  --op=NAME` (validated before the runtime; METRIC names
  `op_KEY_{f32|bf16}_ms_tT`), `bench_embedgemma2 --attn-kernel=rows|tiled`
  (shapes validated up front, tiled checked against rows before timing);
  each prints `METRIC_INCOMPLETE` if it fails after starting.
  `bench/results/attn_tiled_rebench_2026-10-15_m5pro.json`: no regression in
  any `attn_tiled.h` user.
- `src/bin/probe_soak.rs` (the leak-soak driver behind the GPU re-run record in
  `docs/verification.md`), and a fork-safe gate on every self-hosted CI job
  (`tests/ci_workflow_gates.rs`).

- **`bench_embedgemma2_mm`** (`src/bin/bench_embedgemma2_mm.rs`): the facade's
  `encode` on the real checkpoint, three workloads (`mixed_full`, `mixed_short`,
  `media_only`), `bench_paired` format, a HASH line per workload and the
  `sync_waits` of a call.

- **EmbeddingGemma 2 vision tower: a `GemmOperands::Bf16` mode, measured and
  sped up** (`EmbedGemma2VisionModel::load_with(rt, st, cfg, operands)`; `load`
  is `ExactF32`, bit for bit what it was: tiny hash `a1e38036581d2780`, real
  checkpoint 1 image `d9a0d85dd2f8f279` and 8 images `4f48fd6b310f86a5`, equal
  to the build at `433e659`). `Bf16` keeps the 16 layers' projections bf16 on the
  device (335.6 MB against 639.6 MB), runs every GEMM bf16 x bf16 with f32
  accumulation through the text encoder's own `embedgemma2::project`, the
  attention on `encoder_attn_bf16` (head dim 64), and keeps the residual stream,
  norms, RoPE arithmetic, pool and embedder norm f32; new kernel
  `vision_qkv_norm_rope_bf16` stores bf16 with the f32 kernel's rounding (bit
  equal to rounding its output). The facade's one `GemmOperands` now covers the
  text encoder and the vision tower (the audio tower stays exact f32; the C ABI
  `operands` and Python `operands=` documentation say so). Held to the f64
  forward over the same bf16 weights on a tiny checkpoint (relative bounds
  `2e-2` patch embedder, `5e-2` layers and soft tokens, token cosine `0.999`,
  fixed before the first run; observed `9.5e-4`, `6.7e-3`, `5.7e-3`, `0.99998`;
  batched, reversed and split calls bit-identical) and, on the real
  checkpoint, image embeddings end to end (bf16 tower and text encoder) at
  cosine `0.99998769` and `0.99998850` against fp32 (bound `0.9999`), mean
  `1 - cos` `1.19e-5` against transformers' own bf16 run's `2.61e-5`. New bins
  `bench_embedgemma2_vision` and `profile_embedgemma2_vision` (a replica of the
  forward checked bit-equal to `encode_images`, with the quiet-machine gate
  that now lives in `src/bin/common`). Speed on an Apple M5 Pro, real
  checkpoint, paired interleaved rounds on a shared GPU: `Bf16` is 2.8x to 3.1x
  faster than `ExactF32` (1 image 63.5 against 187 ms, 8 images 547 against
  1683 ms over 8 rounds, spreads and the contended rounds in
  `bench/results/embedgemma2_vision_speed_m5pro.txt`); the rope kernel, formerly
  one `cos`/`sin` table per head, now forms one per patch for up to six heads
  (8.4 to 3.1 ms at one image, output bits unchanged). Stage ranking before and
  after: `bench/results/embedgemma2_vision_profile_m5pro.txt`. The multimodal
  tiny fixture's vision tower is two heads of 64 (the bf16 attention is not
  compiled for 32).
- **EmbeddingGemma 2, text, images and audio in one call, from Rust, C and
  Python** (`tessl::embedgemma2_mm`: `EmbedGemma2Multimodal::{load, load_from,
  from_parts, encode, text, vision, audio}`, `MmItem`; audit gaps G3, G8). One
  `config.json` read and one safetensors open load the text encoder and both
  towers; `encode` embeds a batch of text ids, images, clips and caller ids
  with placeholders (`MmItem::Media`, the `j`-th image or audio placeholder
  takes the `j`-th soft row of that modality) in one call, with every refusal
  before any allocation or wait. It is a composition of the existing owners:
  the towers gain `check_images` / `check_clips` (their pre-GPU checks, as a
  soft-token count per item), `queue_images` / `queue_audio` (no wait;
  `PendingImages` / `PendingAudio` hold the buffers, wait on drop, and
  `collect`) and `encode_images` / `encode_audio` become queue + wait +
  collect, now queued under a `GpuRuntime::async_scope` (before, a runtime in
  its default mode committed and waited per dispatch);
  `EmbedGemma2Model::check_sequences` is `encode`'s id checks, shared. A call
  with media waited twice (the towers, then the text encoder: the soft tokens
  crossed the host), a call of texts once, three for the by-hand composition;
  the GPU-resident hand-off that made it one was a later change, recorded in
  Changed above. `GemmOperands`
  applies to the text encoder and, from the vision-tower bf16 entry below, the vision tower; the audio tower stays exact f32.
  C ABI, additive (no version bump of its own; this release ships them at ABI 12, see C ABI 12): `tessl_embedgemma2_load_multimodal` and
  `tessl_embedgemma2_{encode_images, encode_audio}` take a processor's
  `pixel_values` / `image_position_ids` / `input_features` / mask as flat
  arrays with CSR offsets, bound every length before it sizes a slice, and
  write the output only on success; `tests/capi_contract.rs` holds them to the
  signature rules (it learned `c_uint8`). Python: `EmbedGemma2.from_pretrained(dir,
  multimodal=True)`, `encode_images`, `encode_audio` (numpy or torch, duck-typed,
  never imported) and `embed_images` / `embed_audio` (a processor you pass in);
  no new dependency. Bit for bit the by-hand composition on a tiny checkpoint;
  on the real one (opt-in) texts, images and clips in one call within 3.8e-7 of
  sentence-transformers, cosine 1.000000000, in Rust and Python. See
  `docs/embedgemma2.md`, One call.
- **An opt-in small-M exact-f32 NN GEMM path: parallel K-partitions**
  (`gemm_small_m`, `gemm_nn_splitk_par_f32`, kernel
  `matmul2d_tensorops_nn_splitk_par_f32`, `bench_gemm_routes` lane `nnsmall`,
  `profile_embedgemma2` `PROFILE_ROWS`). An exact-f32 NN GEMM whose C is a few
  tiles (N up to 512, K of 512 or more, under 64 tiles of 32x32: M up to 96 at
  N 512) used to run one single-simdgroup threadgroup per tile, each walking
  all of K, so its time was that walk's latency (EmbeddingGemma 2 at 1x16: the
  down, o and per-layer-input GEMMs at 17-22 % of the copy bandwidth). A call
  to `gemm_small_m` now runs every K-partition's tiles in one dispatch into a
  scratch of one slice per partition and `reduce_partitions_f32` adds the
  slices in partition order: the parallel TN split-K's design (whose binder it
  shares, now `splitk_par` over `par_shape`, the one owner of its `u32` and
  scratch limits) for the NN layout. The width is sized from one tile row, so
  it does not move with M inside the gate. **The opt-in is the call's**, a
  parameter and not a setting of the runtime or the process: `gemm`,
  `gemm_tiled`, `GemmOperands::ExactF32.nn`, bf16, f16, relaxed f32, TN, NT and
  every other entry are unchanged and compute the bits they did, so the vision
  and audio towers (through the shared `embedgemma2::project`), Qwen3.5 and
  external consumers are untouched. The scratch is the caller's
  (`scratch: Some(tensor)`, an idle f32 tensor of at least `ceil(K / k_tile) *
  M * N` elements; one too small keeps the single dispatch) or, with `None`, a
  pool allocation. `EmbedGemma2Model`'s text encoder is the one caller that
  opts in (`ExactF32` only), with its idle `qkv | gate_up` buffer as the
  scratch, so a forward allocates nothing more
  (`tests/embedgemma2_small_m.rs`). The sum is rounded in another order than
  the single dispatch, so **the text encoder's bits at these shapes depend on
  M across the gate** (M up to 96 and M of 97 or more at N 512; the result is
  deterministic), within the f32 dot-product bound `gamma_(K+8) * sum|a.b|`
  every exact-f32 GEMM is held to (worst result 0.0039 of it,
  `tests/gemm_small_m.rs`: M in {1, 7, 16, 63, 64} at K/N of 2048/512,
  1024/512 and 512/512, ragged and offset views, a poisoned scratch, a
  caller's scratch, bit-for-bit repeats, rank-one operands, the route by
  traced kernel with and without the opt-in) and within the existing
  alone-against-batched bounds (a text alone against batched: 0 to 1.7e-7 on
  the real checkpoint, 8.9e-8 to 1.3e-7 at the real GEMM shapes with synthetic
  weights, against `BATCH_ABS` 1e-5). Measured on an M5 Pro with the
  interleaved `bench_paired` (`bench/results/embedgemma2_small_m_m5pro.txt`):
  EmbeddingGemma 2's 1x16 encode takes 0.60x of the time (1.66x faster), 1x64
  0.70x; 64x32 and 8x1024 are unchanged (1.000x), the real checkpoint's device
  peak too. A bf16 form of the lane was built and removed (it moved the bf16
  text alone-against-batched check by 2.4e-4 to 5.7e-4; commit `b85f5e1`).

- **EmbeddingGemma 2's multimodal merge** (`EmbedGemma2Model::encode_multimodal`,
  `MmSequence`, `EmbedGemma2Config::{image_sequence, audio_sequence,
  special_tokens}`, `SpecialTokens`, `MmTokens`; audit gap G8, the part that
  does not need a tower). A sequence's soft tokens (`[n, hidden]` f32, the
  towers' 512-wide outputs) replace, in order and unscaled, the rows of its
  image, video and audio placeholder ids, which are embedded as
  `pad_token_id` through the scaled table first, before the per-layer inputs
  and the residual stream are formed: what `EmbeddingGemma2Model.forward`
  does. `encode` is the special case with no soft rows and shares the
  batching, packing and queueing (one host wait per call); the scatter is
  `qwen35_bwd::copy_cols`, one flat copy per run of consecutive placeholder
  rows, so there is no new kernel. A missing id, a ragged or non-finite
  `soft`, or a soft-row count different from the placeholder count is
  refused before any GPU work. The config now keeps the special token ids
  (a text-only config still loads). Held to sentence-transformers' image and
  audio embeddings from golden soft tokens (`tests/embedgemma2_mm_model.rs`,
  opt-in; fp32 embeddings within 2.1e-7 max abs, bf16 cosine 0.999995 or
  better against fp32) and to an f64 forward with an `inputs_embeds`
  override (`tests/embedgemma2_mm_tiny.rs`). The f64 host forward and its
  bounds moved from `tests/embedgemma2_tiny.rs` to `tests/common/embedgemma2.rs`
  to be shared. (The C and Python entries then embedded text only; see the
  entry above.)
- **bf16 operand tiles for the encoder attention, and the matrix-unit kernel at
  head dims 32 and 64** (`kernels/encoder_attn_tiled.h`, `encoder_attn.metal`,
  `kernels/tune/encoder_attn_tune.metal`; `embedgemma2::encoder_attn_bf16`,
  `EncoderAttnKernel::TiledBf16`, `nn::attn_kv_capacity_bf16`).
  `attn_tiled.h`'s template now takes its Q/K/V element type and its P tile
  type separately; the f32 instantiations (Qwen3.5 prefill, the training LSE
  forward, the ExactF32 encoder) compile to the same IR as before.
  Under `GemmOperands::Bf16` the EmbeddingGemma 2 attention reads bf16 Q, K and
  V and keeps scores, softmax, accumulator and output f32; P is rounded to
  bf16 and the row sum is taken over the rounded weights. The q/k-norm-RoPE
  writer emits bf16 directly (`qwen35_attn_qk_norm_rope_bf16`), so there is no
  cast pass. Geometries come from a measured sweep (BQ x BK x NSG): D=64 f32
  and bf16 `q32_k32_sg2`, D=32 f32 `q16_k32_sg1`, D=256 bf16 `q32_k64_sg4`,
  D=512 bf16 `q32_k128_sg8`. `encoder_attn` now routes head dims 32 and 64 to
  the tiled kernel (2.3x and 2.8-2.9x over the rows kernel). Measured on an M5
  Pro, the bf16 kernel is 3.0-3.1x (D=256), 3.6-3.8x (D=512) and 2.5-2.6x
  (vision D=64) faster than the f32 tiled kernel; the bf16 model forward is
  0.87x / 0.66x / 0.54x the time at 32x256 / 8x1024 / 2x4096. ExactF32 output
  hashes are unchanged. New bench `bench_encoder_attn` (paired, min-of-N,
  `BENCH_SWEEP=1` for the tune candidates); numbers and method in
  `bench/results/embedgemma2_attention_m5pro.txt`. The bf16 kernel is held to
  an error bound derived from the bf16 unit roundoff in
  `tests/embedgemma2_kernels.rs`.
- **A bf16-operand EmbeddingGemma 2** (`EmbedGemma2Model::load_with(.., GemmOperands::Bf16)`,
  `operands`, `weight_bytes`; `load` stays `ExactF32`, bit-for-bit unchanged).
  The projections stay bf16 on the device (517 MiB of weights against 778 MiB
  widened to f32) and all eight GEMMs run bf16 x bf16 with f32 accumulation;
  the residual stream, norms, attention and pooling stay f32. Held to
  transformers' bf16 forward (cosine 0.9999 or better against fp32, no worse
  than its mean) in `tests/embedgemma2_model.rs`, and to an f64 forward over
  the same bf16 weights in `tests/embedgemma2_tiny.rs`. `encode` does not
  refuse the runtime's relaxed-f32 switch at `Bf16`, since no f32 GEMM runs.
  `bench_embedgemma2` takes `BENCH_PRECISION=f32|bf16`.
- **EmbeddingGemma 2 from Python and C** (`src/capi_embedgemma2.rs`,
  `python/tessl_torch/embedgemma2.py`; audit gap G3).
  `tessl_embedgemma2_{load,free,embedding_dim,max_tokens,set_max_tokens,encode}`
  load a snapshot directory and embed flat `u32` ids with CSR offsets into
  `f32 [n_seqs, dim]`, validating every pointer and length before any
  allocation or GPU work and writing the output only on success. Python gets
  `EmbedGemma2.from_pretrained(dir).encode(ids, truncate_dim)` and
  `encode_texts(texts, tokenizer, prompt_name)`, which prepends the snapshot's
  prompt and duck-types a `tokenizers` or transformers tokenizer (neither is
  imported). Additive: no version bump of their own (this release ships them at ABI
  12, see C ABI 12), and a library built before these entries is named as
  such by the binding. `capi.rs` only widens
  `Handle`, `guarded`, `free_handle`, `checked_slice` and `checked_slice_mut`
  to `pub(crate)`. The tiny-checkpoint builder moved from
  `tests/embedgemma2_tiny.rs` to `tests/common/embedgemma2.rs`, and
  `tests/capi_contract.rs` holds the new module and Python file to the same
  signature rules as the originals.
- **EmbeddingGemma 2 vision tower and multimodal embedder**
  (`tessl::embedgemma2_vision`: `VisionConfig`, `EmbedGemma2VisionModel::{load,
  encode_images}`, `ImageInput`, `plan_pooling`). The 16-layer ViT
  (`vision_tower.*`) and `embed_vision.*` map the processor's patches and
  positions to the 512-wide soft tokens the text encoder consumes: exact-f32,
  strict loader, ragged batches of images in one forward (padding patches
  stripped on the host, keys masked), several forwards behind one wait. Three
  new kernels in `kernels/embedgemma2_vision.metal`
  (`vision_qkv_norm_rope_f32`: the 2-D axial RoPE with the q/k/v norms and the
  K/V store; `vision_pos_embed_add_f32`; `vision_pool_members_f32`: the
  `k x k` average by patch position times `sqrt(hidden)`); everything else is
  existing kernels (`encoder_attn` at head dim 64, `rms_norm_f32` with a weight
  of ones for the embedder's weightless norm). Against sentence-transformers
  (fp32, CPU) at 2520 patches: every layer within `3e-5` of its largest
  magnitude, pooled and soft tokens within `1e-4`, batched bit-identical to
  alone. See `docs/embedgemma2.md`.
- **EmbeddingGemma 2's audio tower and multimodal embedder**
  (`tessl::embedgemma2_audio`: `AudioConfig`, `EmbedGemma2Audio::{load,
  encode_audio}`, `kernels/embedgemma2_audio.metal`). The 12-layer conformer
  (strided-conv subsampling, clipped linears, chunked local attention with
  Transformer-XL relative positions, light conv) and the 1536 → 512 embedder,
  exact f32, batched over clips of any length with one host wait. It
  implements the intended (bool, sdpa) attention mask: transformers 5.19's
  eager path inverts it (`logical_not()` of a float mask), so
  `tools/embedgemma2_ref/make_mm_reference.py` now runs sdpa by default
  (`EMBEDGEMMA2_MM_ATTN=eager` keeps the old references, which disagree from
  layer 0; text and vision change only by rounding).
  See `docs/embedgemma2.md`.
- **The audio tower's stages are profiled and the top ones fixed**
  (`profile_embedgemma2_audio`, `bench_embedgemma2_audio`; `AudioStage` and
  `EmbedGemma2Audio::{set_stage_mask, clear_stage_mask}` are `#[doc(hidden)]`
  for the profiler). The second subsampling conv is an 8x8 simdgroup-MMA kernel
  (64 → 14 ms on a batch of 8 clips of 30 s, stage alone, two profile runs that
  were not interleaved); the residual adds and
  norms, the GLU with the depthwise conv, norm and SiLU, and the q/k/v input
  clamps are fused (`eg2a_rms_norm` takes two row stages, `eg2a_lconv` replaces
  `eg2a_glu` and `eg2a_dwconv1d_causal`); q, k and v are one GEMM when their
  clamps are equal (they are in all 12 layers of the checkpoint); the attention
  computes only the visible candidates, one butterfly reduce-scatter for the
  logits, and reads the fused projection in place: 28 → 17 dispatches a layer,
  results inside the unchanged bounds, a clip still bit for bit the same alone
  or batched. `AudioConfig::validate` and the attention now refuse more than 16
  past frames and a masked-logit value above `-(cap + 104)`. The end-to-end
  interleaved benchmark was inside the noise of a shared GPU (the control moved
  as much as the change), so no speedup is claimed; the profile shows the
  non-GEMM share of the forward falling from 29 % to 17 % on the batch. See
  `docs/embedgemma2.md` "Speed" and `bench/results/embedgemma2_audio_*`.
- **Per-entry host reads and writes of the Qwen3.5 parameter table**
  (`Qwen35Model::{read_parameter_host, write_parameter_host,
  read_gradient_host, read_adamw_moment_host, write_adamw_moment_host}`).
  Each copies one `parameter_table()` entry through a host mapping, with no
  table-sized shadow tensor, and refuses a bad index, a bad length, a busy
  runtime or a packed-head model before anything moves. A parameter write
  bumps the parameter generation before its first byte, so live `Decode` and
  `Staged` sessions refuse to continue. bf16 writes store exactly the bits the
  GPU copy stores (`tensor::f32_to_bf16_bits_gpu`: NaN → `0x7fc0`, f32
  subnormals flush to zero). `adamw_step_scaled` is documented as Rust-only.
- **Sharded checkpoints** (`safetensors::CheckpointView`, `TensorSource`).
  `Qwen35Model::load`/`load_tower` take `&dyn TensorSource`, so a
  `model.safetensors.index.json` checkpoint loads through the same strict
  loader (existing `&SafeTensors` callers are unchanged). Shard names must be
  bare file names; symlinks are followed, so Hugging Face cache snapshots load.
  Duplicate, missing and unlisted tensors and missing shards are refused at
  open. `tessl_qwen35_load` and the Qwen bench/probe binaries accept a file,
  an index or a directory.
- **`bench_paired` replaces the four paired-round drivers** (`paired_bins.sh`,
  `paired_qwen35_step.sh`, `paired_cross_runtime.py`, `attn_paired.py`):
  `--format metrics|rows:…`, per-lane env (`NAME:ENV=V,…=COMMAND`),
  `--lane-timeout`, `--clean-env`, `--max-spread`, `--evidence` (provenance and
  spread enforcement ported from `benchmark_evidence.py` to Rust, executables
  re-hashed every round) and a per-round raw-output archive under
  `OUT.lanes/`. Benches share `common::{median, fill, parse_count}`; a malformed
  or zero `BENCH_ITERS` exits non-zero naming it. `bench_gemm_epi_tile` is
  `bench_gemm_coop_tile --family epilogue`.
- **Hosted CI runs every device-free test** (`scripts/hosted_tests.sh`,
  `scripts/hosted_tests.txt`: exact per-entry counts, the Metal device hidden
  by `sandbox-exec`), plus a completeness check that fails when a test passes
  without a device but is not selected, the parity harness (a skipped metallib
  cross-check fails under `TESSL_HOSTED=1`) and the kernel IR scan.
- **Kernel IR fast-math check** (`scripts/check_kernel_ir.sh`): every kernel is
  compiled with `build.rs`'s flags and refused on an inf/NaN constant of any
  width, an unordered compare, missing IR, or an `isfinite` that is no longer a
  bit test; exit 77 is a visible SKIPPED. `tests/kernel_fast_math.rs` also bans
  `NAN`, `HUGE_VAL`, `MAXFLOAT`, `isunordered`, inf/NaN builtins and bit casts,
  and self-compares; `isfinite`/`isnan`/`isinf` stay allowed.
- **Seeded mutation tests for the checkpoint parsers** (`src/parser_mutation.rs`;
  dependency-free): JSON, safetensors (including sharded indexes) and npy are
  truncated at every byte, bit-flipped and given boundary integers; every input
  must be `Ok` or a named error with a bounded allocation.

- **bf16 storage's documented default update rule is stochastic rounding**
  (`UpdateRule::bf16_default`). On the 2B loss curve it was the only rule
  inside the 2% per-step bound (worst 0.74%; round-to-nearest worst 4.6%
  on 15–16 of 90 steps). The loss-curve test's fall bound, which failed
  because the f32 run's own held-out mean rose on seed 2, is replaced by
  that 2% bound on the default and the existing 1% held-out comparison.
  The GPU curve has not been re-run. The 4B numbers in `docs/qwen35.md`
  stay the random-weight shape measurement.
- **`tessl_runtime_poisoned` (C ABI 10 → 11).** A poisoned runtime is reported
  by a query function, not a new status code, so existing calls keep the
  statuses they can return. `tessl_torch` asks after any failed call and, if the
  runtime is poisoned, closes the models bound to it and the runtime, then
  raises `TesslPoisoned`; the next call builds a fresh runtime.
  `tests/capi_contract.rs` now parses every `extern "C"` signature and
  `#[repr(C)]` struct in `src/capi.rs` and checks each against the ctypes
  declarations. It found that `tessl_synchronize` had never been declared.
- **`GpuBuffer::try_write_f32`, `try_read_u32`, `try_write_bf16_bits`,
  `try_write_f32_prefix`, `try_write_bytes`.** The panicking accessors now
  delegate to these and are documented as test/bench conveniences.
- **BERT and DistilBERT goldens from transformers.** `tests/bert_model.rs`
  compares the pooled sparse output, the embedding and every layer against
  transformers fp32 on a padded batch, with the bound written before the first
  run (measured error about 3e-6 against a 2e-4 bound). The goldens come from
  `scripts/gen_bert_fixtures.py`, recorded in `tests/fixtures/bert_tiny/manifest.json`.

- **Parser mutation tests** (`tests/parser_mutation.rs`, CPU only, no
  dependency): seeded truncation at every byte, byte and bit flips, extreme
  length/offset/shape values, invalid UTF-8 and deep nesting driven through
  `safetensors` (and its JSON parser) and `npy`, with a counting allocator
  asserting that no declared size is allocated before it is checked against
  the file. `TESSL_MUTATION_CASES` / `TESSL_MUTATION_SEED` set the random pass.
- **`bench/test_parity_harness.py --pure` passes again, and `scripts/ci_local.sh`
  runs it.** The 6c62ab7 import left main's older `gemm_sweep_mlx.py` and
  `paired_cross_runtime.py` beside the harness that tests the imported ones;
  the imported scorers (per-element budget, `adjudicate`, seed-directory
  `parity()`, `missing_coverage`) are restored. `scripts/audit_gemm_tiles.py`
  keeps its stricter main logic, gains `geometry_error`/`geometry_matches`
  and a `__main__` guard so the harness can import it.
- **Measurements: exact f32 at the 2B MLP up/gate shape, and int8.** The
  4.1 vs 5.7 TFLOP/s exact-f32 gap at 2048x6144x2048 reproduces under the
  row-major walk production had on 2026-09-30 and is the B operand (50 MB)
  re-read per tile row; `tile_walk`'s panels already close it with the same
  bits, and the split-K route production takes there is left as a follow-up
  because changing it would change exact-f32 bits
  (`bench/results/f32_exact_mlp_up_gap_m5pro.txt`). int8: exact at sampled
  outputs with and without `b_scale`, 45-57 TOP/s, `b_scale` costing 0-18%;
  the NT variant and a per-row scale are filed there as follow-ups
  (`bench/results/gemm_i8_m5pro.txt`).
- **bf16 MLP inputs, measured and not adopted.** A bf16-output GEMM and a
  SwiGLU reading bf16 gate and up were built, measured and removed: paired on
  the M5 Pro, SwiGLU was 0.65x but the bf16-output GEMMs 1.06x and the
  24-layer forward 1.006-1.009x, with 1.4-1.6x the production forward's KL
  from fp32 on the real 2B (`docs/qwen35.md`, "bf16 activation inputs").
  A fused gate/up/SwiGLU GEMM (same bits, no layout change) was then tried
  and removed too: the 24-layer forward was 1.034-1.037x.
- **Benchmarks:** `bench_qwen35_layers --paired-gate-up` and
  `--model-forward` (the model's prefill as METRIC lines for
  `bench/paired_bins.sh`),
  `bench_qwen35_train --bf16-storage` (since the merge an alias for
  `--storage=bf16`), and op-table rows
  for a bf16 layer input's widen, norm and copy.
  `tools/qwen35_ref/make_reference.py --ids` runs the transformers reference
  on given token ids.

- **Qwen3.5 decode in the model.** `Qwen35Model::prefill(ids, max_new)` runs
  the forward's prefill keeping each layer's GDN state, conv state and K/V,
  and returns the last position's logits with a `Decode` session whose
  `step(id)` runs one token through the same layers on the decode kernels
  (`conv1d_silu` on carried state, `gdn_recurrent` in place,
  `attn_qk_norm_rope_suffix` and `attn_prefix_decode`). tessl owns the
  composition: the layer order is one `Qwen35Model::layer` that the
  prefill, the staged prefill and decode all run. Prefill of `N` tokens
  plus decoded steps matches `forward(N + k)`'s last row within 1e-5
  relative at F32 (`tests/qwen35_model.rs`).
- **Staged prefill: `begin`, `Staged`, `advance_to`, `final_norm_f32`,
  `load_tower`.** `Qwen35Model::begin(ids)` starts a prefill that runs only
  as many layers as asked: `Staged::advance_to(layer)` continues it,
  `final_norm_f32` reads the final norm of the residual stream at the layer
  reached (an early exit; the un-normed residual stays on the device, and
  `forward(ids, true)`'s trace is the way to read it), and `layers_done()`
  says where it is. Advancing in steps is `forward`'s own dispatches in the
  same order, bit for bit at F32. `Qwen35Model::load_tower` loads without
  the packed bf16 head: a bf16 tower has no head (`forward`, `prefill` and
  `Staged::logits` refuse it, and `describe()` says so), an f32 tower uses
  the embedding as its head and runs them, and any tower runs
  `Staged::score_answers`. A `Staged` whose `advance_to` failed part way is
  poisoned the way a `Decode` is (every later entry point refuses before
  encoding), and an F32 `Staged` re-checks the runtime's relaxed precision on
  every entry, not only at `begin`. `begin` and `prefill` errors carry
  their own names. `Qwen35Model::fail_layer_after_for_test` (doc-hidden)
  fails one layer after N, for tests of both poison paths.
- **Chosen logit rows and answer scores.** `Qwen35Model::forward_rows` and
  `Staged::logits` take `LogitRows::{All, Last, Rows}` and run the LM head for
  those positions only, with no `[tokens, vocab]` buffer unless every row is
  asked for. `Staged::score_answers` wires `qwen35::score_answer_rows` into
  the model, on a `load_tower` model too.
- **`nn::DecodeScratch::bytes`**, what `DecodeScratch::new` allocates, for a
  memory check before allocating; and `qwen35::CONV_KERNEL_WIDTHS`, the conv
  widths the forward and backward kernels are compiled for.

- **`Qwen35Model::adamw_step_scaled`.** `adamw_step` with parameter-table
  entry `i` at learning rate `hyper.lr * lr_scale[i]`: torch's AdamW with one
  param group per entry, so the scaled rate forms both the decoupled decay
  factor and the step size. A scale of 0 freezes the entry's bits while its
  moments still update. `lr_scale` is checked (one finite, non-negative value
  per entry; the scaled rate finite) before anything moves.
- **Every recorded benchmark, plotted.** `docs/img/` holds 13 SVG figures
  (light and dark) generated from `bench/results/`, and `docs/benchmarking.md`
  gains a section for the tessl-against-PyTorch GEMM comparison, the Qwen3.5
  step-waste and dispatch host-overhead runs, and one for every other result
  family (tile grids, SIMD-group rewrites, attention speed and tuning,
  loader and training memory, accuracy). The plots exposed that
  `gemm_sweep_m5pro_f32_bf16.json` predates the cooperative-destination bf16
  kernel, and that the `attn_speed_*` files store `tessl_ms / other_ms`;
  both are recorded in the doc.

- **Accumulate GEMM operands.** `GemmOperands::{nn_acc, tn_acc, nt_acc}`
  compute `C += op(A) op(B)` in the GEMM itself on both lanes: the TN/NT
  accumulate kernels, a new exact-f32 `matmul2d_tensorops_nn_accum_f32`
  (and the NN split-K without its zero), and for bf16 operands the
  cooperative epilogue with `beta = 1`. `cross_entropy_rows_accumulating`
  adds the weight gradient into `dW` instead of overwriting it.
- **Allocation and upload primitives.** `GpuRuntime::alloc_tensor_unzeroed`
  for tensors a kernel writes in full, `set_poison_unzeroed` (a test aid
  that fills them with NaN), `alloc_buffer_from_u32` (a fresh buffer filled
  without a GPU wait) and `upload_u32` (a write ordered with the queued
  work, through a bitwise `copy_u32` kernel).
- **`Qwen35Model::adamw_step_unwaited`.** The AdamW step encoded without its
  closing wait; `AdamW::step_count` reports the last count its own waits
  confirmed once the runtime is poisoned.
- **Counters and bench modes.** `infer_trace` counts host waits
  (`sync_waits`) and host-zeroed allocations and bytes, and
  `Snapshot::since` differences two snapshots. `bench_qwen35_train` gains
  `--async`, `--step-only` and `--clip`, and prints each step's waits and
  device peak; `bench/paired_qwen35_step.sh` interleaves configurations and
  a before/after pair of binaries.
- **`qwen35_gdn_recurrent_bv16`, the GDN decode recurrence in 16-column
  slices.** `qwen35::gdn_recurrent_with_slice(.., GdnScanSlice::Cols16)` runs
  it: 64-thread groups, each simdgroup holding two 32-row key blocks of the
  same 16 value columns, so every reduction runs in the 32-column kernel's
  order and the two agree bit for bit (held by every recurrent test in
  `tests/qwen35_kernels.rs` and by the CPU emulator). It launches twice the
  threadgroups of the 32-column kernel. `gdn_recurrent` keeps
  `GDN_RECURRENT_SLICE = Cols32`: a paired batch-1 sweep
  (`bench_qwen35_layers --paired-gdn-recurrent`) found no difference outside
  its noise, so the 16-column kernel is not promoted.
- **`bench_paired`, a Rust round-robin runner for benchmark lanes.** It
  replaces `bench/paired_embedgemma2.py`: lanes are `NAME=COMMAND`s printing
  `bench_embedgemma2`'s JSON, alternated round by round, reported as the
  median of per-round min-of-N with per-round ratios against the first lane.
  Two builds of `bench_embedgemma2` make a before/after A/B.
  `bench_embedgemma2` gains ragged workloads (`BxLO-HI` terms joined by `+`,
  which `bench/embedgemma2_torch.py` reads too), the device's peak allocation
  per workload and the process's peak physical footprint;
  `probe_load_memory --embedgemma2` measures an EmbeddingGemma 2 load.
- **Qwen3.5 training on bf16 storage.** A `Precision::Bf16` model (loaded
  with `load_tower`) trains on `GemmOperands::Bf16` with its matrices kept in
  bf16; arithmetic stays f32. `AdamWConfig` picks an `UpdateRule` for
  bf16-stored weights (`F32Master`, `Bf16Kahan`, `Bf16Stochastic { seed }`)
  and a `MomentStorage` (`F32`, `Bf16`, or `Block8`: 8-bit codes with one f32
  scale per 256 elements). Gradient banks take their weight's dtype, and
  `read_adamw_aux` / `write_adamw_aux` checkpoint the master or compensation
  so a run resumes bit for bit. `Qwen35Model::random_tower` and
  `probe_storage_memory` measure what each variant holds at a config's
  shapes; `probe_storage_memory --step=T` runs a training step and an AdamW
  step on the config's own model and reports the measured peak.
- **bf16 storage through the C ABI and `tessl_torch` (ABI 10).**
  `tessl_torch.Qwen35(..., precision="bf16")` loads a bf16-stored model, and
  `adamw_init(update=..., moments=..., seed=...)` picks the update rule
  (`"f32-master"`, `"bf16-kahan"`, `"bf16-stochastic"`) and the moments
  (`"f32"`, `"bf16"`, `"block8"`). `describe()` names the stored precision and
  the optimizer. `adamw_state()` records the configuration (`"config"`) and
  the f32 masters or Kahan compensations (`"aux"`), and `load_adamw_state`
  refuses a checkpoint made under another configuration, so a bf16 run
  resumes bit for bit under every rule. In the C ABI, `tessl_qwen35_describe`
  and the copy directions `TESSL_READ_ADAMW_AUX` / `TESSL_WRITE_ADAMW_AUX`
  are new.
- **Qwen3.5 training on GDN layers with grouped heads.** A GDN with more
  value heads than key heads (Qwen3.5-4B: 32 over 16) now trains: each key
  head's q and k are repeated across its value heads before `gdn_train`, as
  transformers' `repeat_interleave` does, and their gradients sum back over
  the group. `train_step` no longer refuses such a config. Checked against
  transformers' autograd on a committed tiny fixture with two key heads over
  four value heads (`tests/fixtures/qwen35_train_grouped`,
  `make_train_fixture.py tiny --grouped`).
- **`tessl::bert`: BERT / DistilBERT learned sparse document encoders.**
  `BertForMaskedLM` and `DistilBertForMaskedLM` checkpoints produce
  `max_t log1p(relu(logits))` term weights, in exact f32. New kernels are in
  `kernels/bert.metal`, and `embedgemma2::encoder_attn` gains head dims 32
  and 64.

- **`tessl::safetensors` reads integer and fp8 tensors.** `read_raw` returns
  any tensor's header entry and its stored bytes. `read_u8`, `read_i8` and
  `read_u32` return typed int8 and MLX-packed Q4 weights. `Dtype` gains
  `F8E4M3`, `F8E4M3Fnuz`, `F8E5M2`, `F8E5M2Fnuz`, `F8E8M0` and `C64`
  (`F8_E4M3`, `F8_E4M3FNUZ`, `F8_E5M2`, `F8_E5M2FNUZ`, `F8_E8M0`, `C64` on
  disk), so files holding them now open. The sub-byte `F4` and `F6_*` types
  are still refused. Adding variants to the exhaustive `Dtype` enum breaks an
  external `match` on it.


- **EmbeddingGemma 2 text encoder (`tessl::embedgemma2`).** It loads
  `google/embeddinggemma-2`'s text path from its own `model.safetensors` and
  embeds ragged batches: the `encoder_attn` bidirectional sliding-window
  kernel, `segment_mean_rows`, `l2_normalize_rows`, and length-sorted forward
  packing. It was first run on the GPU on an M5 Pro against
  sentence-transformers 6.1 (fp32 eager): every embedding up to 6147 tokens
  is within `2.5e-7` max abs, every Matryoshka prefix within `3.9e-7`, and
  the per-layer residual stream of a 1658-token text within `4.9e-5` of its
  largest magnitude. `docs/embedgemma2.md` has the bounds and observed
  errors.
  - `tests/embedgemma2_tiny.rs` runs the whole forward on a random tiny
    checkpoint, built in memory, against an f64 host forward, in the plain
    GPU suite.
  - The reference and fixture generators record `provenance.json`: library
    versions and the generator commit.
- **`SafeTensors::from_bytes`** validates and reads a `.safetensors`
  serialization held in memory, through the same header checks as
  `SafeTensors::open`. Every malformed-file test now runs through both.
- **`nn::rope_inv_freq`**: transformers' RoPE frequency table (`1.0 /
  (theta ** ((2p).float() / dim))` in f32), the one every RoPE kernel now
  reads.
- **Device memory accounting.** `GpuRuntime::peak_allocated_bytes` and
  `reset_peak_allocated_bytes` (the high-water mark of `currentAllocatedSize`,
  sampled at every buffer the pool creates), `GpuRuntime::allocated_bytes_for`
  (what a pool allocation of a given size and kind costs),
  `allocated_bytes_for` on `GdnTrainWorkspace`, `CeWorkspace`,
  `AttnTrainWorkspace` and `EmbedBwdWorkspace`, `Qwen35Model::train_step_bytes`,
  `GpuRuntime::set_recommended_working_set_for_test`, and the
  `probe_train_memory` binary that measures the 2B's tables and steps.
- **GPU faults are read from commit feedback.** `MTL4CommandBuffer` has no
  `status` or `error`, so a fault was invisible after a wait. Each commit now
  registers an `MTL4CommitFeedback` handler through `MTL4CommitOptions`
  (features `MTL4CommitFeedback`, `block2`; `block2` is the block crate
  objc2-metal already uses). After a GPU wait, a reported error latches the
  runtime the same way a shared-event timeout does: later encodes and
  allocations refuse reuse, and `GpuRuntime::is_poisoned` lets
  callers read the latch instead of matching the poison string. The wait on the
  callback is bounded and returns on the callback's notify rather than a poll
  tick; a callback that has not run stays pending. Also adds `GpuRuntime::current_allocated_bytes`, a live
  `currentAllocatedSize` reading alongside the startup `memory_info` snapshot.
- **Causal flash attention at head dimension 64 (`tessl::nn::flash_attn_rows`)**:
  new kernel instantiation `flash_attn_rows_h64_r8_g8` at R=8, SGT=8,
  satisfying `D % (4*R) == 0` for float4 lane coverage. Tested in
  `tests/flash_attn_rows_h64.rs`.
- **Packed QK RMSNorm and half-split RoPE (`tessl::qwen35::attn_qk_norm_rope_packed`)**:
  adds support for explicit query-head stride (`head_dim` for packed
  `[T, H, D]`, or `2 * head_dim` for Qwen's query-gate layout) and
  configurable RMSNorm `weight_bias` (`0.0` for learnable scale `* w`,
  `1.0` for Qwen's `*(1 + w)`). Tested in `tests/qk_norm_half_rope.rs`.
- **Non-finite protection in argmax and softcap sampling**:
  `argmax_f32`, `softcap_sample`, and `softcap_argmax_one_pass` guard against
  NaNs and non-finite inputs, writing `0xFFFFFFFF` / returning error instead
  of sampling token 0.
- **Stand-alone host validators for MLX Q4 GEMV**:
  `tessl::nn::validate_gemv_q4_mlx_inputs`, `validate_gemv_q4_mlx_blocked`,
  and `validate_gemv_q4_mlx_simd` expose host-side buffer shape, bank bounds,
  and disjointness validation without encoding.
- **Fallible u32 buffer writing (`GpuBuffer::try_write_u32`)**:
  reports buffer length mismatches and poisoned runtime states via `Result`
  instead of unconditionally panicking.
- **Device AdamW on any f32 tensor (`tessl::qwen35_adamw::adamw_step`)**:
  one step of `qwen35_adamw_f32` on a parameter, its gradient and both
  moments. `Qwen35Model::adamw_step` is that function in a loop over the
  parameter table, including packed windows, so there is no second kernel.
  A step of 0 is refused (the value `u64::MAX + 1` wraps to, which zeroes
  the bias correction). The model method still refuses to increment a count
  that is already `u64::MAX`. Callers synchronize before a host read.
  `tests/adamw_step.rs` checks a non-contracted f32 reference within the
  existing `2e-6` absolute bound, mismatched shapes and dtypes, a zero-length
  view, a byte-offset view, and a gradient that belongs to another runtime.
- **The metallib is embedded.** `GpuRuntime::new` loads the shader library
  with `newLibraryWithData` (`MTLDevice::newLibraryWithData:error:`,
  objc2-metal 0.3.2's `newLibraryWithData_error`, feature `dispatch2`) from
  bytes included at compile time. Moving or deleting the build-directory
  `.metallib` no longer stops a binary from opening a runtime.
  `add_metallib_bytes` is the same load for an adopter overlay.
  `metallib_path` and `DEP_TESSL_METALLIB` still name the on-disk artifact
  for tooling.
- **Metal math mode is explicit.** `build.rs` passes `-fmetal-math-mode=fast`,
  which is the compiler default, on every shader. `-ffp-contract=off` is
  applied to `kernels/qwen35_adamw.metal` only: with contraction on, a step
  at magnitude ~1e4 missed the `2e-6` bound by one ulp (`6.104e-5`); with
  the flag the same step is bit-identical to a non-contracted f32 reference.
  A million-element batched step was 0.2336 ms with the flag and 0.2456 ms
  without (one release run each; the flag was not slower). RMSNorm's gap
  against a serial f32 sum is `3.576e-7` (inside `1e-5`, and it is the
  reduction tree) and cross-entropy's exact-f32 `dh` relative error is
  `1.58e-6` (inside `1e-5`, `precise::exp`/`log` already). Those two were
  not given the flag. Numbers: `bench/results/fp_contract.txt`.
- **LM-head cross-entropy with gradients (`tessl::cross_entropy`)**:
  `cross_entropy_rows` over the supervised `(row, target)` pairs only, in
  vocabulary chunks, never forming `[rows, vocab]` logits. Loss (mean or sum),
  `dh` for the supplied rows and the full `dW`, exact f32, scratch bounded by
  a `CeWorkspace` sized by rows and chunk. Kernels `ce_gather_rows_{f32,bf16}`,
  `ce_lse_update`, `ce_softmax_grad`. `gemm::cast_bf16_to_f32_into` widens
  into a caller's buffer.
- **GDN training forward and backward (`tessl::gdn_train`)**: the gated
  delta rule at transformers' `torch_chunk_gated_delta_rule` seam (`g`,
  `beta` given, in-kernel q/k l2norm), saving only a state per 64 tokens
  (32 MiB/layer at the 2B's shapes, T = 2048, vs 435 MiB in torch), with a
  deterministic chunk-reverse backward. Kernels `gdn_train_fwd`,
  `gdn_train_bwd`, `gdn_train_bwd_finish`.
  From torch: `tessl_torch.chunk_gated_delta_rule` (transformers'
  signature) and `patch_transformers_qwen3_5()`; C ABI 3 adds
  `tessl_gdn_train_forward`/`_backward` and tensors up to rank 6.
- **Backward of the Qwen3.5 row-local ops (`tessl::qwen35_bwd`)**:
  `rms_norm_bwd` (with an accumulate flag for the residual stream),
  `gated_rms_norm_bwd` (the GDN output norm), `swiglu_bwd` and
  `attn_gate_bwd`, in the forward's own row windows: `dgate`/`dup` and the
  output gate's column land in the fused projections' gradient windows.
  Weight gradients are per-block partials summed in order, no atomics.
  Kernels `qwen35_rms_norm_bwd_f32`, `qwen35_col_sum_blocks_f32`,
  `qwen35_gated_rms_norm_bwd_f32`, `qwen35_swiglu_bwd_f32`,
  `qwen35_attn_gate_bwd_f32`.
- **Causal conv + SiLU backward (`qwen35_bwd::conv1d_silu_bwd`)**: the GDN
  conv as training runs it (zero state, as transformers' no-cache path),
  `dx` into a window of the fused projection's gradient and a
  deterministic `dw [C, KW]`; the pre-activation is recomputed from `x`.
  Kernels `qwen35_conv1d_silu_bwd_dx_f32`, `qwen35_conv1d_silu_bwd_dw_f32`.
- **Attention Q/K norm + partial RoPE backward
  (`qwen35_bwd::attn_qk_norm_rope_bwd`)**: from the q/k/v gradients into
  the fused projection's gradient (q, k, v columns; the gate's are
  `attn_gate_bwd`'s), with deterministic `(1 + w)` gradients for both norms.
  The RoPE angle is now one helper, `qwen35_rope_angle` in
  `kernels/qwen35_act.h`, shared by the forward and the backward. Kernel
  `qwen35_attn_qk_norm_rope_bwd_f32`.
- **Embedding backward (`qwen35_bwd::embed_rows_bwd`)**: `dW[id] += dh[r]`,
  added onto the tied LM head's gradient after `cross_entropy_rows`. The
  host checks the ids and groups the rows by id; each id's rows are summed
  in row order without atomics. `EmbedBwdWorkspace` holds the grouping.
  Kernel `qwen35_embed_rows_bwd_f32`.
- **A Qwen3.5 training step (`Qwen35Model::train_step`, `tessl::qwen35_train`)**:
  transformers' causal-LM loss for one sequence and every parameter's
  gradient, in f32, in each weight's own layout (packed fused projections,
  `[vocab, hidden]` for the tied embedding and head). The forward keeps
  only each layer's input and each layer is rebuilt just before its
  backward (0.38 GiB of activations at T = 2048 on the 2B, against 8.54 GiB
  saved, for 10% more time); it uses `gdn_train`, `attn_train` and
  `gdn_gates`, and the backward chains the CE, row-local, conv, gate, Q/K,
  attention and embedding backwards with exact-f32 GEMMs. Checked against
  transformers' autograd on a committed tiny model
  (`tests/fixtures/qwen35_train/`, from
  `tools/qwen35_ref/make_train_fixture.py tiny`), and on the real 2B by an
  ignored test (`make_train_fixture.py 2b`): loss within 4.6e-5, gradients
  within 3.9e-3 of each parameter's largest, a difference finite differences
  of tessl's own loss attribute to the two f32 forwards, not the backward
  (`real_2b_gradients_are_those_of_tessls_forward`);
  `tools/qwen35_ref/train_noise_floor.py` measures transformers' own
  run-to-run gradient disagreement for comparison.
- **A Qwen3.5 model's parameters under transformers' names
  (`tessl::qwen35_params`)**: `parameter_table`, `read_parameters`,
  `read_gradients` and `write_parameters` copy between the model and caller
  f32 tensors on the GPU, with transformers' shapes and values (norms as
  `w`; linear weights as `[in, out]` windows of the
  packed projections), honouring a caller's byte offset and checking every
  tensor before writing any. `gemm::transpose_f32_into` is the checked GPU
  transpose (the TN/NT fallbacks now share it).
- **Training attention (`tessl::attn_train`)**: `attn_train_forward` is
  `attn_prefill`'s tiled kernel at its default geometry with each row's
  log-sum-exp written out (new entry point
  `qwen35_attn_tiled_lse_h256_q32_k32_sg4`; the four existing entry points
  are unchanged and bind no new slot), and bit-identical O.
  `attn_train_backward` is FlashAttention-2's backward on the matrix units
  (`kernels/qwen35_attn_bwd.metal`): dQ per query block, dK and dV per key
  block walking the KV head's query heads in order, each gradient written
  once, no atomics. Kernels `qwen35_attn_bwd_dvec_f32`,
  `qwen35_attn_bwd_{dq,dk,dv}_h256_q32_k32_sg4`.
- **GDN gates for training (`qwen35::gdn_gates`, `qwen35_bwd::gdn_gates_bwd`)**:
  `g = -exp(A_log) * softplus(a + dt_bias)` and `beta = sigmoid(b)` written
  dense `[rows, heads]` for `gdn_train`, and their backward into the fused
  projection's a/b gradient columns with deterministic `dA_log`, `ddt_bias`.
  torch's softplus and the log decay move to `kernels/qwen35_act.h`
  (`qwen35_softplus`, `qwen35_log_decay`), shared with the inference loads.
  `qwen35_bwd::copy_cols` moves a column window between layouts. Kernels
  `qwen35_gdn_gates_f32`, `qwen35_gdn_gates_bwd_f32`, `qwen35_copy_cols_f32`.
- **qwen35_gdn.metal compiles under -Wall -Werror again**: the 16-column
  scan's threadgroup-memory constant gets the 32 KB static_assert its
  siblings have (the current compiler rejected it as unused).
- **`bench_qwen35_train`**: every training op at Qwen3.5-2B's shapes behind a
  NaN-poison gate, and `--step=N` for a whole `train_step` on the checkpoint
  (7.2 s at T = 2048 on an M5 Pro; per-op numbers in docs/qwen35.md).
- **Randomized-shape stress for the backward kernels**
  (`randomized_shapes_stress` in `tests/qwen35_bwd.rs`, `tests/attn_train.rs`):
  shapes drawn within each kernel's contract, each draw checked like the
  targeted tests; `TESSL_FUZZ_ITERS` / `TESSL_FUZZ_SEED` scale it (500 draws
  per sweep pass).
- **msl_emu host contract**: macro instantiations are expanded by a small
  preprocessor (nested macros, object-like parameter macros), the
  TensorOps backward's signatures and `src/attn_train.rs`'s binds are
  checked, and a host bind of a kernel no inspected source declares fails.
- **torch binding (`python/tessl_torch`) over a C ABI (`tessl::capi`)**:
  `tessl_torch.cross_entropy(hidden, weight, targets, mask)`, a
  `torch.autograd.Function` over MPS tensors, loaded with `ctypes` (no C++
  extension, no new dependency). The crate now also builds a `cdylib`
  (`libtessl.dylib`); the ABI is versioned, catches panics at the boundary,
  and refuses calls from a thread other than the handle's.
- **The Qwen3.5 text forward (`tessl::qwen35_model`)**, loaded from the
  Hugging Face checkpoint, checked end to end against transformers
  (`tests/qwen35_model.rs`, opt-in): F32 within 2.3e-6 per layer, Bf16 18x
  closer to fp32 than transformers' own bf16. See `docs/qwen35.md`.
- **`Qwen35Config::from_config_json` / `from_config_file`**: any Qwen3.5 text
  `config.json` (nested `text_config` or text-only), with every feature the
  forward lacks refused by name. The published 2B config parses to exactly
  `qwen35_2b()`; other sizes are parsed, not yet run against transformers.
  The JSON parser is the safetensors one, moved to a crate-private
  `json` module with per-format syntax limits.
- **`tessl::safetensors`**, a strict `.safetensors` reader (exact offsets,
  bounded header, JSON subset, reads through the handle it opened).
- **`qwen35::residual_add`** (`qwen35_residual_add_f32`), the exact-f32
  residual add for strided windows.
- **`tessl_torch.Qwen35.grads(into=...)`** writes a step's gradients into
  tensors an earlier `grads()` returned (the optimizer's `.grad`s), checking
  them all first, instead of allocating another copy of every gradient (8 GB
  on the 2B) while the previous one is still alive. The documented training
  recipe uses it and shows `operands="bf16"`; the default stays exact f32.

### Changed (breaking)

- **ICB stub scaffold removed**: `IcbReplayStub`, `IcbStubPhase`,
  `ArgTableSlotPlan`, `ArgTableSlot`, `IcbCommandTypeHint`,
  `PingPongCbReplay::{icb_stub, icb_stub_mut}`, `from_commands_ex`,
  `mini_copy_chain_ex`, ICB barrier elision and the parked freeze / range-batch
  / coarse ICB flags with their getters. `CbReplayError::NotWired`,
  `survey_cb_replay_api_gaps`, `decode_icb_wired()` and the replay counters are
  unchanged. `PrefillAttnKernel` / `prefill_attn_kernel` are gone:
  `attn_prefill_by_length` calls the tiled kernel directly (same signature).
- **Environment names removed**: `METAL_NATIVE_HAZARD_BARRIERS` no longer
  decides anything (set `TESSL_HAZARD_BARRIERS` or
  `METAL_RUNTIME_HAZARD_BARRIERS`), and `TESSL_COARSE_BARRIERS`,
  `GEMMA_METAL_COARSE_BARRIERS` and `METAL_RUNTIME_COARSE_BARRIERS` went with
  the parked coarse-barrier flag, as did the freeze and range-batch flags. An
  `rg -uu` over the consuming repositories found no setter. gemma-metal follows
  in the same change.
- **npy header framing is enforced**: space padding to a 16-byte data start
  (NumPy writes 64; pre-1.9 NumPy wrote 16), a trailing newline and a closed
  dict, ASCII before v3. A `SafeTensors` read after the file was rewritten in
  place since `open` is refused ("file changed since it was opened"); a
  metadata-only change refuses too.
- **`qwen35::attn_qk_norm_rope_columns` takes an output `DType`** (after
  `targets`): `DType::F32` is the previous behaviour, `DType::BF16` (scalar
  `RopePos` only) writes the rounded values for the bf16 attention kernels.

- **`GpuRuntime::shared_event()` returns a handoff event, not the lifetime
  event.** The allocator reset, const-arena rewind, pool recycling and the
  drop wait now read a crate-private event only tessl's queue signals;
  `shared_event()` is a second event the queue signals after it, at the same
  value, so a foreign `setSignaledValue` (a safe call in objc2-metal) can no
  longer make tessl free memory the GPU is still using. Waiting on
  `(shared_event(), v)` for `v <= last_signaled_value()` still means tessl's
  work through commit `v` is done. Signaling the event does not make tessl
  wait: hand foreign work to tessl by finishing it on the host first
  (`Tensor::from_mtl_buffer`'s handoff doc says so). New:
  `last_completed_value()` and `handoff_foreign_signal()`.
- **A runtime wait that runs out of budget no longer poisons the runtime, and
  the budget is configurable.** One setting, `GpuRuntime::event_wait_ms()` /
  `set_event_wait_ms()`, covers every wait on the GPU timeline. Default
  `DEFAULT_EVENT_WAIT_MS` (30 000 ms); accepted range `EVENT_WAIT_MS_RANGE`
  (1 to 3 600 000 ms); the `TESSL_EVENT_WAIT_MS` environment variable sets it
  at runtime creation (an out-of-range value is ignored with a warning). A
  timeout returns a named error saying how long it waited and for which value,
  `GpuRuntime::is_event_wait_timeout` recognises it, and retrying
  `synchronize` keeps waiting. A GPU fault, or a lifetime event ahead of what
  was submitted, still poisons.
- **`gemm_{tn,nt}_accum_train` always run the accumulate kernels on
  TensorOps;
  `TESSL_GEMM_ACCUM` and `TESSL_GEMM_ACCUM_DX` are retired.** Training has
  reached the accumulate kernels through `GemmOperands::{tn_acc, nt_acc}`
  with no flag since that API landed, weight banks included; the flags gated
  only these two legacy entries, which otherwise computed the product into a
  temporary and added it. They are now `GemmOperands`' path on the precision
  the runtime picks (bit for bit, pinned by
  `accumulate_train_entries_run_the_accumulate_kernels`), so the rounding of
  `C += ...` changes for callers that left the flags unset, as ojas's
  `arch02` trainer does (`model_bwd.rs`). The simdgroup backend, which has
  no accumulate kernel and is `select_backend`'s pick without TensorOps,
  still adds a temporary; `transpose_edges_precision_and_accumulation`
  checks its `C +=` against a CPU sum. The
  variables, including their `METAL_RUNTIME_*` / `METAL_NATIVE_*` spellings,
  are no longer read.
- **`ab_flags::gemm_accum`, `gemm_accum_dx` and `gemm_interior_offsets`
  are removed** with the variables they read (`TESSL_GEMM_ACCUM`,
  `TESSL_GEMM_ACCUM_DX`, `TESSL_GEMM_INTERIOR`, and their `METAL_RUNTIME_*` /
  `METAL_NATIVE_*` spellings), which nothing reads any more. No caller in
  ojas, gemma-metal or tessl-arch02 used them.

- **Binder and runtime soundness narrowing (ft-c125d85).** Safe code could
  make a kernel touch memory it does not own through several doors; none had an
  in-tree or sibling-repo user (`rg -uu` over ojas, qd-metal, MLSystemsLab,
  Lappi-decision, BINN found none).
  - `Binder::bind_addr` is now `unsafe fn` (the address must stay live and
    resident until the command buffer completes; a `compile_fail` doctest pins
    it). `Binder::materialize_bytes` is `pub(crate)`: its address is only valid
    until the next const-arena rewind. `bind_bytes` stays safe: it copies into
    the runtime-owned arena after a bounds check and returns an address inside it.
  - **Decision: `with_binder` stays safe.** `Binder` is the documented
    raw-kernel escape hatch (struct docs and the `lib.rs` "Unsafe code"
    section): scalar counts bound through it are not checked against buffer
    extents. Making it `unsafe` would break about 50 external call sites.
  - `GpuRuntime::{register_residency, register_allocation,
    unregister_residency}` are `pub(crate)`; unregistering a live buffer
    faults in-flight work.
  - `GpuTensor`'s `tensor`, `dims` and `dtype` fields (`quant-prep`) are private;
    use `tensor()`, `dims()` and `data_type()`.
- **`DecodeIcb::from_commands_ex` validates `Buf` binds.** `gpu_addr` is
  recomputed from `buf` + `byte_offset` (a forged value is discarded), and a
  `byte_offset` at or past the buffer, a buffer from another runtime, or a
  `tg_mem` slot past the argument table is an `Err`.
- **`IcbCopySmoke` refuses buffers it cannot copy between.** `encode_copy`
  returns `Err` for a buffer shorter than `n * 4` bytes or owned by another
  runtime, before touching the ICB; `new` refuses `n > u32::MAX`. Re-encoding
  while an earlier `execute` is in flight is documented as the caller's to
  synchronize.


- **`EmbedGemma2Model::encode` takes `truncate_dim: Option<u32>`** between
  the batch and `trace`. `Some(d)` returns `[batch, d]` Matryoshka prefixes,
  normalized on the GPU over the prefix, as sentence-transformers'
  `encode(truncate_dim=d, normalize_embeddings=True)` does. `None` keeps the
  full `embedding_dim`.
- **`embedgemma2::truncate_renormalize` is removed.** It was a host copy of
  what `l2_normalize_rows` does on the GPU; pass `truncate_dim` to `encode`
  instead.
- **`EmbedGemma2Model::load` refuses tensors under the prefix that the
  forward does not read** (a bias, an extra norm), naming them, where it
  ignored them before. Tensors outside the prefix (the vision and audio
  towers) are still not read.
- `set_binder_encode_nop` is no longer public: while armed, every encode on
  its thread returns `Ok(())` having done nothing. `BinderEncodeNopGuard`
  still arms it for a scope, and the new `clear_binder_encode_nop` can only
  disarm it.
- **C ABI 9** (was 3; the binding and library refuse each other across
  versions, so rebuild `libtessl.dylib` with the binding). It adds a
  Qwen3.5 model handle: `tessl_qwen35_load`, `_train_step`,
  `_param_count`, `_param_info` (`TesslParamInfo`), `_copy` (read
  parameters, read gradients, write parameters) and `_free`, with the
  runtime's thread affinity. `tessl_torch.Qwen35` wraps it:
  `parameters()`, `train_step(ids)`, `grads()`, `load_parameters()`, under
  transformers' names and values, for a torch optimizer. Checked against
  transformers' own autograd before and after an AdamW step written back
  (`python/tests/test_qwen35.py`). ABI 9 also runs AdamW inside tessl
  (`tessl_qwen35_adamw_init`, `_step`, `_step_count`, `_set_step_count`,
  `_free`, and `_copy` directions 3-6 reading and writing both moments, over
  `qwen35_adamw`), and `TesslParamInfo` gains `decay_excluded`, Trainer's
  weight-decay exclusion for that entry; `tessl_torch.Qwen35` wraps them
  as `adamw_init()`, `adamw_step(lr, betas, eps, weight_decay)` (a float
  for every non-excluded parameter, or a dict by name), `adamw_free()` and
  `adamw_step_count`, with `adamw_state()` and `load_adamw_state()` for a
  checkpoint (step, `exp_avg`, `exp_avg_sq` by name), so a loop needs no
  torch copy of the parameters or gradients. Checked against
  `torch.optim.AdamW` itself, over three steps within 1e-6, and the ABI
  against the Rust call bit for bit; a run restored from a checkpoint takes
  its next step to the same bits as the run that never stopped
  (`tests/qwen35_adamw.rs`, `python/tests/test_qwen35.py`). Gradient
  clipping: `tessl_qwen35_grad_sq_norm` (`Qwen35.grad_sq_norm()`,
  `Qwen35Model::grad_sq_norm`) gives the global norm's square, and
  `_adamw_step` takes a `grad_scale` (`AdamWHyper::grad_scale`) that
  multiplies every gradient before the update, leaving the stored gradients
  as they are; `tessl_torch.clip_coef(norm, max_norm)` is
  `clip_grad_norm_`'s coefficient. The caller forms it, so gradients
  outside tessl (a head of its own) join the norm. Checked against
  `clip_grad_norm_` then `torch.optim.AdamW` within 1e-6. The handle's
  gradients are a bank, allocated by the first step and reused, and a step
  splits in two: `tessl_qwen35_train_forward` (`TESSL_SUPERVISE_CAUSAL`, or
  `TESSL_SUPERVISE_ROWS` with positions, targets and a scale),
  `tessl_qwen35_hidden` (final-norm rows for a loss outside tessl),
  `tessl_qwen35_train_backward` (that loss's gradient at those rows, and
  `accumulate` to add into the bank) and `tessl_qwen35_train_discard`.
  While a step is pending, another step, the gradients and AdamW are
  refused; accumulating onto a bank a refused or failed step marked is
  refused. `tessl_torch.Qwen35` wraps them as `train_forward(ids,
  positions=, targets=, scale=)`, `hidden(positions)`,
  `train_backward(dh=, positions=, accumulate=)` (summing the rows of a
  repeated position) and `train_discard()`. Checked against transformers on
  a right-padded two-row batch with a letter row through the tied head and
  a span row through a pointer head in torch: run row by row and
  accumulated, the same losses, summed gradients and global norm
  (`python/tests/test_qwen35.py`); the ABI
  against the Rust calls bit for bit (`tests/capi.rs`).
- **msl_emu runs two training kernels**: `qwen35_scatter_add_rows_f32`
  (against torch's `index_add_`, bit for bit) and `qwen35_adamw_f32`
  (against torch's AdamW formula in f64 on a packed window, with
  `grad_scale`, and nothing outside the window moving), so the
  kernel-emulator CI job checks them without a Mac. The shim gains
  `precise::sqrt` and `precise::rsqrt`.
- **msl_emu shares kernel-scope threadgroup arrays across a group**, as a
  GPU does: `build.sh` rewrites each `threadgroup T name[N];` into a static
  registered with `metal::emu::tg_static`, which `launch` poisons with NaN
  before every group, and refuses to build if one is left unrewritten.
  Before, `threadgroup` was defined away and each thread got a private
  array. `qwen35_sq_sum_rows_f32` (the rows of `grad_sq_norm`) now runs
  against f64 on a packed window at four widths, writing nothing outside
  its rows; without the rewrite three of them fail. `barrier_probe` holds
  the sharing and the poison to a probe of its own, and CI's
  ThreadSanitizer step runs `sq_sum_rows`, which reports a race with the
  kernel's barrier removed.
- **`bench_qwen35_train --batch=ROWS,LEN[,SPAN_ROWS]`** times one optimizer
  step's gradients for a batch run row by row into one bank (letter rows
  supervise one position, span rows go through `hidden` and an outside
  gradient), reporting seconds per optimizer step and per row; two lengths
  at one row count separate the per-row fixed cost from the per-token cost.
  An extra run after the timed ones prints each row's wall time, time
  waiting on the GPU, allocations and commits. On an M5 Pro that showed the
  same 1625-token letter row taking 3.0 to 5.9 s between moments in one
  process, with accumulated rows no slower than the first, so a batch-versus-
  single comparison from one run each is within the noise.
- **`Qwen35Model::train_step_into(ids, operands, sup, bank, accumulate)`**
  writes a step's gradients into a bank from `Qwen35Grads::zeros_like`
  (over it, or added to it), so several sequences' gradients sum in place.
  Each layer's gradients go into the bank as soon as its backward is
  encoded and released (freed buffers return to the pool at the next GPU
  wait, so a few layers' are held at once on Qwen3.5, not every gradient).
  A copy is `train_step`'s bits and an add is one f32 rounding
  (`tests/qwen35_train.rs`). `Qwen35Grads::zeros_like` also allocates
  AdamW's moments (it moved from `qwen35_adamw`). What the step scores is a
  `Supervise`: `Causal` (transformers' causal-LM mean, `train_step`'s) or
  `Rows { positions, targets, scale }`, chosen hidden positions against
  given tokens, returning the sum of their cross-entropies with the
  gradients of `scale` times it (so a batch mean over rows spread across
  steps is `scale = 1 / N` in each); an empty selection is allowed (zero
  loss). The rows' gradient goes back through
  `qwen35_bwd::scatter_add_rows` (kernel `qwen35_scatter_add_rows_f32`).
  Checked against the causal step (every row at `1 / (T - 1)`), additivity
  over disjoint sets, and causality (a position on the sequence cut just
  after it). A step also splits in two for a loss outside tessl:
  `train_forward(ids, operands, sup)` returns a `PendingStep` (its loss, and
  `hidden(positions, out)`: rows of the final norm's output, transformers'
  `last_hidden_state`), and `train_backward_into(pending, dh, bank,
  accumulate)` adds that loss's gradient at those rows to the step's own
  before the backward. Checked with the tied head's cross-entropy computed
  on the host from `hidden`'s rows: the same loss and, fed back as `dh`, the
  same gradients as `Supervise::Rows` inside tessl. A step that scores
  nothing in tessl (an outside loss only) adds its embedding gradient
  straight into the bank: no `[vocab, hidden]` tensor (2 GB on the 2B) and
  no copy per row.
- **`Qwen35Model::train_step(ids, operands)` and
  `cross_entropy_rows(.., reduction, operands, ws, grads)`** take a
  `gemm::GemmOperands`: `ExactF32` (the previous behaviour) or `Bf16`, bf16
  GEMM operands with f32 accumulation while weights, activations and
  gradients stay f32. The C ABI carries it: `TesslCeArgs.operands` and an
  `operands` parameter of `tessl_qwen35_train_step` (`TESSL_OPERANDS_EXACT_F32`
  0, `TESSL_OPERANDS_BF16` 1, anything else refused); `tessl_torch` takes
  `operands="f32" | "bf16"` (default `"f32"`) on `Qwen35.train_step`,
  `cross_entropy` and `cross_entropy_rows`. `Bf16` runs the
  2B's cross-entropy with gradients 3.15x faster at T = 2048 (561 against
  1768 ms); on the tiny model its gradients are within 2.3e-2 of
  transformers' f32 (see docs/qwen35.md, "A training step").
  `gemm_bf16`, `gemm_tn_bf16` and `gemm_nt_bf16` are that lane whatever the
  runtime's `PrecisionMode` (the `*_train` GEMMs now delegate to them), and
  `GemmOperands::ExactF32` refuses a runtime with relaxed precision on.
  `bench_qwen35_train --bf16` times it.
- `Tensor::from_mtl_buffer` is an `unsafe fn`: tessl cannot see another
  queue's work on a wrapped buffer, and its `# Safety` section states the
  cross-queue contract.
- `GpuRuntime::metal4` and `Metal4EncodePackage` are crate-private;
  `shared_event()` and `last_signaled_value()` remain the public handoff.
- `cross_entropy_rows` takes the hidden states and weight as `Tensor`s
  (`CeHidden { rows, off }`, `&Tensor`), so views at a storage offset work.
- `ParamsBuffer::push_u32`/`push_f32` take the host-access lease: they wait
  for encoded work and fail with "runtime busy" inside an encoder closure.

### Changed

- The `tinyalpha` GDN fixture is renamed `..._alpha_preclamp` and says it stores
  the pre-clamp alpha; `GateClamp::PUBLISHED` is documented as a deviation from
  transformers / FLA. `scripts/gen_qwen35_fixtures.py` reads stored inputs
  instead of seeded draws (goldens moved by at most 2.6e-7).
- An absent `attn_output_gate` or `mamba_ssm_dtype` takes the implemented value
  (gated, float32); a different value is refused.
- The safetensors shape-overflow check gives the same verdict in any dimension
  order (`plain::checked_numel`).
- `scripts/ci_local.sh` runs the kernel-emulator job's steps and names every
  step it could not run (`ci:local OK WITH SKIPS (not run: ...)`); the emulator
  stops a case that makes no progress for `MSL_EMU_STALL_S` seconds.

- `embedgemma2_vision::axial_qkv_norm_rope` (and the new `_bf16`) refuse a head dim over 512 (the shared rope table holds 8 pairs per lane); the config parser already limited `vision_config.head_dim` to 32, 64, 256 or 512, so no config loads differently.
- README and the crate docs count 298 kernel entry points (the vision tower's bf16 rope kernel and the eight chunked GDN training kernels, `gdn_train_c_*`, are the new ones).
- **EmbeddingGemma 2's device peak is 400 MiB above the model at 8192 rows,
  not 1472** (audit gap G6; `bench/results/embedgemma2_memory_m5pro.{txt,json,png}`).
  Three changes, no output bit moved (hashes of the real checkpoint's embeddings
  and the tiny model's, in both `GemmOperands` modes, equal 16f5190's), and no
  slowdown the shared machine's noise lets through (min-of-N against 16f5190
  0.86 to 1.05 over three paired series; the run-to-run spread there, 1.2 to 28,
  is too wide to verify a 3 % bound, and the results file says so; the
  per-layer-input bound alone, measured on one binary, is 0.96 to 1.03):
  `encode` hands its activations and each forward's uploaded inputs to the
  runtime before its one wait, so that wait recycles them and the next call's
  allocations find them (they were recycled a call late, and two sets were
  live across consecutive calls); `PLE_RUN_BYTES` is 16 MiB, not 256 (a run
  is one layer's inputs from 8192 rows, `-240 MiB` a set); and the `q|k|v`
  GEMM's output shares a buffer with the MLP's `gate | up` (`-96 MiB`). One
  encode and four back to back now peak at the same figure: 400.3 to 400.6 MiB
  at 32x256, 8x1024 and 2x4096 (736 and 1472 before), 112.9 MiB at 64x32 (433),
  2.0 MiB at 1x16 (4.25); Bf16 392.4 at 8x1024 (1456.4). The towers' `encode`s
  still hold their activations across the wait (see the audit). New:
  `tests/embedgemma2_memory.rs` (consecutive calls hold one set; one encode's
  device bytes equal the itemized set to the byte, both modes; an ignored
  hash printer), a `ple_run_elements` unit test, `profile_embedgemma2`'s
  `memory-table`, `hash` and `PROFILE_OPERANDS`, and `bench_embedgemma2`'s
  `BENCH_PLE_RUN_BYTES`.
- **GDN training reads q/k/v where the model keeps them.**
  `gdn_train_forward_windowed` / `gdn_train_backward_windowed` take `(ld, off)`
  column windows of the conv output and write `dq`/`dk`/`dv` back the same way,
  and index key heads as `h / (Hv/Hk)` so grouped heads are never repeated.
  `gdn_train_backward_windowed` sums `dq`/`dk` over each key head's value heads,
  in value-head order. The dense `gdn_train_forward` / `gdn_train_backward` are
  the dense window and delegate to them, so ojas-metal and the C ABI are
  unchanged.
  - The Qwen3.5 training step no longer runs `copy_cols`, `repeat_heads` or
    `sum_heads` for GDN layers. Those helpers and their staging buffers are
    deleted.
  - Gradients are bit-identical on the tiny and grouped fixtures.
  - The adversarial pass confirmed 64-bit kernel indices with rows past 2^32
    elements, that conservative aliasing refusals hold, and that results do
    not depend on the padding columns.
  - Paired step: within ±1% at r=1 (2B, f32 and bf16); 4-5% faster at r=2
    (4B config); `bench/results/qwen35_gdn_windows_m5pro.txt`.
  - `GdnTrainDims` refuses a row width past `u32`.
  - `bench_qwen35_train --random-config=PATH` times a size with no local
    checkpoint.

- **No panicking host access in library code.** Weight loads (`Loader`, BERT,
  Qwen3.5 `load` and `random_tower`) and the ICB triage probe return an error on
  a busy runtime instead of panicking. `tests/host_access_ban.rs` scans non-test
  `src/` (not `src/bin/`) and fails on any call to a panicking accessor. Load
  time on the fixture model did not change (min of 200, 3 runs each, before and
  after).
- **`add_metallib` / `add_metallib_bytes` refuse an overlay that repeats an entry
  point** already in the primary library or an earlier overlay, naming the
  function(s) and both libraries. `pipeline()` resolves the primary first, so
  such a kernel was unreachable. No caller in tessl or ojas shadows a name.
- **AdamW's f32 rule is documented as torch's order, not torch's bits.** The
  final update may differ from `addcdiv_` by up to 1 ulp per parameter.
- **The `mlx-f32` parity lane is scored as tf32-class.** Its GPU matmul measures
  3.5e-3 against 7e-7 on MLX's CPU stream on the M5 Pro, so valid parity dumps
  no longer abort wherever MLX is installed.
- **`scripts/ci_local.sh`** runs `cargo test --no-fail-fast` (failing if any
  binary failed), the torch-binding host tests, and the GPU binding tests under
  the `mac_heavy` lock. The release workflow's two gate edge cases are
  documented under "Releasing" in `docs/verification.md`.

- **Tests that passed without checking now check.**
  - `docs_name_real_tools` scans `README.md` and all of `docs/**`, and fails
    with `file:line` on a dangling bench binary or repo path. Cross-repo
    references are allowlisted, and the marker must be on the same line.
  - The promoted-kernel set is derived from the kernel sources instead of a
    literal 44.
  - TensorOps probes assert instead of returning early.
  - The genuinely hardware-dependent skips print `SKIPPED:`.
  - `set_scan_slice`, `new_inference` and `set_max_tokens` have tests.
  - The SharedEvent wait-budget test asserts the submitted value, a bounded
    deadline, and both ends of `EVENT_WAIT_MS_RANGE`.
  - A clean release build is recorded: 14.1 s min-of-3,
    `bench/results/build_time_m5pro.txt`.

- **Releases require green CI for the exact commit.** `release.yml` runs
  `scripts/require_green_ci.sh "$GITHUB_SHA" ci.yml metal-compile.yml` before
  anything publishes: a missing, running, failed or unqueryable run blocks the
  publish. Every publishing path (tag push, or dispatch with `dry_run=false`)
  must run from a `v*` tag matching `Cargo.toml`, and the GitHub Release is
  created with `--target "$GITHUB_SHA"`. A tag pushed before CI finishes on its
  commit is refused; re-run the release job once CI is green.
- Clippy also runs with `--features quant-prep` (`ci.yml`, `metal-compile.yml`,
  `scripts/ci_local.sh`), so `mtl_tensor` is linted.

- build.rs: every kernel's Metal compiler stderr is now forwarded as
  deduplicated `cargo:warning=` lines (previously only `matmul_tensorops.metal`
  surfaced warnings), replayed from the `.air` cache. `TESSL_METAL_WERROR=1`
  compiles every source, `kernels/tune/` included under `TESSL_GEMM_TUNE=1`,
  with `-Wall -Werror` and no metal3.2 fallback; `metal-compile.yml` runs the
  build that way instead of keeping a second source list and flag set.
  Rebuilds now also trigger on the resolved `metal`/`metallib` binaries, the
  SDK and `SDKROOT`/`TOOLCHAINS`/`MACOSX_DEPLOYMENT_TARGET`. The unread
  `metal_runtime_tensorops` cfg is gone.
- `kernels/flash_attn_decode.metal`: dropped two unused locals in the reduce
  kernel and marked the host-mirror `KV_CHUNK` constant `[[maybe_unused]]`
  (the only `-Wall` warnings in the tree).
- Docs: the tune-bench commands put `TESSL_GEMM_TUNE=1` on every `cargo run`;
  the tune-kernel count (71) is derived from a compile by
  `tests/docs_name_real_tools.rs`.

- **bf16 TN at the split-K-gated dW shapes runs a parallel split-K.** The
  shapes `prefer_tn_splitk` picks (K >= 2048, M, N <= 384, min <= 128) took
  a sequential split-K: a zero pass, then one 64x32 `multiply_accumulate`
  dispatch per 256 rows of K. Every partition now runs in one dispatch as a
  cooperative 64x64 tile set into its own scratch slice
  (`matmul2d_tensorops_tn_splitk_par_bf16_f32`), and `reduce_partitions_f32`
  (or the new `reduce_partitions_add_f32` for `C +=`) sums the slices in
  partition order; `gemm_tn_splitk_par_bf16` takes an explicit width. Paired
  on the M5 Pro it was 2.4-19x faster at its best width at every gated shape,
  overwriting and accumulating (`bench/results/splitk_par_bf16_m5pro.txt`,
  `bench_gemm_routes`, which also re-measured the imported split-K gate: it
  held, and is superseded). The sequential bf16 kernel is removed. Outputs
  at those shapes round differently (partials per partition), within the
  same f32 bound.
- **The cooperative NN paths share one tile rule.** bf16 now also takes
  64x64 when C has fewer than 64 tiles of 128x64, the rule
  `bench/results/bf16_smallm_coop_m5pro.txt` (an import from the
  MLSystemsLab copy) described and `bf16_smallm_coop_reconciled_m5pro.txt`
  re-measured on tessl's kernels (1.16-1.56x in every such case). The fused
  epilogue and `gemm_batched` (batch > 1) now pick the plain path's tile,
  with 64x64 epilogue and batched kernels for bf16, f16 and relaxed f32;
  the epilogue measured 1.11-2.01x faster on 64x64 wherever the rule picks
  it (`bench/results/gemm_coop_tiles_m5pro.txt`), and the relaxed-f32
  batched exceptions are in `gemm_batched`'s docstring. The tiles give the
  same bits, so outputs are unchanged.
- **One panel-walk policy for every GEMM family.** The cooperative NN
  kernels (plain, epilogue, batched) walked tiles by their own gate (8-row
  column panels from 2048 tiles); everything else used `tile_walk` (512-row
  panels once B reaches 2^23 elements, Morton kept). Interleaved per lane,
  `tile_walk` was never worse beyond noise and won where they disagreed:
  NN 1.14x at 1024x8192x2048 and 1.06x at the Qwen3.5-2B MLP up/gate shape,
  while the grid gate lost 0.94x on the 2B TN/NT shapes
  (`bench/results/gemm_panel_walk_m5pro.txt`). NN now walks by `tile_walk`;
  output is bit for bit unchanged (`gemm_coop_header_bits_m5pro.txt`), and
  `panel_walk_matches_row_major_chunks_bit_for_bit` now runs the coop NN
  kernels past the gate too.
- **`TESSL_GEMM_INTERIOR` is retired.** It sent the exact-f32 kernels'
  interior tiles through offset pointer tensors; interleaved on the
  Qwen3.5-2B training shapes, that ran at 0.99-1.01x of the slice path every
  tile takes by default (`bench/results/gemm_interior_m5pro.txt`). The
  branch, the kernels' `buffer(8)` argument and the variable (with its
  `METAL_RUNTIME_*` / `METAL_NATIVE_*` spellings) are gone; the default
  output is unchanged (the 495-output bit snapshot,
  `bench/results/gemm_coop_header_bits_m5pro.txt`), and
  `no_source_reads_a_retired_gemm_flag` pins that nothing reads it, or the
  accumulate flags, again.
- **One source for the GEMM tile bodies: `kernels/gemm_coop.h`.** The tile
  walks, the interior and edge operand views of a tile, the cooperative
  destination body (zero, run, accumulate or epilogue, store) and the
  exact-f32 body moved into a header that `matmul_tensorops.metal` and the
  tune variants both include. The six exact-f32 kernels are one template
  (`mm_exact_f32`, by layout and `multiply` vs `multiply_accumulate`), the
  epilogue is written once for interior and edge tiles, and every tune coop
  variant is an instantiation of the production `mm_coop` with its own
  tile, walk and layout, so a tune winner and the kernel that ships it
  cannot drift. The two kernel files went from 1714 lines to 909 plus the
  396-line header. Output is bit for bit what it was: FNV hashes of every
  production GEMM entry point over 495 (entry, shape) cases matched before
  and after (`probe_gemm_bits`, rebuilt against each commit in
  `bench/results/gemm_coop_header_bits_m5pro.txt`), and the GEMM suites
  pass unchanged.
- **`infer_trace` counts every allocator catch-up as a sync wait.** The
  counter's doc says `sync_waits` and `sync_wait_us` include allocator
  catch-ups, but only the catch-up inside a waiting commit was counted; the
  one a new command buffer makes when both allocators are in flight was not.
  It is now, so a run that mid-commits (`TESSL_MID_COMMIT`, the 100k-dispatch
  cap, or a constant-arena reclaim) reports the host waits it actually makes.
- **Training's layer forward is pinned to inference's.** `train_forward`
  composes the layers on the training kernels, separately from
  `Qwen35Model`'s inference `layer`;
  `train_forward_hidden_states_are_the_inference_forwards` holds its
  final-norm output to `forward`'s at every position within 1e-5 relative
  (observed 1.3e-6 and 1.7e-6 on the tiny and grouped fixtures), where the
  loss comparison saw only the head's average. `qwen35_model`'s docs no
  longer call it the one place the layer order is written.
- **Host-side dispatch overhead.** Owned binds (`bind_tensor`,
  `bind_gpu_buf_offset`) read a GPU address cached on the buffer instead of
  sending `length` and `gpuAddress` per bind; scalar binds use the const
  arena's pointer, GPU address and length read once at init instead of three
  property sends each; `Tensor::validate` compares runtime pointers without
  an `Arc::downgrade`; the per-dispatch and per-GEMM runtime flags
  (`async_encode`, `residency_dirty`, `dispatch_count`, precision,
  `relaxed_precision`, `flash_tensorops`, the residency count) are atomics
  rather than mutexes; `decode_icb_capture_active` reads a `Cell<bool>`
  instead of borrowing a `RefCell`; `BufferPool` reads `maxBufferLength`
  once; split-K GEMMs no longer collect their partition starts. Not yet
  measured on the M5 Pro: `bench_nn_kernels --host-only` (host_us) is the
  check.
  **Breaking:** the public field `GpuRuntime::dispatch_count` is an
  `AtomicUsize` (was `Mutex<usize>`); `take_dispatch_count` is unchanged.
- **bf16 GEMM operands are borrowed.** `gemm_bf16`, `gemm_tn_bf16`,
  `gemm_nt_bf16` and the bf16 accumulate lanes no longer clone a `Tensor`
  (a shape `Vec` and two refcounts) for an operand that is already bf16.
  `tests/host_path_allocs.rs` counts `gemm_bf16`.
- **Model outputs are copied once.** Qwen3.5 logits and traces,
  EmbeddingGemma 2 embeddings and traces, and BERT pooled outputs map the
  buffer and copy the logical prefix, instead of copying the whole buffer and
  then the prefix again (~1 GB per Qwen3.5 forward at t = 1024). A busy or
  poisoned runtime is an `Err` there rather than a panic.
- **Faster weight loading.** `place_linear_part` transposes in 64×64 tiles,
  and `SafeTensors::read_f32` widens bf16/f16 in blocks with the conversion
  inlined. Both produce the same bytes as before; each is ~2.3× faster in a
  standalone check.
- **Residency set committed once per command buffer.** `with_binder` no
  longer commits the residency set before every encode after an
  allocation; `commit_m4` commits it once, just before the command buffer
  (removals still commit after the GPU wait). A training step with ~800
  allocations committed it hundreds of times. `infer_trace`'s
  `residency_flushes` should now track commits.
- **Fewer bf16 casts in training.** On the bf16 lane, cross-entropy rounds
  the gathered rows once per call, each dlogits chunk once for both of its
  GEMMs and an f32 weight chunk once per walk, and passes a bf16 weight as it
  is (it was widened and narrowed back per GEMM); the training step rounds
  each activation several GEMMs read (`x1`, `x2`, `y`, `dresid`, `d_gate`,
  `d_up`, `dproj`) once into a bf16 twin. Same cast kernel, same bits;
  roughly 570 cast dispatches and 800 pool allocations fewer per 2B step.
- **A bank step without temporaries.** Into a bank it overwrites, the step
  writes every vector gradient in the bank's own buffer (133 allocations and
  copies per 2B step), the embedding backward's grouping goes into fresh
  buffers written without staging copies, and a causal step zeroes only the
  last row of `dxf`.
- **Unzeroed inference activations.** `Qwen35Model`'s forward and
  `BertSparseModel`'s encode no longer zero their activations on the host
  (about 1.2 GB per 1024-token Qwen3.5 prefill, 550 MB per 8192-row BERT
  forward); every element read is written earlier in the same forward, held
  by new poison tests (`tiny_forward_reads_no_unwritten_activation`,
  `tests/bert_tiny.rs`). BERT's head resets its pooled rows on the GPU and
  its index uploads, like Qwen3.5's `begin`, no longer wait for the GPU.
- **Decode replay without per-token probes.** `DecodeIcb::execute` reads
  `TESSL_ICB_EXECUTE` / `TESSL_ICB_TRIAGE` once (new `set_icb_execute` /
  `set_icb_triage` overrides) and checks pipeline ICB support once per tape,
  not per command per token; `cb_replay`'s step labels reuse their buffer
  (the label parameters take `impl AsRef<str>`, was `impl Into<String>`).
- **Smaller host costs elsewhere.** The Q4 MLX GEMVs and the AdamW step
  format no error labels on success (`format_args!`), AdamW's kernel names
  come from a fixed table and its pipelines and slot table are built once,
  and `Loader::linear` stages every projection part through one reused
  buffer instead of a fresh `Vec` each.
- **Parallel, incremental shader build.** `build.rs` compiles the kernel
  sources concurrently and reuses an `.air` whose inputs (source, headers,
  compiler, SDK, flags, the script itself) are unchanged; link order,
  diagnostics and the metal3.2 fallback are as before.
- **`tessl_torch` drains torch's queue less.** `train_step`,
  `train_forward` and `adamw_step` no longer call `torch.mps.synchronize()`
  (they share no torch buffer with tessl), and the cross-entropy autograd
  path skips the per-row loss copy (`cross_entropy_rows(..., per_row=False)`).
- **Host-only tests in CI.** The JSON parser tests, the loader's widen and
  transpose checks and AdamW's kernel-name table run on the hosted macOS
  `check` job, which needs no Metal device.
- **`grad_sq_norm` is one command buffer and one wait.** It was a dispatch
  per parameter (a waited commit each under sync encode: 320 on the 2B), a
  read-back of every row's sum and an f64 sum on the host, with each dense
  gradient (the whole `[vocab, hidden]` embedding) summed by one
  threadgroup. Now `qwen35_sq_sum_pieces` sums every gradient's squares in
  pieces of at most 4096 elements in one table-driven dispatch, and
  `qwen35_sum_exact_f32` adds the pieces exactly in 128-bit fixed point
  (integer arithmetic, so no order or fast math can change it), which the
  host rounds to f64 once. The result is within 2^-18 relative of the exact
  sum of the squares, and the exact sum rounded once wherever the pieces are
  exact; on the tiny fixture its bits are unchanged. Replaces
  `qwen35_sq_sum_rows_{f32,bf16}`. On the 2B (bf16 storage, T = 256) it
  takes 41-56 ms where it took 660-770 ms, and a clipped optimizer step
  2.19-2.23 -> 1.54-1.69 s under sync encode, 1.78-1.92 -> 1.33-1.35 s
  under async (`bench/results/qwen35_train_step_wave2_m5pro.txt`).
- **A bf16 bank takes the head gradient a chunk at a time.** A step into a
  bf16 bank (or an f32 bank embedding that is a view) no longer allocates a
  `[vocab, hidden]` f32 head gradient (2.0 GB on the 2B, 2.54 GB at the
  4B's shapes): each cross-entropy chunk is rounded into the bank as its
  GEMM produces it, and the step's token rows wait in f32 until the
  embedding gather has added to them, so every element is still rounded
  once (bit for bit as before on every entry point). The backward also
  releases the final norm's three `[T, hidden]` tensors before the layers.
  At the 4B's shapes the step's peak over resident falls 5.76 -> 3.35 GiB
  at T = 2048 and 3.78 -> 1.40 GiB at T = 512
  (`bench/results/qwen35_4b_storage_memory_wave2_m5pro.txt`); on the 2B
  the device peak falls 1.9 GiB at the same step time.
- **The pool cache makes room for a smaller freed buffer.** A buffer freed
  into a full cache now evicts a larger cached one instead of being
  released itself (the largest buffer freed is still released when it does
  not fit). The cache never evicted, so a large buffer freed once and never
  reused held it for good: loading the 2B leaves its f32 `[vocab, hidden]`
  staging tensor (1.9 GB of the 2 GiB cap) there, after which a bf16-bank
  training step released and re-created 668 buffers (6.8 GB) in its
  backward alone. The cache now holds the step's own freed buffers, so on
  the 2B an f32 step at T = 256 peaks 1 GiB higher (inside
  `train_step_bytes`, which counts the cap) at the same time.
- **`CeWorkspace` is per operand lane (breaking).** `CeWorkspace::new`,
  `bytes_for` and `allocated_bytes_for` take the `GemmOperands` lane, and a
  call on another lane is refused. On bf16 operands the workspace holds the
  rows and one dlogits chunk rounded to bf16 and no f32 copy of a bf16
  weight chunk (80 MiB at the 4B's shapes); the per-call bf16 temporaries
  are gone. The C ABI keys its cached workspace on the lane.
- **Bench and probe modes.** `bench_qwen35_train --storage=bf16` steps a
  bf16-stored 2B into a bf16 bank (and clips with a Kahan, 8-bit AdamW);
  `probe_storage_memory --no-optimizer` measures the step with no optimizer
  resident, under a per-process footprint cap. `infer_trace`'s kernel trace
  counts pipeline requests per kernel (`runtime::traced_kernel_count`).
- **Qwen3.5 training step traffic.** Allocations a kernel writes in full
  are no longer zeroed on the host (808 allocations, 25.86 GB per 2B step at
  T = 2048); the step makes one attention workspace and uploads its indices
  without draining the GPU (dxf is zeroed on the GPU); a step into an f32
  bank writes every weight matrix's and the head's gradient in place (no
  `[vocab, hidden]` head tensor); the residual, MLP-backward and
  cross-entropy dh sums form in their GEMMs (`CeWorkspace` loses `dh_part`).
  Gradients change by rounding: an accumulated f32 bank sits within 8 u of
  the tensor's largest `|x| + |y|` of `f32(x + y)`. AdamW is one
  table-driven dispatch per step kernel, bit-identical. Paired against the
  previous build: -14% / -17% per step sync and -32% / -4% async at
  T = 256 / 2048, peak memory not above before
  (`bench/results/qwen35_train_step_{before,after}_m5pro.txt`).
- **Samplers only encode; `nn::check_argmax_result` reads and refuses the
  token.** `softcap_sample`, `softcap_argmax_one_pass` and `argmax_f32_pass`
  (and their `_with_scalars` forms) no longer synchronize and read their output
  back to refuse a row with no finite logit, so a decode loop can queue the
  sampler behind the step and a multi-pass argmax no longer drains after every
  pass. Read the token with `check_argmax_result(out)`, which returns it or
  the same "no finite value" error; for a multi-pass argmax, pass the final
  pass's `out_idx`. This also fixes a refusal that was wrong: the per-pass
  check rejected a row in which only one 256-logit group had no finite value
  (a masked block of the vocabulary), although the next pass skips such a
  group. A caller that relied on the sampler's own `Err` must now call
  `check_argmax_result`. The first `argmax_f32_pass` binds `logits` in its
  unused `idx_in` slot instead of allocating a 4-byte placeholder.
- **Decode attention takes a caller-owned `nn::DecodeScratch`.**
  `nn::flash_attn_decode`, `nn::flash_attn_decode_with_chunk`,
  `qwen35::attn_prefix_decode` and `qwen35::attn_prefix_decode_varlen` take
  a `&DecodeScratch` (after `o`) instead of allocating the partial pass's
  scratch on every call, so a decode loop binds one buffer at one address
  every token. Size it with `DecodeScratch::new(rt, batch, heads,
  key_capacity, head_dim)` (`with_chunk` for an explicit chunk); a scratch too
  small for the call, or one that is also an operand, is refused before
  anything is encoded. The `nn::flash_attn` router keeps its signature and
  sizes a scratch per call. This breaks callers of the four functions.
- **Decode-path validation allocates nothing on success.** The buffer checks
  take their error label as any `Display`, so call sites pass `format_args!`
  and format only on failure, and the GDN, conv and QK-norm/RoPE entry points
  check optional operands without collecting them into a `Vec`.
  `tests/host_path_allocs.rs` counts heap allocations per call across every
  entry point a batch-1 Qwen3.5 decode token encodes, and holds them at zero.
- **Pipeline cache hits allocate nothing.** `GpuRuntime::pipeline` keeps
  plain and ICB-capable pipelines in separate maps, so the ICB mode no longer
  formats an `icb:{name}` key per call; a miss looks the function up once per
  library instead of twice; and the decode/rows attention, `out`-dtype
  Qwen3.5 kernels resolve static entry-point names. `decode_icb::pipeline_icb`
  now returns the cached ICB pipeline instead of compiling a new one per call.
  Pooled-buffer allocation no longer locks a mutex to read the runtime's weak
  self-handle.
- **EmbeddingGemma 2's forward: fewer passes, no host zeroing, matrix-unit
  attention.** One set of activations per `encode`, allocated with
  `alloc_tensor_unzeroed` for its largest forward and reused by each (the
  activations were reallocated and memset on the host every forward, about
  1.8 GiB at 32,768 rows). One K/V pair serves every layer: `encoder_attn`
  derives the K/V capacity from the buffers' size, as the K/V writer does.
  The weightless V norm runs inside the qk-norm-RoPE kernel
  (`QkvColumns::v_norm`), the `sqrt(hidden)` embedding scale inside the
  gather (`qwen35::embed_rows_scaled`), `gate | up` is one GEMM read by a
  column-window GELU-tanh (`qwen35::gated_act`), and the per-layer inputs of
  a run of layers come from one NT GEMM (all 24 layers up to 5,461 rows;
  `PLE_RUN_BYTES`, `EmbedGemma2Model::set_ple_run_bytes`). On a device with
  TensorOps, `encoder_attn` runs the FlashAttention-2 tiled body
  (`kernels/attn_tiled.h`, shared with Qwen3.5's prefill) at head dims 256
  and 512; `encoder_attn_with` names the kernel. The flash and encoder row
  kernels share their online-softmax body (`kernels/attn_rows.h`).
- **One checkpoint loader for EmbeddingGemma 2 and Qwen3.5**
  (`src/loader.rs`). EmbeddingGemma 2's projections are placed into their
  packed device tensors part by part as they are read, and its embedding and
  per-layer-input projection are read straight into device tensors, with no
  host copy.
- **`qwen35::swiglu` is `qwen35::gated_act` with `GatedAct::Silu`**; the
  kernel takes its activation as a parameter (`qwen35_gelu_tanh_glu_*`).
- **C ABI 10: `tessl_qwen35_load` takes a precision and
  `tessl_qwen35_adamw_init` an optimizer configuration.** `tessl_qwen35_load`
  gains `precision` (`TESSL_F32` or `TESSL_BF16`) before `out`;
  `tessl_qwen35_adamw_init` gains `update` (`TESSL_UPDATE_*`), `seed` and
  `moments` (`TESSL_MOMENTS_*`). A caller built against ABI 9 must pass them
  (`TESSL_F32`; `TESSL_UPDATE_F32, 0, TESSL_MOMENTS_F32` for the old
  behaviour); `tessl_torch` refuses a library of another ABI version.
- **A bf16-stored Qwen3.5 model's forward runs on the f32 residual stream.**
  It still keeps each layer's input in bf16 for the backward, which rebuilds
  the layer from that rounded copy. The forward no longer runs on the stream
  rounded at every layer boundary: on the 2B that rounding compounded
  through the layers above and put four 1-D gradients past the bf16-storage
  test's 2^-4 bound (layer 0's `dt_bias` 1.1e-1 from the f32 step's). The
  loss is now the f32 model's bf16-operand loss, bit for bit.

- **Checkpoint reads fill their destination directly.** `read_bf16_bits` and
  `read_f32` no longer stage the raw bytes and then copy them. A 16-bit tensor
  read as f32 is widened in place inside its own result. `npy` and
  `safetensors` share the one unsafe byte view that does this (`src/plain.rs`).
- **`Qwen35Model::load` streams its weights.** Each projection part is placed
  into its packed tensor as it is read, and the embedding is read straight
  into its device tensor where no transposed head needs a host copy. At 2B the
  peak memory footprint falls from 6.95 to 5.86 GB (bf16), from 5.93 to
  3.83 GB (bf16 `load_tower`) and from 10.79 to 7.63 GB (f32). See
  `bench/results/qwen35_load_rss_m5pro.txt`.
- **One 16-byte operand-alignment rule for every GEMM family.** The
  cooperative (bf16/f16/relaxed-f32), accumulate, epilogue and batched entries
  used to require 64-byte operand views while exact f32 required 16, so
  `gemm_tn_f32` took a view `gemm_tn_accum_train` refused, and
  `set_relaxed_precision(true)` made previously valid views fail. A probe
  (`bench/results/gemm_align_probe_m5pro.txt`) ran every kernel family below
  the host gate on views 4–64 bytes into their buffers, under Metal API and
  shader validation. Every run was bit-identical to its 0-offset run, so 64
  bytes was not a hardware requirement on M5 Pro. Every entry now accepts any
  16-byte-aligned view.
  - **`gemm_batched` checks every batch start, not just the base.** A stride
    whose byte size is not a multiple of 16 is refused, because it puts a later
    batch off the boundary. Contiguous strides on matrices whose byte size is
    not a multiple of 16 (for example bf16 `m * k` odd in 8-element units) are
    now refused. They were accepted before.
  - **Alignment errors name the entry point and operand.** For example,
    `gemm_tn_accum_train: operand A byte_offset 4 is not 16-byte aligned`
    replaces `GEMM cooperative path: byte_offset 4 is not 64-byte aligned`.

- **README Known Gaps no longer claims "no stubs".** `IcbReplayStub`,
  `IcbStubPhase` and `CbReplayError::NotWired` (`tessl::cb_replay`) are public
  and unchanged — no API or semver impact — and full decode-graph ICB replay is
  now listed as a gap. The `IcbStubPhase::Allocated` docs said it was
  unreachable; it is set by a mini `DecodeIcb` attach or execute, and now says
  so.
- **Bf16 operands whose M does not fill a 128-row tile use the 64×64 tile.**
  An NN GEMM with bf16 operands whose $M < 128$ now dispatches
  `matmul2d_tensorops_bf16_f32_64x64_sg4` even when $N > 512$, passing $N$ as
  a runtime extent. Operands with $M \ge 128$ keep the previous rule: 64×64
  only when $N \le 512$. Fused epilogues (`gemm_epilogue`) follow the same
  rule for bf16: shapes with $M < 128$ dispatch the 64×64 instantiation
  `matmul2d_tensorops_bf16_f32_epi_64x64_sg4`, while other shapes remain on
  128×64.
  - **Explicit tile control:** Added `gemm_tiled` and `gemm_epilogue_tiled`
    with the `EpiTile` enum (`Narrow` = 64×64, `Wide` = 128×64) so benchmarks
    and numerical tests can compare both tile geometries on identical buffers.
    Tile overrides require cooperative-destination backends, and an explicit
    tile on an identity epilogue is refused before dispatch.
  - **Speed:** M5 Pro, paired interleaved rounds (`bench_gemm_coop_tile` and
    `bench_gemm_epi_tile`) at the Qwen3.5-2B GDN fused in-projection
    ($K=2048$, $N=8224$): at $M=61$, 64×64 was faster with non-overlapping
    ranges; at $M=200$, 128×64 was faster.
  - **Qwen3.5 GDN chunk scan default switched to 16-column slices:**
    `GdnScanSlice` now defaults to `Cols16` (`qwen35_gdn_chunk_scan_bv16`).
    Paired timing on an M5 Pro (`probe_gdn_scan --paired`) showed `Cols16`
    outperforming `Cols32` at batch 1 ($T=200$ and $T=8192$) and batch 2
    ($T=61$ median 0.0781 vs 0.1005 ms, $T=200$ median 0.3210 vs 0.3839 ms)
    with 0 mismatches in output or final recurrent state. Added
    `GdnWorkspace::set_scan_slice` to retarget scans on an existing workspace.
  - **Prefill attention dispatches tiled kernel at all lengths:**
    `attn_prefill_by_length` and `prefill_attn_kernel` select
    `PrefillAttnKernel::Tiled` (`ATTN_PREFILL_TILE`) across all $t_q$ values
    without a length cutoff. Paired A/B timing (`bench_qwen35_layers --paired-attn`)
    showed `attn_prefill` consistently faster than `flash_attn_rows` at both
    $T=200$ (0.116 vs 0.232 ms) and $T=8192$ (55.3 vs 176.2 ms) with
    non-overlapping ranges.
  - **Audit:** `scripts/audit_gemm_tiles.py` now parses `NN_COOP_EPI_KERNEL`
    macros, so the epilogue pipelines are checked too. The audit's own `PASS` line is
    the current count (33 kernels, 38 Rust dispatch pairs at `2972027`).
  - **Tests:** `short_m_bf16_plain_gemm_matches_the_wide_tile`,
    `narrow_bf16_epilogue_matches_wide_within_unfused_tolerance`,
    `short_m_bf16_ragged_extents_match_the_other_tile_and_the_reference`,
    `f16_and_f32_at_a_short_m_shape_are_not_the_bf16_kernel`,
    `overlapping_short_m_output_is_rejected_before_dispatch`,
    `short_m_epilogue_ragged_bias_keeps_its_guard_and_refuses_the_wrong_tile`,
    `short_m_epilogue_refuses_a_bias_that_aliases_the_output_before_dispatch`.

- **Exact-f32 TN with a small C runs all its K partitions in one dispatch.**
  A TN whose C has fewer than 128 32×32 tiles gave the single dispatch only a
  few threadgroups, each walking all of K: the per-head gate weight gradient
  (12 × 768 over 4096 rows, 24 tiles) ran at ~0.33 TFLOP/s. Such a TN, over a
  K at least two partitions long, now computes every partition at once into a
  scratch (`matmul2d_tensorops_tn_splitk_par_f32`) and adds them in partition
  order (`reduce_partitions_f32`). The width (`tn_par_k_tile` in
  `src/gemm.rs`) aims at ~768 threadgroups in multiples of 256, and the
  scratch is capped at 2^22 floats. `gemm_tn_splitk_par_f32` is public, taking
  the width, so a bench can sweep it.
  - **Replaces the sequential split-K for overwriting f32 TN.** The shapes
    `prefer_tn_splitk` picks (K ≥ 2048, small M·N) took one dispatch per
    256-row partition, adding into C; that was slower than the single dispatch
    there. The sequential kernel stays for bf16 and for `C +=`
    (`gemm_tn_accum_train`), which must add into C.
  - **Speed:** M5 Pro, `metal_bench tn`, min of 20 GPU spans in one quiet run
    (the per-shape table is in ojas
    `bench/results/2026-10-02-gate/tn-sweep/tn-sweep-2.md`): gate dW 225 → 45 µs;
    attention dW (128 × 128 × 4096) 186 µs sequential → 31 µs; MLP dW
    (128 × 384 × 4096) 192 → 74 µs; 64 × 64 × 4096 186 → 19 µs. Shapes of 128
    tiles or more are unchanged. In ojas's gate backward (interleaved A/B,
    4 rounds per side, under outside load), `gw` went from 227–234 to
    43–46 µs and the whole backward from 790–806 to 602–638 µs.
  - **Numerics:** deterministic (fixed partition order), not bit-identical to
    one dispatch over all of K, inside the same f32 bound.
  - **Tests:** `few_tile_long_k_tn_takes_parallel_partitions` (routing, and
    the scratch cap over the old sequential domain);
    `parallel_tn_partitions_cover_all_of_k`,
    `parallel_tn_refuses_bad_widths_and_an_oversized_scratch` and
    `parallel_tn_is_bit_identical_across_100_calls_in_flight` (GPU). The GPU
    coverage test fails with A's partition offset dropped or the last
    partition left out of the sum. Packing the scratch slices unpadded
    survives the tests: the M5 Pro computes correctly from a slice that is
    only 4-byte aligned. The padding is kept for tessl's 16-byte rule on GEMM
    operands, not because a test observes it.

- **Exact-f32 NN with a long K runs as K partitions.** An NN whose B (K×N)
  holds ≥ 2^23 elements, with more than one 32-row tile row and K at least four
  partitions long, now zeroes C and accumulates one dispatch per partition of
  `2^21 / N` rows of B (a multiple of 256; 2560 at N = 768), in order, through
  the new `matmul2d_tensorops_nn_splitk_f32` (`nn_splitk_k_tile` in
  `src/gemm.rs`). The single dispatch re-read all of B for every 32-row tile
  row, and the column-panel walk cannot help there because each tile's A slab
  is K long too. The LM head's input gradient (4096 × 768 × 50304) ran at
  ~3 TFLOP/s.
  - **Speed:** M5 Pro, interleaved A/B, min of 4 per side: that GEMM went from
    99.8 to 45.4 ms (~7 TFLOP/s; per-run ranges 100–113 vs 45–50 ms). K = 16384
    went from 21.0 to 14.6 ms. Shapes that do not route are unchanged.
  - **Numerics:** results are deterministic, since the partitions run in a
    fixed order. They are not bit-identical to the single dispatch, because C
    is rounded once per partition, but they stay inside the same f32 bound.
  - **Shared dispatcher:** the TN split-K lanes (f32 and bf16) now share the
    partition dispatcher, `dispatch_k_partitions`, with their 256-wide
    partitions unchanged. It refuses a K·max(M, N) that the kernels' u32
    offsets cannot hold.
  - **Tests:** `long_k_nn_with_a_large_b_takes_k_partitions` (routing) and
    `long_k_nn_partitions_cover_all_of_k` (GPU, rank-one B so every k
    contributes, short last partition, ragged M and N). The GPU test fails
    with A's partition offset dropped or a partition skipped.

- **Exact-f32 GEMMs walk large B operands in column panels.** The five
  exact-f32 TensorOps kernels (`matmul2d_tensorops_f32`, `_tn_f32`, `_nt_f32`,
  `_tn_accum_f32`, `_nt_accum_f32`) walked tiles row-major. That order re-reads
  all of B once per 32-row tile row, so an LM head (`[4096,768] · [50304,768]^T`)
  ran at ~2.3 TFLOP/s against ~6.5 for transformer-width GEMMs. Once
  `N·K ≥ 2^23` elements they now take 16-tile-row column panels through
  `tile_from_linear_panel`. The bf16 coop NN kernel's 8-row swizzle now calls
  the same helper, which also guards an id past the grid (the current
  dispatch never issues one). Each tile's arithmetic is unchanged; only the
  order threadgroups run in moves. On M5 Pro, interleaved A/B, min of 4:
  NT with B ≥ 50 MB takes 0.39–0.47× the time (LM head 135–138 → 64 ms), and
  TN with a 32 MB B takes 0.77×. Smaller B keeps row-major order: a 12.6 MB B
  measured ~1.08× slower in panels. The threshold was fitted on M5 Pro only;
  25 MB was within noise. Tests: `exact_f32_column_panels_cover_every_tile`
  (`tests/gemm_ragged_shapes.rs`), plus panel shapes in the accumulate and
  interior-branch tests of `tests/gemm_flag_paths.rs`. They guard the tile
  mapping, and all three fail when the partial-band clamp is removed. The
  speed-up itself rests on the A/B; no test pins it.

- **Bf16 TN/NT and int8 GEMMs walk large B operands in column panels too.**
  The bf16 TN/NT coop kernels (`matmul2d_tensorops_tn_bf16_f32`,
  `_nt_bf16_f32`, `_tn_accum_bf16_f32`, `_nt_accum_bf16_f32`) and the int8
  dequant kernel (`matmul2d_tensorops_i8_f32`, behind `nn::gemm_i8_dequant`)
  walked tiles in Morton order on square power-of-two grids and row-major
  otherwise, so they re-read a large B once per tile row, as the exact-f32
  kernels did. All three families now share one walk, `tile_walk<SM>`. Once
  `N·K ≥ 2^23` elements it takes column panels of 512 rows of C (16 tile
  rows of 32, 4 of 128, 8 of 64), unless `tile_from_linear` walks the grid
  in Morton order. That exception changes exact f32 too: past the gate on a
  square power-of-two grid it took panels before and now keeps Morton. Each
  tile's arithmetic is unchanged.
  - **Speed:** M5 Pro, production A/B of back-to-back pairs, median of 6
    rounds. The four controls over 2 ms read 0.99–1.00×, the three under it
    1.04–1.06×. bf16: TN accumulate 768×50304×4096
    took 0.60× the time, NT accumulate 4096×32768×768 0.50× and NT
    1024×248320×2048 0.72×. int8: 0.73–0.77× with B ≥ 24 MiB, 0.91× at
    16 MiB and 0.99–1.00× at 8–12 MiB. Exact f32 on square grids, Morton
    against panels: 0.99–1.02×. The band and the gate come from an
    in-process sweep against bands of 1024 and 2048 rows. The data is in
    ojas `bench/results/2026-10-06-gemm-bf16`.
  - **Tests:** `panel_walk_matches_row_major_chunks_bit_for_bit`
    (`src/gemm.rs`) runs each bf16 TN/NT and exact-f32 lane past the gate. It
    compares the result bit for bit against the same GEMM done in column
    chunks small enough to stay under the gate. It covers ragged edges, a
    partial band and the Morton grids. `column_panels_cover_every_tile_exactly`
    (`tests/gemm_i8.rs`) checks int8 exactly with a rank-one A. Both fail,
    as does `exact_f32_column_panels_cover_every_tile`, when the partial-band
    clamp is removed. The speed-up rests on the A/B; no test pins it.


- `Qwen35Model` in `Precision::F32` holds the tied embedding as one f32
  `[vocab, hidden]` table (the gather reads it through the new
  `qwen35_embed_rows_f32`, training's cross-entropy as its weight, and
  `forward()`'s head through `gemm_nt_f32`) instead of a bf16 gather table
  plus an f32 `[hidden, vocab]` head: 1 GB less on the 2B, a parameter write
  is exact, and gradients are taken at the parameters themselves rather
  than at a bf16 rounding of the embedding. `qwen35::embed_rows` accepts an
  f32 table. The bf16 model is unchanged.
- `Qwen35Model` stores the zero-centred norms (`input_layernorm`,
  `post_attention_layernorm`, the final `norm`) as `w`, as the checkpoint
  holds them, and their kernels form `1 + w` in f32, as transformers does
  and as the attention Q/K norms already did: the new
  `qwen35::rms_norm` (`qwen35_rms_norm_f32`, `_bf16`) for the forward, and
  `qwen35_rms_norm_bwd_f32` (whose `w` is now the stored `w`). Storing
  `1 + w` rounded `w` to ulp(1 + w), so a parameter written and read back,
  or a checkpoint restored into a fresh model, moved by an ulp (a stored
  0.3405694 came back as 0.34056938), and each AdamW update of a norm was
  rounded to about 6e-8. Reads and writes are now exact for every entry and
  AdamW updates the norms as stored; the forward and the gradients are the
  same bits as before on the tiny model (f32 and bf16), and
  `qwen35::rms_norm` in f32 is `nn::rms_norm_f32` on `1 + w` bit for bit.
- **Static documentation drift.** `docs/verification.md` quotes the tile
  audit's own output (33 kernels, 38 Rust dispatch pairs at `2972027`, from
  `python3 scripts/audit_gemm_tiles.py`) and the `#[test]` count (800: 177 in
  `src/`, 623 in `tests/`), and no longer presents 349 passing as current;
  the GPU pass counts are not re-run at that commit. `docs/qwen35.md` states
  that the chunked scan defaults to `Cols16` and that the decode recurrence's
  `Cols32` is the untuned one, and names `tools/mac_heavy.sh` as Lappi-decision's.
  `docs/bert.md` documents the real `BertSparseModel::load(rt, &SafeTensors,
  BertConfig)` and `encode(&[&[u32]], trace)` and the bounds
  `tests/bert_kernels.rs` uses, and drops the nonexistent `encode_topk`. The
  `gemv_q4` and `GdnTrainWorkspace` rustdoc match the code (no threadgroup
  staging of `x`; a workspace is for exactly one `GdnTrainDims`), the two
  `qwen-decision` references now name `Lappi-decision/AUDIT/`, and the
  duplicate `[Unreleased]` headings are merged and given a compare link.
  `bench/results/PROVENANCE.md` lists the 29 files imported from the
  MLSystemsLab copy, `docs/benchmarking.md` labels each one it presents as
  imported, and it no longer quotes or plots attention ratios against MLX or
  PyTorch from that data.

### Fixed

- The emulator's `--fast-math` bound for `encoder_attn_rows` is the on-device
  bound (`2e-5·max|want|`); hosted run 38739221987 failed on the old 2e-6.
- The CI workflow gate is an allowlist, and every self-hosted job carries it.

- **ICB replay.** Tape replay emits exactly the captured barrier markers, and
  capture now records the ordering a command-buffer boundary gave live
  execution (sync encode in hazard mode previously captured no edges). A no-op
  execute is no longer counted as a replay; a no-op scope on a poisoned runtime
  reports the poison; `execute_icb_ex` refuses empty and overflowing ranges;
  barrier elision keeps the barrier after an Immediate command; dropped
  `DecodeIcb`/`IcbCopySmoke` ICBs leave the residency set. The tri-state flag
  getters share one `TriFlag` helper (no env name changed).
- **Checkpoint readers refuse non-regular files** (`plain::open_regular_file`,
  `O_NONBLOCK` + fstat): `SafeTensors::open` and `read_npy` no longer block
  forever on a FIFO.
- **`bench/kernel_coverage.py --check` runs.** It reads the metallib the
  benches embed (`bench_gemm_sweep --dump-metallib`), takes its census from
  `xcrun metal -E` (266 entry points, matching `metal-nm`), and exits 77 when
  it cannot run. `bench_gemm_sweep`/`bench_nn_kernels` emit `KERNEL_TRACE`, and
  `bench_nn_kernels` runs at `BENCH_WARMUP=0`. `bench/benchmark_evidence.py` is
  removed.
- `softcap_sample` drops a redundant `!isfinite(v)`; two kernel comments no
  longer claim fast math folds `isnan`/`isinf`.
- The tile audit is a Rust test (`tests/audit_gemm_tiles.rs`);
  `scripts/audit_gemm_tiles.py` is removed. `dialect_lint.py` fails on a stale
  allowlist entry.

- **C ABI: the owning thread is checked before any reference to a handle is
  formed.** `guarded` and `free_handle` formed `&mut *handle` and then compared
  the owner, so a refused foreign-thread call could alias the owner's live
  `&mut`. The owner is now read through the raw pointer (`Handle::owner_of`)
  first. The module doc states that a freed handle cannot be detected and that
  a second free is undefined behaviour (`tests/capi_contract.rs`, GPU-free
  where possible).
- **`tessl_torch` handle ownership.** `Qwen35` and the per-thread runtime refuse
  `copy`, `deepcopy` and pickle (a copy would double-free), gain an idempotent
  `close()` and context-manager support, and raise `TesslPanic` on a panic
  status, closing the object and dropping that thread's cached runtime (the Rust
  model holds its own `Arc`). Integers bound for the C ABI (`chunk`, `step`,
  `seed`, GDN dimensions, `accumulate`, AdamW `step`) are range-checked and
  refuse floats and `bool`; float index tensors are refused, not truncated.
  `close()` reports a failed free instead of swallowing it
  (`python/tests/test_binding_host.py`, 21 host tests).
- **Four GPU tests that had failed on `main`, unseen because hosted CI runs no
  GPU tests, are corrected; no library code changed.**
  `global_h512_with_scalars_pins_out_bf16_after_the_callback` compared the tiled
  kernel `_with_scalars` dispatches against `flash_attn_global_h512`, which
  routes `Tq > 1` to the rows kernel, so the two accumulated in a different
  order. It now compares against `flash_attn_global_h512_tiled`, and it still
  fails when the slot-12 override is removed. Two `tests/capi.rs` expectations
  still said a step refused for its operands code discards the gradients;
  4551470 made it keep them, as `tessl_qwen35_train_step` documents. A third
  read gradients into transformers' shape instead of the storage shape.

- **Busy host mappings return errors, not panics, in `upload_segments` and
  `DecodeIcb::mini_copy_chain`.** Both wrote a small host buffer through the
  panicking `write_u32`; `upload_segments` now uploads through
  `alloc_buffer_from_u32` and `mini_copy_chain` initialises all four of its
  buffers under one host-access lease, so a live `HostMapping` (or an active
  encoder or submit) is reported as `runtime busy` with the function's name.
  `mini_copy_chain` also refuses an `n` that does not fit a `u32` or overflows
  `n * 4` before allocating (`tests/host_access_fallible.rs`; the two busy tests
  panic at `src/tensor.rs:317` on the previous code). The `set_max_tokens`
  rustdoc names the real cap, `MAX_BATCH_TOKENS` (32768); 262144 is the
  checkpoint's `max_position_embeddings`, which is not a limit here.
- **`bench/test_parity_harness.py --pure` no longer depends on which comparison
  runtimes are installed.** The failure blamed on numpy 2.4.6 was MLX: where
  `mlx` is importable, the scorer contract tests scored the live MLX GPU f32
  matmul, which is reduced precision on the M5 Pro (about 3.5e-3 absolute at
  K=16 against 7e-7 on MLX's CPU stream) and broke the f32 budget by about 600x
  on a valid dump. `parity()` takes `external_lanes`, validated up front (known,
  unique names that do not collide with a dumped lane); the contract tests pass
  deterministic stub lanes, including bf16 ones, and cover a raising lane, a
  partial-seed failure and a dtype flip. An empty list is reported as "no
  comparison runtimes scored", and a tessl-only breach without them is
  "unconfirmed" rather than "a kernel defect". Budgets are unchanged.

- **`rms_qkv_rope` refuses `eps == 0`.** `QkvRopeDims::validate` refused only
  a negative eps, so an eps of 0 with an all-zero head row gave
  `0 * rsqrt(0)` = NaN in q and k and, in the KV-store variant, in the
  persistent cache. All three variants now go through the one RMS eps check
  (`validate_rms_eps`, also behind `validate_rms_scalars`): finite and
  positive, refused before anything is encoded.
- **`Decode::step` reads its logits before counting the token decoded.** A
  failed read used to leave `position()` advanced past a token whose logits
  were lost. It now leaves the position where it was and the session poisoned,
  as a failure in the layers does.
- **`GemmOperands` lane contract and dispatch limits.** `ExactF32.nn` refuses
  bf16 and f16 operands, as `tn` and `nt` do, instead of running the narrow
  cooperative kernel. Every "needs exact f32" refusal (`GemmOperands::ExactF32`,
  `cross_entropy_rows`, BERT, EmbedGemma2, the Qwen3.5 F32 forward and training
  step) asks one predicate, `GpuRuntime::f32_gemm_runs_relaxed`, rather than the
  raw `relaxed_precision` flag, which is ignored under `PrecisionMode::Bf16`:
  `ExactF32.nt` and the rest now run there with the flag on. `dispatch_tg_1d`
  and `dispatch_2d_tg` return an error naming the kernel, the request and
  `maxTotalThreadsPerThreadgroup` for an oversize threadgroup, before encoding.
  `flash_attn_global_h512_with_scalars` binds `out_bf16` (slot 12) itself after
  the callback, as the D=256 wrapper does its slot.
- `npy::read_npy` panicked (slicing inside a multi-byte character) when
  `fortran_order` was followed by non-ASCII text; it now returns the error.
- **C ABI: validate before mutate, allocate or publish.** Every slice built
  from a caller's `n` goes through one checked helper that refuses a byte
  length past `isize::MAX`, a null pointer with elements and a misaligned
  pointer (undefined behaviour in release, an uncatchable abort in debug)
  with `TESSL_ERR`. `tessl_qwen35_copy` checks its direction and the state it
  needs before wrapping any tensor. `tessl_cross_entropy_rows` and
  `tessl_gdn_train_backward` run the same validators as their Rust entry
  points (`validate_ce_inputs`, `validate_gdn_backward`) before replacing the
  cached workspace, so a refused call keeps it; the CE `chunk` is clamped to
  the vocabulary, so it no longer sizes the logits buffer.
  `tessl_qwen35_train_step` parses `operands` before it marks the bank dirty:
  a refused argument leaves the previous gradients readable, and only a step
  that starts and fails leaves none (the doc now says so).
  `tessl_qwen35_load` writes `*out` only after its final synchronize
  succeeds, and frees the model otherwise: `*out` is null whenever the status
  is not `TESSL_OK`.
- **One SiLU, overflow-free.** `nn::mlp_silu` and the GEMM epilogue's
  `Activation::Silu` computed the textbook `x / (1 + exp(-x))`, which divides
  by `exp(-x) = inf` below -88 under the fast math every kernel compiles with.
  Both now call `tessl_silu`, the form the Qwen3.5 kernels already used, which
  keeps the denominator in [1, 2]. `kernels/gelu.h` becomes `kernels/act.h`
  and holds the crate's GELU, sigmoid and SiLU; `qwen35_sigmoid` and
  `qwen35_silu` are now `tessl_sigmoid` and `tessl_silu` there, and
  `qwen35_act.h` keeps softplus, the log decay and the RoPE angle. The new
  tests `mlp_silu_stays_finite_at_extreme_gates` and
  `silu_epilogue_stays_finite_past_exp_overflow` passed on the old kernels
  too: on the M5 Pro, `mlp_silu` returned exactly -0 at the probed gates from
  -88 to -100 and the f32 epilogue at accumulators -88.5 to -100 (reciprocal
  or IEEE lowering, not a NaN), so they stay as guards. The tests' f64 sigmoid and SiLU are one oracle in
  `tests/common`, also sign-branched.
- **The GEMM epilogue's refusal names the real precision API.** It named
  `PrecisionMode::Relaxed`, which does not exist (relaxed f32 is
  `GpuRuntime::set_relaxed_precision`), left out f16, and carried two
  14-space runs from a multi-line literal without `\` continuations; the
  short-bias refusal had one too. `qwen35::project_residual`'s doc had the
  same wording. `epilogue_refusals_name_the_real_precision_api_in_one_line`
  (`tests/gemm_epilogue.rs`) pins both messages. The headers of
  `kernels/matmul_tensorops.metal` and the tune file described the
  pre-cooperative design (64x32 tiles, C zeroed once, a 256x64 NN arm) and
  now describe the shipped kernels.
- **The GEMM tuning rigs are interleaved A/Bs and refuse to run without
  their kernels.** `bench_gemm_tnnt_tune` timed production for all its
  iterations before any variant, though `docs/benchmarking.md` called it
  interleaved, and `bench_gemm_tile_tune` did the same; both printed
  `skip(pipe)` and exited 0 on a metallib without the tune kernels. Both now
  time production and every variant as the arms of one A/B, round by round
  with the order reversed every other round (`BENCH_ROUNDS`, `BENCH_ITERS`),
  and report the median of per-round ratios with their range. A baseline that
  moves more than 10% across a shape's rounds prints `EXCEEDS 10%` and makes
  the run exit 3; a missing variant exits 2 before anything is timed
  (`tests/tune_rigs_refuse_missing_variants.rs`). `bench_gemm_tnnt_tune`'s
  accumulate lanes now time production through `GemmOperands::Bf16`.
- **Async encode without a waiting sync poisoned the runtime.** Every scalar
  bind takes a slot in the 16 MiB constant arena, and only a waiting commit
  rewound it, so a run that never called `synchronize` failed a bind with
  "constant arena exhausted" after ~1M single-scalar dispatches and stayed
  poisoned. A binder scope now first checks for 4 MiB of headroom and, short
  of it, commits the open command buffer, waits for every one in flight and
  rewinds the arena. The 100k-dispatch cap also counted across non-waiting
  commits, so past 100k every dispatch committed its own command buffer; the
  count, like the `TESSL_MID_COMMIT` threshold, is now per command buffer.
- **Dropped buffers waited for a synchronize to be released.** Cold
  temporaries, retired Hot buffers, external wraps and `mtl_tensor`
  allocations queued on drop and left the queue only at a waiting commit, so
  a run that commits without waiting kept every temporary it dropped
  allocated and resident. Each release now records the shared-event value of
  the commit that closes the command buffer open when it dropped, and leaves
  the queue once the GPU has passed it: at every commit, waited or not. A
  final drop that times out now also anchors the external and `mtl_tensor`
  queues, which it had left to free under in-flight work.
- **A poisoned runtime said only "poisoned".** Only the call that poisoned the
  runtime saw the cause; every later refusal, including a command-buffer
  fault reported on Metal's feedback thread, read "runtime is poisoned after
  encode/submit failure". The refusal now names the first cause. The constant
  arena's "exhausted or empty payload" error is split in two, and the
  exhausted one gives the payload size, offset and arena size.
- **`TESSL_MID_COMMIT` split synchronous scopes.** With async encode off,
  each scope commits and waits, but the mid-commit threshold could first
  commit it without waiting; the waiting commit then had no open command
  buffer to stamp, so `take_metal4_stamps` returned nothing. Synchronous
  scopes no longer mid-commit.
- **Qwen3.5 configs and sessions that failed late now fail first.**
  `Qwen35Config` refused only a zero `linear_conv_kernel_dim`, so a width
  the conv kernels lack (1, or past 8) loaded and then failed part way
  through the first forward; it is refused with the config. A `Staged`
  prefill or `Decode` session continued across `write_parameters` or
  `adamw_step` on state the old weights made; both now refuse, as a
  pending training step does. `prefill` allocated whatever `max_new` asked
  (a 16 GiB session went through on a 1 MiB budget); it is checked against
  the recommended working set before any GPU work. Attention heads that do
  not group (query heads not a multiple of the KV heads) are refused with the
  config instead of by attention mid-forward; `forward_rows` checks its rows
  before running the forward (a bad row used to cost the whole forward's
  dispatches); `score_answers` checks its answer count before allocating;
  and `prefill` allocates the session's buffers before the prefill runs.
- **A freeze-binds ICB refused every captured tape.** `bind_tensor` and
  `bind_gpu_buf_offset` finished in the raw-buffer bind, which marks the
  capture incomplete, before recording the owned bind. Every captured
  command then carried `incomplete_binds > 0`, so
  `DecodeIcb::from_commands_ex(.., true)` (and `TESSL_ICB_FREEZE_BINDS`)
  rejected any real tape. Owned binds are recorded without that mark; the
  raw-buffer `Binder::bind_buf` is now test-only. Covered by
  `owned_binds_capture_complete_and_freeze`.
- **`gemm_i8_dequant` bound its exact accumulation with the wrong product.**
  It took 127 × 127 as the largest int8 product, but (−128) × (−128) = 16384
  is larger, so at `k = 131072` an all-(−128) sum wrapped to −2³¹ without an
  error. `MAX_K_EXACT` is now `i32::MAX / 16384 = 131071`. One
  `gemm::require_i32_extent` is the signed-32-bit extent check for
  `validate_gemm`, `gemm_batched` and the raw-buffer `gemm_i8_dequant`, so an
  int8 operand past `i32::MAX` elements is refused before encoding.
- **A busy or poisoned runtime panicked a trainer.** The training paths wrote
  and read host mappings through panicking helpers. `GpuBuffer` gains
  `try_read_f32` and `try_zero` (`read_f32` and `zero` delegate to them), and
  `attn_train`, `cross_entropy`, `qwen35_bwd` and `qwen35_train` use them:
  every training entry point now returns an `Err` naming the state
  (`tests/qwen35_train.rs`).
- **RoPE drifted from transformers with position.** `qwen35_attn_qk_norm_rope`
  (Qwen3.5 and EmbedGemma2), its backward and `rms_qkv_rope` computed
  `inv_freq` on the device. That is about an ulp off torch's host-computed
  table at most pairs, and the angle `pos * inv_freq` multiplies it by the
  position. `rms_qkv_rope` also used fast `pow`/`cos`/`sin`.
  - Before: q/k were `1.2e-4` relative off transformers at position 1600,
    and `rms_qkv_rope` was `4e-4` off at 8000 and `1.5e-3` at 32000.
    EmbedGemma2's 1658-token trace reached `1.8e-4`.
  - Now: every RoPE kernel reads `nn::rope_inv_freq` from the host, bound as
    constant data in the dispatch's arena (no allocation, no host mapping,
    captured as an immediate for ICB replay), and `rms_qkv_rope` uses
    `precise::` cos/sin. The 1658-token trace is within `4.9e-5`, and
    `rms_qkv_rope` holds `3e-5` at position 32000.
  - Tests that fail on the old kernels: `qkv_columns_norm_rope_far_positions`
    (d=256 and d=512 at position 8000 against transformers, with a derived
    bound) and `rms_qkv_rope_holds_at_far_positions`.
  - Kernel ABI: `qwen35_attn_qk_norm_rope{,_posbuf}` and
    `qwen35_attn_qk_norm_rope_bwd_f32` take the table at buffer 18 where
    `theta` was. `rms_qkv_rope*` keep `theta` at 12 (gemma-metal binds it
    from its scalar pool) and take the table at 18, which
    `rms_qkv_rope_with_scalars` binds itself after the callback.
- **Model weights were allocated as mid-step temporaries.** `EmbedGemma2Model`
  and `Qwen35Model` loaded their weights `BufferKind::Cold`, so dropping a
  model parked them in the pool's freelist instead of releasing them (2.2 MB
  and 1.2 MB of the tiny test checkpoints stayed allocated). They are `Hot`
  now, as the runtime documents for weights; activations stay `Cold`.
- **The CPU kernel emulator did not build** after the `-FLT_MAX` change:
  `tools/msl_emu/metal_stdlib` never included `<cfloat>`, whose limits MSL
  predefines. The emulator's Qwen3.5 RoPE checks now pass transformers' own
  `inv_freq` and hold q/k to `2e-5` at position 30000 (was `2e-4`).
- **EmbedGemma2 read the wrong K/V rows at batch > 1 when layers differ in
  `kv_heads * head_dim`.** One K/V buffer was sized for the widest layer. The
  writer derives its per-sequence stride from the buffer's size, while the
  attention and value norm read at stride `seq`, so in narrower layers every
  sequence after the first was misread (cosine `0.27` in the tiny test).
  Each width now gets exactly sized buffers. The released checkpoint has
  width 512 in every layer and was not affected. A missing value-norm buffer
  is now an error instead of a library `.expect`.
- **Four public boundaries now refuse inputs the rest of the API already
  refused.**
  - `nn::row_sum_f32` / `nn::row_max_f32` refuse `out` aliasing `x`.
    Threadgroup `r` writes `out[r]`, which is an element of a row another
    threadgroup may still be reading. `softmax_rows_f32` still runs in place.
  - `Qwen35Model::train_backward_into` refuses a `PendingStep` whose forward
    preceded `write_parameters` or `adamw_step`. The model now carries a
    parameter generation that every weight writer bumps. Before this, the
    backward rebuilt each layer with the new weights and returned gradients of
    a function the forward never evaluated.
  - Tensors from another `GpuRuntime` are refused before anything runs. This
    covers gradient banks (`train_step_into`, `train_backward_into`), gradients
    and AdamW moments (`adamw_step`, `grad_sq_norm`, `read_gradients`,
    `write_adamw_moment`), `write_parameters` / `read_parameters` tensors,
    `PendingStep::hidden`'s `out`, the backward's `dh`, and
    `qwen35_bwd::scatter_add_rows`. Some of these used to be refused only once
    GPU work was encoded, or after the pending step had been consumed.
  - `quant-prep`: `mtl_tensor::alloc_device_tensor` registers the tensor for
    residency. Dropping a `GpuTensor` now holds its `MTLTensor` until submitted
    work completes, since Metal 4 command buffers do not retain bound
    resources; only then does it leave the residency set.
    `bind_mtl_tensor` refuses a tensor from another runtime. New probe kernels
    (`kernels/mtl_tensor.metal`) test a dispatch that writes and then reads a
    device-owned tensor.
- **No kernel relies on an infinity under fast math.** `build.rs` compiles
  every kernel with `-fmetal-math-mode=fast`, whose IR marks float compares
  `fast` (including `ninf`), so `m == -INFINITY` was a compare the GPU
  compiler may fold. Nothing measured had misbehaved. The attention kernels
  (`flash_attn_rows`, `flash_attn_decode`, the SWA h128/h256 and global h512
  tiles, and Qwen3.5's tiled, shared-prefix rows and shared-prefix decode
  kernels) now seed running maxima with `-FLT_MAX`, use `l > 0` as the "has
  seen a key" flag, and zero a masked key's weight by its mask. A split-K
  chunk with no key is marked by `l = 0`. Every existing attention parity
  test still passes; performance was not re-measured.
  The max reductions (`row_max_f32`, `softmax_rows_f32`, `reduce_row_max`)
  seed from the row's own data, so a row of only `-inf` still reports `-inf`.
  - **Contract change, training attention:** a query row with no key now
    saves `lse = f32::MAX` (was `+inf`). The backward gives such a row no
    probability by testing for it rather than by `exp` underflowing.
  - **Contract change, argmax:** `argmax_f32`, `softcap_sample` and
    `softcap_argmax_one_pass` mark a lane with no finite logit by the index
    `0xFFFFFFFF` (which the host already refuses), not by a NaN or `-inf`
    value. A group with no finite logit writes `out_val = -FLT_MAX` (was NaN
    or `-inf`).
  - **Tests:** `no_kernel_spells_an_infinity_in_code` (fails on any
    `INFINITY` left in kernel code),
    `rows_with_no_key_beside_rows_with_keys_are_zero_in_every_kernel`,
    `training_forward_saves_a_finite_lse_for_a_row_with_no_key`,
    `training_backward_gives_a_row_saved_as_empty_no_gradient`,
    `row_max_and_softmax_of_masked_rows_never_invent_a_value`,
    `argmax_padding_lanes_never_beat_the_lowest_finite_logit`.

- **Persistent buffers are no longer rounded to a power of two.** The buffer
  pool bucketed every request to its next power of two, Hot weights,
  gradient banks and AdamW moments included, though they never return to the
  freelist: each f32 table of Qwen3.5-2B took 10.20 GB for 7.53 GB of values.
  Hot buffers and Cold ones over 1 MiB are now made at their size rounded to
  Metal's 16 KiB allocation granule (small temporaries keep power-of-two
  buckets). The 2B's weights, bank and moments went from 40.83 to 30.13 GB
  allocated, and ojas-qwen35's gradient read-back from 50.20 to 37.94 GB
  against a 51.54 GB working set (`docs/qwen35.md`). A recycled large Cold
  buffer now serves only a request of its own rounded size.
- **A Qwen3.5 training step that cannot fit is refused before it runs.**
  `Qwen35Model::train_forward` (so `train_step` and `train_step_into`)
  returns an error, before any GPU work and without poisoning the runtime,
  when `Qwen35Model::train_step_bytes` plus the device's current allocation
  exceeds its recommended working set. Past it, Metal pages the resident set
  and command buffers time out, or the system runs out of memory. The bound
  was never below the measured peak on the 2B at T = 128, 2048 and 8192, and
  exceeded it by at most the freelist cap plus 30%.
- **Fused GEMM + GELU no longer clips at 20.** `Activation::GeluTanh` in
  `gemm_epilogue` ran a private copy of the GELU that multiplied by the
  clamped input, so every pre-activation above 20 came out as exactly 20
  instead of ~x. The epilogue now calls `tessl_gelu_pytorch_tanh` from
  `kernels/gelu.h`, the GELU `mlp_gelu_tanh` and the q4 gate/up GEMVs already
  use, and the copy is gone. The test reference in `tests/gemm_epilogue.rs` had
  the same defect, which is why its GELU case stayed green; the three CPU GELU
  references in `tests/` are now one `common::gelu_pytorch_tanh`, and
  `gelu_epilogue_tracks_x_past_the_cubic_clamp` drives values past the clamp
  through the f32 and both bf16 tile geometries.
- **Accumulate GEMM tests budget for the previous C.** `with_previous` in
  `tests/gemm_flag_paths.rs` added `C0` to the expected value but not to the
  magnitude the f32 error budget scales with. A small-K accumulate onto a
  large `C0` was therefore held to the rounding of `a·b` alone, and failed on
  the final f32 add (1 ulp at K = 3).


- `cross_entropy_rows` adds every vocabulary chunk after the first into
  `dh`'s own view. Those chunks' partial `dh` was added at the start of
  `dh`'s buffer, so in a `dh` view that does not begin at byte 0 they landed
  shifted back by the view's offset, over whatever precedes it.
- `Qwen35Model::adamw_step` at a step count of `u64::MAX` returns an error
  and leaves the count there. Without overflow checks (release builds)
  `step + 1` wrapped to 0, which zeroes the bias correction and sends the
  update to infinity.
- Binder-nop replay suppression belongs to the thread that armed it. The
  flag was process-global, so one model's decode-ICB replay made an
  unrelated model on another thread skip every `with_binder` encode and
  report success over stale device memory.
- `tools/msl_emu` runs under ThreadSanitizer on macOS: TSan does not see
  libc++'s `std::barrier` as synchronisation, so every barrier-separated write
  pair was reported as a race. The shim's barriers are now its own, on acq_rel
  atomics, and `build.sh` holds them to a probe under TSan: clean with each
  barrier, a reported race without it.
- `tools/msl_emu` builds on macOS again: since the fast-math `exp`/`log`
  became the shim's own functions, libc++'s global float overloads made every
  unqualified kernel call ambiguous (Linux's libstdc++ declares none). Kernels
  now compile inside a namespace that resolves them to metal's, and the shim
  declares the global overloads on every host so Linux CI sees this class of
  break too.
- Host mappings of a GPU-private buffer are refused instead of building a
  slice over the null `contents()`.
- Two wraps of one `MTLBuffer` are one allocation: overlap checks compare the
  buffer, not the wrapper, and residency is counted per buffer, so dropping
  one wrap no longer evicts it from under another.
- A `ParamsBuffer` slot is no longer rewritten under an encoded dispatch that
  still reads it.
- `npy`: only versions 1-3, a bounded UTF-8 header, `fortran_order` exactly
  `True`/`False`, checked element counts, and a payload that must be exactly
  the rest of the file (checked before allocating).
- MLX Q4 tiled banks (`Interleaved4`, the blocked layout) must hold their
  padded last tile; a bank sized for the unpadded rows read past its end.
- `qwen35::pack_linear_weights` checks its width sum and skips empty output.
- `attn_qk_norm_rope` rejects a zero-capacity cache and cache positions past
  `u32`.
- `GdnWorkspace::check` computes its sizes with checked products.
- `Tensor::from_mtl_buffer` checks bounds before registering residency, so a
  rejected wrap leaves no residency entry.
- `execute_icb` requires the ICB to be in the residency set, as
  `optimize_icb` already did.

- **Qwen3.5 layer kernels (`tessl::qwen35`)**, the Metal replacement for the
  pure-torch GDN fallback transformers runs on MPS. See `docs/qwen35.md`.
  - Chunked gated delta rule as two dispatches: `qwen35_gdn_chunk_prep`
    (parallel over chunks: l2norm, Qwen3.5's gates, both 64x64 products, and
    the triangular solve in threadgroup memory) and `qwen35_gdn_chunk_scan`
    (the sequential state pass, on `simdgroup_matrix` tiles).
  - `qwen35_gdn_recurrent`, a token-by-token decode that reads a shared
    snapshot state without writing it, so one prefix serves many questions.
  - `qwen35_conv1d_silu` (prefill and decode, snapshot-capable),
    `qwen35_gated_rms_norm_{f32,bf16}`, `qwen35_attn_qk_norm_rope` (zero-centred
    norm, transformers' partial RoPE, K/V cache store),
    `qwen35_attn_gate_{f32,bf16}`, and `qwen35_score_rows_{f32,bf16}` (final
    norm + LM head for the answer tokens at the slot rows only).
  - Fused-projection helpers: weight packing, column layouts that every kernel
    reads in place, and `project_residual` (the residual add as a GEMM
    epilogue).
- **Prefill attention on the TensorOps matrix units** (`qwen35_attn_tiled_h256_*`,
  `qwen35::attn_prefill`): `flash_attn_rows`' contract at head_dim 256 and
  `window = 0`, with `Q·Kᵀ` and `P·V` on `matmul2d` in exact f32 over
  query-by-key blocks (four geometries, `qwen35::AttnTile`,
  `attn_prefill_with_tile`) and an f32 online softmax between them. It skips key blocks
  above the diagonal. `bench_qwen35_layers` now times it, and
  `--attn-tile=LABEL` / `--attn-rows` select another tile or the scalar
  kernel.
- **`qwen35_gdn_chunk_scan_bv16`**, the chunked GDN scan in 16-column value
  slices, selected by `GdnWorkspace::with_scan_slice(GdnScanSlice::Cols16)`.
  It launches twice the threadgroups and is bit-identical to the 32-column
  scan. `probe_gdn_scan` found the 32-column scan underfilling the GPU at
  batch 1; `bench_qwen35_layers --gdn-scan16` selects it. It was opt-in when
  added; `GdnScanSlice` now defaults to `Cols16` (see the default-switch entry
  under Changed), while the decode recurrence's `GDN_RECURRENT_SLICE` stays
  `Cols32`.
- **`qwen35_swiglu_{f32,bf16}` / `qwen35::swiglu`**: Qwen3.5's MLP activation,
  `silu(gate) * up`, from f32 column windows straight to bf16 for the down
  GEMM, replacing `nn::mlp_silu` plus a cast pass in `bench_qwen35_layers`
  (`--mlp-unfused` times the old path).
- `bench_qwen35_layers` times the GDN chunked rule's two dispatches apart
  (`gdn chunk prep`, `gdn chunk scan`) through the doc-hidden
  `qwen35::gdn_chunk_phase`. Run in order, they are bit-identical to
  `gdn_chunk_forward`.
- **Shared-prefix attention for Qwen3.5** (`qwen35_attn_prefix_rows`,
  `qwen35::attn_prefix_rows`). Many questions over one prefilled context now
  share the 6 full-attention layers' KV prefix at batch stride 0, as the GDN
  and conv state already did through `StateIn::Snapshot`. Each row keeps its
  own suffix cache. The result is bit-identical to `nn::flash_attn_rows` over
  a per-row copy of the prefix. `qwen35_attn_prefix_decode_{partial,reduce}`
  (`qwen35::attn_prefix_decode`) are the split-KV single-query form, and are
  bit-identical to `nn::flash_attn_decode` the same way.
  `qwen35::attn_qk_norm_rope_suffix_posbuf` is the suffix writer with the
  position in a device buffer, for ICB replay.
  Ragged continuations take per-row lengths and positions:
  `attn_prefix_rows_varlen`, `attn_prefix_decode_varlen` and
  `attn_qk_norm_rope_suffix_rows`. Each row is bit-identical to that row
  alone. The shared-prefix kernels gained a `row_stride` slot, and both
  `qwen35_attn_qk_norm_rope` kernels gained `pos_stride` (buffer 21).
- **Ragged rows for the stateful layers:** `conv1d_silu_varlen`,
  `gdn_chunk_forward_varlen` and `gdn_recurrent_varlen` take per-row
  `seq_lens`, so the questions in a batch can have different lengths. Each
  row, final state included, is bit-identical to that row run alone. The
  kernels gained a `seq_lens` slot (flag bit 4; `use_lens` in the prep).
- **`qwen35_embed_rows_bf16` (`qwen35::embed_rows`)**, the embedding gather
  from the bf16 vocabulary table on the device. It is exact, and a bad id
  gives a NaN row. A forward no longer needs a host gather and a
  commit-and-wait before its first layer. `qwen35::attn_qk_norm_rope_suffix` caches a
  suffix at prefix-relative slots with absolute RoPE positions, through a new
  `slot_base` argument (buffer 20) to both `qwen35_attn_qk_norm_rope` kernels.
- **`tools/msl_emu`**, a CPU emulator that runs the kernel sources as C++ with
  real threadgroup barriers and simdgroup collectives, and a driver that checks
  them against transformers' own Qwen3.5 code.
- Hardening pass over the Qwen3.5 kernels after an independent audit:
  - **Numerics:** overflow-free sigmoid/SiLU. A `log1p`-accurate softplus (fast
    `log(1+e)` was 1-60% off where `a + dt_bias` routinely lands). Scoring keeps
    invalid answers out of the softmax and stores NaN as integer bits, which fast
    math cannot fold away.
  - **Scan and prep tiles:** the `const` diagonal matmul (a likely compile error)
    is gone. Threadgroup strides are padded against bank conflicts, the per-row
    decays are cached, and redundant tile loads are removed.
  - **Kernel structure:** thread counts are named constants with `static_assert`s
    on the lane mappings, and the conv runs with 256-thread groups.
  - **Host fixes:** the in-place GDN state is now checked against the inputs too.
    `score_answer_rows` rejects `rows == 0` / `vocab == 0`, which read out of
    bounds before. `AttnTargets` derives the KV capacity from the caches instead
    of taking one that could disagree with flash attention. Layouts get
    validating constructors and `GdnProjLayout::dims`. `seq = 0` with a
    `state_out` now copies the state through. Host arithmetic is checked
    throughout.
  - **Emulator:** threadgroup-order invariance, ASan/TSan builds with exactly
    sized allocations, fast-math ulp noise, a host-contract check of binds and
    constants, and a `kernel-emulator` CI job.
- `tools/msl_emu/check_qwen35_model.py`: a whole random `Qwen3_5ForCausalLM`
  through the kernels and tessl's own `flash_attn_rows` (emulated), against
  the model's logits, for prefill, cached decode and a shared snapshot. The
  Mac suite gains the same attention seam:
  `attention_layer_through_flash_attn_rows`.
- Scoring re-derives validity from the indices instead of storing flags. This
  restores the full `MAX_ANSWERS = 4096` inside 32 KB of threadgroup memory,
  and a unit test pins it. `attn_qk_norm_rope` accepts `batch = 0`. Softplus
  uses `precise::exp`.
- `qwen35_attn_qk_norm_rope_posbuf` / `attn_qk_norm_rope_posbuf`: the RoPE
  position comes from a device buffer, so a decode loop replayed from an ICB,
  which freezes scalar binds, advances correctly. Both variants form the
  position in 64 bits before the capacity check. In 32 bits, an offset near
  `u32::MAX` wrapped to slot 0 and passed the check.
- `tools/msl_emu/dialect_lint.py` (in CI): every MSL construct the Qwen3.5
  kernels use that no compiling tessl kernel uses must be on a reviewed list.
- Emulator cases at Qwen3.5's real head counts and at T=4096.
- `.github/workflows/metal-compile.yml`: Apple's Metal compiler on GitHub-hosted
  macOS, never the self-hosted runner. It compiles the Qwen3.5 sources under
  `-std=metal4.0 -Wall -Werror`, links them, checks every entry point is
  exported, and builds everything with no `metal3.2` fallback. Its first run
  found one unused constant; every other line compiled clean.
- `tests/qwen35_kernels.rs` with transformers-generated goldens in
  `tests/fixtures/qwen35/` (`scripts/gen_qwen35_fixtures.py`). The Qwen3.5
  sources join the widened-index-arithmetic inspection in
  `tests/shader_index_arithmetic.rs`.

## [0.2.0] — 2026-09-18

Fail-closed encode / attention / quantized paths, Tensor metadata hygiene, and
SharedEvent interop with sparsl. The minor version moves because public
`Tensor.shape` / `Tensor.byte_offset` fields become `pub(crate)` (use
`shape()` / `byte_offset()`), `BufferKind` gains `External` and is marked
`#[non_exhaustive]`, and several previously-accepted shapes / aliases are now
refused with errors.

### Added

- **Flash attention decode and rows kernels**, with host routing that picks the
  right specialization and refuses aliased / dimensionally inconsistent outputs.
- **`BufferKind::External`** and `Tensor::from_mtl_buffer` for zero-copy
  MTLBuffer handoff (BINN / sparsl SharedEvent paths).
- **Dispatch geometry validation** and traced kernel names so a bad grid fails
  closed instead of writing partial tiles.
- **Q4 / Q8 shape-domain checks** and widened index arithmetic in the Metal
  kernels that previously truncated under large extents.
- **Release tooling:** `scripts/ci_local.sh`, `scripts/check_release_ready.sh`,
  and a tag-triggered `.github/workflows/release.yml` that publishes to
  crates.io and creates the GitHub Release from this changelog.

### Changed

- **`Tensor.shape` / `Tensor.byte_offset` are no longer public fields.** Mutating
  them by assignment could desynchronize the logical view from the buffer;
  construct via allocators / `try_view` and read via accessors.
- **`BufferKind` is `#[non_exhaustive]`** so exhaustive matches stop breaking on
  every new kind.
- **Busy bump-reset and poisoned `alloc_temp` fail closed** instead of
  fallthrough.
- **Hazard tracking** (`hazard_pending`) so decode / attention writers cannot
  race a subsequent reader without a barrier.

### Performance (Apple M5 Pro, 2026-09-18)

Sequential A/B vs clean `v0.1.4` HEAD, `bench_gemm_sweep` with
`BENCH_WARMUP=10` / `BENCH_ITERS=30`, shapes `512³,1024³,2048³`, cooled
cur→prev pass:

| shape / backend | prev (ms) | 0.2.0 (ms) | Δ |
|---|---:|---:|---:|
| 1024³ tensorops-bf16 | 0.217 | 0.196 | **−10%** |
| 1024³ tensorops-f32 | 0.508 | 0.514 | +1% |
| 2048³ tensorops-f32 | 3.883 | 3.859 | ≈0% |
| 2048³ tensorops-bf16 | 0.876 | 0.888 | +1% |

Small square shapes sit near the ~0.25 ms dispatch floor and are not quoted as
wins. Host load was elevated during measurement; treat ±~10% as noise.

## [0.1.4] — 2026-09-14

Manifest metadata only. No code, API or behaviour change; the compiled crate is
identical to 0.1.3.

### Added

- **`homepage` in `Cargo.toml`.** 0.1.3 put a website badge in the README, which
  only reaches someone already reading the rendered README. crates.io shows a
  Homepage link in the crate page sidebar when the manifest declares one, and
  this crate declared none — so the showcase was missing from the one place a
  reader scanning the page looks for it. `sparsl` gained the same field in the
  same change.

## [0.1.3] — 2026-09-14

Documentation and repository hygiene only. No kernel, API or behaviour change,
so the compiled crate is identical to 0.1.2. It ships because the README is
rendered on the crate page, and the links below are the reason to publish it
there rather than only on GitHub.

### Added

- **`documentation` in `Cargo.toml`, and the published docs linked from the
  README.** The crate page fell back to docs.rs implicitly and the README named
  neither, so the one surface a GitHub reader lands on had no path to the API
  reference. The README now carries crates.io, docs.rs, CI and license badges,
  a nav line to docs.rs and the three `docs/` deep dives, and an API reference
  row at the top of the Documentation table.
- **The interactive benchmark showcase at
  [tessl.vbcr.dev](https://tessl.vbcr.dev/) linked from the README**, as a badge
  and as the first entry in the nav line. The measurements the README quotes in
  prose are explorable there; nothing else pointed to it.

### Fixed

- The README Status row said `0.1.0` while crates.io served `0.1.2`. Corrected,
  and linked to the crate page so the number has somewhere to be checked
  against.

## [0.1.2] — 2026-09-01

Documentation only.

### Fixed

- **Internal monorepo paths and audit numbering leaked into the published
  docs.** Eight doc comments referenced `arch_02_value_resid/metal-native`,
  `arch_02`, and "Audit 4 P0 / P1 / 4 6 / 7" — a directory layout and an
  internal review series that mean nothing to a reader arriving from crates.io,
  and cannot be looked up. Each cited note is replaced by the reasoning it stood
  for, so the *why* survives without the unresolvable citation: cold buffers
  recycle only after the command buffer completes because releasing earlier
  hands memory back while the GPU may still read it; one encoder is packed
  across dispatches because opening one per dispatch costs setup on every op.
  References to `gemma-metal` are kept, since that is a real crate and explains
  where the promoted kernels came from.

## [0.1.1] — 2026-09-01

One build-script fix, and documentation. No kernel or API changes.

### Fixed

- **A `DOCS_RS=1` build left the crate permanently broken until `cargo clean`.**
  That branch bakes `TESSL_METALLIB=""` through `cargo:rustc-env`, and
  `build.rs` never declared `rerun-if-env-changed=DOCS_RS` — so cargo had no
  reason to re-run it when the variable disappeared, and every later build kept
  the empty path. The crate still compiled; every `GpuRuntime::new()` then failed
  with `metallib missing at ` and an empty path, thousands of lines from the
  cause. Found by running the docs.rs simulation immediately before the test
  suite, which turned 88 passing lib tests into 62 failures on a tree whose only
  other change was doc comments.

- **The README's headline benchmark claim was wrong.** It read 1.11x against
  PyTorch MPS on bf16 with a worst shape of 1.01x, which reads as "never loses".
  Re-measured with the same paired harness on the same machine: **1.03x, losing
  on 4 of the 8 shapes**, worst 0.86x. The tf32 lane (2.11x, wins every shape)
  and the MLX comparison (2.55x) hold up. The table now carries a "shapes below
  1.0" column, because a geomean above 1.0 with half the ladder underneath it is
  not the same result as a clean win.
- **The published f32 and tf32 peak throughputs were not reproducible.** 10,897
  and 18,040 GFLOP/s; the crate's own committed sweep records 6,606 for f32 and
  a fresh run gives 6,431. Replaced with measured values rather than averaged.
- Two remaining references to `bench_gemm_coop_ab`, a binary that does not
  exist, in the benchmarking doc's tool table and prose. The paired A/B lane is
  `bench_gemm_tnnt_tune`.
- Stale counts in `docs/verification.md` (199 tests across 14 files; it is 228
  across 18) and in the README's known-gaps table.

### Added

- **A docs.rs landing page written for someone outside the monorepo.** Platform
  constraints first, since this crate does not build on Linux or Intel Macs at
  all, and the Metal Toolchain is a separately downloaded Xcode component rather
  than part of Xcode. Adds two `no_run` quickstarts, a module map, the feature
  table, the measured cross-runtime results, and the encode-model note that
  `async_encode` is off by default and worth roughly 49x on a small kernel.
- Contributor notes covering `--test-threads=1`, why `build.rs` tracks every
  `.metal` and `.h` individually, that a name check is not a correctness test,
  seeding outputs with a sentinel, and testing every arm of a kernel family.
- Real module docs for `npy`, `ops` and `infer_trace`, which had one line each.

### Fixed

- **The README's bf16-vs-MPS claim was wrong, and it was the headline.** It read
  1.11x with a worst shape of 1.01x — "never loses". Re-measured with the same
  harness against torch 2.13 MPS on the same machine: **1.03x, losing on 4 of the
  8 shapes**, worst 0.86x at 8192x3072x768. The tf32 lane (2.11x, wins every
  shape) and the MLX comparison (2.55x) hold up; bf16 against Apple's own tuned
  GEMM is parity, and the table says so now.
- **The published f32 and tf32 peak throughputs were not reproducible.** 10,897
  and 18,040 GFLOP/s; the crate's own committed sweep in `bench/results/` records
  6,606 for f32 and a fresh run gives 6,431 — two independent sources agreeing
  against the README. Replaced with measured values (26,642 bf16 / 16,293 tf32 /
  6,431 f32) rather than averaged.
- **Two more references to a binary that does not exist.** `docs/benchmarking.md`
  named `bench_gemm_coop_ab` in its tool table and in the Pitfall 1 prose. An
  earlier pass corrected only the *Reproducing* block at the end of that file, so
  its note could truthfully say the phantom was fixed while two live references
  survived above it. The paired A/B lane is `bench_gemm_tnnt_tune`. Every command
  named in the README and docs is now checked to exist.

## [0.1.0] — 2026-08-31

First published release. The crate has not been on crates.io before, so
everything below ships in it. The entries were written as the work landed and
are kept in that form rather than reflowed, because each records why it was
done — several of them are defects found by giving a kernel its first numeric
test, and the reasoning is the useful part.

### Added

- **`tests/q4_interleaved.rs`** — numeric coverage for the seven `_i4`
  (Interleaved4) MLX Q4 kernels, which had only a name check. They read a
  different weight packing from their row-major twins, and `Q4MlxBank` carries
  no layout tag, so the wrong packing dispatches and returns numbers — the same
  hazard `gemv_q4_mlx_blocked` turned out to be. The packer is transcribed from
  `gemv_q4_mlx_simd_i4`'s indexing: weights at
  `((tile * packs + pack2) * 4 + r) * 8` bytes with `tile = row / 4`, scale/bias
  at `(tile * groups_per_row + g) * 4 + r`; the nibble order inside each 8-byte
  group is unchanged, only the placement moves. Each test checks the `_i4`
  kernel against the dense f64 reference *and* against its row-major twin over
  the same logical weights.

  **All seven kernels were correct.** Verified the tests can fail: feeding the
  `_i4` path a row-major bank fails all six, and mutating the kernel's scale
  stride fails the one test that covers it.

- **The GEMM residual arm.** `gemm_q4_mlx_simd_add` and `_add_i4` are reached
  only by passing `Some(resid)`, and every other GEMM test passed `None`, so
  they were the last two promoted kernels with a name and no number. The
  residual is the full `m x rows`, matching the output — `resid[m * rows + row]`
  in the kernel — not a per-row vector broadcast across `m`, which is what the
  first draft of the test assumed and what the host validation caught.

  With this, all 44 promoted kernels have a numeric test.

### Fixed

- **The sliding-window attention kernels read uninitialised threadgroup memory
  for half of every query block.** `flash_attn_swa_h128` and `_h256` zeroed
  their `scores[BR * BC]` scratch with `if (lid < BR * BC)` — 64 entries — while
  the host dispatches **32 threads per threadgroup**. Entries 32..63 were never
  zeroed, and those are query rows 4..7 of every block, which then accumulated
  QK products into whatever a previous dispatch had left there. The result was
  plausible numbers rather than NaN, so nothing looked wrong. Zeroing is now
  strided, which is correct for any relation between `tptg` and `BR * BC`.
  `flash_attn_global_h512` was unaffected only because its `BR = BC = 4` gives
  16 entries, under the 32 threads; it is fixed the same way so the property
  does not depend on the tile constants.
- **A fully masked block produced NaN in the online softmax.** The FA-2 rescale
  computes `alpha = exp(m_i - m_new)`, and when a row had seen nothing yet and
  the current block was entirely masked for it, both were `-inf` — so `alpha`
  was `exp(NaN)`, which then propagated through `Oacc` and `l_i` and poisoned
  the row. This is reachable whenever the block-level skip admits a block on
  behalf of another row in the same `BR` tile, which the union window makes
  routine at small `window`. Guarded with `m_i == -inf ? 0`, which is also the
  right value for the ordinary first-real-block case.
- **`out_bf16` demanded an output buffer sized for f32.** `flash_attn_global_h512`
  validated `o` through `validate_attn_dims`, which always required
  `require::<f32>`, and then added a `u16` check on top. A caller who sized the
  buffer for bf16 — the whole point of the flag, documented as "half-width act
  scratch" — got "buffer holds 2560 elements, kernel reads/writes 5120". The
  output width now follows `out_bf16`.

### Added

- **`tests/attention.rs`** — six tests against an f64 reference transcribed from
  the kernels' own masking rule: prefill at both sliding-window head dims with
  ragged query and key tails, window bounding (including `window = 1`, which
  must reduce each row to its own V), decode with the device-side position
  offsets, a fully masked decode row that must be zeros rather than NaN, GQA
  head grouping across four `H:Hkv` ratios, and the global kernel's causal rule
  plus its bf16 output arm.
- **`tests/qkv_rope.rs`** — four tests for the fused RMSNorm to QKV to RoPE
  kernels: the constant-position variant against an f64 reference at full and
  partial `rotary_dim`, V normalised but never rotated, `PosBuffer` agreeing
  bit for bit with `PosConst`, and `PosBufferKvStore` writing the rotated K and
  V into the cache at a device offset without touching anything outside the
  slot. All four passed on the first run; these kernels were correct.

- **`gemv_q4_mlx` with `Q4MlxRowVariant::Tiled` left most of its output
  unwritten** — the same defect as `gemv_q4_tiled`, in the sibling family.
  `gemv_q4_mlx_tiled` indexes its output row by `threadgroup_position_in_grid`
  and needs one threadgroup per row; all three row variants were dispatched with
  the one-thread-per-row grid, so `Tiled` wrote `rows / 128` rows and returned
  no error. Found by giving the three variants one shared numeric test instead
  of testing `Standard` alone: 508 of 512 rows never written.

### Documented

- **`gemv_q4_mlx_blocked` requires a block-interleaved bank, and nothing said
  so.** It takes the same `Q4MlxBank` as its row-major siblings — a type with no
  layout tag — and returns wrong numbers rather than an error when given a
  row-major one. The kernel reads scale/bias and nibbles at
  `block * groups_per_row * 16 + group * 16 + row_in_block`. Measured at 64x256
  with `group_size` 64: 63 of 64 rows wrong row-major, 0 of 64 repacked. The two
  layouts coincide only when `groups_per_row == 1`, which is exactly the shape a
  small smoke test would pick. `tests/promoted_numeric.rs` carries a reference
  repacking.

### Added

- **`tests/promoted_numeric.rs`** — numeric coverage for promoted kernels that
  had only a name check in `promoted_kernels.rs`. That file asserts each of the
  44 entry points resolves from tessl's own metallib, which is a real gate on
  the move and not a correctness test: `gemv_q4_tiled` resolved, had adversarial
  coverage, and wrote 4 rows of 512. Eight tests now cover the three MLX Q4 row
  variants, the blocked GEMV, the fused K/V GEMV, `gemm_q4_mlx` against the GEMV
  row by row, `mlp_gelu_tanh_bf16` against its f32 sibling,
  `kv_store_timestep_pair`, `kv_ring_densify`'s rotation, and
  `embed_lookup_q4_mlx` including out-of-range token ids. Every one uses `rows`
  above the 128 the row kernels group by, because below that the competing grids
  coincide and a mismatch is invisible. Shared Q4 scaffolding moved to
  `tests/common/mod.rs`.

- **`gemv_q4` with `tiled = true` silently left most of its output unwritten.**
  `gemv_q4_tiled` indexes its output row by `threadgroup_position_in_grid` and
  so needs one threadgroup per row, but it was dispatched with
  `rows.div_ceil(128)` groups — the geometry the one-thread-per-row `gemv_q4`
  needs. It wrote the first `rows / 128` rows and left the rest of `y` holding
  whatever was there before, with no error returned. Measured at 512 rows: 4
  written, 508 untouched. The dynamic threadgroup allocation and the `cols`
  ceiling that bounds it are now applied only to the kernel that caches `x`;
  the tiled kernel declares its scratch statically and never did.
  **Anyone who passed `tiled: true` was getting wrong results**, and with the
  grid corrected that variant is slower than the default one at every shape
  measured — its apparent speed was the work it was skipping.
- **Nothing tested `gemv_q4_tiled` numerically.** `promoted_kernels.rs` checks
  the pipeline name exists and `nn_adversarial.rs` checks error paths, so a
  kernel writing 0.8% of its rows passed both. Added a test that asserts the two
  variants agree row for row and that neither leaves a seeded sentinel behind,
  at 512x256 and a ragged 300x128.

- **Q8 GEMV ran one thread per row, uncoalesced.** `gemv_q8` dispatched `rows`
  threads, so adjacent threads read addresses `cols` bytes apart and a
  simdgroup's 32 loads touched 32 cache lines. Now one simdgroup per four rows
  with lanes striding K and `char4` loads covering a full cache line per
  instruction, reusing the `simd_gemv_threadgroups` / `SIMD_TPTG` geometry the
  MLX Q4 GEMVs already used. Measured on an M5 Pro: **2.9x at 4096x4096**
  (274.5 -> 95.8 us, 68.9 -> 197.4 GB/s) and 1.25x at 11008x4096. The tall
  case is bound by something else and is recorded as such rather than
  explained away.
- **The Q8 GEMV test covered neither new path.** rows = 24 is a multiple of the
  8 rows a threadgroup owns and group = 16 is divisible by 4, so the ragged row
  tail and the scalar fallback never ran; breaking either left it green. The new
  case sweeps rows 13/37/100 and groups 15/32/64, and seeds `y` past `rows` with
  a sentinel to catch a tail threadgroup writing rows it does not own.

- **RMSNorm ran one thread per row.** All three kernels (`rms_norm_f32`,
  `rms_norm_bf16`, `rms_norm_residual_add_f32`) dispatched `rows` threads, each
  walking its row serially twice, which capped parallelism at the row count and
  ran the entire kernel on a single GPU thread at the decode shape. Now one
  threadgroup per row with the sum of squares reduced as a tree, the pattern
  `reduce.metal` already used. Measured on an M5 Pro: **16.7x at 1x4096**
  (404.3 -> 24.2 us), 10.1x at 512x4096, 2.5x at 2048x4096, with effective
  bandwidth going from 87 GB/s to 216-305 GB/s. RMSNorm runs twice per
  transformer layer on every token. The reduction reassociates, so results
  differ in the low bits from the previous serial sum.
- **The RMSNorm tests could not see the bug they now cover.** Every existing
  case used `dim` of 16 to 64 against a 1024-thread group, so each lane's
  strided loop ran once and deleting the loop entirely left all three green.
  Added `dim` of 4096, a ragged 3000, and 8192 across all three variants; both
  new tests kill that mutation, as do removals of `REDUCE_TREE` and of `eps`.
- **`build.rs` tracked only `.metal` for `rerun-if-changed`.** `REDUCE_TREE` now
  lives in `kernels/reduce_tree.h`, shared by `reduce.metal` and
  `rms_norm.metal`; without tracking headers an edit to the shared reduction
  would leave both dependents stale in the metallib while the suite reported a
  pass. Verified by mutating only the header and confirming a rebuild and five
  failures.

- **`docs/verification.md` documented a command that ran zero tests and
  reported `ok`.** It named a test `gemm_randomized_shape_fuzz` and two
  environment variables `GEMM_FUZZ_SEED` / `GEMM_FUZZ_CASES`; none of the three
  exist, so `cargo test ... gemm_randomized_shape_fuzz` matched nothing and
  printed `test result: ok. 0 passed; 89 filtered out`. The real entry points
  are `gemm_fuzz_quick`, `gemm_fuzz_deep` and `STRESS_SEED`. The same section
  claimed the fuzzer asserts per-kernel coverage at a 1% floor; no such
  assertion is implemented, and the retraction is now in the document rather
  than only in the README.
- **`docs/benchmarking.md` named a binary that does not exist.** The reproducing
  block invoked `bench_gemm_coop_ab` with a `BENCH_ROUNDS` variable. This crate
  builds four binaries and neither the target nor the variable is among them.
- `docs/verification.md` reported 68 unit tests; the suite is 199 across 89 lib
  and 110 integration tests.

### Added

- **Docs cover the promoted library.** `docs/architecture.md` previously
  described only GEMM — no mention of `nn`, the fused epilogue, the reductions,
  or the integer GEMM. It now documents the validate-before-encode boundary,
  the row-stride-0 bias broadcast, softmax's masked-row behaviour, and the
  int32 wrap bound at `k = 131072`.
- **A measured GEMM sweep in `docs/benchmarking.md`**: bf16 TensorOps at 26,642
  GFLOP/s on 4096³, 11.8x the portable simdgroup fallback and 4.2x exact f32,
  with the 512³ inversion explained by the dispatch floor rather than left as
  an anomaly.

- **Quantized int8 GEMM with fused dequantization**: `nn::gemm_i8_dequant`.
  `int8 x int8` accumulates into `int32` natively on TensorOps, and every
  product fits, so the integer result carries **no rounding at all** — tested by
  exact equality against an integer reference, not a tolerance. The per-column
  dequantization is applied in registers between the accumulate and the store.
  `k` above 131072 is refused, past which a full-range accumulation could wrap
  the int32 silently.
- **Corrected a misdiagnosis this crate had been repeating.** Quantized
  TensorOps was documented as blocked because `MTLTensorDataType::Int4` is
  unbound in objc2-metal 0.3. That binding gates host-created `MTLTensor`
  descriptors, and every kernel here builds tensors from raw device pointers
  instead, so it never applied. TensorOps supports
  `uint8_t/int8_t/uint4b_format/int4b_format` per the header's own diagnostic;
  what actually blocks Int4 is the shader-side tensor constructor for a
  sub-byte element type.
- **Strided batched GEMM**: `gemm_batched` with `BatchedGemm` and
  `BatchStrides`. The batch is the grid's second dimension, so it costs a
  pointer offset per threadgroup and nothing else — every element is
  bit-identical to the `gemm` that would have produced it. Per-operand strides,
  because a **zero** stride is the useful case: batched activations against one
  shared weight matrix needs no copies of B. Dimensions are passed explicitly
  rather than read from tensor shapes, since a rank-2 shape cannot distinguish
  `[batch * m, k]` from `[m, k]` and a broadcast B genuinely is `[k, n]`.
- **IEEE binary16 (`DType::F16`)**: `alloc_tensor_f16`, `cast_f32_to_f16` /
  `cast_f16_to_f32`, host `f32_to_f16_bits` / `f16_bits_to_f32` /
  `f32_slice_to_f16`, `GpuBuffer::write_f16_bits`, and f16 GEMM kernels
  (`matmul2d_tensorops_f16_f32`, the 64x64 variant, and the epilogue variant).
  f16 and bf16 are both two bytes and both accumulate in f32, but their bit
  layouts differ, so `nn_coop_kernel` now selects on a three-way `CoopElem`
  rather than a boolean, and `ensure_bf16` refuses f16 operands instead of
  converting them — that conversion would lose three mantissa bits and change
  the exponent range to buy a path the caller did not ask for.
- **Row-wise reductions**: `nn::softmax_rows_f32`, `nn::row_sum_f32`,
  `nn::row_max_f32`. One threadgroup per row, striding, so `cols` is unbounded.
  Softmax subtracts the row maximum before exponentiating — without it a single
  logit above about 88 overflows `exp` in f32 and takes the row to NaN, which is
  an ordinary attention input rather than a pathological one. A fully masked row
  (`-inf` everywhere) yields a uniform distribution rather than NaN.
- **`gemm_epilogue` — fused GEMM epilogue.**
  `C = activation(alpha * A@B + beta * C_prev + bias)` in one dispatch, applied
  while the accumulator is still in registers, so `C` is written once and read
  at most once. Measured 1.57x to 2.43x cheaper than a single elementwise sweep
  over `C` — which is itself strictly less work than any real unfused epilogue.
- `Activation` (`None`, `Relu`, `GeluTanh`, `Silu`) and `Epilogue`, with
  `Epilogue::default()` as the identity, which dispatches to plain `gemm`.
- Per-column bias broadcasts through a row-stride-0 tensor view, reusing the
  cooperative `load` path that fetches `C_prev`.
- `matmul2d_tensorops_bf16_f32_epi` and `matmul2d_tensorops_f32_relaxed_epi`.
  Separate entry points rather than extra parameters on the existing kernels:
  Metal faults on a declared-but-unbound buffer, so widening those signatures
  would force every current caller to bind four operands it does not use. The
  epilogue is a template parameter, so both share one source and the plain path
  compiles to exactly what it did before.
- `examples/epilogue_cost.rs`.
- **`nn` module — 44 kernels promoted out of `gemma-metal`.** RMSNorm (f32,
  bf16, fused residual-add with layer scale), gated MLP activations (SiLU,
  `gelu_pytorch_tanh`), sliding-window and global flash attention, fused
  RMSNorm+QKV+RoPE, MLX-format Q4 GEMV and GEMM, Q8 GEMV, KV-cache timestep
  stores and ring densify, quantized embedding lookup, and softcap/argmax
  sampling. These had lived in one model's crate, reachable only as raw strings
  through an overlay metallib.
- Every `nn` entry point validates operand extents on the host before encoding.
  The kernels guard `gid >= n` and nothing else, so an undersized buffer was
  previously an unchecked out-of-bounds *device* read.
- A `_with_scalars` variant of each entry point, taking a closure that binds the
  scalar operands. Callers that need stable GPU addresses across encodes — an
  Indirect Command Buffer that froze its binds — supply their own persistent
  scalar pool without reimplementing the dispatch.
- `GpuRuntime::max_threadgroup_memory`, so callers can check a kernel's
  threadgroup-memory request before a dispatch-time failure that names neither
  the kernel nor the dimension.
- `examples/gemm.rs` and `examples/nn_layer.rs`, both run by CI. The README's
  snippets are these files.

### Fixed

- Removed a duplicate `cast_f32_to_bf16` that arrived with the promoted kernels.
  `GpuRuntime::pipeline` resolves the primary metallib before any overlay, so
  the copy in `rms_norm.metal` could never have been the one dispatched.
- `docs.rs` builds. The crate is Apple-silicon only and `build.rs` drives the
  Metal toolchain, neither of which exists on docs.rs's x86_64-linux builder;
  the build script now detects `DOCS_RS` and skips the shader compile, and
  `[package.metadata.docs.rs]` targets `aarch64-apple-darwin`.
- 18 rustdoc warnings: matrix-shape notation such as `C[M,N] = A[M,K] @ B[K,N]`
  was being parsed as intra-doc links, and one public doc linked a private item.
- README: the `add_metallib` / `from_metallib_path` snippets passed `&str` to
  functions that take `&Path` and would not have compiled.

### Changed

- `gemma-metal` now delegates seven wrappers to `tessl::nn` rather than
  reimplementing the dispatch, removing 52 lines of duplicate binding code.
- Documented the crate's `unsafe`. It splits into two classes: `objc2` message
  sends, where every site discharges the same obligation and which are covered
  once in the crate docs, and raw pointer/slice construction, where the
  obligation is site-specific. Every block in the second class now carries a
  `// SAFETY:` comment naming the invariant and where it is established —
  8 comments before, 22 now.
- `icb_smoke::verify_copy` bounds-checks its length against the buffer instead
  of relying on both call sites happening to pass the length the buffer was
  allocated from.

### Removed

- `mtl_tensor::try_quant_tensorops_prefill_gemm`. Its entire body was
  `Err("not wired yet")` with every parameter underscored, no caller and no
  test. The fact it encoded — that quantized TensorOps prefill GEMM does not
  exist, because `MTLTensorDataType::Int4` is unbound in objc2-metal 0.3 — is
  now the single `QUANT_PREFILL_GEMM_WIRED` constant that
  `nax_verify_readiness` reports, replacing a second copy of the same sentinel
  that existed alongside it.


[Unreleased]: https://github.com/bharathvbcr/tessl/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/bharathvbcr/tessl/releases/tag/v0.2.0
[0.1.4]: https://github.com/bharathvbcr/tessl/releases/tag/v0.1.4
[0.1.3]: https://github.com/bharathvbcr/tessl/releases/tag/v0.1.3
[0.1.2]: https://github.com/bharathvbcr/tessl/releases/tag/v0.1.2
[0.1.1]: https://github.com/bharathvbcr/tessl/releases/tag/v0.1.1
[0.1.0]: https://github.com/bharathvbcr/tessl/releases/tag/v0.1.0
