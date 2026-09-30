# Qwen3.5 kernels

`tessl::qwen35` holds the Metal kernels for Qwen3.5's layers that transformers
has no fast Mac path for. Sources: `kernels/qwen35_gdn.metal`,
`kernels/qwen35_attn.metal`, `kernels/qwen35_score.metal`.

> **Status: run on a GPU (M5 Pro); performance not yet measured.** The full
> `cargo test --release -- --test-threads=1` passed on the Mac at `86d09fb`
> (41 test binaries, 0 failures), including every kernel here in
> `tests/qwen35_kernels.rs`. The `Metal compile` workflow (GitHub-hosted macOS,
> Xcode 26.6) builds all three sources under `-std=metal4.0 -Wall -Werror`,
> links them, confirms all sixteen entry points are exported, and builds the
> crate and every test target with no `metal3.2` fallback. Every kernel below
> was also compiled as C++ and executed on a CPU emulator of the Metal
> execution model, then compared against transformers' own Qwen3.5 code (see
> [Verification](#verification)), including under AddressSanitizer and
> ThreadSanitizer, in shuffled threadgroup order, and with fast-math-like
> error injected.

## Why

On the M5 Pro, a Qwen3.5 forward pass through transformers does about 3.8 GFLOP
per token. At 870 tok/s that is 3.3 TFLOP/s, against the 26.6 TFLOP/s bf16 this
chip has shown in tessl's GEMM, so torch reaches about 12% of it. The main cause
is the gated delta net (GDN), which is 18 of the 24 layers. It does almost no
arithmetic, but transformers runs it on MPS as a pure-torch fallback:

- a 63-step Python loop for the triangular solve, about 400–500 launches per
  layer (newer transformers use `solve_triangular`, which is still a launch
  storm on MPS);
- a second Python loop over every 64-token chunk;
- everything upcast to fp32.

That comes to roughly 13–16k tiny launches per forward pass, with the GPU idle
between them. On CUDA, `fla` fuses the same work into a few kernels. These are
that fusion for Metal.

## The kernels

| Item | Entry point(s) | Host function | transformers equivalent |
|---|---|---|---|
| 1. Fused chunked GDN forward | `qwen35_gdn_chunk_prep`, `qwen35_gdn_chunk_scan` | `gdn_chunk_forward` | `torch_chunk_gated_delta_rule` |
| 2. Gates, l2norm, q scale folded into the loads | (inside 1 and 5) | — | `beta = b.sigmoid()`, `g = -A_log.exp() * softplus(a + dt_bias)`, `l2norm`, `q * Dk^-0.5` |
| 3. Causal conv + SiLU | `qwen35_conv1d_silu` | `conv1d_silu` | `causal_conv1d_fn` / `causal_conv1d_update` |
| 4. Gated RMSNorm | `qwen35_gated_rms_norm_{f32,bf16}` | `gated_rms_norm` | `Qwen3_5RMSNormGated` |
| 5. Read-only GDN decode | `qwen35_gdn_recurrent` | `gdn_recurrent` | `torch_recurrent_gated_delta_rule` |
| 6. Attention extras | `qwen35_attn_qk_norm_rope`, `qwen35_attn_gate_{f32,bf16}` | `attn_qk_norm_rope`, `attn_output_gate` | `q_norm`/`k_norm` (`1 + w`), `apply_rotary_pos_emb` (partial), `* sigmoid(gate)` |
| 6b. Decode loops replayed from an ICB | `qwen35_attn_qk_norm_rope_posbuf` | `attn_qk_norm_rope_posbuf` | the position comes from a device buffer, like `rms_qkv_rope_posbuf` |
| 6c. Shared-prefix attention | `qwen35_attn_prefix_rows`, `qwen35_attn_prefix_decode_{partial,reduce}` (+ `slot_base` in 6/6b) | `attn_prefix_rows`, `attn_prefix_decode`, `attn_qk_norm_rope_suffix{,_posbuf}` | attention over a per-row copy of a shared KV prefix, without the copy |
| 6d. Prefill attention on the matrix units | `qwen35_attn_tiled_h256_*` (4 tiles) | `attn_prefill`, `attn_prefill_with_tile` | causal `sdpa` over the layer's own K/V; `nn::flash_attn_rows` with both products on TensorOps |
| 6e. MLP activation | `qwen35_swiglu_{f32,bf16}` | `swiglu` | `act_fn(gate_proj(x)) * up_proj(x)` in `Qwen3_5MLP`, stored as bf16 for `down_proj` |
| 7. Score only the answer rows | `qwen35_score_rows_{f32,bf16}` | `score_answer_rows` | final norm + `lm_head`, restricted to the answer tokens |
| 7b. Embedding gather | `qwen35_embed_rows_bf16` | `embed_rows` | `embed_tokens(ids)` from the bf16 table, on the device, so a forward needs no host gather |
| 8. Fused projections | tessl's GEMM | `pack_linear_weights_*`, `fused_projection`, `project_residual` | `in_proj_qkv/z/b/a`, `q/k/v_proj`, `out_proj` + residual |

### Layout: everything reads the fused projection in place

A GDN layer's four input projections are packed into one weight
(`pack_linear_weights_f32`), so a single GEMM writes one row per token:

```text
[ q (Hk*128) | k (Hk*128) | v (Hv*Dv) | z (Hv*Dv) | b (Hv) | a (Hv) ]      GdnProjLayout
```

Every kernel takes its inputs as column windows of such a row (`Cols { buf, ld,
off }`), so nothing is split or copied between steps. The conv reads the first
`conv_dim` columns and writes a dense `[rows, conv_dim]` q|k|v. The GDN kernels
read q, k and v from that, and read `a`/`b` straight out of the projection. The
gated norm reads `z` from the projection too. Attention works the same way with
`AttnProjLayout`: `[q+gate (per head: D query, then D gate) | k | v]`.

Build the layout once with `GdnProjLayout::new(k_heads, v_heads, v_dim)` (it
validates the shape and that every offset fits `u32`) and take the rest from
it: `layout.dims(batch, seq)` for the call shape, `conv_qkv`, `gates` and `z`
for the windows. A layout and a separately stated shape that disagree on
`v_heads` would still pass every capacity check, and read the wrong columns.

### The GDN layer, end to end

```text
proj   = fused_projection(x, W_in)                         GEMM
qkv    = conv1d_silu(proj[:, :conv_dim], conv_w, state)    kernel 3 (writes the next conv state)
o      = gdn_chunk_forward(qkv, gates(proj), ...)          kernels 1+2 (prefill; writes the final state)
       | gdn_recurrent(qkv, gates(proj), ...)              kernel 5 (decode)
y      = gated_rms_norm(o, z(proj), norm_w) -> bf16       kernel 4
resid += y @ W_out                                         project_residual (GEMM, beta = 1 epilogue)
```

### Why the chunked GDN is two dispatches, not one

With `C = 64`, `G` the chunk-local cumulative log decay and
`Γ_ij = exp(G_i − G_j)`:

```text
A = strict_lower(β_i (k_i·k_j) Γ_ij)         W = (I + A)⁻¹          (the solve)
X = β ⊙ (V − e^G ⊙ (K S))                     U = W X
O = e^G ⊙ (Q S) + lower_incl((q_i·k_j) Γ_ij) U
S ← e^{G_last} S + Kᵀ (e^{G_last − G} ⊙ U)
```

This is transformers' formulation with `k_cumdecay = W (β e^G K)` reassociated
as `W (β e^G (K S))`, so the [C, Dk] product is never formed. Every exponent is
≤ 0, so nothing can overflow however strong the decay.

Everything that doesn't depend on the state S (the norms, the gates, both 64×64
products and the solve) is in `qwen35_gdn_chunk_prep`. It runs one threadgroup
per (batch, head, **chunk**), with no ordering between chunks. The solve works
column-parallel in threadgroup memory: `A`ᵀ sits in the upper triangle and `W`
in the lower triangle of one 16 KB block, and a quad of lanes owns each column,
so the 63 steps need no barrier at all. Only `qwen35_gdn_chunk_scan` walks the
chunks in order. Its per-chunk work is four `simdgroup_matrix` products against
a 128×32 state slice held in 30 KB of threadgroup memory (row strides padded
against bank conflicts). A single fused kernel
would have put 63 dependent steps of solve on the sequential critical path of
every chunk.

The prep → scan hand-off goes through a `GdnWorkspace`. That's about 100 MB for
a 2048-token prefill at `Hv = 32`, and it scales linearly with tokens. Allocate
it once and reuse it across layers. Very long prompts can be split into segments
and carried through `StateIn::PerBatch` + `state_out` to bound it; the result is
identical.

### The scan's value slice

The scan's threadgroups each own a slice of the value columns, and its math
is separable by column: every per-chunk product maps value column c only to
column c. So the slice width sets the parallelism without changing any
element's arithmetic. `GdnScanSlice::Cols16` (`qwen35_gdn_chunk_scan_bv16`,
18 KB of threadgroup memory against 30 KB) launches twice the threadgroups of
the default `Cols32`, bit for bit the same result. At Qwen3.5-2B's shapes
the 32-column scan is only 64 threadgroups at batch 1. `probe_gdn_scan`
measured the scan at 1.37–1.55× its batch-1 time at batch 2, and 2.8× at
batch 4, so batch 1 leaves the GPU partly idle. It is a same-session ratio
under UI load. A first A/B at aae935f, also under UI load (GPU 51–59% busy),
put the 16-column scan at 0.82× the 32-column scan at T = 8192, batch 1
(6.79 vs 8.27 ms), but 1.10× at batch 2 and 1.30× at batch 4. At T = 1024 the
batch-1 comparison was inside the noise. So the narrow slice helps a long
batch-1 prefill and hurts larger batches. The default stays `Cols32`; a
batch-dependent choice waits for a clean measurement.

### Many questions from one prefilled snapshot

`StateIn::Snapshot(buf)` makes every batch row start from the same state and
**never writes it**. This works for `gdn_recurrent` and `gdn_chunk_forward`
(the recurrence) and for `conv1d_silu` (the conv history). So one shared context
is prefilled once, and N questions run as a batch of N continuations, each
writing its own `state_out` if it wants one.

The 6 full-attention layers share their KV prefix the same way. The prefix is
prefilled once at batch 1 into caches with no batch dimension,
`[prefix_capacity, kv_heads, 256]`. Each question's tokens go into its own
suffix cache, `[batch, suffix_capacity, kv_heads, 256]`, whose slot `s` is
position `P + s`:

```text
attn_qk_norm_rope(batch 1, pos_offset 0)            -> prefix K/V   (once)
attn_qk_norm_rope_suffix(prefix_len P, offset s0)   -> q, suffix K/V (per question batch)
attn_prefix_rows(q, prefix, suffix, suffix_len, q_pos)   (several tokens per row)
attn_prefix_decode(q, prefix, suffix, suffix_len, q_pos) (one token per row)
```

`attn_qk_norm_rope_suffix` rotates each token at its absolute position
`P + s0 + t` and stores it at slot `s0 + t`. Underneath, it is
`qwen35_attn_qk_norm_rope` with `slot_base = P`; slot_base 0 is the
ordinary cache. `qwen35_attn_prefix_rows` is `flash_attn_rows`' D=256
instantiation (R=16, 32 simdgroups) with only the address of key `t` changed:
the prefix below `P`, at batch stride 0, then the row's suffix. A masked key is
an exact no-op in the online softmax, so it returns **the same bits** as
`nn::flash_attn_rows` over each row's copied `prefix ‖ suffix`. The tests hold
it to exactly that. `attn_prefix_decode` is the same idea applied to
`flash_attn_decode`: at one query per row, the rows kernel walks all `P + S`
keys in series per (row, head), so the decode path splits them into 128-key
chunks across simdgroups and then reduces. Both passes are
`flash_attn_decode`'s D=256 instantiation (chunk 128, R=16, a GQA group per
threadgroup), with the key address and the live key count `P + S` changed.
It returns the same bits as `nn::flash_attn_decode` over the copy.

The live suffix length and query position are one device `u32` each, shared by
every row, like `flash_attn_rows`' `tkv`. For questions of different lengths,
the `_varlen` forms (`attn_prefix_rows_varlen`, `attn_prefix_decode_varlen`)
take them as `[batch]` arrays instead, and `attn_qk_norm_rope_suffix_rows`
writes each row's tokens from its own position. The kernels read element
`b * row_stride`, so the shared form is stride 0 and the same code. Each row of
a ragged batch is bit-identical to that row run alone. Batch the questions
**right**-padded to the longest: causal attention never lets a token see
anything after it, so padding at the end changes no real token's output,
while padding between the prefix and a question would. The decode scratch is
laid out at the capacity's chunk count, not the live one, so rows with
different live lengths cannot overlap.
For an ICB-replayed decode loop, `attn_qk_norm_rope_suffix_posbuf` reads
the absolute position from a device buffer, the one `attn_prefix_decode`
reads as `q_pos_offset`. At B = 16 and P = 8k, the copy
this avoids is 16 × 8k × 2 KV heads × 256 × 4 B = 268 MB per K or V per layer,
or 3.2 GB across the 6 layers.

**Questions of different lengths in one call.** `conv1d_silu_varlen`,
`gdn_chunk_forward_varlen` and `gdn_recurrent_varlen` take `seq_lens`, a
`[batch]` u32 device buffer. Row b is `min(seq_lens[b], seq)` tokens, with
rows still `seq` apart (right-padded). Everything that measured the sequence
by `seq` now uses the row's own length: the conv's output bound and the
window its carried state is taken from, the prep's row masks and the chunks it
runs, the scan's chunk count and masks, and the recurrence's step count. So
each row, `state_out` included, is bit-identical to that row run alone at its
length, and its rows past that length are not written. Those padded rows then
flow through the rest of the layer as garbage (stale memory, NaN on the
emulator). Every later op is row-wise or causal, so they never reach a real
token, but their outputs are unspecified: never score them. The model check's
ragged flow shows both halves. Leaving out `seq_lens` leaves the question
logits intact but moves the next decode step's logits by ~13, because the
carried state was taken at the padded length. The workspace is still
laid out for `seq`. The scan stops at the row's own chunk count, so it never
reads a chunk the prep skipped for that row, which on a workspace reused
across layers would hold an earlier call's data.

State rules, all enforced on the host:

- A snapshot is never a `state_out`: every row reads it.
- A GDN `state_out` may be the `PerBatch` input state itself (in place).
  Each thread reads exactly the elements it later writes. In place, the state
  is still checked against every other input and output.
- A conv `state_out` may never be its input state: output slot `j` is input
  slot `j + seq`, which another thread reads.
- A call with `seq = 0` and a `state_out` copies the start state through. A
  decode loop that alternates two state buffers never finds a stale one after
  an empty step.

### Prefill attention on the matrix units

`nn::flash_attn_rows` is scalar f32: a simdgroup per query row, one
multiply-add per lane per dim. It is exact and simple, but at Qwen3.5-2B's
shapes (8 query / 2 KV heads of 256) at T = 8192 it measured 169 ms per
attention layer at 28732de, about 1.6 TFLOP/s and half the whole forward. The
same machine's exact-f32 TensorOps GEMM runs at 6.4 TFLOP/s
(`docs/benchmarking.md`).

`attn_prefill` (`qwen35_attn_tiled_h256_*`) is FlashAttention-2 with both
products on `mpp::tensor_ops::matmul2d`, in exact f32 (`relaxed_precision`
off). Each threadgroup takes BQ queries of one head. It walks the keys in
blocks of BK, only up to its last query's position, so blocks above the
diagonal are never visited. Per block:

1. `S = Q·Kᵀ` into a cooperative tensor, stored to threadgroup memory.
2. An f32 online softmax, 4 threads per row, masks the diagonal block and the
   partial block at `Tkv`, and writes `P` over `S`.
3. The `[32, 256]` output accumulator, a cooperative tensor, is rescaled per
   row and then `O += P·V`.

Q, K and V are read straight from the `[B, T, H, 256]` buffers through
strided tensor views, so nothing is staged by hand. The buffers, slots and
masking contract are `flash_attn_rows`': live `min(*tkv, capacity)`, query
and key position offsets, causal, and a row that sees no key is zeros. The
matrix units sum in a different order, so the two agree to f32 rounding
rather than bit for bit. On random inputs with softmax scores of a few units,
each is ~1e-6 from an f64 reference (details under Verification).
`dims.window` must be 0.

The shared-prefix kernels (6c) are still the scalar `flash_attn_rows`
instantiation. They serve the questions, a few tokens each over the prefix,
where the tiled kernel's query blocks would sit mostly idle. A prefix
prefilled by `attn_prefill` and continued by `attn_prefix_rows` therefore
mixes the two kernels' rounding. A test holds three 37-token questions over
one 200-token prefix, through `attn_prefix_rows`, to `attn_prefill` over
each row's whole sequence. They agree to 6–8e-7, and the bound is 1e-5.
Moving the prefix/suffix boundary by one moves them by 9e-2.

### The MLP activation

`swiglu` computes `silu(gate) * up` over f32 column windows and stores
either f32 or the bf16 the down projection's GEMM reads. The generic
`nn::mlp_silu` writes f32 only, so a forward used to add a cast pass per
layer. That pass was 0.18 of the MLP's 0.52 ms of elementwise work at
T = 1024 (28732de). The gate and up windows may be two windows of one
buffer, which is what a single `[gate | up]` GEMM would write. The sigmoid is
the overflow-free one in `kernels/qwen35_act.h`, shared with the GDN gates and
the attention output gate. On this GPU the textbook `1 / (1 + exp(-x))`
also returns 0 at x = -120, so no test here can tell the two forms apart.
The safe form is kept because fast math is allowed to assume `exp` never
returns infinity.

### Scoring only the answer rows

`score_answer_rows` applies the final norm (`w_offset = 1.0` for Qwen3.5's
zero-centred `Qwen3_5RMSNorm`) and dots each slot row with just the answer
tokens' LM-head rows. It returns the logits and a log-softmax over the answer
set. That softmax is the distribution restricted to those tokens, not the
full-vocabulary log-probability, which needs every row.

Slot rows and answer ids live in device memory, so they are checked in the
kernel. An out-of-range slot makes that slot's whole row NaN. An out-of-range
answer id makes that entry NaN in both outputs and leaves it out of the
softmax, so the valid answers beside it keep correct log-probabilities.
Validity is tracked as explicit flags and NaN is stored as integer bits,
because under fast math a NaN passing through float arithmetic, or even a
float `select`, may be optimised away.

## Numerics

- **fp32 inside, bf16 only on the way into a GEMM.** Activations are f32, since
  tessl's GEMM writes f32. The gated norm and output gate can round once to bf16
  on store, so they feed a bf16 GEMM directly.
- **The GDN gate is Qwen3.5's**: `g = −exp(A_log)·softplus(a + dt_bias)`,
  `β = sigmoid(b)`, no clamp. This is *not* the nanolab GDN in
  `tests/common/gdn.rs`, whose sigmoid gate and clamp belong to a different
  model. The recurrence itself is that file's `Rule::Published` with
  `α = exp(g)`.
- **RoPE at long positions.** transformers forms the rotary angle as an fp32
  product, whose rounding is ~6e-8 relative. At position 20000 the angle itself
  therefore moves by ~1e-3 rad. The kernel reproduces that fp32 angle and
  matches transformers to 1e-6. An f64-angle reference would disagree with both
  by 1.4e-3. The Mac test at position 20000 allows 4e-3 for exactly this.
- **Fast math is on** (Metal's default). Every sigmoid/SiLU is written
  `e = exp(−|x|)`, so no intermediate is ever infinite, which fast math may
  assume. Softplus needs care too, because MSL has no `log1p`: rounding
  `1 + e^x` plus the fast `log` near 1 is off by 1–60% for `x` in [−15, −8],
  and `a + dt_bias` lands there routinely (Qwen's `dt_bias` sits around −2 to
  −7). Below −3 the kernel uses the 8-term `log1p` series, and above that
  `precise::log`.
- **Chunk-local cumulative decay** is summed in fp32, as transformers sums it.
  After a step with a large decay, later small differences `G_i − G_j` lose
  relative precision, in both implementations alike. The token-by-token decode
  path doesn't have this.
- **Partial RoPE is not `rms_qkv_rope`'s.** transformers pairs `p` with
  `p + rotary_dim/2` and uses `θ^(−2p/rotary_dim)`. tessl's existing kernel
  implements Gemma's proportional RoPE (pairs across `D/2`, denominator `D`).
  At `rotary_dim = 64` of 256, the two agree on nothing.

## Verification

### Off-device: `tools/msl_emu`

`python3 tools/msl_emu/check_qwen35.py` (requirements pinned in
`tools/msl_emu/requirements.txt`) builds the kernel sources as C++20 against a
CPU stand-in for `<metal_stdlib>`. It launches every threadgroup as real
threads with real barriers and simdgroup collectives, using the dispatch
geometry `src/qwen35.rs` uses. The results are compared against transformers'
Qwen3.5 functions, called directly, and against an f64 sequential recurrence.
CI runs it on Linux (`kernel-emulator` job) in four modes:

| Mode | What it adds | Shown to catch |
|---|---|---|
| plain | Every case runs in forward, reverse and shuffled threadgroup order, and the outputs must agree **bit for bit** | a threadgroup writing past its rows into another's; passes in grid order, fails reversed |
| `--fast-math` | Each non-`precise::` transcendental perturbed by ±4 ulps (`exp` by ±(4 + ⌊2|x|⌋), after Metal's documented bound); GDN held to the on-device bound (`1e-4·max|y|`) | whether the Mac test bounds survive approximate math. At 64 ulps the GDN still sits ~10× inside them |
| `MSL_EMU_SANITIZE=address` | Every device buffer and threadgroup allocation exactly sized | an unmasked tail-row store (heap overflow) |
| `MSL_EMU_SANITIZE=thread` | Data-race detection across the real threads | removing the barrier before U = W·X overwrites X |

**Against the model, not only its functions.** `check_qwen35_model.py` builds
a small random `Qwen3_5ForCausalLM` (three GDN layers and one attention layer;
Qwen3.5's head_dim 256 and rotary 64; a shared key head) and runs its forward
pass on the kernels. It uses this module's weight packing and column layouts,
and tessl's own `flash_attn_rows`, emulated, for the attention. torch does only
the GEMMs, the MLP and the pre-attention norms. The answer-row logits match the
model's own:

| Flow | max logit err (scale ~75) |
|---|---:|
| prefill, 70 tokens, 5 slots | 3.1e-5 |
| cached decode, 2 steps after a 40-token prefill | 9.5e-6 |
| 3 questions from one 66-token snapshot (mixed recurrent and chunked GDN paths) | 2.7e-5 |
| the same 3 questions with the attention's KV prefix shared (`qwen35_attn_prefix_rows`) | 1.4e-5 |
| then one decode step per question (`qwen35_attn_prefix_decode`, per-row suffix cache and GDN state) | 3.3e-5 |
| 3 questions of lengths 5, 2, 4 in one right-padded batch, through every `_varlen` path | 1.8e-5 |
| then one decode step each, at each row's own position | 2.1e-5 |

Four injected wiring errors were each caught with O(1) logit errors: gate
columns swapped, the projection packed out of order, rotary width from the
wrong factor, and the final norm missing its `+1`. So was a wrong query
position handed to flash attention during decode. Under the fast-math noise
model, the logits move by less than 4e-5.

**Compile risk, without a compiler.** `tools/msl_emu/dialect_lint.py` lists
the MSL constructs these kernels use that no other tessl kernel uses (the others
compile on every release build). There are 13: `static_assert`, `as_type`, a
`(device uint *)` cast, `fabs`, `log`, `mem_flags::mem_device`, six `precise::`
functions and `uint3`. Each is standard MSL and on a reviewed list. An
unreviewed one fails CI.

The `host_contract` case parses `src/qwen35.rs` and the kernel signatures. It
checks that every `set_*` bind has the kernel's index and kind (buffer, `uint`,
`float`), and that the host's thread counts and threadgroup-memory sizes equal
the kernel's constants. Swapping a single bind fails it. The kernels also
`static_assert` their lane mappings against those constants.

Last run at 3798ca8: all 41 cases, 147 checks passing, 0 failing. GDN rows
show the kernel's max error next to transformers' own fp32 error, both
measured against f64:

| Case | kernel err | transformers fp32 err |
|---|---:|---:|
| chunk, T=1 | 7.5e-9 | 3.8e-9 |
| chunk, T=64 | 1.5e-8 | 1.4e-8 |
| chunk, T=65, grouped heads, B=2 | 5.9e-8 | 5.7e-8 |
| chunk, T=130, 4 heads, per-batch state | 5.0e-8 | 8.8e-8 |
| chunk, T=100, Dv=128, shared snapshot, B=3 | 2.9e-8 | 4.5e-8 |
| chunk, T=150, strong decay | 4.0e-7 | 3.2e-7 |
| chunk, T=130, `a + dt_bias` in the softplus-series range | 8.4e-8 | 6.1e-8 |
| recurrent, T=9, `a + dt_bias` in the softplus-series range | 1.1e-8 | 5.1e-9 |
| chunk, T=200, Dv=128 | 2.3e-8 | 1.8e-8 |
| chunk, T=1000 | 2.2e-8 | 2.7e-8 |
| chunk, T=4096 (64 chunks through one state) | 4.2e-8 | 3.9e-8 |
| chunk at 16 key / 32 value heads, Dv=128 (`Qwen3_5TextConfig()` defaults†), per-batch state | 7.5e-8 | 7.5e-8 |
| recurrent at the same head counts, shared snapshot | 8.1e-9 | 7.5e-9 |
| chunk at the 2B's head counts (16 key / 16 value heads, Dv=128), per-batch state | 7.5e-8 | 8.7e-8 |
| recurrent at the 2B's head counts, shared snapshot | 8.1e-9 | 7.3e-9 |
| recurrent, T=1, snapshot, B=4 | 6.3e-9 | 4.6e-9 |
| recurrent, T=7, per-batch state | 6.7e-9 | 8.1e-9 |
| recurrent, T=20 | 5.1e-9 | 4.3e-9 |
| chunk / recurrent / conv, T=0 with state_out | state copied exactly | — |
| chunk workspace: `W(I+A) = I` | 4.0e-8 | — |
| conv1d (prefill, state, snapshot, T < KW−1) | 4.8e-7 / state exact | — |
| gated norm f32 / bf16 | 2.9e-6 / within one bf16 rounding | — |
| Q/K norm + partial RoPE, D=256, pos 30000 | 7.2e-7 | — |
| output gate f32 / bf16 / in place | 1.2e-7 / exact / 1.2e-7 | — |
| SwiGLU f32 / bf16 (`Qwen3_5MLP.act_fn`), gate and up as windows of one row, gates to ±120 | 1e-6 + 1e-6·\|x\| (max 1.5e-5 at \|x\| ~ 10³) / within one bf16 rounding | — |
| scoring f32 / bf16; a bad slot or answer → NaN, the rest intact | 6.9e-7 / 5.5e-7 | — |
| embedding gather, bf16 table; a bad id → NaN row, the rest intact | exact | — |
| Q/K norm + RoPE with `slot_base`: absolute RoPE, relative slot | bit-identical to slot_base 0 | — |
| shared-prefix attention, 8 query / 2 KV heads of 256; P = 0, 1, 30 (no suffix), 65 | bit-identical to `flash_attn_rows` on a copied prefix; ≤ 5.0e-7 vs torch | — |
| shared-prefix decode, P = 5, 128 (chunk edge), 120 with a chunk straddling the suffix | ≤ 3.4e-7 vs torch | — |
| shared-prefix rows / decode with per-row lengths and query positions | each row bit-identical to it alone | — |
| chunked GDN, recurrent GDN and conv with per-row `seq_lens` (0, 1, 63–65, T, > T) | each row, `state_out` included, bit-identical to it alone | — |

† These are **not the 2B's** head counts. `Qwen/Qwen3.5-2B-Base`'s `config.json`
has `linear_num_key_heads` 16 and `linear_num_value_heads` **16** (Hv/Hk = 1),
and 8 query / 2 KV attention heads. The 16/32 case comes from transformers'
`Qwen3_5TextConfig()` defaults: hidden 4096, 32 layers, 16 query / 4 KV
attention heads. The class docstring cites `Qwen/Qwen3.5-27B`, but whether a
released size uses exactly these defaults hasn't been checked against its
config. The case is kept because Hv = 2·Hk exercises grouped value heads,
which the 2B does not.

On macOS the harness needs a GCC toolchain: `CXX=g++-16` (Homebrew). Apple
clang's libc++ makes the stand-in `exp` ambiguous and the build fails. A whole
run is long enough that CI's Linux job is the place for it. Locally, `-k`
selects cases.

**The checks catch defects.** Mutations injected into the GDN kernel were each
caught: dropping the inter-chunk decay, using the wrong state-update decay,
halving the solve's quad reduction, removing the `simdgroup_barrier` before a
stage read, skipping k's normalization in the recurrent kernel, an unmasked
tail store (ASan, and reverse order), and a missing threadgroup barrier (TSan).
The one survivor (`j < i` → `j <= i` in building `A`) is equivalent, because
the solve never reads the diagonal slot.

### On device: `tests/qwen35_kernels.rs`

Rust f64 references in `tests/common/qwen35.rs`, held to transformers-generated
goldens in `tests/fixtures/qwen35/` (`scripts/gen_qwen35_fixtures.py`). The GDN
reference matches the f64 golden to 1e-16. The RoPE reference matches
transformers to 7e-7. Both were checked off-device. The GPU tests cover every
kernel, including randomized chunk-edge shapes, grouped heads, snapshots,
strided windows, and a chunked prefill followed by in-place recurrent decode
steps that must equal the recurrence over the whole sequence. The attention
seam runs on the device too: `attn_qk_norm_rope` into `nn::flash_attn_rows`
into `attn_output_gate`, continuing a cache that already holds a prefix, at
head_dim 256. Shared-prefix attention is held **bit for bit** to
`flash_attn_rows` over each row's copied `prefix ‖ suffix`, at 8 query / 2 KV
heads of 256. The prefix runs 0, 1, 63–65, 127–129, 300 (no suffix) and
2049–2100 tokens; B runs 1 through 16; output is f32 and bf16. Every K/V slot
past a live length is NaN, so an over-read would show. A second test prefills
the prefix with `attn_qk_norm_rope`, caches the questions with
`attn_qk_norm_rope_suffix`, and requires the same bits, for q and for the
attention output, as the whole sequence cached per row. Shifting the
prefix/suffix boundary by one fails both tests. Shifting the suffix slot by one
fails the second. The decode path is held bit for bit to `flash_attn_decode`
in the same way. The prefix runs 0, 1, 127–129, 255–257, 300 (no suffix), 2100
and 8200; B runs 1 through 16; chunks straddle the prefix/suffix boundary.
Moving the boundary by one fails it, and so does reducing over the suffix
capacity instead of its live length. `attn_prefill` is held to an f64
reference at T = 1, 31, 32, 33 and 300, B = 1 and 2, for:

- a continuation (37 queries from position 63 over 100 keys);
- a key offset;
- leading queries that precede every key, which must be exactly zero.

Its error may be at most 4× `flash_attn_rows`' on the same inputs. It
measures 2–3× (7e-7 to 1.2e-6, against 2.3e-7 to 5.3e-7). At T = 1500,
B = 2, it is checked against `flash_attn_rows`: 1.0e-6. K/V past the live
length is NaN. The bf16 output must be the f32 output rounded. Six kernel
mutations are all caught: no causal mask, the diagonal off by one, no
rescale, the rescale indexed by column, `t_end` ignoring `Tkv`, and
`1/l` on an empty row. The emulator cannot run TensorOps, so this kernel has
no off-device check. `check_qwen35.py`'s host contract still reads its
buffer slots and its `TILED_ATTN_*` constants from the source. There is also a
whole GDN layer (projection GEMM → conv → chunked rule → gated norm) wired only
through `GdnProjLayout` against the same chain in f64. The tests cover
`state_out` discarded, separate and in place on both paths, and `seq = 0`
passthrough. Every host rejection asserts on its error message, so none can be
credited to a different check. Bounds are relative, the shape of fast-math
error.

```sh
cargo test --release --test qwen35_kernels -- --test-threads=1
cargo test --release --test shader_index_arithmetic   # includes the qwen35 sources
```

## Performance

`cargo run --release --bin bench_qwen35_layers` builds Qwen3.5-2B's shapes
with random bf16 weights: 18 GDN and 6 attention layers, batch 1. It times
the whole 24-layer forward in one command buffer, with 307 launches, and each
stage alone. Before timing, every stage must pass a NaN-poison gate. Runs are
on an M5 Pro, with nothing else of ours on the GPU. Forward times are in ms,
throughput in prefill tokens/s. `+lm_head` adds the full-vocab head, timed on
1024 rows and scaled linearly in T.

| T | 28732de: `flash_attn_rows` (2 runs) | 8a881eb: `attn_prefill` | 8a881eb, `--attn-rows` (same state) |
|---|---|---|---|
| 1024 | 168.5 / 165.9 ms, 6078 / 6172 tok/s | 179.0 ms, 5722 tok/s | 164.1 ms, 6240 tok/s |
| 2048 | 344.5 / 363.0 ms, 5945 / 5642 tok/s | 344.4 ms, 5946 tok/s | 340.3 ms, 6018 tok/s |
| 8192 | 1986 / 1998 ms, 4124 / 4099 tok/s | **1318 ms, 6213 tok/s** (+lm_head 5004) | 1956 ms, 4189 tok/s |

The attention stage alone, per layer, `attn_prefill` vs `flash_attn_rows`:
0.91 vs 2.28 ms at T = 1024, 2.89 vs 10.12 at 2048, and 43.1 vs 140.1 at 8192.
At 8192 that is 275 GFLOP in 43 ms, 6.4 TFLOP/s. That equals the measured
exact-f32 GEMM in `docs/benchmarking.md`, but that GEMM is a single-simdgroup
kernel with no register accumulator, and this one is cooperative. Whether
6.4 is the units' exact-f32 limit is untested. The tile sweep (`AttnTile`,
`--attn-tile`) is the test.

The 1k and 2k forwards do not show the stage saving. In that run the GEMM
stages also came out 10–20% slower than in the control run straight after,
and the GPU read 85% busy at the start from something else. They are
unresolved until re-run. For comparison, on the same machine MLX measured
7,606 / 7,528 / 5,285 tok/s at 1k / 2k / 8k, and torch MPS about 2.8k at 1k
(both by the Lappi project).

A second session at c6fc555 ran with the GPU 93–99% busy from the desktop
compositor and app renderers, so its absolute times are not recorded here.
Its same-session stage ratios are: the GDN chunk scan is ~75% of the chunked
rule (prep 0.47 / scan 1.18 ms at T = 1024, 2.32 / 7.42 at 8192). The fused
`swiglu` takes half the time of `mlp_silu` + cast (0.28 vs 0.56 ms at 1024,
2.43 vs 4.05 at 8192). The four attention tiles land within the noise of
each other at 8192 (54–58 ms under that load).

Where the 8192 forward goes now: the GEMMs, at 25–29 TFLOP/s (the bf16
TensorOps peak), take about 0.8 s. Attention takes 6 × 43 ms, the GDN chunked
rule 18 × 8.4 ms, and SwiGLU + cast 24 × 3.6 ms. At 1024 the GDN chunked rule
is the largest non-GEMM share: 18 × 1.4 ms.

## Not done

- **Training.** No backward kernels. Autograd through transformers' loops is
  what made 2k-token fine-tuning hit 48 GB. That's a Mac-training problem and
  is out of scope here.
- **bf16 inputs.** The kernels read f32 activations, which is what tessl's GEMM
  writes. A bf16-activation variant would halve their read traffic.
- **Shared-prefix attention, remaining gaps.** Only head_dim 256 is compiled. The rows of
  a batch read the shared prefix independently: rows that share a head could
  share its K/V lines in one threadgroup, but no measurement says that is
  worth doing yet.
- **Prefill attention, remaining gaps.** `attn_prefill` is exact f32
  TensorOps. The measured exact-f32 GEMM on this machine reaches 6.4 TFLOP/s,
  against 26.6 for bf16. A variant that casts the Q/K/V tiles to bf16 on load, with
  f32 accumulation and softmax, is the next step. It would move the
  attention output by bf16 rounding, which has to be measured at the model's
  logits first. Each of a KV head's 4 query heads reads that head's K/V
  separately, and head_dim 256 is the only size compiled.
- **mRoPE with image positions.** Text positions only. For text, the three mRoPE
  streams are equal and the rotation reduces to plain RoPE.
- **Key head dim other than 128**, and value head dims that aren't multiples of
  32, are rejected on the host.
- **Redundant work left in.** The normalized-k/q workspace is stored per
  value head, so it is duplicated `Hv/Hk` times (2× for Qwen3.5). The prep
  products are uneven across simdgroups (triangular). Both are
  performance-only.
- **Untuned tiles.** GDN's 64-row chunks and 32-column value slices, and the
  prefill attention's tile (`ATTN_PREFILL_TILE`, 32x32), are first
  choices, not swept ones. Four attention tiles are compiled for the sweep.
