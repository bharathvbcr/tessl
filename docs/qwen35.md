# Qwen3.5 kernels

`tessl::qwen35` holds the Metal kernels for Qwen3.5's layers that transformers
has no fast Mac path for, forward and backward. Sources: `kernels/qwen35_*.metal`,
with the training ops in `kernels/gdn_train.metal` and
`kernels/cross_entropy.metal`.

> **Status: run and timed on a GPU (M5 Pro).** The full
> `cargo test --release -- --test-threads=1` passed on the Mac at `26213f1`
> (482 passed, 7 ignored, 0 failures), including every kernel here in
> `tests/qwen35_kernels.rs`; the timings are in
> [Performance](#performance) and [Training-path timing](#training-path-timing). The `Metal compile`
> workflow (GitHub-hosted macOS, Xcode 26.6) builds each of those sources under
> `-std=metal4.0 -Wall -Werror`, links them, confirms that every entry point on
> its list is exported, and builds the crate and every test target with no
> `metal3.2` fallback. Every kernel in [the table below](#the-kernels)
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
| 6d. Prefill attention on the matrix units | `qwen35_attn_tiled_h256_*` (4 tiles) | `attn_prefill`, `attn_prefill_by_length`, `attn_prefill_with_tile` | causal `sdpa` over the layer's own K/V; `nn::flash_attn_rows` with both products on TensorOps |
| 6e. MLP activation | `qwen35_swiglu_{f32,bf16}` | `swiglu` | `act_fn(gate_proj(x)) * up_proj(x)` in `Qwen3_5MLP`, stored as bf16 for `down_proj` |
| 7. Score only the answer rows | `qwen35_score_rows_{f32,bf16}` | `score_answer_rows` | final norm + `lm_head`, restricted to the answer tokens |
| 7b. Embedding gather | `qwen35_embed_rows_{bf16,f32}` | `embed_rows` | `embed_tokens(ids)` from the bf16 (or, in the f32 model, f32) table, on the device, so a forward needs no host gather |
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
`Cols32`, bit for bit the same result. At Qwen3.5-2B's shapes
the 32-column scan is only 64 threadgroups at batch 1. `probe_gdn_scan`
measured the scan at 1.37–1.55× its batch-1 time at batch 2, and 2.8× at
batch 4, so batch 1 leaves the GPU partly idle. It is a same-session ratio
under UI load. A first A/B at aae935f, also under UI load (GPU 51–59% busy),
put the 16-column scan at 0.82× the 32-column scan at T = 8192, batch 1
(6.79 vs 8.27 ms), but 1.10× at batch 2 and 1.30× at batch 4. At T = 1024 the
batch-1 comparison was inside the noise. `probe_gdn_scan --paired` then timed
both widths on one workspace at batch 1 (Apple M5 Pro): Cols16 was faster at
T = 200 and T = 8192 and matched Cols32 bit for bit, so `GdnScanSlice`
defaults to `Cols16`. A later paired run on this Apple M5 Pro, one GDN scan
layer, batch 2, ABBA, 9 rounds, found the opposite of the older probe's
batch-2 result, with bit-identical outputs. At T = 61, Cols32 median
0.1005 ms (min 0.0990, max 0.1247) and Cols16 median 0.0781 ms (min 0.0736,
max 0.1966), Cols16 faster on 7 of 9 rounds (ratio 0.777); the ranges overlap
because of two slow Cols16 rounds. At T = 200, Cols32 median 0.3839 ms
(min 0.3676, max 0.4377) and Cols16 median 0.3210 ms (min 0.2631, max 0.4132),
Cols16 faster on 8 of 9 rounds (ratio 0.836); the ranges overlap on one slow
Cols16 round. Output and final state had 0 mismatches at both lengths. That
1.10× batch-2 figure is the older probe and is not grounds to revert the
`Cols16` default. Batch 4 was not remeasured; the older probe's 1.30× at
batch 4 stays unrechecked.

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

`attn_prefill_by_length` selects `attn_prefill` (`ATTN_PREFILL_TILE`) across
every query length $t_q$ (`prefill_attn_kernel` returns
`PrefillAttnKernel::Tiled` with no length cutoff). Paired and interleaved in
one process on an M5 Pro (`bench_qwen35_layers --paired-attn`, same Q/K/V,
ABBA, 9 rounds), `attn_prefill` beat `nn::flash_attn_rows` at both lengths with
non-overlapping sample ranges (per launch, median / min: $T = 200$,
0.116 / 0.098 ms vs 0.232 / 0.221 ms; $T = 8192$, 55.3 / 53.4 ms vs
176.2 / 173.7 ms). A cutoff that kept the scalar row kernel below 8192 was
the slower kernel at $T = 200$.

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
  `precise::log`. No kernel spells `INFINITY` in code either, since `build.rs`
  passes `-fmetal-math-mode=fast` and the compiler may fold a compare against
  it. Running maxima start at `-FLT_MAX`. Whether a row has seen a key is
  `l > 0` (the attention kernels) or a `first` flag (cross-entropy), and a
  masked score's weight is zeroed by its mask, not by its value. A row with
  no key saves `FLT_MAX` as its log-sum-exp, and the backward gives that row
  no probability. Plain max reductions seed from the row's own data.
  `tests/kernel_fast_math.rs` fails on any `INFINITY` left in kernel code.
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
compile on every release build). There are 11: `as_type`, a
`(device uint *)` cast, `log`, `mem_flags::mem_device`, six `precise::`
functions and `uint3` (`static_assert` and `fabs` are now used elsewhere too).
Each is standard MSL and on a reviewed list. An unreviewed one fails CI.

The `host_contract` case parses `src/qwen35.rs` and the kernel signatures. It
checks that every `set_*` bind has the kernel's index and kind (buffer, `uint`,
`float`), and that the host's thread counts and threadgroup-memory sizes equal
the kernel's constants. Swapping a single bind fails it. The kernels also
`static_assert` their lane mappings against those constants.

Last run at 3798ca8: all 42 cases, 147 checks passing, 0 failing. GDN rows
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
| embedding gather, bf16 and f32 tables; a bad id → NaN row, the rest intact | exact (bits) | — |
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

### The whole model: `tests/qwen35_model.rs`

`tessl::qwen35_model` composes the kernels into the text model's prefill
forward, loaded straight from the Hugging Face `.safetensors` checkpoint
(`tessl::safetensors`, a strict reader; no conversion step). It is the one
place the layer order, the norms' `1 + w` (formed in their kernels), the weight layouts and the
tied LM head are written down. `Precision::Bf16` is the production numerics
(bf16 GEMM inputs, f32 everywhere else); `Precision::F32` keeps every
activation f32 with exact-f32 GEMMs, which makes it the same computation as
transformers' fp32 forward up to operation order.

The test runs Qwen3.5-2B-Base on a 155-token prompt (three GDN chunks) against
transformers' `Qwen3_5ForCausalLM` (`tools/qwen35_ref/make_reference.py`, fp32
and bf16). Its bounds were fixed before the first run.

| | F32 vs transformers fp32 | Bf16 vs transformers fp32 | transformers bf16 vs fp32 |
|---|---:|---:|---:|
| residual stream, worst of 24 layers (relative) | 2.3e-6 | | |
| logits (relative) | 1.9e-6 | | |
| top-1 token differs | 0 / 155 | 0 / 155 | 2 / 155 |
| KL divergence per position, mean (max) | (2.6e-10) | 3.1e-5 (3.5e-4) | 5.7e-4 (2.7e-3) |

So tessl's production path is 18x closer to fp32 than transformers' own bf16
forward, and this is the measurement any relaxed-precision change (bf16
attention tiles, say) must now pass before it is adopted. Swapping the GDN
`a`/`b` gate projections, folding the gated norm's weight to `1 + w`, or RoPE
theta 1e6 instead of 1e7 each fail the F32 test at the first layer they touch.
It needs the 4.5 GB checkpoint, so it is opt-in (a missing file fails it):

```sh
python3 tools/qwen35_ref/make_reference.py
QWEN35_2B_SAFETENSORS=.../model.safetensors-00001-of-00001.safetensors \
  cargo test --release --test qwen35_model -- --ignored --test-threads=1
```

Loading holds little beyond the device weights. Each projection part is
placed into its packed GEMM operand as it is read and then dropped, and the
embedding goes straight into its device tensor, except for the one host copy
that the bf16 transposed LM head needs. At 2B the peak memory footprint is
5.86 GB for bf16 (4.78 GB on the device), 3.83 GB for a bf16 `load_tower`
(3.77 GB) and 7.63 GB for f32 (7.53 GB). Before this change the same loads
peaked at 6.95, 5.93 and 10.79 GB.
`src/bin/probe_load_memory.rs` measures this, and
`bench/results/qwen35_load_rss_m5pro.txt` holds the runs.

## Training memory: where torch's backward spends it

`tools/qwen35_ref/saved_memory.py` attributes every tensor autograd saves
(parameters excepted, deduplicated by storage) to the op that saved it, for
Qwen3.5-2B's dims cut to one repeat of the layer pattern (3 GDN + 1
attention layer, random weights: what a layer saves depends on shapes only),
bf16 on MPS, torch 2.13, transformers 5.15 with its torch GDN fallback.
flash-linear-attention 0.5.2 is installed in the ML venv, and transformers on
this Mac still does not import it because is_flash_linear_attention_available
requires CUDA, and the kernels need Triton, which is not installed. All of it
is linear in T; SDPA saves no `[T, T]` matrix.

| T = 2048 | 4 layers, measured | 24 layers (x6; LM head once) |
|---|---:|---:|
| LM head + full-vocabulary loss | 1.95 GB | 1.95 GB |
| GDN core (`torch_chunk_gated_delta_rule`) | 1.30 GB | 7.8 GB |
| GDN core on tessl (`--tessl`: inputs + a state per 64 tokens) | 0.17 GB | 1.0 GB |
| everything else (norms, MLP, conv, projections, attention) | 1.34 GB | 8.0 GB |
| MPS driver memory after the forward | 11.5 GB | |

The driver holds about 2.5x the saved bytes: the fallback's per-chunk Python
loop leaves transients in torch's caching allocator. The GDN core's minimum
is its inputs plus one state per chunk, about 46 MB per layer at T = 2048
against the 435 MB it saves, which is why the GDN op is the first backward
tessl takes on; `tessl::cross_entropy` already removes the LM head's row.

### The training op: `tessl::gdn_train`

`gdn_train_forward` / `gdn_train_backward` are the gated delta rule at
transformers' seam (`torch_chunk_gated_delta_rule` with
`use_qk_l2norm_in_kernel=True`): `g` and `beta` arrive computed, and q/k are
l2-normalized in the kernel. The forward is a token recurrence per (batch x
head, 16 value columns) that saves the state every 64 tokens and nothing
else: 32 MiB per layer at the 2B's shapes and T = 2048, against the 435 MiB
the torch fallback saves. The backward walks the chunks in reverse,
recomputes each chunk's states from its checkpoint into a bounded
`GdnTrainWorkspace`, and runs the reverse-mode recurrence; the value slices'
partial `dq`, `dk`, `dg`, `dbeta` are summed in a fixed order (no
atomics), so gradients are deterministic.

From torch, `tessl_torch.chunk_gated_delta_rule` has transformers' signature
and `tessl_torch.patch_transformers_qwen3_5()` swaps it in for the fallback
(`python/tests/test_gdn.py`: against transformers' own function through
autograd, every output and gradient within 1.0e-6 relative in f32 and within
bf16 rounding in bf16; a small Qwen3.5 with grouped GDN heads trains with the
same loss and parameter gradients, worst 1e-3, patched or not).

`tests/gdn_train.rs`: the f64 reference's forward equals the
transformers-anchored recurrence to 1e-12, its hand-derived backward equals
central finite differences to 3e-10 for every input and the initial state,
and the kernels match it within 3.4e-7 of the largest magnitude (bound 1e-4)
across chunk edges (T = 1, 63, 64, 65, 130) and at the 2B's 16 heads x 128.
Eleven of twelve injected kernel defects fail it; the twelfth is equivalent.
Both directions are token-sequential, so they are slower than the chunked
inference forward; they have not been timed yet.

### Row-local backward: `tessl::qwen35_bwd`

The layer's row-local ops have their backward in `tessl::qwen35_bwd`: the
RMSNorm (`rms_norm_bwd`, which can add `dx` into an existing residual
gradient), the GDN gated RMSNorm (`gated_rms_norm_bwd`, `dx`, `dz`, `dw`),
SwiGLU (`swiglu_bwd`) and the attention output gate (`attn_gate_bwd`). Each
reads and writes the forward's own windows, so `dgate`/`dup` and the gate
column's gradient are written straight into the fused projection's gradient
buffer for its GEMM backward. Weight gradients are per-block partials in a
caller scratch (`*_part_len`) summed over blocks in order, so they are
deterministic.

`tests/qwen35_bwd.rs`: the f64 references' backward equals central finite
differences (bound 1e-7), and the kernels match them within 3.6e-7 of the
largest magnitude (bound 1e-4) across partial blocks, widths up to 5120 and
the 2B's 16 heads x 128 gated layout; overlapping windows, short buffers and
out-of-range windows are refused. Fourteen of fourteen injected kernel
defects fail it.

`conv1d_silu_bwd` is the GDN causal conv + SiLU backward from a zero state,
which is the conv transformers trains (`causal_conv1d_fn`: padding
`kernel_width - 1`, no bias, sliced to `seq_len`); it takes no carried state
and no ragged lengths. `silu` is not invertible, so both kernels recompute
the pre-activation from `x` (at most 8 multiply-adds) instead of saving it.
`dw` is summed per 256-row block and then in block order. Against its f64
reference (itself checked against finite differences and against the
inference forward's reference) it is within 3.1e-7 of the largest magnitude
from T = 1 (every tap in the padding) and T < KW - 1 across batch rows to
the 2B's 6144 channels, at kernel widths 2, 4 and 8. Nine of ten injected
defects fail it; the tenth drops the padding test, which makes the read
wrap far past the buffer, and this GPU returned zeros there, so no output
can tell.

`attn_qk_norm_rope_bwd` takes the gradients of the rotated q, the rotated k
and v (dense, as the forward wrote them with positions `0..seq`) back
through the inverse rotation and the `(1 + w)` norms into the q, k and v
columns of the fused projection's gradient; `attn_gate_bwd` fills the gate
columns. The rotation's angle is `qwen35_rope_angle` in
`kernels/qwen35_act.h`, the forward's own, so the inverse is its exact
transpose. Against an f64 backward (checked against finite differences, its
forward against the transformers-anchored `norm_rope_row_f64`) it is within
2.4e-7 of the largest magnitude at short positions and 7.4e-6 at the 2B's
heads over 300 positions, where one ulp of the f32 angle is 3e-5 rad; the
cases cover full, partial, lane-splitting and zero rotary widths and blocks
straddling batch rows. Fourteen of fourteen injected defects fail it.

`embed_rows_bwd` adds `dh[r]` into row `ids[r]` of the table's gradient.
Qwen3.5-2B ties its embedding to the LM head, and `cross_entropy_rows`
overwrites the head's `dW`, so the embedding's gradient is added after it
into the same buffer (an untied table is zeroed first). The ids are known
on the host in training, so they are checked there (an id `>= vocab` is an
error, not a NaN row) and the rows are grouped by id: one thread per (id,
column) sums that id's rows in row order and adds once, without atomics, so
the gradient is deterministic and rows no id reads keep their bits. Within
5e-7 of the f64 sum across repeated, reversed, single-id and 2048-wide
cases; eight of eight injected defects (kernel and host grouping) fail it.

`gdn_gates` writes the GDN gates as values, `g = -exp(A_log) *
softplus(a + dt_bias)` and `beta = sigmoid(b)`, dense `[rows, heads]` for
`gdn_train`, with the helpers the inference kernels fold into their loads
(`qwen35_softplus`, `qwen35_log_decay` in `kernels/qwen35_act.h`).
`gdn_gates_bwd` writes `da`, `db` into the fused projection's a and b
gradient columns and sums `dA_log`, `ddt_bias` per block in order. The
softplus derivative is torch's (1 above 20); in f32 that branch and the
smooth one agree to the last bit wherever both are finite (`sigmoid(20)`
rounds to 1), so its test is the forward past 88, where only the threshold
keeps `e^x` from overflowing. Within 7e-7 of the f64 reference through the
threshold, the series range and block edges; ten of twelve injected defects
fail it, the other two (moving or removing the backward's threshold) being
f32-equivalent. `copy_cols` moves a column window between layouts (q, k, v
between the conv output and `gdn_train`'s dense operands).

### Training attention: `tessl::attn_train`

`attn_train_forward` is `attn_prefill`'s tiled kernel at its default
geometry (32 queries by 32 keys, 4 simdgroups) instantiated once more with
each query row's log-sum-exp of the scaled scores written to `[B, H, T]`;
its O is bit-identical to `attn_prefill`'s. `attn_train_backward` is
FlashAttention-2's backward on the same `matmul2d` units
(`kernels/qwen35_attn_bwd.metal`): P is rebuilt per block from the saved
log-sum-exp, `Dr = rowsum(dO ∘ O)` is one pass, dQ is owned per query block
(walking keys to the diagonal) and dK, dV per key block of a KV head
(walking that head's query heads in order, then the queries from the
diagonal on). Each gradient is written once by its owner, so there are no
atomics and the gradients are the same bits on every run; dK and dV are
separate kernels so each keeps the forward's one accumulator.

`tests/attn_train.rs`: the f64 reference's backward equals central finite
differences (1e-7), and the kernels match it within 4e-6 of the largest
magnitude (O, lse, dQ, dK, dV) from T = 1 through partial, exact and
one-row blocks, two batch rows and the 2B's 8 query over 2 KV heads. Fourteen
of fourteen injected defects (masking, scale, `Dr`, head grouping, the
diagonal start, the log-sum-exp reads and store) fail it. The backward has
not been timed yet.

### Training-path timing

`cargo run --release --bin bench_qwen35_train` times each training op at
Qwen3.5-2B's shapes (random bounded f32 inputs, one NaN-poisoned run per op
first: every output must come back finite, and the embedding backward's rows
must move), and `--step=N` a whole `train_step` on the real checkpoint. On
the M5 Pro at T = 2048, nothing else of ours on the GPU, display kept awake
with `caffeinate -d` (ms per call, median of 7):

| op | ms |
|---|---:|
| `gdn_train` forward | 6.5 |
| `gdn_train` backward | 21.0 |
| `attn_train` forward | 3.0 |
| `attn_train` backward | 13.9 |
| cross-entropy + both gradients (vocab 248320) | 1845 |
| `conv1d_silu_bwd` | 3.2 |
| `swiglu_bwd` | 1.2 |
| `attn_qk_norm_rope_bwd` | 0.9 |
| `gated_rms_norm_bwd` | 0.8 |
| `rms_norm_bwd`, `gdn_gates_bwd`, `embed_rows_bwd` | 0.2-0.5 each |

The GDN core is 27.5 ms per layer forward and backward (both
token-sequential), attention's 17 ms. The cross-entropy is four
`[2047, 2048] x [2048, 248320]` products (two logit walks, then dh and dW),
about 8.3 TFLOP, so its 1.8 s is 4.5 TFLOP/s against the 6.4 TFLOP/s the
exact-f32 GEMM reaches. A whole `train_step` at T = 2048 takes 7.2 s (284
tokens/s, median of 3, allocations included): the projections' forward and
backward GEMMs are about 25 TFLOP (3.8 s at that rate), the cross-entropy
1.8 s, the GDN and attention cores 0.6 s.

Activation recomputation was measured against keeping every layer's
intermediates, on the same machine and T, one mode per process under
`/usr/bin/time -l` (median of 3, at 3aa7e97, when both modes existed). The
saving mode was then removed as a losing branch: its gradients were
bit-identical, and it bought 10% speed for 22x the activation memory, which
does not fit at T = 8192 (~34 GiB):

| activations | s / step | activations kept | peak footprint |
|---|---:|---:|---:|
| every layer's (removed) | 7.23 | 8.54 GiB | 34.4 GB |
| layer inputs, recomputed | 7.94 | 0.38 GiB | 30.7 GB |

Recomputing costs 10% (one more forward of every layer, without its `down`
projection). The peak drops by 3.7 GB, not the whole 8.2 GiB of
activations. Where the peak falls was not profiled; the likely reason is
that it is mostly memory both modes hold (the f32 weights and their packed
LM head, and the f32 gradients, which grow as the backward frees
activations), and `/usr/bin/time`'s footprint also counts the loader's
transient host buffers. Swap did not grow in either run. A first
measurement read 40.9 / 43.1 GB because the bench kept its warm-up step's
gradients alive; it now keeps only the loss.

Both footprints predate `d0fe70e`, which replaced the f32 model's bf16
gather table and separate f32 `[2048, 248320]` head (1.0 + 2.0 GB) with one
f32 table (2.0 GB). The recomputing step's peak should now be about 1 GB
lower; that is arithmetic, not a re-measurement.

### Device memory: tables at their size, and a step that cannot fit refused

The buffer pool used to round every request up to a power of two, Hot
buffers included, which never return to the freelist. The 2.03 GB embedding
took 4.29 GB, and each f32 table of the 2B (weights, a gradient bank, each
AdamW moment) 10.20 GB for 7.53 GB of values. Hot buffers, and Cold ones over
1 MiB, are now made at their size rounded to Metal's 16 KiB allocation
granule; small temporaries keep power-of-two buckets
(`runtime::audit_tests::persistent_and_large_buffers_are_page_rounded_not_power_of_two`).
`MTLDevice::currentAllocatedSize` on the 2B in f32, M5 Pro 64 GB (recommended
working set 51.54 GB), `probe_train_memory`:

| after | before the change | now |
|---|---:|---:|
| `Qwen35Model::load` | 10.22 GB | 7.55 GB |
| `Qwen35Grads::zeros_like` (bank) | 20.42 GB | 15.07 GB |
| `AdamW::new` (both moments) | 40.83 GB | 30.13 GB |
| one exact-f32 step at T = 128 into the bank | 43.59 GB | 32.61 GB |
| its waited commit (freed buffers recycled) | 41.44 GB | 30.58 GB |
| one f32 table staged for a gradient read-back | 50.20 GB | 37.94 GB |

That is the "~11.3 GB unattributed" beyond weights, bank and moments in
ojas-qwen35's read-back: 2.67 GB of rounding per table. The "before" column
up to the recycle is this probe on the old pool, and matches ojas-qwen35's
README to the hundredth; its 50.20 GB staging is ojas-qwen35's measurement,
not re-run on the old pool (it left 1.34 GB of headroom). The same staging
now leaves 13.60 GB.

`Qwen35Model::train_step_bytes(t, operands)` bounds what one step allocates
on top of what is allocated when it starts, and `train_forward` (so
`train_step` and `train_step_into`) refuses a step, before any GPU work, when
that bound plus the device's current allocation exceeds the recommended
working set. The bound is what the step holds throughout (each layer's
input, the final norm's output and gradient, the `[vocab, hidden]` head
gradient, the step's one `AttnTrainWorkspace`, the backward's scratch with
`GdnTrainWorkspace`'s split-K parts),
plus the most it allocates between two waited commits (a freed buffer is
only recycled at one), plus the freelist cap. Every buffer counts at its
pool size (`GpuRuntime::allocated_bytes_for`), and each workspace and GEMM
gives its own (`allocated_bytes_for` on `GdnTrainWorkspace`, `CeWorkspace`,
`AttnTrainWorkspace` and `EmbedBwdWorkspace`; `GemmOperands` for its bf16
operand copies and split-K scratch). Without async encode every dispatch waits, so
two adjacent layers bound a window; with it, a step waits only at each
attention layer, and the window is every layer from one attention layer to
the next (a deliberate wait there, where a fresh workspace's host writes
used to drain). `train_step_into` into an f32 bank writes the head gradient
into the bank's embedding instead of holding it, and its pre-flight leaves
it out; `train_step_bytes`, which also bounds `train_forward` followed by
`train_backward_into`, still counts it. Against the measured peak (`GpuRuntime::peak_allocated_bytes`,
sampled at every buffer the pool creates) from an empty freelist, bank
resident, no moments:

| T | operands | encode | peak over start | bound | bound − peak |
|---:|---|---|---:|---:|---:|
| 128 | exact f32 | per dispatch | 2.49 GB | 4.85 GB | 2.36 GB |
| 2048 | exact f32 | per dispatch | 4.11 GB | 6.82 GB | 2.71 GB |
| 8192 | exact f32 | per dispatch | 9.31 GB | 13.13 GB | 3.82 GB |
| 128 | exact f32 | async | 3.20 GB | 5.57 GB | 2.37 GB |
| 2048 | exact f32 | async | 6.01 GB | 8.52 GB | 2.51 GB |
| 8192 | exact f32 | async | 14.72 GB | 17.95 GB | 3.23 GB |
| 2048 | bf16 | per dispatch | 4.41 GB | 7.83 GB | 3.42 GB |

The bound never fell below the peak, and exceeded it by at most the
freelist cap (2.15 GB) plus 30% of the peak. Every peak was in the
backward. `tests/qwen35_train.rs`
(`a_step_over_the_working_set_is_refused_before_it_runs`) holds the
refusal (it fails with the check removed: the step runs) and the bound
against the tiny model's measured peak at three lengths.

`ps` RSS is not a device-memory cap. It counts a Metal shared buffer's pages
only while they are resident and uncompressed: zeroing the 7.53 GB bank on
the host raised it by 7.53 GB, but the 15.05 GB of moments by 6.9 GB, and
after a step it read 7.84 GB with 15.07 GB allocated. A cap on RSS (Lappi's
`tools/mac_heavy.sh` 32 GiB) therefore lets the device allocate well past
it; `currentAllocatedSize` is the figure the working set is measured
against.

### From torch: `tessl_torch.Qwen35`

`src/qwen35_params.rs` exposes the model's parameters and gradients under
transformers' names and values (norms as `w`, which is also what is stored; each
linear weight as its `[in, out]` window of the packed projection) and copies
them between the model and caller tensors on the GPU. The C ABI (version 9)
adds a model handle (`tessl_qwen35_load`, `_train_step`, `_param_count`,
`_param_info`, `_copy`, `_free`, and `_adamw_init`, `_adamw_step`,
`_adamw_step_count`, `_adamw_set_step_count`, `_adamw_free`, `_grad_sq_norm`,
`_train_forward`, `_hidden`, `_train_backward`, `_train_discard`; `_copy`
directions 3-6 read and write both moments, for checkpoints), and
`tessl_torch.Qwen35` wraps it (see `python/README.md`).

`tessl::qwen35_adamw` runs AdamW on the model's own parameters, so a
training loop needs no torch copy of the parameters or gradients: on the 2B
that is params 8 + gradients 8 + moments 16 GB plus the step's scratch,
against about 55 GB with torch's optimizer (estimated, not measured). The
update is `torch.optim.AdamW`'s single-tensor path in its order, per
parameter-table entry, every parameter (the norms' `w` included) as stored;
weight decay is per entry, and the default excludes what transformers'
Trainer excludes (every norm and `linear_attn.dt_bias`). Against an f64
reference of torch's formula over five steps on the tiny model the worst
error is 1.3e-7 (bound 2e-6), and against `torch.optim.AdamW` itself over
three steps it is within 1e-6 (`tests/qwen35_adamw.rs`,
`python/tests/test_qwen35.py`). Clipping is the caller's: `grad_sq_norm`
gives the square of the global gradient norm, and `grad_scale` multiplies
every gradient by `clip_grad_norm_`'s coefficient inside the update, with
no pass over the gradients and the stored ones left unscaled; the caller
forms the coefficient so gradients outside tessl join the norm.

A step also runs in two halves for a loss outside tessl (a head of the
caller's own): `train_forward` scores chosen positions against given tokens
(`Supervise::Rows`: the sum, its gradients scaled by the caller, so a batch
mean split across rows is `1 / N` on each) or takes the causal-LM loss,
`PendingStep::hidden` gives the final norm's output at chosen rows, and
`train_backward_into` adds that loss's gradient there before the backward.
Gradients accumulate across a batch's rows in a bank
(`Qwen35Grads::zeros_like`). Against transformers on a right-padded two-row
batch (a letter row through the tied head, a span row through a pointer head
in torch), run row by row and accumulated, the losses, the summed gradients
and the global norm agree (`python/tests/test_qwen35.py`).

In `Precision::F32` the tied embedding is one f32
`[vocab, hidden]` table: the gather reads it by row
(`qwen35_embed_rows_f32`), the training step's cross-entropy as its weight,
and the inference forward's head as the transposed operand of one NT GEMM,
so a write is exact and moves all three. (It was a bf16 gather table plus an
f32 `[hidden, vocab]` head, 1 GB more, and a write had to round.) The bf16
forward keeps its packed `[hidden, vocab]` head beside the table: an NN GEMM
over it measured 0.035-0.038 ms per row against 0.047 for the NT GEMM over
the table (`bench_qwen35_layers`, 1024 rows, two runs each), and its logits
are unchanged (the parity numbers above re-ran identical). A tile sweep of
the NT kernel at the head's shape (`bench_gemm_tnnt_tune`,
`bench/results/bf16_nt_lm_head_m5pro.txt`) closed most of the gap but not
all of it: 256x64 on 8 simdgroups (fewer passes over the 1 GB table) ran at
35.9 / 37.1 ms against the NN head's 34.2 / 33.4, taller tiles were slower,
and that tile runs the bf16 NT dx shape `gemm_nt_train` serves under
`PrecisionMode::Bf16` (4096 x 128 x 384) at half the production kernel's
speed (0.260 against 0.133 ms; the sweep's "0.51x" is throughput), so it
could only be a head-only kernel. `train_step` on bf16 operands runs that
production NT kernel (`gemm_nt_bf16`), which the tile would have slowed; on
exact f32 it runs `gemm_nt_f32`. Neither that 5-11% nor the 1 GB saved
matters much here, since nothing in production runs the bf16 full-vocabulary
head (Lappi scores answer rows through `score_answer_rows`), so the bf16
model keeps the faster, existing head and no kernel was added. `tests/qwen35_params.rs`, `tests/capi.rs` and
`python/tests/test_qwen35.py` check that the values are the checkpoint's,
that the gradients are transformers' autograd's (the Python test computes its
own oracle), that a byte offset is honoured, that a bad tensor stops a write
before anything is written, and that an AdamW step written back gives
transformers' loss and gradients after the same step.

### Stress: randomized shapes

`randomized_shapes_stress` in `tests/qwen35_bwd.rs` and `tests/attn_train.rs`
draws shapes within each backward kernel's contract (rows, widths and heads
across block edges, batch rows, kernel widths 2-8, rotary widths from 0 to
the head, repeated and scattered ids, grouped KV heads) and checks each draw
as the targeted tests do: the f64 reference, writes only inside the output
window, the same bits on a rerun. CI runs four draws per kernel;
`TESSL_FUZZ_ITERS` and `TESSL_FUZZ_SEED` scale it, and every draw's shape and
seed are printed so a failure reproduces. 500 draws per sweep (seed 2026,
about 16 s) all pass, the worst at 2.7e-5 of the reference's largest
magnitude (attention dK at T = 3, where the few key gradients nearly cancel).

### A training step: `Qwen35Model::train_step`

`train_step(ids)` is one sequence through the model with transformers'
`ForCausalLMLoss` (position t predicts `ids[t + 1]`, mean over `T - 1`
positions) and every parameter's gradient of it. It runs in f32
(`Precision::F32`), or on bf16 storage (see "Training on bf16 storage"). A
GDN with more value heads than key heads (the 4B's 32 over 16) repeats each
key head's q and k across its value heads before `gdn_train`, as transformers'
`repeat_interleave` does, and sums their gradients back over the group
(`tiny_grouped_gdn_step_matches_transformers_autograd`: two key heads over
four value heads, every gradient within 3.6e-6 of transformers' in exact
f32). The forward is the inference
forward's except where a backward needs more: `gdn_gates` + `gdn_train`
(checkpointed state) for the gated delta rule and `attn_train` (log-sum-exp)
for attention. The forward keeps only the residual stream into each layer
(`T x hidden` f32) and reruns one layer's forward just before its backward,
skipping that rerun's `down` projection, whose output the backward never
reads. The kernels are deterministic, so the rebuilt intermediates are the
forward's bits. The backward walks the layers in reverse (MLP, post norm, mixer, input norm, the residual stream's
gradient accumulating through both norms), then adds the embedding's
gradient onto the tied head's `[vocab, hidden]` gradient that the
cross-entropy wrote. Every fused projection's gradient is filled by disjoint
writers (for GDN: the conv's q, k, v columns, the gated norm's z, the gates'
a and b; for attention: the Q/K backward's q, k, v and the output gate's
gate columns) and multiplied out once. Gradients are the same bits on every
run.

`train_step(ids, operands)` takes the GEMMs' operand precision
(`gemm::GemmOperands`): `ExactF32`, or `Bf16`, which rounds each GEMM's
operands to bf16 and accumulates in f32 (the `gemm_bf16` / `gemm_tn_bf16` /
`gemm_nt_bf16` lane, the one `PrecisionMode::Bf16` selects for the
`*_train` GEMMs, chosen per call instead of runtime-wide). Everything else
stays f32: the weights, the activations kept and rebuilt, the gradients, and
every non-GEMM kernel; the cross-entropy's four GEMMs take the same choice.
On the tiny model, `Bf16` gives a loss within 7.3e-5 of transformers' f32 and
gradients within 1.4e-2 (matrices) and 2.3e-2 (the 1-D norms, `A_log`,
`dt_bias`, each one sum over every token) of each parameter's largest
(bounds 2^-8 and 2^-5, set before the first run for this two-layer, 70-token
fixture only: the f32 step already drifts with depth, so a 24-layer or
T = 2048 check sets its own bound before it runs); two runs are the same
bits, and neither is the `ExactF32` step's. At the 2B's shapes and
T = 2048, the cross-entropy with gradients takes 561 ms on bf16 operands
against 1768 ms exact (`bench_qwen35_train --bf16`, two runs each, 3.15x);
the whole 2B step takes 5.94 s on bf16 operands against 14.51 s exact
(2.44x; 345 against 141 tokens/s; peak footprint 25.8 / 25.5 GB), measured
back to back on battery in low-power mode, which made both about 2x slower
than the plugged-in 7.2 s above (`bench/results/qwen35_train_step_bf16_m5pro.txt`).
The weights are still cast to bf16 at every GEMM; a per-step bf16 copy was
to be built only if the ratio came out below ~1.7x, and it did not. Against transformers' float32 2B reference (the ignored
`real_2b_step_on_bf16_operands_stays_near_transformers`, the 128 tokens and
149 tensors of the f32 check) it gives a loss within 1.33e-4 and gradients
within 2.2e-2 (matrices) and 3.6e-2 (1-D norms, `A_log`) of each
parameter's largest, under bounds of 2^-7 and 2^-4 written before the run;
the exact step's gap there is 3.9e-3. Its process peaked at 22.3 GB
resident (`/usr/bin/time -l`).

`tests/qwen35_train.rs` checks it against transformers' own autograd on a
committed tiny `Qwen3_5ForCausalLM` of the 2B's shape family (one GDN and one
attention layer, T = 70 across a GDN chunk and attention blocks; bf16
weights, f32 arithmetic on both sides): the loss within 4.3e-8 relative,
and all 27 parameter gradients within 3.1e-6 of each parameter's largest
magnitude (bound 1e-4). The training forward's loss equals the loss of the
inference forward's logits. Fourteen of fifteen injected composition
defects (dropped accumulations, swapped gradient windows, a skipped
embedding add, wrong norm inputs, the loss shift, the layer order) fail it;
the fifteenth, accumulating the final norm's gradient into the freshly
zeroed residual gradient, cannot change it.
On Qwen3.5-2B-Base (`make_train_fixture.py 2b`, then the ignored
`real_2b_step_matches_transformers`; 128 tokens of the parity prompt, every
1-D and conv parameter, layers 0 and 3 in full, 149 tensors and the used
embedding rows): the training loss equals the inference forward's to 1.7e-7
and transformers' to 4.6e-5, and the gradients agree with transformers' to
at most 3.9e-3 of a parameter's largest (median about 5e-4). That is well
above transformers' own run-to-run disagreement (3.7e-5 worst, SDPA against
eager attention and another thread count, `tools/qwen35_ref/train_noise_floor.py`),
so it was not waved through as float32 noise. It is the forward's: tessl's
and transformers' f32 forwards differ (logits by 1.9e-6 of the largest,
above), and the gradients of two slightly different functions differ by
more. The unit test `real_2b_gradients_are_those_of_tessls_forward` shows
which side each disagreement is on: along `v = (g_tessl - g_torch) / |d|`,
where the two gradients predict slopes `|d|` apart, a Richardson-extrapolated
central difference of tessl's own loss lands on tessl's gradient for every
direction it can resolve: within 0.04-0.21 |d| for layer 20's conv weight,
layer 0's MLP gate and down projections and layer 3's attention output
projection (0.79-1.04 |d| from transformers'), and within 0.03-0.24 |d| for
the final norm and four of the worst layer norms (layers 6, 8, 22, 23;
0.86-1.2 |d| from transformers'), which take steps of 0.05-0.4 along the unit
direction since they act as `1 + w` around 1. Three stay unresolved and are
not claimed either way: `A_log` and `dt_bias` (|d| ~1e-5, at the loss's
rounding), and the GDN gated norm's 128 weights, whose two step scales
extrapolate further apart than its |d|. The 2B test's bounds (loss 1e-4,
gradients 1e-2) were set after that first run, for those reasons.

Depth alone, in a setting with nothing else different
(`make_train_fixture.py tiny --layers 24`, the ignored
`deep_tiny_step_matches_transformers`): the tiny model 24 layers deep, in the
2B's layer pattern, agrees with transformers to 2.7e-5 on every tensor of
more than one element (up from ~3e-6 at two layers) and to 9.6e-4 on the
one-head model's single-element `A_log` / `dt_bias` sums. So depth grows
the disagreement tenfold but not to the 2B's 4e-3, whose remaining factor
comes with the real model's scale (hidden 2048, trained weights); the finite
differences above are what place it in the forward.

### Training on bf16 storage

A `Precision::Bf16` model (`load_tower`) trains on `GemmOperands::Bf16`
with its matrices, its gradient bank's matrices and each layer's saved input
in bf16; every accumulation, the GDN state, the softmax and the log-sum-exp
stay f32, as do the norms, conv and gate parameters. The forward runs on the
f32 residual stream and keeps a bf16 copy of each layer's input; the backward
rebuilds each layer from that copy. (Running the forward on the stream
rounded at every boundary instead compounded the rounding through the layers
above: on the 2B it put layer 0's `dt_bias` gradient 1.1e-1 from the f32
step's, against 2.7e-2 now.) `AdamWConfig` picks the update rule
(`F32Master`, `Bf16Kahan`, `Bf16Stochastic { seed }`) and the moments
(`F32`, `Bf16`, `Block8`); `AdamW::describe` records both.

Parity, bounds written before each first run:
`tiny_step_on_bf16_storage_stays_near_transformers` and
`tiny_grouped_gdn_step_matches_transformers_autograd` (loss 2^-7, gradients
2^-4 of their peak, against transformers' f32 autograd) pass;
`real_2b_step_on_bf16_storage_stays_near_the_f32_step` passes (loss 2.1e-4,
worst gradient 4.5e-2 against tessl's exact-f32 step on 512 natural-text
tokens). `real_2b_loss_curve_tracks_the_f32_step` (three seeds, 30 steps
each on a fresh 128-token chunk, four chunks held out) fails, with its bounds
kept as written. Every stored precision's held-out loss is within 1% of
f32's at every scoring (worst 0.98%), and stochastic rounding's training
loss within 0.74% at every step. The round-to-nearest rules' training loss
exceeds the 2% per-step bound on 15–16 of 90 steps (worst 4.6%): their
forward runs on rounded weights, which hold back updates under half a bf16
ulp, as an f32 model on bf16 GEMM operands does. Its fall bound fails because
the f32 run's own held-out loss rises on one seed. Two earlier designs that
repeated their data measured overfitting instead and were replaced; the
test's comment records them.

Memory at Qwen3.5-4B's shapes (`probe_storage_memory`, random weights;
`bench/results/qwen35_4b_storage_memory_m5pro.txt`), against the M5 Pro's
48 GiB recommended working set, every buffer at its allocated size:

| Weights + bank | Update | Moments | Resident | Fits |
|---|---|---|---|---|
| f32 | f32 | f32 | 62.67 GiB (computed) | no |
| bf16 | f32 master | f32 | 62.67 GiB (computed) | no |
| bf16 | f32 master | 8-bit | 39.29 GiB | yes |
| bf16 | Kahan | bf16 | 39.17 GiB | yes |
| bf16 | Kahan | 8-bit | 31.46 GiB | yes |
| bf16 | stochastic | 8-bit | 23.63 GiB | yes |

With Kahan and 8-bit moments resident, one training step plus one AdamW step
on the 4B peaks at 35.28 GiB at T = 512 and 37.35 GiB at T = 2048 (the
steps' own peaks 3.82 and 5.88 GiB, under their bounds of 6.29 and
9.16 GiB).

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

- **Training, remaining gaps.** The LM-head cross-entropy and its gradients
  exist (`tessl::cross_entropy`, over the supervised rows only, in vocabulary
  chunks), and every layer has a backward, so `train_step` runs the whole
  forward and backward on the GPU (torch sees it through
  `tessl_torch.Qwen35`). A batch runs one sequence at a time: each row
  trimmed to its length (exact for right padding, which a causal model never
  attends), its gradients accumulated in a bank, so a batch of `B` rows costs
  `B` steps rather than one padded one (throughput not measured). What it
  does not do yet: several sequences in one step.
- **Exact f32 at the MLP up/gate shape.** Production's exact-f32 NN runs
  2048 x 6144 x 2048 at 4.1 TFLOP/s against 5.7 at 2048^3, and a
  register-accumulator 128x64 sg8 tile runs it 1.45x faster with the same
  bits (`bench/results/f32_exact_coop_m5pro.txt`). Everywhere else that tile
  is within 0-11%, so exact f32 stays on its 32x32 kernel; why the one shape
  is slow was not diagnosed.
- **Two compositions of the same model.** `bench_qwen35_layers` (random
  weights, timing) and `qwen35_model` (real checkpoint, parity) each wire the
  layers; the bench should run on `Qwen35Model` so a wiring fix lands once.
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
- **Other model sizes are parsed, not validated.**
  `Qwen35Config::from_config_json` reads any Qwen3.5 text `config.json` and
  refuses by name what the forward lacks (untied embeddings, MoE, attention
  bias, an ungated output, non-SiLU activations, `mlp_only_layers`,
  non-default RoPE, a key head dim other than 128, an attention head dim
  other than 256). The published 2B config parses to exactly the config
  `tests/qwen35_model.rs` checks. No other size has been run against
  transformers with its weights, so for those sizes only parsing is checked,
  not the forward.
- **Redundant work left in.** The normalized-k/q workspace is stored per
  value head, so it is duplicated `Hv/Hk` times: 1× for the 2B (16 key /
  16 value heads), 2× at `Qwen3_5TextConfig()`'s defaults†. The prep
  products are uneven across simdgroups (triangular). Both are
  performance-only.
- **Untuned tiles.** GDN's 64-row chunks and 32-column value slices, and the
  prefill attention's tile (`ATTN_PREFILL_TILE`, 32x32), are first
  choices, not swept ones. Four attention tiles are compiled for the sweep.
