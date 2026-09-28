# Qwen3.5 kernels

`tessl::qwen35` holds the Metal kernels for Qwen3.5's layers that transformers
has no fast Mac path for. Sources: `kernels/qwen35_gdn.metal`,
`kernels/qwen35_attn.metal`, `kernels/qwen35_score.metal`.

> **Status: built and checked off-device, not yet run on Apple silicon.** Every
> kernel below was compiled as C++ and executed on a CPU emulator of the Metal
> execution model, then compared against transformers' own Qwen3.5 code (see
> [Verification](#verification)). The Rust wrappers type-check and pass clippy
> for `aarch64-apple-darwin`. Nothing here has been through the Metal compiler
> or a GPU yet. `cargo test --release --test qwen35_kernels -- --test-threads=1`
> on a Mac is the first time either happens.

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
| 7. Score only the answer rows | `qwen35_score_rows_{f32,bf16}` | `score_answer_rows` | final norm + `lm_head`, restricted to the answer tokens |
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
a 128×32 state slice held in 27 KB of threadgroup memory. A single fused kernel
would have put 63 dependent steps of solve on the sequential critical path of
every chunk.

The prep → scan hand-off goes through a `GdnWorkspace`. That's about 100 MB for
a 2048-token prefill at `Hv = 32`, and it scales linearly with tokens. Allocate
it once and reuse it across layers. Very long prompts can be split into segments
and carried through `StateIn::PerBatch` + `state_out` to bound it; the result is
identical.

### Many questions from one prefilled snapshot

`StateIn::Snapshot(buf)` makes every batch row start from the same state and
**never writes it**. This works for `gdn_recurrent` and `gdn_chunk_forward`
(the recurrence) and for `conv1d_silu` (the conv history). So one shared context
is prefilled once, and N questions run as a batch of N continuations, each
writing its own `state_out` if it wants one. The attention layers' KV cache is
not shared this way yet (see [Not done](#not-done)).

### Scoring only the answer rows

`score_answer_rows` applies the final norm (`w_offset = 1.0` for Qwen3.5's
zero-centred `Qwen3_5RMSNorm`) and dots each slot row with just the answer
tokens' LM-head rows. It returns the logits and a log-softmax over the answer
set. That softmax is the distribution restricted to those tokens, not the
full-vocabulary log-probability, which needs every row. An out-of-range slot or
token id produces NaN rather than an out-of-bounds read.

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
- **Partial RoPE is not `rms_qkv_rope`'s.** transformers pairs `p` with
  `p + rotary_dim/2` and uses `θ^(−2p/rotary_dim)`. tessl's existing kernel
  implements Gemma's proportional RoPE (pairs across `D/2`, denominator `D`).
  At `rotary_dim = 64` of 256, the two agree on nothing.

## Verification

### Off-device: `tools/msl_emu`

`python3 tools/msl_emu/check_qwen35.py` (needs torch and transformers) builds the
kernel sources as C++20 against a CPU stand-in for `<metal_stdlib>`. It launches
every threadgroup as real threads with real barriers and simdgroup collectives,
using the dispatch geometry `src/qwen35.rs` uses. The results are compared
against transformers' Qwen3.5 functions, called directly, and against an f64
sequential recurrence. See [tools/msl_emu/README.md](../tools/msl_emu/README.md)
for what the emulator does and does not model.

Last run, all checks passing. GDN rows show the kernel's max error next to
transformers' own fp32 error, both measured against f64:

| Case | kernel err | transformers fp32 err |
|---|---:|---:|
| chunk, T=1 | 1.1e-8 | 1.9e-9 |
| chunk, T=64 | 1.5e-8 | 3.6e-8 |
| chunk, T=65, grouped heads, B=2 | 5.8e-8 | 6.0e-8 |
| chunk, T=130, 4 heads, per-batch state | 4.9e-8 | 8.6e-8 |
| chunk, T=100, Dv=128, shared snapshot, B=3 | 2.5e-8 | 2.9e-8 |
| chunk, T=150, strong decay | 3.9e-7 | 3.2e-7 |
| chunk, T=200, Dv=128 | 2.8e-8 | 2.7e-8 |
| recurrent, T=1, snapshot, B=4 | 5.8e-9 | 8.3e-9 |
| recurrent, T=7, per-batch state | 5.6e-9 | 6.5e-9 |
| chunk workspace: `W(I+A) = I` | 5.4e-8 | — |
| conv1d (prefill, state, snapshot, T < KW−1) | 4.8e-7 / state exact | — |
| gated norm f32 / bf16 | 1e-6 / within one bf16 rounding | — |
| Q/K norm + partial RoPE, D=256, pos 30000 | 9.5e-7 | — |
| output gate f32 / bf16 / in place | 2.4e-7 / exact / 2.4e-7 | — |
| scoring f32 / bf16, out-of-range → NaN | 4.8e-7 | — |

**The checks catch defects.** Mutations injected into the GDN kernel were each
caught: dropping the inter-chunk decay, using the wrong state-update decay,
halving the solve's quad reduction, removing the `simdgroup_barrier` before a
stage read (surfaces as a race), and skipping k's normalization in the
recurrent kernel. The one survivor (`j < i` → `j <= i` in building `A`) is
equivalent, because the solve never reads the diagonal slot.

### On device: `tests/qwen35_kernels.rs`

Rust f64 references in `tests/common/qwen35.rs`, held to transformers-generated
goldens in `tests/fixtures/qwen35/` (`scripts/gen_qwen35_fixtures.py`). The GDN
reference matches the f64 golden to 1e-16. The RoPE reference matches
transformers to 7e-7. Both were checked off-device. The GPU tests cover every
kernel, including randomized chunk-edge shapes, grouped heads, snapshots,
strided windows, and a chunked prefill followed by in-place recurrent decode
steps that must equal the recurrence over the whole sequence. They also check
the projection packing through tessl's GEMM and the host-side rejections.

```sh
cargo test --release --test qwen35_kernels -- --test-threads=1
cargo test --release --test shader_index_arithmetic   # includes the qwen35 sources
```

## Not done

- **Training.** No backward kernels. Autograd through transformers' loops is
  what made 2k-token fine-tuning hit 48 GB. That's a Mac-training problem and
  is out of scope here.
- **bf16 inputs.** The kernels read f32 activations, which is what tessl's GEMM
  writes. A bf16-activation variant would halve their read traffic.
- **Shared-prefix attention.** The snapshot flow shares GDN and conv state, but
  the 6 full-attention layers' KV cache still has a batch dimension. Many
  questions over one prefix need either the prefix copied per row or an
  attention kernel with a batch-stride-0 prefix.
- **mRoPE with image positions.** Text positions only. For text, the three mRoPE
  streams are equal and the rotation reduces to plain RoPE.
- **Key head dim other than 128**, and value head dims that aren't multiples of
  32, are rejected on the host.
- **Performance is unmeasured.** The design targets launch count first: one
  GEMM, one conv, two GDN dispatches and one norm per GDN layer, against
  thousands. Tile sizes (64-row chunks, 32-column value slices) are first
  choices, not tuned ones.
