# Qwen3.5 kernels

`tessl::qwen35` holds the Metal kernels for Qwen3.5's layers that transformers
has no fast Mac path for. Sources: `kernels/qwen35_gdn.metal`,
`kernels/qwen35_attn.metal`, `kernels/qwen35_score.metal`.

> **Status: compiled by Apple's Metal compiler; not yet run on a GPU.** The
> `Metal compile` workflow (GitHub-hosted macOS, Xcode 26.6) builds all three
> sources under `-std=metal4.0 -Wall -Werror`, links them, confirms all thirteen
> entry points are exported, and builds the crate and every test target with
> no `metal3.2` fallback. Its first run caught one diagnostic, an unused
> constant, which was fixed. Every
> kernel below was compiled as C++ and executed on a CPU emulator of the Metal
> execution model, then compared against transformers' own Qwen3.5 code (see
> [Verification](#verification)), including under AddressSanitizer and
> ThreadSanitizer, in shuffled threadgroup order, and with fast-math-like
> error injected. Nothing has run on a GPU yet:
> `cargo test --release --test qwen35_kernels -- --test-threads=1` on a Mac is
> the first time that happens.

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
| 6c. Shared-prefix attention | `qwen35_attn_prefix_rows` (+ `slot_base` in 6/6b) | `attn_prefix_rows`, `attn_qk_norm_rope_suffix` | attention over a per-row copy of a shared KV prefix, without the copy |
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
attn_prefix_rows(q, prefix, suffix, suffix_len, q_pos)
```

`attn_qk_norm_rope_suffix` rotates each token at its absolute position
`P + s0 + t` and stores it at slot `s0 + t`. Underneath, it is
`qwen35_attn_qk_norm_rope` with `slot_base = P`; slot_base 0 is the
ordinary cache. `qwen35_attn_prefix_rows` is `flash_attn_rows`' D=256
instantiation (R=16, 32 simdgroups) with only the address of key `t` changed:
the prefix below `P`, at batch stride 0, then the row's suffix. A masked key is
an exact no-op in the online softmax, so it returns **the same bits** as
`nn::flash_attn_rows` over each row's copied `prefix ‖ suffix`. The tests hold
it to exactly that.

The live suffix length is one device `u32` shared by every row, like
`flash_attn_rows`' `tkv`, so the questions in a batch have equal lengths (pad
the short ones and ignore their padded rows). At B = 16 and P = 8k, the copy
this avoids is 16 × 8k × 2 KV heads × 256 × 4 B = 268 MB per K or V per layer,
or 3.2 GB across the 6 layers.

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

Last run, 84 checks passing. GDN rows show the kernel's max error next to
transformers' own fp32 error, both measured against f64:

| Case | kernel err | transformers fp32 err |
|---|---:|---:|
| chunk, T=1 | 1.1e-8 | 1.9e-9 |
| chunk, T=64 | 1.5e-8 | 3.6e-8 |
| chunk, T=65, grouped heads, B=2 | 5.8e-8 | 6.0e-8 |
| chunk, T=130, 4 heads, per-batch state | 4.9e-8 | 8.6e-8 |
| chunk, T=100, Dv=128, shared snapshot, B=3 | 2.9e-8 | 2.9e-8 |
| chunk, T=150, strong decay | 3.9e-7 | 3.2e-7 |
| chunk, T=130, `a + dt_bias` in the softplus-series range | 5.4e-8 | 5.8e-8 |
| chunk, T=200, Dv=128 | 2.8e-8 | 2.7e-8 |
| chunk, T=1000 | 2.4e-8 | 2.9e-8 |
| chunk, T=4096 (64 chunks through one state) | 3.3e-8 | 3.1e-8 |
| chunk at Qwen3.5's head counts (16 key / 32 value heads, Dv=128), per-batch state | 6.5e-8 | — |
| recurrent at Qwen3.5's head counts, shared snapshot | 8.9e-9 | — |
| recurrent, T=1, snapshot, B=4 | 5.8e-9 | 8.3e-9 |
| recurrent, T=7, per-batch state | 5.7e-9 | 6.5e-9 |
| chunk / recurrent / conv, T=0 with state_out | state copied exactly | — |
| chunk workspace: `W(I+A) = I` | 5.4e-8 | — |
| conv1d (prefill, state, snapshot, T < KW−1) | 4.8e-7 / state exact | — |
| gated norm f32 / bf16 | 1e-6 / within one bf16 rounding | — |
| Q/K norm + partial RoPE, D=256, pos 30000 | 9.5e-7 | — |
| output gate f32 / bf16 / in place | 2.4e-7 / exact / 2.4e-7 | — |
| scoring f32 / bf16; a bad slot or answer → NaN, the rest intact | 4.8e-7 | — |
| Q/K norm + RoPE with `slot_base`: absolute RoPE, relative slot | bit-identical to slot_base 0 | — |
| shared-prefix attention, 8 query / 2 KV heads of 256; P = 0, 1, 30 (no suffix), 65 | bit-identical to `flash_attn_rows` on a copied prefix; ≤ 5.0e-7 vs torch | — |

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
fails the second. There is also a
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

## Not done

- **Training.** No backward kernels. Autograd through transformers' loops is
  what made 2k-token fine-tuning hit 48 GB. That's a Mac-training problem and
  is out of scope here.
- **bf16 inputs.** The kernels read f32 activations, which is what tessl's GEMM
  writes. A bf16-activation variant would halve their read traffic.
- **Shared-prefix attention, remaining gaps.** `attn_prefix_rows` is the
  row-parallel kernel only. A single-token step over a long prefix has
  B·H threadgroups, each walking all P + S keys serially; a split-KV
  (`flash_attn_decode`-style) variant would parallelize that, and nothing
  has measured whether it is needed. Suffix lengths are equal across a batch
  (one device `u32`). Only head_dim 256 is compiled, and there is no `_posbuf`
  form of `attn_qk_norm_rope_suffix` for ICB replay.
- **mRoPE with image positions.** Text positions only. For text, the three mRoPE
  streams are equal and the rotation reduces to plain RoPE.
- **Key head dim other than 128**, and value head dims that aren't multiples of
  32, are rejected on the host.
- **Redundant work left in.** The normalized-k/q workspace is stored per
  value head, so it is duplicated `Hv/Hk` times (2× for Qwen3.5). The prep
  products are uneven across simdgroups (triangular). Both are
  performance-only.
- **Performance is unmeasured.** The design targets launch count first: one
  GEMM, one conv, two GDN dispatches and one norm per GDN layer, against
  thousands. Tile sizes (64-row chunks, 32-column value slices) are first
  choices, not tuned ones.
