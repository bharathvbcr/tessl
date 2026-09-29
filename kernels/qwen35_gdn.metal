// Qwen3.5 gated delta net (GDN) layer kernels.
//
// Qwen3.5's linear-attention layers, as transformers' `modeling_qwen3_5.py`
// defines them (`Qwen3_5GatedDeltaNet.forward`):
//
//   P            = x @ [W_qkv | W_z | W_b | W_a]          (one fused GEMM)
//   qkv          = silu(causal_conv1d(P[:, :conv_dim]))    -> qwen35_conv1d_silu
//   q, k         = l2norm(q), l2norm(k);  q *= Dk^-0.5     -> folded into the loads below
//   g            = -exp(A_log) * softplus(a + dt_bias)     -> folded into the loads below
//   beta         = sigmoid(b)                              -> folded into the loads below
//   o            = gated_delta_rule(q, k, v, g, beta)      -> chunk_prep + chunk_scan (prefill)
//                                                             or qwen35_gdn_recurrent (decode)
//   y            = rms_norm(o) * w * silu(z)               -> qwen35_gated_rms_norm_*
//   out          = y @ W_out
//
// Without these kernels transformers runs the delta rule on the Mac as a torch
// fallback: a Python loop over each 64-token chunk with a 63-step triangular
// solve inside, roughly 13-16k tiny launches per forward pass, all in fp32.
//
// # The recurrence
//
// Per value head, with the state S laid out [Dk, Dv] as transformers stores it:
//
//   S_t = exp(g_t) * S_{t-1}
//   S_t = S_t + k_t (beta_t * (v_t - S_t^T k_t))^T      (the correction reads the *decayed* state)
//   o_t = S_t^T q_t
//
// That is `Rule::Published` in tests/common/gdn.rs with alpha = exp(g), and no
// clamp: Qwen3.5 does not clamp its gates, and exp(g) with g <= 0 is in (0, 1].
//
// # The chunked form (prefill)
//
// With C = 64 rows per chunk, G_i the chunk-local cumulative sum of g, and
// Gamma_ij = exp(G_i - G_j) for j <= i:
//
//   A   = strict_lower(beta_i * (k_i . k_j) * Gamma_ij)        [C, C]
//   W   = (I + A)^-1                                           [C, C]   (the 63-step solve)
//   X   = beta * (V - exp(G) * (K S))                          [C, Dv]
//   U   = W X                                                  [C, Dv]  ("v_new")
//   O   = exp(G) * (Q S) + (lower_incl((q_i . k_j) Gamma_ij)) U
//   S'  = exp(G_last) S + K^T (exp(G_last - G) * U)
//
// which is transformers' `torch_chunk_gated_delta_rule` with its `k_cumdecay`
// term reassociated: W (beta e^G K) S == W (beta e^G (K S)), so the [C, Dk]
// product `k_cumdecay` is never formed. Every exponent above is <= 0, so no
// intermediate can overflow however strong the decay.
//
// Everything that does not depend on S — the norms, the gates, both C x C
// products and the solve — lives in `qwen35_gdn_chunk_prep`, which runs one
// threadgroup per (batch, head, chunk) with no ordering between chunks. Only
// `qwen35_gdn_chunk_scan` walks the chunks in order, and its per-chunk work is
// four simdgroup-matrix products against a state held in threadgroup memory.
// The solve is therefore off the sequential critical path entirely.
//
// # Layout contract
//
// Activations are row-major f32 with one row per token, row index `b*T + t`,
// and every input is addressed as a column window of a wider row (`ld`, `off`)
// so the kernels read the fused projection GEMM's output in place. Heads are
// contiguous within a window: q head `hk` is columns [hk*DK, hk*DK + DK).
// Grouped heads follow transformers' `repeat_interleave`: value head `h`
// reads key head `h / (Hv / Hk)`.
#include <metal_stdlib>
#include "qwen35_act.h"
using namespace metal;

/// Key head dim. Qwen3.5's `linear_key_head_dim` is 128 at every published size;
/// the host rejects anything else rather than launching a kernel whose tiles
/// would silently read the wrong columns.
constant uint GDN_DK = 128u;
/// Rows per chunk. transformers' default `chunk_size`, and the size that makes
/// the solve block 64x64 f32 = 16 KB.
constant uint GDN_C = 64u;
/// Value columns one scan/decode threadgroup owns.
constant uint GDN_BV = 32u;
/// Threads per threadgroup the prep kernel is written for: a quad of lanes per
/// column of the C x C solve, and 8 simdgroups of 8 rows each.
constant uint GDN_PREP_THREADS = 256u;
/// Threads per threadgroup of the scan and recurrent kernels: 4 simdgroups,
/// owning 2 row blocks of each chunk (scan) or 32 key rows (recurrent) apiece.
constant uint GDN_SCAN_THREADS = 128u;

// The lane mappings below are written against these shapes; a change to one
// constant that forgot the others must fail to compile, not mis-index.
static_assert(GDN_PREP_THREADS == 4u * GDN_C, "prep: one quad of lanes per solve column");
static_assert(GDN_PREP_THREADS / 32u * 8u == GDN_C, "prep: one simdgroup per 8-row block");
static_assert(GDN_SCAN_THREADS / 32u * 16u == GDN_C, "scan: two 8-row blocks per simdgroup");
static_assert(GDN_SCAN_THREADS == GDN_DK, "recurrent: 32 key rows per simdgroup");
static_assert(GDN_BV == 32u, "scan and recurrent: one value column per lane, 4 column tiles");
/// transformers' `l2norm` epsilon, added to the sum of squares (not the mean).
constant float GDN_L2_EPS = 1e-6f;

/// torch's `F.softplus` at its defaults (beta = 1, threshold = 20): linear above
/// the threshold, `log1p(e^x)` below it.
///
/// MSL has no `log1p`, and `log(1 + e)` is not a substitute where it matters:
/// rounding `1 + e` alone costs `2^-24 / e` relative, and the fast `log` near 1
/// has an absolute error around 2^-21 — together 1-60% of the result for x in
/// [-15, -8], which `a + dt_bias` reaches routinely (Qwen's dt_bias sits around
/// -2 to -7). Below -3 the series `e - e^2/2 + ... - e^8/8` is exact to f32
/// (the first dropped term is 1e-11 relative); above it `1 + e` loses at most
/// 1.2e-6 relative, and `precise::log` adds half an ulp.
inline float gdn_softplus(float x)
{
    if (x > 20.0f) return x;
    // precise: fast `exp` is documented to 3 + floor(2|x|) ulp, ~33 ulp at
    // x = -15, and below -3 softplus is e^x to within the series. Once per
    // token per head, so the cost is nothing.
    const float e = precise::exp(x);
    if (x < -3.0f) {
        // log1p(e) by Horner: e * (1 - e/2 + e^2/3 - ... - e^7/8).
        float p = -1.0f / 8.0f;
        p = p * e + 1.0f / 7.0f;
        p = p * e - 1.0f / 6.0f;
        p = p * e + 1.0f / 5.0f;
        p = p * e - 1.0f / 4.0f;
        p = p * e + 1.0f / 3.0f;
        p = p * e - 1.0f / 2.0f;
        p = p * e + 1.0f;
        return e * p;
    }
    return precise::log(1.0f + e);
}

/// `g = -exp(A_log) * softplus(a + dt_bias)`: the log of the per-step decay.
/// Always <= 0.
inline float gdn_log_decay(float a, float a_log, float dt_bias)
{
    return -exp(a_log) * gdn_softplus(a + dt_bias);
}

// ------------------------------------------------------------------ conv1d ---

/// Position `pos` of the extended sequence `[state (hist slots), x (T rows)]`
/// for one channel. `xc` points at that channel of row 0, `st` at its state.
inline float conv_ext(
    device const float *xc,
    device const float *st,
    uint ld_x,
    uint hist,
    bool has_state,
    uint pos)
{
    if (pos < hist) {
        return has_state ? st[pos] : 0.0f;
    }
    return xc[(ulong)(pos - hist) * ld_x];
}

/// Depthwise causal conv1d over time, then SiLU. transformers'
/// `causal_conv1d_fn` (prefill) and `causal_conv1d_update` (decode) in one.
///
/// Input channel `c` of token `(b, t)` is `x[(b*T + t) * ld_x + x_off + c]`.
/// The conv looks back `KW - 1` steps; before `t = 0` it reads `state_in`
/// (`[.., C, KW-1]`, oldest first, batch stride `state_bstride` elements — 0
/// broadcasts one snapshot to every batch row) when `flags & 1`, else zeros.
/// When `flags & 2` the last `KW - 1` inputs of each sequence are written to
/// `state_out` (`[B, C, KW-1]`), so a later call can continue from them.
///
/// **Ragged rows.** With `flags & 4`, row b is `min(seq_lens[b], T)` tokens
/// long (rows stay `T` apart): outputs past it are not written, and the state
/// is the last `KW - 1` inputs before it, exactly as a call with that `T`.
///
/// Grid: x = channel, y = `T + KW - 1` (the first T are outputs, the rest are
/// state slots), z = batch.
kernel void qwen35_conv1d_silu(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device const float *state_in [[buffer(2)]],
    device float *y [[buffer(3)]],
    device float *state_out [[buffer(4)]],
    constant uint &B [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &C [[buffer(7)]],
    constant uint &KW [[buffer(8)]],
    constant uint &ld_x [[buffer(9)]],
    constant uint &x_off [[buffer(10)]],
    constant uint &state_bstride [[buffer(11)]],
    constant uint &flags [[buffer(12)]],
    device const uint *seq_lens [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    const uint e = gid.y;
    const uint b = gid.z;
    const uint hist = KW - 1u;
    if (c >= C || e >= T + hist || b >= B) return;
    // Row b's live length; the row stride stays T.
    const uint Tb = (flags & 4u) != 0u ? min(seq_lens[b], T) : T;

    const bool has_state = (flags & 1u) != 0u;
    device const float *st = state_in + (ulong)b * state_bstride + (ulong)c * hist;
    device const float *xc = x + (ulong)b * T * (ulong)ld_x + x_off + c;

    if (e < T) {
        if (e >= Tb) return;
        float acc = 0.0f;
        for (uint j = 0u; j < KW; ++j) {
            acc += w[(ulong)c * KW + j] * conv_ext(xc, st, ld_x, hist, has_state, e + j);
        }
        y[((ulong)b * T + e) * (ulong)C + c] = qwen35_silu(acc);
    } else if ((flags & 2u) != 0u) {
        const uint j = e - T;
        state_out[((ulong)b * C + c) * hist + j] = conv_ext(xc, st, ld_x, hist, has_state, Tb + j);
    }
}

// ------------------------------------------------------------- chunk prep ---

/// Row stride of the prep kernel's C x C threadgroup block. Odd, so the solve's
/// column walks (`blk[j * LD + n]` over j) and the tile stores touch distinct
/// memory banks; at 64 a whole column sits in one bank.
constant uint GDN_BLK_LD = GDN_C + 1u;
/// Threadgroup memory for `qwen35_gdn_chunk_prep`, in floats: one C x BLK_LD
/// block plus four per-row arrays.
constant uint GDN_PREP_TG_FLOATS = GDN_C * GDN_BLK_LD + 4u * GDN_C;
static_assert(GDN_PREP_TG_FLOATS * 4u <= 32768u, "prep threadgroup memory exceeds 32 KB");

/// Per (batch, value head, chunk): gates, norms, both C x C products, the solve.
///
/// Writes, for its chunk (rows past T are zero-padded, with beta = g = 0, which
/// makes them inert exactly as transformers' `F.pad` does):
///
///   ws_k    [B, Hv, Tp, DK]      l2-normalized k
///   ws_q    [B, Hv, Tp, DK]      l2-normalized q * DK^-0.5
///   ws_g    [B, Hv, Tp]          chunk-local cumulative log decay G
///   ws_beta [B, Hv, Tp]          sigmoid(b)
///   ws_w    [B, Hv, NC, C, C]    (I + A)^-1, unit lower triangular
///   ws_aq   [B, Hv, NC, C, C]    lower_incl((q_i . k_j) Gamma_ij)
///
/// with Tp = NC * C and NC = ceil(T / C).
///
/// 256 threads = 8 simdgroups. Grid: x = chunk, y = value head, z = batch.
kernel void qwen35_gdn_chunk_prep(
    device const float *qkv [[buffer(0)]],
    device const float *ab [[buffer(1)]],
    device const float *a_log [[buffer(2)]],
    device const float *dt_bias [[buffer(3)]],
    device float *ws_k [[buffer(4)]],
    device float *ws_q [[buffer(5)]],
    device float *ws_g [[buffer(6)]],
    device float *ws_beta [[buffer(7)]],
    device float *ws_w [[buffer(8)]],
    device float *ws_aq [[buffer(9)]],
    constant uint &T [[buffer(10)]],
    constant uint &Hk [[buffer(11)]],
    constant uint &Hv [[buffer(12)]],
    constant uint &ld_qkv [[buffer(13)]],
    constant uint &q_off [[buffer(14)]],
    constant uint &k_off [[buffer(15)]],
    constant uint &ld_ab [[buffer(16)]],
    constant uint &a_off [[buffer(17)]],
    constant uint &b_off [[buffer(18)]],
    device const uint *seq_lens [[buffer(19)]],
    constant uint &use_lens [[buffer(20)]],
    threadgroup float *tgm [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint chunk = tg.x;
    const uint hv = tg.y;
    const uint b = tg.z;
    // The workspace is laid out for T; row b is live for Tb tokens (ragged
    // rows, `use_lens`), and its chunks past that are never scanned.
    const uint nc = (T + GDN_C - 1u) / GDN_C;
    const uint Tb = use_lens != 0u ? min(seq_lens[b], T) : T;
    // Uniform per threadgroup, before any barrier.
    if (chunk * GDN_C >= Tb) return;
    const ulong tp = (ulong)nc * GDN_C;
    const uint hk = hv / max(Hv / Hk, 1u);
    const uint t0 = chunk * GDN_C;
    const float q_scale = rsqrt((float)GDN_DK);

    threadgroup float *blk = tgm;                    // [C][BLK_LD]
    threadgroup float *G = tgm + GDN_C * GDN_BLK_LD; // [C]
    threadgroup float *beta = G + GDN_C;             // [C]
    threadgroup float *rk = beta + GDN_C;            // [C]
    threadgroup float *rq = rk + GDN_C;              // [C]

    // --- norms and gates: simdgroup `sg` owns rows sg*8 .. sg*8+7 -----------
    for (uint r = 0u; r < 8u; ++r) {
        const uint i = sg * 8u + r;
        const uint t = t0 + i;
        // `t` is uniform across the simdgroup, so the simd_sums below see all
        // 32 lanes on both sides of this branch.
        float ssq = 0.0f;
        float ssk = 0.0f;
        if (t < Tb) {
            device const float *row = qkv + ((ulong)b * T + t) * (ulong)ld_qkv;
            for (uint d = lane; d < GDN_DK; d += 32u) {
                const float qv = row[q_off + hk * GDN_DK + d];
                const float kv = row[k_off + hk * GDN_DK + d];
                ssq += qv * qv;
                ssk += kv * kv;
            }
        }
        ssq = simd_sum(ssq);
        ssk = simd_sum(ssk);
        if (lane == 0u) {
            if (t < Tb) {
                device const float *gr = ab + ((ulong)b * T + t) * (ulong)ld_ab;
                rq[i] = rsqrt(ssq + GDN_L2_EPS) * q_scale;
                rk[i] = rsqrt(ssk + GDN_L2_EPS);
                G[i] = gdn_log_decay(gr[a_off + hv], a_log[hv], dt_bias[hv]);
                beta[i] = qwen35_sigmoid(gr[b_off + hv]);
            } else {
                rq[i] = 0.0f;
                rk[i] = 0.0f;
                G[i] = 0.0f;
                beta[i] = 0.0f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        // Serial on purpose: 63 adds, and the same left-to-right order as
        // torch's `cumsum`.
        float acc = 0.0f;
        for (uint i = 0u; i < GDN_C; ++i) {
            acc += G[i];
            G[i] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- normalized operands to the workspace, zero-padded ------------------
    const ulong head = (ulong)b * Hv + hv;
    const ulong row_base = head * tp + t0;
    device float *wk = ws_k + row_base * GDN_DK;
    device float *wq = ws_q + row_base * GDN_DK;
    for (uint idx = lid; idx < GDN_C * GDN_DK; idx += GDN_PREP_THREADS) {
        const uint i = idx / GDN_DK;
        const uint d = idx % GDN_DK;
        const uint t = t0 + i;
        float kv = 0.0f;
        float qv = 0.0f;
        if (t < Tb) {
            device const float *row = qkv + ((ulong)b * T + t) * (ulong)ld_qkv;
            kv = row[k_off + hk * GDN_DK + d] * rk[i];
            qv = row[q_off + hk * GDN_DK + d] * rq[i];
        }
        wk[idx] = kv;
        wq[idx] = qv;
    }
    if (lid < GDN_C) {
        ws_g[row_base + lid] = G[lid];
        ws_beta[row_base + lid] = beta[lid];
    }
    // The products below read ws_k / ws_q back through simdgroup loads, from
    // rows other threads wrote: order device memory across the threadgroup.
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    const ulong blk_base = (head * nc + chunk) * (ulong)(GDN_C * GDN_C);
    const uint rb = sg; // this simdgroup's 8-row block of the C x C output

    // --- Aq = lower_incl(Q K^T * Gamma) --------------------------------------
    {
        // Constant trip counts with a uniform predicate, rather than `cb <= rb`
        // as the bound: the loops unroll, so `acc` is indexed by constants and
        // stays in registers instead of spilling to the stack.
        simdgroup_float8x8 acc[8];
        for (uint cb = 0u; cb < 8u; ++cb) {
            acc[cb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        for (uint kk = 0u; kk < GDN_DK / 8u; ++kk) {
            simdgroup_float8x8 qa;
            simdgroup_load(qa, wq + (ulong)(rb * 8u) * GDN_DK + kk * 8u, GDN_DK);
            for (uint cb = 0u; cb < 8u; ++cb) {
                if (cb <= rb) {
                    simdgroup_float8x8 kt;
                    simdgroup_load(kt, wk + (ulong)(cb * 8u) * GDN_DK + kk * 8u, GDN_DK,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(acc[cb], qa, kt, acc[cb]);
                }
            }
        }
        for (uint cb = 0u; cb < 8u; ++cb) {
            if (cb <= rb) {
                simdgroup_store(acc[cb], blk + rb * 8u * GDN_BLK_LD + cb * 8u, GDN_BLK_LD);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lid; idx < GDN_C * GDN_C; idx += GDN_PREP_THREADS) {
        const uint i = idx / GDN_C;
        const uint j = idx % GDN_C;
        // Tiles above the diagonal were never stored; the mask never reads them.
        ws_aq[blk_base + idx] = j <= i ? blk[i * GDN_BLK_LD + j] * exp(G[i] - G[j]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- A, stored transposed in the upper triangle --------------------------
    // K K^T is symmetric, so the upper tiles hold (k_i . k_j) at (j, i) for
    // j < i. Scaled in place there, the upper triangle *is* A^T, which leaves
    // the strict lower triangle free for W: one 16 KB block holds both.
    {
        simdgroup_float8x8 acc[8];
        for (uint cb = 0u; cb < 8u; ++cb) {
            acc[cb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        for (uint kk = 0u; kk < GDN_DK / 8u; ++kk) {
            simdgroup_float8x8 ka;
            simdgroup_load(ka, wk + (ulong)(rb * 8u) * GDN_DK + kk * 8u, GDN_DK);
            for (uint cb = 0u; cb < 8u; ++cb) {
                if (cb >= rb) {
                    simdgroup_float8x8 kt;
                    simdgroup_load(kt, wk + (ulong)(cb * 8u) * GDN_DK + kk * 8u, GDN_DK,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(acc[cb], ka, kt, acc[cb]);
                }
            }
        }
        for (uint cb = 0u; cb < 8u; ++cb) {
            if (cb >= rb) {
                simdgroup_store(acc[cb], blk + rb * 8u * GDN_BLK_LD + cb * 8u, GDN_BLK_LD);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lid; idx < GDN_C * GDN_C; idx += GDN_PREP_THREADS) {
        const uint j = idx / GDN_C; // row of the stored element
        const uint i = idx % GDN_C; // column
        if (j < i) {
            blk[j * GDN_BLK_LD + i] = beta[i] * exp(G[i] - G[j]) * blk[j * GDN_BLK_LD + i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- W = (I + A)^-1 by forward substitution, column-parallel -------------
    // Column n of W solves (I + A) w = e_n, independently of every other
    // column, so no barrier separates the 63 steps. A quad of lanes owns a
    // column and splits each step's dot product four ways; lane q handles the
    // terms j with (j - n) % 4 == q, and W[i][n] is written by the lane that
    // will read it back, so each lane only ever reads its own writes.
    {
        const uint n = lid >> 2;
        const uint q = lid & 3u;
        for (uint i = 1u; i < GDN_C; ++i) {
            float s = 0.0f;
            if (i > n) {
                for (uint j = n + q; j < i; j += 4u) {
                    const float wj = j == n ? 1.0f : blk[j * GDN_BLK_LD + n];
                    s += blk[j * GDN_BLK_LD + i] * wj; // A[i][j], from the upper triangle
                }
            }
            // Uniform across the simdgroup: every lane runs every `i`.
            s += simd_shuffle_xor(s, 1u);
            s += simd_shuffle_xor(s, 2u);
            if (i > n && ((i - n) & 3u) == q) {
                blk[i * GDN_BLK_LD + n] = -s;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lid; idx < GDN_C * GDN_C; idx += GDN_PREP_THREADS) {
        const uint i = idx / GDN_C;
        const uint j = idx % GDN_C;
        ws_w[blk_base + idx] = i > j ? blk[i * GDN_BLK_LD + j] : (i == j ? 1.0f : 0.0f);
    }
}

// ------------------------------------------------------------- chunk scan ---

/// Row stride of the scan's [.][BV] threadgroup tiles. Padded past 32 so the
/// eight rows of an 8x8 tile load do not all start in the same memory bank.
constant uint GDN_TLD = GDN_BV + 4u;

/// Threadgroup memory for `qwen35_gdn_chunk_scan`, in floats: the state slice
/// [DK][TLD], one [C][TLD] block, a 2 x 8 x 8 output stage per simdgroup, and
/// four per-row arrays for the chunk (G, beta, exp(G), exp(G_last - G)).
constant uint GDN_SCAN_TG_FLOATS =
    GDN_DK * GDN_TLD + GDN_C * GDN_TLD + (GDN_SCAN_THREADS / 32u) * 2u * 64u + 4u * GDN_C;
static_assert(GDN_SCAN_TG_FLOATS * 4u <= 32768u, "scan threadgroup memory exceeds 32 KB");
/// The same layout for `qwen35_gdn_chunk_scan_bv16`'s 16-column slices.
constant uint GDN_SCAN16_TG_FLOATS =
    GDN_DK * (16u + 4u) + GDN_C * (16u + 4u) + (GDN_SCAN_THREADS / 32u) * 2u * 64u + 4u * GDN_C;

/// The sequential pass over chunks, for one (batch, value head, 32-column
/// slice of Dv). Reads what `qwen35_gdn_chunk_prep` wrote, plus V.
///
/// The initial state is `state_in[b * state_bstride + (hv*DK + k)*Dv + v]` when
/// `flags & 1`, else zero; `state_bstride = 0` starts every batch row from one
/// shared snapshot, which is never written. When `flags & 2` the final state is
/// written to `state_out` ([B, Hv, DK, Dv]). A thread reads the same state
/// elements at the start that it writes at the end, so `state_out` may be
/// `state_in` itself when `state_bstride = Hv*DK*Dv`. With T = 0 there are no
/// chunks, and the start state is copied through.
///
/// Output head `hv` goes to columns [out_off + hv*Dv, +Dv) of row `b*T + t`.
///
/// 128 threads = 4 simdgroups. Grid: x = Dv / 32, y = value head, z = batch.
/// The scan body for one BV-column slice of Dv (BV = 32 or 16); see
/// `GDN_SCAN_KERNEL` below for the contract.
template <uint BV>
inline void gdn_scan_body(
    device const float *qkv, device const float *ws_k, device const float *ws_q,
    device const float *ws_g, device const float *ws_beta, device const float *ws_w,
    device const float *ws_aq, device const float *state_in, device float *out,
    device float *state_out, uint T, uint Hv, uint Dv, uint ld_qkv, uint v_off,
    uint ld_out, uint out_off, uint state_bstride, uint flags,
    device const uint *seq_lens, threadgroup float *tgm, uint3 tg, uint lid, uint sg,
    uint lane)
{
    static_assert(BV % 8u == 0u && BV <= 32u, "8-column tiles, at most 4");
    constexpr uint CT = BV / 8u;   // 8-column tiles per slice
    constexpr uint TLD = BV + 4u;  // tile row stride, padded off the banks
    const uint vs = tg.x;
    const uint hv = tg.y;
    const uint b = tg.z;
    // `nc` lays out the workspace; row b scans only its own `nc_b` chunks
    // (`flags & 4`: ragged rows, Tb = min(seq_lens[b], T)).
    const uint nc = (T + GDN_C - 1u) / GDN_C;
    const uint Tb = (flags & 4u) != 0u ? min(seq_lens[b], T) : T;
    const uint nc_b = (Tb + GDN_C - 1u) / GDN_C;
    const ulong tp = (ulong)nc * GDN_C;
    const ulong head = (ulong)b * Hv + hv;
    const uint v0 = vs * BV;

    threadgroup float *S = tgm;                                  // [DK][TLD]
    threadgroup float *X = S + GDN_DK * TLD;                 // [C][TLD]
    threadgroup float *stage = X + GDN_C * TLD;              // [4][2][64]
    threadgroup float *Gc = stage + (GDN_SCAN_THREADS / 32u) * 2u * 64u; // [C]
    threadgroup float *Bc = Gc + GDN_C;                          // [C] beta
    threadgroup float *Ec = Bc + GDN_C;                          // [C] exp(G)
    threadgroup float *Dc = Ec + GDN_C;                          // [C] exp(G_last - G)
    threadgroup float *st1 = stage + sg * 2u * 64u;
    threadgroup float *st2 = st1 + 64u;

    const ulong state_head = (ulong)hv * GDN_DK * Dv;
    for (uint idx = lid; idx < GDN_DK * BV; idx += GDN_SCAN_THREADS) {
        const uint kk = idx / BV;
        const uint v = idx % BV;
        S[kk * TLD + v] = (flags & 1u) != 0u
            ? state_in[(ulong)b * state_bstride + state_head + (ulong)kk * Dv + v0 + v]
            : 0.0f;
    }

    for (uint chunk = 0u; chunk < nc_b; ++chunk) {
        const uint t0 = chunk * GDN_C;
        const ulong row_base = head * tp + t0;
        device const float *wk = ws_k + row_base * GDN_DK;
        device const float *wq = ws_q + row_base * GDN_DK;
        const ulong blk_base = (head * nc + chunk) * (ulong)(GDN_C * GDN_C);
        device const float *ww = ws_w + blk_base;
        device const float *waq = ws_aq + blk_base;

        if (lid < GDN_C) {
            Gc[lid] = ws_g[row_base + lid];
            Bc[lid] = ws_beta[row_base + lid];
        }
        // Also orders the state initialisation / the previous chunk's update.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Read after the next barrier (the one closing X = K S): computed once
        // a chunk here rather than once an element below. Both exponents are
        // <= 0 because G is non-increasing.
        if (lid < GDN_C) {
            Ec[lid] = exp(Gc[lid]);
            Dc[lid] = exp(Gc[GDN_C - 1u] - Gc[lid]);
        }

        // --- X = K S; simdgroup sg owns row blocks 2sg, 2sg+1 ----------------
        {
            simdgroup_float8x8 acc[2][CT];
            for (uint r = 0u; r < 2u; ++r) {
                for (uint ct = 0u; ct < CT; ++ct) {
                    acc[r][ct] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
            }
            for (uint kk = 0u; kk < GDN_DK / 8u; ++kk) {
                simdgroup_float8x8 a[2];
                for (uint r = 0u; r < 2u; ++r) {
                    simdgroup_load(a[r], wk + (ulong)((2u * sg + r) * 8u) * GDN_DK + kk * 8u, GDN_DK);
                }
                // Each state tile is loaded once for both row blocks.
                for (uint ct = 0u; ct < CT; ++ct) {
                    simdgroup_float8x8 st;
                    simdgroup_load(st, S + kk * 8u * TLD + ct * 8u, TLD);
                    for (uint r = 0u; r < 2u; ++r) {
                        simdgroup_multiply_accumulate(acc[r][ct], a[r], st, acc[r][ct]);
                    }
                }
            }
            for (uint r = 0u; r < 2u; ++r) {
                for (uint ct = 0u; ct < CT; ++ct) {
                    simdgroup_store(acc[r][ct], X + (2u * sg + r) * 8u * TLD + ct * 8u, TLD);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- X = beta * (V - exp(G) * X) --------------------------------------
        for (uint idx = lid; idx < GDN_C * BV; idx += GDN_SCAN_THREADS) {
            const uint i = idx / BV;
            const uint v = idx % BV;
            const uint t = t0 + i;
            const float vv = t < Tb
                ? qkv[((ulong)b * T + t) * (ulong)ld_qkv + v_off + (ulong)hv * Dv + v0 + v]
                : 0.0f;
            X[i * TLD + v] = Bc[i] * (vv - Ec[i] * X[i * TLD + v]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- U = W X (W unit lower triangular: skip the zero tiles) ----------
        {
            simdgroup_float8x8 acc[2][CT];
            for (uint r = 0u; r < 2u; ++r) {
                const uint rt = 2u * sg + r;
                for (uint ct = 0u; ct < CT; ++ct) {
                    acc[r][ct] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
                for (uint kb = 0u; kb <= rt; ++kb) {
                    simdgroup_float8x8 a;
                    simdgroup_load(a, ww + rt * 8u * GDN_C + kb * 8u, GDN_C);
                    for (uint ct = 0u; ct < CT; ++ct) {
                        simdgroup_float8x8 x;
                        simdgroup_load(x, X + kb * 8u * TLD + ct * 8u, TLD);
                        simdgroup_multiply_accumulate(acc[r][ct], a, x, acc[r][ct]);
                    }
                }
            }
            // Every simdgroup reads all of X above; none may overwrite it early.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0u; r < 2u; ++r) {
                for (uint ct = 0u; ct < CT; ++ct) {
                    simdgroup_store(acc[r][ct], X + (2u * sg + r) * 8u * TLD + ct * 8u, TLD);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- O = exp(G) * (Q S) + Aq U ----------------------------------------
        // One row block at a time, four column tiles wide, so each Q / Aq tile
        // is loaded once for all four. Results go through a per-simdgroup stage
        // so the row scale can be applied and rows past T masked: a tile store
        // straight to `out` would write the padding rows into the next sequence.
        for (uint r = 0u; r < 2u; ++r) {
            const uint rt = 2u * sg + r;
            simdgroup_float8x8 o1[CT];
            simdgroup_float8x8 o2[CT];
            for (uint ct = 0u; ct < CT; ++ct) {
                o1[ct] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                o2[ct] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
            for (uint kk = 0u; kk < GDN_DK / 8u; ++kk) {
                simdgroup_float8x8 a;
                simdgroup_load(a, wq + (ulong)(rt * 8u) * GDN_DK + kk * 8u, GDN_DK);
                for (uint ct = 0u; ct < CT; ++ct) {
                    simdgroup_float8x8 st;
                    simdgroup_load(st, S + kk * 8u * TLD + ct * 8u, TLD);
                    simdgroup_multiply_accumulate(o1[ct], a, st, o1[ct]);
                }
            }
            for (uint kb = 0u; kb <= rt; ++kb) {
                simdgroup_float8x8 a;
                simdgroup_load(a, waq + rt * 8u * GDN_C + kb * 8u, GDN_C);
                for (uint ct = 0u; ct < CT; ++ct) {
                    simdgroup_float8x8 u;
                    simdgroup_load(u, X + kb * 8u * TLD + ct * 8u, TLD);
                    simdgroup_multiply_accumulate(o2[ct], a, u, o2[ct]);
                }
            }
            for (uint ct = 0u; ct < CT; ++ct) {
                simdgroup_store(o1[ct], st1, 8);
                simdgroup_store(o2[ct], st2, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = lane; e < 64u; e += 32u) {
                    const uint i = rt * 8u + e / 8u;
                    const uint t = t0 + i;
                    if (t < Tb) {
                        out[((ulong)b * T + t) * (ulong)ld_out + out_off + (ulong)hv * Dv
                            + v0 + ct * 8u + e % 8u] = Ec[i] * st1[e] + st2[e];
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- S = exp(G_last) S + K^T (exp(G_last - G) * U) --------------------
        // Both scalings are elementwise passes: nothing reads S or U until the
        // barrier after them.
        const float decay = Ec[GDN_C - 1u];
        for (uint idx = lid; idx < GDN_C * BV; idx += GDN_SCAN_THREADS) {
            const uint i = idx / BV;
            X[i * TLD + idx % BV] *= Dc[i];
        }
        for (uint idx = lid; idx < GDN_DK * BV; idx += GDN_SCAN_THREADS) {
            S[(idx / BV) * TLD + idx % BV] *= decay;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // simdgroup sg owns state row blocks 4sg .. 4sg+3; nobody else reads or
        // writes them until the next chunk's barrier. Each K^T tile is loaded
        // once for the four column tiles of its row block.
        for (uint r = 0u; r < 4u; ++r) {
            const uint kt = 4u * sg + r;
            simdgroup_float8x8 acc[CT];
            for (uint ct = 0u; ct < CT; ++ct) {
                simdgroup_load(acc[ct], S + kt * 8u * TLD + ct * 8u, TLD);
            }
            for (uint kb = 0u; kb < GDN_C / 8u; ++kb) {
                simdgroup_float8x8 a;
                simdgroup_load(a, wk + (ulong)(kb * 8u) * GDN_DK + kt * 8u, GDN_DK,
                               ulong2(0, 0), true);
                for (uint ct = 0u; ct < CT; ++ct) {
                    simdgroup_float8x8 u;
                    simdgroup_load(u, X + kb * 8u * TLD + ct * 8u, TLD);
                    simdgroup_multiply_accumulate(acc[ct], a, u, acc[ct]);
                }
            }
            for (uint ct = 0u; ct < CT; ++ct) {
                simdgroup_store(acc[ct], S + kt * 8u * TLD + ct * 8u, TLD);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if ((flags & 2u) != 0u) {
        // With no chunks this reads the initialisation above, which the final
        // write needs ordered after it: the barrier is uniform (nc is).
        if (nc == 0u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint idx = lid; idx < GDN_DK * BV; idx += GDN_SCAN_THREADS) {
            const uint kk = idx / BV;
            const uint v = idx % BV;
            state_out[head * GDN_DK * Dv + (ulong)kk * Dv + v0 + v] = S[kk * TLD + v];
        }
    }
}

#define GDN_SCAN_KERNEL(NAME, BV)                                                  \
kernel void NAME(                                                                  \
    device const float *qkv [[buffer(0)]],                                         \
    device const float *ws_k [[buffer(1)]],                                        \
    device const float *ws_q [[buffer(2)]],                                        \
    device const float *ws_g [[buffer(3)]],                                        \
    device const float *ws_beta [[buffer(4)]],                                     \
    device const float *ws_w [[buffer(5)]],                                        \
    device const float *ws_aq [[buffer(6)]],                                       \
    device const float *state_in [[buffer(7)]],                                    \
    device float *out [[buffer(8)]],                                               \
    device float *state_out [[buffer(9)]],                                         \
    constant uint &T [[buffer(10)]],                                               \
    constant uint &Hv [[buffer(11)]],                                              \
    constant uint &Dv [[buffer(12)]],                                              \
    constant uint &ld_qkv [[buffer(13)]],                                          \
    constant uint &v_off [[buffer(14)]],                                           \
    constant uint &ld_out [[buffer(15)]],                                          \
    constant uint &out_off [[buffer(16)]],                                         \
    constant uint &state_bstride [[buffer(17)]],                                   \
    constant uint &flags [[buffer(18)]],                                           \
    device const uint *seq_lens [[buffer(19)]],                                    \
    threadgroup float *tgm [[threadgroup(0)]],                                     \
    uint3 tg [[threadgroup_position_in_grid]],                                     \
    uint lid [[thread_index_in_threadgroup]],                                      \
    uint sg [[simdgroup_index_in_threadgroup]],                                    \
    uint lane [[thread_index_in_simdgroup]])                                       \
{                                                                                  \
    gdn_scan_body<BV>(qkv, ws_k, ws_q, ws_g, ws_beta, ws_w, ws_aq, state_in, out,  \
                      state_out, T, Hv, Dv, ld_qkv, v_off, ld_out, out_off,        \
                      state_bstride, flags, seq_lens, tgm, tg, lid, sg, lane);     \
}

/// `qwen35_gdn_chunk_scan`: 32-column slices, Dv / 32 threadgroups per head.
GDN_SCAN_KERNEL(qwen35_gdn_chunk_scan, 32u)
/// 16-column slices: twice the threadgroups, for when a batch leaves the GPU
/// underfilled (`probe_gdn_scan`). Each output element runs the same
/// arithmetic in either, so the two agree bit for bit.
GDN_SCAN_KERNEL(qwen35_gdn_chunk_scan_bv16, 16u)

// --------------------------------------------------------------- recurrent ---

/// Threadgroup memory for `qwen35_gdn_recurrent`, in floats: two cross-simdgroup
/// partial-sum arrays of 4 x 32.
constant uint GDN_REC_TG_FLOATS = 2u * (GDN_SCAN_THREADS / 32u) * 32u;
static_assert(GDN_REC_TG_FLOATS * 4u <= 32768u, "recurrent threadgroup memory exceeds 32 KB");

/// Token-by-token gated delta rule, for decode and short suffixes, with the
/// gates, norms and q scale folded into the loads exactly as in chunk prep.
///
/// **Read-only snapshot.** `state_in` is only ever read. With
/// `state_bstride = 0`, every batch row starts from the same prefilled state,
/// so many questions can be answered from one snapshot without copying it or
/// perturbing it. When `flags & 2` the state after the last token of each row
/// goes to `state_out` ([B, Hv, DK, Dv]); a thread reads and writes the same
/// state elements, so in-place (`state_out == state_in`, stride Hv*DK*Dv) is
/// also safe. Without `flags & 1` the start state is zero. With `flags & 4`
/// row b runs `min(seq_lens[b], T)` steps and its later rows are not written.
///
/// Mapping: lane = value column `v0 + lane`, simdgroup = 32 key rows. Each lane
/// keeps its 32 x 1 piece of S in registers for all T steps; the two per-token
/// reductions over the key dim cross simdgroups through threadgroup memory.
///
/// 128 threads = 4 simdgroups. Grid: x = Dv / 32, y = value head, z = batch.
kernel void qwen35_gdn_recurrent(
    device const float *qkv [[buffer(0)]],
    device const float *ab [[buffer(1)]],
    device const float *a_log [[buffer(2)]],
    device const float *dt_bias [[buffer(3)]],
    device const float *state_in [[buffer(4)]],
    device float *out [[buffer(5)]],
    device float *state_out [[buffer(6)]],
    constant uint &T [[buffer(7)]],
    constant uint &Hk [[buffer(8)]],
    constant uint &Hv [[buffer(9)]],
    constant uint &Dv [[buffer(10)]],
    constant uint &ld_qkv [[buffer(11)]],
    constant uint &q_off [[buffer(12)]],
    constant uint &k_off [[buffer(13)]],
    constant uint &v_off [[buffer(14)]],
    constant uint &ld_ab [[buffer(15)]],
    constant uint &a_off [[buffer(16)]],
    constant uint &b_off [[buffer(17)]],
    constant uint &ld_out [[buffer(18)]],
    constant uint &out_off [[buffer(19)]],
    constant uint &state_bstride [[buffer(20)]],
    constant uint &flags [[buffer(21)]],
    device const uint *seq_lens [[buffer(22)]],
    threadgroup float *tgm [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint vs = tg.x;
    const uint hv = tg.y;
    const uint b = tg.z;
    const uint hk = hv / max(Hv / Hk, 1u);
    const uint v = vs * GDN_BV + lane;
    const uint k0 = sg * 32u;
    const float q_scale = rsqrt((float)GDN_DK);
    threadgroup float *red_kv = tgm;          // [4][32]
    threadgroup float *red_y = tgm + GDN_REC_TG_FLOATS / 2u; // [4][32]

    float s[32];
    const ulong state_head = (ulong)hv * GDN_DK * Dv;
    for (uint j = 0u; j < 32u; ++j) {
        s[j] = (flags & 1u) != 0u
            ? state_in[(ulong)b * state_bstride + state_head + (ulong)(k0 + j) * Dv + v]
            : 0.0f;
    }

    // Row b's live length (`flags & 4`: ragged rows); the row stride stays T.
    const uint Tb = (flags & 4u) != 0u ? min(seq_lens[b], T) : T;
    for (uint t = 0u; t < Tb; ++t) {
        const ulong row = (ulong)b * T + t;
        device const float *qp = qkv + row * (ulong)ld_qkv + q_off + hk * GDN_DK;
        device const float *kp = qkv + row * (ulong)ld_qkv + k_off + hk * GDN_DK;
        device const float *gr = ab + row * (ulong)ld_ab;

        // Every simdgroup computes both norms itself: 4 loads a lane, and it
        // saves a barrier.
        float ssq = 0.0f;
        float ssk = 0.0f;
        for (uint d = lane; d < GDN_DK; d += 32u) {
            ssq += qp[d] * qp[d];
            ssk += kp[d] * kp[d];
        }
        const float rq = rsqrt(simd_sum(ssq) + GDN_L2_EPS) * q_scale;
        const float rk = rsqrt(simd_sum(ssk) + GDN_L2_EPS);
        const float decay = exp(gdn_log_decay(gr[a_off + hv], a_log[hv], dt_bias[hv]));
        const float beta = qwen35_sigmoid(gr[b_off + hv]);

        // kv_mem = (decay * S)^T k
        float kv = 0.0f;
        for (uint j = 0u; j < 32u; ++j) {
            s[j] *= decay;
            kv += s[j] * (kp[k0 + j] * rk);
        }
        red_kv[sg * 32u + lane] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float kv_mem = red_kv[lane] + red_kv[32u + lane] + red_kv[64u + lane] + red_kv[96u + lane];
        const float delta = beta * (qkv[row * (ulong)ld_qkv + v_off + (ulong)hv * Dv + v] - kv_mem);

        // S += k delta^T; y = S^T q
        float y = 0.0f;
        for (uint j = 0u; j < 32u; ++j) {
            s[j] += (kp[k0 + j] * rk) * delta;
            y += s[j] * (qp[k0 + j] * rq);
        }
        red_y[sg * 32u + lane] = y;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            out[row * (ulong)ld_out + out_off + (ulong)hv * Dv + v] =
                red_y[lane] + red_y[32u + lane] + red_y[64u + lane] + red_y[96u + lane];
        }
        // red_kv is next written after this token's second barrier, by which
        // point every read of it has happened; red_y is next written after the
        // next token's first barrier, which simdgroup 0 reaches only after its
        // read above. Two barriers a token are therefore enough.
    }

    if ((flags & 2u) != 0u) {
        const ulong head = (ulong)b * Hv + hv;
        for (uint j = 0u; j < 32u; ++j) {
            state_out[head * GDN_DK * Dv + (ulong)(k0 + j) * Dv + v] = s[j];
        }
    }
}

// --------------------------------------------------------- gated RMSNorm ---

/// `out = rms_norm(x) * w * silu(z)` per head: transformers'
/// `Qwen3_5RMSNormGated` (weight initialised to ones, applied as `w`, not
/// `1 + w`; norm before gate). One simdgroup per (row, head).
///
/// Row `r`, head `h` reads `x[r*ld_x + x_off + h*D ..]` and
/// `z[r*ld_z + z_off + h*D ..]` and writes `out[r*ld_out + out_off + h*D ..]`.
/// Everything is computed in f32; the bf16 variant rounds once, on store, so it
/// can feed a bf16 `out_proj` GEMM directly.
///
/// Launch whole simdgroups; the host dispatches `ceil(rows*H / per_tg)` groups.

/// Shared body of the two gated-norm kernels; `OutT` is the store type only.
template <typename OutT>
inline void gated_rms_norm_row(
    device const float *xr,
    device const float *zr,
    device const float *w,
    device OutT *orow,
    uint D,
    float eps,
    uint lane)
{
    float ss = 0.0f;
    for (uint d = lane; d < D; d += 32u) {
        ss += xr[d] * xr[d];
    }
    const float inv = rsqrt(simd_sum(ss) / (float)D + eps);
    for (uint d = lane; d < D; d += 32u) {
        orow[d] = (OutT)(w[d] * (xr[d] * inv) * qwen35_silu(zr[d]));
    }
}

#define GATED_RMS_NORM_KERNEL(NAME, OUT_T)                                        \
kernel void NAME(                                                                 \
    device const float *x [[buffer(0)]],                                          \
    device const float *z [[buffer(1)]],                                          \
    device const float *w [[buffer(2)]],                                          \
    device OUT_T *out [[buffer(3)]],                                              \
    constant uint &rows [[buffer(4)]],                                            \
    constant uint &H [[buffer(5)]],                                               \
    constant uint &D [[buffer(6)]],                                               \
    constant uint &ld_x [[buffer(7)]],                                            \
    constant uint &x_off [[buffer(8)]],                                           \
    constant uint &ld_z [[buffer(9)]],                                            \
    constant uint &z_off [[buffer(10)]],                                          \
    constant uint &ld_out [[buffer(11)]],                                         \
    constant uint &out_off [[buffer(12)]],                                        \
    constant float &eps [[buffer(13)]],                                           \
    uint tg [[threadgroup_position_in_grid]],                                     \
    uint sg [[simdgroup_index_in_threadgroup]],                                   \
    uint lane [[thread_index_in_simdgroup]],                                      \
    uint tptg [[threads_per_threadgroup]])                                        \
{                                                                                 \
    const ulong unit = (ulong)tg * (tptg / 32u) + sg;                             \
    /* Uniform per simdgroup, and this kernel has no threadgroup barrier. */      \
    if (unit >= (ulong)rows * H) return;                                          \
    const ulong r = unit / H;                                                     \
    const ulong h = unit % H;                                                     \
    gated_rms_norm_row<OUT_T>(x + r * (ulong)ld_x + x_off + h * D,                \
                              z + r * (ulong)ld_z + z_off + h * D,                \
                              w,                                                  \
                              out + r * (ulong)ld_out + out_off + h * D,          \
                              D, eps, lane);                                      \
}

GATED_RMS_NORM_KERNEL(qwen35_gated_rms_norm_f32, float)
GATED_RMS_NORM_KERNEL(qwen35_gated_rms_norm_bf16, bfloat)
