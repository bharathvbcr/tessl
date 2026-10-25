// The simdgroup-cooperative MLX affine Q4 GEMV (M=1) body shared by the ten
// `gemv_q4_mlx_simd*` entry points in `gemv_q4_mlx.metal` (and an
// eleventh, `gemv_q4_mlx_simd_gate_up_silu`, Qwen3.5's gate/up with SiLU,
// row-major only):
//
//     layout        Store          AddResid            GateUpGelu                    (+ bank routing)
//     row-major     _simd          _simd_add           _simd_gate_up_gelu            _simd_kv, _simd_qkv
//     Interleaved4  _simd_i4       _simd_add_i4        _simd_gate_up_gelu_i4         _simd_kv_i4, _simd_qkv_i4
//
// True `qmv_fast` peel (see mlx quantized.h):
//   - interleaved Hot scale+bias as bfloat2 (one 4B load / group)
//   - packs_per_thread=2 -> 16 nibbles/lane -> K-block 512
//   - 4 results/simdgroup x 2 SG/TG -> 8 rows/TG
//   - load_vector 16^k-prescale x + ushort-mask qdot + bias*sum(x)
//   - row-major: pointer walk (ws/sb/x advance) for occupancy
//
// The ten used to be ten hand-copied bodies. They differ only in the weight
// layout, in what is done with each finished row (store, add a residual,
// gate*up), and in which bank a threadgroup owns. All three are template
// parameters or helpers below, so a fix to the K loop is made once. What stays
// in `gemv_q4_mlx.metal` is exactly what cannot be templated: the entry names
// and `[[buffer(N)]]` slots, which the host and gemma-metal (ojas) bind by
// name and number and which differ per entry.
//
// Math is the same sequence of operations as the bodies this replaced:
// `tests/q4_kernels_template.rs` pins every entry's output bits against values
// recorded before the fold. `qdot16` / `load_x16_qdot` and the simdgroup
// geometry live in `q4_mlx_dot.h`.
//
// Included, never compiled on its own; `build.rs` tracks `.h` changes.
#pragma once

#include <metal_stdlib>
#include "act.h"
#include "q4_mlx_dot.h"
using namespace metal;

/// What a finished row becomes.
enum class Q4Epi : uint {
    /// `out[row] = sum`.
    Store = 0u,
    /// `out[row] = sum + resid[row]`.
    AddResid = 1u,
    /// Two banks over one x: `gelu_pytorch_tanh(gate) * up`, written as f32 or
    /// (`mid_as_bf16`) bf16 into `out`.
    GateUpGelu = 2u,
    /// [`GateUpGelu`] with SiLU: `silu(gate) * up`, Qwen3.5's MLP.
    GateUpSilu = 3u,
};

/// First output row of this simdgroup within the bank its threadgroup owns.
inline uint q4_simd_row0(uint local_tg, uint sgid)
{
    return (local_tg * SIMD_SG_PER_TG + sgid) * SIMD_ROWS;
}

/// Where a threadgroup of a bank-partitioned dispatch (K/V, Q/K/V) works.
/// Bank choice depends only on `tgid` (threadgroup-uniform) and `row0` only on
/// `sgid` (simdgroup-uniform), so the early-out in `q4_simd_gemv` never splits
/// a simdgroup before `simd_sum`.
struct Q4Route {
    Q4Bank bank;
    device float *y;
    uint rows;
    uint local_tg;
};

/// `[0, tg_k)` -> K, `[tg_k, ..)` -> V. Both banks have `rows` rows.
inline Q4Route q4_route_kv(
    uint tgid,
    uint tg_k,
    Q4Bank k,
    device float *k_out,
    Q4Bank v,
    device float *v_out,
    uint rows)
{
    if (tgid < tg_k) return Q4Route{k, k_out, rows, tgid};
    return Q4Route{v, v_out, rows, tgid - tg_k};
}

/// `[0, tg_q)` -> Q, `[tg_q, tg_q + tg_k)` -> K, the rest -> V. Q has `rows_q`
/// rows, K and V `rows_kv`.
inline Q4Route q4_route_qkv(
    uint tgid,
    uint tg_q,
    uint tg_k,
    Q4Bank q,
    device float *q_out,
    uint rows_q,
    Q4Bank k,
    device float *k_out,
    Q4Bank v,
    device float *v_out,
    uint rows_kv)
{
    if (tgid < tg_q) return Q4Route{q, q_out, rows_q, tgid};
    if (tgid < tg_q + tg_k) return Q4Route{k, k_out, rows_kv, tgid - tg_q};
    return Q4Route{v, v_out, rows_kv, tgid - tg_q - tg_k};
}

/// A bank slot an epilogue does not use.
inline Q4Bank q4_no_bank()
{
    return Q4Bank{nullptr, nullptr};
}

/// One simdgroup's SIMD_ROWS rows of a Q4 GEMV.
///
/// `bank1`, `resid` and `mid_as_bf16` are read only by the epilogues that use
/// them (GateUpGelu / AddResid); pass `q4_no_bank()`, `nullptr`, `0u` otherwise.
/// Returns without writing when `row0 >= rows`; rows past `rows` in a partial
/// last group are neither read nor written.
template <Q4Layout L, Q4Epi E>
inline void q4_simd_gemv(
    Q4Bank bank0,
    Q4Bank bank1,
    device const bfloat *x,
    device float *out,
    device const float *resid,
    uint mid_as_bf16,
    uint rows,
    uint cols,
    uint group_size,
    uint row0,
    uint lane)
{
    constexpr bool ROW_MAJOR = (L == Q4Layout::RowMajor);
    constexpr uint NB = (E == Q4Epi::GateUpGelu || E == Q4Epi::GateUpSilu) ? 2u : 1u;

    if (row0 >= rows) return;
    const Q4Bank banks[2] = {bank0, bank1};
    const uint gpr = cols / group_size;
    const uint lane_col0 = lane * SIMD_VPT;

    // Row-major: MLX qmv_fast occupancy, pointer-walk ws/sb/x across K blocks.
    const uint row_bytes = cols >> 1;
    const uint sb_k_step = SIMD_BLOCK / group_size;
    device const uchar *ws[NB];
    device const bfloat2 *sbr[NB];
    device const bfloat *xr = x + lane_col0;
    // Interleaved4: [tile][uint2 pack][r0..r3] weights, [tile][g][r0..r3] sb.
    const uint tile = row0 / SIMD_ROWS;
    const uint packs_u2 = cols >> 4;
    for (uint b = 0u; b < NB; ++b) {
        ws[b] = banks[b].packed + (ulong)row0 * row_bytes + (lane_col0 >> 1);
        sbr[b] = banks[b].sb + (ulong)row0 * gpr + (lane_col0 / group_size);
    }

    float acc[NB][SIMD_ROWS];
    for (uint b = 0u; b < NB; ++b)
        for (uint r = 0u; r < SIMD_ROWS; ++r) acc[b][r] = 0.0f;
    for (ulong k0 = 0ul; k0 < (ulong)cols; k0 += SIMD_BLOCK) {
        const ulong col = k0 + lane_col0;
        if (col + SIMD_VPT <= (ulong)cols) {
            float xt[16];
            float xsum;
            ulong i4_weights = 0ul;
            ulong i4_sb = 0ul;
            if (ROW_MAJOR) {
                xsum = load_x16_qdot(xr, xt);
            } else {
                xsum = load_x16_qdot(x + col, xt);
                const ulong g = col / group_size;
                const ulong pack2 = col >> 4;
                i4_weights = (((ulong)tile * packs_u2 + pack2) * SIMD_ROWS) * 8ul;
                i4_sb = ((ulong)tile * gpr + g) * SIMD_ROWS;
            }
            for (uint r = 0u; r < SIMD_ROWS; ++r) {
                if (row0 + r >= rows) break;
                for (uint b = 0u; b < NB; ++b) {
                    bfloat2 sbv;
                    device const uchar *wp;
                    if (ROW_MAJOR) {
                        sbv = sbr[b][(ulong)r * gpr];
                        wp = ws[b] + (ulong)r * row_bytes;
                    } else {
                        sbv = banks[b].sb[i4_sb + r];
                        wp = banks[b].packed + i4_weights + r * 8u;
                    }
                    acc[b][r] += qdot16(wp, xt, float(sbv.x), float(sbv.y), xsum);
                }
            }
        }
        if (ROW_MAJOR) {
            for (uint b = 0u; b < NB; ++b) {
                ws[b] += SIMD_BLOCK >> 1;
                sbr[b] += sb_k_step;
            }
            xr += SIMD_BLOCK;
        }
    }
    for (uint r = 0u; r < SIMD_ROWS; ++r) {
        float sum[NB];
        for (uint b = 0u; b < NB; ++b) sum[b] = simd_sum(acc[b][r]);
        const uint row = row0 + r;
        if (lane == 0u && row < rows) {
            if (E == Q4Epi::Store) {
                out[row] = sum[0];
            } else if (E == Q4Epi::AddResid) {
                out[row] = sum[0] + resid[row];
            } else {
                const float act = (E == Q4Epi::GateUpSilu) ? tessl_silu(sum[0]) : tessl_gelu_pytorch_tanh(sum[0]);
                const float v = act * sum[NB - 1u];
                if (mid_as_bf16 != 0u) {
                    ((device bfloat *)out)[row] = bfloat(v);
                } else {
                    out[row] = v;
                }
            }
        }
    }
}
