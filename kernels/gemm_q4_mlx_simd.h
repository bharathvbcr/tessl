// The thin Q4 (MLX layout) GEMM body for small M (M <= GEMM_MAX_M) shared by
// the four `gemm_q4_mlx_simd*` entry points in `gemm_q4_mlx.metal`:
//
//     row-major: _simd, _simd_add        Interleaved4: _simd_i4, _simd_add_i4
//
// Same simdgroup structure as `gemv_q4_mlx_simd.h` (geometry and `qdot16` in
// `q4_mlx_dot.h`), with each simdgroup carrying up to GEMM_MAX_M activation rows
// through one weight read. The four used to be four hand-copied bodies that
// differed only in the weight layout (`Q4Layout`) and whether a residual is
// added (`ADD`). What stays in the `.metal` file is what cannot be templated:
// the entry names and `[[buffer(N)]]` slots the host and gemma-metal bind.
//
// `tests/q4_kernels_template.rs` pins every entry's output bits against values
// recorded before the fold.
//
// Included, never compiled on its own; `build.rs` tracks `.h` changes.
#pragma once

#include <metal_stdlib>
#include "q4_mlx_dot.h"
using namespace metal;

constant uint GEMM_MAX_M = 8u;

/// `y[m * rows + row] = sum (+ resid[m * rows + row] when ADD)` for the
/// SIMD_ROWS rows at `row0` and `min(M, GEMM_MAX_M)` activation rows.
/// `resid` is read only when ADD; pass `nullptr` otherwise.
template <Q4Layout L, bool ADD>
inline void q4_simd_gemm(
    Q4Bank bank,
    device const bfloat *x,
    device float *y,
    device const float *resid,
    uint rows,
    uint cols,
    uint group_size,
    uint M,
    uint tgid,
    uint sgid,
    uint lane)
{
    constexpr bool ROW_MAJOR = (L == Q4Layout::RowMajor);
    const uint m_cap = min(M, GEMM_MAX_M);
    if (m_cap == 0u) return;
    const uint row0 = (tgid * SIMD_SG_PER_TG + sgid) * SIMD_ROWS;
    if (row0 >= rows) return;
    const uint tile = row0 / SIMD_ROWS;
    const uint gpr = cols / group_size;
    const uint row_bytes = cols >> 1;
    const uint packs_u2 = cols >> 4;
    const uint lane_col0 = lane * SIMD_VPT;
    float acc[SIMD_ROWS][GEMM_MAX_M];
    for (uint r = 0u; r < SIMD_ROWS; ++r)
        for (uint m = 0u; m < GEMM_MAX_M; ++m) acc[r][m] = 0.0f;
    for (ulong k0 = 0ul; k0 < (ulong)cols; k0 += SIMD_BLOCK) {
        const ulong col = k0 + lane_col0;
        if (col + SIMD_VPT <= (ulong)cols) {
            float xt[GEMM_MAX_M][16];
            float xsum[GEMM_MAX_M];
            for (uint m = 0u; m < m_cap; ++m)
                xsum[m] = load_x16_qdot(x + (ulong)m * cols + col, xt[m]);
            const ulong g = col / group_size;
            // Row-major: a row's weights start `byte_off` into the row.
            const ulong byte_off = col >> 1;
            // Interleaved4: [tile][uint2 pack][r0..r3] weights, [tile][g][r0..r3] sb.
            const ulong pack2 = col >> 4;
            device const uchar *wp0 = bank.packed
                + (((ulong)tile * packs_u2 + pack2) * SIMD_ROWS) * 8ul;
            const ulong sb0 = ((ulong)tile * gpr + g) * SIMD_ROWS;
            for (uint r = 0u; r < SIMD_ROWS; ++r) {
                const uint row = row0 + r;
                if (row >= rows) break;
                bfloat2 sbv;
                device const uchar *wp;
                if (ROW_MAJOR) {
                    sbv = bank.sb[(ulong)row * gpr + g];
                    wp = bank.packed + (ulong)row * row_bytes + byte_off;
                } else {
                    sbv = bank.sb[sb0 + r];
                    wp = wp0 + r * 8u;
                }
                for (uint m = 0u; m < m_cap; ++m)
                    acc[r][m] += qdot16(wp, xt[m], float(sbv.x), float(sbv.y), xsum[m]);
            }
        }
    }
    for (uint r = 0u; r < SIMD_ROWS; ++r) {
        const uint row = row0 + r;
        for (uint m = 0u; m < m_cap; ++m) {
            const float sum = simd_sum(acc[r][m]);
            if (lane == 0u && row < rows) {
                if (ADD) {
                    y[(ulong)m * rows + row] = sum + resid[(ulong)m * rows + row];
                } else {
                    y[(ulong)m * rows + row] = sum;
                }
            }
        }
    }
}
