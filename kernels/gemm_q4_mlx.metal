// DFlash verify thin Q4 GEMM (M≤8). Standalone so gemv_q4_mlx.metal edits do not drop it.
#include <metal_stdlib>
#include "q4_mlx_dot.h"
using namespace metal;

constant uint SIMD_SIZE = 32u;
constant uint SIMD_ROWS = 4u;
constant uint SIMD_SG_PER_TG = 2u;
constant uint SIMD_PACKS = 2u;
constant uint SIMD_VPT = 8u * SIMD_PACKS;     // 16
constant uint SIMD_BLOCK = SIMD_SIZE * SIMD_VPT; // 512
constant uint GEMM_MAX_M = 8u;


kernel void gemm_q4_mlx_simd(
    device const uchar *packed [[buffer(0)]],
    device const bfloat2 *sb [[buffer(1)]],
    device const bfloat *biases_unused [[buffer(2)]],
    device const bfloat *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    constant uint &M [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    (void)biases_unused;
    const uint m_cap = min(M, GEMM_MAX_M);
    if (m_cap == 0u) return;
    const uint row0 = (tgid * SIMD_SG_PER_TG + sgid) * SIMD_ROWS;
    if (row0 >= rows) return;
    const uint gpr = cols / group_size;
    const uint row_bytes = cols >> 1;
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
            const ulong byte_off = col >> 1;
            for (uint r = 0u; r < SIMD_ROWS; ++r) {
                const uint row = row0 + r;
                if (row >= rows) break;
                const bfloat2 sbv = sb[(ulong)row * gpr + g];
                device const uchar *wp = packed + (ulong)row * row_bytes + byte_off;
                for (uint m = 0u; m < m_cap; ++m)
                    acc[r][m] += qdot16(wp, xt[m], float(sbv.x), float(sbv.y), xsum[m]);
            }
        }
    }
    for (uint r = 0u; r < SIMD_ROWS; ++r) {
        const uint row = row0 + r;
        for (uint m = 0u; m < m_cap; ++m) {
            const float sum = simd_sum(acc[r][m]);
            if (lane == 0u && row < rows) y[(ulong)m * rows + row] = sum;
        }
    }
}

kernel void gemm_q4_mlx_simd_add(
    device const uchar *packed [[buffer(0)]],
    device const bfloat2 *sb [[buffer(1)]],
    device const bfloat *biases_unused [[buffer(2)]],
    device const bfloat *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    constant uint &M [[buffer(8)]],
    device const float *resid [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    (void)biases_unused;
    const uint m_cap = min(M, GEMM_MAX_M);
    if (m_cap == 0u) return;
    const uint row0 = (tgid * SIMD_SG_PER_TG + sgid) * SIMD_ROWS;
    if (row0 >= rows) return;
    const uint gpr = cols / group_size;
    const uint row_bytes = cols >> 1;
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
            const ulong byte_off = col >> 1;
            for (uint r = 0u; r < SIMD_ROWS; ++r) {
                const uint row = row0 + r;
                if (row >= rows) break;
                const bfloat2 sbv = sb[(ulong)row * gpr + g];
                device const uchar *wp = packed + (ulong)row * row_bytes + byte_off;
                for (uint m = 0u; m < m_cap; ++m)
                    acc[r][m] += qdot16(wp, xt[m], float(sbv.x), float(sbv.y), xsum[m]);
            }
        }
    }
    for (uint r = 0u; r < SIMD_ROWS; ++r) {
        const uint row = row0 + r;
        for (uint m = 0u; m < m_cap; ++m) {
            const float sum = simd_sum(acc[r][m]);
            if (lane == 0u && row < rows)
                y[(ulong)m * rows + row] = sum + resid[(ulong)m * rows + row];
        }
    }
}

kernel void gemm_q4_mlx_simd_i4(
    device const uchar *packed [[buffer(0)]],
    device const bfloat2 *sb [[buffer(1)]],
    device const bfloat *biases_unused [[buffer(2)]],
    device const bfloat *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    constant uint &M [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    (void)biases_unused;
    const uint m_cap = min(M, GEMM_MAX_M);
    if (m_cap == 0u) return;
    const uint row0 = (tgid * SIMD_SG_PER_TG + sgid) * SIMD_ROWS;
    if (row0 >= rows) return;
    const uint tile = row0 / SIMD_ROWS;
    const uint gpr = cols / group_size;
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
            const ulong pack2 = col >> 4;
            device const uchar *wp0 = packed
                + (((ulong)tile * packs_u2 + pack2) * SIMD_ROWS) * 8ul;
            const ulong sb0 = ((ulong)tile * gpr + g) * SIMD_ROWS;
            for (uint r = 0u; r < SIMD_ROWS; ++r) {
                if (row0 + r >= rows) break;
                const bfloat2 sbv = sb[sb0 + r];
                for (uint m = 0u; m < m_cap; ++m)
                    acc[r][m] += qdot16(wp0 + r * 8u, xt[m], float(sbv.x), float(sbv.y), xsum[m]);
            }
        }
    }
    for (uint r = 0u; r < SIMD_ROWS; ++r) {
        const uint row = row0 + r;
        for (uint m = 0u; m < m_cap; ++m) {
            const float sum = simd_sum(acc[r][m]);
            if (lane == 0u && row < rows) y[(ulong)m * rows + row] = sum;
        }
    }
}

kernel void gemm_q4_mlx_simd_add_i4(
    device const uchar *packed [[buffer(0)]],
    device const bfloat2 *sb [[buffer(1)]],
    device const bfloat *biases_unused [[buffer(2)]],
    device const bfloat *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    constant uint &M [[buffer(8)]],
    device const float *resid [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    (void)biases_unused;
    const uint m_cap = min(M, GEMM_MAX_M);
    if (m_cap == 0u) return;
    const uint row0 = (tgid * SIMD_SG_PER_TG + sgid) * SIMD_ROWS;
    if (row0 >= rows) return;
    const uint tile = row0 / SIMD_ROWS;
    const uint gpr = cols / group_size;
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
            const ulong pack2 = col >> 4;
            device const uchar *wp0 = packed
                + (((ulong)tile * packs_u2 + pack2) * SIMD_ROWS) * 8ul;
            const ulong sb0 = ((ulong)tile * gpr + g) * SIMD_ROWS;
            for (uint r = 0u; r < SIMD_ROWS; ++r) {
                if (row0 + r >= rows) break;
                const bfloat2 sbv = sb[sb0 + r];
                for (uint m = 0u; m < m_cap; ++m)
                    acc[r][m] += qdot16(wp0 + r * 8u, xt[m], float(sbv.x), float(sbv.y), xsum[m]);
            }
        }
    }
    for (uint r = 0u; r < SIMD_ROWS; ++r) {
        const uint row = row0 + r;
        for (uint m = 0u; m < m_cap; ++m) {
            const float sum = simd_sum(acc[r][m]);
            if (lane == 0u && row < rows)
                y[(ulong)m * rows + row] = sum + resid[(ulong)m * rows + row];
        }
    }
}
