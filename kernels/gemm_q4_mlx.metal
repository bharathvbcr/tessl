// Thin Q4 (MLX-layout) GEMM for small M (M≤8), dispatched by `nn::gemm_q4_mlx`. Standalone so
// gemv_q4_mlx.metal edits do not drop it. The entry points below own only their names and
// `[[buffer(N)]]` slots; the body is `gemm_q4_mlx_simd.h` (`q4_simd_gemm<layout, ADD>`).
#include <metal_stdlib>
#include "gemm_q4_mlx_simd.h"
using namespace metal;

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
    q4_simd_gemm<Q4Layout::RowMajor, false>(
        Q4Bank{packed, sb}, x, y, nullptr, rows, cols, group_size, M, tgid, sgid, lane);
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
    q4_simd_gemm<Q4Layout::RowMajor, true>(
        Q4Bank{packed, sb}, x, y, resid, rows, cols, group_size, M, tgid, sgid, lane);
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
    q4_simd_gemm<Q4Layout::Interleaved4, false>(
        Q4Bank{packed, sb}, x, y, nullptr, rows, cols, group_size, M, tgid, sgid, lane);
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
    q4_simd_gemm<Q4Layout::Interleaved4, true>(
        Q4Bank{packed, sb}, x, y, resid, rows, cols, group_size, M, tgid, sgid, lane);
}
