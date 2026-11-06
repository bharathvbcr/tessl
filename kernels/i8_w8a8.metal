// W8A8 int8 x int8 -> int32 on MPP matmul2d, dequantized in the epilogue.
//
// The legacy kernel `matmul2d_tensorops_i8_f32` is unchanged. This file is the
// wider entry: per-row or scalar activation scale, optional per-column scale
// OR per-group scales (one int32 matmul2d per group, then an ascending f32
// flush), optional bias, f32 / f16 / bf16 store, and [N,K] weights
// (transpose_right). M == 1 is the GEMV below, not a tiled matmul.
//
// Device pointers are non-const: const poisons MPP type matching. Direct
// device loads, no threadgroup staging. Every lane of a dispatched simdgroup
// reaches matmul2d; the tile predicate is uniform across the threadgroup.
//
// Epilogue index {n, m} is the Q4 kernel's mapping for a destination whose
// extents are {N, M}. The host tests pin it against the legacy kernel.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "i8_tile_walk.h"

using namespace metal;
using namespace mpp::tensor_ops;

#pragma clang fp contract(off) reassociate(off)

inline void i8_store(device char *C, int m, int n, int N, float v, uint out_kind) {
    ulong i = (ulong)m * (ulong)N + (ulong)n;
    if (out_kind == 1u) {
        ((device half *)C)[i] = half(v);
    } else if (out_kind == 2u) {
        ((device bfloat *)C)[i] = bfloat(v);
    } else {
        ((device float *)C)[i] = v;
    }
}

template <typename Acc>
inline void i8_epilogue_i32(thread Acc &acc, int tx, int ty, int M, int N,
                            device float *a_rows, float a_scalar, uint a_mode,
                            device float *b_scale, uint b_mode, device float *bias, uint has_bias,
                            device char *C, uint out_kind) {
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        const auto idx = acc.get_multidimensional_index(i);
        const int n = tx + (int)idx[0];
        const int m = ty + (int)idx[1];
        if (m < 0 || n < 0 || m >= M || n >= N) continue;
        float ascl = (a_mode != 0u) ? a_rows[m] : a_scalar;
        float bscl = (b_mode == 1u) ? b_scale[n] : 1.0f;
        float v = ((float)acc[i] * ascl) * bscl;
        if (has_bias != 0u) v += bias[n];
        i8_store(C, m, n, N, v, out_kind);
    }
}

template <typename Acc>
inline void i8_epilogue_f32(thread Acc &acc, int tx, int ty, int M, int N, device float *a_rows,
                            float a_scalar, uint a_mode, device float *bias, uint has_bias, device char *C,
                            uint out_kind) {
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        const auto idx = acc.get_multidimensional_index(i);
        const int n = tx + (int)idx[0];
        const int m = ty + (int)idx[1];
        if (m < 0 || n < 0 || m >= M || n >= N) continue;
        float ascl = (a_mode != 0u) ? a_rows[m] : a_scalar;
        float v = acc[i] * ascl;
        if (has_bias != 0u) v += bias[n];
        i8_store(C, m, n, N, v, out_kind);
    }
}

// NN B is [K, N]. NT B is [N, K]. A group window's static K is G because a
// slice keeps the remaining extent, so one full-K matmul cannot carry
// per-group scales.

template <int SM, int SN>
inline void mm_nn_full(device int8_t *A, device int8_t *B, device char *C, device float *a_rows, float a_scalar,
                       uint a_mode, device float *b_scale, uint b_mode, device float *bias, uint has_bias,
                       uint out_kind, uint M, uint N, uint K, uint tiles_n, uint tiles_m, uint tgpig) {
    constexpr auto d = matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, false, false,
                                           matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    uint2 tile = i8_tile_walk<SM>(tgpig, tiles_n, tiles_m, N, K);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    auto tA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
    auto tB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
    auto sA = tA.slice(0, ty);
    auto sB = tB.slice(tx, 0);
    auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(sA)>,
                                                              metal::remove_addrspace_t<decltype(sB)>, int32_t>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc.set(i, 0);
    op.run(sA, sB, acc);
    i8_epilogue_i32(acc, tx, ty, (int)M, (int)N, a_rows, a_scalar, a_mode, b_scale, b_mode, bias, has_bias, C,
                    out_kind);
}

template <int SM, int SN>
inline void mm_nt_full(device int8_t *A, device int8_t *B, device char *C, device float *a_rows, float a_scalar,
                       uint a_mode, device float *b_scale, uint b_mode, device float *bias, uint has_bias,
                       uint out_kind, uint M, uint N, uint K, uint tiles_n, uint tiles_m, uint tgpig) {
    constexpr auto d = matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, true, false,
                                           matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    uint2 tile = i8_tile_walk<SM>(tgpig, tiles_n, tiles_m, N, K);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    auto tA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
    auto tB = tensor(B, dextents<int, 2>{(int)K, (int)N}, array<int, 2>{1, (int)K});
    auto sA = tA.slice(0, ty);
    auto sB = tB.slice(0, tx);
    auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(sA)>,
                                                              metal::remove_addrspace_t<decltype(sB)>, int32_t>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc.set(i, 0);
    op.run(sA, sB, acc);
    i8_epilogue_i32(acc, tx, ty, (int)M, (int)N, a_rows, a_scalar, a_mode, b_scale, b_mode, bias, has_bias, C,
                    out_kind);
}

template <int SM, int SN, int G>
inline void mm_nn_group(device int8_t *A, device int8_t *B, device char *C, device float *a_rows, float a_scalar,
                        uint a_mode, device float *b_scale, device float *bias, uint has_bias, uint out_kind, uint M,
                        uint N, uint K, uint tiles_n, uint tiles_m, uint tgpig) {
    constexpr auto d =
        matmul2d_descriptor(SM, SN, G, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    uint2 tile = i8_tile_walk<SM>(tgpig, tiles_n, tiles_m, N, K);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    auto tA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
    auto tB = tensor(B, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
    auto sA0 = tA.slice(0, ty);
    auto sB0 = tB.slice(tx, 0);
    auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(sA0)>,
                                                              metal::remove_addrspace_t<decltype(sB0)>, float>();
    auto grp = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(sA0)>,
                                                              metal::remove_addrspace_t<decltype(sB0)>, int32_t>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc.set(i, 0.0f);
    const int groups = (int)K / G;
    for (int g = 0; g < groups; ++g) {
        auto sA = tA.slice(g * G, ty);
        auto sB = tB.slice(tx, g * G);
        for (uint16_t i = 0; i < grp.get_capacity(); ++i) grp.set(i, 0);
        op.run(sA, sB, grp);
        for (uint16_t i = 0; i < grp.get_capacity(); ++i) {
            if (!grp.is_valid_element(i)) continue;
            const auto idx = grp.get_multidimensional_index(i);
            const int n = tx + (int)idx[0];
            const int m = ty + (int)idx[1];
            if (m < 0 || n < 0 || m >= (int)M || n >= (int)N) continue;
            acc[i] = acc[i] + (float)grp[i] * b_scale[(ulong)g * N + (ulong)n];
        }
    }
    i8_epilogue_f32(acc, tx, ty, (int)M, (int)N, a_rows, a_scalar, a_mode, bias, has_bias, C, out_kind);
}

template <int SM, int SN, int G>
inline void mm_nt_group(device int8_t *A, device int8_t *B, device char *C, device float *a_rows, float a_scalar,
                        uint a_mode, device float *b_scale, device float *bias, uint has_bias, uint out_kind, uint M,
                        uint N, uint K, uint tiles_n, uint tiles_m, uint tgpig) {
    constexpr auto d = matmul2d_descriptor(SM, SN, G, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    uint2 tile = i8_tile_walk<SM>(tgpig, tiles_n, tiles_m, N, K);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    auto tA = tensor(A, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
    auto tB = tensor(B, dextents<int, 2>{(int)K, (int)N}, array<int, 2>{1, (int)K});
    auto sA0 = tA.slice(0, ty);
    auto sB0 = tB.slice(0, tx);
    auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(sA0)>,
                                                              metal::remove_addrspace_t<decltype(sB0)>, float>();
    auto grp = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(sA0)>,
                                                              metal::remove_addrspace_t<decltype(sB0)>, int32_t>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc.set(i, 0.0f);
    const int groups = (int)K / G;
    for (int g = 0; g < groups; ++g) {
        auto sA = tA.slice(g * G, ty);
        auto sB = tB.slice(g * G, tx);
        for (uint16_t i = 0; i < grp.get_capacity(); ++i) grp.set(i, 0);
        op.run(sA, sB, grp);
        for (uint16_t i = 0; i < grp.get_capacity(); ++i) {
            if (!grp.is_valid_element(i)) continue;
            const auto idx = grp.get_multidimensional_index(i);
            const int n = tx + (int)idx[0];
            const int m = ty + (int)idx[1];
            if (m < 0 || n < 0 || m >= (int)M || n >= (int)N) continue;
            acc[i] = acc[i] + (float)grp[i] * b_scale[(ulong)g * N + (ulong)n];
        }
    }
    i8_epilogue_f32(acc, tx, ty, (int)M, (int)N, a_rows, a_scalar, a_mode, bias, has_bias, C, out_kind);
}

#define I8_ARGS                                                                                                    \
    device int8_t *A [[buffer(0)]], device int8_t *B [[buffer(1)]], device char *C [[buffer(2)]],                 \
        constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],           \
        constant uint &tiles_n [[buffer(6)]], constant uint &tiles_m [[buffer(7)]],                               \
        device float *a_rows [[buffer(8)]], constant float &a_scalar [[buffer(9)]],                               \
        constant uint &a_mode [[buffer(10)]], device float *b_scale [[buffer(11)]],                               \
        constant uint &b_mode [[buffer(12)]], device float *bias [[buffer(13)]], constant uint &has_bias [[buffer(14)]], \
        constant uint &out_kind [[buffer(15)]], uint tgpig [[threadgroup_position_in_grid]]

#define I8_FULL(NAME, SM, SN, FN)                                                                                  \
    kernel void NAME(I8_ARGS) {                                                                                    \
        FN<SM, SN>(A, B, C, a_rows, a_scalar, a_mode, b_scale, b_mode, bias, has_bias, out_kind, M, N, K, tiles_n, \
                   tiles_m, tgpig);                                                                                \
    }

#define I8_GROUP(NAME, SM, SN, G, FN)                                                                              \
    kernel void NAME(I8_ARGS) {                                                                                    \
        FN<SM, SN, G>(A, B, C, a_rows, a_scalar, a_mode, b_scale, bias, has_bias, out_kind, M, N, K, tiles_n,      \
                      tiles_m, tgpig);                                                                             \
    }

I8_FULL(i8_w8a8_64, 64, 64, mm_nn_full)
I8_FULL(i8_w8a8_64_nt, 64, 64, mm_nt_full)
I8_FULL(i8_w8a8_128, 128, 64, mm_nn_full)
I8_FULL(i8_w8a8_128_nt, 128, 64, mm_nt_full)

I8_GROUP(i8_w8a8_g64_64, 64, 64, 64, mm_nn_group)
I8_GROUP(i8_w8a8_g64_64_nt, 64, 64, 64, mm_nt_group)
I8_GROUP(i8_w8a8_g64_128, 128, 64, 64, mm_nn_group)
I8_GROUP(i8_w8a8_g64_128_nt, 128, 64, 64, mm_nt_group)
I8_GROUP(i8_w8a8_g128_64, 64, 64, 128, mm_nn_group)
I8_GROUP(i8_w8a8_g128_64_nt, 64, 64, 128, mm_nt_group)
I8_GROUP(i8_w8a8_g128_128, 128, 64, 128, mm_nn_group)
I8_GROUP(i8_w8a8_g128_128_nt, 128, 64, 128, mm_nt_group)
I8_GROUP(i8_w8a8_g256_64, 64, 64, 256, mm_nn_group)
I8_GROUP(i8_w8a8_g256_64_nt, 64, 64, 256, mm_nt_group)
I8_GROUP(i8_w8a8_g256_128, 128, 64, 256, mm_nn_group)
I8_GROUP(i8_w8a8_g256_128_nt, 128, 64, 256, mm_nt_group)

// One thread per output column. Sequential int32 dots, same flush order as the
// tiled kernels. M is 1; the host does not dispatch this for any other M.
kernel void i8_gemv_dequant(device int8_t *A [[buffer(0)]], device int8_t *B [[buffer(1)]],
                            device char *C [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
                            device float *a_rows [[buffer(5)]], constant float &a_scalar [[buffer(6)]],
                            constant uint &a_mode [[buffer(7)]], device float *b_scale [[buffer(8)]],
                            constant uint &b_mode [[buffer(9)]], constant uint &group [[buffer(10)]],
                            device float *bias [[buffer(11)]], constant uint &has_bias [[buffer(12)]],
                            constant uint &out_kind [[buffer(13)]], constant uint &trans_b [[buffer(14)]],
                            uint n [[thread_position_in_grid]]) {
    if (n >= N) return;
    float ascl = (a_mode != 0u) ? a_rows[0] : a_scalar;
    float acc = 0.0f;
    if (b_mode == 2u) {
        uint groups = K / group;
        for (uint g = 0; g < groups; ++g) {
            int s = 0;
            uint base = g * group;
            for (uint kk = 0; kk < group; ++kk) {
                uint k = base + kk;
                int av = (int)A[k];
                int bv = (trans_b != 0u) ? (int)B[(ulong)n * K + k] : (int)B[(ulong)k * N + n];
                s += av * bv;
            }
            acc = acc + (float)s * b_scale[(ulong)g * N + n];
        }
        acc = acc * ascl;
    } else {
        int s = 0;
        for (uint k = 0; k < K; ++k) {
            int av = (int)A[k];
            int bv = (trans_b != 0u) ? (int)B[(ulong)n * K + k] : (int)B[(ulong)k * N + n];
            s += av * bv;
        }
        float bscl = (b_mode == 1u) ? b_scale[n] : 1.0f;
        acc = ((float)s * ascl) * bscl;
    }
    if (has_bias != 0u) acc += bias[n];
    i8_store(C, 0, (int)n, (int)N, acc, out_kind);
}

// Per-row absmax quantize. scale = absmax/127, or 1 when the row is all zeros.
// q = clamp(rint(x/scale), -127, 127). A non-finite element sets bad[row] and
// that row is not written. One threadgroup of 256 per row.
kernel void i8_quantize_rows(device float *x [[buffer(0)]], device int8_t *q [[buffer(1)]],
                             device float *scales [[buffer(2)]], device uint *bad [[buffer(3)]],
                             constant uint &cols [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[256];
    threadgroup uint bred[256];
    float m = 0.0f;
    uint b = 0u;
    for (uint c = tid; c < cols; c += 256u) {
        float v = x[(ulong)row * cols + c];
        if (!isfinite(v)) b = 1u;
        else m = fmax(m, fabs(v));
    }
    red[tid] = m;
    bred[tid] = b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            red[tid] = fmax(red[tid], red[tid + stride]);
            bred[tid] |= bred[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        float scale = (red[0] == 0.0f) ? 1.0f : red[0] / 127.0f;
        scales[row] = scale;
        bad[row] = bred[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (bad[row] != 0u) return;
    float scale = scales[row];
    for (uint c = tid; c < cols; c += 256u) {
        float v = x[(ulong)row * cols + c] / scale;
        int qi = (int)rint(v);
        if (qi > 127) qi = 127;
        if (qi < -127) qi = -127;
        q[(ulong)row * cols + c] = (int8_t)qi;
    }
}
