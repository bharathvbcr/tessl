// W4A8: int8 activations x int4b weights, int32 accumulate, same epilogue as
// i8_w8a8.metal. Weights are packed [N, K], low nibble first, two's complement
// (nibble 8..15 is -8..-1). The right operand is the only legal 4-bit side.
// Tile is 32x32 on four simdgroups. A static descriptor K is a multiple of 32;
// the host rejects any other K on this arm.
//
// The zero-point-8 unpack is a separate kernel. It is not int4b: unsigned
// nibble minus 8, then the int8 GEMM. The two encodings are not interchangeable.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "i8_tile_walk.h"

using namespace metal;
using namespace mpp::tensor_ops;

#pragma clang fp contract(off) reassociate(off)

inline void w4_store(device char *C, int m, int n, int N, float v, uint out_kind) {
    ulong i = (ulong)m * (ulong)N + (ulong)n;
    if (out_kind == 1u) ((device half *)C)[i] = half(v);
    else if (out_kind == 2u) ((device bfloat *)C)[i] = bfloat(v);
    else ((device float *)C)[i] = v;
}

template <typename Acc>
inline void w4_epilogue_i32(thread Acc &acc, int tx, int ty, int M, int N, device float *a_rows, float a_scalar,
                            uint a_mode, device float *bias, uint has_bias, device char *C, uint out_kind) {
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        const auto idx = acc.get_multidimensional_index(i);
        const int n = tx + (int)idx[0];
        const int m = ty + (int)idx[1];
        if (m < 0 || n < 0 || m >= M || n >= N) continue;
        float ascl = (a_mode != 0u) ? a_rows[m] : a_scalar;
        float v = (float)acc[i] * ascl;
        if (has_bias != 0u) v += bias[n];
        w4_store(C, m, n, N, v, out_kind);
    }
}

template <typename Acc>
inline void w4_epilogue_f32(thread Acc &acc, int tx, int ty, int M, int N, device float *a_rows, float a_scalar,
                            uint a_mode, device float *bias, uint has_bias, device char *C, uint out_kind) {
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        const auto idx = acc.get_multidimensional_index(i);
        const int n = tx + (int)idx[0];
        const int m = ty + (int)idx[1];
        if (m < 0 || n < 0 || m >= M || n >= N) continue;
        float ascl = (a_mode != 0u) ? a_rows[m] : a_scalar;
        float v = acc[i] * ascl;
        if (has_bias != 0u) v += bias[n];
        w4_store(C, m, n, N, v, out_kind);
    }
}

using W4 = tensor<device int4b_format, dextents<int32_t, 2>, tensor_inline>;

inline W4 w4_weights(device uchar *W, int N, int K) {
    return W4((W4::data_handle_type)W, dextents<int32_t, 2>{K, N}, array<int, 2>{1, K});
}

template <int G>
inline void w4_full_body(device int8_t *A, device uchar *W, device char *C, device float *a_rows, float a_scalar,
                         uint a_mode, device float *bias, uint has_bias, uint out_kind, uint M, uint N, uint K,
                         uint tiles_n, uint tiles_m, uint tgpig) {
    constexpr auto d = matmul2d_descriptor(32, 32, dynamic_length_v<int>, false, true, false,
                                           matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    uint2 tile = i8_tile_walk<32>(tgpig, tiles_n, tiles_m, N, K);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * 32;
    int ty = (int)tile.y * 32;
    auto tA = tensor(A, dextents<int32_t, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
    W4 tW = w4_weights(W, (int)N, (int)K);
    auto sA = tA.slice(0, ty);
    auto sW = tW.slice(0, tx);
    auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(sA)>,
                                                              metal::remove_addrspace_t<decltype(sW)>, int32_t>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc.set(i, 0);
    op.run(sA, sW, acc);
    w4_epilogue_i32(acc, tx, ty, (int)M, (int)N, a_rows, a_scalar, a_mode, bias, has_bias, C, out_kind);
    (void)G;
}

template <int G>
inline void w4_group_body(device int8_t *A, device uchar *W, device char *C, device float *a_rows, float a_scalar,
                          uint a_mode, device float *b_scale, device float *bias, uint has_bias, uint out_kind, uint M,
                          uint N, uint K, uint tiles_n, uint tiles_m, uint tgpig) {
    constexpr auto d = matmul2d_descriptor(32, 32, G, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    uint2 tile = i8_tile_walk<32>(tgpig, tiles_n, tiles_m, N, K);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * 32;
    int ty = (int)tile.y * 32;
    auto tA = tensor(A, dextents<int32_t, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
    W4 tW = w4_weights(W, (int)N, (int)K);
    auto sA0 = tA.slice(0, ty);
    auto sW0 = tW.slice(0, tx);
    auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(sA0)>,
                                                              metal::remove_addrspace_t<decltype(sW0)>, float>();
    auto grp = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(sA0)>,
                                                              metal::remove_addrspace_t<decltype(sW0)>, int32_t>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc.set(i, 0.0f);
    const int groups = (int)K / G;
    for (int g = 0; g < groups; ++g) {
        auto sA = tA.slice(g * G, ty);
        auto sW = tW.slice(g * G, tx);
        for (uint16_t i = 0; i < grp.get_capacity(); ++i) grp.set(i, 0);
        op.run(sA, sW, grp);
        for (uint16_t i = 0; i < grp.get_capacity(); ++i) {
            if (!grp.is_valid_element(i)) continue;
            const auto idx = grp.get_multidimensional_index(i);
            const int n = tx + (int)idx[0];
            const int m = ty + (int)idx[1];
            if (m < 0 || n < 0 || m >= (int)M || n >= (int)N) continue;
            acc[i] = acc[i] + (float)grp[i] * b_scale[(ulong)g * N + (ulong)n];
        }
    }
    w4_epilogue_f32(acc, tx, ty, (int)M, (int)N, a_rows, a_scalar, a_mode, bias, has_bias, C, out_kind);
}

#define W4_ARGS                                                                                                    \
    device int8_t *A [[buffer(0)]], device uchar *W [[buffer(1)]], device char *C [[buffer(2)]],                  \
        constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],           \
        constant uint &tiles_n [[buffer(6)]], constant uint &tiles_m [[buffer(7)]],                               \
        device float *a_rows [[buffer(8)]], constant float &a_scalar [[buffer(9)]],                               \
        constant uint &a_mode [[buffer(10)]], device float *b_scale [[buffer(11)]],                               \
        device float *bias [[buffer(12)]], constant uint &has_bias [[buffer(13)]],                                \
        constant uint &out_kind [[buffer(14)]], uint tgpig [[threadgroup_position_in_grid]]

kernel void w4a8_i4_32(W4_ARGS) {
    w4_full_body<0>(A, W, C, a_rows, a_scalar, a_mode, bias, has_bias, out_kind, M, N, K, tiles_n, tiles_m, tgpig);
}

kernel void w4a8_i4_g64(W4_ARGS) {
    w4_group_body<64>(A, W, C, a_rows, a_scalar, a_mode, b_scale, bias, has_bias, out_kind, M, N, K, tiles_n, tiles_m,
                      tgpig);
}

kernel void w4a8_i4_g128(W4_ARGS) {
    w4_group_body<128>(A, W, C, a_rows, a_scalar, a_mode, b_scale, bias, has_bias, out_kind, M, N, K, tiles_n,
                       tiles_m, tgpig);
}

kernel void w4a8_i4_g256(W4_ARGS) {
    w4_group_body<256>(A, W, C, a_rows, a_scalar, a_mode, b_scale, bias, has_bias, out_kind, M, N, K, tiles_n,
                       tiles_m, tgpig);
}

// Unsigned nibble minus 8, low nibble is element 0, into int8 [N, K].
kernel void w4a8_unpack_zp8(device uchar *packed [[buffer(0)]], device int8_t *out [[buffer(1)]],
                            constant uint &N [[buffer(2)]], constant uint &K [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    ulong elems = (ulong)N * (ulong)K;
    if ((ulong)gid >= elems) return;
    uint n = gid / K;
    uint k = gid - n * K;
    uchar byte = packed[(ulong)n * (K / 2u) + (k >> 1u)];
    uint nib = (k & 1u) ? (uint)(byte >> 4u) : (uint)(byte & 0x0fu);
    out[gid] = (int8_t)((int)nib - 8);
}
