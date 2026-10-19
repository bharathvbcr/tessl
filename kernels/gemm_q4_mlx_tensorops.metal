// MLX Q4 GEMM on the TensorOps matrix units, dispatched by `nn::gemm_q4_mlx_tensorops`.
//
//   Y[M, N] = X[M, K] (bf16) @ W[N, K]^T,   W = scale * nibble + bias per group of G along K
//
// The bank is read in place: `uint4b_format` over the bank's own packed bytes (low nibble is
// element 0, nibbles unsigned for MLX; measured in tests/int4b_tensorops.rs), no relayout.
// TensorOps takes the 4-bit operand on the RIGHT only and has no per-group affine, so one
// `matmul2d` runs per K group into a fresh cooperative destination `grp` (P_g = X_g . q_g^T)
// and the epilogue folds the affine in registers:
//
//   acc += s[n, g] * P_g[m, n] + b[n, g] * xsum[m, g]
//
// `xsum[m, g]` (the sum of the group's bf16 activations, the bias term's multiplier) is
// computed IN KERNEL from the activations the threadgroup already addresses: four lanes per
// row sum a quarter of the group each in f32 and combine by xor-shuffle, publishing one value
// per row through threadgroup memory, double-buffered by group parity so one barrier per
// group orders both the write-before-read and the read-before-overwrite.
//
// Tile 32x32 on four simdgroups (128 threads). Rows past M and columns past N are read from
// partial slices and never dereferenced by the epilogue (guarded by `m < M`, `n < N`); the
// final store is clipped by the C slice's extents.
//
// Note: device pointers must be non-const -- `const` poisons MPP type matching.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int TILE = 32;

template <int G>
inline void mlx_q4_tensorops(device bfloat *A, device uchar *W, device float *C,
                             device const bfloat2 *sb, uint M_, uint N_, uint K_,
                             threadgroup float (&xs)[2][TILE], uint2 tg, uint tid) {
    constexpr auto d = matmul2d_descriptor(TILE, TILE, G, false, true, false,
                                           matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    using TW = tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline>;
    const int M = (int)M_, N = (int)N_, K = (int)K_;
    auto tA = tensor(A, dextents<int32_t, 2>{K, M}, array<int, 2>{1, K});
    TW tW((TW::data_handle_type)W, dextents<int32_t, 2>{K, N}, array<int, 2>{1, K});
    auto tC = tensor(C, dextents<int32_t, 2>{N, M}, array<int, 2>{1, N});
    const int tx = (int)tg.x * TILE, ty = (int)tg.y * TILE;
    auto sA0 = tA.slice(0, ty);
    auto sW0 = tW.slice(0, tx);
    auto acc = op.template get_destination_cooperative_tensor<decltype(sA0), decltype(sW0), float>();
    auto grp = op.template get_destination_cooperative_tensor<decltype(sA0), decltype(sW0), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc.set(i, 0.0f);

    // Row `r` of the tile is summed by lanes 4r..4r+3 of the threadgroup (same simdgroup:
    // 8 rows per simdgroup), each over G/4 contiguous activations.
    const int r = (int)(tid >> 2), part = (int)(tid & 3);
    const int row = ty + r;
    const int groups = K / G;
    for (int g = 0; g < groups; ++g) {
        float s = 0.0f;
        if (row < M) {
            device const bfloat *xp = A + (long)row * K + g * G + part * (G / 4);
            for (int j = 0; j < G / 4; ++j) s += float(xp[j]);
        }
        s += simd_shuffle_xor(s, 1);
        s += simd_shuffle_xor(s, 2);
        if (part == 0) xs[g & 1][r] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto a = tA.slice(g * G, ty);
        auto w = tW.slice(g * G, tx);
        op.run(a, w, grp);
        for (uint16_t i = 0; i < grp.get_capacity(); ++i) {
            if (!grp.is_valid_element(i)) continue;
            const auto idx = grp.get_multidimensional_index(i);  // {n, m}
            const int ln = (int)idx[0], lm = (int)idx[1];
            const int n = tx + ln, m = ty + lm;
            if (n >= N || m >= M) continue;
            const bfloat2 p = sb[n * groups + g];
            acc[i] = fma(float(p.x), grp[i], fma(float(p.y), xs[g & 1][lm], acc[i]));
        }
    }
    acc.store(tC.slice(tx, ty));
}

#define MLX_Q4_TENSOROPS(G)                                                                  \
kernel void gemm_q4_mlx_tensorops_g##G(                                                      \
    device uchar *packed [[buffer(0)]],                                                      \
    device const bfloat2 *sb [[buffer(1)]],                                                  \
    device bfloat *x [[buffer(2)]],                                                          \
    device float *y [[buffer(3)]],                                                           \
    constant uint &M [[buffer(4)]],                                                          \
    constant uint &N [[buffer(5)]],                                                          \
    constant uint &K [[buffer(6)]],                                                          \
    uint2 tg [[threadgroup_position_in_grid]],                                               \
    uint tid [[thread_index_in_threadgroup]]) {                                              \
    threadgroup float xs[2][TILE];                                                           \
    mlx_q4_tensorops<G>(x, packed, y, sb, M, N, K, xs, tg, tid);                             \
}

MLX_Q4_TENSOROPS(32)
MLX_Q4_TENSOROPS(64)
MLX_Q4_TENSOROPS(128)
