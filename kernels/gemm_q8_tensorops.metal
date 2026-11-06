// Q8 GEMM on the TensorOps matrix units, dispatched by `nn::gemm_q8_tensorops`.
//
//   Y[M, N] = X[M, K] (f32) @ W[N, K]^T
//   W = scale * (q - zero) per group of G along K, q an int8 code
//
// The bank is read in place as `int8_t`. MPP `matmul2d` accepts float x int8_t
// with a float destination, and it has no per-group affine, so one matmul2d
// runs per K group into a fresh cooperative destination (P_g = X_g . q_g^T)
// and the epilogue folds the affine in registers:
//
//   acc += s[n, g] * P_g[m, n] + (-s[n, g] * z[n, g]) * xsum[m, g]
//
// which is `scale * (q - zero)` summed against the activations. `xsum` is the
// group's activation sum, computed in kernel: four lanes per row sum a quarter
// of the group each and combine by xor-shuffle. One value per row is published
// through threadgroup memory, double-buffered by group parity, so one barrier
// per group orders both the write-before-read and the read-before-overwrite.
//
// Tile 32x32 on four simdgroups (128 threads). Rows past M and columns past N
// are skipped by the epilogue. The final store is clipped by the C slice.
//
// Device pointers are non-const: const poisons MPP type matching. Direct
// device loads, no threadgroup staging of the operands. Every lane reaches
// matmul2d; the group loop is uniform across the threadgroup.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int TILE = 32;

template <int G>
inline void q8_tensorops(device float *A, device int8_t *W, device float *C,
                         device float *scales, device float *zeros, uint M_, uint N_, uint K_,
                         threadgroup float (&xs)[2][TILE], uint2 tg, uint tid) {
    constexpr auto d = matmul2d_descriptor(TILE, TILE, G, false, true, false,
                                           matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    const int M = (int)M_, N = (int)N_, K = (int)K_;
    auto tA = tensor(A, dextents<int32_t, 2>{K, M}, array<int, 2>{1, K});
    auto tW = tensor(W, dextents<int32_t, 2>{K, N}, array<int, 2>{1, K});
    auto tC = tensor(C, dextents<int32_t, 2>{N, M}, array<int, 2>{1, N});
    const int tx = (int)tg.x * TILE, ty = (int)tg.y * TILE;
    auto sA0 = tA.slice(0, ty);
    auto sW0 = tW.slice(0, tx);
    auto acc = op.template get_destination_cooperative_tensor<decltype(sA0), decltype(sW0), float>();
    auto grp = op.template get_destination_cooperative_tensor<decltype(sA0), decltype(sW0), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc.set(i, 0.0f);

    // Row `r` of the tile is summed by lanes 4r..4r+3 (one simdgroup holds 8
    // rows). Each lane sums G/4 contiguous activations.
    const int r = (int)(tid >> 2), part = (int)(tid & 3);
    const int row = ty + r;
    const int groups = K / G;
    for (int g = 0; g < groups; ++g) {
        float s = 0.0f;
        if (row < M) {
            device const float *xp = A + (long)row * K + g * G + part * (G / 4);
            for (int j = 0; j < G / 4; ++j) s += xp[j];
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
            const auto idx = grp.get_multidimensional_index(i); // {n, m}
            const int ln = (int)idx[0], lm = (int)idx[1];
            const int n = tx + ln, m = ty + lm;
            if (n >= N || m >= M) continue;
            const float scl = scales[n * groups + g];
            const float z = zeros[n * groups + g];
            acc[i] = fma(scl, grp[i], fma(-scl * z, xs[g & 1][lm], acc[i]));
        }
    }
    acc.store(tC.slice(tx, ty));
}

#define Q8_TENSOROPS(G)                                                                          \
kernel void gemm_q8_tensorops_g##G(                                                              \
    device int8_t *packed [[buffer(0)]],                                                         \
    device float *scales [[buffer(1)]],                                                          \
    device float *zeros [[buffer(2)]],                                                           \
    device float *x [[buffer(3)]],                                                               \
    device float *y [[buffer(4)]],                                                               \
    constant uint &M [[buffer(5)]],                                                              \
    constant uint &N [[buffer(6)]],                                                              \
    constant uint &K [[buffer(7)]],                                                              \
    uint2 tg [[threadgroup_position_in_grid]],                                                   \
    uint tid [[thread_index_in_threadgroup]]) {                                                  \
    threadgroup float xs[2][TILE];                                                               \
    q8_tensorops<G>(x, packed, y, scales, zeros, M, N, K, xs, tg, tid);                          \
}

Q8_TENSOROPS(32)
Q8_TENSOROPS(64)
Q8_TENSOROPS(128)
