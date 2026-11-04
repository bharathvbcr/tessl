// Throwaway probe kernels (nothing here ships): does Metal Performance Primitives' convolution2d
// tensor op compute a SAM-shaped 3x3 conv correctly, and how fast, against a matmul2d implicit GEMM.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// ---- A. MPP convolution2d: X is NHWC, already padded ([N, Hp, Wp, CI]); W is HWIO ([KH, KW, CI, CO]);
// Y is NHWO ([N, OH, OW, CO]) f32. One threadgroup per (OT channels x WT x HT) destination tile.
// The descriptor carries tile dimensions; the operands are slices at the tile origin.
#define MPP_CONV(NAME, T, CI, OT, WT, HT, NSG)                                                      \
    kernel void NAME(device T *X [[buffer(0)]], device T *Wt [[buffer(1)]],                       \
                     device float *Y [[buffer(2)]], constant int4 &dims [[buffer(3)]],             \
                     constant int2 &pdims [[buffer(4)]], constant int4 &adj [[buffer(5)]],         \
                     uint3 tg [[threadgroup_position_in_grid]]) {                                  \
        constexpr auto desc = convolution2d_descriptor(int4(OT, WT, HT, 1),                         \
                                                       int4(CI, WT + 2, HT + 2, 1), int2(3, 3));   \
        convolution2d<desc, execution_simdgroups<NSG>> op;                                         \
        op.set_offsets(int2(adj.x, adj.y));                                                         \
        const int CO = dims.x, OW = dims.y, OH = dims.z, N = dims.w;                                \
        const int Wp = pdims.x, Hp = pdims.y;                                                       \
        const int o0 = (int)tg.x * OT, w0 = (int)tg.y * WT;                                         \
        const int htiles = (OH + HT - 1) / HT;                                                      \
        const int n = (int)tg.z / htiles, h0 = ((int)tg.z % htiles) * HT;                           \
        auto tX = tensor(X, dextents<int, 4>{CI, Wp, Hp, N},                                        \
                         array<int, 4>{1, CI, CI * Wp, CI * Wp * Hp});                              \
        auto tW = tensor(Wt, dextents<int, 4>{CO, CI, 3, 3},                                        \
                         array<int, 4>{1, CO, CO * CI, CO * CI * 3});                               \
        auto tY = tensor(Y, dextents<int, 4>{CO, OW, OH, N},                                        \
                         array<int, 4>{1, CO, CO * OW, CO * OW * OH});                              \
        auto sX = tX.slice(0, w0 + adj.z, h0 + adj.w, n);                                           \
        auto sW = tW.slice(o0, 0, 0, 0);                                                            \
        auto sY = tY.slice(o0, w0, h0, n);                                                          \
        op.run(sX, sW, sY);                                                                         \
    }

MPP_CONV(mpp_conv_f16_o64_w32_h1, half, 256, 64, 32, 1, 4)
MPP_CONV(mpp_conv_f16_o64_w16_h2, half, 256, 64, 16, 2, 4)
MPP_CONV(mpp_conv_f16_o32_w32_h2, half, 256, 32, 32, 2, 4)
MPP_CONV(mpp_conv_f32_o64_w32_h1, float, 256, 64, 32, 1, 4)

// NCHW f32 -> padded NHWC (T), zero borders. One thread per output element of the padded tensor.
template <typename T>
inline void pad_nhwc(device const float *x, device T *xp, int C, int H, int W, int P, uint gid) {
    const int Wp = W + 2 * P, Hp = H + 2 * P;
    const int c = (int)gid % C;
    const int rest = (int)gid / C;
    const int wp = rest % Wp, hp = (rest / Wp) % Hp, n = rest / (Wp * Hp);
    const int h = hp - P, w = wp - P;
    float v = (h >= 0 && h < H && w >= 0 && w < W) ? x[((n * C + c) * H + h) * W + w] : 0.0f;
    xp[gid] = T(v);
}
kernel void pad_nhwc_f16(device const float *x [[buffer(0)]], device half *xp [[buffer(1)]],
                         constant int4 &d [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    pad_nhwc<half>(x, xp, d.x, d.y, d.z, d.w, gid);
}
kernel void pad_nhwc_f32(device const float *x [[buffer(0)]], device float *xp [[buffer(1)]],
                         constant int4 &d [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    pad_nhwc<float>(x, xp, d.x, d.y, d.z, d.w, gid);
}
