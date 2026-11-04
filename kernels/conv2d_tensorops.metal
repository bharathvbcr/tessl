// 2D convolution on the GPU's matrix units: Metal Performance Primitives' `convolution2d` tensor op
// (Metal 4 / macOS 26+), driven from NCHW f32 tensors. src/conv.rs owns the dispatch.
//
// The op takes NHWC activations, HWIO weights and writes NHWO, and its semantics were pinned by
// probe (bench/results/conv2d_20261104/README.md) rather than documented:
//   - it centres the kernel: destination (h, w) reads source (h + kh - K/2, w + kw - K/2) of the
//     activation slice, and reads before the slice origin come back zero;
//   - `set_offsets(K/2, K/2)` cancels the centring, so on a source padded by P a destination tile at
//     (h0, w0) is the valid convolution of the padded tensor at (h0, w0) — PyTorch's padding P;
//   - a destination tile that runs past the tensor's width computes wrong values (f16 at any width
//     that is not a whole number of tiles, f32 at some), so every tile here is full: the packed
//     source is padded in width to a whole number of tiles, and the store drops the extra columns;
//   - an NCHW destination tensor view (channel stride not 1) is not written correctly, so the op
//     writes a cooperative destination and the kernel stores it to NCHW itself.
// One conv is three dispatches: weights (OIHW f32 -> HWIO T), pack (NCHW f32 -> padded NHWC T, a
// 32x32 threadgroup transpose so both sides are coalesced), and the op with its NCHW store and bias.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// ---------------------------------------------------------------------------------------------
// Pack: x NCHW f32 [N, C, H, W] -> xp NHWC T [N, Hp, Wp, C], x placed at (P, P), zero elsewhere.
// Grid: (ceil(Wp / 32), ceil(C / 32), N * Hp) threadgroups of 32 x 8 threads.
// ---------------------------------------------------------------------------------------------
template <typename T>
inline void conv2d_pack(device const float *x, device T *xp, uint C, uint H, uint W, uint P,
                        uint Hp, uint Wp, uint3 tg, uint2 tid, threadgroup float (*tile)[33]) {
    const uint n = tg.z / Hp, hp = tg.z % Hp;
    const uint wp0 = tg.x * 32, c0 = tg.y * 32;
    const int h = (int)hp - (int)P;
    // Read: thread (tid.x = w, tid.y = c row) walks w contiguously.
    for (uint r = tid.y; r < 32; r += 8) {
        const uint c = c0 + r;
        const int w = (int)(wp0 + tid.x) - (int)P;
        float v = 0.0f;
        if (c < C && h >= 0 && h < (int)H && w >= 0 && w < (int)W)
            v = x[((ulong)(n * C + c) * H + (uint)h) * W + (uint)w];
        tile[r][tid.x] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Write: thread (tid.x = c) walks c contiguously.
    for (uint r = tid.y; r < 32; r += 8) {
        const uint wp = wp0 + r, c = c0 + tid.x;
        if (wp < Wp && c < C)
            xp[((ulong)(n * Hp + hp) * Wp + wp) * C + c] = T(tile[tid.x][r]);
    }
}

#define CONV2D_PACK(NAME, T)                                                                    \
    kernel void NAME(device const float *x [[buffer(0)]], device T *xp [[buffer(1)]],            \
                     constant uint &C [[buffer(2)]], constant uint &H [[buffer(3)]],             \
                     constant uint &W [[buffer(4)]], constant uint &P [[buffer(5)]],             \
                     constant uint &Hp [[buffer(6)]], constant uint &Wp [[buffer(7)]],           \
                     uint3 tg [[threadgroup_position_in_grid]],                                  \
                     uint3 tid3 [[thread_position_in_threadgroup]]) {                            \
        threadgroup float tile[32][33];                                                         \
        conv2d_pack<T>(x, xp, C, H, W, P, Hp, Wp, tg, tid3.xy, tile);                          \
    }

CONV2D_PACK(conv2d_pack_nhwc_f16, half)
CONV2D_PACK(conv2d_pack_nhwc_f32, float)

// ---------------------------------------------------------------------------------------------
// Weights: OIHW f32 [CO, CI, K, K] -> HWIO T [K, K, CI, CO]. One thread per output element.
// ---------------------------------------------------------------------------------------------
template <typename T>
inline void conv2d_weights(device const float *w, device T *wh, uint CO, uint CI, uint K, uint gid) {
    const uint total = CO * CI * K * K;
    if (gid >= total) return;
    const uint o = gid % CO;
    const uint rest = gid / CO;
    const uint c = rest % CI, kk = rest / CI;  // kk = kh * K + kw
    wh[gid] = T(w[(o * CI + c) * K * K + kk]);
}

#define CONV2D_WEIGHTS(NAME, T)                                                                 \
    kernel void NAME(device const float *w [[buffer(0)]], device T *wh [[buffer(1)]],            \
                     constant uint &CO [[buffer(2)]], constant uint &CI [[buffer(3)]],           \
                     constant uint &K [[buffer(4)]], uint gid [[thread_position_in_grid]]) {     \
        conv2d_weights<T>(w, wh, CO, CI, K, gid);                                               \
    }

CONV2D_WEIGHTS(conv2d_weights_hwio_f16, half)
CONV2D_WEIGHTS(conv2d_weights_hwio_f32, float)

// ---------------------------------------------------------------------------------------------
// The op: xp NHWC T [N, Hp, Wp, CI] (Hp = OH + K - 1, Wp = OWp + K - 1), wh HWIO T [K, K, CI, CO],
// y NCHW f32 [N, CO, OH, OW]. Stride 1. One threadgroup of NSG simdgroups per OT x WT x 1 tile;
// CO % OT == 0 and OWp % WT == 0 (the host guarantees both). Grid: (CO / OT, OWp / WT, N * OH).
// The op writes a cooperative destination (registers), which is stored straight to NCHW with the
// bias added. The tile is computed whole; only its columns below OW are stored.
// ---------------------------------------------------------------------------------------------
template <typename T, int K, int OT, int WT, int NSG>
inline void conv2d_tops_nchw(device T *xp, device T *wh, device float *y, device const float *bias,
                             uint has_bias, int CI, int CO, int OH, int OW, int OWp, int N, uint3 tg) {
    constexpr auto desc = convolution2d_descriptor(int4(OT, WT, 1, 1),
                                                   int4(dynamic_length_v<int>, WT + K - 1, K, 1),
                                                   int2(K, K));
    convolution2d<desc, execution_simdgroups<NSG>> op;
    op.set_offsets(int2(K / 2, K / 2));
    const int Hp = OH + K - 1, Wp = OWp + K - 1;
    const int o0 = (int)tg.x * OT, w0 = (int)tg.y * WT;
    const int n = (int)tg.z / OH, h0 = (int)tg.z % OH;
    auto tX = tensor(xp, dextents<int, 4>{CI, Wp, Hp, N}, array<int, 4>{1, CI, CI * Wp, CI * Wp * Hp});
    auto tW = tensor(wh, dextents<int, 4>{CO, CI, K, K}, array<int, 4>{1, CO, CO * CI, CO * CI * K});
    auto sX = tX.slice(0, w0, h0, n);
    auto sW = tW.slice(o0, 0, 0, 0);
    auto cT = op.template get_destination_cooperative_tensor<decltype(sX), decltype(sW), float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    op.run(sX, sW, cT);
    device float *yn = y + (ulong)n * CO * OH * OW + (ulong)h0 * OW;
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (!cT.is_valid_element(i)) continue;
        auto idx = cT.get_multidimensional_index(i);  // (o, w, h, n) within the tile
        const int o = o0 + idx[0], w = w0 + idx[1];
        if (w < OW) yn[(ulong)o * OH * OW + w] = has_bias != 0u ? cT[i] + bias[o] : cT[i];
    }
}

// Device pointers are non-const: `const` poisons MPP type matching (matmul_tensorops.metal).
#define CONV2D_TOPS_NCHW(NAME, T, K, OT, WT, NSG)                                               \
    kernel void NAME(device T *xp [[buffer(0)]], device T *wh [[buffer(1)]],                     \
                     device float *y [[buffer(2)]], device const float *bias [[buffer(3)]],      \
                     constant uint &has_bias [[buffer(4)]], constant uint &CI [[buffer(5)]],     \
                     constant uint &CO [[buffer(6)]], constant uint &OH [[buffer(7)]],           \
                     constant uint &OW [[buffer(8)]], constant uint &OWp [[buffer(9)]],          \
                     constant uint &N [[buffer(10)]],                                            \
                     uint3 tg [[threadgroup_position_in_grid]]) {                                \
        conv2d_tops_nchw<T, K, OT, WT, NSG>(xp, wh, y, bias, has_bias, (int)CI, (int)CO, (int)OH, \
                                            (int)OW, (int)OWp, (int)N, tg);                      \
    }

// The probe's fastest tile at the SAM 3x3 shape (288^2, 256 -> 256: 3.26 ms f16, 30 TFLOP/s), and
// a 16-channel tile for narrow outputs such as the 32 -> 16 conv at 1152^2.
CONV2D_TOPS_NCHW(conv2d_tops_nchw_k3_o64_w32_f16, half, 3, 64, 32, 4)
CONV2D_TOPS_NCHW(conv2d_tops_nchw_k3_o16_w32_f16, half, 3, 16, 32, 4)
CONV2D_TOPS_NCHW(conv2d_tops_nchw_k3_o64_w32_f32, float, 3, 64, 32, 4)
CONV2D_TOPS_NCHW(conv2d_tops_nchw_k3_o16_w32_f32, float, 3, 16, 32, 4)
