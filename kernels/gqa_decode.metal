// Decode-only GQA. Query length is 1: Q is [batch, q_heads, D], K and V are
// [batch, kv_len, kv_heads, D], O is [batch, q_heads, D]. One thread owns one
// (batch, query head) and scans keys in order. Three passes recompute the dot
// so the running max, the exp sum, and the value mix share one reduction
// order. fp-contract is off for this file. Head dimensions outside
// {32, 64, 128, 256, 512} are refused on the host and never dispatched.
// Prefill kernels are not in this file.

#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off) reassociate(off)

template <int D>
inline void gqa_decode_body(device float *Q, device float *K, device float *V, device float *O, uint batch,
                            uint q_heads, uint kv_heads, uint kv_len, uint gid) {
    uint total = batch * q_heads;
    if (gid >= total) return;
    uint b = gid / q_heads;
    uint h = gid - b * q_heads;
    uint group = q_heads / kv_heads;
    uint kv_h = h / group;
    const float scale = 1.0f / sqrt((float)D);
    device float *q = Q + ((ulong)b * q_heads + h) * D;
    device float *o = O + ((ulong)b * q_heads + h) * D;

    float m = -INFINITY;
    for (uint t = 0u; t < kv_len; ++t) {
        device float *k = K + (((ulong)b * kv_len + t) * kv_heads + kv_h) * D;
        float dot = 0.0f;
        for (int i = 0; i < D; ++i) dot = dot + q[i] * k[i];
        m = fmax(m, dot * scale);
    }
    float se = 0.0f;
    for (uint t = 0u; t < kv_len; ++t) {
        device float *k = K + (((ulong)b * kv_len + t) * kv_heads + kv_h) * D;
        float dot = 0.0f;
        for (int i = 0; i < D; ++i) dot = dot + q[i] * k[i];
        se = se + exp(dot * scale - m);
    }
    for (int i = 0; i < D; ++i) o[i] = 0.0f;
    for (uint t = 0u; t < kv_len; ++t) {
        device float *k = K + (((ulong)b * kv_len + t) * kv_heads + kv_h) * D;
        device float *v = V + (((ulong)b * kv_len + t) * kv_heads + kv_h) * D;
        float dot = 0.0f;
        for (int i = 0; i < D; ++i) dot = dot + q[i] * k[i];
        float w = exp(dot * scale - m) / se;
        for (int i = 0; i < D; ++i) o[i] = o[i] + w * v[i];
    }
}

#define GQA_DECODE(D)                                                                                              \
    kernel void gqa_decode_h##D(device float *Q [[buffer(0)]], device float *K [[buffer(1)]],                      \
                                device float *V [[buffer(2)]], device float *O [[buffer(3)]],                      \
                                constant uint &batch [[buffer(4)]], constant uint &q_heads [[buffer(5)]],          \
                                constant uint &kv_heads [[buffer(6)]], constant uint &kv_len [[buffer(7)]],        \
                                uint gid [[thread_position_in_grid]]) {                                            \
        gqa_decode_body<D>(Q, K, V, O, batch, q_heads, kv_heads, kv_len, gid);                                     \
    }

GQA_DECODE(32)
GQA_DECODE(64)
GQA_DECODE(128)
GQA_DECODE(256)
GQA_DECODE(512)
