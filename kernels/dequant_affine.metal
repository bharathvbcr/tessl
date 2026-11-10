// MLX-affine dequantisation: packed codes -> bf16, elementwise.
//
//   out[r, c] = bf16_rne( fma(float(scale[r, c / G]), float(q[r, c]), float(bias[r, c / G])) )
//
// Layout is MLX's, row-major: `packed` is u32 [rows, cols * BITS / 32] with the
// code of column c at bits [BITS * (c % (32 / BITS)), +BITS) of word
// c / (32 / BITS) (little-endian lanes); `scales` and `biases` are bf16
// [rows, cols / G]; `out` is bf16 [rows, cols]. Every row starts on a group
// boundary because the host refuses a `cols` that is not a multiple of G, so a
// flat index never straddles a group: the codes of one word share one scale and
// one bias (G is a multiple of the 8 codes a q4 word holds), and the group of
// flat element e is e >> group_shift.
//
// One rounding to bf16: s * q is exact (8-bit codes times an 8-bit mantissa),
// so the only roundings are the fused multiply-add's to f32 and the single
// f32 -> bf16 conversion; the two-rounding form bf16(bf16(s * q) + b) is wrong
// (qv2 FINDINGS section 3). The multiply-add IS fused on purpose: an unfused
// s * q overflows f32 where the exact product does not (scale 2^125, code 8,
// bias -bf16 max), and mx.dequantize returns the finite fused result there
// (tests/dequant_affine.rs, crafted golden). `bfloat(float)` is
// round-to-nearest-even, overflowing to +-infinity: the tests pin that on
// exact ties.
//
// Non-finite results are stored, not trapped (a sum beyond the bf16 range is
// +-infinity, as in `cast_f32_to_bf16`), and reported: a thread whose output
// holds an infinity or a NaN sets `*fault` to 1. The word is the caller's, is
// never cleared here, and is the same convention as `i8_quantize_rows`'s `bad`
// flag; the caller reads it once, after as many dequants as it likes have
// OR-ed into it. The test is on the exponent bits, an integer test fast math
// cannot fold (see `qwen35_round_stochastic`).
//
// Default (fast) math applies, as in the other kernels: the GPU flushes f32
// subnormals, so a subnormal scale, bias or result behaves as a signed zero.
// tests/dequant_affine.rs records what the device does against the IEEE host
// formula and against mlx.
#include <metal_stdlib>
using namespace metal;

// 8 bytes (q8) or 16 bytes (q4) of bf16 pairs in one vector store. The vector
// pointer type is `packed_`, so it needs only the 4-byte alignment every bf16
// pair already has.
inline void store_pairs(device bfloat *dst, thread const uint (&p)[2])
{
    *reinterpret_cast<device packed_uint2 *>(dst) = packed_uint2(p[0], p[1]);
}

inline void store_pairs(device bfloat *dst, thread const uint (&p)[4])
{
    *reinterpret_cast<device packed_uint4 *>(dst) = packed_uint4(p[0], p[1], p[2], p[3]);
}

// One word per thread: BITS = 8 -> 4 codes, 8 bytes out; BITS = 4 -> 8 codes,
// 16 bytes out.
template <int BITS>
inline void dequant_affine_word(
    device const uint *packed,
    device const bfloat *scales,
    device const bfloat *biases,
    device bfloat *out,
    device atomic_uint *fault,
    uint n_words,
    uint group_shift,
    uint gid)
{
    constexpr uint CODES = 32 / BITS;
    constexpr uint MASK = (1u << BITS) - 1u;
    if (gid >= n_words) return;

    const ulong first = (ulong)gid * CODES;
    const ulong group = first >> group_shift;
    const float s = float(scales[group]);
    const float b = float(biases[group]);
    const uint w = packed[gid];

    uint pairs[CODES / 2];
    uint non_finite = 0u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < CODES; i += 2) {
        const float q0 = float((w >> (BITS * i)) & MASK);
        const float q1 = float((w >> (BITS * (i + 1))) & MASK);
        const ushort lo = as_type<ushort>(bfloat(fma(s, q0, b)));
        const ushort hi = as_type<ushort>(bfloat(fma(s, q1, b)));
        non_finite |= ((uint)lo & 0x7f80u) == 0x7f80u;
        non_finite |= ((uint)hi & 0x7f80u) == 0x7f80u;
        pairs[i / 2] = (uint)lo | ((uint)hi << 16);
    }

    store_pairs(out + first, pairs);
    if (non_finite != 0u) {
        atomic_store_explicit(fault, 1u, memory_order_relaxed);
    }
}

#define TESSL_DEQUANT_AFFINE_KERNEL(NAME, BITS)                     \
    kernel void NAME(                                               \
        device const uint *packed [[buffer(0)]],                    \
        device const bfloat *scales [[buffer(1)]],                  \
        device const bfloat *biases [[buffer(2)]],                  \
        device bfloat *out [[buffer(3)]],                           \
        device atomic_uint *fault [[buffer(4)]],                    \
        constant uint &n_words [[buffer(5)]],                       \
        constant uint &group_shift [[buffer(6)]],                   \
        uint gid [[thread_position_in_grid]])                       \
    {                                                               \
        dequant_affine_word<BITS>(                                  \
            packed, scales, biases, out, fault, n_words, group_shift, gid); \
    }

TESSL_DEQUANT_AFFINE_KERNEL(dequant_affine_q8_bf16, 8)
TESSL_DEQUANT_AFFINE_KERNEL(dequant_affine_q4_bf16, 4)
