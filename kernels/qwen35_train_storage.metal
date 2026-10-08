// Narrow storage for Qwen3.5 training: bf16 weights and gradient banks,
// bf16 or block-quantised 8-bit AdamW moments, and the update rules that keep
// small AdamW steps from vanishing into a bf16 parameter.
//
// Arithmetic is always f32: a narrow value is widened on load and rounded
// once on store. AdamW's per-element math is `qwen35_adamw_math.h`, the same
// code the f32 kernel runs.
//
// # AdamW state layout
//
// The parameter and its gradient are a window of the weight's own tensor
// (`rows` x `width` at leading dimension `ld` from element `off`: a packed
// projection's columns, or a whole dense tensor). Everything the optimizer
// owns (moments, f32 master, Kahan compensation, block scales) is dense in
// the window's row-major order: element `i` is window row `i / width`, column
// `i % width`. That is the order `Qwen35Model::read_parameters` writes a
// parameter in, so a moment reads out as a plain copy.
//
// # Update rules
//
// - plain (f32 parameters only): `p = update(p)`, the f32 kernel's bits.
// - master: an f32 master copy is the parameter; the bf16 weight is the
//   master rounded to nearest after every step.
// - kahan: bf16 parameter and bf16 compensation `c`, as torchao/optimi's
//   Kahan-summed AdamW: `c += delta; p' = bf16(p + c); c -= p' - p`.
// - stochastic: `p' = round(p + delta)` up or down with probability equal to
//   the distance, from 16 hashed bits of (seed, step, tensor salt, element),
//   so a run is reproducible and resumes bit for bit.
//
// # 8-bit moments
//
// Blocks of 256 consecutive state elements share one f32 scale (one
// threadgroup per block, which reduces the block's new maximum before
// encoding). Both codes are companded so small values keep resolution:
// m is signed, `q = rint(127 sign(m) sqrt(|m| / s))`, `m = s (q/127)|q/127|`
// with `s = max |m|`; v is stored through `u = sqrt(v)`,
// `q = rint(255 sqrt(u / s))`, `u = s (q/255)^2` with `s = max u`, and a
// non-zero v never encodes to 0 (it would divide m by eps alone). Encoding a
// decoded block gives the same codes and scale back.
#include <metal_stdlib>
#include "qwen35_adamw_math.h"
using namespace metal;

constant uint QWEN35_Q8_BLOCK = 256;

#define QWEN35_RULE_PLAIN 0u
#define QWEN35_RULE_MASTER 1u
#define QWEN35_RULE_KAHAN 2u
#define QWEN35_RULE_SR 3u
#define QWEN35_MOM_F32 0u
#define QWEN35_MOM_BF16 1u
#define QWEN35_MOM_Q8 2u

/// The stochastic-rounding key: the run's 64-bit seed, the 64-bit step
/// count, and a per-tensor salt (its parameter-table index).
struct Qwen35SrKey {
    uint seed_lo;
    uint seed_hi;
    uint step_lo;
    uint step_hi;
    uint salt;
};

/// A 32-bit integer hash (lowbias32): every input bit reaches every output bit.
static inline uint qwen35_mix(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static inline uint qwen35_sr_bits(constant Qwen35SrKey &k, uint i)
{
    uint h = qwen35_mix(k.seed_lo ^ qwen35_mix(k.seed_hi));
    h = qwen35_mix(h ^ k.step_lo);
    h = qwen35_mix(h ^ k.step_hi);
    h = qwen35_mix(h ^ k.salt);
    return qwen35_mix(h ^ i);
}

/// `x` rounded to bf16 up or down in magnitude, up with probability equal to
/// the dropped fraction. The test is on the exponent bits, not isinf/isnan,
/// which fast math may fold: infinities and NaNs round to nearest.
static inline bfloat qwen35_round_stochastic(float x, uint r)
{
    const uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        return bfloat(x);
    }
    return as_type<bfloat>(ushort((bits + (r & 0xffffu)) >> 16));
}

static inline float qwen35_q8m_decode(char q, float s)
{
    const float t = float(q) / 127.0f;
    return s * t * fabs(t);
}

static inline char qwen35_q8m_encode(float x, float s)
{
    if (!(s > 0.0f)) {
        return 0;
    }
    const float r = rint(precise::sqrt(min(precise::divide(fabs(x), s), 1.0f)) * 127.0f);
    return char(x < 0.0f ? -r : r);
}

/// `sqrt(v)` from its code.
static inline float qwen35_q8v_decode_sqrt(uchar q, float s)
{
    const float t = float(q) / 255.0f;
    return s * t * t;
}

/// The code of `u = sqrt(v)`.
static inline uchar qwen35_q8v_encode(float u, float s)
{
    if (!(s > 0.0f)) {
        return 0;
    }
    float r = rint(precise::sqrt(min(precise::divide(u, s), 1.0f)) * 255.0f);
    if (u > 0.0f && r < 1.0f) {
        r = 1.0f;
    }
    return uchar(r);
}

/// The maximum of `x` over a threadgroup of `n_sg` simdgroups; every thread
/// gets it. Two barriers: `part` is reusable on return.
static inline float qwen35_tg_max(float x, threadgroup float *part, uint t, uint lane, uint sg, uint n_sg)
{
    x = simd_max(x);
    if (lane == 0) {
        part[sg] = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = part[0];
    for (uint s = 1; s < n_sg; ++s) {
        total = max(total, part[s]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

template <uint MOM> struct Qwen35MomTypes;
template <> struct Qwen35MomTypes<QWEN35_MOM_F32> {
    typedef float M;
    typedef float V;
};
template <> struct Qwen35MomTypes<QWEN35_MOM_BF16> {
    typedef bfloat M;
    typedef bfloat V;
};
template <> struct Qwen35MomTypes<QWEN35_MOM_Q8> {
    typedef char M;
    typedef uchar V;
};

/// One AdamW step over `n` state elements; threadgroups of 256 (one 8-bit
/// block each). `aux` is the f32 master (master) or the bf16 compensation
/// (kahan), unused otherwise; the scales are read only by 8-bit moments.
template <typename P, typename G, typename AUX, uint RULE, uint MOM>
static inline void qwen35_adamw_stored_body(
    device P *p,
    device const G *g,
    device typename Qwen35MomTypes<MOM>::M *m,
    device typename Qwen35MomTypes<MOM>::V *v,
    constant Qwen35AdamW &a,
    uint n,
    uint width,
    uint ld,
    uint off,
    device AUX *aux,
    device float *m_scale,
    device float *v_scale,
    constant Qwen35SrKey &key,
    threadgroup float *part,
    uint i,
    uint t,
    uint blk,
    uint lane,
    uint sg,
    uint n_sg)
{
    const bool live = i < n;
    const uint r = live ? i / width : 0u;
    const ulong wi = (ulong)r * ld + off + (live ? i - r * width : 0u);

    float x = 0.0f, gr = 0.0f, mi = 0.0f, vi = 0.0f;
    if (live) {
        if (RULE == QWEN35_RULE_MASTER) {
            x = float(aux[i]);
        } else {
            x = float(p[wi]);
        }
        gr = float(g[wi]);
        if (MOM == QWEN35_MOM_Q8) {
            mi = qwen35_q8m_decode(char(m[i]), m_scale[blk]);
            const float u = qwen35_q8v_decode_sqrt(uchar(v[i]), v_scale[blk]);
            vi = u * u;
        } else {
            mi = float(m[i]);
            vi = float(v[i]);
        }
    }

    if (RULE == QWEN35_RULE_PLAIN || RULE == QWEN35_RULE_MASTER) {
        const float w = qwen35_adamw_update(a, x, gr, mi, vi);
        if (live) {
            if (RULE == QWEN35_RULE_MASTER) {
                aux[i] = AUX(w);
            }
            p[wi] = P(w);
        }
    } else {
        const float d = qwen35_adamw_delta(a, x, gr, mi, vi);
        if (live) {
            if (RULE == QWEN35_RULE_KAHAN) {
                const float c = float(aux[i]) + d;
                const P pn = P(x + c);
                aux[i] = AUX(c - (float(pn) - x));
                p[wi] = pn;
            } else {
                p[wi] = P(qwen35_round_stochastic(x + d, qwen35_sr_bits(key, i)));
            }
        }
    }

    if (MOM == QWEN35_MOM_Q8) {
        const float u = precise::sqrt(vi);
        const float sm = qwen35_tg_max(live ? fabs(mi) : 0.0f, part, t, lane, sg, n_sg);
        const float sv = qwen35_tg_max(live ? u : 0.0f, part, t, lane, sg, n_sg);
        if (live) {
            m[i] = qwen35_q8m_encode(mi, sm);
            v[i] = qwen35_q8v_encode(u, sv);
        }
        if (t == 0) {
            m_scale[blk] = sm;
            v_scale[blk] = sv;
        }
    } else if (live) {
        m[i] = typename Qwen35MomTypes<MOM>::M(mi);
        v[i] = typename Qwen35MomTypes<MOM>::V(vi);
    }
}

#define QWEN35_ADAMW_STORED(NAME, P, G, AUX, RULE, MOM)                                              \
    kernel void NAME(                                                                                \
        device P *p [[buffer(0)]],                                                                   \
        device const G *g [[buffer(1)]],                                                             \
        device Qwen35MomTypes<MOM>::M *m [[buffer(2)]],                                              \
        device Qwen35MomTypes<MOM>::V *v [[buffer(3)]],                                              \
        constant Qwen35AdamW &a [[buffer(4)]],                                                       \
        constant uint &n [[buffer(5)]],                                                              \
        constant uint &width [[buffer(6)]],                                                          \
        constant uint &ld [[buffer(7)]],                                                             \
        constant uint &off [[buffer(8)]],                                                            \
        device AUX *aux [[buffer(9)]],                                                               \
        device float *m_scale [[buffer(10)]],                                                        \
        device float *v_scale [[buffer(11)]],                                                        \
        constant Qwen35SrKey &key [[buffer(12)]],                                                    \
        uint i [[thread_position_in_grid]],                                                          \
        uint t [[thread_index_in_threadgroup]],                                                      \
        uint blk [[threadgroup_position_in_grid]],                                                   \
        uint lane [[thread_index_in_simdgroup]],                                                     \
        uint sg [[simdgroup_index_in_threadgroup]],                                                  \
        uint n_sg [[simdgroups_per_threadgroup]])                                                    \
    {                                                                                                \
        threadgroup float part[32];                                                                  \
        qwen35_adamw_stored_body<P, G, AUX, RULE, MOM>(                                              \
            p, g, m, v, a, n, width, ld, off, aux, m_scale, v_scale, key, part, i, t, blk, lane, sg, n_sg); \
    }

// f32 parameters (their gradients are f32): moments in any storage.
QWEN35_ADAMW_STORED(qwen35_adamw_f32_plain_gf32_mf32, float, float, float, QWEN35_RULE_PLAIN, QWEN35_MOM_F32)
QWEN35_ADAMW_STORED(qwen35_adamw_f32_plain_gf32_mbf16, float, float, float, QWEN35_RULE_PLAIN, QWEN35_MOM_BF16)
QWEN35_ADAMW_STORED(qwen35_adamw_f32_plain_gf32_mq8, float, float, float, QWEN35_RULE_PLAIN, QWEN35_MOM_Q8)

// bf16 parameters: each rule, a bf16 bank's or fresh f32 gradients, each
// moment storage.
#define QWEN35_ADAMW_BF16_RULE(RNAME, AUX, RULE)                                                         \
    QWEN35_ADAMW_STORED(qwen35_adamw_bf16_##RNAME##_gf32_mf32, bfloat, float, AUX, RULE, QWEN35_MOM_F32)    \
    QWEN35_ADAMW_STORED(qwen35_adamw_bf16_##RNAME##_gf32_mbf16, bfloat, float, AUX, RULE, QWEN35_MOM_BF16)  \
    QWEN35_ADAMW_STORED(qwen35_adamw_bf16_##RNAME##_gf32_mq8, bfloat, float, AUX, RULE, QWEN35_MOM_Q8)      \
    QWEN35_ADAMW_STORED(qwen35_adamw_bf16_##RNAME##_gbf16_mf32, bfloat, bfloat, AUX, RULE, QWEN35_MOM_F32)  \
    QWEN35_ADAMW_STORED(qwen35_adamw_bf16_##RNAME##_gbf16_mbf16, bfloat, bfloat, AUX, RULE, QWEN35_MOM_BF16) \
    QWEN35_ADAMW_STORED(qwen35_adamw_bf16_##RNAME##_gbf16_mq8, bfloat, bfloat, AUX, RULE, QWEN35_MOM_Q8)

QWEN35_ADAMW_BF16_RULE(master, float, QWEN35_RULE_MASTER)
QWEN35_ADAMW_BF16_RULE(kahan, bfloat, QWEN35_RULE_KAHAN)
QWEN35_ADAMW_BF16_RULE(sr, float, QWEN35_RULE_SR)

/// `dst[i] (+)= src[i]` into bf16 from f32, one rounding per element: a
/// gradient delivered into a bf16 bank. `add` reads the bank's value widened.
kernel void qwen35_deliver_f32_to_bf16(
    device const float *src [[buffer(0)]],
    device bfloat *dst [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &add [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    dst[i] = bfloat(add != 0u ? float(dst[i]) + src[i] : src[i]);
}

/// A `rows` x `width` window copied between two windows (each a leading
/// dimension and a starting element), converting the element type: f32 to
/// bf16 rounds to nearest, bf16 to f32 is exact.
#define QWEN35_WINDOW_CAST(NAME, S, D)                                                               \
    kernel void NAME(                                                                                \
        device const S *src [[buffer(0)]],                                                           \
        device D *dst [[buffer(1)]],                                                                 \
        constant uint &rows [[buffer(2)]],                                                           \
        constant uint &width [[buffer(3)]],                                                          \
        constant uint &src_ld [[buffer(4)]],                                                         \
        constant uint &src_off [[buffer(5)]],                                                        \
        constant uint &dst_ld [[buffer(6)]],                                                         \
        constant uint &dst_off [[buffer(7)]],                                                        \
        uint2 gid [[thread_position_in_grid]])                                                       \
    {                                                                                                \
        if (gid.x >= width || gid.y >= rows) return;                                                 \
        dst[(ulong)gid.y * dst_ld + dst_off + gid.x] = D(float(src[(ulong)gid.y * src_ld + src_off + gid.x])); \
    }

QWEN35_WINDOW_CAST(qwen35_window_copy_f32_f32, float, float)
QWEN35_WINDOW_CAST(qwen35_window_copy_bf16_f32, bfloat, float)
QWEN35_WINDOW_CAST(qwen35_window_copy_f32_bf16, float, bfloat)

/// `qwen35_sq_sum_rows_f32` over a bf16 window (a bf16 bank's gradient),
/// each element widened: the same lanes, order and adds.
kernel void qwen35_sq_sum_rows_bf16(
    device const bfloat *g [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &width [[buffer(2)]],
    constant uint &ld [[buffer(3)]],
    constant uint &off [[buffer(4)]],
    constant uint &out_off [[buffer(5)]],
    uint r [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint n_sg [[simdgroups_per_threadgroup]])
{
    threadgroup float part[32];
    const ulong base = (ulong)r * ld + off;
    float s = 0.0f;
    for (uint c = t; c < width; c += 256) {
        const float x = float(g[base + c]);
        s = fma(x, x, s);
    }
    s = simd_sum(s);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float total = 0.0f;
        for (uint i = 0; i < n_sg; ++i) total += part[i];
        out[out_off + r] = total;
    }
}

/// An 8-bit moment written from, or read into, dense f32 values (a
/// checkpoint): `first` picks the m code, else the v code (whose f32 value
/// is v itself, not its square root). One threadgroup of 256 per block.
kernel void qwen35_moment_q8_encode(
    device const float *x [[buffer(0)]],
    device uchar *q [[buffer(1)]],
    device float *scale [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant uint &first [[buffer(4)]],
    uint i [[thread_position_in_grid]],
    uint t [[thread_index_in_threadgroup]],
    uint blk [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint n_sg [[simdgroups_per_threadgroup]])
{
    threadgroup float part[32];
    const bool live = i < n;
    const float xi = live ? x[i] : 0.0f;
    const float val = first != 0u ? fabs(xi) : precise::sqrt(max(xi, 0.0f));
    const float s = qwen35_tg_max(live ? val : 0.0f, part, t, lane, sg, n_sg);
    if (live) {
        q[i] = first != 0u ? as_type<uchar>(qwen35_q8m_encode(xi, s)) : qwen35_q8v_encode(val, s);
    }
    if (t == 0) {
        scale[blk] = s;
    }
}

kernel void qwen35_moment_q8_decode(
    device const uchar *q [[buffer(0)]],
    device const float *scale [[buffer(1)]],
    device float *x [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant uint &first [[buffer(4)]],
    uint i [[thread_position_in_grid]],
    uint blk [[threadgroup_position_in_grid]])
{
    if (i >= n) return;
    if (first != 0u) {
        x[i] = qwen35_q8m_decode(as_type<char>(q[i]), scale[blk]);
    } else {
        const float u = qwen35_q8v_decode_sqrt(q[i], scale[blk]);
        x[i] = u * u;
    }
}
