// AdamW for a Qwen3.5 model's parameters, in place, in their own layouts.
//
// One thread per element of a window (`rows` x `width` at `ld`, `off`), which
// the parameter, its gradient and both moments share: the moments mirror the
// parameters' tensors, and the gradients already come in the weights'
// layouts. Elementwise, so a step is deterministic.
//
// The arithmetic is torch.optim.AdamW's single-tensor path
// (`qwen35_adamw_math.h`, shared with the stored-precision family in
// `qwen35_train_storage.metal`) on the gradient times `grad_scale` (what
// clip_grad_norm_ leaves in `.grad`). The host forms the per-step scalars in
// f64 as torch does and passes them as f32.
//
// Every parameter is stored as transformers holds it (the zero-centred norms
// as `w`, their kernels adding the 1), so the update runs on the stored value
// and weight decay pulls `w` toward zero.
#include <metal_stdlib>
#include "qwen35_adamw_math.h"
using namespace metal;

/// Grid: x = column in [0, width), y = row.
kernel void qwen35_adamw_f32(
    device float *p [[buffer(0)]],
    device const float *g [[buffer(1)]],
    device float *m [[buffer(2)]],
    device float *v [[buffer(3)]],
    constant Qwen35AdamW &a [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &width [[buffer(6)]],
    constant uint &ld [[buffer(7)]],
    constant uint &off [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    const uint r = gid.y;
    if (c >= width || r >= rows) return;
    const ulong i = (ulong)r * ld + off + c;

    float mi = m[i];
    float vi = v[i];
    const float w = qwen35_adamw_update(a, p[i], g[i], mi, vi);

    p[i] = w;
    m[i] = mi;
    v[i] = vi;
}

/// One gradient window of a `qwen35_sq_sum_pieces` table: its GPU address,
/// its layout (`rows` x `width` at `ld`, `off`), whether it is bf16 (else
/// f32), and its pieces: `pieces_per_row` of `QWEN35_SQ_PIECE` columns per
/// row (the last one short), the first numbered `first_piece`.
struct Qwen35SqSumSlot {
    ulong g;
    uint rows;
    uint width;
    uint ld;
    uint off;
    uint pieces_per_row;
    uint first_piece;
    uint bf16;
    uint pad;
};
static_assert(sizeof(Qwen35SqSumSlot) == 40, "the host writes 40-byte slots");

/// Elements of one piece: 256 lanes, 16 each.
#define QWEN35_SQ_PIECE 4096u

/// The sum of squares of `n` (at most a piece) elements from `g + base`,
/// lanes striding by 256, each simdgroup's lanes summed, then the
/// simdgroups in order; returned to thread 0. Every element is in f32
/// (bf16 widened exactly) and every add is of non-negative values, so the
/// piece is within `(16 + 31 + 8)` roundings of its exact sum.
template <typename T>
inline float qwen35_sq_sum_piece(device const T *g, ulong base, uint n, threadgroup float *part, uint t, uint lane,
                                 uint sg, uint n_sg)
{
    float s = 0.0f;
    for (uint c = t; c < n; c += 256u) {
        const float x = float(g[base + c]);
        s = fma(x, x, s);
    }
    s = simd_sum(s);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    if (t == 0) {
        for (uint i = 0; i < n_sg; ++i) total += part[i];
    }
    return total;
}

/// The sum of squares of every piece of every window of `slots` (sorted by
/// `first_piece`, the first's 0) in one dispatch, written to `out[piece]`:
/// the parts of a global gradient norm, which `qwen35_sum_exact_f32` adds.
/// One threadgroup of 256 per piece finds its window by binary search. A
/// piece holds at most 4096 elements, so a dense window (one row of a whole
/// tensor) is spread over the GPU rather than summed by one threadgroup.
kernel void qwen35_sq_sum_pieces(
    constant Qwen35SqSumSlot *slots [[buffer(0)]],
    constant uint &n_slots [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint gp [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint n_sg [[simdgroups_per_threadgroup]])
{
    // 8 simdgroups at Apple's 32 lanes; room for 32 at narrower widths.
    threadgroup float part[32];
    uint lo = 0u, hi = n_slots - 1u;
    while (lo < hi) {
        const uint mid = (lo + hi + 1u) >> 1;
        if (slots[mid].first_piece <= gp) {
            lo = mid;
        } else {
            hi = mid - 1u;
        }
    }
    constant Qwen35SqSumSlot &s = slots[lo];
    const uint local = gp - s.first_piece;
    const uint row = local / s.pieces_per_row;
    const uint c0 = (local - row * s.pieces_per_row) * QWEN35_SQ_PIECE;
    const uint n = min(QWEN35_SQ_PIECE, s.width - c0);
    const ulong base = (ulong)row * s.ld + s.off + c0;
    // The window's dtype is the threadgroup's, so the barrier inside is
    // reached by every thread either way.
    const float total = s.bf16 != 0u
        ? qwen35_sq_sum_piece(reinterpret_cast<device const bfloat *>(s.g), base, n, part, t, lane, sg, n_sg)
        : qwen35_sq_sum_piece(reinterpret_cast<device const float *>(s.g), base, n, part, t, lane, sg, n_sg);
    if (t == 0) out[gp] = total;
}

/// `x` (non-negative and finite) in the 128-bit fixed point whose unit is
/// `2^(etop - 226)`: its 24-bit significand shifted so that a value of
/// biased exponent `etop` has its leading bit at bit 99. Bits below the
/// unit are dropped (a value 2^-76 of the largest or smaller).
inline ulong2 qwen35_fixed128(uint bits, uint etop)
{
    uint e = bits >> 23;
    uint m = bits & 0x7fffffu;
    if (e == 0u) {
        e = 1u;
    } else {
        m |= 0x800000u;
    }
    const int sh = int(e) - int(etop) + 76;
    if (sh < 0) return ulong2(sh > -24 ? ulong(m >> uint(-sh)) : 0ul, 0ul);
    if (sh == 0) return ulong2(ulong(m), 0ul);
    if (sh >= 64) return ulong2(0ul, ulong(m) << uint(sh - 64));
    return ulong2(ulong(m) << uint(sh), ulong(m) >> uint(64 - sh));
}

/// The exact sum of `n` non-negative f32 values (the pieces of
/// `qwen35_sq_sum_pieces`), one threadgroup of 256. Integer arithmetic, so
/// the order of the adds (and fast math) cannot change it: each value is
/// put in a 128-bit fixed point set by the largest (`qwen35_fixed128`), the
/// sum is exact (n < 2^27 leaves the top bit clear), and the host rounds it
/// to f64 once. `out` is the sum's four 32-bit words, least significant
/// first, then the largest value's bits; a largest value that is not finite
/// (an infinity, or any NaN) is reported in that last word alone.
kernel void qwen35_sum_exact_f32(
    device const float *x [[buffer(0)]],
    device uint *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint t [[thread_index_in_threadgroup]])
{
    threadgroup uint top_s[256];
    threadgroup ulong lo_s[256];
    threadgroup ulong hi_s[256];
    // The largest value: for non-negative floats, the largest bit pattern
    // (any NaN, either sign, compares above +inf).
    uint m = 0u;
    for (uint i = t; i < n; i += 256u) m = max(m, as_type<uint>(x[i]));
    top_s[t] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = 128u; w > 0u; w >>= 1) {
        if (t < w) top_s[t] = max(top_s[t], top_s[t + w]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint top = top_s[0];
    if (top >= 0x7f800000u) {
        if (t == 0) {
            out[0] = 0u;
            out[1] = 0u;
            out[2] = 0u;
            out[3] = 0u;
            out[4] = top;
        }
        return;
    }
    const uint etop = max(top >> 23, 1u);
    ulong lo = 0ul, hi = 0ul;
    for (uint i = t; i < n; i += 256u) {
        const ulong2 q = qwen35_fixed128(as_type<uint>(x[i]), etop);
        lo += q.x;
        hi += q.y + (lo < q.x ? 1ul : 0ul);
    }
    lo_s[t] = lo;
    hi_s[t] = hi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = 128u; w > 0u; w >>= 1) {
        if (t < w) {
            const ulong a = lo_s[t];
            const ulong l = a + lo_s[t + w];
            hi_s[t] = hi_s[t] + hi_s[t + w] + (l < a ? 1ul : 0ul);
            lo_s[t] = l;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) {
        out[0] = uint(lo_s[0]);
        out[1] = uint(lo_s[0] >> 32);
        out[2] = uint(hi_s[0]);
        out[3] = uint(hi_s[0] >> 32);
        out[4] = top;
    }
}
