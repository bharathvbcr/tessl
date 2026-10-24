// Row-wise reductions: softmax, sum, max.
//
// One threadgroup per row, so a row's reduction never crosses a dispatch
// boundary. The alternative — a global two-pass reduction — costs an extra
// full read and write of the input for shapes where a row already fits one
// group's strided scan.
//
// Every kernel here strides: lane `lid` visits `cols[lid], cols[lid + tptg],
// ...`, so `cols` is unbounded and adjacent lanes read adjacent addresses.
// Reductions are simdgroup-first (`reduce_row_add` / `reduce_row_max` in
// `reduce_tree.h`): lanes fold with simd shuffles and the simdgroup partials
// meet across one barrier.
#include <metal_stdlib>
#include "reduce_tree.h"
using namespace metal;

/// The bit pattern of -INFINITY, tested as bits because fast math may assume
/// no value is infinite and fold a float compare against it away.
constant uint NEG_INF_BITS = 0xFF800000u;

/// The starting value of lane `lid`'s running max over a row of `cols >= 1`:
/// the lane's first element, or the row's first for a lane with none. Max is
/// idempotent, so a duplicate changes nothing; an identity would have to be
/// `-INFINITY`, which fast math may assume never occurs, or `-FLT_MAX`, which
/// would report a row of only `-inf` as `-FLT_MAX`.
inline float max_seed(device const float* xr, uint lid, uint cols) {
    return xr[lid < cols ? lid : 0u];
}

/// `out[r, :] = softmax(x[r, :])`, numerically stable.
///
/// Subtracts the row maximum before exponentiating. Without that, a row
/// containing a logit above ~88 overflows `exp` in f32 and the whole row
/// becomes NaN — which is not a rare input for attention scores or logits, and
/// is why "stable softmax" is the only kind worth having.
///
/// Two passes over the row (max, then sum of exp) plus a third to write. The
/// row is re-read from device memory each pass rather than staged, so `cols`
/// has no ceiling. A register-resident single-pass variant for rows that fit
/// `8 * tptg` was measured on 2026-09-05 and never beat this form: the
/// re-reads are cache hits, and the kernel's cost is the reduction and the
/// threadgroup shape, not row traffic.
///
/// A row of all -INFINITY sums to zero. Dividing would give NaN; the
/// convention here is a uniform distribution, which is what a caller
/// masking every position of an attention row expects to see. That is the
/// only row the uniform fallback is for: a row holding a NaN or a +INFINITY
/// (torch: NaN) has a NaN denominator and every output is NaN. A -INFINITY
/// term is therefore recognised by its bits and contributes exactly 0, so the
/// all--inf row's denominator is 0 rather than the NaN of `-inf - -inf`, and
/// a NaN denominator can no longer be mistaken for it (it was: `!(denom > 0)`
/// took both, silently turning poisoned rows into valid-looking uniform ones).
/// The NaN case is never tested for (an `isnan` is what fast math may fold
/// away): a NaN denominator fails `denom == 0` and flows through the ordinary
/// formula to NaN outputs.
kernel void softmax_rows_f32(
    device const float* x    [[buffer(0)]],
    device float*       out  [[buffer(1)]],
    constant uint&      cols [[buffer(2)]],
    uint  row  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  tptg [[threads_per_threadgroup]]
) {
    threadgroup float scratch[2u * REDUCE_MAX_SIMDGROUPS];
    device const float* xr = x + (ulong)row * cols;
    device float* outr = out + (ulong)row * cols;

    float m = max_seed(xr, lid, cols);
    for (ulong c = lid; c < (ulong)cols; c += tptg) { m = fmax(m, xr[c]); }
    const float row_max = reduce_row_max(m, scratch, sgid, lane, tptg);

    float s = 0.0f;
    for (ulong c = lid; c < (ulong)cols; c += tptg) {
        const float v = xr[c];
        s += as_type<uint>(v) == NEG_INF_BITS ? 0.0f : exp(v - row_max);
    }
    const float denom = reduce_row_add(s, scratch + REDUCE_MAX_SIMDGROUPS, sgid, lane, tptg);
    // Zero only for a row of all -inf (the row's maximum contributes exp(0) = 1 to any other),
    // never for a NaN, which compares false here and takes the ordinary path to NaN outputs.
    const bool degenerate = denom == 0.0f;
    const float inv = degenerate ? 1.0f / (float)cols : 1.0f / denom;

    for (ulong c = lid; c < (ulong)cols; c += tptg) {
        outr[c] = degenerate ? inv : exp(xr[c] - row_max) * inv;
    }
}

/// `out[r] = sum(x[r, :])`.
kernel void row_sum_f32(
    device const float* x    [[buffer(0)]],
    device float*       out  [[buffer(1)]],
    constant uint&      cols [[buffer(2)]],
    uint  row  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  tptg [[threads_per_threadgroup]]
) {
    threadgroup float scratch[REDUCE_MAX_SIMDGROUPS];
    device const float* xr = x + (ulong)row * cols;
    float s = 0.0f;
    for (ulong c = lid; c < (ulong)cols; c += tptg) { s += xr[c]; }
    const float total = reduce_row_add(s, scratch, sgid, lane, tptg);
    if (lid == 0u) { out[row] = total; }
}

/// `out[r] = max(x[r, :])`. The host refuses `cols == 0`, so every row has an
/// element to seed from (`max_seed`).
kernel void row_max_f32(
    device const float* x    [[buffer(0)]],
    device float*       out  [[buffer(1)]],
    constant uint&      cols [[buffer(2)]],
    uint  row  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  tptg [[threads_per_threadgroup]]
) {
    threadgroup float scratch[REDUCE_MAX_SIMDGROUPS];
    device const float* xr = x + (ulong)row * cols;
    float m = max_seed(xr, lid, cols);
    for (ulong c = lid; c < (ulong)cols; c += tptg) { m = fmax(m, xr[c]); }
    const float best = reduce_row_max(m, scratch, sgid, lane, tptg);
    if (lid == 0u) { out[row] = best; }
}
