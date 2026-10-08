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
/// masking every position of an attention row expects to see.
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

    float m = -INFINITY;
    for (ulong c = lid; c < (ulong)cols; c += tptg) { m = fmax(m, xr[c]); }
    const float row_max = reduce_row_max(m, scratch, sgid, lane, tptg);

    float s = 0.0f;
    for (ulong c = lid; c < (ulong)cols; c += tptg) { s += exp(xr[c] - row_max); }
    const float denom = reduce_row_add(s, scratch + REDUCE_MAX_SIMDGROUPS, sgid, lane, tptg);
    const float inv = denom > 0.0f ? 1.0f / denom : 1.0f / (float)cols;
    const bool degenerate = !(denom > 0.0f);

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

/// `out[r] = max(x[r, :])`. An empty row yields -INFINITY, the identity.
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
    float m = -INFINITY;
    for (ulong c = lid; c < (ulong)cols; c += tptg) { m = fmax(m, xr[c]); }
    const float best = reduce_row_max(m, scratch, sgid, lane, tptg);
    if (lid == 0u) { out[row] = best; }
}
