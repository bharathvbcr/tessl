// Row-wise reductions: softmax, sum, max.
//
// One threadgroup per row, so a row's reduction is a tree inside threadgroup
// memory and never crosses a dispatch boundary. The alternative — a global
// two-pass reduction — costs an extra full read and write of the input for
// shapes where a row already fits one group's strided scan.
//
// Every kernel here strides: lane `lid` visits `cols[lid], cols[lid + tptg],
// ...`, so `cols` is unbounded and adjacent lanes read adjacent addresses.
#include <metal_stdlib>
using namespace metal;

/// Threadgroup scratch depth. Every lane writes its own slot before the tree
/// reduction, so the launch must never exceed this.
constant uint REDUCE_MAX_TG = 1024u;

/// Tree-reduce `scratch[0..tptg]` with `op`, leaving the result in slot 0.
///
/// `tptg` must be a power of two: the loop halves it, and an odd width would
/// drop the top element silently rather than failing. The host rounds down to
/// a power of two before dispatch.
#define REDUCE_TREE(scratch, tptg, lid, combine)                               \
    for (uint stride = (tptg) >> 1; stride > 0u; stride >>= 1) {               \
        if ((lid) < stride) {                                                  \
            (scratch)[lid] = combine((scratch)[lid], (scratch)[(lid) + stride]);\
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    }

inline float reduce_add(float a, float b) { return a + b; }
inline float reduce_max(float a, float b) { return fmax(a, b); }

/// `out[r, :] = softmax(x[r, :])`, numerically stable.
///
/// Subtracts the row maximum before exponentiating. Without that, a row
/// containing a logit above ~88 overflows `exp` in f32 and the whole row
/// becomes NaN — which is not a rare input for attention scores or logits, and
/// is why "stable softmax" is the only kind worth having.
///
/// Two passes over the row (max, then sum of exp) plus a third to write. The
/// row is re-read from device memory each pass rather than staged in
/// threadgroup memory, so `cols` has no ceiling.
kernel void softmax_rows_f32(
    device const float* x    [[buffer(0)]],
    device float*       out  [[buffer(1)]],
    constant uint&      cols [[buffer(2)]],
    uint  row  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  tptg [[threads_per_threadgroup]]
) {
    threadgroup float scratch[REDUCE_MAX_TG];
    device const float* xr = x + (ulong)row * cols;
    device float* outr = out + (ulong)row * cols;

    float m = -INFINITY;
    for (uint c = lid; c < cols; c += tptg) { m = fmax(m, xr[c]); }
    scratch[lid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    REDUCE_TREE(scratch, tptg, lid, reduce_max)
    const float row_max = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float s = 0.0f;
    for (uint c = lid; c < cols; c += tptg) { s += exp(xr[c] - row_max); }
    scratch[lid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    REDUCE_TREE(scratch, tptg, lid, reduce_add)
    // A row of all -INFINITY sums to zero. Dividing would give NaN; the
    // convention here is a uniform distribution, which is what a caller
    // masking every position of an attention row expects to see.
    const float denom = scratch[0];
    const float inv = denom > 0.0f ? 1.0f / denom : 1.0f / (float)cols;
    const bool degenerate = !(denom > 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = lid; c < cols; c += tptg) {
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
    uint  tptg [[threads_per_threadgroup]]
) {
    threadgroup float scratch[REDUCE_MAX_TG];
    device const float* xr = x + (ulong)row * cols;
    float s = 0.0f;
    for (uint c = lid; c < cols; c += tptg) { s += xr[c]; }
    scratch[lid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    REDUCE_TREE(scratch, tptg, lid, reduce_add)
    if (lid == 0u) { out[row] = scratch[0]; }
}

/// `out[r] = max(x[r, :])`. An empty row yields -INFINITY, the identity.
kernel void row_max_f32(
    device const float* x    [[buffer(0)]],
    device float*       out  [[buffer(1)]],
    constant uint&      cols [[buffer(2)]],
    uint  row  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  tptg [[threads_per_threadgroup]]
) {
    threadgroup float scratch[REDUCE_MAX_TG];
    device const float* xr = x + (ulong)row * cols;
    float m = -INFINITY;
    for (uint c = lid; c < cols; c += tptg) { m = fmax(m, xr[c]); }
    scratch[lid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    REDUCE_TREE(scratch, tptg, lid, reduce_max)
    if (lid == 0u) { out[row] = scratch[0]; }
}
