// Row-segment means and an L2 normalize of a row prefix.
//
// `segment_mean_rows_f32`: x [rows, D] (row stride D), segments [S][2] u32
// (start, end) row ranges -> out [S, D], out[s] = mean(x[start:end]). The host
// checks 0 <= start < end <= rows; the kernel clamps end to rows and gives
// an empty range zeros, since the buffer is device-writable. Each thread owns
// one (segment, column) and adds the column in row order, so the sum is
// sequential and does not depend on the launch shape. Two callers, one
// kernel: a sentence embedding's masked mean over each sequence's live tokens
// (sentence-transformers' `Pooling(mean)`, whose 1e-9 count clamp never binds
// for a non-empty range), and a decision head's mean over each option's own
// tokens.
//
// `l2_normalize_rows_f32`: in place over the first `dim` columns of each of
// `rows` rows spaced `ld` apart, `x / max(||x||, 1e-12)` as
// `torch.nn.functional.normalize`. Normalizing a prefix (`dim < ld`) is the
// Matryoshka "truncate, then renormalize" in one pass; `dim == ld` is the
// plain `Normalize` module.
#include <metal_stdlib>
#include "reduce_tree.h"
using namespace metal;

kernel void segment_mean_rows_f32(
    device const float *x [[buffer(0)]],
    device const uint *segments [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &S [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &D [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const ulong n = (ulong)S * D;
    if ((ulong)gid >= n) return;
    const uint s = gid / D;
    const uint d = gid % D;
    const uint end = min(segments[2u * s + 1u], rows);
    const uint start = min(segments[2u * s], end);
    float acc = 0.0f;
    for (uint r = start; r < end; ++r) {
        acc += x[(ulong)r * D + d];
    }
    out[(ulong)s * D + d] = (end > start) ? acc / (float)(end - start) : 0.0f;
}

kernel void l2_normalize_rows_f32(
    device float *x [[buffer(0)]],
    constant uint &rows [[buffer(1)]],
    constant uint &dim [[buffer(2)]],
    constant uint &ld [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    if (row >= rows) return;
    threadgroup float scratch[REDUCE_MAX_SIMDGROUPS];
    device float *r = x + (ulong)row * ld;
    float ss = 0.0f;
    for (uint d = lid; d < dim; d += tptg) {
        const float v = r[d];
        ss += v * v;
    }
    const float total = reduce_row_add(ss, scratch, sgid, lane, tptg);
    const float inv = 1.0f / max(sqrt(total), 1e-12f);
    for (uint d = lid; d < dim; d += tptg) {
        r[d] *= inv;
    }
}
