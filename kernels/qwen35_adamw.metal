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

/// The sum of squares of each row of a window (`rows` x `width` at `ld`,
/// `off`), written to `out[out_off + row]`: the pieces of a global gradient
/// norm, which the host adds in f64. One threadgroup of 256 per row, lanes
/// striding the row, so the order of the adds is fixed.
kernel void qwen35_sq_sum_rows_f32(
    device const float *g [[buffer(0)]],
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
    // 8 simdgroups at Apple's 32 lanes; room for 32 at narrower widths.
    threadgroup float part[32];
    const ulong base = (ulong)r * ld + off;
    float s = 0.0f;
    for (uint c = t; c < width; c += 256) {
        const float x = g[base + c];
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
