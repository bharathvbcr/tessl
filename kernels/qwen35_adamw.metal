// AdamW for a Qwen3.5 model's parameters, in place, in their own layouts.
//
// One thread per element of a window (`rows` x `width` at `ld`, `off`), which
// the parameter, its gradient and both moments share: the moments mirror the
// parameters' tensors, and the gradients already come in the weights'
// layouts. Elementwise, so a step is deterministic.
//
// The arithmetic is torch.optim.AdamW's single-tensor path (amsgrad and
// maximize off), in its order: decoupled weight decay, the first moment by
// torch's `lerp` (whose form switches at weight 0.5), the second moment,
// `sqrt(v) / sqrt(bc2) + eps`, then `p += -step_size * (m / denom)`. The host
// forms the per-step scalars in f64 as torch does and passes them as f32.
//
// A norm tessl stores as `1 + w` has `shift = 1`: the update runs on
// `w = p - 1` (exact for the stored range) and stores `1 + w`, so weight
// decay pulls `w`, not `1 + w`, toward zero.
#include <metal_stdlib>
using namespace metal;

/// Per-step, per-tensor scalars.
struct Qwen35AdamW {
    /// `1 - lr * weight_decay`.
    float decay_mul;
    /// `1 - beta1`: the lerp weight.
    float lerp_w;
    float beta2;
    /// `1 - beta2`.
    float one_minus_beta2;
    /// `lr / (1 - beta1^step)`.
    float step_size;
    /// `sqrt(1 - beta2^step)`.
    float bc2_sqrt;
    float eps;
    /// 1 for a parameter stored as `1 + w`, else 0.
    float shift;
};

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

    float w = p[i] - a.shift;
    w = w * a.decay_mul;

    const float gi = g[i];
    float mi = m[i];
    const float diff = gi - mi;
    mi = a.lerp_w < 0.5f ? mi + a.lerp_w * diff : gi - diff * (1.0f - a.lerp_w);
    const float vi = v[i] * a.beta2 + (a.one_minus_beta2 * gi) * gi;

    const float denom = precise::divide(precise::sqrt(vi), a.bc2_sqrt) + a.eps;
    w = w + (-a.step_size) * precise::divide(mi, denom);

    p[i] = a.shift + w;
    m[i] = mi;
    v[i] = vi;
}
