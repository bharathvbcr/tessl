// AdamW's per-element arithmetic, shared by `qwen35_adamw_f32` (dense f32
// tensors) and the stored-precision family in `qwen35_train_storage.metal`, so
// the f32 update has one owner whatever the parameter, moments and
// gradient are stored as.
//
// torch.optim.AdamW's single-tensor path (amsgrad and maximize off), in its
// order, on the gradient times `grad_scale`: decoupled weight decay, the first
// moment by torch's `lerp` (whose form switches at weight 0.5), the second
// moment, `sqrt(v) / sqrt(bc2) + eps`, then `p += -step_size * (m / denom)`.
#pragma once
#include <metal_stdlib>

/// Per-step, per-tensor scalars, formed on the host in f64 and passed as f32.
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
    /// Multiplies the gradient before anything reads it: `clip_grad_norm_`'s
    /// clip coefficient (torch scales `.grad` in place, in f32), or 1.
    float grad_scale;
};

/// The moments of one element from its raw gradient `g` (updated in place,
/// f32), and the step direction `m / denom`.
static inline float qwen35_adamw_moments(constant Qwen35AdamW &a, float g, thread float &m, thread float &v)
{
    const float gi = g * a.grad_scale;
    const float diff = gi - m;
    m = a.lerp_w < 0.5f ? m + a.lerp_w * diff : gi - diff * (1.0f - a.lerp_w);
    v = v * a.beta2 + (a.one_minus_beta2 * gi) * gi;
    const float denom = metal::precise::divide(metal::precise::sqrt(v), a.bc2_sqrt) + a.eps;
    return metal::precise::divide(m, denom);
}

/// One element: `w` is the parameter's value, `g` its raw gradient, `m` and
/// `v` the moments as last stored. Updates `m` and `v` and returns the new
/// parameter value, all in f32.
static inline float qwen35_adamw_update(constant Qwen35AdamW &a, float w, float g, thread float &m, thread float &v)
{
    const float decayed = w * a.decay_mul;
    return decayed + (-a.step_size) * qwen35_adamw_moments(a, g, m, v);
}

/// [`qwen35_adamw_update`] as the change it makes to `w`, formed without
/// first rounding the new value: what a compensated or stochastically rounded
/// narrow parameter adds.
static inline float qwen35_adamw_delta(constant Qwen35AdamW &a, float w, float g, thread float &m, thread float &v)
{
    return w * (a.decay_mul - 1.0f) + (-a.step_size) * qwen35_adamw_moments(a, g, m, v);
}
