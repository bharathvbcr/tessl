#ifndef TESSL_SOFTCAP_H
#define TESSL_SOFTCAP_H

#include <metal_stdlib>
using namespace metal;

/// Canonical softcap used by both the tensor utility and decode/sampling
/// kernels. Keeping the device-buffer and by-value entry points on one helper
/// prevents their saturation and invalid-cap behavior from drifting.
static inline float tessl_apply_softcap(float value, float cap) {
    // A device-backed cap can change after the host encoded the command. Fail
    // safe if it is corrupted: preserving a finite logit is preferable to
    // manufacturing zeros or NaNs. Host-by-value callers reject this earlier.
    if (!(cap > 0.0f) || !isfinite(cap)) {
        return value;
    }

    // Metal's tanh is implemented through an exponential that can overflow on
    // large finite arguments. In f32 tanh has rounded to +/-1 well before 16,
    // so the clamp changes no representable saturated result.
    const float z = clamp(value / cap, -16.0f, 16.0f);
    return cap * tanh(z);
}

#endif
