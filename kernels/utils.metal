// Shared util kernels extracted from metal-native (copy/cast/zero/softcap/transpose).
#include <metal_stdlib>
#include "softcap.h"
using namespace metal;

// The elementwise copy and cast kernels are one template body each, stamped
// out per element type. The stamps own the entry name and the
// `[[buffer(N)]]` slots the host binds (0 = in, 1 = out, 2 = n); a macro rather
// than a `template [[host_name]]` instantiation because the kernel census
// (`xcrun metal -E`, `kernel void NAME`) names each entry by its declaration.
// `tests/q4_kernels_template.rs` pins every one of them bit for bit.
template <typename T>
inline void util_copy(device const T *in, device T *out, uint n, uint gid)
{
    if (gid >= n) return;
    out[gid] = in[gid];
}

template <typename S, typename D>
inline void util_cast(device const S *in, device D *out, uint n, uint gid)
{
    if (gid >= n) return;
    out[gid] = D(in[gid]);
}

#define TESSL_COPY_KERNEL(NAME, T)                                  \
    kernel void NAME(                                               \
        device const T *in [[buffer(0)]],                           \
        device T *out [[buffer(1)]],                                \
        constant uint &n [[buffer(2)]],                             \
        uint gid [[thread_position_in_grid]])                       \
    {                                                               \
        util_copy<T>(in, out, n, gid);                              \
    }

#define TESSL_CAST_KERNEL(NAME, S, D)                               \
    kernel void NAME(                                               \
        device const S *in [[buffer(0)]],                           \
        device D *out [[buffer(1)]],                                \
        constant uint &n [[buffer(2)]],                             \
        uint gid [[thread_position_in_grid]])                       \
    {                                                               \
        util_cast<S, D>(in, out, n, gid);                           \
    }

TESSL_COPY_KERNEL(copy_f32, float)
// Bitwise: indices and other integer data, which a float copy could alter
// (small integers are f32 denormals).
TESSL_COPY_KERNEL(copy_u32, uint)
TESSL_COPY_KERNEL(copy_bf16, bfloat)
TESSL_COPY_KERNEL(copy_f16, half)

TESSL_CAST_KERNEL(cast_f32_to_bf16, float, bfloat)
TESSL_CAST_KERNEL(cast_bf16_to_f32, bfloat, float)

kernel void zero_f32(
    device float *x [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    x[gid] = 0.0f;
}

/// out[i] = ((S[0][i] + S[1][i]) + S[2][i]) + ... over `partitions` slices
/// of `n` floats, `slice` floats apart (`slice >= n`), in slice order, so
/// the sum does not depend on how the slices were produced or scheduled.
kernel void reduce_partitions_f32(
    device const float *S [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &partitions [[buffer(3)]],
    constant uint &slice [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n || partitions == 0u || slice < n) return;
    float s = S[gid];
    for (uint p = 1; p < partitions; p++) {
        s += S[(ulong)p * slice + gid];
    }
    out[gid] = s;
}

/// `reduce_partitions_f32` adding into `out`:
/// out[i] = ((out[i] + S[0][i]) + S[1][i]) + ..., in slice order, so a
/// `C +=` split into partitions adds them to C in the order the sequential
/// split-K's dispatches do.
kernel void reduce_partitions_add_f32(
    device const float *S [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &partitions [[buffer(3)]],
    constant uint &slice [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n || slice < n) return;
    float s = out[gid];
    for (uint p = 0; p < partitions; p++) {
        s += S[(ulong)p * slice + gid];
    }
    out[gid] = s;
}

kernel void add_inplace_f32(
    device float *dst [[buffer(0)]],
    device const float *src [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dst[gid] += src[gid];
}

kernel void transpose2d_f32(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((ulong)gid >= (ulong)rows * cols) return;
    uint i = gid / cols;
    uint j = gid % cols;
    out[(ulong)j * rows + i] = in[(ulong)i * cols + j];
}

/// logits_post = softcap * tanh(logits / softcap).
kernel void softcap_f32(
    device const float *pre [[buffer(0)]],
    device float *post [[buffer(1)]],
    constant float &softcap [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    post[gid] = tessl_apply_softcap(pre[gid], softcap);
}

/// f32 -> IEEE half.
///
/// Unlike bf16, half has only 5 exponent bits, so values above 65504 do not
/// merely lose precision — they become infinity. That is a real difference for
/// activations. The two casts differ in range, not in rounding: both are
/// round-to-nearest-even conversions, and `cast_f32_to_bf16` matches the host
/// `f32_to_bf16_bits` bit for bit (tests/bf16_cast.rs).
TESSL_CAST_KERNEL(cast_f32_to_f16, float, half)
TESSL_CAST_KERNEL(cast_f16_to_f32, half, float)
