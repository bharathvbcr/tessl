// MLX affine Q4 lane dot: the sixteen-nibble `qmv_fast` peel shared by every
// simdgroup-cooperative MLX Q4 GEMV and GEMM kernel.
//
// Canonical owner. `gemv_q4_mlx.metal` and `gemm_q4_mlx.metal` are separate
// translation units and each used to carry its own copy of these two
// functions; two copies of the innermost loop of the decode hot path is a
// defect waiting for one of them to be edited. `build.rs` compiles only
// `*.metal`, so this header is included, never compiled on its own, and
// `track_kernel_sources` emits `rerun-if-changed` for `.h` as well.
#pragma once

#include <metal_stdlib>
using namespace metal;

/// MLX load_vector bits=4: store x with 16^k prescale + return sum(x).
///
/// `x` must be 8-byte aligned: every caller advances it by whole 16-element
/// lane blocks from a buffer base, so the `bfloat4` loads never straddle.
inline float load_x16_qdot(device const bfloat *x, thread float *xp)
{
    bfloat4 x0 = ((device const bfloat4 *)(x))[0];
    bfloat4 x1 = ((device const bfloat4 *)(x + 4u))[0];
    bfloat4 x2 = ((device const bfloat4 *)(x + 8u))[0];
    bfloat4 x3 = ((device const bfloat4 *)(x + 12u))[0];
    float a0 = float(x0.x), a1 = float(x0.y), a2 = float(x0.z), a3 = float(x0.w);
    float a4 = float(x1.x), a5 = float(x1.y), a6 = float(x1.z), a7 = float(x1.w);
    float a8 = float(x2.x), a9 = float(x2.y), a10 = float(x2.z), a11 = float(x2.w);
    float a12 = float(x3.x), a13 = float(x3.y), a14 = float(x3.z), a15 = float(x3.w);
    float sum = (a0 + a1 + a2 + a3) + (a4 + a5 + a6 + a7)
              + (a8 + a9 + a10 + a11) + (a12 + a13 + a14 + a15);
    // values_per_thread chunk of 4: /1, /16, /256, /4096
    xp[0] = a0;             xp[1] = a1 / 16.0f;   xp[2] = a2 / 256.0f;  xp[3] = a3 / 4096.0f;
    xp[4] = a4;             xp[5] = a5 / 16.0f;   xp[6] = a6 / 256.0f;  xp[7] = a7 / 4096.0f;
    xp[8] = a8;             xp[9] = a9 / 16.0f;   xp[10] = a10 / 256.0f; xp[11] = a11 / 4096.0f;
    xp[12] = a12;           xp[13] = a13 / 16.0f; xp[14] = a14 / 256.0f; xp[15] = a15 / 4096.0f;
    return sum;
}

/// Sixteen nibbles of `w` masked against the prescaled x: `scale * accum +
/// sum(x) * bias` (MLX qdot bits=4).
///
/// One 8-byte load. `w` is 8-byte aligned at every call site: the row-major
/// kernels step it by `row_bytes = cols / 2` with `cols % 32 == 0` (the host
/// refuses other group sizes), and the Interleaved4 kernels index 8-byte
/// packs inside 32-byte tiles. The masks pick the nibble in place — `0x00f0`
/// keeps the value times 16, which is why `xp` carries the 16^-k prescale.
inline float qdot16(
    device const uchar *w,
    thread const float *xp,
    float scale,
    float bias,
    float xsum)
{
    const uint2 w2 = ((device const uint2 *)w)[0];
    const uint ww0 = w2.x & 0xffffu;
    const uint ww1 = w2.x >> 16;
    const uint ww2 = w2.y & 0xffffu;
    const uint ww3 = w2.y >> 16;
    float accum = 0.0f;
    accum += xp[0] * float(ww0 & 0x000fu) + xp[1] * float(ww0 & 0x00f0u)
           + xp[2] * float(ww0 & 0x0f00u) + xp[3] * float(ww0 & 0xf000u);
    accum += xp[4] * float(ww1 & 0x000fu) + xp[5] * float(ww1 & 0x00f0u)
           + xp[6] * float(ww1 & 0x0f00u) + xp[7] * float(ww1 & 0xf000u);
    accum += xp[8] * float(ww2 & 0x000fu) + xp[9] * float(ww2 & 0x00f0u)
           + xp[10] * float(ww2 & 0x0f00u) + xp[11] * float(ww2 & 0xf000u);
    accum += xp[12] * float(ww3 & 0x000fu) + xp[13] * float(ww3 & 0x00f0u)
           + xp[14] * float(ww3 & 0x0f00u) + xp[15] * float(ww3 & 0xf000u);
    return scale * accum + xsum * bias;
}
