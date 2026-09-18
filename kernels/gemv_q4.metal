// Q4 GEMV decode (M=1) with inline group-wise affine dequant.
// W layout: row-major [rows, cols], two int4 nibbles per byte (lo = even index),
// signed nibble (sign-extend). `Q4MlxBank` is a different format.
//
// `gemv_q4`: one simdgroup per GEMV_SIMD_ROWS output rows, lanes striding K
// eight nibbles (one uint) at a time. Lane `l` owns columns `l*8 .. l*8+8`,
// `256 + l*8 .. `, so a simdgroup's 32 loads read 128 contiguous bytes of one
// row and the four rows it owns share the same `x` values out of cache.
// Because lanes stride across group boundaries rather than inside one group,
// all 32 lanes are busy at every group size the host admits (a multiple of
// 8), and a lane's eight nibbles never straddle a group; the group's scale
// and zero are fetched per chunk, which the lanes of one group share.
//
// This kernel was one *thread* per row with the whole of `x` staged in
// threadgroup memory until 2026-09-05: adjacent threads read addresses
// `cols / 2` bytes apart, so nothing in a simdgroup's loads coalesced, and
// the 16 KiB `x` cache at `cols = 4096` halved the threadgroup budget for an
// array whose device re-read is a rounding error against the weight stream.
// The same geometry fix on `gemv_q8` (2026-08-31) took that kernel from 132
// GB/s to the machine's ~240.
//
// `gemv_q4_tiled`: one threadgroup per row with a scalar inner loop, kept as
// the A/B baseline the public `tiled` switch selects.
#include <metal_stdlib>
using namespace metal;

constant uint GEMV_TG = 128u;

/// Lanes per simdgroup, fixed by the hardware and named for the arithmetic.
constant uint GEMV_SIMD_SIZE = 32u;
/// Output rows one simdgroup accumulates concurrently; four `x` reads amortise
/// across four rows without spilling the accumulators.
constant uint GEMV_SIMD_ROWS = 4u;
/// Simdgroups per threadgroup. GEMV_SIMD_ROWS * GEMV_SG_PER_TG = 8 rows and
/// GEMV_SIMD_SIZE * GEMV_SG_PER_TG = 64 threads per threadgroup; the host's
/// SIMD_ROWS_PER_TG / SIMD_TPTG in src/nn.rs must agree.
constant uint GEMV_SG_PER_TG = 2u;
/// Nibbles one lane takes per step: one 32-bit word of packed weights.
constant uint GEMV_NIBBLES_PER_LANE = 8u;
/// Columns a simdgroup covers per step.
constant uint GEMV_SIMD_STEP = GEMV_SIMD_SIZE * GEMV_NIBBLES_PER_LANE; // 256

inline float dequant_q4_nibble(
    device const uchar *packed,
    ulong idx,
    float scale,
    float zero)
{
    uchar byte = packed[idx / 2];
    uchar nibble = (idx & 1ul) == 0ul ? (byte & 0x0fu) : ((byte >> 4) & 0x0fu);
    int q = (int)(nibble << 28) >> 28;
    return scale * ((float)q - zero);
}

/// Sign-extended nibble `k` of `w`, lowest nibble first.
inline float q4_signed_nibble(uint w, uint k)
{
    return (float)((int)(w << (28u - 4u * k)) >> 28);
}

/// Eight signed nibbles of `w` dequantized and dotted with `x[0..8)`, held as
/// two float4s.
inline float q4_word_dot8(uint w, float scale, float zero, float4 xa, float4 xb)
{
    const float4 qa = float4(q4_signed_nibble(w, 0u), q4_signed_nibble(w, 1u),
                             q4_signed_nibble(w, 2u), q4_signed_nibble(w, 3u));
    const float4 qb = float4(q4_signed_nibble(w, 4u), q4_signed_nibble(w, 5u),
                             q4_signed_nibble(w, 6u), q4_signed_nibble(w, 7u));
    const float4 wa = scale * (qa - zero);
    const float4 wb = scale * (qb - zero);
    const float4 pa = wa * xa;
    const float4 pb = wb * xb;
    return ((pa.x + pa.y) + (pa.z + pa.w)) + ((pb.x + pb.y) + (pb.z + pb.w));
}

/// y[rows] = W[rows, cols] @ x[cols] with signed Q4 weights.
///
/// Alignment: the host admits only `group_size % 8 == 0` with
/// `cols % group_size == 0`, so `cols % 8 == 0`; a row's packed bytes
/// (`cols / 2`) and a lane's chunk offset (`c / 2`, `c % 8 == 0`) are then
/// multiples of four bytes and the `uint` weight loads are aligned, and
/// `x + c` is a multiple of 32 bytes for the `float4` loads. Buffers bound
/// here are allocation bases, never byte-offset views.
kernel void gemv_q4(
    device const uchar *packed [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *zeros [[buffer(2)]],
    device const float *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row0 = (tgid * GEMV_SG_PER_TG + sgid) * GEMV_SIMD_ROWS;
    if (row0 >= rows) return;
    const uint groups_per_row = cols / group_size;

    float acc[GEMV_SIMD_ROWS];
    for (uint r = 0u; r < GEMV_SIMD_ROWS; ++r) acc[r] = 0.0f;

    for (uint c = lane * GEMV_NIBBLES_PER_LANE; c < cols; c += GEMV_SIMD_STEP) {
        const uint g = c / group_size;
        const float4 xa = ((device const float4 *)(x + c))[0];
        const float4 xb = ((device const float4 *)(x + c + 4u))[0];
        for (uint r = 0u; r < GEMV_SIMD_ROWS; ++r) {
            const uint row = row0 + r;
            if (row >= rows) break;
            const ulong row_base = (ulong)row * cols;
            const ulong gi = (ulong)row * groups_per_row + g;
            const float scale = scales[gi];
            const float zero = zeros[gi];
            const uint w =
                ((device const uint *)(packed + row_base / 2ul + (ulong)(c >> 1)))[0];
            acc[r] += q4_word_dot8(w, scale, zero, xa, xb);
        }
    }

    for (uint r = 0u; r < GEMV_SIMD_ROWS; ++r) {
        const float sum = simd_sum(acc[r]);
        const uint row = row0 + r;
        if (lane == 0u && row < rows) y[row] = sum;
    }
}

kernel void gemv_q4_tiled(
    device const uchar *packed [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *zeros [[buffer(2)]],
    device const float *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    if (tg >= rows) return;
    threadgroup float partial[GEMV_TG];
    const uint row = tg;
    const uint groups_per_row = cols / group_size;
    float acc = 0.0f;
    for (ulong g = tid; g < (ulong)groups_per_row; g += GEMV_TG) {
        const ulong gi = (ulong)row * groups_per_row + g;
        const float scale = scales[gi];
        const float zero = zeros[gi];
        const ulong base = (ulong)row * cols + g * group_size;
        const ulong xbase = g * group_size;
        for (ulong i = 0ul; i < (ulong)group_size; ++i) {
            acc += dequant_q4_nibble(packed, base + i, scale, zero) * x[xbase + i];
        }
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = GEMV_TG / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        y[row] = partial[0];
    }
}
