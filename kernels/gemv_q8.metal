// Q8 GEMV decode (M=1) with inline group-wise affine dequant.
// W layout: row-major [rows, cols] as signed int8 bytes.
// scales/zeros: [rows * (cols / group_size)], group along K (cols).
//
// One simdgroup per Q8_SIMD_ROWS output rows, lanes striding K. This kernel was
// one *thread* per row until 2026-08-31, which was wrong twice over: the
// parallelism available was `rows`, and — worse for a kernel whose whole cost is
// streaming weights — adjacent threads read addresses `cols` bytes apart, so a
// simdgroup's 32 loads touched 32 different cache lines and nothing coalesced.
// Measured on an M5 Pro at 11008x4096 it reached 132 GB/s where the machine
// sustains ~240.
//
// Striding K instead puts lane `l` on column `c0 + l`, so the 32 lanes of a
// simdgroup read 32 contiguous bytes of one row, and the four rows a simdgroup
// owns share the same `x` values out of cache.
#include <metal_stdlib>
using namespace metal;

/// Lanes per simdgroup. Fixed by the hardware, named for the arithmetic below.
constant uint Q8_SIMD_SIZE = 32u;
/// Output rows one simdgroup accumulates concurrently. Four `x` reads amortise
/// across four rows; more would spill the accumulator array.
constant uint Q8_SIMD_ROWS = 4u;
/// Simdgroups per threadgroup.
constant uint Q8_SG_PER_TG = 2u;
// Q8_SIMD_ROWS * Q8_SG_PER_TG = 8 rows per threadgroup and
// Q8_SIMD_SIZE * Q8_SG_PER_TG = 64 threads. The host must agree: those are
// SIMD_ROWS_PER_TG and SIMD_TPTG in src/nn.rs, and a disagreement would leave
// the tail rows of a dispatch unwritten rather than fail. `gemv_q8` with a row
// count that is not a multiple of 8 covers that.

/// y[rows] = W[rows, cols] @ x[cols] with Q8 weights.
kernel void gemv_q8(
    device const char *packed [[buffer(0)]],
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
    const uint row0 = (tgid * Q8_SG_PER_TG + sgid) * Q8_SIMD_ROWS;
    if (row0 >= rows) return;
    const uint groups_per_row = cols / group_size;
    // char4 loads need the row stride, the group base and the group width all
    // divisible by 4. `cols` is a multiple of `group_size` (the host refuses
    // otherwise), so all three follow from group_size alone.
    const bool vec4_ok = (group_size % 4u == 0u) && (cols % 4u == 0u);

    float acc[Q8_SIMD_ROWS];
    for (uint r = 0u; r < Q8_SIMD_ROWS; ++r) acc[r] = 0.0f;

    // Group-major: scale and zero are constant across a group, so hoisting them
    // out of the inner loop turns two device loads per element into two per
    // group per row. Iterating columns first would reload them every element.
    for (uint g = 0u; g < groups_per_row; ++g) {
        float s[Q8_SIMD_ROWS];
        float z[Q8_SIMD_ROWS];
        for (uint r = 0u; r < Q8_SIMD_ROWS; ++r) {
            const uint row = row0 + r;
            if (row < rows) {
                const uint gi = row * groups_per_row + g;
                s[r] = scales[gi];
                z[r] = zeros[gi];
            } else {
                s[r] = 0.0f;
                z[r] = 0.0f;
            }
        }
        const uint c0 = g * group_size;
        if (vec4_ok) {
            // Four bytes per lane, so a simdgroup's 32 loads cover 128
            // contiguous bytes — one cache line per instruction instead of a
            // quarter of one. `vec4_ok` guarantees the char4 access is aligned
            // and that a lane's four columns lie inside a single group, which
            // is what lets `s[r]`/`z[r]` stay hoisted.
            for (uint i = lane * 4u; i < group_size; i += Q8_SIMD_SIZE * 4u) {
                const uint c = c0 + i;
                const float4 xv = float4(x[c], x[c + 1u], x[c + 2u], x[c + 3u]);
                for (uint r = 0u; r < Q8_SIMD_ROWS; ++r) {
                    const uint row = row0 + r;
                    if (row >= rows) break;
                    device const char4 *wp =
                        (device const char4 *)(packed + (ulong)row * cols + c);
                    const char4 q = *wp;
                    const float4 w =
                        s[r] * (float4((float)q.x, (float)q.y, (float)q.z, (float)q.w) - z[r]);
                    const float4 prod = w * xv;
                    acc[r] += (prod.x + prod.y) + (prod.z + prod.w);
                }
            }
        } else {
            // Lanes stride the group one element at a time. When group_size <
            // Q8_SIMD_SIZE the upper lanes idle for that group; correctness
            // holds and the shapes this kernel is for use 32 or more.
            for (uint i = lane; i < group_size; i += Q8_SIMD_SIZE) {
                const float xv = x[c0 + i];
                for (uint r = 0u; r < Q8_SIMD_ROWS; ++r) {
                    const uint row = row0 + r;
                    if (row >= rows) break;
                    const float w = s[r] * ((float)packed[(ulong)row * cols + c0 + i] - z[r]);
                    acc[r] += w * xv;
                }
            }
        }
    }

    for (uint r = 0u; r < Q8_SIMD_ROWS; ++r) {
        const float sum = simd_sum(acc[r]);
        const uint row = row0 + r;
        if (lane == 0u && row < rows) y[row] = sum;
    }
}
