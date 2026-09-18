// GPU-resident KV: store one timestep and optional ring densify (no host roundtrip).
#include <metal_stdlib>
using namespace metal;

/// dst[dst_offset + i] = src[i] for i in [0, n).
/// `dst_offset` is a stable device u32 (ICB / encode-once — not const-arena).
/// `dst_capacity` is the fixed logical element capacity supplied by the host;
/// an invalid live offset makes the complete store a no-op.
kernel void kv_store_timestep(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    device const uint *dst_offset_ptr [[buffer(3)]],
    constant uint &dst_capacity [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const ulong dst_offset = (ulong)*dst_offset_ptr;
    const ulong capacity = (ulong)dst_capacity;
    const ulong count = (ulong)n;
    if (gid >= n) return;
    // Subtraction form avoids wrapping `dst_offset + n` back into bounds.
    if (dst_offset > capacity || count > capacity - dst_offset) return;
    dst[dst_offset + (ulong)gid] = src[gid];
}

/// Store K and V timesteps in one dispatch (producer hot path).
/// `dst_offset` from stable device u32 (ICB freeze).
/// Both destinations share the fixed logical `dst_capacity` contract.
kernel void kv_store_timestep_pair(
    device const float *src_k [[buffer(0)]],
    device const float *src_v [[buffer(1)]],
    device float *dst_k [[buffer(2)]],
    device float *dst_v [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    device const uint *dst_offset_ptr [[buffer(5)]],
    constant uint &dst_capacity [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    const ulong dst_offset = (ulong)*dst_offset_ptr;
    const ulong capacity = (ulong)dst_capacity;
    const ulong count = (ulong)n;
    if (gid >= n) return;
    if (dst_offset > capacity || count > capacity - dst_offset) return;
    const ulong index = dst_offset + (ulong)gid;
    dst_k[index] = src_k[gid];
    dst_v[index] = src_v[gid];
}

/// Chronological densify from a ring: dst[t] = src[(start+t) % capacity].
/// `filled` / `start` are stable device u32s (per-step varying; ICB freeze).
kernel void kv_ring_densify(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant uint &n_slot [[buffer(2)]],
    constant uint &capacity [[buffer(3)]],
    device const uint *filled_ptr [[buffer(4)]],
    device const uint *start_ptr [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    // Both metadata values are live device state. Clamp `filled` to the fixed
    // ring capacity and widen every multiply/add before it can wrap. A start
    // cursor outside the ring is normalized instead of indexing arbitrary
    // memory; the host already rejects capacity == 0.
    const ulong live = (ulong)min(*filled_ptr, capacity);
    const ulong slot_width = (ulong)n_slot;
    const ulong total = live * slot_width;
    const ulong gid64 = (ulong)gid;
    if (gid64 >= total) return;
    const ulong t = gid64 / slot_width;
    const ulong e = gid64 % slot_width;
    const ulong src_t = ((ulong)*start_ptr + t) % (ulong)capacity;
    dst[gid64] = src[src_t * slot_width + e];
}
