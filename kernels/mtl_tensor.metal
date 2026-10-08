// Kernels over a bound `MTLTensor` argument (`mtl_tensor::bind_mtl_tensor`).
//
// Every other kernel builds its tensors from raw device pointers; these take
// the tensor itself, bound by resource ID, which is the only way to reach a
// device-owned tensor (`mtl_tensor::alloc_device_tensor`): Metal picks its
// layout and it has no host pointer. They exist so that binding path is
// exercised by a dispatch that writes and then reads one.
//
// Indices follow `MTLTensorExtents` order, innermost first: `t[c, r]` is
// column `c` of row `r` of a `[cols, rows]` tensor.
#include <metal_stdlib>
#include <metal_tensor>
using namespace metal;

/// `t[c, r] = (r * cols + c) % 127 - 63`: every int8 value a test can tell
/// apart from zeroed or stale memory.
kernel void mtl_tensor_fill_i8(
    tensor<device int8_t, dextents<int32_t, 2>> t [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int cols = t.get_extent(0);
    const int rows = t.get_extent(1);
    const int c = (int)gid.x;
    const int r = (int)gid.y;
    if (c >= cols || r >= rows) { return; }
    t[c, r] = (int8_t)((r * cols + c) % 127 - 63);
}

/// `out[r * cols + c] = t[c, r]`, widened to f32.
kernel void mtl_tensor_read_i8(
    tensor<device int8_t, dextents<int32_t, 2>> t [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const int cols = t.get_extent(0);
    const int rows = t.get_extent(1);
    const int c = (int)gid.x;
    const int r = (int)gid.y;
    if (c >= cols || r >= rows) { return; }
    out[(ulong)r * (ulong)cols + (ulong)c] = (float)t[c, r];
}
