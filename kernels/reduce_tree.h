// Shared threadgroup reduction primitives.
//
// Canonical owner of the tree reduction. `reduce.metal` and `rms_norm.metal`
// both perform a per-row reduction inside one threadgroup, and each .metal file
// is a separate translation unit, so without a header the second one to need
// this copies it. Two copies of a reduction that must agree on power-of-two
// width and barrier placement is a defect waiting for one of them to be
// edited.
//
// `build.rs` compiles only `*.metal`, so this file is included, never compiled
// on its own — and `track_kernel_sources` emits `rerun-if-changed` for `.h` as
// well, without which editing this header would leave every dependent kernel
// silently stale in the metallib while the tests reported a pass.
#pragma once

#include <metal_stdlib>
using namespace metal;

/// Threadgroup scratch depth. Every lane writes its own slot before the tree
/// reduction, so the launch must never exceed this.
constant uint REDUCE_MAX_TG = 1024u;

/// Tree-reduce `scratch[0..tptg]` with `combine`, leaving the result in slot 0.
///
/// `tptg` must be a power of two: the loop halves it, and an odd width would
/// drop the top element silently rather than failing. The host rounds down to
/// a power of two before dispatch.
#define REDUCE_TREE(scratch, tptg, lid, combine)                               \
    for (uint stride = (tptg) >> 1; stride > 0u; stride >>= 1) {               \
        if ((lid) < stride) {                                                  \
            (scratch)[lid] = combine((scratch)[lid], (scratch)[(lid) + stride]);\
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    }

inline float reduce_add(float a, float b) { return a + b; }
inline float reduce_max(float a, float b) { return fmax(a, b); }
