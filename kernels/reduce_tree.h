// Shared threadgroup reduction primitives.
//
// Canonical owner of the per-row reduction. `reduce.metal` and `rms_norm.metal`
// both reduce a row inside one threadgroup, and each .metal file is a separate
// translation unit, so without a header the second one to need this copies
// it. Two copies of a reduction that must agree on lane assignment and
// barrier placement is a defect waiting for one of them to be edited.
//
// `build.rs` compiles only `*.metal`, so this file is included, never compiled
// on its own — and `track_kernel_sources` emits `rerun-if-changed` for `.h` as
// well, without which editing this header would leave every dependent kernel
// silently stale in the metallib while the tests reported a pass.
#pragma once

#include <metal_stdlib>
using namespace metal;

/// Largest threadgroup a row reduction is launched with; the host caps lanes
/// per row well below it (see `reduce_tptg` in src/nn.rs).
constant uint REDUCE_MAX_TG = 1024u;

/// Simdgroups a launch can hold: `REDUCE_MAX_TG` lanes over 32-wide simdgroups.
constant uint REDUCE_MAX_SIMDGROUPS = REDUCE_MAX_TG / 32u;

/// Sum `v` across the threadgroup, returned in every lane.
///
/// Simdgroup-first, two levels: `simd_sum` folds the 32 lanes with no memory
/// traffic, one lane per simdgroup publishes its partial, one barrier, and
/// then every simdgroup folds the partials with a second `simd_sum` (lane
/// `l` takes partial `l`), so the total is uniform in every lane with no
/// serial loop and no second barrier. A launch that fits one simdgroup never
/// touches `scratch` or a barrier.
///
/// The tree this replaced (`REDUCE_TREE`, retired 2026-09-05) halved the whole threadgroup
/// through threadgroup memory with a barrier per round — ten rounds and
/// eleven barriers at 1024 lanes. A first cut folded the partials with a
/// serial loop per thread instead of the second shuffle, and lost to the
/// tree from 32 simdgroups up: 32 dependent threadgroup-memory loads per
/// thread cost more than ten barriers. Measured 2026-09-05, see
/// `bench/results/row_reduction_simd_m5pro.txt`.
///
/// `scratch` must hold `REDUCE_MAX_SIMDGROUPS` floats. The reads of `scratch`
/// happen after the barrier here, so a second reduction must not write the
/// same region until another barrier separates them: give each reduction its
/// own region (`scratch + REDUCE_MAX_SIMDGROUPS`) rather than reusing one.
inline float reduce_row_add(
    float v,
    threadgroup float *scratch,
    uint sgid,
    uint lane,
    uint tptg)
{
    v = simd_sum(v);
    const uint n_sg = (tptg + 31u) / 32u;
    if (n_sg == 1u) return v;
    if (lane == 0u) scratch[sgid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float partial = lane < n_sg ? scratch[lane] : 0.0f;
    return simd_sum(partial);
}

/// Maximum of `v` across the threadgroup, returned in every lane. Same
/// structure and scratch contract as [`reduce_row_add`].
inline float reduce_row_max(
    float v,
    threadgroup float *scratch,
    uint sgid,
    uint lane,
    uint tptg)
{
    v = simd_max(v);
    const uint n_sg = (tptg + 31u) / 32u;
    if (n_sg == 1u) return v;
    if (lane == 0u) scratch[sgid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float partial = lane < n_sg ? scratch[lane] : -INFINITY;
    return simd_max(partial);
}
