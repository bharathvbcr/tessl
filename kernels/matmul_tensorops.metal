// TensorOps GEMM via Metal Performance Primitives `matmul2d` (M5 neural
// accelerators); requires Metal 4 / macOS 26+.
//
// Three kernel families:
//   - Exact f32 (NN/TN/NT, their accumulate forms, split-K): 32x32 tiles on
//     one simdgroup. `mode::multiply` writes into a C the host zeroes first
//     (the zero and the matmul share one binder); the accumulate and split-K
//     kernels use `mode::multiply_accumulate` into C instead.
//   - Cooperative destination (bf16, f16 and relaxed-f32 NN, bf16 TN/NT and
//     their accumulate forms, int8): the accumulator lives in registers for
//     all of K and C is written once, so there is no zero pre-pass; an
//     accumulate or `beta` epilogue loads C once and adds. Tiles 128x64 or
//     64x64 on four simdgroups (host-side selection in src/gemm.rs).
//   - bf16 parallel split-K TN: 64x64 coop on four simdgroups, every K
//     partition in one dispatch into a scratch that utils.metal's
//     reduce_partitions_f32 (or _add_f32, for C +=) sums into C in order.
// Every family walks its tiles by gemm_coop.h's `tile_walk`: column panels
// once B holds 2^23 elements, else `tile_from_linear` (Morton order on
// square power-of-two grids, row-major otherwise).
//
// Note: device pointers must be non-const — `const` poisons MPP type matching.

#include "gemm_coop.h"

// Tile walks (`tile_from_linear`, `tile_from_linear_panel`, `tile_walk`), the
// operand views of one tile and the cooperative and exact-f32 bodies live in
// gemm_coop.h, shared with the tune variants.

// =============================================================================
// f32 exact — execution_simdgroup, SM=SN=32 (golden-safe)
// =============================================================================

#define EXACT_F32_TRANS_A_NN false
#define EXACT_F32_TRANS_B_NN false
#define EXACT_F32_TRANS_A_TN true
#define EXACT_F32_TRANS_B_TN false
#define EXACT_F32_TRANS_A_NT false
#define EXACT_F32_TRANS_B_NT true
#define EXACT_F32_MODE_MULTIPLY false
#define EXACT_F32_MODE_ACCUMULATE true

/// `C (+)= A @ B` exact f32 (`mm_exact_f32`): LAYOUT is NN, TN (A stored
/// `[K, M]`) or NT (B stored `[N, K]`); MODE is MULTIPLY (into a C the host
/// zeroes first) or ACCUMULATE (`multiply_accumulate` into C, no zero). NSG
/// must be 1: the body runs on `execution_simdgroup`.
#define EXACT_F32_KERNEL(NAME, LAYOUT, SM, SN, NSG, MODE)                      \
    kernel void NAME(device float *A [[buffer(0)]],                            \
                     device float *B [[buffer(1)]],                            \
                     device float *C [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &tiles_n [[buffer(6)]],                     \
                     constant uint &tiles_m [[buffer(7)]],                     \
                     uint tgpig [[threadgroup_position_in_grid]]) {            \
        static_assert(NSG == 1, "the exact-f32 body runs on one simdgroup");  \
        mm_exact_f32<EXACT_F32_TRANS_A_##LAYOUT, EXACT_F32_TRANS_B_##LAYOUT,   \
                     EXACT_F32_MODE_##MODE, SM, SN>(                           \
            A, B, C, M, N, K, tiles_n, tiles_m, tgpig);                        \
    }

EXACT_F32_KERNEL(matmul2d_tensorops_f32,          NN, 32, 32, 1, MULTIPLY)
/// C[M,N] = A_stored[K,M]^T @ B[K,N] (TN).
EXACT_F32_KERNEL(matmul2d_tensorops_tn_f32,       TN, 32, 32, 1, MULTIPLY)
/// C[M,N] = A[M,K] @ B_stored[N,K]^T (NT).
EXACT_F32_KERNEL(matmul2d_tensorops_nt_f32,       NT, 32, 32, 1, MULTIPLY)
/// C[M,N] += A[M,K] @ B[K,N] (NN accumulate; no C zero).
EXACT_F32_KERNEL(matmul2d_tensorops_nn_accum_f32, NN, 32, 32, 1, ACCUMULATE)
/// C[M,N] += A_stored[K,M]^T @ B[K,N] (TN accumulate; no C zero).
EXACT_F32_KERNEL(matmul2d_tensorops_tn_accum_f32, TN, 32, 32, 1, ACCUMULATE)
/// C[M,N] += A[M,K] @ B_stored[N,K]^T (NT accumulate; no C zero).
EXACT_F32_KERNEL(matmul2d_tensorops_nt_accum_f32, NT, 32, 32, 1, ACCUMULATE)

/// Split-K TN accumulate for one K-partition.
kernel void matmul2d_tensorops_tn_splitk_f32(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &k0 [[buffer(6)]],
    constant uint &k_tile [[buffer(7)]],
    constant uint &tiles_n [[buffer(8)]],
    constant uint &tiles_m [[buffer(9)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr auto mmul_mode = matmul2d_descriptor::mode::multiply_accumulate;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, true, false, false, mmul_mode);
    matmul2d<desc, execution_simdgroup> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    uint k_len = min(k_tile, K - k0);
    auto mA = tensor(A + k0 * M, dextents<int, 2>{(int)M, (int)k_len}, array<int, 2>{1, (int)M});
    auto mB = tensor(B + k0 * N, dextents<int, 2>{(int)N, (int)k_len}, array<int, 2>{1, (int)N});
    auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});

    auto tA = mA.slice(ty, 0);
    auto tB = mB.slice(tx, 0);
    auto tC = mC.slice(tx, ty);
    op.run(tA, tB, tC);
}

/// Split-K TN with every K-partition in one dispatch, for a C of few tiles
/// over a long K: a weight gradient of a handful of rows, such as
/// `d_pre^T · x` at [12, 768] over 4096 rows (24 tiles). One dispatch over
/// all of K gives such a C only `tiles_n · tiles_m` threadgroups, each
/// walking the whole of K, and the sequential split-K above keeps that count
/// per partition. Here threadgroup `p · tiles + t` computes tile `t` of
/// partition `p` (K from `p · k_tile`) into slice `p` of a zeroed scratch
/// `S` of `partitions` slices `slice` floats apart (M·N rounded up to a
/// multiple of 4, so every slice starts 16-byte aligned, as each partition's
/// start in A and B does with `k_tile` a multiple of 4);
/// `reduce_partitions_f32` then adds the slices into C in partition order.
kernel void matmul2d_tensorops_tn_splitk_par_f32(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *S [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &k_tile [[buffer(6)]],
    constant uint &tiles_n [[buffer(7)]],
    constant uint &tiles_m [[buffer(8)]],
    constant uint &partitions [[buffer(9)]],
    constant uint &slice [[buffer(10)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, true, false, false,
                            matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;

    const uint tiles = tiles_n * tiles_m;
    if (tiles == 0u) return;
    const uint p = tgpig / tiles;
    if (p >= partitions) return;
    uint2 tile = tile_from_linear(tgpig - p * tiles, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    const ulong k0 = (ulong)p * k_tile;
    if (k0 >= K) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    uint k_len = (uint)min((ulong)k_tile, (ulong)K - k0);
    auto mA = tensor(A + k0 * M, dextents<int, 2>{(int)M, (int)k_len}, array<int, 2>{1, (int)M});
    auto mB = tensor(B + k0 * N, dextents<int, 2>{(int)N, (int)k_len}, array<int, 2>{1, (int)N});
    auto mS = tensor(S + (ulong)p * slice, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});

    auto tA = mA.slice(ty, 0);
    auto tB = mB.slice(tx, 0);
    auto tS = mS.slice(tx, ty);
    op.run(tA, tB, tS);
}

/// Split-K NN accumulate for one K-partition: C[M,N] += A[M, k0:k0+len] @
/// B[k0:k0+len, N], with C zeroed by the host first. For a long K the host
/// cuts K into partitions whose slice of B fits in cache. One dispatch over
/// the whole K re-reads all of B for every 32-row tile row, the same B-bound
/// walk as the LM head's NT, and a column-panel walk does not help when each
/// tile's A slab is K long as well.
kernel void matmul2d_tensorops_nn_splitk_f32(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &k0 [[buffer(6)]],
    constant uint &k_tile [[buffer(7)]],
    constant uint &tiles_n [[buffer(8)]],
    constant uint &tiles_m [[buffer(9)]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32;
    constexpr int SN = 32;
    constexpr auto mmul_mode = matmul2d_descriptor::mode::multiply_accumulate;
    constexpr auto desc =
        matmul2d_descriptor(SM, SN, dynamic_length_v<int>, false, false, false, mmul_mode);
    matmul2d<desc, execution_simdgroup> op;

    uint2 tile = tile_from_linear(tgpig, tiles_n, tiles_m);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    // A[M,K] row-major: the partition is columns k0.. of each row (row stride
    // K). B[K,N] row-major: the partition is rows k0.. (contiguous).
    uint k_len = min(k_tile, K - k0);
    auto mA = tensor(A + k0, dextents<int, 2>{(int)k_len, (int)M}, array<int, 2>{1, (int)K});
    auto mB = tensor(B + k0 * N, dextents<int, 2>{(int)N, (int)k_len}, array<int, 2>{1, (int)N});
    auto mC = tensor(C, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});

    auto tA = mA.slice(0, ty);
    auto tB = mB.slice(tx, 0);
    auto tC = mC.slice(tx, ty);
    op.run(tA, tB, tC);
}

// =============================================================================
// bf16 → f32 split-K TN — cooperative 64×64 tiles per K-partition.
// (The NN, TN and NT bf16 kernels are in the cooperative-destination section
// below.)
// =============================================================================

/// The bf16 split-K TN with every K-partition in one dispatch, the bf16 form
/// of `matmul2d_tensorops_tn_splitk_par_f32`: threadgroup `p · tiles + t`
/// computes tile `t` of partition `p` (K from `p · k_tile`) on the
/// cooperative destination (gemm_coop.h's `mm_coop`) and stores it into
/// slice `p` of a scratch `S`, `slice` floats per partition. The store
/// writes every in-bounds element, so `S` needs no zeroing;
/// `reduce_partitions_f32` (or `_add_f32` for `C +=`) then sums the slices
/// in partition order.
#define TN_SPLITK_PAR_KERNEL(NAME, ELEM, SM, SN, NSG, RELAXED)                 \
    kernel void NAME(device ELEM *A [[buffer(0)]],                             \
                     device ELEM *B [[buffer(1)]],                             \
                     device float *S [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &k_tile [[buffer(6)]],                      \
                     constant uint &tiles_n [[buffer(7)]],                     \
                     constant uint &tiles_m [[buffer(8)]],                     \
                     constant uint &partitions [[buffer(9)]],                  \
                     constant uint &slice [[buffer(10)]],                      \
                     uint tgpig [[threadgroup_position_in_grid]]) {            \
        const uint tiles = tiles_n * tiles_m;                                  \
        if (tiles == 0u) return;                                               \
        const uint p = tgpig / tiles;                                          \
        if (p >= partitions) return;                                           \
        const ulong k0 = (ulong)p * k_tile;                                    \
        if (k0 >= K) return;                                                   \
        const uint k_len = (uint)min((ulong)k_tile, (ulong)K - k0);            \
        mm_coop<ELEM, SM, SN, NSG, true, false, RELAXED, false, false,         \
                walk_by_b>(A + k0 * M, B + k0 * N, S + (ulong)p * slice, M, N, \
                           k_len, tiles_n, tiles_m, tgpig - p * tiles,         \
                           nullptr, 1.0f, 0.0f, 0u);                           \
    }

TN_SPLITK_PAR_KERNEL(matmul2d_tensorops_tn_splitk_par_bf16_f32, bfloat, 64, 64, 4, false)

// =============================================================================
// bf16 → f32 and relaxed-f32 NN — cooperative destination tensor
// (2026-08-30 tile tune, bench/results/bf16_tile_tune_m5pro_coop.txt).
// The f32 accumulator lives in registers for the whole K reduction and C is
// written exactly once by cT.store() — this retires both the per-BK device
// round-trips over the C tile (the root cause of the 2× gap vs MPS bf16) and
// the host zero_f32(C) pre-pass (every in-bounds element is overwritten).
// Edge tiles run the same coop path over origin-shifted full-extent slices,
// which bounds-check per the MPPTensorOpsMatMul2d.h cooperative example.
// Geometry is selected host-side (gemm.rs nn_coop_kernel): 128×64 sg4 by
// default, 64×64 sg4 when N ≤ 512 or, for bf16, when M does not fill a
// 128-row tile or the 128×64 grid has under 64 tiles (gemm.rs coop_narrow;
// bench/results/bf16_smallm_coop_reconciled_m5pro.txt). (A 256×64 sg8 arm for large squares was measured and retired;
// bench/results/bf16_tile_tune_FINDINGS.md.) Register cost of the
// accumulator is SM*SN*4/(32*NSG) B/thread.
// =============================================================================

// A call with no explicit template arguments is the production geometry:
// 128×64, sg4, strict, no epilogue (`matmul2d_tensorops_bf16_f32`). The body
// is gemm_coop.h's `mm_coop`, walked by `tile_walk` like every other GEMM
// family (bench/results/gemm_panel_walk_m5pro.txt).
template <typename ElemT, int SM = 128, int SN = 64, int NSG = 4,
          bool RELAXED = false, bool EPILOGUE = false>
inline void mm_nn_coop_f32acc(device ElemT *A, device ElemT *B, device float *C,
                              uint M, uint N, uint K, uint tiles_n,
                              uint tiles_m, uint tgpig,
                              device const float *bias, float alpha, float beta,
                              uint act) {
    mm_coop<ElemT, SM, SN, NSG, false, false, RELAXED, false, EPILOGUE, walk_by_b>(
        A, B, C, M, N, K, tiles_n, tiles_m, tgpig, bias, alpha, beta, act);
}

#define NN_COOP_KERNEL(NAME, ELEM, SM, SN, NSG, RELAXED)                       \
    kernel void NAME(device ELEM *A [[buffer(0)]],                             \
                     device ELEM *B [[buffer(1)]],                             \
                     device float *C [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &tiles_n [[buffer(6)]],                     \
                     constant uint &tiles_m [[buffer(7)]],                     \
                     uint tgpig [[threadgroup_position_in_grid]]) {            \
        mm_nn_coop_f32acc<ELEM, SM, SN, NSG, RELAXED, false>(                   \
            A, B, C, M, N, K, tiles_n, tiles_m, tgpig, nullptr, 1.0f, 0.0f, 0u);\
    }

// Written out, rather than stamped by NN_COOP_KERNEL, so the helper is called
// with no template argument list. The defaults above are this kernel.
kernel void matmul2d_tensorops_bf16_f32(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]]) {
    mm_nn_coop_f32acc(A, B, C, M, N, K, tiles_n, tiles_m, tgpig, nullptr, 1.0f,
                      0.0f, 0u);
}
NN_COOP_KERNEL(matmul2d_tensorops_bf16_f32_64x64_sg4,  bfloat,  64, 64, 4, false)
NN_COOP_KERNEL(matmul2d_tensorops_f32_relaxed,            float, 128, 64, 4, true)
NN_COOP_KERNEL(matmul2d_tensorops_f32_relaxed_64x64_sg4,  float,  64, 64, 4, true)

/// Epilogue variants: `C = act(alpha * A@B + beta * C_prev + bias)`.
///
/// A separate entry point rather than extra parameters on the kernels above.
/// Metal faults on a declared-but-unbound buffer, so widening the existing
/// signatures would force every current caller to bind four operands it does
/// not use — and the tuned geometries above are the crate's most measured
/// code. `EPILOGUE` is a template parameter, so these share one source with
/// them and the plain path compiles to exactly what it did before.
/// Quantized int8 GEMM with the dequantization fused.
///
/// `C[m,n] = (A_i8 @ B_i8) * a_scale * b_scale[n]`
///
/// MPP TensorOps accumulates `int8_t x int8_t` into `int32_t` natively — the
/// header's own diagnostic lists the supported cooperative source types as
/// `uint8_t/int8_t/uint4b_format/int4b_format/float/half/bfloat`. The products
/// are exact in int32 for any K below 2^17 at full int8 range, so the
/// accumulation carries no rounding at all, unlike the f32 paths.
///
/// The dequantization happens in registers between the accumulate and the
/// store, for the same reason the epilogue does: applied afterwards it would be
/// a separate full read and write of C.
///
/// `b_scale` is per output column, which is where a per-channel weight scale
/// lives. It is read through the row-stride-0 broadcast the epilogue uses.
// A call with no explicit template arguments is 128×64 sg4
// (`matmul2d_tensorops_i8_f32`).
template <int SM = 128, int SN = 64, int NSG = 4>
inline void mm_i8_dequant_coop(device int8_t *A, device int8_t *B, device float *C,
                               device const float *b_scale, float a_scale,
                               uint M, uint N, uint K, uint tiles_n,
                               uint tiles_m, uint tgpig) {
    constexpr auto d = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, false, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<NSG>> op;

    uint2 tile = tile_walk<SM>(tgpig, tiles_n, tiles_m, N, K);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    auto tA = gemm_a<false>::edge(A, (int)M, (int)K, ty);
    auto tB = gemm_b<false>::edge(B, (int)N, (int)K, tx);
    auto tC = gemm_c_edge(C, (int)M, (int)N, tx, ty, (int)N);

    auto acc = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(tA)>,
        metal::remove_addrspace_t<decltype(tB)>, int32_t>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc.set(i, 0);
    op.run(tA, tB, acc);

    auto outT = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(tA)>,
        metal::remove_addrspace_t<decltype(tB)>, float>();
    if (b_scale) {
        auto tScale = gemm_c_edge(const_cast<device float *>(b_scale), (int)M, (int)N, tx, ty, 0);
        auto scaleT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA)>,
            metal::remove_addrspace_t<decltype(tB)>, float>();
        scaleT.load(tScale);
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < outT.get_capacity(); ++i)
            if (acc.is_valid_element(i))
                outT[i] = (float)acc[i] * a_scale * scaleT[i];
    } else {
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < outT.get_capacity(); ++i)
            if (acc.is_valid_element(i)) outT[i] = (float)acc[i] * a_scale;
    }
    outT.store(tC);
}

kernel void matmul2d_tensorops_i8_f32(
    device int8_t *A [[buffer(0)]],
    device int8_t *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    device const float *b_scale [[buffer(8)]],
    constant float &a_scale [[buffer(9)]],
    constant uint &has_scale [[buffer(10)]],
    uint tgpig [[threadgroup_position_in_grid]]) {
    mm_i8_dequant_coop(A, B, C, has_scale ? b_scale : nullptr, a_scale, M, N, K,
                       tiles_n, tiles_m, tgpig);
}

/// Strided batched NN GEMM.
///
/// The grid gains a second dimension: `tgpig.y` is the batch index, and each
/// operand is offset by its own stride. Per-operand strides rather than one
/// shared stride because a **zero** stride is the useful case — batched
/// activations against a single shared weight matrix is the common shape, and
/// a zero stride expresses it without materialising `batch` copies of B.
///
/// Everything inside the tile is the existing helper, so batching costs a
/// pointer offset and nothing else. The tuned geometry and the epilogue both
/// apply unchanged.
#define NN_COOP_BATCHED_KERNEL(NAME, ELEM, SM, SN, NSG, RELAXED)               \
    kernel void NAME(device ELEM *A [[buffer(0)]],                             \
                     device ELEM *B [[buffer(1)]],                             \
                     device float *C [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &tiles_n [[buffer(6)]],                     \
                     constant uint &tiles_m [[buffer(7)]],                     \
                     constant uint &stride_a [[buffer(8)]],                    \
                     constant uint &stride_b [[buffer(9)]],                    \
                     constant uint &stride_c [[buffer(10)]],                   \
                     uint2 tgpig [[threadgroup_position_in_grid]]) {           \
        const ulong batch = (ulong)tgpig.y;                                    \
        mm_nn_coop_f32acc<ELEM, SM, SN, NSG, RELAXED, false>(                  \
            A + batch * (ulong)stride_a, B + batch * (ulong)stride_b,          \
            C + batch * (ulong)stride_c, M, N, K, tiles_n, tiles_m, tgpig.x,   \
            nullptr, 1.0f, 0.0f, 0u);                                          \
    }

NN_COOP_BATCHED_KERNEL(matmul2d_tensorops_bf16_f32_batched,   bfloat, 128, 64, 4, false)
NN_COOP_BATCHED_KERNEL(matmul2d_tensorops_f16_f32_batched,      half, 128, 64, 4, false)
NN_COOP_BATCHED_KERNEL(matmul2d_tensorops_f32_relaxed_batched, float, 128, 64, 4, true)
// The 64x64 geometry, which the single-matrix path takes for N <= 512 (and
// bf16 short M).
NN_COOP_BATCHED_KERNEL(matmul2d_tensorops_bf16_f32_batched_64x64_sg4,   bfloat, 64, 64, 4, false)
NN_COOP_BATCHED_KERNEL(matmul2d_tensorops_f16_f32_batched_64x64_sg4,      half, 64, 64, 4, false)
NN_COOP_BATCHED_KERNEL(matmul2d_tensorops_f32_relaxed_batched_64x64_sg4, float, 64, 64, 4, true)

#define NN_COOP_EPI_KERNEL(NAME, ELEM, SM, SN, NSG, RELAXED)                   \
    kernel void NAME(device ELEM *A [[buffer(0)]],                             \
                     device ELEM *B [[buffer(1)]],                             \
                     device float *C [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &tiles_n [[buffer(6)]],                     \
                     constant uint &tiles_m [[buffer(7)]],                     \
                     device const float *bias [[buffer(8)]],                   \
                     constant float &alpha [[buffer(9)]],                      \
                     constant float &beta [[buffer(10)]],                      \
                     constant uint &act [[buffer(11)]],                        \
                     constant uint &has_bias [[buffer(12)]],                   \
                     uint tgpig [[threadgroup_position_in_grid]]) {            \
        mm_nn_coop_f32acc<ELEM, SM, SN, NSG, RELAXED, true>(                   \
            A, B, C, M, N, K, tiles_n, tiles_m, tgpig,                         \
            has_bias ? bias : nullptr, alpha, beta, act);                      \
    }

/// f16 operands with f32 accumulate. Same cooperative machinery as bf16 — the
/// element type is a template parameter — so this is an instantiation, not a
/// second implementation. f16 carries more mantissa than bf16 and far less
/// exponent, which is the trade PyTorch interop asks for.
NN_COOP_KERNEL(matmul2d_tensorops_f16_f32,             half, 128, 64, 4, false)
NN_COOP_KERNEL(matmul2d_tensorops_f16_f32_64x64_sg4,   half,  64, 64, 4, false)
NN_COOP_EPI_KERNEL(matmul2d_tensorops_f16_f32_epi,     half, 128, 64, 4, false)

NN_COOP_EPI_KERNEL(matmul2d_tensorops_bf16_f32_epi,           bfloat, 128, 64, 4, false)
// Same epilogue, 64×64 sg4. The non-epilogue NN kernel already instantiates
// this geometry; this is that instantiation with EPILOGUE=true. Host selects
// it only when M does not fill a 128-row tile.
NN_COOP_EPI_KERNEL(matmul2d_tensorops_bf16_f32_epi_64x64_sg4, bfloat,  64, 64, 4, false)
NN_COOP_EPI_KERNEL(matmul2d_tensorops_f32_relaxed_epi,         float, 128, 64, 4, true)
NN_COOP_EPI_KERNEL(matmul2d_tensorops_f16_f32_epi_64x64_sg4,          half,  64, 64, 4, false)
NN_COOP_EPI_KERNEL(matmul2d_tensorops_f32_relaxed_epi_64x64_sg4,      float,  64, 64, 4, true)

/// TN / NT bf16 GEMMs — cooperative destination tensor (2026-08-30 round 2,
/// bench/results/bf16_tnnt_coop_m5pro.txt): register accumulator, C touched
/// once (plus one load for ACCUM), descriptor transposes per lane, walked by
/// `tile_walk`. The body is gemm_coop.h's `mm_coop`; ACCUM is its load-add-
/// store. Geometry: 128x64 sg4 (plain), 64x64 sg4 (accum).
template <int SM, int SN, int NSG, bool ACCUM>
inline void mm_tn_coop_bf16(device bfloat *A, device bfloat *B, device float *C,
                            uint M, uint N, uint K, uint tiles_n, uint tiles_m,
                            uint tgpig) {
    mm_coop<bfloat, SM, SN, NSG, true, false, false, ACCUM, false, walk_by_b>(
        A, B, C, M, N, K, tiles_n, tiles_m, tgpig, nullptr, 1.0f, 0.0f, 0u);
}

template <int SM, int SN, int NSG, bool ACCUM>
inline void mm_nt_coop_bf16(device bfloat *A, device bfloat *B, device float *C,
                            uint M, uint N, uint K, uint tiles_n, uint tiles_m,
                            uint tgpig) {
    mm_coop<bfloat, SM, SN, NSG, false, true, false, ACCUM, false, walk_by_b>(
        A, B, C, M, N, K, tiles_n, tiles_m, tgpig, nullptr, 1.0f, 0.0f, 0u);
}

#define TN_NT_COOP_KERNEL(NAME, IMPL, SM, SN, NSG, ACCUM)                      \
    kernel void NAME(device bfloat *A [[buffer(0)]],                           \
                     device bfloat *B [[buffer(1)]],                           \
                     device float *C [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &tiles_n [[buffer(6)]],                     \
                     constant uint &tiles_m [[buffer(7)]],                     \
                     uint tgpig [[threadgroup_position_in_grid]]) {            \
        IMPL<SM, SN, NSG, ACCUM>(A, B, C, M, N, K, tiles_n, tiles_m, tgpig);   \
    }

TN_NT_COOP_KERNEL(matmul2d_tensorops_tn_bf16_f32,       mm_tn_coop_bf16, 128, 64, 4, false)
TN_NT_COOP_KERNEL(matmul2d_tensorops_nt_bf16_f32,       mm_nt_coop_bf16, 128, 64, 4, false)
TN_NT_COOP_KERNEL(matmul2d_tensorops_tn_accum_bf16_f32, mm_tn_coop_bf16,  64, 64, 4, true)
TN_NT_COOP_KERNEL(matmul2d_tensorops_nt_accum_bf16_f32, mm_nt_coop_bf16,  64, 64, 4, true)
