// Tile-geometry A/B for the bf16 NN GEMM. Isolates two suspects behind the
// ~2× gap vs PyTorch MPS bf16:
//   (a) output-tile size -> arithmetic intensity  (SM*SN/(SM+SN) FLOP/byte)
//   (b) the host-side zero_f32(C) pre-pass, which only exists because the
//       production kernel runs multiply_accumulate on the FIRST K block too.
// Interior-only (exact divisibility) — this is a measurement rig, not a
// drop-in: the production kernel keeps the ragged-edge paths.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

inline uint2 tune_tile_from_linear(uint linear, uint tiles_n) {
    return uint2(linear % tiles_n, linear / tiles_n);
}

/// ACCUM_FIRST=true reproduces production (all blocks accumulate; C must be
/// pre-zeroed). ACCUM_FIRST=false makes block 0 `multiply`, retiring the zero.
template <int SM, int SN, int BK, int NSG, bool ACCUM_FIRST>
inline void mm_bf16_tune(device bfloat *A, device bfloat *B, device float *C,
                         uint M, uint N, uint K, uint tiles_n, uint tgpig) {
    constexpr auto d_mul = matmul2d_descriptor(
        SM, SN, BK, false, false, false, matmul2d_descriptor::mode::multiply);
    constexpr auto d_acc = matmul2d_descriptor(
        SM, SN, BK, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d_mul, execution_simdgroups<NSG>> op_mul;
    matmul2d<d_acc, execution_simdgroups<NSG>> op_acc;

    uint2 tile = tune_tile_from_linear(tgpig, tiles_n);
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    if (tx + SN > (int)N || ty + SM > (int)M) return;

    auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                     array<int, 2>{1, (int)N});
    for (int k = 0; k + BK <= (int)K; k += BK) {
        auto tA = tensor(A + ty * (int)K + k, dextents<int, 2>{BK, SM},
                         array<int, 2>{1, (int)K});
        auto tB = tensor(B + k * (int)N + tx, dextents<int, 2>{SN, BK},
                         array<int, 2>{1, (int)N});
        if (!ACCUM_FIRST && k == 0) {
            op_mul.run(tA, tB, tC);
        } else {
            op_acc.run(tA, tB, tC);
        }
    }
}

#define TUNE_KERNEL(NAME, SM, SN, BK, NSG, ACCF)                              \
    kernel void NAME(device bfloat *A [[buffer(0)]],                          \
                     device bfloat *B [[buffer(1)]],                          \
                     device float *C [[buffer(2)]],                           \
                     constant uint &M [[buffer(3)]],                          \
                     constant uint &N [[buffer(4)]],                          \
                     constant uint &K [[buffer(5)]],                          \
                     constant uint &tiles_n [[buffer(6)]],                    \
                     uint tgpig [[threadgroup_position_in_grid]]) {           \
        mm_bf16_tune<SM, SN, BK, NSG, ACCF>(A, B, C, M, N, K, tiles_n, tgpig); \
    }

// Control: production geometry + production accumulate-first behaviour.
TUNE_KERNEL(mm_bf16_64x32_bk128_sg4_accf,  64,  32, 128, 4, true)
// Same geometry, zero pre-pass retired.
TUNE_KERNEL(mm_bf16_64x32_bk128_sg4,       64,  32, 128, 4, false)
// Arithmetic-intensity ladder.
TUNE_KERNEL(mm_bf16_64x64_bk64_sg4,        64,  64,  64, 4, false)
TUNE_KERNEL(mm_bf16_128x64_bk64_sg4,      128,  64,  64, 4, false)
TUNE_KERNEL(mm_bf16_128x64_bk64_sg8,      128,  64,  64, 8, false)
TUNE_KERNEL(mm_bf16_128x128_bk64_sg8,     128, 128,  64, 8, false)
TUNE_KERNEL(mm_bf16_128x128_bk32_sg8,     128, 128,  32, 8, false)

// BK ladder at a fixed 64x64 tile: isolates C read-modify-write traffic, which
// scales as K/BK passes over the output tile and is independent of SM/SN.
TUNE_KERNEL(mm_bf16_64x64_bk32_sg4,   64, 64,  32, 4, false)
TUNE_KERNEL(mm_bf16_64x64_bk128_sg4,  64, 64, 128, 4, false)
TUNE_KERNEL(mm_bf16_64x64_bk256_sg4,  64, 64, 256, 4, false)
TUNE_KERNEL(mm_bf16_64x64_bk512_sg4,  64, 64, 512, 4, false)

// Asymptote: at BK >= K the loop runs a single block, so C is touched exactly
// once — the floor for this tile with no accumulate round-trips at all.
TUNE_KERNEL(mm_bf16_64x64_bk1024_sg4, 64, 64, 1024, 4, false)
TUNE_KERNEL(mm_bf16_64x64_bk2048_sg4, 64, 64, 2048, 4, false)
TUNE_KERNEL(mm_bf16_64x64_bk4096_sg4, 64, 64, 4096, 4, false)

// Missing cell: large tile AND large BK together. Tile size raises arithmetic
// intensity; BK cuts C round-trips. Earlier runs varied them one at a time, so
// every big-tile variant was still paying full accumulate traffic.
TUNE_KERNEL(mm_bf16_128x64_bk256_sg4,   128,  64, 256, 4, false)
TUNE_KERNEL(mm_bf16_128x64_bk256_sg8,   128,  64, 256, 8, false)
TUNE_KERNEL(mm_bf16_128x128_bk256_sg8,  128, 128, 256, 8, false)
TUNE_KERNEL(mm_bf16_128x128_bk256_sg4,  128, 128, 256, 4, false)
TUNE_KERNEL(mm_bf16_256x64_bk256_sg8,   256,  64, 256, 8, false)

// =============================================================================
// Cooperative destination tensor (MPPTensorOpsMatMul2d.h "simpleMatMulCooperative"):
// the accumulator lives in registers for the whole K reduction and C is written
// exactly once by cT.store(). No BK loop, no zero pre-pass, no C round-trips —
// the asymptote the BK ladder approaches, at any K. Register cost is
// SM*SN*4 / (32*NSG) bytes per thread, which is why big tiles pair with sg8.
// =============================================================================

template <int SM, int SN, int NSG>
inline void mm_bf16_coop(device bfloat *A, device bfloat *B, device float *C,
                         uint M, uint N, uint K, uint tiles_n, uint tgpig) {
    constexpr auto d = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, false, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<NSG>> op;

    uint2 tile = tune_tile_from_linear(tgpig, tiles_n);
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    if (tx + SN > (int)N || ty + SM > (int)M) return;

    auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                     array<int, 2>{1, (int)K});
    auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K},
                     array<int, 2>{1, (int)N});
    auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                     array<int, 2>{1, (int)N});

    auto cT = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(tA)>,
        metal::remove_addrspace_t<decltype(tB)>, float>();
    // Header doc says get_mask; the shipping metal_cooperative_tensor spells
    // it is_valid_element (set() wraps the same check).
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        cT.set(i, 0.0f);
    }
    op.run(tA, tB, cT);
    cT.store(tC);
}

#define TUNE_COOP_KERNEL(NAME, SM, SN, NSG)                                    \
    kernel void NAME(device bfloat *A [[buffer(0)]],                           \
                     device bfloat *B [[buffer(1)]],                           \
                     device float *C [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &tiles_n [[buffer(6)]],                     \
                     uint tgpig [[threadgroup_position_in_grid]]) {            \
        mm_bf16_coop<SM, SN, NSG>(A, B, C, M, N, K, tiles_n, tgpig);           \
    }

TUNE_COOP_KERNEL(mm_bf16_coop_64x32_sg4,    64,  32, 4)
TUNE_COOP_KERNEL(mm_bf16_coop_64x64_sg4,    64,  64, 4)
TUNE_COOP_KERNEL(mm_bf16_coop_128x64_sg4,  128,  64, 4)
TUNE_COOP_KERNEL(mm_bf16_coop_128x64_sg8,  128,  64, 8)
TUNE_COOP_KERNEL(mm_bf16_coop_128x128_sg8, 128, 128, 8)
TUNE_COOP_KERNEL(mm_bf16_coop_256x64_sg8,  256,  64, 8)

// =============================================================================
// TN / NT / accumulate coop round (training backward lanes), plus a grid
// swizzle for the NN large-square operand-reread question.
// =============================================================================

/// C[M,N] = A_stored[K,M]^T @ B[K,N] — coop destination, interior-only rig.
template <int SM, int SN, int NSG>
inline void mm_bf16_tn_coop(device bfloat *A, device bfloat *B, device float *C,
                            uint M, uint N, uint K, uint tiles_n, uint tgpig) {
    constexpr auto d = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, true, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<NSG>> op;
    uint2 tile = tune_tile_from_linear(tgpig, tiles_n);
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    if (tx + SN > (int)N || ty + SM > (int)M) return;
    auto tA = tensor(A + ty, dextents<int, 2>{SM, (int)K}, array<int, 2>{1, (int)M});
    auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K}, array<int, 2>{1, (int)N});
    auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                     array<int, 2>{1, (int)N});
    auto cT = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(tA)>,
        metal::remove_addrspace_t<decltype(tB)>, float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        cT.set(i, 0.0f);
    op.run(tA, tB, cT);
    cT.store(tC);
}

/// C[M,N] = A[M,K] @ B_stored[N,K]^T — coop destination, interior-only rig.
template <int SM, int SN, int NSG>
inline void mm_bf16_nt_coop(device bfloat *A, device bfloat *B, device float *C,
                            uint M, uint N, uint K, uint tiles_n, uint tgpig) {
    constexpr auto d = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<NSG>> op;
    uint2 tile = tune_tile_from_linear(tgpig, tiles_n);
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    if (tx + SN > (int)N || ty + SM > (int)M) return;
    auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                     array<int, 2>{1, (int)K});
    auto tB = tensor(B + tx * (int)K, dextents<int, 2>{(int)K, SN},
                     array<int, 2>{1, (int)K});
    auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                     array<int, 2>{1, (int)N});
    auto cT = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(tA)>,
        metal::remove_addrspace_t<decltype(tB)>, float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        cT.set(i, 0.0f);
    op.run(tA, tB, cT);
    cT.store(tC);
}

/// C[M,N] += A_stored[K,M]^T @ B[K,N] — coop zero→run, then load-add-store of
/// the prior C (the MPPTensorOpsMatMul2d.h bias pattern): one C read + one
/// C write total, versus multiply_accumulate's internal round-trips.
template <int SM, int SN, int NSG>
inline void mm_bf16_tn_accum_coop(device bfloat *A, device bfloat *B,
                                  device float *C, uint M, uint N, uint K,
                                  uint tiles_n, uint tgpig) {
    constexpr auto d = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, true, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<NSG>> op;
    uint2 tile = tune_tile_from_linear(tgpig, tiles_n);
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    if (tx + SN > (int)N || ty + SM > (int)M) return;
    auto tA = tensor(A + ty, dextents<int, 2>{SM, (int)K}, array<int, 2>{1, (int)M});
    auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K}, array<int, 2>{1, (int)N});
    auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                     array<int, 2>{1, (int)N});
    auto cT = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(tA)>,
        metal::remove_addrspace_t<decltype(tB)>, float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        cT.set(i, 0.0f);
    op.run(tA, tB, cT);
    auto prevT = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(tA)>,
        metal::remove_addrspace_t<decltype(tB)>, float>();
    prevT.load(tC);
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i))
            cT[i] += prevT[i];
    }
    cT.store(tC);
}

#define TUNE_TN_COOP(NAME, SM, SN, NSG)                                        \
    kernel void NAME(device bfloat *A [[buffer(0)]],                           \
                     device bfloat *B [[buffer(1)]],                           \
                     device float *C [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &tiles_n [[buffer(6)]],                     \
                     uint tgpig [[threadgroup_position_in_grid]]) {            \
        mm_bf16_tn_coop<SM, SN, NSG>(A, B, C, M, N, K, tiles_n, tgpig);        \
    }
#define TUNE_NT_COOP(NAME, SM, SN, NSG)                                        \
    kernel void NAME(device bfloat *A [[buffer(0)]],                           \
                     device bfloat *B [[buffer(1)]],                           \
                     device float *C [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &tiles_n [[buffer(6)]],                     \
                     uint tgpig [[threadgroup_position_in_grid]]) {            \
        mm_bf16_nt_coop<SM, SN, NSG>(A, B, C, M, N, K, tiles_n, tgpig);        \
    }
#define TUNE_TN_ACCUM_COOP(NAME, SM, SN, NSG)                                  \
    kernel void NAME(device bfloat *A [[buffer(0)]],                           \
                     device bfloat *B [[buffer(1)]],                           \
                     device float *C [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &tiles_n [[buffer(6)]],                     \
                     uint tgpig [[threadgroup_position_in_grid]]) {            \
        mm_bf16_tn_accum_coop<SM, SN, NSG>(A, B, C, M, N, K, tiles_n, tgpig);  \
    }

TUNE_TN_COOP(mm_bf16_tn_coop_64x64_sg4,   64, 64, 4)
TUNE_TN_COOP(mm_bf16_tn_coop_128x64_sg4, 128, 64, 4)
TUNE_NT_COOP(mm_bf16_nt_coop_64x64_sg4,   64, 64, 4)
TUNE_NT_COOP(mm_bf16_nt_coop_128x64_sg4, 128, 64, 4)
TUNE_TN_ACCUM_COOP(mm_bf16_tn_accum_coop_64x64_sg4, 64, 64, 4)

/// NN coop with a column-panel grid swizzle (PH tile-rows per band): bounds
/// B-tile rereads to tiles_m/PH full passes instead of tiles_m, at the cost
/// of A-panel locality. Tests the square_4096 operand-reread hypothesis.
template <int SM, int SN, int NSG, int PH>
inline void mm_bf16_coop_swz(device bfloat *A, device bfloat *B, device float *C,
                             uint M, uint N, uint K, uint tiles_n, uint tiles_m,
                             uint tgpig) {
    constexpr auto d = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, false, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<NSG>> op;

    uint band = tgpig / (PH * tiles_n);
    uint rem = tgpig - band * PH * tiles_n;
    uint local_h = min((uint)PH, tiles_m - band * PH);
    uint2 tile = uint2(rem / local_h, band * PH + rem % local_h);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    if (tx + SN > (int)N || ty + SM > (int)M) return;

    auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                     array<int, 2>{1, (int)K});
    auto tB = tensor(B + tx, dextents<int, 2>{SN, (int)K},
                     array<int, 2>{1, (int)N});
    auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                     array<int, 2>{1, (int)N});
    auto cT = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(tA)>,
        metal::remove_addrspace_t<decltype(tB)>, float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        cT.set(i, 0.0f);
    op.run(tA, tB, cT);
    cT.store(tC);
}

#define TUNE_SWZ_KERNEL(NAME, SM, SN, NSG, PH)                                 \
    kernel void NAME(device bfloat *A [[buffer(0)]],                           \
                     device bfloat *B [[buffer(1)]],                           \
                     device float *C [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &tiles_n [[buffer(6)]],                     \
                     constant uint &tiles_m [[buffer(7)]],                     \
                     uint tgpig [[threadgroup_position_in_grid]]) {            \
        mm_bf16_coop_swz<SM, SN, NSG, PH>(A, B, C, M, N, K, tiles_n, tiles_m,  \
                                          tgpig);                              \
    }

TUNE_SWZ_KERNEL(mm_bf16_coop_128x64_sg4_swz4,  128, 64, 4, 4)
TUNE_SWZ_KERNEL(mm_bf16_coop_128x64_sg4_swz8,  128, 64, 4, 8)
TUNE_SWZ_KERNEL(mm_bf16_coop_256x64_sg8_swz4,  256, 64, 8, 4)

/// C[M,N] += A[M,K] @ B_stored[N,K]^T — coop zero→run→load-add-store.
template <int SM, int SN, int NSG>
inline void mm_bf16_nt_accum_coop(device bfloat *A, device bfloat *B,
                                  device float *C, uint M, uint N, uint K,
                                  uint tiles_n, uint tgpig) {
    constexpr auto d = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<NSG>> op;
    uint2 tile = tune_tile_from_linear(tgpig, tiles_n);
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;
    if (tx + SN > (int)N || ty + SM > (int)M) return;
    auto tA = tensor(A + ty * (int)K, dextents<int, 2>{(int)K, SM},
                     array<int, 2>{1, (int)K});
    auto tB = tensor(B + tx * (int)K, dextents<int, 2>{(int)K, SN},
                     array<int, 2>{1, (int)K});
    auto tC = tensor(C + ty * (int)N + tx, dextents<int, 2>{SN, SM},
                     array<int, 2>{1, (int)N});
    auto cT = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(tA)>,
        metal::remove_addrspace_t<decltype(tB)>, float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        cT.set(i, 0.0f);
    op.run(tA, tB, cT);
    auto prevT = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(tA)>,
        metal::remove_addrspace_t<decltype(tB)>, float>();
    prevT.load(tC);
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i))
            cT[i] += prevT[i];
    }
    cT.store(tC);
}

#define TUNE_NT_ACCUM_COOP(NAME, SM, SN, NSG)                                  \
    kernel void NAME(device bfloat *A [[buffer(0)]],                           \
                     device bfloat *B [[buffer(1)]],                           \
                     device float *C [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &tiles_n [[buffer(6)]],                     \
                     uint tgpig [[threadgroup_position_in_grid]]) {            \
        mm_bf16_nt_accum_coop<SM, SN, NSG>(A, B, C, M, N, K, tiles_n, tgpig);  \
    }

TUNE_NT_ACCUM_COOP(mm_bf16_nt_accum_coop_64x64_sg4, 64, 64, 4)
