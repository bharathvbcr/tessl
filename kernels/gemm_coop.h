#ifndef TESSL_GEMM_COOP_H
#define TESSL_GEMM_COOP_H

// The GEMM tile machinery shared by the production kernels
// (matmul_tensorops.metal) and the measurement-only variants
// (tune/matmul_tensorops_tune.metal): tile walks, the operand views of one
// tile, the cooperative-destination body and its epilogue, and the exact-f32
// body. A tune variant is an instantiation of these helpers with different
// template arguments, so a winning variant and the kernel that ships it are
// the same source.
//
// Note: device pointers must be non-const — `const` poisons MPP type matching.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "gelu.h"

using namespace metal;
using namespace mpp::tensor_ops;

// =============================================================================
// Tile walks: linear threadgroup id -> (x, y) tile.
// =============================================================================

/// Decode Morton/Z-order code → (x, y) tile coordinates.
inline uint2 morton_decode_2d(uint c) {
    uint x = 0, y = 0;
#pragma unroll
    for (uint i = 0; i < 16; ++i) {
        x |= ((c >> (2 * i)) & 1u) << i;
        y |= ((c >> (2 * i + 1)) & 1u) << i;
    }
    return uint2(x, y);
}

/// True for the grids `tile_from_linear` walks in Morton order: square, with
/// a power-of-two side.
inline bool tile_grid_is_morton(uint tiles_n, uint tiles_m) {
    return tiles_n == tiles_m && tiles_n != 0u && (tiles_n & (tiles_n - 1u)) == 0u;
}

/// Decode linear TG id → (x, y) tile. Uses Morton when the grid is square and
/// power-of-two (cache-friendly); otherwise compact row-major (avoids pad tax).
inline uint2 tile_from_linear(uint linear, uint tiles_n, uint tiles_m) {
    if (tile_grid_is_morton(tiles_n, tiles_m)) {
        return morton_decode_2d(linear);
    }
    return uint2(linear % tiles_n, linear / tiles_n);
}

/// Decode linear TG id → (x, y) tile by column panels: bands of `ph` tile
/// rows, each walked down a column before moving right. Threadgroups that run
/// together then share B tiles, so B is read once per band instead of once
/// per tile row. An id past the grid decodes to (tiles_n, tiles_m), which
/// every caller's bounds check rejects.
inline uint2 tile_from_linear_panel(uint linear, uint tiles_n, uint tiles_m, uint ph) {
    if (linear >= tiles_n * tiles_m) return uint2(tiles_n, tiles_m);
    uint band = linear / (ph * tiles_n);
    uint rem = linear - band * ph * tiles_n;
    uint local_h = min(ph, tiles_m - band * ph);
    return uint2(rem / local_h, band * ph + rem % local_h);
}

/// Tile walk of every GEMM family — exact f32, the cooperative NN (bf16, f16,
/// relaxed f32; plain, epilogue and batched), the bf16 TN/NT coop and their
/// accumulates, and the int8 dequant kernels — `SM` rows per tile. Every tile row reads all of B (the N×K
/// operand), so row-major order re-reads B once per tile row. That is free
/// while B stays in cache and DRAM-bound once it does not: the LM head's
/// 50304×768 f32 weight ran at ~2.3 TFLOP/s against ~6.5 at N ≤ 2304. Column
/// panels of `PANEL_BAND_ROWS` rows of C read B once per band instead, so the
/// walk is chosen by B's size, not the grid's; a grid `tile_from_linear`
/// walks in Morton order keeps Morton.
///
/// M5 Pro, panel time over the walk it replaces:
/// - exact f32 (ojas bench/results/2026-10-02-gemm, interleaved A/B, min of
///   4): 0.39–0.47× for NT with B ≥ 50 MB and 0.77× for TN at 32 MB, but
///   ~1.08× for TN at 12.6 MB; 25 MB was within run-to-run noise.
/// - bf16 coop, 512-row bands (ojas bench/results/2026-10-06-gemm-bf16,
///   in-process sweep, median of per-round ratios over 3–6 rounds):
///   0.45–0.95× with B ≥ 24 MiB (0.45–0.73× at K = 768), 0.73–1.01× at
///   20 MiB, 0.87–1.02× at 16 MiB and 0.91–1.00× at 12 MiB. Against Morton
///   on square power-of-two grids, 0.98–1.04×.
/// - int8 dequant (same directory, production A/B, median of 6 paired
///   rounds): 0.73–0.77× with B ≥ 24 MiB, 0.91× at 16 MiB and 0.99–1.00× at
///   8–12 MiB. The same A/B put Morton at 0.99–1.02× of panels for exact f32
///   on square power-of-two grids, so those keep Morton too.
///
/// So the gate counts B's elements: 2^23 is the 32 MiB of f32 the exact
/// kernels' A/B chose, 16 MiB of bf16, where the coop kernels stop losing,
/// and 8 MiB of int8, where panels neither win nor lose. Bands of 512 rows
/// (16 tile rows of 32, 4 of 128, 8 of 64) were best or within noise of 1024
/// and 2048 everywhere but bf16 NT at K = 768 (0.64× against 0.61×), and
/// 1024 lost up to 10% on bf16 TN accumulate.
constexpr constant ulong PANEL_MIN_B_ELEMS = 1ul << 23;
constexpr constant uint PANEL_BAND_ROWS = 512u;
template <int SM>
inline uint2 tile_walk(uint linear, uint tiles_n, uint tiles_m, uint N, uint K) {
    if ((ulong)N * (ulong)K >= PANEL_MIN_B_ELEMS && !tile_grid_is_morton(tiles_n, tiles_m)) {
        return tile_from_linear_panel(linear, tiles_n, tiles_m, PANEL_BAND_ROWS / (uint)SM);
    }
    return tile_from_linear(linear, tiles_n, tiles_m);
}

/// Walk policies, as the `Walk` template argument of `mm_coop` and
/// `mm_exact_f32`.
///
/// `walk_by_b`: production's `tile_walk` (column panels for a large B), for
/// every GEMM family.
struct walk_by_b {
    template <int SM>
    static inline uint2 tile(uint linear, uint tiles_n, uint tiles_m, uint N, uint K) {
        return tile_walk<SM>(linear, tiles_n, tiles_m, N, K);
    }
};

/// `walk_by_grid`: the gate the coop NN kernels walked by until 2026-10-11.
/// Large grids took a column-panel swizzle (8 tile-rows per band), measured
/// +11% at 4096^3 on the blocked tune rig. Interleaved against `walk_by_b`
/// per lane it tied or lost (0.94x on the 2B dX/dW shapes, `walk_by_b`
/// 1.14x on NN at 1024x8192x2048), so every family now walks by B's size;
/// this stays for the tune A/B (bench/results/gemm_panel_walk_m5pro.txt).
struct walk_by_grid {
    template <int SM>
    static inline uint2 tile(uint linear, uint tiles_n, uint tiles_m, uint N, uint K) {
        if (tiles_n * tiles_m >= 2048u) {
            return tile_from_linear_panel(linear, tiles_n, tiles_m, 8u);
        }
        return tile_from_linear(linear, tiles_n, tiles_m);
    }
};

/// `walk_linear`: `tile_from_linear` whatever B's size (Morton on square
/// power-of-two grids, row-major otherwise): `tile_walk` with no panels.
struct walk_linear {
    template <int SM>
    static inline uint2 tile(uint linear, uint tiles_n, uint tiles_m, uint N, uint K) {
        return tile_from_linear(linear, tiles_n, tiles_m);
    }
};

/// `walk_row_major`: plain row-major, whatever the grid; the tune rig's
/// baseline walk.
struct walk_row_major {
    template <int SM>
    static inline uint2 tile(uint linear, uint tiles_n, uint tiles_m, uint N, uint K) {
        return uint2(linear % tiles_n, linear / tiles_n);
    }
};

/// `walk_panel<PH>`: column panels of a fixed `PH` tile rows, whatever B's
/// size; the tune rig's panel variants.
template <uint PH>
struct walk_panel {
    template <int SM>
    static inline uint2 tile(uint linear, uint tiles_n, uint tiles_m, uint N, uint K) {
        return tile_from_linear_panel(linear, tiles_n, tiles_m, PH);
    }
};

// =============================================================================
// Operand views of one tile. `interior` is an offset pointer tensor exactly
// one tile wide; `edge` is the full operand sliced at the tile, which bounds
// checks per the MPPTensorOpsMatMul2d.h cooperative example.
// =============================================================================

/// A, `M × K` (row-major `[M, K]`), or with `TRANS` stored `[K, M]` (the TN
/// layout; the descriptor's `transpose_left` reads it as `[M, K]`).
template <bool TRANS>
struct gemm_a;

template <>
struct gemm_a<false> {
    template <int SM, typename T>
    static inline auto interior(device T *A, int M, int K, int ty) {
        return tensor(A + ty * K, dextents<int, 2>{K, SM}, array<int, 2>{1, K});
    }
    template <typename T>
    static inline auto edge(device T *A, int M, int K, int ty) {
        return tensor(A, dextents<int, 2>{K, M}, array<int, 2>{1, K}).slice(0, ty);
    }
};

template <>
struct gemm_a<true> {
    // A physical [K,M] row-major; MPP TN view extents {M,K} stride {1,M}.
    // Tile origin (ty, 0) in that view → pointer A + ty (col-major-ish dim0).
    template <int SM, typename T>
    static inline auto interior(device T *A, int M, int K, int ty) {
        return tensor(A + ty, dextents<int, 2>{SM, K}, array<int, 2>{1, M});
    }
    template <typename T>
    static inline auto edge(device T *A, int M, int K, int ty) {
        return tensor(A, dextents<int, 2>{M, K}, array<int, 2>{1, M}).slice(ty, 0);
    }
};

/// B, `K × N` (row-major `[K, N]`), or with `TRANS` stored `[N, K]` (the NT
/// layout, `transpose_right`).
template <bool TRANS>
struct gemm_b;

template <>
struct gemm_b<false> {
    template <int SN, typename T>
    static inline auto interior(device T *B, int N, int K, int tx) {
        return tensor(B + tx, dextents<int, 2>{SN, K}, array<int, 2>{1, N});
    }
    template <typename T>
    static inline auto edge(device T *B, int N, int K, int tx) {
        return tensor(B, dextents<int, 2>{N, K}, array<int, 2>{1, N}).slice(tx, 0);
    }
};

template <>
struct gemm_b<true> {
    // B physical [N,K]; MPP NT view extents {K,N} stride {1,K}.
    template <int SN, typename T>
    static inline auto interior(device T *B, int N, int K, int tx) {
        return tensor(B + tx * K, dextents<int, 2>{K, SN}, array<int, 2>{1, K});
    }
    template <typename T>
    static inline auto edge(device T *B, int N, int K, int tx) {
        return tensor(B, dextents<int, 2>{K, N}, array<int, 2>{1, K}).slice(0, tx);
    }
};

/// C, `M × N` row-major, and anything laid over it the same way: a per-column
/// bias is C's view with row stride 0, so every row of the tile reads the same
/// `SN` values through the same cooperative `load` that fetches C.
template <int SM, int SN, typename T>
inline auto gemm_c_interior(device T *C, int N, int tx, int ty, int row_stride) {
    return tensor(C + ty * row_stride + tx, dextents<int, 2>{SN, SM}, array<int, 2>{1, row_stride});
}
template <typename T>
inline auto gemm_c_edge(device T *C, int M, int N, int tx, int ty, int row_stride) {
    return tensor(C, dextents<int, 2>{N, M}, array<int, 2>{1, row_stride}).slice(tx, ty);
}

// =============================================================================
// The cooperative-destination body: the f32 accumulator lives in registers
// for the whole K reduction and C is written exactly once by `store`, so no
// zero pre-pass is needed (every in-bounds element is overwritten). Register
// cost of the accumulator is SM*SN*4/(32*NSG) B/thread.
// =============================================================================

/// Activation applied by a fused GEMM epilogue.
///
/// Values are ABI: they cross from `tessl::gemm::Activation` as a `uint`, so
/// reordering them silently changes what every caller computes.
enum GemmActivation : uint {
    GEMM_ACT_NONE = 0u,
    GEMM_ACT_RELU = 1u,
    GEMM_ACT_GELU_TANH = 2u,
    GEMM_ACT_SILU = 3u,
};

static inline float gemm_apply_activation(float v, GemmActivation act) {
    switch (act) {
        case GEMM_ACT_RELU: return fmax(v, 0.0f);
        // The crate's one GELU (gelu.h), shared with mlp_gelu_tanh and the q4
        // gate/up GEMVs. A local copy here once clipped the output at 20.
        case GEMM_ACT_GELU_TANH: return tessl_gelu_pytorch_tanh(v);
        // silu(x) = x * sigmoid(x), matching `mlp_silu.metal`.
        case GEMM_ACT_SILU: return v / (1.0f + exp(-v));
        default: return v;
    }
}

/// One tile: zero → run → [ACCUM: + C] / [EPILOGUE] → store, on `op`'s
/// cooperative destination for the views `tA`, `tB`, `tC`. `INTERIOR` says
/// which view family they are, for the bias view built the same way.
///
/// ACCUM adds the prior C via the header's cooperative bias pattern
/// (load-add-store): one C read + one C write total, measured 1.4-1.5x over
/// multiply_accumulate. EPILOGUE is `C = act(alpha * A@B + beta * C_prev +
/// bias)`; `beta * C_prev` reuses the same trick (a second cooperative tensor
/// loaded from C, combined in registers), so C is read once and written once
/// rather than round-tripped per term.
template <bool ACCUM, bool EPILOGUE, bool INTERIOR, int SM, int SN, typename Op,
          typename VA, typename VB, typename VC>
inline void coop_tile(thread Op &op, thread VA &tA, thread VB &tB, thread VC &tC,
                      device const float *bias, int tx, int ty, int M, int N,
                      float alpha, float beta, uint act) {
    using TA = metal::remove_addrspace_t<VA>;
    using TB = metal::remove_addrspace_t<VB>;
    auto cT = op.template get_destination_cooperative_tensor<TA, TB, float>();
    // set() wraps the is_valid_element mask check.
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        cT.set(i, 0.0f);
    op.run(tA, tB, cT);
    if (ACCUM) {
        auto prevT = op.template get_destination_cooperative_tensor<TA, TB, float>();
        prevT.load(tC);
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i))
                cT[i] += prevT[i];
        }
    }
    if (EPILOGUE) {
        if (beta != 0.0f) {
            auto prevT = op.template get_destination_cooperative_tensor<TA, TB, float>();
            prevT.load(tC);
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < cT.get_capacity(); ++i)
                if (cT.is_valid_element(i)) cT[i] = cT[i] * alpha + beta * prevT[i];
        } else if (alpha != 1.0f) {
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < cT.get_capacity(); ++i)
                if (cT.is_valid_element(i)) cT[i] *= alpha;
        }
        if (bias) {
            device float *b = const_cast<device float *>(bias);
            auto biasT = op.template get_destination_cooperative_tensor<TA, TB, float>();
            if (INTERIOR) {
                auto tBias = gemm_c_interior<SM, SN>(b, N, tx, 0, 0);
                biasT.load(tBias);
            } else {
                auto tBias = gemm_c_edge(b, M, N, tx, ty, 0);
                biasT.load(tBias);
            }
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < cT.get_capacity(); ++i)
                if (cT.is_valid_element(i)) cT[i] += biasT[i];
        }
        if (act != GEMM_ACT_NONE) {
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < cT.get_capacity(); ++i)
                if (cT.is_valid_element(i)) cT[i] = gemm_apply_activation(cT[i], (GemmActivation)act);
        }
    }
    cT.store(tC);
}

/// A cooperative-destination GEMM: `C (+)= A @ B` over `SM × SN` tiles on
/// `NSG` simdgroups, `TRANS_A` / `TRANS_B` for the TN / NT layouts, `RELAXED`
/// for the reduced-precision f32 descriptor, the tile order from `Walk`.
/// Interior tiles use offset pointer tensors; edges the bounds-checked slice
/// path. `bias`, `alpha`, `beta` and `act` are read only with `EPILOGUE`.
template <typename ElemT, int SM, int SN, int NSG, bool TRANS_A, bool TRANS_B,
          bool RELAXED, bool ACCUM, bool EPILOGUE, typename Walk>
inline void mm_coop(device ElemT *A, device ElemT *B, device float *C,
                    uint M, uint N, uint K, uint tiles_n, uint tiles_m, uint tgpig,
                    device const float *bias, float alpha, float beta, uint act) {
    constexpr auto d = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, TRANS_A, TRANS_B, RELAXED,
        matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<NSG>> op;

    uint2 tile = Walk::template tile<SM>(tgpig, tiles_n, tiles_m, N, K);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    bool interior = (tx + SN <= (int)N) && (ty + SM <= (int)M);
    if (interior) {
        auto tA = gemm_a<TRANS_A>::template interior<SM>(A, (int)M, (int)K, ty);
        auto tB = gemm_b<TRANS_B>::template interior<SN>(B, (int)N, (int)K, tx);
        auto tC = gemm_c_interior<SM, SN>(C, (int)N, tx, ty, (int)N);
        coop_tile<ACCUM, EPILOGUE, true, SM, SN>(op, tA, tB, tC, bias, tx, ty, (int)M,
                                                 (int)N, alpha, beta, act);
    } else {
        auto tA = gemm_a<TRANS_A>::edge(A, (int)M, (int)K, ty);
        auto tB = gemm_b<TRANS_B>::edge(B, (int)N, (int)K, tx);
        auto tC = gemm_c_edge(C, (int)M, (int)N, tx, ty, (int)N);
        coop_tile<ACCUM, EPILOGUE, false, SM, SN>(op, tA, tB, tC, bias, tx, ty, (int)M,
                                                  (int)N, alpha, beta, act);
    }
}

// =============================================================================
// Exact f32 — execution_simdgroup, SM=SN=32 (golden-safe): `mode::multiply`
// into a C the host zeroed first, or `multiply_accumulate` into C.
// =============================================================================

/// `C (+)= A @ B` exact f32, `TRANS_A` / `TRANS_B` for TN / NT, `ACCUMULATE`
/// for `multiply_accumulate`, the tile order from `Walk` (production:
/// `walk_by_b`). Every tile takes the bounds-checked slice path: offset
/// pointer tensors for interior tiles (`TESSL_GEMM_INTERIOR`) measured
/// 0.99-1.01x on the Qwen3.5-2B training shapes and were retired
/// (bench/results/gemm_interior_m5pro.txt).
template <bool TRANS_A, bool TRANS_B, bool ACCUMULATE, int SM, int SN,
          typename Walk = walk_by_b>
inline void mm_exact_f32(device float *A, device float *B, device float *C,
                         uint M, uint N, uint K, uint tiles_n, uint tiles_m,
                         uint tgpig) {
    constexpr auto desc = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, TRANS_A, TRANS_B, false,
        ACCUMULATE ? matmul2d_descriptor::mode::multiply_accumulate
                   : matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;

    uint2 tile = Walk::template tile<SM>(tgpig, tiles_n, tiles_m, N, K);
    if (tile.x >= tiles_n || tile.y >= tiles_m) return;
    int tx = (int)tile.x * SN;
    int ty = (int)tile.y * SM;

    auto tA = gemm_a<TRANS_A>::edge(A, (int)M, (int)K, ty);
    auto tB = gemm_b<TRANS_B>::edge(B, (int)N, (int)K, tx);
    auto tC = gemm_c_edge(C, (int)M, (int)N, tx, ty, (int)N);
    op.run(tA, tB, tC);
}

#endif
