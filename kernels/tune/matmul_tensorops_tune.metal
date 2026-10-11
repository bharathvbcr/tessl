// Measurement-only GEMM variants, linked only by `TESSL_GEMM_TUNE=1` builds
// and timed against production by bench_gemm_tile_tune and
// bench_gemm_tnnt_tune. The first section is the 2026-08 bf16 NN tile/BK
// ladder, which asked whether output-tile arithmetic intensity
// (SM*SN/(SM+SN) FLOP/byte) or the zero_f32(C) pre-pass of the then-production
// kernel (`multiply_accumulate` on every K block, the first included) caused
// the ~2× gap to PyTorch MPS bf16. Its answer, the cooperative destination
// tensor, is production now: the NN, TN and NT coop kernels in
// matmul_tensorops.metal have no pre-pass. Interior-only (exact
// divisibility) — a measurement rig, not a drop-in: production keeps the
// ragged-edge paths.

#include "../gemm_coop.h"

// Every cooperative variant below is an instantiation of gemm_coop.h's
// `mm_coop`, the body the production coop kernels run, with its own tile,
// walk and operand layout. The rig's shapes divide exactly, so a kernel
// without a `tiles_m` argument takes `M / SM`; every tile is interior.

/// ACCUM_FIRST=true reproduces the pre-coop production NN kernel (every block
/// accumulates, so C must be pre-zeroed); ACCUM_FIRST=false makes block 0
/// `multiply`, retiring the zero.
// A call with no explicit template arguments is the control kernel
// `mm_bf16_64x32_bk128_sg4_accf` (64×32, BK 128, sg4, accumulate-first).
template <int SM = 64, int SN = 32, int BK = 128, int NSG = 4,
          bool ACCUM_FIRST = true>
inline void mm_bf16_tune(device bfloat *A, device bfloat *B, device float *C,
                         uint M, uint N, uint K, uint tiles_n, uint tgpig) {
    constexpr auto d_mul = matmul2d_descriptor(
        SM, SN, BK, false, false, false, matmul2d_descriptor::mode::multiply);
    constexpr auto d_acc = matmul2d_descriptor(
        SM, SN, BK, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d_mul, execution_simdgroups<NSG>> op_mul;
    matmul2d<d_acc, execution_simdgroups<NSG>> op_acc;

    uint2 tile = walk_row_major::tile<SM>(tgpig, tiles_n, M / SM, N, K);
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

// Control: the pre-coop production geometry and accumulate-first behaviour.
kernel void mm_bf16_64x32_bk128_sg4_accf(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]]) {
    mm_bf16_tune(A, B, C, M, N, K, tiles_n, tgpig);
}
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

// A call with no explicit template arguments is `mm_bf16_coop_64x32_sg4`.
// ElemT = float is the exact-f32 (non-relaxed) descriptor: production's exact
// f32 NN runs 32x32 on one simdgroup with a zero pre-pass (`TILE_F32`), with
// no register accumulator; these variants ask whether one pays at f32.
template <int SM = 64, int SN = 32, int NSG = 4, typename ElemT = bfloat>
inline void mm_bf16_coop(device ElemT *A, device ElemT *B, device float *C,
                         uint M, uint N, uint K, uint tiles_n, uint tgpig) {
    mm_coop<ElemT, SM, SN, NSG, false, false, false, false, false, walk_row_major>(
        A, B, C, M, N, K, tiles_n, M / SM, tgpig, nullptr, 1.0f, 0.0f, 0u);
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

kernel void mm_bf16_coop_64x32_sg4(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]]) {
    mm_bf16_coop(A, B, C, M, N, K, tiles_n, tgpig);
}
TUNE_COOP_KERNEL(mm_bf16_coop_64x64_sg4,    64,  64, 4)
TUNE_COOP_KERNEL(mm_bf16_coop_128x64_sg4,  128,  64, 4)
TUNE_COOP_KERNEL(mm_bf16_coop_128x64_sg8,  128,  64, 8)
TUNE_COOP_KERNEL(mm_bf16_coop_128x128_sg8, 128, 128, 8)
TUNE_COOP_KERNEL(mm_bf16_coop_256x64_sg8,  256,  64, 8)

#define TUNE_COOP_KERNEL_F32(NAME, SM, SN, NSG)                                \
    kernel void NAME(device float *A [[buffer(0)]],                            \
                     device float *B [[buffer(1)]],                            \
                     device float *C [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &tiles_n [[buffer(6)]],                     \
                     uint tgpig [[threadgroup_position_in_grid]]) {            \
        mm_bf16_coop<SM, SN, NSG, float>(A, B, C, M, N, K, tiles_n, tgpig);    \
    }
TUNE_COOP_KERNEL_F32(mm_f32_exact_coop_64x64_sg4,   64, 64, 4)
TUNE_COOP_KERNEL_F32(mm_f32_exact_coop_128x64_sg4, 128, 64, 4)
TUNE_COOP_KERNEL_F32(mm_f32_exact_coop_128x64_sg8, 128, 64, 8)

// =============================================================================
// TN / NT / accumulate coop round (training backward lanes), plus a grid
// swizzle for the NN large-square operand-reread question.
// =============================================================================

/// C[M,N] = A_stored[K,M]^T @ B[K,N] — coop destination, interior-only rig.
// A call with no explicit template arguments is `mm_bf16_tn_coop_64x64_sg4`.
template <int SM = 64, int SN = 64, int NSG = 4>
inline void mm_bf16_tn_coop(device bfloat *A, device bfloat *B, device float *C,
                            uint M, uint N, uint K, uint tiles_n, uint tgpig) {
    mm_coop<bfloat, SM, SN, NSG, true, false, false, false, false, walk_row_major>(
        A, B, C, M, N, K, tiles_n, M / SM, tgpig, nullptr, 1.0f, 0.0f, 0u);
}

/// C[M,N] = A[M,K] @ B_stored[N,K]^T — coop destination, interior-only rig.
// A call with no explicit template arguments is `mm_bf16_nt_coop_64x64_sg4`.
template <int SM = 64, int SN = 64, int NSG = 4>
inline void mm_bf16_nt_coop(device bfloat *A, device bfloat *B, device float *C,
                            uint M, uint N, uint K, uint tiles_n, uint tgpig) {
    mm_coop<bfloat, SM, SN, NSG, false, true, false, false, false, walk_row_major>(
        A, B, C, M, N, K, tiles_n, M / SM, tgpig, nullptr, 1.0f, 0.0f, 0u);
}

/// C[M,N] += A_stored[K,M]^T @ B[K,N] — coop zero→run, then load-add-store of
/// the prior C (the MPPTensorOpsMatMul2d.h bias pattern): one C read + one
/// C write total, versus multiply_accumulate's internal round-trips.
// A call with no explicit template arguments is `mm_bf16_tn_accum_coop_64x64_sg4`.
template <int SM = 64, int SN = 64, int NSG = 4>
inline void mm_bf16_tn_accum_coop(device bfloat *A, device bfloat *B,
                                  device float *C, uint M, uint N, uint K,
                                  uint tiles_n, uint tgpig) {
    mm_coop<bfloat, SM, SN, NSG, true, false, false, true, false, walk_row_major>(
        A, B, C, M, N, K, tiles_n, M / SM, tgpig, nullptr, 1.0f, 0.0f, 0u);
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
kernel void mm_bf16_tn_coop_64x64_sg4(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]]) {
    mm_bf16_tn_coop(A, B, C, M, N, K, tiles_n, tgpig);
}
kernel void mm_bf16_nt_coop_64x64_sg4(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]]) {
    mm_bf16_nt_coop(A, B, C, M, N, K, tiles_n, tgpig);
}
kernel void mm_bf16_tn_accum_coop_64x64_sg4(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]]) {
    mm_bf16_tn_accum_coop(A, B, C, M, N, K, tiles_n, tgpig);
}

TUNE_TN_COOP(mm_bf16_tn_coop_128x64_sg4, 128, 64, 4)
TUNE_NT_COOP(mm_bf16_nt_coop_128x64_sg4, 128, 64, 4)
// The tied LM head as an NT product over the [vocab, hidden] table
// (M = rows, N = 248320, K = 2048): wider N tiles and more simdgroups.
TUNE_NT_COOP(mm_bf16_nt_coop_64x128_sg4, 64, 128, 4)
TUNE_NT_COOP(mm_bf16_nt_coop_128x128_sg4, 128, 128, 4)
TUNE_NT_COOP(mm_bf16_nt_coop_128x128_sg8, 128, 128, 8)
TUNE_NT_COOP(mm_bf16_nt_coop_256x64_sg8, 256, 64, 8)
TUNE_NT_COOP(mm_bf16_nt_coop_64x256_sg8, 64, 256, 8)
TUNE_NT_COOP(mm_bf16_nt_coop_128x256_sg8, 128, 256, 8)
TUNE_NT_COOP(mm_bf16_nt_coop_256x128_sg8, 256, 128, 8)
// Round two: the head's winner was the tallest tile (each row tile streams
// the whole 1 GB table once), so taller still.
TUNE_NT_COOP(mm_bf16_nt_coop_256x64_sg4, 256, 64, 4)
TUNE_NT_COOP(mm_bf16_nt_coop_256x32_sg8, 256, 32, 8)
TUNE_NT_COOP(mm_bf16_nt_coop_512x32_sg8, 512, 32, 8)
TUNE_NT_COOP(mm_bf16_nt_coop_512x64_sg8, 512, 64, 8)
TUNE_NT_COOP(mm_bf16_nt_coop_512x64_sg16, 512, 64, 16)
TUNE_NT_COOP(mm_bf16_nt_coop_1024x32_sg16, 1024, 32, 16)

/// NN coop with a column-panel grid swizzle (PH tile-rows per band): bounds
/// B-tile rereads to tiles_m/PH full passes instead of tiles_m, at the cost
/// of A-panel locality. Tests the square_4096 operand-reread hypothesis.
// A call with no explicit template arguments is `mm_bf16_coop_128x64_sg4_swz4`.
template <int SM = 128, int SN = 64, int NSG = 4, int PH = 4>
inline void mm_bf16_coop_swz(device bfloat *A, device bfloat *B, device float *C,
                             uint M, uint N, uint K, uint tiles_n, uint tiles_m,
                             uint tgpig) {
    mm_coop<bfloat, SM, SN, NSG, false, false, false, false, false, walk_panel<(uint)PH>>(
        A, B, C, M, N, K, tiles_n, tiles_m, tgpig, nullptr, 1.0f, 0.0f, 0u);
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

kernel void mm_bf16_coop_128x64_sg4_swz4(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]]) {
    mm_bf16_coop_swz(A, B, C, M, N, K, tiles_n, tiles_m, tgpig);
}
TUNE_SWZ_KERNEL(mm_bf16_coop_128x64_sg4_swz8,  128, 64, 4, 8)
TUNE_SWZ_KERNEL(mm_bf16_coop_256x64_sg8_swz4,  256, 64, 8, 4)

/// C[M,N] += A[M,K] @ B_stored[N,K]^T — coop zero→run→load-add-store.
// A call with no explicit template arguments is `mm_bf16_nt_accum_coop_64x64_sg4`.
template <int SM = 64, int SN = 64, int NSG = 4>
inline void mm_bf16_nt_accum_coop(device bfloat *A, device bfloat *B,
                                  device float *C, uint M, uint N, uint K,
                                  uint tiles_n, uint tgpig) {
    mm_coop<bfloat, SM, SN, NSG, false, true, false, true, false, walk_row_major>(
        A, B, C, M, N, K, tiles_n, M / SM, tgpig, nullptr, 1.0f, 0.0f, 0u);
}

kernel void mm_bf16_nt_accum_coop_64x64_sg4(
    device bfloat *A [[buffer(0)]],
    device bfloat *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]],
    uint tgpig [[threadgroup_position_in_grid]]) {
    mm_bf16_nt_accum_coop(A, B, C, M, N, K, tiles_n, tgpig);
}

// =============================================================================
// TN / NT coop over a column-panel walk: PH tile rows per band, each band
// walked down a column before moving right, so threadgroups that run together
// share B tiles and a B too large for the cache is read once per band rather
// than once per tile row. The rig for the production TN/NT coop walk; the
// row-major baselines are the mm_bf16_{tn,nt}[_accum]_coop kernels above.
// =============================================================================

/// C[M,N] (+)= A[M,K] @ B_stored[N,K]^T over the panel walk.
template <int SM, int SN, int NSG, int PH, bool ACCUM>
inline void mm_bf16_nt_coop_panel(device bfloat *A, device bfloat *B,
                                  device float *C, uint M, uint N, uint K,
                                  uint tiles_n, uint tiles_m, uint tgpig) {
    mm_coop<bfloat, SM, SN, NSG, false, true, false, ACCUM, false, walk_panel<(uint)PH>>(
        A, B, C, M, N, K, tiles_n, tiles_m, tgpig, nullptr, 1.0f, 0.0f, 0u);
}

/// C[M,N] (+)= A_stored[K,M]^T @ B[K,N] over the panel walk.
template <int SM, int SN, int NSG, int PH, bool ACCUM>
inline void mm_bf16_tn_coop_panel(device bfloat *A, device bfloat *B,
                                  device float *C, uint M, uint N, uint K,
                                  uint tiles_n, uint tiles_m, uint tgpig) {
    mm_coop<bfloat, SM, SN, NSG, true, false, false, ACCUM, false, walk_panel<(uint)PH>>(
        A, B, C, M, N, K, tiles_n, tiles_m, tgpig, nullptr, 1.0f, 0.0f, 0u);
}

#define TUNE_PANEL_KERNEL(NAME, IMPL, SM, SN, NSG, PH, ACCUM)                 \
    kernel void NAME(device bfloat *A [[buffer(0)]],                           \
                     device bfloat *B [[buffer(1)]],                           \
                     device float *C [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &tiles_n [[buffer(6)]],                     \
                     constant uint &tiles_m [[buffer(7)]],                     \
                     uint tgpig [[threadgroup_position_in_grid]]) {            \
        IMPL<SM, SN, NSG, PH, ACCUM>(A, B, C, M, N, K, tiles_n, tiles_m,       \
                                     tgpig);                                   \
    }

TUNE_PANEL_KERNEL(mm_bf16_nt_coop_128x64_sg4_ph4,  mm_bf16_nt_coop_panel, 128, 64, 4,  4, false)
TUNE_PANEL_KERNEL(mm_bf16_nt_coop_128x64_sg4_ph8,  mm_bf16_nt_coop_panel, 128, 64, 4,  8, false)
TUNE_PANEL_KERNEL(mm_bf16_nt_coop_128x64_sg4_ph16, mm_bf16_nt_coop_panel, 128, 64, 4, 16, false)
TUNE_PANEL_KERNEL(mm_bf16_tn_coop_128x64_sg4_ph4,  mm_bf16_tn_coop_panel, 128, 64, 4,  4, false)
TUNE_PANEL_KERNEL(mm_bf16_tn_coop_128x64_sg4_ph8,  mm_bf16_tn_coop_panel, 128, 64, 4,  8, false)
TUNE_PANEL_KERNEL(mm_bf16_tn_coop_128x64_sg4_ph16, mm_bf16_tn_coop_panel, 128, 64, 4, 16, false)
TUNE_PANEL_KERNEL(mm_bf16_nt_accum_coop_64x64_sg4_ph8,  mm_bf16_nt_coop_panel, 64, 64, 4,  8, true)
TUNE_PANEL_KERNEL(mm_bf16_nt_accum_coop_64x64_sg4_ph16, mm_bf16_nt_coop_panel, 64, 64, 4, 16, true)
TUNE_PANEL_KERNEL(mm_bf16_tn_accum_coop_64x64_sg4_ph8,  mm_bf16_tn_coop_panel, 64, 64, 4,  8, true)
TUNE_PANEL_KERNEL(mm_bf16_tn_accum_coop_64x64_sg4_ph16, mm_bf16_tn_coop_panel, 64, 64, 4, 16, true)

// =============================================================================
// One panel-walk policy: the production coop kernels (edges included) with
// the other family's walk. The NN coop kernels walk by grid size
// (`walk_by_grid`: >= 2048 tiles take 8-row panels); TN, NT, their
// accumulate forms, exact f32 and int8 walk by B's size (`walk_by_b`,
// `tile_walk`). `_walkb` / `_walkg` swap them; the bench's production arm is
// the shipped walk.
// =============================================================================

#define TUNE_WALK_KERNEL(NAME, ELEM, SM, SN, NSG, TA, TB, ACCUM, WALK)         \
    kernel void NAME(device ELEM *A [[buffer(0)]],                             \
                     device ELEM *B [[buffer(1)]],                             \
                     device float *C [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &tiles_n [[buffer(6)]],                     \
                     constant uint &tiles_m [[buffer(7)]],                     \
                     uint tgpig [[threadgroup_position_in_grid]]) {            \
        mm_coop<ELEM, SM, SN, NSG, TA, TB, false, ACCUM, false, WALK>(         \
            A, B, C, M, N, K, tiles_n, tiles_m, tgpig, nullptr, 1.0f, 0.0f,    \
            0u);                                                               \
    }

TUNE_WALK_KERNEL(mm_bf16_coop_128x64_sg4_walkb, bfloat, 128, 64, 4, false, false, false, walk_by_b);
TUNE_WALK_KERNEL(mm_bf16_coop_64x64_sg4_walkb,  bfloat,  64, 64, 4, false, false, false, walk_by_b);
TUNE_WALK_KERNEL(mm_bf16_tn_coop_128x64_sg4_walkg, bfloat, 128, 64, 4, true, false, false, walk_by_grid);
TUNE_WALK_KERNEL(mm_bf16_nt_coop_128x64_sg4_walkg, bfloat, 128, 64, 4, false, true, false, walk_by_grid);
TUNE_WALK_KERNEL(mm_bf16_tn_accum_coop_64x64_sg4_walkg, bfloat, 64, 64, 4, true, false, true, walk_by_grid);
TUNE_WALK_KERNEL(mm_bf16_nt_accum_coop_64x64_sg4_walkg, bfloat, 64, 64, 4, false, true, true, walk_by_grid);

// =============================================================================
// Exact f32 NN, the production 32x32 body (`mm_exact_f32`, slice path, C
// zeroed by the bench first) under each walk: the 2048x6144x2048 MLP up/gate
// shape ran at 4.1 TFLOP/s against 5.7 at 2048^3 in one run
// (bench/results/f32_exact_coop_m5pro.txt). `_linear`: `tile_from_linear`
// with no panels; `_rowmajor`; `_phN`: N-row panels.
// =============================================================================

#define TUNE_EXACT_F32_KERNEL(NAME, WALK)                                      \
    kernel void NAME(device float *A [[buffer(0)]],                            \
                     device float *B [[buffer(1)]],                            \
                     device float *C [[buffer(2)]],                            \
                     constant uint &M [[buffer(3)]],                           \
                     constant uint &N [[buffer(4)]],                           \
                     constant uint &K [[buffer(5)]],                           \
                     constant uint &tiles_n [[buffer(6)]],                     \
                     constant uint &tiles_m [[buffer(7)]],                     \
                     uint tgpig [[threadgroup_position_in_grid]]) {            \
        mm_exact_f32<false, false, false, 32, 32, WALK>(                       \
            A, B, C, M, N, K, tiles_n, tiles_m, tgpig);                        \
    }

TUNE_EXACT_F32_KERNEL(mm_f32_exact_32x32_linear,   walk_linear);
TUNE_EXACT_F32_KERNEL(mm_f32_exact_32x32_rowmajor, walk_row_major);
TUNE_EXACT_F32_KERNEL(mm_f32_exact_32x32_ph8,      walk_panel<8u>);
TUNE_EXACT_F32_KERNEL(mm_f32_exact_32x32_ph16,     walk_panel<16u>);
TUNE_EXACT_F32_KERNEL(mm_f32_exact_32x32_ph32,     walk_panel<32u>);
