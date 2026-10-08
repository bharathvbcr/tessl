// Qwen3.5 full-attention prefill on the TensorOps matrix units.
//
// `flash_attn_rows` is scalar f32: one simdgroup per query row, one fused
// multiply-add per lane per dim. At T=8192 on an M5 Pro it measured 169 ms per
// layer on Qwen3.5-2B's shapes (8 query / 2 KV heads of 256), about
// 1.6 TFLOP/s and half the whole forward, where the same machine's exact-f32
// TensorOps GEMM runs at 6.4 TFLOP/s (docs/benchmarking.md). The block
// geometry (BQ queries, BK keys, NSG simdgroups) is a template parameter;
// the instantiations are at the end of the file.
//
// This is FlashAttention-2 with both products on `mpp::tensor_ops::matmul2d`:
//
//   S  = Q_blk · K_blkᵀ          BQ x BK, a cooperative tensor (registers)
//   S -> threadgroup; per-row online softmax in scalar code; P overwrites S
//   O  = O · diag(alpha) + P · V_blk      O is BQ x 256, a cooperative tensor
//
// Q, K and V are read straight from device memory through strided tensor
// views: a head's rows are `H*D` (or `Hkv*D`) floats apart, which is just a
// row stride to MPP, so nothing is staged by hand. Only S/P and the per-row
// softmax state live in threadgroup memory.
//
// Same contract as `flash_attn_rows` at `window == 0` and the same buffer
// slots: q_abs = q_pos_offset + t_q, k_abs = kv_pos_offset + t_k, keep
// k_abs <= q_abs; Tkv = min(*Tkv_ptr, kv_capacity); a row with nothing
// unmasked is zeros. The matrix units accumulate in a different order than
// the scalar kernel, so the two agree to rounding, not bit for bit.
//
// Key blocks strictly above a threadgroup's last query are never visited; the
// mask only bites on the blocks that straddle the diagonal and on the partial
// block at Tkv. Out-of-range rows of a K or V block are outside the tensor's
// extents, which MPP bounds-checks, and their scores are masked to -inf so
// their P is exactly 0 either way.
//
// The body is `attn_tiled.h`'s. Its `ENCODER` mode (`encoder_attn_tiled_*` in
// `encoder_attn.metal`, EmbeddingGemma 2) runs `encoder_attn.metal`'s contract
// instead: bidirectional, keys with |t_k - t_q| <= window (every key at window
// 0), and each batch row's live length `lens[b]` bounding both its keys and
// its queries (a query at or past it is written as zeros). A threadgroup
// visits the union of its rows' windows, [q0 - window, q_last + window],
// clipped to the length.
#include "attn_tiled.h"

constant uint TILED_ATTN_D = 256;

/// One entry point per block geometry: BQ queries by BK keys, NSG simdgroups.
/// The name spells the geometry (`_q{BQ}_k{BK}_sg{NSG}`), and the host builds
/// its grid from the same numbers, so the name is the contract between them.
/// Buffer slots are `flash_attn_rows`'; `window` (9) must be 0 and `B` (4)
/// is unused, as there. `LSE_DECL` / `LSE_PTR` add the training forward's
/// log-sum-exp output at slot 15, or nothing.
#define TILED_ATTN_NO_LSE_DECL
#define TILED_ATTN_LSE_DECL device float *lse [[buffer(15)]],
#define TILED_ATTN_KERNEL_IMPL(NAME, BQ, BK, NSG, LSE_DECL, LSE_PTR)          \
kernel void NAME(                                                             \
    device const float *Q [[buffer(0)]],                                      \
    device const float *K [[buffer(1)]],                                      \
    device const float *V [[buffer(2)]],                                      \
    device float *O [[buffer(3)]],                                            \
    constant uint &B [[buffer(4)]],                                           \
    constant uint &Tq [[buffer(5)]],                                          \
    device const uint *Tkv_ptr [[buffer(6)]],                                 \
    constant uint &H [[buffer(7)]],                                           \
    constant uint &Hkv [[buffer(8)]],                                         \
    constant uint &window [[buffer(9)]],                                      \
    constant float &scale [[buffer(10)]],                                     \
    device const uint *q_pos_offset_ptr [[buffer(11)]],                       \
    device const uint *kv_pos_offset_ptr [[buffer(12)]],                      \
    constant uint &out_bf16 [[buffer(13)]],                                   \
    constant uint &kv_capacity [[buffer(14)]],                                \
    LSE_DECL                                                                  \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint tid [[thread_index_in_threadgroup]])                                 \
{                                                                             \
    threadgroup float S[(BQ) * (BK)];                                         \
    threadgroup float m_row[BQ];                                              \
    threadgroup float l_row[BQ];                                              \
    threadgroup float a_row[BQ];                                              \
    /* Clamp mutable device state before it participates in any address. */   \
    const uint Tkv = min(*Tkv_ptr, kv_capacity);                              \
    /* MPP's type matching rejects const element types; the kernel never      \
       writes Q, K or V. */                                                   \
    attn_tiled_body<TILED_ATTN_D, BQ, BK, NSG, false>(                        \
        const_cast<device float *>(Q), const_cast<device float *>(K),         \
        const_cast<device float *>(V), O, LSE_PTR, Tq, Tkv, H, Hkv, scale,    \
        (ulong)(*q_pos_offset_ptr), (ulong)(*kv_pos_offset_ptr), out_bf16,    \
        kv_capacity, 0u, tgpig, tid, S, m_row, l_row, a_row);                 \
    (void)B;                                                                  \
    (void)window;                                                             \
}
#define TILED_ATTN_KERNEL(NAME, BQ, BK, NSG)                                   \
    TILED_ATTN_KERNEL_IMPL(NAME, BQ, BK, NSG, TILED_ATTN_NO_LSE_DECL, nullptr)

// Every instantiation keeps the O accumulator at BQ * 256 / (NSG * 32) = 64
// floats per thread; they differ in K/V reuse (BQ) and in how often the
// softmax pass and the O rescale run per key (BK).
TILED_ATTN_KERNEL(qwen35_attn_tiled_h256_q32_k32_sg4, 32, 32, 4)
TILED_ATTN_KERNEL(qwen35_attn_tiled_h256_q32_k64_sg4, 32, 64, 4)
TILED_ATTN_KERNEL(qwen35_attn_tiled_h256_q64_k32_sg8, 64, 32, 8)
TILED_ATTN_KERNEL(qwen35_attn_tiled_h256_q64_k64_sg8, 64, 64, 8)

// The training forward: the default geometry, also writing the log-sum-exp.
TILED_ATTN_KERNEL_IMPL(qwen35_attn_tiled_lse_h256_q32_k32_sg4, 32, 32, 4, TILED_ATTN_LSE_DECL, lse)
