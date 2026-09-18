// Prefill FlashAttention: one simdgroup per query row.
// K/V use [B,kv_capacity,Hkv,D]; Tkv is the live prefix and never a stride.
//
// The general FA-2 kernels tile BR query rows into a 32-thread threadgroup and
// guard the inner loops with `row_valid = lid < BR`, so **8 lanes in 32 do the
// arithmetic** and the other 24 only help stage tiles. Measured on an M5 Pro,
// `swa128_prefill_4096` reached 241 GFLOP/s -- 3.7% of the same machine's f32
// GEMM peak -- against MLX at 998 GFLOP/s. Occupancy, not bandwidth.
//
// This inverts the mapping: a simdgroup owns one query row, and lane L owns
// head dims L, L+32, L+64, ... The consequences are what make it fast:
//
//   * every lane is live, and the K/V reads are one coalesced 128-byte line
//     per simdgroup step;
//   * the P@V accumulate needs no cross-lane communication -- each lane owns
//     its own slice of the output row, in registers, so there is no `Oacc`
//     threadgroup array and no barrier around it;
//   * a score is one butterfly reduction, which every lane then holds, so the
//     online-softmax state is uniform and the kernel is divergence free;
//   * because a simdgroup owns *one* row rather than a BR tile, its key range
//     is that row's exact `[max(0, q_abs-window+1), q_abs]`. The tiled kernels
//     had to take the union window over their BR rows and then mask inside it,
//     so they iterated key blocks that were fully masked for most of the tile.
//     Here masked keys are never visited at all.
//
// Same masking rule as everywhere else: q_abs = q_pos_offset + t_q,
// k_abs = kv_pos_offset + t_k, keep k_abs <= q_abs, and additionally
// k_abs >= max(0, q_abs - window + 1) when `window` is nonzero. `window == 0`
// is the global rule. A row with nothing unmasked is zeros, not NaN.
#include <metal_stdlib>
using namespace metal;

constant uint SG_W = 32;

// `R` -- lanes per query row -- is the tuning knob. The reduction that turns
// per-lane partial dots into a score costs log2(R) shuffle-and-add steps that
// produce no arithmetic, against 2*D/R fused multiply-adds that do. At R=32
// that reduction is 38% of the inner loop at D=128; at R=8 it is 9%. Narrowing
// R trades reduction steps for per-lane work and lets one simdgroup carry
// 32/R query rows at once.
//
// The cost is that those 32/R rows share a simdgroup, so the loop bound has to
// be their union range with per-row masking inside it -- for consecutive causal
// rows that is at most 32/R extra keys per row. Which R wins is measured, not
// assumed: see `bench_flash_attn` with BENCH_ATTN_ROWS_R.
// `SGT` -- simdgroups per threadgroup -- is the second knob, and it is what
// decides how much K/V reuse one global read buys. Every simdgroup in a
// threadgroup walks the *same* key range, so a threadgroup's K/V lines are
// read once from L2 and served `SGT` times from L1; the rows a threadgroup
// covers are `SGT * 32/R`.
//
// It was a fixed 8 for every head dim, which is why D=512 lagged: at R=32 a
// simdgroup carries one query row, so 8 simdgroups meant a K/V line served 8
// rows, against 32 at D=128/R=8. Same arithmetic per byte, four times the L1
// traffic. `SGT` is now compiled per instantiation and swept with
// BENCH_ATTN_ROWS_SGT; 32 simdgroups is 1024 threads, the Metal maximum.
#define ROWS_KERNEL(NAME, D, R, SGT)                                          \
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
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint2 tpitg [[thread_position_in_threadgroup]])                           \
{                                                                             \
    constexpr uint RPS = SG_W / (R);   /* query rows per simdgroup */         \
    /* Head dims per lane, as float4s -- a lane owns four *consecutive* dims  \
       per step. The scalar mapping issued one memory instruction per FMA at  \
       D=512 (16 K loads and 16 V loads against 32 multiply-adds), which is   \
       an ALU pipeline starved by address arithmetic rather than by DRAM.     \
       D % (4R) == 0 for every instantiation, and K/Q/V row bases are         \
       multiples of D, so the float4 reads are aligned. */                    \
    constexpr uint DPV = (D) / (4u * (R));                                    \
    constexpr uint RPT = (SGT) * RPS;  /* query rows per threadgroup */       \
    const uint tid = tpitg.x;                                                 \
    const uint sg = tid / SG_W;                                               \
    const uint lane = tid % SG_W;                                             \
    const uint sub = lane / (R);       /* which row inside the simdgroup */   \
    const uint dl = lane % (R);        /* which dim slice */                  \
                                                                              \
    /* Clamp mutable device state before it participates in any address. */   \
    const uint Tkv = min(*Tkv_ptr, kv_capacity);                              \
    const uint bh = tgpig.y;                                                  \
    const uint h = bh % H;                                                    \
    const uint b = bh / H;                                                    \
    const ulong base_row = (ulong)tgpig.x * RPT + sg * RPS;                  \
    /* Uniform across the simdgroup: the butterfly below needs every lane of  \
       the simdgroup active, so a partially-out-of-range simdgroup keeps all  \
       its lanes and simply does not store. */                                \
    if (base_row >= (ulong)Tq) { return; }                                    \
    const uint live_rows = min(RPS, (uint)((ulong)Tq - base_row));            \
    const bool row_live = sub < live_rows;                                    \
    const uint t_q = row_live ? (uint)(base_row + sub) : (uint)base_row;      \
                                                                              \
    const uint group = max(H / Hkv, 1u);                                      \
    const uint hkv = h / group;                                               \
    const ulong kv_pos_stride = (ulong)Hkv * (D);                             \
    const ulong kv_head_base =                                                \
        (ulong)b * kv_capacity * kv_pos_stride + (ulong)hkv * (D);            \
    const ulong q_pos_stride = (ulong)H * (D);                                \
    const ulong q_head_base =                                                 \
        (ulong)b * Tq * q_pos_stride + (ulong)h * (D);                        \
    const ulong q_off_i = (ulong)(*q_pos_offset_ptr);                         \
    const ulong kv_off = (ulong)(*kv_pos_offset_ptr);                         \
    const ulong q_abs = q_off_i + (ulong)t_q;                                 \
    const ulong window_back = ((window) == 0u) ? 0ul                          \
                                                : (ulong)window - 1ul;         \
    const ulong my_lo = ((window) == 0u || q_abs < window_back)               \
        ? 0ul                                                                 \
        : q_abs - window_back;                                                 \
                                                                              \
    /* Union key range over the RPS rows this simdgroup owns, so every lane   \
       runs the same trip count and the butterfly is never divergent. Rows    \
       mask individually inside it. */                                        \
    const ulong q_lo = q_off_i + base_row;                                    \
    const ulong q_hi = q_off_i + base_row + live_rows - 1ul;                  \
    const ulong u_lo = ((window) == 0u || q_lo < window_back)                 \
        ? 0ul                                                                 \
        : q_lo - window_back;                                                  \
    const ulong t_start = (u_lo > kv_off)                                     \
        ? min((ulong)Tkv, u_lo - kv_off)                                      \
        : 0ul;                                                                \
    const ulong causal_end = (q_hi >= kv_off) ? q_hi - kv_off + 1ul : 0ul;    \
    const ulong t_end = min((ulong)Tkv, causal_end);                          \
                                                                              \
    const ulong o_off = q_head_base + (ulong)t_q * q_pos_stride;             \
    float4 q_reg[DPV];                                                        \
    float4 acc[DPV];                                                          \
    for (uint j = 0; j < DPV; ++j) { acc[j] = float4(0.0f); }                 \
    float m_i = -INFINITY;                                                    \
    float l_i = 0.0f;                                                         \
                                                                              \
    if (t_start < t_end) {                                                    \
        const ulong q_base = o_off;                                           \
        device const float4 *Q4 = (device const float4 *)(Q + q_base);        \
        for (uint j = 0; j < DPV; ++j) {                                      \
            q_reg[j] = row_live ? Q4[dl + j * (R)] : float4(0.0f);            \
        }                                                                     \
        for (uint t = (uint)t_start; t < (uint)t_end; ++t) {                  \
            const ulong kv_base = kv_head_base + (ulong)t * kv_pos_stride;   \
            device const float4 *K4 = (device const float4 *)(K + kv_base);   \
            float4 dot4 = float4(0.0f);                                       \
            for (uint j = 0; j < DPV; ++j) {                                  \
                dot4 += q_reg[j] * K4[dl + j * (R)];                          \
            }                                                                 \
            float part = dot4.x + dot4.y + dot4.z + dot4.w;                   \
            /* Butterfly over the R lanes of this row: leaves the full sum in \
               every one of them, so no broadcast step is needed and the      \
               softmax state stays uniform across the row's lanes. */         \
            for (uint off = (R) / 2u; off > 0u; off >>= 1) {                  \
                part += simd_shuffle_xor(part, off);                          \
            }                                                                 \
            const ulong k_abs = kv_off + (ulong)t;                            \
            float s = part * scale;                                           \
            if (!row_live || k_abs > q_abs || k_abs < my_lo) {                \
                s = -INFINITY;                                                \
            }                                                                 \
            const float m_new = max(m_i, s);                                  \
            /* Both -inf when this row has seen nothing and this key is       \
               masked for it: exp(-inf - -inf) is NaN, and the accumulator is \
               zero anyway, so the rescale is exactly zero. */                \
            const float alpha = (m_i == -INFINITY) ? 0.0f : exp(m_i - m_new); \
            const float p = (s == -INFINITY) ? 0.0f : exp(s - m_new);         \
            device const float4 *V4 = (device const float4 *)(V + kv_base);   \
            for (uint j = 0; j < DPV; ++j) {                                  \
                acc[j] = acc[j] * alpha + p * V4[dl + j * (R)];               \
            }                                                                 \
            l_i = l_i * alpha + p;                                            \
            m_i = (m_new == -INFINITY) ? -INFINITY : m_new;                   \
        }                                                                     \
    }                                                                         \
                                                                              \
    if (!row_live) { return; }                                                \
    const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;                   \
    if (out_bf16 != 0u) {                                                     \
        device bfloat *Ob = (device bfloat *)O;                               \
        for (uint j = 0; j < DPV; ++j) {                                      \
            const ulong d0 = o_off + 4u * (dl + j * (R));                     \
            const float4 o4 = acc[j] * inv_l;                                 \
            Ob[d0 + 0u] = bfloat(o4.x);                                       \
            Ob[d0 + 1u] = bfloat(o4.y);                                       \
            Ob[d0 + 2u] = bfloat(o4.z);                                       \
            Ob[d0 + 3u] = bfloat(o4.w);                                       \
        }                                                                     \
    } else {                                                                  \
        device float4 *O4 = (device float4 *)(O + o_off);                     \
        for (uint j = 0; j < DPV; ++j) {                                      \
            O4[dl + j * (R)] = acc[j] * inv_l;                                \
        }                                                                     \
    }                                                                         \
    (void)B;                                                                  \
}

ROWS_KERNEL(flash_attn_rows_h128_r32_g8, 128, 32, 8)
ROWS_KERNEL(flash_attn_rows_h128_r32_g16, 128, 32, 16)
ROWS_KERNEL(flash_attn_rows_h128_r32_g32, 128, 32, 32)
ROWS_KERNEL(flash_attn_rows_h128_r16_g8, 128, 16, 8)
ROWS_KERNEL(flash_attn_rows_h128_r16_g16, 128, 16, 16)
ROWS_KERNEL(flash_attn_rows_h128_r16_g32, 128, 16, 32)
ROWS_KERNEL(flash_attn_rows_h128_r8_g8, 128, 8, 8)
ROWS_KERNEL(flash_attn_rows_h128_r8_g16, 128, 8, 16)
ROWS_KERNEL(flash_attn_rows_h128_r8_g32, 128, 8, 32)
ROWS_KERNEL(flash_attn_rows_h256_r32_g8, 256, 32, 8)
ROWS_KERNEL(flash_attn_rows_h256_r32_g16, 256, 32, 16)
ROWS_KERNEL(flash_attn_rows_h256_r32_g32, 256, 32, 32)
ROWS_KERNEL(flash_attn_rows_h256_r16_g8, 256, 16, 8)
ROWS_KERNEL(flash_attn_rows_h256_r16_g16, 256, 16, 16)
ROWS_KERNEL(flash_attn_rows_h256_r16_g32, 256, 16, 32)
ROWS_KERNEL(flash_attn_rows_h256_r8_g8, 256, 8, 8)
ROWS_KERNEL(flash_attn_rows_h256_r8_g16, 256, 8, 16)
ROWS_KERNEL(flash_attn_rows_h256_r8_g32, 256, 8, 32)
ROWS_KERNEL(flash_attn_rows_h512_r32_g8, 512, 32, 8)
ROWS_KERNEL(flash_attn_rows_h512_r32_g16, 512, 32, 16)
ROWS_KERNEL(flash_attn_rows_h512_r32_g32, 512, 32, 32)
ROWS_KERNEL(flash_attn_rows_h512_r16_g8, 512, 16, 8)
ROWS_KERNEL(flash_attn_rows_h512_r16_g16, 512, 16, 16)
ROWS_KERNEL(flash_attn_rows_h512_r16_g32, 512, 16, 32)
ROWS_KERNEL(flash_attn_rows_h512_r8_g8, 512, 8, 8)
ROWS_KERNEL(flash_attn_rows_h512_r8_g16, 512, 8, 16)
ROWS_KERNEL(flash_attn_rows_h512_r8_g32, 512, 8, 32)
