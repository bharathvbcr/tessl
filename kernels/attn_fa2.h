// The tiled FlashAttention-2 body shared by `flash_attn_swa_h128.metal`,
// `flash_attn_swa_h256.metal` and `flash_attn_global_h512.metal`.
//
// Q: [B,Tq,H,D], K/V: [B,kv_capacity,Hkv,D], O: [B,Tq,H,D]
// Tkv is the live prefix; kv_capacity is the fixed per-batch K/V stride.
// Absolute positions: q_abs = q_pos_offset + t_q, k_abs = kv_pos_offset + t_k.
// Prefill dense: Tq=Tkv=T, offsets 0. Decode / ring densify: Tq=1, Tkv=cache_len.
// scale = 1.0 after QK-Norm.
// Sliding window: q_abs - window < k_abs <= q_abs (window == 0: unbounded).
// Global: k_abs <= q_abs, `window` is not read.
//
// The three kernels differ in HEAD_DIM, BR/BC, the sliding-window mask and
// block skip, and whether `out_bf16` exists. They do NOT differ in anything
// else, and the masked-row NaN fix below used to be applied to each copy.
// What stays in the `.metal` files is exactly what cannot be templated: the
// `[[buffer(N)]]` slots, which the host (and gemma-metal in ojas) binds by
// number and which differ per kernel, and the threadgroup arrays, which Metal
// allows only at kernel scope.
//
// Included, never compiled on its own; `build.rs` tracks `.h` changes.
#pragma once

#include <metal_stdlib>
using namespace metal;

constant uint FA2_D_TILE = 32;

/// `kernel_oacc` etc. are the kernel's threadgroup arrays: `Oacc[BR*HEAD_DIM]`,
/// `Ks[BC*D_TILE]`, `Vs[BC*D_TILE]`, `scores[BR*BC]`.
///
/// `window` is read only when SLIDING. `out_bf16` is honoured only when
/// HAS_OUT_BF16 (otherwise the output is always f32).
template <uint HEAD_DIM, uint BR, uint BC, bool SLIDING, bool HAS_OUT_BF16>
inline void flash_attn_fa2_tiled(
    device const float *Q,
    device const float *K,
    device const float *V,
    device float *O,
    uint Tq,
    device const uint *Tkv_ptr,
    uint H,
    uint Hkv,
    uint window,
    float scale,
    device const uint *q_pos_offset_ptr,
    device const uint *kv_pos_offset_ptr,
    uint out_bf16,
    uint kv_capacity,
    uint2 tgpig,
    uint2 tpitg,
    uint2 tptg_vec,
    threadgroup float *Oacc,
    threadgroup float *Ks,
    threadgroup float *Vs,
    threadgroup float *scores)
{
    constexpr uint D_TILE = FA2_D_TILE;
    const bool bf16_out = HAS_OUT_BF16 && out_bf16 != 0u;

    // Tkv is mutable device state. Never trust it past the jointly-backed K/V
    // capacity supplied by the host, even during an ICB replay.
    const uint Tkv = min(*Tkv_ptr, kv_capacity);
    const ulong q_pos_offset = (ulong)(*q_pos_offset_ptr);
    const ulong kv_pos_offset = (ulong)(*kv_pos_offset_ptr);

    const uint lid = tpitg.x;
    const uint tptg = tptg_vec.x;
    const uint q_block = tgpig.x;
    const uint bh = tgpig.y;
    const uint h = bh % H;
    const uint b = bh / H;
    const uint group = max(H / Hkv, 1u);
    const uint hkv = h / group;
    const ulong kv_pos_stride = (ulong)Hkv * HEAD_DIM;
    const ulong kv_head_base =
        (ulong)b * kv_capacity * kv_pos_stride + (ulong)hkv * HEAD_DIM;
    const ulong q_pos_stride = (ulong)H * HEAD_DIM;
    const ulong q_head_base =
        (ulong)b * Tq * q_pos_stride + (ulong)h * HEAD_DIM;

    const uint t_q0 = q_block * BR;
    if (t_q0 >= Tq) return;

    const uint live_rows = min(BR, Tq - t_q0);
    const bool row_valid = lid < live_rows;
    const uint t_q = row_valid ? t_q0 + lid : t_q0;
    const ulong q_row_base = q_head_base + (ulong)t_q * q_pos_stride;

    // Empty history is a valid state. Match the rows/decode kernels by
    // overwriting every live output row rather than exposing recycled bytes.
    if (Tkv == 0) {
        if (row_valid) {
            if (bf16_out) {
                device bfloat *Ob = (device bfloat *)O;
                for (uint d = 0; d < HEAD_DIM; ++d) {
                    Ob[q_row_base + d] = bfloat(0.0f);
                }
            } else {
                for (uint d = 0; d < HEAD_DIM; ++d) {
                    O[q_row_base + d] = 0.0f;
                }
            }
        }
        return;
    }

    if (lid < BR) {
        for (uint d = 0; d < HEAD_DIM; d++) {
            Oacc[lid * HEAD_DIM + d] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -FLT_MAX, not -INFINITY: kernels compile with fast math, which may
    // assume no value is infinite. `l_i > 0` is the "has seen a key" flag
    // (the block maximum contributes exp(0) = 1, and l_i never drops below 1
    // after that), and a masked score's weight is zeroed by recomputing the
    // mask, never by comparing the score against a sentinel.
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    const ulong q_abs = q_pos_offset + (ulong)t_q;
    // Live keys for this row are k_lo <= k_abs <= k_hi. Global: [0, q_abs].
    ulong k_lo = 0ul;
    // Union window over the BR tile (uniform for all TG lanes, so the early
    // `continue` below is safe). Sliding only.
    ulong k_lo_blk = 0ul;
    ulong k_hi_blk = 0ul;
    if (SLIDING) {
        const ulong window_back = (window == 0u) ? 0ul : (ulong)window - 1ul;
        k_lo = (window == 0u || q_abs < window_back) ? 0ul : q_abs - window_back;
        const ulong q_abs_lo = q_pos_offset + (ulong)t_q0;
        const ulong q_abs_hi = q_pos_offset + (ulong)t_q0 + live_rows - 1ul;
        k_lo_blk = (window == 0u || q_abs_lo < window_back)
            ? 0ul
            : q_abs_lo - window_back;
        k_hi_blk = q_abs_hi;
    }
    const ulong k_hi = q_abs;
    const uint n_k_blocks = Tkv / BC + ((Tkv % BC) != 0u ? 1u : 0u);

    for (uint kb = 0; kb < n_k_blocks; ++kb) {
        const uint t_k0 = kb * BC;
        const uint n_k = min(BC, Tkv - t_k0);
        if (SLIDING) {
            // Skip K blocks entirely outside the sliding window (decode / long ctx).
            const ulong block_lo = kv_pos_offset + (ulong)t_k0;
            const ulong block_hi = block_lo + (ulong)n_k - 1ul;
            if (block_hi < k_lo_blk || block_lo > k_hi_blk) {
                continue;
            }
        }

        // Strided, not `if (lid < BR * BC)`. The host dispatches 32 threads
        // per threadgroup while `scores` holds BR*BC entries (64 for the
        // sliding-window kernels), so the guarded form left entries 32..63
        // untouched. Those are query rows 4..7 of every block, and the `+=`
        // below then accumulated into whatever threadgroup memory held from a
        // previous dispatch: plausible numbers, not NaN, so nothing looked
        // wrong. Striding is correct for any relation between `tptg` and
        // BR*BC, which is the property the guarded form quietly depended on.
        for (uint i = lid; i < BR * BC; i += tptg) {
            scores[i] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d0 = 0; d0 < HEAD_DIM; d0 += D_TILE) {
            for (uint i = lid; i < n_k * D_TILE; i += tptg) {
                const uint tk = i / D_TILE;
                const uint d = i % D_TILE;
                const ulong k_off = kv_head_base + (ulong)(t_k0 + tk) * kv_pos_stride;
                Ks[tk * D_TILE + d] = K[k_off + d0 + d];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (row_valid) {
                for (uint tk = 0; tk < n_k; ++tk) {
                    float s = 0.0f;
                    for (uint d = 0; d < D_TILE; ++d) {
                        s += Q[q_row_base + d0 + d] * Ks[tk * D_TILE + d];
                    }
                    scores[lid * BC + tk] += s;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (row_valid) {
            float m_block = -FLT_MAX;
            for (uint tk = 0; tk < n_k; ++tk) {
                const ulong k_abs = kv_pos_offset + (ulong)t_k0 + tk;
                const bool live = k_abs >= k_lo && k_abs <= k_hi;
                const float score = live ? scores[lid * BC + tk] * scale : -FLT_MAX;
                scores[lid * BC + tk] = score;
                m_block = max(m_block, score);
            }
            const float m_new = max(m_i, m_block);
            // A row can reach a block having seen nothing yet with every key
            // in it masked: the sliding block skip admits a block on behalf of
            // another row in the same BR tile (the union window is computed
            // over the whole tile, so a block needed by the last row can be
            // fully masked for the first), and in the global kernel keys can
            // start past a row's position. With an infinite seed that was
            // `exp(-inf - -inf)` = NaN, which propagated through
            // `Oacc *= alpha` and `l_i` and poisoned the row.
            //
            // `l_i == 0` means the accumulator is still zero, so scaling it
            // by zero is exactly right, and it also covers the ordinary
            // first-live-block case.
            const float alpha = (l_i > 0.0f) ? exp(m_i - m_new) : 0.0f;
            float l_block = 0.0f;
            for (uint tk = 0; tk < n_k; ++tk) {
                const ulong k_abs = kv_pos_offset + (ulong)t_k0 + tk;
                const bool live = k_abs >= k_lo && k_abs <= k_hi;
                const float p = live ? exp(scores[lid * BC + tk] - m_new) : 0.0f;
                scores[lid * BC + tk] = p;
                l_block += p;
            }
            for (uint d = 0; d < HEAD_DIM; ++d) {
                Oacc[lid * HEAD_DIM + d] *= alpha;
            }
            l_i = l_i * alpha + l_block;
            m_i = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d0 = 0; d0 < HEAD_DIM; d0 += D_TILE) {
            for (uint i = lid; i < n_k * D_TILE; i += tptg) {
                const uint tk = i / D_TILE;
                const uint d = i % D_TILE;
                const ulong v_off = kv_head_base + (ulong)(t_k0 + tk) * kv_pos_stride;
                Vs[tk * D_TILE + d] = V[v_off + d0 + d];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (row_valid) {
                for (uint tk = 0; tk < n_k; ++tk) {
                    const float p = scores[lid * BC + tk];
                    for (uint d = 0; d < D_TILE; ++d) {
                        Oacc[lid * HEAD_DIM + d0 + d] += p * Vs[tk * D_TILE + d];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (row_valid) {
        const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        if (bf16_out) {
            device bfloat *Ob = (device bfloat *)O;
            for (uint d = 0; d < HEAD_DIM; ++d) {
                Ob[q_row_base + d] = bfloat(Oacc[lid * HEAD_DIM + d] * inv_l);
            }
        } else {
            for (uint d = 0; d < HEAD_DIM; ++d) {
                O[q_row_base + d] = Oacc[lid * HEAD_DIM + d] * inv_l;
            }
        }
    }
}
