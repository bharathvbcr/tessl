// Causal global FlashAttention @ head_dim=512 (FA-2 tiled) + GQA.
// Q: [B,Tq,H,D], K/V: [B,Tkv,Hkv,D], O: [B,Tq,H,D]
// Absolute positions: q_abs = q_pos_offset + t_q, k_abs = kv_pos_offset + t_k.
// Shared-KV consumers pass densified / full-length producer K/V buffers.
// scale = 1.0 after QK-Norm.
// out_bf16=1: write O as bfloat (half-width act scratch for o_proj GEMV).
#include <metal_stdlib>
using namespace metal;

constant uint HEAD_DIM = 512;
constant uint BR = 4;
constant uint BC = 4;
constant uint D_TILE = 32;

/// Per-token varying FA scalars from stable device u32s (ICB / encode-once).
kernel void flash_attn_global_h512(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant uint &B [[buffer(4)]],
    constant uint &Tq [[buffer(5)]],
    device const uint *Tkv_ptr [[buffer(6)]],
    constant uint &H [[buffer(7)]],
    constant uint &Hkv [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    device const uint *q_pos_offset_ptr [[buffer(10)]],
    device const uint *kv_pos_offset_ptr [[buffer(11)]],
    constant uint &out_bf16 [[buffer(12)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tpitg [[thread_position_in_threadgroup]],
    uint2 tptg_vec [[threads_per_threadgroup]])
{
    threadgroup float Oacc[BR * HEAD_DIM];
    threadgroup float Ks[BC * D_TILE];
    threadgroup float Vs[BC * D_TILE];
    threadgroup float scores[BR * BC];

    const uint Tkv = *Tkv_ptr;
    const uint q_pos_offset = *q_pos_offset_ptr;
    const uint kv_pos_offset = *kv_pos_offset_ptr;

    const uint lid = tpitg.x;
    const uint tptg = tptg_vec.x;
    const uint q_block = tgpig.x;
    const uint bh = tgpig.y;
    const uint h = bh % H;
    const uint b = bh / H;
    const uint group = max(H / Hkv, 1u);
    const uint hkv = h / group;

    const uint t_q0 = q_block * BR;
    if (t_q0 >= Tq || Tkv == 0) return;

    const uint t_q = t_q0 + lid;
    const bool row_valid = (lid < BR) && (t_q < Tq);

    if (lid < BR) {
        for (uint d = 0; d < HEAD_DIM; d++) {
            Oacc[lid * HEAD_DIM + d] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_i = -INFINITY;
    float l_i = 0.0f;

    const int q_abs = row_valid ? (int)(q_pos_offset + t_q) : -1;
    const uint n_k_blocks = (Tkv + BC - 1) / BC;

    for (uint kb = 0; kb < n_k_blocks; ++kb) {
        const uint t_k0 = kb * BC;
        const uint n_k = min(BC, Tkv - t_k0);

        // Strided, not `if (lid < BR * BC)`. The host dispatches 32 threads
        // per threadgroup while `scores` holds BR*BC entries — 64 for the
        // sliding-window kernels — so the guarded form left entries 32..63
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
                const uint k_off = ((b * Tkv + (t_k0 + tk)) * Hkv + hkv) * HEAD_DIM;
                Ks[tk * D_TILE + d] = K[k_off + d0 + d];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (row_valid) {
                const uint q_off = ((b * Tq + t_q) * H + h) * HEAD_DIM;
                for (uint tk = 0; tk < n_k; ++tk) {
                    float s = 0.0f;
                    for (uint d = 0; d < D_TILE; ++d) {
                        s += Q[q_off + d0 + d] * Ks[tk * D_TILE + d];
                    }
                    scores[lid * BC + tk] += s;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (row_valid) {
            float m_block = -INFINITY;
            for (uint tk = 0; tk < n_k; ++tk) {
                const int k_abs = (int)(kv_pos_offset + t_k0 + tk);
                float score = scores[lid * BC + tk] * scale;
                if (k_abs > q_abs) {
                    score = -INFINITY;
                }
                scores[lid * BC + tk] = score;
                m_block = max(m_block, score);
            }
            const float m_new = max(m_i, m_block);
            // `exp(m_i - m_new)` is `exp(-inf - -inf)` = `exp(NaN)` = NaN when
            // this row has seen nothing yet and this block is entirely masked
            // for it. That happens whenever the block-level skip admits a block
            // on behalf of another row in the same BR tile — the union window
            // is computed over the whole tile, so a block needed by the last
            // row can be fully masked for the first. The NaN then propagated
            // through `Oacc *= alpha` and `l_i` and poisoned the row.
            //
            // `m_i == -inf` means the accumulator is still zero, so scaling it
            // by zero is exactly right, and it also covers the ordinary
            // first-real-block case where `exp(-inf - finite)` is already 0.
            const float alpha = (m_i == -INFINITY) ? 0.0f : exp(m_i - m_new);
            float l_block = 0.0f;
            for (uint tk = 0; tk < n_k; ++tk) {
                float p = (scores[lid * BC + tk] > -INFINITY)
                    ? exp(scores[lid * BC + tk] - m_new)
                    : 0.0f;
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
                const uint v_off = ((b * Tkv + (t_k0 + tk)) * Hkv + hkv) * HEAD_DIM;
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
        const uint o_off = ((b * Tq + t_q) * H + h) * HEAD_DIM;
        if (out_bf16 != 0u) {
            device bfloat *Ob = (device bfloat *)O;
            for (uint d = 0; d < HEAD_DIM; ++d) {
                Ob[o_off + d] = bfloat(Oacc[lid * HEAD_DIM + d] * inv_l);
            }
        } else {
            for (uint d = 0; d < HEAD_DIM; ++d) {
                O[o_off + d] = Oacc[lid * HEAD_DIM + d] * inv_l;
            }
        }
    }
    (void)B;
}
