// FlashAttention-2 on the TensorOps matrix units: the block body shared by
// Qwen3.5's prefill and training forward (`qwen35_attn_tiled.metal`) and
// EmbeddingGemma 2's encoder attention (`encoder_attn.metal`). The design,
// measurements and masking contracts are written up at the top of
// `qwen35_attn_tiled.metal`.
//
// Included, never compiled on its own; `build.rs` tracks `.h` changes.
#pragma once

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

template <int D, int BQ, int BK, int NSG, bool ENCODER>
inline void attn_tiled_body(
    device float *Q, device float *K, device float *V, device float *O,
    device float *lse,
    uint Tq, uint Tkv, uint H, uint Hkv, float scale, ulong q_off,
    ulong kv_off, uint out_bf16, uint kv_capacity, uint window, uint2 tgpig,
    uint tid, threadgroup float *S, threadgroup float *m_row,
    threadgroup float *l_row, threadgroup float *a_row)
{
    constexpr int THREADS = NSG * 32;
    constexpr int TPR = THREADS / BQ;      // threads per query row, softmax pass
    constexpr int CPT = BK / TPR;          // columns per thread
    static_assert(THREADS % BQ == 0 && BK % TPR == 0, "softmax mapping");
    static_assert(TPR <= 32 && (TPR & (TPR - 1)) == 0, "row threads in one simdgroup");

    const uint q0 = tgpig.x * (uint)BQ;
    if (q0 >= Tq) { return; }              // uniform across the threadgroup
    const uint nq = min((uint)BQ, Tq - q0);
    const uint bh = tgpig.y;
    const uint h = bh % H;
    const uint b = bh / H;
    const uint group = max(H / Hkv, 1u);
    const uint hkv = h / group;

    const ulong q_row = (ulong)H * (ulong)D;
    const ulong kv_row = (ulong)Hkv * (ulong)D;
    const ulong q_base = (ulong)b * Tq * q_row + (ulong)h * D;
    const ulong kv_base = (ulong)b * kv_capacity * kv_row + (ulong)hkv * D;

    // Keys this threadgroup can see at all. Causal: through its last query's
    // position. Encoder (`Tkv` is the row's live length, offsets are 0): its
    // rows' windows, and nothing at all when every row is padding.
    const ulong q_hi = q_off + q0 + nq - 1u;
    uint t_lo = 0u;
    uint t_end;
    if (ENCODER) {
        t_lo = (window == 0u || q0 < window) ? 0u : q0 - window;
        t_end = (q0 >= Tkv) ? 0u
            : ((window == 0u) ? Tkv : (uint)min((ulong)Tkv, q_hi + window + 1ul));
    } else {
        t_end = (q_hi >= kv_off) ? (uint)min((ulong)Tkv, q_hi - kv_off + 1ul) : 0u;
    }

    constexpr auto qk_desc = matmul2d_descriptor(
        BQ, BK, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc = matmul2d_descriptor(
        BQ, D, BK, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<NSG>> qk_op;
    matmul2d<pv_desc, execution_simdgroups<NSG>> pv_op;

    auto mQ = tensor(Q + q_base, dextents<int, 2>{D, (int)Tq},
                     array<int, 2>{1, (int)q_row});
    auto tQ = mQ.slice(0, (int)q0);
    auto mK = tensor(K + kv_base, dextents<int, 2>{D, (int)Tkv},
                     array<int, 2>{1, (int)kv_row});
    auto mV = tensor(V + kv_base, dextents<int, 2>{D, (int)Tkv},
                     array<int, 2>{1, (int)kv_row});
    auto tP = tensor(S, dextents<int, 2>{BK, BQ}, array<int, 2>{1, BK});

    auto oT = pv_op.template get_destination_cooperative_tensor<
        decltype(tP), decltype(mV.slice(0, 0)), float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < oT.get_capacity(); ++i) { oT[i] = 0.0f; }

    // -FLT_MAX, not -INFINITY: kernels compile with fast math, which may
    // assume no value is infinite. `l_row > 0` is a row's "has seen a key"
    // flag (the block maximum contributes exp(0) = 1, and l never drops below
    // 1 after that), and a masked score's weight is zeroed by its mask, never
    // by comparing the score against a sentinel.
    if (tid < (uint)BQ) {
        m_row[tid] = -FLT_MAX;
        l_row[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint r = tid / (uint)TPR;        // this thread's row in the softmax pass
    const uint part = tid % (uint)TPR;
    const ulong q_abs = q_off + q0 + r;
    const bool row_live = r < nq;

    for (uint kb = t_lo; kb < t_end; kb += (uint)BK) {
        auto tK = mK.slice(0, (int)kb);
        auto sT = qk_op.template get_destination_cooperative_tensor<
            decltype(tQ), decltype(tK), float>();
        qk_op.run(tQ, tK, sT);
        sT.store(tP);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax for row r over this block's columns. The TPR threads
        // of a row are adjacent lanes of one simdgroup, so the row max and sum
        // are xor-butterflies over them.
        float s_loc[CPT];
        bool live[CPT];
        float mx = -FLT_MAX;
        for (int j = 0; j < CPT; ++j) {
            const uint c = part * (uint)CPT + (uint)j;
            const uint t = kb + c;
            if (ENCODER) {
                live[j] = row_live && t < t_end && q_abs < (ulong)Tkv
                    && (window == 0u
                        || ((ulong)t + window >= q_abs && (ulong)t <= q_abs + window));
            } else {
                live[j] = row_live && t < t_end && kv_off + (ulong)t <= q_abs;
            }
            s_loc[j] = live[j] ? S[r * (uint)BK + c] * scale : -FLT_MAX;
            mx = max(mx, s_loc[j]);
        }
        for (uint off = (uint)TPR / 2u; off > 0u; off >>= 1) {
            mx = max(mx, simd_shuffle_xor(mx, (ushort)off));
        }
        const float m_old = m_row[r];
        const float m_new = max(m_old, mx);
        // A row that has seen nothing has a zero accumulator, so its rescale
        // is exactly zero.
        const float alpha = (l_row[r] > 0.0f) ? exp(m_old - m_new) : 0.0f;
        float sum = 0.0f;
        for (int j = 0; j < CPT; ++j) {
            const float p = live[j] ? exp(s_loc[j] - m_new) : 0.0f;
            S[r * (uint)BK + part * (uint)CPT + (uint)j] = p;
            sum += p;
        }
        for (uint off = (uint)TPR / 2u; off > 0u; off >>= 1) {
            sum += simd_shuffle_xor(sum, (ushort)off);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (part == 0u) {
            l_row[r] = l_row[r] * alpha + sum;
            m_row[r] = m_new;
            a_row[r] = alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < oT.get_capacity(); ++i) {
            if (oT.is_valid_element(i)) {
                const auto idx = oT.get_multidimensional_index(i);
                oT[i] *= a_row[idx[1]];
            }
        }
        auto tV = mV.slice(0, (int)kb);
        pv_op.run(tP, tV, oT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < oT.get_capacity(); ++i) {
        if (oT.is_valid_element(i)) {
            const auto idx = oT.get_multidimensional_index(i);
            const float l = l_row[idx[1]];
            oT[i] *= (l > 0.0f) ? (1.0f / l) : 0.0f;
        }
    }
    // The training forward also keeps each row's log-sum-exp of the scaled
    // scores, `[B, H, Tq]`, for the backward to rebuild P from. A row with
    // nothing unmasked gets FLT_MAX, the finite stand-in for +inf, which the
    // backward treats as "no probability anywhere in this row".
    if (lse != nullptr && tid < nq) {
        const float l = l_row[tid];
        lse[(ulong)bh * Tq + q0 + tid] = (l > 0.0f) ? m_row[tid] + precise::log(l) : FLT_MAX;
    }
    if (out_bf16 != 0u) {
        auto mO = tensor((device bfloat *)O + q_base, dextents<int, 2>{D, (int)Tq},
                         array<int, 2>{1, (int)q_row});
        auto tO = mO.slice(0, (int)q0);
        auto oB = pv_op.template get_destination_cooperative_tensor<
            decltype(tP), decltype(mV.slice(0, 0)), bfloat>();
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < oB.get_capacity(); ++i) {
            if (oT.is_valid_element(i)) { oB[i] = bfloat(oT[i]); }
        }
        oB.store(tO);
    } else {
        auto mO = tensor(O + q_base, dextents<int, 2>{D, (int)Tq},
                         array<int, 2>{1, (int)q_row});
        oT.store(mO.slice(0, (int)q0));
    }
}

