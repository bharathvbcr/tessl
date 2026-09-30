// Backward of Qwen3.5's full attention for training: causal, no window,
// grouped KV heads, head dim 256, on the TensorOps matrix units.
//
// The forward (`qwen35_attn_tiled_lse_*`) saves O and each row's log-sum-exp
// of the scaled scores, so P is rebuilt block by block and never stored:
//
//   P   = exp(scale * Q Kᵀ - lse)            (0 above the diagonal)
//   Dr  = rowsum(dO ∘ O)                     qwen35_attn_bwd_dvec_f32
//   dP  = dO Vᵀ
//   dS  = P ∘ (dP - Dr) * scale              (the gradient of Q Kᵀ)
//   dQ  = dS K                               qwen35_attn_bwd_dq_*
//   dK  = dSᵀ Q,  dV = Pᵀ dO                 qwen35_attn_bwd_dk_* / _dv_*
//
// FlashAttention-2's backward, split so nothing is accumulated across
// threadgroups: the dQ kernel owns a block of query rows and walks the keys,
// the dK and dV kernels own a block of key rows of one KV head and walk that
// head's query heads (in head order) and their queries. Every gradient is
// written once by the threadgroup that owns it, with no atomics, so the
// result is the same on every run. dK and dV are separate kernels, each with
// one accumulator, which keeps them at the forward's register footprint.
//
// Layouts are the forward's: Q, O, dO, dQ `[B, T, H, D]`, K, V, dK, dV
// `[B, T, Hkv, D]`, lse and Dr `[B, H, T]`. Query t sees keys 0..=t (training
// positions start at 0 on both sides).
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant uint ATTN_BWD_D = 256;

/// `dvec[(b*H + h)*T + t] = dot(dO[b, t, h, :], O[b, t, h, :])`.
///
/// Grid: x = t, y = b*H + h.
kernel void qwen35_attn_bwd_dvec_f32(
    device const float *O [[buffer(0)]],
    device const float *dO [[buffer(1)]],
    device float *dvec [[buffer(2)]],
    constant uint &T [[buffer(3)]],
    constant uint &H [[buffer(4)]],
    constant uint &BH [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint t = gid.x;
    const uint bh = gid.y;
    if (t >= T || bh >= BH) return;
    const ulong b = bh / H, h = bh % H;
    const ulong row = ((b * T + t) * H + h) * (ulong)ATTN_BWD_D;
    float s = 0.0f;
    for (uint d = 0; d < ATTN_BWD_D; ++d) {
        s += dO[row + d] * O[row + d];
    }
    dvec[(ulong)bh * T + t] = s;
}

/// dQ for query rows `[q0, q0 + BQ)` of head h: walk the key blocks up to
/// the last query, rebuild P and dS per block in threadgroup memory, and
/// accumulate dS K in a cooperative tensor.
template <int D, int BQ, int BK, int NSG>
inline void attn_bwd_dq_body(
    device float *Q, device float *K, device float *V, device float *dO,
    device const float *lse, device const float *dvec, device float *dQ,
    uint T, uint H, uint Hkv, float scale, uint2 tgpig, uint tid,
    threadgroup float *S, threadgroup float *dP, threadgroup float *lse_row,
    threadgroup float *d_row)
{
    constexpr uint THREADS = NSG * 32;
    const uint q0 = tgpig.x * (uint)BQ;
    if (q0 >= T) { return; }              // uniform across the threadgroup
    const uint nq = min((uint)BQ, T - q0);
    const uint bh = tgpig.y;
    const uint h = bh % H;
    const uint b = bh / H;
    const uint hkv = h / max(H / Hkv, 1u);
    const ulong q_row = (ulong)H * D, kv_row = (ulong)Hkv * D;
    const ulong q_base = (ulong)b * T * q_row + (ulong)h * D;
    const ulong kv_base = (ulong)b * T * kv_row + (ulong)hkv * D;
    const uint t_end = q0 + nq;           // keys through the last query

    constexpr auto s_desc = matmul2d_descriptor(
        BQ, BK, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto acc_desc = matmul2d_descriptor(
        BQ, D, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<s_desc, execution_simdgroups<NSG>> s_op;
    matmul2d<acc_desc, execution_simdgroups<NSG>> acc_op;

    auto mQ = tensor(Q + q_base, dextents<int, 2>{D, (int)T}, array<int, 2>{1, (int)q_row});
    auto mdO = tensor(dO + q_base, dextents<int, 2>{D, (int)T}, array<int, 2>{1, (int)q_row});
    auto mK = tensor(K + kv_base, dextents<int, 2>{D, (int)T}, array<int, 2>{1, (int)kv_row});
    auto mV = tensor(V + kv_base, dextents<int, 2>{D, (int)T}, array<int, 2>{1, (int)kv_row});
    auto tQ = mQ.slice(0, (int)q0);
    auto tdO = mdO.slice(0, (int)q0);
    auto tS = tensor(S, dextents<int, 2>{BK, BQ}, array<int, 2>{1, BK});
    auto tdP = tensor(dP, dextents<int, 2>{BK, BQ}, array<int, 2>{1, BK});

    auto dq = acc_op.template get_destination_cooperative_tensor<
        decltype(tS), decltype(mK.slice(0, 0)), float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < dq.get_capacity(); ++i) { dq[i] = 0.0f; }

    if (tid < (uint)BQ) {
        const bool live = tid < nq;
        lse_row[tid] = live ? lse[(ulong)bh * T + q0 + tid] : INFINITY;
        d_row[tid] = live ? dvec[(ulong)bh * T + q0 + tid] : 0.0f;
    }

    for (uint kb = 0; kb < t_end; kb += (uint)BK) {
        auto tK = mK.slice(0, (int)kb);
        auto tV = mV.slice(0, (int)kb);
        auto sT = s_op.template get_destination_cooperative_tensor<
            decltype(tQ), decltype(tK), float>();
        s_op.run(tQ, tK, sT);
        sT.store(tS);
        auto pT = s_op.template get_destination_cooperative_tensor<
            decltype(tdO), decltype(tV), float>();
        s_op.run(tdO, tV, pT);
        pT.store(tdP);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < (uint)(BQ * BK); i += THREADS) {
            const uint r = i / (uint)BK, c = i % (uint)BK;
            const uint t = kb + c;
            const bool live = r < nq && t < t_end && t <= q0 + r;
            const float p = live ? exp(S[i] * scale - lse_row[r]) : 0.0f;
            S[i] = p * (dP[i] - d_row[r]) * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc_op.run(tS, tK, dq);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto mdQ = tensor(dQ + q_base, dextents<int, 2>{D, (int)T}, array<int, 2>{1, (int)q_row});
    dq.store(mdQ.slice(0, (int)q0));
}

/// dK (`DK`) or dV (`!DK`) for key rows `[k0, k0 + BK)` of KV head hkv: walk
/// the group's query heads in head order and, per head, the query blocks
/// from the one holding k0 to the end, rebuilding Pᵀ (and dSᵀ for dK) in
/// threadgroup memory; accumulate Pᵀ dO or dSᵀ Q.
template <int D, int BQ, int BK, int NSG, bool DK>
inline void attn_bwd_dkv_body(
    device float *Q, device float *K, device float *V, device float *dO,
    device const float *lse, device const float *dvec, device float *dKV,
    uint T, uint H, uint Hkv, float scale, uint2 tgpig, uint tid,
    threadgroup float *St, threadgroup float *dPt, threadgroup float *lse_col,
    threadgroup float *d_col)
{
    constexpr uint THREADS = NSG * 32;
    const uint k0 = tgpig.x * (uint)BK;
    if (k0 >= T) { return; }              // uniform across the threadgroup
    const uint nk = min((uint)BK, T - k0);
    const uint bkv = tgpig.y;
    const uint hkv = bkv % Hkv;
    const uint b = bkv / Hkv;
    const uint group = max(H / Hkv, 1u);
    const ulong q_row = (ulong)H * D, kv_row = (ulong)Hkv * D;
    const ulong kv_base = (ulong)b * T * kv_row + (ulong)hkv * D;

    constexpr auto s_desc = matmul2d_descriptor(
        BK, BQ, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto acc_desc = matmul2d_descriptor(
        BK, D, BQ, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<s_desc, execution_simdgroups<NSG>> s_op;
    matmul2d<acc_desc, execution_simdgroups<NSG>> acc_op;

    auto mK = tensor(K + kv_base, dextents<int, 2>{D, (int)T}, array<int, 2>{1, (int)kv_row});
    auto mV = tensor(V + kv_base, dextents<int, 2>{D, (int)T}, array<int, 2>{1, (int)kv_row});
    auto tK = mK.slice(0, (int)k0);
    auto tV = mV.slice(0, (int)k0);
    auto tSt = tensor(St, dextents<int, 2>{BQ, BK}, array<int, 2>{1, BQ});
    auto tdPt = tensor(dPt, dextents<int, 2>{BQ, BK}, array<int, 2>{1, BQ});

    auto acc = acc_op.template get_destination_cooperative_tensor<
        decltype(tSt), decltype(mK.slice(0, 0)), float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) { acc[i] = 0.0f; }

    for (uint g = 0; g < group; ++g) {
        const uint h = hkv * group + g;
        const uint bh = b * H + h;
        const ulong q_base = (ulong)b * T * q_row + (ulong)h * D;
        auto mQ = tensor(Q + q_base, dextents<int, 2>{D, (int)T}, array<int, 2>{1, (int)q_row});
        auto mdO = tensor(dO + q_base, dextents<int, 2>{D, (int)T}, array<int, 2>{1, (int)q_row});
        for (uint qb = (k0 / (uint)BQ) * (uint)BQ; qb < T; qb += (uint)BQ) {
            const uint nq = min((uint)BQ, T - qb);
            if (tid < (uint)BQ) {
                const bool live = tid < nq;
                lse_col[tid] = live ? lse[(ulong)bh * T + qb + tid] : INFINITY;
                d_col[tid] = live ? dvec[(ulong)bh * T + qb + tid] : 0.0f;
            }
            auto tQ = mQ.slice(0, (int)qb);
            auto tdO = mdO.slice(0, (int)qb);
            auto sT = s_op.template get_destination_cooperative_tensor<
                decltype(tK), decltype(tQ), float>();
            s_op.run(tK, tQ, sT);
            sT.store(tSt);
            if (DK) {
                auto pT = s_op.template get_destination_cooperative_tensor<
                    decltype(tV), decltype(tdO), float>();
                s_op.run(tV, tdO, pT);
                pT.store(tdPt);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < (uint)(BK * BQ); i += THREADS) {
                const uint c = i / (uint)BQ, r = i % (uint)BQ;
                const bool live = c < nk && r < nq && k0 + c <= qb + r;
                const float p = live ? exp(St[i] * scale - lse_col[r]) : 0.0f;
                St[i] = DK ? p * (dPt[i] - d_col[r]) * scale : p;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (DK) {
                acc_op.run(tSt, tQ, acc);
            } else {
                acc_op.run(tSt, tdO, acc);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    auto mOut = tensor(dKV + kv_base, dextents<int, 2>{D, (int)T}, array<int, 2>{1, (int)kv_row});
    acc.store(mOut.slice(0, (int)k0));
}

/// Buffer slots shared by the three block kernels. The name spells the
/// geometry, and the host builds its grid from the same numbers.
#define ATTN_BWD_ARGS                                                         \
    device const float *Q [[buffer(0)]],                                      \
    device const float *K [[buffer(1)]],                                      \
    device const float *V [[buffer(2)]],                                      \
    device const float *dO [[buffer(3)]],                                     \
    device const float *lse [[buffer(4)]],                                    \
    device const float *dvec [[buffer(5)]],                                   \
    device float *out [[buffer(6)]],                                          \
    constant uint &T [[buffer(7)]],                                           \
    constant uint &H [[buffer(8)]],                                           \
    constant uint &Hkv [[buffer(9)]],                                         \
    constant float &scale [[buffer(10)]],                                     \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint tid [[thread_index_in_threadgroup]]

/* MPP's type matching rejects const element types; the kernels never write
   Q, K, V or dO. */
#define ATTN_BWD_INPUTS                                                       \
    const_cast<device float *>(Q), const_cast<device float *>(K),             \
    const_cast<device float *>(V), const_cast<device float *>(dO), lse, dvec, out

#define ATTN_BWD_DQ_KERNEL(NAME, BQ, BK, NSG)                                 \
kernel void NAME(ATTN_BWD_ARGS)                                               \
{                                                                             \
    threadgroup float S[(BQ) * (BK)];                                         \
    threadgroup float dP[(BQ) * (BK)];                                        \
    threadgroup float r0[BQ];                                                 \
    threadgroup float r1[BQ];                                                 \
    attn_bwd_dq_body<ATTN_BWD_D, BQ, BK, NSG>(ATTN_BWD_INPUTS, T, H, Hkv,     \
        scale, tgpig, tid, S, dP, r0, r1);                                    \
}

#define ATTN_BWD_DKV_KERNEL(NAME, BQ, BK, NSG, DK)                            \
kernel void NAME(ATTN_BWD_ARGS)                                               \
{                                                                             \
    threadgroup float St[(BQ) * (BK)];                                        \
    threadgroup float dPt[(BQ) * (BK)];                                       \
    threadgroup float c0[BQ];                                                 \
    threadgroup float c1[BQ];                                                 \
    attn_bwd_dkv_body<ATTN_BWD_D, BQ, BK, NSG, DK>(ATTN_BWD_INPUTS, T, H,     \
        Hkv, scale, tgpig, tid, St, dPt, c0, c1);                             \
}

ATTN_BWD_DQ_KERNEL(qwen35_attn_bwd_dq_h256_q32_k32_sg4, 32, 32, 4)
ATTN_BWD_DKV_KERNEL(qwen35_attn_bwd_dk_h256_q32_k32_sg4, 32, 32, 4, true)
ATTN_BWD_DKV_KERNEL(qwen35_attn_bwd_dv_h256_q32_k32_sg4, 32, 32, 4, false)
