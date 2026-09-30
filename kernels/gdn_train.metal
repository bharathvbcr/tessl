// The gated delta rule for training, at transformers' op seam
// (`torch_chunk_gated_delta_rule(q, k, v, g, beta, initial_state,
// use_qk_l2norm_in_kernel=True)`): forward and backward.
//
// Per head, with S [Dk, Dv]:
//
//   q^ = l2norm(q) / sqrt(Dk),  k^ = l2norm(k)       l2norm(x) = x rsqrt(|x|^2 + 1e-6)
//   S^ = exp(g_t) S_{t-1}
//   u  = v_t - S^T k^
//   S_t = S^ + k^ (beta_t u)^T
//   o_t = S_t^T q^
//
// The inference kernels (qwen35_gdn.metal) compute g and beta from the raw
// gate logits and run the chunked form; these take g and beta as given, as
// the training seam does, and are built for the backward: the forward saves
// the state every GDN_TRAIN_CKPT tokens (the only activation it keeps beyond
// its inputs), and the backward walks the chunks in reverse, recomputing each
// chunk's states from its checkpoint into a bounded scratch before running
// the reverse-mode recurrence over it.
//
// Geometry: one threadgroup per (value-column slice of GDN_TRAIN_BV columns,
// batch x head), 128 threads, thread i owning row i of the state slice. Every
// row-local term needs no reduction; the column sums (S^T k^, S^T q^, dS^T k^)
// are simd_sums across the four simdgroups. Contributions that sum over all
// value columns (dq, dk, dg, dbeta) are written per slice and reduced in a
// fixed order by gdn_train_bwd_finish, which also applies the l2norm
// backward: nothing uses atomics, so every result is deterministic.
//
// Layouts (all f32, dense): q, k, dq, dk [B, T, H, 128]; v, o, dv, do
// [B, T, H, Dv]; g, beta, dg, dbeta [B, T, H]; states [B, H, 128, Dv];
// checkpoints [B, H, NC, 128, Dv] with NC = ceil(T / GDN_TRAIN_CKPT).
#include <metal_stdlib>
using namespace metal;

constant uint GDN_TRAIN_DK = 128;
constant uint GDN_TRAIN_BV = 16;
constant uint GDN_TRAIN_CKPT = 64;
constant uint GDN_TRAIN_SG = GDN_TRAIN_DK / 32;
constant float GDN_TRAIN_L2_EPS = 1e-6f;

/// Sum each of `N` per-thread values over the threadgroup's 128 threads; every
/// thread receives all `N` totals. `red` holds `4 * N` floats. Callers
/// alternate between two `red` buffers: the barrier here orders this call's
/// writes after every thread's reads of the other buffer.
template <uint N>
inline void gdn_colsum(thread const float (&p)[N], threadgroup float *red, uint sg, uint lane,
                       thread float (&out)[N])
{
    for (uint n = 0; n < N; ++n) {
        const float s = simd_sum(p[n]);
        if (lane == 0) {
            red[sg * N + n] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint n = 0; n < N; ++n) {
        out[n] = (red[n] + red[N + n]) + (red[2 * N + n] + red[3 * N + n]);
    }
}

/// flags: 1 = `s0` holds the initial state (else zeros); 2 = write the final
/// state to `sfin`.
kernel void gdn_train_fwd(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device const float *g [[buffer(3)]],
    device const float *beta [[buffer(4)]],
    device const float *s0 [[buffer(5)]],
    device float *o [[buffer(6)]],
    device float *sfin [[buffer(7)]],
    device float *ckpt [[buffer(8)]],
    constant uint &B [[buffer(9)]],
    constant uint &T [[buffer(10)]],
    constant uint &H [[buffer(11)]],
    constant uint &Dv [[buffer(12)]],
    constant uint &flags [[buffer(13)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint i [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint BV = GDN_TRAIN_BV;
    threadgroup float red[2][GDN_TRAIN_SG * (BV + 2)];
    const uint bh = tg.y;
    if (bh >= B * H || (tg.x + 1) * BV > Dv) return;  // uniform per threadgroup
    const uint b = bh / H, h = bh % H, j0 = tg.x * BV;
    const uint nc = (T + GDN_TRAIN_CKPT - 1) / GDN_TRAIN_CKPT;
    const float scale = rsqrt((float)GDN_TRAIN_DK);
    const ulong srow = ((ulong)bh * GDN_TRAIN_DK + i) * Dv + j0;

    float S[BV];
    for (uint j = 0; j < BV; ++j) {
        S[j] = (flags & 1u) != 0u ? s0[srow + j] : 0.0f;
    }
    for (uint t = 0; t < T; ++t) {
        if (t % GDN_TRAIN_CKPT == 0) {
            const ulong c = ((ulong)bh * nc + t / GDN_TRAIN_CKPT) * GDN_TRAIN_DK + i;
            for (uint j = 0; j < BV; ++j) {
                ckpt[c * Dv + j0 + j] = S[j];
            }
        }
        const ulong r = ((ulong)b * T + t) * H + h;
        const float qi = q[r * GDN_TRAIN_DK + i];
        const float ki = k[r * GDN_TRAIN_DK + i];
        const float a = precise::exp(g[r]);
        const float bt = beta[r];

        float p[BV + 2], tot[BV + 2];
        for (uint j = 0; j < BV; ++j) {
            S[j] *= a;
            p[j] = S[j] * ki;
        }
        p[BV] = ki * ki;
        p[BV + 1] = qi * qi;
        gdn_colsum<BV + 2>(p, red[0], sg, lane, tot);
        const float rk = precise::rsqrt(tot[BV] + GDN_TRAIN_L2_EPS);
        const float rq = precise::rsqrt(tot[BV + 1] + GDN_TRAIN_L2_EPS);
        const float kh = ki * rk;
        const float qh = qi * rq * scale;

        float po[BV], out[BV];
        for (uint j = 0; j < BV; ++j) {
            const float u = v[r * Dv + j0 + j] - rk * tot[j];
            S[j] += kh * (bt * u);
            po[j] = S[j] * qh;
        }
        gdn_colsum<BV>(po, red[1], sg, lane, out);
        if (i < BV) {
            o[r * Dv + j0 + i] = out[i];
        }
    }
    if ((flags & 2u) != 0u) {
        for (uint j = 0; j < BV; ++j) {
            sfin[srow + j] = S[j];
        }
    }
}

/// The reverse-mode recurrence. `scratch` holds `GDN_TRAIN_CKPT * 128 * BV`
/// floats per threadgroup (the states S_{t-1} of one chunk, row i written and
/// read only by thread i). Writes `dv` in full and, per value slice s, the
/// partial gradients with respect to q^ and k^ (`dq_part`, `dk_part`
/// [NS, B, T, H, 128]) and to g and beta (`dg_part`, `dbeta_part`
/// [NS, B, T, H]); gdn_train_bwd_finish sums them.
///
/// flags: 1 = `s0` given (and `ds0` written); 2 = `dfin` holds the final
/// state's gradient (else zeros).
kernel void gdn_train_bwd(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device const float *g [[buffer(3)]],
    device const float *beta [[buffer(4)]],
    device const float *d_o [[buffer(5)]],
    device const float *dfin [[buffer(6)]],
    device const float *ckpt [[buffer(7)]],
    device float *scratch [[buffer(8)]],
    device float *dv [[buffer(9)]],
    device float *dq_part [[buffer(10)]],
    device float *dk_part [[buffer(11)]],
    device float *dg_part [[buffer(12)]],
    device float *dbeta_part [[buffer(13)]],
    device float *ds0 [[buffer(14)]],
    constant uint &B [[buffer(15)]],
    constant uint &T [[buffer(16)]],
    constant uint &H [[buffer(17)]],
    constant uint &Dv [[buffer(18)]],
    constant uint &flags [[buffer(19)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint i [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint BV = GDN_TRAIN_BV;
    constexpr uint C = GDN_TRAIN_CKPT;
    threadgroup float red[2][GDN_TRAIN_SG * (BV + 2)];
    threadgroup float u_c[C * BV];
    threadgroup float rk_c[C];
    threadgroup float rq_c[C];
    const uint bh = tg.y;
    const uint ns = Dv / BV;
    if (bh >= B * H || (tg.x + 1) * BV > Dv) return;  // uniform per threadgroup
    const uint b = bh / H, h = bh % H, j0 = tg.x * BV;
    const uint nc = (T + C - 1) / C;
    const float scale = rsqrt((float)GDN_TRAIN_DK);
    const ulong srow = ((ulong)bh * GDN_TRAIN_DK + i) * Dv + j0;
    const ulong rows = (ulong)B * T * H;
    device float *mine = scratch + ((ulong)bh * ns + tg.x) * C * GDN_TRAIN_DK * BV;

    float dS[BV];
    for (uint j = 0; j < BV; ++j) {
        dS[j] = (flags & 2u) != 0u ? dfin[srow + j] : 0.0f;
    }
    for (uint cc = nc; cc-- > 0;) {
        const uint t0 = cc * C;
        const uint t1 = min(T, t0 + C);
        // Every thread has finished reading the previous chunk's u_c, rk_c and
        // rq_c (its last reduction below orders them) before they are rewritten.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Recompute the chunk's states from its checkpoint.
        float S[BV];
        const ulong c = ((ulong)bh * nc + cc) * GDN_TRAIN_DK + i;
        for (uint j = 0; j < BV; ++j) {
            S[j] = ckpt[c * Dv + j0 + j];
        }
        for (uint t = t0; t < t1; ++t) {
            const uint lt = t - t0;
            for (uint j = 0; j < BV; ++j) {
                mine[((ulong)lt * GDN_TRAIN_DK + i) * BV + j] = S[j];
            }
            const ulong r = ((ulong)b * T + t) * H + h;
            const float qi = q[r * GDN_TRAIN_DK + i];
            const float ki = k[r * GDN_TRAIN_DK + i];
            const float a = precise::exp(g[r]);
            float p[BV + 2], tot[BV + 2];
            for (uint j = 0; j < BV; ++j) {
                S[j] *= a;
                p[j] = S[j] * ki;
            }
            p[BV] = ki * ki;
            p[BV + 1] = qi * qi;
            gdn_colsum<BV + 2>(p, red[lt & 1u], sg, lane, tot);
            const float rk = precise::rsqrt(tot[BV] + GDN_TRAIN_L2_EPS);
            const float kh = ki * rk;
            const float bt = beta[r];
            for (uint j = 0; j < BV; ++j) {
                const float u = v[r * Dv + j0 + j] - rk * tot[j];
                S[j] += kh * (bt * u);
                if (i == j) {
                    u_c[lt * BV + j] = u;
                }
            }
            if (i == 0) {
                rk_c[lt] = rk;
                rq_c[lt] = precise::rsqrt(tot[BV + 1] + GDN_TRAIN_L2_EPS);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reverse over the chunk.
        for (uint t = t1; t-- > t0;) {
            const uint lt = t - t0;
            const ulong r = ((ulong)b * T + t) * H + h;
            const float a = precise::exp(g[r]);
            const float bt = beta[r];
            const float kh = k[r * GDN_TRAIN_DK + i] * rk_c[lt];
            const float qh = q[r * GDN_TRAIN_DK + i] * rq_c[lt] * scale;

            float Sh[BV], delta[BV], dorow[BV];
            float dqh = 0.0f;
            for (uint j = 0; j < BV; ++j) {
                Sh[j] = a * mine[((ulong)lt * GDN_TRAIN_DK + i) * BV + j];
                delta[j] = bt * u_c[lt * BV + j];
                dorow[j] = d_o[r * Dv + j0 + j];
                dqh += (Sh[j] + kh * delta[j]) * dorow[j];  // S_t = S^ + k^ delta^T
            }
            float p[BV], ddelta[BV];
            for (uint j = 0; j < BV; ++j) {
                dS[j] += qh * dorow[j];
                p[j] = dS[j] * kh;
            }
            gdn_colsum<BV>(p, red[0], sg, lane, ddelta);

            float dkh = 0.0f, dgp[1] = {0.0f}, dgt[1];
            for (uint j = 0; j < BV; ++j) {
                dkh += dS[j] * delta[j] - bt * Sh[j] * ddelta[j];
                const float dSh = dS[j] - bt * kh * ddelta[j];
                dgp[0] += dSh * Sh[j];
                dS[j] = a * dSh;
            }
            gdn_colsum<1>(dgp, red[1], sg, lane, dgt);

            if (i < BV) {
                dv[r * Dv + j0 + i] = bt * ddelta[i];
            }
            const ulong part = (ulong)tg.x * rows + r;
            dq_part[part * GDN_TRAIN_DK + i] = dqh;
            dk_part[part * GDN_TRAIN_DK + i] = dkh;
            if (i == 0) {
                float db = 0.0f;
                for (uint j = 0; j < BV; ++j) {
                    db += ddelta[j] * u_c[lt * BV + j];
                }
                dbeta_part[part] = db;
                dg_part[part] = dgt[0];
            }
        }
    }
    if ((flags & 1u) != 0u) {
        for (uint j = 0; j < BV; ++j) {
            ds0[srow + j] = dS[j];
        }
    }
}

/// Sum the per-slice partials in slice order and apply the l2norm backward:
/// with y = x r, r = rsqrt(|x|^2 + eps), dx = r (dy - y (y . dy)); for q the
/// incoming gradient is also scaled by 1/sqrt(Dk). One threadgroup of 128 per
/// (b, t, h) row.
kernel void gdn_train_bwd_finish(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *dq_part [[buffer(2)]],
    device const float *dk_part [[buffer(3)]],
    device const float *dg_part [[buffer(4)]],
    device const float *dbeta_part [[buffer(5)]],
    device float *dq [[buffer(6)]],
    device float *dk [[buffer(7)]],
    device float *dg [[buffer(8)]],
    device float *dbeta [[buffer(9)]],
    constant uint &rows [[buffer(10)]],
    constant uint &ns [[buffer(11)]],
    uint r [[threadgroup_position_in_grid]],
    uint i [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[2][GDN_TRAIN_SG * 2];
    if (r >= rows) return;  // uniform per threadgroup
    const ulong at = (ulong)r * GDN_TRAIN_DK + i;
    float dqh = 0.0f, dkh = 0.0f;
    for (uint s = 0; s < ns; ++s) {
        dqh += dq_part[(ulong)s * rows * GDN_TRAIN_DK + at];
        dkh += dk_part[(ulong)s * rows * GDN_TRAIN_DK + at];
    }
    const float qi = q[at], ki = k[at];
    float p[2] = {qi * qi, ki * ki}, tot[2];
    gdn_colsum<2>(p, red[0], sg, lane, tot);
    const float rq = precise::rsqrt(tot[0] + GDN_TRAIN_L2_EPS);
    const float rk = precise::rsqrt(tot[1] + GDN_TRAIN_L2_EPS);
    const float yq = qi * rq, yk = ki * rk;
    const float dyq = dqh * rsqrt((float)GDN_TRAIN_DK);
    float p2[2] = {yq * dyq, yk * dkh}, dots[2];
    gdn_colsum<2>(p2, red[1], sg, lane, dots);
    dq[at] = rq * (dyq - yq * dots[0]);
    dk[at] = rk * (dkh - yk * dots[1]);
    if (i == 0) {
        float a = 0.0f, bsum = 0.0f;
        for (uint s = 0; s < ns; ++s) {
            a += dg_part[(ulong)s * rows + r];
            bsum += dbeta_part[(ulong)s * rows + r];
        }
        dg[r] = a;
        dbeta[r] = bsum;
    }
}
