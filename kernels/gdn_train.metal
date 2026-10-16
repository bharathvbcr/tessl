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
// Layouts (all f32): o, do [B, T, H, Dv]; g, beta, dg, dbeta [B, T, H];
// states [B, H, 128, Dv]; checkpoints [B, H, NC, 128, Dv] with
// NC = ceil(T / GDN_TRAIN_CKPT). q, k, v and the gradients dq, dk, dv are
// column windows of a row-major matrix with one row per token (b * T + t):
// element (token, head, d) of q is q[token * ldq + offq + head * 128 + d], of
// v v[token * ldv + offv + head * Dv + d], and so on (the Qwen3.5 conv output
// is [q | k | v] with ld = its width). q and k have Hk key heads and H value
// heads share them, H / Hk to a key head: value head h reads key head
// h / (H / Hk) (transformers' repeat_interleave), and dq, dk are summed over
// each key head's value heads in value-head order. A dense [B, T, H, 128]
// operand is ld = H * 128, off = 0, Hk = H.
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
    constant uint &ldq [[buffer(14)]],
    constant uint &offq [[buffer(15)]],
    constant uint &ldk [[buffer(16)]],
    constant uint &offk [[buffer(17)]],
    constant uint &ldv [[buffer(18)]],
    constant uint &offv [[buffer(19)]],
    constant uint &Hk [[buffer(20)]],
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
    const uint hk = h / (H / Hk);
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
        const ulong tok = (ulong)b * T + t;
        const ulong r = tok * H + h;
        const float qi = q[tok * ldq + offq + hk * GDN_TRAIN_DK + i];
        const float ki = k[tok * ldk + offk + hk * GDN_TRAIN_DK + i];
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
            const float u = v[tok * ldv + offv + (ulong)h * Dv + j0 + j] - rk * tot[j];
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
    constant uint &ldq [[buffer(20)]],
    constant uint &offq [[buffer(21)]],
    constant uint &ldk [[buffer(22)]],
    constant uint &offk [[buffer(23)]],
    constant uint &ldv [[buffer(24)]],
    constant uint &offv [[buffer(25)]],
    constant uint &Hk [[buffer(26)]],
    constant uint &lddv [[buffer(27)]],
    constant uint &offdv [[buffer(28)]],
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
    const uint hk = h / (H / Hk);
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
            const ulong tok = (ulong)b * T + t;
            const ulong r = tok * H + h;
            const float qi = q[tok * ldq + offq + hk * GDN_TRAIN_DK + i];
            const float ki = k[tok * ldk + offk + hk * GDN_TRAIN_DK + i];
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
                const float u = v[tok * ldv + offv + (ulong)h * Dv + j0 + j] - rk * tot[j];
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
            const ulong tok = (ulong)b * T + t;
            const ulong r = tok * H + h;
            const float a = precise::exp(g[r]);
            const float bt = beta[r];
            const float kh = k[tok * ldk + offk + hk * GDN_TRAIN_DK + i] * rk_c[lt];
            const float qh = q[tok * ldq + offq + hk * GDN_TRAIN_DK + i] * rq_c[lt] * scale;

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
                dv[tok * lddv + offdv + (ulong)h * Dv + j0 + i] = bt * ddelta[i];
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
/// (token, key head) row: q and k are read through their windows once, and
/// each of the key head's H / Hk value heads (partials, dg, dbeta live per
/// value head, `rows` = B * T * H of them) contributes its dx; they are added
/// in value-head order into the dq and dk windows. The sum goes through
/// threadgroup memory so the compiler cannot fuse the add into the product
/// that made it: the result is the sum of separately rounded gradients.
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
    constant uint &ldq [[buffer(12)]],
    constant uint &offq [[buffer(13)]],
    constant uint &ldk [[buffer(14)]],
    constant uint &offk [[buffer(15)]],
    constant uint &lddq [[buffer(16)]],
    constant uint &offdq [[buffer(17)]],
    constant uint &lddk [[buffer(18)]],
    constant uint &offdk [[buffer(19)]],
    constant uint &toks [[buffer(20)]],
    constant uint &H [[buffer(21)]],
    constant uint &Hk [[buffer(22)]],
    uint r [[threadgroup_position_in_grid]],
    uint i [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[2][GDN_TRAIN_SG * 2];
    threadgroup float stage[2][GDN_TRAIN_DK];
    if (r >= toks * Hk) return;  // uniform per threadgroup
    const uint tok = r / Hk, hk = r % Hk, ratio = H / Hk;
    const ulong qat = (ulong)tok * ldq + offq + (ulong)hk * GDN_TRAIN_DK + i;
    const ulong kat = (ulong)tok * ldk + offk + (ulong)hk * GDN_TRAIN_DK + i;
    const float qi = q[qat], ki = k[kat];
    float p[2] = {qi * qi, ki * ki}, tot[2];
    gdn_colsum<2>(p, red[0], sg, lane, tot);
    const float rq = precise::rsqrt(tot[0] + GDN_TRAIN_L2_EPS);
    const float rk = precise::rsqrt(tot[1] + GDN_TRAIN_L2_EPS);
    const float yq = qi * rq, yk = ki * rk;
    float sq = 0.0f, sk = 0.0f;
    for (uint j = 0; j < ratio; ++j) {
        const ulong vr = (ulong)tok * H + (ulong)hk * ratio + j;
        const ulong at = vr * GDN_TRAIN_DK + i;
        float dqh = 0.0f, dkh = 0.0f;
        for (uint s = 0; s < ns; ++s) {
            dqh += dq_part[(ulong)s * rows * GDN_TRAIN_DK + at];
            dkh += dk_part[(ulong)s * rows * GDN_TRAIN_DK + at];
        }
        const float dyq = dqh * rsqrt((float)GDN_TRAIN_DK);
        float p2[2] = {yq * dyq, yk * dkh}, dots[2];
        // red[0] holds the norms' sums (read above, ordered by this call's
        // barrier for j = 0); successive calls alternate buffers.
        gdn_colsum<2>(p2, red[(j + 1) & 1u], sg, lane, dots);
        stage[0][i] = rq * (dyq - yq * dots[0]);
        stage[1][i] = rk * (dkh - yk * dots[1]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float xq = stage[0][i], xk = stage[1][i];
        sq = j == 0 ? xq : sq + xq;
        sk = j == 0 ? xk : sk + xk;
        if (i == 0) {
            float a = 0.0f, bsum = 0.0f;
            for (uint s = 0; s < ns; ++s) {
                a += dg_part[(ulong)s * rows + vr];
                bsum += dbeta_part[(ulong)s * rows + vr];
            }
            dg[vr] = a;
            dbeta[vr] = bsum;
        }
    }
    dq[(ulong)tok * lddq + offdq + (ulong)hk * GDN_TRAIN_DK + i] = sq;
    dk[(ulong)tok * lddk + offdk + (ulong)hk * GDN_TRAIN_DK + i] = sk;
}

// =============================================================================
// The chunked (WY) form
// =============================================================================
//
// Same operator, same layouts and the same checkpoints as the token-sequential
// kernels above (so a forward of one and a backward of the other compose), but
// the work inside a chunk of C = GDN_TRAIN_CKPT = 64 tokens is matrix products
// on the simdgroup units instead of 64 dependent steps. With G the chunk-local
// cumulative sum of g (G_t = g_0 + ... + g_t), gamma_t = exp(G_t),
// E_ti = exp(G_t - G_i) (<= 1 for i <= t, so nothing overflows), k, q the
// normalized operands (q already scaled by 1/sqrt(Dk)) and S0 the state at the
// chunk's start:
//
//   A_ij = beta_i E_ij (k_i . k_j)           j < i          W = (I + A)^-1
//   R_i  = beta_i (v_i - gamma_i S0^T k_i)                  D = W R
//   M_ti = E_ti (q_t . k_i)                  i <= t
//   o_t  = gamma_t S0^T q_t + sum_{i<=t} M_ti D_i
//   S_C  = gamma_C S0 + sum_i E_Ci k_i D_i^T                 (C = the chunk's last row)
//
// which is `torch_chunk_gated_delta_rule` and kernels/qwen35_gdn.metal's
// inference scan, with g and beta taken as given. The backward is that algebra
// transposed. With dO and dS_C (the gradient of the chunk's final state):
//
//   dD_i  = sum_{t>=i} M_ti dO_t + E_Ci dS_C^T k_i          dR = W^T dD
//   dv_i  = beta_i dR_i
//   dS0   = gamma_C dS_C + sum_t gamma_t q_t dO_t^T - sum_i gamma_i beta_i k_i dR_i^T
//   dM_ti = dO_t . D_i,   dA_ij = -(dR_i . D_j)
//   dq_t  = gamma_t S0 dO_t + sum_{i<=t} E_ti dM_ti k_i
//   dk_i  = sum_{t>=i} E_ti dM_ti q_t + E_Ci dS_C D_i - gamma_i beta_i S0 dR_i
//           + sum_j dA_ij beta_i E_ij k_j + sum_{r>i} dA_ri beta_r E_ri k_r
//   dG_t  = sum_i M_ti dM_ti - sum_r M_rt dM_rt + (the same for A, with dA)
//           + gamma_t (S0^T q_t) . dO_t - gamma_t beta_t (S0^T k_t) . dR_t
//           + (at the last row) gamma_C <S0, dS_C> + sum_i E_Ci (dS_C^T k_i) . D_i
//           - E_Ct (dS_C^T k_t) . D_t
//   dbeta_i = dR_i . (v_i - gamma_i S0^T k_i) + sum_j dA_ij E_ij (k_i . k_j)
//   dg_s  = sum_{t>=s, same chunk} dG_t
//
// Only the state S0 -> S_C (and its gradient, backwards) is sequential across
// chunks, and it is column-local: each (batch x head, 16-column slice of Dv)
// threadgroup owns its columns of S and dS. Everything else is independent per
// chunk:
//
//   gdn_train_c_norm    q, k l2-normalized ONCE per (token, key head), not once
//                       per value slice and not once per value head
//   gdn_train_c_prep    per value head and chunk: G, beta, W and M (the solve)
//   gdn_train_c_fwd     the sequential pass over chunks; writes o, the final
//                       state and the same checkpoints the sequential kernel does
//   gdn_train_c_bwd_a   the sequential reverse pass: per slice, D, dR, dv, dS_C
//                       per chunk and the slice's share of dG and dbeta
//   gdn_train_c_mats    per chunk: the C x C gradients dM, dA that need all of Dv
//   gdn_train_c_gates   per chunk: dG, dbeta -> dg, dbeta
//   gdn_train_c_dqk     per chunk: dq, dk for each value head
//   gdn_train_c_finish  the l2norm backward, summing a key head's value heads
//
// Workspace layout, with Tp = ceil(T / 64) * 64 and every sequence padded to Tp
// with zero k, q, v, dO, g and beta (rows past T are inert, as in
// qwen35_gdn.metal): kh, qh [B, Hk, Tp, 128]; gc, bt [B, H, Tp]; w, aq and the
// C x C gradient matrices [B, H, NC, 64, 64]; dbuf, drbuf, dop [B, H, Tp, Dv].
// No atomics: each element has one writer, so every result is deterministic.
// Nothing here uses device-memory ordering inside a kernel: what a kernel
// reads back across threads goes through threadgroup memory, and what the next
// stage reads is written by an earlier dispatch.

constant uint GDNC_C = GDN_TRAIN_CKPT;
constant uint GDNC_THREADS = 128;
constant uint GDNC_PREP_THREADS = 256;
static_assert(GDN_TRAIN_CKPT == 64, "the chunked kernels are written for 64-token chunks");
static_assert(GDN_TRAIN_BV == 16, "the chunked kernels are written for 16-column slices");
static_assert(GDN_TRAIN_DK == 128, "the chunked kernels are written for a 128-wide key head");

/// acc += op(A) op(B) for the 8x8 output tile (bm, bn), over the 8-wide
/// contraction blocks [k0, k1). `A` is `[M, K]` row-major (or `[K, M]` when
/// `TA`), `B` is `[K, N]` (or `[N, K]` when `TB`), with row strides `lda` and
/// `ldb`. A triangular operand skips its zero tiles through [k0, k1).
template <bool TA, bool TB, typename PA, typename PB>
inline void gdnc_mac(thread simdgroup_float8x8 &acc, PA a, uint lda, PB b, uint ldb, uint bm, uint bn, uint k0,
                     uint k1)
{
    for (uint kk = k0; kk < k1; ++kk) {
        simdgroup_float8x8 ma, mb;
        simdgroup_load(ma, TA ? a + (ulong)kk * 8u * lda + bm * 8u : a + (ulong)bm * 8u * lda + kk * 8u, lda,
                       ulong2(0, 0), TA);
        simdgroup_load(mb, TB ? b + (ulong)bn * 8u * ldb + kk * 8u : b + (ulong)kk * 8u * ldb + bn * 8u, ldb,
                       ulong2(0, 0), TB);
        simdgroup_multiply_accumulate(acc, ma, mb, acc);
    }
}

/// l2norm of q and k, once per (token, key head): kh = k rsqrt(|k|^2 + eps),
/// qh = q rsqrt(|q|^2 + eps) / sqrt(Dk), written head-major and zero-padded
/// past T. One simdgroup per row, 16 rows per simdgroup, 4 simdgroups.
/// Grid: x = chunk, y = batch x key head.
kernel void gdn_train_c_norm(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device float *kh [[buffer(2)]],
    device float *qh [[buffer(3)]],
    constant uint &B [[buffer(4)]],
    constant uint &T [[buffer(5)]],
    constant uint &Hk [[buffer(6)]],
    constant uint &ldq [[buffer(7)]],
    constant uint &offq [[buffer(8)]],
    constant uint &ldk [[buffer(9)]],
    constant uint &offk [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint i [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint chunk = tg.x, kbh = tg.y;
    if (kbh >= B * Hk) return;  // uniform per threadgroup
    const uint b = kbh / Hk, hk = kbh % Hk;
    const uint nc = (T + GDNC_C - 1) / GDNC_C;
    const ulong tp = (ulong)nc * GDNC_C;
    const float scale = rsqrt((float)GDN_TRAIN_DK);
    for (uint rr = 0; rr < GDNC_C / 4; ++rr) {
        const uint ri = sg * (GDNC_C / 4) + rr;
        const uint t = chunk * GDNC_C + ri;
        float qv[4], kv[4], p[2] = {0.0f, 0.0f};
        for (uint m = 0; m < 4; ++m) {
            qv[m] = 0.0f;
            kv[m] = 0.0f;
            if (t < T) {
                const ulong tok = (ulong)b * T + t;
                qv[m] = q[tok * ldq + offq + (ulong)hk * GDN_TRAIN_DK + lane + 32 * m];
                kv[m] = k[tok * ldk + offk + (ulong)hk * GDN_TRAIN_DK + lane + 32 * m];
            }
            p[0] += qv[m] * qv[m];
            p[1] += kv[m] * kv[m];
        }
        // `t` is uniform across the simdgroup: every lane reaches both sums.
        const float rq = precise::rsqrt(simd_sum(p[0]) + GDN_TRAIN_L2_EPS);
        const float rk = precise::rsqrt(simd_sum(p[1]) + GDN_TRAIN_L2_EPS);
        const ulong at = ((ulong)kbh * tp + t) * GDN_TRAIN_DK + lane;
        for (uint m = 0; m < 4; ++m) {
            qh[at + 32 * m] = qv[m] * rq * scale;
            kh[at + 32 * m] = kv[m] * rk;
        }
    }
}

/// Row stride of the prep kernel's C x C threadgroup block: odd, so the solve's
/// column walks touch distinct banks.
constant uint GDNC_BLK_LD = GDNC_C + 1;

/// Per (batch x value head, chunk): the chunk-local cumulative decay, beta, the
/// solve W = (I + A)^-1 (unit lower triangular) and M = lower_incl((q_i . k_j)
/// E_ij). 256 threads = 8 simdgroups. Grid: x = chunk, y = batch x value head.
kernel void gdn_train_c_prep(
    device const float *kh [[buffer(0)]],
    device const float *qh [[buffer(1)]],
    device const float *g [[buffer(2)]],
    device const float *beta [[buffer(3)]],
    device float *gc [[buffer(4)]],
    device float *bt [[buffer(5)]],
    device float *wm [[buffer(6)]],
    device float *aq [[buffer(7)]],
    constant uint &B [[buffer(8)]],
    constant uint &T [[buffer(9)]],
    constant uint &H [[buffer(10)]],
    constant uint &Hk [[buffer(11)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float blk[GDNC_C * GDNC_BLK_LD];
    threadgroup float G[GDNC_C];
    threadgroup float bs[GDNC_C];
    const uint chunk = tg.x, bh = tg.y;
    if (bh >= B * H) return;  // uniform per threadgroup
    const uint b = bh / H, h = bh % H, hk = h / (H / Hk);
    const uint nc = (T + GDNC_C - 1) / GDNC_C;
    const ulong tp = (ulong)nc * GDNC_C;
    const uint t0 = chunk * GDNC_C;
    if (lid < GDNC_C) {
        const uint t = t0 + lid;
        const ulong r = ((ulong)b * T + t) * H + h;
        G[lid] = t < T ? g[r] : 0.0f;
        bs[lid] = t < T ? beta[r] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        // Serial on purpose: 63 adds, in torch's cumsum order.
        float acc = 0.0f;
        for (uint i = 0; i < GDNC_C; ++i) {
            acc += G[i];
            G[i] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong row0 = (ulong)bh * tp + t0;
    if (lid < GDNC_C) {
        gc[row0 + lid] = G[lid];
        bt[row0 + lid] = bs[lid];
    }
    device const float *wk = kh + ((ulong)(b * Hk + hk) * tp + t0) * GDN_TRAIN_DK;
    device const float *wq = qh + ((ulong)(b * Hk + hk) * tp + t0) * GDN_TRAIN_DK;
    const ulong blk_base = ((ulong)bh * nc + chunk) * (GDNC_C * GDNC_C);
    const uint rb = sg;  // this simdgroup's 8-row block of the C x C output

    // M = lower_incl(Q K^T * E). Constant trip counts with a uniform predicate,
    // so the accumulators stay in registers.
    {
        simdgroup_float8x8 acc[8];
        for (uint cb = 0; cb < 8; ++cb) {
            acc[cb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        for (uint cb = 0; cb < 8; ++cb) {
            if (cb <= rb) {
                gdnc_mac<false, true>(acc[cb], wq, GDN_TRAIN_DK, wk, GDN_TRAIN_DK, rb, cb, 0, GDN_TRAIN_DK / 8);
            }
        }
        for (uint cb = 0; cb < 8; ++cb) {
            if (cb <= rb) {
                simdgroup_store(acc[cb], blk + rb * 8 * GDNC_BLK_LD + cb * 8, GDNC_BLK_LD);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lid; idx < GDNC_C * GDNC_C; idx += GDNC_PREP_THREADS) {
        const uint i = idx / GDNC_C, j = idx % GDNC_C;
        // Tiles above the diagonal were never stored; the mask never reads them.
        aq[blk_base + idx] = j <= i ? blk[i * GDNC_BLK_LD + j] * precise::exp(G[i] - G[j]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // A, stored transposed in the upper triangle: K K^T is symmetric, so the
    // upper tiles hold (k_i . k_j) at (j, i) for j < i; scaled in place there
    // the upper triangle is A^T and the lower one is free for W.
    {
        simdgroup_float8x8 acc[8];
        for (uint cb = 0; cb < 8; ++cb) {
            acc[cb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        for (uint cb = 0; cb < 8; ++cb) {
            if (cb >= rb) {
                gdnc_mac<false, true>(acc[cb], wk, GDN_TRAIN_DK, wk, GDN_TRAIN_DK, rb, cb, 0, GDN_TRAIN_DK / 8);
            }
        }
        for (uint cb = 0; cb < 8; ++cb) {
            if (cb >= rb) {
                simdgroup_store(acc[cb], blk + rb * 8 * GDNC_BLK_LD + cb * 8, GDNC_BLK_LD);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lid; idx < GDNC_C * GDNC_C; idx += GDNC_PREP_THREADS) {
        const uint j = idx / GDNC_C;  // row of the stored element
        const uint i = idx % GDNC_C;  // column
        if (j < i) {
            blk[j * GDNC_BLK_LD + i] = bs[i] * precise::exp(G[i] - G[j]) * blk[j * GDNC_BLK_LD + i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // W = (I + A)^-1 by forward substitution, column-parallel: column n solves
    // (I + A) w = e_n independently of every other column. A quad of lanes owns
    // a column and splits each step's dot product four ways; W[i][n] is written
    // by the lane that reads it back.
    {
        const uint n = lid >> 2, q = lid & 3;
        for (uint i = 1; i < GDNC_C; ++i) {
            float s = 0.0f;
            if (i > n) {
                for (uint j = n + q; j < i; j += 4) {
                    const float wj = j == n ? 1.0f : blk[j * GDNC_BLK_LD + n];
                    s += blk[j * GDNC_BLK_LD + i] * wj;  // A[i][j], from the upper triangle
                }
            }
            // Uniform across the simdgroup: every lane runs every `i`.
            s += simd_shuffle_xor(s, 1);
            s += simd_shuffle_xor(s, 2);
            if (i > n && ((i - n) & 3) == q) {
                blk[i * GDNC_BLK_LD + n] = -s;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lid; idx < GDNC_C * GDNC_C; idx += GDNC_PREP_THREADS) {
        const uint i = idx / GDNC_C, j = idx % GDNC_C;
        wm[blk_base + idx] = i > j ? blk[i * GDNC_BLK_LD + j] : (i == j ? 1.0f : 0.0f);
    }
}

/// The forward's sequential pass: per (batch x value head, 16-column slice of
/// Dv), the state slice lives in threadgroup memory and each chunk is
/// X = K S, R = beta (V - exp(G) X), D = W R, o = exp(G) Q S + M D,
/// S = exp(G_C) S + K^T (exp(G_C - G) D). Saves S at every chunk start into
/// `ckpt` ([B, H, NC, 128, Dv], as gdn_train_fwd does). flags: 1 = `s0` holds
/// the initial state; 2 = write the final state to `sfin`.
/// 128 threads = 4 simdgroups. Grid: x = Dv / 16, y = batch x value head.
kernel void gdn_train_c_fwd(
    device const float *v [[buffer(0)]],
    device const float *kh [[buffer(1)]],
    device const float *qh [[buffer(2)]],
    device const float *gc [[buffer(3)]],
    device const float *bt [[buffer(4)]],
    device const float *wm [[buffer(5)]],
    device const float *aq [[buffer(6)]],
    device const float *s0 [[buffer(7)]],
    device float *o [[buffer(8)]],
    device float *sfin [[buffer(9)]],
    device float *ckpt [[buffer(10)]],
    constant uint &B [[buffer(11)]],
    constant uint &T [[buffer(12)]],
    constant uint &H [[buffer(13)]],
    constant uint &Dv [[buffer(14)]],
    constant uint &flags [[buffer(15)]],
    constant uint &ldv [[buffer(16)]],
    constant uint &offv [[buffer(17)]],
    constant uint &Hk [[buffer(18)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint i [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint BV = GDN_TRAIN_BV, C = GDNC_C, DK = GDN_TRAIN_DK;
    threadgroup float S[DK * BV];
    threadgroup float X[C * BV];
    threadgroup float Y[C * BV];
    threadgroup float Dt[C * BV];
    threadgroup float tmp[4 * 64];
    threadgroup float Es[C];
    threadgroup float Bs[C];
    threadgroup float Ws[C];
    const uint bh = tg.y;
    if (bh >= B * H || (tg.x + 1) * BV > Dv) return;  // uniform per threadgroup
    const uint b = bh / H, h = bh % H, j0 = tg.x * BV;
    const uint hk = h / (H / Hk);
    const uint nc = (T + C - 1) / C;
    const ulong tp = (ulong)nc * C;
    const ulong kbh = (ulong)b * Hk + hk;
    const ulong srow = ((ulong)bh * DK + i) * Dv + j0;
    threadgroup float *st = tmp + sg * 64;

    for (uint j = 0; j < BV; ++j) {
        S[i * BV + j] = (flags & 1u) != 0u ? s0[srow + j] : 0.0f;
    }
    for (uint chunk = 0; chunk < nc; ++chunk) {
        const uint t0 = chunk * C;
        const ulong row0 = (ulong)bh * tp + t0;
        device const float *kc = kh + (kbh * tp + t0) * DK;
        device const float *qc = qh + (kbh * tp + t0) * DK;
        device const float *wc = wm + ((ulong)bh * nc + chunk) * (C * C);
        device const float *ac = aq + ((ulong)bh * nc + chunk) * (C * C);
        // Orders the previous chunk's state update (or the initialisation)
        // before this chunk reads S, and its reads of X, Y, Dt before they are rewritten.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (i < C) {
            const float g = gc[row0 + i];
            Es[i] = precise::exp(g);
            Bs[i] = bt[row0 + i];
            Ws[i] = precise::exp(gc[row0 + C - 1] - g);
        }
        {
            const ulong c = ((ulong)bh * nc + chunk) * DK + i;
            for (uint j = 0; j < BV; ++j) {
                ckpt[c * Dv + j0 + j] = S[i * BV + j];
            }
        }
        // X = K S, Y = Q S.
        for (uint job = sg; job < 2 * (C / 8) * (BV / 8); job += GDNC_THREADS / 32) {
            const uint which = job / ((C / 8) * (BV / 8)), r = job % ((C / 8) * (BV / 8));
            const uint bm = r / (BV / 8), bn = r % (BV / 8);
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            gdnc_mac<false, false>(acc, which != 0 ? qc : kc, DK, S, BV, bm, bn, 0, DK / 8);
            simdgroup_store(acc, (which != 0 ? Y : X) + bm * 8 * BV + bn * 8, BV);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // R = beta (V - exp(G) X), in X.
        for (uint idx = i; idx < C * BV; idx += GDNC_THREADS) {
            const uint ri = idx / BV, j = idx % BV, t = t0 + ri;
            const float vv = t < T ? v[((ulong)b * T + t) * ldv + offv + (ulong)h * Dv + j0 + j] : 0.0f;
            X[idx] = Bs[ri] * (vv - Es[ri] * X[idx]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // D = W R (W is unit lower triangular: skip its zero tiles).
        for (uint job = sg; job < (C / 8) * (BV / 8); job += GDNC_THREADS / 32) {
            const uint bm = job / (BV / 8), bn = job % (BV / 8);
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            gdnc_mac<false, false>(acc, wc, C, X, BV, bm, bn, 0, bm + 1);
            simdgroup_store(acc, Dt + bm * 8 * BV + bn * 8, BV);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // o = exp(G) Y + M D, rows past T masked.
        for (uint job = sg; job < (C / 8) * (BV / 8); job += GDNC_THREADS / 32) {
            const uint bm = job / (BV / 8), bn = job % (BV / 8);
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            gdnc_mac<false, false>(acc, ac, C, Dt, BV, bm, bn, 0, bm + 1);
            simdgroup_store(acc, st, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                const uint ri = bm * 8 + e / 8, t = t0 + ri;
                if (t < T) {
                    const ulong r = ((ulong)b * T + t) * H + h;
                    o[r * Dv + j0 + bn * 8 + e % 8] = Es[ri] * Y[ri * BV + bn * 8 + e % 8] + st[e];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = exp(G_C) S + K^T (exp(G_C - G) D): both scalings are elementwise
        // passes, and nothing reads S or D until the barrier after them.
        for (uint idx = i; idx < C * BV; idx += GDNC_THREADS) {
            Dt[idx] *= Ws[idx / BV];
        }
        for (uint j = 0; j < BV; ++j) {
            S[i * BV + j] *= Es[C - 1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Each 8x8 state tile is read and written by the simdgroup that owns it.
        for (uint job = sg; job < (DK / 8) * (BV / 8); job += GDNC_THREADS / 32) {
            const uint bm = job / (BV / 8), bn = job % (BV / 8);
            simdgroup_float8x8 acc;
            simdgroup_load(acc, S + bm * 8 * BV + bn * 8, BV);
            gdnc_mac<true, false>(acc, kc, DK, Dt, BV, bm, bn, 0, C / 8);
            simdgroup_store(acc, S + bm * 8 * BV + bn * 8, BV);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((flags & 2u) != 0u) {
        for (uint j = 0; j < BV; ++j) {
            sfin[srow + j] = S[i * BV + j];
        }
    }
}

/// The backward's sequential pass, walking chunks last to first with the
/// slice's gradient state `dS` in threadgroup memory. Per chunk and 16-column
/// slice it forms D (rebuilt from the chunk's checkpoint) and dR, writes dv,
/// writes `dsbuf` (dS_C, which the full-width stages read), `dbuf`, `drbuf` and
/// `dop` (masked dO, head-major), and this slice's share of dG and dbeta into
/// `dgpart`, `dbpart` [NS, B, H, Tp]. flags: 1 = `s0` was given (write `ds0`);
/// 2 = `dfin` holds the final state's gradient (else zeros).
/// 128 threads = 4 simdgroups. Grid: x = Dv / 16, y = batch x value head.
kernel void gdn_train_c_bwd_a(
    device const float *v [[buffer(0)]],
    device const float *kh [[buffer(1)]],
    device const float *qh [[buffer(2)]],
    device const float *gc [[buffer(3)]],
    device const float *bt [[buffer(4)]],
    device const float *wm [[buffer(5)]],
    device const float *aq [[buffer(6)]],
    device const float *d_o [[buffer(7)]],
    device const float *dfin [[buffer(8)]],
    device const float *ckpt [[buffer(9)]],
    device float *dsbuf [[buffer(10)]],
    device float *dbuf [[buffer(11)]],
    device float *drbuf [[buffer(12)]],
    device float *dop [[buffer(13)]],
    device float *dv [[buffer(14)]],
    device float *dgpart [[buffer(15)]],
    device float *dbpart [[buffer(16)]],
    device float *ds0 [[buffer(17)]],
    constant uint &B [[buffer(18)]],
    constant uint &T [[buffer(19)]],
    constant uint &H [[buffer(20)]],
    constant uint &Dv [[buffer(21)]],
    constant uint &flags [[buffer(22)]],
    constant uint &ldv [[buffer(23)]],
    constant uint &offv [[buffer(24)]],
    constant uint &Hk [[buffer(25)]],
    constant uint &lddv [[buffer(26)]],
    constant uint &offdv [[buffer(27)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint i [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint BV = GDN_TRAIN_BV, C = GDNC_C, DK = GDN_TRAIN_DK;
    threadgroup float dS[DK * BV];
    threadgroup float T1[C * BV];  // X = K S0
    threadgroup float T4[C * BV];  // R, then dD
    threadgroup float T5[C * BV];  // D
    threadgroup float T6[C * BV];  // dO, then exp(G) dO
    threadgroup float T7[C * BV];  // Q S0, then K dS, then dR, then -exp(G) beta dR
    threadgroup float Es[C];
    threadgroup float Bs[C];
    threadgroup float Ws[C];
    threadgroup float xs[C];
    threadgroup float dgs[C];
    threadgroup float red[4];
    const uint bh = tg.y;
    if (bh >= B * H || (tg.x + 1) * BV > Dv) return;  // uniform per threadgroup
    const uint b = bh / H, h = bh % H, j0 = tg.x * BV;
    const uint hk = h / (H / Hk);
    const uint nc = (T + C - 1) / C;
    const ulong tp = (ulong)nc * C;
    const ulong kbh = (ulong)b * Hk + hk;
    const ulong bhn = (ulong)B * H;
    const ulong srow = ((ulong)bh * DK + i) * Dv + j0;

    for (uint j = 0; j < BV; ++j) {
        dS[i * BV + j] = (flags & 2u) != 0u ? dfin[srow + j] : 0.0f;
    }
    for (uint cc = nc; cc-- > 0;) {
        const uint t0 = cc * C;
        const ulong row0 = (ulong)bh * tp + t0;
        device const float *kc = kh + (kbh * tp + t0) * DK;
        device const float *qc = qh + (kbh * tp + t0) * DK;
        device const float *wc = wm + ((ulong)bh * nc + cc) * (C * C);
        device const float *ac = aq + ((ulong)bh * nc + cc) * (C * C);
        const ulong ck = ((ulong)bh * nc + cc) * DK;
        device const float *S0 = ckpt + ck * Dv + j0;
        // The previous chunk's dS update, and its reads of every tile below.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (i < C) {
            const float g = gc[row0 + i];
            Es[i] = precise::exp(g);
            Bs[i] = bt[row0 + i];
            Ws[i] = precise::exp(gc[row0 + C - 1] - g);
        }
        // dS_C out for the full-width stages, and this slice's <S0, dS_C>.
        {
            float p = 0.0f;
            for (uint j = 0; j < BV; ++j) {
                dsbuf[(ck + i) * Dv + j0 + j] = dS[i * BV + j];
                p += S0[(ulong)i * Dv + j] * dS[i * BV + j];
            }
            const float s = simd_sum(p);
            if (lane == 0) {
                red[sg] = s;
            }
        }
        // X = K S0, Q S0.
        for (uint job = sg; job < 2 * (C / 8) * (BV / 8); job += GDNC_THREADS / 32) {
            const uint which = job / ((C / 8) * (BV / 8)), r = job % ((C / 8) * (BV / 8));
            const uint bm = r / (BV / 8), bn = r % (BV / 8);
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            gdnc_mac<false, false>(acc, which != 0 ? qc : kc, DK, S0, Dv, bm, bn, 0, DK / 8);
            simdgroup_store(acc, (which != 0 ? T7 : T1) + bm * 8 * BV + bn * 8, BV);
        }
        // dO (rows past T are zero), saved head-major for the full-width stages.
        for (uint idx = i; idx < C * BV; idx += GDNC_THREADS) {
            const uint ri = idx / BV, j = idx % BV, t = t0 + ri;
            const float x = t < T ? d_o[(((ulong)b * T + t) * H + h) * Dv + j0 + j] : 0.0f;
            T6[idx] = x;
            dop[(row0 + ri) * Dv + j0 + j] = x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // R = beta (V - exp(G) X); and the part of dG that o carries,
        // exp(G_t) (S0^T q_t) . dO_t.
        for (uint idx = i; idx < C * BV; idx += GDNC_THREADS) {
            const uint ri = idx / BV, j = idx % BV, t = t0 + ri;
            const float vv = t < T ? v[((ulong)b * T + t) * ldv + offv + (ulong)h * Dv + j0 + j] : 0.0f;
            T4[idx] = Bs[ri] * (vv - Es[ri] * T1[idx]);
        }
        if (i < C) {
            float s = 0.0f;
            for (uint j = 0; j < BV; ++j) {
                s += T7[i * BV + j] * T6[i * BV + j];
            }
            dgs[i] = Es[i] * s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // D = W R.
        for (uint job = sg; job < (C / 8) * (BV / 8); job += GDNC_THREADS / 32) {
            const uint bm = job / (BV / 8), bn = job % (BV / 8);
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            gdnc_mac<false, false>(acc, wc, C, T4, BV, bm, bn, 0, bm + 1);
            simdgroup_store(acc, T5 + bm * 8 * BV + bn * 8, BV);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = i; idx < C * BV; idx += GDNC_THREADS) {
            dbuf[(row0 + idx / BV) * Dv + j0 + idx % BV] = T5[idx];
        }
        // dD = M^T dO + exp(G_C - G) (K dS_C): M^T first, into R's tile, then
        // K dS_C into the tile Q S0 held.
        for (uint job = sg; job < 2 * (C / 8) * (BV / 8); job += GDNC_THREADS / 32) {
            const uint which = job / ((C / 8) * (BV / 8)), r = job % ((C / 8) * (BV / 8));
            const uint bm = r / (BV / 8), bn = r % (BV / 8);
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            if (which == 0) {
                gdnc_mac<true, false>(acc, ac, C, T6, BV, bm, bn, bm, C / 8);
            } else {
                gdnc_mac<false, false>(acc, kc, DK, dS, BV, bm, bn, 0, DK / 8);
            }
            simdgroup_store(acc, (which != 0 ? T7 : T4) + bm * 8 * BV + bn * 8, BV);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // x_i = (dS_C^T k_i) . D_i, and dD = M^T dO + exp(G_C - G) K dS_C.
        if (i < C) {
            float x = 0.0f;
            for (uint j = 0; j < BV; ++j) {
                x += T7[i * BV + j] * T5[i * BV + j];
                T4[i * BV + j] += Ws[i] * T7[i * BV + j];
            }
            xs[i] = x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // dR = W^T dD (W^T is upper triangular).
        for (uint job = sg; job < (C / 8) * (BV / 8); job += GDNC_THREADS / 32) {
            const uint bm = job / (BV / 8), bn = job % (BV / 8);
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            gdnc_mac<true, false>(acc, wc, C, T4, BV, bm, bn, bm, C / 8);
            simdgroup_store(acc, T7 + bm * 8 * BV + bn * 8, BV);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // dR and dv out, this slice's dG and dbeta, and the scalings the dS
        // update needs. Row i is touched only by thread i from here to the next
        // barrier.
        if (i < C) {
            const uint t = t0 + i;
            float dbr = 0.0f, xr = 0.0f;
            for (uint j = 0; j < BV; ++j) {
                const float vv = t < T ? v[((ulong)b * T + t) * ldv + offv + (ulong)h * Dv + j0 + j] : 0.0f;
                const float z = vv - Es[i] * T1[i * BV + j];
                const float dr = T7[i * BV + j];
                dbr += dr * z;
                xr += T1[i * BV + j] * dr;
                drbuf[(row0 + i) * Dv + j0 + j] = dr;
                if (t < T) {
                    const ulong tok = (ulong)b * T + t;
                    dv[tok * lddv + offdv + (ulong)h * Dv + j0 + j] = Bs[i] * dr;
                }
            }
            float dg = dgs[i] - Es[i] * Bs[i] * xr - Ws[i] * xs[i];
            if (i == C - 1) {
                float sw = 0.0f;
                for (uint r = 0; r < C; ++r) {
                    sw += Ws[r] * xs[r];
                }
                dg += sw + Es[C - 1] * ((red[0] + red[1]) + (red[2] + red[3]));
            }
            dgpart[((ulong)tg.x * bhn + bh) * tp + t0 + i] = dg;
            dbpart[((ulong)tg.x * bhn + bh) * tp + t0 + i] = dbr;
            for (uint j = 0; j < BV; ++j) {
                T6[i * BV + j] *= Es[i];
                T7[i * BV + j] *= -(Es[i] * Bs[i]);
            }
        }
        for (uint j = 0; j < BV; ++j) {
            dS[i * BV + j] *= Es[C - 1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // dS = exp(G_C) dS_C + Q^T (exp(G) dO) - K^T (exp(G) beta dR).
        for (uint job = sg; job < (DK / 8) * (BV / 8); job += GDNC_THREADS / 32) {
            const uint bm = job / (BV / 8), bn = job % (BV / 8);
            simdgroup_float8x8 acc;
            simdgroup_load(acc, dS + bm * 8 * BV + bn * 8, BV);
            gdnc_mac<true, false>(acc, qc, DK, T6, BV, bm, bn, 0, C / 8);
            gdnc_mac<true, false>(acc, kc, DK, T7, BV, bm, bn, 0, C / 8);
            simdgroup_store(acc, dS + bm * 8 * BV + bn * 8, BV);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((flags & 1u) != 0u) {
        for (uint j = 0; j < BV; ++j) {
            ds0[srow + j] = dS[i * BV + j];
        }
    }
}

/// Per (batch x value head, chunk): the C x C gradients that sum over all of
/// Dv. From P = dO D^T, dA' = dR D^T and K K^T it writes, over the lower
/// triangle (zero above the diagonal inside a diagonal tile; tiles above are
/// never written or read): `dscore` = E P, `das` = dA beta E (dA = -dA'),
/// `gtot` = M P + dA A (the (t, i) share of dG_t, and minus that of dG_i) and
/// `dbs` = dA E (k_t . k_i) (the share of dbeta_t). 128 threads.
/// Grid: x = chunk, y = batch x value head.
kernel void gdn_train_c_mats(
    device const float *dop [[buffer(0)]],
    device const float *dbuf [[buffer(1)]],
    device const float *drbuf [[buffer(2)]],
    device const float *kh [[buffer(3)]],
    device const float *gc [[buffer(4)]],
    device const float *bt [[buffer(5)]],
    device const float *aq [[buffer(6)]],
    device float *dscore [[buffer(7)]],
    device float *das [[buffer(8)]],
    device float *gtot [[buffer(9)]],
    device float *dbs [[buffer(10)]],
    constant uint &B [[buffer(11)]],
    constant uint &T [[buffer(12)]],
    constant uint &H [[buffer(13)]],
    constant uint &Dv [[buffer(14)]],
    constant uint &Hk [[buffer(15)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint i [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint C = GDNC_C, DK = GDN_TRAIN_DK;
    threadgroup float tmp[4 * 3 * 64];
    threadgroup float Gs[C];
    threadgroup float Bs[C];
    const uint chunk = tg.x, bh = tg.y;
    if (bh >= B * H) return;  // uniform per threadgroup
    const uint b = bh / H, h = bh % H, hk = h / (H / Hk);
    const uint nc = (T + C - 1) / C;
    const ulong tp = (ulong)nc * C;
    const ulong row0 = (ulong)bh * tp + chunk * C;
    const ulong mat = ((ulong)bh * nc + chunk) * (C * C);
    device const float *kc = kh + ((ulong)(b * Hk + hk) * tp + chunk * C) * DK;
    device const float *dO = dop + row0 * Dv;
    device const float *Dm = dbuf + row0 * Dv;
    device const float *dR = drbuf + row0 * Dv;
    threadgroup float *st = tmp + sg * 3 * 64;
    if (i < C) {
        Gs[i] = gc[row0 + i];
        Bs[i] = bt[row0 + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint job = 0;
    for (uint bm = 0; bm < C / 8; ++bm) {
        for (uint bn = 0; bn <= bm; ++bn, ++job) {
            if (job % (GDNC_THREADS / 32) != sg) {
                continue;
            }
            simdgroup_float8x8 p = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 dp = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 kk = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            gdnc_mac<false, true>(p, dO, Dv, Dm, Dv, bm, bn, 0, Dv / 8);
            gdnc_mac<false, true>(dp, dR, Dv, Dm, Dv, bm, bn, 0, Dv / 8);
            gdnc_mac<false, true>(kk, kc, DK, kc, DK, bm, bn, 0, DK / 8);
            simdgroup_store(p, st, 8);
            simdgroup_store(dp, st + 64, 8);
            simdgroup_store(kk, st + 128, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                const uint t = bm * 8 + e / 8, ii = bn * 8 + e % 8;
                const ulong at = mat + (ulong)t * C + ii;
                float sc = 0.0f, dA = 0.0f, dAs = 0.0f, gm = 0.0f, gt = 0.0f, db = 0.0f;
                if (ii <= t) {
                    const float E = precise::exp(Gs[t] - Gs[ii]);
                    sc = E * st[e];
                    gm = aq[at] * st[e];
                    if (ii < t) {
                        dA = -st[64 + e];
                        dAs = dA * Bs[t] * E;
                        gt = dAs * st[128 + e];
                        db = dA * E * st[128 + e];
                    }
                }
                dscore[at] = sc;
                das[at] = dAs;
                gtot[at] = gm + gt;
                dbs[at] = db;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

/// Per (batch x value head, chunk): dG_t and dbeta_t from the C x C matrices
/// and the slices' partials, then dg_s = sum_{t >= s} dG_t over the chunk.
/// 128 threads (the first 64 work). Grid: x = chunk, y = batch x value head.
kernel void gdn_train_c_gates(
    device const float *gtot [[buffer(0)]],
    device const float *dbs [[buffer(1)]],
    device const float *dgpart [[buffer(2)]],
    device const float *dbpart [[buffer(3)]],
    device float *dg [[buffer(4)]],
    device float *dbeta [[buffer(5)]],
    constant uint &B [[buffer(6)]],
    constant uint &T [[buffer(7)]],
    constant uint &H [[buffer(8)]],
    constant uint &ns [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint i [[thread_index_in_threadgroup]])
{
    constexpr uint C = GDNC_C;
    threadgroup float dgs[C];
    const uint chunk = tg.x, bh = tg.y;
    if (bh >= B * H) return;  // uniform per threadgroup
    const uint b = bh / H, h = bh % H;
    const uint nc = (T + C - 1) / C;
    const ulong tp = (ulong)nc * C;
    const ulong mat = ((ulong)bh * nc + chunk) * (C * C);
    const ulong bhn = (ulong)B * H;
    float db = 0.0f;
    if (i < C) {
        float rs = 0.0f, cs = 0.0f;
        for (uint j = 0; j <= i; ++j) {
            rs += gtot[mat + (ulong)i * C + j];
            db += dbs[mat + (ulong)i * C + j];
        }
        for (uint r = i; r < C; ++r) {
            cs += gtot[mat + (ulong)r * C + i];
        }
        float dgp = 0.0f, dbp = 0.0f;
        for (uint s = 0; s < ns; ++s) {
            const ulong at = (s * bhn + bh) * tp + chunk * C + i;
            dgp += dgpart[at];
            dbp += dbpart[at];
        }
        dgs[i] = (rs - cs) + dgp;
        db += dbp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i < C && chunk * C + i < T) {
        float cum = 0.0f;
        for (uint t = C; t-- > i;) {
            cum += dgs[t];
        }
        const ulong r = ((ulong)b * T + chunk * C + i) * H + h;
        dg[r] = cum;
        dbeta[r] = db;
    }
}

/// Per (batch x value head, chunk): dq and dk for the value head, i.e. the
/// gradients with respect to the normalized q^ and k^, written dense
/// [B, T, H, 128] for gdn_train_c_finish. Each output 8x8 tile is one
/// simdgroup's, accumulated over every term that has no row scale in
/// registers and added to the row-scaled terms after a trip through threadgroup
/// memory. 128 threads. Grid: x = chunk, y = batch x value head.
kernel void gdn_train_c_dqk(
    device const float *kh [[buffer(0)]],
    device const float *qh [[buffer(1)]],
    device const float *gc [[buffer(2)]],
    device const float *bt [[buffer(3)]],
    device const float *dop [[buffer(4)]],
    device const float *dbuf [[buffer(5)]],
    device const float *drbuf [[buffer(6)]],
    device const float *ckpt [[buffer(7)]],
    device const float *dsbuf [[buffer(8)]],
    device const float *dscore [[buffer(9)]],
    device const float *das [[buffer(10)]],
    device float *dqh [[buffer(11)]],
    device float *dkh [[buffer(12)]],
    constant uint &B [[buffer(13)]],
    constant uint &T [[buffer(14)]],
    constant uint &H [[buffer(15)]],
    constant uint &Dv [[buffer(16)]],
    constant uint &Hk [[buffer(17)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint i [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint C = GDNC_C, DK = GDN_TRAIN_DK;
    threadgroup float tmp[4 * 5 * 64];
    threadgroup float Es[C];
    threadgroup float Bs[C];
    threadgroup float Ws[C];
    const uint chunk = tg.x, bh = tg.y;
    if (bh >= B * H) return;  // uniform per threadgroup
    const uint b = bh / H, h = bh % H, hk = h / (H / Hk);
    const uint nc = (T + C - 1) / C;
    const ulong tp = (ulong)nc * C;
    const uint t0 = chunk * C;
    const ulong row0 = (ulong)bh * tp + t0;
    const ulong mat = ((ulong)bh * nc + chunk) * (C * C);
    const ulong ck = ((ulong)bh * nc + chunk) * DK;
    device const float *kc = kh + ((ulong)(b * Hk + hk) * tp + t0) * DK;
    device const float *qc = qh + ((ulong)(b * Hk + hk) * tp + t0) * DK;
    device const float *dO = dop + row0 * Dv;
    device const float *Dm = dbuf + row0 * Dv;
    device const float *dR = drbuf + row0 * Dv;
    device const float *S0 = ckpt + ck * Dv;
    device const float *dSC = dsbuf + ck * Dv;
    device const float *sc = dscore + mat;
    device const float *as = das + mat;
    threadgroup float *st = tmp + sg * 5 * 64;
    if (i < C) {
        const float g = gc[row0 + i];
        Es[i] = precise::exp(g);
        Bs[i] = bt[row0 + i];
        Ws[i] = precise::exp(gc[row0 + C - 1] - g);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint job = sg; job < (C / 8) * (DK / 8); job += GDNC_THREADS / 32) {
        const uint bm = job / (DK / 8), bn = job % (DK / 8);
        simdgroup_float8x8 a1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 a2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 a3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 a4 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 a5 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        // dq = (E dM) K + exp(G) (dO S0^T).
        gdnc_mac<false, false>(a1, sc, C, kc, DK, bm, bn, 0, bm + 1);
        gdnc_mac<false, true>(a2, dO, Dv, S0, Dv, bm, bn, 0, Dv / 8);
        // dk = (E dM)^T Q + dA' K + dA'^T K + E_C-G (D dS_C^T) - exp(G) beta (dR S0^T).
        gdnc_mac<true, false>(a3, sc, C, qc, DK, bm, bn, bm, C / 8);
        gdnc_mac<false, false>(a3, as, C, kc, DK, bm, bn, 0, bm + 1);
        gdnc_mac<true, false>(a3, as, C, kc, DK, bm, bn, bm, C / 8);
        gdnc_mac<false, true>(a4, Dm, Dv, dSC, Dv, bm, bn, 0, Dv / 8);
        gdnc_mac<false, true>(a5, dR, Dv, S0, Dv, bm, bn, 0, Dv / 8);
        simdgroup_store(a1, st, 8);
        simdgroup_store(a2, st + 64, 8);
        simdgroup_store(a3, st + 128, 8);
        simdgroup_store(a4, st + 192, 8);
        simdgroup_store(a5, st + 256, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 64; e += 32) {
            const uint ri = bm * 8 + e / 8, t = t0 + ri;
            if (t < T) {
                const ulong at = (((ulong)b * T + t) * H + h) * DK + bn * 8 + e % 8;
                dqh[at] = st[e] + Es[ri] * st[64 + e];
                dkh[at] = st[128 + e] + Ws[ri] * st[192 + e] - Es[ri] * Bs[ri] * st[256 + e];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/// The l2norm backward, with y = x r, r = rsqrt(|x|^2 + eps): dx = r (dy - y
/// (y . dy)); for q the incoming gradient is first scaled by 1/sqrt(Dk). One
/// threadgroup of 128 per (token, key head) row: q and k are read through their
/// windows once, and each of the key head's H / Hk value heads contributes its
/// dx; they are added in value-head order into the dq and dk windows. The sum
/// goes through threadgroup memory so the compiler cannot fuse the add into
/// the product that made it: the result is the sum of separately rounded
/// gradients (as gdn_train_bwd_finish).
kernel void gdn_train_c_finish(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *dqh [[buffer(2)]],
    device const float *dkh [[buffer(3)]],
    device float *dq [[buffer(4)]],
    device float *dk [[buffer(5)]],
    constant uint &ldq [[buffer(6)]],
    constant uint &offq [[buffer(7)]],
    constant uint &ldk [[buffer(8)]],
    constant uint &offk [[buffer(9)]],
    constant uint &lddq [[buffer(10)]],
    constant uint &offdq [[buffer(11)]],
    constant uint &lddk [[buffer(12)]],
    constant uint &offdk [[buffer(13)]],
    constant uint &toks [[buffer(14)]],
    constant uint &H [[buffer(15)]],
    constant uint &Hk [[buffer(16)]],
    uint r [[threadgroup_position_in_grid]],
    uint i [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[2][GDN_TRAIN_SG * 2];
    threadgroup float stage[2][GDN_TRAIN_DK];
    if (r >= toks * Hk) return;  // uniform per threadgroup
    const uint tok = r / Hk, hk = r % Hk, ratio = H / Hk;
    const ulong qat = (ulong)tok * ldq + offq + (ulong)hk * GDN_TRAIN_DK + i;
    const ulong kat = (ulong)tok * ldk + offk + (ulong)hk * GDN_TRAIN_DK + i;
    const float qi = q[qat], ki = k[kat];
    float p[2] = {qi * qi, ki * ki}, tot[2];
    gdn_colsum<2>(p, red[0], sg, lane, tot);
    const float rq = precise::rsqrt(tot[0] + GDN_TRAIN_L2_EPS);
    const float rk = precise::rsqrt(tot[1] + GDN_TRAIN_L2_EPS);
    const float yq = qi * rq, yk = ki * rk;
    float sq = 0.0f, sk = 0.0f;
    for (uint j = 0; j < ratio; ++j) {
        const ulong at = ((ulong)tok * H + (ulong)hk * ratio + j) * GDN_TRAIN_DK + i;
        const float dyq = dqh[at] * rsqrt((float)GDN_TRAIN_DK), dyk = dkh[at];
        float p2[2] = {yq * dyq, yk * dyk}, dots[2];
        // red[0] holds the norms' sums (read above, ordered by this call's
        // barrier for j = 0); successive calls alternate buffers.
        gdn_colsum<2>(p2, red[(j + 1) & 1u], sg, lane, dots);
        stage[0][i] = rq * (dyq - yq * dots[0]);
        stage[1][i] = rk * (dyk - yk * dots[1]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float xq = stage[0][i], xk = stage[1][i];
        sq = j == 0 ? xq : sq + xq;
        sk = j == 0 ? xk : sk + xk;
    }
    dq[(ulong)tok * lddq + offdq + (ulong)hk * GDN_TRAIN_DK + i] = sq;
    dk[(ulong)tok * lddk + offdk + (ulong)hk * GDN_TRAIN_DK + i] = sk;
}
