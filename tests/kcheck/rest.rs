//! Numeric kernels outside GEMM, flash attention, quant, AdamW and Qwen.
//!
//! Bounds are of the reference peak unless a check says bit-identical.
//! TensorOps attention and convolution use this host reference. A kernel
//! that never writes a lane fails the finite check.

use std::sync::Arc;

use crate::common::embedgemma2_audio as audio_host;
use crate::common::embedgemma2_vision::{pool_dense, rope_axial};
use crate::common::gdn_train::{self, Inputs as GdnIn, Shape as GdnShape};
use crate::common::{
    buf, buf_bf16, buf_u32, gelu_pytorch_tanh, random_f32, round_trip_bf16, silu, tensor_bf16, tensor_f32,
};
use tessl::bert::{self, EmbedTables};
use tessl::conv::{conv2d_nchw, ConvPrecision};
use tessl::cross_entropy::{cross_entropy_rows, CeGrads, CeHidden, CeWorkspace, Reduction};
use tessl::embedgemma2::{self, EncoderAttnDims, EncoderAttnKernel};
use tessl::embedgemma2_audio::{self, AttentionSpec, Clamp, ClipTable, Ew, Lconv, RmsNorm};
use tessl::embedgemma2_vision::{self, AxialQkvShape};
use tessl::gdn_train::{
    gdn_train_backward, gdn_train_forward_with, GdnTrainDims, GdnTrainGrads, GdnTrainInputs, GdnTrainKernel,
    GdnTrainWorkspace,
};
use tessl::gemm::GemmOperands;
use tessl::nn::{self, KvStoreTarget, QkvBuffers, QkvRopeDims, QkvRopeVariant};
use tessl::qwen35::{AttnTargets, Cols};
use tessl::tensor::{bf16_bits_to_f32, f32_to_bf16_bits_gpu};
use tessl::{DType, GpuBuffer, GpuRuntime, PrecisionMode, Tensor};

use super::Ledger;

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().copied().map(f64::from).collect()
}

fn ok(what: &str, r: Result<(), String>) {
    r.unwrap_or_else(|e| panic!("{what}: {e}"));
}

fn close_peak(what: &str, got: &[f32], want: &[f64], rel: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let peak = want.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-30);
    let mut worst = (0.0f64, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what}[{i}]: non-finite {g}");
        let e = (f64::from(g) - w).abs();
        if e > worst.0 {
            worst = (e, i);
        }
    }
    assert!(
        worst.0 <= rel * peak,
        "{what}: max err {:.3e} at {} (got {} want {}), {:.3e} of peak {:.3e} (bound {rel:.1e})",
        worst.0,
        worst.1,
        got[worst.1],
        want[worst.1],
        worst.0 / peak,
        peak
    );
}

fn bits_eq(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}[{i}]: got {g} want {w}");
    }
}

fn read_bf16(buf: &GpuBuffer, n: usize) -> Vec<f32> {
    buf.contents_u16()[..n].iter().copied().map(bf16_bits_to_f32).collect()
}

fn wave(n: usize, amp: f32, phase: f32) -> Vec<f32> {
    (0..n).map(|i| amp * (i as f32 * 0.17 + phase).sin()).collect()
}

fn rms_row(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, wt)| v * inv * wt).collect()
}

fn layer_norm_row(x: &[f64], w: &[f64], b: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(w)
        .zip(b)
        .map(|((v, wt), bias)| (v - mean) * inv * wt + bias)
        .collect()
}

/// Abramowitz and Stegun 7.1.26. Absolute error is about 1.5e-7, independent
/// of the Metal Chebyshev `bert_erfc`.
fn erf_as(x: f64) -> f64 {
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    let y = 1.0 - p * (-a * a).exp();
    if x < 0.0 {
        -y
    } else {
        y
    }
}

fn gelu_erf(x: f64) -> f64 {
    0.5 * x * (1.0 + erf_as(x / std::f64::consts::SQRT_2))
}

fn softcap(x: f64, cap: f64) -> f64 {
    cap * (x / cap).tanh()
}

/// `checked` names this module is responsible for. The device run is the
/// proof; this list is what a missing credit looks like before that run.
pub const NAMES: &[&str] = &[
    "argmax_f32",
    "bert_bias_add_f32",
    "bert_bias_gelu_erf_f32",
    "bert_bias_residual_layer_norm_f32",
    "bert_embed_layer_norm_f32",
    "bert_layer_norm_f32",
    "bert_segment_sparse_max_f32",
    "ce_gather_rows_bf16",
    "ce_gather_rows_f32",
    "ce_lse_update",
    "ce_softmax_grad",
    "conv2d_pack_nhwc_f16",
    "conv2d_pack_nhwc_f32",
    "conv2d_tops_nchw_k3_o16_w32_f16",
    "conv2d_tops_nchw_k3_o16_w32_f32",
    "conv2d_tops_nchw_k3_o64_w32_f16",
    "conv2d_tops_nchw_k3_o64_w32_f32",
    "conv2d_weights_hwio_f16",
    "conv2d_weights_hwio_f32",
    "eg2a_attn",
    "eg2a_ew",
    "eg2a_lconv",
    "eg2a_rms_norm",
    "eg2a_sub_conv0_ln_relu",
    "eg2a_sub_conv1_ln_relu",
    "eg2t_resid_add_bf16_copy",
    "eg2t_resid_norm_bf16",
    "eg2t_resid_norm_f32",
    "eg2t_scale_rms_norm_f32",
    "encoder_attn_rows_h256_r16_g32",
    "encoder_attn_rows_h32_r2_g8",
    "encoder_attn_rows_h512_r32_g32",
    "encoder_attn_rows_h64_r4_g16",
    "encoder_attn_tiled_bf16_h256_q32_k64_sg4",
    "encoder_attn_tiled_bf16_h512_q32_k128_sg8",
    "encoder_attn_tiled_bf16_h64_q32_k32_sg2",
    "encoder_attn_tiled_bf16o_h256_q32_k64_sg4",
    "encoder_attn_tiled_bf16o_h512_q32_k128_sg8",
    "encoder_attn_tiled_bf16o_h64_q32_k32_sg2",
    "encoder_attn_tiled_h256_q32_k32_sg4",
    "encoder_attn_tiled_h32_q16_k32_sg1",
    "encoder_attn_tiled_h512_q32_k32_sg8",
    "encoder_attn_tiled_h64_q32_k32_sg2",
    "gdn_train_bwd",
    "gdn_train_bwd_finish",
    "gdn_train_c_bwd_a",
    "gdn_train_c_dqk",
    "gdn_train_c_finish",
    "gdn_train_c_fwd",
    "gdn_train_c_gates",
    "gdn_train_c_mats",
    "gdn_train_c_norm",
    "gdn_train_c_prep",
    "gdn_train_fwd",
    "kv_ring_densify",
    "kv_store_timestep",
    "kv_store_timestep_pair",
    "l2_normalize_rows_f32",
    "mlp_gelu_tanh",
    "mlp_gelu_tanh_bf16",
    "mlp_silu",
    "rms_norm_bf16",
    "rms_norm_f32",
    "rms_norm_residual_add_f32",
    "rms_qkv_rope",
    "rms_qkv_rope_kv_store",
    "rms_qkv_rope_posbuf",
    "row_max_f32",
    "row_sum_f32",
    "scale_f32_inplace",
    "segment_mean_rows_f32",
    "softcap_argmax_one_pass",
    "softcap_f32",
    "softcap_logits",
    "softcap_sample",
    "softmax_rows_f32",
    "vision_pool_members_f32",
    "vision_pos_embed_add_f32",
    "vision_qkv_norm_rope_bf16",
    "vision_qkv_norm_rope_f32",
];

pub fn run(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    // ExactF32 cross-entropy refuses a relaxed f32 GEMM. Gemm restores the
    // flags at the end of its sweep; set them again here.
    rt.set_precision(PrecisionMode::F32);
    rt.set_relaxed_precision(false);
    assert_eq!(NAMES.len(), 80, "rest coverage list drifted");
    elementwise(rt, ledger);
    bert(rt, ledger);
    convs(rt, ledger);
    cross_entropy(rt, ledger);
    encoder(rt, ledger);
    text_fuse(rt, ledger);
    audio(rt, ledger);
    vision(rt, ledger);
    gdn(rt, ledger);
    rope(rt, ledger);
    kv(rt, ledger);
}

fn elementwise(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let rows = 2u32;
    let dim = 4u32;
    let n = (rows * dim) as usize;
    let eps = 1e-6f32;
    let x = wave(n, 0.4, 0.2);
    let w = wave(dim as usize, 0.15, 1.0)
        .iter()
        .map(|v| v + 1.0)
        .collect::<Vec<_>>();
    let x64 = f64s(&x);
    let w64 = f64s(&w);

    ledger.credit(rt, "rms_norm_f32", &["rms_norm_f32"], || {
        let (xb, wb, out) = (buf(rt, &x), buf(rt, &w), buf(rt, &vec![0.0; n]));
        ok("rms_norm_f32", nn::rms_norm_f32(rt, &xb, &wb, &out, rows, dim, eps));
        let want: Vec<f64> = x64
            .chunks(dim as usize)
            .flat_map(|row| rms_row(row, &w64, f64::from(eps)))
            .collect();
        close_peak("rms_norm_f32", &out.read_f32(), &want, 1e-5);
    });

    ledger.credit(rt, "rms_norm_bf16", &["rms_norm_bf16"], || {
        let (xb, wb, out) = (buf(rt, &x), buf(rt, &w), buf_bf16(rt, &vec![0.0; n]));
        ok("rms_norm_bf16", nn::rms_norm_bf16(rt, &xb, &wb, &out, rows, dim, eps));
        let want: Vec<f64> = x64
            .chunks(dim as usize)
            .flat_map(|row| rms_row(row, &w64, f64::from(eps)))
            .collect();
        // The store rounds the f32 row to bf16. 2e-2 of peak covers that rounding.
        close_peak("rms_norm_bf16", &read_bf16(&out, n), &want, 2e-2);
    });

    ledger.credit(rt, "rms_norm_residual_add_f32", &["rms_norm_residual_add_f32"], || {
        let resid = wave(n, 0.2, 0.7);
        let (xb, wb, rb) = (buf(rt, &x), buf(rt, &w), buf(rt, &resid));
        ok(
            "rms_norm_residual_add_f32",
            nn::rms_norm_residual_add_f32(rt, &xb, &wb, &rb, rows, dim, eps, 0.5),
        );
        let mut want = Vec::new();
        for (row, rr) in x64.chunks(dim as usize).zip(f64s(&resid).chunks(dim as usize)) {
            let normed = rms_row(row, &w64, f64::from(eps));
            want.extend(rr.iter().zip(&normed).map(|(r, y)| 0.5 * (r + y)));
        }
        close_peak("rms_norm_residual_add_f32", &rb.read_f32(), &want, 1e-5);
    });

    let gate = wave(8, 0.8, 0.1);
    let up = wave(8, 0.5, 0.4);
    ledger.credit(rt, "mlp_silu", &["mlp_silu"], || {
        let (g, u, out) = (buf(rt, &gate), buf(rt, &up), buf(rt, &[0.0; 8]));
        ok("mlp_silu", nn::mlp_silu(rt, &g, &u, &out, 8));
        let want: Vec<f64> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| silu(f64::from(g)) * f64::from(u))
            .collect();
        close_peak("mlp_silu", &out.read_f32(), &want, 1e-5);
    });
    ledger.credit(rt, "mlp_gelu_tanh", &["mlp_gelu_tanh"], || {
        let (g, u, out) = (buf(rt, &gate), buf(rt, &up), buf(rt, &[0.0; 8]));
        ok("mlp_gelu_tanh", nn::mlp_gelu_tanh(rt, &g, &u, &out, 8));
        let want: Vec<f64> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| gelu_pytorch_tanh(f64::from(g)) * f64::from(u))
            .collect();
        close_peak("mlp_gelu_tanh", &out.read_f32(), &want, 1e-5);
    });
    ledger.credit(rt, "mlp_gelu_tanh_bf16", &["mlp_gelu_tanh_bf16"], || {
        let (g, u, out) = (buf(rt, &gate), buf(rt, &up), buf_bf16(rt, &[0.0; 8]));
        ok("mlp_gelu_tanh_bf16", nn::mlp_gelu_tanh_bf16(rt, &g, &u, &out, 8));
        let want: Vec<f64> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| gelu_pytorch_tanh(f64::from(g)) * f64::from(u))
            .collect();
        close_peak("mlp_gelu_tanh_bf16", &read_bf16(&out, 8), &want, 2e-2);
    });

    ledger.credit(rt, "scale_f32_inplace", &["scale_f32_inplace"], || {
        // Dyadic values times 0.25 stay dyadic, so the product is bit-identical.
        let src = [1.0f32, 2.0, 4.0, 0.5, 8.0, 0.25, 16.0, 0.125];
        let xb = buf(rt, &src);
        ok("scale_f32_inplace", nn::scale_f32_inplace(rt, &xb, 0.25, 8));
        let want: Vec<f32> = src.iter().map(|v| v * 0.25).collect();
        bits_eq("scale_f32_inplace", &xb.read_f32(), &want);
    });

    let cols = 4u32;
    let mat = wave(n, 0.6, 0.3);
    ledger.credit(rt, "softmax_rows_f32", &["softmax_rows_f32"], || {
        let (xb, out) = (buf(rt, &mat), buf(rt, &vec![0.0; n]));
        ok("softmax_rows_f32", nn::softmax_rows_f32(rt, &xb, &out, rows, cols));
        let mut want = Vec::new();
        for row in f64s(&mat).chunks(cols as usize) {
            let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = row.iter().map(|v| (v - m).exp()).collect();
            let z: f64 = e.iter().sum();
            want.extend(e.iter().map(|v| v / z));
        }
        close_peak("softmax_rows_f32", &out.read_f32(), &want, 1e-5);
    });
    ledger.credit(rt, "row_sum_f32", &["row_sum_f32"], || {
        let (xb, out) = (buf(rt, &mat), buf(rt, &vec![0.0; rows as usize]));
        ok("row_sum_f32", nn::row_sum_f32(rt, &xb, &out, rows, cols));
        let want: Vec<f64> = f64s(&mat).chunks(cols as usize).map(|row| row.iter().sum()).collect();
        close_peak("row_sum_f32", &out.read_f32(), &want, 1e-5);
    });
    ledger.credit(rt, "row_max_f32", &["row_max_f32"], || {
        let (xb, out) = (buf(rt, &mat), buf(rt, &vec![0.0; rows as usize]));
        ok("row_max_f32", nn::row_max_f32(rt, &xb, &out, rows, cols));
        let want: Vec<f32> = mat
            .chunks(cols as usize)
            .map(|row| row.iter().copied().fold(f32::NEG_INFINITY, f32::max))
            .collect();
        bits_eq("row_max_f32", &out.read_f32(), &want);
    });

    let cap = 4.0f32;
    let logits = wave(8, 3.0, 0.5);
    ledger.credit(rt, "softcap_f32", &["softcap_f32"], || {
        let pre = tensor_f32(rt, &[8], &logits);
        let out = tessl::softcap_f32(rt, &pre, cap).expect("softcap_f32");
        let want: Vec<f64> = logits.iter().map(|&v| softcap(f64::from(v), f64::from(cap))).collect();
        close_peak("softcap_f32", &out.read_f32().expect("read"), &want, 1e-5);
    });
    ledger.credit(rt, "softcap_logits", &["softcap_logits"], || {
        let (xb, sb) = (buf(rt, &logits), buf(rt, &[cap]));
        ok("softcap_logits", nn::softcap_logits(rt, &xb, &sb, 8));
        let want: Vec<f64> = logits.iter().map(|&v| softcap(f64::from(v), f64::from(cap))).collect();
        close_peak("softcap_logits", &xb.read_f32(), &want, 1e-5);
    });

    // Softcap is monotone, so the argmax index matches the raw argmax. The
    // value is still the softcapped one.
    let raw_at = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    ledger.credit(rt, "argmax_f32", &["argmax_f32"], || {
        let groups = nn::argmax_pass_groups(8);
        let (xb, idx, val, sb) = (
            buf(rt, &logits),
            buf_u32(rt, &vec![0; groups]),
            buf(rt, &vec![0.0; groups]),
            buf(rt, &[cap]),
        );
        ok("argmax_f32", nn::argmax_f32_pass(rt, &xb, &idx, &val, None, &sb, 8));
        let token = nn::check_argmax_result(&idx).expect("argmax");
        assert_eq!(token, raw_at, "argmax_f32 index");
        let want = softcap(f64::from(logits[raw_at as usize]), f64::from(cap));
        close_peak("argmax_f32 value", &val.read_f32()[..1], &[want], 1e-5);
    });
    ledger.credit(rt, "softcap_sample", &["softcap_sample"], || {
        let (xb, tok, sb) = (buf(rt, &logits), buf_u32(rt, &[u32::MAX]), buf(rt, &[cap]));
        ok("softcap_sample", nn::softcap_sample(rt, &xb, &tok, &sb, 8));
        let token = nn::check_argmax_result(&tok).expect("softcap_sample");
        assert_eq!(token, raw_at, "softcap_sample index");
        let want: Vec<f64> = logits.iter().map(|&v| softcap(f64::from(v), f64::from(cap))).collect();
        close_peak("softcap_sample logits", &xb.read_f32(), &want, 1e-5);
    });
    ledger.credit(rt, "softcap_argmax_one_pass", &["softcap_argmax_one_pass"], || {
        let (xb, tok, sb) = (buf(rt, &logits), buf_u32(rt, &[u32::MAX]), buf(rt, &[cap]));
        ok(
            "softcap_argmax_one_pass",
            nn::softcap_argmax_one_pass(rt, &xb, &tok, &sb, 8),
        );
        let token = nn::check_argmax_result(&tok).expect("softcap_argmax");
        assert_eq!(token, raw_at, "softcap_argmax_one_pass index");
        bits_eq("softcap_argmax_one_pass leaves logits", &xb.read_f32(), &logits);
    });
}

fn bert(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let (vocab, dim, seq, rows) = (4u32, 4u32, 2u32, 2u32);
    let eps = 1e-5f32;
    let word = wave((vocab * dim) as usize, 0.3, 0.1);
    let pos = wave(8, 0.2, 0.4);
    let type_row = wave(dim as usize, 0.1, 0.8);
    let ln_w = wave(dim as usize, 0.1, 0.2).iter().map(|v| v + 1.0).collect::<Vec<_>>();
    let ln_b = wave(dim as usize, 0.05, 1.2);
    let ids = [0u32, 1];

    ledger.credit(rt, "bert_embed_layer_norm_f32", &["bert_embed_layer_norm_f32"], || {
        let (ib, wb, pb, tb, w, b, out) = (
            buf_u32(rt, &ids),
            buf(rt, &word),
            buf(rt, &pos),
            buf(rt, &type_row),
            buf(rt, &ln_w),
            buf(rt, &ln_b),
            buf(rt, &vec![0.0; (rows * dim) as usize]),
        );
        let tables = EmbedTables {
            word: &wb,
            vocab,
            pos: &pb,
            positions: seq,
            type_row: &tb,
            ln_w: &w,
            ln_b: &b,
        };
        ok(
            "embed",
            bert::embed_layer_norm(rt, &ib, tables, &out, rows, dim, seq, eps),
        );
        let (w64, p64, t64) = (f64s(&word), f64s(&pos), f64s(&type_row));
        let mut want = Vec::new();
        for (r, &id) in ids.iter().enumerate() {
            let mut row = vec![0.0; dim as usize];
            for c in 0..dim as usize {
                row[c] = w64[id as usize * dim as usize + c] + p64[(r % seq as usize) * dim as usize + c] + t64[c];
            }
            want.extend(layer_norm_row(&row, &f64s(&ln_w), &f64s(&ln_b), f64::from(eps)));
        }
        close_peak("bert_embed_layer_norm_f32", &out.read_f32(), &want, 1e-5);
    });

    let y = wave(8, 0.4, 0.3);
    let bias = wave(4, 0.1, 0.6);
    let resid = wave(8, 0.25, 0.9);
    ledger.credit(
        rt,
        "bert_bias_residual_layer_norm_f32",
        &["bert_bias_residual_layer_norm_f32"],
        || {
            let (yb, bb, rb, w, b) = (
                buf(rt, &y),
                buf(rt, &bias),
                buf(rt, &resid),
                buf(rt, &ln_w),
                buf(rt, &ln_b),
            );
            ok(
                "bias_resid_ln",
                bert::bias_residual_layer_norm(rt, &yb, &bb, &rb, &w, &b, rows, dim, eps),
            );
            let mut want = Vec::new();
            for (row, rr) in f64s(&y).chunks(4).zip(f64s(&resid).chunks(4)) {
                let mixed: Vec<f64> = row
                    .iter()
                    .zip(&f64s(&bias))
                    .zip(rr)
                    .map(|((y, b), r)| y + b + r)
                    .collect();
                want.extend(layer_norm_row(&mixed, &f64s(&ln_w), &f64s(&ln_b), f64::from(eps)));
            }
            close_peak("bert_bias_residual_layer_norm_f32", &rb.read_f32(), &want, 1e-4);
        },
    );

    ledger.credit(rt, "bert_layer_norm_f32", &["bert_layer_norm_f32"], || {
        let (xb, w, b, out) = (buf(rt, &y), buf(rt, &ln_w), buf(rt, &ln_b), buf(rt, &[0.0; 8]));
        ok("layer_norm", bert::layer_norm(rt, &xb, &w, &b, &out, rows, dim, eps));
        let want: Vec<f64> = f64s(&y)
            .chunks(4)
            .flat_map(|row| layer_norm_row(row, &f64s(&ln_w), &f64s(&ln_b), f64::from(eps)))
            .collect();
        close_peak("bert_layer_norm_f32", &out.read_f32(), &want, 1e-5);
    });

    ledger.credit(rt, "bert_bias_add_f32", &["bert_bias_add_f32"], || {
        let (xb, bb) = (buf(rt, &y), buf(rt, &bias));
        ok("bias_add", bert::bias_add(rt, &xb, &bb, rows, dim));
        let want: Vec<f32> = y
            .chunks(4)
            .flat_map(|row| row.iter().zip(&bias).map(|(a, b)| a + b))
            .collect();
        bits_eq("bert_bias_add_f32", &xb.read_f32(), &want);
    });

    ledger.credit(rt, "bert_bias_gelu_erf_f32", &["bert_bias_gelu_erf_f32"], || {
        let (xb, bb) = (buf(rt, &y), buf(rt, &bias));
        ok("bias_gelu", bert::bias_gelu_erf(rt, &xb, &bb, rows, dim));
        let want: Vec<f64> = y
            .chunks(4)
            .flat_map(|row| row.iter().zip(&bias).map(|(&a, &b)| gelu_erf(f64::from(a + b))))
            .collect();
        close_peak("bert_bias_gelu_erf_f32", &xb.read_f32(), &want, 1e-5);
    });

    ledger.credit(
        rt,
        "bert_segment_sparse_max_f32",
        &["bert_segment_sparse_max_f32"],
        || {
            let logits = [0.2f32, 1.5, -0.4, 0.7, 0.1, 0.4, 2.0, -0.2];
            let bias = [0.1f32, -0.2, 0.3, 0.0];
            let segs = bert::upload_segments(rt, &[(0, 2)], 2).expect("segments");
            // Seed below every log1p(relu(.)) this input produces, so a no-op fails.
            let pooled = buf(rt, &[-1.0; 4]);
            ok(
                "sparse_max",
                bert::segment_sparse_max(rt, &buf(rt, &logits), &buf(rt, &bias), &segs, &pooled, 1, 2, 4),
            );
            let mut col_max = [f64::NEG_INFINITY; 4];
            for row in 0..2 {
                for v in 0..4 {
                    col_max[v] = col_max[v].max(f64::from(logits[row * 4 + v] + bias[v]));
                }
            }
            let want: Vec<f64> = col_max.iter().map(|m| (m.max(0.0)).ln_1p()).collect();
            close_peak("bert_segment_sparse_max_f32", &pooled.read_f32(), &want, 1e-5);
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn conv_ref(
    x: &[f32],
    wt: &[f32],
    bias: &[f32],
    n: usize,
    ci: usize,
    h: usize,
    w: usize,
    co: usize,
    pad: usize,
) -> Vec<f64> {
    let (oh, ow) = (h + 2 * pad - 2, w + 2 * pad - 2);
    let mut y = vec![0.0; n * co * oh * ow];
    for ni in 0..n {
        for o in 0..co {
            for i in 0..oh {
                for j in 0..ow {
                    let mut s = f64::from(bias[o]);
                    for ch in 0..ci {
                        for kh in 0..3 {
                            for kw in 0..3 {
                                let hh = i as isize + kh as isize - pad as isize;
                                let ww = j as isize + kw as isize - pad as isize;
                                if hh < 0 || ww < 0 || hh >= h as isize || ww >= w as isize {
                                    continue;
                                }
                                let xv = x[((ni * ci + ch) * h + hh as usize) * w + ww as usize];
                                s += f64::from(xv) * f64::from(wt[((o * ci + ch) * 3 + kh) * 3 + kw]);
                            }
                        }
                    }
                    y[((ni * co + o) * oh + i) * ow + j] = s;
                }
            }
        }
    }
    y
}

fn conv_one(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, precision: ConvPrecision, co: usize, names: &[&str]) {
    let (n, ci, h, w, pad) = (1usize, 1, 3, 3, 1);
    let (oh, ow) = (h + 2 * pad - 2, w + 2 * pad - 2);
    let x = wave(n * ci * h * w, 0.4, co as f32);
    let wt = wave(co * ci * 9, 0.05, 0.3);
    let bias = wave(co, 0.1, 0.8);
    let label = format!("{precision:?} co{co}");
    let tol = match precision {
        ConvPrecision::F32 => 1e-5,
        ConvPrecision::F16 => 2e-3,
    };
    ledger.credit(rt, &label, names, || {
        let xt = tensor_f32(rt, &[n, ci, h, w], &x);
        let wtb = tensor_f32(rt, &[co, ci, 3, 3], &wt);
        let bb = tensor_f32(rt, &[co], &bias);
        let yt = rt.alloc_tensor_f32(&[n, co, oh, ow]).expect("y");
        yt.write_f32(&vec![f32::NAN; n * co * oh * ow]).expect("poison y");
        ok("conv2d", conv2d_nchw(&xt, &wtb, Some(&bb), &yt, pad, precision));
        rt.synchronize().expect("conv sync");
        let got = yt.read_f32().expect("read y");
        let want = conv_ref(&x, &wt, &bias, n, ci, h, w, co, pad);
        assert!(got.iter().all(|v| v.is_finite()), "{label}: non-finite output");
        let (mut err, mut norm) = (0.0f64, 0.0f64);
        for (&g, &r) in got.iter().zip(&want) {
            let d = f64::from(g) - r;
            err += d * d;
            norm += r * r;
        }
        let rel = (err / norm.max(1e-30)).sqrt();
        assert!(rel < tol, "{label}: relative L2 {rel:.3e} over {tol:.1e}");
    });
}

fn convs(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    conv_one(
        rt,
        ledger,
        ConvPrecision::F32,
        16,
        &[
            "conv2d_pack_nhwc_f32",
            "conv2d_weights_hwio_f32",
            "conv2d_tops_nchw_k3_o16_w32_f32",
        ],
    );
    conv_one(
        rt,
        ledger,
        ConvPrecision::F32,
        64,
        &[
            "conv2d_pack_nhwc_f32",
            "conv2d_weights_hwio_f32",
            "conv2d_tops_nchw_k3_o64_w32_f32",
        ],
    );
    conv_one(
        rt,
        ledger,
        ConvPrecision::F16,
        16,
        &[
            "conv2d_pack_nhwc_f16",
            "conv2d_weights_hwio_f16",
            "conv2d_tops_nchw_k3_o16_w32_f16",
        ],
    );
    conv_one(
        rt,
        ledger,
        ConvPrecision::F16,
        64,
        &[
            "conv2d_pack_nhwc_f16",
            "conv2d_weights_hwio_f16",
            "conv2d_tops_nchw_k3_o64_w32_f16",
        ],
    );
}

fn ce_case(
    rt: &Arc<GpuRuntime>,
    z: &[Vec<f32>],
    targets: &[u32],
    h_dtype: DType,
) -> (tessl::cross_entropy::CeOutput, Vec<Vec<f32>>) {
    const LW: usize = 16;
    let (n, vocab) = (z.len(), z[0].len());
    let mut h = vec![0.0f32; n * LW];
    for r in 0..n {
        h[r * LW + r] = 1.0;
    }
    let mut w = vec![0.0f32; vocab * LW];
    for (r, zr) in z.iter().enumerate() {
        for (j, &l) in zr.iter().enumerate() {
            w[j * LW + r] = l;
        }
    }
    let ht = match h_dtype {
        DType::F32 => tensor_f32(rt, &[n, LW], &h),
        DType::BF16 => tensor_bf16(rt, &[n, LW], &h),
        other => panic!("ce hidden {other:?}"),
    };
    let wt = tensor_f32(rt, &[vocab, LW], &w);
    let rows: Vec<u32> = (0..n as u32).collect();
    let ws = CeWorkspace::new(
        rt,
        n as u32,
        LW as u32,
        vocab as u32,
        DType::F32,
        GemmOperands::ExactF32,
    )
    .expect("ws");
    let dh = tensor_f32(rt, &[n, LW], &vec![0.0; n * LW]);
    let dw = tensor_f32(rt, &[vocab, LW], &vec![0.0; vocab * LW]);
    let out = cross_entropy_rows(
        rt,
        CeHidden { rows: &ht, off: 0 },
        &wt,
        &rows,
        targets,
        Reduction::Mean,
        GemmOperands::ExactF32,
        &ws,
        Some(CeGrads {
            dh: &dh,
            dw: &dw,
            scale: 1.0,
        }),
    )
    .expect("ce");
    let dwv = dw.read_f32().expect("dw");
    let grads = (0..n).map(|r| (0..vocab).map(|j| dwv[j * LW + r]).collect()).collect();
    (out, grads)
}

fn check_ce(label: &str, z: &[Vec<f32>], targets: &[u32], out: &tessl::cross_entropy::CeOutput, grads: &[Vec<f32>]) {
    let n = z.len();
    let eff = 1.0 / n as f64;
    let mut total = 0.0;
    for r in 0..n {
        let zr = f64s(&z[r]);
        let t = targets[r] as usize;
        let (loss, g) = crate::common::ce::ce_oracle(&zr, t);
        total += loss;
        assert!(
            (out.per_row[r] - loss).abs() <= 1e-4 * loss.abs() + 1e-60,
            "{label}: row {r} loss {} want {loss}",
            out.per_row[r]
        );
        let peak = g.iter().fold(0.0f64, |m, x| m.max(x.abs())) * eff;
        let mut sum = 0.0;
        for (j, (&got, &gj)) in grads[r].iter().zip(&g).enumerate() {
            let want = gj * eff;
            let err = (f64::from(got) - want).abs();
            assert!(
                got.is_finite() && err <= 3e-5 * peak + 1e-36,
                "{label}: row {r} class {j} grad {got} want {want}"
            );
            sum += f64::from(got);
        }
        let lo = crate::common::ce::other_classes_lse(&zr, t);
        let tol = 4.0 * f64::from(f32::EPSILON) * (1.0 + lo.abs()) * peak;
        assert!(sum.abs() <= tol, "{label}: row {r} sums to {sum}, tol {tol}");
    }
    let want_loss = total / n as f64;
    assert!(
        (out.loss - want_loss).abs() <= 1e-4 * want_loss.abs() + 1e-60,
        "{label}: loss {} want {want_loss}",
        out.loss
    );
}

fn cross_entropy(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let z = vec![
        vec![0.2f32, -0.1, 0.4, 0.0, -0.3, 0.1, 0.5, -0.2],
        vec![0.3, 0.1, -0.4, 0.2, 0.0, -0.2, 0.15, 0.05],
    ];
    let targets = [1u32, 3];
    ledger.credit(
        rt,
        "ce f32",
        &["ce_gather_rows_f32", "ce_lse_update", "ce_softmax_grad"],
        || {
            let (out, grads) = ce_case(rt, &z, &targets, DType::F32);
            check_ce("ce f32", &z, &targets, &out, &grads);
        },
    );
    ledger.credit(rt, "ce bf16 gather", &["ce_gather_rows_bf16"], || {
        let (out, grads) = ce_case(rt, &z, &targets, DType::BF16);
        check_ce("ce bf16", &z, &targets, &out, &grads);
    });
}

struct Enc {
    name: &'static str,
    kernel: EncoderAttnKernel,
    head: u32,
    bf16_in: bool,
    bf16_out: bool,
    rel: f64,
}

fn encoder(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    assert!(rt.has_tensorops(), "encoder tiled kernels need TensorOps");
    let cases = [
        Enc {
            name: "encoder_attn_rows_h32_r2_g8",
            kernel: EncoderAttnKernel::Rows,
            head: 32,
            bf16_in: false,
            bf16_out: false,
            rel: 2e-5,
        },
        Enc {
            name: "encoder_attn_rows_h64_r4_g16",
            kernel: EncoderAttnKernel::Rows,
            head: 64,
            bf16_in: false,
            bf16_out: false,
            rel: 2e-5,
        },
        Enc {
            name: "encoder_attn_rows_h256_r16_g32",
            kernel: EncoderAttnKernel::Rows,
            head: 256,
            bf16_in: false,
            bf16_out: false,
            rel: 2e-5,
        },
        Enc {
            name: "encoder_attn_rows_h512_r32_g32",
            kernel: EncoderAttnKernel::Rows,
            head: 512,
            bf16_in: false,
            bf16_out: false,
            rel: 2e-5,
        },
        Enc {
            name: "encoder_attn_tiled_h32_q16_k32_sg1",
            kernel: EncoderAttnKernel::Tiled,
            head: 32,
            bf16_in: false,
            bf16_out: false,
            rel: 2e-4,
        },
        Enc {
            name: "encoder_attn_tiled_h64_q32_k32_sg2",
            kernel: EncoderAttnKernel::Tiled,
            head: 64,
            bf16_in: false,
            bf16_out: false,
            rel: 2e-4,
        },
        Enc {
            name: "encoder_attn_tiled_h256_q32_k32_sg4",
            kernel: EncoderAttnKernel::Tiled,
            head: 256,
            bf16_in: false,
            bf16_out: false,
            rel: 2e-4,
        },
        Enc {
            name: "encoder_attn_tiled_h512_q32_k32_sg8",
            kernel: EncoderAttnKernel::Tiled,
            head: 512,
            bf16_in: false,
            bf16_out: false,
            rel: 2e-4,
        },
        Enc {
            name: "encoder_attn_tiled_bf16_h64_q32_k32_sg2",
            kernel: EncoderAttnKernel::TiledBf16,
            head: 64,
            bf16_in: true,
            bf16_out: false,
            rel: 2e-2,
        },
        Enc {
            name: "encoder_attn_tiled_bf16_h256_q32_k64_sg4",
            kernel: EncoderAttnKernel::TiledBf16,
            head: 256,
            bf16_in: true,
            bf16_out: false,
            rel: 2e-2,
        },
        Enc {
            name: "encoder_attn_tiled_bf16_h512_q32_k128_sg8",
            kernel: EncoderAttnKernel::TiledBf16,
            head: 512,
            bf16_in: true,
            bf16_out: false,
            rel: 2e-2,
        },
        Enc {
            name: "encoder_attn_tiled_bf16o_h64_q32_k32_sg2",
            kernel: EncoderAttnKernel::TiledBf16Out,
            head: 64,
            bf16_in: true,
            bf16_out: true,
            rel: 2e-2,
        },
        Enc {
            name: "encoder_attn_tiled_bf16o_h256_q32_k64_sg4",
            kernel: EncoderAttnKernel::TiledBf16Out,
            head: 256,
            bf16_in: true,
            bf16_out: true,
            rel: 2e-2,
        },
        Enc {
            name: "encoder_attn_tiled_bf16o_h512_q32_k128_sg8",
            kernel: EncoderAttnKernel::TiledBf16Out,
            head: 512,
            bf16_in: true,
            bf16_out: true,
            rel: 2e-2,
        },
    ];
    for c in &cases {
        let seq = 2usize;
        let d = c.head as usize;
        let n = seq * d;
        let mut q = wave(n, 0.05, 0.1);
        let mut k = wave(n, 0.05, 0.4);
        let mut v = wave(n, 0.05, 0.8);
        if c.bf16_in {
            q = round_trip_bf16(&q);
            k = round_trip_bf16(&k);
            v = round_trip_bf16(&v);
        }
        let spec = crate::common::embedgemma2::Attn {
            b: 1,
            t: seq,
            h: 1,
            hkv: 1,
            d,
            window: 0,
            lens: vec![seq],
        };
        let want = crate::common::embedgemma2::attn_ref(&spec, &f64s(&q), &f64s(&k), &f64s(&v), None);
        let name = c.name;
        let rel = c.rel;
        let bf16_in = c.bf16_in;
        let bf16_out = c.bf16_out;
        let kernel = c.kernel;
        let head = c.head;
        ledger.credit(rt, name, &[name], || {
            let (qb, kb, vb) = if bf16_in {
                (buf_bf16(rt, &q), buf_bf16(rt, &k), buf_bf16(rt, &v))
            } else {
                (buf(rt, &q), buf(rt, &k), buf(rt, &v))
            };
            let ob = if bf16_out {
                buf_bf16(rt, &vec![0.0; n])
            } else {
                buf(rt, &vec![f32::NAN; n])
            };
            let lens = buf_u32(rt, &[seq as u32]);
            let dims = EncoderAttnDims {
                batch: 1,
                seq: seq as u32,
                heads: 1,
                heads_kv: 1,
                head_dim: head,
                window: 0,
                scale: 1.0,
            };
            ok(
                name,
                embedgemma2::encoder_attn_with(rt, &qb, &kb, &vb, &ob, &lens, dims, kernel),
            );
            let got = if bf16_out { read_bf16(&ob, n) } else { ob.read_f32() };
            close_peak(name, &got, &want, rel);
        });
    }
}

fn text_fuse(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let eps = 1e-6f32;
    let dim = 4u32;
    let rows = 2u32;
    let y = wave(8, 0.35, 0.2);
    let resid = wave(8, 0.2, 0.6);
    let w_post = wave(4, 0.1, 0.1).iter().map(|v| v + 1.0).collect::<Vec<_>>();
    let w_next = wave(4, 0.1, 0.9).iter().map(|v| v + 1.0).collect::<Vec<_>>();
    let (y64, r64, wp, wn) = (f64s(&y), f64s(&resid), f64s(&w_post), f64s(&w_next));

    ledger.credit(rt, "segment_mean_rows_f32", &["segment_mean_rows_f32"], || {
        let x = wave(8, 0.5, 0.2);
        let segs = embedgemma2::upload_segments(rt, &[(0, 2)], 2).expect("segs");
        let out = buf(rt, &[f32::NAN; 4]);
        ok(
            "segment_mean",
            embedgemma2::segment_mean_rows(rt, &buf(rt, &x), &segs, &out, 1, 2, 4),
        );
        let want: Vec<f64> = (0..4).map(|c| 0.5 * (f64::from(x[c]) + f64::from(x[4 + c]))).collect();
        close_peak("segment_mean_rows_f32", &out.read_f32(), &want, 1e-5);
    });

    ledger.credit(rt, "l2_normalize_rows_f32", &["l2_normalize_rows_f32"], || {
        let x = wave(8, 0.7, 0.4);
        let xb = buf(rt, &x);
        ok("l2", embedgemma2::l2_normalize_rows(rt, &xb, 2, 4, 4));
        let mut want = Vec::new();
        for row in f64s(&x).chunks(4) {
            let nrm = row.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
            want.extend(row.iter().map(|v| v / nrm));
        }
        close_peak("l2_normalize_rows_f32", &xb.read_f32(), &want, 1e-5);
    });

    let fused = |scale: f64| -> (Vec<f64>, Vec<f64>) {
        let mut resid_out = Vec::new();
        let mut next = Vec::new();
        for (row, rr) in y64.chunks(4).zip(r64.chunks(4)) {
            let normed = rms_row(row, &wp, f64::from(eps));
            let mixed: Vec<f64> = rr.iter().zip(&normed).map(|(r, y)| scale * (r + y)).collect();
            next.extend(rms_row(&mixed, &wn, f64::from(eps)));
            resid_out.extend(mixed);
        }
        (resid_out, next)
    };

    ledger.credit(rt, "eg2t_resid_norm_f32", &["eg2t_resid_norm_f32"], || {
        let (yb, wp, rb, wn, out) = (
            buf(rt, &y),
            buf(rt, &w_post),
            buf(rt, &resid),
            buf(rt, &w_next),
            buf(rt, &[0.0; 8]),
        );
        ok(
            "resid_f32",
            embedgemma2::resid_add_norm_f32(rt, &yb, &wp, &rb, &wn, &out, rows, dim, eps, 0.5),
        );
        let (want_r, want_o) = fused(0.5);
        close_peak("eg2t resid", &rb.read_f32(), &want_r, 1e-5);
        close_peak("eg2t_resid_norm_f32", &out.read_f32(), &want_o, 1e-5);
    });
    ledger.credit(rt, "eg2t_resid_norm_bf16", &["eg2t_resid_norm_bf16"], || {
        let (yb, wp, rb, wn, out) = (
            buf(rt, &y),
            buf(rt, &w_post),
            buf(rt, &resid),
            buf(rt, &w_next),
            buf_bf16(rt, &[0.0; 8]),
        );
        ok(
            "resid_bf16",
            embedgemma2::resid_add_norm_bf16(rt, &yb, &wp, &rb, &wn, &out, rows, dim, eps, 0.5),
        );
        let (_, want_o) = fused(0.5);
        close_peak("eg2t_resid_norm_bf16", &read_bf16(&out, 8), &want_o, 2e-2);
    });
    ledger.credit(rt, "eg2t_resid_add_bf16_copy", &["eg2t_resid_add_bf16_copy"], || {
        let (yb, wp, rb, copy) = (buf(rt, &y), buf(rt, &w_post), buf(rt, &resid), buf_bf16(rt, &[0.0; 8]));
        ok(
            "resid_copy",
            embedgemma2::resid_add_bf16_copy(rt, &yb, &wp, &rb, &copy, rows, dim, eps, 0.5),
        );
        let (want_r, _) = fused(0.5);
        close_peak("eg2t resid copy", &rb.read_f32(), &want_r, 1e-5);
        close_peak("eg2t_resid_add_bf16_copy", &read_bf16(&copy, 8), &want_r, 2e-2);
    });
    ledger.credit(rt, "eg2t_scale_rms_norm_f32", &["eg2t_scale_rms_norm_f32"], || {
        let xb = buf(rt, &y);
        let wb = buf(rt, &w_post);
        ok(
            "scale_rms",
            embedgemma2::scale_rms_norm_inplace(rt, &xb, &wb, rows, dim, eps, 0.5),
        );
        let mut want = Vec::new();
        for row in y64.chunks(4) {
            let scaled: Vec<f64> = row.iter().map(|v| v * 0.5).collect();
            want.extend(rms_row(&scaled, &wp, f64::from(eps)));
        }
        close_peak("eg2t_scale_rms_norm_f32", &xb.read_f32(), &want, 1e-5);
    });
}

fn audio(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let eps = 1e-5f32;
    let frames = [4usize];
    let mels = 4usize;
    let channels = 4usize;
    ledger.credit(rt, "eg2a_sub_conv0_ln_relu", &["eg2a_sub_conv0_ln_relu"], || {
        let table = ClipTable::new(rt, &frames).expect("clips");
        let feats = wave(frames[0] * mels, 0.4, 0.2);
        let w = wave(channels * 9, 0.2, 0.5);
        let gamma = wave(channels, 0.1, 0.3).iter().map(|v| v + 1.0).collect::<Vec<_>>();
        let mask = vec![1u32; frames[0]];
        let mels1 = mels.div_ceil(2);
        let n_out = table.mid_rows * mels1 * channels;
        let out = buf(rt, &vec![f32::NAN; n_out]);
        ok(
            "conv0",
            embedgemma2_audio::sub_conv0_ln_relu(
                rt,
                &buf(rt, &feats),
                &buf_u32(rt, &mask),
                &buf(rt, &w),
                &buf(rt, &gamma),
                &out,
                &table,
                mels as u32,
                channels as u32,
                eps,
            ),
        );
        let (want, (_, to, fo), _) = audio_host::conv_layer(
            &f64s(&feats),
            (1, frames[0], mels),
            &vec![true; frames[0]],
            &f64s(&w),
            &f64s(&gamma),
            f64::from(eps),
        );
        let got = out.read_f32();
        let mut g = Vec::new();
        let mut e = Vec::new();
        for ti in 0..to {
            for fi in 0..fo {
                for ch in 0..channels {
                    g.push(got[(ti * mels1 + fi) * channels + ch]);
                    e.push(want[(ch * to + ti) * fo + fi]);
                }
            }
        }
        close_peak("eg2a_sub_conv0_ln_relu", &g, &e, 2e-4);
    });

    ledger.credit(rt, "eg2a_sub_conv1_ln_relu", &["eg2a_sub_conv1_ln_relu"], || {
        let frames = [8usize];
        let table = ClipTable::new(rt, &frames).expect("clips");
        let (mels1, c0, c1) = (4usize, 4, 4);
        let mid: Vec<f32> = wave(table.mid_rows * mels1 * c0, 0.5, 0.2)
            .iter()
            .map(|v| v.abs())
            .collect();
        let w_ck = wave(c1 * c0 * 9, 0.15, 0.4);
        let mut w_k = vec![0.0f32; 9 * c0 * c1];
        for co in 0..c1 {
            for ci in 0..c0 {
                for kh in 0..3 {
                    for kw in 0..3 {
                        w_k[((kh * 3 + kw) * c0 + ci) * c1 + co] = w_ck[((co * c0 + ci) * 3 + kh) * 3 + kw];
                    }
                }
            }
        }
        let gamma = wave(c1, 0.1, 0.7).iter().map(|v| v + 1.0).collect::<Vec<_>>();
        let mels2 = mels1.div_ceil(2);
        let n_out = table.rows * mels2 * c1;
        let out = buf(rt, &vec![f32::NAN; n_out]);
        let mask = vec![1u32; frames[0]];
        ok(
            "conv1",
            embedgemma2_audio::sub_conv1_ln_relu(
                rt,
                &buf(rt, &mid),
                &buf_u32(rt, &mask),
                &buf(rt, &w_k),
                &buf(rt, &gamma),
                &out,
                &table,
                mels1 as u32,
                c0 as u32,
                c1 as u32,
                eps,
            ),
        );
        let c = &table.clips[0];
        let mut x = vec![0.0; c0 * c.mid_rows * mels1];
        for ti in 0..c.mid_rows {
            for fi in 0..mels1 {
                for ci in 0..c0 {
                    x[(ci * c.mid_rows + ti) * mels1 + fi] = f64::from(mid[(ti * mels1 + fi) * c0 + ci]);
                }
            }
        }
        let m1: Vec<bool> = mask.iter().step_by(2).map(|_| true).collect();
        let (want, (_, t2, f2), _) = audio_host::conv_layer(
            &x,
            (c0, c.mid_rows, mels1),
            &m1,
            &f64s(&w_ck),
            &f64s(&gamma),
            f64::from(eps),
        );
        let got = out.read_f32();
        let mut g = Vec::new();
        let mut e = Vec::new();
        for ti in 0..t2 {
            for fi in 0..f2 {
                for ch in 0..c1 {
                    g.push(got[ti * mels2 * c1 + fi * c1 + ch]);
                    e.push(want[(ch * t2 + ti) * f2 + fi]);
                }
            }
        }
        close_peak("eg2a_sub_conv1_ln_relu", &g, &e, 2e-4);
    });

    ledger.credit(rt, "eg2a_rms_norm", &["eg2a_rms_norm"], || {
        let x = wave(8, 0.4, 0.2);
        let w = wave(4, 0.1, 0.5).iter().map(|v| v + 1.0).collect::<Vec<_>>();
        let (xb, wb, out) = (buf(rt, &x), buf(rt, &w), buf(rt, &[0.0; 8]));
        let spec = RmsNorm::new(eps, Some(&wb));
        ok("audio rms", embedgemma2_audio::rms_norm(rt, &xb, &out, 2, 4, &spec));
        let want: Vec<f64> = f64s(&x)
            .chunks(4)
            .flat_map(|row| audio_host::rms_eps(row, Some(&f64s(&w)), f64::from(eps)))
            .collect();
        close_peak("eg2a_rms_norm", &out.read_f32(), &want, 1e-5);
    });

    ledger.credit(rt, "eg2a_ew", &["eg2a_ew"], || {
        let x = wave(8, 0.6, 0.3);
        let (xb, out) = (buf(rt, &x), buf(rt, &[0.0; 8]));
        let spec = Ew {
            pre: Clamp::NONE,
            silu: true,
            post: Clamp::NONE,
        };
        ok("ew", embedgemma2_audio::ew(rt, &xb, &out, 8, &spec));
        let want: Vec<f64> = x.iter().map(|&v| silu(f64::from(v))).collect();
        close_peak("eg2a_ew", &out.read_f32(), &want, 1e-5);
    });

    ledger.credit(rt, "eg2a_lconv", &["eg2a_lconv"], || {
        let frames = [8usize];
        let table = ClipTable::new(rt, &frames).expect("clips");
        let (hidden, taps) = (4usize, 3usize);
        let rows = table.rows;
        let x = wave(rows * 2 * hidden, 0.3, 0.2);
        let w = wave(hidden * taps, 0.2, 0.6);
        let nw = wave(hidden, 0.1, 0.1).iter().map(|v| v + 1.0).collect::<Vec<_>>();
        let (xb, wb, nb, out) = (
            buf(rt, &x),
            buf(rt, &w),
            buf(rt, &nw),
            buf(rt, &vec![f32::NAN; rows * hidden]),
        );
        ok(
            "lconv",
            embedgemma2_audio::lconv(
                rt,
                &xb,
                &wb,
                &out,
                &table,
                hidden as u32,
                taps as u32,
                &Lconv {
                    eps,
                    norm_weight: &nb,
                    glu: Clamp::NONE,
                    pre: Clamp::NONE,
                    post: Clamp::NONE,
                },
            ),
        );
        let glu = |row: usize, ch: usize| {
            let a = f64::from(x[row * 2 * hidden + ch]);
            let b = f64::from(x[row * 2 * hidden + hidden + ch]);
            a * audio_host::sigmoid(b)
        };
        let mut want = vec![0.0; rows * hidden];
        for t in 0..rows {
            let conv: Vec<f64> = (0..hidden)
                .map(|ch| {
                    (0..taps)
                        .filter_map(|j| {
                            let src = t as i64 + j as i64 - (taps as i64 - 1);
                            (src >= 0).then(|| f64::from(w[ch * taps + j]) * glu(src as usize, ch))
                        })
                        .sum()
                })
                .collect();
            let normed = audio_host::rms_eps(&conv, Some(&f64s(&nw)), f64::from(eps));
            for (ch, y) in normed.into_iter().enumerate() {
                want[t * hidden + ch] = silu(y);
            }
        }
        close_peak("eg2a_lconv", &out.read_f32(), &want, 2e-4);
    });

    ledger.credit(rt, "eg2a_attn", &["eg2a_attn"], || {
        let frames = [8usize];
        let table = ClipTable::new(rt, &frames).expect("clips");
        let (heads, head_dim, chunk, past) = (2usize, 8usize, 4u32, 3u32);
        let hidden = heads * head_dim;
        let rows = table.rows;
        let ctx = (chunk + past) as usize;
        let q = wave(rows * hidden, 0.05, 0.1);
        let k = wave(rows * hidden, 0.05, 0.4);
        let v = wave(rows * hidden, 0.05, 0.7);
        let mut proj = vec![0.0f64; hidden * hidden];
        for i in 0..hidden {
            proj[i * hidden + i] = 0.01;
        }
        let rel = audio_host::relative_keys(hidden, ctx, &proj);
        let per = vec![0.0f64; head_dim];
        let qm = audio_host::q_mult(&per);
        let omask = vec![true; rows];
        let cap = 5.0f32;
        let invalid = -1.0e9f32;
        let spec = AttentionSpec {
            heads: heads as u32,
            head_dim: head_dim as u32,
            chunk,
            past,
            k_scale: ((1.0f64 + 1.0f64.exp()).ln() / 2.0f64.ln()) as f32,
            cap,
            invalid,
            q: Clamp::NONE,
            k: Clamp::NONE,
            v: Clamp::NONE,
            out: Clamp::NONE,
            stride: hidden as u32,
            q_off: 0,
            k_off: 0,
            v_off: 0,
        };
        let (qb, kb, vb) = (buf(rt, &q), buf(rt, &k), buf(rt, &v));
        let rel_b = buf(rt, &rel.iter().map(|&x| x as f32).collect::<Vec<_>>());
        let qm_b = buf(rt, &qm.iter().map(|&x| x as f32).collect::<Vec<_>>());
        let mask = buf_u32(rt, &vec![1u32; rows]);
        let out = buf(rt, &vec![f32::NAN; rows * hidden]);
        ok(
            "attn",
            embedgemma2_audio::attention(rt, &qb, &kb, &vb, &rel_b, &qm_b, &mask, &out, &table, &spec),
        );
        let want = audio_host::attention_ref(
            &audio_host::AttnSpec {
                heads,
                head_dim,
                chunk: chunk as usize,
                left: past as usize + 1,
                cap: f64::from(cap),
                invalid: f64::from(invalid),
            },
            &f64s(&q),
            &f64s(&k),
            &f64s(&v),
            &rel,
            &qm,
            &omask,
        );
        // f32 k_scale against the f64 reference. 2e-4 of peak still rejects a wrong kernel.
        close_peak("eg2a_attn", &out.read_f32(), &want, 2e-4);
    });
}

fn vision_rope_want(
    proj: &[f32],
    positions: &[i32],
    qw: &[f32],
    kw: &[f32],
    theta: f32,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let d = 4usize;
    let width = 12usize;
    let (qw64, kw64) = (f64s(qw), f64s(kw));
    let (mut q, mut k, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..2 {
        let row = f64s(&proj[r * width..(r + 1) * width]);
        let (px, py) = (f64::from(positions[2 * r]), f64::from(positions[2 * r + 1]));
        q.extend(rope_axial(
            &crate::common::embedgemma2::rms(&row[..d], Some(&qw64)),
            px,
            py,
            f64::from(theta),
        ));
        k.extend(rope_axial(
            &crate::common::embedgemma2::rms(&row[d..2 * d], Some(&kw64)),
            px,
            py,
            f64::from(theta),
        ));
        v.extend(crate::common::embedgemma2::rms(&row[2 * d..3 * d], None));
    }
    (q, k, v)
}

fn vision(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let theta = 100.0f32;
    let eps = 1e-6f32;
    let positions = [0i32, 1, -1, -1];
    let mut proj = wave(24, 0.3, 0.2);
    for v in &mut proj[12..] {
        *v = 0.0;
    }
    let qw = wave(4, 0.1, 0.3).iter().map(|v| v + 1.0).collect::<Vec<_>>();
    let kw = wave(4, 0.1, 0.8).iter().map(|v| v + 1.0).collect::<Vec<_>>();
    let (want_q, want_k, want_v) = vision_rope_want(&proj, &positions, &qw, &kw, theta);
    let shape = AxialQkvShape {
        batch: 1,
        seq: 2,
        q_heads: 1,
        kv_heads: 1,
        head_dim: 4,
    };
    let pos_bits: Vec<u32> = positions.iter().map(|p| *p as u32).collect();

    ledger.credit(rt, "vision_qkv_norm_rope_f32", &["vision_qkv_norm_rope_f32"], || {
        let pb = buf(rt, &proj);
        let (qb, kb) = (buf(rt, &qw), buf(rt, &kw));
        let pos = buf_u32(rt, &pos_bits);
        let q_out = buf(rt, &[3.0; 8]);
        let k_cache = buf(rt, &[3.0; 8]);
        let v_cache = buf(rt, &[3.0; 8]);
        ok(
            "vision rope",
            embedgemma2_vision::axial_qkv_norm_rope(
                rt,
                &shape,
                Cols::dense(&pb, 12),
                4,
                8,
                &qb,
                &kb,
                &AttnTargets {
                    q_out: &q_out,
                    k_cache: &k_cache,
                    v_cache: &v_cache,
                },
                &pos,
                theta,
                eps,
            ),
        );
        let gap = nn::rope_inv_freq(1, 2, theta)
            .iter()
            .enumerate()
            .map(|(p, &f)| (f64::from(f) - 1.0 / f64::from(theta).powf(2.0 * p as f64 / 2.0)).abs())
            .fold(0.0, f64::max);
        let rel = 2e-5 + gap;
        close_peak("vision q", &q_out.read_f32()[..4], &want_q[..4], rel);
        close_peak("vision k", &k_cache.read_f32()[..4], &want_k[..4], rel);
        close_peak("vision v", &v_cache.read_f32()[..4], &want_v[..4], 2e-5);
        bits_eq("vision q pad", &q_out.read_f32()[4..8], &[0.0; 4]);
        bits_eq("vision k pad", &k_cache.read_f32()[4..8], &[0.0; 4]);
        bits_eq("vision v pad", &v_cache.read_f32()[4..8], &[0.0; 4]);
    });

    ledger.credit(rt, "vision_qkv_norm_rope_bf16", &["vision_qkv_norm_rope_bf16"], || {
        let pb = buf(rt, &proj);
        let (qb, kb) = (buf(rt, &qw), buf(rt, &kw));
        let pos = buf_u32(rt, &pos_bits);
        let q_out = buf_bf16(rt, &[3.0; 8]);
        let k_cache = buf_bf16(rt, &[3.0; 8]);
        let v_cache = buf_bf16(rt, &[3.0; 8]);
        ok(
            "vision rope bf16",
            embedgemma2_vision::axial_qkv_norm_rope_bf16(
                rt,
                &shape,
                Cols::dense(&pb, 12),
                4,
                8,
                &qb,
                &kb,
                &AttnTargets {
                    q_out: &q_out,
                    k_cache: &k_cache,
                    v_cache: &v_cache,
                },
                &pos,
                theta,
                eps,
            ),
        );
        close_peak("vision q bf16", &read_bf16(&q_out, 4), &want_q[..4], 2e-2);
        close_peak("vision k bf16", &read_bf16(&k_cache, 4), &want_k[..4], 2e-2);
        close_peak("vision v bf16", &read_bf16(&v_cache, 4), &want_v[..4], 2e-2);
        bits_eq("vision q pad bf16", &read_bf16(&q_out, 8)[4..], &[0.0; 4]);
        bits_eq("vision k pad bf16", &read_bf16(&k_cache, 8)[4..], &[0.0; 4]);
        bits_eq("vision v pad bf16", &read_bf16(&v_cache, 8)[4..], &[0.0; 4]);
    });

    ledger.credit(rt, "vision_pos_embed_add_f32", &["vision_pos_embed_add_f32"], || {
        let dim = 4usize;
        let table_len = 4usize;
        let mut table = vec![0.0f32; 2 * table_len * dim];
        for (i, v) in table.iter_mut().enumerate() {
            *v = [0.5, 1.0, 1.5, 2.0][i % 4];
        }
        let h0 = vec![0.25f32; 8];
        let pos = [1i32, 2, -1, -1];
        let (hb, tb, pb) = (
            buf(rt, &h0),
            buf_bf16(rt, &table),
            buf_u32(rt, &pos.iter().map(|p| *p as u32).collect::<Vec<_>>()),
        );
        ok(
            "pos embed",
            embedgemma2_vision::patch_position_embed_add(rt, &hb, &tb, &pb, 2, dim as u32, table_len as u32),
        );
        let widened: Vec<f32> = table
            .iter()
            .map(|&v| bf16_bits_to_f32(f32_to_bf16_bits_gpu(v)))
            .collect();
        let mut want = h0.clone();
        let (x, y) = (1usize, 2usize);
        for d in 0..dim {
            want[d] += widened[x * dim + d] + widened[(table_len + y) * dim + d];
        }
        want[dim..2 * dim].fill(0.0);
        // Exact bf16 values, so the real row is a dyadic sum and the pad row is +0.
        bits_eq("vision_pos_embed_add_f32", &hb.read_f32(), &want);
    });

    ledger.credit(rt, "vision_pool_members_f32", &["vision_pool_members_f32"], || {
        let positions = [0i32, 0, 1, 0, 0, 1, 1, 1];
        let dim = 4usize;
        let h = wave(4 * dim, 0.4, 0.2);
        let plan = embedgemma2_vision::plan_pooling(&positions, 4, 2).expect("pool plan");
        let (want, _) = pool_dense(&f64s(&h), &positions, 2, dim);
        let out_rows = plan.cells.len();
        let out = buf(rt, &vec![f32::NAN; out_rows * dim]);
        ok(
            "pool",
            embedgemma2_vision::pool_by_members(
                rt,
                &buf(rt, &h),
                &buf_u32(rt, &plan.offsets),
                &buf_u32(rt, &plan.members),
                &out,
                out_rows as u32,
                4,
                dim as u32,
                plan.members.len() as u32,
                0.25,
                1.0,
            ),
        );
        close_peak("vision_pool_members_f32", &out.read_f32(), &want, 1e-5);
    });
}

/// The host-side GDN inputs, in kernel order: q, k, v, g, beta, d_o.
type GdnInputs = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

fn gdn_case(seed: u64) -> GdnInputs {
    let n = 64usize;
    let q = random_f32(n * 128, seed);
    let k = random_f32(n * 128, seed + 1);
    let v = random_f32(n * 16, seed + 2);
    let g = random_f32(n, seed + 3)
        .iter()
        .map(|&x| -0.75 * (x + 1.0) - 1e-3)
        .collect();
    let beta = random_f32(n, seed + 4).iter().map(|&x| 0.5 + 0.45 * x).collect();
    let d_o = random_f32(n * 16, seed + 6);
    (q, k, v, g, beta, d_o)
}

fn gdn_check(label: &str, got: &[f32], want: &[f64]) {
    close_peak(label, got, want, 1e-4);
}

fn gdn(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let dims = GdnTrainDims {
        batch: 1,
        seq: 64,
        heads: 1,
        v_dim: 16,
    };
    let shape = GdnShape {
        b: 1,
        t: 64,
        h: 1,
        dk: 128,
        dv: 16,
    };
    let (q, k, v, g, beta, d_o) = gdn_case(11);
    let q64 = f64s(&q);
    let k64 = f64s(&k);
    let v64 = f64s(&v);
    let g64 = f64s(&g);
    let b64 = f64s(&beta);
    let d64 = f64s(&d_o);
    let inputs = GdnIn {
        s: shape,
        q: &q64,
        k: &k64,
        v: &v64,
        g: &g64,
        beta: &b64,
        s0: None,
    };
    let (want_o, _) = gdn_train::gdn_train_f64(&inputs);
    let grads = gdn_train::gdn_train_bwd_f64(&inputs, &d64, None);
    let seq_ws = GdnTrainWorkspace::new_for(rt, dims, GdnTrainKernel::Sequential).expect("seq ws");
    let chunk_ws = GdnTrainWorkspace::new_for(rt, dims, GdnTrainKernel::Chunked).expect("chunk ws");

    let upload = |data: &[f32], shape: &[usize]| tensor_f32(rt, shape, data);
    let q_t = upload(&q, &[1, 64, 1, 128]);
    let k_t = upload(&k, &[1, 64, 1, 128]);
    let v_t = upload(&v, &[1, 64, 1, 16]);
    let g_t = upload(&g, &[1, 64, 1]);
    let beta_t = upload(&beta, &[1, 64, 1]);
    let x = GdnTrainInputs {
        q: &q_t,
        k: &k_t,
        v: &v_t,
        g: &g_t,
        beta: &beta_t,
        s0: None,
    };

    let run_fwd = |kernel: GdnTrainKernel, ckpt: &Tensor, o: &Tensor| {
        ok("gdn fwd", gdn_train_forward_with(kernel, rt, dims, x, o, None, ckpt));
        rt.synchronize().expect("gdn sync");
        gdn_check("gdn o", &o.read_f32().expect("o"), &want_o);
    };

    ledger.credit(rt, "gdn_train_fwd", &["gdn_train_fwd"], || {
        let ckpt = rt.alloc_tensor_f32(&dims.checkpoint_shape()).expect("ckpt");
        let o = rt.alloc_tensor_f32(&[1, 64, 1, 16]).expect("o");
        run_fwd(GdnTrainKernel::Sequential, &ckpt, &o);
    });
    // The sequential forward above is inside its own credit. Run it again so
    // the backward reads a checkpoint this call produced, then credit only
    // the backward names.
    let seq_ckpt = rt.alloc_tensor_f32(&dims.checkpoint_shape()).expect("ckpt");
    let seq_o = rt.alloc_tensor_f32(&[1, 64, 1, 16]).expect("o");
    run_fwd(GdnTrainKernel::Sequential, &seq_ckpt, &seq_o);

    ledger.credit(rt, "gdn seq bwd", &["gdn_train_bwd", "gdn_train_bwd_finish"], || {
        let d_o_t = upload(&d_o, &[1, 64, 1, 16]);
        let dq = rt.alloc_tensor_f32(&[1, 64, 1, 128]).expect("dq");
        let dk = rt.alloc_tensor_f32(&[1, 64, 1, 128]).expect("dk");
        let dv = rt.alloc_tensor_f32(&[1, 64, 1, 16]).expect("dv");
        let dg = rt.alloc_tensor_f32(&[1, 64, 1]).expect("dg");
        let dbeta = rt.alloc_tensor_f32(&[1, 64, 1]).expect("dbeta");
        ok(
            "gdn bwd",
            gdn_train_backward(
                rt,
                dims,
                x,
                &seq_ckpt,
                &d_o_t,
                None,
                &seq_ws,
                GdnTrainGrads {
                    dq: &dq,
                    dk: &dk,
                    dv: &dv,
                    dg: &dg,
                    dbeta: &dbeta,
                    ds0: None,
                },
            ),
        );
        rt.synchronize().expect("bwd sync");
        gdn_check("dq", &dq.read_f32().expect("dq"), &grads.dq);
        gdn_check("dk", &dk.read_f32().expect("dk"), &grads.dk);
        gdn_check("dv", &dv.read_f32().expect("dv"), &grads.dv);
        gdn_check("dg", &dg.read_f32().expect("dg"), &grads.dg);
        gdn_check("dbeta", &dbeta.read_f32().expect("db"), &grads.dbeta);
    });

    ledger.credit(
        rt,
        "gdn chunk fwd",
        &["gdn_train_c_norm", "gdn_train_c_prep", "gdn_train_c_fwd"],
        || {
            let ckpt = rt.alloc_tensor_f32(&dims.checkpoint_shape()).expect("ckpt");
            let o = rt.alloc_tensor_f32(&[1, 64, 1, 16]).expect("o");
            run_fwd(GdnTrainKernel::Chunked, &ckpt, &o);
        },
    );
    let chunk_ckpt = rt.alloc_tensor_f32(&dims.checkpoint_shape()).expect("ckpt");
    let chunk_o = rt.alloc_tensor_f32(&[1, 64, 1, 16]).expect("o");
    run_fwd(GdnTrainKernel::Chunked, &chunk_ckpt, &chunk_o);

    ledger.credit(
        rt,
        "gdn chunk bwd",
        &[
            "gdn_train_c_bwd_a",
            "gdn_train_c_mats",
            "gdn_train_c_gates",
            "gdn_train_c_dqk",
            "gdn_train_c_finish",
        ],
        || {
            let d_o_t = upload(&d_o, &[1, 64, 1, 16]);
            let dq = rt.alloc_tensor_f32(&[1, 64, 1, 128]).expect("dq");
            let dk = rt.alloc_tensor_f32(&[1, 64, 1, 128]).expect("dk");
            let dv = rt.alloc_tensor_f32(&[1, 64, 1, 16]).expect("dv");
            let dg = rt.alloc_tensor_f32(&[1, 64, 1]).expect("dg");
            let dbeta = rt.alloc_tensor_f32(&[1, 64, 1]).expect("dbeta");
            ok(
                "gdn chunk bwd",
                gdn_train_backward(
                    rt,
                    dims,
                    x,
                    &chunk_ckpt,
                    &d_o_t,
                    None,
                    &chunk_ws,
                    GdnTrainGrads {
                        dq: &dq,
                        dk: &dk,
                        dv: &dv,
                        dg: &dg,
                        dbeta: &dbeta,
                        ds0: None,
                    },
                ),
            );
            rt.synchronize().expect("bwd sync");
            gdn_check("cq", &dq.read_f32().expect("dq"), &grads.dq);
            gdn_check("ck", &dk.read_f32().expect("dk"), &grads.dk);
            gdn_check("cv", &dv.read_f32().expect("dv"), &grads.dv);
            gdn_check("cg", &dg.read_f32().expect("dg"), &grads.dg);
            gdn_check("cb", &dbeta.read_f32().expect("db"), &grads.dbeta);
        },
    );
}

fn rope_ref(row: &mut [f32], weight: &[f32], eps: f32, rotary: usize, pos: usize, theta: f32) {
    let d = row.len();
    let ss: f64 = row.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    let inv = 1.0 / (ss / d as f64 + f64::from(eps)).sqrt();
    for (x, w) in row.iter_mut().zip(weight) {
        *x = (f64::from(*x) * inv * f64::from(*w)) as f32;
    }
    let half = d / 2;
    for i in 0..rotary / 2 {
        let e = (2 * i) as f32 / d as f32;
        let inv_freq = 1.0f32 / (f64::from(theta).powf(f64::from(e)) as f32);
        let angle = f64::from(pos as f32 * inv_freq);
        let (c, s) = (angle.cos(), angle.sin());
        let (x0, x1) = (f64::from(row[i]), f64::from(row[i + half]));
        row[i] = (x0 * c - x1 * s) as f32;
        row[i + half] = (x0 * s + x1 * c) as f32;
    }
}

fn rope_expect(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    qw: &[f32],
    kw: &[f32],
    vw: &[f32],
    pos: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (t, hq, hkv, d, rotary, theta, eps) = (2usize, 2, 1, 32, 32, 10_000.0f32, 1e-6f32);
    let (mut q, mut k, mut v) = (q.to_vec(), k.to_vec(), v.to_vec());
    for ti in 0..t {
        for h in 0..hq {
            let row = &mut q[(ti * hq + h) * d..(ti * hq + h + 1) * d];
            rope_ref(row, qw, eps, rotary, pos + ti, theta);
        }
        for h in 0..hkv {
            let row = &mut k[(ti * hkv + h) * d..(ti * hkv + h + 1) * d];
            rope_ref(row, kw, eps, rotary, pos + ti, theta);
            let row = &mut v[(ti * hkv + h) * d..(ti * hkv + h + 1) * d];
            rope_ref(row, vw, eps, 0, pos + ti, theta);
        }
    }
    (q, k, v)
}

fn rope(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let (t, hq, hkv, d) = (2u32, 2u32, 1u32, 32u32);
    let q = wave((t * hq * d) as usize, 0.3, 0.1);
    let k = wave((t * hkv * d) as usize, 0.3, 0.4);
    let v = wave((t * hkv * d) as usize, 0.3, 0.7);
    let qw = wave(d as usize, 0.1, 0.2).iter().map(|x| x + 1.0).collect::<Vec<_>>();
    let kw = wave(d as usize, 0.1, 0.5).iter().map(|x| x + 1.0).collect::<Vec<_>>();
    let vw = wave(d as usize, 0.1, 0.8).iter().map(|x| x + 1.0).collect::<Vec<_>>();
    let dims = QkvRopeDims {
        t,
        heads_q: hq,
        heads_kv: hkv,
        head_dim: d,
        rotary_dim: d,
        theta: 10_000.0,
        eps: 1e-6,
    };
    let pos = 7u32;
    let (eq, ek, ev) = rope_expect(&q, &k, &v, &qw, &kw, &vw, pos as usize);

    let launch = |variant: QkvRopeVariant, pos_buf: Option<&GpuBuffer>, store: Option<KvStoreTarget<'_>>| {
        let (qb, kb, vb) = (buf(rt, &q), buf(rt, &k), buf(rt, &v));
        let (qwb, kwb, vwb) = (buf(rt, &qw), buf(rt, &kw), buf(rt, &vw));
        ok(
            "rope",
            nn::rms_qkv_rope(
                rt,
                variant,
                QkvBuffers {
                    q: &qb,
                    k: &kb,
                    v: &vb,
                    q_weight: &qwb,
                    k_weight: &kwb,
                    v_weight: &vwb,
                },
                dims,
                pos,
                pos_buf,
                store,
                false,
            ),
        );
        close_f32("q", &qb.read_f32(), &eq);
        close_f32("k", &kb.read_f32(), &ek);
        close_f32("v", &vb.read_f32(), &ev);
        (qb, kb, vb)
    };

    ledger.credit(rt, "rms_qkv_rope", &["rms_qkv_rope"], || {
        launch(QkvRopeVariant::PosConst, None, None);
    });
    ledger.credit(rt, "rms_qkv_rope_posbuf", &["rms_qkv_rope_posbuf"], || {
        let pb = buf_u32(rt, &[pos]);
        launch(QkvRopeVariant::PosBuffer, Some(&pb), None);
    });
    ledger.credit(rt, "rms_qkv_rope_kv_store", &["rms_qkv_rope_kv_store"], || {
        let pb = buf_u32(rt, &[pos]);
        let n = t * hkv * d;
        let off = 4u32;
        let cap = off + n;
        let (dk, dv, ob) = (
            buf(rt, &vec![9.0; cap as usize]),
            buf(rt, &vec![9.0; cap as usize]),
            buf_u32(rt, &[off]),
        );
        let (_q, k, v) = launch(
            QkvRopeVariant::PosBufferKvStore,
            Some(&pb),
            Some(KvStoreTarget {
                dst_k: &dk,
                dst_v: &dv,
                dst_offset: &ob,
                capacity: cap,
            }),
        );
        let ks = k.read_f32();
        let vs = v.read_f32();
        bits_eq("kv k prefix", &dk.read_f32()[..off as usize], &vec![9.0; off as usize]);
        close_f32("kv k stored", &dk.read_f32()[off as usize..(off + n) as usize], &ks);
        close_f32("kv v stored", &dv.read_f32()[off as usize..(off + n) as usize], &vs);
    });
}

fn close_f32(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what}[{i}]: {g}");
        let tol = 3e-5 * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "{what}[{i}]: got {g} want {w}");
    }
}

fn kv(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    ledger.credit(rt, "kv_store_timestep", &["kv_store_timestep"], || {
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let (sb, db, ob) = (buf(rt, &src), buf(rt, &[-1.0; 8]), buf_u32(rt, &[2]));
        ok("kv", nn::kv_store_timestep(rt, &sb, &db, &ob, 4, 8));
        let mut want = vec![-1.0; 8];
        want[2..6].copy_from_slice(&src);
        bits_eq("kv_store_timestep", &db.read_f32(), &want);
    });
    ledger.credit(rt, "kv_store_timestep_pair", &["kv_store_timestep_pair"], || {
        let k = [1.0f32, 2.0, 3.0, 4.0];
        let v = [5.0f32, 6.0, 7.0, 8.0];
        let (kb, vb, dk, dv, ob) = (
            buf(rt, &k),
            buf(rt, &v),
            buf(rt, &[-1.0; 8]),
            buf(rt, &[-1.0; 8]),
            buf_u32(rt, &[2]),
        );
        ok("kv pair", nn::kv_store_timestep_pair(rt, &kb, &vb, &dk, &dv, &ob, 4, 8));
        let mut wk = vec![-1.0; 8];
        let mut wv = vec![-1.0; 8];
        wk[2..6].copy_from_slice(&k);
        wv[2..6].copy_from_slice(&v);
        bits_eq("kv pair k", &dk.read_f32(), &wk);
        bits_eq("kv pair v", &dv.read_f32(), &wv);
    });
    ledger.credit(rt, "kv_ring_densify", &["kv_ring_densify"], || {
        let src: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let (sb, db, filled, start) = (buf(rt, &src), buf(rt, &[-1.0; 8]), buf_u32(rt, &[4]), buf_u32(rt, &[1]));
        ok("ring", nn::kv_ring_densify(rt, &sb, &db, &filled, &start, 2, 4));
        let mut want = vec![0.0; 8];
        for t in 0..4 {
            for e in 0..2 {
                want[t * 2 + e] = src[((1 + t) % 4) * 2 + e];
            }
        }
        bits_eq("kv_ring_densify", &db.read_f32(), &want);
    });
}
