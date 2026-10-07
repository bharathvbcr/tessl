//! `Qwen35Model::begin` / `Staged`: a prefill run a few layers at a time.
//!
//! `forward` is `begin` + `advance_to(last)` + the head, so the two share one
//! layer loop by construction. What is checked: the final norm read after
//! every layer (an early exit) against the same norm of `forward`'s traced
//! residual on the host, at both precisions; at the last layer, **bit for
//! bit** against `forward`'s own final-norm output (F32); and advancing in
//! one step against advancing a layer at a time, bit for bit. `load_tower` must refuse `forward` (it has no head) and
//! still run every staged step. Runs on the tiny two-layer checkpoint in
//! `tests/fixtures/qwen35_train/` (one GDN layer, one attention layer).

mod common;

use std::path::{Path, PathBuf};

use common::with_gpu;
use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::safetensors::SafeTensors;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train")
}

fn ids(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 37 + 11) % 64).collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn staged_final_norm_matches_forward_at_every_layer() {
    let cfg = Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap();
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    let h = cfg.hidden as usize;
    with_gpu(|rt| {
        for precision in [Precision::F32, Precision::Bf16] {
            let full = Qwen35Model::load(rt, &st, "model.", cfg.clone(), precision).unwrap();
            let tower = Qwen35Model::load_tower(rt, &st, "model.", cfg.clone(), precision).unwrap();
            for t in [1usize, 5, 70] {
                let ids = ids(t);
                let want = full.forward(&ids, true).unwrap().trace;
                assert_eq!(want.len(), cfg.layers.len() + 1);
                for model in [&full, &tower] {
                    let mut s = model.begin(&ids).unwrap();
                    assert_eq!((s.tokens() as usize, s.layers_done()), (t, 0));
                    let out = rt.alloc_tensor_f32(&[t, h]).unwrap();
                    for layer in 1..=cfg.layers.len() {
                        s.advance_to(layer).unwrap();
                        s.final_norm_f32(&out).unwrap();
                        rt.synchronize().unwrap();
                        // The residual itself is not exposed; its final norm
                        // is, and at the last layer it is forward's own
                        // final-norm output (F32: the same kernel and dtype).
                        let normed = out.buffer.read_f32()[..t * h].to_vec();
                        let resid = &want[layer - 1];
                        let host = host_norm(resid, &read_norm_w(&st), t, h, cfg.rms_norm_eps);
                        let err = normed
                            .iter()
                            .zip(&host)
                            .map(|(a, b)| (f64::from(*a) - b).abs())
                            .fold(0.0, f64::max);
                        let peak = host.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                        assert!(err <= 2e-6 * peak.max(1.0), "{precision:?} t={t} layer {layer}: {err:.3e}");
                        if layer == cfg.layers.len() && precision == Precision::F32 {
                            assert_eq!(bits(&normed), bits(&want[layer]), "t={t}: final norm vs forward's");
                        }
                    }
                }
            }
            // Advancing in one step and in several gives the same bits.
            let ids = ids(33);
            let (one, many) = (rt.alloc_tensor_f32(&[33, h]).unwrap(), rt.alloc_tensor_f32(&[33, h]).unwrap());
            let mut a = tower.begin(&ids).unwrap();
            a.advance_to(cfg.layers.len()).unwrap();
            a.final_norm_f32(&one).unwrap();
            let mut b = tower.begin(&ids).unwrap();
            for l in 0..=cfg.layers.len() {
                b.advance_to(l).unwrap();
            }
            b.final_norm_f32(&many).unwrap();
            rt.synchronize().unwrap();
            assert_eq!(bits(&one.buffer.read_f32()[..33 * h]), bits(&many.buffer.read_f32()[..33 * h]));
        }
    });
}

#[test]
fn staged_refusals() {
    let cfg = Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap();
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    let n = cfg.layers.len();
    with_gpu(|rt| {
        let tower = Qwen35Model::load_tower(rt, &st, "model.", cfg.clone(), Precision::Bf16).unwrap();
        assert!(tower.forward(&ids(4), false).is_err(), "forward on a bf16 tower has no head");
        // F32 keeps the embedding as its head, so a tower still runs forward.
        let f32_tower = Qwen35Model::load_tower(rt, &st, "model.", cfg.clone(), Precision::F32).unwrap();
        assert!(f32_tower.forward(&ids(4), false).is_ok());

        assert!(tower.begin(&[]).is_err(), "no tokens");
        assert!(tower.begin(&[cfg.vocab]).is_err(), "id past vocab");
        let mut s = tower.begin(&ids(4)).unwrap();
        s.advance_to(n).unwrap();
        assert!(s.advance_to(n - 1).is_err(), "going back");
        assert!(s.advance_to(n + 1).is_err(), "past the last layer");
        assert_eq!(s.layers_done(), n, "a refused advance must not move");
        let wrong = rt.alloc_tensor_f32(&[5, cfg.hidden as usize]).unwrap();
        assert!(s.final_norm_f32(&wrong).is_err(), "out of the wrong shape");
        let bf = rt.alloc_tensor_bf16(&[4, cfg.hidden as usize]).unwrap();
        assert!(s.final_norm_f32(&bf).is_err(), "out of the wrong dtype");
    });
}

fn read_norm_w(st: &SafeTensors) -> Vec<f32> {
    st.read_f32("model.norm.weight").unwrap().1
}

/// `rms_norm(x) * (1 + w)` per row in f64.
fn host_norm(x: &[f32], w: &[f32], t: usize, h: usize, eps: f32) -> Vec<f64> {
    let mut out = Vec::with_capacity(t * h);
    for r in 0..t {
        let row = &x[r * h..(r + 1) * h];
        let ms = row.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / h as f64;
        let inv = 1.0 / (ms + f64::from(eps)).sqrt();
        out.extend(row.iter().zip(w).map(|(v, w)| f64::from(*v) * inv * (1.0 + f64::from(*w))));
    }
    out
}
