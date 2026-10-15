//! Adversarial check of the claim behind removing `!isfinite(v)` from
//! `softcap_argmax_one_pass` (kernels/softcap_sample.metal): for a finite `raw`,
//! `tessl_apply_softcap` never returns a non-finite `v`, whatever the cap
//! (FLT_MAX, subnormal, -0, 0, negative, infinite, NaN) and whatever the raw
//! extremes. `softcap_sample` rewrites the logits in place, so the values the
//! kernel produced can be read back and the one-pass token compared with them.

mod common;

use common::{buf, buf_u32, with_gpu};
use tessl::nn;

#[test]
fn softcap_of_finite_extremes_is_finite_and_one_pass_agrees() {
    let raws = [
        f32::MAX,
        -f32::MAX,
        f32::MIN_POSITIVE,
        1e-45,
        -1e-45,
        0.0,
        -0.0,
        3.0,
        f32::MAX,
        -3.0e38,
        1.0e30,
        2.0,
    ];
    let caps = [
        f32::MAX,
        3.0e38,
        f32::MIN_POSITIVE,
        1.0e-38,
        1e-40,
        1e-45,
        -0.0,
        0.0,
        -1.0,
        -f32::MAX,
        30.0,
        f32::INFINITY,
        f32::NAN,
    ];
    with_gpu(|rt| {
        for cap in caps {
            let n = raws.len() as u32;
            let cb = buf(rt, &[cap]);
            let rewritten = buf(rt, &raws);
            let out_a = buf_u32(rt, &[0]);
            nn::softcap_sample(rt, &rewritten, &out_a, &cb, n).unwrap();
            let one_pass_in = buf(rt, &raws);
            let out_b = buf_u32(rt, &[0]);
            nn::softcap_argmax_one_pass(rt, &one_pass_in, &out_b, &cb, n).unwrap();
            rt.synchronize().unwrap();

            let v = rewritten.read_f32();
            for (i, x) in v[..raws.len()].iter().enumerate() {
                assert!(x.is_finite(), "cap {cap:e}: raw {:e} became {x:e} at {i}", raws[i]);
            }
            let first_max = v[..raws.len()]
                .iter()
                .enumerate()
                .fold(
                    (0usize, f32::NEG_INFINITY),
                    |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) },
                )
                .0 as u32;
            assert_eq!(out_a.read_u32()[0], first_max, "cap {cap:e}: softcap_sample token");
            assert_eq!(
                out_b.read_u32()[0],
                first_max,
                "cap {cap:e}: one-pass token vs rewritten logits"
            );
        }
    });
}
