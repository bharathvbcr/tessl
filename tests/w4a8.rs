//! W4A8: int4b cooperative-tensor inputs and a zero-point-8 unpack.
//!
//! Both arms are checked against one integer oracle on the logical int8
//! matrix. Neither arm is the other's reference. A failed int4b probe is an
//! explicit unsupported error; the unpack arm still has to match.

mod common;

use std::sync::Arc;

use common::{buf, with_gpu};
use tessl::nn::{self, I8AScale, I8BScale, I8Out, W4Arm, W4Gemm};
use tessl::tensor::GpuBuffer;
use tessl::GpuRuntime;

fn seq_i8(n: usize, seed: i32, span: i32) -> Vec<i8> {
    (0..n)
        .map(|i| ((i as i32).wrapping_mul(5).wrapping_add(seed).rem_euclid(span) - span / 2) as i8)
        .collect()
}

fn pack(logical: &[i8], n: usize, k: usize, zp8: bool) -> Vec<u8> {
    assert_eq!(logical.len(), n * k);
    let mut out = vec![0u8; n * (k / 2)];
    for ni in 0..n {
        for kk in (0..k).step_by(2) {
            let nib = |v: i8| -> u8 {
                if zp8 {
                    (v + 8) as u8
                } else if v < 0 {
                    (v + 16) as u8
                } else {
                    v as u8
                }
            };
            let lo = nib(logical[ni * k + kk]);
            let hi = nib(logical[ni * k + kk + 1]);
            out[ni * (k / 2) + kk / 2] = lo | (hi << 4);
        }
    }
    out
}

fn oracle(
    a: &[i8],
    b: &[i8],
    m: usize,
    n: usize,
    k: usize,
    a_scalar: f32,
    group: usize,
    b_scale: Option<&[f32]>,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let v = if group == 0 {
                let mut s: i32 = 0;
                for p in 0..k {
                    s += a[i * k + p] as i32 * b[j * k + p] as i32;
                }
                s as f32 * a_scalar
            } else {
                let mut acc = 0.0f32;
                let scales = b_scale.expect("group scales");
                for g in 0..(k / group) {
                    let mut s: i32 = 0;
                    for p in 0..group {
                        let kk = g * group + p;
                        s += a[i * k + kk] as i32 * b[j * k + kk] as i32;
                    }
                    acc += s as f32 * scales[g * n + j];
                }
                acc * a_scalar
            };
            out[i * n + j] = v + bias.map(|b| b[j]).unwrap_or(0.0);
        }
    }
    out
}

fn assert_exact(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what} [{i}]: got {g} want {w}");
    }
}

fn run(
    rt: &Arc<GpuRuntime>,
    a: &[i8],
    logical: &[i8],
    m: u32,
    n: u32,
    k: u32,
    a_scalar: f32,
    b_scale: I8BScale<'_>,
    bias: Option<&GpuBuffer>,
    out: I8Out,
    arm: W4Arm,
) -> Result<GpuBuffer, String> {
    let zp8 = matches!(arm, W4Arm::UnpackZp8);
    let packed = pack(logical, n as usize, k as usize, zp8);
    let ab = rt.alloc_buffer(a.len().max(1)).unwrap();
    ab.write_bytes(&a.iter().map(|v| *v as u8).collect::<Vec<_>>());
    let pb = rt.alloc_buffer(packed.len().max(1)).unwrap();
    pb.write_bytes(&packed);
    let bytes = (m as usize) * (n as usize) * if out == I8Out::F32 { 4 } else { 2 };
    let cb = rt.alloc_buffer(bytes.max(4)).unwrap();
    cb.write_bytes(&vec![0u8; bytes.max(1)]);
    nn::gemm_w4a8(
        rt,
        W4Gemm {
            a: &ab,
            packed: &pb,
            c: &cb,
            m,
            n,
            k,
            a_scale: I8AScale::Scalar(a_scalar),
            b_scale,
            bias,
            out,
            arm,
        },
    )?;
    Ok(cb)
}

#[test]
fn both_arms_match_one_integer_oracle() {
    with_gpu(|rt| {
        for &(m, n, k) in &[(3u32, 5, 64), (1, 8, 32), (33, 7, 64)] {
            let a = seq_i8((m * k) as usize, 2, 16);
            let b = seq_i8((n * k) as usize, 5, 16);
            let want = oracle(&a, &b, m as usize, n as usize, k as usize, 1.0, 0, None, None);
            let unpacked = run(
                rt,
                &a,
                &b,
                m,
                n,
                k,
                1.0,
                I8BScale::None,
                None,
                I8Out::F32,
                W4Arm::UnpackZp8,
            )
            .expect("unpack");
            assert_exact(&unpacked.read_f32(), &want, &format!("unpack {m}x{n}x{k}"));
            match run(rt, &a, &b, m, n, k, 1.0, I8BScale::None, None, I8Out::F32, W4Arm::Int4b) {
                Ok(got) => assert_exact(&got.read_f32(), &want, &format!("int4b {m}x{n}x{k}")),
                Err(e) => {
                    assert!(e.contains("unsupported"), "{e}");
                    assert!(!e.contains("host"), "{e}");
                }
            }
        }
    });
}

#[test]
fn group_scales_match_the_ordered_flush() {
    with_gpu(|rt| {
        for &(m, n, k, group) in &[(2u32, 3, 128, 64), (4, 2, 128, 128), (2, 2, 256, 256)] {
            let a = vec![1i8; (m * k) as usize];
            let b = vec![1i8; (n * k) as usize];
            let groups = (k / group) as usize;
            let mut scales = vec![0.0f32; groups * n as usize];
            for g in 0..groups {
                for j in 0..n as usize {
                    scales[g * n as usize + j] = [1.0, 0.1, 0.5][g % 3];
                }
            }
            let sb = buf(rt, &scales);
            let want = oracle(
                &a,
                &b,
                m as usize,
                n as usize,
                k as usize,
                1.0,
                group as usize,
                Some(&scales),
                None,
            );
            let scale = I8BScale::PerGroup { scales: &sb, group };
            let unpacked = run(rt, &a, &b, m, n, k, 1.0, scale, None, I8Out::F32, W4Arm::UnpackZp8).expect("unpack");
            assert_exact(&unpacked.read_f32(), &want, &format!("unpack group {group}"));
            if groups > 1 {
                let collapsed = (k as f32) * scales.iter().step_by(n as usize).take(groups).sum::<f32>();
                assert_ne!(want[0].to_bits(), collapsed.to_bits());
            }
            match run(rt, &a, &b, m, n, k, 1.0, scale, None, I8Out::F32, W4Arm::Int4b) {
                Ok(got) => assert_exact(&got.read_f32(), &want, &format!("int4b group {group}")),
                Err(e) => assert!(e.contains("unsupported"), "{e}"),
            }
        }
    });
}

#[test]
fn f16_store_bias_and_two_runs_are_stable() {
    with_gpu(|rt| {
        let (m, n, k) = (2u32, 4, 32);
        let a = vec![1i8; (m * k) as usize];
        let b = vec![1i8; (n * k) as usize];
        let bias_v = vec![0.5f32; n as usize];
        let bias = buf(rt, &bias_v);
        let want = oracle(&a, &b, m as usize, n as usize, k as usize, 1.0, 0, None, Some(&bias_v));
        let mut bits = Vec::new();
        for _ in 0..2 {
            let got = run(
                rt,
                &a,
                &b,
                m,
                n,
                k,
                1.0,
                I8BScale::None,
                Some(&bias),
                I8Out::F16,
                W4Arm::UnpackZp8,
            )
            .expect("f16");
            bits.push(got.contents_u16()[..(m * n) as usize].to_vec());
        }
        assert_eq!(bits[0], bits[1]);
        for (i, w) in want.iter().enumerate() {
            assert_eq!(bits[0][i], tessl::tensor::f32_to_f16_bits(*w), "f16 {i} from {w}");
        }
    });
}

#[test]
fn bad_k_and_per_column_scales_are_refused_before_dispatch() {
    with_gpu(|rt| {
        let (m, n, k) = (2u32, 3, 48);
        let a = vec![1i8; (m * k) as usize];
        let b = vec![1i8; (n * k) as usize];
        let want = oracle(&a, &b, m as usize, n as usize, k as usize, 1.0, 0, None, None);
        let unpacked = run(
            rt,
            &a,
            &b,
            m,
            n,
            k,
            1.0,
            I8BScale::None,
            None,
            I8Out::F32,
            W4Arm::UnpackZp8,
        )
        .expect("k=48");
        assert_exact(&unpacked.read_f32(), &want, "unpack k=48");

        rt.take_dispatch_count();
        let err =
            run(rt, &a, &b, m, n, k, 1.0, I8BScale::None, None, I8Out::F32, W4Arm::Int4b).expect_err("k=48 int4b");
        assert!(err.contains("32"), "{err}");
        assert!(!err.contains("unsupported"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);

        let a32 = vec![1i8; (m * 32) as usize];
        let b32 = vec![1i8; (n * 32) as usize];
        let col = buf(rt, &[1.0, 1.0, 1.0]);
        rt.take_dispatch_count();
        let err = run(
            rt,
            &a32,
            &b32,
            m,
            n,
            32,
            1.0,
            I8BScale::PerColumn(&col),
            None,
            I8Out::F32,
            W4Arm::UnpackZp8,
        )
        .expect_err("per-column");
        assert!(err.contains("per-column"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);

        let err = run(
            rt,
            &a32,
            &b32,
            m,
            n,
            32,
            f32::INFINITY,
            I8BScale::None,
            None,
            I8Out::F32,
            W4Arm::UnpackZp8,
        )
        .expect_err("non-finite");
        assert!(err.contains("finite"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);

        let odd_a = rt.alloc_buffer(64).unwrap();
        let odd_w = rt.alloc_buffer(64).unwrap();
        let odd_c = rt.alloc_buffer(64).unwrap();
        let err = nn::gemm_w4a8(
            rt,
            W4Gemm {
                a: &odd_a,
                packed: &odd_w,
                c: &odd_c,
                m: 2,
                n: 2,
                k: 31,
                a_scale: I8AScale::Scalar(1.0),
                b_scale: I8BScale::None,
                bias: None,
                out: I8Out::F32,
                arm: W4Arm::UnpackZp8,
            },
        )
        .expect_err("odd");
        assert!(err.contains("even"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}
