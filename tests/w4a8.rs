//! W4A8: int4b cooperative-tensor inputs and a zero-point-8 unpack.
//!
//! Both arms are checked against one integer oracle on the logical int8
//! matrix. Neither arm is the other's reference. A failed int4b probe is an
//! explicit unsupported error; the unpack arm still has to match. On a device
//! with TensorOps the int4b arm is required to run, and the kernel trace must
//! show it did.

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

#[allow(clippy::too_many_arguments)]
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
            let v = if let Some(groups) = k.checked_div(group) {
                let mut acc = 0.0f32;
                let scales = b_scale.expect("group scales");
                for g in 0..groups {
                    let mut s: i32 = 0;
                    for p in 0..group {
                        let kk = g * group + p;
                        s += a[i * k + kk] as i32 * b[j * k + kk] as i32;
                    }
                    acc += s as f32 * scales[g * n + j];
                }
                acc * a_scalar
            } else {
                let mut s: i32 = 0;
                for p in 0..k {
                    s += a[i * k + p] as i32 * b[j * k + p] as i32;
                }
                s as f32 * a_scalar
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

#[allow(clippy::too_many_arguments)]
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

/// The int4b result of a call on a device that has TensorOps must be `Ok`: an
/// `unsupported` error there is a failure, not a pass. On a device without
/// TensorOps the explicit unsupported error is the contract, and the skip is
/// printed so it cannot be mistaken for a run.
fn int4b_or_skip(rt: &Arc<GpuRuntime>, r: Result<GpuBuffer, String>) -> Option<GpuBuffer> {
    let has = rt.has_tensorops();
    match r {
        Ok(got) => {
            assert!(has, "int4b returned Ok on a device without TensorOps");
            Some(got)
        }
        Err(e) => {
            assert!(!has, "int4b must run on a TensorOps device, got: {e}");
            assert!(e.contains("unsupported") && !e.contains("host"), "{e}");
            println!("SKIP int4b: no TensorOps on this device ({e})");
            None
        }
    }
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
            if let Some(got) = int4b_or_skip(
                rt,
                run(rt, &a, &b, m, n, k, 1.0, I8BScale::None, None, I8Out::F32, W4Arm::Int4b),
            ) {
                assert_exact(&got.read_f32(), &want, &format!("int4b {m}x{n}x{k}"));
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
            if let Some(got) = int4b_or_skip(rt, run(rt, &a, &b, m, n, k, 1.0, scale, None, I8Out::F32, W4Arm::Int4b)) {
                assert_exact(&got.read_f32(), &want, &format!("int4b group {group}"));
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

/// A raw zeroed buffer for the refusal cases, which never read it.
fn zeros(rt: &Arc<GpuRuntime>, bytes: usize) -> GpuBuffer {
    let b = rt.alloc_buffer(bytes).unwrap();
    b.write_bytes(&vec![0u8; bytes]);
    b
}

#[allow(clippy::too_many_arguments)]
fn call(
    rt: &Arc<GpuRuntime>,
    a: &GpuBuffer,
    packed: &GpuBuffer,
    c: &GpuBuffer,
    (m, n, k): (u32, u32, u32),
    a_scale: I8AScale<'_>,
    b_scale: I8BScale<'_>,
    bias: Option<&GpuBuffer>,
    arm: W4Arm,
) -> Result<(), String> {
    nn::gemm_w4a8(
        rt,
        W4Gemm {
            a,
            packed,
            c,
            m,
            n,
            k,
            a_scale,
            b_scale,
            bias,
            out: I8Out::F32,
            arm,
        },
    )
}

/// The refusal contract: an error naming `needle`, and not one dispatch.
fn assert_refused(rt: &Arc<GpuRuntime>, what: &str, r: Result<(), String>, needle: &str) {
    let e = r.expect_err(what);
    assert!(e.contains(needle), "{what}: wanted {needle:?} in {e:?}");
    assert!(
        !e.contains("unsupported"),
        "{what}: refused as a probe failure, not by validation: {e}"
    );
    assert_eq!(rt.take_dispatch_count(), 0, "{what}: dispatched before refusing ({e})");
}

const ARMS: [W4Arm; 2] = [W4Arm::Int4b, W4Arm::UnpackZp8];

/// Group-0 int32 bound. Int4b: |a*b| <= 128*8, so K <= i32::MAX / 1024. The
/// unpack arm keeps the int8 GEMM's 131071. Both refuse before any allocation
/// or dispatch.
#[test]
fn group_zero_k_past_the_int32_bound_is_refused_before_dispatch() {
    with_gpu(|rt| {
        // 128 * 8 * 2^21 = 2^31: the int32 sum wraps to -2^31 without a bound.
        let k = 1u32 << 21;
        let a = vec![-128i8; k as usize];
        let logical = vec![-8i8; k as usize];
        for arm in ARMS {
            rt.take_dispatch_count();
            let e = run(rt, &a, &logical, 1, 1, k, 1.0, I8BScale::None, None, I8Out::F32, arm)
                .expect_err(&format!("{arm:?} k=2^21 must be refused"));
            assert!(e.contains("exceeds"), "{arm:?}: {e}");
            assert!(!e.contains("unsupported"), "{arm:?}: {e}");
            assert_eq!(rt.take_dispatch_count(), 0, "{arm:?} dispatched before refusing");
        }
        // The unpack arm's own, tighter bound.
        let k = 131_072u32;
        let a = vec![1i8; k as usize];
        let logical = vec![1i8; k as usize];
        rt.take_dispatch_count();
        let e = run(
            rt,
            &a,
            &logical,
            1,
            1,
            k,
            1.0,
            I8BScale::None,
            None,
            I8Out::F32,
            W4Arm::UnpackZp8,
        )
        .expect_err("unpack k=131072");
        assert!(e.contains("131071"), "{e}");
        assert_eq!(rt.take_dispatch_count(), 0, "unpack dispatched before refusing");
    });
}

/// The largest legal K at full range is exact, not merely accepted.
#[test]
fn int4b_at_the_k_bound_is_exact() {
    with_gpu(|rt| {
        // Largest multiple of 32 not above i32::MAX / 1024 = 2097151.
        let k = 2_097_120u32;
        let a = vec![-128i8; k as usize];
        let logical = vec![-8i8; k as usize];
        let want = [(128i64 * 8 * k as i64) as f32];
        assert_eq!(want[0], 2_147_450_880.0);
        let r = run(
            rt,
            &a,
            &logical,
            1,
            1,
            k,
            1.0,
            I8BScale::None,
            None,
            I8Out::F32,
            W4Arm::Int4b,
        );
        if let Some(got) = int4b_or_skip(rt, r) {
            assert_exact(&got.read_f32()[..1], &want, "int4b k at bound");
        }
        // And the unpack arm at its own bound (131071 is odd; 131070 is the last even K).
        let k = 131_070u32;
        let a = vec![-128i8; k as usize];
        let logical = vec![-8i8; k as usize];
        let want = [(128i64 * 8 * k as i64) as f32];
        let got = run(
            rt,
            &a,
            &logical,
            1,
            1,
            k,
            1.0,
            I8BScale::None,
            None,
            I8Out::F32,
            W4Arm::UnpackZp8,
        )
        .expect("unpack at its bound");
        assert_exact(&got.read_f32()[..1], &want, "unpack k at bound");
    });
}

/// `c` may not be any buffer the kernel reads, on either arm, and the refusal
/// comes before the unpack dispatch.
#[test]
fn a_writable_c_that_aliases_any_input_is_refused() {
    with_gpu(|rt| {
        let (m, n, k) = (2u32, 3u32, 64u32);
        let shape = (m, n, k);
        let a = zeros(rt, 128);
        let packed = zeros(rt, 96);
        // Finite floats, so scale and bias finiteness checks pass; large enough
        // to be a (128 B), packed (96 B) and c (24 B).
        let shared = buf(rt, &[1.0f32; 128]);
        for arm in ARMS {
            let tag = format!("{arm:?}");
            let scalar = I8AScale::Scalar(1.0);
            rt.take_dispatch_count();
            let r = call(rt, &shared, &packed, &shared, shape, scalar, I8BScale::None, None, arm);
            assert_refused(
                rt,
                &format!("{tag} c==a"),
                r,
                "writable buffer c overlaps read-only buffer a",
            );
            let r = call(rt, &a, &shared, &shared, shape, scalar, I8BScale::None, None, arm);
            assert_refused(rt, &format!("{tag} c==packed"), r, "read-only buffer packed");
            let r = call(
                rt,
                &a,
                &packed,
                &shared,
                shape,
                I8AScale::Rows(&shared),
                I8BScale::None,
                None,
                arm,
            );
            assert_refused(rt, &format!("{tag} c==a_scale"), r, "read-only buffer a_scale");
            let per_group = I8BScale::PerGroup {
                scales: &shared,
                group: 64,
            };
            let r = call(rt, &a, &packed, &shared, shape, scalar, per_group, None, arm);
            assert_refused(rt, &format!("{tag} c==b_scale"), r, "read-only buffer b_scale");
            let r = call(
                rt,
                &a,
                &packed,
                &shared,
                shape,
                scalar,
                I8BScale::None,
                Some(&shared),
                arm,
            );
            assert_refused(rt, &format!("{tag} c==bias"), r, "read-only buffer bias");
        }
    });
}

/// m*k, n*k and m*n must each fit a signed 32-bit extent. Shape-only: the
/// buffers are tiny because the extent check precedes the size check, so the
/// refusal must name the extent, not the buffer size.
#[test]
fn extents_past_signed_32_bit_are_refused_before_allocation_or_dispatch() {
    with_gpu(|rt| {
        let a = zeros(rt, 64);
        let packed = zeros(rt, 64);
        let c = zeros(rt, 64);
        let scalar = I8AScale::Scalar(1.0);
        for arm in ARMS {
            rt.take_dispatch_count();
            // a: 65536 * 32768 = 2^31.
            let r = call(
                rt,
                &a,
                &packed,
                &c,
                (65_536, 8, 32_768),
                scalar,
                I8BScale::None,
                None,
                arm,
            );
            assert_refused(
                rt,
                &format!("{arm:?} a extent"),
                r,
                "exceeds signed 32-bit kernel indexing",
            );
            // b (n * k): 1 * 2^31 is past i32::MAX and k itself is not.
            let r = call(
                rt,
                &a,
                &packed,
                &c,
                (1, 65_536, 32_768),
                scalar,
                I8BScale::None,
                None,
                arm,
            );
            assert_refused(
                rt,
                &format!("{arm:?} b extent"),
                r,
                "exceeds signed 32-bit kernel indexing",
            );
            // c: 65536 * 65536 = 2^32.
            let r = call(
                rt,
                &a,
                &packed,
                &c,
                (65_536, 65_536, 32),
                scalar,
                I8BScale::None,
                None,
                arm,
            );
            assert_refused(
                rt,
                &format!("{arm:?} c extent"),
                r,
                "exceeds signed 32-bit kernel indexing",
            );
        }
    });
}

/// A busy runtime on the first int4b call must not become "unsupported" for
/// the rest of the process. The probe cache is process-wide, so this runs in a
/// child with a fresh cache.
#[test]
fn a_busy_first_call_does_not_disable_int4b() {
    let name = "a_busy_first_call_does_not_disable_int4b";
    common::in_child(name, &[], || {
        with_gpu(|rt| {
            if !rt.has_tensorops() {
                println!("SKIP busy-probe: no TensorOps on this device");
                return;
            }
            let (m, n, k) = (3u32, 5u32, 64u32);
            let a = seq_i8((m * k) as usize, 2, 16);
            let b = seq_i8((n * k) as usize, 5, 16);
            let want = oracle(&a, &b, m as usize, n as usize, k as usize, 1.0, 0, None, None);
            let ab = rt.alloc_buffer(a.len()).unwrap();
            ab.write_bytes(&a.iter().map(|v| *v as u8).collect::<Vec<_>>());
            let pb = rt.alloc_buffer((n * k / 2) as usize).unwrap();
            pb.write_bytes(&pack(&b, n as usize, k as usize, false));
            let cb = zeros(rt, (m * n * 4) as usize);
            let held_by = zeros(rt, 64);
            let scalar = I8AScale::Scalar(1.0);

            let mapping = held_by.try_contents_u8().expect("hold the runtime lease");
            rt.take_dispatch_count();
            let e = call(rt, &ab, &pb, &cb, (m, n, k), scalar, I8BScale::None, None, W4Arm::Int4b)
                .expect_err("a held host mapping must make the first call busy");
            assert!(e.contains("busy"), "{e}");
            assert!(!e.contains("unsupported"), "busy was reported as unsupported: {e}");
            drop(mapping);
            assert!(!rt.is_poisoned());

            call(rt, &ab, &pb, &cb, (m, n, k), scalar, I8BScale::None, None, W4Arm::Int4b)
                .expect("int4b must work once the runtime is free; a busy probe must not stick");
            assert_exact(&cb.read_f32(), &want, "int4b after a busy first call");
        });
    });
}

/// On a TensorOps device int4b runs, across the full int8 activation range
/// (-128..127) against weights -8..7, with and without group scales, and the
/// kernel trace shows the `w4a8_i4_*` kernel dispatched.
#[test]
fn int4b_runs_its_kernel_over_the_full_activation_range() {
    let name = "int4b_runs_its_kernel_over_the_full_activation_range";
    common::in_traced_child(name, || {
        with_gpu(|rt| {
            if !rt.has_tensorops() {
                println!("SKIP int4b kernel trace: no TensorOps on this device");
                return;
            }
            let count = tessl::runtime::traced_kernel_count;
            for &(m, n, k) in &[(3u32, 5u32, 64u32), (33, 7, 64), (1, 8, 32)] {
                let a = seq_i8((m * k) as usize, 3, 256);
                assert!(a.contains(&-128) || m * k < 256, "activations must reach -128");
                let b = seq_i8((n * k) as usize, 5, 16);
                let want = oracle(&a, &b, m as usize, n as usize, k as usize, 1.0, 0, None, None);
                // The first call may add the probe's own request; measure the second.
                run(rt, &a, &b, m, n, k, 1.0, I8BScale::None, None, I8Out::F32, W4Arm::Int4b).expect("warm");
                let before = count("w4a8_i4_32");
                let got = run(rt, &a, &b, m, n, k, 1.0, I8BScale::None, None, I8Out::F32, W4Arm::Int4b)
                    .expect("int4b must run on a TensorOps device");
                assert_eq!(
                    count("w4a8_i4_32") - before,
                    1,
                    "w4a8_i4_32 not dispatched for {m}x{n}x{k}"
                );
                assert_exact(&got.read_f32(), &want, &format!("int4b full-range A {m}x{n}x{k}"));
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
                assert_exact(&unpacked.read_f32(), &want, &format!("unpack full-range A {m}x{n}x{k}"));
            }
            let (m, n, k, group) = (4u32, 2u32, 128u32, 64u32);
            let a = seq_i8((m * k) as usize, 1, 256);
            let b = seq_i8((n * k) as usize, 5, 16);
            let scales: Vec<f32> = (0..(k / group * n)).map(|i| [1.0, 0.5, 0.25][i as usize % 3]).collect();
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
            run(rt, &a, &b, m, n, k, 1.0, scale, None, I8Out::F32, W4Arm::Int4b).expect("warm group");
            let before = count("w4a8_i4_g64");
            let got = run(rt, &a, &b, m, n, k, 1.0, scale, None, I8Out::F32, W4Arm::Int4b).expect("group");
            assert_eq!(count("w4a8_i4_g64") - before, 1, "w4a8_i4_g64 not dispatched");
            assert_exact(&got.read_f32(), &want, "int4b full-range A group 64");
        });
    });
}
