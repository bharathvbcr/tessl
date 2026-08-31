//! Run-to-run stability: the same GEMM, repeated, must produce the same bits.
//!
//! A GPU is allowed to disagree with a CPU -- that is what the tolerance in
//! `gemm_correctness` is for. It is not allowed to disagree with itself. Every
//! kernel here fixes its reduction order at dispatch time (tile walk, split-K
//! partitions serialized behind barriers, no atomic accumulation), so bitwise
//! equality is the correct assertion, not an approximate one. Anything that
//! made the result depend on scheduling -- an atomic add introduced for speed,
//! a race on a tile, a dependence on whatever the destination buffer happened
//! to contain -- shows up here and nowhere else, because a nondeterministic
//! kernel still lands comfortably inside the numeric tolerance.
//!
//! Comparing `f32::to_bits` rather than `==` is deliberate: it distinguishes
//! +0.0 from -0.0 and does not silently pass a pair of NaNs.

mod common;

use common::{random_f32, round_trip_bf16, tensor_bf16, tensor_f32, with_gpu};
use tessl::gemm::{gemm_nt_f32, gemm_tn_f32, gemm_tn_train};
use tessl::{gemm, gemm_f32, GemmBackend, PrecisionMode, Tensor};

/// Enough repeats to cross command-buffer and allocator-slot boundaries; the
/// Metal 4 package ping-pongs allocators, so a handful of runs could all land
/// in the same slot and prove nothing about the other one.
const REPEATS: usize = 48;

#[track_caller]
fn assert_same_bits(label: &str, run: usize, first: &[f32], got: &[f32]) {
    assert_eq!(
        first.len(),
        got.len(),
        "{label}: length changed on run {run}"
    );
    if let Some((i, (a, b))) = first
        .iter()
        .zip(got.iter())
        .enumerate()
        .find(|(_, (a, b))| a.to_bits() != b.to_bits())
    {
        panic!(
            "{label}: run {run} diverged at element [{i}]: {a} (0x{:08x}) vs {b} (0x{:08x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

#[test]
fn f32_nn_repeats_bit_identically() {
    with_gpu(|rt| {
        let (m, n, k) = (130, 257, 96);
        let a = tensor_f32(rt, &[m, k], &random_f32(m * k, 1));
        let b = tensor_f32(rt, &[k, n], &random_f32(k * n, 2));
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();

        let mut first: Option<Vec<f32>> = None;
        for run in 0..REPEATS {
            gemm_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let got = c.buffer.read_f32();
            match &first {
                None => first = Some(got),
                Some(f) => assert_same_bits("f32 NN", run, f, &got),
            }
        }
        // A GEMM that never wrote anything would also be perfectly stable.
        assert!(
            first.unwrap().iter().any(|&x| x != 0.0),
            "output stayed zero"
        );
    });
}

#[test]
fn bf16_nn_repeats_bit_identically() {
    with_gpu(|rt| {
        // N > 512 puts this on the 128x64 default coop tile, whose register
        // accumulator and single store are the newest part of the pipeline.
        let (m, n, k) = (129, 520, 130);
        let a = tensor_bf16(rt, &[m, k], &round_trip_bf16(&random_f32(m * k, 3)));
        let b = tensor_bf16(rt, &[k, n], &round_trip_bf16(&random_f32(k * n, 4)));
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();

        let mut first: Option<Vec<f32>> = None;
        for run in 0..REPEATS {
            gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let got = c.buffer.read_f32();
            match &first {
                None => first = Some(got),
                Some(f) => assert_same_bits("bf16 NN", run, f, &got),
            }
        }
        assert!(
            first.unwrap().iter().any(|&x| x != 0.0),
            "output stayed zero"
        );
    });
}

#[test]
fn tn_nt_and_splitk_repeat_bit_identically() {
    with_gpu(|rt| {
        // Split-K is the one lane that accumulates across dispatches. It is
        // serialized behind barriers today, so it is deterministic; if that
        // ever became an atomic reduction for throughput, only this assertion
        // would notice.
        let (m, n, k) = (128, 128, 2048);
        let a = tensor_f32(rt, &[k, m], &random_f32(k * m, 5));
        let b = tensor_f32(rt, &[k, n], &random_f32(k * n, 6));
        let nt_b = tensor_f32(rt, &[n, k], &random_f32(n * k, 7));
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        let a_mk = tensor_f32(rt, &[m, k], &random_f32(m * k, 8));

        let mut tn: Option<Vec<f32>> = None;
        let mut nt: Option<Vec<f32>> = None;
        for run in 0..16 {
            gemm_tn_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let got = c.buffer.read_f32();
            match &tn {
                None => tn = Some(got),
                Some(f) => assert_same_bits("f32 TN split-K", run, f, &got),
            }

            gemm_nt_f32(&a_mk, &nt_b, &c, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let got = c.buffer.read_f32();
            match &nt {
                None => nt = Some(got),
                Some(f) => assert_same_bits("f32 NT", run, f, &got),
            }
        }
        assert!(
            tn.unwrap().iter().any(|&x| x != 0.0),
            "TN output stayed zero"
        );
        assert!(
            nt.unwrap().iter().any(|&x| x != 0.0),
            "NT output stayed zero"
        );
    });
}

#[test]
fn bf16_tn_splitk_repeats_bit_identically() {
    with_gpu(|rt| {
        rt.set_precision(PrecisionMode::Bf16);
        let (m, n, k) = (128, 128, 2048);
        let a = tensor_bf16(rt, &[k, m], &round_trip_bf16(&random_f32(k * m, 9)));
        let b = tensor_bf16(rt, &[k, n], &round_trip_bf16(&random_f32(k * n, 10)));
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();

        let mut first: Option<Vec<f32>> = None;
        for run in 0..16 {
            gemm_tn_train(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let got = c.buffer.read_f32();
            match &first {
                None => first = Some(got),
                Some(f) => assert_same_bits("bf16 TN split-K", run, f, &got),
            }
        }
        assert!(
            first.unwrap().iter().any(|&x| x != 0.0),
            "output stayed zero"
        );
    });
}

#[test]
fn results_do_not_depend_on_which_recycled_buffer_backs_the_output() {
    with_gpu(|rt| {
        // The coop kernels claim C is written exactly once with no host pre-zero.
        // If any element were left untouched, the answer would silently inherit
        // whatever the pool handed back -- so each run gets a *different* C,
        // half of them pre-poisoned with a value the correct result cannot be.
        let (m, n, k) = (67, 511, 33);
        let a = tensor_bf16(rt, &[m, k], &round_trip_bf16(&random_f32(m * k, 11)));
        let b = tensor_bf16(rt, &[k, n], &round_trip_bf16(&random_f32(k * n, 12)));

        let mut first: Option<Vec<f32>> = None;
        for run in 0..REPEATS {
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            if run % 2 == 1 {
                c.buffer.write_f32_prefix(&vec![-12_345.0f32; m * n]);
            }
            gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let got = c.buffer.read_f32();
            match &first {
                None => first = Some(got),
                Some(f) => assert_same_bits("bf16 NN over recycled C", run, f, &got),
            }
            // Dropping inside the loop returns the buffer to the freelist, so
            // later runs really are reusing storage a previous run wrote.
            drop(c);
        }
        assert!(
            first.unwrap().iter().any(|&x| x != 0.0),
            "output stayed zero"
        );
    });
}

#[test]
fn results_do_not_depend_on_neighbouring_work_in_the_command_buffer() {
    with_gpu(|rt| {
        // Consumers encode a whole layer graph before syncing, so the GEMM under
        // test is never the only thing in flight. Interleaving unrelated GEMMs
        // of other shapes changes pipeline-cache order, argument-table slots and
        // const-arena offsets around it; the result must not move.
        rt.set_async_encode(true).unwrap();
        let (m, n, k) = (96, 520, 64);
        let a = tensor_bf16(rt, &[m, k], &round_trip_bf16(&random_f32(m * k, 13)));
        let b = tensor_bf16(rt, &[k, n], &round_trip_bf16(&random_f32(k * n, 14)));
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();

        let noise_a = tensor_f32(rt, &[31, 47], &random_f32(31 * 47, 15));
        let noise_b = tensor_f32(rt, &[47, 29], &random_f32(47 * 29, 16));
        let noise_c = rt.alloc_tensor_f32(&[31, 29]).unwrap();

        let mut first: Option<Vec<f32>> = None;
        for run in 0..REPEATS {
            for _ in 0..(run % 3) {
                gemm_f32(&noise_a, &noise_b, &noise_c, GemmBackend::TensorOps).unwrap();
            }
            gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            for _ in 0..(run % 2) {
                gemm_f32(&noise_a, &noise_b, &noise_c, GemmBackend::Simdgroup).unwrap();
            }
            rt.synchronize().unwrap();
            let got: Vec<f32> = c.buffer.read_f32();
            match &first {
                None => first = Some(got),
                Some(f) => assert_same_bits("bf16 NN amid other work", run, f, &got),
            }
        }
        assert!(
            first.unwrap().iter().any(|&x| x != 0.0),
            "output stayed zero"
        );
    });
}

#[test]
fn an_offset_output_view_matches_the_same_gemm_at_offset_zero() {
    with_gpu(|rt| {
        // `byte_offset` reaches the kernel as a bound buffer offset, not as an
        // index the shader recomputes, so writing at an offset must produce the
        // identical bits -- not merely a numerically close answer.
        let (m, n, k) = (65, 63, 31);
        let a = tensor_f32(rt, &[m, k], &random_f32(m * k, 17));
        let b = tensor_f32(rt, &[k, n], &random_f32(k * n, 18));
        let flat = rt.alloc_tensor_f32(&[m, n]).unwrap();
        gemm_f32(&a, &b, &flat, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let baseline = flat.buffer.read_f32();

        let backing = rt.alloc_tensor_f32(&[4 * m * n]).unwrap();
        for slot in 0..4 {
            let view: Tensor = backing.view(&[m, n], slot * m * n);
            gemm_f32(&a, &b, &view, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let all = backing.buffer.read_f32();
            assert_same_bits(
                "f32 NN at an offset",
                slot,
                &baseline,
                &all[slot * m * n..(slot + 1) * m * n],
            );
        }
        assert!(baseline.iter().any(|&x| x != 0.0), "output stayed zero");
    });
}
