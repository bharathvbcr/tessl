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
use std::sync::Arc;
use tessl::gemm::{
    gemm_batched, gemm_nt_f32, gemm_tiled, gemm_tn_f32, gemm_tn_train, BatchStrides, BatchedGemm, EpiTile,
};
use tessl::tensor::f32_slice_to_f16;
use tessl::{gemm, gemm_f32, GemmBackend, GpuRuntime, PrecisionMode, Tensor};

/// Enough repeats to cross command-buffer and allocator-slot boundaries; the
/// Metal 4 package ping-pongs allocators, so a handful of runs could all land
/// in the same slot and prove nothing about the other one.
const REPEATS: usize = 48;

#[track_caller]
fn assert_same_bits(label: &str, run: usize, first: &[f32], got: &[f32]) {
    assert_eq!(first.len(), got.len(), "{label}: length changed on run {run}");
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
        assert!(first.unwrap().iter().any(|&x| x != 0.0), "output stayed zero");
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
        assert!(first.unwrap().iter().any(|&x| x != 0.0), "output stayed zero");
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
        assert!(tn.unwrap().iter().any(|&x| x != 0.0), "TN output stayed zero");
        assert!(nt.unwrap().iter().any(|&x| x != 0.0), "NT output stayed zero");
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
        assert!(first.unwrap().iter().any(|&x| x != 0.0), "output stayed zero");
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
        assert!(first.unwrap().iter().any(|&x| x != 0.0), "output stayed zero");
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
        assert!(first.unwrap().iter().any(|&x| x != 0.0), "output stayed zero");
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

        let backing = rt.alloc_tensor_f32(&[4 * m * n + 16]).unwrap();
        // Stride between slots must keep every view 16-byte aligned.
        let stride = (m * n).div_ceil(4) * 4;
        for slot in 0..4 {
            let view: Tensor = backing.view(&[m, n], slot * stride);
            gemm_f32(&a, &b, &view, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let all = backing.buffer.read_f32();
            assert_same_bits(
                "f32 NN at an offset",
                slot,
                &baseline,
                &all[slot * stride..slot * stride + m * n],
            );
        }
        assert!(baseline.iter().any(|&x| x != 0.0), "output stayed zero");
    });
}

#[test]
fn splitk_under_hazard_skip_auto_stays_bit_identical_amid_neighbours() {
    with_gpu(|rt| {
        // Plan 2.3: under HAZARD_BARRIERS skip-auto the per-dispatch auto
        // barrier is gone; split-K still places explicit barriers between
        // partitions. Pack neighbouring GEMMs around a split-K shape under
        // async encode — if partition serialization ever races, bits drift.
        // Scratch tree reduction is *not* added unless this fails.
        tessl::ab_flags::set_hazard_barriers(true);
        rt.set_async_encode(true).unwrap();

        let (m, n, k) = (128, 128, 2048);
        let a = tensor_f32(rt, &[k, m], &random_f32(k * m, 21));
        let b = tensor_f32(rt, &[k, n], &random_f32(k * n, 22));
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();

        let noise_a = tensor_f32(rt, &[64, 96], &random_f32(64 * 96, 23));
        let noise_b = tensor_f32(rt, &[96, 80], &random_f32(96 * 80, 24));
        let noise_c = rt.alloc_tensor_f32(&[64, 80]).unwrap();

        let mut first: Option<Vec<f32>> = None;
        for run in 0..24 {
            for _ in 0..(run % 3) {
                gemm_f32(&noise_a, &noise_b, &noise_c, GemmBackend::TensorOps).unwrap();
            }
            gemm_tn_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            for _ in 0..(run % 2) {
                gemm_f32(&noise_a, &noise_b, &noise_c, GemmBackend::Simdgroup).unwrap();
            }
            rt.synchronize().unwrap();
            let got = c.buffer.read_f32();
            match &first {
                None => first = Some(got),
                Some(f) => assert_same_bits("f32 TN split-K hazard skip-auto", run, f, &got),
            }
        }
        assert!(first.unwrap().iter().any(|&x| x != 0.0), "split-K output stayed zero");
        tessl::ab_flags::set_hazard_barriers(false);
    });
}

// ------------------------------------------------- cross-route bit identity
//
// Everything above repeats one route against itself. The routes below are the
// ones `RouteDecision` in `src/gemm.rs` classes as arithmetic-preserving
// (`route_arithmetic` returns `Preserving` for `AutoCoopTile` and
// `ForcedMatchingCoopTile` only), so two of them given the same operands must
// agree to the bit, across shapes and across batch sizes.
//
// Excluded, each by that same doc: `PreferTnSplitk` and
// `SplitKVersusOneLaunch` (they change the association of the K sum),
// `Dtype` (changes the products), and `EpiTileWidth` (64x64 against 128x64 was
// not classed bit-equal, so a width change is not a preserving route here;
// `tests/gemm_correctness.rs` measures that pair on its own).

#[derive(Clone, Copy, Debug)]
enum Elem {
    Bf16,
    F16,
}

/// Upload `data` as the narrow element type, rounded so the values are exact.
fn narrow_tensor(rt: &Arc<GpuRuntime>, elem: Elem, shape: &[usize], data: &[f32]) -> Tensor {
    match elem {
        Elem::Bf16 => tensor_bf16(rt, shape, &round_trip_bf16(data)),
        Elem::F16 => {
            let t = rt.alloc_tensor_f16(shape).expect("alloc_tensor_f16");
            t.buffer.write_f16_bits(&f32_slice_to_f16(data));
            t
        }
    }
}

/// The tile `gemm` picks, from its public doc: 64x64 when N <= 512, and for
/// bf16 also when M does not fill a 128-row tile or C has fewer than 64 tiles of
/// 128x64; 128x64 otherwise. A wrong copy of the rule cannot hide: the forced
/// matching tile would then be the other geometry, and the comparison below
/// holds only if the two geometries happen to agree bit for bit.
fn documented_auto_tile(elem: Elem, m: usize, n: usize) -> EpiTile {
    let narrow = n <= 512 || (matches!(elem, Elem::Bf16) && (m < 128 || m.div_ceil(128) * n.div_ceil(64) < 64));
    if narrow {
        EpiTile::Narrow
    } else {
        EpiTile::Wide
    }
}

fn finite_and_written(label: &str, out: &[f32]) {
    assert!(
        out.iter().all(|v| v.is_finite()),
        "{label}: non-finite or unwritten output"
    );
    assert!(out.iter().any(|&v| v != 0.0), "{label}: output stayed zero");
}

#[test]
fn auto_tile_and_the_forced_matching_tile_are_bit_identical() {
    with_gpu(|rt| {
        // Both sides of every clause of the selection rule: N <= 512, short M,
        // few tiles (bf16 only), and a grid that fills the device.
        let shapes: [(Elem, usize, usize, usize); 10] = [
            (Elem::Bf16, 64, 64, 128),
            (Elem::Bf16, 96, 512, 64),
            (Elem::Bf16, 96, 520, 64),
            (Elem::Bf16, 129, 520, 130),
            (Elem::Bf16, 128, 4096, 64),
            (Elem::Bf16, 256, 4096, 96),
            (Elem::F16, 64, 64, 128),
            (Elem::F16, 129, 520, 130),
            (Elem::F16, 128, 4096, 64),
            (Elem::F16, 67, 511, 33),
        ];
        for (case, &(elem, m, n, k)) in shapes.iter().enumerate() {
            let a = narrow_tensor(rt, elem, &[m, k], &random_f32(m * k, 100 + case as u64));
            let b = narrow_tensor(rt, elem, &[k, n], &random_f32(k * n, 200 + case as u64));
            let auto = rt.alloc_tensor_f32(&[m, n]).unwrap();
            let forced = rt.alloc_tensor_f32(&[m, n]).unwrap();
            for c in [&auto, &forced] {
                c.buffer.write_f32_prefix(&vec![f32::NAN; m * n]);
            }
            gemm(&a, &b, &auto, GemmBackend::TensorOps).unwrap();
            gemm_tiled(
                &a,
                &b,
                &forced,
                GemmBackend::TensorOps,
                documented_auto_tile(elem, m, n),
            )
            .unwrap();
            rt.synchronize().unwrap();
            let (auto_out, forced_out) = (auto.buffer.read_f32(), forced.buffer.read_f32());
            let label = format!("{elem:?} {m}x{n}x{k} auto against its forced matching tile");
            finite_and_written(&label, &auto_out[..m * n]);
            assert_same_bits(&label, case, &auto_out[..m * n], &forced_out[..m * n]);
        }
    });
}

#[test]
fn batched_elements_equal_the_single_matrix_gemm_at_every_batch_size() {
    with_gpu(|rt| {
        // `gemm_batched` documents each element as bit-identical to the `gemm`
        // that would have produced it, at the tile `gemm` picks. Strides are
        // multiples of 16 bytes for both element types (M*K and K*N multiples of
        // 8 elements), as the batched alignment rule needs. Batch 1 takes the
        // single-matrix kernel itself; 2 and up take the batched kernels.
        let shapes: [(usize, usize, usize); 3] = [
            (64, 64, 128),   // narrow: N <= 512
            (129, 520, 136), // narrow by grid fill, a partial M and N tile
            (128, 4096, 64), // wide: the grid fills the device
        ];
        for (elem, (m, n, k)) in [Elem::Bf16, Elem::F16]
            .into_iter()
            .flat_map(|e| shapes.into_iter().map(move |s| (e, s)))
        {
            assert_eq!((m * k) % 8, 0);
            assert_eq!((k * n) % 8, 0);
            for batch in [1usize, 2, 3, 5, 8] {
                for shared_b in [false, true] {
                    let a_all = narrow_tensor(
                        rt,
                        elem,
                        &[batch * m, k],
                        &random_f32(batch * m * k, 300 + batch as u64),
                    );
                    let b_batches = if shared_b { 1 } else { batch };
                    let b_all = narrow_tensor(
                        rt,
                        elem,
                        &[b_batches * k, n],
                        &random_f32(b_batches * k * n, 400 + batch as u64),
                    );
                    let c_all = rt.alloc_tensor_f32(&[batch * m, n]).unwrap();
                    c_all.buffer.write_f32_prefix(&vec![f32::NAN; batch * m * n]);
                    let strides = if shared_b {
                        BatchStrides::shared_b(m, n, k)
                    } else {
                        BatchStrides::contiguous(m, n, k)
                    };
                    gemm_batched(
                        &a_all,
                        &b_all,
                        &c_all,
                        GemmBackend::TensorOps,
                        BatchedGemm {
                            m,
                            n,
                            k,
                            batch,
                            strides,
                        },
                    )
                    .unwrap();
                    rt.synchronize().unwrap();
                    let batched = c_all.buffer.read_f32();
                    for i in 0..batch {
                        let a_i = a_all.view(&[m, k], i * m * k);
                        let b_i = b_all.view(&[k, n], if shared_b { 0 } else { i * k * n });
                        let single = rt.alloc_tensor_f32(&[m, n]).unwrap();
                        single.buffer.write_f32_prefix(&vec![f32::NAN; m * n]);
                        gemm(&a_i, &b_i, &single, GemmBackend::TensorOps).unwrap();
                        rt.synchronize().unwrap();
                        let want = single.buffer.read_f32();
                        let label = format!("{elem:?} {m}x{n}x{k} batch {batch} shared_b {shared_b} element {i}");
                        finite_and_written(&label, &want[..m * n]);
                        assert_same_bits(&label, i, &want[..m * n], &batched[i * m * n..(i + 1) * m * n]);
                    }
                }
            }
        }
    });
}
