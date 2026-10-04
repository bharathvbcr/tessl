//! Attacks on the short-M bf16 tile choice.
//!
//! The happy path (M < 128, N = 8224, K = 2048, bit-identical tiles) already
//! lives in `gemm_correctness` and `gemm_epilogue`. These cases keep a partial
//! K, a partial N past the 512 cut, the M = 127 / 128 boundary, and the
//! checks that must fail closed: a dtype that is not bf16, an output that
//! aliases an input, and a bias that runs off the last tile.

mod common;

use std::sync::Arc;

use common::{
    assert_within_bound, random_f32, reference, round_trip_bf16, tensor_bf16, tensor_f32, tolerance, with_gpu, Layout,
    Reference, U_BF16,
};
use tessl::gemm::{gemm_epilogue, gemm_epilogue_tiled, gemm_tiled, Activation, EpiTile, Epilogue, GemmBackend};
use tessl::tensor::{bf16_bits_to_f32, f16_bits_to_f32, f32_slice_to_f16};
use tessl::{gemm, DType, GpuRuntime, Tensor};

/// Past the N ≤ 512 cut, not a multiple of 64, so the last tile is partial
/// and only the bf16 short-M rule selects 64×64.
const N: usize = 520;
/// Not a multiple of 8. The selector ignores K; the kernel must not.
const K: usize = 7;
const PAD: usize = 64;

fn attack_shape() {
    assert!(
        N > 512,
        "N must be past the 512 cut or the short-M branch is not the one under test"
    );
    assert_ne!(N % 64, 0, "N must leave a partial tile");
    assert_ne!(K % 8, 0, "K must not be a multiple of 8");
}

fn require_tensorops(rt: &GpuRuntime) {
    assert!(
        rt.has_tensorops(),
        "TensorOps metallib is missing, so the short-M kernels did not run"
    );
}

fn round_trip_f16(data: &[f32]) -> Vec<f32> {
    f32_slice_to_f16(data).into_iter().map(f16_bits_to_f32).collect()
}

/// The f16 bit pattern read back as bf16. Same bytes, different values.
fn f16_bits_read_as_bf16(data: &[f32]) -> Vec<f32> {
    f32_slice_to_f16(data).into_iter().map(bf16_bits_to_f32).collect()
}

/// What a bf16 kernel would load if it were pointed at an f32 buffer: `cols`
/// bf16 elements per row, so the byte stride is half the f32 row stride.
fn f32_buffer_read_as_bf16(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(data.len(), rows * cols);
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_ne_bytes()).collect();
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for p in 0..cols {
            let byte = i * cols * 2 + p * 2;
            let bits = u16::from_ne_bytes([bytes[byte], bytes[byte + 1]]);
            out[i * cols + p] = bf16_bits_to_f32(bits);
        }
    }
    out
}

/// The two references have to sit more than twice the accepted tolerance
/// apart. A result inside the real bound is then outside the other one. If
/// they are not that far apart, the dtype check cannot tell the kernels apart
/// and must fail rather than pass.
fn assert_references_separated(label: &str, right: &Reference, wrong: &Reference, k: usize, operand_u: f64) {
    assert_eq!(right.c.len(), wrong.c.len(), "{label}");
    let mut best = 0.0f64;
    for (&r, (&w, &mag)) in right.c.iter().zip(wrong.c.iter().zip(right.mag.iter())) {
        let tol = tolerance(k, mag, operand_u).max(1e-12);
        best = best.max((r - w).abs() / tol);
    }
    assert!(
        best > 2.0,
        "{label}: the bf16-bit reading is only {best:.3}× the accepted tolerance away from the real reference, so a numeric pass would not prove which kernel ran"
    );
}

fn same_bits(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite() && w.is_finite(), "{label}: non-finite at {i}");
        assert_eq!(g.to_bits(), w.to_bits(), "{label}: diverged at {i} ({g} vs {w})");
    }
}

fn assert_nan_tail(what: &str, buf: &[f32], logical: usize) {
    assert!(buf.len() >= logical, "{what}: buffer shorter than the logical matrix");
    for (i, v) in buf[logical..].iter().enumerate() {
        assert!(
            v.is_nan(),
            "{what}: guard element {i} past the logical matrix was overwritten with {v}"
        );
    }
}

fn poisoned_out(rt: &Arc<GpuRuntime>, m: usize) -> (Tensor, Tensor) {
    let bank = rt.alloc_tensor_f32(&[m * N + PAD]).expect("alloc C bank");
    bank.buffer.write_f32(&vec![f32::NAN; m * N + PAD]);
    let c = bank.view(&[m, N], 0);
    (bank, c)
}

fn encoded_at_least_once(rt: &GpuRuntime, what: &str) {
    let n = rt.take_dispatch_count();
    assert!(n >= 1, "{what}: returned without encoding a dispatch (count {n})");
}

/// M = 1 and M = 127 take the 64×64 bf16 kernel; M = 128 stays on 128×64.
/// Both geometries have to be finite, inside the bf16 GEMM bound of each
/// other and of the CPU reference, and the automatic dispatch has to match
/// the tile the selector names. A NaN guard past C catches a partial tile
/// that writes past N.
#[test]
fn short_m_bf16_ragged_extents_match_the_other_tile_and_the_reference() {
    attack_shape();
    with_gpu(|rt| {
        require_tensorops(rt);
        let b_h = round_trip_bf16(&random_f32(K * N, 0xA720));
        let b = tensor_bf16(rt, &[K, N], &b_h);
        for (case, m) in [1usize, 127, 128].into_iter().enumerate() {
            if m < 128 {
                assert_ne!(m % 64, 0, "M={m} must leave a partial 64-row tile");
            }
            let a_h = round_trip_bf16(&random_f32(m * K, 0xB720 + case as u64));
            let expect = reference(Layout::Nn, &a_h, &b_h, m, N, K);
            let a = tensor_bf16(rt, &[m, K], &a_h);

            let (wide_bank, wide) = poisoned_out(rt, m);
            let (narrow_bank, narrow) = poisoned_out(rt, m);
            let (auto_bank, auto) = poisoned_out(rt, m);

            rt.take_dispatch_count();
            gemm_tiled(&a, &b, &wide, GemmBackend::TensorOps, EpiTile::Wide).expect("wide");
            gemm_tiled(&a, &b, &narrow, GemmBackend::TensorOps, EpiTile::Narrow).expect("narrow");
            gemm(&a, &b, &auto, GemmBackend::TensorOps).expect("auto");
            encoded_at_least_once(rt, &format!("bf16 M={m}"));
            rt.synchronize().unwrap();

            let wide_full = wide_bank.buffer.read_f32();
            let narrow_full = narrow_bank.buffer.read_f32();
            let auto_full = auto_bank.buffer.read_f32();
            let logical = m * N;
            for (name, full) in [("wide", &wide_full), ("narrow", &narrow_full), ("auto", &auto_full)] {
                assert_nan_tail(&format!("bf16 M={m} {name}"), full, logical);
                assert_within_bound(
                    &format!("bf16 M={m} {name} {m}x{N}x{K}"),
                    &full[..logical],
                    &expect,
                    K,
                    0.0,
                );
            }
            let followed = if m < 128 { &narrow_full } else { &wide_full };
            same_bits(
                &format!("bf16 auto M={m} follows {}", if m < 128 { "64x64" } else { "128x64" }),
                &auto_full[..logical],
                &followed[..logical],
            );
        }
    });
}

/// f16 and f32 at a short-M shape must match their own reference, which is
/// built so it cannot also match a bf16 reading of the same bytes. Exact f32
/// cannot be forced onto a cooperative tile. Relaxed f32 may be forced onto
/// its own 64×64 kernel, but the automatic dispatch stays on 128×64.
#[test]
fn f16_and_f32_at_a_short_m_shape_are_not_the_bf16_kernel() {
    attack_shape();
    with_gpu(|rt| {
        require_tensorops(rt);
        for (case, m) in [1usize, 127].into_iter().enumerate() {
            let a_f = random_f32(m * K, 0xF160 + case as u64);
            let b_f = random_f32(K * N, 0xF161 + case as u64);
            let a16 = round_trip_f16(&a_f);
            let b16 = round_trip_f16(&b_f);
            let right = reference(Layout::Nn, &a16, &b16, m, N, K);
            let wrong = reference(
                Layout::Nn,
                &f16_bits_read_as_bf16(&a_f),
                &f16_bits_read_as_bf16(&b_f),
                m,
                N,
                K,
            );
            assert_references_separated(&format!("f16 M={m}"), &right, &wrong, K, 0.0);

            let a = rt.alloc_tensor_f16(&[m, K]).unwrap();
            let b = rt.alloc_tensor_f16(&[K, N]).unwrap();
            a.buffer.write_f16_bits(&f32_slice_to_f16(&a_f));
            b.buffer.write_f16_bits(&f32_slice_to_f16(&b_f));
            let (wide_bank, wide) = poisoned_out(rt, m);
            let (auto_bank, auto) = poisoned_out(rt, m);
            rt.take_dispatch_count();
            gemm_tiled(&a, &b, &wide, GemmBackend::TensorOps, EpiTile::Wide).expect("f16 wide");
            gemm(&a, &b, &auto, GemmBackend::TensorOps).expect("f16 auto");
            encoded_at_least_once(rt, &format!("f16 M={m}"));
            rt.synchronize().unwrap();
            let logical = m * N;
            let wide_full = wide_bank.buffer.read_f32();
            let auto_full = auto_bank.buffer.read_f32();
            assert_nan_tail(&format!("f16 M={m}"), &auto_full, logical);
            assert_within_bound(&format!("f16 M={m}"), &auto_full[..logical], &right, K, 0.0);
            same_bits(
                &format!("f16 auto stays on 128x64 M={m}"),
                &auto_full[..logical],
                &wide_full[..logical],
            );

            // Exact f32. A bf16 kernel pointed at these bytes reads a different
            // matrix; the reference separation fails the test if that reading
            // would still land inside the f32 bound.
            let a32 = tensor_f32(rt, &[m, K], &a_f);
            let b32 = tensor_f32(rt, &[K, N], &b_f);
            let right32 = reference(Layout::Nn, &a_f, &b_f, m, N, K);
            let wrong32 = reference(
                Layout::Nn,
                &f32_buffer_read_as_bf16(&a_f, m, K),
                &f32_buffer_read_as_bf16(&b_f, K, N),
                m,
                N,
                K,
            );
            assert_references_separated(&format!("f32 M={m}"), &right32, &wrong32, K, 0.0);
            let (c_bank, c) = poisoned_out(rt, m);
            rt.take_dispatch_count();
            gemm(&a32, &b32, &c, GemmBackend::TensorOps).expect("exact f32");
            encoded_at_least_once(rt, &format!("exact f32 M={m}"));
            rt.synchronize().unwrap();
            let got = c_bank.buffer.read_f32();
            assert_nan_tail(&format!("f32 M={m}"), &got, logical);
            assert_within_bound(&format!("exact f32 M={m}"), &got[..logical], &right32, K, 0.0);
        }

        // Exact f32 has no cooperative tile, so it cannot be steered onto the
        // bf16 64×64 kernel. The refusal has to happen before a dispatch.
        let m = 127usize;
        let a = tensor_f32(rt, &[m, K], &vec![1.0; m * K]);
        let b = tensor_f32(rt, &[K, N], &vec![1.0; K * N]);
        let c = rt.alloc_tensor_f32(&[m, N]).unwrap();
        rt.take_dispatch_count();
        let err = gemm_tiled(&a, &b, &c, GemmBackend::TensorOps, EpiTile::Narrow).expect_err("exact f32 narrow tile");
        assert!(err.contains("tile override"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0, "refused exact-f32 tile encoded a dispatch");

        // Relaxed f32 is on the cooperative path. Automatic selection at
        // M < 128 and N > 512 stays on the wide kernel and still matches an
        // f32 reference within the bf16-class bound, not a bf16 reading of
        // the f32 bytes.
        rt.set_relaxed_precision(true);
        let a_f = random_f32(m * K, 0x0E1A);
        let b_f = random_f32(K * N, 0x0E1B);
        let right = reference(Layout::Nn, &a_f, &b_f, m, N, K);
        let wrong = reference(
            Layout::Nn,
            &f32_buffer_read_as_bf16(&a_f, m, K),
            &f32_buffer_read_as_bf16(&b_f, K, N),
            m,
            N,
            K,
        );
        assert_references_separated("relaxed f32", &right, &wrong, K, U_BF16);
        let a = tensor_f32(rt, &[m, K], &a_f);
        let b = tensor_f32(rt, &[K, N], &b_f);
        let (wide_bank, wide) = poisoned_out(rt, m);
        let (auto_bank, auto) = poisoned_out(rt, m);
        rt.take_dispatch_count();
        gemm_tiled(&a, &b, &wide, GemmBackend::TensorOps, EpiTile::Wide).expect("relaxed wide");
        gemm(&a, &b, &auto, GemmBackend::TensorOps).expect("relaxed auto");
        encoded_at_least_once(rt, "relaxed f32");
        rt.synchronize().unwrap();
        let logical = m * N;
        let wide_full = wide_bank.buffer.read_f32();
        let auto_full = auto_bank.buffer.read_f32();
        assert_nan_tail("relaxed f32", &auto_full, logical);
        assert_within_bound("relaxed f32 auto", &auto_full[..logical], &right, K, U_BF16);
        same_bits(
            "relaxed f32 auto stays on 128x64",
            &auto_full[..logical],
            &wide_full[..logical],
        );
    });
}

/// `validate_gemm` rejects an output that aliases either input, and it runs
/// before the tile is chosen. The shape is one the short-M rule would
/// dispatch if validation were skipped.
#[test]
fn overlapping_short_m_output_is_rejected_before_dispatch() {
    attack_shape();
    with_gpu(|rt| {
        require_tensorops(rt);
        let m = 127usize;
        let bytes = m * N * 4 + 256;
        let shared = rt.alloc_buffer(bytes).expect("alloc overlap bank");
        let a_overlap = Tensor::from_buffer(rt, shared.clone(), &[m, K], DType::BF16, 0).unwrap();
        let c_full = Tensor::from_buffer(rt, shared.clone(), &[m, N], DType::F32, 0).unwrap();
        let b_disjoint = tensor_bf16(rt, &[K, N], &vec![0.0; K * N]);
        let a_disjoint = tensor_bf16(rt, &[m, K], &vec![0.0; m * K]);
        let b_overlap = Tensor::from_buffer(rt, shared.clone(), &[K, N], DType::BF16, 0).unwrap();
        // 256 is 64-byte aligned, so a 16-byte or 64-byte alignment check is
        // not what refuses this. The ranges still overlap: A occupies the
        // first 127*7*2 bytes.
        assert!(256 < m * K * 2, "partial overlap was not actually an overlap");
        let c_partial = Tensor::from_buffer(rt, shared.clone(), &[m, N], DType::F32, 256).unwrap();

        let overlap = "GEMM output must not overlap either input";
        rt.take_dispatch_count();
        for (label, a, b, c) in [
            ("A overlaps C", &a_overlap, &b_disjoint, &c_full),
            ("B overlaps C", &a_disjoint, &b_overlap, &c_full),
            ("A overlaps a shifted C", &a_overlap, &b_disjoint, &c_partial),
        ] {
            let err = gemm(a, b, c, GemmBackend::TensorOps).expect_err(label);
            assert_eq!(err, overlap, "{label}");
            let epi = Epilogue {
                beta: 1.0,
                ..Epilogue::default()
            };
            let err = gemm_epilogue(a, b, c, GemmBackend::TensorOps, epi).expect_err(label);
            assert_eq!(err, overlap, "{label} epilogue");
        }
        assert_eq!(
            rt.take_dispatch_count(),
            0,
            "an overlapping short-M GEMM encoded a dispatch"
        );
    });
}

/// Same absolute bound `tests/gemm_epilogue.rs` uses against the unfused
/// sequence. SiLU goes through Metal `exp`, so the GEMM's f32 gamma bound is
/// the wrong yardstick here.
fn epilogue_tol() -> f32 {
    2e-2 * (K as f32).sqrt()
}

fn epilogue_reference(a: &[f32], b: &[f32], c_prev: &[f32], bias: &[f32], m: usize, alpha: f32, beta: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; m * N];
    for i in 0..m {
        for j in 0..N {
            let mut acc = 0.0f32;
            for p in 0..K {
                acc += a[i * K + p] * b[p * N + j];
            }
            let v = alpha * acc + beta * c_prev[i * N + j] + bias[j];
            out[i * N + j] = v / (1.0 + (-v).exp());
        }
    }
    out
}

fn close_abs(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}");
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what}: non-finite at {i}");
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst <= tol,
        "{what}: worst |delta| = {worst} at {at} (got {} want {}), tol {tol}",
        got[at],
        want[at]
    );
}

/// Partial M and partial N, with a bias longer than N and a NaN guard past
/// both the bias and C. The edge tile has to ignore that guard. A short bias
/// and an f16 request for the 64×64 epilogue are refused before dispatch.
#[test]
fn short_m_epilogue_ragged_bias_keeps_its_guard_and_refuses_the_wrong_tile() {
    attack_shape();
    with_gpu(|rt| {
        require_tensorops(rt);
        let m = 127usize;
        let a = tensor_bf16(rt, &[m, K], &vec![0.0; m * K]);
        let b = tensor_bf16(rt, &[K, N], &vec![0.0; K * N]);
        let c = rt.alloc_tensor_f32(&[m, N]).unwrap();
        let short = tensor_f32(rt, &[N - 1], &vec![0.0; N - 1]);
        rt.take_dispatch_count();
        let err = gemm_epilogue(
            &a,
            &b,
            &c,
            GemmBackend::TensorOps,
            Epilogue {
                bias: Some(&short),
                ..Epilogue::default()
            },
        )
        .expect_err("short bias");
        assert!(err.contains("per-column"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0, "a short bias encoded a dispatch");

        let a16 = rt.alloc_tensor_f16(&[m, K]).unwrap();
        let b16 = rt.alloc_tensor_f16(&[K, N]).unwrap();
        let err = gemm_epilogue_tiled(
            &a16,
            &b16,
            &c,
            GemmBackend::TensorOps,
            Epilogue {
                beta: 1.0,
                ..Epilogue::default()
            },
            EpiTile::Narrow,
        )
        .expect_err("f16 narrow epilogue");
        assert!(err.contains("bf16"), "{err}");
        assert_eq!(
            rt.take_dispatch_count(),
            0,
            "a refused f16 narrow epilogue encoded a dispatch"
        );

        let alpha = 0.75f32;
        let beta = 0.5f32;
        let b_h = round_trip_bf16(&random_f32(K * N, 0xE520));
        let b = tensor_bf16(rt, &[K, N], &b_h);
        for (case, m) in [1usize, 127, 128].into_iter().enumerate() {
            let a_h = round_trip_bf16(&random_f32(m * K, 0xE521 + case as u64));
            let c_prev = random_f32(m * N, 0xE522 + case as u64);
            let bias_h = random_f32(N, 0xE523 + case as u64);
            let expect = epilogue_reference(&a_h, &b_h, &c_prev, &bias_h, m, alpha, beta);
            let a = tensor_bf16(rt, &[m, K], &a_h);

            let mut bias_raw = vec![f32::NAN; N + PAD];
            bias_raw[..N].copy_from_slice(&bias_h);
            let bias_bank = rt.alloc_tensor_f32(&[N + PAD]).unwrap();
            bias_bank.buffer.write_f32(&bias_raw);
            let bias = bias_bank.view(&[N], 0);

            let epi = Epilogue {
                alpha,
                beta,
                bias: Some(&bias),
                activation: Activation::Silu,
            };
            let prep = |bank_len: usize, prev: &[f32]| -> (Tensor, Tensor) {
                let bank = rt.alloc_tensor_f32(&[bank_len]).unwrap();
                let mut raw = vec![f32::NAN; bank_len];
                raw[..prev.len()].copy_from_slice(prev);
                bank.buffer.write_f32(&raw);
                let view = bank.view(&[m, N], 0);
                (bank, view)
            };
            let (wide_bank, wide) = prep(m * N + PAD, &c_prev);
            let (narrow_bank, narrow) = prep(m * N + PAD, &c_prev);
            let (auto_bank, auto) = prep(m * N + PAD, &c_prev);
            rt.take_dispatch_count();
            gemm_epilogue_tiled(&a, &b, &wide, GemmBackend::TensorOps, epi, EpiTile::Wide).expect("wide epi");
            gemm_epilogue_tiled(&a, &b, &narrow, GemmBackend::TensorOps, epi, EpiTile::Narrow).expect("narrow epi");
            gemm_epilogue(&a, &b, &auto, GemmBackend::TensorOps, epi).expect("auto epi");
            encoded_at_least_once(rt, &format!("epilogue M={m}"));
            rt.synchronize().unwrap();

            let logical = m * N;
            let wide_full = wide_bank.buffer.read_f32();
            let narrow_full = narrow_bank.buffer.read_f32();
            let auto_full = auto_bank.buffer.read_f32();
            let bias_after = bias_bank.buffer.read_f32();
            assert_nan_tail(&format!("bias M={m}"), &bias_after, N);
            let tol = epilogue_tol();
            for (name, full) in [("wide", &wide_full), ("narrow", &narrow_full), ("auto", &auto_full)] {
                assert_nan_tail(&format!("epi C M={m} {name}"), full, logical);
                close_abs(&format!("epi M={m} {name}"), &full[..logical], &expect, tol);
            }
            let followed = if m < 128 { &narrow_full } else { &wide_full };
            same_bits(&format!("epi auto M={m}"), &auto_full[..logical], &followed[..logical]);
        }
    });
}

/// Bias is read while C is written. Aliasing them is a cross-threadgroup race
/// once M spans two 64-row tiles, which is exactly M = 127 on the short-M
/// kernel. The validator has to refuse that before dispatch.
#[test]
fn short_m_epilogue_refuses_a_bias_that_aliases_the_output_before_dispatch() {
    attack_shape();
    with_gpu(|rt| {
        require_tensorops(rt);
        let m = 127usize;
        assert!(
            m > 64 && m < 128,
            "M=127 is the shape where the 64-row tile has two rows and the 128-row tile has one"
        );
        let a = tensor_bf16(rt, &[m, K], &vec![1.0; m * K]);
        let b = tensor_bf16(rt, &[K, N], &vec![1.0; K * N]);
        let bank = rt.alloc_tensor_f32(&[m * N]).unwrap();
        bank.buffer.write_f32(&vec![3.0; m * N]);
        let c = bank.view(&[m, N], 0);
        // First row of C, length N. It aliases the output the kernel stores.
        let bias = bank.view(&[N], 0);
        rt.take_dispatch_count();
        let err = gemm_epilogue(
            &a,
            &b,
            &c,
            GemmBackend::TensorOps,
            Epilogue {
                bias: Some(&bias),
                ..Epilogue::default()
            },
        )
        .expect_err("bias aliases C");
        assert_eq!(err, "GEMM epilogue: bias must not overlap the output");
        assert_eq!(rt.take_dispatch_count(), 0, "an aliased bias encoded a dispatch");
    });
}
