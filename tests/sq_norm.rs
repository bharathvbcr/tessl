//! `tessl::sq_norm`: the sum of squares of a set of windows in two
//! dispatches, for any number of windows, f32 and bf16, dense and strided.
//! Deterministic whatever the windows' order, finite for any finite values,
//! a non-finite value reported as such, and every refusal made before
//! anything is encoded.
#![cfg(target_os = "macos")]

use std::sync::Arc;

use tessl::sq_norm::{decode_sq_norm, sq_norm, SqNormPlan, SqWindow, SQ_OUT_WORDS};
use tessl::tensor::{bf16_bits_to_f32, f32_to_bf16_bits};
use tessl::{DType, GpuRuntime, Tensor};

/// Deterministic values in roughly [-1, 1): a 32-bit hash.
fn val(seed: u64, i: usize) -> f32 {
    let mut x = (seed ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 31;
    ((x >> 40) as f32 / (1u64 << 23) as f32) - 1.0
}

fn f32_tensor(rt: &Arc<GpuRuntime>, data: &[f32]) -> Tensor {
    let t = rt.alloc_tensor_f32(&[data.len()]).unwrap();
    t.buffer.contents_f32()[..data.len()].copy_from_slice(data);
    t
}

/// bf16 rounding of `data`; returns the tensor and the values it holds.
fn bf16_tensor(rt: &Arc<GpuRuntime>, data: &[f32]) -> (Tensor, Vec<f32>) {
    let t = rt.alloc_tensor_bf16(&[data.len()]).unwrap();
    let bits: Vec<u16> = data.iter().map(|&x| f32_to_bf16_bits(x)).collect();
    t.buffer.contents_u16()[..data.len()].copy_from_slice(&bits);
    (t, bits.iter().map(|&b| bf16_bits_to_f32(b)).collect())
}

/// The squares of the window's elements, in f64.
fn squares(vals: &[f32], rows: usize, width: usize, ld: usize, off: usize) -> f64 {
    let mut s = 0.0f64;
    for r in 0..rows {
        for c in 0..width {
            let x = f64::from(vals[r * ld + off + c]);
            s += x * x;
        }
    }
    s
}

/// 30 windows (more than ojas-metal's 24 slots): dense f32 of many sizes,
/// strided f32, bf16, one at a byte offset. One binder scope (both
/// kernels), the f64 sum
/// within 2^-18 relative, the same bits for the reversed order and on a
/// rerun, and the caller-buffer encode at offsets giving the same bits.
#[test]
fn many_windows_match_f64_in_any_order() {
    let rt = GpuRuntime::new().unwrap();
    let mut keep = Vec::new();
    let mut specs = Vec::new(); // (tensor index, dtype, rows, width, ld, off, byte_off)
    let mut want = 0.0f64;
    for i in 0..26 {
        let n = 1 + (i * 977) % 9000;
        let data: Vec<f32> = (0..n).map(|j| val(i as u64, j)).collect();
        want += squares(&data, 1, n, n, 0);
        specs.push((keep.len(), DType::F32, 1, n, n, 0, 0));
        keep.push(f32_tensor(&rt, &data));
    }
    // Strided: 3 rows of 5000 at ld 5003 from element 2.
    let data: Vec<f32> = (0..3 * 5003 + 2).map(|j| val(100, j)).collect();
    want += squares(&data, 3, 5000, 5003, 2);
    specs.push((keep.len(), DType::F32, 3, 5000, 5003, 2, 0));
    keep.push(f32_tensor(&rt, &data));
    // bf16, dense and strided.
    let (t, held) = bf16_tensor(&rt, &(0..9001).map(|j| 3.0 * val(200, j)).collect::<Vec<_>>());
    want += squares(&held, 1, 9001, 9001, 0) + squares(&held, 2, 4000, 4500, 7);
    specs.push((keep.len(), DType::BF16, 1, 9001, 9001, 0, 0));
    specs.push((keep.len(), DType::BF16, 2, 4000, 4500, 7, 0));
    keep.push(t);
    // A window 64 bytes into its buffer.
    let data: Vec<f32> = (0..1016).map(|j| val(300, j)).collect();
    want += squares(&data[16..], 1, 1000, 1000, 0);
    specs.push((keep.len(), DType::F32, 1, 1000, 1000, 0, 64));
    keep.push(f32_tensor(&rt, &data));
    assert_eq!(specs.len(), 30);

    let windows: Vec<SqWindow<'_>> = specs
        .iter()
        .map(|&(k, dtype, rows, width, ld, off, byte_off)| SqWindow {
            buf: &keep[k].buffer,
            byte_off,
            dtype,
            rows,
            width,
            ld,
            off,
        })
        .collect();
    let _ = rt.take_dispatch_count();
    let got = sq_norm(&rt, &windows, "test").unwrap();
    // `take_dispatch_count` counts binder scopes: both kernels go in one.
    assert_eq!(rt.take_dispatch_count(), 1);
    assert!((got - want).abs() <= 2f64.powi(-18) * want, "{got:e} vs {want:e}");
    assert_eq!(
        sq_norm(&rt, &windows, "test").unwrap().to_bits(),
        got.to_bits(),
        "rerun"
    );
    let reversed: Vec<_> = windows.iter().rev().copied().collect();
    assert_eq!(
        sq_norm(&rt, &reversed, "test").unwrap().to_bits(),
        got.to_bits(),
        "reversed"
    );

    // The same plan into caller buffers at offsets.
    let plan = SqNormPlan::new(&rt, &windows, "test").unwrap();
    let words = plan.table_words();
    let table = rt.alloc_buffer(16 + words.len() * 4).unwrap();
    table.contents_u32()[4..4 + words.len()].copy_from_slice(words);
    let scratch = rt.alloc_buffer(8 * plan.pieces() + 4 * SQ_OUT_WORDS + 32).unwrap();
    let out_at = 8 * plan.pieces() + 16;
    plan.encode(&rt, (&table, 16), (&scratch, 8), (&scratch, out_at), "test")
        .unwrap();
    rt.synchronize().unwrap();
    let r = scratch.contents_u32();
    let at = decode_sq_norm(&r[out_at / 4..out_at / 4 + SQ_OUT_WORDS]).unwrap();
    assert_eq!(at.to_bits(), got.to_bits(), "caller buffers");
}

/// ojas-metal's huge-value contract, now this module's: 1e30 gradients
/// (squares past f32) and 1e-30 ones (squares below its subnormals) give
/// the f64 sum within 2^-18, finite and nonzero.
#[test]
fn huge_and_tiny_values_are_finite_and_right() {
    let rt = GpuRuntime::new().unwrap();
    let big = 1.0e30f32;
    let mut a = vec![big; 5000];
    a[4999] = -2.0 * big;
    let tiny: Vec<f32> = (0..7000).map(|j| 1.0e-30 * (1.0 + val(9, j))).collect();
    let (ta, tb, tc) = (
        f32_tensor(&rt, &a),
        f32_tensor(&rt, &[0.5 * big; 3]),
        f32_tensor(&rt, &tiny),
    );
    let huge = sq_norm(
        &rt,
        &[
            SqWindow::dense(&ta.buffer, 0, DType::F32, 5000),
            SqWindow::dense(&tb.buffer, 0, DType::F32, 3),
        ],
        "test",
    )
    .unwrap();
    let want = squares(&a, 1, 5000, 5000, 0) + 3.0 * f64::from(0.5 * big).powi(2);
    assert!((huge - want).abs() <= 2f64.powi(-18) * want, "{huge:e} vs {want:e}");
    let small = sq_norm(&rt, &[SqWindow::dense(&tc.buffer, 0, DType::F32, 7000)], "test").unwrap();
    let want = squares(&tiny, 1, 7000, 7000, 0);
    assert!(
        small > 0.0 && (small - want).abs() <= 2f64.powi(-18) * want,
        "{small:e} vs {want:e}"
    );
}

/// A NaN anywhere is NaN, an infinity (and no NaN) +inf, from any window.
#[test]
fn a_non_finite_value_is_reported() {
    let rt = GpuRuntime::new().unwrap();
    let base: Vec<f32> = (0..20_000).map(|j| val(5, j)).collect();
    let run = |bad: &[(usize, f32)]| {
        let mut d = base.clone();
        for &(i, x) in bad {
            d[i] = x;
        }
        let t = f32_tensor(&rt, &d);
        let ok = f32_tensor(&rt, &base[..100]);
        sq_norm(
            &rt,
            &[
                SqWindow::dense(&ok.buffer, 0, DType::F32, 100),
                SqWindow::dense(&t.buffer, 0, DType::F32, 20_000),
            ],
            "test",
        )
        .unwrap()
    };
    assert_eq!(run(&[(19_999, f32::INFINITY)]), f64::INFINITY);
    assert_eq!(run(&[(3, f32::NEG_INFINITY)]), f64::INFINITY);
    assert!(run(&[(12_345, f32::NAN)]).is_nan());
    assert!(run(&[(1, f32::INFINITY), (19_000, -f32::NAN)]).is_nan());
}

/// No windows: 0, with nothing encoded. Every refusal names its cause and
/// encodes nothing: a window past its buffer, an empty one, an f16 one, one
/// of another runtime; caller buffers too short, misaligned, overlapping or
/// of another runtime.
#[test]
fn refusals_encode_nothing() {
    let rt = GpuRuntime::new().unwrap();
    let other = GpuRuntime::new().unwrap();
    let t = f32_tensor(&rt, &[1.0; 100]);
    let foreign = f32_tensor(&other, &[1.0; 100]);
    let half = rt.alloc_tensor_f16(&[100]).unwrap();
    let _ = rt.take_dispatch_count();
    assert_eq!(sq_norm(&rt, &[], "test").unwrap(), 0.0);
    let err = |w: SqWindow<'_>, want: &str| {
        let m = sq_norm(&rt, &[SqWindow::dense(&t.buffer, 0, DType::F32, 100), w], "test").unwrap_err();
        assert!(m.contains(want) && m.contains("window 1"), "{want}: {m}");
    };
    err(SqWindow::dense(&t.buffer, 0, DType::F32, 101), "does not fit");
    err(SqWindow::dense(&t.buffer, 8, DType::F32, 99), "does not fit");
    err(SqWindow::dense(&t.buffer, 0, DType::F32, 0), "empty window");
    err(SqWindow::dense(&half.buffer, 0, DType::F16, 100), "not f32 or bf16");
    err(
        SqWindow::dense(&foreign.buffer, 0, DType::F32, 100),
        "belongs to another runtime",
    );
    assert_eq!(rt.take_dispatch_count(), 0, "a refused window encoded work");

    let plan = SqNormPlan::new(&rt, &[SqWindow::dense(&t.buffer, 0, DType::F32, 100)], "test").unwrap();
    assert_eq!(plan.pieces(), 1);
    let table = rt.alloc_buffer_from_u32(plan.table_words()).unwrap();
    let parts = rt.alloc_buffer(8).unwrap();
    let out = rt.alloc_buffer(4 * SQ_OUT_WORDS).unwrap();
    let big = rt.alloc_buffer(256).unwrap();
    let elsewhere = other.alloc_buffer(256).unwrap();
    let _ = rt.take_dispatch_count();
    let enc = |table, parts, out, want: &str| {
        let m = plan.encode(&rt, table, parts, out, "test").unwrap_err();
        assert!(m.contains(want), "{want}: {m}");
    };
    enc((&table, 0), (&parts, 4), (&out, 0), "parts needs 8 bytes at offset 4");
    enc((&table, 0), (&parts, 0), (&out, 4), "out needs 24 bytes");
    enc((&table, 0), (&big, 2), (&out, 0), "parts needs");
    enc((&table, 0), (&big, 0), (&big, 4), "parts and out overlap");
    enc((&big, 0), (&big, 32), (&out, 0), "table and parts overlap");
    enc(
        (&table, 0),
        (&parts, 0),
        (&elsewhere, 0),
        "out: buffer belongs to another runtime",
    );
    assert_eq!(rt.take_dispatch_count(), 0, "a refusal encoded work");
    assert_eq!(plan.parts_words(), 2 * plan.pieces());
    // Disjoint ranges of one buffer are fine.
    plan.encode(&rt, (&table, 0), (&big, 0), (&big, 8), "test").unwrap();
    rt.synchronize().unwrap();
    assert_eq!(decode_sq_norm(&big.contents_u32()[2..2 + SQ_OUT_WORDS]).unwrap(), 100.0);
}

#[test]
fn parts_words_is_two_per_piece_and_encode_refuses_one_short() {
    let rt = GpuRuntime::new().unwrap();
    let n = 3 * tessl::sq_norm::SQ_PIECE;
    let t = f32_tensor(&rt, &vec![1.0; n]);
    let plan = SqNormPlan::new(&rt, &[SqWindow::dense(&t.buffer, 0, DType::F32, n)], "test").unwrap();
    assert_eq!(plan.pieces(), 3);
    assert_eq!(plan.parts_words(), 2 * plan.pieces());
    let table = rt.alloc_buffer_from_u32(plan.table_words()).unwrap();
    let out = rt.alloc_buffer(4 * SQ_OUT_WORDS).unwrap();
    let short = rt.alloc_buffer(4 * (plan.parts_words() - 1)).unwrap();
    let m = plan
        .encode(&rt, (&table, 0), (&short, 0), (&out, 0), "test")
        .unwrap_err();
    assert!(
        m.contains("parts needs 24 bytes at offset 0 of a 20-byte buffer"),
        "{m}"
    );
    let exact = rt.alloc_buffer(4 * plan.parts_words()).unwrap();
    plan.encode(&rt, (&table, 0), (&exact, 0), (&out, 0), "test").unwrap();
    rt.synchronize().unwrap();
    assert_eq!(decode_sq_norm(&out.contents_u32()[..SQ_OUT_WORDS]).unwrap(), n as f64);
}
