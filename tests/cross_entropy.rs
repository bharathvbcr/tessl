//! `tessl::cross_entropy::cross_entropy_rows` against an f64 reference.
//!
//! The reference forms the full `[n, vocab]` logits the kernel never does, in
//! f64, from exactly the operands the GPU reads (bf16 inputs are rounded on the
//! host first). The loss, `dh` and `dW` are all compared, over shapes chosen for
//! the chunk walk's edges: a partial last chunk, targets on the first and last
//! column of a chunk, a single row, a duplicated row, a hidden window inside a
//! wider row, the maximum logit arriving in a late chunk (so the running
//! log-sum-exp has to rescale), logits large enough to overflow a naive `exp`,
//! and the real 248 320-token vocabulary.
//!
//! Bounds: the loss within `1e-5` absolute plus `1e-5` relative, gradients
//! within `1e-4 * max|ref|`. Every GEMM is exact f32 and the exponentials are
//! `precise::`, so the measured error sits near `1e-6` relative.

mod common;

use std::sync::Arc;

use common::{buf, buf_bf16, random_f32, round_trip_bf16, with_gpu};
use tessl::cross_entropy::{
    cross_entropy_rows, CeGrads, CeHidden, CeOutput, CeWeight, CeWorkspace, Reduction,
};
use tessl::tensor::{DType, GpuBuffer, Tensor};
use tessl::GpuRuntime;

/// Written to the gradient outputs first, so an element never written shows.
const SENTINEL: f32 = -7.25e27;

struct Case {
    rows_total: usize,
    ld: usize,
    off: usize,
    hidden: usize,
    vocab: usize,
    chunk: usize,
    rows: Vec<u32>,
    targets: Vec<u32>,
    h_dtype: DType,
    w_dtype: DType,
    reduction: Reduction,
    scale: f32,
    h_scale: f32,
    seed: u64,
    /// `(vocab row, supplied row index, factor)`: set that weight row to
    /// `factor` times the supplied row's hidden vector, planting a dominant
    /// logit in a chosen chunk.
    plant: Option<(usize, usize, f32)>,
}

impl Case {
    fn small(seed: u64) -> Self {
        Self {
            rows_total: 10,
            ld: 64,
            off: 0,
            hidden: 64,
            vocab: 997,
            chunk: 128,
            rows: vec![3, 0, 9, 3],
            targets: vec![0, 127, 128, 996],
            h_dtype: DType::F32,
            w_dtype: DType::F32,
            reduction: Reduction::Mean,
            scale: 1.0,
            h_scale: 1.0,
            seed,
            plant: None,
        }
    }
}

struct Reference {
    per_row: Vec<f64>,
    loss: f64,
    dh: Vec<f64>,
    dw: Vec<f64>,
}

fn maybe_round(v: Vec<f32>, dtype: DType) -> Vec<f32> {
    if dtype == DType::BF16 {
        round_trip_bf16(&v)
    } else {
        v
    }
}

fn upload(rt: &Arc<GpuRuntime>, v: &[f32], dtype: DType) -> GpuBuffer {
    if dtype == DType::BF16 {
        buf_bf16(rt, v)
    } else {
        buf(rt, v)
    }
}

fn reference(c: &Case, h: &[f32], w: &[f32]) -> Reference {
    let (n, hs, vs) = (c.rows.len(), c.hidden, c.vocab);
    let hrow = |i: usize| &h[c.rows[i] as usize * c.ld + c.off..][..hs];
    let scale = f64::from(c.scale)
        / match c.reduction {
            Reduction::Mean => n as f64,
            Reduction::Sum => 1.0,
        };
    let mut per_row = Vec::with_capacity(n);
    let mut dh = vec![0.0f64; n * hs];
    let mut dw = vec![0.0f64; vs * hs];
    for i in 0..n {
        let hr = hrow(i);
        let logits: Vec<f64> = (0..vs)
            .map(|j| {
                let wr = &w[j * hs..][..hs];
                hr.iter().zip(wr).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum()
            })
            .collect();
        let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = m + logits.iter().map(|l| (l - m).exp()).sum::<f64>().ln();
        let t = c.targets[i] as usize;
        per_row.push(lse - logits[t]);
        for (j, &l) in logits.iter().enumerate() {
            let g = ((l - lse).exp() - f64::from(u8::from(j == t))) * scale;
            let wr = &w[j * hs..][..hs];
            for k in 0..hs {
                dh[i * hs + k] += g * f64::from(wr[k]);
                dw[j * hs + k] += g * f64::from(hr[k]);
            }
        }
    }
    let total: f64 = per_row.iter().sum();
    let loss = match c.reduction {
        Reduction::Mean => total / n as f64,
        Reduction::Sum => total,
    };
    Reference { per_row, loss, dh, dw }
}

fn assert_grad(label: &str, got: &[f32], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let peak = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let bound = 1e-4 * peak + 1e-12;
    let mut worst = (0.0f64, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g != SENTINEL, "{label}[{i}]: never written");
        assert!(g.is_finite(), "{label}[{i}]: non-finite {g}");
        let e = (f64::from(g) - w).abs();
        if e > worst.0 {
            worst = (e, i);
        }
    }
    eprintln!("{label}: max err / max|ref| = {:.2e}", worst.0 / peak.max(1e-300));
    assert!(
        worst.0 <= bound,
        "{label}: max err {:.3e} at {} (got {} want {}), bound {bound:.3e}",
        worst.0,
        worst.1,
        got[worst.1],
        want[worst.1]
    );
}

fn assert_loss(label: &str, got: &CeOutput, want: &Reference) {
    let close = |g: f64, w: f64| (g - w).abs() <= 1e-5 + 1e-5 * w.abs();
    let worst = got.per_row.iter().zip(&want.per_row).fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
    eprintln!("{label}: max per-row loss err {worst:.2e}");
    for (i, (&g, &w)) in got.per_row.iter().zip(&want.per_row).enumerate() {
        assert!(close(g, w), "{label}: row {i} loss {g} want {w}");
    }
    assert!(close(got.loss, want.loss), "{label}: loss {} want {}", got.loss, want.loss);
}

fn sentinel_tensor(rt: &Arc<GpuRuntime>, shape: &[usize]) -> Tensor {
    let t = rt.alloc_tensor_f32(shape).expect("alloc");
    t.write_f32(&vec![SENTINEL; shape.iter().product()]).expect("write");
    t
}

/// Run `c` with and without gradients and check everything against the
/// reference. Returns the reference loss for callers that compare cases.
fn check(rt: &Arc<GpuRuntime>, c: &Case) -> f64 {
    let label = format!(
        "V={} H={} chunk={} n={} h={:?} w={:?} {:?} scale={}",
        c.vocab,
        c.hidden,
        c.chunk,
        c.rows.len(),
        c.h_dtype,
        c.w_dtype,
        c.reduction,
        c.scale
    );
    let h: Vec<f32> = random_f32(c.rows_total * c.ld, c.seed)
        .into_iter()
        .map(|x| x * c.h_scale)
        .collect();
    let h = maybe_round(h, c.h_dtype);
    let mut w: Vec<f32> = random_f32(c.vocab * c.hidden, c.seed ^ 0x9e37)
        .into_iter()
        .map(|x| x * 0.25)
        .collect();
    if let Some((vrow, i, factor)) = c.plant {
        let src = c.rows[i] as usize * c.ld + c.off;
        for k in 0..c.hidden {
            w[vrow * c.hidden + k] = factor * h[src + k];
        }
    }
    let w = maybe_round(w, c.w_dtype);
    let want = reference(c, &h, &w);

    let hb = upload(rt, &h, c.h_dtype);
    let wb = upload(rt, &w, c.w_dtype);
    let hid = CeHidden {
        buf: &hb,
        dtype: c.h_dtype,
        rows: c.rows_total as u32,
        ld: c.ld as u32,
        off: c.off as u32,
    };
    let wt = CeWeight { buf: &wb, dtype: c.w_dtype, vocab: c.vocab as u32 };
    let n = c.rows.len();
    // A workspace larger than the call needs, as a training loop reuses one.
    let ws = CeWorkspace::new(rt, n as u32 + 3, c.hidden as u32, c.chunk as u32, c.w_dtype)
        .expect("workspace");

    let fwd = cross_entropy_rows(
        rt, hid, wt, c.hidden as u32, &c.rows, &c.targets, c.reduction, &ws, None,
    )
    .unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_loss(&format!("{label} (forward only)"), &fwd, &want);

    let dh = sentinel_tensor(rt, &[n, c.hidden]);
    let dw = sentinel_tensor(rt, &[c.vocab, c.hidden]);
    let out = cross_entropy_rows(
        rt,
        hid,
        wt,
        c.hidden as u32,
        &c.rows,
        &c.targets,
        c.reduction,
        &ws,
        Some(CeGrads { dh: &dh, dw: &dw, scale: c.scale }),
    )
    .unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_loss(&label, &out, &want);
    assert_grad(&format!("{label} dh"), &dh.read_f32().expect("read"), &want.dh);
    assert_grad(&format!("{label} dW"), &dw.read_f32().expect("read"), &want.dw);
    want.loss
}

#[test]
fn matches_the_f64_reference_across_the_chunk_walk() {
    with_gpu(|rt| {
        // Partial last chunk (997 = 7 * 128 + 101), targets on chunk edges,
        // a duplicated row.
        check(rt, &Case::small(1));
        // Every dtype pairing and both reductions, with an upstream scale.
        for (hd, wd) in [
            (DType::BF16, DType::F32),
            (DType::F32, DType::BF16),
            (DType::BF16, DType::BF16),
        ] {
            for reduction in [Reduction::Mean, Reduction::Sum] {
                check(
                    rt,
                    &Case {
                        h_dtype: hd,
                        w_dtype: wd,
                        reduction,
                        scale: 0.7,
                        ..Case::small(2)
                    },
                );
            }
        }
        // One row; vocabulary an exact multiple of the chunk; chunk of 1.
        check(rt, &Case { rows: vec![5], targets: vec![511], vocab: 512, ..Case::small(3) });
        check(rt, &Case { vocab: 37, chunk: 1, targets: vec![0, 36, 17, 1], ..Case::small(4) });
        // One chunk covering the whole vocabulary, and a chunk wider than it.
        check(rt, &Case { chunk: 997, ..Case::small(5) });
        check(rt, &Case { chunk: 4096, ..Case::small(6) });
        // The hidden columns a window inside a wider row.
        check(rt, &Case { ld: 96, off: 24, h_dtype: DType::BF16, ..Case::small(7) });
    });
}

#[test]
fn the_running_log_sum_exp_survives_large_and_late_maxima() {
    with_gpu(|rt| {
        // Logits in the hundreds: exp without the running max overflows f32.
        check(rt, &Case { h_scale: 12.0, ..Case::small(11) });
        // The dominant logit in the last, partial chunk for row 0 and in the
        // first chunk for row 2: the running max rises late for one and never
        // for the other, and the target of row 0 is not the maximum.
        check(rt, &Case { h_scale: 3.0, plant: Some((990, 0, 2.0)), ..Case::small(12) });
        check(rt, &Case { h_scale: 3.0, plant: Some((5, 2, 2.0)), ..Case::small(13) });
        // The planted maximum is the target itself: the loss approaches 0.
        let loss = check(
            rt,
            &Case {
                h_scale: 3.0,
                plant: Some((996, 0, 4.0)),
                reduction: Reduction::Sum,
                rows: vec![3],
                targets: vec![996],
                ..Case::small(14)
            },
        );
        assert!(loss < 1e-3, "a dominant target logit should give a near-zero loss, got {loss}");
    });
}

#[test]
fn matches_the_reference_at_the_real_vocabulary_and_hidden_width() {
    with_gpu(|rt| {
        // Qwen3.5's 248 320 tokens, at a reduced hidden width to keep the f64
        // reference quick; 248 320 = 121 * 2048 + 512.
        check(
            rt,
            &Case {
                vocab: 248_320,
                hidden: 128,
                ld: 128,
                chunk: 2048,
                rows: vec![0, 7, 2],
                targets: vec![248_319, 0, 123_456],
                w_dtype: DType::BF16,
                h_dtype: DType::BF16,
                ..Case::small(21)
            },
        );
        // The real hidden width (2048), at a small vocabulary.
        check(
            rt,
            &Case {
                vocab: 300,
                hidden: 2048,
                ld: 2048,
                chunk: 128,
                targets: vec![299, 0, 128, 255],
                w_dtype: DType::BF16,
                h_dtype: DType::BF16,
                ..Case::small(22)
            },
        );
    });
}

#[test]
fn the_workspace_is_bounded_by_rows_and_chunk_not_vocabulary() {
    // The whole point: scratch does not grow with the vocabulary.
    let b = CeWorkspace::bytes_for(8, 2048, 4096, DType::BF16);
    let want = 4 * (5 * 8 + 2 * 8 * 2048 + 8 * 4096 + 4096 * 2048);
    assert_eq!(b, want);
    assert_eq!(CeWorkspace::bytes_for(8, 2048, 4096, DType::F32), want - 4 * 4096 * 2048);
    with_gpu(|rt| {
        let ws = CeWorkspace::new(rt, 8, 2048, 4096, DType::BF16).expect("workspace");
        assert_eq!(ws.bytes(), b);
        assert_eq!(ws.chunk(), 4096);
    });
}

#[test]
fn rejects_what_it_cannot_compute() {
    with_gpu(|rt| {
        let c = Case::small(31);
        let h = random_f32(c.rows_total * c.ld, 1);
        let w = random_f32(c.vocab * c.hidden, 2);
        let hb = buf(rt, &h);
        let wb = buf(rt, &w);
        let hid = CeHidden { buf: &hb, dtype: DType::F32, rows: 10, ld: 64, off: 0 };
        let wt = CeWeight { buf: &wb, dtype: DType::F32, vocab: 997 };
        let ws = CeWorkspace::new(rt, 4, 64, 128, DType::F32).expect("workspace");
        let run = |hid: CeHidden<'_>,
                   wt: CeWeight<'_>,
                   rows: &[u32],
                   targets: &[u32],
                   ws: &CeWorkspace,
                   grads: Option<CeGrads<'_>>| {
            cross_entropy_rows(rt, hid, wt, 64, rows, targets, Reduction::Mean, ws, grads)
                .map(|o| o.loss)
        };
        let expect_err = |r: Result<f64, String>, needle: &str| match r {
            Ok(l) => panic!("expected an error containing {needle:?}, got loss {l}"),
            Err(e) => assert!(e.contains(needle), "error {e:?} lacks {needle:?}"),
        };

        expect_err(run(hid, wt, &[], &[], &ws, None), "no supervised rows");
        expect_err(run(hid, wt, &[1, 2], &[1], &ws, None), "2 rows but 1 targets");
        expect_err(run(hid, wt, &[0; 5], &[0; 5], &ws, None), "exceed the workspace");
        expect_err(run(hid, wt, &[1, 10], &[0, 0], &ws, None), "rows[1] = 10");
        expect_err(run(hid, wt, &[1, 2], &[996, 997], &ws, None), "targets[1] = 997");
        expect_err(
            run(CeHidden { off: 8, ..hid }, wt, &[1], &[1], &ws, None),
            "exceeds ld",
        );
        expect_err(
            run(CeHidden { rows: 11, ..hid }, wt, &[1], &[1], &ws, None),
            "cross_entropy hidden",
        );
        expect_err(
            run(hid, CeWeight { vocab: 998, ..wt }, &[1], &[1], &ws, None),
            "cross_entropy weight",
        );
        let ws_bf16 = CeWorkspace::new(rt, 4, 64, 128, DType::BF16).expect("workspace");
        expect_err(run(hid, wt, &[1], &[1], &ws_bf16, None), "workspace is for");

        expect_err(
            CeWorkspace::new(rt, 4, 60, 128, DType::F32).map(|_| 0.0),
            "multiple of 8",
        );
        expect_err(CeWorkspace::new(rt, 0, 64, 128, DType::F32).map(|_| 0.0), "non-zero");
        expect_err(CeWorkspace::new(rt, 4, 64, 0, DType::F32).map(|_| 0.0), "non-zero");
        expect_err(CeWorkspace::new(rt, 4, 64, 8, DType::F16).map(|_| 0.0), "f32 or bf16");

        // Gradients: shape, scale, aliasing of each other and of the inputs.
        let dh = rt.alloc_tensor_f32(&[2, 64]).expect("alloc");
        let dw = rt.alloc_tensor_f32(&[997, 64]).expect("alloc");
        let g = |dh, dw, scale| Some(CeGrads { dh, dw, scale });
        expect_err(run(hid, wt, &[1], &[1], &ws, g(&dh, &dw, 1.0)), "dh must be f32 [1, 64]");
        expect_err(run(hid, wt, &[1, 2], &[1, 2], &ws, g(&dh, &dw, f32::NAN)), "finite");
        let dw_as_dh = dw.try_view(&[2, 64], 0).expect("view");
        expect_err(
            run(hid, wt, &[1, 2], &[1, 2], &ws, g(&dw_as_dh, &dw, 1.0)),
            "writable buffers dh and dw overlap",
        );
        let w_tensor =
            Tensor::from_buffer(rt, wb.clone(), &[997, 64], DType::F32, 0).expect("wrap");
        expect_err(
            run(hid, wt, &[1, 2], &[1, 2], &ws, g(&dh, &w_tensor, 1.0)),
            "writable buffer dw overlaps read-only buffer weight",
        );

        // Non-finite inputs surface as an error, not a NaN loss.
        let mut bad = h.clone();
        bad[64 + 3] = f32::NAN;
        let bad_b = buf(rt, &bad);
        expect_err(
            run(CeHidden { buf: &bad_b, ..hid }, wt, &[1], &[1], &ws, None),
            "not finite",
        );

        // Relaxed-precision GEMMs would silently break the exactness contract.
        rt.set_relaxed_precision(true);
        let r = run(hid, wt, &[1], &[1], &ws, None);
        rt.set_relaxed_precision(false);
        expect_err(r, "relaxed precision");

        // After every rejection the entry point still computes.
        run(hid, wt, &[1, 2], &[3, 4], &ws, g(&dh, &dw, 1.0)).expect("a valid call still runs");
    });
}
