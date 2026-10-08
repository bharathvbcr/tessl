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
    cross_entropy_rows, cross_entropy_rows_accumulating, CeGrads, CeHidden, CeOutput, CeWorkspace, Reduction,
};
use tessl::gemm::GemmOperands;
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
    /// Elements of NaN before the hidden and weight tensors in their buffers,
    /// so a view read from the buffer's base instead of its offset shows.
    h_base: usize,
    w_base: usize,
    /// Sentinel elements before `dh`. Later vocabulary chunks must add into
    /// this view, not into element 0 of the underlying buffer.
    dh_base: usize,
    /// The four GEMMs' operands. Under `Bf16` the reference is formed from
    /// the bf16-rounded hidden rows and weight, which the logit walk reads
    /// exactly; `dh` and `dW` also round the softmax gradient to bf16.
    operands: GemmOperands,
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
            h_base: 0,
            w_base: 0,
            dh_base: 0,
            operands: GemmOperands::ExactF32,
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

/// `v` as a `shape` tensor `base` elements into a buffer whose first `base`
/// elements are NaN.
fn upload_at(rt: &Arc<GpuRuntime>, v: &[f32], dtype: DType, shape: &[usize], base: usize) -> Tensor {
    let mut all = vec![f32::NAN; base];
    all.extend_from_slice(v);
    let buf = upload(rt, &all, dtype);
    let size = if dtype == DType::BF16 { 2 } else { 4 };
    Tensor::from_buffer(rt, buf, shape, dtype, base * size).expect("tensor at offset")
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

fn assert_grad(label: &str, got: &[f32], want: &[f64], rel: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let peak = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let bound = rel * peak + 1e-12;
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
    let worst = got
        .per_row
        .iter()
        .zip(&want.per_row)
        .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
    eprintln!("{label}: max per-row loss err {worst:.2e}");
    for (i, (&g, &w)) in got.per_row.iter().zip(&want.per_row).enumerate() {
        assert!(close(g, w), "{label}: row {i} loss {g} want {w}");
    }
    assert!(
        close(got.loss, want.loss),
        "{label}: loss {} want {}",
        got.loss,
        want.loss
    );
}

fn sentinel_tensor(rt: &Arc<GpuRuntime>, shape: &[usize]) -> Tensor {
    sentinel_tensor_at(rt, shape, 0)
}

/// `shape` sitting `base` f32 elements into a buffer filled with [`SENTINEL`].
/// `base * 4` is the view's byte offset, which must be a multiple of the
/// 16-byte rule every GEMM operand view follows.
fn sentinel_tensor_at(rt: &Arc<GpuRuntime>, shape: &[usize], base: usize) -> Tensor {
    let n: usize = shape.iter().product();
    let storage = rt.alloc_buffer((base + n) * 4).expect("alloc");
    storage.write_f32(&vec![SENTINEL; base + n]);
    Tensor::from_buffer(rt, storage, shape, DType::F32, base * 4).expect("dh view")
}

/// Run `c` with and without gradients and check everything against the
/// reference. Returns the reference loss for callers that compare cases.
fn check(rt: &Arc<GpuRuntime>, c: &Case) -> f64 {
    let label = format!(
        "V={} H={} chunk={} n={} h={:?} w={:?} {:?} scale={} {:?}",
        c.vocab,
        c.hidden,
        c.chunk,
        c.rows.len(),
        c.h_dtype,
        c.w_dtype,
        c.reduction,
        c.scale,
        c.operands
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
    let want = match c.operands {
        GemmOperands::ExactF32 => reference(c, &h, &w),
        GemmOperands::Bf16 => reference(c, &round_trip_bf16(&h), &round_trip_bf16(&w)),
    };
    // Exact f32: the measured error sits near 1e-6. bf16 operands: the softmax
    // gradient is rounded to bf16 (relative 2^-9) before dh and dW, whose sums
    // cancel (the gradient sums to zero over the vocabulary), so the bound is
    // 2^-7 of the peak, written before the first run.
    let grad_bound = match c.operands {
        GemmOperands::ExactF32 => 1e-4,
        GemmOperands::Bf16 => 2f64.powi(-7),
    };

    let ht = upload_at(rt, &h, c.h_dtype, &[c.rows_total, c.ld], c.h_base);
    let wt = upload_at(rt, &w, c.w_dtype, &[c.vocab, c.hidden], c.w_base);
    let hid = CeHidden {
        rows: &ht,
        off: c.off as u32,
    };
    let n = c.rows.len();
    // A workspace larger than the call needs, as a training loop reuses one.
    let ws = CeWorkspace::new(rt, n as u32 + 3, c.hidden as u32, c.chunk as u32, c.w_dtype).expect("workspace");

    let fwd = cross_entropy_rows(rt, hid, &wt, &c.rows, &c.targets, c.reduction, c.operands, &ws, None)
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_loss(&format!("{label} (forward only)"), &fwd, &want);

    let dh = sentinel_tensor_at(rt, &[n, c.hidden], c.dh_base);
    let dw = sentinel_tensor(rt, &[c.vocab, c.hidden]);
    let out = cross_entropy_rows(
        rt,
        hid,
        &wt,
        &c.rows,
        &c.targets,
        c.reduction,
        c.operands,
        &ws,
        Some(CeGrads {
            dh: &dh,
            dw: &dw,
            scale: c.scale,
        }),
    )
    .unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_loss(&label, &out, &want);
    assert_grad(
        &format!("{label} dh"),
        &dh.read_f32().expect("read"),
        &want.dh,
        grad_bound,
    );
    if c.dh_base > 0 {
        let raw = dh.buffer.read_f32();
        assert!(
            raw[..c.dh_base].iter().all(|&x| x == SENTINEL),
            "{label}: a later vocabulary chunk wrote dh at buffer offset 0"
        );
    }
    assert_grad(
        &format!("{label} dW"),
        &dw.read_f32().expect("read"),
        &want.dw,
        grad_bound,
    );
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
        check(
            rt,
            &Case {
                rows: vec![5],
                targets: vec![511],
                vocab: 512,
                ..Case::small(3)
            },
        );
        check(
            rt,
            &Case {
                vocab: 37,
                chunk: 1,
                targets: vec![0, 36, 17, 1],
                ..Case::small(4)
            },
        );
        // One chunk covering the whole vocabulary, and a chunk wider than it.
        check(
            rt,
            &Case {
                chunk: 997,
                ..Case::small(5)
            },
        );
        check(
            rt,
            &Case {
                chunk: 4096,
                ..Case::small(6)
            },
        );
        // The hidden columns a window inside a wider row.
        check(
            rt,
            &Case {
                ld: 96,
                off: 24,
                h_dtype: DType::BF16,
                ..Case::small(7)
            },
        );
        // Tensors that start inside their buffers (a torch view's storage
        // offset), NaN before them: odd for the gather, 16-byte aligned for
        // the weight's GEMM operands.
        check(
            rt,
            &Case {
                h_base: 3,
                w_base: 8,
                h_dtype: DType::BF16,
                w_dtype: DType::BF16,
                ..Case::small(8)
            },
        );
        check(
            rt,
            &Case {
                h_base: 5,
                w_base: 4,
                ..Case::small(9)
            },
        );
        // dh itself starts inside its buffer. The first vocabulary chunk
        // writes through the tensor view; every later chunk must add there
        // too. 16 f32s is 64 bytes, a multiple of the 16-byte GEMM rule.
        check(
            rt,
            &Case {
                dh_base: 16,
                ..Case::small(10)
            },
        );
    });
}

#[test]
fn the_running_log_sum_exp_survives_large_and_late_maxima() {
    with_gpu(|rt| {
        // Logits in the hundreds: exp without the running max overflows f32.
        check(
            rt,
            &Case {
                h_scale: 12.0,
                ..Case::small(11)
            },
        );
        // The dominant logit in the last, partial chunk for row 0 and in the
        // first chunk for row 2: the running max rises late for one and never
        // for the other, and the target of row 0 is not the maximum.
        check(
            rt,
            &Case {
                h_scale: 3.0,
                plant: Some((990, 0, 2.0)),
                ..Case::small(12)
            },
        );
        check(
            rt,
            &Case {
                h_scale: 3.0,
                plant: Some((5, 2, 2.0)),
                ..Case::small(13)
            },
        );
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
        assert!(
            loss < 1e-3,
            "a dominant target logit should give a near-zero loss, got {loss}"
        );
    });
}

/// bf16 GEMM operands, against the f64 reference on the bf16-rounded operands
/// they read: across the chunk walk, every dtype pairing, a chunk of 1 and a
/// late dominant logit. Then that the rounding happens at all: on f32 inputs
/// the exact and bf16 references are further apart than twice the loss bound,
/// so no one GPU result could pass both.
#[test]
fn bf16_operands_match_the_reference_on_the_rounded_operands() {
    with_gpu(|rt| {
        let bf = |c: Case| Case {
            operands: GemmOperands::Bf16,
            ..c
        };
        check(rt, &bf(Case::small(31)));
        for (hd, wd) in [
            (DType::BF16, DType::F32),
            (DType::F32, DType::BF16),
            (DType::BF16, DType::BF16),
        ] {
            check(
                rt,
                &bf(Case {
                    h_dtype: hd,
                    w_dtype: wd,
                    reduction: Reduction::Sum,
                    scale: 0.7,
                    ..Case::small(32)
                }),
            );
        }
        check(
            rt,
            &bf(Case {
                vocab: 37,
                chunk: 1,
                targets: vec![0, 36, 17, 1],
                ..Case::small(33)
            }),
        );
        check(
            rt,
            &bf(Case {
                h_scale: 3.0,
                plant: Some((990, 0, 2.0)),
                ..Case::small(34)
            }),
        );

        let exact = check(rt, &Case::small(35));
        let rounded = check(rt, &bf(Case::small(35)));
        let bound = 1e-5 + 1e-5 * exact.abs();
        assert!(
            (exact - rounded).abs() > 2.0 * bound,
            "bf16 operands moved the loss by only {:.2e} (bound {bound:.2e}): nothing was rounded",
            (exact - rounded).abs()
        );
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
    // rows, targets, m, s, tlogit; the gathered rows; one logit chunk; the
    // widened weight chunk. (dh accumulates in its GEMM: no `dh_part`.)
    let want = 4 * (5 * 8 + 8 * 2048 + 8 * 4096 + 4096 * 2048);
    assert_eq!(b, want);
    assert_eq!(
        CeWorkspace::bytes_for(8, 2048, 4096, DType::F32),
        want - 4 * 4096 * 2048
    );
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
        let ht = tensor_at(rt, &h, &[10, 64]);
        let wt = tensor_at(rt, &w, &[997, 64]);
        let hid = CeHidden { rows: &ht, off: 0 };
        let ws = CeWorkspace::new(rt, 4, 64, 128, DType::F32).expect("workspace");
        let run = |hid: CeHidden<'_>,
                   wt: &Tensor,
                   rows: &[u32],
                   targets: &[u32],
                   ws: &CeWorkspace,
                   grads: Option<CeGrads<'_>>| {
            cross_entropy_rows(
                rt,
                hid,
                wt,
                rows,
                targets,
                Reduction::Mean,
                GemmOperands::ExactF32,
                ws,
                grads,
            )
            .map(|o| o.loss)
        };
        let expect_err = |r: Result<f64, String>, needle: &str| match r {
            Ok(l) => panic!("expected an error containing {needle:?}, got loss {l}"),
            Err(e) => assert!(e.contains(needle), "error {e:?} lacks {needle:?}"),
        };

        expect_err(run(hid, &wt, &[], &[], &ws, None), "no supervised rows");
        expect_err(run(hid, &wt, &[1, 2], &[1], &ws, None), "2 rows but 1 targets");
        expect_err(run(hid, &wt, &[0; 5], &[0; 5], &ws, None), "exceed the workspace");
        expect_err(run(hid, &wt, &[1, 10], &[0, 0], &ws, None), "rows[1] = 10");
        expect_err(run(hid, &wt, &[1, 2], &[996, 997], &ws, None), "targets[1] = 997");
        expect_err(
            run(CeHidden { off: 8, ..hid }, &wt, &[1], &[1], &ws, None),
            "exceeds ld",
        );
        let flat = ht.try_view(&[640], 0).unwrap();
        expect_err(run(CeHidden { rows: &flat, off: 0 }, &wt, &[1], &[1], &ws, None), "2-D");
        let w16 = rt.alloc_tensor_f16(&[997, 64]).unwrap();
        expect_err(run(hid, &w16, &[1], &[1], &ws, None), "f32 or bf16");
        let narrow = wt.try_view(&[997, 32], 0).unwrap();
        expect_err(run(hid, &narrow, &[1], &[1], &ws, None), "workspace is for hidden 64");
        let ws_bf16 = CeWorkspace::new(rt, 4, 64, 128, DType::BF16).expect("workspace");
        expect_err(run(hid, &wt, &[1], &[1], &ws_bf16, None), "workspace is for");
        // Already inside with_gpu's lock, so the second runtime is made here.
        let other = GpuRuntime::new().expect("second runtime");
        let foreign = tensor_at(&other, &h, &[10, 64]);
        expect_err(
            run(CeHidden { rows: &foreign, off: 0 }, &wt, &[1], &[1], &ws, None),
            "hidden belongs to another runtime",
        );

        expect_err(
            CeWorkspace::new(rt, 4, 60, 128, DType::F32).map(|_| 0.0),
            "multiple of 8",
        );
        expect_err(CeWorkspace::new(rt, 0, 64, 128, DType::F32).map(|_| 0.0), "non-zero");
        expect_err(CeWorkspace::new(rt, 4, 64, 0, DType::F32).map(|_| 0.0), "non-zero");
        expect_err(CeWorkspace::new(rt, 4, 64, 8, DType::F16).map(|_| 0.0), "f32 or bf16");

        // Gradients: shape, scale, and overlap with each other and the inputs.
        let dh = rt.alloc_tensor_f32(&[2, 64]).expect("alloc");
        let dw = rt.alloc_tensor_f32(&[997, 64]).expect("alloc");
        let g = |dh, dw, scale| Some(CeGrads { dh, dw, scale });
        expect_err(
            run(hid, &wt, &[1], &[1], &ws, g(&dh, &dw, 1.0)),
            "dh must be f32 [1, 64]",
        );
        expect_err(run(hid, &wt, &[1, 2], &[1, 2], &ws, g(&dh, &dw, f32::NAN)), "finite");
        let dw_as_dh = dw.try_view(&[2, 64], 0).expect("view");
        expect_err(
            run(hid, &wt, &[1, 2], &[1, 2], &ws, g(&dw_as_dh, &dw, 1.0)),
            "dh and dw overlap",
        );
        expect_err(
            run(hid, &wt, &[1, 2], &[1, 2], &ws, g(&dh, &wt, 1.0)),
            "dw overlaps the weight",
        );
        let dh_in_h = ht.try_view(&[2, 64], 64).expect("view");
        expect_err(
            run(hid, &wt, &[1, 2], &[1, 2], &ws, g(&dh_in_h, &dw, 1.0)),
            "dh overlaps the hidden",
        );
        // One storage, disjoint windows: hidden rows [0, 4), dh rows [8, 10).
        let h_head = ht.try_view(&[4, 64], 0).expect("view");
        let dh_tail = ht.try_view(&[2, 64], 8 * 64).expect("view");
        run(
            CeHidden { rows: &h_head, off: 0 },
            &wt,
            &[1, 2],
            &[1, 2],
            &ws,
            g(&dh_tail, &dw, 1.0),
        )
        .expect("disjoint windows of one buffer are not an overlap");

        // Non-finite inputs surface as an error, not a NaN loss.
        let mut bad = h.clone();
        bad[64 + 3] = f32::NAN;
        let bad_t = tensor_at(rt, &bad, &[10, 64]);
        expect_err(
            run(CeHidden { rows: &bad_t, off: 0 }, &wt, &[1], &[1], &ws, None),
            "not finite",
        );

        // Relaxed-precision GEMMs would silently break the exactness contract.
        rt.set_relaxed_precision(true);
        let r = run(hid, &wt, &[1], &[1], &ws, None);
        rt.set_relaxed_precision(false);
        expect_err(r, "relaxed precision");

        // After every rejection the entry point still computes.
        run(hid, &wt, &[1, 2], &[3, 4], &ws, g(&dh, &dw, 1.0)).expect("a valid call still runs");
    });
}

fn tensor_at(rt: &Arc<GpuRuntime>, v: &[f32], shape: &[usize]) -> Tensor {
    upload_at(rt, v, DType::F32, shape, 0)
}

/// `cross_entropy_rows_accumulating` adds the weight gradient into what `dW`
/// holds: within 8 units of f32 rounding (of the tensor's largest
/// `|C0| + |dW|`) of `C0 + dW` from the overwriting call, whose dh and loss
/// it reproduces bit for bit. Several vocabulary chunks, both operand lanes.
#[test]
fn the_accumulating_call_adds_dw_into_what_it_holds() {
    with_gpu(|rt| {
        let (n, hidden, vocab, chunk) = (6usize, 64usize, 200usize, 48u32);
        let h = random_f32(n * hidden, 0xce01);
        let w = random_f32(vocab * hidden, 0xce02);
        let c0 = random_f32(vocab * hidden, 0xce03);
        let rows: Vec<u32> = (0..n as u32).collect();
        let targets: Vec<u32> = (0..n as u32).map(|i| (i * 37 + 5) % vocab as u32).collect();
        let hid = upload_at(rt, &h, DType::F32, &[n, hidden], 0);
        let wt = upload_at(rt, &w, DType::F32, &[vocab, hidden], 0);
        let ws = CeWorkspace::new(rt, n as u32, hidden as u32, chunk, DType::F32).unwrap();
        for op in [GemmOperands::ExactF32, GemmOperands::Bf16] {
            let run = |dw: &Tensor, dh: &Tensor, add: bool| {
                let grads = CeGrads { dh, dw, scale: 0.5 };
                let hidden = CeHidden { rows: &hid, off: 0 };
                let out = if add {
                    cross_entropy_rows_accumulating(rt, hidden, &wt, &rows, &targets, Reduction::Mean, op, &ws, grads)
                } else {
                    cross_entropy_rows(rt, hidden, &wt, &rows, &targets, Reduction::Mean, op, &ws, Some(grads))
                };
                out.unwrap()
            };
            let (dw1, dh1) = (sentinel_tensor(rt, &[vocab, hidden]), sentinel_tensor(rt, &[n, hidden]));
            let l1 = run(&dw1, &dh1, false);
            let dw2 = upload_at(rt, &c0, DType::F32, &[vocab, hidden], 0);
            let dh2 = sentinel_tensor(rt, &[n, hidden]);
            let l2 = run(&dw2, &dh2, true);
            assert_eq!(l1.loss.to_bits(), l2.loss.to_bits(), "{op:?}: the loss differs");
            assert!(
                dh1.read_f32().unwrap() == dh2.read_f32().unwrap(),
                "{op:?}: dh differs between the overwriting and accumulating calls"
            );
            let (g1, g2) = (dw1.read_f32().unwrap(), dw2.read_f32().unwrap());
            let scale = c0
                .iter()
                .zip(&g1)
                .map(|(a, b)| f64::from(a.abs() + b.abs()))
                .fold(0.0, f64::max);
            let u = f64::from(f32::EPSILON) / 2.0;
            for (k, ((&prev, &fresh), &got)) in c0.iter().zip(&g1).zip(&g2).enumerate() {
                let want = f64::from(prev) + f64::from(fresh);
                let err = (f64::from(got) - want).abs();
                assert!(
                    err <= 8.0 * u * scale,
                    "{op:?}: dW[{k}] = {got}, want {want} (err {err:.3e})"
                );
            }
        }
    });
}
