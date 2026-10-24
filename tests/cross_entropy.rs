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

use common::ce::{ce_oracle, other_classes_lse};
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
        let (loss, grad) = ce_oracle(&logits, c.targets[i] as usize);
        per_row.push(loss);
        for (j, &gj) in grad.iter().enumerate() {
            let g = gj * scale;
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
    let ws =
        CeWorkspace::new(rt, n as u32 + 3, c.hidden as u32, c.chunk as u32, c.w_dtype, c.operands).expect("workspace");

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
    use GemmOperands::{Bf16, ExactF32};
    // The whole point: scratch does not grow with the vocabulary.
    // rows, targets, m, s, tlogit; the gathered rows; one logit chunk. (dh
    // accumulates in its GEMM: no `dh_part`.)
    let base = 4 * (5 * 8 + 8 * 2048 + 8 * 4096);
    // Exact f32: a bf16 weight's chunk widened to f32; an f32 weight is read
    // as it is.
    let exact_bf16 = base + 4 * 4096 * 2048;
    assert_eq!(CeWorkspace::bytes_for(8, 2048, 4096, DType::BF16, ExactF32), exact_bf16);
    assert_eq!(CeWorkspace::bytes_for(8, 2048, 4096, DType::F32, ExactF32), base);
    // bf16 operands: the gathered rows and one dlogits chunk rounded once,
    // and no f32 copy of a bf16 weight (the 32 MiB `w32` here, 80 MiB at the
    // 4B's hidden 2560 and 8192-column chunks); an f32 weight's chunk is
    // rounded to bf16.
    let rounded = 2 * (8 * 2048 + 8 * 4096);
    assert_eq!(CeWorkspace::bytes_for(8, 2048, 4096, DType::BF16, Bf16), base + rounded);
    assert_eq!(
        CeWorkspace::bytes_for(8, 2048, 4096, DType::F32, Bf16),
        base + rounded + 2 * 4096 * 2048
    );
    assert!(CeWorkspace::bytes_for(8, 2048, 4096, DType::BF16, Bf16) < exact_bf16);
    with_gpu(|_| {
        for (dtype, op) in [
            (DType::BF16, ExactF32),
            (DType::BF16, Bf16),
            (DType::F32, Bf16),
            (DType::F32, ExactF32),
        ] {
            // A fresh runtime: nothing in its pool to reuse, so every buffer
            // the workspace makes is a fresh allocation.
            let rt = &GpuRuntime::new().expect("runtime");
            let before = rt.current_allocated_bytes();
            let ws = CeWorkspace::new(rt, 8, 2048, 4096, dtype, op).expect("workspace");
            let held = rt.current_allocated_bytes() - before;
            assert_eq!(ws.bytes(), CeWorkspace::bytes_for(8, 2048, 4096, dtype, op));
            assert_eq!(ws.chunk(), 4096);
            assert_eq!(ws.operands(), op);
            // What the device charges, as the step's memory bound counts it:
            // the bound rounds the five 32-byte buffers up to 256 B each,
            // which the device charges less for below 16 KiB.
            let bound = CeWorkspace::allocated_bytes_for(8, 2048, 4096, dtype, op);
            assert!(
                held <= bound && bound - held <= 5 * 256,
                "{dtype:?} weight, {op:?}: held {held} B, bound {bound} B"
            );
        }
    });
}

/// A bf16 weight goes to bf16-operand GEMMs as it is: the loss, `dh` and
/// `dW` are bit for bit those of the same weight widened to f32 (exact)
/// and then rounded per chunk by the workspace (exact back), which is the
/// path every bf16 weight took before. Likewise on exact f32, where the
/// widening is the workspace's. Several chunks, a partial last one.
#[test]
fn a_bf16_weight_computes_the_bits_of_its_exact_widening() {
    with_gpu(|rt| {
        let (n, hidden, vocab, chunk) = (5usize, 64usize, 300usize, 64u32);
        let h = random_f32(n * hidden, 0xb16a);
        let w = round_trip_bf16(&random_f32(vocab * hidden, 0xb16b));
        let rows: Vec<u32> = vec![0, 3, 1, 4, 2];
        let targets: Vec<u32> = vec![0, 299, 64, 63, 200];
        let hid = upload_at(rt, &h, DType::F32, &[n, hidden], 0);
        let w16 = upload_at(rt, &w, DType::BF16, &[vocab, hidden], 0);
        let w32 = upload_at(rt, &w, DType::F32, &[vocab, hidden], 0);
        for op in [GemmOperands::Bf16, GemmOperands::ExactF32] {
            let run = |wt: &Tensor| {
                let ws = CeWorkspace::new(rt, n as u32, hidden as u32, chunk, wt.dtype, op).unwrap();
                let (dh, dw) = (sentinel_tensor(rt, &[n, hidden]), sentinel_tensor(rt, &[vocab, hidden]));
                let grads = CeGrads {
                    dh: &dh,
                    dw: &dw,
                    scale: 0.75,
                };
                let hidden = CeHidden { rows: &hid, off: 0 };
                let out =
                    cross_entropy_rows(rt, hidden, wt, &rows, &targets, Reduction::Mean, op, &ws, Some(grads)).unwrap();
                let bits = |t: &Tensor| t.read_f32().unwrap().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                (out.loss.to_bits(), bits(&dh), bits(&dw))
            };
            let (a, b) = (run(&w16), run(&w32));
            assert_eq!(a.0, b.0, "{op:?}: the loss differs");
            assert!(a.1 == b.1, "{op:?}: dh differs");
            assert!(a.2 == b.2, "{op:?}: dW differs");
        }
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
        let ws = CeWorkspace::new(rt, 4, 64, 128, DType::F32, GemmOperands::ExactF32).expect("workspace");
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
        let ws_bf16 = CeWorkspace::new(rt, 4, 64, 128, DType::BF16, GemmOperands::ExactF32).expect("workspace");
        expect_err(run(hid, &wt, &[1], &[1], &ws_bf16, None), "workspace is for");
        // A workspace holds one lane's operand copies.
        let ws_lane = CeWorkspace::new(rt, 4, 64, 128, DType::F32, GemmOperands::Bf16).expect("workspace");
        expect_err(
            run(hid, &wt, &[1], &[1], &ws_lane, None),
            "workspace is for Bf16 operands",
        );
        // Already inside with_gpu's lock, so the second runtime is made here.
        let other = GpuRuntime::new().expect("second runtime");
        let foreign = tensor_at(&other, &h, &[10, 64]);
        expect_err(
            run(CeHidden { rows: &foreign, off: 0 }, &wt, &[1], &[1], &ws, None),
            "hidden belongs to another runtime",
        );

        let new = |max_rows, hidden, chunk, dtype| {
            CeWorkspace::new(rt, max_rows, hidden, chunk, dtype, GemmOperands::ExactF32).map(|_| 0.0)
        };
        expect_err(new(4, 60, 128, DType::F32), "multiple of 8");
        expect_err(new(0, 64, 128, DType::F32), "non-zero");
        expect_err(new(4, 64, 0, DType::F32), "non-zero");
        expect_err(new(4, 64, 8, DType::F16), "f32 or bf16");

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
        for op in [GemmOperands::ExactF32, GemmOperands::Bf16] {
            let ws = CeWorkspace::new(rt, n as u32, hidden as u32, chunk, DType::F32, op).unwrap();
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

// ---------------------------------------------------------------------------
// Softmax collapse (Barron, "A Fix for Softmax and Cross-Entropy").
//
// The logits are chosen directly: hidden width `LW`, hidden row `r` is the unit
// vector `e_r`, and weight row `j` holds `z[r][j]` in column `r`, so the exact
// f32 GEMM returns exactly `z` (one product of 1.0 and `z`, the rest 0 * 0).
// With `h = e_r`, `dW[j][r]` is the gradient of the loss with respect to
// `z[r][j]`, so the whole `[rows, vocab]` gradient is read back from `dW`.
// ---------------------------------------------------------------------------

/// Hidden width of the logit-row cases: at least the number of rows in one.
const LW: usize = 16;

/// Run `cross_entropy_rows` on the rows of logits `z` (every row the same
/// length), returning the output and the gradient of the loss with respect
/// to each logit, `[row][class]`.
#[allow(clippy::too_many_arguments)]
fn run_logit_rows(
    rt: &Arc<GpuRuntime>,
    z: &[Vec<f32>],
    targets: &[u32],
    chunk: usize,
    reduction: Reduction,
    scale: f32,
    h_dtype: DType,
    operands: GemmOperands,
) -> Result<(CeOutput, Vec<Vec<f32>>), String> {
    let (n, vocab) = (z.len(), z[0].len());
    assert!(n <= LW);
    let mut h = vec![0.0f32; n * LW];
    for r in 0..n {
        h[r * LW + r] = 1.0;
    }
    let mut w = vec![0.0f32; vocab * LW];
    for (r, zr) in z.iter().enumerate() {
        for (j, &l) in zr.iter().enumerate() {
            w[j * LW + r] = l;
        }
    }
    let ht = upload_at(rt, &h, h_dtype, &[n, LW], 0);
    let wt = upload_at(rt, &w, DType::F32, &[vocab, LW], 0);
    let rows: Vec<u32> = (0..n as u32).collect();
    let ws = CeWorkspace::new(rt, n as u32, LW as u32, chunk as u32, DType::F32, operands)?;
    let dh = sentinel_tensor(rt, &[n, LW]);
    let dw = sentinel_tensor(rt, &[vocab, LW]);
    let out = cross_entropy_rows(
        rt,
        CeHidden { rows: &ht, off: 0 },
        &wt,
        &rows,
        targets,
        reduction,
        operands,
        &ws,
        Some(CeGrads {
            dh: &dh,
            dw: &dw,
            scale,
        }),
    )?;
    let dwv = dw.read_f32()?;
    let grads = (0..n).map(|r| (0..vocab).map(|j| dwv[j * LW + r]).collect()).collect();
    Ok((out, grads))
}

/// Run the rows and compare loss, mean/sum reduction and every gradient with
/// the f64 oracle on the operands the GPU reads. Gradients are held to
/// `3e-5` of the row's peak (`2^-7` under bf16 operands, which round the
/// gradient before `dW`), and, on exact f32, the row must sum to zero within
/// `4 * eps * (1 + |L|)` of its peak: `L`, the log-sum-exp of the other
/// classes, is formed in f32, and its rounding moves every `exp(z_j - L)` by
/// that relative amount. A target gradient the oracle puts above the f32
/// normal range must be nonzero and negative: that is the collapse.
#[allow(clippy::too_many_arguments)]
fn check_logit_rows(
    rt: &Arc<GpuRuntime>,
    label: &str,
    z: &[Vec<f32>],
    targets: &[u32],
    chunk: usize,
    reduction: Reduction,
    scale: f32,
    h_dtype: DType,
    operands: GemmOperands,
) -> Vec<Vec<f32>> {
    let label = format!("{label} chunk={chunk} {reduction:?} scale={scale} h={h_dtype:?} {operands:?}");
    let (out, grads) = run_logit_rows(rt, z, targets, chunk, reduction, scale, h_dtype, operands)
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    let n = z.len();
    let eff = f64::from(scale)
        / match reduction {
            Reduction::Mean => n as f64,
            Reduction::Sum => 1.0,
        };
    let (grad_rel, check_sum) = match operands {
        GemmOperands::ExactF32 => (3e-5, true),
        GemmOperands::Bf16 => (2f64.powi(-7), false),
    };
    let mut total = 0.0;
    for r in 0..n {
        let zr: Vec<f64> = match operands {
            GemmOperands::ExactF32 => z[r].iter().map(|&x| f64::from(x)).collect(),
            GemmOperands::Bf16 => round_trip_bf16(&z[r]).into_iter().map(f64::from).collect(),
        };
        let t = targets[r] as usize;
        let (loss, g) = ce_oracle(&zr, t);
        total += loss;
        let got_loss = out.per_row[r];
        assert!(
            (got_loss - loss).abs() <= 1e-4 * loss.abs() + 1e-60,
            "{label}: row {r} loss {got_loss:e} want {loss:e}"
        );
        let peak = g.iter().fold(0.0f64, |m, x| m.max(x.abs())) * eff.abs();
        let mut sum = 0.0f64;
        for (j, (&got, &gj)) in grads[r].iter().zip(&g).enumerate() {
            let want = gj * eff;
            assert!(got != SENTINEL && got.is_finite(), "{label}: row {r} class {j}: {got}");
            assert!(
                (f64::from(got) - want).abs() <= grad_rel * peak + 1e-36,
                "{label}: row {r} class {j}: grad {got:e} want {want:e} (peak {peak:e})"
            );
            sum += f64::from(got);
        }
        let want_t = g[t] * eff;
        if want_t.abs() > 1e-30 {
            let got_t = grads[r][t];
            assert!(
                got_t != 0.0 && (got_t < 0.0) == (want_t < 0.0),
                "{label}: row {r}: the target gradient collapsed to {got_t:e}, want {want_t:e}"
            );
        }
        if check_sum && peak > 0.0 {
            let lo = other_classes_lse(&zr, t);
            let tol = 4.0 * f64::from(f32::EPSILON) * (1.0 + lo.abs()) * peak;
            assert!(
                sum.abs() <= tol,
                "{label}: row {r} sums to {sum:e}, want 0 within {tol:e} (peak {peak:e})"
            );
        }
    }
    let want_loss = match reduction {
        Reduction::Mean => total / n as f64,
        Reduction::Sum => total,
    };
    assert!(
        (out.loss - want_loss).abs() <= 1e-4 * want_loss.abs() + 1e-60,
        "{label}: reduced loss {:e} want {want_loss:e}",
        out.loss
    );
    grads
}

/// The paper's `[9, -9, -9]` with the target on class 0, and its rotations so
/// the target is also the middle and last class. `1 + exp(-17.3)` is 1 in
/// f32, so `softmax_t - 1` is 0 while the other two gradients are 1.5e-8: the
/// row must instead carry `g_t = -3.0e-8` and sum to 0.
#[test]
fn a_target_past_the_f32_cliff_keeps_its_gradient() {
    with_gpu(|rt| {
        let z = vec![vec![9.0f32, -9.0, -9.0], vec![-9.0, 9.0, -9.0], vec![-9.0, -9.0, 9.0]];
        let targets = [0u32, 1, 2];
        for (h_dtype, operands) in [
            (DType::F32, GemmOperands::ExactF32),
            (DType::BF16, GemmOperands::ExactF32),
            (DType::F32, GemmOperands::Bf16),
            (DType::BF16, GemmOperands::Bf16),
        ] {
            for chunk in [3, 1] {
                let g = check_logit_rows(rt, "paper", &z, &targets, chunk, Reduction::Sum, 1.0, h_dtype, operands);
                // The numbers the paper's row has: q = sigmoid(-(9 - (-9 + ln 2))).
                let q = 1.0 / (1.0 + (9.0f64 - (-9.0 + 2f64.ln())).exp());
                let rel = if operands == GemmOperands::Bf16 {
                    2f64.powi(-7)
                } else {
                    3e-5
                };
                assert!(
                    (f64::from(g[0][0]) + q).abs() <= rel * q,
                    "target gradient {:e}, want {:e}",
                    g[0][0],
                    -q
                );
                if operands == GemmOperands::ExactF32 {
                    let sum: f32 = g[0].iter().sum();
                    assert!(sum.abs() <= 1e-6 * q as f32, "row sum {sum:e}");
                }
            }
        }
    });
}

/// Margins on both sides of the cliff (0, 16, 17, 20, 40, 100 and their
/// negatives), a row of ties, the target in the first, a middle and the last
/// class, chunks that split the row and one that does not; mean reduction and
/// the sum over rows.
#[test]
fn the_margin_sweep_matches_the_oracle_across_the_cliff() {
    with_gpu(|rt| {
        let others = [0.0f32, -1.0, -2.5, 0.5, 3.0, -0.25];
        let lo = other_classes_lse(&others.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(), usize::MAX);
        let margins = [
            0.0f64, 16.0, -16.0, 17.0, -17.0, 20.0, -20.0, 40.0, -40.0, 100.0, -100.0,
        ];
        for t in [0usize, 3, 6] {
            let mut z: Vec<Vec<f32>> = margins
                .iter()
                .map(|&m| {
                    let mut row = others.to_vec();
                    row.insert(t, (lo + m) as f32);
                    row
                })
                .collect();
            z.push(vec![3.0f32; 7]);
            let targets = vec![t as u32; z.len()];
            for chunk in [2, 7] {
                for (reduction, scale) in [(Reduction::Sum, 1.0), (Reduction::Mean, 2.5)] {
                    check_logit_rows(
                        rt,
                        &format!("margins target={t}"),
                        &z,
                        &targets,
                        chunk,
                        reduction,
                        scale,
                        DType::F32,
                        GemmOperands::ExactF32,
                    );
                }
            }
            check_logit_rows(
                rt,
                &format!("margins target={t}"),
                &z,
                &targets,
                2,
                Reduction::Sum,
                1.0,
                DType::BF16,
                GemmOperands::Bf16,
            );
        }
    });
}

/// A vocabulary of several chunks (the last partial), the target in the first
/// chunk, the middle, the last and on a chunk edge, collapsed margins either
/// way, chunk sizes that do and do not divide the vocabulary.
#[test]
fn collapsed_margins_survive_a_multi_chunk_walk() {
    with_gpu(|rt| {
        let vocab = 300usize;
        let targets = [0u32, 150, 299, 64];
        let margins = [20.0f64, 20.0, -20.0, 30.0];
        let z: Vec<Vec<f32>> = targets
            .iter()
            .zip(margins)
            .enumerate()
            .map(|(r, (&t, m))| {
                let mut row: Vec<f32> = random_f32(vocab, 0xc011 + r as u64)
                    .into_iter()
                    .map(|x| x * 5.0)
                    .collect();
                let zr: Vec<f64> = row.iter().map(|&x| f64::from(x)).collect();
                row[t as usize] = (other_classes_lse(&zr, t as usize) + m) as f32;
                row
            })
            .collect();
        for chunk in [64, 7, 300] {
            for (reduction, scale) in [(Reduction::Mean, 2.5), (Reduction::Sum, 1.0)] {
                check_logit_rows(
                    rt,
                    "multi-chunk",
                    &z,
                    &targets,
                    chunk,
                    reduction,
                    scale,
                    DType::F32,
                    GemmOperands::ExactF32,
                );
            }
        }
        check_logit_rows(
            rt,
            "multi-chunk",
            &z,
            &targets,
            64,
            Reduction::Mean,
            1.0,
            DType::BF16,
            GemmOperands::Bf16,
        );
    });
}

/// `-inf` logits. Some non-targets `-inf` is an ordinary row; a row whose
/// other classes are all `-inf` has `L = -inf`, margin `+inf`: loss 0 and
/// every gradient 0 (not NaN from `exp(-inf - -inf)`).
#[test]
fn minus_infinite_logits_and_a_row_of_only_the_target() {
    with_gpu(|rt| {
        let ninf = f32::NEG_INFINITY;
        let z = vec![
            vec![ninf, 2.0, ninf, ninf, ninf],
            vec![1.0, ninf, 2.0, ninf, 0.0],
            vec![0.5, ninf, ninf, ninf, ninf],
            vec![1.0, 2.0, 3.0, ninf, ninf],
        ];
        let targets = [1u32, 2, 0, 0];
        for chunk in [5, 2, 1] {
            let g = check_logit_rows(
                rt,
                "inf",
                &z,
                &targets,
                chunk,
                Reduction::Sum,
                1.0,
                DType::F32,
                GemmOperands::ExactF32,
            );
            for r in [0usize, 2] {
                assert!(g[r].iter().all(|&x| x == 0.0), "chunk {chunk} row {r}: {:?}", g[r]);
            }
            let (out, _) = run_logit_rows(
                rt,
                &z,
                &targets,
                chunk,
                Reduction::Sum,
                1.0,
                DType::F32,
                GemmOperands::ExactF32,
            )
            .unwrap();
            assert!(out.per_row[0] == 0.0 && out.per_row[2] == 0.0, "{:?}", out.per_row);
        }
    });
}

/// A NaN logit, in a non-target or in the target, is an error, never a clean
/// loss (a max that drops NaN must not hide it).
#[test]
fn a_nan_logit_is_an_error() {
    with_gpu(|rt| {
        for (z, t) in [
            (vec![1.0f32, f32::NAN, 2.0, 0.0], 2u32),
            (vec![1.0f32, 3.0, f32::NAN, 0.0], 2u32),
            (vec![f32::NAN, 3.0, 1.0, 0.0], 0u32),
        ] {
            for chunk in [4, 2] {
                let r = run_logit_rows(
                    rt,
                    &[z.clone()],
                    &[t],
                    chunk,
                    Reduction::Sum,
                    1.0,
                    DType::F32,
                    GemmOperands::ExactF32,
                );
                let e = r
                    .err()
                    .unwrap_or_else(|| panic!("{z:?} target {t} chunk {chunk}: a NaN logit gave a number"));
                assert!(e.contains("not finite"), "{e}");
            }
        }
    });
}

/// Timing of the two cross-entropy kernels at Qwen3.5's vocabulary, for the
/// before/after comparison of a kernel change. Run on purpose
/// (`--ignored --nocapture`), under the GPU lock. The hidden width is 16, so
/// the GEMMs are a sliver of the step and `ce_lse_update` and `ce_softmax_grad`
/// (one pass each over `[16, 248 320]` logits) dominate. Prints
/// `METRIC ce_fwd_ms` (forward only) and `METRIC ce_fwd_bwd_ms` (with
/// gradients), each the minimum of 15 runs after 3 warmups.
#[test]
#[ignore = "timing run, not a correctness test"]
fn ce_kernel_timing_at_the_real_vocabulary() {
    with_gpu(|rt| {
        let (n, vocab, chunk) = (16usize, 248_320usize, 8192usize);
        let z: Vec<Vec<f32>> = (0..n)
            .map(|r| {
                random_f32(vocab, 0x71 + r as u64)
                    .into_iter()
                    .map(|x| x * 4.0)
                    .collect()
            })
            .collect();
        let mut h = vec![0.0f32; n * LW];
        for r in 0..n {
            h[r * LW + r] = 1.0;
        }
        let mut w = vec![0.0f32; vocab * LW];
        for (r, zr) in z.iter().enumerate() {
            for (j, &l) in zr.iter().enumerate() {
                w[j * LW + r] = l;
            }
        }
        let ht = upload_at(rt, &h, DType::F32, &[n, LW], 0);
        let wt = upload_at(rt, &w, DType::F32, &[vocab, LW], 0);
        let rows: Vec<u32> = (0..n as u32).collect();
        let targets: Vec<u32> = (0..n as u32).map(|r| r * 15_000 + 3).collect();
        let ws = CeWorkspace::new(
            rt,
            n as u32,
            LW as u32,
            chunk as u32,
            DType::F32,
            GemmOperands::ExactF32,
        )
        .unwrap();
        let dh = sentinel_tensor(rt, &[n, LW]);
        let dw = sentinel_tensor(rt, &[vocab, LW]);
        let run = |grads: bool| {
            let g = grads.then_some(CeGrads {
                dh: &dh,
                dw: &dw,
                scale: 1.0,
            });
            let t0 = std::time::Instant::now();
            cross_entropy_rows(
                rt,
                CeHidden { rows: &ht, off: 0 },
                &wt,
                &rows,
                &targets,
                Reduction::Mean,
                GemmOperands::ExactF32,
                &ws,
                g,
            )
            .unwrap();
            t0.elapsed().as_secs_f64() * 1e3
        };
        for (name, grads) in [("ce_fwd_ms", false), ("ce_fwd_bwd_ms", true)] {
            for _ in 0..3 {
                run(grads);
            }
            let best = (0..15).map(|_| run(grads)).fold(f64::INFINITY, f64::min);
            println!("METRIC {name} {best:.4}");
        }
    });
}
