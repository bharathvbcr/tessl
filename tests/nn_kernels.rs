//! Numeric and validation tests for [`tessl::nn`].
//!
//! Every kernel here ran for months inside `gemma-metal` with no test that
//! executed it against an independent reference — correctness was established
//! end-to-end, by whole-model golden parity, which cannot say *which* kernel
//! drifted. These check each one on its own.

mod common;

use common::{random_f32, with_gpu};
use std::sync::Arc;
use tessl::nn;
use tessl::tensor::bf16_bits_to_f32;
use tessl::GpuRuntime;

fn buf(rt: &Arc<GpuRuntime>, data: &[f32]) -> tessl::tensor::GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1) * 4).expect("alloc");
    b.write_f32(data);
    b
}

fn empty(rt: &Arc<GpuRuntime>, elems: usize) -> tessl::tensor::GpuBuffer {
    let b = rt.alloc_buffer(elems.max(1) * 4).expect("alloc");
    b.zero();
    b
}

/// Assert `got ≈ want` with an absolute tolerance scaled to the accumulation.
fn close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}]: got {g} want {w} (tol {tol})"
        );
    }
}

// ---------------------------------------------------------------- RMSNorm ---

fn rms_norm_ref(x: &[f32], weight: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let ss: f32 = row.iter().map(|v| v * v).sum();
        let inv = 1.0 / (ss / dim as f32 + eps).sqrt();
        for d in 0..dim {
            out[r * dim + d] = row[d] * inv * weight[d];
        }
    }
    out
}

#[test]
fn rms_norm_f32_matches_cpu_reference() {
    with_gpu(|rt| {
        let (rows, dim, eps) = (7usize, 64usize, 1e-6f32);
        let mut x = random_f32(rows * dim, 0x21);
        // Row 3 is all zeros. `eps` exists precisely so that `rsqrt(0 + eps)`
        // is finite; without it this row is `rsqrt(0)` = inf and the whole row
        // becomes inf or NaN. A test built only from well-scaled random rows
        // cannot tell the two apart — the eps term shifts a mean-square of
        // ~0.33 by 3e-6 relative, which hides under any sane tolerance.
        for d in 0..dim {
            x[3 * dim + d] = 0.0;
        }
        let w = random_f32(dim, 0x22);
        let xb = buf(rt, &x);
        let wb = buf(rt, &w);
        let ob = empty(rt, rows * dim);

        nn::rms_norm_f32(rt, &xb, &wb, &ob, rows as u32, dim as u32, eps).unwrap();
        rt.synchronize().unwrap();

        // The kernel sums `dim` squares sequentially in f32; the reference does
        // the same, so only the rsqrt and the final products differ in rounding.
        let got = ob.read_f32();
        assert!(
            got[..rows * dim].iter().all(|v| v.is_finite()),
            "rms_norm_f32 produced a non-finite value; the zero row needs eps"
        );
        close(
            "rms_norm_f32",
            &got[..rows * dim],
            &rms_norm_ref(&x, &w, rows, dim, eps),
            1e-5,
        );
    });
}

#[test]
fn rms_norm_bf16_matches_the_f32_kernel_within_bf16_resolution() {
    with_gpu(|rt| {
        let (rows, dim, eps) = (5usize, 32usize, 1e-6f32);
        let mut x = random_f32(rows * dim, 0x31);
        // Zero row, for the same reason as in the f32 test above.
        for d in 0..dim {
            x[dim + d] = 0.0;
        }
        let w = random_f32(dim, 0x32);
        let xb = buf(rt, &x);
        let wb = buf(rt, &w);
        let ob = rt.alloc_buffer(rows * dim * 2).unwrap();
        ob.zero();

        nn::rms_norm_bf16(rt, &xb, &wb, &ob, rows as u32, dim as u32, eps).unwrap();
        rt.synchronize().unwrap();

        let got: Vec<f32> = ob.read_u32()[..rows * dim / 2]
            .iter()
            .flat_map(|packed| {
                [
                    bf16_bits_to_f32((*packed & 0xffff) as u16),
                    bf16_bits_to_f32((*packed >> 16) as u16),
                ]
            })
            .collect();
        let want = rms_norm_ref(&x, &w, rows, dim, eps);
        // bf16 keeps 8 mantissa bits: relative resolution is ~2^-8.
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let tol = 1e-2 * w.abs().max(1e-3);
            assert!((g - w).abs() <= tol, "rms_norm_bf16[{i}]: {g} vs {w}");
        }
    });
}

#[test]
fn rms_norm_residual_add_folds_the_layer_scale() {
    with_gpu(|rt| {
        let (rows, dim, eps, scale) = (4usize, 16usize, 1e-6f32, 0.75f32);
        let mut x = random_f32(rows * dim, 0x41);
        for d in 0..dim {
            x[2 * dim + d] = 0.0;
        }
        let w = random_f32(dim, 0x42);
        let resid = random_f32(rows * dim, 0x43);
        let xb = buf(rt, &x);
        let wb = buf(rt, &w);
        let rb = buf(rt, &resid);

        nn::rms_norm_residual_add_f32(rt, &xb, &wb, &rb, rows as u32, dim as u32, eps, scale)
            .unwrap();
        rt.synchronize().unwrap();

        let norm = rms_norm_ref(&x, &w, rows, dim, eps);
        let want: Vec<f32> = resid
            .iter()
            .zip(&norm)
            .map(|(r, n)| scale * (r + n))
            .collect();
        let got = rb.read_f32();
        assert!(
            got[..rows * dim].iter().all(|v| v.is_finite()),
            "rms_norm_residual_add_f32 produced a non-finite value; the zero row needs eps"
        );
        close("rms_norm_residual_add", &got[..rows * dim], &want, 1e-5);
    });
}

/// `dim` beyond one threadgroup's width, which is where the strided loop is the
/// only thing summing the tail.
///
/// The three tests above all use `dim` of 16 to 64. `reduce_tptg` hands the
/// kernel 1024 threads for a row that wide, so every lane's `for (d = lid; d <
/// dim; d += tptg)` executes exactly once and a kernel that dropped the loop
/// entirely would still pass them. Verified: replacing the loop with a single
/// `xin[lid]` left all three green. These shapes take 4 strided iterations at
/// 4096 and a ragged 3 at 3000, so the tail cannot go unsummed unnoticed.
#[test]
fn rms_norm_sums_rows_wider_than_one_threadgroup() {
    with_gpu(|rt| {
        for &(rows, dim) in &[(2usize, 4096usize), (3, 3000), (1, 8192)] {
            let eps = 1e-6f32;
            let mut x = random_f32(rows * dim, 0x9001 + dim as u64);
            // A zero row here too: eps has to survive the reduction rewrite.
            x[..dim].fill(0.0);
            let w = random_f32(dim, 0x9002);
            let xb = buf(rt, &x);
            let wb = buf(rt, &w);
            let ob = empty(rt, rows * dim);

            nn::rms_norm_f32(rt, &xb, &wb, &ob, rows as u32, dim as u32, eps).unwrap();
            rt.synchronize().unwrap();

            let got = ob.read_f32();
            assert!(
                got[..rows * dim].iter().all(|v| v.is_finite()),
                "{rows}x{dim}: non-finite output"
            );
            // The kernel now reduces as a tree, so it does NOT match a
            // sequential f32 sum bit for bit. The reference is f64 and the
            // tolerance is relative, which measures the error rather than
            // agreeing with the kernel's own ordering.
            let want = rms_norm_ref_f64(&x, &w, rows, dim, eps);
            for (i, (g, wv)) in got[..rows * dim].iter().zip(&want).enumerate() {
                let tol = 1e-5 * wv.abs().max(1e-3);
                assert!(
                    (g - wv).abs() <= tol,
                    "rms_norm {rows}x{dim} [{i}]: got {g} want {wv}"
                );
            }
        }
    });
}

/// Same width sweep for the two sibling kernels, because the strided loop and
/// the tree reduction are shared code and a fix applied to one of three is not
/// a fix.
#[test]
fn rms_norm_siblings_handle_rows_wider_than_one_threadgroup() {
    with_gpu(|rt| {
        let (rows, dim, eps) = (2usize, 4096usize, 1e-6f32);
        let x = random_f32(rows * dim, 0x9101);
        let w = random_f32(dim, 0x9102);
        let want = rms_norm_ref_f64(&x, &w, rows, dim, eps);
        let xb = buf(rt, &x);
        let wb = buf(rt, &w);

        // bf16: same reduction, narrower store.
        let ob = rt.alloc_buffer(rows * dim * 2).unwrap();
        ob.zero();
        nn::rms_norm_bf16(rt, &xb, &wb, &ob, rows as u32, dim as u32, eps).unwrap();
        rt.synchronize().unwrap();
        let got: Vec<f32> = ob.read_u32()[..rows * dim / 2]
            .iter()
            .flat_map(|p| {
                [
                    bf16_bits_to_f32((*p & 0xffff) as u16),
                    bf16_bits_to_f32((*p >> 16) as u16),
                ]
            })
            .collect();
        for (i, (g, wv)) in got.iter().zip(&want).enumerate() {
            let tol = 1e-2 * wv.abs().max(1e-3);
            assert!(
                (g - wv).abs() <= tol,
                "rms_norm_bf16 wide [{i}]: {g} vs {wv}"
            );
        }

        // residual_add with layer_scale = 1: resid += norm.
        let resid = vec![0.0f32; rows * dim];
        let rb = buf(rt, &resid);
        nn::rms_norm_residual_add_f32(rt, &xb, &wb, &rb, rows as u32, dim as u32, eps, 1.0)
            .unwrap();
        rt.synchronize().unwrap();
        let got = rb.read_f32();
        for (i, (g, wv)) in got[..rows * dim].iter().zip(&want).enumerate() {
            let tol = 1e-5 * wv.abs().max(1e-3);
            assert!(
                (g - wv).abs() <= tol,
                "rms_norm_residual_add wide [{i}]: {g} vs {wv}"
            );
        }
    });
}

/// f64 reference. Distinct from `rms_norm_ref` above, which accumulates in f32
/// in the same order the old serial kernel did — a comparison that agreed with
/// the kernel's rounding instead of measuring it.
fn rms_norm_ref_f64(x: &[f32], weight: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let ss: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let inv = 1.0 / (ss / dim as f64 + eps as f64).sqrt();
        for d in 0..dim {
            out[r * dim + d] = (row[d] as f64 * inv * weight[d] as f64) as f32;
        }
    }
    out
}

// ------------------------------------------------------------ MLP gating ---

#[test]
fn mlp_silu_matches_cpu_reference() {
    with_gpu(|rt| {
        let n = 512usize;
        let gate = random_f32(n, 0x51);
        let up = random_f32(n, 0x52);
        let gb = buf(rt, &gate);
        let ub = buf(rt, &up);
        let ob = empty(rt, n);

        nn::mlp_silu(rt, &gb, &ub, &ob, n as u32).unwrap();
        rt.synchronize().unwrap();

        let want: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        close("mlp_silu", &ob.read_f32()[..n], &want, 1e-6);
    });
}

#[test]
fn mlp_gelu_tanh_stays_finite_where_fast_tanh_would_nan() {
    with_gpu(|rt| {
        // The kernel's own header records the bug this guards: at -O2, MSL
        // lowers `tanh` to `air.fast_tanh`, which NaNs past |arg| ~ 10, and the
        // GELU inner term reaches ~301 at |x| = 20. Feed it exactly that range.
        let gate: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.5).collect();
        let up = vec![1.0f32; gate.len()];
        let n = gate.len();
        let gb = buf(rt, &gate);
        let ub = buf(rt, &up);
        let ob = empty(rt, n);

        nn::mlp_gelu_tanh(rt, &gb, &ub, &ob, n as u32).unwrap();
        rt.synchronize().unwrap();

        let got = ob.read_f32();
        assert!(
            got[..n].iter().all(|v| v.is_finite()),
            "mlp_gelu_tanh produced a non-finite value on |x| up to 64"
        );

        // Match the kernel's clamp so this is a comparison, not a restatement.
        let want: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(x, u)| {
                // f64 reference: the kernel works in f32, so computing the
                // expected value at the same precision would hide a real f32
                // ordering bug behind matching rounding.
                let xc = (*x as f64).clamp(-20.0, 20.0);
                let inner = 0.7978845608028654 * (xc + 0.044715 * xc * xc * xc);
                (0.5 * xc * (1.0 + inner.clamp(-10.0, 10.0).tanh()) * (*u as f64)) as f32
            })
            .collect();
        close("mlp_gelu_tanh", &got[..n], &want, 1e-4);
    });
}

// ------------------------------------------------------------ Elementwise ---

#[test]
fn scale_f32_inplace_scales_exactly_and_leaves_the_tail_alone() {
    with_gpu(|rt| {
        let data = random_f32(128, 0x61);
        // Allocate more than we scale: the kernel must not touch the tail.
        let b = rt.alloc_buffer(256 * 4).unwrap();
        let mut padded = data.clone();
        padded.extend(std::iter::repeat_n(7.0f32, 128));
        b.write_f32(&padded);

        nn::scale_f32_inplace(rt, &b, 0.25, 128).unwrap();
        rt.synchronize().unwrap();

        let got = b.read_f32();
        // Multiplication by a power of two is exact in f32 — no tolerance.
        for (i, v) in data.iter().enumerate() {
            assert_eq!(got[i], v * 0.25, "scaled[{i}]");
        }
        for (i, v) in got[128..256].iter().enumerate() {
            assert_eq!(*v, 7.0, "tail[{i}] was modified");
        }
    });
}

// ------------------------------------------------------------------ GEMV ---

#[test]
fn gemv_q8_matches_cpu_dequant_reference() {
    with_gpu(|rt| {
        let (rows, cols, group) = (24usize, 64usize, 16usize);
        let groups = rows * (cols / group);
        let packed: Vec<i8> = (0..rows * cols)
            .map(|i| (i as i32 % 251 - 125) as i8)
            .collect();
        let scales: Vec<f32> = (0..groups).map(|i| 0.01 + (i % 7) as f32 * 0.003).collect();
        let zeros: Vec<f32> = (0..groups).map(|i| (i % 5) as f32 - 2.0).collect();
        let x = random_f32(cols, 0x71);

        let pb = rt.alloc_buffer(packed.len()).unwrap();
        pb.write_bytes(&packed.iter().map(|v| *v as u8).collect::<Vec<u8>>());
        let sb = buf(rt, &scales);
        let zb = buf(rt, &zeros);
        let xb = buf(rt, &x);
        let yb = empty(rt, rows);

        nn::gemv_q8(
            rt,
            &pb,
            &sb,
            &zb,
            &xb,
            &yb,
            rows as u32,
            cols as u32,
            group as u32,
        )
        .unwrap();
        rt.synchronize().unwrap();

        let mut want = vec![0.0f32; rows];
        for r in 0..rows {
            let mut acc = 0.0f32;
            for g in 0..cols / group {
                let gi = r * (cols / group) + g;
                for i in 0..group {
                    let w = scales[gi] * (packed[r * cols + g * group + i] as f32 - zeros[gi]);
                    acc += w * x[g * group + i];
                }
            }
            want[r] = acc;
        }
        close("gemv_q8", &yb.read_f32()[..rows], &want, 1e-3);
    });
}

/// CPU reference for Q8 GEMV, in f64 so it measures the kernel's error rather
/// than sharing its rounding.
fn gemv_q8_ref(
    packed: &[i8],
    scales: &[f32],
    zeros: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    group: usize,
) -> Vec<f32> {
    let gpr = cols / group;
    (0..rows)
        .map(|r| {
            let mut acc = 0.0f64;
            for g in 0..gpr {
                let gi = r * gpr + g;
                for i in 0..group {
                    let w = scales[gi] as f64
                        * (packed[r * cols + g * group + i] as f64 - zeros[gi] as f64);
                    acc += w * x[g * group + i] as f64;
                }
            }
            acc as f32
        })
        .collect()
}

/// Shapes that reach the two paths the original test could not.
///
/// `gemv_q8` is one simdgroup per four rows, and it takes a `char4` fast path
/// when `group_size % 4 == 0`. The only pre-existing test used rows = 24 and
/// group = 16 — a multiple of the 8 rows a threadgroup covers, and a group
/// width divisible by 4 — so neither the row tail nor the scalar fallback ran.
/// Verified: forcing `xv = 0` in the fallback, and removing the `row < rows`
/// writeback guard, both left that test green.
#[test]
fn gemv_q8_covers_the_row_tail_and_the_scalar_fallback() {
    with_gpu(|rt| {
        // group 15 is not divisible by 4 -> scalar path; rows 13 and 37 are not
        // multiples of 8 -> partially filled final threadgroup.
        for &(rows, cols, group) in &[
            (13usize, 60usize, 15usize),
            (37, 128, 32),
            (13, 120, 15),
            (100, 4096, 64),
        ] {
            let packed: Vec<i8> = (0..rows * cols)
                .map(|i| (i as i32 % 251 - 125) as i8)
                .collect();
            let groups = rows * (cols / group);
            let scales: Vec<f32> = (0..groups).map(|i| 0.01 + (i % 7) as f32 * 0.003).collect();
            let zeros: Vec<f32> = (0..groups).map(|i| (i % 5) as f32 - 2.0).collect();
            let x = random_f32(cols, 0xB100 + cols as u64);

            let pb = rt.alloc_buffer(packed.len()).unwrap();
            pb.write_bytes(&packed.iter().map(|v| *v as u8).collect::<Vec<u8>>());
            let sb = buf(rt, &scales);
            let zb = buf(rt, &zeros);
            let xb = buf(rt, &x);

            // y is allocated past `rows` and seeded with a sentinel: a kernel
            // whose tail threadgroup writes rows it does not own would land
            // here, and a bounds bug that only ever wrote plausible numbers
            // inside the live range would otherwise be invisible.
            const SENTINEL: f32 = -12345.0;
            let yb = buf(rt, &vec![SENTINEL; rows + 16]);

            nn::gemv_q8(
                rt,
                &pb,
                &sb,
                &zb,
                &xb,
                &yb,
                rows as u32,
                cols as u32,
                group as u32,
            )
            .unwrap();
            rt.synchronize().unwrap();

            let got = yb.read_f32();
            let want = gemv_q8_ref(&packed, &scales, &zeros, &x, rows, cols, group);
            for (i, w) in want.iter().enumerate() {
                let tol = 1e-3 * w.abs().max(1.0);
                assert!(
                    (got[i] - w).abs() <= tol,
                    "gemv_q8 {rows}x{cols} g{group} [{i}]: got {} want {w}",
                    got[i]
                );
            }
            for (i, v) in got[rows..rows + 16].iter().enumerate() {
                assert_eq!(
                    *v, SENTINEL,
                    "gemv_q8 {rows}x{cols} g{group} wrote past row {rows} at +{i}"
                );
            }
        }
    });
}

// -------------------------------------------------------------- KV cache ---

#[test]
fn kv_store_timestep_writes_at_the_device_side_offset() {
    with_gpu(|rt| {
        let n = 32usize;
        let src = random_f32(n, 0x81);
        let sb = buf(rt, &src);
        let dst = rt.alloc_buffer(4 * n * 4).unwrap();
        dst.zero();
        let off = rt.alloc_buffer(4).unwrap();
        off.write_u32(&[(2 * n) as u32]);

        nn::kv_store_timestep(rt, &sb, &dst, &off, n as u32).unwrap();
        rt.synchronize().unwrap();

        let got = dst.read_f32();
        // Slot 2 holds the data...
        close("kv_store slot2", &got[2 * n..3 * n], &src, 0.0);
        // ...and nothing else was written.
        assert!(
            got[..2 * n].iter().chain(&got[3 * n..]).all(|v| *v == 0.0),
            "kv_store_timestep wrote outside its slot"
        );
    });
}

// ------------------------------------------------------------- Validation ---

#[test]
fn undersized_buffers_are_refused_before_any_dispatch() {
    with_gpu(|rt| {
        let (rows, dim) = (8u32, 64u32);
        let full = empty(rt, (rows * dim) as usize);
        let short = empty(rt, (rows * dim) as usize - 1);
        let w = empty(rt, dim as usize);

        // Each operand checked independently: a single guard covering only the
        // first would pass this test if the others were unchecked.
        for (name, x, weight, out) in [
            ("x", &short, &w, &full),
            ("weight", &full, &empty(rt, dim as usize - 1), &full),
            ("out", &full, &w, &short),
        ] {
            let err = nn::rms_norm_f32(rt, x, weight, out, rows, dim, 1e-6)
                .expect_err("undersized {name} must be refused");
            assert!(
                err.contains("buffer holds"),
                "{name}: unexpected error {err:?}"
            );
        }
        assert_eq!(
            rt.take_dispatch_count(),
            0,
            "a rejected call must not have encoded anything"
        );
    });
}

#[test]
fn gemv_q8_refuses_a_ragged_final_group() {
    with_gpu(|rt| {
        // cols = 65 with group_size 16 leaves a group of 1 that the kernel's
        // `cols / group_size` would silently drop, returning a plausible but
        // wrong answer. That is worse than an error.
        let b = empty(rt, 65 * 24);
        let err = nn::gemv_q8(rt, &b, &b, &b, &b, &b, 24, 65, 16).expect_err("ragged tail");
        assert!(err.contains("not a multiple"), "unexpected error: {err:?}");

        let err = nn::gemv_q8(rt, &b, &b, &b, &b, &b, 24, 64, 0).expect_err("zero group");
        assert!(err.contains("non-zero"), "unexpected error: {err:?}");
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}

#[test]
fn kv_ring_densify_refuses_zero_capacity() {
    with_gpu(|rt| {
        let b = empty(rt, 64);
        let u = rt.alloc_buffer(4).unwrap();
        let err = nn::kv_ring_densify(rt, &b, &b, &u, &u, 8, 0).expect_err("zero capacity");
        assert!(err.contains("non-zero"), "unexpected error: {err:?}");
    });
}
