//! Shared scaffolding for tessl's integration tests.
//!
//! Everything here goes through the public API only — the same surface
//! `gemma-metal` and `tessl-arch02` see across a crate boundary. Nothing in
//! this module may reach into `pub(crate)` internals; if a check cannot be
//! written from outside, it belongs in a `src/` unit test instead.

#![allow(dead_code)]

use std::sync::{Arc, Mutex, MutexGuard};

use tessl::tensor::{bf16_bits_to_f32, f32_slice_to_bf16, GpuBuffer};
use tessl::{GpuRuntime, Tensor};

/// One GPU at a time, whatever `--test-threads` says.
///
/// `GpuRuntime` guards its *own* encoder with an access flag, but two runtimes
/// alive on two threads still contend for the same device and the same default
/// command queue. The crate README tells humans to pass `--test-threads=1`; a
/// test suite that only passes when the caller remembers a flag is a test suite
/// that fails in CI, so the serialization is enforced here instead.
static GPU: Mutex<()> = Mutex::new(());

fn gpu_lock() -> MutexGuard<'static, ()> {
    // A panicking test poisons the mutex. Recovering keeps one real failure
    // from cascading into a screen of misleading "poisoned" errors that hide it.
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Run `f` against a freshly constructed runtime, exclusively.
///
/// Fresh per test on purpose: `set_precision` / `set_relaxed_precision` /
/// `set_async_encode` are runtime-wide mutable state, so a shared runtime would
/// leak one test's mode into the next and make failures order-dependent.
pub fn with_gpu<R>(f: impl FnOnce(&Arc<GpuRuntime>) -> R) -> R {
    let _guard = gpu_lock();
    let rt = GpuRuntime::new().expect("GpuRuntime::new (needs a Metal 4 device + built metallib)");
    f(&rt)
}

/// Like [`with_gpu`], but hands out two independent runtimes.
///
/// Cross-runtime rejection is a real hazard for consumers that hold an
/// inference runtime and a training runtime at once, and it cannot be provoked
/// with a single runtime.
pub fn with_two_gpus<R>(f: impl FnOnce(&Arc<GpuRuntime>, &Arc<GpuRuntime>) -> R) -> R {
    let _guard = gpu_lock();
    let a = GpuRuntime::new().expect("GpuRuntime::new (first)");
    let b = GpuRuntime::new().expect("GpuRuntime::new (second)");
    f(&a, &b)
}

/// splitmix64 — a deterministic stream so a failure reproduces exactly.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [-1, 1). Bounded operands keep the derived error bound
    /// meaningful; unbounded ones would let one outlier product dominate it.
    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / 8_388_608.0 - 1.0
    }
}

pub fn random_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.unit()).collect()
}

/// Round through bf16 and back, so the CPU reference sees exactly the operands
/// the GPU will read. Without this the test would be measuring host-side
/// quantization error, not the kernel.
pub fn round_trip_bf16(data: &[f32]) -> Vec<f32> {
    f32_slice_to_bf16(data)
        .into_iter()
        .map(bf16_bits_to_f32)
        .collect()
}

pub fn tensor_f32(rt: &Arc<GpuRuntime>, shape: &[usize], data: &[f32]) -> Tensor {
    let t = rt.alloc_tensor_f32(shape).expect("alloc_tensor_f32");
    t.buffer.write_f32(data);
    t
}

pub fn tensor_bf16(rt: &Arc<GpuRuntime>, shape: &[usize], data: &[f32]) -> Tensor {
    let t = rt.alloc_tensor_bf16(shape).expect("alloc_tensor_bf16");
    t.buffer.write_bf16_bits(&f32_slice_to_bf16(data));
    t
}

/// Operand storage order, i.e. which GEMM entry point produced the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// A[M,K] @ B[K,N]
    Nn,
    /// A[K,M]^T @ B[K,N]
    Tn,
    /// A[M,K] @ B[N,K]^T
    Nt,
}

/// A CPU reference product plus the magnitude term its error bound scales with.
pub struct Reference {
    pub c: Vec<f64>,
    /// Per output element, `sum_k |a_ik * b_kj|`.
    ///
    /// The f32 accumulation bound is proportional to this, not to `|c|`: a row
    /// whose terms cancel has a small result and a large error budget, and
    /// judging it by `|c|` alone would demand accuracy the format cannot give.
    pub mag: Vec<f64>,
}

/// Reference GEMM in f64, indexing the operands by `layout`.
///
/// f64 rather than f32 so the reference is not itself a source of the error the
/// bound is meant to attribute to the GPU: at K <= 4096 its own accumulation
/// error is ~1e-13 relative, four orders below the f32 budget below.
pub fn reference(layout: Layout, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Reference {
    let a_at = |i: usize, p: usize| -> f64 {
        match layout {
            Layout::Nn | Layout::Nt => a[i * k + p] as f64,
            Layout::Tn => a[p * m + i] as f64,
        }
    };
    let b_at = |p: usize, j: usize| -> f64 {
        match layout {
            Layout::Nn | Layout::Tn => b[p * n + j] as f64,
            Layout::Nt => b[j * k + p] as f64,
        }
    };
    let mut c = vec![0.0f64; m * n];
    let mut mag = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let (mut acc, mut abs) = (0.0f64, 0.0f64);
            for p in 0..k {
                let term = a_at(i, p) * b_at(p, j);
                acc += term;
                abs += term.abs();
            }
            c[i * n + j] = acc;
            mag[i * n + j] = abs;
        }
    }
    Reference { c, mag }
}

/// f32 unit roundoff, 2^-24.
pub const U_F32: f64 = 5.960_464_477_539_063e-8;
/// bf16 unit roundoff, 2^-8. Also a safe upper bound for tf32-class formats
/// (2^-11), which is why the relaxed-precision path can share it.
pub const U_BF16: f64 = 3.906_25e-3;

/// Standard `gamma_n = n*u / (1 - n*u)` from the f32 dot-product error analysis.
fn gamma(n: usize, u: f64) -> f64 {
    let nu = n as f64 * u;
    assert!(nu < 1.0, "error bound degenerate at K={n}; shrink K");
    nu / (1.0 - nu)
}

/// Per-element tolerance for a K-term dot product accumulated in f32.
///
/// Derivation. Every kernel under test reduces K products into an f32
/// accumulator, so the classical bound applies elementwise:
///
///   |fl(sum) - sum| <= gamma_K * sum_k |a_k b_k|,   gamma_n = n*u / (1 - n*u)
///
/// with u = 2^-24 for f32. That single gamma already absorbs the K product
/// roundings and the K-1 additions, and an FMA-based kernel only does better.
/// `n = K + 8` rather than K: the split-K lanes reduce partial sums in a second
/// pass and every path rounds once more when storing C, and eight spare terms
/// covers both with room left over — it costs ~0.4% of the budget at K=2048 and
/// removes the need to know which lane the dispatcher picked.
///
/// `operand_u` is nonzero only when the *inputs* are narrower than f32. For the
/// bf16 kernels it is zero, because the test rounds its operands to bf16 before
/// upload: the GPU then reads exactly the reference's values, and a bf16 x bf16
/// product is exact in f32 (8 + 8 significand bits fit in 24), so operand width
/// contributes nothing. It is nonzero for the relaxed-precision f32 path, where
/// the kernel — not the test — narrows the operands.
pub fn tolerance(k: usize, mag: f64, operand_u: f64) -> f64 {
    // Narrowing both operands perturbs each product by at most 2*u_in relative
    // (first order); that error is then carried through the same summation.
    (gamma(k + 8, U_F32) + 2.0 * operand_u) * mag
}

/// Assert every element of `got` is within the derived bound of the reference.
///
/// Reports the worst offender scaled by its own budget, so the failure message
/// says how far out of tolerance the kernel is rather than just that it is.
pub fn assert_within_bound(label: &str, got: &[f32], r: &Reference, k: usize, operand_u: f64) {
    assert_eq!(got.len(), r.c.len(), "{label}: result length mismatch");
    let mut worst = 0.0f64;
    let mut worst_at = 0usize;
    for (idx, (&g, (&want, &mag))) in got.iter().zip(r.c.iter().zip(r.mag.iter())).enumerate() {
        assert!(g.is_finite(), "{label}: non-finite output at [{idx}] = {g}");
        let tol = tolerance(k, mag, operand_u);
        let err = (g as f64 - want).abs();
        // An all-zero magnitude means every product was exactly zero, so the
        // only acceptable output is exactly zero and `tol` is correctly 0.
        let scaled = if tol > 0.0 {
            err / tol
        } else if err == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        if scaled > worst {
            worst = scaled;
            worst_at = idx;
        }
    }
    assert!(
        worst <= 1.0,
        "{label}: element [{worst_at}] is {worst:.3}x its error budget \
         (got {}, want {}, |a.b| sum {}, tol {:.3e})",
        got[worst_at],
        r.c[worst_at],
        r.mag[worst_at],
        tolerance(k, r.mag[worst_at], operand_u),
    );
}

// --------------------------------------------------- Buffers and Q4 banks ---

/// f32 buffer holding `data`.
pub fn buf(rt: &Arc<GpuRuntime>, data: &[f32]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1) * 4).expect("alloc");
    b.write_f32(data);
    b
}

/// Zeroed f32 buffer of `elems` elements.
pub fn empty(rt: &Arc<GpuRuntime>, elems: usize) -> GpuBuffer {
    let b = rt.alloc_buffer(elems.max(1) * 4).expect("alloc");
    b.zero();
    b
}

/// bf16 buffer holding `data` rounded to bf16.
pub fn buf_bf16(rt: &Arc<GpuRuntime>, data: &[f32]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1) * 2).expect("alloc");
    b.write_bf16_bits(&f32_slice_to_bf16(data));
    b
}

/// Buffer seeded with `value`, for detecting elements a kernel never wrote.
pub fn seeded(rt: &Arc<GpuRuntime>, elems: usize, value: f32) -> GpuBuffer {
    buf(rt, &vec![value; elems.max(1)])
}

/// Two nibbles per byte, low nibble first — the MLX Q4 packing.
pub fn pack_nibbles(nibbles: &[u8]) -> Vec<u8> {
    nibbles
        .chunks(2)
        .map(|p| (p[0] & 0x0f) | ((p.get(1).copied().unwrap_or(0) & 0x0f) << 4))
        .collect()
}

/// An MLX Q4 weight bank plus the dense matrix it dequantizes to.
///
/// Scales and biases are rounded through bf16 first, so the reference sees the
/// values the kernel actually loads rather than their f32 originals — otherwise
/// the test would be measuring host-side quantization, not the kernel.
pub fn q4_mlx_matrix(rows: usize, cols: usize, group: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let nibbles: Vec<u8> = (0..rows * cols).map(|i| ((i * 5) % 16) as u8).collect();
    let groups = rows * (cols / group);
    let sb_f32: Vec<f32> = (0..groups * 2)
        .map(|i| {
            if i % 2 == 0 {
                0.03 + (i % 7) as f32 * 0.002
            } else {
                (i % 5) as f32 * 0.1 - 0.2
            }
        })
        .collect();
    let sb_round = round_trip_bf16(&sb_f32);
    let mut dense = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let gi = r * (cols / group) + c / group;
            dense[r * cols + c] =
                sb_round[gi * 2] * nibbles[r * cols + c] as f32 + sb_round[gi * 2 + 1];
        }
    }
    (pack_nibbles(&nibbles), sb_f32, dense)
}

/// `y[r] = sum_c dense[r, c] * x[c]`, in f64.
pub fn dense_gemv(dense: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| dense[r * cols + c] as f64 * x[c] as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

/// Assert `got ~= want` elementwise with a relative-plus-absolute tolerance.
pub fn close_rel(what: &str, got: &[f32], want: &[f32], rel: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what}[{i}]: non-finite {g}");
        let tol = rel * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "{what}[{i}]: got {g} want {w}");
    }
}
