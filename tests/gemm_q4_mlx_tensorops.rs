//! `nn::gemm_q4_mlx_tensorops` (kernels/gemm_q4_mlx_tensorops.metal): the
//! production per-group MLX Q4 TensorOps GEMM that the probe in
//! `tests/int4b_tensorops.rs` proved out. GPU tests (a Metal 4 TensorOps device);
//! every test goes through `with_gpu`, so none is a hosted-CI test.
//!
//! Parity is against `nn::gemm_q4_mlx` on the same bank bytes and the same bf16
//! activation buffer, and against an f64 reference of the dequantized product.
//! Both paths load bit-identical operands (the activations go through
//! `buf_bf16` for both), so the derived bound has no operand-rounding term.
//!
//! # Error bound
//!
//! Every output is formed from at most `t` f32 additions of magnitude <= S, in
//! an unspecified order (matmul2d internals, simd_sum tree, fma contraction), so
//! `|fl(sum) - sum| <= gamma_t * S`, `gamma_t = t u / (1 - t u)`, `u = 2^-24`,
//! `S = sum_g |s| sum|q x| + |b| sum|x|` (the magnitude of every term the
//! epilogue adds). Products `x q` (bf16 x 4-bit) and the bf16 `s`, `b` are exact
//! in f32. For the TensorOps kernel `t = K + 2 K/G` (the accumulator, the
//! `s P` and `b xsum` additions per group) `+ G` for the in-kernel `xsum`, whose
//! own error is at most `gamma_G sum|x|` per group and is scaled by `|b|`. The
//! simd path has `t = K + 2 K/G`, as in `tests/int4b_tensorops.rs`.

mod common;

use common::{buf_bf16, pack_nibbles, round_trip_bf16, seeded, with_gpu, SplitMix};
use std::sync::Arc;
use tessl::nn::{self, Q4MlxBank, Q4MlxLayout, QuantShape};
use tessl::GpuRuntime;

const SENTINEL: f32 = -7.7e28;

/// f64 product of the loaded operands and the per-output magnitude `S`.
fn reference(
    x: &[f32],
    nibbles: &[u8],
    sb: &[f32],
    (m, n, k, group): (usize, usize, usize, usize),
) -> (Vec<f64>, Vec<f64>) {
    let groups = k / group;
    let mut want = vec![0.0f64; m * n];
    let mut mag = vec![0.0f64; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let (mut acc, mut s_mag) = (0.0f64, 0.0f64);
            for g in 0..groups {
                let (s, b) = (sb[(ni * groups + g) * 2] as f64, sb[(ni * groups + g) * 2 + 1] as f64);
                let (mut qx, mut aqx, mut xsum, mut ax) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for j in 0..group {
                    let xv = x[mi * k + g * group + j] as f64;
                    let q = nibbles[ni * k + g * group + j] as f64;
                    qx += q * xv;
                    aqx += q * xv.abs();
                    xsum += xv;
                    ax += xv.abs();
                }
                acc += s * qx + b * xsum;
                s_mag += s.abs() * aqx + b.abs() * ax;
            }
            want[mi * n + ni] = acc;
            mag[mi * n + ni] = s_mag;
        }
    }
    (want, mag)
}

fn gamma(terms: usize) -> f64 {
    let tu = terms as f64 * 2f64.powi(-24);
    tu / (1.0 - tu)
}

struct Fractions {
    to_ref: f64,
    simd_ref: f64,
    to_simd: f64,
    max_abs_to_ref: f64,
    max_abs_to_simd: f64,
}

/// One shape: returns the largest `err / bound` for TensorOps vs f64, simd vs
/// f64 and TensorOps vs simd.
fn run_case(rt: &Arc<GpuRuntime>, m: usize, n: usize, k: usize, group: usize, random: bool) -> Fractions {
    let groups = k / group;
    let mut rng = SplitMix::new(0x71e5 ^ (m * 1000003 + n * 4099 + k * 17 + group) as u64);
    // Exact: multiples of 1/64 (7 bits, exact in bf16). Random: any f32, rounded
    // to bf16 by `buf_bf16` for both paths and by `round_trip_bf16` for the
    // reference, so every path reads the same operand.
    let x: Vec<f32> = (0..m * k)
        .map(|_| {
            let v = rng.unit();
            if random {
                v
            } else {
                (v * 64.0).round() / 64.0
            }
        })
        .collect();
    let x_bf = round_trip_bf16(&x);
    let nibbles: Vec<u8> = (0..n * k).map(|_| rng.range(0, 15) as u8).collect();
    let sb_f32: Vec<f32> = (0..n * groups * 2)
        .map(|i| {
            let u = (rng.unit() + 1.0) * 0.5;
            if i % 2 == 0 {
                0.01 + 0.04 * u
            } else {
                -0.5 + 0.6 * u
            }
        })
        .collect();
    let sb = round_trip_bf16(&sb_f32);
    let (want, mag) = reference(&x_bf, &nibbles, &sb, (m, n, k, group));

    let packed = pack_nibbles(&nibbles);
    let pb = rt.alloc_buffer(packed.len()).unwrap();
    pb.write_bytes(&packed);
    let sbb = buf_bf16(rt, &sb);
    let bank = Q4MlxBank {
        packed: &pb,
        scales_biases: &sbb,
    };
    let shape = QuantShape {
        rows: n as u32,
        cols: k as u32,
        group_size: group as u32,
    };
    let xb = buf_bf16(rt, &x);

    let yb = seeded(rt, m * n, SENTINEL);
    nn::gemm_q4_mlx_tensorops(rt, bank, &xb, &yb, shape, m as u32).unwrap();
    rt.synchronize().unwrap();
    let to: Vec<f32> = yb.read_f32()[..m * n].to_vec();
    assert!(
        to.iter().all(|v| *v != SENTINEL && v.is_finite()),
        "TensorOps kernel left outputs unwritten or non-finite for M={m} N={n} K={k} G={group}"
    );

    let mut simd = vec![0.0f32; m * n];
    for r0 in (0..m).step_by(nn::GEMM_Q4_MLX_MAX_M as usize) {
        let mc = (m - r0).min(nn::GEMM_Q4_MLX_MAX_M as usize);
        let xc = buf_bf16(rt, &x[r0 * k..(r0 + mc) * k]);
        let yc = seeded(rt, mc * n, SENTINEL);
        nn::gemm_q4_mlx(rt, bank, &xc, &yc, shape, mc as u32, Q4MlxLayout::RowMajor, None).unwrap();
        rt.synchronize().unwrap();
        simd[r0 * n..(r0 + mc) * n].copy_from_slice(&yc.read_f32()[..mc * n]);
    }
    assert!(
        simd.iter().all(|v| *v != SENTINEL && v.is_finite()),
        "simd path unwritten"
    );

    let g_simd = gamma(k + 2 * groups);
    let g_to = gamma(k + 2 * groups + group);
    let mut f = Fractions {
        to_ref: 0.0,
        simd_ref: 0.0,
        to_simd: 0.0,
        max_abs_to_ref: 0.0,
        max_abs_to_simd: 0.0,
    };
    for i in 0..m * n {
        let e_to = (to[i] as f64 - want[i]).abs();
        let e_simd = (simd[i] as f64 - want[i]).abs();
        let e_pair = (to[i] as f64 - simd[i] as f64).abs();
        f.to_ref = f.to_ref.max(e_to / (g_to * mag[i]));
        f.simd_ref = f.simd_ref.max(e_simd / (g_simd * mag[i]));
        f.to_simd = f.to_simd.max(e_pair / ((g_to + g_simd) * mag[i]));
        f.max_abs_to_ref = f.max_abs_to_ref.max(e_to);
        f.max_abs_to_simd = f.max_abs_to_simd.max(e_pair);
    }
    f
}

fn sweep(random: bool) {
    with_gpu(|rt| {
        let mut failures = Vec::new();
        for group in [32usize, 64, 128] {
            // M > 33 and a partial last M tile (70, 33, 1, 8); N a multiple of 32
            // and not (33, 40, 100: partial last N tile, ragged weight rows).
            for m in [1usize, 8, 33, 70] {
                for n in [32usize, 64, 33, 40, 100] {
                    for k in [128usize, 256] {
                        let f = run_case(rt, m, n, k, group, random);
                        eprintln!(
                            "{} M={m} N={n} K={k} G={group}: TO-vs-f64 abs {:.3e} ({:.3} of bound); \
                             simd-vs-f64 ({:.3}); TO-vs-simd abs {:.3e} ({:.3})",
                            if random { "random" } else { "exact " },
                            f.max_abs_to_ref,
                            f.to_ref,
                            f.simd_ref,
                            f.max_abs_to_simd,
                            f.to_simd
                        );
                        if f.to_ref > 1.0 || f.simd_ref > 1.0 || f.to_simd > 1.0 {
                            failures.push(format!(
                                "M={m} N={n} K={k} G={group}: {:.3} / {:.3} / {:.3}",
                                f.to_ref, f.simd_ref, f.to_simd
                            ));
                        }
                    }
                }
            }
        }
        assert!(failures.is_empty(), "bound exceeded: {failures:#?}");
    });
}

#[test]
fn matches_gemm_q4_mlx_and_f64_on_exact_activations() {
    sweep(false);
}

#[test]
fn matches_gemm_q4_mlx_and_f64_on_random_activations() {
    sweep(true);
}

/// The store at a partial tile must be clipped to `M x N`. `y` is allocated
/// with `PAD_WORDS` of NaN-pattern sentinel after the `m * n` outputs (a full
/// 32x32 tile plus 128 words, so any unclipped tile overrun lands inside it);
/// every pad word must be bit-identical afterwards, and every output word
/// written. `seeded`-sized buffers cannot see this: an overrun there lands in
/// whatever neighbours the allocation, and only `[..m * n]` is read back.
#[test]
fn partial_tiles_never_write_past_y() {
    const PAD_WORDS: usize = 32 * 32 + 128;
    const PAD: u32 = 0x7fc0_dead;
    with_gpu(|rt| {
        for group in [32usize, 64, 128] {
            for m in [1usize, 8, 33, 70] {
                for n in [33usize, 40, 100] {
                    let k = 256;
                    let mut rng = SplitMix::new(0x9ad ^ (m * 131 + n * 7 + group) as u64);
                    let x: Vec<f32> = (0..m * k).map(|_| rng.unit()).collect();
                    let nibbles: Vec<u8> = (0..n * k).map(|_| rng.range(0, 15) as u8).collect();
                    let sb: Vec<f32> = (0..n * (k / group) * 2).map(|_| rng.unit() * 0.1).collect();
                    let packed = rt.alloc_buffer(n * k / 2).unwrap();
                    packed.write_bytes(&pack_nibbles(&nibbles));
                    let sbb = buf_bf16(rt, &sb);
                    let xb = buf_bf16(rt, &x);
                    let total = m * n + PAD_WORDS;
                    let y = rt.alloc_buffer(total * 4).unwrap();
                    y.write_u32(&vec![PAD; total]);
                    nn::gemm_q4_mlx_tensorops(
                        rt,
                        Q4MlxBank {
                            packed: &packed,
                            scales_biases: &sbb,
                        },
                        &xb,
                        &y,
                        QuantShape {
                            rows: n as u32,
                            cols: k as u32,
                            group_size: group as u32,
                        },
                        m as u32,
                    )
                    .unwrap();
                    rt.synchronize().unwrap();
                    let words = y.read_u32();
                    let first_bad = words[m * n..total].iter().position(|w| *w != PAD);
                    assert!(
                        first_bad.is_none(),
                        "M={m} N={n} G={group}: pad word {:?} past y was overwritten",
                        first_bad.map(|i| (i, words[m * n + i]))
                    );
                    assert!(
                        words[..m * n].iter().all(|w| *w != PAD),
                        "M={m} N={n} G={group}: an output word was left unwritten"
                    );
                }
            }
        }
    });
}

/// A longer K, so many groups accumulate (and the xsum is over many groups).
#[test]
fn matches_at_a_long_k() {
    with_gpu(|rt| {
        for group in [32usize, 64, 128] {
            let f = run_case(rt, 40, 72, 2048, group, true);
            eprintln!(
                "random M=40 N=72 K=2048 G={group}: {:.3} / {:.3} / {:.3}",
                f.to_ref, f.simd_ref, f.to_simd
            );
            assert!(f.to_ref <= 1.0 && f.simd_ref <= 1.0 && f.to_simd <= 1.0, "G={group}");
        }
    });
}

// ------------------------------------------------------------ the guards ---

/// Buffers of the right size for a tiny shape, plus a call under test.
struct Operands {
    packed: tessl::GpuBuffer,
    sb: tessl::GpuBuffer,
    x: tessl::GpuBuffer,
    y: tessl::GpuBuffer,
}

fn operands(rt: &Arc<GpuRuntime>, m: usize, n: usize, k: usize, group: usize) -> Operands {
    Operands {
        packed: rt.alloc_buffer((n * k / 2).max(4)).unwrap(),
        sb: rt.alloc_buffer((n * (k / group) * 4).max(4)).unwrap(),
        x: rt.alloc_buffer((m * k * 2).max(4)).unwrap(),
        y: seeded(rt, (m * n).max(1), SENTINEL),
    }
}

fn call(rt: &Arc<GpuRuntime>, o: &Operands, n: u32, k: u32, group: u32, m: u32) -> Result<(), String> {
    nn::gemm_q4_mlx_tensorops(
        rt,
        Q4MlxBank {
            packed: &o.packed,
            scales_biases: &o.sb,
        },
        &o.x,
        &o.y,
        QuantShape {
            rows: n,
            cols: k,
            group_size: group,
        },
        m,
    )
}

#[test]
fn the_first_refused_k_is_the_derived_int32_extent_bound() {
    with_gpu(|rt| {
        // m = n = 1: the bound is i32::MAX / 1 rounded down to a multiple of G,
        // so the first refused K is exactly 2^31.
        for g in [32u32, 64, 128] {
            let max_k = nn::gemm_q4_mlx_tensorops_max_k(1, 1, g);
            assert_eq!(max_k, (i32::MAX as u32) / g * g, "G={g}");
            assert_eq!(max_k + g, 1 << 31, "G={g}: the first refused K is 2^31");
        }
        // Wider shapes: the bound follows max(m, n), and the entry agrees with it
        // at the boundary and one group past it, without touching the GPU. The
        // operands are tiny; an accepted K is stopped by the buffer-length
        // check instead, which names a buffer, not the 32-bit limit.
        rt.take_dispatch_count();
        let cases = [
            (1u32, 1u32, 32u32),
            (1, 1, 128),
            (33, 70, 64),
            (4096, 2, 32),
            (3, 100_000, 128),
        ];
        for (m, n, g) in cases {
            let max_k = nn::gemm_q4_mlx_tensorops_max_k(m, n, g);
            assert_eq!(max_k % g, 0);
            let o = operands(rt, 1, 1, 32, 32);
            let ok = call(rt, &o, n, max_k, g, m).expect_err("tiny buffers cannot hold the bound");
            assert!(
                !ok.contains("32-bit") && ok.contains("gemm_q4_mlx_tensorops"),
                "m={m} n={n} G={g} K={max_k} must pass the extent guard, got: {ok}"
            );
            let over = call(rt, &o, n, max_k + g, g, m).expect_err("one group past the bound");
            assert!(
                over.contains("32-bit"),
                "m={m} n={n} G={g} K={} must be refused as past signed 32-bit indexing: {over}",
                max_k + g
            );
        }
        // Brute force the claim itself: every product the kernel indexes with is
        // within i32 at max_k, and the binding one (m*K or n*K) is past it one group on.
        for (m, n, g) in cases {
            let k = nn::gemm_q4_mlx_tensorops_max_k(m, n, g) as u64;
            let widest = m.max(n) as u64;
            assert!(widest * k <= i32::MAX as u64);
            assert!(widest * (k + g as u64) > i32::MAX as u64);
        }
        assert_eq!(rt.take_dispatch_count(), 0, "a refused call encoded work");
    });
}

#[test]
fn every_refusal_happens_before_any_gpu_work() {
    with_gpu(|rt| {
        let o = operands(rt, 40, 40, 256, 64);
        rt.take_dispatch_count();
        let cases: Vec<(&str, Result<(), String>, &str)> = vec![
            (
                "group 256 (MLX domain, not instantiated)",
                call(rt, &o, 40, 256, 256, 40),
                "group_size 256",
            ),
            ("group 16", call(rt, &o, 40, 256, 16, 40), "group_size 16"),
            ("group 0", call(rt, &o, 40, 256, 0, 40), "group_size"),
            (
                "cols not a multiple of the group",
                call(rt, &o, 40, 250, 64, 40),
                "multiple of group_size",
            ),
            ("cols = 0", call(rt, &o, 40, 0, 64, 40), "cols must be non-zero"),
            (
                "packed too short",
                call(
                    rt,
                    &operands_with(rt, |p| p.packed = rt.alloc_buffer(64).unwrap()),
                    40,
                    256,
                    64,
                    40,
                ),
                "packed",
            ),
            (
                "scales_biases too short",
                call(
                    rt,
                    &operands_with(rt, |p| p.sb = rt.alloc_buffer(64).unwrap()),
                    40,
                    256,
                    64,
                    40,
                ),
                "scales_biases",
            ),
            (
                "x too short",
                call(
                    rt,
                    &operands_with(rt, |p| p.x = rt.alloc_buffer(64).unwrap()),
                    40,
                    256,
                    64,
                    40,
                ),
                "x_bf16",
            ),
            (
                "y too short",
                call(
                    rt,
                    &operands_with(rt, |p| p.y = rt.alloc_buffer(64).unwrap()),
                    40,
                    256,
                    64,
                    40,
                ),
                " y",
            ),
            ("m * cols past i32", call(rt, &o, 40, 256, 64, u32::MAX), "32-bit"),
        ];
        for (what, got, want) in cases {
            let err = got.expect_err(what);
            assert!(err.contains(want), "{what}: expected {want:?} in: {err}");
        }
        // y overlapping an input.
        let alias = Operands {
            packed: rt.alloc_buffer(40 * 256 / 2).unwrap(),
            sb: rt.alloc_buffer(40 * 4 * 4).unwrap(),
            x: rt.alloc_buffer(40 * 256 * 2).unwrap(),
            y: rt.alloc_buffer(40 * 40 * 4).unwrap(),
        };
        let err = nn::gemm_q4_mlx_tensorops(
            rt,
            Q4MlxBank {
                packed: &alias.packed,
                scales_biases: &alias.sb,
            },
            &alias.x,
            &alias.x,
            QuantShape {
                rows: 40,
                cols: 256,
                group_size: 64,
            },
            40,
        )
        .expect_err("y aliasing x");
        assert!(err.contains("gemm_q4_mlx_tensorops"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0, "a refused call encoded work");
        assert!(
            o.y.read_f32().iter().take(1600).all(|v| *v == SENTINEL),
            "y written by a refused call"
        );

        // m = 0 or rows = 0 is a validated no-op: Ok, no dispatch, y untouched.
        call(rt, &o, 40, 256, 64, 0).expect("m = 0");
        call(rt, &o, 0, 256, 64, 40).expect("rows = 0");
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}

fn operands_with(rt: &Arc<GpuRuntime>, edit: impl FnOnce(&mut Operands)) -> Operands {
    let mut o = operands(rt, 40, 40, 256, 64);
    edit(&mut o);
    o
}

/// A runtime whose metallib has no TensorOps kernels refuses by name, before
/// validation or encode. The library is a one-kernel overlay compiled here, so
/// `has_tensorops` (the presence of `matmul2d_tensorops_f32`) is false.
#[test]
fn hardware_without_tensorops_is_refused_before_encode() {
    with_gpu(|full| {
        let dir = std::env::temp_dir().join(format!("tessl-q4to-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (src, air, lib) = (dir.join("k.metal"), dir.join("k.air"), dir.join("k.metallib"));
        std::fs::write(
            &src,
            "#include <metal_stdlib>\nkernel void noop(device float *a [[buffer(0)]]) { a[0] = 1.0f; }\n",
        )
        .unwrap();
        let sdk = std::process::Command::new("xcrun")
            .args(["--sdk", "macosx", "--show-sdk-path"])
            .output()
            .expect("xcrun");
        let sdk = String::from_utf8(sdk.stdout).unwrap().trim().to_string();
        for args in [
            vec![
                "metal",
                "-c",
                src.to_str().unwrap(),
                "-isysroot",
                &sdk,
                "-o",
                air.to_str().unwrap(),
            ],
            vec!["metallib", air.to_str().unwrap(), "-o", lib.to_str().unwrap()],
        ] {
            let out = std::process::Command::new("xcrun").args(&args).output().unwrap();
            assert!(
                out.status.success(),
                "{args:?}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        let rt = GpuRuntime::from_metallib_path(&lib).expect("runtime over a TensorOps-free metallib");
        let _ = std::fs::remove_dir_all(&dir);
        assert!(!rt.has_tensorops() && full.has_tensorops());
        let o = operands(&rt, 8, 32, 128, 32);
        rt.take_dispatch_count();
        let err = call(&rt, &o, 32, 128, 32, 8).expect_err("no TensorOps");
        assert!(
            err.contains("gemm_q4_mlx_tensorops") && err.contains("TensorOps is unavailable"),
            "{err}"
        );
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}
