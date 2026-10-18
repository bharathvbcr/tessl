//! First GPU execution of the `int4b_format` / `uint4b_format` TensorOps probe
//! (`tools/probes/int4b/`, which only ever compiled).
//!
//! Two questions, answered by running:
//!
//! 1. Inside a byte, which nibble is tensor element 0, and is `int4b_format`
//!    two's-complement (stored 8..15 -> -8..-1)?
//! 2. Does a per-group `matmul2d` (half activations x `uint4b_format` weights,
//!    `out = sum_g s*P_g + b*xsum_g`) reproduce tessl's MLX Q4 GEMM
//!    (`nn::gemm_q4_mlx`) on the same packed bank bytes, and a CPU f64
//!    reference of the dequantized product?
//!
//! The kernels live here as source strings, compiled with `xcrun metal` at
//! test time and registered through `GpuRuntime::add_metallib_bytes` (the
//! overlay path `tests/public_api_errors.rs` already uses). Nothing is added
//! under `kernels/`, so the production kernel census is untouched. The tests
//! need a Metal 4 device with TensorOps; they are not hosted-CI tests.
//!
//! # Measured on Apple GPU (see `MEASURED_*` below)
//!
//! The constants record what the device returned; the tests assert them, so a
//! different answer on another GPU generation fails loudly instead of silently
//! mis-decoding weights.

mod common;

use common::{buf_bf16, pack_nibbles, round_trip_bf16, seeded, with_gpu, SplitMix};
use tessl::dispatch::{dispatch_2d_tg, set_gpu_buf, set_u32};
use tessl::nn::{self, Q4MlxBank, Q4MlxLayout, QuantShape};
use tessl::tensor::f32_slice_to_f16;
use tessl::GpuRuntime;

/// Element 0 of a packed byte is the LOW nibble (`byte & 0xF`); element 1 is
/// the high nibble. This is the same order as `Q4MlxBank` / MLX, so a bank's
/// bytes feed `uint4b_format` unchanged.
const MEASURED_LOW_NIBBLE_FIRST: bool = true;
/// `int4b_format` reads a stored nibble 8..=15 as -8..=-1 (two's complement).
const MEASURED_INT4_SIGNED: bool = true;

/// Compile `source` to a metallib with the flags `build.rs` uses (no -Werror).
fn compile_metallib(tag: &str, source: &str) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("tessl-int4b-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    let src = dir.join("k.metal");
    let air = dir.join("k.air");
    let lib = dir.join("k.metallib");
    std::fs::write(&src, source).unwrap();
    let sdk = std::process::Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-path"])
        .output()
        .expect("xcrun --show-sdk-path");
    assert!(sdk.status.success(), "xcrun --show-sdk-path failed");
    let sdk = String::from_utf8(sdk.stdout).unwrap().trim().to_string();
    let steps: [Vec<&str>; 2] = [
        vec![
            "metal",
            "-std=metal4.0",
            "-O2",
            "-fmetal-math-mode=fast",
            "-isysroot",
            &sdk,
            "-mmacosx-version-min=26.0",
            "-c",
            src.to_str().unwrap(),
            "-o",
            air.to_str().unwrap(),
        ],
        vec!["metallib", air.to_str().unwrap(), "-o", lib.to_str().unwrap()],
    ];
    for args in steps {
        let out = std::process::Command::new("xcrun").args(&args).output().expect("xcrun");
        assert!(
            out.status.success(),
            "xcrun {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let bytes = std::fs::read(&lib).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

fn load_kernels(rt: &GpuRuntime, tag: &str, source: &str) {
    assert!(rt.has_tensorops(), "int4b TensorOps tests need a TensorOps device");
    rt.add_metallib_bytes(&compile_metallib(tag, source))
        .expect("register overlay metallib");
}

// ------------------------------------------------------- nibble order ---

/// `C[32,32] = I[32,32](half) x B[32,32](int4b | uint4b)`. The LHS is the
/// identity, so C reads B back element for element, in tensor order
/// (row-major, dim0 = n fastest, one nibble per element).
const NIBBLE_SRC: &str = r#"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#define NIB(NAME, RT, DT)                                                           \
kernel void NAME(device uchar *A [[buffer(0)]], device uchar *B [[buffer(1)]],      \
                 device uchar *C [[buffer(2)]]) {                                   \
    constexpr auto d = matmul2d_descriptor(32, 32, dynamic_length_v<int>, false,    \
                                           false, false,                            \
                                           matmul2d_descriptor::mode::multiply);    \
    matmul2d<d, execution_simdgroups<4>> op;                                        \
    using TA = tensor<device half, dextents<int32_t, 2>, tensor_inline>;            \
    using TB = tensor<device RT, dextents<int32_t, 2>, tensor_inline>;              \
    using TC = tensor<device DT, dextents<int32_t, 2>, tensor_inline>;              \
    TA tA((TA::data_handle_type)A, dextents<int32_t, 2>{32, 32});                   \
    TB tB((TB::data_handle_type)B, dextents<int32_t, 2>{32, 32});                   \
    TC tC((TC::data_handle_type)C, dextents<int32_t, 2>{32, 32});                   \
    op.run(tA, tB, tC);                                                             \
}
NIB(nib_uint4b, uint4b_format, half)
NIB(nib_int4b, int4b_format, float)
"#;

/// Run one 32x32 identity product; returns C as f32.
fn read_back_b(rt: &std::sync::Arc<GpuRuntime>, kernel: &str, packed: &[u8], half_out: bool) -> Vec<f32> {
    let mut ident = vec![0.0f32; 32 * 32];
    for i in 0..32 {
        ident[i * 32 + i] = 1.0;
    }
    let a = rt.alloc_buffer(32 * 32 * 2).unwrap();
    a.write_f16_bits(&f32_slice_to_f16(&ident));
    let b = rt.alloc_buffer(packed.len()).unwrap();
    b.write_bytes(packed);
    let c = rt.alloc_buffer(32 * 32 * 4).unwrap();
    c.zero();
    let p = rt.pipeline(kernel).unwrap();
    dispatch_2d_tg(rt, &p, 1, 1, 128, |bnd| {
        set_gpu_buf(bnd, &a, 0);
        set_gpu_buf(bnd, &b, 1);
        set_gpu_buf(bnd, &c, 2);
    })
    .unwrap();
    rt.synchronize().unwrap();
    if half_out {
        c.read_u32()
            .iter()
            .take(32 * 32 / 2)
            .flat_map(|w| [(w & 0xffff) as u16, (w >> 16) as u16])
            .map(tessl::tensor::f16_bits_to_f32)
            .collect()
    } else {
        c.read_f32()
    }
}

/// Nibble value of element `i` of the test pattern, in "element order": every
/// value 0..=15 occurs, and adjacent elements differ, so swapping the nibbles
/// of a byte is detectable.
fn pattern(i: usize) -> u8 {
    ((i * 7 + 3) % 16) as u8
}

fn decode(nib: u8, signed: bool) -> f32 {
    if signed && nib >= 8 {
        nib as f32 - 16.0
    } else {
        nib as f32
    }
}

#[test]
fn int4b_and_uint4b_nibble_order_and_signedness_are_as_measured() {
    with_gpu(|rt| {
        load_kernels(rt, "nib", NIBBLE_SRC);
        let elems: Vec<u8> = (0..32 * 32).map(pattern).collect();
        // `pack_nibbles` is low nibble first: element 2j = byte & 0xF.
        let packed_low_first = pack_nibbles(&elems);

        let got_u = read_back_b(rt, "nib_uint4b", &packed_low_first, true);
        let got_i = read_back_b(rt, "nib_int4b", &packed_low_first, false);

        // The four hypotheses for (nibble order, signedness), for the report.
        let mut verdict = Vec::new();
        for (low_first, name) in [(true, "low-nibble-first"), (false, "high-nibble-first")] {
            let order = |i: usize| -> u8 {
                let byte = packed_low_first[i / 2];
                let (lo, hi) = (byte & 0xf, byte >> 4);
                match (i % 2 == 0) == low_first {
                    true => lo,
                    false => hi,
                }
            };
            let u_ok = (0..1024).all(|i| got_u[i] == decode(order(i), false));
            let i_ok = (0..1024).all(|i| got_i[i] == decode(order(i), true));
            let i_unsigned_ok = (0..1024).all(|i| got_i[i] == decode(order(i), false));
            verdict.push(format!(
                "{name}: uint4b matches={u_ok} int4b-signed matches={i_ok} int4b-unsigned matches={i_unsigned_ok}"
            ));
        }
        eprintln!(
            "int4b nibble probe: {verdict:?}\nfirst 16 uint4b: {:?}\nfirst 16 int4b: {:?}",
            &got_u[..16],
            &got_i[..16]
        );

        let order = |i: usize| -> u8 {
            let byte = packed_low_first[i / 2];
            if (i % 2 == 0) == MEASURED_LOW_NIBBLE_FIRST {
                byte & 0xf
            } else {
                byte >> 4
            }
        };
        for i in 0..1024 {
            assert_eq!(got_u[i], decode(order(i), false), "uint4b element {i}: {verdict:?}");
            assert_eq!(
                got_i[i],
                decode(order(i), MEASURED_INT4_SIGNED),
                "int4b element {i}: {verdict:?}"
            );
        }
        // Both signs must actually have been exercised.
        assert!(got_i.iter().any(|v| *v < 0.0) && got_i.iter().any(|v| *v > 0.0));
    });
}

// ----------------------------------------------------------- MLX parity ---

/// One matmul2d per K group into a fresh cooperative destination, then the
/// affine epilogue in registers. `W` is [N,K] nibbles, K contiguous
/// (transpose_right), i.e. exactly `Q4MlxBank`'s row-major bytes. The scale/
/// bias pair is read as `bfloat2` from the bank's own `scales_biases` buffer.
fn mlx_group_src(group: usize) -> String {
    format!(
        r#"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
constant constexpr int G = {group};
kernel void int4b_mlx_g{group}(device half *A [[buffer(0)]], device uchar *W [[buffer(1)]],
              device float *C [[buffer(2)]], device const bfloat2 *sb [[buffer(3)]],
              device const float *xsum [[buffer(4)]], constant int &M [[buffer(5)]],
              constant int &N [[buffer(6)]], constant int &K [[buffer(7)]],
              uint2 tg [[threadgroup_position_in_grid]]) {{
    constexpr auto d = matmul2d_descriptor(32, 32, G, false, true, false,
                                           matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    using TW = tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline>;
    auto tA = tensor(A, dextents<int32_t, 2>{{K, M}}, array<int, 2>{{1, K}});
    TW tW((TW::data_handle_type)W, dextents<int32_t, 2>{{K, N}}, array<int, 2>{{1, K}});
    auto tC = tensor(C, dextents<int32_t, 2>{{N, M}}, array<int, 2>{{1, N}});
    const int tx = tg.x * 32, ty = tg.y * 32;
    auto sA0 = tA.slice(0, ty);
    auto sW0 = tW.slice(0, tx);
    auto acc = op.template get_destination_cooperative_tensor<decltype(sA0), decltype(sW0), float>();
    auto grp = op.template get_destination_cooperative_tensor<decltype(sA0), decltype(sW0), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc.set(i, 0.0f);
    const int groups = K / G;
    for (int g = 0; g < groups; ++g) {{
        auto a = tA.slice(g * G, ty);
        auto w = tW.slice(g * G, tx);
        op.run(a, w, grp);
        for (uint16_t i = 0; i < grp.get_capacity(); ++i) {{
            if (!grp.is_valid_element(i)) continue;
            const auto idx = grp.get_multidimensional_index(i);  // {{n, m}}
            const int n = tx + (int)idx[0], m = ty + (int)idx[1];
            const bfloat2 p = sb[n * groups + g];
            acc[i] = fma(float(p.x), grp[i], fma(float(p.y), xsum[m * groups + g], acc[i]));
        }}
    }}
    acc.store(tC.slice(tx, ty));
}}
"#
    )
}

struct Case {
    m: usize,
    n: usize,
    k: usize,
    group: usize,
}

#[derive(Default)]
struct Stats {
    max_abs: f64,
    max_rel: f64,
    /// Largest `err / (gamma_n * S)`: 1.0 means the derived bound is exactly met.
    max_bound_frac: f64,
}

impl Stats {
    fn add(&mut self, got: f64, want: f64, bound: f64) {
        let e = (got - want).abs();
        self.max_abs = self.max_abs.max(e);
        self.max_rel = self.max_rel.max(e / want.abs().max(1.0));
        self.max_bound_frac = self.max_bound_frac.max(e / bound);
    }
}

fn run_case(rt: &std::sync::Arc<GpuRuntime>, c: &Case) -> (Stats, Stats, Stats) {
    let &Case { m, n, k, group } = c;
    let groups = k / group;
    let mut rng = SplitMix::new(0x4b17 ^ (m * 1000003 + n * 4099 + k * 17 + group) as u64);

    // Activations: multiples of 1/64 in [-1, 1]. 7 significant bits, so exact in
    // both half (11) and bf16 (8): the TensorOps path and the simd path read
    // bit-identical operands, and every product x*q (q <= 15) is exact in f32.
    let x: Vec<f32> = (0..m * k).map(|_| (rng.unit() * 64.0).round() / 64.0).collect();
    let nibbles: Vec<u8> = (0..n * k).map(|_| (rng.range(0, 15)) as u8).collect();
    // bf16-valued scale in ~[0.01, 0.05], bias in ~[-0.5, 0.1]; rounded through
    // bf16 so the CPU reference sees the loaded values.
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

    // CPU f64 reference and the magnitude S that drives the error bound.
    let mut want = vec![0.0f64; m * n];
    let mut s_mag = vec![0.0f64; m * n];
    let mut xsum = vec![0.0f32; m * groups];
    for mi in 0..m {
        for g in 0..groups {
            xsum[mi * groups + g] = (0..group).map(|j| x[mi * k + g * group + j] as f64).sum::<f64>() as f32;
        }
    }
    for mi in 0..m {
        for ni in 0..n {
            let (mut acc, mut mag) = (0.0f64, 0.0f64);
            for g in 0..groups {
                let (s, b) = (sb[(ni * groups + g) * 2] as f64, sb[(ni * groups + g) * 2 + 1] as f64);
                let (mut qx, mut aqx, mut ax) = (0.0f64, 0.0f64, 0.0f64);
                for j in 0..group {
                    let xv = x[mi * k + g * group + j] as f64;
                    let q = nibbles[ni * k + g * group + j] as f64;
                    qx += q * xv;
                    aqx += q * xv.abs();
                    ax += xv.abs();
                }
                acc += s * qx + b * xsum[mi * groups + g] as f64;
                // |s| * sum|q x| + |b| * sum|x|: the magnitude of every term the
                // TensorOps epilogue adds (it can exceed sum|w x| when b cancels s*q).
                mag += s.abs() * aqx + b.abs() * ax;
            }
            want[mi * n + ni] = acc;
            s_mag[mi * n + ni] = mag;
        }
    }

    // Shared bank bytes: tessl's row-major Q4MlxBank, low nibble first.
    let packed = pack_nibbles(&nibbles);
    let pb = rt.alloc_buffer(packed.len()).unwrap();
    pb.write_bytes(&packed);
    let sbb = buf_bf16(rt, &sb);

    // TensorOps path.
    let a = rt.alloc_buffer(m * k * 2).unwrap();
    a.write_f16_bits(&f32_slice_to_f16(&x));
    let xs = common::buf(rt, &xsum);
    let c_buf = seeded(rt, m * n, -7.7e28);
    let p = rt.pipeline(&format!("int4b_mlx_g{group}")).unwrap();
    dispatch_2d_tg(rt, &p, n / 32, m.div_ceil(32), 128, |bnd| {
        set_gpu_buf(bnd, &a, 0);
        set_gpu_buf(bnd, &pb, 1);
        set_gpu_buf(bnd, &c_buf, 2);
        set_gpu_buf(bnd, &sbb, 3);
        set_gpu_buf(bnd, &xs, 4);
        set_u32(bnd, m as u32, 5);
        set_u32(bnd, n as u32, 6);
        set_u32(bnd, k as u32, 7);
    })
    .unwrap();
    rt.synchronize().unwrap();
    let tensorops: Vec<f32> = c_buf.read_f32()[..m * n].to_vec();
    assert!(
        tensorops.iter().all(|v| *v != -7.7e28 && v.is_finite()),
        "TensorOps kernel left outputs unwritten or non-finite for {m}x{n}x{k} g{group}"
    );

    // tessl's own MLX path, <= GEMM_Q4_MLX_MAX_M rows per call.
    let bank = Q4MlxBank {
        packed: &pb,
        scales_biases: &sbb,
    };
    let mut simd = vec![0.0f32; m * n];
    for r0 in (0..m).step_by(nn::GEMM_Q4_MLX_MAX_M as usize) {
        let mc = (m - r0).min(nn::GEMM_Q4_MLX_MAX_M as usize);
        let xb = buf_bf16(rt, &x[r0 * k..(r0 + mc) * k]);
        let yb = seeded(rt, mc * n, -7.7e28);
        nn::gemm_q4_mlx(
            rt,
            bank,
            &xb,
            &yb,
            QuantShape {
                rows: n as u32,
                cols: k as u32,
                group_size: group as u32,
            },
            mc as u32,
            Q4MlxLayout::RowMajor,
            None,
        )
        .unwrap();
        rt.synchronize().unwrap();
        simd[r0 * n..(r0 + mc) * n].copy_from_slice(&yb.read_f32()[..mc * n]);
    }
    assert!(
        simd.iter().all(|v| *v != -7.7e28 && v.is_finite()),
        "simd path unwritten"
    );

    // Error model. Every path forms each output from at most `terms` f32 adds
    // of magnitude <= S, so |fl(sum) - sum| <= gamma_terms * S with
    // gamma_t = t*u / (1 - t*u), u = 2^-24, whatever the summation order
    // (simd_sum tree, matmul2d internal order, fma contraction). The operand
    // products are exact (7-bit x times 4-bit q fit f32; bf16 s,b are f32
    // values), so no operand-rounding term enters.
    let terms = (k + 2 * groups) as f64;
    let u = 2f64.powi(-24);
    let gamma = terms * u / (1.0 - terms * u);
    let (mut t_ref, mut s_ref, mut t_s) = (Stats::default(), Stats::default(), Stats::default());
    for i in 0..m * n {
        let b1 = gamma * s_mag[i];
        t_ref.add(tensorops[i] as f64, want[i], b1);
        s_ref.add(simd[i] as f64, want[i], b1);
        t_s.add(tensorops[i] as f64, simd[i] as f64, 2.0 * b1);
    }
    (t_ref, s_ref, t_s)
}

#[test]
fn tensorops_mlx_q4_matches_gemm_q4_mlx_and_the_f64_reference() {
    with_gpu(|rt| {
        load_kernels(rt, "mlx32", &mlx_group_src(32));
        load_kernels(rt, "mlx64", &mlx_group_src(64));
        let mut report = Vec::new();
        for group in [32usize, 64] {
            for m in [1usize, 8, 33] {
                for n in [32usize, 64] {
                    for k in [64usize, 128] {
                        let (t_ref, s_ref, t_s) = run_case(rt, &Case { m, n, k, group });
                        let line = format!(
                            "M={m} N={n} K={k} G={group}: TO-vs-f64 abs {:.3e} rel {:.3e} ({:.3} of bound); \
                             simd-vs-f64 abs {:.3e} rel {:.3e} ({:.3}); TO-vs-simd abs {:.3e} rel {:.3e} ({:.3} of 2x bound)",
                            t_ref.max_abs,
                            t_ref.max_rel,
                            t_ref.max_bound_frac,
                            s_ref.max_abs,
                            s_ref.max_rel,
                            s_ref.max_bound_frac,
                            t_s.max_abs,
                            t_s.max_rel,
                            t_s.max_bound_frac
                        );
                        eprintln!("{line}");
                        report.push((line, t_ref.max_bound_frac, s_ref.max_bound_frac, t_s.max_bound_frac));
                    }
                }
            }
        }
        for (line, a, b, c) in &report {
            assert!(*a <= 1.0, "TensorOps exceeds the f32 error bound vs f64: {line}");
            assert!(*b <= 1.0, "simd path exceeds the f32 error bound vs f64: {line}");
            assert!(*c <= 1.0, "TensorOps vs simd exceeds twice the f32 bound: {line}");
        }
    });
}
