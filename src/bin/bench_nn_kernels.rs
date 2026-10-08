//! Throughput of the `nn` kernel library.
//!
//! These kernels are small. A 1x4096 RMSNorm moves 32 KB and finishes in
//! microseconds, which is far under the ~0.25 ms submit-and-wait floor that
//! `docs/benchmarking.md` measures for this protocol. Timed one dispatch at a
//! time with a `synchronize()` after each, every kernel here would report
//! roughly 0.25 ms and the table would describe the driver rather than the
//! shaders.
//!
//! So each kernel is timed twice, and both numbers are printed:
//!
//! * **batched** — `set_async_encode(true)`, `BATCH` dispatches accumulated
//!   into one command buffer, then a single `synchronize()`, divided by
//!   `BATCH`. This is what a decode loop actually pays, and it is the number
//!   the GB/s column derives from.
//! * **solo** — `set_async_encode(false)`, one dispatch, one synchronize. This
//!   is the dispatch floor, printed rather than hidden so nobody reads the
//!   batched figure as a latency.
//!
//! The gap between the two columns is what `async_encode` buys, and it is
//! large. Note that it defaults to **off**.
//!
//! Every kernel is checked for a plausible result before being timed. A kernel
//! that silently wrote nothing would otherwise post the best number in the
//! table.

mod common;

use common::{env_usize, median};
use std::sync::Arc;
use std::time::Instant;

use tessl::tensor::GpuBuffer;
use tessl::{nn, GpuRuntime};

const BATCH: usize = 64;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 32) as u32) as f64 / (u32::MAX as f64) * 2.0 - 1.0) as f32
        })
        .collect()
}

fn buf(rt: &Arc<GpuRuntime>, data: &[f32]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1) * 4).expect("alloc");
    b.write_f32(data);
    b
}

struct Row {
    name: String,
    shape: String,
    batched_us: f64,
    solo_us: f64,
    gb_s: f64,
}

/// Time `f` batched and solo. `bytes` is the traffic one dispatch must move at
/// minimum — operands read plus results written, counted once each.
fn measure(
    rt: &Arc<GpuRuntime>,
    name: &str,
    shape: &str,
    bytes: f64,
    warmup: usize,
    iters: usize,
    mut f: impl FnMut() -> Result<(), String>,
) -> Result<Row, String> {
    for _ in 0..warmup {
        f()?;
        rt.synchronize()?;
    }

    // `async_encode` defaults to false, in which case every dispatch gets its
    // own command buffer and commits — so a loop of BATCH dispatches costs
    // BATCH times one dispatch and this arm would silently measure the same
    // thing as `solo`. The first version of this benchmark did exactly that and
    // reported batched == solo across the board. Turning it on is the whole
    // point of the measurement.
    rt.set_async_encode(true)?;
    let mut batched = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        for _ in 0..BATCH {
            f()?;
        }
        rt.synchronize()?;
        batched.push(t0.elapsed().as_secs_f64() * 1e6 / BATCH as f64);
    }
    rt.set_async_encode(false)?;

    let mut solo = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f()?;
        rt.synchronize()?;
        solo.push(t0.elapsed().as_secs_f64() * 1e6);
    }

    let b = median(batched)?;
    Ok(Row {
        name: name.to_string(),
        shape: shape.to_string(),
        batched_us: b,
        solo_us: median(solo)?,
        gb_s: bytes / (b * 1e-6) / 1e9,
    })
}

/// Sentinel seeded into every output before the kernel runs. Chosen so no
/// kernel here could legitimately produce it.
const UNWRITTEN: f32 = -1.234_567_9e30;

/// A kernel that skips work is fast for the wrong reason. Refuse to report one.
///
/// The first version of this only required *some* element to be non-zero, and
/// that was not enough: `gemv_q4_tiled` was dispatched with `rows / 128`
/// threadgroups instead of `rows`, wrote 4 of 512 rows, and passed — then
/// posted 3,077 GB/s, which is several times what this machine can do. Seeding
/// the whole output and requiring every live element to change is what makes a
/// partial write fail instead of winning the table.
fn assert_wrote_everything(what: &str, out: &GpuBuffer, elems: usize) {
    let got = out.read_f32();
    let live = &got[..elems.min(got.len())];
    assert!(
        live.iter().all(|v| v.is_finite()),
        "{what}: produced a non-finite value"
    );
    if let Some(i) = live.iter().position(|v| *v == UNWRITTEN) {
        let n = live.iter().filter(|v| **v == UNWRITTEN).count();
        panic!(
            "{what}: {n} of {elems} output elements were never written (first at \
             {i}). A kernel that skips work is fast for the wrong reason, so its \
             timing is not reported."
        );
    }
}

/// Seed the output, run the kernel once, and assert it wrote every element.
///
/// Seeding *here* rather than at allocation matters: `gemv_q4` times two arms
/// against one `y`, so a buffer seeded once and filled by the first arm would
/// let the second pass no matter what it wrote.
fn verify(
    rt: &Arc<GpuRuntime>,
    what: &str,
    out: &GpuBuffer,
    elems: usize,
    mut f: impl FnMut() -> Result<(), String>,
) -> Result<(), String> {
    out.write_f32(&vec![UNWRITTEN; elems.max(1)]);
    f()?;
    rt.synchronize()?;
    assert_wrote_everything(what, out, elems);
    Ok(())
}

/// Output buffer seeded with [`UNWRITTEN`], so a partial write is detectable.
fn out_buf(rt: &Arc<GpuRuntime>, elems: usize) -> GpuBuffer {
    buf(rt, &vec![UNWRITTEN; elems.max(1)])
}

/// Emit the kernel trace for `bench/kernel_coverage.py`. Prints nothing when
/// tracing is off, so normal runs are unchanged.
fn emit_kernel_trace() {
    if tessl::runtime::kernel_trace_enabled() {
        eprintln!(
            "KERNEL_TRACE {}",
            tessl::runtime::traced_kernels().join(",")
        );
    }
}

fn run() -> Result<(), String> {
    let warmup = env_usize("BENCH_WARMUP", 10, 0)?;
    let iters = env_usize("BENCH_ITERS", 50, 1)?;
    let rt = GpuRuntime::new()?;

    println!("device: {}", rt.device_name());
    println!(
        "batched = {BATCH} dispatches per command buffer; solo = 1 dispatch + \
         synchronize\n"
    );

    let mut rows: Vec<Row> = Vec::new();

    // ---------------------------------------------------------- RMSNorm ---
    for &(r, d) in &[(1usize, 4096usize), (512, 4096), (2048, 4096)] {
        let x = buf(&rt, &fill(r * d, 0x11));
        let w = buf(&rt, &fill(d, 0x12));
        let o = out_buf(&rt, r * d);
        verify(&rt, "rms_norm_f32", &o, r * d, || {
            nn::rms_norm_f32(&rt, &x, &w, &o, r as u32, d as u32, 1e-6)
        })?;
        // reads x and weight, writes out
        let bytes = ((2 * r * d + d) * 4) as f64;
        rows.push(measure(
            &rt,
            "rms_norm_f32",
            &format!("{r}x{d}"),
            bytes,
            warmup,
            iters,
            || nn::rms_norm_f32(&rt, &x, &w, &o, r as u32, d as u32, 1e-6),
        )?);
    }

    // ------------------------------------------------------- MLP gating ---
    for &n in &[4096usize, 1 << 20, 8 << 20] {
        let g = buf(&rt, &fill(n, 0x21));
        let u = buf(&rt, &fill(n, 0x22));
        let o = out_buf(&rt, n);
        let bytes = (3 * n * 4) as f64;

        verify(&rt, "mlp_silu", &o, n, || {
            nn::mlp_silu(&rt, &g, &u, &o, n as u32)
        })?;
        rows.push(measure(
            &rt,
            "mlp_silu",
            &format!("n={n}"),
            bytes,
            warmup,
            iters,
            || nn::mlp_silu(&rt, &g, &u, &o, n as u32),
        )?);

        verify(&rt, "mlp_gelu_tanh", &o, n, || {
            nn::mlp_gelu_tanh(&rt, &g, &u, &o, n as u32)
        })?;
        rows.push(measure(
            &rt,
            "mlp_gelu_tanh",
            &format!("n={n}"),
            bytes,
            warmup,
            iters,
            || nn::mlp_gelu_tanh(&rt, &g, &u, &o, n as u32),
        )?);
    }

    // -------------------------------------------------------- Reductions ---
    for &(r, c) in &[(32usize, 1024usize), (512, 4096), (2048, 8192)] {
        let x = buf(&rt, &fill(r * c, 0x31));
        let o = out_buf(&rt, r * c);
        verify(&rt, "softmax_rows_f32", &o, r * c, || {
            nn::softmax_rows_f32(&rt, &x, &o, r as u32, c as u32)
        })?;
        rows.push(measure(
            &rt,
            "softmax_rows_f32",
            &format!("{r}x{c}"),
            (2 * r * c * 4) as f64,
            warmup,
            iters,
            || nn::softmax_rows_f32(&rt, &x, &o, r as u32, c as u32),
        )?);

        let s = out_buf(&rt, r);
        verify(&rt, "row_sum_f32", &s, r, || {
            nn::row_sum_f32(&rt, &x, &s, r as u32, c as u32)
        })?;
        rows.push(measure(
            &rt,
            "row_sum_f32",
            &format!("{r}x{c}"),
            ((r * c + r) * 4) as f64,
            warmup,
            iters,
            || nn::row_sum_f32(&rt, &x, &s, r as u32, c as u32),
        )?);

        nn::row_max_f32(&rt, &x, &s, r as u32, c as u32)?;
        rt.synchronize()?;
        rows.push(measure(
            &rt,
            "row_max_f32",
            &format!("{r}x{c}"),
            ((r * c + r) * 4) as f64,
            warmup,
            iters,
            || nn::row_max_f32(&rt, &x, &s, r as u32, c as u32),
        )?);
    }

    // -------------------------------------------------------- Q8 GEMV ---
    for &(r, c) in &[(4096usize, 4096usize), (11008, 4096)] {
        let group = 64usize;
        let groups = r * (c / group);
        let packed: Vec<u8> = (0..r * c).map(|i| ((i % 251) as i32 - 125) as u8).collect();
        let pb = rt.alloc_buffer(packed.len())?;
        pb.write_bytes(&packed);
        let sb = buf(&rt, &vec![0.01f32; groups]);
        let zb = buf(&rt, &vec![1.0f32; groups]);
        let xb = buf(&rt, &fill(c, 0x41));
        let yb = out_buf(&rt, r);

        verify(&rt, "gemv_q8", &yb, r, || {
            nn::gemv_q8(
                &rt,
                &pb,
                &sb,
                &zb,
                &xb,
                &yb,
                r as u32,
                c as u32,
                group as u32,
            )
        })?;
        // int8 weights dominate: r*c bytes, plus scales/zeros, x and y in f32
        let bytes = (r * c + 2 * groups * 4 + c * 4 + r * 4) as f64;
        rows.push(measure(
            &rt,
            "gemv_q8",
            &format!("{r}x{c}"),
            bytes,
            warmup,
            iters,
            || {
                nn::gemv_q8(
                    &rt,
                    &pb,
                    &sb,
                    &zb,
                    &xb,
                    &yb,
                    r as u32,
                    c as u32,
                    group as u32,
                )
            },
        )?);
    }

    // -------------------------------------------------------- Q4 GEMV ---
    // Both arms of `tiled`, because the crate already offers a threadgroup-per-
    // row-tile alternative to the one-thread-per-row kernel and the question is
    // whether the default arm is the one anybody should use.
    for &(r, c) in &[(4096usize, 4096usize), (11008, 4096)] {
        let group = 64usize;
        let groups = r * (c / group);
        let packed = rt.alloc_buffer(r * c / 2)?;
        packed.write_bytes(&(0..r * c / 2).map(|i| (i % 251) as u8).collect::<Vec<u8>>());
        let sc = buf(&rt, &vec![0.02f32; groups]);
        let ze = buf(&rt, &vec![7.0f32; groups]);
        let xb = buf(&rt, &fill(c, 0x51));
        let yb = out_buf(&rt, r);
        let shape = nn::QuantShape {
            rows: r as u32,
            cols: c as u32,
            group_size: group as u32,
        };
        let bank = nn::Q4Bank {
            packed: &packed,
            scales: &sc,
            zeros: &ze,
        };
        // 4-bit weights: half a byte each, plus scales/zeros, x and y.
        let bytes = (r * c / 2 + 2 * groups * 4 + c * 4 + r * 4) as f64;
        for &tiled in &[false, true] {
            verify(&rt, "gemv_q4", &yb, r, || {
                nn::gemv_q4(&rt, bank, &xb, &yb, shape, tiled)
            })?;
            let name = if tiled {
                "gemv_q4 [tiled]"
            } else {
                "gemv_q4 [row]"
            };
            rows.push(measure(
                &rt,
                name,
                &format!("{r}x{c}"),
                bytes,
                warmup,
                iters,
                || nn::gemv_q4(&rt, bank, &xb, &yb, shape, tiled),
            )?);
        }
    }

    // ------------------------------------------------------- int8 GEMM ---
    for &(m, n, k) in &[(512usize, 512usize, 512usize), (2048, 2048, 2048)] {
        let a = rt.alloc_buffer(m * k)?;
        a.write_bytes(&(0..m * k).map(|i| (i % 127) as u8).collect::<Vec<u8>>());
        let b = rt.alloc_buffer(k * n)?;
        b.write_bytes(&(0..k * n).map(|i| (i % 113) as u8).collect::<Vec<u8>>());
        let c = out_buf(&rt, m * n);
        verify(&rt, "gemm_i8_dequant", &c, m * n, || {
            nn::gemm_i8_dequant(&rt, &a, &b, &c, m as u32, n as u32, k as u32, 0.01, None)
        })?;
        let bytes = (m * k + k * n + m * n * 4) as f64;
        let mut row = measure(
            &rt,
            "gemm_i8_dequant",
            &format!("{m}x{n}x{k}"),
            bytes,
            warmup,
            iters,
            || nn::gemm_i8_dequant(&rt, &a, &b, &c, m as u32, n as u32, k as u32, 0.01, None),
        )?;
        // Compute-bound, so report GFLOP/s in the GB/s slot's place via name.
        let gflops = 2.0 * (m * n * k) as f64 / (row.batched_us * 1e-6) / 1e9;
        row.name = format!("gemm_i8_dequant [{gflops:.0} GFLOP/s]");
        rows.push(row);
    }

    // ---------------------------------------------------- Q4 MLX GEMV ---
    // Nineteen kernels reached through seven wrappers, and none of them had a
    // timing lane: the MLX-format Q4 family is the quantized decode path, and
    // `bench/kernel_coverage.py` showed it dispatched by nothing.
    //
    // Every arm is enumerated rather than sampled. `Q4MlxRowVariant` and
    // `Q4MlxLayout` each *select a kernel* rather than hint at one, so a lane
    // that fixes them measures one entry point and silently leaves its siblings
    // unmeasured -- which is exactly how they came to be uncovered.
    {
        let (rows_w, cols_w, group) = (4096usize, 4096usize, 64usize);
        let groups = rows_w * (cols_w / group);
        let mk_bank = |seed: u8| -> Result<(GpuBuffer, GpuBuffer), String> {
            let packed = rt.alloc_buffer(rows_w * cols_w / 2)?;
            packed.write_bytes(
                &(0..rows_w * cols_w / 2)
                    .map(|i| (i as u8).wrapping_mul(seed))
                    .collect::<Vec<u8>>(),
            );
            // One bfloat2 (scale, bias) per group, packed as a u32.
            let sb = rt.alloc_buffer(groups * 4)?;
            sb.write_u32(&vec![0x3d80_3c00u32; groups]);
            Ok((packed, sb))
        };
        let (p0, s0) = mk_bank(7)?;
        let (p1, s1) = mk_bank(11)?;
        let (p2, s2) = mk_bank(13)?;
        let bank = nn::Q4MlxBank {
            packed: &p0,
            scales_biases: &s0,
        };
        let bank1 = nn::Q4MlxBank {
            packed: &p1,
            scales_biases: &s1,
        };
        let bank2 = nn::Q4MlxBank {
            packed: &p2,
            scales_biases: &s2,
        };
        let shape = nn::QuantShape {
            rows: rows_w as u32,
            cols: cols_w as u32,
            group_size: group as u32,
        };
        let xf = buf(&rt, &fill(cols_w, 0x71));
        let xb = rt.alloc_buffer(cols_w * 2)?;
        xb.write_bf16_bits(&vec![0x3f00u16; cols_w]);
        let y = out_buf(&rt, rows_w);
        let y2 = out_buf(&rt, rows_w);
        let y3 = out_buf(&rt, rows_w);
        let resid = buf(&rt, &fill(rows_w, 0x72));
        let bytes = (rows_w * cols_w / 2 + groups * 4 + cols_w * 4 + rows_w * 4) as f64;

        for (label, variant) in [
            ("standard", nn::Q4MlxRowVariant::Standard),
            ("wide", nn::Q4MlxRowVariant::Wide),
            ("tiled", nn::Q4MlxRowVariant::Tiled),
        ] {
            verify(&rt, "gemv_q4_mlx", &y, rows_w, || {
                nn::gemv_q4_mlx(&rt, bank, &xf, &y, shape, variant)
            })?;
            rows.push(measure(
                &rt,
                &format!("gemv_q4_mlx [{label}]"),
                &format!("{rows_w}x{cols_w}"),
                bytes,
                warmup,
                iters,
                || nn::gemv_q4_mlx(&rt, bank, &xf, &y, shape, variant),
            )?);
        }

        verify(&rt, "gemv_q4_mlx_blocked", &y, rows_w, || {
            nn::gemv_q4_mlx_blocked(&rt, bank, &xf, &y, shape)
        })?;
        rows.push(measure(
            &rt,
            "gemv_q4_mlx_blocked",
            &format!("{rows_w}x{cols_w}"),
            bytes,
            warmup,
            iters,
            || nn::gemv_q4_mlx_blocked(&rt, bank, &xf, &y, shape),
        )?);

        for (label, layout) in [
            ("rowmajor", nn::Q4MlxLayout::RowMajor),
            ("i4", nn::Q4MlxLayout::Interleaved4),
        ] {
            for (rlabel, r) in [("", None), (" +resid", Some(&resid))] {
                verify(&rt, "gemv_q4_mlx_simd", &y, rows_w, || {
                    nn::gemv_q4_mlx_simd(&rt, bank, &xb, &y, shape, layout, r)
                })?;
                rows.push(measure(
                    &rt,
                    &format!("gemv_q4_mlx_simd [{label}{rlabel}]"),
                    &format!("{rows_w}x{cols_w}"),
                    bytes,
                    warmup,
                    iters,
                    || nn::gemv_q4_mlx_simd(&rt, bank, &xb, &y, shape, layout, r),
                )?);
            }
            // GEMM form: the same weights against M rows of activations.
            let m = 8u32;
            let ym = out_buf(&rt, rows_w * m as usize);
            let xm = rt.alloc_buffer(cols_w * m as usize * 2)?;
            xm.write_bf16_bits(&vec![0x3f00u16; cols_w * m as usize]);
            // The GEMM form's residual spans all M rows, not one. The wrapper
            // rejects the GEMV-sized buffer rather than reading past it.
            let resid_m = buf(&rt, &fill(rows_w * m as usize, 0x73));
            for (rlabel, r) in [("", None), (" +resid", Some(&resid_m))] {
                rows.push(measure(
                    &rt,
                    &format!("gemm_q4_mlx [{label}{rlabel}] m={m}"),
                    &format!("{rows_w}x{cols_w}"),
                    bytes * m as f64,
                    warmup,
                    iters,
                    || nn::gemm_q4_mlx(&rt, bank, &xm, &ym, shape, m, layout, r),
                )?);
            }
            // Fused K and V projections in one dispatch.
            rows.push(measure(
                &rt,
                &format!("gemv_q4_mlx_kv [{label}]"),
                &format!("{rows_w}x{cols_w}"),
                bytes * 2.0,
                warmup,
                iters,
                || nn::gemv_q4_mlx_kv(&rt, bank1, bank2, &xb, &y2, &y3, shape, layout),
            )?);
            // Fused Q, K and V.
            rows.push(measure(
                &rt,
                &format!("gemv_q4_mlx_qkv [{label}]"),
                &format!("{rows_w}x{cols_w}"),
                bytes * 3.0,
                warmup,
                iters,
                || {
                    nn::gemv_q4_mlx_qkv(
                        &rt,
                        bank,
                        bank1,
                        bank2,
                        &xb,
                        nn::QkvOutputs {
                            q_out: &y,
                            k_out: &y2,
                            v_out: &y3,
                        },
                        rows_w as u32,
                        rows_w as u32,
                        cols_w as u32,
                        group as u32,
                        layout,
                    )
                },
            )?);
        }

        // Gate/up MLP projection, both dispatch strategies.
        let mid = out_buf(&rt, rows_w);
        for (label, dispatch) in [
            ("simd", nn::GateUpDispatch::Simd(nn::Q4MlxLayout::RowMajor)),
            (
                "simd_i4",
                nn::GateUpDispatch::Simd(nn::Q4MlxLayout::Interleaved4),
            ),
            ("blocked", nn::GateUpDispatch::Blocked),
        ] {
            let x_for = if matches!(dispatch, nn::GateUpDispatch::Blocked) {
                &xf
            } else {
                &xb
            };
            rows.push(measure(
                &rt,
                &format!("gemv_q4_mlx_gate_up_gelu [{label}]"),
                &format!("{rows_w}x{cols_w}"),
                bytes * 2.0,
                warmup,
                iters,
                || {
                    nn::gemv_q4_mlx_gate_up_gelu(
                        &rt, bank1, bank2, x_for, &mid, shape, dispatch, false,
                    )
                },
            )?);
        }
    }

    // ------------------------------- decode-path and elementwise kernels ---
    // Individually small; all of them on the per-token critical path, and none
    // of them previously timed. `bench/kernel_coverage.py` is what surfaced
    // them -- a name-based scan had reported several of these as covered.
    {
        let (rows_n, dim) = (2048usize, 2048usize);
        let n = rows_n * dim;
        let xf = buf(&rt, &fill(n, 0x81));
        let upf = buf(&rt, &fill(n, 0x82));
        let w = buf(&rt, &fill(dim, 0x82));
        let outf = out_buf(&rt, n);
        // These kernels take f32 inputs and write bf16. Keep the output at its
        // exact half-width extent so the coverage benchmark cannot hide an
        // overrun in an oversized allocation, and never reuse it as an input:
        // doing so races the kernel's read and write streams.
        let obf = rt.alloc_buffer(n * 2)?;
        let bytes_n = (n * 4 * 2) as f64;

        rows.push(measure(
            &rt,
            "rms_norm_bf16",
            &format!("{rows_n}x{dim}"),
            bytes_n / 2.0,
            warmup,
            iters,
            || nn::rms_norm_bf16(&rt, &xf, &w, &obf, rows_n as u32, dim as u32, 1e-6),
        )?);
        // Residual add is in-place on `resid`, which is what makes it one pass.
        let resid = buf(&rt, &fill(n, 0x83));
        rows.push(measure(
            &rt,
            "rms_norm_residual_add_f32",
            &format!("{rows_n}x{dim}"),
            bytes_n * 1.5,
            warmup,
            iters,
            || {
                nn::rms_norm_residual_add_f32(
                    &rt,
                    &xf,
                    &w,
                    &resid,
                    rows_n as u32,
                    dim as u32,
                    1e-6,
                    1.0,
                )
            },
        )?);
        rows.push(measure(
            &rt,
            "mlp_gelu_tanh_bf16",
            &format!("{n}"),
            bytes_n / 2.0,
            warmup,
            iters,
            || nn::mlp_gelu_tanh_bf16(&rt, &xf, &upf, &obf, n as u32),
        )?);
        rows.push(measure(
            &rt,
            "scale_f32_inplace",
            &format!("{n}"),
            (n * 4 * 2) as f64,
            warmup,
            iters,
            || nn::scale_f32_inplace(&rt, &outf, 1.000_976_6, n as u32),
        )?);

        // --- KV cache stores. `dst_offset` is a device buffer because during
        // decode it changes every token and an ICB freezes its binds.
        let slot = 8192usize;
        let cache_elems = slot * 64;
        let src_k = buf(&rt, &fill(slot, 0x84));
        let src_v = buf(&rt, &fill(slot, 0x85));
        let dst_k = out_buf(&rt, cache_elems);
        let dst_v = out_buf(&rt, cache_elems);
        let off = rt.alloc_buffer(4)?;
        off.write_u32(&[0]);
        let kv_bytes = (slot * 4 * 2) as f64;
        rows.push(measure(
            &rt,
            "kv_store_timestep",
            &format!("{slot}"),
            kv_bytes,
            warmup,
            iters,
            || nn::kv_store_timestep(&rt, &src_k, &dst_k, &off, slot as u32, cache_elems as u32),
        )?);
        rows.push(measure(
            &rt,
            "kv_store_timestep_pair",
            &format!("{slot}"),
            kv_bytes * 2.0,
            warmup,
            iters,
            || {
                nn::kv_store_timestep_pair(
                    &rt,
                    &src_k,
                    &src_v,
                    &dst_k,
                    &dst_v,
                    &off,
                    slot as u32,
                    cache_elems as u32,
                )
            },
        )?);
        let filled = rt.alloc_buffer(4)?;
        filled.write_u32(&[64]);
        let start = rt.alloc_buffer(4)?;
        start.write_u32(&[0]);
        rows.push(measure(
            &rt,
            "kv_ring_densify",
            &format!("{slot}x64"),
            (slot * 64 * 4 * 2) as f64,
            warmup,
            iters,
            || nn::kv_ring_densify(&rt, &dst_k, &dst_v, &filled, &start, slot as u32, 64),
        )?);

        // --- Sampling tail. One token's logits over the vocabulary.
        let vocab = 262144usize;
        let logits = buf(&rt, &fill(vocab, 0x86));
        let cap = buf(&rt, &[30.0f32]);
        let tok = rt.alloc_buffer(4)?;
        let idx = rt.alloc_buffer(4 * nn::argmax_pass_groups(vocab as u32))?;
        let val = rt.alloc_buffer(4 * nn::argmax_pass_groups(vocab as u32))?;
        let vb = (vocab * 4) as f64;
        rows.push(measure(
            &rt,
            "softcap_logits",
            &format!("{vocab}"),
            vb * 2.0,
            warmup,
            iters,
            || nn::softcap_logits(&rt, &logits, &cap, vocab as u32),
        )?);
        rows.push(measure(
            &rt,
            "argmax_f32_pass",
            &format!("{vocab}"),
            vb,
            warmup,
            iters,
            || nn::argmax_f32_pass(&rt, &logits, &idx, &val, None, &cap, vocab as u32),
        )?);
        // `softcap_sample` reduces over a single 256-lane threadgroup and
        // rejects anything wider -- it is the small-vocabulary path, and
        // `softcap_argmax_one_pass` below is the full-vocabulary one.
        let small = 256usize;
        rows.push(measure(
            &rt,
            "softcap_sample",
            &format!("{small}"),
            (small * 4) as f64,
            warmup,
            iters,
            || nn::softcap_sample(&rt, &logits, &tok, &cap, small as u32),
        )?);
        rows.push(measure(
            &rt,
            "softcap_argmax_one_pass",
            &format!("{vocab}"),
            vb,
            warmup,
            iters,
            || nn::softcap_argmax_one_pass(&rt, &logits, &tok, &cap, vocab as u32),
        )?);

        // --- Reverse casts. The forward ones are already exercised by the
        // GEMM lanes; these are the paths back to f32.
        let tf = rt.alloc_tensor_f32(&[rows_n, dim])?;
        tf.buffer.write_f32(&fill(n, 0x87));
        let tbf = tessl::gemm::cast_f32_to_bf16(&tf)?;
        let th = tessl::gemm::cast_f32_to_f16(&tf)?;
        rows.push(measure(
            &rt,
            "cast_bf16_to_f32",
            &format!("{rows_n}x{dim}"),
            (n * 6) as f64,
            warmup,
            iters,
            || tessl::gemm::cast_bf16_to_f32(&tbf).map(|_| ()),
        )?);
        rows.push(measure(
            &rt,
            "cast_f16_to_f32",
            &format!("{rows_n}x{dim}"),
            (n * 6) as f64,
            warmup,
            iters,
            || tessl::gemm::cast_f16_to_f32(&th).map(|_| ()),
        )?);

        // --- Typed copies. One kernel per dtype, selected by the tensor's own
        // dtype rather than by a flag, so all three need their own lane.
        let dst_f = rt.alloc_tensor_f32(&[rows_n, dim])?;
        let dst_bf = rt.alloc_tensor_bf16(&[rows_n, dim])?;
        let dst_h = rt.alloc_tensor_f16(&[rows_n, dim])?;
        for (name, src, dst, w) in [
            ("copy_f32", &tf, &dst_f, 4usize),
            ("copy_bf16", &tbf, &dst_bf, 2),
            ("copy_f16", &th, &dst_h, 2),
        ] {
            rows.push(measure(
                &rt,
                name,
                &format!("{rows_n}x{dim}"),
                (n * w * 2) as f64,
                warmup,
                iters,
                || tessl::tensor::gpu_copy(src, dst),
            )?);
        }

        // --- Elementwise softcap on a tensor, distinct from the sampling-path
        // `softcap_logits` above: this one allocates its result.
        rows.push(measure(
            &rt,
            "softcap_f32",
            &format!("{rows_n}x{dim}"),
            (n * 8) as f64,
            warmup,
            iters,
            || tessl::ops::softcap_f32(&rt, &tf, 30.0).map(|_| ()),
        )?);

        // --- Embedding lookup, both quantization formats.
        let (vocab_e, hidden, grp) = (32768usize, 2048usize, 64usize);
        let egroups = vocab_e * (hidden / grp);
        let epacked = rt.alloc_buffer(vocab_e * hidden / 2)?;
        epacked.write_bytes(
            &(0..vocab_e * hidden / 2)
                .map(|i| (i % 251) as u8)
                .collect::<Vec<u8>>(),
        );
        let esc = buf(&rt, &vec![0.02f32; egroups]);
        let eze = buf(&rt, &vec![7.0f32; egroups]);
        let esb = rt.alloc_buffer(egroups * 4)?;
        esb.write_u32(&vec![0x3d80_3c00u32; egroups]);
        let n_tok = 64u32;
        let ids = rt.alloc_buffer(n_tok as usize * 4)?;
        ids.write_u32(&(0..n_tok).collect::<Vec<u32>>());
        let eout = out_buf(&rt, n_tok as usize * hidden);
        let ebytes = (n_tok as usize * hidden * 4 + n_tok as usize * hidden / 2) as f64;
        rows.push(measure(
            &rt,
            "embed_lookup_q4",
            &format!("{vocab_e}x{hidden}"),
            ebytes,
            warmup,
            iters,
            || {
                nn::embed_lookup_q4(
                    &rt,
                    nn::Q4Bank {
                        packed: &epacked,
                        scales: &esc,
                        zeros: &eze,
                    },
                    &ids,
                    &eout,
                    vocab_e as u32,
                    hidden as u32,
                    grp as u32,
                    n_tok,
                )
            },
        )?);
        rows.push(measure(
            &rt,
            "embed_lookup_q4_mlx",
            &format!("{vocab_e}x{hidden}"),
            ebytes,
            warmup,
            iters,
            || {
                nn::embed_lookup_q4_mlx(
                    &rt,
                    nn::Q4MlxBank {
                        packed: &epacked,
                        scales_biases: &esb,
                    },
                    &ids,
                    &eout,
                    vocab_e as u32,
                    hidden as u32,
                    grp as u32,
                    n_tok,
                )
            },
        )?);

        // --- Fused RMSNorm + QKV scale + RoPE, all three position variants.
        // The variant selects the kernel, so each needs its own lane.
        let (t, hq, hkv, hd) = (256u32, 32u32, 8u32, 128u32);
        let qn = (t * hq * hd) as usize;
        let kvn = (t * hkv * hd) as usize;
        let qb = buf(&rt, &fill(qn, 0x91));
        let kb = buf(&rt, &fill(kvn, 0x92));
        let vbuf = buf(&rt, &fill(kvn, 0x93));
        let qw = buf(&rt, &fill(hd as usize, 0x94));
        let kw = buf(&rt, &fill(hd as usize, 0x95));
        let vw = buf(&rt, &fill(hd as usize, 0x96));
        let posb = rt.alloc_buffer(4)?;
        posb.write_u32(&[0]);
        let kvcap = kvn * 8;
        let dk = out_buf(&rt, kvcap);
        let dv = out_buf(&rt, kvcap);
        let dofs = rt.alloc_buffer(4)?;
        dofs.write_u32(&[0]);
        let dims = nn::QkvRopeDims {
            t,
            heads_q: hq,
            heads_kv: hkv,
            head_dim: hd,
            rotary_dim: hd,
            theta: 10000.0,
            eps: 1e-6,
        };
        let qkv = nn::QkvBuffers {
            q: &qb,
            k: &kb,
            v: &vbuf,
            q_weight: &qw,
            k_weight: &kw,
            v_weight: &vw,
        };
        let rbytes = ((qn + 2 * kvn) * 4 * 2) as f64;
        for (name, variant, pos, store) in [
            (
                "rms_qkv_rope [posconst]",
                nn::QkvRopeVariant::PosConst,
                None,
                None,
            ),
            (
                "rms_qkv_rope [posbuf]",
                nn::QkvRopeVariant::PosBuffer,
                Some(&posb),
                None,
            ),
            (
                "rms_qkv_rope [posbuf+kvstore]",
                nn::QkvRopeVariant::PosBufferKvStore,
                Some(&posb),
                Some(nn::KvStoreTarget {
                    dst_k: &dk,
                    dst_v: &dv,
                    dst_offset: &dofs,
                    capacity: kvcap as u32,
                }),
            ),
        ] {
            rows.push(measure(
                &rt,
                name,
                &format!("T={t} Hq={hq}"),
                rbytes,
                warmup,
                iters,
                || nn::rms_qkv_rope(&rt, variant, qkv, dims, 0, pos, store, false),
            )?);
        }
    }

    println!(
        "{:<38} {:>16} {:>12} {:>12} {:>10}",
        "kernel", "shape", "batched us", "solo us", "GB/s"
    );
    for r in &rows {
        println!(
            "{:<38} {:>16} {:>12.3} {:>12.3} {:>10.1}",
            r.name, r.shape, r.batched_us, r.solo_us, r.gb_s
        );
    }

    let floor = median(rows.iter().map(|r| r.solo_us).collect())?;
    println!(
        "\nMedian solo dispatch: {floor:.1} us. Every kernel whose batched time \
         is below that is\nentirely inside the submit-and-wait floor when issued \
         alone."
    );
    Ok(())
}

fn main() -> Result<(), String> {
    // Wrapped rather than called before each `return`: --dump-parity exits
    // early and an error exits earlier still, and a trace that silently skips
    // those paths would under-report coverage exactly where it matters.
    let outcome = run();
    emit_kernel_trace();
    outcome
}
