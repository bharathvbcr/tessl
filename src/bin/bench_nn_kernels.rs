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

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
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

    let b = median(batched);
    Ok(Row {
        name: name.to_string(),
        shape: shape.to_string(),
        batched_us: b,
        solo_us: median(solo),
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

fn main() -> Result<(), String> {
    let rt = GpuRuntime::new()?;
    let warmup: usize = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

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

    let floor = median(rows.iter().map(|r| r.solo_us).collect());
    println!(
        "\nMedian solo dispatch: {floor:.1} us. Every kernel whose batched time \
         is below that is\nentirely inside the submit-and-wait floor when issued \
         alone."
    );
    Ok(())
}
