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
//! A third column, **host us**, is the batched arm's encode time alone: the
//! `BATCH` calls before the `synchronize()`, divided by `BATCH`. Nothing runs
//! on the GPU until that commit, so it is what the host pays per dispatch
//! (validation, pipeline lookup, binder setup, scratch allocation), not the
//! kernel.
//!
//! After the table, a **host path** section times the pieces of that cost
//! directly, and a **decode-style sample loop**: one sampler call per token,
//! encoded with async encode on and the token read back, as a GPU-resident
//! decode loop would. Every figure there is also printed as a `METRIC name
//! value` line so `bench_paired --format metrics` can take interleaved rounds
//! across two frozen binaries. `--host-only` skips the kernel table.
//!
//! Every kernel is checked for a plausible result before being timed. A kernel
//! that silently wrote nothing would otherwise post the best number in the
//! table.

mod common;

use std::sync::Arc;
use std::time::Instant;

use tessl::gemm::cast_f16_to_f32;
use tessl::tensor::GpuBuffer;
use tessl::{nn, softcap_f32, GpuRuntime};

const BATCH: usize = 64;

fn buf(rt: &Arc<GpuRuntime>, data: &[f32]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1) * 4).expect("alloc");
    b.write_f32(data);
    b
}

struct Row {
    name: String,
    shape: String,
    batched_us: f64,
    /// Encode time per dispatch in the batched arm, before its synchronize.
    host_us: f64,
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
    let mut host = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        for _ in 0..BATCH {
            f()?;
        }
        host.push(t0.elapsed().as_secs_f64() * 1e6 / BATCH as f64);
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

    let b = common::median(batched)?;
    Ok(Row {
        name: name.to_string(),
        shape: shape.to_string(),
        batched_us: b,
        host_us: common::median(host)?,
        solo_us: common::median(solo)?,
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
    // Wrapped so every exit path, an error included, prints the trace.
    let outcome = run();
    common::emit_kernel_trace();
    outcome
}

fn run() -> Result<(), String> {
    let warmup = common::env_usize("BENCH_WARMUP", 10, 0)?;
    let iters = common::env_usize("BENCH_ITERS", 50, 1)?;
    let rt = GpuRuntime::new()?;

    let mut host_only = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--host-only" => host_only = true,
            other => return Err(format!("unknown argument {other:?}; expected --host-only")),
        }
    }

    println!("device: {}", rt.device_name());
    if !host_only {
        kernel_table(&rt, warmup, iters)?;
    }
    host_path(&rt, warmup, iters)
}

fn kernel_table(rt: &Arc<GpuRuntime>, warmup: usize, iters: usize) -> Result<(), String> {
    let rt = rt.clone();
    println!(
        "batched = {BATCH} dispatches per command buffer; solo = 1 dispatch + \
         synchronize; host = batched encode before its synchronize\n"
    );

    let mut rows: Vec<Row> = Vec::new();

    // ---------------------------------------------------------- RMSNorm ---
    for &(r, d) in &[(1usize, 4096usize), (512, 4096), (2048, 4096)] {
        let x = buf(&rt, &common::fill(r * d, 0x11));
        let w = buf(&rt, &common::fill(d, 0x12));
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
        let g = buf(&rt, &common::fill(n, 0x21));
        let u = buf(&rt, &common::fill(n, 0x22));
        let o = out_buf(&rt, n);
        let bytes = (3 * n * 4) as f64;

        verify(&rt, "mlp_silu", &o, n, || nn::mlp_silu(&rt, &g, &u, &o, n as u32))?;
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
        let x = buf(&rt, &common::fill(r * c, 0x31));
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
        let xb = buf(&rt, &common::fill(c, 0x41));
        let yb = out_buf(&rt, r);

        verify(&rt, "gemv_q8", &yb, r, || {
            nn::gemv_q8(&rt, &pb, &sb, &zb, &xb, &yb, r as u32, c as u32, group as u32)
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
            || nn::gemv_q8(&rt, &pb, &sb, &zb, &xb, &yb, r as u32, c as u32, group as u32),
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
        let xb = buf(&rt, &common::fill(c, 0x51));
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
            let name = if tiled { "gemv_q4 [tiled]" } else { "gemv_q4 [row]" };
            rows.push(measure(&rt, name, &format!("{r}x{c}"), bytes, warmup, iters, || {
                nn::gemv_q4(&rt, bank, &xb, &yb, shape, tiled)
            })?);
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

    decode_lanes(&rt, &mut rows, warmup, iters)?;
    q4_mlx_lanes(&rt, &mut rows, warmup, iters)?;
    encoder_lanes(&rt, &mut rows, warmup, iters)?;
    util_lanes(&rt, &mut rows, warmup, iters)?;

    println!(
        "{:<38} {:>16} {:>12} {:>10} {:>12} {:>10}",
        "kernel", "shape", "batched us", "host us", "solo us", "GB/s"
    );
    for r in &rows {
        println!(
            "{:<38} {:>16} {:>12.3} {:>10.3} {:>12.3} {:>10.1}",
            r.name, r.shape, r.batched_us, r.host_us, r.solo_us, r.gb_s
        );
    }
    let host = common::median(rows.iter().map(|r| r.host_us).collect())?;
    println!("\nMedian host encode per batched dispatch: {host:.2} us.");
    for r in &rows {
        // The bracketed GFLOP/s in a GEMM row's name changes run to run.
        let base = r.name.split(" [").next().unwrap_or(&r.name);
        let tag = r.name.find(" [").map(|i| &r.name[i..]).filter(|t| !t.contains("GFLOP"));
        let name = format!("{base}{}", tag.unwrap_or("")).replace([' ', '[', ']'], "");
        println!("METRIC host_us/{name}/{} {:.4}", r.shape, r.host_us);
    }
    println!("METRIC host_us/median {host:.4}");

    let floor = common::median(rows.iter().map(|r| r.solo_us).collect())?;
    println!(
        "\nMedian solo dispatch: {floor:.1} us. Every kernel whose batched time \
         is below that is\nentirely inside the submit-and-wait floor when issued \
         alone."
    );
    Ok(())
}

// ======================================================================
// Coverage lanes: the public `nn::`, `bert::` and `embedgemma2::` ops a
// consumer calls that the table above did not time. The `kernel_coverage` bin
// reads which kernels this binary dispatched, so each lane is an entry point's
// reason to be in the metallib. Shapes are a model's own (a decode step's, a
// 4096 x 2048 projection, BERT-base's 768); correctness is checked by the
// tests, so a lane only asserts that its kernel wrote finite output.
// ======================================================================

/// Elements of `buf` read back as f32, which must all be finite.
fn assert_finite(what: &str, got: &[f32]) {
    assert!(got.iter().all(|v| v.is_finite()), "{what}: produced a non-finite value");
}

/// For an op that rewrites its operand: seed `buf` with `seed`, run the op once
/// and require finite output that differs from the seed somewhere (an op that
/// wrote nothing leaves the seed). `reset` is not needed between timed calls:
/// every in-place lane here is stable under repetition.
fn verify_inplace(
    rt: &Arc<GpuRuntime>,
    what: &str,
    buf_: &GpuBuffer,
    seed: &[f32],
    mut f: impl FnMut() -> Result<(), String>,
) -> Result<(), String> {
    buf_.write_f32(seed);
    f()?;
    rt.synchronize()?;
    let got = buf_.read_f32();
    assert_finite(what, &got);
    assert!(
        got.iter().zip(seed).any(|(g, s)| g != s),
        "{what}: the operand was not rewritten"
    );
    Ok(())
}

/// [`verify`] for a bf16 output: seeded with 0xFFFF (a NaN), so every element
/// must have been written, and written finite.
fn verify_bf16(
    rt: &Arc<GpuRuntime>,
    what: &str,
    out: &GpuBuffer,
    elems: usize,
    mut f: impl FnMut() -> Result<(), String>,
) -> Result<(), String> {
    out.write_bytes(&vec![0xFF; elems * 2]);
    f()?;
    rt.synchronize()?;
    let got = out.contents_u16().to_vec();
    let finite = |h: u16| h & 0x7F80 != 0x7F80;
    assert!(
        got[..elems].iter().all(|&h| finite(h)),
        "{what}: bf16 output has a NaN, Inf, or element that was never written"
    );
    Ok(())
}

/// Seed `out`, run, and require that the kernel wrote some elements and left
/// every element either unwritten or finite: for a kernel that writes only a
/// live part of a larger buffer.
fn verify_partial(
    rt: &Arc<GpuRuntime>,
    what: &str,
    out: &GpuBuffer,
    elems: usize,
    mut f: impl FnMut() -> Result<(), String>,
) -> Result<(), String> {
    out.write_f32(&vec![UNWRITTEN; elems]);
    f()?;
    rt.synchronize()?;
    let got = out.read_f32();
    assert_finite(what, &got);
    assert!(got.iter().any(|v| *v != UNWRITTEN), "{what}: wrote nothing");
    Ok(())
}

/// Time `f` into `rows` once its checker has passed.
#[allow(clippy::too_many_arguments)]
fn lane(
    rt: &Arc<GpuRuntime>,
    rows: &mut Vec<Row>,
    name: &str,
    shape: &str,
    bytes: f64,
    warmup: usize,
    iters: usize,
    f: impl FnMut() -> Result<(), String>,
) -> Result<(), String> {
    rows.push(measure(rt, name, shape, bytes, warmup, iters, f)?);
    Ok(())
}

fn u32_buf(rt: &Arc<GpuRuntime>, data: &[u32]) -> Result<GpuBuffer, String> {
    let b = rt.alloc_buffer(data.len().max(1) * 4)?;
    b.write_u32(data);
    Ok(b)
}

fn bf16_buf(rt: &Arc<GpuRuntime>, n: usize, seed: u64) -> Result<GpuBuffer, String> {
    let b = rt.alloc_buffer(n * 2)?;
    b.write_bf16_bits(&tessl::tensor::f32_slice_to_bf16(&common::fill(n, seed)));
    Ok(b)
}

/// An MLX Q4 bank of `rows x cols` weights in `group`-sized groups, stored
/// padded to a multiple of `tile` rows: random nibbles, and a finite bf16
/// (scale, bias) pair per group.
fn mlx_bank(
    rt: &Arc<GpuRuntime>,
    rows: usize,
    cols: usize,
    group: usize,
    tile: usize,
) -> Result<(GpuBuffer, GpuBuffer), String> {
    let stored = rows.next_multiple_of(tile);
    let packed = rt.alloc_buffer(stored * cols / 2)?;
    packed.write_bytes(&(0..stored * cols / 2).map(|i| (i % 251) as u8).collect::<Vec<u8>>());
    let groups = stored * (cols / group);
    let word =
        u32::from(tessl::tensor::f32_to_bf16_bits(0.02)) | (u32::from(tessl::tensor::f32_to_bf16_bits(-0.1)) << 16);
    let sb = u32_buf(rt, &vec![word; groups])?;
    Ok((packed, sb))
}

/// The decode step's elementwise, KV-cache, RoPE, sampler and attention ops.
fn decode_lanes(rt: &Arc<GpuRuntime>, rows: &mut Vec<Row>, warmup: usize, iters: usize) -> Result<(), String> {
    // ---- bf16-output norm and MLP gate, the residual-fused norm, scale ----
    for &(r, d) in &[(1usize, 4096usize), (512, 4096)] {
        let x = buf(rt, &common::fill(r * d, 0x71));
        let w = buf(rt, &common::fill(d, 0x72));
        let o = rt.alloc_buffer(r * d * 2)?;
        let run = || nn::rms_norm_bf16(rt, &x, &w, &o, r as u32, d as u32, 1e-6);
        verify_bf16(rt, "rms_norm_bf16", &o, r * d, run)?;
        lane(
            rt,
            rows,
            "rms_norm_bf16",
            &format!("{r}x{d}"),
            ((r * d * 6) + d * 4) as f64,
            warmup,
            iters,
            run,
        )?;

        let resid = buf(rt, &common::fill(r * d, 0x73));
        let seed = common::fill(r * d, 0x73);
        let run = || nn::rms_norm_residual_add_f32(rt, &x, &w, &resid, r as u32, d as u32, 1e-6, 0.01);
        verify_inplace(rt, "rms_norm_residual_add_f32", &resid, &seed, run)?;
        lane(
            rt,
            rows,
            "rms_norm_residual_add_f32",
            &format!("{r}x{d}"),
            ((3 * r * d + d) * 4) as f64,
            warmup,
            iters,
            run,
        )?;
    }
    let n = 1usize << 20;
    let g = buf(rt, &common::fill(n, 0x74));
    let u = buf(rt, &common::fill(n, 0x75));
    let o = rt.alloc_buffer(n * 2)?;
    let run = || nn::mlp_gelu_tanh_bf16(rt, &g, &u, &o, n as u32);
    verify_bf16(rt, "mlp_gelu_tanh_bf16", &o, n, run)?;
    lane(
        rt,
        rows,
        "mlp_gelu_tanh_bf16",
        &format!("n={n}"),
        (n * 10) as f64,
        warmup,
        iters,
        run,
    )?;

    let x = buf(rt, &common::fill(n, 0x76));
    let seed = common::fill(n, 0x76);
    let run = || nn::scale_f32_inplace(rt, &x, 0.999, n as u32);
    verify_inplace(rt, "scale_f32_inplace", &x, &seed, run)?;
    lane(
        rt,
        rows,
        "scale_f32_inplace",
        &format!("n={n}"),
        (n * 8) as f64,
        warmup,
        iters,
        run,
    )?;

    // ---- KV cache: one timestep in, a ring read back dense ----
    let (n_kv, cap) = (1024usize, 1024usize * 64);
    let src_k = buf(rt, &common::fill(n_kv, 0x77));
    let src_v = buf(rt, &common::fill(n_kv, 0x78));
    let dst_k = out_buf(rt, cap);
    let dst_v = out_buf(rt, cap);
    let at = 1024usize * 3;
    let off = u32_buf(rt, &[at as u32])?;
    let want_k = src_k.read_f32();
    let want_v = src_v.read_f32();
    dst_k.write_f32(&vec![UNWRITTEN; cap]);
    nn::kv_store_timestep(rt, &src_k, &dst_k, &off, n_kv as u32, cap as u32)?;
    rt.synchronize()?;
    assert_eq!(
        &dst_k.read_f32()[at..at + n_kv],
        &want_k[..],
        "kv_store_timestep: wrong timestep"
    );
    lane(
        rt,
        rows,
        "kv_store_timestep",
        &format!("n={n_kv}"),
        (n_kv * 8) as f64,
        warmup,
        iters,
        || nn::kv_store_timestep(rt, &src_k, &dst_k, &off, n_kv as u32, cap as u32),
    )?;
    dst_k.write_f32(&vec![UNWRITTEN; cap]);
    dst_v.write_f32(&vec![UNWRITTEN; cap]);
    nn::kv_store_timestep_pair(rt, &src_k, &src_v, &dst_k, &dst_v, &off, n_kv as u32, cap as u32)?;
    rt.synchronize()?;
    assert_eq!(
        &dst_k.read_f32()[at..at + n_kv],
        &want_k[..],
        "kv_store_timestep_pair: wrong K"
    );
    assert_eq!(
        &dst_v.read_f32()[at..at + n_kv],
        &want_v[..],
        "kv_store_timestep_pair: wrong V"
    );
    lane(
        rt,
        rows,
        "kv_store_timestep_pair",
        &format!("n={n_kv}"),
        (n_kv * 16) as f64,
        warmup,
        iters,
        || nn::kv_store_timestep_pair(rt, &src_k, &src_v, &dst_k, &dst_v, &off, n_kv as u32, cap as u32),
    )?;

    let (slot, ring) = (1024usize, 64usize);
    let ring_src = buf(rt, &common::fill(slot * ring, 0x79));
    let dense = out_buf(rt, slot * ring);
    let filled = u32_buf(rt, &[48])?;
    let start = u32_buf(rt, &[20])?;
    let run = || nn::kv_ring_densify(rt, &ring_src, &dense, &filled, &start, slot as u32, ring as u32);
    verify_partial(rt, "kv_ring_densify", &dense, slot * ring, run)?;
    lane(
        rt,
        rows,
        "kv_ring_densify",
        &format!("{ring}x{slot}"),
        (slot * 48 * 8) as f64,
        warmup,
        iters,
        run,
    )?;

    // ---- fused per-head RMSNorm + RoPE, the three entry points ----
    let dims = nn::QkvRopeDims {
        t: 1,
        heads_q: 8,
        heads_kv: 2,
        head_dim: 256,
        rotary_dim: 64,
        theta: 10_000.0,
        eps: 1e-6,
    };
    let (q_n, kv_n) = (8 * 256, 2 * 256);
    let q = buf(rt, &common::fill(q_n, 0x7A));
    let k = buf(rt, &common::fill(kv_n, 0x7B));
    let v = buf(rt, &common::fill(kv_n, 0x7C));
    let (qw, kw, vw) = (
        buf(rt, &common::fill(256, 0x7D)),
        buf(rt, &common::fill(256, 0x7E)),
        buf(rt, &common::fill(256, 0x7F)),
    );
    let pos = u32_buf(rt, &[5])?;
    let kcap = kv_n * 16;
    let (ck, cv) = (out_buf(rt, kcap), out_buf(rt, kcap));
    let kv_at = u32_buf(rt, &[kv_n as u32])?;
    let seed_q = common::fill(q_n, 0x7A);
    for (variant, name) in [
        (nn::QkvRopeVariant::PosConst, "rms_qkv_rope"),
        (nn::QkvRopeVariant::PosBuffer, "rms_qkv_rope_posbuf"),
        (nn::QkvRopeVariant::PosBufferKvStore, "rms_qkv_rope_kv_store"),
    ] {
        let run = || {
            nn::rms_qkv_rope(
                rt,
                variant,
                nn::QkvBuffers {
                    q: &q,
                    k: &k,
                    v: &v,
                    q_weight: &qw,
                    k_weight: &kw,
                    v_weight: &vw,
                },
                dims,
                5,
                (variant != nn::QkvRopeVariant::PosConst).then_some(&pos),
                (variant == nn::QkvRopeVariant::PosBufferKvStore).then_some(nn::KvStoreTarget {
                    dst_k: &ck,
                    dst_v: &cv,
                    dst_offset: &kv_at,
                    capacity: kcap as u32,
                }),
                false,
            )
        };
        verify_inplace(rt, name, &q, &seed_q, run)?;
        lane(
            rt,
            rows,
            name,
            "1 pos, 8q/2kv x 256",
            ((q_n + 2 * kv_n) * 8) as f64,
            warmup,
            iters,
            run,
        )?;
    }

    // ---- samplers over the vocabulary ----
    let logits = buf(rt, &common::fill(VOCAB, 0x81));
    let cap_buf = buf(rt, &[30.0]);
    let seed_logits = common::fill(VOCAB, 0x81);
    let run = || nn::softcap_logits(rt, &logits, &cap_buf, VOCAB as u32);
    verify_inplace(rt, "softcap_logits", &logits, &seed_logits, run)?;
    lane(
        rt,
        rows,
        "softcap_logits",
        &format!("n={VOCAB}"),
        (VOCAB * 8) as f64,
        warmup,
        iters,
        run,
    )?;
    logits.write_f32(&seed_logits);
    let tok = u32_buf(rt, &[u32::MAX])?;
    // One threadgroup reduces the whole row, so `n` is bounded by its lane count
    // (the full-vocabulary samplers are the multi-pass and one-pass argmax).
    // 256 is the largest valid n, so this lane sits on the boundary.
    const SAMPLE_N: usize = 256;
    let run = || nn::softcap_sample(rt, &logits, &tok, &cap_buf, SAMPLE_N as u32);
    run()?;
    rt.synchronize()?;
    let t = nn::check_argmax_result(&tok)?;
    assert!((t as usize) < SAMPLE_N, "softcap_sample: token {t} is outside the row");
    lane(
        rt,
        rows,
        "softcap_sample",
        &format!("n={SAMPLE_N}"),
        (SAMPLE_N * 4) as f64,
        warmup,
        iters,
        run,
    )?;

    // ---- D=64 row-parallel attention: the one head dim bench_flash_attn has no config for ----
    let (tq, h, hkv, d) = (128usize, 8usize, 2usize, 64usize);
    let dims = nn::AttnDims {
        batch: 1,
        tq: tq as u32,
        heads: h as u32,
        heads_kv: hkv as u32,
        window: 0,
        scale: 0.125,
    };
    let aq = buf(rt, &common::fill_affine(tq * h * d, 0x91, 0.125, 0.0));
    let ak = buf(rt, &common::fill_affine(tq * hkv * d, 0x92, 0.125, 0.0));
    let av = buf(rt, &common::fill(tq * hkv * d, 0x93));
    let ao = out_buf(rt, tq * h * d);
    let (tkv, zero) = (u32_buf(rt, &[tq as u32])?, u32_buf(rt, &[0])?);
    let run = || nn::flash_attn_rows(rt, &aq, &ak, &av, &ao, &tkv, &zero, &zero, dims, d as u32, false);
    verify(rt, "flash_attn_rows d=64", &ao, tq * h * d, run)?;
    lane(
        rt,
        rows,
        "flash_attn_rows",
        &format!("d={d} tq={tq}"),
        ((2 * tq * h + 2 * tq * hkv) * d * 4) as f64,
        warmup,
        iters,
        run,
    )?;
    Ok(())
}

/// MLX Q4 projections, the embedding lookups, and the Q4 GEMM.
fn q4_mlx_lanes(rt: &Arc<GpuRuntime>, rows: &mut Vec<Row>, warmup: usize, iters: usize) -> Result<(), String> {
    use nn::{GateUpDispatch, Q4MlxBank, Q4MlxLayout, Q4MlxRowVariant};
    let (r, c, group) = (4096usize, 2048usize, 64usize);
    let shape = nn::QuantShape {
        rows: r as u32,
        cols: c as u32,
        group_size: group as u32,
    };
    let x_f32 = buf(rt, &common::fill(c, 0xA1));
    let x_bf16 = bf16_buf(rt, c, 0xA1)?;
    let y = out_buf(rt, r);
    let weight_bytes = (r * c / 2 + r * (c / group) * 4 + c * 4 + r * 4) as f64;
    let sh = format!("{r}x{c}");

    // Row-major bank: the three one-thread-per-row variants and the simdgroup kernel.
    let (packed, sb) = mlx_bank(rt, r, c, group, 1)?;
    let bank = Q4MlxBank {
        packed: &packed,
        scales_biases: &sb,
    };
    for (variant, name) in [
        (Q4MlxRowVariant::Standard, "gemv_q4_mlx"),
        (Q4MlxRowVariant::Wide, "gemv_q4_mlx_wide"),
        (Q4MlxRowVariant::Tiled, "gemv_q4_mlx_tiled"),
    ] {
        let run = || nn::gemv_q4_mlx(rt, bank, &x_f32, &y, shape, variant);
        verify(rt, name, &y, r, run)?;
        lane(rt, rows, name, &sh, weight_bytes, warmup, iters, run)?;
    }
    let run = || nn::gemv_q4_mlx_simd(rt, bank, &x_bf16, &y, shape, Q4MlxLayout::RowMajor, None);
    verify(rt, "gemv_q4_mlx_simd", &y, r, run)?;
    lane(rt, rows, "gemv_q4_mlx_simd", &sh, weight_bytes, warmup, iters, run)?;
    // The residual-adding form accumulates into `y`, so it starts from finite values.
    let seed_y = common::fill(r, 0xA2);
    let run = || nn::gemv_q4_mlx_simd(rt, bank, &x_bf16, &y, shape, Q4MlxLayout::RowMajor, Some(&y));
    verify_inplace(rt, "gemv_q4_mlx_simd_add", &y, &seed_y, run)?;
    lane(rt, rows, "gemv_q4_mlx_simd_add", &sh, weight_bytes, warmup, iters, run)?;

    // Block-interleaved (16-row tile) bank for the blocked kernels, f32 activations.
    let (bpacked, bsb) = mlx_bank(rt, r, c, group, 16)?;
    let blocked = Q4MlxBank {
        packed: &bpacked,
        scales_biases: &bsb,
    };
    let run = || nn::gemv_q4_mlx_blocked(rt, blocked, &x_f32, &y, shape);
    verify(rt, "gemv_q4_mlx_blocked", &y, r, run)?;
    lane(rt, rows, "gemv_q4_mlx_blocked", &sh, weight_bytes, warmup, iters, run)?;

    // 4-row interleaved bank: the `_i4` entry points.
    let (ipacked, isb) = mlx_bank(rt, r, c, group, 4)?;
    let i4 = Q4MlxBank {
        packed: &ipacked,
        scales_biases: &isb,
    };
    let run = || nn::gemv_q4_mlx_simd(rt, i4, &x_bf16, &y, shape, Q4MlxLayout::Interleaved4, None);
    verify(rt, "gemv_q4_mlx_simd_i4", &y, r, run)?;
    lane(rt, rows, "gemv_q4_mlx_simd_i4", &sh, weight_bytes, warmup, iters, run)?;
    let run = || nn::gemv_q4_mlx_simd(rt, i4, &x_bf16, &y, shape, Q4MlxLayout::Interleaved4, Some(&y));
    verify_inplace(rt, "gemv_q4_mlx_simd_add_i4", &y, &seed_y, run)?;
    lane(
        rt,
        rows,
        "gemv_q4_mlx_simd_add_i4",
        &sh,
        weight_bytes,
        warmup,
        iters,
        run,
    )?;

    // Fused gate/up GELU: simdgroup (both layouts) and blocked.
    let mid = out_buf(rt, r);
    let (gpacked, gsb) = mlx_bank(rt, r, c, group, 1)?;
    let gate_rm = Q4MlxBank {
        packed: &gpacked,
        scales_biases: &gsb,
    };
    let (gi_packed, gi_sb) = mlx_bank(rt, r, c, group, 4)?;
    let gate_i4 = Q4MlxBank {
        packed: &gi_packed,
        scales_biases: &gi_sb,
    };
    let (bg_packed, bg_sb) = mlx_bank(rt, r, c, group, 16)?;
    let gate_blocked = Q4MlxBank {
        packed: &bg_packed,
        scales_biases: &bg_sb,
    };
    for (dispatch, gate, up, x, name) in [
        (
            GateUpDispatch::Simd(Q4MlxLayout::RowMajor),
            gate_rm,
            bank,
            &x_bf16,
            "gemv_q4_mlx_simd_gate_up_gelu",
        ),
        (
            GateUpDispatch::Simd(Q4MlxLayout::Interleaved4),
            gate_i4,
            i4,
            &x_bf16,
            "gemv_q4_mlx_simd_gate_up_gelu_i4",
        ),
        (
            GateUpDispatch::Blocked,
            gate_blocked,
            blocked,
            &x_f32,
            "gemv_q4_mlx_blocked_gate_up_gelu",
        ),
    ] {
        let run = || nn::gemv_q4_mlx_gate_up_gelu(rt, gate, up, x, &mid, shape, dispatch, false);
        verify(rt, name, &mid, r, run)?;
        lane(rt, rows, name, &sh, 2.0 * weight_bytes, warmup, iters, run)?;
    }

    // Fused K+V and Q+K+V projections, both layouts.
    let rows_kv = 1024usize;
    let kv_shape = nn::QuantShape {
        rows: rows_kv as u32,
        cols: c as u32,
        group_size: group as u32,
    };
    let (k_out, v_out) = (out_buf(rt, rows_kv), out_buf(rt, rows_kv));
    for (layout, tile, name) in [
        (Q4MlxLayout::RowMajor, 1usize, "gemv_q4_mlx_simd_kv"),
        (Q4MlxLayout::Interleaved4, 4, "gemv_q4_mlx_simd_kv_i4"),
    ] {
        let (kp, ks) = mlx_bank(rt, rows_kv, c, group, tile)?;
        let (vp, vs) = mlx_bank(rt, rows_kv, c, group, tile)?;
        let (kb, vb) = (
            Q4MlxBank {
                packed: &kp,
                scales_biases: &ks,
            },
            Q4MlxBank {
                packed: &vp,
                scales_biases: &vs,
            },
        );
        v_out.write_f32(&vec![UNWRITTEN; rows_kv]);
        let run = || nn::gemv_q4_mlx_kv(rt, kb, vb, &x_bf16, &k_out, &v_out, kv_shape, layout);
        verify(rt, name, &k_out, rows_kv, run)?;
        assert_wrote_everything(name, &v_out, rows_kv);
        lane(
            rt,
            rows,
            name,
            &format!("2x{rows_kv}x{c}"),
            weight_bytes / 2.0,
            warmup,
            iters,
            run,
        )?;

        let (qp, qs) = mlx_bank(rt, r, c, group, tile)?;
        let qb = Q4MlxBank {
            packed: &qp,
            scales_biases: &qs,
        };
        let name = if tile == 1 {
            "gemv_q4_mlx_simd_qkv"
        } else {
            "gemv_q4_mlx_simd_qkv_i4"
        };
        let q_out = out_buf(rt, r);
        let run = || {
            nn::gemv_q4_mlx_qkv(
                rt,
                qb,
                kb,
                vb,
                &x_bf16,
                nn::QkvOutputs {
                    q_out: &q_out,
                    k_out: &k_out,
                    v_out: &v_out,
                },
                r as u32,
                rows_kv as u32,
                c as u32,
                group as u32,
                layout,
            )
        };
        k_out.write_f32(&vec![UNWRITTEN; rows_kv]);
        v_out.write_f32(&vec![UNWRITTEN; rows_kv]);
        verify(rt, name, &q_out, r, run)?;
        assert_wrote_everything(name, &k_out, rows_kv);
        assert_wrote_everything(name, &v_out, rows_kv);
        lane(
            rt,
            rows,
            name,
            &format!("{r}+2x{rows_kv}x{c}"),
            weight_bytes * 1.5,
            warmup,
            iters,
            run,
        )?;
    }

    // The Q4 GEMM, a few activation rows at a time (a speculative-decode or tiny-batch step).
    let m = 4usize;
    let xm = bf16_buf(rt, m * c, 0xA3)?;
    let ym = out_buf(rt, m * r);
    let seed_ym = common::fill(m * r, 0xA4);
    for (layout, b_, tag) in [
        (Q4MlxLayout::RowMajor, bank, ""),
        (Q4MlxLayout::Interleaved4, i4, "_i4"),
    ] {
        let name = format!("gemm_q4_mlx_simd{tag}");
        let run = || nn::gemm_q4_mlx(rt, b_, &xm, &ym, shape, m as u32, layout, None);
        verify(rt, &name, &ym, m * r, run)?;
        lane(
            rt,
            rows,
            &name,
            &format!("m={m} {sh}"),
            weight_bytes,
            warmup,
            iters,
            run,
        )?;
        let name = format!("gemm_q4_mlx_simd_add{tag}");
        let run = || nn::gemm_q4_mlx(rt, b_, &xm, &ym, shape, m as u32, layout, Some(&ym));
        verify_inplace(rt, &name, &ym, &seed_ym, run)?;
        lane(
            rt,
            rows,
            &name,
            &format!("m={m} {sh}"),
            weight_bytes,
            warmup,
            iters,
            run,
        )?;
    }

    // Embedding lookups: the f32-scale/zero bank and the MLX bank.
    let (vocab, hidden, n_tok) = (32_768usize, 2048usize, 8usize);
    let ids = u32_buf(
        rt,
        &(0..n_tok as u32).map(|i| i * 4001 % vocab as u32).collect::<Vec<_>>(),
    )?;
    let out = out_buf(rt, n_tok * hidden);
    let gs = 64usize;
    let epacked = rt.alloc_buffer(vocab * hidden / 2)?;
    epacked.write_bytes(&(0..vocab * hidden / 2).map(|i| (i % 251) as u8).collect::<Vec<u8>>());
    let groups = vocab * (hidden / gs);
    let escales = buf(rt, &vec![0.02f32; groups]);
    let ezeros = buf(rt, &vec![7.0f32; groups]);
    let run = || {
        nn::embed_lookup_q4(
            rt,
            nn::Q4Bank {
                packed: &epacked,
                scales: &escales,
                zeros: &ezeros,
            },
            &ids,
            &out,
            vocab as u32,
            hidden as u32,
            gs as u32,
            n_tok as u32,
        )
    };
    verify(rt, "embed_lookup_q4", &out, n_tok * hidden, run)?;
    lane(
        rt,
        rows,
        "embed_lookup_q4",
        &format!("{n_tok}x{hidden}"),
        (n_tok * hidden * 5) as f64,
        warmup,
        iters,
        run,
    )?;
    let (mp, ms) = mlx_bank(rt, vocab, hidden, gs, 1)?;
    let run = || {
        nn::embed_lookup_q4_mlx(
            rt,
            Q4MlxBank {
                packed: &mp,
                scales_biases: &ms,
            },
            &ids,
            &out,
            vocab as u32,
            hidden as u32,
            gs as u32,
            n_tok as u32,
        )
    };
    verify(rt, "embed_lookup_q4_mlx", &out, n_tok * hidden, run)?;
    lane(
        rt,
        rows,
        "embed_lookup_q4_mlx",
        &format!("{n_tok}x{hidden}"),
        (n_tok * hidden * 5) as f64,
        warmup,
        iters,
        run,
    )
}

/// The encoder ops: BERT's layer norms, biases and sparse pooling, and
/// EmbedGemma2's attention, mean pooling and normalization.
fn encoder_lanes(rt: &Arc<GpuRuntime>, rows: &mut Vec<Row>, warmup: usize, iters: usize) -> Result<(), String> {
    use tessl::{bert, embedgemma2};
    let (batch, seq, dim) = (4usize, 128usize, 768usize);
    let n_rows = batch * seq;
    let eps = 1e-12;
    let (w, b) = (buf(rt, &common::fill(dim, 0xB1)), buf(rt, &common::fill(dim, 0xB2)));
    let bias = buf(rt, &common::fill_affine(dim, 0xB3, 0.001, 0.0));

    let (vocab, positions) = (30_522usize, 512usize);
    let ids = u32_buf(
        rt,
        &(0..n_rows as u32).map(|i| i * 7919 % vocab as u32).collect::<Vec<_>>(),
    )?;
    let (word, pos) = (
        buf(rt, &common::fill(vocab * dim, 0xB4)),
        buf(rt, &common::fill(positions * dim, 0xB5)),
    );
    let tables = bert::EmbedTables {
        word: &word,
        vocab: vocab as u32,
        pos: &pos,
        positions: positions as u32,
        type_row: &b,
        ln_w: &w,
        ln_b: &b,
    };
    let out = out_buf(rt, n_rows * dim);
    let run = || bert::embed_layer_norm(rt, &ids, tables, &out, n_rows as u32, dim as u32, seq as u32, eps);
    verify(rt, "bert_embed_layer_norm_f32", &out, n_rows * dim, run)?;
    let sh = format!("{n_rows}x{dim}");
    let act = (n_rows * dim * 8) as f64;
    lane(rt, rows, "bert_embed_layer_norm_f32", &sh, act, warmup, iters, run)?;

    let x = buf(rt, &common::fill(n_rows * dim, 0xB6));
    let run = || bert::layer_norm(rt, &x, &w, &b, &out, n_rows as u32, dim as u32, eps);
    verify(rt, "bert_layer_norm_f32", &out, n_rows * dim, run)?;
    lane(rt, rows, "bert_layer_norm_f32", &sh, act, warmup, iters, run)?;

    let seed = common::fill(n_rows * dim, 0xB7);
    let resid = buf(rt, &seed);
    let run = || bert::bias_residual_layer_norm(rt, &x, &bias, &resid, &w, &b, n_rows as u32, dim as u32, eps);
    verify_inplace(rt, "bert_bias_residual_layer_norm_f32", &resid, &seed, run)?;
    lane(
        rt,
        rows,
        "bert_bias_residual_layer_norm_f32",
        &sh,
        act * 1.5,
        warmup,
        iters,
        run,
    )?;

    let ffn = 4 * dim;
    let seed = common::fill(n_rows * ffn, 0xB8);
    let h = buf(rt, &seed);
    let fb = buf(rt, &common::fill_affine(ffn, 0xB9, 0.001, 0.0));
    let sh_ffn = format!("{n_rows}x{ffn}");
    let run = || bert::bias_add(rt, &h, &fb, n_rows as u32, ffn as u32);
    verify_inplace(rt, "bert_bias_add_f32", &h, &seed, run)?;
    lane(
        rt,
        rows,
        "bert_bias_add_f32",
        &sh_ffn,
        (n_rows * ffn * 8) as f64,
        warmup,
        iters,
        run,
    )?;
    let run = || bert::bias_gelu_erf(rt, &h, &fb, n_rows as u32, ffn as u32);
    verify_inplace(rt, "bert_bias_gelu_erf_f32", &h, &seed, run)?;
    lane(
        rt,
        rows,
        "bert_bias_gelu_erf_f32",
        &sh_ffn,
        (n_rows * ffn * 8) as f64,
        warmup,
        iters,
        run,
    )?;

    // Sparse pooling over a masked-LM head's logits: one segment per sequence.
    let segments: Vec<(u32, u32)> = (0..batch as u32)
        .map(|i| (i * seq as u32, (i + 1) * seq as u32))
        .collect();
    let seg = bert::upload_segments(rt, &segments, n_rows as u32)?;
    let logits = buf(rt, &common::fill(n_rows * vocab, 0xBA));
    let pbias = buf(rt, &vec![0.5f32; vocab]);
    let pooled = buf(rt, &vec![0.0f32; batch * vocab]);
    let run = || {
        bert::segment_sparse_max(
            rt,
            &logits,
            &pbias,
            &seg,
            &pooled,
            batch as u32,
            n_rows as u32,
            vocab as u32,
        )
    };
    verify_inplace(
        rt,
        "bert_segment_sparse_max_f32",
        &pooled,
        &vec![0.0f32; batch * vocab],
        run,
    )?;
    lane(
        rt,
        rows,
        "bert_segment_sparse_max_f32",
        &format!("{n_rows}x{vocab}"),
        (n_rows * vocab * 4) as f64,
        warmup,
        iters,
        run,
    )?;

    // EmbedGemma2 encoder attention: every instantiated head dim on the rows
    // kernel, and the TensorOps kernel where the device has it.
    let (ab, at, heads, heads_kv) = (2usize, 128usize, 8usize, 4usize);
    let lens = u32_buf(rt, &vec![at as u32; ab])?;
    for d in [32usize, 64, 256, 512] {
        let scale = 1.0 / (d as f32).sqrt();
        let q = buf(rt, &common::fill_affine(ab * at * heads * d, 0xC1, scale, 0.0));
        let k = buf(rt, &common::fill_affine(ab * at * heads_kv * d, 0xC2, scale, 0.0));
        let v = buf(rt, &common::fill_affine(ab * at * heads_kv * d, 0xC3, scale, 0.0));
        let o = out_buf(rt, ab * at * heads * d);
        let dims = embedgemma2::EncoderAttnDims {
            batch: ab as u32,
            seq: at as u32,
            heads: heads as u32,
            heads_kv: heads_kv as u32,
            head_dim: d as u32,
            window: if d == 256 { 64 } else { 0 },
            scale: 1.0,
        };
        let kernels: &[(embedgemma2::EncoderAttnKernel, &str)] = if d >= 256 && rt.has_tensorops() {
            &[
                (embedgemma2::EncoderAttnKernel::Rows, "rows"),
                (embedgemma2::EncoderAttnKernel::Tiled, "tiled"),
            ]
        } else {
            &[(embedgemma2::EncoderAttnKernel::Rows, "rows")]
        };
        for &(kernel, tag) in kernels {
            let name = format!("encoder_attn_{tag}_h{d}");
            let run = || embedgemma2::encoder_attn_with(rt, &q, &k, &v, &o, &lens, dims, kernel);
            verify(rt, &name, &o, ab * at * heads * d, run)?;
            lane(
                rt,
                rows,
                &name,
                &format!("{ab}x{at} h{heads}/{heads_kv}"),
                (ab * at * (2 * heads + 2 * heads_kv) * d * 4) as f64,
                warmup,
                iters,
                run,
            )?;
        }
    }

    // Mean pooling and L2 normalization of the encoder output.
    let (e_rows, e_dim, n_seg) = (2048usize, 1024usize, 8usize);
    let ex = buf(rt, &common::fill(e_rows * e_dim, 0xC4));
    let per = (e_rows / n_seg) as u32;
    let segs: Vec<(u32, u32)> = (0..n_seg as u32).map(|i| (i * per, (i + 1) * per)).collect();
    let eseg = embedgemma2::upload_segments(rt, &segs, e_rows as u32)?;
    let pooled = out_buf(rt, n_seg * e_dim);
    let run = || embedgemma2::segment_mean_rows(rt, &ex, &eseg, &pooled, n_seg as u32, e_rows as u32, e_dim as u32);
    verify(rt, "segment_mean_rows_f32", &pooled, n_seg * e_dim, run)?;
    lane(
        rt,
        rows,
        "segment_mean_rows_f32",
        &format!("{e_rows}x{e_dim}"),
        (e_rows * e_dim * 4) as f64,
        warmup,
        iters,
        run,
    )?;
    let seed = common::fill(512 * e_dim, 0xC5);
    let nx = buf(rt, &seed);
    let run = || embedgemma2::l2_normalize_rows(rt, &nx, 512, e_dim as u32, e_dim as u32);
    verify_inplace(rt, "l2_normalize_rows_f32", &nx, &seed, run)?;
    lane(
        rt,
        rows,
        "l2_normalize_rows_f32",
        &format!("512x{e_dim}"),
        (512 * e_dim * 8) as f64,
        warmup,
        iters,
        run,
    )
}

/// Qwen3.5's vocabulary: the row a decode sampler reduces each token.
const VOCAB: usize = 248_320;
/// The `utils.metal` elementwise ops a caller runs between the big kernels: the
/// logit softcap, the device blit in each storage dtype, and the f16 -> f32
/// widening cast. Each lane asserts its result is the exact value it should be
/// before it is timed.
fn util_lanes(rt: &Arc<GpuRuntime>, rows: &mut Vec<Row>, warmup: usize, iters: usize) -> Result<(), String> {
    use tessl::tensor::{f16_bits_to_f32, f32_slice_to_bf16, f32_to_f16_bits, gpu_copy, Tensor};
    let n = 1usize << 20;
    let sh = format!("n={n}");
    let vals = common::fill(n, 0x91);

    let pre = rt.alloc_tensor_f32(&[n])?;
    pre.buffer.write_f32(&vals);
    let post = softcap_f32(rt, &pre, 30.0)?;
    rt.synchronize()?;
    let got = post.buffer.read_f32();
    assert_finite("softcap_f32", &got);
    assert!(
        got.iter()
            .zip(&vals)
            .all(|(g, v)| (g - 30.0 * (v / 30.0).tanh()).abs() <= 1e-4),
        "softcap_f32: result is not softcap * tanh(x / softcap)"
    );
    lane(rt, rows, "softcap_f32", &sh, (2 * n * 4) as f64, warmup, iters, || {
        softcap_f32(rt, &pre, 30.0).map(|_| ())
    })?;

    let bf_bits = f32_slice_to_bf16(&vals);
    let f16_bits: Vec<u16> = vals.iter().map(|v| f32_to_f16_bits(*v)).collect();
    let src_bf = rt.alloc_tensor_bf16(&[n])?;
    src_bf.buffer.write_bf16_bits(&bf_bits);
    let src_h = rt.alloc_tensor_f16(&[n])?;
    src_h.buffer.write_f16_bits(&f16_bits);
    let blit = |name: &str, src: &Tensor, dst: &Tensor, bits: &[u16]| -> Result<(), String> {
        dst.buffer.write_bytes(&vec![0xFF; n * 2]);
        gpu_copy(src, dst)?;
        rt.synchronize()?;
        assert!(
            dst.buffer.contents_u16()[..n] == *bits,
            "{name}: the copy differs from its source"
        );
        Ok(())
    };
    let dst_bf = rt.alloc_tensor_bf16(&[n])?;
    blit("copy_bf16", &src_bf, &dst_bf, &bf_bits)?;
    lane(rt, rows, "copy_bf16", &sh, (2 * n * 2) as f64, warmup, iters, || {
        gpu_copy(&src_bf, &dst_bf)
    })?;
    let dst_h = rt.alloc_tensor_f16(&[n])?;
    blit("copy_f16", &src_h, &dst_h, &f16_bits)?;
    lane(rt, rows, "copy_f16", &sh, (2 * n * 2) as f64, warmup, iters, || {
        gpu_copy(&src_h, &dst_h)
    })?;

    let wide = cast_f16_to_f32(&src_h)?;
    rt.synchronize()?;
    let got = wide.buffer.read_f32();
    assert!(
        got.iter().zip(&f16_bits).all(|(g, h)| *g == f16_bits_to_f32(*h)),
        "cast_f16_to_f32: widening is not exact"
    );
    lane(rt, rows, "cast_f16_to_f32", &sh, (n * 6) as f64, warmup, iters, || {
        cast_f16_to_f32(&src_h).map(|_| ())
    })?;
    Ok(())
}

/// Tokens per timed sample-loop iteration.
const TOKENS: usize = 32;
/// Calls per timed iteration of a host-only microbenchmark.
const HOST_CALLS: usize = 20_000;

/// Median over `iters` of the mean nanoseconds per call of `f`, `HOST_CALLS`
/// calls an iteration.
fn ns_per_call(iters: usize, mut f: impl FnMut() -> Result<(), String>) -> Result<f64, String> {
    let mut v = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        for _ in 0..HOST_CALLS {
            f()?;
        }
        v.push(t0.elapsed().as_secs_f64() * 1e9 / HOST_CALLS as f64);
    }
    common::median(v)
}

/// One decode-style token loop: `encode` queues a token's work with async
/// encode on, `read` returns the token (and is where the wait belongs).
/// Returns median-over-iterations (encode us, wall us) per token.
fn sample_loop(
    rt: &Arc<GpuRuntime>,
    warmup: usize,
    iters: usize,
    encode: impl FnMut() -> Result<(), String>,
    read: impl FnMut() -> Result<u32, String>,
) -> Result<(f64, f64, u32), String> {
    rt.set_async_encode(true)?;
    let timed = timed_token_loop(warmup, iters, encode, read);
    // Restore before surfacing a loop error, so a failed lane cannot leave
    // async encode on for the next one.
    rt.set_async_encode(false)?;
    timed
}

/// The loop under [`sample_loop`], with no runtime in it. The token every read
/// must agree on is seeded by the first read, warmup or not: `BENCH_WARMUP=0`
/// (what the `kernel_coverage` bin passes) has no warmup read to seed it.
fn timed_token_loop(
    warmup: usize,
    iters: usize,
    mut encode: impl FnMut() -> Result<(), String>,
    mut read: impl FnMut() -> Result<u32, String>,
) -> Result<(f64, f64, u32), String> {
    let mut token: Option<u32> = None;
    for _ in 0..warmup {
        encode()?;
        token = Some(read()?);
    }
    let (mut enc, mut wall) = (Vec::with_capacity(iters), Vec::with_capacity(iters));
    for _ in 0..iters {
        let (mut e, mut w) = (0.0, 0.0);
        for _ in 0..TOKENS {
            let t0 = Instant::now();
            encode()?;
            let t1 = Instant::now();
            let got = read()?;
            let t2 = Instant::now();
            match token {
                None => token = Some(got),
                Some(expected) if expected != got => {
                    return Err(format!("sample loop: token changed from {expected} to {got}"));
                }
                Some(_) => {}
            }
            e += (t1 - t0).as_secs_f64();
            w += (t2 - t0).as_secs_f64();
        }
        enc.push(e * 1e6 / TOKENS as f64);
        wall.push(w * 1e6 / TOKENS as f64);
    }
    let token = token.ok_or("sample loop: no token was read (iters and TOKENS must be nonzero)")?;
    Ok((common::median(enc)?, common::median(wall)?, token))
}

/// The host side of a dispatch, timed apart from any kernel.
fn host_path(rt: &Arc<GpuRuntime>, warmup: usize, iters: usize) -> Result<(), String> {
    println!("\n-- host path --");

    // Pipeline lookup on a cache hit, both cache modes. The ICB mode builds
    // its own pipeline on first use, so warm it before timing.
    tessl::decode_icb::set_icb_pipelines(false);
    rt.pipeline("rms_norm_f32")?;
    let plain = ns_per_call(iters, || rt.pipeline("rms_norm_f32").map(drop))?;
    tessl::decode_icb::set_icb_pipelines(true);
    rt.pipeline("rms_norm_f32")?;
    let icb = ns_per_call(iters, || rt.pipeline("rms_norm_f32").map(drop))?;
    tessl::decode_icb::set_icb_pipelines(false);
    println!("pipeline hit:          {plain:>9.1} ns");
    println!("pipeline hit (ICB):    {icb:>9.1} ns");
    println!("METRIC pipeline_hit_ns {plain:.3}");
    println!("METRIC pipeline_hit_icb_ns {icb:.3}");

    // Decode-style sampling over a full vocabulary: a final-norm-sized
    // dispatch, then the sampler, then read the token.
    let logits = buf(rt, &common::fill(VOCAB, 0x61));
    let cap = buf(rt, &[30.0]);
    let hx = buf(rt, &common::fill(2048, 0x62));
    let hw = buf(rt, &common::fill(2048, 0x63));
    let ho = out_buf(rt, 2048);
    let norm = || nn::rms_norm_f32(rt, &hx, &hw, &ho, 1, 2048, 1e-6);
    let tok = rt.alloc_buffer(4)?;

    let (enc, wall, t) = sample_loop(
        rt,
        warmup,
        iters,
        || {
            norm()?;
            nn::softcap_argmax_one_pass(rt, &logits, &tok, &cap, VOCAB as u32)
        },
        || read_token(&tok),
    )?;
    println!("softcap_argmax_one_pass token loop: encode {enc:>8.2} us, wall {wall:>8.2} us per token (token {t})");
    println!("METRIC sample_one_pass_encode_us {enc:.4}");
    println!("METRIC sample_one_pass_wall_us {wall:.4}");

    // Multi-pass: V -> ceil(V/256) -> ... -> 1.
    let mut ns = vec![VOCAB as u32];
    while *ns.last().unwrap() > 1 {
        ns.push(nn::argmax_pass_groups(*ns.last().unwrap()) as u32);
    }
    let idx: Vec<GpuBuffer> = ns[1..]
        .iter()
        .map(|&g| rt.alloc_buffer(g as usize * 4))
        .collect::<Result<_, _>>()?;
    let val: Vec<GpuBuffer> = ns[1..]
        .iter()
        .map(|&g| rt.alloc_buffer(g as usize * 4))
        .collect::<Result<_, _>>()?;
    let passes = idx.len();
    let (enc, wall, t) = sample_loop(
        rt,
        warmup,
        iters,
        || {
            norm()?;
            for p in 0..passes {
                let (input, idx_in) = if p == 0 {
                    (&logits, None)
                } else {
                    (&val[p - 1], Some(&idx[p - 1]))
                };
                nn::argmax_f32_pass(rt, input, &idx[p], &val[p], idx_in, &cap, ns[p])?;
            }
            Ok(())
        },
        || read_token(&idx[passes - 1]),
    )?;
    println!("argmax_f32_pass x{passes} token loop:  encode {enc:>8.2} us, wall {wall:>8.2} us per token (token {t})");
    println!("METRIC sample_multi_pass_encode_us {enc:.4}");
    println!("METRIC sample_multi_pass_wall_us {wall:.4}");
    Ok(())
}

/// The token a sampler wrote, with its no-finite-logit refusal.
fn read_token(out: &GpuBuffer) -> Result<u32, String> {
    nn::check_argmax_result(out)
}

#[cfg(test)]
mod tests {
    use super::timed_token_loop;

    /// `BENCH_WARMUP=0` has no warmup read to seed the expected token, so the
    /// first timed read must. (It used to compare against a literal 0 and fail
    /// with "token changed from 0 to 117223".)
    #[test]
    fn token_loop_seeds_from_the_first_read_without_warmup() {
        let (_, _, token) = timed_token_loop(0, 2, || Ok(()), || Ok(117_223)).unwrap();
        assert_eq!(token, 117_223);
        let (_, _, token) = timed_token_loop(1, 1, || Ok(()), || Ok(5)).unwrap();
        assert_eq!(token, 5);
    }

    #[test]
    fn token_loop_still_rejects_a_token_that_changes() {
        let mut n = 0u32;
        let err = timed_token_loop(
            0,
            1,
            || Ok(()),
            || {
                n += 1;
                Ok(if n == 3 { 9 } else { 4 })
            },
        )
        .unwrap_err();
        assert!(err.contains("token changed from 4 to 9"), "{err}");
    }
}
