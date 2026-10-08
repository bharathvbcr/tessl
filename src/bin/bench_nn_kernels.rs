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
//! value` line so `bench/paired_bins.sh` can take an interleaved min-of-N
//! across two frozen binaries. `--host-only` skips the kernel table.
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
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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

    let b = median(batched);
    Ok(Row {
        name: name.to_string(),
        shape: shape.to_string(),
        batched_us: b,
        host_us: median(host),
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
    let host = median(rows.iter().map(|r| r.host_us).collect());
    println!("\nMedian host encode per batched dispatch: {host:.2} us.");
    for r in &rows {
        // The bracketed GFLOP/s in a GEMM row's name changes run to run.
        let base = r.name.split(" [").next().unwrap_or(&r.name);
        let tag = r.name.find(" [").map(|i| &r.name[i..]).filter(|t| !t.contains("GFLOP"));
        let name = format!("{base}{}", tag.unwrap_or("")).replace([' ', '[', ']'], "");
        println!("METRIC host_us/{name}/{} {:.4}", r.shape, r.host_us);
    }
    println!("METRIC host_us/median {host:.4}");

    let floor = median(rows.iter().map(|r| r.solo_us).collect());
    println!(
        "\nMedian solo dispatch: {floor:.1} us. Every kernel whose batched time \
         is below that is\nentirely inside the submit-and-wait floor when issued \
         alone."
    );
    Ok(())
}

/// Qwen3.5's vocabulary: the row a decode sampler reduces each token.
const VOCAB: usize = 248_320;
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
    Ok(median(v))
}

/// One decode-style token loop: `encode` queues a token's work with async
/// encode on, `read` returns the token (and is where the wait belongs).
/// Returns median-over-iterations (encode us, wall us) per token.
fn sample_loop(
    rt: &Arc<GpuRuntime>,
    warmup: usize,
    iters: usize,
    mut encode: impl FnMut() -> Result<(), String>,
    mut read: impl FnMut() -> Result<u32, String>,
) -> Result<(f64, f64, u32), String> {
    rt.set_async_encode(true)?;
    let mut token = 0;
    for _ in 0..warmup {
        encode()?;
        token = read()?;
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
            if got != token {
                return Err(format!("sample loop: token changed from {token} to {got}"));
            }
            e += (t1 - t0).as_secs_f64();
            w += (t2 - t0).as_secs_f64();
        }
        enc.push(e * 1e6 / TOKENS as f64);
        wall.push(w * 1e6 / TOKENS as f64);
    }
    rt.set_async_encode(false)?;
    Ok((median(enc), median(wall), token))
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
    let logits = buf(rt, &fill(VOCAB, 0x61));
    let cap = buf(rt, &[30.0]);
    let hx = buf(rt, &fill(2048, 0x62));
    let hw = buf(rt, &fill(2048, 0x63));
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
