//! Effective bandwidth of `dequant_affine_{q8,q4}_bf16` against the device copy.
//!
//! The kernels are memory bound, so the figure of merit is effective GB/s,
//! (bytes read + bytes written) / time, next to what the same session's device
//! copy (`copy_bf16`, the blit every `gpu_copy` runs) achieves. The ratio is
//! the answer; neither number means much without the other, because the
//! machine is shared and its bandwidth moves with whatever else runs.
//!
//! * Each timed call is `BATCH` back-to-back dispatches in one command buffer
//!   (`set_async_encode(true)`) and one `synchronize()`, divided by `BATCH`, so
//!   the ~0.25 ms submit-and-wait floor measured in `docs/benchmarking.md` does
//!   not dominate a 0.1 ms kernel (the protocol of `bench_nn_kernels`' batched
//!   column).
//! * The arms are interleaved by `common::run_paired`: every round times each
//!   arm once, forward on even rounds and reversed on odd ones, after
//!   `BENCH_WARMUP` interleaved rounds that are discarded. The figure is the
//!   best of `BENCH_ITERS` rounds (default 30, the protocol's N >= 20); median,
//!   max and each arm's `self_spread` are printed beside it.
//! * Before timing, each kernel's whole output is compared against a host
//!   formula (a kernel that wrote nothing would otherwise win the table).
//!
//! Bytes: q8 reads `rows * cols` code bytes, q4 half that, both read 2 x 2
//! bytes of scale and bias per 64 columns, and write `2 * rows * cols`. The
//! copy arm at the same output size moves `2 * 2 * rows * cols`; a second copy
//! of 256 MiB is the streaming ceiling once the working set is out of the
//! system-level cache.

mod common;

use std::sync::Arc;

use tessl::dequant::{dequant_affine_bf16, AffineBits, DequantAffineBank};
use tessl::nn::QuantShape;
use tessl::tensor::{bf16_bits_to_f32, f32_to_bf16_bits, gpu_copy, GpuBuffer};
use tessl::GpuRuntime;

const BATCH: usize = 16;
const GROUP: usize = 64;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Finite bf16 with exponent in 100..=140: ordinary scales and biases.
    fn ordinary(&mut self) -> u16 {
        let r = self.next();
        (((r & 1) as u16) << 15) | ((100 + ((r >> 1) % 41) as u16) << 7) | (((r >> 8) & 0x7f) as u16)
    }
}

struct Problem {
    bits: AffineBits,
    rows: usize,
    cols: usize,
    packed: GpuBuffer,
    scales: GpuBuffer,
    biases: GpuBuffer,
    out: GpuBuffer,
    fault: GpuBuffer,
}

fn problem(rt: &Arc<GpuRuntime>, rng: &mut Rng, bits: AffineBits, rows: usize, cols: usize) -> Result<Problem, String> {
    let words = rows * cols * bits.bits() as usize / 32;
    let words_data: Vec<u32> = (0..words).map(|_| rng.next() as u32).collect();
    let groups = rows * cols / GROUP;
    let scales: Vec<u16> = (0..groups).map(|_| rng.ordinary()).collect();
    let biases: Vec<u16> = (0..groups).map(|_| rng.ordinary()).collect();
    let sb = rt.alloc_buffer(groups * 2)?;
    sb.write_bf16_bits(&scales);
    let bb = rt.alloc_buffer(groups * 2)?;
    bb.write_bf16_bits(&biases);
    let p = Problem {
        bits,
        rows,
        cols,
        packed: rt.alloc_buffer_from_u32(&words_data)?,
        scales: sb,
        biases: bb,
        out: rt.alloc_buffer(rows * cols * 2)?,
        fault: rt.alloc_buffer(4)?,
    };
    verify(rt, &p, &words_data, &scales, &biases)?;
    Ok(p)
}

fn shape(p: &Problem) -> QuantShape {
    QuantShape {
        rows: p.rows as u32,
        cols: p.cols as u32,
        group_size: GROUP as u32,
    }
}

fn call(rt: &Arc<GpuRuntime>, p: &Problem) -> Result<(), String> {
    let bank = DequantAffineBank {
        packed: &p.packed,
        scales: &p.scales,
        biases: &p.biases,
    };
    dequant_affine_bf16(rt, p.bits, bank, &p.out, &p.fault, shape(p))
}

/// Run once into a sentinel-filled output and compare every element with the
/// host formula (`fma` in f32, one bf16 rounding; inputs here are ordinary, so
/// no subnormal flush is involved).
fn verify(rt: &Arc<GpuRuntime>, p: &Problem, words: &[u32], scales: &[u16], biases: &[u16]) -> Result<(), String> {
    let n = p.rows * p.cols;
    p.out.write_bf16_bits(&vec![0xDEADu16; n]);
    p.fault.write_u32(&[0]);
    call(rt, p)?;
    rt.synchronize()?;
    if p.fault.read_u32()[0] != 0 {
        return Err(format!(
            "{:?} {}x{}: the fault word is raised on finite data; timing refused",
            p.bits, p.rows, p.cols
        ));
    }
    let got = p.out.try_contents_u16()?;
    let bits = p.bits.bits() as usize;
    let per = 32 / bits;
    let mask = (1u32 << bits) - 1;
    for (i, &g) in got[..n].iter().enumerate() {
        let q = (words[i / per] >> (bits * (i % per))) & mask;
        let (s, b) = (bf16_bits_to_f32(scales[i / GROUP]), bf16_bits_to_f32(biases[i / GROUP]));
        let want = f32_to_bf16_bits(s.mul_add(q as f32, b));
        if g != want {
            return Err(format!(
                "{:?} {}x{}: element {i} is {g:#06x}, host formula says {want:#06x}; timing refused",
                p.bits, p.rows, p.cols
            ));
        }
    }
    Ok(())
}

fn uptime_load() -> String {
    std::process::Command::new("uptime")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|e| format!("uptime failed: {e}"))
}

fn run() -> Result<(), String> {
    let warmup = common::env_usize("BENCH_WARMUP", 3, 0)?;
    let iters = common::env_usize("BENCH_ITERS", 30, 1)?;
    let rt = GpuRuntime::new()?;
    let mut rng = Rng(0xbe7c);

    println!("device: {}", rt.device_name());
    println!("machine: {}", uptime_load());
    println!("GPU utilisation before: {}%", common::gpu_utilization());
    if iters < 20 {
        println!("WARNING: BENCH_ITERS={iters} is below the protocol's 20; not a reportable run");
    }

    // The 2B's two big shapes: MLP gate/up (6144x2048) and down (2048x6144).
    let shapes = [(6144usize, 2048usize), (2048, 6144)];
    let mut problems = Vec::new();
    for bits in [AffineBits::Q8, AffineBits::Q4] {
        for (rows, cols) in shapes {
            problems.push(problem(&rt, &mut rng, bits, rows, cols)?);
        }
    }

    let n = shapes[0].0 * shapes[0].1;
    let copy_src = rt.alloc_tensor_bf16(&[n])?;
    let copy_dst = rt.alloc_tensor_bf16(&[n])?;
    copy_src.buffer.write_bf16_bits(&vec![0x3F80u16; n]);
    let big = 256 * 1024 * 1024 / 2;
    let big_src = rt.alloc_tensor_bf16(&[big])?;
    let big_dst = rt.alloc_tensor_bf16(&[big])?;
    big_src.buffer.write_bf16_bits(&vec![0x3F80u16; big]);

    // One timed call of an arm: BATCH dispatches in one command buffer, one wait.
    let batched = |encode: &dyn Fn() -> Result<(), String>| -> Result<(), String> {
        rt.set_async_encode(true)?;
        let encoded = (0..BATCH).try_for_each(|_| encode());
        let waited = rt.synchronize();
        rt.set_async_encode(false)?;
        encoded.and(waited)
    };

    let mut names = vec![
        format!("copy_bf16 {n} elems (same output size)"),
        "copy_bf16 256 MiB (streaming ceiling)".to_string(),
    ];
    let mut bytes = vec![(4 * n) as f64, (4 * big) as f64];
    let mut arms: Vec<common::Arm<'_>> = vec![
        common::Arm {
            name: names[0].clone(),
            reps: BATCH,
            run: Box::new(|| batched(&|| gpu_copy(&copy_src, &copy_dst))),
        },
        common::Arm {
            name: names[1].clone(),
            reps: BATCH,
            run: Box::new(|| batched(&|| gpu_copy(&big_src, &big_dst))),
        },
    ];
    for p in &problems {
        let elems = (p.rows * p.cols) as f64;
        let code_bytes = elems * p.bits.bits() as f64 / 8.0;
        let sb_bytes = 2.0 * 2.0 * elems / GROUP as f64;
        let name = format!("{} {}x{}", p.bits.entry(), p.rows, p.cols);
        names.push(name.clone());
        bytes.push(code_bytes + sb_bytes + 2.0 * elems);
        arms.push(common::Arm {
            name,
            reps: BATCH,
            run: Box::new({
                let (rt, batched) = (&rt, &batched);
                move || batched(&|| call(rt, p))
            }),
        });
    }

    // `warmup` interleaved rounds first, discarded; run_paired's own per-arm
    // warmup is the untimed first touch of each pipeline.
    let paired = common::run_paired(&mut arms, 1, 1, warmup + iters)?;
    println!("GPU utilisation after: {}%", common::gpu_utilization());
    println!(
        "{BATCH} dispatches per call, {warmup} discarded interleaved rounds, {iters} timed rounds, arms interleaved"
    );
    println!(
        "{:<44} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8}",
        "arm", "bytes MB", "min us", "med us", "max us", "GB/s best", "GB/s med", "spread"
    );
    let mut best = Vec::new();
    let mut typical = Vec::new();
    for (i, rounds) in paired.rounds.iter().enumerate() {
        let kept = &rounds[warmup..];
        let us: Vec<f64> = kept.iter().map(|ms| ms * 1e3).collect();
        let (lo, med, hi) = (
            common::min_of(&us),
            common::median_of(&us),
            us.iter().copied().fold(0.0, f64::max),
        );
        best.push(bytes[i] / (lo * 1e-6) / 1e9);
        typical.push(bytes[i] / (med * 1e-6) / 1e9);
        println!(
            "{:<44} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>7.0}%",
            names[i],
            bytes[i] / 1e6,
            lo,
            med,
            hi,
            best[i],
            typical[i],
            common::self_spread(kept) * 100.0
        );
    }
    println!("ratio to the same-size copy / to the 256 MiB copy:");
    for i in 2..names.len() {
        println!(
            "  {:<44} best-of-{iters} {:.2} / {:.2}   median {:.2} / {:.2}",
            names[i],
            best[i] / best[0],
            best[i] / best[1],
            typical[i] / typical[0],
            typical[i] / typical[1]
        );
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let outcome = run();
    common::emit_kernel_trace();
    outcome
}
