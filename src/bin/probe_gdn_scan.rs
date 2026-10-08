//! Is the GDN chunk scan underfilling the GPU?
//!
//! ```text
//! cargo run --release --bin probe_gdn_scan            # T = 1024 and 8192
//! cargo run --release --bin probe_gdn_scan -- 2048
//! cargo run --release --bin probe_gdn_scan -- --paired          # T = 200 and 8192
//! cargo run --release --bin probe_gdn_scan -- --paired 200 8192
//! cargo run --release --bin probe_gdn_scan -- --paired --batch 2 61 200
//! ```
//!
//! `--paired` times one layer's scan at one batch (default 1), both widths, in this process:
//! one prep into one workspace, then the scan kernels ABBA (which width goes
//! first rotates each round) on those same buffers. It prints median and min,
//! and compares the two widths' output and final state bit for bit.
//!
//! `qwen35_gdn_chunk_scan` walks the chunks in order and launches one
//! threadgroup per (batch, value head, 32-column value slice): at Qwen3.5-2B's
//! shapes (16 value heads of 128) that is only 64 threadgroups at batch 1. This
//! times the scan alone (`qwen35::gdn_chunk_phase`, after one prep over the same
//! inputs) at batch 1, 2 and 4 with T fixed. Time that stays flat as the batch
//! grows means the extra threadgroups ran on otherwise idle cores, so the scan
//! is occupancy-bound and more parallelism per sequence would pay. Time that
//! grows linearly means it already fills the machine.
//!
//! Each B also runs the 16-column-slice scan (`GdnScanSlice::Cols16`), twice
//! the threadgroups for the same work; its last column is its time over the
//! 32-column scan's in the same session.
//!
//! Only the ratios are meaningful: the inputs are random, and a
//! contended GPU slows every B alike.

use std::sync::Arc;
use std::time::Instant;

use tessl::qwen35::{
    self, Cols, GdnChunkPhase, GdnDims, GdnGateLogits, GdnParams, GdnQkv, GdnScanSlice, GdnWorkspace, StateIn,
};
use tessl::tensor::GpuBuffer;
use tessl::GpuRuntime;

// Qwen3.5-2B's GDN shapes.
const K_HEADS: u32 = 16;
const V_HEADS: u32 = 16;
const KEY_DIM: u32 = 128;
const V_DIM: u32 = 128;
const REPS: usize = 8;
const WARMUP: usize = 2;
const ITERS: usize = 7;

type Res<T> = Result<T, String>;

fn fill(n: usize, seed: u64, scale: f32, shift: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 32) as u32) as f64 / u32::MAX as f64 * 2.0 - 1.0;
            u as f32 * scale + shift
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Median ms of one scan over `batch` sequences of `t` tokens.
fn scan_ms(rt: &Arc<GpuRuntime>, batch: u32, t: u32, slice: GdnScanSlice) -> Res<f64> {
    let dims = GdnDims {
        batch,
        seq: t,
        k_heads: K_HEADS,
        v_heads: V_HEADS,
        v_dim: V_DIM,
    };
    let rows = batch as usize * t as usize;
    let key_w = K_HEADS * KEY_DIM;
    let qkv_ld = 2 * key_w + V_HEADS * V_DIM;
    let gate_ld = 2 * V_HEADS;
    let out_w = V_HEADS * V_DIM;
    let qkv = rt.alloc_tensor_f32(&[rows, qkv_ld as usize])?;
    qkv.buffer.write_f32(&fill(rows * qkv_ld as usize, 1, 1.0, 0.0));
    let gates = rt.alloc_tensor_f32(&[rows, gate_ld as usize])?;
    gates.buffer.write_f32(&fill(rows * gate_ld as usize, 2, 2.0, 0.0));
    let a_log = rt.alloc_tensor_f32(&[V_HEADS as usize])?;
    a_log.buffer.write_f32(&fill(V_HEADS as usize, 3, 0.5, 0.0));
    let dt_bias = rt.alloc_tensor_f32(&[V_HEADS as usize])?;
    dt_bias.buffer.write_f32(&fill(V_HEADS as usize, 4, 1.0, -3.0));
    let out = rt.alloc_tensor_f32(&[rows, out_w as usize])?;
    let ws = GdnWorkspace::new(rt, &dims)?.with_scan_slice(slice);
    let q = GdnQkv {
        buf: &qkv.buffer,
        ld: qkv_ld,
        q_off: 0,
        k_off: key_w,
        v_off: 2 * key_w,
    };
    let g = GdnGateLogits {
        buf: &gates.buffer,
        ld: gate_ld,
        a_off: 0,
        b_off: V_HEADS,
    };
    let p = GdnParams {
        a_log: &a_log.buffer,
        dt_bias: &dt_bias.buffer,
    };
    let run = |phase| {
        qwen35::gdn_chunk_phase(
            rt,
            &dims,
            &q,
            &g,
            &p,
            StateIn::Zero,
            &ws,
            Cols::dense(&out.buffer, out_w),
            None,
            phase,
        )
    };
    run(GdnChunkPhase::Prep)?;
    rt.synchronize()?;
    for _ in 0..WARMUP {
        for _ in 0..REPS {
            run(GdnChunkPhase::Scan)?;
        }
        rt.synchronize()?;
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        for _ in 0..REPS {
            run(GdnChunkPhase::Scan)?;
        }
        rt.synchronize()?;
        samples.push(t0.elapsed().as_secs_f64() * 1e3 / REPS as f64);
    }
    // A scan that wrote nothing would post the best time of all.
    let o = out.buffer.read_f32();
    if !o[..rows * out_w as usize].iter().all(|x| x.is_finite()) {
        return Err(format!("B={batch} T={t}: scan output is not all finite"));
    }
    Ok(median(samples))
}

#[allow(clippy::too_many_arguments)]
fn phase(
    rt: &Arc<GpuRuntime>,
    dims: &GdnDims,
    q: &GdnQkv<'_>,
    gates: &GdnGateLogits<'_>,
    params: &GdnParams<'_>,
    ws: &mut GdnWorkspace,
    out: &GpuBuffer,
    out_w: u32,
    state_out: Option<&GpuBuffer>,
    slice: GdnScanSlice,
    which: GdnChunkPhase,
) -> Res<()> {
    ws.set_scan_slice(slice);
    qwen35::gdn_chunk_phase(
        rt,
        dims,
        q,
        gates,
        params,
        StateIn::Zero,
        ws,
        Cols::dense(out, out_w),
        state_out,
        which,
    )
}

fn poison(buf: &GpuBuffer) {
    buf.write_f32(&vec![f32::from_bits(0x7fc0_0001); buf.nbytes() / 4]);
}

/// One Qwen3.5-2B GDN layer, both scan widths, same buffers, ABBA.
///
/// Prep runs once into `ws`. Each timed sample is `REPS` scans in one command
/// buffer, so the submit-and-wait floor is not counted once per launch. The
/// bit check runs after the timing and covers the output and the final state.
fn run_paired(rt: &Arc<GpuRuntime>, batch: u32, ts: &[u32]) -> Res<()> {
    use std::io::Write;

    const WARMUP_ROUNDS: usize = 3;
    const ROUNDS: usize = 9;
    const LAYERS: f64 = 18.0;
    println!(
        "paired gdn chunk scan: Cols32 (qwen35_gdn_chunk_scan) vs Cols16 \
         (qwen35_gdn_chunk_scan_bv16)"
    );
    println!(
        "one process, one GDN layer, batch {batch}, {K_HEADS}k/{V_HEADS}v heads x {V_DIM}, \
         one workspace, ABBA (first width rotates each round)"
    );
    println!(
        "warmup rounds {WARMUP_ROUNDS}, timed rounds {ROUNDS}, {REPS} scans per command buffer; \
         x{LAYERS:.0} is 18 GDN layers in Qwen3.5-2B"
    );

    for &t in ts {
        let dims = GdnDims {
            batch,
            seq: t,
            k_heads: K_HEADS,
            v_heads: V_HEADS,
            v_dim: V_DIM,
        };
        let rows = batch as usize * t as usize;
        let key_w = K_HEADS * KEY_DIM;
        let qkv_ld = 2 * key_w + V_HEADS * V_DIM;
        let gate_ld = 2 * V_HEADS;
        let out_w = V_HEADS * V_DIM;
        let qkv = rt.alloc_tensor_f32(&[rows, qkv_ld as usize])?;
        qkv.buffer.write_f32(&fill(rows * qkv_ld as usize, 1, 1.0, 0.0));
        let gates = rt.alloc_tensor_f32(&[rows, gate_ld as usize])?;
        gates.buffer.write_f32(&fill(rows * gate_ld as usize, 2, 2.0, 0.0));
        let a_log = rt.alloc_tensor_f32(&[V_HEADS as usize])?;
        a_log.buffer.write_f32(&fill(V_HEADS as usize, 3, 0.5, 0.0));
        let dt_bias = rt.alloc_tensor_f32(&[V_HEADS as usize])?;
        dt_bias.buffer.write_f32(&fill(V_HEADS as usize, 4, 1.0, -3.0));
        let out = rt.alloc_tensor_f32(&[rows, out_w as usize])?;
        let state_elems = dims.state_elems_per_row() * batch as usize;
        let state = rt.alloc_buffer(state_elems * 4)?;
        let mut ws = GdnWorkspace::new(rt, &dims)?;
        let q = GdnQkv {
            buf: &qkv.buffer,
            ld: qkv_ld,
            q_off: 0,
            k_off: key_w,
            v_off: 2 * key_w,
        };
        let g = GdnGateLogits {
            buf: &gates.buffer,
            ld: gate_ld,
            a_off: 0,
            b_off: V_HEADS,
        };
        let p = GdnParams {
            a_log: &a_log.buffer,
            dt_bias: &dt_bias.buffer,
        };
        // Prep does not read the slice. One fill feeds both scans.
        phase(
            rt,
            &dims,
            &q,
            &g,
            &p,
            &mut ws,
            &out.buffer,
            out_w,
            None,
            GdnScanSlice::Cols32,
            GdnChunkPhase::Prep,
        )?;
        rt.synchronize()?;

        let mut wide = Vec::with_capacity(ROUNDS);
        let mut narrow = Vec::with_capacity(ROUNDS);
        for round in 0..(WARMUP_ROUNDS + ROUNDS) {
            let record = round >= WARMUP_ROUNDS;
            let order = if round % 2 == 0 {
                [GdnScanSlice::Cols32, GdnScanSlice::Cols16]
            } else {
                [GdnScanSlice::Cols16, GdnScanSlice::Cols32]
            };
            for slice in order {
                rt.synchronize()?;
                let t0 = Instant::now();
                for _ in 0..REPS {
                    phase(
                        rt,
                        &dims,
                        &q,
                        &g,
                        &p,
                        &mut ws,
                        &out.buffer,
                        out_w,
                        None,
                        slice,
                        GdnChunkPhase::Scan,
                    )?;
                }
                rt.synchronize()?;
                let ms = t0.elapsed().as_secs_f64() * 1e3 / REPS as f64;
                if record {
                    println!(
                        "sample B={batch} T={t} round={} {slice:?} {ms:.4} ms (buffer {:.2} ms, {REPS} launches)",
                        round - WARMUP_ROUNDS,
                        ms * REPS as f64
                    );
                    let _ = std::io::stdout().flush();
                    match slice {
                        GdnScanSlice::Cols32 => wide.push(ms),
                        GdnScanSlice::Cols16 => narrow.push(ms),
                    }
                }
            }
        }

        for (name, samples) in [("Cols32", &wide), ("Cols16", &narrow)] {
            let med = median(samples.clone());
            let lo = samples.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let wins = samples
                .iter()
                .zip(if name == "Cols32" { &narrow } else { &wide })
                .filter(|(a, b)| a < b)
                .count();
            println!(
                "PAIRED B={batch} t={t} slice={name} n={} reps={REPS} median_ms={med:.4} min_ms={lo:.4} max_ms={hi:.4} \
                 rounds_faster={wins} x18_median_ms={:.4} x18_min_ms={:.4}",
                samples.len(),
                med * LAYERS,
                lo * LAYERS
            );
        }
        let ratio = median(narrow.clone()) / median(wide.clone());
        println!("PAIRED B={batch} t={t} ratio_cols16_over_cols32={ratio:.4}");

        let mut capture = |slice: GdnScanSlice| -> Res<(Vec<u32>, Vec<u32>)> {
            poison(&out.buffer);
            poison(&state);
            phase(
                rt,
                &dims,
                &q,
                &g,
                &p,
                &mut ws,
                &out.buffer,
                out_w,
                Some(&state),
                slice,
                GdnChunkPhase::Scan,
            )?;
            rt.synchronize()?;
            let y = out.buffer.read_f32();
            let st = state.read_f32();
            let y_n = rows * out_w as usize;
            let st_n = state_elems;
            if y.len() < y_n || st.len() < st_n {
                return Err(format!(
                    "B={batch} T={t} {slice:?}: readback shorter than the logical output"
                ));
            }
            Ok((
                y[..y_n].iter().map(|x| x.to_bits()).collect(),
                st[..st_n].iter().map(|x| x.to_bits()).collect(),
            ))
        };
        let (y32, s32) = capture(GdnScanSlice::Cols32)?;
        let (y16, s16) = capture(GdnScanSlice::Cols16)?;
        let diff = |a: &[u32], b: &[u32]| a.iter().zip(b).filter(|(x, y)| x != y).count();
        let y_mis = diff(&y32, &y16);
        let s_mis = diff(&s32, &s16);
        println!(
            "BITS B={batch} t={t} out_elems={} state_elems={} out_mismatches={y_mis} state_mismatches={s_mis}",
            y32.len(),
            s32.len()
        );
        if y_mis != 0 || s_mis != 0 {
            let first = |a: &[u32], b: &[u32]| {
                a.iter()
                    .zip(b)
                    .position(|(x, y)| x != y)
                    .map(|i| format!("index {i}: {:08x} vs {:08x}", a[i], b[i]))
                    .unwrap_or_else(|| "length".into())
            };
            return Err(format!(
                "B={batch} T={t}: widths differ (out {y_mis}, state {s_mis}); first out {}, first state {}",
                first(&y32, &y16),
                first(&s32, &s16)
            ));
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut paired = false;
    let mut batch = 1u32;
    let mut ts = Vec::new();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--paired" {
            paired = true;
            i += 1;
            continue;
        }
        if arg == "--batch" {
            let raw = args.get(i + 1).ok_or("--batch needs a positive batch size")?;
            batch = raw
                .parse()
                .map_err(|_| format!("--batch needs a positive integer, got {raw:?}"))?;
            if batch == 0 {
                return Err("--batch must be positive".into());
            }
            i += 2;
            continue;
        }
        let t: u32 = arg
            .parse()
            .map_err(|_| format!("expected a token count, --paired, or --batch N, got {arg:?}"))?;
        if t == 0 {
            return Err("T must be positive".into());
        }
        ts.push(t);
        i += 1;
    }
    if ts.is_empty() {
        ts = if paired { vec![200, 8192] } else { vec![1024, 8192] };
    }
    if paired {
        let rt = GpuRuntime::new()?;
        rt.set_async_encode(true)?;
        println!("device: {}", rt.device_name());
        run_paired(&rt, batch, &ts)?;
        return Ok(());
    }
    let rt = GpuRuntime::new()?;
    rt.set_async_encode(true)?;
    println!("device: {}", rt.device_name());
    println!(
        "gdn chunk scan alone, {K_HEADS}k/{V_HEADS}v heads x {V_DIM}, median of {ITERS}, \
         {REPS} per command buffer"
    );
    println!(
        "{:>6} {:>3} {:>5} {:>7} {:>10} {:>12} {:>14}",
        "T", "B", "cols", "TGs", "ms", "ms / B=1", "ms / cols 32"
    );
    for &t in &ts {
        let mut base = None;
        for b in [1u32, 2, 4] {
            let mut wide = None;
            for (slice, cols) in [(GdnScanSlice::Cols32, 32u32), (GdnScanSlice::Cols16, 16)] {
                let ms = scan_ms(&rt, b, t, slice)?;
                let w = *wide.get_or_insert(ms);
                let tgs = b * V_HEADS * (V_DIM / cols);
                let rel_b = if cols == 32 {
                    format!("{:.2}", ms / *base.get_or_insert(ms))
                } else {
                    String::new()
                };
                println!(
                    "{t:>6} {b:>3} {cols:>5} {tgs:>7} {ms:>10.3} {rel_b:>12} {:>14.2}",
                    ms / w
                );
            }
        }
    }
    Ok(())
}
