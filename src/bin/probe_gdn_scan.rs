//! Is the GDN chunk scan underfilling the GPU?
//!
//! ```text
//! cargo run --release --bin probe_gdn_scan            # T = 1024 and 8192
//! cargo run --release --bin probe_gdn_scan -- 2048
//! ```
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
    self, Cols, GdnChunkPhase, GdnDims, GdnGateLogits, GdnParams, GdnQkv, GdnScanSlice,
    GdnWorkspace, StateIn,
};
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
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
    qkv.buffer
        .write_f32(&fill(rows * qkv_ld as usize, 1, 1.0, 0.0));
    let gates = rt.alloc_tensor_f32(&[rows, gate_ld as usize])?;
    gates
        .buffer
        .write_f32(&fill(rows * gate_ld as usize, 2, 2.0, 0.0));
    let a_log = rt.alloc_tensor_f32(&[V_HEADS as usize])?;
    a_log.buffer.write_f32(&fill(V_HEADS as usize, 3, 0.5, 0.0));
    let dt_bias = rt.alloc_tensor_f32(&[V_HEADS as usize])?;
    dt_bias
        .buffer
        .write_f32(&fill(V_HEADS as usize, 4, 1.0, -3.0));
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ts = Vec::new();
    for arg in std::env::args().skip(1) {
        let t: u32 = arg
            .parse()
            .map_err(|_| format!("expected a token count, got {arg:?}"))?;
        if t == 0 {
            return Err("T must be positive".into());
        }
        ts.push(t);
    }
    if ts.is_empty() {
        ts = vec![1024, 8192];
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
