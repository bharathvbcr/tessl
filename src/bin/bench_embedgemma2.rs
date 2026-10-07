//! End-to-end latency of `tessl::embedgemma2` on the real checkpoint.
//!
//! ```text
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//!   cargo run --release --bin bench_embedgemma2
//! ```
//!
//! `BENCH_WORKLOADS` (default `1x16,64x32,32x256,8x1024,2x4096`) lists
//! `sequences x tokens`; every sequence of a workload has the same length, so
//! no lane pads and both run one forward. `BENCH_ITERS` (default 10) and
//! `BENCH_WARMUP` (default 3) are per workload. Prints one JSON array on
//! stdout: `{workload, backend, ms_min, ms_median, iters, forwards}` per
//! workload. `bench/paired_embedgemma2.py` alternates this with the PyTorch
//! lane (`bench/embedgemma2_torch.py`, the same token ids) round by round.
//!
//! What is timed: one `encode` call, from host token ids to host embeddings:
//! activation allocation, upload, the forward, pooling, projection and
//! normalization, and the read back. Not timed: tokenization (both lanes take
//! ids) and the model load.
//!
//! Before anything is timed, every workload's embeddings must be finite and of
//! unit norm; the run aborts otherwise, so a forward that wrote nothing cannot
//! post the best time.

use std::path::PathBuf;
use std::time::Instant;

use tessl::embedgemma2::{EmbedGemma2Config, EmbedGemma2Model};
use tessl::safetensors::SafeTensors;
use tessl::GpuRuntime;

/// The token ids both lanes use: BOS, then a spread of ordinary ids, then EOS.
fn ids(seq: usize, len: usize) -> Vec<u32> {
    (0..len)
        .map(|i| {
            if i == 0 {
                2
            } else if i == len - 1 {
                1
            } else {
                (1000 + (seq * 7919 + i * 104_729) % 200_000) as u32
            }
        })
        .collect()
}

fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    match std::env::var(name) {
        Ok(v) => v.parse().map_err(|e| format!("{name}={v}: {e}")),
        Err(_) => Ok(default),
    }
}

fn parse_workloads(spec: &str) -> Result<Vec<(usize, usize)>, String> {
    spec.split(',')
        .map(|w| {
            let (b, t) = w.trim().split_once('x').ok_or_else(|| format!("workload {w:?}: expected BxT"))?;
            let (b, t): (usize, usize) = (
                b.parse().map_err(|e| format!("workload {w:?}: {e}"))?,
                t.parse().map_err(|e| format!("workload {w:?}: {e}"))?,
            );
            if b == 0 || t < 2 {
                return Err(format!("workload {w:?}: need at least 1 sequence of 2 tokens"));
            }
            Ok((b, t))
        })
        .collect()
}

fn main() -> Result<(), String> {
    let snap = PathBuf::from(
        std::env::var("EMBEDGEMMA2_SNAPSHOT").map_err(|_| "EMBEDGEMMA2_SNAPSHOT must name the snapshot directory")?,
    );
    let workloads = parse_workloads(
        &std::env::var("BENCH_WORKLOADS").unwrap_or_else(|_| "1x16,64x32,32x256,8x1024,2x4096".into()),
    )?;
    let iters = env_usize("BENCH_ITERS", 10)?.max(1);
    let warmup = env_usize("BENCH_WARMUP", 3)?;

    let cfg = EmbedGemma2Config::from_config_file(&snap.join("config.json"))?;
    let st = SafeTensors::open(&snap.join("model.safetensors"))?;
    let rt = GpuRuntime::new_inference()?;
    let model = EmbedGemma2Model::load(&rt, &st, "language_model.", cfg.clone())?;
    let dim = cfg.embedding_dim as usize;

    let mut rows = Vec::new();
    for &(b, t) in &workloads {
        let seqs: Vec<Vec<u32>> = (0..b).map(|s| ids(s, t)).collect();
        let batch: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();

        let out = model.encode(&batch, false)?;
        for (s, e) in out.embeddings.chunks(dim).enumerate() {
            let norm = e.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
            if !e.iter().all(|v| v.is_finite()) || (norm - 1.0).abs() > 1e-4 {
                return Err(format!("{b}x{t}: sequence {s} embedding is not a finite unit vector (norm {norm})"));
            }
        }

        for _ in 0..warmup {
            model.encode(&batch, false)?;
        }
        let mut ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            model.encode(&batch, false)?;
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(f64::total_cmp);
        rows.push(format!(
            "{{\"workload\":\"{b}x{t}\",\"backend\":\"tessl-f32\",\"ms_min\":{:.4},\"ms_median\":{:.4},\"iters\":{iters},\"forwards\":{}}}",
            ms[0],
            ms[ms.len() / 2],
            out.forwards
        ));
    }
    println!("[{}]", rows.join(","));
    Ok(())
}
