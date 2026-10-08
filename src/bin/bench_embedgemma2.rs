//! End-to-end latency and memory of `tessl::embedgemma2` on the real checkpoint.
//!
//! ```text
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//!   cargo run --release --bin bench_embedgemma2
//! ```
//!
//! `BENCH_WORKLOADS` lists workloads separated by `,`. A workload is one
//! `encode` call: terms joined by `+`, each `BxT` (`B` sequences of `T`
//! tokens) or `BxLO-HI` (`B` sequences, the `i`-th of
//! `LO + (i * 2654435761) % (HI - LO + 1)` tokens). The uniform workloads run
//! one forward; the ragged ones exercise `encode`'s split into forwards
//! (`forwards` in the output says how many ran). The default:
//!
//! * uniform: `1x16`, `64x32`, `32x256`, `8x1024`, `2x4096`;
//! * ragged: `128x8-512` (short texts, more padded rows than one forward
//!   holds), `1x4096+63x32` (a long document beside short queries), and
//!   `1x6147+1x1658+6x10-42` (the shape of `tools/embedgemma2_ref`'s eight
//!   texts).
//!
//! `BENCH_ITERS` (default 10) and `BENCH_WARMUP` (default 3) are per
//! workload. Prints one JSON array on stdout, one object per workload:
//! `{workload, backend, ms_min, ms_median, iters, forwards, device_peak_mib,
//! peak_footprint_mib}`. `device_peak_mib` is the device's peak allocation
//! during that workload above the loaded model
//! ([`GpuRuntime::peak_allocated_bytes`]); `peak_footprint_mib` is the
//! process's lifetime peak physical footprint so far (what `/usr/bin/time -l`
//! reports as `peak memory footprint`), so only its last value covers the
//! whole run. `bench_paired` alternates this binary with the PyTorch lane
//! (`bench/embedgemma2_torch.py`, the same token ids) or with another build
//! of itself, round by round.
//!
//! What is timed: one `encode` call, from host token ids to host embeddings:
//! activation allocation, upload, the forwards, pooling, projection and
//! normalization, and the read back. Not timed: tokenization (both lanes take
//! ids) and the model load.
//!
//! Before anything is timed, every workload's embeddings must be finite and of
//! unit norm; the run aborts otherwise, so a forward that wrote nothing cannot
//! post the best time.

mod common;

use std::path::PathBuf;
use std::time::Instant;

use tessl::embedgemma2::{EmbedGemma2Config, EmbedGemma2Model};
use tessl::safetensors::SafeTensors;
use tessl::GpuRuntime;

use common::{env_usize, peak_footprint};

const DEFAULT_WORKLOADS: &str =
    "1x16,64x32,32x256,8x1024,2x4096,128x8-512,1x4096+63x32,1x6147+1x1658+6x10-42";

const MIB: f64 = (1u64 << 20) as f64;

/// The token ids every lane uses for sequence `seq` of a workload: BOS, then
/// a spread of ordinary ids, then EOS.
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

/// The sequence lengths of one workload (see the module docs for the grammar).
fn parse_workload(w: &str) -> Result<Vec<usize>, String> {
    let mut lens = Vec::new();
    for term in w.trim().split('+') {
        let (b, t) = term
            .trim()
            .split_once('x')
            .ok_or_else(|| format!("workload {w:?}: term {term:?} is not BxT or BxLO-HI"))?;
        let num = |s: &str| -> Result<usize, String> { s.parse().map_err(|e| format!("workload {w:?}: {s:?}: {e}")) };
        let b = num(b)?;
        let (lo, hi) = match t.split_once('-') {
            Some((lo, hi)) => (num(lo)?, num(hi)?),
            None => (num(t)?, num(t)?),
        };
        if b == 0 || lo < 2 || hi < lo {
            return Err(format!(
                "workload {w:?}: term {term:?} needs at least 1 sequence and 2 <= LO <= HI tokens"
            ));
        }
        lens.extend((0..b).map(|i| lo + (i * 2_654_435_761) % (hi - lo + 1)));
    }
    Ok(lens)
}

fn main() -> Result<(), String> {
    let snap = PathBuf::from(
        std::env::var("EMBEDGEMMA2_SNAPSHOT").map_err(|_| "EMBEDGEMMA2_SNAPSHOT must name the snapshot directory")?,
    );
    let spec = std::env::var("BENCH_WORKLOADS").unwrap_or_else(|_| DEFAULT_WORKLOADS.into());
    let workloads = spec
        .split(',')
        .map(|w| Ok((w.trim().to_string(), parse_workload(w)?)))
        .collect::<Result<Vec<_>, String>>()?;
    let iters = env_usize("BENCH_ITERS", 10, 1)?;
    let warmup = env_usize("BENCH_WARMUP", 3, 0)?;

    let cfg = EmbedGemma2Config::from_config_file(&snap.join("config.json"))?;
    let st = SafeTensors::open(&snap.join("model.safetensors"))?;
    let rt = GpuRuntime::new_inference()?;
    let model = EmbedGemma2Model::load(&rt, &st, "language_model.", cfg.clone())?;
    let dim = cfg.embedding_dim as usize;
    let loaded = rt.current_allocated_bytes();

    let mut rows = Vec::new();
    for (name, lens) in &workloads {
        let seqs: Vec<Vec<u32>> = lens.iter().enumerate().map(|(s, &t)| ids(s, t)).collect();
        let batch: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();

        rt.reset_peak_allocated_bytes();
        let out = model.encode(&batch, None, false)?;
        for (s, e) in out.embeddings.chunks(dim).enumerate() {
            let norm = e.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
            if !e.iter().all(|v| v.is_finite()) || (norm - 1.0).abs() > 1e-4 {
                return Err(format!(
                    "{name}: sequence {s} embedding is not a finite unit vector (norm {norm})"
                ));
            }
        }

        for _ in 0..warmup {
            model.encode(&batch, None, false)?;
        }
        let mut ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            model.encode(&batch, None, false)?;
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(f64::total_cmp);
        let device_peak = rt.peak_allocated_bytes().saturating_sub(loaded);
        rows.push(format!(
            "{{\"workload\":\"{name}\",\"backend\":\"tessl-f32\",\"ms_min\":{:.4},\"ms_median\":{:.4},\"iters\":{iters},\"forwards\":{},\"device_peak_mib\":{:.1},\"peak_footprint_mib\":{:.1}}}",
            ms[0],
            ms[ms.len() / 2],
            out.forwards,
            device_peak as f64 / MIB,
            peak_footprint()? as f64 / MIB,
        ));
    }
    println!("[{}]", rows.join(","));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_workload;

    #[test]
    fn workload_grammar() {
        assert_eq!(parse_workload("3x5").unwrap(), vec![5, 5, 5]);
        assert_eq!(parse_workload("1x9+2x4").unwrap(), vec![9, 4, 4]);
        // The i-th length is LO + (i * 2654435761) % (HI - LO + 1), which
        // bench/embedgemma2_torch.py repeats.
        assert_eq!(parse_workload("4x10-42").unwrap(), vec![10, 20, 30, 40]);
        for bad in ["0x5", "2x1", "2x9-3", "2", "x5", "2x5+"] {
            assert!(parse_workload(bad).is_err(), "{bad}");
        }
    }
}
