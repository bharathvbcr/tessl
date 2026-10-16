//! End-to-end latency of `tessl::embedgemma2_audio` on the real checkpoint.
//!
//! ```text
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//! EMBEDGEMMA2_MM_REF_DIR=/path/to/embedgemma2_mm_ref \
//!   cargo run --release --bin bench_embedgemma2_audio
//! ```
//!
//! `BENCH_WORKLOADS` lists workloads separated by `,` (default all):
//! `clip_1.0s` and `clip_3.3s` (the golden clips of the reference directory)
//! and `batch_8x30s` (eight synthetic clips of 3000 frames; see
//! `common/audio_workloads.rs`). `BENCH_ITERS` (default 10) and `BENCH_WARMUP`
//! (default 3) are per workload. The runtime queues dispatches
//! (`set_async_encode(true)`, which `BENCH_ASYNC=0` turns off) so that every
//! build of this binary, whichever way its `encode_audio` waits, times the
//! pipelined forward; each object carries `"async":0|1`.
//!
//! What is timed: one `encode_audio` call, from host features to host soft
//! tokens: allocation, upload, the forwards, the wait and the read back. Not
//! timed: the model load and the log-mel front end (the caller's).
//!
//! Prints one JSON array on stdout, one object per workload: `{workload,
//! ms_min, ms_median, iters, async, forwards, frames, soft_err_max,
//! device_peak_mib}`, the format `bench_paired` reads (`--format workloads`).
//! Before anything is timed the soft tokens must be finite and, for the golden
//! clips, within `1e-4` of their largest magnitude of transformers' (the bound
//! of `tests/embedgemma2_audio_model.rs`); `soft_err_max` is the observed
//! worst. The binary uses only the library's public `load` and `encode_audio`,
//! so it builds unchanged against older revisions of the crate.

mod common;

#[path = "common/audio_workloads.rs"]
mod audio_workloads;

use std::path::PathBuf;
use std::time::Instant;

use tessl::embedgemma2_audio::{AudioConfig, EmbedGemma2Audio};
use tessl::safetensors::SafeTensors;
use tessl::GpuRuntime;

use common::env_usize;

const MIB: f64 = (1u64 << 20) as f64;
/// The soft-token bound `tests/embedgemma2_audio_model.rs` states.
const SOFT_BOUND: f64 = 1e-4;

fn main() -> Result<(), String> {
    let snap = PathBuf::from(
        std::env::var("EMBEDGEMMA2_SNAPSHOT").map_err(|_| "EMBEDGEMMA2_SNAPSHOT must name the snapshot directory")?,
    );
    let workloads = audio_workloads::selected("BENCH_WORKLOADS")?;
    let iters = env_usize("BENCH_ITERS", 10, 1)?;
    let warmup = env_usize("BENCH_WARMUP", 3, 0)?;
    let async_encode = match std::env::var("BENCH_ASYNC").as_deref() {
        Err(_) | Ok("1") => true,
        Ok("0") => false,
        Ok(other) => return Err(format!("BENCH_ASYNC={other:?}: expected 0 or 1")),
    };

    let cfg = AudioConfig::from_config_file(&snap.join("config.json"))?;
    let st = SafeTensors::open(&snap.join("model.safetensors"))?;
    let rt = GpuRuntime::new_inference()?;
    rt.set_async_encode(async_encode)?;
    let model = EmbedGemma2Audio::load(&rt, &st, "", cfg)?;
    let loaded = rt.current_allocated_bytes();

    let mut rows = Vec::new();
    for w in &workloads {
        rt.reset_peak_allocated_bytes();
        let out = model.encode_audio(&w.clips, false)?;
        let err = audio_workloads::json_opt(audio_workloads::check(w, &out.soft_tokens, SOFT_BOUND)?);
        for _ in 0..warmup {
            model.encode_audio(&w.clips, false)?;
        }
        let mut ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            model.encode_audio(&w.clips, false)?;
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(f64::total_cmp);
        let device_peak = rt.peak_allocated_bytes().saturating_sub(loaded);
        rows.push(format!(
            "{{\"workload\":\"{}\",\"ms_min\":{:.4},\"ms_median\":{:.4},\"iters\":{iters},\"async\":{},\"forwards\":{},\"frames\":{},\"soft_err_max\":{err},\"device_peak_mib\":{:.1}}}",
            w.name,
            ms[0],
            ms[ms.len() / 2],
            u8::from(async_encode),
            out.forwards,
            w.frames(),
            device_peak as f64 / MIB,
        ));
    }
    println!("[{}]", rows.join(","));
    Ok(())
}
