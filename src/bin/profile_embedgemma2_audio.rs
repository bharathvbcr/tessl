//! Where an `encode_audio` call's GPU time goes, per stage, on the real checkpoint.
//!
//! ```text
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//! EMBEDGEMMA2_MM_REF_DIR=/path/to/embedgemma2_mm_ref \
//!   cargo run --release --bin profile_embedgemma2_audio
//! ```
//!
//! The method is `profile_embedgemma2`'s, on the library's own forward instead
//! of a replica: the forward is `AudioStage`-tagged, and the model's
//! `set_stage_mask` runs only the stages whose bits are set. Per workload
//! (`PROFILE_WORKLOADS`, default `clip_1.0s,clip_3.3s,batch_8x30s`, see
//! `common/audio_workloads.rs`) it times, interleaved round by round (order
//! reversed on odd rounds) and min-of-`PROFILE_ITERS` per sample:
//!
//! * `encode`: `encode_audio` as shipped (unzeroed activations). It is also
//!   checked against transformers' soft tokens (golden clips) before anything
//!   is timed;
//! * `full`: every stage on, but with the profiler's zeroed activations: the
//!   baseline of the variants below, all of which allocate the same way;
//! * `none`: no dispatch at all (allocation, upload, one wait, read back): the
//!   harness floor;
//! * `only:<stage>`: one stage on. `only - none` is that stage's cost alone,
//!   with the real shapes and weights of all 12 layers;
//! * `skip:<stage>`: everything but that stage. `full - skip` is what removing
//!   it saves with the rest unchanged (its marginal cost in the pipelined
//!   forward), which is what fusing it away can win.
//!
//! The runtime queues dispatches (`set_async_encode(true)`), so a sample is
//! the host encoding the forward plus the GPU running it, ended by the wait;
//! at 25 rows the host is the limit and `only` of a stage then measures its
//! dispatch count, which the table shows beside it. `PROFILE_SKIP=0` leaves
//! out the `skip:` variants. `PROFILE_OPERANDS=f32|bf16` (default `f32`) is the
//! tower's operand precision ([`GemmOperands`]): the stage names are the same
//! in both, and under `bf16` `input_proj GEMM` and `output_proj GEMM` include
//! the one cast pass that rounds their operand. `PROFILE_JSON` names the output file. Other tessl
//! GPU processes seen before or after a sample flag it `contended`; the
//! statistics use every sample (min-of-N discards a contended one only when a
//! quiet one exists).

mod common;

#[path = "common/audio_workloads.rs"]
mod audio_workloads;

use std::path::PathBuf;
use std::time::Instant;

use tessl::embedgemma2_audio::{sub_rows, AudioConfig, AudioStage, EmbedGemma2Audio};
use tessl::gemm::GemmOperands;
use tessl::safetensors::SafeTensors;
use tessl::GpuRuntime;

use common::{env_usize, gpu_utilization, others_running, round_order};

const SOFT_BOUND: f64 = 1e-4;
/// A bf16 forward's soft tokens against transformers' fp32 ones (the floor of
/// `tests/embedgemma2_audio_model.rs`).
const SOFT_COS_FLOOR: f64 = 0.9997;

struct Variant {
    name: String,
    kind: &'static str,
    /// `None`: the shipped model, no stage mask (`encode`).
    mask: Option<u32>,
    /// Per round: the fastest of `iters` calls, in ms.
    rounds_ms: Vec<f64>,
    contended: usize,
    dispatches: usize,
}

fn min_of(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

fn median_of(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

fn apply(model: &EmbedGemma2Audio, mask: Option<u32>) {
    match mask {
        Some(m) => model.set_stage_mask(m),
        None => model.clear_stage_mask(),
    }
}

/// FLOPs of the GEMM stages for `rows` tower rows (2 per multiply-add); zero
/// for a stage that is not a GEMM.
fn gemm_flops(cfg: &AudioConfig, rows: usize, s: AudioStage) -> f64 {
    let (h, l, r) = (f64::from(cfg.hidden), f64::from(cfg.layers), rows as f64);
    let mm = |k: f64, n: f64| 2.0 * r * k * n;
    match s {
        AudioStage::InputProj => mm(f64::from(cfg.proj_in()), h),
        AudioStage::FfnUp | AudioStage::FfnDown => 2.0 * l * mm(h, 4.0 * h),
        AudioStage::QkvGemm => 3.0 * l * mm(h, h),
        AudioStage::AttnOutGemm | AudioStage::LconvEnd => l * mm(h, h),
        AudioStage::LconvStart => l * mm(h, 2.0 * h),
        AudioStage::OutProj => mm(h, f64::from(cfg.out_dims)),
        AudioStage::EmbedProj => mm(f64::from(cfg.out_dims), f64::from(cfg.embed_dim)),
        _ => 0.0,
    }
}

fn jnum(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.6}")
    } else {
        "null".into()
    }
}

fn jarr(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|&x| jnum(x)).collect::<Vec<_>>().join(","))
}

fn main() -> Result<(), String> {
    let snap = PathBuf::from(
        std::env::var("EMBEDGEMMA2_SNAPSHOT").map_err(|_| "EMBEDGEMMA2_SNAPSHOT must name the snapshot directory")?,
    );
    let workloads = audio_workloads::selected("PROFILE_WORKLOADS")?;
    let rounds = env_usize("PROFILE_ROUNDS", 5, 1)?;
    let operands = match std::env::var("PROFILE_OPERANDS").as_deref() {
        Err(_) | Ok("f32") => GemmOperands::ExactF32,
        Ok("bf16") => GemmOperands::Bf16,
        Ok(other) => return Err(format!("PROFILE_OPERANDS={other:?}: expected f32 or bf16")),
    };
    let with_skip = std::env::var("PROFILE_SKIP").as_deref() != Ok("0");
    let out_path = std::env::var("PROFILE_JSON").unwrap_or_else(|_| "target/profile_audio.json".into());

    let cfg = AudioConfig::from_config_file(&snap.join("config.json"))?;
    let st = SafeTensors::open(&snap.join("model.safetensors"))?;
    let rt = GpuRuntime::new_inference()?;
    rt.set_async_encode(true)?;
    let model = EmbedGemma2Audio::load_with(&rt, &st, "", cfg.clone(), operands)?;
    let util0 = gpu_utilization();
    eprintln!("gpu utilization at start: {util0}%");

    let mut json_wl = Vec::new();
    for w in &workloads {
        let rows: usize = w.clips.iter().map(|c| sub_rows(c.mask.len())).sum();
        let iters = env_usize(
            "PROFILE_ITERS",
            match w.frames() {
                0..=1000 => 12,
                1001..=5000 => 6,
                _ => 2,
            },
            1,
        )?;

        model.clear_stage_mask();
        let out = model.encode_audio(&w.clips, false)?;
        let err = match operands {
            GemmOperands::ExactF32 => {
                audio_workloads::json_opt(audio_workloads::check(w, &out.soft_tokens, SOFT_BOUND)?)
            }
            GemmOperands::Bf16 => audio_workloads::json_opt(
                audio_workloads::check_cosine(w, &out.soft_tokens, SOFT_COS_FLOOR)?.map(|(e, _)| e),
            ),
        };
        eprintln!(
            "{}: {} frames, {rows} tower rows, {} forward(s); soft tokens within {err} of transformers (null: no reference; relative to its largest magnitude)",
            w.name,
            w.frames(),
            out.forwards
        );

        let mut variants: Vec<Variant> = Vec::new();
        let mut add = |name: String, kind: &'static str, mask: Option<u32>| {
            variants.push(Variant {
                name,
                kind,
                mask,
                rounds_ms: Vec::new(),
                contended: 0,
                dispatches: 0,
            });
        };
        add("encode".into(), "encode", None);
        add("full".into(), "full", Some(AudioStage::MASK_ALL));
        add("none".into(), "none", Some(0));
        for s in AudioStage::ALL {
            add(format!("only:{}", s.name()), "only", Some(s.bit()));
        }
        if with_skip {
            for s in AudioStage::ALL {
                add(
                    format!("skip:{}", s.name()),
                    "skip",
                    Some(AudioStage::MASK_ALL & !s.bit()),
                );
            }
        }

        // Dispatch counts, and a warm-up of every variant (pipelines, pool).
        for v in &mut variants {
            apply(&model, v.mask);
            model.encode_audio(&w.clips, false)?;
            rt.take_dispatch_count();
            model.encode_audio(&w.clips, false)?;
            v.dispatches = rt.take_dispatch_count();
        }

        let util_a = gpu_utilization();
        for round in 0..rounds {
            for vi in round_order(round, variants.len()) {
                let v = &mut variants[vi];
                apply(&model, v.mask);
                let busy_before = !others_running().is_empty();
                let mut best = f64::INFINITY;
                for _ in 0..iters {
                    let t0 = Instant::now();
                    model.encode_audio(&w.clips, false)?;
                    best = best.min(t0.elapsed().as_secs_f64() * 1e3);
                }
                if busy_before || !others_running().is_empty() {
                    v.contended += 1;
                }
                v.rounds_ms.push(best);
            }
            eprintln!("{}: round {}/{rounds} done", w.name, round + 1);
        }
        model.clear_stage_mask();
        let util_b = gpu_utilization();

        // Report.
        let get = |name: &str| variants.iter().find(|v| v.name == name).expect("variant exists");
        let full = min_of(&get("full").rounds_ms);
        let none = min_of(&get("none").rounds_ms);
        let enc = min_of(&get("encode").rounds_ms);
        let spread =
            |v: &Variant| (v.rounds_ms.iter().cloned().fold(0.0, f64::max) / min_of(&v.rounds_ms) - 1.0) * 100.0;
        println!("\n== {} ({} frames, {rows} tower rows) ==", w.name, w.frames());
        println!(
            "encode (shipped) {enc:8.3} ms (median over rounds {:.3}, round spread {:.0} %); full (zeroed acts) {full:8.3} ms; floor (none) {none:8.3} ms; gpu util {util_a}% -> {util_b}%",
            median_of(&get("encode").rounds_ms),
            spread(get("encode")),
        );
        println!(
            "{:<32} {:>6} {:>10} {:>7} {:>10} {:>7} {:>9}",
            "stage", "disp", "alone ms", "share", "marg ms", "share", "TFLOP/s"
        );
        let mut rows_out = Vec::new();
        let mut stage_json = Vec::new();
        for s in AudioStage::ALL {
            let only = get(&format!("only:{}", s.name()));
            let alone = (min_of(&only.rounds_ms) - none).max(0.0);
            let marg = if with_skip {
                (full - min_of(&get(&format!("skip:{}", s.name())).rounds_ms)).max(0.0)
            } else {
                f64::NAN
            };
            let flops = gemm_flops(&cfg, rows, s);
            let tflops = if flops > 0.0 && alone > 0.0 {
                flops / (alone * 1e-3) / 1e12
            } else {
                f64::NAN
            };
            rows_out.push((s, only.dispatches, alone, marg, tflops));
        }
        let sum_alone: f64 = rows_out.iter().map(|r| r.2).sum();
        let sum_marg: f64 = rows_out.iter().filter(|r| r.3.is_finite()).map(|r| r.3).sum();
        rows_out.sort_by(|a, b| b.2.total_cmp(&a.2));
        for (s, disp, alone, marg, tflops) in &rows_out {
            println!(
                "{:<32} {:>6} {:>10.3} {:>6.1}% {:>10.3} {:>6.1}% {:>9}",
                s.name(),
                disp,
                alone,
                100.0 * alone / sum_alone,
                marg,
                100.0 * marg / sum_marg,
                if tflops.is_finite() {
                    format!("{tflops:.2}")
                } else {
                    "-".into()
                },
            );
            stage_json.push(format!(
                "{{\"stage\":\"{}\",\"dispatches\":{disp},\"alone_ms\":{},\"marginal_ms\":{},\"tflops\":{}}}",
                s.name(),
                jnum(*alone),
                jnum(*marg),
                jnum(*tflops)
            ));
        }
        println!(
            "sum of stages alone {sum_alone:.3} ms; sum of marginals {sum_marg:.3} ms; full - none {:.3} ms; total dispatches per forward set: {}",
            full - none,
            get("full").dispatches
        );
        let gemm_alone: f64 = rows_out.iter().filter(|r| r.4.is_finite()).map(|r| r.2).sum();
        println!(
            "GEMM stages alone: {gemm_alone:.3} ms ({:.1} % of the stage sum)",
            100.0 * gemm_alone / sum_alone
        );

        let var_json: Vec<String> = variants
            .iter()
            .map(|v| {
                format!(
                    "{{\"name\":{:?},\"kind\":\"{}\",\"dispatches\":{},\"contended_rounds\":{},\"rounds_ms\":{}}}",
                    v.name,
                    v.kind,
                    v.dispatches,
                    v.contended,
                    jarr(&v.rounds_ms)
                )
            })
            .collect();
        json_wl.push(format!(
            "{{\"workload\":\"{}\",\"frames\":{},\"rows\":{rows},\"iters\":{iters},\"rounds\":{rounds},\"precision\":\"{}\",\"soft_err\":{err},\"gpu_util_start\":\"{util_a}\",\"gpu_util_end\":\"{util_b}\",\"stages\":[{}],\"variants\":[{}]}}",
            w.name,
            w.frames(),
            if operands == GemmOperands::Bf16 { "bf16" } else { "f32" },
            stage_json.join(","),
            var_json.join(",")
        ));
    }
    let body = format!("{{\"workloads\":[{}]}}\n", json_wl.join(","));
    common::atomic_write(std::path::Path::new(&out_path), body.as_bytes())?;
    eprintln!("wrote {out_path}");
    Ok(())
}
