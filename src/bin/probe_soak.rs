//! Leak soak: a long loop on the committed tiny Qwen3.5 fixture, for `leaks`.
//!
//! ```text
//! cargo build --release --bin probe_soak
//! MallocStackLogging=1 leaks --atExit -- target/release/probe_soak --train=4000
//! MallocStackLogging=1 leaks --atExit -- target/release/probe_soak --decode=20000
//! ```
//!
//! `--train=N` runs N steps of `train_step_into` into one resident bank, each
//! followed by an AdamW step (a waited commit), on the fixture's 70-token
//! sequence. `--decode=N` runs sessions of `--session=M` tokens (default 512):
//! one prefill and M `Decode::step`s, the session dropped, ceil(N / M) times
//! (every session is the same size, so N is rounded up to a whole session),
//! so allocation and release of a whole session recurs too.
//!
//! Prints, at the start, every `--every=K` iterations (default N/8) and at the
//! end, one line: the process resident set (`ps`), the device's
//! `currentAllocatedSize` ([`GpuRuntime::current_allocated_bytes`]) and the
//! loss or the last logits' checksum. Exits non-zero if the device allocation
//! at the end is above its value after the first iteration (the first
//! iteration warms every pipeline and scratch pool; a decode session is also
//! measured after the second, which recycles the first's buffers), or if a loss
//! or logit is not finite.

use std::path::Path;

use tessl::gemm::GemmOperands;
use tessl::npy::read_npy;
use tessl::qwen35_adamw::{AdamW, AdamWHyper};
use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::qwen35_train::{Qwen35Grads, Supervise};
use tessl::safetensors::SafeTensors;
use tessl::GpuRuntime;

type Res<T> = Result<T, String>;

fn ps_rss() -> Res<u64> {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .map_err(|e| format!("ps: {e}"))?;
    let kib: u64 = String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .map_err(|e| format!("ps rss: {e}"))?;
    Ok(kib * 1024)
}

fn point(rt: &GpuRuntime, what: &str, i: u64, value: f64) -> Res<u64> {
    let dev = rt.current_allocated_bytes();
    println!(
        "SOAK {what} iter={i} rss_bytes={} device_bytes={dev} value={value:.6}",
        ps_rss()?
    );
    Ok(dev)
}

fn main() -> Res<()> {
    let (mut train, mut decode, mut session, mut every) = (0u64, 0u64, 512u64, None::<u64>);
    for arg in std::env::args().skip(1) {
        let num = |v: &str| v.parse::<u64>().map_err(|_| format!("{arg}: expected a count"));
        if let Some(v) = arg.strip_prefix("--train=") {
            train = num(v)?;
        } else if let Some(v) = arg.strip_prefix("--decode=") {
            decode = num(v)?;
        } else if let Some(v) = arg.strip_prefix("--session=") {
            session = num(v)?;
        } else if let Some(v) = arg.strip_prefix("--every=") {
            every = Some(num(v)?);
        } else {
            return Err(format!(
                "expected --train=N, --decode=N, --session=M or --every=K, got {arg:?}"
            ));
        }
    }
    if (train == 0) == (decode == 0) || session == 0 {
        return Err("give exactly one of --train=N or --decode=N (N >= 1), and --session=M >= 1".into());
    }

    if decode > 0 && decode.div_ceil(session) < 3 {
        return Err("--decode needs at least three sessions (N > 2 * M) to have a warmed baseline".into());
    }

    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train");
    let cfg = Qwen35Config::from_config_file(&dir.join("config.json"))?;
    let rt = GpuRuntime::new()?;
    let st = SafeTensors::open(&dir.join("model.safetensors"))?;
    let model = Qwen35Model::load(&rt, &st, "model.", cfg.clone(), Precision::F32)?;
    let ids: Vec<u32> = read_npy(&dir.join("ids.npy"))?
        .data_i64
        .ok_or("ids.npy is not int64")?
        .iter()
        .map(|&x| u32::try_from(x).map_err(|_| format!("token id {x}")))
        .collect::<Res<_>>()?;

    if train > 0 {
        let bank = Qwen35Grads::zeros_like(&model)?;
        let mut state = AdamW::new(&model)?;
        let wd = model.default_weight_decay(0.01)?;
        let hyper = AdamWHyper {
            lr: 1e-4,
            ..AdamWHyper::default()
        };
        let every = every.unwrap_or((train / 8).max(1));
        let mut baseline = 0;
        let mut last = 0.0;
        for i in 0..train {
            last = model.train_step_into(&ids, GemmOperands::ExactF32, Supervise::Causal, &bank, false)?;
            model.adamw_step(&bank, &mut state, &hyper, &wd)?;
            if !last.is_finite() {
                return Err(format!("train: loss {last} at step {i}"));
            }
            if i == 0 {
                baseline = point(&rt, "train", 1, last)?;
            } else if (i + 1) % every == 0 {
                point(&rt, "train", i + 1, last)?;
            }
        }
        let end = point(&rt, "train-end", train, last)?;
        if end > baseline {
            return Err(format!("train: device allocation grew from {baseline} to {end} bytes"));
        }
    } else {
        let vocab = cfg.vocab as usize;
        let prompt = &ids[..8.min(ids.len())];
        let sessions = decode.div_ceil(session);
        let every = every.unwrap_or((sessions / 8).max(1));
        let mut baseline = 0;
        let mut sum = 0.0f64;
        for s in 0..sessions {
            let steps = session as u32;
            let (mut d, first) = model.prefill(prompt, steps)?;
            let mut logits = first;
            for k in 0..steps {
                let next = logits
                    .iter()
                    .enumerate()
                    .fold(
                        (0usize, f32::NEG_INFINITY),
                        |m, (j, &x)| if x > m.1 { (j, x) } else { m },
                    )
                    .0;
                debug_assert!(next < vocab);
                logits = d.step(next as u32)?;
                if k % 64 == 0 {
                    sum = logits.iter().map(|&x| f64::from(x)).sum();
                    if !sum.is_finite() {
                        return Err(format!("decode: non-finite logits in session {s} at step {k}"));
                    }
                }
            }
            drop(d);
            if s == 0 {
                point(&rt, "decode", 1, sum)?;
            } else if s == 1 {
                baseline = point(&rt, "decode", 2, sum)?;
            } else if (s + 1) % every == 0 {
                point(&rt, "decode", s + 1, sum)?;
            }
        }
        let end = point(&rt, "decode-end", sessions, sum)?;
        if end > baseline {
            return Err(format!("decode: device allocation grew from {baseline} to {end} bytes"));
        }
    }
    Ok(())
}
