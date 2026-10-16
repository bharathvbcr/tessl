//! Sweep the attention kernels' tuning knobs through `bench_flash_attn` and
//! report the winner per config.
//!
//! ```text
//! attn_tune --knob KNOB [--rounds N] [--iters N] [--warmup N] [--batched N]
//!           [--out PATH] [--binary PATH]
//! ```
//!
//! `KNOB` is one of `rows`, `rows-g`, `decode`, `decode-r`, `decode-sgs`,
//! `reduce-w` or `batched`; what each sweeps, and why `batched` is a decomposition
//! rather than a tuning parameter, is in `common/tune.rs`. Each value is a
//! compile-time constant in the shader, so it is its own kernel and is measured in
//! its own process; the values are visited round-robin (not all rounds of one and
//! then the next) so thermal drift cannot favour whichever was measured first.
//!
//! ```text
//! cargo run --release --bin attn_tune -- --knob rows
//! cargo run --release --bin attn_tune -- --knob decode --rounds 3
//! cargo run --release --bin attn_tune -- --knob batched --rounds 3
//! ```
//!
//! `bench_flash_attn` is the binary next to this one unless `--binary` names
//! another. Exit status: 0 done, 1 a run failed or did not report a lane (no
//! winner is picked from a partial sweep), 2 usage.

#[path = "../json.rs"]
#[allow(dead_code)]
mod json;
#[path = "common/proc.rs"]
mod proc;
#[path = "common/pyjson.rs"]
mod pyjson;
#[path = "common/tune.rs"]
mod tune;

use std::path::PathBuf;
use tune::{Settings, RUN_SECS};

fn usage() -> String {
    format!(
        "usage: attn_tune --knob {{{}}} [--rounds N] [--iters N] [--warmup N] [--batched N] [--out PATH] [--binary PATH]",
        tune::knob_names().join(",")
    )
}

fn usage_error(msg: &str) -> ! {
    eprintln!("attn_tune: error: {msg}\n{}", usage());
    std::process::exit(2);
}

struct Args {
    knob: &'static tune::Knob,
    rounds: i64,
    iters: i64,
    warmup: i64,
    batched: i64,
    out: Option<PathBuf>,
    binary: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut knob = None;
    let (mut rounds, mut iters, mut warmup, mut batched) = (3, 20, 5, 1);
    let (mut out, mut binary) = (None, None);
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let (name, inline) = match arg.split_once('=') {
            Some((n, v)) if n.starts_with("--") => (n.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut value = |what: &str| {
            inline
                .clone()
                .or_else(|| it.next())
                .unwrap_or_else(|| usage_error(&format!("argument {what}: expected one argument")))
        };
        let number = |what: &str, v: String| {
            v.parse::<i64>()
                .unwrap_or_else(|_| usage_error(&format!("argument {what}: invalid int value: {v:?}")))
        };
        match name.as_str() {
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            "--knob" => {
                let v = value("--knob");
                knob = Some(tune::knob(&v).unwrap_or_else(|| {
                    usage_error(&format!(
                        "argument --knob: invalid choice: {} (choose from {})",
                        pyjson::quote(&v),
                        tune::knob_names()
                            .iter()
                            .map(|n| pyjson::quote(n))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                }));
            }
            "--rounds" => rounds = number("--rounds", value("--rounds")),
            "--iters" => iters = number("--iters", value("--iters")),
            "--warmup" => warmup = number("--warmup", value("--warmup")),
            "--batched" => batched = number("--batched", value("--batched")),
            "--out" => out = Some(PathBuf::from(value("--out"))),
            "--binary" => binary = Some(PathBuf::from(value("--binary"))),
            _ => usage_error(&format!("unrecognized arguments: {arg}")),
        }
    }
    let Some(knob) = knob else {
        usage_error("the following arguments are required: --knob")
    };
    Args {
        knob,
        rounds,
        iters,
        warmup,
        batched,
        out,
        binary,
    }
}

fn run(a: &Args) -> Result<(), String> {
    if a.rounds < 1 {
        return Err(format!("--rounds must be >= 1, got {}", a.rounds));
    }
    if a.batched < 1 {
        return Err(format!("--batched must be >= 1, got {}", a.batched));
    }
    if a.knob.name == "batched" && a.batched != 1 {
        return Err(
            "--knob batched sweeps launches-per-submit; --batched would fix the very thing being \
                    swept. Drop one of them."
                .to_string(),
        );
    }
    // `bench_flash_attn` refuses a count it cannot run; these only have to fit.
    let count = |what: &str, n: i64| usize::try_from(n).map_err(|_| format!("{what} must be >= 0, got {n}"));
    let settings = Settings {
        iters: count("--iters", a.iters)?,
        warmup: count("--warmup", a.warmup)?,
        batched: a.batched as usize,
        secs: RUN_SECS,
    };
    let rounds = a.rounds as usize;
    let binary = match &a.binary {
        Some(b) => b.clone(),
        None => std::env::current_exe()
            .map_err(|e| format!("current_exe: {e}"))?
            .with_file_name("bench_flash_attn"),
    };
    if !binary.exists() {
        return Err(format!(
            "{} not built. cargo build --release --bin bench_flash_attn",
            binary.display()
        ));
    }

    let k = a.knob;
    // Round-robin over the values rather than all rounds of one then the next, so
    // thermal drift cannot favour whichever was measured first.
    let mut per: tune::PerRound = vec![vec![Vec::new(); k.cfgs.len()]; k.values.len()];
    for r in 0..rounds {
        for (vi, v) in k.values.iter().enumerate() {
            let got = tune::run_once(&binary, (k.env, v), k.lane, k.cfgs, &settings)?;
            for (ci, c) in k.cfgs.iter().enumerate() {
                per[vi][ci].push(got[*c]);
            }
        }
        eprintln!("round {}/{rounds} done", r + 1);
    }

    let report = tune::render(k, rounds, settings.iters, settings.batched, &per)?;
    print!("{}", report.text);
    if let Some(out) = &a.out {
        std::fs::write(out, report.doc.dumps()?).map_err(|e| format!("write {}: {e}", out.display()))?;
        eprintln!("\nwrote {}", out.display());
    }
    Ok(())
}

fn main() {
    let args = parse_args();
    if let Err(msg) = run(&args) {
        eprintln!("{msg}");
        std::process::exit(1);
    }
}
