//! Rebuild the EmbedGemma 2 result files under `bench/results/` from the raw
//! runs behind them. Analysis only; ships nothing.
//!
//! ```text
//! results_report memory      RAW_DIR OUT_DIR
//! results_report small-m     RESULTS_DIR RAW_DIR LANE_BEFORE LANE_AFTER LANE_CONTROL
//! results_report stage       OUT_DIR PASS.json... [--gemm G.json] [--memory M.json] [--layout L.json]
//!                            [--ledger LG.json] [--alloc A.json] [--ple P.json] [--contended STEM...]
//! results_report vision      OUT_DIR LABEL=PASS.json...
//! results_report text-fusion OUT_DIR LABEL=FILE.json...
//! ```
//!
//! Each subcommand is the port of one stdlib-only Python script (`memory` of
//! `embedgemma2_memory_report.py`, `small-m` of `embedgemma2_small_m_report.py`,
//! `stage` of `embedgemma2_stage_profile_report.py`, `vision` of
//! `embedgemma2_vision_profile_report.py`, `text-fusion` of
//! `embedgemma2_text_fusion_profile_report.py`); the module docs of each say
//! what its inputs are. The outputs are byte-identical to what the scripts
//! wrote: `py.rs` reproduces the pieces of Python they leaned on (`int` and
//! `float` kept apart, `sum`, `statistics.median`, float `repr` and `format`,
//! `json.dumps`), and `tests/results_report.rs` holds it to that against the
//! committed files and against golden outputs the scripts produced.
//!
//! Where a script would have raised (a missing key, a division by zero, `min`
//! of an empty list), this exits 1 with the reason and writes nothing more.

// The crate's strict parser, shared the way bench_paired shares it; this tool
// does not use its object lookup.
#[allow(dead_code)]
#[path = "../../json.rs"]
mod json;

mod fusion;
mod memory;
mod py;
mod small_m;
mod stage;
mod vision;

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.split_first() {
        Some((cmd, rest)) => match cmd.as_str() {
            "memory" => memory::run(rest),
            "small-m" => small_m::run(rest),
            "stage" => stage::run(rest),
            "vision" => vision::run(rest),
            "text-fusion" => fusion::run(rest),
            other => Err(format!(
                "unknown report {other:?}; one of: memory, small-m, stage, vision, text-fusion"
            )),
        },
        None => Err("usage: results_report memory|small-m|stage|vision|text-fusion ...".to_string()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("results_report: {msg}");
            ExitCode::FAILURE
        }
    }
}
