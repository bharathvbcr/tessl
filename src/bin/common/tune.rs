//! Sweep the attention kernels' tuning knobs and report the winner per config.
//!
//! Both fast paths have one free parameter each, and neither is derivable:
//!
//! * `BENCH_ATTN_ROWS_R`: lanes per query row. The reduction costs log2(R) steps
//!   against 2*D/R multiply-adds, so small R trades reduction for per-lane work
//!   until `q_reg + acc` spills the register file.
//! * `BENCH_ATTN_ROWS_SGT`: simdgroups per threadgroup. Every simdgroup in a
//!   threadgroup walks the same key range, so this is how many query rows one
//!   global K/V read serves. It interacts with `R` (rows per threadgroup are
//!   `SGT * 32/R`), so the two are swept against each other, not alone.
//! * `BENCH_ATTN_DECODE_CHUNK`: keys per KV chunk. Smaller buys grid parallelism,
//!   which is what a latency-bound decode wants, at the cost of more partials for
//!   the reduce pass to combine.
//!
//! Each is a *compile-time* constant in the shader, so a value is a kernel, not a
//! flag, which is why they are swept rather than guessed.
//!
//! `--knob batched` is not a tuning parameter: it sweeps launches-per-submit to
//! *decompose* a measurement rather than to pick a winner. One launch per submit is
//! what `mx.eval` and `torch.mps.synchronize` do, so it charges the host round
//! trip; 32 amortises it away. The ratio is the share of the wall clock that is
//! actually kernel, and it is printed as such instead of as a winner.
//!
//! This is the Rust owner of what `bench/attn_tune.py` did, and it stays a tool of
//! its own rather than a lane of `bench_paired`: `bench_paired` compares *variants*
//! (the `runtime` column) across lane processes, while this sweeps one variant's
//! compile-time constant across processes of the same binary, picks the argmin per
//! config, and for `batched` reports a decomposition instead of a ratio.
//!
//! Included with `#[path]` by the `attn_tune` bin and by `tests/attn_tune.rs`.

#![allow(dead_code)] // The binary and the tests each use a subset.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use crate::json::{self, Json, Syntax};
use crate::proc::{first_chars, run_bounded};
use crate::pyjson::{median, obj, quote, Doc};

/// One bench run is `iters` timed launches of up to seven configs; an hour is a hang.
pub const RUN_SECS: u64 = 3600;

/// A swept knob: the environment variable that sets it, the values tried, the
/// lane of `bench_flash_attn`'s output that shows its effect, and the configs
/// worth timing it on.
pub struct Knob {
    pub name: &'static str,
    pub env: &'static str,
    pub values: &'static [&'static str],
    pub lane: &'static str,
    pub cfgs: &'static [&'static str],
}

const PREFILL: &[&str] = &[
    "swa128_prefill_512",
    "swa128_prefill_2048",
    "swa128_prefill_4096",
    "swa256_prefill_2048",
    "global512_prefill_1024",
];

const DECODE: &[&str] = &[
    "swa128_decode_1k",
    "swa128_decode_4k",
    "swa128_decode_b8_4k",
    "swa256_decode_4k",
    "global512_decode_4k",
];

const DECODE_MHA: &[&str] = &[
    "swa128_decode_1k",
    "swa128_decode_4k",
    "swa128_decode_b8_4k",
    "swa256_decode_4k",
    "global512_decode_4k",
    "global512_decode_4k_mha",
];

/// Decomposition, not tuning: the shipping routed lane over both decode and
/// prefill, because prefill is the control that says the decode split is real and
/// not an artefact of the batched arm.
const BATCHED: &[&str] = &[
    "swa128_decode_1k",
    "swa128_decode_4k",
    "swa256_decode_4k",
    "global512_decode_4k",
    "swa128_prefill_2048",
    "swa256_prefill_2048",
    "global512_prefill_1024",
];

const THREES: &[&str] = &["8", "16", "32"];

/// In the order `bench/attn_tune.py` declared them (also `--help`'s order after sorting).
pub const KNOBS: &[Knob] = &[
    Knob {
        name: "rows",
        env: "BENCH_ATTN_ROWS_R",
        values: THREES,
        lane: "tessl-rows",
        cfgs: PREFILL,
    },
    Knob {
        name: "decode",
        env: "BENCH_ATTN_DECODE_CHUNK",
        values: &["64", "128", "256"],
        lane: "tessl-decode",
        cfgs: DECODE,
    },
    Knob {
        name: "rows-g",
        env: "BENCH_ATTN_ROWS_SGT",
        values: THREES,
        lane: "tessl-rows",
        cfgs: PREFILL,
    },
    Knob {
        name: "decode-r",
        env: "BENCH_ATTN_DECODE_R",
        values: THREES,
        lane: "tessl-decode",
        cfgs: DECODE,
    },
    // Which query heads share a threadgroup. `group` is the H/Hkv heads that share
    // a KV head; `all` additionally makes the per-key read the whole contiguous
    // [Hkv][D] row, at the cost of H-simdgroup threadgroups.
    Knob {
        name: "decode-sgs",
        env: "BENCH_ATTN_DECODE_SGS",
        values: &["one", "group", "all"],
        lane: "tessl-decode",
        cfgs: DECODE_MHA,
    },
    Knob {
        name: "reduce-w",
        env: "BENCH_ATTN_REDUCE_W",
        values: &["32", "128", "256"],
        lane: "tessl-decode",
        cfgs: DECODE,
    },
    Knob {
        name: "batched",
        env: "BENCH_ATTN_BATCHED",
        values: &["1", "32"],
        lane: "tessl",
        cfgs: BATCHED,
    },
];

pub fn knob(name: &str) -> Option<&'static Knob> {
    KNOBS.iter().find(|k| k.name == name)
}

/// The knob names, sorted: what `--knob` accepts.
pub fn knob_names() -> Vec<&'static str> {
    let mut v: Vec<&str> = KNOBS.iter().map(|k| k.name).collect();
    v.sort_unstable();
    v
}

/// The one canonical list of the knob environment variables, so a run clears
/// every knob it is not sweeping. A hardcoded second copy silently leaks
/// whichever knob was added to [`KNOBS`] but forgotten there.
pub fn knob_env() -> Vec<&'static str> {
    let mut v: Vec<&str> = KNOBS.iter().map(|k| k.env).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// What to do to the caller's environment before one run, in order: a value to
/// set, or `None` to remove the variable.
///
/// Batching is applied to every arm of the sweep, then the swept knob is set last
/// so `--knob batched` still overrides it. A decode sweep at one launch per
/// submit is ~85% host round trip, which compresses every gain it reports towards
/// 1.0x; the knob that wins can still be read off it, but the margin cannot.
pub fn env_edits(
    env_name: &str,
    value: &str,
    cfgs: &[&str],
    iters: usize,
    warmup: usize,
    batched: usize,
) -> Vec<(String, Option<String>)> {
    let mut edits = vec![
        ("BENCH_ITERS".to_string(), Some(iters.to_string())),
        ("BENCH_WARMUP".to_string(), Some(warmup.to_string())),
        ("BENCH_ATTN_CFGS".to_string(), Some(cfgs.join(","))),
    ];
    for other in knob_env() {
        edits.push((other.to_string(), None));
    }
    edits.push(("BENCH_ATTN_BATCHED".to_string(), Some(batched.to_string())));
    edits.push((env_name.to_string(), Some(value.to_string())));
    edits
}

/// `median_ms` per config of the rows in `stdout` whose `runtime` is `lane`.
fn lane_medians(stdout: &str, lane: &str) -> Result<BTreeMap<String, f64>, String> {
    let syntax = Syntax {
        what: "bench_flash_attn output",
        max_depth: 4,
        uints_only: false,
        literals: true,
    };
    let Json::Array(rows) = json::parse(stdout.trim(), syntax)? else {
        return Err("bench_flash_attn printed JSON that is not an array of rows".to_string());
    };
    let mut out = BTreeMap::new();
    for (i, row) in rows.iter().enumerate() {
        let field = |key: &str| row.get(key);
        let (Some(Json::Str(cfg)), Some(Json::Str(runtime)), Some(Json::Num { value, .. })) =
            (field("cfg"), field("runtime"), field("median_ms"))
        else {
            return Err(format!(
                "bench_flash_attn row {i} lacks a string cfg, a string runtime or a numeric median_ms"
            ));
        };
        if runtime == lane {
            out.insert(cfg.clone(), *value);
        }
    }
    Ok(out)
}

/// How each run of the bench is configured, whatever knob is swept.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub iters: usize,
    pub warmup: usize,
    /// Launches per submit, for every arm of the sweep.
    pub batched: usize,
    /// How long one run may take.
    pub secs: u64,
}

/// Run `binary` once with `env_name=value` and return `lane`'s median per config.
pub fn run_once(
    binary: &Path,
    (env_name, value): (&str, &str),
    lane: &str,
    cfgs: &[&str],
    s: &Settings,
) -> Result<BTreeMap<String, f64>, String> {
    let secs = s.secs;
    let mut cmd = Command::new(binary);
    for (k, v) in env_edits(env_name, value, cfgs, s.iters, s.warmup, s.batched) {
        match v {
            Some(v) => cmd.env(k, v),
            None => cmd.env_remove(k),
        };
    }
    let ran = run_bounded(&mut cmd, secs).map_err(|e| format!("{env_name}={value}: {e}"))?;
    if ran.code != Some(0) {
        return Err(format!(
            "{env_name}={value}: exited {}\n{}",
            ran.code.map_or("by signal".to_string(), |c| c.to_string()),
            ran.stderr.trim()
        ));
    }
    let out = lane_medians(&ran.stdout, lane)
        .map_err(|e| format!("{env_name}={value}: {e}: {}", first_chars(&ran.stdout, 200)))?;
    let missing: Vec<&str> = cfgs.iter().copied().filter(|c| !out.contains_key(*c)).collect();
    if !missing.is_empty() {
        return Err(format!(
            "{env_name}={value}: lane {} absent for {}. Refusing to pick a winner from a partial sweep.",
            quote(lane),
            crate::pyjson::quote_list(&missing)
        ));
    }
    Ok(out)
}

/// What the sweep measured: per value, per config, one median per round.
pub type PerRound = Vec<Vec<Vec<f64>>>;

/// The printed report and the document written by `--out`.
pub struct Report {
    pub text: String,
    pub doc: Doc,
}

fn strs(v: &[&str]) -> Doc {
    Doc::strs(v)
}

/// Median ms per (value, config) over the interleaved rounds, printed as a table.
pub fn render(k: &Knob, rounds: usize, iters: usize, batched: usize, per: &PerRound) -> Result<Report, String> {
    let (values, cfgs) = (k.values, k.cfgs);
    let med: Vec<Vec<f64>> = per
        .iter()
        .map(|by_cfg| by_cfg.iter().map(|xs| median(xs)).collect())
        .collect();
    let med_doc = Doc::Obj(
        values
            .iter()
            .enumerate()
            .map(|(vi, v)| {
                (
                    v.to_string(),
                    Doc::Obj(
                        cfgs.iter()
                            .enumerate()
                            .map(|(ci, c)| (c.to_string(), Doc::Float(med[vi][ci])))
                            .collect(),
                    ),
                )
            })
            .collect(),
    );
    let what = if batched == 1 {
        "wall clock, submit-and-wait per call".to_string()
    } else {
        format!("kernel only, {batched} launches per submit")
    };
    let mut text = format!("\n{}  (median ms over {rounds} interleaved rounds, {what})\n", k.env);

    if k.name == "batched" {
        // No winner is reported here on purpose. 32 launches per submit is a
        // measurement protocol, not a setting to ship, and an artifact that named
        // it "winner" would read as a recommendation.
        if values != ["1", "32"] {
            return Err(format!(
                "--knob batched expects values ('1', '32'), got {}. The share is defined against one launch per submit.",
                crate::pyjson::quote_list(values)
            ));
        }
        let (solo, amortised) = (0, values.len() - 1);
        text.push_str(&format!(
            "  {:<24}{:>10}{:>12}{:>11}{:>14}\n",
            "config", "solo ms", "batched ms", "submit ms", "kernel share"
        ));
        let mut share = Vec::new();
        for (ci, c) in cfgs.iter().enumerate() {
            let (hi, lo) = (med[solo][ci], med[amortised][ci]);
            share.push((c.to_string(), Doc::Float(lo / hi)));
            text.push_str(&format!(
                "  {c:<24}{hi:>10.4}{lo:>12.4}{:>11.4}{:>13.1}%\n",
                hi - lo,
                100.0 * (lo / hi)
            ));
        }
        let doc = obj([
            ("knob", Doc::str(k.name)),
            ("env", Doc::str(k.env)),
            ("values", strs(values)),
            ("rounds", Doc::uint(rounds)),
            ("iters", Doc::uint(iters)),
            ("batched", Doc::uint(batched)),
            ("median_ms", med_doc),
            ("winner", Doc::Null),
            ("kernel_share", Doc::Obj(share)),
        ]);
        return Ok(Report { text, doc });
    }

    text.push_str(&format!("  {:<24}", "config"));
    for v in values {
        text.push_str(&format!("{v:>10}"));
    }
    text.push_str(&format!("{:>10}{:>8}\n", "winner", "gain"));
    let mut winners = Vec::new();
    for (ci, c) in cfgs.iter().enumerate() {
        // This config's median under each value, in the order the values were swept.
        let col: Vec<f64> = med.iter().map(|by_cfg| by_cfg[ci]).collect();
        // The first of equal minima, as Python's min() keeps it.
        let mut best = 0;
        for (vi, x) in col.iter().enumerate().skip(1) {
            if *x < col[best] {
                best = vi;
            }
        }
        let worst = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        winners.push((c.to_string(), Doc::str(values[best])));
        text.push_str(&format!("  {c:<24}"));
        for x in &col {
            text.push_str(&format!("{x:>10.3}"));
        }
        text.push_str(&format!("{:>10}{:>7.2}x\n", values[best], worst / col[best]));
    }
    let doc = obj([
        ("knob", Doc::str(k.name)),
        ("env", Doc::str(k.env)),
        ("values", strs(values)),
        ("rounds", Doc::uint(rounds)),
        ("iters", Doc::uint(iters)),
        ("batched", Doc::uint(batched)),
        ("median_ms", med_doc),
        ("winner", Doc::Obj(winners)),
    ]);
    Ok(Report { text, doc })
}
