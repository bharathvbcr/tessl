//! Alternate benchmark lanes round by round, so clock drift and warm-up land
//! on every lane alike instead of in the ratio between them.
//!
//! ```text
//! bench_paired [--rounds N] [--out PATH] [--format F] [--max-spread X]
//!              [--clean-env] [--evidence] [--compare A:B]... [--keys K,..]
//!              [--require FIELD=VALUE]... LANE...
//! LANE = NAME=COMMAND  |  NAME:ENV=V[,ENV=V...]=COMMAND
//! ```
//!
//! Each lane is a command run as a fresh process, split on whitespace (single
//! or double quotes group a word, so a path may hold spaces). It inherits this
//! process's environment, with the lane's `ENV=V` pairs set on top (an empty
//! `V` unsets the variable; a value holding `,` or `=` is quoted, as in
//! `'tessl:BENCH_SHAPES="64x64x64,128x128x128"=bench_gemm_sweep'`). `--clean-env` first
//! drops the inherited `BENCH_*`, `TESSL_*`, `METAL_*`, `MTL_*`, `MLX_*`,
//! `PYTORCH_*`, `OMP_*`, `MKL_*`, `VECLIB_*` and `ACCELERATE_*` variables, so
//! only what the lane lines name reaches a lane.
//!
//! Every round runs every lane once, in the given order on even rounds and
//! reversed on odd ones. What a lane prints is chosen by `--format`:
//!
//! * `workloads` (default): the last stdout line is a JSON array of objects,
//!   each with a string `workload` and a positive number `ms_min` (the
//!   workload's fastest iteration in that process: min-of-N). Other numeric
//!   fields (`device_peak_mib`, `peak_footprint_mib`, `forwards`) are carried
//!   into the report. `bench_embedgemma2` and `bench/embedgemma2_torch.py`
//!   print this.
//! * `metrics`: stdout carries `METRIC name value` lines (`bench_nn_kernels`,
//!   `bench_qwen35_layers`, `bench_qwen35_train --step`,
//!   `bench/metallib_clean_time.sh`). Each name is a workload; lower is
//!   better for every metric those benches print.
//! * `rows:KEY,VARIANT,VALUE`: the whole of stdout is one JSON array of
//!   objects; the field `KEY` names the shape or config, `VARIANT` the
//!   implementation, `VALUE` the positive number measured
//!   (`bench_gemm_sweep`, `bench_flash_attn`, `bench/gemm_sweep_mlx.py`,
//!   `bench/flash_attn_torch_mlx.py`). Rows from different lanes are compared
//!   across processes with `--compare A:B`, where `A` and `B` are variants.
//!   `--require FIELD=VALUE` refuses any row whose numeric `FIELD` is not
//!   `VALUE` (both lanes must have run the same batching). `--keys` names the
//!   whole ladder; a key absent from any round is an error, so a geomean
//!   never covers part of it silently.
//!
//! Per workload (or key) the report gives the median over rounds; per lane
//! after the first (the base) it gives the median of the per-round ratios
//! `lane / base` with the lowest and highest round as the spread. A spread
//! that straddles 1.0 is not a result. With three or more lanes it also gives
//! each lane against the one before it (`vs_prev`), so a chain of builds, one
//! change each, benches every change separately in one interleaved run. A
//! workload a lane did not report in some round is an error, not a gap.
//!
//! `--max-spread X` (1 to 1.25) makes a ratio series whose max/min exceeds `X`
//! an error: exit 3, nothing printed as a result, nothing published.
//! `--evidence` adds the whole publication contract for `--out`: an even
//! round count of at least 4 (default 6), `BENCH_ITERS >= 3` where set, a
//! 1.10 drift gate unless `--max-spread` says otherwise, a `.attempt.json`
//! marker that makes an older output stale until the new one is published
//! atomically, and a provenance block (git revision, dirty flag and diff
//! hash, the hash of every lane executable, which embeds its metallib, and
//! of every script argument, host, power, load). Exit status: 2 usage, 1 a lane or the output failed, 3 drift.

mod common;

#[path = "common/evidence.rs"]
mod evidence;
#[path = "../json.rs"]
#[allow(dead_code)]
mod json;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use common::median;
use evidence::{Jv, Summary};
use json::{Json, Syntax};

type Res<T> = Result<T, String>;

const SYNTAX: Syntax = Syntax {
    what: "lane output",
    max_depth: 4,
    uints_only: false,
    literals: true,
};

/// Numeric fields of a workload row that are carried into the report.
const EXTRA_FIELDS: [&str; 3] = ["device_peak_mib", "peak_footprint_mib", "forwards"];

enum Fail {
    Usage(String),
    Run(String),
    Drift(Vec<String>),
}

impl From<String> for Fail {
    fn from(e: String) -> Self {
        Fail::Run(e)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Format {
    Workloads,
    Metrics,
    Rows {
        key: String,
        variant: String,
        value: String,
    },
}

impl Format {
    fn parse(s: &str) -> Res<Self> {
        match s {
            "workloads" => Ok(Format::Workloads),
            "metrics" => Ok(Format::Metrics),
            _ => {
                let fields = s
                    .strip_prefix("rows:")
                    .map(|f| f.split(',').map(str::to_string).collect::<Vec<_>>());
                match fields.as_deref() {
                    Some([key, variant, value]) if ![key, variant, value].iter().any(|f| f.is_empty()) => {
                        Ok(Format::Rows {
                            key: key.clone(),
                            variant: variant.clone(),
                            value: value.clone(),
                        })
                    }
                    _ => Err(format!(
                        "--format {s:?}: expected workloads, metrics or rows:KEY,VARIANT,VALUE"
                    )),
                }
            }
        }
    }

    /// The field the statistics are over.
    fn primary(&self) -> &'static str {
        match self {
            Format::Workloads => "ms_min",
            _ => "value",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Lane {
    name: String,
    argv: Vec<String>,
    env: Vec<(String, String)>,
}

/// Split on whitespace; a single or double quote groups a word.
fn split_command(cmd: &str) -> Res<Vec<String>> {
    let mut words = Vec::new();
    let mut cur = String::new();
    let mut in_word = false;
    let mut quote: Option<char> = None;
    for c in cmd.chars() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), c) => cur.push(c),
            (None, '\'' | '"') => {
                quote = Some(c);
                in_word = true;
            }
            (None, c) if c.is_whitespace() => {
                if in_word {
                    words.push(std::mem::take(&mut cur));
                    in_word = false;
                }
            }
            (None, c) => {
                cur.push(c);
                in_word = true;
            }
        }
    }
    if quote.is_some() {
        return Err(format!("{cmd:?}: unterminated quote"));
    }
    if in_word {
        words.push(cur);
    }
    Ok(words)
}

fn valid_env_name(k: &str) -> bool {
    let mut chars = k.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `NAME=COMMAND` or `NAME:ENV=V[,ENV=V...]=COMMAND`.
fn parse_lane(arg: &str) -> Res<Lane> {
    let eq = arg.find('=');
    let (name, env, cmd) = match (arg.find(':'), eq) {
        (Some(colon), Some(eq)) if colon < eq => {
            let mut env = Vec::new();
            let mut rest = &arg[colon + 1..];
            let cmd = loop {
                let (k, after) = rest
                    .split_once('=')
                    .ok_or_else(|| format!("{arg:?}: expected NAME:ENV=V[,ENV=V]=COMMAND"))?;
                if !valid_env_name(k) {
                    return Err(format!("{arg:?}: {k:?} is not an environment variable name"));
                }
                // A quoted value may hold `,` and `=` (`BENCH_SHAPES="64x64x64,128x128x128"`).
                let (value, after_value) = match after.chars().next() {
                    Some(q @ ('"' | '\'')) => {
                        let close = after[1..]
                            .find(q)
                            .ok_or_else(|| format!("{arg:?}: unterminated quote in the value of {k}"))?;
                        (&after[1..=close], &after[close + 2..])
                    }
                    _ => {
                        let end = after
                            .find([',', '='])
                            .ok_or_else(|| format!("{arg:?}: expected =COMMAND after the environment"))?;
                        (&after[..end], &after[end..])
                    }
                };
                env.push((k.to_string(), value.to_string()));
                match after_value.as_bytes().first() {
                    Some(b'=') => break &after_value[1..],
                    Some(b',') => rest = &after_value[1..],
                    _ => return Err(format!("{arg:?}: expected , or =COMMAND after the value of {k}")),
                }
            };
            (&arg[..colon], env, cmd)
        }
        (_, Some(eq)) => (&arg[..eq], Vec::new(), &arg[eq + 1..]),
        _ => return Err(format!("{arg:?}: expected NAME=COMMAND or NAME:ENV=V=COMMAND")),
    };
    let argv = split_command(cmd)?;
    if name.is_empty() || argv.is_empty() {
        return Err(format!("{arg:?}: empty lane name or command"));
    }
    Ok(Lane {
        name: name.to_string(),
        argv,
        env,
    })
}

/// The environment a lane's process gets.
fn lane_env(
    lane: &Lane,
    inherited: impl IntoIterator<Item = (String, String)>,
    clean: bool,
) -> BTreeMap<String, String> {
    let mut env: BTreeMap<String, String> = if clean {
        evidence::clean_benchmark_env(inherited)
    } else {
        inherited.into_iter().collect()
    };
    for (k, v) in &lane.env {
        if v.is_empty() {
            env.remove(k);
        } else {
            env.insert(k.clone(), v.clone());
        }
    }
    env
}

/// `(workload or key, variant)`; the variant is empty outside `rows`.
type Id = (String, String);

/// One lane's numbers in one round: id -> field -> value, in print order.
struct LaneRun {
    fields: BTreeMap<Id, BTreeMap<String, f64>>,
    order: Vec<Id>,
    raw: Jv,
}

fn positive(what: &str, lane: &str, v: f64) -> Res<f64> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(format!("lane {lane}: {what} is not a positive finite number: {v}"))
    }
}

fn parse_workloads(lane: &str, stdout: &str) -> Res<LaneRun> {
    let raw = stdout
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| format!("lane {lane} printed nothing"))?
        .trim()
        .to_string();
    let Json::Array(rows) = json::parse(&raw, SYNTAX).map_err(|e| format!("lane {lane}: {e}"))? else {
        return Err(format!("lane {lane}: the last line is not a JSON array"));
    };
    let (mut fields, mut order) = (BTreeMap::new(), Vec::new());
    for row in &rows {
        let Some(Json::Str(w)) = row.get("workload") else {
            return Err(format!("lane {lane}: a row has no string \"workload\""));
        };
        let Json::Object(kv) = row else {
            unreachable!("get() succeeded on an object")
        };
        let nums: BTreeMap<String, f64> = kv
            .iter()
            .filter_map(|(k, v)| match v {
                Json::Num { value, .. } => Some((k.clone(), *value)),
                _ => None,
            })
            .collect();
        match nums.get("ms_min") {
            Some(m) if m.is_finite() && *m > 0.0 => {}
            _ => return Err(format!("lane {lane}: workload {w:?} has no positive \"ms_min\"")),
        }
        let id = (w.clone(), String::new());
        if fields.insert(id.clone(), nums).is_some() {
            return Err(format!("lane {lane}: workload {w:?} reported twice"));
        }
        order.push(id);
    }
    Ok(LaneRun {
        fields,
        order,
        raw: Jv::Raw(raw),
    })
}

fn parse_metrics(lane: &str, stdout: &str) -> Res<LaneRun> {
    let (mut fields, mut order) = (BTreeMap::new(), Vec::new());
    for line in stdout.lines().filter(|l| l.starts_with("METRIC ")) {
        let mut tok = line.split_whitespace().skip(1);
        let (Some(name), Some(v)) = (tok.next(), tok.next()) else {
            return Err(format!("lane {lane}: malformed METRIC line {line:?}"));
        };
        let v: f64 = v
            .parse()
            .map_err(|_| format!("lane {lane}: METRIC {name} value {v:?} is not a number"))?;
        if !v.is_finite() {
            return Err(format!("lane {lane}: METRIC {name} is not finite: {v}"));
        }
        let id = (name.to_string(), String::new());
        if fields
            .insert(id.clone(), BTreeMap::from([("value".to_string(), v)]))
            .is_some()
        {
            return Err(format!("lane {lane}: METRIC {name} reported twice"));
        }
        order.push(id);
    }
    if order.is_empty() {
        return Err(format!("lane {lane} printed no METRIC lines"));
    }
    let raw = Jv::Obj(
        order
            .iter()
            .map(|id| (id.0.clone(), Jv::Num(fields[id]["value"])))
            .collect(),
    );
    Ok(LaneRun { fields, order, raw })
}

fn scalar_text(v: &Json) -> Option<String> {
    match v {
        Json::Str(s) => Some(s.clone()),
        Json::Num { value, uint: Some(u) } if value.is_finite() => Some(u.to_string()),
        Json::Num { value, .. } if value.is_finite() => Some(value.to_string()),
        _ => None,
    }
}

fn parse_rows(
    lane: &str,
    stdout: &str,
    key: &str,
    variant: &str,
    value: &str,
    require: &[(String, f64)],
) -> Res<LaneRun> {
    let Json::Array(rows) =
        json::parse(stdout.trim(), SYNTAX).map_err(|e| format!("lane {lane}: stdout is not JSON: {e}"))?
    else {
        return Err(format!("lane {lane}: stdout is not a JSON array"));
    };
    let (mut fields, mut order) = (BTreeMap::new(), Vec::new());
    for row in &rows {
        let text = |f: &str| {
            row.get(f)
                .and_then(scalar_text)
                .ok_or_else(|| format!("lane {lane}: a row has no scalar {f:?}"))
        };
        let (k, var) = (text(key)?, text(variant)?);
        let v = match row.get(value) {
            Some(Json::Num { value, .. }) => positive(&format!("{k}/{var} {value:?}"), lane, *value)?,
            _ => return Err(format!("lane {lane}: {k}/{var} has no numeric {value:?}")),
        };
        // A ratio across lanes that ran different batching is an apples to
        // oranges number that looks plausible; a row that does not say what it
        // ran cannot be checked and is refused.
        for (field, want) in require {
            match row.get(field) {
                Some(Json::Num { value: got, .. }) if got == want => {}
                Some(Json::Num { value: got, .. }) => {
                    return Err(format!(
                        "lane {lane}: {k}/{var} ran with {field}={got}, but --require asked for {want}; \
                         a ratio across different {field} is meaningless"
                    ))
                }
                _ => {
                    return Err(format!(
                        "lane {lane}: {k}/{var} carries no numeric {field:?}, so --require {field}={want} \
                         cannot be confirmed; refusing to form a ratio"
                    ))
                }
            }
        }
        let id = (k.clone(), var.clone());
        if fields
            .insert(id.clone(), BTreeMap::from([("value".to_string(), v)]))
            .is_some()
        {
            return Err(format!("lane {lane}: duplicate measurement for {k}/{var}"));
        }
        order.push(id);
    }
    if order.is_empty() {
        return Err(format!("lane {lane}: no rows"));
    }
    let raw = Jv::Arr(
        order
            .iter()
            .map(|(k, var)| {
                Jv::obj([
                    ("key", Jv::str(k.clone())),
                    ("variant", Jv::str(var.clone())),
                    ("value", Jv::Num(fields[&(k.clone(), var.clone())]["value"])),
                ])
            })
            .collect(),
    );
    Ok(LaneRun { fields, order, raw })
}

struct Options {
    rounds: usize,
    out: Option<PathBuf>,
    format: Format,
    compares: Vec<(String, String)>,
    keys: Option<Vec<String>>,
    require: Vec<(String, f64)>,
    max_spread: Option<f64>,
    clean_env: bool,
    evidence: bool,
    lanes: Vec<Lane>,
}

fn parse_args(args: &[String]) -> Res<Options> {
    let mut o = Options {
        rounds: 0,
        out: None,
        format: Format::Workloads,
        compares: Vec::new(),
        keys: None,
        require: Vec::new(),
        max_spread: None,
        clean_env: false,
        evidence: false,
        lanes: Vec::new(),
    };
    let mut rounds = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |flag: &str| it.next().cloned().ok_or_else(|| format!("{flag} needs a value"));
        match a.as_str() {
            "--rounds" => {
                rounds = Some(parse_rounds(&value("--rounds")?)?);
            }
            "--out" => o.out = Some(PathBuf::from(value("--out")?)),
            "--format" => o.format = Format::parse(&value("--format")?)?,
            "--compare" => {
                let v = value("--compare")?;
                match v.split_once(':') {
                    Some((x, y)) if !x.is_empty() && !y.is_empty() && x != y => {
                        o.compares.push((x.to_string(), y.to_string()))
                    }
                    _ => return Err(format!("--compare {v:?}: expected two different variants A:B")),
                }
            }
            "--keys" => {
                let v = value("--keys")?;
                let keys: Vec<String> = v.split(',').map(|k| k.trim().to_string()).collect();
                if keys.iter().any(String::is_empty) || keys.iter().collect::<BTreeSet<_>>().len() != keys.len() {
                    return Err(format!("--keys {v:?}: empty or duplicate key"));
                }
                o.keys = Some(keys);
            }
            "--require" => {
                let v = value("--require")?;
                let (f, n) = v
                    .split_once('=')
                    .ok_or_else(|| format!("--require {v:?}: expected FIELD=VALUE"))?;
                let n: f64 = n
                    .parse()
                    .map_err(|_| format!("--require {v:?}: VALUE is not a number"))?;
                o.require.push((f.to_string(), n));
            }
            "--max-spread" => o.max_spread = Some(evidence::parse_ratio_spread_limit(&value("--max-spread")?)?),
            "--clean-env" => o.clean_env = true,
            "--evidence" => o.evidence = true,
            _ if a.starts_with("--") && !a.contains('=') => {
                return Err(format!("unknown option {a}; lanes are NAME=COMMAND"));
            }
            _ => {
                let lane = parse_lane(a)?;
                if o.lanes.iter().any(|l| l.name == lane.name) {
                    return Err(format!("lane {:?} given twice", lane.name));
                }
                o.lanes.push(lane);
            }
        }
    }
    if o.lanes.len() < 2 {
        return Err("give at least two lanes; the first is the base".into());
    }
    o.rounds = rounds.unwrap_or(if o.evidence { evidence::DEFAULT_OUTER_ROUNDS } else { 5 });
    if o.rounds < 2 {
        return Err("--rounds must be at least 2: a single round has no spread".into());
    }
    let rows = matches!(o.format, Format::Rows { .. });
    if rows && o.compares.is_empty() {
        return Err("--format rows needs at least one --compare A:B".into());
    }
    if !rows && (!o.compares.is_empty() || o.keys.is_some() || !o.require.is_empty()) {
        return Err("--compare, --keys and --require apply to --format rows only".into());
    }
    if o.evidence {
        if o.out.is_none() {
            return Err("--evidence publishes an artifact and needs --out".into());
        }
        o.max_spread.get_or_insert(evidence::DEFAULT_MAX_RATIO_SPREAD);
    }
    Ok(o)
}

/// A positive decimal integer: ASCII digits, no sign, no leading zero, at most
/// six digits. `str::parse` would also take `+2` and `007`.
fn parse_rounds(v: &str) -> Res<usize> {
    let ok = !v.is_empty() && v.len() <= 6 && v.bytes().all(|b| b.is_ascii_digit()) && !v.starts_with('0');
    if !ok {
        return Err(format!("--rounds must be a positive integer (1-999999), got {v:?}"));
    }
    Ok(v.parse().expect("validated digits"))
}

/// A lane's `BENCH_ITERS`, from the lane line or this process.
fn lane_iters(env: &BTreeMap<String, String>) -> Res<Option<usize>> {
    env.get("BENCH_ITERS")
        .map(|v| {
            v.trim()
                .parse()
                .map_err(|_| format!("BENCH_ITERS={v:?} is not a non-negative integer"))
        })
        .transpose()
}

fn run_lane(lane: &Lane, env: &BTreeMap<String, String>, o: &Options) -> Res<LaneRun> {
    let out = Command::new(&lane.argv[0])
        .args(&lane.argv[1..])
        .env_clear()
        .envs(env)
        .output()
        .map_err(|e| format!("lane {}: {}: {e}", lane.name, lane.argv[0]))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let tail: String = err
            .chars()
            .rev()
            .take(4000)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        return Err(format!("lane {} failed ({}):\n{tail}", lane.name, out.status));
    }
    let stdout = String::from_utf8(out.stdout).map_err(|e| format!("lane {}: stdout: {e}", lane.name))?;
    match &o.format {
        Format::Workloads => parse_workloads(&lane.name, &stdout),
        Format::Metrics => parse_metrics(&lane.name, &stdout),
        Format::Rows { key, variant, value } => parse_rows(&lane.name, &stdout, key, variant, value, &o.require),
    }
}

// ------------------------------------------------------------------ report

/// Everything a report prints and publishes, and every drift violation found.
struct Report {
    table: Vec<String>,
    summary: Vec<Jv>,
    violations: Vec<String>,
    extra: Vec<(String, Jv)>,
}

fn ratio_series(a: &[f64], b: &[f64], label: &str) -> Res<Summary> {
    let r: Vec<f64> = a.iter().zip(b).map(|(x, y)| x / y).collect();
    evidence::series_summary(&r, label)
}

fn summary_json(s: &Summary) -> Jv {
    Jv::obj([
        ("median", Jv::Num(s.median)),
        ("min", Jv::Num(s.min)),
        ("max", Jv::Num(s.max)),
        ("spread", Jv::Num(s.spread)),
        ("outer_round_values", Jv::nums(&s.values)),
    ])
}

fn gate(s: &Summary, limit: Option<f64>, label: &str, violations: &mut Vec<String>) -> Res<()> {
    if let Some(limit) = limit {
        if let Err(e) = evidence::enforce_ratio_spread(s, limit, label) {
            violations.push(e);
        }
    }
    Ok(())
}

/// Workloads and metrics: every lane's per-workload median over rounds, and
/// each lane's per-round ratio against the base (and, with a chain of three or
/// more, against the lane before it).
fn report_lanes(o: &Options, runs: &[Vec<LaneRun>]) -> Res<Report> {
    let lanes = &o.lanes;
    let primary = o.format.primary();
    let unit = if o.format == Format::Workloads { " ms" } else { "" };
    let chain = lanes.len() > 2;
    let mut rep = Report {
        table: Vec::new(),
        summary: Vec::new(),
        violations: Vec::new(),
        extra: Vec::new(),
    };
    rep.table.push(format!(
        "{:>24} {}",
        "workload",
        lanes
            .iter()
            .enumerate()
            .map(|(i, l)| match i {
                0 => format!("{:>22}", format!("{}{unit}", l.name)),
                _ if chain && i > 1 => format!(
                    "{:>22} {:>22} {:>22}",
                    format!("{}{unit}", l.name),
                    "vs base [lo, hi]",
                    "vs prev [lo, hi]"
                ),
                _ => format!("{:>22} {:>22}", format!("{}{unit}", l.name), "vs base [lo, hi]"),
            })
            .collect::<Vec<_>>()
            .join(" ")
    ));
    // Report in the order the base lane printed in round 1.
    for id in &runs[0][0].order {
        let w = &id.0;
        let get = |r: usize, l: usize, f: &str| -> Res<Option<f64>> {
            match runs[r][l].fields.get(id) {
                None => Err(format!(
                    "lane {} did not report workload {w:?} in round {}",
                    lanes[l].name,
                    r + 1
                )),
                Some(m) => Ok(m.get(f).copied()),
            }
        };
        let series: Vec<Vec<f64>> = (0..lanes.len())
            .map(|l| {
                (0..o.rounds)
                    .map(|r| get(r, l, primary).map(|v| v.expect("checked in the lane parser")))
                    .collect::<Res<Vec<f64>>>()
            })
            .collect::<Res<_>>()?;
        let mut obj = vec![("workload".to_string(), Jv::str(w.clone()))];
        let mut line = format!("{w:>24}");
        for (l, lane) in lanes.iter().enumerate() {
            let all = &series[l];
            let med = median(all.clone())?;
            let lo = all.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = all.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut fields = vec![
                (format!("{primary}_median"), Jv::Num(med)),
                (format!("{primary}_min"), Jv::Num(lo)),
                (format!("{primary}_max"), Jv::Num(hi)),
                (format!("{primary}_all"), Jv::nums(all)),
            ];
            if o.format == Format::Workloads {
                for extra in EXTRA_FIELDS {
                    let vals: Vec<f64> = (0..o.rounds)
                        .map(|r| get(r, l, extra))
                        .collect::<Res<Vec<_>>>()?
                        .into_iter()
                        .flatten()
                        .collect();
                    if !vals.is_empty() {
                        fields.push((
                            format!("{extra}_max"),
                            Jv::Num(vals.iter().copied().fold(f64::MIN, f64::max)),
                        ));
                    }
                }
            }
            line += &format!(" {med:>22.3}");
            let mut versus = |name: &str, base: &[f64], fields: &mut Vec<(String, Jv)>, line: &mut String| -> Res<()> {
                let label = format!("{} {name} {}", lane.name, w);
                let s = ratio_series(all, base, &label)?;
                gate(&s, o.max_spread, &label, &mut rep.violations)?;
                fields.push((
                    name.to_string(),
                    Jv::obj([
                        ("median", Jv::Num(s.median)),
                        ("lo", Jv::Num(s.min)),
                        ("hi", Jv::Num(s.max)),
                    ]),
                ));
                *line += &format!(" {:>22}", format!("{:.3}x [{:.3}, {:.3}]", s.median, s.min, s.max));
                Ok(())
            };
            if l > 0 {
                versus("vs_base", &series[0], &mut fields, &mut line)?;
            }
            if chain && l > 1 {
                versus("vs_prev", &series[l - 1], &mut fields, &mut line)?;
            }
            obj.push((lane.name.clone(), Jv::Obj(fields)));
        }
        rep.summary.push(Jv::Obj(obj));
        rep.table.push(line);
    }
    // Workloads a lane reports that the base does not are not compared; say so.
    for (l, lane) in lanes.iter().enumerate().skip(1) {
        let extra: Vec<&str> = runs[0][l]
            .order
            .iter()
            .filter(|id| !runs[0][0].fields.contains_key(*id))
            .map(|id| id.0.as_str())
            .collect();
        if !extra.is_empty() {
            eprintln!(
                "note: lane {} reported {} workload(s) the base lane did not, not compared: {}",
                lane.name,
                extra.len(),
                extra.join(", ")
            );
        }
    }
    Ok(rep)
}

/// `(key, variant)` pairs absent from at least one round. Formed before any
/// ratio is, because a shape missing from one side is a measurement that did
/// not happen, and a geomean that quietly drops it reads like one over all.
fn missing_coverage(per_round: &[BTreeMap<Id, f64>], keys: &[String], variants: &[&str]) -> Vec<String> {
    let mut gaps = BTreeSet::new();
    for key in keys {
        for round in per_round {
            for v in variants {
                if !round.contains_key(&(key.clone(), v.to_string())) {
                    gaps.insert(format!("{key}/{v}"));
                }
            }
        }
    }
    gaps.into_iter().collect()
}

/// Rows: each `--compare A:B` as a per-key ratio series and a paired geomean.
fn report_rows(o: &Options, runs: &[Vec<LaneRun>]) -> Res<Report> {
    let mut variant_lane: BTreeMap<String, usize> = BTreeMap::new();
    for round in runs {
        for (l, run) in round.iter().enumerate() {
            for (_, var) in run.fields.keys() {
                if let Some(prev) = variant_lane.insert(var.clone(), l) {
                    if prev != l {
                        return Err(format!(
                            "variant {var:?} reported by both lane {} and lane {}",
                            o.lanes[prev].name, o.lanes[l].name
                        ));
                    }
                }
            }
        }
    }
    let per_round: Vec<BTreeMap<Id, f64>> = runs
        .iter()
        .map(|round| {
            round
                .iter()
                .flat_map(|run| run.fields.iter().map(|(id, f)| (id.clone(), f["value"])))
                .collect()
        })
        .collect();
    let keys: Vec<String> = match &o.keys {
        Some(k) => k.clone(),
        None => {
            let mut seen = Vec::new();
            for id in runs.iter().flatten().flat_map(|r| r.order.iter()) {
                if !seen.contains(&id.0) {
                    seen.push(id.0.clone());
                }
            }
            seen
        }
    };
    let mut rep = Report {
        table: Vec::new(),
        summary: Vec::new(),
        violations: Vec::new(),
        extra: Vec::new(),
    };
    for (a, b) in &o.compares {
        for v in [a, b] {
            if !variant_lane.contains_key(v) {
                return Err(format!(
                    "--compare {a}:{b}: no lane reported variant {v:?}; reported: {:?}",
                    variant_lane.keys().collect::<Vec<_>>()
                ));
            }
        }
        let missing = missing_coverage(&per_round, &keys, &[a, b]);
        if !missing.is_empty() {
            return Err(format!(
                "{a}:{b}: {} (key, variant) pair(s) absent from at least one round -- {:?}. Refusing to \
                 report a geomean over part of the ladder as if it covered all of it.",
                missing.len(),
                &missing[..missing.len().min(6)]
            ));
        }
        let label = format!("{a} / {b}");
        let value = |round: usize, key: &String, v: &String| per_round[round][&(key.clone(), v.clone())];
        let mut per_key = Vec::new();
        let mut ratios_by_round: Vec<Vec<f64>> = vec![Vec::new(); o.rounds];
        rep.table.push(format!(
            "\n{label}   ({} alternating rounds, {}/{} keys)",
            o.rounds,
            keys.len(),
            keys.len()
        ));
        rep.table
            .push(format!("  {:<24}{:>10}{:>10}{:>10}", "key", "median", "min", "max"));
        for key in &keys {
            let (av, bv): (Vec<f64>, Vec<f64>) = (0..o.rounds).map(|r| (value(r, key, a), value(r, key, b))).unzip();
            let klabel = format!("{label}/{key}");
            let s = ratio_series(&av, &bv, &klabel)?;
            gate(&s, o.max_spread, &klabel, &mut rep.violations)?;
            for (r, (x, y)) in av.iter().zip(&bv).enumerate() {
                ratios_by_round[r].push(x / y);
            }
            rep.table
                .push(format!("  {key:<24}{:>9.3}x{:>9.3}x{:>9.3}x", s.median, s.min, s.max));
            per_key.push(Jv::obj([
                ("key", Jv::str(key.clone())),
                ("ratio", summary_json(&s)),
                (
                    "a_value",
                    summary_json(&evidence::series_summary(&av, &format!("{klabel}/{a}"))?),
                ),
                (
                    "b_value",
                    summary_json(&evidence::series_summary(&bv, &format!("{klabel}/{b}"))?),
                ),
            ]));
        }
        let geos: Vec<f64> = ratios_by_round
            .iter()
            .enumerate()
            .map(|(r, v)| evidence::geometric_mean(v, &format!("{label}/round-{}", r + 1)))
            .collect::<Res<_>>()?;
        let glabel = format!("{label}/paired-geomean");
        let g = evidence::series_summary(&geos, &glabel)?;
        gate(&g, o.max_spread, &glabel, &mut rep.violations)?;
        rep.table.push(format!(
            "  {:<24}{:>9.3}x   range {:.3}x..{:.3}x",
            "PAIRED GEOMEAN", g.median, g.min, g.max
        ));
        rep.summary.push(Jv::obj([
            ("label", Jv::str(label)),
            ("a", Jv::str(a.clone())),
            ("b", Jv::str(b.clone())),
            ("ratio_direction", Jv::str("a / b per round")),
            ("keys_requested", Jv::Int(keys.len() as u64)),
            ("keys_covered", Jv::Int(keys.len() as u64)),
            ("paired_geomean", summary_json(&g)),
            ("per_key", Jv::Arr(per_key)),
        ]));
    }
    rep.extra.push((
        "keys".into(),
        Jv::Arr(keys.iter().map(|k| Jv::str(k.clone())).collect()),
    ));
    Ok(rep)
}

fn lane_json(lane: &Lane) -> Jv {
    Jv::obj([
        ("name", Jv::str(lane.name.clone())),
        ("argv", Jv::Arr(lane.argv.iter().map(|a| Jv::str(a.clone())).collect())),
        (
            "env",
            Jv::Obj(lane.env.iter().map(|(k, v)| (k.clone(), Jv::str(v.clone()))).collect()),
        ),
    ])
}

fn methodology(o: &Options) -> Jv {
    let rows = matches!(o.format, Format::Rows { .. });
    Jv::obj([
        ("ratio_pairing", Jv::str("same outer round")),
        (
            "execution_order",
            Jv::str("lanes forward on even rounds, reversed on odd rounds"),
        ),
        ("central_estimate", Jv::str("median of paired per-round ratios")),
        (
            "canonical_aggregate",
            Jv::str(if rows {
                "median of per-round geometric means"
            } else {
                "none; per workload"
            }),
        ),
        (
            "recorded_round_value",
            Jv::str("one child-process aggregate per lane per round; inner iteration timings are not emitted"),
        ),
        (
            "drift_gate",
            Jv::str("max/min paired ratio spread per series and aggregate"),
        ),
        ("environment_probe_policy", Jv::str(evidence::ENVIRONMENT_PROBE_POLICY)),
    ])
}

/// The files whose bytes a run is built from: each lane's executable (which
/// embeds the metallib) and every argument that names an existing file (a
/// Python lane's script).
fn provenance_files(lanes: &[Lane]) -> Vec<(String, PathBuf)> {
    let mut files = Vec::new();
    for lane in lanes {
        for (i, arg) in lane.argv.iter().enumerate() {
            let p = Path::new(arg);
            if (i == 0 && arg.contains('/')) || (i > 0 && p.is_file()) {
                let base = p.file_name().map_or(arg.clone(), |n| n.to_string_lossy().into_owned());
                files.push((format!("lane:{}:{base}", lane.name), p.to_path_buf()));
            }
        }
    }
    files
}

fn real_main(args: &[String]) -> Result<(), Fail> {
    // The marker goes down before anything else is validated, so even a
    // malformed rerun makes a preserved older output visibly stale.
    let mut publisher = None;
    if args.iter().any(|a| a == "--evidence") {
        if let Some(path) = evidence::requested_output_path(args) {
            let p = evidence::EvidenceOutput::new(Path::new(&path), args)?;
            p.begin()?;
            publisher = Some(p);
        }
    }
    let o = parse_args(args).map_err(Fail::Usage)?;
    let inherited: Vec<(String, String)> = std::env::vars().collect();
    let envs: Vec<BTreeMap<String, String>> = o
        .lanes
        .iter()
        .map(|l| lane_env(l, inherited.clone(), o.clean_env))
        .collect();
    for (lane, env) in o.lanes.iter().zip(&envs) {
        let exe = &lane.argv[0];
        if exe.contains('/') && !Path::new(exe).is_file() {
            return Err(Fail::Usage(format!(
                "lane {}: {exe} is missing or not a file",
                lane.name
            )));
        }
        if o.evidence {
            let iters = lane_iters(env).map_err(Fail::Usage)?;
            evidence::validate_evidence_sample_counts(o.rounds, iters).map_err(Fail::Usage)?;
        }
    }
    let mut provenance = None;
    if o.evidence {
        let config = Jv::obj([
            ("rounds", Jv::Int(o.rounds as u64)),
            (
                "max_spread",
                Jv::Num(o.max_spread.unwrap_or(evidence::DEFAULT_MAX_RATIO_SPREAD)),
            ),
            ("clean_env", Jv::Bool(o.clean_env)),
            ("format", Jv::str(format!("{:?}", o.format))),
            ("lanes", Jv::Arr(o.lanes.iter().map(lane_json).collect())),
            (
                "compare",
                Jv::Arr(o.compares.iter().map(|(a, b)| Jv::str(format!("{a}:{b}"))).collect()),
            ),
        ]);
        let mut files = provenance_files(&o.lanes);
        files.sort();
        files.dedup();
        provenance = Some(evidence::start_provenance(&evidence::ProvenanceInputs {
            argv: args,
            scope: Path::new("."),
            files,
            config,
            probe_device: true,
        })?);
        for lane in &o.lanes {
            if lane.argv[0].rsplit('/').next().is_some_and(|n| n.starts_with("python")) {
                if let Some(Jv::Obj(runtime)) = provenance.as_mut().and_then(|p| match p {
                    Jv::Obj(f) => f.iter_mut().find(|(k, _)| k == "runtime").map(|(_, v)| v),
                    _ => None,
                }) {
                    runtime.push((
                        format!("python_packages:{}", lane.name),
                        evidence::python_packages(&lane.argv[0]),
                    ));
                }
            }
        }
    }

    // runs[round][lane]
    let mut runs: Vec<Vec<LaneRun>> = Vec::with_capacity(o.rounds);
    for r in 0..o.rounds {
        let mut order: Vec<usize> = (0..o.lanes.len()).collect();
        if r % 2 == 1 {
            order.reverse();
        }
        let mut this: Vec<Option<LaneRun>> = (0..o.lanes.len()).map(|_| None).collect();
        for i in order {
            this[i] = Some(run_lane(&o.lanes[i], &envs[i], &o)?);
        }
        runs.push(this.into_iter().map(|x| x.expect("every lane ran")).collect());
        eprintln!("round {}/{} done", r + 1, o.rounds);
    }

    let mut report = match o.format {
        Format::Rows { .. } => report_rows(&o, &runs)?,
        _ => report_lanes(&o, &runs)?,
    };
    // Numbers are presented only after every drift gate has passed.
    if !report.violations.is_empty() {
        return Err(Fail::Drift(std::mem::take(&mut report.violations)));
    }
    println!("{}", report.table.join("\n"));

    if let Some(path) = &o.out {
        let raw: Vec<Jv> = runs
            .iter()
            .map(|round| {
                Jv::Obj(
                    round
                        .iter()
                        .zip(&o.lanes)
                        .map(|(run, l)| (l.name.clone(), run.raw.clone()))
                        .collect(),
                )
            })
            .collect();
        let env_json: Vec<(String, Jv)> = ["BENCH_WORKLOADS", "BENCH_ITERS", "BENCH_WARMUP"]
            .iter()
            .filter_map(|k| std::env::var(k).ok().map(|v| (k.to_string(), Jv::str(v))))
            .collect();
        let mut doc = vec![
            ("rounds".to_string(), Jv::Int(o.rounds as u64)),
            ("format".into(), Jv::str(format!("{:?}", o.format))),
            ("lanes".into(), Jv::Arr(o.lanes.iter().map(lane_json).collect())),
            ("env".into(), Jv::Obj(env_json)),
        ];
        doc.extend(report.extra);
        if let Some(spread) = o.max_spread {
            doc.push(("max_ratio_spread".into(), Jv::Num(spread)));
        }
        if o.evidence {
            doc.push(("methodology".into(), methodology(&o)));
        }
        doc.push((
            if matches!(o.format, Format::Rows { .. }) {
                "comparisons"
            } else {
                "summary"
            }
            .into(),
            Jv::Arr(report.summary),
        ));
        doc.push(("raw".into(), Jv::Arr(raw)));
        if let Some(mut prov) = provenance {
            evidence::finish_provenance(&mut prov);
            doc.push(("provenance".into(), prov));
        }
        let doc = Jv::Obj(doc);
        match publisher.as_mut() {
            Some(p) => {
                p.publish(&doc)?;
                eprintln!(
                    "wrote {} (attempt marker {})",
                    path.display(),
                    p.marker_path().display()
                );
            }
            None => {
                evidence::atomic_write(path, doc.render()?.as_bytes())?;
                eprintln!("wrote {}", path.display());
            }
        }
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match real_main(&args) {
        Ok(()) => {}
        Err(Fail::Usage(e)) => {
            eprintln!("bench_paired: {e}");
            std::process::exit(2);
        }
        Err(Fail::Run(e)) => {
            eprintln!("bench_paired: {e}");
            std::process::exit(1);
        }
        Err(Fail::Drift(v)) => {
            eprintln!(
                "bench_paired: {} ratio series moved more than the bounded spread, so none is reported:",
                v.len()
            );
            for line in v {
                eprintln!("  {line}");
            }
            std::process::exit(3);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn lanes_parse_with_and_without_environment() {
        let l = parse_lane("tessl=target/release/bench --flag").unwrap();
        assert_eq!((l.name.as_str(), l.argv.len(), l.env.len()), ("tessl", 2, 0));
        let l = parse_lane("mid:TESSL_MID_COMMIT=3,BENCH_ITERS=5=bin a=b").unwrap();
        assert_eq!(l.name, "mid");
        assert_eq!(
            l.env,
            vec![
                ("TESSL_MID_COMMIT".into(), "3".into()),
                ("BENCH_ITERS".into(), "5".into())
            ]
        );
        assert_eq!(l.argv, vec!["bin", "a=b"]);
        // An empty value unsets; a colon after the first '=' belongs to the command.
        assert_eq!(parse_lane("x:K==cmd").unwrap().env, vec![("K".into(), String::new())]);
        assert_eq!(parse_lane("x=/p:q/bin").unwrap().argv, vec!["/p:q/bin"]);
        let l = parse_lane("x:S=\"1x1,2x2\",N='a=b',Z=1=bin").unwrap();
        assert_eq!(
            l.env,
            vec![
                ("S".into(), "1x1,2x2".into()),
                ("N".into(), "a=b".into()),
                ("Z".into(), "1".into())
            ]
        );
        assert!(parse_lane("x:S=\"open=bin").is_err());
        assert!(parse_lane("x:S=\"v\"junk=bin").is_err());
        for bad in [
            "noeq",
            "=cmd",
            "n=",
            "n:K=1",
            "n:K=1,=cmd",
            "n:1K=1=cmd",
            "n:K=1=",
            "n=\"open",
        ] {
            assert!(parse_lane(bad).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn quoted_words_may_hold_spaces() {
        assert_eq!(
            split_command("\"/a b/bin\" --x 'y z' ''").unwrap(),
            vec!["/a b/bin", "--x", "y z", ""]
        );
        assert!(split_command("   ").unwrap().is_empty());
    }

    #[test]
    fn lane_environment_overrides_unsets_and_cleans() {
        let inherited =
            [("PATH", "/bin"), ("BENCH_ITERS", "9"), ("TESSL_X", "1")].map(|(k, v)| (k.to_string(), v.to_string()));
        let lane = parse_lane("a:BENCH_ITERS=3,TESSL_X==bin").unwrap();
        let env = lane_env(&lane, inherited.clone(), false);
        assert_eq!(env["BENCH_ITERS"], "3");
        assert!(!env.contains_key("TESSL_X") && env["PATH"] == "/bin");
        let plain = parse_lane("a=bin").unwrap();
        let cleaned = lane_env(&plain, inherited, true);
        assert_eq!(cleaned.keys().collect::<Vec<_>>(), vec!["PATH"]);
    }

    #[test]
    fn metric_lines_are_strict() {
        let run = parse_metrics("l", "noise\nMETRIC a/x 1.5\nMETRIC b 2\n").unwrap();
        assert_eq!(
            run.order,
            vec![("a/x".into(), String::new()), ("b".into(), String::new())]
        );
        assert_eq!(run.fields[&("a/x".to_string(), String::new())]["value"], 1.5);
        for bad in [
            "hi\n",
            "METRIC a\n",
            "METRIC a x\n",
            "METRIC a NaN\n",
            "METRIC a 1\nMETRIC a 2\n",
        ] {
            assert!(parse_metrics("l", bad).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn workload_rows_need_a_positive_ms_min() {
        let ok = parse_workloads("l", "log\n[{\"workload\":\"w\",\"ms_min\":2.5,\"forwards\":3}]\n").unwrap();
        assert_eq!(ok.fields[&("w".to_string(), String::new())]["forwards"], 3.0);
        for bad in [
            "[{\"workload\":\"w\"}]",
            "[{\"workload\":\"w\",\"ms_min\":0}]",
            "[{\"workload\":\"w\",\"ms_min\":1},{\"workload\":\"w\",\"ms_min\":1}]",
            "{\"workload\":\"w\"}",
            "",
        ] {
            assert!(parse_workloads("l", bad).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn rows_require_the_same_batching_and_refuse_duplicates() {
        let rows = |s: &str| parse_rows("l", s, "cfg", "runtime", "median_ms", &[("batched".into(), 32.0)]);
        let good = "[{\"cfg\":\"c\",\"runtime\":\"t\",\"median_ms\":1.5,\"batched\":32}]";
        assert!(rows(good).is_ok());
        let wrong = rows("[{\"cfg\":\"c\",\"runtime\":\"t\",\"median_ms\":1.5,\"batched\":1}]")
            .err()
            .unwrap();
        assert!(wrong.contains("batched=1") && wrong.contains("32"), "{wrong}");
        let silent = rows("[{\"cfg\":\"c\",\"runtime\":\"t\",\"median_ms\":1.5}]")
            .err()
            .unwrap();
        assert!(silent.contains("cannot be confirmed"), "{silent}");
        let dup = "[{\"cfg\":\"c\",\"runtime\":\"t\",\"median_ms\":1,\"batched\":32},\
                   {\"cfg\":\"c\",\"runtime\":\"t\",\"median_ms\":2,\"batched\":32}]";
        assert!(rows(dup).err().unwrap().contains("duplicate"));
        assert!(rows("[{\"cfg\":\"c\",\"runtime\":\"t\",\"median_ms\":-1,\"batched\":32}]").is_err());
        assert!(rows("not json").is_err());
        // The row key may be numeric in the lane's output.
        assert!(parse_rows("l", "[{\"n\":3,\"v\":\"a\",\"x\":2}]", "n", "v", "x", &[]).is_ok());
    }

    #[test]
    fn a_missing_shape_in_any_round_is_named() {
        let round = |ids: &[(&str, &str)]| -> BTreeMap<Id, f64> {
            ids.iter().map(|(k, v)| ((k.to_string(), v.to_string()), 1.0)).collect()
        };
        let keys = ["s1".to_string(), "s2".to_string()];
        let full = round(&[("s1", "t"), ("s2", "t"), ("s1", "o"), ("s2", "o")]);
        assert!(missing_coverage(&[full.clone(), full.clone()], &keys, &["t", "o"]).is_empty());
        let holed = round(&[("s1", "t"), ("s2", "t"), ("s1", "o")]);
        assert_eq!(missing_coverage(&[full, holed], &keys, &["t", "o"]), vec!["s2/o"]);
    }

    fn rows_options(keys: Option<Vec<String>>, spread: Option<f64>) -> Options {
        Options {
            rounds: 4,
            out: None,
            format: Format::Rows {
                key: "k".into(),
                variant: "v".into(),
                value: "x".into(),
            },
            compares: vec![("t".into(), "o".into())],
            keys,
            require: Vec::new(),
            max_spread: spread,
            clean_env: false,
            evidence: false,
            lanes: vec![parse_lane("tessl=a").unwrap(), parse_lane("other=b").unwrap()],
        }
    }

    fn rows_run(vals: &[(&str, &str, f64)]) -> LaneRun {
        let mut fields = BTreeMap::new();
        let mut order = Vec::new();
        for (k, v, x) in vals {
            let id = (k.to_string(), v.to_string());
            fields.insert(id.clone(), BTreeMap::from([("value".to_string(), *x)]));
            order.push(id);
        }
        LaneRun {
            fields,
            order,
            raw: Jv::Null,
        }
    }

    #[test]
    fn rows_report_paired_ratios_geomean_and_the_drift_gate() {
        // Ratios t/o per round: shape a is 2.0 throughout, shape b drifts 1..1.5.
        let drift = [1.0, 1.1, 1.3, 1.5];
        let runs: Vec<Vec<LaneRun>> = drift
            .iter()
            .map(|b| {
                vec![
                    rows_run(&[("a", "t", 4.0), ("b", "t", *b)]),
                    rows_run(&[("a", "o", 2.0), ("b", "o", 1.0)]),
                ]
            })
            .collect();
        let o = rows_options(None, None);
        let rep = report_rows(&o, &runs).unwrap();
        assert!(rep.violations.is_empty());
        assert!(rep.table.iter().any(|l| l.contains("PAIRED GEOMEAN")));
        // geomean per round: sqrt(2 * b); median over rounds of the four.
        let want = {
            let mut g: Vec<f64> = drift.iter().map(|b| (2.0 * b).sqrt()).collect();
            g.sort_by(f64::total_cmp);
            (g[1] + g[2]) / 2.0
        };
        let Jv::Obj(cmp) = &rep.summary[0] else {
            panic!("comparison is an object")
        };
        let Some((_, Jv::Obj(geo))) = cmp.iter().find(|(k, _)| k == "paired_geomean") else {
            panic!("no geomean")
        };
        let Some((_, Jv::Num(got))) = geo.iter().find(|(k, _)| k == "median") else {
            panic!("no median")
        };
        assert!((got - want).abs() < 1e-12, "paired geomean median {got}, want {want}");

        let gated = report_rows(&rows_options(None, Some(1.10)), &runs).unwrap();
        assert!(
            gated.violations.iter().any(|v| v.contains("t / o/b")),
            "{:?}",
            gated.violations
        );
        assert!(gated.violations.iter().any(|v| v.contains("paired-geomean")));
    }

    #[test]
    fn rows_refuse_partial_coverage_unknown_variants_and_shared_variants() {
        let full = || {
            vec![
                rows_run(&[("a", "t", 2.0), ("b", "t", 2.0)]),
                rows_run(&[("a", "o", 1.0), ("b", "o", 1.0)]),
            ]
        };
        let mut runs: Vec<Vec<LaneRun>> = (0..4).map(|_| full()).collect();
        runs[2][1] = rows_run(&[("a", "o", 1.0)]);
        let err = report_rows(&rows_options(Some(vec!["a".into(), "b".into()]), None), &runs)
            .err()
            .unwrap();
        assert!(err.contains("b/o") && err.contains("Refusing"), "{err}");
        // A key the lanes dropped in every round is only caught by --keys.
        let all: Vec<Vec<LaneRun>> = (0..4).map(|_| full()).collect();
        let err = report_rows(
            &rows_options(Some(vec!["a".into(), "b".into(), "c".into()]), None),
            &all,
        )
        .err()
        .unwrap();
        assert!(err.contains("c/t"), "{err}");
        let mut unknown = rows_options(None, None);
        unknown.compares = vec![("t".into(), "ghost".into())];
        assert!(report_rows(&unknown, &all).err().unwrap().contains("ghost"));
        let shared: Vec<Vec<LaneRun>> = (0..4)
            .map(|_| {
                vec![
                    rows_run(&[("a", "t", 2.0)]),
                    rows_run(&[("a", "t", 1.0), ("a", "o", 1.0)]),
                ]
            })
            .collect();
        assert!(report_rows(&rows_options(None, None), &shared)
            .err()
            .unwrap()
            .contains("both lane"));
    }

    #[test]
    fn lane_reports_give_median_ratios_and_flag_a_workload_a_lane_dropped() {
        let o = Options {
            format: Format::Metrics,
            compares: Vec::new(),
            ..rows_options(None, Some(1.10))
        };
        let m = |a: f64, b: f64| rows_run(&[("m1", "", a), ("m2", "", b)]);
        let runs: Vec<Vec<LaneRun>> = (0..4).map(|_| vec![m(2.0, 4.0), m(1.0, 2.0)]).collect();
        let rep = report_lanes(&o, &runs).unwrap();
        assert!(rep.violations.is_empty());
        assert!(rep.table[1].contains("0.500x [0.500, 0.500]"), "{:?}", rep.table);
        let mut dropped: Vec<Vec<LaneRun>> = (0..4).map(|_| vec![m(2.0, 4.0), m(1.0, 2.0)]).collect();
        dropped[1][1] = rows_run(&[("m1", "", 1.0)]);
        assert!(report_lanes(&o, &dropped)
            .err()
            .unwrap()
            .contains("did not report workload"));
    }

    #[test]
    fn argument_policy() {
        let two = ["a=x", "b=y"];
        let with = |extra: &[&str]| parse_args(&args(&[extra, &two].concat()));
        assert_eq!(with(&[]).unwrap().rounds, 5);
        assert!(with(&["--rounds", "1"]).is_err());
        for bad in [
            "0", "00", "-1", "", "abc", "1.5", "+2", " 2", "2 ", "007", "0x10", "1e3", "1000000", "\u{661}", "1\n2",
        ] {
            assert!(parse_rounds(bad).is_err(), "--rounds {bad:?} accepted");
        }
        assert_eq!(parse_rounds("12").unwrap(), 12);
        assert!(parse_args(&args(&["a=x"])).is_err());
        assert!(with(&["--format", "bogus"]).is_err());
        assert!(with(&["--format", "rows:k,v,x"]).is_err(), "rows needs --compare");
        assert!(with(&["--compare", "t:o"]).is_err(), "--compare needs rows");
        assert!(with(&["--format", "rows:k,v,x", "--compare", "t:t"]).is_err());
        assert!(with(&["--evidence"]).is_err(), "--evidence needs --out");
        assert!(with(&["--max-spread", "1.3"]).is_err());
        assert!(with(&["--bogus"]).is_err());
        let ev = with(&["--evidence", "--out", "o.json"]).unwrap();
        assert_eq!((ev.rounds, ev.max_spread), (6, Some(1.10)));
        assert!(parse_args(&args(&["a=x", "a=y"])).is_err(), "duplicate lane name");
    }
}
