//! Alternate benchmark lanes round by round, so clock drift and warm-up land
//! on every lane alike instead of in the ratio between them.
//!
//! ```text
//! cargo build --release --bin bench_embedgemma2
//! EMBEDGEMMA2_SNAPSHOT=... BENCH_ITERS=10 BENCH_WARMUP=3 \
//!   target/release/bench_paired --rounds 5 --out bench/results/embedgemma2_<machine>.json \
//!     'tessl-f32=target/release/bench_embedgemma2' \
//!     'torch-mps-f32=/path/to/python bench/embedgemma2_torch.py --dtype f32'
//! ```
//!
//! Each `NAME=COMMAND` is a lane; the command is split on whitespace (no
//! quoting) and runs from the current directory with this process's
//! environment, so `BENCH_WORKLOADS`, `BENCH_ITERS` and `BENCH_WARMUP` reach
//! every lane. A lane prints, as the last line of its stdout, a JSON array of
//! objects, each with a string `workload` and a number `ms_min` (the
//! workload's fastest iteration in that process: min-of-N). Other numeric
//! fields (`device_peak_mib`, `peak_footprint_mib`, `forwards`) are carried
//! into the report. `bench_embedgemma2` and `bench/embedgemma2_torch.py` both
//! print this; two builds of `bench_embedgemma2` make a before/after A/B.
//!
//! Every round runs every lane once as a fresh process, in the given order on
//! even rounds and reversed on odd ones. Per lane and workload the report
//! gives the median over rounds of `ms_min`; per lane after the first (the
//! base) it gives the median of the per-round ratios `ms_min(lane) /
//! ms_min(base)` with the lowest and highest round as the spread. Below 1 the
//! lane is faster. A spread that straddles 1.0 is not a result. With three or
//! more lanes it also gives each lane against the one before it (`vs_prev`),
//! so a chain of builds, one change each, benches every change separately in
//! one interleaved run. A workload a lane did not report in some round is an
//! error, not a gap.

mod common;

#[path = "../json.rs"]
#[allow(dead_code)]
mod json;

use std::collections::BTreeMap;
use std::process::Command;

use common::median;
use json::{Json, Syntax};

type Res<T> = Result<T, String>;

const SYNTAX: Syntax = Syntax {
    what: "lane output",
    max_depth: 4,
    uints_only: false,
    literals: true,
};

struct Lane {
    name: String,
    argv: Vec<String>,
}

/// One lane's numbers in one round: workload -> field -> value, plus the
/// line it printed (kept verbatim in the report).
struct LaneRun {
    fields: BTreeMap<String, BTreeMap<String, f64>>,
    raw: String,
}

fn run_lane(lane: &Lane) -> Res<LaneRun> {
    let out = Command::new(&lane.argv[0])
        .args(&lane.argv[1..])
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
    let raw = stdout
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| format!("lane {} printed nothing", lane.name))?
        .trim()
        .to_string();
    let Json::Array(rows) = json::parse(&raw, SYNTAX).map_err(|e| format!("lane {}: {e}", lane.name))? else {
        return Err(format!("lane {}: the last line is not a JSON array", lane.name));
    };
    let mut fields = BTreeMap::new();
    for row in &rows {
        let Some(Json::Str(w)) = row.get("workload") else {
            return Err(format!("lane {}: a row has no string \"workload\"", lane.name));
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
            _ => return Err(format!("lane {}: workload {w:?} has no positive \"ms_min\"", lane.name)),
        }
        if fields.insert(w.clone(), nums).is_some() {
            return Err(format!("lane {}: workload {w:?} reported twice", lane.name));
        }
    }
    Ok(LaneRun { fields, raw })
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn json_nums(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(","))
}

fn main() -> Res<()> {
    let mut rounds = 5usize;
    let mut out_path = None;
    let mut lanes = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--rounds" => {
                let v = args.next().ok_or("--rounds needs a value")?;
                rounds = v.parse().map_err(|e| format!("--rounds {v}: {e}"))?;
            }
            "--out" => out_path = Some(args.next().ok_or("--out needs a path")?),
            _ => {
                let (name, cmd) = a
                    .split_once('=')
                    .ok_or_else(|| format!("{a:?}: expected NAME=COMMAND, --rounds N or --out PATH"))?;
                let argv: Vec<String> = cmd.split_whitespace().map(String::from).collect();
                if name.is_empty() || argv.is_empty() {
                    return Err(format!("{a:?}: empty lane name or command"));
                }
                if lanes.iter().any(|l: &Lane| l.name == name) {
                    return Err(format!("lane {name:?} given twice"));
                }
                lanes.push(Lane {
                    name: name.to_string(),
                    argv,
                });
            }
        }
    }
    if lanes.len() < 2 {
        return Err("give at least two NAME=COMMAND lanes; the first is the base".into());
    }
    if rounds < 2 {
        return Err("--rounds must be at least 2: a single round has no spread".into());
    }

    // runs[round][lane]
    let mut runs: Vec<Vec<LaneRun>> = Vec::with_capacity(rounds);
    for r in 0..rounds {
        let mut order: Vec<usize> = (0..lanes.len()).collect();
        if r % 2 == 1 {
            order.reverse();
        }
        let mut this: Vec<Option<LaneRun>> = (0..lanes.len()).map(|_| None).collect();
        for i in order {
            this[i] = Some(run_lane(&lanes[i])?);
        }
        runs.push(this.into_iter().map(|x| x.expect("every lane ran")).collect());
        eprintln!("round {}/{rounds} done", r + 1);
    }

    // Report workloads in the order the base lane printed them in round 1
    // (`LaneRun::fields` is sorted by name).
    let base_order: Vec<String> = {
        let Json::Array(rows) = json::parse(&runs[0][0].raw, SYNTAX)? else {
            unreachable!()
        };
        rows.iter()
            .filter_map(|r| match r.get("workload") {
                Some(Json::Str(w)) => Some(w.clone()),
                _ => None,
            })
            .collect()
    };
    let get = |r: usize, l: usize, w: &str, f: &str| -> Res<Option<f64>> {
        match runs[r][l].fields.get(w) {
            None => Err(format!(
                "lane {} did not report workload {w:?} in round {}",
                lanes[l].name,
                r + 1
            )),
            Some(m) => Ok(m.get(f).copied()),
        }
    };

    let chain = lanes.len() > 2;
    // (median, lowest, highest) of the per-round ratios a / b.
    let ratio = |a: &[f64], b: &[f64]| -> Res<(f64, f64, f64)> {
        let r: Vec<f64> = a.iter().zip(b).map(|(x, y)| x / y).collect();
        let (lo, hi) = (
            r.iter().copied().fold(f64::MAX, f64::min),
            r.iter().copied().fold(f64::MIN, f64::max),
        );
        Ok((median(r)?, lo, hi))
    };
    let mut summary = Vec::new();
    let mut table = vec![format!(
        "{:>24} {}",
        "workload",
        lanes
            .iter()
            .enumerate()
            .map(|(i, l)| match i {
                0 => format!("{:>22}", format!("{} ms", l.name)),
                1 => format!("{:>22} {:>22}", format!("{} ms", l.name), "vs base [lo, hi]"),
                _ if chain => format!(
                    "{:>22} {:>22} {:>22}",
                    format!("{} ms", l.name),
                    "vs base [lo, hi]",
                    "vs prev [lo, hi]"
                ),
                _ => format!("{:>22} {:>22}", format!("{} ms", l.name), "vs base [lo, hi]"),
            })
            .collect::<Vec<_>>()
            .join(" ")
    )];
    for w in &base_order {
        let mut obj = vec![format!("\"workload\":{}", json_str(w))];
        let mut line = format!("{w:>24}");
        let lane_mins: Vec<Vec<f64>> = (0..lanes.len())
            .map(|l| {
                (0..rounds)
                    .map(|r| get(r, l, w, "ms_min").map(|v| v.expect("checked in run_lane")))
                    .collect::<Res<Vec<f64>>>()
            })
            .collect::<Res<_>>()?;
        for (l, lane) in lanes.iter().enumerate() {
            let all = &lane_mins[l];
            let med = median(all.clone())?;
            let mut fields = vec![
                format!("\"ms_min_median\":{med}"),
                format!("\"ms_min_all\":{}", json_nums(all)),
            ];
            for extra in ["device_peak_mib", "peak_footprint_mib", "forwards"] {
                let vals: Vec<f64> = (0..rounds)
                    .map(|r| get(r, l, w, extra))
                    .collect::<Res<Vec<_>>>()?
                    .into_iter()
                    .flatten()
                    .collect();
                if !vals.is_empty() {
                    fields.push(format!(
                        "\"{extra}_max\":{}",
                        vals.iter().copied().fold(f64::MIN, f64::max)
                    ));
                }
            }
            line += &format!(" {med:>22.3}");
            if l > 0 {
                let (rmed, lo, hi) = ratio(all, &lane_mins[0])?;
                fields.push(format!("\"vs_base\":{{\"median\":{rmed},\"lo\":{lo},\"hi\":{hi}}}"));
                line += &format!(" {:>22}", format!("{rmed:.3}x [{lo:.3}, {hi:.3}]"));
            }
            if chain && l > 1 {
                let (rmed, lo, hi) = ratio(all, &lane_mins[l - 1])?;
                fields.push(format!("\"vs_prev\":{{\"median\":{rmed},\"lo\":{lo},\"hi\":{hi}}}"));
                line += &format!(" {:>22}", format!("{rmed:.3}x [{lo:.3}, {hi:.3}]"));
            }
            obj.push(format!("{}:{{{}}}", json_str(&lane.name), fields.join(",")));
        }
        summary.push(format!("{{{}}}", obj.join(",")));
        table.push(line);
    }
    println!("{}", table.join("\n"));

    if let Some(path) = out_path {
        let lanes_json: Vec<String> = lanes
            .iter()
            .map(|l| {
                format!(
                    "{{\"name\":{},\"argv\":[{}]}}",
                    json_str(&l.name),
                    l.argv.iter().map(|a| json_str(a)).collect::<Vec<_>>().join(",")
                )
            })
            .collect();
        let env_json: Vec<String> = ["BENCH_WORKLOADS", "BENCH_ITERS", "BENCH_WARMUP"]
            .iter()
            .filter_map(|k| {
                std::env::var(k)
                    .ok()
                    .map(|v| format!("{}:{}", json_str(k), json_str(&v)))
            })
            .collect();
        let raw: Vec<String> = runs
            .iter()
            .map(|round| {
                format!(
                    "{{{}}}",
                    round
                        .iter()
                        .zip(&lanes)
                        .map(|(run, l)| format!("{}:{}", json_str(&l.name), run.raw))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .collect();
        let doc = format!(
            "{{\"rounds\":{rounds},\"lanes\":[{}],\"env\":{{{}}},\"summary\":[\n{}\n],\"raw\":[\n{}\n]}}\n",
            lanes_json.join(","),
            env_json.join(","),
            summary.join(",\n"),
            raw.join(",\n")
        );
        std::fs::write(&path, doc).map_err(|e| format!("{path}: {e}"))?;
        eprintln!("wrote {path}");
    }
    Ok(())
}
