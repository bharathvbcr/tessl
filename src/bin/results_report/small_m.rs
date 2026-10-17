//! `embedgemma2_small_m_m5pro.json` and the tables its `.txt` quotes.
//!
//! ```text
//! results_report small-m RESULTS_DIR RAW_DIR LANE_BEFORE LANE_AFTER LANE_CONTROL
//! ```
//!
//! RAW_DIR holds, from `bench_paired` (one run per precision and workload group):
//!
//! ```text
//! paired_lat_f32.json    (1x16, 1x64)
//! paired_thr64_f32.json  (64x32)
//! paired_thr8_f32.json   (8x1024)
//! ```
//!
//! and, from `bench_gemm_routes` (`BENCH_LANES=nnsmall`), the stdout of
//! `nnsmall_f32.txt`. It writes `embedgemma2_small_m_m5pro.json` into
//! RESULTS_DIR and prints the tables the `.txt` quotes.
//!
//! Statistics, all computed here from the per-round figures in the raw files:
//!
//! * "all rounds": the median over rounds of the per-round ratio after /
//!   before, with the lowest and highest round, and the same for the control
//!   lane (the before binary run a second time), which is what noise looks like.
//! * "min of rounds": the house method, min over rounds of each lane's
//!   min-of-N, after against before.
//! * "clean rounds": rounds in which the control lane is within 10 % of the
//!   before lane on that workload (nothing else was loading the GPU unevenly
//!   across the three runs of that round); the median ratio over them and how
//!   many there were. A delta whose [lo, hi] over all rounds straddles 1.0 and
//!   whose control spread is as wide is noise.

use crate::py::*;
use std::path::Path;

const CLEAN: f64 = 0.10;

fn ms_all(row: &Val, lane: &str) -> R<Vec<Num>> {
    row.get(lane)?.get("ms_min_all")?.nums()
}

struct Stats {
    rounds: usize,
    before_min: Num,
    after_min: Num,
    control_min: Num,
    before_median: Num,
    after_median: Num,
    ratio_median: f64,
    ratio_lo: f64,
    ratio_hi: f64,
    control_median: f64,
    control_lo: f64,
    control_hi: f64,
    min_ratio: f64,
    clean: Vec<f64>,
}

impl Stats {
    fn clean_stat(&self, f: impl Fn(&[Num]) -> R<Num>) -> R<Val> {
        if self.clean.is_empty() {
            return Ok(Val::Null);
        }
        let nums: Vec<Num> = self.clean.iter().map(|x| F(*x)).collect();
        Ok(Val::num(f(&nums)?))
    }

    fn val(&self) -> R<Val> {
        Ok(Val::obj([
            ("rounds", Val::int(self.rounds)),
            ("before_min", Val::num(self.before_min)),
            ("after_min", Val::num(self.after_min)),
            ("control_min", Val::num(self.control_min)),
            ("before_median", Val::num(self.before_median)),
            ("after_median", Val::num(self.after_median)),
            ("ratio_median", Val::float(self.ratio_median)),
            ("ratio_lo", Val::float(self.ratio_lo)),
            ("ratio_hi", Val::float(self.ratio_hi)),
            ("control_median", Val::float(self.control_median)),
            ("control_lo", Val::float(self.control_lo)),
            ("control_hi", Val::float(self.control_hi)),
            ("min_ratio", Val::float(self.min_ratio)),
            ("clean_rounds", Val::int(self.clean.len())),
            ("clean_ratio_median", self.clean_stat(median)?),
            ("clean_ratio_lo", self.clean_stat(min)?),
            ("clean_ratio_hi", self.clean_stat(max)?),
        ]))
    }

    /// `s["clean_ratio_median"]` as a truth value.
    fn clean_median_truthy(&self) -> R<Option<f64>> {
        if self.clean.is_empty() {
            return Ok(None);
        }
        let nums: Vec<Num> = self.clean.iter().map(|x| F(*x)).collect();
        let m = median(&nums)?;
        Ok(m.truthy().then(|| m.f()))
    }
}

fn stats(row: &Val, lanes: &[String; 3]) -> R<Stats> {
    let b = ms_all(row, &lanes[0])?;
    let a = ms_all(row, &lanes[1])?;
    let c = ms_all(row, &lanes[2])?;
    let ratio = a.iter().zip(&b).map(|(x, y)| x.div(*y)).collect::<R<Vec<Num>>>()?;
    let noise = c.iter().zip(&b).map(|(x, y)| x.div(*y)).collect::<R<Vec<Num>>>()?;
    let clean: Vec<f64> = ratio
        .iter()
        .zip(&noise)
        .filter(|(_, n)| (n.f() - 1.0).abs() <= CLEAN)
        .map(|(r, _)| r.f())
        .collect();
    Ok(Stats {
        rounds: b.len(),
        before_min: min(&b)?,
        after_min: min(&a)?,
        control_min: min(&c)?,
        before_median: median(&b)?,
        after_median: median(&a)?,
        ratio_median: median(&ratio)?.f(),
        ratio_lo: min(&ratio)?.f(),
        ratio_hi: max(&ratio)?.f(),
        control_median: median(&noise)?.f(),
        control_lo: min(&noise)?.f(),
        control_hi: max(&noise)?.f(),
        min_ratio: min(&a)?.div(min(&b)?)?.f(),
        clean,
    })
}

struct Gemm {
    dtype: String,
    name: String,
    m: i128,
    n: i128,
    k: i128,
    arms: OMap<f64>,
    ratios: OMap<[f64; 3]>,
}

fn digits(s: &str) -> Option<(i128, &str)> {
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if end == 0 {
        return None;
    }
    Some((s[..end].parse().ok()?, &s[end..]))
}

fn number_run(s: &str) -> Option<(&str, &str)> {
    let end = s.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(s.len());
    (end > 0).then(|| (&s[..end], &s[end..]))
}

/// `re.match(r"nnsmall (f32|bf16) (.+?) M=(\d+) N=(\d+) K=(\d+)", line)`.
fn header(line: &str) -> Option<Gemm> {
    let rest = line.strip_prefix("nnsmall ")?;
    let (dtype, after) = if let Some(r) = rest.strip_prefix("f32 ") {
        ("f32", r)
    } else {
        ("bf16", rest.strip_prefix("bf16 ")?)
    };
    // `(.+?)`: the shortest name (at least one character) the rest can follow.
    for e in after.char_indices().map(|(i, c)| i + c.len_utf8()) {
        let Some(tail) = after[e..].strip_prefix(" M=") else {
            continue;
        };
        let Some((m, tail)) = digits(tail) else { continue };
        let Some(tail) = tail.strip_prefix(" N=") else { continue };
        let Some((n, tail)) = digits(tail) else { continue };
        let Some(tail) = tail.strip_prefix(" K=") else { continue };
        let Some((k, _)) = digits(tail) else { continue };
        return Some(Gemm {
            dtype: dtype.to_string(),
            name: after[..e].to_string(),
            m,
            n,
            k,
            arms: OMap::default(),
            ratios: OMap::default(),
        });
    }
    None
}

/// `re.match(r"\s+(\S+)\s+([0-9.]+) us\s+plain/this ([0-9.]+) \[([0-9.]+)–([0-9.]+)\]", line)`:
/// the arm, its microseconds and its three ratios.
fn arm_line(line: &str) -> Option<(&str, &str, [&str; 3])> {
    let body = line.trim_start_matches(is_space);
    if body.len() == line.len() {
        return None;
    }
    let name_end = body.find(is_space)?;
    let (name, rest) = body.split_at(name_end);
    let rest = rest.trim_start_matches(is_space);
    let (us, rest) = number_run(rest)?;
    let rest = rest.strip_prefix(" us")?;
    let trimmed = rest.trim_start_matches(is_space);
    if trimmed.len() == rest.len() {
        return None;
    }
    let rest = trimmed.strip_prefix("plain/this ")?;
    let (r0, rest) = number_run(rest)?;
    let rest = rest.strip_prefix(" [")?;
    let (r1, rest) = number_run(rest)?;
    let rest = rest.strip_prefix('\u{2013}')?;
    let (r2, rest) = number_run(rest)?;
    rest.strip_prefix(']')?;
    Some((name, us, [r0, r1, r2]))
}

fn float(s: &str) -> R<f64> {
    s.parse::<f64>()
        .map_err(|_| format!("ValueError: could not convert string to float: {s:?}"))
}

/// `[(label, {arm: us})]` from `bench_gemm_routes` lane nnsmall's stdout.
fn parse_nnsmall(path: &Path) -> R<Vec<Gemm>> {
    let text = read_text(path)?;
    let mut out: Vec<Gemm> = Vec::new();
    for line in splitlines(&text) {
        if let Some(g) = header(line) {
            out.push(g);
            continue;
        }
        if let (Some((arm, us, ratios)), Some(cur)) = (arm_line(line), out.last_mut()) {
            cur.arms.set(arm, float(us)?);
            cur.ratios
                .set(arm, [float(ratios[0])?, float(ratios[1])?, float(ratios[2])?]);
        }
    }
    Ok(out)
}

pub fn run(args: &[String]) -> R<()> {
    let [results, raw, before, after, control] = args else {
        return Err(
            "usage: results_report small-m RESULTS_DIR RAW_DIR LANE_BEFORE LANE_AFTER LANE_CONTROL".to_string(),
        );
    };
    let (results, raw) = (Path::new(results), Path::new(raw));
    let lanes = [before.clone(), after.clone(), control.clone()];

    let prec = "f32";
    let mut rows: Vec<Val> = Vec::new();
    let mut env: Vec<(String, Val)> = Vec::new();
    for grp in ["lat", "thr64", "thr8"] {
        let d = load_file(&raw.join(format!("paired_{grp}_{prec}.json")))?;
        rows.extend(d.get("summary")?.arr()?.iter().cloned());
        let this_env = if d.has("env") {
            d.get("env")?.clone()
        } else {
            Val::Obj(Vec::new())
        };
        env.push((format!("{grp}_{prec}"), this_env));
    }
    let mut stats_by_workload: OMap<Stats> = OMap::default();
    for r in &rows {
        stats_by_workload.set(r.get("workload")?.str()?, stats(r, &lanes)?);
    }
    let workloads: Vec<Val> = rows
        .iter()
        .map(|r| Ok(Val::Str(r.get("workload")?.str()?.to_string())))
        .collect::<R<_>>()?;

    let gemm = parse_nnsmall(&raw.join(format!("nnsmall_{prec}.txt")))?;
    let mut g16 = Vec::new();
    for c in gemm.iter().filter(|c| c.m == 16) {
        let plain = c.arms.get("plain").ok_or("KeyError: \"plain\"")?;
        let production = c.arms.get("production").ok_or("KeyError: \"production\"")?;
        g16.push(Val::obj([
            ("name", Val::Str(c.name.clone())),
            ("plain_us", Val::float(*plain)),
            ("production_us", Val::float(*production)),
        ]));
    }

    let gemm_val = Val::Arr(
        gemm.iter()
            .map(|c| {
                Val::obj([
                    ("dtype", Val::Str(c.dtype.clone())),
                    ("name", Val::Str(c.name.clone())),
                    ("m", Val::num(I(c.m))),
                    ("n", Val::num(I(c.n))),
                    ("k", Val::num(I(c.k))),
                    (
                        "arms",
                        Val::Obj(c.arms.items.iter().map(|(k, v)| (k.clone(), Val::float(*v))).collect()),
                    ),
                    (
                        "ratios",
                        Val::Obj(
                            c.ratios
                                .items
                                .iter()
                                .map(|(k, v)| (k.clone(), Val::Arr(v.iter().map(|x| Val::float(*x)).collect())))
                                .collect(),
                        ),
                    ),
                ])
            })
            .collect(),
    );
    let mut stats_val = Vec::new();
    for (w, s) in &stats_by_workload.items {
        stats_val.push((w.clone(), s.val()?));
    }
    let doc = Val::obj([
        (
            "lane_names",
            Val::obj([
                ("before", Val::Str(before.clone())),
                ("after", Val::Str(after.clone())),
                ("control", Val::Str(control.clone())),
            ]),
        ),
        ("workloads", Val::Arr(workloads)),
        (
            "encode",
            Val::obj([(prec, Val::obj([("summary", Val::Arr(rows.clone()))]))]),
        ),
        ("stats", Val::obj([(prec, Val::Obj(stats_val))])),
        ("env", Val::Obj(env)),
        ("gemm", Val::obj([(prec, gemm_val)])),
        ("gemm_16_rows", Val::obj([(prec, Val::Arr(g16))])),
    ]);
    write_text(&results.join("embedgemma2_small_m_m5pro.json"), &dumps(&doc, Some(1)))?;

    println!("WHOLE ENCODE: after / before (ms = min of rounds' min-of-N)");
    println!(
        "{:5} {:8} {:>6} {:>9} {:>9} {:>14} {:>24} {:>22} {:>8} {:>13}",
        "prec",
        "workload",
        "rounds",
        "before",
        "after",
        "min-of-rounds",
        "median ratio [lo,hi]",
        "control [lo,hi]",
        "clean n",
        "clean median"
    );
    for (w, s) in &stats_by_workload.items {
        let cm = match s.clean_median_truthy()? {
            Some(x) => fixed(x, 3),
            None => "-".to_string(),
        };
        println!(
            "{:5} {:8} {:>6} {:>9} {:>9} {:>14} {:>8} [{},{}] {:>8} [{},{}] {:>8} {:>13}",
            prec,
            w,
            s.rounds,
            fixed(s.before_min.f(), 3),
            fixed(s.after_min.f(), 3),
            fixed(s.min_ratio, 3),
            fixed(s.ratio_median, 3),
            fixed(s.ratio_lo, 2),
            fixed(s.ratio_hi, 2),
            fixed(s.control_median, 3),
            fixed(s.control_lo, 2),
            fixed(s.control_hi, 2),
            s.clean.len(),
            cm
        );
    }
    println!();
    println!(
        "ONE GEMM, us per call (median of interleaved rounds); plain / this = median of per-round ratios [lo, hi]"
    );
    for c in &gemm {
        let mut parts = Vec::new();
        for (k, v) in &c.arms.items {
            let mut part = format!("{k} {}", fixed(*v, 1));
            if k != "plain" {
                let r = c.ratios.get(k).ok_or_else(|| format!("KeyError: {k:?}"))?;
                part.push_str(&format!(
                    " ({}x [{},{}])",
                    fixed(r[0], 2),
                    fixed(r[1], 2),
                    fixed(r[2], 2)
                ));
            }
            parts.push(part);
        }
        println!(
            "{:5} {:<14} M={:<3} K={:<5} N={:<5} {}",
            prec,
            c.name,
            c.m,
            c.k,
            c.n,
            parts.join("  ")
        );
    }
    Ok(())
}
