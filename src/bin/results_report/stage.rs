//! Merge and tabulate the output of `profile_embedgemma2`.
//!
//! ```text
//! results_report stage OUT_DIR PASS.json [PASS.json ...]
//!     [--gemm G.json] [--memory M.json] [--layout L.json] [--ledger LG.json]
//!     [--alloc A.json] [--ple P.json] [--contended PASS_STEM ...]
//! ```
//!
//! Writes `OUT_DIR/embedgemma2_stage_profile_m5pro.json` (every raw sample of
//! every pass, and the other modes' files) and
//! `OUT_DIR/embedgemma2_stage_profile_tables.txt`, and prints the ranking
//! tables. A stage's time is the minimum over every kept sample of every pass;
//! the table also shows the spread of the per-pass minima, which is the honest
//! error bar on this machine. A pass is named by its file stem; `--contended`
//! names the passes taken while the machine was busy, which the tables leave out.

use crate::py::*;
use std::path::Path;

/// Class groups for the ranking (prefixes of the class names).
const GROUPS: &[(&str, &[&str])] = &[
    ("gate|up GEMM", &["gate|up GEMM"]),
    ("attention sliding (hd256)", &["attention sliding"]),
    ("attention global (hd512)", &["attention global"]),
    ("down GEMM", &["down GEMM"]),
    ("qkv GEMMs (sliding + global)", &["qkv GEMM"]),
    ("o GEMMs (sliding + global)", &["o GEMM"]),
    (
        "per-layer-input GEMMs (proj, ple_gate, ple_out)",
        &["PLE projection", "ple_gate", "ple_out"],
    ),
    ("gelu*up (mlp)", &["gelu*up"]),
    ("qk-norm + rope + kv store", &["qk-norm"]),
    ("rms_norm x48 + norm-residual x72", &["rms_norm", "norm+residual"]),
    ("per-layer-input scale+norm, gelu*input", &["PLE scale", "ple gelu"]),
    ("gather, final norm, pool, projection, l2", &["gather", "tail"]),
];

const OPTIONS: [&str; 7] = [
    "--gemm",
    "--memory",
    "--layout",
    "--ledger",
    "--alloc",
    "--ple",
    "--contended",
];

struct Args {
    out_dir: String,
    passes: Vec<String>,
    extras: Vec<(&'static str, String)>,
    contended: Vec<String>,
}

/// The argparse grammar of the script: positionals (the first is the output
/// directory), `--name value` options (an unambiguous prefix or `--name=value`
/// also works), and `--contended` taking every following word up to the next option.
fn parse_args(args: &[String]) -> R<Args> {
    let mut positionals = Vec::new();
    let mut extras: Vec<(&'static str, String)> = Vec::new();
    let mut contended = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        i += 1;
        if !arg.starts_with("--") || arg.len() == 2 {
            positionals.push(arg.clone());
            continue;
        }
        let (name, inline) = match arg.split_once('=') {
            Some((n, v)) => (n, Some(v.to_string())),
            None => (arg.as_str(), None),
        };
        let matches: Vec<&&str> = OPTIONS.iter().filter(|o| o.starts_with(name)).collect();
        let opt = match OPTIONS.iter().find(|o| **o == name) {
            Some(o) => *o,
            None if matches.len() == 1 => *matches[0],
            None => return Err(format!("unrecognized or ambiguous option {name}")),
        };
        if opt == "--contended" {
            if let Some(v) = inline {
                contended.push(v);
            }
            while i < args.len() && !args[i].starts_with("--") {
                contended.push(args[i].clone());
                i += 1;
            }
        } else {
            let value = match inline {
                Some(v) => v,
                None => {
                    let v = args
                        .get(i)
                        .filter(|v| !v.starts_with("--"))
                        .ok_or_else(|| format!("{opt} expects one argument"))?;
                    i += 1;
                    v.clone()
                }
            };
            extras.retain(|(k, _)| *k != opt);
            extras.push((opt, value));
        }
    }
    let mut it = positionals.into_iter();
    let out_dir = it
        .next()
        .ok_or("the following arguments are required: out_dir, passes")?;
    let passes: Vec<String> = it.collect();
    if passes.is_empty() {
        return Err("the following arguments are required: passes".to_string());
    }
    Ok(Args {
        out_dir,
        passes,
        extras,
        contended,
    })
}

fn group_of(cls: &str) -> R<&'static str> {
    GROUPS
        .iter()
        .find(|(_, keys)| keys.iter().any(|k| cls.starts_with(k)))
        .map(|(g, _)| *g)
        .ok_or_else(|| format!("KeyError: {cls:?}"))
}

/// `(int(a) * int(b), int(b))` of a workload named `AxB`.
fn dims(w: &str) -> R<(i64, i64)> {
    let mut parts = w.split('x');
    let mut next = || -> R<i64> {
        parts
            .next()
            .and_then(|p| p.trim().parse().ok())
            .ok_or_else(|| format!("ValueError: workload {w:?} is not AxB"))
    };
    let (a, b) = (next()?, next()?);
    Ok((a * b, b))
}

fn variants(ww: &Val) -> R<&[Val]> {
    ww.get("variants")?.arr()
}

fn kind(v: &Val) -> R<&str> {
    v.get("kind")?.str()
}

fn ms_min(v: &Val, field: &str) -> R<Num> {
    min(&v.get(field)?.nums()?)
}

/// The text after the first colon of a variant's name; an error when there is none.
fn class_of(v: &Val) -> R<&str> {
    v.get("name")?
        .str()?
        .split_once(':')
        .map(|(_, c)| c)
        .ok_or_else(|| "IndexError: variant name without a colon".to_string())
}

struct Summary {
    full: OMap<Num>,
    full_best: Num,
    encode_best: Num,
    only: OMap<OMap<Num>>,
    sum_only: Num,
}

pub fn run(argv: &[String]) -> R<()> {
    let args = parse_args(argv)?;
    let out_dir = Path::new(&args.out_dir);

    let mut passes: OMap<Val> = OMap::default();
    for p in &args.passes {
        passes.set(&stem(p), load_file(Path::new(p))?);
    }
    let contended = |name: &str| args.contended.iter().any(|c| c == name);

    // Minimum ms over the kept samples of one variant, per pass.
    let var_min = |w: &str, kind_want: &str, cls: Option<&str>| -> R<OMap<Num>> {
        let mut per_pass = OMap::default();
        for (pname, doc) in &passes.items {
            if contended(pname) {
                continue;
            }
            for ww in doc.get("workloads")?.arr()? {
                if ww.get("workload")?.str()? != w {
                    continue;
                }
                for v in variants(ww)? {
                    let wanted = kind(v)? == kind_want
                        && match cls {
                            None => true,
                            Some(c) => after_colon(v.get("name")?.str()?) == c,
                        };
                    if wanted {
                        per_pass.set(pname, ms_min(v, "ms_per_forward")?);
                    }
                }
            }
        }
        Ok(per_pass)
    };

    let mut names: Vec<String> = Vec::new();
    for doc in passes.values() {
        for ww in doc.get("workloads")?.arr()? {
            let n = ww.get("workload")?.str()?.to_string();
            if !names.contains(&n) {
                names.push(n);
            }
        }
    }
    let mut keyed = Vec::new();
    for n in names {
        keyed.push((dims(&n)?, n));
    }
    keyed.sort();
    let workloads: Vec<String> = keyed.into_iter().map(|(_, n)| n).collect();
    let first_doc = passes.values().next().ok_or("no passes")?;
    let mut classes: Vec<String> = Vec::new();
    for c in first_doc.get("classes")?.arr()? {
        classes.push(c.str()?.to_string());
    }

    let mut lines: Vec<String> = Vec::new();
    let mut summary: Vec<(String, Summary)> = Vec::new();
    for w in &workloads {
        let full = var_min(w, "full", None)?;
        let full_nums: Vec<Num> = full.values().copied().collect();
        let full_best = min(&full_nums)?;
        let mut enc = Vec::new();
        for (pn, d) in &passes.items {
            if contended(pn) {
                continue;
            }
            for ww in d.get("workloads")?.arr()? {
                if ww.get("workload")?.str()? == w {
                    enc.push(ms_min(ww, "encode_ms")?);
                }
            }
        }
        let mut only: OMap<OMap<Num>> = OMap::default();
        for c in &classes {
            only.set(c, var_min(w, "only", Some(c))?);
        }
        let mut bests = Vec::new();
        for v in only.values() {
            let nums: Vec<Num> = v.values().copied().collect();
            bests.push(min(&nums)?);
        }
        summary.push((
            w.clone(),
            Summary {
                full,
                full_best,
                encode_best: min(&enc)?,
                only,
                sum_only: sum(&bests)?,
            },
        ));
    }
    let sum_of = |w: &str| -> R<&Summary> {
        summary
            .iter()
            .find(|(n, _)| n == w)
            .map(|(_, s)| s)
            .ok_or_else(|| format!("KeyError: {w:?}"))
    };
    let only_min = |s: &Summary, c: &str| -> R<Num> {
        let v = s.only.get(c).ok_or_else(|| format!("KeyError: {c:?}"))?;
        let nums: Vec<Num> = v.values().copied().collect();
        min(&nums)
    };

    // 1. Whole forward, replica vs library, sum of stages.
    lines.push(
        "Whole forward: replica (async, forwards queued) against EmbedGemma2Model::encode, and the sum of its stages"
            .into(),
    );
    lines.push(format!(
        "  {:<9} {:>6} {:>11} {:>22} {:>12} {:>15} {:>12}",
        "workload", "rows", "replica ms", "per-pass min", "encode() ms", "sum(stages) ms", "sum/replica"
    ));
    for w in &workloads {
        let s = sum_of(w)?;
        let rows = dims(w)?.0;
        let pp = s.full.values().map(|v| fixed(v.f(), 1)).collect::<Vec<_>>().join(" ");
        lines.push(format!(
            "  {:<9} {:>6} {:>11} {:>22} {:>12} {:>15} {:>11}",
            w,
            rows,
            fixed(s.full_best.f(), 2),
            pp,
            fixed(s.encode_best.f(), 2),
            fixed(s.sum_only.f(), 2),
            percent(s.sum_only.div(s.full_best)?.f(), 1)
        ));
    }
    lines.push(String::new());

    // 2. Ranking by group.
    lines.push("Ranking by stage group: ms per forward (best over passes) and % of the replica's forward".into());
    let mut hdr = format!("  {:<50}", "group");
    for w in &workloads {
        hdr.push_str(&format!("{w:>17}"));
    }
    lines.push(hdr);
    // (group, per workload (ms, % of fwd))
    let mut rank_rows: Vec<(&str, Vec<(Num, Num)>)> = Vec::new();
    for (g, _) in GROUPS {
        let mut cells = Vec::new();
        for w in &workloads {
            let s = sum_of(w)?;
            let mut parts = Vec::new();
            for c in &classes {
                if group_of(c)? == *g {
                    parts.push(only_min(s, c)?);
                }
            }
            let ms = sum(&parts)?;
            cells.push((ms, F(100.0).mul(ms)?.div(s.full_best)?));
        }
        rank_rows.push((g, cells));
    }
    if workloads.len() < 2 {
        return Err("IndexError: the ranking needs at least two workloads".to_string());
    }
    let by_workload = |rows: &mut Vec<(&str, Vec<(Num, Num)>)>, at: usize| {
        rows.sort_by(|a, b| {
            a.1[at]
                .0
                .neg()
                .cmp(b.1[at].0.neg())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    };
    by_workload(&mut rank_rows, workloads.len() - 2);
    for (g, cells) in &rank_rows {
        let mut line = format!("  {g:<50}");
        for (ms, pct) in cells {
            line.push_str(&format!("{:>10} {:>5}%", fixed(ms.f(), 2), fixed(pct.f(), 1)));
        }
        lines.push(line);
    }
    lines.push(String::new());
    lines.push("Top 5 groups at each workload".into());
    for (wi, w) in workloads.iter().enumerate() {
        let mut ranked = rank_rows.clone();
        by_workload(&mut ranked, wi);
        let top: Vec<String> = ranked
            .iter()
            .take(5)
            .map(|(g, c)| format!("{g} {}%", fixed(c[wi].1.f(), 1)))
            .collect();
        lines.push(format!("  {:<8} {}", w, top.join("; ")));
    }
    lines.push(String::new());

    // 3. Per class tables with rates.
    for w in &workloads {
        let s = sum_of(w)?;
        let mut ww = None;
        'find: for d in passes.values() {
            for x in d.get("workloads")?.arr()? {
                if x.get("workload")?.str()? == w {
                    ww = Some(x);
                    break 'find;
                }
            }
        }
        let ww = ww.ok_or("StopIteration: no pass has this workload")?;
        let mut costs: OMap<&Val> = OMap::default();
        for c in ww.get("costs")?.arr()? {
            costs.set(c.get("class")?.str()?, c);
        }
        let mut disp: OMap<Num> = OMap::default();
        for v in variants(ww)? {
            if kind(v)? == "only" {
                disp.set(class_of(v)?, v.get("dispatches")?.num_val()?);
            }
        }
        lines.push(format!(
            "Per class, {w} ({} rows, {} dispatches per forward); only = that class alone, best over passes [lowest, highest per-pass minimum]",
            ww.get("rows")?.display(),
            ww.get("dispatches_full")?.display()
        ));
        lines.push(format!(
            "  {:<34} {:>4} {:>9} {:>8} {:>16} {:>8} {:>6} {:>8}",
            "class", "disp", "only ms", "% of fwd", "pass range", "us/disp", "GB/s", "TFLOP/s"
        ));
        let mut order: Vec<&String> = classes.iter().collect();
        let mut keyed = Vec::new();
        for c in order.drain(..) {
            keyed.push((only_min(s, c)?.neg(), c));
        }
        keyed.sort_by(|a, b| a.0.cmp(b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, c) in keyed {
            let vals: Vec<Num> = s
                .only
                .get(c)
                .ok_or_else(|| format!("KeyError: {c:?}"))?
                .values()
                .copied()
                .collect();
            let ms = min(&vals)?;
            let co = costs.get(c).ok_or_else(|| format!("KeyError: {c:?}"))?;
            let ms_s = F(ms.f() * 1e-3);
            let gbs = co.get("bytes_min")?.num_val()?.div(ms_s)?.div(F(1e9))?;
            let tf = co.get("flops")?.num_val()?.div(ms_s)?.div(F(1e12))?;
            let d = disp.get(c).copied().ok_or_else(|| format!("KeyError: {c:?}"))?;
            let d1 = if d.lt(I(1)) { I(1) } else { d };
            let tf_shown = if tf.gt(I(0)) { tf.f() } else { f64::NAN };
            lines.push(format!(
                "  {:<34} {:>4} {:>9} {:>7}% {:>8}-{:<7} {:>8} {:>6} {:>8}",
                c,
                str_num(d),
                fixed(ms.f(), 3),
                fixed(I(100).mul(ms)?.div(s.full_best)?.f(), 1),
                fixed(min(&vals)?.f(), 3),
                fixed(max(&vals)?.f(), 3),
                fixed(F(1e3).mul(ms)?.div(d1)?.f(), 1),
                fixed(gbs.f(), 0),
                fixed(tf_shown, 2)
            ));
        }
        lines.push(String::new());
    }

    // 4. Ablation cross-check where a pass has skip variants.
    for (pname, doc) in &passes.items {
        if contended(pname) {
            continue;
        }
        for ww in doc.get("workloads")?.arr()? {
            let vs = variants(ww)?;
            let mut has_skip = false;
            for v in vs {
                has_skip |= kind(v)? == "skip";
            }
            if !has_skip {
                continue;
            }
            let mut full_v = None;
            for v in vs {
                if kind(v)? == "full" {
                    full_v = Some(v);
                    break;
                }
            }
            let full = ms_min(full_v.ok_or("StopIteration: no full variant")?, "ms_per_forward")?;
            lines.push(format!(
                "Ablation cross-check, pass {pname}, {}: full {} ms; for the six largest stages, full - skip against the stage alone",
                ww.get("workload")?.str()?,
                fixed(full.f(), 1)
            ));
            let mut tops = Vec::new();
            for v in vs {
                if kind(v)? == "only" {
                    tops.push((ms_min(v, "ms_per_forward")?.neg(), v));
                }
            }
            tops.sort_by(|a, b| a.0.cmp(b.0).unwrap_or(std::cmp::Ordering::Equal));
            for (neg_ms, v) in tops.into_iter().take(6) {
                let c = class_of(v)?;
                let want = format!("skip:{c}");
                let mut sk = None;
                for x in vs {
                    if kind(x)? == "skip" && x.get("name")?.str()? == want {
                        sk = Some(x);
                        break;
                    }
                }
                let sk = sk.ok_or("StopIteration: no matching skip variant")?;
                lines.push(format!(
                    "    {:<34} alone {:>8}   full-skip {:>8}",
                    c,
                    fixed(neg_ms.neg().f(), 2),
                    fixed(full.sub(ms_min(sk, "ms_per_forward")?)?.f(), 2)
                ));
            }
            let mut diffs = Vec::new();
            for x in vs {
                if kind(x)? == "skip" {
                    diffs.push(full.sub(ms_min(x, "ms_per_forward")?)?);
                }
            }
            let tot = sum(&diffs)?;
            lines.push(format!(
                "    sum over all 21 stages of full-skip: {} ms = {} of full",
                fixed(tot.f(), 1),
                percent(tot.div(full)?.f(), 0)
            ));
        }
    }
    lines.push(String::new());

    let text = lines.join("\n");
    println!("{text}");
    write_text(
        &out_dir.join("embedgemma2_stage_profile_tables.txt"),
        &format!("{text}\n"),
    )?;

    let mut merged: Vec<(String, Val)> = vec![
        ("passes".to_string(), Val::Obj(passes.items.clone())),
        (
            "contended_passes".to_string(),
            Val::Arr(args.contended.iter().map(|c| Val::Str(c.clone())).collect()),
        ),
    ];
    for key in ["gemm", "memory", "layout", "ledger", "alloc", "ple"] {
        let flag = format!("--{key}");
        if let Some((_, path)) = args.extras.iter().find(|(k, _)| *k == flag) {
            merged.push((key.to_string(), load_file(Path::new(path))?));
        }
    }
    let json_path = out_dir.join("embedgemma2_stage_profile_m5pro.json");
    write_text(&json_path, &dumps(&Val::Obj(merged), None))?;
    eprintln!("wrote {}", json_path.display());
    Ok(())
}
