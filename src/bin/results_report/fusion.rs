//! Tabulate `profile_embedgemma2` stage profiles of the text encoder, before
//! and after the row-pass fusion (G7).
//!
//! ```text
//! results_report text-fusion OUT_DIR LABEL=FILE.json [LABEL=FILE.json ...]
//! ```
//!
//! Each FILE is one `PROFILE_MODE=stages` run (`PROFILE_JSON`); a LABEL may be
//! given for several files (passes), and its numbers are then the minimum over
//! all of them (the house method: min-of-N over every kept sample of every
//! pass), with the range of the per-pass minima next to it. A class's time is
//! the minimum over a pass's samples of its `only:` variant (the stage alone,
//! every other dispatch skipped); a category sums its classes; the share is of
//! the sum of all classes (the stages add up to the whole forward to within the
//! run-to-run noise, as `embedgemma2_stage_profile_m5pro.txt` showed). The
//! classes are grouped as:
//!
//! ```text
//! GEMM              every exact-f32 or bf16 GEMM (qkv, o, gate|up, down, the three
//!                   per-layer-input ones; the tail's projection is in `tail`)
//! attention         sliding and global
//! row passes        every norm and residual add, the per-layer inputs' scale+norm
//!                   and (bf16) the casts: the classes the fusion changed
//! gelu*up           the two gated activations
//! other elementwise gather+copy and qk-norm+rope+kv store
//! tail              final norm (unfused), pooling, projection, l2 normalize
//! ```
//!
//! Prints the tables; writes `OUT_DIR/embedgemma2_text_fusion_profile_m5pro.json`
//! (every input file merged under its label) and the tables to
//! `OUT_DIR/embedgemma2_text_fusion_profile_tables.txt`.

use crate::py::*;
use std::path::Path;

const CATEGORIES: &[(&str, &[&str])] = &[
    (
        "GEMM",
        &[
            "qkv GEMM",
            "o GEMM",
            "gate|up GEMM",
            "down GEMM",
            "PLE projection",
            "ple_gate",
            "ple_out",
        ],
    ),
    ("attention", &["attention"]),
    // `rms_norm (input, pre-ffn)` / `norm+residual` are the names before the
    // fusion lane renamed the classes; both spellings are the same stages.
    (
        "row passes",
        &["rms_norm", "residual add", "norm+residual", "PLE scale", "bf16 casts"],
    ),
    ("gelu*up", &["gelu*up", "ple gelu"]),
    ("other elementwise", &["gather", "qk-norm"]),
    ("tail", &["tail"]),
];

fn category(cls: &str) -> R<&'static str> {
    CATEGORIES
        .iter()
        .find(|(_, prefixes)| prefixes.iter().any(|p| cls.starts_with(p)))
        .map(|(name, _)| *name)
        .ok_or_else(|| format!("KeyError: {cls:?}"))
}

struct Stages<'a> {
    /// `{class: [per-pass min ms]}` in first-seen order.
    per: OMap<Vec<Num>>,
    disp: OMap<Num>,
    full: Vec<Num>,
    first: Option<&'a Val>,
}

fn stages<'a>(docs: &'a [Val], wl: &str) -> R<Stages<'a>> {
    // For one label and workload: every pass's class minima (in first-seen
    // order), dispatch counts, `full` minima, and the first pass's record.
    let mut s = Stages {
        per: OMap::default(),
        disp: OMap::default(),
        full: Vec::new(),
        first: None,
    };
    for doc in docs {
        let mut found = None;
        for x in doc.get("workloads")?.arr()? {
            if x.get("workload")?.str()? == wl {
                found = Some(x);
                break;
            }
        }
        let Some(w) = found else { continue };
        s.first = s.first.or(Some(w));
        for v in w.get("variants")?.arr()? {
            let kind = v.get("kind")?.str()?;
            if kind == "only" {
                let name = v.get("name")?.str()?;
                let cls = name
                    .split_once(':')
                    .map(|(_, c)| c)
                    .ok_or("IndexError: variant name without a colon")?;
                s.per.entry(cls, Vec::new).push(min(&v.get("ms_per_forward")?.nums()?)?);
                s.disp.set(cls, v.get("dispatches")?.num_val()?);
            } else if kind == "full" {
                s.full.push(min(&v.get("ms_per_forward")?.nums()?)?);
            }
        }
    }
    Ok(s)
}

struct Summary {
    best: OMap<Num>,
    total: Num,
    cat_ms: Vec<(&'static str, Num)>,
    cat_disp: Vec<(&'static str, Num)>,
    full: Vec<Num>,
    dispatches: String,
    n_pass: usize,
}

fn cat_of(v: &[(&'static str, Num)], c: &str) -> Num {
    v.iter().find(|(n, _)| *n == c).map_or(I(0), |(_, x)| *x)
}

pub fn run(args: &[String]) -> R<()> {
    let Some((out_dir, rest)) = args.split_first() else {
        return Err("usage: results_report text-fusion OUT_DIR LABEL=FILE.json [LABEL=FILE.json ...]".to_string());
    };
    let out_dir = Path::new(out_dir);
    let mut passes: OMap<Vec<Val>> = OMap::default();
    for arg in rest {
        let (label, path) = arg
            .split_once('=')
            .ok_or_else(|| format!("ValueError: {arg:?} is not LABEL=FILE"))?;
        passes.entry(label, Vec::new).push(load_file(Path::new(path))?);
    }
    let labels: Vec<String> = passes.keys().cloned().collect();

    let mut names: Vec<String> = Vec::new();
    for docs in passes.values() {
        for d in docs {
            for w in d.get("workloads")?.arr()? {
                let n = w.get("workload")?.str()?.to_string();
                if !names.contains(&n) {
                    names.push(n);
                }
            }
        }
    }
    let mut keyed = Vec::new();
    for n in names {
        let mut parts = n.split('x');
        let mut next = || -> R<i64> {
            parts
                .next()
                .and_then(|p| p.trim().parse().ok())
                .ok_or_else(|| format!("ValueError: workload {n:?} is not AxB"))
        };
        let (a, b) = (next()?, next()?);
        keyed.push(((a * b, n.clone()), n));
    }
    keyed.sort();
    let workloads: Vec<String> = keyed.into_iter().map(|(_, n)| n).collect();

    let mut summary: Vec<((String, String), Summary)> = Vec::new();
    for wl in &workloads {
        for label in &labels {
            let docs = passes.get(label).ok_or("KeyError")?;
            let st = stages(docs, wl)?;
            if st.per.is_empty() {
                continue;
            }
            let mut best: OMap<Num> = OMap::default();
            for (c, v) in &st.per.items {
                best.set(c, min(v)?);
            }
            let best_nums: Vec<Num> = best.values().copied().collect();
            let total = sum(&best_nums)?;
            let mut cat_ms: Vec<(&'static str, Num)> = CATEGORIES.iter().map(|(c, _)| (*c, F(0.0))).collect();
            let mut cat_disp: Vec<(&'static str, Num)> = CATEGORIES.iter().map(|(c, _)| (*c, I(0))).collect();
            for (cls, ms) in &best.items {
                let cat = category(cls)?;
                if let Some(slot) = cat_ms.iter_mut().find(|(n, _)| *n == cat) {
                    slot.1 = slot.1.add(*ms)?;
                }
                let d = *st.disp.get(cls).ok_or_else(|| format!("KeyError: {cls:?}"))?;
                if let Some(slot) = cat_disp.iter_mut().find(|(n, _)| *n == cat) {
                    slot.1 = slot.1.add(d)?;
                }
            }
            let first = st.first.ok_or("TypeError: no workload record")?;
            summary.push((
                (label.clone(), wl.clone()),
                Summary {
                    best,
                    total,
                    cat_ms,
                    cat_disp,
                    full: st.full,
                    dispatches: first.get("dispatches_full")?.display(),
                    n_pass: docs.len(),
                },
            ));
        }
    }
    let get = |label: &str, wl: &str| summary.iter().find(|((l, w), _)| l == label && w == wl).map(|(_, s)| s);

    let mut lines: Vec<String> = Vec::new();
    for wl in &workloads {
        let n_pass = summary
            .iter()
            .filter(|((_, w), _)| w == wl)
            .map(|(_, s)| s.n_pass)
            .max()
            .ok_or("ValueError: max() of an empty sequence")?;
        lines.push(format!(
            "== workload {wl}   (ms per forward; stage alone, min over {n_pass} pass(es))"
        ));
        let mut header = format!("  {:<30} {:>5} {:>9} {:>10} ", "run", "disp", "full ms", "sum(only)");
        for (c, _) in CATEGORIES {
            header.push_str(&format!("{c:>19}"));
        }
        lines.push(header);
        for label in &labels {
            let Some(s) = get(label, wl) else { continue };
            let mut cells = String::new();
            for (c, _) in CATEGORIES {
                let ms = cat_of(&s.cat_ms, c);
                cells.push_str(&format!(
                    "{:>10} {:>5}% ",
                    fixed(ms.f(), 3),
                    fixed(I(100).mul(ms)?.div(s.total)?.f(), 1)
                ));
            }
            lines.push(format!(
                "  {:<30} {:>5} {:>9} {:>10} {}",
                label,
                s.dispatches,
                fixed(min(&s.full)?.f(), 3),
                fixed(s.total.f(), 3),
                cells
            ));
        }
        lines.push(String::new());
    }

    lines.push("Row passes (the classes the fusion changed): dispatches, ms (min over passes) and [range of the per-pass sums], share of the sum of stages".into());
    lines.push(format!(
        "  {:<9} {:<30} {:>5} {:>9} {:>16} {:>7}   classes (ms)",
        "workload", "run", "disp", "ms", "passes", "share"
    ));
    for wl in &workloads {
        for label in &labels {
            let Some(s) = get(label, wl) else { continue };
            let mut rp = Vec::new();
            for c in s.best.keys() {
                if category(c)? == "row passes" {
                    rp.push(c.clone());
                }
            }
            let st = stages(passes.get(label).ok_or("KeyError")?, wl)?;
            let mut passes_rp = Vec::new();
            for i in 0..s.n_pass {
                let mut parts = Vec::new();
                for c in &rp {
                    let v = st.per.get(c).ok_or_else(|| format!("KeyError: {c:?}"))?;
                    if v.len() > i {
                        parts.push(v[i]);
                    }
                }
                passes_rp.push(sum(&parts)?);
            }
            let row_ms = cat_of(&s.cat_ms, "row passes");
            let classes: Vec<String> = rp
                .iter()
                .map(|c| {
                    let short = c.split(" (").next().unwrap_or(c);
                    let ms = s.best.get(c).copied().unwrap_or(I(0));
                    format!("{short} {}", fixed(ms.f(), 3))
                })
                .collect();
            lines.push(format!(
                "  {:<9} {:<30} {:>5} {:>9} {:>7}-{:<7} {:>6}%   {}",
                wl,
                label,
                str_num(cat_of(&s.cat_disp, "row passes")),
                fixed(row_ms.f(), 3),
                fixed(min(&passes_rp)?.f(), 2),
                fixed(max(&passes_rp)?.f(), 2),
                fixed(I(100).mul(row_ms)?.div(s.total)?.f(), 1),
                classes.join("; ")
            ));
        }
    }
    lines.push(String::new());
    lines.push("Machine: GPU utilization at the start and end of each workload (ioreg, %, first pass), and samples taken with another tessl GPU process seen".into());
    for wl in &workloads {
        for label in &labels {
            if get(label, wl).is_none() {
                continue;
            }
            let first = stages(passes.get(label).ok_or("KeyError")?, wl)?
                .first
                .ok_or("TypeError: no workload record")?;
            lines.push(format!(
                "  {:<9} {:<30} util {}% -> {}%   contended samples {} of {}",
                wl,
                label,
                first.get("gpu_util_start_pct")?.display(),
                first.get("gpu_util_end_pct")?.display(),
                first.get("samples_contended")?.display(),
                first.get("samples_total")?.display()
            ));
        }
    }

    let text = lines.join("\n");
    println!("{text}");
    write_text(
        &out_dir.join("embedgemma2_text_fusion_profile_tables.txt"),
        &format!("{text}\n"),
    )?;
    let merged = Val::obj([
        (
            "runs",
            Val::Obj(
                labels
                    .iter()
                    .map(|l| {
                        let docs = passes.get(l).ok_or("KeyError")?;
                        Ok((l.clone(), docs.last().ok_or("IndexError")?.clone()))
                    })
                    .collect::<R<Vec<_>>>()?,
            ),
        ),
        (
            "passes",
            Val::Obj(
                labels
                    .iter()
                    .map(|l| {
                        let docs = passes.get(l).ok_or("KeyError")?;
                        Ok((l.clone(), Val::Arr(docs.clone())))
                    })
                    .collect::<R<Vec<_>>>()?,
            ),
        ),
        (
            "categories",
            Val::Arr(CATEGORIES.iter().map(|(c, _)| Val::Str((*c).to_string())).collect()),
        ),
    ]);
    let json_path = out_dir.join("embedgemma2_text_fusion_profile_m5pro.json");
    write_text(&json_path, &dumps(&merged, None))?;
    eprintln!("wrote {}", json_path.display());
    Ok(())
}
