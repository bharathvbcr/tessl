//! Merge and tabulate the output of `profile_embedgemma2_vision`.
//!
//! ```text
//! results_report vision OUT_DIR LABEL=PASS.json [LABEL=PASS.json ...]
//! ```
//!
//! A LABEL names a build and operand mode (`before-f32`, `after-f32`,
//! `after-bf16`) and may repeat: every pass with that label is merged. Writes
//! `OUT_DIR/embedgemma2_vision_profile_m5pro.json` (every pass's raw document
//! under its label) and prints the ranking tables.
//!
//! A stage's time is the minimum over every kept sample of every pass with the
//! label (the harness keeps the samples taken while no other tessl GPU process
//! ran, or all of them when none was clean); the table also shows the range of
//! the per-pass minima, the honest error bar on a GPU other sessions share, and
//! how many samples the harness flagged as taken beside another process. `only`
//! is a stage run alone with the real shapes and weights of all 16 layers; the
//! share column divides by the sum of the stages' `only` times (what the
//! forward costs if its stages add up) and `full` is the whole forward's own
//! minimum, so the gap between them is the cost the stages do not show alone
//! (overlap, caches). `full - skip` is reported by the harness and kept in the
//! JSON; it is not printed, because on a shared GPU its difference of two noisy
//! minima is not trustworthy (the raw samples are there to check).

use crate::py::*;
use std::path::Path;

/// `variant(doc_wl, kind, cls)`: the first variant of that kind (and class index).
fn variant<'a>(wl: &'a Val, kind: &str, cls: Option<usize>) -> R<&'a Val> {
    for v in wl.get("variants")?.arr()? {
        if v.get("kind")?.str()? == kind {
            let ok = match cls {
                None => true,
                Some(ci) => v.get("class")?.num_val()?.eq(I(ci as i128)),
            };
            if ok {
                return Ok(v);
            }
        }
    }
    Err(format!("KeyError: ({kind:?}, {cls:?})"))
}

/// `label, _, path = arg.partition("=")`.
pub fn partition(arg: &str) -> (&str, &str) {
    arg.split_once('=').unwrap_or((arg, ""))
}

pub fn run(args: &[String]) -> R<()> {
    let Some((out_dir, rest)) = args.split_first() else {
        return Err("usage: results_report vision OUT_DIR LABEL=PASS.json [LABEL=PASS.json ...]".to_string());
    };
    let out_dir = Path::new(out_dir);
    let mut passes: OMap<Vec<Val>> = OMap::default();
    for arg in rest {
        let (label, path) = partition(arg);
        let doc = load_file(Path::new(path))?;
        passes.entry(label, Vec::new).push(doc);
    }
    let runs = Val::Obj(
        passes
            .items
            .iter()
            .map(|(k, v)| (k.clone(), Val::Arr(v.clone())))
            .collect(),
    );
    write_text(
        &out_dir.join("embedgemma2_vision_profile_m5pro.json"),
        &format!("{}\n", dumps(&Val::obj([("runs", runs)]), None)),
    )?;

    for (label, docs) in &passes.items {
        let mut names: Vec<&str> = Vec::new();
        for d in docs {
            for w in d.get("workloads")?.arr()? {
                let n = w.get("workload")?.str()?;
                if !names.contains(&n) {
                    names.push(n);
                }
            }
        }
        for name in names {
            let mut wls: Vec<&Val> = Vec::new();
            for d in docs {
                for w in d.get("workloads")?.arr()? {
                    if w.get("workload")?.str()? == name {
                        wls.push(w);
                    }
                }
            }
            let first = *wls.first().ok_or("IndexError: no workload")?;
            let classes = docs
                .first()
                .ok_or("StopIteration: no documents")?
                .get("classes")?
                .arr()?;
            let mut full_min = Vec::new();
            let mut enc_min = Vec::new();
            for w in &wls {
                full_min.push(min(&variant(w, "full", None)?.get("ms_per_call")?.nums()?)?);
                enc_min.push(min(&w.get("encode_images_ms")?.nums()?)?);
            }
            let mut flagged = Vec::new();
            let mut total = Vec::new();
            for w in &wls {
                flagged.push(w.get("samples_contended")?.num_val()?);
                total.push(w.get("samples_total")?.num_val()?);
            }
            let (flagged, total) = (sum(&flagged)?, sum(&total)?);
            let operands = if first.has("operands") {
                first.get("operands")?.display()
            } else {
                "f32 (the build before the bf16 mode)".to_string()
            };
            println!(
                "== {label}, workload {name}: {} forward(s), {} padded rows, {} pass(es), operands {operands}",
                first.get("forwards")?.display(),
                first.get("padded_rows")?.display(),
                wls.len()
            );
            let (fmin, fmax) = (min(&full_min)?, max(&full_min)?);
            let (emin, emax) = (min(&enc_min)?, max(&enc_min)?);
            let mut head = format!(
                "   full replica {} ms (min; per-pass minima {}..{}); encode_images {} ms (per-pass {}..{})",
                fixed(fmin.f(), 2),
                fixed(fmin.f(), 1),
                fixed(fmax.f(), 1),
                fixed(emin.f(), 2),
                fixed(emin.f(), 1),
                fixed(emax.f(), 1)
            );
            if first.has("host_io_ms") {
                let mut io = Vec::new();
                for w in &wls {
                    io.push(w.get("host_io_ms")?.num_val()?);
                }
                head.push_str(&format!("; host io build {} ms", fixed(min(&io)?.f(), 2)));
            }
            println!("{head}");
            println!(
                "   samples beside another tessl GPU process: {} of {}; gpu util at start/end {}/{} %",
                str_num(flagged),
                str_num(total),
                first.get("gpu_util_start_pct")?.display(),
                first.get("gpu_util_end_pct")?.display()
            );
            // (class name, dispatches, only, lo, hi, cost)
            let mut rows: Vec<(String, Num, Num, Num, Num, &Val)> = Vec::new();
            for (ci, cname) in classes.iter().enumerate() {
                let mut vs = Vec::new();
                for w in &wls {
                    vs.push(variant(w, "only", Some(ci))?);
                }
                let disp = vs[0].get("dispatches")?.num_val()?;
                if disp.eq(I(0)) {
                    continue;
                }
                let mut per_pass = Vec::new();
                for v in &vs {
                    per_pass.push(min(&v.get("ms_per_call")?.nums()?)?);
                }
                let cost = first
                    .get("costs")?
                    .arr()?
                    .get(ci)
                    .ok_or_else(|| format!("IndexError: costs[{ci}]"))?;
                let only = min(&per_pass)?;
                rows.push((
                    cname.str()?.to_string(),
                    disp,
                    only,
                    min(&per_pass)?,
                    max(&per_pass)?,
                    cost,
                ));
            }
            let only_all: Vec<Num> = rows.iter().map(|r| r.2).collect();
            let total_only = sum(&only_all)?;
            println!(
                "   {:<38} {:>5} {:>9} {:>6} {:>15} {:>6} {:>8}",
                "stage", "disp", "only ms", "share", "pass range", "GB/s", "TFLOP/s"
            );
            let mut order: Vec<usize> = (0..rows.len()).collect();
            order.sort_by(|a, b| {
                rows[*a]
                    .2
                    .neg()
                    .cmp(rows[*b].2.neg())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for at in order {
                let (cname, disp, only, lo, hi, cost) = &rows[at];
                let only_s = F(only.f() * 1e-3);
                let gbs = cost.get("bytes_min")?.num_val()?.div(only_s)?.div(F(1e9))?;
                let flops = cost.get("flops")?.num_val()?;
                let tf = flops.div(only_s)?.div(F(1e12))?;
                let tf_text = if flops.truthy() {
                    fixed(tf.f(), 2)
                } else {
                    "-".to_string()
                };
                println!(
                    "   {:<38} {:>5} {:>9} {:>5}% {:>15} {:>6} {:>8}",
                    cname,
                    str_num(*disp),
                    fixed(only.f(), 3),
                    fixed(I(100).mul(*only)?.div(total_only)?.f(), 1),
                    format!("{}..{}", fixed(lo.f(), 2), fixed(hi.f(), 2)),
                    fixed(gbs.f(), 0),
                    tf_text
                );
            }
            println!(
                "   sum(only) {} ms = {}% of the full replica's minimum",
                fixed(total_only.f(), 2),
                fixed(I(100).mul(total_only)?.div(fmin)?.f(), 1)
            );
            println!();
        }
    }
    Ok(())
}
