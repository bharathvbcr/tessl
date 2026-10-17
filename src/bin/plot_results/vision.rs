//! `embedgemma2_vision_speed_m5pro.json` (bench_paired's report for the vision
//! tower) with `embedgemma2_vision_profile_m5pro.json` beside it (written by
//! `results_report vision`): three panels.
//!
//! 1. `encode_images` time per lane at 1 image and at 8 images: the bar is the
//!    median over rounds of each round's min-of-N, the dots are the rounds;
//! 2. the same as ratios: each lane's time over the first lane's, per round
//!    (median, and the lowest and highest round as the whisker), where a
//!    whisker that crosses 1.0 is not a result;
//! 3. where the 1-image forward's time goes, per stage, for the profile's
//!    labels (each stage alone on the real shapes and weights, ms).

use crate::chart::{
    nice_floor, spread, Axes, AxisCfg, Figure, LegendItem, Paint, Stroke, Ts, Va, M_BLUE, M_GREY, M_ORANGE,
};
use crate::data::{max_of, min_of, N, R};
use crate::paired::{lane_names, load, log_span};
use crate::registry::Output;
use std::path::Path;

const LANE_COLORS: [&str; 3] = [M_GREY, M_BLUE, M_ORANGE];

fn lane_color(i: usize) -> &'static str {
    LANE_COLORS[i % LANE_COLORS.len()]
}

/// `1img` as "1 image", `8img` as "8 images".
fn workload_label(w: &str) -> String {
    w.replace("img", if w == "1img" { " image" } else { " images" })
}

/// Stage name to the minimum over passes of its `only` ms per call, for one
/// profile label and workload, in the order the stages first appear.
fn stage_ms(label: &N, workload: &str) -> R<Vec<(String, f64)>> {
    let mut out: Vec<(String, f64)> = Vec::new();
    for doc in label.nonempty()? {
        let wl = doc.get("workloads")?.find("workload", workload)?;
        let variants = wl.get("variants")?.items()?;
        for (ci, name) in doc.get("classes")?.nonempty()?.iter().enumerate() {
            let name = name.s()?.to_string();
            for v in &variants {
                if v.get("kind")?.s()? == "only" && v.get("class")?.f()? == ci as f64 && v.get("dispatches")?.f()? > 0.0
                {
                    let best = min_of(&v.get("ms_per_call")?.floats()?)?;
                    match out.iter_mut().find(|(n, _)| *n == name) {
                        Some((_, m)) => *m = m.min(best),
                        None => out.push((name.clone(), best)),
                    }
                }
            }
        }
    }
    Ok(out)
}

pub fn vision_speed(json: &Path) -> R<Output> {
    let speed_doc = load(json)?;
    let speed = N::root(&speed_doc, "vision_speed");
    let profile_path = json.with_file_name("embedgemma2_vision_profile_m5pro.json");
    let profile_doc = load(&profile_path)?;
    let profile = N::root(&profile_doc, "vision_profile").get("runs")?;

    let lanes = lane_names(&speed)?;
    let rows = speed.get("summary")?.nonempty()?;
    let workloads: Vec<String> = rows
        .iter()
        .map(|r| r.get("workload")?.s().map(str::to_string))
        .collect::<R<_>>()?;
    let labels: Vec<String> = workloads.iter().map(|w| workload_label(w)).collect();
    let base = lanes[0].clone();

    let legend: Vec<LegendItem> = lanes
        .iter()
        .enumerate()
        .map(|(i, l)| LegendItem::boxed(l, lane_color(i)))
        .collect();
    let mut fig = Figure::new(
        1600.0,
        640.0,
        "EmbeddingGemma 2 vision tower on an Apple M5 Pro, real checkpoint",
        &[],
        &legend,
    );
    let cells = fig.cells(&[1.1, 1.1, 1.6], &[1.0]);

    // 1. Wall time, log.
    let mut all = Vec::new();
    for r in &rows {
        for l in &lanes {
            all.extend(r.get(l)?.get("ms_min_all")?.positives()?);
            all.push(r.get(l)?.get("ms_min_median")?.pos()?);
        }
    }
    let (lo, hi) = log_span(min_of(&all)?, max_of(&all)?)?;
    let mut ax = Axes::new(
        cells[0],
        "Wall time of one encode_images call",
        &[],
        AxisCfg::categories(&labels, false)?,
        AxisCfg::log(lo, hi)?.labelled("encode_images, ms (median over rounds of min-of-N; dots = rounds)"),
    )?;
    let width = 0.8 / lanes.len() as f64;
    for (li, lane) in lanes.iter().enumerate() {
        for (wi, r) in rows.iter().enumerate() {
            let d = r.get(lane)?;
            let x = wi as f64 + spread(li, lanes.len(), width);
            let med = d.get("ms_min_median")?.pos()?;
            ax.bar(x, width * 0.92, (0.0, med), &Paint::new(lane_color(li)));
            let rounds = d.get("ms_min_all")?.positives()?;
            for v in &rounds {
                ax.dot(x, *v, 2.2, &Paint::ink());
            }
            ax.text(
                x,
                max_of(&rounds)?.max(med),
                (0.0, -6.0),
                &format!("{med:.0}"),
                &Ts::new(10.0).mid().ink().va(Va::Bottom),
            );
        }
    }
    fig.add(ax);

    // 2. Ratios against the first lane.
    let others: Vec<(usize, &String)> = lanes.iter().enumerate().skip(1).collect();
    let mut tops = Vec::new();
    for r in &rows {
        for (_, l) in &others {
            tops.push(r.get(l)?.get("vs_base")?.get("hi")?.f()?);
        }
    }
    let top = max_of(&tops)?.max(1.0);
    let legend2: Vec<LegendItem> = others
        .iter()
        .map(|(i, l)| LegendItem::boxed(&format!("{l} / {base}"), lane_color(*i)))
        .collect();
    let mut ax = Axes::new(
        cells[1],
        "Paired, interleaved rounds (order reversed on odd rounds)",
        &legend2,
        AxisCfg::categories(&labels, false)?,
        AxisCfg::linear(0.0, top * 1.14)?.labelled(&format!(
            "time relative to {base} (lower is faster); whiskers = lowest/highest round"
        )),
    )?;
    ax.axhline(1.0, &Stroke::ink(1.0).dashed());
    let w2 = 0.34;
    for (k, (li, lane)) in others.iter().enumerate() {
        for (wi, r) in rows.iter().enumerate() {
            let v = r.get(lane)?.get("vs_base")?;
            let x = wi as f64 + spread(k, others.len(), 0.38);
            let med = v.get("median")?.f()?;
            ax.bar(x, w2, (0.0, med), &Paint::new(lane_color(*li)));
            ax.whisker(x, (v.get("lo")?.f()?, v.get("hi")?.f()?), 4.0, &Stroke::ink(1.0));
            ax.text(
                x,
                v.get("hi")?.f()?,
                (0.0, -8.0),
                &format!("{med:.2}x"),
                &Ts::new(10.0).mid().ink().va(Va::Bottom),
            );
        }
    }
    fig.add(ax);

    // 3. Where the 1-image forward's time goes, per stage.
    let mut per_label = Vec::new();
    for (k, v) in profile.entries()? {
        per_label.push((k, stage_ms(&v, "1img")?));
    }
    if per_label.is_empty() {
        return Err(format!("{}: no profile labels", profile.path()));
    }
    let mut stages: Vec<(String, f64)> = Vec::new();
    for (_, st) in &per_label {
        for (name, ms) in st {
            match stages.iter_mut().find(|(n, _)| n == name) {
                Some((_, m)) => *m = m.max(*ms),
                None => stages.push((name.clone(), *ms)),
            }
        }
    }
    stages.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if stages.is_empty() {
        return Err(format!("{}: no stage has a time above zero", profile.path()));
    }
    let names: Vec<String> = stages.iter().map(|(n, _)| n.clone()).collect();
    let vals: Vec<f64> = per_label
        .iter()
        .flat_map(|(_, st)| st.iter().map(|(_, ms)| *ms))
        .filter(|v| *v > 0.0)
        .collect();
    let (lo, hi) = (nice_floor(min_of(&vals)? * 0.6)?, max_of(&vals)? * 1.5);
    let rows_padded = padded_rows(&profile)?;
    let legend3: Vec<LegendItem> = per_label
        .iter()
        .enumerate()
        .map(|(i, (l, _))| LegendItem::boxed(l, lane_color(i)))
        .collect();
    let mut ax = Axes::new(
        cells[2],
        "Where the time goes (profile_embedgemma2_vision)",
        &legend3,
        AxisCfg::log(lo, hi)?.labelled(&format!(
            "ms for all 16 layers of one stage, run alone (1 image, {rows_padded} padded rows)"
        )),
        AxisCfg::categories(&names, true)?,
    )?;
    let bar = 0.8 / per_label.len() as f64;
    for (li, (_, st)) in per_label.iter().enumerate() {
        for (si, name) in names.iter().enumerate() {
            if let Some((_, v)) = st.iter().find(|(n, _)| n == name) {
                if *v > 0.0 {
                    ax.hbar(
                        si as f64 + spread(li, per_label.len(), bar),
                        bar * 0.92,
                        (0.0, *v),
                        &Paint::new(lane_color(li)),
                    );
                }
            }
        }
    }
    fig.add(ax);
    Ok(Output::svg(fig.finish()))
}

/// `padded_rows` of the 1-image workload of the first profile document.
fn padded_rows(profile: &N) -> R<f64> {
    let first = profile
        .entries()?
        .into_iter()
        .next()
        .ok_or_else(|| format!("{}: no profile labels", profile.path()))?
        .1;
    first
        .nonempty()?
        .first()
        .ok_or_else(|| format!("{}: empty", first.path()))?
        .get("workloads")?
        .find("workload", "1img")?
        .get("padded_rows")?
        .f()
}
