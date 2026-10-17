//! Figures drawn from `bench_paired --format workloads` output: one summary row
//! per workload, one entry per lane with every round's min-of-N time
//! (`ms_min_all`), the median over rounds (`ms_min_median`) and, for the lanes
//! after the first, the median per-round ratio to the first lane with its
//! lowest and highest round (`vs_base`).
//!
//! * [`final_`]: four lanes of `bench_embedgemma2` over the text workloads.
//! * [`audio_speed`]: before / after / control for the audio tower.
//! * [`mm_onewait`]: before / after / control for the multimodal facade.
//! * [`pre`]: tessl's preprocessors against the Hugging Face processors.

use crate::chart::{
    nice_ceil, nice_floor, Axes, AxisCfg, Figure, Kind, LegendItem, Paint, Scale, Stroke, Ts, Va, BLUE, GREEN, GREY,
    ORANGE,
};
use crate::data::{max_of, min_of, N, R};
use crate::json::Json;
use crate::registry::Output;
use std::path::Path;

/// Pixels a panel loses to its title, legend and tick labels: what the plot
/// area is short of its cell, for sizing a range before the axes exist.
pub const CHROME_PX: f64 = 150.0;

/// A series colour by lane name; a lane the figure has no colour for is grey.
pub fn color_of(table: &[(&str, &'static str)], lane: &str) -> &'static str {
    table.iter().find(|(n, _)| *n == lane).map(|(_, c)| *c).unwrap_or(GREY)
}

pub fn lane_names(doc: &N) -> R<Vec<String>> {
    doc.get("lanes")?
        .nonempty()?
        .iter()
        .map(|l| l.get("name")?.s().map(str::to_string))
        .collect()
}

/// A log range with round ends around `[min, max]`.
pub fn log_span(min: f64, max: f64) -> R<(f64, f64)> {
    let lo = nice_floor(min * 0.8)?;
    let hi = nice_ceil(max * 1.2)?;
    if lo == hi {
        return Ok((lo / 2.0, hi * 2.0));
    }
    Ok((lo, hi))
}

pub fn load(path: &Path) -> R<Json> {
    crate::data::parse_file(path)
}

/// `"{:.2}x\n[{:.2}, {:.2}]"`: a ratio with the lowest and highest round.
fn ratio_label(r: &N) -> R<String> {
    Ok(format!(
        "{:.2}x\n[{:.2}, {:.2}]",
        r.get("median")?.f()?,
        r.get("lo")?.f()?,
        r.get("hi")?.f()?
    ))
}

/// One lane of a strip panel: the median as a thick bar, every round as a dot,
/// and above the dots the ratio to the base lane (none for the base itself).
fn strip_lane(ax: &mut Axes, x: f64, d: &N, color: &str, label: Option<String>) -> R<()> {
    let med = d.get("ms_min_median")?.f()?;
    let all = d.get("ms_min_all")?.floats()?;
    ax.line("median", (x - 0.3, med), (x + 0.3, med), &Stroke::color(color, 3.0));
    let n = all.len() as f64;
    for (i, v) in all.iter().enumerate() {
        ax.dot(
            x + (i as f64 - (n - 1.0) / 2.0) * 0.045,
            *v,
            4.0,
            &Paint::new(color).bg_edge(1.2),
        );
    }
    if let Some(label) = label {
        let top = max_of(&all)?;
        ax.text(x, top, (0.0, -7.0), &label, &Ts::new(10.5).mid().ink().va(Va::Bottom));
    }
    Ok(())
}

fn blank(s: &str) -> String {
    s.replace('_', " ")
}

fn rounds_of(doc: &N) -> String {
    match doc.opt("rounds").map(|r| r.f()) {
        Some(Ok(v)) => format!("{v}"),
        _ => "?".to_string(),
    }
}

// ---------------------------------------------------------------------------
// final
// ---------------------------------------------------------------------------

const FINAL_COLORS: &[(&str, &str)] = &[
    ("before_async", BLUE),
    ("control", GREY),
    ("after_f32", ORANGE),
    ("after_bf16", GREEN),
];

/// `embedgemma2_final_m5pro.json`: text encode time per workload, speed-up
/// against the first lane, and device memory.
pub fn final_(json: &Path) -> R<Output> {
    let doc = load(json)?;
    let root = N::root(&doc, "final");
    let rows = root.get("summary")?.nonempty()?;
    let lanes = lane_names(&root)?;
    if lanes.len() < 2 {
        return Err(format!(
            "{}: needs a base lane and at least one more",
            root.get("lanes")?.path()
        ));
    }
    let base = lanes[0].clone();
    let works: Vec<String> = rows
        .iter()
        .map(|r| r.get("workload")?.s().map(str::to_string))
        .collect::<R<_>>()?;

    let legend: Vec<LegendItem> = lanes
        .iter()
        .map(|l| LegendItem::boxed(l, color_of(FINAL_COLORS, l)))
        .collect();
    let mut fig = Figure::new(
        1500.0,
        560.0,
        &format!(
            "EmbeddingGemma 2 text encode on Apple M5 Pro: 60a6240 against eg2/integrate ({} interleaved rounds)",
            rounds_of(&root)
        ),
        &[
            "Left: each lane's median over rounds of its min-of-N encode time, log scale. Middle: speed-up against \
             the first lane (1 / the median per-round ratio, whiskers at the lowest and highest round). Right: \
             device peak above the loaded model.",
        ],
        &legend,
    );
    let cells = fig.cells(&[1.2, 1.2, 1.0], &[1.0]);

    // Encode time, log.
    let mut times = Vec::new();
    for r in &rows {
        for l in &lanes {
            times.push(r.get(l)?.get("ms_min_median")?.pos()?);
        }
    }
    let (lo, hi) = log_span(min_of(&times)?, max_of(&times)?)?;
    let mut ax = Axes::new(
        cells[0],
        "Encode time per workload",
        &[],
        AxisCfg::categories(&works, false)?.rotated(35.0),
        AxisCfg::log(lo, hi)?.labelled("encode, ms (median over rounds of min-of-N; log)"),
    )?;
    let w = 0.8 / lanes.len() as f64;
    for (k, r) in rows.iter().enumerate() {
        for (i, l) in lanes.iter().enumerate() {
            let x = k as f64 + (i as f64 - (lanes.len() as f64 - 1.0) / 2.0) * w;
            ax.bar(
                x,
                w * 0.92,
                (0.0, r.get(l)?.get("ms_min_median")?.pos()?),
                &Paint::new(color_of(FINAL_COLORS, l)),
            );
        }
    }
    fig.add(ax);

    // Speed-up against the base lane, log, with whiskers.
    let others: Vec<&String> = lanes.iter().filter(|l| **l != base).collect();
    let mut tops = Vec::new();
    let mut bottoms = Vec::new();
    for r in &rows {
        for l in &others {
            let v = r.get(l)?.get("vs_base")?;
            tops.push(1.0 / v.get("lo")?.pos()?);
            bottoms.push(1.0 / v.get("hi")?.pos()?);
        }
    }
    let lo = min_of(&bottoms)?.min(1.0) * 0.92;
    let top = max_of(&tops)?.max(1.0);
    let scale = Scale::headroom(Kind::Log, lo, top, cells[1].h - CHROME_PX, 48.0)?;
    let mut ax = Axes::new(
        cells[1],
        &format!("Speed-up against 60a6240 ({base})"),
        &[],
        AxisCfg::categories(&works, false)?.rotated(35.0),
        AxisCfg::fixed(scale, &[0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
            .labelled(&format!("speed-up vs {base} (log; whiskers: lowest/highest round)")),
    )?;
    ax.axhline(1.0, &Stroke::muted(1.0).dashed());
    let w = 0.8 / others.len() as f64;
    for (k, r) in rows.iter().enumerate() {
        for (i, l) in others.iter().enumerate() {
            let x = k as f64 + (i as f64 - (others.len() as f64 - 1.0) / 2.0) * w;
            let v = r.get(l)?.get("vs_base")?;
            let sp = 1.0 / v.get("median")?.pos()?;
            let (wlo, whi) = (1.0 / v.get("hi")?.pos()?, 1.0 / v.get("lo")?.pos()?);
            ax.bar(x, w * 0.92, (0.0, sp), &Paint::new(color_of(FINAL_COLORS, l)));
            ax.whisker(x, (wlo, whi), 3.0, &Stroke::ink(1.0));
            ax.text(x, whi, (3.0, -5.0), &format!("{sp:.2}x"), &Ts::new(9.5).ink().rot(90.0));
        }
    }
    fig.add(ax);

    // Device memory, linear.
    let mut peaks = Vec::new();
    for r in &rows {
        for l in &lanes {
            peaks.push(match r.get(l)?.opt("device_peak_mib_max") {
                Some(p) => p.f()?,
                None => 0.0,
            });
        }
    }
    let top = max_of(&peaks)?;
    let mut ax = Axes::new(
        cells[2],
        "Device memory",
        &[],
        AxisCfg::categories(&works, false)?.rotated(35.0),
        AxisCfg::linear(0.0, if top > 0.0 { top * 1.06 } else { 1.0 })?
            .labelled("device peak above the model, MiB (max over rounds)"),
    )?;
    let w = 0.8 / lanes.len() as f64;
    for (k, r) in rows.iter().enumerate() {
        for (i, l) in lanes.iter().enumerate() {
            let x = k as f64 + (i as f64 - (lanes.len() as f64 - 1.0) / 2.0) * w;
            let p = match r.get(l)?.opt("device_peak_mib_max") {
                Some(p) => p.f()?,
                None => 0.0,
            };
            ax.bar(x, w * 0.92, (0.0, p), &Paint::new(color_of(FINAL_COLORS, l)));
        }
    }
    fig.add(ax);
    Ok(Output::svg(fig.finish()))
}

// ---------------------------------------------------------------------------
// audio_speed
// ---------------------------------------------------------------------------

const BEFORE_AFTER_COLORS: &[(&str, &str)] = &[("before", BLUE), ("after", ORANGE), ("control", GREY)];

/// `embedgemma2_audio_speed_m5pro.json`: every round, the median and the ratio
/// against `before` for each audio workload.
pub fn audio_speed(json: &Path) -> R<Output> {
    let doc = load(json)?;
    let root = N::root(&doc, "audio_speed");
    let rows = root.get("summary")?.nonempty()?;
    let lanes = lane_names(&root)?;
    let base = lanes[0].clone();
    let legend: Vec<LegendItem> = lanes
        .iter()
        .map(|l| LegendItem::boxed(l, color_of(BEFORE_AFTER_COLORS, l)))
        .collect();
    let mut fig = Figure::new(
        390.0 * rows.len() as f64,
        470.0,
        "EmbeddingGemma 2 audio tower: before (df5d329) vs after, interleaved rounds, Apple M5 Pro",
        &[
            "Label: median per-round ratio to before, [lowest, highest round]. Control = the before binary again: \
             its own ratio and range are the noise floor of this run; a difference inside it is not a result.",
        ],
        &legend,
    );
    let cells = fig.cells(&vec![1.0; rows.len()], &[1.0]);
    for (i, row) in rows.iter().enumerate() {
        let mut top = Vec::new();
        for l in &lanes {
            top.push(max_of(&row.get(l)?.get("ms_min_all")?.floats()?)?);
        }
        let ymax = max_of(&top)? * 1.34;
        let mut ax = Axes::new(
            cells[i],
            &blank(row.get("workload")?.s()?),
            &[],
            AxisCfg::categories(&lanes, false)?,
            if i == 0 {
                AxisCfg::linear(0.0, ymax)?.labelled("encode_audio, ms (min of N per round; median bar)")
            } else {
                AxisCfg::linear(0.0, ymax)?
            },
        )?;
        for (x, l) in lanes.iter().enumerate() {
            let d = row.get(l)?;
            let label = if *l != base {
                Some(ratio_label(&d.get("vs_base")?)?)
            } else {
                None
            };
            strip_lane(&mut ax, x as f64, &d, color_of(BEFORE_AFTER_COLORS, l), label)?;
        }
        fig.add(ax);
    }
    Ok(Output::svg(fig.finish()))
}

// ---------------------------------------------------------------------------
// mm_onewait
// ---------------------------------------------------------------------------

/// Host waits of one call per lane. `bench_paired` does not carry the
/// `sync_waits` field the lanes printed into its JSON, so this restates what
/// all 24 processes of each lane printed (see the results `.txt`).
const MM_WAITS: &[(&str, u32)] = &[("before", 2), ("after", 1), ("control", 2)];

/// `embedgemma2_mm_onewait_m5pro.json`: the facade's call time per workload and
/// lane, and device peak with host waits.
pub fn mm_onewait(json: &Path) -> R<Output> {
    let doc = load(json)?;
    let root = N::root(&doc, "mm_onewait");
    let rows = root.get("summary")?.nonempty()?;
    let lanes = lane_names(&root)?;
    let base = lanes[0].clone();
    let legend: Vec<LegendItem> = lanes
        .iter()
        .map(|l| LegendItem::boxed(l, color_of(BEFORE_AFTER_COLORS, l)))
        .collect();
    let mut fig = Figure::new(
        390.0 * rows.len() as f64,
        760.0,
        &format!(
            "EmbeddingGemma 2 multimodal facade: before (27a0407, 2 host waits) vs after (1 wait), {} interleaved rounds, Apple M5 Pro",
            rounds_of(&root)
        ),
        &[
            "Label: median per-round ratio to before, [lowest, highest round]; 'best' = fastest round. Control = \
             the before binary again: its own ratio and range are the noise floor of this run; a difference \
             inside it is not a result. Time axes are zoomed.",
        ],
        &legend,
    );
    let cells = fig.cells(&vec![1.0; rows.len()], &[3.0, 1.4]);
    let n = rows.len();
    for (col, row) in rows.iter().enumerate() {
        let mut all_lo = Vec::new();
        let mut all_hi = Vec::new();
        for l in &lanes {
            let v = row.get(l)?.get("ms_min_all")?.floats()?;
            all_lo.push(min_of(&v)?);
            all_hi.push(max_of(&v)?);
        }
        let (lo_all, hi_all) = (min_of(&all_lo)?, max_of(&all_hi)?);
        let span = if hi_all > lo_all {
            hi_all - lo_all
        } else {
            hi_all * 0.05
        };
        let mut ax = Axes::new(
            cells[col],
            &blank(row.get("workload")?.s()?),
            &[],
            AxisCfg::categories(&lanes, false)?,
            if col == 0 {
                AxisCfg::linear(lo_all - 0.13 * span, hi_all + 0.4 * span)?
                    .labelled("encode, ms (min of N per round; bar = median)")
            } else {
                AxisCfg::linear(lo_all - 0.13 * span, hi_all + 0.4 * span)?
            },
        )?;
        for (x, l) in lanes.iter().enumerate() {
            let d = row.get(l)?;
            let label = if *l != base {
                Some(ratio_label(&d.get("vs_base")?)?)
            } else {
                None
            };
            strip_lane(&mut ax, x as f64, &d, color_of(BEFORE_AFTER_COLORS, l), label)?;
            let best = min_of(&d.get("ms_min_all")?.floats()?)?;
            ax.text(
                x as f64,
                best,
                (0.0, 14.0),
                &format!("best {best:.1}"),
                &Ts::new(10.0).mid().va(Va::Top),
            );
        }
        fig.add(ax);

        let mut peaks = Vec::new();
        for l in &lanes {
            peaks.push(row.get(l)?.get("device_peak_mib_max")?.f()?);
        }
        let ptop = max_of(&peaks)?;
        let mut bx = Axes::new(
            cells[n + col],
            "",
            &[],
            AxisCfg::categories(&lanes, false)?,
            if col == 0 {
                AxisCfg::linear(0.0, if ptop > 0.0 { ptop * 1.45 } else { 1.0 })?
                    .labelled("device peak above model, MiB")
            } else {
                AxisCfg::linear(0.0, if ptop > 0.0 { ptop * 1.45 } else { 1.0 })?
            },
        )?;
        for (x, (l, p)) in lanes.iter().zip(&peaks).enumerate() {
            bx.bar(x as f64, 0.55, (0.0, *p), &Paint::new(color_of(BEFORE_AFTER_COLORS, l)));
            let waits = MM_WAITS.iter().find(|(name, _)| name == l).map(|(_, w)| *w);
            let label = match waits {
                Some(w) => format!("{p:.1}\n{w} wait{}", if w == 1 { "" } else { "s" }),
                None => format!("{p:.1}"),
            };
            bx.text(
                x as f64,
                *p,
                (0.0, -4.0),
                &label,
                &Ts::new(10.5).mid().ink().va(Va::Bottom),
            );
        }
        fig.add(bx);
    }
    Ok(Output::svg(fig.finish()))
}

// ---------------------------------------------------------------------------
// pre
// ---------------------------------------------------------------------------

const PRE_COLORS: &[(&str, &str)] = &[("tessl", ORANGE), ("hf", BLUE)];

/// `embedgemma2_pre_m5pro.json`: preprocessing time, tessl against the Hugging
/// Face processors, per workload on a log axis.
pub fn pre(json: &Path) -> R<Output> {
    let doc = load(json)?;
    let root = N::root(&doc, "pre");
    let rows = root.get("summary")?.nonempty()?;
    let names: Vec<String> = rows
        .iter()
        .map(|r| r.get("workload")?.s().map(str::to_string))
        .collect::<R<_>>()?;
    let mut vals = Vec::new();
    let mut tops = Vec::new();
    for r in &rows {
        let mut row_top = Vec::new();
        for lane in ["tessl", "hf"] {
            let d = r.get(lane)?;
            vals.push(d.get("ms_min_median")?.pos()?);
            let all = d.get("ms_min_all")?.positives()?;
            vals.extend(&all);
            row_top.push(max_of(&all)?);
        }
        tops.push(max_of(&row_top)?);
    }
    let lo = nice_floor(min_of(&vals)? * 0.7)?;
    let top = max_of(&tops)?;
    let cell_h = 520.0;
    let scale = Scale::headroom(Kind::Log, lo, top * 1.1, cell_h - CHROME_PX, 40.0)?;
    let mut fig = Figure::new(
        1150.0,
        cell_h,
        "EmbeddingGemma 2 preprocessing, tessl native vs transformers processors, same inputs, Apple M5 Pro (shared host)",
        &["Label: median per-round ratio hf / tessl [lowest, highest round]; above 1 = tessl faster. Bar and tick: median over rounds of min-of-N; dots: every round."],
        &[
            LegendItem::boxed("tessl (embedgemma2_pre)", ORANGE),
            LegendItem::boxed("transformers 5.19 (AutoProcessor)", BLUE),
        ],
    );
    let cells = fig.cells(&[1.0], &[1.0]);
    let mut ax = Axes::new(
        cells[0],
        "",
        &[],
        AxisCfg::categories(&names, false)?.rotated(20.0),
        AxisCfg::from_scale(scale)?
            .labelled("ms per call (min-of-N per round; bar = median over rounds; dots = rounds)"),
    )?;
    let width = 0.38;
    for (i, r) in rows.iter().enumerate() {
        for (j, lane) in ["tessl", "hf"].iter().enumerate() {
            let d = r.get(lane)?;
            let c = color_of(PRE_COLORS, lane);
            let x = i as f64 + (j as f64 - 0.5) * width;
            let med = d.get("ms_min_median")?.pos()?;
            ax.bar(x, width * 0.92, (0.0, med), &Paint::new(c).alpha(0.28));
            ax.line(
                "median",
                (x - width * 0.46, med),
                (x + width * 0.46, med),
                &Stroke::color(c, 2.5),
            );
            let all = d.get("ms_min_all")?.positives()?;
            let n = all.len() as f64;
            for (k, v) in all.iter().enumerate() {
                ax.dot(
                    x + (k as f64 - (n - 1.0) / 2.0) * width * 0.9 / n,
                    *v,
                    3.2,
                    &Paint::new(c).bg_edge(0.8),
                );
            }
        }
        let ratio = ratio_label(&r.get("hf")?.get("vs_base")?)?;
        ax.text(
            i as f64,
            tops[i],
            (0.0, -7.0),
            &ratio,
            &Ts::new(10.5).mid().ink().va(Va::Bottom),
        );
    }
    fig.add(ax);
    Ok(Output::svg(fig.finish()))
}
