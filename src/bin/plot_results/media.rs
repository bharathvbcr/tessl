//! `media-decode`: native decode time per workload (`bench_media_decode`),
//! one horizontal bar per workload at its minimum, a tick at its median, on a
//! log axis because the workloads span four decades.

use crate::chart::{Axes, AxisCfg, Figure, LegendItem, Paint, Stroke, Ts, Va, BLUE, GREEN, ORANGE};
use crate::data::{max_of, min_of, N, R};
use crate::paired::{load, log_span};
use crate::registry::Output;
use std::path::Path;

/// The bar colour of a workload's unit.
fn unit_color(unit: &str) -> R<&'static str> {
    match unit {
        "megapixel" => Ok(ORANGE),
        "audio_s" => Ok(BLUE),
        "frame" => Ok(GREEN),
        other => Err(format!("unknown unit {other:?}; one of megapixel, audio_s, frame")),
    }
}

pub fn media_decode(json: &Path) -> R<Output> {
    let doc = load(json)?;
    let rows = N::root(&doc, "media_decode").nonempty()?;
    let mut labels = Vec::with_capacity(rows.len());
    let mut bars = Vec::with_capacity(rows.len());
    for r in &rows {
        let unit = r.get("unit")?.s()?;
        let units = r.get("units")?.pos()?;
        labels.push(format!(
            "{}  ({} {})",
            r.get("workload")?.s()?,
            crate::chart::fmt_tick(units),
            unit.replace('_', " ")
        ));
        bars.push((r.get("ms_min")?.pos()?, r.get("ms_median")?.pos()?, unit_color(unit)?));
    }
    let lo = min_of(&bars.iter().map(|b| b.0).collect::<Vec<_>>())?;
    let hi = max_of(&bars.iter().map(|b| b.0.max(b.1)).collect::<Vec<_>>())?;
    // Room on the right for the value label of the longest bar.
    let (lo, hi) = log_span(lo, hi * 4.0)?;

    let legend = [
        LegendItem::boxed("image (megapixels)", ORANGE),
        LegendItem::boxed("audio (seconds of audio)", BLUE),
        LegendItem::boxed("video (frames)", GREEN),
        LegendItem::ink_line("median", false),
    ];
    let mut fig = Figure::new(
        1100.0,
        60.0 + 34.0 * rows.len() as f64,
        "tessl::media native decoding, one call per workload (Apple M5 Pro, shared machine)",
        &["Bars are the minimum of N; the tick is the median. Indicative, not a paired comparison."],
        &legend,
    );
    let cell = fig.cells(&[1.0], &[1.0])[0];
    let mut ax = Axes::new(
        cell,
        "",
        &[],
        AxisCfg::log(lo, hi)?.labelled("wall time per call, ms (min of N; tick = median)"),
        AxisCfg::categories(&labels, true)?,
    )?;
    for (y, (min, med, color)) in bars.iter().enumerate() {
        let y = y as f64;
        ax.hbar(y, 0.62, (lo, *min), &Paint::new(color).alpha(0.85));
        ax.line("median", (*med, y - 0.31), (*med, y + 0.31), &Stroke::ink(2.0));
        ax.text(
            min.max(*med),
            y,
            (8.0, 0.0),
            &format!("{min:.1} ms"),
            &Ts::new(10.5).va(Va::Middle),
        );
    }
    fig.add(ax);
    Ok(Output::svg(fig.finish()))
}
