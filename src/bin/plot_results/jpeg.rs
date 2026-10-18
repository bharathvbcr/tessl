//! `jpeg-decode`: the native JPEG decoder against the ImageIO path it replaced
//! (`bench_jpeg_decode`), two horizontal bars per file at their minimum with a
//! tick at their median, on a log axis because the files span three decades of
//! size.

use crate::chart::{Axes, AxisCfg, Figure, LegendItem, Paint, Stroke, Ts, Va, BLUE, ORANGE};
use crate::data::{max_of, min_of, N, R};
use crate::paired::{load, log_span};
use crate::registry::Output;
use std::path::Path;

/// Vertical offset of the two bars of one file from the file's row centre.
const LANE: f64 = 0.19;

pub fn jpeg_decode(json: &Path) -> R<Output> {
    let doc = load(json)?;
    let rows = N::root(&doc, "jpeg_decode").nonempty()?;
    let mut labels = Vec::with_capacity(rows.len());
    // (native min, native median, ImageIO min, ImageIO median) per file.
    let mut times = Vec::with_capacity(rows.len());
    for r in &rows {
        labels.push(format!(
            "{}  ({:.2} MP)",
            r.get("workload")?.s()?,
            r.get("megapixels")?.pos()?
        ));
        times.push((
            r.get("native_ms_min")?.pos()?,
            r.get("native_ms_median")?.pos()?,
            r.get("imageio_ms_min")?.pos()?,
            r.get("imageio_ms_median")?.pos()?,
        ));
    }
    let all: Vec<f64> = times.iter().flat_map(|t| [t.0, t.1, t.2, t.3]).collect();
    // Room on the right for the value label of the longest bar.
    let (lo, hi) = log_span(min_of(&all)?, max_of(&all)? * 4.0)?;

    let legend = [
        LegendItem::boxed("native (tessl::media::jpeg, libjpeg-turbo exact)", BLUE),
        LegendItem::boxed("ImageIO (the path it replaced; differs by up to 43 levels)", ORANGE),
        LegendItem::ink_line("median", false),
    ];
    let mut fig = Figure::new(
        1100.0,
        90.0 + 56.0 * rows.len() as f64,
        "JPEG decoding, native against ImageIO, one call per file (Apple M5 Pro, shared machine)",
        &["Bars are the minimum of N interleaved calls; the tick is the median. Indicative, not a paired comparison."],
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
    for (y, t) in times.iter().enumerate() {
        let y = y as f64;
        for (dy, min, med, color) in [(-LANE, t.0, t.1, BLUE), (LANE, t.2, t.3, ORANGE)] {
            ax.hbar(y + dy, 0.34, (lo, min), &Paint::new(color).alpha(0.85));
            ax.line("median", (med, y + dy - 0.17), (med, y + dy + 0.17), &Stroke::ink(2.0));
            ax.text(
                min.max(med),
                y + dy,
                (8.0, 0.0),
                &format!("{min:.1} ms"),
                &Ts::new(10.5).va(Va::Middle),
            );
        }
    }
    fig.add(ax);
    Ok(Output::svg(fig.finish()))
}
