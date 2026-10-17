//! `resample`: wall time of `decode_audio_file` to 16 kHz mono per file and
//! sample-rate converter (`bench_media_resample`): the file's own decode alone,
//! libswresample's algorithm in Rust, the system `AudioConverter`, and the Rust
//! converter by itself. One horizontal bar per (file, stage) at its minimum, a
//! tick at its median, on a log axis because the stages span two decades.

use crate::chart::{Axes, AxisCfg, Figure, LegendItem, Paint, Stroke, Ts, Va, BLUE, GOLD, GREEN, ORANGE};
use crate::data::{max_of, min_of, N, R};
use crate::paired::{load, log_span};
use crate::registry::Output;
use std::path::Path;

/// The bar colour of a stage.
fn stage_color(stage: &str) -> R<&'static str> {
    match stage {
        "native decode" => Ok(GOLD),
        "swresample" => Ok(BLUE),
        "audiotoolbox" => Ok(ORANGE),
        "swresample converter alone" => Ok(GREEN),
        other => Err(format!(
            "unknown stage {other:?}; one of native decode, swresample, audiotoolbox, swresample converter alone"
        )),
    }
}

pub fn resample(json: &Path) -> R<Output> {
    let doc = load(json)?;
    let rows = N::root(&doc, "media_resample").nonempty()?;
    let mut labels = Vec::with_capacity(rows.len());
    let mut bars = Vec::with_capacity(rows.len());
    for r in &rows {
        let stage = r.get("stage")?.s()?;
        let secs = r.get("audio_s")?.pos()?;
        labels.push(format!(
            "{}  {}  ({} s)",
            r.get("workload")?.s()?,
            stage,
            crate::chart::fmt_tick(secs)
        ));
        bars.push((r.get("ms_min")?.pos()?, r.get("ms_median")?.pos()?, stage_color(stage)?));
    }
    let lo = min_of(&bars.iter().map(|b| b.0).collect::<Vec<_>>())?;
    let hi = max_of(&bars.iter().map(|b| b.0.max(b.1)).collect::<Vec<_>>())?;
    // Room on the right for the value label of the longest bar.
    let (lo, hi) = log_span(lo, hi * 4.0)?;

    let legend = [
        LegendItem::boxed("decode at the file's rate (no conversion)", GOLD),
        LegendItem::boxed("libswresample's algorithm, in Rust (default)", BLUE),
        LegendItem::boxed("AudioToolbox AudioConverter", ORANGE),
        LegendItem::boxed("the Rust converter alone, one channel", GREEN),
        LegendItem::ink_line("median", false),
    ];
    let mut fig = Figure::new(
        1100.0,
        60.0 + 26.0 * rows.len() as f64,
        "tessl::media sample-rate conversion to 16 kHz mono, 30 s of 44.1 kHz stereo (Apple M5 Pro, shared machine)",
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
