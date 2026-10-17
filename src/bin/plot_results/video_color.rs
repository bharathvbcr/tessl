//! `video-color`: video decoding to RGB under the two colour paths
//! (`bench_media_decode` on `bench/make_video_bench_inputs.py`'s clips): per
//! clip and call, one bar for the exact path (VideoToolbox planes converted by
//! libswscale's arithmetic) and one for AVFoundation's own BGRA conversion, at
//! the minimum of N, a tick at the median.
//!
//! The workloads are `FILE:video_sampled_N:PATH` and `FILE:video_first_K:PATH`
//! with `PATH` `exact` or `bgra`; the `FILE:video_open` rows (no colour path)
//! are not drawn.

use crate::chart::{Axes, AxisCfg, Figure, LegendItem, Paint, Stroke, Ts, Va, BLUE, ORANGE};
use crate::data::{max_of, N, R};
use crate::registry::Output;
use std::path::Path;

/// One call under one colour path.
struct Call {
    min: f64,
    median: f64,
    frames: f64,
}

/// One clip and call (`FILE  video_first_30`) with its two paths.
struct Group {
    label: String,
    exact: Option<Call>,
    bgra: Option<Call>,
}

fn groups(rows: &[N]) -> R<Vec<Group>> {
    let mut out: Vec<Group> = Vec::new();
    for r in rows {
        let workload = r.get("workload")?.s()?;
        let parts: Vec<&str> = workload.split(':').collect();
        let (file, call, path) = match parts.as_slice() {
            [_, "video_open"] => continue,
            [file, call, path] => (*file, *call, *path),
            _ => {
                return Err(format!(
                    "workload {workload:?}: expected FILE:video_CALL:PATH or FILE:video_open"
                ))
            }
        };
        if r.get("unit")?.s()? != "frame" {
            return Err(format!("workload {workload:?}: a video workload counts frames"));
        }
        let call_row = Call {
            min: r.get("ms_min")?.pos()?,
            median: r.get("ms_median")?.pos()?,
            frames: r.get("units")?.pos()?,
        };
        let label = format!("{file}  {call} ({} frames)", crate::chart::fmt_tick(call_row.frames));
        let at = match out.iter().position(|g| g.label == label) {
            Some(at) => at,
            None => {
                out.push(Group {
                    label,
                    exact: None,
                    bgra: None,
                });
                out.len() - 1
            }
        };
        let slot = match path {
            "exact" => &mut out[at].exact,
            "bgra" => &mut out[at].bgra,
            other => {
                return Err(format!(
                    "workload {workload:?}: unknown colour path {other:?}; one of exact, bgra"
                ))
            }
        };
        if slot.replace(call_row).is_some() {
            return Err(format!("workload {workload:?} appears twice"));
        }
    }
    if out.is_empty() {
        return Err("no video_sampled/video_first workloads to plot".to_string());
    }
    for g in &out {
        if g.exact.is_none() || g.bgra.is_none() {
            return Err(format!("{}: needs both an exact and a bgra row", g.label));
        }
    }
    Ok(out)
}

pub fn video_color(json: &Path) -> R<Output> {
    let doc = crate::paired::load(json)?;
    let rows = N::root(&doc, "video_color").nonempty()?;
    let groups = groups(&rows)?;
    let labels: Vec<String> = groups.iter().map(|g| g.label.clone()).collect();
    let calls = || {
        groups
            .iter()
            .flat_map(|g| [g.exact.as_ref(), g.bgra.as_ref()])
            .flatten()
    };
    let hi = max_of(&calls().map(|c| c.min.max(c.median)).collect::<Vec<_>>())?;
    // Room on the right for the value label of the longest bar.
    let hi = crate::chart::nice_ceil(hi * 1.6)?;

    let legend = [
        LegendItem::boxed("exact: VideoToolbox planes, libswscale arithmetic", BLUE),
        LegendItem::boxed("bgra: AVFoundation's own conversion", ORANGE),
        LegendItem::line("median", "#0b0b0b", false),
    ];
    let mut fig = Figure::new(
        1100.0,
        90.0 + 56.0 * groups.len() as f64,
        "Video to RGB frames under the two colour paths (Apple M5 Pro, shared machine)",
        &[
            "Wall time of one call, a freshly opened source each: bar = minimum of N, tick = median. Indicative, not a paired comparison.",
        ],
        &legend,
    );
    let cell = fig.cells(&[1.0], &[1.0])[0];
    let mut ax = Axes::new(
        cell,
        "",
        &[],
        AxisCfg::linear(0.0, hi)?.labelled("wall time per call, ms (min of N; tick = median)"),
        AxisCfg::categories(&labels, true)?,
    )?;
    for (y, g) in groups.iter().enumerate() {
        for (offset, call, color) in [(-0.2, g.exact.as_ref(), BLUE), (0.2, g.bgra.as_ref(), ORANGE)] {
            let call = call.ok_or("a group lost a row")?;
            let y = y as f64 + offset;
            ax.hbar(y, 0.36, (0.0, call.min), &Paint::new(color).alpha(0.85));
            ax.line(
                "median",
                (call.median, y - 0.18),
                (call.median, y + 0.18),
                &Stroke::ink(2.0),
            );
            ax.text(
                call.min.max(call.median),
                y,
                (8.0, 0.0),
                &format!("{:.0} ms ({:.1} ms/frame)", call.min, call.min / call.frames),
                &Ts::new(10.5).va(Va::Middle),
            );
        }
    }
    fig.add(ax);
    Ok(Output::svg(fig.finish()))
}
