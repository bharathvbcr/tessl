//! `video-color`: video decoding to RGB under the colour paths
//! (`bench_media_decode` on `bench/make_video_bench_inputs.py`'s clips): per
//! clip and call, one bar for the exact path (VideoToolbox planes converted by
//! libswscale's arithmetic, rows split over threads), optionally one for the same
//! on one thread, and one for AVFoundation's own BGRA conversion, at the minimum
//! of N, a tick at the median.
//!
//! The workloads are `FILE:video_sampled_N:PATH` and `FILE:video_first_K:PATH`
//! with `PATH` `exact`, `exact1` or `bgra`; the `FILE:video_open` rows (no colour
//! path) and the converter-alone rows `yuv:LAYOUT_WxH:single|threads` are not
//! drawn. A run made before the converter was threaded has no `exact1` rows and
//! draws two bars per call; a run with any `exact1` row must have one for every
//! call and draws three.

use crate::chart::{Axes, AxisCfg, Figure, LegendItem, Paint, Stroke, Ts, Va, BLUE, GREEN, ORANGE};
use crate::data::{max_of, N, R};
use crate::registry::Output;
use std::path::Path;

/// One call under one colour path.
struct Call {
    min: f64,
    median: f64,
    frames: f64,
}

/// One clip and call (`FILE  video_first_30`) with its paths.
struct Group {
    label: String,
    exact: Option<Call>,
    exact1: Option<Call>,
    bgra: Option<Call>,
}

fn groups(rows: &[N]) -> R<Vec<Group>> {
    let mut out: Vec<Group> = Vec::new();
    for r in rows {
        let workload = r.get("workload")?.s()?;
        let parts: Vec<&str> = workload.split(':').collect();
        let (file, call, path) = match parts.as_slice() {
            [_, "video_open"] | ["yuv", _, _] => continue,
            [file, call, path] => (*file, *call, *path),
            _ => {
                return Err(format!(
                    "workload {workload:?}: expected FILE:video_CALL:PATH, FILE:video_open or yuv:LAYOUT:PATH"
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
                    exact1: None,
                    bgra: None,
                });
                out.len() - 1
            }
        };
        let slot = match path {
            "exact" => &mut out[at].exact,
            "exact1" => &mut out[at].exact1,
            "bgra" => &mut out[at].bgra,
            other => {
                return Err(format!(
                    "workload {workload:?}: unknown colour path {other:?}; one of exact, exact1, bgra"
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
    let single = out.iter().any(|g| g.exact1.is_some());
    for g in &out {
        if g.exact.is_none() || g.bgra.is_none() {
            return Err(format!("{}: needs both an exact and a bgra row", g.label));
        }
        if single && g.exact1.is_none() {
            return Err(format!(
                "{}: other calls have an exact1 row, this one needs it too",
                g.label
            ));
        }
    }
    Ok(out)
}

pub fn video_color(json: &Path) -> R<Output> {
    let doc = crate::paired::load(json)?;
    let rows = N::root(&doc, "video_color").nonempty()?;
    let groups = groups(&rows)?;
    let three = groups.iter().any(|g| g.exact1.is_some());
    let labels: Vec<String> = groups.iter().map(|g| g.label.clone()).collect();
    let calls = || {
        groups
            .iter()
            .flat_map(|g| [g.exact.as_ref(), g.exact1.as_ref(), g.bgra.as_ref()])
            .flatten()
    };
    let hi = max_of(&calls().map(|c| c.min.max(c.median)).collect::<Vec<_>>())?;
    // Room on the right for the value label of the longest bar.
    let hi = crate::chart::nice_ceil(hi * 1.6)?;

    // (offset, half height of the median tick, bar thickness) per bar of a call.
    let (offsets, tick, thickness): (&[f64], f64, f64) = if three {
        (&[-0.27, 0.0, 0.27], 0.12, 0.24)
    } else {
        (&[-0.2, 0.2], 0.18, 0.36)
    };
    let mut legend = vec![LegendItem::boxed(
        if three {
            "exact: VideoToolbox planes, libswscale arithmetic, threaded rows"
        } else {
            "exact: VideoToolbox planes, libswscale arithmetic"
        },
        BLUE,
    )];
    if three {
        legend.push(LegendItem::boxed("exact1: the same on one thread", GREEN));
    }
    legend.push(LegendItem::boxed("bgra: AVFoundation's own conversion", ORANGE));
    legend.push(LegendItem::ink_line("median", false));
    let mut fig = Figure::new(
        1100.0,
        if three {
            100.0 + 78.0 * groups.len() as f64
        } else {
            90.0 + 56.0 * groups.len() as f64
        },
        if three {
            "Video to RGB frames under the colour paths (Apple M5 Pro)"
        } else {
            "Video to RGB frames under the two colour paths (Apple M5 Pro)"
        },
        &[
            "Wall time of one call, a freshly opened source each: bar = minimum of N, tick = median. Indicative, not a paired comparison; \
             the machine's load is recorded in the .txt beside the data.",
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
        let bars: Vec<(Option<&Call>, _)> = if three {
            vec![
                (g.exact.as_ref(), BLUE),
                (g.exact1.as_ref(), GREEN),
                (g.bgra.as_ref(), ORANGE),
            ]
        } else {
            vec![(g.exact.as_ref(), BLUE), (g.bgra.as_ref(), ORANGE)]
        };
        for (&offset, (call, color)) in offsets.iter().zip(bars) {
            let call = call.ok_or("a group lost a row")?;
            let y = y as f64 + offset;
            ax.hbar(y, thickness, (0.0, call.min), &Paint::new(color).alpha(0.85));
            ax.line(
                "median",
                (call.median, y - tick),
                (call.median, y + tick),
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
