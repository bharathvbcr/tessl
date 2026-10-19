//! `go-overhead`: what the Go binding adds to a tessl call
//! (`embedgemma2_go_m5pro.json`, written by `go/cmd/bench_overhead`). Left, the
//! time of the same call made straight from Rust against tessl's C ABI (with the
//! model on the main thread, and on a thread of its own like the engine's worker)
//! and through the Go package and gusset, per workload on a log axis: bars at the
//! minimum of N, a tick at the median. Right, what the Go call adds: its wall
//! time minus the engine's own timing of the same call, per call, so the GPU
//! time is the same on both sides of the difference; and the cost of a round trip
//! with no work in it.

use crate::chart::{
    nice_top, Axes, AxisCfg, Figure, LegendItem, Paint, Scale, Stroke, Ts, Va, BLUE, GREY, LIGHT_GREY, ORANGE,
};
use crate::data::{max_of, min_of, N, R};
use crate::paired::{load, log_span};
use crate::registry::Output;
use std::fmt::Write as _;
use std::path::Path;

pub fn go_overhead(json: &Path) -> R<Output> {
    let doc = load(json)?;
    let root = N::root(&doc, "go_overhead");
    let rows = root.get("rows")?.nonempty()?;
    let ping = root.get("ping")?.get("min_us")?.pos()?;
    let (rounds, iters) = (root.get("rounds")?.pos()?, root.get("iters_per_round")?.pos()?);
    // The machine's state is the run's record (`vm.loadavg` before and after), so
    // the figure says what it was rather than assuming it.
    let host = root.get("host")?;
    let (load_before, load_after) = (
        one_minute_load(&host.get("loadavg_before")?)?,
        one_minute_load(&host.get("loadavg_after")?)?,
    );

    struct Row {
        label: String,
        main_min: f64,
        main_med: f64,
        direct_min: f64,
        direct_med: f64,
        go_min: f64,
        go_med: f64,
        vs_direct: f64,
        paired_min: f64,
        paired_med: f64,
        ratio: f64,
    }
    let mut data = Vec::with_capacity(rows.len());
    for r in &rows {
        data.push(Row {
            label: r.get("workload")?.s()?.to_string(),
            main_min: r.get("direct_main_min_us")?.pos()?,
            main_med: r.get("direct_main_median_us")?.pos()?,
            direct_min: r.get("direct_min_us")?.pos()?,
            direct_med: r.get("direct_median_us")?.pos()?,
            go_min: r.get("go_min_us")?.pos()?,
            go_med: r.get("go_median_us")?.pos()?,
            // Go minus direct is between two processes' GPU times as well, so it
            // may be negative; it is quoted in the table and not drawn.
            vs_direct: r.get("vs_direct_min_us")?.f()?,
            // The paired figure cannot be: Go wall includes the engine's timing
            // of the same call. Zero or less means the file is wrong.
            paired_min: r.get("paired_overhead_min_us")?.pos()?,
            paired_med: r.get("paired_overhead_median_us")?.pos()?,
            ratio: r.get("ratio_min")?.pos()?,
        });
    }
    let labels: Vec<String> = data.iter().map(|d| d.label.clone()).collect();

    let times: Vec<f64> = data
        .iter()
        .flat_map(|d| [d.main_min, d.main_med, d.direct_min, d.direct_med, d.go_min, d.go_med])
        .collect();
    let (lo, hi) = log_span(min_of(&times)?, max_of(&times)?)?;

    let mut table = String::new();
    writeln!(
        table,
        "{:<14}{:>13}{:>13}{:>11}{:>14}{:>14}{:>11}",
        "workload", "main us", "direct us", "go us", "go-direct us", "paired us", "go/direct"
    )
    .map_err(|e| e.to_string())?;
    for d in &data {
        writeln!(
            table,
            "{:<14}{:>13.1}{:>13.1}{:>11.1}{:>14.1}{:>14.1}{:>11.3}",
            d.label, d.main_min, d.direct_min, d.go_min, d.vs_direct, d.paired_min, d.ratio
        )
        .map_err(|e| e.to_string())?;
    }
    writeln!(
        table,
        "{:<14}min {ping:.1} us (a round trip with no work)",
        "ping (noop)"
    )
    .map_err(|e| e.to_string())?;

    let left_legend = [
        LegendItem::boxed("tessl's C ABI from Rust, model on the main thread", LIGHT_GREY),
        LegendItem::boxed("the same, on a thread of its own (the engine's worker)", GREY),
        LegendItem::boxed("the Go package through gusset", BLUE),
        LegendItem::ink_line("median", false),
    ];
    let right_legend = [
        LegendItem::boxed("Go wall minus the engine's timing of the same call", ORANGE),
        LegendItem::ink_line("median", false),
        LegendItem::line("a round trip with no work (Ping, minimum)", GREY, true),
    ];
    let notes = format!(
        "Tiny multimodal checkpoint, one call per point; minimum of {} calls ({} rounds of {}) per arm, the arms alternating in separate \
         processes, and all arms' outputs are bit-identical. 1-minute load average {load_before:.1} before, {load_after:.1} after.",
        rounds as u64 * iters as u64,
        rounds as u64,
        iters as u64
    );
    let mut fig = Figure::new(
        1400.0,
        580.0,
        "tessl from Go over gusset: the cost of the boundary against the same call made from Rust (Apple M5 Pro)",
        &[notes.as_str()],
        &[],
    );
    let cells = fig.cells(&[1.3, 1.0], &[1.0]);

    let mut ax = Axes::new(
        cells[0],
        "Time per call (log scale; bar: minimum; black tick: median; label: Go / worker-thread direct)",
        &left_legend,
        AxisCfg::categories(&labels, false)?.rotated(30.0),
        AxisCfg::from_scale(Scale::log(lo, hi)?)?.labelled("wall time per call, us"),
    )?;
    let w = 0.26;
    for (i, d) in data.iter().enumerate() {
        let x = i as f64;
        for (dx, min, med, color) in [
            (-0.28, d.main_min, d.main_med, LIGHT_GREY),
            (0.0, d.direct_min, d.direct_med, GREY),
            (0.28, d.go_min, d.go_med, BLUE),
        ] {
            ax.bar(x + dx, w, (lo, min), &Paint::new(color).alpha(0.9));
            ax.line(
                "median",
                (x + dx - w / 2.0, med),
                (x + dx + w / 2.0, med),
                &Stroke::ink(2.0),
            );
        }
        ax.text(
            x + 0.28,
            d.go_min,
            (0.0, -6.0),
            &format!("{:.2}x", d.ratio),
            &Ts::new(10.0).mid().va(Va::Bottom),
        );
    }
    fig.add(ax);

    let tops: Vec<f64> = data
        .iter()
        .flat_map(|d| [d.paired_min, d.paired_med])
        .chain([ping])
        .collect();
    let top = nice_top(max_of(&tops)? * 1.2)?;
    let mut bx = Axes::new(
        cells[1],
        "What the Go call adds (same call, same GPU time)",
        &right_legend,
        AxisCfg::categories(&labels, false)?.rotated(30.0),
        AxisCfg::linear(0.0, top)?.labelled("added wall time per call, us"),
    )?;
    bx.axhline(ping, &Stroke::muted(1.2).dashed());
    for (i, d) in data.iter().enumerate() {
        let x = i as f64;
        bx.bar(x, 0.5, (0.0, d.paired_min), &Paint::new(ORANGE).alpha(0.9));
        bx.line(
            "median",
            (x - 0.25, d.paired_med),
            (x + 0.25, d.paired_med),
            &Stroke::ink(2.0),
        );
        bx.text(
            x,
            d.paired_min.max(d.paired_med),
            (0.0, -6.0),
            &format!("{:.0}", d.paired_min),
            &Ts::new(10.0).mid().va(Va::Bottom),
        );
    }
    fig.add(bx);
    Ok(Output {
        svg: fig.finish(),
        table,
    })
}

/// The 1-minute figure of a `vm.loadavg` record, `"{ 3.87 5.25 6.69 }"`.
fn one_minute_load(v: &N) -> R<f64> {
    let text = v.s()?;
    let first = text.trim().trim_start_matches('{').split_whitespace().next();
    match first.map(str::parse::<f64>) {
        Some(Ok(x)) if x.is_finite() && x >= 0.0 => Ok(x),
        _ => Err(format!("{}: {text:?} is not a vm.loadavg record", v.path())),
    }
}
