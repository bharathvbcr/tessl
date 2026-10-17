//! `embedgemma2_small_m_m5pro.json` (`results_report small-m`): left, the
//! whole-encode time after the small-M GEMM lane against before it (the ratio
//! of every interleaved round, with the median and the control lane's spread
//! as the noise band); right, the per-GEMM time at 16 rows (`bench_gemm_routes`
//! lane `nnsmall`: plain single dispatch against the split-K lane, interleaved).

use crate::chart::{nice_top, spread, Axes, AxisCfg, Figure, LegendItem, Paint, Scale, Stroke, M_BLUE, M_ORANGE};
use crate::data::{max_of, min_of, N, R};
use crate::paired::load;
use crate::registry::Output;
use std::path::Path;

fn prec_color(prec: &str) -> &'static str {
    if prec == "f32" {
        M_BLUE
    } else {
        M_ORANGE
    }
}

pub fn small_m(json: &Path) -> R<Output> {
    let doc = load(json)?;
    let root = N::root(&doc, "small_m");
    let lane = |k: &str| -> R<String> { Ok(root.get("lane_names")?.get(k)?.s()?.to_string()) };
    let (before, after, control) = (lane("before")?, lane("after")?, lane("control")?);
    let workloads: Vec<String> = root
        .get("workloads")?
        .nonempty()?
        .iter()
        .map(|w| w.s().map(str::to_string))
        .collect::<R<_>>()?;
    let precs = root.get("encode")?.entries()?;
    if precs.is_empty() {
        return Err(format!("{}: no precision", root.get("encode")?.path()));
    }

    // Whole encode: after / before per round, and the control against before.
    struct Group {
        prec: String,
        ratios: Vec<Vec<f64>>,
        noise: Vec<Vec<f64>>,
    }
    let mut groups = Vec::new();
    for (prec, paired) in &precs {
        let summary = paired.get("summary")?;
        let (mut ratios, mut noise) = (Vec::new(), Vec::new());
        for w in &workloads {
            let row = summary.find("workload", w)?;
            let b = row.get(&before)?.get("ms_min_all")?.positives()?;
            let a = row.get(&after)?.get("ms_min_all")?.positives()?;
            let c = row.get(&control)?.get("ms_min_all")?.positives()?;
            if a.len() != b.len() || c.len() != b.len() {
                return Err(format!("{}: the lanes ran different numbers of rounds", row.path()));
            }
            ratios.push(a.iter().zip(&b).map(|(x, y)| x / y).collect::<Vec<f64>>());
            noise.push(c.iter().zip(&b).map(|(x, y)| x / y).collect::<Vec<f64>>());
        }
        groups.push(Group {
            prec: prec.clone(),
            ratios,
            noise,
        });
    }
    let all: Vec<f64> = groups
        .iter()
        .flat_map(|g| g.ratios.iter().chain(g.noise.iter()).flatten().copied())
        .chain([1.0])
        .collect();
    let lo = min_of(&all)? * 0.9;
    let hi = max_of(&all)? * 1.1;

    // Right: per-GEMM time at 16 rows.
    let gemm = root.get("gemm_16_rows")?;
    let first = precs[0].0.clone();
    let names: Vec<String> = gemm
        .get(&first)?
        .nonempty()?
        .iter()
        .map(|r| r.get("name")?.s().map(str::to_string))
        .collect::<R<_>>()?;
    let mut gtop = Vec::new();
    for (prec, _) in &precs {
        for r in gemm.get(prec)?.nonempty()? {
            gtop.push(r.get("plain_us")?.f()?);
            gtop.push(r.get("production_us")?.f()?);
        }
    }

    let mut left_legend = Vec::new();
    for g in &groups {
        left_legend.push(LegendItem::dot(
            &format!("{}: after / before", g.prec),
            prec_color(&g.prec),
        ));
    }
    left_legend.push(LegendItem::cross("control: before again / before", "#555555"));
    let mut right_legend = Vec::new();
    for (prec, _) in &precs {
        right_legend.push(LegendItem::faint(
            &format!("{prec}: single dispatch"),
            prec_color(prec),
            0.45,
        ));
        right_legend.push(LegendItem::boxed(&format!("{prec}: split-K lane"), prec_color(prec)));
    }
    let mut fig = Figure::new(
        1300.0,
        560.0,
        "EmbeddingGemma 2 small-M GEMM lane: whole encode before and after, and one GEMM at 16 rows (Apple M5 Pro)",
        &[],
        &[],
    );
    let cells = fig.cells(&[1.15, 1.0], &[1.0]);

    let mut ax = Axes::new(
        cells[0],
        "Whole encode, interleaved rounds (log scale; bar: range; black: median)",
        &left_legend,
        AxisCfg::categories(&workloads, false)?.labelled("workload (sequences x tokens)"),
        AxisCfg::from_scale(Scale::log(lo, hi)?)?.labelled("encode time, after / before (lower is faster)"),
    )?;
    ax.axhline(1.0, &Stroke::ink(0.8).dashed());
    let width = 0.36;
    for (j, g) in groups.iter().enumerate() {
        let c = prec_color(&g.prec);
        for (i, ratios) in g.ratios.iter().enumerate() {
            let x = i as f64 + spread(j, groups.len(), width);
            ax.line(
                "range",
                (x, min_of(ratios)?),
                (x, max_of(ratios)?),
                &Stroke::color(c, 6.0).alpha(0.35),
            );
            for r in ratios {
                ax.dot(x, *r, 3.0, &Paint::new(c));
            }
            // The upper median, as the script's `sorted(ratios)[len // 2]`.
            let mut sorted = ratios.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let med = sorted[sorted.len() / 2];
            ax.line(
                "median",
                (x - width / 3.0, med),
                (x + width / 3.0, med),
                &Stroke::ink(2.0),
            );
            // The control lane against the first: what noise looks like.
            for r in &g.noise[i] {
                ax.cross(x + width / 2.2, *r, 3.0, "#555555");
            }
        }
    }
    fig.add(ax);

    let gtop = nice_top(max_of(&gtop)? * 1.1)?;
    let mut bx = Axes::new(
        cells[1],
        "One GEMM at 16 rows (median of interleaved rounds)",
        &right_legend,
        AxisCfg::categories(&names, false)?.rotated(30.0),
        AxisCfg::linear(0.0, gtop)?.labelled("us per GEMM (median of interleaved rounds)"),
    )?;
    for (j, (prec, _)) in precs.iter().enumerate() {
        let c = prec_color(prec);
        let rows = gemm.get(prec)?.nonempty()?;
        let off = spread(j, precs.len(), 0.36);
        for (p, r) in rows.iter().enumerate() {
            let x = p as f64 + off;
            bx.bar(
                x - 0.08,
                0.16,
                (0.0, r.get("plain_us")?.f()?),
                &Paint::new(c).alpha(0.45),
            );
            bx.bar(x + 0.08, 0.16, (0.0, r.get("production_us")?.f()?), &Paint::new(c));
        }
    }
    fig.add(bx);
    Ok(Output::svg(fig.finish()))
}
