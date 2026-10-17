//! The device-memory figure (`results_report memory`'s JSON) and the small-M
//! GEMM figure (`results_report small-m`'s JSON).

use crate::chart::{
    nice_top, spread, Axes, AxisCfg, Figure, LegendItem, Paint, Scale, Stroke, Ts, Va, M_BLUE, M_GREEN, M_GREY, M_RED,
};
use crate::data::{max_of, N, R};
use crate::paired::load;
use crate::registry::Output;
use std::path::Path;

fn ops_title(ops: &str) -> &'static str {
    if ops == "f32" {
        "ExactF32"
    } else {
        "Bf16"
    }
}

const MEM_WLS: [&str; 5] = ["1x16", "64x32", "32x256", "8x1024", "2x4096"];
const DIAMOND: &str = "#dd8452";

/// The activation-set itemisation at 8192 rows, ExactF32: what 16f5190 held,
/// the two reductions the lane made (the per-layer-input bound from 256 to
/// 16 MiB, and q|k|v sharing a buffer with gate|up), and what the changed
/// build holds. The two ends are measured; the steps are the lane's
/// accounting, and the figure refuses to draw when they do not add up to the
/// measured end.
const PLE_BOUND_STEP: f64 = -240.0;

/// One precision's per-layer-input sweep: its name, its colour, and the points
/// (workload, the sweep's `vs_ple256.ple16` entry) in padded-row order.
type Sweep<'a> = (&'a str, &'a str, Vec<(String, N<'a>)>);
const QKV_ALIAS_STEP: f64 = -96.0;

/// `embedgemma2_memory_m5pro.json`: device peak per workload (before and
/// after), the activation-set waterfall, paired time ratios per workload, and
/// the cost of the per-layer-input bound.
pub fn memory(json: &Path) -> R<Output> {
    let doc = load(json)?;
    let root = N::root(&doc, "memory");
    let mut cells: Vec<((String, String, String), N)> = Vec::new();
    for c in root.get("memory")?.get("cells")?.nonempty()? {
        cells.push((
            (
                c.get("operands")?.s()?.to_string(),
                c.get("workload")?.s()?.to_string(),
                c.get("config")?.s()?.to_string(),
            ),
            c,
        ));
    }
    let cell = |ops: &str, wl: &str, cfg: &str| -> R<N> {
        cells
            .iter()
            .find(|((o, w, c), _)| o == ops && w == wl && c == cfg)
            .map(|(_, n)| n.clone())
            .ok_or_else(|| {
                format!(
                    "{}: no cell {ops}/{wl}/{cfg}",
                    root.get("memory").map(|m| m.path().to_string()).unwrap_or_default()
                )
            })
    };
    let wls: Vec<String> = MEM_WLS.iter().map(|s| s.to_string()).collect();

    let mut fig = Figure::new(
        1500.0,
        900.0,
        "EmbeddingGemma 2 encode, device memory and time: lane H against 16f5190 (M5 Pro, real checkpoint)",
        &[],
        &[],
    );
    let grid = fig.cells(&[1.0, 1.0, 1.0], &[1.0, 1.05]);

    // Row 1: device peak above the model, back-to-back encodes, both builds.
    for (col, ops) in ["f32", "bf16"].iter().enumerate() {
        let (mut b, mut a, mut b1) = (Vec::new(), Vec::new(), Vec::new());
        for w in MEM_WLS {
            b.push(cell(ops, w, "back-to-back")?.get("before_mib")?.f()?);
            a.push(cell(ops, w, "back-to-back")?.get("after_mib")?.f()?);
            b1.push(cell(ops, w, "single")?.get("before_mib")?.f()?);
        }
        let top = max_of(&b)?.max(max_of(&a)?).max(max_of(&b1)?);
        let legend = if col == 0 {
            vec![
                LegendItem::boxed("16f5190, back-to-back encodes", M_RED),
                LegendItem::boxed("changed, back-to-back encodes", M_BLUE),
                LegendItem::line("16f5190, one encode", "#0b0b0b", false),
            ]
        } else {
            Vec::new()
        };
        let mut ax = Axes::new(
            grid[col],
            &format!("{}: peak of the last of four encodes", ops_title(ops)),
            &legend,
            AxisCfg::categories(&wls, false)?,
            AxisCfg::linear(0.0, nice_top(top * 1.12)?)?.labelled("device peak above the loaded model (MiB)"),
        )?;
        for i in 0..wls.len() {
            let x = i as f64;
            ax.bar(x - 0.2, 0.38, (0.0, b[i]), &Paint::new(M_RED));
            ax.bar(x + 0.2, 0.38, (0.0, a[i]), &Paint::new(M_BLUE));
            ax.tick(x - 0.2, b1[i], 13.0, &Stroke::ink(2.0));
            ax.text(
                x - 0.2,
                b[i],
                (0.0, -4.0),
                &format!("{:.0}", b[i]),
                &Ts::new(9.5).mid().ink().va(Va::Bottom),
            );
            ax.text(
                x + 0.2,
                a[i],
                (0.0, -4.0),
                &format!("{:.0}", a[i]),
                &Ts::new(9.5).mid().ink().va(Va::Bottom),
            );
        }
        fig.add(ax);
    }

    // Row 1, right: the 736 -> 400 MiB itemisation of one activation set.
    let start = cell("f32", "32x256", "single")?.get("before_mib")?.f()?;
    let end = cell("f32", "32x256", "single")?.get("after_mib")?.f()?;
    let predicted = start + PLE_BOUND_STEP + QKV_ALIAS_STEP;
    if (predicted - end).abs() > 0.5 {
        return Err(format!(
            "the itemisation does not add up: {start:.2} {PLE_BOUND_STEP} {QKV_ALIAS_STEP} = {predicted:.2}, but the cell holds {end:.2}"
        ));
    }
    let labels: Vec<String> = [
        "16f5190\none set",
        "PLE bound\n256 -> 16 MiB",
        "q|k|v = gate|up\none buffer",
        "changed\none set",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let mut ax = Axes::new(
        grid[2],
        "one activation set (single encode)",
        &[],
        AxisCfg::categories(&labels, false)?,
        AxisCfg::linear(0.0, nice_top(start * 1.1)?)?.labelled("MiB, 8192 rows, ExactF32"),
    )?;
    ax.bar(0.0, 0.8, (0.0, start), &Paint::new(M_RED));
    ax.text(
        0.0,
        start,
        (0.0, -4.0),
        &format!("{start:.0}"),
        &Ts::new(10.5).mid().ink().va(Va::Bottom),
    );
    let mut run = start;
    for (i, v) in [PLE_BOUND_STEP, QKV_ALIAS_STEP].iter().enumerate() {
        ax.bar((i + 1) as f64, 0.8, (run + v, run), &Paint::new(M_GREY));
        ax.text(
            (i + 1) as f64,
            run,
            (0.0, -4.0),
            &format!("{v:+.0}"),
            &Ts::new(10.5).mid().ink().va(Va::Bottom),
        );
        run += v;
    }
    ax.bar(3.0, 0.8, (0.0, end), &Paint::new(M_BLUE));
    ax.text(
        3.0,
        end,
        (0.0, -4.0),
        &format!("{end:.0}"),
        &Ts::new(10.5).mid().ink().va(Va::Bottom),
    );
    fig.add(ax);

    // Row 2: time, after / before per workload, both modes.
    for (col, ops) in ["f32", "bf16"].iter().enumerate() {
        let p = root.get("paired")?.get(ops)?;
        let series = p.get("series_ms")?;
        let before = series.get("before")?.entries()?;
        let names: Vec<String> = before.iter().map(|(k, _)| k.clone()).collect();
        if names.is_empty() {
            return Err(format!("{}: no workloads", series.get("before")?.path()));
        }
        let n_rounds = p.get("rounds")?.nonempty()?.len();
        let ld = p.get("load_1m_normalized")?;
        let (lo, hi) = (0.5, 2.0);
        let mut ax = Axes::new(
            grid[3 + col],
            &format!(
                "{}: {n_rounds} paired rounds (dots), median (bar), min-of-N (diamond)\nshared machine, load/core median {:.1}, max {:.1}",
                ops_title(ops),
                ld.get("median")?.f()?,
                ld.get("max")?.f()?
            ),
            &[],
            AxisCfg::categories(&names, false)?.rotated(15.0),
            AxisCfg::fixed(Scale::linear(lo, hi)?, &[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
                .labelled("changed / 16f5190 (encode ms, per round)"),
        )?;
        let after_series = series.get("after")?;
        let mut off_scale = 0usize;
        let mut total = 0usize;
        for (i, wl) in names.iter().enumerate() {
            let a = after_series.get(wl)?.floats()?;
            let b = series.get("before")?.get(wl)?.floats()?;
            if a.len() != b.len() {
                return Err(format!(
                    "{}: {} rounds after against {} before",
                    series.path(),
                    a.len(),
                    b.len()
                ));
            }
            for (x, y) in a.iter().zip(&b) {
                if *y <= 0.0 {
                    return Err(format!("{}.{wl}: a before time of {y} makes no ratio", series.path()));
                }
                let r = x / y;
                total += 1;
                if r < lo || r > hi {
                    off_scale += 1;
                }
                ax.dot(i as f64, r, 3.2, &Paint::new(M_BLUE).alpha(0.45));
            }
            let s = p.get("vs_before")?.get("after")?.get(wl)?;
            let med = s.get("median_of_round_ratios")?.f()?;
            ax.line(
                "median",
                (i as f64 - 0.3, med),
                (i as f64 + 0.3, med),
                &Stroke::ink(2.0),
            );
            ax.diamond(i as f64, s.get("min_of_n_ratio")?.f()?, 5.5, &Paint::new(DIAMOND));
        }
        ax.axhline(1.0, &Stroke::ink(0.8));
        ax.axhline(1.03, &Stroke::color(M_RED, 0.8).dashed());
        ax.note(
            0.99,
            0.02,
            &format!("{off_scale} of {total} rounds off scale"),
            &Ts::new(10.0).end().dim(),
        );
        fig.add(ax);
    }

    // Row 2, right: the per-layer-input bound's own cost by padded rows.
    const ROWS: [(&str, u32); 6] = [
        ("32x32", 1024),
        ("64x32", 2048),
        ("128x32", 4096),
        ("32x256", 8192),
        ("8x1024", 8192),
        ("2x4096", 8192),
    ];
    let sweeps = root.get("ple_sweeps")?.entries()?;
    let mut by_ops: Vec<Sweep> = Vec::new();
    for (ops, color) in [("f32", M_BLUE), ("bf16", M_GREEN)] {
        let mut pts: Vec<(String, N)> = Vec::new();
        for (name, sweep) in &sweeps {
            // The first bf16 run at 1024-4096 rows ran at a load of 14-38 per core and is in the text
            // only; a "repeat" is the same cell run again (text only).
            if !name.starts_with(ops) || name.contains("first run") || name.contains("repeat") {
                continue;
            }
            for (wl, s) in sweep.get("vs_ple256")?.get("ple16")?.entries()? {
                pts.push((wl, s));
            }
        }
        let rows_of = |wl: &str| -> R<u32> {
            ROWS.iter()
                .find(|(w, _)| *w == wl)
                .map(|(_, r)| *r)
                .ok_or_else(|| format!("no padded-row count known for workload {wl:?}"))
        };
        let mut keyed = Vec::new();
        for (wl, s) in pts {
            keyed.push(((rows_of(&wl)?, wl.clone()), (wl, s)));
        }
        keyed.sort_by(|a, b| a.0.cmp(&b.0));
        by_ops.push((ops, color, keyed.into_iter().map(|(_, v)| v).collect()));
    }
    let labels: Vec<String> = by_ops
        .last()
        .map(|(_, _, pts)| pts.iter().map(|(wl, _)| wl.clone()).collect())
        .unwrap_or_default();
    if labels.is_empty() {
        return Err(format!("{}: no PLE sweep points", root.get("ple_sweeps")?.path()));
    }
    for (ops, _, pts) in &by_ops {
        let l: Vec<&String> = pts.iter().map(|(wl, _)| wl).collect();
        if l.len() != labels.len() || l.iter().zip(&labels).any(|(a, b)| *a != b) {
            return Err(format!(
                "{}: the {ops} sweep covers different workloads than the bf16 sweep",
                root.get("ple_sweeps")?.path()
            ));
        }
    }
    let legend = vec![
        LegendItem::boxed("ExactF32: median of rounds", M_BLUE),
        LegendItem::boxed("Bf16: median of rounds", M_GREEN),
        LegendItem::diamond("min-of-N", DIAMOND),
    ];
    let mut ax = Axes::new(
        grid[5],
        "time cost of the 16 MiB per-layer-input bound\n(dashed: the 1.03 limit fixed in advance)",
        &legend,
        AxisCfg::categories(&labels, false)?.rotated(20.0),
        AxisCfg::fixed(
            Scale::linear(0.8, 1.25)?,
            &[0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25],
        )
        .labelled("16 MiB bound / 256 MiB bound (encode time)"),
    )?;
    for (k, (_, color, pts)) in by_ops.iter().enumerate() {
        for (i, (_, s)) in pts.iter().enumerate() {
            let x = i as f64 + spread(k, by_ops.len(), 0.3);
            ax.bar(
                x,
                0.28,
                (0.0, s.get("median_of_round_ratios")?.f()?),
                &Paint::new(color),
            );
            ax.diamond(x, s.get("min_of_n_ratio")?.f()?, 5.0, &Paint::new(DIAMOND));
        }
    }
    ax.axhline(1.0, &Stroke::ink(0.8));
    ax.axhline(1.03, &Stroke::color(M_RED, 0.8).dashed());
    fig.add(ax);
    Ok(Output::svg(fig.finish()))
}
