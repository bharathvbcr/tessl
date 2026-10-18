//! Per-stage profile figures: where a forward's GPU time goes.
//!
//! * [`audio_profile`]: the audio tower before and after, stages grouped by the
//!   work they do, with the grouped table the results file quotes.
//! * [`stage_profile`]: the text encoder's stage profile (`results_report stage`).
//! * [`text_fusion`]: the text encoder before and after the row-pass fusion
//!   (`results_report text-fusion`).

use crate::chart::{
    nice_top, Axes, AxisCfg, Figure, Kind, LegendItem, Paint, Scale, Stroke, Ts, Va, BLUE, GOLD, GREEN, GREY,
    LIGHT_GREY, M_BLUE, M_GREEN, M_GREY, M_ORANGE, M_RED, ORANGE,
};
use crate::data::{max_of, min_of, N, R};
use crate::paired::{load, log_span};
use crate::registry::Output;
use std::path::Path;

/// The smallest of an array of numbers, or an error naming the path.
fn min_floats(n: &N) -> R<f64> {
    min_of(&n.floats()?)
}

// ---------------------------------------------------------------------------
// audio_profile
// ---------------------------------------------------------------------------

const AUDIO_GROUPS: [&str; 4] = ["GEMMs (exact f32)", "attention", "subsampling convs", "row passes"];
const AUDIO_COLORS: [&str; 4] = [BLUE, ORANGE, "#1baf7a", "#eda100"];

fn audio_group(stage: &str) -> usize {
    if stage.contains("GEMM") {
        0
    } else if stage.starts_with("attention") {
        1
    } else if stage.starts_with("subsample") {
        2
    } else {
        3
    }
}

/// A workload's alone-ms per group and its best encode time.
fn audio_grouped(profile: &N, workload: &str) -> R<([f64; 4], f64)> {
    let w = profile.get("workloads")?.find("workload", workload)?;
    let mut out = [0.0; 4];
    for s in w.get("stages")?.nonempty()? {
        out[audio_group(s.get("stage")?.s()?)] += s.get("alone_ms")?.f()?;
    }
    let enc = min_floats(&w.get("variants")?.find("name", "encode")?.get("rounds_ms")?)?;
    Ok((out, enc))
}

/// `embedgemma2_audio_profile_m5pro.json`: the figure, and the grouped table
/// the results file quotes.
pub fn audio_profile(json: &Path) -> R<Output> {
    let doc = load(json)?;
    let root = N::root(&doc, "audio_profile");
    let before = root.get("before")?;
    let after = root.get("after")?;
    let workloads: Vec<String> = before
        .get("workloads")?
        .nonempty()?
        .iter()
        .map(|w| w.get("workload")?.s().map(str::to_string))
        .collect::<R<_>>()?;

    let mut table = format!("{:<14}{:<8}", "workload", "side");
    for g in AUDIO_GROUPS {
        table.push_str(&format!("{g:>22}"));
    }
    table.push_str(&format!("{:>12}{:>12}\n", "encode ms", "other/GEMM"));

    let legend: Vec<LegendItem> = AUDIO_GROUPS
        .iter()
        .zip(AUDIO_COLORS)
        .map(|(g, c)| LegendItem::boxed(g, c))
        .collect();
    let row_h = 205.0;
    let mut fig = Figure::new(
        950.0,
        row_h * workloads.len() as f64,
        "EmbeddingGemma 2 audio tower, per-stage GPU time (Apple M5 Pro, shared GPU)",
        &[
            "The GEMM kernels did not change (q|k|v is one wider GEMM): read the other bars against the GEMM bar of \
             the same row.",
        ],
        &legend,
    );
    let cells = fig.cells(&[1.0], &vec![1.0; workloads.len()]);
    for (wl, cell) in workloads.iter().zip(&cells) {
        let mut rows = Vec::new();
        for (side, profile) in [("before", &before), ("after", &after)] {
            let (g, enc) = audio_grouped(profile, wl)?;
            let other: f64 = g[1..].iter().sum();
            if g[0] == 0.0 {
                return Err(format!(
                    "{}: no GEMM time in {side} of {wl}, other/GEMM undefined",
                    profile.path()
                ));
            }
            table.push_str(&format!("{wl:<14}{side:<8}"));
            for v in g {
                table.push_str(&format!("{v:>22.2}"));
            }
            table.push_str(&format!("{enc:>12.2}{:>12.3}\n", other / g[0]));
            rows.push((side, g));
        }
        let max_sum = max_of(&rows.iter().map(|(_, g)| g.iter().sum::<f64>()).collect::<Vec<_>>())?;
        if max_sum <= 0.0 {
            return Err(format!("{}: {wl} has no stage time", before.path()));
        }
        let scale = Scale::headroom(Kind::Linear, 0.0, max_sum, cell.w - 120.0, 175.0)?;
        let mut ax = Axes::new(
            *cell,
            &wl.replace('_', " "),
            &[],
            AxisCfg::from_scale(scale)?.labelled("stage time alone, ms (min over rounds, harness floor removed)"),
            AxisCfg::categories(&["before".to_string(), "after".to_string()], true)?,
        )?;
        for (yi, (_, g)) in rows.iter().enumerate() {
            let mut left = 0.0;
            for (k, c) in AUDIO_COLORS.iter().enumerate() {
                ax.hseg(yi as f64, 0.62, (left, left + g[k]), &Paint::new(c).bg_edge(2.0));
                left += g[k];
            }
            ax.text(
                left,
                yi as f64,
                (8.0, 0.0),
                &format!("{left:.1} ms (sum of stages)"),
                &Ts::new(10.5).va(Va::Middle),
            );
        }
        fig.add(ax);
    }
    Ok(Output {
        svg: fig.finish(),
        table,
    })
}

// ---------------------------------------------------------------------------
// stage_profile
// ---------------------------------------------------------------------------

const STAGE_GROUPS: [(&str, &[&str]); 4] = [
    (
        "GEMMs, exact f32",
        &[
            "qkv GEMM",
            "o GEMM",
            "gate|up GEMM",
            "down GEMM",
            "ple_gate GEMM",
            "ple_out GEMM",
            "PLE projection",
        ],
    ),
    ("attention sliding (hd256)", &["attention sliding"]),
    ("attention global (hd512)", &["attention global"]),
    (
        "elementwise: norms, residual adds, gelu, rope, scale, gather, tail",
        &[
            "gelu*up",
            "ple gelu",
            "rms_norm",
            "norm+residual",
            "qk-norm",
            "PLE scale",
            "gather",
            "tail",
        ],
    ),
];
const STAGE_COLORS: [&str; 4] = [M_BLUE, M_ORANGE, M_RED, M_GREY];

fn stage_group(name: &str) -> R<usize> {
    STAGE_GROUPS
        .iter()
        .position(|(_, prefixes)| prefixes.iter().any(|p| name.starts_with(p)))
        .ok_or_else(|| format!("stage class {name:?} belongs to no group of the figure"))
}

/// The best (lowest) `ms_per_forward` over all passes for a workload and
/// variant `kind` (and class, for `only`).
fn best(passes: &[(String, N)], workload: &str, kind: &str, class: Option<&str>) -> R<f64> {
    let mut vals = Vec::new();
    for (_, p) in passes {
        for w in p.get("workloads")?.items()? {
            if w.get("workload")?.s()? != workload {
                continue;
            }
            for v in w.get("variants")?.items()? {
                if v.get("kind")?.s()? != kind {
                    continue;
                }
                if let Some(c) = class {
                    let name = v.get("name")?.s()?;
                    let tail = name.split_once(':').map(|(_, t)| t).unwrap_or(name);
                    if tail != c {
                        continue;
                    }
                }
                vals.push(min_floats(&v.get("ms_per_forward")?)?);
            }
        }
    }
    min_of(&vals).map_err(|_| format!("no {kind} variant {class:?} for workload {workload} in any pass"))
}

/// Workloads `AxB` ordered by rows (`A*B`) and then by `B`.
fn workload_key(w: &str) -> R<(u64, u64)> {
    let (a, b) = w.split_once('x').ok_or_else(|| format!("workload {w:?} is not AxB"))?;
    let a: u64 = a.parse().map_err(|_| format!("workload {w:?} is not AxB"))?;
    let b: u64 = b.parse().map_err(|_| format!("workload {w:?} is not AxB"))?;
    Ok((a * b, b))
}

fn stacked(ax: &mut Axes, series: &[Vec<f64>], colors: &[&str]) {
    let mut bottom = vec![0.0; series.first().map(|s| s.len()).unwrap_or(0)];
    for (vals, c) in series.iter().zip(colors) {
        for (i, v) in vals.iter().enumerate() {
            ax.seg(i as f64, 0.8, (bottom[i], bottom[i] + v), &Paint::new(c));
            bottom[i] += v;
        }
    }
}

/// `embedgemma2_stage_profile_m5pro.json`: four panels (share of the forward,
/// absolute time, GEMM throughput, device memory).
pub fn stage_profile(json: &Path) -> R<Output> {
    let doc = load(json)?;
    let root = N::root(&doc, "stage_profile");
    let contended: Vec<String> = root
        .get("contended_passes")?
        .items()?
        .iter()
        .map(|c| c.s().map(str::to_string))
        .collect::<R<_>>()?;
    let passes: Vec<(String, N)> = root
        .get("passes")?
        .entries()?
        .into_iter()
        .filter(|(k, _)| !contended.contains(k))
        .collect();
    if passes.is_empty() {
        return Err(format!(
            "{}: every pass is contended, nothing to plot",
            root.get("passes")?.path()
        ));
    }
    let gemm = root.get("gemm")?;
    let mem = root.get("memory")?;
    let layout = root.get("layout")?;

    let mut wls: Vec<String> = Vec::new();
    for (_, p) in &passes {
        for w in p.get("workloads")?.items()? {
            let name = w.get("workload")?.s()?.to_string();
            if !wls.contains(&name) {
                wls.push(name);
            }
        }
    }
    let mut keyed = Vec::new();
    for w in &wls {
        keyed.push((workload_key(w)?, w.clone()));
    }
    keyed.sort();
    let wls: Vec<String> = keyed.into_iter().map(|(_, w)| w).collect();
    let classes: Vec<String> = passes[0]
        .1
        .get("classes")?
        .nonempty()?
        .iter()
        .map(|c| c.s().map(str::to_string))
        .collect::<R<_>>()?;
    let mut class_group = Vec::new();
    for c in &classes {
        class_group.push(stage_group(c)?);
    }

    let mut share = vec![vec![0.0; wls.len()]; STAGE_GROUPS.len()];
    let mut ms = vec![vec![0.0; wls.len()]; STAGE_GROUPS.len()];
    let mut fulls = Vec::new();
    for (i, w) in wls.iter().enumerate() {
        let full = best(&passes, w, "full", None)?;
        fulls.push(full);
        for g in 0..STAGE_GROUPS.len() {
            let mut t = 0.0;
            for (c, cg) in classes.iter().zip(&class_group) {
                if *cg == g {
                    t += best(&passes, w, "only", Some(c))?;
                }
            }
            ms[g][i] = t;
            share[g][i] = 100.0 * t / full;
        }
    }

    let legend: Vec<LegendItem> = STAGE_GROUPS
        .iter()
        .zip(STAGE_COLORS)
        .map(|((g, _), c)| LegendItem::boxed(g, c))
        .collect();
    let mut fig = Figure::new(
        1400.0,
        900.0,
        "EmbeddingGemma 2 text encoder, stage profile (Apple M5 Pro, async, each stage alone on real shapes and weights)",
        &[],
        &legend,
    );
    let cells = fig.cells(&[1.0, 1.0], &[1.0, 1.0]);

    // Share of the forward.
    let totals: Vec<f64> = (0..wls.len()).map(|i| share.iter().map(|s| s[i]).sum()).collect();
    let mut ax = Axes::new(
        cells[0],
        "Where a forward's GPU time goes (async; each stage alone, real shapes and weights)",
        &[],
        AxisCfg::categories(&wls, false)?,
        AxisCfg::linear(0.0, nice_top(max_of(&totals)? * 1.04)?)?
            .labelled("% of the full forward (stage alone, summed)"),
    )?;
    stacked(&mut ax, &share, &STAGE_COLORS);
    fig.add(ax);

    // Absolute time, log, with the whole forward as a tick.
    let sums: Vec<f64> = (0..wls.len()).map(|i| ms.iter().map(|s| s[i]).sum()).collect();
    let mut all = fulls.clone();
    all.extend(&sums);
    // Only the bottom class of a stack reaches the axis floor; the classes above it sit higher, so a
    // small segment there must not pull the floor down and squash every bar.
    all.extend(ms.first().into_iter().flatten().filter(|v| **v > 0.0));
    let (lo, hi) = log_span(min_of(&all)?, max_of(&all)?)?;
    let mut ax = Axes::new(
        cells[1],
        "Absolute time; black ticks are the whole replica forward",
        &[LegendItem::ink_line("full forward (min)", false)],
        AxisCfg::categories(&wls, false)?,
        AxisCfg::log(lo, hi)?.labelled("ms per forward (log)"),
    )?;
    stacked(&mut ax, &ms, &STAGE_COLORS);
    for (i, f) in fulls.iter().enumerate() {
        ax.tick(i as f64, *f, 20.0, &Stroke::ink(2.5));
    }
    fig.add(ax);

    // GEMM throughput at 8192 rows.
    let shapes: Vec<N> = gemm
        .get("gemm_shapes")?
        .nonempty()?
        .into_iter()
        .filter(|r| r.get("rows").and_then(|v| v.f()).map(|v| v == 8192.0).unwrap_or(false))
        .collect();
    if shapes.is_empty() {
        return Err(format!("{}: no GEMM shape at 8192 rows", gemm.path()));
    }
    let mut names = Vec::new();
    let mut f32t = Vec::new();
    let mut bft = Vec::new();
    for r in &shapes {
        names.push(r.get("gemm")?.s()?.to_string());
        let flops = r.get("flops")?.f()?;
        f32t.push(flops / (min_floats(&r.get("f32_ms")?)? * 1e-3) / 1e12);
        bft.push(flops / (min_floats(&r.get("bf16_ms")?)? * 1e-3) / 1e12);
    }
    let ceil_f32 = gemm.get("ceilings")?.get("f32_gemm_4096_tflops")?.f()?;
    let ceil_bf = gemm.get("ceilings")?.get("bf16_gemm_4096_tflops")?.f()?;
    let top = max_of(&f32t)?.max(max_of(&bft)?).max(ceil_f32).max(ceil_bf);
    let mut ax = Axes::new(
        cells[2],
        "The forward's GEMM shapes at 8192 rows, a distinct weight per layer",
        &[
            LegendItem::boxed("exact f32 (what runs today)", M_BLUE),
            LegendItem::boxed("bf16 operands, f32 accumulate (not in the forward)", M_GREEN),
            LegendItem::line("f32 4096^3 (measured)", M_BLUE, true),
            LegendItem::line("bf16 4096^3 (measured)", M_GREEN, true),
        ],
        AxisCfg::categories(&names, false)?.rotated(30.0),
        AxisCfg::linear(0.0, nice_top(top * 1.05)?)?.labelled("TFLOP/s"),
    )?;
    for i in 0..names.len() {
        ax.bar(i as f64 - 0.2, 0.4, (0.0, f32t[i]), &Paint::new(M_BLUE));
        ax.bar(i as f64 + 0.2, 0.4, (0.0, bft[i]), &Paint::new(M_GREEN));
    }
    ax.axhline(ceil_f32, &Stroke::color(M_BLUE, 1.0).dashed());
    ax.axhline(ceil_bf, &Stroke::color(M_GREEN, 1.0).dashed());
    fig.add(ax);

    // Device memory: one activation set, and what the bench table measured.
    let wl8 = layout
        .get("workloads")?
        .nonempty()?
        .into_iter()
        .find(|w| w.get("rows").and_then(|v| v.f()).map(|v| v == 8192.0).unwrap_or(false))
        .ok_or_else(|| format!("{}: no layout workload at 8192 rows", layout.path()))?;
    let mut runs: Vec<(String, N)> = Vec::new();
    for r in mem.get("runs")?.nonempty()? {
        let key = format!("{}|{}", r.get("label")?.s()?, r.get("result")?.get("workload")?.s()?);
        runs.push((key, r.get("result")?));
    }
    let run = |key: &str| -> R<N> {
        runs.iter()
            .find(|(k, _)| k == key)
            .map(|(_, r)| r.clone())
            .ok_or_else(|| format!("{}: no run {key:?}", mem.path()))
    };
    let mem_labels: Vec<String> = [
        "2a7cb8b\nset",
        "HEAD\nset",
        "HEAD set,\nPLE run 16 MiB",
        "HEAD,\nencode;encode",
        "bench table\n(after 1x16, 64x32)",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let vals = [
        wl8.get("2a7cb8b_total_mib")?.f()?,
        wl8.get("head_total_mib")?.f()?,
        run("ple16|8x1024")?.get("trace")?.nonempty()?[0]
            .get("allocated_above_model_mib")?
            .f()?,
        run("alone-back-to-back|8x1024")?.get("peak_above_model_mib")?.f()?,
        run("bench-order|32x256")?.get("peak_above_model_mib")?.f()?,
    ];
    let colors = [M_GREY, M_BLUE, M_GREEN, M_RED, M_RED];
    let top = max_of(&vals)?;
    let mut ax = Axes::new(
        cells[3],
        "Device memory: one activation set, and what the bench table measured",
        &[],
        AxisCfg::categories(&mem_labels, false)?,
        AxisCfg::linear(0.0, nice_top(top * 1.1)?)?.labelled("MiB above the loaded model, 8192 rows"),
    )?;
    for (i, (v, c)) in vals.iter().zip(colors).enumerate() {
        ax.bar(i as f64, 0.8, (0.0, *v), &Paint::new(c));
        ax.text(
            i as f64,
            *v,
            (0.0, -4.0),
            &format!("{v:.0}"),
            &Ts::new(10.5).mid().ink().va(Va::Bottom),
        );
    }
    fig.add(ax);
    Ok(Output::svg(fig.finish()))
}

// ---------------------------------------------------------------------------
// text_fusion
// ---------------------------------------------------------------------------

const FUSION_PREFIXES: [(&str, &[&str]); 6] = [
    (
        "GEMM",
        &[
            "qkv GEMM",
            "o GEMM",
            "gate|up GEMM",
            "down GEMM",
            "PLE projection",
            "ple_gate",
            "ple_out",
        ],
    ),
    ("attention", &["attention"]),
    (
        "row passes",
        &["rms_norm", "residual add", "norm+residual", "PLE scale", "bf16 casts"],
    ),
    ("gelu*up", &["gelu*up", "ple gelu"]),
    ("other elementwise", &["gather", "qk-norm"]),
    ("tail", &["tail"]),
];
const FUSION_COLORS: [(&str, &str); 6] = [
    ("GEMM", BLUE),
    ("attention", GREY),
    ("row passes", ORANGE),
    ("gelu*up", GREEN),
    ("other elementwise", GOLD),
    ("tail", LIGHT_GREY),
];

fn fusion_category(class: &str) -> R<&'static str> {
    FUSION_PREFIXES
        .iter()
        .find(|(_, prefixes)| prefixes.iter().any(|p| class.starts_with(p)))
        .map(|(name, _)| *name)
        .ok_or_else(|| format!("stage class {class:?} belongs to no category of the figure"))
}

/// Stage time per category and the dispatch count of one run label at one
/// workload: each class's time is its minimum over every pass of the label.
fn per_category(doc: &N, cats: &[String], label: &str, wl: &str) -> R<(Vec<f64>, f64)> {
    let mut best: Vec<(String, f64)> = Vec::new();
    let mut dispatches = None;
    for run in doc.get("passes")?.get(label)?.nonempty()? {
        let w = run.get("workloads")?.find("workload", wl)?;
        dispatches = Some(w.get("dispatches_full")?.f()?);
        for v in w.get("variants")?.items()? {
            if v.get("kind")?.s()? != "only" {
                continue;
            }
            let name = v.get("name")?.s()?;
            let cls = name
                .split_once(':')
                .map(|(_, c)| c)
                .ok_or_else(|| format!("{}: variant name {name:?} has no class", v.path()))?
                .to_string();
            let t = min_floats(&v.get("ms_per_forward")?)?;
            match best.iter_mut().find(|(c, _)| *c == cls) {
                Some((_, b)) => *b = b.min(t),
                None => best.push((cls, t)),
            }
        }
    }
    let mut ms = vec![0.0; cats.len()];
    for (cls, t) in best {
        let cat = fusion_category(&cls)?;
        let idx = cats
            .iter()
            .position(|c| c == cat)
            .ok_or_else(|| format!("{}: category {cat:?} is not in the file's categories", doc.path()))?;
        ms[idx] += t;
    }
    let d = dispatches.ok_or_else(|| format!("{}: no passes for {label}", doc.path()))?;
    Ok((ms, d))
}

/// `embedgemma2_text_fusion_profile_m5pro.json`: stage time and dispatches per
/// forward, before and after the row-pass fusion, at three workloads.
pub fn text_fusion(json: &Path) -> R<Output> {
    let doc = load(json)?;
    let root = N::root(&doc, "text_fusion");
    let cats: Vec<String> = root
        .get("categories")?
        .nonempty()?
        .iter()
        .map(|c| c.s().map(str::to_string))
        .collect::<R<_>>()?;
    let runs = root.get("runs")?.entries()?;
    let labels: Vec<String> = ["f32 unfused", "f32 fused", "bf16 unfused", "bf16 fused"]
        .iter()
        .filter(|l| runs.iter().any(|(k, _)| k == *l))
        .map(|l| l.to_string())
        .collect();
    if labels.is_empty() {
        return Err(format!(
            "{}: none of the four run labels is present",
            root.get("runs")?.path()
        ));
    }
    let workloads = ["1x16", "64x32", "8x1024"];
    let cat_color = |c: &str| FUSION_COLORS.iter().find(|(n, _)| *n == c).map(|(_, col)| *col);
    let mut colors = Vec::new();
    for c in &cats {
        colors.push(cat_color(c).ok_or_else(|| format!("{}: no colour for category {c:?}", root.path()))?);
    }

    let legend: Vec<LegendItem> = cats
        .iter()
        .zip(&colors)
        .map(|(c, col)| LegendItem::boxed(c, col))
        .collect();
    let mut fig = Figure::new(
        1500.0,
        860.0,
        "EmbeddingGemma 2 text encoder, stage profile before and after the row-pass fusion (M5 Pro, shared GPU, min over 3 passes)",
        &[],
        &legend,
    );
    let cells = fig.cells(&vec![1.0; workloads.len()], &[1.0, 1.0]);
    for (j, wl) in workloads.iter().enumerate() {
        let mut per = Vec::new();
        for l in &labels {
            per.push(per_category(&root, &cats, l, wl)?);
        }
        let totals: Vec<f64> = per.iter().map(|(ms, _)| ms.iter().sum()).collect();
        let mut ax = Axes::new(
            cells[j],
            &format!("{wl}: stage time per forward (ms, sum of stages alone)"),
            &[],
            AxisCfg::categories(&labels, false)?.rotated(20.0),
            AxisCfg::linear(0.0, nice_top(max_of(&totals)? * 1.08)?)?,
        )?;
        let by_cat: Vec<Vec<f64>> = (0..cats.len())
            .map(|c| per.iter().map(|(ms, _)| ms[c]).collect())
            .collect();
        let mut bottom = vec![0.0; labels.len()];
        for (vals, col) in by_cat.iter().zip(&colors) {
            for (k, v) in vals.iter().enumerate() {
                ax.seg(k as f64, 0.7, (bottom[k], bottom[k] + v), &Paint::new(col));
                bottom[k] += v;
            }
        }
        for (k, t) in bottom.iter().enumerate() {
            ax.text(
                k as f64,
                *t,
                (0.0, -4.0),
                &format!("{t:.1}"),
                &Ts::new(10.0).mid().ink().va(Va::Bottom),
            );
        }
        fig.add(ax);

        let disp: Vec<f64> = per.iter().map(|(_, d)| *d).collect();
        let mut bx = Axes::new(
            cells[workloads.len() + j],
            &format!("{wl}: dispatches per forward"),
            &[],
            AxisCfg::categories(&labels, false)?.rotated(20.0),
            AxisCfg::linear(0.0, nice_top(max_of(&disp)? * 1.1)?)?,
        )?;
        for (k, d) in disp.iter().enumerate() {
            let c = if k % 2 == 0 { GREY } else { ORANGE };
            bx.bar(k as f64, 0.7, (0.0, *d), &Paint::new(c));
            bx.text(
                k as f64,
                *d,
                (0.0, -4.0),
                &format!("{d}"),
                &Ts::new(10.0).mid().ink().va(Va::Bottom),
            );
        }
        fig.add(bx);
    }
    Ok(Output::svg(fig.finish()))
}
