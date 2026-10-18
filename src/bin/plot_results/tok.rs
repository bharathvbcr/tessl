//! The tokenizer figure and the video figure.

use crate::chart::{
    nice_floor, spread, Axes, AxisCfg, Figure, LegendItem, Paint, Ts, Va, BLUE, M_BLUE, M_GREY, M_ORANGE, ORANGE,
};
use crate::data::{max_of, min_of, N, R};
use crate::paired::{load, log_span};
use crate::registry::Output;
use std::path::Path;

// ---------------------------------------------------------------------------
// tok
// ---------------------------------------------------------------------------

const TOK_LANES: [(&str, &str, &str); 3] = [
    ("tessl", "tessl (Rust, 1 thread)", M_BLUE),
    ("hf_loop", "tokenizers encode() loop (1 thread)", M_ORANGE),
    ("hf_batch", "tokenizers encode_batch() (all cores)", M_GREY),
];

/// One lane's own output for a workload: its token count, byte count and id hash.
fn lane_row<'a>(lane: &N<'a>, workload: &str) -> R<N<'a>> {
    lane.find("workload", workload)
}

/// `embedgemma2_tok_m5pro.json` with the two lanes' own output beside it
/// (`embedgemma2_tok_m5pro_tessl_lane.json`, `embedgemma2_tok_m5pro_hf_lane.json`):
/// left, encode throughput (tokens per second at the fastest of the interleaved
/// rounds, every round a dot); right, the load time of `tokenizer.json`. It
/// refuses to draw if the two lanes' id hashes or token counts differ: the
/// lanes must have produced the same ids before the times mean anything.
pub fn tok(json: &Path) -> R<Output> {
    let doc = load(json)?;
    let root = N::root(&doc, "tok");
    let stem = json
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("{}: not a file name", json.display()))?;
    let rust_doc = load(&json.with_file_name(format!("{stem}_tessl_lane.json")))?;
    let hf_doc = load(&json.with_file_name(format!("{stem}_hf_lane.json")))?;
    let rust = N::root(&rust_doc, "tessl lane");
    let hf = N::root(&hf_doc, "hf lane");
    let summary = root.get("summary")?.nonempty()?;
    let mut workloads = Vec::new();
    let mut has_load = false;
    for row in &summary {
        let w = row.get("workload")?.s()?.to_string();
        if w == "load" {
            has_load = true;
        } else {
            workloads.push(w);
        }
    }
    if workloads.is_empty() || !has_load {
        return Err(format!(
            "{}: needs a load row and at least one encode workload",
            root.get("summary")?.path()
        ));
    }
    let find_row = |w: &str| -> R<N> { root.get("summary")?.find("workload", w) };

    let mut tokens = Vec::new();
    for w in &workloads {
        let (r, h) = (lane_row(&rust, w)?, lane_row(&hf, w)?);
        let (rh, hh) = (r.get("ids_fnv32")?.f()?, h.get("ids_fnv32")?.f()?);
        let (rt, ht) = (r.get("tokens")?.f()?, h.get("tokens")?.f()?);
        if rh != hh || rt != ht {
            return Err(format!(
                "{w}: the lanes produced different ids ({rh} vs {hh}, {rt} vs {ht} tokens)"
            ));
        }
        tokens.push(rt);
    }
    let load_row = lane_row(&rust, "load")?;
    let (load_tokens, load_bytes) = (load_row.get("tokens")?.f()?, load_row.get("bytes")?.f()?);

    let legend: Vec<LegendItem> = TOK_LANES
        .iter()
        .map(|(_, label, c)| LegendItem::boxed(label, c))
        .collect();
    let mut fig = Figure::new(
        1300.0,
        520.0,
        &format!(
            "embedgemma2_tok vs Hugging Face tokenizers 0.23.2, {} interleaved rounds, loaded M5 Pro",
            match root.opt("rounds").map(|r| r.f()) {
                Some(Ok(v)) => format!("{v}"),
                _ => "?".to_string(),
            }
        ),
        &[],
        &legend,
    );
    let cells = fig.cells(&[2.4, 1.0], &[1.0]);

    // Throughput, log.
    let mut rates: Vec<Vec<Vec<f64>>> = Vec::new();
    for (lane, _, _) in TOK_LANES {
        let mut per_w = Vec::new();
        for (i, w) in workloads.iter().enumerate() {
            let rounds = find_row(w)?.get(lane)?.get("ms_min_all")?.positives()?;
            per_w.push(rounds.iter().map(|ms| tokens[i] / (ms / 1e3)).collect());
        }
        rates.push(per_w);
    }
    let flat: Vec<f64> = rates.iter().flatten().flatten().copied().collect();
    let (lo, hi) = log_span(min_of(&flat)?, max_of(&flat)? * 1.5)?;
    let labels: Vec<String> = workloads
        .iter()
        .zip(&tokens)
        .map(|(w, t)| format!("{w}\n{t} tokens"))
        .collect();
    let mut ax = Axes::new(
        cells[0],
        "Encode throughput, interleaved rounds (identical ids on every workload)",
        &[],
        AxisCfg::categories(&labels, false)?,
        AxisCfg::log(lo, hi)?.labelled("tokens per second (log; bar: best round, dots: every round)"),
    )?;
    let width = 0.26;
    for (j, (_, _, color)) in TOK_LANES.iter().enumerate() {
        for (i, per_round) in rates[j].iter().enumerate() {
            let x = i as f64 + spread(j, TOK_LANES.len(), width);
            let best = max_of(per_round)?;
            ax.bar(x, width * 0.9, (0.0, best), &Paint::new(color).alpha(0.85));
            for r in per_round {
                ax.dot(x, *r, 2.6, &Paint::ink().alpha(0.45));
            }
            ax.text(
                x,
                best,
                (0.0, -5.0),
                &format!("{:.2}M", best / 1e6),
                &Ts::new(9.5).mid().ink().va(Va::Bottom),
            );
        }
    }
    fig.add(ax);

    // Load time, linear.
    let mut loads = Vec::new();
    for (lane, _, _) in TOK_LANES {
        loads.push(find_row("load")?.get(lane)?.get("ms_min_all")?.positives()?);
    }
    let top = max_of(&loads.iter().flatten().copied().collect::<Vec<_>>())?;
    let names: Vec<String> = ["tessl", "tokenizers\n(loop lane)", "tokenizers\n(batch lane)"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut bx = Axes::new(
        cells[1],
        &format!("Load tokenizer.json ({:.0} MB, {load_tokens} tokens)", load_bytes / 1e6),
        &[],
        AxisCfg::categories(&names, false)?,
        AxisCfg::linear(0.0, top * 1.12)?.labelled("ms (best round; dots: every round)"),
    )?;
    for (j, ((_, _, color), rounds)) in TOK_LANES.iter().zip(&loads).enumerate() {
        let best = min_of(rounds)?;
        bx.bar(j as f64, 0.7, (0.0, best), &Paint::new(color).alpha(0.85));
        for r in rounds {
            bx.dot(j as f64, *r, 2.6, &Paint::ink().alpha(0.45));
        }
        bx.text(
            j as f64,
            best,
            (0.0, -9.0),
            &format!("{best:.0} ms"),
            &Ts::new(10.0).mid().ink().va(Va::Bottom),
        );
    }
    fig.add(bx);
    let _ = nice_floor;
    Ok(Output::svg(fig.finish()))
}

// ---------------------------------------------------------------------------
// video
// ---------------------------------------------------------------------------

/// The runs of one (workload, precision): each run's min-of-N ms and the load average.
type Series = ((String, String), Vec<(f64, f64)>);

const VIDEO_PANELS: [(&str, &str); 3] = [
    ("preprocess", "preprocess_video, CPU\n(32 frames, 640x360)"),
    ("tower", "vision tower\n(32 frames)"),
    ("encode", "whole call\n(tower + text encoder, one wait)"),
];

/// `embedgemma2_video_m5pro.json`: per workload and precision each run's
/// min-of-N as a dot and the best run as a bar, with the machine's load average
/// at the start of each run's workload beside it: the spread between runs is the
/// point, the bar is the least-disturbed run, and none of it is a clean
/// measurement.
pub fn video(json: &Path) -> R<Output> {
    let doc = load(json)?;
    let root = N::root(&doc, "video");
    let mut by: Vec<Series> = Vec::new();
    for run in root.get("runs")?.nonempty()? {
        for r in run.get("rows")?.nonempty()? {
            let key = (
                r.get("workload")?.s()?.to_string(),
                r.get("precision")?.s()?.to_string(),
            );
            let pt = (r.get("ms_min")?.f()?, r.get("load_avg_1m")?.f()?);
            match by.iter_mut().find(|(k, _)| *k == key) {
                Some((_, v)) => v.push(pt),
                None => by.push((key, vec![pt])),
            }
        }
    }
    let mut fig = Figure::new(
        1150.0,
        460.0,
        "One 32-frame video, EmbeddingGemma 2 on an Apple M5 Pro",
        &["Dots: the runs, labelled by the 1-minute load average when the workload started; bar: the best run."],
        &[],
    );
    let cells = fig.cells(&[1.0, 1.0, 1.0], &[1.0]);
    for (i, (w, title)) in VIDEO_PANELS.iter().enumerate() {
        let precisions: &[&str] = if *w == "preprocess" { &["f32"] } else { &["f32", "bf16"] };
        let mut pts = Vec::new();
        for p in precisions {
            let v = by
                .iter()
                .find(|((kw, kp), _)| kw == w && kp == p)
                .map(|(_, v)| v.clone())
                .ok_or_else(|| format!("{}.runs: no {w} row for {p}", root.path()))?;
            pts.push(v);
        }
        let top = max_of(&pts.iter().flatten().map(|(ms, _)| *ms).collect::<Vec<_>>())?;
        let cats: Vec<String> = if *w == "preprocess" {
            vec!["CPU".to_string()]
        } else {
            precisions.iter().map(|s| s.to_string()).collect()
        };
        let mut ax = Axes::new(
            cells[i],
            title,
            &[],
            AxisCfg::categories(&cats, false)?,
            if i == 0 {
                AxisCfg::linear(0.0, top * 1.18)?.labelled("ms, min of 15 calls")
            } else {
                AxisCfg::linear(0.0, top * 1.18)?
            },
        )?;
        for (x, (p, v)) in precisions.iter().zip(&pts).enumerate() {
            let color = if *p == "f32" { BLUE } else { ORANGE };
            let best = min_of(&v.iter().map(|(ms, _)| *ms).collect::<Vec<_>>())?;
            ax.bar(x as f64, 0.55, (0.0, best), &Paint::new(color).alpha(0.35));
            for (k, (ms, load)) in v.iter().enumerate() {
                let dx = spread(k, v.len(), 0.12);
                ax.dot(x as f64 + dx, *ms, 4.5, &Paint::new(color));
                ax.text(
                    x as f64 + dx,
                    *ms,
                    (0.0, -9.0),
                    &format!("{load:.0}"),
                    &Ts::new(10.0).mid().va(Va::Bottom),
                );
            }
            ax.text(
                x as f64,
                0.0,
                (0.0, -6.0),
                &format!("{best:.0} ms"),
                &Ts::new(11.0).mid().ink().va(Va::Bottom),
            );
        }
        fig.add(ax);
    }
    Ok(Output::svg(fig.finish()))
}
