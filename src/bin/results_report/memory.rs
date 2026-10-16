//! `embedgemma2_memory_m5pro.{json,txt}` from the lane H raw runs.
//!
//! ```text
//! results_report memory RAW_DIR OUT_DIR
//! ```
//!
//! The raw directory holds what the lane H session produced:
//!
//! * `table_{before,after}_{1,2}.json`: `profile_embedgemma2`
//!   `PROFILE_MODE=memory-table`, two runs of each build (`before` is 16f5190,
//!   `after` the change);
//! * `hash_{before,after,after_2}.txt`: `PROFILE_MODE=hash` on the real
//!   checkpoint;
//! * `tiny_{before,after}.txt`: the `TINY` lines of
//!   `tests/embedgemma2_memory.rs tiny_embedding_hashes`;
//! * `paired_{f32,bf16}_a.json.lanes/`: `bench_paired --out`'s per-lane run
//!   files (`rNN-I-lane.stdout` and `.meta.json`), lanes `before`, `after`,
//!   `after_ple256`;
//! * `paired_f32.json.lanes/`: an earlier 6-round run with a fourth lane,
//!   `after_ple64`;
//! * `ple_*.json.lanes/`: the per-layer-input bound sweeps;
//! * `bounds.txt`: the bounds, written before the changed code was measured.

use crate::py::*;
use std::path::Path;

/// B3/B4: median of per-round ratios.
const MIB_BOUND: f64 = 1.03;

type Cell = ((String, String, String), Val);

/// `{(operands, workload, config): peak_above_model_mib}`, in first-seen order.
fn table(path: &Path) -> R<Vec<Cell>> {
    let doc = load_file(path)?;
    let mut out: Vec<Cell> = Vec::new();
    for r in doc.get("runs")?.arr()? {
        let x = r.get("result")?;
        let key = (
            r.get("operands")?.str()?.to_string(),
            x.get("workload")?.str()?.to_string(),
            r.get("label")?.str()?.to_string(),
        );
        let value = x.get("peak_above_model_mib")?.clone();
        match out.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = value,
            None => out.push((key, value)),
        }
    }
    Ok(out)
}

/// Python dict equality.
fn tables_equal(a: &[Cell], b: &[Cell]) -> bool {
    a.len() == b.len() && a.iter().all(|(k, v)| b.iter().any(|(k2, v2)| k == k2 && v.same(v2)))
}

type Series = OMap<OMap<Vec<Num>>>;

/// `{lane: {workload: [ms_min per round]}}`, the per-round normalized 1-minute
/// load, and the rounds.
fn lanes_dir(dir: &Path) -> R<(Series, Vec<Num>, Vec<i128>)> {
    let mut metas: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| format!("cannot read {}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".meta.json") && !n.starts_with('.'))
        .collect();
    metas.sort();
    // lane -> workload -> [(round, ms_min)], a later file overwriting an earlier round.
    let mut runs: OMap<OMap<Vec<(i128, Num)>>> = OMap::default();
    let mut loads = Vec::new();
    for name in &metas {
        let meta = load_file(&dir.join(name))?;
        let stdout_name = format!("{}.stdout", &name[..name.len() - ".meta.json".len()]);
        let stdout = read_text(&dir.join(&stdout_name))?;
        let stripped = strip(&stdout);
        let last = *splitlines(stripped)
            .last()
            .ok_or_else(|| format!("{stdout_name}: IndexError: no output lines"))?;
        let rows = parse_json(last, &stdout_name)?;
        let lane = meta.get("lane")?.str()?;
        let round = match meta.get("round")?.num_val()? {
            I(r) => r,
            F(_) => return Err("a lane round must be an integer".to_string()),
        };
        for r in rows.arr()? {
            let by_round = runs
                .entry(lane, OMap::default)
                .entry(r.get("workload")?.str()?, Vec::new);
            let ms = r.get("ms_min")?.num_val()?;
            match by_round.iter_mut().find(|(k, _)| *k == round) {
                Some(slot) => slot.1 = ms,
                None => by_round.push((round, ms)),
            }
        }
        loads.push(meta.get("load_at_start")?.get("normalized_loadavg_1m")?.num_val()?);
    }
    let mut rounds: Vec<i128> = runs
        .values()
        .flat_map(|lane| lane.values().flat_map(|w| w.iter().map(|(k, _)| *k)))
        .collect();
    rounds.sort_unstable();
    rounds.dedup();
    let mut series: Series = OMap::default();
    for (lane, workloads) in &runs.items {
        let mut by_workload = OMap::default();
        for (w, d) in &workloads.items {
            let values = rounds
                .iter()
                .map(|k| {
                    d.iter()
                        .find(|(r, _)| r == k)
                        .map(|(_, v)| *v)
                        .ok_or_else(|| format!("KeyError: round {k} for {lane}/{w}"))
                })
                .collect::<R<Vec<Num>>>()?;
            by_workload.set(w, values);
        }
        series.set(lane, by_workload);
    }
    Ok((series, loads, rounds))
}

struct Stat {
    min_of_n_ms: Num,
    base_min_of_n_ms: Num,
    min_of_n_ratio: f64,
    median: f64,
    lo: f64,
    hi: f64,
    spread: f64,
    within: usize,
    rounds: usize,
}

impl Stat {
    fn val(&self) -> Val {
        Val::obj([
            ("min_of_n_ms", Val::num(self.min_of_n_ms)),
            ("base_min_of_n_ms", Val::num(self.base_min_of_n_ms)),
            ("min_of_n_ratio", Val::float(self.min_of_n_ratio)),
            ("median_of_round_ratios", Val::float(self.median)),
            ("ratio_lo", Val::float(self.lo)),
            ("ratio_hi", Val::float(self.hi)),
            ("spread", Val::float(self.spread)),
            ("rounds_within_bound", Val::int(self.within)),
            ("rounds", Val::int(self.rounds)),
        ])
    }
}

fn ratio_stats(series: &Series, lane: &str, base: &str) -> R<Vec<(String, Stat)>> {
    let lane_series = series.get(lane).ok_or_else(|| format!("KeyError: {lane:?}"))?;
    let base_series = series.get(base).ok_or_else(|| format!("KeyError: {base:?}"))?;
    let mut out = Vec::new();
    for (w, xs) in &lane_series.items {
        let bs = base_series.get(w).ok_or_else(|| format!("KeyError: {w:?}"))?;
        let ratios = xs
            .iter()
            .zip(bs)
            .map(|(a, b)| a.div(*b).map(Num::f))
            .collect::<R<Vec<f64>>>()?;
        let as_nums: Vec<Num> = ratios.iter().map(|r| F(*r)).collect();
        let lo = min(&as_nums)?.f();
        let hi = max(&as_nums)?.f();
        out.push((
            w.clone(),
            Stat {
                min_of_n_ms: min(xs)?,
                base_min_of_n_ms: min(bs)?,
                min_of_n_ratio: min(xs)?.div(min(bs)?)?.f(),
                median: median(&as_nums)?.f(),
                lo,
                hi,
                spread: F(hi).div(F(lo))?.f(),
                within: ratios.iter().filter(|r| **r <= MIB_BOUND).count(),
                rounds: ratios.len(),
            },
        ));
    }
    Ok(out)
}

fn stats_val(stats: &[(String, Stat)]) -> Val {
    Val::Obj(stats.iter().map(|(w, s)| (w.clone(), s.val())).collect())
}

fn series_val(series: &Series) -> Val {
    Val::Obj(
        series
            .items
            .iter()
            .map(|(lane, ws)| {
                (
                    lane.clone(),
                    Val::Obj(
                        ws.items
                            .iter()
                            .map(|(w, xs)| (w.clone(), Val::Arr(xs.iter().map(|x| Val::num(*x)).collect())))
                            .collect(),
                    ),
                )
            })
            .collect(),
    )
}

struct LoadSpread {
    min: Num,
    median: Num,
    max: Num,
}

impl LoadSpread {
    fn of(loads: &[Num]) -> R<LoadSpread> {
        Ok(LoadSpread {
            min: min(loads)?,
            median: median(loads)?,
            max: max(loads)?,
        })
    }

    fn val(&self) -> Val {
        Val::obj([
            ("min", Val::num(self.min)),
            ("median", Val::num(self.median)),
            ("max", Val::num(self.max)),
        ])
    }
}

fn rounds_val(rounds: &[i128]) -> Val {
    Val::Arr(rounds.iter().map(|r| Val::num(I(*r))).collect())
}

fn read_hashes(path: &Path) -> R<Vec<String>> {
    let text = read_text(path)?;
    Ok(splitlines(&text)
        .into_iter()
        .filter(|ln| ln.starts_with("HASH") || ln.starts_with("TINY"))
        .map(|ln| strip(ln).to_string())
        .collect())
}

fn strings_val(v: &[String]) -> Val {
    Val::Arr(v.iter().map(|s| Val::Str(s.clone())).collect())
}

/// `{x:{w}.{prec}f}`.
fn fx(x: Num, w: usize, prec: usize) -> String {
    format!("{:>w$}", fixed(x.f(), prec))
}

fn fx64(x: f64, w: usize, prec: usize) -> String {
    format!("{:>w$}", fixed(x, prec))
}

fn py_bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

pub fn run(args: &[String]) -> R<()> {
    let [raw, out] = args else {
        return Err("usage: results_report memory RAW_DIR OUT_DIR".to_string());
    };
    let (raw, out) = (Path::new(raw), Path::new(out));

    let bounds = read_text(&raw.join("bounds.txt"))?;
    let tb = [
        table(&raw.join("table_before_1.json"))?,
        table(&raw.join("table_before_2.json"))?,
    ];
    let ta = [
        table(&raw.join("table_after_1.json"))?,
        table(&raw.join("table_after_2.json"))?,
    ];
    let before_stable = tables_equal(&tb[0], &tb[1]);
    let after_stable = tables_equal(&ta[0], &ta[1]);
    // Cells in the first before-table's order, each with its after value.
    struct MemCell {
        operands: String,
        workload: String,
        config: String,
        before: Val,
        after: Val,
    }
    let mut cells = Vec::new();
    for (k, v) in &tb[0] {
        let after = ta[0]
            .iter()
            .find(|(k2, _)| k2 == k)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| format!("KeyError: {k:?}"))?;
        cells.push(MemCell {
            operands: k.0.clone(),
            workload: k.1.clone(),
            config: k.2.clone(),
            before: v.clone(),
            after,
        });
    }
    let real_before = read_hashes(&raw.join("hash_before.txt"))?;
    let real_after = read_hashes(&raw.join("hash_after.txt"))?;
    let real_after_repeat = read_hashes(&raw.join("hash_after_2.txt"))?;
    let tiny_before = read_hashes(&raw.join("tiny_before.txt"))?;
    let tiny_after = read_hashes(&raw.join("tiny_after.txt"))?;
    let real_identical = real_before == real_after;
    let tiny_identical = tiny_before == tiny_after;

    struct Paired {
        rounds: Vec<i128>,
        series: Series,
        loads: LoadSpread,
        vs_before: Vec<(String, Vec<(String, Stat)>)>,
        ple256_vs_after: Vec<(String, Stat)>,
    }
    let mut paired: Vec<(&str, Paired)> = Vec::new();
    for (name, d) in [
        ("f32", "paired_f32_a.json.lanes"),
        ("bf16", "paired_bf16_a.json.lanes"),
        ("f32_six_rounds_four_lanes", "paired_f32.json.lanes"),
    ] {
        let (series, loads, rounds) = lanes_dir(&raw.join(d))?;
        let mut vs_before = Vec::new();
        for l in series.keys().filter(|l| *l != "before") {
            vs_before.push((l.clone(), ratio_stats(&series, l, "before")?));
        }
        let ple256_vs_after = ratio_stats(&series, "after_ple256", "after")?;
        paired.push((
            name,
            Paired {
                rounds,
                loads: LoadSpread::of(&loads)?,
                series,
                vs_before,
                ple256_vs_after,
            },
        ));
    }

    struct Sweep {
        rounds: Vec<i128>,
        series: Series,
        loads: LoadSpread,
        vs_ple256: Vec<(String, Vec<(String, Stat)>)>,
    }
    let mut sweeps: Vec<(&str, Sweep)> = Vec::new();
    for (name, d) in [
        ("f32 1024-4096 rows", "ple_f32_a.json.lanes"),
        (
            "bf16 1024-4096 rows, first run (machine at load 14-38 per core)",
            "ple_bf16_a.json.lanes",
        ),
        ("bf16 1024-4096 rows", "ple_bf16_b.json.lanes"),
        ("f32 8192 rows", "ple_f32_big.json.lanes"),
        (
            "f32 2x4096 repeat (the first run's median was 1.0303)",
            "ple_f32_big2.json.lanes",
        ),
        ("bf16 8192 rows", "ple_bf16_big.json.lanes"),
    ] {
        let (series, loads, rounds) = lanes_dir(&raw.join(d))?;
        let mut vs_ple256 = Vec::new();
        for l in ["ple16", "ple64"] {
            vs_ple256.push((l.to_string(), ratio_stats(&series, l, "ple256")?));
        }
        sweeps.push((
            name,
            Sweep {
                rounds,
                loads: LoadSpread::of(&loads)?,
                series,
                vs_ple256,
            },
        ));
    }

    // ---- json ----
    let stats_by_lane =
        |v: &[(String, Vec<(String, Stat)>)]| Val::Obj(v.iter().map(|(l, s)| (l.clone(), stats_val(s))).collect());
    let doc = Val::obj([
        ("bounds", Val::Str(bounds.clone())),
        (
            "memory",
            Val::obj([
                (
                    "cells",
                    Val::Arr(
                        cells
                            .iter()
                            .map(|c| {
                                Val::obj([
                                    ("operands", Val::Str(c.operands.clone())),
                                    ("workload", Val::Str(c.workload.clone())),
                                    ("config", Val::Str(c.config.clone())),
                                    ("before_mib", c.before.clone()),
                                    ("after_mib", c.after.clone()),
                                ])
                            })
                            .collect(),
                    ),
                ),
                ("before_stable", Val::Bool(before_stable)),
                ("after_stable", Val::Bool(after_stable)),
            ]),
        ),
        (
            "hashes",
            Val::obj([
                ("real_before", strings_val(&real_before)),
                ("real_after", strings_val(&real_after)),
                ("real_after_repeat", strings_val(&real_after_repeat)),
                ("tiny_before", strings_val(&tiny_before)),
                ("tiny_after", strings_val(&tiny_after)),
                ("real_identical", Val::Bool(real_identical)),
                ("tiny_identical", Val::Bool(tiny_identical)),
            ]),
        ),
        (
            "paired",
            Val::Obj(
                paired
                    .iter()
                    .map(|(name, p)| {
                        (
                            name.to_string(),
                            Val::obj([
                                ("rounds", rounds_val(&p.rounds)),
                                ("series_ms", series_val(&p.series)),
                                ("load_1m_normalized", p.loads.val()),
                                ("vs_before", stats_by_lane(&p.vs_before)),
                                ("ple256_vs_after", stats_val(&p.ple256_vs_after)),
                            ]),
                        )
                    })
                    .collect(),
            ),
        ),
        (
            "ple_sweeps",
            Val::Obj(
                sweeps
                    .iter()
                    .map(|(name, s)| {
                        (
                            name.to_string(),
                            Val::obj([
                                ("rounds", rounds_val(&s.rounds)),
                                ("series_ms", series_val(&s.series)),
                                ("load_1m_normalized", s.loads.val()),
                                ("vs_ple256", stats_by_lane(&s.vs_ple256)),
                            ]),
                        )
                    })
                    .collect(),
            ),
        ),
    ]);
    write_text(
        &out.join("embedgemma2_memory_m5pro.json"),
        &format!("{}\n", dumps(&doc, Some(1))),
    )?;

    // ---- text ----
    let mut l: Vec<String> = Vec::new();
    let mut w = |s: String| l.push(s);
    w("EMBEDGEMMA 2: DEVICE MEMORY OF encode, lane H (M5 Pro, real checkpoint)".into());
    w(String::new());
    w("Before = 16f5190 (the library), built in its own worktree and target dir. After = the lane H change:".into());
    w("  (a) encode hands its activations and uploaded inputs back to the runtime before its one wait;".into());
    w("  (b) PLE_RUN_BYTES 256 -> 16 MiB; (c) the q|k|v GEMM output and the MLP gate|up share one buffer.".into());
    w(
        "The harness is the same source for both builds (src/bin/profile_embedgemma2.rs mode memory-table / hash,"
            .into(),
    );
    w("src/bin/bench_embedgemma2.rs through bench_paired); only the library differs.".into());
    w(String::new());
    w(rstrip(&bounds).to_string());
    w(String::new());
    w("RESULTS: DEVICE PEAK ABOVE THE LOADED MODEL, MiB (one fresh process per cell, deterministic)".into());
    w("  single = one encode; back-to-back = the last of four encodes with no wait between;".into());
    w(
        "  sync-between = the last of three with rt.synchronize() between. Two runs of each build agree on every cell:"
            .into(),
    );
    w(format!(
        "  before stable: {}, after stable: {}.",
        py_bool(before_stable),
        py_bool(after_stable)
    ));
    w(String::new());
    w(format!(
        "  {:5} {:8} | {:>9} {:>10} {:>10} | {:>9} {:>10} {:>10} | back2back",
        "mode", "workload", "single", "back2back", "sync-betw", "single", "back2back", "sync-betw"
    ));
    w(format!(
        "  {:5} {:8} | {:>31} | {:>31} | change",
        "", "", "BEFORE", "AFTER"
    ));
    let cell = |ops: &str, wl: &str, cfg: &str| {
        cells
            .iter()
            .find(|c| c.operands == ops && c.workload == wl && c.config == cfg)
            .ok_or_else(|| format!("KeyError: {:?}", (ops, wl, cfg)))
    };
    for ops in ["f32", "bf16"] {
        for wl in ["1x16", "64x32", "32x256", "8x1024", "2x4096"] {
            let mut b = Vec::new();
            let mut a = Vec::new();
            for cfg in ["single", "back-to-back", "sync-between"] {
                let c = cell(ops, wl, cfg)?;
                b.push(c.before.num_val()?);
                a.push(c.after.num_val()?);
            }
            let change = a[1].div(b[1])?.sub(I(1))?.f();
            w(format!(
                "  {:5} {:8} | {} {} {} | {} {} {} | {}",
                ops,
                wl,
                fx(b[0], 9, 2),
                fx(b[1], 10, 2),
                fx(b[2], 10, 2),
                fx(a[0], 9, 2),
                fx(a[1], 10, 2),
                fx(a[2], 10, 2),
                signed_percent(change, 1)
            ));
        }
    }
    w(String::new());
    w("RESULTS: OUTPUT BITS (FNV-1a of the f32 bit patterns; three consecutive encodes agree in every cell)".into());
    w(format!(
        "  real checkpoint, 6 workloads x 2 modes x {{full, truncate_dim 128}}: before == after: {} (after run twice: {})",
        py_bool(real_identical),
        py_bool(real_after == real_after_repeat)
    ));
    w(format!(
        "  tiny model, 3 batches x 2 modes: before == after: {}",
        py_bool(tiny_identical)
    ));
    for ln in &real_after {
        w(format!("    {ln}"));
    }
    for ln in &tiny_after {
        w(format!("    {ln}"));
    }
    w(String::new());
    for (name, title) in [
        ("f32", "RESULTS: TIME, ExactF32"),
        ("bf16", "RESULTS: TIME, Bf16"),
        (
            "f32_six_rounds_four_lanes",
            "SUPPLEMENT: TIME, ExactF32, first 6-round run, with a 64 MiB lane",
        ),
    ] {
        let p = &paired
            .iter()
            .find(|(n, _)| *n == name)
            .ok_or_else(|| format!("KeyError: {name:?}"))?
            .1;
        w(format!(
            "{title}: bench_paired, {} rounds, order reversed on odd rounds, each value the min of 10 encodes",
            p.rounds.len()
        ));
        w(format!(
            "  machine load at lane start (1-minute load average / 18 cores): min {}, median {}, max {};",
            fixed(p.loads.min.f(), 2),
            fixed(p.loads.median.f(), 2),
            fixed(p.loads.max.f(), 2)
        ));
        w(
            "  other lanes and builds share the GPU and the cores, so a single round can be several times slower."
                .into(),
        );
        for (lane, stats) in &p.vs_before {
            w(format!("  lane {lane} against before:"));
            w(format!(
                "    {:14} {:>10} {:>9} | {:>9} {:>7} {:>16} {:>7} {:>13}",
                "workload", "before ms", "lane ms", "min-of-N", "median", "[lo, hi]", "spread", "rounds<=1.03"
            ));
            for (wl, s) in stats {
                w(format!(
                    "    {:14} {} {} | {} {} [{},{}] {} {:>9}/{}",
                    wl,
                    fx(s.base_min_of_n_ms, 10, 2),
                    fx(s.min_of_n_ms, 9, 2),
                    fx64(s.min_of_n_ratio, 9, 3),
                    fx64(s.median, 7, 3),
                    fx64(s.lo, 6, 3),
                    fx64(s.hi, 6, 3),
                    fx64(s.spread, 7, 2),
                    s.within,
                    s.rounds
                ));
            }
        }
        w(
            "  after_ple256 (the change with the old 256 MiB bound) against after (the PLE bound's own cost, B4):"
                .into(),
        );
        w(format!(
            "    {:14} | {:>9} {:>7} {:>16} {:>7}",
            "workload", "min-of-N", "median", "[lo, hi]", "spread"
        ));
        for (wl, s) in &p.ple256_vs_after {
            w(format!(
                "    {:14} | {} {} [{},{}] {}",
                wl,
                fx64(s.min_of_n_ratio, 9, 3),
                fx64(s.median, 7, 3),
                fx64(s.lo, 6, 3),
                fx64(s.hi, 6, 3),
                fx64(s.spread, 7, 2)
            ));
        }
        w(String::new());
    }
    w("RESULTS: THE PER-LAYER-INPUT BOUND ALONE (B4): one changed binary at 16, 64 and 256 MiB (BENCH_PLE_RUN_BYTES), interleaved".into());
    w("  Each ratio is lane / the 256 MiB lane, over the rounds; min-of-N is the minimum over all rounds of each lane's per-process minimum.".into());
    for (name, sweep) in &sweeps {
        w(format!(
            "  {name}: {} rounds, load/core median {}, max {}",
            sweep.rounds.len(),
            fixed(sweep.loads.median.f(), 1),
            fixed(sweep.loads.max.f(), 1)
        ));
        w(format!(
            "    {:10} {:6} | {:>10} {:>9} | {:>9} {:>7} {:>16} {:>7}",
            "workload", "lane", "256 MiB ms", "lane ms", "min-of-N", "median", "[lo, hi]", "spread"
        ));
        for (lane, stats) in &sweep.vs_ple256 {
            for (wl, s) in stats {
                w(format!(
                    "    {:10} {:6} | {} {} | {} {} [{},{}] {}",
                    wl,
                    lane,
                    fx(s.base_min_of_n_ms, 10, 2),
                    fx(s.min_of_n_ms, 9, 2),
                    fx64(s.min_of_n_ratio, 9, 3),
                    fx64(s.median, 7, 3),
                    fx64(s.lo, 6, 3),
                    fx64(s.hi, 6, 3),
                    fx64(s.spread, 7, 2)
                ));
            }
        }
    }
    w(String::new());
    for line in READING {
        w((*line).to_string());
    }
    w(String::new());
    let text = format!("{}\n", l.join("\n"));
    write_text(&out.join("embedgemma2_memory_m5pro.txt"), &text)?;
    println!("{}", l.join("\n"));
    Ok(())
}

/// The closing reading, as the script wrote it.
const READING: &[&str] = &[
    "READING (each claim labelled verified / inferred / unverified)",
    "  * Memory (verified, deterministic, repeated): B2 holds. One encode and four back to back peak at the same figure within 0.3 MiB",
    "    (the second call's own small output buffer); 400.5 / 400.3 / 400.3 MiB at 32x256 / 8x1024 / 2x4096 against the 400 + 1 bound; no cell",
    "    above its 16f5190 value. 736.5 -> 400.5 is the 240 (PLE) + 96 (alias) the lane D accounting predicted; the 96 was inferred there and is measured here.",
    "  * Output bits (verified): B1 holds on the real checkpoint (6 workloads, both modes, full and truncated, each from three consecutive",
    "    encodes) and on the tiny model (3 batches, both modes, one of them two forwards). The 16 and 64 MiB PLE bounds were also hashed on 16f5190",
    "    itself (the library unchanged) before any change was made: identical, so the PLE bound moves no bit by itself.",
    "  * PLE bound time (B4, verified to the extent the noise allows): one binary at 16 MiB against 256 MiB. The three small-row sweeps and the",
    "    8192-row sweeps put the ratio at 0.96 to 1.03 (median of rounds) and 0.90 to 1.00 (min-of-N); the only cell at 1.0303 (f32 2x4096, 8 rounds, load",
    "    21 per core, spread 1.75) was repeated with 12 rounds and read 1.005 / 0.991, so 16 MiB is the default. A first bf16 sweep at 1024-4096 rows ran at a",
    "    load of 14-38 per core and gave min-of-N 1.047 at 32x32 with medians 0.90-0.96 (spreads 2.2-4.6): it was repeated once (spreads 1.04-1.10, ratios 0.98-1.00).",
    "    The 64x32 f32 cell that read 1.045 and 1.074 in the two paired runs is not the PLE bound: the quiet sweep reads 0.993 / 0.984 there.",
    "  * Time against 16f5190 (B3): NOT met on the registered statistic, and the registered response (repeat) did not converge. Every series of",
    "    the paired runs has a max/min round spread of 1.2 to 28 (the machine ran at 100 to 360 load average on 18 cores, with other lanes on the GPU),",
    "    so the median over rounds of lane/before is not a stable statistic here: f32 medians 0.81-1.25 (1x16: 1.247 with a spread of 28), 2x4096 1.065 in the first run;",
    "    bf16 medians 0.94-1.00. The statistic that survives the contention is min-of-N over all rounds (10 rounds x 10 encodes per lane): after / before",
    "    = 0.92 / 1.05 / 0.90 / 0.98 / 0.96 / 0.86 (f32, 10 rounds) and 1.00 / 1.05 / 1.00 / 1.02 / 1.04 / 1.04 (f32, 6 rounds), 1.03 / 0.98 / 1.00 / 1.01 / 0.88 / 0.88 (bf16)",
    "    at 1x16 / 64x32 / 32x256 / 8x1024 / 2x4096 / ragged. So: no slowdown beyond +5.4 % shows in any series (verified as a bound of about 5 %, which is",
    "    the noise floor of this machine), and the 5.4 % cell (64x32) is not the PLE bound (above) and not the alias or the early release either: the changed",
    "    build with the old 256 MiB bound reads 0.98 and 1.01 there. Whether the change is within 3 % of 16f5190 at every workload is UNVERIFIED on this machine.",
    "  * Why none is expected (inferred): the change adds no dispatch to a forward except 22 more per-layer-input GEMM+scale+norm triples at 8192 rows",
    "    (66 extra dispatches of a few us each, against 372 in a forward and hundreds of ms), the alias changes which buffer a GEMM writes, not what it computes,",
    "    and the early release moves buffer ownership, not GPU work.",
];
