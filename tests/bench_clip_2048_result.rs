//! GPU-free consistency check of bench/results/qwen35_train_clip_2048_m5pro.txt:
//! the committed summary table must equal the minima of the raw runs below it,
//! every timed run exited 0, and the headline wait counts hold in every run.

use std::collections::BTreeMap;

const FILE: &str = include_str!("../bench/results/qwen35_train_clip_2048_m5pro.txt");

#[derive(Default, Debug)]
struct Run {
    label: String,
    total_ms: f64,
    gsn_ms: f64,
    gsn_waits: u32,
    total_waits: u32,
    footprint: u64,
    exit: Option<i32>,
}

fn num_after(line: &str, key: &str) -> f64 {
    let rest = line.trim_start().strip_prefix(key).expect("key");
    rest.split_whitespace().next().unwrap().parse().unwrap()
}

fn runs() -> Vec<Run> {
    let mut out: Vec<Run> = Vec::new();
    for line in FILE.lines() {
        if let Some(h) = line.strip_prefix("# ---- clip round ") {
            let parts: Vec<&str> = h.split(',').collect();
            let mode = if parts[2].contains("async") { "async" } else { "sync" };
            out.push(Run {
                label: format!("{}/{}/{}", parts[0].trim(), parts[1].trim(), mode),
                ..Default::default()
            });
        } else if let Some(r) = out.last_mut() {
            let t = line.trim_start();
            if t.starts_with("grad_sq_norm") && t.contains("waits") {
                r.gsn_ms = num_after(t, "grad_sq_norm");
                r.gsn_waits = t
                    .split(" ms, ")
                    .nth(1)
                    .unwrap()
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
            } else if t.starts_with("total ") && t.contains("waits") {
                r.total_ms = num_after(t, "total");
                r.total_waits = t
                    .split(" ms, ")
                    .nth(1)
                    .unwrap()
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
            } else if t.ends_with("peak memory footprint") {
                r.footprint = t.split_whitespace().next().unwrap().parse().unwrap();
            } else if let Some(c) = line.strip_prefix("# exit: ") {
                r.exit = Some(c.trim().parse().unwrap());
            }
        }
    }
    out
}

#[test]
fn every_run_parsed_and_exited_zero() {
    let rs = runs();
    assert_eq!(rs.len(), 14, "1 probe pair + 3 rounds x 4");
    for r in &rs {
        assert_eq!(r.exit, Some(0), "{}", r.label);
        assert!(r.total_ms.is_finite() && r.total_ms > 0.0, "{}", r.label);
        assert!(
            r.footprint > 0 && r.footprint < 32 << 30,
            "{} under memguard cap",
            r.label
        );
    }
}

#[test]
fn summary_rows_are_minima_of_raw_runs() {
    let rs: Vec<Run> = runs().into_iter().filter(|r| !r.label.starts_with("P/")).collect();
    let mut min: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for r in &rs {
        let key = r.label.split_once('/').unwrap().1.to_string();
        let e = min.entry(key).or_insert((f64::MAX, f64::MAX));
        e.0 = e.0.min(r.total_ms);
        e.1 = e.1.min(r.gsn_ms);
    }
    for (key, (tot, gsn)) in &min {
        let (variant, mode) = key.split_once('/').unwrap();
        let row = FILE
            .lines()
            .find(|l| l.starts_with(&format!("# {variant}/{mode}")) && l.contains(" min "))
            .unwrap_or_else(|| panic!("no summary row for {key}"));
        let f: Vec<&str> = row.split_whitespace().collect();
        let (s_tot, s_gsn): (f64, f64) = (f[3].parse().unwrap(), f[6].parse().unwrap());
        assert!((s_tot - tot).abs() < 0.06, "{key} total {s_tot} vs raw {tot}");
        assert!((s_gsn - gsn).abs() < 0.06, "{key} grad_sq {s_gsn} vs raw {gsn}");
    }
}

#[test]
fn grad_sq_norm_waits_hold_in_every_run() {
    for r in runs() {
        let after = r.label.contains("after");
        let sync = r.label.ends_with("/sync");
        match (after, sync) {
            (true, true) => assert_eq!((r.gsn_waits, r.total_waits), (1, 2104), "{}", r.label),
            (false, true) => assert_eq!((r.gsn_waits, r.total_waits), (320, 2392), "{}", r.label),
            (_, false) => assert_eq!((r.gsn_waits, r.total_waits), (1, 16), "{}", r.label),
        }
    }
}

#[test]
fn prose_matches_table_and_states_columns_are_independent_minima() {
    assert!(
        FILE.contains("INDEPENDENT minimum"),
        "summary must say its columns are per-column minima from different runs"
    );
    assert!(
        FILE.contains("grad_sq_norm 589.2 -> 20.7 ms"),
        "findings prose must cite the table's grad_sq minima, not rounded figures"
    );
    assert!(
        FILE.contains("neither shown to be\n#   real nor shown to be noise"),
        "train_step_into slowdown must be reported as unresolved"
    );
}

// ---- adversarial additions: the remaining summary columns and the prose figures ----

fn summary_row(variant: &str, mode: &str) -> Vec<f64> {
    let row = FILE
        .lines()
        .find(|l| l.starts_with(&format!("# {variant}/{mode}")) && l.contains(" min "))
        .unwrap_or_else(|| panic!("no summary row {variant}/{mode}"));
    // total, max, train, grad_sq, adamw, fp GB, waits
    row.split_whitespace().skip(3).map(|x| x.parse().unwrap()).collect()
}

fn timed(variant: &str, mode: &str) -> Vec<(Run, f64, f64)> {
    // (run, train_step_into ms, adamw ms) for non-probe runs
    let mut res = Vec::new();
    let mut cur: Option<usize> = None;
    let all = runs();
    let mut idx = 0usize;
    let (mut train, mut adamw) = (0.0, 0.0);
    let mut pending: Vec<(usize, f64, f64)> = Vec::new();
    for line in FILE.lines() {
        if line.starts_with("# ---- clip round ") {
            if let Some(i) = cur {
                pending.push((i, train, adamw));
            }
            cur = Some(idx);
            idx += 1;
        } else {
            let t = line.trim_start();
            if t.starts_with("train_step_into") && t.contains("waits") {
                train = num_after(t, "train_step_into");
            } else if t.starts_with("adamw_step") && t.contains("waits") {
                adamw = num_after(t, "adamw_step");
            }
        }
    }
    if let Some(i) = cur {
        pending.push((i, train, adamw));
    }
    for (i, t, a) in pending {
        let r = &all[i];
        if r.label.starts_with("P/") {
            continue;
        }
        let key = r.label.split_once('/').unwrap().1;
        if key == format!("{variant}/{mode}") {
            res.push((
                Run {
                    label: r.label.clone(),
                    total_ms: r.total_ms,
                    gsn_ms: r.gsn_ms,
                    gsn_waits: r.gsn_waits,
                    total_waits: r.total_waits,
                    footprint: r.footprint,
                    exit: r.exit,
                },
                t,
                a,
            ));
        }
    }
    res
}

#[test]
fn every_summary_column_is_the_minimum_of_its_raw_runs() {
    for variant in ["before", "after"] {
        for mode in ["sync", "async"] {
            let rows = timed(variant, mode);
            assert_eq!(rows.len(), 3, "{variant}/{mode}: 3 timed rounds");
            let s = summary_row(variant, mode);
            let fmin = |f: &dyn Fn(&(Run, f64, f64)) -> f64| rows.iter().map(f).fold(f64::MAX, f64::min);
            let fmax = |f: &dyn Fn(&(Run, f64, f64)) -> f64| rows.iter().map(f).fold(f64::MIN, f64::max);
            let close = |a: f64, b: f64, what: &str| {
                assert!((a - b).abs() < 0.06, "{variant}/{mode} {what}: summary {a} vs raw {b}")
            };
            close(s[0], fmin(&|r| r.0.total_ms), "total");
            close(s[1], fmax(&|r| r.0.total_ms), "max total");
            close(s[2], fmin(&|r| r.1), "train_step_into");
            close(s[3], fmin(&|r| r.0.gsn_ms), "grad_sq_norm");
            close(s[4], fmin(&|r| r.2), "adamw_step");
            let fp = rows.iter().map(|r| r.0.footprint).min().unwrap() as f64 / 1e9;
            close(s[5], fp, "footprint GB");
            let waits = rows[0].0.total_waits as f64;
            assert!(rows.iter().all(|r| r.0.total_waits as f64 == waits), "waits stable");
            close(s[6], waits, "waits");
        }
    }
}

#[test]
fn prose_percentages_and_ranges_match_the_table() {
    let (bs, as_) = (summary_row("before", "sync"), summary_row("after", "sync"));
    let (ba, aa) = (summary_row("before", "async"), summary_row("after", "async"));
    let pct = |b: f64, a: f64| (b - a) / b * 100.0;
    assert!((pct(bs[0], as_[0]) - 8.7).abs() < 0.06, "sync -8.7 %");
    assert!((pct(ba[0], aa[0]) - 15.5).abs() < 0.06, "async -15.5 %");
    assert!(FILE.contains("-8.7 %") && FILE.contains("-15.5 %"));
    // overlap claim: before train range must overlap after train range
    let b = timed("before", "sync");
    let a = timed("after", "sync");
    let rng = |v: &Vec<(Run, f64, f64)>| {
        (
            v.iter().map(|r| r.1).fold(f64::MAX, f64::min),
            v.iter().map(|r| r.1).fold(f64::MIN, f64::max),
        )
    };
    let (bl, bh) = rng(&b);
    let (al, ah) = rng(&a);
    assert!(al <= bh && bl <= ah, "ranges overlap, as the prose says");
    assert!(FILE.contains("2461-3118") && FILE.contains("2684-2968"));
    assert!((bl - 2461.0).abs() < 0.6 && (bh - 3118.0).abs() < 0.6);
    assert!((al - 2684.0).abs() < 0.6 && (ah - 2968.0).abs() < 0.6);
}

#[test]
fn independent_minimum_example_is_real_and_probe_is_excluded() {
    // The example in the summary: the run with total 2974.6 has grad_sq 21.5 / train 2683.7,
    // and the 20.7 grad_sq minimum belongs to a run with total 3018.6.
    let a = timed("after", "sync");
    let r1 = a
        .iter()
        .find(|r| (r.0.total_ms - 2974.6).abs() < 0.06)
        .expect("total run");
    assert!((r1.0.gsn_ms - 21.5).abs() < 0.06 && (r1.1 - 2683.7).abs() < 0.06);
    let r2 = a.iter().find(|r| (r.0.gsn_ms - 20.7).abs() < 0.06).expect("gsn run");
    assert!((r2.0.total_ms - 3018.6).abs() < 0.06);
    // the probe's 20.3 ms must be lower than every timed grad_sq, i.e. excluding it matters
    let probe = runs()
        .into_iter()
        .find(|r| r.label == "P/probe-after/sync")
        .map(|r| r.gsn_ms);
    let probe = probe.expect("probe-after run");
    assert!((probe - 20.3).abs() < 0.06 && probe < 20.7);
}

#[test]
fn every_run_has_ac_power_and_no_peers() {
    assert_eq!(FILE.matches("Now drawing from 'AC Power'").count(), 14);
    assert_eq!(FILE.lines().filter(|l| l.starts_with("# peers: ")).count(), 14);
    assert!(FILE
        .lines()
        .filter(|l| l.starts_with("# peers: "))
        .all(|l| l == "# peers: 0"));
}

#[test]
fn sync_gap_prose_arithmetic_is_derived_from_the_table() {
    let b = summary_row("before", "sync");
    let a = summary_row("after", "sync");
    // summary_row columns: total, max, train, grad_sq, adamw, footprint
    let gap = b[0] - a[0];
    let gsn = b[3] - a[3];
    let train = a[2] - b[2];
    let adamw = a[4] - b[4];
    let net = gsn - train - adamw;
    for (v, s) in [
        (gap, "282.6 ms"),
        (gsn, "568.5 ms"),
        (train, "+222.8 ms"),
        (adamw, "+21.6 ms"),
        (net, "net 324.1 ms"),
    ] {
        assert!(FILE.contains(s), "prose must state {s} (computed {v:.1})");
        let stated: f64 = s
            .trim_start_matches("net ")
            .trim_start_matches('+')
            .trim_end_matches(" ms")
            .parse()
            .unwrap();
        assert!((stated - v).abs() < 0.06, "{s} disagrees with table: {v:.1}");
    }
    assert!(
        !FILE.contains("gap, 282 ms sync, is"),
        "the old claim that the gap equals the removed phase minus the train difference must stay gone"
    );
}

#[test]
fn every_parenthesised_subtraction_in_the_prose_is_arithmetically_true_and_in_the_table() {
    // Any "(A - B)" with two decimal numbers in the result file must satisfy the
    // stated figure's arithmetic and cite numbers that exist in the file.
    let mut checked = 0;
    for line in FILE.lines() {
        let mut rest = line;
        while let Some(i) = rest.find('(') {
            let inner = &rest[i + 1..];
            let Some(j) = inner.find(')') else { break };
            let parts: Vec<&str> = inner[..j].split(" - ").collect();
            if parts.len() == 2 {
                if let (Ok(a), Ok(b)) = (parts[0].trim().parse::<f64>(), parts[1].trim().parse::<f64>()) {
                    let d = a - b;
                    assert!(d > 0.0, "subtraction ({a} - {b}) must be positive in {line}");
                    assert!(
                        FILE.contains(&format!("{d:.1} ms")),
                        "difference {d:.1} of ({a} - {b}) not stated: {line}"
                    );
                    for n in [parts[0].trim(), parts[1].trim()] {
                        assert!(
                            FILE.matches(n).count() >= 2,
                            "{n} is cited only inside its own subtraction"
                        );
                    }
                    checked += 1;
                }
            }
            rest = &inner[j..];
        }
    }
    assert!(checked >= 3, "expected at least 3 subtractions, found {checked}");
}

#[test]
fn cross_file_wave2_claim_matches_the_wave2_file() {
    let wave2 = include_str!("../bench/results/qwen35_train_step_wave2_m5pro.txt");
    assert!(FILE.contains("2.583 -> 2.563 s sync"));
    assert!(wave2.contains("after/2048:-:sync:--storage=bf16        2.563"));
    assert!(wave2.contains("before/2048:-:sync:--storage=bf16       2.583"));
}
