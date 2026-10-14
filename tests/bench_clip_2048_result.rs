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
