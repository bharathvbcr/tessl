//! The numeric-parity scorer and ladder (`src/bin/common/parity.rs`,
//! `src/bin/gemm_parity.rs`), with no GPU.
//!
//! These are the cases of the retired `bench/test_parity_harness.py` that
//! tested the scorer, the external-lane contract, the verdicts and the grid
//! merge, plus the checks that the Rust port is the same function as the Python
//! it replaced: `oracle_dump` was scored by that Python (`parity()` at b219aff,
//! numpy 2.5.3) and its rows are kept as the golden answer, and the numpy
//! goldens below are numpy's own output for the same inputs.
//!
//! The bar every case defends is the same one: a check that could not run must
//! never report the same result as a check that ran and passed. Each case below
//! corresponds to a way an earlier harness returned a clean-looking report over
//! a dump it had not actually verified.

#[path = "../src/json.rs"]
#[allow(dead_code)]
mod json;
#[path = "../src/bin/common/parity.rs"]
mod parity;
#[path = "../src/bin/common/pyjson.rs"]
mod pyjson;

use parity::{adjudicate, CellRow, LaneRow, Precision, Scored};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

const M: usize = 32;
const N: usize = 24;
const K: usize = 16;
const LANES: [&str; 4] = ["tensorops-f32", "simdgroup-f32", "tensorops-bf16", "tensorops-tf32"];

// ------------------------------------------------------------------ fixtures

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("tessl-gemm-parity-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Deterministic uniform [-1, 1) stream.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 32) as u32) as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32
    }

    fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn f64s(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| v as f64).collect()
}

fn product(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    parity::reference_gemm(&f64s(a), &f64s(b), m, k, n).0
}

fn save(path: &Path, shape: &[usize], data: &[f32]) {
    tessl::npy::write_npy_f32(path, shape, data).unwrap();
}

fn load(path: &Path) -> Vec<f32> {
    tessl::npy::read_npy(path).unwrap().data_f32.unwrap()
}

fn seed_name(i: usize) -> String {
    format!("seed_{i:02}")
}

/// A manifest as raw text, so a test can break any one field.
fn manifest_text(lanes: &str, seeds: &str, k: &str) -> String {
    format!(r#"{{"shape":"synthetic","dist":"uniform","m":{M},"n":{N},"k":{k},"lanes":{lanes},"seeds":{seeds}}}"#)
}

fn lanes_json(lanes: &[&str]) -> String {
    format!(
        "[{}]",
        lanes.iter().map(|l| format!("\"{l}\"")).collect::<Vec<_>>().join(",")
    )
}

fn seeds_json(n: usize) -> String {
    format!(
        "[{}]",
        (0..n)
            .map(|i| format!("\"{}\"", seed_name(i)))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// A structurally valid dump. `err_scale[i]` perturbs seed i's results so a test
/// can pin down which seed the aggregate reports.
fn build_dump(root: &Path, seeds: usize, lanes: &[&str], err_scale: Option<&[f64]>) -> PathBuf {
    std::fs::create_dir_all(root).unwrap();
    for i in 0..seeds {
        let sd = root.join(seed_name(i));
        std::fs::create_dir_all(&sd).unwrap();
        let mut rng = Rng(1000 + i as u64);
        let (a, b) = (rng.fill(M * K), rng.fill(K * N));
        save(&sd.join("parity_a.npy"), &[M, K], &a);
        save(&sd.join("parity_b.npy"), &[K, N], &b);
        let mut c = product(&a, &b, M, K, N);
        if let Some(e) = err_scale {
            // Relative to the reference, so the perturbation scales with each
            // element's own error budget instead of blowing it on small ones.
            for v in &mut c {
                *v *= 1.0 + e[i];
            }
        }
        let c32: Vec<f32> = c.iter().map(|&v| v as f32).collect();
        for lane in lanes {
            save(&sd.join(format!("parity_c_{lane}.npy")), &[M, N], &c32);
        }
    }
    std::fs::write(
        root.join("parity_manifest.json"),
        manifest_text(&lanes_json(lanes), &seeds_json(seeds), &K.to_string()),
    )
    .unwrap();
    root.to_path_buf()
}

fn score(dir: &Path) -> Result<Vec<LaneRow>, String> {
    parity::parity(dir, &Precision::standard())
}

fn scored<'a>(rows: &'a [LaneRow], lane: &str) -> &'a Scored {
    match rows.iter().find(|r| r.lane() == lane) {
        Some(LaneRow::Scored(s)) => s,
        other => panic!("{lane} is not a scored row: {other:?}"),
    }
}

/// The scorer must refuse, and say why: `needle` is a case-insensitive part of
/// the message, as the Python harness checked.
fn refuses(label: &str, got: Result<Vec<LaneRow>, String>, needle: &str) {
    match got {
        Ok(rows) => panic!("{label}: ACCEPTED a dump it should have refused: {rows:?}"),
        Err(msg) => assert!(
            msg.to_lowercase().contains(&needle.to_lowercase()),
            "{label}: rejected, but the message lacks {needle:?}: {msg}"
        ),
    }
}

fn accepts(label: &str, got: Result<Vec<LaneRow>, String>) -> Vec<LaneRow> {
    got.unwrap_or_else(|e| panic!("{label}: rejected a valid dump -- {e}"))
}

// ---------------------------------------------------- stand-in runtimes

/// bf16 (nearest-even) of an f32, as an f64.
fn bf16(x: f32) -> f64 {
    let u = x.to_bits() as u64;
    let u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(u as u32) as f64
}

/// tf32 (10 explicit mantissa bits, nearest-even) of an f32, as an f64.
fn tf32(x: f32) -> f64 {
    let u = x.to_bits() as u64;
    let u = (u + 0xFFF + ((u >> 13) & 1)) & 0xFFFF_E000;
    f32::from_bits(u as u32) as f64
}

type Output = Result<(Vec<f32>, &'static str), String>;

/// The float64 product rounded once to f32: a comparison runtime that is exactly
/// right. The scorer-contract fixtures must not depend on which of MLX / torch is
/// installed or on how the local GPU rounds an f32 matmul.
fn exact(a: &[f32], b: &[f32]) -> Output {
    Ok((product(a, b, M, K, N).iter().map(|&v| v as f32).collect(), "f32"))
}

/// bf16 operands, exact product, bf16 result.
fn bf16_runtime(a: &[f32], b: &[f32]) -> Output {
    let (a, b): (Vec<f64>, Vec<f64>) = (
        a.iter().map(|&v| bf16(v)).collect(),
        b.iter().map(|&v| bf16(v)).collect(),
    );
    let c = parity::reference_gemm(&a, &b, M, K, N).0;
    Ok((c.iter().map(|&v| bf16(v as f32) as f32).collect(), "bf16"))
}

/// MLX's measured GPU f32 matmul: tf32 operands, exact product.
fn tf32_runtime(a: &[f32], b: &[f32]) -> Output {
    let (a, b): (Vec<f64>, Vec<f64>) = (
        a.iter().map(|&v| tf32(v)).collect(),
        b.iter().map(|&v| tf32(v)).collect(),
    );
    let c = parity::reference_gemm(&a, &b, M, K, N).0;
    Ok((c.iter().map(|&v| v as f32).collect(), "f32"))
}

/// A comparison runtime: seed operands in, product (and the dtype it came back in) out.
type Runtime = Box<dyn FnMut(&[f32], &[f32]) -> Output>;

struct Ext {
    lane: &'static str,
    dtype: &'static str,
    /// Called once per seed, in order, with that seed's operands.
    run: Runtime,
}

fn ext(lane: &'static str, dtype: &'static str, f: fn(&[f32], &[f32]) -> Output) -> Ext {
    Ext {
        lane,
        dtype,
        run: Box::new(f),
    }
}

fn boom(_: &[f32], _: &[f32]) -> Output {
    Err("stub boom".to_string())
}

/// What `bench/gemm_sweep_mlx.py --external-lanes DIR` writes, for stand-in runtimes.
fn write_externals(root: &Path, seeds: usize, mut exts: Vec<Ext>) {
    let mut entries = Vec::new();
    let mut per_lane: Vec<Vec<String>> = vec![Vec::new(); exts.len()];
    for i in 0..seeds {
        let sd = root.join(seed_name(i));
        let (a, b) = (load(&sd.join("parity_a.npy")), load(&sd.join("parity_b.npy")));
        for (j, e) in exts.iter_mut().enumerate() {
            match (e.run)(&a, &b) {
                Ok((c, out)) => {
                    save(&sd.join(format!("parity_c_{}.npy", e.lane)), &[M, N], &c);
                    per_lane[j].push(format!(r#"{{"seed":"{}","out_dtype":"{out}"}}"#, seed_name(i)));
                }
                Err(reason) => per_lane[j].push(format!(r#"{{"seed":"{}","skipped":"{reason}"}}"#, seed_name(i))),
            }
        }
    }
    for (j, e) in exts.iter().enumerate() {
        entries.push(format!(
            r#"{{"lane":"{}","dtype":"{}","seeds":[{}]}}"#,
            e.lane,
            e.dtype,
            per_lane[j].join(",")
        ));
    }
    write_externals_raw(root, &format!(r#"{{"lanes":[{}]}}"#, entries.join(",")));
}

fn write_externals_raw(root: &Path, text: &str) {
    std::fs::write(root.join("external_lanes.json"), text).unwrap();
}

fn stub_f32() -> Vec<Ext> {
    vec![ext("mlx-f32", "f32", exact), ext("torch-mps-f32", "f32", exact)]
}

fn stub_bf16() -> Vec<Ext> {
    vec![
        ext("mlx-bf16", "bf16", bf16_runtime),
        ext("torch-mps-bf16", "bf16", bf16_runtime),
    ]
}

/// A dump with `exts` written beside it.
fn dump_with(tag: &str, seeds: usize, exts: Vec<Ext>) -> PathBuf {
    let d = build_dump(&scratch(tag), seeds, &LANES, None);
    write_externals(&d, seeds, exts);
    d
}

// ============================================================ scorer cases

#[test]
fn a_valid_dump_is_accepted_and_reported_once_per_lane() {
    let d = dump_with("valid", 3, stub_f32());
    let rows = accepts("valid dump", score(&d));
    let names: Vec<&str> = rows.iter().map(|r| r.lane()).collect();
    let mut sorted = names.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), names.len(), "duplicate lane rows: {names:?}");
    assert_eq!(
        names,
        [
            "tensorops-f32",
            "simdgroup-f32",
            "tensorops-bf16",
            "tensorops-tf32",
            "mlx-f32",
            "torch-mps-f32"
        ]
    );
    for lane in LANES {
        assert_eq!(scored(&rows, lane).seeds, 3, "{lane} did not report all 3 seeds");
    }
}

#[test]
fn the_aggregate_reports_the_worst_seed_not_the_last_or_the_mean() {
    // Seed 1 is perturbed hardest; a scorer that keeps the last seed or averages
    // would under-report it.
    let d = build_dump(&scratch("worst"), 3, &LANES, Some(&[1e-8, 5e-7, 1e-8]));
    write_externals(&d, 3, stub_f32());
    let rows = accepts("worst-of-seeds", score(&d));
    let row = scored(&rows, "tensorops-f32");
    let per = &row.per_seed_rel_err;
    assert!(
        (row.max_rel_err - per.iter().cloned().fold(0.0, f64::max)).abs() <= 1e-12,
        "max_rel_err {} is not the worst seed of {per:?}",
        row.max_rel_err
    );
    assert!(per[1] > per[0] * 10.0, "the perturbation did not register: {per:?}");
}

#[test]
fn a_manifest_that_is_absent_malformed_or_ill_shaped_is_refused() {
    let tmp = scratch("manifest");
    let cases: Vec<(&str, Option<String>, &str)> = vec![
        ("missing manifest", None, "manifest"),
        ("malformed manifest", Some("{not json".to_string()), "unreadable"),
        (
            "manifest lanes not a list",
            Some(manifest_text("\"tensorops-f32\"", &seeds_json(2), "16")),
            "non-empty list",
        ),
        (
            "manifest lanes not strings",
            Some(manifest_text("[1,2]", &seeds_json(2), "16")),
            "strings",
        ),
        (
            "manifest lanes empty",
            Some(manifest_text("[]", &seeds_json(2), "16")),
            "non-empty list",
        ),
        (
            "manifest seeds empty",
            Some(manifest_text(&lanes_json(&LANES), "[]", "16")),
            "non-empty list",
        ),
        (
            "manifest zero dim",
            Some(manifest_text(&lanes_json(&LANES), &seeds_json(2), "0")),
            "positive integer",
        ),
        (
            "manifest negative dim",
            Some(manifest_text(&lanes_json(&LANES), &seeds_json(2), "-4")),
            "positive integer",
        ),
        (
            "manifest fractional dim",
            Some(manifest_text(&lanes_json(&LANES), &seeds_json(2), "16.5")),
            "positive integer",
        ),
        ("manifest not an object", Some("[1,2,3]".to_string()), "object"),
    ];
    for (i, (label, text, needle)) in cases.into_iter().enumerate() {
        let d = build_dump(&tmp.join(format!("c{i}")), 2, &LANES, None);
        match text {
            None => std::fs::remove_file(d.join("parity_manifest.json")).unwrap(),
            Some(t) => std::fs::write(d.join("parity_manifest.json"), t).unwrap(),
        }
        refuses(label, score(&d), needle);
    }
}

#[test]
fn a_manifest_cannot_name_a_path_outside_the_dump_or_a_lane_twice() {
    let tmp = scratch("manifest-hardening");
    for (i, (label, lanes, seeds, needle)) in [
        (
            "lane with a path separator",
            lanes_json(&["../escape"]),
            seeds_json(2),
            "plain name",
        ),
        (
            "seed with a path separator",
            lanes_json(&LANES),
            "[\"../seed_00\"]".to_string(),
            "plain name",
        ),
        (
            "seed called ..",
            lanes_json(&LANES),
            "[\"..\"]".to_string(),
            "plain name",
        ),
        ("empty lane name", lanes_json(&[""]), seeds_json(2), "plain name"),
        (
            "lane listed twice",
            lanes_json(&["tensorops-f32", "tensorops-f32"]),
            seeds_json(2),
            "more than once",
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let d = build_dump(&tmp.join(format!("c{i}")), 2, &LANES, None);
        std::fs::write(d.join("parity_manifest.json"), manifest_text(&lanes, &seeds, "16")).unwrap();
        refuses(label, score(&d), needle);
    }
}

#[test]
fn a_manifest_that_promises_more_than_the_dump_holds_is_refused() {
    // Exactly the pre-fix state: the two reduced-precision lanes absent while the
    // manifest still claims four.
    let d = build_dump(&scratch("shortlanes"), 2, &LANES, None);
    for s in 0..2 {
        for lane in ["tensorops-bf16", "tensorops-tf32"] {
            std::fs::remove_file(d.join(seed_name(s)).join(format!("parity_c_{lane}.npy"))).unwrap();
        }
    }
    refuses("lane in manifest, npy absent", score(&d), "missing");

    let d = build_dump(&scratch("shortseed"), 3, &LANES, None);
    std::fs::remove_dir_all(d.join("seed_02")).unwrap();
    refuses("seed in manifest, directory absent", score(&d), "absent");
}

#[test]
fn non_finite_results_operands_and_degenerate_references_are_refused() {
    for (label, bad) in [("NaN", f32::NAN), ("+Inf", f32::INFINITY), ("-Inf", f32::NEG_INFINITY)] {
        let d = build_dump(&scratch(&format!("nf{label}")), 2, &LANES, None);
        let fnm = d.join("seed_00").join("parity_c_tensorops-tf32.npy");
        let mut c = load(&fnm);
        c[3 * N + 4] = bad;
        save(&fnm, &[M, N], &c);
        refuses(&format!("{label} in a result"), score(&d), "non-finite");
    }

    let d = build_dump(&scratch("wrongshape"), 2, &LANES, None);
    save(
        &d.join("seed_00/parity_c_tensorops-f32.npy"),
        &[M, N + 3],
        &vec![0.0; M * (N + 3)],
    );
    refuses("result shape mismatch", score(&d), "shape");

    let d = build_dump(&scratch("zeroops"), 2, &LANES, None);
    save(&d.join("seed_00/parity_a.npy"), &[M, K], &vec![0.0; M * K]);
    refuses("all-zero operands (scale 0)", score(&d), "division by zero");

    let d = build_dump(&scratch("nanops"), 2, &LANES, None);
    let mut a = load(&d.join("seed_00/parity_a.npy"));
    a[0] = f32::NAN;
    save(&d.join("seed_00/parity_a.npy"), &[M, K], &a);
    refuses("non-finite operands", score(&d), "non-finite");

    let d = build_dump(&scratch("opshape"), 2, &LANES, None);
    save(&d.join("seed_00/parity_b.npy"), &[K + 2, N], &vec![1.0; (K + 2) * N]);
    refuses("operand shape contradicts manifest", score(&d), "contradict");
}

#[test]
fn a_lane_over_its_per_element_budget_is_refused() {
    // A lane that writes nothing. The Rust side pre-zeroes C so this cannot
    // inherit the previous lane's result; the budget gate then rejects it
    // outright rather than merely scoring it 1.0.
    let d = dump_with("zeroed", 2, stub_f32());
    for s in 0..2 {
        save(
            &d.join(seed_name(s)).join("parity_c_tensorops-tf32.npy"),
            &[M, N],
            &vec![0.0; M * N],
        );
    }
    refuses("no-op lane (all-zero result)", score(&d), "budget exceeded");

    // A drift of 1e-5 relative: innocuous by the normwise metric, and ~2x the f32
    // lane's per-element budget. This is the case the old harness could not see.
    let d = dump_with("drifted", 2, stub_f32());
    for s in 0..2 {
        let fnm = d.join(seed_name(s)).join("parity_c_tensorops-f32.npy");
        let c: Vec<f32> = load(&fnm).iter().map(|&v| (v as f64 * (1.0 + 1e-5)) as f32).collect();
        save(&fnm, &[M, N], &c);
    }
    refuses(
        "f32 lane drifted 1e-5 (normwise would shrug)",
        score(&d),
        "kernel defect",
    );
}

#[test]
fn a_lane_with_no_declared_roundoff_is_not_scored_against_a_default() {
    let mut lanes = LANES.to_vec();
    lanes.push("tensorops-fp8");
    let d = build_dump(&scratch("unknownlane"), 2, &lanes, None);
    refuses("lane with no LANE_PRECISION entry", score(&d), "LANE_PRECISION");
}

// ============================================================ verdicts

fn over(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
    pairs.iter().map(|(l, v)| (l.to_string(), *v)).collect()
}

#[test]
fn the_three_budget_verdicts_name_the_response_they_call_for() {
    for (name, o, needle) in [
        ("tessl only", over(&[("tensorops-bf16", 2.0)]), "kernel defect"),
        (
            "comparison runtime only",
            over(&[("torch-mps-bf16", 2.0)]),
            "other runtime is the outlier",
        ),
        (
            "every runtime",
            over(&[("tensorops-bf16", 2.0), ("torch-mps-bf16", 2.1), ("mlx-bf16", 2.2)]),
            "points at the bound",
        ),
    ] {
        let msg = adjudicate(&o, true);
        assert!(msg.contains(needle), "adjudicate({name}) lacks {needle:?}: {msg}");
        assert!(msg.starts_with("per-element error budget exceeded: "), "{msg}");
    }
}

#[test]
fn a_verdict_never_claims_corroboration_it_lacks() {
    let msg = adjudicate(&over(&[("tensorops-f32", 2.0)]), false);
    assert!(
        !msg.contains("stayed inside"),
        "claims the externals stayed inside with none scored: {msg}"
    );
    assert!(msg.contains("unconfirmed"), "{msg}");
    assert!(msg.contains("tensorops-f32 2.000x"), "{msg}");
}

// ====================================================== external-lane contract

#[test]
fn bf16_externals_are_scored_through_the_output_roundoff_budget() {
    let mut exts = stub_f32();
    exts.extend(stub_bf16());
    let d = dump_with("ext-bf16", 3, exts);
    let rows = accepts("f32 + bf16 stub lanes", score(&d));
    for lane in ["mlx-bf16", "torch-mps-bf16"] {
        let r = scored(&rows, lane);
        assert_eq!(
            (r.out_dtype.as_str(), r.seeds),
            ("bf16", 3),
            "{lane} stub not scored as bf16 on 3 seeds"
        );
    }
    for r in &rows {
        if let LaneRow::Scored(s) = r {
            assert_eq!(
                s.external_lanes_scored, 4,
                "{}: rows do not report 4 comparison runtimes scored",
                s.lane
            );
        }
    }
}

#[test]
fn a_lane_that_cannot_run_on_any_seed_is_a_visible_skip_with_its_reason() {
    let d = dump_with(
        "ext-boom",
        3,
        vec![ext("mlx-f32", "f32", boom), ext("torch-mps-f32", "f32", exact)],
    );
    let rows = accepts("raises on every seed", score(&d));
    match rows.iter().find(|r| r.lane() == "mlx-f32") {
        Some(LaneRow::Skipped { reason, .. }) => assert!(reason.contains("stub boom"), "{reason}"),
        other => panic!("skip reason not surfaced: {other:?}"),
    }
    assert_eq!(
        scored(&rows, "torch-mps-f32").external_lanes_scored,
        1,
        "a skipped runtime was counted as scored"
    );
}

#[test]
fn a_lane_that_covers_some_seeds_and_not_others_is_refused() {
    let mut calls = 0;
    let flaky: Ext = Ext {
        lane: "mlx-f32",
        dtype: "f32",
        run: Box::new(move |a, b| {
            calls += 1;
            if calls == 3 {
                Err("stub seed-2 failure".to_string())
            } else {
                exact(a, b)
            }
        }),
    };
    let d = dump_with("ext-partial", 3, vec![flaky]);
    refuses("raises on seed 2 only", score(&d), "cover every seed");
}

#[test]
fn a_runtime_whose_output_dtype_flips_between_seeds_is_refused() {
    let mut calls = 0;
    let flip = Ext {
        lane: "mlx-f32",
        dtype: "f32",
        run: Box::new(move |a, b| {
            calls += 1;
            let (c, _) = exact(a, b)?;
            Ok((c, if calls == 1 { "f32" } else { "bf16" }))
        }),
    };
    let d = dump_with("ext-flip", 3, vec![flip]);
    refuses("dtype label flips between seeds", score(&d), "dtype changed");
}

#[test]
fn a_malformed_external_lane_list_is_refused_up_front() {
    let ok = || ext("mlx-f32", "f32", exact);
    for (label, exts, needle) in [
        ("duplicate name", vec![ok(), ok()], "more than once"),
        (
            "collides with a dumped lane",
            vec![ext("tensorops-f32", "f32", exact)],
            "collides",
        ),
        ("unknown name", vec![ext("cuda-f32", "f32", exact)], "LANE_PRECISION"),
        ("bad dtype", vec![ext("mlx-f32", "f16", exact)], "dtype"),
    ] {
        let d = dump_with(&format!("ext-bad-{}", label.replace(' ', "-")), 3, exts);
        refuses(&format!("external_lanes {label}"), score(&d), needle);
    }
}

#[test]
fn an_external_lane_file_that_is_not_the_contract_is_refused() {
    // The Python harness refused a lane entry that was not a (lane, fn, dtype)
    // 3-tuple, a lane list that was not iterable, and an fn that was not
    // callable. The file contract has the same three ways to be malformed.
    let seeds = r#"[{"seed":"seed_00","out_dtype":"f32"},{"seed":"seed_01","out_dtype":"f32"}]"#;
    for (label, text, needle) in [
        ("not JSON", "{nope".to_string(), "unreadable"),
        ("lanes is not a list", r#"{"lanes":7}"#.to_string(), "must be a list"),
        ("entry is not an object", r#"{"lanes":[3]}"#.to_string(), "object"),
        ("lane is not a string", format!(r#"{{"lanes":[{{"lane":4,"dtype":"f32","seeds":{seeds}}}]}}"#), "must be a string"),
        ("seeds is not a list", r#"{"lanes":[{"lane":"mlx-f32","dtype":"f32","seeds":"x"}]}"#.to_string(), "seeds"),
        (
            "a seed is reported twice",
            r#"{"lanes":[{"lane":"mlx-f32","dtype":"f32","seeds":[{"seed":"seed_00","out_dtype":"f32"},{"seed":"seed_00","out_dtype":"f32"}]}]}"#.to_string(),
            "twice",
        ),
        (
            "a seed is both scored and skipped",
            r#"{"lanes":[{"lane":"mlx-f32","dtype":"f32","seeds":[{"seed":"seed_00","out_dtype":"f32","skipped":"x"},{"seed":"seed_01","out_dtype":"f32"}]}]}"#.to_string(),
            "exactly one of",
        ),
        (
            "a seed is neither scored nor skipped",
            r#"{"lanes":[{"lane":"mlx-f32","dtype":"f32","seeds":[{"seed":"seed_00"},{"seed":"seed_01","out_dtype":"f32"}]}]}"#.to_string(),
            "exactly one of",
        ),
        (
            "the writer did not finish (a seed has no entry)",
            r#"{"lanes":[{"lane":"mlx-f32","dtype":"f32","seeds":[{"seed":"seed_00","out_dtype":"f32"}]}]}"#.to_string(),
            "no result for seed",
        ),
    ] {
        let d = build_dump(&scratch(&format!("ext-file-{}", label.len())), 2, &LANES, None);
        write_externals_raw(&d, &text);
        refuses(label, score(&d), needle);
    }
}

#[test]
fn a_scored_external_whose_output_file_is_missing_is_refused() {
    let d = dump_with("ext-nofile", 2, stub_f32());
    std::fs::remove_file(d.join("seed_01/parity_c_mlx-f32.npy")).unwrap();
    refuses("reported output absent", score(&d), "missing");
}

#[test]
fn no_external_file_means_no_comparison_runtimes_and_says_so() {
    let d = build_dump(&scratch("ext-none"), 2, &LANES, None);
    let rows = accepts("empty external lanes", score(&d));
    assert_eq!(rows.len(), 4);
    for r in &rows {
        if let LaneRow::Scored(s) = r {
            assert_eq!(
                s.external_lanes_scored, 0,
                "{} claims runtimes that were not scored",
                s.lane
            );
        }
    }
    // The note is on stderr of the binary: loud, not silent.
    let out = Command::new(env!("CARGO_BIN_EXE_gemm_parity"))
        .args(["score"])
        .arg(&d)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("NO COMPARISON RUNTIMES SCORED"), "no loud note: {err}");
}

#[test]
fn a_tessl_only_breach_with_no_externals_is_unconfirmed_not_a_kernel_defect() {
    let d = build_dump(&scratch("ext-none-drift"), 2, &LANES, None);
    for s in 0..2 {
        let fnm = d.join(seed_name(s)).join("parity_c_tensorops-f32.npy");
        let c: Vec<f32> = load(&fnm).iter().map(|&v| (v as f64 * (1.0 + 1e-5)) as f32).collect();
        save(&fnm, &[M, N], &c);
    }
    match score(&d) {
        Err(msg) => {
            assert!(msg.contains("unconfirmed"), "{msg}");
            assert!(
                !msg.contains("stayed inside"),
                "verdict claims corroboration it lacks: {msg}"
            );
        }
        Ok(rows) => panic!("drifted lane accepted with no comparison runtimes: {rows:?}"),
    }
}

// ------------------------------------------------ mlx-f32 is a tf32-class lane

#[test]
fn mlx_f32_is_scored_as_the_tf32_class_lane_it_measures_as() {
    // tf32-class error is inside its budget...
    let d = dump_with(
        "mlx-tf32",
        3,
        vec![ext("mlx-f32", "f32", tf32_runtime), ext("torch-mps-f32", "f32", exact)],
    );
    accepts("tf32-class mlx-f32 stub", score(&d));

    // ...the same error against an f32-exact claim aborts...
    let exact_claim = Precision::standard().with("mlx-f32", (0.0, 0.0));
    refuses(
        "f32-exact claim for mlx-f32",
        parity::parity(&d, &exact_claim),
        "mlx-f32",
    );

    // ...and torch-mps-f32 stays f32-exact: a tf32-class torch lane aborts.
    let d = dump_with(
        "torch-tf32",
        3,
        vec![ext("mlx-f32", "f32", exact), ext("torch-mps-f32", "f32", tf32_runtime)],
    );
    refuses("tf32-class torch-mps-f32 stub", score(&d), "torch-mps-f32");
}

// ============================================================= ladder merge

fn cell(shape: &str, dist: &str) -> parity::Cell {
    (shape.to_string(), dist.to_string())
}

fn row(lane: &str, cell: &parity::Cell, rel: f64, dtype: &str) -> CellRow {
    CellRow {
        row: LaneRow::Scored(Scored {
            lane: lane.to_string(),
            out_dtype: dtype.to_string(),
            seeds: 2,
            max_rel_err: rel,
            median_rel_err: rel,
            min_rel_err: rel,
            max_abs_err: rel,
            mean_abs_err: rel,
            max_elem_rel_err: rel,
            max_higham_ratio: rel,
            per_seed_rel_err: vec![rel],
            per_seed_higham: vec![rel],
            external_lanes_scored: 0,
        }),
        shape: cell.0.clone(),
        dist: cell.1.clone(),
        m: 1,
        n: 1,
        k: 1,
    }
}

fn skipped_row(lane: &str, cell: &parity::Cell) -> CellRow {
    CellRow {
        row: LaneRow::Skipped {
            lane: lane.to_string(),
            reason: "mlx missing".to_string(),
        },
        shape: cell.0.clone(),
        dist: cell.1.clone(),
        m: 1,
        n: 1,
        k: 1,
    }
}

fn grid() -> Vec<parity::Cell> {
    vec![
        cell("square_512", "uniform"),
        cell("square_1024", "uniform"),
        cell("mlp_down", "heavy_tail"),
    ]
}

#[test]
fn a_full_grid_merges_to_the_worst_cell() {
    let cells = grid();
    let full: Vec<CellRow> = cells
        .iter()
        .zip([1e-6, 5e-6, 9e-6])
        .map(|(c, r)| row("L", c, r, "f32"))
        .collect();
    let out = parity::merge(&full, &cells).expect("full grid merges");
    match &out[0] {
        parity::LaneMerge::Scored {
            worst_shape,
            worst_rel_err,
            cells: n,
            per_cell_rel_err,
            ..
        } => {
            assert_eq!(worst_shape, "mlp_down");
            assert!((worst_rel_err - 9e-6).abs() <= 1e-18, "worst is {worst_rel_err}");
            assert_eq!(*n, 3);
            assert_eq!(
                per_cell_rel_err[0].0, "square_512/uniform",
                "per-cell keys keep the grid order"
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_lane_must_cover_the_whole_grid_or_none_of_it() {
    let cells = grid();
    let full: Vec<CellRow> = cells
        .iter()
        .zip([1e-6, 5e-6, 9e-6])
        .map(|(c, r)| row("L", c, r, "f32"))
        .collect();

    let msg = parity::merge(&full[..2], &cells).unwrap_err();
    assert!(msg.contains("whole grid"), "lane absent from one cell: {msg}");

    let mut partial: Vec<CellRow> = full[..2].to_vec();
    partial.push(skipped_row("L", &cells[2]));
    let msg = parity::merge(&partial, &cells).unwrap_err();
    assert!(
        msg.contains("whole grid"),
        "lane scored on some cells, skipped on others: {msg}"
    );
}

#[test]
fn a_lane_skipped_on_the_whole_grid_is_reported_skipped_not_scored() {
    let cells = grid();
    let allskip: Vec<CellRow> = cells.iter().map(|c| skipped_row("L", c)).collect();
    let out = parity::merge(&allskip, &cells).expect("an all-skipped lane merges");
    assert!(
        matches!(&out[0], parity::LaneMerge::Skipped { reason, .. } if reason == "mlx missing"),
        "{out:?}"
    );
}

#[test]
fn an_output_dtype_that_varies_across_cells_is_refused() {
    let cells = grid();
    let mixed = vec![
        row("L", &cells[0], 1e-6, "f32"),
        row("L", &cells[1], 1e-6, "bf16"),
        row("L", &cells[2], 1e-6, "f32"),
    ];
    let msg = parity::merge(&mixed, &cells).unwrap_err();
    assert!(msg.contains("dtype varies"), "{msg}");
}

#[test]
fn the_first_of_tied_cells_names_the_worst_one() {
    // Python's max(key=...) keeps the earliest of a tie; a later cell with the
    // same number must not take the name.
    let cells = grid();
    let rows: Vec<CellRow> = cells.iter().map(|c| row("L", c, 3e-6, "f32")).collect();
    match &parity::merge(&rows, &cells).unwrap()[0] {
        parity::LaneMerge::Scored {
            worst_shape,
            worst_higham_shape,
            worst_elem_shape,
            ..
        } => {
            assert_eq!(
                (
                    worst_shape.as_str(),
                    worst_higham_shape.as_str(),
                    worst_elem_shape.as_str()
                ),
                ("square_512", "square_512", "square_512")
            );
        }
        other => panic!("{other:?}"),
    }
}

// ================================================== the port is the same function

#[test]
fn the_oracle_dump_scores_to_the_rows_the_python_scorer_produced() {
    // `tests/fixtures/gemm_parity/oracle_dump` is 2 seeds of 24x20x48 heavy-tailed
    // operands with four tessl lanes and four comparison runtimes (one of which
    // is skipped), scored by bench/gemm_sweep_mlx.py's `parity()` at b219aff.
    // The rows are compared as text: the same numbers in the same bytes
    // `json.dumps(rows, indent=2)` wrote.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gemm_parity/oracle_dump");
    let rows = accepts("oracle dump", score(&dir));
    let want = std::fs::read_to_string(dir.join("oracle_rows.json")).unwrap();
    let got = parity::rows_json(&rows).unwrap() + "\n";
    assert_eq!(
        got, want,
        "the Rust scorer disagrees with the Python scorer it replaced"
    );
    assert!(matches!(rows.last(), Some(LaneRow::Skipped { reason, .. }) if reason == "stub boom"));
}

fn series(n: usize) -> Vec<f64> {
    const SCALES: [f64; 7] = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0];
    (0..n)
        .map(|i| (((i as u64 * 2654435761) % (1 << 32)) as f64 / 4294967296.0 - 0.5) * SCALES[i % 7])
        .collect()
}

#[test]
fn the_pairwise_mean_is_numpys_to_the_last_bit() {
    // ndarray.mean() of the same series, bit patterns from numpy 2.5.3. A plain
    // left-to-right sum would differ in the last bits from n = 8 up.
    const MEAN_GOLDENS: &[(usize, u64)] = &[
        (1, 0xbf40624dd2f1a9fc),
        (2, 0x3f364b1abdb22d0e),
        (7, 0x4037e553bc169c99),
        (8, 0x4034e8a7d82bf8dd),
        (9, 0x4032960ae11797cf),
        (15, 0xc03443c9e90342fd),
        (16, 0xc032ff96b0d41e52),
        (17, 0xc031e0e8fb599e0b),
        (127, 0x400f731340276372),
        (128, 0x400f342cb41ca66a),
        (129, 0x400ef5a1069d05f7),
        (130, 0x400ebc3aa3404ec8),
        (255, 0x400f9953f6898f5b),
        (256, 0x400f7a84b44d7dcb),
        (257, 0x400f4496257805c5),
        (511, 0x40074792162dfc3c),
        (1000, 0x3fe0240d88dd4568),
        (1024, 0xbfdb12bf7733aa70),
        (4097, 0x3fd9feaf28e8a3d6),
        (100003, 0x3f8153fd58b96d35),
    ];
    for &(n, bits) in MEAN_GOLDENS {
        let got = parity::np_mean(&series(n));
        assert_eq!(
            got.to_bits(),
            bits,
            "np.mean of {n}: {got:e} vs {:e}",
            f64::from_bits(bits)
        );
    }
    // The naive left-to-right sum really is different, so the goldens bite.
    let naive: f64 = series(100003).iter().sum::<f64>() / 100003.0;
    assert_ne!(
        naive.to_bits(),
        0x3f8153fd58b96d35,
        "the goldens would not catch a plain sum"
    );
}

#[test]
fn fsum_and_median_are_pythons() {
    // statistics.fmean(list) == math.fsum(list) / len; bit patterns from Python 3.13.
    const FMEAN_GOLDENS: &[(usize, u64)] = &[
        (1, 0xbf40624dd2f1a9fc),
        (2, 0x3f364b1abdb22d0e),
        (3, 0xbf818da601ea2799),
        (7, 0x4037e553bc169c99),
        (10, 0x4030ba65d2f9a4ca),
        (100, 0x400f3607c738b84d),
        (1000, 0x3fe0240d88dd4561),
    ];
    for &(n, bits) in FMEAN_GOLDENS {
        let s = series(n);
        let got = parity::fsum(&s) / n as f64;
        assert_eq!(got.to_bits(), bits, "fmean of {n}: {got:e}");
    }
    assert_eq!(parity::median(&series(9)).to_bits(), 0xbf26c67d028f5c29);
    assert_eq!(parity::median(&series(10)).to_bits(), 0x3f407de4a8000000);
}

#[test]
fn numbers_print_the_way_python_prints_them() {
    const REPRS: &[(u64, &str)] = &[
        (0x0000000000000000, "0.0"),
        (0x8000000000000000, "-0.0"),
        (0x3ff0000000000000, "1.0"),
        (0x3fe0000000000000, "0.5"),
        (0x3e7ad7f29abcaf48, "1e-07"),
        (0x3ee9e3abe16fc70d, "1.2345e-05"),
        (0x3f1a36e2eb1c432d, "0.0001"),
        (0x3f202e4b6ce5dc68, "0.00012345"),
        (0x42dc12218377de40, "123456789012345.0"),
        (0x430c6bf526340000, "1000000000000000.0"),
        (0x4341c37937e08000, "1e+16"),
        (0x434aa535d3d0c000, "1.5e+16"),
        (0x437b69b4ba630f34, "1.2345678901234566e+17"),
        (0x4480f0cf064dd592, "1e+22"),
        (0x0000000000000001, "5e-324"),
        (0x7fefffffffffffff, "1.7976931348623157e+308"),
        (0xbe25798ee2308c3a, "-2.5e-09"),
        (0x4059000000000000, "100.0"),
        (0x3fd3333333333334, "0.30000000000000004"),
        (0x3eff75104d551d69, "3e-05"),
        (0x40c81cd6c8b43958, "12345.678"),
    ];
    for &(bits, want) in REPRS {
        assert_eq!(pyjson::float_repr(f64::from_bits(bits)), want, "repr of {bits:#x}");
    }
    const FMT_E: &[(u64, &str)] = &[
        (0x3e8091b5aeffdb8e, "1.23e-07"),
        (0x0000000000000000, "0.00e+00"),
        (0x4012000000000000, "4.50e+00"),
        (0x40fe240000000000, "1.23e+05"),
        (0x54b249354bbf381a, "1.00e+100"),
        (0x2b2bff2ee48e0530, "1.00e-100"),
        (0x3f201f31f46ed246, "1.23e-04"),
        (0x0000000000000001, "4.94e-324"),
    ];
    for &(bits, want) in FMT_E {
        assert_eq!(pyjson::fmt_e(f64::from_bits(bits), 2), want, "%.2e of {bits:#x}");
    }
    assert_eq!(pyjson::quote("tensorops-f32"), "'tensorops-f32'");
    assert_eq!(pyjson::quote_list(&["a", "b"]), "['a', 'b']");
}

#[test]
fn json_output_is_what_json_dumps_with_indent_2_writes() {
    use pyjson::{obj, Doc};
    let doc = obj([
        ("a", Doc::Arr(vec![])),
        ("b", Doc::Obj(vec![])),
        (
            "c",
            Doc::Arr(vec![Doc::Int(1), Doc::Float(2.5), Doc::Null, Doc::Bool(true)]),
        ),
        ("d\n\"é", Doc::str("tab\there \u{1f600}")),
    ]);
    let want = "{\n  \"a\": [],\n  \"b\": {},\n  \"c\": [\n    1,\n    2.5,\n    null,\n    true\n  ],\n  \"d\\n\\\"\\u00e9\": \"tab\\there \\ud83d\\ude00\"\n}";
    assert_eq!(doc.dumps().unwrap(), want);
    assert!(
        Doc::Float(f64::NAN).dumps().is_err(),
        "a non-finite number must not reach a JSON file"
    );
    assert!(Doc::Float(f64::INFINITY).dumps().is_err());
}

#[test]
fn the_reference_product_matches_a_plain_triple_loop_on_awkward_shapes() {
    // Shapes that are not multiples of the 4-row band, the 512-column block or
    // the 256-deep block, including degenerate ones.
    for &(m, k, n) in &[
        (1, 1, 1),
        (3, 5, 7),
        (5, 300, 9),
        (9, 17, 515),
        (4, 256, 512),
        (13, 257, 1030),
        (2, 0, 3),
    ] {
        let mut rng = Rng((m * 1000 + k * 10 + n) as u64);
        let a: Vec<f64> = rng.fill(m * k).iter().map(|&v| v as f64 * 3.0).collect();
        let b: Vec<f64> = rng.fill(k * n).iter().map(|&v| v as f64).collect();
        let (c, s) = parity::reference_gemm(&a, &b, m, k, n);
        for i in 0..m {
            for j in 0..n {
                let (mut cc, mut ss) = (0.0f64, 0.0f64);
                for kk in 0..k {
                    cc = a[i * k + kk].mul_add(b[kk * n + j], cc);
                    ss = a[i * k + kk].abs().mul_add(b[kk * n + j].abs(), ss);
                }
                // Same ascending order, same fused step: identical, not close.
                assert_eq!(c[i * n + j].to_bits(), cc.to_bits(), "C[{i}][{j}] at {m}x{k}x{n}");
                assert_eq!(s[i * n + j].to_bits(), ss.to_bits(), "S[{i}][{j}] at {m}x{k}x{n}");
            }
        }
    }
}

#[test]
fn the_reference_does_not_depend_on_how_many_threads_ran_it() {
    // 70 rows is 18 bands; a one-thread run and a many-thread run take the bands
    // in different orders and must agree to the bit.
    let (m, k, n) = (70, 130, 40);
    let mut rng = Rng(7);
    let a: Vec<f64> = rng.fill(m * k).iter().map(|&v| v as f64).collect();
    let b: Vec<f64> = rng.fill(k * n).iter().map(|&v| v as f64).collect();
    let first = parity::reference_gemm(&a, &b, m, k, n);
    for _ in 0..5 {
        let again = parity::reference_gemm(&a, &b, m, k, n);
        assert_eq!(
            first.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            again.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn the_budget_degenerates_loudly_at_an_impossible_k() {
    // n*u >= 1: the classical bound says nothing, and a number built on it would
    // be a plausible-looking nothing.
    let msg = parity::gamma(1 << 24, parity::U_F32).unwrap_err();
    assert!(msg.contains("degenerate"), "{msg}");
    assert!(parity::gamma(1 << 20, parity::U_F32).unwrap() > 0.0);
}

// ================================================================== the binary

fn gemm_parity() -> Command {
    Command::new(env!("CARGO_BIN_EXE_gemm_parity"))
}

fn run_ok(cmd: &mut Command) -> std::process::Output {
    let out = cmd.output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    out
}

#[test]
fn score_prints_the_rows_or_writes_them_without_a_trailing_newline() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gemm_parity/oracle_dump");
    let want = std::fs::read_to_string(dir.join("oracle_rows.json")).unwrap();
    let out = run_ok(gemm_parity().arg("score").arg(&dir));
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        want,
        "stdout is json.dumps plus the print newline"
    );

    let tmp = scratch("score-out");
    let file = tmp.join("rows.json");
    run_ok(gemm_parity().arg("score").arg(&dir).arg("--out").arg(&file));
    assert_eq!(
        std::fs::read_to_string(&file).unwrap(),
        want.trim_end_matches('\n'),
        "--out wrote a trailing newline"
    );
}

#[test]
fn score_exits_nonzero_with_the_message_on_a_refused_dump() {
    let out = gemm_parity().arg("score").arg(scratch("score-empty")).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("parity_manifest.json"),
        "{:?}",
        out.stderr
    );
}

#[test]
fn usage_errors_exit_2() {
    for args in [
        &[][..],
        &["nonsense"],
        &["score"],
        &["score", "a", "b"],
        &["ladder", "--bogus"],
        &["ladder", "--seeds"],
    ] {
        let out = gemm_parity().args(args).output().unwrap();
        assert_eq!(
            out.status.code(),
            Some(2),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn ladder_refuses_bad_arguments_before_running_anything() {
    let fake = scratch("ladder-args").join("not_a_binary");
    std::fs::write(&fake, "").unwrap();
    for (args, needle) in [
        (vec!["--seeds", "0"], "--seeds must be >= 1"),
        (vec!["--seeds", "abc"], "--seeds must be a positive integer"),
        (vec!["--shapes", "square_512,nonsense"], "unknown ladder label"),
        (vec!["--dists", "uniform,gaussian"], "unknown distribution"),
        (vec!["--external", "maybe"], "--external must be auto or none"),
    ] {
        let out = gemm_parity()
            .arg("ladder")
            .args(&args)
            .arg("--binary")
            .arg(&fake)
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(1), "{args:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains(needle), "{args:?}: {err}");
    }
    let out = gemm_parity()
        .args(["ladder", "--binary", "/definitely/not/built"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("not built"));
}

/// A stand-in for `bench_gemm_sweep` that does what the real one does for the
/// parity dump (reads `--dump-parity DIR` and the BENCH_PARITY_* variables) by
/// copying a template dump and stamping the manifest with what it was asked for,
/// or with something else, to play a drifted binary.
fn fake_sweep(dir: &Path, template: &Path, shape: &str, dist: &str, exit: i32) -> PathBuf {
    let script = dir.join("bench_gemm_sweep");
    let body = format!(
        "#!/bin/sh\n\
         [ \"{exit}\" = 0 ] || {{ echo 'fake sweep failed' >&2; exit {exit}; }}\n\
         [ \"$1\" = --dump-parity ] || exit 9\n\
         cp -R '{}'/. \"$2\" || exit 8\n\
         sed -e 's/\"shape\":\"[^\"]*\"/\"shape\":\"{shape}\"/' -e 's/\"dist\":\"[^\"]*\"/\"dist\":\"{dist}\"/' \
         '{}'/parity_manifest.json > \"$2\"/parity_manifest.json\n",
        template.display(),
        template.display()
    );
    std::fs::write(&script, body).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    script
}

/// A conforming 512x512x512 single-seed dump: the only shape a stand-in can
/// stand in for without the real ladder's dimensions being wrong.
fn ladder_template(tag: &str) -> PathBuf {
    let root = scratch(tag);
    let t = root.join("template");
    let sd = t.join("seed_00");
    std::fs::create_dir_all(&sd).unwrap();
    let (m, k, n) = (512, 512, 512);
    let mut rng = Rng(5);
    let (a, b) = (rng.fill(m * k), rng.fill(k * n));
    save(&sd.join("parity_a.npy"), &[m, k], &a);
    save(&sd.join("parity_b.npy"), &[k, n], &b);
    let c: Vec<f32> = product(&a, &b, m, k, n).iter().map(|&v| v as f32).collect();
    for lane in ["tensorops-f32", "simdgroup-f32"] {
        save(&sd.join(format!("parity_c_{lane}.npy")), &[m, n], &c);
    }
    std::fs::write(
        t.join("parity_manifest.json"),
        r#"{"shape":"square_512","dist":"uniform","m":512,"n":512,"k":512,"lanes":["tensorops-f32","simdgroup-f32"],"seeds":["seed_00"]}"#,
    )
    .unwrap();
    t
}

#[test]
fn ladder_scores_a_cell_merges_it_and_writes_the_grid_document() {
    let t = ladder_template("ladder-ok");
    let root = t.parent().unwrap().to_path_buf();
    let bin = fake_sweep(&root, &t, "square_512", "uniform", 0);
    let out_file = root.join("grid.json");
    let out = run_ok(
        gemm_parity()
            .args([
                "ladder",
                "--shapes",
                "square_512",
                "--dists",
                "uniform",
                "--seeds",
                "1",
                "--external",
                "none",
            ])
            .arg("--binary")
            .arg(&bin)
            .arg("--out")
            .arg(&out_file),
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("===== [1/1] square_512 / uniform x 1 seeds ====="),
        "{err}"
    );
    assert!(
        err.contains("NO COMPARISON RUNTIMES SCORED"),
        "--external none must say so: {err}"
    );
    let text = std::fs::read_to_string(&out_file).unwrap();
    let doc = json::parse(
        &text,
        json::Syntax {
            what: "grid",
            max_depth: 8,
            uints_only: false,
            literals: true,
        },
    )
    .unwrap();
    let keys: Vec<&str> = match &doc {
        json::Json::Object(f) => f.iter().map(|(k, _)| k.as_str()).collect(),
        _ => panic!("not an object"),
    };
    assert_eq!(keys, ["ladder", "dists", "seeds", "cells", "per_lane", "per_shape"]);
    assert!(text.contains("\"worst_higham_shape\": \"square_512\""), "{text}");
    assert!(
        text.contains("\"m\": 512"),
        "per_shape rows carry the cell's dimensions"
    );
}

#[test]
fn ladder_refuses_a_binary_that_dumped_a_different_shape_or_distribution_or_failed() {
    let t = ladder_template("ladder-drift");
    let root = t.parent().unwrap().to_path_buf();
    let run = |bin: &Path| {
        gemm_parity()
            .args([
                "ladder",
                "--shapes",
                "square_512",
                "--dists",
                "uniform",
                "--seeds",
                "1",
                "--external",
                "none",
            ])
            .arg("--binary")
            .arg(bin)
            .output()
            .unwrap()
    };
    // The manifest names another shape: every number would be attributed to the
    // wrong one.
    let out = run(&fake_sweep(&root, &t, "square_1024", "uniform", 0));
    assert_eq!(out.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("ladder drift"),
        "{:?}",
        String::from_utf8_lossy(&out.stderr)
    );
    // A silently ignored BENCH_PARITY_DIST would attribute uniform numbers to heavy_tail.
    let out = run(&fake_sweep(&root, &t, "square_512", "heavy_tail", 0));
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("distribution drift"));
    // The dump itself failed.
    let out = run(&fake_sweep(&root, &t, "square_512", "uniform", 7));
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("exited 7") && err.contains("fake sweep failed"), "{err}");
}

// =================================================== the three ladders agree

fn between<'a>(text: &'a str, start: &str, end: &str) -> &'a str {
    let i = text.find(start).unwrap_or_else(|| panic!("no {start:?}")) + start.len();
    let j = text[i..]
        .find(end)
        .unwrap_or_else(|| panic!("no {end:?} after {start:?}"))
        + i;
    &text[i..j]
}

/// `(M, N, K, label)` tuples out of a source's ladder, in order.
fn ladder_in(src: &str, open: &str, close: &str) -> Vec<(usize, usize, usize, String)> {
    let mut out = Vec::new();
    for line in between(src, open, close).lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix('(') else { continue };
        let parts: Vec<&str> = rest.trim_end_matches([',', ')']).split(',').map(str::trim).collect();
        if parts.len() == 4 {
            out.push((
                parts[0].parse().unwrap(),
                parts[1].parse().unwrap(),
                parts[2].parse().unwrap(),
                parts[3].trim_matches('"').to_string(),
            ));
        }
    }
    out
}

#[test]
fn the_shape_ladder_is_the_same_in_the_sweep_the_scorer_and_the_python_lanes() {
    // Three hand-maintained copies. If they drift, every number in a ladder
    // artifact is attributed to the wrong shape; the ladder checks each dump
    // against its own copy, and this checks the copies against each other.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let rs = |p: &str| std::fs::read_to_string(root.join(p)).unwrap();
    let sweep = ladder_in(
        &rs("src/bin/bench_gemm_sweep.rs"),
        "const SHAPES: &[(usize, usize, usize, &str)] = &[",
        "];",
    );
    let parity = ladder_in(
        &rs("src/bin/gemm_parity.rs"),
        "const LADDER: &[(usize, usize, usize, &str)] = &[",
        "];",
    );
    let py = rs("bench/gemm_sweep_mlx.py");
    let python = ladder_in(&py, "SHAPES = [", "]");
    assert_eq!(sweep.len(), 8);
    assert_eq!(
        sweep, parity,
        "gemm_parity's LADDER drifted from bench_gemm_sweep's SHAPES"
    );
    assert_eq!(
        sweep, python,
        "bench/gemm_sweep_mlx.py's SHAPES drifted from bench_gemm_sweep's"
    );
}

#[test]
fn the_distribution_names_are_the_dumpers() {
    // gemm_parity walks `Dist::ALL`; the dumper parses the same names.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let common = std::fs::read_to_string(root.join("src/bin/common/mod.rs")).unwrap();
    let names = between(&common, "pub const ALL: [Dist; 5] = [", "];");
    for variant in ["Uniform", "Normal", "LogUniform", "NearCancel", "HeavyTail"] {
        assert!(names.contains(&format!("Dist::{variant}")), "Dist::ALL lacks {variant}");
    }
}
