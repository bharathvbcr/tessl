//! `src/bin/results_report` rebuilds the EmbedGemma 2 result files from the raw
//! runs behind them. It replaced five stdlib-only Python scripts
//! (`bench/results/embedgemma2_{memory,small_m,stage_profile,vision_profile,
//! text_fusion_profile}_report.py`), and its outputs are held byte for byte to
//! two oracles, neither of which needs Python or a GPU:
//!
//! * the committed results: the stage, vision, text-fusion and small-m JSON
//!   files embed every input they were built from, so each case here splits
//!   one back into raw files, runs the tool, and requires the committed JSON
//!   back (and, where the committed `.txt` quotes the tables, those too);
//! * golden outputs under `tests/fixtures/results_report/*/golden`, which the
//!   Python scripts wrote from the small raw inputs beside them, with number
//!   spellings chosen to tell Python's `int`, `float`, `repr` and `json.dumps`
//!   apart from a naive port (`5` against `5.0`, `1e-05`, `1e+16`, `é`).

#[allow(dead_code)]
#[path = "../src/json.rs"]
mod json;

use json::{Json, Syntax};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const SYNTAX: Syntax = Syntax {
    what: "test JSON",
    max_depth: 64,
    uints_only: false,
    literals: true,
};

struct Run {
    code: Option<i32>,
    out: String,
    err: String,
}

fn manifest(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn fresh(case: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("results_report").join(case);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn run<S: AsRef<str>>(args: &[S]) -> Run {
    let out = Command::new(env!("CARGO_BIN_EXE_results_report"))
        .args(args.iter().map(|a| a.as_ref()))
        .output()
        .unwrap();
    Run {
        code: out.status.code(),
        out: String::from_utf8(out.stdout).unwrap(),
        err: String::from_utf8(out.stderr).unwrap(),
    }
}

fn s(p: &Path) -> String {
    p.to_str().unwrap().to_string()
}

fn read(p: &Path) -> String {
    fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// The tool's outputs against the golden directory: stdout and every file the
/// script wrote.
fn assert_golden(case: &str, run: &Run, out_dir: &Path, golden: &str) {
    assert_eq!(run.code, Some(0), "{case}: {}", run.err);
    let golden = manifest(&format!("tests/fixtures/results_report/{golden}/golden"));
    assert_eq!(run.out, read(&golden.join("stdout.txt")), "{case}: stdout");
    let mut compared = 0;
    for entry in fs::read_dir(&golden).unwrap() {
        let name = entry.unwrap().file_name();
        if name == "stdout.txt" {
            continue;
        }
        let want = fs::read(golden.join(&name)).unwrap();
        let got = fs::read(out_dir.join(&name))
            .unwrap_or_else(|e| panic!("{case}: {} was not written: {e}", name.to_string_lossy()));
        assert!(
            want == got,
            "{case}: {} differs from the golden output (first difference at byte {})",
            name.to_string_lossy(),
            want.iter()
                .zip(&got)
                .position(|(a, b)| a != b)
                .unwrap_or(want.len().min(got.len()))
        );
        compared += 1;
    }
    assert!(compared >= 1, "{case}: the golden directory holds no output file");
}

// ---------------------------------------------------------------------------
// Golden outputs from the small fixtures.
// ---------------------------------------------------------------------------

fn fx(rel: &str) -> String {
    s(&manifest(&format!("tests/fixtures/results_report/{rel}")))
}

#[test]
fn memory_matches_the_script_on_the_fixture() {
    let out = fresh("memory_fixture");
    let r = run(&["memory".to_string(), fx("memory/raw"), s(&out)]);
    assert_golden("memory", &r, &out, "memory");
}

#[test]
fn small_m_matches_the_script_on_the_fixture() {
    let out = fresh("small_m_fixture");
    let r = run(&[
        "small-m".to_string(),
        s(&out),
        fx("small_m/raw"),
        "before-27a0407".to_string(),
        "after".to_string(),
        "control-before-again".to_string(),
    ]);
    assert_golden("small-m", &r, &out, "small_m");
}

#[test]
fn stage_matches_the_script_on_the_fixture() {
    let out = fresh("stage_fixture");
    let mut args = vec!["stage".to_string(), s(&out)];
    for p in ["p1", "p2", "p3"] {
        args.push(fx(&format!("stage/raw/stages_{p}.json")));
    }
    for (flag, file) in [("--gemm", "gemm"), ("--ledger", "ledger"), ("--ple", "ple")] {
        args.push(flag.to_string());
        args.push(fx(&format!("stage/raw/{file}.json")));
    }
    args.extend(["--contended".to_string(), "stages_p1".to_string()]);
    assert_golden("stage", &run(&args), &out, "stage");
}

#[test]
fn vision_matches_the_script_on_the_fixture() {
    let out = fresh("vision_fixture");
    let r = run(&[
        "vision".to_string(),
        s(&out),
        format!("before-f32={}", fx("vision/raw/before-f32_0.json")),
        format!("after-f32={}", fx("vision/raw/after-f32_0.json")),
        format!("after-f32={}", fx("vision/raw/after-f32_1.json")),
        format!("after-bf16={}", fx("vision/raw/after-bf16_0.json")),
    ]);
    assert_golden("vision", &r, &out, "vision");
    // The fixture's number spellings, as Python writes them back.
    let json = read(&out.join("embedgemma2_vision_profile_m5pro.json"));
    assert!(
        json.contains(
            "\"number_forms\":[1e-05,100000.0,-0.0,0,1e-06,1e+16,12345678901234567890,1.5e+300,5e-324,0.1,100,1.0,1.0,-5,2.5,\
             1.7976931348623157e+308,1.2345678901234568e+17,0.0001,1e-05,9999999999999998.0,\
             \"caf\\u00e9 \\ud83d\\ude00 \\u0000 \\u007f \\u2028 tab\\t\",true,null,[],{}]"
        ),
        "{}",
        &json[json.find("number_forms").unwrap_or(0)..]
    );
}

#[test]
fn text_fusion_matches_the_script_on_the_fixture() {
    let out = fresh("fusion_fixture");
    let mut args = vec!["text-fusion".to_string(), s(&out)];
    for (label, file) in [
        ("f32 unfused", "f32_unfused_0"),
        ("f32 fused", "f32_fused_1"),
        ("f32 fused", "f32_fused_2"),
        ("bf16 fused", "bf16_fused_3"),
    ] {
        args.push(format!("{label}={}", fx(&format!("fusion/raw/{file}.json"))));
    }
    assert_golden("text-fusion", &run(&args), &out, "fusion");
}

// ---------------------------------------------------------------------------
// The committed results rebuilt from their own embedded inputs.
// ---------------------------------------------------------------------------

fn parse(text: &str) -> Json {
    json::parse(text, SYNTAX).unwrap()
}

fn write_json(v: &Json, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(&b.to_string()),
        Json::Num { .. } if v.exact_int().is_some() => out.push_str(&v.exact_int().unwrap_or_default().to_string()),
        Json::Num { value, .. } => out.push_str(&format!("{value:?}")),
        Json::Str(t) => {
            out.push('"');
            for c in t.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        Json::Array(items) => {
            out.push('[');
            for (k, item) in items.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                write_json(item, out);
            }
            out.push(']');
        }
        Json::Object(fields) => {
            out.push('{');
            for (k, (key, value)) in fields.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                write_json(&Json::Str(key.clone()), out);
                out.push(':');
                write_json(value, out);
            }
            out.push('}');
        }
    }
}

fn json_text(v: &Json) -> String {
    let mut out = String::new();
    write_json(v, &mut out);
    out
}

fn field<'a>(v: &'a Json, key: &str) -> &'a Json {
    v.get(key).unwrap_or_else(|| panic!("no field {key}"))
}

fn fields(v: &Json) -> &[(String, Json)] {
    match v {
        Json::Object(f) => f,
        other => panic!("not an object: {other:?}"),
    }
}

fn items(v: &Json) -> &[Json] {
    match v {
        Json::Array(a) => a,
        other => panic!("not an array: {other:?}"),
    }
}

fn text_of(v: &Json) -> &str {
    match v {
        Json::Str(t) => t,
        other => panic!("not a string: {other:?}"),
    }
}

/// The committed JSON is Python-written: compare ours byte for byte.
fn assert_bytes_equal(what: &str, got: &Path, committed: &str) {
    let want = fs::read(manifest(committed)).unwrap();
    let got = fs::read(got).unwrap_or_else(|e| panic!("{what}: {e}"));
    assert!(
        want == got,
        "{what}: differs from {committed} (first difference at byte {})",
        want.iter()
            .zip(&got)
            .position(|(a, b)| a != b)
            .unwrap_or(want.len().min(got.len()))
    );
}

#[test]
fn the_committed_stage_profile_is_rebuilt_from_its_own_passes() {
    let committed = read(&manifest("bench/results/embedgemma2_stage_profile_m5pro.json"));
    let doc = parse(&committed);
    let dir = fresh("stage_rebuild");
    let out = dir.join("out");
    fs::create_dir_all(&out).unwrap();
    let mut args = vec!["stage".to_string(), s(&out)];
    for (stem, pass) in fields(field(&doc, "passes")) {
        let path = dir.join(format!("{stem}.json"));
        fs::write(&path, json_text(pass)).unwrap();
        args.push(s(&path));
    }
    for key in ["gemm", "memory", "layout", "ledger", "alloc", "ple"] {
        if let Some(v) = doc.get(key) {
            let path = dir.join(format!("{key}.json"));
            fs::write(&path, json_text(v)).unwrap();
            args.push(format!("--{key}"));
            args.push(s(&path));
        }
    }
    let contended = items(field(&doc, "contended_passes"));
    assert!(!contended.is_empty(), "the committed profile excludes a contended pass");
    args.push("--contended".to_string());
    args.extend(contended.iter().map(|c| text_of(c).to_string()));
    let r = run(&args);
    assert_eq!(r.code, Some(0), "{}", r.err);
    assert_bytes_equal(
        "stage json",
        &out.join("embedgemma2_stage_profile_m5pro.json"),
        "bench/results/embedgemma2_stage_profile_m5pro.json",
    );
    // The committed write-up quotes these tables verbatim.
    let txt = read(&manifest("bench/results/embedgemma2_stage_profile_m5pro.txt"));
    assert!(
        txt.contains(r.out.trim_end()),
        "the stage tables are not the ones the write-up quotes"
    );
    assert_eq!(read(&out.join("embedgemma2_stage_profile_tables.txt")), r.out);
}

#[test]
fn the_committed_vision_profile_is_rebuilt_from_its_own_passes() {
    let doc = parse(&read(&manifest("bench/results/embedgemma2_vision_profile_m5pro.json")));
    let dir = fresh("vision_rebuild");
    let out = dir.join("out");
    fs::create_dir_all(&out).unwrap();
    let mut args = vec!["vision".to_string(), s(&out)];
    for (label, passes) in fields(field(&doc, "runs")) {
        for (k, pass) in items(passes).iter().enumerate() {
            let path = dir.join(format!("{label}_{k}.json"));
            fs::write(&path, json_text(pass)).unwrap();
            args.push(format!("{label}={}", s(&path)));
        }
    }
    let r = run(&args);
    assert_eq!(r.code, Some(0), "{}", r.err);
    assert_bytes_equal(
        "vision json",
        &out.join("embedgemma2_vision_profile_m5pro.json"),
        "bench/results/embedgemma2_vision_profile_m5pro.json",
    );
    let txt = read(&manifest("bench/results/embedgemma2_vision_profile_m5pro.txt"));
    assert!(
        txt.contains(r.out.trim_end()),
        "the vision tables are not the ones the write-up quotes"
    );
}

#[test]
fn the_committed_text_fusion_profile_is_rebuilt_from_its_own_passes() {
    let doc = parse(&read(&manifest(
        "bench/results/embedgemma2_text_fusion_profile_m5pro.json",
    )));
    let dir = fresh("fusion_rebuild");
    let out = dir.join("out");
    fs::create_dir_all(&out).unwrap();
    let mut args = vec!["text-fusion".to_string(), s(&out)];
    for (label, passes) in fields(field(&doc, "passes")) {
        for (k, pass) in items(passes).iter().enumerate() {
            let path = dir.join(format!("{}_{k}.json", label.replace(' ', "_")));
            fs::write(&path, json_text(pass)).unwrap();
            args.push(format!("{label}={}", s(&path)));
        }
    }
    let r = run(&args);
    assert_eq!(r.code, Some(0), "{}", r.err);
    assert_bytes_equal(
        "text-fusion json",
        &out.join("embedgemma2_text_fusion_profile_m5pro.json"),
        "bench/results/embedgemma2_text_fusion_profile_m5pro.json",
    );
    assert_bytes_equal(
        "text-fusion tables",
        &out.join("embedgemma2_text_fusion_profile_tables.txt"),
        "bench/results/embedgemma2_text_fusion_profile_tables.txt",
    );
    // The tables go to stdout and to the file alike.
    assert_eq!(read(&out.join("embedgemma2_text_fusion_profile_tables.txt")), r.out);
}

#[test]
fn the_committed_small_m_json_is_rebuilt_from_its_own_rows() {
    let committed = "bench/results/embedgemma2_small_m_m5pro.json";
    let doc = parse(&read(&manifest(committed)));
    let lanes = field(&doc, "lane_names");
    let (before, after, control) = (
        text_of(field(lanes, "before")).to_string(),
        text_of(field(lanes, "after")).to_string(),
        text_of(field(lanes, "control")).to_string(),
    );
    let dir = fresh("small_m_rebuild");
    let raw = dir.join("raw");
    let out = dir.join("out");
    fs::create_dir_all(&raw).unwrap();
    fs::create_dir_all(&out).unwrap();
    // Which paired run each workload came from (`lat`, `thr64`, `thr8`).
    let rows = items(field(field(field(&doc, "encode"), "f32"), "summary"));
    let group_of = |w: &str| match w {
        "1x16" | "1x64" => "lat",
        "64x32" => "thr64",
        "8x1024" => "thr8",
        other => panic!("unexpected workload {other}"),
    };
    for grp in ["lat", "thr64", "thr8"] {
        let mine: Vec<Json> = rows
            .iter()
            .filter(|r| group_of(text_of(field(r, "workload"))) == grp)
            .cloned()
            .collect();
        let env = field(field(&doc, "env"), &format!("{grp}_f32")).clone();
        let file = Json::Object(vec![
            ("summary".to_string(), Json::Array(mine)),
            ("env".to_string(), env),
        ]);
        fs::write(raw.join(format!("paired_{grp}_f32.json")), json_text(&file)).unwrap();
    }
    // The `nnsmall` stdout, laid out the way bench_gemm_routes prints it.
    let mut text = String::new();
    for case in items(field(field(&doc, "gemm"), "f32")) {
        text.push_str(&format!(
            "nnsmall f32 {} M={} N={} K={}\n",
            text_of(field(case, "name")),
            json_text(field(case, "m")),
            json_text(field(case, "n")),
            json_text(field(case, "k"))
        ));
        let ratios = field(case, "ratios");
        for (arm, us) in fields(field(case, "arms")) {
            let r = items(field(ratios, arm));
            let num = |j: &Json| match j {
                Json::Num { value, .. } => *value,
                other => panic!("{other:?}"),
            };
            text.push_str(&format!(
                "  {arm} {:.1} us  plain/this {:.2} [{:.2}\u{2013}{:.2}]\n",
                num(us),
                num(&r[0]),
                num(&r[1]),
                num(&r[2])
            ));
        }
    }
    fs::write(raw.join("nnsmall_f32.txt"), text).unwrap();
    let r = run(&["small-m".to_string(), s(&out), s(&raw), before, after, control]);
    assert_eq!(r.code, Some(0), "{}", r.err);
    assert_bytes_equal("small-m json", &out.join("embedgemma2_small_m_m5pro.json"), committed);
    // The write-up quotes the whole-encode table.
    let txt = read(&manifest("bench/results/embedgemma2_small_m_m5pro.txt"));
    let table: Vec<&str> = r.out.lines().take_while(|l| !l.is_empty()).collect();
    assert!(table.len() >= 5, "{}", r.out);
    assert!(
        txt.contains(&table.join("\n")),
        "the whole-encode table is not the one the write-up quotes"
    );
}

// ---------------------------------------------------------------------------
// A small run whose numbers can be checked by hand.
// ---------------------------------------------------------------------------

fn hand_checked_small_m(dir: &Path, before: &str) -> Run {
    let raw = dir.join("raw");
    let out = dir.join("out");
    fs::create_dir_all(&raw).unwrap();
    fs::create_dir_all(&out).unwrap();
    let lane = |xs: &str| format!("{{\"ms_min_all\":{xs}}}");
    for (grp, rows) in [
        (
            "lat",
            format!(
                "[{{\"workload\":\"w1\",\"b\":{},\"a\":{},\"c\":{}}}]",
                lane(before),
                lane("[1,3]"),
                lane("[2,4]")
            ),
        ),
        ("thr64", "[]".to_string()),
        ("thr8", "[]".to_string()),
    ] {
        fs::write(
            raw.join(format!("paired_{grp}_f32.json")),
            format!("{{\"summary\":{rows}}}"),
        )
        .unwrap();
    }
    fs::write(
        raw.join("nnsmall_f32.txt"),
        "nnsmall f32 down M=16 N=512 K=2048\n  plain 90.0 us  plain/this 1.00 [1.00\u{2013}1.00]\n  production 30.0 us  plain/this 3.00 [2.50\u{2013}3.50]\n",
    )
    .unwrap();
    run(&[
        "small-m".to_string(),
        s(&out),
        s(&raw),
        "b".to_string(),
        "a".to_string(),
        "c".to_string(),
    ])
}

#[test]
fn a_small_m_report_with_numbers_checked_by_hand() {
    let dir = fresh("small_m_hand");
    let r = hand_checked_small_m(&dir, "[2,4]");
    assert_eq!(r.code, Some(0), "{}", r.err);
    // before [2,4], after [1,3], control [2,4]: round ratios 0.5 and 0.75,
    // median 0.625; control ratios 1 and 1, so both rounds are clean.
    let want = "WHOLE ENCODE: after / before (ms = min of rounds' min-of-N)\n\
prec  workload rounds    before     after  min-of-rounds     median ratio [lo,hi]        control [lo,hi]  clean n  clean median\n\
f32   w1            2     2.000     1.000          0.500    0.625 [0.50,0.75]    1.000 [1.00,1.00]        2         0.625\n\
\n\
ONE GEMM, us per call (median of interleaved rounds); plain / this = median of per-round ratios [lo, hi]\n\
f32   down           M=16  K=2048  N=512   plain 90.0  production 30.0 (3.00x [2.50,3.50])\n";
    assert_eq!(r.out, want);
    let json = read(&dir.join("out/embedgemma2_small_m_m5pro.json"));
    // Integers stay integers (the min of [2, 4]) and ratios are floats; no trailing newline.
    assert!(json.contains("\"before_min\": 2,"), "{json}");
    assert!(json.contains("\"ratio_median\": 0.625,"), "{json}");
    assert!(json.contains("\"before_median\": 3.0,"), "{json}");
    assert!(json.ends_with("}") && !json.ends_with("\n"));
}

#[test]
fn a_zero_in_a_denominator_is_an_error_not_an_infinity() {
    let dir = fresh("small_m_zero");
    let r = hand_checked_small_m(&dir, "[0,4]");
    assert_eq!(r.code, Some(1), "{}", r.out);
    assert!(r.err.contains("division by zero"), "{}", r.err);
    assert!(
        !dir.join("out/embedgemma2_small_m_m5pro.json").exists(),
        "a failed run left a result file"
    );
}

// ---------------------------------------------------------------------------
// Bad invocations and bad inputs fail loudly.
// ---------------------------------------------------------------------------

#[test]
fn bad_invocations_are_refused() {
    let none: [&str; 0] = [];
    assert_eq!(run(&none).code, Some(1));
    let r = run(&["frobnicate"]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("unknown report"), "{}", r.err);
    for sub in ["memory", "small-m", "stage", "vision", "text-fusion"] {
        let r = run(&[sub]);
        assert_eq!(r.code, Some(1), "{sub}");
        assert!(r.err.starts_with("results_report: "), "{sub}: {}", r.err);
    }
    // A pass argument without `=`, where the script split on it.
    let dir = fresh("no_equals");
    let r = run(&["text-fusion".to_string(), s(&dir), "just-a-path.json".to_string()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("LABEL=FILE"), "{}", r.err);
    // The stage tool wants an output directory and at least one pass.
    let r = run(&["stage".to_string(), s(&dir)]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("passes"), "{}", r.err);
    let r = run(&[
        "stage".to_string(),
        s(&dir),
        "p.json".to_string(),
        "--bogus".to_string(),
        "x".to_string(),
    ]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("--bogus"), "{}", r.err);
}

#[test]
fn missing_unreadable_and_malformed_inputs_name_the_file() {
    let dir = fresh("bad_inputs");
    let r = run(&["memory".to_string(), s(&dir), s(&dir)]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("bounds.txt"), "{}", r.err);

    let bad = dir.join("bad.json");
    fs::write(&bad, b"{\"workloads\": [\xff]}").unwrap();
    let r = run(&["vision".to_string(), s(&dir), format!("x={}", s(&bad))]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("not UTF-8"), "{}", r.err);

    fs::write(&bad, "{\"workloads\": [}").unwrap();
    let r = run(&["vision".to_string(), s(&dir), format!("x={}", s(&bad))]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("bad.json"), "{}", r.err);

    // A duplicate key, which Python would silently resolve, is refused.
    fs::write(&bad, "{\"workloads\": [], \"workloads\": []}").unwrap();
    let r = run(&["vision".to_string(), s(&dir), format!("x={}", s(&bad))]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("duplicate key"), "{}", r.err);

    // An integer this tool cannot hold exactly is refused, never rounded.
    fs::write(
        &bad,
        "{\"workloads\": [], \"big\": 1000000000000000000000000000000000000000}",
    )
    .unwrap();
    let r = run(&["vision".to_string(), s(&dir), format!("x={}", s(&bad))]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("too large"), "{}", r.err);

    // A document of the wrong shape is a named missing key.
    fs::write(&bad, "{\"nothing\": 1}").unwrap();
    let r = run(&["vision".to_string(), s(&dir), format!("x={}", s(&bad))]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("workloads"), "{}", r.err);
}
