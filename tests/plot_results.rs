//! `src/bin/plot_results` draws the benchmark figures as SVG with no plotting
//! dependency. These tests run every figure against the committed result files
//! and hold it to what a reader of the figure would rely on:
//!
//! * the SVG is well formed (a small tag-balance and attribute checker below, not
//!   a trust in the writer), every figure has a background and a title;
//! * the marks drawn are the data's: bar, dot, whisker and legend-key counts equal
//!   what the JSON holds (a bar or a series dropped by a layout change fails);
//! * the same data is the same bytes, from the library path and from the binary;
//! * the axis arithmetic (log and linear ticks at awkward ranges, number
//!   spelling) and the XML escaping are right on their own;
//! * every refusal the figure's script had (lanes that disagree on the token ids,
//!   an itemisation that does not add up, an unknown stage class, a missing key,
//!   empty data, a time that cannot go on a log axis) still stops it, writes
//!   nothing, and names the reason.
//!
//! Nothing here needs a GPU or Python.

// The bin's own module list, so this test compiles every figure the registry names.
include!("../src/bin/plot_results/modules.rs");

use chart::{fmt_tick, linear_ticks, log_ticks, nice_ceil, nice_floor, nice_top};
use json::{Json, Syntax};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn manifest(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn results(file: &str) -> PathBuf {
    manifest("bench/results").join(file)
}

fn fresh(case: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("plot_results").join(case);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

struct Run {
    code: Option<i32>,
    out: String,
    err: String,
}

fn run<S: AsRef<str>>(args: &[S]) -> Run {
    let out = Command::new(env!("CARGO_BIN_EXE_plot_results"))
        .args(args.iter().map(|a| a.as_ref()))
        .output()
        .unwrap();
    Run {
        code: out.status.code(),
        out: String::from_utf8(out.stdout).unwrap(),
        err: String::from_utf8(out.stderr).unwrap(),
    }
}

const SYNTAX: Syntax = Syntax {
    what: "test JSON",
    max_depth: 64,
    uints_only: false,
    literals: true,
};

fn parse(file: &str) -> Json {
    json::parse(&fs::read_to_string(results(file)).unwrap(), SYNTAX).unwrap()
}

/// `j.a.b[2].c`-style lookup into a parsed document.
fn at<'a>(j: &'a Json, path: &str) -> &'a Json {
    let mut cur = j;
    for part in path.split('.') {
        let (key, idx) = match part.split_once('[') {
            Some((k, rest)) => (k, Some(rest.trim_end_matches(']').parse::<usize>().unwrap())),
            None => (part, None),
        };
        if !key.is_empty() {
            cur = cur.get(key).unwrap_or_else(|| panic!("no key {key} in {path}"));
        }
        if let Some(i) = idx {
            cur = match cur {
                Json::Array(a) => &a[i],
                _ => panic!("{path}: not an array at {part}"),
            };
        }
    }
    cur
}

fn arr(j: &Json) -> &Vec<Json> {
    match j {
        Json::Array(a) => a,
        _ => panic!("not an array"),
    }
}

fn fields(j: &Json) -> &Vec<(String, Json)> {
    match j {
        Json::Object(f) => f,
        _ => panic!("not an object"),
    }
}

fn s(j: &Json) -> &str {
    match j {
        Json::Str(s) => s,
        _ => panic!("not a string"),
    }
}

fn n(j: &Json) -> f64 {
    match j {
        Json::Num { value, .. } => *value,
        _ => panic!("not a number"),
    }
}

/// A parsed element: its name and attributes.
#[derive(Debug)]
struct Elem {
    name: String,
    attrs: Vec<(String, String)>,
}

impl Elem {
    fn attr(&self, k: &str) -> Option<&str> {
        self.attrs.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
    }

    fn has_class(&self, class: &str) -> bool {
        self.attr("class").is_some_and(|c| c.split(' ').any(|t| t == class))
    }
}

/// A well-formedness check small enough to read: tags balance, names match,
/// attributes are quoted and unique within a tag, text and values hold no bare
/// `<` or `&` (only the five predefined entities and numeric references), no
/// character XML 1.0 forbids, one `svg` root and nothing after it. Returns the
/// elements in document order.
fn check_xml(doc: &str) -> Result<Vec<Elem>, String> {
    for c in doc.chars() {
        if (c as u32) < 0x20 && c != '\n' && c != '\t' && c != '\r'
            || c == '\u{7f}'
            || c == '\u{fffe}'
            || c == '\u{ffff}'
        {
            return Err(format!("forbidden character U+{:04X}", c as u32));
        }
    }
    let b = doc.as_bytes();
    let mut i = 0;
    let mut stack: Vec<String> = Vec::new();
    let mut elems = Vec::new();
    let mut roots = 0;
    let entity_ok = |text: &str| -> Result<(), String> {
        let mut rest = text;
        while let Some(p) = rest.find('&') {
            let tail = &rest[p..];
            let end = tail.find(';').ok_or_else(|| format!("bare & in {text:?}"))?;
            let ent = &tail[1..end];
            let ok = matches!(ent, "amp" | "lt" | "gt" | "quot" | "apos")
                || (ent.starts_with('#') && ent.len() > 1 && ent[1..].chars().all(|c| c.is_ascii_digit()));
            if !ok {
                return Err(format!("unknown entity &{ent}; in {text:?}"));
            }
            rest = &tail[end + 1..];
        }
        Ok(())
    };
    while i < b.len() {
        if b[i] != b'<' {
            let start = i;
            while i < b.len() && b[i] != b'<' {
                i += 1;
            }
            let text = &doc[start..i];
            if stack.is_empty() && !text.trim().is_empty() {
                return Err(format!("text outside the root: {text:?}"));
            }
            if text.contains('>') {
                return Err(format!("bare > in text {text:?}"));
            }
            entity_ok(text)?;
            continue;
        }
        if doc[i..].starts_with("<?") {
            i += doc[i..].find("?>").ok_or("unterminated <?")? + 2;
            continue;
        }
        if doc[i..].starts_with("<!--") {
            i += doc[i..].find("-->").ok_or("unterminated comment")? + 3;
            continue;
        }
        if doc[i..].starts_with("<!") {
            return Err("DOCTYPE, CDATA and other <! constructs are not used".to_string());
        }
        if doc[i..].starts_with("</") {
            let end = doc[i..].find('>').ok_or("unterminated end tag")? + i;
            let name = doc[i + 2..end].trim();
            match stack.pop() {
                Some(open) if open == name => {}
                Some(open) => return Err(format!("</{name}> closes <{open}>")),
                None => return Err(format!("</{name}> with nothing open")),
            }
            i = end + 1;
            continue;
        }
        // An open or empty-element tag.
        i += 1;
        let ns = i;
        while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'-' || b[i] == b':' || b[i] == b'_') {
            i += 1;
        }
        let name = doc[ns..i].to_string();
        if name.is_empty() {
            return Err(format!("empty tag name at byte {ns}"));
        }
        let mut attrs: Vec<(String, String)> = Vec::new();
        loop {
            while i < b.len() && b[i].is_ascii_whitespace() {
                i += 1;
            }
            match b.get(i) {
                None => return Err(format!("unterminated <{name}")),
                Some(b'/') => {
                    if b.get(i + 1) != Some(&b'>') {
                        return Err(format!("stray / in <{name}"));
                    }
                    i += 2;
                    if stack.is_empty() {
                        roots += 1;
                    }
                    elems.push(Elem { name, attrs });
                    break;
                }
                Some(b'>') => {
                    i += 1;
                    if stack.is_empty() {
                        roots += 1;
                    }
                    stack.push(name.clone());
                    elems.push(Elem { name, attrs });
                    break;
                }
                Some(_) => {
                    let ks = i;
                    while i < b.len()
                        && (b[i].is_ascii_alphanumeric()
                            || b[i] == b'-'
                            || b[i] == b':'
                            || b[i] == b'_'
                            || b[i] == b'.')
                    {
                        i += 1;
                    }
                    let key = doc[ks..i].to_string();
                    if key.is_empty() || b.get(i) != Some(&b'=') {
                        return Err(format!("bad attribute in <{name} at byte {ks}"));
                    }
                    i += 1;
                    let q = *b.get(i).ok_or("unterminated attribute")?;
                    if q != b'"' && q != b'\'' {
                        return Err(format!("unquoted attribute {key} in <{name}"));
                    }
                    let vs = i + 1;
                    let ve = doc[vs..].find(q as char).ok_or("unterminated attribute value")? + vs;
                    let val = &doc[vs..ve];
                    if val.contains('<') {
                        return Err(format!("< in attribute {key}"));
                    }
                    entity_ok(val)?;
                    if attrs.iter().any(|(k, _)| *k == key) {
                        return Err(format!("duplicate attribute {key} in <{name}"));
                    }
                    attrs.push((key, val.to_string()));
                    i = ve + 1;
                }
            }
        }
    }
    if let Some(open) = stack.last() {
        return Err(format!("<{open}> is never closed"));
    }
    if roots != 1 || elems.first().map(|e| e.name.as_str()) != Some("svg") {
        return Err("expected exactly one svg root".to_string());
    }
    Ok(elems)
}

fn count(elems: &[Elem], class: &str) -> usize {
    elems.iter().filter(|e| e.has_class(class)).count()
}

/// Render a figure in-process, from the committed file.
fn render(name: &str) -> registry::Output {
    let plot = registry::find(name).unwrap_or_else(|| panic!("no figure {name}"));
    (plot.render)(&results(plot.file)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn elems(name: &str) -> Vec<Elem> {
    let out = render(name);
    check_xml(&out.svg).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// A directory holding copies of the named result files.
fn copy_of(case: &str, files: &[&str]) -> PathBuf {
    let dir = fresh(case);
    for f in files {
        fs::copy(results(f), dir.join(f)).unwrap();
    }
    dir
}

/// Replace `from` by `to` in a copied file; the replacement must have happened.
fn tamper(path: &Path, from: &str, to: &str, all: bool) {
    let text = fs::read_to_string(path).unwrap();
    assert!(text.contains(from), "{} does not contain {from:?}", path.display());
    let new = if all {
        text.replace(from, to)
    } else {
        text.replacen(from, to, 1)
    };
    fs::write(path, new).unwrap();
}

fn no_svg_in(dir: &Path) -> bool {
    fs::read_dir(dir)
        .unwrap()
        .all(|e| e.unwrap().path().extension().is_none_or(|x| x != "svg"))
}

const ALL: &[(&str, &str)] = &[
    ("final", "embedgemma2_final_m5pro"),
    ("integrate-text", "embedgemma2_integrate_text_m5pro"),
    ("integrate-mm", "embedgemma2_integrate_mm_m5pro"),
    ("vs-torch", "embedgemma2_vs_torch_m5pro"),
    ("audio-speed", "embedgemma2_audio_speed_m5pro"),
    ("mm-onewait", "embedgemma2_mm_onewait_m5pro"),
    ("pre", "embedgemma2_pre_m5pro"),
    ("vision-speed", "embedgemma2_vision_speed_m5pro"),
    ("audio-profile", "embedgemma2_audio_profile_m5pro"),
    ("stage-profile", "embedgemma2_stage_profile_m5pro"),
    ("text-fusion", "embedgemma2_text_fusion_profile_m5pro"),
    ("memory", "embedgemma2_memory_m5pro"),
    ("small-m", "embedgemma2_small_m_m5pro"),
    ("tok", "embedgemma2_tok_m5pro"),
    ("video", "embedgemma2_video_m5pro"),
    ("media-decode", "media_decode_m5pro"),
    ("media-resample", "media_resample_m5pro"),
    ("video-color", "media_video_color_m5pro"),
    ("jpeg-decode", "jpeg_decode_m5pro"),
    ("go-overhead", "embedgemma2_go_m5pro"),
];

// ---------------------------------------------------------------------------
// Every figure renders every committed result file
// ---------------------------------------------------------------------------

#[test]
fn the_registry_holds_every_figure_once() {
    let mut names: Vec<&str> = registry::PLOTS.iter().map(|p| p.name).collect();
    let listed: Vec<&str> = ALL.iter().map(|(n, _)| *n).collect();
    names.sort();
    let mut want = listed.clone();
    want.sort();
    assert_eq!(names, want);
    for (name, stem) in ALL {
        let plot = registry::find(name).unwrap();
        assert_eq!(plot.file, format!("{stem}.json"), "{name}");
        assert!(results(plot.file).is_file(), "{} is not committed", plot.file);
    }
}

#[test]
fn every_figure_is_a_well_formed_svg_with_a_background_a_title_and_a_dark_theme() {
    for (name, _) in ALL {
        let out = render(name);
        let elems = check_xml(&out.svg).unwrap_or_else(|e| panic!("{name}: {e}"));
        let root = &elems[0];
        let (w, h) = (root.attr("width").unwrap(), root.attr("height").unwrap());
        assert!(
            w.parse::<f64>().unwrap() > 300.0 && h.parse::<f64>().unwrap() > 200.0,
            "{name}: {w}x{h}"
        );
        assert_eq!(root.attr("viewBox"), Some(format!("0 0 {w} {h}").as_str()), "{name}");
        assert!(elems.iter().any(|e| e.name == "title"), "{name}: no <title>");
        let bg = elems
            .iter()
            .find(|e| e.has_class("bg"))
            .unwrap_or_else(|| panic!("{name}: no background"));
        assert_eq!(bg.name, "rect");
        assert!(out.svg.contains("prefers-color-scheme:dark"), "{name}: no dark theme");
        assert!(
            out.svg.contains(".bg{fill:#fff}"),
            "{name}: the light background is not white"
        );
        assert!(
            !out.svg.contains("NaN") && !out.svg.contains("inf"),
            "{name}: a non-finite number reached the SVG"
        );
        assert!(count(&elems, "panel") >= 1, "{name}: no panel");
        // The ink colour is a class the dark theme redefines; written as a
        // literal it would vanish on the dark background.
        for e in &elems {
            for attr in ["stroke", "fill"] {
                assert!(
                    e.attr(attr) != Some("#0b0b0b"),
                    "{name}: a <{}> has {attr}=\"#0b0b0b\" instead of the ink class",
                    e.name
                );
            }
        }
    }
}

#[test]
fn the_same_data_is_the_same_bytes_from_the_library_and_from_the_binary() {
    let out_a = fresh("determinism_a");
    let out_b = fresh("determinism_b");
    for (name, stem) in ALL {
        let first = render(name);
        let second = render(name);
        assert_eq!(first.svg, second.svg, "{name}: two renders differ");
        assert_eq!(first.table, second.table, "{name}: the table differs");
        let json = results(&format!("{stem}.json"));
        for dir in [&out_a, &out_b] {
            let r = run(&[name, json.to_str().unwrap(), "--out-dir", dir.to_str().unwrap()]);
            assert_eq!(r.code, Some(0), "{name}: {}", r.err);
            assert!(
                r.out
                    .contains(&format!("wrote {}", dir.join(format!("{stem}.svg")).display())),
                "{name}: {}",
                r.out
            );
        }
        let (a, b) = (
            fs::read(out_a.join(format!("{stem}.svg"))).unwrap(),
            fs::read(out_b.join(format!("{stem}.svg"))).unwrap(),
        );
        assert_eq!(a, b, "{name}: two runs of the binary differ");
        assert_eq!(a, first.svg.as_bytes(), "{name}: the binary and the library differ");
    }
}

#[test]
fn the_committed_svgs_are_what_the_tool_draws_from_the_committed_json() {
    for (name, stem) in ALL {
        let committed = fs::read_to_string(results(&format!("{stem}.svg")))
            .unwrap_or_else(|e| panic!("{stem}.svg is not committed: {e}"));
        assert!(
            committed == render(name).svg,
            "{stem}.svg is stale: run `cargo run --release --bin plot_results -- all bench/results`"
        );
    }
}

#[test]
fn all_draws_every_figure_and_writes_beside_the_inputs_by_default() {
    let files: Vec<String> = registry::PLOTS.iter().map(|p| p.file.to_string()).collect();
    let mut want: Vec<&str> = files.iter().map(String::as_str).collect();
    want.extend([
        "embedgemma2_tok_m5pro_tessl_lane.json",
        "embedgemma2_tok_m5pro_hf_lane.json",
        "embedgemma2_vision_profile_m5pro.json",
    ]);
    let dir = copy_of("all_default", &want);
    let r = run(&["all", dir.to_str().unwrap()]);
    assert_eq!(r.code, Some(0), "{}", r.err);
    for (_, stem) in ALL {
        let svg = fs::read_to_string(dir.join(format!("{stem}.svg"))).unwrap();
        check_xml(&svg).unwrap();
    }
    assert!(
        r.out.contains("GEMMs (exact f32)"),
        "the audio table is printed: {}",
        r.out
    );
}

// ---------------------------------------------------------------------------
// The marks are the data's
// ---------------------------------------------------------------------------

fn lane_names(doc: &Json) -> Vec<String> {
    arr(at(doc, "lanes"))
        .iter()
        .map(|l| s(at(l, "name")).to_string())
        .collect()
}

fn rounds_len(row: &Json, lane: &str) -> usize {
    arr(at(row, &format!("{lane}.ms_min_all"))).len()
}

#[test]
fn final_draws_a_bar_per_lane_and_workload_in_each_panel_and_a_whisker_per_speedup() {
    let doc = parse("embedgemma2_final_m5pro.json");
    let rows = arr(at(&doc, "summary")).len();
    let lanes = lane_names(&doc).len();
    assert_eq!(
        (rows, lanes),
        (8, 4),
        "the committed file changed shape; update the expectation"
    );
    let e = elems("final");
    assert_eq!(count(&e, "bar"), rows * lanes + rows * (lanes - 1) + rows * lanes);
    assert_eq!(count(&e, "whisker"), rows * (lanes - 1));
    assert_eq!(count(&e, "key"), lanes);
    assert_eq!(count(&e, "panel"), 3);
}

#[test]
fn audio_speed_draws_every_round_of_every_lane_and_a_median_per_lane() {
    let doc = parse("embedgemma2_audio_speed_m5pro.json");
    let lanes = lane_names(&doc);
    let mut dots = 0;
    for row in arr(at(&doc, "summary")) {
        for l in &lanes {
            dots += rounds_len(row, l);
        }
    }
    let rows = arr(at(&doc, "summary")).len();
    let e = elems("audio-speed");
    assert_eq!(count(&e, "dot"), dots);
    assert_eq!(count(&e, "median"), rows * lanes.len());
    assert_eq!(count(&e, "key"), lanes.len());
    assert_eq!(count(&e, "panel"), rows);
}

#[test]
fn mm_onewait_draws_every_round_and_a_peak_bar_per_lane() {
    let doc = parse("embedgemma2_mm_onewait_m5pro.json");
    let lanes = lane_names(&doc);
    let rows = arr(at(&doc, "summary"));
    let dots: usize = rows
        .iter()
        .map(|r| lanes.iter().map(|l| rounds_len(r, l)).sum::<usize>())
        .sum();
    let e = elems("mm-onewait");
    assert_eq!(count(&e, "dot"), dots);
    assert_eq!(count(&e, "bar"), rows.len() * lanes.len());
    assert_eq!(count(&e, "median"), rows.len() * lanes.len());
    assert_eq!(count(&e, "key"), lanes.len());
    assert_eq!(count(&e, "panel"), rows.len() * 2);
}

#[test]
fn pre_draws_a_bar_per_lane_and_workload_and_every_round() {
    let doc = parse("embedgemma2_pre_m5pro.json");
    let rows = arr(at(&doc, "summary"));
    let dots: usize = rows.iter().map(|r| rounds_len(r, "tessl") + rounds_len(r, "hf")).sum();
    let e = elems("pre");
    assert_eq!(count(&e, "bar"), rows.len() * 2);
    assert_eq!(count(&e, "dot"), dots);
    assert_eq!(count(&e, "median"), rows.len() * 2);
    assert_eq!(count(&e, "key"), 2);
}

#[test]
fn vision_speed_draws_the_lanes_the_ratios_and_every_profile_stage() {
    let speed = parse("embedgemma2_vision_speed_m5pro.json");
    let profile = parse("embedgemma2_vision_profile_m5pro.json");
    let lanes = lane_names(&speed);
    let rows = arr(at(&speed, "summary"));
    let dots: usize = rows
        .iter()
        .map(|r| lanes.iter().map(|l| rounds_len(r, l)).sum::<usize>())
        .sum();
    let labels = fields(at(&profile, "runs")).len();
    let e = elems("vision-speed");
    let bars = count(&e, "bar");
    let first_two = rows.len() * lanes.len() + rows.len() * (lanes.len() - 1);
    assert!(bars > first_two, "the stage panel drew no bars");
    // One bar per (profile label, stage) with a time above zero: at most labels x stages, at least one per stage.
    let stages = {
        let mut names: Vec<String> = Vec::new();
        for (_, runs) in fields(at(&profile, "runs")) {
            for doc in arr(runs) {
                for c in arr(at(doc, "classes")) {
                    if !names.contains(&s(c).to_string()) {
                        names.push(s(c).to_string());
                    }
                }
            }
        }
        names.len()
    };
    assert!(
        bars - first_two <= labels * stages && bars - first_two >= stages - 2,
        "{bars} bars, {stages} stages"
    );
    assert_eq!(count(&e, "dot"), dots);
    assert_eq!(count(&e, "whisker"), rows.len() * (lanes.len() - 1));
    assert_eq!(count(&e, "key"), lanes.len() + (lanes.len() - 1) + labels);
}

#[test]
fn audio_profile_draws_four_segments_per_side_and_prints_the_table_the_results_file_quotes() {
    let doc = parse("embedgemma2_audio_profile_m5pro.json");
    let wl = arr(at(&doc, "before.workloads")).len();
    let out = render("audio-profile");
    let e = check_xml(&out.svg).unwrap();
    assert_eq!(count(&e, "seg"), wl * 2 * 4);
    assert_eq!(count(&e, "key"), 4);
    // The results file quotes this table; it must be the lines of the printed one.
    let txt = fs::read_to_string(results("embedgemma2_audio_profile_m5pro.txt")).unwrap();
    let lines: Vec<&str> = out.table.lines().collect();
    assert_eq!(lines.len(), 1 + wl * 2);
    let at_header = txt
        .lines()
        .position(|l| l == lines[0])
        .expect("the header line is quoted in the .txt");
    for (k, line) in lines.iter().enumerate() {
        assert_eq!(txt.lines().nth(at_header + k), Some(*line), "table line {k}");
    }
}

#[test]
fn stage_profile_draws_four_groups_per_workload_in_two_stacked_panels() {
    let doc = parse("embedgemma2_stage_profile_m5pro.json");
    let contended: Vec<&str> = arr(at(&doc, "contended_passes")).iter().map(s).collect();
    let mut wls: Vec<&str> = Vec::new();
    for (name, pass) in fields(at(&doc, "passes")) {
        if contended.contains(&name.as_str()) {
            continue;
        }
        for w in arr(at(pass, "workloads")) {
            if !wls.contains(&s(at(w, "workload"))) {
                wls.push(s(at(w, "workload")));
            }
        }
    }
    let e = elems("stage-profile");
    assert_eq!(count(&e, "seg"), wls.len() * 4 * 2);
    assert_eq!(count(&e, "marker"), wls.len(), "one whole-forward tick per workload");
    // Throughput bars: two per GEMM shape at 8192 rows; memory: five.
    let shapes = arr(at(&doc, "gemm.gemm_shapes"))
        .iter()
        .filter(|r| n(at(r, "rows")) == 8192.0)
        .count();
    assert_eq!(count(&e, "bar"), shapes * 2 + 5);
    assert_eq!(count(&e, "key"), 4 + 1 + 4);
}

#[test]
fn text_fusion_draws_a_segment_per_category_and_label_and_a_dispatch_bar_per_label() {
    let doc = parse("embedgemma2_text_fusion_profile_m5pro.json");
    let cats = arr(at(&doc, "categories")).len();
    let labels = fields(at(&doc, "runs")).len();
    let e = elems("text-fusion");
    assert_eq!(count(&e, "seg"), 3 * labels * cats);
    assert_eq!(count(&e, "bar"), 3 * labels);
    assert_eq!(count(&e, "key"), cats);
}

#[test]
fn memory_draws_the_peaks_the_waterfall_and_every_round_that_is_on_scale() {
    let doc = parse("embedgemma2_memory_m5pro.json");
    let out = render("memory");
    let e = check_xml(&out.svg).unwrap();
    // Row 1: five workloads, before and after, in two modes; the waterfall has four bars.
    // The sweep panel has one bar per (mode, workload) of the two modes' shared workloads.
    let sweep_points = {
        let mut pts = 0;
        for (name, sweep) in fields(at(&doc, "ple_sweeps")) {
            if name.starts_with("f32") && !name.contains("first run") && !name.contains("repeat") {
                pts += fields(at(sweep, "vs_ple256.ple16")).len();
            }
        }
        pts
    };
    assert_eq!(count(&e, "bar"), 5 * 2 * 2 + 4 + sweep_points * 2);
    // Ticks: one per workload and mode. Diamonds: one per sweep point and mode, and one per workload of
    // each mode's paired series that has its min-of-N ratio on the scale.
    assert!(
        count(&e, "marker") >= 5 * 2 + sweep_points * 2,
        "{} markers",
        count(&e, "marker")
    );
    // Row 2: a dot per round that is on the 0.5..2 scale, and the panel says how many were not.
    let mut total = 0usize;
    let mut off = 0usize;
    for (at_, _) in out.svg.match_indices(" rounds off scale") {
        let head = &out.svg[..at_];
        let text = &head[head.rfind('>').unwrap() + 1..];
        let (o, t) = text.split_once(" of ").unwrap();
        off += o.parse::<usize>().unwrap();
        total += t.parse::<usize>().unwrap();
    }
    let mut want_total = 0;
    for ops in ["f32", "bf16"] {
        for (_, series) in fields(at(&doc, &format!("paired.{ops}.series_ms.before"))) {
            want_total += arr(series).len();
        }
    }
    assert_eq!(total, want_total);
    assert_eq!(count(&e, "dot"), total - off);
    assert_eq!(count(&e, "key"), 3 + 3);
}

#[test]
fn small_m_draws_every_round_a_control_cross_and_a_pair_of_bars_per_gemm() {
    let doc = parse("embedgemma2_small_m_m5pro.json");
    let before = s(at(&doc, "lane_names.before")).to_string();
    let mut rounds = 0;
    let mut workloads = 0;
    for (_, paired) in fields(at(&doc, "encode")) {
        for row in arr(at(paired, "summary")) {
            rounds += rounds_len(row, &before);
            workloads += 1;
        }
    }
    let gemms: usize = fields(at(&doc, "gemm_16_rows")).iter().map(|(_, v)| arr(v).len()).sum();
    let e = elems("small-m");
    assert_eq!(count(&e, "dot"), rounds);
    assert_eq!(count(&e, "marker"), rounds, "one control cross per round");
    assert_eq!(count(&e, "median"), workloads);
    assert_eq!(count(&e, "bar"), gemms * 2);
}

#[test]
fn tok_draws_a_bar_per_lane_and_workload_and_every_round() {
    let doc = parse("embedgemma2_tok_m5pro.json");
    let rows = arr(at(&doc, "summary"));
    let lanes = ["tessl", "hf_loop", "hf_batch"];
    let dots: usize = rows
        .iter()
        .map(|r| lanes.iter().map(|l| rounds_len(r, l)).sum::<usize>())
        .sum();
    let e = elems("tok");
    assert_eq!(
        count(&e, "bar"),
        rows.len() * lanes.len(),
        "the load row has one bar per lane too"
    );
    assert_eq!(count(&e, "dot"), dots);
    assert_eq!(count(&e, "key"), lanes.len());
}

#[test]
fn video_draws_a_dot_per_run_and_a_bar_per_workload_and_precision() {
    let doc = parse("embedgemma2_video_m5pro.json");
    let runs = arr(at(&doc, "runs")).len();
    let e = elems("video");
    assert_eq!(count(&e, "bar"), 5);
    assert_eq!(count(&e, "dot"), 5 * runs);
}

// ---------------------------------------------------------------------------
// Axes and text
// ---------------------------------------------------------------------------

fn strictly_increasing(t: &[f64]) -> bool {
    t.windows(2).all(|w| w[0] < w[1])
}

#[test]
fn log_ticks_are_powers_of_ten_over_a_wide_range() {
    assert_eq!(log_ticks(1.0, 1000.0).unwrap(), vec![1.0, 10.0, 100.0, 1000.0]);
    assert_eq!(
        log_ticks(1000.0, 1.0).unwrap(),
        vec![1.0, 10.0, 100.0, 1000.0],
        "a reversed range is the same range"
    );
    let t = log_ticks(1e-9, 1e9).unwrap();
    assert!(t.len() <= 10 && t.len() >= 4, "{t:?}");
    assert!(
        strictly_increasing(&t) && t.iter().all(|v| (v.log10() - v.log10().round()).abs() < 1e-9),
        "{t:?}"
    );
    let t = log_ticks(1e-300, 1e300).unwrap();
    assert!(t.len() <= 10 && strictly_increasing(&t), "{t:?}");
}

#[test]
fn log_ticks_get_finer_as_the_range_narrows_and_stay_inside_it() {
    for (lo, hi) in [
        (2.0, 5000.0),
        (0.5, 4.2),
        (3.0, 60.0),
        (0.55, 1.1),
        (1.2, 1.9),
        (1.0, 1.0001),
        (7.0, 9.0),
        (0.9, 1000.0),
    ] {
        let t = log_ticks(lo, hi).unwrap();
        assert!(t.len() >= 2, "{lo}..{hi}: {t:?}");
        assert!(strictly_increasing(&t), "{lo}..{hi}: {t:?}");
        assert!(
            t.iter().all(|v| *v >= lo * (1.0 - 1e-9) && *v <= hi * (1.0 + 1e-9)),
            "{lo}..{hi}: {t:?}"
        );
    }
    assert_eq!(
        log_ticks(2.0, 5000.0).unwrap().len(),
        11,
        "1-2-5 over three and a half decades"
    );
    let narrow = log_ticks(1.2, 1.9).unwrap();
    assert!(narrow.len() >= 4, "{narrow:?}");
}

#[test]
fn ticks_refuse_ranges_that_cannot_be_drawn() {
    for (lo, hi) in [
        (0.0, 10.0),
        (-1.0, 10.0),
        (5.0, 5.0),
        (f64::NAN, 3.0),
        (1.0, f64::INFINITY),
    ] {
        assert!(log_ticks(lo, hi).is_err(), "log {lo}..{hi}");
        assert!(chart::Scale::log(lo, hi).is_err(), "scale {lo}..{hi}");
    }
    for (lo, hi) in [(5.0, 5.0), (f64::NAN, 3.0), (0.0, f64::NEG_INFINITY)] {
        assert!(linear_ticks(lo, hi, 5).is_err(), "linear {lo}..{hi}");
        assert!(chart::Scale::linear(lo, hi).is_err(), "scale {lo}..{hi}");
    }
    assert!(chart::AxisCfg::categories(&[], false).is_err());
}

#[test]
fn linear_ticks_are_round_and_inside_the_range() {
    assert_eq!(linear_ticks(0.0, 10.0, 5).unwrap(), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    assert_eq!(linear_ticks(0.0, 1.0, 5).unwrap().len(), 6);
    let t = linear_ticks(-3.0, 7.0, 5).unwrap();
    assert!(t.contains(&0.0) && strictly_increasing(&t), "{t:?}");
    let tiny = linear_ticks(1.0, 1.0 + 1e-9, 5).unwrap();
    assert!(
        tiny.len() >= 2 && tiny.iter().all(|v| (1.0..=1.0 + 1e-9).contains(v)),
        "{tiny:?}"
    );
    let t = linear_ticks(0.0, 1e12, 5).unwrap();
    assert!(t.len() <= 12 && t.len() >= 3, "{t:?}");
}

#[test]
fn round_numbers_for_limits() {
    assert_eq!(nice_floor(3.1).unwrap(), 2.0);
    assert_eq!(nice_floor(5.0).unwrap(), 5.0);
    assert_eq!(nice_floor(0.07).unwrap(), 0.05);
    assert_eq!(nice_ceil(1820.0 * 1.2).unwrap(), 5000.0);
    assert_eq!(nice_ceil(100.0).unwrap(), 100.0);
    assert_eq!(nice_top(114.0).unwrap(), 120.0);
    assert!(nice_floor(0.0).is_err() && nice_ceil(-1.0).is_err() && nice_top(f64::NAN).is_err());
}

#[test]
fn tick_labels_are_clean_numbers() {
    assert_eq!(fmt_tick(0.1 + 0.2), "0.3");
    assert_eq!(fmt_tick(0.0), "0");
    assert_eq!(fmt_tick(1.5), "1.5");
    assert_eq!(fmt_tick(0.0001), "0.0001");
    assert_eq!(fmt_tick(20_000_000.0), "20M");
    assert_eq!(fmt_tick(150_000.0), "150k");
    assert_eq!(fmt_tick(1000.0), "1000");
    assert_eq!(fmt_tick(-2.5), "-2.5");
}

#[test]
fn scales_map_the_ends_and_the_middle_and_clip_to_the_axis() {
    let lin = chart::Scale::linear(0.0, 10.0).unwrap();
    assert_eq!((lin.frac(0.0), lin.frac(5.0), lin.frac(10.0)), (0.0, 0.5, 1.0));
    let log = chart::Scale::log(1.0, 100.0).unwrap();
    assert!((log.frac(10.0) - 0.5).abs() < 1e-12);
    assert_eq!(
        log.frac(0.0),
        0.0,
        "zero on a log axis sits at the bottom, not at minus infinity"
    );
    let rev = chart::Scale::linear(4.5, -0.5).unwrap();
    assert!(
        rev.frac(0.0) > 0.89 && rev.frac(4.0) < 0.2,
        "a reversed axis puts the first category first"
    );
    assert_eq!(log.clamp(1e9), 100.0);
    assert!(!lin.contains(10.5) && lin.contains(10.0));
    let head = chart::Scale::headroom(chart::Kind::Log, 1.0, 100.0, 400.0, 100.0).unwrap();
    assert!(
        (head.frac(100.0) - 0.75).abs() < 1e-9,
        "100 px of a 400 px axis are left above the top mark"
    );
}

// ---------------------------------------------------------------------------
// XML escaping and the checker itself
// ---------------------------------------------------------------------------

#[test]
fn xml_text_is_escaped() {
    assert_eq!(
        svg::esc("a<b & c>d \"q\" 'r'"),
        "a&lt;b &amp; c&gt;d &quot;q&quot; &apos;r&apos;"
    );
    assert_eq!(svg::esc("tab\there\nbreak"), "tab here break");
    assert_eq!(
        svg::esc("bell\u{7}nul\u{0}del\u{7f}"),
        "bell\u{fffd}nul\u{fffd}del\u{fffd}"
    );
    assert_eq!(
        svg::esc("é — 日本 \u{1f600}"),
        "é — 日本 \u{1f600}",
        "legal characters pass through"
    );
    assert_eq!(
        svg::esc("&amp;"),
        "&amp;amp;",
        "an entity in the input is text, not markup"
    );
}

#[test]
fn hostile_labels_come_out_as_well_formed_text() {
    let hostile = "</text><script>alert(1)</script> & \"x\" 'y' \u{1}";
    let labels = vec![hostile.to_string(), "b".to_string()];
    let mut fig = chart::Figure::new(
        900.0,
        400.0,
        hostile,
        &[hostile],
        &[chart::LegendItem::boxed(hostile, "#2a78d6")],
    );
    let cells = fig.cells(&[1.0], &[1.0]);
    let mut ax = chart::Axes::new(
        cells[0],
        hostile,
        &[],
        chart::AxisCfg::categories(&labels, false).unwrap().labelled(hostile),
        chart::AxisCfg::linear(0.0, 10.0).unwrap().labelled(hostile),
    )
    .unwrap();
    ax.bar(0.0, 0.5, (0.0, 5.0), &chart::Paint::new("#2a78d6"));
    ax.text(0.0, 5.0, (0.0, -4.0), hostile, &chart::Ts::new(11.0));
    fig.add(ax);
    let svg = fig.finish();
    let elems = check_xml(&svg).expect("escaped text is well formed");
    assert!(!elems.iter().any(|e| e.name == "script"), "text became an element");
    assert!(svg.contains("&lt;/text&gt;&lt;script&gt;"));
}

#[test]
fn the_checker_rejects_what_it_is_there_to_reject() {
    for bad in [
        "<svg><g></svg>",
        "<svg></g></svg>",
        "<svg><g></svg></g>",
        "<svg a=\"1\" a=\"2\"/>",
        "<svg a=1/>",
        "<svg>a & b</svg>",
        "<svg>a &bogus; b</svg>",
        "<svg a=\"x<y\"/>",
        "<svg/><svg/>",
        "<g/>",
        "<svg>\u{1}</svg>",
        "<svg>",
        "text<svg/>",
    ] {
        assert!(check_xml(bad).is_err(), "accepted {bad:?}");
    }
    let ok = check_xml("<?xml version=\"1.0\"?><svg a=\"1\"><g b='2'/><t>x &amp; &#65; y</t></svg>\n").unwrap();
    assert_eq!(ok.len(), 3);
}

#[test]
fn text_is_wrapped_to_a_width_and_never_overflows_it() {
    let lines = svg::wrap(
        "a fairly long title that has to be broken over several lines",
        120.0,
        11.0,
    );
    assert!(lines.len() >= 3, "{lines:?}");
    assert!(lines.iter().all(|l| svg::text_width(l, 11.0) <= 120.0), "{lines:?}");
    assert_eq!(svg::wrap("a\nb", 500.0, 11.0), vec!["a", "b"]);
    let cut = svg::wrap("Pneumonoultramicroscopicsilicovolcanoconiosis", 60.0, 11.0);
    assert!(
        cut.len() > 1 && cut.iter().all(|l| svg::text_width(l, 11.0) <= 60.0),
        "{cut:?}"
    );
    assert_eq!(svg::num(-0.0), "0");
    assert_eq!(svg::num(1.0), "1");
    assert_eq!(svg::num(1.2345), "1.23");
    assert_eq!(svg::num(-0.001), "0");
}

#[test]
fn a_panel_too_small_for_its_labels_is_refused() {
    let cell = chart::Rect {
        x: 0.0,
        y: 0.0,
        w: 80.0,
        h: 70.0,
    };
    let r = chart::Axes::new(
        cell,
        "a title that takes the room",
        &[],
        chart::AxisCfg::linear(0.0, 1.0).unwrap().labelled("x"),
        chart::AxisCfg::linear(0.0, 1.0).unwrap().labelled("y"),
    );
    assert!(r.is_err_and(|e| e.contains("too small")));
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn tok_refuses_to_draw_when_the_lanes_produced_different_ids() {
    let dir = copy_of(
        "tok_hash",
        &[
            "embedgemma2_tok_m5pro.json",
            "embedgemma2_tok_m5pro_tessl_lane.json",
            "embedgemma2_tok_m5pro_hf_lane.json",
        ],
    );
    let r = run(&["tok", dir.join("embedgemma2_tok_m5pro.json").to_str().unwrap()]);
    assert_eq!(r.code, Some(0), "the untouched copy draws: {}", r.err);
    fs::remove_file(dir.join("embedgemma2_tok_m5pro.svg")).unwrap();
    tamper(
        &dir.join("embedgemma2_tok_m5pro_hf_lane.json"),
        "\"ids_fnv32\":",
        "\"ids_fnv32\":1",
        true,
    );
    let r = run(&["tok", dir.join("embedgemma2_tok_m5pro.json").to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("the lanes produced different ids"), "{}", r.err);
    assert!(no_svg_in(&dir), "a refused figure must not leave a file");
}

#[test]
fn tok_refuses_when_the_token_counts_differ_even_if_the_hashes_agree() {
    let dir = copy_of(
        "tok_count",
        &[
            "embedgemma2_tok_m5pro.json",
            "embedgemma2_tok_m5pro_tessl_lane.json",
            "embedgemma2_tok_m5pro_hf_lane.json",
        ],
    );
    tamper(
        &dir.join("embedgemma2_tok_m5pro_hf_lane.json"),
        "\"tokens\":",
        "\"tokens\":9",
        true,
    );
    let r = run(&["tok", dir.join("embedgemma2_tok_m5pro.json").to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("different ids"), "{}", r.err);
}

#[test]
fn tok_refuses_a_missing_lane_file() {
    let dir = copy_of(
        "tok_missing",
        &["embedgemma2_tok_m5pro.json", "embedgemma2_tok_m5pro_tessl_lane.json"],
    );
    let r = run(&["tok", dir.join("embedgemma2_tok_m5pro.json").to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("hf_lane.json"), "{}", r.err);
    assert!(no_svg_in(&dir));
}

#[test]
fn a_paired_figure_is_titled_by_its_runs_label_and_refused_without_one() {
    let f = "embedgemma2_final_m5pro.json";
    let dir = copy_of("label", &[f]);
    let p = dir.join(f);
    tamper(
        &p,
        "60a6240 against eg2/integrate\"",
        "branch point against the integrated tree\"",
        false,
    );
    let r = run(&["final", p.to_str().unwrap()]);
    assert_eq!(r.code, Some(0), "{}", r.err);
    let svg = fs::read_to_string(dir.join("embedgemma2_final_m5pro.svg")).unwrap();
    assert!(
        svg.contains("branch point against the integrated tree (6 interleaved rounds)"),
        "title is not the label"
    );
    assert!(!svg.contains("60a6240"), "a revision the run did not record is drawn");
    // The three runs it draws each carry their own label.
    for (name, want) in [
        ("final", "60a6240 against eg2/integrate"),
        ("integrate-text", "42f1d52 (branch point) against eg2/integrate"),
        ("integrate-mm", "multimodal encode on Apple M5 Pro: 42f1d52"),
    ] {
        assert!(render(name).svg.contains(want), "{name}");
    }
    fs::remove_file(dir.join("embedgemma2_final_m5pro.svg")).unwrap();
    tamper(&p, "\"label\":", "\"label_gone\":", false);
    let r = run(&["final", p.to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("final.label: missing key"), "{}", r.err);
    assert!(no_svg_in(&dir));
}

#[test]
fn lanes_the_colour_table_does_not_name_are_still_drawn_apart() {
    // A run's lane names are its own (`tessl_f32`, `torch_bf16`, ...); a figure
    // that had no colour for them drew every such lane the control's grey.
    let f = "embedgemma2_final_m5pro.json";
    let dir = copy_of("lane_colours", &[f]);
    let p = dir.join(f);
    tamper(&p, "after_f32", "lane_x", true);
    tamper(&p, "after_bf16", "lane_y", true);
    let r = run(&["final", p.to_str().unwrap()]);
    assert_eq!(r.code, Some(0), "{}", r.err);
    let svg = fs::read_to_string(dir.join("embedgemma2_final_m5pro.svg")).unwrap();
    // before_async keeps its blue and control its grey; the two unnamed lanes
    // take the next series colours no lane of the figure uses.
    for (colour, what) in [
        ("#2a78d6", "blue"),
        ("#8a8985", "grey"),
        ("#eb6834", "orange"),
        ("#1f9e6e", "green"),
    ] {
        assert!(svg.contains(colour), "no {what} series");
    }
}

#[test]
fn a_lane_that_recorded_no_device_peak_is_not_drawn_as_zero() {
    // The torch lanes print no device peak; a missing peak drawn as a 0 MiB bar
    // reads as "used no memory".
    let f = "embedgemma2_final_m5pro.json";
    let dir = copy_of("no_peak", &[f]);
    let p = dir.join(f);
    let r = run(&["final", p.to_str().unwrap()]);
    assert_eq!(r.code, Some(0), "{}", r.err);
    let with = fs::read_to_string(dir.join("embedgemma2_final_m5pro.svg")).unwrap();
    assert!(!with.contains("not recorded"), "a recorded peak is called missing");
    fs::remove_file(dir.join("embedgemma2_final_m5pro.svg")).unwrap();
    tamper(&p, "\"device_peak_mib_max\"", "\"device_peak_gone\"", true);
    let r = run(&["final", p.to_str().unwrap()]);
    assert_eq!(r.code, Some(0), "{}", r.err);
    let svg = fs::read_to_string(dir.join("embedgemma2_final_m5pro.svg")).unwrap();
    // Eight workloads, four lanes: every bar of the panel is a "not recorded".
    assert_eq!(
        svg.matches("not recorded").count(),
        8 * 4,
        "missing peaks are not marked"
    );
}

#[test]
fn figures_whose_data_records_no_machine_state_claim_none() {
    // These result files hold timings only; the load they ran at is in the .txt
    // beside them, so a title asserting it would be a constant, not a record.
    for name in ["jpeg-decode", "video-color"] {
        let svg = render(name).svg;
        assert!(!svg.contains("shared machine"), "{name}");
        assert!(svg.contains("recorded in the .txt beside the data"), "{name}");
    }
}

#[test]
fn a_missing_key_is_a_named_error_not_a_panic() {
    let dir = copy_of("missing_key", &["embedgemma2_final_m5pro.json"]);
    tamper(
        &dir.join("embedgemma2_final_m5pro.json"),
        "\"summary\":",
        "\"summary_gone\":",
        false,
    );
    let r = run(&["final", dir.join("embedgemma2_final_m5pro.json").to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("final.summary: missing key"), "{}", r.err);
    assert!(!r.err.contains("panicked"));
    assert!(no_svg_in(&dir));
}

#[test]
fn a_missing_key_deep_in_a_row_names_the_row() {
    let dir = copy_of("missing_deep", &["embedgemma2_pre_m5pro.json"]);
    tamper(
        &dir.join("embedgemma2_pre_m5pro.json"),
        "\"ms_min_all\":",
        "\"ms_min_every\":",
        false,
    );
    let r = run(&["pre", dir.join("embedgemma2_pre_m5pro.json").to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(
        r.err.contains("pre.summary[0].tessl.ms_min_all: missing key"),
        "{}",
        r.err
    );
}

fn write_json(dir: &Path, name: &str, text: &str) -> PathBuf {
    let p = dir.join(name);
    fs::write(&p, text).unwrap();
    p
}

#[test]
fn empty_data_is_refused_for_every_figure_that_reads_a_summary() {
    let dir = fresh("empty");
    for (name, stem) in [
        ("final", "embedgemma2_final_m5pro"),
        ("audio-speed", "embedgemma2_audio_speed_m5pro"),
        ("mm-onewait", "embedgemma2_mm_onewait_m5pro"),
        ("pre", "embedgemma2_pre_m5pro"),
    ] {
        let p = write_json(
            &dir,
            &format!("{stem}.json"),
            "{\"rounds\":1,\"lanes\":[{\"name\":\"a\"},{\"name\":\"b\"}],\"summary\":[]}",
        );
        let r = run(&[name, p.to_str().unwrap()]);
        assert_eq!(r.code, Some(1), "{name}");
        assert!(r.err.contains("empty list, nothing to plot"), "{name}: {}", r.err);
    }
    assert!(no_svg_in(&dir));
}

#[test]
fn media_decode_draws_a_bar_and_a_median_per_workload() {
    let rows = arr(&parse("media_decode_m5pro.json")).len();
    assert!(rows > 0);
    let e = elems("media-decode");
    assert_eq!(count(&e, "bar"), rows);
    assert_eq!(count(&e, "median"), rows);
}

#[test]
fn media_decode_refuses_an_empty_list_and_an_unknown_unit() {
    let dir = fresh("media_bad");
    let p = write_json(&dir, "media_decode_m5pro.json", "[]");
    let r = run(&["media-decode", p.to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("empty list, nothing to plot"), "{}", r.err);
    let p = write_json(
        &dir,
        "media_decode_m5pro.json",
        "[{\"workload\":\"x.wav\",\"ms_min\":1.0,\"ms_median\":1.5,\"iters\":3,\"units\":30.0,\"unit\":\"bytes\"}]",
    );
    let r = run(&["media-decode", p.to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("unknown unit \"bytes\""), "{}", r.err);
    assert!(no_svg_in(&dir));
}

#[test]
fn video_color_draws_a_bar_and_a_median_per_path_of_every_call() {
    let doc = parse("media_video_color_m5pro.json");
    let rows: Vec<&str> = arr(&doc)
        .iter()
        .map(|r| s(at(r, "workload")))
        .filter(|w| !w.ends_with(":video_open") && !w.starts_with("yuv:"))
        .collect();
    let drawn = rows.len();
    // Two paths (exact, bgra) for every call, or three (and exact1) in a run that timed the
    // converter on one thread as well.
    let paths = if rows.iter().any(|w| w.ends_with(":exact1")) {
        3
    } else {
        2
    };
    assert!(drawn > 0 && drawn % paths == 0);
    let e = elems("video-color");
    assert_eq!(count(&e, "bar"), drawn);
    assert_eq!(count(&e, "median"), drawn);
    video_color_draws_three_bars_per_call_when_the_single_thread_path_was_timed();
}

/// A run that also timed the single-threaded converter draws three bars per call (called from
/// `video_color_draws_a_bar_and_a_median_per_path_of_every_call`, which keeps the test count).
fn video_color_draws_three_bars_per_call_when_the_single_thread_path_was_timed() {
    let dir = fresh("video_color_three");
    let row = |workload: &str| {
        format!("{{\"workload\":\"{workload}\",\"ms_min\":{},\"ms_median\":1.5,\"iters\":3,\"units\":30.0,\"unit\":\"frame\"}}", workload.len())
    };
    // Two calls of one clip, the converter-alone rows and the open row of a bench run with BENCH_YUV.
    let rows = [
        "a.mp4:video_open",
        "a.mp4:video_first_30:exact",
        "a.mp4:video_first_30:exact1",
        "a.mp4:video_first_30:bgra",
        "a.mp4:video_sampled_9:exact",
        "a.mp4:video_sampled_9:exact1",
        "a.mp4:video_sampled_9:bgra",
        "yuv:420_8bit_1280x720:single",
        "yuv:420_8bit_1280x720:threads",
    ]
    .map(row);
    let p = write_json(&dir, "media_video_color_m5pro.json", &format!("[{}]", rows.join(",")));
    let plot = registry::find("video-color").expect("video-color");
    let out = (plot.render)(&p).unwrap_or_else(|e| panic!("video-color: {e}"));
    let e = check_xml(&out.svg).unwrap_or_else(|e| panic!("video-color: {e}"));
    assert_eq!(count(&e, "bar"), 6);
    assert_eq!(count(&e, "median"), 6);
    assert!(out.svg.contains("exact1: the same on one thread"));
}

#[test]
fn video_color_refuses_what_it_cannot_pair_or_name() {
    let dir = fresh("video_color_bad");
    let row = |workload: &str, unit: &str| {
        format!("{{\"workload\":\"{workload}\",\"ms_min\":1.0,\"ms_median\":1.5,\"iters\":3,\"units\":30.0,\"unit\":\"{unit}\"}}")
    };
    let cases = [
        ("[]".to_string(), "empty list, nothing to plot"),
        (
            format!("[{}]", row("a.mp4:video_open", "frame")),
            "no video_sampled/video_first workloads",
        ),
        (
            format!("[{}]", row("a.mp4:video_first_30:exact", "frame")),
            "needs both an exact and a bgra row",
        ),
        (
            // One call with exact1, another without: the run is half threaded-timed.
            format!(
                "[{},{},{},{},{}]",
                row("a.mp4:video_first_30:exact", "frame"),
                row("a.mp4:video_first_30:exact1", "frame"),
                row("a.mp4:video_first_30:bgra", "frame"),
                row("a.mp4:video_sampled_9:exact", "frame"),
                row("a.mp4:video_sampled_9:bgra", "frame")
            ),
            "other calls have an exact1 row, this one needs it too",
        ),
        (
            format!("[{}]", row("yuv:420_8bit_1280x720:single", "frame")),
            "no video_sampled/video_first workloads",
        ),
        (
            format!(
                "[{},{}]",
                row("a.mp4:video_first_30:exact", "frame"),
                row("a.mp4:video_first_30:cv2", "frame")
            ),
            "unknown colour path \"cv2\"",
        ),
        (
            format!(
                "[{},{}]",
                row("a.mp4:video_first_30:exact", "frame"),
                row("a.mp4:video_first_30:exact", "frame")
            ),
            "appears twice",
        ),
        (
            format!("[{}]", row("a.mp4:video_first_30:exact", "megapixel")),
            "a video workload counts frames",
        ),
        (format!("[{}]", row("a.mp4", "frame")), "expected FILE:video_CALL:PATH"),
    ];
    for (json, reason) in cases {
        let p = write_json(&dir, "media_video_color_m5pro.json", &json);
        let r = run(&["video-color", p.to_str().unwrap()]);
        assert_eq!(r.code, Some(1), "{json}");
        assert!(r.err.contains(reason), "{json}: {}", r.err);
    }
    assert!(no_svg_in(&dir));
}

#[test]
fn jpeg_decode_draws_a_native_and_an_imageio_bar_and_median_per_file() {
    let rows = arr(&parse("jpeg_decode_m5pro.json")).len();
    assert!(rows > 0);
    let e = elems("jpeg-decode");
    assert_eq!(count(&e, "bar"), 2 * rows);
    assert_eq!(count(&e, "median"), 2 * rows);
    let svg = render("jpeg-decode").svg;
    assert!(
        svg.contains("native (tessl::media::jpeg"),
        "legend names the native decoder"
    );
    assert!(svg.contains("ImageIO"), "legend names ImageIO");
    assert!(
        svg.contains("Indicative, not a paired comparison"),
        "the figure says the numbers are indicative"
    );
}

#[test]
fn jpeg_decode_refuses_an_empty_list_a_missing_key_and_a_time_that_cannot_go_on_a_log_axis() {
    let dir = fresh("jpeg_bad");
    let p = write_json(&dir, "jpeg_decode_m5pro.json", "[]");
    let r = run(&["jpeg-decode", p.to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("empty list, nothing to plot"), "{}", r.err);
    let row = |extra: &str| {
        format!(
            "[{{\"workload\":\"x.jpg\",\"megapixels\":2.0,\"native_ms_min\":1.0,\"native_ms_median\":1.5,{extra}\"iters\":3}}]"
        )
    };
    let p = write_json(&dir, "jpeg_decode_m5pro.json", &row(""));
    let r = run(&["jpeg-decode", p.to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("imageio_ms_min: missing key"), "{}", r.err);
    let p = write_json(
        &dir,
        "jpeg_decode_m5pro.json",
        &row("\"imageio_ms_min\":0.0,\"imageio_ms_median\":1.0,"),
    );
    let r = run(&["jpeg-decode", p.to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("imageio_ms_min"), "{}", r.err);
    assert!(no_svg_in(&dir));
}

#[test]
fn tok_refuses_an_empty_summary() {
    let dir = copy_of(
        "tok_empty",
        &[
            "embedgemma2_tok_m5pro_tessl_lane.json",
            "embedgemma2_tok_m5pro_hf_lane.json",
        ],
    );
    let p = write_json(&dir, "embedgemma2_tok_m5pro.json", "{\"rounds\":1,\"summary\":[]}");
    let r = run(&["tok", p.to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("empty list, nothing to plot"), "{}", r.err);
}

#[test]
fn empty_round_lists_and_empty_lanes_are_refused() {
    let dir = fresh("empty_rounds");
    let p = write_json(
        &dir,
        "embedgemma2_pre_m5pro.json",
        "{\"summary\":[{\"workload\":\"w\",\"tessl\":{\"ms_min_median\":1,\"ms_min_all\":[]},\
         \"hf\":{\"ms_min_median\":2,\"ms_min_all\":[2],\"vs_base\":{\"median\":2,\"lo\":2,\"hi\":2}}}]}",
    );
    let r = run(&["pre", p.to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("empty list"), "{}", r.err);
    let q = write_json(
        &dir,
        "embedgemma2_final_m5pro.json",
        "{\"lanes\":[],\"summary\":[{\"workload\":\"w\"}]}",
    );
    let r = run(&["final", q.to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("empty list"), "{}", r.err);
}

#[test]
fn a_time_that_cannot_go_on_a_log_axis_is_refused() {
    let dir = copy_of("log_zero", &["embedgemma2_final_m5pro.json"]);
    tamper(
        &dir.join("embedgemma2_final_m5pro.json"),
        "\"ms_min_median\": 7.3713",
        "\"ms_min_median\": 0",
        false,
    );
    let r = run(&["final", dir.join("embedgemma2_final_m5pro.json").to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("is not above zero"), "{}", r.err);
}

#[test]
fn a_number_that_is_not_finite_is_refused() {
    let dir = copy_of("not_finite", &["embedgemma2_pre_m5pro.json"]);
    tamper(
        &dir.join("embedgemma2_pre_m5pro.json"),
        "\"ms_min_median\": 0.7464500000000001",
        "\"ms_min_median\": 1e999",
        false,
    );
    let r = run(&["pre", dir.join("embedgemma2_pre_m5pro.json").to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("overflows f64"), "{}", r.err);
}

#[test]
fn malformed_json_is_refused_with_its_path() {
    let dir = fresh("malformed");
    let p = write_json(&dir, "embedgemma2_pre_m5pro.json", "{\"summary\": [1, 2,");
    let r = run(&["pre", p.to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("embedgemma2_pre_m5pro.json"), "{}", r.err);
}

#[test]
fn an_unknown_stage_class_is_refused_by_the_stage_profile_and_the_fusion_figure() {
    for (name, file, from) in [
        ("stage-profile", "embedgemma2_stage_profile_m5pro.json", "gather+copy"),
        (
            "text-fusion",
            "embedgemma2_text_fusion_profile_m5pro.json",
            "gather+copy",
        ),
    ] {
        let dir = copy_of(&format!("class_{name}"), &[file]);
        tamper(&dir.join(file), from, "zz unknown stage", true);
        let r = run(&[name, dir.join(file).to_str().unwrap()]);
        assert_eq!(r.code, Some(1), "{name}");
        assert!(
            r.err.contains("zz unknown stage") && r.err.contains("belongs to no"),
            "{name}: {}",
            r.err
        );
        assert!(no_svg_in(&dir), "{name}");
    }
}

#[test]
fn a_workload_the_stage_profile_cannot_order_is_refused() {
    let dir = copy_of("stage_workload", &["embedgemma2_stage_profile_m5pro.json"]);
    tamper(
        &dir.join("embedgemma2_stage_profile_m5pro.json"),
        "\"workload\":\"1x16\"",
        "\"workload\":\"one-by-sixteen\"",
        true,
    );
    let r = run(&[
        "stage-profile",
        dir.join("embedgemma2_stage_profile_m5pro.json").to_str().unwrap(),
    ]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("is not AxB"), "{}", r.err);
}

#[test]
fn the_memory_itemisation_must_add_up_to_the_measured_end() {
    let dir = copy_of("memory_sum", &["embedgemma2_memory_m5pro.json"]);
    let r = run(&["memory", dir.join("embedgemma2_memory_m5pro.json").to_str().unwrap()]);
    assert_eq!(r.code, Some(0), "{}", r.err);
    fs::remove_file(dir.join("embedgemma2_memory_m5pro.svg")).unwrap();
    tamper(
        &dir.join("embedgemma2_memory_m5pro.json"),
        "\"after_mib\": 400.47",
        "\"after_mib\": 420.47",
        false,
    );
    let r = run(&["memory", dir.join("embedgemma2_memory_m5pro.json").to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("does not add up"), "{}", r.err);
    assert!(no_svg_in(&dir));
}

#[test]
fn the_memory_figure_refuses_series_of_different_lengths() {
    let dir = copy_of("memory_series", &["embedgemma2_memory_m5pro.json"]);
    // Drop one round from the first "after" series: the ratios no longer pair up.
    let path = dir.join("embedgemma2_memory_m5pro.json");
    let text = fs::read_to_string(&path).unwrap();
    let series = text.find("\"series_ms\"").expect("a series_ms object");
    let at_after = text[series..].find("\"after\": {").expect("an after series") + series;
    let open = text[at_after..].find('[').unwrap() + at_after + 1;
    let comma = text[open..].find(',').unwrap() + open;
    fs::write(&path, format!("{}{}", &text[..open], &text[comma + 1..])).unwrap();
    let r = run(&["memory", path.to_str().unwrap()]);
    assert_eq!(r.code, Some(1), "{}", r.out);
    assert!(r.err.contains("rounds after against"), "{}", r.err);
}

#[test]
fn all_is_all_or_nothing() {
    let files: Vec<String> = registry::PLOTS.iter().map(|p| p.file.to_string()).collect();
    let mut want: Vec<&str> = files.iter().map(String::as_str).collect();
    want.extend([
        "embedgemma2_tok_m5pro_tessl_lane.json",
        "embedgemma2_tok_m5pro_hf_lane.json",
        "embedgemma2_vision_profile_m5pro.json",
    ]);
    let dir = copy_of("all_refused", &want);
    tamper(
        &dir.join("embedgemma2_video_m5pro.json"),
        "\"runs\":",
        "\"runs_gone\":",
        false,
    );
    let r = run(&["all", dir.to_str().unwrap()]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("video") && r.err.contains("missing key"), "{}", r.err);
    assert!(no_svg_in(&dir), "figures drawn before the refusal must not be written");
}

#[test]
fn bad_command_lines_are_refused() {
    let none: [&str; 0] = [];
    let r = run(&none);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("usage"), "{}", r.err);
    let r = run(&["nonesuch", "x.json"]);
    assert_eq!(r.code, Some(1));
    assert!(
        r.err.contains("unknown figure") && r.err.contains("audio-speed"),
        "{}",
        r.err
    );
    let r = run(&["final"]);
    assert_eq!(r.code, Some(1));
    let r = run(&["final", "a.json", "--out-dir"]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("--out-dir needs"), "{}", r.err);
    let r = run(&["final", "a.json", "--frob"]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("unknown flag"), "{}", r.err);
    let r = run(&["final", "no_such_file.json"]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("no_such_file.json"), "{}", r.err);
}

#[test]
fn the_default_output_is_beside_the_result_file() {
    let dir = copy_of("beside", &["embedgemma2_pre_m5pro.json"]);
    let r = run(&["pre", dir.join("embedgemma2_pre_m5pro.json").to_str().unwrap()]);
    assert_eq!(r.code, Some(0), "{}", r.err);
    let svg = fs::read_to_string(dir.join("embedgemma2_pre_m5pro.svg")).unwrap();
    check_xml(&svg).unwrap();
}
