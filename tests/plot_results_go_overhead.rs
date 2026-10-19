//! The `go-overhead` figure of `src/bin/plot_results` (what the Go binding adds
//! to a tessl call, from `go/cmd/bench_overhead`): the marks drawn are the result
//! file's, and a file it cannot draw is refused with the reason and nothing
//! written. That the figure is well formed, deterministic and current with
//! `bench/results/` is held by the table-driven tests of `tests/plot_results.rs`,
//! which lists it.
//!
//! No GPU, no Python, no Go.

// The bin's own module list, so this test compiles every figure the registry names.
include!("../src/bin/plot_results/modules.rs");

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn committed() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/results/embedgemma2_go_m5pro.json")
}

fn fresh(case: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("plot_results_go_overhead")
        .join(case);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(args: &[&str]) -> (Option<i32>, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_plot_results"))
        .args(args)
        .output()
        .unwrap();
    (
        out.status.code(),
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
    )
}

fn svg_of(path: &Path) -> registry::Output {
    (registry::find("go-overhead").unwrap().render)(path).unwrap()
}

/// A result document with the given rows (already JSON) and ping minimum.
fn doc(rows: &str, ping: &str) -> String {
    doc_with_load(rows, ping, "{ 3.87 5.25 6.69 }")
}

/// [`doc`] with the `vm.loadavg` record taken before the run (after: `{ 5.56 5.54 6.70 }`).
fn doc_with_load(rows: &str, ping: &str, before: &str) -> String {
    format!(
        "{{\"benchmark\":\"embedgemma2_go_overhead\",\"rounds\":3,\"iters_per_round\":10,\
         \"host\":{{\"loadavg_before\":\"{before}\",\"loadavg_after\":\"{{ 5.56 5.54 6.70 }}\"}},\
         \"ping\":{{\"min_us\":{ping}}},\"rows\":[{rows}]}}"
    )
}

#[test]
fn the_figure_states_the_recorded_load_and_refuses_a_bad_record() {
    let dir = fresh("load");
    let write = |body: &str| {
        let p = dir.join("embedgemma2_go_m5pro.json");
        fs::write(&p, body).unwrap();
        p
    };
    let svg = svg_of(&write(&doc(&row("texts x1", "60.0", "10.0"), "3.0"))).svg;
    assert!(
        svg.contains("1-minute load average 3.9 before, 5.6 after"),
        "load not drawn"
    );
    assert!(!svg.contains("shared machine"), "a condition the run did not record");
    for bad in ["", "{ }", "{ busy 1 2 }", "{ -1 0 0 }"] {
        let p = write(&doc_with_load(&row("texts x1", "60.0", "10.0"), "3.0", bad));
        let (code, _, err) = run(&["go-overhead", p.to_str().unwrap()]);
        assert_eq!(code, Some(1), "{bad:?}");
        assert!(err.contains("is not a vm.loadavg record"), "{bad:?}: {err}");
    }
}

/// One workload's row; `paired` is the paired overhead minimum, `vs_direct` the
/// difference of the minima (which may be negative: two processes' GPU times).
fn row(name: &str, paired: &str, vs_direct: &str) -> String {
    format!(
        "{{\"workload\":\"{name}\",\"direct_main_min_us\":480.0,\"direct_main_median_us\":500.0,\
         \"direct_min_us\":500.0,\"direct_median_us\":520.0,\"go_min_us\":560.0,\"go_median_us\":590.0,\
         \"vs_direct_min_us\":{vs_direct},\"paired_overhead_min_us\":{paired},\"paired_overhead_median_us\":55.0,\"ratio_min\":1.12}}"
    )
}

#[test]
fn it_draws_three_bars_and_medians_and_an_overhead_bar_and_median_per_workload() {
    let text = fs::read_to_string(committed()).unwrap();
    let rows = text.matches("\"workload\"").count();
    assert!(rows >= 5, "{rows} rows");
    let out = svg_of(&committed());
    // Main-thread direct, worker-thread direct and Go on the left; the added
    // time on the right; each bar has its median tick.
    assert_eq!(out.svg.matches("class=\"bar\"").count(), 4 * rows);
    assert_eq!(out.svg.matches("class=\"median sk\"").count(), 4 * rows);
    // Four keys on the left, three on the right.
    assert_eq!(
        out.svg.matches("class=\"key\"").count() + out.svg.matches("class=\"key sk\"").count(),
        7
    );
    // The table the result's .txt quotes has a line per workload and the ping.
    assert_eq!(out.table.lines().count(), rows + 2, "{}", out.table);
    assert!(out.table.contains("ping (noop)"));
}

#[test]
fn it_refuses_an_empty_list_a_missing_key_and_a_boundary_that_cost_nothing() {
    let dir = fresh("bad");
    let write = |body: &str| {
        let p = dir.join("embedgemma2_go_m5pro.json");
        fs::write(&p, body).unwrap();
        p
    };
    let cases = [
        (doc("", "3.0"), "empty list, nothing to plot"),
        // Go wall includes the engine's timing of the same call, so the paired
        // overhead cannot be zero or less: the file is wrong.
        (doc(&row("texts x1", "0.0", "10.0"), "3.0"), "paired_overhead_min_us"),
        (doc(&row("texts x1", "-12.5", "10.0"), "3.0"), "not above zero"),
        (doc(&row("texts x1", "60.0", "10.0"), "0.0"), "min_us"),
        (
            "{\"rounds\":3,\"iters_per_round\":10,\"ping\":{\"min_us\":3.0},\"rows\":[{\"workload\":\"x\"}]}"
                .to_string(),
            "missing key",
        ),
    ];
    for (body, reason) in cases {
        let p = write(&body);
        let (code, _, err) = run(&["go-overhead", p.to_str().unwrap()]);
        assert_eq!(code, Some(1), "{body}");
        assert!(err.contains(reason), "{body}: {err}");
    }
    assert!(fs::read_dir(&dir)
        .unwrap()
        .all(|e| e.unwrap().path().extension().is_none_or(|x| x != "svg")));
}

#[test]
fn it_draws_a_small_document_with_the_marks_of_its_rows_and_quotes_a_negative_difference() {
    let dir = fresh("small");
    let p = dir.join("embedgemma2_go_m5pro.json");
    // A negative go-minus-direct is two processes' GPU times differing: it is
    // not drawn, and the table says it.
    fs::write(
        &p,
        doc(
            &format!("{},{}", row("a", "60.0", "-31.5"), row("b", "75.5", "12.0")),
            "3.2",
        ),
    )
    .unwrap();
    let out = svg_of(&p);
    assert_eq!(out.svg.matches("class=\"bar\"").count(), 8);
    assert!(
        out.table.contains("60.0") && out.table.contains("75.5"),
        "{}",
        out.table
    );
    assert!(out.table.contains("-31.5"), "{}", out.table);
}

#[test]
fn the_binary_writes_the_svg_the_library_draws() {
    let dir = fresh("binary");
    let (code, out, err) = run(&[
        "go-overhead",
        committed().to_str().unwrap(),
        "--out-dir",
        dir.to_str().unwrap(),
    ]);
    assert_eq!(code, Some(0), "{err}");
    assert!(out.contains("wrote "), "{out}");
    assert!(out.contains("ping (noop)"), "the table is printed: {out}");
    assert_eq!(
        fs::read_to_string(dir.join("embedgemma2_go_m5pro.svg")).unwrap(),
        svg_of(&committed()).svg
    );
}
