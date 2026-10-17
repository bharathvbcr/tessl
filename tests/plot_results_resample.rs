//! The `media-resample` figure of `src/bin/plot_results` (libswresample's
//! algorithm in Rust against the system `AudioConverter`, from
//! `bench_media_resample`): the marks drawn are the result file's, and a file it
//! cannot draw is refused with the reason and nothing written. That the figure
//! is well formed, deterministic and current with `bench/results/` is held by
//! the table-driven tests of `tests/plot_results.rs`, which lists it.
//!
//! No GPU, no Python.

#[allow(dead_code)]
#[path = "../src/bin/plot_results/chart.rs"]
mod chart;
#[allow(dead_code)]
#[path = "../src/bin/plot_results/data.rs"]
mod data;
#[allow(dead_code)]
#[path = "../src/bin/plot_results/go_overhead.rs"]
mod go_overhead;
#[allow(dead_code)]
#[path = "../src/json.rs"]
mod json;
#[allow(dead_code)]
#[path = "../src/bin/plot_results/media.rs"]
mod media;
#[allow(dead_code)]
#[path = "../src/bin/plot_results/memory.rs"]
mod memory;
#[allow(dead_code)]
#[path = "../src/bin/plot_results/paired.rs"]
mod paired;
#[allow(dead_code)]
#[path = "../src/bin/plot_results/registry.rs"]
mod registry;
#[allow(dead_code)]
#[path = "../src/bin/plot_results/resample.rs"]
mod resample;
#[allow(dead_code)]
#[path = "../src/bin/plot_results/small_m.rs"]
mod small_m;
#[allow(dead_code)]
#[path = "../src/bin/plot_results/stage.rs"]
mod stage;
#[allow(dead_code)]
#[path = "../src/bin/plot_results/svg.rs"]
mod svg;
#[allow(dead_code)]
#[path = "../src/bin/plot_results/tok.rs"]
mod tok;
#[allow(dead_code)]
#[path = "../src/bin/plot_results/vision.rs"]
mod vision;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn committed() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/results/media_resample_m5pro.json")
}

fn fresh(case: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("plot_results_resample")
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

fn svg_of(path: &Path) -> String {
    (registry::find("media-resample").unwrap().render)(path).unwrap().svg
}

#[test]
fn it_draws_a_bar_and_a_median_per_row_and_a_key_per_stage() {
    let rows = fs::read_to_string(committed()).unwrap().matches("\"workload\"").count();
    assert!(rows >= 4, "{rows} rows");
    let svg = svg_of(&committed());
    assert_eq!(svg.matches("class=\"bar\"").count(), rows);
    assert_eq!(svg.matches("class=\"median sk\"").count(), rows);
    // Four stages and the median tick.
    assert_eq!(
        svg.matches("class=\"key\"").count() + svg.matches("class=\"key sk\"").count(),
        5
    );
    for stage in ["swresample", "audiotoolbox", "native decode"] {
        assert!(svg.contains(stage), "{stage}");
    }
}

#[test]
fn it_refuses_an_empty_list_an_unknown_stage_and_a_time_that_is_not_positive() {
    let dir = fresh("bad");
    let write = |body: &str| {
        let p = dir.join("media_resample_m5pro.json");
        fs::write(&p, body).unwrap();
        p
    };
    let row = |stage: &str, ms: &str| {
        format!("[{{\"workload\":\"x.wav\",\"stage\":\"{stage}\",\"ms_min\":{ms},\"ms_median\":2.0,\"iters\":3,\"audio_s\":30.0}}]")
    };
    let cases = [
        ("[]".to_string(), "empty list, nothing to plot"),
        (row("sox", "1.0"), "unknown stage \"sox\""),
        (row("swresample", "0.0"), "ms_min"),
        (
            "[{\"workload\":\"x.wav\",\"stage\":\"swresample\"}]".to_string(),
            "missing key",
        ),
    ];
    for (body, reason) in cases {
        let p = write(&body);
        let (code, _, err) = run(&["media-resample", p.to_str().unwrap()]);
        assert_eq!(code, Some(1), "{body}");
        assert!(err.contains(reason), "{body}: {err}");
    }
    assert!(fs::read_dir(&dir)
        .unwrap()
        .all(|e| e.unwrap().path().extension().is_none_or(|x| x != "svg")));
}

#[test]
fn the_binary_writes_the_svg_the_library_draws() {
    let dir = fresh("binary");
    let (code, out, err) = run(&[
        "media-resample",
        committed().to_str().unwrap(),
        "--out-dir",
        dir.to_str().unwrap(),
    ]);
    assert_eq!(code, Some(0), "{err}");
    assert!(out.contains("wrote "), "{out}");
    assert_eq!(
        fs::read_to_string(dir.join("media_resample_m5pro.svg")).unwrap(),
        svg_of(&committed())
    );
}
