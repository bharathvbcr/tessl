//! `kernel_coverage` (`src/bin/common/kcov.rs`): the kernel census and the
//! measurement of which kernels the benchmark suite dispatches.
//!
//! These are the cases of the retired `bench/test_kernel_coverage_adversarial.py`
//! (the phantom and missed kernels the source scan once reported wrongly, the
//! exit statuses, the stale-binary refusal) and of the kernel-inventory section
//! of `bench/test_parity_harness.py`, plus the suite, report and command-line
//! paths the Python tool never had a test for.
//!
//! The census runs the real `xcrun metal -E`. That is not an optional extra: the
//! crate cannot be built without the Metal compiler, so where this test runs it
//! is there, and a test that skipped itself without it would read as a pass over
//! a census it never took. (The Python harness skipped the metallib cross-check
//! when `metal-nm` was missing and failed only under `TESSL_HOSTED`.)

#[path = "../src/bin/common/kcov.rs"]
mod kcov;
#[path = "../src/bin/common/proc.rs"]
mod proc;
#[path = "../src/bin/common/pyjson.rs"]
mod pyjson;

use kcov::{Census, Failure, Traced};
use std::collections::{BTreeMap, BTreeSet};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime};

fn tmp(tag: &str) -> kcov::TempDir {
    kcov::scratch_dir(&format!("tessl-kcov-test-{tag}")).expect("scratch dir")
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|s| s.to_string()).collect()
}

/// The entry points of one source, through the real preprocessor and no metallib.
fn scan(text: &str) -> BTreeSet<String> {
    let d = tmp("scan");
    write(&d.path().join("a.metal"), text);
    kcov::inventory(d.path(), Some(Path::new("/nonexistent")), false)
        .unwrap_or_else(|e| panic!("scan refused: {e:?}\n{text}"))
        .names
}

fn scan_err(text: &str) -> Failure {
    let d = tmp("scan-err");
    write(&d.path().join("a.metal"), text);
    match kcov::inventory(d.path(), Some(Path::new("/nonexistent")), false) {
        Ok(c) => panic!("accepted: {:?}", c.names),
        Err(e) => e,
    }
}

const STAMP: &str = "#define STAMP(NAME, T) \\\nkernel void NAME(device T* x) {}\n";

// --------------------------------------- phantom kernels (adversarial tests)

#[test]
fn an_invocation_in_a_block_comment_is_not_a_kernel() {
    assert_eq!(
        scan(&format!("{STAMP}STAMP(live, float)\n/*\nSTAMP(dead, float)\n*/\n")),
        set(&["live"])
    );
}

#[test]
fn an_invocation_under_if_0_is_not_a_kernel() {
    assert_eq!(
        scan(&format!(
            "{STAMP}STAMP(live, float)\n#if 0\nSTAMP(dead, float)\n#endif\n"
        )),
        set(&["live"])
    );
}

#[test]
fn an_invocation_under_an_unset_ifdef_is_not_a_kernel() {
    assert_eq!(
        scan(&format!(
            "{STAMP}STAMP(live, float)\n#ifdef NOPE\nSTAMP(dead, float)\n#endif\n"
        )),
        set(&["live"])
    );
}

#[test]
fn the_word_kernel_inside_a_macro_body_comment_is_not_a_kernel() {
    let body =
        "#define S2(NAME) \\\n  /* kernel void ghost(x) */ \\\n kernel void NAME(device float* x) {}\nS2(real)\n";
    assert_eq!(scan(body), set(&["real"]));
}

#[test]
fn a_plain_kernel_in_a_block_comment_is_not_a_kernel() {
    assert_eq!(
        scan("kernel void live(device float* x) {}\n/*\nkernel void dead(device float* x) {}\n*/\n"),
        set(&["live"])
    );
}

// ---------------------------------------- missed kernels (adversarial tests)

#[test]
fn two_invocations_on_one_line_are_both_kernels() {
    assert_eq!(
        scan(&format!("{STAMP}STAMP(a1, float) STAMP(a2, float)\n")),
        set(&["a1", "a2"])
    );
}

#[test]
fn an_indented_invocation_is_a_kernel() {
    assert_eq!(
        scan(&format!("{STAMP}STAMP(top, float)\n  STAMP(indented, float)\n")),
        set(&["top", "indented"])
    );
}

#[test]
fn an_indented_plain_kernel_is_a_kernel() {
    assert_eq!(
        scan("kernel void top(device float* x) {}\n  kernel void indented(device float* x) {}\n"),
        set(&["top", "indented"])
    );
}

#[test]
fn a_host_name_alias_is_the_exported_name() {
    // metal-nm exports the host_name, not the C++ symbol.
    assert_eq!(
        scan("[[host_name(\"alias\")]] kernel void inner(device float* x) {}\n"),
        set(&["alias"])
    );
}

// ------------------------------------------------------------------ the census

#[test]
fn a_template_definition_is_not_an_entry_point_but_its_instantiation_is() {
    let src = "template <typename T> kernel void gen(device T* x) {}\n\
               template [[host_name(\"gen_f32\")]] kernel void gen<float>(device float*);\n";
    assert_eq!(scan(src), set(&["gen_f32"]));
}

#[test]
fn an_instantiation_without_a_host_name_is_refused() {
    let err = scan_err(
        "template <typename T> kernel void gen(device T* x) {}\n\
         template kernel void gen<float>(device float*);\n",
    );
    assert!(err.message().contains("host_name"), "{err:?}");
    assert_eq!(
        err.exit_code(),
        kcov::EXIT_GAPS,
        "a census that cannot name a kernel is a finding, not a skip"
    );
}

#[test]
fn plain_and_macro_declared_kernels_are_both_counted() {
    let src = "kernel void alpha(device float* x) {}\n\
               #define NN_COOP_KERNEL(NAME, T, A, B, C, D) kernel void NAME(device T* x) {}\n\
               #define TN_NT_COOP_KERNEL(NAME, T, A, B) kernel void NAME(device T* x) {}\n\
               NN_COOP_KERNEL(beta, bfloat, 64, 64, 4, false)\n\
               TN_NT_COOP_KERNEL( gamma , float, 128, 64)\n";
    assert_eq!(scan(src), set(&["alpha", "beta", "gamma"]));
}

#[test]
fn a_macro_bodys_parameter_is_not_counted_as_a_kernel() {
    // NAME is the macro parameter, and counting it invents a kernel that is in no
    // metallib. The metallib cross-check caught this for real when the decode
    // kernels landed.
    let src = "#define MK_KERNEL(NAME, D) \\\nkernel void NAME(device float* x) {}\nMK_KERNEL(real_one, 128)\n";
    assert_eq!(scan(src), set(&["real_one"]));
}

#[test]
fn a_source_the_preprocessor_rejects_is_a_failed_census_never_a_smaller_one() {
    let err = scan_err(
        "kernel void alpha(device float* x) {}\nWEIRD_NEW_KERNEL(\"delta\", bfloat)\n#error not a kernel source\n",
    );
    assert!(err.message().contains("does not preprocess"), "{err:?}");
}

#[test]
fn macro_families_that_stamp_kernels_under_unexpected_names_are_expanded() {
    // A name not ending in KERNEL, `##` pasting, and a macro that stamps through
    // another macro: the shapes of the 29 qwen35 kernels the old scan missed.
    let src = "#define STAMP(NAME, T) \\\nkernel void NAME(device T* x) {}\n\
               #define STAMP_RULE(R, T) \\\n    STAMP(fam_##R##_a, T) \\\n    STAMP(fam_##R##_b, T)\n\
               STAMP(plain_stamp, float)\n\
               STAMP_RULE(kahan, bfloat)\n\
               STAMP_RULE(sr,\n           float)\n";
    assert_eq!(
        scan(src),
        set(&["plain_stamp", "fam_kahan_a", "fam_kahan_b", "fam_sr_a", "fam_sr_b"])
    );
}

#[test]
fn an_empty_kernel_directory_is_refused_not_read_as_zero() {
    let d = tmp("empty");
    let err = kcov::inventory(d.path(), Some(Path::new("/nonexistent")), false).expect_err("accepted");
    assert!(err.message().contains("no kernels found"), "{err:?}");
}

#[test]
fn a_directory_of_sources_that_declare_nothing_is_refused_not_read_as_zero() {
    let d = tmp("nothing");
    write(&d.path().join("a.metal"), "// no kernels here\n");
    let err = kcov::inventory(d.path(), Some(Path::new("/nonexistent")), false).expect_err("accepted");
    assert!(err.message().contains("no kernels found"), "{err:?}");
}

#[test]
fn a_binary_that_dumps_no_metallib_fails_naming_the_rebuild() {
    let d = tmp("stale-dump");
    let err = kcov::dump_embedded_metallib(Path::new("/usr/bin/true"), d.path()).expect_err("accepted");
    assert!(err.message().contains("cargo build --release --bins"), "{err:?}");
    assert_eq!(err.exit_code(), kcov::EXIT_CANNOT_RUN);
}

#[test]
fn a_required_metallib_cross_check_fails_closed_when_the_library_is_absent() {
    // Reporting source coverage without checking the compiled library used to
    // print the same PASS when the library or metal-nm was unavailable. Library
    // consumers may request a source-only census; the CLI's coverage measurement
    // must fail closed.
    let d = tmp("required");
    write(&d.path().join("a.metal"), "kernel void alpha(device float* x) {}\n");
    let err = kcov::inventory(d.path(), Some(Path::new("/definitely/missing.metallib")), true).expect_err("accepted");
    assert!(err.message().contains("metallib missing"), "{err:?}");
    assert_eq!(err.exit_code(), 77);
}

#[test]
fn a_source_scan_that_disagrees_with_the_compiled_metallib_is_wrong_whichever_way() {
    let lib = Path::new(tessl::metallib_path());
    let d = tmp("disagree");
    write(
        &d.path().join("a.metal"),
        "kernel void not_a_real_kernel(device float* x) {}\n",
    );
    let err = kcov::inventory(d.path(), Some(lib), false).expect_err("accepted");
    assert!(
        err.message().contains("disagrees with the compiled metallib"),
        "{err:?}"
    );
    assert!(
        err.message().contains("not_a_real_kernel"),
        "names the phantom: {err:?}"
    );
    assert_eq!(err.exit_code(), kcov::EXIT_GAPS);
}

#[test]
fn the_real_trees_census_agrees_with_the_library_the_binaries_embed() {
    // The library the runtime embeds, dumped by a bench binary the way the
    // measurement reads it, must be the build artifact the census is compared to.
    let d = tmp("real");
    let dumped = kcov::dump_embedded_metallib(Path::new(env!("CARGO_BIN_EXE_bench_gemm_sweep")), d.path())
        .expect("bench_gemm_sweep --dump-metallib");
    let exported = kcov::metallib_symbols(Some(&dumped), true)
        .expect("metal-nm")
        .expect("symbols");
    let census = kcov::inventory(&kcov::kernel_dir(), Some(&dumped), true).expect("the real tree's census");
    assert!(
        census.names.len() >= 80,
        "real inventory looks wrong: {} names",
        census.names.len()
    );
    assert!(
        census.names.contains("matmul2d_tensorops_bf16_f32_64x64_sg4"),
        "macro-declared kernels are included"
    );
    assert_eq!(census.names, exported, "the census is exactly what the library exports");
    let union: BTreeSet<String> = census.by_file.values().flatten().cloned().collect();
    assert_eq!(union, census.names, "every kernel belongs to a file");
}

// ------------------------------------------------------------- the entry scanner

fn entries(text: &str) -> Result<BTreeSet<String>, Failure> {
    kcov::entry_points(text, "t.metal")
}

#[test]
fn the_entry_scanner_follows_the_regex_it_replaced() {
    // `\bkernel`: the word must start at a boundary.
    assert_eq!(
        entries("mykernel void nope(); kernel void yes();").unwrap(),
        set(&["yes"])
    );
    // `kernel\s+void\s+`: any whitespace, but at least one on each side.
    assert_eq!(entries("kernel\n\tvoid   spaced();").unwrap(), set(&["spaced"]));
    assert!(entries("kernelvoid x(); kernel voidx();").unwrap().is_empty());
    // Several attributes before the keyword; the host_name may be any of them.
    assert_eq!(
        entries("[[max_total_threads_per_threadgroup(64)]] [[host_name( \"a b\" )]] kernel void inner();").unwrap(),
        set(&["a b"])
    );
    // An empty host_name is not a name.
    assert_eq!(
        entries("[[host_name(\"\")]] kernel void inner();").unwrap(),
        set(&["inner"])
    );
    // A malformed attribute list is not a prefix: the kernel is still a kernel.
    assert_eq!(entries("[[unclosed kernel void k1();").unwrap(), set(&["k1"]));
    // `template <>` has no parameter, so it is a (full) specialisation naming a kernel.
    assert_eq!(
        entries("template <> [[host_name(\"spec\")]] kernel void f<>();").unwrap(),
        set(&["spec"])
    );
    // `template <typename T>` is a generic definition even with a host_name on it.
    assert!(
        entries("template <typename T> [[host_name(\"g\")]] kernel void gen(T* x);")
            .unwrap()
            .is_empty()
    );
    // A template prefix that is not followed by a kernel does not swallow a later one.
    assert_eq!(
        entries("template <typename T> struct S {}; kernel void after();").unwrap(),
        set(&["after"])
    );
    // Identifiers stop at the first non-identifier character.
    assert_eq!(
        entries("kernel void name_1(device float* x)").unwrap(),
        set(&["name_1"])
    );
    // Non-ASCII text in the preprocessed output is not a reason to panic.
    assert_eq!(
        entries("// caf\u{e9} \u{1f600}\nkernel void ok();").unwrap(),
        set(&["ok"])
    );
    assert!(entries("").unwrap().is_empty());
}

// ----------------------------------------------------------- build.rs flags

const BUILD_RS: &str = r#"
fn metal_args(sdk: &str, src: &Path, metal_std: &str, werror: bool) -> Vec<String> {
    let mut args: Vec<String> = [
        &format!("-std={metal_std}")[..],
        "-O2",
        "-fmetal-math-mode=fast",
        "-isysroot",
        sdk,
        "-mmacosx-version-min=26.0",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    args
}
"#;

#[test]
fn the_preprocessor_flags_are_read_from_build_rs() {
    let flags = kcov::build_rs_flags(BUILD_RS, "/sdk/path").unwrap();
    assert_eq!(
        flags,
        [
            "-O2",
            "-fmetal-math-mode=fast",
            "-isysroot",
            "/sdk/path",
            "-mmacosx-version-min=26.0"
        ]
    );
    // And from the real file, so a refactor of build.rs that this reader cannot
    // follow fails here rather than in a measurement.
    let real = std::fs::read_to_string(kcov::crate_root().join("build.rs")).unwrap();
    let flags = kcov::build_rs_flags(&real, "/sdk/path").unwrap();
    assert!(
        flags.contains(&"-isysroot".to_string()) && flags.contains(&"/sdk/path".to_string()),
        "{flags:?}"
    );
    assert!(
        !flags.iter().any(|f| f.starts_with("-std=")),
        "-std is chosen per compile: {flags:?}"
    );
}

#[test]
fn a_build_rs_this_reader_cannot_follow_is_could_not_run() {
    for (label, text) in [
        ("no metal_args", "fn other() {}"),
        ("no list", "fn metal_args() { let x = 1; }"),
        (
            "an unterminated list",
            "fn metal_args() { let mut args: Vec<String> = [ \"-O2\", ",
        ),
        (
            "no -isysroot",
            "fn metal_args() { let mut args: Vec<String> = [ \"-O2\", sdk ]\n.iter()",
        ),
        (
            "no sdk",
            "fn metal_args() { let mut args: Vec<String> = [ \"-O2\", \"-isysroot\" ]\n.iter()",
        ),
    ] {
        let err = kcov::build_rs_flags(text, "/sdk")
            .err()
            .unwrap_or_else(|| panic!("{label}: accepted"));
        assert_eq!(err.exit_code(), kcov::EXIT_CANNOT_RUN, "{label}: {err:?}");
    }
    // `sdk` inside another word or a string is not the SDK path.
    let flags = kcov::build_rs_flags(
        "fn metal_args() { let mut args: Vec<String> = [ \"-isysroot\", sdk, \"-Dsdk_x\", sdk_other, ]\n.iter()",
        "/sdk",
    )
    .unwrap();
    assert_eq!(flags, ["-isysroot", "/sdk", "-Dsdk_x"]);
}

// ----------------------------------------------------------------- freshness

fn at(secs: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(secs)
}

fn touch(path: &Path, secs: u64) {
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(at(secs))
        .unwrap();
}

fn mini_crate(tag: &str) -> (kcov::TempDir, PathBuf) {
    let d = tmp(tag);
    for rel in ["build.rs", "kernels/a.metal", "kernels/a.h"] {
        write(&d.path().join(rel), "x");
    }
    let binary = d.path().join("bin");
    write(&binary, "x");
    (d, binary)
}

#[test]
fn a_binary_older_than_a_kernel_source_is_refused() {
    let (d, binary) = mini_crate("stale");
    touch(&binary, 1000);
    for rel in ["build.rs", "kernels/a.metal", "kernels/a.h"] {
        for other in ["build.rs", "kernels/a.metal", "kernels/a.h"] {
            touch(&d.path().join(other), 500);
        }
        touch(&d.path().join(rel), 2000);
        let err = kcov::require_fresh(&binary, d.path())
            .err()
            .unwrap_or_else(|| panic!("{rel}: accepted"));
        assert!(err.message().contains("cargo build --release --bins"), "{rel}: {err:?}");
        assert_eq!(err.exit_code(), kcov::EXIT_CANNOT_RUN);
    }
}

#[test]
fn a_binary_newer_than_every_input_is_accepted() {
    let (d, binary) = mini_crate("fresh");
    for rel in ["build.rs", "kernels/a.metal", "kernels/a.h"] {
        touch(&d.path().join(rel), 500);
    }
    touch(&binary, 1000);
    kcov::require_fresh(&binary, d.path()).expect("fresh");
    // A header or source that is not a kernel input does not make it stale.
    write(&d.path().join("kernels/notes.txt"), "later");
    touch(&d.path().join("kernels/notes.txt"), 5000);
    kcov::require_fresh(&binary, d.path()).expect("an unrelated file is not an input");
}

// -------------------------------------------------------------------- suites

/// A `target/release` of fake benchmark binaries: shell scripts that print
/// `trace` on stderr (and `--dump-metallib` the real library, as the real
/// `bench_gemm_sweep` does).
fn fake_release(tag: &str, trace: &str) -> kcov::TempDir {
    let d = tmp(tag);
    for name in [
        "bench_gemm_sweep",
        "bench_flash_attn",
        "bench_nn_kernels",
        "bench_encoder_attn",
        "bench_embedgemma2",
        "bench_embedgemma2_audio",
        "bench_embedgemma2_vision",
        "bench_gemm_variants",
        "bench_gemm_routes",
        "bench_qwen35_layers",
        "bench_qwen35_train",
    ] {
        fake_bin(
            &d.path().join("release"),
            name,
            &format!("echo 'KERNEL_TRACE {trace}' >&2"),
        );
    }
    d
}

fn fake_bin(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    write(
        &p,
        &format!(
            "#!/bin/sh\ncase \"$1\" in --dump-metallib) cp '{}' \"$2\"; exit 0;; esac\n{body}\n",
            tessl::metallib_path()
        ),
    );
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p
}

fn one_suite(name: &'static str, argv: &[&'static str], env: &[(&'static str, &str)]) -> kcov::Suite {
    kcov::Suite {
        name,
        argv: argv.to_vec(),
        env: env.iter().map(|(k, v)| (*k, v.to_string())).collect(),
    }
}

fn run_one(bin: &Path, s: kcov::Suite) -> Traced {
    kcov::run_suites(&[s], bin, bin, &kcov::crate_root(), 60)
}

#[test]
fn a_suite_label_keeps_every_argument_but_scratch_paths_and_cuts_a_config_to_its_name() {
    let d = tmp("label-args");
    let bin = d.path().join("release");
    fake_bin(&bin, "bench_qwen35_train", "echo 'KERNEL_TRACE a' >&2");
    let t = kcov::run_suites(
        &[
            one_suite("bench_qwen35_train", &["64", "--check-only"], &[]),
            one_suite("bench_qwen35_train", &["64", "--check-only", "--bf16"], &[]),
            one_suite(
                "bench_qwen35_train",
                &[
                    "64",
                    "--step=64",
                    "--random-config={crate}/tests/fixtures/qwen35_train/config.json",
                ],
                &[],
            ),
        ],
        &bin,
        &bin,
        &kcov::crate_root(),
        60,
    );
    assert!(t.failures.is_empty(), "{:?}", t.failures);
    let labels: Vec<&str> = t.per_suite.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        labels,
        [
            "bench_qwen35_train 64 --check-only",
            "bench_qwen35_train 64 --check-only --bf16",
            "bench_qwen35_train 64 --step=64 --random-config=config.json",
        ],
        "suites that differ past their first argument keep their own rows"
    );
}

#[test]
fn a_suite_is_labelled_by_its_binary_arguments_and_knobs_and_its_trace_is_the_last_line() {
    let d = tmp("label");
    let bin = d.path().join("release");
    fake_bin(
        &bin,
        "bench_flash_attn",
        "echo 'KERNEL_TRACE stale_a' >&2\necho 'KERNEL_TRACE b,a,,c' >&2",
    );
    let t = run_one(
        &bin,
        one_suite(
            "bench_flash_attn",
            &["--dump-parity", "{tmp}/x"],
            &[
                ("BENCH_ITERS", "1"),
                ("BENCH_ATTN_ROWS_R", "16"),
                ("TESSL_ATTN_TILED", "1"),
                ("BENCH_ATTN_CFGS", "x"),
            ],
        ),
    );
    assert!(t.failures.is_empty(), "{:?}", t.failures);
    assert_eq!(t.per_suite.len(), 1);
    assert_eq!(
        t.per_suite[0].0, "bench_flash_attn --dump-parity BENCH_ATTN_ROWS_R=16 TESSL_ATTN_TILED=1",
        "only the TESSL_* and the named attention knobs are in the label"
    );
    assert_eq!(
        t.per_suite[0].1,
        set(&["a", "b", "c"]),
        "the last trace line wins, empty names dropped"
    );
    assert_eq!(t.traced, set(&["a", "b", "c"]));
}

#[test]
fn every_way_a_suite_can_fail_is_a_failure_not_an_empty_trace() {
    let d = tmp("fail");
    let bin = d.path().join("release");
    std::fs::create_dir_all(&bin).unwrap();

    let t = run_one(&bin, one_suite("bench_nn_kernels", &[], &[]));
    assert!(
        t.failures[0].contains("bench_nn_kernels: not built (cargo build --release --bins)"),
        "{:?}",
        t.failures
    );

    fake_bin(&bin, "bench_nn_kernels", "echo 'boom on stderr' >&2\nexit 3");
    let t = run_one(&bin, one_suite("bench_nn_kernels", &["--x"], &[]));
    assert!(
        t.failures[0].starts_with("bench_nn_kernels --x: exited 3\n") && t.failures[0].contains("boom on stderr"),
        "{:?}",
        t.failures
    );

    // An empty trace and a missing trace must not read the same.
    fake_bin(&bin, "bench_nn_kernels", "echo 'no trace here' >&2");
    let t = run_one(&bin, one_suite("bench_nn_kernels", &[], &[]));
    assert!(
        t.failures[0].contains("emitted no KERNEL_TRACE line"),
        "{:?}",
        t.failures
    );
    assert!(t.traced.is_empty() && t.per_suite.is_empty());

    fake_bin(&bin, "bench_nn_kernels", "echo 'KERNEL_TRACE ' >&2");
    let t = run_one(&bin, one_suite("bench_nn_kernels", &[], &[]));
    assert!(
        t.failures.is_empty() && t.per_suite[0].1.is_empty(),
        "a present, empty trace is a trace: {:?}",
        t.failures
    );

    // A stale binary is refused, not run.
    let stale = fake_bin(&bin, "bench_nn_kernels", "echo 'KERNEL_TRACE a' >&2");
    touch(&stale, 1000);
    let t = run_one(&bin, one_suite("bench_nn_kernels", &[], &[]));
    assert!(t.failures[0].contains("older than kernels/*.metal"), "{:?}", t.failures);
    assert!(t.traced.is_empty());

    // And a hang is bounded.
    fake_bin(&bin, "bench_nn_kernels", "sleep 30");
    let t = kcov::run_suites(
        &[one_suite("bench_nn_kernels", &[], &[])],
        &bin,
        &bin,
        &kcov::crate_root(),
        1,
    );
    assert!(t.failures[0].contains("timed out after 1s"), "{:?}", t.failures);
}

#[test]
fn the_suite_list_reaches_the_breadth_it_claims() {
    let s = kcov::suites();
    // 2 sweep runs, 1 base + 9 decode (R x chunk) + 9 rows (R x SGT) + 1 tiled
    // attention runs, nn kernels, variants, routes, the encoder attention, 3 EmbedGemma 2 kernel
    // lanes (text, audio, vision), 6 Qwen3.5 inference and 5
    // Qwen3.5 training runs.
    assert_eq!(s.len(), 2 + 1 + 9 + 9 + 1 + 3 + 4 + 6 + 5);
    // Every suite is a probe, not a measurement: bounded by its environment or,
    // for the Qwen3.5 benches, by its arguments.
    let bounded_by_args = |a: &str| {
        a == "--iters=1"
            || a == "--check-only"
            || a == "--adamw-matrix"
            || a == "--attn-tiles"
            || a.starts_with("--step=")
            || a.starts_with("--paired-")
    };
    for x in &s {
        assert!(
            x.env
                .iter()
                .any(|(k, _)| *k == "BENCH_ITERS" || *k == "BENCH_PARITY_SEEDS" || *k == "BENCH_REPS")
                || x.argv.iter().any(|a| bounded_by_args(a)),
            "{} {:?} is unbounded",
            x.name,
            x.argv
        );
        // A `{crate}` path names a file that exists.
        for a in &x.argv {
            if let Some((_, rel)) = a.split_once("{crate}/") {
                assert!(kcov::crate_root().join(rel).is_file(), "{a}");
            }
        }
    }
    let attn: Vec<&kcov::Suite> = s.iter().filter(|x| x.name == "bench_flash_attn").collect();
    let combos: BTreeSet<(String, String)> = attn
        .iter()
        .filter_map(|x| {
            let get = |k: &str| x.env.iter().find(|(n, _)| *n == k).map(|(_, v)| v.clone());
            Some((get("BENCH_ATTN_DECODE_R")?, get("BENCH_ATTN_DECODE_CHUNK")?))
        })
        .collect();
    assert_eq!(
        combos.len(),
        9,
        "every decode (lanes-per-key x chunk) pair is its own kernel"
    );
    // Every suite names a bench binary that exists as a source.
    for x in &s {
        assert!(
            kcov::crate_root()
                .join("src/bin")
                .join(format!("{}.rs", x.name))
                .is_file(),
            "{}",
            x.name
        );
    }
}

// -------------------------------------------------------------------- report

fn census(files: &[(&str, &[&str])]) -> Census {
    let mut names = BTreeSet::new();
    let mut by_file = std::collections::BTreeMap::new();
    for (f, ks) in files {
        names.extend(ks.iter().map(|k| k.to_string()));
        by_file.insert(f.to_string(), set(ks));
    }
    Census { names, by_file }
}

fn traced(suites: &[(&str, &[&str])]) -> Traced {
    let per_suite: Vec<(String, BTreeSet<String>)> = suites.iter().map(|(k, v)| (k.to_string(), set(v))).collect();
    let traced = per_suite.iter().flat_map(|(_, v)| v.iter().cloned()).collect();
    Traced {
        traced,
        per_suite,
        failures: vec![],
    }
}

#[test]
fn the_report_counts_names_the_uncovered_by_file_and_truncates_at_four() {
    let c = census(&[
        ("zeta", &["z1"]),
        ("alpha", &["a1", "a2", "a3", "a4", "a5", "a6"]),
        ("beta", &["b1"]),
    ]);
    let t = traced(&[("bench_x", &["a1", "b1"]), ("bench_y --flag", &["z1", "a1"])]);
    let r = kcov::report(&c, &t, &BTreeMap::new()).unwrap();
    // `print(f"  {suite:<40}{len(got):>4} kernels")` and
    // `print(f"  {f:<24}{len(gap):>3}  {names}")`, spelled out.
    let expected = format!(
        "kernel entry points : 8\ndispatched by a bench: 3  (38%)\nallowlisted          : 0\nremaining            : 5\n\n  {:<40}{:>4} kernels\n  {:<40}{:>4} kernels\n\nuncovered and not allowlisted, by file:\n  {:<24}{:>3}  a2, a3, a4, a5 ...\n",
        "bench_x", 2, "bench_y --flag", 2, "alpha", 5
    );
    assert_eq!(r.text, expected);
    assert_eq!(
        r.problems,
        ["5 kernel(s) have no timing lane and are not allowlisted: ['a2', 'a3', 'a4', 'a5', 'a6']"]
    );
    assert_eq!(
        r.doc.dumps().unwrap(),
        "{\n  \"entry_points\": 8,\n  \"covered\": [\n    \"a1\",\n    \"b1\",\n    \"z1\"\n  ],\n  \"missing\": [\n    \"a2\",\n    \"a3\",\n    \"a4\",\n    \"a5\",\n    \"a6\"\n  ],\n  \"allowlisted\": {},\n  \"unallowlisted_missing\": [\n    \"a2\",\n    \"a3\",\n    \"a4\",\n    \"a5\",\n    \"a6\"\n  ],\n  \"stale_allowlist\": [],\n  \"redundant_allowlist\": [],\n  \"per_suite\": {\n    \"bench_x\": [\n      \"a1\",\n      \"b1\"\n    ],\n    \"bench_y --flag\": [\n      \"a1\",\n      \"z1\"\n    ]\n  }\n}"
    );
}

#[test]
fn the_percentage_rounds_half_to_even_as_python_did() {
    // 100*c/n lands exactly on .5 at c = 1, n = 200: Python's '%.0f' gives 0, and
    // 3/200 gives 2 (1.5 -> 2), 5/200 gives 2 (2.5 -> 2).
    let names: Vec<String> = (0..200).map(|i| format!("k{i}")).collect();
    let all: Vec<&str> = names.iter().map(String::as_str).collect();
    let c = census(&[("f", &all)]);
    for (n, want) in [(1, "0%"), (3, "2%"), (5, "2%"), (7, "4%"), (200, "100%")] {
        let r = kcov::report(&c, &traced(&[("s", &all[..n])]), &BTreeMap::new()).unwrap();
        assert!(r.text.contains(&format!("({want})")), "{n}/200: {}", r.text);
    }
}

#[test]
fn a_kernel_traced_but_absent_from_the_census_means_the_census_is_wrong() {
    let c = census(&[("f", &["a", "b"])]);
    let t = traced(&[("s", &["a", "ghost"])]);
    let Err(err) = kcov::report(&c, &t, &BTreeMap::new()) else {
        panic!("accepted")
    };
    assert!(err.message().contains("absent from the source inventory"), "{err:?}");
    assert!(err.message().contains("['ghost']"), "{err:?}");
    assert_eq!(err.exit_code(), kcov::EXIT_GAPS);
}

#[test]
fn a_fully_covered_census_reports_no_gap_section() {
    let c = census(&[("f", &["a", "b"])]);
    let r = kcov::report(&c, &traced(&[("s", &["a", "b"])]), &BTreeMap::new()).unwrap();
    assert!(r.problems.is_empty(), "{:?}", r.problems);
    assert!(!r.text.contains("uncovered"), "{}", r.text);
    assert!(r.text.contains("(100%)"));
}

// ---------------------------------------------------------- the command line

/// The tool with `CARGO_TARGET_DIR` pointed at `target` and nothing else about
/// the caller's environment changed.
fn tool(target: &Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_kernel_coverage"));
    c.env("CARGO_TARGET_DIR", target);
    c
}

fn text(o: &std::process::Output) -> (String, String) {
    (
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

#[test]
fn no_binary_in_the_cargo_target_dir_cannot_run() {
    // CARGO_TARGET_DIR is honoured: an empty one has no bench binary.
    let empty = tmp("empty-target");
    let out = tool(empty.path()).arg("--check").output().unwrap();
    let (_, err) = text(&out);
    assert_eq!(out.status.code(), Some(kcov::EXIT_CANNOT_RUN), "{err}");
    assert!(
        err.contains(&empty.path().join("release").display().to_string()),
        "{err}"
    );
}

#[test]
fn a_missing_toolchain_cannot_run_and_never_falls_back_to_a_scan() {
    let out = tool(Path::new("/nonexistent"))
        .arg("--check")
        .env("PATH", "/nonexistent")
        .output()
        .unwrap();
    let (_, err) = text(&out);
    assert_eq!(out.status.code(), Some(kcov::EXIT_CANNOT_RUN), "{err}");
    assert!(err.contains("xcrun metal"), "{err}");
}

#[test]
fn an_unrunnable_check_has_its_own_exit_status() {
    // A missing explicit metallib: the check cannot run at all.
    let out = tool(Path::new("/nonexistent"))
        .args(["--check", "--metallib", "/nonexistent/x.metallib"])
        .output()
        .unwrap();
    let code = out.status.code().expect("exited");
    assert_eq!(code, kcov::EXIT_CANNOT_RUN, "{:?}", text(&out));
    assert_ne!(code, 0, "must never report success");
    // Gaps found is exit 1; the unrunnable case must not share that status, or a
    // CI wrapper cannot tell "uncovered" from "broken".
    assert_ne!(code, kcov::EXIT_GAPS, "could-not-run shares exit 1 with found-gaps");
}

#[test]
fn a_bad_command_line_exits_2() {
    for args in [
        &["--bogus"][..],
        &["--metallib"],
        &["--check=1"],
        &["--audit-allowlist=1"],
        &["--allowlist"],
        &["extra"],
    ] {
        let out = Command::new(env!("CARGO_BIN_EXE_kernel_coverage"))
            .args(args)
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(2), "{args:?}: {:?}", text(&out));
    }
    let out = Command::new(env!("CARGO_BIN_EXE_kernel_coverage"))
        .arg("--help")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0));
    assert!(text(&out)
        .0
        .contains("exit status: 0 ran and every kernel is dispatched"));
}

#[test]
fn the_command_line_reports_gaps_with_exit_1_only_under_check_and_writes_the_document() {
    let d = fake_release("cli-gaps", "rms_norm_bf16,add_inplace_f32");
    let out_json = d.path().join("cov.json");

    let out = tool(d.path()).arg("--out").arg(&out_json).output().unwrap();
    let (stdout, err) = text(&out);
    assert_eq!(
        out.status.code(),
        Some(0),
        "gaps without --check are a report, not a failure: {err}"
    );
    assert!(stdout.starts_with("kernel entry points : "), "{stdout}");
    assert!(stdout.contains("dispatched by a bench: 2  ("), "{stdout}");
    assert!(stdout.contains("uncovered and not allowlisted, by file:"), "{stdout}");
    let last = stdout.lines().last().unwrap_or_default();
    assert!(
        last.contains("kernel(s) have no timing lane and are not allowlisted: [") && last.ends_with(']'),
        "the gaps end stdout as a problem line: {stdout:?}"
    );
    assert!(!stdout.contains("PASS"), "{stdout}");
    let doc = std::fs::read_to_string(&out_json).unwrap();
    assert!(
        doc.starts_with("{\n  \"entry_points\": ") && !doc.ends_with('\n'),
        "json.dumps without a trailing newline"
    );
    assert!(
        doc.contains("\"covered\": [\n    \"add_inplace_f32\",\n    \"rms_norm_bf16\"\n  ]"),
        "{doc}"
    );

    let out = tool(d.path()).arg("--check").output().unwrap();
    let (_, err) = text(&out);
    assert_eq!(out.status.code(), Some(kcov::EXIT_GAPS), "{err}");
    assert!(err.contains("kernel(s) have no timing lane"), "{err}");
}

#[test]
fn the_command_line_refuses_a_trace_naming_a_kernel_the_census_does_not_have() {
    let d = fake_release("cli-stray", "rms_norm_bf16,not_a_real_kernel");
    let out = tool(d.path()).output().unwrap();
    let (_, err) = text(&out);
    assert_eq!(out.status.code(), Some(kcov::EXIT_GAPS), "{err}");
    assert!(
        err.contains("absent from the source inventory") && err.contains("not_a_real_kernel"),
        "{err}"
    );
}

#[test]
fn the_command_line_refuses_a_partial_sweep_and_lists_every_failed_suite() {
    let d = fake_release("cli-partial", "rms_norm_bf16");
    fake_bin(&d.path().join("release"), "bench_nn_kernels", "echo 'no trace' >&2");
    fake_bin(
        &d.path().join("release"),
        "bench_gemm_variants",
        "echo oops >&2\nexit 5",
    );
    let out = tool(d.path()).arg("--check").output().unwrap();
    let (stdout, err) = text(&out);
    assert_eq!(out.status.code(), Some(kcov::EXIT_CANNOT_RUN), "{err}");
    assert!(
        err.contains("ERROR  bench_nn_kernels: emitted no KERNEL_TRACE line"),
        "{err}"
    );
    assert!(
        err.contains("ERROR  bench_gemm_variants : exited 5") || err.contains("ERROR  bench_gemm_variants: exited 5"),
        "{err}"
    );
    assert!(err.contains("COULD NOT RUN: coverage run incomplete"), "{err}");
    assert!(stdout.is_empty(), "a partial sweep prints no percentage: {stdout:?}");
}

#[test]
fn a_failed_suite_quotes_the_tail_of_its_stderr_where_the_cause_is() {
    let d = tmp("stderr-tail");
    let bin = d.path().join("release");
    // 700 characters of progress noise, then the reason: the head alone would hide it.
    fake_bin(
        &bin,
        "bench_nn_kernels",
        "printf 'BEGIN%0700d\\n' 0 >&2\necho 'error: the real cause' >&2\nexit 3",
    );
    let t = run_one(&bin, one_suite("bench_nn_kernels", &[], &[]));
    assert_eq!(t.failures.len(), 1, "{:?}", t.failures);
    let f = &t.failures[0];
    assert!(f.contains("exited 3"), "{f}");
    assert!(f.ends_with("error: the real cause"), "{f}");
    assert!(!f.contains("BEGIN"), "only the last 600 characters are quoted: {f}");
}

#[test]
fn the_command_line_refuses_a_stale_bench_binary() {
    let d = fake_release("cli-stale", "rms_norm_bf16");
    touch(&d.path().join("release/bench_flash_attn"), 1000);
    let out = tool(d.path()).output().unwrap();
    let (_, err) = text(&out);
    assert_eq!(out.status.code(), Some(kcov::EXIT_CANNOT_RUN), "{err}");
    assert!(err.contains("older than kernels/*.metal"), "{err}");
}

// ------------------------------------------------------------------ allowlist

/// The committed allowlist stands in for a bench lane only while it is true.
fn allow_crate(tag: &str, allowlist_text: &str) -> (kcov::TempDir, PathBuf) {
    let d = tmp(tag);
    write(&d.path().join("tests/t.rs"), "// the kernel's only caller\n");
    let path = d.path().join("allow.txt");
    write(&path, allowlist_text);
    (d, path)
}

fn load(tag: &str, text: &str) -> Result<BTreeMap<String, String>, Failure> {
    let (d, path) = allow_crate(tag, text);
    kcov::load_allowlist(&path, d.path())
}

#[test]
fn an_allowlist_entry_names_its_class_and_an_existing_file() {
    let got = load(
        "allow-ok",
        "# comment\nk_one: test-only: tests/t.rs the only caller\n\nk_two: probe: tests/t.rs  # trailing comment\n",
    )
    .unwrap();
    let want: BTreeMap<String, String> = [
        ("k_one", "test-only: tests/t.rs the only caller"),
        ("k_two", "probe: tests/t.rs"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    assert_eq!(got, want);
}

#[test]
fn a_bare_allowlist_class_is_not_a_reason() {
    for line in [
        "k: test-only\n",
        "k: test-only:\n",
        "k:\n",
        "k\n",
        "2k: probe: tests/t.rs\n",
    ] {
        let err = load("allow-bare", line).expect_err(line);
        assert_eq!(err.exit_code(), kcov::EXIT_GAPS, "{line:?}: {err:?}");
    }
}

#[test]
fn an_allowlist_reason_must_name_a_file_that_exists() {
    let err = load("allow-gone", "k: test-only: tests/gone.rs it used to be here\n").unwrap_err();
    assert!(err.message().contains("exist"), "{err:?}");
    assert_eq!(err.exit_code(), kcov::EXIT_GAPS);
}

#[test]
fn an_unknown_allowlist_class_is_refused() {
    let err = load("allow-class", "k: whim: tests/t.rs\n").unwrap_err();
    assert!(err.message().contains("class 'whim'"), "{err:?}");
}

#[test]
fn a_kernel_allowlisted_twice_is_refused() {
    let err = load("allow-twice", "k: probe: tests/t.rs\nk: probe: tests/t.rs\n").unwrap_err();
    assert!(err.message().contains("allow.txt:2: k is listed twice"), "{err:?}");
}

#[test]
fn an_allowlist_that_is_not_utf8_is_malformed() {
    let (d, path) = allow_crate("allow-bytes", "");
    std::fs::write(&path, b"k: probe: tests/t.rs \xff\n").unwrap();
    let err = kcov::load_allowlist(&path, d.path()).unwrap_err();
    assert_eq!(err.exit_code(), kcov::EXIT_GAPS, "{err:?}");
    assert!(err.message().contains("not UTF-8"), "{err:?}");
}

#[test]
fn the_committed_allowlist_loads() {
    kcov::load_allowlist(&kcov::allowlist_path(), &kcov::crate_root()).expect("bench/kernel_coverage_allowlist.txt");
}

#[test]
fn a_missing_allowlist_is_empty() {
    let got = kcov::load_allowlist(Path::new("/nonexistent/allow.txt"), &kcov::crate_root()).unwrap();
    assert!(got.is_empty());
}

#[test]
fn stale_and_redundant_allowlist_entries_are_found() {
    let allowed: BTreeMap<String, String> = ["gone", "b", "c"]
        .into_iter()
        .map(|k| (k.to_string(), "x".to_string()))
        .collect();
    let (stale, redundant) = kcov::audit_allowlist(&allowed, &set(&["a", "b", "c"]), &set(&["a", "c"]));
    assert_eq!((stale, redundant), (vec!["gone".to_string()], vec!["c".to_string()]));
    let consistent: BTreeMap<String, String> = [("b".to_string(), "x".to_string())].into_iter().collect();
    assert_eq!(
        kcov::audit_allowlist(&consistent, &set(&["a", "b"]), &set(&["a"])),
        (vec![], vec![])
    );
}

#[test]
fn the_report_subtracts_the_allowlist_and_names_stale_and_redundant_entries() {
    let c = census(&[("f", &["a", "b", "c", "d"])]);
    let t = traced(&[("s", &["a", "c"])]);
    let allowed: BTreeMap<String, String> = [("b", "test-only: tests/t.rs"), ("c", "probe: x"), ("gone", "probe: x")]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let r = kcov::report(&c, &t, &allowed).unwrap();
    assert!(
        r.text.contains("allowlisted          : 1\nremaining            : 1\n"),
        "{}",
        r.text
    );
    assert!(
        r.text.contains("uncovered and not allowlisted, by file:\n  f"),
        "{}",
        r.text
    );
    assert!(r.text.contains("  1  d\n"), "only d is a gap: {}", r.text);
    assert_eq!(
        r.problems,
        [
            "1 kernel(s) have no timing lane and are not allowlisted: ['d']",
            "1 allowlist entr(ies) are not kernels any more: ['gone']",
            "1 allowlisted kernel(s) are dispatched by a bench; remove them from the allowlist: ['c']",
        ]
    );
    let doc = r.doc.dumps().unwrap();
    assert!(
        doc.contains("\"allowlisted\": {\n    \"b\": \"test-only: tests/t.rs\"\n  }"),
        "{doc}"
    );
    assert!(doc.contains("\"unallowlisted_missing\": [\n    \"d\"\n  ]"), "{doc}");

    // Every gap allowlisted and nothing stale or redundant is a pass.
    let exact: BTreeMap<String, String> = [("b", "probe: x"), ("d", "probe: x")]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let r = kcov::report(&c, &t, &exact).unwrap();
    assert!(r.problems.is_empty(), "{:?}", r.problems);
    assert!(!r.text.contains("uncovered"), "{}", r.text);
}

#[test]
fn a_malformed_allowlist_stops_the_command_line_before_any_suite_runs() {
    // An empty target would be "could not run" (77); the allowlist is read first.
    let empty = tmp("allow-cli-target");
    let (_d, path) = allow_crate("allow-cli", "k: whim: tests/t.rs\n");
    let out = tool(empty.path())
        .arg("--check")
        .arg("--allowlist")
        .arg(&path)
        .output()
        .unwrap();
    let (_, err) = text(&out);
    assert_eq!(out.status.code(), Some(kcov::EXIT_GAPS), "{err}");
    assert!(err.contains("class 'whim'"), "{err}");
}

#[test]
fn the_allowlist_audit_runs_no_suite_and_fails_only_on_a_stale_entry() {
    // No bench binary at all: the audit needs the census only.
    let empty = tmp("audit-target");
    let d = tmp("audit");
    let good = d.path().join("good.txt");
    write(
        &good,
        "rms_norm_f32: test-only: tests/kernel_coverage.rs the census test\n",
    );
    let out = tool(empty.path())
        .arg("--audit-allowlist")
        .arg("--allowlist")
        .arg(&good)
        .output()
        .unwrap();
    let (stdout, err) = text(&out);
    assert_eq!(out.status.code(), Some(0), "{err}");
    assert!(
        stdout.starts_with("allowlist: 1 entries, every one a kernel of the "),
        "{stdout}"
    );

    let stale = d.path().join("stale.txt");
    write(&stale, "not_a_kernel_name: probe: tests/kernel_coverage.rs\n");
    let out = tool(empty.path())
        .arg("--audit-allowlist")
        .arg("--allowlist")
        .arg(&stale)
        .output()
        .unwrap();
    let (_, err) = text(&out);
    assert_eq!(out.status.code(), Some(kcov::EXIT_GAPS), "{err}");
    assert!(err.contains("['not_a_kernel_name']"), "{err}");
}
