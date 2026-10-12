//! Build-script regression tests that do not require a GPU or Metal toolchain.
//!
//! A dependency build used to publish `default.metallib` into
//! `CARGO_MANIFEST_DIR`. Cargo's package-verification layout happened to be
//! special-cased, but registry, vendored, and ordinary path dependencies still
//! mutated their source and raced one another. These tests execute the real
//! build script with fake compiler tools from two concurrent build directories.

use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("tessl-build-artifact-contract-{}-{id}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir(&path).expect("create scratch directory");
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn write_executable(path: &Path, body: &str) {
    fs::write(path, body).expect("write fake tool");
    let mut permissions = fs::metadata(path).expect("stat fake tool").permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(path, permissions).expect("make fake tool executable");
}

fn source_snapshot(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn visit(base: &Path, dir: &Path, out: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in fs::read_dir(dir).expect("read source snapshot") {
            let path = entry.expect("read source entry").path();
            if path.is_dir() {
                visit(base, &path, out);
            } else {
                out.insert(
                    path.strip_prefix(base).expect("source-relative path").into(),
                    fs::read(&path).expect("read source file"),
                );
            }
        }
    }

    let mut files = BTreeMap::new();
    visit(root, root, &mut files);
    files
}

fn build_command(binary: &Path, manifest: &Path, out: &Path, tools: &Path) -> Command {
    let mut command = Command::new(binary);
    let inherited_path = std::env::var_os("PATH").unwrap_or_default();
    let mut path = tools.as_os_str().to_os_string();
    path.push(":");
    path.push(inherited_path);
    command
        .env("CARGO_MANIFEST_DIR", manifest)
        .env("OUT_DIR", out)
        .env("DEVELOPER_DIR", "/tmp")
        .env("PATH", path)
        .env_remove("DOCS_RS")
        .env_remove("TESSL_SKIP_AOT")
        .env_remove("METAL_RUNTIME_SKIP_AOT")
        .env_remove("TESSL_PREBUILT_METALLIB")
        .env_remove("TESSL_GEMM_TUNE")
        .env_remove("METAL_NATIVE_GEMM_TUNE")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("build output is UTF-8")
}

fn metadata_value<'a>(stdout: &'a str, prefix: &str) -> &'a str {
    stdout
        .lines()
        .find_map(|line| line.strip_prefix(prefix))
        .unwrap_or_else(|| panic!("missing {prefix:?} in build output:\n{stdout}"))
}

fn assert_normal_build(output: &Output, manifest: &Path, out: &Path) -> PathBuf {
    assert!(
        output.status.success(),
        "fake AOT build failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = stdout(output);
    assert_eq!(
        metadata_value(&stdout, "cargo:kernels="),
        manifest.join("kernels").to_string_lossy()
    );
    let embedded_path = PathBuf::from(metadata_value(&stdout, "cargo:rustc-env=TESSL_METALLIB="));
    if let Some(metadata_path) = stdout
        .lines()
        .find_map(|line| line.strip_prefix("cargo:metallib="))
        .map(PathBuf::from)
    {
        assert_eq!(metadata_path, embedded_path);
    }
    assert!(embedded_path.starts_with(out), "artifact escaped OUT_DIR");
    assert!(embedded_path.is_file(), "embedded artifact names no file");
    embedded_path
}

#[test]
fn concurrent_builds_never_mutate_dependency_source_and_publish_out_dir_metadata() {
    let scratch = Scratch::new();
    let manifest = scratch.0.join("registry-source/tessl-0.1.0");
    let kernels = manifest.join("kernels");
    fs::create_dir_all(&kernels).expect("create synthetic source");
    fs::write(
        kernels.join("matmul_tensorops.metal"),
        b"kernel void synthetic_tensorops() {}\n",
    )
    .expect("write required synthetic shader");

    let build_script = if let Some(path) = std::env::var_os("TESSL_BUILD_SCRIPT_UNDER_TEST") {
        PathBuf::from(path)
    } else {
        let path = scratch.0.join("tessl-build-script");
        let compile = Command::new("rustc")
            .arg("--edition=2021")
            .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("build.rs"))
            .arg("-o")
            .arg(&path)
            .output()
            .expect("compile tessl build script");
        assert!(
            compile.status.success(),
            "build.rs did not compile: {}",
            String::from_utf8_lossy(&compile.stderr)
        );
        path
    };

    let tools = scratch.0.join("fake-tools");
    fs::create_dir(&tools).expect("create fake tool directory");
    let tool = r#"#!/bin/sh
set -eu
write_out() {
  output=''
  while [ "$#" -gt 0 ]; do
    if [ "$1" = "-o" ]; then
      shift
      output="$1"
    fi
    shift
  done
  test -n "$output"
  printf 'synthetic-%s\n' "$(basename "$0")" > "$output"
}
case "$(basename "$0")" in
  xcrun)
    if [ "${1:-}" = "-f" ]; then
      printf '%s/%s\n' "$(dirname "$0")" "$2"
    else
      printf '/tmp\n'
    fi
    ;;
  metal)
    # The math mode must be an argument, not the compiler's implicit default.
    # Throughput kernels (this synthetic TensorOps source) keep contraction.
    case " $* " in
      *" -fmetal-math-mode=fast "*) ;;
      *) echo "missing -fmetal-math-mode=fast: $*" >&2; exit 1 ;;
    esac
    case " $* " in
      *" -ffp-contract=off "*)
        echo "throughput kernel must not disable contraction: $*" >&2
        exit 1
        ;;
    esac
    write_out "$@"
    ;;
  metallib)
    write_out "$@"
    ;;
esac
"#;
    for name in ["xcrun", "metal", "metallib"] {
        write_executable(&tools.join(name), tool);
    }

    let out_a = scratch.0.join("target/a/out");
    let out_b = scratch.0.join("target/b/out");
    fs::create_dir_all(&out_a).expect("create first OUT_DIR");
    fs::create_dir_all(&out_b).expect("create second OUT_DIR");
    let before = source_snapshot(&manifest);

    let first = build_command(&build_script, &manifest, &out_a, &tools)
        .spawn()
        .expect("spawn first dependency build");
    let second = build_command(&build_script, &manifest, &out_b, &tools)
        .spawn()
        .expect("spawn second dependency build");
    let first_output = first.wait_with_output().expect("wait for first build");
    let second_output = second.wait_with_output().expect("wait for second build");

    let first_metallib = assert_normal_build(&first_output, &manifest, &out_a);
    let second_metallib = assert_normal_build(&second_output, &manifest, &out_b);
    assert_ne!(first_metallib, second_metallib);
    assert_eq!(
        source_snapshot(&manifest),
        before,
        "a dependency build wrote into its registry/vendored source directory"
    );
    for output in [&first_output, &second_output] {
        let output = stdout(output);
        assert!(
            output.contains("cargo:rerun-if-env-changed=DOCS_RS"),
            "DOCS_RS changes must invalidate cached build-script output"
        );
        assert!(
            output.contains("cargo:metallib="),
            "the immutable artifact must be exposed as DEP_TESSL_METALLIB"
        );
    }

    // Offline reuse must be an explicit, absolute input and must publish the
    // same metadata contract without reconstructing a crate-root convention.
    let skip_out = scratch.0.join("target/skip/out");
    fs::create_dir_all(&skip_out).expect("create skip OUT_DIR");
    let skip = build_command(&build_script, &manifest, &skip_out, &tools)
        .env("TESSL_SKIP_AOT", "1")
        .env("TESSL_PREBUILT_METALLIB", &first_metallib)
        .output()
        .expect("run explicit offline build");
    assert!(
        skip.status.success(),
        "explicit offline build failed: {}",
        String::from_utf8_lossy(&skip.stderr)
    );
    let skip_stdout = stdout(&skip);
    assert_eq!(
        PathBuf::from(metadata_value(&skip_stdout, "cargo:metallib=")),
        first_metallib.canonicalize().expect("canonical prebuilt path")
    );
    assert_eq!(source_snapshot(&manifest), before);

    let missing = build_command(&build_script, &manifest, &skip_out, &tools)
        .env("TESSL_SKIP_AOT", "1")
        .output()
        .expect("run underspecified offline build");
    assert!(
        !missing.status.success(),
        "TESSL_SKIP_AOT accepted an implicit source-tree artifact"
    );
    assert!(
        String::from_utf8_lossy(&missing.stderr).contains("TESSL_PREBUILT_METALLIB"),
        "missing offline path produced the wrong diagnostic: {}",
        String::from_utf8_lossy(&missing.stderr)
    );
    assert_eq!(source_snapshot(&manifest), before);
}

/// A fake `metal` that answers from the source it is given: `#include
/// "shared.h"` reports a header warning while `shared.h` says `WARN`,
/// `WARN_SELF` reports one of the source's own, `FAIL_METAL4` fails under
/// `-std=metal4.0`, and `-Werror` turns any warning into a failure. It prints
/// the include chain and the `N warnings generated.` tally the way clang does,
/// and logs each invocation (source, then flags) to `$TESSL_FAKE_METAL_LOG`.
const DIAGNOSING_TOOLS: &str = r#"#!/bin/sh
set -eu
case "$(basename "$0")" in
  xcrun)
    if [ "${1:-}" = "-f" ]; then
      printf '%s/%s\n' "$(dirname "$0")" "$2"
    else
      printf '/tmp\n'
    fi
    exit 0
    ;;
esac
src=''
out=''
std=''
werror=0
prev=''
for a in "$@"; do
  case "$prev" in
    -c) src="$a" ;;
    -o) out="$a" ;;
  esac
  case "$a" in
    -std=*) std="${a#-std=}" ;;
    -Werror) werror=1 ;;
  esac
  prev="$a"
done
test -n "$out"
if [ "$(basename "$0")" = "metallib" ]; then
  printf 'synthetic-metallib\n' > "$out"
  exit 0
fi
printf '%s %s\n' "$(basename "$src")" "$*" >> "$TESSL_FAKE_METAL_LOG"
if [ "$std" = "metal4.0" ] && grep -q FAIL_METAL4 "$src"; then
  echo "$src:1:1: error: synthetic metal4.0-only failure" >&2
  exit 1
fi
n=0
if grep -q '#include "shared.h"' "$src" && grep -q WARN "$(dirname "$src")/shared.h"; then
  echo "In file included from $src:1:" >&2
  echo "$(dirname "$src")/shared.h:1:5: warning: synthetic header warning [-Wunused-variable]" >&2
  echo "int shared_unused;" >&2
  echo "    ^" >&2
  n=$((n + 1))
fi
if grep -q WARN_SELF "$src"; then
  echo "$src:2:5: warning: synthetic own warning [-Wunused-variable]" >&2
  echo "int WARN_SELF;" >&2
  echo "    ^" >&2
  n=$((n + 1))
fi
if [ "$n" -eq 1 ]; then echo "1 warning generated." >&2; fi
if [ "$n" -gt 1 ]; then echo "$n warnings generated." >&2; fi
if [ "$werror" -eq 1 ] && [ "$n" -gt 0 ]; then exit 1; fi
printf 'synthetic-air\n' > "$out"
"#;

fn log_lines(log: &Path) -> Vec<String> {
    fs::read_to_string(log)
        .unwrap_or_default()
        .lines()
        .map(str::to_string)
        .collect()
}

/// Every source's compiler warnings reach `cargo:warning=`, a header's once
/// however many sources include it, and a reused `.air` replays its own. A
/// header-only edit recompiles and drops a warning the header no longer
/// produces. `TESSL_METAL_WERROR=1` passes `-Wall -Werror` to every source,
/// the tune rig included, and fails rather than falling back to metal3.2;
/// without it a metal4.0 failure still falls back.
#[test]
fn metal_diagnostics_are_forwarded_cached_and_strict_on_request() {
    let scratch = Scratch::new();
    let manifest = scratch.0.join("crate");
    let kernels = manifest.join("kernels");
    fs::create_dir_all(kernels.join("tune")).expect("create synthetic kernels/tune");
    let write = |rel: &str, body: &str| fs::write(kernels.join(rel), body).expect("write synthetic kernel");
    write("matmul_tensorops.metal", "kernel void synthetic_tensorops() {}\n");
    write("tune/matmul_tensorops_tune.metal", "kernel void synthetic_tune() {}\n");
    write("shared.h", "WARN\n");
    write("a.metal", "#include \"shared.h\"\nWARN_SELF\n");
    write("b.metal", "#include \"shared.h\"\n");
    write("c.metal", "FAIL_METAL4\n");

    let build_script = scratch.0.join("tessl-build-script");
    let compile = Command::new("rustc")
        .arg("--edition=2021")
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("build.rs"))
        .arg("-o")
        .arg(&build_script)
        .output()
        .expect("compile tessl build script");
    assert!(
        compile.status.success(),
        "build.rs did not compile: {}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let tools = scratch.0.join("fake-tools");
    fs::create_dir(&tools).expect("create fake tool directory");
    for name in ["xcrun", "metal", "metallib"] {
        write_executable(&tools.join(name), DIAGNOSING_TOOLS);
    }
    let out = scratch.0.join("target/out");
    fs::create_dir_all(&out).expect("create OUT_DIR");
    let log = scratch.0.join("metal.log");
    let run = |env: &[(&str, &str)]| {
        let mut command = build_command(&build_script, &manifest, &out, &tools);
        command.env("TESSL_FAKE_METAL_LOG", &log).env_remove("TESSL_METAL_WERROR");
        for (k, v) in env {
            command.env(k, v);
        }
        command.output().expect("run build script")
    };
    let header = "synthetic header warning";
    let own = "synthetic own warning";

    // 1. Fresh build: both warnings, the header's once, no tally lines; the
    //    metal4.0-only failure falls back to metal3.2 as before.
    let first = run(&[]);
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let first_out = stdout(&first);
    assert_eq!(
        first_out.matches(header).count(),
        1,
        "a header warning is printed once, not once per includer:\n{first_out}"
    );
    assert_eq!(first_out.matches(own).count(), 1, "{first_out}");
    assert!(
        first_out.contains("cargo:warning=  In file included from"),
        "the include chain stays with its warning:\n{first_out}"
    );
    assert!(
        !first_out.contains("generated."),
        "the per-source tally is not a diagnostic:\n{first_out}"
    );
    assert!(
        first_out.contains("falling back to -std=metal3.2"),
        "{first_out}"
    );
    assert!(first_out.contains("cargo:rerun-if-env-changed=TESSL_METAL_WERROR"));
    let compiled = log_lines(&log).len();
    assert!(
        log_lines(&log)
            .iter()
            .any(|l| l.starts_with("c.metal ") && l.contains("-std=metal3.2")),
        "c.metal must have been compiled under the metal3.2 fallback"
    );
    assert!(
        log_lines(&log).iter().all(|l| !l.contains("-Werror")),
        "strict flags are opt-in"
    );

    // 2. Nothing changed: every `.air` is reused and still reports.
    let second = run(&[]);
    assert!(second.status.success());
    let second_out = stdout(&second);
    assert_eq!(log_lines(&log).len(), compiled, "an unchanged tree recompiled");
    assert_eq!(second_out.matches(header).count(), 1, "{second_out}");
    assert_eq!(second_out.matches(own).count(), 1, "{second_out}");

    // 3. A header-only edit recompiles its includers and the warning it no
    //    longer produces is not replayed.
    write("shared.h", "quiet\n");
    let third = run(&[]);
    assert!(third.status.success());
    let third_out = stdout(&third);
    assert!(log_lines(&log).len() > compiled, "a header edit must recompile");
    assert_eq!(
        third_out.matches(header).count(),
        0,
        "stale header warning replayed:\n{third_out}"
    );
    assert_eq!(third_out.matches(own).count(), 1, "{third_out}");

    // 4. Strict mode: a warning is an error and there is no metal3.2 fallback.
    write("shared.h", "WARN\n");
    let before = log_lines(&log).len();
    let strict = run(&[("TESSL_METAL_WERROR", "1"), ("TESSL_GEMM_TUNE", "1")]);
    assert!(!strict.status.success(), "a warning passed under TESSL_METAL_WERROR=1");
    assert!(
        String::from_utf8_lossy(&strict.stderr).contains("TESSL_METAL_WERROR"),
        "{}",
        String::from_utf8_lossy(&strict.stderr)
    );
    let strict_log = log_lines(&log).split_off(before);
    assert!(
        strict_log.iter().all(|l| !l.contains("-std=metal3.2")),
        "strict mode fell back to metal3.2: {strict_log:?}"
    );

    // 5. Strict mode on a clean tree builds, and every compile in it, the
    //    tune rig included, carried -Wall -Werror. The header edit changes
    //    every source's cache key, so step 4's strict `.air`s are not reused.
    write("shared.h", "quiet\n");
    write("a.metal", "#include \"shared.h\"\n");
    write("c.metal", "kernel void c() {}\n");
    let before = log_lines(&log).len();
    let clean = run(&[("TESSL_METAL_WERROR", "1"), ("TESSL_GEMM_TUNE", "1")]);
    assert!(
        clean.status.success(),
        "{}",
        String::from_utf8_lossy(&clean.stderr)
    );
    let clean_log = log_lines(&log).split_off(before);
    for src in [
        "matmul_tensorops.metal",
        "matmul_tensorops_tune.metal",
        "a.metal",
        "b.metal",
        "c.metal",
    ] {
        let line = clean_log
            .iter()
            .find(|l| l.starts_with(&format!("{src} ")))
            .unwrap_or_else(|| panic!("{src} was not compiled in strict mode: {clean_log:?}"));
        assert!(
            line.contains(" -Wall ") && line.contains(" -Werror "),
            "{src} compiled without -Wall -Werror: {line}"
        );
    }
}
