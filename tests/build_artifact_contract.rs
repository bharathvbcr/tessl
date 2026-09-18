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
        let path = std::env::temp_dir().join(format!(
            "tessl-build-artifact-contract-{}-{id}",
            std::process::id()
        ));
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
                    path.strip_prefix(base)
                        .expect("source-relative path")
                        .into(),
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
case "$(basename "$0")" in
  xcrun)
    if [ "${1:-}" = "-f" ]; then
      printf '%s/%s\n' "$(dirname "$0")" "$2"
    else
      printf '/tmp\n'
    fi
    ;;
  metal|metallib)
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
        first_metallib
            .canonicalize()
            .expect("canonical prebuilt path")
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
