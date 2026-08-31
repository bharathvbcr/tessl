//! AOT-compile `kernels/*.metal` → `default.metallib`.
//!
//! Targets Metal 4 / macOS 26 for TensorOps (`matmul2d`) with bf16 enabled in
//! the language dialect. The portable `simdgroup_matrix` kernel is always
//! included for A/B.
//!
//! Requires the Xcode Metal Toolchain component:
//!   `xcodebuild -downloadComponent MetalToolchain`
//!
//! Important: do **not** invoke `xcrun -sdk macosx metal` — the `-sdk` switch
//! breaks cryptex Metal Toolchain resolution on Xcode 26+. Use `xcrun metal`
//! plus an explicit `-isysroot`.
//!
//! `TESSL_SKIP_AOT` hazard: when set, this script skips compilation and
//! points `TESSL_METALLIB` at the crate-root `default.metallib`. That
//! file may be stale or missing — only use for intentional offline/CI skips
//! after a known-good metallib is already present at the crate root.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=TESSL_SKIP_AOT");
    println!("cargo:rerun-if-env-changed=METAL_RUNTIME_SKIP_AOT");
    println!("cargo:rerun-if-env-changed=TESSL_GEMM_TUNE");
    println!("cargo:rerun-if-env-changed=METAL_NATIVE_GEMM_TUNE");

    // CoreGraphics is required for MTLCreateSystemDefaultDevice.
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");

    // Legacy spelling still honoured: this one is set by hand in CI/offline runs.
    if env::var_os("TESSL_SKIP_AOT").is_some() || env::var_os("METAL_RUNTIME_SKIP_AOT").is_some() {
        println!("cargo:warning=TESSL_SKIP_AOT set; skipping metallib AOT");
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let crate_lib = manifest_dir.join("default.metallib");
        println!(
            "cargo:rustc-env=TESSL_METALLIB={}",
            crate_lib.display()
        );
        return;
    }

    ensure_developer_dir();

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernels_dir = manifest_dir.join("kernels");
    track_kernel_sources(&kernels_dir);

    // Canonical GEMM kernel sources. Dependents that build their own metallib
    // read this as `DEP_TESSL_KERNELS` (Cargo derives the name from `links`).
    println!("cargo:kernels={}", kernels_dir.display());

    let sdk = xcrun_stdout(&["--sdk", "macosx", "--show-sdk-path"]);
    let metal = resolve_metal();
    let metallib = resolve_metallib();

    let mut air_files: Vec<PathBuf> = Vec::new();

    // TensorOps kernels — Metal 4 dialect (macOS 26+ / MPP). Hard-fail: NAX GEMM
    // is the hot path; a simdgroup-only metallib is not acceptable.
    // The GEMM A/B rig (kernels/tune/) is 92 measurement-only kernels that
    // nothing dispatches at runtime. Linking it took the shipped metallib from
    // 0.22 MB to 1.09 MB, so it is opt-in. It lives in a subdirectory precisely
    // so the directory glob below cannot pick it up by accident.
    let want_tune = env::var_os("TESSL_GEMM_TUNE").is_some()
        || env::var_os("METAL_NATIVE_GEMM_TUNE").is_some();
    let mut tensorops_sources: Vec<PathBuf> = vec![kernels_dir.join("matmul_tensorops.metal")];
    if want_tune {
        tensorops_sources.push(kernels_dir.join("tune/matmul_tensorops_tune.metal"));
    }
    for src in &tensorops_sources {
        let name = src.file_name().and_then(|n| n.to_str()).unwrap_or("<unnamed>");
        if !src.exists() {
            panic!(
                "required TensorOps source missing: {}; Metal 4 / macOS 26 toolchain required",
                src.display()
            );
        }
        let air = out_dir.join(format!("{}.air", src.file_stem().unwrap().to_string_lossy()));
        let status = Command::new(&metal)
            .args([
                "-std=metal4.0",
                "-O2",
                "-isysroot",
                &sdk,
                "-mmacosx-version-min=26.0",
                "-c",
            ])
            .arg(src)
            .arg("-o")
            .arg(&air)
            .status()
            .unwrap_or_else(|e| panic!("failed to spawn metal for {name}: {e}"));
        if !status.success() {
            panic!(
                "{name} failed to compile (need Metal 4 / macOS 26 SDK + MetalToolchain); \
                 refusing simdgroup-only metallib"
            );
        }
        air_files.push(air);
    }
    println!("cargo:rustc-cfg=metal_runtime_tensorops");

    // All other .metal sources (simdgroup GEMM + util kernels).
    let skip: &[&str] = &["matmul_tensorops.metal"];
    let mut others: Vec<PathBuf> = fs::read_dir(&kernels_dir)
        .unwrap_or_else(|e| panic!("read kernels/: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("metal")
                && !skip
                    .iter()
                    .any(|s| p.file_name().and_then(|n| n.to_str()) == Some(*s))
        })
        .collect();
    others.sort();

    for src in &others {
        let stem = src.file_stem().unwrap().to_string_lossy();
        let air = out_dir.join(format!("{stem}.air"));
        let ok_m4 = try_metal_compile(&metal, &sdk, src, &air, "metal4.0");
        if !ok_m4 {
            println!(
                "cargo:warning={} failed under -std=metal4.0; falling back to -std=metal3.2 \
                 (shader dialect only; encode remains Metal 4)",
                src.file_name().and_then(|n| n.to_str()).unwrap_or(&stem)
            );
            run(
                Command::new(&metal)
                    .args([
                        "-std=metal3.2",
                        "-O2",
                        "-isysroot",
                        &sdk,
                        "-mmacosx-version-min=26.0",
                        "-c",
                    ])
                    .arg(src)
                    .arg("-o")
                    .arg(&air),
                &format!("metal compile {} (metal3.2 fallback)", src.display()),
            );
        }
        air_files.push(air);
    }

    // Metal can retain file-backed library data after loading. Never relink a
    // pathname baked into a prior binary: each build owns an immutable artifact.
    let build_id = format!("{}-{}", std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before Unix epoch").as_nanos());
    let metallib_out = out_dir.join(format!("default-{build_id}.metallib"));
    fs::File::create_new(&metallib_out).expect("reserve unique metallib output");
    let mut link = Command::new(&metallib);
    for air in &air_files {
        link.arg(air);
    }
    link.arg("-o").arg(&metallib_out);
    run(&mut link, "metallib link");

    // Offline compatibility copy at the crate root, for the TESSL_SKIP_AOT path.
    //
    // This writes outside OUT_DIR, which Cargo forbids while verifying a package
    // ("Source directory was modified by build.rs during cargo publish") — it
    // made `cargo package` fail outright. Skipped when building from a packaging
    // directory, where the copy is useless anyway: a consumer building tessl
    // from a registry always compiles the metallib fresh into OUT_DIR, and only
    // this repository's offline/CI runs ever set TESSL_SKIP_AOT.
    if !is_packaging_dir(&manifest_dir) {
        // Do not truncate an inode a running process may still have mapped;
        // stage beside it and rename. Failure is surfaced, never swallowed.
        let crate_copy = manifest_dir.join("default.metallib");
        let staged_copy = manifest_dir.join(format!(".default-{build_id}.metallib"));
        fs::File::create_new(&staged_copy).expect("reserve offline metallib staging file");
        fs::copy(&metallib_out, &staged_copy).expect("stage offline metallib");
        fs::rename(&staged_copy, &crate_copy).expect("publish offline metallib");
    }

    println!(
        "cargo:rustc-env=TESSL_METALLIB={}",
        metallib_out.display()
    );
}

fn try_metal_compile(metal: &Path, sdk: &str, src: &Path, air: &Path, metal_std: &str) -> bool {
    let std_flag = format!("-std={metal_std}");
    Command::new(metal)
        .args([
            std_flag.as_str(),
            "-O2",
            "-isysroot",
            sdk,
            "-mmacosx-version-min=26.0",
            "-c",
        ])
        .arg(src)
        .arg("-o")
        .arg(air)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn ensure_developer_dir() {
    if env::var_os("DEVELOPER_DIR").is_some() {
        return;
    }
    let xcode = Path::new("/Applications/Xcode.app/Contents/Developer");
    if xcode.is_dir() {
        unsafe { env::set_var("DEVELOPER_DIR", xcode) };
    }
}

fn resolve_metal() -> PathBuf {
    if let Ok(p) = xcrun_try(&["-f", "metal"]) {
        return PathBuf::from(p);
    }
    panic!(
        "metal compiler not found. Install Xcode and run:\n  \
         sudo xcode-select -s /Applications/Xcode.app/Contents/Developer\n  \
         xcodebuild -downloadComponent MetalToolchain"
    );
}

fn resolve_metallib() -> PathBuf {
    PathBuf::from(xcrun_stdout(&["-f", "metallib"]))
}

fn xcrun_stdout(args: &[&str]) -> String {
    let out = Command::new("xcrun")
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("xcrun {:?} failed to spawn: {e}", args));
    if !out.status.success() {
        panic!(
            "xcrun {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn xcrun_try(args: &[&str]) -> Result<String, ()> {
    let out = Command::new("xcrun").args(args).output().map_err(|_| ())?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        Err(())
    }
}

fn run(cmd: &mut Command, label: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("{label}: failed to spawn: {e}"));
    if !status.success() {
        panic!("{label}: exited with {status}");
    }
}

/// Emit `rerun-if-changed` for every `.metal` file, not just the directory.
///
/// A bare `cargo:rerun-if-changed=kernels/` tracks the *directory*, whose mtime
/// only moves when a file is created, deleted or renamed — editing a kernel in
/// place does not touch it. The result is a metallib that silently stays stale
/// while `cargo test` reports a pass, which is how a broken kernel can look
/// green. Listing the files individually is the only reliable form.
fn track_kernel_sources(dir: &Path) {
    println!("cargo:rerun-if-changed={}", dir.display());
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => panic!("read {} for change tracking: {e}", dir.display()),
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let p = entry.path();
        if p.is_dir() {
            track_kernel_sources(&p);
        } else if p.extension().and_then(|s| s.to_str()) == Some("metal") {
            println!("cargo:rerun-if-changed={}", p.display());
        }
    }
}

/// True when this build is Cargo verifying a package tarball.
///
/// `cargo package` unpacks into `<target>/package/<name>-<version>/` and builds
/// there, then fails the run if the build script touched anything in that
/// directory. Detecting it by path is the available signal — Cargo exposes no
/// "am I packaging" variable — and it is precise: a normal checkout is not
/// nested under `target/package/`.
fn is_packaging_dir(manifest_dir: &Path) -> bool {
    let mut it = manifest_dir.components().rev();
    // .../target/package/<name>-<version>
    it.next().is_some()
        && it.next().map(|c| c.as_os_str() == "package").unwrap_or(false)
        && it.next().map(|c| c.as_os_str() == "target").unwrap_or(false)
}
