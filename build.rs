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
//! `TESSL_SKIP_AOT` is an explicit offline escape hatch. It requires
//! `TESSL_PREBUILT_METALLIB` to name an existing absolute path; the build
//! script never creates or replaces files in `CARGO_MANIFEST_DIR`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(tessl_embedded_metallib)");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-env-changed=TESSL_SKIP_AOT");
    println!("cargo:rerun-if-env-changed=METAL_RUNTIME_SKIP_AOT");
    println!("cargo:rerun-if-env-changed=TESSL_PREBUILT_METALLIB");
    println!("cargo:rerun-if-env-changed=TESSL_GEMM_TUNE");
    println!("cargo:rerun-if-env-changed=METAL_NATIVE_GEMM_TUNE");

    // CoreGraphics is required for MTLCreateSystemDefaultDevice.
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels_dir = manifest_dir.join("kernels");

    // Canonical kernel sources. Dependents that build their own metallib read
    // this as `DEP_TESSL_KERNELS` (Cargo derives the name from `links`). Emit
    // it on every branch, including docs.rs and offline builds, so a global
    // skip setting cannot silently erase the downstream contract.
    println!("cargo:kernels={}", kernels_dir.display());

    // docs.rs builds on x86_64 Linux with no Xcode or Metal toolchain. Every
    // other branch below shells out to `xcrun`, which does not exist there — so
    // without this the crate has no path to a rendered docs page, only a red
    // build.
    //
    // Documentation does not run kernels, so an empty metallib path is the
    // honest answer: `metallib_path()` returns "" and `GpuRuntime::new` fails
    // loudly if anything ever tried. `DOCS_RS` is set only by docs.rs, so this
    // cannot silently swallow a real build.
    if env::var_os("DOCS_RS").is_some() {
        println!("cargo:warning=DOCS_RS set; skipping metallib AOT (docs only, no GPU)");
        println!("cargo:metallib=");
        println!("cargo:rustc-env=TESSL_METALLIB=");
        return;
    }

    // Legacy spelling still honoured: this one is set by hand in CI/offline runs.
    if env::var_os("TESSL_SKIP_AOT").is_some() || env::var_os("METAL_RUNTIME_SKIP_AOT").is_some() {
        println!("cargo:warning=TESSL_SKIP_AOT set; skipping metallib AOT");
        let configured = env::var_os("TESSL_PREBUILT_METALLIB").unwrap_or_else(|| {
            panic!(
                "TESSL_SKIP_AOT is set but TESSL_PREBUILT_METALLIB is not. \
                 Name an existing absolute metallib path explicitly, or unset \
                 TESSL_SKIP_AOT and build the shaders"
            )
        });
        let configured = PathBuf::from(configured);
        if !configured.is_absolute() {
            panic!(
                "TESSL_PREBUILT_METALLIB must be an absolute path, got {}",
                configured.display()
            );
        }
        let prebuilt = configured.canonicalize().unwrap_or_else(|e| {
            panic!(
                "TESSL_PREBUILT_METALLIB={} is not accessible: {e}",
                configured.display()
            )
        });
        if !prebuilt.is_file() {
            panic!("TESSL_PREBUILT_METALLIB={} is not a file", prebuilt.display());
        }
        println!("cargo:rerun-if-changed={}", prebuilt.display());
        println!("cargo:rustc-cfg=tessl_embedded_metallib");
        println!("cargo:metallib={}", prebuilt.display());
        println!("cargo:rustc-env=TESSL_METALLIB={}", prebuilt.display());
        return;
    }

    ensure_developer_dir();

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    track_kernel_sources(&kernels_dir);

    let sdk = xcrun_stdout(&["--sdk", "macosx", "--show-sdk-path"]);
    let metal = resolve_metal();
    let metallib = resolve_metallib();

    let mut air_files: Vec<PathBuf> = Vec::new();

    // TensorOps kernels — Metal 4 dialect (macOS 26+ / MPP). Hard-fail: NAX GEMM
    // is the hot path; a simdgroup-only metallib is not acceptable.
    // The GEMM A/B rig (kernels/tune/) is 50 measurement-only kernels that
    // nothing dispatches at runtime, so it stays opt-in to keep the shipped
    // metallib small. It lives in a subdirectory precisely so the directory
    // glob below cannot pick it up by accident.
    let want_tune = env::var_os("TESSL_GEMM_TUNE").is_some() || env::var_os("METAL_NATIVE_GEMM_TUNE").is_some();
    let mut tensorops_sources: Vec<PathBuf> = vec![kernels_dir.join("matmul_tensorops.metal")];
    if want_tune {
        tensorops_sources.push(kernels_dir.join("tune/matmul_tensorops_tune.metal"));
    }
    for src in &tensorops_sources {
        if !src.exists() {
            panic!(
                "required TensorOps source missing: {}; Metal 4 / macOS 26 toolchain required",
                src.display()
            );
        }
    }

    // All other .metal sources (simdgroup GEMM + util kernels).
    let skip: &[&str] = &["matmul_tensorops.metal"];
    let mut others: Vec<PathBuf> = fs::read_dir(&kernels_dir)
        .unwrap_or_else(|e| panic!("read kernels/: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("metal")
                && !skip.iter().any(|s| p.file_name().and_then(|n| n.to_str()) == Some(*s))
        })
        .collect();
    others.sort();

    // Every source compiles under metal4.0 at once (bounded by the machine's
    // parallelism), each `.air` reused when nothing it depends on changed;
    // the results are then taken in this fixed order, so the link order, the
    // diagnostics and the metal3.2 fallback are what a one-at-a-time build
    // produced.
    let sources: Vec<PathBuf> = tensorops_sources.iter().chain(&others).cloned().collect();
    let cache = AirCache::new(&metal, &sdk, &kernels_dir);
    let outcomes = compile_metal4_all(&metal, &sdk, &out_dir, &cache, &sources);

    // TensorOps kernels: hard-fail, no fallback.
    for (src, outcome) in sources.iter().zip(&outcomes).take(tensorops_sources.len()) {
        let name = src.file_name().and_then(|n| n.to_str()).unwrap_or("<unnamed>");
        match outcome {
            Air::Ready { air, .. } => air_files.push(air.clone()),
            Air::Metal4Failed { diag, .. } => panic!(
                "{name} failed to compile (need Metal 4 / macOS 26 SDK + MetalToolchain); \
                 refusing simdgroup-only metallib\n{diag}"
            ),
        }
    }
    println!("cargo:rustc-cfg=metal_runtime_tensorops");

    for (src, outcome) in sources.iter().zip(&outcomes).skip(tensorops_sources.len()) {
        let stem = src.file_stem().unwrap().to_string_lossy();
        let name = src.file_name().and_then(|n| n.to_str()).unwrap_or(&stem);
        match outcome {
            Air::Ready { air, std } => {
                if std != "metal4.0" {
                    // Reused from a build that already printed the metal4.0
                    // diagnostic below; say so again rather than go quiet.
                    println!(
                        "cargo:warning={name} is compiled under -std={std} (reused from an earlier \
                         build whose -std=metal4.0 compile failed; edit the source to retry)"
                    );
                }
                air_files.push(air.clone());
            }
            Air::Metal4Failed {
                air,
                diag: metal4_diag,
                key,
            } => {
                // The fallback is a dialect downgrade, so its cause is printed with
                // it rather than discarded: a green build whose kernel silently
                // compiled under metal3.2 should be diagnosable from the log, and
                // when the fallback fails too the real (metal4.0) error must not
                // hide behind an unrelated metal3.2 one.
                println!(
                    "cargo:warning={name} failed under -std=metal4.0; falling back to -std=metal3.2 \
                     (shader dialect only; encode remains Metal 4). metal4.0 diagnostic follows."
                );
                for line in metal4_diag.lines().take(40) {
                    println!("cargo:warning=  {line}");
                }
                if let Err(metal32_diag) = try_metal_compile(&metal, &sdk, src, air, "metal3.2") {
                    panic!(
                        "{name} failed under both -std=metal4.0 and -std=metal3.2.\n\
                         --- metal4.0 ---\n{metal4_diag}\n--- metal3.2 ---\n{metal32_diag}"
                    );
                }
                AirCache::record(air, *key, "metal3.2");
                air_files.push(air.clone());
            }
        }
    }

    // Each build owns an immutable artifact. The runtime embeds these bytes
    // (`include_bytes!` of `TESSL_METALLIB`); the path stays for tooling and
    // `DEP_TESSL_METALLIB`.
    //
    // Immutable, not eternal: every earlier build's artifact in this OUT_DIR is
    // removed first. A binary that referenced one of them is rebuilt by Cargo
    // whenever this script reruns (`TESSL_METALLIB` is baked in through
    // `rustc-env`, and dependents read `DEP_TESSL_METALLIB`), so nothing
    // current can still name a swept path, and OUT_DIR no longer grows by one
    // metallib per build.
    sweep_previous_metallibs(&out_dir);
    let build_id = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before Unix epoch")
            .as_nanos()
    );
    let metallib_out = out_dir.join(format!("default-{build_id}.metallib"));
    fs::File::create_new(&metallib_out).expect("reserve unique metallib output");
    let mut link = Command::new(&metallib);
    for air in &air_files {
        link.arg(air);
    }
    link.arg("-o").arg(&metallib_out);
    run(&mut link, "metallib link");

    // Keep the compiled artifact inside Cargo's build-owned directory. Besides
    // making registry and vendored sources immutable, this isolates concurrent
    // profiles/targets/builds from one another. `links = "tessl"` exposes the
    // same path to direct dependents as `DEP_TESSL_METALLIB`.
    println!("cargo:rustc-cfg=tessl_embedded_metallib");
    println!("cargo:metallib={}", metallib_out.display());
    println!("cargo:rustc-env=TESSL_METALLIB={}", metallib_out.display());
}

/// `-ffp-contract=off` where a non-contracted f32 reference disagreed.
///
/// `qwen35_adamw.metal`: with contraction on, a step at magnitude ~1e4 missed
/// the `2e-6` absolute bound by one ulp (`6.104e-5` at element 20 of step 1).
/// The measured step time is in `bench/results/fp_contract.txt`.
fn disable_fp_contract(src: &Path) -> bool {
    const OFF: &[&str] = &["qwen35_adamw.metal"];
    src.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| OFF.contains(&name))
}

/// Delete `default-*.metallib` left in `out_dir` by previous builds.
fn sweep_previous_metallibs(out_dir: &Path) {
    let Ok(entries) = fs::read_dir(out_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("default-") && name.ends_with(".metallib") {
            // A stale artifact that cannot be removed is not an error worth a
            // red build; it costs disk, not correctness.
            let _ = fs::remove_file(entry.path());
        }
    }
}

/// One source after the parallel metal4.0 pass.
enum Air {
    /// `air` holds this source compiled under `std`: just now, or reused.
    Ready { air: PathBuf, std: String },
    /// metal4.0 failed with `diag`; `key` is what to record for `air` once a
    /// fallback compile succeeds.
    Metal4Failed { air: PathBuf, diag: String, key: u64 },
}

/// Compile every source under metal4.0 into `out_dir/<stem>.air`, reusing an
/// `.air` whose recorded key still matches, on up to
/// `available_parallelism()` threads. Results are in `sources` order.
fn compile_metal4_all(metal: &Path, sdk: &str, out_dir: &Path, cache: &AirCache, sources: &[PathBuf]) -> Vec<Air> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(sources.len())
        .max(1);
    let mut results: Vec<(usize, Air)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(src) = sources.get(i) else {
                            return done;
                        };
                        let air = out_dir.join(format!("{}.air", src.file_stem().unwrap().to_string_lossy()));
                        let key = cache.key(src);
                        let outcome = if let Some(std) = AirCache::reusable(&air, key) {
                            Air::Ready { air, std }
                        } else {
                            AirCache::forget(&air);
                            match try_metal_compile(metal, sdk, src, &air, "metal4.0") {
                                Ok(()) => {
                                    AirCache::record(&air, key, "metal4.0");
                                    Air::Ready {
                                        air,
                                        std: "metal4.0".into(),
                                    }
                                }
                                Err(diag) => Air::Metal4Failed { air, diag, key },
                            }
                        };
                        done.push((i, outcome));
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("metal compile worker panicked"))
            .collect()
    });
    results.sort_by_key(|(i, _)| *i);
    assert_eq!(results.len(), sources.len(), "every source compiled exactly once");
    results.into_iter().map(|(_, air)| air).collect()
}

/// When an `.air` from an earlier build can stand in for a fresh compile.
///
/// Its key covers everything the compile reads or is told: this build
/// script's own source (so a changed flag invalidates every entry), the
/// compiler binary (path, size and modification time), the SDK (path and
/// `SDKSettings.json`), every header under `kernels/` (any header edit
/// invalidates every kernel; the conservative choice), the source's path and
/// bytes and whether it is built without FP contraction. The key is written
/// next to the `.air` only after a successful compile, and removed before a
/// compile starts, so an interrupted or failed compile is never reused.
struct AirCache {
    shared: u64,
}

impl AirCache {
    fn new(metal: &Path, sdk: &str, kernels_dir: &Path) -> Self {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        include_str!("build.rs").hash(&mut h);
        metal.hash(&mut h);
        if let Ok(meta) = fs::metadata(metal) {
            meta.len().hash(&mut h);
            if let Ok(modified) = meta.modified() {
                modified.hash(&mut h);
            }
        }
        sdk.hash(&mut h);
        fs::read(Path::new(sdk).join("SDKSettings.json")).ok().hash(&mut h);
        let mut headers = Vec::new();
        collect_headers(kernels_dir, &mut headers);
        headers.sort();
        for header in &headers {
            header.hash(&mut h);
            fs::read(header)
                .unwrap_or_else(|e| panic!("read {}: {e}", header.display()))
                .hash(&mut h);
        }
        Self { shared: h.finish() }
    }

    fn key(&self, src: &Path) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.shared.hash(&mut h);
        src.hash(&mut h);
        fs::read(src)
            .unwrap_or_else(|e| panic!("read {}: {e}", src.display()))
            .hash(&mut h);
        disable_fp_contract(src).hash(&mut h);
        h.finish()
    }

    fn key_path(air: &Path) -> PathBuf {
        let mut p = air.as_os_str().to_owned();
        p.push(".key");
        PathBuf::from(p)
    }

    /// The `-std` the existing `air` was compiled under, if its key matches.
    fn reusable(air: &Path, key: u64) -> Option<String> {
        let recorded = fs::read_to_string(Self::key_path(air)).ok()?;
        let (hex, std) = recorded.trim().split_once(' ')?;
        (u64::from_str_radix(hex, 16).ok()? == key && air.is_file()).then(|| std.to_string())
    }

    fn forget(air: &Path) {
        let _ = fs::remove_file(Self::key_path(air));
    }

    fn record(air: &Path, key: u64, std: &str) {
        // A key that cannot be written costs a recompile next time, nothing more.
        let _ = fs::write(Self::key_path(air), format!("{key:016x} {std}\n"));
    }
}

/// Every `.h` under `dir`, recursively.
fn collect_headers(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_headers(&p, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some("h") {
            out.push(p);
        }
    }
}

/// Compile one kernel under `metal_std`; on failure return the compiler's
/// diagnostic (or the spawn error) instead of swallowing it.
fn try_metal_compile(metal: &Path, sdk: &str, src: &Path, air: &Path, metal_std: &str) -> Result<(), String> {
    let std_flag = format!("-std={metal_std}");
    let mut cmd = Command::new(metal);
    cmd.args([
        std_flag.as_str(),
        "-O2",
        "-fmetal-math-mode=fast",
        "-isysroot",
        sdk,
        "-mmacosx-version-min=26.0",
    ]);
    if disable_fp_contract(src) {
        cmd.arg("-ffp-contract=off");
    }
    let out = cmd
        .arg("-c")
        .arg(src)
        .arg("-o")
        .arg(air)
        .output()
        .map_err(|e| format!("failed to spawn {}: {e}", metal.display()))?;
    if out.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stderr = stderr.trim();
    Err(if stderr.is_empty() {
        format!("metal exited with {}", out.status)
    } else {
        stderr.to_string()
    })
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
        panic!("xcrun {:?} failed: {}", args, String::from_utf8_lossy(&out.stderr));
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
    let status = cmd.status().unwrap_or_else(|e| panic!("{label}: failed to spawn: {e}"));
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
        } else if matches!(p.extension().and_then(|s| s.to_str()), Some("metal") | Some("h")) {
            // `.h` as well as `.metal`: shared reduction/activation helpers are
            // included by multiple kernels and compiled as neither. Tracking
            // only sources would let a helper edit leave every dependent stale
            // in the metallib while `cargo test` reported a pass.
            println!("cargo:rerun-if-changed={}", p.display());
        }
    }
}
