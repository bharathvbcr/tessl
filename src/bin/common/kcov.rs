//! Which kernels does the benchmark suite actually dispatch?
//!
//! Coverage used to be inferred: grep a kernel's name out of the bench sources.
//! That was wrong in both directions: `matmul2d_tensorops_*` is reached through a
//! dispatcher and looked untimed, while a name in a comment looked timed. Worse,
//! the inventory itself was wrong: kernels declared through the `NN_COOP_KERNEL`
//! family of macros have no `kernel void` line, so a source scan missed 16 of 83
//! entry points outright.
//!
//! This measures instead. `TESSL_KERNEL_TRACE=1` makes the runtime record every
//! name passed to `GpuRuntime::pipeline`, which is the single site where a kernel
//! is selected; each bench binary prints its trace on exit, on every exit path.
//!
//! The kernel census comes from the real preprocessor: `xcrun metal -E`, with the
//! flags `build.rs`'s `metal_args` gives every compile, on each `kernels/*.metal`,
//! and the entry points are read from the preprocessed text (comments, `#if 0`,
//! and disabled `#ifdef` blocks are gone; macro stamping, `##` pasting, and
//! `[[host_name]]` / `template` instantiations are resolved). It is not
//! reimplemented here, so a toolchain that is missing is "could not run".
//!
//! The census is cross-checked against the metallib the binaries run. The runtime
//! embeds the library's bytes (`include_bytes!`), so no binary carries a path to
//! look up; `bench_gemm_sweep --dump-metallib PATH` writes those bytes out and
//! `metal-nm` reads them.
//!
//! This is the Rust owner of what `bench/kernel_coverage.py` did. Included with
//! `#[path]` by the `kernel_coverage` bin and by `tests/kernel_coverage.rs`.

#![allow(dead_code)] // The binary and the tests each use a subset.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

use crate::proc::{first_chars, run_bounded};
use crate::pyjson::{obj, quote_list, Doc};

/// Exit status of a run that found gaps.
pub const EXIT_GAPS: i32 = 1;
/// Exit status of a check that could not run. 77 is the repo's SKIPPED status
/// (`scripts/check_kernel_ir.sh`): never success, and never the same status as
/// "ran and found gaps".
pub const EXIT_CANNOT_RUN: i32 = 77;

/// Why a measurement stopped.
#[derive(Debug, Clone, PartialEq)]
pub enum Failure {
    /// The check could not run: no toolchain, no binary, a stale binary, no
    /// metallib, a suite that failed or printed no trace.
    CannotRun(String),
    /// The check ran and what it found is wrong (a census that disagrees with the
    /// metallib, a kernel traced that is not in the census, a source that does not
    /// preprocess, or kernels with no timing lane).
    Found(String),
}

impl Failure {
    pub fn message(&self) -> &str {
        match self {
            Failure::CannotRun(m) | Failure::Found(m) => m,
        }
    }

    pub fn exit_code(&self) -> i32 {
        match self {
            Failure::CannotRun(_) => EXIT_CANNOT_RUN,
            Failure::Found(_) => EXIT_GAPS,
        }
    }
}

pub type Res<T> = Result<T, Failure>;

fn cannot<T>(msg: impl Into<String>) -> Res<T> {
    Err(Failure::CannotRun(msg.into()))
}

fn found<T>(msg: impl Into<String>) -> Res<T> {
    Err(Failure::Found(msg.into()))
}

/// This crate's root, where `build.rs` and `kernels/` live.
pub fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn kernel_dir() -> PathBuf {
    crate_root().join("kernels")
}

/// `target/release` of this crate, honouring `CARGO_TARGET_DIR`.
pub fn bin_dir() -> PathBuf {
    let target = match std::env::var_os("CARGO_TARGET_DIR") {
        Some(t) if !t.is_empty() => std::path::absolute(t).unwrap_or_else(|_| crate_root().join("target")),
        _ => crate_root().join("target"),
    };
    target.join("release")
}

// -------------------------------------------------------------------- freshness

/// Newest mtime among what the metallib and the binaries are built from.
pub fn newest_input(crate_dir: &Path) -> SystemTime {
    let mut paths = vec![crate_dir.join("build.rs")];
    if let Ok(rd) = std::fs::read_dir(crate_dir.join("kernels")) {
        for e in rd.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".metal") || name.ends_with(".h") {
                paths.push(e.path());
            }
        }
    }
    paths
        .iter()
        .filter_map(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
        .max()
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

/// A bench binary older than the kernel sources or `build.rs` embeds a metallib
/// (and dispatches kernels) the source tree no longer describes.
pub fn require_fresh(binary: &Path, crate_dir: &Path) -> Res<()> {
    let built = std::fs::metadata(binary)
        .and_then(|m| m.modified())
        .map_err(|e| Failure::CannotRun(format!("{}: {e}", binary.display())))?;
    if built < newest_input(crate_dir) {
        return cannot(format!(
            "{} is older than kernels/*.metal, kernels/*.h or build.rs, so it embeds a stale \
             metallib; rebuild with `cargo build --release --bins`",
            binary.display()
        ));
    }
    Ok(())
}

// ------------------------------------------------------------------ toolchain

/// `(metal, sdk)`: the compiler and the SDK path, or why there is neither.
pub fn metal_toolchain() -> Res<(String, String)> {
    let xcrun = |args: &[&str]| {
        Command::new("xcrun").args(args).output().map_err(|e| {
            Failure::CannotRun(format!(
                "kernel census needs `xcrun metal` (Xcode Metal Toolchain): {e}"
            ))
        })
    };
    let metal = xcrun(&["-f", "metal"])?;
    let sdk = xcrun(&["--sdk", "macosx", "--show-sdk-path"])?;
    let metal_path = String::from_utf8_lossy(&metal.stdout).trim().to_string();
    if !metal.status.success() || metal_path.is_empty() || !sdk.status.success() {
        return cannot("kernel census needs `xcrun metal` (Xcode Metal Toolchain); no kernel source was preprocessed");
    }
    Ok((metal_path, String::from_utf8_lossy(&sdk.stdout).trim().to_string()))
}

fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b >= 0x80
}

/// Python's `\s` over ASCII.
fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c | 0x1c..=0x1f)
}

fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && is_space(b[i]) {
        i += 1;
    }
    i
}

/// The flag list `build.rs`'s `metal_args` gives every compile, read from
/// `build.rs` itself (no second copy to drift). `-std` is chosen by the caller.
pub fn build_rs_flags(build_rs: &str, sdk: &str) -> Res<Vec<String>> {
    let unreadable =
        || Failure::CannotRun("could not read build.rs `metal_args`; the preprocessor flags are unknown".into());
    let start = build_rs.find("fn metal_args(").ok_or_else(unreadable)?;
    let after = &build_rs[start..];
    let marker = "let mut args: Vec<String> = [";
    let list_at = after.find(marker).ok_or_else(unreadable)? + marker.len();
    let rest = &after[list_at..];
    // The list ends at the first `]` that is followed by `.iter()`.
    let mut end = None;
    for (i, c) in rest.char_indices() {
        if c == ']' && rest[i + 1..].trim_start().starts_with(".iter()") {
            end = Some(i);
            break;
        }
    }
    let body = &rest[..end.ok_or_else(unreadable)?];

    let b = body.as_bytes();
    let mut flags: Vec<String> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'"' {
            let close = body[i + 1..].find('"').map(|j| i + 1 + j);
            let Some(close) = close else { break };
            let lit = &body[i + 1..close];
            // `-std={metal_std}` is chosen per compile.
            if !lit.contains('{') {
                flags.push(lit.to_string());
            }
            i = close + 1;
        } else if b[i..].starts_with(b"sdk")
            && (i == 0 || !is_word(b[i - 1]))
            && (i + 3 >= b.len() || !is_word(b[i + 3]))
        {
            flags.push(sdk.to_string());
            i += 3;
        } else {
            i += 1;
        }
    }
    if !flags.iter().any(|f| f == "-isysroot") || !flags.iter().any(|f| f == sdk) {
        return cannot(format!(
            "build.rs metal_args parsed to {}, with no -isysroot SDK",
            quote_list(&flags)
        ));
    }
    Ok(flags)
}

// ----------------------------------------------------------------- the census

/// One `kernel void NAME` in preprocessed MSL, with the `template` prefix and the
/// `[[attributes]]` in front of it.
#[derive(Debug, PartialEq)]
struct EntryMatch<'a> {
    end: usize,
    template: Option<&'a str>,
    attrs: &'a str,
    name: &'a str,
}

/// `(template\s*(?:<[^<>]*>)?\s*)?((?:\[\[[^\]]*\]\]\s*)*)\bkernel\s+void\s+([A-Za-z_][A-Za-z0-9_]*)`
/// matched at `p`, with the same preferences the regex has.
fn entry_at(text: &str, p: usize) -> Option<EntryMatch<'_>> {
    let b = text.as_bytes();
    let mut i = p;
    let mut template = None;
    if b[i..].starts_with(b"template") {
        let mut j = skip_ws(b, i + 8);
        if b.get(j) == Some(&b'<') {
            let mut k = j + 1;
            while k < b.len() && b[k] != b'<' && b[k] != b'>' {
                k += 1;
            }
            if b.get(k) == Some(&b'>') {
                j = k + 1;
            }
        }
        j = skip_ws(b, j);
        template = Some(&text[i..j]);
        i = j;
    }
    let attrs_from = i;
    while b[i..].starts_with(b"[[") {
        // `\[\[[^\]]*\]\]`: up to the first `]`, which must be followed by another.
        let Some(close) = b[i + 2..].iter().position(|&c| c == b']').map(|k| i + 2 + k) else {
            break;
        };
        if b.get(close + 1) != Some(&b']') {
            break;
        }
        i = skip_ws(b, close + 2);
    }
    let attrs = &text[attrs_from..i];
    if !b[i..].starts_with(b"kernel") || (i > 0 && is_word(b[i - 1])) {
        return None;
    }
    let mut j = i + 6;
    let ws = skip_ws(b, j);
    if ws == j || !b[ws..].starts_with(b"void") {
        return None;
    }
    j = ws + 4;
    let ws = skip_ws(b, j);
    if ws == j {
        return None;
    }
    j = ws;
    if !(b.get(j).is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_')) {
        return None;
    }
    let mut k = j + 1;
    while k < b.len() && (b[k].is_ascii_alphanumeric() || b[k] == b'_') {
        k += 1;
    }
    Some(EntryMatch {
        end: k,
        template,
        attrs,
        name: &text[j..k],
    })
}

/// `host_name\(\s*"([^"]+)"\s*\)`, first occurrence.
fn host_name(attrs: &str) -> Option<&str> {
    let b = attrs.as_bytes();
    let mut from = 0;
    while let Some(at) = attrs[from..].find("host_name") {
        let mut i = from + at + 9;
        from += at + 1;
        if b.get(i) != Some(&b'(') {
            continue;
        }
        i = skip_ws(b, i + 1);
        if b.get(i) != Some(&b'"') {
            continue;
        }
        let Some(len) = attrs[i + 1..].find('"') else { continue };
        if len == 0 {
            continue;
        }
        let name = &attrs[i + 1..i + 1 + len];
        let close = skip_ws(b, i + 1 + len + 1);
        if b.get(close) == Some(&b')') {
            return Some(name);
        }
    }
    None
}

/// `<\s*[^>\s]`: a template parameter list that names a parameter.
fn has_template_params(template: &str) -> bool {
    let b = template.as_bytes();
    b.iter().enumerate().any(|(i, &c)| {
        c == b'<' && {
            let j = skip_ws(b, i + 1);
            j < b.len() && b[j] != b'>'
        }
    })
}

/// Exported kernel names in preprocessed MSL.
///
/// An entry point is `kernel void NAME`; `[[host_name("x")]]` renames it to x
/// (what the metallib exports). A `template <params>` kernel is a generic
/// definition, not an entry point, until an explicit instantiation
/// (`template [[host_name("x")]] kernel void f<...>`) names it.
pub fn entry_points(preprocessed: &str, where_: &str) -> Res<BTreeSet<String>> {
    let b = preprocessed.as_bytes();
    let mut out = BTreeSet::new();
    let mut p = 0;
    while p < b.len() {
        if !matches!(b[p], b't' | b'[' | b'k') {
            p += 1;
            continue;
        }
        let Some(m) = entry_at(preprocessed, p) else {
            p += 1;
            continue;
        };
        p = m.end;
        let host = host_name(m.attrs);
        if let Some(t) = m.template {
            if has_template_params(t) {
                continue; // generic definition
            }
            if host.is_none() {
                return found(format!(
                    "{where_}: template instantiation of `{}` has no [[host_name]], so its exported name cannot be derived",
                    m.name
                ));
            }
        }
        out.insert(host.unwrap_or(m.name).to_string());
    }
    Ok(out)
}

/// `metal -E` of one source, with build.rs's std fallback (metal4.0 then
/// metal3.2). Returns the preprocessed text.
pub fn preprocess(metal: &str, flags: &[String], src: &Path) -> Res<String> {
    let dir = scratch_dir("kernel-census")?;
    let out = dir.path().join("pp.i");
    let mut errs = Vec::new();
    for std in ["metal4.0", "metal3.2"] {
        let mut cmd = Command::new(metal);
        cmd.arg("-E")
            .arg(format!("-std={std}"))
            .args(flags)
            .arg(src)
            .arg("-o")
            .arg(&out);
        let ran = run_bounded(&mut cmd, 120)
            .map_err(|e| Failure::CannotRun(format!("could not run {metal} -E on {}: {e}", src.display())))?;
        if ran.code == Some(0) && out.is_file() {
            let bytes = std::fs::read(&out).map_err(|e| Failure::CannotRun(format!("{}: {e}", out.display())))?;
            return Ok(String::from_utf8_lossy(&bytes).into_owned());
        }
        errs.push(format!("{std}: {}", first_chars(&ran.stderr, 300)));
    }
    found(format!(
        "{} does not preprocess ({})",
        src.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
        errs.join("; ")
    ))
}

/// A temporary directory removed on drop.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn scratch_dir(tag: &str) -> Res<TempDir> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let d = std::env::temp_dir().join(format!(
        "{tag}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&d).map_err(|e| Failure::CannotRun(format!("mkdir {}: {e}", d.display())))?;
    Ok(TempDir(d))
}

/// The kernel census: every entry point, and which source file declares it.
#[derive(Debug)]
pub struct Census {
    pub names: BTreeSet<String>,
    /// Source file stem -> the entry points it declares, in file-name order.
    pub by_file: BTreeMap<String, BTreeSet<String>>,
}

/// Every kernel entry point in the shipped kernel sources.
///
/// Read from the real preprocessor's output, so the set is right by
/// construction and needs no cross-check to be trusted. `metallib` is still
/// compared when it can be read (the library is what the runtime looks a
/// pipeline up in), and `require_metallib` makes an unreadable library a
/// failure; the command line always sets it. A missing toolchain is
/// `Failure::CannotRun`: that is never answered with a different scan.
///
/// The metallib is read before the sources are preprocessed, so a tree with no
/// library to compare against fails at once instead of after 38 compiler runs.
pub fn inventory(kdir: &Path, metallib: Option<&Path>, require_metallib: bool) -> Res<Census> {
    let mut sources: Vec<String> = std::fs::read_dir(kdir)
        .map_err(|e| Failure::Found(format!("no kernels found under {}: {e}", kdir.display())))?
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|f| f.ends_with(".metal"))
        .collect();
    sources.sort();
    if sources.is_empty() {
        return found(format!("no kernels found under {}", kdir.display()));
    }
    let (metal, sdk) = metal_toolchain()?;
    let build_rs = std::fs::read_to_string(crate_root().join("build.rs"))
        .map_err(|e| Failure::CannotRun(format!("could not read build.rs: {e}")))?;
    let flags = build_rs_flags(&build_rs, &sdk)?;
    let exported = metallib_symbols(metallib, require_metallib)?;

    // Each source preprocesses on its own; the first failure in file order is the
    // one reported, as it would be if they ran one after another.
    let results: Vec<Res<BTreeSet<String>>> = {
        let next = std::sync::atomic::AtomicUsize::new(0);
        let slots: Vec<std::sync::Mutex<Option<Res<BTreeSet<String>>>>> =
            sources.iter().map(|_| std::sync::Mutex::new(None)).collect();
        let workers = std::thread::available_parallelism()
            .map_or(1, |p| p.get())
            .min(8)
            .min(sources.len());
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(f) = sources.get(i) else { break };
                    let r = preprocess(&metal, &flags, &kdir.join(f)).and_then(|text| entry_points(&text, f));
                    *slots[i].lock().unwrap_or_else(|p| p.into_inner()) = Some(r);
                });
            }
        });
        slots
            .into_iter()
            .map(|s| {
                s.into_inner()
                    .unwrap_or_else(|p| p.into_inner())
                    .unwrap_or_else(|| cannot("internal: a source was not scanned"))
            })
            .collect()
    };
    let mut names = BTreeSet::new();
    let mut by_file = BTreeMap::new();
    for (f, r) in sources.iter().zip(results) {
        let found_in = r?;
        names.extend(found_in.iter().cloned());
        by_file.insert(f.trim_end_matches(".metal").to_string(), found_in);
    }
    if names.is_empty() {
        return found(format!("no kernels found under {}", kdir.display()));
    }

    // Cross-check against the compiled metallib, which is ground truth for what
    // the runtime can look up. Any disagreement means the census is wrong.
    if let Some(exported) = exported {
        let only_lib: Vec<&String> = exported.difference(&names).collect();
        let only_src: Vec<&String> = names.difference(&exported).collect();
        if !only_lib.is_empty() || !only_src.is_empty() {
            let head = |v: &[&String]| quote_list(&v.iter().take(8).map(|s| s.as_str()).collect::<Vec<_>>());
            return found(format!(
                "kernel inventory disagrees with the compiled metallib.\n  in metallib, missed by the census ({}): {}\n  \
                 in census, absent from metallib ({}): {}\nThe census is wrong; fix it rather than measuring coverage against it.",
                only_lib.len(),
                head(&only_lib),
                only_src.len(),
                head(&only_src)
            ));
        }
    }
    Ok(Census { names, by_file })
}

/// Exported kernel names from the built metallib, or `None` if unavailable.
///
/// `None` and an empty set are deliberately different: a toolchain without
/// `metal-nm` must not read as a library with no kernels in it. Set `required`
/// for a coverage measurement: a cross-check that could not run is then
/// `CannotRun` rather than the same PASS as one that actually verified the
/// binary. (An empty listing is `None` when not required, as it was.)
pub fn metallib_symbols(path: Option<&Path>, required: bool) -> Res<Option<BTreeSet<String>>> {
    let Some(path) = path else {
        // The runtime embeds the library's bytes, so the census is read from
        // those bytes, dumped by the binary being measured.
        let binary = bin_dir().join("bench_gemm_sweep");
        if !binary.is_file() {
            if required {
                return cannot(format!(
                    "benchmark binary missing at {}; run `cargo build --release --bins` before measuring kernel coverage",
                    binary.display()
                ));
            }
            return Ok(None);
        }
        let dir = scratch_dir("kernel-coverage-lib")?;
        let dumped = require_fresh(&binary, &crate_root()).and_then(|()| dump_embedded_metallib(&binary, dir.path()));
        return match dumped {
            Ok(lib) => metallib_symbols(Some(&lib), required),
            Err(e) if required => Err(e),
            Err(_) => Ok(None),
        };
    };
    if !path.exists() {
        if required {
            return cannot(format!(
                "compiled metallib missing at {}; build the release binaries before measuring kernel coverage",
                path.display()
            ));
        }
        return Ok(None);
    }
    let ran = run_bounded(Command::new("xcrun").arg("metal-nm").arg(path), 120);
    let ran = match ran {
        Ok(r) => r,
        Err(e) if required => return cannot(format!("could not execute xcrun metal-nm: {e}")),
        Err(_) => return Ok(None),
    };
    if ran.code != Some(0) {
        if required {
            let detail = first_chars(&ran.stderr, 400);
            let detail = if detail.is_empty() {
                "no diagnostic".to_string()
            } else {
                detail
            };
            return cannot(format!("metal-nm failed for {}: {detail}", path.display()));
        }
        return Ok(None);
    }
    let mut out = BTreeSet::new();
    for line in ran.stdout.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        // `<addr> T <name>` marks an exported kernel entry point.
        if parts.len() == 3 && parts[1] == "T" {
            out.insert(parts[2].to_string());
        }
    }
    if out.is_empty() {
        if required {
            return cannot(format!(
                "metal-nm exported no kernel entry points from {}; refusing to treat an unreadable or empty library as verified",
                path.display()
            ));
        }
        return Ok(None);
    }
    Ok(Some(out))
}

/// Write the metallib `binary` embeds into `directory`; return its path.
///
/// `CannotRun` names the fix when the binary cannot produce it, so a stale
/// binary (built before `--dump-metallib` existed) is not mistaken for a library
/// with nothing in it.
pub fn dump_embedded_metallib(binary: &Path, directory: &Path) -> Res<PathBuf> {
    let out = directory.join("embedded.metallib");
    let fix = "rebuild with `cargo build --release --bins`";
    let ran = run_bounded(Command::new(binary).arg("--dump-metallib").arg(&out), 60).map_err(|e| {
        Failure::CannotRun(format!(
            "could not run {} --dump-metallib: {e}; {fix}",
            binary.display()
        ))
    })?;
    if ran.code != Some(0) || std::fs::metadata(&out).map_or(true, |m| !m.is_file() || m.len() == 0) {
        let detail = first_chars(&ran.stderr, 400);
        let detail = if detail.is_empty() {
            "no diagnostic".to_string()
        } else {
            detail
        };
        return cannot(format!(
            "{} --dump-metallib exited {} without a library ({detail}); the binary is stale or embeds no metallib: {fix}",
            binary.display(),
            ran.code.map_or("by signal".to_string(), |c| c.to_string())
        ));
    }
    Ok(out)
}

// --------------------------------------------------------------------- suites

/// One benchmark invocation: the binary, its arguments (`{tmp}` is the scratch
/// directory), and the environment that makes it cover its own breadth.
pub struct Suite {
    pub name: &'static str,
    pub argv: Vec<&'static str>,
    pub env: Vec<(&'static str, String)>,
}

fn suite(name: &'static str, argv: &[&'static str], env: &[(&'static str, &str)]) -> Suite {
    Suite {
        name,
        argv: argv.to_vec(),
        env: env.iter().map(|(k, v)| (*k, v.to_string())).collect(),
    }
}

/// Benchmarks to run, and the arguments that make them cover their own breadth.
/// Kept small: this is a coverage probe, not a measurement.
pub fn suites() -> Vec<Suite> {
    let mut v = vec![
        suite(
            "bench_gemm_sweep",
            &[],
            &[("BENCH_SHAPES", "256x256x256,192x520x1024"), ("BENCH_ITERS", "1"), ("BENCH_WARMUP", "0")],
        ),
        suite(
            "bench_gemm_sweep",
            &["--dump-parity", "{tmp}/gemmparity"],
            &[("BENCH_PARITY_SHAPE", "square_512"), ("BENCH_PARITY_SEEDS", "1")],
        ),
        // Decode configs reach the KV-split kernels; one prefill config at each
        // head dim reaches the row-parallel ones. Both families are needed: the
        // public entry points route by Tq, so a decode-only run leaves the prefill
        // kernels undispatched.
        suite(
            "bench_flash_attn",
            &[],
            &[
                ("BENCH_ITERS", "1"),
                ("BENCH_WARMUP", "0"),
                (
                    "BENCH_ATTN_CFGS",
                    "swa128_decode_1k,swa256_decode_4k,global512_decode_4k,swa128_prefill_512,swa256_prefill_2048,global512_prefill_1024",
                ),
            ],
        ),
    ];
    // Every decode (lanes-per-key x chunk) pair, and every rows lanes-per-row.
    // Both are compile-time constants, so each combination is its own kernel and
    // sweeping one axis at the other's default would leave the cross product
    // undispatched -- which is exactly what the gate caught.
    for r in ["8", "16", "32"] {
        for ch in ["64", "128", "256"] {
            v.push(suite(
                "bench_flash_attn",
                &[],
                &[
                    ("BENCH_ITERS", "1"),
                    ("BENCH_WARMUP", "0"),
                    ("BENCH_ATTN_DECODE_R", r),
                    ("BENCH_ATTN_DECODE_CHUNK", ch),
                    (
                        "BENCH_ATTN_CFGS",
                        "swa128_decode_1k,swa256_decode_4k,global512_decode_4k",
                    ),
                ],
            ));
        }
    }
    for r in ["8", "16", "32"] {
        for g in ["8", "16", "32"] {
            v.push(suite(
                "bench_flash_attn",
                &[],
                &[
                    ("BENCH_ITERS", "1"),
                    ("BENCH_WARMUP", "0"),
                    ("BENCH_ATTN_ROWS_R", r),
                    ("BENCH_ATTN_ROWS_SGT", g),
                    (
                        "BENCH_ATTN_CFGS",
                        "swa128_prefill_512,swa256_prefill_2048,global512_prefill_1024",
                    ),
                ],
            ));
        }
    }
    v.extend([
        // The tiled kernels remain the A/B baseline and are still dispatched.
        suite(
            "bench_flash_attn",
            &[],
            &[
                ("BENCH_ITERS", "1"),
                ("BENCH_WARMUP", "0"),
                ("TESSL_ATTN_TILED", "1"),
                (
                    "BENCH_ATTN_CFGS",
                    "swa128_decode_1k,swa256_decode_4k,global512_decode_4k",
                ),
            ],
        ),
        suite("bench_nn_kernels", &[], &[("BENCH_ITERS", "1"), ("BENCH_WARMUP", "0")]),
        // The TN/NT accumulate lanes reach the accumulate kernels with no opt-in
        // since TESSL_GEMM_ACCUM / _ACCUM_DX were retired (2026-10-10).
        suite(
            "bench_gemm_variants",
            &[],
            &[("BENCH_ITERS", "1"), ("BENCH_WARMUP", "0")],
        ),
        // The routing lanes dispatch the parallel bf16 split-K and its
        // accumulating reduce, and the f16 / relaxed-f32 64x64 epilogue kernels,
        // which no other suite reaches. One round, so the drift gate cannot fire.
        suite(
            "bench_gemm_routes",
            &[],
            &[
                ("BENCH_REPS", "1"),
                ("BENCH_ITERS", "1"),
                ("BENCH_WARMUP", "0"),
                ("BENCH_ROUNDS", "1"),
            ],
        ),
    ]);
    v
}

/// Environment variables worth naming in a suite's label.
fn labelled(key: &str) -> bool {
    key.starts_with("TESSL_")
        || matches!(
            key,
            "BENCH_ATTN_ROWS_R" | "BENCH_ATTN_ROWS_SGT" | "BENCH_ATTN_DECODE_CHUNK" | "BENCH_ATTN_DECODE_R"
        )
}

/// What every suite dispatched.
pub struct Traced {
    pub traced: BTreeSet<String>,
    /// `(label, kernels)` in suite order; a label seen twice keeps its first slot.
    pub per_suite: Vec<(String, BTreeSet<String>)>,
    pub failures: Vec<String>,
}

/// Run `suites` from `bin` and union the kernels they dispatched.
pub fn run_suites(suites: &[Suite], bin: &Path, tmp: &Path, crate_dir: &Path, suite_secs: u64) -> Traced {
    let mut t = Traced {
        traced: BTreeSet::new(),
        per_suite: Vec::new(),
        failures: Vec::new(),
    };
    for s in suites {
        let binary = bin.join(s.name);
        if !binary.exists() {
            t.failures
                .push(format!("{}: not built (cargo build --release --bins)", s.name));
            continue;
        }
        if let Err(e) = require_fresh(&binary, crate_dir) {
            t.failures.push(e.message().to_string());
            continue;
        }
        let argv: Vec<String> = s
            .argv
            .iter()
            .map(|a| a.replace("{tmp}", &tmp.to_string_lossy()))
            .collect();
        let mut cmd = Command::new(&binary);
        cmd.args(&argv).env("TESSL_KERNEL_TRACE", "1");
        for (k, v) in &s.env {
            cmd.env(k, v);
        }
        let shown = argv.join(" ");
        let ran = match run_bounded(&mut cmd, suite_secs) {
            Ok(r) => r,
            Err(e) => {
                t.failures.push(format!("{} {shown}: {e}", s.name));
                continue;
            }
        };
        if ran.code != Some(0) {
            t.failures.push(format!(
                "{} {shown}: exited {}\n{}",
                s.name,
                ran.code.map_or("by signal".to_string(), |c| c.to_string()),
                first_chars(&ran.stderr, 400)
            ));
            continue;
        }
        let Some(line) = ran.stderr.lines().rev().find(|l| l.starts_with("KERNEL_TRACE ")) else {
            // An empty trace and a missing trace must not read the same.
            t.failures.push(format!(
                "{}: emitted no KERNEL_TRACE line; is the runtime honouring TESSL_KERNEL_TRACE?",
                s.name
            ));
            continue;
        };
        let got: BTreeSet<String> = line["KERNEL_TRACE ".len()..]
            .split(',')
            .filter(|x| !x.is_empty())
            .map(str::to_string)
            .collect();
        let mut env_sorted: Vec<&(&str, String)> = s.env.iter().collect();
        env_sorted.sort();
        let tag = env_sorted
            .iter()
            .filter(|(k, _)| labelled(k))
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ");
        let key = format!(
            "{}{}{}",
            s.name,
            argv.first().map_or(String::new(), |a| format!(" {a}")),
            if tag.is_empty() {
                String::new()
            } else {
                format!(" {tag}")
            }
        );
        match t.per_suite.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = got.clone(),
            None => t.per_suite.push((key, got.clone())),
        }
        t.traced.extend(got);
    }
    t
}

// --------------------------------------------------------------------- report

/// The coverage report, as the lines the tool prints and the JSON it writes.
pub struct Report {
    pub text: String,
    pub doc: Doc,
    pub missing: usize,
}

/// Compare a trace against the census. A traced name absent from the inventory
/// means the inventory is wrong, which is exactly the failure this tool exists
/// to prevent.
pub fn report(census: &Census, traced: &Traced) -> Res<Report> {
    let stray: Vec<&String> = traced.traced.difference(&census.names).collect();
    if !stray.is_empty() {
        let head: Vec<&str> = stray.iter().take(8).map(|s| s.as_str()).collect();
        return found(format!(
            "dispatched {} kernel(s) absent from the source inventory: {}. The inventory is undercounting; \
             fix it before reading any coverage number from it.",
            stray.len(),
            quote_list(&head)
        ));
    }
    let covered: Vec<&String> = traced.traced.iter().collect();
    let missing: Vec<&String> = census.names.difference(&traced.traced).collect();
    let pct = 100.0 * covered.len() as f64 / census.names.len() as f64;
    let mut text = String::new();
    text.push_str(&format!("kernel entry points : {}\n", census.names.len()));
    text.push_str(&format!("dispatched by a bench: {}  ({pct:.0}%)\n", covered.len()));
    text.push_str(&format!("never dispatched     : {}\n\n", missing.len()));
    for (suite, got) in &traced.per_suite {
        text.push_str(&format!("  {suite:<40}{:>4} kernels\n", got.len()));
    }
    if !missing.is_empty() {
        text.push_str("\nuncovered, by file:\n");
        for (f, ks) in &census.by_file {
            let gap: Vec<&String> = ks.difference(&traced.traced).collect();
            if !gap.is_empty() {
                let head: Vec<&str> = gap.iter().take(4).map(|s| s.as_str()).collect();
                text.push_str(&format!(
                    "  {f:<24}{:>3}  {}{}\n",
                    gap.len(),
                    head.join(", "),
                    if gap.len() > 4 { " ..." } else { "" }
                ));
            }
        }
    }
    let strs = |v: &[&String]| Doc::Arr(v.iter().map(|s| Doc::str(s.as_str())).collect());
    let doc = obj([
        ("entry_points", Doc::uint(census.names.len())),
        ("covered", strs(&covered)),
        ("missing", strs(&missing)),
        (
            "per_suite",
            Doc::Obj(
                traced
                    .per_suite
                    .iter()
                    .map(|(k, v)| (k.clone(), Doc::Arr(v.iter().map(Doc::str).collect())))
                    .collect(),
            ),
        ),
    ]);
    Ok(Report {
        text,
        doc,
        missing: missing.len(),
    })
}
