//! Evidence for a paired benchmark: the drift policy, the provenance block and
//! the atomic, fail-closed publication of the artifact.
//!
//! This is the Rust owner of what `bench/benchmark_evidence.py` did for the
//! retired `paired_cross_runtime.py` and `attn_paired.py` drivers. Shader
//! provenance is the hash of each lane executable: the runtime bakes the
//! metallib into the binary (`include_bytes!` in `src/runtime.rs`), so the
//! binary's bytes are the shaders' bytes. (The retired drivers resolved a
//! metallib *path* embedded in the binary, which no current binary carries.) Only
//! `bench_paired` includes it (`#[path]`), so the rest of `src/bin` does not
//! compile it. No dependency: SHA-256 is implemented here (checked against the
//! FIPS 180-4 vectors below) and every host probe is a short-lived command.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::common::median;
use crate::json::{Json, Syntax};

/// Default and ceiling for a paired ratio's max/min spread.
pub const DEFAULT_MAX_RATIO_SPREAD: f64 = 1.10;
pub const MAX_RATIO_SPREAD: f64 = 1.25;
pub const DEFAULT_OUTER_ROUNDS: usize = 6;
pub const MIN_OUTER_ROUNDS: usize = 4;
pub const MIN_INNER_ITERS: usize = 3;
pub const ENVIRONMENT_PROBE_POLICY: &str =
    "thermal/power/load are provenance-only: pmset is macOS-specific and may emit \
     non-numeric state, while normalized host load is workload- and CPU-count-dependent; \
     paired ratio spread is the portable enforced stability gate";
/// Inherited variables with these prefixes are dropped by `--clean-env`.
pub const BENCHMARK_ENV_PREFIXES: [&str; 10] = [
    "BENCH_",
    "TESSL_",
    "METAL_",
    "MTL_",
    "MLX_",
    "PYTORCH_",
    "OMP_",
    "MKL_",
    "VECLIB_",
    "ACCELERATE_",
];

pub type Res<T> = Result<T, String>;

// ------------------------------------------------------------------ policy

/// A finite ratio-spread limit in `[1, MAX_RATIO_SPREAD]`.
pub fn parse_ratio_spread_limit(text: &str) -> Res<f64> {
    let value: f64 = text
        .trim()
        .parse()
        .map_err(|_| format!("ratio-spread limit: expected a number, got {text:?}"))?;
    if !value.is_finite() || !(1.0..=MAX_RATIO_SPREAD).contains(&value) {
        return Err(format!(
            "ratio-spread limit must be finite and in [1, {MAX_RATIO_SPREAD}], got {text:?}"
        ));
    }
    Ok(value)
}

/// Even rounds, at least four, give exact AB/BA balance, a paired median and a
/// drift estimate; the inner count is each child process's median sample size.
pub fn validate_evidence_sample_counts(rounds: usize, iters: Option<usize>) -> Res<()> {
    if rounds < MIN_OUTER_ROUNDS || rounds % 2 != 0 {
        return Err(format!(
            "rounds must be an even integer >= {MIN_OUTER_ROUNDS} for exact AB/BA balance, a paired median, \
             and drift estimation; got {rounds}"
        ));
    }
    if let Some(iters) = iters.filter(|i| *i < MIN_INNER_ITERS) {
        return Err(format!(
            "BENCH_ITERS must be >= {MIN_INNER_ITERS} for each child-process median, got {iters}"
        ));
    }
    Ok(())
}

/// `base` with inherited benchmark/tuning knobs removed.
pub fn clean_benchmark_env(base: impl IntoIterator<Item = (String, String)>) -> BTreeMap<String, String> {
    base.into_iter()
        .filter(|(k, _)| !BENCHMARK_ENV_PREFIXES.iter().any(|p| k.starts_with(p)))
        .collect()
}

/// The last usable `--out` value, found before any other option is validated,
/// so even a malformed rerun marks an older output stale.
pub fn requested_output_path(args: &[String]) -> Option<String> {
    let mut selected = None;
    let mut i = 0;
    while i < args.len() {
        let token = &args[i];
        if token == "--out" {
            if let Some(next) = args.get(i + 1).filter(|n| !n.starts_with('-')) {
                selected = Some(next.clone());
                i += 2;
                continue;
            }
        } else if let Some(v) = token.strip_prefix("--out=").filter(|v| !v.is_empty()) {
            selected = Some(v.to_string());
        }
        i += 1;
    }
    selected
}

// -------------------------------------------------------------- statistics

pub fn positive_finite(values: &[f64], label: &str) -> Res<()> {
    if values.is_empty() {
        return Err(format!("{label}: no samples"));
    }
    match values.iter().find(|v| !v.is_finite() || **v <= 0.0) {
        Some(bad) => Err(format!("{label}: expected positive finite sample, got {bad}")),
        None => Ok(()),
    }
}

pub fn geometric_mean(values: &[f64], label: &str) -> Res<f64> {
    positive_finite(values, label)?;
    Ok((values.iter().map(|v| v.ln()).sum::<f64>() / values.len() as f64).exp())
}

/// Robust location plus every outer-round value of a positive series.
#[derive(Clone, Debug)]
pub struct Summary {
    pub median: f64,
    pub min: f64,
    pub max: f64,
    /// `max / min`.
    pub spread: f64,
    pub values: Vec<f64>,
}

pub fn series_summary(values: &[f64], label: &str) -> Res<Summary> {
    positive_finite(values, label)?;
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(Summary {
        median: median(values.to_vec())?,
        min,
        max,
        spread: max / min,
        values: values.to_vec(),
    })
}

pub fn enforce_ratio_spread(summary: &Summary, limit: f64, label: &str) -> Res<()> {
    if !limit.is_finite() || !(1.0..=MAX_RATIO_SPREAD).contains(&limit) {
        return Err(format!(
            "{label}: ratio-spread limit must be in [1, {MAX_RATIO_SPREAD}]"
        ));
    }
    if !summary.spread.is_finite() {
        return Err(format!("{label}: missing/non-finite ratio spread"));
    }
    if summary.spread > limit {
        return Err(format!(
            "{label}: paired ratio spread {:.3}x exceeds bounded limit {limit:.3}x",
            summary.spread
        ));
    }
    Ok(())
}

// -------------------------------------------------------------------- JSON

/// A JSON document under construction. Rendering refuses a non-finite number,
/// so one can never replace prior evidence.
#[derive(Clone, Debug)]
pub enum Jv {
    Null,
    Bool(bool),
    Num(f64),
    Int(u64),
    Str(String),
    /// Text already validated as JSON by the strict parser.
    Raw(String),
    Arr(Vec<Jv>),
    Obj(Vec<(String, Jv)>),
}

impl Jv {
    pub fn str(s: impl Into<String>) -> Jv {
        Jv::Str(s.into())
    }

    pub fn nums(v: &[f64]) -> Jv {
        Jv::Arr(v.iter().map(|x| Jv::Num(*x)).collect())
    }

    pub fn obj<const N: usize>(fields: [(&str, Jv); N]) -> Jv {
        Jv::Obj(fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    pub fn opt_str(s: Option<String>) -> Jv {
        s.map_or(Jv::Null, Jv::Str)
    }

    pub fn render(&self) -> Res<String> {
        let mut out = String::new();
        self.write(&mut out, 0, "$")?;
        out.push('\n');
        Ok(out)
    }

    fn write(&self, out: &mut String, depth: usize, at: &str) -> Res<()> {
        let pad = |out: &mut String, d: usize| out.push_str(&"  ".repeat(d));
        match self {
            Jv::Null => out.push_str("null"),
            Jv::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Jv::Int(i) => out.push_str(&i.to_string()),
            Jv::Num(x) if x.is_finite() => out.push_str(&format!("{x}")),
            Jv::Num(x) => return Err(format!("non-finite number {x} at {at}")),
            Jv::Str(s) => json_escape(s, out),
            Jv::Raw(s) => out.push_str(s),
            Jv::Arr(items) if items.is_empty() => out.push_str("[]"),
            Jv::Arr(items) => {
                out.push_str("[\n");
                for (i, item) in items.iter().enumerate() {
                    pad(out, depth + 1);
                    item.write(out, depth + 1, &format!("{at}[{i}]"))?;
                    out.push_str(if i + 1 < items.len() { ",\n" } else { "\n" });
                }
                pad(out, depth);
                out.push(']');
            }
            Jv::Obj(fields) if fields.is_empty() => out.push_str("{}"),
            Jv::Obj(fields) => {
                out.push_str("{\n");
                for (i, (k, v)) in fields.iter().enumerate() {
                    pad(out, depth + 1);
                    json_escape(k, out);
                    out.push_str(": ");
                    v.write(out, depth + 1, &format!("{at}.{k}"))?;
                    out.push_str(if i + 1 < fields.len() { ",\n" } else { "\n" });
                }
                pad(out, depth);
                out.push('}');
            }
        }
        Ok(())
    }
}

fn json_escape(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

// ------------------------------------------------------------------ SHA-256

const K256: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

pub fn sha256_hex(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K256[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (a, b) in h.iter_mut().zip(v) {
            *a = a.wrapping_add(b);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

// -------------------------------------------------------- time and redaction

/// `2026-10-15T01:17:26Z`, from the system clock (civil-from-days, no crate).
pub fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64);
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// Paths beneath the invoking user's home are recorded `$HOME`-relative.
pub fn redact_text(value: &str) -> String {
    match std::env::var("HOME") {
        Ok(home) if home.len() > 1 => value.replace(&home, "$HOME"),
        _ => value.to_string(),
    }
}

fn display_path(path: &Path) -> String {
    let abs = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    redact_text(&abs.to_string_lossy())
}

pub fn shell_join(argv: &[String]) -> String {
    argv.iter()
        .map(|a| {
            if !a.is_empty() && a.chars().all(|c| c.is_ascii_alphanumeric() || "@%+=:,./-_".contains(c)) {
                a.clone()
            } else {
                format!("'{}'", a.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

// ------------------------------------------------------------- host probes

fn mtime_ns(meta: &std::fs::Metadata) -> Jv {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(Jv::Null, |d| Jv::Int(d.as_nanos() as u64))
}

/// `{path, size_bytes, mtime_ns, sha256}`, or `{path, unavailable}`.
pub fn file_record(path: &Path) -> Jv {
    let shown = Jv::str(display_path(path));
    match std::fs::metadata(path).and_then(|m| std::fs::read(path).map(|d| (m, d))) {
        Ok((meta, data)) => Jv::obj([
            ("path", shown),
            ("size_bytes", Jv::Int(meta.len())),
            ("mtime_ns", mtime_ns(&meta)),
            ("sha256", Jv::str(sha256_hex(&data))),
        ]),
        Err(e) => Jv::obj([("path", shown), ("unavailable", Jv::str(redact_text(&e.to_string())))]),
    }
}

/// The recorded sha256 of a [`file_record`], if it has one.
pub fn record_sha256(rec: &Jv) -> Option<&str> {
    match rec {
        Jv::Obj(f) => f.iter().find(|(k, _)| k == "sha256").and_then(|(_, v)| match v {
            Jv::Str(s) => Some(s.as_str()),
            _ => None,
        }),
        _ => None,
    }
}

pub struct RawRun {
    pub returncode: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub error: Option<String>,
}

/// Run `argv`, draining both pipes on threads (a large `git diff` would fill
/// one) and killing the child, which this function started, past `timeout`.
pub fn run_raw(argv: &[&str], cwd: Option<&Path>, timeout: Duration) -> RawRun {
    let fail = |e: String| RawRun {
        returncode: None,
        stdout: Vec::new(),
        stderr: Vec::new(),
        error: Some(e),
    };
    let mut cmd = Command::new(argv[0]);
    cmd.args(&argv[1..]).stdout(Stdio::piped()).stderr(Stdio::piped());
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return fail(e.to_string()),
    };
    let drain = |mut pipe: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = pipe.read_to_end(&mut buf);
            buf
        })
    };
    let out = drain(Box::new(child.stdout.take().expect("piped")));
    let err = drain(Box::new(child.stderr.take().expect("piped")));
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Ok(s),
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                break Err(format!("timed out after {}s", timeout.as_secs()));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(e) => break Err(e.to_string()),
        }
    };
    let (stdout, stderr) = (out.join().unwrap_or_default(), err.join().unwrap_or_default());
    match status {
        Ok(s) => RawRun {
            returncode: s.code(),
            stdout,
            stderr,
            error: None,
        },
        Err(e) => RawRun {
            returncode: None,
            stdout,
            stderr,
            error: Some(e),
        },
    }
}

fn clip(bytes: &[u8], limit: usize) -> String {
    String::from_utf8_lossy(bytes).trim().chars().take(limit).collect()
}

fn command_text(argv: &[&str]) -> Option<String> {
    let raw = run_raw(argv, None, Duration::from_secs(10));
    (raw.error.is_none() && raw.returncode == Some(0)).then(|| clip(&raw.stdout, 16_384))
}

pub fn command_record(argv: &[&str]) -> Jv {
    let raw = run_raw(argv, None, Duration::from_secs(10));
    let mut fields = vec![("argv".to_string(), Jv::Arr(argv.iter().map(|a| Jv::str(*a)).collect()))];
    if let Some(e) = raw.error {
        fields.push(("error".into(), Jv::str(redact_text(&e))));
        return Jv::Obj(fields);
    }
    fields.push((
        "returncode".into(),
        raw.returncode.map_or(Jv::Null, |c| Jv::Num(f64::from(c))),
    ));
    fields.push(("stdout".into(), Jv::str(redact_text(&clip(&raw.stdout, 16_384)))));
    let stderr = clip(&raw.stderr, 4_096);
    if !stderr.is_empty() {
        fields.push(("stderr".into(), Jv::str(redact_text(&stderr))));
    }
    Jv::Obj(fields)
}

/// Git revision, dirtiness and the hash of the tracked diff beneath `scope`.
/// `Err` when any probe could not run: provenance that names no revision is
/// not provenance.
pub fn git_snapshot(scope: &Path) -> Res<Jv> {
    let scope = std::path::absolute(scope).map_err(|e| e.to_string())?;
    let t = Duration::from_secs(20);
    let top = run_raw(&["git", "rev-parse", "--show-toplevel"], Some(&scope), t);
    let root = match (top.error.as_deref(), top.returncode) {
        (None, Some(0)) => clip(&top.stdout, 4096),
        _ => {
            return Err(format!(
                "git revision/dirty/diff provenance unavailable under {}",
                scope.display()
            ))
        }
    };
    let rel = scope
        .strip_prefix(&root)
        .map(|p| {
            if p.as_os_str().is_empty() {
                ".".into()
            } else {
                p.to_string_lossy().into_owned()
            }
        })
        .unwrap_or_else(|_| ".".into());
    let root_path = PathBuf::from(&root);
    let rev = run_raw(&["git", "rev-parse", "HEAD"], Some(&root_path), t);
    let status = run_raw(
        &["git", "status", "--porcelain=v1", "--untracked-files=all", "--", &rel],
        Some(&root_path),
        t,
    );
    let diff = run_raw(&["git", "diff", "--binary", "HEAD", "--", &rel], Some(&root_path), t);
    for (what, run) in [("rev-parse HEAD", &rev), ("status", &status), ("diff", &diff)] {
        if run.error.is_some() || run.returncode != Some(0) {
            return Err(format!(
                "git {what} failed: {}",
                redact_text(&run.error.clone().unwrap_or_else(|| clip(&run.stderr, 400)))
            ));
        }
    }
    let status_text = String::from_utf8_lossy(&status.stdout).into_owned();
    let lines: Vec<&str> = status_text.lines().collect();
    Ok(Jv::obj([
        ("available", Jv::Bool(true)),
        ("root", Jv::str(display_path(&root_path))),
        ("scope", Jv::str(rel)),
        ("revision", Jv::str(clip(&rev.stdout, 128))),
        ("dirty", Jv::Bool(!status_text.trim().is_empty())),
        (
            "status_porcelain",
            Jv::Arr(lines.iter().take(2000).map(|l| Jv::str(redact_text(l))).collect()),
        ),
        ("status_line_count", Jv::Int(lines.len() as u64)),
        ("status_truncated", Jv::Bool(lines.len() > 2000)),
        ("tracked_diff_sha256", Jv::str(sha256_hex(&diff.stdout))),
        ("tracked_diff_bytes", Jv::Int(diff.stdout.len() as u64)),
    ]))
}

fn load_snapshot() -> Jv {
    let mut l = [0f64; 3];
    // SAFETY: `getloadavg` writes at most `nelem` doubles into `l`, which holds 3.
    let n = unsafe { libc::getloadavg(l.as_mut_ptr(), 3) };
    if n != 3 {
        return Jv::obj([("unavailable", Jv::str("getloadavg"))]);
    }
    let cpus = std::thread::available_parallelism().map_or(0, |n| n.get());
    let norm = |x: f64| if cpus > 0 { Jv::Num(x / cpus as f64) } else { Jv::Null };
    Jv::obj([
        ("loadavg_1m", Jv::Num(l[0])),
        ("loadavg_5m", Jv::Num(l[1])),
        ("loadavg_15m", Jv::Num(l[2])),
        ("cpu_count_normalizer", Jv::Int(cpus as u64)),
        ("normalized_loadavg_1m", norm(l[0])),
        ("normalized_loadavg_5m", norm(l[1])),
        ("normalized_loadavg_15m", norm(l[2])),
    ])
}

fn power_source_record() -> Jv {
    let mut rec = command_record(&["pmset", "-g", "batt"]);
    if let Jv::Obj(fields) = &mut rec {
        for (k, v) in fields.iter_mut() {
            if let (true, Jv::Str(s)) = (k == "stdout", &*v) {
                // The battery id changes run to run and says nothing about speed.
                let cleaned: String = s
                    .lines()
                    .map(|l| match l.find(" (id=") {
                        Some(i) => format!("{}{}", &l[..i], l[i..].split_once(')').map_or("", |(_, r)| r)),
                        None => l.to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                *v = Jv::Str(cleaned);
            }
        }
    }
    rec
}

/// GPU model fields only: the display serials and host name do not affect
/// kernel speed and must not leak into an artifact.
fn device_snapshot() -> Jv {
    let argv = ["system_profiler", "SPDisplaysDataType", "-json"];
    let raw = run_raw(&argv, None, Duration::from_secs(10));
    let probe = Jv::Arr(argv.iter().map(|a| Jv::str(*a)).collect());
    if let Some(e) = raw.error {
        return Jv::obj([("probe_argv", probe), ("error", Jv::str(redact_text(&e)))]);
    }
    let syntax = Syntax {
        what: "system_profiler",
        max_depth: 16,
        uints_only: false,
        literals: true,
    };
    let text = String::from_utf8_lossy(&raw.stdout);
    let Ok(Json::Object(root)) = crate::json::parse(&text, syntax) else {
        return Jv::obj([("probe_argv", probe), ("parse_error", Jv::str("not a JSON object"))]);
    };
    const ALLOWED: [&str; 7] = [
        "_name",
        "spdisplays_mtlgpufamilysupport",
        "spdisplays_vendor",
        "sppci_bus",
        "sppci_cores",
        "sppci_device_type",
        "sppci_model",
    ];
    let gpus = root
        .iter()
        .find(|(k, _)| k == "SPDisplaysDataType")
        .and_then(|(_, v)| match v {
            Json::Array(a) => Some(a),
            _ => None,
        })
        .map(|gpus| {
            gpus.iter()
                .filter_map(|g| match g {
                    Json::Object(f) => Some(Jv::Obj(
                        f.iter()
                            .filter(|(k, _)| ALLOWED.contains(&k.as_str()))
                            .filter_map(|(k, v)| match v {
                                Json::Str(s) => Some((k.clone(), Jv::str(s.clone()))),
                                _ => None,
                            })
                            .collect(),
                    )),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default();
    Jv::obj([("probe_argv", probe), ("gpus", Jv::Arr(gpus))])
}

/// Versions of the packages a Python lane's interpreter imports.
pub fn python_packages(interpreter: &str) -> Jv {
    let code = "import importlib.metadata as m\n\
                for n in ('numpy','torch','mlx'):\n\
                \ttry: print(n, m.version(n))\n\
                \texcept m.PackageNotFoundError: print(n, '-')";
    let raw = run_raw(&[interpreter, "-I", "-c", code], None, Duration::from_secs(20));
    let text = String::from_utf8_lossy(&raw.stdout).into_owned();
    Jv::Obj(
        text.lines()
            .filter_map(|l| l.split_once(' '))
            .map(|(n, v)| (n.to_string(), if v == "-" { Jv::Null } else { Jv::str(v) }))
            .collect(),
    )
}

// ------------------------------------------------------------- provenance

/// What a run is built from; fixed at the start.
pub struct ProvenanceInputs<'a> {
    pub argv: &'a [String],
    pub scope: &'a Path,
    /// Name and file of each input whose bytes are hashed.
    pub files: Vec<(String, PathBuf)>,
    pub config: Jv,
    pub probe_device: bool,
}

/// Capture the run inputs and environment. `Err` when the git state or any
/// input hash is unavailable.
pub fn start_provenance(inp: &ProvenanceInputs) -> Res<Jv> {
    let git = git_snapshot(inp.scope)?;
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let invocation: Vec<String> = std::iter::once(exe.to_string_lossy().into_owned())
        .chain(inp.argv.iter().cloned())
        .map(|a| redact_text(&a))
        .collect();
    let mut inputs = vec![("driver".to_string(), file_record(&exe))];
    inputs.extend(inp.files.iter().map(|(n, p)| (n.clone(), file_record(p))));
    let missing: Vec<&str> = inputs
        .iter()
        .filter(|(_, r)| record_sha256(r).is_none())
        .map(|(n, _)| n.as_str())
        .collect();
    if !missing.is_empty() {
        return Err(format!("executable input hashes unavailable: {missing:?}"));
    }
    let sysctl_text = |name: &str| Jv::opt_str(command_text(&["sysctl", "-n", name]).filter(|s| !s.is_empty()));
    let sysctl_int = |name: &str| {
        command_text(&["sysctl", "-n", name])
            .and_then(|s| s.parse::<u64>().ok())
            .map_or(Jv::Null, Jv::Int)
    };
    let cpus = std::thread::available_parallelism().map_or(0, |n| n.get() as u64);
    Ok(Jv::obj([
        ("schema_version", Jv::Int(1)),
        ("started_at_utc", Jv::str(utc_now())),
        ("invocation_shell", Jv::str(shell_join(&invocation))),
        (
            "invocation_argv",
            Jv::Arr(invocation.iter().map(|a| Jv::str(a.clone())).collect()),
        ),
        (
            "cwd",
            Jv::str(display_path(&std::env::current_dir().map_err(|e| e.to_string())?)),
        ),
        (
            "path_policy",
            Jv::str(
                "paths beneath the invoking user's home are recorded as $HOME-relative; \
                 hashes are computed from the real files",
            ),
        ),
        (
            "environment_policy",
            Jv::str(format!(
                "full process environment not captured; with --clean-env the inherited benchmark/tuning \
                 prefixes {BENCHMARK_ENV_PREFIXES:?} are cleared and benchmark_config records only the \
                 explicit lane overrides"
            )),
        ),
        ("benchmark_config", inp.config.clone()),
        ("git", git),
        (
            "host",
            Jv::obj([
                ("machine", Jv::str(std::env::consts::ARCH)),
                ("cpu_count", Jv::Int(cpus)),
                ("hardware_model", sysctl_text("hw.model")),
                ("memory_bytes", sysctl_int("hw.memsize")),
            ]),
        ),
        (
            "os",
            Jv::obj([
                ("system", Jv::str(std::env::consts::OS)),
                ("release", sysctl_text("kern.osrelease")),
                ("version", sysctl_text("kern.version")),
                ("sw_vers", command_record(&["sw_vers"])),
            ]),
        ),
        ("runtime", Jv::obj([("rustc", command_record(&["rustc", "--version"]))])),
        (
            "power",
            Jv::obj([
                ("thermal_pressure", command_record(&["pmset", "-g", "therm"])),
                ("power_source", power_source_record()),
            ]),
        ),
        ("load", Jv::obj([("start", load_snapshot())])),
        ("inputs", Jv::Obj(inputs)),
        (
            "device",
            if inp.probe_device {
                device_snapshot()
            } else {
                Jv::obj([("probe", Jv::str("disabled_for_test"))])
            },
        ),
    ]))
}

/// Stamp the finish time, load and power state onto a started provenance.
pub fn finish_provenance(prov: &mut Jv) {
    let Jv::Obj(fields) = prov else { return };
    fields.push(("finished_at_utc".into(), Jv::str(utc_now())));
    for (k, v) in fields.iter_mut() {
        match (k.as_str(), v) {
            ("load", Jv::Obj(f)) => f.push(("finish".into(), load_snapshot())),
            ("power", Jv::Obj(f)) => {
                f.push((
                    "thermal_pressure_finish".into(),
                    command_record(&["pmset", "-g", "therm"]),
                ));
                f.push(("power_source_finish".into(), power_source_record()));
            }
            _ => {}
        }
    }
}

// ------------------------------------------------------------- publication

/// Replace `path` with `bytes` durably, exposing no partial content.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Res<()> {
    let abs = std::path::absolute(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let dir = abs
        .parent()
        .ok_or_else(|| format!("{}: no parent directory", abs.display()))?;
    let name = abs.file_name().and_then(|n| n.to_str()).unwrap_or("out");
    if !dir.is_dir() {
        return Err(format!(
            "{}: parent directory does not exist (never created implicitly)",
            abs.display()
        ));
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let staged = dir.join(format!(".{name}.{}.{nanos}", std::process::id()));
    let result = (|| -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(&staged)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        std::fs::rename(&staged, &abs)?;
        std::fs::File::open(dir)?.sync_all()
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&staged);
        return Err(format!("{}: {e}", abs.display()));
    }
    Ok(())
}

/// Atomic publication plus a fail-closed marker for the latest attempt. An
/// older valid output survives a failed rerun, but the sidecar starts in
/// `not_published`, so that file cannot be read as evidence from the newer
/// attempt, even after a crash.
pub struct EvidenceOutput {
    output: PathBuf,
    marker: PathBuf,
    record: Vec<(String, Jv)>,
}

impl EvidenceOutput {
    pub fn new(output: &Path, argv: &[String]) -> Res<Self> {
        let output = std::path::absolute(output).map_err(|e| e.to_string())?;
        let marker = PathBuf::from(format!("{}.attempt.json", output.display()));
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let invocation: Vec<String> = std::iter::once(exe.to_string_lossy().into_owned())
            .chain(argv.iter().cloned())
            .map(|a| redact_text(&a))
            .collect();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let record = vec![
            ("schema_version".to_string(), Jv::Int(1)),
            (
                "attempt_id".into(),
                Jv::str(format!("{:x}-{:x}", std::process::id(), nanos)),
            ),
            ("status".into(), Jv::str("not_published")),
            ("started_at_utc".into(), Jv::str(utc_now())),
            ("invocation_shell".into(), Jv::str(shell_join(&invocation))),
            (
                "invocation_argv".into(),
                Jv::Arr(invocation.iter().map(|a| Jv::str(a.clone())).collect()),
            ),
            ("output_path".into(), Jv::str(display_path(&output))),
            (
                "prior_output".into(),
                if output.exists() {
                    file_record(&output)
                } else {
                    Jv::Null
                },
            ),
            (
                "interpretation".into(),
                Jv::str("Unless status is published, output_path does not represent this attempt."),
            ),
        ];
        Ok(Self { output, marker, record })
    }

    pub fn marker_path(&self) -> &Path {
        &self.marker
    }

    fn write_marker(&self) -> Res<()> {
        atomic_write(&self.marker, Jv::Obj(self.record.clone()).render()?.as_bytes())
    }

    pub fn begin(&self) -> Res<()> {
        self.write_marker()
    }

    /// Render first: a document with a non-finite number never reaches disk.
    pub fn publish(&mut self, doc: &Jv) -> Res<()> {
        let text = doc.render()?;
        atomic_write(&self.output, text.as_bytes())?;
        for (k, v) in &mut self.record {
            if k == "status" {
                *v = Jv::str("published");
            }
        }
        self.record.push(("finished_at_utc".into(), Jv::str(utc_now())));
        self.record.push(("published_output".into(), file_record(&self.output)));
        self.write_marker()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("tessl-evidence-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn sha256_matches_the_fips_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn utc_formatting_is_rfc3339() {
        let s = utc_now();
        assert_eq!(s.len(), 20, "{s}");
        assert!(s.starts_with("20") && s.ends_with('Z') && &s[10..11] == "T");
    }

    #[test]
    fn drift_policy_defaults_and_caps() {
        assert_eq!(
            (DEFAULT_MAX_RATIO_SPREAD, MAX_RATIO_SPREAD, DEFAULT_OUTER_ROUNDS),
            (1.10, 1.25, 6)
        );
        assert_eq!(parse_ratio_spread_limit("1.25").unwrap(), 1.25);
        for bad in ["nan", "inf", "0.99", "1.251", "", "x"] {
            assert!(parse_ratio_spread_limit(bad).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn low_or_odd_round_counts_and_thin_inner_samples_are_refused() {
        assert!(validate_evidence_sample_counts(4, Some(3)).is_ok());
        assert!(validate_evidence_sample_counts(6, None).is_ok());
        for rounds in [0, 1, 2, 3, 5] {
            assert!(
                validate_evidence_sample_counts(rounds, Some(3)).is_err(),
                "rounds {rounds}"
            );
        }
        for iters in [0, 1, 2] {
            assert!(
                validate_evidence_sample_counts(4, Some(iters)).is_err(),
                "iters {iters}"
            );
        }
    }

    #[test]
    fn output_discovery_uses_the_last_usable_value() {
        let a = args(&["--out", "first.json", "--out=second.json"]);
        assert_eq!(requested_output_path(&a).as_deref(), Some("second.json"));
        assert_eq!(requested_output_path(&args(&["--out", "--rounds"])), None);
        assert_eq!(requested_output_path(&args(&["--out="])), None);
    }

    #[test]
    fn clean_env_drops_inherited_knobs_and_keeps_the_rest() {
        let base = [
            ("TESSL_ATTN_TILED", "1"),
            ("BENCH_ITERS", "9"),
            ("PATH", "/bin"),
            ("MLX_X", "1"),
        ]
        .map(|(k, v)| (k.to_string(), v.to_string()));
        let clean = clean_benchmark_env(base);
        assert_eq!(clean.keys().map(String::as_str).collect::<Vec<_>>(), vec!["PATH"]);
    }

    #[test]
    fn spread_gate_and_aggregates_refuse_bad_series() {
        let s = series_summary(&[1.0, 1.05, 1.02], "x").unwrap();
        assert_eq!(s.median, 1.02);
        assert!((s.spread - 1.05).abs() < 1e-12);
        enforce_ratio_spread(&s, 1.10, "x").unwrap();
        let wide = series_summary(&[1.0, 1.3], "x").unwrap();
        assert!(enforce_ratio_spread(&wide, 1.10, "x")
            .unwrap_err()
            .contains("exceeds bounded limit"));
        assert!(enforce_ratio_spread(&s, 1.5, "x").is_err());
        assert!(series_summary(&[], "x").is_err());
        assert!(series_summary(&[1.0, 0.0], "x").is_err());
        assert!(series_summary(&[1.0, f64::NAN], "x").is_err());
        assert!((geometric_mean(&[2.0, 8.0], "g").unwrap() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn provenance_records_revision_inputs_load_and_redacts_home() {
        let d = scratch("prov");
        let fixture = d.join("fixture");
        std::fs::write(&fixture, b"fixture bytes").unwrap();
        let argv = args(&["--synthetic", "value with spaces"]);
        let mut prov = start_provenance(&ProvenanceInputs {
            argv: &argv,
            scope: Path::new("."),
            files: vec![("fixture".into(), fixture.clone())],
            config: Jv::obj([("rounds", Jv::Int(6))]),
            probe_device: false,
        })
        .unwrap();
        finish_provenance(&mut prov);
        let text = prov.render().unwrap();
        for needle in [
            "\"revision\"",
            "\"dirty\"",
            "\"tracked_diff_sha256\"",
            "\"thermal_pressure_finish\"",
            "\"power_source_finish\"",
            "\"finish\"",
            "\"environment_policy\"",
            "\"path_policy\"",
            "'value with spaces'",
            &sha256_hex(b"fixture bytes"),
        ] {
            assert!(text.contains(needle), "provenance lacks {needle}");
        }
        assert!(
            !text.contains("\"environment\":"),
            "the full environment must not be captured"
        );
        if let Ok(home) = std::env::var("HOME") {
            if home.len() > 1 {
                assert!(!text.contains(&home), "home path leaked into provenance");
            }
        }
        let missing = start_provenance(&ProvenanceInputs {
            argv: &argv,
            scope: Path::new("."),
            files: vec![("gone".into(), d.join("absent"))],
            config: Jv::Null,
            probe_device: false,
        });
        assert!(missing.unwrap_err().contains("hashes unavailable"));
    }

    #[test]
    fn a_new_attempt_marks_an_older_output_stale_then_publishes_atomically() {
        let d = scratch("publish");
        let out = d.join("result.json");
        std::fs::write(&out, "{\"generation\":\"old\"}").unwrap();
        let mut publisher = EvidenceOutput::new(&out, &args(&["--out", out.to_str().unwrap()])).unwrap();
        publisher.begin().unwrap();
        assert_eq!(std::fs::read_to_string(&out).unwrap(), "{\"generation\":\"old\"}");
        let pending = std::fs::read_to_string(publisher.marker_path()).unwrap();
        assert!(pending.contains("\"status\": \"not_published\""));
        assert!(
            pending.contains(&sha256_hex(b"{\"generation\":\"old\"}")),
            "prior output hash"
        );

        publisher.publish(&Jv::obj([("generation", Jv::str("new"))])).unwrap();
        assert!(std::fs::read_to_string(&out).unwrap().contains("\"new\""));
        let marker = std::fs::read_to_string(publisher.marker_path()).unwrap();
        assert!(marker.contains("\"status\": \"published\"") && marker.contains("published_output"));
        let staged: Vec<_> = std::fs::read_dir(&d)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(".result.json."))
            .collect();
        assert!(staged.is_empty(), "staged files left behind");
    }

    #[test]
    fn non_finite_json_cannot_replace_prior_evidence() {
        let d = scratch("nonfinite");
        let out = d.join("nonfinite.json");
        std::fs::write(&out, "{\"generation\":\"old\"}").unwrap();
        let mut publisher = EvidenceOutput::new(&out, &[]).unwrap();
        publisher.begin().unwrap();
        let err = publisher.publish(&Jv::obj([("ratio", Jv::Num(f64::NAN))])).unwrap_err();
        assert!(err.contains("non-finite"), "{err}");
        assert_eq!(std::fs::read_to_string(&out).unwrap(), "{\"generation\":\"old\"}");
        assert!(std::fs::read_to_string(publisher.marker_path())
            .unwrap()
            .contains("not_published"));
    }

    #[test]
    fn output_parents_must_pre_exist() {
        let d = scratch("parent");
        let out = d.join("missing-parent").join("result.json");
        let publisher = EvidenceOutput::new(&out, &[]).unwrap();
        assert!(publisher.begin().unwrap_err().contains("never created implicitly"));
        assert!(!d.join("missing-parent").exists());
    }
}
