//! Which kernels does the benchmark suite actually dispatch?
//!
//! ```text
//! kernel_coverage [--check] [--metallib PATH] [--out PATH]
//! ```
//!
//! Runs the small bench suites in `common/kcov.rs` with `TESSL_KERNEL_TRACE=1`,
//! unions the kernels each dispatched, and compares that with the census of every
//! kernel entry point in `kernels/*.metal` (read from the real preprocessor and
//! cross-checked against the metallib the release binaries embed). The method,
//! and why a source scan was not good enough, is in `common/kcov.rs`.
//!
//! Exit status (the same on every path; documented in docs/benchmarking.md):
//!
//! * 0   the check ran and every kernel entry point is dispatched
//! * 1   the check ran and found gaps (undispatched kernels, a census that
//!   disagrees with the metallib, a kernel traced that is not in the census)
//! * 77  the check could not run: no toolchain (`xcrun metal` / `metal-nm`), no
//!   binary, a stale binary, no metallib, a suite that failed or printed no
//!   trace. 77 is the repo's SKIPPED status (`scripts/check_kernel_ir.sh`) and is
//!   never success, and never the same status as "ran and found gaps"
//! * 2   bad command line
//!
//! Without `--check` a gap is reported and the exit status stays 0.

#[path = "common/kcov.rs"]
mod kcov;
#[path = "common/proc.rs"]
mod proc;
#[path = "common/pyjson.rs"]
mod pyjson;

use kcov::{Failure, Res};
use std::path::PathBuf;

const USAGE: &str = "usage: kernel_coverage [-h] [--check] [--metallib METALLIB] [--out OUT]\n\n\
    options:\n  \
    -h, --help           show this help message and exit\n  \
    --check              exit 1 if any kernel is never dispatched (77 if the check cannot run)\n  \
    --metallib METALLIB  explicit compiled metallib to inventory (default: the library bench_gemm_sweep\n                       \
    embeds, via --dump-metallib)\n  \
    --out OUT            also write the coverage document here as JSON\n\n\
    exit status: 0 ran and every kernel is dispatched; 1 ran and found gaps (or a census/metallib\n\
    disagreement); 77 could not run (no toolchain, binary, metallib or trace, a stale binary, or a\n\
    failed suite); 2 bad command line.";

/// One suite is a single bench run at one iteration; an hour is a hang.
const SUITE_SECS: u64 = 3600;

struct Args {
    check: bool,
    metallib: Option<PathBuf>,
    out: Option<PathBuf>,
}

fn usage_error(msg: &str) -> ! {
    eprintln!("kernel_coverage: error: {msg}\n{USAGE}");
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut a = Args {
        check: false,
        metallib: None,
        out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let (name, inline) = match arg.split_once('=') {
            Some((n, v)) if n.starts_with("--") => (n.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut value = |what: &str| {
            inline
                .clone()
                .or_else(|| it.next())
                .unwrap_or_else(|| usage_error(&format!("{what} needs a value")))
        };
        match name.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "--check" if inline.is_none() => a.check = true,
            "--metallib" => a.metallib = Some(PathBuf::from(value("--metallib"))),
            "--out" => a.out = Some(PathBuf::from(value("--out"))),
            _ => usage_error(&format!("unrecognized arguments: {arg}")),
        }
    }
    a
}

fn run(args: &Args) -> Res<()> {
    let tmp = kcov::scratch_dir("kernel-coverage")?;
    // A command-line coverage result is evidence about the shipped binary, not
    // just a source grep. If the compiled-library census cannot run, stop before
    // executing suites or printing a percentage.
    let census = kcov::inventory(&kcov::kernel_dir(), args.metallib.as_deref(), true)?;
    let traced = kcov::run_suites(
        &kcov::suites(),
        &kcov::bin_dir(),
        tmp.path(),
        &kcov::crate_root(),
        SUITE_SECS,
    );
    drop(tmp);

    if !traced.failures.is_empty() {
        for f in &traced.failures {
            eprintln!("ERROR  {f}");
        }
        return Err(Failure::CannotRun(
            "coverage run incomplete; refusing to report a partial sweep as a coverage measurement".to_string(),
        ));
    }

    let report = kcov::report(&census, &traced)?;
    print!("{}", report.text);
    if let Some(out) = &args.out {
        let text = report.doc.dumps().map_err(Failure::CannotRun)?;
        std::fs::write(out, text).map_err(|e| Failure::CannotRun(format!("write {}: {e}", out.display())))?;
    }
    if args.check && report.missing > 0 {
        return Err(Failure::Found(format!(
            "\n{} kernel(s) have no timing lane",
            report.missing
        )));
    }
    if report.missing == 0 {
        println!("\nPASS: every kernel entry point is dispatched by a benchmark");
    } else {
        println!();
    }
    Ok(())
}

fn main() {
    let args = parse_args();
    match run(&args) {
        Ok(()) => {}
        Err(Failure::CannotRun(msg)) => {
            eprintln!("COULD NOT RUN: {msg}");
            std::process::exit(kcov::EXIT_CANNOT_RUN);
        }
        Err(Failure::Found(msg)) => {
            eprintln!("{msg}");
            std::process::exit(kcov::EXIT_GAPS);
        }
    }
}
