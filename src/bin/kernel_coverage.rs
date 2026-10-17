//! Which kernels does the benchmark suite actually dispatch?
//!
//! ```text
//! kernel_coverage [--check] [--metallib PATH] [--out PATH] [--allowlist PATH]
//! kernel_coverage --audit-allowlist [--metallib PATH] [--allowlist PATH]
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
//! * 0   the check ran and every kernel entry point is dispatched by a bench or
//!   listed in `bench/kernel_coverage_allowlist.txt`
//! * 1   the check ran and found gaps (undispatched kernels that are not
//!   allowlisted, an allowlist entry that is stale or that a bench dispatches
//!   after all, a malformed allowlist, a census that disagrees with the
//!   metallib, a kernel traced that is not in the census)
//! * 77  the check could not run: no toolchain (`xcrun metal` / `metal-nm`), no
//!   binary, a stale binary, no metallib, a suite that failed or printed no
//!   trace. 77 is the repo's SKIPPED status (`scripts/check_kernel_ir.sh`) and is
//!   never success, and never the same status as "ran and found gaps"
//! * 2   bad command line
//!
//! Without `--check` a gap is reported and the exit status stays 0.
//! `--audit-allowlist` runs no benchmark: it checks the allowlist against the
//! census alone, which a runner with the Metal toolchain but no GPU can do.

#[path = "common/kcov.rs"]
mod kcov;
#[path = "common/proc.rs"]
mod proc;
#[path = "common/pyjson.rs"]
mod pyjson;

use kcov::{Failure, Res};
use std::collections::BTreeSet;
use std::path::PathBuf;

const USAGE: &str = "usage: kernel_coverage [-h] [--check] [--metallib METALLIB] [--out OUT] [--audit-allowlist]\n\
    \x20                      [--allowlist ALLOWLIST]\n\n\
    options:\n  \
    -h, --help           show this help message and exit\n  \
    --check              exit 1 if a kernel is neither dispatched by a bench nor allowlisted, or an\n                       \
    allowlist entry is stale or redundant (77 if the check cannot run)\n  \
    --metallib METALLIB  explicit compiled metallib to inventory (default: the library bench_gemm_sweep\n                       \
    embeds, via --dump-metallib)\n  \
    --out OUT            also write the coverage document here as JSON\n  \
    --audit-allowlist    runs no benchmark: exit 1 if an allowlist entry is not a kernel in the census\n                       \
    (what a runner with the Metal toolchain but no GPU can check; 77 if the toolchain is missing)\n  \
    --allowlist ALLOWLIST\n                       \
    kernels no bench dispatches, each with a reason (default: bench/kernel_coverage_allowlist.txt)\n\n\
    exit status: 0 ran and every kernel is dispatched or allowlisted; 1 ran and found gaps (or a census/metallib\n\
    disagreement); 77 could not run (no toolchain, binary, metallib or trace, a stale binary, or a\n\
    failed suite); 2 bad command line.";

/// One suite is a single bench run at one iteration; an hour is a hang.
const SUITE_SECS: u64 = 3600;

struct Args {
    check: bool,
    metallib: Option<PathBuf>,
    out: Option<PathBuf>,
    audit_allowlist: bool,
    allowlist: PathBuf,
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
        audit_allowlist: false,
        allowlist: kcov::allowlist_path(),
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
            "--audit-allowlist" if inline.is_none() => a.audit_allowlist = true,
            "--allowlist" => a.allowlist = PathBuf::from(value("--allowlist")),
            _ => usage_error(&format!("unrecognized arguments: {arg}")),
        }
    }
    a
}

fn run(args: &Args) -> Res<()> {
    // A malformed allowlist stops here, before any suite runs.
    let allowed = kcov::load_allowlist(&args.allowlist, &kcov::crate_root())?;

    if args.audit_allowlist {
        // The census is the preprocessor's, so it needs no binary; an explicit
        // --metallib is still cross-checked.
        let census = kcov::inventory(&kcov::kernel_dir(), args.metallib.as_deref(), false)?;
        let (stale, _) = kcov::audit_allowlist(&allowed, &census.names, &BTreeSet::new());
        if !stale.is_empty() {
            return Err(Failure::Found(format!(
                "{} allowlist entr(ies) are not kernels any more: {}",
                stale.len(),
                pyjson::quote_list(&stale)
            )));
        }
        println!(
            "allowlist: {} entries, every one a kernel of the {}-entry census",
            allowed.len(),
            census.names.len()
        );
        return Ok(());
    }

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

    let report = kcov::report(&census, &traced, &allowed)?;
    print!("{}", report.text);
    if let Some(out) = &args.out {
        let text = report.doc.dumps().map_err(Failure::CannotRun)?;
        std::fs::write(out, text).map_err(|e| Failure::CannotRun(format!("write {}: {e}", out.display())))?;
    }
    let problems = format!("\n{}", report.problems.join("\n"));
    if args.check && !report.problems.is_empty() {
        return Err(Failure::Found(problems));
    }
    if report.problems.is_empty() {
        println!("\nPASS: every kernel entry point is dispatched by a benchmark or allowlisted");
    } else {
        println!("{problems}");
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
