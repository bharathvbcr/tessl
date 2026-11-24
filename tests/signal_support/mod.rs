//! Start a test's child with SIGINT, SIGQUIT and SIGHUP at their default action,
//! whatever this test process inherited, and prove a test does not depend on it.
//!
//! A `cargo test` started as a background job (`&` in a non-interactive shell,
//! or a wrapper such as the cargo slot gate, which runs cargo that way) inherits
//! SIGINT and SIGQUIT ignored, and one under `nohup` inherits SIGHUP ignored.
//! exec keeps an ignored signal ignored, bash cannot trap a signal that was
//! ignored on entry, and `src/bin/common/proc.rs` deliberately does not install
//! its handler over `SIG_IGN` (`tests/kernel_coverage.rs` pins that). So a test
//! that signals a child it started must hand the child the defaults a terminal
//! would; otherwise it tests whatever the harness happened to inherit.
//!
//! Used by the test binaries through `mod signal_support;`. Not every binary
//! uses every item.
#![allow(dead_code)]

use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};

/// The signals a background or `nohup` launch hands down ignored.
pub const INHERITED_IGNORABLE: [libc::c_int; 3] = [libc::SIGINT, libc::SIGQUIT, libc::SIGHUP];

/// Reset [`INHERITED_IGNORABLE`] to `SIG_DFL` in the child `cmd` starts.
pub fn default_signals(cmd: &mut Command) -> &mut Command {
    // SAFETY: the closure runs in the forked child before exec. It calls only
    // `signal` and, on failure, reads errno into an `io::Error` (no allocation);
    // both are async-signal-safe.
    unsafe { cmd.pre_exec(|| set_all(libc::SIG_DFL)) }
}

/// Set [`INHERITED_IGNORABLE`] to `SIG_IGN` in the child `cmd` starts, as a
/// background launch would.
pub fn ignored_signals(cmd: &mut Command) -> &mut Command {
    // SAFETY: as in `default_signals`: only `signal` and an errno read run
    // between fork and exec.
    unsafe { cmd.pre_exec(|| set_all(libc::SIG_IGN)) }
}

fn set_all(action: libc::sighandler_t) -> std::io::Result<()> {
    for sig in INHERITED_IGNORABLE {
        // SAFETY: `signal` with a valid signal number and SIG_DFL or SIG_IGN.
        if unsafe { libc::signal(sig, action) } == libc::SIG_ERR {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

/// Set by [`rerun_with_signals_ignored`] for its re-run, so that the re-run
/// tests can check, with [`expect_ignored_if_rerun`], that they really started
/// with the signals ignored.
const EXPECT_IGNORED_ENV: &str = "TESSL_SIGNAL_SUPPORT_EXPECT_IGNORED";

/// In a re-run started by [`rerun_with_signals_ignored`], require that this
/// process started with every one of [`INHERITED_IGNORABLE`] ignored, so a
/// passing re-run proves what it claims. Does nothing in a direct run, which
/// may have inherited either disposition. Call it first in a re-run test.
pub fn expect_ignored_if_rerun() {
    if std::env::var_os(EXPECT_IGNORED_ENV).is_none() {
        return;
    }
    for sig in INHERITED_IGNORABLE {
        // SAFETY: `sigaction` is a plain C struct; all-zero is a valid value.
        let mut old: libc::sigaction = unsafe { std::mem::zeroed() };
        // SAFETY: a query only (a null new action); `old` is a valid out-parameter.
        let r = unsafe { libc::sigaction(sig, std::ptr::null(), &mut old) };
        assert_eq!(r, 0, "sigaction({sig}): {}", std::io::Error::last_os_error());
        assert_eq!(
            old.sa_sigaction,
            libc::SIG_IGN,
            "the re-run did not start with signal {sig} ignored"
        );
    }
}

/// Re-run `names` from this test binary, each with `--exact`, in a child that
/// starts with [`INHERITED_IGNORABLE`] ignored, and require that exactly
/// `names.len()` tests ran and all passed (a name that matches nothing also
/// exits 0, so the count is checked). `env` is added to the child's environment.
pub fn rerun_with_signals_ignored(names: &[&str], env: &[(&str, &str)]) {
    let mut cmd = Command::new(std::env::current_exe().expect("the running test binary"));
    cmd.arg("--exact")
        .arg("--test-threads=1")
        .args(names)
        .envs(env.iter().copied())
        .env(EXPECT_IGNORED_ENV, "1")
        .stdin(Stdio::null());
    ignored_signals(&mut cmd);
    let out = cmd.output().expect("re-run the test binary");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let want = format!("test result: ok. {} passed; 0 failed", names.len());
    assert!(
        out.status.success() && stdout.contains(&want),
        "with SIGINT, SIGQUIT and SIGHUP ignored on entry, {names:?} did not all pass \
         (status {:?}, wanted {want:?}):\n{stdout}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
}
