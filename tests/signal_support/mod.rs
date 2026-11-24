//! Start a test's child with the signals a test sends at their default action,
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

use std::io::{Read, Write};
use std::os::unix::process::CommandExt;
use std::process::{Child, ChildStdin, Command, Stdio};

/// The signals a background or `nohup` launch hands down ignored.
pub const INHERITED_IGNORABLE: [libc::c_int; 3] = [libc::SIGINT, libc::SIGQUIT, libc::SIGHUP];

/// The signals [`default_signals`] resets: [`INHERITED_IGNORABLE`] and SIGTERM,
/// which the tests also send and which a parent may have ignored.
pub const SENT_BY_TESTS: [libc::c_int; 4] = [libc::SIGINT, libc::SIGQUIT, libc::SIGHUP, libc::SIGTERM];

/// Reset [`SENT_BY_TESTS`] to `SIG_DFL` in the child `cmd` starts. A caller
/// that wants some of them ignored, or other set-up, registers its own
/// `pre_exec` after this one: the closures run in the order they were added.
pub fn default_signals(cmd: &mut Command) -> &mut Command {
    // SAFETY: the closure runs in the forked child before exec. It calls only
    // `signal` and, on failure, reads errno into an `io::Error` (no allocation);
    // both are async-signal-safe.
    unsafe { cmd.pre_exec(|| set_all(&SENT_BY_TESTS, libc::SIG_DFL)) }
}

/// Set [`INHERITED_IGNORABLE`] to `SIG_IGN` in the child `cmd` starts, as a
/// background launch would.
pub fn ignored_signals(cmd: &mut Command) -> &mut Command {
    // SAFETY: as in `default_signals`: only `signal` and an errno read run
    // between fork and exec.
    unsafe { cmd.pre_exec(|| set_all(&INHERITED_IGNORABLE, libc::SIG_IGN)) }
}

fn set_all(sigs: &[libc::c_int], action: libc::sighandler_t) -> std::io::Result<()> {
    for &sig in sigs {
        // SAFETY: `signal` with a valid signal number and SIG_DFL or SIG_IGN.
        if unsafe { libc::signal(sig, action) } == libc::SIG_ERR {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

/// Set by [`spawn_rerun`] for its re-run, so that the re-run tests can check,
/// with [`prepare_if_rerun`], that they really started with the signals ignored.
const EXPECT_IGNORED_ENV: &str = "TESSL_SIGNAL_SUPPORT_EXPECT_IGNORED";

/// What a re-run prints to stderr when it stops because its parent is gone.
pub const PARENT_GONE: &str = "signal_support: the test that started this re-run is gone; stopping the re-run";

/// The exit status of a re-run that stopped because its parent is gone.
pub const PARENT_GONE_EXIT: i32 = 3;

/// In a re-run started by [`spawn_rerun`]: require that this process started
/// with every one of [`INHERITED_IGNORABLE`] ignored, so a passing re-run proves
/// what it claims, and stop this process once the test that started it is gone.
/// Does nothing in a direct run, which may have inherited either disposition.
/// Call it first in a re-run test.
pub fn prepare_if_rerun() {
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
    stop_when_parent_is_gone();
}

/// A Ctrl-C kills the test that started a re-run, but not the re-run, which
/// starts with SIGINT ignored: its tests would run on to their own deadlines.
/// Its stdin is a pipe whose write end only that parent holds ([`Rerun`]), so
/// end of file on stdin means the parent is gone, and the re-run exits.
fn stop_when_parent_is_gone() {
    static WATCH: std::sync::Once = std::sync::Once::new();
    WATCH.call_once(|| {
        std::thread::Builder::new()
            .name("rerun-parent-watch".into())
            .spawn(|| {
                let mut buf = [0u8; 64];
                loop {
                    match std::io::stdin().read(&mut buf) {
                        Ok(0) => break,
                        Ok(_) => continue,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(_) => break,
                    }
                }
                // Straight to fd 2: the test harness captures `eprintln!`, and
                // the capture is lost at exit.
                let _ = std::io::stderr().write_all(format!("{PARENT_GONE}\n").as_bytes());
                std::process::exit(PARENT_GONE_EXIT);
            })
            .expect("start the re-run's parent watch");
    });
}

/// A started re-run of some tests of this binary (see [`spawn_rerun`]).
/// `parent_alive` is the write end of the re-run's stdin: the re-run stops soon
/// after it is closed, which this process's death also does.
pub struct Rerun {
    pub child: Child,
    pub parent_alive: ChildStdin,
}

/// Start `names` from this test binary, each with `--exact`, in a child that
/// starts with [`INHERITED_IGNORABLE`] ignored, with stdout and stderr piped.
/// `env` is added to the child's environment.
pub fn spawn_rerun(names: &[&str], env: &[(&str, &str)]) -> Rerun {
    let mut cmd = Command::new(std::env::current_exe().expect("the running test binary"));
    cmd.arg("--exact")
        .arg("--test-threads=1")
        .args(names)
        .envs(env.iter().copied())
        .env(EXPECT_IGNORED_ENV, "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    ignored_signals(&mut cmd);
    let mut child = cmd.spawn().expect("re-run the test binary");
    let parent_alive = child.stdin.take().expect("the re-run's stdin is piped");
    Rerun { child, parent_alive }
}

/// Re-run `names` with [`spawn_rerun`] and require that exactly `names.len()`
/// tests ran and all passed (a name that matches nothing also exits 0, so the
/// count is checked).
pub fn rerun_with_signals_ignored(names: &[&str], env: &[(&str, &str)]) {
    let Rerun { child, parent_alive } = spawn_rerun(names, env);
    // `wait_with_output` would close a stdin it held; this one is held apart,
    // open until the re-run has ended.
    let out = child.wait_with_output().expect("wait for the re-run");
    drop(parent_alive);
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
