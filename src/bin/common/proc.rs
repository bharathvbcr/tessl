//! Run a child process to completion within a bound.
//!
//! The tools that drive other programs (`kernel_coverage` runs the compiler,
//! `metal-nm` and the bench suites; `attn_tune` runs `bench_flash_attn`) must not
//! wait forever on one that hangs, and must not deadlock on one that writes more
//! than a pipe holds. Included with `#[path]` by those bins and by the
//! integration tests of each; it has no dependency of its own.

#![allow(dead_code)] // Each includer uses a subset.

use std::io::Read;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// What a finished process left behind.
pub struct Ran {
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

/// How long a timed-out child gets to act on SIGTERM before its group is killed.
const TERM_GRACE: Duration = Duration::from_secs(2);

/// End a timed-out child and everything it started. The child leads its own
/// process group (see [`run_bounded`]), so the group is signalled, not only the
/// child: a shell that runs `sleep 30` dies on a kill of the shell alone while
/// the `sleep` lives on, reparented to launchd.
fn stop_group(child: &mut Child) {
    let group = -(child.id() as libc::pid_t);
    // SAFETY: `kill` takes a pid and a signal; the negated pid is the group of
    // the child spawned with `process_group(0)`, whose pid is its group id.
    unsafe { libc::kill(group, libc::SIGTERM) };
    let grace = Instant::now() + TERM_GRACE;
    while Instant::now() < grace && !matches!(child.try_wait(), Ok(Some(_))) {
        std::thread::sleep(Duration::from_millis(15));
    }
    // The leader may have exited on TERM with a descendant still running, so
    // the group is killed either way; the child is then reaped.
    // SAFETY: as above.
    unsafe { libc::kill(group, libc::SIGKILL) };
    let _ = child.kill();
    let _ = child.wait();
}

/// Run `cmd` to completion within `secs`. `Err` is a process that could not be
/// started or did not finish; a non-zero exit is a normal `Ok`. The child runs
/// in its own process group, and a timeout ends the whole group.
pub fn run_bounded(cmd: &mut Command, secs: u64) -> Result<Ran, String> {
    let mut child = cmd
        .process_group(0)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    fn drain(mut pipe: impl Read + Send + 'static) -> std::thread::JoinHandle<Vec<u8>> {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = pipe.read_to_end(&mut buf);
            buf
        })
    }
    let out = child.stdout.take().map(drain);
    let err = child.stderr.take().map(drain);
    let deadline = Instant::now() + Duration::from_secs(secs);
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) if Instant::now() >= deadline => {
                // The group is gone, so its pipes close and the readers finish by
                // themselves; they are not joined.
                stop_group(&mut child);
                return Err(format!("timed out after {secs}s"));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(15)),
            Err(e) => return Err(e.to_string()),
        }
    };
    let text = |h: Option<std::thread::JoinHandle<Vec<u8>>>| {
        h.and_then(|h| h.join().ok())
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .unwrap_or_default()
    };
    let (stdout, stderr) = (text(out), text(err));
    Ok(Ran {
        code: status.code(),
        stdout,
        stderr,
    })
}

/// The first `n` characters of `s`, trimmed: what a diagnostic quotes of a
/// child's stderr.
pub fn first_chars(s: &str, n: usize) -> String {
    s.trim().chars().take(n).collect()
}

/// The last `n` characters of `s`, trimmed: what a diagnostic quotes of a
/// failed child's stderr, whose cause is usually its final lines.
pub fn last_chars(s: &str, n: usize) -> String {
    let s = s.trim();
    let skip = s.chars().count().saturating_sub(n);
    s.chars().skip(skip).collect()
}
