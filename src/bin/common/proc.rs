//! Run a child process to completion within a bound.
//!
//! The tools that drive other programs (`kernel_coverage` runs the compiler,
//! `metal-nm` and the bench suites; `attn_tune` runs `bench_flash_attn`) must not
//! wait forever on one that hangs, and must not deadlock on one that writes more
//! than a pipe holds. Included with `#[path]` by those bins and by the
//! integration tests of each; it has no dependency of its own.

#![allow(dead_code)] // Each includer uses a subset.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// What a finished process left behind.
pub struct Ran {
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

/// Run `cmd` to completion within `secs`. `Err` is a process that could not be
/// started or did not finish; a non-zero exit is a normal `Ok`.
pub fn run_bounded(cmd: &mut Command, secs: u64) -> Result<Ran, String> {
    let mut child = cmd
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
                // Our own child, by handle. A grandchild that outlives it may
                // still hold the pipes open, so the readers are left to finish
                // by themselves rather than joined.
                let _ = child.kill();
                let _ = child.wait();
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
