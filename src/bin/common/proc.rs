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
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Once;
use std::time::{Duration, Instant};

/// What a finished process left behind.
pub struct Ran {
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

/// How long a timed-out child gets to act on SIGTERM before its group is killed.
const TERM_GRACE: Duration = Duration::from_secs(2);

/// Process groups of the children now running, for the terminal-signal handler.
/// A slot holds a group id, 0 when free, or -1 while a call has claimed it and
/// not yet spawned its child. A fixed table of atomics, because
/// the handler may only call async-signal-safe functions: no lock, no
/// allocation, and `kill` is on the safe list. `run_bounded` may be called from
/// several threads at once (`kcov::inventory` preprocesses on up to 8), and each
/// call owns one slot for as long as its child runs.
static ACTIVE_GROUPS: [AtomicI32; 64] = [const { AtomicI32::new(0) }; 64];

/// Signals whose default action ends this process and which a terminal or a
/// supervisor sends to it: Ctrl-C, a `kill`, a closed terminal, Ctrl-\.
const FATAL_SIGNALS: [libc::c_int; 4] = [libc::SIGINT, libc::SIGTERM, libc::SIGHUP, libc::SIGQUIT];

/// The handler: kill every registered child group, then die of the signal as
/// the default action would have. `SA_RESETHAND` has already put the default
/// disposition back and `SA_NODEFER` leaves the signal unblocked, so the
/// `raise` ends the process with the original signal for the parent shell to see.
extern "C" fn kill_children_and_die(sig: libc::c_int) {
    for slot in &ACTIVE_GROUPS {
        let group = slot.load(Ordering::Relaxed);
        if group > 0 {
            // SAFETY: `kill` is async-signal-safe; `-group` names a group this
            // module spawned (its leader's pid).
            unsafe { libc::kill(-group, libc::SIGKILL) };
        }
    }
    // SAFETY: `raise` is async-signal-safe.
    unsafe { libc::raise(sig) };
}

/// Install [`kill_children_and_die`], once per process, for each of
/// [`FATAL_SIGNALS`] that still has its default disposition. A signal the
/// process ignores (`nohup`) or already handles is left as it is: the owner of
/// that disposition decides what happens to the children.
fn install_signal_handlers() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        for sig in FATAL_SIGNALS {
            // SAFETY: plain `sigaction` calls on zero-initialised structs; the
            // handler is an `extern "C" fn(c_int)`, as `sa_sigaction` requires
            // without `SA_SIGINFO`.
            unsafe {
                let mut old: libc::sigaction = std::mem::zeroed();
                if libc::sigaction(sig, std::ptr::null(), &mut old) != 0 || old.sa_sigaction != libc::SIG_DFL {
                    continue;
                }
                let mut act: libc::sigaction = std::mem::zeroed();
                act.sa_sigaction = kill_children_and_die as extern "C" fn(libc::c_int) as usize;
                act.sa_flags = libc::SA_RESETHAND | libc::SA_NODEFER;
                libc::sigemptyset(&mut act.sa_mask);
                libc::sigaction(sig, &act, std::ptr::null_mut());
            }
        }
    });
}

/// A claimed slot of [`ACTIVE_GROUPS`], released when dropped.
struct Registered(usize);

impl Registered {
    /// Claim a free slot, or `None` when all are taken.
    fn reserve() -> Option<Registered> {
        ACTIVE_GROUPS
            .iter()
            .position(|s| s.compare_exchange(0, -1, Ordering::AcqRel, Ordering::Relaxed).is_ok())
            .map(Registered)
    }

    /// Record the child's group in the claimed slot, from now on killed by the handler.
    fn set(&self, group: libc::pid_t) {
        ACTIVE_GROUPS[self.0].store(group, Ordering::Release);
    }

    /// Take the group back out of the handler's reach, before the leader is
    /// reaped: once it is, the group id may be handed to another process group,
    /// and a terminal signal arriving then must not kill that one.
    fn release_group(&self) {
        ACTIVE_GROUPS[self.0].store(-1, Ordering::Release);
    }
}

impl Drop for Registered {
    fn drop(&mut self) {
        ACTIVE_GROUPS[self.0].store(0, Ordering::Release);
    }
}

/// End a child and everything it started, then reap it. The child leads its own
/// process group (see [`run_bounded`]), so the group is signalled, not only the
/// child: a shell that runs `sleep 30` dies on a kill of the shell alone while
/// the `sleep` lives on, reparented to launchd. SIGTERM first, SIGKILL after
/// [`TERM_GRACE`]: a child that ignores TERM (and passes that on to its own
/// children) is still ended. The group is killed even when the leader exited on
/// TERM, since a descendant may not have.
///
/// A process that leaves the group (`setsid`, `setpgid`) is not reached: the
/// group is the only handle on a descendant. Nothing in this repository's
/// suites does that.
fn stop_group(child: &mut Child, slot: &Registered) {
    let group = -(child.id() as libc::pid_t);
    // SAFETY: `kill` takes a pid and a signal; the negated pid is the group of
    // the child spawned with `process_group(0)`, whose pid is its group id.
    unsafe { libc::kill(group, libc::SIGTERM) };
    let grace = Instant::now() + TERM_GRACE;
    // The leader is waited for without reaping it (`exited_unreaped`): once it
    // is reaped nothing holds its pid, and so the group id, and a later
    // `kill(-group)` could signal an unrelated process group that was handed
    // the same number. A zombie leader keeps the id reserved.
    while Instant::now() < grace && !matches!(exited_unreaped(child.id() as libc::pid_t), Ok(true)) {
        std::thread::sleep(Duration::from_millis(15));
    }
    // SAFETY: as above; the leader is not reaped yet, so the group id is still ours.
    unsafe { libc::kill(group, libc::SIGKILL) };
    slot.release_group();
    let _ = child.kill();
    let _ = child.wait();
}

/// Whether `pid` has exited, without reaping it: `waitid` with `WNOWAIT` leaves
/// the zombie in place, so its pid (here, the child's process group id) stays
/// reserved until the caller `wait`s. `try_wait` reaps, and a group signalled
/// after that may by then belong to someone else.
pub fn exited_unreaped(pid: libc::pid_t) -> Result<bool, String> {
    // SAFETY: a zeroed `siginfo_t` is a valid out-parameter; `waitid` only
    // writes it. `WNOHANG` makes it return at once with `si_pid` 0 while the
    // child still runs.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    let rc = unsafe {
        libc::waitid(
            libc::P_PID,
            pid as libc::id_t,
            &mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error().to_string());
    }
    // SAFETY: `si_pid` is set by a successful `waitid`.
    Ok(unsafe { info.si_pid() } != 0)
}

/// Everything a bounded run left behind, bytes and signal included, and also
/// when it did not finish: a hung child's partial output is the evidence of the
/// hang. `error` is a process that could not be started, did not finish
/// (`timed out after Ns`) or could not be waited for; a non-zero exit is not an
/// error.
pub struct RanRaw {
    pub code: Option<i32>,
    /// The signal that ended the child, when one did.
    pub signal: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub error: Option<String>,
}

/// How long the output readers get to finish after a timeout or an error, once
/// the group is dead: a descendant outside it (`setsid`) can hold the pipes open,
/// and then the output so far is returned without the rest.
const DRAIN_GRACE: Duration = Duration::from_secs(2);

/// Run `cmd` to completion within `secs`. `Err` is a process that could not be
/// started or did not finish; a non-zero exit is a normal `Ok`.
///
/// The child runs in its own process group, so that a timeout can end the whole
/// tree ([`stop_group`]) and so that a terminal interrupt, which goes to the
/// parent's group only, does not leave it running: for as long as the child
/// runs, SIGINT, SIGTERM, SIGHUP and SIGQUIT to this process kill the child's group before
/// the process dies of the signal (see [`install_signal_handlers`]). There is a
/// window between `spawn` and the registration of the group, a few instructions
/// wide, in which such a signal still leaves the child behind.
///
/// When the child exits by itself, whatever it left in its group is killed too,
/// which also closes the pipes a backgrounded descendant would hold open. A
/// descendant outside the group (one that called `setsid`) can still hold them,
/// and then the output readers are joined only when it exits: a timeout returns
/// after [`DRAIN_GRACE`] without them, a normal exit waits.
pub fn run_bounded(cmd: &mut Command, secs: u64) -> Result<Ran, String> {
    let raw = run_bounded_raw(cmd, Duration::from_secs(secs));
    if let Some(e) = raw.error {
        return Err(e);
    }
    Ok(Ran {
        code: raw.code,
        stdout: String::from_utf8_lossy(&raw.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&raw.stderr).into_owned(),
    })
}

/// [`run_bounded`] with the output as bytes, the signal that ended the child, and
/// the output of a child that timed out. The one implementation: every other
/// bounded run (`bench_paired`'s lanes, through `evidence::run_raw`) goes
/// through here, so every one of them is killed with its group on a timeout and
/// on the parent's own interrupt.
pub fn run_bounded_raw(cmd: &mut Command, timeout: Duration) -> RanRaw {
    let fail = |e: String| RanRaw {
        code: None,
        signal: None,
        stdout: Vec::new(),
        stderr: Vec::new(),
        error: Some(e),
    };
    install_signal_handlers();
    // Claimed before the spawn: with no free slot nothing is started.
    let Some(registered) = Registered::reserve() else {
        return fail("too many children running at once".to_string());
    };
    let mut child = match cmd
        .process_group(0)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return fail(e.to_string()),
    };
    let group = child.id() as libc::pid_t;
    registered.set(group);
    type Drain = std::sync::mpsc::Receiver<Vec<u8>>;
    fn drain(mut pipe: impl Read + Send + 'static) -> Drain {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = pipe.read_to_end(&mut buf);
            let _ = tx.send(buf);
        });
        rx
    }
    let out = child.stdout.take().map(drain);
    let err = child.stderr.take().map(drain);
    let deadline = Instant::now() + timeout;
    // After a normal exit the readers end by themselves (the group was swept,
    // closing every pipe a descendant held), and a descendant outside the group
    // that still holds one is waited for, as it always was. After a timeout or an
    // error the wait is capped by [`DRAIN_GRACE`].
    let text = |rx: Option<Drain>, grace: Option<Duration>| {
        rx.map_or_else(Vec::new, |rx| match grace {
            None => rx.recv().unwrap_or_default(),
            Some(g) => rx.recv_timeout(g).unwrap_or_default(),
        })
    };
    // Exit is observed with `waitid(WNOWAIT)`, the group is killed, and only
    // then is the leader reaped (see `stop_group`).
    let outcome: Result<std::process::ExitStatus, String> = loop {
        match exited_unreaped(group) {
            Ok(true) => {
                // The leader is gone but not reaped, so the group id is still
                // ours; anything it backgrounded is still in the group.
                // SAFETY: as in `stop_group`.
                unsafe { libc::kill(-group, libc::SIGKILL) };
                registered.release_group();
                break child.wait().map_err(|e| e.to_string());
            }
            Ok(false) if Instant::now() >= deadline => {
                stop_group(&mut child, &registered);
                break Err(format!("timed out after {}s", timeout.as_secs()));
            }
            Ok(false) => std::thread::sleep(Duration::from_millis(15)),
            Err(e) => {
                stop_group(&mut child, &registered);
                break Err(e);
            }
        }
    };
    match outcome {
        Ok(status) => {
            use std::os::unix::process::ExitStatusExt;
            RanRaw {
                code: status.code(),
                signal: status.signal(),
                stdout: text(out, None),
                stderr: text(err, None),
                error: None,
            }
        }
        Err(e) => RanRaw {
            code: None,
            signal: None,
            stdout: text(out, Some(DRAIN_GRACE)),
            stderr: text(err, Some(DRAIN_GRACE)),
            error: Some(e),
        },
    }
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
