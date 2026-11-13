//! `scripts/check_release_ready.sh` against fixture repos (ft-57146ec0).
//!
//! The script cannot run over the real checkout in a test: there it needs the
//! network and fails at `cargo package --locked` (ft-c89dd818). So each case
//! copies the real script into a throwaway git repo and puts a fake `rustc` and
//! a fake `cargo` first on `PATH`. The fake `cargo` replays a canned package
//! list, can fail any step, can park a run on a bounded file barrier, and can
//! hang so a signal has something to interrupt.
//!
//! Every wait is bounded and every child runs in its own process group, which
//! is killed on overrun or on drop, so a red test never leaves a script behind.
//! Crate names are unique per case and never `tessl`, so the pre-fix script's
//! fixed `/tmp/<crate>-*` files cannot clobber a peer's live ones.
#![cfg(unix)]

mod scratch_support;

use scratch_support::{remove_path, Scratch};
use std::ffi::{OsStr, OsString};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static NEXT: AtomicU64 = AtomicU64::new(0);

const FAKE_RUSTC: &str = "#!/bin/sh\n\
out=\n\
i=0\n\
while [ -n \"$FAKE_RUSTC_WAIT\" ] && [ ! -e \"$FAKE_RUSTC_WAIT\" ]; do\n\
  i=$((i + 1))\n\
  [ \"$i\" -gt 600 ] && exit 99\n\
  sleep 0.1\n\
done\n\
while [ $# -gt 0 ]; do\n\
  if [ \"$1\" = -o ]; then out=\"$2\"; fi\n\
  shift\n\
done\n\
printf '#!/bin/sh\\nexit 0\\n' > \"$out\"\n\
chmod 755 \"$out\"\n";

const FAKE_CARGO: &str = r#"#!/bin/sh
case "$*" in
"package --locked --list")
  # stdout is the list itself, so the tag goes to stderr here.
  echo "tag $FAKE_TAG" >&2
  cat "$FAKE_LIST"
  if [ "${FAKE_LIST_RC:-0}" != 0 ]; then
    echo "error: fake list failure" >&2
    exit "$FAKE_LIST_RC"
  fi
  if [ -n "$FAKE_LISTED" ]; then
    : > "$FAKE_LISTED"
    i=0
    while [ ! -e "$FAKE_RELEASE" ]; do
      i=$((i + 1))
      if [ "$i" -gt 600 ]; then
        echo "barrier timeout" >&2
        exit 98
      fi
      sleep 0.1
    done
  fi
  exit 0
  ;;
"package --locked")
  echo "tag $FAKE_TAG"
  if [ -n "$FAKE_PARK" ]; then
    : > "$FAKE_STARTED"
    i=0
    while [ ! -e "$FAKE_PARK" ]; do
      i=$((i + 1))
      if [ "$i" -gt 600 ]; then
        echo "park timeout" >&2
        exit 98
      fi
      sleep 0.1
    done
  fi
  if [ "$FAKE_HANG" = package ]; then
    : > "$FAKE_STARTED"
    sleep 60
  fi
  rc="${FAKE_PACKAGE_RC:-0}"
  if [ "$rc" != 0 ]; then
    if [ "$rc" = 101 ]; then
      echo "error: no matching package named \`ojas-core\` found"
    fi
    echo "tag $FAKE_TAG" >&2
  fi
  exit "$rc"
  ;;
"check --locked")
  echo "tag $FAKE_TAG"
  rc="${FAKE_DOCSRS_RC:-0}"
  if [ "$rc" != 0 ]; then
    echo "error: fake docs.rs failure" >&2
    echo "tag $FAKE_TAG" >&2
  fi
  exit "$rc"
  ;;
*)
  echo "fake cargo: unexpected arguments: $*" >&2
  exit 97
  ;;
esac
"#;

/// Shell text that polls (bounded to 60 s) until `$1` exists, then continues.
fn wait_for_shell(path: &Path) -> String {
    format!(
        "i=0\nwhile [ ! -e '{}' ]; do\n  i=$((i + 1))\n  [ \"$i\" -gt 600 ] && exit 99\n  sleep 0.1\ndone\n",
        path.display()
    )
}

fn unique_crate(tag: &str) -> String {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("relready-fx-{}-{nanos}-{n}-{tag}", std::process::id())
}

/// Names under `dir` that start with `prefix`.
fn leftovers(dir: &Path, prefix: &str) -> Vec<PathBuf> {
    match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with(prefix))
            .map(|e| e.path())
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Everything the pre-fix script (fixed names) or a run that fell back to
/// `/tmp` (empty or unset TMPDIR) can leave under `/tmp` for this crate. The
/// fixed paths are reserved up front and everything with the crate prefix is
/// removed on drop, pass or panic.
struct CrateTmp {
    crate_name: String,
    _fixed: Vec<Scratch>,
}

impl CrateTmp {
    fn new(crate_name: &str) -> CrateTmp {
        let fixed = ["package.list", "package.log", "docsrs.log"]
            .iter()
            .map(|n| Scratch::file_at(PathBuf::from(format!("/tmp/{crate_name}-{n}"))))
            .collect();
        CrateTmp {
            crate_name: crate_name.to_string(),
            _fixed: fixed,
        }
    }

    fn present(&self) -> Vec<PathBuf> {
        leftovers(Path::new("/tmp"), &format!("{}-", self.crate_name))
    }
}

impl Drop for CrateTmp {
    fn drop(&mut self) {
        for p in self.present() {
            remove_path(&p);
        }
    }
}

fn write_exec(path: &Path, text: &str) {
    std::fs::write(path, text).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

fn git(repo: &Path, args: &[&str]) {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t.invalid",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?}: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A committed throwaway repo holding the real script, plus the fake tools.
struct Fixture {
    root: Scratch,
    tag: String,
}

impl Fixture {
    /// `manifest_name` is what Cargo.toml says; `tag` identifies this fixture in
    /// every fake tool output; `list` is the canned `cargo package --list`.
    fn new(manifest_name: &str, tag: &str, list: &[&str]) -> Fixture {
        let root = Scratch::dir("relready-fx");
        let repo = root.join("repo");
        std::fs::create_dir_all(repo.join("scripts")).unwrap();
        std::fs::create_dir_all(repo.join("src/bin")).unwrap();
        std::fs::create_dir_all(root.join("fakebin")).unwrap();
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/check_release_ready.sh");
        std::fs::copy(&script, repo.join("scripts/check_release_ready.sh")).unwrap();
        std::fs::write(
            repo.join("Cargo.toml"),
            format!("[package]\nname = \"{manifest_name}\"\nversion = \"0.0.0\"\n"),
        )
        .unwrap();
        std::fs::write(
            repo.join("CHANGELOG.md"),
            "## [0.0.0]\n\n[0.0.0]: https://example.invalid\n",
        )
        .unwrap();
        std::fs::write(repo.join("README.md"), "fixture\n").unwrap();
        std::fs::write(repo.join("src/bin/release_check.rs"), "fn main() {}\n").unwrap();
        write_exec(&root.join("fakebin/rustc"), FAKE_RUSTC);
        write_exec(&root.join("fakebin/cargo"), FAKE_CARGO);
        std::fs::write(
            root.join("list.txt"),
            list.iter().map(|l| format!("{l}\n")).collect::<String>(),
        )
        .unwrap();
        git(&repo, &["init", "-q"]);
        git(&repo, &["add", "-A"]);
        git(&repo, &["commit", "-qm", "fx"]);
        let status = Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args(["status", "--porcelain"])
            .output()
            .unwrap();
        assert!(
            status.status.success() && status.stdout.is_empty(),
            "fixture tree is not clean: {}",
            String::from_utf8_lossy(&status.stdout)
        );
        Fixture {
            root,
            tag: tag.to_string(),
        }
    }

    fn path(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }
}

enum Tmp {
    Unset,
    Set(OsString),
}

fn ev(k: &'static str, v: impl AsRef<OsStr>) -> (&'static str, OsString) {
    (k, v.as_ref().to_os_string())
}

struct Out {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

impl Out {
    fn all(&self) -> String {
        format!("--- stdout ---\n{}--- stderr ---\n{}", self.stdout, self.stderr)
    }

    /// The directory printed by the `==> run dir:` line.
    fn run_dir(&self) -> PathBuf {
        let line = self
            .stdout
            .lines()
            .find_map(|l| l.strip_prefix("==> run dir: "))
            .unwrap_or_else(|| panic!("no `==> run dir:` line\n{}", self.all()));
        let (dir, _) = line
            .rsplit_once(" (removed on success")
            .unwrap_or_else(|| panic!("malformed run dir line {line:?}"));
        PathBuf::from(dir)
    }
}

/// A running script in its own process group. Dropping it kills the group.
struct Spawned {
    child: Option<Child>,
    pgid: i32,
    out: Option<JoinHandle<String>>,
    err: Option<JoinHandle<String>>,
}

fn drain(mut r: impl std::io::Read + Send + 'static) -> JoinHandle<String> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = r.read_to_end(&mut buf);
        String::from_utf8_lossy(&buf).into_owned()
    })
}

fn spawn(fx: &Fixture, tmp: &Tmp, env: &[(&'static str, OsString)]) -> Spawned {
    spawn_in(fx, &fx.path("."), tmp, env)
}

/// `PATH` with `dirs` (wrapper tools) first, then the fixture's fake tools.
fn path_with(fx: &Fixture, dirs: &[&Path]) -> OsString {
    let mut path = OsString::new();
    for d in dirs {
        path.push(d);
        path.push(":");
    }
    path.push(fx.path("fakebin"));
    path.push(":");
    path.push(std::env::var_os("PATH").unwrap_or_default());
    path
}

/// How the test reads the script's stdout.
#[derive(Clone, Copy, PartialEq)]
enum Stdout {
    /// Read everything.
    All,
    /// Read up to and including the `==> run dir:` line, then close the pipe.
    CloseAfterRunDir,
}

fn spawn_in(fx: &Fixture, cwd: &Path, tmp: &Tmp, env: &[(&'static str, OsString)]) -> Spawned {
    spawn_with(fx, cwd, tmp, env, Stdout::All, false)
}

fn drain_until_run_dir(mut r: impl std::io::Read + Send + 'static) -> JoinHandle<String> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let mut byte = [0u8; 1];
        while r.read(&mut byte).map(|n| n == 1).unwrap_or(false) {
            buf.push(byte[0]);
            let text = String::from_utf8_lossy(&buf);
            if text.lines().any(|l| l.starts_with("==> run dir: ")) && text.ends_with('\n') {
                break;
            }
        }
        String::from_utf8_lossy(&buf).into_owned()
    })
}

fn spawn_with(
    fx: &Fixture,
    cwd: &Path,
    tmp: &Tmp,
    env: &[(&'static str, OsString)],
    stdout: Stdout,
    ignore_sigpipe: bool,
) -> Spawned {
    let path = path_with(fx, &[]);
    let mut cmd = Command::new("bash");
    cmd.arg(fx.path("repo/scripts/check_release_ready.sh"))
        .current_dir(cwd)
        .env("PATH", path)
        .env("FAKE_TAG", &fx.tag)
        .env("FAKE_LIST", fx.path("list.txt"))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    match tmp {
        Tmp::Unset => {
            cmd.env_remove("TMPDIR");
        }
        Tmp::Set(v) => {
            cmd.env("TMPDIR", v);
        }
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    // SAFETY: the closure runs in the forked child before exec and only calls
    // the async-signal-safe `signal`. A `cargo test` started in the background
    // (`&` in a non-interactive shell, or nohup) inherits SIGINT, SIGQUIT and
    // SIGHUP as ignored, and bash cannot trap a signal that was ignored on
    // entry; the script's own traps need them all default.
    unsafe {
        cmd.pre_exec(move || {
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            libc::signal(libc::SIGTERM, libc::SIG_DFL);
            libc::signal(libc::SIGQUIT, libc::SIG_DFL);
            libc::signal(libc::SIGHUP, libc::SIG_DFL);
            libc::signal(
                libc::SIGPIPE,
                if ignore_sigpipe { libc::SIG_IGN } else { libc::SIG_DFL },
            );
            Ok(())
        });
    }
    let mut child = cmd.spawn().unwrap();
    let pgid = child.id() as i32;
    let so = child.stdout.take().unwrap();
    let out = Some(match stdout {
        Stdout::All => drain(so),
        Stdout::CloseAfterRunDir => drain_until_run_dir(so),
    });
    let err = Some(drain(child.stderr.take().unwrap()));
    Spawned {
        child: Some(child),
        pgid,
        out,
        err,
    }
}

impl Spawned {
    fn signal_group(&self, sig: i32) {
        // SAFETY: plain syscall on a process group this struct created; a
        // vanished group just yields ESRCH.
        unsafe {
            libc::kill(-self.pgid, sig);
        }
    }

    fn signal_pid(&self, sig: i32) {
        // SAFETY: plain syscall on the script's own pid, which is still unreaped.
        unsafe {
            libc::kill(self.pgid, sig);
        }
    }

    /// Wait at most `secs`; on overrun kill the whole group and panic.
    fn wait(mut self, secs: u64) -> Out {
        let mut child = self.child.take().unwrap();
        let deadline = Instant::now() + Duration::from_secs(secs);
        let status = loop {
            if let Some(s) = child.try_wait().unwrap() {
                break Some(s);
            }
            if Instant::now() >= deadline {
                self.signal_group(libc::SIGKILL);
                let _ = child.wait();
                break None;
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        // Whatever is left of the group must not hold the pipes open.
        self.signal_group(libc::SIGKILL);
        let stdout = self.out.take().unwrap().join().unwrap();
        let stderr = self.err.take().unwrap().join().unwrap();
        let out = Out {
            code: status.and_then(|s| s.code()),
            stdout,
            stderr,
        };
        match status {
            None => panic!("script still running after {secs}s\n{}", out.all()),
            Some(s) if s.code().is_none() => {
                panic!("script died of signal {:?}\n{}", s.signal(), out.all())
            }
            Some(_) => out,
        }
    }

    /// Wait at most `secs` for `path` to appear; the script dying first or the
    /// overrun is a panic.
    fn wait_for_file(&mut self, path: &Path, secs: u64, what: &str) {
        let deadline = Instant::now() + Duration::from_secs(secs);
        while !path.exists() {
            let exited = self.child.as_mut().unwrap().try_wait().unwrap();
            assert!(exited.is_none(), "script exited ({exited:?}) before {what}");
            assert!(Instant::now() < deadline, "no {what} within {secs}s");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for Spawned {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            self.signal_group(libc::SIGKILL);
            let _ = child.wait();
        }
    }
}

fn run(fx: &Fixture, tmp: &Tmp, env: &[(&'static str, OsString)]) -> Out {
    spawn(fx, tmp, env).wait(120)
}

fn entries(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

fn tmpdir_with_sentinel(prefix: &str) -> Scratch {
    let d = Scratch::dir(prefix);
    std::fs::write(d.join("sentinel"), "keep me\n").unwrap();
    d
}

fn assert_released(out: &Out, crate_name: &str) {
    assert_eq!(out.code, Some(0), "{}", out.all());
    assert!(
        out.stdout.contains(&format!("release-ready: {crate_name} 0.0.0")),
        "{}",
        out.all()
    );
}

#[test]
fn concurrent_runs_false_fail() {
    // A's own list is clean; B's holds a denied path. Both share crate name and
    // TMPDIR, and B runs to completion while A sits between listing and checking.
    let c = unique_crate("cff");
    let _tmp = CrateTmp::new(&c);
    let a = Fixture::new(&c, "tag-A-cff", &["src/lib.rs"]);
    let b = Fixture::new(&c, "tag-B-cff", &["src/lib.rs", ".devmap/tag-B-cff"]);
    let tmpdir = Scratch::dir("relready-tmp");
    let tmp = Tmp::Set(tmpdir.path().into());
    let listed = a.path("listed");
    let release = a.path("release");

    let mut sa = spawn(&a, &tmp, &[ev("FAKE_LISTED", &listed), ev("FAKE_RELEASE", &release)]);
    sa.wait_for_file(&listed, 60, "A to finish listing");

    let ob = run(&b, &tmp, &[]);
    assert_eq!(ob.code, Some(1), "B must fail on its own denied path\n{}", ob.all());
    assert!(ob.all().contains(".devmap/tag-B-cff"), "{}", ob.all());

    std::fs::write(&release, "go\n").unwrap();
    let oa = sa.wait(60);
    assert!(!oa.all().contains("tag-B-cff"), "A read B's list\n{}", oa.all());
    assert_released(&oa, &c);
}

#[test]
fn concurrent_runs_false_pass() {
    // A's list holds a denied path; B's is clean and overwrites the shared
    // file while A waits. A must still fail on its own list.
    let c = unique_crate("cfp");
    let _tmp = CrateTmp::new(&c);
    let a = Fixture::new(&c, "tag-A-cfp", &["src/lib.rs", ".devmap/tag-A-cfp"]);
    let b = Fixture::new(&c, "tag-B-cfp", &["src/lib.rs"]);
    let tmpdir = Scratch::dir("relready-tmp");
    let tmp = Tmp::Set(tmpdir.path().into());
    let listed = a.path("listed");
    let release = a.path("release");

    let mut sa = spawn(&a, &tmp, &[ev("FAKE_LISTED", &listed), ev("FAKE_RELEASE", &release)]);
    sa.wait_for_file(&listed, 60, "A to finish listing");

    let ob = run(&b, &tmp, &[]);
    assert_released(&ob, &c);

    std::fs::write(&release, "go\n").unwrap();
    let oa = sa.wait(60);
    assert_eq!(oa.code, Some(1), "the gate passed on another run's list\n{}", oa.all());
    assert!(oa.all().contains(".devmap/tag-A-cfp"), "{}", oa.all());
}

#[test]
fn success_leaves_nothing_and_no_fixed_tmp_path() {
    let c = unique_crate("ok");
    let tmp_guard = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-ok", &["src/lib.rs"]);
    let tmpdir = tmpdir_with_sentinel("relready-tmp");

    let out = run(&fx, &Tmp::Set(tmpdir.path().into()), &[]);
    assert_released(&out, &c);
    assert_eq!(
        tmp_guard.present(),
        Vec::<PathBuf>::new(),
        "fixed /tmp files written\n{}",
        out.all()
    );
    let run_dir = out.run_dir();
    assert_eq!(
        run_dir.parent().unwrap().canonicalize().unwrap(),
        tmpdir.canonicalize().unwrap(),
        "run dir is not under TMPDIR\n{}",
        out.all()
    );
    assert!(!run_dir.exists(), "run dir {} survived success", run_dir.display());
    assert_eq!(entries(&tmpdir), vec!["sentinel".to_string()]);
}

#[test]
fn hostile_tmpdirs() {
    let c = unique_crate("host");
    let tmp_guard = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-host", &["src/lib.rs"]);
    let slash_or_not = |out: &Out, under: &Path| {
        assert_released(out, &c);
        let run_dir = out.run_dir();
        assert_eq!(
            run_dir.parent().unwrap().canonicalize().unwrap(),
            under.canonicalize().unwrap(),
            "{}",
            out.all()
        );
        assert!(
            !run_dir.exists(),
            "run dir {} survived\n{}",
            run_dir.display(),
            out.all()
        );
    };

    // Spaces and non-ASCII.
    let parent = Scratch::dir("relready-hostile");
    let odd = parent.join("tmp dir ü 名");
    std::fs::create_dir(&odd).unwrap();
    std::fs::write(odd.join("sentinel"), "keep\n").unwrap();
    let out = run(&fx, &Tmp::Set(odd.clone().into()), &[]);
    slash_or_not(&out, &odd);
    assert_eq!(entries(&odd), vec!["sentinel".to_string()]);

    // A symlink to a real directory: both survive, the run dir is gone.
    let real = parent.join("real");
    std::fs::create_dir(&real).unwrap();
    std::fs::write(real.join("sentinel"), "keep\n").unwrap();
    let link = parent.join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let out = run(&fx, &Tmp::Set(link.clone().into()), &[]);
    slash_or_not(&out, &real);
    assert!(std::fs::symlink_metadata(&link).unwrap().file_type().is_symlink());
    assert_eq!(entries(&real), vec!["sentinel".to_string()]);

    // A trailing slash, as the macOS default has.
    let mut slashed = OsString::from(&odd);
    slashed.push("/");
    let out = run(&fx, &Tmp::Set(slashed), &[]);
    slash_or_not(&out, &odd);
    assert_eq!(entries(&odd), vec!["sentinel".to_string()]);

    // Empty and unset both mean /tmp, and nothing is left there.
    for tmp in [Tmp::Set(OsString::new()), Tmp::Unset] {
        let out = run(&fx, &tmp, &[]);
        slash_or_not(&out, Path::new("/tmp"));
        assert_eq!(tmp_guard.present(), Vec::<PathBuf>::new(), "{}", out.all());
    }
}

#[test]
fn nonexistent_tmpdir_fails_closed() {
    let c = unique_crate("nx");
    let tmp_guard = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-nx", &["src/lib.rs"]);
    let parent = Scratch::dir("relready-nx");
    let missing = parent.join("does-not-exist");

    let out = run(&fx, &Tmp::Set(missing.clone().into()), &[]);
    assert_eq!(out.code, Some(1), "{}", out.all());
    assert!(
        out.stderr.contains("error:") && out.stderr.contains(&missing.display().to_string()),
        "the error must name TMPDIR\n{}",
        out.all()
    );
    assert!(!out.stdout.contains("release-ready:"), "{}", out.all());
    assert_eq!(
        tmp_guard.present(),
        Vec::<PathBuf>::new(),
        "fell back to /tmp\n{}",
        out.all()
    );
    assert!(!missing.exists());
}

#[test]
fn each_failing_step_keeps_only_its_log_and_names_it() {
    struct Case {
        name: &'static str,
        env: Vec<(&'static str, OsString)>,
        list: Vec<&'static str>,
        log: &'static str,
        msg: &'static str,
        tail: Option<&'static str>,
    }
    let cases = vec![
        Case {
            name: "list",
            env: vec![ev("FAKE_LIST_RC", "101")],
            list: vec!["src/lib.rs"],
            log: "package-list.log",
            msg: "cargo package --locked --list failed",
            tail: Some("error: fake list failure"),
        },
        Case {
            name: "denied",
            env: vec![],
            list: vec!["src/lib.rs", ".devmap/tag-denied"],
            log: "package.list",
            msg: "package list contains excluded/stale paths",
            tail: None,
        },
        Case {
            name: "package",
            env: vec![ev("FAKE_PACKAGE_RC", "101")],
            list: vec!["src/lib.rs"],
            log: "package.log",
            msg: "cargo package --locked failed",
            tail: Some("no matching package named `ojas-core` found"),
        },
        Case {
            name: "docsrs",
            env: vec![ev("FAKE_DOCSRS_RC", "3")],
            list: vec!["src/lib.rs"],
            log: "docsrs.log",
            msg: "DOCS_RS=1 cargo check failed",
            tail: Some("error: fake docs.rs failure"),
        },
    ];
    for case in cases {
        let c = unique_crate(case.name);
        let tmp_guard = CrateTmp::new(&c);
        let tag = format!("tag-{}", case.name);
        let fx = Fixture::new(&c, &tag, &case.list);
        let tmpdir = tmpdir_with_sentinel("relready-tmp");
        let out = run(&fx, &Tmp::Set(tmpdir.path().into()), &case.env);

        if case.name == "docsrs" && !cfg!(target_os = "macos") {
            // The step is Darwin-only: elsewhere the same failure is not reached.
            assert_released(&out, &c);
            continue;
        }
        assert_ne!(out.code, Some(0), "[{}] {}", case.name, out.all());
        assert!(!out.stdout.contains("release-ready: "), "[{}] {}", case.name, out.all());
        assert_eq!(
            tmp_guard.present(),
            Vec::<PathBuf>::new(),
            "[{}] {}",
            case.name,
            out.all()
        );
        let run_dir = out.run_dir();
        let kept = run_dir.join(case.log);
        assert!(
            out.stderr.contains(&format!("error: {}", case.msg))
                && out.stderr.contains(&format!("; see {}", kept.display())),
            "[{}] no `error: {}; see {}`\n{}",
            case.name,
            case.msg,
            kept.display(),
            out.all()
        );
        assert!(
            out.stderr
                .contains(&format!("release-ready FAILED: kept {}", kept.display())),
            "[{}] no `release-ready FAILED: kept` line\n{}",
            case.name,
            out.all()
        );
        if let Some(tail) = case.tail {
            assert!(
                out.stderr.contains("--- last 40 lines of") && out.stderr.contains(tail),
                "[{}] tail missing\n{}",
                case.name,
                out.all()
            );
        }
        let text = std::fs::read_to_string(&kept)
            .unwrap_or_else(|e| panic!("[{}] kept log {}: {e}\n{}", case.name, kept.display(), out.all()));
        assert!(
            text.contains(&tag),
            "[{}] kept log lacks this run's tag: {text:?}",
            case.name
        );
        assert_eq!(
            entries(&run_dir),
            vec![case.log.to_string()],
            "[{}] extra files kept",
            case.name
        );
        remove_path(&run_dir);
    }
}

#[test]
fn signals_remove_the_run_dir() {
    for (name, sig, code) in [
        ("INT", libc::SIGINT, 130),
        ("TERM", libc::SIGTERM, 143),
        ("HUP", libc::SIGHUP, 129),
    ] {
        let c = unique_crate(name);
        let tmp_guard = CrateTmp::new(&c);
        let fx = Fixture::new(&c, &format!("tag-{name}"), &["src/lib.rs"]);
        let tmpdir = tmpdir_with_sentinel("relready-tmp");
        let started = fx.path("started");
        let mut s = spawn(
            &fx,
            &Tmp::Set(tmpdir.path().into()),
            &[ev("FAKE_HANG", "package"), ev("FAKE_STARTED", &started)],
        );
        s.wait_for_file(&started, 60, "the hung `cargo package` to start");
        s.signal_group(sig);
        let out = s.wait(10);
        assert_eq!(out.code, Some(code), "[{name}] {}", out.all());
        let run_dir = out.run_dir();
        assert!(
            !run_dir.exists(),
            "[{name}] run dir {} survived\n{}",
            run_dir.display(),
            out.all()
        );
        assert_eq!(tmp_guard.present(), Vec::<PathBuf>::new(), "[{name}] {}", out.all());
        assert_eq!(entries(&tmpdir), vec!["sentinel".to_string()], "[{name}]");
    }
}

#[test]
fn crate_name_is_validated() {
    // A path separator in the name would move the temp dir out of TMPDIR.
    let unique = unique_crate("bad");
    let bad = format!("../{unique}");
    let tmpdir = tmpdir_with_sentinel("relready-tmp");
    let escaped = tmpdir.parent().unwrap().to_path_buf();
    let escape_prefix = format!("{unique}-release-ready.");
    struct Cleanup(PathBuf, String);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            for p in leftovers(&self.0, &self.1) {
                remove_path(&p);
            }
        }
    }
    let _escape_guard = Cleanup(escaped.clone(), escape_prefix.clone());
    let fx = Fixture::new(&bad, "tag-bad", &["src/lib.rs"]);

    let out = run(&fx, &Tmp::Set(tmpdir.path().into()), &[]);
    assert_eq!(out.code, Some(1), "{}", out.all());
    assert!(
        out.stderr.contains("error:") && out.stderr.contains(&bad),
        "the error must name the crate\n{}",
        out.all()
    );
    assert_eq!(
        leftovers(&escaped, &escape_prefix),
        Vec::<PathBuf>::new(),
        "{}",
        out.all()
    );
    assert_eq!(entries(&tmpdir), vec!["sentinel".to_string()]);
}

/// A directory holding one wrapper tool; `body` is the shell text after the shebang.
fn wrapper(fx: &Fixture, tool: &str, body: &str) -> PathBuf {
    let dir = fx.path(&format!("wrap-{tool}"));
    std::fs::create_dir_all(&dir).unwrap();
    write_exec(&dir.join(tool), &format!("#!/bin/sh\n{body}"));
    dir
}

fn assert_only_sentinel(tmpdir: &Path, what: &str, out: &Out) {
    assert_eq!(entries(tmpdir), vec!["sentinel".to_string()], "{what}\n{}", out.all());
}

#[test]
fn empty_package_list_fails() {
    let c = unique_crate("empty");
    let _tmp = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-empty", &[]);
    let tmpdir = tmpdir_with_sentinel("relready-tmp");
    let out = run(&fx, &Tmp::Set(tmpdir.path().into()), &[]);
    assert_eq!(out.code, Some(1), "an empty package list passed\n{}", out.all());
    let kept = out.run_dir().join("package.list");
    assert!(
        out.stderr.contains("error:") && out.stderr.contains(&format!("; see {}", kept.display())),
        "{}",
        out.all()
    );
    assert_eq!(entries(&out.run_dir()), vec!["package.list".to_string()]);
    remove_path(&out.run_dir());
}

#[test]
fn run_dir_is_verified_before_anything_is_removed() {
    // A mktemp that prints nothing, a file, a directory outside TMPDIR, a
    // foreign directory inside TMPDIR, or a symlink (inside TMPDIR, or outside
    // it, pointing at a TMPDIR child) must be refused, and the recording `rm`
    // must never be called.
    for variant in ["empty", "file", "elsewhere", "foreign", "symlink_in", "symlink_out"] {
        let c = unique_crate(variant);
        let _tmp = CrateTmp::new(&c);
        let fx = Fixture::new(&c, &format!("tag-{variant}"), &["src/lib.rs"]);
        let tmpdir = tmpdir_with_sentinel("relready-tmp");
        let foreign = tmpdir.join("foreign-dir");
        std::fs::create_dir(&foreign).unwrap();
        std::fs::write(foreign.join("precious"), "x\n").unwrap();
        let outside = Scratch::dir("relready-outside");
        let file = outside.join("a-file");
        std::fs::write(&file, "x\n").unwrap();
        let link_in = tmpdir.join(format!("{c}-release-ready.AAAAAA"));
        let link_out = outside.join(format!("{c}-release-ready.BBBBBB"));
        std::os::unix::fs::symlink(&foreign, &link_in).unwrap();
        std::os::unix::fs::symlink(&foreign, &link_out).unwrap();
        let before = entries(&tmpdir);
        let rmlog = fx.path("rm.log");
        let printed = match variant {
            "empty" => String::new(),
            "file" => file.display().to_string(),
            "elsewhere" => outside.display().to_string(),
            "foreign" => foreign.display().to_string(),
            "symlink_in" => link_in.display().to_string(),
            _ => link_out.display().to_string(),
        };
        let mk_body = if printed.is_empty() {
            "exit 0\n".to_string()
        } else {
            format!("echo '{printed}'\n")
        };
        let mk = wrapper(&fx, "mktemp", &mk_body);
        let rm = wrapper(&fx, "rm", &format!("echo \"rm $*\" >> '{}'\n", rmlog.display()));
        let out = run(
            &fx,
            &Tmp::Set(tmpdir.path().into()),
            &[ev("PATH", path_with(&fx, &[&rm, &mk]))],
        );
        assert_eq!(out.code, Some(1), "[{variant}] {}", out.all());
        assert!(out.stderr.contains("error:"), "[{variant}] {}", out.all());
        assert!(!out.stdout.contains("release-ready:"), "[{variant}] {}", out.all());
        let calls = std::fs::read_to_string(&rmlog).unwrap_or_default();
        assert_eq!(calls, "", "[{variant}] rm was called: {calls}\n{}", out.all());
        assert!(
            file.exists() && outside.exists() && foreign.join("precious").exists(),
            "[{variant}] a path outside the run was touched"
        );
        assert_eq!(entries(&tmpdir), before, "[{variant}] TMPDIR changed\n{}", out.all());
    }
}

#[test]
fn run_dir_check_handles_trailing_slash_symlink_and_dotdot_tmpdirs() {
    // The real mktemp, under TMPDIRs that are spelled in awkward ways: each run
    // must succeed and clean up (a wrongly refused run dir would leak).
    let c = unique_crate("shapes");
    let _tmp = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-shapes", &["src/lib.rs"]);
    let parent = Scratch::dir("relready-shapes");
    let real = parent.join("real");
    std::fs::create_dir_all(real.join("sub")).unwrap();
    std::fs::write(real.join("sentinel"), "x\n").unwrap();
    let link = parent.join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let via_sub = parent.join("subs");
    std::os::unix::fs::symlink(real.join("sub"), &via_sub).unwrap();
    let mut slash = OsString::from(&real);
    slash.push("/");
    let mut dotdot = OsString::from(&via_sub);
    dotdot.push("/..");
    for (name, tmp) in [
        ("trailing slash", slash),
        ("symlinked", OsString::from(&link)),
        ("symlink/..", dotdot),
    ] {
        let out = run(&fx, &Tmp::Set(tmp), &[]);
        assert_released(&out, &c);
        assert!(
            !out.stderr.contains("warning:"),
            "[{name}] unexpected warning\n{}",
            out.all()
        );
        assert_eq!(
            entries(&real),
            vec!["sentinel".to_string(), "sub".to_string()],
            "[{name}] leaked\n{}",
            out.all()
        );
    }
}

#[test]
fn closed_stdout_still_cleans_up_without_a_false_warning() {
    // The reader closes the pipe right after the run-dir line (`| head -1`).
    // Whether SIGPIPE is default (the script traps it) or ignored (the echo
    // fails with EPIPE and errexit ends the run), the run dir must go, with no
    // `warning: not removing`.
    for ignore in [false, true] {
        let c = unique_crate(if ignore { "pipe-ign" } else { "pipe-dfl" });
        let tmp_guard = CrateTmp::new(&c);
        let fx = Fixture::new(&c, "tag-pipe", &["src/lib.rs"]);
        let tmpdir = tmpdir_with_sentinel("relready-tmp");
        let release = fx.path("release");
        // The fake rustc parks the script until the reader has closed the pipe,
        // so the next echo certainly hits a closed stdout.
        let s = spawn_with(
            &fx,
            &fx.path("."),
            &Tmp::Set(tmpdir.path().into()),
            &[ev("FAKE_RUSTC_WAIT", &release)],
            Stdout::CloseAfterRunDir,
            ignore,
        );
        let deadline = Instant::now() + Duration::from_secs(30);
        while !s.out.as_ref().unwrap().is_finished() {
            assert!(Instant::now() < deadline, "stdout reader never closed");
            std::thread::sleep(Duration::from_millis(20));
        }
        std::fs::write(&release, "go\n").unwrap();
        let out = s.wait(60);
        let what = format!("[sigpipe ignored: {ignore}]");
        // SIGPIPE default: the PIPE trap exits 141. Ignored: the failing echo
        // ends the run through errexit with 1.
        let want = if ignore { 1 } else { 141 };
        assert_eq!(out.code, Some(want), "{what} unexpected rc\n{}", out.all());
        assert!(
            !out.stderr.contains("warning: not removing"),
            "{what} false warning\n{}",
            out.all()
        );
        assert_only_sentinel(&tmpdir, &format!("{what} run dir leaked"), &out);
        assert_eq!(tmp_guard.present(), Vec::<PathBuf>::new());
    }
}

#[test]
fn trap_is_installed_before_mktemp() {
    // mktemp has made the dir but is slow to print it; TERM goes to the script
    // pid alone. With the trap already in, bash defers it until mktemp is done
    // and cleanup removes the dir; without it TERM kills bash at once.
    let c = unique_crate("early");
    let _tmp = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-early", &["src/lib.rs"]);
    let tmpdir = tmpdir_with_sentinel("relready-tmp");
    let made = fx.path("made");
    let release = fx.path("release");
    let slow = wrapper(
        &fx,
        "mktemp",
        &format!(
            "d=$(PATH=/usr/bin:/bin mktemp \"$@\") || exit 1\n: > '{}'\n{}echo \"$d\"\n",
            made.display(),
            wait_for_shell(&release)
        ),
    );
    let mut s = spawn(
        &fx,
        &Tmp::Set(tmpdir.path().into()),
        &[ev("PATH", path_with(&fx, &[&slow]))],
    );
    s.wait_for_file(&made, 60, "mktemp to make its dir");
    s.signal_pid(libc::SIGTERM);
    std::fs::write(&release, "go\n").unwrap();
    let out = s.wait(20);
    assert_eq!(out.code, Some(143), "{}", out.all());
    assert_only_sentinel(&tmpdir, "dir leaked between mkdir and trap", &out);
}

#[test]
fn cleanup_failure_on_success_keeps_rc_and_warns() {
    let c = unique_crate("rmfail");
    let _tmp = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-rmfail", &["src/lib.rs"]);
    let tmpdir = tmpdir_with_sentinel("relready-tmp");
    let rm = wrapper(&fx, "rm", "exit 1\n");
    let out = run(
        &fx,
        &Tmp::Set(tmpdir.path().into()),
        &[ev("PATH", path_with(&fx, &[&rm]))],
    );
    assert_released(&out, &c);
    assert!(
        out.stderr.contains("warning: could not remove run dir"),
        "a failed cleanup was silent\n{}",
        out.all()
    );
    remove_path(&out.run_dir());
}

#[test]
fn second_signal_during_cleanup_is_ignored() {
    let c = unique_crate("twice");
    let tmp_guard = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-twice", &["src/lib.rs"]);
    let tmpdir = tmpdir_with_sentinel("relready-tmp");
    let started = fx.path("started");
    let in_rm = fx.path("in-rm");
    let release = fx.path("release");
    let rm = wrapper(
        &fx,
        "rm",
        &format!(
            ": > '{}'\n{}exec /bin/rm \"$@\"\n",
            in_rm.display(),
            wait_for_shell(&release)
        ),
    );
    let mut s = spawn(
        &fx,
        &Tmp::Set(tmpdir.path().into()),
        &[
            ev("PATH", path_with(&fx, &[&rm])),
            ev("FAKE_HANG", "package"),
            ev("FAKE_STARTED", &started),
        ],
    );
    s.wait_for_file(&started, 60, "the hung `cargo package` to start");
    s.signal_group(libc::SIGINT);
    s.wait_for_file(&in_rm, 30, "cleanup to reach rm");
    // The wrapper rm is parked until `release`, so this INT lands while the
    // cleanup is certainly still running.
    s.signal_group(libc::SIGINT);
    std::thread::sleep(Duration::from_millis(300));
    std::fs::write(&release, "go\n").unwrap();
    let out = s.wait(20);
    assert_eq!(out.code, Some(130), "{}", out.all());
    assert_only_sentinel(&tmpdir, "a second INT aborted the cleanup", &out);
    assert_eq!(tmp_guard.present(), Vec::<PathBuf>::new());
}

#[test]
fn signal_inside_fail_log_leaves_nothing() {
    // The fail path has already set `keep`; a signal while it prints the tail
    // must still clear it, so the aborted run keeps nothing.
    for (name, sig, code) in [("INT", libc::SIGINT, 130), ("TERM", libc::SIGTERM, 143)] {
        let c = unique_crate(name);
        let _tmp = CrateTmp::new(&c);
        let fx = Fixture::new(&c, &format!("tag-{name}"), &["src/lib.rs"]);
        let tmpdir = tmpdir_with_sentinel("relready-tmp");
        let in_tail = fx.path("in-tail");
        let tail = wrapper(
            &fx,
            "tail",
            &format!(
                ": > '{}'\n{}exec /usr/bin/tail \"$@\"\n",
                in_tail.display(),
                wait_for_shell(&fx.path("never"))
            ),
        );
        let mut s = spawn(
            &fx,
            &Tmp::Set(tmpdir.path().into()),
            &[ev("PATH", path_with(&fx, &[&tail])), ev("FAKE_PACKAGE_RC", "3")],
        );
        s.wait_for_file(&in_tail, 60, "fail_log to reach tail");
        s.signal_group(sig);
        let out = s.wait(20);
        assert_eq!(out.code, Some(code), "[{name}] {}", out.all());
        assert_only_sentinel(&tmpdir, name, &out);
    }
}

#[test]
fn relative_tmpdir_means_the_callers_cwd() {
    let c = unique_crate("rel");
    let _tmp = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-rel", &["src/lib.rs"]);
    let cwd = Scratch::dir("relready-cwd");
    let rel = cwd.join("reltmp");
    std::fs::create_dir(&rel).unwrap();
    std::fs::write(rel.join("sentinel"), "keep\n").unwrap();
    let out = spawn_in(&fx, &cwd, &Tmp::Set("reltmp".into()), &[]).wait(120);
    assert_released(&out, &c);
    assert_eq!(
        out.run_dir().parent().unwrap().canonicalize().unwrap(),
        rel.canonicalize().unwrap(),
        "{}",
        out.all()
    );
    assert_only_sentinel(&rel, "relative TMPDIR", &out);
}

#[test]
fn leading_dash_tmpdir_gets_our_own_error() {
    let c = unique_crate("dash");
    let _tmp = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-dash", &["src/lib.rs"]);
    let cwd = Scratch::dir("relready-cwd");
    std::fs::create_dir(cwd.join("-d")).unwrap();
    let out = spawn_in(&fx, &cwd, &Tmp::Set("-d".into()), &[]).wait(120);
    assert_eq!(out.code, Some(1), "{}", out.all());
    assert!(
        out.stderr.contains("error: TMPDIR '-d' starts with '-'") && !out.stderr.contains("usage:"),
        "{}",
        out.all()
    );
    assert_eq!(entries(&cwd.join("-d")), Vec::<String>::new());
}

/// Crash reports macOS writes when a process dies of a core-dumping signal
/// (SIGQUIT included). `None` when the directory cannot be read.
fn crash_reports() -> Option<Vec<String>> {
    let home = std::env::var_os("HOME")?;
    let dir = Path::new(&home).join("Library/Logs/DiagnosticReports");
    let rd = std::fs::read_dir(&dir).ok()?;
    let mut v: Vec<String> = rd
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".ips") && ["sleep-", "sh-", "bash-"].iter().any(|p| n.starts_with(p)))
        .collect();
    v.sort();
    Some(v)
}

#[test]
fn quit_is_trapped_and_nothing_dies_of_sigquit() {
    // SIGQUIT to the script pid alone, while `cargo package` is parked on a
    // release file: bash defers the QUIT trap until the child exits normally,
    // so no process dies of SIGQUIT (which on macOS goes through ReportCrash,
    // slow to reap and noisy in DiagnosticReports). The trap exits 131 and
    // cleanup removes the run dir.
    let before = crash_reports();
    if before.is_none() {
        eprintln!("skipping the crash-report assertion: ~/Library/Logs/DiagnosticReports is not readable");
    }
    let c = unique_crate("quit");
    let tmp_guard = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-quit", &["src/lib.rs"]);
    let tmpdir = tmpdir_with_sentinel("relready-tmp");
    let started = fx.path("started");
    let park = fx.path("park");
    let mut s = spawn(
        &fx,
        &Tmp::Set(tmpdir.path().into()),
        &[ev("FAKE_PARK", &park), ev("FAKE_STARTED", &started)],
    );
    s.wait_for_file(&started, 60, "the parked `cargo package` to start");
    s.signal_pid(libc::SIGQUIT);
    std::fs::write(&park, "go\n").unwrap();
    let out = s.wait(20);
    assert_eq!(out.code, Some(131), "{}", out.all());
    let run_dir = out.run_dir();
    assert!(
        !run_dir.exists(),
        "run dir {} survived\n{}",
        run_dir.display(),
        out.all()
    );
    assert_only_sentinel(&tmpdir, "QUIT", &out);
    assert_eq!(tmp_guard.present(), Vec::<PathBuf>::new());
    if let (Some(before), Some(after)) = (before, crash_reports()) {
        let new: Vec<&String> = after.iter().filter(|n| !before.contains(n)).collect();
        assert!(new.is_empty(), "a process died of SIGQUIT: new crash reports {new:?}");
    }
}

// ---------------------------------------------------------------------------
// ft-bc7a0de6: a check that could not run must not read as a check that passed.
// The dirty-tree check, the denylist grep and the other tool calls of
// check_release_ready.sh, then the same class in dev_fast_check.sh and
// guard_break_it.sh.
// ---------------------------------------------------------------------------

/// Stand-in `grep`: exits 2 (a read error) when any argument contains
/// `$FAKE_GREP_FAIL_ON`, otherwise runs the system grep.
const FAKE_GREP: &str = "#!/bin/sh\n\
for a in \"$@\"; do\n\
  case \"$a\" in\n\
    *\"$FAKE_GREP_FAIL_ON\"*) echo \"grep: $a: fake read error\" >&2; exit 2 ;;\n\
  esac\n\
done\n\
for g in /usr/bin/grep /bin/grep; do\n\
  [ -x \"$g\" ] && exec \"$g\" \"$@\"\n\
done\n\
exit 99\n";

/// A tool that always fails with `rc`, saying so on stderr.
fn failing_tool(rc: i32) -> String {
    format!("#!/bin/sh\necho \"fake: tool failed\" >&2\nexit {rc}\n")
}

/// A directory holding one stand-in tool, to put first on `PATH`.
fn tool_dir(name: &str, text: &str) -> Scratch {
    let d = Scratch::dir("relready-tool");
    write_exec(&d.join(name), text);
    d
}

/// Run `fx` with `dir` first on `PATH` and `extra` in the environment.
fn run_with_tool(fx: &Fixture, dir: &Path, extra: &[(&'static str, OsString)]) -> Out {
    let tmpdir = Scratch::dir("relready-tmp");
    let mut env = vec![ev("PATH", path_with(fx, &[dir]))];
    env.extend(extra.iter().map(|(k, v)| (*k, v.clone())));
    run(fx, &Tmp::Set(tmpdir.path().into()), &env)
}

#[test]
fn clean_checkout_is_released() {
    let c = unique_crate("clean");
    let _tmp = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-clean", &["src/lib.rs"]);
    let out = run_with_tool(&fx, Path::new("/nonexistent-tool-dir"), &[]);
    assert_released(&out, &c);
}

#[test]
fn dirty_tracked_file_fails_before_any_build() {
    let c = unique_crate("dirty");
    let _tmp = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-dirty", &["src/lib.rs"]);
    std::fs::write(fx.path("repo/README.md"), "edited after the commit\n").unwrap();
    let out = run_with_tool(&fx, Path::new("/nonexistent-tool-dir"), &[]);
    assert_eq!(out.code, Some(1), "{}", out.all());
    assert!(out.stderr.contains("working tree is dirty"), "{}", out.all());
    assert!(out.stderr.contains(" M README.md"), "{}", out.all());
    assert!(!out.all().contains("tag-dirty"), "a fake cargo ran\n{}", out.all());
    assert!(!out.stdout.contains("release-ready:"), "{}", out.all());
}

#[test]
fn untracked_file_counts_as_dirty() {
    let c = unique_crate("untracked");
    let _tmp = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-untracked", &["src/lib.rs"]);
    std::fs::write(fx.path("repo/stray.txt"), "not committed\n").unwrap();
    let out = run_with_tool(&fx, Path::new("/nonexistent-tool-dir"), &[]);
    assert_eq!(out.code, Some(1), "{}", out.all());
    assert!(out.stderr.contains("working tree is dirty"), "{}", out.all());
    assert!(out.stderr.contains("?? stray.txt"), "{}", out.all());
}

#[test]
fn failing_git_status_is_not_a_clean_tree() {
    // A git that exits 128 prints nothing on stdout; that must not pass as clean.
    let c = unique_crate("gitfail");
    let _tmp = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-gitfail", &["src/lib.rs"]);
    let dir = tool_dir("git", &failing_tool(128));
    let out = run_with_tool(&fx, &dir, &[]);
    assert_eq!(out.code, Some(1), "{}", out.all());
    assert!(
        out.stderr.contains("'git status --porcelain' failed (rc 128)"),
        "{}",
        out.all()
    );
    assert!(!out.stdout.contains("release-ready:"), "{}", out.all());
    assert!(!out.all().contains("tag-gitfail"), "a fake cargo ran\n{}", out.all());
}

#[test]
fn not_a_repository_is_not_a_clean_tree() {
    // The fixture repo with its .git removed: the real git fails (rc 128).
    // The parent of the scratch dir must not itself be a work tree, or git
    // would find that one; the guard below skips the case rather than lie.
    let c = unique_crate("norepo");
    let _tmp = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-norepo", &["src/lib.rs"]);
    std::fs::remove_dir_all(fx.path("repo/.git")).unwrap();
    let probe = Command::new("git")
        .arg("-C")
        .arg(fx.path("repo"))
        .args(["rev-parse", "--is-inside-work-tree"])
        .output()
        .unwrap();
    if probe.status.success() {
        eprintln!("skipping: the scratch dir sits inside another git work tree");
        return;
    }
    let out = run_with_tool(&fx, Path::new("/nonexistent-tool-dir"), &[]);
    assert_eq!(out.code, Some(1), "{}", out.all());
    assert!(out.stderr.contains("'git status --porcelain' failed"), "{}", out.all());
    assert!(!out.stdout.contains("release-ready:"), "{}", out.all());
}

#[test]
fn denylist_grep_read_error_fails_closed() {
    let c = unique_crate("grepdeny");
    let _tmp = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-grepdeny", &["src/lib.rs"]);
    let dir = tool_dir("grep", FAKE_GREP);
    let out = run_with_tool(&fx, &dir, &[ev("FAKE_GREP_FAIL_ON", "package.list")]);
    assert_eq!(out.code, Some(1), "{}", out.all());
    assert!(
        out.stderr.contains("grep failed (rc 2) reading ") && out.stderr.contains("package.list"),
        "{}",
        out.all()
    );
    assert!(out.stderr.contains("the denylist was not checked"), "{}", out.all());
    assert!(!out.stdout.contains("release-ready:"), "{}", out.all());
    assert!(
        !out.stdout.contains("cargo package --locked (verify build)"),
        "the run went on past the unchecked denylist\n{}",
        out.all()
    );
}

#[test]
fn denylist_match_and_clean_still_decide() {
    // rc 0 (a denied path) fails, rc 1 (clean) passes: the rc-2 branch did not
    // swallow either.
    let c = unique_crate("denyrc");
    let _tmp = CrateTmp::new(&c);
    let denied = Fixture::new(&c, "tag-denied", &["src/lib.rs", "assets/concepts/a.png"]);
    let out = run_with_tool(&denied, Path::new("/nonexistent-tool-dir"), &[]);
    assert_eq!(out.code, Some(1), "{}", out.all());
    assert!(out.stderr.contains("excluded/stale paths"), "{}", out.all());
    assert!(out.all().contains("assets/concepts/a.png"), "{}", out.all());
    let clean = Fixture::new(&c, "tag-denyclean", &["src/lib.rs"]);
    let out = run_with_tool(&clean, Path::new("/nonexistent-tool-dir"), &[]);
    assert_released(&out, &c);
}

#[test]
fn changelog_footer_grep_read_error_is_not_a_missing_footer() {
    // An unreadable CHANGELOG.md is a grep error and says so; it must not be
    // reported as, or pass as, a footer verdict.
    let c = unique_crate("grepfoot");
    let _tmp = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-grepfoot", &["src/lib.rs"]);
    let dir = tool_dir("grep", FAKE_GREP);
    let out = run_with_tool(&fx, &dir, &[ev("FAKE_GREP_FAIL_ON", "CHANGELOG.md")]);
    assert_eq!(out.code, Some(1), "{}", out.all());
    assert!(
        out.stderr.contains("grep failed (rc 2) reading CHANGELOG.md"),
        "{}",
        out.all()
    );
    assert!(!out.stderr.contains("missing footer link"), "{}", out.all());
    assert!(!out.stdout.contains("release-ready:"), "{}", out.all());
}

#[test]
fn failing_uname_does_not_skip_the_docsrs_check() {
    // A uname that fails used to read as "not Darwin" and skip DOCS_RS=1.
    let c = unique_crate("uname");
    let _tmp = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-uname", &["src/lib.rs"]);
    let dir = tool_dir("uname", &failing_tool(1));
    let out = run_with_tool(&fx, &dir, &[]);
    assert_eq!(out.code, Some(1), "{}", out.all());
    assert!(out.stderr.contains("'uname -s' failed (rc 1)"), "{}", out.all());
    assert!(!out.stdout.contains("release-ready:"), "{}", out.all());
}

// --- dev_fast_check.sh and guard_break_it.sh --------------------------------

/// A throwaway committed repo holding one real script plus the files it needs,
/// with a fake `cargo` that logs what it sees.
struct ScriptRepo {
    root: Scratch,
}

/// Stand-in `cargo`: `check` passes; `test` prints a passing result and writes
/// the listing of `$TMPDIR` to `$FAKE_LS` so a test can see what the script put
/// there while it ran.
const FAKE_DEV_CARGO: &str = r#"#!/bin/sh
case "$1" in
check) exit 0 ;;
test)
  if [ -n "$FAKE_LS" ]; then ls "$TMPDIR" > "$FAKE_LS"; fi
  echo "running 3 tests"
  echo "test result: ok. 3 passed; 0 failed"
  exit 0
  ;;
*) echo "fake cargo: unexpected arguments: $*" >&2; exit 97 ;;
esac
"#;

/// Stand-in `cargo` for guard_break_it: every run is "no test result", so each
/// baseline is red and the driver stops without editing any test. Records the
/// listing of `$TMPDIR` on every call.
const FAKE_GUARD_CARGO: &str = r#"#!/bin/sh
if [ -n "$FAKE_LS" ]; then ls "$TMPDIR" >> "$FAKE_LS"; fi
echo "fake cargo: no test result"
exit 0
"#;

impl ScriptRepo {
    /// `real`: the listed files are copies of the checkout's own (for the guard
    /// driver, whose perturbation table needs the real test text) instead of
    /// one-line stand-ins.
    fn new(script: &str, files: &[&str], real: bool, cargo: &str) -> ScriptRepo {
        let root = Scratch::dir("relready-script");
        let repo = root.join("repo");
        std::fs::create_dir_all(repo.join("scripts")).unwrap();
        std::fs::create_dir_all(repo.join("tests")).unwrap();
        std::fs::create_dir_all(root.join("fakebin")).unwrap();
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts").join(script);
        std::fs::copy(&src, repo.join("scripts").join(script)).unwrap();
        for f in files {
            if real {
                std::fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join(f), repo.join(f)).unwrap();
            } else {
                std::fs::write(repo.join(f), "// fixture\n").unwrap();
            }
        }
        write_exec(&root.join("fakebin/cargo"), cargo);
        git(&repo, &["init", "-q"]);
        git(&repo, &["add", "-A"]);
        git(&repo, &["commit", "-qm", "fx"]);
        ScriptRepo { root }
    }

    /// Start `scripts/<script> <args>` in `cwd` with `dirs` ahead of the fake
    /// cargo on PATH. `FAKE_COUNT_DIR` is this repo's root, where the counting
    /// stand-in tools keep their call counters.
    fn spawn(
        &self,
        script: &str,
        args: &[&str],
        cwd: &Path,
        dirs: &[&Path],
        tmpdir: Option<&Path>,
        env: &[(&'static str, OsString)],
    ) -> Spawned {
        let mut path = OsString::new();
        for d in dirs {
            path.push(d);
            path.push(":");
        }
        path.push(self.root.join("fakebin"));
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        let mut cmd = Command::new("bash");
        cmd.arg(self.root.join("repo/scripts").join(script))
            .args(args)
            .current_dir(cwd)
            .env("PATH", path)
            .env("FAKE_COUNT_DIR", self.root.path())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0);
        match tmpdir {
            Some(t) => cmd.env("TMPDIR", t),
            None => cmd.env_remove("TMPDIR"),
        };
        for (k, v) in env {
            cmd.env(k, v);
        }
        // As in spawn_with: a test started in the background inherits ignored
        // signals, and the scripts' traps need them all default.
        // SAFETY: the closure only calls the async-signal-safe `signal`.
        unsafe {
            cmd.pre_exec(|| {
                libc::signal(libc::SIGINT, libc::SIG_DFL);
                libc::signal(libc::SIGTERM, libc::SIG_DFL);
                libc::signal(libc::SIGHUP, libc::SIG_DFL);
                Ok(())
            });
        }
        let mut child = cmd.spawn().unwrap();
        let pgid = child.id() as i32;
        let out = drain(child.stdout.take().unwrap());
        let err = drain(child.stderr.take().unwrap());
        Spawned {
            child: Some(child),
            pgid,
            out: Some(out),
            err: Some(err),
        }
    }

    /// Run in the repo itself and wait.
    fn run(
        &self,
        script: &str,
        args: &[&str],
        dirs: &[&Path],
        tmpdir: Option<&Path>,
        env: &[(&'static str, OsString)],
    ) -> Out {
        self.spawn(script, args, &self.root.join("repo"), dirs, tmpdir, env)
            .wait(120)
    }
}

fn dev_repo() -> ScriptRepo {
    ScriptRepo::new("dev_fast_check.sh", &[], false, FAKE_DEV_CARGO)
}

#[test]
fn dev_fast_check_passes_when_tests_ran() {
    let repo = dev_repo();
    let tmp = Scratch::dir("relready-devtmp");
    let out = repo.run("dev_fast_check.sh", &[], &[], Some(tmp.path()), &[]);
    assert_eq!(out.code, Some(0), "{}", out.all());
    assert!(out.stdout.contains("SUCCESS"), "{}", out.all());
}

#[test]
fn dev_fast_check_grep_read_error_is_not_tests_ran() {
    // grep rc 2 on the log used to read as "tests ran", so the run reported SUCCESS.
    let repo = dev_repo();
    let tmp = Scratch::dir("relready-devtmp");
    let dir = tool_dir("grep", FAKE_GREP);
    let out = repo.run(
        "dev_fast_check.sh",
        &[],
        &[dir.path()],
        Some(tmp.path()),
        &[ev("FAKE_GREP_FAIL_ON", "tessl_fast_check_")],
    );
    assert_ne!(out.code, Some(0), "{}", out.all());
    assert!(out.stderr.contains("grep failed (rc 2)"), "{}", out.all());
    assert!(!out.stdout.contains("SUCCESS"), "{}", out.all());
}

#[test]
fn dev_fast_check_log_lives_under_tmpdir_and_is_removed() {
    let repo = dev_repo();
    let tmp = Scratch::dir("relready-devtmp");
    let ls = repo.root.join("ls.txt");
    let out = repo.run("dev_fast_check.sh", &[], &[], Some(tmp.path()), &[ev("FAKE_LS", &ls)]);
    assert_eq!(out.code, Some(0), "{}", out.all());
    let seen = std::fs::read_to_string(&ls).unwrap();
    assert!(
        seen.lines().any(|l| l.starts_with("tessl_fast_check_")),
        "the log was not made under TMPDIR; it held: {seen:?}"
    );
    assert_eq!(entries(tmp.path()), Vec::<String>::new());
}

#[test]
fn dev_fast_check_unusable_tmpdir_fails() {
    let repo = dev_repo();
    let missing = repo.root.join("no-such-tmpdir");
    let out = repo.run("dev_fast_check.sh", &[], &[], Some(&missing), &[]);
    assert_eq!(out.code, Some(1), "{}", out.all());
    assert!(out.stderr.contains("could not create a temp log"), "{}", out.all());
    assert!(!out.stdout.contains("SUCCESS"), "{}", out.all());
}

fn guard_repo() -> ScriptRepo {
    ScriptRepo::new(
        "guard_break_it.sh",
        &["tests/reductions.rs", "tests/nn_adversarial.rs"],
        false,
        FAKE_GUARD_CARGO,
    )
}

#[test]
fn guard_break_it_failing_git_status_refuses() {
    // A git that cannot run printed nothing, which read as "both test files
    // match HEAD" and let the run go on to edit them.
    let repo = guard_repo();
    let tmp = Scratch::dir("relready-guardtmp");
    let dir = tool_dir("git", &failing_tool(128));
    let out = repo.run("guard_break_it.sh", &[], &[dir.path()], Some(tmp.path()), &[]);
    assert_eq!(out.code, Some(2), "{}", out.all());
    assert!(
        out.stderr.contains("'git status --porcelain' failed (rc 128)"),
        "{}",
        out.all()
    );
    assert_eq!(entries(tmp.path()), Vec::<String>::new());
}

#[test]
fn guard_break_it_failing_shasum_refuses() {
    // An empty hash from a failed shasum used to equal the empty hash after the
    // restore, so a restore that changed nothing proved nothing.
    let repo = guard_repo();
    let tmp = Scratch::dir("relready-guardtmp");
    let dir = tool_dir("shasum", &failing_tool(1));
    let out = repo.run("guard_break_it.sh", &[], &[dir.path()], Some(tmp.path()), &[]);
    assert_eq!(out.code, Some(2), "{}", out.all());
    assert!(out.stderr.contains("shasum failed on the test files"), "{}", out.all());
    // The run died right after making its backup dir: the exit trap, installed
    // before the first temp path, must have removed it.
    assert_eq!(entries(tmp.path()), Vec::<String>::new(), "{}", out.all());
}

#[test]
fn guard_break_it_temp_paths_live_under_tmpdir() {
    let repo = guard_repo();
    let tmp = Scratch::dir("relready-guardtmp");
    let ls = repo.root.join("ls.txt");
    let out = repo.run("guard_break_it.sh", &[], &[], Some(tmp.path()), &[ev("FAKE_LS", &ls)]);
    // Every baseline is red (no test result), so the verdict is failure.
    assert_eq!(out.code, Some(1), "{}", out.all());
    let seen = std::fs::read_to_string(&ls).unwrap();
    for prefix in ["tessl-guard-bak.", "tessl-guard-log.", "tessl-guard-plan."] {
        assert!(
            seen.lines().any(|l| l.starts_with(prefix)),
            "no {prefix}* under TMPDIR while the driver ran; it held: {seen:?}"
        );
    }
    // The backup dir and the plan file are removed; only the log dir (which the
    // summary points at) stays.
    let left = entries(tmp.path());
    assert!(
        left.len() == 1 && left[0].starts_with("tessl-guard-log."),
        "left behind: {left:?}\n{}",
        out.all()
    );
}

// --- ft-bc7a0de6 round 2: one test per fail-closed site ----------------------

const FAKE_COUNTING_SHASUM: &str = "#!/bin/sh\n\
n=$(cat \"$FAKE_COUNT_DIR/shasum\" 2>/dev/null || echo 0)\n\
n=$((n + 1))\n\
echo \"$n\" > \"$FAKE_COUNT_DIR/shasum\"\n\
if [ \"$n\" -ge \"$FAKE_FAIL_FROM\" ]; then echo \"shasum: fake failure at call $n\" >&2; exit 1; fi\n\
for g in /usr/bin/shasum /usr/local/bin/shasum /opt/homebrew/bin/shasum; do\n\
  [ -x \"$g\" ] && exec \"$g\" \"$@\"\n\
done\n\
exit 99\n";

const FAKE_COUNTING_CP: &str = "#!/bin/sh\n\
n=$(cat \"$FAKE_COUNT_DIR/cp\" 2>/dev/null || echo 0)\n\
n=$((n + 1))\n\
echo \"$n\" > \"$FAKE_COUNT_DIR/cp\"\n\
if [ \"$n\" -ge \"$FAKE_FAIL_FROM\" ]; then echo \"cp: fake failure at call $n\" >&2; exit 1; fi\n\
exec /bin/cp \"$@\"\n";

/// `python3` whose `list` subcommand (the perturbation table's needles) fails
/// (`FAKE_PY_LIST=fail`) or lists no row at all (`empty`); everything else is
/// the real interpreter, `$FAKE_REAL_PYTHON`.
const FAKE_PY: &str = "#!/bin/sh\n\
if [ \"$3\" = list ]; then\n\
  case \"$FAKE_PY_LIST\" in\n\
    fail) echo \"python3: fake list failure\" >&2; exit 3 ;;\n\
    empty) printf 'nosuchid\\tb\\tt\\tn\\n'; exit 0 ;;\n\
  esac\n\
fi\n\
exec \"$FAKE_REAL_PYTHON\" \"$@\"\n";

/// Every call passes ("test result: ok"); call number `$FAKE_PARK_AT` first
/// waits (bounded) for `$FAKE_RELEASE`, after creating `$FAKE_STARTED`.
const FAKE_GUARD_OK_CARGO: &str = r#"#!/bin/sh
n=$(cat "$FAKE_COUNT_DIR/cargo" 2>/dev/null || echo 0)
n=$((n + 1))
echo "$n" > "$FAKE_COUNT_DIR/cargo"
if [ -n "$FAKE_PARK_AT" ] && [ "$n" = "$FAKE_PARK_AT" ]; then
  : > "$FAKE_STARTED"
  i=0
  while [ ! -e "$FAKE_RELEASE" ]; do
    i=$((i + 1))
    [ "$i" -gt 600 ] && exit 98
    sleep 0.1
  done
fi
echo "running 1 test"
echo "test result: ok. 1 passed; 0 failed"
exit 0
"#;

/// A directory holding several stand-in tools.
fn tools_dir(tools: &[(&str, &str)]) -> Scratch {
    let d = Scratch::dir("relready-tools");
    for (name, text) in tools {
        write_exec(&d.join(name), text);
    }
    d
}

/// The guard driver in a repo with the checkout's real test files (its
/// perturbation table edits them) and a fake cargo that passes every run.
fn guard_real_repo() -> ScriptRepo {
    ScriptRepo::new(
        "guard_break_it.sh",
        &["tests/reductions.rs", "tests/nn_adversarial.rs"],
        true,
        FAKE_GUARD_OK_CARGO,
    )
}

/// The id of the first perturbation row, from the driver's own `--list`.
fn first_row_id(repo: &ScriptRepo) -> String {
    let out = Command::new("bash")
        .arg(repo.root.join("repo/scripts/guard_break_it.sh"))
        .arg("--list")
        .current_dir(repo.root.join("repo"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "--list failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.lines()
        .next()
        .and_then(|l| l.split('\t').next())
        .unwrap_or_else(|| panic!("no rows in --list output: {text:?}"))
        .to_string()
}

fn real_python() -> OsString {
    let out = Command::new("sh").args(["-c", "command -v python3"]).output().unwrap();
    assert!(out.status.success(), "no python3 on PATH");
    OsString::from(String::from_utf8_lossy(&out.stdout).trim())
}

/// Run `--only <first row>` with a shasum that fails from call `from` on.
/// Calls: 1 = before, 2 = after the baseline, 3 = after the row's restore,
/// 4 = the exit trap's check.
fn guard_with_failing_shasum(from: u32) -> (Out, Scratch) {
    let repo = guard_real_repo();
    let id = first_row_id(&repo);
    let tmp = Scratch::dir("relready-guardtmp");
    let dir = tools_dir(&[("shasum", FAKE_COUNTING_SHASUM)]);
    let out = repo.run(
        "guard_break_it.sh",
        &["--only", &id],
        &[dir.path()],
        Some(tmp.path()),
        &[ev("FAKE_FAIL_FROM", from.to_string())],
    );
    (out, tmp)
}

#[test]
fn guard_shasum_failing_before_the_backup_is_hashed_is_that_error() {
    // Call 1 fails. Not the later "after the baseline" error a swallowed
    // failure here would turn into.
    let (out, tmp) = guard_with_failing_shasum(1);
    assert_eq!(out.code, Some(2), "{}", out.all());
    assert!(out.stderr.contains("shasum failed on the test files"), "{}", out.all());
    assert_eq!(entries(tmp.path()), Vec::<String>::new(), "{}", out.all());
}

#[test]
fn guard_shasum_failing_after_the_baseline_is_fatal() {
    let (out, _tmp) = guard_with_failing_shasum(2);
    assert_eq!(out.code, Some(2), "{}", out.all());
    assert!(
        out.stderr.contains("shasum failed after the baseline of"),
        "{}",
        out.all()
    );
}

#[test]
fn guard_shasum_failing_after_a_rows_restore_is_fatal() {
    // Not "restore after <id> failed", which an empty hash compared with the
    // real one would give.
    let (out, _tmp) = guard_with_failing_shasum(3);
    assert_eq!(out.code, Some(2), "{}", out.all());
    assert!(
        out.stderr.contains("shasum failed after the restore for"),
        "{}",
        out.all()
    );
}

#[test]
fn guard_shasum_failing_in_the_exit_restore_reports_a_failed_restore() {
    // The exit trap's own hash fails: that is RESTORE FAILED with the reason,
    // not an empty "after" that happens to differ.
    let (out, _tmp) = guard_with_failing_shasum(4);
    assert_eq!(out.code, Some(2), "{}", out.all());
    assert!(out.stderr.contains("guard_break_it: RESTORE FAILED"), "{}", out.all());
    assert!(out.stderr.contains("after:  (shasum failed)"), "{}", out.all());
}

#[test]
fn guard_backup_copy_failure_is_fatal_and_cleans_up() {
    let repo = guard_real_repo();
    let tmp = Scratch::dir("relready-guardtmp");
    let dir = tools_dir(&[("cp", FAKE_COUNTING_CP)]);
    let out = repo.run(
        "guard_break_it.sh",
        &[],
        &[dir.path()],
        Some(tmp.path()),
        &[ev("FAKE_FAIL_FROM", "1")],
    );
    assert_eq!(out.code, Some(2), "{}", out.all());
    assert!(out.stderr.contains("could not back up the test files"), "{}", out.all());
    assert_eq!(entries(tmp.path()), Vec::<String>::new(), "{}", out.all());
}

#[test]
fn guard_per_row_restore_copy_failure_is_fatal() {
    // cp call 1 is the backup; call 2 is the first row's restore.
    let repo = guard_real_repo();
    let id = first_row_id(&repo);
    let tmp = Scratch::dir("relready-guardtmp");
    let dir = tools_dir(&[("cp", FAKE_COUNTING_CP)]);
    let out = repo.run(
        "guard_break_it.sh",
        &["--only", &id],
        &[dir.path()],
        Some(tmp.path()),
        &[ev("FAKE_FAIL_FROM", "2")],
    );
    assert_eq!(out.code, Some(2), "{}", out.all());
    assert!(
        out.stderr
            .contains(&format!("could not restore tests/reductions.rs after {id}"))
            || out
                .stderr
                .contains(&format!("could not restore tests/nn_adversarial.rs after {id}")),
        "{}",
        out.all()
    );
}

#[test]
fn guard_needle_producer_failure_is_fatal() {
    // `py list | awk` failing used to leave an empty needle.
    let repo = guard_real_repo();
    let id = first_row_id(&repo);
    let tmp = Scratch::dir("relready-guardtmp");
    let dir = tools_dir(&[("python3", FAKE_PY)]);
    let out = repo.run(
        "guard_break_it.sh",
        &["--only", &id],
        &[dir.path()],
        Some(tmp.path()),
        &[ev("FAKE_PY_LIST", "fail"), ev("FAKE_REAL_PYTHON", real_python())],
    );
    assert_eq!(out.code, Some(2), "{}", out.all());
    assert!(out.stderr.contains("could not read the needle for"), "{}", out.all());
}

#[test]
fn guard_empty_needle_is_refused_and_the_plan_file_is_removed() {
    // A table with no such row gives an empty needle, which `grep -F -q ""`
    // would match in every log. The die must also leave no plan file or backup.
    let repo = guard_real_repo();
    let id = first_row_id(&repo);
    let tmp = Scratch::dir("relready-guardtmp");
    let dir = tools_dir(&[("python3", FAKE_PY)]);
    let out = repo.run(
        "guard_break_it.sh",
        &["--only", &id],
        &[dir.path()],
        Some(tmp.path()),
        &[ev("FAKE_PY_LIST", "empty"), ev("FAKE_REAL_PYTHON", real_python())],
    );
    assert_eq!(out.code, Some(2), "{}", out.all());
    assert!(out.stderr.contains("empty needle for"), "{}", out.all());
    let left = entries(tmp.path());
    assert!(
        left.len() == 1 && left[0].starts_with("tessl-guard-log."),
        "left behind: {left:?}\n{}",
        out.all()
    );
}

#[test]
fn guard_signal_restores_the_tests_and_exits() {
    // SIGTERM while a row's cargo run is parked (the test files are perturbed
    // then): the handler must restore, clean up and exit 128+15, not return
    // into the loop and carry on perturbing with the backup gone.
    let repo = guard_real_repo();
    let id = first_row_id(&repo);
    let tmp = Scratch::dir("relready-guardtmp");
    let started = repo.root.join("started");
    let release = repo.root.join("release");
    let mut s = repo.spawn(
        "guard_break_it.sh",
        &["--only", &id],
        &repo.root.join("repo"),
        &[],
        Some(tmp.path()),
        &[
            // cargo call 1 is the baseline, call 2 the perturbed row.
            ev("FAKE_PARK_AT", "2"),
            ev("FAKE_STARTED", &started),
            ev("FAKE_RELEASE", &release),
        ],
    );
    s.wait_for_file(&started, 60, "the perturbed row's cargo run to start");
    let perturbed = Command::new("git")
        .arg("-C")
        .arg(repo.root.join("repo"))
        .args(["status", "--porcelain", "--", "tests"])
        .output()
        .unwrap();
    assert!(
        !perturbed.stdout.is_empty(),
        "the row was not applied while cargo ran; the test proves nothing"
    );
    s.signal_pid(libc::SIGTERM);
    std::fs::write(&release, "go\n").unwrap();
    let out = s.wait(60);
    assert_eq!(out.code, Some(143), "{}", out.all());
    let after = Command::new("git")
        .arg("-C")
        .arg(repo.root.join("repo"))
        .args(["status", "--porcelain", "--", "tests"])
        .output()
        .unwrap();
    assert!(
        after.stdout.is_empty(),
        "tests left perturbed: {}\n{}",
        String::from_utf8_lossy(&after.stdout),
        out.all()
    );
    let left = entries(tmp.path());
    assert!(
        left.len() == 1 && left[0].starts_with("tessl-guard-log."),
        "left behind: {left:?}\n{}",
        out.all()
    );
}

#[test]
fn guard_relative_tmpdir_means_the_callers_cwd() {
    let repo = guard_repo();
    let cwd = repo.root.join("cwd");
    std::fs::create_dir_all(cwd.join("rel-tmp")).unwrap();
    let out = repo
        .spawn("guard_break_it.sh", &[], &cwd, &[], Some(Path::new("rel-tmp")), &[])
        .wait(120);
    // Every baseline is red (no test result), so the verdict is failure; what
    // matters is where the temp paths went.
    assert_eq!(out.code, Some(1), "{}", out.all());
    let left = entries(&cwd.join("rel-tmp"));
    assert!(
        left.len() == 1 && left[0].starts_with("tessl-guard-log."),
        "left in the caller's rel-tmp: {left:?}\n{}",
        out.all()
    );
    assert!(!repo.root.join("repo/rel-tmp").exists(), "{}", out.all());
}

#[test]
fn guard_dash_tmpdir_gets_our_own_error() {
    let repo = guard_repo();
    let out = repo.run("guard_break_it.sh", &[], &[], Some(Path::new("-d")), &[]);
    assert_eq!(out.code, Some(2), "{}", out.all());
    assert!(out.stderr.contains("TMPDIR '-d' starts with '-'"), "{}", out.all());
}

#[test]
fn dev_fast_check_relative_tmpdir_means_the_callers_cwd() {
    // The script cd's to its repo root; a relative TMPDIR is the caller's.
    let repo = dev_repo();
    let cwd = repo.root.join("cwd");
    std::fs::create_dir_all(cwd.join("rel-tmp")).unwrap();
    let out = repo
        .spawn("dev_fast_check.sh", &[], &cwd, &[], Some(Path::new("rel-tmp")), &[])
        .wait(120);
    assert_eq!(out.code, Some(0), "{}", out.all());
    assert_eq!(entries(&cwd.join("rel-tmp")), Vec::<String>::new(), "{}", out.all());
}

#[test]
fn dev_fast_check_dash_tmpdir_gets_our_own_error() {
    let repo = dev_repo();
    let out = repo.run("dev_fast_check.sh", &[], &[], Some(Path::new("-d")), &[]);
    assert_eq!(out.code, Some(1), "{}", out.all());
    assert!(out.stderr.contains("TMPDIR '-d' starts with '-'"), "{}", out.all());
    assert!(!out.stdout.contains("SUCCESS"), "{}", out.all());
}

#[test]
fn dev_fast_check_failing_tee_is_not_tests_ran() {
    // tee drains its input and fails: cargo passed, the log is not trustworthy.
    let repo = dev_repo();
    let tmp = Scratch::dir("relready-devtmp");
    let dir = tool_dir(
        "tee",
        "#!/bin/sh\ncat > /dev/null\necho \"tee: fake write failure\" >&2\nexit 1\n",
    );
    let out = repo.run("dev_fast_check.sh", &[], &[dir.path()], Some(tmp.path()), &[]);
    assert_eq!(out.code, Some(1), "{}", out.all());
    assert!(out.stderr.contains("tee failed (rc 1)"), "{}", out.all());
    assert!(!out.stdout.contains("SUCCESS"), "{}", out.all());
}

#[test]
fn missing_changelog_footer_still_fails_the_gate() {
    // grep rc 1 under the three-way split must still be a failure.
    let c = unique_crate("nofoot");
    let _tmp = CrateTmp::new(&c);
    let fx = Fixture::new(&c, "tag-nofoot", &["src/lib.rs"]);
    let repo = fx.path("repo");
    std::fs::write(repo.join("CHANGELOG.md"), "## [0.0.0]\n\nno footer link here\n").unwrap();
    git(&repo, &["add", "-A"]);
    git(&repo, &["commit", "-qm", "drop the footer"]);
    let out = run_with_tool(&fx, Path::new("/nonexistent-tool-dir"), &[]);
    assert_eq!(out.code, Some(1), "{}", out.all());
    assert!(
        out.stderr.contains("CHANGELOG.md missing footer link [0.0.0]:"),
        "{}",
        out.all()
    );
    assert!(!out.stderr.contains("grep failed"), "{}", out.all());
    assert!(!out.stdout.contains("release-ready:"), "{}", out.all());
}
