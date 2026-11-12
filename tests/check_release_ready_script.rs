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
    // inherits SIGINT as ignored; the script's own INT trap needs it default.
    unsafe {
        cmd.pre_exec(move || {
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            libc::signal(libc::SIGTERM, libc::SIG_DFL);
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
        ("QUIT", libc::SIGQUIT, 131),
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
        assert!(
            matches!(out.code, Some(1) | Some(141)),
            "{what} unexpected rc\n{}",
            out.all()
        );
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
