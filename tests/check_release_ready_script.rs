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
while [ $# -gt 0 ]; do\n\
  if [ \"$1\" = -o ]; then out=\"$2\"; fi\n\
  shift\n\
done\n\
printf '#!/bin/sh\\nexit 0\\n' > \"$out\"\n\
chmod 755 \"$out\"\n";

const FAKE_CARGO: &str = r#"#!/bin/sh
echo "tag $FAKE_TAG"
case "$*" in
"package --locked --list")
  cat "$FAKE_LIST"
  if [ "${FAKE_LIST_RC:-0}" != 0 ]; then
    echo "error: fake list failure" >&2
    echo "tag $FAKE_TAG" >&2
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
        CrateTmp { crate_name: crate_name.to_string(), _fixed: fixed }
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
        std::fs::write(repo.join("CHANGELOG.md"), "## [0.0.0]\n\n[0.0.0]: https://example.invalid\n").unwrap();
        std::fs::write(repo.join("README.md"), "fixture\n").unwrap();
        std::fs::write(repo.join("src/bin/release_check.rs"), "fn main() {}\n").unwrap();
        write_exec(&root.join("fakebin/rustc"), FAKE_RUSTC);
        write_exec(&root.join("fakebin/cargo"), FAKE_CARGO);
        std::fs::write(root.join("list.txt"), list.iter().map(|l| format!("{l}\n")).collect::<String>()).unwrap();
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
        Fixture { root, tag: tag.to_string() }
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
    let mut path = OsString::from(fx.path("fakebin"));
    path.push(":");
    path.push(std::env::var_os("PATH").unwrap_or_default());
    let mut cmd = Command::new("bash");
    cmd.arg(fx.path("repo/scripts/check_release_ready.sh"))
        .current_dir(fx.path("."))
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
        cmd.pre_exec(|| {
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            libc::signal(libc::SIGTERM, libc::SIG_DFL);
            Ok(())
        });
    }
    let mut child = cmd.spawn().unwrap();
    let pgid = child.id() as i32;
    let out = Some(drain(child.stdout.take().unwrap()));
    let err = Some(drain(child.stderr.take().unwrap()));
    Spawned { child: Some(child), pgid, out, err }
}

impl Spawned {
    fn signal_group(&self, sig: i32) {
        // SAFETY: plain syscall on a process group this struct created; a
        // vanished group just yields ESRCH.
        unsafe {
            libc::kill(-self.pgid, sig);
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
        let out = Out { code: status.and_then(|s| s.code()), stdout, stderr };
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
    assert_eq!(
        oa.code,
        Some(1),
        "the gate passed on another run's list\n{}",
        oa.all()
    );
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
    assert_eq!(tmp_guard.present(), Vec::<PathBuf>::new(), "fixed /tmp files written\n{}", out.all());
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
        assert!(!run_dir.exists(), "run dir {} survived\n{}", run_dir.display(), out.all());
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
    assert_eq!(tmp_guard.present(), Vec::<PathBuf>::new(), "fell back to /tmp\n{}", out.all());
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
            list: vec!["src/lib.rs", ".devmap/denied"],
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
        assert_eq!(tmp_guard.present(), Vec::<PathBuf>::new(), "[{}] {}", case.name, out.all());
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
        assert!(text.contains(&tag), "[{}] kept log lacks this run's tag: {text:?}", case.name);
        assert_eq!(entries(&run_dir), vec![case.log.to_string()], "[{}] extra files kept", case.name);
        remove_path(&run_dir);
    }
}

#[test]
fn signals_remove_the_run_dir() {
    for (name, sig, code) in [("INT", libc::SIGINT, 130), ("TERM", libc::SIGTERM, 143)] {
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
        assert!(!run_dir.exists(), "[{name}] run dir {} survived\n{}", run_dir.display(), out.all());
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
    assert_eq!(leftovers(&escaped, &escape_prefix), Vec::<PathBuf>::new(), "{}", out.all());
    assert_eq!(entries(&tmpdir), vec!["sentinel".to_string()]);
}
