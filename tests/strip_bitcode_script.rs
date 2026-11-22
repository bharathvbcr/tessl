//! `go/engine/strip_bitcode.sh` holds the Go binding's engine archive to being
//! bitcode-free. This holds the script to doing that from any working directory
//! and to failing loudly otherwise.
//!
//! It exists because the script shipped resolving a relative input archive after
//! it had already `cd`ed into its work directory, so `strip_bitcode.sh in.a out.a`
//! failed with relative paths (the Makefile passes exactly those). A test that
//! runs it with absolute paths cannot see that.
//!
//! Each case builds a tiny archive whose members really carry an
//! `__LLVM,__bitcode` section (`clang -fembed-bitcode`), runs the real script from
//! a different directory, and checks the result with `otool`. std-only, no device,
//! no cargo in the fixtures.
#![cfg(target_os = "macos")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn script() -> PathBuf {
    repo().join("go/engine/strip_bitcode.sh")
}

fn run(dir: &Path, program: impl AsRef<std::ffi::OsStr>, args: &[&str]) -> Output {
    let program = program.as_ref();
    Command::new(program)
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap_or_else(|e| panic!("spawn {program:?}: {e}"))
}

fn ok(out: Output, what: &str) -> String {
    assert!(
        out.status.success(),
        "{what} failed: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn bitcode_sections(archive: &Path) -> usize {
    let out = Command::new("otool")
        .arg("-l")
        .arg(archive)
        .output()
        .expect("spawn otool");
    assert!(out.status.success(), "otool -l {archive:?} failed");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| l.contains("__bitcode"))
        .count()
}

/// A fixture directory under target/, with `in/libfix.a` holding two members that
/// each carry embedded bitcode.
fn fixture(name: &str) -> PathBuf {
    let dir = repo().join("target/strip_bitcode_fixtures").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("in")).expect("create fixture");
    for (stem, body) in [
        ("one", "int one(void){return 1;}\n"),
        ("two", "int two(void){return 2;}\n"),
    ] {
        fs::write(dir.join(format!("in/{stem}.c")), body).expect("write source");
        ok(
            run(
                &dir.join("in"),
                "clang",
                &[
                    "-c",
                    "-fembed-bitcode",
                    "-arch",
                    "arm64",
                    &format!("{stem}.c"),
                    "-o",
                    &format!("{stem}.o"),
                ],
            ),
            "clang -fembed-bitcode",
        );
    }
    ok(
        run(
            &dir.join("in"),
            "libtool",
            &["-static", "-o", "libfix.a", "one.o", "two.o"],
        ),
        "libtool",
    );
    assert_eq!(
        bitcode_sections(&dir.join("in/libfix.a")),
        2,
        "the fixture archive must carry a bitcode section in each member, or the test proves nothing"
    );
    dir
}

fn leftovers(dir: &Path) -> Vec<String> {
    fs::read_dir(dir)
        .expect("read output dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(".work."))
        .collect()
}

#[test]
fn relative_paths_from_another_directory_strip_every_member() {
    let dir = fixture("relative");
    fs::create_dir_all(dir.join("elsewhere/deeper")).expect("create cwd");
    // cwd is neither the fixture root nor the output directory: both paths are
    // relative to it, as the Makefile's are relative to go/.
    let cwd = dir.join("elsewhere/deeper");
    let out = run(&cwd, script(), &["../../in/libfix.a", "../../stage/libfix.a"]);
    let text = ok(out, "strip_bitcode.sh with relative paths");
    assert!(text.contains("(2 members)"), "unexpected output: {text}");
    let stripped = dir.join("stage/libfix.a");
    assert!(stripped.is_file(), "no output archive at {stripped:?}");
    assert_eq!(bitcode_sections(&stripped), 0, "output still has bitcode");
    let members = ok(run(&dir, "ar", &["-t", "stage/libfix.a"]), "ar -t");
    assert_eq!(
        members.lines().filter(|l| l.ends_with(".o")).count(),
        2,
        "members: {members}"
    );
    assert!(leftovers(&dir.join("stage")).is_empty(), "work directory left behind");
}

#[test]
fn bare_filenames_in_the_current_directory_work() {
    let dir = fixture("bare");
    let cwd = dir.join("in");
    ok(
        run(&cwd, script(), &["libfix.a", "stripped.a"]),
        "strip_bitcode.sh with bare file names",
    );
    assert_eq!(bitcode_sections(&cwd.join("stripped.a")), 0);
    assert!(leftovers(&cwd).is_empty(), "work directory left behind");
}

#[test]
fn missing_input_fails_and_writes_nothing() {
    let dir = fixture("missing");
    let out = run(&dir, script(), &["in/nope.a", "stage/nope.a"]);
    assert!(!out.status.success(), "a missing input must fail");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("no such archive"), "stderr: {err}");
    assert!(!dir.join("stage/nope.a").exists());
}

#[test]
fn a_stale_output_is_removed_when_the_run_fails() {
    let dir = fixture("stale");
    fs::create_dir_all(dir.join("stage")).expect("create stage");
    fs::write(dir.join("in/empty.a"), "!<arch>\n").expect("write empty archive");
    fs::write(dir.join("stage/empty.a"), "old").expect("write stale output");
    let out = run(&dir, script(), &["in/empty.a", "stage/empty.a"]);
    assert!(!out.status.success(), "an archive with no objects must fail");
    assert!(
        !dir.join("stage/empty.a").exists(),
        "a failed run must not leave an output that reads as built"
    );
    assert!(leftovers(&dir.join("stage")).is_empty(), "work directory left behind");
}

/// `script` run with a directory of stand-in tools ahead of the real ones on PATH.
fn run_with_shim(dir: &Path, tool: &str, body: &str, args: &[&str]) -> Output {
    let shim = dir.join("shim");
    fs::create_dir_all(&shim).expect("create shim dir");
    let path = shim.join(tool);
    fs::write(&path, format!("#!/bin/bash\n{body}\n")).expect("write shim");
    ok(run(dir, "chmod", &["+x", path.to_str().unwrap()]), "chmod");
    let old = std::env::var("PATH").unwrap_or_default();
    Command::new(script())
        .args(args)
        .current_dir(dir)
        .env("PATH", format!("{}:{old}", shim.display()))
        .output()
        .expect("run strip_bitcode.sh")
}

#[test]
fn an_otool_that_fails_fails_the_run_and_writes_nothing() {
    let dir = fixture("otool_fails");
    let out = run_with_shim(
        &dir,
        "otool",
        "echo boom >&2; exit 1",
        &["in/libfix.a", "stage/libfix.a"],
    );
    assert!(!out.status.success(), "a failing otool must not read as 'no bitcode'");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("otool -l failed"), "stderr: {err}");
    assert!(!dir.join("stage/libfix.a").exists());
    assert!(leftovers(&dir.join("stage")).is_empty(), "work directory left behind");
}

#[test]
fn bitcode_followed_by_a_large_listing_is_still_found() {
    let dir = fixture("otool_large");
    // An early-closing reader would SIGPIPE this shim and lose the answer.
    let out = run_with_shim(
        &dir,
        "otool",
        "echo '  sectname __bitcode'; yes 'padding padding padding' | head -c 500000",
        &["in/libfix.a", "stage/libfix.a"],
    );
    assert!(!out.status.success(), "a member with bitcode must fail the run");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("still has a bitcode section"), "stderr: {err}");
    assert!(!dir.join("stage/libfix.a").exists());
}

#[test]
fn a_member_that_is_not_an_object_fails_instead_of_being_dropped() {
    let dir = fixture("non_object");
    // A Mach-O object under a name that does not end in .o: ranlib on this machine
    // drops members that are not Mach-O, so a real object is what survives to be
    // ignored by a script that only looks at *.o.
    fs::copy(dir.join("in/one.o"), dir.join("in/extra.obj")).expect("copy member");
    ok(
        run(&dir.join("in"), "ar", &["rc", "mixed.a", "one.o", "extra.obj"]),
        "ar rc",
    );
    let out = run(&dir, script(), &["in/mixed.a", "stage/mixed.a"]);
    assert!(!out.status.success(), "a non-object member must fail the run");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("not an object: extra.obj"), "stderr: {err}");
    assert!(!dir.join("stage/mixed.a").exists());
}

#[test]
fn a_path_that_starts_with_a_dash_is_a_path_not_an_option() {
    let dir = fixture("dash");
    let cwd = dir.join("in");
    fs::rename(cwd.join("libfix.a"), cwd.join("-dash.a")).expect("rename");
    ok(
        run(&cwd, script(), &["-dash.a", "-clean.a"]),
        "strip_bitcode.sh with dash-leading names",
    );
    assert_eq!(bitcode_sections(&cwd.join("-clean.a")), 0);
}
