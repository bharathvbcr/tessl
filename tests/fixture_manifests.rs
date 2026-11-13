//! Every generator-backed fixture directory carries a manifest that says where its
//! arrays came from, and the manifest accounts for every file in the directory.
//!
//! `tests/fixtures/qwen35/`, `qwen35_train/` and `qwen35_train_grouped/` are written by
//! `scripts/gen_qwen35_fixtures.py` and `tools/qwen35_ref/make_train_fixture.py`. A
//! golden is only as good as the record of what produced it: a torch generator's
//! stream moves across torch builds (a rerun on another build changed 16 of 18 files
//! of `qwen35/`), so a directory without versions, a generator commit and an inputs
//! source cannot be told apart from one somebody edited by hand. `MANIFEST.json`,
//! in the format `tools/qwen35_ref/fixture_manifest.py` writes, is that record, and
//! this test holds it to the directory: it parses, names the generator and the
//! versions, and lists exactly the files present, each with a role and a
//! `published` status. No device is opened.
//!
//! The directories are discovered, not listed: every directory under
//! `tests/fixtures/` must carry `MANIFEST.json` in this format, or be named in
//! [`OWN_RECORD`] (a directory whose generator predates this format and writes a
//! record of its own, which must exist and parse), in [`SPLIT_RECORDS`] (several
//! generators, each with its own record and its own file-name prefix, every file
//! owned by exactly one), or in [`EXEMPT`] (not generator-backed, with the
//! reason). A new directory with none of the four fails, so a fixture cannot
//! arrive without its provenance.
//!
//! Two provenance checks run over every `MANIFEST.json` (ft-bab435ce):
//!
//! * the pinned `generator_commit` resolves and is on this checkout's history
//!   ([`judge_pin`]). That needs history, so on a shallow clone it is **NOT RUN**
//!   and fails, naming the fix (`fetch-depth: 0`); it never passes there;
//! * every `sha256` a manifest records matches its file ([`verify_fixture_bytes`]).
//!   Only `dequant_affine` records any; the directories in [`NO_FILE_SHA256`] do
//!   not, and that gap is a named table, not a silent skip.
//!
//! `dequant_affine`'s pin must also be the commit that last changed its generator
//! ([`judge_last_touch`]): its goldens are checked bit for bit, so a generator
//! edited after them has to be regenerated, not re-pinned. No device is opened.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

#[path = "../src/bin/common/json.rs"]
mod json;
mod scratch_support;

use json::{Json, Syntax};
use scratch_support::Scratch;

/// Directories whose generator writes its own record, in its own format. Each file
/// must exist and parse; moving them onto `MANIFEST.json` is a change to their
/// generator (`scripts/gen_bert_fixtures.py`) and to the nanolab corpus manifest,
/// not to this test.
const OWN_RECORD: [(&str, &str); 2] = [("bert_tiny", "manifest.json"), ("gdn", "gdn_published_MANIFEST.json")];

/// Directories that several generators write into, each its own family of files
/// (`(record, file-name prefix, generator)`) with a record of its own. Every record
/// must exist, parse and name its generator, which must exist in the repo, and
/// every other file must start with exactly one family's prefix, so a file no
/// generator accounts for cannot arrive unnoticed.
type Family = (&'static str, &'static str, &'static str);
const SPLIT_RECORDS: [(&str, &[Family]); 1] = [(
    "embedgemma2",
    &[
        ("provenance.json", "eg2_", "scripts/gen_embedgemma2_fixtures.py"),
        (
            "eg2a_audio_provenance.json",
            "eg2a_",
            "scripts/gen_embedgemma2_audio_fixtures.py",
        ),
        (
            "eg2p_provenance.json",
            "eg2p_",
            "scripts/gen_embedgemma2_pre_fixtures.py",
        ),
    ],
)];

/// [`SPLIT_RECORDS`]' rule for one directory: `files` is every file in it and
/// `read` returns a record's text.
fn check_split(
    files: &BTreeSet<String>,
    families: &[Family],
    read: impl Fn(&str) -> Result<String, String>,
    repo: &Path,
) -> Result<(), String> {
    for (record, _, generator) in families {
        if !files.contains(*record) {
            return Err(format!("its record {record} is missing"));
        }
        let doc = parse(&read(record)?).map_err(|e| format!("{record} does not parse: {e}"))?;
        match doc.get("generator") {
            Some(Json::Str(g)) if g == generator => {}
            other => return Err(format!("{record} must name generator {generator:?}, found {other:?}")),
        }
        if !repo.join(generator).is_file() {
            return Err(format!("{record} names {generator}, which is not in the repo"));
        }
    }
    for f in files {
        if families.iter().any(|(record, _, _)| record == f) {
            continue;
        }
        let owners = families.iter().filter(|(_, prefix, _)| f.starts_with(prefix)).count();
        if owners != 1 {
            return Err(format!(
                "{f} belongs to {owners} generator families (prefixes {:?}); it needs exactly one",
                families.iter().map(|(_, p, _)| *p).collect::<Vec<_>>()
            ));
        }
    }
    Ok(())
}

/// Directories with no generator to record. Nothing that has a generator belongs
/// here; each entry says why it is not generator-backed.
const EXEMPT: [(&str, &str); 2] = [
    (
        "gemm_parity",
        "a captured GPU dump (bench_gemm_sweep --dump-parity lanes and the MLX/torch comparison \
         lanes, 2 seeds of 24x20x48) scored once by bench/gemm_sweep_mlx.py's parity() at b219aff; \
         the device outputs are captures, so no generator run reproduces them",
    ),
    (
        "results_report",
        "goldens written by the five stdlib report scripts retired in 1b42066 \
         (bench/results/embedgemma2_*_report.py); src/bin/results_report is held to them byte for byte, \
         and no generator remains to record",
    ),
];

const ROLES: [&str; 4] = ["input", "golden", "model", "config"];

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn parse(text: &str) -> Result<Json, String> {
    json::parse(
        text,
        Syntax {
            what: "MANIFEST.json",
            max_depth: 8,
            uints_only: false,
            literals: true,
        },
    )
}

fn string<'a>(v: &'a Json, key: &str) -> Result<&'a str, String> {
    match v.get(key) {
        Some(Json::Str(s)) if !s.trim().is_empty() => Ok(s),
        Some(other) => Err(format!("{key} must be a non-empty string, got {other:?}")),
        None => Err(format!("{key} is missing")),
    }
}

/// `Ok` when `text` is a well-formed manifest whose `files` are exactly `present`.
fn check_manifest(text: &str, present: &BTreeSet<String>, repo: &Path) -> Result<(), String> {
    let m = parse(text)?;
    if m.get("schema").and_then(Json::exact_int) != Some(1) {
        return Err("schema must be 1".into());
    }
    let generator = string(&m, "generator")?;
    if !repo.join(generator).is_file() {
        return Err(format!("generator {generator} is not a file in the repo"));
    }
    let commit = string(&m, "generator_commit")?;
    if commit.len() != 40 || !commit.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return Err(format!(
            "generator_commit {commit:?} is not a lowercase 40-hex commit id"
        ));
    }
    let versions = m.get("versions").ok_or("versions is missing")?;
    for tool in ["torch", "transformers", "numpy"] {
        string(versions, tool).map_err(|e| format!("versions.{e}"))?;
    }
    match string(&m, "versions_role")? {
        "produced" => {}
        "reproduced" if m.get("reproduction").is_some() => {}
        "reproduced" => return Err("versions_role is reproduced but there is no reproduction record".into()),
        other => return Err(format!("versions_role {other:?} is neither produced nor reproduced")),
    }
    string(&m, "inputs_source")?;

    let Some(Json::Object(files)) = m.get("files") else {
        return Err("files must be an object".into());
    };
    let mut listed = BTreeSet::new();
    for (name, meta) in files {
        let role = string(meta, "role").map_err(|e| format!("{name}: {e}"))?;
        if !ROLES.contains(&role) {
            return Err(format!("{name}: role {role:?} is not one of {ROLES:?}"));
        }
        if !matches!(meta.get("published"), Some(Json::Bool(_))) {
            return Err(format!("{name}: published must be a boolean"));
        }
        string(meta, "note").map_err(|e| format!("{name}: {e}"))?;
        listed.insert(name.clone());
    }
    let unlisted: Vec<_> = present.difference(&listed).collect();
    let absent: Vec<_> = listed.difference(present).collect();
    if !unlisted.is_empty() || !absent.is_empty() {
        return Err(format!(
            "files and directory disagree: present but unlisted {unlisted:?}, listed but absent {absent:?}"
        ));
    }
    Ok(())
}

fn files_in(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.expect("readable entry").file_name().to_string_lossy().into_owned())
        .filter(|n| n != "MANIFEST.json")
        .collect()
}

fn has_entry_named(dir: &Path, name: &str) -> bool {
    std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.expect("readable entry"))
        .any(|e| e.file_name() == name && e.path().is_file())
}

#[test]
fn every_fixture_directory_has_a_manifest_that_lists_every_file() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut dirs: Vec<String> = std::fs::read_dir(fixtures())
        .expect("tests/fixtures exists")
        .map(|e| e.expect("readable entry"))
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    dirs.sort();
    assert!(
        !dirs.is_empty(),
        "no fixture directories found; the test would pass vacuously"
    );
    for d in &dirs {
        let dir = fixtures().join(d);
        let manifest = dir.join("MANIFEST.json");
        // By the directory's own entry name, not `is_file()`: on a
        // case-insensitive volume (APFS's default) `MANIFEST.json` would find
        // bert_tiny's `manifest.json`, its own record in another format.
        if has_entry_named(&dir, "MANIFEST.json") {
            let text =
                std::fs::read_to_string(&manifest).unwrap_or_else(|e| panic!("tests/fixtures/{d}/MANIFEST.json: {e}"));
            if let Err(e) = check_manifest(&text, &files_in(&dir), repo) {
                panic!("tests/fixtures/{d}/MANIFEST.json: {e}");
            }
        } else if let Some((_, families)) = SPLIT_RECORDS.iter().find(|(n, _)| n == d) {
            let read = |r: &str| std::fs::read_to_string(dir.join(r)).map_err(|e| format!("{r}: {e}"));
            if let Err(e) = check_split(&files_in(&dir), families, read, repo) {
                panic!("tests/fixtures/{d}: {e}");
            }
        } else if let Some((_, record)) = OWN_RECORD.iter().find(|(n, _)| n == d) {
            let text = std::fs::read_to_string(dir.join(record))
                .unwrap_or_else(|e| panic!("tests/fixtures/{d} names {record} as its record: {e}"));
            parse(&text).unwrap_or_else(|e| panic!("tests/fixtures/{d}/{record} does not parse: {e}"));
        } else if !EXEMPT.iter().any(|(n, _)| n == d) {
            panic!(
                "tests/fixtures/{d} has no MANIFEST.json and is in neither OWN_RECORD nor EXEMPT; \
                 write one with tools/qwen35_ref/fixture_manifest.py"
            );
        }
    }
    // A stale table entry would hide a removed or renamed directory.
    for n in OWN_RECORD
        .iter()
        .chain(EXEMPT.iter())
        .map(|(n, _)| *n)
        .chain(SPLIT_RECORDS.iter().map(|(n, _)| *n))
    {
        assert!(
            dirs.iter().any(|d| d == n),
            "the table names tests/fixtures/{n}, which does not exist"
        );
    }
}

/// The split-record rule refuses a missing record, a record naming another or no
/// generator, a generator not in the repo, and a file that no family (or two)
/// owns, so the green run above is not vacuous.
#[test]
fn the_split_record_check_refuses_an_unrecorded_file_or_a_bad_record() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fams: &[Family] = &[
        ("provenance.json", "eg2_", "scripts/gen_embedgemma2_fixtures.py"),
        ("eg2a_rec.json", "eg2a_", "scripts/gen_embedgemma2_audio_fixtures.py"),
    ];
    let files = |names: &[&str]| names.iter().map(|s| s.to_string()).collect::<BTreeSet<_>>();
    let ok = |r: &str| {
        Ok(match r {
            "provenance.json" => r#"{"generator":"scripts/gen_embedgemma2_fixtures.py"}"#,
            _ => r#"{"generator":"scripts/gen_embedgemma2_audio_fixtures.py"}"#,
        }
        .to_string())
    };
    let all = files(&["provenance.json", "eg2a_rec.json", "eg2_x.npy", "eg2a_y.npy"]);
    check_split(&all, fams, ok, repo).unwrap();

    let e = check_split(&files(&["provenance.json", "eg2_x.npy"]), fams, ok, repo).unwrap_err();
    assert!(e.contains("eg2a_rec.json is missing"), "{e}");

    let mut stray = all.clone();
    stray.insert("other.npy".into());
    let e = check_split(&stray, fams, ok, repo).unwrap_err();
    assert!(e.contains("other.npy belongs to 0"), "{e}");

    let overlap: &[Family] = &[
        ("provenance.json", "eg2", "scripts/gen_embedgemma2_fixtures.py"),
        ("eg2a_rec.json", "eg2a_", "scripts/gen_embedgemma2_audio_fixtures.py"),
    ];
    let e = check_split(&all, overlap, ok, repo).unwrap_err();
    assert!(e.contains("eg2a_y.npy belongs to 2"), "{e}");

    let wrong = |r: &str| {
        Ok(if r == "provenance.json" {
            r#"{"generator":"scripts/elsewhere.py"}"#.to_string()
        } else {
            ok(r)?
        })
    };
    let e = check_split(&all, fams, wrong, repo).unwrap_err();
    assert!(e.contains("must name generator"), "{e}");

    let unnamed = |r: &str| Ok(if r == "eg2a_rec.json" { "{}".to_string() } else { ok(r)? });
    let e = check_split(&all, fams, unnamed, repo).unwrap_err();
    assert!(e.contains("found None"), "{e}");

    let gone: &[Family] = &[("provenance.json", "eg2_", "scripts/no_such_generator.py")];
    let gone_read = |_: &str| Ok(r#"{"generator":"scripts/no_such_generator.py"}"#.to_string());
    let e = check_split(&files(&["provenance.json", "eg2_x.npy"]), gone, gone_read, repo).unwrap_err();
    assert!(e.contains("not in the repo"), "{e}");
}

/// The checker refuses what it exists to catch, so a green run above is not vacuous.
#[test]
fn the_manifest_check_refuses_a_missing_extra_or_malformed_entry() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let good = |files: &str| {
        format!(
            r#"{{"schema":1,"generator":"scripts/gen_qwen35_fixtures.py",
            "generator_commit":"{}","versions":{{"torch":"2","transformers":"5","numpy":"2"}},
            "versions_role":"produced","inputs_source":"stored","files":{{{files}}}}}"#,
            "a".repeat(40)
        )
    };
    let entry = |n: &str, published: &str| format!(r#""{n}":{{"role":"golden","published":{published},"note":"n"}}"#);
    let set = |names: &[&str]| names.iter().map(|s| s.to_string()).collect::<BTreeSet<_>>();

    check_manifest(&good(&entry("a.npy", "true")), &set(&["a.npy"]), repo).expect("a sound manifest passes");
    let unlisted = check_manifest(&good(&entry("a.npy", "true")), &set(&["a.npy", "b.npy"]), repo);
    assert!(
        unlisted.unwrap_err().contains("b.npy"),
        "a file nobody listed must be named"
    );
    let absent = check_manifest(&good(&entry("a.npy", "true")), &set(&[]), repo);
    assert!(
        absent.unwrap_err().contains("a.npy"),
        "a listed file that is gone must be named"
    );
    let untyped = check_manifest(&good(&entry("a.npy", "\"yes\"")), &set(&["a.npy"]), repo);
    assert!(
        untyped.unwrap_err().contains("published"),
        "published must be a boolean"
    );
    let no_commit = good(&entry("a.npy", "true")).replace(&"a".repeat(40), "HEAD");
    assert!(
        check_manifest(&no_commit, &set(&["a.npy"]), repo).is_err(),
        "a commit id must be 40 hex"
    );
    assert!(
        check_manifest("{", &set(&[]), repo).is_err(),
        "unparseable text is refused"
    );
}

/// The owner's decision on `qwen35_gdn_*` (ft-b8561dbb): the legacy GDN arrays are
/// marked `published: false`, and the gate golden the kernel is held to is `true`.
#[test]
fn the_legacy_gdn_arrays_are_not_published_and_the_gate_golden_is() {
    let text = std::fs::read_to_string(fixtures().join("qwen35/MANIFEST.json")).expect("qwen35 manifest");
    let Some(Json::Object(files)) = parse(&text).expect("parses").get("files").cloned() else {
        panic!("files must be an object");
    };
    let published = |name: &str| -> bool {
        let (_, meta) = files
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("{name} is not listed"));
        matches!(meta.get("published"), Some(Json::Bool(true)))
    };
    let mut gdn = 0;
    for (name, _) in &files {
        if name.starts_with("qwen35_gdn_") {
            gdn += 1;
            assert!(
                !published(name),
                "{name} must be published: false (ft-b8561dbb decision 3)"
            );
        }
    }
    assert_eq!(gdn, 12, "qwen35_gdn_* has 8 inputs and 4 goldens");
    for g in ["qwen35_gates_g.npy", "qwen35_gates_beta.npy"] {
        assert!(
            published(g),
            "{g} is the qwen35_gdn_gates_f32 golden and must be published"
        );
    }
}

// ------------------------------------------------------------- provenance ---
//
// Everything below is device-free and is the one owner of the git and checksum
// checks: `tests/dequant_affine.rs` carried a copy for its own manifest until
// ft-bab435ce moved it here, where it covers every manifest.

/// What a git call gives back: `(exit code, trimmed stdout)`, or why it could not run.
type GitOutput = Result<(i32, String), String>;
/// A way of running git, so a verdict can be tested against scripted outcomes.
type GitRun<'a> = dyn Fn(&[&str]) -> GitOutput + 'a;
/// Longest any one git call may take before provenance is "could not verify".
const GIT_DEADLINE: Duration = Duration::from_secs(30);

/// Every environment variable whose name starts with this is removed from a git child.
/// git reads a long list of them, and several change what a history question answers or
/// where it lands, not only which repository is opened: `GIT_DIR`, `GIT_WORK_TREE`,
/// `GIT_INDEX_FILE`, `GIT_OBJECT_DIRECTORY`, `GIT_ALTERNATE_OBJECT_DIRECTORIES`,
/// `GIT_COMMON_DIR`, `GIT_NAMESPACE`, `GIT_CEILING_DIRECTORIES`, `GIT_GRAFT_FILE` (fakes
/// parents, so a commit that is not an ancestor can pass), `GIT_REPLACE_REF_BASE`,
/// `GIT_CONFIG_PARAMETERS`, `GIT_CONFIG_COUNT`/`KEY_n`/`VALUE_n` (any config key). A
/// prefix removes them all, and any that a later git adds.
const GIT_ENV_PREFIX: &str = "GIT_";

/// Set on a git child after the scrub, so the user's and the system's git config cannot
/// change an answer (an `alias`, a `core.fsmonitor` daemon, a `core.worktree`, `grafts`):
/// `GIT_CONFIG_NOSYSTEM` skips `/etc/gitconfig`, and `GIT_CONFIG_GLOBAL=/dev/null`
/// skips `~/.gitconfig` and the XDG one. The repository's own `.git/config` still
/// applies. Hooks and the fsmonitor are also switched off with `-c` in [`git_in`].
const GIT_ENV_AFTER_SCRUB: [(&str, &str); 2] = [("GIT_CONFIG_NOSYSTEM", "1"), ("GIT_CONFIG_GLOBAL", "/dev/null")];

/// Run `program` in `dir` and give `(exit code, trimmed stdout)`, the whole call bounded by
/// `deadline`: waiting for the exit and reading the output both count. `inherited` is
/// what the caller's environment would add (applied first). With `scrub_prefix`, every
/// variable of the real environment or of `inherited` whose name starts with it is then
/// removed, and `after_scrub` is set last. A program that cannot start, that a signal
/// ends, that runs past the deadline, or whose output pipe is still open at the deadline
/// is an `Err`. A program still running at the deadline is killed. One that has already
/// exited is not: an orphaned child holds the pipe open, stays running, and its reader
/// thread ends when the orphan closes it.
fn run_bounded(
    program: &str,
    args: &[&str],
    inherited: &[(&str, &str)],
    scrub_prefix: Option<&str>,
    after_scrub: &[(&str, &str)],
    dir: &Path,
    deadline: Duration,
) -> GitOutput {
    let what = format!("{program} {args:?}");
    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    for (name, value) in inherited {
        command.env(name, value);
    }
    if let Some(prefix) = scrub_prefix {
        let real = std::env::vars_os().map(|(name, _)| name);
        let modelled = inherited.iter().map(|(name, _)| std::ffi::OsString::from(name));
        for name in real.chain(modelled) {
            if name.to_string_lossy().starts_with(prefix) {
                command.env_remove(name);
            }
        }
    }
    for (name, value) in after_scrub {
        command.env(name, value);
    }
    let stop = Instant::now() + deadline;
    let mut child = command.spawn().map_err(|e| format!("{what} did not start: {e}"))?;
    let mut stdout = child.stdout.take().ok_or_else(|| format!("{what}: no stdout pipe"))?;
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let mut out = String::new();
        let read = std::io::Read::read_to_string(&mut stdout, &mut out).map(|_| out);
        let _ = sender.send(read);
    });
    let kill = |child: &mut Child| {
        let _ = child.kill();
        let _ = child.wait();
    };
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= stop => {
                kill(&mut child);
                return Err(format!("{what} still running after {deadline:?}; killed"));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(e) => {
                kill(&mut child);
                return Err(format!("{what} could not be waited on: {e}"));
            }
        }
    };
    let out = match receiver.recv_timeout(stop.saturating_duration_since(Instant::now())) {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => return Err(format!("{what} output unreadable: {e}")),
        Err(_) => {
            return Err(format!(
                "{what} exited but its output pipe was still open after {deadline:?} (an orphaned child holds it)"
            ))
        }
    };
    let code = status.code().ok_or_else(|| format!("{what} was killed by a signal"))?;
    Ok((code, out.trim().to_string()))
}

/// git in `dir` with every `GIT_*` variable removed ([`GIT_ENV_PREFIX`]), `inherited`
/// modelling what the caller's environment held, the user and system config switched
/// off ([`GIT_ENV_AFTER_SCRUB`]), and hooks and the fsmonitor disabled, bounded by
/// `deadline`.
fn git_in(dir: &Path, args: &[&str], inherited: &[(&str, &str)], deadline: Duration) -> GitOutput {
    let mut full = vec!["-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false"];
    full.extend_from_slice(args);
    run_bounded(
        "git",
        &full,
        inherited,
        Some(GIT_ENV_PREFIX),
        &GIT_ENV_AFTER_SCRUB,
        dir,
        deadline,
    )
}

fn repo() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// What `git rev-parse --is-shallow-repository` said, as a verdict on whether history
/// questions can be answered. `Ok` only for a plain `false`. A shallow clone is a
/// **NOT RUN** that names its fix, so it can never read as "the commit resolves"; any
/// other answer (a failure, an old git printing the option back) is "could not verify".
fn shallow_verdict(code: i32, printed: &str) -> Result<(), String> {
    match (code, printed) {
        (0, "false") => Ok(()),
        (0, "true") => Err(
            "NOT RUN: provenance was not checked because this clone is shallow (git rev-parse \
             --is-shallow-repository printed true), so a pinned commit cannot be found or ruled out. \
             Fetch full history: `fetch-depth: 0` on actions/checkout, or `git fetch --unshallow`"
                .into(),
        ),
        _ => Err(format!(
            "could not verify provenance: this is not known to be a full clone (rev-parse \
             --is-shallow-repository exited {code}, printed {printed:?})"
        )),
    }
}

/// The verdict on a manifest's `generator_commit` `pin`, given `run` (git in `root`) and
/// `root`, the crate's directory. `Ok` only when git sees the repository rooted at
/// `root`, the clone is not shallow, and `pin` is a commit on the history of `HEAD`. Every
/// other outcome, including one where a check could not run, is an `Err` naming it.
fn judge_pin(pin: &str, root: &Path, run: &GitRun<'_>) -> Result<(), String> {
    let unverified = |why: String| format!("could not verify provenance: {why}");
    let (code, top) = run(&["rev-parse", "--show-toplevel"]).map_err(unverified)?;
    if code != 0 {
        return Err(unverified(format!("rev-parse --show-toplevel exited {code}")));
    }
    let seen = std::fs::canonicalize(&top).map_err(|e| unverified(format!("git's toplevel {top:?}: {e}")))?;
    let want = std::fs::canonicalize(root).map_err(|e| unverified(format!("crate directory {root:?}: {e}")))?;
    if seen != want {
        return Err(unverified(format!(
            "git sees the repository at {}, not at the crate directory {}",
            seen.display(),
            want.display()
        )));
    }
    let (code, shallow) = run(&["rev-parse", "--is-shallow-repository"]).map_err(unverified)?;
    shallow_verdict(code, &shallow)?;
    let commit = format!("{pin}^{{commit}}");
    let (code, _) = run(&["cat-file", "-e", &commit]).map_err(unverified)?;
    if code != 0 {
        return Err(format!(
            "generator_commit {pin} does not resolve to a commit in this repository (cat-file -e exited {code}); \
             re-pin it to the commit that holds the same generator source"
        ));
    }
    let (code, _) = run(&["merge-base", "--is-ancestor", pin, "HEAD"]).map_err(unverified)?;
    match code {
        0 => Ok(()),
        1 => Err(format!("generator_commit {pin} is not on this history")),
        other => Err(unverified(format!("merge-base --is-ancestor exited {other}"))),
    }
}

/// `Ok` when `pin` is the commit that last changed `generator`. Run after [`judge_pin`]
/// has passed, which settles that the history is complete.
fn judge_last_touch(pin: &str, generator: &str, run: &GitRun<'_>) -> Result<(), String> {
    let (code, last) = run(&["log", "-1", "--format=%H", "--", generator])
        .map_err(|e| format!("could not verify provenance: {e}"))?;
    if code != 0 || last.len() != 40 {
        return Err(format!(
            "could not verify provenance: git log of {generator} exited {code} and printed {last:?}"
        ));
    }
    if last != pin {
        return Err(format!(
            "generator changed after the fixtures: the manifest pins {pin}, the generator was last changed in {last}; regenerate"
        ));
    }
    Ok(())
}

fn is_lower_hex(text: &str, len: usize) -> bool {
    text.len() == len && text.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Every fixture directory that carries a `MANIFEST.json` in this format, with its
/// path and parsed manifest, in name order. Found by the directory's own entry name
/// ([`has_entry_named`]), so `bert_tiny/manifest.json` is not mistaken for one on a
/// case-insensitive volume.
fn manifests() -> Vec<(String, PathBuf, Json)> {
    let mut names: Vec<String> = std::fs::read_dir(fixtures())
        .expect("tests/fixtures exists")
        .map(|e| e.expect("readable entry"))
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    let mut found = Vec::new();
    for name in names {
        let dir = fixtures().join(&name);
        if has_entry_named(&dir, "MANIFEST.json") {
            let manifest = read_manifest(&dir).unwrap_or_else(|e| panic!("tests/fixtures/{name}: {e}"));
            found.push((name, dir, manifest));
        }
    }
    found
}

/// The parsed `MANIFEST.json` of the fixture directory `dir`.
fn read_manifest(dir: &Path) -> Result<Json, String> {
    let path = dir.join("MANIFEST.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&text).map_err(|e| format!("MANIFEST.json does not parse: {e}"))
}

/// Every file the manifest lists has the sha256 it records, and the directory holds
/// no file the manifest does not list: the pin names the generator, this names the bytes.
fn verify_fixture_bytes(dir: &Path, manifest: &Json) -> Result<(), String> {
    let Some(Json::Object(files)) = manifest.get("files") else {
        return Err("MANIFEST.json has no files object".into());
    };
    let mut listed = BTreeSet::new();
    for (name, meta) in files {
        if name.is_empty() || name.contains(['/', '\\']) || name == "." || name == ".." || name == "MANIFEST.json" {
            return Err(format!(
                "MANIFEST.json lists {name:?}, which is not a plain fixture file name"
            ));
        }
        let Some(Json::Str(want)) = meta.get("sha256") else {
            return Err(format!("{name}: MANIFEST.json records no sha256"));
        };
        if !is_lower_hex(want, 64) {
            return Err(format!(
                "{name}: the recorded sha256 {want:?} is not 64 lowercase hex digits"
            ));
        }
        let bytes = std::fs::read(dir.join(name)).map_err(|e| format!("{name}: {e}"))?;
        let got = ojas_io::sha256::sha256_hex(&bytes);
        if &got != want {
            return Err(format!("{name}: sha256 is {got}, MANIFEST.json records {want}"));
        }
        if !listed.insert(name.clone()) {
            return Err(format!("MANIFEST.json lists {name} twice"));
        }
    }
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name != "MANIFEST.json" && !listed.contains(&name) {
            return Err(format!("{name} is in the fixture directory but not in MANIFEST.json"));
        }
    }
    Ok(())
}

/// Whether any file entry of `manifest` records a `sha256`.
fn records_any_sha256(manifest: &Json) -> bool {
    matches!(manifest.get("files"), Some(Json::Object(files))
        if files.iter().any(|(_, meta)| meta.get("sha256").is_some()))
}

/// Manifests that record no per-file `sha256`, so check (b) cannot cover their bytes.
/// This is a coverage gap named in one place: a manifest that gains checksums must leave
/// this table (the test refuses a stale entry), and a new manifest without them must
/// join it on purpose. The fix is a change to each generator's manifest writer
/// (`tools/qwen35_ref/fixture_manifest.py` and the media generators), not to this test.
const NO_FILE_SHA256: [&str; 10] = [
    "embedgemma2_tok",
    "jpeg",
    "media",
    "media_resample",
    "media_resample_cr",
    "media_video",
    "qwen35",
    "qwen35_train",
    "qwen35_train_grouped",
    "qwen35_train_untied",
];

/// The directory whose pin must also be its generator's last change (see
/// [`judge_last_touch`]).
const LAST_TOUCH_PINNED: &str = "dequant_affine";

fn dequant_dir() -> PathBuf {
    fixtures().join(LAST_TOUCH_PINNED)
}

/// (a) Every manifest's `generator_commit` resolves to a commit on this checkout's
/// history. Needs a full clone: on a shallow one this is NOT RUN and fails, loudly.
#[test]
fn every_manifest_pins_a_generator_commit_on_this_history() {
    let all = manifests();
    // 11 manifests at the time of writing; losing one must not read as a smaller green run.
    assert!(all.len() >= 11, "only {} MANIFEST.json files found", all.len());
    let run = |args: &[&str]| git_in(repo(), args, &[], GIT_DEADLINE);
    let mut failures = Vec::new();
    for (name, _, manifest) in &all {
        let pin = string(manifest, "generator_commit").unwrap_or_else(|e| panic!("tests/fixtures/{name}: {e}"));
        if let Err(why) = judge_pin(pin, repo(), &run) {
            failures.push(format!("tests/fixtures/{name}/MANIFEST.json: {why}"));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The dequant goldens are checked bit for bit, so its pin must be the last change to
/// its generator.
#[test]
fn the_dequant_pin_is_the_generators_last_touch() {
    let manifest = read_manifest(&dequant_dir()).expect("dequant manifest");
    let pin = string(&manifest, "generator_commit").expect("pin");
    let generator = string(&manifest, "generator").expect("generator");
    let run = |args: &[&str]| git_in(repo(), args, &[], GIT_DEADLINE);
    let verdict = judge_pin(pin, repo(), &run).and_then(|()| judge_last_touch(pin, generator, &run));
    if let Err(why) = verdict {
        panic!("tests/fixtures/{LAST_TOUCH_PINNED}/MANIFEST.json: {why}");
    }
}

/// (b) Every recorded `sha256` matches its file, every manifest that records none is
/// named in [`NO_FILE_SHA256`], and the table has no stale entry.
#[test]
fn every_recorded_file_checksum_matches_and_the_gap_is_named() {
    assert_eq!(
        ojas_io::sha256::sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "the hash the check relies on must be SHA-256"
    );
    let mut verified = 0;
    let mut without = Vec::new();
    for (name, dir, manifest) in manifests() {
        if records_any_sha256(&manifest) {
            if let Err(why) = verify_fixture_bytes(&dir, &manifest) {
                panic!("tests/fixtures/{name}: {why}");
            }
            verified += 1;
        } else {
            without.push(name);
        }
    }
    assert!(verified >= 1, "no manifest records a sha256; the check ran on nothing");
    let named: Vec<String> = NO_FILE_SHA256.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        without, named,
        "the manifests that record no sha256 must be exactly NO_FILE_SHA256 (a manifest that gained \
         checksums leaves the table; one that has none joins it on purpose)"
    );
}

/// The shallow verdict, and the whole check against a real `--depth 1` clone: the
/// clone's own HEAD resolves there, so only the shallow rule keeps it from passing.
#[test]
fn a_shallow_clone_is_not_run_and_never_passes() {
    assert!(shallow_verdict(0, "false").is_ok());
    let not_run = shallow_verdict(0, "true").expect_err("a shallow clone must not pass");
    assert!(
        not_run.starts_with("NOT RUN") && not_run.contains("shallow") && not_run.contains("fetch-depth: 0"),
        "{not_run}"
    );
    for (code, printed) in [(0, ""), (0, "--is-shallow-repository"), (128, "false"), (1, "true")] {
        let why = shallow_verdict(code, printed).expect_err("anything but a plain false must not pass");
        assert!(why.contains("could not verify provenance") || why.starts_with("NOT RUN"), "{why}");
    }

    let (code, head) = git_in(repo(), &["rev-parse", "HEAD"], &[], GIT_DEADLINE).expect("rev-parse HEAD");
    assert_eq!(code, 0, "rev-parse HEAD");
    let clone = Scratch::dir("tessl_shallow_clone");
    let url = format!("file://{}", repo().display());
    let (code, _) = git_in(
        &clone,
        &["clone", "--quiet", "--depth", "1", "--no-checkout", &url, "."],
        &[],
        Duration::from_secs(300),
    )
    .expect("clone runs");
    assert_eq!(code, 0, "git clone --depth 1 {url}");
    let run = |args: &[&str]| git_in(&clone, args, &[], GIT_DEADLINE);
    assert_eq!(
        run(&["rev-parse", "--is-shallow-repository"]).expect("runs"),
        (0, "true".to_string()),
        "the clone must really be shallow"
    );
    assert_eq!(
        run(&["cat-file", "-e", &format!("{head}^{{commit}}")]).expect("runs").0,
        0,
        "the pin must resolve in the shallow clone, so that only the shallow rule can fail it"
    );
    let why = judge_pin(&head, &clone, &run).expect_err("a shallow clone must never pass");
    assert!(why.starts_with("NOT RUN") && why.contains("fetch-depth: 0"), "{why}");
}

/// The verdict passes in one case, and every other outcome, including each check that
/// could not run, is an error.
#[test]
fn provenance_passes_only_for_an_ancestor_of_head_in_a_full_clone() {
    let pin = "a".repeat(40);
    let top = repo().to_string_lossy().into_owned();
    // A scripted git: `shallow`, `has_commit` and `ancestor` pick each answer.
    let scripted = |toplevel: String, shallow: &'static str, has_commit: i32, ancestor: i32| {
        move |args: &[&str]| -> GitOutput {
            Ok(match (args[0], args.get(1).copied()) {
                ("rev-parse", Some("--show-toplevel")) => (0, toplevel.clone()),
                ("rev-parse", _) => (0, shallow.to_string()),
                ("cat-file", _) => (has_commit, String::new()),
                ("merge-base", _) => (ancestor, String::new()),
                (other, _) => panic!("unexpected git {other}"),
            })
        }
    };
    let good = |shallow, has_commit, ancestor| scripted(top.clone(), shallow, has_commit, ancestor);
    judge_pin(&pin, repo(), &good("false", 0, 0)).expect("the one passing outcome");

    let fails = |run: &GitRun<'_>, needle: &str| {
        let e = judge_pin(&pin, repo(), run).expect_err("must not pass");
        assert!(e.contains(needle), "wanted {needle:?} in {e:?}");
    };
    fails(&good("false", 0, 1), "not on this history");
    fails(&good("false", 1, 0), "does not resolve to a commit");
    fails(&good("false", 128, 0), "does not resolve to a commit");
    fails(&good("true", 0, 0), "NOT RUN");
    fails(&good("", 0, 0), "could not verify provenance");
    fails(&good("false", 0, 128), "could not verify provenance");
    fails(&|_| Err("git is missing".into()), "could not verify provenance");
    fails(
        &|args| {
            if args.get(1) == Some(&"--is-shallow-repository") {
                Err("deadline".into())
            } else {
                Ok((0, top.clone()))
            }
        },
        "could not verify provenance",
    );
    // A git that landed in some other repository (or an empty directory) is not this one,
    // whatever it then says about the pin.
    let elsewhere = std::env::temp_dir().to_string_lossy().into_owned();
    fails(&scripted(elsewhere, "false", 0, 0), "not at the crate directory");
    fails(
        &scripted("/this/path/does/not/exist".into(), "false", 0, 0),
        "could not verify provenance",
    );

    // The last-touch rule: the pin, another commit, an empty answer, a failure.
    let other = "b".repeat(40);
    let last = |code: i32, out: String| move |_: &[&str]| -> GitOutput { Ok((code, out.clone())) };
    judge_last_touch(&pin, "g.py", &last(0, pin.clone())).expect("the pin is the last touch");
    let e = judge_last_touch(&pin, "g.py", &last(0, other)).expect_err("a later change must fail");
    assert!(e.contains("generator changed after the fixtures"), "{e}");
    for (code, out) in [(0, String::new()), (128, pin.clone())] {
        let e = judge_last_touch(&pin, "g.py", &last(code, out)).expect_err("must not pass");
        assert!(e.contains("could not verify provenance"), "{e}");
    }
    let e = judge_last_touch(&pin, "g.py", &|_| Err("git is missing".into())).expect_err("must not pass");
    assert!(e.contains("could not verify provenance"), "{e}");
}

#[test]
fn git_helper_ignores_repository_redirecting_environment() {
    let root = std::fs::canonicalize(repo()).expect("crate dir");
    let empty = Scratch::dir("tessl_fixture_empty");
    let empty_s = empty.to_string_lossy().into_owned();
    let missing = empty.join("missing").to_string_lossy().into_owned();
    let git_ok = |args: &[&str], inherited: &[(&str, &str)]| {
        let (code, out) = git_in(repo(), args, inherited, GIT_DEADLINE).expect("git runs");
        (code, out)
    };
    let toplevel = |inherited: &[(&str, &str)]| {
        let (code, top) = git_ok(&["rev-parse", "--show-toplevel"], inherited);
        assert_eq!(code, 0, "rev-parse failed under {inherited:?}: {top}");
        std::fs::canonicalize(top).expect("toplevel exists")
    };
    assert_eq!(toplevel(&[]), root, "baseline: git sees this crate's repository");
    let config_parameters = format!("'core.worktree'='{empty_s}'");
    let redirects: [(&str, &str); 14] = [
        ("GIT_DIR", &empty_s),
        ("GIT_WORK_TREE", &empty_s),
        ("GIT_INDEX_FILE", &missing),
        ("GIT_OBJECT_DIRECTORY", &missing),
        ("GIT_ALTERNATE_OBJECT_DIRECTORIES", &missing),
        ("GIT_COMMON_DIR", &empty_s),
        ("GIT_NAMESPACE", "elsewhere"),
        ("GIT_CEILING_DIRECTORIES", &empty_s),
        ("GIT_DISCOVERY_ACROSS_FILESYSTEM", "0"),
        ("GIT_REPLACE_REF_BASE", "refs/nothing/"),
        ("GIT_CONFIG_PARAMETERS", &config_parameters),
        ("GIT_CONFIG_COUNT", "1"),
        ("GIT_CONFIG_KEY_0", "core.worktree"),
        ("GIT_CONFIG_VALUE_0", &empty_s),
    ];
    for one in &redirects {
        assert_eq!(toplevel(&[*one]), root, "{} redirected the provenance check", one.0);
    }
    assert_eq!(toplevel(&redirects), root, "all of them together redirected it");
    // The provenance verdict itself is not fooled either: with a GIT_DIR in the
    // environment, the same helper still reaches the real repository and its commits.
    let (code, _) = git_ok(&["rev-parse", "--verify", "HEAD^{commit}"], &redirects);
    assert_eq!(code, 0, "HEAD must resolve in the real repository");

    // A graft file makes a history question answer differently: with HEAD grafted to have
    // no parents, its real parent stops being an ancestor. The helper must not see it.
    let (code, head) = git_ok(&["rev-parse", "HEAD"], &[]);
    assert_eq!(code, 0);
    let graft = Scratch::file("tessl_fixture_graft", ".txt");
    std::fs::write(&*graft, format!("{head}\n")).expect("graft file");
    let graft_s = graft.to_string_lossy().into_owned();
    let (code, _) = git_ok(
        &["merge-base", "--is-ancestor", "HEAD~1", "HEAD"],
        &[("GIT_GRAFT_FILE", &graft_s)],
    );
    assert_eq!(code, 0, "GIT_GRAFT_FILE changed which commits are ancestors");

    // Every GIT_* variable is gone from the child, and only the two set after the scrub
    // remain; the rest of the environment is untouched.
    let (code, env) = run_bounded(
        "sh",
        &["-c", "env"],
        &[("GIT_ANYTHING_AT_ALL", "1"), ("GIT_CONFIG_NOSYSTEM", "0")],
        Some(GIT_ENV_PREFIX),
        &GIT_ENV_AFTER_SCRUB,
        repo(),
        Duration::from_secs(20),
    )
    .expect("env runs");
    assert_eq!(code, 0);
    let mut git_vars: Vec<&str> = env.lines().filter(|l| l.starts_with("GIT_")).collect();
    git_vars.sort_unstable();
    assert_eq!(
        git_vars,
        ["GIT_CONFIG_GLOBAL=/dev/null", "GIT_CONFIG_NOSYSTEM=1"],
        "{env}"
    );
    assert!(
        env.lines().any(|l| l.starts_with("PATH=")),
        "the scrub must not clear the environment"
    );
}

#[test]
fn the_deadline_covers_reading_the_output_not_only_the_exit() {
    let run = |script: &str, deadline: Duration| {
        let started = Instant::now();
        let got = run_bounded("sh", &["-c", script], &[], None, &[], repo(), deadline);
        (got, started.elapsed())
    };
    let (got, _) = run("echo hi", Duration::from_secs(20));
    assert_eq!(got, Ok((0, "hi".to_string())));
    // The program outlives the deadline: killed, an error.
    let (got, took) = run("sleep 20", Duration::from_secs(1));
    assert!(got.is_err(), "{got:?}");
    assert!(took < Duration::from_secs(10), "took {took:?}");
    // The program exits at once but an orphan keeps its stdout open for 15 s: the call
    // must fail at the deadline, not wait for the pipe.
    let (got, took) = run("sleep 15 & echo hi", Duration::from_secs(2));
    let why = got.expect_err("an orphan-held pipe must not read as a success");
    assert!(why.contains("output pipe was still open"), "{why}");
    assert!(
        took < Duration::from_secs(10),
        "took {took:?}, the 2 s deadline did not bound the read"
    );
}

/// A copy of the dequant fixture directory to damage.
fn copy_fixture_dir() -> Scratch {
    let copy = Scratch::dir("tessl_dequant_fixtures");
    for entry in std::fs::read_dir(dequant_dir()).expect("fixture dir") {
        let entry = entry.expect("entry");
        std::fs::copy(entry.path(), copy.join(entry.file_name())).expect("copy");
    }
    copy
}

#[test]
fn the_manifest_pins_every_fixture_byte_and_every_file() {
    let pristine = copy_fixture_dir();
    let check = |dir: &Path| verify_fixture_bytes(dir, &read_manifest(dir).expect("manifest"));
    check(&pristine).expect("the committed fixtures match their manifest");

    let victim = "crafted_q8_g32_out.bin";
    let damaged = copy_fixture_dir();
    let mut bytes = std::fs::read(damaged.join(victim)).expect("read");
    bytes[0] ^= 1;
    std::fs::write(damaged.join(victim), bytes).expect("write");
    let why = check(&damaged).expect_err("a flipped bit must fail");
    assert!(why.contains(victim) && why.contains("sha256"), "{why}");

    let extra = copy_fixture_dir();
    std::fs::write(extra.join("stowaway.bin"), [0u8; 4]).expect("write");
    let why = check(&extra).expect_err("an unlisted file must fail");
    assert!(why.contains("stowaway.bin"), "{why}");

    let missing = copy_fixture_dir();
    std::fs::remove_file(missing.join(victim)).expect("remove");
    let why = check(&missing).expect_err("a listed file that is gone must fail");
    assert!(why.contains(victim), "{why}");

    let unrecorded = copy_fixture_dir();
    let text = std::fs::read_to_string(unrecorded.join("MANIFEST.json")).expect("manifest");
    assert!(text.contains("\"sha256\""));
    std::fs::write(
        unrecorded.join("MANIFEST.json"),
        text.replace("\"sha256\"", "\"sha_256\""),
    )
    .expect("write");
    let why = check(&unrecorded).expect_err("a manifest without checksums must fail");
    assert!(why.contains("no sha256"), "{why}");

    let traversal = copy_fixture_dir();
    let text = std::fs::read_to_string(traversal.join("MANIFEST.json")).expect("manifest");
    std::fs::write(traversal.join("MANIFEST.json"), text.replace(victim, "../x.bin")).expect("write");
    let why = check(&traversal).expect_err("a path in a file name must fail");
    assert!(why.contains("plain fixture file name"), "{why}");
}
