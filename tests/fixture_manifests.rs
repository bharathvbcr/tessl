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
//! record of its own, which must exist and parse), or in [`EXEMPT`] (not
//! generator-backed, with the reason). A new directory with none of the three
//! fails, so a fixture cannot arrive without its provenance.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

#[allow(dead_code)]
#[path = "../src/json.rs"]
mod json;

use json::{Json, Syntax};

/// Directories whose generator writes its own record, in its own format. Each file
/// must exist and parse; moving them onto `MANIFEST.json` is a change to their
/// generators (`scripts/gen_bert_fixtures.py`, `scripts/gen_embedgemma2_fixtures.py`)
/// and to the nanolab corpus manifest, not to this test.
const OWN_RECORD: [(&str, &str); 3] = [
    ("bert_tiny", "manifest.json"),
    ("embedgemma2", "provenance.json"),
    ("gdn", "gdn_published_MANIFEST.json"),
];

/// Directories with no generator to record. Nothing that has a generator belongs
/// here; each entry says why it is not generator-backed.
const EXEMPT: [(&str, &str); 0] = [];

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
    if m.get("schema")
        != Some(&Json::Num {
            value: 1.0,
            uint: Some(1),
        })
    {
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
    for (n, _) in OWN_RECORD.iter().chain(EXEMPT.iter()) {
        assert!(
            dirs.iter().any(|d| d == n),
            "the table names tests/fixtures/{n}, which does not exist"
        );
    }
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
