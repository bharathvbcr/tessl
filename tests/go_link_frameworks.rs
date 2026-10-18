//! The Go binding links tessl from a static archive (go/engine's libgusset.a),
//! and a static archive carries no record of the system frameworks it needs: cgo
//! names them itself, in `go/embedgemma2/engine.go`'s `#cgo darwin LDFLAGS`.
//! That list is written by hand, so a framework tessl starts to link (VideoToolbox
//! did, for exact video) breaks the Go build without touching any Rust test.
//! This holds the list to the sources: every framework a `#[link(name = "...",
//! kind = "framework")]` under `src/` names must be on the cgo line. No device.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let p = e.expect("readable entry").path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// Every `name` of a `link(name = "...", kind = "framework")` in `text`.
fn linked_frameworks(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for (i, _) in text.match_indices("link(name = \"") {
        let rest = &text[i + "link(name = \"".len()..];
        let Some(end) = rest.find('"') else { continue };
        if rest[end..].starts_with("\", kind = \"framework\")") {
            out.insert(rest[..end].to_string());
        }
    }
    out
}

/// The frameworks of the `#cgo darwin LDFLAGS:` line in `text`.
fn cgo_frameworks(text: &str) -> BTreeSet<String> {
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with("#cgo darwin LDFLAGS:"))
        .expect("engine.go has a #cgo darwin LDFLAGS line");
    let words: Vec<&str> = line.split_whitespace().collect();
    words
        .windows(2)
        .filter(|w| w[0] == "-framework")
        .map(|w| w[1].to_string())
        .collect()
}

#[test]
fn every_framework_tessl_links_is_on_the_go_link_line() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_files(&repo.join("src"), &mut files);
    let mut linked = BTreeSet::new();
    for f in &files {
        linked.extend(linked_frameworks(&std::fs::read_to_string(f).unwrap()));
    }
    assert!(
        linked.contains("VideoToolbox") && linked.contains("ImageIO"),
        "the scan found {linked:?}; it should see the media decoders' frameworks"
    );
    let engine = std::fs::read_to_string(repo.join("go/embedgemma2/engine.go")).unwrap();
    let cgo = cgo_frameworks(&engine);
    let missing: Vec<&String> = linked.difference(&cgo).collect();
    assert!(
        missing.is_empty(),
        "go/embedgemma2/engine.go's #cgo darwin LDFLAGS lacks {missing:?}: the Go binding \
         would not link (tessl's sources link {linked:?}; the cgo line names {cgo:?})"
    );
}

#[test]
fn the_scanners_read_what_they_should() {
    let rust = r#"
        #[link(name = "VideoToolbox", kind = "framework")] extern "C" {}
        #[link(name = "c++")] extern "C" {}
        #[link(name = "ImageIO", kind = "framework")]
    "#;
    let want: BTreeSet<String> = ["ImageIO", "VideoToolbox"].iter().map(|s| s.to_string()).collect();
    assert_eq!(linked_frameworks(rust), want);
    let go = "#cgo LDFLAGS: -lgusset\n#cgo darwin LDFLAGS: -framework ImageIO -framework Metal -lobjc\n";
    let cgo: BTreeSet<String> = ["ImageIO", "Metal"].iter().map(|s| s.to_string()).collect();
    assert_eq!(cgo_frameworks(go), cgo);
    assert_eq!(
        linked_frameworks(rust)
            .difference(&cgo_frameworks(go))
            .collect::<Vec<_>>(),
        ["VideoToolbox"],
        "a framework missing from the cgo line is found"
    );
}
