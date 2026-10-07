//! No kernel source spells an infinity in code.
//!
//! `build.rs` compiles every kernel with `-fmetal-math-mode=fast`. The IR that
//! produces marks every float compare `fast`, which includes `ninf`: the
//! compiler may assume no operand is infinite, so `m == -INFINITY` is a
//! comparison it is free to fold, and a running max seeded with `-INFINITY` is
//! a value it is free to assume away. Nothing measured has gone wrong, but
//! nothing obliges the GPU compiler to keep it that way.
//!
//! The safe forms are a finite identity (`-FLT_MAX`) with an explicit "has
//! seen a value" flag — `cross_entropy.metal`'s `first`, the attention
//! kernels' `l > 0` — or a seed taken from the data itself, as `reduce.metal`
//! and `qwen35_score.metal` do. This test keeps the infinities from coming
//! back: it strips comments and fails on any `INFINITY` token left.

use std::path::{Path, PathBuf};

/// `src` with `//` and `/* */` comments replaced by spaces, newlines kept so
/// line numbers still match the file.
fn strip_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
            i += 2;
            while i < b.len() && !(b[i] == b'*' && b.get(i + 1) == Some(&b'/')) {
                out.push(if b[i] == b'\n' { '\n' } else { ' ' });
                i += 1;
            }
            i += 2;
        } else {
            let ch = src[i..].chars().next().expect("index is on a char boundary");
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// Every `.metal` and `.h` file under `kernels/`, recursively.
fn kernel_sources(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("kernels directory is readable") {
            let path = entry.expect("readable directory entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "metal" || e == "h") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// Lines of `code` holding `INFINITY` as a whole identifier.
fn infinity_lines(code: &str) -> Vec<usize> {
    let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    code.lines()
        .enumerate()
        .filter(|(_, line)| {
            line.match_indices("INFINITY").any(|(at, m)| {
                let before = line[..at].chars().next_back();
                let after = line[at + m.len()..].chars().next();
                !before.is_some_and(is_ident) && !after.is_some_and(is_ident)
            })
        })
        .map(|(n, _)| n + 1)
        .collect()
}

#[test]
fn the_scanner_finds_code_and_skips_comments() {
    let src = "a = -INFINITY; // INFINITY\n/* INFINITY\n INFINITY */ b = MY_INFINITY;\nc = INFINITY_X;\n#define X \\\n    (INFINITY)\n";
    assert_eq!(infinity_lines(&strip_comments(src)), vec![1, 6]);
}

#[test]
fn no_kernel_spells_an_infinity_in_code() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
    let sources = kernel_sources(&root);
    assert!(
        sources.len() > 20,
        "found only {} kernel sources under {}",
        sources.len(),
        root.display()
    );
    let mut hits = Vec::new();
    for path in &sources {
        let src = std::fs::read_to_string(path).expect("kernel source is readable UTF-8");
        for line in infinity_lines(&strip_comments(&src)) {
            hits.push(format!("{}:{line}", path.strip_prefix(&root).unwrap_or(path).display()));
        }
    }
    assert!(
        hits.is_empty(),
        "{} INFINITY use(s) in kernel code, which fast math may fold away; seed with -FLT_MAX \
         and a `seen` flag, or from the data, instead: {hits:#?}",
        hits.len()
    );
}
