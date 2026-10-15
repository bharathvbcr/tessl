//! The static tile audit: every Rust `TileGeom` against the `SM`/`SN` compiled
//! into the kernel it dispatches. A mismatch makes the host launch the wrong
//! number of threadgroups and silently leave output tiles unwritten, so this
//! is the one correctness gate that needs no GPU: it reads `kernels/
//! matmul_tensorops.metal` and `src/` and compares text.
//!
//! The Rust side is read from the selectors themselves, not from a hand-kept
//! copy: the `(kernel, TILE_*)` arms of `coop_kernel`, `epi_kernel`, the
//! batched `match` in `gemm_batched` and `SplitKPar::kernel`, plus every
//! `pipeline("matmul2d_tensorops_...")` site under `src/` paired with the
//! `TILE_*` that follows it. An anchor that disappears fails the audit (a
//! selector it cannot find is a selector it did not check), and so does any
//! kernel entry point that no pair reaches.
//!
//! Each fault case below mutates one input and expects the audit to go red
//! with a named reason; `PASS` from a check that cannot print `FAIL` means
//! nothing (`GAP-TESSL-AUDIT-BKC-CHECK-DEAD` printed `PASS: 0 mismatch(es)`
//! while half of it examined nothing).

use std::fs;
use std::path::{Path, PathBuf};

/// How far below a `pipeline("name")` line to look for the `TileGeom` it is
/// dispatched with. 12 reaches the split-K and accumulate call sites, whose
/// tile argument sits 8-10 lines down; a literal site that pairs with none
/// fails the audit.
const PAIR_WINDOW: usize = 12;

/// Kernels this audit deliberately does not check, each with the reason. An
/// entry whose kernel no longer exists fails the audit, so this cannot rot
/// into a list of names that once meant something.
const EXEMPT: &[(&str, &str)] = &[];

/// Where a kernel's geometry came from, and the numbers: `(SM, SN, simdgroups)`
/// with `None` where the source does not state one.
type KernelGeom = (Option<usize>, Option<usize>, Option<usize>, String);

struct Tree {
    metal: String,
    gemm_rs: String,
    /// Every Rust file under `src/`, `(path relative to the crate, contents)`.
    /// `gemm.rs` is the entry whose path is `src/gemm.rs`.
    files: Vec<(String, String)>,
}

impl Tree {
    fn load() -> Tree {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files = Vec::new();
        collect_rs(root, &root.join("src"), &mut files);
        files.sort();
        let gemm_rs = files
            .iter()
            .find(|(p, _)| p == "src/gemm.rs")
            .expect("src/gemm.rs is part of the tree")
            .1
            .clone();
        Tree {
            metal: fs::read_to_string(root.join("kernels/matmul_tensorops.metal"))
                .expect("read matmul_tensorops.metal"),
            gemm_rs,
            files,
        }
    }

    /// Replace one occurrence in one file, so a fault that silently failed to
    /// inject cannot pass as a fault the audit missed.
    fn inject_rs(&self, file: &str, from: &str, to: &str) -> Tree {
        let mut t = Tree {
            metal: self.metal.clone(),
            gemm_rs: self.gemm_rs.clone(),
            files: self.files.clone(),
        };
        let entry = t.files.iter_mut().find(|(p, _)| p == file).expect("file in tree");
        entry.1 = inject(&entry.1, from, to);
        if file == "src/gemm.rs" {
            t.gemm_rs = entry.1.clone();
        }
        t
    }

    fn inject_metal(&self, from: &str, to: &str) -> Tree {
        let mut t = Tree {
            metal: inject(&self.metal, from, to),
            gemm_rs: self.gemm_rs.clone(),
            files: self.files.clone(),
        };
        t.files = self.files.clone();
        t
    }

    fn with_metal(&self, metal: String) -> Tree {
        Tree {
            metal,
            gemm_rs: self.gemm_rs.clone(),
            files: self.files.clone(),
        }
    }
}

fn collect_rs(root: &Path, dir: &Path, out: &mut Vec<(String, String)>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display())) {
        let path: PathBuf = entry.expect("dir entry").path();
        if path.is_dir() {
            collect_rs(root, &path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let rel = path.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
            out.push((
                rel,
                fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display())),
            ));
        }
    }
}

fn inject(src: &str, from: &str, to: &str) -> String {
    assert_eq!(src.matches(from).count(), 1, "injection site not unique: {from:?}");
    src.replacen(from, to, 1)
}

/// Comments carry example geometries that would otherwise match as code.
fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(at) = rest.find("/*") {
        out.push_str(&rest[..at]);
        match rest[at + 2..].find("*/") {
            Some(end) => rest = &rest[at + 2 + end + 2..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out.lines()
        .map(|l| l.find("//").map_or(l, |i| &l[..i]))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The `{...}` block that opens at or after `start`, or `None` if unbalanced.
fn braced_body(src: &str, start: usize) -> Option<&str> {
    let open = start + src[start..].find('{')?;
    let mut depth = 0usize;
    for (i, c) in src[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&src[open..open + i + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

fn is_word(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Byte offsets of `word` as a whole identifier.
fn word_positions<'a>(hay: &'a str, word: &'a str) -> impl Iterator<Item = usize> + 'a {
    hay.match_indices(word).map(|(i, _)| i).filter(move |&i| {
        !hay[..i].chars().next_back().is_some_and(is_word) && !hay[i + word.len()..].chars().next().is_some_and(is_word)
    })
}

/// A scanner over text, for the few fixed shapes the audit reads.
struct Cur<'a> {
    s: &'a str,
    at: usize,
}

impl<'a> Cur<'a> {
    fn new(s: &'a str, at: usize) -> Self {
        Cur { s, at }
    }
    fn ws(&mut self) {
        while self.s[self.at..].starts_with(|c: char| c.is_whitespace()) {
            self.at += self.s[self.at..].chars().next().unwrap().len_utf8();
        }
    }
    fn eat(&mut self, lit: &str) -> bool {
        self.ws();
        if self.s[self.at..].starts_with(lit) {
            self.at += lit.len();
            true
        } else {
            false
        }
    }
    fn take_while(&mut self, f: impl Fn(char) -> bool) -> &'a str {
        let rest = &self.s[self.at..];
        let n = rest.find(|c| !f(c)).unwrap_or(rest.len());
        self.at += n;
        &rest[..n]
    }
    fn ident(&mut self) -> Option<&'a str> {
        self.ws();
        let w = self.take_while(is_word);
        (!w.is_empty()).then_some(w)
    }
    fn num(&mut self) -> Option<usize> {
        self.ws();
        self.take_while(|c| c.is_ascii_digit()).parse().ok()
    }
}

/// `int NAME = N` inside a template parameter list.
fn int_default(params: &str, name: &str) -> Option<usize> {
    for at in word_positions(params, name) {
        let before = params[..at].trim_end();
        if before.ends_with("int") && !before[..before.len() - 3].ends_with(is_word) {
            let mut c = Cur::new(params, at + name.len());
            if c.eat("=") {
                return c.num();
            }
        }
    }
    None
}

/// Helpers whose template parameters default SM, SN and NSG. A kernel that
/// calls one with no template argument list runs that default geometry.
fn template_defaults(src: &str) -> Vec<(String, (usize, usize, usize))> {
    let mut out = Vec::new();
    for (at, _) in src.match_indices("template") {
        let mut c = Cur::new(src, at + "template".len());
        if !c.eat("<") {
            continue;
        }
        let params_start = c.at;
        let Some(close) = src[params_start..].find('>') else {
            continue;
        };
        let params = &src[params_start..params_start + close];
        c.at = params_start + close + 1;
        if !c.eat("inline") || c.ident().is_none() {
            continue;
        }
        let Some(name) = c.ident() else { continue };
        if !c.eat("(") {
            continue;
        }
        if let (Some(sm), Some(sn), Some(nsg)) = (
            int_default(params, "SM"),
            int_default(params, "SN"),
            int_default(params, "NSG"),
        ) {
            out.push((name.to_string(), (sm, sn, nsg)));
        }
    }
    out
}

fn constexpr_int(body: &str, name: &str) -> Option<usize> {
    let key = format!("constexpr int {name} = ");
    let at = body.find(&key)? + key.len();
    let digits: String = body[at..].chars().take_while(char::is_ascii_digit).collect();
    (body[at + digits.len()..].starts_with(';'))
        .then(|| digits.parse().ok())
        .flatten()
}

/// `{kernel: geometry}` for every entry point, plus the `*_KERNEL(...)` macro
/// invocations that could not be parsed. The geometry is `None` when it is not
/// a compile-time constant this parser can see.
fn kernel_tiles(metal: &str) -> (Vec<(String, KernelGeom)>, Vec<String>) {
    let src = strip_comments(metal);
    let mut out: Vec<(String, KernelGeom)> = Vec::new();
    let mut unparsed = Vec::new();
    let defaults = template_defaults(&src);

    let mut line_start = 0;
    for line in src.split_inclusive('\n') {
        let at = line_start;
        line_start += line.len();
        if let Some(rest) = line.strip_prefix("kernel void ") {
            let name: String = rest.chars().take_while(|&c| is_word(c)).collect();
            let after_name = rest[name.len()..].trim_start();
            if name.is_empty() || !after_name.starts_with('(') {
                continue;
            }
            let paren = at + "kernel void ".len() + name.len() + (rest[name.len()..].len() - after_name.len()) + 1;
            let body = braced_body(&src, paren).unwrap_or("");
            let sm = constexpr_int(body, "SM");
            let sn = constexpr_int(body, "SN");
            let sg = body.find("execution_simdgroups<").and_then(|i| {
                let d: String = body[i + "execution_simdgroups<".len()..]
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect();
                d.parse::<usize>().ok()
            });
            let one = body.contains("execution_simdgroup>");
            let geom = if sm.is_some() && sn.is_some() {
                (
                    sm,
                    sn,
                    if one && sg.is_none() { Some(1) } else { sg },
                    "constexpr".to_string(),
                )
            } else if let Some((h, d)) = defaults
                .iter()
                .find(|(h, _)| word_positions(body, h).any(|i| body[i + h.len()..].trim_start().starts_with('(')))
            {
                (Some(d.0), Some(d.1), Some(d.2), format!("{h} template defaults"))
            } else {
                (None, None, None, "no compile-time SM/SN".to_string())
            };
            out.push((name, geom));
            continue;
        }
        // Macro-stamped kernels: NAME(KERNEL, X, SM, SN, NSG, Y). Any `*_KERNEL(`
        // invocation that does not fit that shape is reported rather than
        // skipped, or a new macro would drop its kernels from the audit.
        let trimmed = line.trim_start_matches([' ', '\t']);
        let word: String = trimmed.chars().take_while(|&c| is_word(c)).collect();
        if !word.ends_with("_KERNEL") {
            continue;
        }
        let args_at = at + (line.len() - trimmed.len()) + word.len();
        let mut c = Cur::new(&src, args_at);
        if !c.eat("(") {
            continue;
        }
        let shape = (|| {
            let name = c.ident()?;
            c.eat(",").then_some(())?;
            c.ident()?;
            c.eat(",").then_some(())?;
            let sm = c.num()?;
            c.eat(",").then_some(())?;
            let sn = c.num()?;
            c.eat(",").then_some(())?;
            let nsg = c.num()?;
            c.eat(",").then_some(())?;
            c.ident()?;
            c.eat(")").then_some(())?;
            Some((name.to_string(), sm, sn, nsg))
        })();
        match shape {
            Some((name, sm, sn, nsg)) => out.push((name, (Some(sm), Some(sn), Some(nsg), word))),
            None => unparsed.push(format!("{word}: {}", trimmed.lines().next().unwrap_or("").trim())),
        }
    }
    (out, unparsed)
}

/// `TILE_*: TileGeom` constants from `gemm.rs`.
fn tile_consts(gemm_rs: &str) -> Vec<(String, (usize, usize, usize))> {
    let src = strip_comments(gemm_rs);
    let mut out = Vec::new();
    for at in word_positions(&src, "const") {
        let mut c = Cur::new(&src, at + "const".len());
        let Some(name) = c.ident().filter(|n| n.starts_with("TILE_")) else {
            continue;
        };
        if !(c.eat(":") && c.eat("TileGeom") && c.eat("=") && c.eat("TileGeom") && c.eat("{")) {
            continue;
        }
        let field = |c: &mut Cur, key: &str| -> Option<usize> {
            c.eat(key).then_some(())?;
            c.eat(":").then_some(())?;
            let n = c.num()?;
            c.eat(",");
            Some(n)
        };
        if let (Some(sm), Some(sn), Some(sg)) = (field(&mut c, "sm"), field(&mut c, "sn"), field(&mut c, "simdgroups"))
        {
            if c.eat("}") {
                out.push((name.to_string(), (sm, sn, sg)));
            }
        }
    }
    out
}

/// The first `TILE_*` identifier in `text`.
fn first_tile(text: &str) -> Option<&str> {
    let mut from = 0;
    while let Some(i) = text[from..].find("TILE_") {
        let at = from + i;
        if !text[..at].chars().next_back().is_some_and(is_word) {
            let n = text[at..].find(|c| !is_word(c)).unwrap_or(text.len() - at);
            return Some(&text[at..at + n]);
        }
        from = at + 5;
    }
    None
}

/// Every `"matmul2d_tensorops_*"` literal in `body`.
fn kernel_literals(body: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(i) = body[from..].find("\"matmul2d_tensorops_") {
        let start = from + i + 1;
        let Some(len) = body[start..].find('"') else { break };
        let name = &body[start..start + len];
        if name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        {
            out.push((start, name));
        }
        from = start + len + 1;
    }
    out
}

/// The selector bodies of `gemm.rs` whose arms pair a kernel literal with its
/// `TILE_*`, each by an anchor that must be found.
const ANCHORS: &[(&str, &str)] = &[
    ("coop_kernel", "fn coop_kernel("),
    ("epi_kernel", "fn epi_kernel("),
    (
        "the batched match in gemm_batched",
        "match (elem, coop_narrow(m, n, elem))",
    ),
    ("SplitKPar::kernel", "impl SplitKPar"),
];

struct Pair {
    kernel: String,
    tile: String,
    site: String,
}

struct Rust {
    consts: Vec<(String, (usize, usize, usize))>,
    pairs: Vec<Pair>,
    /// Literal `pipeline(...)` sites with no `TILE_*` within `PAIR_WINDOW`.
    unpaired: Vec<(String, String)>,
    /// Anchors this audit could not find in `gemm.rs`.
    missing_anchors: Vec<&'static str>,
}

fn rust_tiles(tree: &Tree) -> Rust {
    let consts = tile_consts(&tree.gemm_rs);
    let mut pairs = Vec::new();
    let mut missing_anchors = Vec::new();

    let gemm = strip_comments(&tree.gemm_rs);
    for (label, anchor) in ANCHORS {
        let Some(at) = gemm.find(anchor) else {
            missing_anchors.push(*label);
            continue;
        };
        let Some(body) = braced_body(&gemm, at) else {
            missing_anchors.push(*label);
            continue;
        };
        let before = pairs.len();
        for (lit_at, kernel) in kernel_literals(body) {
            if let Some(tile) = first_tile(&body[lit_at..]) {
                pairs.push(Pair {
                    kernel: kernel.to_string(),
                    tile: tile.to_string(),
                    site: format!("{label}"),
                });
            }
        }
        if pairs.len() == before {
            missing_anchors.push(*label);
        }
    }

    // `kernel_name_f32()` is the TensorOps arm's literal: resolve the call
    // through it, so the f32 NN dispatch is paired like any literal site.
    let f32_kernel = gemm
        .find("fn kernel_name_f32(")
        .and_then(|at| braced_body(&gemm, at))
        .and_then(|b| b.find("TensorOps => ").map(|i| &b[i..]))
        .and_then(|b| kernel_literals(b).first().map(|(_, n)| n.to_string()));

    let mut unpaired = Vec::new();
    for (rel, text) in &tree.files {
        let lines: Vec<String> = strip_comments(text).split('\n').map(str::to_string).collect();
        for (i, line) in lines.iter().enumerate() {
            let kernel = if let Some(at) = line.find("pipeline(\"matmul2d_tensorops_") {
                kernel_literals(&line[at..]).first().map(|(_, n)| n.to_string())
            } else if line.contains("pipeline(backend.kernel_name_f32())") {
                f32_kernel.clone()
            } else {
                continue;
            };
            let Some(kernel) = kernel else { continue };
            let window = lines[i..(i + PAIR_WINDOW).min(lines.len())].iter().enumerate();
            match window.filter_map(|(d, l)| first_tile(l).map(|t| (d, t))).next() {
                Some((d, tile)) => pairs.push(Pair {
                    kernel,
                    tile: tile.to_string(),
                    site: format!("{rel}:{}", i + d + 1),
                }),
                None => unpaired.push((kernel, format!("{rel}:{}", i + 1))),
            }
        }
    }
    Rust {
        consts,
        pairs,
        unpaired,
        missing_anchors,
    }
}

/// Why `kern` cannot be compared with Rust's `tile`, or `None` when it can.
/// Missing data is an error, never an unaudited row.
fn geometry_error(
    kernels: &[(String, KernelGeom)],
    consts: &[(String, (usize, usize, usize))],
    kern: &str,
    tile: &str,
) -> Option<String> {
    let Some((_, geom)) = kernels.iter().find(|(k, _)| k == kern) else {
        return Some("kernel not in matmul_tensorops.metal".into());
    };
    if !consts.iter().any(|(t, _)| t == tile) {
        return Some("no such TileGeom in src/gemm.rs".into());
    }
    if geom.0.is_none() {
        return Some(format!("{}: cannot check", geom.3));
    }
    None
}

struct Audit {
    ok: bool,
    report: String,
}

fn audit(tree: &Tree) -> Audit {
    let mut report = String::new();
    let mut bad = 0usize;
    let mut say = |bad_row: bool, line: String| {
        report.push_str(&line);
        report.push('\n');
        if bad_row {
            bad += 1;
        }
    };

    let (kt, unparsed) = kernel_tiles(&tree.metal);
    let rust = rust_tiles(tree);
    say(false, format!("    tile constants: {}", rust.consts.len()));
    say(false, format!("    kernels in matmul_tensorops.metal: {}", kt.len()));

    for label in &rust.missing_anchors {
        say(
            true,
            format!("  FAIL  anchor not found or paired no kernel in src/gemm.rs: {label}"),
        );
    }
    for u in &unparsed {
        say(true, format!("  FAIL  unparsed kernel macro {u}"));
    }
    // A literal dispatch with no TileGeom beside it launches a geometry this
    // audit cannot see. Its kernel may still be checked through another site,
    // which is exactly why the site itself has to fail.
    for (kern, site) in &rust.unpaired {
        say(
            true,
            format!("  FAIL  {kern:<44} no TILE_* within {PAIR_WINDOW} lines of pipeline() ({site})"),
        );
    }

    let mut checked: Vec<&str> = Vec::new();
    let mut n_pairs = 0usize;
    for p in &rust.pairs {
        if let Some(why) = geometry_error(&kt, &rust.consts, &p.kernel, &p.tile) {
            say(
                true,
                format!("  FAIL  {:<44} {:<18} {why} ({})", p.kernel, p.tile, p.site),
            );
            continue;
        }
        let kg = &kt.iter().find(|(k, _)| *k == p.kernel).unwrap().1;
        let rg = rust.consts.iter().find(|(t, _)| *t == p.tile).unwrap().1;
        let (ksm, ksn, ksg) = (kg.0.unwrap(), kg.1.unwrap(), kg.2);
        // A simdgroup count the kernel does not state is not compared.
        let ok = (ksm, ksn) == (rg.0, rg.1) && ksg.is_none_or(|g| g == rg.2);
        n_pairs += 1;
        if !checked.contains(&p.kernel.as_str()) {
            checked.push(&p.kernel);
        }
        say(
            !ok,
            format!(
                "  {}  {:<44} {:<18} kernel={ksm}x{ksn}/sg{ksg:?}  rust={}x{}/sg{}  ({})",
                if ok { "OK" } else { "MISMATCH" },
                p.kernel,
                p.tile,
                rg.0,
                rg.1,
                rg.2,
                p.site
            ),
        );
    }

    // Every entry point is checked or exempt. Kernels the host selects through
    // a variable never sit next to a pipeline("literal"), so without this they
    // would escape the audit entirely.
    let mut names: Vec<&str> = kt.iter().map(|(k, _)| k.as_str()).collect();
    names.sort_unstable();
    for kern in &names {
        match EXEMPT.iter().find(|(k, _)| k == kern) {
            Some((_, why)) if checked.contains(kern) => {
                say(true, format!("  FAIL  {kern:<44} both checked and EXEMPT ({why}); drop the exemption"))
            }
            Some((_, why)) => say(false, format!("  EXEMPT  {kern:<42} {why}")),
            None if !checked.contains(kern) => say(
                true,
                format!("  FAIL  {kern:<44} not checked: no Rust selector pairs it with a TILE_* (anchor it in ANCHORS or EXEMPT it with a reason)"),
            ),
            None => {}
        }
    }
    for (kern, _) in EXEMPT {
        if !names.contains(kern) {
            say(
                true,
                format!("  FAIL  {kern:<44} EXEMPT names a kernel that no longer exists"),
            );
        }
    }
    if kt.is_empty() || n_pairs == 0 {
        say(
            true,
            "  FAIL  examined nothing: no kernels parsed or no Rust dispatch pairs checked".into(),
        );
    }

    let coverage = format!(
        "tile geometry: {} kernels, {n_pairs} Rust dispatch pairs; {} exempt",
        checked.len(),
        EXEMPT.len()
    );
    report.push_str(&if bad > 0 {
        format!("\nFAIL: {bad} problem(s); {coverage}\n")
    } else {
        format!("\nPASS: {coverage}\n")
    });
    Audit { ok: bad == 0, report }
}

/// The arch_02 stale-copy guard. tessl-arch02 compiles these same sources
/// through `DEP_TESSL_KERNELS`, so it cannot drift by construction; a local
/// copy reappearing there is itself the regression. Skipped when the sibling
/// is absent, as it is inside a published crate.
fn stale_arch02_copy(crate_root: &Path) -> Option<PathBuf> {
    let stale = crate_root
        .ancestors()
        .nth(2)?
        .join("arch_02_value_resid/metal-native/kernels/matmul_tensorops.metal");
    stale.exists().then_some(stale)
}

#[test]
fn clean_tree_passes_and_names_what_it_checked() {
    let run = audit(&Tree::load());
    eprintln!("{}", run.report);
    assert!(run.ok, "the audit fails on the unmodified tree:\n{}", run.report);
    let last = run.report.lines().rev().find(|l| !l.is_empty()).unwrap();
    assert!(
        last.starts_with("PASS: tile geometry: ") && last.contains(" kernels"),
        "PASS line does not name what it checked: {last:?}"
    );
}

#[test]
fn tessl_arch02_has_no_matmul_tensorops_copy_of_its_own() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    if let Some(stale) = stale_arch02_copy(root) {
        panic!(
            "tessl-arch02 has its own {} again. It must compile tessl's copy via DEP_TESSL_KERNELS; a local one silently drifts.",
            stale.display()
        );
    }
}

#[test]
fn the_arch02_guard_sees_a_planted_copy() {
    let base = Path::new(env!("CARGO_TARGET_TMPDIR")).join("arch02_guard");
    let _ = fs::remove_dir_all(&base);
    let crate_root = base.join("a/b/tessl");
    fs::create_dir_all(&crate_root).unwrap();
    assert!(stale_arch02_copy(&crate_root).is_none());
    let planted = base.join("a/arch_02_value_resid/metal-native/kernels");
    fs::create_dir_all(&planted).unwrap();
    fs::write(planted.join("matmul_tensorops.metal"), "").unwrap();
    assert!(stale_arch02_copy(&crate_root).is_some(), "a planted copy went unseen");
}

#[test]
fn tile_drift_fails() {
    let tree = Tree::load().inject_rs(
        "src/gemm.rs",
        "const TILE_COOP_NARROW: TileGeom = TileGeom {\n    sm: 64,",
        "const TILE_COOP_NARROW: TileGeom = TileGeom {\n    sm: 32,",
    );
    let run = audit(&tree);
    assert!(!run.ok);
    assert!(run.report.contains("MISMATCH"));
}

/// The selectors themselves are read, so a wrong tile in an arm of
/// `coop_kernel` is a MISMATCH on that arm's kernel. The hand-kept pin list
/// this replaced would have kept passing on its own copy.
#[test]
fn a_mismatched_coop_kernel_arm_fails() {
    let tree = Tree::load().inject_rs(
        "src/gemm.rs",
        "                CoopElem::RelaxedF32 => \"matmul2d_tensorops_f32_relaxed_64x64_sg4\",\n            },\n            TILE_COOP_NARROW,",
        "                CoopElem::RelaxedF32 => \"matmul2d_tensorops_f32_relaxed_64x64_sg4\",\n            },\n            TILE_COOP_DEFAULT,",
    );
    let run = audit(&tree);
    assert!(!run.ok);
    assert!(
        run.report.contains("MISMATCH  matmul2d_tensorops_bf16_f32_64x64_sg4"),
        "{}",
        run.report
    );
}

#[test]
fn a_mismatched_epi_kernel_arm_fails() {
    let tree = Tree::load().inject_rs(
        "src/gemm.rs",
        "(\"matmul2d_tensorops_f16_f32_epi\", TILE_COOP_DEFAULT)",
        "(\"matmul2d_tensorops_f16_f32_epi\", TILE_COOP_NARROW)",
    );
    let run = audit(&tree);
    assert!(!run.ok);
    assert!(
        run.report.contains("MISMATCH  matmul2d_tensorops_f16_f32_epi "),
        "{}",
        run.report
    );
}

#[test]
fn a_mismatched_batched_arm_fails() {
    let tree = Tree::load().inject_rs(
        "src/gemm.rs",
        "(CoopElem::Bf16, true) => (\"matmul2d_tensorops_bf16_f32_batched_64x64_sg4\", TILE_COOP_NARROW)",
        "(CoopElem::Bf16, true) => (\"matmul2d_tensorops_bf16_f32_batched_64x64_sg4\", TILE_COOP_DEFAULT)",
    );
    let run = audit(&tree);
    assert!(!run.ok);
    assert!(
        run.report
            .contains("MISMATCH  matmul2d_tensorops_bf16_f32_batched_64x64_sg4"),
        "{}",
        run.report
    );
}

#[test]
fn a_mismatched_splitk_par_arm_fails() {
    let tree = Tree::load().inject_rs(
        "src/gemm.rs",
        "Self::Bf16 => (\"matmul2d_tensorops_tn_splitk_par_bf16_f32\", TILE_COOP_SPLITK)",
        "Self::Bf16 => (\"matmul2d_tensorops_tn_splitk_par_bf16_f32\", TILE_F32)",
    );
    let run = audit(&tree);
    assert!(!run.ok);
    assert!(
        run.report
            .contains("MISMATCH  matmul2d_tensorops_tn_splitk_par_bf16_f32"),
        "{}",
        run.report
    );
}

/// A selector the audit cannot find is a selector it did not check.
#[test]
fn a_renamed_selector_fails_instead_of_dropping_out() {
    let tree = Tree::load().inject_rs("src/gemm.rs", "fn coop_kernel(", "fn coop_kernel_renamed(");
    let run = audit(&tree);
    assert!(!run.ok);
    assert!(run.report.contains("anchor not found"), "{}", run.report);
    assert!(run.report.contains("coop_kernel"), "{}", run.report);
}

/// `matmul2d_tensorops_bf16_f32` is the production NN kernel and takes its
/// geometry from `mm_nn_coop_f32acc`'s template defaults, not from a
/// `constexpr` or a macro argument. It was once skipped as "no compile-time
/// SM/SN" — and a skipped kernel counted as passed.
#[test]
fn template_default_drift_fails() {
    let tree = Tree::load().inject_metal(
        "template <typename ElemT, int SM = 128, int SN = 64, int NSG = 4,",
        "template <typename ElemT, int SM = 256, int SN = 64, int NSG = 4,",
    );
    let run = audit(&tree);
    assert!(!run.ok);
    assert!(run.report.contains("matmul2d_tensorops_bf16_f32 "), "{}", run.report);
}

/// A kernel the host dispatches through a variable never appears next to a
/// `pipeline("literal")`, so nothing checks its tile unless something insists
/// every entry point is accounted for.
#[test]
fn unaccounted_kernel_fails() {
    let base = Tree::load();
    let tree = base.with_metal(format!(
        "{}\nNN_COOP_KERNEL(matmul2d_tensorops_audit_probe, bfloat, 64, 64, 4, false)\n",
        base.metal
    ));
    let run = audit(&tree);
    assert!(!run.ok);
    assert!(run.report.contains("matmul2d_tensorops_audit_probe"));
}

/// A new cooperative macro whose arguments the parser does not understand
/// must stop the audit, not drop its kernels from it.
#[test]
fn unparsed_kernel_macro_fails() {
    let base = Tree::load();
    let tree = base.with_metal(format!(
        "{}\nNN_COOP_SPLIT_KERNEL(matmul2d_tensorops_audit_probe, bfloat, 64, 64)\n",
        base.metal
    ));
    let run = audit(&tree);
    assert!(!run.ok);
    assert!(run.report.contains("NN_COOP_SPLIT_KERNEL"));
}

/// Nothing to examine is a failure, never `PASS: 0 mismatch(es)`.
#[test]
fn no_kernels_fails() {
    let run = audit(&Tree::load().with_metal(String::new()));
    assert!(!run.ok);
    assert!(run.report.contains("examined nothing"));
}

#[test]
fn no_rust_dispatch_fails() {
    let mut tree = Tree::load();
    tree.gemm_rs.clear();
    tree.files.clear();
    let run = audit(&tree);
    assert!(!run.ok);
    assert!(run.report.contains("examined nothing"));
}

/// `nn::gemm_i8_dequant` dispatches from src/nn.rs: a wrong tile at that site
/// has to be a `MISMATCH` like one in gemm.rs.
#[test]
fn tile_drift_outside_gemm_rs_fails() {
    let tree = Tree::load().inject_rs(
        "src/nn.rs",
        "let tile = crate::gemm::TILE_COOP_DEFAULT;",
        "let tile = crate::gemm::TILE_COOP_NARROW;",
    );
    let run = audit(&tree);
    assert!(!run.ok);
    assert!(
        run.report.contains("MISMATCH  matmul2d_tensorops_i8_f32"),
        "{}",
        run.report
    );
}

/// A literal dispatch with no TileGeom near it launches a geometry the audit
/// cannot see, so the site fails even though nothing mismatched.
#[test]
fn dispatch_without_a_tile_geom_fails() {
    let tree = Tree::load().inject_rs(
        "src/nn.rs",
        "let tile = crate::gemm::TILE_COOP_DEFAULT;",
        "let tile = local_geometry();",
    );
    let run = audit(&tree);
    assert!(!run.ok);
    assert!(run.report.contains("no TILE_* within"), "{}", run.report);
    assert!(run.report.contains("src/nn.rs:"), "{}", run.report);
}

/// An EXEMPT entry for a kernel that does not exist must fail rather than rot.
#[test]
fn geometry_error_rejects_missing_or_unverifiable_data() {
    let kernels = vec![
        ("ok".to_string(), (Some(32), Some(64), Some(4), "constexpr".to_string())),
        (
            "blind".to_string(),
            (None, None, None, "no compile-time SM/SN".to_string()),
        ),
    ];
    let consts = vec![("TILE_OK".to_string(), (32, 64, 4))];
    assert!(geometry_error(&kernels, &consts, "ok", "TILE_OK").is_none());
    assert!(geometry_error(&kernels, &consts, "missing", "TILE_OK").is_some());
    assert!(geometry_error(&kernels, &consts, "ok", "TILE_MISSING").is_some());
    assert!(geometry_error(&kernels, &consts, "blind", "TILE_OK").is_some());
}
