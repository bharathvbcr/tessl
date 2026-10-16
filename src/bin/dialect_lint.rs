//! Flag MSL constructs in the Qwen3.5 kernels that no compiling kernel uses.
//!
//! ```text
//! dialect_lint [--reviewed NAME=REASON]... [ROOT]
//! ```
//!
//! The emulator compiles the kernels as C++, so it cannot see a construct
//! Apple's Metal compiler rejects. The next best evidence without a Mac: every
//! function, qualified name, attribute, cast and language feature the Qwen3.5
//! kernels use is either already used by one of tessl's other kernels (which
//! compile on every release build) or listed in `REVIEWED` below with the
//! reason it is standard MSL. A new construct that is neither fails this
//! check, so it gets a deliberate look before it gets a compiler.
//!
//! Exit status: 1 for a construct that is new and unreviewed, and equally for
//! a `REVIEWED` entry that is no longer new (another kernel now uses it, or
//! the Qwen3.5 sources dropped it). A stale allowlist entry is an unchecked
//! claim, so it fails rather than printing a note: delete the entry. It also
//! fails when nothing is new at all, because then the Qwen3.5 sources were
//! never examined. ROOT defaults to the current directory;
//! `tests/dialect_lint_script.rs` passes a planted tree and uses
//! `--reviewed` to plant a stale entry.
//!
//! Standalone on purpose: standard library only, no `tessl` import. The
//! kernel-emulator CI job runs on Linux, where the crate does not build, and
//! compiles this file with a bare `rustc --edition 2021`.
//!
//! The scanning is a port of a Python script that used `re`. Its patterns are
//! kept as the same text and run by the small backtracking matcher below
//! (classes, groups, alternation, greedy and lazy `*` `+` `?`, `^` under
//! multi-line, `\b`, and a one-character negative look-behind: everything the
//! patterns use and nothing more), so the leftmost-first, backtracking
//! behaviour of the originals, which the run-on cases of `defined_here`
//! depend on, is the same. `\w` is alphanumeric-or-underscore as Rust's
//! `char::is_alphanumeric` has it, `\s` is Python's `str.isspace`, and `\d` is
//! ASCII: the kernels are ASCII source, and these differ from Python's tables
//! only on exotic code points.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// The sources under test: the Qwen3.5 kernels.
const NEW: [&str; 4] = ["qwen35_gdn", "qwen35_attn", "qwen35_mlp", "qwen35_score"];

/// Constructs the Qwen3.5 kernels introduce, each checked against the Metal
/// Shading Language specification when it was added.
const REVIEWED: [(&str, &str); 3] = [
    ("log", "MSL math function (fast variant under the default fast math)"),
    (
        "mem_flags::mem_device",
        "threadgroup_barrier flag ordering device memory across the threadgroup",
    ),
    (
        "uint3",
        "vector type for the 3-D [[threadgroup_position_in_grid]] builtin",
    ),
];

const KEYWORDS: [&str; 10] = [
    "if", "for", "while", "return", "switch", "sizeof", "defined", "kernel", "void", "else",
];
const FEATURES: [&str; 10] = [
    "static_assert",
    "constexpr",
    "template",
    "#define",
    "uint3",
    "uint2",
    "ulong2",
    "float4",
    "bfloat",
    "bool",
];

// ---------------------------------------------------------------------------
// The matcher.
// ---------------------------------------------------------------------------

/// Python's `str.isspace`: Unicode white space plus `\x1c`..`\x1f`.
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

#[derive(Clone, Debug)]
enum Item {
    Ch(char),
    Range(char, char),
    Word,
    Space,
    Digit,
}

/// One character set: `[...]`, `[^...]`, `\w`, `\s`, `\d` or a literal.
#[derive(Clone, Debug)]
struct Class {
    items: Vec<Item>,
    negate: bool,
}

impl Class {
    fn matches(&self, c: char) -> bool {
        let hit = self.items.iter().any(|i| match *i {
            Item::Ch(x) => c == x,
            Item::Range(a, b) => a <= c && c <= b,
            Item::Word => is_word(c),
            Item::Space => is_space(c),
            Item::Digit => c.is_ascii_digit(),
        });
        hit != self.negate
    }
}

#[derive(Debug)]
enum Node {
    Set(Class),
    Seq(Vec<Node>),
    Alt(Vec<Node>),
    Group(usize, Box<Node>),
    Rep {
        node: Box<Node>,
        min_one: bool,
        max_one: bool,
        greedy: bool,
    },
    Bol,
    WordBoundary,
    NotBehind(Class),
}

#[derive(Debug)]
enum Inst {
    Set(Class),
    Split(usize, usize),
    Jmp(usize),
    Save(usize),
    Bol,
    WordBoundary,
    NotBehind(Class),
    Match,
}

/// A compiled pattern.
pub struct Regex {
    prog: Vec<Inst>,
    groups: usize,
}

/// One match: the span of the whole match and of each capture group.
pub struct Match {
    spans: Vec<Option<(usize, usize)>>,
}

impl Match {
    pub fn start(&self) -> usize {
        self.spans[0].map_or(0, |s| s.0)
    }
    pub fn end(&self) -> usize {
        self.spans[0].map_or(0, |s| s.1)
    }
    /// Group `i` (0 is the whole match) as text.
    pub fn text(&self, chars: &[char], i: usize) -> Option<String> {
        self.spans
            .get(i)
            .copied()
            .flatten()
            .map(|(a, b)| chars[a..b].iter().collect())
    }
}

struct Parser<'a> {
    p: &'a [char],
    i: usize,
    groups: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.p.get(self.i).copied()
    }

    fn alt(&mut self) -> Result<Node, String> {
        let mut arms = vec![self.seq()?];
        while self.peek() == Some('|') {
            self.i += 1;
            arms.push(self.seq()?);
        }
        Ok(if arms.len() == 1 {
            arms.remove(0)
        } else {
            Node::Alt(arms)
        })
    }

    fn seq(&mut self) -> Result<Node, String> {
        let mut items = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let atom = self.atom()?;
            items.push(self.quantified(atom)?);
        }
        Ok(Node::Seq(items))
    }

    fn quantified(&mut self, atom: Node) -> Result<Node, String> {
        let (min_one, max_one) = match self.peek() {
            Some('*') => (false, false),
            Some('+') => (true, false),
            Some('?') => (false, true),
            _ => return Ok(atom),
        };
        self.i += 1;
        let greedy = if self.peek() == Some('?') {
            self.i += 1;
            false
        } else {
            true
        };
        Ok(Node::Rep {
            node: Box::new(atom),
            min_one,
            max_one,
            greedy,
        })
    }

    fn escape_item(&mut self) -> Result<Item, String> {
        let c = self.peek().ok_or("dangling backslash")?;
        self.i += 1;
        Ok(match c {
            'w' => Item::Word,
            's' => Item::Space,
            'd' => Item::Digit,
            'n' => Item::Ch('\n'),
            other => Item::Ch(other),
        })
    }

    fn class(&mut self) -> Result<Class, String> {
        let negate = self.peek() == Some('^');
        if negate {
            self.i += 1;
        }
        let mut items = Vec::new();
        loop {
            let c = self.peek().ok_or("unterminated character class")?;
            self.i += 1;
            let item = match c {
                ']' if !items.is_empty() => break,
                '\\' => self.escape_item()?,
                other => Item::Ch(other),
            };
            if let (Item::Ch(lo), Some('-'), Some(&hi)) = (&item, self.peek(), self.p.get(self.i + 1)) {
                if hi != ']' {
                    self.i += 2;
                    items.push(Item::Range(*lo, hi));
                    continue;
                }
            }
            items.push(item);
        }
        Ok(Class { items, negate })
    }

    fn atom(&mut self) -> Result<Node, String> {
        let c = self.peek().ok_or("unexpected end of pattern")?;
        self.i += 1;
        match c {
            '(' => {
                let node = if self.p[self.i..].starts_with(&['?', ':']) {
                    self.i += 2;
                    self.alt()?
                } else if self.p[self.i..].starts_with(&['?', '<', '!']) {
                    self.i += 3;
                    self.expect('[')?;
                    let class = self.class()?;
                    self.expect(')')?;
                    return Ok(Node::NotBehind(class));
                } else {
                    self.groups += 1;
                    let idx = self.groups;
                    Node::Group(idx, Box::new(self.alt()?))
                };
                self.expect(')')?;
                Ok(node)
            }
            '[' => Ok(Node::Set(self.class()?)),
            '\\' => match self.peek() {
                Some('b') => {
                    self.i += 1;
                    Ok(Node::WordBoundary)
                }
                _ => Ok(Node::Set(Class {
                    items: vec![self.escape_item()?],
                    negate: false,
                })),
            },
            '^' => Ok(Node::Bol),
            other => Ok(Node::Set(Class {
                items: vec![Item::Ch(other)],
                negate: false,
            })),
        }
    }

    fn expect(&mut self, c: char) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected {c:?} at {}", self.i))
        }
    }
}

fn emit(node: &Node, prog: &mut Vec<Inst>) {
    match node {
        Node::Set(c) => prog.push(Inst::Set(c.clone())),
        Node::Seq(items) => items.iter().for_each(|n| emit(n, prog)),
        Node::Alt(arms) => {
            let mut jumps = Vec::new();
            for (k, arm) in arms.iter().enumerate() {
                if k + 1 < arms.len() {
                    let split = prog.len();
                    prog.push(Inst::Split(split + 1, 0));
                    emit(arm, prog);
                    jumps.push(prog.len());
                    prog.push(Inst::Jmp(0));
                    let next = prog.len();
                    prog[split] = Inst::Split(split + 1, next);
                } else {
                    emit(arm, prog);
                }
            }
            let end = prog.len();
            for j in jumps {
                prog[j] = Inst::Jmp(end);
            }
        }
        Node::Group(idx, inner) => {
            prog.push(Inst::Save(2 * idx));
            emit(inner, prog);
            prog.push(Inst::Save(2 * idx + 1));
        }
        Node::Rep {
            node,
            min_one,
            max_one,
            greedy,
        } => {
            // x?   : split(body, end); body
            // x*   : L: split(body, end); body; jmp L
            // x+   : L: body; split(L, end)
            let prefer = |body: usize, end: usize| {
                if *greedy {
                    Inst::Split(body, end)
                } else {
                    Inst::Split(end, body)
                }
            };
            if *max_one {
                let split = prog.len();
                prog.push(Inst::Match);
                emit(node, prog);
                let end = prog.len();
                prog[split] = prefer(split + 1, end);
            } else if *min_one {
                let top = prog.len();
                emit(node, prog);
                let split = prog.len();
                prog.push(Inst::Match);
                prog[split] = prefer(top, split + 1);
            } else {
                let split = prog.len();
                prog.push(Inst::Match);
                emit(node, prog);
                prog.push(Inst::Jmp(split));
                let end = prog.len();
                prog[split] = prefer(split + 1, end);
            }
        }
        Node::Bol => prog.push(Inst::Bol),
        Node::WordBoundary => prog.push(Inst::WordBoundary),
        Node::NotBehind(c) => prog.push(Inst::NotBehind(c.clone())),
    }
}

enum Frame {
    Branch(usize, usize),
    Restore(usize, Option<usize>),
}

impl Regex {
    /// Compile `pattern`; the patterns are compile-time constants, so a bad one
    /// is a bug in this file and panics.
    pub fn new(pattern: &str) -> Regex {
        let chars: Vec<char> = pattern.chars().collect();
        let mut parser = Parser {
            p: &chars,
            i: 0,
            groups: 0,
        };
        let ast = parser.alt().unwrap_or_else(|e| panic!("bad pattern {pattern:?}: {e}"));
        assert!(parser.i == chars.len(), "bad pattern {pattern:?}: unbalanced ')'");
        let mut prog = vec![Inst::Save(0)];
        emit(&ast, &mut prog);
        prog.push(Inst::Save(1));
        prog.push(Inst::Match);
        Regex {
            prog,
            groups: parser.groups,
        }
    }

    /// The match that starts exactly at `start`, with backtracking.
    fn match_at(&self, text: &[char], start: usize) -> Option<Match> {
        let mut caps: Vec<Option<usize>> = vec![None; 2 * (self.groups + 1)];
        let mut stack: Vec<Frame> = Vec::new();
        let (mut pc, mut pos) = (0usize, start);
        loop {
            let advance = match &self.prog[pc] {
                Inst::Match => {
                    let spans = (0..=self.groups).map(|g| caps[2 * g].zip(caps[2 * g + 1])).collect();
                    return Some(Match { spans });
                }
                Inst::Set(c) => match text.get(pos) {
                    Some(&ch) if c.matches(ch) => {
                        pos += 1;
                        true
                    }
                    _ => false,
                },
                Inst::Split(first, second) => {
                    stack.push(Frame::Branch(*second, pos));
                    pc = *first;
                    continue;
                }
                Inst::Jmp(to) => {
                    pc = *to;
                    continue;
                }
                Inst::Save(slot) => {
                    stack.push(Frame::Restore(*slot, caps[*slot]));
                    caps[*slot] = Some(pos);
                    true
                }
                Inst::Bol => pos == 0 || text[pos - 1] == '\n',
                Inst::WordBoundary => {
                    let before = pos > 0 && is_word(text[pos - 1]);
                    let after = text.get(pos).is_some_and(|&c| is_word(c));
                    before != after
                }
                Inst::NotBehind(c) => pos == 0 || !c.matches(text[pos - 1]),
            };
            if advance {
                pc += 1;
                continue;
            }
            // Backtrack to the most recent alternative.
            loop {
                match stack.pop() {
                    None => return None,
                    Some(Frame::Restore(slot, old)) => caps[slot] = old,
                    Some(Frame::Branch(to, at)) => {
                        pc = to;
                        pos = at;
                        break;
                    }
                }
            }
        }
    }

    /// `re.finditer`: leftmost, non-overlapping matches, each search resuming at
    /// the end of the last one.
    pub fn find_iter(&self, text: &[char]) -> Vec<Match> {
        let mut out = Vec::new();
        let mut from = 0;
        while from <= text.len() {
            let Some(m) = (from..=text.len()).find_map(|s| self.match_at(text, s)) else {
                break;
            };
            from = if m.end() > m.start() { m.end() } else { m.end() + 1 };
            out.push(m);
        }
        out
    }

    /// `re.search(...) is not None`.
    pub fn is_match(&self, text: &[char]) -> bool {
        (0..=text.len()).any(|s| self.match_at(text, s).is_some())
    }
}

// ---------------------------------------------------------------------------
// The lint.
// ---------------------------------------------------------------------------

/// Drop `//` comments to the end of their line, then `/* */` comments. In that
/// order: a `//` inside a block comment is cut first and can take the `*/`
/// with it.
fn strip(src: &str) -> String {
    let mut no_line = String::with_capacity(src.len());
    for (k, line) in src.split('\n').enumerate() {
        if k > 0 {
            no_line.push('\n');
        }
        no_line.push_str(line.find("//").map_or(line, |at| &line[..at]));
    }
    let mut out = String::with_capacity(no_line.len());
    let mut rest = no_line.as_str();
    while let Some(open) = rest.find("/*") {
        // `/\*.*?\*/`: the closer is searched after the two characters of the opener.
        match rest[open + 2..].find("*/") {
            Some(close) => {
                out.push_str(&rest[..open]);
                rest = &rest[open + 2 + close + 2..];
            }
            None => break,
        }
    }
    out.push_str(rest);
    out
}

fn find_text(re: &Regex, chars: &[char], group: usize) -> BTreeSet<String> {
    re.find_iter(chars)
        .iter()
        .filter_map(|m| m.text(chars, group))
        .collect()
}

/// `re.sub(r"\(\d+\)", "(n)", s)`.
fn number_args(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '(' {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 1 && chars.get(j) == Some(&')') {
                out.push_str("(n)");
                i = j + 1;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// `re.sub(r"\s+", " ", s)`.
fn squeeze_space(s: &str) -> String {
    let mut out = String::new();
    let mut in_space = false;
    for c in s.chars() {
        if is_space(c) {
            if !in_space {
                out.push(' ');
            }
            in_space = true;
        } else {
            out.push(c);
            in_space = false;
        }
    }
    out
}

/// Every construct a source uses: calls, qualified names, attributes, pointer
/// casts and language features.
pub fn constructs(src: &str) -> BTreeSet<String> {
    let chars: Vec<char> = strip(src).chars().collect();
    let mut found = BTreeSet::new();
    found.extend(find_text(&Regex::new(r"\b([A-Za-z_]\w*(?:::\w+)*)\s*\("), &chars, 1));
    found.extend(find_text(
        &Regex::new(r"\b([A-Za-z_]\w*)\s*<[\w\s,:]*>\s*\("),
        &chars,
        1,
    ));
    found.extend(find_text(&Regex::new(r"\b\w+::\w+\b"), &chars, 0));
    found.extend(
        find_text(&Regex::new(r"\[\[([^\]]+)\]\]"), &chars, 1)
            .iter()
            .map(|a| format!("[[{}]]", number_args(a))),
    );
    found.extend(
        find_text(
            &Regex::new(r"\((?:device|threadgroup|constant)\s+(?:const\s+)?\w+\s*\*\)"),
            &chars,
            0,
        )
        .iter()
        .map(|c| format!("cast {}", squeeze_space(c))),
    );
    for f in FEATURES {
        if Regex::new(&format!(r"(?<![\w#]){f}\b")).is_match(&chars) {
            found.insert(f.to_string());
        }
    }
    for k in KEYWORDS {
        found.remove(k);
    }
    found
}

/// Helpers, kernels and macros the new sources define themselves.
pub fn defined_here(src: &str) -> BTreeSet<String> {
    let chars: Vec<char> = strip(src).chars().collect();
    let mut names = BTreeSet::new();
    names.extend(find_text(
        &Regex::new(r"^\s*(?:inline\s+)?[\w<>\s]*?\b(\w+)\s*\([^;{]*\)\s*\{"),
        &chars,
        1,
    ));
    // Kernels by name as well: the signature pattern above can start at a
    // macro instantiation with no trailing semicolon and run on through the
    // next kernel's parameter list, which then never gets its own match.
    names.extend(find_text(&Regex::new(r"\bkernel\s+void\s+(\w+)\s*\("), &chars, 1));
    // Inline helpers by name too, for the same reason: a helper defined right
    // after a macro instantiation is inside that run-on match.
    names.extend(find_text(&Regex::new(r"\binline\s+[\w<>:\s]*?\b(\w+)\s*\("), &chars, 1));
    names.extend(find_text(&Regex::new(r"^#define\s+(\w+)"), &chars, 1));
    names.extend(find_text(&Regex::new(r"^(\w+)\(\w+,"), &chars, 1)); // macro instantiations
                                                                      // A function-like macro's parameters (`NAME`, a kernel macro's `ACT`) are
                                                                      // its own names, called or not; what they expand to is checked where the
                                                                      // macro is instantiated.
    for params in find_text(&Regex::new(r"^#define\s+\w+\(([^)]*)\)"), &chars, 1) {
        for p in params.split(',') {
            let p = p.trim_matches(is_space);
            if !p.is_empty() {
                names.insert(p.to_string());
            }
        }
    }
    names
}

/// `open(path).read()`: UTF-8, with `\r\n` and `\r` read as `\n`.
fn read_source(path: &Path) -> Result<String, String> {
    let raw = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let text = String::from_utf8(raw).map_err(|e| format!("{} is not UTF-8: {e}", path.display()))?;
    Ok(text.replace("\r\n", "\n").replace('\r', "\n"))
}

/// `glob(dir/**/*.metal, recursive=True)` (hidden names skipped, as glob does)
/// plus `glob(dir/*.h)`.
fn kernel_sources(kernels: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>, top: bool) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let path = entry.path();
            if std::fs::metadata(&path).is_ok_and(|m| m.is_dir()) {
                walk(&path, out, false);
            } else if name.ends_with(".metal") || (top && name.ends_with(".h")) {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    walk(kernels, &mut out, true);
    out.sort();
    out
}

/// What a run prints and how it ended.
pub struct Report {
    pub ok: bool,
    pub text: String,
}

/// Lint the tree at `root`. `extra` are allowlist entries added to `REVIEWED`.
pub fn lint(root: &Path, extra: &[(String, String)]) -> Report {
    let kernels = root.join("kernels");
    let new: Vec<PathBuf> = NEW.iter().map(|k| kernels.join(format!("{k}.metal"))).collect();
    let fail = |msg: String| Report {
        ok: false,
        text: format!("error: {msg}\n"),
    };

    let mut known = BTreeSet::new();
    for f in kernel_sources(&kernels).iter().filter(|f| !new.contains(f)) {
        match read_source(f) {
            Ok(src) => known.extend(constructs(&src)),
            Err(e) => return fail(e),
        }
    }
    let mut sources = Vec::new();
    for f in &new {
        match read_source(f) {
            Ok(src) => sources.push((f.file_name().unwrap().to_string_lossy().into_owned(), src)),
            Err(e) => return fail(e),
        }
    }
    let mut own = BTreeSet::new();
    for (_, src) in &sources {
        own.extend(defined_here(src));
    }
    // A construct is attributed to the first of the new files that has it.
    let mut novel: BTreeMap<String, String> = BTreeMap::new();
    for (name, src) in &sources {
        for c in constructs(src) {
            if !known.contains(&c) && !own.contains(&c) {
                novel.entry(c).or_insert_with(|| name.clone());
            }
        }
    }

    let reviewed: BTreeMap<&str, &str> = REVIEWED
        .iter()
        .map(|&(k, v)| (k, v))
        .chain(extra.iter().map(|(k, v)| (k.as_str(), v.as_str())))
        .collect();
    let unreviewed: Vec<&String> = novel.keys().filter(|c| !reviewed.contains_key(c.as_str())).collect();
    let stale: Vec<&str> = reviewed.keys().copied().filter(|c| !novel.contains_key(*c)).collect();

    let mut text = String::new();
    for (c, file) in &novel {
        let (mark, why) = match reviewed.get(c.as_str()) {
            Some(why) => ("ok  ", *why),
            None => ("FAIL", "not reviewed: check the MSL spec, then list it"),
        };
        text.push_str(&format!("  [{mark}] {c:<24} {file:<20} {why}\n"));
    }
    if !stale.is_empty() {
        text.push_str(&format!(
            "  FAIL  reviewed entries no longer needed (now used by another kernel, or removed \
             from the Qwen3.5 sources); delete them from REVIEWED: {}\n",
            stale.join(", ")
        ));
    }
    if !unreviewed.is_empty() {
        text.push_str(&format!(
            "\n{} construct(s) new to tessl's kernels and not reviewed\n",
            unreviewed.len()
        ));
    }
    if !stale.is_empty() || !unreviewed.is_empty() {
        return Report { ok: false, text };
    }
    if novel.is_empty() {
        text.push_str(
            "\nexamined nothing: no construct is new to tessl's kernels, so the Qwen3.5 sources were not read\n",
        );
        return Report { ok: false, text };
    }
    text.push_str(&format!(
        "\n{} construct(s) new to tessl's kernels, all reviewed\n",
        novel.len()
    ));
    Report { ok: true, text }
}

fn main() -> ExitCode {
    let mut root = PathBuf::from(".");
    let mut extra = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--reviewed" {
            let Some((name, why)) = args
                .next()
                .and_then(|a| a.split_once('=').map(|(n, w)| (n.to_string(), w.to_string())))
            else {
                eprintln!("usage: dialect_lint [--reviewed NAME=REASON]... [ROOT]");
                return ExitCode::FAILURE;
            };
            extra.push((name, why));
        } else {
            root = PathBuf::from(arg);
        }
    }
    let report = lint(&root, &extra);
    if report.ok {
        print!("{}", report.text);
        ExitCode::SUCCESS
    } else {
        print!("{}", report.text);
        ExitCode::FAILURE
    }
}
