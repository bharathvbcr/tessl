//! The C ABI's handle contract: null handles, foreign threads, freed handles.
//!
//! No test here creates a runtime or needs a GPU (a guard test scans this file
//! for runtime creation); the foreign-thread test, which needs one real runtime
//! handle (a Metal device), lives in `tests/capi.rs`. The aliasing the
//! thread check used to cause (a `&mut` formed on a foreign thread while the
//! owner may be inside a call) is undefined behaviour Miri cannot exercise
//! through Metal, so what is pinned here is the observable half: the refusal
//! is a status, carries the thread message before any argument is looked at,
//! and leaves the handle usable.
//!
//! A freed handle is not tested for: the ABI keeps no registry of live
//! handles, so a call on a freed pointer is undefined behaviour by contract
//! (documented in `src/capi.rs`), and a test would have to commit it.

use std::ffi::{c_char, CStr};
use std::ptr;

use tessl::capi::{
    tessl_cross_entropy_rows, tessl_gdn_train_backward, tessl_gdn_train_forward, tessl_qwen35_adamw_free,
    tessl_qwen35_adamw_set_step_count, tessl_qwen35_adamw_step_count, tessl_qwen35_free, tessl_qwen35_grad_sq_norm,
    tessl_qwen35_load, tessl_qwen35_param_count, tessl_qwen35_train_discard, tessl_qwen35_train_step,
    tessl_runtime_free, tessl_runtime_poisoned, tessl_synchronize, TesslQwen35, TesslRuntime, TESSL_ABI_VERSION,
    TESSL_ERR, TESSL_F32, TESSL_OK,
};

const ERR_LEN: usize = 512;

type Call = (&'static str, i32, String);

fn msg(err: &[c_char]) -> String {
    unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned()
}

fn run(out: &mut Vec<Call>, name: &'static str, f: &dyn Fn(*mut c_char, usize) -> i32) {
    let mut err = [0 as c_char; ERR_LEN];
    let s = f(err.as_mut_ptr(), ERR_LEN);
    out.push((name, s, msg(&err)));
}

/// Every runtime-handle entry point called with `rt` and no valid arguments
/// beyond it: the handle check runs first, so status and message depend only
/// on the handle.
fn runtime_entries(rt: *mut TesslRuntime) -> Vec<Call> {
    let mut out = Vec::new();
    unsafe {
        run(&mut out, "tessl_synchronize", &|e, n| tessl_synchronize(rt, e, n));
        run(&mut out, "tessl_runtime_poisoned", &|e, n| {
            tessl_runtime_poisoned(rt, ptr::null_mut(), e, n)
        });
        run(&mut out, "tessl_cross_entropy_rows", &|e, n| {
            tessl_cross_entropy_rows(rt, ptr::null(), ptr::null_mut(), ptr::null_mut(), e, n)
        });
        run(&mut out, "tessl_gdn_train_forward", &|e, n| {
            tessl_gdn_train_forward(rt, ptr::null(), e, n)
        });
        run(&mut out, "tessl_gdn_train_backward", &|e, n| {
            tessl_gdn_train_backward(rt, ptr::null(), e, n)
        });
        run(&mut out, "tessl_qwen35_load", &|e, n| {
            let mut m = ptr::null_mut();
            tessl_qwen35_load(rt, ptr::null(), ptr::null(), ptr::null(), TESSL_F32, &mut m, e, n)
        });
    }
    out
}

/// Every model-handle entry point, as [`runtime_entries`].
fn model_entries(model: *mut TesslQwen35) -> Vec<Call> {
    let mut out = Vec::new();
    unsafe {
        run(&mut out, "tessl_qwen35_param_count", &|e, n| {
            tessl_qwen35_param_count(model, ptr::null_mut(), e, n)
        });
        run(&mut out, "tessl_qwen35_train_step", &|e, n| {
            tessl_qwen35_train_step(model, ptr::null(), 0, 0, ptr::null_mut(), e, n)
        });
        run(&mut out, "tessl_qwen35_train_discard", &|e, n| {
            tessl_qwen35_train_discard(model, e, n)
        });
        run(&mut out, "tessl_qwen35_adamw_free", &|e, n| {
            tessl_qwen35_adamw_free(model, e, n)
        });
        run(&mut out, "tessl_qwen35_grad_sq_norm", &|e, n| {
            tessl_qwen35_grad_sq_norm(model, ptr::null_mut(), e, n)
        });
        run(&mut out, "tessl_qwen35_adamw_step_count", &|e, n| {
            tessl_qwen35_adamw_step_count(model, ptr::null_mut(), e, n)
        });
        run(&mut out, "tessl_qwen35_adamw_set_step_count", &|e, n| {
            tessl_qwen35_adamw_set_step_count(model, 0, e, n)
        });
    }
    out
}

#[test]
fn a_null_handle_is_a_refusal_with_a_message_and_a_null_free_is_a_no_op() {
    let calls: Vec<Call> = runtime_entries(ptr::null_mut())
        .into_iter()
        .chain(model_entries(ptr::null_mut()))
        .collect();
    assert_eq!(calls.len(), 13);
    for (name, status, message) in calls {
        assert_eq!(status, TESSL_ERR, "{name}");
        assert!(message.starts_with("null "), "{name}: {message:?}");
        assert!(message.contains("handle"), "{name}: {message:?}");
    }
    assert_eq!(unsafe { tessl_runtime_free(ptr::null_mut()) }, TESSL_OK);
    assert_eq!(unsafe { tessl_qwen35_free(ptr::null_mut()) }, TESSL_OK);
}

#[test]
fn a_null_handle_with_no_error_buffer_is_still_a_status() {
    assert_eq!(
        unsafe { tessl_synchronize(ptr::null_mut(), ptr::null_mut(), 0) },
        TESSL_ERR
    );
}

/// `None` when the binding's `_ABI_VERSION` line is `ver`.
fn abi_line_problem(src: &str, ver: u32) -> Option<String> {
    let src = strip_py_comments(src);
    let Some(line) = src.lines().find(|l| l.starts_with("_ABI_VERSION = ")) else {
        return Some("no `_ABI_VERSION = ` line in the binding".to_string());
    };
    let want = format!("_ABI_VERSION = {ver}");
    let line = line.trim_end();
    (line != want).then(|| format!("binding says `{line}`, TESSL_ABI_VERSION says `{want}`"))
}

#[test]
fn the_python_binding_expects_this_abi_version() {
    let src = read("python/tessl_torch/__init__.py");
    assert_eq!(abi_line_problem(&src, TESSL_ABI_VERSION), None);
}

// ---------------------------------------------------------------------------
// The C signatures in `src/capi.rs` against the ctypes declarations in the
// Python binding, both parsed as text (no GPU, no torch). ctypes trusts the
// `argtypes`/`restype` it is given: a missing argument, a 32-bit field where
// the C side has 64, or a changed struct layout is a silent wrong call, not an
// error. `TESSL_ABI_VERSION` only catches a bump somebody remembered.

use std::collections::BTreeMap;

/// A C type reduced to what the calling convention sees.
#[derive(Clone, Debug, PartialEq)]
enum Ty {
    Void,
    /// A scalar by its Rust name (`u32`, `f64`, `c_char`, ...).
    Scalar(String),
    /// A pointer; `None` is an untyped one (`c_void_p`).
    Ptr(Option<Box<Ty>>),
    Struct(String),
    Array(Box<Ty>, usize),
}

impl Ty {
    /// Same ABI type. An untyped or void pointee matches any pointee (a
    /// handle to an opaque Rust struct is `c_void_p` in ctypes).
    fn agrees(&self, other: &Ty) -> bool {
        match (self, other) {
            (Ty::Ptr(a), Ty::Ptr(b)) => match (a.as_deref(), b.as_deref()) {
                (None, _) | (_, None) | (Some(Ty::Void), _) | (_, Some(Ty::Void)) => true,
                (Some(a), Some(b)) => a.agrees(b),
            },
            (Ty::Array(a, n), Ty::Array(b, m)) => n == m && a.agrees(b),
            (a, b) => a == b,
        }
    }
}

#[derive(Debug)]
struct Sig {
    args: Vec<(String, Ty)>,
    ret: Ty,
}

/// Split at `sep` outside any bracket pair or string.
fn split_top(s: &str, sep: char) -> Vec<String> {
    let (mut out, mut depth, mut quote, mut cur) = (Vec::new(), 0i32, None::<char>, String::new());
    for c in s.chars() {
        match quote {
            Some(q) => {
                if c == q {
                    quote = None;
                }
            }
            None => match c {
                '"' | '\'' => quote = Some(c),
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                c if c == sep && depth == 0 => {
                    out.push(std::mem::take(&mut cur));
                    continue;
                }
                _ => {}
            },
        }
        cur.push(c);
    }
    out.push(cur);
    out.into_iter()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

/// The text inside the bracket pair opening at byte `open`, and the index
/// just past its closer.
fn bracketed(s: &str, open: usize) -> (&str, usize) {
    let (mut depth, mut quote) = (0i32, None::<char>);
    for (i, c) in s[open..].char_indices() {
        match quote {
            Some(q) => {
                if c == q {
                    quote = None;
                }
            }
            None => match c {
                '"' | '\'' => quote = Some(c),
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return (&s[open + 1..open + i], open + i + 1);
                    }
                }
                _ => {}
            },
        }
    }
    panic!(
        "unbalanced bracket at byte {open}: {:?}",
        &s[open..(open + 60).min(s.len())]
    );
}

/// Drops Python comments and the contents of triple-quoted strings (docstrings
/// are text, not declarations). Strings are lexed with backslash escapes, so a
/// `#` or quote after `\"` is read the way Python reads it. Newlines are kept
/// so later line structure still lines up.
fn strip_py_comments(src: &str) -> String {
    let c: Vec<char> = src.chars().collect();
    let (mut out, mut i) = (String::new(), 0usize);
    while i < c.len() {
        let ch = c[i];
        if ch == '#' {
            while i < c.len() && c[i] != '\n' {
                i += 1;
            }
        } else if ch == '"' || ch == '\'' {
            let triple = i + 2 < c.len() && c[i + 1] == ch && c[i + 2] == ch;
            if triple {
                i += 3;
                while i < c.len() && !(c[i] == ch && c.get(i + 1) == Some(&ch) && c.get(i + 2) == Some(&ch)) {
                    if c[i] == '\\' {
                        i += 1;
                    }
                    if let Some(&n) = c.get(i) {
                        if n == '\n' {
                            out.push('\n');
                        }
                    }
                    i += 1;
                }
                i += 3;
            } else {
                out.push(ch);
                i += 1;
                while i < c.len() && c[i] != ch && c[i] != '\n' {
                    if c[i] == '\\' && i + 1 < c.len() {
                        // An escape, or a backslash-newline continuation:
                        // both chars belong to the string.
                        out.push(c[i]);
                        i += 1;
                    }
                    out.push(c[i]);
                    i += 1;
                }
                if i < c.len() && c[i] == ch {
                    out.push(ch);
                    i += 1;
                }
            }
        } else {
            out.push(ch);
            i += 1;
        }
    }
    out
}

/// True when the `r` at `i` continues an identifier. A `b` or `c` directly
/// before it is the byte / C string prefix (`br"..."`, `cr"..."`), not an
/// identifier, unless that prefix letter itself follows an identifier char
/// (`abr#` and `acr#` are identifiers).
fn prefix_is_ident(c: &[char], i: usize) -> bool {
    let ident = |j: Option<usize>| {
        j.and_then(|j| c.get(j))
            .is_some_and(|p| p.is_alphanumeric() || *p == '_')
    };
    match i.checked_sub(1) {
        Some(b) if c[b] == 'b' || c[b] == 'c' => ident(b.checked_sub(1)),
        p => ident(p),
    }
}

/// Drops Rust `//` and (nesting) `/* */` comments, keeping newlines. One
/// lexer, so a comment opener inside a string, a char literal or another
/// comment is not an opener, and a string's text is kept whole.
fn strip_line_comments(src: &str) -> String {
    let c: Vec<char> = src.chars().collect();
    let (mut out, mut i) = (String::new(), 0usize);
    while i < c.len() {
        let at = |j: usize| c.get(j).copied();
        if c[i] == '/' && at(i + 1) == Some('/') {
            while i < c.len() && c[i] != '\n' {
                i += 1;
            }
        } else if c[i] == '/' && at(i + 1) == Some('*') {
            let mut depth = 0usize;
            loop {
                if at(i) == Some('/') && at(i + 1) == Some('*') {
                    depth += 1;
                    i += 2;
                } else if at(i) == Some('*') && at(i + 1) == Some('/') {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else if let Some(x) = at(i) {
                    if x == '\n' {
                        out.push('\n');
                    }
                    i += 1;
                } else {
                    break;
                }
            }
        } else if c[i] == 'r' && matches!(at(i + 1), Some('"' | '#')) && !prefix_is_ident(&c, i) {
            // Raw string: r"..", r#".."#, and the br / cr forms (the b or c
            // is already in `out`). b".." and c".." are ordinary strings.
            let mut j = i + 1;
            while at(j) == Some('#') {
                j += 1;
            }
            if at(j) != Some('"') {
                out.push(c[i]);
                i += 1;
                continue;
            }
            let hashes = j - i - 1;
            j += 1;
            while j < c.len() && !(c[j] == '"' && (1..=hashes).all(|k| at(j + k) == Some('#'))) {
                j += 1;
            }
            let end = (j + 1 + hashes).min(c.len());
            out.extend(&c[i..end]);
            i = end;
        } else if c[i] == '"' {
            out.push('"');
            i += 1;
            while i < c.len() && c[i] != '"' {
                if c[i] == '\\' && i + 1 < c.len() {
                    out.push(c[i]);
                    i += 1;
                }
                out.push(c[i]);
                i += 1;
            }
            if i < c.len() {
                out.push('"');
                i += 1;
            }
        } else if c[i] == '\'' {
            // A char literal ('x', '\n', '\'') or a lifetime ('a).
            let end = match (at(i + 1), at(i + 2)) {
                (Some('\\'), _) => (i + 3..c.len().min(i + 14)).find(|&j| c[j] == '\''),
                (Some(_), Some('\'')) => Some(i + 2),
                _ => None,
            };
            let to = end.map_or(i + 1, |e| e + 1);
            out.extend(&c[i..to]);
            i = to;
        } else {
            out.push(c[i]);
            i += 1;
        }
    }
    out
}

/// `<decl>NAME[: T] = <integer>` constants, for array lengths.
fn int_consts(src: &str, decl: &str) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for line in src.lines() {
        let Some(rest) = line.strip_prefix(decl) else { continue };
        let Some((name, value)) = rest.split_once('=') else {
            continue;
        };
        let name = name.split(':').next().unwrap().trim();
        if let Ok(v) = value.trim().trim_end_matches(';').trim().parse() {
            out.insert(name.to_string(), v);
        }
    }
    out
}

struct RustAbi {
    fns: BTreeMap<String, Sig>,
    structs: BTreeMap<String, Vec<(String, Ty)>>,
}

fn rust_ty(t: &str, consts: &BTreeMap<String, usize>, repr: &[String]) -> Ty {
    let t = t.trim();
    if let Some(rest) = t.strip_prefix("*const ").or_else(|| t.strip_prefix("*mut ")) {
        return Ty::Ptr(Some(Box::new(rust_ty(rest, consts, repr))));
    }
    if let Some(inner) = t.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
        let (elem, len) = inner.rsplit_once(';').expect("array type has a length");
        let len = len.trim();
        let n = len
            .parse()
            .ok()
            .or_else(|| consts.get(len).copied())
            .unwrap_or_else(|| panic!("array length {len:?}"));
        return Ty::Array(Box::new(rust_ty(elem, consts, repr)), n);
    }
    match t {
        "i8" | "u8" | "i16" | "u16" | "i32" | "u32" | "i64" | "u64" | "usize" | "isize" | "f32" | "f64" | "c_char" => {
            Ty::Scalar(t.to_string())
        }
        "c_void" => Ty::Void,
        name if repr.iter().any(|r| r == name) => Ty::Struct(name.trim_start_matches("Tessl").to_string()),
        // A handle type with no `#[repr(C)]` layout: opaque to C.
        _ => Ty::Void,
    }
}

fn parse_rust_abi(src: &str) -> RustAbi {
    let src = strip_line_comments(src);
    let consts = int_consts(&src, "pub const ");
    let mut repr = Vec::new();
    let mut raw_structs = Vec::new();
    let mut rest = src.as_str();
    while let Some(i) = rest.find("#[repr(C)]") {
        rest = &rest[i + "#[repr(C)]".len()..];
        let s = rest.find("pub struct ").expect("a struct after #[repr(C)]");
        let after = &rest[s + "pub struct ".len()..];
        let name = after[..after.find(|c: char| !c.is_alphanumeric() && c != '_').unwrap()].to_string();
        let (body, _) = bracketed(after, after.find('{').unwrap());
        repr.push(name.clone());
        raw_structs.push((name, body.to_string()));
    }
    let structs = raw_structs
        .into_iter()
        .map(|(name, body)| {
            let fields = split_top(&body, ',')
                .into_iter()
                .map(|f| {
                    let f = f.trim_start_matches("pub ").trim();
                    let (n, t) = f.split_once(':').unwrap_or_else(|| panic!("field {f:?}"));
                    (n.trim().to_string(), rust_ty(t, &consts, &repr))
                })
                .collect();
            (name.trim_start_matches("Tessl").to_string(), fields)
        })
        .collect();

    let mut fns = BTreeMap::new();
    let mut rest = src.as_str();
    while let Some(i) = rest.find("extern \"C\" fn tessl_") {
        rest = &rest[i + "extern \"C\" fn ".len()..];
        let open = rest.find('(').unwrap();
        let name = rest[..open].trim().to_string();
        let (params, end) = bracketed(rest, open);
        let args = split_top(params, ',')
            .into_iter()
            .map(|p| {
                let (n, t) = p.split_once(':').unwrap_or_else(|| panic!("{name}: parameter {p:?}"));
                (n.trim().to_string(), rust_ty(t, &consts, &repr))
            })
            .collect();
        let tail = rest[end..].trim_start();
        let ret = match tail.strip_prefix("->") {
            Some(r) => rust_ty(&r[..r.find('{').unwrap()], &consts, &repr),
            None => Ty::Void,
        };
        assert!(
            fns.insert(name.clone(), Sig { args, ret }).is_none(),
            "{name} defined twice"
        );
        rest = &rest[end..];
    }
    RustAbi { fns, structs }
}

fn py_ty(expr: &str, consts: &BTreeMap<String, usize>) -> Ty {
    let e = expr.trim().trim_start_matches("ctypes.");
    if let Some(inner) = e.strip_prefix("POINTER(").and_then(|r| r.strip_suffix(')')) {
        return Ty::Ptr(Some(Box::new(py_ty(inner, consts))));
    }
    if let Some((elem, n)) = e.rsplit_once('*') {
        let n = n.trim();
        let n = n
            .parse()
            .ok()
            .or_else(|| consts.get(n).copied())
            .unwrap_or_else(|| panic!("array length {n:?}"));
        return Ty::Array(Box::new(py_ty(elem, consts)), n);
    }
    match e {
        "None" => Ty::Void,
        "c_void_p" => Ty::Ptr(None),
        "c_char_p" => Ty::Ptr(Some(Box::new(Ty::Scalar("c_char".into())))),
        "c_char" => Ty::Scalar("c_char".into()),
        "c_int32" | "c_int" => Ty::Scalar("i32".into()),
        "c_uint32" | "c_uint" => Ty::Scalar("u32".into()),
        "c_int64" => Ty::Scalar("i64".into()),
        "c_uint64" => Ty::Scalar("u64".into()),
        "c_size_t" => Ty::Scalar("usize".into()),
        "c_float" => Ty::Scalar("f32".into()),
        "c_double" => Ty::Scalar("f64".into()),
        name if name.starts_with('_') => Ty::Struct(name.trim_start_matches('_').to_string()),
        other => panic!("unrecognised ctypes type {other:?}"),
    }
}

struct PyAbi {
    /// name -> (argtypes, restype), as declared. A function with no
    /// `restype` line keeps `None`: ctypes would assume a C int.
    fns: BTreeMap<String, (Vec<Ty>, Option<Ty>)>,
    structs: BTreeMap<String, Vec<(String, Ty)>>,
}

fn unquote(s: &str) -> String {
    s.trim().trim_matches(|c| c == '"' || c == '\'').to_string()
}

/// Inline local aliases such as `u32p = ctypes.POINTER(ctypes.c_uint32)`:
/// every later whole-word use of the alias becomes its definition.
fn inline_aliases(src: &str) -> String {
    let mut out = src.to_string();
    for line in src.lines() {
        let t = line.trim();
        let Some((name, value)) = t.split_once(" = ") else {
            continue;
        };
        let is_ident = !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_');
        if !is_ident
            || name.starts_with('_')
            || !(value.starts_with("ctypes.POINTER(") || value.starts_with("ctypes.c_"))
        {
            continue;
        }
        let Some(at) = out.find(t) else { continue };
        let (head, tail) = out.split_at(at + t.len());
        let mut replaced = String::new();
        let mut rest = tail;
        while let Some(i) = rest.find(name) {
            let before = rest[..i].chars().next_back();
            let after = rest[i + name.len()..].chars().next();
            let word = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '.');
            replaced.push_str(&rest[..i]);
            replaced.push_str(if word(before) || word(after) { name } else { value });
            rest = &rest[i + name.len()..];
        }
        replaced.push_str(rest);
        out = format!("{head}{replaced}");
    }
    out
}

fn parse_py_abi(src: &str) -> PyAbi {
    let src: &str = &inline_aliases(&strip_py_comments(src));
    // Module-level constants sit at column 0 as `_NAME = 6`.
    let consts = int_consts(src, "");

    let mut structs = BTreeMap::new();
    let mut rest = src;
    while let Some(i) = rest.find("(ctypes.Structure):") {
        let head = &rest[..i];
        let name = head[head.rfind("class ").unwrap() + "class ".len()..]
            .trim()
            .trim_start_matches('_')
            .to_string();
        rest = &rest[i..];
        let f = rest.find("_fields_ = [").expect("_fields_ after a Structure");
        let (body, end) = bracketed(rest, f + "_fields_ = ".len());
        let mut fields = Vec::new();
        for entry in split_top(body, ',') {
            let (inner, _) = bracketed(&entry, entry.find('(').unwrap());
            let parts = split_top(inner, ',');
            fields.push((unquote(&parts[0]), py_ty(&parts[1], &consts)));
        }
        // `+ [(name, T) for name in ("a", "b")]`: fields generated from a name list.
        let after = rest[end..].trim_start();
        if after.starts_with("+ [") {
            let (inner, _) = bracketed(after, "+ ".len());
            let (pair, pair_end) = bracketed(inner, inner.find('(').unwrap());
            let ty = py_ty(&split_top(pair, ',')[1], &consts);
            let tail = &inner[pair_end..];
            let (names, _) = bracketed(tail, tail.find(" in (").expect("a name tuple") + " in ".len());
            for n in split_top(names, ',') {
                fields.push((unquote(&n), ty.clone()));
            }
        }
        structs.insert(name, fields);
        rest = &rest[end..];
    }

    // `err_args = [...]`, appended to some argtypes with `+ err_args`.
    let err_args: Vec<Ty> = {
        let i = src.find("err_args = [").expect("err_args");
        let (body, _) = bracketed(src, i + "err_args = ".len());
        split_top(body, ',').iter().map(|t| py_ty(t, &consts)).collect()
    };
    let argtypes = |after_eq: &str| -> Vec<Ty> {
        let (body, end) = bracketed(after_eq, after_eq.find('[').unwrap());
        let mut tys: Vec<Ty> = split_top(body, ',').iter().map(|t| py_ty(t, &consts)).collect();
        if after_eq[end..].trim_start().starts_with("+ err_args") {
            tys.extend(err_args.iter().cloned());
        }
        tys
    };
    let line_value = |after: &str| -> String { after.lines().next().unwrap().trim().to_string() };

    let mut fns: BTreeMap<String, (Vec<Ty>, Option<Ty>)> = BTreeMap::new();
    // `lib.tessl_x.argtypes = [...]` / `.restype = T`.
    let mut rest = src;
    while let Some(i) = rest.find("lib.tessl_") {
        rest = &rest[i + "lib.".len()..];
        let dot = rest.find('.').unwrap_or(rest.len());
        let name = rest[..dot].to_string();
        if let Some(v) = rest[dot..].strip_prefix(".restype = ") {
            fns.entry(name).or_default().1 = Some(py_ty(&line_value(v), &consts));
        } else if let Some(v) = rest[dot..].strip_prefix(".argtypes = ") {
            fns.entry(name).or_default().0 = argtypes(v);
        }
        // Anything else is a call such as `lib.tessl_runtime_new(...)`.
    }
    // `for name in ("tessl_a", "tessl_b"):` then `fn.restype` / `fn.argtypes`.
    let mut rest = src;
    while let Some(i) = rest.find("for name in (\"tessl_") {
        let from = &rest[i + "for name in ".len()..];
        let (names, end) = bracketed(from, 0);
        let body = &from[end..];
        let r = body.find("fn.restype = ").expect("fn.restype in the loop") + "fn.restype = ".len();
        let a = body.find("fn.argtypes = ").expect("fn.argtypes in the loop") + "fn.argtypes = ".len();
        let (ret, args) = (py_ty(&line_value(&body[r..]), &consts), argtypes(&body[a..]));
        for n in split_top(names, ',') {
            fns.insert(unquote(&n), (args.clone(), Some(ret.clone())));
        }
        rest = body;
    }
    PyAbi { fns, structs }
}

fn read(path: &str) -> String {
    std::fs::read_to_string(format!("{}/{path}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

/// Exports the binding deliberately does not declare. Empty: every export is
/// bound, `tessl_synchronize` included. An entry that is bound, or not
/// exported, is itself a problem, so the list cannot go stale.
const UNBOUND_EXPORTS: &[&str] = &[];

/// Every way a bound function can disagree with its C signature, as data so
/// that a fault can be fed in as edited source text.
fn fn_problems(rust: &RustAbi, py: &PyAbi, unbound_ok: &[&str]) -> Vec<String> {
    let mut problems = Vec::new();
    for name in rust.fns.keys().filter(|k| !py.fns.contains_key(*k)) {
        if !unbound_ok.contains(&name.as_str()) {
            problems.push(format!(
                "{name}: in src/capi.rs, not declared in the Python binding and not on the allowlist"
            ));
        }
    }
    for name in unbound_ok {
        if !rust.fns.contains_key(*name) {
            problems.push(format!(
                "{name}: on the unbound allowlist, but not exported by src/capi.rs"
            ));
        } else if py.fns.contains_key(*name) {
            problems.push(format!("{name}: on the unbound allowlist, but the binding declares it"));
        }
    }
    for name in py.fns.keys().filter(|k| !rust.fns.contains_key(*k)) {
        problems.push(format!("{name}: declared in the Python binding, not in src/capi.rs"));
    }
    for (name, sig) in &rust.fns {
        let Some((pargs, pret)) = py.fns.get(name) else {
            continue;
        };
        match pret {
            None => problems.push(format!("{name}: no restype (ctypes would assume a C int)")),
            Some(pret) if !sig.ret.agrees(pret) => {
                problems.push(format!("{name}: returns {:?} in C, restype is {pret:?}", sig.ret));
            }
            Some(_) => {}
        }
        if sig.args.len() != pargs.len() {
            problems.push(format!(
                "{name}: C takes {} arguments, ctypes declares {}",
                sig.args.len(),
                pargs.len()
            ));
            continue;
        }
        for (i, ((an, cty), pty)) in sig.args.iter().zip(pargs).enumerate() {
            if !cty.agrees(pty) {
                problems.push(format!(
                    "{name}: argument {i} `{an}` is {cty:?} in C, {pty:?} in ctypes"
                ));
            }
        }
    }
    problems
}

/// Struct names, field order, field types.
fn struct_problems(rust: &RustAbi, py: &PyAbi) -> Vec<String> {
    let mut problems = Vec::new();
    let rust_names: Vec<_> = rust.structs.keys().collect();
    let py_names: Vec<_> = py.structs.keys().collect();
    if rust_names != py_names {
        problems.push(format!(
            "#[repr(C)] structs {rust_names:?} vs ctypes Structures {py_names:?}"
        ));
    }
    for (name, rfields) in &rust.structs {
        let Some(pfields) = py.structs.get(name) else {
            continue;
        };
        if rfields.len() != pfields.len() {
            problems.push(format!(
                "{name}: {} fields in C, {} in ctypes",
                rfields.len(),
                pfields.len()
            ));
            continue;
        }
        for (i, ((rn, rt), (pn, pt))) in rfields.iter().zip(pfields).enumerate() {
            if rn != pn || !rt.agrees(pt) {
                problems.push(format!(
                    "{name}: field {i} is `{rn}`: {rt:?} in C, `{pn}`: {pt:?} in ctypes"
                ));
            }
        }
    }
    problems
}

type Structs = BTreeMap<String, Vec<(String, Ty)>>;

/// Size and alignment under C's natural-alignment rules, which is what both
/// rustc (`repr(C)`) and ctypes (no `_pack_`) apply.
fn c_layout(ty: &Ty, structs: &Structs) -> (usize, usize) {
    match ty {
        Ty::Scalar(n) => {
            let w = match n.as_str() {
                "i8" | "u8" | "c_char" => 1,
                "i16" | "u16" => 2,
                "i32" | "u32" | "f32" => 4,
                "i64" | "u64" | "f64" | "usize" | "isize" => 8,
                other => panic!("no width for scalar {other:?}"),
            };
            (w, w)
        }
        Ty::Ptr(_) => (8, 8),
        Ty::Void => panic!("a void value has no layout"),
        Ty::Array(elem, n) => {
            let (size, align) = c_layout(elem, structs);
            (size * n, align)
        }
        Ty::Struct(name) => {
            let fields = structs.get(name).unwrap_or_else(|| panic!("unknown struct {name:?}"));
            let (_, size, align) = field_offsets(fields, structs);
            (size, align)
        }
    }
}

/// Offsets, total size and alignment of a struct's fields in declaration order.
fn field_offsets(fields: &[(String, Ty)], structs: &Structs) -> (Vec<usize>, usize, usize) {
    let (mut at, mut max_align, mut offsets) = (0usize, 1usize, Vec::new());
    for (_, ty) in fields {
        let (size, align) = c_layout(ty, structs);
        at = at.next_multiple_of(align);
        offsets.push(at);
        at += size;
        max_align = max_align.max(align);
    }
    (offsets, at.next_multiple_of(max_align), max_align)
}

/// The ctypes side's offsets and size.
fn py_offsets(fields: &[(String, Ty)], structs: &Structs) -> (Vec<usize>, usize) {
    let (offsets, size, _) = field_offsets(fields, structs);
    (offsets, size)
}

/// `rustc`'s offsets and size (from `offset_of!`) against ctypes's.
fn offset_problems(
    name: &str,
    rust_offsets: &[(String, usize)],
    rust_size: usize,
    py_fields: &[(String, Ty)],
    py_structs: &Structs,
) -> Vec<String> {
    let (offsets, size) = py_offsets(py_fields, py_structs);
    let mut problems = Vec::new();
    for (field, at) in rust_offsets {
        match py_fields.iter().position(|(n, _)| n == field) {
            None => problems.push(format!("{name}.{field}: at byte {at} in C, absent from ctypes")),
            Some(i) if offsets[i] != *at => problems.push(format!(
                "{name}.{field}: at byte {at} in C, at byte {} in ctypes",
                offsets[i]
            )),
            Some(_) => {}
        }
    }
    if py_fields.len() != rust_offsets.len() {
        problems.push(format!(
            "{name}: {} fields in C, {} in ctypes",
            rust_offsets.len(),
            py_fields.len()
        ));
    }
    if size != rust_size {
        problems.push(format!("{name}: {rust_size} bytes in C, {size} in ctypes"));
    }
    problems
}

/// `_pack_` or `_anonymous_` would change ctypes's layout rules and make
/// `py_offsets` wrong without any field changing.
fn layout_guard_problem(py_src: &str) -> Option<String> {
    let code = strip_py_comments(py_src);
    ["_pack_", "_anonymous_"]
        .into_iter()
        .find(|k| code.contains(k))
        .map(|k| format!("the binding uses `{k}`, which the layout check does not model"))
}

/// `(name, [(field, offset_of!)], size_of)` straight from rustc.
macro_rules! rust_layout {
    ($t:ident { $($f:ident),+ $(,)? }) => {(
        stringify!($t).trim_start_matches("Tessl").to_string(),
        vec![$((stringify!($f).to_string(), std::mem::offset_of!(tessl::capi::$t, $f))),+],
        std::mem::size_of::<tessl::capi::$t>(),
    )};
}

/// A struct's name, its `(field, offset)` pairs and its size.
type RustLayout = (String, Vec<(String, usize)>, usize);

fn rust_layouts() -> Vec<RustLayout> {
    vec![
        rust_layout!(TesslTensorRef {
            buffer,
            byte_offset,
            dtype,
            ndim,
            shape
        }),
        rust_layout!(TesslCeArgs {
            hidden,
            col_off,
            weight,
            rows,
            targets,
            n,
            reduction,
            operands,
            chunk,
            want_grads,
            scale,
            dh,
            dw
        }),
        rust_layout!(TesslGdnArgs {
            batch,
            seq,
            heads,
            v_dim,
            q,
            k,
            v,
            g,
            beta,
            s0,
            ckpt,
            o,
            s_fin,
            d_o,
            d_fin,
            dq,
            dk,
            dv,
            dg,
            dbeta,
            ds0
        }),
        rust_layout!(TesslParamInfo {
            name,
            ndim,
            transposed,
            decay_excluded,
            shape
        }),
    ]
}

const PY: &str = "python/tessl_torch/__init__.py";
const RS: &str = "src/capi.rs";

#[test]
fn every_c_function_is_declared_in_python_with_the_same_arity_and_widths() {
    let rust = parse_rust_abi(&read(RS));
    let py = parse_py_abi(&read(PY));
    // A parser that quietly found nothing would pass vacuously.
    assert!(rust.fns.len() >= 26, "parsed only {} C functions", rust.fns.len());
    assert!(rust.fns.contains_key("tessl_runtime_poisoned"));
    let problems = fn_problems(&rust, &py, UNBOUND_EXPORTS);
    assert!(problems.is_empty(), "C ABI / ctypes mismatch:\n{}", problems.join("\n"));
}

#[test]
fn every_repr_c_struct_matches_its_ctypes_structure_field_for_field() {
    let rust = parse_rust_abi(&read(RS));
    let py = parse_py_abi(&read(PY));
    assert!(rust.structs.len() >= 4, "parsed only {} structs", rust.structs.len());
    let problems = struct_problems(&rust, &py);
    assert!(problems.is_empty(), "struct layout mismatch:\n{}", problems.join("\n"));
}

#[test]
fn every_ctypes_structure_has_the_rust_field_offsets_and_size() {
    let rust = parse_rust_abi(&read(RS));
    let py = parse_py_abi(&read(PY));
    assert_eq!(layout_guard_problem(&read(PY)), None);
    let layouts = rust_layouts();
    assert_eq!(
        layouts.len(),
        rust.structs.len(),
        "a #[repr(C)] struct is missing from rust_layouts"
    );
    let mut problems = Vec::new();
    for (name, offsets, size) in &layouts {
        let parsed = rust
            .structs
            .get(name)
            .unwrap_or_else(|| panic!("{name} is not a parsed #[repr(C)] struct"));
        let parsed_names: Vec<_> = parsed.iter().map(|(n, _)| n.as_str()).collect();
        let listed: Vec<_> = offsets.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            listed, parsed_names,
            "{name}: rust_layouts lists different fields than src/capi.rs"
        );
        // The layout model must agree with rustc, or it proves nothing about ctypes.
        let (model, model_size, _) = field_offsets(parsed, &rust.structs);
        for ((field, at), m) in offsets.iter().zip(&model) {
            assert_eq!(at, m, "{name}.{field}: offset_of! vs the layout model");
        }
        assert_eq!(*size, model_size, "{name}: size_of vs the layout model");
        let pfields = py
            .structs
            .get(name)
            .unwrap_or_else(|| panic!("{name}: no ctypes Structure"));
        problems.extend(offset_problems(name, offsets, *size, pfields, &py.structs));
    }
    assert!(problems.is_empty(), "struct offsets differ:\n{}", problems.join("\n"));
}

// ---------------------------------------------------------------------------
// Injected faults: an edited copy of each source with one thing changed must
// be reported. Every edit asserts its needle is present, so a rename fails
// loudly rather than letting the fault go vacuous.

fn fault(src: &str, needle: &str, with: &str) -> String {
    assert!(src.contains(needle), "fault needle not in the source: {needle:?}");
    src.replacen(needle, with, 1)
}

fn py_faulted(needle: &str, with: &str) -> PyAbi {
    parse_py_abi(&fault(&read(PY), needle, with))
}

fn real() -> (RustAbi, PyAbi) {
    (parse_rust_abi(&read(RS)), parse_py_abi(&read(PY)))
}

fn mentions(problems: &[String], what: &str) -> bool {
    problems.iter().any(|p| p.contains(what))
}

#[test]
fn the_unmodified_sources_are_clean() {
    let (rust, py) = real();
    assert!(fn_problems(&rust, &py, UNBOUND_EXPORTS).is_empty());
    assert!(struct_problems(&rust, &py).is_empty());
}

#[test]
fn a_narrowed_argument_width_is_caught() {
    let (rust, _) = real();
    let py = py_faulted(
        "lib.tessl_qwen35_adamw_set_step_count.argtypes = [ctypes.c_void_p, ctypes.c_uint64]",
        "lib.tessl_qwen35_adamw_set_step_count.argtypes = [ctypes.c_void_p, ctypes.c_uint32]",
    );
    let p = fn_problems(&rust, &py, UNBOUND_EXPORTS);
    assert!(
        mentions(&p, "tessl_qwen35_adamw_set_step_count: argument 1 `step`"),
        "{p:?}"
    );
}

#[test]
fn a_signedness_change_is_caught() {
    let (rust, _) = real();
    let py = py_faulted(
        "lib.tessl_synchronize.restype = ctypes.c_int32",
        "lib.tessl_synchronize.restype = ctypes.c_uint32",
    );
    let p = fn_problems(&rust, &py, UNBOUND_EXPORTS);
    assert!(mentions(&p, "tessl_synchronize: returns"), "{p:?}");
}

#[test]
fn a_dropped_argument_is_caught() {
    let (rust, _) = real();
    let py = py_faulted(
        "lib.tessl_mtl_buffer_length.argtypes = [ctypes.c_void_p]",
        "lib.tessl_mtl_buffer_length.argtypes = []",
    );
    let p = fn_problems(&rust, &py, UNBOUND_EXPORTS);
    assert!(
        mentions(&p, "tessl_mtl_buffer_length: C takes 1 arguments, ctypes declares 0"),
        "{p:?}"
    );
}

#[test]
fn a_changed_rust_width_is_caught() {
    let (_, py) = real();
    let rs = fault(
        &read(RS),
        "tessl_qwen35_adamw_set_step_count(\n    model: *mut TesslQwen35,\n    step: u64,",
        "tessl_qwen35_adamw_set_step_count(\n    model: *mut TesslQwen35,\n    step: u32,",
    );
    let p = fn_problems(&parse_rust_abi(&rs), &py, UNBOUND_EXPORTS);
    assert!(
        mentions(&p, "tessl_qwen35_adamw_set_step_count: argument 1 `step`"),
        "{p:?}"
    );
}

#[test]
fn a_removed_restype_is_reported_not_assumed() {
    let (rust, _) = real();
    let py = py_faulted("        lib.tessl_synchronize.restype = ctypes.c_int32\n", "");
    let p = fn_problems(&rust, &py, UNBOUND_EXPORTS);
    assert!(mentions(&p, "tessl_synchronize: no restype"), "{p:?}");
}

#[test]
fn an_unbound_export_fails_unless_allowlisted() {
    let (rust, _) = real();
    let src = fault(
        &read(PY),
        "        lib.tessl_synchronize.restype = ctypes.c_int32\n        lib.tessl_synchronize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]\n",
        "",
    );
    let py = parse_py_abi(&src);
    let p = fn_problems(&rust, &py, UNBOUND_EXPORTS);
    assert!(mentions(&p, "tessl_synchronize: in src/capi.rs, not declared"), "{p:?}");
    assert_eq!(fn_problems(&rust, &py, &["tessl_synchronize"]), Vec::<String>::new());
}

#[test]
fn a_stale_allowlist_entry_fails() {
    let (rust, py) = real();
    let bound = fn_problems(&rust, &py, &["tessl_synchronize"]);
    assert!(
        mentions(
            &bound,
            "tessl_synchronize: on the unbound allowlist, but the binding declares it"
        ),
        "{bound:?}"
    );
    let gone = fn_problems(&rust, &py, &["tessl_no_such_export"]);
    assert!(
        mentions(
            &gone,
            "tessl_no_such_export: on the unbound allowlist, but not exported"
        ),
        "{gone:?}"
    );
}

#[test]
fn a_reordered_struct_field_is_caught() {
    let (rust, _) = real();
    let src = fault(
        &read(PY),
        "        (\"col_off\", ctypes.c_uint32),\n        (\"weight\", _TensorRef),\n",
        "        (\"weight\", _TensorRef),\n        (\"col_off\", ctypes.c_uint32),\n",
    );
    let py = parse_py_abi(&src);
    let p = struct_problems(&rust, &py);
    assert!(mentions(&p, "CeArgs: field 1"), "{p:?}");
    let (_, offsets, size) = rust_layouts().into_iter().find(|(n, ..)| n == "CeArgs").unwrap();
    let p = offset_problems("CeArgs", &offsets, size, &py.structs["CeArgs"], &py.structs);
    assert!(mentions(&p, "CeArgs.col_off"), "{p:?}");
    assert!(mentions(&p, "CeArgs.weight"), "{p:?}");
}

#[test]
fn a_narrowed_struct_field_changes_the_offsets_and_size() {
    let src = fault(
        &read(PY),
        "        (\"byte_offset\", ctypes.c_uint64),\n        (\"dtype\", ctypes.c_uint32),",
        "        (\"byte_offset\", ctypes.c_uint32),\n        (\"dtype\", ctypes.c_uint32),",
    );
    let py = parse_py_abi(&src);
    let (_, offsets, size) = rust_layouts().into_iter().find(|(n, ..)| n == "TensorRef").unwrap();
    let p = offset_problems("TensorRef", &offsets, size, &py.structs["TensorRef"], &py.structs);
    assert!(mentions(&p, "TensorRef.dtype"), "{p:?}");
}

#[test]
fn natural_alignment_pads_a_narrow_field_before_a_wide_one() {
    let mut structs = Structs::new();
    let fields = vec![
        ("a".to_string(), Ty::Scalar("u32".into())),
        ("b".to_string(), Ty::Scalar("u64".into())),
    ];
    structs.insert("X".to_string(), fields.clone());
    assert_eq!(py_offsets(&fields, &structs), (vec![0, 8], 16));
    // A nested struct takes its own alignment and rounds its size up.
    let outer = vec![
        ("c".to_string(), Ty::Scalar("u8".into())),
        ("x".to_string(), Ty::Struct("X".into())),
        ("p".to_string(), Ty::Ptr(None)),
    ];
    assert_eq!(py_offsets(&outer, &structs), (vec![0, 8, 24], 32));
}

#[test]
fn packed_ctypes_layout_is_refused() {
    assert!(layout_guard_problem("class _A(ctypes.Structure):\n    _pack_ = 1\n").is_some());
    assert!(layout_guard_problem("# _pack_ is not used\nx = 1\n").is_none());
}

#[test]
fn an_abi_version_mismatch_is_caught() {
    let src = read(PY);
    assert_eq!(abi_line_problem(&src, TESSL_ABI_VERSION), None);
    let stale = fault(
        &src,
        &format!("_ABI_VERSION = {TESSL_ABI_VERSION}"),
        "_ABI_VERSION = 10",
    );
    assert!(abi_line_problem(&stale, TESSL_ABI_VERSION).is_some());
    assert!(abi_line_problem("x = 1\n", TESSL_ABI_VERSION).is_some());
}

// ---------------------------------------------------------------------------
// Adversarial cases from the review of this checker: comment handling, layout
// edges and the void-pointer wildcard.

#[test]
fn a_zero_width_array_and_empty_struct_have_no_size() {
    let structs = Structs::new();
    assert_eq!(py_offsets(&[], &structs), (vec![], 0));
    let arr = vec![("a".to_string(), Ty::Array(Box::new(Ty::Scalar("u32".into())), 0))];
    assert_eq!(py_offsets(&arr, &structs), (vec![0], 0));
}

#[test]
fn a_trailing_narrow_field_is_padded_to_the_struct_alignment() {
    let fields = vec![
        ("a".to_string(), Ty::Scalar("u64".into())),
        ("b".to_string(), Ty::Scalar("u8".into())),
    ];
    assert_eq!(py_offsets(&fields, &Structs::new()), (vec![0, 8], 16));
}

#[test]
fn a_pointer_field_is_eight_bytes_whatever_its_pointee() {
    let s = Structs::new();
    for p in [
        Ty::Ptr(None),
        Ty::Ptr(Some(Box::new(Ty::Void))),
        Ty::Ptr(Some(Box::new(Ty::Scalar("u8".into())))),
    ] {
        assert_eq!(c_layout(&p, &s), (8, 8));
    }
}

#[test]
fn a_pointer_to_the_wrong_scalar_is_not_masked_by_the_void_wildcard() {
    let (rust, _) = real();
    // set_step_count takes a u64 by value; make Python pass a pointer instead.
    let py = py_faulted(
        "lib.tessl_qwen35_adamw_set_step_count.argtypes = [ctypes.c_void_p, ctypes.c_uint64]",
        "lib.tessl_qwen35_adamw_set_step_count.argtypes = [ctypes.c_void_p, ctypes.c_void_p]",
    );
    let p = fn_problems(&rust, &py, UNBOUND_EXPORTS);
    assert!(mentions(&p, "tessl_qwen35_adamw_set_step_count: argument 1"), "{p:?}");
}

#[test]
fn a_swapped_pair_of_equal_width_arguments_is_invisible_by_design() {
    // Documents a limit, not a defect: two u32 arguments swapped in Python
    // agree on every width, so only the names-in-order check could catch it,
    // and ctypes has no names. If this ever starts failing, the checker grew
    // a name-based comparison and this test should assert the new behaviour.
    let (rust, py) = real();
    let same = rust
        .fns
        .iter()
        .any(|(n, s)| s.args.windows(2).any(|w| w[0].1 == w[1].1) && py.fns.contains_key(n));
    assert!(same, "no function has two adjacent same-typed arguments to swap");
}

#[test]
fn a_python_comment_ending_a_line_does_not_leak_into_the_type() {
    let (rust, _) = real();
    let py = py_faulted(
        "lib.tessl_synchronize.restype = ctypes.c_int32\n",
        "lib.tessl_synchronize.restype = ctypes.c_int32  # status\n",
    );
    assert!(fn_problems(&rust, &py, UNBOUND_EXPORTS).is_empty());
}

#[test]
fn an_abi_line_with_a_trailing_comment_is_still_read() {
    let src = format!("_ABI_VERSION = {TESSL_ABI_VERSION}  # keep in step\n");
    assert_eq!(abi_line_problem(&src, TESSL_ABI_VERSION), None);
}

#[test]
fn a_leading_zero_abi_literal_is_not_the_same_line() {
    // Python reads `011` as a syntax error, so the checker must not bless it.
    let src = format!("_ABI_VERSION = 0{TESSL_ABI_VERSION}\n");
    assert!(abi_line_problem(&src, TESSL_ABI_VERSION).is_some());
}

#[test]
fn a_commented_out_restype_is_reported_not_assumed() {
    let (rust, _) = real();
    let py = py_faulted(
        "        lib.tessl_synchronize.restype = ctypes.c_int32\n",
        "        # lib.tessl_synchronize.restype = ctypes.c_int32\n",
    );
    let p = fn_problems(&rust, &py, UNBOUND_EXPORTS);
    assert!(mentions(&p, "tessl_synchronize: no restype"), "{p:?}");
}

#[test]
fn a_commented_out_argtypes_is_a_missing_declaration() {
    let (rust, _) = real();
    let src = fault(
        &read(PY),
        "        lib.tessl_synchronize.restype = ctypes.c_int32\n        lib.tessl_synchronize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]\n",
        "        # lib.tessl_synchronize.restype = ctypes.c_int32\n        # lib.tessl_synchronize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]\n",
    );
    let p = fn_problems(&rust, &parse_py_abi(&src), UNBOUND_EXPORTS);
    assert!(mentions(&p, "tessl_synchronize: in src/capi.rs, not declared"), "{p:?}");
}

#[test]
fn a_block_commented_export_is_not_an_export() {
    let (_, py) = real();
    let rs = format!(
        "{}\n/* pub extern \"C\" fn tessl_phantom(x: u32) -> i32 {{ 0 }} */\n",
        read(RS)
    );
    let p = fn_problems(&parse_rust_abi(&rs), &py, UNBOUND_EXPORTS);
    assert!(!mentions(&p, "tessl_phantom"), "{p:?}");
}

// ---------------------------------------------------------------------------
// Adversarial comment-stripping cases (tester round).

const PHANTOM_FN: &str = "pub extern \"C\" fn tessl_phantom(x: u32) -> i32 { 0 }";

#[test]
fn a_block_opener_inside_a_line_comment_does_not_swallow_a_real_export() {
    let (_, py) = real();
    let rs = format!("{}\n// see the /* legacy path\n{PHANTOM_FN}\n", read(RS));
    let p = fn_problems(&parse_rust_abi(&rs), &py, UNBOUND_EXPORTS);
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
}

#[test]
fn a_block_opener_inside_a_string_literal_does_not_swallow_a_real_export() {
    let (_, py) = real();
    let rs = format!("{}\nconst GLOB: &str = \"/*\";\n{PHANTOM_FN}\n", read(RS));
    let p = fn_problems(&parse_rust_abi(&rs), &py, UNBOUND_EXPORTS);
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
}

#[test]
fn a_line_comment_marker_inside_a_rust_string_does_not_hide_the_rest_of_the_line() {
    let (_, py) = real();
    let rs = format!("{}\nconst U: &str = \"http://x\"; {PHANTOM_FN}\n", read(RS));
    let p = fn_problems(&parse_rust_abi(&rs), &py, UNBOUND_EXPORTS);
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
}

#[test]
fn an_escaped_quote_before_a_hash_in_a_python_string_does_not_end_the_string() {
    let (rust, _) = real();
    let py = py_faulted(
        "lib.tessl_synchronize.restype = ctypes.c_int32\n",
        "lib.tessl_synchronize.restype = ctypes.c_int32\n        _note = \"a\\\"#b\"; lib.tessl_synchronize.restype = ctypes.c_int64\n",
    );
    let p = fn_problems(&rust, &py, UNBOUND_EXPORTS);
    assert!(mentions(&p, "tessl_synchronize: returns"), "{p:?}");
}

#[test]
fn crlf_line_endings_in_the_binding_are_read_like_lf() {
    let (rust, _) = real();
    let py = parse_py_abi(&read(PY).replace('\n', "\r\n"));
    assert!(fn_problems(&rust, &py, UNBOUND_EXPORTS).is_empty());
}

#[test]
fn multibyte_text_in_comments_does_not_break_either_stripper() {
    let (rust, py) = real();
    let rs = format!("/* \u{1F980} \u{00e9} */\n// \u{4e2d}\u{6587}\n{}", read(RS));
    let pys = format!("# \u{1F980} \u{00e9}\n{}  # \u{4e2d}\n", read(PY));
    assert!(fn_problems(&parse_rust_abi(&rs), &parse_py_abi(&pys), UNBOUND_EXPORTS).is_empty());
    let _ = (rust, py);
}

#[test]
fn a_docstring_that_looks_like_a_binding_is_not_live() {
    let (rust, _) = real();
    let py = py_faulted(
        "lib.tessl_synchronize.restype = ctypes.c_int32\n",
        "lib.tessl_synchronize.restype = ctypes.c_int32\n        \"\"\"\n        lib.tessl_synchronize.restype = ctypes.c_int64\n        \"\"\"\n",
    );
    assert!(fn_problems(&rust, &py, UNBOUND_EXPORTS).is_empty());
}

#[test]
fn a_char_literal_and_raw_string_do_not_confuse_the_rust_stripper() {
    let (_, py) = real();
    let rs = format!(
        "{}\nconst Q: char = '\\''; const R: &str = r#\"/* \"#; fn f<'a>(x: &'a u8) {{}} {PHANTOM_FN}\n",
        read(RS)
    );
    let p = fn_problems(&parse_rust_abi(&rs), &py, UNBOUND_EXPORTS);
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
}

// ---------------------------------------------------------------------------
// Adversarial comment-stripping cases (tester round 3).

fn rs_with(extra: &str) -> Vec<String> {
    let (_, py) = real();
    let rs = format!("{}\n{extra}\n", read(RS));
    fn_problems(&parse_rust_abi(&rs), &py, UNBOUND_EXPORTS)
}

#[test]
fn a_double_quote_char_literal_does_not_open_a_string() {
    let p = rs_with(&format!("const Q: char = '\"'; {PHANTOM_FN}"));
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
}

#[test]
fn a_comment_opener_inside_a_nested_block_comment_is_inert() {
    let p = rs_with(&format!("/* a /* b // c */ d */ {PHANTOM_FN}"));
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
}

#[test]
fn an_unterminated_block_comment_swallows_to_eof_without_panicking() {
    let p = rs_with(&format!("/* never closed {PHANTOM_FN}"));
    assert!(!mentions(&p, "tessl_phantom"), "{p:?}");
}

#[test]
fn a_raw_string_with_hashes_hides_inner_quotes_and_comment_markers() {
    let p = rs_with(&format!("const R: &str = r##\"a \"# /* \"##; {PHANTOM_FN}"));
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
}

#[test]
fn a_raw_identifier_is_not_a_raw_string() {
    let p = rs_with(&format!("fn f(r#type: u8) {{}} {PHANTOM_FN}"));
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
}

#[test]
fn a_raw_string_ending_in_a_backslash_does_not_escape_its_quote() {
    let p = rs_with(&format!("const R: &str = r\"\\\"; {PHANTOM_FN}"));
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
}

#[test]
fn a_byte_raw_string_ending_in_a_backslash_does_not_escape_its_quote() {
    let p = rs_with(&format!("const R: &[u8] = br\"\\\"; {PHANTOM_FN}"));
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
}

#[test]
fn a_lifetime_pair_does_not_pair_up_as_a_char_literal() {
    let p = rs_with(&format!("fn g<'a, 'b>(x: &'a u8, y: &'b u8) {{}} /* c */ {PHANTOM_FN}"));
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
}

#[test]
fn a_unicode_escape_char_literal_is_one_token() {
    let p = rs_with(&format!(
        "const C: char = '\\u{{1F980}}'; const D: char = '\u{4e2d}'; {PHANTOM_FN}"
    ));
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
}

#[test]
fn a_python_triple_single_quote_docstring_is_not_live() {
    let (rust, _) = real();
    let py = py_faulted(
        "lib.tessl_synchronize.restype = ctypes.c_int32\n",
        "lib.tessl_synchronize.restype = ctypes.c_int32\n        '''\n        lib.tessl_synchronize.restype = ctypes.c_int64\n        '''\n",
    );
    assert!(fn_problems(&rust, &py, UNBOUND_EXPORTS).is_empty());
}

#[test]
fn a_python_docstring_holding_the_other_triple_quote_stays_one_string() {
    let (rust, _) = real();
    let py = py_faulted(
        "lib.tessl_synchronize.restype = ctypes.c_int32\n",
        "lib.tessl_synchronize.restype = ctypes.c_int32\n        \"\"\" ''' \n        lib.tessl_synchronize.restype = ctypes.c_int64\n        \"\"\"\n",
    );
    assert!(fn_problems(&rust, &py, UNBOUND_EXPORTS).is_empty());
}

#[test]
fn a_python_docstring_with_an_escaped_quote_run_stays_one_string() {
    let (rust, _) = real();
    let py = py_faulted(
        "lib.tessl_synchronize.restype = ctypes.c_int32\n",
        "lib.tessl_synchronize.restype = ctypes.c_int32\n        \"\"\" \\\"\"\" \n        lib.tessl_synchronize.restype = ctypes.c_int64\n        \"\"\"\n",
    );
    assert!(fn_problems(&rust, &py, UNBOUND_EXPORTS).is_empty());
}

#[test]
fn an_unterminated_python_triple_quote_does_not_panic() {
    let py = format!("{}\n\"\"\"never closed", read(PY));
    let _ = parse_py_abi(&py);
}

#[test]
fn a_python_hash_inside_a_single_quoted_string_is_not_a_comment() {
    let (rust, _) = real();
    let py = py_faulted(
        "lib.tessl_synchronize.restype = ctypes.c_int32\n",
        "lib.tessl_synchronize.restype = ctypes.c_int32\n        _n = '#'; lib.tessl_synchronize.restype = ctypes.c_int64\n",
    );
    let p = fn_problems(&rust, &py, UNBOUND_EXPORTS);
    assert!(mentions(&p, "tessl_synchronize: returns"), "{p:?}");
}

#[test]
fn a_python_backslash_continued_string_keeps_its_hash_inside() {
    let (rust, _) = real();
    let py = py_faulted(
        "lib.tessl_synchronize.restype = ctypes.c_int32\n",
        "lib.tessl_synchronize.restype = ctypes.c_int32\n        _n = 'a\\\n#b'; lib.tessl_synchronize.restype = ctypes.c_int64\n",
    );
    let p = fn_problems(&rust, &py, UNBOUND_EXPORTS);
    assert!(mentions(&p, "tessl_synchronize: returns"), "{p:?}");
}

#[test]
fn a_byte_raw_string_hides_a_trailing_comment_and_what_follows() {
    // Read right, the raw string ends at its second quote and `// ...` is a
    // comment, so the phantom is hidden. Mis-lexed as a plain string it shows.
    let p = rs_with(&format!("const R: &[u8] = br\"\\\"; // \" {PHANTOM_FN}"));
    assert!(!mentions(&p, "tessl_phantom"), "{p:?}");
}

#[test]
fn an_identifier_ending_in_br_is_not_a_byte_raw_string() {
    // Read as `abr` then an ordinary string `"\"; // "`, the phantom after it is
    // live. Mis-lexed as the byte raw string `br"\"` the rest is a comment and
    // the phantom is hidden, so this fails if the identifier check is removed.
    let p = rs_with(&format!("const X: u8 = abr\"\\\"; // \" {PHANTOM_FN}"));
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
}

#[test]
fn an_identifier_ending_in_cr_is_not_a_c_raw_string() {
    let p = rs_with(&format!("const X: u8 = acr\"\\\"; // \" {PHANTOM_FN}"));
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
}

#[test]
fn a_c_raw_string_hides_a_trailing_comment_and_what_follows() {
    // `cr"\"` ends at its second quote, then `// ...` is a comment: phantom hidden.
    let p = rs_with(&format!("const R: &core::ffi::CStr = cr\"\\\"; // \" {PHANTOM_FN}"));
    assert!(!mentions(&p, "tessl_phantom"), "{p:?}");
}

#[test]
fn a_c_raw_string_with_hashes_hides_inner_quotes() {
    let p = rs_with(&format!("const R: &core::ffi::CStr = cr##\"\"# /* \"##; {PHANTOM_FN}"));
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
    let q = rs_with(&format!("const R: &core::ffi::CStr = cr#\"\\\"#; /* {PHANTOM_FN} */"));
    assert!(!mentions(&q, "tessl_phantom"), "{q:?}");
}

#[test]
fn a_c_string_is_an_ordinary_string_with_escapes() {
    // `c"\"; // "` is one string (the escaped quote does not end it), so the
    // phantom after it is live.
    let p = rs_with(&format!("const R: &core::ffi::CStr = c\"\\\"; // \"; {PHANTOM_FN}"));
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
}

#[test]
fn a_layout_keyword_after_a_hash_inside_a_string_is_still_code() {
    let src = "class _A(ctypes.Structure):\n    _n = '#'; _pack_ = 1\n";
    assert!(layout_guard_problem(src).is_some());
}

#[test]
fn the_real_sources_survive_stripping_idempotently() {
    for (src, f) in [
        (read(RS), strip_line_comments as fn(&str) -> String),
        (read(PY), strip_py_comments as fn(&str) -> String),
    ] {
        let once = f(&src);
        assert_eq!(f(&once), once, "stripping is not idempotent");
    }
}

// ---- tester round 4: adversarial lexer cases ----

#[test]
fn a_byte_char_quote_does_not_open_a_string() {
    let p = rs_with(&format!("const Q: u8 = b'\"'; /* c */ {PHANTOM_FN}"));
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
}

#[test]
fn a_raw_identifier_with_a_block_comment_keeps_the_phantom_visible() {
    let p = rs_with(&format!("fn r#type() {{}} /* c */ {PHANTOM_FN}"));
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
}

#[test]
fn a_block_comment_hides_a_quote_and_nests() {
    let p = rs_with(&format!("/* \" /* \" */ \" */ {PHANTOM_FN}"));
    assert!(mentions(&p, "tessl_phantom: in src/capi.rs"), "{p:?}");
}

#[test]
fn an_unterminated_rust_block_comment_does_not_panic() {
    for s in ["/*", "/* /*", "/", "r#\"", "\"", "'", "'\\", "br", "br#", "b'"] {
        let _ = strip_line_comments(s);
        let _ = strip_py_comments(s);
    }
}

#[test]
fn a_python_raw_string_ending_in_an_escaped_quote_stays_open() {
    // r"a\"" is the string a\" ; the trailing comment must not hide the next line.
    let src = "x = r\"a\\\"\" # c\n_pack_ = 1\n";
    assert!(layout_guard_problem(src).is_some());
}

#[test]
fn a_python_comment_hides_a_layout_keyword() {
    assert!(layout_guard_problem("x = 1  # _pack_ = 1\n").is_none());
    assert!(layout_guard_problem("x = '''\n_pack_\n'''\n").is_none());
}

#[test]
fn a_python_layout_keyword_after_a_closed_string_is_code() {
    assert!(layout_guard_problem("x = 'a' ; _pack_ = 1\n").is_some());
    assert!(layout_guard_problem("x = '' ; _anonymous_ = []\n").is_some());
    assert!(layout_guard_problem("x = \"\"\"d\"\"\"; _pack_ = 1\n").is_some());
}

#[test]
fn python_stripping_keeps_line_count() {
    for s in [
        "a # b\nc\n",
        "'''x\ny\nz'''\nw\n",
        "'a\\\nb'\nc\n",
        "x = '''a\\\nb'''\n",
    ] {
        assert_eq!(
            strip_py_comments(s).matches('\n').count(),
            s.matches('\n').count(),
            "{s:?}"
        );
    }
}

#[test]
fn rust_stripping_keeps_line_count_and_is_idempotent() {
    for s in [
        "a // b\nc\n",
        "/* x\ny\n*/ z\n",
        "\"a\nb\" // c\n",
        "r#\"a\nb\"# /* c\n*/\n",
    ] {
        let once = strip_line_comments(s);
        assert_eq!(once.matches('\n').count(), s.matches('\n').count(), "{s:?}");
        assert_eq!(strip_line_comments(&once), once, "{s:?}");
    }
}

#[test]
fn this_file_never_creates_a_runtime() {
    // The contract file is GPU-free: a runtime is a Metal device. Runtime
    // creation lives in tests/capi.rs. The needles are assembled so this
    // guard's own text does not contain them.
    let src = read("tests/capi_contract.rs");
    for needle in [["tessl_runtime", "_new"].concat(), ["GpuRuntime", "::new"].concat()] {
        let hits: Vec<_> = src
            .lines()
            .enumerate()
            .filter(|(_, l)| l.contains(&needle) && !l.trim_start().starts_with("//"))
            .map(|(n, _)| n + 1)
            .collect();
        assert!(hits.is_empty(), "{needle} used at lines {hits:?}");
    }
}

// ---- adversarial: generated token streams with a known expected output ----

/// A tiny deterministic generator (no dependency).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % n
    }
}

/// One Rust token as (source text, text the stripper must keep).
fn rust_token(g: &mut Lcg) -> (String, String) {
    let keep = |s: String| (s.clone(), s);
    match g.next(13) {
        0 => keep("ident_x".to_string()),
        1 => keep(["abr", "acr", "xr", "_r", "bar", "for"][g.next(6)].to_string()),
        2 => {
            let h = "#".repeat(g.next(4));
            let body = ["//", "/*", "\"", "\\", " \" // ", "\"#", "x"][g.next(7)];
            // A body may hold `"` only if the terminator needs more hashes.
            let body = if h.len() < 2 && body.contains('"') { "x" } else { body };
            keep(format!("r{h}\"{body}\"{h}"))
        }
        3 => {
            let h = "#".repeat(1 + g.next(3));
            keep(format!("br{h}\"a // b \\\"{h}"))
        }
        4 => {
            let h = "#".repeat(1 + g.next(3));
            keep(format!("cr{h}\"a /* b \\\"{h}"))
        }
        5 => keep(r#""a // \" /* b""#.to_string()),
        6 => keep(r#"b"x\"//""#.to_string()),
        7 => keep(r#"c"x\"//""#.to_string()),
        8 => keep(
            [
                "'\"'",
                "'\\''",
                "'\\\\'",
                "'x'",
                "b'\"'",
                "'\\u{1F600}'",
                "'é'",
                "&'a str",
            ][g.next(8)]
            .to_string(),
        ),
        9 => (
            format!("// {} \" r#\" /*\n", ["x", "'", "\\"][g.next(3)]),
            "\n".to_string(),
        ),
        10 => ("/* a \" /* nested // */ \" */".to_string(), String::new()),
        11 => keep("r#type".to_string()),
        _ => keep(["+", "{", "}", "::", "0x1r"][g.next(5)].to_string()),
    }
}

#[test]
fn generated_rust_token_streams_strip_to_exactly_their_non_comment_text() {
    let mut g = Lcg(0x5eed);
    for case in 0..4000 {
        let (mut src, mut want) = (String::new(), String::new());
        for _ in 0..1 + g.next(8) {
            let (s, k) = rust_token(&mut g);
            src.push_str(&s);
            want.push_str(&k);
            // A separator keeps adjacent tokens from fusing into one.
            src.push(' ');
            want.push(' ');
        }
        assert_eq!(strip_line_comments(&src), want, "case {case}: {src:?}");
    }
}

/// One Python token as (source text, text the stripper must keep).
fn py_token(g: &mut Lcg) -> (String, String) {
    let keep = |s: String| (s.clone(), s);
    match g.next(8) {
        0 => keep("name_1".to_string()),
        1 => keep(
            [
                "'a # b'",
                "\"a # b\"",
                "r'a # \\' # b'",
                "b\"\\\"#\"",
                "f'{x} # y'",
                "''",
                "\"\"",
            ][g.next(7)]
            .to_string(),
        ),
        2 => ("# a ' \" '''\n".to_string(), "\n".to_string()),
        3 => ("\"\"\"doc ' # \\\"\"\" \"\"\"".to_string(), String::new()),
        4 => ("'''doc \" # '''".to_string(), String::new()),
        5 => keep("1 + 2".to_string()),
        6 => keep("'it\\'s # not a comment'".to_string()),
        _ => keep("(".to_string()),
    }
}

#[test]
fn generated_python_token_streams_strip_to_exactly_their_non_comment_text() {
    let mut g = Lcg(0xfeed);
    for case in 0..4000 {
        let (mut src, mut want) = (String::new(), String::new());
        for _ in 0..1 + g.next(8) {
            let (s, k) = py_token(&mut g);
            src.push_str(&s);
            want.push_str(&k);
            src.push('\n');
            want.push('\n');
        }
        // A comment token already ends in a newline, so the separator adds one.
        assert_eq!(strip_py_comments(&src), want, "case {case}: {src:?}");
    }
}

// ---------------------------------------------------------------------------
// The EmbeddingGemma 2 entries live in their own module and Python file, so the
// tests above (which read `RS` and `PY` alone) never see them. These hold the
// pair to the same rules, with the same injected faults, so a mismatch there
// cannot pass for want of being looked at.

const RS_EG2: &str = "src/capi_embedgemma2.rs";
const PY_EG2: &str = "python/tessl_torch/embedgemma2.py";

fn eg2_abi() -> (RustAbi, PyAbi) {
    (parse_rust_abi(&read(RS_EG2)), parse_py_abi(&read(PY_EG2)))
}

const EG2_FNS: [&str; 6] = [
    "tessl_embedgemma2_load",
    "tessl_embedgemma2_free",
    "tessl_embedgemma2_embedding_dim",
    "tessl_embedgemma2_max_tokens",
    "tessl_embedgemma2_set_max_tokens",
    "tessl_embedgemma2_encode",
];

#[test]
fn every_embedgemma2_c_function_is_declared_in_python_with_the_same_arity_and_widths() {
    let (rust, py) = eg2_abi();
    // A parser that quietly found nothing would pass vacuously.
    assert_eq!(
        rust.fns.keys().map(String::as_str).collect::<Vec<_>>().len(),
        EG2_FNS.len()
    );
    for name in EG2_FNS {
        assert!(rust.fns.contains_key(name), "{name} not parsed from {RS_EG2}");
        assert!(py.fns.contains_key(name), "{name} not parsed from {PY_EG2}");
    }
    let problems = fn_problems(&rust, &py, &[]);
    assert!(problems.is_empty(), "C ABI / ctypes mismatch:\n{}", problems.join("\n"));
}

#[test]
fn an_embedgemma2_signature_that_drifts_on_either_side_is_reported() {
    let (rust, py) = eg2_abi();
    let py_src = read(PY_EG2);
    let rs_src = read(RS_EG2);

    // A narrower restype, a narrower length, a dropped argument.
    let p = fn_problems(
        &rust,
        &parse_py_abi(&fault(
            &py_src,
            "lib.tessl_embedgemma2_encode.restype = ctypes.c_int32",
            "lib.tessl_embedgemma2_encode.restype = ctypes.c_int64",
        )),
        &[],
    );
    assert!(mentions(&p, "tessl_embedgemma2_encode: returns"), "{p:?}");
    let p = fn_problems(
        &rust,
        &parse_py_abi(&fault(
            &py_src,
            "ctypes.c_void_p, u32p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,",
            "ctypes.c_void_p, u32p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,",
        )),
        &[],
    );
    assert!(mentions(&p, "tessl_embedgemma2_encode: argument 2 `n_ids`"), "{p:?}");
    let p = fn_problems(
        &rust,
        &parse_py_abi(&fault(
            &py_src,
            "lib.tessl_embedgemma2_set_max_tokens.argtypes = [ctypes.c_void_p, ctypes.c_uint32] + err_args",
            "lib.tessl_embedgemma2_set_max_tokens.argtypes = [ctypes.c_void_p] + err_args",
        )),
        &[],
    );
    assert!(mentions(&p, "tessl_embedgemma2_set_max_tokens: C takes"), "{p:?}");

    // The same on the Rust side.
    let p = fn_problems(
        &parse_rust_abi(&fault(&rs_src, "    n_seqs: u64,\n", "    n_seqs: u32,\n")),
        &py,
        &[],
    );
    assert!(mentions(&p, "tessl_embedgemma2_encode: argument 4 `n_seqs`"), "{p:?}");
    let p = fn_problems(
        &parse_rust_abi(&fault(
            &rs_src,
            "    truncate_dim: u32,\n",
            "    truncate_dim: u64,\n",
        )),
        &py,
        &[],
    );
    assert!(mentions(&p, "tessl_embedgemma2_encode: argument 5 `truncate_dim`"), "{p:?}");

    // An export with no declaration, and a declaration with no export.
    let phantom = "pub unsafe extern \"C\" fn tessl_embedgemma2_phantom(x: u32) -> i32 { 0 }";
    let p = fn_problems(&parse_rust_abi(&format!("{rs_src}\n{phantom}\n")), &py, &[]);
    assert!(mentions(&p, "tessl_embedgemma2_phantom: in src/capi.rs"), "{p:?}");
    let p = fn_problems(
        &rust,
        &parse_py_abi(&fault(
            &py_src,
            "        lib.tessl_embedgemma2_free.restype = ctypes.c_int32\n",
            "        lib.tessl_embedgemma2_free.restype = ctypes.c_int32\n        lib.tessl_embedgemma2_ghost.restype = ctypes.c_int32\n        lib.tessl_embedgemma2_ghost.argtypes = [ctypes.c_void_p]\n",
        )),
        &[],
    );
    assert!(
        mentions(&p, "tessl_embedgemma2_ghost: declared in the Python binding"),
        "{p:?}"
    );
}

/// Every module that exports `tessl_*` entry points, found by scanning `src/`.
/// A third one would be checked by neither contract above, so adding one must
/// start here.
#[test]
fn only_the_two_known_modules_export_c_entry_points() {
    fn walk(dir: &std::path::Path, out: &mut Vec<String>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs")
                && strip_line_comments(&std::fs::read_to_string(&p).unwrap()).contains("extern \"C\" fn tessl_")
            {
                out.push(
                    p.strip_prefix(env!("CARGO_MANIFEST_DIR"))
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
    }
    let mut found = Vec::new();
    walk(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut found,
    );
    found.sort();
    assert_eq!(found, [RS, RS_EG2]);
}
