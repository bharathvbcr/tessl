//! The C ABI's handle contract: null handles, foreign threads, freed handles.
//!
//! The null-handle tests create no runtime and need no GPU. The foreign-thread
//! test needs one real runtime handle (a Metal device). The aliasing the
//! thread check used to cause (a `&mut` formed on a foreign thread while the
//! owner may be inside a call) is undefined behaviour Miri cannot exercise
//! through Metal, so what is pinned here is the observable half: the refusal
//! is a status, carries the thread message before any argument is looked at,
//! and leaves the handle usable.
//!
//! A freed handle is not tested for: the ABI keeps no registry of live
//! handles, so a call on a freed pointer is undefined behaviour by contract
//! (documented in `src/capi.rs`), and a test would have to commit it.

mod common;

use std::ffi::{c_char, CStr};
use std::ptr;

use tessl::capi::{
    tessl_cross_entropy_rows, tessl_gdn_train_backward, tessl_gdn_train_forward, tessl_qwen35_adamw_free,
    tessl_qwen35_adamw_set_step_count, tessl_qwen35_adamw_step_count, tessl_qwen35_free, tessl_qwen35_grad_sq_norm,
    tessl_qwen35_load, tessl_qwen35_param_count, tessl_qwen35_train_discard, tessl_qwen35_train_step,
    tessl_runtime_free, tessl_runtime_new, tessl_runtime_poisoned, tessl_synchronize, TesslQwen35, TesslRuntime,
    TESSL_ABI_VERSION, TESSL_ERR, TESSL_F32, TESSL_OK,
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

#[test]
fn every_runtime_entry_refuses_a_foreign_thread_before_looking_at_its_arguments() {
    let _gpu = common::gpu_lock();
    let mut err = [0 as c_char; ERR_LEN];
    let rt = unsafe { tessl_runtime_new(err.as_mut_ptr(), ERR_LEN) };
    assert!(!rt.is_null(), "{}", msg(&err));
    let raw = rt as usize;
    // The arguments are all null: were they looked at first, the message
    // would be "null args", not the thread's.
    let calls = std::thread::spawn(move || runtime_entries(raw as *mut TesslRuntime))
        .join()
        .unwrap();
    assert_eq!(calls.len(), 6);
    for (name, status, message) in &calls {
        assert_eq!(*status, TESSL_ERR, "{name}");
        assert!(
            message.contains("other than the one that created it"),
            "{name}: {message:?}"
        );
    }
    // A foreign free leaks rather than drops, and the owner still works.
    let freed = std::thread::spawn(move || unsafe { tessl_runtime_free(raw as *mut TesslRuntime) })
        .join()
        .unwrap();
    assert_eq!(freed, TESSL_ERR);
    assert_eq!(
        unsafe { tessl_synchronize(rt, err.as_mut_ptr(), ERR_LEN) },
        TESSL_OK,
        "{}",
        msg(&err)
    );
    assert_eq!(unsafe { tessl_runtime_free(rt) }, TESSL_OK);
}

#[test]
fn the_python_binding_expects_this_abi_version() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/python/tessl_torch/__init__.py")).unwrap();
    let line = src
        .lines()
        .find(|l| l.starts_with("_ABI_VERSION = "))
        .expect("_ABI_VERSION in the binding");
    assert_eq!(line, format!("_ABI_VERSION = {TESSL_ABI_VERSION}"));
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

fn strip_line_comments(src: &str) -> String {
    src.lines()
        .map(|l| l.find("//").map_or(l, |i| &l[..i]))
        .collect::<Vec<_>>()
        .join("\n")
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
    /// name -> (argtypes, restype), as declared.
    fns: BTreeMap<String, (Vec<Ty>, Ty)>,
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
    let src: &str = &inline_aliases(src);
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
    let fns = fns
        .into_iter()
        .map(|(n, (a, r))| {
            let r = r.unwrap_or_else(|| panic!("{n}: no restype (ctypes would assume a C int)"));
            (n, (a, r))
        })
        .collect();
    PyAbi { fns, structs }
}

fn read(path: &str) -> String {
    std::fs::read_to_string(format!("{}/{path}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

#[test]
fn every_c_function_is_declared_in_python_with_the_same_arity_and_widths() {
    let rust = parse_rust_abi(&read("src/capi.rs"));
    let py = parse_py_abi(&read("python/tessl_torch/__init__.py"));
    // A parser that quietly found nothing would pass vacuously.
    assert!(rust.fns.len() >= 26, "parsed only {} C functions", rust.fns.len());
    assert!(rust.fns.contains_key("tessl_runtime_poisoned"));

    let mut problems = Vec::new();
    for name in rust.fns.keys().filter(|k| !py.fns.contains_key(*k)) {
        problems.push(format!("{name}: in src/capi.rs, not declared in the Python binding"));
    }
    for name in py.fns.keys().filter(|k| !rust.fns.contains_key(*k)) {
        problems.push(format!("{name}: declared in the Python binding, not in src/capi.rs"));
    }
    for (name, sig) in &rust.fns {
        let Some((pargs, pret)) = py.fns.get(name) else {
            continue;
        };
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
        if !sig.ret.agrees(pret) {
            problems.push(format!("{name}: returns {:?} in C, restype is {pret:?}", sig.ret));
        }
    }
    assert!(problems.is_empty(), "C ABI / ctypes mismatch:\n{}", problems.join("\n"));
}

#[test]
fn every_repr_c_struct_matches_its_ctypes_structure_field_for_field() {
    let rust = parse_rust_abi(&read("src/capi.rs"));
    let py = parse_py_abi(&read("python/tessl_torch/__init__.py"));
    assert!(rust.structs.len() >= 4, "parsed only {} structs", rust.structs.len());
    let rust_names: Vec<_> = rust.structs.keys().collect();
    let py_names: Vec<_> = py.structs.keys().collect();
    assert_eq!(rust_names, py_names, "#[repr(C)] structs vs ctypes Structures");
    let mut problems = Vec::new();
    for (name, rfields) in &rust.structs {
        let pfields = &py.structs[name];
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
    assert!(problems.is_empty(), "struct layout mismatch:\n{}", problems.join("\n"));
}
