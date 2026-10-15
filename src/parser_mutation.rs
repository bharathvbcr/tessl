//! Seeded mutation tests for the three hand-written parsers of downloaded
//! files: `json` (config.json, safetensors headers), `safetensors` and `npy`.
//!
//! Decision (card ft-04c475ac, AC1): a dependency-free, deterministic mutation
//! test rather than cargo-fuzz or a proptest dev-dependency. The repo takes no
//! new dependency for this, and a failing case replays from its seed and
//! mutation description alone. It lives in-crate because `json` is private;
//! it constructs no `GpuRuntime`, so it runs on hosted CI (the `Host-only unit
//! tests` step filters on `parser_mutation::`).
//!
//! Property, for every input: the parser returns `Ok` or a non-empty `Err`,
//! never panics, and no single allocation exceeds [`alloc_bound`] (a function
//! of the input length only, so a header field cannot size an allocation).
//! Stronger invariants are asserted where they hold: a strict truncation of a
//! valid file is always refused, and `SafeTensors::open` agrees with
//! `from_bytes` on the same bytes.
//!
//! Mutations: truncation at every byte, every single-bit flip, every position
//! overwritten with boundary integers (u16/u32/u64 extremes, input length +-1),
//! seeded random byte/range edits, plus hand-built hostile headers (length
//! fields at 0/1/len-1/len+1/u64::MAX, reversed/overlapping/overflowing
//! offsets, overflowing shapes, bad npy magic/version, lone surrogates, 100k
//! nesting, huge exponents, duplicate keys, trailing garbage).
//!
//! Tunables: `PARSER_MUTATION_SEED` (default fixed) and `PARSER_MUTATION_CASES`
//! (random cases per corpus entry, default 1500).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;

use crate::json::{self, Syntax};
use crate::npy::read_npy;
use crate::safetensors::{Dtype, SafeTensors};

// ---------------------------------------------------------------- allocation

thread_local! {
    static MAX_ALLOC: Cell<usize> = const { Cell::new(0) };
}

fn note(size: usize) {
    let _ = MAX_ALLOC.try_with(|m| m.set(m.get().max(size)));
}

/// Records, per thread, the largest single allocation requested.
struct Tracking;

// SAFETY: every method delegates to `System` with the arguments it was given;
// the only addition is a thread-local counter update.
unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        note(l.size());
        System.alloc(l)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        note(l.size());
        System.alloc_zeroed(l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        note(new_size);
        System.realloc(p, l, new_size)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
}

#[global_allocator]
static ALLOC: Tracking = Tracking;

/// The stated bound on any single allocation while parsing an input of `len`
/// bytes: 64 bytes per input byte (a `Json` node costs tens of bytes and
/// consumes at least two input bytes, `Vec` doubles) plus 8 KiB of slack. A
/// parser sizing a buffer from a header field would exceed this on the
/// hostile inputs below.
fn alloc_bound(len: usize) -> usize {
    64 * len + 8192
}

// ----------------------------------------------------------------------- rng

/// splitmix64: tiny, deterministic, good enough to pick mutation sites.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn env_u64(key: &str, default: u64) -> u64 {
    match std::env::var(key) {
        Ok(v) => {
            let v = v.trim();
            let parsed = match v.strip_prefix("0x") {
                Some(h) => u64::from_str_radix(h, 16),
                None => v.parse(),
            };
            parsed.unwrap_or_else(|e| panic!("{key}={v:?} is not an integer: {e}"))
        }
        Err(_) => default,
    }
}

fn seed() -> u64 {
    env_u64("PARSER_MUTATION_SEED", 0x7e55_1ab5_eed5)
}

fn cases() -> usize {
    env_u64("PARSER_MUTATION_CASES", 1500) as usize
}

// ------------------------------------------------------------------ mutation

/// Boundary integers written over every position.
fn interesting(len: usize) -> Vec<u64> {
    let l = len as u64;
    vec![
        0,
        1,
        2,
        l.wrapping_sub(1),
        l,
        l + 1,
        0x7f,
        0x80,
        0xff,
        0xffff,
        0x1_0000,
        0x7fff_ffff,
        0x8000_0000,
        0xffff_ffff,
        0x1_0000_0000,
        crate::safetensors::MAX_HEADER_BYTES,
        crate::safetensors::MAX_HEADER_BYTES + 1,
        i64::MAX as u64,
        1 << 63,
        u64::MAX - 1,
        u64::MAX,
    ]
}

/// Every mutation of `seed`, with a label that replays it. Exhaustive passes
/// first (truncations, bit flips, boundary-integer overwrites), then `n`
/// seeded random edits.
fn mutations(seed_bytes: &[u8], rng_seed: u64, n: usize) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    for cut in 0..seed_bytes.len() {
        out.push((format!("truncate to {cut}"), seed_bytes[..cut].to_vec()));
    }
    for pos in 0..seed_bytes.len() {
        for bit in 0..8 {
            let mut b = seed_bytes.to_vec();
            b[pos] ^= 1 << bit;
            out.push((format!("flip bit {bit} of byte {pos}"), b));
        }
    }
    for v in interesting(seed_bytes.len()) {
        for width in [1usize, 2, 4, 8] {
            if width < 8 && v >> (8 * width) != 0 {
                continue;
            }
            for pos in 0..seed_bytes.len().saturating_sub(width - 1) {
                let mut b = seed_bytes.to_vec();
                b[pos..pos + width].copy_from_slice(&v.to_le_bytes()[..width]);
                out.push((format!("write {v:#x} ({width} bytes LE) at {pos}"), b));
            }
        }
    }
    let mut rng = Rng(rng_seed);
    for i in 0..n {
        let mut b = seed_bytes.to_vec();
        let edits = 1 + rng.below(4);
        let mut label = format!("random#{i} seed={rng_seed:#x}:");
        for _ in 0..edits {
            if b.is_empty() {
                break;
            }
            let pos = rng.below(b.len());
            match rng.below(5) {
                0 => {
                    let v = rng.next() as u8;
                    b[pos] = v;
                    label += &format!(" set[{pos}]={v:#x}");
                }
                1 => {
                    let end = (pos + 1 + rng.below(16)).min(b.len());
                    b.drain(pos..end);
                    label += &format!(" delete[{pos}..{end}]");
                }
                2 => {
                    let v = rng.next() as u8;
                    b.insert(pos, v);
                    label += &format!(" insert[{pos}]={v:#x}");
                }
                3 => {
                    let end = (pos + 1 + rng.below(32)).min(b.len());
                    let chunk = b[pos..end].to_vec();
                    let at = rng.below(b.len());
                    b.splice(at..at, chunk);
                    label += &format!(" dup[{pos}..{end}]@{at}");
                }
                _ => {
                    let vals = interesting(b.len());
                    let v = vals[rng.below(vals.len())];
                    let w = (1 + rng.below(8)).min(b.len() - pos);
                    b[pos..pos + w].copy_from_slice(&v.to_le_bytes()[..w]);
                    label += &format!(" put[{pos};{w}]={v:#x}");
                }
            }
        }
        out.push((label, b));
    }
    out
}

// -------------------------------------------------------------------- driver

/// What one parse of one input produced.
struct Outcome {
    ok: bool,
    /// The parser's error text, when it returned one.
    err: Option<String>,
    /// Any property violation (panic, empty error, oversized allocation).
    violations: Vec<String>,
}

/// Why a checked closure stopped. A parser refusing its input is expected;
/// two parser entry points disagreeing, or a read-back contradicting the
/// header the parser accepted, is a defect and must fail the test rather than
/// pass as one more refusal.
enum Fail {
    /// The parser returned a named `Err` for the input.
    Refused(String),
    /// An internal-consistency check failed on an input the parser accepted.
    Inconsistent(String),
}

impl From<String> for Fail {
    fn from(e: String) -> Self {
        Fail::Refused(e)
    }
}

/// Run `f` on `input` under catch_unwind and the allocation tracker.
fn run<E: Into<Fail>>(input_len: usize, f: impl FnOnce() -> Result<(), E>) -> Outcome {
    MAX_ALLOC.with(|m| m.set(0));
    let r = catch_unwind(AssertUnwindSafe(|| f().map_err(Into::into)));
    let max = MAX_ALLOC.with(Cell::get);
    let mut violations = Vec::new();
    let mut err = None;
    let ok = match r {
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            violations.push(format!("panic: {msg}"));
            false
        }
        Ok(Ok(())) => true,
        Ok(Err(Fail::Inconsistent(msg))) => {
            violations.push(format!("inconsistent: {msg}"));
            false
        }
        Ok(Err(Fail::Refused(e))) => {
            if e.trim().is_empty() {
                violations.push("Err with an empty message".into());
            }
            err = Some(e);
            false
        }
    };
    if max > alloc_bound(input_len) {
        violations.push(format!(
            "single allocation of {max} bytes exceeds the bound {} for a {input_len}-byte input",
            alloc_bound(input_len)
        ));
    }
    Outcome { ok, err, violations }
}

/// Collects failures across a pass so one run reports many, not the first.
#[derive(Default)]
struct Failures(Vec<String>);

impl Failures {
    fn add(&mut self, what: &str, label: &str, msg: String) {
        self.0.push(format!("[{what}] {label}: {msg}"));
    }
    fn finish(self, total: usize) {
        if self.0.is_empty() {
            return;
        }
        let shown: Vec<&str> = self.0.iter().take(12).map(String::as_str).collect();
        panic!(
            "{} of {total} mutated inputs violated the parser contract (seed {:#x}); first {}:\n{}",
            self.0.len(),
            seed(),
            shown.len(),
            shown.join("\n")
        );
    }
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("tessl-parser-mutation-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let p = self.0.join(name);
        std::fs::write(&p, bytes).unwrap();
        p
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// --------------------------------------------------------------------- json

const CONFIG_SYNTAX: Syntax = Syntax {
    what: "config.json",
    max_depth: 16,
    uints_only: false,
    literals: true,
};

const HEADER_SYNTAX: Syntax = Syntax {
    what: "header JSON",
    max_depth: 3,
    uints_only: true,
    literals: false,
};

/// A config.json shaped like a real Qwen3.5 text config: nested objects, int,
/// float and exponent numbers, booleans, null, arrays, escapes.
const QWEN_CONFIG: &str = r#"{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "tie_word_embeddings": true,
  "eos_token_id": null,
  "note": "caf\u00e9 \ud83d\ude00 \"quoted\" \\ \/ \n\t",
  "text_config": {
    "hidden_size": 2048, "num_hidden_layers": 24, "vocab_size": 248320,
    "rms_norm_eps": 1e-06, "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25,
      "mrope_section": [11, 11, 10]},
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "initializer_range": 0.02, "use_cache": false, "scale": -1.5E+3
  }
}"#;

const BERT_CONFIG: &str = r#"{"hidden_size":384,"num_attention_heads":12,"layer_norm_eps":1e-12,"hidden_act":"gelu","position_embedding_type":"absolute","id2label":{"0":"a","1":"b"},"x":[[],[{}],[0.5,-0,2e2]]}"#;

fn check_json(input: &[u8], full_len: Option<usize>) -> Outcome {
    let input = input.to_vec();
    let mut o = run(input.len(), || {
        // Callers hand the parser a &str, so invalid UTF-8 never reaches it;
        // the lossy form (U+FFFD in place of bad bytes) is the nearest valid text.
        let text = String::from_utf8_lossy(&input);
        for syntax in [CONFIG_SYNTAX, HEADER_SYNTAX] {
            json::parse(&text, syntax)?;
        }
        Ok::<(), String>(())
    });
    // A strict truncation of a document whose root is an object is not itself
    // a document (the closing brace is last), under either syntax.
    if let Some(full) = full_len {
        if input.len() < full && o.ok {
            o.violations.push("a strict truncation of the document parsed".into());
        }
    }
    o
}

#[test]
fn json_parser_survives_mutation() {
    let mut f = Failures::default();
    let mut total = 0;
    for (name, text) in [("qwen", QWEN_CONFIG), ("bert", BERT_CONFIG)] {
        let seed_bytes = text.as_bytes();
        assert!(
            json::parse(text, CONFIG_SYNTAX).is_ok(),
            "the {name} seed config must parse"
        );
        let trimmed = text.trim_end().len();
        for (label, b) in mutations(seed_bytes, seed() ^ name.len() as u64, cases()) {
            total += 1;
            // Truncation never makes a valid document; trailing whitespace is not
            // part of the root, so cuts into it still parse and are exempt.
            let full = (label.starts_with("truncate")).then_some(trimmed);
            for v in check_json(&b, full).violations {
                f.add(name, &label, v);
            }
        }
    }
    f.finish(total);
}

/// Parse `text` with the tracker, expecting `Err` containing `needle`.
fn expect_json_err(label: &str, text: &str, syntax: Syntax, needle: &str) {
    let o = run(text.len(), || json::parse(text, syntax).map(|_| ()));
    assert!(o.violations.is_empty(), "{label}: {:?}", o.violations);
    let e = o.err.unwrap_or_else(|| panic!("{label}: parsed, expected an error"));
    assert!(e.contains(needle), "{label}: error {e:?} lacks {needle:?}");
}

#[test]
fn json_hostile_documents_are_named_errors() {
    let deep_arrays = "[".repeat(100_000);
    expect_json_err("100k [", &deep_arrays, CONFIG_SYNTAX, "nesting deeper");
    let deep_objects = "{\"a\":".repeat(100_000);
    expect_json_err("100k {\"a\":", &deep_objects, CONFIG_SYNTAX, "nesting deeper");
    // Properly closed deep nesting is refused too, not only unterminated.
    let closed = format!("{}{}", "[".repeat(50_000), "]".repeat(50_000));
    expect_json_err("closed 50k nesting", &closed, CONFIG_SYNTAX, "nesting deeper");
    expect_json_err("lone low surrogate", r#"{"a":"\udc00"}"#, CONFIG_SYNTAX, "unpaired low");
    expect_json_err(
        "lone high surrogate",
        r#"{"a":"\ud800"}"#,
        CONFIG_SYNTAX,
        "unpaired high",
    );
    expect_json_err(
        "high surrogate then non-surrogate",
        "{\"a\":\"\\ud800\\u0041\"}",
        CONFIG_SYNTAX,
        "invalid low surrogate",
    );
    expect_json_err(
        "huge exponent",
        r#"{"a":1e999999999999}"#,
        CONFIG_SYNTAX,
        "overflows f64",
    );
    expect_json_err("exponent with no digits", r#"{"a":1e}"#, CONFIG_SYNTAX, "exponent");
    expect_json_err("duplicate key", r#"{"a":1,"a":2}"#, CONFIG_SYNTAX, "duplicate key");
    expect_json_err("trailing garbage", r#"{"a":1} x"#, CONFIG_SYNTAX, "trailing bytes");
    expect_json_err("two roots", r#"{"a":1}{"b":2}"#, CONFIG_SYNTAX, "trailing bytes");
    expect_json_err(
        "control char in string",
        "{\"a\":\"x\ty\"}",
        CONFIG_SYNTAX,
        "control character",
    );
    expect_json_err("leading zero", r#"{"a":01}"#, CONFIG_SYNTAX, "leading zero");
    expect_json_err(
        "u64 overflow in header",
        r#"{"a":18446744073709551616}"#,
        HEADER_SYNTAX,
        "overflows u64",
    );
    expect_json_err("empty input", "", CONFIG_SYNTAX, "unexpected end");
    // A long digit run and a long string are linear in the input, not a blowup.
    let digits = format!("{{\"a\":{}}}", "9".repeat(200_000));
    let o = run(digits.len(), || json::parse(&digits, CONFIG_SYNTAX).map(|_| ()));
    assert!(o.violations.is_empty(), "{:?}", o.violations);
    let long_string = format!("{{\"a\":\"{}\"}}", "\u{1F600}".repeat(50_000));
    let o = run(long_string.len(), || {
        json::parse(&long_string, CONFIG_SYNTAX).map(|_| ())
    });
    assert!(o.ok && o.violations.is_empty(), "{:?}", o.violations);
    // Huge negative exponent underflows to zero, which is a valid f64.
    let o = run(24, || json::parse(r#"{"a":1e-999999999}"#, CONFIG_SYNTAX).map(|_| ()));
    assert!(o.ok && o.violations.is_empty(), "{:?}", o.violations);
}

// -------------------------------------------------------------- safetensors

/// Serialize a header text plus data buffer with the 8-byte length prefix.
fn st_bytes(header: &str, data_len: usize) -> Vec<u8> {
    let mut b = (header.len() as u64).to_le_bytes().to_vec();
    b.extend_from_slice(header.as_bytes());
    b.extend((0..data_len).map(|i| (i * 7 + 3) as u8));
    b
}

/// Four tensors of different dtypes plus metadata, tiling a 53-byte buffer:
/// f32 [2,3] at 0..24, bf16 [4] at 24..32, u8 [5] at 32..37, i64 [2] at 37..53.
fn st_seed() -> Vec<u8> {
    let h = r#"{"__metadata__":{"format":"pt"},"a.f32":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"b.bf16":{"dtype":"BF16","shape":[4],"data_offsets":[24,32]},"c.u8":{"dtype":"U8","shape":[5],"data_offsets":[32,37]},"d.i64":{"dtype":"I64","shape":[2],"data_offsets":[37,53]}}"#;
    st_bytes(h, 53)
}

/// Parse with `from_bytes`, then read every tensor every way; and require
/// `open` on the same bytes to agree.
fn check_safetensors(bytes: &[u8], scratch: &Scratch) -> Outcome {
    let input = bytes.to_vec();
    run(input.len(), || {
        let from_mem = SafeTensors::from_bytes("mutated", input.clone());
        let path = scratch.write("m.safetensors", &input);
        let from_file = SafeTensors::open(&path);
        if from_mem.is_ok() != from_file.is_ok() {
            return Err(Fail::Inconsistent(format!(
                "from_bytes ok={} but open ok={} (mem: {:?}; file: {:?})",
                from_mem.is_ok(),
                from_file.is_ok(),
                from_mem.as_ref().err(),
                from_file.as_ref().err()
            )));
        }
        let (st, file_st) = match (from_mem, from_file) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(e), Err(_)) => return Err(Fail::Refused(e)),
            _ => unreachable!(),
        };
        for s in [&st, &file_st] {
            let names: Vec<String> = s.names().map(str::to_string).collect();
            for n in names {
                // The header was accepted, so every listed tensor reads back whole.
                let bad = |e: String| Fail::Inconsistent(format!("{n}: {e}"));
                let info = s.info(&n).map_err(bad)?.clone();
                let (info2, raw) = s.read_raw(&n).map_err(bad)?;
                if info2 != info || raw.len() as u64 != info.end - info.begin {
                    return Err(bad("read_raw disagrees with its header entry".into()));
                }
                if raw.len() as u64 != info.numel() as u64 * info.dtype.size() {
                    return Err(bad("byte length is not numel x dtype size".into()));
                }
                // The typed readers may refuse a dtype; they must not panic.
                let _ = s.read_f32(&n);
                let _ = s.read_u8(&n);
                let _ = s.read_i8(&n);
                let _ = s.read_u32(&n);
                let _ = s.read_bf16_bits(&n);
            }
        }
        Ok(())
    })
}

#[test]
fn safetensors_parser_survives_mutation() {
    let scratch = Scratch::new("st");
    let seed_bytes = st_seed();
    {
        let st = SafeTensors::from_bytes("seed", seed_bytes.clone()).expect("the seed file must be valid");
        assert_eq!(st.names().count(), 4);
        assert_eq!(st.info("b.bf16").unwrap().dtype, Dtype::BF16);
    }
    let mut f = Failures::default();
    let all = mutations(&seed_bytes, seed(), cases());
    let total = all.len();
    for (label, b) in all {
        let o = check_safetensors(&b, &scratch);
        let mut violations = o.violations;
        if label.starts_with("truncate") && o.ok {
            violations.push("a strict truncation of a valid file was accepted".into());
        }
        for v in violations {
            f.add("safetensors", &label, v);
        }
    }
    f.finish(total);
}

#[test]
fn safetensors_hostile_headers_are_named_errors() {
    let scratch = Scratch::new("st-hostile");
    let valid = st_seed();
    let body = valid.len() - 8;
    let hdr_len = u64::from_le_bytes(valid[..8].try_into().unwrap());
    let mut cases: Vec<(String, Vec<u8>)> = Vec::new();

    // Header length field at its extremes, data untouched.
    for n in [
        0u64,
        1,
        hdr_len - 1,
        hdr_len + 1,
        body as u64,
        body as u64 + 1,
        crate::safetensors::MAX_HEADER_BYTES,
        crate::safetensors::MAX_HEADER_BYTES + 1,
        1 << 32,
        1 << 63,
        u64::MAX - 7,
        u64::MAX - 8,
        u64::MAX,
    ] {
        if n == hdr_len {
            continue;
        }
        let mut b = valid.clone();
        b[..8].copy_from_slice(&n.to_le_bytes());
        cases.push((format!("header length {n}"), b));
    }
    for len in 0..8 {
        cases.push((format!("{len}-byte file"), valid[..len].to_vec()));
    }

    // Header JSON whose fields are individually well-formed but jointly hostile.
    let t = |entries: &str, data: usize| st_bytes(&format!("{{{entries}}}"), data);
    let u = u64::MAX;
    let one = |dt: &str, shape: &str, a: &str, b: &str| {
        format!(r#""x":{{"dtype":"{dt}","shape":{shape},"data_offsets":[{a},{b}]}}"#)
    };
    for (label, entries, data) in [
        ("reversed offsets", one("U8", "[8]", "8", "0"), 8),
        ("offsets past the buffer", one("U8", "[8]", "0", "16"), 8),
        ("begin past the buffer", one("U8", "[0]", "100", "100"), 8),
        (
            "u64::MAX begin and end",
            one("U8", "[0]", &u.to_string(), &u.to_string()),
            8,
        ),
        ("end u64::MAX", one("U8", "[8]", "0", &u.to_string()), 8),
        ("begin u64::MAX, end 0", one("U8", "[8]", &u.to_string(), "0"), 8),
        ("hole at the start", one("U8", "[4]", "4", "8"), 8),
        ("hole at the end", one("U8", "[4]", "0", "4"), 8),
        (
            "shape product overflows u64",
            one("U8", "[4294967296,4294967296]", "0", "0"),
            0,
        ),
        ("dimension u64::MAX", one("U8", &format!("[{u},2]"), "0", "8"), 8),
        (
            "dimension u64::MAX x dtype size overflows",
            one("F64", &format!("[{u}]"), "0", "8"),
            8,
        ),
        (
            "dimension 2^61 x 8 wraps to zero",
            one("F64", "[2305843009213693952]", "0", "0"),
            0,
        ),
        ("empty shape with 2 bytes", one("U8", "[]", "0", "2"), 2),
        ("negative dim", one("U8", "[-1]", "0", "1"), 1),
        ("float dim", one("U8", "[1.0]", "0", "1"), 1),
        ("unknown dtype", one("F4", "[1]", "0", "1"), 1),
        (
            "one offset",
            r#""x":{"dtype":"U8","shape":[1],"data_offsets":[0]}"#.to_string(),
            1,
        ),
        (
            "three offsets",
            r#""x":{"dtype":"U8","shape":[1],"data_offsets":[0,1,1]}"#.to_string(),
            1,
        ),
        (
            "extra field",
            r#""x":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"z":1}"#.to_string(),
            1,
        ),
        (
            "duplicate tensor name",
            format!("{},{}", one("U8", "[1]", "0", "1"), one("U8", "[1]", "1", "2")),
            2,
        ),
        (
            "overlapping tensors",
            r#""p":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"q":{"dtype":"U8","shape":[4],"data_offsets":[2,6]}"#
                .to_string(),
            6,
        ),
        ("metadata not strings", r#""__metadata__":{"k":1}"#.to_string(), 0),
        ("metadata not an object", r#""__metadata__":[]"#.to_string(), 0),
    ] {
        cases.push((label.to_string(), t(&entries, data)));
    }
    cases.push(("array root".into(), st_bytes("[]", 0)));
    cases.push((
        "header 1e999".into(),
        st_bytes(r#"{"x":{"dtype":"U8","shape":[1e999],"data_offsets":[0,1]}}"#, 1),
    ));
    cases.push((
        "lone surrogate in a name".into(),
        st_bytes(r#"{"\ud800":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#, 1),
    ));
    cases.push(("deep nesting in a header".into(), st_bytes(&"[".repeat(100_000), 0)));
    cases.push(("trailing garbage after header".into(), st_bytes(r#"{} x"#, 0)));
    // Invalid UTF-8 inside the header.
    let mut bad = st_bytes(r#"{"xx":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#, 1);
    bad[10] = 0xff;
    bad[11] = 0xc0;
    cases.push(("invalid UTF-8 in a tensor name".into(), bad));

    let mut f = Failures::default();
    let total = cases.len();
    for (label, b) in cases {
        let o = check_safetensors(&b, &scratch);
        for v in o.violations {
            f.add("safetensors", &label, v);
        }
        if o.ok {
            f.add("safetensors", &label, "a hostile file was accepted".into());
        }
    }
    f.finish(total);
}

// ---------------------------------------------------------------------- npy

/// An npy file: magic, version, header padded with spaces to a 64-byte
/// multiple and ending in a newline (as NumPy writes it), then the payload.
fn npy_bytes(version: u8, dict: &str, payload: &[u8]) -> Vec<u8> {
    let prefix = if version == 1 { 10 } else { 12 };
    let mut header = dict.as_bytes().to_vec();
    while (prefix + header.len() + 1) % 64 != 0 {
        header.push(b' ');
    }
    header.push(b'\n');
    let mut b = b"\x93NUMPY".to_vec();
    b.extend_from_slice(&[version, 0]);
    if version == 1 {
        b.extend_from_slice(&(header.len() as u16).to_le_bytes());
    } else {
        b.extend_from_slice(&(header.len() as u32).to_le_bytes());
    }
    b.extend_from_slice(&header);
    b.extend_from_slice(payload);
    b
}

fn npy_seeds() -> Vec<(&'static str, Vec<u8>)> {
    let p = |n: usize| (0..n).map(|i| (i * 5 + 1) as u8).collect::<Vec<u8>>();
    vec![
        (
            "v1 f4 (3,2)",
            npy_bytes(1, "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 2), }", &p(24)),
        ),
        (
            "v2 f8 (4,)",
            npy_bytes(2, "{'descr': '<f8', 'fortran_order': False, 'shape': (4,), }", &p(32)),
        ),
        (
            "v3 i8 scalar",
            npy_bytes(3, "{'descr': '<i8', 'fortran_order': False, 'shape': (), }", &p(8)),
        ),
    ]
}

fn check_npy(bytes: &[u8], scratch: &Scratch) -> Outcome {
    let input = bytes.to_vec();
    run(input.len(), || {
        let path = scratch.write("m.npy", &input);
        let a = read_npy(&path).map_err(Fail::Refused)?;
        let numel: usize = a.shape.iter().product();
        let len = a
            .data_f32
            .as_ref()
            .map(Vec::len)
            .or(a.data_f64.as_ref().map(Vec::len))
            .or(a.data_i64.as_ref().map(Vec::len));
        if len != Some(numel) {
            return Err(Fail::Inconsistent(format!(
                "shape {:?} but payload of {len:?} elements",
                a.shape
            )));
        }
        Ok(())
    })
}

#[test]
fn npy_parser_survives_mutation() {
    let scratch = Scratch::new("npy");
    let mut f = Failures::default();
    let mut total = 0;
    for (name, seed_bytes) in npy_seeds() {
        assert!(check_npy(&seed_bytes, &scratch).ok, "the {name} seed must read back");
        for (label, b) in mutations(&seed_bytes, seed() ^ name.len() as u64, cases()) {
            total += 1;
            let o = check_npy(&b, &scratch);
            let mut violations = o.violations;
            if label.starts_with("truncate") && o.ok {
                violations.push("a strict truncation of a valid file was accepted".into());
            }
            for v in violations {
                f.add(name, &label, v);
            }
        }
    }
    f.finish(total);
}

#[test]
fn npy_hostile_headers_are_named_errors() {
    let scratch = Scratch::new("npy-hostile");
    let ok_dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }";
    let mut cases: Vec<(String, Vec<u8>)> = Vec::new();
    let valid = npy_bytes(1, ok_dict, &[0u8; 8]);

    // Magic and version.
    for len in 0..12 {
        cases.push((format!("{len}-byte file"), valid[..len].to_vec()));
    }
    let mut b = valid.clone();
    b[0] = 0x92;
    cases.push(("bad magic".into(), b));
    for v in [0u8, 4, 9, 255] {
        let mut b = valid.clone();
        b[6] = v;
        cases.push((format!("version {v}"), b));
    }

    // Header-length field at its extremes, for each width.
    for n in [0u16, 1, 0xffff, (valid.len() - 10 + 1) as u16] {
        let mut b = valid.clone();
        b[8..10].copy_from_slice(&n.to_le_bytes());
        cases.push((format!("v1 header length {n}"), b));
    }
    let v2 = npy_bytes(2, ok_dict, &[0u8; 8]);
    for n in [
        0u32,
        1,
        0xffff,
        0x10_0000,
        0x10_0001,
        0x7fff_ffff,
        0x8000_0000,
        u32::MAX,
    ] {
        let mut b = v2.clone();
        b[8..12].copy_from_slice(&n.to_le_bytes());
        cases.push((format!("v2 header length {n}"), b));
    }

    // Shapes and dtypes.
    for (label, dict, payload) in [
        ("shape product overflows usize", "{'descr': '<f4', 'fortran_order': False, 'shape': (4294967296, 4294967296), }", 0),
        ("dimension usize::MAX", "{'descr': '<f8', 'fortran_order': False, 'shape': (18446744073709551615,), }", 0),
        ("dimension past usize", "{'descr': '<f4', 'fortran_order': False, 'shape': (18446744073709551616,), }", 0),
        ("2^61 x 8 wraps to zero", "{'descr': '<f8', 'fortran_order': False, 'shape': (2305843009213693952,), }", 0),
        ("negative dimension", "{'descr': '<f4', 'fortran_order': False, 'shape': (-1,), }", 0),
        ("huge shape, tiny payload", "{'descr': '<f4', 'fortran_order': False, 'shape': (1000000000000,), }", 16),
        ("fortran order", "{'descr': '<f4', 'fortran_order': True, 'shape': (2,), }", 8),
        ("fortran order not a bool", "{'descr': '<f4', 'fortran_order': Maybe, 'shape': (2,), }", 8),
        ("big-endian dtype", "{'descr': '>f4', 'fortran_order': False, 'shape': (2,), }", 8),
        ("structured dtype", "{'descr': [('a','<f4')], 'fortran_order': False, 'shape': (2,), }", 8),
        ("missing shape", "{'descr': '<f4', 'fortran_order': False, }", 8),
        ("missing descr", "{'fortran_order': False, 'shape': (2,), }", 8),
        ("unterminated shape", "{'descr': '<f4', 'fortran_order': False, 'shape': (2, }", 8),
        ("unterminated descr", "{'descr': '<f4, 'fortran_order': False, 'shape': (2,), }", 8),
        ("multi-byte text after fortran_order", "{'descr': '<f4', 'shape': (2,), 'fortran_order': \u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}}", 8),
        ("non-numeric dimension", "{'descr': '<f4', 'fortran_order': False, 'shape': (a,), }", 8),
    ] {
        cases.push((label.to_string(), npy_bytes(1, dict, &vec![0u8; payload])));
    }
    // Payload one byte short or long for a valid header.
    cases.push(("payload one byte short".into(), npy_bytes(1, ok_dict, &[0u8; 7])));
    cases.push(("payload one byte long".into(), npy_bytes(1, ok_dict, &[0u8; 9])));
    // Invalid UTF-8 in the header.
    let mut bad = valid.clone();
    bad[12] = 0xff;
    bad[13] = 0xfe;
    cases.push(("invalid UTF-8 in the header".into(), bad));
    // Maximum-size header (1 MiB + 1) on a file that is long enough to hold it.
    let mut big = b"\x93NUMPY\x02\x00".to_vec();
    big.extend_from_slice(&((crate::npy::MAX_NPY_HEADER_BYTES as u32) + 1).to_le_bytes());
    big.extend(std::iter::repeat_n(b' ', crate::npy::MAX_NPY_HEADER_BYTES + 1));
    cases.push(("header over the cap".into(), big));

    let mut f = Failures::default();
    let total = cases.len();
    for (label, b) in cases {
        let o = check_npy(&b, &scratch);
        for v in o.violations {
            f.add("npy", &label, v);
        }
        if o.ok {
            f.add("npy", &label, "a hostile file was accepted".into());
        }
    }
    f.finish(total);
}

/// Second-pass adversarial review: inputs the mutation operators cannot reach,
/// the public file-path entry points, and positive controls for the harness
/// itself (an internal-consistency failure inside `check_safetensors` comes
/// back as an ordinary `Err`, which the mutation passes accept, so the seed
/// files must be shown to read back clean separately).
#[cfg(test)]
mod adversarial {
    use super::*;
    use std::path::Path;
    use std::time::Duration;

    /// Positive control: a valid safetensors file passes every read-back
    /// check in `check_safetensors`. Without this, a parser fault that made
    /// `read_raw` return the wrong bytes surfaces only as an `Err` that the
    /// mutation passes treat as a refusal (verified by injecting
    /// `bytes.pop()` into `read_raw`: all six existing tests stayed green).
    #[test]
    fn safetensors_seed_reads_back_clean() {
        let scratch = Scratch::new("adv-st-seed");
        let o = check_safetensors(&st_seed(), &scratch);
        assert!(
            o.ok,
            "valid seed refused or inconsistent: {:?} {:?}",
            o.err, o.violations
        );
        assert!(o.violations.is_empty());
    }

    /// The mutation passes must generate a non-trivial number of inputs: a
    /// regression to zero cases would otherwise pass silently.
    #[test]
    fn mutation_corpora_are_non_trivial() {
        assert!(cases() > 0, "PARSER_MUTATION_CASES=0 runs no random cases");
        let st = st_seed();
        let n = mutations(&st, seed(), 0).len();
        assert!(
            n >= st.len() * 9,
            "only {n} exhaustive mutations of a {}-byte file",
            st.len()
        );
        assert!(mutations(QWEN_CONFIG.as_bytes(), seed(), 0).len() >= 5000);
        for (_, b) in npy_seeds() {
            assert!(mutations(&b, seed(), 0).len() >= b.len() * 9);
        }
    }

    fn timed<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx.recv_timeout(Duration::from_secs(2)).ok()
    }

    /// Directories, character devices and a file shrunk or rewritten after
    /// `open` are named errors (no mmap, so no SIGBUS: reads are `pread`s on
    /// the validated handle).
    #[test]
    fn file_entry_points_refuse_non_regular_and_shrunk_files() {
        let sc = Scratch::new("adv-files");
        assert!(SafeTensors::open(&sc.0).is_err());
        assert!(read_npy(&sc.0).is_err());
        for dev in ["/dev/zero", "/dev/null"] {
            assert!(SafeTensors::open(Path::new(dev)).is_err(), "{dev}");
            assert!(read_npy(Path::new(dev)).is_err(), "{dev}");
        }
        assert!(SafeTensors::open(Path::new("")).is_err());
        assert!(SafeTensors::open(&sc.0.join("a".repeat(300))).is_err());
        let bytes = st_seed();
        assert_eq!(
            SafeTensors::open(&sc.write("caf\u{e9}-\u{1F600}.safetensors", &bytes))
                .unwrap()
                .names()
                .count(),
            4
        );
        let p = sc.write("shrink.safetensors", &bytes);
        let s = SafeTensors::open(&p).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&p)
            .unwrap()
            .set_len(60)
            .unwrap();
        assert!(
            s.read_raw("d.i64").is_err(),
            "a read past the shrunk end must be an Err"
        );
        std::fs::write(&p, b"").unwrap();
        assert!(s.read_raw("a.f32").is_err());
    }

    /// Opening a FIFO with no writer used to block forever in `File::open`, in
    /// both `SafeTensors::open` and `read_npy`; both now refuse a non-regular
    /// file with a named error before any blocking open.
    #[test]
    fn file_entry_points_do_not_block_on_a_fifo() {
        let sc = Scratch::new("adv-fifo");
        let fifo = sc.0.join("f.bin");
        assert!(std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success());
        let p1 = fifo.clone();
        let st = timed(move || SafeTensors::open(&p1).map(|_| ()));
        let p2 = fifo.clone();
        let npy = timed(move || read_npy(&p2).map(|_| ()));
        // Release any blocked readers so the process can exit cleanly. A
        // read+write open of a FIFO never blocks, and one open releases every
        // blocked reader (a write-only open would block with none left).
        drop(std::fs::OpenOptions::new().read(true).write(true).open(&fifo));
        for (name, r) in [("SafeTensors::open", st), ("read_npy", npy)] {
            let r = r.unwrap_or_else(|| panic!("{name} on a FIFO blocked"));
            let e = r.expect_err("a FIFO must be refused");
            assert!(e.contains("not a regular file"), "{name}: {e}");
        }
    }

    /// A header exactly at the cap is read; one byte more is refused, and in
    /// both cases the one big allocation is the header itself.
    #[test]
    fn header_length_at_the_cap() {
        let cap = crate::safetensors::MAX_HEADER_BYTES as usize;
        for (extra, ok) in [(0usize, true), (1, false)] {
            let h = format!("{{}}{}", " ".repeat(cap + extra - 2));
            let b = st_bytes(&h, 0);
            let o = run(b.len(), || SafeTensors::from_bytes("m", b.clone()).map(|_| ()));
            assert!(o.violations.is_empty(), "{:?}", o.violations);
            assert_eq!(o.ok, ok, "header of cap+{extra}: {:?}", o.err);
        }
    }

    /// Text-to-structure amplification is linear and bounded: 5M `0` dims
    /// (10 MB) make one allocation of about 27x the input (the `Json` array),
    /// well inside the harness's 64x bound but worth pinning, since a 100 MB
    /// header reaches the same ratio. A growth in node size shows up here.
    #[test]
    fn header_amplification_is_pinned() {
        let dims = vec!["0"; 5_000_000].join(",");
        let h = format!(r#"{{"x":{{"dtype":"U8","shape":[{dims}],"data_offsets":[0,0]}}}}"#);
        let b = st_bytes(&h, 0);
        let o = run(b.len(), || SafeTensors::from_bytes("m", b.clone()).map(|_| ()));
        let max = MAX_ALLOC.with(Cell::get);
        assert!(o.ok && o.violations.is_empty(), "{:?} {:?}", o.err, o.violations);
        assert!(
            max <= 32 * b.len(),
            "single allocation {max} B for a {} B header",
            b.len()
        );
    }

    /// JSON number edges under the config syntax: huge exponent is a named
    /// error, `-0`, 2^64 and an enormous negative exponent are values, a 1 MB
    /// digit run and a 10 MB string are linear.
    #[test]
    fn json_number_and_string_edges() {
        let parse = |t: &str| run(t.len(), || json::parse(t, CONFIG_SYNTAX).map(|_| ()));
        assert!(!parse(r#"{"a":1e999999999}"#).ok);
        for t in [r#"{"a":-0}"#, r#"{"a":18446744073709551616}"#, r#"{"a":-0e-999999999}"#] {
            let o = parse(t);
            assert!(o.ok && o.violations.is_empty(), "{t}: {:?}", o.err);
        }
        let digits = format!("{{\"a\":{}}}", "9".repeat(1_000_000));
        let o = parse(&digits);
        assert!(o.err.is_some() && o.violations.is_empty(), "{:?}", o.violations);
        let s = format!("{{\"a\":\"{}\"}}", "x".repeat(10_000_000));
        let o = parse(&s);
        assert!(o.ok && o.violations.is_empty(), "{:?}", o.violations);
        let mixed = "{\"a\":[".repeat(100_000);
        assert!(parse(&mixed).err.unwrap().contains("nesting"));
    }
}
