//! Seeded mutation and property tests for the hand-written parsers that read
//! untrusted files: `.safetensors` (and, through its header, `src/json.rs`)
//! and `.npy`. CPU only: no Metal device is opened, so this runs on hosted CI.
//!
//! No dependency: a small xorshift PRNG drives truncation at every byte,
//! byte and bit flips, spliced and duplicated ranges, extreme 64-bit values
//! written over any offset, and generated headers whose lengths, offsets and
//! shapes sit at 0, `u64::MAX` and the overflow edges of a multiply. Each input
//! must
//!
//! * never panic or overflow (the profile checks arithmetic);
//! * return `Ok` or a non-empty `Err`;
//! * never request one allocation beyond a bound proportional to the input
//!   (a counting global allocator records the largest single request made on
//!   the test's thread), so a declared size is checked against the file before
//!   anything is allocated;
//! * when `Ok`, hold the invariants the parser promises (tensors tile the data
//!   buffer; an array's length is its shape's product).
//!
//! `TESSL_MUTATION_CASES` sets the random cases per section (default 20 000)
//! and `TESSL_MUTATION_SEED` the seed. The deterministic sweeps always run in
//! full.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use tessl::npy::{read_npy, MAX_NPY_HEADER_BYTES};
use tessl::safetensors::{SafeTensors, MAX_HEADER_BYTES};

struct Counting;

thread_local! {
    static LARGEST: Cell<usize> = const { Cell::new(0) };
}

fn note(size: usize) {
    // `try_with`: the slot is gone during thread teardown.
    let _ = LARGEST.try_with(|c| c.set(c.get().max(size)));
}

// SAFETY: every method forwards to `System` unchanged; `note` only updates a
// const-initialised thread-local `Cell`, which neither allocates nor panics.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        note(l.size());
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        note(l.size());
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        note(new);
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Run `f` and return its result with the largest single allocation it made.
fn measure<T>(f: impl FnOnce() -> T) -> (T, usize) {
    LARGEST.with(|c| c.set(0));
    let r = f();
    (r, LARGEST.with(|c| c.get()))
}

/// The largest single allocation a parse of `len` input bytes may make. Parsed
/// JSON is a tree of 32-byte-plus nodes, one per two input bytes at worst, so
/// the bound is generous in `len` but a declared size of gigabytes is far
/// outside it.
fn alloc_bound(len: usize) -> usize {
    64 * len + (1 << 20)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len())]
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Ok(v) => v.parse().unwrap_or_else(|_| panic!("{name}={v:?} is not an integer")),
        Err(_) => default,
    }
}

fn cases() -> u64 {
    env_u64("TESSL_MUTATION_CASES", 20_000)
}

fn rng() -> Rng {
    Rng(env_u64("TESSL_MUTATION_SEED", 0x9e37_79b9_7f4a_7c15) | 1)
}

const EXTREMES: [u64; 14] = [
    0,
    1,
    2,
    255,
    256,
    65_535,
    65_536,
    u32::MAX as u64,
    1 << 32,
    (1 << 62) + 1,
    i64::MAX as u64,
    1 << 63,
    u64::MAX - 1,
    u64::MAX,
];

const INTERESTING: [u8; 16] = [
    0, 1, 0x7f, 0x80, 0xff, b'"', b'\\', b'{', b'}', b'[', b']', b',', b':', b'-', b'9', b' ',
];

/// A mutation of `seed`, chosen by `r`.
fn mutate(r: &mut Rng, seed: &[u8]) -> Vec<u8> {
    let mut b = seed.to_vec();
    for _ in 0..1 + r.below(4) {
        match r.below(9) {
            0 if !b.is_empty() => {
                let i = r.below(b.len());
                b[i] ^= 1 << r.below(8);
            }
            1 if !b.is_empty() => {
                let i = r.below(b.len());
                b[i] = r.pick(&INTERESTING);
            }
            2 if !b.is_empty() => {
                let i = r.below(b.len());
                b[i] = r.next() as u8;
            }
            3 => {
                let cut = r.below(b.len() + 1);
                b.truncate(cut);
            }
            4 if !b.is_empty() => {
                let i = r.below(b.len());
                let j = (i + 1 + r.below(16)).min(b.len());
                b.drain(i..j);
            }
            5 => {
                let i = r.below(b.len() + 1);
                let n = 1 + r.below(16);
                let fill: Vec<u8> = (0..n).map(|_| r.pick(&INTERESTING)).collect();
                b.splice(i..i, fill);
            }
            6 if !b.is_empty() => {
                // Duplicate a range in place.
                let i = r.below(b.len());
                let j = (i + 1 + r.below(32)).min(b.len());
                let chunk = b[i..j].to_vec();
                b.splice(i..i, chunk);
            }
            7 if b.len() >= 8 => {
                // An extreme little-endian u64 or u32 over any offset.
                let i = r.below(b.len() - 7);
                let v = r.pick(&EXTREMES);
                if r.below(2) == 0 {
                    b[i..i + 8].copy_from_slice(&v.to_le_bytes());
                } else {
                    b[i..i + 4].copy_from_slice(&(v as u32).to_le_bytes());
                }
            }
            8 if !b.is_empty() => {
                // Overwrite a decimal digit run: lengths, offsets and dims in
                // text headers.
                let i = r.below(b.len());
                if let Some(p) = b[i..].iter().position(u8::is_ascii_digit) {
                    let text = r.pick(&EXTREMES).to_string();
                    let start = i + p;
                    let end = start + b[start..].iter().take_while(|c| c.is_ascii_digit()).count();
                    b.splice(start..end, text.into_bytes());
                }
            }
            _ => {}
        }
    }
    b
}

// ---------------------------------------------------------------- safetensors

/// A `.safetensors` serialization: header length, JSON, then `data`.
fn st_file(header: &str, data: &[u8]) -> Vec<u8> {
    let mut b = (header.len() as u64).to_le_bytes().to_vec();
    b.extend_from_slice(header.as_bytes());
    b.extend_from_slice(data);
    b
}

fn st_seeds() -> Vec<Vec<u8>> {
    let two = st_file(
        r#"{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"b":{"dtype":"BF16","shape":[3],"data_offsets":[16,22]}}"#,
        &(0..22u8).collect::<Vec<_>>(),
    );
    let scalar_and_empty = st_file(
        r#"{"s":{"dtype":"I64","shape":[],"data_offsets":[0,8]},"z":{"dtype":"F32","shape":[0,5],"data_offsets":[8,8]}}"#,
        &[7u8; 8],
    );
    let spaced = st_file(
        "{ \"w\" : { \"dtype\" : \"U8\" , \"shape\" : [ 4 ] , \"data_offsets\" : [ 0 , 4 ] } }    ",
        &[1, 2, 3, 4],
    );
    let none = st_file("{}", &[]);
    let escapes = st_file(
        r#"{"naé😀me":{"dtype":"F8_E5M2","shape":[2],"data_offsets":[0,2]}}"#,
        &[9, 9],
    );
    vec![two, scalar_and_empty, spaced, none, escapes]
}

/// Parse `bytes` and check every promise; the result is whether it was `Ok`.
fn check_st(label: &str, bytes: &[u8]) -> bool {
    let (res, largest) = measure(|| SafeTensors::from_bytes("mem", bytes.to_vec()));
    // `to_vec` is the harness's own copy, `bytes.len()` at most.
    assert!(
        largest <= alloc_bound(bytes.len()),
        "{label}: one {largest}-byte allocation for a {}-byte input",
        bytes.len()
    );
    match res {
        Err(e) => {
            assert!(!e.is_empty(), "{label}: empty error");
            false
        }
        Ok(st) => {
            let data_len = bytes.len() as u64 - 8 - u64::from_le_bytes(bytes[..8].try_into().unwrap());
            let mut spans: Vec<(u64, u64)> = Vec::new();
            for name in st.names() {
                let info = st.info(name).unwrap();
                assert!(
                    info.begin <= info.end && info.end <= data_len,
                    "{label}: {name} outside buffer"
                );
                let numel: u64 = info.shape.iter().map(|&d| d as u64).product();
                assert_eq!(numel * info.dtype.size(), info.end - info.begin, "{label}: {name} size");
                spans.push((info.begin, info.end));
                let ((_, raw), largest) = measure(|| st.read_raw(name).unwrap());
                assert_eq!(raw.len() as u64, info.end - info.begin, "{label}: {name} raw length");
                assert!(
                    largest <= alloc_bound(bytes.len()),
                    "{label}: read of {name} allocated {largest}"
                );
            }
            spans.sort_unstable();
            let mut at = 0;
            for (b, e) in spans {
                assert_eq!(b, at, "{label}: tensors do not tile the buffer");
                at = e;
            }
            assert_eq!(at, data_len, "{label}: tensors do not cover the buffer");
            true
        }
    }
}

static SEQ: AtomicUsize = AtomicUsize::new(0);

fn tmp(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("tessl-mut-{}-{tag}-{n}", std::process::id()))
}

/// `open` and `from_bytes` must reach the same verdict on the same bytes.
fn check_st_file(label: &str, bytes: &[u8]) {
    let p = tmp("st");
    std::fs::write(&p, bytes).unwrap();
    let (res, largest) = measure(|| SafeTensors::open(&p));
    let _ = std::fs::remove_file(&p);
    assert!(
        largest <= alloc_bound(bytes.len()),
        "{label}: open allocated {largest} for a {}-byte file",
        bytes.len()
    );
    let from_mem = SafeTensors::from_bytes("mem", bytes.to_vec());
    assert_eq!(res.is_ok(), from_mem.is_ok(), "{label}: open and from_bytes disagree");
    if let (Ok(a), Ok(b)) = (res, from_mem) {
        assert_eq!(a.names().collect::<Vec<_>>(), b.names().collect::<Vec<_>>(), "{label}");
    }
}

#[test]
fn seeds_are_valid() {
    for (i, s) in st_seeds().iter().enumerate() {
        assert!(check_st(&format!("seed {i}"), s), "safetensors seed {i} must parse");
    }
    for (i, (_, s)) in npy_seeds().iter().enumerate() {
        assert!(check_npy(&format!("seed {i}"), s), "npy seed {i} must parse");
    }
}

#[test]
fn safetensors_truncated_at_every_byte() {
    for (i, seed) in st_seeds().iter().enumerate() {
        for cut in 0..seed.len() {
            let ok = check_st(&format!("seed {i} cut {cut}"), &seed[..cut]);
            assert!(!ok, "seed {i} truncated to {cut} bytes parsed as valid");
        }
    }
}

#[test]
fn safetensors_every_byte_replaced() {
    for (i, seed) in st_seeds().iter().enumerate() {
        for pos in 0..seed.len() {
            for v in INTERESTING {
                let mut m = seed.clone();
                m[pos] = v;
                check_st(&format!("seed {i} byte {pos}={v:#x}"), &m);
            }
            for bit in 0..8 {
                let mut m = seed.clone();
                m[pos] ^= 1 << bit;
                check_st(&format!("seed {i} byte {pos} bit {bit}"), &m);
            }
        }
    }
}

#[test]
fn safetensors_header_length_extremes() {
    for seed in st_seeds() {
        for v in EXTREMES {
            let mut m = seed.clone();
            m[..8].copy_from_slice(&v.to_le_bytes());
            check_st(&format!("header length {v}"), &m);
            check_st_file(&format!("header length {v}"), &m);
        }
        for v in [MAX_HEADER_BYTES, MAX_HEADER_BYTES + 1, MAX_HEADER_BYTES - 1] {
            let mut m = seed.clone();
            m[..8].copy_from_slice(&v.to_le_bytes());
            check_st(&format!("header length {v}"), &m);
        }
    }
}

#[test]
fn safetensors_random_mutations() {
    let seeds = st_seeds();
    let mut r = rng();
    for n in 0..cases() {
        let seed = &seeds[r.below(seeds.len())];
        let m = mutate(&mut r, seed);
        check_st(&format!("case {n}"), &m);
        if n % 16 == 0 {
            check_st_file(&format!("case {n}"), &m);
        }
    }
}

/// A decimal or bare extreme for a header field.
fn num(r: &mut Rng) -> String {
    match r.below(4) {
        0 => r.pick(&EXTREMES).to_string(),
        1 => r.below(40).to_string(),
        2 => (r.next() % 1_000_000).to_string(),
        _ => format!("{}", r.pick(&EXTREMES) as u128 * 10 + r.below(10) as u128),
    }
}

/// A header written from the format's grammar with hostile values: shape
/// products that wrap, offsets past the end, reversed, overlapping, out of
/// order and duplicated, followed by a data buffer of a size the generator
/// sometimes gets exactly right.
fn generated_header(r: &mut Rng) -> Vec<u8> {
    let dtypes = ["F32", "BF16", "U8", "I64", "F8_E5M2", "C64", "F4", "BOGUS", ""];
    let mut parts: Vec<String> = Vec::new();
    let mut at = 0u64;
    let mut data_len = 0u64;
    for t in 0..r.below(5) {
        let dt = r.pick(&dtypes);
        let rank = r.below(4);
        let shape: Vec<String> = (0..rank)
            .map(|_| {
                if r.below(3) == 0 {
                    num(r)
                } else {
                    (1 + r.below(4)).to_string()
                }
            })
            .collect();
        let numel: Option<u64> = shape
            .iter()
            .try_fold(1u64, |a, d| a.checked_mul(d.parse::<u64>().ok()?));
        let size = match dt {
            "F32" => 4,
            "BF16" => 2,
            "I64" | "C64" => 8,
            _ => 1,
        };
        let (b, e) = match (r.below(4), numel.and_then(|n| n.checked_mul(size))) {
            (0, _) => (num(r), num(r)),
            (1, Some(want)) => (
                (at.saturating_sub(r.below(2) as u64)).to_string(),
                (at + want + r.below(2) as u64).to_string(),
            ),
            (_, Some(want)) => (at.to_string(), (at + want).to_string()),
            (_, None) => (at.to_string(), num(r)),
        };
        if let (Ok(bb), Ok(ee)) = (b.parse::<u64>(), e.parse::<u64>()) {
            at = ee.max(bb).min(1 << 20);
            data_len = data_len.max(at);
        }
        let name = if r.below(8) == 0 {
            "dup".to_string()
        } else {
            format!("t{t}")
        };
        let extra = if r.below(10) == 0 { r#","extra":1"# } else { "" };
        parts.push(format!(
            r#""{name}":{{"dtype":"{dt}","shape":[{}],"data_offsets":[{b},{e}]{extra}}}"#,
            shape.join(",")
        ));
    }
    if r.below(4) == 0 {
        parts.insert(0, r#""__metadata__":{"k":"v"}"#.to_string());
    }
    let header = format!("{{{}}}", parts.join(","));
    let declared = match r.below(6) {
        0 => r.pick(&EXTREMES),
        1 => header.len() as u64 + r.below(3) as u64,
        _ => header.len() as u64,
    };
    let data = vec![
        0u8;
        if r.below(3) == 0 {
            r.below(64)
        } else {
            data_len.min(4096) as usize
        }
    ];
    let mut b = declared.to_le_bytes().to_vec();
    b.extend_from_slice(header.as_bytes());
    b.extend_from_slice(&data);
    b
}

#[test]
fn safetensors_generated_hostile_headers() {
    let mut r = rng();
    let mut accepted = 0u64;
    for n in 0..cases() {
        let m = generated_header(&mut r);
        if check_st(&format!("generated {n}"), &m) {
            accepted += 1;
        }
    }
    println!("generated headers accepted: {accepted} of {}", cases());
}

#[test]
fn safetensors_declared_sizes_do_not_allocate() {
    // A tiny file declaring a petabyte tensor, and shapes whose product wraps
    // a u64 to a small number.
    for (shape, end) in [
        ("[1125899906842624]", "1125899906842624"),
        ("[4294967296,4294967296]", "0"),
        ("[4294967296,4294967297]", "4294967296"),
        ("[18446744073709551615]", "18446744073709551615"),
        ("[18446744073709551615,2]", "18446744073709551614"),
        ("[18446744073709551616]", "0"),
        ("[0,18446744073709551615]", "0"),
    ] {
        let h = format!(r#"{{"x":{{"dtype":"U8","shape":{shape},"data_offsets":[0,{end}]}}}}"#);
        let m = st_file(&h, &[0; 4]);
        assert!(!check_st(shape, &m), "{shape} must be refused");
    }
}

#[test]
fn safetensors_deep_nesting_and_hostile_json() {
    let deep = 100_000;
    let cases: Vec<(&str, String)> = vec![
        ("deep arrays", format!("{}1{}", "[".repeat(deep), "]".repeat(deep))),
        ("deep unclosed arrays", "[".repeat(1 << 20)),
        ("deep objects", format!("{}{}", r#"{"a":"#.repeat(deep), "}".repeat(deep))),
        ("unclosed string", format!("{{\"{}", "a".repeat(1 << 16))),
        ("huge digit run", format!(r#"{{"x":{{"dtype":"U8","shape":[{}],"data_offsets":[0,0]}}}}"#, "9".repeat(1 << 16))),
        ("many keys", format!("{{{}}}", (0..5000).map(|i| format!(r#""k{i}":{{}}"#)).collect::<Vec<_>>().join(","))),
        ("duplicate keys", r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#.to_string()),
        ("lone surrogate", r#"{"\ud800":{}}"#.to_string()),
        ("control char", "{\"a\u{1}\":{}}".to_string()),
        ("float", r#"{"a":{"dtype":"U8","shape":[1.5],"data_offsets":[0,1]}}"#.to_string()),
        ("literal", r#"{"a":null}"#.to_string()),
    ];
    for (label, h) in cases {
        let m = st_file(&h, &[0; 8]);
        assert!(!check_st(label, &m), "{label} must be refused");
    }
    // Header bytes that are not UTF-8.
    for bad in [
        &[0xffu8, 0xfe][..],
        &[b'{', 0xc0, 0x80, b'}'],
        &[b'{', 0xed, 0xa0, 0x80, b'}'],
        &[b'{', 0xf4, 0x90, 0x80, 0x80, b'}'],
    ] {
        let mut m = (bad.len() as u64).to_le_bytes().to_vec();
        m.extend_from_slice(bad);
        assert!(!check_st("invalid UTF-8", &m));
    }
}

// ------------------------------------------------------------------------ npy

/// (header, payload) pairs and their serialized forms in npy 1.0, 2.0 and 3.0.
fn npy_bytes(version: u8, header: &str, payload: &[u8]) -> Vec<u8> {
    let prefix = if version == 1 { 10 } else { 12 };
    let mut h = header.to_string();
    h.push_str(&" ".repeat((64 - (prefix + h.len() + 1) % 64) % 64));
    h.push('\n');
    let mut b = b"\x93NUMPY".to_vec();
    b.extend_from_slice(&[version, 0]);
    if version == 1 {
        b.extend_from_slice(&(h.len() as u16).to_le_bytes());
    } else {
        b.extend_from_slice(&(h.len() as u32).to_le_bytes());
    }
    b.extend_from_slice(h.as_bytes());
    b.extend_from_slice(payload);
    b
}

fn npy_seeds() -> Vec<(&'static str, Vec<u8>)> {
    let f4: Vec<u8> = (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let f8: Vec<u8> = (0..4).flat_map(|i| (i as f64).to_le_bytes()).collect();
    let i8s: Vec<u8> = (0..3).flat_map(|i: i64| (i - 1).to_le_bytes()).collect();
    vec![
        (
            "f4 v1",
            npy_bytes(1, "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }", &f4),
        ),
        (
            "f8 v2",
            npy_bytes(2, "{'descr': '<f8', 'fortran_order': False, 'shape': (4,), }", &f8),
        ),
        (
            "i8 v3",
            npy_bytes(3, "{'descr': '<i8', 'fortran_order': False, 'shape': (3,), }", &i8s),
        ),
        (
            "scalar",
            npy_bytes(
                1,
                "{'descr': '<f4', 'fortran_order': False, 'shape': (), }",
                &1.5f32.to_le_bytes(),
            ),
        ),
        (
            "empty",
            npy_bytes(1, "{'descr': '|f4', 'fortran_order': False, 'shape': (0, 5), }", &[]),
        ),
        (
            "utf8 header",
            npy_bytes(
                3,
                "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), 'n\u{e9}me': '\u{1f600}', }",
                &[0; 4],
            ),
        ),
    ]
}

fn check_npy(label: &str, bytes: &[u8]) -> bool {
    let p = tmp("npy");
    std::fs::write(&p, bytes).unwrap();
    let (res, largest) = measure(|| read_npy(&p));
    let _ = std::fs::remove_file(&p);
    assert!(
        largest <= alloc_bound(bytes.len()),
        "{label}: one {largest}-byte allocation for a {}-byte file",
        bytes.len()
    );
    match res {
        Err(e) => {
            assert!(!e.is_empty(), "{label}: empty error");
            false
        }
        Ok(a) => {
            let numel: usize = a.shape.iter().product();
            let got = a
                .data_f32
                .as_ref()
                .map(Vec::len)
                .or(a.data_f64.as_ref().map(Vec::len))
                .or(a.data_i64.as_ref().map(Vec::len));
            assert_eq!(got, Some(numel), "{label}: data length is not the shape's product");
            true
        }
    }
}

#[test]
fn npy_truncated_and_replaced_at_every_byte() {
    for (name, seed) in npy_seeds() {
        for cut in 0..seed.len() {
            assert!(
                !check_npy(&format!("{name} cut {cut}"), &seed[..cut]),
                "{name} cut {cut} parsed"
            );
        }
        // One byte past the end: trailing bytes are refused.
        let mut padded = seed.clone();
        padded.push(0);
        assert!(!check_npy(&format!("{name} padded"), &padded));
        for pos in 0..seed.len() {
            for v in INTERESTING {
                let mut m = seed.clone();
                m[pos] = v;
                check_npy(&format!("{name} byte {pos}={v:#x}"), &m);
            }
            for bit in 0..8 {
                let mut m = seed.clone();
                m[pos] ^= 1 << bit;
                check_npy(&format!("{name} byte {pos} bit {bit}"), &m);
            }
        }
    }
}

#[test]
fn npy_header_length_extremes() {
    for (name, seed) in npy_seeds() {
        let (off, width) = if seed[6] == 1 { (8, 2) } else { (8, 4) };
        for v in EXTREMES
            .into_iter()
            .chain([MAX_NPY_HEADER_BYTES as u64, MAX_NPY_HEADER_BYTES as u64 + 1])
        {
            let mut m = seed.clone();
            m[off..off + width].copy_from_slice(&v.to_le_bytes()[..width]);
            check_npy(&format!("{name} header length {v}"), &m);
        }
        for ver in 0..=255u8 {
            let mut m = seed.clone();
            m[6] = ver;
            check_npy(&format!("{name} version {ver}"), &m);
        }
    }
}

#[test]
fn npy_declared_shapes_do_not_allocate() {
    for shape in [
        "(18446744073709551615,)",
        "(18446744073709551616,)",
        "(4294967296, 4294967296)",
        "(4611686018427387904, 4)",
        "(2305843009213693952,)",
        "(0, 18446744073709551615)",
        "(1, 1, 1, 18446744073709551615)",
        "(-1,)",
        "(,)",
        "(1,,2)",
    ] {
        for descr in ["<f4", "<f8", "<i8"] {
            let m = npy_bytes(
                1,
                &format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}"),
                &[0; 8],
            );
            check_npy(&format!("{descr} {shape}"), &m);
        }
    }
}

#[test]
fn npy_hostile_headers() {
    for (label, header) in [
        ("multibyte after fortran_order", "{'descr': '<f4', 'shape': (1,), 'fortran_order': \u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}}"),
        ("multibyte straddling 16 bytes", "{'descr': '<f4', 'shape': (1,), 'fortran_order': 123456789012345\u{e9}}"),
        ("no keys", "{}"),
        ("unterminated descr", "{'descr': '<f4"),
        ("unterminated shape", "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 2"),
        ("shape before paren", "{'descr': '<f4', 'fortran_order': False, 'shape': 7)("),
        ("duplicate keys", "{'descr': '<f4', 'descr': '<f8', 'fortran_order': False, 'shape': (1,), }"),
        ("deep parens", &format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}), }}", "(".repeat(100_000))),
        ("huge shape list", &format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}), }}", "1,".repeat(300_000))),
    ] {
        check_npy(label, &npy_bytes(1, header, &[0; 4]));
        check_npy(label, &npy_bytes(3, header, &[0; 4]));
    }
    // A header that is not UTF-8.
    let mut m = npy_bytes(3, "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }", &[0; 4]);
    m[20] = 0xff;
    assert!(!check_npy("invalid UTF-8 header", &m));
}

#[test]
fn npy_random_mutations() {
    let seeds = npy_seeds();
    let mut r = rng();
    for n in 0..cases() {
        let (name, seed) = &seeds[r.below(seeds.len())];
        let m = mutate(&mut r, seed);
        check_npy(&format!("{name} case {n}"), &m);
    }
}
