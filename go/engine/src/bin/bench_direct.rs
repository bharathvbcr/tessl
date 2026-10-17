//! The direct arm of the Go overhead benchmark: the same tessl calls the Go
//! package makes through gusset, made straight from Rust against tessl's C ABI,
//! timed around the call alone.
//!
//! ```text
//! bench_direct --snapshot DIR [--iters N] [--warmup W]
//! ```
//!
//! prints one JSON object on stdout with two sets of rows, `main` and `worker`:
//! for each workload its minimum and median wall time over `N` calls (after `W`
//! warm-up calls) and a checksum of the last result's bits. `main` makes and
//! uses the model on the process's main thread; `worker` does the same on a
//! thread of its own with the stack size gusset gives a worker (8 MiB), which is
//! what the engine's model lives on, so `worker` differs from the Go arm in the
//! boundary and not in the kind of thread. `go/cmd/bench_overhead` runs this once
//! per round and refuses a workload whose Go checksum differs: the arms must be
//! doing the same work.
//!
//! The workloads and their inputs are defined here and in
//! `go/cmd/bench_overhead/main.go`, byte for byte: texts `bench text {i}: ` plus
//! three repetitions of `lorem ipsum dolor sit amet `, and 224x224x3 images whose
//! byte `k` of image `j` is `(131 k + 17 j) mod 251`.

use std::ffi::c_char;
use std::ptr;
use std::time::Instant;

use tessl::capi::{tessl_runtime_free, tessl_runtime_new, TESSL_OK};
use tessl::capi_embedgemma2::{self as c, TesslEmbedGemma2};

const ERR_LEN: usize = 4096;
const SIDE: usize = 224;

fn text(i: usize) -> String {
    format!("bench text {i}: {}", "lorem ipsum dolor sit amet ".repeat(3))
}

fn image(j: usize) -> Vec<u8> {
    (0..SIDE * SIDE * 3).map(|k| ((131 * k + 17 * j) % 251) as u8).collect()
}

/// FNV-1a over the little-endian bytes of the values' bits.
fn checksum(out: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in out {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn message(buf: &[c_char]) -> String {
    // SAFETY: the buffer is zero-filled before each call and tessl writes a
    // NUL-terminated string into it, so a terminator always exists.
    unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

struct Stats {
    name: String,
    min_ns: u64,
    median_ns: u64,
    iters: usize,
    checksum: u64,
}

/// Runs `once` (which returns the nanoseconds its tessl call took) `warmup`
/// times unmeasured, then `iters` times, and returns the minimum and median.
/// The caller sets the checksum from the output the last call left behind.
fn measure(name: &str, warmup: usize, iters: usize, once: &mut impl FnMut() -> u64) -> Stats {
    for _ in 0..warmup {
        once();
    }
    let mut times: Vec<u64> = (0..iters).map(|_| once()).collect();
    times.sort_unstable();
    Stats {
        name: name.to_string(),
        min_ns: times[0],
        median_ns: times[times.len() / 2],
        iters,
        checksum: 0,
    }
}

/// Makes a runtime and a model on the calling thread, runs every workload and
/// frees both. Returns the embedding width and a row per workload.
fn run(dir: &str, iters: usize, warmup: usize) -> (usize, Vec<Stats>) {
    let mut err = [0 as c_char; ERR_LEN];
    // SAFETY: `err` is ERR_LEN writable bytes.
    let rt = unsafe { tessl_runtime_new(err.as_mut_ptr(), ERR_LEN) };
    assert!(!rt.is_null(), "runtime: {}", message(&err));
    let mut model: *mut TesslEmbedGemma2 = ptr::null_mut();
    // SAFETY: `dir` is valid UTF-8 alive for the call; `model` and `err` are writable.
    let status = unsafe {
        c::tessl_embedgemma2_load_multimodal(
            rt,
            dir.as_ptr().cast(),
            dir.len() as u64,
            0,
            &mut model,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    assert_eq!(status, TESSL_OK, "load: {}", message(&err));
    let mut dim = 0u32;
    // SAFETY: the handle is live; `dim` and `err` are writable.
    let status = unsafe { c::tessl_embedgemma2_embedding_dim(model, &mut dim, err.as_mut_ptr(), ERR_LEN) };
    assert_eq!(status, TESSL_OK, "dim: {}", message(&err));
    let dim = dim as usize;

    let mut results = Vec::new();

    for &n in &[1usize, 4, 16, 64, 256] {
        let texts: Vec<String> = (0..n).map(text).collect();
        let mut bytes = Vec::new();
        let mut offsets = vec![0u64];
        for t in &texts {
            bytes.extend_from_slice(t.as_bytes());
            offsets.push(bytes.len() as u64);
        }
        let mut out = vec![0f32; n * dim];
        let mut call = || -> u64 {
            err.fill(0);
            let t = Instant::now();
            // SAFETY: every pointer addresses a live buffer of the length passed.
            let status = unsafe {
                c::tessl_embedgemma2_embed_texts(
                    model,
                    bytes.as_ptr(),
                    bytes.len() as u64,
                    offsets.as_ptr(),
                    n as u64,
                    ptr::null(),
                    0,
                    0,
                    out.as_mut_ptr(),
                    out.len() as u64,
                    err.as_mut_ptr(),
                    ERR_LEN,
                )
            };
            let ns = t.elapsed().as_nanos() as u64;
            assert_eq!(status, TESSL_OK, "embed_texts x{n}: {}", message(&err));
            ns
        };
        let mut s = measure(&format!("texts x{n}"), warmup, iters, &mut call);
        s.checksum = checksum(&out);
        results.push(s);
    }

    for &n in &[1usize, 8] {
        let pix: Vec<u8> = (0..n).flat_map(image).collect();
        let dims: Vec<u32> = (0..n).flat_map(|_| [SIDE as u32, SIDE as u32, 3]).collect();
        let mut out = vec![0f32; n * dim];
        let mut call = || -> u64 {
            err.fill(0);
            let t = Instant::now();
            // SAFETY: every pointer addresses a live buffer of the length passed.
            let status = unsafe {
                c::tessl_embedgemma2_embed_images(
                    model,
                    pix.as_ptr(),
                    pix.len() as u64,
                    dims.as_ptr(),
                    n as u64,
                    0,
                    out.as_mut_ptr(),
                    out.len() as u64,
                    err.as_mut_ptr(),
                    ERR_LEN,
                )
            };
            let ns = t.elapsed().as_nanos() as u64;
            assert_eq!(status, TESSL_OK, "embed_images x{n}: {}", message(&err));
            ns
        };
        let mut s = measure(&format!("images x{n}"), warmup, iters, &mut call);
        s.checksum = checksum(&out);
        results.push(s);
    }

    // SAFETY: made on this thread above, freed once, model before runtime.
    unsafe {
        assert_eq!(c::tessl_embedgemma2_free(model), TESSL_OK);
        assert_eq!(tessl_runtime_free(rt), TESSL_OK);
    }
    (dim, results)
}

fn json(rows: &[Stats]) -> String {
    let rows: Vec<String> = rows
        .iter()
        .map(|s| {
            format!(
                "{{\"workload\":\"{}\",\"min_ns\":{},\"median_ns\":{},\"iters\":{},\"checksum\":\"{:016x}\"}}",
                s.name, s.min_ns, s.median_ns, s.iters, s.checksum
            )
        })
        .collect();
    format!("[{}]", rows.join(","))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut snapshot = None;
    let (mut iters, mut warmup) = (100usize, 5usize);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--snapshot" => snapshot = it.next().cloned(),
            "--iters" => iters = it.next().and_then(|v| v.parse().ok()).expect("--iters needs a number"),
            "--warmup" => warmup = it.next().and_then(|v| v.parse().ok()).expect("--warmup needs a number"),
            other => panic!("unknown argument {other:?}"),
        }
    }
    let dir = snapshot.expect("--snapshot DIR is required");
    assert!(iters >= 1, "--iters must be at least 1");

    let (dim, main_rows) = run(&dir, iters, warmup);
    let worker_dir = dir.clone();
    let (worker_dim, worker_rows) = std::thread::Builder::new()
        .name("bench-direct".to_string())
        .stack_size(8 << 20)
        .spawn(move || run(&worker_dir, iters, warmup))
        .expect("spawning the worker thread")
        .join()
        .expect("the worker thread panicked");
    assert_eq!(dim, worker_dim);
    println!(
        "{{\"dim\":{dim},\"main\":{},\"worker\":{}}}",
        json(&main_rows),
        json(&worker_rows)
    );
}
