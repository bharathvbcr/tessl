//! The Hugging Face `tokenizers` crate, natively, on the corpus
//! `bench_embedgemma2_tok` and `bench/embedgemma2_tok_hf.py` read.
//!
//! ```text
//! cargo build --release --manifest-path bench/tok_hf_native/Cargo.toml
//! BENCH_TOK_CORPUS=target/embedgemma2_tok_bench_corpus.bin \
//!   bench/tok_hf_native/target/release/tok_hf_native
//! ```
//!
//! The Python lane times `tokenizers` through its binding, so its numbers
//! carry the per-call Python and PyO3 cost; this lane is the same library
//! version (0.23.2) without it. Environment, corpus, snapshot lookup and
//! output are the Python lane's: `BENCH_TOK_CORPUS` (required),
//! `EMBEDGEMMA2_SNAPSHOT` (default the newest Hugging Face cache snapshot
//! holding a `config.json`), `BENCH_ITERS` and `BENCH_LOAD_ITERS` (default
//! 5), `BENCH_TOK_CHECK=1` or `--check` (hashes only, no timing). The last
//! stdout line is one JSON array, `{workload, ms_min, ms_median, iters,
//! tokens, bytes, ids_fnv32}` per workload and `load`, which
//! `bench_paired --format workloads` takes; `ms_median` is
//! `sorted[len / 2]`, as the Python lane takes it.
//!
//! `BENCH_TOK_HF_MODE` picks what a pass calls, all with specials added
//! (`add_special_tokens=True` in Python, `encode_raw(text, true)` in tessl):
//! - `loop` (default): `encode_char_offsets` per text, which is what Python's
//!   `Tokenizer.encode` runs. Against the Python lane's `loop`, the
//!   difference is the binding.
//! - `loop_fast`: `encode_fast` per text, which skips the character offsets.
//!   tessl returns ids only, so this is the like-for-like lane against it.
//! - `batch`: `encode_batch_char_offsets` over the workload, which is what
//!   Python's `encode_batch` runs, on all cores (rayon).
//! - `batch_fast`: `encode_batch_fast` over the workload.
//!
//! What is timed: the encode of every text of the workload, from `&str` to
//! the library's `Encoding`; not the corpus read, not the hash.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::exit;
use std::time::Instant;

use tokenizers::{Encoding, Tokenizer};

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Loop,
    LoopFast,
    Batch,
    BatchFast,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Loop => "loop",
            Mode::LoopFast => "loop_fast",
            Mode::Batch => "batch",
            Mode::BatchFast => "batch_fast",
        }
    }
}

fn fail(msg: impl std::fmt::Display) -> ! {
    eprintln!("tok_hf_native: {msg}");
    exit(2)
}

fn env_count(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Err(_) => default,
        Ok(v) => match v.parse::<usize>() {
            Ok(n) if n >= 1 => n,
            _ => fail(format!("{name}={v:?}: expected an integer of at least 1")),
        },
    }
}

/// `EMBEDGEMMA2_SNAPSHOT`, or the last (sorted) snapshot directory of the
/// Hugging Face cache that holds a `config.json`, as the Python lane picks it.
fn snapshot() -> PathBuf {
    if let Some(p) = std::env::var_os("EMBEDGEMMA2_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var_os("HOME").unwrap_or_else(|| fail("HOME is not set"));
    let root = PathBuf::from(home).join(".cache/huggingface/hub/models--google--embeddinggemma-2/snapshots");
    let entries =
        std::fs::read_dir(&root).unwrap_or_else(|e| fail(format!("{}: {e}; set EMBEDGEMMA2_SNAPSHOT", root.display())));
    let mut hits = Vec::new();
    for entry in entries {
        let dir = entry
            .unwrap_or_else(|e| fail(format!("{}: {e}", root.display())))
            .path();
        if dir.join("config.json").is_file() {
            hits.push(dir);
        }
    }
    hits.sort();
    hits.pop()
        .unwrap_or_else(|| fail("no google/embeddinggemma-2 snapshot; set EMBEDGEMMA2_SNAPSHOT"))
}

/// Records `name TAB text` separated by NUL, grouped by name in the order of
/// first appearance.
fn read_corpus(path: &Path) -> Vec<(String, Vec<String>)> {
    let raw = std::fs::read(path).unwrap_or_else(|e| fail(format!("{}: {e}", path.display())));
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    for rec in raw.split(|&b| b == 0).filter(|r| !r.is_empty()) {
        let rec = std::str::from_utf8(rec)
            .unwrap_or_else(|e| fail(format!("{}: a record is not UTF-8: {e}", path.display())));
        let (name, text) = rec.split_once('\t').unwrap_or((rec, ""));
        // Names go into the JSON unescaped.
        if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            fail(format!("workload name {name:?} is not [A-Za-z0-9_]+"));
        }
        // `load` is this lane's own row; a workload of that name would be a
        // second row with the same key.
        if name == "load" {
            fail("the corpus has a workload named \"load\", which is the tokenizer-load row's name");
        }
        match out.iter_mut().find(|(n, _)| n == name) {
            Some((_, texts)) => texts.push(text.to_string()),
            None => out.push((name.to_string(), vec![text.to_string()])),
        }
    }
    if out.is_empty() {
        fail(format!("{}: no records", path.display()));
    }
    out
}

fn fnv32(hash: &mut u32, ids: &[u32]) {
    for id in ids {
        for b in id.to_le_bytes() {
            *hash = (*hash ^ u32::from(b)).wrapping_mul(16_777_619);
        }
    }
}

fn encode_pass(tok: &Tokenizer, mode: Mode, texts: &[String]) -> Vec<Encoding> {
    // The batch calls take the inputs by value, as the binding builds them
    // from its Python list on every call; the loops borrow each text.
    let batch = || texts.iter().map(String::as_str).collect::<Vec<_>>();
    let got = match mode {
        Mode::Loop => texts
            .iter()
            .map(|t| tok.encode_char_offsets(t.as_str(), true))
            .collect(),
        Mode::LoopFast => texts.iter().map(|t| tok.encode_fast(t.as_str(), true)).collect(),
        Mode::Batch => tok.encode_batch_char_offsets(batch(), true),
        Mode::BatchFast => tok.encode_batch_fast(batch(), true),
    };
    got.unwrap_or_else(|e| fail(format!("encode ({}): {e}", mode.name())))
}

struct Row {
    workload: String,
    ms_min: f64,
    ms_median: f64,
    iters: usize,
    tokens: usize,
    bytes: usize,
    ids_fnv32: u32,
}

fn min_median(mut ms: Vec<f64>) -> (f64, f64) {
    if ms.is_empty() {
        return (0.0, 0.0);
    }
    ms.sort_by(f64::total_cmp);
    (ms[0], ms[ms.len() / 2])
}

fn main() {
    let corpus = std::env::var_os("BENCH_TOK_CORPUS")
        .map(PathBuf::from)
        .unwrap_or_else(|| fail("BENCH_TOK_CORPUS must name the corpus (see the module docs)"));
    let check = std::env::args().any(|a| a == "--check") || std::env::var("BENCH_TOK_CHECK").is_ok_and(|v| v == "1");
    let mode = match std::env::var("BENCH_TOK_HF_MODE").as_deref() {
        Err(_) | Ok("loop") => Mode::Loop,
        Ok("loop_fast") => Mode::LoopFast,
        Ok("batch") => Mode::Batch,
        Ok("batch_fast") => Mode::BatchFast,
        Ok(other) => fail(format!(
            "BENCH_TOK_HF_MODE={other:?}: expected loop, loop_fast, batch or batch_fast"
        )),
    };
    let iters = env_count("BENCH_ITERS", 5);
    let load_iters = env_count("BENCH_LOAD_ITERS", 5);
    let json_path = snapshot().join("tokenizer.json");
    let workloads = read_corpus(&corpus);

    let mut loads = Vec::with_capacity(load_iters);
    let mut tok = None;
    for _ in 0..load_iters {
        let t0 = Instant::now();
        let t = Tokenizer::from_file(&json_path).unwrap_or_else(|e| fail(format!("{}: {e}", json_path.display())));
        loads.push(t0.elapsed().as_secs_f64() * 1e3);
        tok = Some(t);
    }
    let tok = tok.unwrap_or_else(|| fail("BENCH_LOAD_ITERS is at least 1"));
    let json_bytes = std::fs::metadata(&json_path)
        .unwrap_or_else(|e| fail(format!("{}: {e}", json_path.display())))
        .len();
    let (load_min, load_median) = min_median(loads);
    let mut rows = vec![Row {
        workload: "load".to_string(),
        ms_min: load_min,
        ms_median: load_median,
        iters: load_iters,
        tokens: tok.get_vocab_size(true),
        bytes: usize::try_from(json_bytes)
            .unwrap_or_else(|_| fail(format!("{}: {json_bytes} bytes do not fit usize", json_path.display()))),
        ids_fnv32: 0,
    }];

    for (name, texts) in &workloads {
        // The untimed first pass is the warm-up and gives the hash.
        let mut hash = 2_166_136_261u32;
        let mut tokens = 0usize;
        for enc in encode_pass(&tok, mode, texts) {
            tokens += enc.get_ids().len();
            fnv32(&mut hash, enc.get_ids());
        }
        let mut ms = Vec::with_capacity(iters);
        if !check {
            for _ in 0..iters {
                let t0 = Instant::now();
                black_box(encode_pass(&tok, mode, black_box(texts)));
                ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        let (ms_min, ms_median) = min_median(ms);
        rows.push(Row {
            workload: name.clone(),
            ms_min,
            ms_median,
            iters,
            tokens,
            bytes: texts.iter().map(String::len).sum(),
            ids_fnv32: hash,
        });
    }

    for r in &rows {
        let rate = if r.workload == "load" || r.ms_min == 0.0 {
            String::new()
        } else {
            format!("  {:.0} tokens/s", r.tokens as f64 / (r.ms_min / 1e3))
        };
        eprintln!(
            "{:<14} min {:>10.3} ms  median {:>10.3} ms  {} tokens, {} bytes{rate}  [native {}]",
            r.workload,
            r.ms_min,
            r.ms_median,
            r.tokens,
            r.bytes,
            mode.name()
        );
    }
    let objs: Vec<String> = rows
        .iter()
        .map(|r| {
            format!(
                "{{\"workload\":\"{}\",\"ms_min\":{:.6},\"ms_median\":{:.6},\"iters\":{},\"tokens\":{},\"bytes\":{},\"ids_fnv32\":{}}}",
                r.workload, r.ms_min, r.ms_median, r.iters, r.tokens, r.bytes, r.ids_fnv32
            )
        })
        .collect();
    println!("[{}]", objs.join(","));
}
