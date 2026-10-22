//! Load time and encode throughput of `tessl::embedgemma2_tok` on a fixed corpus.
//!
//! ```text
//! ~/.venvs/ml-2026-10/bin/python bench/embedgemma2_tok_hf.py corpus target/embedgemma2_tok_bench_corpus.bin
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//! BENCH_TOK_CORPUS=target/embedgemma2_tok_bench_corpus.bin \
//!   cargo run --release --bin bench_embedgemma2_tok
//! ```
//!
//! The corpus is written by `bench/embedgemma2_tok_hf.py corpus` (seeded and
//! deterministic) so both lanes read the same bytes: records `name TAB text`
//! separated by NUL, in the workload order of first appearance (`short_queries`,
//! `documents`, `multilingual`, `long_100k`). The snapshot's `tokenizer.json`
//! is the only file read, as the Python lane's `Tokenizer.from_file` reads it.
//!
//! Prints one JSON array on the last stdout line, in the shape
//! `bench_paired --format workloads` takes: one object per workload,
//! `{workload, ms_min, ms_median, iters, tokens, bytes, ids_fnv32}`, plus the
//! `load` workload. `ms_min` is the fastest of `BENCH_ITERS` (default 5) full
//! passes over the workload's texts, each encoded alone with `encode_raw`
//! (the library's `encode` with specials, one thread), after one warm-up pass;
//! for `load` it is the fastest of `BENCH_LOAD_ITERS` (default 5) parses of
//! `tokenizer.json` into a [`Tokenizer`]. `tokens` is the ids a pass produced
//! and `ids_fnv32` the FNV-1a hash of all of them in order, which the Python
//! lane computes the same way: a lane that produced other ids than the
//! other would be caught by the comparison, not timed as if it had not.
//!
//! `BENCH_TOK_MODE=batch` times `Tokenizer::encode_batch` over the workload
//! instead (the default, `loop`, is the one-thread loop above): it spreads the
//! batch over the machine's cores (`TESSL_TOK_THREADS` sets the count), the
//! lane to read against `bench/tok_hf_native` in `BENCH_TOK_HF_MODE=batch` or
//! `batch_fast`. The hash is the same on every mode.
//!
//! What is timed: the encode of every text of the workload, from `&str` to
//! `Vec<u32>`; not the corpus read and not the hash.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use tessl::embedgemma2_tok::Tokenizer;

fn env_count(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Err(_) => default,
        Ok(v) => match v.parse::<usize>() {
            Ok(n) if n >= 1 => n,
            _ => {
                eprintln!("{name}={v:?}: expected an integer of at least 1");
                std::process::exit(2);
            }
        },
    }
}

fn env_path(name: &str) -> PathBuf {
    match std::env::var_os(name) {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("{name} must be set (see the module docs)");
            std::process::exit(2);
        }
    }
}

fn fnv32(hash: &mut u32, ids: &[u32]) {
    for &id in ids {
        for b in id.to_le_bytes() {
            *hash ^= u32::from(b);
            *hash = hash.wrapping_mul(16_777_619);
        }
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

struct Row {
    workload: String,
    ms_min: f64,
    ms_median: f64,
    iters: usize,
    tokens: usize,
    bytes: usize,
    fnv: u32,
}

fn main() {
    let snapshot = env_path("EMBEDGEMMA2_SNAPSHOT");
    let corpus_path = env_path("BENCH_TOK_CORPUS");
    let iters = env_count("BENCH_ITERS", 5);
    let load_iters = env_count("BENCH_LOAD_ITERS", 5);
    let json_path = snapshot.join("tokenizer.json");
    let batch = match std::env::var("BENCH_TOK_MODE").as_deref() {
        Err(_) | Ok("loop") => false,
        Ok("batch") => true,
        Ok(other) => {
            eprintln!("BENCH_TOK_MODE={other:?}: expected loop or batch");
            std::process::exit(2);
        }
    };

    let raw = std::fs::read(&corpus_path).unwrap_or_else(|e| {
        eprintln!("{}: {e}", corpus_path.display());
        std::process::exit(1);
    });
    let mut workloads: Vec<(String, Vec<String>)> = Vec::new();
    for record in raw.split(|&b| b == 0).filter(|r| !r.is_empty()) {
        let text = std::str::from_utf8(record).expect("the corpus is UTF-8");
        let (name, body) = text.split_once('\t').expect("a record is `name TAB text`");
        match workloads.iter_mut().find(|(n, _)| n == name) {
            Some((_, texts)) => texts.push(body.to_string()),
            None => workloads.push((name.to_string(), vec![body.to_string()])),
        }
    }
    assert!(!workloads.is_empty(), "the corpus is empty");

    let mut rows: Vec<Row> = Vec::new();

    let mut loads = Vec::with_capacity(load_iters);
    let mut tok = None;
    for _ in 0..load_iters {
        let t0 = Instant::now();
        let t = Tokenizer::from_file(&json_path).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
        loads.push(t0.elapsed().as_secs_f64() * 1e3);
        tok = Some(black_box(t));
    }
    let tok = tok.expect("at least one load");
    let load_bytes = std::fs::metadata(&json_path).map(|m| m.len() as usize).unwrap_or(0);
    rows.push(Row {
        workload: "load".into(),
        ms_min: loads.iter().copied().fold(f64::INFINITY, f64::min),
        ms_median: median(&mut loads.clone()),
        iters: load_iters,
        tokens: tok.vocab_size(),
        bytes: load_bytes,
        fnv: 0,
    });

    for (name, texts) in &workloads {
        let bytes: usize = texts.iter().map(String::len).sum();
        // Warm-up pass, which also fixes the token count and the hash.
        let mut fnv = 2_166_136_261u32;
        let mut tokens = 0usize;
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        if batch {
            for ids in tok.encode_batch(&refs).expect("encode_batch") {
                tokens += ids.len();
                fnv32(&mut fnv, &ids);
            }
        } else {
            for t in texts {
                let ids = tok.encode_raw(t, true);
                tokens += ids.len();
                fnv32(&mut fnv, &ids);
            }
        }
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            if batch {
                black_box(tok.encode_batch(black_box(&refs)).expect("encode_batch"));
            } else {
                for t in texts {
                    black_box(tok.encode_raw(black_box(t), true));
                }
            }
            times.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        rows.push(Row {
            workload: name.clone(),
            ms_min: times.iter().copied().fold(f64::INFINITY, f64::min),
            ms_median: median(&mut times.clone()),
            iters,
            tokens,
            bytes,
            fnv,
        });
    }

    for r in &rows {
        eprintln!(
            "{:<14} min {:>10.3} ms  median {:>10.3} ms  {} tokens, {} bytes{}",
            r.workload,
            r.ms_min,
            r.ms_median,
            r.tokens,
            r.bytes,
            if r.workload == "load" {
                String::new()
            } else {
                format!("  {:.0} tokens/s", r.tokens as f64 / (r.ms_min / 1e3))
            }
        );
    }
    let objs: Vec<String> = rows
        .iter()
        .map(|r| {
            format!(
                "{{\"workload\":\"{}\",\"ms_min\":{:.6},\"ms_median\":{:.6},\"iters\":{},\"tokens\":{},\"bytes\":{},\"ids_fnv32\":{}}}",
                r.workload, r.ms_min, r.ms_median, r.iters, r.tokens, r.bytes, r.fnv
            )
        })
        .collect();
    println!("[{}]", objs.join(","));
}
