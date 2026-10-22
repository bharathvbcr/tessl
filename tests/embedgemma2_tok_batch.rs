//! `Tokenizer::encode_batch` on the tiny fixture tokenizers, with no snapshot
//! and no device: the ids it returns are the ids `encode` returns text by text,
//! whatever the thread count. The 200-document workload is generated here from
//! a fixed seed (the shape of `bench/embedgemma2_tok_hf.py`'s `documents`:
//! 120 to 400 words each, with digits, punctuation, code lines and scripts
//! beyond ASCII), so the pinned hashes need no corpus file.

use std::path::{Path, PathBuf};

use tessl::embedgemma2_tok::Tokenizer;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embedgemma2_tok")
}

fn load(name: &str) -> Tokenizer {
    Tokenizer::from_file(&fixtures().join(format!("{name}.json"))).unwrap()
}

/// SplitMix64: the same seeded generator on every platform.
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
        (self.next() % n as u64) as usize
    }
}

const WORDS: &[&str] = &[
    "metal",
    "kernel",
    "attention",
    "window",
    "encoder",
    "residual",
    "gradient",
    "tensor",
    "embedding",
    "retrieval",
    "query",
    "document",
    "cosine",
    "latency",
    "batch",
    "pooling",
    "sequence",
    "token",
    "position",
    "rotary",
    "apple",
    "silicon",
    "model",
    "error",
    "memory",
    "cache",
    "layer",
    "weights",
    "float",
    "benchmark",
    "throughput",
    "a",
    "b",
    "ab",
    "abc",
    "<a>",
    "<a><b>",
    "<bos>",
];
const CODE: &[&str] = &[
    "fn main() { let v: Vec<u32> = (0..10).collect(); }",
    "SELECT a, COUNT(*) FROM t WHERE b >= 10 GROUP BY a;",
    "{\"a\": [1, 2.5e-3, null], \"b\": {\"c\": \"d\"}}",
];
const SCRIPTS: &[&str] = &[
    "混合精度训练与推理 東京で会いましょう",
    "नमस्ते दुनिया हिन्दी भाषा में पाठ",
    "مرحبا بالعالم اللغة العربية",
    "Привет мир Γειά σου Κόσμε",
    "😀 🚀 👨‍👩‍👧‍👦 🇯🇵 ❤️",
];

/// `n` documents of `lo..hi` words.
fn documents(seed: u64, n: usize, lo: usize, hi: usize) -> Vec<String> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let words = lo + rng.below(hi - lo);
            let mut s = String::new();
            for w in 0..words {
                if w > 0 {
                    s.push(if rng.below(40) == 0 { '\n' } else { ' ' });
                }
                match rng.below(30) {
                    0 => s.push_str(CODE[rng.below(CODE.len())]),
                    1 => s.push_str(SCRIPTS[rng.below(SCRIPTS.len())]),
                    2 => s.push_str(&rng.below(1000).to_string()),
                    _ => {
                        s.push_str(WORDS[rng.below(WORDS.len())]);
                        if rng.below(12) == 0 {
                            s.push(',');
                        }
                    }
                }
            }
            s
        })
        .collect()
}

fn fnv32(batch: &[Vec<u32>]) -> (u32, usize) {
    let mut h = 2_166_136_261u32;
    let mut tokens = 0;
    for ids in batch {
        tokens += ids.len();
        for &id in ids {
            for b in id.to_le_bytes() {
                h ^= u32::from(b);
                h = h.wrapping_mul(16_777_619);
            }
        }
    }
    (h, tokens)
}

fn refs(texts: &[String]) -> Vec<&str> {
    texts.iter().map(String::as_str).collect()
}

const DOCS_SEED: u64 = 20_261_020;

#[test]
fn the_200_document_workload_hashes_as_recorded() {
    let docs = documents(DOCS_SEED, 200, 120, 400);
    let bytes: usize = docs.iter().map(String::len).sum();
    let mut seen = Vec::new();
    for name in ["tiny_gemma", "tiny_legacy", "tiny_bytes"] {
        let tok = load(name);
        let one_by_one: Vec<Vec<u32>> = docs.iter().map(|d| tok.encode(d).unwrap()).collect();
        let batch = tok.encode_batch(&refs(&docs)).unwrap();
        assert_eq!(batch, one_by_one, "{name}: encode_batch is encode, text by text");
        seen.push((name, bytes, fnv32(&batch)));
    }
    eprintln!("{seen:?}");
    assert_eq!(seen, PINNED);
}

/// Recorded from the single-threaded `encode_batch` before it was spread
/// across threads: (fixture, bytes of the workload, (FNV-1a of all ids, ids)).
const PINNED: [(&str, usize, (u32, usize)); 3] = [
    ("tiny_gemma", 525_726, (1_912_934_318, 499_764)),
    ("tiny_legacy", 525_726, (4_280_043_287, 214_558)),
    ("tiny_bytes", 525_726, (1_359_547_695, 458_276)),
];

/// The thread counts every comparison runs: serial, small, odd, many, the cap.
const COUNTS: [usize; 6] = [1, 2, 3, 8, 64, 1000];

fn serial(tok: &Tokenizer, texts: &[&str]) -> Vec<Vec<u32>> {
    texts.iter().map(|t| tok.encode(t).unwrap()).collect()
}

#[test]
fn the_hash_is_the_recorded_one_at_every_thread_count() {
    let docs = documents(DOCS_SEED, 200, 120, 400);
    let texts = refs(&docs);
    for (name, bytes, hash) in PINNED {
        let tok = load(name);
        assert_eq!(bytes, docs.iter().map(String::len).sum::<usize>());
        for threads in COUNTS {
            let got = tok.encode_batch_with_threads(None, &texts, threads).unwrap();
            assert_eq!(fnv32(&got), hash, "{name}, {threads} threads");
        }
    }
}

#[test]
fn an_empty_batch_a_single_text_and_empty_strings() {
    let tok = load("tiny_gemma");
    for threads in COUNTS {
        assert_eq!(
            tok.encode_batch_with_threads(None, &[], threads).unwrap(),
            Vec::<Vec<u32>>::new()
        );
        let one = tok
            .encode_batch_with_threads(None, &["hello <a><b> world"], threads)
            .unwrap();
        assert_eq!(one, vec![tok.encode("hello <a><b> world").unwrap()]);
        let blanks = vec![""; 40];
        let got = tok.encode_batch_with_threads(None, &blanks, threads).unwrap();
        assert_eq!(got, vec![tok.encode("").unwrap(); 40]);
    }
}

#[test]
fn skewed_unicode_and_over_long_texts_keep_their_order() {
    let tok = load("tiny_bytes");
    let long = documents(7, 1, 40_000, 40_001).remove(0);
    let docs = documents(11, 30, 5, 3000);
    let mut texts: Vec<&str> = vec!["", "x", &long, "", "é", "😀 🚀 👨‍👩‍👧‍👦", &long];
    texts.extend(docs.iter().map(String::as_str));
    texts.push("");
    texts.push(&long);
    let want = serial(&tok, &texts);
    for threads in COUNTS {
        assert_eq!(
            tok.encode_batch_with_threads(None, &texts, threads).unwrap(),
            want,
            "{threads} threads"
        );
    }
}

/// `tiny_gemma` with a `<bos>` rule, three prompts and `<a><b>` as a refused
/// placeholder.
fn configured() -> Tokenizer {
    load("tiny_gemma")
        .with_tokenizer_config(
            r#"{"bos_token": "<bos>", "image_token": "<a><b>", "model_max_length": 1000000000000000019884624838656}"#,
        )
        .unwrap()
        .with_prompts(r#"{"prompts": {"query": "hello ", "empty": "", "code": "<bos>"}, "default_prompt_name": null}"#)
        .unwrap()
}

#[test]
fn a_prompt_applies_to_every_text_whatever_the_thread_count() {
    let tok = configured();
    // No refused placeholder: leave out the generator's `<a><b>` word.
    let docs: Vec<String> = documents(5, 60, 100, 300)
        .iter()
        .map(|d| d.replace("<a><b>", "<a>"))
        .collect();
    let texts = refs(&docs);
    for prompt in [None, Some("query"), Some("empty"), Some("code")] {
        let want: Vec<Vec<u32>> = texts
            .iter()
            .map(|t| tok.encode_with_prompt(prompt, t).unwrap())
            .collect();
        for threads in COUNTS {
            assert_eq!(
                tok.encode_batch_with_threads(prompt, &texts, threads).unwrap(),
                want,
                "{prompt:?}"
            );
        }
    }
}

#[test]
fn the_first_failing_text_is_the_error_at_every_thread_count() {
    let tok = configured();
    let mut docs: Vec<String> = documents(3, 100, 100, 300)
        .iter()
        .map(|d| d.replace("<a><b>", "<a>"))
        .collect();
    for i in [70, 31, 99] {
        docs[i].push_str(" <a><b> ");
    }
    let texts = refs(&docs);
    for threads in COUNTS {
        let e = tok.encode_batch_with_threads(None, &texts, threads).unwrap_err();
        assert!(
            e.starts_with("text 31: ") && e.contains("placeholder"),
            "{threads} threads: {e}"
        );
    }
    // An unknown prompt refuses the first text, and an empty batch has none to refuse.
    for threads in COUNTS {
        let e = tok
            .encode_batch_with_threads(Some("nope"), &texts, threads)
            .unwrap_err();
        assert!(e.starts_with("text 0: no prompt named"), "{threads} threads: {e}");
        assert!(tok
            .encode_batch_with_threads(Some("nope"), &[], threads)
            .unwrap()
            .is_empty());
    }
}

#[test]
fn parallel_equals_serial_over_seeded_random_batches() {
    let toks = [load("tiny_gemma"), load("tiny_legacy"), load("tiny_bytes")];
    let mut rng = Rng(0xC0FFEE);
    for case in 0..60 {
        let n = rng.below(70);
        let texts: Vec<String> = (0..n)
            .map(|_| match rng.below(6) {
                0 => String::new(),
                1 => documents(rng.next(), 1, 1, 8).remove(0),
                2 => documents(rng.next(), 1, 2000, 9000).remove(0),
                _ => documents(rng.next(), 1, 20, 400).remove(0),
            })
            .collect();
        let texts = refs(&texts);
        let tok = &toks[case % 3];
        let want = serial(tok, &texts);
        let threads = 1 + rng.below(70);
        assert_eq!(
            tok.encode_batch_with_threads(None, &texts, threads).unwrap(),
            want,
            "case {case}: {n} texts, {threads} threads"
        );
        assert_eq!(
            tok.encode_batch(&texts).unwrap(),
            want,
            "case {case}: the configured thread count"
        );
    }
}

#[test]
fn the_tokenizer_is_shared_across_threads() {
    fn shared<T: Send + Sync>() {}
    shared::<Tokenizer>();
}
