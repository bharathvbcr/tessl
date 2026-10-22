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
    "metal", "kernel", "attention", "window", "encoder", "residual", "gradient", "tensor", "embedding", "retrieval",
    "query", "document", "cosine", "latency", "batch", "pooling", "sequence", "token", "position", "rotary", "apple",
    "silicon", "model", "error", "memory", "cache", "layer", "weights", "float", "benchmark", "throughput", "a", "b",
    "ab", "abc", "<a>", "<a><b>", "<bos>",
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
