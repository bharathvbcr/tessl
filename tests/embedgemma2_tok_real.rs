//! `tessl::embedgemma2_tok` against the real `google/embeddinggemma-2`
//! tokenizer: exact id equality with what sentence-transformers feeds the model
//! (and with the Hugging Face `tokenizers` library alone), and exact string
//! equality of decode, over the corpus `scripts/gen_embedgemma2_tok_fixtures.py
//! real` writes. Opt-in:
//!
//! ```text
//! ~/.venvs/ml-2026-10/bin/python scripts/gen_embedgemma2_tok_fixtures.py real   # once
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//!   cargo test --release --test embedgemma2_tok_real -- --ignored --test-threads=1
//! ```
//!
//! `EMBEDGEMMA2_TOK_REF_DIR` overrides `target/embedgemma2_tok_ref`. A missing
//! variable or file fails the test rather than skipping it. There is no
//! tolerance: one differing id fails the case, and a failure prints the first
//! difference. The corpus: every task prompt with sample texts, whitespace and
//! line endings, CJK, Devanagari, Arabic, Thai and other scripts, combining
//! marks, emoji (ZWJ sequences, flags), control and private-use characters,
//! characters with no vocabulary entry (byte fallback), special-token strings
//! whole and in pieces, the three refused placeholders, digits, code, texts of
//! 50k to 100k characters, and 4000 seeded random strings over all of those.

mod common;

use std::path::PathBuf;
use std::time::Instant;

use common::embedgemma2_tok::{brief, first_diff, Corpus, Tally};
use tessl::embedgemma2_tok::Tokenizer;

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("EMBEDGEMMA2_SNAPSHOT")
            .expect("EMBEDGEMMA2_SNAPSHOT must name the google/embeddinggemma-2 snapshot directory"),
    )
}

fn ref_dir() -> PathBuf {
    std::env::var_os("EMBEDGEMMA2_TOK_REF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/embedgemma2_tok_ref"))
}

fn load() -> Tokenizer {
    let t0 = Instant::now();
    let tok = Tokenizer::from_snapshot(&snapshot()).expect("load the snapshot's tokenizer");
    eprintln!(
        "load: {:.1} ms, {} tokens",
        t0.elapsed().as_secs_f64() * 1e3,
        tok.vocab_size()
    );
    tok
}

#[test]
#[ignore]
fn encode_is_what_sentence_transformers_feeds_the_model() {
    let tok = load();
    let c = Corpus::load(&ref_dir());
    let mut st = Tally::default();
    let mut raw = Tally::default();
    for i in 0..c.texts.len() {
        let (text, group) = (&c.texts[i], &c.group[i]);
        let prompt = c.prompt[i].as_deref();
        let rendered = format!(
            "{}{}",
            prompt.map_or("", |p| tok.prompt(p).expect("a prompt of the snapshot")),
            text
        );

        // The library alone, on the rendered text, with and without specials.
        let got = tok.encode_raw(&rendered, true);
        raw.record(group, got == c.raw_ids[i], || {
            format!("raw {} {}", brief(&rendered), first_diff(&got, &c.raw_ids[i]))
        });
        let got = tok.encode_raw(&rendered, false);
        raw.record(group, got == c.plain_ids[i], || {
            format!("plain {} {}", brief(&rendered), first_diff(&got, &c.plain_ids[i]))
        });

        // sentence-transformers' path.
        let res = tok.encode_with_prompt(prompt, text);
        if c.status[i] == 0 {
            let got = res.unwrap_or_else(|e| panic!("{}: refused: {e}", brief(text)));
            st.record(group, got == c.st_ids[i], || {
                format!("{} (prompt {prompt:?}) {}", brief(text), first_diff(&got, &c.st_ids[i]))
            });
        } else {
            // The processor raises; so do we, and the ids the library alone gives (with the
            // `<bos>` rule) are the oracle's for `encode_raw`.
            st.record(group, res.is_err(), || format!("{} should be refused", brief(text)));
            let got = tok.encode_raw(&rendered, !rendered.starts_with("<bos>"));
            st.record(group, got == c.st_ids[i], || {
                format!("refused {} {}", brief(text), first_diff(&got, &c.st_ids[i]))
            });
        }
    }
    raw.finish("encode_raw vs tokenizers");
    st.finish("encode vs sentence-transformers");
}

#[test]
#[ignore]
fn decode_is_the_tokenizers_decode() {
    let tok = load();
    let c = Corpus::load(&ref_dir());
    let mut t = Tally::default();
    for i in 0..c.dec_ids.len() {
        let got = tok.decode_with(&c.dec_ids[i], c.dec_skip[i]);
        t.record(&c.dec_group[i], got == c.dec_out[i], || {
            format!(
                "decode of {:?} (skip {}): got {} want {}",
                &c.dec_ids[i][..c.dec_ids[i].len().min(12)],
                c.dec_skip[i],
                brief(&got),
                brief(&c.dec_out[i])
            )
        });
    }
    // The default skips special tokens, as `Tokenizer.decode` does.
    for ids in [vec![2u32, 2202, 1], vec![105, 3, 4]] {
        assert_eq!(tok.decode(&ids), tok.decode_with(&ids, true));
    }
    t.finish("decode vs tokenizers");
}

/// sentence-transformers decides `<bos>`/`<eos>` for a whole batch from its
/// first text (`scripts/gen_embedgemma2_tok_fixtures.py` asserts this against
/// the installed transformers: `["<bos>hi", "plain"]` gives `plain` neither
/// `<bos>` nor `<eos>`). Here each text stands alone.
#[test]
#[ignore]
fn a_batch_is_each_text_alone() {
    let tok = load();
    let hi = tok.encode("<bos>hi").unwrap();
    let plain = tok.encode("plain").unwrap();
    assert_eq!(
        hi,
        [2, 2202],
        "a text that starts with <bos> gets no second <bos> and no <eos>"
    );
    assert_eq!(plain, [2, 30448, 1]);
    assert_eq!(
        tok.encode_batch(&["<bos>hi", "plain"]).unwrap(),
        [hi.clone(), plain.clone()]
    );
    assert_eq!(tok.encode_batch(&["plain", "<bos>hi"]).unwrap(), [plain, hi]);
}

#[test]
#[ignore]
fn the_named_prompts_are_the_checkpoints() {
    let tok = load();
    assert_eq!(tok.prompt_names().count(), 20);
    assert_eq!(tok.prompt("query"), Some("task: search result | query: "));
    assert_eq!(tok.prompt("document"), Some("title: none | text: "));
    assert_eq!(tok.prompt("CodeRetrieval"), Some("task: code retrieval | query: "));
    assert_eq!(tok.prompt("nope"), None);
    let err = tok.encode_with_prompt(Some("nope"), "x").unwrap_err();
    assert!(err.contains("no prompt named \"nope\""), "{err}");
    // A prompt is text in front of the text: its ids are the prompt's then the text's.
    let with = tok.encode_with_prompt(Some("query"), "hello").unwrap();
    let rendered = tok.encode("task: search result | query: hello").unwrap();
    assert_eq!(with, rendered);
}
