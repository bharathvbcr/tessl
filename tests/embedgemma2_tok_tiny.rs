//! `tessl::embedgemma2_tok` on three hand-written tiny tokenizers, with no
//! snapshot and no device: the ids and decoded strings are the Hugging Face
//! `tokenizers` library's (`scripts/gen_embedgemma2_tok_fixtures.py tiny`
//! wrote them into `tests/fixtures/embedgemma2_tok/<name>/`), and exact
//! equality is the bound.
//!
//! * `tiny_gemma.json` has the snapshot's shape: a space-to-`U+2581` normalizer,
//!   the inert space split, byte fallback for every byte, `<bos>`/`<eos>`
//!   from a template, and added tokens, two of them in a prefix relation
//!   (`<a>`, `<a><b>`) so the leftmost-longest match is exercised.
//! * `tiny_legacy.json` has a real space split (merged with the previous
//!   piece), no normalizer, `"a b"` merge strings with a repeated merge,
//!   `ignore_merges`, `fuse_unk` and a plain added token.
//! * `tiny_bytes.json` has byte tokens for some characters only, so a run of
//!   unknown characters meets byte-fallback tokens (the library emits the
//!   run after them), `fuse_unk` off, and a two-step normalizer.
//!
//! The rest is what the library does not have: sentence-transformers' rules on
//! top (prompts, `<bos>`, refused placeholders), and the refusals of anything
//! this tokenizer does not implement.

mod common;

use std::path::{Path, PathBuf};

use common::embedgemma2_tok::{brief, first_diff, Corpus, Tally};
use tessl::embedgemma2_tok::Tokenizer;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embedgemma2_tok")
}

fn json_text(name: &str) -> String {
    std::fs::read_to_string(fixtures().join(format!("{name}.json"))).unwrap()
}

fn load(name: &str) -> Tokenizer {
    Tokenizer::from_file(&fixtures().join(format!("{name}.json"))).unwrap()
}

#[test]
fn the_tiny_tokenizers_match_the_library_exactly() {
    for name in ["tiny_gemma", "tiny_legacy", "tiny_bytes"] {
        let tok = load(name);
        let c = Corpus::load(&fixtures().join(name));
        assert!(
            c.texts.len() > 600 && c.dec_ids.len() > 700,
            "{name}: the corpus is there"
        );
        let mut t = Tally::default();
        for i in 0..c.texts.len() {
            let got = tok.encode_raw(&c.texts[i], true);
            t.record(name, got == c.raw_ids[i], || {
                format!("{} {}", brief(&c.texts[i]), first_diff(&got, &c.raw_ids[i]))
            });
            let got = tok.encode_raw(&c.texts[i], false);
            t.record(name, got == c.plain_ids[i], || {
                format!(
                    "{} (no specials) {}",
                    brief(&c.texts[i]),
                    first_diff(&got, &c.plain_ids[i])
                )
            });
        }
        for i in 0..c.dec_ids.len() {
            let got = tok.decode_with(&c.dec_ids[i], c.dec_skip[i]);
            t.record(name, got == c.dec_out[i], || {
                format!(
                    "decode {:?} skip {}: got {} want {}",
                    c.dec_ids[i],
                    c.dec_skip[i],
                    brief(&got),
                    brief(&c.dec_out[i])
                )
            });
        }
        t.finish(name);
    }
}

#[test]
fn the_fixtures_exercise_what_they_claim() {
    // The leftmost-longest match: `<a><b>` is one token, not `<a>` then `<b>`.
    let g = load("tiny_gemma");
    let (a, ab, b) = (
        g.token_to_id("<a>").unwrap(),
        g.token_to_id("<a><b>").unwrap(),
        g.token_to_id("<b>").unwrap(),
    );
    assert_eq!(g.encode_raw("<a><b>", false), [ab]);
    assert_eq!(g.encode_raw("<a><b><a>", false), [ab, a]);
    assert_eq!(g.encode_raw("<b><a>", false), [b, a]);
    // Byte fallback: 日 is E6 97 A5, no vocabulary entry of its own.
    let bytes: Vec<u32> = [0xE6u8, 0x97, 0xA5]
        .iter()
        .map(|b| g.token_to_id(&format!("<0x{b:02X}>")).unwrap())
        .collect();
    assert_eq!(g.encode_raw("日", false), bytes);
    // An added token is matched in the raw text, before the normalizer turns spaces into U+2581.
    assert_eq!(g.encode_raw("<|turn> <|turn>", false).len(), 3);

    // A real split: a space ends its piece ("ba " is b, a, space, merged to ba + space), and a
    // space after a space is a piece of its own; "ba" alone is one vocabulary entry.
    let l = load("tiny_legacy");
    let (sp, ba) = (l.token_to_id(" ").unwrap(), l.token_to_id("ba").unwrap());
    assert_eq!(l.encode_raw("ba  ba", false), [ba, sp, sp, ba]);
    // fuse_unk: a run of unknown characters is one <unk>.
    let unk = l.token_to_id("<unk>").unwrap();
    assert_eq!(l.encode_raw("zzzz", false), [unk]);
    // The template's trailing token, and a plain added token that decode keeps.
    let end = l.token_to_id("<end>").unwrap();
    assert_eq!(*l.encode_raw("ba", true).last().unwrap(), end);
    let x = l.token_to_id("[X]").unwrap();
    assert_eq!(l.decode(&[ba, end, x]), "ba[X]");
    assert_eq!(l.decode_with(&[ba, end, x], false), "ba<end>[X]");

    // An unknown-character run after byte tokens: the library writes the run last.
    let by = load("tiny_bytes");
    // 😀 is four byte tokens (all present), 日 has no byte tokens, so it is an <unk>.
    let unk = by.token_to_id("<unk>").unwrap();
    let emoji: Vec<u32> = [0xF0u8, 0x9F, 0x98, 0x80]
        .iter()
        .map(|b| by.token_to_id(&format!("<0x{b:02X}>")).unwrap())
        .collect();
    assert_eq!(by.encode_raw("日😀", false), [emoji.clone(), vec![unk]].concat());
    assert_eq!(by.encode_raw("日😀日", false), [emoji, vec![unk, unk]].concat());
}

const GEMMA_CONFIG: &str =
    r#"{"bos_token": "<bos>", "image_token": "<a><b>", "model_max_length": 1000000000000000019884624838656}"#;
const GEMMA_PROMPTS: &str =
    r#"{"prompts": {"query": "hello ", "empty": "", "code": "<bos>"}, "default_prompt_name": null}"#;

fn configured() -> Tokenizer {
    load("tiny_gemma")
        .with_tokenizer_config(GEMMA_CONFIG)
        .unwrap()
        .with_prompts(GEMMA_PROMPTS)
        .unwrap()
}

#[test]
fn sentence_transformers_wraps_the_text_in_bos_and_eos_unless_it_starts_with_bos() {
    let t = configured();
    let (bos, eos) = (2u32, 1u32);
    let body = |s: &str| t.encode_raw(s, false);
    let plain = t.encode("hello world").unwrap();
    assert_eq!(plain, [vec![bos], body("hello world"), vec![eos]].concat());
    // Starting with <bos>: the text's own <bos> only, no <eos>.
    assert_eq!(t.encode("<bos>hello world").unwrap(), body("<bos>hello world"));
    assert_eq!(t.encode("<bos>hello world").unwrap()[0], bos);
    assert_ne!(*t.encode("<bos>hello world").unwrap().last().unwrap(), eos);
    // <bos> elsewhere or <eos> at the end do not switch it off.
    assert_eq!(
        t.encode("a<bos>").unwrap(),
        [vec![bos], body("a<bos>"), vec![eos]].concat()
    );
    assert_eq!(t.encode("<eos>").unwrap(), [bos, eos, eos]);
    assert_eq!(t.encode("").unwrap(), [bos, eos]);
    // Only the exact bos token counts.
    assert_eq!(t.encode("<bo").unwrap()[0], bos);
    assert_eq!(*t.encode("<bo").unwrap().last().unwrap(), eos);
    // Without a tokenizer config there is no rule: the library alone.
    let bare = load("tiny_gemma");
    assert_eq!(bare.encode("<bos>x").unwrap(), bare.encode_raw("<bos>x", true));
}

#[test]
fn a_prompt_is_text_in_front_of_the_text() {
    let t = configured();
    assert_eq!(t.prompt("query"), Some("hello "));
    assert_eq!(t.prompt_names().collect::<Vec<_>>(), ["query", "empty", "code"]);
    assert_eq!(
        t.encode_with_prompt(Some("query"), "world").unwrap(),
        t.encode("hello world").unwrap()
    );
    assert_eq!(
        t.encode_with_prompt(Some("empty"), "world").unwrap(),
        t.encode("world").unwrap()
    );
    assert_eq!(t.encode_with_prompt(None, "world").unwrap(), t.encode("world").unwrap());
    // The rule looks at prompt + text: a prompt that starts with <bos> switches the wrap off.
    assert_eq!(
        t.encode_with_prompt(Some("code"), "x").unwrap(),
        t.encode_raw("<bos>x", false)
    );
    // And a text starting with <bos> after a prompt that does not has it in the middle, so it is wrapped.
    let ids = t.encode_with_prompt(Some("query"), "<bos>x").unwrap();
    assert_eq!(*ids.last().unwrap(), 1);
    let err = t.encode_with_prompt(Some("missing"), "x").unwrap_err();
    assert!(
        err.contains("no prompt named \"missing\"") && err.contains("query"),
        "{err}"
    );
}

/// `default_prompt_name` is what `SentenceTransformer.encode` applies when it
/// is given no name. The tokenizer reports it and refuses a file whose default
/// is not one of its prompts, as the Python binding did before this moved here.
#[test]
fn the_default_prompt_name_is_read_and_must_name_a_prompt() {
    let t = configured();
    assert_eq!(t.default_prompt_name(), None);
    let named = |default: &str| {
        format!(r#"{{"prompts": {{"query": "hello ", "empty": ""}}, "default_prompt_name": {default}}}"#)
    };
    let t = load("tiny_gemma").with_prompts(&named("\"query\"")).unwrap();
    assert_eq!(t.default_prompt_name(), Some("query"));
    // The encode methods take the name they are given: the default is the caller's to apply.
    assert_eq!(t.encode("world").unwrap(), t.encode_raw("world", true));
    let t = load("tiny_gemma").with_prompts(&named("null")).unwrap();
    assert_eq!(t.default_prompt_name(), None);
    // Absent is none, and a later file replaces an earlier default.
    let t = t.with_prompts(r#"{"prompts": {"q": "a "}}"#).unwrap();
    assert_eq!(t.default_prompt_name(), None);
    for (bad, pattern) in [
        ("\"nope\"", "default_prompt_name \"nope\" is not one of its prompts"),
        ("5", "default_prompt_name\" is not a string or null"),
        ("[]", "default_prompt_name\" is not a string or null"),
    ] {
        let e = load("tiny_gemma")
            .with_prompts(&named(bad))
            .expect_err("a bad default_prompt_name was accepted");
        assert!(e.contains(pattern), "{bad}: {e}");
    }
}

#[test]
fn the_placeholder_tokens_of_the_config_are_refused() {
    let t = configured();
    // The config names `<a><b>` as the image token.
    for text in ["<a><b>", "x <a><b> y", "<a><b><a><b>"] {
        let err = t.encode(text).unwrap_err();
        assert!(
            err.contains("\"<a><b>\"") && err.contains("placeholder"),
            "{text}: {err}"
        );
    }
    // It is the prompt + text that is searched, and `<a>` alone is not the placeholder.
    assert!(t.encode("<a>").is_ok());
    let with_prompt = configured().with_prompts(r#"{"prompts": {"img": "<a><b>"}}"#).unwrap();
    assert!(with_prompt.encode_with_prompt(Some("img"), "x").is_err());
    // Not refused by the library-level call, which has no such rule.
    assert!(!t.encode_raw("<a><b>", true).is_empty());
    // A config without the token refuses nothing.
    let none = load("tiny_gemma")
        .with_tokenizer_config(r#"{"bos_token": "<bos>"}"#)
        .unwrap();
    assert!(none.encode("<a><b>").is_ok());
}

#[test]
fn a_batch_is_each_text_alone_and_an_error_names_its_text() {
    let t = configured();
    let texts = ["<bos>hello", "hello", "", "<bos>"];
    let batch = t.encode_batch(&texts).unwrap();
    for (i, text) in texts.iter().enumerate() {
        assert_eq!(batch[i], t.encode(text).unwrap(), "{text:?}");
    }
    // The order does not matter either (transformers lets the first text decide for the batch).
    let rev = t.encode_batch(&["hello", "<bos>hello"]).unwrap();
    assert_eq!(rev[0], t.encode("hello").unwrap());
    assert_eq!(rev[1], t.encode("<bos>hello").unwrap());
    let with = t.encode_batch_with_prompt(Some("query"), &["a", "b"]).unwrap();
    assert_eq!(with[1], t.encode("hello b").unwrap());
    let err = t.encode_batch(&["ok", "<a><b>"]).unwrap_err();
    assert!(err.starts_with("text 1: "), "{err}");
    assert!(t.encode_batch(&[]).unwrap().is_empty());
}

#[test]
fn snapshot_loading_reads_the_three_files_and_names_a_missing_one() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("embedgemma2_tok_snapshot");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), json_text("tiny_gemma")).unwrap();
    let err = Tokenizer::from_snapshot(&dir).unwrap_err();
    assert!(err.contains("tokenizer_config.json"), "{err}");
    std::fs::write(dir.join("tokenizer_config.json"), GEMMA_CONFIG).unwrap();
    let err = Tokenizer::from_snapshot(&dir).unwrap_err();
    assert!(err.contains("config_sentence_transformers.json"), "{err}");
    std::fs::write(dir.join("config_sentence_transformers.json"), GEMMA_PROMPTS).unwrap();
    let t = Tokenizer::from_snapshot(&dir).unwrap();
    assert_eq!(t.prompt("query"), Some("hello "));
    assert_eq!(t.encode("<bos>q").unwrap()[0], 2);
    // A tokenizer.json that is a directory (or anything but a regular file) is refused, not read.
    std::fs::remove_file(dir.join("tokenizer.json")).unwrap();
    std::fs::create_dir(dir.join("tokenizer.json")).unwrap();
    assert!(Tokenizer::from_snapshot(&dir)
        .unwrap_err()
        .contains("not a regular file"));
    // The sidecars are strict JSON too.
    for bad in [
        r#"{"bos_token": 1}"#,
        r#"{"bos_token": "<bos>",}"#,
        r#"[]"#,
        r#"{"image_token": {"content": 2}}"#,
    ] {
        assert!(load("tiny_gemma").with_tokenizer_config(bad).is_err(), "{bad}");
    }
    for bad in [
        r#"{"prompts": []}"#,
        r#"{"prompts": {"a": 1}}"#,
        r#"{"prompts": {"a": "x", "a": "y"}}"#,
    ] {
        assert!(load("tiny_gemma").with_prompts(bad).is_err(), "{bad}");
    }
    assert_eq!(load("tiny_gemma").with_prompts("{}").unwrap().prompt_names().count(), 0);
    // An added-token object is read for its content.
    let t = load("tiny_gemma")
        .with_tokenizer_config(r#"{"bos_token": {"content": "<|turn>"}}"#)
        .unwrap();
    assert_eq!(t.encode("<|turn>x").unwrap(), t.encode_raw("<|turn>x", false));
}

/// Every feature the tokenizer does not implement is a named error at load, and
/// each edit below is made to the real fixture, so the refusal is what changed.
#[test]
fn what_is_not_implemented_is_refused_by_name_at_load() {
    let base = json_text("tiny_gemma");
    Tokenizer::from_json(&base).unwrap();
    let cases: &[(&str, &str, &str)] = &[
        (r#""dropout": null"#, r#""dropout": 1"#, "dropout"),
        (
            r#""continuing_subword_prefix": null"#,
            r###""continuing_subword_prefix": "##""###,
            "continuing_subword_prefix",
        ),
        (
            r#""end_of_word_suffix": null"#,
            r#""end_of_word_suffix": "</w>""#,
            "end_of_word_suffix",
        ),
        (r#""truncation": null"#, r#""truncation": {}"#, "truncation"),
        (r#""padding": null"#, r#""padding": {}"#, "padding"),
        (r#""type": "BPE""#, r#""type": "WordPiece""#, "only a BPE model"),
        (
            r#""behavior": "MergedWithPrevious""#,
            r#""behavior": "Isolated""#,
            "MergedWithPrevious",
        ),
        (r#""invert": false"#, r#""invert": true"#, "MergedWithPrevious"),
        (r#""type": "Split""#, r#""type": "ByteLevel""#, "pre_tokenizer"),
        (
            r#""type": "TemplateProcessing""#,
            r#""type": "BertProcessing""#,
            "TemplateProcessing",
        ),
        (r#""type": "ByteFallback""#, r#""type": "Strip""#, "decoder"),
        (r#""type": "Fuse""#, r#""type": "Metaspace""#, "decoder"),
        (r#""lstrip": false"#, r#""lstrip": true"#, "lstrip"),
        (r#""rstrip": false"#, r#""rstrip": true"#, "rstrip"),
        (r#""single_word": false"#, r#""single_word": true"#, "single_word"),
        (r#""normalized": false"#, r#""normalized": true"#, "normalized"),
        (
            r#""pattern": {"String": " "}"#,
            r#""pattern": {"Regex": " "}"#,
            "literal",
        ),
        (
            r#""pattern": {"String": " "}"#,
            r#""pattern": {"String": ""}"#,
            "empty pattern",
        ),
        (r#""unk_token": "<unk>""#, r#""unk_token": "<nope>""#, "unk_token"),
        (r#""fuse_unk": true"#, r#""fuse_unk": 1"#, "fuse_unk"),
        (r#""byte_fallback": true"#, r#""byte_fallback": "yes""#, "byte_fallback"),
        (r#"["l", "l"]"#, r#"["l", "zz"]"#, "not in the vocabulary"),
        (r#"["he", "ll"]"#, r#"["he", "he"]"#, "not in the vocabulary"),
        (r#"["l", "l"]"#, r#"["l"]"#, "not a pair"),
        (r#"["l", "l"]"#, r#"3"#, "neither a pair"),
        (r#""<pad>": 0"#, r#""<pad>": 1"#, "used twice"),
        (r#""<b>": 7"#, r#""<b>": 9999"#, "dense"),
        (r#""content": "<a>""#, r#""content": "<zz>""#, "vocabulary's entry"),
        (r#""ids": [2]"#, r#""ids": [999999]"#, "outside the vocabulary"),
        (
            r#""id": "<bos>", "type_id": 0"#,
            r#""id": "<nobos>", "type_id": 0"#,
            "special_tokens",
        ),
        (r#""version": "1.0""#, r#""version": -1"#, "unsupported JSON value"),
        (r#""special": true"#, r#""special": null"#, "special"),
        (
            r#""version": "1.0""#,
            r#""version": "1.0", "version": "2""#,
            "duplicate key",
        ),
    ];
    for &(old, new, want) in cases {
        assert!(base.contains(old), "the fixture no longer contains {old}");
        let err = Tokenizer::from_json(&base.replace(old, new))
            .err()
            .unwrap_or_else(|| panic!("{new} was accepted"));
        assert!(err.contains(want), "{new}: {err:?} should mention {want:?}");
    }
    for (text, want) in [
        ("", "unexpected end"),
        ("[]", "not an object"),
        ("{}", "no model"),
        ("null", "not an object"),
        ("{\"model\": 3}", "model"),
    ] {
        let err = Tokenizer::from_json(text)
            .err()
            .unwrap_or_else(|| panic!("{text:?} was accepted"));
        assert!(err.to_lowercase().contains(want), "{text:?}: {err}");
    }
    assert!(Tokenizer::from_json(&format!("{base}x")).is_err());
    assert!(Tokenizer::from_json(&base[..base.len() / 2]).is_err());
}

#[test]
fn a_missing_post_processor_or_normalizer_is_no_wrap_and_no_rewrite() {
    // The legacy fixture has neither a normalizer nor byte fallback; strip the template too.
    let base = json_text("tiny_legacy");
    let start = base.find(r#""post_processor""#).unwrap();
    let end = base.find(r#""decoder""#).unwrap();
    let stripped = format!(
        "{}\"post_processor\": null,\n \"decoder\": {}",
        &base[..start],
        &base[end + r#""decoder": "#.len()..]
    );
    let t = Tokenizer::from_json(&stripped).unwrap();
    let full = Tokenizer::from_json(&base).unwrap();
    assert_eq!(t.encode_raw("ab", true), full.encode_raw("ab", false));
    // A null decoder is refused: the library would join tokens with spaces.
    let null_decoder = base.replace(r#""decoder": {"#, r#""decoder": null, "unused": {"#);
    assert!(Tokenizer::from_json(&null_decoder)
        .unwrap_err()
        .contains("null decoder"));
}

#[test]
fn hostile_texts_are_encoded_not_crashed_on() {
    let t = configured();
    // Long single pieces (the BPE runs over the whole text between added tokens).
    for text in [
        "l".repeat(300_000),
        " ".repeat(200_000),
        "<".repeat(100_000),
        "<bos>".repeat(40_000),
        "日".repeat(60_000),
    ] {
        let ids = t.encode(&text).unwrap();
        assert!(ids.len() >= 2, "{}", brief(&text));
    }
    // 4 bytes per character, the highest code point, and an embedded NUL.
    let t2 = t.encode("\u{10FFFF}\0\u{10FFFF}").unwrap();
    assert_eq!(
        t2.len(),
        2 + 4 + 1 + 4,
        "bytes of the two characters and the NUL: {t2:?}"
    );
    // The merges run left to right at equal rank: `l` x 4 is (ll)(ll), x 5 is (ll)(ll)l.
    let ll = t.token_to_id("ll").unwrap();
    let l = t.token_to_id("l").unwrap();
    assert_eq!(t.encode_raw("llll", false), [ll, ll]);
    assert_eq!(t.encode_raw("lllll", false), [ll, ll, l]);
}

#[test]
fn a_text_past_the_size_limit_is_an_error_and_a_panic_at_the_raw_call() {
    use tessl::embedgemma2_tok::MAX_TEXT_BYTES;
    let t = configured();
    let big = "a".repeat(MAX_TEXT_BYTES);
    let err = t.encode(&big).unwrap_err();
    assert!(err.contains("byte limit"), "{err}");
    // Under the limit by less than the prompt's length: the prompt counts.
    let near = "a".repeat(MAX_TEXT_BYTES - 3);
    assert!(t
        .encode_with_prompt(Some("query"), &near)
        .unwrap_err()
        .contains("with its prompt"));
    assert!(std::panic::catch_unwind(|| load("tiny_gemma").encode_raw(&big, true)).is_err());
}

#[test]
fn the_tokenizer_is_shared_across_threads() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Tokenizer>();
    let t = configured();
    let want = t.encode("hello world <|turn> 日本").unwrap();
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                for _ in 0..200 {
                    assert_eq!(t.encode("hello world <|turn> 日本").unwrap(), want);
                }
            });
        }
    });
}
