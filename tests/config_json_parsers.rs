//! Characterization of the two `config.json` readers, `BertConfig` and
//! `Qwen35Config`: the shared shape (non-object root, missing keys, wrong
//! types, out-of-range u32, unknown keys) must read the same in both.

use tessl::bert::BertConfig;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use tessl::embedgemma2::EmbedGemma2Config;
use tessl::embedgemma2_audio::AudioConfig;
use tessl::embedgemma2_vision::VisionConfig;
use tessl::qwen35_model::Qwen35Config;

const MINI: &str = r#"{"hidden_act":"gelu","hidden_size":384,
"intermediate_size":1536,"layer_norm_eps":1e-12,"max_position_embeddings":512,"model_type":"bert",
"num_attention_heads":12,"num_hidden_layers":6,"position_embedding_type":"absolute","type_vocab_size":2,
"vocab_size":30522}"#;

const DISTIL: &str = r#"{"activation":"gelu","dim":768,
"hidden_dim":3072,"max_position_embeddings":512,"model_type":"distilbert","n_heads":12,"n_layers":6,
"sinusoidal_pos_embds":false,"vocab_size":30522}"#;

const QWEN: &str = include_str!("fixtures/qwen35/config_qwen35_2b_base.json");

fn bert_err(text: &str) -> String {
    BertConfig::from_config_json(text).map(|_| ()).unwrap_err()
}
fn qwen_err(text: &str) -> String {
    Qwen35Config::from_config_json(text).map(|_| ()).unwrap_err()
}

/// Replace the one occurrence of `from` in `base`.
fn edit(base: &str, from: &str, to: &str) -> String {
    assert_eq!(base.matches(from).count(), 1, "{from:?} must match once");
    base.replace(from, to)
}

#[test]
fn both_refuse_a_non_object_root_with_the_same_text() {
    for root in ["[]", "3", "\"x\"", "null", "true"] {
        assert_eq!(bert_err(root), "config.json: the root is not an object", "{root}");
        assert_eq!(qwen_err(root), "config.json: the root is not an object", "{root}");
    }
    assert!(bert_err("{").contains("config.json"));
    assert!(qwen_err("{").contains("config.json"));
}

#[test]
fn both_name_a_missing_required_key() {
    assert_eq!(
        bert_err(&edit(MINI, "\"vocab_size\":30522", "\"x\":1")),
        "config.json: missing \"vocab_size\""
    );
    assert_eq!(
        bert_err(&edit(MINI, "\"layer_norm_eps\":1e-12,", "")),
        "config.json: missing \"layer_norm_eps\""
    );
    assert_eq!(
        bert_err(&edit(MINI, "\"type_vocab_size\":2,", "")),
        "config.json: missing \"type_vocab_size\""
    );
    assert_eq!(
        bert_err(&edit(DISTIL, "\"dim\":768,", "")),
        "config.json: missing \"dim\""
    );
    let q = edit(QWEN, "\"vocab_size\"", "\"vocab_sizeX\"");
    assert_eq!(qwen_err(&q), "config.json: missing \"vocab_size\"");
    let q = edit(QWEN, "\"rms_norm_eps\"", "\"rms_norm_epsX\"");
    assert_eq!(qwen_err(&q), "config.json: missing \"rms_norm_eps\"");
}

#[test]
fn both_refuse_a_non_integer_or_negative_count() {
    for bad in ["-1", "1.5", "1e3", "\"6\"", "true", "null", "[]"] {
        let b = bert_err(&edit(
            MINI,
            "\"num_hidden_layers\":6",
            &format!("\"num_hidden_layers\":{bad}"),
        ));
        assert!(
            b.starts_with("config.json: num_hidden_layers must be a non-negative integer, got "),
            "{bad}: {b}"
        );
    }
    for bad in ["1.5", "null"] {
        let q = qwen_err(&edit(QWEN, "\"vocab_size\": 248320", &format!("\"vocab_size\": {bad}")));
        assert!(
            q.starts_with("config.json: vocab_size must be a non-negative integer, got "),
            "{bad}: {q}"
        );
    }
}

#[test]
fn both_refuse_a_count_above_u32() {
    assert_eq!(
        bert_err(&edit(MINI, "\"vocab_size\":30522", "\"vocab_size\":4294967296")),
        "config.json: vocab_size = 4294967296 exceeds u32"
    );
    assert_eq!(
        qwen_err(&edit(QWEN, "\"vocab_size\": 248320", "\"vocab_size\": 4294967296")),
        "config.json: vocab_size = 4294967296 exceeds u32"
    );
    // u32::MAX itself is in range.
    BertConfig::from_config_json(&edit(MINI, "\"vocab_size\":30522", "\"vocab_size\":4294967295")).unwrap();
}

#[test]
fn both_refuse_a_non_number_epsilon() {
    let b = bert_err(&edit(MINI, "1e-12", "\"1e-12\""));
    assert!(
        b.starts_with("config.json: layer_norm_eps must be a number, got "),
        "{b}"
    );
    let b = bert_err(&edit(MINI, "1e-12", "-1e-12"));
    assert!(b.starts_with("config.json: layer_norm_eps must be positive"), "{b}");
    let q = qwen_err(&edit(QWEN, "\"rms_norm_eps\": 1e-06", "\"rms_norm_eps\": \"x\""));
    assert!(q.starts_with("config.json: rms_norm_eps must be a number, got "), "{q}");
}

#[test]
fn both_refuse_a_non_string_where_a_string_is_read() {
    let b = bert_err(&edit(MINI, "\"model_type\":\"bert\"", "\"model_type\":3"));
    assert!(b.starts_with("config.json: model_type must be a string, got "), "{b}");
    let b = bert_err(&edit(MINI, "\"hidden_act\":\"gelu\"", "\"hidden_act\":false"));
    assert!(b.starts_with("config.json: hidden_act must be a string, got "), "{b}");
    let q = qwen_err(&edit(QWEN, "\"hidden_act\": \"silu\"", "\"hidden_act\": 1"));
    assert!(q.starts_with("config.json: hidden_act must be a string, got "), "{q}");
}

#[test]
fn unknown_keys_are_ignored_by_both() {
    let b = edit(MINI, "{\"hidden_act\"", "{\"made_up\":[1,{\"a\":null}],\"hidden_act\"");
    assert_eq!(
        BertConfig::from_config_json(&b).unwrap(),
        BertConfig::from_config_json(MINI).unwrap()
    );
    let q = QWEN.replacen("{", "{\"made_up\": {\"a\": [1]},", 1);
    assert_eq!(
        Qwen35Config::from_config_json(&q).unwrap(),
        Qwen35Config::from_config_json(QWEN).unwrap()
    );
}

#[test]
fn bert_reads_both_families() {
    assert_eq!(BertConfig::from_config_json(MINI).unwrap().hidden, 384);
    assert_eq!(BertConfig::from_config_json(DISTIL).unwrap().hidden, 768);
    assert_eq!(
        bert_err(&edit(MINI, "\"model_type\":\"bert\"", "\"model_type\":\"roberta\"")),
        "config.json: model_type must be bert or distilbert, got Some(\"roberta\")"
    );
}

/// DRIFT: a flag BERT reads (`sinusoidal_pos_embds`) was matched only as
/// `true`, so `"true"`, `1`, `null` were taken as false and learned
/// positions ran; Qwen's flag reader refuses any non-boolean.
#[test]
fn bert_flag_must_be_a_boolean() {
    for bad in ["\"true\"", "1", "null", "[]"] {
        let r = BertConfig::from_config_json(&edit(
            DISTIL,
            "\"sinusoidal_pos_embds\":false",
            &format!("\"sinusoidal_pos_embds\":{bad}"),
        ));
        assert!(
            r.as_ref()
                .is_err_and(|e| e.starts_with("config.json: sinusoidal_pos_embds must be true or false, got ")),
            "{bad}: {r:?}"
        );
    }
    assert!(bert_err(&edit(
        DISTIL,
        "\"sinusoidal_pos_embds\":false",
        "\"sinusoidal_pos_embds\":true"
    ))
    .contains("sinusoidal_pos_embds is not supported"));
}

/// DRIFT: Qwen refuses a `null` string (wrong type); BERT read `null` as
/// absent, so `"position_embedding_type": null` was taken as absolute.
#[test]
fn bert_null_string_is_a_wrong_type_not_absent() {
    let r = BertConfig::from_config_json(&edit(
        MINI,
        "\"position_embedding_type\":\"absolute\"",
        "\"position_embedding_type\":null",
    ));
    assert!(
        r.as_ref()
            .is_err_and(|e| e.starts_with("config.json: position_embedding_type must be a string, got ")),
        "{r:?}"
    );
    let q = qwen_err(&edit(QWEN, "\"hidden_act\": \"silu\"", "\"hidden_act\": null"));
    assert!(q.starts_with("config.json: hidden_act must be a string, got "), "{q}");
}

// Card ft-16a434be: every `from_config_file` reads through one bounded reader.

const CAP: u64 = 1 << 20;

type FileReader = fn(&Path) -> Result<(), String>;

/// The five `from_config_file` readers. A refusal test over all five is what
/// shows each one goes through the bounded reader.
const READERS: [(&str, FileReader); 5] = [
    ("bert", |p| BertConfig::from_config_file(p).map(|_| ())),
    ("qwen35", |p| Qwen35Config::from_config_file(p).map(|_| ())),
    ("embedgemma2", |p| EmbedGemma2Config::from_config_file(p).map(|_| ())),
    ("vision", |p| VisionConfig::from_config_file(p).map(|_| ())),
    ("audio", |p| AudioConfig::from_config_file(p).map(|_| ())),
];

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("tessl-config-cap-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn sparse(path: &Path, len: u64) {
    std::fs::File::create(path).unwrap().set_len(len).unwrap();
}

#[test]
fn every_file_reader_refuses_a_config_past_the_cap_by_its_length() {
    let s = Scratch::new("over");
    let f = s.path("config.json");
    sparse(&f, CAP + 1);
    for (name, read) in READERS {
        let e = read(&f).unwrap_err();
        assert!(
            e.contains(&format!("{}: {} bytes, above the {CAP}-byte config cap", f.display(), CAP + 1)),
            "{name}: {e}"
        );
    }
}

#[test]
fn every_file_reader_refuses_a_directory() {
    let s = Scratch::new("dir");
    for (name, read) in READERS {
        let e = read(&s.0).unwrap_err();
        assert!(e.contains("not a regular file"), "{name}: {e}");
    }
}

#[test]
fn every_file_reader_refuses_invalid_utf8() {
    let s = Scratch::new("utf8");
    let f = s.path("config.json");
    std::fs::write(&f, [b'{', 0xff, 0xfe, b'}']).unwrap();
    for (name, read) in READERS {
        let e = read(&f).unwrap_err();
        assert!(e.starts_with(&format!("{}: ", f.display())), "{name}: {e}");
        assert!(!e.contains("config.json:"), "{name} reached the parser: {e}");
    }
}

#[test]
fn every_file_reader_refuses_a_fifo_without_blocking() {
    let s = Scratch::new("fifo");
    let f = s.path("config.json");
    let made = std::process::Command::new("mkfifo").arg(&f).status().unwrap();
    assert!(made.success(), "mkfifo failed");
    for (name, read) in READERS {
        let (tx, rx) = mpsc::channel();
        let p = f.clone();
        std::thread::spawn(move || {
            let _ = tx.send(read(&p));
        });
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(r) => {
                let e = r.unwrap_err();
                assert!(e.contains("not a regular file"), "{name}: {e}");
            }
            Err(_) => {
                // The reader is blocked in open(2); a writer releases it so
                // the thread ends, then the test fails.
                let _ = std::fs::OpenOptions::new().write(true).open(&f);
                panic!("{name}: from_config_file blocked on a FIFO");
            }
        }
    }
}

#[test]
fn a_config_symlink_is_followed() {
    let s = Scratch::new("link");
    let blob = s.path("blob");
    std::fs::write(&blob, MINI).unwrap();
    let link = s.path("config.json");
    std::os::unix::fs::symlink(&blob, &link).unwrap();
    BertConfig::from_config_file(&link).unwrap();
}

#[test]
fn a_config_of_exactly_the_cap_is_read() {
    let s = Scratch::new("exact");
    let f = s.path("config.json");
    let mut text = MINI.to_string();
    text.push_str(&" ".repeat(CAP as usize - text.len()));
    assert_eq!(text.len() as u64, CAP);
    std::fs::write(&f, &text).unwrap();
    BertConfig::from_config_file(&f).unwrap();
    text.push(' ');
    std::fs::write(&f, &text).unwrap();
    let e = BertConfig::from_config_file(&f).unwrap_err();
    assert!(e.contains("above the 1048576-byte config cap"), "{e}");
}
