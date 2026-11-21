//! Characterization of the two `config.json` readers, `BertConfig` and
//! `Qwen35Config`: the shared shape (non-object root, missing keys, wrong
//! types, out-of-range u32, unknown keys) must read the same in both.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;
use tessl::bert::BertConfig;

use tessl::embedgemma2::EmbedGemma2Config;
use tessl::embedgemma2_audio::AudioConfig;
use tessl::embedgemma2_pre::PreConfig;
use tessl::embedgemma2_tok::Tokenizer;
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
            e.contains(&format!(
                "{}: {} bytes, above the {CAP}-byte config cap",
                f.display(),
                CAP + 1
            )),
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
        assert!(e.ends_with("stream did not contain valid UTF-8"), "{name}: {e}");
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

// Card ft-16a434be, adversarial review (B-1): edge cases on the seam.

/// Run every reader on `f` on its own thread; a reader that has not returned
/// in 10 s is blocked or reading forever. Returns each reader's refusal.
fn refusals_without_blocking(f: &Path) -> Vec<(&'static str, String)> {
    READERS
        .iter()
        .map(|&(name, read)| {
            let (tx, rx) = mpsc::channel();
            let p = f.to_path_buf();
            std::thread::spawn(move || {
                let _ = tx.send(read(&p));
            });
            let r = rx
                .recv_timeout(Duration::from_secs(10))
                .unwrap_or_else(|_| panic!("{name}: from_config_file did not return on {}", f.display()));
            (name, r.expect_err(name))
        })
        .collect()
}

fn assert_every_refusal_names(f: &Path, needle: &str) {
    for (name, e) in refusals_without_blocking(f) {
        assert!(
            e.contains(&f.display().to_string()),
            "{name}: the refusal omits the path: {e}"
        );
        assert!(e.contains(needle), "{name}: expected {needle:?}: {e}");
    }
}

#[test]
fn edge_a_symlink_loop_is_refused_naming_the_path() {
    let s = Scratch::new("loop");
    let (a, b) = (s.path("a.json"), s.path("b.json"));
    std::os::unix::fs::symlink(&b, &a).unwrap();
    std::os::unix::fs::symlink(&a, &b).unwrap();
    assert_every_refusal_names(&a, "");
}

#[test]
fn edge_a_symlink_to_a_fifo_is_refused_without_blocking() {
    let s = Scratch::new("linkfifo");
    let fifo = s.path("fifo");
    assert!(std::process::Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .unwrap()
        .success());
    let link = s.path("config.json");
    std::os::unix::fs::symlink(&fifo, &link).unwrap();
    assert_every_refusal_names(&link, "not a regular file");
}

#[test]
fn edge_a_symlink_to_a_char_device_is_refused_not_read_forever() {
    let s = Scratch::new("chardev");
    for dev in ["/dev/zero", "/dev/urandom"] {
        let link = s.path("config.json");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(dev, &link).unwrap();
        assert_every_refusal_names(&link, "not a regular file");
        assert_every_refusal_names(Path::new(dev), "not a regular file");
    }
}

#[test]
fn edge_a_unix_socket_is_refused() {
    let s = Scratch::new("sock");
    let sock = s.path("s");
    let _listener = std::os::unix::net::UnixListener::bind(&sock).unwrap();
    assert_every_refusal_names(&sock, "not a regular file");
}

#[test]
fn edge_a_dangling_symlink_is_refused_naming_the_path() {
    let s = Scratch::new("dangling");
    let link = s.path("config.json");
    std::os::unix::fs::symlink(s.path("nowhere"), &link).unwrap();
    assert_every_refusal_names(&link, "");
}

#[test]
fn edge_an_empty_file_is_an_error_for_every_reader() {
    let s = Scratch::new("empty");
    let f = s.path("config.json");
    std::fs::write(&f, b"").unwrap();
    for (name, e) in refusals_without_blocking(&f) {
        assert!(
            e.starts_with(&format!("{}: config.json: JSON:", f.display())),
            "{name}: {e}"
        );
    }
}

#[test]
fn edge_exactly_the_cap_plus_one_behind_a_symlink_is_refused_naming_the_link() {
    let s = Scratch::new("linkover");
    let blob = s.path("blob");
    sparse(&blob, CAP + 1);
    let link = s.path("config.json");
    std::os::unix::fs::symlink(&blob, &link).unwrap();
    assert_every_refusal_names(&link, &format!("{} bytes, above the {CAP}-byte config cap", CAP + 1));
}

#[test]
fn edge_a_non_utf8_path_does_not_panic_and_reads_when_it_exists() {
    use std::os::unix::ffi::OsStrExt;
    let s = Scratch::new("badname");
    let name = std::ffi::OsStr::from_bytes(b"conf\xffig.json");
    let f = s.0.join(name);
    for (rname, e) in refusals_without_blocking(&f) {
        assert!(e.contains("conf"), "{rname}: {e}");
    }
    // APFS refuses non-UTF-8 names (EILSEQ); where the name can be created the
    // config must read.
    if std::fs::write(&f, MINI).is_ok() {
        BertConfig::from_config_file(&f).unwrap();
    }
}

#[test]
fn edge_a_utf8_bom_is_refused_as_a_config_error() {
    let s = Scratch::new("bom");
    let f = s.path("config.json");
    let mut bytes = vec![0xEF, 0xBB, 0xBF];
    bytes.extend_from_slice(MINI.as_bytes());
    std::fs::write(&f, bytes).unwrap();
    for (name, e) in refusals_without_blocking(&f) {
        assert!(
            e.starts_with(&format!("{}: config.json: JSON:", f.display())),
            "{name}: {e}"
        );
    }
}

#[test]
fn edge_a_config_without_read_permission_is_refused_naming_the_path() {
    use std::os::unix::fs::PermissionsExt;
    let root = std::process::Command::new("id").arg("-u").output().unwrap();
    if String::from_utf8_lossy(&root.stdout).trim() == "0" {
        return; // root reads a mode-000 file
    }
    let s = Scratch::new("noperm");
    let f = s.path("config.json");
    std::fs::write(&f, MINI).unwrap();
    std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o000)).unwrap();
    assert_every_refusal_names(&f, "ermission denied");
}

#[test]
fn edge_a_path_past_path_max_is_refused_naming_the_path() {
    let s = Scratch::new("long");
    let mut f = s.0.clone();
    for _ in 0..8 {
        f.push("a".repeat(200));
    }
    f.push("config.json");
    assert!(f.as_os_str().len() > 1024);
    assert_every_refusal_names(&f, "");
}

// Card ft-57e1d032: a small, well-formed config can name huge, zero or
// non-finite numbers; each is a typed refusal before any size is taken from it.

/// The tiny EmbedGemma2 text config of `tests/common/embedgemma2.rs`.
const EG2: &str = r#"{"model_type":"embedding_gemma2","text_config":{
"model_type":"embedding_gemma2_text","attention_bias":false,"hidden_activation":"gelu_pytorch_tanh",
"hidden_size":64,"hidden_size_per_layer_input":32,"intermediate_size":128,"vocab_size":48,"embedding_dim":96,
"num_attention_heads":2,"num_key_value_heads":1,"head_dim":256,"num_hidden_layers":3,"sliding_window":3,
"rms_norm_eps":1e-06,
"layer_types":["full_attention","sliding_attention","sliding_attention"],
"per_layer_config":{"00":{"head_dim":512},"02":{"num_key_value_heads":2}},
"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"default"},
"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}}}}"#;

/// `processor_config.json` as the snapshot ships it (the image and video
/// sections both spell `patch_size`, `pooling_kernel_size` and
/// `rescale_factor`: an edit names which section by occurrence).
const PRE: &str = r#"{
  "audio_ms_per_token": 40, "audio_seq_length": 280,
  "feature_extractor": {"dither": 0.0, "feature_extractor_type": "Gemma4AudioFeatureExtractor",
    "feature_size": 128, "fft_length": 512, "fft_overdrive": false, "frame_length": 320,
    "hop_length": 160, "input_scale_factor": 1.0, "max_frequency": 8000.0, "mel_floor": 0.001,
    "min_frequency": 0.0, "padding_side": "right", "padding_value": 0.0, "per_bin_mean": null,
    "per_bin_stddev": null, "preemphasis": 0.0, "preemphasis_htk_flavor": true,
    "return_attention_mask": true, "sampling_rate": 16000},
  "image_processor": {"do_convert_rgb": true, "do_normalize": false, "do_rescale": true,
    "do_resize": true, "image_mean": [0.0, 0.0, 0.0], "image_processor_type": "Gemma4ImageProcessor",
    "image_seq_length": 280, "image_std": [1.0, 1.0, 1.0], "max_soft_tokens": 280, "patch_size": 16,
    "pooling_kernel_size": 3, "resample": 3, "rescale_factor": 0.00392156862745098},
  "video_processor": {"add_timestamps": false, "do_convert_rgb": true, "do_normalize": false,
    "do_rescale": true, "do_resize": true, "do_sample_frames": true, "fps": 1,
    "image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0], "max_frames": 32,
    "max_soft_tokens": 140, "overflow_strategy": "uniform", "patch_size": 16,
    "pooling_kernel_size": 3, "resample": 3, "rescale_factor": 0.00392156862745098,
    "return_metadata": false, "video_processor_type": "EmbeddingGemma2VideoProcessor"}
}"#;

/// The audio tower's config (`tests/embedgemma2_audio_tiny.rs`'s TINY).
const AUDIO: &str = r#"{"text_config":{"hidden_size":40},"audio_config":{
"attention_chunk_size":4,"attention_context_left":5,"attention_context_right":0,
"attention_invalid_logits_value":-1000000000.0,"attention_logit_cap":4.0,"conv_kernel_size":5,
"gradient_clipping":2.5,"hidden_act":"silu","hidden_size":64,"model_type":"gemma4_audio",
"num_attention_heads":4,"num_hidden_layers":2,"output_proj_dims":48,"residual_weight":0.5,
"rms_norm_eps":1e-05,"subsampling_conv_channels":[32,8],"use_clipped_linears":true}}"#;

/// Replace the `nth` (0-based) occurrence of `from` in `base`.
fn edit_nth(base: &str, from: &str, to: &str, nth: usize) -> String {
    let at = base
        .match_indices(from)
        .nth(nth)
        .unwrap_or_else(|| panic!("{from:?} has no occurrence {nth}"))
        .0;
    format!("{}{to}{}", &base[..at], &base[at + from.len()..])
}

type Reader = fn(&str) -> Result<(), String>;

fn read_bert(t: &str) -> Result<(), String> {
    BertConfig::from_config_json(t).map(|_| ())
}
fn read_eg2(t: &str) -> Result<(), String> {
    EmbedGemma2Config::from_config_json(t).map(|_| ())
}
fn read_pre(t: &str) -> Result<(), String> {
    PreConfig::from_json(t).map(|_| ())
}
fn read_audio(t: &str) -> Result<(), String> {
    AudioConfig::from_config_json(t).map(|_| ())
}
fn read_qwen(t: &str) -> Result<(), String> {
    Qwen35Config::from_config_json(t).map(|_| ())
}
fn read_tok(t: &str) -> Result<(), String> {
    Tokenizer::from_json(t).map(|_| ())
}

/// The smallest `tokenizer.json` the reader takes: one token, id 0.
const TOK: &str = r#"{"model":{"type":"BPE","vocab":{"a":0},"merges":[],"fuse_unk":false,"byte_fallback":false},
"decoder":{"type":"Fuse"}}"#;

/// One hostile case: the reader, the file it reads with the `nth` occurrence
/// of `from` replaced by `to` (an empty `from` means `base` is the file as
/// it stands), and what the refusal must name.
struct Hostile {
    reader: &'static str,
    read: Reader,
    base: &'static str,
    from: &'static str,
    nth: usize,
    to: &'static str,
    needle: &'static str,
}

const fn case(
    reader: &'static str,
    read: Reader,
    base: &'static str,
    from: &'static str,
    nth: usize,
    to: &'static str,
    needle: &'static str,
) -> Hostile {
    Hostile {
        reader,
        read,
        base,
        from,
        nth,
        to,
        needle,
    }
}

/// A case whose file is already edited.
const fn whole(
    reader: (&'static str, Reader),
    file: &'static str,
    label: &'static str,
    needle: &'static str,
) -> Hostile {
    case(reader.0, reader.1, file, "", 0, label, needle)
}

fn hostile_cases() -> Vec<Hostile> {
    const B: (&str, Reader) = ("bert", read_bert);
    const E: (&str, Reader) = ("embedgemma2", read_eg2);
    const P: (&str, Reader) = ("pre", read_pre);
    const Q: (&str, Reader) = ("qwen35", read_qwen);
    const T: (&str, Reader) = ("tokenizer", read_tok);
    let mut v = vec![
        // BERT.
        case(
            B.0,
            B.1,
            MINI,
            "\"num_hidden_layers\":6",
            0,
            "\"num_hidden_layers\":4294967295",
            "num_hidden_layers",
        ),
        case(
            B.0,
            B.1,
            MINI,
            "\"num_hidden_layers\":6",
            0,
            "\"num_hidden_layers\":1025",
            "num_hidden_layers",
        ),
        case(
            B.0,
            B.1,
            MINI,
            "\"num_hidden_layers\":6",
            0,
            "\"num_hidden_layers\":0",
            "num_hidden_layers is zero",
        ),
        case(
            B.0,
            B.1,
            MINI,
            "\"hidden_size\":384",
            0,
            "\"hidden_size\":0",
            "hidden_size is zero",
        ),
        case(
            B.0,
            B.1,
            MINI,
            "\"num_attention_heads\":12",
            0,
            "\"num_attention_heads\":0",
            "num_attention_heads is zero",
        ),
        case(
            B.0,
            B.1,
            MINI,
            "\"num_attention_heads\":12",
            0,
            "\"num_attention_heads\":4294967295",
            "multiple",
        ),
        case(
            B.0,
            B.1,
            MINI,
            "\"hidden_size\":384",
            0,
            "\"hidden_size\":4294967295",
            "multiple",
        ),
        case(
            B.0,
            B.1,
            MINI,
            "\"intermediate_size\":1536",
            0,
            "\"intermediate_size\":0",
            "intermediate_size is zero",
        ),
        case(
            B.0,
            B.1,
            MINI,
            "\"type_vocab_size\":2",
            0,
            "\"type_vocab_size\":0",
            "type_vocab_size is zero",
        ),
        case(B.0, B.1, MINI, "1e-12", 0, "1e39", "layer_norm_eps"),
        case(B.0, B.1, MINI, "1e-12", 0, "1e999", "overflows f64"),
        case(B.0, B.1, MINI, "1e-12", 0, "1e-50", "layer_norm_eps must be positive"),
        case(B.0, B.1, MINI, "1e-12", 0, "0", "layer_norm_eps must be positive"),
        case(
            B.0,
            B.1,
            DISTIL,
            "\"n_layers\":6",
            0,
            "\"n_layers\":4294967295",
            "n_layers",
        ),
        case(
            B.0,
            B.1,
            DISTIL,
            "\"n_layers\":6",
            0,
            "\"n_layers\":0",
            "n_layers is zero",
        ),
        case(
            B.0,
            B.1,
            DISTIL,
            "\"n_heads\":12",
            0,
            "\"n_heads\":0",
            "n_heads is zero",
        ),
        case(B.0, B.1, DISTIL, "\"dim\":768", 0, "\"dim\":0", "dim is zero"),
        // EmbedGemma2 text config.
        case(
            E.0,
            E.1,
            EG2,
            "\"num_hidden_layers\":3",
            0,
            "\"num_hidden_layers\":4294967295",
            "num_hidden_layers",
        ),
        case(
            E.0,
            E.1,
            EG2,
            "\"num_hidden_layers\":3",
            0,
            "\"num_hidden_layers\":0",
            "num_hidden_layers",
        ),
        case(
            E.0,
            E.1,
            EG2,
            "\"hidden_size\":64",
            0,
            "\"hidden_size\":0",
            "hidden_size is zero",
        ),
        case(
            E.0,
            E.1,
            EG2,
            "\"hidden_size_per_layer_input\":32",
            0,
            "\"hidden_size_per_layer_input\":0",
            "hidden_size_per_layer_input is zero",
        ),
        case(
            E.0,
            E.1,
            EG2,
            "\"intermediate_size\":128",
            0,
            "\"intermediate_size\":0",
            "intermediate_size is zero",
        ),
        case(
            E.0,
            E.1,
            EG2,
            "\"vocab_size\":48",
            0,
            "\"vocab_size\":0",
            "vocab_size is zero",
        ),
        case(
            E.0,
            E.1,
            EG2,
            "\"embedding_dim\":96",
            0,
            "\"embedding_dim\":0",
            "embedding_dim is zero",
        ),
        case(
            E.0,
            E.1,
            EG2,
            "\"num_attention_heads\":2",
            0,
            "\"num_attention_heads\":0",
            "num_attention_heads is zero",
        ),
        case(
            E.0,
            E.1,
            EG2,
            "\"num_attention_heads\":2",
            0,
            "\"num_attention_heads\":4294967295",
            "overflows u32",
        ),
        case(
            E.0,
            E.1,
            EG2,
            "\"num_key_value_heads\":1",
            0,
            "\"num_key_value_heads\":0",
            "kv heads",
        ),
        case(
            E.0,
            E.1,
            EG2,
            "\"num_key_value_heads\":1",
            0,
            "\"num_key_value_heads\":4294967295",
            "kv heads",
        ),
        case(
            E.0,
            E.1,
            EG2,
            "\"sliding_window\":3",
            0,
            "\"sliding_window\":0",
            "sliding_window is zero",
        ),
        case(
            E.0,
            E.1,
            EG2,
            "\"head_dim\":256",
            0,
            "\"head_dim\":4294967295",
            "head_dim",
        ),
        case(E.0, E.1, EG2, "\"head_dim\":256", 0, "\"head_dim\":0", "head_dim"),
        case(E.0, E.1, EG2, "1e-06", 0, "1e39", "rms_norm_eps"),
        case(E.0, E.1, EG2, "1e-06", 0, "1e999", "overflows f64"),
        case(E.0, E.1, EG2, "1e-06", 0, "1e-50", "rms_norm_eps must be positive"),
        case(E.0, E.1, EG2, "1e-06", 0, "0", "rms_norm_eps must be positive"),
        case(E.0, E.1, EG2, "1e-06", 0, "-1e-06", "rms_norm_eps must be positive"),
        case(E.0, E.1, EG2, "1000000.0", 0, "1e39", "rope_theta"),
        case(E.0, E.1, EG2, "1000000.0", 0, "1e999", "overflows f64"),
        case(E.0, E.1, EG2, "1000000.0", 0, "0", "rope_theta must be positive"),
        case(E.0, E.1, EG2, "10000.0", 0, "-10000.0", "rope_theta must be positive"),
        // Qwen3.5 (the same fixture the file readers use).
        case(
            Q.0,
            Q.1,
            QWEN,
            "\"num_hidden_layers\": 24",
            0,
            "\"num_hidden_layers\": 4294967295",
            "num_hidden_layers is 4294967295",
        ),
        case(
            Q.0,
            Q.1,
            QWEN,
            "\"hidden_size\": 2048",
            0,
            "\"hidden_size\": 0",
            "must be non-zero",
        ),
        case(
            Q.0,
            Q.1,
            QWEN,
            "\"num_attention_heads\": 8",
            0,
            "\"num_attention_heads\": 0",
            "must be non-zero",
        ),
        case(
            Q.0,
            Q.1,
            QWEN,
            "\"linear_num_key_heads\": 16",
            0,
            "\"linear_num_key_heads\": 0",
            "non-zero multiple",
        ),
        case(
            Q.0,
            Q.1,
            QWEN,
            "\"rms_norm_eps\": 1e-06",
            0,
            "\"rms_norm_eps\": 1e39",
            "must be positive",
        ),
        case(
            Q.0,
            Q.1,
            QWEN,
            "\"rope_theta\": 10000000",
            0,
            "\"rope_theta\": 1e39",
            "must be positive",
        ),
        case(
            Q.0,
            Q.1,
            QWEN,
            "\"rope_theta\": 10000000",
            0,
            "\"rope_theta\": 0",
            "must be positive",
        ),
        case(
            Q.0,
            Q.1,
            QWEN,
            "\"linear_conv_kernel_dim\": 4",
            0,
            "\"linear_conv_kernel_dim\": 9",
            "conv_kernel 9",
        ),
        // Tokenizer: a vocabulary id sizes nothing, but it indexes the span table.
        case(T.0, T.1, TOK, "\"a\":0", 0, "\"a\":4294967295", "vocabulary id"),
        case(T.0, T.1, TOK, "\"a\":0", 0, "\"a\":4294967296", "vocabulary id"),
        case(T.0, T.1, TOK, "\"a\":0", 0, "\"a\":1", "outside 0..1"),
        // Processor config: image section (first occurrence).
        case(
            P.0,
            P.1,
            PRE,
            "\"patch_size\": 16",
            0,
            "\"patch_size\": 4294967295",
            "patch_size",
        ),
        case(
            P.0,
            P.1,
            PRE,
            "\"patch_size\": 16",
            0,
            "\"patch_size\": 257",
            "patch_size",
        ),
        case(
            P.0,
            P.1,
            PRE,
            "\"patch_size\": 16",
            0,
            "\"patch_size\": 0",
            "patch_size",
        ),
        case(
            P.0,
            P.1,
            PRE,
            "\"pooling_kernel_size\": 3",
            0,
            "\"pooling_kernel_size\": 4294967295",
            "pooling_kernel_size",
        ),
        case(
            P.0,
            P.1,
            PRE,
            "\"pooling_kernel_size\": 3",
            0,
            "\"pooling_kernel_size\": 17",
            "pooling_kernel_size",
        ),
        case(
            P.0,
            P.1,
            PRE,
            "\"pooling_kernel_size\": 3",
            0,
            "\"pooling_kernel_size\": 0",
            "pooling_kernel_size",
        ),
        case(
            P.0,
            P.1,
            PRE,
            "\"max_soft_tokens\": 280",
            0,
            "\"max_soft_tokens\": 4294967295",
            "max_soft_tokens",
        ),
        case(P.0, P.1, PRE, "0.00392156862745098", 0, "1e39", "rescale_factor"),
        case(P.0, P.1, PRE, "0.00392156862745098", 0, "1e-50", "rescale_factor"),
        case(P.0, P.1, PRE, "0.00392156862745098", 0, "0", "rescale_factor"),
        case(P.0, P.1, PRE, "0.00392156862745098", 0, "-1", "rescale_factor"),
        // Processor config: video section (second occurrence).
        case(
            P.0,
            P.1,
            PRE,
            "\"patch_size\": 16",
            1,
            "\"patch_size\": 4294967295",
            "patch_size",
        ),
        case(
            P.0,
            P.1,
            PRE,
            "\"pooling_kernel_size\": 3",
            1,
            "\"pooling_kernel_size\": 4294967295",
            "pooling_kernel_size",
        ),
        case(P.0, P.1, PRE, "0.00392156862745098", 1, "1e39", "rescale_factor"),
        case(
            P.0,
            P.1,
            PRE,
            "\"max_frames\": 32",
            0,
            "\"max_frames\": 18446744073709551615",
            "max_frames",
        ),
        case(
            P.0,
            P.1,
            PRE,
            "\"max_frames\": 32",
            0,
            "\"max_frames\": 1048577",
            "max_frames",
        ),
        // Processor config: audio feature extractor.
        case(
            P.0,
            P.1,
            PRE,
            "\"feature_size\": 128",
            0,
            "\"feature_size\": 4294967295",
            "feature_size",
        ),
        case(
            P.0,
            P.1,
            PRE,
            "\"feature_size\": 128",
            0,
            "\"feature_size\": 18446744073709551615",
            "feature_size",
        ),
        case(
            P.0,
            P.1,
            PRE,
            "\"feature_size\": 128",
            0,
            "\"feature_size\": 513",
            "feature_size",
        ),
        case(
            P.0,
            P.1,
            PRE,
            "\"feature_size\": 128",
            0,
            "\"feature_size\": 0",
            "feature_size",
        ),
        case(
            P.0,
            P.1,
            PRE,
            "\"sampling_rate\": 16000",
            0,
            "\"sampling_rate\": 0",
            "out of range",
        ),
    ];
    // Sizes whose product, not any one of them, overflows usize (each alone is a
    // valid u32; the config is refused before a tensor of that shape is sized).
    for (other, label, first) in [
        (
            "\"hidden_size_per_layer_input\":32",
            "hidden x per-layer input at u32::MAX",
            "\"hidden_size_per_layer_input\":4294967295",
        ),
        (
            "\"intermediate_size\":128",
            "hidden x intermediate at u32::MAX",
            "\"intermediate_size\":4294967295",
        ),
    ] {
        let text = edit_nth(
            &edit_nth(EG2, "\"hidden_size\":64", "\"hidden_size\":4294967295", 0),
            other,
            first,
            0,
        );
        let text: &'static str = Box::leak(text.into_boxed_str());
        v.push(whole(E, text, label, "overflows usize"));
    }
    // Each bound holds on its own and the product still must not: the largest
    // patch, pooling kernel and soft-token count together are a 220 GB image.
    let big_image = edit_nth(
        &edit_nth(
            &edit_nth(PRE, "\"patch_size\": 16", "\"patch_size\": 256", 0),
            "\"pooling_kernel_size\": 3",
            "\"pooling_kernel_size\": 16",
            0,
        ),
        "\"max_soft_tokens\": 280",
        "\"max_soft_tokens\": 1120",
        0,
    );
    let big_image: &'static str = Box::leak(big_image.into_boxed_str());
    v.push(whole(
        P,
        big_image,
        "patch 256, kernel 16, 1120 soft tokens",
        "pixel_values",
    ));
    // 512 mel bins at a one-sample hop over 30 s is 983 MB of features.
    let dense_audio = edit_nth(
        &edit_nth(PRE, "\"feature_size\": 128", "\"feature_size\": 512", 0),
        "\"hop_length\": 160",
        "\"hop_length\": 1, \"hop_length_ms\": 0.0625",
        0,
    );
    let dense_audio: &'static str = Box::leak(dense_audio.into_boxed_str());
    v.push(whole(P, dense_audio, "512 mel bins, hop of one sample", "features"));
    v
}

#[test]
fn every_reader_refuses_a_hostile_number_with_a_typed_error() {
    let mut failures = Vec::new();
    for c in hostile_cases() {
        let text = if c.from.is_empty() {
            c.base.to_string()
        } else {
            edit_nth(c.base, c.from, c.to, c.nth)
        };
        match std::panic::catch_unwind(|| (c.read)(&text)) {
            Err(_) => failures.push(format!("{}: {} -> panicked", c.reader, c.to)),
            Ok(Ok(())) => failures.push(format!("{}: {} -> accepted", c.reader, c.to)),
            Ok(Err(e)) if !e.contains(c.needle) => {
                failures.push(format!("{}: {} -> {e:?} lacks {:?}", c.reader, c.to, c.needle));
            }
            Ok(Err(_)) => {}
        }
    }
    assert!(
        failures.is_empty(),
        "{} of the hostile cases:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The caps are inclusive: the value at the bound reads, one past does not.
#[test]
fn the_bounds_are_inclusive() {
    BertConfig::from_config_json(&edit(MINI, "\"num_hidden_layers\":6", "\"num_hidden_layers\":1024")).unwrap();
    let at = edit_nth(
        &edit_nth(PRE, "\"patch_size\": 16", "\"patch_size\": 32", 0),
        "\"pooling_kernel_size\": 3",
        "\"pooling_kernel_size\": 8",
        0,
    );
    // 280 soft tokens x 64 patches x 3 x 32^2 x 4 B = 220 MB: under the 268 MB cap.
    PreConfig::from_json(&at).unwrap();
    // 560 soft tokens at the same geometry is 440 MB: over it.
    let over = edit_nth(&at, "\"max_soft_tokens\": 280", "\"max_soft_tokens\": 560", 0);
    assert!(PreConfig::from_json(&over).unwrap_err().contains("pixel_values"));
    PreConfig::from_json(&edit_nth(PRE, "\"feature_size\": 128", "\"feature_size\": 512", 0)).unwrap();
    PreConfig::from_json(&edit_nth(PRE, "\"max_frames\": 32", "\"max_frames\": 1048576", 0)).unwrap();
}

/// The audio reader is held (`src/embedgemma2_audio.rs`): only the hostile
/// numbers it already refuses are pinned here. It accepts, on unmodified code,
/// `num_hidden_layers`, `output_proj_dims`, `attention_chunk_size` and
/// `conv_kernel_size` at u32::MAX (reported as findings, not pinned).
#[test]
fn the_audio_reader_refuses_the_hostile_numbers_it_already_refuses() {
    let cases = [
        ("\"hidden_size\":64", "\"hidden_size\":0"),
        ("\"num_attention_heads\":4", "\"num_attention_heads\":0"),
        ("\"num_hidden_layers\":2", "\"num_hidden_layers\":0"),
        ("\"attention_chunk_size\":4", "\"attention_chunk_size\":0"),
        ("\"attention_context_left\":5", "\"attention_context_left\":0"),
        ("\"attention_context_left\":5", "\"attention_context_left\":4294967295"),
        ("\"conv_kernel_size\":5", "\"conv_kernel_size\":0"),
        ("\"output_proj_dims\":48", "\"output_proj_dims\":0"),
        ("\"hidden_size\":64", "\"hidden_size\":4294967295"),
        ("\"num_attention_heads\":4", "\"num_attention_heads\":4294967295"),
        (
            "\"subsampling_conv_channels\":[32,8]",
            "\"subsampling_conv_channels\":[4294967292,8]",
        ),
        (
            "\"subsampling_conv_channels\":[32,8]",
            "\"subsampling_conv_channels\":[32,4294967295]",
        ),
        ("\"rms_norm_eps\":1e-05", "\"rms_norm_eps\":-1"),
        ("\"rms_norm_eps\":1e-05", "\"rms_norm_eps\":0"),
        ("\"rms_norm_eps\":1e-05", "\"rms_norm_eps\":1e39"),
        ("\"residual_weight\":0.5", "\"residual_weight\":1e39"),
        ("\"attention_logit_cap\":4.0", "\"attention_logit_cap\":1e39"),
        (
            "\"attention_invalid_logits_value\":-1000000000.0",
            "\"attention_invalid_logits_value\":-1e39",
        ),
    ];
    for (from, to) in cases {
        let text = edit_nth(AUDIO, from, to, 0);
        let r = std::panic::catch_unwind(|| read_audio(&text));
        assert!(matches!(r, Ok(Err(_))), "{to}: {r:?}");
    }
}

/// Every hostile case edits one number of a file that is itself accepted, so a
/// refusal is the number's, not the file's.
#[test]
fn every_hostile_base_is_itself_accepted() {
    for (name, read, base) in [
        ("bert", read_bert as Reader, MINI),
        ("distilbert", read_bert, DISTIL),
        ("embedgemma2", read_eg2, EG2),
        ("pre", read_pre, PRE),
        ("audio", read_audio, AUDIO),
        ("qwen35", read_qwen, QWEN),
        ("tokenizer", read_tok, TOK),
    ] {
        read(base).unwrap_or_else(|e| panic!("{name}: the base is refused: {e}"));
    }
}

/// A bare `tokenizer_config.json` keeps a non-ASCII `bos_token` intact
/// through the rewrite that makes a 20-digit `model_max_length` parse: the
/// rewrite once pushed each byte as a Latin-1 char, so `é` became `Ã©` and
/// the text that starts with it no longer matched.
#[test]
fn a_non_ascii_bos_token_survives_a_twenty_digit_model_max_length() {
    let tok = Tokenizer::from_file(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embedgemma2_tok/tiny_gemma.json"),
    )
    .unwrap();
    // 1e30 as an integer, 31 digits, as the real file spells it.
    let cfg = r#"{"bos_token": "é", "model_max_length": 1000000000000000019884624838656,
        "tag": "id12345678901234567890"}"#;
    let configured = tok.with_tokenizer_config(cfg).unwrap();
    // A text that starts with the bos token gets no second one added.
    let with_bos = configured.encode("éa").unwrap();
    let without = configured.encode_raw("éa", false);
    assert_eq!(with_bos, without, "the bos_token was not recognised");
}

/// The fields are `pub`, so a config built or edited by hand never went
/// through parsing: `validate()` (which `load` calls first) is the one owner of
/// every bound, and refuses these without a panic.
#[test]
fn a_hand_built_config_is_refused_by_validate_not_a_panic() {
    let refused = |r: Result<(), String>, needle: &str, what: &str| {
        let e = r.expect_err(what);
        assert!(e.contains(needle), "{what}: {e:?} lacks {needle:?}");
    };

    let good = BertConfig::from_config_json(MINI).unwrap();
    good.validate().unwrap();
    for (what, needle, edit) in [
        (
            "zero heads",
            "num_attention_heads is zero",
            (|c: &mut BertConfig| c.heads = 0) as fn(&mut BertConfig),
        ),
        ("zero hidden", "hidden_size is zero", |c| c.hidden = 0),
        ("layers at u32::MAX", "num_hidden_layers", |c| c.layers = u32::MAX),
        ("layers one past the cap", "num_hidden_layers", |c| c.layers = 1025),
        ("infinite eps", "layer_norm_eps", |c| c.layer_norm_eps = f32::INFINITY),
        ("NaN eps", "layer_norm_eps", |c| c.layer_norm_eps = f32::NAN),
        ("head dim without a kernel", "head dim", |c| c.hidden = 12 * 48),
    ] {
        let mut c = good.clone();
        edit(&mut c);
        // head_dim is a pub fn on the same value: it must not panic on zero heads.
        let _ = c.head_dim();
        refused(c.validate(), needle, what);
    }

    let good = EmbedGemma2Config::from_config_json(EG2).unwrap();
    good.validate().unwrap();
    for (what, needle, edit) in [
        (
            "zero q heads",
            "num_attention_heads is zero",
            (|c: &mut EmbedGemma2Config| c.q_heads = 0) as fn(&mut EmbedGemma2Config),
        ),
        ("zero hidden", "hidden_size is zero", |c| c.hidden = 0),
        ("no layers", "num_hidden_layers is zero", |c| c.layers.clear()),
        ("too many layers", "above the cap", |c| {
            let one = c.layers[0];
            c.layers = vec![one; 1025];
        }),
        ("infinite eps", "rms_norm_eps", |c| c.rms_norm_eps = f32::INFINITY),
        ("zero kv heads", "kv heads", |c| c.layers[1].kv_heads = 0),
        ("infinite theta", "rope_theta", |c| {
            c.layers[1].rope_theta = f32::INFINITY
        }),
        ("a head dim with no kernel", "head_dim", |c| c.layers[1].head_dim = 100),
        ("a q|k|v width past u32", "overflows u32", |c| c.q_heads = u32::MAX / 2),
    ] {
        let mut c = good.clone();
        edit(&mut c);
        refused(c.validate(), needle, what);
    }
}
