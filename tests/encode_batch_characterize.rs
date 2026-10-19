//! Characterization, recorded on the code before the `config.json` and
//! batch-check convergence: what the three EmbedGemma2 config readers
//! (`EmbedGemma2Config`, `VisionConfig`, `AudioConfig`) answer for edited
//! configs, the cached real configs, and the exact texts of the two
//! `upload_segments` (`bert`, `embedgemma2`).

mod common;

use common::embedgemma2::CONFIG;
use common::embedgemma2_mm::{config_json, AUDIO_JSON, VISION_JSON};
use common::with_gpu;
use std::path::PathBuf;
use tessl::embedgemma2::EmbedGemma2Config;
use tessl::embedgemma2_audio::AudioConfig;
use tessl::embedgemma2_vision::VisionConfig;

/// Each reader's answer to `text`: `Ok(Debug of the config)` or `Err(text)`.
fn answers(text: &str) -> [Result<String, String>; 3] {
    [
        EmbedGemma2Config::from_config_json(text).map(|c| format!("{c:?}")),
        VisionConfig::from_config_json(text).map(|c| format!("{c:?}")),
        AudioConfig::from_config_json(text).map(|c| format!("{c:?}")),
    ]
}

fn edit(base: &str, from: &str, to: &str) -> String {
    assert_eq!(base.matches(from).count(), 1, "{from:?} must match once");
    base.replace(from, to)
}

/// `(label, text)` pairs, each one edit of the tiny multimodal config.
fn cases() -> Vec<(&'static str, String)> {
    let base = config_json();
    let mut v: Vec<(&'static str, String)> = Vec::new();
    for root in ["[]", "3", "null", "\"x\"", "{", ""] {
        v.push(("root", root.to_string()));
    }
    v.push(("text_config missing", edit(&base, "\"text_config\":{", "\"text_configX\":{")));
    v.push(("vocab missing", edit(&base, "\"vocab_size\":48,", "")));
    v.push(("vocab float", edit(&base, "\"vocab_size\":48,", "\"vocab_size\":48.5,")));
    v.push(("vocab negative", edit(&base, "\"vocab_size\":48,", "\"vocab_size\":-1,")));
    v.push(("vocab null", edit(&base, "\"vocab_size\":48,", "\"vocab_size\":null,")));
    v.push(("vocab big", edit(&base, "\"vocab_size\":48,", "\"vocab_size\":4294967296,")));
    v.push(("vocab zero", edit(&base, "\"vocab_size\":48,", "\"vocab_size\":0,")));
    v.push(("eps string", edit(&base, "\"rms_norm_eps\":1e-06,\n\"layer", "\"rms_norm_eps\":\"x\",\n\"layer")));
    v.push(("eps negative", edit(&base, "\"rms_norm_eps\":1e-06,\n\"layer", "\"rms_norm_eps\":-1.0,\n\"layer")));
    v.push(("eps missing", edit(&base, "\"rms_norm_eps\":1e-06,\n\"layer", "\n\"layer")));
    v.push((
        "text model_type wrong",
        edit(&base, "\"embedding_gemma2_text\"", "\"gemma\""),
    ));
    v.push((
        "text model_type number",
        edit(&base, "\"model_type\":\"embedding_gemma2_text\"", "\"model_type\":3"),
    ));
    v.push((
        "activation wrong",
        edit(&base, "\"hidden_activation\":\"gelu_pytorch_tanh\",\n", "\"hidden_activation\":\"relu\",\n"),
    ));
    v.push(("attention_bias true", edit(&base, "\"attention_bias\":false,\"hidden_activation\"", "\"attention_bias\":true,\"hidden_activation\"")));
    v.push((
        "layer_types short",
        edit(&base, "[\"full_attention\",\"sliding_attention\",\"sliding_attention\"]", "[\"full_attention\"]"),
    ));
    v.push((
        "layer_types unknown",
        edit(&base, "[\"full_attention\",\"sliding_attention\",\"sliding_attention\"]", "[\"full_attention\",\"x\",\"sliding_attention\"]"),
    ));
    v.push((
        "per_layer unknown key",
        edit(&base, "\"00\":{\"head_dim\":512}", "\"00\":{\"hidden\":512}"),
    ));
    v.push(("per_layer index", edit(&base, "\"00\":{\"head_dim\":512}", "\"9\":{\"head_dim\":512}")));
    v.push(("rope missing", edit(&base, "\"rope_parameters\":{\"full", "\"rope_parametersX\":{\"full")));
    v.push(("rope_type", edit(&base, "\"rope_type\":\"default\"},\n\"slid", "\"rope_type\":\"yarn\"},\n\"slid")));
    v.push(("pad_token_id string", edit(&base, "\"pad_token_id\":0", "\"pad_token_id\":\"0\"")));
    v.push(("pad_token_id null", edit(&base, "\"pad_token_id\":0", "\"pad_token_id\":null")));
    v.push(("image_token_id negative", edit(&base, "\"image_token_id\":60", "\"image_token_id\":-1")));
    // Vision tower.
    v.push(("vision missing", edit(&base, "\"vision_config\":", "\"vision_configX\":")));
    v.push(("vision patch string", edit(&base, "\"patch_size\":4", "\"patch_size\":\"4\"")));
    v.push(("vision patch missing", edit(&base, "\"patch_size\":4,", "")));
    v.push(("vision patch zero", edit(&base, "\"patch_size\":4", "\"patch_size\":0")));
    v.push(("vision eps string", edit(&base, "\"rms_norm_eps\":1e-06,\n\"attention_bias\"", "\"rms_norm_eps\":\"e\",\n\"attention_bias\"")));
    v.push(("vision model_type", edit(&base, "\"gemma4_vision\"", "\"gemma\"")));
    v.push(("vision activation", edit(&base, "\"hidden_activation\":\"gelu_pytorch_tanh\",\"hidden_size\":96", "\"hidden_activation\":\"relu\",\"hidden_size\":96")));
    v.push(("vision standardize number", edit(&base, "\"standardize\":false", "\"standardize\":1")));
    v.push(("vision standardize true", edit(&base, "\"standardize\":false", "\"standardize\":true")));
    v.push(("vision attention_bias true", edit(&base, "\"attention_bias\":false,\"attention_dropout\"", "\"attention_bias\":true,\"attention_dropout\"")));
    // Audio tower.
    v.push(("audio missing", edit(&base, "\"audio_config\":", "\"audio_configX\":")));
    v.push(("audio hidden_act", edit(&base, "\"hidden_act\":\"silu\"", "\"hidden_act\":\"gelu\"")));
    v.push(("audio model_type", edit(&base, "\"gemma4_audio\"", "\"x\"")));
    v.push(("audio clipped number", edit(&base, "\"use_clipped_linears\":true", "\"use_clipped_linears\":1")));
    v.push(("audio chunk missing", edit(&base, "\"attention_chunk_size\":4,", "")));
    v.push(("audio chunk negative", edit(&base, "\"attention_chunk_size\":4", "\"attention_chunk_size\":-4")));
    v.push(("audio logit_cap string", edit(&base, "\"attention_logit_cap\":4.0", "\"attention_logit_cap\":\"4\"")));
    v.push(("audio channels short", edit(&base, "[32,8]", "[32]")));
    v.push(("audio eps zero", edit(&base, "\"rms_norm_eps\":1e-05", "\"rms_norm_eps\":0")));
    v
}

/// One line per (edit, reader): the exact error text, or the length of the
/// accepted config's `Debug`; then the accepted tiny config's `Debug` per
/// reader. Recorded from the code before the convergence into
/// `tests/fixtures/encode_batch_characterize.golden`.
fn transcript() -> String {
    let mut out = String::new();
    for (label, text) in cases() {
        for (i, a) in answers(&text).iter().enumerate() {
            match a {
                Ok(c) => out += &format!("CASE\t{label}\t{i}\tOK\t{}\n", c.len()),
                Err(e) => out += &format!("CASE\t{label}\t{i}\tERR\t{e}\n"),
            }
        }
    }
    for (i, a) in answers(&config_json()).iter().enumerate() {
        out += &format!("BASE\t{i}\t{}\n", a.as_ref().unwrap());
    }
    let _ = (CONFIG, VISION_JSON, AUDIO_JSON);
    out
}

#[test]
fn the_three_readers_answer_as_recorded() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/encode_batch_characterize.golden");
    let now = transcript();
    if std::env::var_os("TESSL_RECORD_GOLDEN").is_some() {
        std::fs::write(&path, &now).unwrap();
        return;
    }
    let golden = std::fs::read_to_string(&path).unwrap();
    for (n, (g, w)) in golden.lines().zip(now.lines()).enumerate() {
        assert_eq!(g, w, "line {}", n + 1);
    }
    assert_eq!(golden.lines().count(), now.lines().count());
}

type UploadFn = fn(&std::sync::Arc<tessl::GpuRuntime>, &[(u32, u32)], u32) -> Result<tessl::tensor::GpuBuffer, String>;

fn cached_configs()-> Vec<PathBuf> {
    let Some(home) = std::env::var_os("HOME") else {
        return Vec::new();
    };
    let hub = PathBuf::from(home).join(".cache/huggingface/hub");
    let mut out = Vec::new();
    let Ok(models) = std::fs::read_dir(&hub) else {
        return out;
    };
    for m in models.flatten() {
        if !m.file_name().to_string_lossy().starts_with("models--google--embeddinggemma") {
            continue;
        }
        let Ok(snaps) = std::fs::read_dir(m.path().join("snapshots")) else {
            continue;
        };
        for s in snaps.flatten() {
            let c = s.path().join("config.json");
            if c.is_file() {
                out.push(c);
            }
        }
    }
    out.sort();
    out
}

#[test]
fn text_reader_answers_the_cached_real_configs() {
    let paths = cached_configs();
    if paths.is_empty() {
        eprintln!("no cached google/embeddinggemma* config.json: real-config check not run");
        return;
    }
    for p in paths {
        let text = std::fs::read_to_string(&p).unwrap();
        let [t, v, a] = answers(&text);
        let t = t.unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        assert!(t.contains("EmbedGemma2Config"), "{t}");
        // Pinned shapes of google/embeddinggemma-2.
        assert!(t.contains("embedding_dim: 768"), "{}: {t}", p.display());
        eprintln!("{}: text ok; vision {:?}; audio {:?}", p.display(), v.map(|x| x.len()), a.map(|x| x.len()));
    }
}

#[test]
fn upload_segments_texts_and_bounds() {
    with_gpu(|rt| {
        for (name, f) in [
            ("bert", tessl::bert::upload_segments as UploadFn),
            ("embedgemma2", tessl::embedgemma2::upload_segments as UploadFn),
        ] {
            let e = |segs: &[(u32, u32)], rows: u32| f(rt, segs, rows).map(|_| ()).unwrap_err();
            assert_eq!(
                e(&[(0, 4), (4, 9)], 8),
                format!("{name}::upload_segments: segment 1 is [4, 9); need start < end <= 8")
            );
            assert_eq!(
                e(&[(3, 3)], 8),
                format!("{name}::upload_segments: segment 0 is [3, 3); need start < end <= 8")
            );
            assert_eq!(
                e(&[(0, 1), (5, 2)], 8),
                format!("{name}::upload_segments: segment 1 is [5, 2); need start < end <= 8")
            );
            assert!(f(rt, &[], 0).is_ok(), "{name}: an empty list uploads nothing and is not an error");
            assert!(f(rt, &[(0, 2), (2, 8)], 8).is_ok(), "{name}");
        }
    });
}

/// `EmbedGemma2Model::encode`'s refusals before any GPU work, exact texts.
#[test]
fn embedgemma2_encode_refuses_a_bad_batch_with_exact_texts() {
    use common::embedgemma2::{open, random_weights, safetensors_bytes, PREFIX};
    use tessl::embedgemma2::{EmbedGemma2Model, MAX_BATCH};
    const W: &str = "EmbedGemma2Model::encode";
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let st = open("batch_texts", safetensors_bytes(&random_weights(&cfg, 5, false), &[]));
    with_gpu(|rt| {
        let model = EmbedGemma2Model::load(rt, &st, PREFIX, cfg.clone()).unwrap();
        let err = |b: &[&[u32]], dim: Option<u32>, trace: bool| model.encode(b, dim, trace).map(|_| ()).unwrap_err();
        let ok: Vec<u32> = vec![1, 2, 3];
        assert_eq!(err(&[], None, false), format!("{W}: empty batch"));
        let many: Vec<&[u32]> = vec![ok.as_slice(); MAX_BATCH + 1];
        assert_eq!(
            err(&many, None, false),
            format!("{W}: {} sequences exceeds the batch limit {MAX_BATCH}", MAX_BATCH + 1)
        );
        assert_eq!(err(&[&ok, &ok], None, true), format!("{W}: trace takes one sequence, got 2"));
        assert_eq!(err(&[&ok, &[]], None, false), format!("{W}: sequence 1 is empty"));
        assert_eq!(
            err(&[&ok], Some(0), false),
            format!("{W}: truncate_dim 0 must be in 1..={}", cfg.embedding_dim)
        );
        assert_eq!(
            err(&[&[cfg.vocab]], None, false),
            format!("{W}: sequence 0: token id {} >= vocab {}", cfg.vocab, cfg.vocab)
        );
        let long = vec![1u32; model.max_tokens() as usize + 1];
        assert_eq!(
            err(&[&long], None, false),
            format!(
                "{W}: sequence 0 has {} tokens; the limit is {} (truncate it, or raise it with set_max_tokens)",
                long.len(),
                model.max_tokens()
            )
        );
    });
}
