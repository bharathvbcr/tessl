//! `Qwen35Config::from_config_json` against Hugging Face configs (host only).
//!
//! `tests/fixtures/qwen35/config_qwen35_2b_base.json` is `Qwen/Qwen3.5-2B-Base`'s
//! `config.json` as published. It must parse to exactly the hand-written
//! `Qwen35Config::qwen35_2b()` that `tests/qwen35_model.rs` checks against
//! transformers; every other case is that file edited, so each assertion names
//! one field.
//!
//! What this covers is parsing: a config for another size is read correctly
//! and refused where the kernels cannot run it. Whether such a model's forward
//! is right is only known after running it against transformers with its
//! weights, as `tests/qwen35_model.rs` does for the 2B.

use tessl::qwen35_model::{LayerKind, Qwen35Config};

const REAL: &str = include_str!("fixtures/qwen35/config_qwen35_2b_base.json");

fn edited(pairs: &[(&str, &str)]) -> String {
    let mut s = REAL.to_string();
    for (from, to) in pairs {
        assert_eq!(s.matches(from).count(), 1, "fixture edit {from:?} must match once");
        s = s.replace(from, to);
    }
    s
}

fn refused(text: &str, needle: &str) {
    match Qwen35Config::from_config_json(text) {
        Ok(c) => panic!("expected {needle:?}, parsed {c:?}"),
        Err(e) => assert!(e.contains(needle), "{e:?} lacks {needle:?}"),
    }
}

#[test]
fn the_published_2b_config_is_the_hand_written_one() {
    let parsed = Qwen35Config::from_config_json(REAL).expect("the real config.json");
    assert_eq!(parsed, Qwen35Config::qwen35_2b().unwrap());
}

#[test]
fn a_text_only_config_at_the_root_parses_the_same() {
    // What a text-only checkpoint's config.json holds: text_config's fields
    // at the root.
    let start = REAL.find("\"text_config\": {").expect("text_config") + "\"text_config\": ".len();
    let body = &REAL[start..];
    let mut depth = 0;
    let end = body
        .char_indices()
        .find_map(|(i, ch)| {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i + 1);
                    }
                }
                _ => {}
            }
            None
        })
        .expect("end of text_config");
    let text_only = &body[..end];
    assert_eq!(
        Qwen35Config::from_config_json(text_only).unwrap(),
        Qwen35Config::qwen35_2b().unwrap()
    );
}

#[test]
fn layer_types_default_from_the_interval_and_other_sizes_parse() {
    // No layer_types: transformers derives them from full_attention_interval
    // (full when (i + 1) % interval == 0).
    let a = REAL.find("\"layer_types\": [").unwrap();
    let b = a + REAL[a..].find(']').unwrap() + 1;
    let no_types = format!("{}\"_unused\": 0{}", &REAL[..a], &REAL[b..]);
    assert_eq!(
        Qwen35Config::from_config_json(&no_types).unwrap(),
        Qwen35Config::qwen35_2b().unwrap()
    );

    // A different size: every field it reads comes from the file.
    let other = edited(&[
        ("\"hidden_size\": 2048", "\"hidden_size\": 1024"),
        ("\"intermediate_size\": 6144", "\"intermediate_size\": 3584"),
        ("\"linear_num_value_heads\": 16", "\"linear_num_value_heads\": 32"),
        ("\"linear_value_head_dim\": 128", "\"linear_value_head_dim\": 64"),
        ("\"num_attention_heads\": 8", "\"num_attention_heads\": 16"),
        ("\"num_key_value_heads\": 2", "\"num_key_value_heads\": 4"),
        ("\"vocab_size\": 248320", "\"vocab_size\": 151936"),
        ("\"rms_norm_eps\": 1e-06", "\"rms_norm_eps\": 1e-05"),
        ("\"rope_theta\": 10000000", "\"rope_theta\": 1000000.0"),
        ("\"partial_rotary_factor\": 0.25", "\"partial_rotary_factor\": 0.5"),
        ("\"linear_conv_kernel_dim\": 4", "\"linear_conv_kernel_dim\": 3"),
    ]);
    let c = Qwen35Config::from_config_json(&other).unwrap();
    assert_eq!((c.hidden, c.intermediate, c.vocab), (1024, 3584, 151_936));
    assert_eq!((c.gdn.k_heads(), c.gdn.v_heads(), c.gdn.v_dim()), (16, 32, 64));
    assert_eq!((c.attn.q_heads(), c.attn.kv_heads(), c.attn.head_dim()), (16, 4, 256));
    assert_eq!((c.rotary_dim, c.conv_kernel), (128, 3));
    assert_eq!((c.rope_theta, c.rms_norm_eps), (1e6, 1e-5));
    assert_eq!(c.layers.len(), 24);
    assert_eq!(c.layers.iter().filter(|&&k| k == LayerKind::FullAttention).count(), 6);
}

#[test]
fn features_the_forward_lacks_are_refused_by_name() {
    let untied_text = edited(&[(
        "\"rms_norm_eps\": 1e-06,\n        \"tie_word_embeddings\": true",
        "\"rms_norm_eps\": 1e-06,\n        \"tie_word_embeddings\": false",
    )]);
    refused(&untied_text, "untied embeddings");
    refused(&edited(&[("\"attention_bias\": false", "\"attention_bias\": true")]), "attention_bias");
    refused(&edited(&[("\"attn_output_gate\": true", "\"attn_output_gate\": false")]), "ungated attention");
    refused(&edited(&[("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"")]), "hidden_act \"gelu\"");
    refused(&edited(&[("\"mlp_only_layers\": []", "\"mlp_only_layers\": [3]")]), "mlp_only_layers");
    refused(&edited(&[("\"mlp_only_layers\": []", "\"mlp_only_layers\": [], \"num_experts\": 256")]), "mixture-of-experts");
    refused(&edited(&[("\"rope_type\": \"default\"", "\"rope_type\": \"yarn\"")]), "rope_type");
    refused(&edited(&[("\"linear_key_head_dim\": 128", "\"linear_key_head_dim\": 64")]), "linear_key_head_dim 64");
    refused(&edited(&[("\"head_dim\": 256", "\"head_dim\": 128")]), "head_dim 128 is not supported");
    refused(&edited(&[("\"partial_rotary_factor\": 0.25", "\"partial_rotary_factor\": 0.3")]), "whole number of rotated dims");
    refused(&edited(&[("\"partial_rotary_factor\": 0.25", "\"partial_rotary_factor\": 0")]), "whole number of rotated dims");
}

#[test]
fn malformed_configs_are_refused() {
    refused(&edited(&[("\"num_hidden_layers\": 24", "\"num_hidden_layers\": 23")]), "has 24 entries but num_hidden_layers is 23");
    refused(&edited(&[("\"head_dim\": 256", "\"head_dim\": \"256\"")]), "head_dim must be a non-negative integer");
    refused(&edited(&[("\"head_dim\": 256", "\"head_dim\": 256.5")]), "head_dim must be a non-negative integer");
    refused(&edited(&[("\"head_dim\": 256", "\"head_dim\": -256")]), "head_dim must be a non-negative integer");
    refused(&edited(&[("\"head_dim\": 256", "\"head_dim\": 4294967296")]), "exceeds u32");
    refused(&edited(&[("\"head_dim\": 256,", "")]), "missing \"head_dim\"");
    refused(&edited(&[("\"rms_norm_eps\": 1e-06", "\"rms_norm_eps\": \"1e-6\"")]), "rms_norm_eps must be a number");
    refused(&edited(&[("\"rms_norm_eps\": 1e-06", "\"rms_norm_eps\": 0")]), "rms_norm_eps and rope_theta must be positive");
    refused(&edited(&[("\"attention_bias\": false", "\"attention_bias\": 0")]), "attention_bias must be true or false");
    refused(&edited(&[("\"vocab_size\": 248320", "\"vocab_size\": 0")]), "must be non-zero");
    let first = "\"linear_attention\",";
    let pos = REAL.find(first).unwrap();
    let bad_kind = format!("{}\"sliding_attention\",{}", &REAL[..pos], &REAL[pos + first.len()..]);
    refused(&bad_kind, "layer_types[0]");
    refused("[1, 2]", "the root is not an object");
    refused("{\"text_config\": {}} x", "trailing bytes");
    refused(&REAL.replacen("\"hidden_size\": 2048", "\"hidden_size\": 2048, \"hidden_size\": 2048", 1), "duplicate key");
}
