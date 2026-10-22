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

/// `tie_word_embeddings` is read, not refused: false is the 9B's layout (its
/// own `lm_head.weight`); the root and `text_config` must agree, and one must
/// say.
#[test]
fn tie_word_embeddings_is_read_and_must_be_consistent() {
    assert!(Qwen35Config::from_config_json(REAL).unwrap().tie_word_embeddings);
    let both = |from: &str, to: &str| -> String {
        REAL.replace(
            &format!("\"tie_word_embeddings\": {from}"),
            &format!("\"tie_word_embeddings\": {to}"),
        )
    };
    assert_eq!(
        REAL.matches("\"tie_word_embeddings\": true").count(),
        2,
        "root and text_config"
    );
    let untied = Qwen35Config::from_config_json(&both("true", "false")).unwrap();
    assert!(!untied.tie_word_embeddings);
    assert_eq!(
        Qwen35Config {
            tie_word_embeddings: true,
            ..untied
        },
        Qwen35Config::qwen35_2b().unwrap()
    );
    // Only text_config flipped: the two disagree.
    refused(
        &edited(&[(
            "\"rms_norm_eps\": 1e-06,\n        \"tie_word_embeddings\": true",
            "\"rms_norm_eps\": 1e-06,\n        \"tie_word_embeddings\": false",
        )]),
        "disagrees",
    );
    refused(&both("true", "\"yes\""), "must be true or false");
}

/// The REAL config with `tie_word_embeddings` at the root (its first
/// occurrence) and in `text_config` (its second) set to the given JSON value,
/// or, for `None`, replaced by an unrelated key so the key is absent.
fn tie_edit(root: Option<&str>, text: Option<&str>) -> String {
    let key = "\"tie_word_embeddings\": true";
    let spots: Vec<usize> = REAL.match_indices(key).map(|(i, _)| i).collect();
    assert_eq!(spots.len(), 2, "the key at root and in text_config");
    let put = |v: Option<&str>| match v {
        Some(v) => format!("\"tie_word_embeddings\": {v}"),
        None => "\"unrelated_key\": 0".to_string(),
    };
    format!(
        "{}{}{}{}{}",
        &REAL[..spots[0]],
        put(root),
        &REAL[spots[0] + key.len()..spots[1]],
        put(text),
        &REAL[spots[1] + key.len()..]
    )
}

/// transformers 5.17.0 defaults the key to False (untied) in both config
/// classes (configuration_qwen3_5.py:93 and :184), so a config that is silent
/// is ambiguous and refused; and the three ways to disagree or misspell it
/// are each refused by name: only the root flipped, only `text_config`
/// flipped (above), and a number where a bool belongs.
#[test]
fn tie_word_embeddings_refusals_name_the_key() {
    // Sanity: the helper reproduces the real file when it changes nothing.
    assert_eq!(tie_edit(Some("true"), Some("true")), REAL);
    // Silent at both places: refused, not defaulted to either layout.
    refused(&tie_edit(None, None), "tie_word_embeddings is not set");
    // One place alone states it: accepted, either place, either value.
    assert!(
        Qwen35Config::from_config_json(&tie_edit(None, Some("true")))
            .unwrap()
            .tie_word_embeddings
    );
    assert!(
        !Qwen35Config::from_config_json(&tie_edit(Some("false"), None))
            .unwrap()
            .tie_word_embeddings
    );
    // Only the root flipped: it disagrees with text_config.
    refused(&tie_edit(Some("false"), Some("true")), "disagrees");
    // A number is not a bool, at either place and for both 0 and 1.
    for v in ["0", "1"] {
        refused(&tie_edit(Some(v), Some("true")), "must be true or false");
        refused(&tie_edit(Some("true"), Some(v)), "must be true or false");
        refused(&tie_edit(Some(v), Some(v)), "must be true or false");
    }
}

/// An absent `attn_output_gate` or `mamba_ssm_dtype` takes the value the code
/// implements (gated attention, a float32 GDN state), and a present key with a
/// different value is refused (tessl ft-b8561dbb's rule). transformers 5.17.0
/// applies the gate unconditionally and casts the GDN rules to float32
/// (`modeling_qwen3_5.py`), so a config that omits either key describes this
/// model.
#[test]
fn absent_gate_and_state_dtype_keys_take_the_implemented_values() {
    let want = Qwen35Config::qwen35_2b().unwrap();
    for (key, text) in [
        ("attn_output_gate", edited(&[("\"attn_output_gate\": true,", "")])),
        ("mamba_ssm_dtype", edited(&[("\"mamba_ssm_dtype\": \"float32\",", "")])),
        (
            "both",
            edited(&[
                ("\"attn_output_gate\": true,", ""),
                ("\"mamba_ssm_dtype\": \"float32\",", ""),
            ]),
        ),
    ] {
        assert!(!text.contains(key) || key == "both", "the edit removed {key}");
        let got = Qwen35Config::from_config_json(&text).unwrap_or_else(|e| panic!("absent {key}: {e}"));
        assert_eq!(got, want, "absent {key}");
    }
    refused(
        &edited(&[("\"mamba_ssm_dtype\": \"float32\"", "\"mamba_ssm_dtype\": \"bfloat16\"")]),
        "mamba_ssm_dtype",
    );
}

#[test]
fn features_the_forward_lacks_are_refused_by_name() {
    refused(
        &edited(&[("\"attention_bias\": false", "\"attention_bias\": true")]),
        "attention_bias",
    );
    refused(
        &edited(&[("\"attn_output_gate\": true", "\"attn_output_gate\": false")]),
        "ungated attention",
    );
    refused(
        &edited(&[("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"")]),
        "hidden_act \"gelu\"",
    );
    refused(
        &edited(&[("\"mlp_only_layers\": []", "\"mlp_only_layers\": [3]")]),
        "mlp_only_layers",
    );
    refused(
        &edited(&[(
            "\"mlp_only_layers\": []",
            "\"mlp_only_layers\": [], \"num_experts\": 256",
        )]),
        "mixture-of-experts",
    );
    refused(
        &edited(&[("\"rope_type\": \"default\"", "\"rope_type\": \"yarn\"")]),
        "rope_type",
    );
    refused(
        &edited(&[("\"linear_key_head_dim\": 128", "\"linear_key_head_dim\": 64")]),
        "linear_key_head_dim 64",
    );
    refused(
        &edited(&[("\"head_dim\": 256", "\"head_dim\": 128")]),
        "head_dim 128 is not supported",
    );
    refused(
        &edited(&[("\"partial_rotary_factor\": 0.25", "\"partial_rotary_factor\": 0.3")]),
        "whole number of rotated dims",
    );
    refused(
        &edited(&[("\"partial_rotary_factor\": 0.25", "\"partial_rotary_factor\": 0")]),
        "whole number of rotated dims",
    );
}

#[test]
fn malformed_configs_are_refused() {
    refused(
        &edited(&[("\"num_hidden_layers\": 24", "\"num_hidden_layers\": 23")]),
        "has 24 entries but num_hidden_layers is 23",
    );
    refused(
        &edited(&[("\"head_dim\": 256", "\"head_dim\": \"256\"")]),
        "head_dim must be a non-negative integer",
    );
    refused(
        &edited(&[("\"head_dim\": 256", "\"head_dim\": 256.5")]),
        "head_dim must be a non-negative integer",
    );
    refused(
        &edited(&[("\"head_dim\": 256", "\"head_dim\": -256")]),
        "head_dim must be a non-negative integer",
    );
    refused(
        &edited(&[("\"head_dim\": 256", "\"head_dim\": 4294967296")]),
        "exceeds u32",
    );
    refused(&edited(&[("\"head_dim\": 256,", "")]), "missing \"head_dim\"");
    refused(
        &edited(&[("\"rms_norm_eps\": 1e-06", "\"rms_norm_eps\": \"1e-6\"")]),
        "rms_norm_eps must be a number",
    );
    refused(
        &edited(&[("\"rms_norm_eps\": 1e-06", "\"rms_norm_eps\": 0")]),
        "rms_norm_eps and rope_theta must be positive",
    );
    refused(
        &edited(&[("\"attention_bias\": false", "\"attention_bias\": 0")]),
        "attention_bias must be true or false",
    );
    refused(
        &edited(&[("\"vocab_size\": 248320", "\"vocab_size\": 0")]),
        "must be non-zero",
    );
    let first = "\"linear_attention\",";
    let pos = REAL.find(first).unwrap();
    let bad_kind = format!("{}\"sliding_attention\",{}", &REAL[..pos], &REAL[pos + first.len()..]);
    refused(&bad_kind, "layer_types[0]");
    refused("[1, 2]", "the root is not an object");
    refused("{\"text_config\": {}} x", "trailing bytes");
    refused(
        &REAL.replacen(
            "\"hidden_size\": 2048",
            "\"hidden_size\": 2048, \"hidden_size\": 2048",
            1,
        ),
        "duplicate key",
    );
}

// ------------------------------------------------ the real 27B config ---

/// `mlx-community/Qwen3.8-27B-MTP-4bit`'s `config.json` as published (snapshot
/// b643c01): `model_type` `qwen3_5_mtp`, the 27B's `text_config`, the MLX
/// `quantization` block and `mtp_num_hidden_layers: 1`.
const Q27B: &str = include_str!("fixtures/qwen35/config_qwen38_27b_mtp_4bit.json");

fn edited27(pairs: &[(&str, &str)]) -> String {
    let mut s = Q27B.to_string();
    for (from, to) in pairs {
        assert_eq!(s.matches(from).count(), 1, "fixture edit {from:?} must match once");
        s = s.replace(from, to);
    }
    s
}

#[test]
fn the_real_27b_config_parses_to_the_27b_shape() {
    let c = Qwen35Config::from_config_json(Q27B).expect("the real 27B config.json");
    assert_eq!((c.hidden, c.intermediate, c.vocab), (5120, 17408, 248_320));
    assert_eq!(c.layers.len(), 64);
    // Every fourth layer is full attention: 16 attention, 48 GDN.
    for (l, kind) in c.layers.iter().enumerate() {
        let want = if l % 4 == 3 {
            LayerKind::FullAttention
        } else {
            LayerKind::LinearAttention
        };
        assert_eq!(*kind, want, "layer {l}");
    }
    assert_eq!(c.layers.iter().filter(|k| **k == LayerKind::FullAttention).count(), 16);
    // GDN: 16 key heads, 48 value heads of 128 (three value heads per key head).
    assert_eq!((c.gdn.k_heads(), c.gdn.v_heads(), c.gdn.v_dim()), (16, 48, 128));
    assert_eq!(c.gdn.conv_dim(), 2 * 16 * 128 + 48 * 128);
    // Attention: 24 query and 4 KV heads of 256, a quarter of each head rotated.
    assert_eq!((c.attn.q_heads(), c.attn.kv_heads(), c.attn.head_dim()), (24, 4, 256));
    assert_eq!(c.rotary_dim, 64);
    assert_eq!(c.conv_kernel, 4);
    assert_eq!(c.rope_theta, 1e7);
    assert_eq!(c.rms_norm_eps, 1e-6);
    // Untied: its own 248,320-row head.
    assert!(!c.tie_word_embeddings);
}

/// The decision on the MTP head: it is not part of the tower. The config's
/// `mtp_*` keys (and `block_size`, `model_type`) describe a drafter the forward
/// does not run, so they are read by nobody, on purpose, and changing them
/// changes nothing; the checkpoint's `mtp.*` tensors are skipped by name and
/// reported at load (`classify_outside`, `tests/qwen35_mlx_q4_layout.rs`).
#[test]
fn the_mtp_keys_are_ignored_on_purpose_and_change_nothing() {
    let base = Qwen35Config::from_config_json(Q27B).unwrap();
    for pairs in [
        &[("\"mtp_num_hidden_layers\": 1", "\"mtp_num_hidden_layers\": 3")][..],
        &[(
            "\"mtp_use_dedicated_embeddings\": false",
            "\"mtp_use_dedicated_embeddings\": true",
        )],
        &[("\"block_size\": 3", "\"block_size\": 5")],
        &[("\"model_type\": \"qwen3_5_mtp\"", "\"model_type\": \"qwen3_5\"")],
    ] {
        assert_eq!(
            Qwen35Config::from_config_json(&edited27(pairs)).unwrap(),
            base,
            "{pairs:?}"
        );
    }
}

/// `output_gate_type: "swish"` is in the 27B config only. No reader of it exists
/// in mlx-lm 0.32.0, mlx-vlm 0.7.6 or transformers' qwen3_5 (checked with a
/// source search), all of which gate the attention output with `sigmoid(gate)`
/// whatever the key says. So the config parses and the forward is the sigmoid
/// one; whether the 27B's weights were trained with a swish gate is
/// unverified. This pins the current reading so a change to it is a decision.
#[test]
fn output_gate_type_is_not_read_and_the_sigmoid_gate_is_what_runs() {
    let with = Qwen35Config::from_config_json(Q27B).unwrap();
    let without = Qwen35Config::from_config_json(&edited27(&[("\"output_gate_type\": \"swish\",", "")])).unwrap();
    assert_eq!(with, without);
    let other = Qwen35Config::from_config_json(&edited27(&[(
        "\"output_gate_type\": \"swish\"",
        "\"output_gate_type\": \"sigmoid\"",
    )]))
    .unwrap();
    assert_eq!(with, other);
}

#[test]
fn the_27b_config_refuses_what_the_forward_lacks_by_key() {
    for (pairs, needle) in [
        (
            &[("\"attention_bias\": false", "\"attention_bias\": true")][..],
            "attention_bias",
        ),
        (
            &[("\"attn_output_gate\": true", "\"attn_output_gate\": false")],
            "ungated attention",
        ),
        (
            &[("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"")],
            "hidden_act \"gelu\"",
        ),
        (
            &[("\"rope_type\": \"default\"", "\"rope_type\": \"yarn\"")],
            "rope_type",
        ),
        (
            &[("\"linear_key_head_dim\": 128", "\"linear_key_head_dim\": 64")],
            "linear_key_head_dim 64",
        ),
        (
            &[("\"head_dim\": 256", "\"head_dim\": 128")],
            "head_dim 128 is not supported",
        ),
        (
            &[("\"mamba_ssm_dtype\": \"float32\"", "\"mamba_ssm_dtype\": \"bfloat16\"")],
            "mamba_ssm_dtype",
        ),
        (
            &[("\"num_hidden_layers\": 64", "\"num_hidden_layers\": 63")],
            "has 64 entries but num_hidden_layers is 63",
        ),
        // Untied is stated at the root and in text_config; one flipped disagrees.
        (
            &[(
                "\"tie_word_embeddings\": false,\n    \"vision_config\"",
                "\"tie_word_embeddings\": true,\n    \"vision_config\"",
            )],
            "disagrees",
        ),
    ] {
        match Qwen35Config::from_config_json(&edited27(pairs)) {
            Ok(c) => panic!("expected {needle:?}, parsed {c:?}"),
            Err(e) => assert!(e.contains(needle), "{e:?} lacks {needle:?}"),
        }
    }
}
