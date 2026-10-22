//! The MLX affine 4-bit checkpoint layout the Qwen3.5 tower loads
//! (`tessl::qwen35_q4`), and what it refuses. Host only: headers, names and
//! config text, no device and no tensor data.
//!
//! The layout facts under test are the ones read out of mlx-lm 0.32.0's source
//! (`docs/qwen35.md`, "MLX 4-bit weights"): `.weight` is U32 `[rows, cols / 8]`,
//! `.scales` and `.biases` are BF16 `[rows, cols / 64]`, `conv1d.weight` moves
//! to `[conv_dim, k, 1]` and the zero-centred norms are stored as `1 + w`.

use tessl::qwen35_q4::{
    check_part, classify_outside, sidecar_names, MlxNames, MlxQ4Error, MlxQuant, NormConvention, OutsideTensors,
};
use tessl::safetensors::{Dtype, TensorInfo};

fn info(dtype: Dtype, shape: &[usize]) -> TensorInfo {
    let numel: usize = shape.iter().product();
    TensorInfo {
        dtype,
        shape: shape.to_vec(),
        begin: 0,
        end: (numel as u64) * dtype.size(),
    }
}

const NAME: &str = "language_model.model.layers.0.mlp.gate_proj.weight";

fn good_part(rows: usize, cols: usize) -> (TensorInfo, TensorInfo, TensorInfo) {
    (
        info(Dtype::U32, &[rows, cols / 8]),
        info(Dtype::BF16, &[rows, cols / 64]),
        info(Dtype::BF16, &[rows, cols / 64]),
    )
}

fn quant(body: &str) -> String {
    format!("{{\"model_type\": \"qwen3_5\", {body}}}")
}

// ------------------------------------------------------------- config.json ---

#[test]
fn the_converters_quantization_block_is_accepted_in_either_spelling() {
    let q = "{\"group_size\": 64, \"bits\": 4, \"mode\": \"affine\"}";
    let both = quant(&format!("\"quantization\": {q}, \"quantization_config\": {q}"));
    assert_eq!(
        MlxQuant::from_config_json(&both).unwrap(),
        MlxQuant {
            bits: 4,
            group_size: 64
        }
    );
    // Either alone, and mlx-lm's default mode when the key is absent.
    for body in [
        format!("\"quantization\": {q}"),
        format!("\"quantization_config\": {q}"),
        "\"quantization\": {\"group_size\": 64, \"bits\": 4}".to_string(),
    ] {
        MlxQuant::from_config_json(&quant(&body)).unwrap();
    }
}

#[test]
fn other_bits_groups_and_modes_are_refused_by_name() {
    let refuse = |body: &str| MlxQuant::from_config_json(&quant(body)).unwrap_err();
    assert_eq!(
        refuse("\"quantization\": {\"group_size\": 64, \"bits\": 8}"),
        MlxQ4Error::Bits { bits: 8 }
    );
    assert_eq!(
        refuse("\"quantization\": {\"group_size\": 32, \"bits\": 4}"),
        MlxQ4Error::GroupSize { group_size: 32 }
    );
    assert_eq!(
        refuse("\"quantization\": {\"group_size\": 128, \"bits\": 4, \"mode\": \"affine\"}"),
        MlxQ4Error::GroupSize { group_size: 128 }
    );
    assert_eq!(
        refuse("\"quantization\": {\"group_size\": 32, \"bits\": 4, \"mode\": \"mxfp4\"}"),
        MlxQ4Error::Mode { mode: "mxfp4".into() }
    );
    assert_eq!(refuse("\"hidden_size\": 8"), MlxQ4Error::NotQuantized);
}

#[test]
fn per_layer_overrides_are_a_mixed_checkpoint_and_refused() {
    // mlx-lm records a layer quantized differently as `"<path>": {...}` beside
    // the global entries, and one left dense as `"<path>": false`.
    let e = MlxQuant::from_config_json(&quant(
        "\"quantization\": {\"group_size\": 64, \"bits\": 4, \
         \"language_model.model.layers.0.mlp.gate\": {\"group_size\": 64, \"bits\": 8}, \
         \"language_model.model.embed_tokens\": false}",
    ))
    .unwrap_err();
    match &e {
        MlxQ4Error::Mixed { overrides } => assert_eq!(
            overrides,
            &[
                "language_model.model.layers.0.mlp.gate".to_string(),
                "language_model.model.embed_tokens".to_string()
            ]
        ),
        other => panic!("expected Mixed, got {other:?}"),
    }
    assert!(e.to_string().contains("layers.0.mlp.gate"), "{e}");
}

#[test]
fn malformed_quantization_blocks_are_refused() {
    let m = |body: &str| MlxQuant::from_config_json(&quant(body)).unwrap_err();
    assert!(matches!(m("\"quantization\": 4"), MlxQ4Error::MalformedConfig { .. }));
    assert!(matches!(
        m("\"quantization\": {\"group_size\": 64, \"bits\": \"4\"}"),
        MlxQ4Error::MalformedConfig { .. }
    ));
    assert!(matches!(
        m("\"quantization\": {\"group_size\": 64}"),
        MlxQ4Error::MalformedConfig { .. }
    ));
    // The two spellings must agree: a config whose copies differ is not read
    // as either.
    assert!(matches!(
        m("\"quantization\": {\"group_size\": 64, \"bits\": 4}, \
           \"quantization_config\": {\"group_size\": 64, \"bits\": 8}"),
        MlxQ4Error::MalformedConfig { .. }
    ));
    assert!(MlxQuant::from_config_json("[]").is_err());
}

// -------------------------------------------------------- tensor headers ---

#[test]
fn a_well_formed_part_is_accepted() {
    let (w, s, b) = good_part(6144, 2048);
    check_part(NAME, &w, Some(&s), Some(&b), 6144, 2048).unwrap();
}

#[test]
fn sidecars_are_named_after_the_weight() {
    assert_eq!(
        sidecar_names(NAME).unwrap(),
        (
            "language_model.model.layers.0.mlp.gate_proj.scales".to_string(),
            "language_model.model.layers.0.mlp.gate_proj.biases".to_string()
        )
    );
    assert!(sidecar_names("language_model.model.norm.scales").is_none());
}

#[test]
fn a_dense_weight_in_a_quantized_checkpoint_is_refused() {
    // A layer the converter left unquantized: BF16 `[rows, cols]`.
    let (_, s, b) = good_part(8, 128);
    let dense = info(Dtype::BF16, &[8, 128]);
    match check_part(NAME, &dense, Some(&s), Some(&b), 8, 128).unwrap_err() {
        MlxQ4Error::DenseWeight { tensor, dtype } => {
            assert_eq!(tensor, NAME);
            assert!(dtype.contains("BF16"), "{dtype}");
        }
        other => panic!("expected DenseWeight, got {other:?}"),
    }
}

#[test]
fn a_missing_scales_or_biases_is_refused_by_which() {
    let (w, s, b) = good_part(8, 128);
    assert_eq!(
        check_part(NAME, &w, None, Some(&b), 8, 128).unwrap_err(),
        MlxQ4Error::MissingSidecar {
            tensor: NAME.into(),
            missing: "scales"
        }
    );
    assert_eq!(
        check_part(NAME, &w, Some(&s), None, 8, 128).unwrap_err(),
        MlxQ4Error::MissingSidecar {
            tensor: NAME.into(),
            missing: "biases"
        }
    );
}

#[test]
fn eight_bit_widths_group_widths_and_wrong_dtypes_fail_closed() {
    let (w, s, b) = good_part(8, 128);
    // 8-bit packs four values per word: `[8, 128 / 4]`, not `[8, 128 / 8]`.
    let eight_bit = info(Dtype::U32, &[8, 32]);
    assert!(matches!(
        check_part(NAME, &eight_bit, Some(&s), Some(&b), 8, 128).unwrap_err(),
        MlxQ4Error::Shape { .. }
    ));
    // Group 32 scales: `[8, 128 / 32]`.
    let g32 = info(Dtype::BF16, &[8, 4]);
    let e = check_part(NAME, &w, Some(&g32), Some(&b), 8, 128).unwrap_err();
    assert!(
        matches!(&e, MlxQ4Error::Shape { tensor, .. } if tensor.ends_with(".scales")),
        "{e:?}"
    );
    // float16 scales (a float16 model) are another dtype, not widened here.
    let f16 = info(Dtype::F16, &[8, 2]);
    let e = check_part(NAME, &w, Some(&s), Some(&f16), 8, 128).unwrap_err();
    assert!(
        matches!(&e, MlxQ4Error::Dtype { tensor, .. } if tensor.ends_with(".biases")),
        "{e:?}"
    );
    // The rows the model expects must be the rows the file has.
    assert!(matches!(
        check_part(NAME, &w, Some(&s), Some(&b), 16, 128).unwrap_err(),
        MlxQ4Error::Shape { .. }
    ));
    // A weight whose columns no group divides has no layout at all.
    let (w, s, b) = good_part(8, 128);
    assert!(matches!(
        check_part(NAME, &w, Some(&s), Some(&b), 8, 120).unwrap_err(),
        MlxQ4Error::Cols { cols: 120, .. }
    ));
}

// ------------------------------------------------- names and conventions ---

#[test]
fn the_key_spelling_is_detected_and_ambiguity_refused() {
    let mlx = [
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
    ];
    assert_eq!(
        MlxNames::detect(mlx).unwrap(),
        MlxNames {
            prefix: "language_model.model.",
            head: "language_model.lm_head"
        }
    );
    let hf = ["model.language_model.embed_tokens.weight", "lm_head.weight"];
    assert_eq!(
        MlxNames::detect(hf).unwrap(),
        MlxNames {
            prefix: "model.language_model.",
            head: "lm_head"
        }
    );
    assert_eq!(MlxNames::detect(["model.visual.x"]).unwrap_err(), MlxQ4Error::NoTower);
    assert!(matches!(
        MlxNames::detect([mlx[0], hf[0]]).unwrap_err(),
        MlxQ4Error::Convention { .. }
    ));
}

#[test]
fn the_conv_layout_says_which_norm_convention_the_checkpoint_is_in() {
    let (c, k) = (6144, 4);
    assert_eq!(
        NormConvention::from_conv_shape("conv", &[c, 1, k], c, k).unwrap(),
        NormConvention::Hf
    );
    assert_eq!(
        NormConvention::from_conv_shape("conv", &[c, k, 1], c, k).unwrap(),
        NormConvention::MlxShifted
    );
    assert!(matches!(
        NormConvention::from_conv_shape("conv", &[c, 2, 2], c, k).unwrap_err(),
        MlxQ4Error::Shape { .. }
    ));
    // The kernels add 1 themselves, so the shifted convention is undone, and
    // exactly: a bf16 value near 1 minus 1 is exact in f32, and adding the 1
    // back gives the stored value again bit for bit.
    for stored in [0.0f32, 0.5, 1.0, 1.0078125, 1.75, 2.5, 33.0, -0.25, 0.003_906_25] {
        let w = NormConvention::MlxShifted.unshift(stored);
        assert_eq!((1.0 + w).to_bits(), stored.to_bits(), "stored {stored}");
        assert_eq!(NormConvention::Hf.unshift(stored).to_bits(), stored.to_bits());
    }
    assert_eq!(NormConvention::MlxShifted.conv_shape(c, k), [c, k, 1]);
    assert_eq!(NormConvention::Hf.conv_shape(c, k), [c, 1, k]);
}

// ------------------------------------------- what is skipped, by name ---

#[test]
fn mtp_and_vision_tensors_are_skipped_by_name_and_counted() {
    // What a Hugging Face Qwen3.5 checkpoint carries outside the text tower:
    // the multi-token-prediction head at the root, and the vision tower.
    let unread = [
        "mtp.fc.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "mtp.norm.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "model.visual.blocks.1.attn.qkv.weight",
        "vision_tower.merger.weight",
    ];
    let got = classify_outside(unread).unwrap();
    assert_eq!(
        got,
        OutsideTensors {
            mtp: vec![
                "mtp.fc.weight".into(),
                "mtp.layers.0.mlp.down_proj.weight".into(),
                "mtp.norm.weight".into()
            ],
            vision: vec![
                "model.visual.blocks.0.attn.qkv.weight".into(),
                "model.visual.blocks.1.attn.qkv.weight".into(),
                "vision_tower.merger.weight".into()
            ]
        }
    );
    assert_eq!(classify_outside(Vec::<&str>::new()).unwrap(), OutsideTensors::default());
}

#[test]
fn any_other_tensor_outside_the_tower_is_refused() {
    let e = classify_outside(["mtp.fc.weight", "lm_head.bias", "audio_tower.x"]).unwrap_err();
    match &e {
        MlxQ4Error::Unexpected { count, first } => {
            assert_eq!(*count, 2);
            assert_eq!(first, &["lm_head.bias".to_string(), "audio_tower.x".to_string()]);
        }
        other => panic!("expected Unexpected, got {other:?}"),
    }
    assert!(e.to_string().contains("lm_head.bias"), "{e}");
}

/// The tensor names of `mlx-community/Qwen3.8-27B-MTP-4bit`'s index (snapshot
/// b643c01): a 239 MB drafter, not the 27B. It has no embedding and no
/// `language_model.model.` or `model.language_model.` prefix, so the tower
/// loader fails closed on it ([`MlxQ4Error::NoTower`]) instead of reading it as
/// a tower that happens to lack layers.
#[test]
fn the_mtp_drafter_checkpoint_is_not_mistaken_for_a_tower() {
    let names = [
        "fc.biases",
        "fc.scales",
        "fc.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.mlp.down_proj.biases",
        "layers.0.mlp.down_proj.scales",
        "layers.0.mlp.down_proj.weight",
        "layers.0.self_attn.q_proj.weight",
        "norm.weight",
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
    ];
    assert_eq!(MlxNames::detect(names).unwrap_err(), MlxQ4Error::NoTower);
}

#[test]
fn errors_render_with_the_offending_key_or_tensor() {
    let msgs = [
        MlxQ4Error::Bits { bits: 8 }.to_string(),
        MlxQ4Error::GroupSize { group_size: 32 }.to_string(),
        MlxQ4Error::Mode { mode: "nvfp4".into() }.to_string(),
        MlxQ4Error::DenseWeight {
            tensor: NAME.into(),
            dtype: "BF16".into(),
        }
        .to_string(),
    ];
    assert!(msgs[0].contains("quantization.bits") && msgs[0].contains('8'));
    assert!(msgs[1].contains("quantization.group_size") && msgs[1].contains("32"));
    assert!(msgs[2].contains("quantization.mode") && msgs[2].contains("nvfp4"));
    assert!(msgs[3].contains(NAME));
    let s: String = MlxQ4Error::NoTower.into();
    assert!(s.contains("language_model.model."));
}
