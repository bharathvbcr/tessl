//! Adversarial, host-only attacks on the MLX 4-bit loader's untrusted-input
//! surface (`tessl::qwen35_q4`): the checkpoint headers, `config.json`'s
//! `quantization` block, the key spellings, the skip list and the norm
//! convention. No device, no tensor data.
//!
//! The harness is the seeded, dependency-free mutation style of
//! `src/parser_mutation.rs` (card ft-04c475ac): a failing case replays from
//! `MLX_Q4_ADV_SEED` and its printed description alone. Tunable:
//! `MLX_Q4_ADV_CASES` (random cases per property, default 20000).
//!
//! Tests marked `#[ignore = "BREAK: ..."]` are attacks that currently succeed:
//! each asserts the behaviour a fail-closed loader would have, so it fails
//! today and passes when the hole is closed.

use std::panic::{catch_unwind, AssertUnwindSafe};

use tessl::qwen35_q4::{check_part, classify_outside, sidecar_names, MlxNames, MlxQ4Error, MlxQuant, NormConvention};
use tessl::safetensors::{Dtype, SafeTensors, TensorInfo, TensorSource};
use tessl::tensor::{bf16_bits_to_f32, f32_to_bf16_bits};

const ALL_DTYPES: [Dtype; 19] = [
    Dtype::Bool,
    Dtype::U8,
    Dtype::I8,
    Dtype::F8E4M3,
    Dtype::F8E4M3Fnuz,
    Dtype::F8E5M2,
    Dtype::F8E5M2Fnuz,
    Dtype::F8E8M0,
    Dtype::I16,
    Dtype::U16,
    Dtype::F16,
    Dtype::BF16,
    Dtype::I32,
    Dtype::U32,
    Dtype::F32,
    Dtype::F64,
    Dtype::I64,
    Dtype::U64,
    Dtype::C64,
];

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
        (self.next() % n.max(1) as u64) as usize
    }
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().parse().unwrap_or_else(|e| panic!("{key}={v:?}: {e}")))
        .unwrap_or(default)
}

fn seed() -> u64 {
    env_u64("MLX_Q4_ADV_SEED", 0x4d4c_5834_a5ed)
}

fn cases() -> usize {
    env_u64("MLX_Q4_ADV_CASES", 20_000) as usize
}

fn info(dtype: Dtype, shape: &[usize]) -> TensorInfo {
    // Offsets are the loader's `TensorSource::tensor_info` shape of the
    // truth: the header parse already proved `end - begin == numel * size`.
    // These are synthetic, so saturate instead of panicking on huge shapes.
    let numel = shape
        .iter()
        .try_fold(1u64, |a, &d| a.checked_mul(d as u64))
        .unwrap_or(0);
    TensorInfo {
        dtype,
        shape: shape.to_vec(),
        begin: 0,
        end: numel.saturating_mul(dtype.size()),
    }
}

const NAME: &str = "language_model.model.layers.0.mlp.gate_proj.weight";

/// A dimension drawn from the boundary values a header field could carry.
fn dim(rng: &mut Rng) -> usize {
    const EDGE: [usize; 20] = [
        0,
        1,
        2,
        7,
        8,
        9,
        31,
        32,
        63,
        64,
        65,
        127,
        128,
        4096,
        (u32::MAX as usize) - 1,
        u32::MAX as usize,
        (u32::MAX as usize) + 1,
        1 << 40,
        usize::MAX / 8,
        usize::MAX,
    ];
    if rng.below(3) == 0 {
        rng.below(300)
    } else {
        EDGE[rng.below(EDGE.len())]
    }
}

/// The independent oracle: what `check_part` must accept, written from the
/// layout table in the module docs and nothing else.
fn oracle_ok(w: &TensorInfo, s: Option<&TensorInfo>, b: Option<&TensorInfo>, rows: usize, cols: usize) -> bool {
    let fits = |t: &TensorInfo, dt: Dtype, last: usize| t.dtype == dt && t.shape == [rows, last];
    cols % 64 == 0
        && cols % 8 == 0
        && fits(w, Dtype::U32, cols / 8)
        && s.is_some_and(|t| fits(t, Dtype::BF16, cols / 64))
        && b.is_some_and(|t| fits(t, Dtype::BF16, cols / 64))
}

/// Attack 1: `check_part` over boundary shapes, every dtype and missing or
/// swapped sidecars. It must never panic, never overflow, and agree with the
/// oracle in both directions.
#[test]
fn check_part_agrees_with_the_layout_oracle_and_never_panics() {
    let mut rng = Rng(seed());
    let mut accepted = 0usize;
    for case in 0..cases() {
        let (rows, cols) = (dim(&mut rng), dim(&mut rng));
        // Mostly a correct triple, then one to three mutations.
        let mut w = info(Dtype::U32, &[rows, cols / 8]);
        let mut s = Some(info(Dtype::BF16, &[rows, cols / 64]));
        let mut b = Some(info(Dtype::BF16, &[rows, cols / 64]));
        let mut what = vec![format!("rows={rows} cols={cols}")];
        for _ in 0..rng.below(4) {
            let pick = rng.below(3);
            let dtype = ALL_DTYPES[rng.below(ALL_DTYPES.len())];
            let (r, c) = (dim(&mut rng), dim(&mut rng));
            let t: Option<&mut TensorInfo> = match pick {
                0 => Some(&mut w),
                1 => s.as_mut(),
                _ => b.as_mut(),
            };
            let Some(t) = t else { continue };
            match rng.below(6) {
                0 => {
                    t.dtype = dtype;
                    what.push(format!("tensor {pick} dtype {dtype:?}"));
                }
                1 => {
                    t.shape = vec![r, c];
                    what.push(format!("tensor {pick} shape [{r}, {c}]"));
                }
                2 => {
                    t.shape = vec![r];
                    what.push(format!("tensor {pick} rank 1 [{r}]"));
                }
                3 => {
                    t.shape = vec![1, r, c];
                    what.push(format!("tensor {pick} rank 3"));
                }
                4 => {
                    t.shape = vec![];
                    what.push(format!("tensor {pick} scalar"));
                }
                _ => {
                    if pick == 1 {
                        s = None;
                    } else if pick == 2 {
                        b = None;
                    }
                    what.push(format!("sidecar {pick} dropped"));
                }
            }
        }
        let got = catch_unwind(AssertUnwindSafe(|| {
            check_part(NAME, &w, s.as_ref(), b.as_ref(), rows, cols)
        }));
        let got = got.unwrap_or_else(|_| panic!("case {case} (seed {:#x}): check_part panicked on {what:?}", seed()));
        let want = oracle_ok(&w, s.as_ref(), b.as_ref(), rows, cols);
        assert_eq!(
            got.is_ok(),
            want,
            "case {case} (seed {:#x}): {what:?}: check_part says {got:?}, oracle says {want}",
            seed()
        );
        accepted += usize::from(got.is_ok());
    }
    assert!(accepted > 0, "the generator never produced an acceptable part");
}

/// Attack 1 (cols not divisible by 8 or 64): every such width is a typed
/// `Cols` refusal, whatever the tensors say, including the widths whose
/// `cols / 8` and `cols / 64` truncate to something that would "fit".
#[test]
fn widths_the_group_does_not_divide_are_refused_by_cols() {
    for cols in [1usize, 7, 8, 9, 56, 63, 65, 72, 100, 127, 191, 4095, 4104, 4160 - 8] {
        // The header an attacker writes: shapes exactly what truncating
        // division gives, so a shape check alone would accept.
        let w = info(Dtype::U32, &[4, cols / 8]);
        let s = info(Dtype::BF16, &[4, cols / 64]);
        let e = check_part(NAME, &w, Some(&s), Some(&s), 4, cols).unwrap_err();
        assert!(
            matches!(e, MlxQ4Error::Cols { cols: c, .. } if c == cols),
            "{cols}: {e:?}"
        );
    }
}

/// Attack 1 (scale/bias rows and groups): a scales tensor with the wrong rows
/// or group count, in each position, is refused and names which tensor.
#[test]
fn scales_and_biases_with_the_wrong_rows_or_groups_are_refused_by_name() {
    let (rows, cols) = (6usize, 256usize);
    let w = info(Dtype::U32, &[rows, cols / 8]);
    let good = info(Dtype::BF16, &[rows, cols / 64]);
    for bad_shape in [
        vec![rows + 1, cols / 64],
        vec![rows - 1, cols / 64],
        vec![rows, cols / 64 + 1],
        vec![rows, cols / 64 - 1],
        vec![cols / 64, rows],
        vec![rows * (cols / 64)],
        vec![rows, cols / 64, 1],
        vec![],
    ] {
        let bad = info(Dtype::BF16, &bad_shape);
        let e = check_part(NAME, &w, Some(&bad), Some(&good), rows, cols).unwrap_err();
        assert!(
            matches!(&e, MlxQ4Error::Shape { tensor, .. } if tensor.ends_with(".scales")),
            "{bad_shape:?}: {e:?}"
        );
        let e = check_part(NAME, &w, Some(&good), Some(&bad), rows, cols).unwrap_err();
        assert!(
            matches!(&e, MlxQ4Error::Shape { tensor, .. } if tensor.ends_with(".biases")),
            "{bad_shape:?}: {e:?}"
        );
    }
}

/// Attack 1 (zero-row and zero-column tensors). A zero dimension has no
/// group layout worth loading; a fail-closed boundary refuses it by name
/// rather than building an empty bank. `check_part` accepts both today
/// (`0 % 64 == 0`, `[0, 0]` shapes match), so an empty `Q4Weight` is sized
/// from them in `Loader::mlx_q4_named`. Unreachable through a validated
/// `Qwen35Config` (every width is non-zero), so this is a hardening gap at the
/// boundary, not a reachable load.
#[test]
fn zero_row_and_zero_column_parts_are_refused() {
    for (rows, cols) in [(0usize, 64usize), (8, 0), (0, 0)] {
        let w = info(Dtype::U32, &[rows, cols / 8]);
        let s = info(Dtype::BF16, &[rows, cols / 64]);
        let r = check_part(NAME, &w, Some(&s), Some(&s), rows, cols);
        assert!(r.is_err(), "{rows} x {cols} was accepted");
    }
}

// ------------------------------------------------------ header-level fuzz ---

/// A safetensors file of one quantized part, in the converter's layout.
fn part_file(rows: usize, cols: usize) -> Vec<u8> {
    let (w, sb) = (rows * cols / 8 * 4, rows * (cols / 64) * 2);
    let header = format!(
        "{{\"x.weight\":{{\"dtype\":\"U32\",\"shape\":[{rows},{}],\"data_offsets\":[0,{w}]}},\
          \"x.scales\":{{\"dtype\":\"BF16\",\"shape\":[{rows},{}],\"data_offsets\":[{w},{}]}},\
          \"x.biases\":{{\"dtype\":\"BF16\",\"shape\":[{rows},{}],\"data_offsets\":[{},{}]}}}}",
        cols / 8,
        cols / 64,
        w + sb,
        cols / 64,
        w + sb,
        w + 2 * sb
    );
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header.into_bytes());
    out.extend(vec![0u8; w + 2 * sb]);
    out
}

/// What the loader does with one `(weight, scales, biases)` triple of a parsed
/// file; returns whether it was accepted.
fn load_check(st: &SafeTensors, rows: usize, cols: usize) -> Result<bool, String> {
    let w = st.tensor_info("x.weight")?;
    let (sn, bn) = sidecar_names("x.weight").unwrap();
    let (s, b) = (st.tensor_info(&sn).ok(), st.tensor_info(&bn).ok());
    match check_part("x.weight", &w, s.as_ref(), b.as_ref(), rows, cols) {
        Ok(()) => {
            // The invariant the loader's `read_u32_into` and
            // `read_bf16_bits_into` lean on: an accepted part's byte ranges are
            // exactly the bytes the shapes imply.
            assert_eq!(w.end - w.begin, (rows * cols / 2) as u64, "weight bytes");
            for t in [s.unwrap(), b.unwrap()] {
                assert_eq!(t.end - t.begin, (rows * (cols / 64) * 2) as u64, "scale/bias bytes");
            }
            Ok(true)
        }
        Err(_) => Ok(false),
    }
}

/// Attack 1 (a header that lies; huge declared shape on a tiny file): seeded
/// byte edits, digit edits and boundary-integer overwrites over a valid
/// header. Parse plus `check_part` never panics; an accepted part's byte
/// ranges are exactly the shapes' bytes (so a "U32 claimed, data truncated"
/// header cannot get past the header parse, which tiles the buffer); a header
/// that declares a vast shape over a tiny buffer is refused at parse.
#[test]
fn mutated_part_headers_never_panic_and_accepted_ones_are_byte_exact() {
    let (rows, cols) = (4usize, 128usize);
    let base = part_file(rows, cols);
    let mut rng = Rng(seed() ^ 0x1);
    let (mut parsed, mut accepted) = (0usize, 0usize);
    for case in 0..cases() {
        let mut bytes = base.clone();
        let mut what = Vec::new();
        for _ in 0..1 + rng.below(3) {
            match rng.below(4) {
                0 => {
                    let i = 8 + rng.below(bytes.len() - 8);
                    let v = rng.next() as u8;
                    what.push(format!("byte {i} = {v:#x}"));
                    bytes[i] = v;
                }
                1 => {
                    // Overwrite an ASCII digit in the header with another digit.
                    let hlen = (u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize).min(bytes.len() - 8);
                    let digits: Vec<usize> = (8..8 + hlen).filter(|&i| bytes[i].is_ascii_digit()).collect();
                    if digits.is_empty() {
                        continue;
                    }
                    let i = digits[rng.below(digits.len())];
                    let d = b'0' + rng.below(10) as u8;
                    what.push(format!("digit {i} = {}", d as char));
                    bytes[i] = d;
                }
                2 => {
                    // A boundary integer in place of a whole shape field: re-emit
                    // the header with weight shape [x, y].
                    let (x, y) = (dim(&mut rng), dim(&mut rng));
                    let hlen = (u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize).min(bytes.len() - 8);
                    let text = String::from_utf8_lossy(&bytes[8..8 + hlen]).to_string();
                    let needle = format!("\"shape\":[{rows},{}]", cols / 8);
                    if let Some(at) = text.find(&needle) {
                        let t = format!("{}\"shape\":[{x},{y}]{}", &text[..at], &text[at + needle.len()..]);
                        let mut out = (t.len() as u64).to_le_bytes().to_vec();
                        out.extend(t.into_bytes());
                        out.extend(&bytes[8 + hlen..]);
                        bytes = out;
                        what.push(format!("weight shape [{x}, {y}]"));
                    }
                }
                _ => {
                    let at = 8 + rng.below(bytes.len() - 8);
                    what.push(format!("truncate at {at}"));
                    bytes.truncate(at);
                }
            }
            if bytes.len() < 9 {
                break;
            }
        }
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            match SafeTensors::from_bytes("adv", bytes.clone()) {
                Ok(st) => load_check(&st, rows, cols).map(|a| (true, a)),
                Err(_) => Ok((false, false)),
            }
        }));
        let (p, a) = outcome
            .unwrap_or_else(|_| panic!("case {case} (seed {:#x}): panicked on {what:?}", seed()))
            .unwrap_or((true, false));
        parsed += usize::from(p);
        accepted += usize::from(a);
    }
    assert!(parsed > 0 && accepted > 0, "no mutated header parsed or was accepted");
}

/// Attack 1 (a huge declared shape on a tiny file): refused by the header
/// parse, before any allocation sized from the shape.
#[test]
fn a_vast_declared_shape_over_a_tiny_buffer_is_refused_at_parse() {
    for (r, c) in [
        (1usize << 40, 1usize << 20),
        (usize::MAX, 8),
        (u32::MAX as usize, u32::MAX as usize),
    ] {
        let header = format!("{{\"x.weight\":{{\"dtype\":\"U32\",\"shape\":[{r},{c}],\"data_offsets\":[0,16]}}}}");
        let mut f = (header.len() as u64).to_le_bytes().to_vec();
        f.extend(header.into_bytes());
        f.extend([0u8; 16]);
        assert!(SafeTensors::from_bytes("adv", f).is_err(), "{r} x {c} parsed");
    }
}

// ---------------------------------------------------------------- config ---

/// Attack 4: the `quantization` block, mutated. Never panics; accepted only
/// when it is exactly affine / 4 / 64 with no other key; every near-miss is a
/// named refusal.
#[test]
fn quantization_blocks_are_accepted_only_when_exactly_affine_4_bit_group_64() {
    let mut rng = Rng(seed() ^ 0x2);
    let bits = [
        "4",
        "8",
        "2",
        "3",
        "0",
        "-4",
        "4.0",
        "4.5",
        "\"4\"",
        "null",
        "true",
        "4e0",
        "4294967300",
        "1e400",
    ];
    let groups = [
        "64",
        "32",
        "128",
        "16",
        "0",
        "-64",
        "64.0",
        "\"64\"",
        "null",
        "6.4e1",
        "4294967360",
    ];
    let modes = [
        None,
        Some("\"affine\""),
        Some("\"Affine\""),
        Some("\"mxfp4\""),
        Some("\"nvfp4\""),
        Some("\"mxfp8\""),
        Some("null"),
        Some("\"\""),
        Some("\"affine \""),
        Some("1"),
    ];
    let extras = [
        None,
        Some("\"language_model.model.layers.0.mlp.gate\": {\"bits\": 8, \"group_size\": 64}"),
        Some("\"language_model.model.lm_head\": false"),
        Some("\"quant_method\": \"gptq\""),
        Some("\"bits\": 4"),
        Some("\"embed_tokens\": true"),
    ];
    let mut accepted = 0usize;
    for case in 0..cases() {
        let (b, g) = (bits[rng.below(bits.len())], groups[rng.below(groups.len())]);
        let m = modes[rng.below(modes.len())];
        let x = extras[rng.below(extras.len())];
        let mut fields = vec![format!("\"bits\": {b}"), format!("\"group_size\": {g}")];
        if let Some(m) = m {
            fields.push(format!("\"mode\": {m}"));
        }
        if let Some(x) = x {
            fields.push(x.to_string());
        }
        // Rotate so duplicate-key and ordering paths vary.
        let k = rng.below(fields.len());
        fields.rotate_left(k);
        let block = format!("{{{}}}", fields.join(", "));
        let key = ["quantization", "quantization_config", "both"][rng.below(3)];
        let body = match key {
            "both" => format!("\"quantization\": {block}, \"quantization_config\": {block}"),
            k => format!("\"{k}\": {block}"),
        };
        let text = format!("{{\"model_type\": \"qwen3_5\", {body}}}");
        let got = catch_unwind(|| MlxQuant::from_config_json(&text));
        let got = got.unwrap_or_else(|_| panic!("case {case}: panicked on {text}"));
        let pure = |v: &str, want: &[&str]| want.contains(&v);
        let mode_ok = matches!(m, None | Some("\"affine\""));
        // A numeric spelling of 4 or 64 may go either way (the JSON layer owns
        // number grammar), but nothing else may be accepted, an extra key (a
        // per-layer entry, or a repeated "bits") never is, and the canonical
        // block always is.
        if got.is_ok() {
            accepted += 1;
            assert!(
                x.is_none() && pure(b, &["4", "4.0", "4e0"]) && pure(g, &["64", "64.0", "6.4e1"]) && mode_ok,
                "case {case}: accepted {text}"
            );
        } else {
            assert!(
                !(x.is_none() && b == "4" && g == "64" && mode_ok),
                "case {case}: refused the canonical block {text}: {got:?}"
            );
        }
    }
    assert!(accepted > 0, "the generator never produced a canonical block");
}

/// Attack 4: `mode` missing is mlx-lm's default (affine) and is accepted; the
/// refusals for the other formats name the key.
#[test]
fn a_missing_mode_is_affine_and_every_other_mode_is_named() {
    let ok = r#"{"quantization": {"bits": 4, "group_size": 64}}"#;
    assert!(MlxQuant::from_config_json(ok).is_ok());
    for m in ["mxfp4", "nvfp4", "mxfp8", "affine2", "", "AFFINE"] {
        let t = format!(r#"{{"quantization": {{"bits": 4, "group_size": 64, "mode": "{m}"}}}}"#);
        let e = MlxQuant::from_config_json(&t).unwrap_err();
        assert_eq!(e, MlxQ4Error::Mode { mode: m.into() });
    }
}

// ------------------------------------------------------------- spellings ---

/// Attack 1 (both key spellings present at once): refused, in either order and
/// whichever other tensors are around.
#[test]
fn both_key_spellings_at_once_are_refused() {
    let mlx = "language_model.model.layers.0.input_layernorm.weight";
    let hf = "model.language_model.layers.0.input_layernorm.weight";
    assert!(matches!(
        MlxNames::detect([mlx, hf]),
        Err(MlxQ4Error::Convention { .. })
    ));
    assert!(matches!(
        MlxNames::detect([hf, mlx]),
        Err(MlxQ4Error::Convention { .. })
    ));
    assert!(matches!(
        MlxNames::detect(["lm_head.weight", "mtp.fc.weight"]),
        Err(MlxQ4Error::NoTower)
    ));
    assert_eq!(MlxNames::detect([mlx]).unwrap().prefix, "language_model.model.");
    assert_eq!(MlxNames::detect([hf]).unwrap().prefix, "model.language_model.");
}

// ------------------------------------------------------------- skip list ---

/// Attack 6: the skip rule is an exact prefix on tensors OUTSIDE the tower,
/// and nothing a misspelling can reach: near-miss spellings of the skip
/// groups, and of the tower prefix itself, are refused.
#[test]
fn the_skip_list_is_exact_prefix_and_misspellings_are_refused() {
    let skipped = [
        "mtp.fc.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "vision_tower.blocks.0.attn.qkv.weight",
    ];
    let r = classify_outside(skipped).unwrap();
    assert_eq!(
        r.mtp,
        vec![
            "mtp.fc.weight".to_string(),
            "mtp.layers.0.mlp.up_proj.weight".to_string()
        ]
    );
    assert_eq!(r.vision, vec!["vision_tower.blocks.0.attn.qkv.weight".to_string()]);
    for near in [
        "mtp",
        "mtpx.fc.weight",
        "Mtp.fc.weight",
        "language_model.mtp.fc.weight",
        "language_model.model.layers.0.mtp.weight",
        "language_model.mdel.layers.0.mlp.gate_proj.weight",
        "language_model.model.layer.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "lm_head.weight",
        "language_model.lm_head.biases",
        "vision",
        "vision_towers.x",
        "xvisual.x",
        "model.visual",
        "language_model.visual.x",
    ] {
        let r = classify_outside([near]);
        // "model.visual" has no trailing dot and "mtp" has none either.
        assert!(
            matches!(r, Err(MlxQ4Error::Unexpected { count: 1, .. })),
            "{near:?} was skipped or classified: {r:?}"
        );
    }
}

/// Attack 6: the report lists every skipped tensor. Today vision tensors are
/// counted but not named, and the first name list of `Unexpected` is capped at
/// 8 with the count carried beside it. A skipped tensor that is not named is a
/// tensor an auditor cannot find.
#[test]
fn every_skipped_tensor_is_named_in_the_report() {
    let r = classify_outside(["mtp.a", "vision_tower.b", "visual.c"]).unwrap();
    // A `Vec<String>` of vision names would let this compile; today the field
    // is a count, so the assertion is on the Debug text.
    let text = format!("{r:?}");
    assert!(text.contains("vision_tower.b") && text.contains("visual.c"), "{text}");
}

/// Attack 6: a many-tensor skip is reported with its true count, not a capped
/// sample passed off as the whole (`Unexpected` carries `count` beside its
/// first eight names).
#[test]
fn unexpected_tensors_carry_the_full_count_beside_the_capped_names() {
    let names: Vec<String> = (0..40).map(|i| format!("stray.{i}")).collect();
    match classify_outside(names.iter().map(String::as_str)).unwrap_err() {
        MlxQ4Error::Unexpected { count, first } => {
            assert_eq!(count, 40);
            assert_eq!(first.len(), 8);
        }
        other => panic!("{other:?}"),
    }
}

// ------------------------------------------------------- norm convention ---

/// Attack 2: the +1 undo is exact. For every finite bf16 `stored` the loader
/// can see, `unshift` then the kernel's `1 +` returns `stored` bit for bit,
/// except where `|stored| < 2^-24` (a norm weight within 2^-24 of 0, whose
/// `stored - 1` rounds to -1 in f32); there the effective multiplier is off by
/// less than one f32 ulp of 1 (inexact only below 2^-16). Exhaustive over all 65 536 bf16 patterns.
#[test]
fn the_norm_unshift_is_exact_for_every_finite_bf16() {
    let (mut exact, mut tiny) = (0usize, 0usize);
    for bits in 0..=u16::MAX {
        let stored = bf16_bits_to_f32(bits);
        if !stored.is_finite() {
            continue;
        }
        let w = NormConvention::MlxShifted.unshift(stored);
        let back = 1.0f32 + w;
        if back == stored || (back == 0.0 && stored == 0.0) {
            exact += 1;
        } else {
            // The inexact ones: tiny stored values (|v| < 2^-16, error under
            // 2^-24) and huge ones (|v| >= 2^23, off by one in the last place).
            let small = stored.abs() < 2.0f32.powi(-16) && (back - stored).abs() <= 2.0f32.powi(-24);
            let huge = stored.abs() >= 2.0f32.powi(23) && (back - stored).abs() <= stored.abs() * 2.0f32.powi(-23);
            assert!(
                small || huge,
                "bf16 {bits:#06x} = {stored:e}: unshift {w:e}, 1 + w = {back:e}"
            );
            tiny += 1;
        }
        // The Hf convention is the identity.
        assert_eq!(NormConvention::Hf.unshift(stored).to_bits(), stored.to_bits());
    }
    assert!(exact > 30_000, "exact={exact} tiny={tiny}");
    // The values a norm actually holds (|w| < 8 around 1) are all exact.
    for bits in 0..=u16::MAX {
        let v = bf16_bits_to_f32(bits);
        if v.is_finite() && v.abs() >= 2.0f32.powi(-16) && v.abs() < 2.0f32.powi(23) {
            assert_eq!(1.0f32 + NormConvention::MlxShifted.unshift(v), v);
        }
    }
    let _ = f32_to_bf16_bits(1.0);
}

/// Attack 2: the converter rounds `w + 1` to bf16, so a small `w` is gone
/// before this loader sees the file (`1 + 0.003` is `1.0` in bf16). The loader
/// returns the stored value minus one and the kernel's `1 +` reproduces the
/// converter's own rounded multiplier exactly, which is what mlx-lm runs: no
/// loader-side error, and no way to recover the lost bits.
#[test]
fn the_converters_bf16_rounding_of_w_plus_one_is_what_the_kernel_reproduces() {
    for w in [0.003f32, -0.003, 0.0038, 0.0039, 0.0041, 0.2, -0.3] {
        let stored = bf16_bits_to_f32(f32_to_bf16_bits(1.0 + w));
        assert_eq!(1.0f32 + NormConvention::MlxShifted.unshift(stored), stored, "w={w}");
    }
    assert_eq!(bf16_bits_to_f32(f32_to_bf16_bits(1.0 + 0.003)), 1.0);
}

/// Attack 2: the convention is read from `conv1d.weight`'s shape alone. The
/// function has no way to see the norms, so a checkpoint whose conv layout
/// says one convention and whose norms are the other's is loaded under the
/// layout's, silently: documented here as the characterization (the same
/// weights are `[c, 1, k]` and `[c, k, 1]` under two conventions, and nothing
/// else in the header tells them apart).
#[test]
fn the_convention_is_a_function_of_the_conv_shape_alone() {
    let (c, k) = (512usize, 4usize);
    assert_eq!(
        NormConvention::from_conv_shape("c", &[c, 1, k], c, k).unwrap(),
        NormConvention::Hf
    );
    assert_eq!(
        NormConvention::from_conv_shape("c", &[c, k, 1], c, k).unwrap(),
        NormConvention::MlxShifted
    );
    for bad in [
        vec![c, k],
        vec![k, 1, c],
        vec![c, 2, 2],
        vec![c, k, 1, 1],
        vec![c + 1, k, 1],
        vec![],
    ] {
        assert!(NormConvention::from_conv_shape("c", &bad, c, k).is_err(), "{bad:?}");
    }
    // k == 1: `[c, 1, 1]` is both layouts; mlx-lm's `shape[-1] != 1` test
    // treats it as already sanitized and leaves the norms unshifted, and the
    // loader picks `Hf` (identity), which agrees.
    assert_eq!(
        NormConvention::from_conv_shape("c", &[c, 1, 1], c, 1).unwrap(),
        NormConvention::Hf
    );
}
