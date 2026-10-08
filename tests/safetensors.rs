//! `tessl::safetensors` against hand-built files: every malformed shape the
//! module documents as rejected, plus exact round trips. CPU only.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use tessl::safetensors::{Dtype, SafeTensors, MAX_HEADER_BYTES};

static SEQ: AtomicUsize = AtomicUsize::new(0);

fn tmp(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("tessl-st-{}-{tag}-{n}.safetensors", std::process::id()))
}

/// Write `len8 | header | data` and return the path.
fn raw(tag: &str, len8: u64, header: &[u8], data: &[u8]) -> PathBuf {
    let p = tmp(tag);
    let mut b = len8.to_le_bytes().to_vec();
    b.extend_from_slice(header);
    b.extend_from_slice(data);
    std::fs::write(&p, b).unwrap();
    p
}

/// A well-formed file from a header string and data bytes.
fn file(tag: &str, header: &str, data: &[u8]) -> PathBuf {
    raw(tag, header.len() as u64, header.as_bytes(), data)
}

fn open_err(p: PathBuf) -> String {
    let r = SafeTensors::open(&p);
    let _ = std::fs::remove_file(&p);
    match r {
        Ok(_) => panic!("{} opened, expected an error", p.display()),
        Err(e) => e,
    }
}

fn expect_rejected(tag: &str, header: &str, data: &[u8], needle: &str) {
    let e = open_err(file(tag, header, data));
    assert!(e.contains(needle), "{tag}: error {e:?} does not mention {needle:?}");
}

const ONE_F32: &str = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;

#[test]
fn round_trips_f32_bf16_f16_scalars_empty_tensors_and_metadata() {
    let a = [1.5f32, -2.25, f32::MAX, f32::MIN_POSITIVE];
    let b_bits: [u16; 3] = [0x3f80, 0xc0a0, 0x7f80]; // 1, -5, +inf
    let h_bits: [u16; 2] = [0x3c00, 0xfbff]; // 1, -65504
    let s = 7.0f32;
    let mut data = Vec::new();
    for v in a {
        data.extend_from_slice(&v.to_le_bytes());
    }
    for v in b_bits {
        data.extend_from_slice(&v.to_le_bytes());
    }
    for v in h_bits {
        data.extend_from_slice(&v.to_le_bytes());
    }
    data.extend_from_slice(&[9, 8, 7, 6]); // "pad", U8 [4]
    data.extend_from_slice(&s.to_le_bytes());
    // Keys out of offset order, a zero-size tensor, metadata with escapes, and
    // the space padding real writers add after the object.
    let header = concat!(
        r#"{"__metadata__":{"format":"pt","note":"q\"\\\/é😀"},"#,
        r#""s":{"dtype":"F32","shape":[],"data_offsets":[30,34]},"#,
        r#""e":{"dtype":"BF16","shape":[0,5],"data_offsets":[30,30]},"#,
        r#""h":{"dtype":"F16","shape":[2],"data_offsets":[22,26]},"#,
        r#""b":{"dtype":"BF16","shape":[3,1],"data_offsets":[16,22]},"#,
        r#""a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"#,
        r#""pad":{"dtype":"U8","shape":[4],"data_offsets":[26,30]}}    "#
    );
    let p = file("ok", header, &data);
    let st = SafeTensors::open(&p).unwrap();
    assert_eq!(st.names().collect::<Vec<_>>(), ["a", "b", "e", "h", "pad", "s"]);
    assert_eq!(st.metadata()["note"], "q\"\\/\u{e9}\u{1f600}");
    assert_eq!(st.read_f32("a").unwrap(), (vec![2, 2], a.to_vec()));
    assert_eq!(st.read_bf16_bits("b").unwrap(), (vec![3, 1], b_bits.to_vec()));
    assert_eq!(st.read_f32("b").unwrap().1, vec![1.0, -5.0, f32::INFINITY]);
    assert_eq!(st.read_f32("h").unwrap().1, vec![1.0, -65504.0]);
    assert_eq!(st.read_f32("s").unwrap(), (vec![], vec![7.0]));
    assert_eq!(st.read_bf16_bits("e").unwrap(), (vec![0, 5], vec![]));
    assert_eq!(st.info("pad").unwrap().dtype, Dtype::U8);
    assert_eq!(st.info("s").unwrap().numel(), 1);

    // Narrowing is refused, and non-float types are not read as numbers.
    assert!(st.read_bf16_bits("a").unwrap_err().contains("expected BF16"));
    assert!(st.read_f32("pad").unwrap_err().contains("cannot read U8"));
    assert!(st.info("missing").unwrap_err().contains("no tensor"));
    std::fs::remove_file(&p).unwrap();
}

/// One file holding every integer dtype; returns its path and each tensor's
/// little-endian bytes, keyed by name.
fn integer_fixture() -> (PathBuf, Vec<(&'static str, Dtype, Vec<usize>, Vec<u8>)>) {
    let le = |vs: &[u64], w: usize| {
        vs.iter()
            .flat_map(|v| v.to_le_bytes()[..w].to_vec())
            .collect::<Vec<u8>>()
    };
    let tensors = vec![
        ("bool", Dtype::Bool, vec![3], vec![1, 0, 1]),
        ("u8", Dtype::U8, vec![2, 2], vec![0, 1, 0x7f, 0xff]),
        ("i8", Dtype::I8, vec![3], vec![0x80, 0xff, 0x7f]), // -128, -1, 127
        ("i16", Dtype::I16, vec![2], le(&[0x8000, 0x1234], 2)),
        ("u16", Dtype::U16, vec![1], le(&[0xbeef], 2)),
        ("i32", Dtype::I32, vec![1], le(&[0xffff_fffe], 4)), // -2
        // Eight 4-bit codes per word, as MLX packs Q4 weights.
        (
            "u32",
            Dtype::U32,
            vec![2, 2],
            le(&[0x7654_3210, 0xfedc_ba98, 0, u32::MAX as u64], 4),
        ),
        ("i64", Dtype::I64, vec![1], le(&[i64::MIN as u64], 8)),
        ("u64", Dtype::U64, vec![2], le(&[u64::MAX, 0x0102_0304_0506_0708], 8)),
    ];
    let (mut header, mut data) = (String::from("{"), Vec::new());
    for (name, dtype, shape, bytes) in &tensors {
        // The format's tag is the variant's name in capitals for every
        // integer type ("BOOL", "U8", ..).
        let tag = format!("{dtype:?}").to_uppercase();
        let dims = shape.iter().map(usize::to_string).collect::<Vec<_>>().join(",");
        let (b, e) = (data.len(), data.len() + bytes.len());
        header.push_str(&format!(
            r#""{name}":{{"dtype":"{tag}","shape":[{dims}],"data_offsets":[{b},{e}]}},"#
        ));
        data.extend_from_slice(bytes);
    }
    header.pop();
    header.push('}');
    (file("ints", &header, &data), tensors)
}

#[test]
fn integer_tensors_read_back_typed_and_raw() {
    let (p, tensors) = integer_fixture();
    let st = SafeTensors::open(&p).unwrap();
    std::fs::remove_file(&p).unwrap();
    for (name, dtype, shape, bytes) in &tensors {
        let (info, raw) = st.read_raw(name).unwrap();
        assert_eq!((info.dtype, &info.shape, &raw), (*dtype, shape, bytes), "{name}");
    }
    assert_eq!(st.read_u8("u8").unwrap(), (vec![2, 2], vec![0, 1, 0x7f, 0xff]));
    assert_eq!(st.read_i8("i8").unwrap(), (vec![3], vec![-128, -1, 127]));
    assert_eq!(
        st.read_u32("u32").unwrap(),
        (vec![2, 2], vec![0x7654_3210, 0xfedc_ba98, 0, u32::MAX])
    );
    // A typed read never reinterprets another dtype's bytes.
    assert!(st.read_u8("i8").unwrap_err().contains("expected U8, found I8"));
    assert!(st.read_i8("u8").unwrap_err().contains("expected I8, found U8"));
    assert!(st.read_u32("i32").unwrap_err().contains("expected U32, found I32"));
    assert!(st.read_f32("u32").unwrap_err().contains("cannot read U32"));
    assert!(st.read_raw("missing").unwrap_err().contains("no tensor"));
}

#[test]
fn fp8_and_complex_tensors_open_and_read_raw() {
    let header = concat!(
        r#"{"e4m3":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,2]},"#,
        r#""e4m3fnuz":{"dtype":"F8_E4M3FNUZ","shape":[1],"data_offsets":[2,3]},"#,
        r#""e5m2":{"dtype":"F8_E5M2","shape":[3],"data_offsets":[3,6]},"#,
        r#""e5m2fnuz":{"dtype":"F8_E5M2FNUZ","shape":[1],"data_offsets":[6,7]},"#,
        r#""e8m0":{"dtype":"F8_E8M0","shape":[1],"data_offsets":[7,8]},"#,
        r#""c64":{"dtype":"C64","shape":[1],"data_offsets":[8,16]}}"#
    );
    let mut data = vec![0x38, 0xb8, 0x80, 0x3c, 0x7b, 0xfc, 0x01, 0x7f];
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&(-2.0f32).to_le_bytes());
    let p = file("fp8", header, &data);
    let st = SafeTensors::open(&p).unwrap();
    std::fs::remove_file(&p).unwrap();
    let want: &[(&str, Dtype, &[u8])] = &[
        ("e4m3", Dtype::F8E4M3, &data[0..2]),
        ("e4m3fnuz", Dtype::F8E4M3Fnuz, &data[2..3]),
        ("e5m2", Dtype::F8E5M2, &data[3..6]),
        ("e5m2fnuz", Dtype::F8E5M2Fnuz, &data[6..7]),
        ("e8m0", Dtype::F8E8M0, &data[7..8]),
        ("c64", Dtype::C64, &data[8..16]),
    ];
    for &(name, dtype, bytes) in want {
        let (info, raw) = st.read_raw(name).unwrap();
        assert_eq!((info.dtype, raw.as_slice()), (dtype, bytes), "{name}");
        assert_eq!(info.dtype.size() * info.numel() as u64, bytes.len() as u64, "{name}");
    }
    // Not widened by a reader that does not know the format.
    assert!(st.read_f32("e4m3").unwrap_err().contains("cannot read F8E4M3"));
    // The sub-byte types stay refused: their sizes are not whole bytes.
    for tag in ["F4", "F6_E2M3", "F6_E3M2"] {
        let h = format!(r#"{{"a":{{"dtype":"{tag}","shape":[2],"data_offsets":[0,1]}}}}"#);
        expect_rejected(tag, &h, &[0], "unsupported dtype");
    }
}

/// `read_f32` widens a 16-bit tensor inside its own result; a long tensor of
/// every bit pattern checks that no widened value overwrote an unread one.
#[test]
fn sixteen_bit_tensors_widen_exactly_across_every_bit_pattern() {
    let n = 1 << 16;
    let mut data = Vec::with_capacity(4 * n);
    for bits in 0..=u16::MAX {
        data.extend_from_slice(&bits.to_le_bytes());
    }
    for bits in (0..=u16::MAX).rev() {
        data.extend_from_slice(&bits.to_le_bytes());
    }
    let header = format!(
        r#"{{"b":{{"dtype":"BF16","shape":[{n}],"data_offsets":[0,{}]}},"h":{{"dtype":"F16","shape":[256,256],"data_offsets":[{},{}]}}}}"#,
        2 * n,
        2 * n,
        4 * n
    );
    let p = file("widen", &header, &data);
    let st = SafeTensors::open(&p).unwrap();
    std::fs::remove_file(&p).unwrap();
    let (shape, b) = st.read_f32("b").unwrap();
    assert_eq!(shape, vec![n]);
    let (shape, h) = st.read_f32("h").unwrap();
    assert_eq!(shape, vec![256, 256]);
    for (i, bits) in (0..=u16::MAX).enumerate() {
        let want = tessl::tensor::bf16_bits_to_f32(bits);
        assert_eq!(b[i].to_bits(), want.to_bits(), "bf16 {bits:#06x}");
        let rbits = u16::MAX - bits;
        let want = tessl::tensor::f16_bits_to_f32(rbits);
        assert_eq!(h[i].to_bits(), want.to_bits(), "f16 {rbits:#06x}");
    }
    assert_eq!(st.read_bf16_bits("b").unwrap().1, (0..=u16::MAX).collect::<Vec<_>>());
}

#[test]
fn unicode_escapes_and_surrogate_pairs_decode() {
    // Built from parts so each escape stays an escape in the file.
    let bs = '\\';
    let header = format!(
        "{{\"__metadata__\":{{\"n\":\"{bs}u00e9{bs}ud83d{bs}ude00{bs}u0041{bs}n{bs}t\"}},\
         \"{bs}u0061\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}}}"
    );
    assert!(header.contains("\\ud83d\\ude00"), "{header}");
    let p = file("esc", &header, &2.0f32.to_le_bytes());
    let st = SafeTensors::open(&p).unwrap();
    std::fs::remove_file(&p).unwrap();
    assert_eq!(st.metadata()["n"], "\u{e9}\u{1f600}A\n\t");
    // "a" is the key "a".
    assert_eq!(st.read_f32("a").unwrap().1, vec![2.0]);
}

#[test]
fn header_length_is_bounded_by_the_file_and_the_cap() {
    // Shorter than the length field itself.
    let p = tmp("short");
    std::fs::write(&p, [1u8, 0, 0]).unwrap();
    assert!(open_err(p).contains("header length"));

    let h = ONE_F32.as_bytes();
    let data = 1.0f32.to_le_bytes();
    assert!(open_err(raw("zero", 0, h, &data)).contains("outside 1..="));
    assert!(open_err(raw("cap", MAX_HEADER_BYTES + 1, h, &data)).contains("outside 1..="));
    assert!(open_err(raw("huge", u64::MAX, h, &data)).contains("outside 1..="));
    let past = h.len() as u64 + data.len() as u64 + 1;
    assert!(open_err(raw("past", past, h, &data)).contains("past the end"));
    // One byte short of the header: the JSON is cut and fails to parse.
    assert!(open_err(raw("cut", h.len() as u64 - 1, h, &data)).contains("header JSON"));
}

#[test]
fn json_outside_the_formats_subset_is_rejected() {
    let d = 1.0f32.to_le_bytes();
    let cases: &[(&str, &str, &str)] = &[
        ("not_obj", r#"[1]"#, "not a JSON object"),
        ("trailing", &format!("{ONE_F32}x"), "trailing bytes"),
        (
            "float",
            r#"{"a":{"dtype":"F32","shape":[1.0],"data_offsets":[0,4]}}"#,
            "non-integer",
        ),
        (
            "exp",
            r#"{"a":{"dtype":"F32","shape":[1e0],"data_offsets":[0,4]}}"#,
            "non-integer",
        ),
        (
            "neg",
            r#"{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#,
            "unsupported JSON value",
        ),
        (
            "lead0",
            r#"{"a":{"dtype":"F32","shape":[01],"data_offsets":[0,4]}}"#,
            "leading zero",
        ),
        ("null", r#"{"a":null}"#, "unsupported JSON value"),
        ("bool", r#"{"a":true}"#, "unsupported JSON value"),
        (
            "u64",
            r#"{"a":{"dtype":"F32","shape":[18446744073709551616],"data_offsets":[0,4]}}"#,
            "overflows u64",
        ),
        (
            "dup",
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            "duplicate key",
        ),
        (
            "dup_field",
            r#"{"a":{"dtype":"F32","dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            "duplicate key",
        ),
        (
            "deep",
            r#"{"a":{"dtype":"F32","shape":[[1]],"data_offsets":[0,4]}}"#,
            "nesting deeper",
        ),
        ("unterminated", r#"{"a":{"dtype":"F32"#, "unterminated string"),
        (
            "bad_escape",
            r#"{"a\q":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            "invalid escape",
        ),
        (
            "plus_hex",
            r#"{"a\u+041":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            "bad \\u escape",
        ),
        (
            "lone_hi",
            r#"{"a\ud800":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            "unpaired high surrogate",
        ),
        (
            "lone_lo",
            r#"{"a\udc00":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            "unpaired low surrogate",
        ),
        (
            "hi_then_char",
            r#"{"a\ud800A":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            "unpaired high surrogate",
        ),
        (
            "ctrl",
            "{\"a\tb\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}",
            "control character",
        ),
        ("no_colon", r#"{"a" {"dtype":"F32"}}"#, "expected ':'"),
        (
            "no_comma",
            r#"{"a":{"dtype":"F32" "shape":[1]}}"#,
            "expected ',' or '}'",
        ),
        (
            "arr_comma",
            r#"{"a":{"dtype":"F32","shape":[1 2],"data_offsets":[0,4]}}"#,
            "expected ',' or ']'",
        ),
        ("key", r#"{1:2}"#, "expected a string key"),
    ];
    for (tag, h, needle) in cases {
        expect_rejected(tag, h, &d, needle);
    }
    // A high surrogate followed by an escape that is not a low surrogate. Built
    // from parts so the second escape stays a six-byte escape in the source.
    let bs = '\\';
    let bad_lo = format!("{{\"a{bs}ud800{bs}u0041\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}}}");
    assert!(bad_lo.contains("\\u0041"), "{bad_lo}");
    expect_rejected("bad_lo", &bad_lo, &d, "invalid low surrogate");
    // Invalid UTF-8 in the header.
    let bad = b"{\"\xff\":1}";
    assert!(open_err(raw("utf8", bad.len() as u64, bad, &[])).contains("not UTF-8"));
}

#[test]
fn entries_must_be_exactly_dtype_shape_and_offsets() {
    let d = 1.0f32.to_le_bytes();
    let cases: &[(&str, &str, &str)] = &[
        (
            "unknown_field",
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"x":"y"}}"#,
            "unexpected or mistyped field \"x\"",
        ),
        (
            "missing",
            r#"{"a":{"dtype":"F32","shape":[1]}}"#,
            "needs dtype, shape and data_offsets",
        ),
        (
            "dtype_type",
            r#"{"a":{"dtype":4,"shape":[1],"data_offsets":[0,4]}}"#,
            "mistyped field \"dtype\"",
        ),
        (
            "shape_type",
            r#"{"a":{"dtype":"F32","shape":"1","data_offsets":[0,4]}}"#,
            "mistyped field \"shape\"",
        ),
        (
            "shape_str",
            r#"{"a":{"dtype":"F32","shape":["1"],"data_offsets":[0,4]}}"#,
            "non-integer",
        ),
        (
            "offs3",
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4,4]}}"#,
            "not [begin, end]",
        ),
        (
            "dtype_name",
            r#"{"a":{"dtype":"F8","shape":[1],"data_offsets":[0,4]}}"#,
            "unsupported dtype",
        ),
        ("entry_type", r#"{"a":"F32"}"#, "not an object"),
        ("meta_type", r#"{"__metadata__":[1]}"#, "__metadata__ is not an object"),
        ("meta_value", r#"{"__metadata__":{"k":1}}"#, "not a string"),
    ];
    for (tag, h, needle) in cases {
        expect_rejected(tag, h, &d, needle);
    }
}

#[test]
fn offsets_must_match_the_shape_and_tile_the_data_exactly() {
    let d8 = [0u8; 8];
    let cases: &[(&str, &str, &[u8], &str)] = &[
        (
            "reversed",
            r#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[4,0]}}"#,
            &d8[..4],
            "outside",
        ),
        (
            "past",
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            &d8[..4],
            "outside",
        ),
        (
            "size",
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
            &d8[..4],
            "shape [2] x F32 is 8",
        ),
        (
            "overlap",
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}"#,
            &d8[..6],
            "no gap or overlap",
        ),
        (
            "gap",
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[2],"data_offsets":[6,8]}}"#,
            &d8,
            "no gap or overlap",
        ),
        ("unindexed", ONE_F32, &d8, "cover 4 of 8"),
        (
            "numel",
            r#"{"a":{"dtype":"U8","shape":[4294967296,4294967296],"data_offsets":[0,0]}}"#,
            &[],
            "element count overflows",
        ),
        (
            "bytes",
            r#"{"a":{"dtype":"F64","shape":[2305843009213693952,2],"data_offsets":[0,0]}}"#,
            &[],
            "overflows",
        ),
    ];
    for (tag, h, d, needle) in cases {
        expect_rejected(tag, h, d, needle);
    }
}

#[test]
fn reads_come_from_the_opened_file_even_if_the_path_is_replaced() {
    let p = file("swap", ONE_F32, &3.0f32.to_le_bytes());
    let st = SafeTensors::open(&p).unwrap();
    // A different file now sits at the path (new inode, other bytes and a
    // longer header, so the old offsets would land on the wrong bytes).
    let other = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}          "#;
    let q = file("swap_new", other, &9.0f32.to_le_bytes());
    std::fs::rename(&q, &p).unwrap();
    let got = st.read_f32("a");
    std::fs::remove_file(&p).unwrap();
    assert_eq!(got.unwrap().1, vec![3.0]);
}

#[test]
fn a_file_truncated_after_open_fails_the_read_not_silently() {
    let p = file("trunc", ONE_F32, &1.0f32.to_le_bytes());
    let st = SafeTensors::open(&p).unwrap();
    let len = std::fs::metadata(&p).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&p)
        .unwrap()
        .set_len(len - 2)
        .unwrap();
    let e = st.read_f32("a").unwrap_err();
    std::fs::remove_file(&p).unwrap();
    assert!(e.contains("data"), "{e}");
}

/// The real checkpoint. Opt-in because it needs the ~4.5 GB file from the
/// Hugging Face cache (it reads only the header and two small tensors):
///
/// ```text
/// QWEN35_2B_SAFETENSORS=/path/to/model.safetensors \
///   cargo test --release --test safetensors -- --ignored
/// ```
///
/// With the variable unset the test fails rather than passing unexamined.
#[test]
#[ignore]
fn opens_the_qwen35_2b_checkpoint() {
    let path = std::env::var("QWEN35_2B_SAFETENSORS")
        .expect("set QWEN35_2B_SAFETENSORS to the checkpoint's .safetensors file");
    let st = SafeTensors::open(std::path::Path::new(&path)).unwrap();
    let n = st.names().count();
    assert_eq!(n, 632, "tensor count");
    let (shape, bits) = st.read_bf16_bits("model.language_model.norm.weight").unwrap();
    assert_eq!(shape, vec![2048]);
    assert!(bits.iter().any(|&b| b != 0));
    let (shape, a_log) = st.read_f32("model.language_model.layers.0.linear_attn.A_log").unwrap();
    assert_eq!(shape, vec![16]);
    assert!(a_log.iter().all(|v| v.is_finite()));
}
