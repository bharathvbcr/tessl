//! The C ABI's multimodal EmbeddingGemma 2 entries
//! (`tessl_embedgemma2_load_multimodal`, `tessl_embedgemma2_encode_images`,
//! `tessl_embedgemma2_encode_audio`), called the way a foreign caller calls
//! them, on the tiny multimodal checkpoint of `tests/common/embedgemma2_mm.rs`
//! written to a directory under `target/`.
//!
//! What this file pins is the boundary, as `tests/capi_embedgemma2.rs` does for
//! the text entries: every refusal is a status and a message that leaves the
//! caller's output untouched and the handle usable, each length is bounded
//! before it sizes a slice (the over-limit cases pass buffers far smaller than
//! the length they declare, which would fault if anything read them first),
//! and a successful call is `EmbedGemma2Multimodal::encode` bit for bit on the
//! same weights. The real checkpoint is `python/tests/test_embedgemma2_mm.py`.
//!
//! The tests with a null handle or a null runtime need no device (they are in
//! the hosted selection, `scripts/hosted_tests.txt`); the rest create a runtime.

mod common;

use std::ffi::{c_char, CStr};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::OnceLock;

use common::embedgemma2_mm::{checkpoint_bytes, clip, config_json, config_json_without_ids, configs, Image};
use common::gpu_lock;
use tessl::capi::{
    tessl_runtime_free, tessl_runtime_new, TesslRuntime, TESSL_ERR, TESSL_OK, TESSL_OPERANDS_BF16,
    TESSL_OPERANDS_EXACT_F32,
};
use tessl::capi_embedgemma2::{
    tessl_embedgemma2_encode, tessl_embedgemma2_encode_audio, tessl_embedgemma2_encode_images, tessl_embedgemma2_free,
    tessl_embedgemma2_load, tessl_embedgemma2_load_multimodal, test_embedgemma2_multimodal, TesslEmbedGemma2,
};
use tessl::embedgemma2_audio::{AudioInput, MAX_CLIPS, MAX_FRAMES_PER_FORWARD};
use tessl::embedgemma2_mm::MmItem;
use tessl::embedgemma2_vision::{MAX_BATCH_PATCHES, MAX_IMAGES};
use tessl::gemm::GemmOperands;

const ERR_LEN: usize = 512;
/// The tiny checkpoint's `embedding_dim`.
const DIM: usize = 96;
/// Written to the output of every call that must leave it alone.
const SENTINEL: f32 = -7.25;

fn msg(err: &[c_char]) -> String {
    unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

// ---------------------------------------------------------------------------
// Raw calls: every argument is the caller's.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn images_raw(
    model: *mut TesslEmbedGemma2,
    pixels: *const f32,
    pixels_len: u64,
    positions: *const i32,
    positions_len: u64,
    offsets: *const u64,
    n: u64,
    trunc: u32,
    out: *mut f32,
    out_len: u64,
) -> (i32, String) {
    let mut err = [0 as c_char; ERR_LEN];
    let s = unsafe {
        tessl_embedgemma2_encode_images(
            model,
            pixels,
            pixels_len,
            positions,
            positions_len,
            offsets,
            n,
            trunc,
            out,
            out_len,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    (s, msg(&err))
}

#[allow(clippy::too_many_arguments)]
fn audio_raw(
    model: *mut TesslEmbedGemma2,
    feats: *const f32,
    feats_len: u64,
    mask: *const u8,
    mask_len: u64,
    offsets: *const u64,
    n: u64,
    trunc: u32,
    out: *mut f32,
    out_len: u64,
) -> (i32, String) {
    let mut err = [0 as c_char; ERR_LEN];
    let s = unsafe {
        tessl_embedgemma2_encode_audio(
            model,
            feats,
            feats_len,
            mask,
            mask_len,
            offsets,
            n,
            trunc,
            out,
            out_len,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    (s, msg(&err))
}

/// Images laid end to end, as the C entry takes them.
struct FlatImages {
    pixels: Vec<f32>,
    positions: Vec<i32>,
    offsets: Vec<u64>,
}

fn flat_images(images: &[&Image]) -> FlatImages {
    let mut f = FlatImages {
        pixels: Vec::new(),
        positions: Vec::new(),
        offsets: vec![0],
    };
    for im in images {
        f.pixels.extend_from_slice(&im.pixel_values);
        f.positions.extend_from_slice(&im.position_ids);
        f.offsets
            .push(f.offsets.last().unwrap() + (im.position_ids.len() / 2) as u64);
    }
    f
}

impl FlatImages {
    fn encode(&self, model: *mut TesslEmbedGemma2, trunc: u32) -> (i32, String, Vec<f32>) {
        let dim = if trunc == 0 { DIM } else { trunc as usize };
        let n = self.offsets.len() - 1;
        let mut out = vec![SENTINEL; n * dim];
        let (s, m) = images_raw(
            model,
            self.pixels.as_ptr(),
            self.pixels.len() as u64,
            self.positions.as_ptr(),
            self.positions.len() as u64,
            self.offsets.as_ptr(),
            n as u64,
            trunc,
            out.as_mut_ptr(),
            out.len() as u64,
        );
        (s, m, out)
    }
}

/// Clips laid end to end.
struct FlatClips {
    feats: Vec<f32>,
    mask: Vec<u8>,
    offsets: Vec<u64>,
}

fn flat_clips(clips: &[&AudioInput]) -> FlatClips {
    let mut f = FlatClips {
        feats: Vec::new(),
        mask: Vec::new(),
        offsets: vec![0],
    };
    for c in clips {
        f.feats.extend_from_slice(&c.input_features);
        f.mask.extend(c.mask.iter().map(|&m| u8::from(m)));
        f.offsets.push(f.offsets.last().unwrap() + c.mask.len() as u64);
    }
    f
}

impl FlatClips {
    fn encode(&self, model: *mut TesslEmbedGemma2, trunc: u32) -> (i32, String, Vec<f32>) {
        let dim = if trunc == 0 { DIM } else { trunc as usize };
        let n = self.offsets.len() - 1;
        let mut out = vec![SENTINEL; n * dim];
        let (s, m) = audio_raw(
            model,
            self.feats.as_ptr(),
            self.feats.len() as u64,
            self.mask.as_ptr(),
            self.mask.len() as u64,
            self.offsets.as_ptr(),
            n as u64,
            trunc,
            out.as_mut_ptr(),
            out.len() as u64,
        );
        (s, m, out)
    }
}

// ---------------------------------------------------------------------------
// No device: a null handle, a null runtime.
// ---------------------------------------------------------------------------

#[test]
fn a_null_handle_is_refused_by_the_multimodal_entries() {
    let null: *mut TesslEmbedGemma2 = ptr::null_mut();
    let mut out = [SENTINEL; DIM];
    let (px, pos, feats, mask, off) = ([0f32; 4], [0i32; 2], [0f32; 4], [1u8; 2], [0u64, 1]);
    let results = [
        (
            "images",
            images_raw(
                null,
                px.as_ptr(),
                4,
                pos.as_ptr(),
                2,
                off.as_ptr(),
                1,
                0,
                out.as_mut_ptr(),
                DIM as u64,
            ),
        ),
        (
            "audio",
            audio_raw(
                null,
                feats.as_ptr(),
                4,
                mask.as_ptr(),
                2,
                off.as_ptr(),
                1,
                0,
                out.as_mut_ptr(),
                DIM as u64,
            ),
        ),
    ];
    for (name, (status, message)) in results {
        assert_eq!(status, TESSL_ERR, "{name}");
        assert!(message.contains("null embedgemma2 model handle"), "{name}: {message}");
    }
    assert!(out.iter().all(|&v| v == SENTINEL), "a refused call wrote its output");
}

#[test]
fn loading_multimodal_on_a_null_runtime_is_refused_and_leaves_out_null() {
    let dir = b"/nonexistent";
    let mut err = [0 as c_char; ERR_LEN];
    let mut out = 0x10 as *mut TesslEmbedGemma2;
    let s = unsafe {
        tessl_embedgemma2_load_multimodal(
            ptr::null_mut(),
            dir.as_ptr().cast(),
            dir.len() as u64,
            TESSL_OPERANDS_EXACT_F32,
            &mut out,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    assert_eq!(s, TESSL_ERR);
    assert!(out.is_null(), "*out is non-null after a refused load");
    assert!(msg(&err).contains("null runtime handle"), "{}", msg(&err));
}

// ---------------------------------------------------------------------------
// The tiny checkpoint on disk.
// ---------------------------------------------------------------------------

/// A snapshot directory `name` under `target/`, written once per process. The
/// text projections are stored bf16, which both `operands` codes load.
fn snapshot(name: &'static str, config: String) -> &'static Path {
    static DIRS: OnceLock<std::sync::Mutex<Vec<(&'static str, &'static PathBuf)>>> = OnceLock::new();
    let dirs = DIRS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let mut dirs = dirs.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(&(_, p)) = dirs.iter().find(|(n, _)| *n == name) {
        return p.as_path();
    }
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target").join(name);
    std::fs::create_dir_all(&dir).expect("create the snapshot directory");
    std::fs::write(dir.join("config.json"), config).expect("write config.json");
    std::fs::write(
        dir.join("model.safetensors"),
        checkpoint_bytes(&configs(), 2026, GemmOperands::Bf16),
    )
    .expect("write model.safetensors");
    let dir: &'static PathBuf = Box::leak(Box::new(dir));
    dirs.push((name, dir));
    dir.as_path()
}

fn mm_snapshot() -> &'static Path {
    snapshot("capi_embedgemma2_mm_tiny", config_json())
}

fn load_mm_raw(rt: *mut TesslRuntime, dir: &[u8], operands: u32) -> (i32, String, *mut TesslEmbedGemma2) {
    let mut err = [0 as c_char; ERR_LEN];
    let mut out = 0x10 as *mut TesslEmbedGemma2;
    let s = unsafe {
        tessl_embedgemma2_load_multimodal(
            rt,
            dir.as_ptr().cast(),
            dir.len() as u64,
            operands,
            &mut out,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    (s, msg(&err), out)
}

/// A runtime handle and a model loaded on it, both freed on drop.
struct Loaded {
    rt: *mut TesslRuntime,
    model: *mut TesslEmbedGemma2,
}

impl Loaded {
    fn new(operands: u32) -> Self {
        let mut err = [0 as c_char; ERR_LEN];
        let rt = unsafe { tessl_runtime_new(err.as_mut_ptr(), ERR_LEN) };
        assert!(!rt.is_null(), "tessl_runtime_new: {}", msg(&err));
        let (s, m, model) = load_mm_raw(rt, mm_snapshot().to_str().unwrap().as_bytes(), operands);
        assert_eq!(s, TESSL_OK, "tessl_embedgemma2_load_multimodal: {m}");
        assert!(!model.is_null());
        Self { rt, model }
    }
}

impl Drop for Loaded {
    fn drop(&mut self) {
        assert_eq!(unsafe { tessl_embedgemma2_free(self.model) }, TESSL_OK);
        assert_eq!(unsafe { tessl_runtime_free(self.rt) }, TESSL_OK);
    }
}

/// The inputs the tests share.
struct Inputs {
    images: Vec<Image>,
    clips: Vec<AudioInput>,
}

fn inputs() -> Inputs {
    let c = configs();
    let pd = c.vision.patch_dim() as usize;
    Inputs {
        images: vec![
            Image::new(8, 6, 16, pd, 1),
            Image::new(5, 4, 28, pd, 2),
            Image::new(1, 1, 3, pd, 3),
            Image::new(2, 2, 0, pd, 4),
        ],
        clips: vec![
            clip(&c.audio, 41, 1, |_| true),
            clip(&c.audio, 58, 2, |f| f < 45),
            clip(&c.audio, 30, 3, |f| !(8..16).contains(&f)),
            clip(&c.audio, 1, 4, |_| true),
        ],
    }
}

#[test]
fn images_and_audio_through_the_c_entries_are_the_facade_bit_for_bit() {
    let _g = gpu_lock();
    for operands in [TESSL_OPERANDS_EXACT_F32, TESSL_OPERANDS_BF16] {
        let l = Loaded::new(operands);
        let mm = unsafe { test_embedgemma2_multimodal(l.model) }.expect("a multimodal handle");
        assert_eq!(
            mm.text().operands(),
            if operands == TESSL_OPERANDS_BF16 {
                GemmOperands::Bf16
            } else {
                GemmOperands::ExactF32
            },
            "the text encoder's operands are the load's"
        );
        let inp = inputs();
        let all_images: Vec<&Image> = inp.images.iter().collect();
        let all_clips: Vec<&AudioInput> = inp.clips.iter().collect();
        for trunc in [0u32, 40, 1] {
            let t = (trunc != 0).then_some(trunc);
            let (s, m, got) = flat_images(&all_images).encode(l.model, trunc);
            assert_eq!(s, TESSL_OK, "{operands} images, truncate_dim {trunc}: {m}");
            let items: Vec<MmItem<'_>> = inp.images.iter().map(|i| MmItem::Image(i.input())).collect();
            let want = mm.encode(&items, t).unwrap();
            assert_eq!(
                bits(&got),
                bits(&want.embeddings),
                "{operands} images, truncate_dim {trunc}"
            );

            let (s, m, got) = flat_clips(&all_clips).encode(l.model, trunc);
            assert_eq!(s, TESSL_OK, "{operands} audio, truncate_dim {trunc}: {m}");
            let items: Vec<MmItem<'_>> = inp.clips.iter().map(MmItem::Audio).collect();
            let want = mm.encode(&items, t).unwrap();
            assert_eq!(
                bits(&got),
                bits(&want.embeddings),
                "{operands} audio, truncate_dim {trunc}"
            );
            assert!(got.iter().all(|v| v.is_finite()));
        }
        // One image and one clip alone; the unit-norm contract through the boundary.
        let (s, _, one) = flat_images(&all_images[2..3]).encode(l.model, 0);
        assert_eq!(s, TESSL_OK);
        assert_eq!(
            bits(&one),
            bits(
                &mm.encode(&[MmItem::Image(inp.images[2].input())], None)
                    .unwrap()
                    .embeddings
            )
        );
        let (s, _, one) = flat_clips(&all_clips[1..2]).encode(l.model, 0);
        assert_eq!(s, TESSL_OK);
        let norm = one.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm {norm}");
        // The same handle still embeds text, through the text entry.
        let ids = [2u32, 5, 7, 1];
        let (offsets, mut out) = ([0u64, 4], vec![SENTINEL; DIM]);
        let mut err = [0 as c_char; ERR_LEN];
        let s = unsafe {
            tessl_embedgemma2_encode(
                l.model,
                ids.as_ptr(),
                4,
                offsets.as_ptr(),
                1,
                0,
                out.as_mut_ptr(),
                DIM as u64,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(s, TESSL_OK, "{}", msg(&err));
        let want = mm.encode(&[MmItem::Text(&ids)], None).unwrap();
        assert_eq!(bits(&out), bits(&want.embeddings));
    }
}

#[test]
fn image_refusals_leave_the_output_alone_and_the_handle_usable() {
    let _g = gpu_lock();
    let l = Loaded::new(TESSL_OPERANDS_EXACT_F32);
    let inp = inputs();
    let imgs: Vec<&Image> = inp.images.iter().collect();
    let good = flat_images(&imgs[..2]);
    let (s, m, want) = good.encode(l.model, 0);
    assert_eq!(s, TESSL_OK, "{m}");
    let pd = configs().vision.patch_dim() as u64;

    let (px, pos, off) = (good.pixels.as_ptr(), good.positions.as_ptr(), good.offsets.as_ptr());
    let (px_len, pos_len) = (good.pixels.len() as u64, good.positions.len() as u64);
    let out_len = (2 * DIM) as u64;
    // Buffers far smaller than the lengths declared below: a refusal that read
    // them first would fault.
    let tiny_px = [0f32; 4];
    let tiny_pos = [0i32; 2];
    let big_off = [0u64, MAX_BATCH_PATCHES as u64 + 1];
    #[allow(clippy::type_complexity)]
    let run = |what: &str, args: (*const f32, u64, *const i32, u64, *const u64, u64, u32, u64), want_msg: &str| {
        let (p, pl, q, ql, o, n, trunc, olen) = args;
        let mut out = vec![SENTINEL; 2 * DIM];
        let (s, m) = images_raw(
            l.model,
            p,
            pl,
            q,
            ql,
            o,
            n,
            trunc,
            out.as_mut_ptr(),
            olen.min(out.len() as u64),
        );
        assert_eq!(s, TESSL_ERR, "{what}: accepted");
        assert!(m.contains(want_msg), "{what}: {m:?} does not mention {want_msg:?}");
        assert!(out.iter().all(|&v| v == SENTINEL), "{what}: wrote the output");
    };
    run(
        "zero images",
        (px, px_len, pos, pos_len, off, 0, 0, out_len),
        "n_images is 0",
    );
    run(
        "too many images",
        (px, px_len, pos, pos_len, off, MAX_IMAGES as u64 + 1, 0, out_len),
        "n_images is 257",
    );
    run(
        "an overflowing count",
        (px, px_len, pos, pos_len, off, u64::MAX, 0, out_len),
        "n_images",
    );
    run(
        "truncate_dim past the width",
        (px, px_len, pos, pos_len, off, 2, DIM as u32 + 1, out_len),
        "truncate_dim",
    );
    run(
        "out_len off by one",
        (px, px_len, pos, pos_len, off, 2, 0, out_len - 1),
        "out_len is",
    );
    run(
        "null pixel_values",
        (ptr::null(), px_len, pos, pos_len, off, 2, 0, out_len),
        "null pixel_values",
    );
    run(
        "null position_ids",
        (px, px_len, ptr::null(), pos_len, off, 2, 0, out_len),
        "null pixel_values",
    );
    run(
        "null offsets",
        (px, px_len, pos, pos_len, ptr::null(), 2, 0, out_len),
        "null pixel_values",
    );
    run(
        "an image past the patch limit, with buffers it would overrun",
        (
            tiny_px.as_ptr(),
            4,
            tiny_pos.as_ptr(),
            2,
            big_off.as_ptr(),
            1,
            0,
            DIM as u64,
        ),
        "image 0 has 16385 patch rows; the limit is 16384",
    );
    run(
        "pixel_values_len short by one",
        (px, px_len - 1, pos, pos_len, off, 2, 0, out_len),
        "pixel_values_len is",
    );
    run(
        "position_ids_len long by one",
        (px, px_len, pos, pos_len + 1, off, 2, 0, out_len),
        "position_ids_len is",
    );
    // Offsets: not starting at 0, not increasing, an empty image.
    for (what, offs, want_msg) in [
        (
            "offsets not from zero",
            vec![1u64, good.offsets[1], good.offsets[2]],
            "offsets[0] is 1",
        ),
        (
            "offsets that do not increase",
            vec![0u64, good.offsets[2], good.offsets[1]],
            "does not exceed",
        ),
        ("an empty image", vec![0u64, 0, good.offsets[2]], "does not exceed"),
    ] {
        run(what, (px, px_len, pos, pos_len, offs.as_ptr(), 2, 0, out_len), want_msg);
    }

    // The towers' own refusals, through the boundary.
    let mut nan = flat_images(&imgs[..1]);
    nan.pixels[3] = f32::NAN;
    let (s, m, o) = nan.encode(l.model, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("pixel value 3") && m.contains("NaN"), "{m}");
    assert!(o.iter().all(|&v| v == SENTINEL));
    let mut far = flat_images(&imgs[3..4]);
    far.positions[2] = 999;
    let (s, m, o) = far.encode(l.model, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("outside the position table"), "{m}");
    assert!(o.iter().all(|&v| v == SENTINEL));
    // Padding that is not a suffix.
    let mut gap = flat_images(&imgs[..1]);
    gap.positions[2] = -1;
    gap.positions[3] = -1;
    let (s, m, _) = gap.encode(l.model, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("padding must be a suffix"), "{m}");
    // Rows past the patch width: the arrays do not hold what the offsets say.
    let mut shape = flat_images(&imgs[..1]);
    shape.pixels.truncate(shape.pixels.len() - pd as usize);
    let (s, m, _) = shape.encode(l.model, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("pixel_values_len"), "{m}");

    // The handle works, and gives what it gave.
    let (s, _, again) = good.encode(l.model, 0);
    assert_eq!(s, TESSL_OK);
    assert_eq!(bits(&again), bits(&want));
}

#[test]
fn audio_refusals_leave_the_output_alone_and_the_handle_usable() {
    let _g = gpu_lock();
    let l = Loaded::new(TESSL_OPERANDS_EXACT_F32);
    let inp = inputs();
    let clips: Vec<&AudioInput> = inp.clips.iter().collect();
    let good = flat_clips(&clips[..2]);
    let (s, m, want) = good.encode(l.model, 0);
    assert_eq!(s, TESSL_OK, "{m}");
    let mels = configs().audio.mel_bins() as u64;

    let (f, k, off) = (good.feats.as_ptr(), good.mask.as_ptr(), good.offsets.as_ptr());
    let (f_len, k_len) = (good.feats.len() as u64, good.mask.len() as u64);
    let out_len = (2 * DIM) as u64;
    let tiny_f = [0f32; 4];
    let tiny_k = [1u8; 2];
    let big_off = [0u64, MAX_FRAMES_PER_FORWARD as u64 + 1];
    #[allow(clippy::type_complexity)]
    let run = |what: &str, args: (*const f32, u64, *const u8, u64, *const u64, u64, u32, u64), want_msg: &str| {
        let (p, pl, q, ql, o, n, trunc, olen) = args;
        let mut out = vec![SENTINEL; 2 * DIM];
        let (s, m) = audio_raw(
            l.model,
            p,
            pl,
            q,
            ql,
            o,
            n,
            trunc,
            out.as_mut_ptr(),
            olen.min(out.len() as u64),
        );
        assert_eq!(s, TESSL_ERR, "{what}: accepted");
        assert!(m.contains(want_msg), "{what}: {m:?} does not mention {want_msg:?}");
        assert!(out.iter().all(|&v| v == SENTINEL), "{what}: wrote the output");
    };
    run("zero clips", (f, f_len, k, k_len, off, 0, 0, out_len), "n_clips is 0");
    run(
        "too many clips",
        (f, f_len, k, k_len, off, MAX_CLIPS as u64 + 1, 0, out_len),
        "n_clips is 257",
    );
    run(
        "an overflowing count",
        (f, f_len, k, k_len, off, u64::MAX, 0, out_len),
        "n_clips",
    );
    run(
        "truncate_dim past the width",
        (f, f_len, k, k_len, off, 2, DIM as u32 + 1, out_len),
        "truncate_dim",
    );
    run(
        "out_len off by one",
        (f, f_len, k, k_len, off, 2, 0, out_len - 1),
        "out_len is",
    );
    run(
        "null features",
        (ptr::null(), f_len, k, k_len, off, 2, 0, out_len),
        "null input_features",
    );
    run(
        "null mask",
        (f, f_len, ptr::null(), k_len, off, 2, 0, out_len),
        "null input_features",
    );
    run(
        "null offsets",
        (f, f_len, k, k_len, ptr::null(), 2, 0, out_len),
        "null input_features",
    );
    run(
        "a clip past the frame limit, with buffers it would overrun",
        (
            tiny_f.as_ptr(),
            4,
            tiny_k.as_ptr(),
            2,
            big_off.as_ptr(),
            1,
            0,
            DIM as u64,
        ),
        "clip 0 has 16385 frames; the limit is 16384",
    );
    run(
        "features short by one",
        (f, f_len - 1, k, k_len, off, 2, 0, out_len),
        "input_features_len is",
    );
    run(
        "mask long by one",
        (f, f_len, k, k_len + 1, off, 2, 0, out_len),
        "mask_len is",
    );
    for (what, offs, want_msg) in [
        (
            "offsets not from zero",
            vec![2u64, good.offsets[1], good.offsets[2]],
            "offsets[0] is 2",
        ),
        (
            "offsets that do not increase",
            vec![0u64, good.offsets[1], good.offsets[1]],
            "does not exceed",
        ),
    ] {
        run(what, (f, f_len, k, k_len, offs.as_ptr(), 2, 0, out_len), want_msg);
    }

    // A mask byte that is not 0 or 1 is not a bool.
    let mut bad = flat_clips(&clips[..1]);
    bad.mask[4] = 2;
    let (s, m, o) = bad.encode(l.model, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("input_features_mask[4] is 2"), "{m}");
    assert!(o.iter().all(|&v| v == SENTINEL));
    // The tower's own refusals.
    let mut nan = flat_clips(&clips[..1]);
    nan.feats[9] = f32::INFINITY;
    let (s, m, _) = nan.encode(l.model, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("input_features[9]") && m.contains("not finite"), "{m}");
    let silent = AudioInput {
        input_features: vec![0.5; 3 * mels as usize],
        mask: vec![false; 3],
    };
    let (s, m, o) = flat_clips(&[&silent]).encode(l.model, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("n_soft is zero"), "{m}");
    assert!(o.iter().all(|&v| v == SENTINEL));

    let (s, _, again) = good.encode(l.model, 0);
    assert_eq!(s, TESSL_OK);
    assert_eq!(bits(&again), bits(&want));
}

#[test]
fn a_text_only_handle_refuses_the_multimodal_entries_and_still_embeds_text() {
    let _g = gpu_lock();
    let mut err = [0 as c_char; ERR_LEN];
    let rt = unsafe { tessl_runtime_new(err.as_mut_ptr(), ERR_LEN) };
    assert!(!rt.is_null(), "{}", msg(&err));
    let dir = mm_snapshot().to_str().unwrap().as_bytes();
    let mut model = ptr::null_mut();
    let s = unsafe {
        tessl_embedgemma2_load(
            rt,
            dir.as_ptr().cast(),
            dir.len() as u64,
            &mut model,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    assert_eq!(s, TESSL_OK, "{}", msg(&err));
    assert!(unsafe { test_embedgemma2_multimodal(model) }.is_none());
    let inp = inputs();
    let (s, m, out) = flat_images(&[&inp.images[2]]).encode(model, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("tessl_embedgemma2_load_multimodal"), "{m}");
    assert!(out.iter().all(|&v| v == SENTINEL));
    let (s, m, out) = flat_clips(&[&inp.clips[0]]).encode(model, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("tessl_embedgemma2_load_multimodal"), "{m}");
    assert!(out.iter().all(|&v| v == SENTINEL));
    // Text rows agree between the two kinds of handle.
    let ids = [2u32, 9, 4, 4, 1];
    let (offsets, mut a) = ([0u64, 5], vec![SENTINEL; DIM]);
    let s = unsafe {
        tessl_embedgemma2_encode(
            model,
            ids.as_ptr(),
            5,
            offsets.as_ptr(),
            1,
            0,
            a.as_mut_ptr(),
            DIM as u64,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    assert_eq!(s, TESSL_OK, "{}", msg(&err));
    let l = Loaded::new(TESSL_OPERANDS_EXACT_F32);
    let mm = unsafe { test_embedgemma2_multimodal(l.model) }.unwrap();
    assert_eq!(
        bits(&a),
        bits(&mm.encode(&[MmItem::Text(&ids)], None).unwrap().embeddings)
    );
    assert_eq!(unsafe { tessl_embedgemma2_free(model) }, TESSL_OK);
    assert_eq!(unsafe { tessl_runtime_free(rt) }, TESSL_OK);
}

#[test]
fn loading_multimodal_refuses_what_it_cannot_serve() {
    let _g = gpu_lock();
    let mut err = [0 as c_char; ERR_LEN];
    let rt = unsafe { tessl_runtime_new(err.as_mut_ptr(), ERR_LEN) };
    assert!(!rt.is_null(), "{}", msg(&err));
    let dir = mm_snapshot().to_str().unwrap().as_bytes();
    // An operands code that is neither exact f32 nor bf16.
    let (s, m, out) = load_mm_raw(rt, dir, 2);
    assert_eq!(s, TESSL_ERR);
    assert!(out.is_null());
    assert!(m.contains("operands code 2"), "{m}");
    // A config with the towers but not the multimodal ids.
    let no_ids = snapshot("capi_embedgemma2_mm_no_ids", config_json_without_ids());
    let (s, m, out) = load_mm_raw(rt, no_ids.to_str().unwrap().as_bytes(), TESSL_OPERANDS_EXACT_F32);
    assert_eq!(s, TESSL_ERR);
    assert!(out.is_null());
    assert!(m.contains("image_token_id") && m.contains("eoa_token_index"), "{m}");
    // Bad directories, as for the text load.
    for (what, bytes, want_msg) in [
        ("empty", &b""[..], "snapshot_dir is 0 bytes"),
        ("missing", &b"/nonexistent/embedgemma2"[..], "nonexistent"),
        ("NUL", &b"a\0b"[..], "NUL"),
    ] {
        let (s, m, out) = load_mm_raw(rt, bytes, TESSL_OPERANDS_EXACT_F32);
        assert_eq!(s, TESSL_ERR, "{what}");
        assert!(out.is_null(), "{what}: *out is non-null");
        assert!(m.contains(want_msg), "{what}: {m}");
    }
    // Still usable after all of that.
    let (s, m, model) = load_mm_raw(rt, dir, TESSL_OPERANDS_BF16);
    assert_eq!(s, TESSL_OK, "{m}");
    assert_eq!(unsafe { tessl_embedgemma2_free(model) }, TESSL_OK);
    assert_eq!(unsafe { tessl_runtime_free(rt) }, TESSL_OK);
}

#[test]
fn a_model_used_from_another_thread_is_refused_by_the_multimodal_entries() {
    let _g = gpu_lock();
    let l = Loaded::new(TESSL_OPERANDS_EXACT_F32);
    let model = l.model as usize;
    let (s, m, out, s2, m2, out2) = std::thread::spawn(move || {
        let inp = inputs();
        let model = model as *mut TesslEmbedGemma2;
        let (s, m, out) = flat_images(&[&inp.images[2]]).encode(model, 0);
        let (s2, m2, out2) = flat_clips(&[&inp.clips[0]]).encode(model, 0);
        (s, m, out, s2, m2, out2)
    })
    .join()
    .unwrap();
    for (s, m, out) in [(s, m, out), (s2, m2, out2)] {
        assert_eq!(s, TESSL_ERR);
        assert!(m.contains("thread other than the one that created it"), "{m}");
        assert!(out.iter().all(|&v| v == SENTINEL));
    }
    let inp = inputs();
    assert_eq!(flat_images(&[&inp.images[2]]).encode(l.model, 0).0, TESSL_OK);
}
