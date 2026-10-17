//! The C ABI's media entries that read files and bytes, count soft tokens and build
//! items from text interleaved with media
//! (`tessl_embedgemma2_{embed_image_sources, embed_audio_sources, embed_video_sources,
//! image_soft_token_count, audio_soft_token_count, video_soft_token_count, embed_segments}`),
//! called the way a foreign caller calls them, on the tiny multimodal checkpoint of
//! `tests/common/embedgemma2_mm.rs` written to a directory under `target/` with a tokenizer
//! that has every marker as an added token, its sidecars and a `processor_config.json` that
//! fits the tiny towers; the media are `tests/fixtures/media/`.
//!
//! What this file pins is the boundary, as `tests/capi_embedgemma2_raw.rs` does for the raw
//! entries: every refusal is a status and a message that leaves the caller's output untouched,
//! the handle usable and the runtime unpoisoned; each declared length is bounded before it sizes
//! or reads anything (the over-limit cases declare lengths far past tiny buffers, which would
//! fault if anything read first); an error names the file or bytes it is about; and a
//! successful call is the Rust facade (`embedgemma2_media::prepare_*`, `MmItem`) bit for bit on
//! the same weights. The real checkpoint is `tests/embedgemma2_segments_model.rs` and
//! `python/tests/test_embedgemma2_media.py`.
//!
//! The tests with a null handle need no device (they are in the hosted selection,
//! `scripts/hosted_tests.txt`). The video tests decode H.264 through VideoToolbox, which the
//! hosted sandbox denies; they and every other test that creates a runtime run locally.

mod common;
mod media_support;

use std::ffi::{c_char, CStr};
use std::path::{Path, PathBuf};
use std::ptr;

use common::embedgemma2_mm::{
    checkpoint_bytes, config_json, configs, full_tokenizer_json, processor_config_json, sentence_transformers_json,
    TOKENIZER_CONFIG,
};
use common::{gpu_lock, SplitMix};
use tessl::capi::{
    tessl_runtime_free, tessl_runtime_new, tessl_runtime_poisoned, TesslRuntime, TESSL_ERR, TESSL_OK,
    TESSL_OPERANDS_BF16, TESSL_OPERANDS_EXACT_F32,
};
use tessl::capi_embedgemma2::{
    tessl_embedgemma2_audio_soft_token_count, tessl_embedgemma2_embed_audio_sources,
    tessl_embedgemma2_embed_image_sources, tessl_embedgemma2_embed_segments, tessl_embedgemma2_embed_video_sources,
    tessl_embedgemma2_free, tessl_embedgemma2_image_soft_token_count, tessl_embedgemma2_load,
    tessl_embedgemma2_load_multimodal, tessl_embedgemma2_video_soft_token_count, test_embedgemma2_multimodal,
    test_embedgemma2_runtime, TesslEmbedGemma2, TESSL_EMBEDGEMMA2_MAX_PATH_BYTES, TESSL_EMBEDGEMMA2_MAX_SEGMENTS,
    TESSL_EMBEDGEMMA2_SEGMENT_AUDIO, TESSL_EMBEDGEMMA2_SEGMENT_IMAGE, TESSL_EMBEDGEMMA2_SEGMENT_TEXT,
    TESSL_EMBEDGEMMA2_SEGMENT_VIDEO, TESSL_EMBEDGEMMA2_SEGMENT_VIDEO_TIMESTAMPED, TESSL_EMBEDGEMMA2_SOURCE_BYTES,
    TESSL_EMBEDGEMMA2_SOURCE_PATHS, TESSL_EMBEDGEMMA2_SOURCE_RAW,
};
use tessl::embedgemma2::MAX_BATCH;
use tessl::embedgemma2_audio::AudioInput;
use tessl::embedgemma2_media::{prepare_audio, prepare_image, MediaSource, OpenedVideo};
use tessl::embedgemma2_mm::{EmbedGemma2Multimodal, MmItem, MmSegment};
use tessl::embedgemma2_pre::{
    preprocess_image, preprocess_video, AudioFrontEnd, PreConfig, PreparedImage, PreparedVideo, RawImage,
};
use tessl::embedgemma2_tok::Tokenizer;
use tessl::embedgemma2_vision::ImageInput;

const ERR_LEN: usize = 512;
/// The tiny checkpoint's `embedding_dim`.
const DIM: usize = 96;
/// Written to the output of every call that must leave it alone.
const SENTINEL: f32 = -7.25;

/// A refusal case: what it is, what the call returned (status, message, output) and the
/// words the message must hold.
type Case<'a, Out> = (&'a str, (i32, String, Out), &'a str);

fn msg(err: &[c_char]) -> String {
    unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn untouched(out: &[f32]) -> bool {
    out.iter().all(|&v| v == SENTINEL)
}

fn call(f: impl FnOnce(*mut c_char, usize) -> i32) -> (i32, String) {
    let mut err = [0 as c_char; ERR_LEN];
    let s = f(err.as_mut_ptr(), ERR_LEN);
    (s, msg(&err))
}

fn pixels(len: usize, seed: u64) -> Vec<u8> {
    let mut r = SplitMix::new(seed);
    (0..len).map(|_| ((r.unit() + 1.0) * 127.5) as u8).collect()
}

fn pcm(len: usize, seed: u64) -> Vec<f32> {
    let mut r = SplitMix::new(seed);
    (0..len)
        .map(|i| 0.4 * (i as f32 * 0.05).sin() + 0.1 * r.unit())
        .collect()
}

fn ptr_or_null<T>(v: &[T]) -> *const T {
    if v.is_empty() {
        ptr::null()
    } else {
        v.as_ptr()
    }
}

// ---------------------------------------------------------------------------
// The tiny snapshot on disk
// ---------------------------------------------------------------------------

/// The processor config of the tiny towers, sampling a video at 5 fps (the
/// fixture video is 1.2 s: six frames, which the config's cap cuts to four).
fn processor_config() -> String {
    processor_config_json(false, 70).replacen("\"fps\": 1,", "\"fps\": 5,", 1)
}

/// A fresh snapshot directory `name` under `target/`: the tiny multimodal checkpoint, the tokenizer
/// with every marker as an added token, its sidecars and the processor config.
fn snapshot(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the snapshot directory");
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    std::fs::write(
        dir.join("model.safetensors"),
        checkpoint_bytes(&configs(), 2026, tessl::gemm::GemmOperands::Bf16),
    )
    .unwrap();
    std::fs::write(dir.join("tokenizer.json"), full_tokenizer_json()).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), TOKENIZER_CONFIG).unwrap();
    std::fs::write(
        dir.join("config_sentence_transformers.json"),
        sentence_transformers_json("null"),
    )
    .unwrap();
    std::fs::write(dir.join("processor_config.json"), processor_config()).unwrap();
    dir
}

fn tokenizer() -> Tokenizer {
    Tokenizer::from_json(&full_tokenizer_json())
        .unwrap()
        .with_tokenizer_config(TOKENIZER_CONFIG)
        .unwrap()
}

fn pre() -> PreConfig {
    PreConfig::from_json(&processor_config()).unwrap()
}

/// `operands` `None` loads the text encoder alone.
fn load(rt: *mut TesslRuntime, dir: &Path, operands: Option<u32>) -> (i32, String, *mut TesslEmbedGemma2) {
    let bytes = dir.to_str().unwrap().as_bytes();
    let mut err = [0 as c_char; ERR_LEN];
    let mut out = 0x10 as *mut TesslEmbedGemma2;
    let s = unsafe {
        match operands {
            None => tessl_embedgemma2_load(
                rt,
                bytes.as_ptr().cast(),
                bytes.len() as u64,
                &mut out,
                err.as_mut_ptr(),
                ERR_LEN,
            ),
            Some(op) => tessl_embedgemma2_load_multimodal(
                rt,
                bytes.as_ptr().cast(),
                bytes.len() as u64,
                op,
                &mut out,
                err.as_mut_ptr(),
                ERR_LEN,
            ),
        }
    };
    (s, msg(&err), out)
}

/// A runtime handle and a model loaded on it, both freed on drop.
struct Loaded {
    rt: *mut TesslRuntime,
    model: *mut TesslEmbedGemma2,
}

impl Loaded {
    fn new(dir: &Path, operands: Option<u32>) -> Self {
        let mut err = [0 as c_char; ERR_LEN];
        let rt = unsafe { tessl_runtime_new(err.as_mut_ptr(), ERR_LEN) };
        assert!(!rt.is_null(), "tessl_runtime_new: {}", msg(&err));
        let (s, m, model) = load(rt, dir, operands);
        assert_eq!(s, TESSL_OK, "load: {m}");
        assert!(!model.is_null());
        Self { rt, model }
    }

    fn mm(&self) -> &'static EmbedGemma2Multimodal {
        unsafe { test_embedgemma2_multimodal(self.model) }.expect("a multimodal handle")
    }

    /// Whether a refused call poisoned the runtime (it must never).
    fn poisoned(&self) -> bool {
        let mut out = -1i32;
        let (s, m) = call(|e, n| unsafe { tessl_runtime_poisoned(self.rt, &mut out, e, n) });
        assert_eq!(s, TESSL_OK, "{m}");
        out == 1
    }
}

impl Drop for Loaded {
    fn drop(&mut self) {
        assert_eq!(unsafe { tessl_embedgemma2_free(self.model) }, TESSL_OK);
        assert_eq!(unsafe { tessl_runtime_free(self.rt) }, TESSL_OK);
    }
}

// ---------------------------------------------------------------------------
// Encoded-media tables
// ---------------------------------------------------------------------------

/// Media laid end to end as the C entries take them: the blob and its cuts.
#[derive(Clone)]
struct Table {
    blob: Vec<u8>,
    offsets: Vec<u64>,
}

impl Table {
    fn n(&self) -> u64 {
        (self.offsets.len() - 1) as u64
    }
}

fn table(parts: &[Vec<u8>]) -> Table {
    let mut t = Table {
        blob: Vec::new(),
        offsets: vec![0],
    };
    for p in parts {
        t.blob.extend_from_slice(p);
        t.offsets.push(t.blob.len() as u64);
    }
    t
}

fn path_table(names: &[&str]) -> Table {
    table(
        &names
            .iter()
            .map(|n| media_support::fixture(n).to_str().unwrap().as_bytes().to_vec())
            .collect::<Vec<_>>(),
    )
}

fn bytes_table(names: &[&str]) -> Table {
    table(&names.iter().map(|n| media_support::bytes(n)).collect::<Vec<_>>())
}

fn path_of(name: &str) -> PathBuf {
    media_support::fixture(name)
}

fn out_vec(n: usize, trunc: u32) -> Vec<f32> {
    vec![SENTINEL; n * if trunc == 0 { DIM } else { trunc as usize }]
}

/// `embed_image_sources` on well-formed arguments.
fn embed_images_from(model: *mut TesslEmbedGemma2, source: u32, t: &Table, trunc: u32) -> (i32, String, Vec<f32>) {
    let mut out = out_vec(t.offsets.len() - 1, trunc);
    let (s, m) = call(|e, n_e| unsafe {
        tessl_embedgemma2_embed_image_sources(
            model,
            source,
            t.blob.as_ptr(),
            t.blob.len() as u64,
            t.offsets.as_ptr(),
            t.n(),
            trunc,
            out.as_mut_ptr(),
            out.len() as u64,
            e,
            n_e,
        )
    });
    (s, m, out)
}

fn embed_clips_from(model: *mut TesslEmbedGemma2, source: u32, t: &Table, trunc: u32) -> (i32, String, Vec<f32>) {
    let mut out = out_vec(t.offsets.len() - 1, trunc);
    let (s, m) = call(|e, n_e| unsafe {
        tessl_embedgemma2_embed_audio_sources(
            model,
            source,
            t.blob.as_ptr(),
            t.blob.len() as u64,
            t.offsets.as_ptr(),
            t.n(),
            trunc,
            out.as_mut_ptr(),
            out.len() as u64,
            e,
            n_e,
        )
    });
    (s, m, out)
}

fn embed_videos_from(
    model: *mut TesslEmbedGemma2,
    source: u32,
    t: &Table,
    ext: Option<&str>,
    timestamps: u32,
    trunc: u32,
) -> (i32, String, Vec<f32>) {
    let mut out = out_vec(t.offsets.len() - 1, trunc);
    let (ep, el) = ext.map_or((ptr::null(), 0), |x| (x.as_ptr().cast::<c_char>(), x.len() as u64));
    let (s, m) = call(|e, n_e| unsafe {
        tessl_embedgemma2_embed_video_sources(
            model,
            source,
            t.blob.as_ptr(),
            t.blob.len() as u64,
            t.offsets.as_ptr(),
            t.n(),
            ep,
            el,
            timestamps,
            trunc,
            out.as_mut_ptr(),
            out.len() as u64,
            e,
            n_e,
        )
    });
    (s, m, out)
}

// ---------------------------------------------------------------------------
// The Rust facade's side
// ---------------------------------------------------------------------------

fn prepared_images(names: &[&str]) -> Vec<PreparedImage> {
    let cfg = pre();
    names
        .iter()
        .map(|n| prepare_image(&cfg.image, MediaSource::Path(&path_of(n))).unwrap())
        .collect()
}

fn prepared_clips(names: &[&str]) -> Vec<AudioInput> {
    let fe = AudioFrontEnd::new(&pre().audio).unwrap();
    names
        .iter()
        .map(|n| prepare_audio(&fe, MediaSource::Path(&path_of(n))).unwrap())
        .collect()
}

fn prepared_video(name: &str) -> PreparedVideo {
    let cfg = pre();
    OpenedVideo::open(MediaSource::Path(&path_of(name)), "mp4")
        .unwrap()
        .prepare(&cfg.video)
        .unwrap()
}

fn encode(mm: &EmbedGemma2Multimodal, items: &[MmItem<'_>], trunc: u32) -> Vec<f32> {
    mm.encode(items, (trunc != 0).then_some(trunc)).unwrap().embeddings
}

// ---------------------------------------------------------------------------
// No device: a null handle
// ---------------------------------------------------------------------------

#[test]
fn a_null_handle_is_refused_by_every_media_entry() {
    let null: *mut TesslEmbedGemma2 = ptr::null_mut();
    let t = path_table(&["rgb.png"]);
    let mut counts = [0u64; 2];
    let dims = [8u32, 8];
    let samples = [3200u64];
    let frames = [4u64];
    let fps = [1.0f64];
    let mut results = Vec::new();
    results.push(("embed_image_sources", {
        let (s, m, o) = embed_images_from(null, TESSL_EMBEDGEMMA2_SOURCE_PATHS, &t, 0);
        assert!(untouched(&o));
        (s, m)
    }));
    results.push(("embed_audio_sources", {
        let (s, m, o) = embed_clips_from(null, TESSL_EMBEDGEMMA2_SOURCE_PATHS, &t, 0);
        assert!(untouched(&o));
        (s, m)
    }));
    results.push(("embed_video_sources", {
        let (s, m, o) = embed_videos_from(null, TESSL_EMBEDGEMMA2_SOURCE_PATHS, &t, None, 0, 0);
        assert!(untouched(&o));
        (s, m)
    }));
    results.push(("image_soft_token_count", {
        call(|e, n| unsafe {
            tessl_embedgemma2_image_soft_token_count(null, dims.as_ptr(), 1, counts.as_mut_ptr(), 1, e, n)
        })
    }));
    results.push(("audio_soft_token_count", {
        call(|e, n| unsafe {
            tessl_embedgemma2_audio_soft_token_count(null, samples.as_ptr(), 1, 16000, counts.as_mut_ptr(), 1, e, n)
        })
    }));
    results.push(("video_soft_token_count", {
        call(|e, n| unsafe {
            tessl_embedgemma2_video_soft_token_count(
                null,
                dims.as_ptr(),
                frames.as_ptr(),
                fps.as_ptr(),
                1,
                counts.as_mut_ptr(),
                2,
                e,
                n,
            )
        })
    }));
    results.push(("embed_segments", {
        let call_args = SegCall::text_only(&["12"]);
        let (s, m, o) = call_args.run(null, 0);
        assert!(untouched(&o));
        (s, m)
    }));
    assert_eq!(results.len(), 7);
    for (name, (status, message)) in results {
        assert_eq!(status, TESSL_ERR, "{name}");
        assert!(message.contains("null embedgemma2 model handle"), "{name}: {message}");
    }
    assert_eq!(counts, [0, 0], "a refused query wrote its output");
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

const IMAGES: [&str; 5] = ["rgb.png", "gray.png", "rgb.bmp", "rgb.tiff", "rgb.jpg"];

#[test]
fn image_files_and_bytes_through_the_c_entry_are_decode_preprocess_and_encode_bit_for_bit() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_media_images");
    for operands in [TESSL_OPERANDS_EXACT_F32, TESSL_OPERANDS_BF16] {
        let l = Loaded::new(&dir, Some(operands));
        let prepared = prepared_images(&IMAGES);
        let items: Vec<MmItem<'_>> = prepared.iter().map(|p| MmItem::Image(p.input())).collect();
        for trunc in [0u32, 40, 1] {
            let want = encode(l.mm(), &items, trunc);
            for (what, source, t) in [
                ("paths", TESSL_EMBEDGEMMA2_SOURCE_PATHS, path_table(&IMAGES)),
                ("bytes", TESSL_EMBEDGEMMA2_SOURCE_BYTES, bytes_table(&IMAGES)),
            ] {
                let (s, m, got) = embed_images_from(l.model, source, &t, trunc);
                assert_eq!(s, TESSL_OK, "{what}: {m}");
                assert_eq!(bits(&got), bits(&want), "{what}, {operands}, truncate {trunc}");
            }
        }
        // One image alone is its row of the batch (up to the text side's batch bound).
        let (s, m, one) = embed_images_from(l.model, TESSL_EMBEDGEMMA2_SOURCE_PATHS, &path_table(&["gray.png"]), 0);
        assert_eq!(s, TESSL_OK, "{m}");
        let all = encode(l.mm(), &items, 0);
        let worst = one
            .iter()
            .zip(&all[DIM..2 * DIM])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= 1e-5, "alone vs in a batch: {worst}");
        assert!(!l.poisoned());
    }
}

/// Refusals shared by the three source entries (they share the table reader), run through the
/// image entry: the call is refused, the output untouched, the message names the reason.
#[test]
fn encoded_table_refusals_leave_the_output_alone_the_handle_usable_and_the_runtime_unpoisoned() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_media_tables");
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    let good = path_table(&["rgb.png", "gray.png"]);
    let (s, m, want) = embed_images_from(l.model, TESSL_EMBEDGEMMA2_SOURCE_PATHS, &good, 0);
    assert_eq!(s, TESSL_OK, "{m}");

    let run = |source: u32, blob: &[u8], blob_len: u64, offsets: &[u64], n: u64, trunc: u32, out_len: u64| {
        let mut out = vec![SENTINEL; 2 * DIM + 8];
        let (s, m) = call(|e, n_e| unsafe {
            tessl_embedgemma2_embed_image_sources(
                l.model,
                source,
                ptr_or_null(blob),
                blob_len,
                ptr_or_null(offsets),
                n,
                trunc,
                out.as_mut_ptr(),
                out_len,
                e,
                n_e,
            )
        });
        (s, m, out)
    };
    let paths = TESSL_EMBEDGEMMA2_SOURCE_PATHS;
    let bytes = TESSL_EMBEDGEMMA2_SOURCE_BYTES;
    let (blob, offs) = (good.blob.clone(), good.offsets.clone());
    let blen = blob.len() as u64;
    let two = 2 * DIM as u64;
    let mut cases: Vec<Case<'_, Vec<f32>>> =
        vec![("n_images 0", run(paths, &blob, blen, &offs, 0, 0, 0), "n_images is 0")];
    cases.push((
        "n_images past the batch",
        run(paths, &blob, blen, &offs, 257, 0, 257 * DIM as u64),
        "n_images is 257",
    ));
    cases.push((
        "source raw",
        run(TESSL_EMBEDGEMMA2_SOURCE_RAW, &blob, blen, &offs, 2, 0, two),
        "image_source is 0",
    ));
    cases.push(("source 3", run(3, &blob, blen, &offs, 2, 0, two), "image_source is 3"));
    cases.push((
        "out_len wrong",
        run(paths, &blob, blen, &offs, 2, 0, DIM as u64),
        "out_len is 96",
    ));
    cases.push((
        "truncate_dim past the width",
        run(paths, &blob, blen, &offs, 2, 97, 2 * 97),
        "truncate_dim 97",
    ));
    cases.push(("null blob", run(paths, &[], blen, &offs, 2, 0, two), "null"));
    cases.push(("null offsets", run(paths, &blob, blen, &[], 2, 0, two), "null"));
    let mut bad = offs.clone();
    bad[0] = 1;
    cases.push((
        "offsets not from 0",
        run(paths, &blob, blen, &bad, 2, 0, two),
        "offsets[0] is 1",
    ));
    let mut bad = offs.clone();
    bad[1] = bad[0];
    cases.push(("an empty path", run(paths, &blob, blen, &bad, 2, 0, two), "never empty"));
    let mut bad = offs.clone();
    bad[1] = bad[2] + 1;
    cases.push((
        "offsets decreasing",
        run(paths, &blob, blen, &bad, 2, 0, two),
        "does not exceed",
    ));
    cases.push((
        "offsets end short of the blob",
        run(paths, &blob, blen - 1, &offs, 2, 0, two),
        "end at",
    ));
    cases.push((
        "offsets end past the blob",
        run(paths, &blob, blen + 1, &offs, 2, 0, two),
        "end at",
    ));
    // A declared length past every bound, with a tiny buffer: nothing may be read.
    let path_limit: &'static str = format!("the limit is {TESSL_EMBEDGEMMA2_MAX_PATH_BYTES}").leak();
    cases.push((
        "a path past the path bound",
        run(paths, &blob[..4], 1 << 40, &[0, 1 << 40], 1, 0, DIM as u64),
        path_limit,
    ));
    cases.push((
        "bytes past the image file bound",
        run(bytes, &blob[..4], 1 << 40, &[0, 1 << 40], 1, 0, DIM as u64),
        "the limit is 1073741824",
    ));
    let mut nul = b"/tmp/a\0b".to_vec();
    nul.extend_from_slice(b"x");
    cases.push((
        "a NUL byte in a path",
        run(paths, &nul, nul.len() as u64, &[0, nul.len() as u64], 1, 0, DIM as u64),
        "image 0: the path contains a NUL byte",
    ));
    let bad_utf8 = [0xffu8, 0xfe, b'x'];
    cases.push((
        "a path that is not UTF-8",
        run(paths, &bad_utf8, 3, &[0, 3], 1, 0, DIM as u64),
        "image 0: the path is not UTF-8",
    ));
    // The decoders' own refusals name the index and the reason.
    let missing = table(&[
        media_support::fixture("rgb.png").to_str().unwrap().as_bytes().to_vec(),
        b"/no/such/dir/missing.png".to_vec(),
    ]);
    cases.push((
        "a missing file",
        run(
            paths,
            &missing.blob,
            missing.blob.len() as u64,
            &missing.offsets,
            2,
            0,
            two,
        ),
        "file 1: cannot read /no/such/dir/missing.png",
    ));
    let dir_path = table(&[media_support::dir().to_str().unwrap().as_bytes().to_vec()]);
    cases.push((
        "a directory",
        run(
            paths,
            &dir_path.blob,
            dir_path.blob.len() as u64,
            &dir_path.offsets,
            1,
            0,
            DIM as u64,
        ),
        "file 0: image: invalid argument",
    ));
    let garbage = table(&[media_support::bytes("rgb.png"), b"this is not an image".to_vec()]);
    cases.push((
        "bytes that are not an image",
        run(
            bytes,
            &garbage.blob,
            garbage.blob.len() as u64,
            &garbage.offsets,
            2,
            0,
            two,
        ),
        "bytes 1: image: not a format the system decodes",
    ));
    let cut = media_support::bytes("rgb.png");
    let cut = table(&[cut[..cut.len() / 2].to_vec()]);
    cases.push((
        "a cut-short image",
        run(bytes, &cut.blob, cut.blob.len() as u64, &cut.offsets, 1, 0, DIM as u64),
        "bytes 0: image:",
    ));
    for (what, (status, message, out), want) in &cases {
        assert_eq!(*status, TESSL_ERR, "{what}: {message}");
        assert!(message.contains(want), "{what}: {message}");
        assert!(
            out.iter().all(|&v| v == SENTINEL),
            "{what}: a refused call wrote its output"
        );
    }
    assert!(!l.poisoned(), "a refusal poisoned the runtime");
    let (s, m, got) = embed_images_from(l.model, paths, &good, 0);
    assert_eq!(s, TESSL_OK, "the handle is unusable afterwards: {m}");
    assert_eq!(bits(&got), bits(&want));
}

#[test]
fn a_text_only_handle_refuses_the_media_entries_naming_the_multimodal_load() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_media_text_only");
    let l = Loaded::new(&dir, None);
    let t = path_table(&["rgb.png"]);
    let results = [
        (
            "embed_image_sources",
            embed_images_from(l.model, TESSL_EMBEDGEMMA2_SOURCE_PATHS, &t, 0),
        ),
        (
            "embed_audio_sources",
            embed_clips_from(l.model, TESSL_EMBEDGEMMA2_SOURCE_PATHS, &t, 0),
        ),
        (
            "embed_video_sources",
            embed_videos_from(l.model, TESSL_EMBEDGEMMA2_SOURCE_PATHS, &t, None, 0, 0),
        ),
        ("embed_segments", SegCall::text_only(&["12"]).run(l.model, 0)),
    ];
    for (name, (s, m, o)) in results {
        assert_eq!(s, TESSL_ERR, "{name}");
        assert!(m.contains("tessl_embedgemma2_load_multimodal"), "{name}: {m}");
        assert!(untouched(&o), "{name}");
    }
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

const CLIPS: [&str; 4] = [
    "pcm16_16k_mono.wav",
    "pcm16_44k_stereo.wav",
    "tone_44k_stereo.flac",
    "tone_16k_mono.mp3",
];

#[test]
fn audio_files_and_bytes_through_the_c_entry_are_decode_extract_and_encode_bit_for_bit() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_media_audio");
    for operands in [TESSL_OPERANDS_EXACT_F32, TESSL_OPERANDS_BF16] {
        let l = Loaded::new(&dir, Some(operands));
        let clips = prepared_clips(&CLIPS);
        let items: Vec<MmItem<'_>> = clips.iter().map(MmItem::Audio).collect();
        for trunc in [0u32, 40] {
            let want = encode(l.mm(), &items, trunc);
            for (what, source, t) in [
                ("paths", TESSL_EMBEDGEMMA2_SOURCE_PATHS, path_table(&CLIPS)),
                ("bytes", TESSL_EMBEDGEMMA2_SOURCE_BYTES, bytes_table(&CLIPS)),
            ] {
                let (s, m, got) = embed_clips_from(l.model, source, &t, trunc);
                assert_eq!(s, TESSL_OK, "{what}: {m}");
                assert_eq!(bits(&got), bits(&want), "{what}, {operands}, truncate {trunc}");
            }
        }
        assert!(!l.poisoned());
    }
}

#[test]
fn audio_refusals_name_the_clip_and_leave_the_output_alone() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_media_audio_refusals");
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    let good = path_table(&["pcm16_16k_mono.wav"]);
    let (s, m, want) = embed_clips_from(l.model, TESSL_EMBEDGEMMA2_SOURCE_PATHS, &good, 0);
    assert_eq!(s, TESSL_OK, "{m}");
    let wv = path_table(&["pcm16_16k_mono.wav", "tone_16k_mono.wv"]);
    let empty = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/capi_embedgemma2_media_empty.wav");
    std::fs::write(&empty, b"").unwrap();
    let empties = table(&[empty.to_str().unwrap().as_bytes().to_vec()]);
    let garbage = table(&[b"RIFF not a wave file".to_vec()]);
    let cases = [
        (
            "an unsupported format",
            embed_clips_from(l.model, 1, &wv, 0),
            "file 1: audio: not a format the system decodes",
        ),
        (
            "an empty file",
            embed_clips_from(l.model, 1, &empties, 0),
            "file 0: audio: no bytes to decode",
        ),
        (
            "garbage bytes",
            embed_clips_from(l.model, 2, &garbage, 0),
            "bytes 0: audio:",
        ),
        (
            "source raw",
            embed_clips_from(l.model, 0, &good, 0),
            "audio_source is 0",
        ),
    ];
    for (what, (status, message, out), want) in cases {
        assert_eq!(status, TESSL_ERR, "{what}: {message}");
        assert!(message.contains(want), "{what}: {message}");
        assert!(untouched(&out), "{what}: a refused call wrote its output");
    }
    assert!(!l.poisoned());
    let (s, m, got) = embed_clips_from(l.model, 1, &good, 0);
    assert_eq!(s, TESSL_OK, "{m}");
    assert_eq!(bits(&got), bits(&want));
}

// ---------------------------------------------------------------------------
// Video (VideoToolbox: local only)
// ---------------------------------------------------------------------------

#[test]
fn video_files_and_bytes_through_the_c_entry_are_decode_sample_and_encode_bit_for_bit() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_media_video");
    let tok = tokenizer();
    for operands in [TESSL_OPERANDS_EXACT_F32, TESSL_OPERANDS_BF16] {
        let l = Loaded::new(&dir, Some(operands));
        let video = prepared_video("colors.mp4");
        assert_eq!(video.frames.len(), 4, "six sampled frames cut to the config's four");
        let frames: Vec<ImageInput<'_>> = video.frames.iter().map(PreparedImage::input).collect();
        for (timestamps, trunc) in [(0u32, 0u32), (1, 0), (0, 40), (1, 40)] {
            let item = if timestamps == 1 {
                MmItem::TimestampedVideo {
                    frames: &frames,
                    timestamps: &video.timestamps,
                    tokenizer: &tok,
                }
            } else {
                MmItem::Video(&frames)
            };
            let want = encode(l.mm(), &[item], trunc);
            for (what, source, t, ext) in [
                (
                    "path",
                    TESSL_EMBEDGEMMA2_SOURCE_PATHS,
                    path_table(&["colors.mp4"]),
                    None,
                ),
                (
                    "bytes",
                    TESSL_EMBEDGEMMA2_SOURCE_BYTES,
                    bytes_table(&["colors.mp4"]),
                    Some("mp4"),
                ),
                (
                    "bytes, default extension",
                    TESSL_EMBEDGEMMA2_SOURCE_BYTES,
                    bytes_table(&["colors.mp4"]),
                    None,
                ),
            ] {
                let (s, m, got) = embed_videos_from(l.model, source, &t, ext, timestamps, trunc);
                assert_eq!(s, TESSL_OK, "{what}: {m}");
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "{what}, {operands}, timestamps {timestamps}, truncate {trunc}"
                );
            }
        }
        assert!(!l.poisoned());
    }
    assert_eq!(media_support::temp_video_files(), 0, "a temporary video file was left");
}

#[test]
fn video_refusals_name_the_video_and_leave_the_output_alone() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_media_video_refusals");
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    let good = path_table(&["colors.mp4"]);
    let (s, m, want) = embed_videos_from(l.model, 1, &good, None, 0, 0);
    assert_eq!(s, TESSL_OK, "{m}");
    let bytes = bytes_table(&["colors.mp4"]);
    let garbage = table(&[b"not a video".to_vec()]);
    let missing = table(&[b"/no/such/video.mp4".to_vec()]);
    let before = media_support::temp_video_files();
    let cases = [
        (
            "timestamps 2",
            embed_videos_from(l.model, 1, &good, None, 2, 0),
            "timestamps is 2",
        ),
        (
            "an extension with paths",
            embed_videos_from(l.model, 1, &good, Some("mp4"), 0, 0),
            "extension hint goes with encoded bytes",
        ),
        (
            "an extension past the bound",
            embed_videos_from(l.model, 2, &bytes, Some("mp4mp4mp4"), 0, 0),
            "extension_len is 9",
        ),
        (
            "an extension that is not a type",
            embed_videos_from(l.model, 2, &bytes, Some("m p4"), 0, 0),
            "bytes 0: video: invalid argument",
        ),
        (
            "source raw",
            embed_videos_from(l.model, 0, &good, None, 0, 0),
            "video_source is 0",
        ),
        (
            "a missing file",
            embed_videos_from(l.model, 1, &missing, None, 0, 0),
            "file 0: cannot read /no/such/video.mp4",
        ),
        (
            "garbage bytes",
            embed_videos_from(l.model, 2, &garbage, Some("mp4"), 0, 0),
            "bytes 0: video:",
        ),
    ];
    for (what, (status, message, out), want) in cases {
        assert_eq!(status, TESSL_ERR, "{what}: {message}");
        assert!(message.contains(want), "{what}: {message}");
        assert!(untouched(&out), "{what}: a refused call wrote its output");
    }
    assert_eq!(
        media_support::temp_video_files(),
        before,
        "a refused call left a temporary file"
    );
    assert!(!l.poisoned());
    let (s, m, got) = embed_videos_from(l.model, 1, &good, None, 0, 0);
    assert_eq!(s, TESSL_OK, "{m}");
    assert_eq!(bits(&got), bits(&want));
}

// ---------------------------------------------------------------------------
// Soft-token counts
// ---------------------------------------------------------------------------

fn image_counts(model: *mut TesslEmbedGemma2, dims: &[u32], n: u64, out_len: u64) -> (i32, String, Vec<u64>) {
    let mut out = vec![u64::MAX; (out_len as usize).max(1)];
    let (s, m) = call(|e, n_e| unsafe {
        tessl_embedgemma2_image_soft_token_count(model, ptr_or_null(dims), n, out.as_mut_ptr(), out_len, e, n_e)
    });
    (s, m, out)
}

fn audio_counts(model: *mut TesslEmbedGemma2, samples: &[u64], rate: u32, out_len: u64) -> (i32, String, Vec<u64>) {
    let mut out = vec![u64::MAX; (out_len as usize).max(1)];
    let (s, m) = call(|e, n_e| unsafe {
        tessl_embedgemma2_audio_soft_token_count(
            model,
            ptr_or_null(samples),
            samples.len() as u64,
            rate,
            out.as_mut_ptr(),
            out_len,
            e,
            n_e,
        )
    });
    (s, m, out)
}

fn video_counts(
    model: *mut TesslEmbedGemma2,
    dims: &[u32],
    frames: &[u64],
    fps: &[f64],
    out_len: u64,
) -> (i32, String, Vec<u64>) {
    let mut out = vec![u64::MAX; (out_len as usize).max(2)];
    let (s, m) = call(|e, n_e| unsafe {
        tessl_embedgemma2_video_soft_token_count(
            model,
            ptr_or_null(dims),
            ptr_or_null(frames),
            ptr_or_null(fps),
            frames.len() as u64,
            out.as_mut_ptr(),
            out_len,
            e,
            n_e,
        )
    });
    (s, m, out)
}

#[test]
fn soft_token_counts_are_what_the_towers_make() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_media_counts");
    let cfg = pre();
    // Counts need no tower: a text-only handle answers them, and a multimodal one agrees.
    let text_only = Loaded::new(&dir, None);
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));

    // Images: the count of the C entry, of preprocessing, and of the vision tower's check.
    let shapes = [
        (8usize, 8usize),
        (13, 40),
        (200, 31),
        (3, 400),
        (64, 64),
        (1, 9),
        (23, 37),
    ];
    let dims: Vec<u32> = shapes.iter().flat_map(|&(h, w)| [h as u32, w as u32]).collect();
    for h in [text_only.model, l.model] {
        let (s, m, got) = image_counts(h, &dims, shapes.len() as u64, shapes.len() as u64);
        assert_eq!(s, TESSL_OK, "{m}");
        for (i, &(height, width)) in shapes.iter().enumerate() {
            let data = pixels(height * width * 3, i as u64);
            let p = preprocess_image(&cfg.image, &RawImage::rgb(width, height, &data)).unwrap();
            assert_eq!(got[i] as usize, p.soft_tokens, "{height}x{width}");
            // The tiny tower's position table is 128 wide: the one very wide shape is past it
            // (the count is a property of the processor, not of the tower, and still agrees).
            match l.mm().vision().check_images(&[p.input()]) {
                Ok(tower) => assert_eq!(got[i] as usize, tower[0], "{height}x{width} against the tower's check"),
                Err(e) => assert!(
                    (height, width) == (3, 400) && e.contains("position table"),
                    "{height}x{width}: {e}"
                ),
            }
        }
    }

    // Clips: the extractor's mask, the tower's check.
    let samples: Vec<u64> = vec![
        400, 1599, 1600, 3200, 6400, 16_000, 47_999, 48_000, 480_000, 480_001, 1_000_000,
    ];
    let fe = AudioFrontEnd::new(&cfg.audio).unwrap();
    let (s, m, got) = audio_counts(l.model, &samples, 16000, samples.len() as u64);
    assert_eq!(s, TESSL_OK, "{m}");
    let (s, m, text_only_got) = audio_counts(text_only.model, &samples, 16000, samples.len() as u64);
    assert_eq!(s, TESSL_OK, "{m}");
    assert_eq!(got, text_only_got);
    for (i, &n) in samples.iter().enumerate() {
        let clip = fe.extract(&pcm(n as usize, i as u64), 16000).unwrap();
        let tower = l.mm().audio().check_clips(&[&clip]).unwrap();
        assert_eq!(got[i] as usize, tower[0], "{n} samples against the tower's check");
    }

    // Videos: the sampler's frames and the per-frame count of preprocessing.
    let spec = [
        (12usize, 48usize, 64usize, 10.0f64),
        (40, 8, 8, 1.0),
        (3, 9, 33, 30.0),
        (100, 20, 12, 2.5),
    ];
    let dims: Vec<u32> = spec.iter().flat_map(|&(_, h, w, _)| [h as u32, w as u32]).collect();
    let frames: Vec<u64> = spec.iter().map(|s| s.0 as u64).collect();
    let fps: Vec<f64> = spec.iter().map(|s| s.3).collect();
    let (s, m, got) = video_counts(l.model, &dims, &frames, &fps, 2 * spec.len() as u64);
    assert_eq!(s, TESSL_OK, "{m}");
    for (i, &(n, h, w, rate)) in spec.iter().enumerate() {
        let data = pixels(n * h * w * 3, 90 + i as u64);
        let raws: Vec<RawImage<'_>> = (0..n)
            .map(|f| RawImage::rgb(w, h, &data[f * h * w * 3..(f + 1) * h * w * 3]))
            .collect();
        let video = preprocess_video(&cfg.video, &raws, rate).unwrap();
        assert_eq!(
            (got[2 * i] as usize, got[2 * i + 1] as usize),
            (video.frames.len(), video.soft_tokens_per_frame),
            "video {i}"
        );
        let inputs: Vec<ImageInput<'_>> = video.frames.iter().map(PreparedImage::input).collect();
        let tower = l.mm().vision().check_images(&inputs).unwrap();
        assert!(
            tower.iter().all(|&t| t == got[2 * i + 1] as usize),
            "video {i} against the tower"
        );
    }
}

#[test]
fn soft_token_count_refusals_name_the_item_and_leave_the_output_alone() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_media_count_refusals");
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    let m = l.model;
    let untouched_u64 = |v: &[u64]| v.iter().all(|&x| x == u64::MAX);
    let image_cases: Vec<Case<'_, Vec<u64>>> = vec![
        ("n 0", image_counts(m, &[], 0, 0), "n_images is 0"),
        (
            "n past the bound",
            image_counts(m, &[8, 8], 4097, 4097),
            "n_images is 4097",
        ),
        (
            "out_len wrong",
            image_counts(m, &[8, 8, 8, 8], 2, 1),
            "out_len is 1, expected exactly 2",
        ),
        ("a zero height", image_counts(m, &[8, 8, 0, 8], 2, 2), "image 1 is 0x8"),
        ("a zero width", image_counts(m, &[8, 0], 1, 1), "image 0 is 8x0"),
        (
            "past the pixel limit",
            image_counts(m, &[70_000, 70_000], 1, 1),
            "pixel limit",
        ),
        ("null dims", image_counts(m, &[], 2, 2), "null"),
    ];
    let audio_cases: Vec<Case<'_, Vec<u64>>> = vec![
        ("n 0", audio_counts(m, &[], 16000, 0), "n_clips is 0"),
        ("another rate", audio_counts(m, &[16_000], 8000, 1), "8000 Hz audio"),
        (
            "a clip of no samples",
            audio_counts(m, &[16_000, 0], 16000, 2),
            "clip 1 has 0 samples",
        ),
        (
            "a clip past the limit",
            audio_counts(m, &[(1 << 28) + 1], 16000, 1),
            "clip 0 has 268435457 samples",
        ),
        (
            "a clip too short for a frame",
            audio_counts(m, &[10], 16000, 1),
            "clip 0 (10 samples) is too short",
        ),
        (
            "out_len wrong",
            audio_counts(m, &[16_000, 16_000], 16000, 1),
            "out_len is 1, expected exactly 2",
        ),
    ];
    let video_cases: Vec<Case<'_, Vec<u64>>> = vec![
        ("n 0", video_counts(m, &[], &[], &[], 0), "n_videos is 0"),
        (
            "out_len wrong",
            video_counts(m, &[8, 8], &[4], &[1.0], 1),
            "out_len is 1, expected exactly 2",
        ),
        (
            "a zero dimension",
            video_counts(m, &[0, 8], &[4], &[1.0], 2),
            "video 0 is 0x8",
        ),
        (
            "no frames",
            video_counts(m, &[8, 8], &[0], &[1.0], 2),
            "video 0 has 0 frames",
        ),
        (
            "too many frames",
            video_counts(m, &[8, 8], &[(1 << 20) + 1], &[1.0], 2),
            "video 0 has 1048577 frames",
        ),
        ("fps 0", video_counts(m, &[8, 8], &[4], &[0.0], 2), "video 0:"),
        ("fps NaN", video_counts(m, &[8, 8], &[4], &[f64::NAN], 2), "video 0:"),
    ];
    for (what, (status, message, out), want) in image_cases.into_iter().chain(audio_cases).chain(video_cases) {
        assert_eq!(status, TESSL_ERR, "{what}: {message}");
        assert!(message.contains(want), "{what}: {message}");
        assert!(untouched_u64(&out), "{what}: a refused call wrote its output");
    }
    assert!(!l.poisoned());
}

// ---------------------------------------------------------------------------
// Segments
// ---------------------------------------------------------------------------

/// Texts laid end to end.
fn text_table(t: &[&str]) -> Table {
    table(&t.iter().map(|s| s.as_bytes().to_vec()).collect::<Vec<_>>())
}

/// Every argument of `embed_segments`, as plain vectors a test can edit.
#[derive(Clone)]
struct SegCall {
    texts: Table,
    kinds: Vec<u32>,
    refs: Vec<u64>,
    item_offsets: Vec<u64>,
    image_source: u32,
    image: Table,
    image_dims: Vec<u32>,
    audio_source: u32,
    audio: Table,
    sampling_rate: u32,
    video_source: u32,
    video: Table,
    video_dims: Vec<u32>,
    video_fps: Vec<f64>,
    extension: Option<String>,
    n_items: Option<u64>,
}

impl SegCall {
    fn empty() -> Self {
        Self {
            texts: text_table(&[]),
            kinds: Vec::new(),
            refs: Vec::new(),
            item_offsets: vec![0],
            image_source: TESSL_EMBEDGEMMA2_SOURCE_PATHS,
            image: table(&[]),
            image_dims: Vec::new(),
            audio_source: TESSL_EMBEDGEMMA2_SOURCE_PATHS,
            audio: table(&[]),
            sampling_rate: 0,
            video_source: TESSL_EMBEDGEMMA2_SOURCE_PATHS,
            video: table(&[]),
            video_dims: Vec::new(),
            video_fps: Vec::new(),
            extension: None,
            n_items: None,
        }
    }

    /// One item per text.
    fn text_only(texts: &[&str]) -> Self {
        let mut c = Self::empty();
        c.texts = text_table(texts);
        for i in 0..texts.len() {
            c.kinds.push(TESSL_EMBEDGEMMA2_SEGMENT_TEXT);
            c.refs.push(i as u64);
            c.item_offsets.push(i as u64 + 1);
        }
        c
    }

    /// Add an item made of `(kind, ref)` segments.
    fn item(mut self, segments: &[(u32, u64)]) -> Self {
        for &(k, r) in segments {
            self.kinds.push(k);
            self.refs.push(r);
        }
        self.item_offsets.push(self.kinds.len() as u64);
        self
    }

    fn items(&self) -> usize {
        self.item_offsets.len() - 1
    }

    fn run(&self, model: *mut TesslEmbedGemma2, trunc: u32) -> (i32, String, Vec<f32>) {
        let n = self.n_items.unwrap_or(self.items() as u64);
        let mut out = out_vec(self.items().max(1), trunc);
        let out_len = out.len() as u64;
        let (ep, el) = self
            .extension
            .as_deref()
            .map_or((ptr::null(), 0), |x| (x.as_ptr().cast::<c_char>(), x.len() as u64));
        let (s, m) = call(|e, n_e| unsafe {
            tessl_embedgemma2_embed_segments(
                model,
                ptr_or_null(&self.texts.blob),
                self.texts.blob.len() as u64,
                ptr_or_null(&self.texts.offsets[..]),
                self.texts.n(),
                ptr_or_null(&self.kinds),
                ptr_or_null(&self.refs),
                self.kinds.len() as u64,
                self.item_offsets.as_ptr(),
                n,
                self.image_source,
                ptr_or_null(&self.image.blob),
                self.image.blob.len() as u64,
                ptr_or_null(&self.image.offsets[..]),
                ptr_or_null(&self.image_dims),
                self.image.n(),
                self.audio_source,
                ptr_or_null(&self.audio.blob),
                self.audio.blob.len() as u64,
                ptr_or_null(&self.audio.offsets[..]),
                self.sampling_rate,
                self.audio.n(),
                self.video_source,
                ptr_or_null(&self.video.blob),
                self.video.blob.len() as u64,
                ptr_or_null(&self.video.offsets[..]),
                ptr_or_null(&self.video_dims),
                ptr_or_null(&self.video_fps),
                self.video.n(),
                ep,
                el,
                trunc,
                out.as_mut_ptr(),
                out_len,
                e,
                n_e,
            )
        });
        (s, m, out)
    }
}

const TEXT: u32 = TESSL_EMBEDGEMMA2_SEGMENT_TEXT;
const IMAGE: u32 = TESSL_EMBEDGEMMA2_SEGMENT_IMAGE;
const AUDIO: u32 = TESSL_EMBEDGEMMA2_SEGMENT_AUDIO;
const VIDEO: u32 = TESSL_EMBEDGEMMA2_SEGMENT_VIDEO;
const VIDEO_T: u32 = TESSL_EMBEDGEMMA2_SEGMENT_VIDEO_TIMESTAMPED;

/// Raw images, PCM and frames as one `Table` and side tables each.
struct RawImages {
    table: Table,
    dims: Vec<u32>,
    shapes: Vec<(usize, usize, usize)>,
    data: Vec<Vec<u8>>,
}

fn raw_images(shapes: &[(usize, usize, usize)]) -> RawImages {
    let data: Vec<Vec<u8>> = shapes
        .iter()
        .enumerate()
        .map(|(k, &(h, w, c))| pixels(h * w * c, 400 + k as u64))
        .collect();
    RawImages {
        table: table(&data),
        dims: shapes
            .iter()
            .flat_map(|&(h, w, c)| [h as u32, w as u32, c as u32])
            .collect(),
        shapes: shapes.to_vec(),
        data,
    }
}

impl RawImages {
    fn prepared(&self) -> Vec<PreparedImage> {
        let cfg = pre();
        self.shapes
            .iter()
            .zip(&self.data)
            .map(|(&(height, width, channels), data)| {
                preprocess_image(
                    &cfg.image,
                    &RawImage {
                        width,
                        height,
                        channels,
                        data,
                    },
                )
                .unwrap()
            })
            .collect()
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

#[test]
fn segments_of_raw_media_are_the_facade_bit_for_bit() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_media_segments_raw");
    let tok = tokenizer();
    let cfg = pre();
    for operands in [TESSL_OPERANDS_EXACT_F32, TESSL_OPERANDS_BF16] {
        let l = Loaded::new(&dir, Some(operands));
        let im = raw_images(&[(16, 16, 3), (24, 8, 1), (9, 33, 4)]);
        let pcms = [pcm(3200, 1), pcm(8000, 2)];
        let videos = [(4usize, 16usize, 16usize, 1.0f64), (3, 8, 24, 2.0)];
        let frames: Vec<Vec<u8>> = videos
            .iter()
            .enumerate()
            .map(|(k, &(n, h, w, _))| pixels(n * h * w * 3, 500 + k as u64))
            .collect();
        let mut c = SegCall::empty();
        c.texts = text_table(&["a 0", "0 b", "1: 2", "00"]);
        c.image_source = TESSL_EMBEDGEMMA2_SOURCE_RAW;
        c.image = im.table.clone();
        c.image_dims = im.dims.clone();
        c.audio_source = TESSL_EMBEDGEMMA2_SOURCE_RAW;
        c.audio = table(&pcms.iter().map(|p| f32_bytes(p)).collect::<Vec<_>>());
        c.sampling_rate = 16000;
        c.video_source = TESSL_EMBEDGEMMA2_SOURCE_RAW;
        c.video = table(&frames);
        c.video_dims = videos.iter().flat_map(|&(_, h, w, _)| [h as u32, w as u32]).collect();
        c.video_fps = videos.iter().map(|v| v.3).collect();
        // Item 0: text, image, text. Item 1: a clip alone with text. Item 2: all kinds, with a
        // medium named twice. Item 3: a timestamped video between texts.
        let c = c
            .item(&[(TEXT, 0), (IMAGE, 0), (TEXT, 1)])
            .item(&[(TEXT, 2), (AUDIO, 1)])
            .item(&[
                (IMAGE, 2),
                (VIDEO, 0),
                (TEXT, 3),
                (AUDIO, 0),
                (IMAGE, 1),
                (VIDEO, 1),
                (IMAGE, 2),
            ])
            .item(&[(TEXT, 0), (VIDEO_T, 1), (TEXT, 1)]);

        // The facade's side, from the same raw media.
        let images = im.prepared();
        let fe = AudioFrontEnd::new(&cfg.audio).unwrap();
        let clips: Vec<AudioInput> = pcms.iter().map(|p| fe.extract(p, 16000).unwrap()).collect();
        let prepared_videos: Vec<PreparedVideo> = videos
            .iter()
            .zip(&frames)
            .map(|(&(n, h, w, fps), data)| {
                let raws: Vec<RawImage<'_>> = (0..n)
                    .map(|f| RawImage::rgb(w, h, &data[f * h * w * 3..(f + 1) * h * w * 3]))
                    .collect();
                preprocess_video(&cfg.video, &raws, fps).unwrap()
            })
            .collect();
        let frame_inputs: Vec<Vec<ImageInput<'_>>> = prepared_videos
            .iter()
            .map(|v| v.frames.iter().map(PreparedImage::input).collect())
            .collect();
        let texts = ["a 0", "0 b", "1: 2", "00"];
        let segs: Vec<Vec<MmSegment<'_>>> = vec![
            vec![
                MmSegment::Text(texts[0]),
                MmSegment::Image(images[0].input()),
                MmSegment::Text(texts[1]),
            ],
            vec![MmSegment::Text(texts[2]), MmSegment::Audio(&clips[1])],
            vec![
                MmSegment::Image(images[2].input()),
                MmSegment::Video(&frame_inputs[0]),
                MmSegment::Text(texts[3]),
                MmSegment::Audio(&clips[0]),
                MmSegment::Image(images[1].input()),
                MmSegment::Video(&frame_inputs[1]),
                MmSegment::Image(images[2].input()),
            ],
            vec![
                MmSegment::Text(texts[0]),
                MmSegment::TimestampedVideo {
                    frames: &frame_inputs[1],
                    timestamps: &prepared_videos[1].timestamps,
                },
                MmSegment::Text(texts[1]),
            ],
        ];
        let items: Vec<MmItem<'_>> = segs
            .iter()
            .map(|s| MmItem::Segments {
                segments: s,
                tokenizer: &tok,
            })
            .collect();
        for trunc in [0u32, 40] {
            let want = encode(l.mm(), &items, trunc);
            let (s, m, got) = c.run(l.model, trunc);
            assert_eq!(s, TESSL_OK, "{m}");
            assert_eq!(bits(&got), bits(&want), "{operands}, truncate {trunc}");
        }
        assert!(!l.poisoned());
    }
}

#[test]
fn segments_of_files_and_bytes_in_mixed_sources_are_the_facade_bit_for_bit() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_media_segments_files");
    let tok = tokenizer();
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));

    // Images by path, clips as bytes, videos as raw frames: each modality its own source.
    let image_names = ["rgb.png", "gray.png"];
    let clip_names = ["pcm16_16k_mono.wav", "tone_44k_stereo.flac"];
    let raw_video = (4usize, 16usize, 16usize, 1.0f64);
    let frames = pixels(raw_video.0 * raw_video.1 * raw_video.2 * 3, 77);
    let mut c = SegCall::empty();
    c.texts = text_table(&["see", "and hear", "0"]);
    c.image = path_table(&image_names);
    c.audio_source = TESSL_EMBEDGEMMA2_SOURCE_BYTES;
    c.audio = bytes_table(&clip_names);
    c.video_source = TESSL_EMBEDGEMMA2_SOURCE_RAW;
    c.video = table(std::slice::from_ref(&frames));
    c.video_dims = vec![raw_video.1 as u32, raw_video.2 as u32];
    c.video_fps = vec![raw_video.3];
    let c =
        c.item(&[(TEXT, 0), (IMAGE, 1), (TEXT, 1), (AUDIO, 0)])
            .item(&[(IMAGE, 0), (AUDIO, 1), (VIDEO, 0), (TEXT, 2)]);

    let images = prepared_images(&image_names);
    let clips = prepared_clips(&clip_names);
    let cfg = pre();
    let raws: Vec<RawImage<'_>> = (0..raw_video.0)
        .map(|f| {
            let n = raw_video.1 * raw_video.2 * 3;
            RawImage::rgb(raw_video.2, raw_video.1, &frames[f * n..(f + 1) * n])
        })
        .collect();
    let video = preprocess_video(&cfg.video, &raws, raw_video.3).unwrap();
    let vframes: Vec<ImageInput<'_>> = video.frames.iter().map(PreparedImage::input).collect();
    let segs: Vec<Vec<MmSegment<'_>>> = vec![
        vec![
            MmSegment::Text("see"),
            MmSegment::Image(images[1].input()),
            MmSegment::Text("and hear"),
            MmSegment::Audio(&clips[0]),
        ],
        vec![
            MmSegment::Image(images[0].input()),
            MmSegment::Audio(&clips[1]),
            MmSegment::Video(&vframes),
            MmSegment::Text("0"),
        ],
    ];
    let items: Vec<MmItem<'_>> = segs
        .iter()
        .map(|s| MmItem::Segments {
            segments: s,
            tokenizer: &tok,
        })
        .collect();
    let want = encode(l.mm(), &items, 0);
    let (s, m, got) = c.run(l.model, 0);
    assert_eq!(s, TESSL_OK, "{m}");
    assert_eq!(bits(&got), bits(&want));

    // The same images as bytes.
    let mut as_bytes = c.clone();
    as_bytes.image_source = TESSL_EMBEDGEMMA2_SOURCE_BYTES;
    as_bytes.image = bytes_table(&image_names);
    let (s, m, got) = as_bytes.run(l.model, 0);
    assert_eq!(s, TESSL_OK, "{m}");
    assert_eq!(bits(&got), bits(&want));
    assert!(!l.poisoned());
}

#[test]
fn segments_of_a_video_file_are_the_facade_bit_for_bit() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_media_segments_video");
    let tok = tokenizer();
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    let video = prepared_video("colors.mp4");
    let vframes: Vec<ImageInput<'_>> = video.frames.iter().map(PreparedImage::input).collect();
    let images = prepared_images(&["rgb.png"]);
    let segs: Vec<Vec<MmSegment<'_>>> = vec![
        vec![
            MmSegment::Text("0"),
            MmSegment::Video(&vframes),
            MmSegment::Image(images[0].input()),
        ],
        vec![
            MmSegment::Image(images[0].input()),
            MmSegment::TimestampedVideo {
                frames: &vframes,
                timestamps: &video.timestamps,
            },
            MmSegment::Text("1: 2"),
        ],
    ];
    let items: Vec<MmItem<'_>> = segs
        .iter()
        .map(|s| MmItem::Segments {
            segments: s,
            tokenizer: &tok,
        })
        .collect();
    let want = encode(l.mm(), &items, 0);
    for (what, source, vt, ext) in [
        (
            "path",
            TESSL_EMBEDGEMMA2_SOURCE_PATHS,
            path_table(&["colors.mp4"]),
            None,
        ),
        (
            "bytes",
            TESSL_EMBEDGEMMA2_SOURCE_BYTES,
            bytes_table(&["colors.mp4"]),
            Some("mp4".to_string()),
        ),
    ] {
        let mut c = SegCall::empty();
        c.texts = text_table(&["0", "1: 2"]);
        c.image = path_table(&["rgb.png"]);
        c.video_source = source;
        c.video = vt;
        c.extension = ext;
        let c = c
            .item(&[(TEXT, 0), (VIDEO, 0), (IMAGE, 0)])
            .item(&[(IMAGE, 0), (VIDEO_T, 0), (TEXT, 1)]);
        let (s, m, got) = c.run(l.model, 0);
        assert_eq!(s, TESSL_OK, "{what}: {m}");
        assert_eq!(bits(&got), bits(&want), "{what}");
    }
    assert_eq!(media_support::temp_video_files(), 0);
}

#[test]
fn segment_refusals_leave_the_output_alone_the_handle_usable_and_the_runtime_unpoisoned() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_media_segment_refusals");
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    let im = raw_images(&[(16, 16, 3), (24, 8, 1)]);
    let mut base = SegCall::empty();
    base.texts = text_table(&["a 0", "0 b"]);
    base.image_source = TESSL_EMBEDGEMMA2_SOURCE_RAW;
    base.image = im.table.clone();
    base.image_dims = im.dims.clone();
    let base = base.item(&[(TEXT, 0), (IMAGE, 0), (TEXT, 1)]).item(&[(IMAGE, 1)]);
    let (s, m, want) = base.run(l.model, 0);
    assert_eq!(s, TESSL_OK, "{m}");

    let edit = |f: &dyn Fn(&mut SegCall)| {
        let mut c = base.clone();
        f(&mut c);
        c.run(l.model, 0)
    };
    let mut cases: Vec<Case<'_, Vec<f32>>> = Vec::new();
    cases.push(("no items", edit(&|c| c.n_items = Some(0)), "n_items is 0"));
    cases.push((
        "items past the batch",
        edit(&|c| c.n_items = Some(257)),
        "n_items is 257",
    ));
    cases.push(("an unknown kind", edit(&|c| c.kinds[1] = 9), "segment 1 has kind 9"));
    cases.push((
        "a text past its table",
        edit(&|c| c.refs[0] = 2),
        "segment 0 names text 2, but the call has 2",
    ));
    cases.push((
        "an image past its table",
        edit(&|c| c.refs[1] = 5),
        "segment 1 names image 5, but the call has 2",
    ));
    cases.push((
        "a clip when there are none",
        edit(&|c| c.kinds[1] = AUDIO),
        "segment 1 names clip 0, but the call has 0",
    ));
    cases.push((
        "a video when there are none",
        edit(&|c| c.kinds[1] = VIDEO_T),
        "segment 1 names video 0, but the call has 0",
    ));
    cases.push((
        "an image no segment names",
        edit(&|c| {
            c.refs[1] = 1;
        }),
        "image 0 is not named by any segment",
    ));
    cases.push((
        "a text no segment names",
        edit(&|c| c.refs[2] = 0),
        "text 1 is not named by any segment",
    ));
    cases.push((
        "an item with no segments",
        edit(&|c| c.item_offsets[2] = c.item_offsets[1]),
        "never empty",
    ));
    cases.push((
        "item_offsets end short",
        edit(&|c| {
            c.kinds.push(TEXT);
            c.refs.push(0);
        }),
        "item_offsets end at 4 but n_segments is 5",
    ));
    cases.push((
        "item_offsets not from 0",
        edit(&|c| c.item_offsets[0] = 1),
        "offsets[0] is 1",
    ));
    cases.push((
        "raw image offsets that disagree with dims",
        edit(&|c| c.image.offsets[1] += 1),
        "image_offsets[1]",
    ));
    cases.push((
        "raw images with null dims",
        edit(&|c| c.image_dims.clear()),
        "null image_dims",
    ));
    cases.push((
        "an image of no pixels",
        edit(&|c| c.image_dims[0] = 0),
        "image 0 is 0x16",
    ));
    cases.push((
        "an image source that is none",
        edit(&|c| c.image_source = 7),
        "image_source is 7",
    ));
    cases.push((
        "a sampling rate with files",
        edit(&|c| {
            c.audio_source = TESSL_EMBEDGEMMA2_SOURCE_PATHS;
            c.audio = path_table(&["pcm16_16k_mono.wav"]);
            c.sampling_rate = 16000;
            c.kinds.push(AUDIO);
            c.refs.push(0);
            c.item_offsets.push(c.kinds.len() as u64);
        }),
        "sampling_rate is 16000; it describes raw PCM",
    ));
    cases.push((
        "an extension with no videos",
        edit(&|c| c.extension = Some("mp4".into())),
        "an extension hint goes with encoded videos",
    ));
    cases.push((
        "blob bytes with no media of the kind",
        edit(&|c| c.audio.blob = vec![0; 4]),
        "audio_blob_len is 4 but the call has no audio media",
    ));
    cases.push((
        "raw clip bytes that are not whole samples",
        edit(&|c| {
            c.audio_source = TESSL_EMBEDGEMMA2_SOURCE_RAW;
            c.audio = table(&[vec![0u8; 7]]);
            c.sampling_rate = 16000;
            c.kinds.push(AUDIO);
            c.refs.push(0);
            c.item_offsets.push(c.kinds.len() as u64);
        }),
        "clip 0 is 7 bytes, not a whole number of 4-byte f32 samples",
    ));
    cases.push((
        "a missing image file",
        edit(&|c| {
            c.image_source = TESSL_EMBEDGEMMA2_SOURCE_PATHS;
            c.image = table(&[
                b"/no/such/a.png".to_vec(),
                path_of("rgb.png").to_str().unwrap().as_bytes().to_vec(),
            ]);
        }),
        "file 0: cannot read /no/such/a.png",
    ));
    cases.push((
        "text bytes with no texts",
        edit(&|c| {
            c.texts = Table {
                blob: b"x".to_vec(),
                offsets: vec![0],
            };
            c.kinds[0] = IMAGE;
            c.refs[0] = 1;
            c.kinds[2] = IMAGE;
            c.refs[2] = 0;
        }),
        "text_bytes_len is 1 but n_texts is 0",
    ));
    for (what, (status, message, out), want) in &cases {
        assert_eq!(*status, TESSL_ERR, "{what}: {message}");
        assert!(message.contains(want), "{what}: {message}");
        assert!(
            out.iter().all(|&v| v == SENTINEL),
            "{what}: a refused call wrote its output"
        );
    }

    // A medium named by thousands of segments is refused by the call's bound, before one image
    // is resized: 4000 pictures of 280 padded patches are past MAX_CALL_PATCHES.
    let mut many = SegCall::empty();
    many.texts = text_table(&["0"]);
    many.image_source = TESSL_EMBEDGEMMA2_SOURCE_RAW;
    many.image = table(&[pixels(8 * 8 * 3, 1)]);
    many.image_dims = vec![8, 8, 3];
    let refs: Vec<(u32, u64)> = std::iter::repeat_n((IMAGE, 0), 4000).chain([(TEXT, 0)]).collect();
    let many = many.item(&refs);
    let (s, m, out) = many.run(l.model, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("MAX_CALL_PATCHES"), "{m}");
    assert!(untouched(&out));

    // The segment count and the text are bounded before they size anything.
    let mut c = base.clone();
    let (s, m, out) = {
        let mut out = out_vec(2, 0);
        let (s, m) = call(|e, n_e| unsafe {
            tessl_embedgemma2_embed_segments(
                l.model,
                c.texts.blob.as_ptr(),
                c.texts.blob.len() as u64,
                c.texts.offsets.as_ptr(),
                c.texts.n(),
                c.kinds.as_ptr(),
                c.refs.as_ptr(),
                TESSL_EMBEDGEMMA2_MAX_SEGMENTS + 1,
                c.item_offsets.as_ptr(),
                2,
                c.image_source,
                c.image.blob.as_ptr(),
                c.image.blob.len() as u64,
                c.image.offsets.as_ptr(),
                c.image_dims.as_ptr(),
                c.image.n(),
                1,
                ptr::null(),
                0,
                ptr::null(),
                0,
                0,
                1,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                ptr::null(),
                0,
                0,
                out.as_mut_ptr(),
                out.len() as u64,
                e,
                n_e,
            )
        });
        (s, m, out)
    };
    c.n_items = None;
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("n_segments is 65537"), "{m}");
    assert!(untouched(&out));

    assert!(!l.poisoned(), "a refusal poisoned the runtime");
    let (s, m, got) = base.run(l.model, 0);
    assert_eq!(s, TESSL_OK, "the handle is unusable afterwards: {m}");
    assert_eq!(bits(&got), bits(&want));
}

// ---------------------------------------------------------------------------
// Threads and precision
// ---------------------------------------------------------------------------

#[test]
fn a_model_used_from_another_thread_is_refused_by_the_media_entries() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_media_thread");
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    let model = l.model as usize;
    let results = std::thread::spawn(move || {
        let model = model as *mut TesslEmbedGemma2;
        let t = path_table(&["rgb.png"]);
        let mut counts = [u64::MAX; 2];
        let dims = [8u32, 8];
        let samples = [3200u64];
        let frames = [4u64];
        let fps = [1.0f64];
        vec![
            ("embed_image_sources", {
                let (s, m, o) = embed_images_from(model, 1, &t, 0);
                assert!(untouched(&o));
                (s, m)
            }),
            ("embed_audio_sources", {
                let (s, m, o) = embed_clips_from(model, 1, &t, 0);
                assert!(untouched(&o));
                (s, m)
            }),
            ("embed_video_sources", {
                let (s, m, o) = embed_videos_from(model, 1, &t, None, 0, 0);
                assert!(untouched(&o));
                (s, m)
            }),
            ("image_soft_token_count", {
                call(|e, n| unsafe {
                    tessl_embedgemma2_image_soft_token_count(model, dims.as_ptr(), 1, counts.as_mut_ptr(), 1, e, n)
                })
            }),
            ("audio_soft_token_count", {
                call(|e, n| unsafe {
                    tessl_embedgemma2_audio_soft_token_count(
                        model,
                        samples.as_ptr(),
                        1,
                        16000,
                        counts.as_mut_ptr(),
                        1,
                        e,
                        n,
                    )
                })
            }),
            ("video_soft_token_count", {
                call(|e, n| unsafe {
                    tessl_embedgemma2_video_soft_token_count(
                        model,
                        dims.as_ptr(),
                        frames.as_ptr(),
                        fps.as_ptr(),
                        1,
                        counts.as_mut_ptr(),
                        2,
                        e,
                        n,
                    )
                })
            }),
            ("embed_segments", {
                let (s, m, o) = SegCall::text_only(&["12"]).run(model, 0);
                assert!(untouched(&o));
                (s, m)
            }),
        ]
    })
    .join()
    .unwrap();
    for (name, (s, m)) in results {
        assert_eq!(s, TESSL_ERR, "{name}");
        assert!(m.contains("thread other than the one that created it"), "{name}: {m}");
    }
    assert_eq!(SegCall::text_only(&["12"]).run(l.model, 0).0, TESSL_OK);
}

/// Resets the runtime's relaxed precision when a test ends, however it ends.
struct Relaxed(std::sync::Arc<tessl::GpuRuntime>);

impl Relaxed {
    fn on(model: *mut TesslEmbedGemma2) -> Self {
        let rt = unsafe { test_embedgemma2_runtime(model) }.clone();
        rt.set_relaxed_precision(true);
        Self(rt)
    }
}

impl Drop for Relaxed {
    fn drop(&mut self) {
        self.0.set_relaxed_precision(false);
    }
}

#[test]
fn a_bf16_handle_accepts_a_relaxed_runtime_for_the_media_entries_and_an_exact_one_refuses_it() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_media_relaxed");
    let imgs = path_table(&["rgb.png", "gray.png"]);
    let clps = path_table(&["pcm16_16k_mono.wav"]);
    let seg = SegCall::text_only(&["0"]);
    let seg = {
        let mut s = seg;
        s.image = path_table(&["rgb.png"]);
        s.item(&[(TEXT, 0), (IMAGE, 0)])
    };
    let run_all = |model| {
        (
            embed_images_from(model, 1, &imgs, 0),
            embed_clips_from(model, 1, &clps, 0),
            seg.run(model, 0),
        )
    };
    let bf = Loaded::new(&dir, Some(TESSL_OPERANDS_BF16));
    let strict = run_all(bf.model);
    for (what, (s, m, _)) in [("images", &strict.0), ("clips", &strict.1), ("segments", &strict.2)] {
        assert_eq!(*s, TESSL_OK, "bf16 {what}: {m}");
    }
    {
        let _relaxed = Relaxed::on(bf.model);
        let relaxed = run_all(bf.model);
        assert_eq!(relaxed.0 .0, TESSL_OK, "{}", relaxed.0 .1);
        assert_eq!(bits(&relaxed.0 .2), bits(&strict.0 .2));
        assert_eq!(bits(&relaxed.1 .2), bits(&strict.1 .2));
        assert_eq!(bits(&relaxed.2 .2), bits(&strict.2 .2));
    }
    let ex = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    let _relaxed = Relaxed::on(ex.model);
    let refused = run_all(ex.model);
    for (what, (s, m, o)) in [("images", refused.0), ("clips", refused.1), ("segments", refused.2)] {
        assert_eq!(s, TESSL_ERR, "{what}");
        assert!(m.contains("relaxed precision"), "{what}: {m}");
        assert!(untouched(&o), "{what}: a refused call wrote its output");
    }
    let _ = MAX_BATCH;
}
