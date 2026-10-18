//! The C ABI's entries that run the snapshot's own front ends
//! (`tessl_embedgemma2_{embed_texts, tokenize, decode, prompt_*, embed_images,
//! embed_audio, embed_videos, embed_media}`), called the way a foreign caller
//! calls them, on the tiny multimodal checkpoint of `tests/common/embedgemma2_mm.rs`
//! written to a directory under `target/` with the tiny tokenizer, its sidecars
//! and a `processor_config.json` that fits the tiny towers.
//!
//! What this file pins is the boundary, as `tests/capi_embedgemma2.rs` does for
//! the id entries: every refusal is a status and a message that leaves the
//! caller's output untouched and the handle usable, each declared length is
//! bounded before it sizes or reads anything (the over-limit cases pass
//! buffers far smaller than the length they declare, which would fault if
//! anything read them first), a front end whose file is missing is refused by
//! name and tried again on the next call, and a successful call is the Rust
//! facade (`Tokenizer`, `preprocess_*`, `EmbedGemma2Multimodal::encode`) bit for
//! bit on the same weights. The real checkpoint is
//! `python/tests/test_embedgemma2_raw.py`.
//!
//! The tests with a null handle need no device (they are in the hosted
//! selection, `scripts/hosted_tests.txt`); the rest create a runtime.

mod common;

use std::ffi::{c_char, CStr};
use std::path::{Path, PathBuf};
use std::ptr;

use common::embedgemma2_mm::{
    checkpoint_bytes, config_json, configs, processor_config_json, sentence_transformers_json,
    timestamp_tokenizer_json, TOKENIZER_CONFIG,
};
use common::{gpu_lock, SplitMix};
use tessl::capi::{
    tessl_runtime_free, tessl_runtime_new, TesslRuntime, TESSL_ERR, TESSL_OK, TESSL_OPERANDS_BF16,
    TESSL_OPERANDS_EXACT_F32,
};
use tessl::capi_embedgemma2::{
    tessl_embedgemma2_decode, tessl_embedgemma2_default_prompt, tessl_embedgemma2_embed_audio,
    tessl_embedgemma2_embed_images, tessl_embedgemma2_embed_media, tessl_embedgemma2_embed_texts,
    tessl_embedgemma2_embed_videos, tessl_embedgemma2_free, tessl_embedgemma2_load, tessl_embedgemma2_load_multimodal,
    tessl_embedgemma2_prompt_count, tessl_embedgemma2_prompt_get, tessl_embedgemma2_prompt_name,
    tessl_embedgemma2_set_max_tokens, tessl_embedgemma2_tokenize, test_embedgemma2_model, test_embedgemma2_multimodal,
    test_embedgemma2_runtime, TesslEmbedGemma2, TESSL_EMBEDGEMMA2_MAX_CLIP_SAMPLES, TESSL_EMBEDGEMMA2_MAX_DECODE_IDS,
    TESSL_EMBEDGEMMA2_MAX_NAME_BYTES, TESSL_EMBEDGEMMA2_MAX_TOKENIZE_TEXTS,
};
use tessl::embedgemma2::MAX_BATCH;
use tessl::embedgemma2_audio::AudioInput;
use tessl::embedgemma2_mm::MmItem;
use tessl::embedgemma2_pre::{preprocess_image, preprocess_video, AudioFrontEnd, PreConfig, RawImage};
use tessl::embedgemma2_tok::{Tokenizer, MAX_TEXT_BYTES};
use tessl::embedgemma2_vision::{ImageInput, MAX_CALL_PATCHES};

const ERR_LEN: usize = 512;
/// The tiny checkpoint's `embedding_dim`.
const DIM: usize = 96;
/// Written to the output of every call that must leave it alone.
const SENTINEL: f32 = -7.25;
/// The ids `common::embedgemma2_mm::config_json` carries.
const BOS: u32 = 2;
const EOS: u32 = 1;
const BOI: u32 = 40;
const EOI: u32 = 41;
const BOA: u32 = 42;
const EOA: u32 = 43;
const IMAGE: u32 = 60;
const AUDIO: u32 = 61;
const VIDEO: u32 = 62;

fn msg(err: &[c_char]) -> String {
    unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn untouched(out: &[f32]) -> bool {
    out.iter().all(|&v| v == SENTINEL)
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

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

/// Texts laid end to end, as the C entries take them.
struct Texts {
    bytes: Vec<u8>,
    offsets: Vec<u64>,
}

fn texts(t: &[&str]) -> Texts {
    let mut out = Texts {
        bytes: Vec::new(),
        offsets: vec![0],
    };
    for s in t {
        out.bytes.extend_from_slice(s.as_bytes());
        out.offsets.push(out.bytes.len() as u64);
    }
    out
}

/// Images laid end to end: the bytes, and the `[n, 3]` dims table.
struct Images {
    data: Vec<u8>,
    dims: Vec<u32>,
    shapes: Vec<(usize, usize, usize)>,
}

fn images(shapes: &[(usize, usize, usize)]) -> Images {
    let mut im = Images {
        data: Vec::new(),
        dims: Vec::new(),
        shapes: shapes.to_vec(),
    };
    for (k, &(h, w, c)) in shapes.iter().enumerate() {
        im.data.extend(pixels(h * w * c, 100 + k as u64));
        im.dims.extend([h as u32, w as u32, c as u32]);
    }
    im
}

impl Images {
    fn raws(&self) -> Vec<RawImage<'_>> {
        let mut at = 0;
        self.shapes
            .iter()
            .map(|&(height, width, channels)| {
                let n = height * width * channels;
                let data = &self.data[at..at + n];
                at += n;
                RawImage {
                    width,
                    height,
                    channels,
                    data,
                }
            })
            .collect()
    }
}

/// Clips laid end to end.
struct Clips {
    samples: Vec<f32>,
    offsets: Vec<u64>,
}

fn clips(lens: &[usize]) -> Clips {
    let mut c = Clips {
        samples: Vec::new(),
        offsets: vec![0],
    };
    for (k, &n) in lens.iter().enumerate() {
        c.samples.extend(pcm(n, 200 + k as u64));
        c.offsets.push(c.samples.len() as u64);
    }
    c
}

impl Clips {
    fn slices(&self) -> Vec<&[f32]> {
        self.offsets
            .windows(2)
            .map(|w| &self.samples[w[0] as usize..w[1] as usize])
            .collect()
    }
}

/// Videos laid end to end: `(frames, height, width, fps)` each.
struct Videos {
    frames: Vec<u8>,
    dims: Vec<u32>,
    offsets: Vec<u64>,
    fps: Vec<f64>,
    shapes: Vec<(usize, usize, usize)>,
}

fn videos(spec: &[(usize, usize, usize, f64)]) -> Videos {
    let mut v = Videos {
        frames: Vec::new(),
        dims: Vec::new(),
        offsets: vec![0],
        fps: Vec::new(),
        shapes: Vec::new(),
    };
    for (k, &(n, h, w, fps)) in spec.iter().enumerate() {
        v.frames.extend(pixels(n * h * w * 3, 300 + k as u64));
        v.dims.extend([h as u32, w as u32]);
        v.offsets.push(v.offsets[k] + n as u64);
        v.fps.push(fps);
        v.shapes.push((n, h, w));
    }
    v
}

impl Videos {
    /// Video `k`'s frames as the facade takes them.
    fn raws(&self, k: usize) -> Vec<RawImage<'_>> {
        let skip: usize = self.shapes[..k].iter().map(|&(n, h, w)| n * h * w * 3).sum();
        let (n, height, width) = self.shapes[k];
        (0..n)
            .map(|f| {
                let at = skip + f * height * width * 3;
                RawImage::rgb(width, height, &self.frames[at..at + height * width * 3])
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Raw calls: every argument is the caller's.
// ---------------------------------------------------------------------------

fn call(f: impl FnOnce(*mut c_char, usize) -> i32) -> (i32, String) {
    let mut err = [0 as c_char; ERR_LEN];
    let s = f(err.as_mut_ptr(), ERR_LEN);
    (s, msg(&err))
}

#[allow(clippy::too_many_arguments)]
fn embed_texts_raw(
    model: *mut TesslEmbedGemma2,
    bytes: *const u8,
    bytes_len: u64,
    offsets: *const u64,
    n: u64,
    name: *const c_char,
    name_len: u64,
    trunc: u32,
    out: *mut f32,
    out_len: u64,
) -> (i32, String) {
    call(|e, n_e| unsafe {
        tessl_embedgemma2_embed_texts(
            model, bytes, bytes_len, offsets, n, name, name_len, trunc, out, out_len, e, n_e,
        )
    })
}

/// `embed_texts` on well-formed arguments: the status, the message and the output.
fn embed_texts(model: *mut TesslEmbedGemma2, t: &Texts, prompt: Option<&str>, trunc: u32) -> (i32, String, Vec<f32>) {
    let n = t.offsets.len() - 1;
    let dim = if trunc == 0 { DIM } else { trunc as usize };
    let mut out = vec![SENTINEL; n * dim];
    let (np, nl) = prompt.map_or((ptr::null(), 0), |p| (p.as_ptr().cast::<c_char>(), p.len() as u64));
    let (s, m) = embed_texts_raw(
        model,
        t.bytes.as_ptr(),
        t.bytes.len() as u64,
        t.offsets.as_ptr(),
        n as u64,
        np,
        nl,
        trunc,
        out.as_mut_ptr(),
        out.len() as u64,
    );
    (s, m, out)
}

/// The two-call tokenize: `(ids, offsets)`.
fn tokenize(model: *mut TesslEmbedGemma2, t: &Texts, prompt: Option<&str>) -> Result<(Vec<u32>, Vec<u64>), String> {
    let n = (t.offsets.len() - 1) as u64;
    let (np, nl) = prompt.map_or((ptr::null(), 0), |p| (p.as_ptr().cast::<c_char>(), p.len() as u64));
    let mut total = 0u64;
    let (s, m) = call(|e, n_e| unsafe {
        tessl_embedgemma2_tokenize(
            model,
            t.bytes.as_ptr(),
            t.bytes.len() as u64,
            t.offsets.as_ptr(),
            n,
            np,
            nl,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            &mut total,
            e,
            n_e,
        )
    });
    if s != TESSL_OK {
        return Err(m);
    }
    let mut ids = vec![u32::MAX; total as usize];
    let mut cuts = vec![u64::MAX; n as usize + 1];
    let mut again = 0u64;
    let (s, m) = call(|e, n_e| unsafe {
        tessl_embedgemma2_tokenize(
            model,
            t.bytes.as_ptr(),
            t.bytes.len() as u64,
            t.offsets.as_ptr(),
            n,
            np,
            nl,
            ids.as_mut_ptr(),
            ids.len() as u64,
            cuts.as_mut_ptr(),
            &mut again,
            e,
            n_e,
        )
    });
    if s != TESSL_OK {
        return Err(m);
    }
    assert_eq!(again, total);
    Ok((ids, cuts))
}

/// A string-valued query through `ask(buf, buf_len, &mut needed)`, size first.
fn string_query(mut ask: impl FnMut(*mut u8, u64, *mut u64, *mut c_char, usize) -> i32) -> Result<String, String> {
    let mut needed = 0u64;
    let (s, m) = call(|e, n| ask(ptr::null_mut(), 0, &mut needed, e, n));
    if s != TESSL_OK {
        return Err(m);
    }
    let mut buf = vec![0xAAu8; needed as usize];
    let mut again = 0u64;
    let (s, m) = call(|e, n| ask(buf.as_mut_ptr(), buf.len() as u64, &mut again, e, n));
    if s != TESSL_OK {
        return Err(m);
    }
    assert_eq!(again, needed);
    Ok(String::from_utf8(buf).expect("UTF-8 back"))
}

fn prompt_count(model: *mut TesslEmbedGemma2) -> Result<u32, String> {
    let mut n = u32::MAX;
    let (s, m) = call(|e, n_e| unsafe { tessl_embedgemma2_prompt_count(model, &mut n, e, n_e) });
    (s == TESSL_OK).then_some(n).ok_or(m)
}

fn prompt_name(model: *mut TesslEmbedGemma2, index: u32) -> Result<String, String> {
    string_query(|b, bl, need, e, n| unsafe { tessl_embedgemma2_prompt_name(model, index, b, bl, need, e, n) })
}

fn prompt_get(model: *mut TesslEmbedGemma2, name: &str) -> Result<String, String> {
    string_query(|b, bl, need, e, n| unsafe {
        tessl_embedgemma2_prompt_get(model, name.as_ptr().cast(), name.len() as u64, b, bl, need, e, n)
    })
}

fn default_prompt(model: *mut TesslEmbedGemma2) -> Result<i32, String> {
    let mut i = i32::MIN;
    let (s, m) = call(|e, n| unsafe { tessl_embedgemma2_default_prompt(model, &mut i, e, n) });
    (s == TESSL_OK).then_some(i).ok_or(m)
}

fn decode(model: *mut TesslEmbedGemma2, ids: &[u32], skip: u32) -> Result<String, String> {
    string_query(|b, bl, need, e, n| unsafe {
        tessl_embedgemma2_decode(model, ids.as_ptr(), ids.len() as u64, skip, b, bl, need, e, n)
    })
}

fn embed_images(model: *mut TesslEmbedGemma2, im: &Images, trunc: u32) -> (i32, String, Vec<f32>) {
    let n = im.shapes.len();
    let dim = if trunc == 0 { DIM } else { trunc as usize };
    let mut out = vec![SENTINEL; n * dim];
    let (s, m) = call(|e, n_e| unsafe {
        tessl_embedgemma2_embed_images(
            model,
            im.data.as_ptr(),
            im.data.len() as u64,
            im.dims.as_ptr(),
            n as u64,
            trunc,
            out.as_mut_ptr(),
            out.len() as u64,
            e,
            n_e,
        )
    });
    (s, m, out)
}

fn embed_audio(model: *mut TesslEmbedGemma2, c: &Clips, rate: u32, trunc: u32) -> (i32, String, Vec<f32>) {
    let n = c.offsets.len() - 1;
    let dim = if trunc == 0 { DIM } else { trunc as usize };
    let mut out = vec![SENTINEL; n * dim];
    let (s, m) = call(|e, n_e| unsafe {
        tessl_embedgemma2_embed_audio(
            model,
            c.samples.as_ptr(),
            c.samples.len() as u64,
            c.offsets.as_ptr(),
            n as u64,
            rate,
            trunc,
            out.as_mut_ptr(),
            out.len() as u64,
            e,
            n_e,
        )
    });
    (s, m, out)
}

fn embed_videos(model: *mut TesslEmbedGemma2, v: &Videos, timestamps: u32, trunc: u32) -> (i32, String, Vec<f32>) {
    let n = v.fps.len();
    let dim = if trunc == 0 { DIM } else { trunc as usize };
    let mut out = vec![SENTINEL; n * dim];
    let (s, m) = call(|e, n_e| unsafe {
        tessl_embedgemma2_embed_videos(
            model,
            v.frames.as_ptr(),
            v.frames.len() as u64,
            v.dims.as_ptr(),
            v.offsets.as_ptr(),
            v.fps.as_ptr(),
            n as u64,
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

/// One `embed_media` item: its ids and which of the call's media it takes
/// (as index ranges over the call's images, clips and videos).
struct MediaItem {
    ids: Vec<u32>,
    images: std::ops::Range<usize>,
    clips: std::ops::Range<usize>,
    videos: std::ops::Range<usize>,
}

fn cuts(items: &[MediaItem], f: impl Fn(&MediaItem) -> std::ops::Range<usize>) -> Vec<u64> {
    let mut c = vec![0u64];
    for it in items {
        let r = f(it);
        assert_eq!(
            r.start as u64,
            *c.last().unwrap(),
            "test bug: media ranges are consecutive"
        );
        c.push(r.end as u64);
    }
    c
}

fn embed_media(
    model: *mut TesslEmbedGemma2,
    items: &[MediaItem],
    im: &Images,
    c: &Clips,
    v: &Videos,
    trunc: u32,
) -> (i32, String, Vec<f32>) {
    let ids: Vec<u32> = items.iter().flat_map(|i| i.ids.iter().copied()).collect();
    let mut id_cuts = vec![0u64];
    for it in items {
        id_cuts.push(id_cuts.last().unwrap() + it.ids.len() as u64);
    }
    let (image_cuts, clip_cuts, video_cuts) = (
        cuts(items, |i| i.images.clone()),
        cuts(items, |i| i.clips.clone()),
        cuts(items, |i| i.videos.clone()),
    );
    let dim = if trunc == 0 { DIM } else { trunc as usize };
    let mut out = vec![SENTINEL; items.len() * dim];
    let (s, m) = call(|e, n_e| unsafe {
        tessl_embedgemma2_embed_media(
            model,
            ids.as_ptr(),
            ids.len() as u64,
            id_cuts.as_ptr(),
            items.len() as u64,
            im.data.as_ptr(),
            im.data.len() as u64,
            im.dims.as_ptr(),
            im.shapes.len() as u64,
            image_cuts.as_ptr(),
            c.samples.as_ptr(),
            c.samples.len() as u64,
            c.offsets.as_ptr(),
            (c.offsets.len() - 1) as u64,
            clip_cuts.as_ptr(),
            16000,
            v.frames.as_ptr(),
            v.frames.len() as u64,
            v.dims.as_ptr(),
            v.offsets.as_ptr(),
            v.fps.as_ptr(),
            v.fps.len() as u64,
            video_cuts.as_ptr(),
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
// No device: a null handle
// ---------------------------------------------------------------------------

#[test]
fn a_null_handle_is_refused_by_every_front_end_entry() {
    let null: *mut TesslEmbedGemma2 = ptr::null_mut();
    let t = texts(&["12"]);
    let im = images(&[(8, 8, 3)]);
    let c = clips(&[1600]);
    let v = videos(&[(2, 8, 8, 1.0)]);
    let mut out = [SENTINEL; DIM];
    let mut n_u32 = 0u32;
    let mut n_i32 = 0i32;
    let mut n_u64 = 0u64;
    let (mut ids, mut cuts_out) = ([0u32; 4], [0u64; 2]);
    let item = [MediaItem {
        ids: vec![BOS, EOS],
        images: 0..0,
        clips: 0..0,
        videos: 0..0,
    }];
    let results = vec![
        ("prompt_count", {
            let (s, m) = call(|e, n| unsafe { tessl_embedgemma2_prompt_count(null, &mut n_u32, e, n) });
            (s, m)
        }),
        ("prompt_name", {
            call(|e, n| unsafe { tessl_embedgemma2_prompt_name(null, 0, ptr::null_mut(), 0, &mut n_u64, e, n) })
        }),
        ("prompt_get", {
            call(|e, n| unsafe {
                tessl_embedgemma2_prompt_get(null, b"q".as_ptr().cast(), 1, ptr::null_mut(), 0, &mut n_u64, e, n)
            })
        }),
        ("default_prompt", {
            call(|e, n| unsafe { tessl_embedgemma2_default_prompt(null, &mut n_i32, e, n) })
        }),
        ("tokenize", {
            call(|e, n| unsafe {
                tessl_embedgemma2_tokenize(
                    null,
                    t.bytes.as_ptr(),
                    2,
                    t.offsets.as_ptr(),
                    1,
                    ptr::null(),
                    0,
                    ids.as_mut_ptr(),
                    4,
                    cuts_out.as_mut_ptr(),
                    &mut n_u64,
                    e,
                    n,
                )
            })
        }),
        ("decode", {
            call(|e, n| unsafe {
                tessl_embedgemma2_decode(null, ids.as_ptr(), 1, 1, ptr::null_mut(), 0, &mut n_u64, e, n)
            })
        }),
        ("embed_texts", {
            embed_texts_raw(
                null,
                t.bytes.as_ptr(),
                2,
                t.offsets.as_ptr(),
                1,
                ptr::null(),
                0,
                0,
                out.as_mut_ptr(),
                DIM as u64,
            )
        }),
        ("embed_images", {
            let (s, m, o) = embed_images(null, &im, 0);
            assert!(untouched(&o));
            (s, m)
        }),
        ("embed_audio", {
            let (s, m, o) = embed_audio(null, &c, 16000, 0);
            assert!(untouched(&o));
            (s, m)
        }),
        ("embed_videos", {
            let (s, m, o) = embed_videos(null, &v, 0, 0);
            assert!(untouched(&o));
            (s, m)
        }),
        ("embed_media", {
            let (s, m, o) = embed_media(null, &item, &images(&[]), &clips(&[]), &videos(&[]), 0);
            assert!(untouched(&o));
            (s, m)
        }),
    ];
    assert_eq!(results.len(), 11);
    for (name, (status, message)) in results {
        assert_eq!(status, TESSL_ERR, "{name}");
        assert!(message.contains("null embedgemma2 model handle"), "{name}: {message}");
    }
    assert!(untouched(&out), "a refused call wrote its output");
    assert_eq!((n_u32, n_i32, n_u64), (0, 0, 0), "a refused query wrote its output");
    assert_eq!(ids, [0; 4]);
}

// ---------------------------------------------------------------------------
// The tiny snapshot on disk
// ---------------------------------------------------------------------------

/// Which of the sidecars a snapshot directory gets.
#[derive(Clone)]
struct Sidecars {
    tokenizer: bool,
    tokenizer_config: bool,
    prompts: Option<String>,
    processor: Option<String>,
}

impl Sidecars {
    fn full() -> Self {
        Self {
            tokenizer: true,
            tokenizer_config: true,
            prompts: Some(sentence_transformers_json("null")),
            processor: Some(processor_config_json(false, 70)),
        }
    }
}

/// A fresh snapshot directory `name` under `target/`: the tiny multimodal
/// checkpoint and the sidecars `s` names. (One directory per test, so tests
/// that edit theirs do not race.)
fn snapshot(name: &str, s: &Sidecars) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the snapshot directory");
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    std::fs::write(
        dir.join("model.safetensors"),
        checkpoint_bytes(&configs(), 2026, tessl::gemm::GemmOperands::Bf16),
    )
    .unwrap();
    if s.tokenizer {
        std::fs::write(dir.join("tokenizer.json"), timestamp_tokenizer_json()).unwrap();
    }
    if s.tokenizer_config {
        std::fs::write(dir.join("tokenizer_config.json"), TOKENIZER_CONFIG).unwrap();
    }
    if let Some(p) = &s.prompts {
        std::fs::write(dir.join("config_sentence_transformers.json"), p).unwrap();
    }
    if let Some(p) = &s.processor {
        std::fs::write(dir.join("processor_config.json"), p).unwrap();
    }
    dir
}

/// The tokenizer the snapshot's files make, built the way the facade's tests do.
fn tokenizer(prompts: Option<&str>) -> Tokenizer {
    let t = Tokenizer::from_json(&timestamp_tokenizer_json())
        .unwrap()
        .with_tokenizer_config(TOKENIZER_CONFIG)
        .unwrap();
    match prompts {
        Some(p) => t.with_prompts(p).unwrap(),
        None => t,
    }
}

fn pre() -> PreConfig {
    PreConfig::from_json(&processor_config_json(false, 70)).unwrap()
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

    fn mm(&self) -> &'static tessl::embedgemma2_mm::EmbedGemma2Multimodal {
        unsafe { test_embedgemma2_multimodal(self.model) }.expect("a multimodal handle")
    }
}

impl Drop for Loaded {
    fn drop(&mut self) {
        assert_eq!(unsafe { tessl_embedgemma2_free(self.model) }, TESSL_OK);
        assert_eq!(unsafe { tessl_runtime_free(self.rt) }, TESSL_OK);
    }
}

const SAMPLE_TEXTS: [&str; 7] = [
    "",
    "1: 2:3",
    "00 0 00 1000",
    "h\u{e9}llo w\u{f6}rld \u{4e2d}\u{6587} \u{1f389}",
    "<bos>12",
    "<|image>00",
    "0123456789:0123456789 0123456789",
];

// ---------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------

#[test]
fn texts_through_the_c_entry_are_the_tokenizer_and_encode_bit_for_bit() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_raw_text", &Sidecars::full());
    let tok = tokenizer(Some(&sentence_transformers_json("null")));
    // A text-only handle and a multimodal one agree with the Rust facade.
    for operands in [None, Some(TESSL_OPERANDS_EXACT_F32), Some(TESSL_OPERANDS_BF16)] {
        let l = Loaded::new(&dir, operands);
        let model = unsafe { test_embedgemma2_model(l.model) };
        for prompt in [None, Some("query"), Some("document")] {
            for trunc in [0u32, 40, 1] {
                let t = texts(&SAMPLE_TEXTS);
                let (s, m, got) = embed_texts(l.model, &t, prompt, trunc);
                assert_eq!(s, TESSL_OK, "{operands:?} {prompt:?} truncate_dim {trunc}: {m}");
                let want = model
                    .encode_texts(&tok, prompt, &SAMPLE_TEXTS, (trunc != 0).then_some(trunc))
                    .unwrap();
                assert_eq!(bits(&got), bits(&want.embeddings), "{operands:?} {prompt:?} {trunc}");
            }
        }
        // One text alone, and the batch limit.
        let one = texts(&["1: 2"]);
        assert_eq!(embed_texts(l.model, &one, None, 0).0, TESSL_OK);
        let many: Vec<&str> = (0..MAX_BATCH).map(|_| "12 0").collect();
        let (s, m, got) = embed_texts(l.model, &texts(&many), None, 0);
        assert_eq!(s, TESSL_OK, "{m}");
        assert_eq!(got.len(), MAX_BATCH * DIM);
    }
}

#[test]
fn the_default_prompt_applies_when_no_name_is_given() {
    let _g = gpu_lock();
    let side = Sidecars {
        prompts: Some(sentence_transformers_json("\"document\"")),
        ..Sidecars::full()
    };
    let dir = snapshot("capi_embedgemma2_raw_default_prompt", &side);
    let tok = tokenizer(Some(&sentence_transformers_json("\"document\"")));
    let l = Loaded::new(&dir, None);
    let model = unsafe { test_embedgemma2_model(l.model) };
    assert_eq!(default_prompt(l.model), Ok(1), "document is the second prompt");
    let t = texts(&SAMPLE_TEXTS);
    let (s, m, with_default) = embed_texts(l.model, &t, None, 0);
    assert_eq!(s, TESSL_OK, "{m}");
    let want = model.encode_texts(&tok, Some("document"), &SAMPLE_TEXTS, None).unwrap();
    assert_eq!(bits(&with_default), bits(&want.embeddings));
    let plain = model.encode_texts(&tok, None, &SAMPLE_TEXTS, None).unwrap();
    assert_ne!(
        bits(&with_default),
        bits(&plain.embeddings),
        "the default prompt changed nothing"
    );
    // A named prompt overrides the default, and tokenize applies it too.
    let (s, _, named) = embed_texts(l.model, &t, Some("query"), 0);
    assert_eq!(s, TESSL_OK);
    let want = model.encode_texts(&tok, Some("query"), &SAMPLE_TEXTS, None).unwrap();
    assert_eq!(bits(&named), bits(&want.embeddings));
    let (ids, cuts_out) = tokenize(l.model, &texts(&["12"]), None).unwrap();
    assert_eq!(ids, tok.encode_with_prompt(Some("document"), "12").unwrap());
    assert_eq!(cuts_out, [0, ids.len() as u64]);
}

#[test]
fn tokenize_and_decode_are_the_tokenizer() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_raw_tokenize", &Sidecars::full());
    let tok = tokenizer(Some(&sentence_transformers_json("null")));
    let l = Loaded::new(&dir, None);
    let t = texts(&SAMPLE_TEXTS);
    for prompt in [None, Some("query"), Some("document")] {
        let (ids, cuts_out) = tokenize(l.model, &t, prompt).unwrap();
        let want = tok.encode_batch_with_prompt(prompt, &SAMPLE_TEXTS).unwrap();
        assert_eq!(cuts_out.len(), want.len() + 1);
        for (i, w) in want.iter().enumerate() {
            assert_eq!(
                &ids[cuts_out[i] as usize..cuts_out[i + 1] as usize],
                w.as_slice(),
                "{prompt:?} text {i}"
            );
        }
        // Decoding the ids of each text, with and without the special tokens.
        for (i, w) in want.iter().enumerate() {
            for skip in [0u32, 1] {
                assert_eq!(
                    decode(l.model, w, skip).unwrap(),
                    tok.decode_with(w, skip == 1),
                    "{prompt:?} text {i} skip {skip}"
                );
            }
        }
    }
    // No ids decode to the empty string; an id past the vocabulary is skipped.
    assert_eq!(decode(l.model, &[], 1).unwrap(), "");
    assert_eq!(decode(l.model, &[4, 4_000_000, 5], 1).unwrap(), tok.decode(&[4, 5]));

    // A buffer one id short is refused with the size written, and untouched.
    let one = texts(&["1: 2:3"]);
    let need = tok.encode("1: 2:3").unwrap().len() as u64;
    let (mut ids, mut cuts_out, mut total) = (vec![77u32; need as usize - 1], [77u64; 2], 0u64);
    let (s, m) = call(|e, n| unsafe {
        tessl_embedgemma2_tokenize(
            l.model,
            one.bytes.as_ptr(),
            one.bytes.len() as u64,
            one.offsets.as_ptr(),
            1,
            ptr::null(),
            0,
            ids.as_mut_ptr(),
            ids.len() as u64,
            cuts_out.as_mut_ptr(),
            &mut total,
            e,
            n,
        )
    });
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains(&format!("{need} are needed")), "{m}");
    assert_eq!(total, need, "the size is written so the caller can retry");
    assert!(
        ids.iter().all(|&v| v == 77) && cuts_out == [77; 2],
        "a refused tokenize wrote its buffers"
    );
    // A null ids_out with a capacity is not a size query.
    let (s, m) = call(|e, n| unsafe {
        tessl_embedgemma2_tokenize(
            l.model,
            one.bytes.as_ptr(),
            one.bytes.len() as u64,
            one.offsets.as_ptr(),
            1,
            ptr::null(),
            0,
            ptr::null_mut(),
            5,
            ptr::null_mut(),
            &mut total,
            e,
            n,
        )
    });
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("size query"), "{m}");
    // The same short-buffer rule for decode.
    let text_ids = tok.encode("1: 2:3").unwrap();
    let (mut buf, mut need_bytes) = ([0x55u8; 1], 0u64);
    let (s, m) = call(|e, n| unsafe {
        tessl_embedgemma2_decode(
            l.model,
            text_ids.as_ptr(),
            text_ids.len() as u64,
            0,
            buf.as_mut_ptr(),
            1,
            &mut need_bytes,
            e,
            n,
        )
    });
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("are needed") && need_bytes > 1, "{m}");
    assert_eq!(buf, [0x55]);
    // Bad arguments.
    for (what, ids_ptr, n_ids, skip, want_msg) in [
        ("skip 2", text_ids.as_ptr(), 1u64, 2u32, "expected 0 or 1"),
        (
            "too many ids",
            text_ids.as_ptr(),
            TESSL_EMBEDGEMMA2_MAX_DECODE_IDS + 1,
            1,
            "expected at most",
        ),
        ("null ids", ptr::null(), 3, 1, "null ids"),
    ] {
        let (s, m) = call(|e, n| unsafe {
            tessl_embedgemma2_decode(l.model, ids_ptr, n_ids, skip, ptr::null_mut(), 0, &mut need_bytes, e, n)
        });
        assert_eq!(s, TESSL_ERR, "{what}");
        assert!(m.contains(want_msg), "{what}: {m}");
    }
    // A null buffer with a length, and a null length cell.
    let (s, m) = call(|e, n| unsafe {
        tessl_embedgemma2_decode(
            l.model,
            text_ids.as_ptr(),
            2,
            1,
            ptr::null_mut(),
            4,
            &mut need_bytes,
            e,
            n,
        )
    });
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("null buf"), "{m}");
    let (s, m) = call(|e, n| unsafe {
        tessl_embedgemma2_decode(
            l.model,
            text_ids.as_ptr(),
            2,
            1,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            e,
            n,
        )
    });
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("null out_len"), "{m}");
    assert_eq!(
        decode(l.model, &text_ids, 1).unwrap(),
        tok.decode(&text_ids),
        "usable afterwards"
    );
}

#[test]
fn the_prompt_entries_list_get_and_refuse() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_raw_prompts", &Sidecars::full());
    let l = Loaded::new(&dir, None);
    assert_eq!(prompt_count(l.model), Ok(2));
    assert_eq!(prompt_name(l.model, 0).unwrap(), "query");
    assert_eq!(prompt_name(l.model, 1).unwrap(), "document");
    assert_eq!(prompt_get(l.model, "query").unwrap(), "00 ");
    assert_eq!(prompt_get(l.model, "document").unwrap(), "1: ");
    assert_eq!(default_prompt(l.model), Ok(-1), "the checkpoint's default is null");
    let e = prompt_name(l.model, 2).unwrap_err();
    assert!(e.contains("index 2 is past the snapshot's 2 prompts"), "{e}");
    let e = prompt_get(l.model, "nope").unwrap_err();
    assert!(e.contains("no prompt named \"nope\"") && e.contains("query"), "{e}");
    // A short buffer is refused with the length written; the buffer is untouched.
    let (mut buf, mut need) = ([0x55u8; 3], 0u64);
    let (s, m) =
        call(|e, n| unsafe { tessl_embedgemma2_prompt_name(l.model, 1, buf.as_mut_ptr(), 3, &mut need, e, n) });
    assert_eq!((s, need, buf), (TESSL_ERR, 8, [0x55; 3]), "{m}");
    assert!(m.contains("8 are needed"), "{m}");
    // A buffer that fits is copied into without a terminator.
    let mut big = [0x55u8; 12];
    let (s, m) =
        call(|e, n| unsafe { tessl_embedgemma2_prompt_name(l.model, 1, big.as_mut_ptr(), 12, &mut need, e, n) });
    assert_eq!(s, TESSL_OK, "{m}");
    assert_eq!((&big[..8], &big[8..]), (&b"document"[..], &[0x55u8; 4][..]));
    // Name arguments: empty, too long, not UTF-8, null with a length, null cell.
    let long = vec![b'a'; TESSL_EMBEDGEMMA2_MAX_NAME_BYTES as usize + 1];
    for (what, p, len, want_msg) in [
        ("empty", b"q".as_ptr(), 0u64, "expected 1..="),
        ("long", long.as_ptr(), long.len() as u64, "expected 1..="),
        ("not UTF-8", [0xffu8, 0xfe].as_ptr(), 2, "not UTF-8"),
        ("null", ptr::null(), 3, "null name"),
    ] {
        let (s, m) = call(|e, n| unsafe {
            tessl_embedgemma2_prompt_get(l.model, p.cast(), len, ptr::null_mut(), 0, &mut need, e, n)
        });
        assert_eq!(s, TESSL_ERR, "{what}");
        assert!(m.contains(want_msg), "{what}: {m}");
    }
    let (s, m) = call(|e, n| unsafe { tessl_embedgemma2_prompt_count(l.model, ptr::null_mut(), e, n) });
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("null out"), "{m}");
    let (s, m) =
        call(|e, n| unsafe { tessl_embedgemma2_prompt_name(l.model, 0, ptr::null_mut(), 0, ptr::null_mut(), e, n) });
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("null out_len"), "{m}");
    assert_eq!(prompt_count(l.model), Ok(2), "usable afterwards");
}

#[test]
fn a_snapshot_without_prompts_has_none_and_a_prompt_name_is_refused() {
    let _g = gpu_lock();
    let side = Sidecars {
        prompts: None,
        ..Sidecars::full()
    };
    let dir = snapshot("capi_embedgemma2_raw_no_prompts", &side);
    let l = Loaded::new(&dir, None);
    assert_eq!(prompt_count(l.model), Ok(0));
    assert_eq!(default_prompt(l.model), Ok(-1));
    let e = prompt_get(l.model, "query").unwrap_err();
    assert!(e.contains("no prompt named"), "{e}");
    let (s, m, out) = embed_texts(l.model, &texts(&["12"]), Some("query"), 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("no prompt named \"query\""), "{m}");
    assert!(untouched(&out));
    // Texts without a prompt still embed.
    let (s, m, _) = embed_texts(l.model, &texts(&["12"]), None, 0);
    assert_eq!(s, TESSL_OK, "{m}");
}

#[test]
fn a_missing_front_end_file_is_refused_by_name_and_read_again_on_the_next_call() {
    let _g = gpu_lock();
    let side = Sidecars {
        tokenizer: false,
        tokenizer_config: false,
        prompts: None,
        processor: None,
    };
    let dir = snapshot("capi_embedgemma2_raw_late", &side);
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    // Ids still embed on a snapshot with no front end at all.
    let ids = [BOS, 5, 6, EOS];
    let (offsets, mut out) = ([0u64, 4], vec![SENTINEL; DIM]);
    let (s, m) = call(|e, n| unsafe {
        tessl::capi_embedgemma2::tessl_embedgemma2_encode(
            l.model,
            ids.as_ptr(),
            4,
            offsets.as_ptr(),
            1,
            0,
            out.as_mut_ptr(),
            DIM as u64,
            e,
            n,
        )
    });
    assert_eq!(s, TESSL_OK, "{m}");

    let t = texts(&["12"]);
    let (s, m, o) = embed_texts(l.model, &t, None, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("tokenizer.json"), "{m}");
    assert!(untouched(&o));
    let (s, m, o) = embed_images(l.model, &images(&[(16, 16, 3)]), 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("processor_config.json"), "{m}");
    assert!(untouched(&o));
    let (s, m, o) = embed_audio(l.model, &clips(&[3200]), 16000, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("processor_config.json"), "{m}");
    assert!(untouched(&o));

    // The tokenizer.json alone is not enough: tokenizer_config.json is named next.
    std::fs::write(dir.join("tokenizer.json"), timestamp_tokenizer_json()).unwrap();
    let (s, m, _) = embed_texts(l.model, &t, None, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("tokenizer_config.json"), "{m}");
    std::fs::write(dir.join("tokenizer_config.json"), TOKENIZER_CONFIG).unwrap();
    // The prompts file is optional.
    let (s, m, o) = embed_texts(l.model, &t, None, 0);
    assert_eq!(s, TESSL_OK, "the file appeared and the next call read it: {m}");
    assert!(!untouched(&o));
    // A prompts file that is there and wrong is an error, not an absence.
    std::fs::write(dir.join("config_sentence_transformers.json"), "{\"prompts\": 3}").unwrap();
    assert_eq!(
        prompt_count(l.model),
        Ok(0),
        "the tokenizer loaded before the file was bad is kept"
    );

    std::fs::write(dir.join("processor_config.json"), processor_config_json(false, 70)).unwrap();
    let (s, m, o) = embed_images(l.model, &images(&[(16, 16, 3)]), 0);
    assert_eq!(s, TESSL_OK, "{m}");
    assert!(!untouched(&o));
}

#[test]
fn a_snapshot_whose_prompts_file_is_bad_is_refused_by_name() {
    let _g = gpu_lock();
    let side = Sidecars {
        prompts: Some("{\"prompts\": {\"q\": 3}}".to_string()),
        ..Sidecars::full()
    };
    let dir = snapshot("capi_embedgemma2_raw_bad_prompts", &side);
    let l = Loaded::new(&dir, None);
    let (s, m, o) = embed_texts(l.model, &texts(&["12"]), None, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(
        m.contains("config_sentence_transformers.json") && m.contains("not a string"),
        "{m}"
    );
    assert!(untouched(&o));
    let side = Sidecars {
        prompts: Some(sentence_transformers_json("\"missing\"")),
        ..Sidecars::full()
    };
    let dir = snapshot("capi_embedgemma2_raw_bad_default", &side);
    let l2 = Loaded::new(&dir, None);
    let e = default_prompt(l2.model).unwrap_err();
    assert!(e.contains("default_prompt_name"), "{e}");
}

#[test]
fn text_refusals_leave_the_output_alone_and_the_handle_usable() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_raw_text_refusals", &Sidecars::full());
    let l = Loaded::new(&dir, None);
    let t = texts(&["12", "0:1"]);
    let want = embed_texts(l.model, &t, None, 0);
    assert_eq!(want.0, TESSL_OK, "{}", want.1);
    let good = |out: &mut Vec<f32>| {
        embed_texts_raw(
            l.model,
            t.bytes.as_ptr(),
            t.bytes.len() as u64,
            t.offsets.as_ptr(),
            2,
            ptr::null(),
            0,
            0,
            out.as_mut_ptr(),
            out.len() as u64,
        )
    };
    let tiny = [0u8; 4];
    let tiny_off = [0u64; 3];
    #[allow(clippy::type_complexity)]
    let cases: Vec<(&str, Box<dyn Fn(*mut f32) -> (i32, String)>, &str)> = vec![
        (
            "n_texts 0",
            Box::new(|o| {
                embed_texts_raw(
                    l.model,
                    t.bytes.as_ptr(),
                    5,
                    t.offsets.as_ptr(),
                    0,
                    ptr::null(),
                    0,
                    0,
                    o,
                    0,
                )
            }),
            "n_texts is 0",
        ),
        (
            "n_texts past the batch",
            Box::new(|o| {
                embed_texts_raw(
                    l.model,
                    tiny.as_ptr(),
                    0,
                    tiny_off.as_ptr(),
                    257,
                    ptr::null(),
                    0,
                    0,
                    o,
                    257 * 96,
                )
            }),
            "n_texts is 257",
        ),
        (
            "bytes_len past the limit, buffers tiny",
            Box::new(|o| {
                embed_texts_raw(
                    l.model,
                    tiny.as_ptr(),
                    MAX_TEXT_BYTES as u64,
                    tiny_off.as_ptr(),
                    2,
                    ptr::null(),
                    0,
                    0,
                    o,
                    2 * 96,
                )
            }),
            "less than",
        ),
        (
            "offsets not from 0",
            Box::new(|o| {
                embed_texts_raw(
                    l.model,
                    t.bytes.as_ptr(),
                    5,
                    [1u64, 3, 5].as_ptr(),
                    2,
                    ptr::null(),
                    0,
                    0,
                    o,
                    2 * 96,
                )
            }),
            "text_offsets[0] is 1",
        ),
        (
            "offsets decreasing",
            Box::new(|o| {
                embed_texts_raw(
                    l.model,
                    t.bytes.as_ptr(),
                    5,
                    [0u64, 4, 3].as_ptr(),
                    2,
                    ptr::null(),
                    0,
                    0,
                    o,
                    2 * 96,
                )
            }),
            "is below",
        ),
        (
            "offsets end short",
            Box::new(|o| {
                embed_texts_raw(
                    l.model,
                    t.bytes.as_ptr(),
                    5,
                    [0u64, 2, 3].as_ptr(),
                    2,
                    ptr::null(),
                    0,
                    0,
                    o,
                    2 * 96,
                )
            }),
            "end at 3 but text_bytes_len is 5",
        ),
        (
            "not UTF-8",
            Box::new(|o| {
                embed_texts_raw(
                    l.model,
                    [0xffu8, 0xfe].as_ptr(),
                    2,
                    [0u64, 1, 2].as_ptr(),
                    2,
                    ptr::null(),
                    0,
                    0,
                    o,
                    192,
                )
            }),
            "text 0 is not UTF-8",
        ),
        (
            "null bytes",
            Box::new(|o| {
                embed_texts_raw(
                    l.model,
                    ptr::null(),
                    5,
                    t.offsets.as_ptr(),
                    2,
                    ptr::null(),
                    0,
                    0,
                    o,
                    2 * 96,
                )
            }),
            "null text_bytes",
        ),
        (
            "null offsets",
            Box::new(|o| {
                embed_texts_raw(
                    l.model,
                    t.bytes.as_ptr(),
                    5,
                    ptr::null(),
                    2,
                    ptr::null(),
                    0,
                    0,
                    o,
                    2 * 96,
                )
            }),
            "null text_offsets",
        ),
        (
            "null out",
            Box::new(|_| {
                embed_texts_raw(
                    l.model,
                    t.bytes.as_ptr(),
                    5,
                    t.offsets.as_ptr(),
                    2,
                    ptr::null(),
                    0,
                    0,
                    ptr::null_mut(),
                    2 * 96,
                )
            }),
            "null out",
        ),
        (
            "out_len wrong",
            Box::new(|o| {
                embed_texts_raw(
                    l.model,
                    t.bytes.as_ptr(),
                    5,
                    t.offsets.as_ptr(),
                    2,
                    ptr::null(),
                    0,
                    0,
                    o,
                    96,
                )
            }),
            "expected exactly 192",
        ),
        (
            "truncate_dim past the width",
            Box::new(|o| {
                embed_texts_raw(
                    l.model,
                    t.bytes.as_ptr(),
                    5,
                    t.offsets.as_ptr(),
                    2,
                    ptr::null(),
                    0,
                    97,
                    o,
                    2 * 97,
                )
            }),
            "truncate_dim 97",
        ),
        (
            "unknown prompt",
            Box::new(|o| {
                embed_texts_raw(
                    l.model,
                    t.bytes.as_ptr(),
                    5,
                    t.offsets.as_ptr(),
                    2,
                    b"nope".as_ptr().cast(),
                    4,
                    0,
                    o,
                    2 * 96,
                )
            }),
            "no prompt named \"nope\"",
        ),
        (
            "prompt name length 0",
            Box::new(|o| {
                embed_texts_raw(
                    l.model,
                    t.bytes.as_ptr(),
                    5,
                    t.offsets.as_ptr(),
                    2,
                    b"q".as_ptr().cast(),
                    0,
                    0,
                    o,
                    2 * 96,
                )
            }),
            "prompt_name_len is 0",
        ),
        (
            "prompt name too long, buffer tiny",
            Box::new(|o| {
                embed_texts_raw(
                    l.model,
                    t.bytes.as_ptr(),
                    5,
                    t.offsets.as_ptr(),
                    2,
                    tiny.as_ptr().cast(),
                    TESSL_EMBEDGEMMA2_MAX_NAME_BYTES + 1,
                    0,
                    o,
                    2 * 96,
                )
            }),
            "prompt_name_len is 257",
        ),
        (
            "null prompt name with a length",
            Box::new(|o| {
                embed_texts_raw(
                    l.model,
                    t.bytes.as_ptr(),
                    5,
                    t.offsets.as_ptr(),
                    2,
                    ptr::null(),
                    3,
                    0,
                    o,
                    2 * 96,
                )
            }),
            "null prompt_name",
        ),
    ];
    for (what, f, want_msg) in &cases {
        let mut out = vec![SENTINEL; 2 * DIM];
        let (s, m) = f(out.as_mut_ptr());
        assert_eq!(s, TESSL_ERR, "{what}: {m}");
        assert!(m.contains(want_msg), "{what}: {m}");
        assert!(untouched(&out), "{what}: a refused call wrote its output");
        let mut again = vec![SENTINEL; 2 * DIM];
        let (s, m) = good(&mut again);
        assert_eq!(s, TESSL_OK, "{what}: the handle is unusable afterwards: {m}");
        assert_eq!(bits(&again), bits(&want.2), "{what}");
    }
    // A text over the token limit is named; the limit is the model's.
    let long = texts(&["12", "0123456789 0123456789 0123456789"]);
    let (s, m) = call(|e, n| unsafe { tessl_embedgemma2_set_max_tokens(l.model, 12, e, n) });
    assert_eq!(s, TESSL_OK, "{m}");
    let (s, m, o) = embed_texts(l.model, &long, None, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("text 1 has") && m.contains("the limit is 12"), "{m}");
    assert!(untouched(&o));
    let (s, m) = call(|e, n| unsafe { tessl_embedgemma2_set_max_tokens(l.model, 8192, e, n) });
    assert_eq!(s, TESSL_OK, "{m}");
    assert_eq!(embed_texts(l.model, &long, None, 0).0, TESSL_OK);
}

#[test]
fn tokenize_refuses_its_bad_arguments() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_raw_tokenize_refusals", &Sidecars::full());
    let l = Loaded::new(&dir, None);
    let t = texts(&["12"]);
    let mut total = 0u64;
    let tiny = [0u8; 4];
    for (what, n, len, want_msg) in [
        ("zero texts", 0u64, 2u64, "n_texts is 0"),
        (
            "too many texts",
            TESSL_EMBEDGEMMA2_MAX_TOKENIZE_TEXTS + 1,
            2,
            "expected 1..=",
        ),
        ("bytes_len past the limit", 1, MAX_TEXT_BYTES as u64, "less than"),
    ] {
        let (s, m) = call(|e, n_e| unsafe {
            tessl_embedgemma2_tokenize(
                l.model,
                tiny.as_ptr(),
                len,
                [0u64, 0].as_ptr(),
                n,
                ptr::null(),
                0,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                &mut total,
                e,
                n_e,
            )
        });
        assert_eq!(s, TESSL_ERR, "{what}");
        assert!(m.contains(want_msg), "{what}: {m}");
    }
    let (s, m) = call(|e, n| unsafe {
        tessl_embedgemma2_tokenize(
            l.model,
            t.bytes.as_ptr(),
            2,
            t.offsets.as_ptr(),
            1,
            ptr::null(),
            0,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            e,
            n,
        )
    });
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("null n_ids_out"), "{m}");
    // A buffer with no offsets cell, and offsets that are not 0..n.
    let mut ids = [0u32; 8];
    let (s, m) = call(|e, n| unsafe {
        tessl_embedgemma2_tokenize(
            l.model,
            t.bytes.as_ptr(),
            2,
            t.offsets.as_ptr(),
            1,
            ptr::null(),
            0,
            ids.as_mut_ptr(),
            8,
            ptr::null_mut(),
            &mut total,
            e,
            n,
        )
    });
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("null id_offsets_out"), "{m}");
    assert_eq!(ids, [0; 8], "a refused tokenize wrote its ids");
    assert!(tokenize(l.model, &t, None).is_ok(), "usable afterwards");
    let (s, m) = call(|e, n| unsafe {
        tessl_embedgemma2_tokenize(
            l.model,
            t.bytes.as_ptr(),
            2,
            [1u64, 2].as_ptr(),
            1,
            ptr::null(),
            0,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            &mut total,
            e,
            n,
        )
    });
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("text_offsets[0] is 1"), "{m}");
    // A placeholder in a text is refused naming the text.
    let e = tokenize(l.model, &texts(&["12", "a <|image|> b"]), None).unwrap_err();
    assert!(e.contains("text 1") && e.contains("placeholder"), "{e}");
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

#[test]
fn raw_images_through_the_c_entry_are_preprocess_and_encode_bit_for_bit() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_raw_images", &Sidecars::full());
    let cfg = pre();
    // RGB that needs no resize, gray, gray+alpha, RGBA, and two that resize.
    let im = images(&[
        (64, 56, 3),
        (40, 40, 1),
        (32, 32, 2),
        (100, 50, 4),
        (13, 77, 3),
        (8, 8, 3),
    ]);
    for operands in [TESSL_OPERANDS_EXACT_F32, TESSL_OPERANDS_BF16] {
        let l = Loaded::new(&dir, Some(operands));
        let mm = l.mm();
        let prepared: Vec<_> = im
            .raws()
            .iter()
            .map(|r| preprocess_image(&cfg.image, r).unwrap())
            .collect();
        let items: Vec<MmItem<'_>> = prepared.iter().map(|p| MmItem::Image(p.input())).collect();
        for trunc in [0u32, 40, 1] {
            let (s, m, got) = embed_images(l.model, &im, trunc);
            assert_eq!(s, TESSL_OK, "{operands} truncate_dim {trunc}: {m}");
            let want = mm.encode(&items, (trunc != 0).then_some(trunc)).unwrap();
            assert_eq!(bits(&got), bits(&want.embeddings), "{operands} truncate_dim {trunc}");
        }
        // An image alone is its row of the batch to the call's batch tolerance.
        let one = images(&[(100, 50, 4)]);
        let (s, m, alone) = embed_images(l.model, &one, 0);
        assert_eq!(s, TESSL_OK, "{m}");
        let norm = alone.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm {norm}");
    }
}

#[test]
fn image_refusals_leave_the_output_alone_and_the_handle_usable() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_raw_image_refusals", &Sidecars::full());
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    let im = images(&[(16, 16, 3), (24, 8, 1)]);
    let (s, m, want) = embed_images(l.model, &im, 0);
    assert_eq!(s, TESSL_OK, "{m}");
    let run = |data: &[u8], len: u64, dims: *const u32, n: u64, trunc: u32, out_len: u64| {
        let mut out = vec![SENTINEL; 2 * DIM];
        let (s, m) = call(|e, n_e| unsafe {
            tessl_embedgemma2_embed_images(
                l.model,
                data.as_ptr(),
                len,
                dims,
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
    let bytes = im.data.len() as u64;
    let mut dims = im.dims.clone();
    let cases: Vec<(&str, i32, String, Vec<f32>)> = {
        let mut v = Vec::new();
        let mut push = |what: &'static str, r: (i32, String, Vec<f32>)| v.push((what, r.0, r.1, r.2));
        push("n_images 0", run(&im.data, bytes, im.dims.as_ptr(), 0, 0, 0));
        push(
            "n_images past the batch",
            run(&im.data, 0, im.dims.as_ptr(), 257, 0, 257 * 96),
        );
        push(
            "pixels_len short",
            run(&im.data, bytes - 1, im.dims.as_ptr(), 2, 0, 192),
        );
        push("pixels_len long", run(&im.data, bytes + 1, im.dims.as_ptr(), 2, 0, 192));
        push("null dims", run(&im.data, bytes, ptr::null(), 2, 0, 192));
        push("out_len wrong", run(&im.data, bytes, im.dims.as_ptr(), 2, 0, 96));
        push(
            "truncate_dim past the width",
            run(&im.data, bytes, im.dims.as_ptr(), 2, 97, 2 * 97),
        );
        dims[0] = 0;
        push("height 0", run(&im.data, bytes, dims.as_ptr(), 2, 0, 192));
        dims[0] = 16;
        dims[2] = 5;
        push("5 channels", run(&im.data, bytes, dims.as_ptr(), 2, 0, 192));
        dims[2] = 0;
        push("0 channels", run(&im.data, bytes, dims.as_ptr(), 2, 0, 192));
        // Declared sizes far past the data and the pixel limit, with a tiny buffer.
        let huge = [70_000u32, 70_000, 1, 8, 8, 1];
        push(
            "over the pixel limit",
            run(&im.data[..64], 64, huge.as_ptr(), 2, 0, 192),
        );
        v
    };
    for (what, s, m, out) in cases {
        assert_eq!(s, TESSL_ERR, "{what}: {m}");
        assert!(untouched(&out), "{what}: a refused call wrote its output");
        let _ = m;
    }
    let (s, m, out) = run(&im.data, bytes, im.dims.as_ptr(), 2, 0, 192);
    assert_eq!(s, TESSL_OK, "the handle is unusable afterwards: {m}");
    assert_eq!(bits(&out), bits(&want));
    // The reasons, for the ones a message must name.
    let named = |what: &str, r: (i32, String, Vec<f32>), pat: &str| {
        assert_eq!(r.0, TESSL_ERR, "{what}");
        assert!(r.1.contains(pat), "{what}: {}", r.1);
    };
    let mut d = im.dims.clone();
    d[0] = 0;
    named(
        "height 0",
        run(&im.data, bytes, d.as_ptr(), 2, 0, 192),
        "image 0 is 0x16",
    );
    d[0] = 16;
    d[2] = 5;
    named("5 channels", run(&im.data, bytes, d.as_ptr(), 2, 0, 192), "5 channels");
    named(
        "short",
        run(&im.data, bytes - 1, im.dims.as_ptr(), 2, 0, 192),
        "expected 960",
    );
    let huge = [70_000u32, 70_000, 1, 8, 8, 1];
    named("huge", run(&im.data[..64], 64, huge.as_ptr(), 2, 0, 192), "pixel limit");
    named(
        "batch",
        run(&im.data, 0, im.dims.as_ptr(), 257, 0, 257 * 96),
        "n_images is 257",
    );
}

#[test]
fn the_per_call_patch_bound_is_checked_before_any_image_is_resized() {
    let _g = gpu_lock();
    // 1120 soft tokens at a 2x2 pooling cell is 4480 padded patches an image:
    // 256 images are past MAX_CALL_PATCHES, however small they are.
    let side = Sidecars {
        processor: Some(processor_config_json(false, 1120)),
        ..Sidecars::full()
    };
    let dir = snapshot("capi_embedgemma2_raw_patch_bound", &side);
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    let n = MAX_BATCH;
    assert!(n * 4480 > MAX_CALL_PATCHES);
    let shapes = vec![(1, 1, 1); n];
    let im = images(&shapes);
    let (s, m, out) = embed_images(l.model, &im, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("MAX_CALL_PATCHES"), "{m}");
    assert!(untouched(&out));
    // Half as many fit the bound (and are tiny enough to run).
    let ok = images(&[(8, 8, 3); 8]);
    let (s, m, _) = embed_images(l.model, &ok, 0);
    assert_eq!(s, TESSL_OK, "{m}");
}

#[test]
fn a_processor_that_does_not_fit_the_towers_is_refused_by_name() {
    let _g = gpu_lock();
    let wrong_patch = processor_config_json(false, 70).replace("\"patch_size\": 4", "\"patch_size\": 8");
    let side = Sidecars {
        processor: Some(wrong_patch),
        ..Sidecars::full()
    };
    let dir = snapshot("capi_embedgemma2_raw_wrong_patch", &side);
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    let (s, m, o) = embed_images(l.model, &images(&[(16, 16, 3)]), 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("image_processor") && m.contains("patch_size 8"), "{m}");
    assert!(untouched(&o));
    let (s, m, o) = embed_videos(l.model, &videos(&[(2, 16, 16, 1.0)]), 0, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("video_processor") && m.contains("patch_size 8"), "{m}");
    assert!(untouched(&o));
    let wrong_mel = processor_config_json(false, 70).replace("\"feature_size\": 32", "\"feature_size\": 128");
    let side = Sidecars {
        processor: Some(wrong_mel),
        ..Sidecars::full()
    };
    let dir = snapshot("capi_embedgemma2_raw_wrong_mel", &side);
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    let (s, m, o) = embed_audio(l.model, &clips(&[3200]), 16000, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(
        m.contains("feature_size 128") && m.contains("audio tower takes 32"),
        "{m}"
    );
    assert!(untouched(&o));
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

#[test]
fn raw_audio_through_the_c_entry_is_the_front_end_and_encode_bit_for_bit() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_raw_audio", &Sidecars::full());
    let cfg = pre();
    let c = clips(&[1600, 4000, 16000, 2500]);
    let front = AudioFrontEnd::new(&cfg.audio).unwrap();
    for operands in [TESSL_OPERANDS_EXACT_F32, TESSL_OPERANDS_BF16] {
        let l = Loaded::new(&dir, Some(operands));
        let mm = l.mm();
        let feats: Vec<AudioInput> = c.slices().iter().map(|p| front.extract(p, 16000).unwrap()).collect();
        let items: Vec<MmItem<'_>> = feats.iter().map(MmItem::Audio).collect();
        for trunc in [0u32, 40, 1] {
            let (s, m, got) = embed_audio(l.model, &c, 16000, trunc);
            assert_eq!(s, TESSL_OK, "{operands} truncate_dim {trunc}: {m}");
            let want = mm.encode(&items, (trunc != 0).then_some(trunc)).unwrap();
            assert_eq!(bits(&got), bits(&want.embeddings), "{operands} truncate_dim {trunc}");
        }
    }
}

#[test]
fn audio_refusals_leave_the_output_alone_and_the_handle_usable() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_raw_audio_refusals", &Sidecars::full());
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    let c = clips(&[3200, 4800]);
    let (s, m, want) = embed_audio(l.model, &c, 16000, 0);
    assert_eq!(s, TESSL_OK, "{m}");
    let run = |samples: &[f32], len: u64, offsets: &[u64], n: u64, rate: u32, trunc: u32, out_len: u64| {
        let mut out = vec![SENTINEL; 2 * DIM];
        let (s, m) = call(|e, n_e| unsafe {
            tessl_embedgemma2_embed_audio(
                l.model,
                if samples.is_empty() {
                    ptr::null()
                } else {
                    samples.as_ptr()
                },
                len,
                offsets.as_ptr(),
                n,
                rate,
                trunc,
                out.as_mut_ptr(),
                out_len,
                e,
                n_e,
            )
        });
        (s, m, out)
    };
    let len = c.samples.len() as u64;
    let mut nan = c.samples.clone();
    nan[7] = f32::NAN;
    let cases = [
        (
            "n_clips 0",
            run(&c.samples, len, &c.offsets, 0, 16000, 0, 0),
            "n_clips is 0",
        ),
        (
            "n_clips past the batch",
            run(&c.samples, len, &c.offsets, 257, 16000, 0, 257 * 96),
            "n_clips is 257",
        ),
        (
            "another rate",
            run(&c.samples, len, &c.offsets, 2, 8000, 0, 192),
            "does not resample",
        ),
        ("rate 0", run(&c.samples, len, &c.offsets, 2, 0, 0, 192), "0 Hz"),
        ("a NaN sample", run(&nan, len, &c.offsets, 2, 16000, 0, 192), "sample 7"),
        (
            "offsets end short",
            run(&c.samples, len - 1, &c.offsets, 2, 16000, 0, 192),
            "end at 8000 but samples_len is 7999",
        ),
        (
            "an empty clip",
            run(&c.samples, len, &[0, 3200, 3200], 2, 16000, 0, 192),
            "does not exceed",
        ),
        (
            "offsets not from 0",
            run(&c.samples, len, &[1, 3200, 8000], 2, 16000, 0, 192),
            "offsets[0] is 1",
        ),
        (
            "a clip past the sample limit, buffer tiny",
            run(
                &c.samples[..8],
                TESSL_EMBEDGEMMA2_MAX_CLIP_SAMPLES + 1,
                &[0, TESSL_EMBEDGEMMA2_MAX_CLIP_SAMPLES + 1],
                1,
                16000,
                0,
                96,
            ),
            "the limit is",
        ),
        (
            "too short for a frame",
            run(&c.samples[..100], 100, &[0, 100], 1, 16000, 0, 96),
            "",
        ),
        (
            "out_len wrong",
            run(&c.samples, len, &c.offsets, 2, 16000, 0, 96),
            "expected exactly 192",
        ),
        (
            "truncate_dim past the width",
            run(&c.samples, len, &c.offsets, 2, 16000, 97, 2 * 97),
            "truncate_dim 97",
        ),
    ];
    for (what, (s, m, out), want_msg) in cases {
        assert_eq!(s, TESSL_ERR, "{what}: {m}");
        assert!(m.contains(want_msg), "{what}: {m}");
        if !what.contains("too short") {
            assert!(untouched(&out), "{what}: a refused call wrote its output");
        }
    }
    let (s, m, out) = run(&c.samples, len, &c.offsets, 2, 16000, 0, 192);
    assert_eq!(s, TESSL_OK, "the handle is unusable afterwards: {m}");
    assert_eq!(bits(&out), bits(&want));
}

// ---------------------------------------------------------------------------
// Video
// ---------------------------------------------------------------------------

#[test]
fn raw_video_through_the_c_entry_is_preprocess_and_encode_bit_for_bit() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_raw_video", &Sidecars::full());
    let cfg = pre();
    let tok = tokenizer(Some(&sentence_transformers_json("null")));
    // Four frames at 2 fps (two seconds: sampled to the cap at 1 fps), and
    // 8 frames at 4 fps of another size.
    let v = videos(&[(4, 24, 32, 2.0), (8, 40, 16, 4.0)]);
    for operands in [TESSL_OPERANDS_EXACT_F32, TESSL_OPERANDS_BF16] {
        let l = Loaded::new(&dir, Some(operands));
        let mm = l.mm();
        let prepared: Vec<_> = (0..2)
            .map(|k| preprocess_video(&cfg.video, &v.raws(k), v.fps[k]).unwrap())
            .collect();
        assert!(
            prepared.iter().all(|p| p.frames.len() >= 2),
            "test bug: the videos have frames"
        );
        let inputs: Vec<Vec<ImageInput<'_>>> = prepared
            .iter()
            .map(|p| p.frames.iter().map(|f| f.input()).collect())
            .collect();
        let plain: Vec<MmItem<'_>> = inputs.iter().map(|f| MmItem::Video(f)).collect();
        let stamped: Vec<MmItem<'_>> = prepared
            .iter()
            .zip(&inputs)
            .map(|(p, f)| MmItem::TimestampedVideo {
                frames: f,
                timestamps: &p.timestamps,
                tokenizer: &tok,
            })
            .collect();
        for trunc in [0u32, 40] {
            for (flag, items) in [(0u32, &plain), (1, &stamped)] {
                let (s, m, got) = embed_videos(l.model, &v, flag, trunc);
                assert_eq!(s, TESSL_OK, "{operands} timestamps {flag} truncate_dim {trunc}: {m}");
                let want = mm.encode(items, (trunc != 0).then_some(trunc)).unwrap();
                assert_eq!(
                    bits(&got),
                    bits(&want.embeddings),
                    "{operands} timestamps {flag} {trunc}"
                );
            }
        }
        let a = embed_videos(l.model, &v, 0, 0).2;
        let b = embed_videos(l.model, &v, 1, 0).2;
        assert_ne!(bits(&a), bits(&b), "timestamps changed nothing");
    }
}

#[test]
fn video_refusals_leave_the_output_alone_and_the_handle_usable() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_raw_video_refusals", &Sidecars::full());
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    let v = videos(&[(2, 16, 16, 1.0), (3, 8, 24, 2.0)]);
    let (s, m, want) = embed_videos(l.model, &v, 0, 0);
    assert_eq!(s, TESSL_OK, "{m}");
    let run = |frames: &[u8], len: u64, dims: &[u32], offsets: &[u64], fps: &[f64], n: u64, ts: u32, out_len: u64| {
        let mut out = vec![SENTINEL; 2 * DIM];
        let (s, m) = call(|e, n_e| unsafe {
            tessl_embedgemma2_embed_videos(
                l.model,
                frames.as_ptr(),
                len,
                dims.as_ptr(),
                offsets.as_ptr(),
                fps.as_ptr(),
                n,
                ts,
                0,
                out.as_mut_ptr(),
                out_len,
                e,
                n_e,
            )
        });
        (s, m, out)
    };
    let len = v.frames.len() as u64;
    let go = |dims: &[u32], offsets: &[u64], fps: &[f64], n: u64, ts: u32| {
        run(&v.frames, len, dims, offsets, fps, n, ts, 192)
    };
    let cases = [
        ("n_videos 0", go(&v.dims, &v.offsets, &v.fps, 0, 0), "n_videos is 0"),
        ("timestamps 2", go(&v.dims, &v.offsets, &v.fps, 2, 2), "timestamps is 2"),
        (
            "fps 0",
            go(&v.dims, &v.offsets, &[0.0, 2.0], 2, 0),
            "fps must be positive",
        ),
        ("fps NaN", go(&v.dims, &v.offsets, &[1.0, f64::NAN], 2, 0), "video 1"),
        (
            "fps inf",
            go(&v.dims, &v.offsets, &[f64::INFINITY, 1.0], 2, 0),
            "video 0",
        ),
        ("no frames", go(&v.dims, &[0, 2, 2], &v.fps, 2, 0), "does not exceed"),
        (
            "offsets not from 0",
            go(&v.dims, &[1, 2, 5], &v.fps, 2, 0),
            "offsets[0] is 1",
        ),
        (
            "a dimension 0",
            go(&[16, 0, 8, 24], &v.offsets, &v.fps, 2, 0),
            "video 0 is 16x0",
        ),
        (
            "frames_len short",
            run(&v.frames, len - 1, &v.dims, &v.offsets, &v.fps, 2, 0, 192),
            "expected",
        ),
        (
            "more frames than a call takes, buffer tiny",
            run(&v.frames[..8], 8, &v.dims[..2], &[0, 1 << 21], &v.fps[..1], 1, 0, 96),
            "limit is",
        ),
        (
            "frames over the pixel limit, buffer tiny",
            run(&v.frames[..8], 8, &[70_000, 70_000], &[0, 1], &v.fps[..1], 1, 0, 96),
            "pixel limit",
        ),
        (
            "out_len wrong",
            run(&v.frames, len, &v.dims, &v.offsets, &v.fps, 2, 0, 96),
            "expected exactly 192",
        ),
    ];
    for (what, (s, m, out), want_msg) in cases {
        assert_eq!(s, TESSL_ERR, "{what}: {m}");
        assert!(m.contains(want_msg), "{what}: {m}");
        assert!(untouched(&out), "{what}: a refused call wrote its output");
    }
    let (s, m, out) = go(&v.dims, &v.offsets, &v.fps, 2, 0);
    assert_eq!(s, TESSL_OK, "the handle is unusable afterwards: {m}");
    assert_eq!(bits(&out), bits(&want));
}

#[test]
fn timestamps_need_the_tokenizer_and_plain_video_does_not() {
    let _g = gpu_lock();
    let side = Sidecars {
        tokenizer: false,
        tokenizer_config: false,
        prompts: None,
        ..Sidecars::full()
    };
    let dir = snapshot("capi_embedgemma2_raw_video_no_tokenizer", &side);
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    let v = videos(&[(2, 16, 16, 1.0)]);
    let (s, m, o) = embed_videos(l.model, &v, 0, 0);
    assert_eq!(s, TESSL_OK, "{m}");
    assert!(!untouched(&o));
    let (s, m, o) = embed_videos(l.model, &v, 1, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("tokenizer.json"), "{m}");
    assert!(untouched(&o));
}

// ---------------------------------------------------------------------------
// Media
// ---------------------------------------------------------------------------

#[test]
fn media_items_through_the_c_entry_are_the_facade_bit_for_bit() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_raw_media", &Sidecars::full());
    let cfg = pre();
    let im = images(&[(64, 56, 3), (13, 77, 4), (40, 40, 1)]);
    let c = clips(&[4000, 16000]);
    let v = videos(&[(4, 24, 32, 2.0)]);
    for operands in [TESSL_OPERANDS_EXACT_F32, TESSL_OPERANDS_BF16] {
        let l = Loaded::new(&dir, Some(operands));
        let mm = l.mm();
        let front = AudioFrontEnd::new(&cfg.audio).unwrap();
        let prepared: Vec<_> = im
            .raws()
            .iter()
            .map(|r| preprocess_image(&cfg.image, r).unwrap())
            .collect();
        let feats: Vec<AudioInput> = c.slices().iter().map(|p| front.extract(p, 16000).unwrap()).collect();
        let video = preprocess_video(&cfg.video, &v.raws(0), v.fps[0]).unwrap();
        let clip_tokens = mm.audio().check_clips(&feats.iter().collect::<Vec<_>>()).unwrap();
        let image_tokens: Vec<usize> = prepared.iter().map(|p| p.soft_tokens).collect();
        let video_tokens = video.frames.len() * video.soft_tokens_per_frame;

        let rep = |id: u32, n: usize| std::iter::repeat_n(id, n);
        let items = vec![
            // An image between text.
            MediaItem {
                ids: [BOS, 5, BOI]
                    .into_iter()
                    .chain(rep(IMAGE, image_tokens[0]))
                    .chain([EOI, 6, 7, EOS])
                    .collect(),
                images: 0..1,
                clips: 0..0,
                videos: 0..0,
            },
            // Two images and a clip in one sequence.
            MediaItem {
                ids: [BOS, BOI]
                    .into_iter()
                    .chain(rep(IMAGE, image_tokens[1] + image_tokens[2]))
                    .chain([EOI, BOA])
                    .chain(rep(AUDIO, clip_tokens[0]))
                    .chain([EOA, EOS])
                    .collect(),
                images: 1..3,
                clips: 0..1,
                videos: 0..0,
            },
            // Text only.
            MediaItem {
                ids: vec![BOS, 8, 9, 10, EOS],
                images: 3..3,
                clips: 1..1,
                videos: 0..0,
            },
            // A clip and a video.
            MediaItem {
                ids: [BOS, BOA]
                    .into_iter()
                    .chain(rep(AUDIO, clip_tokens[1]))
                    .chain([EOA, 11, BOI])
                    .chain(rep(VIDEO, video_tokens))
                    .chain([EOI, EOS])
                    .collect(),
                images: 3..3,
                clips: 1..2,
                videos: 0..1,
            },
        ];
        let inputs: Vec<ImageInput<'_>> = prepared.iter().map(|p| p.input()).collect();
        let frame_inputs: Vec<ImageInput<'_>> = video.frames.iter().map(|f| f.input()).collect();
        let video_slices: Vec<&[ImageInput<'_>]> = vec![&frame_inputs];
        let want_items: Vec<MmItem<'_>> = items
            .iter()
            .map(|it| MmItem::Media {
                ids: &it.ids,
                images: &inputs[it.images.clone()],
                audio: &feats[it.clips.clone()],
                videos: &video_slices[it.videos.clone()],
            })
            .collect();
        for trunc in [0u32, 40] {
            let (s, m, got) = embed_media(l.model, &items, &im, &c, &v, trunc);
            assert_eq!(s, TESSL_OK, "{operands} truncate_dim {trunc}: {m}");
            let want = mm.encode(&want_items, (trunc != 0).then_some(trunc)).unwrap();
            assert_eq!(bits(&got), bits(&want.embeddings), "{operands} truncate_dim {trunc}");
        }
        // A call with no media at all is the id entry's.
        let text_only = [MediaItem {
            ids: vec![BOS, 5, 6, EOS],
            images: 0..0,
            clips: 0..0,
            videos: 0..0,
        }];
        let (s, m, got) = embed_media(l.model, &text_only, &images(&[]), &clips(&[]), &videos(&[]), 0);
        assert_eq!(s, TESSL_OK, "{m}");
        let want = mm.encode(&[MmItem::Text(&[BOS, 5, 6, EOS])], None).unwrap();
        assert_eq!(bits(&got), bits(&want.embeddings));
    }
}

#[test]
fn media_refusals_leave_the_output_alone_and_the_handle_usable() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_raw_media_refusals", &Sidecars::full());
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    let cfg = pre();
    let im = images(&[(64, 56, 3)]);
    let s_img = preprocess_image(&cfg.image, &im.raws()[0]).unwrap().soft_tokens;
    let good_ids: Vec<u32> = [BOS, BOI]
        .into_iter()
        .chain(std::iter::repeat_n(IMAGE, s_img))
        .chain([EOI, EOS])
        .collect();
    let item = |ids: Vec<u32>, images: std::ops::Range<usize>| MediaItem {
        ids,
        images,
        clips: 0..0,
        videos: 0..0,
    };
    let (none_c, none_v) = (clips(&[]), videos(&[]));
    let (s, m, want) = embed_media(l.model, &[item(good_ids.clone(), 0..1)], &im, &none_c, &none_v, 0);
    assert_eq!(s, TESSL_OK, "{m}");

    // The placeholder count is the media's soft-token count, exactly.
    let mut short = good_ids.clone();
    short.remove(3);
    let (s, m, o) = embed_media(l.model, &[item(short, 0..1)], &im, &none_c, &none_v, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("image placeholder ids"), "{m}");
    assert!(untouched(&o));
    // Media given and no placeholder for it.
    let (s, m, o) = embed_media(l.model, &[item(vec![BOS, 5, EOS], 0..1)], &im, &none_c, &none_v, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("image placeholder ids"), "{m}");
    assert!(untouched(&o));

    // Raw calls for the layout checks.
    let ids = good_ids.clone();
    let id_cuts = [0u64, ids.len() as u64];
    let image_cuts = [0u64, 1];
    let zero = [0u64, 0];
    let run = |n_items: u64,
               id_cuts: &[u64],
               image_cuts: &[u64],
               n_images: u64,
               ids_ptr: *const u32,
               out_len: u64,
               trunc: u32| {
        let mut out = vec![SENTINEL; DIM];
        let (s, m) = call(|e, n_e| unsafe {
            tessl_embedgemma2_embed_media(
                l.model,
                ids_ptr,
                ids.len() as u64,
                id_cuts.as_ptr(),
                n_items,
                im.data.as_ptr(),
                im.data.len() as u64,
                im.dims.as_ptr(),
                n_images,
                image_cuts.as_ptr(),
                ptr::null(),
                0,
                zero.as_ptr(),
                0,
                zero.as_ptr(),
                16000,
                ptr::null(),
                0,
                ptr::null(),
                zero.as_ptr(),
                ptr::null(),
                0,
                zero.as_ptr(),
                trunc,
                out.as_mut_ptr(),
                out_len,
                e,
                n_e,
            )
        });
        (s, m, out)
    };
    let ok = run(1, &id_cuts, &image_cuts, 1, ids.as_ptr(), DIM as u64, 0);
    assert_eq!(ok.0, TESSL_OK, "{}", ok.1);
    assert_eq!(bits(&ok.2), bits(&want));
    let cases = [
        (
            "n_items 0",
            run(0, &id_cuts, &image_cuts, 1, ids.as_ptr(), 0, 0),
            "n_items is 0",
        ),
        (
            "n_images past a call",
            run(1, &id_cuts, &image_cuts, 4097, ids.as_ptr(), DIM as u64, 0),
            "n_images is 4097",
        ),
        (
            "image cuts end elsewhere",
            run(1, &id_cuts, &[0, 2], 1, ids.as_ptr(), DIM as u64, 0),
            "image_offsets end at 2 but the call passes 1",
        ),
        (
            "image cuts not from 0",
            run(1, &id_cuts, &[1, 1], 1, ids.as_ptr(), DIM as u64, 0),
            "image_offsets[0] is 1",
        ),
        (
            "id cuts end elsewhere",
            run(1, &[0, 2], &image_cuts, 1, ids.as_ptr(), DIM as u64, 0),
            "id_offsets end at 2",
        ),
        (
            "null ids",
            run(1, &id_cuts, &image_cuts, 1, ptr::null(), DIM as u64, 0),
            "null ids",
        ),
        (
            "out_len wrong",
            run(1, &id_cuts, &image_cuts, 1, ids.as_ptr(), 5, 0),
            "expected exactly 96",
        ),
        (
            "truncate_dim past the width",
            run(1, &id_cuts, &image_cuts, 1, ids.as_ptr(), 97, 97),
            "truncate_dim 97",
        ),
    ];
    for (what, (s, m, out), want_msg) in cases {
        assert_eq!(s, TESSL_ERR, "{what}: {m}");
        assert!(m.contains(want_msg), "{what}: {m}");
        assert!(untouched(&out), "{what}: a refused call wrote its output");
    }
    let again = run(1, &id_cuts, &image_cuts, 1, ids.as_ptr(), DIM as u64, 0);
    assert_eq!(again.0, TESSL_OK, "the handle is unusable afterwards: {}", again.1);
    assert_eq!(bits(&again.2), bits(&want));
}

// ---------------------------------------------------------------------------
// Text-only handles, threads, precision
// ---------------------------------------------------------------------------

#[test]
fn a_text_only_handle_refuses_the_media_entries_naming_the_multimodal_load() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_raw_text_only", &Sidecars::full());
    let l = Loaded::new(&dir, None);
    assert!(unsafe { test_embedgemma2_multimodal(l.model) }.is_none());
    let results = [
        ("images", {
            let (s, m, o) = embed_images(l.model, &images(&[(16, 16, 3)]), 0);
            assert!(untouched(&o));
            (s, m)
        }),
        ("audio", {
            let (s, m, o) = embed_audio(l.model, &clips(&[3200]), 16000, 0);
            assert!(untouched(&o));
            (s, m)
        }),
        ("videos", {
            let (s, m, o) = embed_videos(l.model, &videos(&[(2, 16, 16, 1.0)]), 0, 0);
            assert!(untouched(&o));
            (s, m)
        }),
        ("media", {
            let item = [MediaItem {
                ids: vec![BOS, EOS],
                images: 0..0,
                clips: 0..0,
                videos: 0..0,
            }];
            let (s, m, o) = embed_media(l.model, &item, &images(&[]), &clips(&[]), &videos(&[]), 0);
            assert!(untouched(&o));
            (s, m)
        }),
    ];
    for (name, (s, m)) in results {
        assert_eq!(s, TESSL_ERR, "{name}");
        assert!(m.contains("tessl_embedgemma2_load_multimodal"), "{name}: {m}");
    }
    assert_eq!(embed_texts(l.model, &texts(&["12"]), None, 0).0, TESSL_OK);
}

#[test]
fn a_model_used_from_another_thread_is_refused_by_the_front_end_entries() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_raw_thread", &Sidecars::full());
    let l = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    let model = l.model as usize;
    let results = std::thread::spawn(move || {
        let model = model as *mut TesslEmbedGemma2;
        let mut n = 0u32;
        let mut r = vec![
            ("embed_texts", {
                let (s, m, o) = embed_texts(model, &texts(&["12"]), None, 0);
                assert!(untouched(&o));
                (s, m)
            }),
            (
                "tokenize",
                match tokenize(model, &texts(&["12"]), None) {
                    Ok(_) => (TESSL_OK, String::new()),
                    Err(m) => (TESSL_ERR, m),
                },
            ),
            ("prompt_count", {
                call(|e, n_e| unsafe { tessl_embedgemma2_prompt_count(model, &mut n, e, n_e) })
            }),
        ];
        r.push(("embed_images", {
            let (s, m, o) = embed_images(model, &images(&[(16, 16, 3)]), 0);
            assert!(untouched(&o));
            (s, m)
        }));
        r.push(("embed_audio", {
            let (s, m, o) = embed_audio(model, &clips(&[3200]), 16000, 0);
            assert!(untouched(&o));
            (s, m)
        }));
        r.push(("embed_videos", {
            let (s, m, o) = embed_videos(model, &videos(&[(2, 16, 16, 1.0)]), 0, 0);
            assert!(untouched(&o));
            (s, m)
        }));
        (r, n)
    })
    .join()
    .unwrap();
    for (name, (s, m)) in results.0 {
        assert_eq!(s, TESSL_ERR, "{name}");
        assert!(m.contains("thread other than the one that created it"), "{name}: {m}");
    }
    assert_eq!(results.1, 0, "a refused query wrote its output");
    assert_eq!(embed_texts(l.model, &texts(&["12"]), None, 0).0, TESSL_OK);
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
fn a_bf16_handle_accepts_a_relaxed_runtime_for_every_call_and_an_exact_one_refuses_it() {
    let _g = gpu_lock();
    let dir = snapshot("capi_embedgemma2_raw_relaxed", &Sidecars::full());
    let im = images(&[(64, 56, 3), (13, 77, 4)]);
    let c = clips(&[4000, 16000]);
    let v = videos(&[(4, 24, 32, 2.0)]);
    let t = texts(&["12", "1: 2:3"]);

    // bf16: no GEMM of any model is an f32 one, so the switch changes nothing,
    // for texts, images, clips and videos alike.
    let bf = Loaded::new(&dir, Some(TESSL_OPERANDS_BF16));
    let strict = (
        embed_texts(bf.model, &t, None, 0).2,
        embed_images(bf.model, &im, 0).2,
        embed_audio(bf.model, &c, 16000, 0).2,
        embed_videos(bf.model, &v, 0, 0).2,
        embed_videos(bf.model, &v, 1, 0).2,
    );
    {
        let _relaxed = Relaxed::on(bf.model);
        let relaxed = (
            embed_texts(bf.model, &t, None, 0),
            embed_images(bf.model, &im, 0),
            embed_audio(bf.model, &c, 16000, 0),
            embed_videos(bf.model, &v, 0, 0),
            embed_videos(bf.model, &v, 1, 0),
        );
        for (what, (s, m, _)) in [
            ("texts", &relaxed.0),
            ("images", &relaxed.1),
            ("clips", &relaxed.2),
            ("videos", &relaxed.3),
            ("timestamped videos", &relaxed.4),
        ]
        .map(|(w, r)| (w, r.clone()))
        {
            assert_eq!(s, TESSL_OK, "bf16 {what} under relaxed precision: {m}");
        }
        assert_eq!(bits(&relaxed.0 .2), bits(&strict.0));
        assert_eq!(bits(&relaxed.1 .2), bits(&strict.1));
        assert_eq!(bits(&relaxed.2 .2), bits(&strict.2));
        assert_eq!(bits(&relaxed.3 .2), bits(&strict.3));
        assert_eq!(bits(&relaxed.4 .2), bits(&strict.4));
    }

    // Exact f32: every call that runs a model refuses, by name, and nothing is written.
    let ex = Loaded::new(&dir, Some(TESSL_OPERANDS_EXACT_F32));
    let _relaxed = Relaxed::on(ex.model);
    for (what, (s, m, o)) in [
        ("texts", embed_texts(ex.model, &t, None, 0)),
        ("images", embed_images(ex.model, &im, 0)),
        ("clips", embed_audio(ex.model, &c, 16000, 0)),
        ("videos", embed_videos(ex.model, &v, 0, 0)),
    ] {
        assert_eq!(s, TESSL_ERR, "{what}");
        assert!(m.contains("relaxed precision"), "{what}: {m}");
        assert!(untouched(&o), "{what}: a refused call wrote its output");
    }
}
