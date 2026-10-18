//! Writes the fixtures the Go binding's device tests read (`go/embedgemma2`):
//! the tiny multimodal checkpoint as a snapshot directory, the raw media the
//! tests feed it, and what tessl's C ABI answers for them, called directly from
//! Rust. The Go tests then hold the binding to those answers bit for bit: the
//! only thing between the two is the Go package, gusset, and the umbrella
//! engine, which is what is under test.
//!
//! Everything lands in `target/go_fixtures/` (`snapshot/`, the inputs, the
//! expected rows as raw little-endian `f32` files, and `manifest.json`). It is
//! one test because the Go tests need the whole directory or none of it, and
//! `go/Makefile` runs it before them. It needs a device; the Go tests that read
//! its output fail, not skip, when the directory is missing.

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
    tessl_runtime_free, tessl_runtime_new, TesslRuntime, TESSL_OK, TESSL_OPERANDS_BF16, TESSL_OPERANDS_EXACT_F32,
};
use tessl::capi_embedgemma2::{
    tessl_embedgemma2_audio_soft_token_count, tessl_embedgemma2_decode, tessl_embedgemma2_embed_audio,
    tessl_embedgemma2_embed_audio_sources, tessl_embedgemma2_embed_image_sources, tessl_embedgemma2_embed_images,
    tessl_embedgemma2_embed_segments, tessl_embedgemma2_embed_texts, tessl_embedgemma2_embed_video_sources,
    tessl_embedgemma2_embed_videos, tessl_embedgemma2_free, tessl_embedgemma2_image_soft_token_count,
    tessl_embedgemma2_load_multimodal, tessl_embedgemma2_tokenize, tessl_embedgemma2_video_soft_token_count,
    TesslEmbedGemma2, TESSL_EMBEDGEMMA2_SEGMENT_AUDIO, TESSL_EMBEDGEMMA2_SEGMENT_IMAGE, TESSL_EMBEDGEMMA2_SEGMENT_TEXT,
    TESSL_EMBEDGEMMA2_SEGMENT_VIDEO, TESSL_EMBEDGEMMA2_SEGMENT_VIDEO_TIMESTAMPED, TESSL_EMBEDGEMMA2_SOURCE_BYTES,
    TESSL_EMBEDGEMMA2_SOURCE_PATHS, TESSL_EMBEDGEMMA2_SOURCE_RAW,
};

const ERR_LEN: usize = 512;
const DIM: usize = 96;

fn msg(err: &[c_char]) -> String {
    unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned()
}

fn call(what: &str, f: impl FnOnce(*mut c_char, usize) -> i32) {
    let mut err = [0 as c_char; ERR_LEN];
    let s = f(err.as_mut_ptr(), ERR_LEN);
    assert_eq!(s, TESSL_OK, "{what}: {}", msg(&err));
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

fn opt<T>(v: &[T]) -> *const T {
    if v.is_empty() {
        ptr::null()
    } else {
        v.as_ptr()
    }
}

/// A runtime and a model loaded on it, both freed on drop.
struct Loaded {
    rt: *mut TesslRuntime,
    model: *mut TesslEmbedGemma2,
}

impl Loaded {
    fn new(dir: &Path, operands: u32) -> Self {
        let mut err = [0 as c_char; ERR_LEN];
        let rt = unsafe { tessl_runtime_new(err.as_mut_ptr(), ERR_LEN) };
        assert!(!rt.is_null(), "tessl_runtime_new: {}", msg(&err));
        let bytes = dir.to_str().unwrap().as_bytes();
        let mut model = ptr::null_mut();
        call("load", |e, n| unsafe {
            tessl_embedgemma2_load_multimodal(
                rt,
                bytes.as_ptr().cast(),
                bytes.len() as u64,
                operands,
                &mut model,
                e,
                n,
            )
        });
        Self { rt, model }
    }

    /// Run an embedding entry that writes `rows` rows of the truncated width.
    fn embed(
        &self,
        what: &str,
        rows: usize,
        trunc: u32,
        f: impl FnOnce(*mut f32, u64, u32, *mut c_char, usize) -> i32,
    ) -> Vec<f32> {
        let dim = if trunc == 0 { DIM } else { trunc as usize };
        let mut out = vec![-7.25f32; rows * dim];
        call(what, |e, n| f(out.as_mut_ptr(), out.len() as u64, trunc, e, n));
        out
    }
}

impl Drop for Loaded {
    fn drop(&mut self) {
        assert_eq!(unsafe { tessl_embedgemma2_free(self.model) }, TESSL_OK);
        assert_eq!(unsafe { tessl_runtime_free(self.rt) }, TESSL_OK);
    }
}

/// Media laid end to end and cut: the blob and its `n + 1` offsets.
struct Table {
    blob: Vec<u8>,
    offsets: Vec<u64>,
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

fn paths(names: &[&str]) -> Table {
    table(
        &names
            .iter()
            .map(|n| media_support::fixture(n).to_str().unwrap().as_bytes().to_vec())
            .collect::<Vec<_>>(),
    )
}

fn encoded(names: &[&str]) -> Table {
    table(&names.iter().map(|n| media_support::bytes(n)).collect::<Vec<_>>())
}

impl Table {
    fn n(&self) -> u64 {
        (self.offsets.len() - 1) as u64
    }

    /// The offsets, or null for a modality with no media (the C ABI reads none).
    fn cuts(&self) -> *const u64 {
        if self.n() == 0 {
            ptr::null()
        } else {
            self.offsets.as_ptr()
        }
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The directory the fixtures are written to, and a manifest builder.
struct Out {
    dir: PathBuf,
    expected: Vec<String>,
}

impl Out {
    fn file(&self, name: &str, bytes: &[u8]) {
        std::fs::write(self.dir.join(name), bytes).unwrap();
    }

    fn rows(&mut self, key: &str, rows: usize, data: &[f32]) {
        assert!(data.iter().all(|&v| v != -7.25), "{key}: an output was not written");
        let dim = data.len() / rows;
        self.file(&format!("{key}.f32"), &f32_bytes(data));
        self.expected.push(format!(
            "\"{key}\": {{\"file\": \"{key}.f32\", \"rows\": {rows}, \"dim\": {dim}}}"
        ));
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
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

const IMAGE_SHAPES: [(usize, usize, usize); 6] = [
    (64, 56, 3),
    (40, 40, 1),
    (32, 32, 2),
    (100, 50, 4),
    (13, 77, 3),
    (8, 8, 3),
];
const CLIP_LENS: [usize; 4] = [1600, 4000, 16000, 2500];
const VIDEOS: [(usize, usize, usize, f64); 2] = [(4, 24, 32, 2.0), (8, 40, 16, 4.0)];

/// The processor config of the tiny towers, sampling a video at 5 fps (the
/// fixture video is 1.2 s: six frames, which the config's cap cuts to four).
fn processor_config() -> String {
    processor_config_json(false, 70).replacen("\"fps\": 1,", "\"fps\": 5,", 1)
}

fn write_snapshot(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
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
}

/// Texts laid end to end.
fn text_table(texts: &[&str]) -> (Vec<u8>, Vec<u64>) {
    let mut bytes = Vec::new();
    let mut offsets = vec![0u64];
    for t in texts {
        bytes.extend_from_slice(t.as_bytes());
        offsets.push(bytes.len() as u64);
    }
    (bytes, offsets)
}

fn embed_texts(l: &Loaded, texts: &[&str], prompt: Option<&str>, trunc: u32) -> Vec<f32> {
    let (bytes, offsets) = text_table(texts);
    let (np, nl) = prompt.map_or((ptr::null(), 0), |p| (p.as_ptr().cast::<c_char>(), p.len() as u64));
    l.embed("embed_texts", texts.len(), trunc, |out, out_len, trunc, e, n| unsafe {
        tessl_embedgemma2_embed_texts(
            l.model,
            bytes.as_ptr(),
            bytes.len() as u64,
            offsets.as_ptr(),
            texts.len() as u64,
            np,
            nl,
            trunc,
            out,
            out_len,
            e,
            n,
        )
    })
}

/// One `embed_segments` call over three raw/encoded tables.
struct Segments<'a> {
    texts: &'a [&'a str],
    /// `(kind, ref)` per segment, then the item cuts.
    segments: Vec<(u32, u64)>,
    items: Vec<u64>,
    image: (u32, &'a Table, Vec<u32>),
    audio: (u32, &'a Table, u32),
    video: (u32, &'a Table, Vec<u32>, Vec<f64>),
}

impl Segments<'_> {
    fn run(&self, l: &Loaded, trunc: u32) -> Vec<f32> {
        let (tb, to) = text_table(self.texts);
        let kinds: Vec<u32> = self.segments.iter().map(|s| s.0).collect();
        let refs: Vec<u64> = self.segments.iter().map(|s| s.1).collect();
        let n_items = self.items.len() - 1;
        let (img_src, img, img_dims) = (&self.image.0, self.image.1, &self.image.2);
        let (aud_src, aud, rate) = (&self.audio.0, self.audio.1, self.audio.2);
        let (vid_src, vid, vid_dims, vid_fps) = (&self.video.0, self.video.1, &self.video.2, &self.video.3);
        l.embed("embed_segments", n_items, trunc, |out, out_len, trunc, e, n| unsafe {
            tessl_embedgemma2_embed_segments(
                l.model,
                opt(&tb),
                tb.len() as u64,
                if self.texts.is_empty() {
                    ptr::null()
                } else {
                    to.as_ptr()
                },
                self.texts.len() as u64,
                kinds.as_ptr(),
                refs.as_ptr(),
                kinds.len() as u64,
                self.items.as_ptr(),
                n_items as u64,
                *img_src,
                opt(&img.blob),
                img.blob.len() as u64,
                img.cuts(),
                opt(img_dims),
                img.n(),
                *aud_src,
                opt(&aud.blob),
                aud.blob.len() as u64,
                aud.cuts(),
                rate,
                aud.n(),
                *vid_src,
                opt(&vid.blob),
                vid.blob.len() as u64,
                vid.cuts(),
                opt(vid_dims),
                opt(vid_fps),
                vid.n(),
                ptr::null(),
                0,
                trunc,
                out,
                out_len,
                e,
                n,
            )
        })
    }
}

#[test]
fn write_the_go_binding_fixtures() {
    let _g = gpu_lock();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("go_fixtures");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_snapshot(&dir.join("snapshot"));
    let snapshot = dir.join("snapshot");
    let mut out = Out {
        dir: dir.clone(),
        expected: Vec::new(),
    };

    // Raw inputs, deterministic, written for the Go tests to read back.
    let image_data: Vec<Vec<u8>> = IMAGE_SHAPES
        .iter()
        .enumerate()
        .map(|(k, &(h, w, c))| pixels(h * w * c, 100 + k as u64))
        .collect();
    for (k, d) in image_data.iter().enumerate() {
        out.file(&format!("image{k}.u8"), d);
    }
    let clips: Vec<Vec<f32>> = CLIP_LENS
        .iter()
        .enumerate()
        .map(|(k, &n)| pcm(n, 200 + k as u64))
        .collect();
    for (k, c) in clips.iter().enumerate() {
        out.file(&format!("audio{k}.f32"), &f32_bytes(c));
    }
    let video_data: Vec<Vec<u8>> = VIDEOS
        .iter()
        .enumerate()
        .map(|(k, &(n, h, w, _))| pixels(n * h * w * 3, 300 + k as u64))
        .collect();
    for (k, d) in video_data.iter().enumerate() {
        out.file(&format!("video{k}.u8"), d);
    }

    // The raw tables.
    let img_t = table(&image_data);
    let img_dims: Vec<u32> = IMAGE_SHAPES
        .iter()
        .flat_map(|&(h, w, c)| [h as u32, w as u32, c as u32])
        .collect();
    let clip_bytes: Vec<Vec<u8>> = clips.iter().map(|c| f32_bytes(c)).collect();
    let mut sample_offsets = vec![0u64];
    for c in &clips {
        sample_offsets.push(sample_offsets.last().unwrap() + c.len() as u64);
    }
    let all_samples: Vec<f32> = clips.iter().flatten().copied().collect();
    let all_frames: Vec<u8> = video_data.iter().flatten().copied().collect();
    let vid_dims: Vec<u32> = VIDEOS.iter().flat_map(|&(_, h, w, _)| [h as u32, w as u32]).collect();
    let mut frame_offsets = vec![0u64];
    for &(n, ..) in &VIDEOS {
        frame_offsets.push(frame_offsets.last().unwrap() + n as u64);
    }
    let fps: Vec<f64> = VIDEOS.iter().map(|v| v.3).collect();

    for (label, operands) in [("", TESSL_OPERANDS_EXACT_F32), ("bf16_", TESSL_OPERANDS_BF16)] {
        let l = Loaded::new(&snapshot, operands);
        // Texts: every prompt, full width and a Matryoshka prefix.
        for (pname, prompt) in [("none", None), ("query", Some("query")), ("document", Some("document"))] {
            for trunc in [0u32, 40] {
                let rows = embed_texts(&l, &SAMPLE_TEXTS, prompt, trunc);
                out.rows(&format!("{label}texts_{pname}_{trunc}"), SAMPLE_TEXTS.len(), &rows);
            }
        }
        // Raw images, clips and videos.
        for trunc in [0u32, 40] {
            let rows = l.embed("embed_images", IMAGE_SHAPES.len(), trunc, |o, ol, t, e, n| unsafe {
                tessl_embedgemma2_embed_images(
                    l.model,
                    img_t.blob.as_ptr(),
                    img_t.blob.len() as u64,
                    img_dims.as_ptr(),
                    IMAGE_SHAPES.len() as u64,
                    t,
                    o,
                    ol,
                    e,
                    n,
                )
            });
            out.rows(&format!("{label}images_{trunc}"), IMAGE_SHAPES.len(), &rows);
            let rows = l.embed("embed_audio", CLIP_LENS.len(), trunc, |o, ol, t, e, n| unsafe {
                tessl_embedgemma2_embed_audio(
                    l.model,
                    all_samples.as_ptr(),
                    all_samples.len() as u64,
                    sample_offsets.as_ptr(),
                    CLIP_LENS.len() as u64,
                    16000,
                    t,
                    o,
                    ol,
                    e,
                    n,
                )
            });
            out.rows(&format!("{label}audio_{trunc}"), CLIP_LENS.len(), &rows);
            for (name, stamps) in [("plain", 0u32), ("ts", 1)] {
                let rows = l.embed("embed_videos", VIDEOS.len(), trunc, |o, ol, t, e, n| unsafe {
                    tessl_embedgemma2_embed_videos(
                        l.model,
                        all_frames.as_ptr(),
                        all_frames.len() as u64,
                        vid_dims.as_ptr(),
                        frame_offsets.as_ptr(),
                        fps.as_ptr(),
                        VIDEOS.len() as u64,
                        stamps,
                        t,
                        o,
                        ol,
                        e,
                        n,
                    )
                });
                out.rows(&format!("{label}video_{name}_{trunc}"), VIDEOS.len(), &rows);
            }
        }
        if !label.is_empty() {
            continue;
        }

        // Files and encoded bytes.
        let image_names = ["rgb.png", "gray.png"];
        let clip_names = ["pcm16_16k_mono.wav", "tone_44k_stereo.flac"];
        for (key, source, t) in [
            ("image_files", TESSL_EMBEDGEMMA2_SOURCE_PATHS, paths(&image_names)),
            ("image_bytes", TESSL_EMBEDGEMMA2_SOURCE_BYTES, encoded(&image_names)),
        ] {
            let rows = l.embed("embed_image_sources", 2, 0, |o, ol, tr, e, n| unsafe {
                tessl_embedgemma2_embed_image_sources(
                    l.model,
                    source,
                    t.blob.as_ptr(),
                    t.blob.len() as u64,
                    t.offsets.as_ptr(),
                    t.n(),
                    tr,
                    o,
                    ol,
                    e,
                    n,
                )
            });
            out.rows(key, 2, &rows);
        }
        for (key, source, t) in [
            ("audio_files", TESSL_EMBEDGEMMA2_SOURCE_PATHS, paths(&clip_names)),
            ("audio_bytes", TESSL_EMBEDGEMMA2_SOURCE_BYTES, encoded(&clip_names)),
        ] {
            let rows = l.embed("embed_audio_sources", 2, 0, |o, ol, tr, e, n| unsafe {
                tessl_embedgemma2_embed_audio_sources(
                    l.model,
                    source,
                    t.blob.as_ptr(),
                    t.blob.len() as u64,
                    t.offsets.as_ptr(),
                    t.n(),
                    tr,
                    o,
                    ol,
                    e,
                    n,
                )
            });
            out.rows(key, 2, &rows);
        }
        for (key, source, t, ext) in [
            (
                "video_files",
                TESSL_EMBEDGEMMA2_SOURCE_PATHS,
                paths(&["colors.mp4"]),
                None,
            ),
            (
                "video_bytes",
                TESSL_EMBEDGEMMA2_SOURCE_BYTES,
                encoded(&["colors.mp4"]),
                Some("mp4"),
            ),
        ] {
            for (suffix, stamps) in [("plain", 0u32), ("ts", 1)] {
                let (ep, el) = ext.map_or((ptr::null(), 0), |x: &str| {
                    (x.as_ptr().cast::<c_char>(), x.len() as u64)
                });
                let rows = l.embed("embed_video_sources", 1, 0, |o, ol, tr, e, n| unsafe {
                    tessl_embedgemma2_embed_video_sources(
                        l.model,
                        source,
                        t.blob.as_ptr(),
                        t.blob.len() as u64,
                        t.offsets.as_ptr(),
                        t.n(),
                        ep,
                        el,
                        stamps,
                        tr,
                        o,
                        ol,
                        e,
                        n,
                    )
                });
                out.rows(&format!("{key}_{suffix}"), 1, &rows);
            }
        }

        // Tokenize and decode.
        let (tb, to) = text_table(&SAMPLE_TEXTS);
        let (np, nl) = ("document".as_ptr().cast::<c_char>(), 8u64);
        let mut total = 0u64;
        call("tokenize size", |e, n| unsafe {
            tessl_embedgemma2_tokenize(
                l.model,
                tb.as_ptr(),
                tb.len() as u64,
                to.as_ptr(),
                SAMPLE_TEXTS.len() as u64,
                np,
                nl,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                &mut total,
                e,
                n,
            )
        });
        let mut ids = vec![0u32; total as usize];
        let mut cuts = vec![0u64; SAMPLE_TEXTS.len() + 1];
        let mut again = 0u64;
        call("tokenize", |e, n| unsafe {
            tessl_embedgemma2_tokenize(
                l.model,
                tb.as_ptr(),
                tb.len() as u64,
                to.as_ptr(),
                SAMPLE_TEXTS.len() as u64,
                np,
                nl,
                ids.as_mut_ptr(),
                total,
                cuts.as_mut_ptr(),
                &mut again,
                e,
                n,
            )
        });
        out.file(
            "tokenize_document_ids.u32",
            &ids.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
        );
        out.file(
            "tokenize_document_cuts.u64",
            &cuts.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
        );
        let first = &ids[cuts[0] as usize..cuts[1] as usize];
        for (name, skip) in [("skip", 1u32), ("keep", 0)] {
            let mut needed = 0u64;
            call("decode size", |e, n| unsafe {
                tessl_embedgemma2_decode(
                    l.model,
                    first.as_ptr(),
                    first.len() as u64,
                    skip,
                    ptr::null_mut(),
                    0,
                    &mut needed,
                    e,
                    n,
                )
            });
            let mut buf = vec![0u8; needed as usize];
            let mut got = 0u64;
            call("decode", |e, n| unsafe {
                tessl_embedgemma2_decode(
                    l.model,
                    first.as_ptr(),
                    first.len() as u64,
                    skip,
                    buf.as_mut_ptr(),
                    needed,
                    &mut got,
                    e,
                    n,
                )
            });
            out.file(&format!("decode_{name}.txt"), &buf);
        }

        // Soft-token counts.
        let sizes: [(u32, u32); 3] = [(64, 56), (100, 50), (13, 77)];
        let dims: Vec<u32> = sizes.iter().flat_map(|&(h, w)| [h, w]).collect();
        let mut image_soft = [0u64; 3];
        call("image soft", |e, n| unsafe {
            tessl_embedgemma2_image_soft_token_count(l.model, dims.as_ptr(), 3, image_soft.as_mut_ptr(), 3, e, n)
        });
        let counts: Vec<u64> = CLIP_LENS.iter().map(|&n| n as u64).collect();
        let mut audio_soft = vec![0u64; counts.len()];
        call("audio soft", |e, n| unsafe {
            tessl_embedgemma2_audio_soft_token_count(
                l.model,
                counts.as_ptr(),
                counts.len() as u64,
                16000,
                audio_soft.as_mut_ptr(),
                counts.len() as u64,
                e,
                n,
            )
        });
        let frame_counts: Vec<u64> = VIDEOS.iter().map(|v| v.0 as u64).collect();
        let mut video_soft = vec![0u64; 2 * VIDEOS.len()];
        call("video soft", |e, n| unsafe {
            tessl_embedgemma2_video_soft_token_count(
                l.model,
                vid_dims.as_ptr(),
                frame_counts.as_ptr(),
                fps.as_ptr(),
                VIDEOS.len() as u64,
                video_soft.as_mut_ptr(),
                2 * VIDEOS.len() as u64,
                e,
                n,
            )
        });
        // JSON, not Rust Debug: tuples print as (a, b), which no JSON reader takes.
        let pairs = sizes
            .iter()
            .map(|&(h, w)| format!("[{h}, {w}]"))
            .collect::<Vec<_>>()
            .join(", ");
        let quads = VIDEOS
            .iter()
            .map(|v| format!("[{}, {}, {}, {:?}]", v.1, v.2, v.0, v.3))
            .collect::<Vec<_>>()
            .join(", ");
        out.file(
            "soft_tokens.json",
            format!(
                "{{\"image_sizes\": [{pairs}], \"image\": {image_soft:?}, \"audio_samples\": {CLIP_LENS:?}, \"audio\": {audio_soft:?}, \"videos\": [{quads}], \"video\": {video_soft:?}}}"
            )
            .as_bytes(),
        );

        // Text interleaved with media: all raw, then images by path, clips as bytes, videos raw.
        let texts = ["a 0", "0 b", "1: 2", "00"];
        let img3 = table(&image_data[..3]);
        let img3_dims = img_dims[..9].to_vec();
        let aud2 = table(&clip_bytes[..2]);
        let vid2 = table(&video_data);
        let raw = Segments {
            texts: &texts,
            segments: vec![
                (TESSL_EMBEDGEMMA2_SEGMENT_TEXT, 0),
                (TESSL_EMBEDGEMMA2_SEGMENT_IMAGE, 0),
                (TESSL_EMBEDGEMMA2_SEGMENT_TEXT, 1),
                (TESSL_EMBEDGEMMA2_SEGMENT_TEXT, 2),
                (TESSL_EMBEDGEMMA2_SEGMENT_AUDIO, 1),
                (TESSL_EMBEDGEMMA2_SEGMENT_IMAGE, 1),
                (TESSL_EMBEDGEMMA2_SEGMENT_IMAGE, 2),
                (TESSL_EMBEDGEMMA2_SEGMENT_VIDEO, 0),
                (TESSL_EMBEDGEMMA2_SEGMENT_TEXT, 3),
                (TESSL_EMBEDGEMMA2_SEGMENT_AUDIO, 0),
                (TESSL_EMBEDGEMMA2_SEGMENT_TEXT, 0),
                (TESSL_EMBEDGEMMA2_SEGMENT_VIDEO_TIMESTAMPED, 1),
                (TESSL_EMBEDGEMMA2_SEGMENT_TEXT, 1),
            ],
            items: vec![0, 3, 6, 10, 13],
            image: (TESSL_EMBEDGEMMA2_SOURCE_RAW, &img3, img3_dims),
            audio: (TESSL_EMBEDGEMMA2_SOURCE_RAW, &aud2, 16000),
            video: (TESSL_EMBEDGEMMA2_SOURCE_RAW, &vid2, vid_dims.clone(), fps.clone()),
        };
        for trunc in [0u32, 40] {
            let rows = raw.run(&l, trunc);
            out.rows(&format!("segments_raw_{trunc}"), 4, &rows);
        }
        let files_img = paths(&image_names);
        let bytes_aud = encoded(&clip_names);
        let one_video = table(&video_data[..1]);
        let mixed = Segments {
            texts: &["see", "and hear", "0"],
            segments: vec![
                (TESSL_EMBEDGEMMA2_SEGMENT_TEXT, 0),
                (TESSL_EMBEDGEMMA2_SEGMENT_IMAGE, 0),
                (TESSL_EMBEDGEMMA2_SEGMENT_TEXT, 1),
                (TESSL_EMBEDGEMMA2_SEGMENT_AUDIO, 1),
                (TESSL_EMBEDGEMMA2_SEGMENT_IMAGE, 1),
                (TESSL_EMBEDGEMMA2_SEGMENT_TEXT, 2),
                (TESSL_EMBEDGEMMA2_SEGMENT_VIDEO, 0),
                (TESSL_EMBEDGEMMA2_SEGMENT_AUDIO, 0),
            ],
            items: vec![0, 4, 8],
            image: (TESSL_EMBEDGEMMA2_SOURCE_PATHS, &files_img, Vec::new()),
            audio: (TESSL_EMBEDGEMMA2_SOURCE_BYTES, &bytes_aud, 0),
            video: (
                TESSL_EMBEDGEMMA2_SOURCE_RAW,
                &one_video,
                vid_dims[..2].to_vec(),
                fps[..1].to_vec(),
            ),
        };
        let rows = mixed.run(&l, 0);
        out.rows("segments_mixed_0", 2, &rows);
    }

    let manifest = format!(
        "{{\n  \"dim\": {DIM},\n  \"texts\": [{}],\n  \"images\": [{}],\n  \"audio\": [{}],\n  \"videos\": [{}],\n  \"expected\": {{\n    {}\n  }}\n}}\n",
        SAMPLE_TEXTS.iter().map(|t| json_str(t)).collect::<Vec<_>>().join(", "),
        IMAGE_SHAPES
            .iter()
            .enumerate()
            .map(|(k, &(h, w, c))| format!("{{\"file\": \"image{k}.u8\", \"h\": {h}, \"w\": {w}, \"c\": {c}}}"))
            .collect::<Vec<_>>()
            .join(", "),
        CLIP_LENS
            .iter()
            .enumerate()
            .map(|(k, &n)| format!("{{\"file\": \"audio{k}.f32\", \"samples\": {n}}}"))
            .collect::<Vec<_>>()
            .join(", "),
        VIDEOS
            .iter()
            .enumerate()
            .map(|(k, &(n, h, w, fps))| format!("{{\"file\": \"video{k}.u8\", \"frames\": {n}, \"h\": {h}, \"w\": {w}, \"fps\": {fps:?}}}"))
            .collect::<Vec<_>>()
            .join(", "),
        out.expected.join(",\n    "),
    );
    std::fs::write(dir.join("manifest.json"), manifest).unwrap();
}
