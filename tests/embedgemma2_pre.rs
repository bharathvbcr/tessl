//! Native preprocessing (`tessl::embedgemma2_pre`) against transformers 5.19's
//! `AutoProcessor` for google/embeddinggemma-2, on the CPU, with no device.
//!
//! The oracle is `scripts/gen_embedgemma2_pre_fixtures.py`: its inputs and the
//! processors' outputs are in `tests/fixtures/embedgemma2/eg2p_*`. Two image
//! geometries are checked. `S` (patch 4, pooling 2, 70 soft tokens) is stored
//! whole; `D` (the checkpoint's 16, 3, 280) outputs are 7.7 MB per image, so the
//! index holds their CRC-32 and the test compares that.
//!
//! Exact where transformers' arithmetic is integer or reproducible: the target
//! size, the uint8 resize (ATen's fixed-point bicubic), the rescaled pixels and
//! the positions are compared bit for bit, the video's sampled indices and
//! timestamps exactly. The audio features are float: transformers runs numpy's
//! f32 FFT, tessl an f64 one, so they are held to the bound measured below.
//!
//! With `EMBEDGEMMA2_PRE_BIG_DIR` (the generator's big-case directory) the
//! `#[ignore]`d tests also check the full D arrays of every image, large
//! inputs, a clip past the 30 s limit and a 140-token video.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tessl::embedgemma2_pre::{
    preprocess_audio, preprocess_image, preprocess_video, sample_frame_indices, ImageConfig, Overflow, PreConfig,
    PreparedImage, RawImage, VideoConfig,
};
use tessl::npy::read_npy;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embedgemma2")
}

fn config() -> PreConfig {
    PreConfig::from_file(&fixtures().join("eg2p_processor_config.json")).expect("the snapshot's processor_config.json")
}

/// The generator's small geometry.
fn small(image: &ImageConfig) -> ImageConfig {
    ImageConfig {
        patch_size: 4,
        pooling_kernel: 2,
        max_soft_tokens: 70,
        ..image.clone()
    }
}

/// One index line: its kind, name (or key) and `key=value` fields.
struct Line {
    kind: String,
    name: String,
    geom: String,
    fields: HashMap<String, String>,
}

impl Line {
    fn get(&self, k: &str) -> &str {
        self.fields
            .get(k)
            .unwrap_or_else(|| panic!("{} {}: no {k}", self.kind, self.name))
    }

    fn num<T: std::str::FromStr>(&self, k: &str) -> T
    where
        T::Err: std::fmt::Debug,
    {
        self.get(k)
            .parse()
            .unwrap_or_else(|e| panic!("{} {} {k}: {e:?}", self.kind, self.name))
    }
}

fn index(path: &Path) -> Vec<Line> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .map(|l| {
            let mut words = l.split(' ');
            let kind = words.next().unwrap().to_string();
            let mut name = String::new();
            let mut geom = String::new();
            let mut fields = HashMap::new();
            for w in words {
                match w.split_once('=') {
                    Some((k, v)) => {
                        fields.insert(k.to_string(), v.to_string());
                    }
                    None if name.is_empty() => name = w.to_string(),
                    None => geom = w.to_string(),
                }
            }
            Line {
                kind,
                name,
                geom,
                fields,
            }
        })
        .collect()
}

fn of_kind<'a>(lines: &'a [Line], kind: &str) -> Vec<&'a Line> {
    lines.iter().filter(|l| l.kind == kind).collect()
}

/// CRC-32 (IEEE, as `zlib.crc32`).
fn crc32(bytes: &[u8]) -> u32 {
    let table: Vec<u32> = (0..256u32)
        .map(|n| (0..8).fold(n, |c, _| if c & 1 == 1 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 }))
        .collect();
    !bytes
        .iter()
        .fold(!0u32, |c, &b| table[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8))
}

fn crc_f32(v: &[f32]) -> String {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    format!("{:08x}", crc32(&bytes))
}

/// The CRC of positions as the generator hashes them: int64 little-endian.
fn crc_pos(v: &[i32]) -> String {
    let bytes: Vec<u8> = v.iter().flat_map(|&x| i64::from(x).to_le_bytes()).collect();
    format!("{:08x}", crc32(&bytes))
}

fn npy(dir: &Path, name: &str) -> tessl::npy::NpyArray {
    let p = dir.join(name);
    read_npy(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// Bytes stored as int64.
fn bytes_of(a: &tessl::npy::NpyArray) -> Vec<u8> {
    a.i64_slice()
        .expect("int64 bytes")
        .iter()
        .map(|&v| u8::try_from(v).expect("a byte"))
        .collect()
}

fn to_i32(v: &[i64]) -> Vec<i32> {
    v.iter().map(|&x| i32::try_from(x).unwrap()).collect()
}

/// An image fixture's raw input in `dir`.
fn load_image(dir: &Path, name: &str, channels: usize, h: usize, w: usize) -> Vec<u8> {
    let a = npy(dir, &format!("eg2p_img_{name}_in.npy"));
    let want: Vec<usize> = if channels == 1 {
        vec![h, w]
    } else {
        vec![h, w, channels]
    };
    assert_eq!(a.shape, want, "{name} shape");
    bytes_of(&a)
}

fn assert_bits_eq(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a.to_bits() != b.to_bits()) {
        panic!("{what}: first difference at {i}: {} vs {}", got[i], want[i]);
    }
}

/// An image compared with its index line: the metadata and the CRCs, and for
/// the stored small geometry the arrays bit for bit.
fn check_image(line: &Line, got: &PreparedImage, stored: Option<(&[f32], &[i32])>) {
    let tag = format!("{} {}", line.name, line.geom);
    let resized = format!("{}x{}", got.resized.0, got.resized.1);
    assert_eq!(resized, line.get("resized"), "{tag}: resized size");
    assert_eq!(got.patches, line.num::<usize>("patches"), "{tag}: patches");
    assert_eq!(got.soft_tokens, line.num::<usize>("soft"), "{tag}: soft tokens");
    assert_eq!(crc_pos(&got.position_ids), line.get("crc_pos"), "{tag}: positions");
    if let Some((px, pos)) = stored {
        assert_eq!(got.position_ids, pos, "{tag}: positions");
        assert_bits_eq(&got.pixel_values, px, &format!("{tag}: pixel_values"));
    }
    assert_eq!(
        crc_f32(&got.pixel_values),
        line.get("crc_px"),
        "{tag}: pixel_values CRC"
    );
}

fn run_image_lines(lines: &[Line], dir: &Path, geom: &str, stored: bool) -> usize {
    let cfg = config();
    let image = if geom == "S" {
        small(&cfg.image)
    } else {
        cfg.image.clone()
    };
    let mut n = 0;
    for line in of_kind(lines, "image").into_iter().filter(|l| l.geom == geom) {
        let (c, h, w) = (line.num("channels"), line.num("h"), line.num("w"));
        let data = load_image(dir, &line.name, c, h, w);
        let got = preprocess_image(
            &image,
            &RawImage {
                width: w,
                height: h,
                channels: c,
                data: &data,
            },
        )
        .unwrap_or_else(|e| panic!("{}: {e}", line.name));
        let arrays = stored.then(|| {
            let px = npy(&fixtures(), &format!("eg2p_img_{}_S_px.npy", line.name));
            let pos = npy(&fixtures(), &format!("eg2p_img_{}_S_pos.npy", line.name));
            (px.f32_slice().unwrap().to_vec(), to_i32(pos.i64_slice().unwrap()))
        });
        check_image(line, &got, arrays.as_ref().map(|(p, q)| (p.as_slice(), q.as_slice())));
        n += 1;
    }
    n
}

/// Every image of the small geometry equals the processor's, bit for bit.
/// This is the resize parity: a different filter, precision or pass order moves
/// pixels by whole grey levels, so one wrong tap fails it.
#[test]
fn small_geometry_images_match_the_processor_bit_for_bit() {
    let lines = index(&fixtures().join("eg2p_index.txt"));
    let n = run_image_lines(&lines, &fixtures(), "S", true);
    assert_eq!(n, 13, "every committed image has a small-geometry line");
}

/// The checkpoint's own geometry (patch 16, pooling 3, 280 soft tokens): the
/// CRC-32 of the whole pixel array and the positions, which are equal only if
/// every one of the 1.9 million floats is.
#[test]
fn checkpoint_geometry_images_match_the_processor_crc() {
    let lines = index(&fixtures().join("eg2p_index.txt"));
    let n = run_image_lines(&lines, &fixtures(), "D", false);
    assert_eq!(n, 13);
}

/// Images in the big directory: full arrays, bit for bit, for the committed
/// inputs at D geometry and for large inputs (a 2000x1500 photo, the exact
/// 768x768 target, wide and noisy images, RGBA).
#[test]
#[ignore = "needs EMBEDGEMMA2_PRE_BIG_DIR from the generator"]
fn big_images_match_the_processor_bit_for_bit() {
    let big = PathBuf::from(std::env::var("EMBEDGEMMA2_PRE_BIG_DIR").expect("EMBEDGEMMA2_PRE_BIG_DIR"));
    let cfg = config();
    let committed = index(&fixtures().join("eg2p_index.txt"));
    let mut checked = 0;
    let mut cases: Vec<(String, usize, usize, usize, PathBuf)> = of_kind(&committed, "image")
        .into_iter()
        .filter(|l| l.geom == "D")
        .map(|l| (l.name.clone(), l.num("channels"), l.num("h"), l.num("w"), fixtures()))
        .collect();
    let big_lines = index(&big.join("index_big.txt"));
    cases.extend(
        of_kind(&big_lines, "image")
            .into_iter()
            .map(|l| (l.name.clone(), l.num("channels"), l.num("h"), l.num("w"), big.clone())),
    );
    for (name, c, h, w, dir) in cases {
        let data = load_image(&dir, &name, c, h, w);
        let got = preprocess_image(
            &cfg.image,
            &RawImage {
                width: w,
                height: h,
                channels: c,
                data: &data,
            },
        )
        .unwrap();
        let px = npy(&big, &format!("eg2p_img_{name}_D_px.npy"));
        let pos = npy(&big, &format!("eg2p_img_{name}_D_pos.npy"));
        assert_bits_eq(
            &got.pixel_values,
            px.f32_slice().unwrap(),
            &format!("{name} pixel_values"),
        );
        assert_eq!(got.position_ids, to_i32(pos.i64_slice().unwrap()), "{name} positions");
        checked += 1;
    }
    assert!(checked >= 18, "{checked} big images checked");
}

/// Largest absolute and largest relative (to the larger magnitude, floored at 1)
/// difference between two feature arrays.
fn diffs(got: &[f32], want: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len());
    let mut abs = 0.0f64;
    let mut rel = 0.0f64;
    for (&g, &w) in got.iter().zip(want) {
        let d = (f64::from(g) - f64::from(w)).abs();
        abs = abs.max(d);
        rel = rel.max(d / f64::from(g).abs().max(f64::from(w).abs()).max(1.0));
    }
    (abs, rel)
}

/// Log-mel bound. Measured over the 15 committed clips and the two big ones (a
/// 30 s clip and a 35 s clip cut to 30 s): the largest absolute difference was
/// 4.768e-7, which is 2^-21, one unit in the last place of an f32 of magnitude
/// 4 to 8, and the largest relative difference (to the larger magnitude, at
/// least 1) was 2.35e-7, about two ulps. The bound is four times the largest
/// absolute difference seen. The difference is numpy's float32 FFT against
/// tessl's float64 one (transformers' own rounding), on log-mel values of
/// magnitude up to about 20.
const AUDIO_ABS_BOUND: f64 = 2e-6;

fn check_audio(line: &Line, dir: &Path, cfg: &PreConfig) -> (f64, f64) {
    let clip_arr = npy(dir, &format!("eg2p_aud_{}_in.npy", line.name));
    let clip = clip_arr.f32_slice().unwrap();
    assert_eq!(clip.len(), line.num::<usize>("samples"), "{} samples", line.name);
    let mut audio = cfg.audio.clone();
    let max_length: usize = line.num("max_length");
    if max_length != 0 {
        audio.max_samples = max_length;
    }
    let got = preprocess_audio(&audio, clip, 16000).unwrap();
    let frames: usize = line.num("frames");
    assert_eq!(got.mask.len(), frames, "{} frames", line.name);
    assert_eq!(
        got.mask.iter().filter(|&&m| m).count(),
        line.num::<usize>("valid"),
        "{} valid",
        line.name
    );
    let mask = npy(dir, &format!("eg2p_aud_{}_mask.npy", line.name));
    let want_mask: Vec<bool> = mask.i64_slice().unwrap().iter().map(|&m| m != 0).collect();
    assert_eq!(got.mask, want_mask, "{} mask", line.name);
    assert_eq!(got.input_features.len(), frames * 128);
    if frames == 0 {
        return (0.0, 0.0);
    }
    let feat = npy(dir, &format!("eg2p_aud_{}_feat.npy", line.name));
    assert_eq!(feat.shape, [frames, 128]);
    let (abs, rel) = diffs(&got.input_features, feat.f32_slice().unwrap());
    // Masked frames are exact zeros, not just close.
    for (i, row) in got.input_features.chunks_exact(128).enumerate() {
        if !got.mask[i] {
            assert!(row.iter().all(|&v| v == 0.0), "{} masked frame {i}", line.name);
        }
    }
    (abs, rel)
}

/// Frames, masks and log-mel features against the feature extractor: silence,
/// a sweep, noise, clips shorter than a frame, exactly one frame, odd lengths,
/// DC, Nyquist, loud and near-silent input, and a clip cut at `max_length`.
#[test]
fn audio_features_match_the_processor() {
    let cfg = config();
    let lines = index(&fixtures().join("eg2p_index.txt"));
    let mut worst = (0.0f64, 0.0f64);
    let mut n = 0;
    for line in of_kind(&lines, "audio") {
        let (abs, rel) = check_audio(line, &fixtures(), &cfg);
        eprintln!(
            "{:<20} frames {:>4}  max abs {abs:.3e}  max rel {rel:.3e}",
            line.name,
            line.num::<usize>("frames")
        );
        assert!(
            abs <= AUDIO_ABS_BOUND,
            "{}: max abs {abs:.3e} exceeds {AUDIO_ABS_BOUND:e}",
            line.name
        );
        worst = (worst.0.max(abs), worst.1.max(rel));
        n += 1;
    }
    assert_eq!(n, 15);
    eprintln!("worst over {n} clips: abs {:.3e}, rel {:.3e}", worst.0, worst.1);
}

/// The 35 s clip (cut to 30 s) and a clip of exactly 30 s, at the default limit.
#[test]
#[ignore = "needs EMBEDGEMMA2_PRE_BIG_DIR from the generator"]
fn big_audio_matches_the_processor() {
    let big = PathBuf::from(std::env::var("EMBEDGEMMA2_PRE_BIG_DIR").expect("EMBEDGEMMA2_PRE_BIG_DIR"));
    let cfg = config();
    let lines = index(&big.join("index_big.txt"));
    let clips = of_kind(&lines, "audio");
    assert_eq!(clips.len(), 2);
    for line in clips {
        let (abs, rel) = check_audio(line, &big, &cfg);
        eprintln!(
            "{}: frames {}, max abs {abs:.3e}, max rel {rel:.3e}",
            line.name,
            line.get("frames")
        );
        assert!(abs <= AUDIO_ABS_BOUND, "{}: {abs:.3e}", line.name);
        assert_eq!(line.get("frames"), "2999", "30 s is 2999 frames");
    }
}

fn parse_fps(s: &str) -> f64 {
    let (n, d) = s.split_once('/').unwrap();
    n.parse::<f64>().unwrap() / d.parse::<f64>().unwrap()
}

fn video_config(base: &VideoConfig, mode: &str) -> VideoConfig {
    match mode {
        "uniform" => base.clone(),
        "truncate" => VideoConfig {
            overflow: Overflow::Truncate,
            ..base.clone()
        },
        "nofps" => VideoConfig {
            fps: None,
            ..base.clone()
        },
        "uniform1" => VideoConfig {
            max_frames: 1,
            ..base.clone()
        },
        other => panic!("mode {other}"),
    }
}

/// `sample_frames` over 36 frame counts and 21 source rates (NTSC and other
/// fractional ones among them), under the uniform and truncating caps, with fps
/// sampling off, and with a one-frame cap: every index list equal.
#[test]
fn sampled_frame_indices_equal_the_processors() {
    let base = config().video;
    let lines = index(&fixtures().join("eg2p_index.txt"));
    let mut n = 0;
    let (mut below, mut at, mut above) = (0, 0, 0);
    for line in of_kind(&lines, "video_indices") {
        let total: usize = line.num("n");
        let fps = parse_fps(line.get("fps"));
        let want: Vec<usize> = line.get("idx").split(',').map(|v| v.parse().unwrap()).collect();
        let got = sample_frame_indices(&video_config(&base, line.get("mode")), total, fps).unwrap();
        assert_eq!(got, want, "n={total} fps={} mode={}", line.get("fps"), line.get("mode"));
        if line.get("mode") == "uniform" {
            match want.len() {
                0..=31 => below += 1,
                32 => at += 1,
                _ => unreachable!(),
            }
            above += usize::from(total as f64 / fps > 32.0);
        }
        n += 1;
    }
    assert_eq!(n, 36 * 21 + 15 + 5 + 2);
    // The grid reaches under, exactly at and over the cap (a video of more than 32 sampled
    // frames is capped to exactly 32 and counted under `at`).
    assert!(
        below > 100 && at > 100 && above > 100,
        "grid coverage: {below} below, {at} at, {above} above"
    );
}

/// Whole videos: indices, timestamps (exactly), pixels and positions.
#[test]
fn videos_match_the_processor() {
    let cfg = config();
    let lines = index(&fixtures().join("eg2p_index.txt"));
    let mut n = 0;
    for line in of_kind(&lines, "video") {
        check_video(line, &fixtures(), &cfg, true);
        n += 1;
    }
    assert_eq!(n, 7);
}

fn check_video(line: &Line, dir: &Path, cfg: &PreConfig, stored_ok: bool) {
    let tag = format!("{} {}", line.name, line.geom);
    let (t, h, w): (usize, usize, usize) = (line.num("frames"), line.num("h"), line.num("w"));
    let a = npy(dir, &format!("eg2p_vid_{}_in.npy", line.name));
    assert_eq!(a.shape, [t, h, w, 3], "{tag}");
    let bytes = bytes_of(&a);
    let frames: Vec<RawImage<'_>> = bytes.chunks_exact(h * w * 3).map(|f| RawImage::rgb(w, h, f)).collect();
    let video_cfg = VideoConfig {
        image: if line.geom == "S" {
            small(&cfg.video.image)
        } else {
            cfg.video.image.clone()
        },
        ..cfg.video.clone()
    };
    let got = preprocess_video(&video_cfg, &frames, parse_fps(line.get("fps"))).unwrap();
    let want_idx: Vec<usize> = line.get("indices").split(',').map(|v| v.parse().unwrap()).collect();
    assert_eq!(got.indices, want_idx, "{tag}: sampled indices");
    assert_eq!(got.frames.len(), line.num::<usize>("sampled"), "{tag}: frames");
    let want_stamps: Vec<f64> = line.get("stamps").split(',').map(|v| v.parse().unwrap()).collect();
    assert_eq!(
        got.timestamps.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        want_stamps.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "{tag}: timestamps"
    );
    assert_eq!(
        got.soft_tokens_per_frame,
        line.num::<usize>("soft"),
        "{tag}: soft tokens"
    );
    assert!(
        got.frames.iter().all(|f| f.patches == line.num::<usize>("patches")),
        "{tag}: patches"
    );
    let px: Vec<f32> = got.frames.iter().flat_map(|f| f.pixel_values.iter().copied()).collect();
    let pos: Vec<i32> = got.frames.iter().flat_map(|f| f.position_ids.iter().copied()).collect();
    assert_eq!(crc_pos(&pos), line.get("crc_pos"), "{tag}: positions CRC");
    assert_eq!(crc_f32(&px), line.get("crc_px"), "{tag}: pixel CRC");
    let stored = fixtures_have(dir, &format!("eg2p_vid_{}_px.npy", line.name));
    if stored && stored_ok {
        let want = npy(dir, &format!("eg2p_vid_{}_px.npy", line.name));
        assert_bits_eq(&px, want.f32_slice().unwrap(), &format!("{tag}: pixel_values_videos"));
        let wpos = npy(dir, &format!("eg2p_vid_{}_pos.npy", line.name));
        assert_eq!(pos, to_i32(wpos.i64_slice().unwrap()), "{tag}: video_position_ids");
    }
}

fn fixtures_have(dir: &Path, name: &str) -> bool {
    dir.join(name).is_file()
}

/// A 140-token video of 100 frames and a 600-frame one, full arrays.
#[test]
#[ignore = "needs EMBEDGEMMA2_PRE_BIG_DIR from the generator"]
fn big_videos_match_the_processor() {
    let big = PathBuf::from(std::env::var("EMBEDGEMMA2_PRE_BIG_DIR").expect("EMBEDGEMMA2_PRE_BIG_DIR"));
    let cfg = config();
    let lines = index(&big.join("index_big.txt"));
    let videos = of_kind(&lines, "video");
    assert_eq!(videos.len(), 2);
    for line in videos {
        check_video(line, &big, &cfg, true);
    }
}
