//! Video decoding that matches the reference loader byte for byte
//! (`tessl::media::video`, `tessl::media::yuv`), on H.264 and HEVC from
//! AVFoundation (VideoToolbox, so not on `scripts/hosted_tests.txt`: run it with
//! `cargo test --release --test media_video_exact` on a Mac).
//!
//! The reference is torchcodec (`transformers`' `fetch_videos`), not installed
//! here; the oracle is the `ffmpeg` CLI it wraps, with the settings that make it
//! equal to torchcodec (`scripts/gen_media_video_exact_fixtures.py`, docstring).
//! The fixtures and goldens are in `tests/fixtures/media_video/`. The converter on
//! its own, with no codec, is held to the same goldens by `tests/media_yuv_exact.rs`.

mod video_exact_support;

use tessl::embedgemma2_pre::VideoConfig;
use tessl::media::video::{ColorPath, Conversion, Rotation, VideoOptions, VideoSource};
use tessl::media::yuv::{Flavor, Matrix, Range};
use video_exact_support::*;

fn open(name: &str, options: &VideoOptions) -> VideoSource {
    VideoSource::open(&fixture(&format!("{name}.mp4")), options).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Fixtures with a frame-by-frame RGB golden (not the end-to-end file, not the fallback one).
fn exact_fixtures() -> Vec<Fact> {
    facts()
        .into_iter()
        .filter(|f| f.name != "e2e_video" && f.name != "hevc_main10_64x48")
        .collect()
}

fn video_config() -> VideoConfig {
    let p =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embedgemma2/eg2p_processor_config.json");
    tessl::embedgemma2_pre::PreConfig::from_file(&p)
        .expect("processor_config.json")
        .video
}

#[test]
fn the_decoders_planes_are_the_planes_ffmpeg_decodes() {
    for f in exact_fixtures() {
        let src = open(&f.name, &VideoOptions::default());
        let i = src.info();
        assert!(i.eight_bit_420, "{}: expected an 8-bit 4:2:0 stream", f.name);
        let frames = src
            .decode_native_planes(&(0..f.frames).collect::<Vec<_>>())
            .unwrap_or_else(|e| panic!("{}: {e}", f.name));
        let planes = bytes(&format!("{}.nv12", f.name));
        let (y_len, uv_len) = (f.width * f.height, f.width * f.height / 2);
        assert_eq!(planes.len(), (y_len + uv_len) * f.frames, "{}: golden size", f.name);
        for (k, fr) in frames.iter().enumerate() {
            assert_eq!((fr.width, fr.height), (f.width, f.height), "{} frame {k}", f.name);
            assert_eq!(fr.full_range, f.range() == Range::Full, "{} frame {k}: range", f.name);
            let at = k * (y_len + uv_len);
            assert_eq!(
                first_diff(&fr.y, &planes[at..at + y_len]),
                None,
                "{} frame {k}: luma (first differing byte, differing bytes, largest difference)",
                f.name
            );
            assert_eq!(
                first_diff(&fr.uv, &planes[at + y_len..at + y_len + uv_len]),
                None,
                "{} frame {k}: chroma (first differing byte, differing bytes, largest difference)",
                f.name
            );
        }
    }
}

#[test]
fn rgb_frames_are_byte_equal_to_the_oracle() {
    let mut classes = std::collections::BTreeSet::new();
    for f in exact_fixtures() {
        let src = open(&f.name, &VideoOptions::default());
        let all = src
            .decode_frames(&(0..f.frames).collect::<Vec<_>>())
            .unwrap_or_else(|e| panic!("{}: {e}", f.name));
        let (w, h) = f.display_size();
        assert_eq!((all.width, all.height), (w, h), "{}: size after rotation", f.name);
        assert_eq!(
            all.conversion,
            Conversion::Swscale {
                matrix: f.matrix(),
                range: f.range(),
                flavor: Flavor::Aarch64
            },
            "{}",
            f.name
        );
        let want = bytes(&format!("{}.rgb24", f.name));
        let per = w * h * 3;
        assert_eq!(want.len(), per * f.frames, "{}: golden size", f.name);
        for (k, frame) in all.frames.iter().enumerate() {
            assert_eq!(
                first_diff(frame, &want[k * per..(k + 1) * per]),
                None,
                "{} frame {k}: (first differing byte, differing bytes, largest difference)",
                f.name
            );
        }
        classes.insert(format!(
            "{:?}/{:?}/{:?}/{}",
            f.matrix(),
            f.range(),
            if f.name.starts_with("hevc") { "hevc" } else { "h264" },
            if f.width % 16 == 0 { "neon" } else { "table" }
        ));
    }
    // H.264 and HEVC, both ranges, both matrices, both converters, an untagged stream, rotated streams.
    assert!(classes.len() >= 8, "classes covered: {classes:?}");
}

#[test]
fn rotation_ignored_returns_the_frames_as_stored() {
    for name in ["h264_709_tv_64x48_rot90", "h264_601_tv_66x38_rot270"] {
        let f = fact(name);
        let opts = VideoOptions {
            rotation: Rotation::Ignore,
            ..VideoOptions::default()
        };
        let stored = open(name, &opts).decode_frames(&[0, 3]).unwrap();
        assert_eq!((stored.width, stored.height), (f.width, f.height), "{name}");
        let applied = open(name, &VideoOptions::default()).decode_frames(&[0, 3]).unwrap();
        assert_eq!((applied.width, applied.height), (f.height, f.width), "{name}");
        // The same picture, turned: the 90 degree transpose of one is the other.
        let turned = |frame: &[u8], clockwise: bool| -> Vec<u8> {
            let (w, h) = (f.width, f.height);
            let mut out = vec![0u8; frame.len()];
            for y in 0..h {
                for x in 0..w {
                    let (nx, ny) = if clockwise { (h - 1 - y, x) } else { (y, w - 1 - x) };
                    let dst = (ny * h + nx) * 3;
                    out[dst..dst + 3].copy_from_slice(&frame[(y * w + x) * 3..(y * w + x) * 3 + 3]);
                }
            }
            out
        };
        // rot90 is a display rotation of 90 degrees counter-clockwise, rot270 clockwise.
        let want = turned(&stored.frames[0], f.rotation == 270);
        assert_eq!(first_diff(&applied.frames[0], &want), None, "{name}");
    }
}

#[test]
fn frame_count_duration_and_fps_are_torchcodecs_exact_mode() {
    for f in facts() {
        let src = open(&f.name, &VideoOptions::default());
        let i = src.info();
        assert_eq!(i.total_frames, f.frames, "{}", f.name);
        assert_eq!(
            i.duration.to_bits(),
            f.duration.to_bits(),
            "{}: duration {} vs {}",
            f.name,
            i.duration,
            f.duration
        );
        assert_eq!(
            i.fps.to_bits(),
            f.average_fps.to_bits(),
            "{}: fps {} vs {}",
            f.name,
            i.fps,
            f.average_fps
        );
        assert_eq!(i.timestamps.len(), f.pts.len(), "{}", f.name);
        for (k, (a, b)) in i.timestamps.iter().zip(&f.pts).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{} frame {k}: {a} vs ffprobe {b}", f.name);
        }
    }
    // The point of using the exact figures: the header's rate is not the reference's.
    let e2e = open("e2e_video", &VideoOptions::default());
    assert_ne!(e2e.info().fps.to_bits(), e2e.info().nominal_fps.to_bits());
}

#[test]
fn sampled_indices_are_the_transformers_processors() {
    let base = video_config();
    let settings = [
        ("sampled", Some(1.0), 32usize, tessl::embedgemma2_pre::Overflow::Uniform),
        ("sampled_f5m4u", Some(5.0), 4, tessl::embedgemma2_pre::Overflow::Uniform),
        (
            "sampled_f5m4t",
            Some(5.0),
            4,
            tessl::embedgemma2_pre::Overflow::Truncate,
        ),
        ("sampled_all_m3u", None, 3, tessl::embedgemma2_pre::Overflow::Uniform),
    ];
    let mut seen = 0;
    for f in facts() {
        let src = open(&f.name, &VideoOptions::default());
        for (key, fps, max_frames, overflow) in &settings {
            let want = f
                .sampled
                .get(*key)
                .unwrap_or_else(|| panic!("{}: no {key} in facts.txt", f.name));
            let mut cfg = base.clone();
            cfg.fps = *fps;
            cfg.max_frames = *max_frames;
            cfg.overflow = *overflow;
            assert_eq!(&src.sample_indices(&cfg).unwrap(), want, "{} {key}", f.name);
            seen += 1;
        }
    }
    assert!(seen >= 40);
    // The end-to-end file samples three frames of 80, at the reference's step.
    assert_eq!(fact("e2e_video").sampled["sampled"], vec![0, 23, 47]);
}

#[test]
fn an_untagged_stream_is_converted_with_bt601_as_libswscale_does() {
    let src = open("h264_untagged_64x48", &VideoOptions::default());
    assert_eq!(src.info().tags.matrix_code, None);
    let d = src.decode_frames(&[0]).unwrap();
    assert_eq!(
        d.conversion,
        Conversion::Swscale {
            matrix: Matrix::Bt601,
            range: Range::Limited,
            flavor: Flavor::Aarch64
        }
    );
}

#[test]
fn avfoundations_own_conversion_stays_available_and_is_close_but_not_equal() {
    let f = fact("h264_709_tv_64x48");
    let opts = VideoOptions {
        color: ColorPath::AvFoundation,
        ..VideoOptions::default()
    };
    let d = open(&f.name, &opts)
        .decode_frames(&(0..f.frames).collect::<Vec<_>>())
        .unwrap();
    assert_eq!(d.conversion, Conversion::AvFoundation);
    let want = bytes(&format!("{}.rgb24", f.name));
    let got: Vec<u8> = d.frames.concat();
    let (n, max) = first_diff(&got, &want).map_or((0, 0), |(_, n, max)| (n, max));
    let mean = got
        .iter()
        .zip(&want)
        .map(|(a, b)| f64::from(a.abs_diff(*b)))
        .sum::<f64>()
        / want.len() as f64;
    eprintln!(
        "AvFoundation vs the oracle: {n} of {} bytes differ, largest difference {max}, mean {mean:.3}",
        want.len()
    );
    // Measured: 39941 of 55296 bytes, largest 151. The reader's chroma is not the decoder's
    // (see src/media/vt.rs), so this path is not close, which is why it is not the default.
    assert!(
        n > 0,
        "if AVFoundation now matches libswscale, the exact path is redundant"
    );
    assert!(mean < 6.0, "mean abs difference {mean}");
}

#[test]
fn a_stream_that_is_not_8_bit_420_falls_back_to_the_system_conversion() {
    let f = fact("hevc_main10_64x48");
    let src = open(&f.name, &VideoOptions::default());
    assert!(!src.info().eight_bit_420);
    let d = src.decode_frames(&(0..f.frames).collect::<Vec<_>>()).unwrap();
    assert_eq!(d.conversion, Conversion::AvFoundation);
    let want = bytes(&format!("{}.rgb24", f.name));
    let got: Vec<u8> = d.frames.concat();
    let mean = got
        .iter()
        .zip(&want)
        .map(|(a, b)| f64::from(a.abs_diff(*b)))
        .sum::<f64>()
        / want.len() as f64;
    eprintln!("10-bit HEVC through AVFoundation vs the oracle: mean abs difference {mean:.3}");
    // Measured 4.06: 10-bit samples through AVFoundation's converter against swscale's dithered one.
    assert!(mean < 6.0, "mean abs difference {mean}");
}
