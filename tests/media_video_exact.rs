//! Video decoding that matches the reference loader byte for byte
//! (`tessl::media::video`, `tessl::media::yuv`), on H.264 and HEVC from
//! AVFoundation (VideoToolbox, so not on `scripts/hosted_tests.txt`: run it with
//! `cargo test --release --test media_video_exact` on a Mac).
//!
//! The reference is torchcodec (`transformers`' `fetch_videos`), not installed
//! here; the oracle is the `ffmpeg` CLI it wraps, with the settings that make it
//! equal to torchcodec (`scripts/gen_media_video_exact_fixtures.py`, docstring).
//! The fixtures and goldens are in `tests/fixtures/media_video/`. The converter on
//! its own, with no codec, is held to the same goldens by `tests/media_yuv_exact.rs`
//! and `tests/media_yuv_formats.rs`.
//!
//! The classes: H.264 and HEVC at 8 bits in 4:2:0; HEVC at 8 and 10 bits in 4:2:0,
//! 4:2:2 and 4:4:4 (VideoToolbox's H.264 decoder takes 8-bit 4:2:0 only); every
//! matrix the converter holds (BT.601, BT.709, BT.2020, FCC, SMPTE 240M) in both
//! ranges; the shapes H.264 and HEVC can code with an odd side (4:4:4 any, 4:2:2 an
//! odd height); rotation and flips of each layout.

mod video_exact_support;

use std::collections::BTreeSet;

use tessl::embedgemma2_pre::VideoConfig;
use tessl::media::video::{ColorPath, Conversion, Rotation, Samples, VideoOptions, VideoSource};
use tessl::media::yuv::{Chroma, ChromaFilter, Flavor, Layout, Matrix, Range, Threads};
use tessl::media::MediaError;
use video_exact_support::*;

fn open(name: &str, options: &VideoOptions) -> VideoSource {
    VideoSource::open(&fixture(&format!("{name}.mp4")), options).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Fixtures with frame-by-frame goldens (not the end-to-end file, not the refused streams).
fn exact_fixtures() -> Vec<Fact> {
    facts().into_iter().filter(|f| f.golden).collect()
}

/// The MPEG-4 Part 2 stream AVFoundation cannot decode at all: not even its first frame, so
/// a source cannot be opened on it, under either colour path.
const UNDECODABLE: &str = "refuse_mpeg4_709_tv_64x48";

/// Every fixture a source can be opened on.
fn openable_facts() -> Vec<Fact> {
    facts().into_iter().filter(|f| f.name != UNDECODABLE).collect()
}

fn video_config() -> VideoConfig {
    let p =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embedgemma2/eg2p_processor_config.json");
    tessl::embedgemma2_pre::PreConfig::from_file(&p)
        .expect("processor_config.json")
        .video
}

/// The golden of a fixture's frames under `options`: the 10-bit 4:2:0 ones depend on the
/// chroma filter (`.bilinear.rgb24` is bilinear, `.rgb24` is the CLI's own default, bicubic).
fn rgb_golden(f: &Fact, filter: ChromaFilter) -> Vec<u8> {
    let ten_420 = f.layout()
        == Layout {
            chroma: Chroma::C420,
            depth: 10,
        };
    if ten_420 && filter == ChromaFilter::Bilinear {
        bytes(&format!("{}.bilinear.rgb24", f.name))
    } else {
        bytes(&format!("{}.rgb24", f.name))
    }
}

fn expected_conversion(f: &Fact, filter: ChromaFilter) -> Conversion {
    Conversion::Swscale {
        layout: f.layout(),
        matrix: f.matrix(),
        range: f.range(),
        flavor: Flavor::Aarch64,
        filter,
    }
}

#[test]
fn the_decoders_planes_are_the_planes_ffmpeg_decodes() {
    let mut layouts = BTreeSet::new();
    for f in exact_fixtures() {
        let src = open(&f.name, &VideoOptions::default());
        let i = src.info();
        assert_eq!(
            i.layout,
            Ok(f.layout()),
            "{}: layout from the codec configuration",
            f.name
        );
        let frames = src
            .decode_native_planes(&(0..f.frames).collect::<Vec<_>>())
            .unwrap_or_else(|e| panic!("{}: {e}", f.name));
        let planes = bytes(&format!("{}.{}", f.name, f.planes_ext()));
        let layout = f.layout();
        let (cw, ch) = match layout.chroma {
            Chroma::C420 => (f.width.div_ceil(2), f.height.div_ceil(2)),
            Chroma::C422 => (f.width.div_ceil(2), f.height),
            Chroma::C444 => (f.width, f.height),
        };
        let (y_len, uv_len) = (f.width * f.height, 2 * cw * ch);
        let bps = if layout.depth == 8 { 1 } else { 2 };
        assert_eq!(
            planes.len(),
            (y_len + uv_len) * bps * f.frames,
            "{}: golden size",
            f.name
        );
        for (k, fr) in frames.iter().enumerate() {
            assert_eq!((fr.width, fr.height), (f.width, f.height), "{} frame {k}", f.name);
            assert_eq!(fr.layout, layout, "{} frame {k}", f.name);
            assert_eq!(fr.full_range, f.range() == Range::Full, "{} frame {k}: range", f.name);
            let at = k * (y_len + uv_len) * bps;
            // A golden's samples as values: bytes at 8 bits, 16-bit words holding the sample in
            // their high ten bits (p010le and kin) at 10.
            let golden = |from: usize, n: usize| -> Vec<u32> {
                let raw = &planes[at + from * bps..at + (from + n) * bps];
                if bps == 1 {
                    raw.iter().map(|&b| u32::from(b)).collect()
                } else {
                    raw.chunks_exact(2)
                        .map(|c| u32::from(u16::from_le_bytes([c[0], c[1]]) >> 6))
                        .collect()
                }
            };
            let values = |s: &Samples| -> Vec<u32> {
                match s {
                    Samples::U8(v) => v.iter().map(|&b| u32::from(b)).collect(),
                    Samples::U16(v) => v.iter().map(|&b| u32::from(b)).collect(),
                }
            };
            let first_difference = |got: Vec<u32>, want: Vec<u32>| -> Option<(usize, usize)> {
                assert_eq!(got.len(), want.len());
                let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
                bad.first().map(|&i| (i, bad.len()))
            };
            assert_eq!(
                first_difference(values(&fr.y), golden(0, y_len)),
                None,
                "{} frame {k}: luma (first differing sample, differing samples)",
                f.name
            );
            assert_eq!(
                first_difference(values(&fr.uv), golden(y_len, uv_len)),
                None,
                "{} frame {k}: chroma (first differing sample, differing samples)",
                f.name
            );
        }
        layouts.insert(format!("{:?}/{}", layout.chroma, layout.depth));
    }
    // 4:2:0, 4:2:2 and 4:4:4 at 8 and 10 bits.
    assert_eq!(layouts.len(), 6, "layouts covered: {layouts:?}");
}

#[test]
fn rgb_frames_are_byte_equal_to_the_oracle() {
    let mut classes = BTreeSet::new();
    let mut odd_sides = 0;
    for f in exact_fixtures() {
        let src = open(&f.name, &VideoOptions::default());
        let all = src
            .decode_frames(&(0..f.frames).collect::<Vec<_>>())
            .unwrap_or_else(|e| panic!("{}: {e}", f.name));
        let (w, h) = f.display_size();
        assert_eq!((all.width, all.height), (w, h), "{}: size after rotation", f.name);
        assert_eq!(
            all.conversion,
            expected_conversion(&f, ChromaFilter::Bilinear),
            "{}",
            f.name
        );
        let want = rgb_golden(&f, ChromaFilter::Bilinear);
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
        odd_sides += usize::from(w % 2 == 1 || h % 2 == 1);
        classes.insert(format!(
            "{:?}/{}/{:?}/{:?}/{}/{}",
            f.layout().chroma,
            f.layout().depth,
            f.matrix(),
            f.range(),
            if f.name.starts_with("hevc") { "hevc" } else { "h264" },
            if f.width % 16 == 0 { "neon" } else { "table" }
        ));
    }
    // Every layout and depth, all five matrices, both ranges, H.264 and HEVC, both converter
    // widths, rotated and flipped streams, odd sides.
    assert!(classes.len() >= 28, "classes covered: {} {classes:?}", classes.len());
    assert!(odd_sides >= 5, "{odd_sides} fixtures with an odd side");
}

#[test]
fn ten_bit_420_under_the_clis_default_bicubic_filter_is_exact_too() {
    let mut checked = 0;
    for f in exact_fixtures() {
        if f.layout()
            != (Layout {
                chroma: Chroma::C420,
                depth: 10,
            })
        {
            continue;
        }
        let options = VideoOptions {
            chroma_filter: ChromaFilter::Bicubic,
            ..VideoOptions::default()
        };
        let all = open(&f.name, &options)
            .decode_frames(&(0..f.frames).collect::<Vec<_>>())
            .unwrap_or_else(|e| panic!("{}: {e}", f.name));
        assert_eq!(
            all.conversion,
            expected_conversion(&f, ChromaFilter::Bicubic),
            "{}",
            f.name
        );
        let want = rgb_golden(&f, ChromaFilter::Bicubic);
        assert_eq!(first_diff(&all.frames.concat(), &want), None, "{}: bicubic", f.name);
        // And the two filters really are different pictures, so the test could fail.
        assert_ne!(rgb_golden(&f, ChromaFilter::Bilinear), want, "{}", f.name);
        checked += 1;
    }
    assert!(checked >= 8, "{checked} 10-bit 4:2:0 fixtures");
}

#[test]
fn the_thread_count_never_changes_the_frames() {
    // The byte-identical guarantee of the threaded converter, through the whole decode.
    for name in [
        "hevc_main10_64x48",
        "hevc_444_709_tv_64x48",
        "hevc_422_601_pc_66x38",
        "h264_709_tv_64x48",
    ] {
        let f = fact(name);
        let all: Vec<usize> = (0..f.frames).collect();
        let single = open(
            name,
            &VideoOptions {
                threads: Threads::Single,
                ..VideoOptions::default()
            },
        )
        .decode_frames(&all)
        .unwrap();
        for threads in [
            Threads::Auto,
            Threads::Count(2),
            Threads::Count(5),
            Threads::Count(1000),
        ] {
            let got = open(
                name,
                &VideoOptions {
                    threads,
                    ..VideoOptions::default()
                },
            )
            .decode_frames(&all)
            .unwrap();
            assert_eq!(got.frames, single.frames, "{name} with {threads:?}");
        }
    }
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
fn rotation_follows_the_clis_autorotate_where_the_order_of_turning_and_converting_shows() {
    // 4:2:0 is turned before it is converted (so the converter, chosen by width, and at 10 bits the
    // vertical chroma filter act on the upright picture); 4:2:2 and 4:4:4 are converted first.
    // At 48x40 the two orders give different bytes for 4:2:0 (a NEON width in one orientation and
    // not the other), which is what these fixtures hold the order to. The upright goldens are the
    // CLI's autorotate (`rgb_frames_are_byte_equal_to_the_oracle`); here the other order is shown
    // to be a different picture, so that test could fail.
    for name in [
        "hevc_420_709_tv_48x40_rot90",
        "hevc_p10_709_tv_48x40_rot90",
        "hevc_p10_601_pc_40x48_rot270",
    ] {
        let f = fact(name);
        let src = open(
            name,
            &VideoOptions {
                rotation: Rotation::Ignore,
                ..VideoOptions::default()
            },
        );
        let stored = src.decode_frames(&[0]).unwrap();
        // Convert-then-turn: turn the stored RGB.
        let (w, h) = (f.width, f.height);
        let turned: Vec<u8> = {
            let frame = &stored.frames[0];
            let mut out = vec![0u8; frame.len()];
            for y in 0..h {
                for x in 0..w {
                    let (nx, ny) = if f.rotation == 270 {
                        (h - 1 - y, x)
                    } else {
                        (y, w - 1 - x)
                    };
                    let dst = (ny * h + nx) * 3;
                    out[dst..dst + 3].copy_from_slice(&frame[(y * w + x) * 3..(y * w + x) * 3 + 3]);
                }
            }
            out
        };
        let want = rgb_golden(&f, ChromaFilter::Bilinear);
        let per = w * h * 3;
        assert_ne!(
            turned,
            &want[..per],
            "{name}: the order of turning and converting must show in these bytes"
        );
    }
}

#[test]
fn frame_count_duration_and_fps_are_torchcodecs_exact_mode() {
    for f in openable_facts() {
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
    for f in openable_facts() {
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
            layout: Layout {
                chroma: Chroma::C420,
                depth: 8
            },
            matrix: Matrix::Bt601,
            range: Range::Limited,
            flavor: Flavor::Aarch64,
            filter: ChromaFilter::Bilinear,
        }
    );
}

#[test]
fn the_matrix_tags_the_system_reports_include_fcc_and_smpte240m() {
    for (name, code, matrix) in [
        ("hevc_fcc_pc_64x48", 4, Matrix::Fcc),
        ("hevc_240m_pc_66x38", 7, Matrix::Smpte240m),
        ("hevc_p10_2020_tv_64x32", 9, Matrix::Bt2020Nc),
    ] {
        let src = open(name, &VideoOptions::default());
        assert_eq!(src.info().tags.matrix_code, Some(code), "{name}");
        let d = src.decode_frames(&[0]).unwrap();
        assert!(
            matches!(d.conversion, Conversion::Swscale { matrix: m, .. } if m == matrix),
            "{name}: {:?}",
            d.conversion
        );
    }
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
fn ten_bit_through_avfoundation_is_not_the_oracle_and_is_never_the_default() {
    // The 10-bit fixture was 4.06 mean absolute difference away when AVFoundation's conversion was
    // the fallback; now it is exact by default and the system's conversion only by request.
    let f = fact("hevc_main10_64x48");
    let opts = VideoOptions {
        color: ColorPath::AvFoundation,
        ..VideoOptions::default()
    };
    let d = open(&f.name, &opts)
        .decode_frames(&(0..f.frames).collect::<Vec<_>>())
        .unwrap();
    assert_eq!(d.conversion, Conversion::AvFoundation);
    let want = rgb_golden(&f, ChromaFilter::Bilinear);
    let got: Vec<u8> = d.frames.concat();
    let mean = got
        .iter()
        .zip(&want)
        .map(|(a, b)| f64::from(a.abs_diff(*b)))
        .sum::<f64>()
        / want.len() as f64;
    eprintln!("10-bit HEVC through AVFoundation vs the oracle: mean abs difference {mean:.3}");
    assert!(mean > 0.5 && mean < 6.0, "mean abs difference {mean}");
}

#[test]
fn streams_that_cannot_be_exact_are_refused_by_name_and_converted_only_on_request() {
    // (fixture, what the refusal says)
    for (name, reason) in [
        ("refuse_hevc_12bit_709_tv_64x48", "12-bit HEVC"),
        ("refuse_hevc_gray_64x48", "monochrome (4:0:0) HEVC"),
    ] {
        let src = open(name, &VideoOptions::default());
        let layout_error = src.info().layout.clone().expect_err(name);
        assert!(layout_error.contains(reason), "{name}: {layout_error}");
        match src.decode_frames(&[0]) {
            Err(MediaError::Unsupported { what: "video", detail }) => {
                assert!(
                    detail.contains(reason) && detail.contains("ColorPath::AvFoundation"),
                    "{name}: {detail}"
                );
            }
            other => panic!("{name}: expected a refusal by name, got {other:?}"),
        }
        assert!(src.decode_native_planes(&[0]).is_err(), "{name}");
        // Asking for the system's conversion is the way to get its frames, and says so.
        let on_request = open(
            name,
            &VideoOptions {
                color: ColorPath::AvFoundation,
                ..VideoOptions::default()
            },
        )
        .decode_frames(&[0, 1])
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(on_request.conversion, Conversion::AvFoundation, "{name}");
        assert_eq!(
            (on_request.width, on_request.height, on_request.frames.len()),
            (64, 48, 2),
            "{name}"
        );
    }
    // MPEG-4 Part 2 is not refused by this module but by the system: AVFoundation cannot decode
    // it, so no source opens on it, loudly and under either colour path.
    for color in [ColorPath::default(), ColorPath::AvFoundation] {
        let options = VideoOptions {
            color,
            ..VideoOptions::default()
        };
        match VideoSource::open(&fixture(&format!("{UNDECODABLE}.mp4")), &options) {
            Err(MediaError::Decode { what: "video", .. }) => {}
            Err(other) => panic!("{color:?}: expected a decode error, got {other}"),
            Ok(_) => panic!("{color:?}: a source opened on a stream nothing decodes"),
        }
    }
}
