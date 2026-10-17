//! Media decoding that needs the system's codec hardware: HEIC images and H.264 /
//! HEVC video (`tessl::media`), against the same oracles as `tests/media_decode.rs`.
//!
//! These are a separate binary, and not on `scripts/hosted_tests.txt`, because
//! ImageIO's HEIC decoder and AVFoundation's H.264 and HEVC decoders go through
//! VideoToolbox, which opens IOKit. The hosted job's sandbox denies `iokit-open`,
//! and under it every test here fails with a decode error (measured: HEIC
//! "pixel data is shorter than its size", video "no frame could be decoded");
//! it is a loud failure, never a silent one. Run them with
//! `cargo test --release --test media_codecs` on a Mac. What does not need a
//! codec (refusing cut or foreign files, every image and audio format) stays in
//! `tests/media_decode.rs`, which runs hosted.

mod media_support;

use media_support::*;
use tessl::embedgemma2_pre::{preprocess_video, sample_frame_indices};
use tessl::media::image::{decode_image, decode_image_file, ImageOptions};
use tessl::media::video::{preprocess_decoded_video, Rotation, VideoOptions, VideoSource};
use tessl::media::MediaError;

#[test]
fn heic_decodes_to_the_source_within_the_codec_error() {
    let d = decode_image_file(&fixture("photo.heic"), &ImageOptions::default()).unwrap();
    assert_eq!((d.width, d.height), (64, 48));
    assert_eq!(d.info.format, "public.heic");
    let want = bytes("photo.heic.src.rgb");
    let (max, n) = diff_bytes(&d.data, &want);
    let mean = d
        .data
        .iter()
        .zip(&want)
        .map(|(a, b)| f64::from(a.abs_diff(*b)))
        .sum::<f64>()
        / want.len() as f64;
    eprintln!("photo.heic: vs source max diff {max}, {n} bytes, mean {mean:.3}");
    assert!(mean < 6.0, "mean abs error {mean}");
}

#[test]
fn video_frames_match_ffmpeg_and_the_source_colours() {
    let oracle = bytes("colors.mp4.rgb24");
    let per = VW * VH * 3;
    for name in ["colors.mp4", "colors_hevc.mp4"] {
        let src = VideoSource::open(&fixture(name), &VideoOptions::default()).unwrap_or_else(|e| panic!("{name}: {e}"));
        let info = src.info();
        assert_eq!(info.total_frames, 12, "{name}");
        assert!((info.fps - 10.0).abs() < 1e-9, "{name}: fps {}", info.fps);
        assert_eq!(info.orientation, 1, "{name}");
        let want_pts = manifest_pts(name);
        assert_eq!(info.timestamps.len(), want_pts.len());
        for (i, (a, b)) in info.timestamps.iter().zip(&want_pts).enumerate() {
            assert!((a - b).abs() < 1e-6, "{name}: frame {i} at {a} s, ffprobe {b}");
        }
        let all = src.decode_frames(&(0..12).collect::<Vec<_>>()).unwrap();
        assert_eq!((all.width, all.height, all.frames.len()), (VW, VH, 12), "{name}");
        assert_eq!(all.indices, (0..12).collect::<Vec<_>>());
        assert_eq!(all.timestamps, info.timestamps);
        let mut worst_colour = 0u8;
        let mut worst_ffmpeg = (0u8, 0f64);
        for (i, frame) in all.frames.iter().enumerate() {
            assert_eq!(frame.len(), per);
            let c = centre_pixel(frame, VW, VH);
            let (r, g, b) = FRAME_COLORS[i];
            let d = c[0].abs_diff(r).max(c[1].abs_diff(g)).max(c[2].abs_diff(b));
            worst_colour = worst_colour.max(d);
            if name == "colors.mp4" {
                let (max, mean) = byte_stats(frame, &oracle[i * per..(i + 1) * per]);
                worst_ffmpeg = (worst_ffmpeg.0.max(max), worst_ffmpeg.1.max(mean));
            }
        }
        let vs_ffmpeg = if name == "colors.mp4" {
            format!("vs ffmpeg (max, mean) {worst_ffmpeg:?}")
        } else {
            "no ffmpeg oracle for this file".to_string()
        };
        eprintln!("{name}: centre colour worst {worst_colour} from the source; {vs_ffmpeg}");
        assert!(
            worst_colour <= 8,
            "{name}: a decoded colour is {worst_colour} from the source"
        );
        if name == "colors.mp4" {
            // The default path decodes the planes with VideoToolbox and converts
            // them with libswscale's arithmetic (src/media/yuv.rs), so every
            // byte equals the ffmpeg CLI's.
            assert_eq!(worst_ffmpeg, (0, 0.0), "{name}: vs ffmpeg (max, mean)");
        }
    }
}

#[test]
fn rotation_metadata_is_applied_or_ignored_as_asked() {
    let name = "quad_rot90.mp4";
    let applied = VideoSource::open(&fixture(name), &VideoOptions::default()).unwrap();
    let ignored = VideoSource::open(
        &fixture(name),
        &VideoOptions {
            rotation: Rotation::Ignore,
            ..VideoOptions::default()
        },
    )
    .unwrap();
    eprintln!("{name}: orientation {}", applied.info().orientation);
    assert_ne!(applied.info().orientation, 1);
    assert_eq!(applied.info().orientation, ignored.info().orientation);
    let idx: Vec<usize> = (0..12).collect();
    let a = applied.decode_frames(&idx).unwrap();
    let s = ignored.decode_frames(&idx).unwrap();
    assert_eq!((a.width, a.height), (VH, VW), "rotated");
    assert_eq!((s.width, s.height), (VW, VH), "as stored");
    let upright = bytes("quad_rot90.mp4.rgb24");
    let stored = bytes("quad_rot90.mp4.stored.rgb24");
    let cv2 = bytes("quad_rot90.mp4.cv2.rgb24");
    let per = VW * VH * 3;
    // (max, mean, share of bytes more than 8 off), worst over the frames. The
    // last row is FFmpeg against OpenCV, two independent decoders of the same
    // file, for scale: the frames have hard edges between saturated quadrants,
    // where decoders upsample the half-resolution chroma differently.
    let mut worst = [(0u8, 0f64, 0f64); 4];
    for i in 0..12 {
        let at = i * per..(i + 1) * per;
        let pairs: [(&[u8], &[u8]); 4] = [
            (&a.frames[i], &upright[at.clone()]),
            (&s.frames[i], &stored[at.clone()]),
            (&a.frames[i], &cv2[at.clone()]),
            (&upright[at.clone()], &cv2[at]),
        ];
        for (slot, (got, want)) in pairs.into_iter().enumerate() {
            let (max, mean) = byte_stats(got, want);
            let big = got.iter().zip(want).filter(|(x, y)| x.abs_diff(**y) > 8).count() as f64 / got.len() as f64;
            worst[slot] = (worst[slot].0.max(max), worst[slot].1.max(mean), worst[slot].2.max(big));
        }
    }
    eprintln!(
        "(max, mean, share > 8): rotated vs ffmpeg autorotate {:?}; stored vs ffmpeg -noautorotate {:?}; rotated vs OpenCV {:?}; ffmpeg vs OpenCV {:?}",
        worst[0], worst[1], worst[2], worst[3]
    );
    for (what, (_, mean, big)) in [("rotated/ffmpeg", worst[0]), ("stored/ffmpeg", worst[1])] {
        assert!(mean < 6.0 && big < 0.12, "{what}: mean {mean}, share over 8 {big}");
    }
    // OpenCV's colour conversion is its own (it maps 255 to 233 and 200 to 226 on
    // these frames, 9.4 mean from FFmpeg's), so it is an oracle for the rotation
    // only: against it the right orientation is 12.5 mean away and the wrong
    // ones 87 to 96 (the other readings of the same bytes, measured offline).
    assert!(worst[2].1 < 20.0, "rotated/opencv: mean {}", worst[2].1);
}

#[test]
fn sampled_frames_are_the_processors_indices_and_the_same_pixels() {
    let src = VideoSource::open(&fixture("colors.mp4"), &VideoOptions::default()).unwrap();
    let all = src.decode_frames(&(0..12).collect::<Vec<_>>()).unwrap();
    let mut cfg = pre_config().video;
    for fps in [Some(5.0), Some(2.0), Some(1.0), None] {
        cfg.fps = fps;
        cfg.do_sample_frames = true;
        let want = sample_frame_indices(&cfg, 12, 10.0).unwrap();
        let got = src.decode_sampled(&cfg).unwrap();
        assert_eq!(got.indices, want, "fps {fps:?}");
        assert_eq!(got.total_frames, 12);
        for (k, &i) in want.iter().enumerate() {
            assert_eq!(got.frames[k], all.frames[i], "fps {fps:?}, frame {i}");
            assert_eq!(got.timestamps[k], src.info().timestamps[i]);
        }
    }
    // Frames asked for out of order and twice come back in the order asked.
    let some = src.decode_frames(&[7, 2, 7, 11]).unwrap();
    assert_eq!(some.indices, vec![7, 2, 7, 11]);
    assert_eq!(some.frames[0], all.frames[7]);
    assert_eq!(some.frames[1], all.frames[2]);
    assert_eq!(some.frames[2], all.frames[7]);
    assert_eq!(some.frames[3], all.frames[11]);
    // Sampling off: every frame.
    cfg.do_sample_frames = false;
    assert_eq!(src.decode_sampled(&cfg).unwrap().indices, (0..12).collect::<Vec<_>>());
}

#[test]
fn bytes_and_path_give_the_same_video() {
    let by_path = VideoSource::open(&fixture("colors.mp4"), &VideoOptions::default()).unwrap();
    let by_bytes = VideoSource::open_bytes(&bytes("colors.mp4"), "mp4", &VideoOptions::default()).unwrap();
    assert_eq!(by_bytes.info(), by_path.info());
    let idx: Vec<usize> = (0..12).collect();
    assert_eq!(
        by_bytes.decode_frames(&idx).unwrap(),
        by_path.decode_frames(&idx).unwrap()
    );
    // The temporary file is gone once the source is dropped.
    let before = temp_video_files();
    drop(by_bytes);
    assert!(
        temp_video_files() < before || before == 0,
        "{before} -> {}",
        temp_video_files()
    );
}

#[test]
fn preprocessing_decoded_video_equals_preprocessing_all_its_frames() {
    let src = VideoSource::open(&fixture("colors.mp4"), &VideoOptions::default()).unwrap();
    let mut cfg = pre_config().video;
    cfg.fps = Some(5.0);
    cfg.do_sample_frames = true;
    // The reference: every frame decoded, sampled and preprocessed in one call.
    let all = src.decode_frames(&(0..12).collect::<Vec<_>>()).unwrap();
    let reference = preprocess_video(&cfg, &all.raw_frames(), src.info().fps).unwrap();
    // The helper: only the sampled frames decoded.
    let sampled = src.decode_sampled(&cfg).unwrap();
    assert!(sampled.frames.len() < all.frames.len());
    let got = preprocess_decoded_video(&cfg, &sampled).unwrap();
    assert_eq!(got.indices, reference.indices);
    assert_eq!(got.timestamps, reference.timestamps);
    assert_eq!(got.soft_tokens_per_frame, reference.soft_tokens_per_frame);
    assert_eq!(got.frames, reference.frames);
    // Without the helper the sampled frames would be sampled again and the
    // indices would be those of the subset, not of the video.
    let twice = preprocess_video(&cfg, &sampled.raw_frames(), src.info().fps).unwrap();
    assert_ne!(twice.indices, reference.indices);
}

#[test]
fn asking_a_video_for_what_is_not_there_is_an_error() {
    let src = VideoSource::open(&fixture("colors.mp4"), &VideoOptions::default()).unwrap();
    assert!(matches!(src.decode_frames(&[]), Err(MediaError::Invalid { .. })));
    assert!(matches!(src.decode_frames(&[12]), Err(MediaError::Invalid { .. })));
    let tight = VideoSource::open(
        &fixture("colors.mp4"),
        &VideoOptions {
            max_frames: 3,
            ..VideoOptions::default()
        },
    )
    .unwrap();
    assert!(matches!(
        tight.decode_frames(&[0, 1, 2, 3]),
        Err(MediaError::Limit { .. })
    ));
    let small = VideoSource::open(
        &fixture("colors.mp4"),
        &VideoOptions {
            max_bytes: 3 * VW * VH * 3 - 1,
            ..VideoOptions::default()
        },
    )
    .unwrap();
    assert!(matches!(small.decode_frames(&[0, 1, 2]), Err(MediaError::Limit { .. })));
    assert_eq!(small.decode_frames(&[0, 1]).unwrap().frames.len(), 2);
}

#[test]
fn every_cut_of_a_heic_is_an_error_or_the_whole_image() {
    let data = bytes("photo.heic");
    let full = decode_image(&data, &ImageOptions::default()).unwrap();
    let (mut failed, mut whole) = (0, 0);
    for cut in (1..data.len()).step_by((data.len() / 97).max(1)) {
        match decode_image(&data[..cut], &ImageOptions::default()) {
            Ok(d) if d.data == full.data => whole += 1,
            Ok(_) => panic!("photo.heic cut at {cut} of {} is a partial picture", data.len()),
            Err(_) => failed += 1,
        }
    }
    eprintln!("photo.heic: {failed} cuts failed, {whole} decoded to the whole image");
    assert!(failed > whole);
}
