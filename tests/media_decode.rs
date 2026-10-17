//! Native media decoding (`tessl::media`) against oracles, with no device.
//!
//! The fixtures under `tests/fixtures/media/` and the oracle outputs next to
//! them are made by `scripts/gen_media_fixtures.py` (see its docstring for what
//! each oracle is: `torchvision.io.decode_image(mode=RGB)`, which is what
//! sentence-transformers runs on an image path, PIL for TIFF and BMP, and for
//! audio and video FFmpeg, torchaudio and OpenCV, since torchcodec is not
//! installed there). Every bound below was set from a measurement printed by
//! the test that uses it (run with `--nocapture`), then fixed.
//!
//! Everything here runs in the hosted job's sandbox (IOKit denied). The HEIC
//! and H.264 / HEVC decodes, which need VideoToolbox, are in
//! `tests/media_codecs.rs`.

mod media_support;

use std::collections::HashMap;

use media_support::*;
use tessl::media::audio::{decode_audio_bytes, decode_audio_file, AudioContainer, AudioOptions, Downmix, Resampler};
use tessl::media::image::{
    decode_image, decode_image_file, decode_image_imageio, ColorHandling, DecodePath, ExifOrientation, Gray16,
    ImageLayout, ImageOptions,
};
use tessl::media::video::{VideoOptions, VideoSource};
use tessl::media::MediaError;

/// The channel mean, taken before resampling: what the `ffmpeg_mean` goldens
/// (`pan=mono|c0=0.5*c0+0.5*c1`, then `aresample`) are. The default
/// [`AudioOptions`] mix like torchcodec instead (libswresample's matrix).
fn mean() -> AudioOptions {
    AudioOptions {
        downmix: Downmix::Mean,
        ..AudioOptions::default()
    }
}

/// The same mean, resampled by the system `AudioConverter`: the path the
/// bounds of the three tests that use it were measured on.
fn apple_mean() -> AudioOptions {
    AudioOptions {
        resampler: Resampler::AudioToolbox,
        ..mean()
    }
}

fn opts(layout: ImageLayout) -> ImageOptions {
    ImageOptions {
        layout,
        ..ImageOptions::default()
    }
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

const LOSSLESS: &[&str] = &[
    "rgb.png",
    "rgba.png",
    "gray.png",
    "graya.png",
    "pal.png",
    "pal_trns.png",
    "rgb16.png",
    "gray16.png",
    "rgba16.png",
    "rgb.tiff",
    "rgba.tiff",
    "gray.tiff",
    "rgb.bmp",
    "gray.bmp",
    "rgba.bmp",
    "anim.gif",
    "p3.png",
    "srgb_icc.png",
];

#[test]
fn lossless_images_equal_the_oracle_byte_for_byte() {
    let m = manifest();
    let mut failures = Vec::new();
    for name in LOSSLESS {
        let (w, h): (usize, usize) = (m[*name]["w"].parse().unwrap(), m[*name]["h"].parse().unwrap());
        for (layout, ext, ch) in [(ImageLayout::Rgb, "rgb", 3), (ImageLayout::Rgba, "rgba", 4)] {
            let want = bytes(&format!("{name}.{ext}"));
            let got = decode_image_file(&fixture(name), &opts(layout)).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!((got.width, got.height, got.channels), (w, h, ch), "{name} {ext} shape");
            let (max, n) = diff_bytes(&got.data, &want);
            eprintln!(
                "{name:14} {ext:4} {:?} bpc={} {:?} alpha={} -> {n} of {} bytes differ (max {max})",
                got.info.path,
                got.info.bits_per_component,
                got.info.color_model,
                got.info.has_alpha,
                want.len()
            );
            if n != 0 {
                failures.push(format!("{name} {ext}: {n} bytes differ, max {max}"));
            }
        }
    }
    assert!(failures.is_empty(), "not bit exact:\n{}", failures.join("\n"));
}

#[test]
fn bytes_and_path_give_the_same_image() {
    for name in ["rgb.png", "rgba.png", "rgb.jpg", "anim.gif", "rgb.tiff"] {
        let a = decode_image_file(&fixture(name), &ImageOptions::default()).unwrap();
        let b = decode_image(&bytes(name), &ImageOptions::default()).unwrap();
        assert_eq!(a, b, "{name}");
    }
}

#[test]
fn every_lossless_case_takes_the_direct_path() {
    for name in LOSSLESS.iter().filter(|n| **n != "p3.png") {
        let d = decode_image_file(&fixture(name), &ImageOptions::default()).unwrap();
        assert_eq!(d.info.path, DecodePath::Direct, "{name}");
    }
}

#[test]
fn jpeg_equals_the_oracle_byte_for_byte_and_imageio_does_not() {
    // libjpeg-turbo (torchvision and PIL, identical to each other here) against
    // the native decoder: the same bytes. Against ImageIO's decoder, which uses
    // a different IDCT and chroma upsampling, the measured gap below.
    for name in ["rgb.jpg", "rgb444.jpg", "gray.jpg"] {
        let want = bytes(&format!("{name}.rgb"));
        let got = decode_image_file(&fixture(name), &ImageOptions::default()).unwrap();
        assert_eq!(got.info.path, DecodePath::Jpeg, "{name}");
        assert_eq!(got.info.format, "public.jpeg", "{name}");
        assert_eq!(
            diff_bytes(&got.data, &want),
            (0, 0),
            "{name}: native decoder vs libjpeg-turbo"
        );

        // ImageIO is still reachable for comparison, and still not libjpeg-turbo.
        let io = decode_image_imageio(&bytes(name), &ImageOptions::default()).unwrap();
        let (max, n) = diff_bytes(&io.data, &want);
        let mean = io
            .data
            .iter()
            .zip(&want)
            .map(|(a, b)| f64::from(a.abs_diff(*b)))
            .sum::<f64>()
            / want.len() as f64;
        eprintln!(
            "{name:12} ImageIO vs libjpeg-turbo: max diff {max}, {n} of {} bytes differ, mean abs {mean:.3}",
            want.len()
        );
        assert!(
            max <= IMAGEIO_MAX,
            "{name}: max diff {max} over the bound {IMAGEIO_MAX}"
        );
        assert!(
            mean <= IMAGEIO_MEAN,
            "{name}: mean abs diff {mean:.3} over {IMAGEIO_MEAN}"
        );
    }
}

/// ImageIO's decoder against libjpeg-turbo: 4:2:0 `rgb.jpg` max 43 (3083 of
/// 9216 bytes, mean 0.745, at the chroma edges of the disc), 4:4:4 max 3 (mean
/// 0.169), gray max 1 (mean 0.057). The bound leaves a little room for another
/// machine's decoder (Apple has hardware and software JPEG paths).
const IMAGEIO_MAX: u8 = 48;
const IMAGEIO_MEAN: f64 = 1.0;

#[test]
fn exif_orientation_is_ignored_by_default_and_applied_on_request() {
    let m = manifest();
    for o in 1..=8 {
        let name = format!("exif{o}.jpg");
        check_exif(&name, &m[&name]);
    }
    check_exif("exif6.png", &m["exif6.png"]);
}

fn check_exif(name: &str, kv: &HashMap<String, String>) {
    let p = |k: &str| -> usize { kv[k].parse().unwrap() };
    let orientation: u32 = kv["orientation"].parse().unwrap();
    // The default is torchvision.io.decode_image(path): the stored pixels.
    let stored = decode_image_file(&fixture(name), &ImageOptions::default()).unwrap();
    assert_eq!((stored.width, stored.height), (p("w"), p("h")), "{name} stored size");
    assert_eq!(stored.info.exif_orientation, orientation, "{name} reported orientation");
    let (max, n) = diff_bytes(&stored.data, &bytes(&format!("{name}.rgb")));
    // JPEG and PNG alike: the stored pixels are the oracle's, byte for byte.
    assert_eq!((max, n), (0, 0), "{name} ignore: max {max}, {n} bytes");
    // `Apply` is PIL's exif_transpose of the same file.
    let applied = decode_image_file(
        &fixture(name),
        &ImageOptions {
            orientation: ExifOrientation::Apply,
            ..ImageOptions::default()
        },
    )
    .unwrap();
    assert_eq!(
        (applied.width, applied.height),
        (p("tw"), p("th")),
        "{name} transposed size"
    );
    let want = bytes(&format!("{name}.exif.rgb"));
    let (max, n) = diff_bytes(&applied.data, &want);
    eprintln!("{name}: orientation {orientation}, Apply vs exif_transpose max diff {max} ({n} bytes)");
    assert_eq!((max, n), (0, 0), "{name} apply: max {max}, {n} bytes");
}

#[test]
fn colour_handling_native_matches_pil_and_srgb_converts() {
    // Native: the stored samples, as libpng/PIL return them, even for Display P3.
    let native = decode_image_file(&fixture("p3.png"), &ImageOptions::default()).unwrap();
    assert_eq!(native.info.path, DecodePath::Direct);
    assert_eq!(diff_bytes(&native.data, &bytes("p3.png.rgb")), (0, 0));
    // Srgb: Quartz converts the Display P3 samples.
    let srgb = decode_image_file(
        &fixture("p3.png"),
        &ImageOptions {
            color: ColorHandling::Srgb,
            ..ImageOptions::default()
        },
    )
    .unwrap();
    assert_eq!(srgb.info.path, DecodePath::SrgbContext);
    let (max, n) = diff_bytes(&srgb.data, &native.data);
    eprintln!(
        "p3.png: sRGB conversion moves {n} of {} bytes, max {max}",
        native.data.len()
    );
    assert!(
        n > native.data.len() / 4 && max >= 8,
        "the P3 -> sRGB conversion should be visible"
    );
    // An image already in sRGB is not touched.
    let s = decode_image_file(
        &fixture("srgb_icc.png"),
        &ImageOptions {
            color: ColorHandling::Srgb,
            ..ImageOptions::default()
        },
    )
    .unwrap();
    assert_eq!(s.info.path, DecodePath::Direct);
    assert_eq!(diff_bytes(&s.data, &bytes("srgb_icc.png.rgb")), (0, 0));
}

#[test]
fn sixteen_bit_gray_follows_pil_unless_told_otherwise() {
    let clip = decode_image_file(&fixture("gray16.png"), &ImageOptions::default()).unwrap();
    assert_eq!(diff_bytes(&clip.data, &bytes("gray16.png.rgb")), (0, 0));
    let high = decode_image_file(
        &fixture("gray16.png"),
        &ImageOptions {
            gray16: Gray16::HighByte,
            ..ImageOptions::default()
        },
    )
    .unwrap();
    // The high byte of a ramp that spans 0..65535 is a ramp; PIL's clip is white
    // past the first 256 samples.
    assert_ne!(clip.data, high.data);
    let white = clip.data.iter().filter(|&&v| v == 255).count();
    let ramp_max = high.data.iter().copied().max().unwrap();
    assert!(
        white > clip.data.len() / 2,
        "PIL's clip blows most of the ramp out: {white}"
    );
    assert!(ramp_max >= 250, "the high byte keeps the ramp: max {ramp_max}");
}

#[test]
fn cmyk_jpeg_is_converted_the_way_torchvision_does_it_and_equals_pil() {
    // Quartz converts CMYK through a colour profile and lands near PIL's plain
    // conversion; the native path applies torchvision's `k * cmy / 255` and
    // equals it exactly.
    let d = decode_image_file(&fixture("cmyk.jpg"), &ImageOptions::default()).unwrap();
    assert_eq!(d.info.path, DecodePath::Jpeg);
    assert_eq!(d.info.color_model, tessl::media::image::ColorModel::Cmyk);
    let want = bytes("cmyk.jpg.rgb");
    assert_eq!(diff_bytes(&d.data, &want), (0, 0), "cmyk.jpg vs the oracle");
    // `ColorHandling::Srgb` is colour management, which stays with Quartz.
    let q = decode_image_file(
        &fixture("cmyk.jpg"),
        &ImageOptions {
            color: ColorHandling::Srgb,
            ..ImageOptions::default()
        },
    )
    .unwrap();
    assert_eq!(q.info.path, DecodePath::SrgbContext);
}

#[test]
fn undecodable_images_are_errors_never_empty_results() {
    let png = bytes("rgb.png");
    let jpg = bytes("rgb.jpg");
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", Vec::new()),
        (
            "garbage",
            (0..4096u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect(),
        ),
        ("text", b"this is not an image at all".to_vec()),
        ("png header only", png[..40].to_vec()),
        ("png truncated", png[..png.len() / 2].to_vec()),
        ("jpeg truncated", jpg[..jpg.len() / 2].to_vec()),
        ("jpeg header only", jpg[..200].to_vec()),
    ];
    for (what, data) in cases {
        match decode_image(&data, &ImageOptions::default()) {
            Ok(d) => panic!("{what}: decoded to a {}x{} image instead of failing", d.width, d.height),
            Err(e) => eprintln!("{what:18} -> {e}"),
        }
    }
    let missing = decode_image_file(&fixture("does_not_exist.png"), &ImageOptions::default());
    assert!(matches!(missing, Err(MediaError::Io { .. })), "{missing:?}");
    assert!(matches!(
        decode_image(&[], &ImageOptions::default()),
        Err(MediaError::Empty { .. })
    ));
    let dir_err = decode_image_file(&dir(), &ImageOptions::default());
    assert!(matches!(dir_err, Err(MediaError::Invalid { .. })), "{dir_err:?}");
}

#[test]
fn every_truncation_is_an_error_or_the_whole_image() {
    // Cut each format at every few bytes. A cut file either fails or, when the
    // cut only drops bytes after the last pixel (a trailer), decodes to exactly
    // the full image; it never yields a partial picture.
    let mut partial = Vec::new();
    for name in [
        "rgb.png",
        "rgba16.png",
        "pal.png",
        "rgb.jpg",
        "gray.jpg",
        "rgb.tiff",
        "rgba.tiff",
        "gray.tiff",
        "rgb.bmp",
        "gray.bmp",
        "rgba.bmp",
        "anim.gif",
    ] {
        let full_bytes = bytes(name);
        let full = decode_image(&full_bytes, &ImageOptions::default()).unwrap();
        let mut ok_cuts = 0;
        let mut err_cuts = 0;
        let mut bad_cuts = 0;
        let step = (full_bytes.len() / 97).max(1);
        for cut in (1..full_bytes.len()).step_by(step) {
            match decode_image(&full_bytes[..cut], &ImageOptions::default()) {
                Ok(d) if d.data == full.data => ok_cuts += 1,
                Ok(_) => {
                    bad_cuts += 1;
                    if bad_cuts == 1 {
                        partial.push(format!(
                            "{name}: cut at {cut} of {} is a partial picture",
                            full_bytes.len()
                        ));
                    }
                }
                Err(_) => err_cuts += 1,
            }
        }
        eprintln!("{name}: {err_cuts} cuts failed, {ok_cuts} decoded to the whole image, {bad_cuts} partial pictures");
        assert!(err_cuts > ok_cuts, "{name}: most cuts must fail");
    }
    assert!(partial.is_empty(), "{}", partial.join("\n"));
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

fn snr_db(reference: &[f32], got: &[f32]) -> f64 {
    let n = reference.len().min(got.len());
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    for i in 0..n {
        let r = f64::from(reference[i]);
        sig += r * r;
        err += (f64::from(got[i]) - r).powi(2);
    }
    10.0 * (sig / err.max(1e-30)).log10()
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
        .fold(0.0, f64::max)
}

/// Drop `edge` samples at each end: a windowed-sinc resampler's first and last
/// milliseconds see zeros past the clip.
fn interior(x: &[f32], edge: usize) -> &[f32] {
    &x[edge..x.len() - edge]
}

const EDGE: usize = 160; // 10 ms at 16 kHz

const EXACT_AUDIO: &[&str] = &[
    "pcm16_16k_mono.wav",
    "pcm24_16k_mono.wav",
    "pcm32_16k_mono.wav",
    "pcm8_16k_mono.wav",
    "f32_16k_mono.wav",
    "pcm16_16k_mono.aiff",
    "pcm16_16k_mono.caf",
    "pcm16_16k_mono.flac",
];

#[test]
fn audio_already_at_16k_mono_decodes_bit_for_bit() {
    for name in EXACT_AUDIO {
        let want = f32s(&format!("{name}.f32"));
        let got = decode_audio_file(&fixture(name), &AudioOptions::default()).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(got.sample_rate, 16_000);
        assert_eq!((got.source_channels, got.source_rate as u32), (1, 16_000), "{name}");
        assert_eq!(got.samples.len(), want.len(), "{name} length");
        let bad = got
            .samples
            .iter()
            .zip(&want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            bad,
            0,
            "{name}: {bad} of {} samples differ from the reference's (max abs {:.3e})",
            want.len(),
            max_abs(&got.samples, &want)
        );
        // The same bytes through the callback path.
        let from_bytes = decode_audio_bytes(&bytes(name), AudioContainer::Auto, &AudioOptions::default()).unwrap();
        assert_eq!(from_bytes.samples, got.samples, "{name}: bytes vs path");
    }
}

/// `(file, source rate, channels)` of the resampling fixtures.
const RESAMPLED: &[(&str, u32, u32)] = &[
    ("pcm16_44k_stereo.wav", 44_100, 2),
    ("pcm16_48k_stereo.wav", 48_000, 2),
    ("pcm16_22k_mono.wav", 22_050, 1),
    ("pcm16_8k_mono.wav", 8_000, 1),
];

#[test]
fn resampled_audio_is_measured_against_three_references() {
    for &(name, rate, ch) in RESAMPLED {
        let truth = f32s(&format!("{name}.truth.f32"));
        let ta = f32s(&format!("{name}.torchaudio.f32"));
        let ffm = f32s(&format!("{name}.ffmpeg_mean.f32"));
        let got = decode_audio_file(&fixture(name), &apple_mean()).unwrap();
        assert_eq!((got.source_rate as u32, got.source_channels), (rate, ch), "{name}");
        assert_eq!(got.samples.len(), truth.len(), "{name}: output length");
        let s = |r: &[f32]| {
            (
                snr_db(interior(r, EDGE), interior(&got.samples, EDGE)),
                snr_db(r, &got.samples),
            )
        };
        let (t_in, t_all) = s(&truth);
        let (a_in, a_all) = s(&ta);
        let (f_in, f_all) = s(&ffm);
        // For scale: the references against the truth and each other.
        let ta_truth = snr_db(interior(&truth, EDGE), interior(&ta, EDGE));
        let ff_truth = snr_db(interior(&truth, EDGE), interior(&ffm, EDGE));
        eprintln!(
            "{name:22} vs truth {t_in:6.1} dB ({t_all:6.1} with edges) | vs torchaudio {a_in:6.1} ({a_all:6.1}) | vs ffmpeg {f_in:6.1} ({f_all:6.1}) | torchaudio vs truth {ta_truth:5.1}, ffmpeg vs truth {ff_truth:5.1}; max abs vs truth {:.2e}",
            max_abs(interior(&truth, EDGE), interior(&got.samples, EDGE))
        );
    }
}

#[test]
fn swresample_downmix_matches_ffmpegs_default_and_mean_is_quieter_by_root_two() {
    let name = "pcm16_44k_stereo.wav";
    let ffmpeg_default = f32s(&format!("{name}.ffmpeg.f32"));
    let sw = decode_audio_file(
        &fixture(name),
        &AudioOptions {
            downmix: Downmix::Swresample,
            resampler: Resampler::AudioToolbox,
            ..AudioOptions::default()
        },
    )
    .unwrap();
    let mean = decode_audio_file(&fixture(name), &apple_mean()).unwrap();
    let snr = snr_db(interior(&ffmpeg_default, EDGE), interior(&sw.samples, EDGE));
    eprintln!(
        "Swresample downmix vs ffmpeg -ac 1: {snr:.1} dB; mean vs ffmpeg -ac 1: {:.1} dB",
        snr_db(interior(&ffmpeg_default, EDGE), interior(&mean.samples, EDGE))
    );
    assert!(snr > 78.0, "{snr}");
    // The two downmixes differ by exactly sqrt(2) in gain.
    let ratio: f64 = interior(&sw.samples, EDGE)
        .iter()
        .zip(interior(&mean.samples, EDGE))
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum::<f64>()
        / interior(&mean.samples, EDGE)
            .iter()
            .map(|b| f64::from(*b).powi(2))
            .sum::<f64>();
    assert!((ratio - std::f64::consts::SQRT_2).abs() < 1e-3, "gain ratio {ratio}");
}

#[test]
fn resampled_audio_is_within_the_measured_bounds() {
    // Measured (see `resampled_audio_is_measured_against_three_references`):
    // interior SNR against the analytic 16 kHz signal 86.1 dB (44.1 and 48 kHz
    // stereo), 91.8 (22.05 kHz), 69.7 (8 kHz up); against FFmpeg's resampler
    // 83.8, 83.8, 93.0, 69.8; against torchaudio's 66.1, 66.1, 66.8, 51.3.
    // Bounds are the measurement less 3 to 6 dB. The first and last 10 ms see
    // the converter's start-up and flush; with them included the SNR is 41 to
    // 59 dB.
    let bounds: [(&str, f64, f64, f64, f64); 4] = [
        // name, vs truth, vs ffmpeg, vs torchaudio, with edges vs truth
        ("pcm16_44k_stereo.wav", 82.0, 80.0, 62.0, 42.0),
        ("pcm16_48k_stereo.wav", 82.0, 80.0, 62.0, 42.0),
        ("pcm16_22k_mono.wav", 86.0, 87.0, 62.0, 54.0),
        ("pcm16_8k_mono.wav", 65.0, 65.0, 47.0, 37.0),
    ];
    for (name, b_truth, b_ffmpeg, b_ta, b_edges) in bounds {
        let truth = f32s(&format!("{name}.truth.f32"));
        let ta = f32s(&format!("{name}.torchaudio.f32"));
        let ffm = f32s(&format!("{name}.ffmpeg_mean.f32"));
        let got = decode_audio_file(&fixture(name), &apple_mean()).unwrap();
        assert_eq!(got.samples.len(), truth.len(), "{name}: length");
        let g = interior(&got.samples, EDGE);
        let checks = [
            ("truth", snr_db(interior(&truth, EDGE), g), b_truth),
            ("ffmpeg", snr_db(interior(&ffm, EDGE), g), b_ffmpeg),
            ("torchaudio", snr_db(interior(&ta, EDGE), g), b_ta),
            ("truth with edges", snr_db(&truth, &got.samples), b_edges),
        ];
        for (what, snr, bound) in checks {
            assert!(snr >= bound, "{name} vs {what}: {snr:.1} dB under the bound {bound}");
        }
    }
}

/// `(file, channels, floor of SNR vs ffmpeg, floor of SNR vs the analytic
/// signal)` of the compressed fixtures, decoded with the channel mean and the
/// libswresample resampler. Floors sit a few dB under what was measured
/// (138.8/33.2, 92.6/26.0, 87.7/52.2, 89.5/26.0, 135.9/30.9, 14.6/14.5): the
/// lossy codecs are 26-33 dB from the signal they encoded, FFmpeg's own decode
/// being as far (mp3: 26.0). The stereo FLAC's 87.7 dB is the golden's, not the
/// decode's: `pan` in `gen_media_fixtures.py` quantises the mix to 16 bits
/// (`tests/media_resample.rs` holds the float mean to the bit). Ogg Opus is
/// discussed in the test below.
const COMPRESSED: &[(&str, u32, f64, f64)] = &[
    ("tone_44k_stereo.m4a", 2, 130.0, 32.0),
    ("tone_44k_stereo.mp3", 2, 88.0, 25.0),
    ("tone_44k_stereo.flac", 2, 84.0, 50.0),
    ("tone_16k_mono.mp3", 1, 85.0, 25.0),
    ("tone_16k_mono.m4a", 1, 130.0, 30.0),
    ("tone_16k_mono.opus", 1, 14.0, 14.0),
];

#[test]
fn compressed_audio_decodes_with_the_expected_length_and_content() {
    let truth_st = f32s("tone_stereo_long.truth.f32");
    let truth_mo = f32s("tone_mono_long.truth.f32");
    for &(name, ch, floor_ffmpeg, floor_truth) in COMPRESSED {
        let ffm = f32s(&format!("{name}.ffmpeg_mean.f32"));
        let truth = if ch == 2 { &truth_st } else { &truth_mo };
        let got = decode_audio_file(&fixture(name), &mean()).unwrap_or_else(|e| panic!("{name}: {e}"));
        let bytes_path = decode_audio_bytes(&bytes(name), AudioContainer::Auto, &mean()).unwrap();
        // Codec output is float arithmetic whose last bit depends on buffer
        // alignment (the same file differs by ~2e-8 between runs of the test
        // binary that allocate differently), so path and bytes agree to a
        // tolerance here; the integer-PCM cases above are compared bit for bit.
        assert_eq!(
            bytes_path.samples.len(),
            got.samples.len(),
            "{name}: bytes vs path length"
        );
        assert!(
            max_abs(&bytes_path.samples, &got.samples) < 1e-6,
            "{name}: bytes vs path"
        );
        assert_eq!(got.source_channels, ch, "{name}");
        // Best alignment against the truth, in 16 kHz samples (an encoder delay
        // not trimmed would show here).
        let lag = best_lag(truth, &got.samples, 64);
        let (a, b) = if lag >= 0 {
            (&truth[lag as usize..], &got.samples[..])
        } else {
            (&truth[..], &got.samples[(-lag) as usize..])
        };
        eprintln!(
            "{name:22} {} frames (ffmpeg {}, truth {}), source {:?} frames, lag {lag}; SNR vs ffmpeg {:.1} dB, vs truth {:.1} dB (aligned {:.1})",
            got.samples.len(),
            ffm.len(),
            truth.len(),
            got.source_frames,
            snr_db(&ffm, &got.samples),
            snr_db(truth, &got.samples),
            snr_db(a, b)
        );
        assert_eq!(got.samples.len(), truth.len(), "{name}: frames");
        assert_eq!(got.samples.len(), ffm.len(), "{name}: frames vs ffmpeg");
        assert_eq!(lag, 0, "{name}: best alignment against the signal");
        let (vs_ffmpeg, vs_truth) = (snr_db(&ffm, &got.samples), snr_db(truth, &got.samples));
        assert!(
            vs_ffmpeg >= floor_ffmpeg,
            "{name}: {vs_ffmpeg:.1} dB vs ffmpeg under {floor_ffmpeg}"
        );
        assert!(
            vs_truth >= floor_truth,
            "{name}: {vs_truth:.1} dB vs the signal under {floor_truth}"
        );
        if name.ends_with(".opus") {
            // The golden is FFmpeg's *native* Opus decoder (what `ffmpeg` and
            // torchcodec run by default), which is two 48 kHz samples behind
            // libopus in the SILK frames of this file, its first 18000 samples
            // (6000 at 16 kHz), and identical to AudioToolbox's after that.
            // AudioToolbox's decode is not behind libopus: it matches it to
            // the bit in SILK frames (tests/media_resample.rs measures this
            // against libopus itself).
            let tail = snr_db(&ffm[6000..], &got.samples[6000..]);
            eprintln!("{name}: SNR vs ffmpeg after the first 6000 samples {tail:.1} dB");
            assert!(
                tail >= 130.0,
                "{name}: {tail:.1} dB vs ffmpeg after the first 6000 samples"
            );
        }
    }
}

/// The shift `k` in `-max..=max` that maximises the correlation of `got[i]`
/// with `truth[i + k]`.
fn best_lag(truth: &[f32], got: &[f32], max: i64) -> i64 {
    let n = truth.len().min(got.len()) as i64 - max;
    (-max..=max)
        .max_by(|&p, &q| {
            let c = |k: i64| -> f64 {
                (max..n)
                    .map(|i| f64::from(got[i as usize]) * f64::from(truth[(i + k) as usize]))
                    .sum()
            };
            c(p).total_cmp(&c(q))
        })
        .unwrap()
}

#[test]
fn an_unsupported_container_is_an_error_not_silence() {
    // WavPack: a real file in a format AudioToolbox does not decode.
    let r = decode_audio_file(&fixture("tone_16k_mono.wv"), &AudioOptions::default());
    eprintln!("wavpack -> {r:?}");
    assert!(r.is_err());
}

#[test]
fn undecodable_audio_is_an_error_never_an_empty_result() {
    let wav = bytes("pcm16_16k_mono.wav");
    let mut header_only = wav[..44].to_vec();
    // Declare zero data bytes: a valid WAV with no samples.
    header_only[40..44].copy_from_slice(&0u32.to_le_bytes());
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", Vec::new()),
        (
            "garbage",
            (0..8192u32).map(|i| (i.wrapping_mul(2654435761) >> 11) as u8).collect(),
        ),
        ("text", b"RIFF but not really a wave file".to_vec()),
        ("wav header only", header_only),
        ("wav truncated", wav[..wav.len() / 2].to_vec()),
        ("wav cut in the header", wav[..30].to_vec()),
        ("m4a truncated", bytes("tone_16k_mono.m4a")[..3000].to_vec()),
        ("flac truncated", bytes("pcm16_16k_mono.flac")[..9000].to_vec()),
        ("aiff truncated", bytes("pcm16_16k_mono.aiff")[..6000].to_vec()),
        ("caf truncated", bytes("pcm16_16k_mono.caf")[..6000].to_vec()),
    ];
    for (what, data) in cases {
        match decode_audio_bytes(&data, AudioContainer::Auto, &AudioOptions::default()) {
            Ok(a) => panic!("{what}: decoded {} samples instead of failing", a.samples.len()),
            Err(e) => eprintln!("{what:22} -> {e}"),
        }
        // The same bytes as a file.
        let path = std::env::temp_dir().join(format!(
            "tessl-media-test-{}-{}.bin",
            std::process::id(),
            what.replace(' ', "_")
        ));
        std::fs::write(&path, &data).unwrap();
        let r = decode_audio_file(&path, &AudioOptions::default());
        let _ = std::fs::remove_file(&path);
        assert!(
            r.is_err(),
            "{what}: the file path decoded {} samples",
            r.map(|a| a.samples.len()).unwrap_or(0)
        );
    }
    let missing = decode_audio_file(&fixture("does_not_exist.wav"), &AudioOptions::default());
    assert!(matches!(missing, Err(MediaError::Io { .. })), "{missing:?}");
    assert!(matches!(
        decode_audio_bytes(&[], AudioContainer::Auto, &AudioOptions::default()),
        Err(MediaError::Empty { .. })
    ));
}

#[test]
fn the_length_limit_errors_or_truncates_as_asked() {
    let name = "pcm16_16k_mono.wav"; // 0.4 s
    let tight = AudioOptions {
        max_seconds: 0.1,
        ..AudioOptions::default()
    };
    assert!(matches!(
        decode_audio_file(&fixture(name), &tight),
        Err(MediaError::Limit { .. })
    ));
    let cut = decode_audio_file(
        &fixture(name),
        &AudioOptions {
            truncate: true,
            ..tight
        },
    )
    .unwrap();
    assert!(cut.truncated);
    assert_eq!(cut.samples.len(), 1600);
    let full = f32s(&format!("{name}.f32"));
    assert_eq!(cut.samples, full[..1600]);
    // Exactly at the limit is not a truncation.
    let exact = decode_audio_file(
        &fixture(name),
        &AudioOptions {
            max_seconds: 0.4,
            ..AudioOptions::default()
        },
    )
    .unwrap();
    assert!(!exact.truncated);
    assert_eq!(exact.samples.len(), 6400);
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let r = decode_audio_file(
            &fixture(name),
            &AudioOptions {
                max_seconds: bad,
                ..AudioOptions::default()
            },
        );
        assert!(matches!(r, Err(MediaError::Invalid { .. })), "{bad}: {r:?}");
    }
    let r = decode_audio_file(
        &fixture(name),
        &AudioOptions {
            sample_rate: 0,
            ..AudioOptions::default()
        },
    );
    assert!(matches!(r, Err(MediaError::Invalid { .. })));
}

#[test]
fn an_oversized_image_is_refused_before_it_is_decoded() {
    let small = ImageOptions {
        max_pixels: 100,
        ..ImageOptions::default()
    };
    let r = decode_image_file(&fixture("rgb.png"), &small);
    assert!(matches!(r, Err(MediaError::Limit { .. })), "{r:?}");
}

// ---------------------------------------------------------------------------
// Video
// ---------------------------------------------------------------------------

#[test]
fn every_cut_of_a_video_is_an_error_or_the_whole_video() {
    // colors.mp4 has moov first, so a cut anywhere in mdat leaves a track table
    // that promises frames the file no longer holds. quad_rot90.mp4 has moov
    // last, so any cut loses the table.
    let mut cuts = 0;
    for name in ["colors.mp4", "quad_rot90.mp4", "colors_hevc.mp4"] {
        let data = bytes(name);
        for len in (1..data.len()).step_by(7).chain([data.len() - 1]) {
            cuts += 1;
            if let Ok(v) = VideoSource::open_bytes(&data[..len], "mp4", &VideoOptions::default()) {
                panic!(
                    "{name} cut to {len} of {} bytes opened with {} frames",
                    data.len(),
                    v.info().total_frames
                );
            }
        }
    }
    eprintln!("{cuts} cuts, all refused");
}

#[test]
fn undecodable_video_is_an_error_never_an_empty_result() {
    let good = bytes("colors.mp4");
    let m4a = bytes("tone_16k_mono.m4a");
    let png = bytes("rgb.png");
    let cases: Vec<(&str, Vec<u8>, &str)> = vec![
        ("empty", Vec::new(), "mp4"),
        (
            "garbage",
            (0..8192u32).map(|i| (i.wrapping_mul(2654435761) >> 11) as u8).collect(),
            "mp4",
        ),
        ("a png", png, "mp4"),
        ("audio only", m4a, "mp4"),
        ("truncated in mdat", good[..good.len() / 2].to_vec(), "mp4"),
        ("truncated to the header", good[..40].to_vec(), "mp4"),
        ("bad extension", good.clone(), "../x"),
    ];
    for (what, data, ext) in cases {
        match VideoSource::open_bytes(&data, ext, &VideoOptions::default()) {
            Ok(v) => panic!("{what}: opened a video of {} frames", v.info().total_frames),
            Err(e) => eprintln!("{what:24} -> {e}"),
        }
    }
    let missing = VideoSource::open(&fixture("does_not_exist.mp4"), &VideoOptions::default());
    assert!(matches!(missing, Err(MediaError::Io { .. })), "{missing:?}");
    let dir_as_file = VideoSource::open(&dir(), &VideoOptions::default());
    assert!(
        matches!(dir_as_file, Err(MediaError::Invalid { .. })),
        "{dir_as_file:?}"
    );
}
