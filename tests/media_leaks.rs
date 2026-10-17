//! Decoding in a loop must not grow the process: every CoreFoundation and
//! AudioToolbox object `tessl::media` creates is released on every path.
//!
//! One test, so no other thread allocates while the resident set is measured.
//! Each loop is warmed up (framework caches fill once), then the growth over
//! many more rounds is bounded. A leak of one decoded image per call would be
//! tens of MiB over the rounds below; the bound is 256 KiB for the
//! least-growing of three batches of 100 (see `rss_growth_kib`). The failing
//! paths (a cut file, a refused size) are looped too, because an error return that
//! skips a release is the usual leak. The HEIC and video loops are in
//! `tests/media_codecs_leaks.rs`; `leaks --atExit` was run on this binary as
//! well (see `docs/embedgemma2.md`, "Media decoding").

mod media_support;

use media_support::*;
use tessl::media::audio::{decode_audio_bytes, decode_audio_file, AudioContainer, AudioOptions, Downmix};
use tessl::media::image::{decode_image, decode_image_file, ColorHandling, ExifOrientation, ImageOptions};

const BOUND_KIB: i64 = 256;

fn check(what: &str, warmup: usize, rounds: usize, f: impl FnMut()) {
    let (growth, per_round) = rss_growth_kib(warmup, rounds, f);
    eprintln!("{what:48} {rounds} rounds: {growth:+} KiB ({per_round:+.2} KiB per round)");
    assert!(
        growth < BOUND_KIB,
        "{what}: resident set grew {growth} KiB over {rounds} rounds"
    );
}

#[test]
fn decode_loops_do_not_grow_the_process() {
    let default = ImageOptions::default();
    let srgb = ImageOptions {
        color: ColorHandling::Srgb,
        orientation: ExifOrientation::Apply,
        ..ImageOptions::default()
    };
    let png = bytes("rgb.png");
    let jpg = bytes("rgb.jpg");
    let cmyk = bytes("cmyk.jpg");
    let cut_tiff = {
        let t = bytes("rgb.tiff");
        t[..t.len() / 2].to_vec()
    };
    check("image: png from bytes", 30, 100, || {
        decode_image(&png, &default).unwrap();
    });
    check("image: jpeg from a path", 30, 100, || {
        decode_image_file(&fixture("rgb.jpg"), &default).unwrap();
    });
    check("image: jpeg from bytes", 30, 100, || {
        decode_image(&jpg, &default).unwrap();
    });
    check("image: cmyk jpeg through the sRGB context", 30, 100, || {
        decode_image(&cmyk, &default).unwrap();
    });
    check("image: p3 png converted to sRGB, exif applied", 30, 100, || {
        decode_image_file(&fixture("p3.png"), &srgb).unwrap();
    });
    check("image: a cut tiff (refused)", 30, 100, || {
        assert!(decode_image(&cut_tiff, &default).is_err());
    });
    check("image: not an image (refused)", 30, 100, || {
        assert!(decode_image(b"not an image at all", &default).is_err());
    });

    let wav = bytes("pcm16_44k_stereo.wav");
    let m4a = bytes("tone_44k_stereo.m4a");
    let mp3 = bytes("tone_16k_mono.mp3");
    let cut_wav = wav[..wav.len() / 2].to_vec();
    let audio = AudioOptions::default();
    let swr = AudioOptions {
        downmix: Downmix::Swresample,
        ..AudioOptions::default()
    };
    check("audio: wav 44.1k stereo resampled, from a path", 30, 100, || {
        decode_audio_file(&fixture("pcm16_44k_stereo.wav"), &audio).unwrap();
    });
    check("audio: wav 44.1k stereo resampled, from bytes", 30, 100, || {
        decode_audio_bytes(&wav, AudioContainer::Auto, &swr).unwrap();
    });
    check("audio: m4a 44.1k stereo", 30, 100, || {
        decode_audio_bytes(&m4a, AudioContainer::Auto, &audio).unwrap();
    });
    check("audio: mp3 16k mono (tag trim)", 30, 100, || {
        decode_audio_bytes(&mp3, AudioContainer::Auto, &audio).unwrap();
    });
    check("audio: a cut wav (refused)", 30, 100, || {
        assert!(decode_audio_bytes(&cut_wav, AudioContainer::Auto, &audio).is_err());
    });
    let tight = AudioOptions {
        max_seconds: 0.1,
        ..AudioOptions::default()
    };
    check("audio: over the length limit (refused)", 30, 100, || {
        assert!(decode_audio_file(&fixture("pcm16_44k_stereo.wav"), &tight).is_err());
    });
}
