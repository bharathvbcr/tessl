//! The leak loops of `tests/media_leaks.rs` for the decoders that need
//! VideoToolbox: HEIC images and H.264 / HEVC video (AVAssetReader, sample
//! buffers, pixel buffers, autorelease pools). One test, local only, for the
//! reason given in `tests/media_codecs.rs`.

mod media_support;

use media_support::*;
use tessl::media::image::{decode_image, ImageOptions};
use tessl::media::video::{VideoOptions, VideoSource};

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
fn heic_and_video_decode_loops_do_not_grow_the_process() {
    let heic = bytes("photo.heic");
    check("image: heic from bytes", 20, 100, || {
        decode_image(&heic, &ImageOptions::default()).unwrap();
    });
    let all: Vec<usize> = (0..12).collect();
    for name in ["colors.mp4", "colors_hevc.mp4", "quad_rot90.mp4"] {
        check(&format!("video: open + decode 12 frames of {name}"), 4, 12, || {
            let v = VideoSource::open(&fixture(name), &VideoOptions::default()).unwrap();
            assert_eq!(v.decode_frames(&all).unwrap().frames.len(), 12);
        });
    }
    let data = bytes("colors.mp4");
    check("video: from bytes (temporary file)", 4, 12, || {
        let v = VideoSource::open_bytes(&data, "mp4", &VideoOptions::default()).unwrap();
        assert_eq!(v.decode_frames(&[3]).unwrap().frames.len(), 1);
    });
    let cut = data[..data.len() - 30].to_vec();
    check("video: a cut file (refused)", 4, 12, || {
        assert!(VideoSource::open_bytes(&cut, "mp4", &VideoOptions::default()).is_err());
    });
    assert_eq!(temp_video_files(), 0, "temporary video files left behind");
}
