//! `tessl::media::yuv`: YCbCr 4:2:0 to RGB, byte for byte what libswscale computes
//! (the converter torchcodec and the `ffmpeg` CLI use), with no codec and no device.
//!
//! The goldens are in `tests/fixtures/media_video/` (made by
//! `scripts/gen_media_video_exact_fixtures.py` with the `ffmpeg` CLI): the planes
//! `ffmpeg` decodes from every video fixture and its RGB frames, and random planes
//! converted with and without the CPU's assembly (`-cpuflags 0`). The exhaustive
//! test (`--ignored`) checks every `(Y, Cb, Cr)` triple against a live `ffmpeg`.

mod video_exact_support;

use tessl::media::yuv::{nv12_to_rgb24, Flavor, Matrix, Nv12, Range, YuvError};
use video_exact_support::*;

fn convert(w: usize, h: usize, planes: &[u8], m: Matrix, r: Range, flavor: Flavor) -> Vec<u8> {
    let (y, uv) = planes.split_at(w * h);
    nv12_to_rgb24(
        &Nv12 {
            width: w,
            height: h,
            y,
            y_stride: w,
            uv,
            uv_stride: w,
        },
        m,
        r,
        flavor,
    )
    .expect("a well-formed picture")
}

#[test]
fn the_planes_ffmpeg_decodes_convert_to_the_rgb_it_converts() {
    let mut checked = 0;
    for f in facts() {
        // The goldens of a rotated stream are rotated; rotation is checked on the codec side.
        if f.name == "e2e_video" || f.rotation != 0 || f.pix_fmt.contains("10") {
            continue;
        }
        let planes = bytes(&format!("{}.nv12", f.name));
        let want = bytes(&format!("{}.rgb24", f.name));
        let per_planes = f.width * f.height * 3 / 2;
        let per_rgb = f.width * f.height * 3;
        assert_eq!(planes.len(), per_planes * f.frames, "{}: nv12 golden size", f.name);
        for k in 0..f.frames {
            let got = convert(
                f.width,
                f.height,
                &planes[k * per_planes..(k + 1) * per_planes],
                f.matrix(),
                f.range(),
                Flavor::Aarch64,
            );
            assert_eq!(
                first_diff(&got, &want[k * per_rgb..(k + 1) * per_rgb]),
                None,
                "{} frame {k}: (first differing byte, differing bytes, largest difference)",
                f.name
            );
        }
        checked += 1;
    }
    assert!(checked >= 8, "only {checked} fixtures were checked");
}

#[test]
fn random_planes_convert_as_ffmpeg_does_on_both_converters() {
    let cases = synthetics();
    assert!(cases.len() >= 8);
    let mut neon_widths_differ = false;
    for c in &cases {
        let planes = bytes(&format!("{}.nv12", c.name));
        for (flavor, golden) in [
            (Flavor::Aarch64, format!("{}.rgb24", c.name)),
            (Flavor::Portable, format!("{}.portable.rgb24", c.name)),
        ] {
            let got = convert(c.width, c.height, &planes, c.matrix, c.range, flavor);
            assert_eq!(
                first_diff(&got, &bytes(&golden)),
                None,
                "{} as {flavor:?}: (first differing byte, differing bytes, largest difference)",
                c.name
            );
        }
        // The two converters are different arithmetic: they must differ exactly where the
        // aarch64 build uses its NEON one (a width that is a multiple of 16), so a test of
        // one flavor against the other's golden would fail rather than pass by luck.
        let a = convert(c.width, c.height, &planes, c.matrix, c.range, Flavor::Aarch64);
        let p = convert(c.width, c.height, &planes, c.matrix, c.range, Flavor::Portable);
        assert_eq!(
            a == p,
            c.width % 16 != 0,
            "{}: flavors equal only off the NEON widths",
            c.name
        );
        neon_widths_differ |= c.width % 16 == 0 && a != p;
    }
    assert!(neon_widths_differ, "no case exercised the NEON converter");
}

#[test]
fn a_stride_wider_than_the_row_is_skipped_not_read() {
    let c = &synthetics()[0];
    let planes = bytes(&format!("{}.nv12", c.name));
    let (y, uv) = planes.split_at(c.width * c.height);
    // Pad every row of both planes by 5 bytes of garbage.
    let pad = |plane: &[u8], rows: usize| -> Vec<u8> {
        let mut out = Vec::new();
        for r in 0..rows {
            out.extend_from_slice(&plane[r * c.width..(r + 1) * c.width]);
            out.extend_from_slice(&[0xA5; 5]);
        }
        out
    };
    let (y2, uv2) = (pad(y, c.height), pad(uv, c.height / 2));
    let got = nv12_to_rgb24(
        &Nv12 {
            width: c.width,
            height: c.height,
            y: &y2,
            y_stride: c.width + 5,
            uv: &uv2,
            uv_stride: c.width + 5,
        },
        c.matrix,
        c.range,
        Flavor::Aarch64,
    )
    .unwrap();
    assert_eq!(
        got,
        convert(c.width, c.height, &planes, c.matrix, c.range, Flavor::Aarch64)
    );
}

#[test]
fn what_cannot_be_converted_as_libswscale_does_is_refused() {
    let ok = |w, h, y: &[u8], uv: &[u8], ys, uvs| {
        nv12_to_rgb24(
            &Nv12 {
                width: w,
                height: h,
                y,
                y_stride: ys,
                uv,
                uv_stride: uvs,
            },
            Matrix::Bt709,
            Range::Limited,
            Flavor::Aarch64,
        )
    };
    let y = [0u8; 64];
    let uv = [0u8; 32];
    assert_eq!(ok(4, 4, &y, &uv, 4, 4).map(|v| v.len()), Ok(48));
    for (w, h) in [(0, 4), (4, 0), (3, 4), (4, 3), (3, 3)] {
        assert_eq!(
            ok(w, h, &y, &uv, 4, 4),
            Err(YuvError::Size { width: w, height: h }),
            "{w}x{h}"
        );
    }
    // A plane one byte short, a stride narrower than the row, a chroma plane too small.
    assert!(matches!(
        ok(4, 4, &y[..15], &uv, 4, 4),
        Err(YuvError::Plane { plane: "luma", .. })
    ));
    assert!(matches!(
        ok(4, 4, &y, &uv, 3, 4),
        Err(YuvError::Plane { plane: "luma", .. })
    ));
    assert!(matches!(
        ok(4, 4, &y, &uv[..7], 4, 4),
        Err(YuvError::Plane { plane: "chroma", .. })
    ));
    // A stride so large that rows * stride overflows is an error, not a wrapped slice.
    assert!(matches!(
        ok(4, 4, &y, &uv, usize::MAX / 2, 4),
        Err(YuvError::Plane { plane: "luma", .. })
    ));
}

#[test]
fn matrices_are_chosen_by_the_h273_code_as_libswscale_does() {
    assert_eq!(Matrix::from_code(1), Some(Matrix::Bt709));
    // 2 is "unspecified", which libswscale converts with BT.601; 5 and 6 are BT.601's two spellings.
    for code in [2, 5, 6] {
        assert_eq!(Matrix::from_code(code), Some(Matrix::Bt601), "code {code}");
    }
    assert_eq!(Matrix::from_code(9), Some(Matrix::Bt2020Nc));
    // FCC, SMPTE 240M, YCgCo, BT.2020 constant luminance, identity: not held.
    for code in [0, 3, 4, 7, 8, 10, 11, 255] {
        assert_eq!(Matrix::from_code(code), None, "code {code}");
    }
}

/// Every `(Y, Cb, Cr)` triple of 8 bits, converted by a live `ffmpeg` and by this
/// module, for every matrix and range held, on both converters. It is the proof
/// that the constants in `src/media/yuv.rs` are libswscale's, and takes about a
/// minute per matrix and range. Needs `ffmpeg` (9.0.2 on aarch64 was used).
#[test]
#[ignore]
fn exhaustive_against_a_live_ffmpeg() {
    use std::io::Write;
    use std::process::Command;
    let work = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/yuv_exhaustive");
    std::fs::create_dir_all(&work).unwrap();
    let input = work.join("cube.yuv");
    // 64 frames of 512x512: frame b holds luma 4b..4b+3 in the four samples of each 2x2 block,
    // and Cb = column, Cr = row of the chroma plane, so the 64 frames hold all 2^24 triples.
    // The CLI is given planar yuv420p (what libavcodec produces); this module takes the same
    // samples interleaved.
    let (mut planar, mut nv12) = (Vec::new(), Vec::new());
    for b in 0..64usize {
        let mut y = vec![0u8; 512 * 512];
        for (i, v) in y.iter_mut().enumerate() {
            let (row, col) = (i / 512, i % 512);
            *v = (b * 4 + (row % 2) * 2 + col % 2) as u8;
        }
        let cb: Vec<u8> = (0..256 * 256).map(|i| (i % 256) as u8).collect();
        let cr: Vec<u8> = (0..256 * 256).map(|i| (i / 256) as u8).collect();
        planar.extend_from_slice(&y);
        planar.extend_from_slice(&cb);
        planar.extend_from_slice(&cr);
        nv12.extend_from_slice(&y);
        nv12.extend(cb.iter().zip(&cr).flat_map(|(&u, &v)| [u, v]));
    }
    std::fs::File::create(&input).unwrap().write_all(&planar).unwrap();
    let per_nv12 = 512 * 512 * 3 / 2;
    let mut failures = Vec::new();
    for (matrix, tag) in [
        (Matrix::Bt709, "bt709"),
        (Matrix::Bt601, "smpte170m"),
        (Matrix::Bt601, "bt470bg"),
        (Matrix::Bt2020Nc, "bt2020nc"),
    ] {
        for (range, range_tag) in [(Range::Limited, "tv"), (Range::Full, "pc")] {
            for (flavor, cpu) in [(Flavor::Aarch64, false), (Flavor::Portable, true)] {
                let mut cmd = Command::new("ffmpeg");
                cmd.args(["-v", "error", "-y"]);
                if cpu {
                    cmd.args(["-cpuflags", "0"]);
                }
                cmd.args([
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "yuv420p",
                    "-s",
                    "512x512",
                    "-colorspace",
                    tag,
                ]);
                cmd.args(["-color_range", range_tag, "-i"]).arg(&input);
                cmd.args(["-pix_fmt", "rgb24", "-f", "rawvideo", "-"]);
                let out = cmd.output().expect("ffmpeg runs").stdout;
                assert_eq!(out.len(), 64 * 512 * 512 * 3, "{tag} {range_tag}: ffmpeg output size");
                let mut bad = 0usize;
                for b in 0..64 {
                    let mine = convert(512, 512, &nv12[b * per_nv12..(b + 1) * per_nv12], matrix, range, flavor);
                    let want = &out[b * 512 * 512 * 3..(b + 1) * 512 * 512 * 3];
                    bad += mine.iter().zip(want).filter(|(a, b)| a != b).count();
                }
                eprintln!(
                    "{tag} {range_tag} {flavor:?}: {bad} of {} bytes differ",
                    64 * 512 * 512 * 3
                );
                if bad != 0 {
                    failures.push(format!("{tag} {range_tag} {flavor:?}: {bad} bytes"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}
