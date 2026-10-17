//! `tessl::media::yuv` for every sample format it converts: 8 and 10 bits, 4:2:0,
//! 4:2:2 and 4:4:4, byte for byte what libswscale computes (the `ffmpeg` CLI's
//! conversion), with no codec and no device.
//!
//! The goldens are in `tests/fixtures/media_video/` (made by
//! `scripts/gen_media_video_exact_fixtures.py`): random planes of every format and
//! the CLI's `rgb24` for them. The exhaustive tests (`--ignored`) run a live `ffmpeg`
//! over every sample combination a format can hold (or, at ten bits with subsampled
//! chroma, over what its arithmetic can tell apart) for every matrix and range:
//!
//! ```text
//! cargo test --release --test media_yuv_formats -- --ignored --nocapture
//! ```

mod video_exact_support;

use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::process::{Command, Stdio};

use tessl::media::yuv::{
    check_shape, to_rgb24, Chroma, ChromaFilter, Flavor, Layout, Matrix, Params, Picture, Range, Threads, YuvError,
};
use video_exact_support::*;

fn params(c: &FormatCase) -> Params {
    Params::new(c.matrix, c.range)
}

fn convert(c: &FormatCase, p: &Params) -> Vec<u8> {
    let planes = c.planes();
    let uv_stride = match c.layout.chroma {
        Chroma::C444 => 2 * c.width,
        _ => c.width,
    };
    to_rgb24(
        &Picture {
            width: c.width,
            height: c.height,
            layout: c.layout,
            y: &planes.y,
            y_stride: c.width,
            uv: &planes.uv,
            uv_stride,
        },
        p,
    )
    .unwrap_or_else(|e| panic!("{}: {e}", c.name))
}

#[test]
fn every_format_converts_as_the_cli_does() {
    let cases = format_cases();
    assert!(cases.len() >= 30, "{} cases", cases.len());
    let mut classes = BTreeSet::new();
    for c in &cases {
        let mut checks: Vec<(String, Params, String)> = Vec::new();
        match (c.layout.chroma, c.layout.depth) {
            (Chroma::C420 | Chroma::C422, 8) => {
                for (flavor, suffix) in [(Flavor::Aarch64, ".rgb24"), (Flavor::Portable, ".portable.rgb24")] {
                    checks.push((format!("{flavor:?}"), Params { flavor, ..params(c) }, suffix.into()));
                }
            }
            (Chroma::C420, _) => {
                for (filter, suffix) in [
                    (ChromaFilter::Bicubic, ".rgb24"),
                    (ChromaFilter::Bilinear, ".bilinear.rgb24"),
                ] {
                    checks.push((format!("{filter:?}"), Params { filter, ..params(c) }, suffix.into()));
                }
            }
            _ => checks.push(("-".into(), params(c), ".rgb24".into())),
        }
        for (label, p, suffix) in checks {
            let want = bytes(&format!("{}{suffix}", c.name));
            assert_eq!(want.len(), c.width * c.height * 3, "{}: golden size", c.name);
            assert_eq!(
                first_diff(&convert(c, &p), &want),
                None,
                "{} ({label}): (first differing byte, differing bytes, largest difference)",
                c.name
            );
            classes.insert(format!(
                "{:?}/{}/{:?}/{:?}/{}",
                c.layout.chroma,
                c.layout.depth,
                c.matrix,
                c.range,
                if Flavor::Aarch64.uses_neon(c.width, c.height) {
                    "neon"
                } else {
                    "table"
                }
            ));
        }
    }
    // Every layout, both depths, all five matrices, both ranges, both converter widths.
    assert!(classes.len() >= 34, "classes covered: {}", classes.len());
}

/// What the 8-bit 4:4:4 and the 10-bit 4:4:4 and 4:2:2 paths do not depend on: the
/// converter flavor, and the 10-bit 4:2:2 and 4:4:4 ones not on the chroma filter either.
#[test]
fn paths_that_do_not_depend_on_the_flavor_or_filter_do_not() {
    for c in format_cases() {
        let base = convert(&c, &params(&c));
        let depends_on_flavor = c.layout.depth == 8 && c.layout.chroma != Chroma::C444;
        let depends_on_filter = c.layout.depth == 10 && c.layout.chroma == Chroma::C420 && c.height > 2;
        let portable = convert(
            &c,
            &Params {
                flavor: Flavor::Portable,
                ..params(&c)
            },
        );
        // The two converters differ exactly where the aarch64 build would use its NEON one.
        assert_eq!(
            portable == base,
            !(depends_on_flavor && Flavor::Aarch64.uses_neon(c.width, c.height)),
            "{}: flavor",
            c.name
        );
        let bicubic = convert(
            &c,
            &Params {
                filter: ChromaFilter::Bicubic,
                ..params(&c)
            },
        );
        if !depends_on_filter {
            assert_eq!(bicubic, base, "{}: filter", c.name);
        }
    }
}

#[test]
fn threading_never_changes_a_byte() {
    // The committed cases, and large pictures of every layout (random planes, no golden: the
    // single-threaded converter is the reference the others must equal).
    let mut checked = 0;
    for c in format_cases() {
        let single = convert(
            &c,
            &Params {
                threads: Threads::Single,
                ..params(&c)
            },
        );
        for threads in [
            Threads::Auto,
            Threads::Count(1),
            Threads::Count(2),
            Threads::Count(3),
            Threads::Count(7),
            Threads::Count(c.height),
            Threads::Count(c.height + 5),
            Threads::Count(usize::MAX),
        ] {
            assert_eq!(
                convert(&c, &Params { threads, ..params(&c) }),
                single,
                "{} with {threads:?}",
                c.name
            );
            checked += 1;
        }
    }
    assert!(checked >= 200);
    for layout in [
        Layout {
            chroma: Chroma::C420,
            depth: 8,
        },
        Layout {
            chroma: Chroma::C422,
            depth: 8,
        },
        Layout {
            chroma: Chroma::C444,
            depth: 8,
        },
        Layout {
            chroma: Chroma::C420,
            depth: 10,
        },
        Layout {
            chroma: Chroma::C422,
            depth: 10,
        },
        Layout {
            chroma: Chroma::C444,
            depth: 10,
        },
    ] {
        let (w, h) = (1280, 720);
        let (cw, ch) = match layout.chroma {
            Chroma::C420 => (w / 2, h / 2),
            Chroma::C422 => (w / 2, h),
            Chroma::C444 => (w, h),
        };
        let max = if layout.depth == 8 { 256u32 } else { 1024 };
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 33) as u32 % max
        };
        let y: Vec<u16> = (0..w * h).map(|_| next() as u16).collect();
        let uv: Vec<u16> = (0..2 * cw * ch).map(|_| next() as u16).collect();
        let uv_stride = 2 * cw;
        for filter in [ChromaFilter::Bilinear, ChromaFilter::Bicubic] {
            let run = |threads| {
                to_rgb24(
                    &Picture {
                        width: w,
                        height: h,
                        layout,
                        y: &y,
                        y_stride: w,
                        uv: &uv,
                        uv_stride,
                    },
                    &Params {
                        filter,
                        threads,
                        ..Params::new(Matrix::Bt709, Range::Limited)
                    },
                )
                .unwrap()
            };
            let single = run(Threads::Single);
            for threads in [Threads::Auto, Threads::Count(2), Threads::Count(5), Threads::Count(16)] {
                assert!(run(threads) == single, "{layout:?} {filter:?} with {threads:?}");
            }
        }
    }
}

#[test]
fn shapes_libswscale_does_not_define_here_are_refused_by_name() {
    let l = |chroma, depth| Layout { chroma, depth };
    let ok = |layout, w, h, f| check_shape(layout, w, h, f);
    // Fine: 4:4:4 of any size, 4:2:2 of an odd height, tiny 4:2:0.
    for (layout, w, h) in [
        (l(Chroma::C444, 8), 1, 1),
        (l(Chroma::C444, 10), 33, 17),
        (l(Chroma::C422, 8), 64, 33),
        (l(Chroma::C422, 10), 2, 1),
        (l(Chroma::C420, 8), 2, 2),
        (l(Chroma::C420, 10), 6, 2),
        (l(Chroma::C420, 10), 6, 14),
    ] {
        assert_eq!(ok(layout, w, h, ChromaFilter::Bicubic), Ok(()), "{layout:?} {w}x{h}");
    }
    // Zero sides; an odd width where the chroma is halved; an odd height for 4:2:0.
    for (layout, w, h) in [
        (l(Chroma::C444, 8), 0, 4),
        (l(Chroma::C444, 8), 4, 0),
        (l(Chroma::C422, 8), 3, 4),
        (l(Chroma::C420, 8), 3, 4),
        (l(Chroma::C420, 8), 4, 3),
        (l(Chroma::C420, 10), 4, 5),
        (l(Chroma::C422, 10), 5, 4),
    ] {
        assert_eq!(
            ok(layout, w, h, ChromaFilter::Bilinear),
            Err(YuvError::Size { width: w, height: h }),
            "{layout:?} {w}x{h}"
        );
    }
    // 10-bit 4:2:0 below the heights the chroma filter was measured for.
    let ten = l(Chroma::C420, 10);
    for h in [4, 6, 8, 10, 12] {
        assert_eq!(
            ok(ten, 16, h, ChromaFilter::Bicubic),
            Err(YuvError::TooShort {
                height: h,
                filter: ChromaFilter::Bicubic,
                min: 14
            }),
            "bicubic {h}"
        );
    }
    for h in [4, 6] {
        assert!(
            matches!(
                ok(ten, 16, h, ChromaFilter::Bilinear),
                Err(YuvError::TooShort { min: 8, .. })
            ),
            "bilinear {h}"
        );
    }
    assert_eq!(ok(ten, 16, 8, ChromaFilter::Bilinear), Ok(()));
    // Depths other than 8 and 10.
    assert_eq!(
        ok(l(Chroma::C444, 12), 4, 4, ChromaFilter::Bilinear),
        Err(YuvError::Depth { depth: 12 })
    );
    assert_eq!(
        ok(l(Chroma::C444, 9), 4, 4, ChromaFilter::Bilinear),
        Err(YuvError::Depth { depth: 9 })
    );
    // The messages name the reason.
    let text = ok(ten, 16, 6, ChromaFilter::Bicubic).unwrap_err().to_string();
    assert!(text.contains("14 rows") && text.contains("Bicubic"), "{text}");
}

#[test]
fn planes_and_samples_are_checked() {
    let y = vec![0u16; 16];
    let uv = vec![0u16; 16];
    let pic = |layout: Layout, y: &[u16], uv: &[u16], ys, uvs| {
        to_rgb24(
            &Picture {
                width: 4,
                height: 4,
                layout,
                y,
                y_stride: ys,
                uv,
                uv_stride: uvs,
            },
            &Params::new(Matrix::Bt709, Range::Limited),
        )
    };
    let c444 = Layout {
        chroma: Chroma::C444,
        depth: 10,
    };
    assert_eq!(
        pic(c444, &y, &uv, 4, 8).map(|v| v.len()),
        Err(YuvError::Plane {
            plane: "chroma",
            have: 16,
            need: 32
        })
    );
    assert!(matches!(
        pic(c444, &y[..15], &uv, 4, 8),
        Err(YuvError::Plane { plane: "luma", .. })
    ));
    assert!(matches!(
        pic(c444, &y, &uv, 3, 8),
        Err(YuvError::Plane { plane: "luma", .. })
    ));
    assert!(matches!(
        pic(c444, &y, &uv, usize::MAX / 2, 8),
        Err(YuvError::Plane { .. })
    ));
    // A stride wider than the row is skipped, a sample wider than the depth is an error.
    let wide = vec![1023u16; 4 * 7];
    assert!(pic(c444, &wide, &[512u16; 4 * 16], 7, 16).is_ok());
    let mut bad = vec![0u16; 16];
    bad[9] = 1024;
    assert_eq!(
        pic(c444, &bad, &[0u16; 32], 4, 8),
        Err(YuvError::Sample {
            plane: "luma",
            value: 1024,
            depth: 10
        })
    );
    let mut bad_uv = vec![0u16; 32];
    bad_uv[31] = 4000;
    assert_eq!(
        pic(c444, &y, &bad_uv, 4, 8),
        Err(YuvError::Sample {
            plane: "chroma",
            value: 4000,
            depth: 10
        })
    );
    // Samples of u8 cannot exceed 8 bits; a u16 picture at depth 8 is checked.
    let c8 = Layout {
        chroma: Chroma::C444,
        depth: 8,
    };
    assert_eq!(
        pic(c8, &[256u16; 16], &uv.repeat(2), 4, 8),
        Err(YuvError::Sample {
            plane: "luma",
            value: 256,
            depth: 8
        })
    );
}

// ---------------------------------------------------------------------------
// Live `ffmpeg`
// ---------------------------------------------------------------------------

const MATRICES: [(Matrix, &str); 6] = [
    (Matrix::Bt709, "bt709"),
    (Matrix::Bt601, "smpte170m"),
    (Matrix::Bt601, "bt470bg"),
    (Matrix::Bt2020Nc, "bt2020nc"),
    (Matrix::Fcc, "fcc"),
    (Matrix::Smpte240m, "smpte240m"),
];

/// Run `ffmpeg` on raw planes fed from `frames` (called with each frame index, returns its
/// planar bytes) and call `check(index, rgb)` with each output frame, streaming both ways.
fn through_ffmpeg(
    pix_fmt: &str,
    (w, h): (usize, usize),
    (tag, range): (&str, &str),
    cpu_flags_off: bool,
    nframes: usize,
    frames: impl Fn(usize) -> Vec<u8> + Send + Sync,
    mut check: impl FnMut(usize, &[u8]),
) {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-v", "error", "-y"]);
    if cpu_flags_off {
        cmd.args(["-cpuflags", "0"]);
    }
    cmd.args([
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-s",
        &format!("{w}x{h}"),
        "-colorspace",
        tag,
    ]);
    cmd.args([
        "-color_range",
        range,
        "-i",
        "-",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]);
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("ffmpeg runs");
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let frames = &frames;
    std::thread::scope(|scope| {
        let writer = scope.spawn(move || {
            for i in 0..nframes {
                stdin.write_all(&frames(i)).expect("ffmpeg takes the planes");
            }
            drop(stdin);
        });
        let mut rgb = vec![0u8; w * h * 3];
        for i in 0..nframes {
            stdout.read_exact(&mut rgb).expect("ffmpeg writes a frame");
            check(i, &rgb);
        }
        writer.join().unwrap();
    });
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "ffmpeg: {}", String::from_utf8_lossy(&out.stderr));
}

fn range_tags() -> [(Range, &'static str); 2] {
    [(Range::Limited, "tv"), (Range::Full, "pc")]
}

/// 8-bit 4:2:2 and 4:4:4: every `(Y, Cb, Cr)`, for every matrix and range (and, for 4:2:2, both
/// converters). Frame `b` of a 512 x 512 picture holds luma `4b..4b+3` over each 2 x 2 block
/// (4:4:4 and 4:2:2 take the pairs they need), `Cb` = chroma column and `Cr` = chroma row.
#[test]
#[ignore]
fn exhaustive_8_bit_422_and_444_against_a_live_ffmpeg() {
    let mut failures = Vec::new();
    for (chroma, pix_fmt) in [(Chroma::C422, "yuv422p"), (Chroma::C444, "yuv444p")] {
        // 4:4:4: one 4096 x 4096 frame holds every triple, pixel i = (Y << 16) | (Cb << 8) | Cr.
        // 4:2:2: 256 frames of 512 x 256, the pair of luma values (2k, 2k+1) in frame k % 128 ... below.
        let (size, nframes) = match chroma {
            Chroma::C444 => ((4096, 4096), 1),
            _ => ((512, 256), 128),
        };
        let (w, h) = size;
        let planar = |i: usize| -> Vec<u8> {
            let mut out = Vec::new();
            match chroma {
                Chroma::C444 => {
                    out.extend((0..w * h).map(|p| (p >> 16) as u8));
                    out.extend((0..w * h).map(|p| (p >> 8) as u8));
                    out.extend((0..w * h).map(|p| p as u8));
                }
                _ => {
                    // Frame i: luma 2i (left of each pair) and 2i+1 (right); Cb = pair column, Cr = row.
                    out.extend((0..w * h).map(|p| (2 * i + p % 2) as u8));
                    out.extend((0..(w / 2) * h).map(|p| (p % (w / 2)) as u8));
                    out.extend((0..(w / 2) * h).map(|p| (p / (w / 2)) as u8));
                }
            }
            out
        };
        for (matrix, tag) in MATRICES {
            for (range, range_tag) in range_tags() {
                for (flavor, cpu_off) in [(Flavor::Aarch64, false), (Flavor::Portable, true)] {
                    if chroma == Chroma::C444 && cpu_off {
                        continue;
                    }
                    let mut bad = 0usize;
                    let p = Params {
                        flavor,
                        ..Params::new(matrix, range)
                    };
                    through_ffmpeg(pix_fmt, size, (tag, range_tag), cpu_off, nframes, planar, |i, rgb| {
                        // Mine, from the same samples as bi-planar.
                        let raw = planar(i);
                        let (y, rest) = raw.split_at(w * h);
                        let (cb, cr) = rest.split_at(rest.len() / 2);
                        let y16: Vec<u16> = y.iter().map(|&v| u16::from(v)).collect();
                        let uv: Vec<u16> = cb
                            .iter()
                            .zip(cr)
                            .flat_map(|(&b, &r)| [u16::from(b), u16::from(r)])
                            .collect();
                        let mine = to_rgb24(
                            &Picture {
                                width: w,
                                height: h,
                                layout: Layout { chroma, depth: 8 },
                                y: &y16,
                                y_stride: w,
                                uv: &uv,
                                uv_stride: if chroma == Chroma::C444 { 2 * w } else { w },
                            },
                            &p,
                        )
                        .unwrap();
                        bad += mine.iter().zip(rgb).filter(|(a, b)| a != b).count();
                    });
                    eprintln!(
                        "{pix_fmt} {tag} {range_tag} {flavor:?}: {bad} of {} bytes differ",
                        nframes * w * h * 3
                    );
                    if bad != 0 {
                        failures.push(format!("{pix_fmt} {tag} {range_tag} {flavor:?}: {bad} bytes"));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// 10-bit 4:4:4: every `(Y, Cb, Cr)` of 10 bits (2^30 triples) for every matrix and range, as
/// 1024 frames of 1024 x 1024: frame `y` has luma `y`, `Cb` = column, `Cr` = row.
#[test]
#[ignore]
fn exhaustive_10_bit_444_against_a_live_ffmpeg() {
    let (w, h) = (1024usize, 1024usize);
    let mut failures = Vec::new();
    for (matrix, tag) in MATRICES {
        for (range, range_tag) in range_tags() {
            let frame = |y: usize| -> Vec<u8> {
                let mut out = Vec::with_capacity(w * h * 6);
                out.extend((0..w * h).flat_map(|_| (y as u16).to_le_bytes()));
                out.extend((0..w * h).flat_map(|p| ((p % w) as u16).to_le_bytes()));
                out.extend((0..w * h).flat_map(|p| ((p / w) as u16).to_le_bytes()));
                out
            };
            // Frame-independent parts of the picture, built once per check.
            let cb: Vec<u16> = (0..w * h).map(|p| (p % w) as u16).collect();
            let cr: Vec<u16> = (0..w * h).map(|p| (p / w) as u16).collect();
            let uv: Vec<u16> = cb.iter().zip(&cr).flat_map(|(&b, &r)| [b, r]).collect();
            let p = Params::new(matrix, range);
            let mut bad = 0usize;
            through_ffmpeg("yuv444p10le", (w, h), (tag, range_tag), false, 1024, frame, |y, rgb| {
                let luma = vec![y as u16; w * h];
                let mine = to_rgb24(
                    &Picture {
                        width: w,
                        height: h,
                        layout: Layout {
                            chroma: Chroma::C444,
                            depth: 10,
                        },
                        y: &luma,
                        y_stride: w,
                        uv: &uv,
                        uv_stride: 2 * w,
                    },
                    &p,
                )
                .unwrap();
                bad += mine.iter().zip(rgb).filter(|(a, b)| a != b).count();
            });
            eprintln!(
                "yuv444p10le {tag} {range_tag}: {bad} of {} bytes differ",
                1024 * w * h * 3
            );
            if bad != 0 {
                failures.push(format!("{tag} {range_tag}: {bad} bytes"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// 10-bit 4:2:2: the arithmetic rounds each sample to 8 bits (luma may reach 256) and then runs the
/// table converter, so (1) the rounding of all 1024 values of luma, of Cb and of Cr, and (2) the
/// converter on every rounded triple (luma 0..=256, chroma 0..=255) are what tell the paths apart.
#[test]
#[ignore]
fn exhaustive_10_bit_422_against_a_live_ffmpeg() {
    let (w, h) = (512usize, 256usize);
    let mut failures = Vec::new();
    let mine = |matrix, range, y: &[u16], uv: &[u16], width, height| {
        to_rgb24(
            &Picture {
                width,
                height,
                layout: Layout {
                    chroma: Chroma::C422,
                    depth: 10,
                },
                y,
                y_stride: width,
                uv,
                uv_stride: width,
            },
            &Params::new(matrix, range),
        )
        .unwrap()
    };
    for (matrix, tag) in MATRICES {
        for (range, range_tag) in range_tags() {
            // (2): frame k has luma pair (2k, 2k+1) on the rounded scale (257 is luma 256, twice:
            // 1022 rounds to 256), Cb = pair column, Cr = row, each at 4 * value.
            let level = |v: usize| -> u16 {
                if v >= 256 {
                    1022
                } else {
                    (4 * v) as u16
                }
            };
            let frame = |k: usize| -> Vec<u8> {
                let mut out = Vec::new();
                out.extend((0..w * h).flat_map(|p| level(2 * k + p % 2).to_le_bytes()));
                out.extend((0..(w / 2) * h).flat_map(|p| (4 * (p % (w / 2)) as u16).to_le_bytes()));
                out.extend((0..(w / 2) * h).flat_map(|p| (4 * (p / (w / 2)) as u16).to_le_bytes()));
                out
            };
            let cb_row: Vec<u16> = (0..(w / 2) * h).map(|p| (4 * (p % (w / 2))) as u16).collect();
            let cr_row: Vec<u16> = (0..(w / 2) * h).map(|p| (4 * (p / (w / 2))) as u16).collect();
            let uv: Vec<u16> = cb_row.iter().zip(&cr_row).flat_map(|(&b, &r)| [b, r]).collect();
            let mut bad = 0usize;
            through_ffmpeg("yuv422p10le", (w, h), (tag, range_tag), false, 129, frame, |k, rgb| {
                let y: Vec<u16> = (0..w * h).map(|p| level(2 * k + p % 2)).collect();
                let got = mine(matrix, range, &y, &uv, w, h);
                bad += got.iter().zip(rgb).filter(|(a, b)| a != b).count();
            });
            // (1): the rounding of every value, one sample varying at a time over 1024 columns.
            // A 1024 x 2 picture; luma sweeps 0..1023 along both rows. A chroma component sweeps
            // 0..1023 over the 512 pair columns twice: row 0 the even values, row 1 the odd.
            let (rw, rh) = (1024usize, 2usize);
            for which in 0..3 {
                let mut y = vec![512u16; rw * rh];
                let (mut cb, mut cr) = (vec![512u16; rw / 2 * rh], vec![512u16; rw / 2 * rh]);
                match which {
                    0 => (0..rw * rh).for_each(|p| y[p] = (p % rw) as u16),
                    _ => {
                        let target = if which == 1 { &mut cb } else { &mut cr };
                        for c in 0..rw / 2 {
                            target[c] = (2 * c) as u16;
                            target[rw / 2 + c] = (2 * c + 1) as u16;
                        }
                    }
                }
                let planar: Vec<u8> = y.iter().chain(&cb).chain(&cr).flat_map(|v| v.to_le_bytes()).collect();
                let uv: Vec<u16> = cb.iter().zip(&cr).flat_map(|(&b, &r)| [b, r]).collect();
                let mut sweep_bad = 0usize;
                through_ffmpeg(
                    "yuv422p10le",
                    (rw, rh),
                    (tag, range_tag),
                    false,
                    1,
                    |_| planar.clone(),
                    |_, rgb| {
                        let got = mine(matrix, range, &y, &uv, rw, rh);
                        sweep_bad += got.iter().zip(rgb).filter(|(a, b)| a != b).count();
                    },
                );
                bad += sweep_bad;
            }
            eprintln!("yuv422p10le {tag} {range_tag}: {bad} bytes differ");
            if bad != 0 {
                failures.push(format!("{tag} {range_tag}: {bad} bytes"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// 10-bit 4:2:0 and 4:2:2 against random pictures of many shapes, both chroma filters: the
/// vertical filter at every even height from 14 to 120, at 480p, 720p and 1080p, with both
/// converter widths.
#[test]
#[ignore]
fn ten_bit_420_filters_against_a_live_ffmpeg_at_many_heights() {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move |max: u32| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 33) as u32 % max
    };
    let mut shapes: Vec<(usize, usize)> = (14..=120).step_by(2).map(|h| (34usize, h)).collect();
    shapes.extend([
        (32, 64),
        (640, 480),
        (1280, 720),
        (1920, 1080),
        (1918, 1080),
        (3840, 2160),
    ]);
    let mut failures = Vec::new();
    for (w, h) in shapes {
        let y: Vec<u16> = (0..w * h).map(|_| next(1024) as u16).collect();
        let (cw, ch) = (w / 2, h / 2);
        let cb: Vec<u16> = (0..cw * ch).map(|_| next(1024) as u16).collect();
        let cr: Vec<u16> = (0..cw * ch).map(|_| next(1024) as u16).collect();
        let uv: Vec<u16> = cb.iter().zip(&cr).flat_map(|(&b, &r)| [b, r]).collect();
        let planar: Vec<u8> = y.iter().chain(&cb).chain(&cr).flat_map(|v| v.to_le_bytes()).collect();
        let combos: Vec<_> = MATRICES
            .iter()
            .flat_map(|m| range_tags().into_iter().map(move |r| (m, r)))
            .collect();
        // Small heights rotate through the twelve (matrix, range) pairs; the big pictures take all.
        let take = if h < 200 { 1 } else { combos.len() };
        for (filter, flags) in [(ChromaFilter::Bilinear, "bilinear"), (ChromaFilter::Bicubic, "bicubic")] {
            for ((matrix, tag), (range, range_tag)) in (0..take).map(|k| combos[(k + h / 2) % combos.len()]) {
                let out = Command::new("ffmpeg")
                    .args([
                        "-v",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "yuv420p10le",
                        "-s",
                        &format!("{w}x{h}"),
                    ])
                    .args([
                        "-colorspace",
                        tag,
                        "-color_range",
                        range_tag,
                        "-i",
                        "-",
                        "-sws_flags",
                        flags,
                    ])
                    .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .spawn()
                    .and_then(|mut c| {
                        let mut si = c.stdin.take().unwrap();
                        let data = planar.clone();
                        let t = std::thread::spawn(move || si.write_all(&data));
                        let o = c.wait_with_output();
                        t.join().unwrap().ok();
                        o
                    })
                    .expect("ffmpeg runs");
                let got = to_rgb24(
                    &Picture {
                        width: w,
                        height: h,
                        layout: Layout {
                            chroma: Chroma::C420,
                            depth: 10,
                        },
                        y: &y,
                        y_stride: w,
                        uv: &uv,
                        uv_stride: w,
                    },
                    &Params {
                        filter,
                        ..Params::new(*matrix, range)
                    },
                )
                .unwrap();
                let bad = got.iter().zip(&out.stdout).filter(|(a, b)| a != b).count();
                if bad != 0 || out.stdout.len() != got.len() {
                    failures.push(format!("{w}x{h} {flags} {tag} {range_tag}: {bad} bytes"));
                }
            }
        }
        eprintln!("{w}x{h}: done ({} failures so far)", failures.len());
    }
    assert!(failures.is_empty(), "{failures:?}");
}
