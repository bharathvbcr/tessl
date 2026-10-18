//! Fixture access shared by `tests/media_yuv_exact.rs` (the converter, no codec) and
//! `tests/media_video_exact.rs` (AVFoundation decoding).
//!
//! The fixtures under `tests/fixtures/media_video/` are made by
//! `scripts/gen_media_video_exact_fixtures.py`; `facts.txt` there has one line per
//! fixture, `kind name key=value ...`.

#![allow(dead_code)] // each of the two test binaries uses a different part

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tessl::media::yuv::{Chroma, Layout, Matrix, Range};

pub fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/media_video")
}

pub fn fixture(name: &str) -> PathBuf {
    dir().join(name)
}

pub fn bytes(name: &str) -> Vec<u8> {
    let p = fixture(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e} (run scripts/gen_media_video_exact_fixtures.py)", p.display()))
}

/// One `video` line of `facts.txt`.
#[derive(Clone, Debug)]
pub struct Fact {
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub rotation: u32,
    /// `h`, `v` or `none`: a display flip (with no rotation).
    pub flip: String,
    /// Whether the fixture has frame-by-frame goldens (the end-to-end file and the
    /// streams the exact conversion refuses do not).
    pub golden: bool,
    pub frames: usize,
    pub pix_fmt: String,
    pub color_range: String,
    pub color_space: String,
    pub duration: f64,
    pub average_fps: f64,
    pub pts: Vec<f64>,
    /// The indices `EmbeddingGemma2VideoProcessor.sample_frames` picks, by setting name
    /// (`sampled`, `sampled_f5m4u`, ...): see SAMPLINGS in the generator.
    pub sampled: HashMap<String, Vec<usize>>,
}

impl Fact {
    /// The output size after the stream's rotation.
    pub fn display_size(&self) -> (usize, usize) {
        if self.rotation == 90 || self.rotation == 270 {
            (self.height, self.width)
        } else {
            (self.width, self.height)
        }
    }

    /// The matrix libswscale converts this stream with: the tag, or BT.601 for
    /// none.
    pub fn matrix(&self) -> Matrix {
        matrix_named(&self.color_space)
    }

    pub fn range(&self) -> Range {
        range_named(&self.color_range)
    }

    /// The sample layout the stream is coded in.
    pub fn layout(&self) -> Layout {
        layout_of(&self.pix_fmt)
    }

    /// The extension of the CLI's bi-planar golden of this fixture (`nv12`, `p010`, ...).
    pub fn planes_ext(&self) -> &'static str {
        match (self.layout().chroma, self.layout().depth) {
            (Chroma::C420, 8) => "nv12",
            (Chroma::C422, 8) => "nv16",
            (Chroma::C444, 8) => "nv24",
            (Chroma::C420, _) => "p010",
            (Chroma::C422, _) => "p210",
            (Chroma::C444, _) => "p410",
        }
    }
}

/// The layout of an ffmpeg pixel format name (`yuv420p`, `yuvj420p`, `yuv422p10le`, ...).
pub fn layout_of(pix_fmt: &str) -> Layout {
    let name = pix_fmt.replace("yuvj", "yuv");
    let chroma = match &name[3..6] {
        "420" => Chroma::C420,
        "422" => Chroma::C422,
        "444" => Chroma::C444,
        other => panic!("pixel format {pix_fmt:?} has chroma {other:?}"),
    };
    Layout {
        chroma,
        depth: if name.contains("10") { 10 } else { 8 },
    }
}

pub fn matrix_named(tag: &str) -> Matrix {
    match tag {
        "bt709" => Matrix::Bt709,
        "bt2020nc" => Matrix::Bt2020Nc,
        "fcc" => Matrix::Fcc,
        "smpte240m" => Matrix::Smpte240m,
        "smpte170m" | "bt470bg" | "unspecified" => Matrix::Bt601,
        other => panic!("matrix tag {other:?} is not one the fixtures use"),
    }
}

pub fn range_named(tag: &str) -> Range {
    match tag {
        "pc" => Range::Full,
        "tv" | "unspecified" => Range::Limited,
        other => panic!("range tag {other:?} is not one the fixtures use"),
    }
}

fn kv(line: &str) -> (String, String, HashMap<String, String>) {
    let mut it = line.split(' ');
    let kind = it.next().expect("kind").to_string();
    let name = it.next().expect("name").to_string();
    let map = it
        .filter_map(|p| p.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    (kind, name, map)
}

fn facts_text() -> String {
    String::from_utf8(bytes("facts.txt")).expect("facts.txt is UTF-8")
}

pub fn facts() -> Vec<Fact> {
    facts_text()
        .lines()
        .filter(|l| l.starts_with("video "))
        .map(|l| {
            let (_, name, m) = kv(l);
            let get = |k: &str| {
                m.get(k)
                    .unwrap_or_else(|| panic!("{name}: facts.txt has no {k}"))
                    .as_str()
            };
            let list = |k: &str| -> Vec<&str> { get(k).split(',').filter(|s| !s.is_empty()).collect() };
            Fact {
                width: get("w").parse().unwrap(),
                height: get("h").parse().unwrap(),
                rotation: get("rotation").parse().unwrap(),
                flip: get("flip").to_string(),
                golden: get("golden") == "yes",
                frames: get("frames").parse().unwrap(),
                pix_fmt: get("pix_fmt").to_string(),
                color_range: get("color_range").to_string(),
                color_space: get("color_space").to_string(),
                duration: get("duration").parse().unwrap(),
                average_fps: get("average_fps").parse().unwrap(),
                pts: list("pts").iter().map(|s| s.parse().unwrap()).collect(),
                sampled: m
                    .iter()
                    .filter(|(k, _)| k.starts_with("sampled"))
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            v.split(',')
                                .filter(|x| !x.is_empty())
                                .map(|x| x.parse().unwrap())
                                .collect(),
                        )
                    })
                    .collect(),
                name,
            }
        })
        .collect()
}

pub fn fact(name: &str) -> Fact {
    facts()
        .into_iter()
        .find(|f| f.name == name)
        .unwrap_or_else(|| panic!("facts.txt has no video {name}"))
}

/// One `synthetic` line of `facts.txt`: random NV12 planes and the CLI's
/// conversion of them.
#[derive(Clone, Debug)]
pub struct Synthetic {
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub matrix: Matrix,
    pub range: Range,
}

pub fn synthetics() -> Vec<Synthetic> {
    facts_text()
        .lines()
        .filter(|l| l.starts_with("synthetic "))
        .map(|l| {
            let (_, name, m) = kv(l);
            Synthetic {
                width: m["w"].parse().unwrap(),
                height: m["h"].parse().unwrap(),
                matrix: matrix_named(&m["matrix"]),
                range: range_named(&m["range"]),
                name,
            }
        })
        .collect()
}

/// One `format` line of `facts.txt`: random planar planes of one layout
/// (`syn_<fmt>_<matrix>_<range>_<w>x<h>.planar`) and the CLI's conversion of them.
#[derive(Clone, Debug)]
pub struct FormatCase {
    pub name: String,
    pub pix_fmt: String,
    pub layout: Layout,
    pub width: usize,
    pub height: usize,
    pub matrix: Matrix,
    pub range: Range,
}

pub fn format_cases() -> Vec<FormatCase> {
    facts_text()
        .lines()
        .filter(|l| l.starts_with("format "))
        .map(|l| {
            let (_, name, m) = kv(l);
            FormatCase {
                pix_fmt: m["fmt"].clone(),
                layout: layout_of(&m["fmt"]),
                width: m["w"].parse().unwrap(),
                height: m["h"].parse().unwrap(),
                matrix: matrix_named(&m["matrix"]),
                range: range_named(&m["range"]),
                name,
            }
        })
        .collect()
}

/// A format case's planes as the converter takes them: luma, and interleaved Cb, Cr.
pub struct Planes {
    pub y: Vec<u16>,
    pub uv: Vec<u16>,
}

impl FormatCase {
    /// Width and height of the chroma planes.
    pub fn chroma_size(&self) -> (usize, usize) {
        match self.layout.chroma {
            Chroma::C420 => (self.width.div_ceil(2), self.height.div_ceil(2)),
            Chroma::C422 => (self.width.div_ceil(2), self.height),
            Chroma::C444 => (self.width, self.height),
        }
    }

    /// The `.planar` file, read and interleaved.
    pub fn planes(&self) -> Planes {
        let raw = bytes(&format!("{}.planar", self.name));
        let samples: Vec<u16> = if self.layout.depth == 8 {
            raw.iter().map(|&b| u16::from(b)).collect()
        } else {
            raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
        };
        let (cw, ch) = self.chroma_size();
        let (ny, nc) = (self.width * self.height, cw * ch);
        assert_eq!(samples.len(), ny + 2 * nc, "{}: planar size", self.name);
        let (y, rest) = samples.split_at(ny);
        let (cb, cr) = rest.split_at(nc);
        Planes {
            y: y.to_vec(),
            uv: cb.iter().zip(cr).flat_map(|(&b, &r)| [b, r]).collect(),
        }
    }
}

/// `(index of the first differing byte, number of differing bytes, largest difference)`,
/// or `None` when equal.
pub fn first_diff(got: &[u8], want: &[u8]) -> Option<(usize, usize, u8)> {
    assert_eq!(got.len(), want.len(), "length");
    let mut first = None;
    let (mut n, mut max) = (0usize, 0u8);
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        if a != b {
            first.get_or_insert(i);
            n += 1;
            max = max.max(a.abs_diff(*b));
        }
    }
    first.map(|f| (f, n, max))
}
