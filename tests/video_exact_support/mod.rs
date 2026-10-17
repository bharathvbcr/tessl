//! Fixture access shared by `tests/media_yuv_exact.rs` (the converter, no codec) and
//! `tests/media_video_exact.rs` (AVFoundation decoding).
//!
//! The fixtures under `tests/fixtures/media_video/` are made by
//! `scripts/gen_media_video_exact_fixtures.py`; `facts.txt` there has one line per
//! fixture, `kind name key=value ...`.

#![allow(dead_code)] // each of the two test binaries uses a different part

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tessl::media::yuv::{Matrix, Range};

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
}

pub fn matrix_named(tag: &str) -> Matrix {
    match tag {
        "bt709" => Matrix::Bt709,
        "bt2020nc" => Matrix::Bt2020Nc,
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
            let get = |k: &str| m.get(k).unwrap_or_else(|| panic!("{name}: facts.txt has no {k}")).as_str();
            let list = |k: &str| -> Vec<&str> { get(k).split(',').filter(|s| !s.is_empty()).collect() };
            Fact {
                width: get("w").parse().unwrap(),
                height: get("h").parse().unwrap(),
                rotation: get("rotation").parse().unwrap(),
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
                            v.split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect(),
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
