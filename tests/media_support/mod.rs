//! Fixture access shared by `tests/media_decode.rs` and `tests/media_codecs.rs`.
//!
//! The fixtures under `tests/fixtures/media/` and the oracle outputs next to them
//! are made by `scripts/gen_media_fixtures.py`.

#![allow(dead_code)] // each of the two test binaries uses a different part

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tessl::embedgemma2_pre::PreConfig;

pub fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/media")
}

pub fn fixture(name: &str) -> PathBuf {
    dir().join(name)
}

pub fn bytes(name: &str) -> Vec<u8> {
    let p = fixture(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e} (run scripts/gen_media_fixtures.py)", p.display()))
}

/// `manifest.txt`: `kind name key=value ...` per line.
pub fn manifest() -> HashMap<String, HashMap<String, String>> {
    let text = String::from_utf8(bytes("manifest.txt")).unwrap();
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split(' ');
            let _kind = it.next().unwrap();
            let name = it.next().unwrap().to_string();
            let kv = it
                .filter_map(|p| p.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            (name, kv)
        })
        .collect()
}

pub fn f32s(name: &str) -> Vec<f32> {
    bytes(name)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `(max abs difference, number of differing bytes)`.
pub fn diff_bytes(a: &[u8], b: &[u8]) -> (u8, usize) {
    assert_eq!(a.len(), b.len(), "length");
    let mut max = 0u8;
    let mut n = 0;
    for (x, y) in a.iter().zip(b) {
        let d = x.abs_diff(*y);
        if d > 0 {
            n += 1;
            max = max.max(d);
        }
    }
    (max, n)
}

pub fn pre_config() -> PreConfig {
    PreConfig::from_file(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embedgemma2/eg2p_processor_config.json"),
    )
    .expect("the snapshot's processor_config.json")
}

pub fn centre_pixel(frame: &[u8], w: usize, h: usize) -> [u8; 3] {
    let o = ((h / 2) * w + w / 2) * 3;
    [frame[o], frame[o + 1], frame[o + 2]]
}

/// `(max abs difference, mean abs difference)` of two RGB buffers.
pub fn byte_stats(a: &[u8], b: &[u8]) -> (u8, f64) {
    assert_eq!(a.len(), b.len(), "length");
    let max = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0);
    let sum: u64 = a.iter().zip(b).map(|(x, y)| u64::from(x.abs_diff(*y))).sum();
    (max, sum as f64 / a.len().max(1) as f64)
}

pub fn manifest_pts(name: &str) -> Vec<f64> {
    manifest()[name]["pts"].split(',').map(|p| p.parse().unwrap()).collect()
}

/// Files this process left in the temp directory for video bytes.
pub fn temp_video_files() -> usize {
    let prefix = format!("tessl-media-{}-", std::process::id());
    std::fs::read_dir(std::env::temp_dir())
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
        .count()
}

/// `struct mach_task_basic_info` (`<mach/task_info.h>`); `time_value_t` is two
/// `integer_t`.
#[repr(C)]
#[derive(Default)]
struct MachTaskBasicInfo {
    virtual_size: u64,
    resident_size: u64,
    resident_size_max: u64,
    user_time: [i32; 2],
    system_time: [i32; 2],
    policy: i32,
    suspend_count: i32,
}

extern "C" {
    static mach_task_self_: u32;
    fn task_info(task: u32, flavor: u32, info: *mut MachTaskBasicInfo, count: *mut u32) -> i32;
}

/// This process's resident set size in KiB (`task_info(MACH_TASK_BASIC_INFO)`;
/// `ps` is not allowed under the hosted job's sandbox).
pub fn rss_kib() -> u64 {
    const MACH_TASK_BASIC_INFO: u32 = 20;
    let mut info = MachTaskBasicInfo::default();
    let mut count = (std::mem::size_of::<MachTaskBasicInfo>() / 4) as u32; // MACH_TASK_BASIC_INFO_COUNT
                                                                           // SAFETY: `info` is a `mach_task_basic_info` of `count` natural_t words;
                                                                           // `mach_task_self_` is the task port libSystem initialises before main.
    let status = unsafe { task_info(mach_task_self_, MACH_TASK_BASIC_INFO, &mut info, &mut count) };
    assert_eq!(status, 0, "task_info failed");
    info.resident_size / 1024
}

/// Run `f` `warmup` times, then in `BATCHES` batches of `rounds`, and return
/// the smallest resident-set growth of any batch in KiB, with the per-round
/// figure. A leak grows every batch, so the minimum sees it; the one-time rise
/// of allocator and framework caches (measured: 2.5 MiB in the first 400 calls
/// of a resampling decode, 0 after) lands in one batch and does not. It
/// detects a leak of more than about `bound / rounds` per call.
pub fn rss_growth_kib(warmup: usize, rounds: usize, mut f: impl FnMut()) -> (i64, f64) {
    const BATCHES: usize = 3;
    for _ in 0..warmup {
        f();
    }
    let mut least = i64::MAX;
    for _ in 0..BATCHES {
        let before = rss_kib() as i64;
        for _ in 0..rounds {
            f();
        }
        least = least.min(rss_kib() as i64 - before);
    }
    (least, least as f64 / rounds as f64)
}

pub const FRAME_COLORS: [(u8, u8, u8); 12] = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 128, 128),
    (255, 128, 0),
    (0, 128, 255),
    (128, 0, 255),
    (32, 32, 32),
    (224, 224, 224),
];
pub const VW: usize = 64;
pub const VH: usize = 48;
