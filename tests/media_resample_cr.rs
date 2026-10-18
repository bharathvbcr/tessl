//! `tessl::media::resample` against the `ffmpeg` binary on the black-box fixtures of
//! `tests/fixtures/media_resample_cr/` (made by `scripts/gen_resample_cr_fixtures.py`).
//! Every comparison is bit exact: a differing sample fails the test. No device is
//! opened and no `ffmpeg` is run; the goldens are committed.

use std::path::{Path, PathBuf};

use tessl::media::resample::{convert_interleaved, resample, SwrConverter, SwrMix, SwrResampler, MAX_RATE};
use tessl::media::MediaError;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/media_resample_cr")
}

fn read(name: &str) -> Vec<u8> {
    let p = dir().join(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e} (run scripts/gen_resample_cr_fixtures.py)", p.display()))
}

fn f32s(name: &str) -> Vec<f32> {
    read(name)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn text(name: &str) -> String {
    String::from_utf8(read(name)).unwrap()
}

/// `(name, channels, in rate, out rate)` of every plain case.
fn cases() -> Vec<(String, usize, u32, u32)> {
    text("cases.csv")
        .lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (
                f[0].to_string(),
                f[1].parse().unwrap(),
                f[2].parse().unwrap(),
                f[3].parse().unwrap(),
            )
        })
        .collect()
}

/// `(rate in, rate out)` of every length sweep.
fn sweeps() -> Vec<(u32, u32)> {
    let mut v: Vec<(u32, u32)> = std::fs::read_dir(dir())
        .unwrap()
        .filter_map(|e| {
            let n = e.unwrap().file_name().to_string_lossy().into_owned();
            let stem = n.strip_prefix("lens_")?.strip_suffix(".lens.csv")?.to_string();
            let (a, b) = stem.split_once('_')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .collect();
    v.sort();
    v
}

/// `(differing samples, max ULP)` between two equal-length outputs.
fn ulps(got: &[f32], want: &[f32]) -> (usize, u64) {
    assert_eq!(got.len(), want.len(), "length");
    let key = |v: f32| -> i64 {
        let b = v.to_bits() as i32;
        i64::from(if b < 0 { i32::MIN - b } else { b })
    };
    let (mut n, mut max) = (0, 0u64);
    for (a, b) in got.iter().zip(want) {
        if a.to_bits() != b.to_bits() {
            n += 1;
            max = max.max(key(*a).abs_diff(key(*b)));
        }
    }
    (n, max)
}

#[test]
fn every_plain_case_equals_ffmpeg_bit_for_bit() {
    let cs = cases();
    assert!(cs.len() >= 90, "{} cases", cs.len());
    let mut failures = Vec::new();
    let (mut samples, mut worst) = (0usize, 0u64);
    for (name, ch, rin, rout) in &cs {
        let x = f32s(&format!("{name}.in.f32"));
        let want = f32s(&format!("{name}.ffmpeg.f32"));
        let got = convert_interleaved(&x, *ch, *rin, *rout).unwrap_or_else(|e| panic!("{name}: {e}"));
        if got.len() != want.len() {
            failures.push(format!("{name}: {} samples out, ffmpeg {}", got.len(), want.len()));
            continue;
        }
        let (n, max) = ulps(&got, &want);
        samples += want.len();
        worst = worst.max(max);
        if n != 0 {
            failures.push(format!("{name}: {n} of {} samples differ, max {max} ULP", want.len()));
        }
    }
    eprintln!("{} cases, {samples} samples compared, max {worst} ULP", cs.len());
    assert!(failures.is_empty(), "not bit exact:\n{}", failures.join("\n"));
}

#[test]
fn every_length_gives_ffmpegs_sample_count_and_samples() {
    let sw = sweeps();
    assert!(sw.len() >= 12, "{} sweeps", sw.len());
    let mut failures = Vec::new();
    let mut runs = 0;
    for (rin, rout) in sw {
        let stem = format!("lens_{rin}_{rout}");
        let base = f32s(&format!("{stem}.in.f32"));
        let want = f32s(&format!("{stem}.ffmpeg.f32"));
        let mut at = 0;
        for line in text(&format!("{stem}.lens.csv")).lines().skip(1) {
            let (n, len) = line.split_once(',').unwrap();
            let (n, len): (usize, usize) = (n.parse().unwrap(), len.parse().unwrap());
            let got = resample(&base[..n], rin, rout).unwrap();
            runs += 1;
            if got.len() != len {
                failures.push(format!("{rin} -> {rout}, n = {n}: {} out, ffmpeg {len}", got.len()));
            } else if ulps(&got, &want[at..at + len]).0 != 0 {
                failures.push(format!("{rin} -> {rout}, n = {n}: samples differ"));
            }
            at += len;
        }
        assert_eq!(at, want.len(), "{stem}: the lengths account for every golden sample");
    }
    eprintln!("{runs} lengths compared");
    assert!(failures.is_empty(), "not bit exact:\n{}", failures.join("\n"));
}

#[test]
fn chunking_never_changes_the_output() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for (name, ch, rin, rout) in cases() {
        // A third of the cases, and every layout and interpolating pair.
        if !(name.starts_with("layout_") || name.starts_with("lens") || rin % 3 == 0 || rin == 44101 || rin == 12345) {
            continue;
        }
        let x = f32s(&format!("{name}.in.f32"));
        let want = f32s(&format!("{name}.ffmpeg.f32"));
        let frames = x.len() / ch;
        for _ in 0..3 {
            let mut conv = SwrConverter::new(ch, rin, rout).unwrap();
            let mut out = Vec::new();
            let mut at = 0;
            while at < frames {
                let take = ((next() % 301) as usize).min(frames - at);
                conv.push(&x[at * ch..(at + take) * ch], &mut out);
                at += take;
            }
            conv.finish(&mut out);
            assert_eq!(ulps(&out, &want).0, 0, "{name}");
        }
    }
}

#[test]
fn the_mix_weights_are_the_measured_ones() {
    // Found by feeding a unit impulse into each channel of the `ffmpeg` binary.
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let table: [(usize, &[f32]); 8] = [
        (1, &[1.0]),
        (2, &[s, s]),
        (3, &[s, s, 0.0]),
        (4, &[s, s, 1.0, 0.5]),
        (5, &[s, s, 1.0, 0.5, 0.5]),
        (6, &[s, s, 1.0, 0.0, 0.5, 0.5]),
        (7, &[s, s, 1.0, 0.0, 0.5, 0.5, 0.5]),
        (8, &[s, s, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5]),
    ];
    for (n, weights) in table {
        let mix = SwrMix::new(n).unwrap();
        assert_eq!(mix.channels(), n);
        for (c, w) in weights.iter().enumerate() {
            let mut frames = vec![0f32; n];
            frames[c] = 1.0;
            let mut out = Vec::new();
            mix.mix_interleaved(&frames, &mut out);
            assert_eq!(out, vec![*w], "{n} channels, channel {c}");
        }
    }
}

#[test]
fn planar_and_interleaved_mixes_agree() {
    for n in 1..=8 {
        let mix = SwrMix::new(n).unwrap();
        let planar: Vec<Vec<f32>> = (0..n)
            .map(|c| (0..40).map(|i| ((i * 7 + c * 13) % 17) as f32 / 9.0 - 0.9).collect())
            .collect();
        let mut interleaved = Vec::new();
        for i in 0..40 {
            for c in &planar {
                interleaved.push(c[i]);
            }
        }
        let (mut a, mut b) = (Vec::new(), Vec::new());
        mix.mix_planar(&planar, &mut a);
        mix.mix_interleaved(&interleaved, &mut b);
        assert_eq!(a, b, "{n} channels");
    }
}

#[test]
fn equal_rates_pass_the_samples_through() {
    let x: Vec<f32> = (0..50).map(|i| i as f32 * 0.01 - 0.2).collect();
    assert_eq!(resample(&x, 16_000, 16_000).unwrap(), x);
    let a = std::f32::consts::FRAC_1_SQRT_2;
    let stereo = convert_interleaved(&[0.5, -0.25, 1.0, 1.0], 2, 16_000, 16_000).unwrap();
    assert_eq!(stereo, vec![a.mul_add(0.5, a * -0.25), a.mul_add(1.0, a * 1.0)]);
}

#[test]
fn the_shape_of_the_bank_is_what_the_formula_gives_for_other_pairs() {
    // Beyond the table in tests/media_resample.rs: ceil(32 / min(1, 0.97 * out / in)) up to even taps,
    // out / gcd phases capped at 1024.
    for (rin, rout, taps, phases) in [
        (97_000, 32_000, 100, 32),
        (97_000, 50_000, 64, 50),
        (192_000, 8_000, 792, 1),
        (16_000, 48_000, 32, 3),
        (44_100, 48_000, 32, 160),
        (12_345, 16_000, 32, 1024),
        (16_000, 1_000, 528, 1),
    ] {
        let r = SwrResampler::new(rin, rout).unwrap();
        assert_eq!((r.taps(), r.phases()), (taps, phases), "{rin} -> {rout}");
    }
}

#[test]
fn a_frame_split_across_pushes_is_carried() {
    let frames: Vec<f32> = (0..3 * 900).map(|i| ((i * 31) % 101) as f32 / 60.0 - 0.8).collect();
    let want = convert_interleaved(&frames, 3, 44_100, 16_000).unwrap();
    let mut conv = SwrConverter::new(3, 44_100, 16_000).unwrap();
    let mut out = Vec::new();
    for piece in frames.chunks(7) {
        conv.push(piece, &mut out);
    }
    conv.finish(&mut out);
    assert_eq!(out, want);
}

#[test]
fn oversized_filters_are_refused_and_the_largest_rate_is_not() {
    assert!(matches!(
        SwrResampler::new(1_000_000, 1),
        Err(MediaError::Invalid { .. })
    ));
    assert!(SwrResampler::new(MAX_RATE, 16_000).is_ok());
    assert!(SwrConverter::new(8, 44_100, 16_000).is_ok());
    assert!(matches!(
        SwrConverter::new(9, 44_100, 16_000),
        Err(MediaError::Unsupported { .. })
    ));
}

#[test]
fn a_converter_can_be_reused_after_finish() {
    let (name, _, rin, rout) = cases().into_iter().find(|c| c.0 == "rate_44100_to_16000").unwrap();
    let x = f32s(&format!("{name}.in.f32"));
    let want = f32s(&format!("{name}.ffmpeg.f32"));
    let mut r = SwrResampler::new(rin, rout).unwrap();
    for _ in 0..2 {
        let mut out = Vec::new();
        r.push(&x, &mut out);
        r.finish(&mut out);
        assert_eq!(ulps(&out, &want).0, 0);
    }
}
