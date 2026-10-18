//! `tessl::media::resample` and the audio path built on it, against the `ffmpeg`
//! binary (libswresample, the library torchcodec's `AudioDecoder` calls), with
//! no device.
//!
//! The fixtures under `tests/fixtures/media_resample/` and the goldens next to
//! them are made by `scripts/gen_resample_fixtures.py` (see its docstring); each
//! `<name>.ffmpeg.f32` is `ffmpeg -i <name>.wav -ac 1 -ar 16000 -f f32le`. The
//! converter is held to those bits: every rate, layout and length below must
//! come out with zero differing samples, so the measured bound is 0 ULP. (That
//! is the AArch64 build of FFmpeg, whose float dot product sums in NEON order;
//! an x86 build sums in another order and would differ in the last bits.)
//!
//! Where the audio path cannot be bit-equal, the bound is a measurement
//! printed by the test that uses it (run with `--nocapture`): the codecs
//! AudioToolbox decodes in place of FFmpeg's (AAC, MP3, Opus).

use std::path::{Path, PathBuf};

use tessl::media::audio::{decode_audio_bytes, decode_audio_file, AudioContainer, AudioOptions, Downmix, Resampler};
use tessl::media::resample::{convert_interleaved, resample, SwrConverter, SwrMix, SwrResampler, MAX_RATE};
use tessl::media::MediaError;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/media_resample")
}

fn read(name: &str) -> Vec<u8> {
    let p = dir().join(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e} (run scripts/gen_resample_fixtures.py)", p.display()))
}

fn f32s(name: &str) -> Vec<f32> {
    read(name)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `(interleaved samples, channels, rate)` of a 16-bit PCM WAV the generator wrote
/// (a 44-byte header, or an extensible one: the `data` chunk is found).
fn wav(name: &str) -> (Vec<f32>, usize, u32) {
    let b = read(name);
    assert_eq!(&b[..4], b"RIFF");
    let mut at = 12;
    let (mut channels, mut rate) = (0usize, 0u32);
    while at + 8 <= b.len() {
        let id = &b[at..at + 4];
        let len = u32::from_le_bytes(b[at + 4..at + 8].try_into().unwrap()) as usize;
        if id == b"fmt " {
            assert_eq!(
                u16::from_le_bytes([b[at + 22], b[at + 23]]),
                16,
                "{name}: 16-bit PCM expected"
            );
            channels = u16::from_le_bytes([b[at + 10], b[at + 11]]) as usize;
            rate = u32::from_le_bytes(b[at + 12..at + 16].try_into().unwrap());
        } else if id == b"data" {
            let samples = b[at + 8..at + 8 + len]
                .chunks_exact(2)
                .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
                .collect();
            return (samples, channels, rate);
        }
        at += 8 + len + (len & 1);
    }
    panic!("{name}: no data chunk");
}

/// Every `*.wav` that has an `.ffmpeg.f32` golden, sorted.
fn cases() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| {
            n.ends_with(".wav")
                && dir()
                    .join(format!("{}.ffmpeg.f32", n.trim_end_matches(".wav")))
                    .is_file()
        })
        .collect();
    names.sort();
    names
}

/// `(differing samples, max ULP)` between two equal-length outputs.
fn ulps(got: &[f32], want: &[f32]) -> (usize, u64) {
    assert_eq!(got.len(), want.len(), "length");
    let key = |v: f32| -> i64 {
        let b = v.to_bits() as i32;
        i64::from(if b < 0 { i32::MIN - b } else { b })
    };
    let mut n = 0;
    let mut max = 0u64;
    for (a, b) in got.iter().zip(want) {
        if a.to_bits() != b.to_bits() {
            n += 1;
            max = max.max(key(*a).abs_diff(key(*b)));
        }
    }
    (n, max)
}

fn snr_db(reference: &[f32], got: &[f32]) -> f64 {
    let n = reference.len().min(got.len());
    let (mut s, mut e) = (0f64, 0f64);
    for i in 0..n {
        s += f64::from(reference[i]).powi(2);
        e += f64::from(reference[i] - got[i]).powi(2);
    }
    10.0 * (s / e.max(1e-300)).log10()
}

// ---------------------------------------------------------------------------
// The converter against libswresample
// ---------------------------------------------------------------------------

#[test]
fn every_rate_layout_and_length_equals_libswresample_bit_for_bit() {
    let names = cases();
    assert_eq!(names.len(), 44, "the generator writes 44 inputs with a golden");
    let (mut samples, mut worst) = (0usize, 0u64);
    let mut failures = Vec::new();
    for name in &names {
        let stem = name.trim_end_matches(".wav");
        let (x, ch, rate) = wav(name);
        let want = f32s(&format!("{stem}.ffmpeg.f32"));
        let got = convert_interleaved(&x, ch, rate, 16_000).unwrap_or_else(|e| panic!("{name}: {e}"));
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
        let up = dir().join(format!("{stem}.ffmpeg48k.f32"));
        if up.is_file() {
            let want = f32s(&format!("{stem}.ffmpeg48k.f32"));
            let got = convert_interleaved(&x, ch, rate, 48_000).unwrap();
            if got.len() != want.len() || ulps(&got, &want).0 != 0 {
                failures.push(format!("{name}: to 48 kHz differs"));
            }
        }
    }
    eprintln!(
        "{} inputs, {samples} output samples compared with ffmpeg's, max {worst} ULP",
        names.len()
    );
    assert!(failures.is_empty(), "not bit exact:\n{}", failures.join("\n"));
}

#[test]
fn the_filter_bank_shape_follows_the_reduced_ratio() {
    // (in, out) -> (taps, phases): `ceil(32 / (out * 0.97 / in))` rounded up to
    // even; the reduced numerator of out/in when it is at most 1024, else 1024.
    for (rin, rout, taps, phases) in [
        (44_100, 16_000, 92, 160),
        (48_000, 16_000, 100, 1),
        (96_000, 16_000, 198, 1),
        (32_000, 16_000, 66, 1),
        (22_050, 16_000, 46, 320),
        (11_025, 16_000, 32, 640),
        (8_000, 16_000, 32, 2),
        (44_101, 16_000, 92, 1024),
        (16_000, 16_000, 34, 1),
    ] {
        let r = SwrResampler::new(rin, rout).unwrap();
        assert_eq!((r.taps(), r.phases()), (taps, phases), "{rin} -> {rout}");
    }
}

#[test]
fn the_output_does_not_depend_on_how_the_input_is_chunked() {
    for stem in [
        "sweep_44100_mono",
        "sweep_8000_mono",
        "sweep_96000_mono",
        "sweep_44101_stereo",
        "noise_48000_mono_long",
    ] {
        let (x, ch, rate) = wav(&format!("{stem}.wav"));
        let want = f32s(&format!("{stem}.ffmpeg.f32"));
        for seed in [1u64, 2, 3] {
            let mut conv = SwrConverter::new(ch, rate, 16_000).unwrap();
            let mut out = Vec::new();
            let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let frames = x.len() / ch;
            let mut at = 0;
            while at < frames {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Chunks of 0 to 2500 frames, some empty.
                let take = ((state % 2501) as usize).min(frames - at);
                conv.push(&x[at * ch..(at + take) * ch], &mut out);
                at += take;
            }
            conv.finish(&mut out);
            assert_eq!(out.len(), want.len(), "{stem} seed {seed}");
            assert_eq!(ulps(&out, &want).0, 0, "{stem} seed {seed}");
        }
    }
}

#[test]
fn a_different_rate_is_not_equal() {
    // A control: the comparison above can fail.
    let (x, ch, _) = wav("sweep_44100_mono.wav");
    let want = f32s("sweep_44100_mono.ffmpeg.f32");
    let got = convert_interleaved(&x, ch, 44_101, 16_000).unwrap();
    assert!(got.len() != want.len() || ulps(&got, &want).0 > 0);
}

#[test]
fn a_signal_too_short_for_one_window_gives_nothing_as_libswresample_does() {
    // 44.1 -> 16 kHz: 92 taps; after the end mirror a signal of n samples has
    // n + (n + 1) / 2 of them, and the first window needs 93.
    for n in [0usize, 1, 2, 61] {
        let x = vec![0.25f32; n];
        assert!(resample(&x, 44_100, 16_000).unwrap().is_empty(), "n = {n}");
    }
    assert_eq!(resample(&vec![0.25f32; 62], 44_100, 16_000).unwrap().len(), 18);
}

#[test]
fn a_constant_stays_constant_and_silence_stays_silent() {
    let out = resample(&vec![0.5f32; 20_000], 48_000, 16_000).unwrap();
    assert!(!out.is_empty());
    for (i, v) in out.iter().enumerate() {
        assert!((v - 0.5).abs() < 1e-6, "sample {i}: {v}");
    }
    assert!(resample(&vec![0f32; 5_000], 44_100, 16_000)
        .unwrap()
        .iter()
        .all(|v| *v == 0.0));
}

#[test]
fn rates_and_sizes_are_validated_before_anything_is_allocated() {
    for (a, b) in [(0, 16_000), (16_000, 0), (MAX_RATE + 1, 16_000), (16_000, MAX_RATE + 1)] {
        assert!(
            matches!(SwrResampler::new(a, b), Err(MediaError::Invalid { .. })),
            "{a} -> {b}"
        );
    }
    // 1 MHz down to 1 Hz would need a 33-million-tap filter.
    assert!(matches!(
        SwrResampler::new(1_000_000, 1),
        Err(MediaError::Invalid { .. })
    ));
    assert!(matches!(SwrMix::new(0), Err(MediaError::Unsupported { .. })));
    assert!(matches!(SwrMix::new(9), Err(MediaError::Unsupported { .. })));
    assert!(matches!(
        convert_interleaved(&[0.0; 5], 2, 44_100, 16_000),
        Err(MediaError::Invalid { .. })
    ));
    assert!(matches!(
        convert_interleaved(&[], 0, 44_100, 16_000),
        Err(MediaError::Invalid { .. })
    ));
}

// ---------------------------------------------------------------------------
// The audio path: AudioToolbox decodes, this converts
// ---------------------------------------------------------------------------

fn mean() -> AudioOptions {
    AudioOptions {
        downmix: Downmix::Mean,
        ..AudioOptions::default()
    }
}

#[test]
fn a_wav_file_through_decode_audio_file_equals_ffmpegs_decode_bit_for_bit() {
    let (mut compared, mut empty) = (0, 0);
    for name in cases() {
        let stem = name.trim_end_matches(".wav");
        let want = f32s(&format!("{stem}.ffmpeg.f32"));
        let from_path = decode_audio_file(&dir().join(&name), &AudioOptions::default());
        if want.is_empty() {
            // A clip the converter cannot fill one window of is an error, not silence.
            assert!(
                matches!(from_path, Err(MediaError::Decode { .. })),
                "{name}: {from_path:?}"
            );
            empty += 1;
            continue;
        }
        let got = from_path.unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(got.sample_rate, 16_000);
        assert_eq!(got.samples.len(), want.len(), "{name}");
        assert_eq!(ulps(&got.samples, &want).0, 0, "{name}: path");
        let from_bytes = decode_audio_bytes(&read(&name), AudioContainer::Auto, &AudioOptions::default()).unwrap();
        assert_eq!(from_bytes.samples, got.samples, "{name}: bytes vs path");
        compared += 1;
    }
    assert_eq!(compared + empty, 44);
    assert_eq!(empty, 5, "n = 1, 2 and 61 at 44.1 kHz and n = 1 and 2 at 8 kHz");
}

#[test]
fn the_channel_mean_taken_before_resampling_equals_ffmpegs_float_mean() {
    let mut compared = 0;
    for name in cases() {
        let stem = name.trim_end_matches(".wav");
        let golden = format!("{stem}.ffmpeg_mean.f32");
        if !dir().join(&golden).is_file() {
            continue;
        }
        let want = f32s(&golden);
        let got = decode_audio_file(&dir().join(&name), &mean()).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(got.samples.len(), want.len(), "{name}");
        let (n, max) = ulps(&got.samples, &want);
        assert_eq!(n, 0, "{name}: {n} samples differ, max {max} ULP");
        compared += 1;
    }
    assert!(compared >= 6, "{compared}");
}

#[test]
fn the_swresample_mix_is_root_two_louder_than_the_mean() {
    let (x, _, rate) = wav("sweep_48000_stereo.wav");
    let sw = decode_audio_file(&dir().join("sweep_48000_stereo.wav"), &AudioOptions::default()).unwrap();
    let mn = decode_audio_file(&dir().join("sweep_48000_stereo.wav"), &mean()).unwrap();
    assert_eq!(rate, 48_000);
    assert_eq!(x.len() / 2, 9_600);
    let dot: f64 = sw
        .samples
        .iter()
        .zip(&mn.samples)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let nn: f64 = mn.samples.iter().map(|b| f64::from(*b).powi(2)).sum();
    assert!((dot / nn - std::f64::consts::SQRT_2).abs() < 1e-5, "{}", dot / nn);
}

#[test]
fn the_audiotoolbox_resampler_is_still_selectable_and_close() {
    let name = "sweep_44100_mono.wav";
    let want = f32s("sweep_44100_mono.ffmpeg.f32");
    let apple = decode_audio_file(
        &dir().join(name),
        &AudioOptions {
            resampler: Resampler::AudioToolbox,
            ..AudioOptions::default()
        },
    )
    .unwrap();
    assert!(
        apple.samples.len().abs_diff(want.len()) <= 1,
        "{} vs {}",
        apple.samples.len(),
        want.len()
    );
    // The interior, 20 ms from each end (the converters mirror the edges differently).
    let snr = snr_db(&want[320..want.len() - 320], &apple.samples[320..want.len() - 320]);
    eprintln!("AudioToolbox converter vs libswresample, interior: {snr:.1} dB");
    assert!(snr > 40.0, "{snr}");
}

#[test]
fn truncation_with_resampling_returns_the_head_of_the_full_decode() {
    let name = "sweep_44100_stereo_long.wav";
    let full = decode_audio_file(&dir().join(name), &AudioOptions::default()).unwrap();
    assert_eq!(full.samples.len(), f32s("sweep_44100_stereo_long.ffmpeg.f32").len());
    let tight = AudioOptions {
        max_seconds: 0.1,
        ..AudioOptions::default()
    };
    assert!(matches!(
        decode_audio_file(&dir().join(name), &tight),
        Err(MediaError::Limit { .. })
    ));
    let cut = decode_audio_file(
        &dir().join(name),
        &AudioOptions {
            truncate: true,
            ..tight
        },
    )
    .unwrap();
    assert!(cut.truncated);
    assert_eq!(cut.samples, full.samples[..1600]);
}

#[test]
fn the_mean_of_any_channel_count_is_summed_in_f32_then_resampled() {
    for name in ["sweep_48000_51", "sweep_48000_c8", "sweep_44100_51"] {
        let (x, ch, rate) = wav(&format!("{name}.wav"));
        let mono: Vec<f32> = x.chunks_exact(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect();
        let want = resample(&mono, rate, 16_000).unwrap();
        let got = decode_audio_file(&dir().join(format!("{name}.wav")), &mean()).unwrap();
        assert_eq!(got.source_channels as usize, ch);
        assert_eq!(got.samples, want, "{name}");
    }
}

// ---------------------------------------------------------------------------
// Ogg Opus
// ---------------------------------------------------------------------------

/// The shift `k` in `-max..=max` that makes `got[i + k]` best match `want[i]`
/// over `from..to`.
fn best_lag(want: &[f32], got: &[f32], from: usize, to: usize, max: i64) -> i64 {
    (-max..=max)
        .max_by(|&p, &q| {
            let s = |k: i64| snr_db_range(want, got, from, to, k);
            s(p).total_cmp(&s(q))
        })
        .unwrap()
}

fn snr_db_range(want: &[f32], got: &[f32], from: usize, to: usize, k: i64) -> f64 {
    let (mut s, mut e) = (0f64, 0f64);
    for i in from..to {
        let g = f64::from(got[(i as i64 + k) as usize]);
        s += f64::from(want[i]).powi(2);
        e += (f64::from(want[i]) - g).powi(2);
    }
    10.0 * (s / e.max(1e-300)).log10()
}

fn opus48(name: &str) -> Vec<f32> {
    decode_audio_file(
        &dir().join(name),
        &AudioOptions {
            sample_rate: 48_000,
            ..AudioOptions::default()
        },
    )
    .unwrap_or_else(|e| panic!("{name}: {e}"))
    .samples
}

/// `(stem, ffmpeg's native decoder's lag behind libopus in samples at 48 kHz,
/// floor of AudioToolbox's SNR against libopus after the first 2000 samples)`:
/// narrow-band SILK, wide-band SILK, CELT. AudioToolbox's SILK decode is
/// libopus's to the last bit (3025 dB is the floating-point floor of the
/// measurement); its CELT decode is FFmpeg's native one (138.7 dB) and 82.2 dB
/// from libopus's.
const OPUS: [(&str, i64, f64); 3] = [
    ("opus_silknb_48000_mono", 4, 150.0),
    ("opus_silkwb_48000_mono", 2, 150.0),
    ("opus_celt_48000_mono", 0, 78.0),
];

#[test]
fn ogg_opus_is_aligned_with_libopus_and_has_the_pre_skip_removed() {
    for (stem, _, floor) in OPUS {
        let got = opus48(&format!("{stem}.opus"));
        let lib = f32s(&format!("{stem}.libopus48k.f32"));
        assert_eq!(got.len(), lib.len(), "{stem}: length (the pre-skip is trimmed)");
        let lag = best_lag(&lib, &got, 2_000, lib.len() - 2_000, 8);
        let after = snr_db(&lib[2_000..], &got[2_000..]);
        let whole = snr_db(&lib, &got);
        eprintln!("{stem}: best lag vs libopus {lag}; SNR {whole:.1} dB whole, {after:.1} dB after 2000 samples");
        assert_eq!(lag, 0, "{stem}: best alignment against libopus at 48 kHz");
        assert!(after > floor, "{stem}: {after:.1} dB under {floor}");
    }
}

#[test]
fn the_lag_behind_libopus_is_ffmpegs_native_opus_decoder_not_audiotoolbox() {
    // ffmpeg's default Opus decoder (the one `ffmpeg` and torchcodec use) is its
    // own; in SILK frames it runs behind libopus (its resampler out of SILK's
    // internal rate), in CELT frames it is identical to it up to float noise.
    // AudioToolbox is aligned with libopus in all three.
    for (stem, native_lag, _) in OPUS {
        let lib = f32s(&format!("{stem}.libopus48k.f32"));
        let nat = f32s(&format!("{stem}.native48k.f32"));
        assert_eq!(lib.len(), nat.len(), "{stem}");
        // native[i + k] matches libopus[i]: k samples behind.
        let lag = best_lag(&lib, &nat, 2_000, lib.len() - 2_000, 8);
        let tess = opus48(&format!("{stem}.opus"));
        let vs_native = snr_db(&nat, &tess);
        eprintln!("{stem}: ffmpeg native is {lag} samples behind libopus; AudioToolbox vs native {vs_native:.1} dB");
        assert_eq!(lag, native_lag, "{stem}");
        if native_lag == 0 {
            assert!(vs_native > 120.0, "{stem}: {vs_native:.1} dB");
        } else {
            assert!(vs_native < 40.0, "{stem}: {vs_native:.1} dB");
        }
    }
}

#[test]
fn ogg_opus_resampled_to_16k_meets_its_measured_bounds() {
    // Against ffmpeg's native decoder, the reference's: CELT matches to float
    // noise; SILK differs by that decoder's own lag, not by anything here.
    for (stem, floor) in [
        ("opus_celt_48000_mono", 120.0),
        ("opus_silkwb_48000_mono", 18.0),
        ("opus_silknb_48000_mono", 13.0),
    ] {
        let want = f32s(&format!("{stem}.ffmpeg.f32"));
        let got = decode_audio_file(&dir().join(format!("{stem}.opus")), &AudioOptions::default()).unwrap();
        assert_eq!(got.samples.len(), want.len(), "{stem}");
        let snr = snr_db(&want, &got.samples);
        eprintln!("{stem}: 16 kHz SNR vs ffmpeg (native opus decoder) {snr:.1} dB");
        assert!(snr > floor, "{stem}: {snr:.1} dB");
    }
}

// ---------------------------------------------------------------------------
// AAC and MP3
// ---------------------------------------------------------------------------

#[test]
fn aac_and_mp3_meet_their_measured_bounds_against_ffmpegs_decode() {
    // The decoders differ (AudioToolbox's against FFmpeg's `aac` and `mp3float`);
    // the resampling and the mix now do not. `(file, floor of the whole-file
    // SNR, floor of the interior SNR, ceiling of the max abs difference)`; the
    // interior leaves out 1024 samples at each end, where the AAC decoders
    // treat the first and last frames differently (measured on the 44.1 kHz
    // decode too, before any resampling).
    for (name, whole_floor, inner_floor, max_ceiling) in [
        ("codec_44100_stereo.m4a", 54.0, 98.0, 2e-2f32),
        ("codec_44100_stereo.mp3", 92.0, 92.0, 1e-4f32),
    ] {
        let want = f32s(&format!("{name}.ffmpeg.f32"));
        let got = decode_audio_file(&dir().join(name), &AudioOptions::default()).unwrap();
        assert_eq!(got.samples.len(), want.len(), "{name}: frames");
        let (whole, inner) = (
            snr_db(&want, &got.samples),
            snr_db(&want[1024..want.len() - 1024], &got.samples[1024..want.len() - 1024]),
        );
        let max = want
            .iter()
            .zip(&got.samples)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("{name}: SNR vs ffmpeg {whole:.1} dB whole, {inner:.1} dB interior, max abs {max:.2e}");
        assert!(whole > whole_floor, "{name}: {whole:.1} dB whole");
        assert!(inner > inner_floor, "{name}: {inner:.1} dB interior");
        assert!(max < max_ceiling, "{name}: max abs {max:.2e}");
    }
}
