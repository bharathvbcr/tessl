#!/usr/bin/env python3
"""Fixtures for tests/media_resample.rs: audio at many rates, channel layouts and
lengths, and what FFmpeg's libswresample (the library torchcodec calls) makes of
each.

    ~/.venvs/ml-2026-10/bin/python scripts/gen_resample_fixtures.py

Writes tests/fixtures/media_resample/. Every input is a 16-bit PCM WAV built from
numpy's default_rng and closed-form signals (an impulse, a log sine sweep that
crosses the converter's transition band, white noise, a step). Every golden is
the output of the `ffmpeg` binary (Homebrew, FFmpeg 9.0.2 here) for that WAV:

  <name>.ffmpeg.f32       `ffmpeg -i <name>.wav -ac 1 -ar 16000 -f f32le`: the
                          decode torchcodec's AudioDecoder(sample_rate=16000,
                          num_channels=1) makes, i.e. libswresample with its
                          defaults (32-tap Kaiser sinc at the cutoff 0.97, phases
                          from the reduced ratio, libswresample's mono matrix).
  <name>.ffmpeg48k.f32    the same to 48 kHz (an up-conversion by a rational ratio).
  <name>.ffmpeg_mean.f32  a stereo file's channel mean, taken in float before
                          resampling: `aformat=fltp,pan=mono|c0=0.5*c0+0.5*c1`
                          then the default resampler; the librosa and torchaudio
                          convention, with FFmpeg's resampler.

and the codec cases, for the bounds of the codecs AudioToolbox decodes in place of
FFmpeg's:

  <name>.m4a/.mp3         the 44.1 kHz stereo sweep encoded by ffmpeg (aac, libmp3lame)
  <name>.opus             the 48 kHz mono clips encoded with libopus (SILK and CELT)
  <name>.ffmpeg.f32       FFmpeg's default decoder (for Opus, its native one) to
                          16 kHz mono
  <name>.libopus48k.f32   libopus's decode (`-c:a libopus`) to 48 kHz mono
  <name>.native48k.f32    FFmpeg's native Opus decoder to 48 kHz mono

FFmpeg's native Opus decoder runs two samples (at 48 kHz) behind libopus in SILK
frames; see docs/embedgemma2.md ("Media decoding").
"""

import os
import shutil
import subprocess
import sys

import numpy as np
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools", "qwen35_ref"))
import fixture_manifest as fm  # noqa: E402

OUT = os.path.join(HERE, "..", "tests", "fixtures", "media_resample")
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
GENERATOR = "scripts/gen_resample_fixtures.py"
TARGET = 16000


def path(name):
    return os.path.join(OUT, name)


def quantize(x):
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)


def sweep(n, rate, phase):
    """A log sweep from 60 Hz to 0.45 of the Nyquist rate: it crosses the 16 kHz
    converter's transition band whenever the input rate is above 16 kHz."""
    t = np.arange(n) / rate
    dur = max(n / rate, 1e-9)
    f0, f1 = 60.0, 0.45 * rate / 2
    k = np.log(f1 / f0)
    return 0.6 * np.sin(2 * np.pi * f0 * dur / k * (np.exp(k * t / dur) - 1.0) + phase)


def noise(n, seed):
    return np.random.default_rng(seed).integers(-12000, 12000, n).astype(np.float64) / 32767.0


def impulse(n, at):
    x = np.zeros(n)
    x[at] = 0.5
    return x


def step(n, at):
    x = np.zeros(n)
    x[at:] = 0.5
    return x


def signal(kind, n, rate, nch):
    """(n, nch) float64 in [-1, 1] for a named signal."""
    cols = []
    for c in range(nch):
        if kind == "sweep":
            cols.append(sweep(n, rate, 0.4 * c))
        elif kind == "noise":
            cols.append(noise(n, 1000 * c + rate % 997 + n % 101))
        elif kind == "impulse":
            cols.append(impulse(n, max(1, n // 3) + c))
        elif kind == "step":
            cols.append(step(n, max(1, n // 3)) * (1.0 - 0.3 * c))
        else:
            raise SystemExit(f"unknown signal {kind}")
    return np.stack(cols, axis=1)


LAYOUT_CHANNELS = {"mono": 1, "stereo": 2, "51": 6}


def ffmpeg(args):
    subprocess.run([FFMPEG, "-v", "error", "-y"] + args, check=True)


def golden(wav, out, rate=TARGET, extra=None):
    ffmpeg(["-i", wav] + (extra or ["-ac", "1"]) + ["-ar", str(rate), "-f", "f32le", out])


files = {}


def note(name, role, text):
    files[name] = {"role": role, "published": True, "note": text}


def add_wav(name, kind, n, rate, layout, mean=False, also_48k=False):
    nch = LAYOUT_CHANNELS[layout]
    x = signal(kind, n, rate, nch)
    wavfile.write(path(name + ".wav"), rate, quantize(x if nch > 1 else x[:, 0]))
    note(name + ".wav", "input", f"{kind}, {n} frames at {rate} Hz, {layout}, 16-bit PCM")
    golden(path(name + ".wav"), path(name + ".ffmpeg.f32"))
    note(name + ".ffmpeg.f32", "golden", "ffmpeg -ac 1 -ar 16000: libswresample's defaults, f32 little endian")
    if also_48k:
        golden(path(name + ".wav"), path(name + ".ffmpeg48k.f32"), rate=48000)
        note(name + ".ffmpeg48k.f32", "golden", "the same to 48 kHz")
    if mean:
        ffmpeg(
            [
                "-i", path(name + ".wav"),
                "-af", "aformat=sample_fmts=fltp,pan=mono|c0=0.5*c0+0.5*c1",
                "-ar", str(TARGET), "-f", "f32le", path(name + ".ffmpeg_mean.f32"),
            ]
        )
        note(name + ".ffmpeg_mean.f32", "golden", "channel mean in float, then libswresample's resampler")


def codec_cases():
    """AAC and MP3 of a 44.1 kHz stereo sweep; Opus clips in SILK and CELT modes."""
    sr = 44100
    n = sr // 2
    x = signal("sweep", n, sr, 2)
    wavfile.write(path("codec_src_44100_stereo.wav"), sr, quantize(x))
    note("codec_src_44100_stereo.wav", "input", "the sweep the AAC and MP3 files were encoded from")
    for ext, args in (("m4a", ["-c:a", "aac", "-b:a", "128k"]), ("mp3", ["-c:a", "libmp3lame", "-b:a", "128k"])):
        name = f"codec_44100_stereo.{ext}"
        ffmpeg(["-i", path("codec_src_44100_stereo.wav")] + args + [path(name)])
        note(name, "input", f"{ext} encode of codec_src_44100_stereo.wav by ffmpeg")
        golden(path(name), path(name + ".ffmpeg.f32"))
        note(name + ".ffmpeg.f32", "golden", "ffmpeg's decode of the file to 16 kHz mono (default mix)")
    # Opus: speech-like harmonic clips, 48 kHz mono.
    sr = 48000
    n = sr * 6 // 10
    t = np.arange(n) / sr
    rng = np.random.default_rng(5)
    f0 = 120 + 40 * np.sin(2 * np.pi * 1.3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = sum(np.sin(k * phase) / k for k in range(1, 12)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) * 0.2
    sig = sig + 0.02 * rng.standard_normal(n)
    wavfile.write(path("opus_src_48000_mono.wav"), sr, quantize(sig))
    note("opus_src_48000_mono.wav", "input", "the clip the Opus files were encoded from")
    for name, args in (
        ("opus_silk_48000_mono", ["-application", "voip", "-frame_duration", "20", "-b:a", "16k"]),
        ("opus_celt_48000_mono", ["-application", "audio", "-frame_duration", "20", "-b:a", "64k"]),
    ):
        ffmpeg(["-i", path("opus_src_48000_mono.wav"), "-c:a", "libopus"] + args + [path(name + ".opus")])
        note(name + ".opus", "input", "libopus encode of opus_src_48000_mono.wav (" + " ".join(args) + ")")
        ffmpeg(["-i", path(name + ".opus"), "-ac", "1", "-ar", str(TARGET), "-f", "f32le", path(name + ".ffmpeg.f32")])
        note(name + ".ffmpeg.f32", "golden", "ffmpeg's default decoder (native opus) to 16 kHz mono")
        ffmpeg(["-c:a", "libopus", "-i", path(name + ".opus"), "-ac", "1", "-ar", "48000", "-f", "f32le",
                path(name + ".libopus48k.f32")])
        note(name + ".libopus48k.f32", "golden", "libopus's decode at 48 kHz, pre-skip removed")
        ffmpeg(["-c:a", "opus", "-i", path(name + ".opus"), "-ac", "1", "-ar", "48000", "-f", "f32le",
                path(name + ".native48k.f32")])
        note(name + ".native48k.f32", "golden", "ffmpeg's native opus decoder at 48 kHz, pre-skip removed")


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f != "MANIFEST.json":
            os.remove(os.path.join(OUT, f))
    # Rates, mono: a quarter second plus a frame, so the length is odd for even rates.
    for rate in (8000, 11025, 22050, 32000, 44100, 48000, 96000):
        add_wav(f"sweep_{rate}_mono", "sweep", rate // 4 + 1, rate, "mono", also_48k=(rate == 44100))
    # Rates, stereo and 5.1.
    for rate in (8000, 22050, 44100, 48000, 96000):
        add_wav(f"sweep_{rate}_stereo", "sweep", rate // 5, rate, "stereo", mean=True)
    for rate in (44100, 48000):
        add_wav(f"sweep_{rate}_51", "sweep", rate // 8, rate, "51")
    # Impulses and a step.
    for rate in (8000, 44100, 48000):
        add_wav(f"impulse_{rate}_mono", "impulse", rate // 10, rate, "mono")
    add_wav("step_44100_mono", "step", 4410, 44100, "mono")
    add_wav("impulse_44100_stereo", "impulse", 4410, 44100, "stereo", mean=True)
    # Lengths around the converter's reach: the window is 92 taps at 44.1 -> 16 kHz
    # and 32 at 8 -> 16 kHz, and a signal too short to fill one after both mirrors gives nothing.
    for n in (1, 2, 61, 62, 63, 93, 200, 1001):
        add_wav(f"noise_44100_mono_n{n}", "noise", n, 44100, "mono")
    for n in (1, 2, 31, 32, 33, 64):
        add_wav(f"noise_8000_mono_n{n}", "noise", n, 8000, "mono")
    # Not an exact rational ratio (16000/44101 has a numerator over 1024): 1024
    # phases and linear interpolation. And an odd rate, the rate itself, and long clips.
    add_wav("sweep_44101_stereo", "sweep", 8000, 44101, "stereo")
    add_wav("sweep_12345_mono", "sweep", 6000, 12345, "mono")
    add_wav("sweep_16000_stereo", "sweep", 4000, 16000, "stereo")
    add_wav("sweep_16000_51", "sweep", 4000, 16000, "51")
    add_wav("noise_48000_mono_long", "noise", 48000 * 3, 48000, "mono")
    add_wav("sweep_44100_stereo_long", "sweep", 44100 * 2, 44100, "stereo", mean=True)
    codec_cases()
    fm.write_manifest(
        OUT,
        GENERATOR,
        "generated: closed-form signals and numpy default_rng draws quantised to 16-bit PCM WAV; the goldens "
        "are the ffmpeg binary's output for those exact files (see the generator docstring); the AAC, MP3 "
        "and Opus files are ffmpeg's encodes, stable for a given FFmpeg version",
        files,
        "produced",
        extra={"ffmpeg": subprocess.run([FFMPEG, "-version"], capture_output=True, text=True).stdout.splitlines()[0]},
    )
    total = sum(os.path.getsize(path(f)) for f in files)
    print(f"wrote {len(files)} files, {total / 1e6:.2f} MB, to {OUT}")


if __name__ == "__main__":
    main()
