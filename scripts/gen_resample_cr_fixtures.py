#!/usr/bin/env python3
"""Black-box fixtures for tests/media_resample_cr.rs: what the `ffmpeg` binary makes of
float audio at many rate pairs, channel counts and lengths. The only tool that produces
a golden is `ffmpeg` (`-ac 1 -ar R -f f32le`, default resampler options); numpy only draws
the inputs.

    ~/.venvs/ml-2026-10/bin/python scripts/gen_resample_cr_fixtures.py

Writes tests/fixtures/media_resample_cr/ (commit this script first: the manifest names
its commit). Every input is raw little-endian f32, interleaved; the WAV that `ffmpeg`
reads is built in a scratch directory and not kept.

  cases.csv                      name,channels,in_rate,out_rate for every `<name>.in.f32`
                                 / `<name>.ffmpeg.f32` pair below
  rate_*, layout_*, signal_*     one clip per rate pair, channel count or signal shape
  special_*                      noise with a NaN, an infinity or negative zeros in it
  train_*                        impulses spaced so that no output sees two of them: the
                                 output samples are filter coefficients, so the pair
                                 exercises the filter bank of that rate pair
  lens_<in>_<out>.in.f32         the first N samples of one clip; `.lens.csv` lists the
                                 lengths n run through `ffmpeg` (`n,output samples`) and
                                 `.ffmpeg.f32` is the outputs for n = each, back to back
"""

import os
import shutil
import struct
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools", "qwen35_ref"))
import fixture_manifest as fm  # noqa: E402

OUT = os.path.join(HERE, "..", "tests", "fixtures", "media_resample_cr")
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
GENERATOR = "scripts/gen_resample_cr_fixtures.py"
TARGET = 16000

files = {}


def path(name):
    return os.path.join(OUT, name)


def note(name, role, text):
    files[name] = {"role": role, "published": True, "note": text}


def wav_f32(p, data, rate):
    data = np.ascontiguousarray(data, dtype="<f4")
    n, ch = data.shape
    pcm = data.tobytes()
    hdr = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 3, ch, rate, rate * ch * 4, ch * 4, 32)
    hdr += b"data" + struct.pack("<I", len(pcm))
    with open(p, "wb") as f:
        f.write(hdr + pcm)


def ffmpeg_convert(scratch, tag, data, in_rate, out_rate):
    """`ffmpeg -i in.wav -ac 1 -ar out_rate -f f32le out.raw` on (n, ch) float data."""
    wav = os.path.join(scratch, tag + ".wav")
    raw = os.path.join(scratch, tag + ".raw")
    wav_f32(wav, data, in_rate)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", wav, "-ac", "1", "-ar", str(out_rate), "-f", "f32le", raw],
                   check=True)
    out = np.fromfile(raw, dtype="<f4")
    os.remove(wav)
    os.remove(raw)
    return out


def signal(kind, n, ch, seed):
    """(n, ch) float32 in [-1, 1]."""
    rng = np.random.default_rng(seed)
    if kind == "noise":
        return (rng.uniform(-0.9, 0.9, (n, ch))).astype(np.float32)
    if kind == "grid":  # values on the 16-bit grid, as a decoded WAV has them
        return (rng.integers(-32768, 32768, (n, ch)) / 32768.0).astype(np.float32)
    if kind.startswith("special_"):  # noise with one non-finite or signed-zero sample set
        x = rng.uniform(-0.9, 0.9, (n, ch)).astype(np.float32)
        what = kind[len("special_"):]
        if what == "inf_last_channel":
            x[n // 6, ch - 1] = np.inf
        elif what == "nan_last_channel":
            x[n // 6, ch - 1] = np.nan
        elif what == "nan_first_channel":
            x[n // 3, 0] = np.nan
        elif what == "inf_first_channel":
            x[n // 3, 0] = np.inf
        elif what == "opposite_infs":
            x[n // 6, 0], x[n // 6, 1] = np.inf, -np.inf
        elif what == "negative_zero":
            x[:, 0] = -0.0
            x[:, 1] = 0.0
        else:
            raise SystemExit(f"unknown special {what}")
        return x
    if kind == "tiny":  # subnormal f32 values
        return (rng.uniform(-1, 1, (n, ch)) * 1e-40).astype(np.float32)
    if kind == "square":
        t = np.arange(n)[:, None]
        return np.where((t // 19 + np.arange(ch)[None, :]) % 2 == 0, 1.0, -1.0).astype(np.float32)
    if kind == "dc":
        return np.full((n, ch), 0.3, np.float32)
    if kind == "step":
        x = np.zeros((n, ch), np.float32)
        x[n // 3:, :] = 0.5
        return x
    if kind == "sweep":
        t = np.arange(n) / 44100.0
        dur = max(n / 44100.0, 1e-9)
        k = np.log(0.45 * 22050 / 60.0)
        s = 0.6 * np.sin(2 * np.pi * 60.0 * dur / k * (np.exp(k * t / dur) - 1.0))
        return np.stack([np.roll(s, 7 * c) for c in range(ch)], axis=1).astype(np.float32)
    raise SystemExit(f"unknown signal {kind}")


def taps_for(rin, rout):
    import math

    f = min(1.0, 0.97 * rout / rin)
    t = math.ceil(32 / f)
    return t + (t & 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))
    jobs = []  # (name, kind, n, channels, in_rate, out_rate, seed)

    def case(name, kind, n, ch, rin, rout, seed):
        jobs.append((name, kind, n, ch, rin, rout, seed))

    seed = 100
    for rate in (8000, 11025, 12345, 16000, 22050, 32000, 44100, 44101, 48000, 96000):
        seed += 1
        case(f"rate_{rate}_to_16000", "noise", 2200, 1, rate, TARGET, seed)
    for rin, rout in ((44100, 48000), (16000, 48000), (22050, 44100), (48000, 44100), (97000, 50000),
                      (97000, 32000), (192000, 8000), (7999, 16000), (16000, 15999), (1000, 16000),
                      (16000, 1000), (44100, 44100), (3, 16000)):
        seed += 1
        case(f"rate_{rin}_to_{rout}", "noise", 2200 if rin >= 1000 else 40, 1, rin, rout, seed)
    for kind in ("sweep", "step", "square", "dc", "tiny", "grid"):
        for rin in (48000, 44100, 8000, 44101):
            seed += 1
            case(f"signal_{kind}_{rin}", kind, 2500, 1, rin, TARGET, seed)
    for ch in range(1, 9):
        for rin in (48000, 22050, 44101, 16000, 8000):
            seed += 1
            case(f"layout_c{ch}_{rin}", "noise", 700, ch, rin, TARGET, seed)

    # Non-finite samples and signed zeros, with and without a resampler; the last channel of 3 and
    # of 6 channels is the one the default mix gives no weight.
    for what, ch, rin in (("inf_last_channel", 3, 22050), ("nan_last_channel", 3, 22050),
                          ("inf_last_channel", 6, 16000), ("nan_first_channel", 1, 48000),
                          ("inf_first_channel", 1, 48000), ("opposite_infs", 2, 16000),
                          ("nan_first_channel", 2, 16000), ("negative_zero", 2, 16000),
                          ("negative_zero", 2, 44100)):
        seed += 1
        case(f"special_{what}_c{ch}_{rin}", f"special_{what}", 600, ch, rin, TARGET, seed)

    # Impulse trains: each output sees at most one impulse.
    trains = {}
    for rin, rout in ((44100, 16000), (22050, 16000), (11025, 16000), (8000, 16000), (44100, 48000),
                      (44101, 16000), (12345, 16000), (48000, 16000), (96000, 16000)):
        taps = taps_for(rin, rout)
        pos, ks = 600, []
        for k in range(220):
            ks.append(pos)
            pos += taps + 6 + (k * 7) % 11
        x = np.zeros((pos + 400, 1), np.float32)
        for k in ks:
            x[k, 0] = 0.75
        trains[f"train_{rin}_to_{rout}"] = (x, rin, rout)

    # Length sweeps.
    lens = {}
    for rin, rout in ((48000, 16000), (44100, 16000), (22050, 16000), (96000, 16000), (8000, 16000),
                      (44101, 16000), (12345, 16000), (16000, 48000), (44100, 48000), (11025, 16000),
                      (32000, 16000), (16000, 16000)):
        taps = taps_for(rin, rout)
        ns = list(range(0, max(141, taps * 3 // 2 + 11))) + [200, 257, 300, 513, 1000, 1023, 1024, 1025, 2049]
        lens[f"lens_{rin}_{rout}"] = (rin, rout, ns, signal("noise", max(ns), 1, 7000 + rin % 977))

    with tempfile.TemporaryDirectory() as scratch, ThreadPoolExecutor(max_workers=12) as pool:
        def run_case(j):
            name, kind, n, ch, rin, rout, sd = j
            x = signal(kind, n, ch, sd)
            y = ffmpeg_convert(scratch, name, x, rin, rout)
            return name, x, y, ch, rin, rout, kind

        rows = []
        for name, x, y, ch, rin, rout, kind in pool.map(run_case, jobs):
            x.astype("<f4").tofile(path(name + ".in.f32"))
            y.astype("<f4").tofile(path(name + ".ffmpeg.f32"))
            note(name + ".in.f32", "input", f"{kind}, {len(x)} frames of {ch} channel(s) at {rin} Hz, raw f32")
            note(name + ".ffmpeg.f32", "golden", f"ffmpeg -ac 1 -ar {rout}, default resampler options, raw f32")
            rows.append(f"{name},{ch},{rin},{rout}")

        def run_train(item):
            name, (x, rin, rout) = item
            return name, x, ffmpeg_convert(scratch, name, x, rin, rout), rin, rout

        for name, x, y, rin, rout in pool.map(run_train, trains.items()):
            x.astype("<f4").tofile(path(name + ".in.f32"))
            y.astype("<f4").tofile(path(name + ".ffmpeg.f32"))
            note(name + ".in.f32", "input", f"220 impulses of 0.75, spaced beyond the filter length, at {rin} Hz")
            note(name + ".ffmpeg.f32", "golden", f"ffmpeg -ac 1 -ar {rout} of the impulse train: filter coefficients")
            rows.append(f"{name},1,{rin},{rout}")

        def run_len(item):
            name, (rin, rout, ns, base) = item
            outs = [ffmpeg_convert(scratch, f"{name}_{n}", base[:n], rin, rout) for n in ns]
            return name, rin, rout, ns, base, outs

        with ThreadPoolExecutor(max_workers=12) as inner:
            for name, rin, rout, ns, base, outs in inner.map(run_len, lens.items()):
                base.astype("<f4").tofile(path(name + ".in.f32"))
                np.concatenate(outs).astype("<f4").tofile(path(name + ".ffmpeg.f32"))
                with open(path(name + ".lens.csv"), "w") as f:
                    f.write("n,output_samples\n")
                    for n, o in zip(ns, outs):
                        f.write(f"{n},{len(o)}\n")
                note(name + ".in.f32", "input", f"{len(base)} noise samples at {rin} Hz; length n runs the first n")
                note(name + ".ffmpeg.f32", "golden", f"ffmpeg -ac 1 -ar {rout} for each n of the .lens.csv, back to back")
                note(name + ".lens.csv", "config", "the lengths run and the number of samples ffmpeg made for each")

    with open(path("cases.csv"), "w") as f:
        f.write("name,channels,in_rate,out_rate\n")
        f.write("\n".join(sorted(rows)) + "\n")
    note("cases.csv", "config", "name,channels,in_rate,out_rate of every .in.f32/.ffmpeg.f32 pair that is not a lens sweep")
    fm.write_manifest(
        OUT,
        GENERATOR,
        "generated: numpy default_rng draws and closed-form signals written as raw f32; the goldens are the "
        "`ffmpeg` binary's output for those exact samples (see the generator docstring)",
        files,
        "produced",
        extra={"ffmpeg": subprocess.run([FFMPEG, "-version"], capture_output=True, text=True).stdout.splitlines()[0]},
    )
    total = sum(os.path.getsize(path(f)) for f in files)
    print(f"wrote {len(files)} files, {total / 1e6:.2f} MB, to {OUT}")


if __name__ == "__main__":
    main()
