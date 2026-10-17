#!/usr/bin/env python3
"""How far libswresample's 16 kHz mono decode (what tessl::media reproduces to the
bit) is from the other loaders `transformers.audio_utils.load_audio` can use.

    ~/.venvs/voice-mac/bin/python tools/embedgemma2_ref/audio_loader_gap.py   # librosa 0.11 + soxr 1.1
    ~/.venvs/ml-2026-10/bin/python tools/embedgemma2_ref/audio_loader_gap.py  # torchaudio

Each interpreter runs the loaders it has and says which it skipped. For each
fixture of tests/fixtures/media_resample/ and four of tests/fixtures/media/ (16-bit WAV, whose `.ffmpeg.f32` and
`.ffmpeg_mean.f32` goldens tessl matches sample for sample) it prints the SNR of

  librosa.load(path, sr=16000)       soundfile decode, channel mean, soxr_hq   [backend="librosa"]
  torchaudio.load + .mean(0) + torchaudio.functional.resample                 [backend="torchaudio"]

against the golden that has the same channel mix (`.ffmpeg_mean.f32` for stereo,
`.ffmpeg.f32` for mono), over the whole signal and over the interior (the first and
last 10 ms left out, where the converters mirror their edges differently). Neither
loader is installed in the environment the repository's tests run in, so this is a
measurement script, not a test.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "..", "tests", "fixtures")
# (directory, name). The sweeps of media_resample/ run to 0.45 of the input's Nyquist
# rate, so they carry energy above the 8 kHz output Nyquist rate, which each converter
# attenuates or folds differently: a hard case. The tones of media/ (the earlier
# fixtures, all below 5.4 kHz) are the band-limited one, as speech mostly is; their
# `.ffmpeg_mean.f32` goldens went through a 16-bit `pan` stage, which caps an SNR
# measured against them at about 87 dB.
NAMES = [
    ("media_resample", n)
    for n in (
        "sweep_8000_mono", "sweep_11025_mono", "sweep_22050_mono", "sweep_32000_mono",
        "sweep_44100_mono", "sweep_48000_mono", "sweep_96000_mono",
        "sweep_44100_stereo", "sweep_48000_stereo", "sweep_96000_stereo",
    )
] + [("media", n) for n in ("pcm16_8k_mono", "pcm16_22k_mono", "pcm16_44k_stereo", "pcm16_48k_stereo")]
EDGE = 160  # 10 ms at 16 kHz


def golden(d, name):
    base = os.path.join(FIXTURES, d, name)
    if d == "media":
        return np.fromfile(base + ".wav.ffmpeg_mean.f32", dtype="<f4")
    mean = base + ".ffmpeg_mean.f32"
    return np.fromfile(mean if os.path.exists(mean) else base + ".ffmpeg.f32", dtype="<f4")


def snr(ref, x):
    n = min(len(ref), len(x))
    ref = ref[:n].astype(np.float64)
    x = x[:n].astype(np.float64)
    return 10 * np.log10((ref**2).sum() / max(((ref - x) ** 2).sum(), 1e-300))


def report(label, loader):
    print(label)
    for d, name in NAMES:
        want = golden(d, name)
        got = np.asarray(loader(os.path.join(FIXTURES, d, name + ".wav")), dtype=np.float32)
        print(f"  {name:22} frames {len(got)} (ffmpeg {len(want)})  SNR whole {snr(want, got):6.1f} dB, "
              f"interior {snr(want[EDGE:-EDGE], got[EDGE:-EDGE]):6.1f} dB")


def main():
    try:
        import librosa

        report(f"librosa {librosa.__version__}, load(path, sr=16000) [soxr_hq] vs libswresample (channel mean for stereo)",
               lambda p: librosa.load(p, sr=16000)[0])
    except ImportError as e:
        print("librosa skipped:", e)
    try:
        import torch
        import torchaudio

        def ta(p):
            from scipy.io import wavfile

            sr, x = wavfile.read(p)
            w = torch.from_numpy(x.astype(np.float32) / 32768.0)
            w = w.mean(dim=1) if w.ndim == 2 else w
            return torchaudio.functional.resample(w, orig_freq=sr, new_freq=16000).numpy()

        report(f"torchaudio {torchaudio.__version__}, mean + functional.resample vs libswresample (channel mean for stereo)", ta)
    except ImportError as e:
        print("torchaudio skipped:", e)


if __name__ == "__main__":
    sys.exit(main())
