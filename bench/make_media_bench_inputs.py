#!/usr/bin/env python3
"""Inputs for `bench_media_decode` (src/bin/bench_media_decode.rs).

    ~/.venvs/ml-2026-10/bin/python bench/make_media_bench_inputs.py [OUT_DIR]

Writes into OUT_DIR (default target/media_bench) deterministic files of realistic
size: a 1920x1080 picture as PNG, JPEG (4:2:0, quality 90) and TIFF; 30 s of
44.1 kHz stereo audio as WAV, FLAC, AAC (m4a) and MP3, and as 16 kHz mono WAV;
10 s of 1280x720 H.264 video at 30 fps (300 frames). The audio and video are
made with FFmpeg (`/opt/homebrew/bin/ffmpeg`), the picture with numpy and PIL.
"""

import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.io import wavfile

FFMPEG = "/opt/homebrew/bin/ffmpeg"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "media_bench")


def ff(*args):
    subprocess.run([FFMPEG, "-v", "error", "-y", *args], check=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    p = lambda n: os.path.join(OUT, n)  # noqa: E731
    rng = np.random.default_rng(0)
    h, w = 1080, 1920
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.stack([127.5 * (1 + np.sin(2 * math.pi * x / w * 3)), 255.0 * y / h,
                    127.5 * (1 + np.cos(2 * math.pi * (x + y) / (w + h) * 5))], axis=-1)
    img += rng.normal(0, 6.0, img.shape)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")
    im.save(p("photo_1080p.png"))
    im.save(p("photo_1080p.jpg"), quality=90, subsampling=2)
    im.save(p("photo_1080p.tiff"))

    sr, secs = 44100, 30
    t = np.arange(sr * secs) / sr
    left = 0.4 * np.sin(2 * math.pi * (200 + 40 * t) * t) + 0.2 * np.sin(2 * math.pi * 1250 * t) + 0.02 * rng.standard_normal(t.shape)
    right = 0.7 * left + 0.05 * np.sin(2 * math.pi * 330 * t)
    stereo = np.round(np.stack([left, right], axis=1) * 32767).astype(np.int16)
    wavfile.write(p("audio_30s_44k_stereo.wav"), sr, stereo)
    for ext, codec in (("flac", ["-c:a", "flac"]), ("m4a", ["-c:a", "aac", "-b:a", "128k"]),
                       ("mp3", ["-c:a", "libmp3lame", "-b:a", "128k"])):
        ff("-i", p("audio_30s_44k_stereo.wav"), *codec, p(f"audio_30s_44k_stereo.{ext}"))
    ff("-i", p("audio_30s_44k_stereo.wav"), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", p("audio_30s_16k_mono.wav"))

    ff("-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30", "-t", "10", "-pix_fmt", "yuv420p", "-c:v", "libx264",
       "-crf", "23", p("video_10s_720p_h264.mp4"))
    for n in sorted(os.listdir(OUT)):
        print(f"{n:32} {os.path.getsize(p(n)):>10} bytes")


if __name__ == "__main__":
    main()
