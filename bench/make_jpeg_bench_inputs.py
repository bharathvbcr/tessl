#!/usr/bin/env python3
"""Inputs for `bench_jpeg_decode` (src/bin/bench_jpeg_decode.rs).

    ~/.venvs/ml-2026-10/bin/python bench/make_jpeg_bench_inputs.py [OUT_DIR]

Writes into OUT_DIR (default target/jpeg_bench) deterministic JPEG files of one
picture (gradients, a disc and noise, the picture bench/make_media_bench_inputs.py
uses) at the sizes and codings the decoder meets: 1920x1080 as baseline 4:2:0,
4:2:2 and 4:4:4 at quality 90, as progressive 4:2:0 and as grayscale; 3840x2160
baseline 4:2:0; 224x224 baseline 4:2:0 (the size a vision tower sees). Encoded by
PIL 12.3 (libjpeg-turbo). Python is the input generator only; the timings come
from the Rust bench.
"""

import math
import os
import sys

import numpy as np
from PIL import Image

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "jpeg_bench")


def picture(h, w, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.stack([127.5 * (1 + np.sin(2 * math.pi * x / w * 3)), 255.0 * y / h,
                    127.5 * (1 + np.cos(2 * math.pi * (x + y) / (w + h) * 5))], axis=-1)
    img += rng.normal(0, 6.0, img.shape)
    disc = (x - 0.6 * w) ** 2 + (y - 0.4 * h) ** 2 < (0.2 * min(h, w)) ** 2
    img[disc] = [235.0, 40.0, 70.0]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def main():
    os.makedirs(OUT, exist_ok=True)
    p = lambda n: os.path.join(OUT, n)  # noqa: E731
    hd = picture(1080, 1920)
    hd.save(p("hd_420_baseline.jpg"), quality=90, subsampling=2)
    hd.save(p("hd_422_baseline.jpg"), quality=90, subsampling=1)
    hd.save(p("hd_444_baseline.jpg"), quality=90, subsampling=0)
    hd.save(p("hd_420_progressive.jpg"), quality=90, subsampling=2, progressive=True)
    hd.convert("L").save(p("hd_gray_baseline.jpg"), quality=90)
    picture(2160, 3840).save(p("uhd_420_baseline.jpg"), quality=90, subsampling=2)
    picture(224, 224).save(p("small_224_420_baseline.jpg"), quality=90, subsampling=2)
    for n in sorted(os.listdir(OUT)):
        print(f"{n:32} {os.path.getsize(p(n)):>10} bytes")


if __name__ == "__main__":
    main()
