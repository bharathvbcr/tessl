#!/usr/bin/env python3
"""The transformers lane that `bench_paired` alternates with `bench_embedgemma2_pre`.

Times what `AutoProcessor.from_pretrained(google/embeddinggemma-2)` does with the
same raw inputs as tessl's native preprocessing (src/embedgemma2_pre.rs), on the
CPU: the torchvision `Gemma4ImageProcessor` on a PIL image, the
`Gemma4AudioFeatureExtractor` on a float32 clip (the batched form; a bare 1-D
array hits a squeeze bug in transformers 5.19), and the
`EmbeddingGemma2VideoProcessor` on decoded frames with a `VideoMetadata` carrying
`fps` and `duration`. Prints one JSON array like the Rust lane's.

    ~/.venvs/ml-2026-10/bin/python bench/embedgemma2_pre_hf.py
    BENCH_WORKLOADS=image_640x480,audio_10s BENCH_ITERS=30 \\
        ~/.venvs/ml-2026-10/bin/python bench/embedgemma2_pre_hf.py

Environment, as for the Rust lane: BENCH_WORKLOADS (default all), BENCH_ITERS
(default 15), BENCH_WARMUP (default 3), and EMBEDGEMMA2_SNAPSHOT for the
snapshot (default: the newest in the Hugging Face cache).

The inputs are the Rust lane's, bit for bit: integer formulas (src/bin/
bench_embedgemma2_pre.rs: `pixel` and `sample`), and each row carries the CRC-32
of the input bytes (`input_crc`), so a comparison can check both lanes timed the
same data. Not timed: input construction (the PIL image, the numpy arrays).
transformers uses torch's intra-op threads (`threads` is
`torch.get_num_threads()`); tessl uses at most 8 of its own.
"""
import glob
import json
import os
import time
import zlib

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata

SAMPLE_RATE = 16000
VIDEO_FRAMES = 40
IMAGES = [(224, 224), (640, 480), (1920, 1080), (4000, 3000)]
AUDIO_SECONDS = [1, 10, 30]
VIDEOS = [(640, 360), (1280, 720)]


def snapshot():
    if os.environ.get("EMBEDGEMMA2_SNAPSHOT"):
        return os.environ["EMBEDGEMMA2_SNAPSHOT"]
    hits = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/*/config.json")))
    if not hits:
        raise SystemExit("no google/embeddinggemma-2 snapshot; set EMBEDGEMMA2_SNAPSHOT")
    return os.path.dirname(hits[-1])


def frame(f, w, h):
    """bench_embedgemma2_pre's `pixel`: ((y*7 + x*13 + c*29 + ((x*y) >> 4) + f*11) & 255)."""
    y = np.arange(h, dtype=np.int64)[:, None, None]
    x = np.arange(w, dtype=np.int64)[None, :, None]
    c = np.arange(3, dtype=np.int64)[None, None, :]
    return ((y * 7 + x * 13 + c * 29 + ((x * y) >> 4) + f * 11) & 255).astype(np.uint8)


def samples(n):
    """bench_embedgemma2_pre's `sample`: a 24-bit hash of the index scaled by 2^-25 (exact in f32)."""
    i = np.arange(n, dtype=np.uint64)
    h = ((i * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)) >> np.uint64(8)
    return ((h.astype(np.int64) - (1 << 23)).astype(np.float64) * (1.0 / 33554432.0)).astype(np.float32)


def count(name, default, minimum):
    v = os.environ.get(name)
    if v is None:
        return default
    n = int(v)
    if n < minimum:
        raise SystemExit(f"{name}={n}: at least {minimum} expected")
    return n


def timed(warmup, iters, run):
    for _ in range(warmup):
        run()
    ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run()
        ms.append((time.perf_counter() - t0) * 1e3)
    return sorted(ms)


def crc(*arrays):
    c = 0
    for a in arrays:
        c = zlib.crc32(np.ascontiguousarray(a).tobytes(), c)
    return format(c & 0xFFFFFFFF, "08x")


def main():
    iters, warmup = count("BENCH_ITERS", 15, 1), count("BENCH_WARMUP", 3, 0)
    names = ([f"image_{w}x{h}" for w, h in IMAGES] + [f"audio_{s}s" for s in AUDIO_SECONDS]
             + [f"video_{VIDEO_FRAMES}f_{w}x{h}" for w, h in VIDEOS])
    wanted = [w.strip() for w in os.environ.get("BENCH_WORKLOADS", ",".join(names)).split(",") if w.strip()]
    bad = [w for w in wanted if w not in names]
    if bad:
        raise SystemExit(f"BENCH_WORKLOADS: unknown workload {bad[0]!r}; known: {','.join(names)}")
    proc = AutoProcessor.from_pretrained(snapshot())
    assert type(proc.image_processor).__name__ == "Gemma4ImageProcessor"
    threads = torch.get_num_threads()

    rows = []
    for name in wanted:
        if name.startswith("image_"):
            w, h = map(int, name[len("image_"):].split("x"))
            arr = frame(0, w, h)
            img = Image.fromarray(arr, "RGB")
            ms = timed(warmup, iters, lambda: proc.image_processor(img, return_tensors="np"))
            units, unit, c = 1, "image", crc(arr)
        elif name.startswith("audio_"):
            seconds = int(name[len("audio_"):-1])
            pcm = samples(seconds * SAMPLE_RATE)
            ms = timed(warmup, iters, lambda: proc.feature_extractor([pcm], sampling_rate=SAMPLE_RATE))
            units, unit, c = seconds, "audio_s", crc(pcm.astype("<f4"))
        else:
            w, h = map(int, name[len(f"video_{VIDEO_FRAMES}f_"):].split("x"))
            frames = np.stack([frame(f, w, h) for f in range(VIDEO_FRAMES)])
            meta = {"total_num_frames": VIDEO_FRAMES, "fps": 1.0, "duration": float(VIDEO_FRAMES),
                    "height": h, "width": w}

            def run():
                return proc.video_processor(videos=[frames], video_metadata=[meta], return_tensors="np")

            units = int(run()["num_frames_per_video"][0])
            ms = timed(warmup, iters, run)
            unit, c = "frame", crc(frames)
        rows.append({"workload": name, "ms_min": round(ms[0], 4), "ms_median": round(ms[len(ms) // 2], 4),
                     "iters": iters, "units": units, "unit": unit, "threads": threads, "input_crc": c})
        print(f"{name:<22} min {ms[0]:>9.3f} ms  median {ms[len(ms) // 2]:>9.3f} ms  "
              f"per {unit} {ms[0] / units:.3f} ms", file=__import__("sys").stderr)
    print(json.dumps(rows, separators=(",", ":")))


if __name__ == "__main__":
    main()
