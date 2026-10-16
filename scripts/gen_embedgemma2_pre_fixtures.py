#!/usr/bin/env python3
"""Generate the EmbeddingGemma 2 preprocessing fixtures in tests/fixtures/embedgemma2/ from transformers.

    ~/.venvs/ml-2026-10/bin/python scripts/gen_embedgemma2_pre_fixtures.py [SNAPSHOT_DIR]
    EMBEDGEMMA2_PRE_BIG_DIR=target/embedgemma2_pre_big \\
        ~/.venvs/ml-2026-10/bin/python scripts/gen_embedgemma2_pre_fixtures.py [SNAPSHOT_DIR]

The oracle of `src/embedgemma2_pre.rs` (tessl's native image, audio and video
preprocessing): the inputs and the outputs of the processors that
`AutoProcessor.from_pretrained(snapshot)` builds for google/embeddinggemma-2,
which is what sentence-transformers calls. SNAPSHOT_DIR defaults to the newest
google/embeddinggemma-2 snapshot of the Hugging Face cache (EMBEDGEMMA2_SNAPSHOT
overrides it). Needs torch, torchvision, numpy, pillow and transformers 5.19 or
later. Deterministic: every input is built from a fixed seed, and rerunning
rewrites identical files.

Which image path: `AutoProcessor` loads `Gemma4ImageProcessor`, the torchvision
("fast") processor, not `Gemma4ImageProcessorPil`; this script calls that
object. The video processor and the audio feature extractor are the snapshot's
own. Inputs go in as PIL images (so `do_convert_rgb` acts, as it only does on
PIL input), as a list of float32 arrays (the audio extractor's batched path; a
bare 1-D array hits a squeeze bug in transformers 5.19), and as decoded uint8
arrays with a `VideoMetadata` carrying `fps`, `duration = frames / fps` and
`total_num_frames` (without a duration the video processor silently skips fps
sampling).

Committed files, prefix `eg2p_`, kept small:

  processor_config.json    the snapshot's, which `PreConfig::from_json` reads
  index.txt                one line per case, `kind name key=value ...`; the
                           Rust test parses it
  img_<name>_in.npy        the raw image, int64 [H, W] / [H, W, C] bytes (tessl's
                           npy reader takes f32, f64 and i64)
  img_<name>_S_px.npy      the processor's pixel_values [P, patch_dim] f32 and
  img_<name>_S_pos.npy     image_position_ids [P, 2] int64, for the small geometry
                           S: patch 4, pooling 2, 70 soft tokens (280 patches of
                           48 values), which exercises the same arithmetic as the
                           real geometry on inputs of a few kilobytes
  D (the checkpoint's patch 16, pooling 3, 280 soft tokens) outputs are 7.7 MB per
  image, so the index holds their CRC-32 (zlib, over the C-order little-endian
  bytes) and shapes instead; `crc_px` / `crc_pos`
  aud_<name>_in.npy        the clip, f32
  aud_<name>_feat.npy      input_features [F, 128] f32 (absent when F = 0)
  aud_<name>_mask.npy      input_features_mask [F] int64
  vid_<name>_in.npy        decoded frames [T, H, W, 3] int64
  vid_<name>_px.npy        pixel_values_videos, vid_<name>_pos.npy video_position_ids
                           (geometry S), the sampled indices and timestamps in index.txt
  provenance.json          library versions

The index also carries a grid of `video_indices` lines: `sample_frames` (fps
sampling, then the uniform or truncating cap) over source frame counts and
frame rates, including fractional rates.

With EMBEDGEMMA2_PRE_BIG_DIR the big cases are written there instead of
committed: the D geometry's full arrays for every image, plus a 2000x1500 photo-
like image, a 768x768 image (already the exact target), wide and noisy images,
a 35 s clip (past the 30 s limit) and a 140-token video. The Rust tests that read
them are `#[ignore]` and take the directory in `EMBEDGEMMA2_PRE_BIG_DIR`.
"""

import glob
import importlib.metadata
import json
import math
import os
import shutil
import sys
import zlib

import numpy as np
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures", "embedgemma2")
BIG = os.environ.get("EMBEDGEMMA2_PRE_BIG_DIR")
SR = 16000
S_GEOMETRY = dict(patch_size=4, pooling_kernel_size=2, max_soft_tokens=70)
S_VIDEO = dict(patch_size=4, pooling_kernel_size=2, max_soft_tokens=70)


def snapshot():
    if len(sys.argv) > 1:
        return sys.argv[1]
    if os.environ.get("EMBEDGEMMA2_SNAPSHOT"):
        return os.environ["EMBEDGEMMA2_SNAPSHOT"]
    hits = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/*/config.json")))
    if not hits:
        sys.exit("no google/embeddinggemma-2 snapshot; pass its directory")
    return os.path.dirname(hits[-1])


def save(path, arr):
    np.save(path, np.ascontiguousarray(arr))


def crc(arr):
    return format(zlib.crc32(np.ascontiguousarray(arr).tobytes()) & 0xFFFFFFFF, "08x")


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def photo(h, w, seed, noise=4.0):
    """Smooth gradients, a disc and a checker patch plus noise: low and high frequencies."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 127.5 * (1 + np.sin(2 * math.pi * x / max(w, 1)))
    g = 255.0 * y / max(h - 1, 1)
    b = 127.5 * (1 + np.cos(2 * math.pi * (x + y) / (w + h)))
    img = np.stack([r, g, b], axis=-1)
    cx, cy, rad = 0.6 * w, 0.4 * h, 0.2 * min(w, h)
    img[(x - cx) ** 2 + (y - cy) ** 2 < rad**2] = [240.0, 30.0, 60.0]
    img[((x // 8 + y // 8) % 2 == 0) & (x < w // 4) & (y > 3 * h // 4)] = [10.0, 10.0, 10.0]
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def noise_image(h, w, c, seed):
    return np.random.default_rng(seed).integers(0, 256, (h, w, c), dtype=np.uint8)


def pil(arr, mode):
    """A PIL image of `mode` from an array [H, W] (L) or [H, W, C]."""
    return Image.fromarray(arr, mode)


# (name, array, PIL mode); committed, so a few kilobytes each.
IMAGES = [
    ("one", noise_image(1, 1, 3, 1), "RGB"),
    ("wide1x2000", noise_image(1, 2000, 3, 2), "RGB"),
    ("tall2000x1", noise_image(2000, 1, 3, 3), "RGB"),
    ("exact56x80", photo(56, 80, 4), "RGB"),
    ("exact48", photo(48, 48, 5), "RGB"),
    ("gray37x53", noise_image(37, 53, 1, 6)[:, :, 0], "L"),
    ("la20x30", noise_image(20, 30, 2, 7), "LA"),
    ("rgba41x29", noise_image(41, 29, 4, 8), "RGBA"),
    ("photo100x60", photo(60, 100, 9), "RGB"),
    ("noise31x17", noise_image(17, 31, 3, 10), "RGB"),
    ("tall90x7", noise_image(90, 7, 3, 11), "RGB"),
    ("down100x75", photo(75, 100, 12), "RGB"),
    ("up9x13", photo(13, 9, 13), "RGB"),
]
# Big cases (EMBEDGEMMA2_PRE_BIG_DIR only).
BIG_IMAGES = [
    ("photo2000x1500", photo(1500, 2000, 20, noise=6.0), "RGB"),
    ("exact768", photo(768, 768, 21), "RGB"),
    ("wide3000x200", noise_image(200, 3000, 3, 22), "RGB"),
    ("noise1024", noise_image(1024, 1024, 3, 23), "RGB"),
    ("rgba640x360", noise_image(360, 640, 4, 24), "RGBA"),
]


def image_case(proc, name, arr, mode, geom, out_dir, big_only=False):
    """Run the processor on one image; write the arrays and return the index line."""
    kwargs = S_GEOMETRY if geom == "S" else {}
    img = pil(arr, mode)
    res = proc.image_processor(img, return_tensors="np", **kwargs)
    px = res["pixel_values"][0].astype(np.float32)
    pos = res["image_position_ids"][0].astype(np.int64)
    soft = int(res["num_soft_tokens_per_image"][0])
    patches = int((pos[:, 0] >= 0).sum())
    assert np.all(pos[patches:] == -1) and np.all(px[patches:] == 0)
    p = kwargs.get("patch_size", 16)
    resized = None
    # The resized size, from the real patches' extent.
    rows, cols = int(pos[:patches, 1].max()) + 1, int(pos[:patches, 0].max()) + 1
    assert rows * cols == patches
    resized = f"{rows * p}x{cols * p}"
    h, w = arr.shape[:2]
    c = 1 if arr.ndim == 2 else arr.shape[2]
    line = (f"image {name} {geom} channels={c} h={h} w={w} resized={resized} patches={patches} soft={soft} "
            f"crc_px={crc(px)} crc_pos={crc(pos)}")
    if geom == "S" and not big_only:
        save(os.path.join(out_dir, f"eg2p_img_{name}_S_px.npy"), px)
        save(os.path.join(out_dir, f"eg2p_img_{name}_S_pos.npy"), pos)
    if BIG and geom == "D":
        save(os.path.join(BIG, f"eg2p_img_{name}_D_px.npy"), px)
        save(os.path.join(BIG, f"eg2p_img_{name}_D_pos.npy"), pos)
    return line


def images(proc, lines):
    for name, arr, mode in IMAGES:
        save(os.path.join(OUT, f"eg2p_img_{name}_in.npy"), arr.astype(np.int64))
        for geom in ("S", "D"):
            lines.append(image_case(proc, name, arr, mode, geom, OUT))
    if BIG:
        # The committed inputs' D arrays went to BIG above (image_case); the big
        # inputs carry their input and D arrays there, indexed in index_big.txt.
        for name, arr, mode in BIG_IMAGES:
            save(os.path.join(BIG, f"eg2p_img_{name}_in.npy"), arr.astype(np.int64))
            with open(os.path.join(BIG, "index_big.txt"), "a") as f:
                f.write(image_case(proc, name, arr, mode, "D", BIG, big_only=True) + "\n")
    # Once through the whole processor, as sentence-transformers calls it: the same pixels.
    full = proc(images=[pil(IMAGES[8][1], IMAGES[8][2])], text=None)
    direct = proc.image_processor(pil(IMAGES[8][1], IMAGES[8][2]), return_tensors="np")
    assert np.array_equal(np.asarray(full["pixel_values"]), direct["pixel_values"])
    assert np.array_equal(np.asarray(full["image_position_ids"]), direct["image_position_ids"])


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def chirp(n, f0, f1, amp):
    t = np.arange(n) / SR
    k = (f1 - f0) / (n / SR)
    return (amp * np.sin(2 * math.pi * (f0 * t + 0.5 * k * t**2))).astype(np.float32)


def noise(n, seed, amp=0.1):
    return (amp * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)


def tones(n, amp):
    t = np.arange(n) / SR
    return (amp * (np.sin(2 * math.pi * 440 * t) + 0.5 * np.sin(2 * math.pi * 3300 * t + 0.4))).astype(np.float32)


# (name, clip, max_length or None)
CLIPS = [
    ("silence1000", np.zeros(1000, np.float32), None),
    ("sweep1s", chirp(16000, 100, 7000, 0.5), None),
    ("noise0p5s", noise(8000, 30), None),
    ("short100", noise(100, 31), None),
    ("pad150", noise(150, 32), None),
    ("one161", noise(161, 33), None),
    ("one320", noise(320, 34), None),
    ("odd4097", noise(4097, 35), None),
    ("odd8191", chirp(8191, 300, 4000, 0.3), None),
    ("loud", tones(4800, 8.0), None),
    ("dc", np.full(4000, 0.3, np.float32), None),
    ("nyquist", np.where(np.arange(4000) % 2 == 0, 0.9, -0.9).astype(np.float32), None),
    ("quiet", noise(3000, 36, amp=1e-5), None),
    ("trunc6000_max4000", noise(6000, 37), 4000),
    ("mixed3s", chirp(48000, 200, 5000, 0.4) + noise(48000, 38, 0.02), None),
]
BIG_CLIPS = [
    ("long35s", noise(35 * SR, 40, 0.1) + chirp(35 * SR, 100, 7500, 0.3), None),
    ("max30s", noise(30 * SR, 41, 0.1), None),
]


def audio_case(proc, name, clip, max_length, out_dir):
    fe = proc.feature_extractor
    kwargs = {} if max_length is None else {"max_length": max_length}
    res = fe([clip], sampling_rate=SR, **kwargs)
    feat = np.asarray(res["input_features"][0], dtype=np.float32)
    mask = np.asarray(res["input_features_mask"][0]).astype(np.int64)
    frames = int(mask.shape[0])
    assert feat.shape == (frames, 128)
    save(os.path.join(out_dir, f"eg2p_aud_{name}_in.npy"), clip)
    if frames:
        save(os.path.join(out_dir, f"eg2p_aud_{name}_feat.npy"), feat)
    save(os.path.join(out_dir, f"eg2p_aud_{name}_mask.npy"), mask)
    return (f"audio {name} samples={clip.shape[0]} max_length={max_length or 0} frames={frames} "
            f"valid={int(mask.sum())}")


def audio(proc, lines):
    for name, clip, max_length in CLIPS:
        lines.append(audio_case(proc, name, clip, max_length, OUT))
    # The whole processor on a clip: the same features.
    clip = CLIPS[1][1]
    full = proc(audio=[clip])
    direct = proc.feature_extractor([clip], sampling_rate=SR)
    assert np.array_equal(np.asarray(full["input_features"][0]), np.asarray(direct["input_features"][0]))
    if BIG:
        for name, clip, max_length in BIG_CLIPS:
            line = audio_case(proc, name, clip, max_length, BIG)
            with open(os.path.join(BIG, "index_big.txt"), "a") as f:
                f.write(line + "\n")


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

# Source frame rates as (numerator, denominator): NTSC rates are fractional.
FPS_GRID = [(1, 1), (1, 2), (1, 10), (1, 4), (2, 1), (5, 1), (10, 1), (12, 1), (15, 1), (24000, 1001), (24, 1),
            (25, 1), (30000, 1001), (30, 1), (48, 1), (50, 1), (60000, 1001), (60, 1), (120, 1), (3, 2), (7, 3)]
COUNT_GRID = [1, 2, 3, 5, 24, 29, 30, 31, 32, 33, 34, 59, 60, 61, 63, 64, 65, 90, 95, 100, 127, 128, 129, 200,
              299, 300, 301, 500, 719, 720, 1000, 1799, 1800, 2500, 3600, 5000]


def video_indices(proc, lines):
    vp = proc.video_processor
    from transformers.video_utils import VideoMetadata

    def run(total, num, den, **kw):
        fps = num / den
        meta = VideoMetadata(total_num_frames=total, fps=fps, duration=total / fps)
        return [int(i) for i in vp.sample_frames(meta, **kw)]

    for total in COUNT_GRID:
        for num, den in FPS_GRID:
            idx = run(total, num, den, fps=1, max_frames=32, overflow_strategy="uniform")
            lines.append(f"video_indices n={total} fps={num}/{den} mode=uniform idx={','.join(map(str, idx))}")
    for total in (31, 33, 100, 1000, 3600):
        for num, den in ((24, 1), (30000, 1001), (1, 2)):
            idx = run(total, num, den, fps=1, max_frames=32, overflow_strategy="truncate")
            lines.append(f"video_indices n={total} fps={num}/{den} mode=truncate idx={','.join(map(str, idx))}")
    # fps sampling off in the config's sense: every frame, then the cap.
    for total in (7, 32, 33, 100, 1000):
        meta = VideoMetadata(total_num_frames=total, fps=25.0, duration=total / 25.0)
        idx = [int(i) for i in vp.sample_frames(meta, fps=None, max_frames=32, overflow_strategy="uniform")]
        lines.append(f"video_indices n={total} fps=25/1 mode=nofps idx={','.join(map(str, idx))}")
    # max_frames=1 under uniform: np.linspace(0, n - 1, 1) is [0].
    for total in (5, 100):
        idx = run(total, 24, 1, fps=1, max_frames=1, overflow_strategy="uniform")
        lines.append(f"video_indices n={total} fps=24/1 mode=uniform1 idx={','.join(map(str, idx))}")


def make_video(t, h, w, seed):
    """Frames that differ: a photo with a moving disc and per-frame noise."""
    rng = np.random.default_rng(seed)
    base = photo(h, w, seed).astype(np.float64)
    frames = []
    for f in range(t):
        fr = base.copy()
        cx = (f * 3) % w
        yy, xx = np.mgrid[0:h, 0:w]
        fr[(xx - cx) ** 2 + (yy - h // 2) ** 2 < (min(h, w) // 4) ** 2] = [250.0, 250.0, 20.0]
        fr += rng.normal(0, 3.0, fr.shape)
        frames.append(np.clip(fr, 0, 255).astype(np.uint8))
    return np.stack(frames)


# (name, frames, source fps (num, den), geometry)
VIDEOS = [
    ("v9_3fps_S", make_video(9, 30, 40, 50), (3, 1), "S"),
    ("v40_1fps_S", make_video(40, 12, 16, 51), (1, 1), "S"),      # 40 sampled, capped to 32
    ("v70_ntsc_S", make_video(70, 20, 12, 52), (24000, 1001), "S"),
    ("v33_1fps_S", make_video(33, 9, 14, 53), (1, 1), "S"),       # 33: one over the cap
    ("v32_1fps_S", make_video(32, 9, 14, 54), (1, 1), "S"),       # exactly the cap
    ("v5_2fps_D", make_video(5, 36, 48, 55), (2, 1), "D"),
    ("v1_D", make_video(1, 20, 30, 56), (24, 1), "D"),
]
BIG_VIDEOS = [
    ("v100_30fps_D", make_video(100, 90, 160, 60), (30, 1), "D"),
    ("v600_25fps_D", make_video(600, 24, 32, 61), (25, 1), "D"),
]


def video_case(proc, name, frames, fps_frac, geom, out_dir, big):
    vp = proc.video_processor
    num, den = fps_frac
    fps = num / den
    t, h, w, _ = frames.shape
    kwargs = S_VIDEO if geom == "S" else {}
    meta = {"total_num_frames": t, "fps": fps, "duration": t / fps, "height": h, "width": w}
    res = vp(videos=[frames], video_metadata=[meta], return_tensors="np", return_metadata=True, **kwargs)
    px = np.asarray(res["pixel_values_videos"], dtype=np.float32)
    pos = np.asarray(res["video_position_ids"]).astype(np.int64)
    md = res["video_metadata"][0]
    indices = [int(i) for i in md["frames_indices"]]
    stamps = [float(v) for v in md.timestamps]
    nsoft = int(res["num_soft_tokens_per_video"][0])
    nframes = int(res["num_frames_per_video"][0])
    assert nframes == len(indices) == px.shape[0]
    patches = int((pos[0, :, 0] >= 0).sum())
    save(os.path.join(out_dir, f"eg2p_vid_{name}_in.npy"), frames.astype(np.int64))
    if geom == "S" and not big and nframes <= 4:   # a 32-frame S video is 1.7 MB: its CRC only
        save(os.path.join(out_dir, f"eg2p_vid_{name}_px.npy"), px)
        save(os.path.join(out_dir, f"eg2p_vid_{name}_pos.npy"), pos)
    if geom == "D" and big:
        save(os.path.join(out_dir, f"eg2p_vid_{name}_px.npy"), px)
        save(os.path.join(out_dir, f"eg2p_vid_{name}_pos.npy"), pos)
    return (f"video {name} {geom} fps={num}/{den} frames={t} h={h} w={w} sampled={nframes} soft={nsoft} "
            f"patches={patches} indices={','.join(map(str, indices))} "
            f"stamps={','.join(repr(s) for s in stamps)} crc_px={crc(px)} crc_pos={crc(pos)}")


def video(proc, lines):
    video_indices(proc, lines)
    for name, frames, fps_frac, geom in VIDEOS:
        lines.append(video_case(proc, name, frames, fps_frac, geom, OUT, big=False))
    if BIG:
        for name, frames, fps_frac, geom in BIG_VIDEOS:
            line = video_case(proc, name, frames, fps_frac, geom, BIG, big=True)
            with open(os.path.join(BIG, "index_big.txt"), "a") as f:
                f.write(line + "\n")


def main():
    path = snapshot()
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(path)
    assert type(proc.image_processor).__name__ == "Gemma4ImageProcessor", type(proc.image_processor)
    os.makedirs(OUT, exist_ok=True)
    if BIG:
        os.makedirs(BIG, exist_ok=True)
        open(os.path.join(BIG, "index_big.txt"), "w").close()
    shutil.copyfile(os.path.join(path, "processor_config.json"), os.path.join(OUT, "eg2p_processor_config.json"))
    lines = []
    images(proc, lines)
    audio(proc, lines)
    video(proc, lines)
    with open(os.path.join(OUT, "eg2p_index.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "eg2p_provenance.json"), "w") as f:
        json.dump({"versions": {p: importlib.metadata.version(p)
                                for p in ("torch", "torchvision", "numpy", "pillow", "transformers")},
                   "image_processor": type(proc.image_processor).__name__,
                   "generator": "scripts/gen_embedgemma2_pre_fixtures.py"}, f, indent=1)
        f.write("\n")
    print(f"{len(lines)} index lines; fixtures in {os.path.abspath(OUT)}" + (f"; big cases in {BIG}" if BIG else ""))


if __name__ == "__main__":
    main()
