#!/usr/bin/env python3
"""Video files and sentence-transformers' embeddings of them, for tessl's native
video decoding (src/media/video.rs, tests/media_e2e_video.rs).

    EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
      ~/.venvs/ml-2026-10/bin/python tools/embedgemma2_ref/make_video_reference.py

Writes into target/media_e2e_ref/ (override: EMBEDGEMMA2_MEDIA_REF_DIR)
`video_{name}.emb_f32.npy` and `video_{name}.indices.npy` for each case below,
from sentence-transformers 6.x / transformers 5.x, fp32, sdpa, CPU, batch size 1.
The inputs are the committed fixtures of tests/fixtures/media_video/, so tessl
reads the very same files.

What the reference is given, and why it is what the reference loader produces
------------------------------------------------------------------------------
transformers' `fetch_videos` decodes a video path with torchcodec
(`VideoDecoder(seek_mode="exact")`), which is not installed here. The array
given to sentence-transformers is torchcodec's frames as its source (v0.17.0, BSD-3)
makes them, with the `ffmpeg` CLI as the libswscale: the stored picture converted to
rgb24 with the scaler flags unset (`swsFlags = 0`, or a `format=rgb24` filtergraph, which
sets none), then the display rotation applied to the RGB
(`gen.torchcodec_rgb24`; scripts/gen_media_video_exact_fixtures.py cites the files and
the probes that the graph, `sws_scale` flags 0 and the CLI agree). The metadata is torchcodec's
exact-mode figures (`num_frames` = packets, `duration` = last pts + duration -
first pts, `fps` = num_frames / duration), from the same script. The video
processor then samples frames from those with `sample_frames`, as it does for a
decoded path, and this script asserts the indices it kept equal the ones the
fixture's `facts.txt` records.

Cases (`name`: file, timestamps)
  e2e_video      e2e_video.mp4, 96x64, 80 frames at 24000/1001, BT.709 limited:
                 three frames sampled (0, 23, 47)
  e2e_video_ts   the same with `add_timestamps` on: each frame's block is
                 preceded by its `mm:ss` label
  rot90          h264_709_tv_64x48_rot90.mp4, a stored 64x48 turned by its
                 rotation matrix: the model sees 48x64. A 4:2:0 picture whose
                 stored width is a NEON width and upright width is not, so converting
                 before the turn (torchcodec) and after (the CLI's autorotate) differ
  hevc_pc        hevc_601_pc_34x26.mp4, HEVC, BT.601 full range, a width off
                 libswscale's NEON widths
  hevc_main10    hevc_p10_601_pc_66x38.mp4, HEVC Main 10 (yuv420p10le), BT.601 full
                 range: the 10-bit planes VideoToolbox decodes, converted with the
                 bicubic chroma filter that unset scaler flags give
  odd_444_10bit  hevc_p410_240m_tv_33x17.mp4, HEVC 4:4:4 10-bit at an odd 33x17,
                 SMPTE 240M: odd widths and heights exist only in 4:4:4 (and odd
                 heights in 4:2:2)
  odd_422        hevc_422_709_tv_64x33.mp4, HEVC 4:2:2 at an odd height of 33
"""

import os
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers.video_utils import VideoMetadata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import gen_media_video_exact_fixtures as gen  # noqa: E402
from make_mm_reference import set_timestamps  # noqa: E402
from make_reference import snapshot  # noqa: E402

OUT = os.environ.get("EMBEDGEMMA2_MEDIA_REF_DIR", os.path.join(ROOT, "target", "media_e2e_ref"))
FIXTURES = os.path.join(ROOT, "tests", "fixtures", "media_video")
CASES = [  # (name, fixture, add_timestamps)
    ("e2e_video", "e2e_video", False),
    ("e2e_video_ts", "e2e_video", True),
    ("rot90", "h264_709_tv_64x48_rot90", False),
    ("hevc_pc", "hevc_601_pc_34x26", False),
    ("hevc_main10", "hevc_p10_601_pc_66x38", False),
    ("odd_444_10bit", "hevc_p410_240m_tv_33x17", False),
    ("odd_422", "hevc_422_709_tv_64x33", False),
]


def facts_line(name):
    with open(os.path.join(FIXTURES, "facts.txt")) as f:
        for line in f:
            parts = line.split()
            if parts[:2] == ["video", name]:
                return dict(p.split("=", 1) for p in parts[2:])
    sys.exit(f"{name} is not in facts.txt")


def reference_item(fixture):
    """The array and the metadata the reference loader would hand the processor for this file."""
    path = os.path.join(FIXTURES, fixture + ".mp4")
    facts = facts_line(fixture)
    frames, duration, fps, _, _, _ = gen.torchcodec_exact_metadata(path)
    assert frames == int(facts["frames"]) and duration == float(facts["duration"]), (fixture, facts)
    w, h, rot = int(facts["w"]), int(facts["h"]), int(facts["rotation"])
    if rot in (90, 270):
        w, h = h, w
    # torchcodec's frames: the stored picture converted by libswscale with no scaler flags (the
    # CLI's default; 10-bit 4:2:0 is the one layout whose RGB depends on them), then the
    # display rotation applied to the RGB (the generator's docstring cites torchcodec's source).
    raw = np.frombuffer(
        gen.torchcodec_rgb24(path, int(facts["w"]), int(facts["h"]), frames, rot, "" if facts["flip"] == "none" else facts["flip"]),
        dtype=np.uint8,
    )
    assert raw.size == frames * h * w * 3, (fixture, raw.size, frames, h, w)
    item = {
        "array": raw.reshape(frames, h, w, 3).copy(),
        "video_metadata": {"total_num_frames": frames, "fps": fps, "duration": duration,
                           "frames_indices": list(range(frames))},
    }
    return item, [int(i) for i in facts["sampled"].split(",")]


def main():
    os.makedirs(OUT, exist_ok=True)
    model = SentenceTransformer(snapshot(), device="cpu",
                                model_kwargs={"attn_implementation": "sdpa", "dtype": torch.float32})
    with torch.no_grad():
        for name, fixture, stamps in CASES:
            item, indices = reference_item(fixture)
            meta = VideoMetadata(**{k: v for k, v in item["video_metadata"].items() if k != "frames_indices"})
            kept = [int(i) for i in model[0].processor.video_processor.sample_frames(
                meta, fps=1, max_frames=32, overflow_strategy="uniform")]
            assert kept == indices, (name, kept, indices)
            set_timestamps(model, stamps)
            emb = model.encode([item], convert_to_tensor=True, batch_size=1)[0]
            np.save(os.path.join(OUT, f"video_{name}.emb_f32.npy"), emb.detach().float().cpu().numpy().astype(np.float32))
            np.save(os.path.join(OUT, f"video_{name}.indices.npy"), np.array(indices, dtype=np.int64))
            print(name, fixture, tuple(emb.shape), "frames sampled", indices)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
