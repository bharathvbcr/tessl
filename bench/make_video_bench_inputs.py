#!/usr/bin/env python3
"""Video inputs for `bench_media_decode`'s colour-conversion rows
(src/bin/bench_media_decode.rs, bench/results/media_video_color_m5pro.json).

    ~/.venvs/ml-2026-10/bin/python bench/make_video_bench_inputs.py [OUT_DIR]

Writes into OUT_DIR (default target/media_bench_video) deterministic ffmpeg
`testsrc2` clips, yuv420p, 30 fps, BT.709 tagged limited range:

  video_10s_720p_h264.mp4     1280x720, 300 frames (a width libswscale's aarch64
                              NEON converter takes)
  video_10s_720p_hevc.mp4     the same as HEVC (hvc1)
  video_5s_1080p_h264.mp4     1920x1080, 150 frames
  video_5s_1270x720_h264.mp4  1270x720, 150 frames (a width it does not take: the
                              table converter)
"""

import os
import subprocess
import sys

FFMPEG = "/opt/homebrew/bin/ffmpeg"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "target", "media_bench_video")

CLIPS = [  # name, size, seconds, encoder arguments
    ("video_10s_720p_h264.mp4", "1280x720", "10", ["-c:v", "libx264", "-crf", "23"]),
    ("video_10s_720p_hevc.mp4", "1280x720", "10", ["-c:v", "libx265", "-crf", "26", "-tag:v", "hvc1"]),
    ("video_5s_1080p_h264.mp4", "1920x1080", "5", ["-c:v", "libx264", "-crf", "23"]),
    ("video_5s_1270x720_h264.mp4", "1270x720", "5", ["-c:v", "libx264", "-crf", "23"]),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, size, secs, enc in CLIPS:
        path = os.path.join(OUT, name)
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30",
                        "-t", secs, "-pix_fmt", "yuv420p", *enc,
                        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                        "-color_range", "tv", path], check=True)
        print(f"{name:32} {os.path.getsize(path):>10} bytes")


if __name__ == "__main__":
    main()
