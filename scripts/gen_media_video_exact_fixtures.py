#!/usr/bin/env python3
"""Fixtures and oracle frames for tests/media_video_exact.rs and the video end-to-end case.

    ~/.venvs/ml-2026-10/bin/python scripts/gen_media_video_exact_fixtures.py

Writes into tests/fixtures/media_video/ (small and deterministic: the frames are numpy
`default_rng` draws and closed-form shapes, the encoders are libx264 and libx265):

  <name>.mp4        an encoded input
  <name>.rgb24      the oracle's RGB frames, concatenated (what the test compares)
  <name>.nv12       the oracle's decoded planes (Y, then interleaved Cb/Cr) of the
                    fixtures listed in NV12_GOLDENS (what the plane-equality test compares)
  facts.txt         one line per fixture: the facts torchcodec's exact seek mode derives
                    (frame count, duration, average fps) computed from ffprobe's packets,
                    and the tags the stream carries

The oracle (what the reference path of google/embeddinggemma-2 runs, traced in
docs/embedgemma2.md "Media decoding"):

  transformers' `VideoProcessorMixin.fetch_videos` asks for torchcodec, and
  `transformers.video_utils.read_video_torchcodec` opens `VideoDecoder(path,
  seek_mode="exact")`. torchcodec is not installed here and torchvision 0.29 removed
  video decoding, so the oracle is the `ffmpeg` CLI, which runs the same libavcodec and
  libswscale. torchcodec converts with libswscale (`SwScale.cpp`; the frame's own
  colorspace and range, flags 0) when input and output widths are multiples of 32 and
  with an `format=rgb24` filtergraph otherwise, and the two give the same bytes for a
  same-size YUV 4:2:0 to RGB conversion (checked: bytes from `sws_getContext` flags 0,
  1, 2 and 4 and the CLI's bicubic are equal on even sizes). The CLI's `-pix_fmt rgb24`
  is that conversion. A rotated stream is rotated by torchcodec after the colour
  conversion (`RotationTransform`, `transpose=...`) and by the CLI's autorotate before
  it; for even sizes the two agree, because the chroma is replicated, not interpolated.

Frame count, duration and average fps follow torchcodec's exact seek mode
(`Metadata.cpp`): the number of packets, `max(pts + duration) - min(pts)` in seconds,
and their quotient. They are computed here from `ffprobe -show_packets`.
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools", "qwen35_ref"))
import fixture_manifest as fm  # noqa: E402

OUT = os.path.join(HERE, "..", "tests", "fixtures", "media_video")
SCRATCH = os.path.join(HERE, "..", "target", "media_video_exact")
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
GENERATOR = "scripts/gen_media_video_exact_fixtures.py"

# name, codec, width, height, frames, fps, matrix tag (None: untagged), range, rotation
# Widths that are multiples of 16 take libswscale's aarch64 NEON converter and the rest the
# portable table converter; both are covered, as are cropped sizes (66x38 and 50x34 code a
# 80x48 and 64x48 picture and crop it) and both ranges.
FIXTURES = [
    ("h264_709_tv_64x48", "libx264", 64, 48, 6, 10, "bt709", "tv", 0),
    ("h264_601_tv_66x38", "libx264", 66, 38, 6, 10, "smpte170m", "tv", 0),
    ("h264_709_pc_80x48", "libx264", 80, 48, 6, 10, "bt709", "pc", 0),
    ("h264_601_pc_50x34", "libx264", 50, 34, 6, 10, "smpte170m", "pc", 0),
    ("h264_untagged_64x48", "libx264", 64, 48, 6, 10, None, "tv", 0),
    ("hevc_709_tv_64x48", "libx265", 64, 48, 6, 10, "bt709", "tv", 0),
    ("hevc_601_pc_34x26", "libx265", 34, 26, 6, 10, "smpte170m", "pc", 0),
    ("hevc_709_pc_96x64", "libx265", 96, 64, 6, 10, "bt709", "pc", 0),
    ("h264_709_tv_64x48_rot90", "libx264", 64, 48, 6, 10, "bt709", "tv", 90),
    ("h264_601_tv_66x38_rot270", "libx264", 66, 38, 6, 10, "smpte170m", "tv", 270),
    # The end-to-end file: 80 frames at 24000/1001 fps (3.337 s, 3 sampled frames at 1 fps), so
    # the source rate is not a round number and sampling is not the same as `index / 10`.
    ("e2e_video", "libx264", 96, 64, 80, "24000/1001", "bt709", "tv", 0),
]
NO_RGB24_GOLDEN = {"e2e_video"}  # decoded by the reference script, not compared frame by frame
CRF = {"e2e_video": "24"}
NV12_GOLDENS ={"h264_709_tv_64x48", "h264_601_tv_66x38", "hevc_601_pc_34x26", "hevc_709_pc_96x64"}


def frame(w, h, k, seed):
    """One RGB frame: a gradient, three saturated moving shapes and a little noise."""
    rng = np.random.default_rng(seed * 1000 + k)
    yy, xx = np.mgrid[0:h, 0:w]
    f = np.zeros((h, w, 3), np.float64)
    f[..., 0] = 40 + 160 * xx / max(w - 1, 1)
    f[..., 1] = 30 + 150 * yy / max(h - 1, 1)
    f[..., 2] = 90 + 60 * np.sin((xx + 2 * k) / 7.0)
    cx = (3 * k) % max(w - 12, 1)
    f[h // 4: h // 4 + max(h // 3, 4), cx: cx + 10] = (235, 30, 40)
    f[h // 2: h // 2 + max(h // 4, 3), w // 3: w // 3 + 12] = (30, 220, 60)
    r = max(min(w, h) // 6, 3)
    disc = (xx - (w - 8 - k)) ** 2 + (yy - (h // 3)) ** 2 < r * r
    f[disc] = (250, 230, 20)
    f += rng.normal(0, 4.0, f.shape)
    return np.clip(np.round(f), 0, 255).astype(np.uint8)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def encode(name, codec, w, h, n, fps, matrix, rng_, rot, seed):
    path = os.path.join(OUT, name + ".mp4")
    raw = b"".join(frame(w, h, k, seed).tobytes() for k in range(n))
    # The RGB to YCbCr conversion before the encoder uses the matrix and range the stream
    # is tagged with (untagged streams: the conversion's default, BT.601 limited).
    vf = "scale="
    if matrix:
        vf += f"out_color_matrix={'bt709' if matrix == 'bt709' else 'bt601'}:"
    vf += f"out_range={'full' if rng_ == 'pc' else 'limited'},format=yuv420p"
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-framerate", str(fps),
           "-i", "-", "-vf", vf, "-c:v", codec, "-crf", CRF.get(name, "12"), "-g", "5", "-bf", "2"]
    if matrix:
        prim = "bt709" if matrix == "bt709" else "smpte170m"
        trc = "bt709" if matrix == "bt709" else "smpte170m"
        cmd += ["-colorspace", matrix, "-color_primaries", prim, "-color_trc", trc]
    if rng_ == "pc":
        cmd += ["-color_range", "pc"]
    elif matrix:
        cmd += ["-color_range", "tv"]
    if codec == "libx265":
        cmd += ["-tag:v", "hvc1", "-x265-params", "log-level=error"]
    cmd += ["-movflags", "+faststart", "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact", path]
    run(cmd, input=raw)
    if rot:
        tmp = os.path.join(SCRATCH, name + "_norot.mp4")
        shutil.copy(path, tmp)
        run([FFMPEG, "-v", "error", "-y", "-display_rotation", str(rot), "-i", tmp, "-c", "copy",
             "-fflags", "+bitexact", "-movflags", "+faststart", path])
    return path


def probe_stream(path):
    out = run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,pix_fmt,color_range,color_space,color_primaries,color_transfer,time_base,nb_frames",
               "-of", "json", path], capture_output=True, text=True).stdout
    return json.loads(out)["streams"][0]


def torchcodec_exact_metadata(path):
    """num_frames, duration and average fps the way torchcodec's exact seek mode computes them."""
    s = probe_stream(path)
    num, den = (int(x) for x in s["time_base"].split("/"))
    out = run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts,dts,duration,flags",
               "-of", "json", path], capture_output=True, text=True).stdout
    pkts = [p for p in json.loads(out)["packets"] if "D" not in p.get("flags", "")]
    pts = [int(p["pts"]) if p.get("pts") not in (None, "N/A") else int(p["dts"]) for p in pkts]
    ends = [t + int(p["duration"]) for t, p in zip(pts, pkts)]
    secs = lambda x: float(x) * num / den  # Metadata.cpp pts_to_seconds
    begin, end = secs(min(pts)), secs(max(ends))
    duration = end - begin
    return len(pkts), duration, len(pkts) / duration, begin, end, sorted(secs(t) for t in pts)


def decode(path, pix_fmt, autorotate=True):
    cmd = [FFMPEG, "-v", "error"]
    if not autorotate:
        cmd.append("-noautorotate")
    cmd += ["-i", path, "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    return run(cmd, capture_output=True).stdout


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SCRATCH, exist_ok=True)
    facts = []
    for seed, (name, codec, w, h, n, fps, matrix, rng_, rot) in enumerate(FIXTURES, start=1):
        path = encode(name, codec, w, h, n, fps, matrix, rng_, rot, seed)
        s = probe_stream(path)
        want = {"smpte170m": "smpte170m", "bt709": "bt709", None: None}[matrix]
        assert (s.get("color_space") if matrix else None) in (want, None) and s["width"] * s["height"] > 0
        rgb = decode(path, "rgb24")
        ow, oh = (h, w) if rot in (90, 270) else (w, h)
        assert len(rgb) == ow * oh * 3 * n, (name, len(rgb), ow, oh, n)
        if name not in NO_RGB24_GOLDEN:
            with open(os.path.join(OUT, name + ".rgb24"), "wb") as f:
                f.write(rgb)
        if name in NV12_GOLDENS:
            nv = decode(path, "nv12", autorotate=False)
            assert len(nv) == (w * h + 2 * ((w + 1) // 2) * ((h + 1) // 2)) * n, name
            with open(os.path.join(OUT, name + ".nv12"), "wb") as f:
                f.write(nv)
        frames, duration, avg, begin, end, pts = torchcodec_exact_metadata(path)
        facts.append(
            f"video {name} codec={codec} w={w} h={h} rotation={rot} frames={frames} "
            f"pix_fmt={s['pix_fmt']} color_range={s.get('color_range', 'unspecified')} "
            f"color_space={s.get('color_space', 'unspecified')} "
            f"duration={duration!r} average_fps={avg!r} begin={begin!r} end={end!r} "
            f"pts={','.join(repr(t) for t in pts)}"
        )
        print(facts[-1][:200])
    with open(os.path.join(OUT, "facts.txt"), "w") as f:
        f.write("\n".join(facts) + "\n")
    write_fixture_manifest()


def file_table():
    table = {}
    for name in sorted(os.listdir(OUT)):
        if name == "MANIFEST.json":
            continue
        if name == "facts.txt":
            role, note = "config", "per fixture: size, rotation, tags and torchcodec's exact-mode frame count, duration and fps"
        elif name.endswith(".rgb24"):
            role, note = "golden", "the ffmpeg CLI's rgb24 frames for the input of the same name (autorotated), concatenated"
        elif name.endswith(".nv12"):
            role, note = "golden", "the ffmpeg CLI's nv12 frames for the input of the same name (as stored, not rotated)"
        else:
            role, note = "input", "an encoded video the decoders under test are given"
        table[name] = {"role": role, "published": True, "note": note}
    return table


def write_fixture_manifest():
    fm.write_manifest(
        OUT,
        GENERATOR,
        "generated: numpy default_rng draws and closed-form shapes, encoded by ffmpeg (libx264, libx265) with "
        "the tags the file name says; rotation is a display matrix added by a stream copy. Encoder output is "
        "stable for a given tool version; the goldens are the ffmpeg CLI's output for those exact files.",
        file_table(),
        "produced",
    )


if __name__ == "__main__":
    main()
