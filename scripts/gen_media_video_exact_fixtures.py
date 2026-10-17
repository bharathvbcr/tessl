#!/usr/bin/env python3
"""Fixtures and oracle frames for tests/media_video_exact.rs, tests/media_yuv_exact.rs,
tests/media_yuv_formats.rs and the video end-to-end case.

    ~/.venvs/ml-2026-10/bin/python scripts/gen_media_video_exact_fixtures.py

Writes into tests/fixtures/media_video/ (small and deterministic: the frames are numpy
`default_rng` draws and closed-form shapes, the encoders are libx264 and libx265):

  <name>.mp4        an encoded input
  <name>.rgb24      the oracle's RGB frames, concatenated (what the test compares)
  <name>.bilinear.rgb24
                    the same with `-sws_flags bilinear`, for the 10-bit 4:2:0 fixtures (the
                    one class whose bytes depend on the scaler flags)
  <name>.<fmt>      the oracle's decoded planes, `-pix_fmt <fmt>`, in the fixture's own
                    format (nv12, nv16, nv24; p010, p210, p410 for 10 bits, the sample in the
                    high 10 bits of a 16-bit little-endian word)
  syn_*.planar      random planes of one format (planar, 16-bit little-endian words at 10
                    bits) handed to the converter tests without a codec; syn_*.rgb24 is the
                    CLI's conversion of the same planes (`.portable.rgb24`: with the CPU's
                    assembly off, for the 8-bit 4:2:0 and 4:2:2 formats; `.bilinear.rgb24`:
                    `-sws_flags bilinear`, for yuv420p10le)
  facts.txt         one line per fixture: the facts torchcodec's exact seek mode derives
                    (frame count, duration, average fps) computed from ffprobe's packets,
                    and the tags the stream carries

The oracle (what the reference path of google/embeddinggemma-2 runs, traced in
docs/embedgemma2.md "Media decoding"):

  transformers' `VideoProcessorMixin.fetch_videos` asks for torchcodec, and
  `transformers.video_utils.read_video_torchcodec` opens `VideoDecoder(path,
  seek_mode="exact")`. torchcodec is not installed here and torchvision 0.29 removed
  video decoding, so the oracle is the `ffmpeg` CLI, which runs the same libavcodec and
  libswscale, arranged as torchcodec arranges it. From torchcodec v0.17.0's own source
  (BSD-3; read by the integrator, copies in the lane's `target/tc_*.cpp`):

    * `SwScale.cpp` makes the colour-conversion context with `swsFlags = 0`; frames whose
      width is a multiple of 32, with no transform, are converted by it
      (`CpuDeviceInterface.cpp`, `get_color_conversion_library`);
    * any other width, and any transform, go through a libavfilter graph whose string starts
      `format=rgb24,` with no scaler flags set (`FilterGraph.cpp`);
    * a rotation is a transform (`Transform.cpp`: `transpose=cclock`, `transpose=clock` or
      `hflip,vflip`), applied after the `format=rgb24,` (Note [Transform and Format
      Conversion Order]): **convert to RGB at the stored size, then rotate the RGB**, for
      every layout.

  Black-box checks through the public libswscale/libavfilter API (`target/v2/graph_probe.c`
  in the lane, not committed; 34 random-plane classes of every format here, odd sizes
  included): a `format=rgb24` graph fed a frame carrying its own colorspace and range gives
  the same bytes as `sws_scale` with flags 0 and as the CLI's `-pix_fmt rgb24` (0 differing
  bytes in all), and the graph `format=rgb24,transpose=...` equals the CLI's conversion
  rotated afterwards. So the CLI's `-noautorotate -pix_fmt rgb24` (flags unset: libswscale's
  default, bicubic) followed by a numpy turn of the RGB is torchcodec's output.

  What the scaler flags change: only 10-bit 4:2:0, whose chroma rows are resampled to luma
  rows by the scaler (bilinear and bicubic differ in about 65 percent of the bytes of a random
  picture). torchcodec passes no flags, which is the bicubic filter: `<name>.rgb24` is that;
  `<name>.bilinear.rgb24` is the `-sws_flags bilinear` variant tessl keeps selectable.
  8-bit 4:2:0 and 4:2:2 and all 4:4:4 are the same under every flag tried.

  Rotation and flips: `<name>.rgb24` is the unrotated conversion with the display rotation
  or flip applied to the RGB (above). For 4:2:0 fixtures that are rotated or flipped,
  `<name>.cli_autorotate.rgb24` keeps what the CLI's autorotate gives instead (it turns the
  4:2:0 planes before converting), which differs; it is the evidence that the order matters.
  For 4:2:2 and 4:4:4 the CLI converts first as well and this script asserts the two agree.

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
# 80x48 and 64x48 picture and crop it) and both ranges. Beyond the original 8-bit 4:2:0 H.264
# and HEVC set: HEVC at 8 and 10 bits in 4:2:0, 4:2:2 and 4:4:4 (the formats VideoToolbox
# decodes here; its H.264 decoder takes 8-bit 4:2:0 only), the matrices FCC, SMPTE 240M and
# BT.2020, odd sizes (4:4:4 any, 4:2:2 an odd height: the only shapes H.264 and HEVC can code
# with an odd side), and rotation and flips of each layout.
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
    # 10-bit HEVC (Main 10): the 4:2:0 class whose chroma the scaler resamples.
    ("hevc_main10_64x48", "libx265", 64, 48, 6, 10, "bt709", "tv", 0),
    # The matrices the exact converter holds beyond BT.601, BT.709 and BT.2020.
    ("hevc_fcc_pc_64x48", "libx265", 64, 48, 4, 10, "fcc", "pc", 0),
    ("hevc_fcc_tv_66x38", "libx265", 66, 38, 4, 10, "fcc", "tv", 0),
    ("hevc_240m_pc_66x38", "libx265", 66, 38, 4, 10, "smpte240m", "pc", 0),
    ("hevc_240m_tv_64x48", "libx265", 64, 48, 4, 10, "smpte240m", "tv", 0),
    # 10-bit 4:2:0: both ranges, cropped sizes, BT.2020, FCC.
    ("hevc_p10_601_pc_66x38", "libx265", 66, 38, 4, 10, "smpte170m", "pc", 0),
    ("hevc_p10_2020_tv_64x32", "libx265", 64, 32, 4, 10, "bt2020nc", "tv", 0),
    ("hevc_p10_fcc_pc_50x34", "libx265", 50, 34, 4, 10, "fcc", "pc", 0),
    # 4:2:2 (8 and 10 bits; an odd height is codable).
    ("hevc_422_709_tv_64x48", "libx265", 64, 48, 4, 10, "bt709", "tv", 0),
    ("hevc_422_601_pc_66x38", "libx265", 66, 38, 4, 10, "smpte170m", "pc", 0),
    ("hevc_422_709_tv_64x33", "libx265", 64, 33, 4, 10, "bt709", "tv", 0),
    ("hevc_422_240m_pc_50x17", "libx265", 50, 17, 4, 10, "smpte240m", "pc", 0),
    ("hevc_p210_709_tv_64x48", "libx265", 64, 48, 4, 10, "bt709", "tv", 0),
    ("hevc_p210_601_pc_66x38", "libx265", 66, 38, 4, 10, "smpte170m", "pc", 0),
    ("hevc_p210_2020_tv_48x33", "libx265", 48, 33, 4, 10, "bt2020nc", "tv", 0),
    # 4:4:4 (8 and 10 bits, any size).
    ("hevc_444_709_tv_64x48", "libx265", 64, 48, 4, 10, "bt709", "tv", 0),
    ("hevc_444_601_pc_50x34", "libx265", 50, 34, 4, 10, "smpte170m", "pc", 0),
    ("hevc_444_fcc_pc_33x17", "libx265", 33, 17, 4, 10, "fcc", "pc", 0),
    ("hevc_p410_709_tv_64x48", "libx265", 64, 48, 4, 10, "bt709", "tv", 0),
    ("hevc_p410_2020_pc_50x34", "libx265", 50, 34, 4, 10, "bt2020nc", "pc", 0),
    ("hevc_p410_240m_tv_33x17", "libx265", 33, 17, 4, 10, "smpte240m", "tv", 0),
    # Rotation and flips. Sizes with a width that is a NEON width in one orientation and not
    # in the other (48x40), so the order of turning and converting shows.
    ("hevc_420_709_tv_48x40_rot90", "libx265", 48, 40, 3, 10, "bt709", "tv", 90),
    ("hevc_p10_709_tv_48x40_rot90", "libx265", 48, 40, 3, 10, "bt709", "tv", 90),
    ("hevc_p10_601_pc_40x48_rot270", "libx265", 40, 48, 3, 10, "smpte170m", "pc", 270),
    ("hevc_p10_709_tv_48x40_rot180", "libx265", 48, 40, 3, 10, "bt709", "tv", 180),
    ("hevc_p10_709_tv_48x40_vflip", "libx265", 48, 40, 3, 10, "bt709", "tv", 0),
    ("hevc_420_709_tv_48x40_hflip", "libx265", 48, 40, 3, 10, "bt709", "tv", 0),
    ("hevc_422_709_tv_48x40_rot90", "libx265", 48, 40, 3, 10, "bt709", "tv", 90),
    ("hevc_p210_709_tv_48x40_rot270", "libx265", 48, 40, 3, 10, "bt709", "tv", 270),
    ("hevc_444_709_tv_48x40_rot90", "libx265", 48, 40, 3, 10, "bt709", "tv", 90),
    ("hevc_p410_709_tv_48x40_rot270", "libx265", 48, 40, 3, 10, "bt709", "tv", 270),
    # Streams the system decodes and the exact conversion refuses by name (no golden): 12-bit video,
    # monochrome video and a codec other than H.264 and HEVC. (x265 codes no picture under 16 rows,
    # so the 10-bit 4:2:0 heights below the chroma filter's measured range are tested on the
    # converter only.)
    ("refuse_hevc_12bit_709_tv_64x48", "libx265", 64, 48, 3, 10, "bt709", "tv", 0),
    ("refuse_hevc_gray_64x48", "libx265", 64, 48, 3, 10, None, "tv", 0),
    ("refuse_mpeg4_709_tv_64x48", "mpeg4", 64, 48, 3, 10, "bt709", "tv", 0),
    # Motion JPEG: a codec AVFoundation does decode and the exact conversion does not take.
    ("refuse_mjpeg_pc_64x48", "mjpeg", 64, 48, 3, 10, None, "pc", 0),
]
REFUSED = {f[0] for f in FIXTURES if f[0].startswith("refuse_")}
# decoded by the reference script, or refused by the exact conversion: no frame-by-frame golden
NO_RGB24_GOLDEN = {"e2e_video"} | REFUSED
CRF = {"e2e_video": "24"}


def pix_fmt_of(name):
    """The pixel format a fixture is encoded in, from its name."""
    if "_12bit_" in name:
        return "yuv420p12le"
    if "_gray_" in name:
        return "gray"
    if name.startswith("hevc_main10") or "_p10_" in name:
        return "yuv420p10le"
    if "_p210_" in name:
        return "yuv422p10le"
    if "_p410_" in name:
        return "yuv444p10le"
    if "_422_" in name:
        return "yuv422p"
    if "_444_" in name:
        return "yuv444p"
    return "yuv420p"


def flip_of(name):
    return "h" if name.endswith("_hflip") else "v" if name.endswith("_vflip") else ""


# ffmpeg's bi-planar pixel format for each decoded format, and the golden file's extension.
PLANES = {
    "yuv420p": ("nv12", "nv12"),
    "yuv422p": ("nv16", "nv16"),
    "yuv444p": ("nv24", "nv24"),
    "yuv420p10le": ("p010le", "p010"),
    "yuv422p10le": ("p210le", "p210"),
    "yuv444p10le": ("p410le", "p410"),
}
NV12_GOLDENS = {f[0] for f in FIXTURES} - NO_RGB24_GOLDEN
# Planes with random samples, outside the legal ranges too, run through the CLI as raw NV12
# input: name, width, height, matrix tag, range. They hold the converter to libswscale for every
# Y, Cb and Cr that natural pictures leave out, on both converters (width a multiple of 16 or not).
SYNTHETIC = [
    ("rand_709_tv_32x16", 32, 16, "bt709", "tv"),
    ("rand_709_tv_34x16", 34, 16, "bt709", "tv"),
    ("rand_601_pc_48x8", 48, 8, "smpte170m", "pc"),
    ("rand_601_pc_50x8", 50, 8, "smpte170m", "pc"),
    ("rand_2020_tv_16x16", 16, 16, "bt2020nc", "tv"),
    ("rand_2020_pc_18x12", 18, 12, "bt2020nc", "pc"),
    ("rand_untagged_16x8", 16, 8, None, "tv"),
    ("rand_untagged_30x6", 30, 6, None, "tv"),
]
# Random planes of every format the converter takes, handed to the CLI as planar raw video:
# format, matrix tag, range, width, height. Odd sides where the layout allows them, widths on
# and off the NEON widths, every matrix and both ranges; 10-bit 4:2:0 at heights from 14 on
# (below them the scaler builds a different filter, which the converter refuses by name).
FORMATS = [
    ("yuv422p", "bt709", "tv", 32, 16),
    ("yuv422p", "smpte170m", "pc", 34, 16),
    ("yuv422p", "fcc", "pc", 48, 9),
    ("yuv422p", "smpte240m", "tv", 50, 7),
    ("yuv422p", "bt2020nc", "tv", 16, 8),
    ("yuv422p", "bt709", "pc", 2, 1),
    ("yuv444p", "bt709", "tv", 32, 16),
    ("yuv444p", "smpte170m", "pc", 17, 9),
    ("yuv444p", "fcc", "pc", 5, 3),
    ("yuv444p", "smpte240m", "tv", 33, 17),
    ("yuv444p", "bt2020nc", "pc", 16, 16),
    ("yuv444p", "bt709", "pc", 1, 1),
    ("yuv444p10le", "bt709", "tv", 16, 8),
    ("yuv444p10le", "smpte170m", "pc", 17, 9),
    ("yuv444p10le", "bt2020nc", "tv", 33, 17),
    ("yuv444p10le", "fcc", "pc", 32, 3),
    ("yuv444p10le", "smpte240m", "pc", 5, 5),
    ("yuv444p10le", "bt709", "pc", 1, 1),
    ("yuv422p10le", "bt709", "tv", 32, 16),
    ("yuv422p10le", "smpte170m", "pc", 34, 9),
    ("yuv422p10le", "bt2020nc", "pc", 16, 16),
    ("yuv422p10le", "fcc", "tv", 50, 7),
    ("yuv422p10le", "smpte240m", "pc", 6, 5),
    ("yuv420p10le", "bt709", "tv", 32, 16),
    ("yuv420p10le", "smpte170m", "pc", 34, 14),
    ("yuv420p10le", "bt2020nc", "tv", 16, 24),
    ("yuv420p10le", "fcc", "pc", 48, 34),
    ("yuv420p10le", "smpte240m", "tv", 50, 18),
    ("yuv420p10le", "bt709", "pc", 16, 100),
    ("yuv420p10le", "smpte170m", "tv", 8, 2),
    ("yuv420p", "fcc", "pc", 48, 8),
    ("yuv420p", "smpte240m", "pc", 50, 8),
    ("yuv420p", "fcc", "tv", 32, 16),
    ("yuv420p", "smpte240m", "tv", 34, 16),
]
MATRIX_OUT = {"bt709": "bt709", "smpte170m": "bt601", "bt2020nc": "bt2020", "fcc": "fcc", "smpte240m": "smpte240m"}
PRIMARIES = {"bt709": "bt709", "smpte170m": "smpte170m", "bt2020nc": "bt2020", "fcc": "bt470m", "smpte240m": "smpte240m"}
TRC = {"bt709": "bt709", "smpte170m": "smpte170m", "bt2020nc": "bt709", "fcc": "bt709", "smpte240m": "smpte240m"}


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
        vf += f"out_color_matrix={MATRIX_OUT[matrix]}:"
    # (The Motion JPEG encoder takes only the full-range names of its formats.)
    vf += f"out_range={'full' if rng_ == 'pc' else 'limited'},format={'yuvj420p' if codec == 'mjpeg' else pix_fmt_of(name)}"
    quality = ["-q:v", "2"] if codec in ("mpeg4", "mjpeg") else ["-crf", CRF.get(name, "12")]
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-framerate", str(fps),
           "-i", "-", "-vf", vf, "-c:v", codec, *quality, "-g", "5" if n > 3 else "3",
           "-bf", "0" if codec == "mjpeg" else "2"]
    if matrix:
        cmd += ["-colorspace", matrix, "-color_primaries", PRIMARIES[matrix], "-color_trc", TRC[matrix]]
    if rng_ == "pc":
        cmd += ["-color_range", "pc"]
    elif matrix:
        cmd += ["-color_range", "tv"]
    if codec == "libx265":
        cmd += ["-tag:v", "hvc1", "-x265-params", "log-level=error"]
    cmd += ["-movflags", "+faststart", "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact", path]
    run(cmd, input=raw)
    flip = flip_of(name)
    if rot or flip:
        tmp = os.path.join(SCRATCH, name + "_norot.mp4")
        shutil.copy(path, tmp)
        turn = ["-display_rotation", str(rot)] if rot else [f"-display_{flip}flip"]
        run([FFMPEG, "-v", "error", "-y", *turn, "-i", tmp, "-c", "copy",
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


def decode(path, pix_fmt, autorotate=True, flags=None):
    cmd = [FFMPEG, "-v", "error"]
    if not autorotate:
        cmd.append("-noautorotate")
    cmd += ["-i", path]
    if flags:
        cmd += ["-sws_flags", flags]
    cmd += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    return run(cmd, capture_output=True).stdout


def torchcodec_rgb24(path, w, h, n, rot, flip, flags=None):
    """The frames torchcodec returns: the stored picture converted to rgb24 (flags unset unless
    asked), then the display rotation or flip applied to the RGB (module docstring)."""
    frames = np.frombuffer(decode(path, "rgb24", autorotate=False, flags=flags), np.uint8).reshape(n, h, w, 3)
    if rot == 90:
        turn = lambda a: np.rot90(a, 1)  # noqa: E731
    elif rot == 270:
        turn = lambda a: np.rot90(a, -1)  # noqa: E731
    elif rot == 180:
        turn = lambda a: a[::-1, ::-1]  # noqa: E731
    elif flip == "h":
        turn = lambda a: a[:, ::-1]  # noqa: E731
    elif flip == "v":
        turn = lambda a: a[::-1]  # noqa: E731
    else:
        return frames.tobytes()
    return b"".join(np.ascontiguousarray(turn(f)).tobytes() for f in frames)


# Sampling settings recorded per fixture: the processor's own `sample_frames` is run on the
# metadata above (fps 1 with a 32 frame cap is processor_config.json's; the others make the cap bite).
SAMPLINGS = {
    "sampled": dict(fps=1, max_frames=32, overflow_strategy="uniform"),
    "sampled_f5m4u": dict(fps=5, max_frames=4, overflow_strategy="uniform"),
    "sampled_f5m4t": dict(fps=5, max_frames=4, overflow_strategy="truncate"),
    "sampled_all_m3u": dict(fps=None, max_frames=3, overflow_strategy="uniform"),
}


def sample(total, fps_avg, duration, **kw):
    """transformers' EmbeddingGemma2VideoProcessor.sample_frames on torchcodec-style metadata."""
    from transformers.models.embedding_gemma2.video_processing_embedding_gemma2 import EmbeddingGemma2VideoProcessor
    from transformers.video_utils import VideoMetadata

    meta = VideoMetadata(total_num_frames=total, fps=fps_avg, duration=duration)
    return EmbeddingGemma2VideoProcessor().sample_frames(meta, **kw)


def synthetic(facts):
    for seed, (name, w, h, matrix, rng_) in enumerate(SYNTHETIC, start=100):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 256, (h, w), dtype=np.uint8)
        uv = rng.integers(0, 256, (h // 2, w), dtype=np.uint8)  # interleaved Cb, Cr
        with open(os.path.join(OUT, name + ".nv12"), "wb") as f:
            f.write(y.tobytes() + uv.tobytes())
        # The oracle is given planar yuv420p, the form libavcodec's H.264 and HEVC decoders
        # produce and torchcodec converts; ffmpeg's NV12 input takes a different swscale path.
        planes = y.tobytes() + np.ascontiguousarray(uv[:, 0::2]).tobytes() + np.ascontiguousarray(uv[:, 1::2]).tobytes()
        for suffix, cpu in ((".rgb24", []), (".portable.rgb24", ["-cpuflags", "0"])):
            # `-cpuflags 0` turns the CPU's assembly off, so libswscale runs its C table
            # converter at every width (the aarch64 NEON one needs a width multiple of 16).
            cmd = [FFMPEG, "-v", "error", "-y", *cpu, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}"]
            if matrix:
                cmd += ["-colorspace", matrix]
            cmd += ["-color_range", rng_, "-i", "-", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
            rgb = run(cmd, input=planes, capture_output=True).stdout
            assert len(rgb) == w * h * 3, name
            with open(os.path.join(OUT, name + suffix), "wb") as f:
                f.write(rgb)
        facts.append(f"synthetic {name} w={w} h={h} matrix={matrix or 'unspecified'} range={rng_}")


def plane_shapes(fmt, w, h):
    kind = fmt[3:6]
    cw = w if kind == "444" else (w + 1) // 2
    ch = (h + 1) // 2 if kind == "420" else h
    return (h, w), (ch, cw)


def formats(facts):
    """Random planes of every format, with the CLI's conversion (default flags, bilinear, no assembly)."""
    for seed, (fmt, matrix, rng_, w, h) in enumerate(FORMATS, start=500):
        rng = np.random.default_rng(seed)
        hi = 1024 if "10" in fmt else 256
        dt = "<u2" if hi == 1024 else "u1"
        (yh, yw), (ch, cw) = plane_shapes(fmt, w, h)
        planes = b"".join(rng.integers(0, hi, shape).astype(dt).tobytes() for shape in ((yh, yw), (ch, cw), (ch, cw)))
        name = f"syn_{fmt}_{matrix}_{rng_}_{w}x{h}"
        with open(os.path.join(OUT, name + ".planar"), "wb") as f:
            f.write(planes)
        variants = [(".rgb24", [], [])]
        if fmt in ("yuv420p", "yuv422p"):
            variants.append((".portable.rgb24", ["-cpuflags", "0"], []))
        if fmt == "yuv420p10le":
            variants.append((".bilinear.rgb24", [], ["-sws_flags", "bilinear"]))
        for suffix, pre, post in variants:
            cmd = [FFMPEG, "-v", "error", "-y", *pre, "-f", "rawvideo", "-pix_fmt", fmt, "-s", f"{w}x{h}",
                   "-colorspace", matrix, "-color_range", rng_, "-i", "-", *post, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
            rgb = run(cmd, input=planes, capture_output=True).stdout
            assert len(rgb) == w * h * 3, name
            with open(os.path.join(OUT, name + suffix), "wb") as f:
                f.write(rgb)
        facts.append(f"format {name} fmt={fmt} w={w} h={h} matrix={matrix} range={rng_}")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SCRATCH, exist_ok=True)
    facts = []
    synthetic(facts)
    formats(facts)
    for seed, (name, codec, w, h, n, fps, matrix, rng_, rot) in enumerate(FIXTURES, start=1):
        path = encode(name, codec, w, h, n, fps, matrix, rng_, rot, seed)
        s = probe_stream(path)
        want = {"smpte170m": "smpte170m", "bt709": "bt709", "fcc": "fcc", "smpte240m": "smpte240m", "bt2020nc": "bt2020nc",
                None: None}[matrix]
        assert (s.get("color_space") if matrix else None) in (want, None) and s["width"] * s["height"] > 0
        # ffprobe names a full-range 8-bit stream yuvj420p (and so on): the same samples, the range is a tag.
        assert s["pix_fmt"].replace("yuvj", "yuv") == pix_fmt_of(name), (name, s["pix_fmt"])
        flip = flip_of(name)
        cli = decode(path, "rgb24")  # the CLI's autorotate
        rgb = torchcodec_rgb24(path, w, h, n, rot, flip)
        ow, oh = (h, w) if rot in (90, 270) else (w, h)
        assert len(rgb) == ow * oh * 3 * n, (name, len(rgb), ow, oh, n)
        if name not in NO_RGB24_GOLDEN:
            if pix_fmt_of(name)[:7] != "yuv420p":
                # The CLI converts 4:2:2 and 4:4:4 first as well: the two orders agree (and this
                # checks the rotation directions above).
                assert cli == rgb, name
            with open(os.path.join(OUT, name + ".rgb24"), "wb") as f:
                f.write(rgb)
            if (rot or flip) and pix_fmt_of(name)[:7] == "yuv420p":
                with open(os.path.join(OUT, name + ".cli_autorotate.rgb24"), "wb") as f:
                    f.write(cli)
            if pix_fmt_of(name) == "yuv420p10le":
                with open(os.path.join(OUT, name + ".bilinear.rgb24"), "wb") as f:
                    f.write(torchcodec_rgb24(path, w, h, n, rot, flip, flags="bilinear"))
        if name in NV12_GOLDENS and name not in NO_RGB24_GOLDEN:
            fmt, ext = PLANES[pix_fmt_of(name)]
            nv = decode(path, fmt, autorotate=False)
            sub_w, sub_h = {"yuv420p": (2, 2), "yuv422p": (2, 1), "yuv444p": (1, 1)}[pix_fmt_of(name)[:7]]
            cw, ch = (w + sub_w - 1) // sub_w, (h + sub_h - 1) // sub_h
            bps = 2 if "10" in pix_fmt_of(name) else 1
            assert len(nv) == (w * h + 2 * cw * ch) * bps * n, name
            with open(os.path.join(OUT, f"{name}.{ext}"), "wb") as f:
                f.write(nv)
        frames, duration, avg, begin, end, pts = torchcodec_exact_metadata(path)
        sampled = {
            tag: ",".join(str(int(i)) for i in sample(frames, avg, duration, **kw))
            for tag, kw in SAMPLINGS.items()
        }
        facts.append(
            f"video {name} codec={codec} w={w} h={h} rotation={rot} flip={flip_of(name) or 'none'} "
            f"golden={'no' if name in NO_RGB24_GOLDEN else 'yes'} frames={frames} "
            f"pix_fmt={s['pix_fmt']} color_range={s.get('color_range', 'unspecified')} "
            f"color_space={s.get('color_space', 'unspecified')} "
            f"duration={duration!r} average_fps={avg!r} begin={begin!r} end={end!r} "
            + " ".join(f"{tag}={v}" for tag, v in sampled.items())
            + f" pts={','.join(repr(t) for t in pts)}"
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
        elif name.endswith(".portable.rgb24"):
            role, note = "golden", "the ffmpeg CLI's conversion of the input of the same name with the CPU's assembly off (`-cpuflags 0`)"
        elif name.endswith(".cli_autorotate.rgb24"):
            role, note = "golden", "the ffmpeg CLI's autorotated rgb24 frames of a rotated or flipped 4:2:0 input (it turns the planes before converting; torchcodec converts first, so this is NOT the reference, kept to show the orders differ)"
        elif name.endswith(".bilinear.rgb24"):
            role, note = "golden", "the ffmpeg CLI's rgb24 conversion with `-sws_flags bilinear` (10-bit 4:2:0 depends on the flag)"
        elif name.endswith(".rgb24"):
            role, note = "golden", "the rgb24 frames torchcodec returns for the input of the same name (converted at the stored size by the ffmpeg CLI, then rotated or flipped as RGB), concatenated"
        elif name.startswith("syn_") and name.endswith(".planar"):
            role, note = "input", "random planar planes of one format (16-bit little-endian at 10 bits) the converter is given"
        elif name.startswith("rand_") and name.endswith(".nv12"):
            role, note = "input", "random NV12 planes the converter is given (its rgb24 golden is the ffmpeg CLI conversion of the same samples as planar yuv420p)"
        elif name.endswith((".nv12", ".nv16", ".nv24", ".p010", ".p210", ".p410")):
            role, note = "golden", "the ffmpeg CLI's bi-planar decode (nv12/nv16/nv24, p010le/p210le/p410le) for the input of the same name (as stored, not rotated)"
        else:
            role, note = "input", "an encoded video the decoders under test are given"
        table[name] = {"role": role, "published": True, "note": note}
    return table


def write_fixture_manifest():
    fm.write_manifest(
        OUT,
        GENERATOR,
        "generated: numpy default_rng draws and closed-form shapes, encoded by ffmpeg (libx264, libx265) with "
        "the tags the file name says; rotation and flips are a display matrix added by a stream copy. Encoder output is "
        "stable for a given tool version; the goldens are the ffmpeg CLI's output for those exact files.",
        file_table(),
        "produced",
    )


if __name__ == "__main__":
    main()
