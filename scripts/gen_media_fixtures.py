#!/usr/bin/env python3
"""Fixtures and oracle outputs for tests/media_decode.rs (src/media.rs).

    ~/.venvs/ml-2026-10/bin/python scripts/gen_media_fixtures.py [images|audio|video|all]

Writes into tests/fixtures/media/ (small, deterministic where the tool is):

  manifest.txt      one line per fixture: `kind name key=value ...`
  *.png/.tiff/...   encoded inputs
  *.rgb / *.rgba    the oracle's decoded bytes, HWC (what the test compares)
  *.f32             the oracle's decoded audio, little-endian f32
  *.rgb24           the oracle's decoded video frames, concatenated

Oracles (what the reference path of google/embeddinggemma-2 runs):

  images  sentence-transformers hands a path string to the processor, whose
          `fetch_images` calls `transformers.image_utils.load_image_as_tensor`,
          i.e. `torchvision.io.decode_image(path, mode=ImageReadMode.RGB)`
          (libpng / libjpeg-turbo, no colour management, no EXIF transpose).
          Traced in this environment (see docs/embedgemma2.md, "Media decoding").
          A PIL image given directly goes through `PIL.ImageOps.exif_transpose`
          and `convert("RGB")` instead. TIFF and BMP are not torchvision formats,
          so PIL decodes those. Where both decode a file, they are compared and
          the difference is printed.
  audio   `transformers.audio_utils.load_audio` resolves to torchcodec (FFmpeg's
          libswresample, `num_channels=1`) else librosa (soundfile, mean over
          channels, soxr_hq). Neither is installed here, so the oracle is the
          `ffmpeg` CLI (`-ac 1 -ar 16000 -f f32le`), which is the same library
          torchcodec calls, and `torchaudio.functional.resample` on the channel
          mean (the "torchaudio" backend of load_audio) as a second reference.
  video   torchcodec is not installed and torchvision 0.29 removed video
          decoding, so the oracle is `ffmpeg` (libavcodec, libswscale) plus
          `ffprobe`, and OpenCV (`cv2.VideoCapture`, the "opencv" backend of
          transformers.video_utils.load_video).
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import zlib

import numpy as np
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools", "qwen35_ref"))
import fixture_manifest as fm  # noqa: E402

OUT =os.path.join(HERE, "..", "tests", "fixtures", "media")
SCRATCH = os.path.join(HERE, "..", "target", "media_fixtures")
manifest = []


def path(name):
    return os.path.join(OUT, name)


def note(kind, name, **kv):
    manifest.append(" ".join([kind, name] + [f"{k}={v}" for k, v in kv.items()]))


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def photo(h, w, seed, c=3):
    """Smooth gradients, a disc, hard edges and a little noise: a stand-in for a photo."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    chans = [
        127.5 * (1 + np.sin(2 * np.pi * x / w + k)) * (0.6 + 0.4 * y / h) + 20 * np.cos(x / (3 + k))
        for k in range(max(c, 3))
    ]
    img = np.stack(chans, axis=-1)
    disc = (x - 0.6 * w) ** 2 + (y - 0.4 * h) ** 2 < (0.25 * min(h, w)) ** 2
    img[disc] = [240.0, 30.0, 60.0][: img.shape[-1]] + [0.0] * (img.shape[-1] - 3)
    img[: h // 5, : w // 3] = 8.0
    img += rng.normal(0, 3.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)[..., :c]


def noise(h, w, c, seed):
    return np.random.default_rng(seed).integers(0, 256, (h, w, c), dtype=np.uint8)


def png_chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def write_png16(file, arr16, color_type):
    """A 16-bit PNG (PIL has no 16-bit RGB mode to write). arr16: HxWxC uint16."""
    h, w = arr16.shape[:2]
    ihdr = struct.pack(">IIBBBBB", w, h, 16, color_type, 0, 0, 0)
    raw = b"".join(b"\x00" + arr16[r].astype(">u2").tobytes() for r in range(h))
    with open(file, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(png_chunk(b"IHDR", ihdr))
        f.write(png_chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(png_chunk(b"IEND", b""))


def tv_decode(file):
    """torchvision.io.decode_image(mode=RGB) or None when it is not a torchvision format."""
    import torchvision.io as tio

    try:
        t = tio.decode_image(file, mode=tio.ImageReadMode.RGB)
    except Exception:
        return None
    if t.dim() == 4:  # animated GIF: [frames, C, H, W]
        t = t[0]
    return t.permute(1, 2, 0).numpy()


def emit_image(name, exact_rgb=None, tag="exact"):
    """Save the oracle's RGB (torchvision when it can, else PIL) and RGBA (PIL) of fixture `name`."""
    f = path(name)
    pil = Image.open(f)
    pil.load()
    pil_rgb = np.asarray(pil.convert("RGB"))
    pil_rgba = np.asarray(pil.convert("RGBA"))
    tv = tv_decode(f)
    note_extra = {}
    if tv is not None:
        if tv.dtype != np.uint8:
            note_extra["torchvision"] = f"dtype_{tv.dtype}"
            ref = pil_rgb
        else:
            d = np.abs(tv.astype(int) - pil_rgb.astype(int))
            note_extra["tv_vs_pil_max"] = int(d.max())
            note_extra["tv_vs_pil_differing"] = int((d.max(axis=-1) > 0).sum())
            ref = tv
            if name.endswith(".png") or name.endswith(".gif"):
                print(f"{name}: torchvision vs PIL max diff {d.max()}")
            else:
                print(f"{name}: torchvision vs PIL max diff {d.max()} ({(d.max(axis=-1) > 0).sum()} px)")
    else:
        ref = pil_rgb
    h, w = ref.shape[:2]
    stem = name
    with open(path(stem + ".rgb"), "wb") as o:
        o.write(np.ascontiguousarray(ref).tobytes())
    with open(path(stem + ".rgba"), "wb") as o:
        o.write(np.ascontiguousarray(pil_rgba).tobytes())
    note("image", name, w=w, h=h, tag=tag, oracle="torchvision" if tv is not None and tv.dtype == np.uint8 else "pil",
         **note_extra)


def images():
    os.makedirs(OUT, exist_ok=True)
    # --- lossless -----------------------------------------------------------
    rgb = noise(23, 37, 3, 1)
    Image.fromarray(rgb, "RGB").save(path("rgb.png"))
    emit_image("rgb.png")

    # Alpha: opaque, transparent with a colour under it (PIL keeps the colour), and partial.
    rgba = noise(23, 37, 4, 2)
    rgba[::3, ::2, 3] = 255
    rgba[1::3, ::2, 3] = 0
    rgba[2::3, 1::2, 3] = 128
    rgba[2, 3] = [200, 100, 50, 0]
    Image.fromarray(rgba, "RGBA").save(path("rgba.png"))
    emit_image("rgba.png")

    Image.fromarray(noise(23, 37, 1, 3)[..., 0], "L").save(path("gray.png"))
    emit_image("gray.png")
    Image.fromarray(noise(23, 37, 2, 4), "LA").save(path("graya.png"))
    emit_image("graya.png")

    pal = Image.fromarray(noise(23, 37, 1, 5)[..., 0] % 16, "L").convert("P")
    pal.putpalette(list(noise(16, 1, 3, 6).reshape(-1)) + [0] * (768 - 48))
    pal.save(path("pal.png"))
    emit_image("pal.png")
    pal.save(path("pal_trns.png"), transparency=3)
    emit_image("pal_trns.png")

    # 16-bit: a ramp that spans far past 255 plus noise in the low byte, so a
    # clip, a high-byte truncation and a rounding scale all give different bytes.
    g = (np.linspace(0, 65535, 23 * 37).reshape(23, 37, 1)).astype(np.uint16)
    lo = np.random.default_rng(7).integers(0, 256, (23, 37, 4), dtype=np.uint16)
    rgb16 = np.concatenate([g, 65535 - g, (g // 2) + 1000], axis=-1) | (lo[..., :3] & 0xFF)
    rgb16[0, 0] = [65535, 65535, 65535]
    rgb16[0, 1] = [128, 127, 255]
    rgb16[0, 2] = [32767, 32768, 257]
    write_png16(path("rgb16.png"), rgb16, 2)
    emit_image("rgb16.png")
    write_png16(path("gray16.png"), rgb16[..., :1], 0)
    emit_image("gray16.png")
    rgba16 = np.concatenate([rgb16, (lo[..., 3:4] << 8) | lo[..., 3:4]], axis=-1)
    write_png16(path("rgba16.png"), rgba16, 6)
    emit_image("rgba16.png")

    Image.fromarray(rgb, "RGB").save(path("rgb.tiff"))
    emit_image("rgb.tiff")
    Image.fromarray(rgba, "RGBA").save(path("rgba.tiff"))
    emit_image("rgba.tiff")
    Image.fromarray(noise(23, 37, 1, 3)[..., 0], "L").save(path("gray.tiff"))
    emit_image("gray.tiff")
    Image.fromarray(rgb, "RGB").save(path("rgb.bmp"))
    emit_image("rgb.bmp")
    Image.fromarray(noise(23, 37, 1, 3)[..., 0], "L").save(path("gray.bmp"))
    emit_image("gray.bmp")
    Image.fromarray(rgba, "RGBA").save(path("rgba.bmp"))
    emit_image("rgba.bmp")

    # GIF: two frames, only the first counts. A palette of 32 colours.
    f0 = Image.fromarray(photo(23, 37, 11), "RGB").quantize(32, dither=Image.Dither.NONE)
    f1 = Image.fromarray(photo(23, 37, 12), "RGB").quantize(32, dither=Image.Dither.NONE)
    f0.save(path("anim.gif"), save_all=True, append_images=[f1], duration=100, loop=0)
    emit_image("anim.gif")

    # --- colour management --------------------------------------------------
    p3 = open("/System/Library/ColorSync/Profiles/Display P3.icc", "rb").read()
    srgb = open("/System/Library/ColorSync/Profiles/sRGB Profile.icc", "rb").read()
    sat = np.zeros((23, 37, 3), np.uint8)
    sat[..., 0] = np.linspace(0, 255, 37).astype(np.uint8)[None, :]
    sat[..., 1] = np.linspace(255, 0, 23).astype(np.uint8)[:, None]
    sat[..., 2] = 40
    Image.fromarray(sat, "RGB").save(path("p3.png"), icc_profile=p3)
    emit_image("p3.png", tag="colour")
    Image.fromarray(sat, "RGB").save(path("srgb_icc.png"), icc_profile=srgb)
    emit_image("srgb_icc.png", tag="colour")

    # --- JPEG ---------------------------------------------------------------
    ph = photo(48, 64, 21)
    Image.fromarray(ph, "RGB").save(path("rgb.jpg"), quality=90)
    emit_image("rgb.jpg", tag="jpeg")
    Image.fromarray(ph, "RGB").save(path("rgb444.jpg"), quality=90, subsampling=0)
    emit_image("rgb444.jpg", tag="jpeg")
    Image.fromarray(ph[..., 0], "L").save(path("gray.jpg"), quality=90)
    emit_image("gray.jpg", tag="jpeg")
    Image.fromarray(ph, "RGB").convert("CMYK").save(path("cmyk.jpg"), quality=90)
    emit_image("cmyk.jpg", tag="jpeg_cmyk")

    # EXIF orientation 1..8 of an asymmetric image: the file's pixels are the
    # unrotated ones, the tag says how to display them.
    base = photo(24, 40, 31)
    for o in range(1, 9):
        ex = Image.Exif()
        ex[0x0112] = o
        Image.fromarray(base, "RGB").save(path(f"exif{o}.jpg"), quality=95, subsampling=0, exif=ex)
        f = path(f"exif{o}.jpg")
        pil = Image.open(f)
        transposed = np.asarray(ImageOps.exif_transpose(pil).convert("RGB"))
        plain = tv_decode(f)  # torchvision, apply_exif_orientation=False (the ST path)
        with open(path(f"exif{o}.jpg.rgb"), "wb") as out:
            out.write(np.ascontiguousarray(plain).tobytes())
        with open(path(f"exif{o}.jpg.exif.rgb"), "wb") as out:
            out.write(np.ascontiguousarray(transposed).tobytes())
        note("exif", f"exif{o}.jpg", orientation=o, h=plain.shape[0], w=plain.shape[1], th=transposed.shape[0],
             tw=transposed.shape[1])
    # PNG with an eXIf chunk orientation 6 (PIL reads it for exif_transpose).
    ex = Image.Exif()
    ex[0x0112] = 6
    Image.fromarray(base, "RGB").save(path("exif6.png"), exif=ex)
    pil = Image.open(path("exif6.png"))
    transposed = np.asarray(ImageOps.exif_transpose(pil).convert("RGB"))
    with open(path("exif6.png.rgb"), "wb") as out:
        out.write(np.ascontiguousarray(base).tobytes())
    with open(path("exif6.png.exif.rgb"), "wb") as out:
        out.write(np.ascontiguousarray(transposed).tobytes())
    note("exif", "exif6.png", orientation=6, h=base.shape[0], w=base.shape[1], th=transposed.shape[0],
         tw=transposed.shape[1])

    # --- HEIC, encoded by sips from the same photo ---
    os.makedirs(SCRATCH, exist_ok=True)
    Image.fromarray(ph, "RGB").save(os.path.join(SCRATCH, "heic_src.png"))
    subprocess.run(["sips", "-s", "format", "heic", os.path.join(SCRATCH, "heic_src.png"), "--out",
                    path("photo.heic")], check=True, capture_output=True)
    # No Python oracle in this environment decodes HEIC (torchvision's
    # decode_heic needs an extra package), so the reference is the source pixels
    # and the test bounds the lossy round trip.
    with open(path("photo.heic.src.rgb"), "wb") as o:
        o.write(np.ascontiguousarray(ph).tobytes())
    note("heic", "photo.heic", w=ph.shape[1], h=ph.shape[0], oracle="source_pixels")


# ---------------------------------------------------------------------------
# Audio and video are appended below by their own functions.
# ---------------------------------------------------------------------------


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("images", "all"):
        images()
    if what in ("audio", "all"):
        audio()
    if what in ("video", "all"):
        video()
    prior = []
    mf = path("manifest.txt")
    if os.path.exists(mf) and what != "all":
        keep = {"images": ("image", "exif", "heic"), "audio": ("audio",), "video": ("video",)}[what]
        prior = [l for l in open(mf).read().splitlines() if l.split(" ")[0] not in keep]
    with open(mf, "w") as f:
        f.write("\n".join(prior + manifest) + "\n")
    print("wrote", len(prior) + len(manifest), "manifest lines to", mf)
    write_fixture_manifest()


GENERATOR = "scripts/gen_media_fixtures.py"
GOLDEN_SUFFIXES = (".rgb", ".rgba", ".f32", ".rgb24")


def file_table():
    """The role, published status and note of every file now in the fixture directory."""
    table = {}
    for name in sorted(os.listdir(OUT)):
        if name == "MANIFEST.json":
            continue
        if name == "manifest.txt":
            role, note_ = "config", "one line per fixture: kind, name and the facts about it that the tests read"
        elif name.endswith(GOLDEN_SUFFIXES):
            role = "golden"
            note_ = (
                "the reference decoder's output for the input of the same name "
                "(torchvision/PIL for images, ffmpeg for audio and video; see the generator docstring)"
            )
        else:
            role, note_ = "input", "an encoded media file the decoders under test are given"
        table[name] = {"role": role, "published": True, "note": note_}
    return table


def write_fixture_manifest():
    fm.write_manifest(
        OUT,
        GENERATOR,
        "generated: the images are numpy default_rng draws and closed-form gradients encoded by PIL, the "
        "audio is synthesised tones encoded by afconvert, ffmpeg and PIL's encoders, and the video is "
        "solid and quadrant frames encoded by ffmpeg (libx264, libx265). Encoder output is stable for a "
        "given tool version; the goldens are the reference decoders' output for those exact files.",
        file_table(),
        "produced",
    )


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

TARGET_SR = 16000
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
AFCONVERT = "/usr/bin/afconvert"


def tone_a(t, dur, k=1.0):
    """Left channel: tones plus a chirp, all below 5.4 kHz * k. `k` scales every frequency."""
    return (0.35 * np.sin(2 * np.pi * 440 * k * t + 0.3) + 0.2 * np.sin(2 * np.pi * 1250 * k * t + 0.7)
            + 0.15 * np.sin(2 * np.pi * 3300 * k * t)
            + 0.1 * np.sin(2 * np.pi * k * (200 * t + 0.5 * (5200 - 200) / dur * t ** 2)))


def tone_b(t, dur, k=1.0):
    """Right channel: different frequencies, so the mean is not either channel."""
    return (0.3 * np.sin(2 * np.pi * 523.25 * k * t) + 0.25 * np.sin(2 * np.pi * 2000 * k * t + 1.1)
            + 0.1 * np.sin(2 * np.pi * 4700 * k * t + 2.0)
            + 0.1 * np.sin(2 * np.pi * k * (5000 - 4000 * t / dur) * t))


def signal(sr, dur, channels, k=1.0):
    """(frames, channels) float64 at `sr`; mono is the left channel."""
    n = int(round(sr * dur))
    t = np.arange(n) / sr
    cols = [tone_a(t, dur, k)] + ([tone_b(t, dur, k)] if channels == 2 else [])
    return np.stack(cols, axis=-1)


def truth_16k(dur, channels, k=1.0):
    """The exact 16 kHz mono (channel mean) version of `signal`: the band-limited ground truth."""
    return signal(TARGET_SR, dur, channels, k).mean(axis=1)


def write_wav_pcm(file, x, sr, bits):
    """x: (frames, channels) float in [-1, 1) -> integer PCM WAV of `bits` bits (8 is unsigned)."""
    import wave

    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    ch = x.shape[1]
    if bits == 16:
        data = np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2")
        raw = data.tobytes()
    elif bits == 24:
        v = np.clip(np.round(x * 8388608.0), -8388608, 8388607).astype("<i4")
        raw = b"".join(row.tobytes()[:3] for row in v.reshape(-1, 1))
    elif bits == 32:
        raw = np.clip(np.round(x * 2147483648.0), -2147483648, 2147483647).astype("<i4").tobytes()
    elif bits == 8:
        raw = np.clip(np.round(x * 128.0) + 128, 0, 255).astype("u1").tobytes()
    else:
        raise ValueError(bits)
    with wave.open(file, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(bits // 8)
        w.setframerate(sr)
        w.writeframes(raw)


def read_pcm_as_f32(file):
    """What soundfile's `read(dtype='float32')` returns for a PCM WAV: integer / 2^(bits-1)."""
    import wave

    with wave.open(file, "rb") as w:
        ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 1:
        v = (np.frombuffer(raw, "u1").astype(np.int64) - 128) / 128.0
    elif width == 2:
        v = np.frombuffer(raw, "<i2").astype(np.int64) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, "u1").reshape(-1, 3).astype(np.int64)
        v = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16))
        v = np.where(v >= 1 << 23, v - (1 << 24), v) / 8388608.0
    else:
        v = np.frombuffer(raw, "<i4").astype(np.int64) / 2147483648.0
    return v.reshape(-1, ch).astype(np.float32), sr


def save_f32(name, arr):
    with open(path(name), "wb") as o:
        o.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def ffmpeg_audio_16k(file, mean_downmix=False, channels=2):
    """The `-ac 1 -ar 16000` decode that torchcodec's AudioDecoder(sample_rate=16000, num_channels=1) does.

    swresample's default stereo-to-mono matrix is 0.7071 * (L + R) for float output (measured: 7.7 dB
    below the channel mean, which is a factor 1.414). `mean_downmix` asks for the channel mean instead
    (`pan=mono|c0=0.5*c0+0.5*c1`), the librosa and torchaudio convention, with FFmpeg's resampler.
    """
    af = ["-af", "pan=mono|c0=0.5*c0+0.5*c1"] if mean_downmix and channels == 2 else ["-ac", "1"]
    out = subprocess.run([FFMPEG, "-v", "error", "-i", file, *af, "-ar", str(TARGET_SR), "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(out, "<f4").copy()


def torchaudio_16k(x, sr):
    """`load_audio(backend="torchaudio")` after the file is read: channel mean, torchaudio.functional.resample."""
    import torch
    import torchaudio

    w = torch.from_numpy(np.ascontiguousarray(x.astype(np.float32).T)).mean(dim=0)
    if sr != TARGET_SR:
        w = torchaudio.functional.resample(w, orig_freq=sr, new_freq=TARGET_SR)
    return w.numpy().astype(np.float32)


def snr_db(ref, got):
    n = min(len(ref), len(got))
    e = got[:n].astype(np.float64) - ref[:n].astype(np.float64)
    return 10 * np.log10((ref[:n].astype(np.float64) ** 2).sum() / max((e ** 2).sum(), 1e-30))


def audio():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SCRATCH, exist_ok=True)
    DUR = 0.4
    # --- exact: already 16 kHz mono ---------------------------------------
    x = signal(TARGET_SR, DUR, 1)
    for name, bits in [("pcm16_16k_mono.wav", 16), ("pcm24_16k_mono.wav", 24), ("pcm32_16k_mono.wav", 32),
                       ("pcm8_16k_mono.wav", 8)]:
        write_wav_pcm(path(name), x, TARGET_SR, bits)
        ref, _ = read_pcm_as_f32(path(name))
        save_f32(name + ".f32", ref[:, 0])
        note("audio", name, tag="exact", rate=TARGET_SR, channels=1, frames=len(ref))
    import scipy.io.wavfile as wf

    f32 = x[:, 0].astype(np.float32)
    wf.write(path("f32_16k_mono.wav"), TARGET_SR, f32)
    save_f32("f32_16k_mono.wav.f32", f32)
    note("audio", "f32_16k_mono.wav", tag="exact", rate=TARGET_SR, channels=1, frames=len(f32))
    # AIFF (big-endian PCM16) and CAF (little-endian PCM16): the other PCM containers.
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", path("pcm16_16k_mono.wav"), "-c:a", "pcm_s16be",
                    path("pcm16_16k_mono.aiff")], check=True)
    subprocess.run([AFCONVERT, "-f", "caff", "-d", "LEI16", path("pcm16_16k_mono.wav"), path("pcm16_16k_mono.caf")],
                   check=True)
    ref, _ = read_pcm_as_f32(path("pcm16_16k_mono.wav"))
    for name in ("pcm16_16k_mono.aiff", "pcm16_16k_mono.caf"):
        save_f32(name + ".f32", ref[:, 0])
        note("audio", name, tag="exact", rate=TARGET_SR, channels=1, frames=len(ref))
    # FLAC is lossless, so the 16 kHz mono file decodes to exactly the PCM16 values.
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", path("pcm16_16k_mono.wav"), "-c:a", "flac",
                    path("pcm16_16k_mono.flac")], check=True)
    save_f32("pcm16_16k_mono.flac.f32", ref[:, 0])
    note("audio", "pcm16_16k_mono.flac", tag="exact", rate=TARGET_SR, channels=1, frames=len(ref))

    # --- resampled and downmixed: bounded by measurement -------------------
    cases = [("pcm16_44k_stereo.wav", 44100, 2, 16), ("pcm16_48k_stereo.wav", 48000, 2, 16),
             ("pcm16_22k_mono.wav", 22050, 1, 16), ("pcm16_8k_mono.wav", 8000, 1, 16)]
    for name, sr, ch, bits in cases:
        k = 3400 / 5400 if sr == 8000 else 1.0  # an 8 kHz file cannot hold 5 kHz tones
        s = signal(sr, DUR, ch, k)
        write_wav_pcm(path(name), s, sr, bits)
        src, _ = read_pcm_as_f32(path(name))
        truth = truth_16k(DUR, ch, k)
        ff = ffmpeg_audio_16k(path(name))
        ffm = ffmpeg_audio_16k(path(name), mean_downmix=True, channels=ch)
        ta = torchaudio_16k(src, sr)
        if name == "pcm16_44k_stereo.wav":  # FFmpeg's default downmix (0.7071 * (L + R)), kept for one file
            save_f32(name + ".ffmpeg.f32", ff)
        save_f32(name + ".ffmpeg_mean.f32", ffm)
        save_f32(name + ".torchaudio.f32", ta)
        save_f32(name + ".truth.f32", truth)
        print(f"{name}: frames ffmpeg {len(ff)} torchaudio {len(ta)} truth {len(truth)}; "
              f"SNR vs truth ffmpeg(mean) {snr_db(truth, ffm):.1f} dB torchaudio {snr_db(truth, ta):.1f} dB; "
              f"ffmpeg(mean) vs torchaudio {snr_db(ta, ffm):.1f} dB; ffmpeg(default) vs truth {snr_db(truth, ff):.1f} dB")
        note("audio", name, tag="resample", rate=sr, channels=ch, frames=len(src), ffmpeg_frames=len(ff),
             torchaudio_frames=len(ta), truth_frames=len(truth))

    # --- compressed: decode succeeds, length and error bounded by measurement ---
    LONG = 1.0
    s = signal(44100, LONG, 2)
    write_wav_pcm(os.path.join(SCRATCH, "long44.wav"), s, 44100, 16)
    src44 = os.path.join(SCRATCH, "long44.wav")
    subprocess.run([AFCONVERT, "-f", "m4af", "-d", "aac", "-b", "96000", src44, path("tone_44k_stereo.m4a")], check=True)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", src44, "-c:a", "libmp3lame", "-b:a", "128k",
                    path("tone_44k_stereo.mp3")], check=True)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", src44, "-c:a", "flac", path("tone_44k_stereo.flac")], check=True)
    write_wav_pcm(os.path.join(SCRATCH, "long16.wav"), signal(TARGET_SR, LONG, 1), TARGET_SR, 16)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", os.path.join(SCRATCH, "long16.wav"), "-c:a", "libmp3lame",
                    "-b:a", "64k", path("tone_16k_mono.mp3")], check=True)
    subprocess.run([AFCONVERT, "-f", "m4af", "-d", "aac", "-b", "32000", os.path.join(SCRATCH, "long16.wav"),
                    path("tone_16k_mono.m4a")], check=True)
    truth = truth_16k(LONG, 2)
    for name, src_rate, ch in [("tone_44k_stereo.m4a", 44100, 2), ("tone_44k_stereo.mp3", 44100, 2),
                               ("tone_44k_stereo.flac", 44100, 2), ("tone_16k_mono.mp3", 16000, 1),
                               ("tone_16k_mono.m4a", 16000, 1)]:
        ff = ffmpeg_audio_16k(path(name), mean_downmix=True, channels=ch)
        save_f32(name + ".ffmpeg_mean.f32", ff)
        t = truth if ch == 2 else truth_16k(LONG, 1)
        lag = int(np.argmax(np.correlate(ff, t, "full")) - (len(t) - 1))
        print(f"{name}: ffmpeg(mean) frames {len(ff)}, SNR vs truth {snr_db(t, ff):.1f} dB, "
              f"best alignment lag {lag} samples")
        note("audio", name, tag="compressed", rate=src_rate, channels=ch, ffmpeg_frames=len(ff),
             truth_frames=len(t))
    save_f32("tone_stereo_long.truth.f32", truth)
    save_f32("tone_mono_long.truth.f32", truth_16k(LONG, 1))
    # Ogg Opus (macOS decodes it), at 48 kHz inside the container.
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", os.path.join(SCRATCH, "long16.wav"), "-c:a", "libopus",
                    "-b:a", "24k", path("tone_16k_mono.opus")], check=True)
    ff = ffmpeg_audio_16k(path("tone_16k_mono.opus"), channels=1)
    save_f32("tone_16k_mono.opus.ffmpeg_mean.f32", ff)
    note("audio", "tone_16k_mono.opus", tag="compressed", rate=48000, channels=1, ffmpeg_frames=len(ff),
         truth_frames=16000)
    # WavPack: a real audio file in a format AudioToolbox does not decode.
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", path("pcm16_16k_mono.wav"), "-c:a", "wavpack",
                    path("tone_16k_mono.wv")], check=True)
    note("audio", "tone_16k_mono.wv", tag="unsupported")


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

FRAME_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
                (128, 128, 128), (255, 128, 0), (0, 128, 255), (128, 0, 255), (32, 32, 32), (224, 224, 224)]
VW, VH, VFPS = 64, 48, 10


def quadrant_frame(k):
    """An asymmetric frame (four different quadrant colours plus a bar) so rotation is visible."""
    f = np.zeros((VH, VW, 3), np.uint8)
    f[: VH // 2, : VW // 2] = (220, 40, 40)
    f[: VH // 2, VW // 2:] = (40, 200, 60)
    f[VH // 2:, : VW // 2] = (50, 60, 230)
    f[VH // 2:, VW // 2:] = (230, 220, 40)
    f[VH // 2 - 3: VH // 2 + 3, : VW // 4] = (255, 255, 255)
    f[:, :4] = FRAME_COLORS[k % len(FRAME_COLORS)]
    return f


def make_video(name, frames, extra_in=(), extra_out=(), codec="libx264"):
    d = os.path.join(SCRATCH, "frames_" + name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for i, f in enumerate(frames):
        Image.fromarray(f, "RGB").save(os.path.join(d, f"f{i:03d}.png"))
    cmd = [FFMPEG, "-v", "error", "-y", "-framerate", str(VFPS), "-i", os.path.join(d, "f%03d.png"), *extra_in,
           "-c:v", codec, "-pix_fmt", "yuv420p", "-crf", "10", "-g", "5", "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", "-movflags", "+faststart",
           *extra_out, path(name)]
    subprocess.run(cmd, check=True)


def probe_pts(file):
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time",
                          "-of", "csv=p=0", file], check=True, capture_output=True, text=True).stdout
    return sorted(float(x.strip(",")) for x in out.split() if x.strip(","))


def ffmpeg_rgb(file, autorotate=True):
    cmd = [FFMPEG, "-v", "error"]
    if not autorotate:
        cmd.append("-noautorotate")
    cmd += ["-i", file, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def video():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SCRATCH, exist_ok=True)
    solid = [np.full((VH, VW, 3), c, np.uint8) for c in FRAME_COLORS]
    make_video("colors.mp4", solid)
    raw = ffmpeg_rgb(path("colors.mp4"))
    with open(path("colors.mp4.rgb24"), "wb") as o:
        o.write(raw)
    pts = probe_pts(path("colors.mp4"))
    print("colors.mp4: ffmpeg frames", len(raw) // (VW * VH * 3), "pts", pts[:4], "...", pts[-1])
    note("video", "colors.mp4", w=VW, h=VH, frames=len(solid), fps=VFPS, rotation=0,
         pts=",".join(f"{p:.6f}" for p in pts))
    # (The source colours are FRAME_COLORS; tests/media_decode.rs carries the same table.)

    # Asymmetric frames with rotation metadata (a 90 degrees counter-clockwise display matrix).
    quad = [quadrant_frame(k) for k in range(len(FRAME_COLORS))]
    make_video("quad_rot90.mp4", quad)
    rot_tmp = os.path.join(SCRATCH, "quad_rot90_tmp.mp4")
    shutil.copy(path("quad_rot90.mp4"), rot_tmp)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-display_rotation", "90", "-i", rot_tmp, "-c", "copy",
                    path("quad_rot90.mp4")], check=True)
    for auto in (True, False):
        raw = ffmpeg_rgb(path("quad_rot90.mp4"), autorotate=auto)
        with open(path("quad_rot90.mp4" + (".rgb24" if auto else ".stored.rgb24")), "wb") as o:
            o.write(raw)
    note("video", "quad_rot90.mp4", w=VW, h=VH, frames=len(quad), fps=VFPS, rotation=90,
         pts=",".join(f"{p:.6f}" for p in probe_pts(path("quad_rot90.mp4"))))
    # OpenCV (the "opencv" backend of transformers.video_utils.load_video) on the rotated file.
    try:
        import cv2

        cap = cv2.VideoCapture(path("quad_rot90.mp4"))
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        print("cv2 on quad_rot90.mp4: frames", len(frames), "shape", frames[0].shape,
              "fps", cap.get(cv2.CAP_PROP_FPS), "count", cap.get(cv2.CAP_PROP_FRAME_COUNT),
              "orientation_auto", cap.get(cv2.CAP_PROP_ORIENTATION_AUTO))
        with open(path("quad_rot90.mp4.cv2.rgb24"), "wb") as o:
            o.write(b"".join(f.tobytes() for f in frames))
    except Exception as e:
        print("cv2 oracle unavailable:", type(e).__name__, e)

    # HEVC.
    make_video("colors_hevc.mp4", solid, codec="libx265", extra_out=("-tag:v", "hvc1", "-x265-params", "log-level=error"))
    note("video", "colors_hevc.mp4", w=VW, h=VH, frames=len(solid), fps=VFPS, rotation=0,
         pts=",".join(f"{p:.6f}" for p in probe_pts(path("colors_hevc.mp4"))))


if __name__ == "__main__":
    main()
