#!/usr/bin/env python3
"""Fixtures and oracle outputs for tests/media_jpeg.rs (src/media/jpeg/).

    ~/.venvs/ml-2026-10/bin/python scripts/gen_jpeg_fixtures.py
    ~/.venvs/ml-2026-10/bin/python scripts/gen_jpeg_fixtures.py --live-fuzz DIR COUNT SEED

The first form writes tests/fixtures/jpeg/ and its MANIFEST.json (commit this script
first: the manifest names the commit that last touched it). The second writes COUNT
mutated JPEGs and the oracle's verdict on each into DIR, for the ignored live-fuzz
test; nothing in the repository changes.

The oracle is the reference path of google/embeddinggemma-2:

    sentence-transformers -> image processor `fetch_images` ->
    `torchvision.io.decode_image(path, mode=ImageReadMode.RGB)`

whose JPEG decoder is libjpeg-turbo 3.1.3 (the libjpeg 8 build bundled in torchvision
0.29.1's wheel, ISLOW IDCT, fancy upsampling, `apply_exif_orientation=False`).
`decode_image` hands data to it only when the bytes start `FF D8 FF`. PIL 12.3
(libjpeg-turbo 3.1.4.1) and `djpeg` 3.2.0 are cross-checks; where they disagree with
torchvision the manifest says so. Every golden is torchvision's output.

Inputs come from four encoders, so the grid is not shaped by one writer's habits:

  PIL        baseline / progressive / optimized / restart intervals, 4:4:4 4:2:2 4:2:0
             4:1:1, grayscale, Adobe CMYK, quality 1..100
  cjpeg      (libjpeg-turbo 3.2.0) arbitrary sampling factors, arithmetic coding,
             progressive, restart rows, lossless and 12-bit streams
  synth      the coefficient-level baseline writer below: exact control of every
             coefficient, quantizer, table, marker and component layout (YCCK, RGB,
             non-interleaved scans, 16-bit DQT, extreme IDCT inputs, malformed headers)
  mutation   a seeded byte-level mutator over the files above (the fuzz corpus)

Layout of tests/fixtures/jpeg/:

  NAME.jpg      the input
  NAME.jpg.rgb  the oracle's decoded bytes, HWC, when the oracle decodes it
  manifest.txt  one line per case: `case NAME expect=ok|error key=value ...`
  fuzz_corpus.bin  mutated inputs with the oracle's verdicts (format in tests/media_jpeg.rs)
"""

import io
import itertools
import json
import os
import random
import shutil
import struct
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools", "qwen35_ref"))
import fixture_manifest as fm  # noqa: E402

OUT = os.path.join(HERE, "..", "tests", "fixtures", "jpeg")
GENERATOR = "scripts/gen_jpeg_fixtures.py"
CJPEG = shutil.which("cjpeg") or "/opt/homebrew/Cellar/jpeg-turbo/3.2.0/bin/cjpeg"
DJPEG = shutil.which("djpeg") or "/opt/homebrew/Cellar/jpeg-turbo/3.2.0/bin/djpeg"

# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------


def tv_decode(data):
    """torchvision.io.decode_image(mode=RGB) -> HWC uint8 array, or an error string."""
    import torch
    import torchvision.io as tio

    try:
        t = tio.decode_image(torch.frombuffer(bytearray(data), dtype=torch.uint8), mode=tio.ImageReadMode.RGB)
        return t.permute(1, 2, 0).numpy()
    except Exception as e:  # the oracle's refusal is the datum
        return "ERR " + str(e).splitlines()[0].split("decode_jpeg.cpp:171, ")[-1][:100]


def pil_decode(data):
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        return np.asarray(im.convert("RGB"))
    except Exception as e:
        return "ERR " + str(e)[:80]


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


def photo(h, w, seed):
    """Smooth gradients, a disc, hard edges and noise: a stand-in for a photograph."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    ch = [127.5 * (1 + np.sin(2 * np.pi * x / max(w, 1) * (k + 1) + k)) * (0.6 + 0.4 * y / max(h, 1)) for k in range(3)]
    img = np.stack(ch, -1) + rng.normal(0, 10, (h, w, 3))
    disc = (x - 0.6 * w) ** 2 + (y - 0.4 * h) ** 2 < (0.25 * min(h, w)) ** 2
    img[disc] = [240.0, 30.0, 60.0]
    return np.clip(img, 0, 255).astype(np.uint8)


def pil_enc(img, **kw):
    b = io.BytesIO()
    mode = "CMYK" if img.ndim == 3 and img.shape[2] == 4 else None
    (Image.fromarray(img, mode) if mode else Image.fromarray(img)).save(b, "JPEG", **kw)
    return b.getvalue()


def cj(img, *flags):
    h, w = img.shape[:2]
    ppm = b"P6\n%d %d\n255\n" % (w, h) + img.tobytes()
    p = subprocess.run([CJPEG, *flags], input=ppm, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("cjpeg " + " ".join(flags) + ": " + p.stderr.decode().strip())
    return p.stdout


# ---------------------------------------------------------------------------
# synth: a baseline JPEG writer working on quantized coefficients
# ---------------------------------------------------------------------------

ZZ = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
      28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
      54, 47, 55, 62, 63]  # zigzag index -> natural index


def seg(marker, payload):
    return b"\xff" + bytes([marker]) + struct.pack(">H", len(payload) + 2) + payload


# DC table: symbols 0..15, all 5-bit codes. AC table: symbols 0x00..0xFE with 8-bit codes, 0xFF with 9 bits. Between
# them every (run, size) pair and every DC category up to 15 is codable, so any coefficient block can be written.
DC_BITS = [0] * 16
DC_BITS[4] = 16
DC_VALS = list(range(16))
AC_BITS = [0] * 16
AC_BITS[7] = 255
AC_BITS[8] = 1
AC_VALS = list(range(256))


def _codes(bits, vals):
    code, k, out = 0, 0, {}
    for ln in range(1, 17):
        for _ in range(bits[ln - 1]):
            out[vals[k]] = (code, ln)
            code += 1
            k += 1
        code <<= 1
    return out


DC_CODE = _codes(DC_BITS, DC_VALS)
AC_CODE = _codes(AC_BITS, AC_VALS)


def dht(tc, th, bits, vals):
    return seg(0xC4, bytes([(tc << 4) | th]) + bytes(bits) + bytes(vals))


class Bits:
    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.n = 0

    def put(self, code, ln):
        self.acc = (self.acc << ln) | (code & ((1 << ln) - 1))
        self.n += ln
        while self.n >= 8:
            b = (self.acc >> (self.n - 8)) & 0xFF
            self.out.append(b)
            if b == 0xFF:
                self.out.append(0)
            self.n -= 8
        self.acc &= (1 << self.n) - 1

    def flush(self):
        if self.n:
            self.put((1 << (8 - self.n)) - 1, 8 - self.n)


def _encode_block(bw, blk_zz, prev_dc):
    d = int(blk_zz[0]) - prev_dc
    c = abs(d).bit_length()
    bw.put(*DC_CODE[c])
    if c:
        bw.put(d if d > 0 else d + (1 << c) - 1, c)
    run = 0
    last = max([i for i in range(1, 64) if blk_zz[i] != 0], default=0)
    for i in range(1, last + 1):
        v = int(blk_zz[i])
        if v == 0:
            run += 1
            continue
        while run > 15:
            bw.put(*AC_CODE[0xF0])
            run -= 16
        c = abs(v).bit_length()
        bw.put(*AC_CODE[(run << 4) | c])
        bw.put(v if v > 0 else v + (1 << c) - 1, c)
        run = 0
    if last != 63:
        bw.put(*AC_CODE[0x00])
    return int(blk_zz[0])


def write_jpeg(width, height, comps, coefs, qtabs, *, adobe=None, jfif=False, restart=0, sof=0xC0,
               interleave=True, extra_before_sof=b"", precision=8, comp_ids=None, qprecision=0, eoi=True):
    """comps: [(h, v, tq)]. coefs[c]: int array [block rows, block cols, 64] in natural order, padded to the MCU
    grid. qtabs: {tq: 64 values in natural order}."""
    out = bytearray(b"\xff\xd8")
    if jfif:
        out += seg(0xE0, b"JFIF\0\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    if adobe is not None:
        out += seg(0xEE, b"Adobe" + struct.pack(">HHHB", 100, 0, 0, adobe))
    out += extra_before_sof
    for tq, q in qtabs.items():
        zz = [q[ZZ[i]] for i in range(64)]
        if qprecision:
            out += seg(0xDB, bytes([(1 << 4) | tq]) + b"".join(struct.pack(">H", v) for v in zz))
        else:
            out += seg(0xDB, bytes([tq]) + bytes(zz))
    ids = comp_ids or list(range(1, len(comps) + 1))
    sofp = bytes([precision]) + struct.pack(">HH", height, width) + bytes([len(comps)])
    for i, (h, v, tq) in enumerate(comps):
        sofp += bytes([ids[i], ((h & 15) << 4) | (v & 15), tq])
    out += seg(sof, sofp)
    out += dht(0, 0, DC_BITS, DC_VALS) + dht(1, 0, AC_BITS, AC_VALS)
    if restart:
        out += seg(0xDD, struct.pack(">H", restart))
    hmax = max(c[0] for c in comps)
    vmax = max(c[1] for c in comps)
    mcux = -(-width // (8 * hmax))
    mcuy = -(-height // (8 * vmax))

    def scan(cidx, units):
        hdr = bytes([len(cidx)]) + b"".join(bytes([ids[c], 0x00]) for c in cidx) + bytes([0, 63, 0])
        bw = Bits()
        prev = {c: 0 for c in cidx}
        n = rst = 0
        for unit in units:
            if restart and n and n % restart == 0:
                bw.flush()
                bw.out += bytes([0xFF, 0xD0 + (rst & 7)])
                rst += 1
                prev = {c: 0 for c in cidx}
            for (c, by, bx) in unit:
                blk = coefs[c][by, bx]
                prev[c] = _encode_block(bw, [blk[ZZ[i]] for i in range(64)], prev[c])
            n += 1
        bw.flush()
        return seg(0xDA, hdr) + bytes(bw.out)

    if interleave or len(comps) == 1:
        units = []
        for my in range(mcuy):
            for mx in range(mcux):
                units.append([(c, my * v + y, mx * h + x) for c, (h, v, _) in enumerate(comps)
                              for y in range(v) for x in range(h)])
        out += scan(list(range(len(comps))), units)
    else:
        for c, (h, v, _) in enumerate(comps):
            cw = -(-(-(-width * h // hmax)) // 8)
            chh = -(-(-(-height * v // vmax)) // 8)
            out += scan([c], [[(c, by, bx)] for by in range(chh) for bx in range(cw)])
    if eoi:
        out += b"\xff\xd9"
    return bytes(out)


def rand_coefs(rng, comps, w, h, dc=80, ac=5, nac=8):
    hmax = max(c[0] for c in comps)
    vmax = max(c[1] for c in comps)
    mcux, mcuy = -(-w // (8 * hmax)), -(-h // (8 * vmax))
    out = []
    for (ch, cv, _) in comps:
        a = np.zeros((mcuy * cv, mcux * ch, 64), dtype=np.int32)
        a[:, :, 0] = rng.integers(-dc, dc, size=a.shape[:2])
        for k in range(1, nac):
            a[:, :, k] = rng.integers(-ac, ac, size=a.shape[:2]) // (1 + k // 3)
        out.append(a)
    return out


def flat_coefs(vals, w, h, comps):
    """Coefficients that decode to a flat 8x8 block of value `vals[block row, block col, comp]` (q = 1)."""
    out = []
    for c in range(len(comps)):
        a = np.zeros((vals.shape[0], vals.shape[1], 64), dtype=np.int32)
        a[:, :, 0] = (vals[:, :, c].astype(np.int32) - 128) * 8
        out.append(a)
    return out


# ---------------------------------------------------------------------------
# Header surgery
# ---------------------------------------------------------------------------


def segments(b):
    """(marker, start, end) of every header segment up to and including the first SOS."""
    i, out = 2, []
    while i < len(b) - 1:
        m = b[i + 1]
        if m == 0xDA:
            out.append((m, i, len(b)))
            break
        ln = struct.unpack(">H", b[i + 2:i + 4])[0]
        out.append((m, i, i + 2 + ln))
        i += 2 + ln
    return out


def strip_dht(b):
    out = bytearray(b[:2])
    for m, s, e in segments(b):
        if m != 0xC4:
            out += b[s:e]
    return bytes(out)


def scan_offsets(b):
    """Offsets of every SOS marker of a (progressive) file."""
    pos, i = [], 2
    while i < len(b) - 1:
        if b[i] != 0xFF:
            i += 1
            continue
        m = b[i + 1]
        if m in (0x00, 0xFF) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m == 0xD9:
            break
        ln = struct.unpack(">H", b[i + 2:i + 4])[0]
        if m == 0xDA:
            pos.append(i)
        i += 2 + ln
        if m == 0xDA:
            while i < len(b) - 1 and not (b[i] == 0xFF and b[i + 1] != 0x00 and not (0xD0 <= b[i + 1] <= 0xD7)):
                i += 1
    return pos


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------

QT = {0: [6 + (i % 8) for i in range(64)], 1: [10] * 64, 2: [3] * 64}
CASES = []  # (name, bytes, klass, note)


def case(name, data, klass, note="", tessl="equal"):
    assert name not in {c[0] for c in CASES}, name
    CASES.append((name, data, klass, note, tessl))


def build_cases():
    img = photo(33, 47, 5)
    # --- subsampling x coding ------------------------------------------------
    for s in ["4:4:4", "4:2:2", "4:2:0", "4:1:1"]:
        t = s.replace(":", "")
        case(f"sub_{t}_baseline", pil_enc(img, quality=75, subsampling=s), "subsampling", s)
        case(f"sub_{t}_progressive", pil_enc(img, quality=75, subsampling=s, progressive=True), "progressive", s)
    case("sub_440_baseline", cj(img, "-sample", "1x2,1x1,1x1"), "subsampling", "4:4:0 (cjpeg)")
    case("sub_440_progressive", cj(img, "-progressive", "-sample", "1x2,1x1,1x1"), "progressive", "4:4:0")
    samples = ["2x1,1x1,1x1", "1x2,1x1,1x1", "4x1,1x1,1x1", "1x4,1x1,1x1", "4x2,1x1,1x1", "2x4,1x1,1x1", "3x1,1x1,1x1",
               "3x3,1x1,1x1", "1x1,2x2,2x2", "2x1,1x2,1x1", "2x2,2x1,1x2", "4x4,2x2,1x1", "4x4,1x1,1x1", "2x2,1x1,2x2",
               "3x2,1x1,1x1", "4x3,2x1,1x1", "1x3,1x1,1x1"]
    img2 = photo(31, 33, 8)
    for s in samples:
        tag = s.replace(",", "_")
        try:
            data = cj(img2, "-sample", s, "-quality", "80")
        except RuntimeError as e:  # cjpeg refuses an MCU over 10 blocks; the over-limit ones are covered by the err_mcu_over_10_blocks case
            print("cjpeg cannot encode", s, "-", e)
            continue
        case(f"samp_{tag}", data, "sampling", f"cjpeg -sample {s}")
    for s in ["2x2,1x1,1x1", "1x1,2x2,2x2", "4x2,1x1,1x1", "2x1,1x2,1x1"]:
        case(f"samp_{s.replace(',', '_')}_prog", cj(img2, "-sample", s, "-progressive"), "sampling", f"progressive {s}")
    # --- sizes -----------------------------------------------------------------
    for (h, w) in [(1, 1), (1, 9), (9, 1), (2, 2), (3, 3), (7, 13), (13, 7), (8, 8), (9, 9), (17, 15), (65, 129),
                   (3, 4097), (4097, 3)]:
        im = photo(h, w, h * 31 + w)
        case(f"size_{h}x{w}_420", pil_enc(im, quality=80, subsampling="4:2:0"), "size", f"{h}x{w} 4:2:0")
        case(f"size_{h}x{w}_444", pil_enc(im, quality=80, subsampling="4:4:4"), "size", f"{h}x{w} 4:4:4")
        case(f"size_{h}x{w}_prog", pil_enc(im, quality=80, subsampling="4:2:0", progressive=True), "size", f"{h}x{w} progressive")
        case(f"size_{h}x{w}_gray", pil_enc(np.asarray(Image.fromarray(im).convert("L")), quality=80), "size", f"{h}x{w} gray")
    # --- restart markers ---------------------------------------------------------
    for r in (1, 2, 3, 7):
        case(f"rst_{r}_420", pil_enc(img, quality=80, subsampling="4:2:0", restart_marker_blocks=r), "restart", f"{r} MCUs")
    case("rst_rows_cjpeg", cj(img, "-restart", "1B"), "restart", "cjpeg -restart 1B")
    case("rst_progressive", cj(img, "-progressive", "-restart", "2"), "restart", "progressive + DRI")
    case("rst_arith", cj(img, "-arithmetic", "-restart", "2"), "restart", "arithmetic + DRI")
    case("rst_arith_progressive", cj(img, "-arithmetic", "-progressive", "-restart", "3"), "restart", "arithmetic progressive + DRI")
    # --- grayscale, quality, tables ------------------------------------------------
    g = np.asarray(Image.fromarray(img).convert("L"))
    case("gray_baseline", pil_enc(g, quality=75), "grayscale")
    case("gray_progressive", pil_enc(g, quality=75, progressive=True), "grayscale")
    case("gray_arith", cj(img, "-arithmetic", "-grayscale"), "grayscale")
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 100):
        case(f"quality_{q}_420", pil_enc(img, quality=q, subsampling="4:2:0"), "quality")
    case("quality_100_444", pil_enc(img, quality=100, subsampling=0), "quality")
    case("optimized_tables", pil_enc(img, quality=60, optimize=True), "tables", "per-image Huffman tables")
    base = pil_enc(img, quality=75, subsampling="4:2:0")
    case("std_tables_only", strip_dht(base), "tables", "DHT removed: libjpeg-turbo installs the standard tables")
    case("std_tables_only_gray", strip_dht(pil_enc(g, quality=75)), "tables", "gray, DHT removed")
    case("sof1_extended", base.replace(b"\xff\xc0", b"\xff\xc1", 1), "markers", "SOF1 marker on an 8-bit stream")
    # --- arithmetic ------------------------------------------------------------------
    case("arith_sequential_420", cj(img, "-arithmetic", "-sample", "2x2,1x1,1x1"), "arithmetic")
    case("arith_sequential_444", cj(img, "-arithmetic", "-sample", "1x1,1x1,1x1"), "arithmetic")
    case("arith_progressive", cj(img, "-arithmetic", "-progressive"), "arithmetic")
    case("arith_progressive_gray", cj(img, "-arithmetic", "-progressive", "-grayscale"), "arithmetic")
    case("arith_422", cj(img, "-arithmetic", "-sample", "2x1,1x1,1x1"), "arithmetic")
    # --- colour spaces -----------------------------------------------------------------
    rng = np.random.default_rng(3)
    W, H = 24, 16
    c3 = [(2, 2, 0), (1, 1, 1), (1, 1, 1)]
    c3f = [(1, 1, 0)] * 3
    c4 = [(1, 1, 0)] * 4
    case("cmyk_pil_adobe", pil_enc(np.asarray(Image.fromarray(img).convert("CMYK")), quality=80), "colour", "Adobe CMYK from PIL")
    for tr in (None, 0, 2, 3):
        case(f"cmyk4_adobe_{tr}", write_jpeg(W, H, c4, rand_coefs(rng, c4, W, H), QT, adobe=tr), "colour",
             f"4 components, Adobe transform {tr}")
    for tr in (None, 0, 1, 2, 9):
        for jf in (False, True):
            case(f"rgb3_adobe_{tr}_jfif{int(jf)}", write_jpeg(W, H, c3f, rand_coefs(rng, c3f, W, H), QT, adobe=tr, jfif=jf),
                 "colour", f"3 components, Adobe {tr}, JFIF {jf}")
    case("rgb3_ids_RGB", write_jpeg(W, H, c3f, rand_coefs(rng, c3f, W, H), QT, comp_ids=[82, 71, 66]), "colour", "component ids R G B")
    case("rgb3_ids_other", write_jpeg(W, H, c3f, rand_coefs(rng, c3f, W, H), QT, comp_ids=[7, 8, 9]), "colour", "unrecognised ids")
    # flat YCbCr blocks over the colour cube
    for y in (0, 128, 255):
        nb = 20
        cb, cr = np.meshgrid(np.linspace(0, 255, nb).astype(int), np.linspace(0, 255, nb).astype(int), indexing="ij")
        vals = np.stack([np.full((nb, nb), y), cb, cr], -1)
        case(f"ycc_cube_y{y}", write_jpeg(nb * 8, nb * 8, c3f, flat_coefs(vals, nb * 8, nb * 8, c3f), {0: [1] * 64}, jfif=True),
             "colour", f"flat blocks, Y={y}, Cb and Cr on a 20x20 grid")
    nb = 16
    vals = np.random.default_rng(9).integers(0, 256, size=(nb, nb, 4))
    case("ycck_flat", write_jpeg(nb * 8, nb * 8, c4, flat_coefs(vals, nb * 8, nb * 8, c4), {0: [1] * 64}, adobe=2),
         "colour", "flat YCCK blocks")
    case("cmyk_flat", write_jpeg(nb * 8, nb * 8, c4, flat_coefs(vals, nb * 8, nb * 8, c4), {0: [1] * 64}, adobe=0),
         "colour", "flat CMYK blocks")
    # --- synth layouts -------------------------------------------------------------------
    case("noninterleaved_scans", write_jpeg(W, H, c3, rand_coefs(rng, c3, W, H), QT, interleave=False, jfif=True), "layout", "one scan per component")
    case("dqt_16bit", write_jpeg(W, H, c3, rand_coefs(rng, c3, W, H), QT, qprecision=1), "tables", "16-bit quantizers")
    for r in (1, 2, 5):
        case(f"synth_rst_{r}", write_jpeg(W, H, c3, rand_coefs(rng, c3, W, H), QT, restart=r), "restart", f"{r} MCUs (synth)")
    # --- extreme coefficients: the IDCT's 16-bit lanes and saturating output ----------------
    one = [(1, 1, 0)]
    r2 = np.random.default_rng(21)
    for name, scale, q in (("dense32767_q255", 32767, 255), ("dense32767_q1", 32767, 1), ("dense2000_q40", 2000, 40),
                           ("dense1023_q200", 1023, 200)):
        nbx, nby = 6, 5
        a = r2.integers(-scale, scale + 1, size=(nby, nbx, 64)).astype(np.int32)
        a[:, :, 0] = np.clip(a[:, :, 0], -2047, 2047)
        case(f"idct_{name}", write_jpeg(nbx * 8, nby * 8, one, [a], {0: [q] * 64}), "idct", f"{scale} coefficients, quantizer {q}")
    nbx, nby = 6, 5
    a = r2.integers(-32767, 32768, size=(nby, nbx, 64)).astype(np.int32)
    a[:, :, 0] = np.clip(a[:, :, 0], -2047, 2047)
    case("idct_q65535", write_jpeg(nbx * 8, nby * 8, one, [a], {0: [int(x) for x in r2.integers(1, 65536, 64)]}, qprecision=1), "idct", "16-bit quantizers up to 65535")
    dc = np.zeros((4, 8, 64), dtype=np.int32)
    dc[:, :, 0] = np.linspace(-2047, 2047, 32).reshape(4, 8).astype(np.int32)
    case("idct_dc_ramp_q16", write_jpeg(64, 32, one, [dc], {0: [16] * 64}), "idct", "DC ramp whose dequantized value saturates and wraps")
    # --- header surgery that the reference accepts -----------------------------------------
    sosat = base.find(b"\xff\xda")
    case("marker_com_app_before_sof", base[:2] + seg(0xFE, b"hello") + seg(0xE5, b"x" * 40) + seg(0xE1, b"Exif\0\0junk") + base[2:], "markers", "COM and unknown APPn segments")
    case("marker_fill_ff", base[:sosat] + b"\xff\xff\xff" + base[sosat:], "markers", "FF fill bytes before SOS")
    case("marker_trailing_garbage", base + b"\x00\x01garbage after EOI", "markers", "bytes after the EOI")
    case("marker_garbage_before_eoi", base[:-2] + b"zzz" + base[-2:], "markers", "extraneous bytes before the EOI")
    case("marker_dnl", base[:-2] + b"\xff\xdc\x00\x04\x00\x10\xff\xd9", "markers", "DNL after the scan")
    case("marker_dri_zero", base[:2] + seg(0xDD, b"\x00\x00") + base[2:], "markers", "DRI 0")
    case("marker_dup_eoi", base + b"\xff\xd9", "markers", "two EOIs")
    sy = write_jpeg(W, H, c3, rand_coefs(rng, c3, W, H), QT, interleave=False)
    sp = [i for i in range(len(sy) - 1) if sy[i] == 0xFF and sy[i + 1] == 0xDA]
    for order in itertools.permutations([1, 2, 3]):
        j = bytearray(sy)
        for p, cid in zip(sp, order):
            j[p + 5] = cid
        case("scan_order_" + "".join(map(str, order)), bytes(j), "layout", f"non-interleaved scans in component order {order}")
    # --- progressive files that stop early: libjpeg-turbo smooths them ------------------------
    pg = cj(photo(40, 56, 9), "-grayscale", "-progressive")
    pos = scan_offsets(pg)
    for k in (1, 2, 3):
        case(f"smoothing_gray_{k}scans", pg[:pos[k]] + b"\xff\xd9", "smoothing",
             f"progressive grayscale cut after {k} scans + EOI: block smoothing", tessl="refused")
    pc = cj(photo(40, 56, 9), "-progressive", "-sample", "2x2,1x1,1x1")
    pos = scan_offsets(pc)
    for k in (1, 4):
        case(f"smoothing_420_{k}scans", pc[:pos[k]] + b"\xff\xd9", "smoothing", f"4:2:0 cut after {k} scans + EOI", tessl="refused")
    # --- lossless: the reference decodes it, tessl names the refusal ---------------------------
    case("lossless_sof3", cj(photo(16, 24, 2), "-lossless", "1"), "lossless", "SOF3, 8-bit, predictor 1", tessl="refused")
    # --- the reference refuses these, so must tessl ---------------------------------------------
    case("err_12bit_cjpeg", cj(photo(16, 24, 2), "-precision", "12"), "error", "12-bit precision")
    case("err_precision_12_sof0", write_jpeg(W, H, c3, rand_coefs(rng, c3, W, H), QT, precision=12), "error", "SOF0 with 12-bit precision")
    case("err_precision_16", write_jpeg(W, H, c3, rand_coefs(rng, c3, W, H), QT, precision=16), "error", "precision 16")
    case("err_no_eoi", base[:-2], "error", "no EOI: the reference's source manager fails at the end of the data")
    for frac in (0.1, 0.3, 0.5, 0.8, 0.97):
        case(f"err_cut_{int(frac * 100)}", base[:int(len(base) * frac)], "error", f"cut at {int(frac * 100)}%")
    case("err_cut_progressive_half", (lambda p: p[:len(p) // 2])(pil_enc(img, quality=75, progressive=True)), "error", "progressive cut in the middle")
    case("err_header_only", base[:sosat + 14], "error", "stops after the SOS header")
    for nc in (2, 5, 6, 10):
        comps = [(1, 1, 0)] * nc
        case(f"err_{nc}_components", write_jpeg(W, H, comps, rand_coefs(rng, comps, W, H), QT), "error", f"{nc} components: no RGB conversion")
    case("err_fractional_sampling_3_2_1", write_jpeg(W, H, [(3, 1, 0), (2, 1, 1), (1, 1, 1)], rand_coefs(rng, [(3, 1, 0), (2, 1, 1), (1, 1, 1)], W, H), QT), "error", "3:2 sampling ratio")
    c12 = [(2, 2, 0), (2, 2, 1), (2, 2, 1)]
    c10 = [(2, 2, 0), (2, 2, 1), (2, 1, 1)]
    case("err_mcu_over_10_blocks", write_jpeg(W, H, c12, rand_coefs(rng, c12, W, H), QT), "error", "12 blocks in an MCU")
    case("layout_mcu_10_blocks", write_jpeg(W, H, c10, rand_coefs(rng, c10, W, H), QT), "layout", "exactly 10 blocks in an MCU")
    case("err_sampling_zero", write_jpeg(W, H, [(0, 1, 0), (1, 1, 1), (1, 1, 1)], [np.zeros((4, 8, 64), dtype=np.int32)] * 3, QT), "error", "sampling factor 0")
    case("err_sampling_5", write_jpeg(W, H, [(5, 1, 0), (1, 1, 1), (1, 1, 1)], [np.zeros((4, 8, 64), dtype=np.int32)] * 3, QT), "error", "sampling factor 5")
    case("err_zero_width", write_jpeg(0, H, c3f, rand_coefs(rng, c3f, 8, H), QT), "error", "zero width")
    case("err_missing_dqt", write_jpeg(W, H, c3, rand_coefs(rng, c3, W, H), {0: QT[0]}), "error", "quantizer table 1 undefined")
    case("err_dht_missing_progressive", strip_dht(pil_enc(img, quality=75, progressive=True)), "error", "progressive stream without DHT")
    case("err_bad_signature", b"\xff\xd8\x00" + base[3:], "error", "FF D8 then not FF: not handed to libjpeg")
    case("err_png_signature", b"\x89PNG\r\n\x1a\n" + base, "error", "not a JPEG")
    case("err_unknown_marker_f0", base[:2] + b"\xff\xf0\x00\x04ab" + base[2:], "error", "marker F0 (JPG0)")
    case("err_second_sof", base[:sosat] + base[base.find(b"\xff\xc0"):base.find(b"\xff\xc0") + 19] + base[sosat:], "error", "two SOF segments")
    case("err_scan_before_sof", base[:2] + base[sosat:], "error", "SOS before SOF")
    case("err_sof_hierarchical_c5", base.replace(b"\xff\xc0", b"\xff\xc5", 1), "error", "differential sequential SOF5")
    case("err_sof11_lossless_arith", base.replace(b"\xff\xc0", b"\xff\xcb", 1), "error", "SOF11")
    case("err_dht_overfull", base[:sosat] + seg(0xC4, b"\x00" + bytes([255] + [255] * 15) + bytes(range(256))[:10]) + base[sosat:], "error", "Huffman table with more codes than symbols")
    case("err_dqt_bad_len", base[:2] + b"\xff\xdb\x00\x05\x00\x01\x02" + base[2:], "error", "DQT length wrong")
    case("err_progression_bad_params", scan_bad_progression(), "error", "progressive scan with Ss > Se")


def scan_bad_progression():
    p = bytearray(cj(photo(24, 32, 5), "-progressive", "-grayscale"))
    pos = scan_offsets(bytes(p))
    s = pos[1]
    # SOS: FF DA len(2) n(1) [id,tbl] x n Ss Se AhAl
    n = p[s + 4]
    ss_at = s + 5 + 2 * n
    p[ss_at], p[ss_at + 1] = 9, 3
    return bytes(p)


# ---------------------------------------------------------------------------
# Fuzz corpus
# ---------------------------------------------------------------------------

MARKERS = [0x00, 0x01, 0xC0, 0xC2, 0xC4, 0xD0, 0xD1, 0xD2, 0xD7, 0xD8, 0xD9, 0xDA, 0xDB, 0xDD, 0xE0, 0xE1, 0xEE, 0xFE,
           0xFF, 0xC9, 0xCC, 0xDC]


def mutate(rng, b):
    b = bytearray(b)
    kind = rng.choice(["flip", "replace", "replace", "insert", "delete", "marker", "header", "swap", "dup", "bitflip",
                       "zero", "ff"])
    n = rng.choice([1, 1, 1, 2, 3, 6])
    sos = bytes(b).find(b"\xff\xda")
    for _ in range(n):
        if kind == "header":
            p = rng.randrange(2, max(3, min(len(b), (sos if sos > 0 else 200) + 14)))
            b[p] = rng.randrange(256)
        elif kind in ("flip", "replace"):
            b[rng.randrange(2, len(b))] = rng.randrange(256)
        elif kind == "bitflip":
            b[rng.randrange(2, len(b))] ^= 1 << rng.randrange(8)
        elif kind == "zero":
            b[rng.randrange(2, len(b))] = 0
        elif kind == "ff":
            b[rng.randrange(2, len(b))] = 0xFF
        elif kind == "insert":
            b.insert(rng.randrange(2, len(b)), rng.randrange(256))
        elif kind == "delete":
            if len(b) > 4:
                del b[rng.randrange(2, len(b))]
        elif kind == "marker":
            p = rng.randrange(2, len(b))
            b[p:p] = bytes([0xFF, rng.choice(MARKERS)])
        elif kind == "swap":
            i, j = rng.randrange(2, len(b)), rng.randrange(2, len(b))
            b[i], b[j] = b[j], b[i]
        elif kind == "dup":
            i = rng.randrange(2, len(b))
            b[i:i] = b[i:i + rng.randrange(1, 40)]
    return bytes(b)


def fuzz_bases():
    """Small bases covering every coding and layout."""
    img = photo(24, 40, 6)
    g = np.asarray(Image.fromarray(img).convert("L"))
    bases = {}
    for s in ["4:4:4", "4:2:2", "4:2:0", "4:1:1"]:
        bases["pil" + s.replace(":", "")] = pil_enc(img, quality=70, subsampling=s)
    bases["prog420"] = pil_enc(img, quality=70, subsampling="4:2:0", progressive=True)
    bases["prog444"] = pil_enc(img, quality=70, subsampling="4:4:4", progressive=True)
    bases["rst"] = pil_enc(img, quality=70, subsampling="4:2:0", restart_marker_blocks=2)
    bases["gray"] = pil_enc(g, quality=70)
    bases["cmyk"] = pil_enc(np.asarray(Image.fromarray(img).convert("CMYK")), quality=70)
    for name, fl in {"s440": ["-sample", "1x2,1x1,1x1"], "s2x1_1x2": ["-sample", "2x1,1x2,1x1"], "s3x1": ["-sample", "3x1,1x1,1x1"],
                     "s4x2": ["-sample", "4x2,1x1,1x1"], "prog_rst": ["-progressive", "-restart", "2"],
                     "arith": ["-arithmetic"], "arith_prog": ["-arithmetic", "-progressive"],
                     "arith_rst": ["-arithmetic", "-restart", "2"], "arith_prog_rst": ["-arithmetic", "-progressive", "-restart", "3"],
                     "arith_gray": ["-arithmetic", "-grayscale"], "prog440": ["-progressive", "-sample", "1x2,1x1,1x1"],
                     "rgb": ["-rgb"], "prog_gray": ["-grayscale", "-progressive"]}.items():
        bases[name] = cj(img, *fl)
    rng = np.random.default_rng(1)
    W, H = 24, 24
    c3 = [(2, 2, 0), (1, 1, 1), (1, 1, 1)]
    c4 = [(1, 1, 0)] * 4
    bases["synth_noninterleaved"] = write_jpeg(W, H, c3, rand_coefs(rng, c3, W, H), QT, interleave=False, jfif=True)
    bases["synth_ycck"] = write_jpeg(W, H, c4, rand_coefs(rng, c4, W, H), QT, adobe=2)
    bases["synth_cmyk"] = write_jpeg(W, H, c4, rand_coefs(rng, c4, W, H), QT, adobe=0)
    bases["synth_rst1"] = write_jpeg(W, H, c3, rand_coefs(rng, c3, W, H), QT, restart=1)
    return bases


def fuzz_stream(seed):
    """Yield (base name, mutant bytes) forever, round-robin over the bases."""
    rng = random.Random(seed)
    bases = fuzz_bases()
    names = sorted(bases)
    while True:
        for n in names:
            yield n, mutate(rng, bases[n])


def build_fuzz_corpus(path, per_base_ok=12, per_base_err=6, seed=2026):
    bases = fuzz_bases()
    rng = random.Random(seed)
    entries, seen = [], set()
    stats = {}
    for name in sorted(bases):
        ok = err = 0
        tries = 0
        while (ok < per_base_ok or err < per_base_err) and tries < 4000:
            tries += 1
            m = mutate(rng, bases[name])
            if m in seen:
                continue
            t = tv_decode(m)
            is_err = isinstance(t, str)
            if is_err and err >= per_base_err or (not is_err and ok >= per_base_ok):
                continue
            seen.add(m)
            entries.append((m, None if is_err else t))
            if is_err:
                err += 1
            else:
                ok += 1
        stats[name] = (ok, err)
    with open(path, "wb") as f:
        f.write(b"JFZ1")
        f.write(struct.pack("<I", len(entries)))
        for data, t in entries:
            f.write(struct.pack("<I", len(data)))
            f.write(data)
            if t is None:
                f.write(b"\x00")
            else:
                h, w = t.shape[:2]
                f.write(b"\x01" + struct.pack("<II", w, h))
                f.write(np.ascontiguousarray(t).tobytes())
    return len(entries), stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def path(name):
    return os.path.join(OUT, name)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--live-fuzz":
        d, count, seed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        os.makedirs(d, exist_ok=True)
        gen = fuzz_stream(seed)
        ok = 0
        for i in range(count):
            _, m = next(gen)
            t = tv_decode(m)
            open(f"{d}/mut_{i:06d}.jpg", "wb").write(m)
            if isinstance(t, str):
                open(f"{d}/mut_{i:06d}.err", "w").write(t)
            else:
                ok += 1
                open(f"{d}/mut_{i:06d}.rgb", "wb").write(np.ascontiguousarray(t).tobytes())
                open(f"{d}/mut_{i:06d}.dim", "w").write(f"{t.shape[1]} {t.shape[0]}")
        print(f"wrote {count} mutants ({ok} decoded by the oracle) to {d}")
        return

    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(path(f))
    build_cases()
    lines, files = [], {}
    n_ok = n_err = 0
    pil_differs = 0
    for name, data, klass, note, tessl in CASES:
        with open(path(name + ".jpg"), "wb") as f:
            f.write(data)
        files[name + ".jpg"] = {"role": "input", "published": True, "note": f"{klass}: {note}" if note else klass}
        t = tv_decode(data)
        if isinstance(t, str):
            assert klass == "error", f"{name}: the oracle refuses a case not filed under error: {t}"
            lines.append(f"case {name} expect=error class={klass} oracle={t.replace(' ', '_')}")
            n_err += 1
            continue
        assert klass != "error", f"{name}: filed under error but the oracle decodes it"
        h, w = t.shape[:2]
        with open(path(name + ".jpg.rgb"), "wb") as f:
            f.write(np.ascontiguousarray(t).tobytes())
        files[name + ".jpg.rgb"] = {"role": "golden", "published": True, "note": "torchvision.io.decode_image(mode=RGB) of the input of the same name"}
        p = pil_decode(data)
        pil = "error" if isinstance(p, str) else ("equal" if np.array_equal(p, t) else "differs")
        pil_differs += pil == "differs"
        lines.append(f"case {name} expect=ok tessl={tessl} class={klass} w={w} h={h} pil={pil}")
        n_ok += 1
    n_fuzz, stats = build_fuzz_corpus(path("fuzz_corpus.bin"))
    files["fuzz_corpus.bin"] = {"role": "golden", "published": True, "note": "mutated JPEGs with the oracle's verdict and pixels (format in tests/media_jpeg.rs)"}
    with open(path("manifest.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    files["manifest.txt"] = {"role": "config", "published": True, "note": "one line per case: expectation, class, size"}
    print(f"{n_ok} cases decoded by the oracle, {n_err} refused; {n_fuzz} fuzz entries; PIL differs on {pil_differs}")
    print("fuzz entries per base (ok, error):", stats)
    fm.write_manifest(
        OUT,
        GENERATOR,
        "generated: images are numpy default_rng draws and closed-form gradients encoded by PIL 12.3 (libjpeg-turbo "
        "3.1.4.1), cjpeg 3.2.0 (libjpeg-turbo) and this script's coefficient-level writer; the fuzz corpus is a "
        "seeded byte mutator over those files (random.Random, seed 2026). Encoder output is stable for a given tool "
        "version; every golden is torchvision 0.29.1's decode (libjpeg-turbo 3.1.3) of the exact input file.",
        files,
        "produced",
    )


if __name__ == "__main__":
    main()
