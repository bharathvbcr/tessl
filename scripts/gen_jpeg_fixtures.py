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
             non-interleaved scans, 16-bit DQT, extreme IDCT inputs, malformed headers);
             a progressive writer whose AC data is all zero and whose scan list sets the
             progression state (the probes of libjpeg-turbo's block smoothing); and a
             lossless (SOF3) writer with every predictor, point transform, restart
             interval and arbitrary differences
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
import math
import os
import random
import shutil
import struct
import subprocess
import zlib
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
# synth: a progressive writer whose AC data is all zero (block smoothing probes)
# ---------------------------------------------------------------------------


def write_progressive_gray(dc, q, scans, qprecision=0, eoi=True, vsamp=1):
    """A grayscale progressive file made of the given scans, in which every AC coefficient is zero.

    dc: [block rows, block cols] full-precision DC values. q: 64 quantizers, natural order. scans: ('dc', ah, al) or
    ('ac', ss, se, ah, al); an AC scan is one EOB per block, which is a valid first scan or refinement of zeros. What the
    decoder is left with is the DC image plus a `coef_bits` state chosen by the scan list: exactly what block
    smoothing reads. Used to measure libjpeg-turbo's estimator one impulse at a time."""
    h, w = dc.shape[0] * 8, dc.shape[1] * 8
    out = bytearray(b"\xff\xd8")
    zz = [q[ZZ[i]] for i in range(64)]
    if qprecision or max(zz) > 255:
        out += seg(0xDB, bytes([0x10]) + b"".join(struct.pack(">H", v) for v in zz))
    else:
        out += seg(0xDB, bytes([0]) + bytes(zz))
    # vsamp > 1 gives the lone component several block rows per iMCU row, as a luma plane has in 4:2:0 or 4:4:0
    out += seg(0xC2, bytes([8]) + struct.pack(">HH", h, w) + bytes([1, 1, 0x10 | vsamp, 0]))
    out += dht(0, 0, DC_BITS, DC_VALS) + dht(1, 0, AC_BITS, AC_VALS)
    for s in scans:
        bw = Bits()
        if s[0] == "dc":
            _, ah, al = s
            out += seg(0xDA, bytes([1, 1, 0x00, 0, 0, (ah << 4) | al]))
            prev = 0
            for by in range(dc.shape[0]):
                for bx in range(dc.shape[1]):
                    v = int(dc[by, bx])
                    if ah == 0:
                        v >>= al
                        d = v - prev
                        prev = v
                        c = abs(d).bit_length()
                        bw.put(*DC_CODE[c])
                        if c:
                            bw.put(d if d > 0 else d + (1 << c) - 1, c)
                    else:
                        bw.put((v >> al) & 1, 1)
        else:
            _, ss, se, ah, al = s
            out += seg(0xDA, bytes([1, 1, 0x00, ss, se, (ah << 4) | al]))
            for _ in range(dc.shape[0] * dc.shape[1]):
                bw.put(*AC_CODE[0x00])
        bw.flush()
        out += bytes(bw.out)
    if eoi:
        out += b"\xff\xd9"
    return bytes(out)


# ---------------------------------------------------------------------------
# synth: a lossless (SOF3, T.81 Annex H) writer
# ---------------------------------------------------------------------------

# Every difference category 0..16 gets a 5-bit code, so any difference can be coded.
LL_BITS = [0] * 16
LL_BITS[4] = 17
LL_VALS = list(range(17))
LL_CODE = _codes(LL_BITS, LL_VALS)


def ll_predict(psv, ra, rb, rc):
    """Predictor selection value 1..7 (Table H.1), integer arithmetic as the decoder does."""
    return {1: ra, 2: rb, 3: rc, 4: ra + rb - rc, 5: ra + ((rb - rc) >> 1), 6: rb + ((ra - rc) >> 1), 7: (ra + rb) >> 1}[psv]


def ll_put_diff(bw, d):
    """One difference d (-32768..32768) as a category and its extra bits."""
    c = abs(d).bit_length()
    bw.put(*LL_CODE[c])
    if 0 < c < 16:
        bw.put(d if d > 0 else d + (1 << c) - 1, c)


def write_lossless(width, height, comps, planes, *, psv=1, pt=0, precision=8, restart=0, interleave=True, adobe=None,
                   jfif=False, comp_ids=None, diffs=None, eoi=True, scan_ss=None, scan_se=0, scan_ah=0, dht_extra=b"",
                   restart_rows=None, scans=None):
    """A lossless JPEG: comps [(h, v)], planes[c] the coded samples (after the point transform), padded to the MCU grid
    ([mcu rows * v, mcu cols * h]). `diffs[c]`, when given, are the differences written instead of the ones the
    samples imply (arbitrary values, the decoder's hostile input). `restart` is in MCUs and a multiple of the MCUs per
    row; the first sample row of every restart interval is predicted from the left only, its first sample from 2^(P-Pt-1)."""
    out = bytearray(b"\xff\xd8")
    if jfif:
        out += seg(0xE0, b"JFIF\0\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    if adobe is not None:
        out += seg(0xEE, b"Adobe" + struct.pack(">HHHB", 100, 0, 0, adobe))
    ids = comp_ids or list(range(1, len(comps) + 1))
    sofp = bytes([precision]) + struct.pack(">HH", height, width) + bytes([len(comps)])
    for i, (h, v) in enumerate(comps):
        sofp += bytes([ids[i], ((h & 15) << 4) | (v & 15), 0])
    out += seg(0xC3, sofp)
    out += dht(0, 0, LL_BITS, LL_VALS) + dht_extra
    if restart:
        out += seg(0xDD, struct.pack(">H", restart))
    hmax = max(c[0] for c in comps)
    vmax = max(c[1] for c in comps)
    mcux = -(-width // hmax)
    mcuy = -(-height // vmax)
    start = 1 << max(precision - pt - 1, 0)
    ss = psv if scan_ss is None else scan_ss

    def code_scan(cidx):
        hdr = bytes([len(cidx)]) + b"".join(bytes([ids[c], 0x00]) for c in cidx) + bytes([ss, scan_se, (scan_ah << 4) | pt])
        bw = Bits()
        if len(cidx) == 1:
            c = cidx[0]
            h, v = comps[c]
            cw = -(-width * h // hmax)
            ch = -(-height * v // vmax)
            mcols, mrows = cw, ch
            geom = {c: (1, 1)}
        else:
            mcols, mrows = mcux, mcuy
            geom = {c: comps[c] for c in cidx}
        rows_per_interval = restart_rows if restart_rows else ((restart // mcols) if restart else 0)
        rst = 0
        first_row = {c: 0 for c in cidx}  # first sample row of the current restart interval, per component
        for my in range(mrows):
            if rows_per_interval and my and my % rows_per_interval == 0:
                bw.flush()
                bw.out += bytes([0xFF, 0xD0 + (rst & 7)])
                rst += 1
                for c in cidx:
                    first_row[c] = my * geom[c][1]
            for mx in range(mcols):
                for c in cidx:
                    h, v = geom[c]
                    for dy in range(v):
                        for dx in range(h):
                            y, x = my * v + dy, mx * h + dx
                            p = planes[c]
                            if diffs is not None:
                                d = int(diffs[c][y, x])
                            else:
                                if y == first_row[c]:
                                    pred = start if x == 0 else int(p[y, x - 1])
                                elif x == 0:
                                    pred = int(p[y - 1, 0])
                                else:
                                    pred = ll_predict(psv, int(p[y, x - 1]), int(p[y - 1, x]), int(p[y - 1, x - 1]))
                                d = (int(p[y, x]) - pred) & 0xFFFF
                                if d >= 0x8000:
                                    d -= 0x10000
                            ll_put_diff(bw, d)
        bw.flush()
        return seg(0xDA, hdr) + bytes(bw.out)

    if scans is not None:  # explicit scans, each a list of component indices (a component may be left out or repeated)
        for s in scans:
            out += code_scan(list(s))
    elif interleave or len(comps) == 1:
        out += code_scan(list(range(len(comps))))
    else:
        for c in range(len(comps)):
            out += code_scan([c])
    if eoi:
        out += b"\xff\xd9"
    return bytes(out)


def ll_planes(rng, comps, width, height, precision=8, pt=0, smooth=True):
    """Image-like coded samples for a lossless writer: padded planes of values in [0, 2^(P-Pt))."""
    hmax = max(c[0] for c in comps)
    vmax = max(c[1] for c in comps)
    mcux, mcuy = -(-width // hmax), -(-height // vmax)
    top = (1 << (precision - pt)) - 1
    out = []
    for (h, v) in comps:
        ph, pw = mcuy * v, mcux * h
        y, x = np.mgrid[0:ph, 0:pw].astype(np.float64)
        base = (0.5 + 0.35 * np.sin(x / 5.0 + y / 7.0 + rng.uniform(0, 6))) if smooth else rng.uniform(0, 1, (ph, pw))
        a = np.clip(np.rint(base * top + (rng.normal(0, 0.02 * top, (ph, pw)) if smooth else 0)), 0, top).astype(np.int64)
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


SCAN_DIR = os.path.join(HERE, "..", "target", "jpeg_scans")

# cjpeg -scans scripts (wizard.txt): "components: Ss Se Ah Al;". Every one is a valid progression.
SCAN_SCRIPTS = {
    # No successive approximation: coef_bits ends at 0 for what has arrived and -1 for the rest.
    "spectral": "0 1 2: 0 0 0 0;\n0: 1 5 0 0;\n1: 1 63 0 0;\n2: 1 63 0 0;\n0: 6 63 0 0;\n",
    # DC at Al=2 and AC 1..9 at Al=3, refined one bit per scan.
    "deep": "0 1 2: 0 0 0 2;\n0: 1 9 0 3;\n1: 1 9 0 3;\n2: 1 9 0 3;\n0: 10 63 0 3;\n1: 10 63 0 3;\n2: 10 63 0 3;\n"
            "0: 1 63 3 2;\n1: 1 63 3 2;\n2: 1 63 3 2;\n0 1 2: 0 0 2 1;\n0: 1 63 2 1;\n1: 1 63 2 1;\n2: 1 63 2 1;\n"
            "0 1 2: 0 0 1 0;\n0: 1 63 1 0;\n1: 1 63 1 0;\n2: 1 63 1 0;\n",
    # The high band arrives before the low one.
    "high_first": "0 1 2: 0 0 0 1;\n0: 6 63 0 2;\n0: 1 5 0 2;\n1: 1 63 0 2;\n2: 1 63 0 2;\n0: 1 63 2 1;\n1: 1 63 2 1;\n"
                  "2: 1 63 2 1;\n0 1 2: 0 0 1 0;\n0: 1 63 1 0;\n1: 1 63 1 0;\n2: 1 63 1 0;\n",
    # The chroma planes lag the luma plane by whole bands.
    "luma_first": "0 1 2: 0 0 0 1;\n0: 1 63 0 2;\n0: 1 63 2 1;\n0: 1 63 1 0;\n1: 1 63 0 0;\n2: 1 63 0 0;\n0 1 2: 0 0 1 0;\n",
}
GRAY_SCRIPTS = {
    "spectral": "0: 0 0 0 0;\n0: 1 5 0 0;\n0: 6 63 0 0;\n",
    "deep": "0: 0 0 0 2;\n0: 1 9 0 3;\n0: 10 63 0 3;\n0: 1 63 3 2;\n0: 0 0 2 1;\n0: 1 63 2 1;\n0: 0 0 1 0;\n0: 1 63 1 0;\n",
    "high_first": "0: 0 0 0 1;\n0: 6 63 0 2;\n0: 1 5 0 2;\n0: 1 63 2 1;\n0: 0 0 1 0;\n0: 1 63 1 0;\n",
}


def cj_script(img, script, *flags):
    """cjpeg with a progression script (-scans)."""
    os.makedirs(SCAN_DIR, exist_ok=True)
    p = os.path.join(SCAN_DIR, "s%08x.txt" % zlib.crc32(script.encode()))
    with open(p, "w") as f:
        f.write(script)
    return cj(img, "-scans", p, *flags)


def cut_all(name, data, klass, note, ks=None):
    """The file cut after k scans and closed with an EOI, for every k (or those in ks)."""
    pos = scan_offsets(data)
    for k in ks if ks is not None else range(1, len(pos)):
        case(f"{name}_{k}scans", data[:pos[k]] + b"\xff\xd9", klass, f"{note} cut after {k} of {len(pos)} scans + EOI")


def cut_mid(name, data, klass, note, ks=None):
    """The file cut inside the entropy data of scan k (a third and two thirds of the way), closed with an EOI: the scan runs
    out of data at a marker, so the rest of it decodes as zeros and libjpeg-turbo falls back to the earlier progression."""
    pos = scan_offsets(data) + [len(data) - 2]
    for k in ks if ks is not None else range(len(pos) - 1):
        # the entropy-coded data starts after the SOS header, whose length is in its second field
        body = pos[k] + 2 + struct.unpack(">H", data[pos[k] + 2:pos[k] + 4])[0]
        for third in (1, 2):
            at = body + (pos[k + 1] - body) * third // 3
            case(f"{name}_mid{k}_{third}", data[:at] + b"\xff\xd9", klass, f"{note} cut inside scan {k + 1}, {third}/3 of it + EOI", tessl="either")


def cut_hole(name, data, klass, note, ks=None):
    """Scan k cut short a third and two thirds of the way into its entropy data, the scans after it kept: the cut scan
    runs out of data at the next marker, and later scans decode normally (found by live fuzzing: a later scan with data
    moves libjpeg-turbo's last_good_iMCU_row on)."""
    pos = scan_offsets(data) + [len(data) - 2]
    for k in ks if ks is not None else range(len(pos) - 1):
        body = pos[k] + 2 + struct.unpack(">H", data[pos[k] + 2:pos[k] + 4])[0]
        for third in (1, 2):
            at = body + (pos[k + 1] - body) * third // 3
            case(f"{name}_hole{k}_{third}", data[:at] + data[pos[k + 1]:], klass, f"{note} scan {k + 1} cut at {third}/3, the scans after it kept", tessl="either")


def smoothing_cases():
    """Progressive files whose scans stop early: libjpeg-turbo smooths them (block smoothing)."""
    k = "smoothing"
    img = photo(33, 47, 5)
    # the real encoders' default script, every prefix, every subsampling
    cut_all("smoothing_gray", cj(photo(40, 56, 9), "-grayscale", "-progressive"), k, "grayscale 40x56")
    cut_all("smoothing_420", cj(photo(40, 56, 9), "-progressive", "-sample", "2x2,1x1,1x1"), k, "4:2:0 40x56")
    for tag, s in (("444", "1x1,1x1,1x1"), ("422", "2x1,1x1,1x1"), ("420", "2x2,1x1,1x1"), ("411", "4x1,1x1,1x1"),
                   ("440", "1x2,1x1,1x1"), ("21_12", "2x1,1x2,1x1")):
        cut_all(f"smoothing_s{tag}", cj(img, "-progressive", "-sample", s), k, f"cjpeg -sample {s}, 33x47")
    # sizes around the block and MCU edges (the picture's edge rows and columns are replicated)
    # (even numbers of luma block rows - 32, 48, 64, 96 - put the rows below the last iMCU row's real ones outside the picture)
    for (h, w) in [(8, 8), (9, 9), (17, 15), (24, 40), (65, 129), (1, 9), (9, 1), (33, 200), (200, 33), (72, 40),
                   (32, 32), (48, 24), (64, 40), (96, 17)]:
        im = photo(h, w, h * 7 + w)
        cut_all(f"smoothing_size_{h}x{w}_420", cj(im, "-progressive", "-sample", "2x2,1x1,1x1"), k, f"{h}x{w} 4:2:0", ks=(1, 2, 5, 6))
        cut_all(f"smoothing_size_{h}x{w}_gray", cj(im, "-progressive", "-grayscale"), k, f"{h}x{w} gray", ks=(1, 2))
        cut_all(f"smoothing_size_{h}x{w}_440", cj(im, "-progressive", "-sample", "1x2,1x1,1x1"), k, f"{h}x{w} 4:4:0", ks=(1, 5))
    # components of three and four block rows per iMCU row (the row logic is measured on two)
    for tag, s in (("v3", "1x3,1x1,1x1"), ("v4", "1x4,1x1,1x1"), ("v4c2", "1x4,1x2,1x1"), ("h2v3", "2x3,1x1,1x1")):
        for (h, w) in [(33, 47), (96, 24), (104, 40), (56, 17)]:
            cut_all(f"smoothing_tall_{tag}_{h}x{w}", cj(photo(h, w, h + w), "-progressive", "-sample", s), k, f"cjpeg -sample {s}, {h}x{w}", ks=(1, 2, 5))
    # progression scripts that leave coef_bits in states the default script never does
    for tag, s in (("420", "2x2,1x1,1x1"), ("444", "1x1,1x1,1x1")):
        for name, script in SCAN_SCRIPTS.items():
            cut_all(f"smoothing_{tag}_{name}", cj_script(img, script, "-sample", s), k, f"script {name}, {tag}")
    for name, script in GRAY_SCRIPTS.items():
        cut_all(f"smoothing_gray_{name}", cj_script(img, script, "-grayscale"), k, f"script {name}, gray")
    # arithmetic coding and restart intervals
    cut_all("smoothing_arith_420", cj(img, "-arithmetic", "-progressive", "-sample", "2x2,1x1,1x1"), k, "arithmetic 4:2:0")
    cut_all("smoothing_arith_gray", cj(img, "-arithmetic", "-progressive", "-grayscale"), k, "arithmetic gray", ks=(1, 2, 4))
    cut_all("smoothing_rst_420", cj(img, "-progressive", "-restart", "2", "-sample", "2x2,1x1,1x1"), k, "restart 2", ks=(1, 3, 5))
    cut_all("smoothing_rst_arith", cj(img, "-arithmetic", "-progressive", "-restart", "3"), k, "arithmetic, restart 3", ks=(1, 3))
    # a scan that runs out of data in the middle
    cut_mid("smoothing_gray", cj(photo(40, 56, 9), "-grayscale", "-progressive"), k, "grayscale 40x56")
    cut_mid("smoothing_420", cj(photo(40, 56, 9), "-progressive", "-sample", "2x2,1x1,1x1"), k, "4:2:0 40x56")
    cut_mid("smoothing_440", cj(photo(65, 40, 3), "-progressive", "-sample", "1x2,1x1,1x1"), k, "4:4:0 65x40", ks=(0, 1, 3, 6))
    cut_hole("smoothing_gray", cj(photo(40, 56, 9), "-grayscale", "-progressive"), k, "grayscale 40x56")
    cut_hole("smoothing_420", cj(photo(40, 56, 9), "-progressive", "-sample", "2x2,1x1,1x1"), k, "4:2:0 40x56")
    cut_hole("smoothing_420_24x40", cj(photo(24, 40, 4), "-progressive", "-sample", "2x2,1x1,1x1"), k, "4:2:0 24x40")
    cut_hole("smoothing_rst", cj(photo(40, 56, 9), "-progressive", "-restart", "2", "-grayscale"), k, "restart 2 gray", ks=(0, 1, 3))
    cut_mid("smoothing_arith_gray", cj(photo(40, 56, 9), "-arithmetic", "-progressive", "-grayscale"), k, "arithmetic gray", ks=(0, 2))
    cut_mid("smoothing_rst", cj(photo(40, 56, 9), "-progressive", "-restart", "2", "-grayscale"), k, "restart 2 gray", ks=(0, 1, 3))

    # --- synthetic: a DC image and a progression state chosen by the scan list ---------------------
    rng = np.random.default_rng(31)
    for (h, w), (q00, qac), amp in [((1, 1), (16, 8), 6), ((1, 2), (16, 8), 6), ((2, 1), (16, 8), 6), ((1, 3), (3, 1), 40),
                                    ((3, 1), (3, 1), 40), ((3, 3), (3, 1), 40), ((4, 7), (40, 17), 40), ((9, 6), (7, 60), 40),
                                    ((10, 10), (1, 1), 40), ((6, 6), (255, 255), 40), ((8, 8), (16, 40), 600)]:
        dc = rng.integers(-amp, amp + 1, (h, w))
        case(f"smoothing_dc_{h}x{w}_q{q00}_{qac}", write_progressive_gray(dc, [q00] + [qac] * 63, [("dc", 0, 0)]),
             k, f"DC-only, {h}x{w} blocks, Q00 {q00}, AC quantizers {qac}")
    for d, q00, qac in ((5, 64, 4), (-5, 64, 4), (2, 256, 1), (1, 255, 255), (30, 8, 2)):
        dc = np.zeros((13, 13), dtype=int)
        dc[6, 6] = d
        case(f"smoothing_impulse_d{d}_q{q00}_{qac}", write_progressive_gray(dc, [q00] + [qac] * 63, [("dc", 0, 0)]), k,
             f"one DC impulse {d}, Q00 {q00}, AC quantizers {qac}")
    for al in (1, 2, 4):
        dc = rng.integers(-60, 61, (7, 9))
        case(f"smoothing_dc_al{al}", write_progressive_gray(dc, [8] + [3] * 63, [("dc", 0, al)]), k, f"DC scan with point transform {al}")
    dc = rng.integers(-60, 61, (7, 9))
    case("smoothing_dc_al2_refined_once", write_progressive_gray(dc, [8] + [3] * 63, [("dc", 0, 2), ("dc", 2, 1)]), k, "DC Al=2 then refined to Al=1")
    # DC plus AC bands in various states: Al = bits not yet sent, the estimate is limited to 2^Al - 1
    for al, hi in ((1, 2), (2, 5), (3, 5), (5, 5), (6, 9), (4, 9), (7, 3)):
        dc = rng.integers(-40, 41, (10, 10))
        case(f"smoothing_state_ac1to{hi}_al{al}", write_progressive_gray(dc, [64] + [4] * 63, [("dc", 0, 0), ("ac", 1, hi, 0, al)]), k,
             f"DC, then AC 1..{hi} at Al={al}")
    for zz in range(1, 10):
        dc = rng.integers(-40, 41, (9, 9))
        scans = [("dc", 0, 0)] + [("ac", i, i, 0, 0) for i in range(1, 10) if i != zz] + [("ac", zz, zz, 0, 4)]
        case(f"smoothing_only_zz{zz}", write_progressive_gray(dc, [64] + [4] * 63, scans), k, f"zigzag {zz} the only coefficient still unknown")
    dc = rng.integers(-40, 41, (9, 9))
    case("smoothing_state_known_1to5_missing_6to9", write_progressive_gray(dc, [64] + [4] * 63, [("dc", 0, 0), ("ac", 1, 5, 0, 0)]), k,
         "AC 1..5 complete, 6..9 never sent: smoothing is on and estimates nothing")
    case("smoothing_state_high_band_only", write_progressive_gray(dc, [64] + [4] * 63, [("dc", 0, 0), ("ac", 6, 63, 0, 0)]), k,
         "AC 6..63 complete, 1..5 never sent")
    case("smoothing_state_no_dc_scan", write_progressive_gray(dc, [64] + [4] * 63, [("ac", 1, 5, 0, 2)]), k, "no DC scan: smoothing is off")
    case("smoothing_state_refined_to_known", write_progressive_gray(dc, [64] + [4] * 63, [("dc", 0, 0), ("ac", 1, 63, 0, 1), ("ac", 1, 63, 1, 0)]),
         k, "every AC coefficient complete: smoothing is off")
    # one component with several block rows per iMCU row, every height from one block row to a few iMCU rows past two: the
    # row logic at the top and bottom of the picture and at the iMCU-row boundaries, without a second component in the way
    for vs in (2, 3, 4):
        for rows in range(1, 4 * vs + 2):
            dc = rng.integers(-60, 61, (rows, 5))
            for tag, scans in (("dc", [("dc", 0, 0)]), ("ac5", [("dc", 0, 0), ("ac", 1, 5, 0, 2)]),
                               ("ac9", [("dc", 0, 0), ("ac", 1, 9, 0, 1)])):
                case(f"smoothing_vrows{vs}_{rows:02d}_{tag}", write_progressive_gray(dc, [16] + [5] * 63, scans, vsamp=vs), k,
                     f"one component, {vs} block rows per iMCU row, {rows} block rows, state {tag}")
    # a zero quantizer in the guarded set switches smoothing off; elsewhere it does not
    dc = rng.integers(-20, 21, (8, 8))
    for nat in (0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 4, 5, 25, 63):
        q = [6] * 64
        q[0] = 16
        q[nat] = 0
        case(f"smoothing_zero_quantizer_{nat}", write_progressive_gray(dc, q, [("dc", 0, 0)]), k, f"quantizer at natural position {nat} is 0")
    for nat in (3, 24, 1):
        q = [6] * 64
        q[0] = 16
        q[nat] = 0
        case(f"smoothing_zero_quantizer_{nat}_part", write_progressive_gray(dc, q, [("dc", 0, 0), ("ac", 1, 5, 0, 2)]), k,
             f"quantizer at natural position {nat} is 0, AC 1..5 at Al=2")
    # extremes: the estimate is stored in 16 bits
    for amp, q00, qac in ((2047, 255, 1), (2047, 65535, 1), (16000, 1, 65535), (16000, 255, 3), (2047, 40, 255)):
        dc = rng.integers(-amp, amp + 1, (6, 6))
        case(f"smoothing_extreme_dc{amp}_q{q00}_{qac}", write_progressive_gray(dc, [q00] + [qac] * 63, [("dc", 0, 0)]), k,
             f"DC up to {amp}, Q00 {q00}, AC quantizers {qac}")


def lossless_cases():
    """Lossless JPEG (SOF3, T.81 Annex H) from this script's writer and from cjpeg -lossless."""
    k = "lossless"
    rng = np.random.default_rng(77)
    g1, c3, c4 = [(1, 1)], [(1, 1)] * 3, [(1, 1)] * 4
    # libjpeg-turbo's lossless decoder copies samples and converts nothing: RGB (three components, Adobe transform 0 or the
    # ids R G B) and CMYK decode; one component (gray to RGB), YCbCr and YCCK are refused when the decompressor starts.
    # So the matrix is RGB, and a few of the others are filed to show the refusal.
    for psv in range(1, 8):
        p3 = ll_planes(rng, c3, 17, 9)
        case(f"lossless_rgb_psv{psv}", write_lossless(17, 9, c3, p3, psv=psv, adobe=0), k, f"17x9 RGB (Adobe 0), predictor {psv}")
        case(f"lossless_rgb_ids_psv{psv}", write_lossless(17, 9, c3, p3, psv=psv, comp_ids=[82, 71, 66]), k, f"17x9 RGB (ids R G B, no Adobe), predictor {psv}")
        case(f"lossless_noise_psv{psv}", write_lossless(9, 8, c3, ll_planes(rng, c3, 9, 8, smooth=False), psv=psv, adobe=0), k, f"noise, predictor {psv}")
    for psv in (1, 4):
        case(f"lossless_refused_gray_psv{psv}", write_lossless(13, 11, g1, ll_planes(rng, g1, 13, 11), psv=psv), k, "one component: gray to RGB is not converted", tessl="either")
        case(f"lossless_refused_ycc_psv{psv}", write_lossless(17, 9, c3, ll_planes(rng, c3, 17, 9), psv=psv, jfif=True), k, "YCbCr to RGB is not converted", tessl="either")
        case(f"lossless_refused_adobe1_psv{psv}", write_lossless(17, 9, c3, ll_planes(rng, c3, 17, 9), psv=psv, adobe=1), k, "Adobe transform 1 (YCbCr)", tessl="either")
    # the standard-table fallback belongs to the DCT Huffman decoder: a lossless stream with no DHT is refused (found by live fuzzing)
    case("lossless_no_dht", strip_dht(write_lossless(17, 9, c3, p3, psv=4, adobe=0)), k, "lossless, DHT removed: no standard tables", tessl="either")
    # every component must be in some scan ("Bogus virtual array access" otherwise); a component may be in several
    for name, scans, note in (("missing_first", [[1], [2]], "the first component is in no scan"),
                              ("missing_last", [[0], [1]], "the last component is in no scan"),
                              ("missing_middle_pair", [[0], [2]], "the middle component is in no scan"),
                              ("missing_in_interleaved", [[1, 2]], "one interleaved scan of the last two components"),
                              ("single_then_pair", [[0], [1, 2]], "one component, then the other two interleaved"),
                              ("pair_then_single", [[0, 1], [2]], "two components interleaved, then the third"),
                              ("repeated_scan", [[0], [0], [1], [2]], "the first component in two scans")):
        case(f"lossless_scans_{name}", write_lossless(17, 9, c3, p3, psv=4, adobe=0, scans=scans), k, note, tessl="either")
    case("lossless_ids_123_no_marker", write_lossless(17, 9, c3, ll_planes(rng, c3, 17, 9), psv=2), k, "no JFIF or Adobe marker, ids 1 2 3: decoded as RGB")
    case("lossless_ids_789_no_marker", write_lossless(17, 9, c3, ll_planes(rng, c3, 17, 9), psv=3, comp_ids=[7, 8, 9]), k, "no marker, unrecognised ids")
    for pt in (1, 2, 4, 7):
        for psv in (1, 4, 6, 7):
            case(f"lossless_pt{pt}_psv{psv}", write_lossless(12, 10, c3, ll_planes(rng, c3, 12, 10, pt=pt), psv=psv, pt=pt, adobe=0), k, f"point transform {pt}, predictor {psv}")
    for psv in (1, 3, 7):
        pl = ll_planes(rng, c3, 12, 7)
        case(f"lossless_rgb_noninterleaved_psv{psv}", write_lossless(12, 7, c3, pl, psv=psv, adobe=0, interleave=False), k, "one scan per component")
    # sizes
    for (h, w) in [(1, 1), (1, 7), (7, 1), (2, 2), (8, 8), (3, 33), (33, 3)]:
        case(f"lossless_size_{h}x{w}", write_lossless(w, h, c3, ll_planes(rng, c3, w, h), psv=4, adobe=0), k, f"{h}x{w}, predictor 4")
    # subsampled components: the reference upsamples without the fancy filter
    for tag, comps in (("h2v1", [(2, 1), (1, 1), (1, 1)]), ("h2v2", [(2, 2), (1, 1), (1, 1)]), ("h1v2", [(1, 2), (1, 1), (1, 1)]),
                       ("h4v1", [(4, 1), (1, 1), (1, 1)]), ("h2v2_all", [(2, 2), (2, 1), (1, 2)])):
        for (w, h) in ((16, 12), (13, 9)):
            pl = ll_planes(rng, comps, w, h)
            case(f"lossless_sub_{tag}_{h}x{w}", write_lossless(w, h, comps, pl, psv=2, adobe=0), k, f"{tag}, {h}x{w}, predictor 2")
    pl = ll_planes(rng, [(2, 2), (1, 1), (1, 1)], 16, 12)
    case("lossless_sub_h2v2_noninterleaved", write_lossless(16, 12, [(2, 2), (1, 1), (1, 1)], pl, psv=5, adobe=0, interleave=False), k,
         "4:2:0 in three scans")
    # four components
    p4 = ll_planes(rng, c4, 10, 6)
    case("lossless_cmyk", write_lossless(10, 6, c4, p4, psv=1, adobe=0), k, "CMYK (Adobe 0)")
    case("lossless_refused_ycck", write_lossless(10, 6, c4, p4, psv=3, adobe=2), k, "YCCK (Adobe 2) to CMYK is not converted", tessl="either")
    case("lossless_cmyk_ids", write_lossless(10, 6, c4, p4, psv=5), k, "four components, no Adobe marker: CMYK")
    # restart intervals, in MCUs, whole MCU rows
    for restart, (w, h) in ((6, (6, 12)), (12, (6, 12)), (9, (9, 8)), (18, (9, 8))):
        pl = ll_planes(rng, c3, w, h)
        for psv in (1, 4, 7):
            case(f"lossless_rst{restart}_{w}x{h}_psv{psv}", write_lossless(w, h, c3, pl, psv=psv, restart=restart, adobe=0), k, f"restart every {restart} MCUs, predictor {psv}")
    pl = ll_planes(rng, c3, 5, 9)
    case("lossless_rst_rgb_interleaved", write_lossless(5, 9, c3, pl, psv=6, adobe=0, restart=10), k, "3 components, restart every 2 rows")
    case("lossless_rst_rgb_noninterleaved", write_lossless(5, 9, c3, pl, psv=6, adobe=0, restart=5, interleave=False), k, "one scan per component, restart every row")
    pl = ll_planes(rng, [(2, 2), (1, 1), (1, 1)], 16, 12)
    case("lossless_rst_sub_interleaved", write_lossless(16, 12, [(2, 2), (1, 1), (1, 1)], pl, psv=4, adobe=0, restart=16), k, "4:2:0, restart every 2 MCU rows")
    case("lossless_rst_sub_noninterleaved", write_lossless(16, 12, [(2, 2), (1, 1), (1, 1)], pl, psv=4, adobe=0, restart=16, interleave=False), k,
         "4:2:0 in three scans with a restart interval (the interval is checked against the last scan's MCU row)", tessl="either")
    # a scan that ends in a marker before the picture does: the rest is zero differences, i.e. prediction only
    for name, data in (("rgb", write_lossless(17, 9, c3, ll_planes(rng, c3, 17, 9), psv=4, adobe=0)),
                       ("ps7", write_lossless(17, 9, c3, ll_planes(rng, c3, 17, 9), psv=7, adobe=0)),
                       ("rst", write_lossless(6, 12, c3, ll_planes(rng, c3, 6, 12), psv=2, restart=12, adobe=0)),
                       ("sub", write_lossless(16, 12, [(2, 2), (1, 1), (1, 1)], ll_planes(rng, [(2, 2), (1, 1), (1, 1)], 16, 12), psv=3, adobe=0))):
        sos = data.find(b"\xff\xda")
        body = sos + 2 + struct.unpack(">H", data[sos + 2:sos + 4])[0]
        for third in (1, 2):
            at = body + (len(data) - 2 - body) * third // 3
            case(f"lossless_cut_{name}_{third}", data[:at] + b"\xff\xd9", k, f"{name} cut {third}/3 of the way into the scan + EOI")
    # what the reference does with a restart interval that is not a whole number of MCU rows
    pl = ll_planes(rng, c3, 6, 12)
    case("lossless_rst_partial_row", write_lossless(6, 12, c3, pl, psv=1, restart=4, restart_rows=2, adobe=0), k, "restart interval 4 with 6 MCUs per row", tessl="either")
    case("lossless_rst_one_mcu", write_lossless(6, 12, c3, pl, psv=1, restart=1, restart_rows=1, adobe=0), k, "restart interval 1 with 6 MCUs per row", tessl="either")
    # hostile differences: wrap-around at 16 bits, the largest category, values the point transform cannot take back
    zeros = lambda h, w: [np.zeros((h, w), dtype=np.int64)] * 3  # noqa: E731
    for psv in (1, 2, 3, 4, 5, 6, 7):
        d = [rng.integers(-300, 301, (8, 9)) for _ in range(3)]
        case(f"lossless_wild_diffs_psv{psv}", write_lossless(9, 8, c3, zeros(8, 9), psv=psv, diffs=d, adobe=0), k, f"differences of +-300, predictor {psv}")
        d = [rng.integers(-32768, 32769, (6, 7)) for _ in range(3)]
        case(f"lossless_huge_diffs_psv{psv}", write_lossless(7, 6, c3, zeros(6, 7), psv=psv, diffs=d, adobe=0), k, f"differences over the whole 16-bit range, predictor {psv}")
    d = [rng.integers(-300, 301, (8, 9)) for _ in range(3)]
    case("lossless_wild_diffs_pt3", write_lossless(9, 8, c3, zeros(8, 9), psv=4, pt=3, diffs=d, adobe=0), k, "differences of +-300, point transform 3")
    d = [rng.integers(-3, 4, (8, 9)) for _ in range(3)]
    for prec in (2, 5, 7):
        case(f"lossless_small_diffs_p{prec}", write_lossless(9, 8, c3, zeros(8, 9), psv=4, precision=prec, diffs=d, adobe=0), k,
             f"{prec}-bit samples driven past their range by differences of +-3")
    # scan parameters
    pl = ll_planes(rng, c3, 8, 8)
    for name, kw in (("psv0", dict(scan_ss=0)), ("psv8", dict(scan_ss=8)), ("se1", dict(scan_se=1)), ("ah1", dict(scan_ah=1)),
                     ("pt_eq_precision_minus1", dict(pt=7)), ("pt_eq_precision", dict(pt=8)), ("pt_over_precision", dict(pt=9)),
                     ("pt15", dict(pt=15))):
        case(f"lossless_params_{name}", write_lossless(8, 8, c3, pl, psv=1, adobe=0, **kw), k, f"scan header: {name}", tessl="either")
    for pt in (1, 3):
        case(f"lossless_params_p4_pt{pt}", write_lossless(8, 8, c3, ll_planes(rng, c3, 8, 8, precision=4, pt=pt), psv=1, adobe=0, precision=4, pt=pt), k,
             f"4-bit samples, point transform {pt}")
    for pt in (4, 5):
        case(f"lossless_params_p4_pt{pt}", write_lossless(8, 8, c3, pl, psv=1, adobe=0, precision=4, pt=pt), k, f"4-bit samples, point transform {pt}", tessl="either")
    # precisions: torchvision refuses above 8 bits
    for prec in range(2, 17):
        pl = ll_planes(rng, c3, 8, 6, precision=prec)
        case(f"lossless_precision_{prec}_rgb", write_lossless(8, 6, c3, pl, psv=4, precision=prec, adobe=0), k, f"{prec}-bit RGB", tessl="either")
        case(f"lossless_precision_{prec}_rgb_p1", write_lossless(8, 6, c3, pl, psv=1, precision=prec, adobe=0), k, f"{prec}-bit RGB, predictor 1", tessl="either")
    # cjpeg's own lossless encoder
    def cj_case(name, im, flags, note, klass=k):
        # cjpeg may refuse an option combination (the lossless encoder has its own limits); that is not a case
        try:
            case(name, cj(im, *flags), klass, note, tessl="either")
        except RuntimeError as e:
            print("cjpeg cannot encode", name, "-", e)

    for psv in range(1, 8):
        cj_case(f"lossless_cjpeg_psv{psv}", photo(16, 24, 2 + psv), ["-lossless", str(psv)], f"cjpeg -lossless {psv}")
    for psv, pt in ((1, 1), (4, 3), (7, 2)):
        cj_case(f"lossless_cjpeg_psv{psv}_pt{pt}", photo(16, 24, 9), ["-lossless", f"{psv},{pt}"], f"cjpeg -lossless {psv},{pt}")
    im = photo(17, 23, 4)
    cj_case("lossless_cjpeg_gray_flag", im, ["-lossless", "5", "-grayscale"], "cjpeg -grayscale is ignored under -lossless: an RGB file, predictor 5")
    cj_case("lossless_cjpeg_rgb", im, ["-lossless", "6", "-rgb"], "cjpeg -rgb, predictor 6")
    cj_case("lossless_cjpeg_420", im, ["-lossless", "2", "-sample", "2x2,1x1,1x1"], "cjpeg 4:2:0, predictor 2")
    cj_case("lossless_cjpeg_422", im, ["-lossless", "2", "-sample", "2x1,1x1,1x1"], "cjpeg 4:2:2, predictor 2")
    cj_case("lossless_cjpeg_restart", im, ["-lossless", "4", "-restart", "2"], "cjpeg restart every 2 rows")
    cj_case("lossless_cjpeg_arith", im, ["-lossless", "4", "-arithmetic"], "SOF11 lossless arithmetic", klass="error")
    for tag, script, extra in (("scans_split", "0: 4 0 0 0;\n1: 4 0 0 0;\n2: 4 0 0 0;\n", []),
                               ("scans_pt", "0 1 2: 5 0 0 2;\n", []),
                               ("scans_pt_split", "0: 3 0 0 1;\n1 2: 3 0 0 1;\n", []),
                               ("scans_420", "0 1 2: 2 0 0 0;\n", ["-sample", "2x2,1x1,1x1"]),
                               ("scans_420_split", "0: 2 0 0 0;\n1: 2 0 0 0;\n2: 2 0 0 0;\n", ["-sample", "2x2,1x1,1x1"])):
        try:
            case(f"lossless_cjpeg_{tag}", cj_script(im, script, *extra), k, f"cjpeg -scans {tag}", tessl="either")
        except RuntimeError as e:
            print("cjpeg cannot encode", tag, "-", e)
    for prec in (2, 4, 7, 12, 16):
        cj_case(f"lossless_cjpeg_precision_{prec}", photo(9, 11, 4), ["-lossless", "1", "-precision", str(prec)], f"cjpeg -precision {prec}")


def lossless_restart_subsampled_cases():
    """Lossless frames whose components differ in size, in several scans, with a restart interval.

    libjpeg-turbo undifferences a scan one iMCU row at a time (jddiffct.c decompress_data): a non-interleaved scan of a
    component with v_samp_factor Vi decodes Vi MCU rows (restart markers are honoured between them, and each restart
    resets every component's predictor to the first-row state) and only then undifferences the Vi rows together. A restart
    that falls inside an iMCU row therefore moves the first-row treatment to the iMCU row's first row. The grid varies what
    decides that: the component's v (1 to 4), the restart interval in rows of that component (1, 2, 3 -- odd and even,
    aligned or not with the iMCU row), the scan order and grouping, and picture sizes that are not a multiple of the MCU."""
    k = "lossless"
    rng = np.random.default_rng(1103)
    layouts = {
        "420": [(2, 2), (1, 1), (1, 1)],
        "h1v2": [(1, 2), (1, 1), (1, 1)],
        "h2v1": [(2, 1), (1, 1), (1, 1)],
        "v3": [(1, 3), (1, 1), (1, 1)],
        "v4": [(1, 4), (1, 1), (1, 1)],
        "mixed": [(2, 2), (2, 1), (1, 2)],
        "cbv2": [(1, 1), (1, 2), (1, 1)],
        "h2v3": [(2, 3), (1, 1), (1, 1)],
    }
    scan_lists = {
        "sep": [[0], [1], [2]],
        "rev": [[2], [1], [0]],
        "yfirst_pair": [[0], [1, 2]],
        "pair_yl": [[1, 2], [0]],
        "mid": [[1], [0], [2]],
    }
    sizes = {  # widths multiples of hmax keep the restart interval valid for every scan with a small lcm
        "420": ((16, 12), (16, 25), (8, 9)),
        "h1v2": ((5, 7), (13, 12)),
        "h2v1": ((16, 9), (8, 5)),
        "v3": ((7, 10), (5, 13)),
        "v4": ((6, 9), (4, 17)),
        "mixed": ((16, 12), (8, 11)),
        "cbv2": ((7, 12), (9, 5)),
        "h2v3": ((16, 13), (8, 7)),
    }

    def widths(comps, w, scans):
        hmax = max(h for h, _ in comps)
        mcux = -(-w // hmax)
        return {len(s) > 1 and mcux or -(-w * comps[s[0]][0] // hmax) for s in scans}

    for name, comps in layouts.items():
        for (w, h) in sizes[name]:
            for sname, scans in scan_lists.items():
                if sname in ("rev", "mid") and (w, h) != sizes[name][0]:
                    continue  # the order does not interact with the size; keep the grid to a few hundred KB
                base = math.lcm(*widths(comps, w, scans))
                for m in (1, 2, 3):
                    if m == 3 and sname not in ("sep", "yfirst_pair"):
                        continue
                    psv = 1 + (hash_name(f"{name}{w}{h}{sname}{m}") % 7)
                    pl = ll_planes(rng, comps, w, h)
                    case(f"lossless_rstsub_{name}_{w}x{h}_{sname}_r{m}",
                         write_lossless(w, h, comps, pl, psv=psv, adobe=0, restart=base * m, scans=scans), k,
                         f"{name} {w}x{h}, scans {scans}, restart every {base * m} MCUs, predictor {psv}")
    # a point transform, a cut inside the scans and a restart interval that is not a whole number of some scan's rows
    comps = layouts["420"]
    pl = ll_planes(rng, comps, 16, 12, pt=3)
    case("lossless_rstsub_420_pt3", write_lossless(16, 12, comps, pl, psv=4, pt=3, adobe=0, restart=16, scans=scan_lists["sep"]), k,
         "4:2:0 in three scans, restart every Y row, point transform 3")
    for sname in ("sep", "yfirst_pair"):
        data = write_lossless(16, 25, comps, ll_planes(rng, comps, 16, 25), psv=5, adobe=0, restart=16, scans=scan_lists[sname])
        sos = scan_offsets(data)
        for si in (0, 1):
            body = sos[si] + 2 + struct.unpack(">H", data[sos[si] + 2:sos[si] + 4])[0]
            nxt = sos[si + 1] if si + 1 < len(sos) else len(data) - 2
            for third in (1, 2):
                at = body + (nxt - body) * third // 3
                case(f"lossless_rstsub_cut_{sname}_s{si}_{third}", data[:at] + data[nxt:], k,
                     f"4:2:0 16x25 {sname}, restart every Y row, scan {si + 1} cut at {third}/3 and the scans after it kept", tessl="either")
    for restart in (8, 24, 12):  # Y's rows are 16 samples wide: 8 and 24 are not whole Y rows, 12 neither; chroma's 8 divides 8 and 24
        case(f"lossless_rstsub_420_16x12_bad_r{restart}",
             write_lossless(16, 12, comps, ll_planes(rng, comps, 16, 12), psv=4, adobe=0, restart=restart, restart_rows=1, scans=scan_lists["sep"]), k,
             f"4:2:0 in three scans, restart interval {restart} (not a multiple of the Y scan's 16 MCUs per row)", tessl="either")


def hash_name(s):
    """A stable small hash (Python's hash() is salted per process)."""
    return zlib.crc32(s.encode())


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
    smoothing_cases()
    # --- lossless (SOF3): libjpeg-turbo 3.x decodes it ----------------------------------------
    lossless_cases()
    lossless_restart_subsampled_cases()
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
    # progressive files cut short and closed (block smoothing), and lossless streams
    pg = cj(img, "-grayscale", "-progressive")
    bases["smooth_gray_cut3"] = pg[:scan_offsets(pg)[3]] + b"\xff\xd9"
    pc = cj(img, "-progressive", "-sample", "2x2,1x1,1x1")
    bases["smooth_420_cut2"] = pc[:scan_offsets(pc)[2]] + b"\xff\xd9"
    bases["smooth_420_cut6"] = pc[:scan_offsets(pc)[6]] + b"\xff\xd9"
    pa = cj(img, "-arithmetic", "-progressive", "-sample", "2x2,1x1,1x1")
    bases["smooth_arith_cut3"] = pa[:scan_offsets(pa)[3]] + b"\xff\xd9"
    lrng = np.random.default_rng(5)
    g1, l3, l420 = [(1, 1)], [(1, 1)] * 3, [(2, 2), (1, 1), (1, 1)]
    bases["lossless_rgb"] = write_lossless(21, 9, l3, ll_planes(lrng, l3, 21, 9), psv=4, adobe=0)
    bases["lossless_rgb_rst"] = write_lossless(8, 10, l3, ll_planes(lrng, l3, 8, 10), psv=6, adobe=0, restart=16)
    bases["lossless_420_pt"] = write_lossless(18, 10, l420, ll_planes(lrng, l420, 18, 10, pt=2), psv=7, pt=2, adobe=0)
    bases["lossless_noninterleaved"] = write_lossless(9, 7, l3, ll_planes(lrng, l3, 9, 7), psv=2, adobe=0, interleave=False)
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
    problems = []  # reported after everything is written, so one surprise does not cost the whole run
    for name, data, klass, note, tessl in CASES:
        with open(path(name + ".jpg"), "wb") as f:
            f.write(data)
        files[name + ".jpg"] = {"role": "input", "published": True, "note": f"{klass}: {note}" if note else klass}
        t = tv_decode(data)
        if isinstance(t, str):
            if not (klass == "error" or tessl == "either"):
                problems.append(f"{name}: the oracle refuses a case not filed under error: {t}")
            lines.append(f"case {name} expect=error class={klass} oracle={t.replace(' ', '_')}")
            n_err += 1
            continue
        if not (klass != "error" or tessl == "either"):
            problems.append(f"{name}: filed under error but the oracle decodes it")
        h, w = t.shape[:2]
        with open(path(name + ".jpg.rgb"), "wb") as f:
            f.write(np.ascontiguousarray(t).tobytes())
        files[name + ".jpg.rgb"] = {"role": "golden", "published": True, "note": "torchvision.io.decode_image(mode=RGB) of the input of the same name"}
        p = pil_decode(data)
        pil = "error" if isinstance(p, str) else ("equal" if np.array_equal(p, t) else "differs")
        pil_differs += pil == "differs"
        lines.append(f"case {name} expect=ok class={klass} w={w} h={h} pil={pil}")
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
    if problems:
        print("\n".join(problems))
        sys.exit(f"{len(problems)} cases are filed against the oracle's verdict")


if __name__ == "__main__":
    main()
