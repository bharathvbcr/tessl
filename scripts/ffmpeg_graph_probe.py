#!/usr/bin/env python3
"""Black-box check, through the public libavfilter and libswscale API (Homebrew ffmpeg's
headers and libraries; no FFmpeg source), of what torchcodec's colour conversion gives:

    ~/.venvs/ml-2026-10/bin/python scripts/ffmpeg_graph_probe.py

Builds scripts/ffmpeg_graph_probe.c into target/graph_probe/ and, for each random-plane
class of tests/fixtures/media_video/ (`format` lines of facts.txt), compares

  * the graph `format=rgb24` (a `buffer` source given format, size, aspect and time base
    only, the frame carrying its colorspace and range, a plain `buffersink`: torchcodec's
    arrangement for widths that are not multiples of 32), and
  * `sws_scale` with flags 0 (torchcodec's colour-conversion context),

with the ffmpeg CLI's `-pix_fmt rgb24` golden (`<name>.rgb24`), and the graphs
`format=rgb24,transpose=cclock|clock` and `format=rgb24,hflip,vflip` with that golden turned
afterwards. Exits 1 on any difference, and on an empty facts file or a golden or probe output of the wrong size. `--self-test` runs the planted-fault checks of those refusals without ffmpeg. Needs `cc` and Homebrew ffmpeg at /opt/homebrew/opt/ffmpeg.
"""

import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(ROOT, "tests", "fixtures", "media_video") + "/"
OUT = os.path.join(ROOT, "target", "graph_probe") + "/"
FF = "/opt/homebrew/opt/ffmpeg"


def parse_classes(text):
    """The `format` lines of facts.txt as (name, {key: value}). Refuses a facts file with no
    class, or a class missing a key the probe is run with: an empty set compares nothing and
    would report "0 differing"."""
    classes = []
    for line in text.splitlines():
        p = line.split()
        if p and p[0] == "format":
            kv = dict(x.split("=", 1) for x in p[2:])
            missing = [k for k in ("fmt", "w", "h", "matrix", "range") if k not in kv]
            if len(p) < 2 or missing:
                raise ValueError(f"facts.txt: format line {line!r} lacks {missing or ['a name']}")
            classes.append((p[1], kv))
    if not classes:
        raise ValueError("facts.txt: no `format` line, so there is nothing to compare")
    return classes


def expect_golden(name, kv, gold):
    """The CLI's rgb24 golden must be exactly w*h*3 bytes: a short or empty golden would equal a
    short or empty probe output and pass."""
    want = int(kv["w"]) * int(kv["h"]) * 3
    if want == 0 or len(gold) != want:
        raise ValueError(f"{name}: golden is {len(gold)} bytes, a {kv['w']}x{kv['h']} rgb24 frame is {want}")
    return gold


def expect_output(label, got, want_len):
    """A probe output must be the size of the frame it converted (never empty)."""
    if len(got) != want_len or want_len == 0:
        raise ValueError(f"{label}: {len(got)} bytes written, expected {want_len}")
    return got


def self_test():
    """Planted faults for the three refusals above, needing neither ffmpeg nor numpy."""
    good = "format a fmt=yuv420p w=2 h=2 matrix=bt709 range=tv\n"
    assert parse_classes(good)[0][0] == "a"
    for bad in ("", "\n\n", "other x\n", "format\n", "format a fmt=yuv420p w=2 h=2\n"):
        try:
            parse_classes(bad)
        except ValueError:
            continue
        raise SystemExit(f"self-test: parse_classes accepted {bad!r}")
    kv = {"w": "2", "h": "2"}
    assert expect_golden("a", kv, b"x" * 12) == b"x" * 12
    for gold in (b"", b"x" * 11, b"x" * 13):
        try:
            expect_golden("a", kv, gold)
        except ValueError:
            continue
        raise SystemExit(f"self-test: expect_golden accepted {len(gold)} bytes")
    assert expect_output("a", b"x" * 12, 12)
    for got, n in ((b"", 0), (b"", 12), (b"x" * 11, 12)):
        try:
            expect_output("a", got, n)
        except ValueError:
            continue
        raise SystemExit(f"self-test: expect_output accepted {len(got)} of {n}")
    print("ffmpeg_graph_probe self-test: 3 refusals planted, all caught")


def main():
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return
    import numpy as np

    os.makedirs(OUT, exist_ok=True)
    subprocess.run(["cc", "-O1", f"-I{FF}/include", os.path.join(ROOT, "scripts", "ffmpeg_graph_probe.c"),
                    f"-L{FF}/lib", "-lavfilter", "-lavutil", "-lswscale", f"-Wl,-rpath,{FF}/lib",
                    "-o", OUT + "probe"], check=True)
    classes = parse_classes(open(D + "facts.txt").read())
    bad = 0
    for name, kv in classes:
        gold = expect_golden(name, kv, open(D + name + ".rgb24", "rb").read())
        subprocess.run([OUT + "probe", kv["fmt"], kv["w"], kv["h"], kv["matrix"], kv["range"], D + name + ".planar",
                        OUT + "graph.rgb", OUT + "sws.rgb"], check=True, capture_output=True)
        for label, path in (("graph", "graph.rgb"), ("sws_scale flags 0", "sws.rgb")):
            if expect_output(f"{name}: {label}", open(OUT + path, "rb").read(), len(gold)) != gold:
                print(f"DIFFERS: {name}: {label} against the CLI's rgb24")
                bad += 1
    print(f"{len(classes)} classes, graph and sws_scale(flags 0) against the CLI: {bad} differing")
    turns = [("transpose=cclock", lambda a: np.rot90(a, 1)), ("transpose=clock", lambda a: np.rot90(a, -1)),
             ("hflip,vflip", lambda a: a[::-1, ::-1])]
    checked = 0
    for name, kv in classes:
        w, h = int(kv["w"]), int(kv["h"])
        gold = np.frombuffer(expect_golden(name, kv, open(D + name + ".rgb24", "rb").read()), np.uint8).reshape(h, w, 3)
        for tf, fn in turns:
            subprocess.run([OUT + "probe", kv["fmt"], str(w), str(h), kv["matrix"], kv["range"], D + name + ".planar",
                            OUT + "t.rgb", "/dev/null", tf], check=True, capture_output=True)
            want = fn(gold)
            got = np.frombuffer(expect_output(f"{name}: {tf}", open(OUT + "t.rgb", "rb").read(), want.size), np.uint8).reshape(want.shape)
            checked += 1
            if not np.array_equal(got, want):
                print(f"DIFFERS: {name}: {tf}")
                bad += 1
    print(f"{checked} graph-with-transform checks (convert in the graph, then turn): {bad} differing in all")
    sys.exit(1 if bad else 0)


main()
