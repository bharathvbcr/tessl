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
afterwards. Exits 1 on any difference. Needs `cc` and Homebrew ffmpeg at /opt/homebrew/opt/ffmpeg.
"""

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(ROOT, "tests", "fixtures", "media_video") + "/"
OUT = os.path.join(ROOT, "target", "graph_probe") + "/"
FF = "/opt/homebrew/opt/ffmpeg"


def main():
    os.makedirs(OUT, exist_ok=True)
    subprocess.run(["cc", "-O1", f"-I{FF}/include", os.path.join(ROOT, "scripts", "ffmpeg_graph_probe.c"),
                    f"-L{FF}/lib", "-lavfilter", "-lavutil", "-lswscale", f"-Wl,-rpath,{FF}/lib",
                    "-o", OUT + "probe"], check=True)
    classes = []
    for line in open(D + "facts.txt"):
        p = line.split()
        if p[0] == "format":
            classes.append((p[1], dict(x.split("=", 1) for x in p[2:])))
    bad = 0
    for name, kv in classes:
        gold = open(D + name + ".rgb24", "rb").read()
        subprocess.run([OUT + "probe", kv["fmt"], kv["w"], kv["h"], kv["matrix"], kv["range"], D + name + ".planar",
                        OUT + "graph.rgb", OUT + "sws.rgb"], check=True, capture_output=True)
        for label, path in (("graph", "graph.rgb"), ("sws_scale flags 0", "sws.rgb")):
            if open(OUT + path, "rb").read() != gold:
                print(f"DIFFERS: {name}: {label} against the CLI's rgb24")
                bad += 1
    print(f"{len(classes)} classes, graph and sws_scale(flags 0) against the CLI: {bad} differing")
    turns = [("transpose=cclock", lambda a: np.rot90(a, 1)), ("transpose=clock", lambda a: np.rot90(a, -1)),
             ("hflip,vflip", lambda a: a[::-1, ::-1])]
    checked = 0
    for name, kv in classes:
        w, h = int(kv["w"]), int(kv["h"])
        gold = np.frombuffer(open(D + name + ".rgb24", "rb").read(), np.uint8).reshape(h, w, 3)
        for tf, fn in turns:
            subprocess.run([OUT + "probe", kv["fmt"], str(w), str(h), kv["matrix"], kv["range"], D + name + ".planar",
                            OUT + "t.rgb", "/dev/null", tf], check=True, capture_output=True)
            want = fn(gold)
            got = np.frombuffer(open(OUT + "t.rgb", "rb").read(), np.uint8).reshape(want.shape)
            checked += 1
            if not np.array_equal(got, want):
                print(f"DIFFERS: {name}: {tf}")
                bad += 1
    print(f"{checked} graph-with-transform checks (convert in the graph, then turn): {bad} differing in all")
    sys.exit(1 if bad else 0)


main()
