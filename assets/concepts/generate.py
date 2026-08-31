#!/usr/bin/env python3
"""Generate the tessl logo concepts: 4 marks x 3 palettes, SVG then PNG.

Run from this directory:  python3 generate.py
Requires macOS (qlmanage renders the SVG; sips builds the size ladder).
"""
import pathlib, re, shutil, subprocess

PALETTES = {
    # Silicon Precision — Metal 4 / hardware
    "silicon": dict(bg="#0E1117", surface="#21262D", accent="#00E5FF",
                    light="#E6EDF3", grid="#30363D", glow=0.55),
    # Rust Oxide & Cold Titanium
    "oxide":   dict(bg="#0A0A0C", surface="#1F2428", accent="#FF5722",
                    light="#F0F6FC", grid="#39424A", glow=0.45),
    # Glassmorphic Matrix
    "glass":   dict(bg="#000000", surface="rgba(255,255,255,0.08)", accent="#5865F2",
                    light="#FFFFFF", grid="rgba(255,255,255,0.22)", glow=0.75),
}

def head(u, p, title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="128" height="128"
     role="img" aria-label="tessl — {title}">
  <title>tessl — {title}</title>
  <defs>
    <filter id="{u}glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="3.2" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <clipPath id="{u}clip"><rect x="6" y="6" width="116" height="116" rx="26"/></clipPath>
  </defs>
  <rect x="6" y="6" width="116" height="116" rx="26" fill="{p['bg']}"/>
  <g clip-path="url(#{u}clip)">
'''

def foot(p):
    return ("  </g>\n"
            f'  <rect x="6.5" y="6.5" width="115" height="115" rx="25.5" fill="none" '
            f'stroke="{p["grid"]}" stroke-width="1"/>\n</svg>\n')

def tile(x, y, w, h, p, kind="surface", r=2.6):
    """Gridlines go on every structural tile in every palette. Surface-on-bg is
    only a ~4% luminance step in these palettes: without the stroke the tiling
    is technically present and visually absent."""
    if kind == "accent":
        return f'    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{p["accent"]}"/>\n'
    op = ' opacity="0.62"' if kind == "ghost" else ""
    return (f'    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{p["surface"]}" '
            f'stroke="{p["grid"]}" stroke-width="1.4"{op}/>\n')

# A — lowercase t from an offset grid of matrix tiles. The crossbar spans the
# full width: a 1-cell-either-side bar on a tall stem reads as a domino column.
# The focal tile at the intersection is the register accumulator, C_tile.
T_CELLS = [(2,0),(0,1),(1,1),(2,1),(3,1),(4,1),(2,2),(2,3),(2,4),(3,4)]
def concept_a(u, p):
    s, g, x0, y0 = 18, 3.0, 17, 20
    body = '  <g transform="skewX(-5) translate(4,0)">\n'
    for (c, r) in T_CELLS:
        body += tile(x0+c*(s+g), y0+r*(s+g), s, s, p,
                     "accent" if (c, r) == (2, 1) else "surface", r=3.0)
    body += "  </g>\n"
    fx, fy = x0+2*(s+g), y0+1*(s+g)
    body += (f'  <g transform="skewX(-5) translate(4,0)" filter="url(#{u}glow)" '
             f'opacity="{p["glow"]}">\n' + tile(fx, fy, s, s, p, "accent", r=3.0) + "  </g>\n")
    return body

# B — 7x7 diamond lattice whose omitted cells spell a capital T. A sparse
# lattice cannot carry negative space, so this is dense and the tiles are lit.
T_NEG = {(c,1) for c in range(1,6)} | {(3,r) for r in range(2,6)}
def concept_b(u, p):
    body, step, x0, y0, half = "", 15.5, 17, 17, 6.6
    for r in range(7):
        for c in range(7):
            if (c, r) in T_NEG:
                continue
            cx, cy = x0+c*step, y0+r*step
            body += (f'    <path d="M{cx} {cy-half} L{cx+half} {cy} L{cx} {cy+half} '
                     f'L{cx-half} {cy} Z" fill="{p["surface"]}" stroke="{p["grid"]}" '
                     f'stroke-width="1.3"/>\n')
    d = (f'M{x0+0.5*step} {y0+0.35*step} L{x0+5.5*step} {y0+0.35*step} '
         f'L{x0+5.5*step} {y0+1.6*step} L{x0+3.62*step} {y0+1.6*step} '
         f'L{x0+3.62*step} {y0+5.6*step} L{x0+2.38*step} {y0+5.6*step} '
         f'L{x0+2.38*step} {y0+1.6*step} L{x0+0.5*step} {y0+1.6*step} Z')
    for extra in (f' filter="url(#{u}glow)" opacity="{p["glow"]}"', ""):
        body += (f'    <path d="{d}" fill="none" stroke="{p["accent"]}" stroke-width="2.6" '
                 f'stroke-linejoin="round"{extra}/>\n')
    return body

# C — 4x4 tiles threaded by the Z-order walk the kernels actually use
# (`tile_from_linear` in matmul_tensorops.metal).
def morton_order(n=4):
    return [((i & 1) | ((i >> 1) & 2), ((i >> 1) & 1) | ((i >> 2) & 2)) for i in range(n*n)]
def concept_c(u, p):
    s, g, x0, y0 = 20, 4, 22, 22
    body = "".join(tile(x0+c*(s+g), y0+r*(s+g), s, s, p) for r in range(4) for c in range(4))
    d = " ".join(("M" if i == 0 else "L") + f"{x0+x*(s+g)+s/2} {y0+y*(s+g)+s/2}"
                 for i, (x, y) in enumerate(morton_order()))
    for extra in (f' filter="url(#{u}glow)" opacity="{p["glow"]}"', ""):
        body += (f'    <path d="{d}" fill="none" stroke="{p["accent"]}" stroke-width="3.4" '
                 f'stroke-linecap="round" stroke-linejoin="round"{extra}/>\n')
    return body

# D — A and B bar arrays overlapping into the output grid, the accumulator block
# solid at its centre. Sized to sit inside 16..114; an earlier pass ran the grid
# to x=127 against a plate ending at 122 and clipped its rightmost column.
def concept_d(u, p):
    body, bw, g = "", 13, 3.5
    gx = gy = 56
    for i in range(4):
        body += tile(16, gy+i*(bw+g), 32, bw, p, "ghost", r=2.0)
        body += tile(gx+i*(bw+g), 16, bw, 32, p, "ghost", r=2.0)
    for r in range(4):
        for c in range(4):
            body += tile(gx+c*(bw+g), gy+r*(bw+g), bw, bw, p, "surface", r=2.0)
    bx, by, bs = gx+(bw+g), gy+(bw+g), bw*2+g
    body += (f'  <g filter="url(#{u}glow)" opacity="{p["glow"]}">\n'
             + tile(bx, by, bs, bs, p, "accent", r=2.8) + "  </g>\n")
    body += tile(bx, by, bs, bs, p, "accent", r=2.8)
    return body

CONCEPTS = [("a-tessellated-t", "tessellated t", concept_a),
            ("b-lattice-t", "lattice negative-space T", concept_b),
            ("c-morton-path", "Morton walk", concept_c),
            ("d-block-matrix", "cooperative block matrix", concept_d)]

def main():
    here = pathlib.Path(__file__).parent
    for cname, ctitle, fn in CONCEPTS:
        for pname, p in PALETTES.items():
            u = f"{cname[0]}{pname[0]}"
            (here / f"{cname}--{pname}.svg").write_text(
                head(u, p, f"{ctitle} · {pname}") + fn(u, p) + foot(p))
    print(f"  {len(CONCEPTS)*len(PALETTES)} SVGs")

    png, tmp = here / "png", here / "tmp"
    shutil.rmtree(png, ignore_errors=True); shutil.rmtree(tmp, ignore_errors=True)
    png.mkdir(); tmp.mkdir()
    # qlmanage honours the SVG's own width/height, so render from a scaled copy.
    for f in sorted(here.glob("*.svg")):
        (tmp / f.name).write_text(
            re.sub(r'width="128" height="128"', 'width="1024" height="1024"',
                   f.read_text(), count=1))
    for f in sorted(tmp.glob("*.svg")):
        subprocess.run(["qlmanage", "-t", "-s", "1024", "-o", str(tmp), str(f)],
                       capture_output=True)
        out = tmp / (f.name + ".png")
        if out.exists():
            shutil.move(str(out), str(png / f"{f.stem}@1024.png"))
    for f in sorted(png.glob("*@1024.png")):
        for s in (512, 256, 128, 64, 32):
            subprocess.run(["sips", "-Z", str(s), str(f), "--out",
                            str(png / f.name.replace("@1024", f"@{s}"))], capture_output=True)
    shutil.rmtree(tmp)
    print(f"  {len(list(png.glob('*.png')))} PNGs")

if __name__ == "__main__":
    main()
