#!/usr/bin/env python3
"""Build the tessl mark and wordmark lockups from the chosen concept.

Chosen: Morton walk, Silicon Precision palette (concepts/c-morton-path--silicon.svg).
Run from this directory:  python3 build_logo.py
Requires macOS (qlmanage) + Pillow.
"""
import pathlib, re, shutil, subprocess
from PIL import Image, ImageDraw, ImageFont

HERE   = pathlib.Path(__file__).parent
SOURCE = HERE / "concepts" / "c-morton-path--silicon.svg"
PNG    = HERE / "png"
FONT   = "/System/Library/Fonts/SFNSMono.ttf"
OBSIDIAN  = (0x0E, 0x11, 0x17)   # wordmark on light pages
ALUMINIUM = (0xE6, 0xED, 0xF3)   # wordmark on dark pages

def render_mark(size=1024):
    """qlmanage is the only renderer here that handles the glow filter and clip
    path, but it flattens onto opaque white — there is no alpha in its output.
    The art is a rounded rect at x=6..122, rx=26 in a 128 viewBox and the glow is
    clipped to that same path, so the alpha is rebuilt analytically rather than
    flood-filled (which would eat into near-white pixels)."""
    tmp = HERE / "_tmp"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
    svg = re.sub(r'width="128" height="128"', f'width="{size}" height="{size}"',
                 SOURCE.read_text(), count=1)
    p = tmp / "mark.svg"; p.write_text(svg)
    subprocess.run(["qlmanage", "-t", "-s", str(size), "-o", str(tmp), str(p)],
                   capture_output=True)
    im = Image.open(tmp / "mark.svg.png").convert("RGBA").resize((size, size), Image.LANCZOS)
    ss = 4
    m = Image.new("L", (size*ss, size*ss), 0)
    k = size*ss / 128.0
    ImageDraw.Draw(m).rounded_rectangle([6*k, 6*k, 122*k, 122*k], radius=26*k, fill=255)
    im.putalpha(m.resize((size, size), Image.LANCZOS))
    shutil.rmtree(tmp)
    return im

def draw_tracked(d, xy, text, font, fill, tracking):
    """Monospace advances are far too loose for a wordmark; tighten per glyph."""
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += font.getlength(ch) * (1.0 + tracking)
    return x

def lockup(mark, rgb, out, mark_px=1024, tracking=-0.14):
    font = ImageFont.truetype(FONT, int(mark_px * 0.52))
    font.set_variation_by_name("Semibold")
    width = sum(font.getlength(c) * (1 + tracking) for c in "tessl")
    asc, desc = font.getmetrics()
    gap, pad = int(mark_px * 0.13), int(mark_px * 0.05)
    canvas = Image.new("RGBA", (pad*2 + mark_px + gap + int(width), mark_px + pad*2), (0,0,0,0))
    canvas.alpha_composite(mark.resize((mark_px, mark_px), Image.LANCZOS), (pad, pad))
    d = ImageDraw.Draw(canvas)
    draw_tracked(d, (pad + mark_px + gap, pad + (mark_px - asc)//2 - int(mark_px*0.03)),
                 "tessl", font, rgb + (255,), tracking)
    canvas.crop(canvas.getbbox()).save(out)
    return Image.open(out).size

def main():
    PNG.mkdir(exist_ok=True)
    mark = render_mark(1024)
    mark.save(PNG / "logo-mark@1024.png")
    for s in (512, 256, 128, 64, 32):
        mark.resize((s, s), Image.LANCZOS).save(PNG / f"logo-mark@{s}.png")
    print("  mark      ", mark.size, "+ ladder 512/256/128/64/32")
    print("  logo      ", lockup(mark, OBSIDIAN,  PNG / "logo@2048.png"))
    print("  logo-dark ", lockup(mark, ALUMINIUM, PNG / "logo-dark@2048.png"))
    for n in ("logo", "logo-dark"):
        im = Image.open(PNG / f"{n}@2048.png")
        im.resize((900, int(im.height * 900 / im.width)), Image.LANCZOS).save(PNG / f"{n}@900.png")
    print("  900px lockups")

if __name__ == "__main__":
    main()
