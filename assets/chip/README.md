# Chip-style marks

The tesseract rendered the way the Apple M-series marks are: a gradient mesh
with depth, in a dark/light hue pair, rather than a flat vector fill.

Two restrained palettes. The hue spread is roughly 60°, not 300° — a hue *shift*
across the mark, not a spectrum. An earlier pass ran violet→cyan→magenta→amber
and read as a rainbow; these are neighbouring hues at partial saturation.

| palette | hues |
| --- | --- |
| `indigo-*` | violet → indigo → slate-blue |
| `steel-*` | slate-blue → teal → graphite |

Four forms per palette, `dark` and `light` meaning *the background it is meant to
sit on*, not the mark's own value:

| form | use |
| --- | --- |
| `*-free-dark` | README and docs on a dark page. Transparent background. **Recommended.** |
| `*-free-light` | README and docs on a light page. Transparent background. **Recommended.** |
| `*-plate-dark` | App icon, GitHub avatar, favicon. Near-black rounded plate. |
| `*-plate-light` | Light rounded plate. **Weakest of the four** — a pale mesh inside a pale plate reads as a smudge rather than a mark. Kept for completeness; prefer `free-light` on light pages. |

## Files

SVG is the source; the PNGs are rendered from it, not drawn separately.

```
*.svg              vector source, 128×128 viewBox
png/*@1024.png     master raster
png/*@{512,256,128,64,32}.png
```

Regenerate the PNGs after editing any SVG:

```bash
# renders every SVG at 1024 then downsamples the ladder
python3 - <<'PY'
import pathlib, re, subprocess, shutil
pathlib.Path("tmp").mkdir(exist_ok=True)
for f in sorted(pathlib.Path(".").glob("*.svg")):
    pathlib.Path("tmp", f.name).write_text(
        re.sub(r'width="128" height="128"', 'width="1024" height="1024"', f.read_text(), count=1))
for f in sorted(pathlib.Path("tmp").glob("*.svg")):
    subprocess.run(["qlmanage", "-t", "-s", "1024", "-o", "tmp", str(f)],
                   capture_output=True)
    src = pathlib.Path("tmp", f.name + ".png")
    if src.exists():
        shutil.move(src, f"png/{f.stem}@1024.png")
for f in sorted(pathlib.Path("png").glob("*@1024.png")):
    for s in (512, 256, 128, 64, 32):
        subprocess.run(["sips", "-Z", str(s), str(f),
                        "--out", f"png/{f.name.replace('@1024','@'+str(s))}"],
                       capture_output=True)
shutil.rmtree("tmp")
PY
```

`qlmanage` is macOS's own renderer and handles the masks, filters and radial
gradients correctly; ImageMagick's built-in SVG renderer does not, and silently
produces a flatter image rather than failing.

## Note on generation

These are hand-authored SVG, not AI-generated raster. Image generation needs an
`OPENROUTER_API_KEY`, which is not configured on this machine — see
`~/.claude/skills/generate-image`. Vector also stays the better source for a
logo: one file drives every size, edges stay crisp at 32px, and the palette is a
few hex values rather than a re-generation.
