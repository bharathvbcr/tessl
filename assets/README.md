# Brand assets

**Mark:** Morton walk · **Palette:** Silicon Precision.

A 4×4 tiled block threaded by the Z-order walk the kernels actually use —
`tile_from_linear` in `matmul_tensorops.metal`. The mark and the code agree.

| | |
| --- | --- |
| Base | Deep Obsidian Slate `#0E1117` |
| Surfaces | Anodized Space Gray `#21262D` |
| Accent | Electric Aqua `#00E5FF` |
| Gridlines | `#30363D` · wordmark on dark: Brushed Aluminum `#E6EDF3` |

## Files

```
logo-mark.svg            vector source (128×128)
png/logo-mark@{1024,512,256,128,64,32}.png    icon / avatar / favicon
png/logo@{2048,900}.png       wordmark lockup for light pages
png/logo-dark@{2048,900}.png  wordmark lockup for dark pages
build_logo.py            regenerates every PNG above
concepts/                all four concepts × three palettes, plus generate.py
```

```bash
python3 build_logo.py
```

## Why the wordmark is PNG and not SVG

An SVG `<text>` element renders with whatever font the viewer has. GitHub serves
README SVGs through `<img>`, so the wordmark would reflow or fall back on any
machine without SF Mono. The lockups are composed in Pillow with the font baked
to pixels; the mark stays vector.

## Two renderer facts worth keeping

`qlmanage` is macOS's own renderer and the only one here that handles the glow
filter and clip path. ImageMagick's built-in SVG path flattens them silently
rather than failing, and its `rsvg-convert` delegate is not installed.

**`qlmanage` produces no alpha** — it composites onto opaque white. `build_logo.py`
rebuilds the alpha analytically from the known plate geometry (rounded rect at
x=6..122, rx=26 in a 128 viewBox, with the glow clipped to that same path) rather
than flood-filling, which would eat into near-white pixels.

## Superseded

Three earlier exploration sets (`alternates/`, `alternates-icon/`, `chip/`) were
removed once the Morton/silicon mark was chosen. They are in git history:

```bash
git log --diff-filter=D --name-only -- assets/
git checkout <commit>^ -- assets/chip
```
