# Logo concepts

Four concepts × three palettes, to the spec. `generate.py` produces every SVG
and the PNG ladder; edit it rather than the generated files.

```bash
python3 generate.py     # 12 SVGs + 72 PNGs @1024/512/256/128/64/32
```

## Concepts

| | concept | reads as |
| --- | --- | --- |
| **A** | `a-tessellated-t` | Lowercase **t** from an offset grid of matrix tiles, skewed 5°. The accent tile at the crossbar intersection is the persistent register accumulator, C_tile. |
| **B** | `b-lattice-t` | A 7×7 diamond lattice whose **omitted cells spell a capital T**. Negative space as memory layout. |
| **C** | `c-morton-path` | A 4×4 tiled block threaded by the **Z-order (Morton) walk** — the real traversal from `tile_from_linear`, not a decorative curve. |
| **D** | `d-block-matrix` | A and B bar arrays overlapping into the output grid, the accumulator block solid at its centre. |

## Palettes

| | base | surface | accent | gridlines |
| --- | --- | --- | --- | --- |
| `silicon` | `#0E1117` | `#21262D` | `#00E5FF` | `#30363D` |
| `oxide` | `#0A0A0C` | `#1F2428` | `#FF5722` | `#39424A` |
| `glass` | `#000000` | `rgba(255,255,255,0.08)` | `#5865F2` | `rgba(255,255,255,0.22)` |

## Two things learned rendering these

**Gridlines are load-bearing, not decoration.** In all three palettes the
surface-to-background step is about 4% luminance. The first pass drew structural
tiles as fill only, and concepts A, B and D came out as a floating accent chip on
an empty plate — the tessellation was present in the file and invisible on
screen. Every structural tile now carries a 1.4px gridline stroke, which is what
made B legible at all.

**Composition has to fit inside the plate.** D's grid originally ran to x=127
against a plate ending at x=122, silently clipping its rightmost column. It is
now laid out inside 16..114.

`qlmanage` is macOS's own renderer and handles the masks, filters and clip paths
correctly. ImageMagick's built-in SVG path does not, and flattens them silently
rather than failing.
