# Logo alternates

Six concepts on the same theme: how a GEMM output matrix is cut into tiles and
in what order those tiles are walked. Shared palette — deep navy `#0E1526`,
blue `#5B8DEF → #3A5FCD`, amber accent `#F5A524 → #E8760A`.

| file | concept | notes |
| --- | --- | --- |
| `a-z-order-walk.svg` | Four sub-tiles with the Morton traversal drawn over them | Currently shipped. Most literal: it is `tile_from_linear`. |
| `b-resident-tile.svg` | A 4×4 output grid with one tile ringed and held | The register-resident accumulator. Busiest of the six at small sizes. |
| `c-k-slab-collapse.svg` | Four K-slabs narrowing into one stored tile | Draws the actual fix: many K blocks, one store. |
| `d-a-at-b.svg` | Tall operand, wide operand, output tile in the corner | The GEMM diagram itself. Reads instantly to anyone who knows BLAS. |
| `e-subdivision.svg` | A tile quartered, then quartered again | Tessellation and Morton recursion in one mark. |
| `f-tile-monogram.svg` | A lowercase "t" cut from tiles | Most conventional and the most legible at 16px. |

To adopt one, copy it over `assets/logo-mark.svg` and replace the mark group in
`assets/logo.svg` / `assets/logo-dark.svg` (the wordmark sits to the right of a
128×128 mark at x=0).
