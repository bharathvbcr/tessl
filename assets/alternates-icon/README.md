# Iconic marks — the PyTorch idiom

PyTorch's logo works because the *name* contains a physical object and the mark
simply **is** that object: one free-standing silhouette, one colour, no
container. These six take the same approach for `tessl`.

Colour is a single variable (`#6C4CF1`, chosen clear of PyTorch's orange-red and
JAX's green); each mark also has a flat single-colour form for favicons, print
and light/dark inversion.

| file | mark | why |
| --- | --- | --- |
| `1-tesseract.svg` | Tesseract | **tessl → tesseract** is the same hook as **torch → flame**. A cube of cubes: nesting and tiling in one of the most recognisable objects in geometry. |
| `2-tessera.svg` | Tessera | The single mosaic tile the word *tessellation* comes from. The simplest possible mark. |
| `3-penrose.svg` | Penrose pair | The two rhombs that tile a plane and never repeat. Distinctive and unmistakably about tessellation. |
| `4-cube.svg` | Isometric tile | A tile with depth — the K dimension made visible. One hexagonal silhouette. |
| `5-spread.svg` | Spreading tile | One tile and the pattern it generates. Tessellation as a verb. |
| `6-interlock.svg` | Interlock | Two identical shapes keyed together: the minimal proof that a motif tiles. |

These differ from `../alternates/` in idiom, not just shape. Those are app-icon
style — a mark inside a rounded-rect container. These are free-standing, which is
what PyTorch, JAX and Triton all use and what reads better at favicon size and on
a plain README.
