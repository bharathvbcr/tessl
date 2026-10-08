"""Natural-text token ids for the 2B training tests that need no reference.

    python tools/qwen35_ref/make_text_ids.py TOKENIZER_JSON OUT_NPY [N]

Tokenizes the opening of tessl's own docs/architecture.md (prose, so the
step's gradients have the magnitudes text gives, not those of uniform random
ids) with a Qwen3.5 checkpoint's tokenizer.json and writes the first N ids
(default 512) as an int64 .npy. Needs only the `tokenizers` package; the
.npy is written by hand so numpy is not required.
"""

import struct
import sys
from pathlib import Path

from tokenizers import Tokenizer


def write_npy(path: Path, ids: list[int]) -> None:
    header = "{'descr': '<i8', 'fortran_order': False, 'shape': (%d,), }" % len(ids)
    # The header, its length field and the magic pad to a multiple of 64.
    pad = 64 - (10 + len(header) + 1) % 64
    header = header + " " * pad + "\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00")
        f.write(struct.pack("<H", len(header)))
        f.write(header.encode("latin1"))
        f.write(struct.pack("<%dq" % len(ids), *ids))


def main() -> None:
    tok_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    text = (Path(__file__).resolve().parents[2] / "docs" / "architecture.md").read_text()
    ids = Tokenizer.from_file(str(tok_path)).encode(text).ids
    if len(ids) < n:
        sys.exit(f"docs/architecture.md gives {len(ids)} tokens, fewer than {n}")
    write_npy(out, ids[:n])
    print(f"{out}: {n} ids from docs/architecture.md")


if __name__ == "__main__":
    main()
