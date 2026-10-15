#!/usr/bin/env python3
"""Resolve the metallib path a benchmark binary embeds.

The drift policy, provenance and atomic publication that used to live here
moved to `src/bin/common/evidence.rs` (driven by `bench_paired --evidence`).
What stays is the one helper `bench/kernel_coverage.py` imports.

Note: the runtime now bakes the metallib in by `include_bytes!` and a current
binary carries no path, so this raises `ValueError` ("found []") on one. That
is `kernel_coverage.py`'s to resolve (it needs the library's bytes' source, not
a path), and is why this module was not extended.
"""

from __future__ import annotations

import os

# macOS PATH_MAX is below this, while the explicit bound prevents a malformed
# or unrelated binary from turning suffix discovery into an unbounded reverse
# scan. Cargo/Rust environment string literals are UTF-8, so non-UTF-8 paths
# cannot be embedded by `env!` and are deliberately outside this contract.
MAX_EMBEDDED_METALLIB_PATH_BYTES = 4096
MAX_EMBEDDED_METALLIB_MARKERS = 1024
MAX_EMBEDDED_METALLIB_PATH_PROBES = 16384


def embedded_metallib_path(binary_path: str) -> str:
    """Resolve the bounded absolute UTF-8 metallib path compiled into a binary.

    A POSIX path component may contain every byte except NUL and ``/``. Do not
    guess a filename alphabet: locate each ``.metallib`` suffix, consider the
    bounded slash-started substrings ending there, and accept only an existing
    file. Multiple distinct live candidates fail closed rather than selecting
    whichever byte string happened to appear first in the executable.
    """
    try:
        with open(binary_path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise ValueError(f"cannot inspect benchmark binary {binary_path}: {exc}") from exc

    suffix = b".metallib"
    marker_count = data.count(suffix)
    if marker_count > MAX_EMBEDDED_METALLIB_MARKERS:
        raise ValueError(
            f"embedded metallib scan found {marker_count} suffix markers, above the "
            f"bounded limit {MAX_EMBEDDED_METALLIB_MARKERS}"
        )

    candidates = set()
    probes = 0
    cursor = 0
    while True:
        suffix_start = data.find(suffix, cursor)
        if suffix_start < 0:
            break
        suffix_end = suffix_start + len(suffix)
        window_start = max(0, suffix_end - MAX_EMBEDDED_METALLIB_PATH_BYTES)
        slash = data.find(b"/", window_start, suffix_start + 1)
        while slash >= 0:
            probes += 1
            if probes > MAX_EMBEDDED_METALLIB_PATH_PROBES:
                raise ValueError(
                    "embedded metallib path scan exceeded the bounded candidate-probe "
                    f"limit {MAX_EMBEDDED_METALLIB_PATH_PROBES}"
                )
            raw = data[slash:suffix_end]
            try:
                path = raw.decode("utf-8", errors="strict")
                if os.path.isfile(path):
                    candidates.add(os.path.realpath(path))
            except (OSError, ValueError, UnicodeError):
                # Most slash-started substrings are not paths; malformed UTF-8,
                # embedded NULs, overlong components, and inaccessible names are
                # ordinary rejected candidates, not alternate interpretations.
                pass
            slash = data.find(b"/", slash + 1, suffix_start + 1)
        cursor = suffix_end

    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one existing metallib path embedded in {binary_path}, "
            f"found {sorted(candidates)}"
        )
    return candidates.pop()
