#!/usr/bin/env python3
"""Cross-check every Rust TileGeom against the SM/SN compiled into the kernel it
dispatches. A mismatch means the host launches the wrong number of threadgroups,
silently leaving output tiles unwritten."""
import pathlib
import re
import sys

# Resolve everything from this script's own location, never the caller's cwd.
# `scripts/` sits directly under the crate root both in this repository and in a
# published .crate, so the audit runs from anywhere and survives packaging.
CRATE = pathlib.Path(__file__).resolve().parents[1]

# The GEMM kernels and dispatch have exactly one owner: this crate. The audit
# used to compare two mirrored copies; the mirror is gone, so what remains is
# the check that still matters — that each Rust TileGeom agrees with the SM/SN
# and simdgroup count compiled into the kernel it dispatches.
#
# tessl-arch02 compiles these same sources through DEP_TESSL_KERNELS, so it
# cannot drift by construction. A local copy reappearing there is itself the
# regression, so fail if one shows up (skipped when the sibling is absent, as
# it is inside a published crate).
CRATES = [CRATE]
_stale = CRATE.parents[2] / "arch_02_value_resid/metal-native/kernels/matmul_tensorops.metal"
if _stale.exists():
    raise SystemExit(
        "audit_gemm_tiles: tessl-arch02 has its own matmul_tensorops.metal again. "
        "It must compile tessl's copy via DEP_TESSL_KERNELS; a local one silently drifts."
    )

def strip_comments(src):
    """Comments carry example geometries that would otherwise match as code."""
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
    return re.sub(r'//[^\n]*', '', src)

def kernel_tiles(metal):
    src = strip_comments(pathlib.Path(metal).read_text())
    out = {}
    parts = re.split(r'(?m)^kernel void ', src)
    for p in parts[1:]:
        name = p.split('(')[0].strip()
        sm = re.search(r'constexpr int SM = (\d+);', p)
        sn = re.search(r'constexpr int SN = (\d+);', p)
        sg = re.search(r'execution_simdgroups<(\d+)>', p)
        one = 'execution_simdgroup>' in p
        out[name] = (int(sm.group(1)) if sm else None,
                     int(sn.group(1)) if sn else None,
                     1 if one and not sg else (int(sg.group(1)) if sg else None))

    for m in re.finditer(r'NN_COOP_KERNEL\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\w+)\s*\)', src):
        name = m.group(1)
        out[name] = (int(m.group(3)), int(m.group(4)), int(m.group(5)))

    for m in re.finditer(r'TN_NT_COOP_KERNEL\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\w+)\s*\)', src):
        name = m.group(1)
        out[name] = (int(m.group(3)), int(m.group(4)), int(m.group(5)))

    return out

def rust_tiles(rs):
    src = strip_comments(pathlib.Path(rs).read_text())
    consts = {}
    for m in re.finditer(r'const (TILE_\w+): TileGeom = TileGeom \{\s*sm: (\d+),\s*sn: (\d+),\s*simdgroups: (\d+),?\s*\}', src):
        consts[m.group(1)] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
    # pipeline("name") ... TILE_X  within a small window
    pairs = []
    lines = src.split('\n')
    for i, line in enumerate(lines):
        km = re.search(r'pipeline\("([a-z0-9_]+)"\)', line)
        if not km:
            continue
        kname = km.group(1)
        if not kname.startswith("matmul"):
            continue
        for j in range(i, min(i + 8, len(lines))):
            tm = re.search(r'\b(TILE_\w+)\b', lines[j])
            if tm:
                pairs.append((kname, tm.group(1), j + 1))
                break
    return consts, pairs


def geometry_error(kernel_tiles_by_name, rust_tile_constants, kernel, tile):
    """Return why one host/kernel geometry cannot be verified, or ``None``.

    Missing data is an error, not an unaudited row: the old audit printed
    ``COOP_BKC = None`` and still returned PASS because both sides of that
    retired comparison were optional.
    """
    if kernel not in kernel_tiles_by_name:
        return f"kernel {kernel!r} is absent from matmul_tensorops.metal"
    if tile not in rust_tile_constants:
        return f"Rust tile constant {tile!r} is absent from gemm.rs"
    ksm, ksn, ksg = kernel_tiles_by_name[kernel]
    if ksm is None or ksn is None or ksg is None:
        return f"kernel {kernel!r} has incomplete SM/SN/simdgroup geometry"
    rsm, rsn, rsg = rust_tile_constants[tile]
    if (ksm, ksn, ksg) != (rsm, rsn, rsg):
        return (
            f"kernel {kernel!r} is {ksm}x{ksn}/sg{ksg}, but {tile} is "
            f"{rsm}x{rsn}/sg{rsg}"
        )
    return None

# NN and coop paths select their kernel through helper functions / expressions,
# so they do not always appear as a literal pipeline("name") next to TILE_*,
# and are pinned by hand.
NN_PAIRS = [
    ("matmul2d_tensorops_f32", "TILE_F32"),
    ("matmul2d_tensorops_bf16_f32", "TILE_COOP_DEFAULT"),
    ("matmul2d_tensorops_bf16_f32_64x64_sg4", "TILE_COOP_NARROW"),
    ("matmul2d_tensorops_f32_relaxed", "TILE_COOP_DEFAULT"),
    ("matmul2d_tensorops_f32_relaxed_64x64_sg4", "TILE_COOP_NARROW"),
    ("matmul2d_tensorops_tn_bf16_f32", "TILE_COOP_TN_NT"),
    ("matmul2d_tensorops_nt_bf16_f32", "TILE_COOP_TN_NT"),
    ("matmul2d_tensorops_tn_accum_bf16_f32", "TILE_COOP_ACCUM"),
    ("matmul2d_tensorops_nt_accum_bf16_f32", "TILE_COOP_ACCUM"),
]

def main():
    bad = 0
    for crate in CRATES:
        kt = kernel_tiles(crate / "kernels/matmul_tensorops.metal")
        consts, pairs = rust_tiles(crate / "src/gemm.rs")
        print(f"\n=== {crate}")
        print(f"    tile constants: {consts}")

        pairs = [(k, t, "pinned") for k, t in NN_PAIRS] + pairs
        for kern, tile, line in pairs:
            error = geometry_error(kt, consts, kern, tile)
            if error is not None:
                print(f"  MISMATCH  {kern:<44} {tile:<18} ({line}): {error}")
                bad += 1
                continue
            ksm, ksn, ksg = kt[kern]
            rsm, rsn, rsg = consts[tile]
            print(f"  OK  {kern:<44} {tile:<18} "
                  f"kernel={ksm}x{ksn}/sg{ksg}  rust={rsm}x{rsn}/sg{rsg}  ({line})")
    print(f"\n{'FAIL' if bad else 'PASS'}: {bad} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
