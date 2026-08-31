#!/usr/bin/env python3
"""Cross-check every Rust TileGeom against the SM/SN compiled into the kernel it
dispatches. A mismatch means the host launches the wrong number of threadgroups,
silently leaving output tiles unwritten."""
import re, sys, pathlib

# Resolve everything from this script's own location, never the caller's cwd.
# `scripts/` sits directly under the crate root both in this repository and in a
# published .crate, so the audit runs from anywhere and survives packaging.
CRATE = pathlib.Path(__file__).resolve().parents[1]

# The GEMM kernels and dispatch have exactly one owner: this crate. The audit
# used to compare two mirrored copies; the mirror is gone, so what remains is
# the check that still matters — that each Rust TileGeom agrees with the SM/SN
# compiled into the kernel it dispatches, and that COOP_BKC agrees with each
# coop kernel's BKC.
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
        bk = re.search(r'constexpr int BKC = (\d+);', p)
        one = 'execution_simdgroup>' in p
        out[name] = (int(sm.group(1)) if sm else None,
                     int(sn.group(1)) if sn else None,
                     1 if one and not sg else (int(sg.group(1)) if sg else None),
                     int(bk.group(1)) if bk else None)

    for m in re.finditer(r'NN_COOP_KERNEL\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\w+)\s*\)', src):
        name = m.group(1)
        out[name] = (int(m.group(3)), int(m.group(4)), int(m.group(5)), None)

    for m in re.finditer(r'TN_NT_COOP_KERNEL\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\w+)\s*\)', src):
        name = m.group(1)
        out[name] = (int(m.group(3)), int(m.group(4)), int(m.group(5)), None)

    return out

def rust_coop_bkc(rs):
    """`COOP_BKC` is the host's promise that K divides evenly into the coop
    kernels' K loop. If it drifts above a kernel's own BKC the loop silently
    drops the tail: the gate admits K, the kernel computes only part of it, and
    the result is quietly wrong with no error anywhere."""
    src = strip_comments(pathlib.Path(rs).read_text())
    m = re.search(r'const COOP_BKC: usize = (\d+);', src)
    return int(m.group(1)) if m else None

def rust_tiles(rs):
    src = strip_comments(pathlib.Path(rs).read_text())
    consts = {}
    for m in re.finditer(r'const (TILE_\w+): TileGeom = TileGeom \{\s*sm: (\d+),\s*sn: (\d+),\s*simdgroups: (\d+),?\s*\}', src):
        consts[m.group(1)] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
    # pipeline("name") ... TILE_X  within a small window
    pairs = []
    lines = src.split('\n')
    for i, l in enumerate(lines):
        km = re.search(r'pipeline\("([a-z0-9_]+)"\)', l)
        if not km: continue
        kname = km.group(1)
        if not kname.startswith("matmul"): continue
        for j in range(i, min(i + 8, len(lines))):
            tm = re.search(r'\b(TILE_\w+)\b', lines[j])
            if tm:
                pairs.append((kname, tm.group(1), j + 1)); break
    return consts, pairs

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

bad = 0
for crate in CRATES:
    kt = kernel_tiles(crate / "kernels/matmul_tensorops.metal")
    consts, pairs = rust_tiles(crate / "src/gemm.rs")
    coop_bkc = rust_coop_bkc(crate / "src/gemm.rs")
    print(f"\n=== {crate}")
    print(f"    tile constants: {consts}")
    print(f"    COOP_BKC = {coop_bkc}")

    # Every `*_coop` kernel must be pinned above, or it escapes this audit
    # entirely (they are dispatched via a variable, never pipeline("literal")).
    pinned = {k for k, _ in NN_PAIRS}
    for kern in kt:
        if kern.endswith("_coop") and kern not in pinned:
            print(f"  MISMATCH  {kern:<40} not pinned in NN_PAIRS (audit would skip it)")
            bad += 1

    pairs = [(k, t, "pinned") for k, t in NN_PAIRS] + pairs
    for kern, tile, line in pairs:
        if kern not in kt:
            print(f"  ?  {kern:<44} {tile:<20} (kernel not in matmul_tensorops.metal)"); continue
        ksm, ksn, ksg, kbkc = kt[kern]
        if kbkc is not None and coop_bkc is not None and kbkc != coop_bkc:
            print(f"  MISMATCH  {kern:<40} BKC={kbkc} but Rust COOP_BKC={coop_bkc} "
                  f"— gate would admit K values whose tail the kernel drops")
            bad += 1
        elif kern.endswith("_coop") and kbkc is None:
            print(f"  MISMATCH  {kern:<40} has no `constexpr int BKC` to check "
                  f"against COOP_BKC")
            bad += 1
        if ksm is None:
            print(f"  -  {kern:<44} {tile:<20} (no compile-time SM/SN)"); continue
        rsm, rsn, rsg = consts[tile]
        ok = (ksm, ksn) == (rsm, rsn) and (ksg is None or ksg == rsg)
        if not ok: bad += 1
        print(f"  {'OK' if ok else 'MISMATCH'}  {kern:<44} {tile:<18} "
              f"kernel={ksm}x{ksn}/sg{ksg}  rust={rsm}x{rsn}/sg{rsg}  ({line})")
print(f"\n{'FAIL' if bad else 'PASS'}: {bad} mismatch(es)")
sys.exit(1 if bad else 0)
