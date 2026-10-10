#!/usr/bin/env python3
"""Cross-check every Rust TileGeom against the SM/SN compiled into the kernel it
dispatches. A mismatch means the host launches the wrong number of threadgroups,
silently leaving output tiles unwritten.

Every kernel entry point in matmul_tensorops.metal must be checked or named in
EXEMPT with a reason, and an audit that examined nothing fails: a check that
could not run must not print what a check that ran and passed prints."""
import re
import sys
import pathlib

# Resolve everything from this script's own location, never the caller's cwd.
# `scripts/` sits directly under the crate root both in this repository and in a
# published .crate, so the audit runs from anywhere and survives packaging.
CRATE = pathlib.Path(__file__).resolve().parents[1]

# The GEMM kernels and dispatch have exactly one owner: this crate. The audit
# used to compare two mirrored copies; the mirror is gone, so what remains is
# the check that still matters — that each Rust TileGeom agrees with the SM/SN
# compiled into the kernel it dispatches.
#
# There is no K-block relationship to check alongside it. Every kernel here
# passes K to `matmul2d_descriptor` as `dynamic_length_v<int>` and reduces the
# whole of it inside `op.run()`, so there is no `BKC` loop whose tail a host
# gate could drop. The `COOP_BKC`/`BKC` half this script once carried described
# the blocked kernels of the code tessl was extracted from; no commit of tessl
# ever defined either (GAP-TESSL-AUDIT-BKC-CHECK-DEAD).
#
# tessl-arch02 compiles these same sources through DEP_TESSL_KERNELS, so it
# cannot drift by construction. A local copy reappearing there is itself the
# regression, so fail if one shows up (skipped when the sibling is absent, as
# it is inside a published crate).
_stale = CRATE.parents[2] / "arch_02_value_resid/metal-native/kernels/matmul_tensorops.metal"
if _stale.exists():
    raise SystemExit(
        "audit_gemm_tiles: tessl-arch02 has its own matmul_tensorops.metal again. "
        "It must compile tessl's copy via DEP_TESSL_KERNELS; a local one silently drifts."
    )

METAL = CRATE / "kernels/matmul_tensorops.metal"
GEMM_RS = CRATE / "src/gemm.rs"
# Every Rust source is scanned for literal `pipeline("matmul2d_tensorops_...")`
# dispatches, not only gemm.rs: `nn::gemm_i8_dequant` dispatches one from
# src/nn.rs, and a site this audit did not scan is a site it did not check.
SRC = CRATE / "src"

# How far below a `pipeline("name")` line to look for the TileGeom it is
# dispatched with. 12 reaches the split-K and accumulate call sites, whose tile
# argument sits 8-10 lines down; every literal site pairs with its dispatch tile,
# and one that pairs with none fails the audit.
PAIR_WINDOW = 12

# NN and coop paths select their kernel through helper functions / expressions,
# so they do not always appear as a literal pipeline("name") next to TILE_*,
# and are pinned by hand.
NN_PAIRS = [
    ("matmul2d_tensorops_f32", "TILE_F32"),
    ("matmul2d_tensorops_bf16_f32", "TILE_COOP_DEFAULT"),
    ("matmul2d_tensorops_bf16_f32_64x64_sg4", "TILE_COOP_NARROW"),
    ("matmul2d_tensorops_f16_f32", "TILE_COOP_DEFAULT"),
    ("matmul2d_tensorops_f16_f32_64x64_sg4", "TILE_COOP_NARROW"),
    ("matmul2d_tensorops_f32_relaxed", "TILE_COOP_DEFAULT"),
    ("matmul2d_tensorops_f32_relaxed_64x64_sg4", "TILE_COOP_NARROW"),
    ("matmul2d_tensorops_bf16_f32_batched", "TILE_COOP_DEFAULT"),
    ("matmul2d_tensorops_f16_f32_batched", "TILE_COOP_DEFAULT"),
    ("matmul2d_tensorops_f32_relaxed_batched", "TILE_COOP_DEFAULT"),
    ("matmul2d_tensorops_tn_bf16_f32", "TILE_COOP_TN_NT"),
    ("matmul2d_tensorops_nt_bf16_f32", "TILE_COOP_TN_NT"),
    ("matmul2d_tensorops_tn_accum_bf16_f32", "TILE_COOP_ACCUM"),
    ("matmul2d_tensorops_nt_accum_bf16_f32", "TILE_COOP_ACCUM"),
    ("matmul2d_tensorops_bf16_f32_epi", "TILE_COOP_DEFAULT"),
    ("matmul2d_tensorops_bf16_f32_epi_64x64_sg4", "TILE_COOP_NARROW"),
    ("matmul2d_tensorops_f16_f32_epi", "TILE_COOP_DEFAULT"),
    ("matmul2d_tensorops_f32_relaxed_epi", "TILE_COOP_DEFAULT"),
]

# Kernels this audit deliberately does not check, each with the reason. An
# entry whose kernel no longer exists fails the audit, so this cannot rot into
# a list of names that once meant something.
EXEMPT = {}

def strip_comments(src):
    """Comments carry example geometries that would otherwise match as code."""
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
    return re.sub(r'//[^\n]*', '', src)

def braced_body(src, start):
    """The `{...}` block that opens at or after `start`, or None if unbalanced."""
    open_at = src.find('{', start)
    if open_at < 0:
        return None
    depth = 0
    for i in range(open_at, len(src)):
        if src[i] == '{':
            depth += 1
        elif src[i] == '}':
            depth -= 1
            if depth == 0:
                return src[open_at:i + 1]
    return None

def template_defaults(src):
    """Helpers whose template parameters default SM, SN and NSG. A kernel that
    calls one with no template argument list runs that default geometry."""
    out = {}
    for m in re.finditer(r'template\s*<([^>]*)>\s*inline\s+\w+\s+(\w+)\s*\(', src):
        params, name = m.group(1), m.group(2)
        vals = [re.search(rf'\bint\s+{p}\s*=\s*(\d+)', params) for p in ("SM", "SN", "NSG")]
        if all(vals):
            out[name] = tuple(int(v.group(1)) for v in vals)
    return out

def kernel_tiles(metal):
    """{kernel: (SM, SN, simdgroups, source)} for every entry point, plus the
    `*_KERNEL(...)` macro invocations that could not be parsed. SM is None when
    the geometry is not a compile-time constant this parser can see."""
    src = strip_comments(pathlib.Path(metal).read_text())
    out, unparsed = {}, []
    defaults = template_defaults(src)

    for m in re.finditer(r'(?m)^kernel void (\w+)\s*\(', src):
        name = m.group(1)
        body = braced_body(src, m.end()) or ""
        sm = re.search(r'constexpr int SM = (\d+);', body)
        sn = re.search(r'constexpr int SN = (\d+);', body)
        sg = re.search(r'execution_simdgroups<(\d+)>', body)
        one = 'execution_simdgroup>' in body
        if sm and sn:
            out[name] = (int(sm.group(1)), int(sn.group(1)),
                         1 if one and not sg else (int(sg.group(1)) if sg else None),
                         "constexpr")
            continue
        helper = next((h for h in defaults if re.search(rf'\b{h}\s*\(', body)), None)
        if helper:
            out[name] = (*defaults[helper], f"{helper} template defaults")
        else:
            out[name] = (None, None, None, "no compile-time SM/SN")

    # Macro-stamped kernels: NAME(KERNEL, X, SM, SN, NSG, Y). Any `*_KERNEL(`
    # invocation that does not fit that shape is reported rather than skipped,
    # or a new macro would drop its kernels from the audit without a word.
    shape = re.compile(r'\(\s*(\w+)\s*,\s*\w+\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*\w+\s*\)')
    for m in re.finditer(r'(?m)^[ \t]*(\w+_KERNEL)\s*(?=\()', src):
        args = shape.match(src, m.end())
        if not args:
            line = src[m.start():src.find('\n', m.start())].strip()
            unparsed.append(f"{m.group(1)}: {line}")
            continue
        out[args.group(1)] = (int(args.group(2)), int(args.group(3)), int(args.group(4)),
                              m.group(1))
    return out, unparsed

def rust_tiles(gemm_rs, src_dir):
    """TileGeom constants from gemm.rs, the (kernel, TILE_*, site) pairs from
    every literal `pipeline("matmul2d_tensorops_...")` under `src_dir`, and the
    literal sites with no TILE_* within PAIR_WINDOW lines."""
    consts = {}
    src = strip_comments(gemm_rs.read_text())
    for m in re.finditer(r'const (TILE_\w+): TileGeom = TileGeom \{\s*sm: (\d+),\s*sn: (\d+),\s*simdgroups: (\d+),?\s*\}', src):
        consts[m.group(1)] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
    pairs, unpaired = [], []
    for rs in sorted(src_dir.rglob('*.rs')):
        rel = rs.relative_to(CRATE)
        lines = strip_comments(rs.read_text()).split('\n')
        for i, line in enumerate(lines):
            km = re.search(r'pipeline\("(matmul2d_tensorops_[a-z0-9_]+)"\)', line)
            if not km:
                continue
            for j in range(i, min(i + PAIR_WINDOW, len(lines))):
                tm = re.search(r'\b(TILE_\w+)\b', lines[j])
                if tm:
                    pairs.append((km.group(1), tm.group(1), f"{rel}:{j + 1}"))
                    break
            else:
                unpaired.append((km.group(1), f"{rel}:{i + 1}"))
    return consts, pairs, unpaired

bad = 0
kt, unparsed = kernel_tiles(METAL)
consts, pairs, unpaired = rust_tiles(GEMM_RS, SRC)
print(f"\n=== {CRATE}")
print(f"    tile constants: {consts}")
print(f"    kernels in matmul_tensorops.metal: {len(kt)}")

for u in unparsed:
    print(f"  FAIL  unparsed kernel macro {u}")
    bad += 1
# A literal dispatch with no TileGeom beside it launches a geometry this audit
# cannot see. Its kernel may still be checked through another site, which is
# exactly why the site itself has to fail.
for kern, site in unpaired:
    print(f"  FAIL  {kern:<44} no TILE_* within {PAIR_WINDOW} lines of pipeline() ({site})")
    bad += 1

pairs = [(k, t, "pinned") for k, t in NN_PAIRS] + pairs
checked, n_pairs = set(), 0
for kern, tile, line in pairs:
    if kern not in kt:
        unchecked = "kernel not in matmul_tensorops.metal"
    elif tile not in consts:
        unchecked = "no such TileGeom in src/gemm.rs"
    elif kt[kern][0] is None:
        unchecked = f"{kt[kern][3]}: cannot check"
    else:
        unchecked = None
    if unchecked:
        print(f"  FAIL  {kern:<44} {tile:<18} {unchecked} ({line})")
        bad += 1
        continue
    ksm, ksn, ksg, _ = kt[kern]
    rsm, rsn, rsg = consts[tile]
    ok = (ksm, ksn) == (rsm, rsn) and (ksg is None or ksg == rsg)
    n_pairs += 1
    checked.add(kern)
    if not ok:
        bad += 1
    print(f"  {'OK' if ok else 'MISMATCH'}  {kern:<44} {tile:<18} "
          f"kernel={ksm}x{ksn}/sg{ksg}  rust={rsm}x{rsn}/sg{rsg}  ({line})")

# Every entry point is checked or exempt. Kernels the host selects through a
# variable never sit next to a pipeline("literal"), so without this they would
# escape the audit entirely.
for kern in sorted(kt):
    if kern in EXEMPT:
        if kern in checked:
            print(f"  FAIL  {kern:<44} both checked and EXEMPT; drop the exemption")
            bad += 1
        else:
            print(f"  EXEMPT  {kern:<42} {EXEMPT[kern]}")
    elif kern not in checked:
        print(f"  FAIL  {kern:<44} not checked: pin it in NN_PAIRS or EXEMPT it with a reason")
        bad += 1
for kern in sorted(set(EXEMPT) - set(kt)):
    print(f"  FAIL  {kern:<44} EXEMPT names a kernel that no longer exists")
    bad += 1

if not kt or n_pairs == 0:
    print("  FAIL  examined nothing: no kernels parsed or no Rust dispatch pairs checked")
    bad += 1

exempt = sorted(set(EXEMPT) & set(kt))
coverage = (f"tile geometry: {len(checked)} kernels, {n_pairs} Rust dispatch pairs; "
            f"{len(exempt)} exempt ({', '.join(exempt) or 'none'})")
print(f"\nFAIL: {bad} problem(s); {coverage}" if bad else f"\nPASS: {coverage}")
sys.exit(1 if bad else 0)
