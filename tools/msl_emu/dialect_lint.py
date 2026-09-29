#!/usr/bin/env python3
"""Flag MSL constructs in the Qwen3.5 kernels that no compiling kernel uses.

    python3 tools/msl_emu/dialect_lint.py

The emulator compiles the kernels as C++, so it cannot see a construct Apple's
Metal compiler rejects. The next best evidence without a Mac: every function,
qualified name, attribute, cast and language feature the Qwen3.5 kernels use
is either already used by one of tessl's other kernels (which compile on every
release build) or listed in REVIEWED below with the reason it is standard MSL.
A new construct that is neither fails this check, so it gets a deliberate look
before it gets a compiler.
"""

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NEW = [os.path.join(ROOT, "kernels", f"{k}.metal") for k in ("qwen35_gdn", "qwen35_attn", "qwen35_score")]

# Constructs the Qwen3.5 kernels introduce, each checked against the Metal
# Shading Language specification when it was added.
REVIEWED = {
    "as_type": "MSL 2.19 reinterpret-cast function (as_type<T>(x))",
    "fabs": "MSL common math function",
    "log": "MSL math function (fast variant under the default fast math)",
    "mem_flags::mem_device": "threadgroup_barrier flag ordering device memory across the threadgroup",
    "precise::cos": "metal::precise math namespace",
    "precise::divide": "metal::precise math namespace",
    "precise::exp": "metal::precise math namespace",
    "precise::log": "metal::precise math namespace",
    "precise::pow": "metal::precise math namespace",
    "precise::sin": "metal::precise math namespace",
    "static_assert": "C++14 language feature; operands are program-scope `constant uint`s, which the "
                     "existing kernels already use as array bounds (constant expressions)",
    "uint3": "vector type for the 3-D [[threadgroup_position_in_grid]] builtin",
    "cast (device uint *)": "pointer cast within the device address space",
}

KEYWORDS = {"if", "for", "while", "return", "switch", "sizeof", "defined", "kernel", "void", "else"}
FEATURES = ["static_assert", "constexpr", "template", "#define", "uint3", "uint2", "ulong2", "float4",
            "bfloat", "bool"]


def strip(src):
    src = re.sub(r"//[^\n]*", "", src)
    return re.sub(r"/\*.*?\*/", "", src, flags=re.S)


def constructs(src):
    s = strip(src)
    found = set()
    found |= {m.group(1) for m in re.finditer(r"\b([A-Za-z_]\w*(?:::\w+)*)\s*\(", s)}
    found |= {m.group(1) for m in re.finditer(r"\b([A-Za-z_]\w*)\s*<[\w\s,:]*>\s*\(", s)}
    found |= {m.group(0) for m in re.finditer(r"\b\w+::\w+\b", s)}
    found |= {"[[" + re.sub(r"\(\d+\)", "(n)", m.group(1)) + "]]" for m in re.finditer(r"\[\[([^\]]+)\]\]", s)}
    found |= {"cast " + re.sub(r"\s+", " ", m.group(0))
              for m in re.finditer(r"\((?:device|threadgroup|constant)\s+(?:const\s+)?\w+\s*\*\)", s)}
    found |= {f for f in FEATURES if re.search(r"(?<![\w#])" + re.escape(f) + r"\b", s)}
    return found - KEYWORDS


def defined_here(src):
    """Helpers, kernels and macros the new sources define themselves."""
    s = strip(src)
    names = {m.group(1) for m in re.finditer(r"^\s*(?:inline\s+)?[\w<>\s]*?\b(\w+)\s*\([^;{]*\)\s*\{", s, re.M)}
    names |= {m.group(1) for m in re.finditer(r"^#define\s+(\w+)", s, re.M)}
    names |= {m.group(1) for m in re.finditer(r"^(\w+)\(\w+,", s, re.M)}  # macro instantiations
    names |= {"NAME"}
    return names


def main():
    old_files = [f for f in glob.glob(os.path.join(ROOT, "kernels", "**", "*.metal"), recursive=True)
                 + glob.glob(os.path.join(ROOT, "kernels", "*.h")) if f not in NEW]
    known = set()
    for f in old_files:
        known |= constructs(open(f).read())
    own = set()
    for f in NEW:
        own |= defined_here(open(f).read())
    novel = {}
    for f in NEW:
        for c in constructs(open(f).read()):
            if c not in known and c not in own:
                novel.setdefault(c, os.path.basename(f))
    unreviewed = sorted(c for c in novel if c not in REVIEWED)
    stale = sorted(c for c in REVIEWED if c not in novel)
    for c in sorted(novel):
        mark = "ok  " if c in REVIEWED else "FAIL"
        print(f"  [{mark}] {c:<24} {novel[c]:<20} {REVIEWED.get(c, 'not reviewed: check the MSL spec, then list it')}")
    if stale:
        print("  note: reviewed entries no longer needed (now used elsewhere, or removed):", ", ".join(stale))
    if unreviewed:
        print(f"\n{len(unreviewed)} construct(s) new to tessl's kernels and not reviewed")
        sys.exit(1)
    print(f"\n{len(novel)} construct(s) new to tessl's kernels, all reviewed")


if __name__ == "__main__":
    main()
