#!/usr/bin/env bash
# Build the CPU harness for tessl's Qwen3.5 kernels (see README.md).
#
# The kernel sources are compiled unmodified except for one mechanical step:
# `[[attribute]]` annotations (buffer indices, builtin roles) are stripped,
# because C++ has no meaning for them. The harness supplies arguments in
# declaration order instead, which is also how the Rust host binds them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${MSL_EMU_OUT:-$HERE/build}"
mkdir -p "$OUT/gen"
for k in qwen35_gdn qwen35_attn qwen35_score; do
  sed -E 's/\[\[[^]]*\]\]//g' "$ROOT/kernels/$k.metal" > "$OUT/gen/$k.cpp"
done
"${CXX:-g++}" -std=c++20 -O2 -g -pthread -Wall -Wno-unused-variable -Wno-unused-parameter \
  -Wno-unknown-pragmas -Wno-sign-compare \
  -I "$HERE" -I "$ROOT/kernels" -I "$OUT/gen" \
  "$HERE/harness.cpp" -o "$OUT/harness"
echo "$OUT/harness"
