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
# flash_attn_rows is tessl's existing attention kernel, built here so the
# model-level check runs the attention the Qwen3.5 kernels actually feed.
for k in qwen35_gdn qwen35_attn qwen35_mlp qwen35_score flash_attn_rows; do
  sed -E 's/\[\[[^]]*\]\]//g' "$ROOT/kernels/$k.metal" > "$OUT/gen/$k.cpp"
done
# MSL_EMU_SANITIZE=address makes any out-of-bounds device-buffer or
# threadgroup-memory access fatal; =thread reports data races, which is what a
# missing barrier looks like.
# Expanded as ${SAN[@]+...}: macOS's bash 3.2 treats an empty array as unbound
# under `set -u`.
SAN=()
if [[ -n "${MSL_EMU_SANITIZE:-}" ]]; then
  SAN=(-fsanitize="$MSL_EMU_SANITIZE" -fno-omit-frame-pointer -O1)
fi
CXXFLAGS=(-std=c++20 -O2 -g -pthread -fno-strict-aliasing -Wall -Wno-unused-variable -Wno-unused-parameter
  ${SAN[@]+"${SAN[@]}"} -Wno-unknown-pragmas -Wno-sign-compare -I "$HERE")
"${CXX:-g++}" "${CXXFLAGS[@]}" -I "$ROOT/kernels" -I "$OUT/gen" "$HERE/harness.cpp" -o "$OUT/harness"
# Under TSan, a race report is only evidence if the barriers are visible to it
# and add nothing a real barrier lacks: barrier_probe.cpp holds them to both
# before any kernel is judged by them.
"${CXX:-g++}" "${CXXFLAGS[@]}" "$HERE/barrier_probe.cpp" -o "$OUT/barrier_probe"
if [[ "${MSL_EMU_SANITIZE:-}" == thread ]]; then
  for mode in tg tg_missing simd simd_missing drop; do
    want=0
    [[ "$mode" == *_missing ]] && want=66
    got=0
    # abort_on_error defaults to 1 on macOS, which would exit 134 instead.
    TSAN_OPTIONS="halt_on_error=0 abort_on_error=0 exitcode=66" "$OUT/barrier_probe" "$mode" 2> "$OUT/barrier_probe.$mode.log" || got=$?
    if [[ "$got" != "$want" ]]; then
      cat "$OUT/barrier_probe.$mode.log" >&2
      echo "msl_emu: barrier_probe $mode exited $got under TSan, expected $want" \
        "(66 = race reported): TSan's view of the emulator's barriers is wrong" >&2
      exit 1
    fi
  done
else
  "$OUT/barrier_probe" drop
fi
echo "$OUT/harness"
