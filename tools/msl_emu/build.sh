#!/usr/bin/env bash
# Build the CPU harness for tessl's Qwen3.5 kernels (see README.md).
#
# The kernel sources are compiled unmodified except for two mechanical steps:
# `[[attribute]]` annotations (buffer indices, builtin roles) are stripped,
# because C++ has no meaning for them; the harness supplies arguments in
# declaration order instead, which is also how the Rust host binds them. And a
# kernel-scope `threadgroup T name[N];` becomes a static registered with
# metal::emu::tg_static, so a group's threads share it (with `threadgroup`
# defined away it would be one private array per thread).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${MSL_EMU_OUT:-$HERE/build}"
mkdir -p "$OUT/gen"
# flash_attn_rows is tessl's existing attention kernel, built here so the
# model-level check runs the attention the Qwen3.5 kernels actually feed.
# qwen35_adamw and qwen35_bwd hold the training kernels.
for k in qwen35_gdn qwen35_attn qwen35_mlp qwen35_score flash_attn_rows qwen35_adamw qwen35_bwd; do
  sed -E -e 's/\[\[[^]]*\]\]//g' \
    -e 's/^([[:space:]]*)threadgroup ([A-Za-z_][A-Za-z0-9_]*) ([A-Za-z_][A-Za-z0-9_]*)\[([^];]*)\];/\1static \2 \3[\4]; ::metal::emu::tg_static(\3);/' \
    "$ROOT/kernels/$k.metal" > "$OUT/gen/$k.cpp"
done
# Any threadgroup array the rewrite did not take (2-D, several declarators on
# one line, an initializer) would silently be per-thread: refuse to build
# instead.
if grep -nE 'threadgroup +[A-Za-z_][A-Za-z0-9_]* +[A-Za-z_][A-Za-z0-9_]*\[' "$OUT"/gen/*.cpp >&2; then
  echo "msl_emu: kernel-scope threadgroup arrays above are not rewritten to shared statics" >&2
  exit 1
fi
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
  for mode in tg tg_missing simd simd_missing drop tg_static; do
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
  "$OUT/barrier_probe" tg_static
fi
echo "$OUT/harness"
