#!/usr/bin/env bash
# Compile every kernels/*.metal to LLVM IR with build.rs's flags and check what
# fast math (-fmetal-math-mode=fast) actually does to the float classification
# code. Standalone; scripts/ci_local.sh runs it next to the MSL dialect lint.
#
#   scripts/check_kernel_ir.sh [KERNELS_DIR]     (default: <repo>/kernels)
#
# Fails (exit 1) when any kernel's IR holds
#   * an inf / NaN float constant, found by parsing the exponent rather than
#     matching fixed spellings: any `0x` + 16 hex digits (f32/f64) whose 11
#     exponent bits are all ones (0x7FF0000000000000, 0x7FF4... signalling NaN,
#     0x7FF0000000000001, 0xFFF8...), any half `0xH....` with 5 exponent bits
#     all ones (0xH7C00, 0xHFC00) and any bfloat `0xR....` with 8 (0xR7F80).
#     A compare against INFINITY / NAN / HUGE_VAL / MAXFLOAT is poison under
#     `fcmp fast`.
#   * an unordered/ordered compare (`fcmp uno` / `fcmp ord`: what a self
#     compare `v != v` or a std::isnan lowers to when it is not folded), or
#   * a source that calls isfinite / isnan / isinf (headers included) while its
#     IR has no `icmp ... 2139095040` (0x7F800000): Metal defines those three
#     as integer tests on the bit pattern, which fast math cannot fold. This is
#     a per-kernel-file lower bound, not a per-call-site proof: LLVM may merge
#     two identical tests, so the check says "the bit test survived", and the
#     inf/NaN-constant and fcmp checks above say "nothing replaced it".
#   * a compile that exited 0 but left no IR (a missing or empty .ll is a
#     failure, never a pass).
#
# Sources: what build.rs compiles. kernels/*.metal always; kernels/tune/*.metal
# only when TESSL_GEMM_TUNE or METAL_NATIVE_GEMM_TUNE is set, as build.rs does.
# matmul_tensorops.metal (and the tune rig) compile under metal4.0 only;
# every other source falls back to metal3.2, exactly as build.rs does.
#
# Exit status: 0 clean, 1 violation or compile failure, 77 SKIPPED (no
# `xcrun metal`). 77 is deliberately not success; ci_local.sh lets it fail the
# run because the build step before it already needs the same toolchain.
#
# The flag list below mirrors `metal_args` in build.rs. tests/kernel_fast_math.rs
# (`check_kernel_ir_uses_build_rs_flags`) fails when the two drift apart.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KERNELS="${1:-$ROOT/kernels}"

if ! METAL="$(xcrun -f metal 2>/dev/null)" || [[ -z "$METAL" ]]; then
  echo "SKIPPED: check_kernel_ir.sh needs 'xcrun metal' (Xcode Metal Toolchain); no kernel IR was checked" >&2
  exit 77
fi
SDK="$(xcrun --sdk macosx --show-sdk-path)"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/kernel_ir.XXXXXX")"
trap 'rm -rf "$OUT"' EXIT

# build.rs `metal_args`: -std, -O2, fast math, sysroot, deployment target, and
# -ffp-contract=off for the sources in `disable_fp_contract`.
indent() { awk '{ print "    " $0 }'; }

# inf_nan_constants <ll>: print each float constant whose exponent is all ones.
inf_nan_constants() {
  local ll="$1" tok d
  while IFS= read -r tok; do
    d="${tok:2}"
    case "$tok" in
      0x[0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f])
        (( (16#${d:0:3} & 16#7FF) == 16#7FF )) && echo "$tok" ;;
      0xH*) (( ((16#${d:1} >> 10) & 16#1F) == 16#1F )) && echo "$tok" ;;
      0xR*) (( ((16#${d:1} >> 7) & 16#FF) == 16#FF )) && echo "$tok" ;;
    esac
  done < <(grep -oE '0x(H|R)?[0-9A-Fa-f]+' "$ll" | sort -u)
}

compile() { # compile <src> <std> <out.ll>
  local src="$1" std="$2" ll="$3" extra=()
  [[ "$(basename "$src")" == "qwen35_adamw.metal" ]] && extra=(-ffp-contract=off)
  "$METAL" "-std=$std" -O2 -fmetal-math-mode=fast -isysroot "$SDK" \
    -mmacosx-version-min=26.0 ${extra[@]+"${extra[@]}"} \
    -S -emit-llvm "$src" -o "$ll" 2>"$ll.err"
}

check_one() { # check_one <src>; prints violations, exit 1 if any
  local src="$1" name ll rc=0 std
  name="$(basename "$src" .metal)"
  ll="$OUT/$name.ll"
  # build.rs falls back to metal3.2 for non-TensorOps sources; mirror it.
  local stds=(metal4.0 metal3.2)
  # build.rs: the TensorOps sources hard-fail, no metal3.2 fallback.
  [[ "$name" == matmul_tensorops || "$src" == */tune/* ]] && stds=(metal4.0)
  for std in "${stds[@]}"; do
    rm -f "$ll"
    compile "$src" "$std" "$ll" && break
    std=""
  done
  if [[ -z "$std" ]]; then
    echo "FAIL $name: does not compile under metal4.0 or metal3.2:" >&2
    indent <"$ll.err" >&2
    return 1
  fi
  if [[ ! -s "$ll" ]]; then
    echo "FAIL $name: metal exited 0 but wrote no IR ($ll missing or empty); nothing was checked" >&2
    return 1
  fi
  local hits
  hits="$(inf_nan_constants "$ll" | head -5)"
  if [[ -n "$hits" ]]; then
    echo "FAIL $name: inf/NaN constant in IR (fast math treats a compare against it as poison):" >&2
    while IFS= read -r tok; do
      grep -m1 -F -- "$tok" "$ll" | indent >&2
    done <<<"$hits"
    rc=1
  fi
  hits="$(grep -nE 'fcmp( [a-z]+)* (uno|ord) ' "$ll" | head -5)"
  if [[ -n "$hits" ]]; then
    echo "FAIL $name: unordered/ordered fcmp in IR (self-compare or NaN test fast math may fold):" >&2
    indent <<<"$hits" >&2
    rc=1
  fi
  # isfinite / isnan / isinf in the source or a header it includes (one level
  # of local `#include "x.h"`, which is all the kernels use).
  local files=("$src") inc
  while IFS= read -r inc; do
    [[ -f "$(dirname "$src")/$inc" ]] && files+=("$(dirname "$src")/$inc")
  done < <(sed -n 's/^[[:space:]]*#include[[:space:]]*"\([^"]*\)".*/\1/p' "$src")
  local sites
  sites="$(cat "${files[@]}" | sed -e 's://.*$::' | grep -cE '\bis(finite|nan|inf)[[:space:]]*\(' || true)"
  if [[ "${sites:-0}" -gt 0 ]]; then
    if ! grep -qE 'icmp [a-z]+ i32 %[A-Za-z0-9_.]+, 2139095040' "$ll"; then
      echo "FAIL $name: $sites isfinite/isnan/isinf call(s) in source but no 0x7F800000 bit test in IR" >&2
      rc=1
    fi
  fi
  return $rc
}

shopt -s nullglob
SOURCES=("$KERNELS"/*.metal)
if [[ -n "${TESSL_GEMM_TUNE:-}${METAL_NATIVE_GEMM_TUNE:-}" ]]; then
  SOURCES+=("$KERNELS"/tune/*.metal)
fi
if [[ ${#SOURCES[@]} -eq 0 ]]; then
  echo "error: no .metal sources in $KERNELS" >&2
  exit 1
fi

JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
pids=()
failed=0
for src in "${SOURCES[@]}"; do
  while [[ ${#pids[@]} -ge $JOBS ]]; do
    wait "${pids[0]}" || failed=1
    pids=("${pids[@]:1}")
  done
  check_one "$src" &
  pids+=("$!")
done
for pid in ${pids[@]+"${pids[@]}"}; do wait "$pid" || failed=1; done

if [[ $failed -ne 0 ]]; then
  echo "check_kernel_ir: FAILED" >&2
  exit 1
fi
echo "check_kernel_ir: OK (${#SOURCES[@]} kernels compiled to IR with build.rs flags; no inf/NaN constants, no fcmp uno/ord, isfinite sites are bit tests)"
