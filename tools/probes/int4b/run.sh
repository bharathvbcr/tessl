#!/bin/bash
# Re-runnable evidence check for the int4b_format TensorOps probe. CPU-only:
# every variant is compiled, nothing is executed. Exits non-zero if any variant
# deviates from the expected outcome below (a negative control must fail AND
# print its expected diagnostic).
#
# Flags mirror build.rs `metal_args` (build.rs:479-499) with `werror` on:
# -std=metal4.0 (the first of the two stds try_metal_compile tries), -O2,
# -fmetal-math-mode=fast, -isysroot <sdk>, -mmacosx-version-min=26.0,
# -Wall -Werror. Uses `xcrun metal` without `-sdk` (build.rs header, lines 10-11).
# Override the SDK with SDKROOT=<path>. Works from any cwd; outputs go to
# <repo>/target/w13-int4probe/ (gitignored).
here=$(dirname "$0")
repo=$(cd "$here/../../.." && pwd)
src="$here/probe.metal"
out="$repo/target/w13-int4probe"
sdk="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
mkdir -p "$out"

fail=0
# check VARIANT OUTCOME [REQUIRED_STDERR_SUBSTRING]
check() {
  v=$1; want=$2; needle=$3
  if xcrun metal -std=metal4.0 -O2 -fmetal-math-mode=fast -isysroot "$sdk" \
      -mmacosx-version-min=26.0 -Wall -Werror -DV_$v=1 -c "$src" \
      -o "$out/$v.air" 2> "$out/$v.err"; then got=COMPILES; else got=FAILS; fi
  if [ "$got" != "$want" ]; then
    echo "MISMATCH $v: expected $want, got $got"; fail=1; return
  fi
  if [ -n "$needle" ] && ! grep -qF -- "$needle" "$out/$v.err"; then
    echo "MISMATCH $v: failed but stderr lacks '$needle'"; fail=1; return
  fi
  echo "ok $v: $got${needle:+ ($needle)}"
}

check HALF_I4_F32 COMPILES
check HALF_U4_HALF COMPILES
check BF_I4_F32 COMPILES
check BF_I4_BF COMPILES
check I8_I4_I32 COMPILES
check U8_U4_I32 COMPILES
check MLX_GROUP_LOOP COMPILES
check LHS_I4 FAILS "Unsupported type"
check BOTH_I4 FAILS "Unsupported type"
check HALF_U4_I32 FAILS "Unsupported type"
check TENSOR_ONLY_TYPED_PTR FAILS "no matching constructor for initialization of 'tensor<device metal::int4b_format"
check STATIC_EXT_16 FAILS "must be a multiple of 32"

[ "$fail" -eq 0 ] && echo "ALL VARIANTS AS EXPECTED" || echo "PROBE DEVIATED"
exit "$fail"
