#!/bin/sh
# Re-runnable evidence check: compile each int4b_format probe variant with
# tessl's build.rs flags (-std=metal4.0 -O2 ... -mmacosx-version-min=26.0).
# CPU-only; nothing is executed. Output lands under target/w13-int4probe/.
# Run from the repo root. Uses `xcrun metal` without `-sdk` (see build.rs header).
SDK=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
[ -n "$SDKROOT" ] && SDK="$SDKROOT"
mkdir -p target/w13-int4probe
for v in HALF_I4_F32 HALF_U4_HALF BF_I4_F32 BF_I4_BF I8_I4_I32 U8_U4_I32 LHS_I4 BOTH_I4 HALF_U4_I32 TENSOR_ONLY_TYPED_PTR STATIC_EXT_16 MLX_GROUP_LOOP; do
  if xcrun metal -std=metal4.0 -O2 -fmetal-math-mode=fast -isysroot "$SDK" \
      -mmacosx-version-min=26.0 -DV_$v=1 -c tools/probes/int4b/probe.metal \
      -o target/w13-int4probe/$v.air 2> target/w13-int4probe/$v.err; then
    echo "$v: COMPILES"
  else
    echo "$v: FAILS"; grep -m3 'error' target/w13-int4probe/$v.err
  fi
done
