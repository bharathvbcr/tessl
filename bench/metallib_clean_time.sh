#!/usr/bin/env bash
# Wall time of one clean metallib build, isolated from rustc: drives a
# compiled build script directly (the way tests/build_artifact_contract.rs
# does) with an empty OUT_DIR, so no .air from an earlier run can be reused.
#
#   bench/metallib_clean_time.sh BUILD_SCRIPT_BIN MANIFEST_DIR
#
# Prints the METRIC lines bench/paired_bins.sh consumes. Build the script with
#   rustc --edition 2021 -O MANIFEST_DIR/build.rs -o BUILD_SCRIPT_BIN
# then pair arms by wrapping this script per arm (paired_bins.sh hands every
# binary the same ARGS). Pairing a build.rs with a different tree's kernels
# (BUILD_SCRIPT_BIN from one commit, MANIFEST_DIR from another) is the control
# that separates the build script's change from the kernel sources' change.
#
# macOS only: it samples `sysctl vm.loadavg`, times with perl Time::HiRes and
# needs the Metal toolchain; elsewhere it exits 2 before doing anything.
set -euo pipefail

if [[ $(uname -s) != Darwin ]]; then
  echo "$0 needs macOS (sysctl vm.loadavg, Metal toolchain)" >&2
  exit 2
fi
if [[ $# -ne 2 ]]; then
  echo "usage: $0 BUILD_SCRIPT_BIN MANIFEST_DIR" >&2
  exit 2
fi
bs=$1
manifest=$2
[[ -x $bs ]] || { echo "$bs is missing or not executable" >&2; exit 2; }
[[ -d $manifest/kernels ]] || { echo "$manifest has no kernels/" >&2; exit 2; }

out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
log=$out/stdout
errlog=$out/stderr
load_start=$(sysctl -n vm.loadavg | awk '{print $2}')

mkdir "$out/o"
start=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')
env -u TESSL_GEMM_TUNE -u METAL_NATIVE_GEMM_TUNE -u TESSL_METAL_WERROR \
  -u TESSL_SKIP_AOT -u METAL_RUNTIME_SKIP_AOT -u TESSL_PREBUILT_METALLIB \
  -u DOCS_RS -u DEVELOPER_DIR \
  CARGO_MANIFEST_DIR="$manifest" OUT_DIR="$out/o" \
  "$bs" >"$log" 2>"$errlog" || { echo "build script failed:" >&2; cat "$errlog" >&2; exit 1; }
end=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')

metallib=$(sed -n 's/^cargo:rustc-env=TESSL_METALLIB=//p' "$log" | tail -1)
[[ -n $metallib && -f $metallib ]] || { echo "no metallib at '$metallib'" >&2; exit 1; }
# The artifact must come from this run's scratch OUT_DIR, not a stale or
# prebuilt path the script happened to print.
case $metallib in
  "$out"/o/*) ;;
  *) echo "metallib '$metallib' is not under the scratch OUT_DIR '$out/o'" >&2; exit 1 ;;
esac
# One .air per top-level kernels/*.metal (the build script does not compile
# subdirectories), so the expected count follows the sources.
# Regular files and symlinks (build.rs follows them); a directory named x.metal is not a source.
want=$(find "$manifest/kernels" -maxdepth 1 -name '*.metal' \( -type f -o -type l \) | wc -l | tr -d ' ')
((want > 0)) || { echo "$manifest/kernels has no top-level .metal sources" >&2; exit 1; }
airs=$(find "$out/o" -name '*.air' | wc -l | tr -d ' ')
[[ $airs -eq $want ]] || { echo "expected $want .air, found $airs" >&2; exit 1; }
# The cache-key files exist only in the parallel build script; when present
# every one must be fresh (this OUT_DIR started empty, so none can be reused).
keys=$(find "$out/o" -name '*.air.key' | wc -l | tr -d ' ')
[[ $keys -eq 0 || $keys -eq $want ]] || { echo "expected 0 or $want .air.key, found $keys" >&2; exit 1; }
fallbacks=$(grep -c 'falling back to -std=metal3.2' "$log" || true)

echo "METRIC metallib_clean_s $(awk -v a="$start" -v b="$end" 'BEGIN { printf "%.3f", b - a }')"
echo "METRIC load1_start $load_start"
echo "METRIC metal32_fallbacks $fallbacks"
echo "METRIC air_keys $keys"
