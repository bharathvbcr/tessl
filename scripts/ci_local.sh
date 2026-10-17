#!/usr/bin/env bash
# Local mirror of .github/workflows/ci.yml gates that can run on this machine.
# Named ci:local to match the monorepo convention; invoke via:
#   ./scripts/ci_local.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Every step that does not run is named here and in the last line, so a run with skips
# never ends with the same words as a run that checked everything.
SKIPS=()

echo "==> fmt"
cargo fmt --all --check

echo "==> build (release, all targets)"
cargo build --release --all-targets

echo "==> clippy"
cargo clippy --release --all-targets -- -D warnings

echo "==> clippy (quant-prep; mtl_tensor is compiled only with the feature)"
cargo clippy --release --all-targets --features quant-prep -- -D warnings

echo "==> doc"
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps

echo "==> device-free selection with the device hidden (CI: check job; includes the static tile audit)"
# The same invocation the hosted macOS job runs: scripts/hosted_tests.txt under a sandbox that
# denies IOKit, every filter guarded to have run tests. Any test in it that opens a device fails.
./scripts/hosted_tests.sh

echo "==> hosted manifest completeness (every test that passes with the device hidden is selected)"
./scripts/hosted_tests.sh --outside

echo "==> MSL dialect lint (CI: kernel-emulator job, which compiles it with a bare rustc as here)"
mkdir -p target
rustc --edition 2021 -O -o target/dialect_lint src/bin/dialect_lint.rs
target/dialect_lint

echo "==> kernel IR check (fast-math: no inf/NaN constants, no fcmp uno/ord, isfinite stays a bit test)"
# Exit 77 means no Metal toolchain: SKIPPED is printed and the run still fails, because the
# build step above already needed that toolchain and a skipped check must not read as passed.
./scripts/check_kernel_ir.sh

echo "==> emulator host contract (CI: kernel-emulator job; needs the torch oracle)"
# check_qwen35.py imports torch at module top, so no torch means no check. That
# must fail loudly: a skipped check must never read as a passed one.
ORACLE_PY=""
for cand in "${TESSL_ORACLE_PYTHON:-}" "$HOME/.venvs/rsijev-oracle/bin/python" python3; do
  [[ -n "$cand" ]] || continue
  if "$cand" -c 'import torch' >/dev/null 2>&1; then ORACLE_PY="$cand"; break; fi
done
if [[ -z "$ORACLE_PY" ]]; then
  echo "error: no python with torch found (tried \$TESSL_ORACLE_PYTHON, ~/.venvs/rsijev-oracle, python3)." >&2
  echo "       pip install -r tools/msl_emu/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu" >&2
  exit 1
fi
# -k host_contract selects both the parser self-test and the src/qwen35.rs binds/constants case.
"$ORACLE_PY" tools/msl_emu/check_qwen35.py -k host_contract

echo "==> emulator kernel cases (CI: kernel-emulator job; exact math, whole model, fast-math bounds)"
# The same three runs as the hosted job's non-sanitizer steps. A case added here but never run
# locally is how a wave-7 tolerance first failed on main, so they run by default. They take long
# on a loaded machine; TESSL_CI_SKIP_EMULATOR=1 skips them, and the last line says so.
if [[ "${TESSL_CI_SKIP_EMULATOR:-}" == 1 ]]; then
  echo "SKIPPED: emulator kernel cases (TESSL_CI_SKIP_EMULATOR=1)"
  SKIPS+=("emulator kernel cases")
else
  EMU_OUT="$ROOT/target/msl_emu"
  MSL_EMU_OUT="$EMU_OUT/exact" "$ORACLE_PY" tools/msl_emu/check_qwen35.py
  MSL_EMU_OUT="$EMU_OUT/exact" "$ORACLE_PY" tools/msl_emu/check_qwen35_model.py
  MSL_EMU_OUT="$EMU_OUT/fast" "$ORACLE_PY" tools/msl_emu/check_qwen35.py --fast-math
fi

echo "==> comparator lanes, host-only (masking rule and external-lane files; the scorer, ladder, census and tuner are cargo tests)"
# Needs numpy, which the oracle environment carries. A non-zero exit stops the script.
"$ORACLE_PY" -I bench/test_comparator_lanes.py

echo "==> torch binding host tests (python/tests/test_binding_host.py; no GPU, needs the torch oracle)"
# The same oracle python as above: a missing torch already stopped the script, so this
# cannot be skipped silently. The recording stand-in replaces libtessl; nothing is built.
"$ORACLE_PY" -m unittest python/tests/test_binding_host.py python/tests/test_embedgemma2_host.py python/tests/test_embedgemma2_mm_host.py -v

echo "==> torch binding GPU tests (python/tests: test_cross_entropy, test_gdn, test_qwen35, test_embedgemma2, test_embedgemma2_mm)"
# These load target/release/libtessl.dylib (built above) and run on MPS next to the Metal
# runtime, so they belong under the machine-wide heavy-job lock, which exports MAC_HEAVY_LABEL.
# Outside the lock they are not run, and that is printed rather than left to read as a pass.
if [[ -n "${MAC_HEAVY_LABEL:-}" ]]; then
  "$ORACLE_PY" -m unittest discover -s python/tests -v
else
  echo "SKIPPED: python/tests GPU tests (MAC_HEAVY_LABEL unset; rerun under tools/mac_heavy.sh to run them)"
  SKIPS+=("python/tests GPU tests")
fi

echo "==> test (release, serialized, every binary even after a failure)"
# --no-fail-fast runs every test binary so one red binary does not hide the rest; the
# script then still ends non-zero (below) if any of them failed.
TEST_FAILED=0
cargo test --release --no-fail-fast -- --test-threads=1 || TEST_FAILED=1

echo "==> test mtl_tensor (quant-prep; the default suite does not compile it)"
mtl_out=$(cargo test --release --features quant-prep --lib mtl_tensor:: -- --test-threads=1 2>&1) || { echo "$mtl_out"; exit 1; }
echo "$mtl_out"
grep -Eq 'test result: ok\. [1-9][0-9]* passed' <<<"$mtl_out" || { echo "mtl_tensor filter matched no tests"; exit 1; }

echo "==> examples"
cargo run --release --example gemm
cargo run --release --example nn_layer
cargo run --release --example epilogue_cost

echo "==> release-ready gate (requires clean tree + version triad)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "SKIPPED: check_release_ready.sh (tree is dirty)"
  SKIPS+=("release-ready gate")
else
  ./scripts/check_release_ready.sh
fi

if [[ "$TEST_FAILED" -ne 0 ]]; then
  echo "ci:local FAILED: one or more cargo test binaries failed (see the full run above)" >&2
  exit 1
fi
if [[ ${#SKIPS[@]} -gt 0 ]]; then
  joined=$(printf '%s; ' "${SKIPS[@]}")
  echo "ci:local OK WITH SKIPS (not run: ${joined%; })"
else
  echo "ci:local OK"
fi
