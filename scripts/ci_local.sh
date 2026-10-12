#!/usr/bin/env bash
# Local mirror of .github/workflows/ci.yml gates that can run on this machine.
# Named ci:local to match the monorepo convention; invoke via:
#   ./scripts/ci_local.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> fmt"
cargo fmt --all --check

echo "==> build (release, all targets)"
cargo build --release --all-targets

echo "==> clippy"
cargo clippy --release --all-targets -- -D warnings

echo "==> doc"
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps

echo "==> static tile audit"
python3 scripts/audit_gemm_tiles.py

echo "==> MSL dialect lint (CI: kernel-emulator job)"
python3 tools/msl_emu/dialect_lint.py

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

echo "==> parity harness, host-only (adversarial scorer, ladder, tile-audit and evidence contracts)"
# Needs numpy, which the oracle environment carries. A non-zero exit (any assertion failure,
# or a required section that did not run) stops the script; GPU sections are skipped visibly.
"$ORACLE_PY" -I bench/test_parity_harness.py --pure

echo "==> test (release, serialized)"
cargo test --release -- --test-threads=1

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
  echo "note: skipping check_release_ready.sh while tree is dirty"
else
  ./scripts/check_release_ready.sh
fi

echo "ci:local OK"
