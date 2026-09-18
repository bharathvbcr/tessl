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

echo "==> test (release, serialized)"
cargo test --release -- --test-threads=1

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
