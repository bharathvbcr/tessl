#!/usr/bin/env bash
# Fail closed before tag / cargo publish / GitHub Release.
# Catches version triad drift, dirty trees, and oversized/stale package contents.
# Portable: uses rustc + grep/sed only (no ripgrep, no Python) so GitHub runners work.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

crate="$(sed -n 's/^name = "\(.*\)"/\1/p' Cargo.toml | head -1)"
version="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
if [[ -z "$crate" || -z "$version" ]]; then
  echo "error: could not parse name/version from Cargo.toml" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: working tree is dirty; commit or stash before release" >&2
  git status --porcelain >&2
  exit 1
fi

# The README / CHANGELOG / Cargo.lock text checks are src/bin/release_check.rs,
# built here with a bare rustc: it is std-only, so it needs neither the crate's
# dependencies nor the Metal toolchain, and it runs before either is built.
RC_DIR="$(mktemp -d)"
trap 'rm -rf "$RC_DIR"' EXIT
rustc --edition 2021 -O -o "$RC_DIR/release_check" src/bin/release_check.rs

"$RC_DIR/release_check" readme README.md "$version"
"$RC_DIR/release_check" changelog CHANGELOG.md "$version"

if ! grep -q "^\[${version}\]:" CHANGELOG.md; then
  echo "error: CHANGELOG.md missing footer link [$version]:" >&2
  exit 1
fi

if [[ -f Cargo.lock ]]; then
  "$RC_DIR/release_check" lock Cargo.lock "$crate" "$version"
fi

echo "==> cargo package --locked --list ($crate $version)"
cargo package --locked --list >"/tmp/${crate}-package.list"
if grep -E '\.devmap/|@2048\.png|build_logo\.py|assets/concepts/' "/tmp/${crate}-package.list"; then
  echo "error: package list contains excluded/stale paths (listed above)" >&2
  exit 1
fi

echo "==> cargo package --locked (verify build)"
cargo package --locked >/tmp/"${crate}-package.log" 2>&1

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "==> DOCS_RS=1 cargo check (docs.rs AOT skip)"
  DOCS_RS=1 cargo check --locked >/tmp/"${crate}-docsrs.log" 2>&1 || {
    echo "error: DOCS_RS=1 cargo check failed; see /tmp/${crate}-docsrs.log" >&2
    exit 1
  }
fi

echo "release-ready: $crate $version"
