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

# The crate name is used in a temp-dir template below: fail closed on anything
# that is not a plain package name (a path separator would escape the template).
if ! [[ "$crate" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "error: crate name '$crate' from Cargo.toml is not [A-Za-z0-9_-]+" >&2
  exit 1
fi

# One run dir holds the release_check binary and every log of this run, so two
# runs (other checkouts, other agents) never share a file. mktemp's template
# form is the one both BSD (the macos-26 runners) and GNU accept.
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/${crate}-release-ready.XXXXXX")" || {
  echo "error: could not create a temp dir under ${TMPDIR:-/tmp}" >&2
  exit 1
}
echo "==> run dir: $RUN_DIR (removed on success; a failing step's log is kept)"

# keep: the one log a failing step wants to leave behind (empty: remove all).
# INT/TERM/HUP clear it, so an aborted run leaves nothing. SIGKILL cannot be
# trapped and leaves the dir behind; that is accepted.
keep=""
cleanup() {
  if [[ -n "$keep" ]]; then
    local f
    for f in "$RUN_DIR"/*; do
      [[ "$f" == "$keep" ]] || rm -rf -- "$f"
    done
    echo "release-ready: kept $keep" >&2
  else
    rm -rf -- "$RUN_DIR"
  fi
}
trap cleanup EXIT
trap 'keep=""; exit 130' INT
trap 'keep=""; exit 143' TERM HUP

# fail_log PATH MSG: keep PATH, say where it is, show its tail, exit 1.
fail_log() {
  keep="$1"
  echo "error: $2; see $1" >&2
  if [[ -s "$1" ]]; then
    echo "--- last 40 lines of $1 ---" >&2
    tail -n 40 -- "$1" >&2
  fi
  exit 1
}

# The README / CHANGELOG / Cargo.lock text checks are src/bin/release_check.rs,
# built here with a bare rustc: it is std-only, so it needs neither the crate's
# dependencies nor the Metal toolchain, and it runs before either is built.
rustc --edition 2021 -O -o "$RUN_DIR/release_check" src/bin/release_check.rs

"$RUN_DIR/release_check" readme README.md "$version"
"$RUN_DIR/release_check" changelog CHANGELOG.md "$version"

if ! grep -q "^\[${version}\]:" CHANGELOG.md; then
  echo "error: CHANGELOG.md missing footer link [$version]:" >&2
  exit 1
fi

if [[ -f Cargo.lock ]]; then
  "$RUN_DIR/release_check" lock Cargo.lock "$crate" "$version"
fi

echo "==> cargo package --locked --list ($crate $version)"
cargo package --locked --list >"$RUN_DIR/package.list" 2>"$RUN_DIR/package-list.log" ||
  fail_log "$RUN_DIR/package-list.log" "cargo package --locked --list failed"
if [[ -s "$RUN_DIR/package-list.log" ]]; then
  cat "$RUN_DIR/package-list.log" >&2
fi
if grep -E '\.devmap/|@2048\.png|build_logo\.py|assets/concepts/' "$RUN_DIR/package.list"; then
  keep="$RUN_DIR/package.list"
  echo "error: package list contains excluded/stale paths (listed above); see $RUN_DIR/package.list" >&2
  exit 1
fi

echo "==> cargo package --locked (verify build)"
cargo package --locked >"$RUN_DIR/package.log" 2>&1 ||
  fail_log "$RUN_DIR/package.log" "cargo package --locked failed"

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "==> DOCS_RS=1 cargo check (docs.rs AOT skip)"
  DOCS_RS=1 cargo check --locked >"$RUN_DIR/docsrs.log" 2>&1 ||
    fail_log "$RUN_DIR/docsrs.log" "DOCS_RS=1 cargo check failed"
fi

echo "release-ready: $crate $version"
