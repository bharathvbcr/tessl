#!/usr/bin/env bash
# Fail closed before tag / cargo publish / GitHub Release.
# Catches version triad drift, dirty trees, and oversized/stale package contents.
# Portable: uses rustc + grep/sed only (no ripgrep, no Python) so GitHub runners work.
set -euo pipefail

# TMPDIR is read before the cd below, so a relative one means the caller's cwd.
# A leading '-' would be parsed by mktemp as a flag: refuse it with our own error.
tmp_base="${TMPDIR:-/tmp}"
case "$tmp_base" in
  -*)
    echo "error: TMPDIR '$tmp_base' starts with '-'; use an absolute path or ./-name" >&2
    exit 1
    ;;
  /*) ;;
  *) tmp_base="$PWD/$tmp_base" ;;
esac

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
#
# The traps go in BEFORE mktemp, and RUN_DIR is assigned straight from it (bash
# runs a signal trap that arrived during the command substitution only after the
# assignment), so a signal right after the dir is made still reaches cleanup.
# Nothing is removed unless run_dir_ok holds, in cleanup as well as at the
# start: `rm -rf -- ""` and `"$RUN_DIR"/*` can never become `/*`, and a path
# that is not a fresh directory directly under the resolved TMPDIR is never
# deleted.
#
# keep: the one log a failing step wants to leave behind (empty: remove all).
# INT/TERM/HUP clear it, so an aborted run leaves nothing. SIGKILL cannot be
# trapped and leaves the dir behind; that is accepted. A signal sent to the
# script's pid alone (not its process group, as Ctrl-C does) is deferred by
# bash until the foreground child exits; that is also accepted.
#
# A cleanup failure must not change the verdict: the gate already decided, and
# failing a ready release over a temp dir (or turning a failure into a
# different one) would mislead. So cleanup keeps the exit status it started
# with, warns loudly on stderr, and `set +e` stops errexit from altering it.
RUN_DIR=""
keep=""
run_dir_ok() {
  local tmp_real run_real
  [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] || return 1
  tmp_real="$(cd "$tmp_base" 2>/dev/null && pwd -P)" || return 1
  run_real="$(cd "$RUN_DIR" 2>/dev/null && pwd -P)" || return 1
  [[ -n "$tmp_real" && "$(dirname "$run_real")" == "$tmp_real" ]]
}
cleanup() {
  local rc=$?
  trap '' INT TERM HUP # a second signal must not abort the removal
  set +e
  if [[ -n "$RUN_DIR" ]]; then
    if ! run_dir_ok; then
      echo "warning: not removing '$RUN_DIR': not a directory directly under TMPDIR '$tmp_base'" >&2
    elif [[ -n "$keep" ]]; then
      local f
      for f in "$RUN_DIR"/*; do
        [[ "$f" == "$keep" ]] || rm -rf -- "$f" ||
          echo "warning: could not remove $f" >&2
      done
      echo "release-ready FAILED: kept $keep" >&2
    else
      rm -rf -- "$RUN_DIR" || echo "warning: could not remove run dir $RUN_DIR" >&2
    fi
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'keep=""; exit 130' INT
trap 'keep=""; exit 129' HUP
trap 'keep=""; exit 143' TERM

RUN_DIR="$(mktemp -d "$tmp_base/${crate}-release-ready.XXXXXX")" || {
  echo "error: could not create a temp dir under $tmp_base" >&2
  exit 1
}
if ! run_dir_ok; then
  echo "error: mktemp gave '$RUN_DIR', not a directory directly under TMPDIR '$tmp_base'" >&2
  exit 1
fi
echo "==> run dir: $RUN_DIR (removed on success; a failing step's log is kept)"

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
# A package with no files is never release-ready (and an empty list would let
# the denylist grep below pass vacuously).
[[ -s "$RUN_DIR/package.list" ]] ||
  fail_log "$RUN_DIR/package.list" "cargo package --locked --list produced an empty package list"
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
