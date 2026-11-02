#!/usr/bin/env bash
# Preflight for the sibling checkout this crate path-depends on.
#
# tessl does not stand alone. `Cargo.toml` declares mandatory, non-optional path
# dependencies that leave the repository:
#
#   ojas-core = { version = "0.1.0", path = "../ojas/ojas-core" }
#   ojas-io   = { version = "0.1.0", path = "../ojas/ojas-io" }
#
# Cargo resolves both before it builds, lints or tests anything, so a missing or
# misplaced ojas fails ahead of the first gate with an error about a manifest
# rather than about layout. This script says what is actually wrong, first, in
# one place.
#
# It exists because of this repository's own P0: those dependencies were
# mandatory for weeks while `.github/workflows/ci.yml` had seven
# `actions/checkout` steps and none checked out ojas. Every cargo job failed
# before its own first step, and nothing named the cause — the error was about
# a package, so it read like a dependency problem. ojas carries the mirror of
# this file (`ojas/scripts/check_siblings.sh`) for the same reason in the other
# direction; as of 2026-11-01 that file is unpushed, on ojas's local main only.
#
# Invoke: ./scripts/check_siblings.sh [--network]
#   --network  let the resolution checks reach the registry. Required where the
#              cargo registry cache is cold, which is every fresh CI runner:
#              measured, a cold CARGO_HOME fails `--offline` with "no matching
#              package named `block2` found / location searched: crates.io
#              index", which is a registry failure wearing a resolution
#              failure's clothes. Locally the default `--offline` is right: a
#              network outage must not be reported as a broken layout.
#
# Exits non-zero on the first hard failure. Nothing here is skippable: a check
# that cannot run must not report what a check that ran and passed reports.
# tests/check_siblings_script.rs holds every failure path below to failing.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OFFLINE=--offline
case "${1:-}" in
  --network) OFFLINE= ;;
  "") ;;
  *) echo "usage: $0 [--network]" >&2; exit 2 ;;
esac

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

# Lockfiles whose workspace this script resolves, and why only these. Every
# tracked lockfile is accounted for by step 5, which fails if the set changes.
#
#   Cargo.lock            the crate itself; `cargo check --locked` in the msrv
#                         job and `cargo publish --locked` in release.yml.
#   go/engine/Cargo.lock  its own `[workspace]`, and the `go` and `gpu` jobs
#                         build it with --locked (go/Makefile: `cargo
#                         build/test --release --locked --manifest-path
#                         $(ENGINE)/Cargo.toml`). Resolving only the root
#                         lockfile would leave that job's authority unchecked —
#                         and that lockfile really was stale, named neither
#                         ojas-core nor ojas-io, until commit d289a9c.
RESOLVE_LOCKS="Cargo.lock go/engine/Cargo.lock"
# Tracked lockfiles deliberately NOT resolved here, with the reason:
#   numcheck/Cargo.lock             a path dev-dependency of the root crate; the
#                                   root graph above resolves it, and nothing
#                                   builds it through its own lockfile.
#   bench/tok_hf_native/Cargo.lock  no CI job, script or Makefile builds it
#                                   (verified: `rg tok_hf_native` over
#                                   .github/, scripts/, go/Makefile matches
#                                   nothing), and its `tokenizers` dependency is
#                                   absent from a cache that has never built it,
#                                   so resolving it here would fail --offline
#                                   for a crate nothing gates on.
EXPECTED_LOCKS="Cargo.lock bench/tok_hf_native/Cargo.lock go/engine/Cargo.lock numcheck/Cargo.lock"

# ---------------------------------------------------------------------------
# 1. What kind of checkout is this?
#
# Reported, NOT refused, and that is a deliberate difference from ojas's copy.
# ojas cannot build from a worktree at any path: `ojas-metal -> ../../tessl`
# resolves to nothing under `.claude/worktrees/<lane>/`, and at a sibling depth
# where it does resolve, tessl pulls the primary checkout's `ojas-core` while
# the worktree has an `ojas-core` member of its own, so cargo refuses the
# lockfile collision.
#
# tessl has neither half of that. Measured on 2026-11-01 with
# `cargo metadata --locked --offline` from throwaway worktrees of this commit:
#   * a worktree whose `../ojas` does not exist  -> exit 101, "no matching
#     package named `ojas-core` found; location searched: <parent>/ojas/ojas-core"
#   * a worktree placed beside a real ojas checkout -> exit 0
# There is no collision to hit, because tessl's workspace has no `ojas-core` of
# its own. So a tessl worktree is fine wherever `../ojas` resolves, and step 2
# is what decides that. Refusing every worktree here would refuse a layout that
# demonstrably works — including CI's own, which is a fresh checkout at a nested
# path.
# ---------------------------------------------------------------------------
echo "==> checkout kind"
WORKTREE=no
if [ -f .git ]; then
  WORKTREE=yes
  echo "linked git worktree: $ROOT"
  echo "    (fine for tessl as long as ../ojas resolves from here; step 2 checks it)"
else
  echo "primary checkout: $ROOT"
fi

# ---------------------------------------------------------------------------
# 2. Every path dependency that leaves this repository resolves to a real crate.
#
# Reported in full rather than stopping at the first, because a relocated
# sibling usually breaks several at once and the set is the diagnosis.
#
# The manifest list comes from `git ls-files`, not a glob: tessl's manifests sit
# at several depths and a glob that silently stops covering one is the failure
# this file exists to catch, one level up.
#
# Two counters, and the guard is on the second. `inside` counts path
# dependencies that stay in this repository — `go/engine -> ../..` is one, and
# it resolves whatever happens to ojas — so guarding on a total would be a
# coverage check that can never fire: the parser could go blind to BOTH ojas
# dependencies and still report success for having looked at `../..`. Only a
# dependency whose resolved target lies outside $ROOT counts as a sibling, and
# finding none of those is a hard failure.
#
# Both TOML string forms are matched. `path = '../ojas/ojas-core'` is valid TOML
# and a double-quote-only parser goes blind to it, which is exactly the hole
# this comment is about.
# ---------------------------------------------------------------------------
echo "==> path dependencies resolve"
missing=0
inside=0
external=0
siblings=""

manifests="$(git ls-files '*Cargo.toml' 2>/dev/null || true)"
[ -n "$manifests" ] || fail "git ls-files found no Cargo.toml at all. Either this is not a
       checkout of tessl, or the manifest list this script walks has stopped
       matching. Fix it, do not ignore this."

while IFS= read -r manifest; do
  [ -n "$manifest" ] || continue
  [ -f "$manifest" ] || continue
  dir="$(dirname "$manifest")"
  while IFS= read -r raw; do
    [ -n "$raw" ] || continue
    case "$raw" in *..*) ;; *) continue ;; esac
    target="$(cd "$dir" 2>/dev/null && cd "$raw" 2>/dev/null && pwd || true)"
    if [ -z "$target" ] || [ ! -f "$target/Cargo.toml" ]; then
      echo "  MISSING   $manifest -> $raw" >&2
      missing=$((missing + 1))
      continue
    fi
    case "$target" in
      "$ROOT"/*|"$ROOT")
        inside=$((inside + 1))
        echo "  ok        $manifest -> $raw (inside this repository)"
        ;;
      *)
        external=$((external + 1))
        echo "  ok        $manifest -> $raw (sibling)"
        repo="$(git -C "$target" rev-parse --show-toplevel 2>/dev/null || true)"
        [ -n "$repo" ] && siblings="$siblings$repo
"
        ;;
    esac
  done <<EOF_PATHS
$(grep -oE "path[[:space:]]*=[[:space:]]*(\"[^\"]+\"|'[^']+')" "$manifest" 2>/dev/null | sed -E "s/.*[\"']([^\"']+)[\"'].*/\1/" || true)
EOF_PATHS
done <<EOF_MANIFESTS
$manifests
EOF_MANIFESTS

# Order matters: a missing sibling is the layout diagnosis and must be reported
# as that, not as a blind parser.
if [ "$missing" -ne 0 ]; then
  echo >&2
  echo "The sibling layout this crate needs:" >&2
  echo "    <parent>/tessl   <- this checkout, $ROOT" >&2
  echo "    <parent>/ojas    <- github.com/bharathvbcr/ojas" >&2
  echo >&2
  if [ "$WORKTREE" = yes ]; then
    echo "This is a linked worktree, which is the usual reason: a worktree under" >&2
    echo ".claude/worktrees/<lane>/ sits at a depth where ../ojas is nothing." >&2
    echo "A worktree placed beside a real ojas checkout resolves and builds" >&2
    echo "fine; the depth is the problem, not the worktree." >&2
    echo >&2
  fi
  echo "Cargo's own error for this names a package and a searched path, which" >&2
  echo "reads like a dependency or MSRV problem on every toolchain. It is not:" >&2
  echo "it is this layout. Clone or move ojas beside this checkout." >&2
  fail "$missing path dependencies do not resolve (see above)"
fi

if [ "$external" -eq 0 ]; then
  fail "parsed $inside path dependencies, and not one of them leaves this
       repository. Cargo.toml declares ojas-core and ojas-io at ../ojas/, so
       this parser has gone blind to them and is reporting success for having
       looked only at in-repository paths. Fix the parser, do not ignore this."
fi
echo "$((inside + external)) path dependencies resolve, $external of them leaving this repository"

# ---------------------------------------------------------------------------
# 3. What state is that sibling in?
#
# Not a hard failure: a dirty or branched sibling is normal while work is in
# flight. It is reported loudly because anything read from such a checkout — a
# version, a line number, a manifest field — is not what that repository
# publishes. That mistake was made about THIS repository on 2026-11-01: a
# session read tessl's `rust-version` and the ojas dependency's line numbers out
# of a working tree sitting on an unpushed branch and reported them as the
# repository's state, when origin/main said something else.
# ---------------------------------------------------------------------------
echo "==> sibling checkout state"
printf '%s' "$siblings" | sort -u | while IFS= read -r repo; do
  [ -n "$repo" ] || continue
  branch="$(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  head="$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo '?')"
  dirty="$(git -C "$repo" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  ahead="$(git -C "$repo" rev-list --count origin/main..HEAD 2>/dev/null || echo '?')"
  echo "  $repo"
  echo "    branch $branch at $head; $dirty uncommitted path(s); $ahead commit(s) not on origin/main"
  if [ "$branch" != "main" ] || [ "$dirty" != "0" ] || { [ "$ahead" != "0" ] && [ "$ahead" != "?" ]; }; then
    echo "    NOTE: this checkout does not match what it publishes. Cite origin/main," >&2
    echo "          or a commit, for any claim about this repository." >&2
  fi
done

# ---------------------------------------------------------------------------
# 4. The authoritative check: cargo itself resolves each lockfile CI builds
#    with --locked.
#
# Steps 1-3 can all pass while resolution still fails — a version mismatch, a
# lockfile that disagrees with a manifest, two packages of one name.
# `cargo metadata` answers that without compiling, and is not one of the
# subcommands the agent build-slot gate serialises (verified against its list:
# build, check, clippy, test, bench, doc, rustc, rustdoc, fix, install,
# nextest, llvm-cov, mutants, fuzz, miri), so a preflight cannot end up waiting
# for a slot.
#
# Per lockfile, not once at the root: `go/engine` declares its own
# `[workspace]`, so the root resolve says nothing about the lockfile the `go`
# and `gpu` jobs actually build. See RESOLVE_LOCKS above for the set and the
# reasons, and step 5 for what keeps that set honest.
#
# --locked because the promise is about the committed lockfiles, not about
# whatever a fresh resolve happens to pick today.
# ---------------------------------------------------------------------------
echo "==> cargo resolves each gating lockfile (--locked ${OFFLINE:-online})"
for lock in $RESOLVE_LOCKS; do
  [ -f "$lock" ] || fail "$lock is in this script's resolve set but not on disk. Either it was
       removed — in which case update RESOLVE_LOCKS and say why — or this
       checkout is incomplete."
  manifest="$(dirname "$lock")/Cargo.toml"
  if ! err="$(cargo metadata --locked $OFFLINE --format-version 1 --manifest-path "$manifest" 2>&1 >/dev/null)"; then
    echo "$err" >&2
    # A cold registry cache fails --offline for a reason that has nothing to do
    # with this repository's layout, and must not be reported as if it had.
    if [ -n "$OFFLINE" ] && printf '%s' "$err" | grep -q 'location searched: crates.io index'; then
      fail "the cargo registry cache has no copy of a dependency $lock names, so
       --offline cannot resolve the graph. This is NOT a layout or lockfile
       problem: steps 1-3 passed. Re-run as
           $0 --network
       which is what CI does, every runner starting with a cold cache."
    fi
    fail "cargo cannot resolve $manifest against $lock. The layout checks above
       passed, so this is a manifest or lockfile problem, not a missing
       sibling."
  fi
  echo "  resolved  $lock"
done

# ---------------------------------------------------------------------------
# 5. Every tracked lockfile is accounted for.
#
# Step 4 resolves a named subset, because two of the four tracked lockfiles are
# not what anything builds with --locked and one of them cannot resolve offline
# at all. A named subset is exactly the hand-written list that goes stale, so
# this check fails when the set of tracked lockfiles changes: a new workspace
# then forces a decision about whether it gates CI, instead of quietly sitting
# outside the preflight the way go/engine's did.
# ---------------------------------------------------------------------------
echo "==> tracked lockfiles are accounted for"
found="$(git ls-files '*Cargo.lock' | sort | tr '\n' ' ')"
want="$(printf '%s' "$EXPECTED_LOCKS" | tr ' ' '\n' | sort | tr '\n' ' ')"
if [ "$found" != "$want" ]; then
  echo "  tracked:  $found" >&2
  echo "  expected: $want" >&2
  fail "the set of tracked lockfiles has changed. Decide for each new one whether
       CI builds it with --locked: if it does, add it to RESOLVE_LOCKS so this
       preflight covers it; either way add it to EXPECTED_LOCKS with the
       reason. Do not widen EXPECTED_LOCKS without reading that question."
fi
echo "  4 tracked, 2 resolved above, 2 accounted for with reasons"

echo "==> siblings ok"
