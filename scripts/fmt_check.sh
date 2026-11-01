#!/usr/bin/env bash
# rustfmt check over tessl's own packages only.
#
# `cargo fmt --all` formats a package's local path dependencies as well, and
# tessl has three: `numcheck` (dev) and ojas's `ojas-core` and `ojas-io`. So
# `--all` would hold this repository's fmt gate to a sibling repository's
# formatting — and its `cargo metadata` call loads the whole ojas workspace,
# which path-depends on gusset, so without a third checkout it does not even
# run ("failed to load manifest for workspace member .../ojas-capi").
#
# `numcheck` is a path dev-dependency and not a workspace member (this crate
# declares no `[workspace]` table), so `cargo fmt -p numcheck` cannot select it;
# it gets its own invocation by manifest path.
#
# Invoke: ./scripts/fmt_check.sh [--fix]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ "${1:-}" = "--fix" ]; then
  cargo fmt -p tessl
  cargo fmt --manifest-path numcheck/Cargo.toml
else
  cargo fmt -p tessl --check
  cargo fmt --manifest-path numcheck/Cargo.toml --check
fi
