#!/usr/bin/env bash
# Fail unless every named workflow has a completed, successful run for exactly
# the given commit. Used by release.yml so nothing publishes from a commit that
# CI has not verified.
#
#   require_green_ci.sh <sha> <workflow-file>...
#
# Fail-closed: no run, an in-progress run, a failed run, or an API error all
# exit non-zero. "No run" and "could not ask" are different messages but the
# same exit status; a check that could not run never reads as a pass.
# A workflow with several runs for the sha (re-runs) passes if any completed
# run concluded `success` and no run of it is still in progress.
#
# Env: GH_REPO (optional, owner/name) selects the repository; GH_TOKEN
# authenticates (needs `actions: read`). RUN_LIMIT bounds the runs inspected
# per workflow (default 30); callers bound wall time (release.yml: timeout-minutes).
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <sha> <workflow-file>..." >&2
  exit 2
fi
SHA="$1"; shift
if [[ ! "$SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: '$SHA' is not a full 40-hex commit sha" >&2
  exit 2
fi
LIMIT="${RUN_LIMIT:-30}"

fail=0
for wf in "$@"; do
  if ! rows=$(gh run list --commit "$SHA" --workflow "$wf" --limit "$LIMIT" \
      --json status,conclusion --jq '.[] | "\(.status)/\(.conclusion)"' 2>&1); then
    echo "error: could not query runs of $wf for $SHA: $rows" >&2
    fail=1; continue
  fi
  if [[ -z "$rows" ]]; then
    echo "error: $wf has no run for $SHA (path-filtered, not yet started, or wrong commit)" >&2
    fail=1; continue
  fi
  if grep -Ev '^completed/' <<<"$rows" >/dev/null; then
    echo "error: $wf has a run for $SHA that is not completed:" >&2
    sort <<<"$rows" | uniq -c >&2
    fail=1; continue
  fi
  if ! grep -qx 'completed/success' <<<"$rows"; then
    echo "error: $wf has no successful run for $SHA:" >&2
    sort <<<"$rows" | uniq -c >&2
    fail=1; continue
  fi
  echo "ok: $wf succeeded for $SHA"
done
exit $fail
