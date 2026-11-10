#!/usr/bin/env bash
# Reads `python -m unittest -v` output on stdin and fails on every skip except the ones whose
# cause this ci_local run cannot produce itself: an external checkpoint or reference, whose
# reason starts `set EMBEDGEMMA2_` (EMBEDGEMMA2_SNAPSHOT, *_REF_DIR, EMBEDGEMMA2_COMPARE_ST).
#
# Why an allow-list: a skip prints OK at the end of the run. The tiny EmbeddingGemma 2 snapshots
# come from `cargo test --test capi_embedgemma2*`, a stage that ran after the Python stage, so on a
# fresh target dir 22 tests skipped and the stage passed (wave 5: OK (skipped=42), 20 expected).
# A deny-list of the wordings seen so far would pass the next one, so everything else fails:
# a missing tiny fixture, an unbuilt library, a missing reference script output, an oracle
# package that is not installed. tests/ci_local_python_stage.rs derives the reasons from
# python/tests and holds this classification to them.
#
# unittest prints `... skipped 'reason'` and, when the reason holds a `'`, `skipped "reason"`.
# A class-level or setUpClass skip is printed once per test or once per class, and the summary's
# `skipped=N` counts the same. The skip lines parsed must equal that N, or the output was not
# the whole story (a line lost or cut) and it was not examined.
#
# Exit 0: examined; every skip is an external-checkpoint skip. Exit 1: some skip is not, listed on
# stdout with `unexpected_skips=N`. Exit 2: not examined (no `Ran N tests` line, no OK/FAILED
# verdict line, or the parsed skip count differs from the summary's); never the same as a pass.
set -euo pipefail

input=$(cat)
unexamined() {
  echo "unittest_fixture_skips: $1; nothing was examined" >&2
  exit 2
}

grep -Eq '^Ran [0-9]+ tests? in ' <<<"$input" || unexamined "no 'Ran N tests' summary in the output"
verdict=$(grep -E '^(OK|FAILED)( \(.*\))?$' <<<"$input" | tail -n 1 || true)
[[ -n "$verdict" ]] || unexamined "no OK/FAILED verdict line in the output"
declared=0
if [[ "$verdict" =~ skipped=([0-9]+) ]]; then declared=${BASH_REMATCH[1]}; fi

skip_re="(^| )skipped ['\"]"
allowed_re="(^| )skipped ['\"]set EMBEDGEMMA2_"
parsed=$(grep -Ec "$skip_re" <<<"$input" || true)
unexpected=$(grep -E "$skip_re" <<<"$input" | grep -Ev "$allowed_re" || true)

if [[ -n "$unexpected" ]]; then
  echo "$unexpected"
  echo "unexpected_skips=$(grep -c '' <<<"$unexpected")"
  exit 1
fi
[[ "$parsed" -eq "$declared" ]] || unexamined "parsed $parsed skip lines but the summary says skipped=$declared"
echo "unexpected_skips=0 (external-checkpoint skips: $parsed)"
