#!/usr/bin/env bash
# The Python GPU stage of scripts/ci_local.sh: run python/tests, keep the output, and fail unless
# every skip in it is an external-checkpoint skip (scripts/unittest_fixture_skips.sh). ci_local.sh
# owns the MAC_HEAVY_LABEL guard and runs this only under the heavy-job lock; tests/ci_local_python_stage.rs
# runs this very file with a stand-in python.
#
# Needs ORACLE_PY (the torch python ci_local found). Exit codes, which ci_local.sh names:
#   0  tests passed and every skip is an external-checkpoint skip
#   1  the Python run itself failed (non-zero exit, a signal, or python could not start)
#   2  the output could not be kept (the log is unwritable): the skip report was not examined
#   3  tests passed but some skip is not an external-checkpoint skip (a missing tiny fixture, ...)
#   4  tests passed but the skip report could not be examined (no summary, count mismatch)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
: "${ORACLE_PY:?ORACLE_PY is not set}"

log="target/python_gpu_unittest.log"
# A log left by an earlier run must never be read as this run's output.
rm -f "$log" 2>/dev/null || true
mkdir -p target 2>/dev/null || true

"$ORACLE_PY" -m unittest discover -s python/tests -v 2>&1 | tee "$log"
status=("${PIPESTATUS[@]}")

if [[ "${status[0]}" -ne 0 ]]; then
  echo "python/tests failed (python exit ${status[0]})" >&2
  exit 1
fi
if [[ "${status[1]}" -ne 0 || ! -f "$log" ]]; then
  echo "python/tests output could not be written to $log (tee exit ${status[1]:-?}); the skip report was not examined" >&2
  exit 2
fi

detector_rc=0
scripts/unittest_fixture_skips.sh < "$log" || detector_rc=$?
case "$detector_rc" in
  0) exit 0 ;;
  1) exit 3 ;;
  *) exit 4 ;;
esac
