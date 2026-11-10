#!/usr/bin/env bash
# Reads `python -m unittest -v` output on stdin and fails if any test skipped because a
# fixture or the library it needs was not there when the stage ran.
#
# Those skips print OK at the end of the run, so a stage that ran before the fixtures were
# written (the tiny EmbeddingGemma 2 snapshots come from `cargo test --test capi_embedgemma2*`)
# passed with 22 tests not run. Skips for the real checkpoint (EMBEDGEMMA2_SNAPSHOT, *_REF_DIR)
# are expected on this machine and are not matched here: they are the only skips left allowed.
#
# Matched by the start of the reason, so a truncated line still counts (fail closed):
#   skipped 'run `cargo test ...             the tiny snapshot/checkpoint was never written
#   skipped 'libtessl.dylib is not built     the library the stage loads was not built
#
# Exit 0: output examined, no such skip. Exit 1: some, listed on stdout with a count.
# Exit 2: no "Ran N tests" summary on stdin, so nothing was examined (empty input or a
# stage that died before unittest finished): never the same result as a clean run.
set -euo pipefail

input=$(cat)

if ! grep -Eq '^Ran [0-9]+ tests? in ' <<<"$input"; then
  echo "unittest_fixture_skips: no 'Ran N tests' summary in the output; nothing was examined" >&2
  exit 2
fi

pattern="skipped '(run \`cargo test|libtessl\\.dylib is not built)"
hits=$(grep -E "$pattern" <<<"$input" || true)
if [[ -z "$hits" ]]; then
  echo "fixture_skips=0"
  exit 0
fi

count=$(grep -c '' <<<"$hits")
echo "$hits"
echo "fixture_skips=$count"
exit 1
