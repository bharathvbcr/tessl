#!/usr/bin/env bash
# Run the device-free cargo tests listed in scripts/hosted_tests.txt, the way the
# hosted macOS job does, and fail unless every invocation actually ran tests.
#
#   scripts/hosted_tests.sh [MANIFEST]     (default: scripts/hosted_tests.txt)
#
# One guard for every invocation. A cargo filter that matches nothing exits 0
# with "0 passed", and a renamed test then reads as a pass; here
#   bin    NAME COUNT     the whole integration binary tests/NAME.rs: exactly COUNT tests ran
#   bins   COUNT          every unit test of the src/bin targets (`--bins`): COUNT in total
#   lib    FILTER COUNT   `--lib` filtered by name prefix: exactly COUNT tests ran
#   exact  lib|BIN NAME.. each NAME run with --exact: exactly as many tests ran as
#                         names given, so one name that matches nothing fails
#   ignored BIN NAME      one #[ignore]d test, run on purpose with --ignored --exact: it ran
#   outside BIN NAME      like `ignored`, but run without the sandbox and only with
#                         `--outside` (tests/hosted_completeness.rs: it starts its own
#                         sandboxed children, and sandbox-exec does not nest)
# COUNT is exact, so a deleted or renamed test fails here, and a new one fails here
# until the manifest is updated; the completeness check finds the ones a list of
# names would miss. A malformed manifest line fails too.
#
# Usage: scripts/hosted_tests.sh [--outside] [MANIFEST]
#   default    every entry except `outside`, under the sandbox
#   --outside  only the `outside` entries, after the sandboxed step
#
# No device. Every invocation runs under a sandbox profile that denies IOKit, so
# MTLCreateSystemDefaultDevice returns nil even on a runner that has a GPU, and
# with TESSL_HOSTED_NO_DEVICE=1 so tests/common's runtime helpers refuse by name.
# A test in the list that opens a device therefore fails; it is never skipped.
# `ignored hosted_selection hosted_selection_has_no_device` proves the sandbox
# hides the device before anything is trusted to run under it.
#
# Environment (for tests/hosted_tests_script.rs, not for CI):
#   TESSL_HOSTED_CARGO   the cargo to run (default: cargo)
#   TESSL_HOSTED_WRAP    the command prefix that hides the device (default:
#                        sandbox-exec with the IOKit-deny profile)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE=sandboxed
if [[ "${1:-}" == --outside ]]; then
  MODE=outside
  shift
fi
MANIFEST="${1:-$ROOT/scripts/hosted_tests.txt}"
CARGO="${TESSL_HOSTED_CARGO:-cargo}"
PROFILE='(version 1)(allow default)(deny iokit-open)'

fail() {
  echo "hosted_tests: $*" >&2
  exit 1
}

[[ -f "$MANIFEST" ]] || fail "manifest not found: $MANIFEST"

if [[ "$MODE" == outside ]]; then
  WRAP=(env)
elif [[ -n "${TESSL_HOSTED_WRAP:-}" ]]; then
  read -r -a WRAP <<<"$TESSL_HOSTED_WRAP"
else
  command -v sandbox-exec >/dev/null 2>&1 || fail "sandbox-exec not found; refusing to run device-free tests with a device visible"
  WRAP=(sandbox-exec -p "$PROFILE")
fi

# Build outside the sandbox (build.rs shells out to the Metal compiler), then run
# inside it. A build failure is a failure of this script, not a skip.
echo "==> build the test binaries"
"$CARGO" test --release --no-run || fail "building the test binaries failed"

# run_guarded WANT_N LABEL CARGO_ARGS... : run, require a clean result, and require that
# exactly WANT_N tests ran.
run_guarded() {
  local want="$1" label="$2"
  shift 2
  local out ran n results=0
  echo "==> $label"
  out="$(TESSL_HOSTED_NO_DEVICE=1 "${WRAP[@]}" "$CARGO" test --release "$@" 2>&1)" || {
    echo "$out"
    fail "$label: cargo test failed"
  }
  echo "$out"
  ran=0
  while IFS= read -r n; do
    ran=$((ran + n))
    results=$((results + 1))
  done < <(sed -n 's/^test result: ok\. \([0-9][0-9]*\) passed.*/\1/p' <<<"$out")
  [[ "$results" -gt 0 ]] || fail "$label: no 'test result: ok' line; nothing ran"
  [[ "$ran" -gt 0 ]] || fail "$label: 0 tests ran (the filter matched nothing)"
  if [[ "$ran" -ne "$want" ]]; then
    fail "$label: $ran test(s) ran, expected exactly $want (a test was added, removed or renamed: update the manifest)"
  fi
  TOTAL=$((TOTAL + ran))
}

is_count() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

TOTAL=0
INVOCATIONS=0
lineno=0
while IFS= read -r line || [[ -n "$line" ]]; do
  lineno=$((lineno + 1))
  line="${line%%#*}"
  read -r -a f <<<"$line"
  [[ ${#f[@]} -gt 0 ]] || continue
  at="$MANIFEST:$lineno"
  case "${f[0]}" in
    bin)
      if [[ ${#f[@]} -ne 3 ]] || ! is_count "${f[2]}"; then fail "$at: 'bin NAME COUNT' takes a name and a positive count"; fi
      [[ "$MODE" == sandboxed ]] || continue
      run_guarded "${f[2]}" "bin ${f[1]}" --test "${f[1]}" -- --test-threads=1
      ;;
    bins)
      if [[ ${#f[@]} -ne 2 ]] || ! is_count "${f[1]}"; then fail "$at: 'bins COUNT' takes a positive count"; fi
      [[ "$MODE" == sandboxed ]] || continue
      run_guarded "${f[1]}" "bins (src/bin unit tests)" --bins -- --test-threads=1
      ;;
    lib)
      if [[ ${#f[@]} -ne 3 ]] || ! is_count "${f[2]}"; then fail "$at: 'lib FILTER COUNT' takes a filter and a positive count"; fi
      [[ "$MODE" == sandboxed ]] || continue
      run_guarded "${f[2]}" "lib ${f[1]}" --lib -- --test-threads=1 "${f[1]}"
      ;;
    exact)
      [[ ${#f[@]} -ge 3 ]] || fail "$at: 'exact lib|BIN NAME...' needs a target and a name"
      [[ "$MODE" == sandboxed ]] || continue
      if [[ "${f[1]}" == lib ]]; then target=(--lib); else target=(--test "${f[1]}"); fi
      run_guarded $((${#f[@]} - 2)) "exact ${f[1]} ($((${#f[@]} - 2)) names)" "${target[@]}" -- --test-threads=1 --exact "${f[@]:2}"
      ;;
    ignored)
      [[ ${#f[@]} -eq 3 ]] || fail "$at: 'ignored BIN NAME' takes a binary and one name"
      [[ "$MODE" == sandboxed ]] || continue
      run_guarded 1 "ignored ${f[1]} ${f[2]}" --test "${f[1]}" -- --test-threads=1 --ignored --exact "${f[2]}"
      ;;
    outside)
      [[ ${#f[@]} -eq 3 ]] || fail "$at: 'outside BIN NAME' takes a binary and one name"
      [[ "$MODE" == outside ]] || continue
      run_guarded 1 "outside ${f[1]} ${f[2]}" --test "${f[1]}" -- --test-threads=1 --ignored --exact "${f[2]}"
      ;;
    *) fail "$at: unknown entry kind '${f[0]}'" ;;
  esac
  INVOCATIONS=$((INVOCATIONS + 1))
done <"$MANIFEST"

[[ "$INVOCATIONS" -gt 0 ]] || fail "$MANIFEST has no $MODE entries; nothing to run"
echo "hosted_tests ($MODE): OK ($INVOCATIONS invocations, $TOTAL tests ran)"
