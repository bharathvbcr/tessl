#!/usr/bin/env bash
# Run the device-free cargo tests listed in scripts/hosted_tests.txt, the way the
# hosted macOS job does, and fail unless every invocation actually ran tests.
#
#   scripts/hosted_tests.sh [MANIFEST]     (default: scripts/hosted_tests.txt)
#
# One guard for every invocation. A cargo filter that matches nothing exits 0
# with "0 passed", and a renamed test then reads as a pass; here
#   bin    NAME           the whole integration binary tests/NAME.rs: at least one test ran
#   lib    FILTER         `--lib` filtered by name prefix: at least one test ran
#   exact  lib|BIN NAME.. each NAME run with --exact: exactly as many tests ran as
#                         names given, so one name that matches nothing fails
#   ignored BIN NAME      one #[ignore]d test, run on purpose with --ignored --exact: it ran
# A malformed manifest line fails too.
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
MANIFEST="${1:-$ROOT/scripts/hosted_tests.txt}"
CARGO="${TESSL_HOSTED_CARGO:-cargo}"
PROFILE='(version 1)(allow default)(deny iokit-open)'

fail() {
  echo "hosted_tests: $*" >&2
  exit 1
}

[[ -f "$MANIFEST" ]] || fail "manifest not found: $MANIFEST"

if [[ -n "${TESSL_HOSTED_WRAP:-}" ]]; then
  read -r -a WRAP <<<"$TESSL_HOSTED_WRAP"
else
  command -v sandbox-exec >/dev/null 2>&1 || fail "sandbox-exec not found; refusing to run device-free tests with a device visible"
  WRAP=(sandbox-exec -p "$PROFILE")
fi

# Build outside the sandbox (build.rs shells out to the Metal compiler), then run
# inside it. A build failure is a failure of this script, not a skip.
echo "==> build the test binaries"
"$CARGO" test --release --no-run || fail "building the test binaries failed"

# run_guarded WANT_KIND WANT_N CARGO_ARGS... : run, require a clean result, require the ran count.
#   WANT_KIND "min": at least WANT_N tests ran. "eq": exactly WANT_N ran.
run_guarded() {
  local kind="$1" want="$2" label="$3"
  shift 3
  local out ran
  echo "==> $label"
  out="$(TESSL_HOSTED_NO_DEVICE=1 "${WRAP[@]}" "$CARGO" test --release "$@" 2>&1)" || {
    echo "$out"
    fail "$label: cargo test failed"
  }
  echo "$out"
  local n results=0
  ran=0
  while IFS= read -r n; do
    ran=$((ran + n))
    results=$((results + 1))
  done < <(sed -n 's/^test result: ok\. \([0-9][0-9]*\) passed.*/\1/p' <<<"$out")
  [[ "$results" -gt 0 ]] || fail "$label: no 'test result: ok' line; nothing ran"
  if [[ "$kind" == "eq" && "$ran" -ne "$want" ]]; then
    fail "$label: $ran test(s) ran, expected exactly $want (a name that matches nothing, or a renamed test)"
  fi
  if [[ "$kind" == "min" && "$ran" -lt "$want" ]]; then
    fail "$label: $ran test(s) ran, expected at least $want (the filter matched nothing)"
  fi
  TOTAL=$((TOTAL + ran))
}

TOTAL=0
INVOCATIONS=0
lineno=0
while IFS= read -r line || [[ -n "$line" ]]; do
  lineno=$((lineno + 1))
  line="${line%%#*}"
  read -r -a f <<<"$line"
  [[ ${#f[@]} -gt 0 ]] || continue
  INVOCATIONS=$((INVOCATIONS + 1))
  case "${f[0]}" in
    bin)
      [[ ${#f[@]} -eq 2 ]] || fail "$MANIFEST:$lineno: 'bin NAME' takes one name"
      run_guarded min 1 "bin ${f[1]}" --test "${f[1]}" -- --test-threads=1
      ;;
    lib)
      [[ ${#f[@]} -eq 2 ]] || fail "$MANIFEST:$lineno: 'lib FILTER' takes one filter"
      run_guarded min 1 "lib ${f[1]}" --lib -- --test-threads=1 "${f[1]}"
      ;;
    exact)
      [[ ${#f[@]} -ge 3 ]] || fail "$MANIFEST:$lineno: 'exact lib|BIN NAME...' needs a target and a name"
      if [[ "${f[1]}" == lib ]]; then target=(--lib); else target=(--test "${f[1]}"); fi
      run_guarded eq $((${#f[@]} - 2)) "exact ${f[1]} ($((${#f[@]} - 2)) names)" "${target[@]}" -- --test-threads=1 --exact "${f[@]:2}"
      ;;
    ignored)
      [[ ${#f[@]} -eq 3 ]] || fail "$MANIFEST:$lineno: 'ignored BIN NAME' takes a binary and one name"
      run_guarded eq 1 "ignored ${f[1]} ${f[2]}" --test "${f[1]}" -- --test-threads=1 --ignored --exact "${f[2]}"
      ;;
    *) fail "$MANIFEST:$lineno: unknown entry kind '${f[0]}'" ;;
  esac
done <"$MANIFEST"

[[ "$INVOCATIONS" -gt 0 ]] || fail "$MANIFEST lists no tests; nothing to run"
echo "hosted_tests: OK ($INVOCATIONS invocations, $TOTAL tests ran, none with a device)"
