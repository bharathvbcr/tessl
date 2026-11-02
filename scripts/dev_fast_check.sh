#!/usr/bin/env bash
# Fast inner development loop for tessl (< 15s).
# Checks syntax (cargo check), runs surgical lints (cargo clippy --all-targets -- -D warnings),
# and executes unit tests (cargo test --lib -- --test-threads=1).
#
# Usage:
#   ./scripts/dev_fast_check.sh [OPTIONS] [MODULE_OR_TEST_FILTER...]
#
# Examples:
#   ./scripts/dev_fast_check.sh                     # Defaults to runtime::tests (fast smoke)
#   ./scripts/dev_fast_check.sh tensor              # Tests tensor module
#   ./scripts/dev_fast_check.sh gemm::tests         # Tests gemm unit tests
#   ./scripts/dev_fast_check.sh --all               # Runs full lib test suite
#   ./scripts/dev_fast_check.sh --skip-clippy
#   ./scripts/dev_fast_check.sh --check-only

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

START_TIME=$(date +%s)

TMP_LOG_FILES=()
cleanup_logs() {
  if [[ ${#TMP_LOG_FILES[@]} -gt 0 ]]; then
    for f in "${TMP_LOG_FILES[@]}"; do
      [[ -f "$f" ]] && rm -f "$f"
    done
  fi
  return 0
}
trap cleanup_logs EXIT INT TERM

show_help() {
  cat <<'EOF'
dev_fast_check.sh: Fast inner dev loop syntax & unit check for tessl (< 15s)

Usage:
  ./scripts/dev_fast_check.sh [OPTIONS] [MODULE_OR_TEST_FILTER...]

Options:
  -h, --help       Show this help message
  --all            Run all unit tests in the library
  --check-only     Run only syntax/type checking (cargo check)
  --clippy         Run clippy linting stage (omitted by default for sub-15s inner loop)
  --skip-clippy    Skip clippy linting stage (default)
  --skip-tests     Skip unit tests stage

Default test target if none specified: runtime::tests
EOF
}

TARGETS=()
CHECK_ONLY=0
SKIP_CLIPPY=1
SKIP_TESTS=0
ALL_TESTS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    --all)
      ALL_TESTS=1
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --clippy)
      SKIP_CLIPPY=0
      shift
      ;;
    --skip-clippy)
      SKIP_CLIPPY=1
      shift
      ;;
    --skip-tests)
      SKIP_TESTS=1
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      show_help >&2
      exit 1
      ;;
    *)
      TARGETS+=("$1")
      shift
      ;;
  esac
done

if [[ "$ALL_TESTS" -eq 1 ]]; then
  TEST_ARGS=()
  TARGET_DESC="all lib tests"
elif [[ "${#TARGETS[@]}" -eq 0 ]]; then
  TEST_ARGS=(runtime::tests)
  TARGET_DESC="runtime::tests"
else
  TEST_ARGS=("${TARGETS[@]}")
  TARGET_DESC="${TARGETS[*]}"
fi

echo "======================================================================"
echo "==> [Fast Check - tessl] Target: $TARGET_DESC"
echo "======================================================================"

# Stage 1: Syntax & Typecheck
echo "==> [Stage 1/3] Syntax & Typecheck (cargo check --lib)"
STAGE1_START=$(date +%s)
cargo check --lib
STAGE1_END=$(date +%s)
echo "    -> Completed in $((STAGE1_END - STAGE1_START))s"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  TOTAL_TIME=$(($(date +%s) - START_TIME))
  echo "==> Fast check completed in ${TOTAL_TIME}s (check-only mode)."
  exit 0
fi

# Stage 2: Fast Clippy
if [[ "$SKIP_CLIPPY" -eq 0 ]]; then
  echo "==> [Stage 2/3] Clippy (cargo clippy --lib -- -D warnings)"
  STAGE2_START=$(date +%s)
  cargo clippy --lib -- -D warnings
  STAGE2_END=$(date +%s)
  echo "    -> Completed in $((STAGE2_END - STAGE2_START))s"
else
  echo "==> [Stage 2/3] Clippy skipped (fast inner loop default; pass --clippy to enable)"
fi

# Stage 3: In-crate Unit Tests (serialized on Apple Silicon)
if [[ "$SKIP_TESTS" -eq 0 ]]; then
  echo "==> [Stage 3/3] Unit Tests (cargo test --lib ${TEST_ARGS[*]:-} -- --test-threads=1)"
  STAGE3_START=$(date +%s)

  local_log=$(mktemp /tmp/tessl_fast_check_XXXXXX)
  TMP_LOG_FILES+=("$local_log")

  set +e
  cargo test --lib "${TEST_ARGS[@]+"${TEST_ARGS[@]}"}" -- --test-threads=1 2>&1 | tee "$local_log"
  TEST_RC=${PIPESTATUS[0]}
  set -e

  if [[ $TEST_RC -ne 0 ]]; then
    rm -f "$local_log"
    echo "ERROR: Unit tests failed with exit code $TEST_RC" >&2
    exit "$TEST_RC"
  fi

  if grep -qE "(running 0 tests|test result: .* 0 passed; 0 failed)" "$local_log"; then
    rm -f "$local_log"
    echo "" >&2
    echo "======================================================================" >&2
    echo "ERROR: No unit tests matched filter '${TEST_ARGS[*]:-}' in tessl!" >&2
    echo "0 unit tests were executed." >&2
    echo "======================================================================" >&2
    exit 1
  fi

  rm -f "$local_log"
  STAGE3_END=$(date +%s)
  echo "    -> Completed in $((STAGE3_END - STAGE3_START))s"
else
  echo "==> [Stage 3/3] Unit tests skipped (--skip-tests)"
fi

TOTAL_TIME=$(($(date +%s) - START_TIME))
echo "======================================================================"
echo "==> [Fast Check - tessl] SUCCESS in ${TOTAL_TIME}s (target: < 15s)"
echo "======================================================================"
