#!/bin/bash
# Reconstructed guard-test break-it driver.
#
# The original 14-perturbation script, target/lead_run_guard_round3.sh, lived in
# the wave-25 worktree and was not recovered (untracked, worktree deleted, not in
# git). This table is reconstructed from the assertions in tests/reductions.rs and
# tests/nn_adversarial.rs. It is not a claim that these are the same 14 edits.
#
# Each row inverts one condition so the same assertion message still fires.
# A non-zero exit alone is not a catch: the failing test's output must contain
# that message. Flipping only the comparison of the tiled f16 witness while
# leaving `!got.is_finite() ||` in place would still pass when the tile sum is
# inf, so that row requires the tiled sum to be finite and close.
#
#   scripts/guard_break_it.sh --check        each old string is unique (no device)
#   scripts/guard_break_it.sh --self-test    driver rules, no device, no repo edits
#   scripts/guard_break_it.sh --list
#   scripts/guard_break_it.sh [--only id,id] baselines, then one perturbation at a time
#
# The two test files are copied aside and restored by an EXIT/INT/TERM/HUP trap.
# The run refuses to start when either file differs from HEAD unless
# TESSL_GUARD_ALLOW_DIRTY=1. Restore puts back the bytes from the start of the
# run, and the script exits non-zero if those hashes do not match.
#
# GPU rows take the device. Run them under the machine lock, not beside another
# tessl GPU job:
#   TESSL_GUARD_ALLOW_DIRTY=1 bash <Lappi>/tools/mac_heavy.sh tessl-w28-guards \
#       bash scripts/guard_break_it.sh
#
# A baseline that fails is NOT RUN for its perturbations. Do not loosen a bound
# to make a baseline pass. Timeouts: TESSL_GUARD_TEST_TIMEOUT (default 1800).
# CARGO_TARGET_DIR is unset so this worktree does not reuse another checkout's
# target. CARGO_BUILD_JOBS is left as the caller set it (mac_heavy defaults it).

set -uo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root" || exit 2

test_timeout=${TESSL_GUARD_TEST_TIMEOUT:-1800}
mode=run
only=""

die() { echo "guard_break_it: $*" >&2; exit 2; }

usage() {
  echo "usage: scripts/guard_break_it.sh [--check | --self-test | --list | [--only id,id]]"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --check) mode=check ;;
    --self-test) mode=self ;;
    --list) mode=list ;;
    --plan) mode=plan ;;
    --only)
      shift
      [ $# -gt 0 ] || die "--only needs a comma-separated id list"
      only=$1
      ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

if [ -n "${CARGO_TARGET_DIR:-}" ]; then
  echo "guard_break_it: unsetting CARGO_TARGET_DIR=$CARGO_TARGET_DIR" >&2
  unset CARGO_TARGET_DIR
fi

# py SUBCMD [ARG] — the perturbation table and the byte-exact replace.
py() {
  python3 - "$root" "$@" <<'PY'
import sys
root, cmd = sys.argv[1], sys.argv[2]
arg = sys.argv[3] if len(sys.argv) > 3 else ""

def R(*lines):
    return "\n".join(lines)

ROWS = [
    {
        "id": "nan-sum",
        "file": "tests/reductions.rs",
        "bin": "reductions",
        "test": "planted_non_finite_inputs_poison_exactly_their_rows",
        "needle": "NaN + finite = NaN",
        "old": '            assert!(s[1].is_nan(), "{label}: NaN + finite = NaN, got {}", s[1]);',
        "new": '            assert!(!s[1].is_nan(), "{label}: NaN + finite = NaN, got {}", s[1]);',
    },
    {
        "id": "pinf-sum",
        "file": "tests/reductions.rs",
        "bin": "reductions",
        "test": "planted_non_finite_inputs_poison_exactly_their_rows",
        "needle": "sum of a +inf row",
        "old": '            assert_eq!(s[2], f32::INFINITY, "{label}: sum of a +inf row");',
        "new": '            assert_eq!(s[2], 0.0, "{label}: sum of a +inf row");',
    },
    {
        "id": "ninf-sum",
        "file": "tests/reductions.rs",
        "bin": "reductions",
        "test": "planted_non_finite_inputs_poison_exactly_their_rows",
        "needle": "sum of a -inf row",
        "old": '            assert_eq!(s[3], f32::NEG_INFINITY, "{label}: sum of a -inf row");',
        "new": '            assert_eq!(s[3], 0.0, "{label}: sum of a -inf row");',
    },
    {
        "id": "softmax-nan",
        "file": "tests/reductions.rs",
        "bin": "reductions",
        "test": "planted_non_finite_inputs_poison_exactly_their_rows",
        "needle": "softmax of a row holding",
        "old": R(
            "                assert!(",
            "                    out.iter().all(|p| p.is_nan()),",
            '                    "{label}: softmax of a row holding {what} must be NaN throughout, got {} NaN of {cols}, first {:?}",',
        ),
        "new": R(
            "                assert!(",
            "                    out.iter().all(|p| !p.is_nan()),",
            '                    "{label}: softmax of a row holding {what} must be NaN throughout, got {} NaN of {cols}, first {:?}",',
        ),
    },
    {
        "id": "masked-bound",
        "file": "tests/reductions.rs",
        "bin": "reductions",
        "test": "planted_non_finite_inputs_poison_exactly_their_rows",
        "needle": "softmax of a masked row is not a distribution",
        "old": R(
            "                out3.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p))",
            "                    && (out3.iter().map(|p| f64::from(*p)).sum::<f64>() - 1.0).abs() <= sum_bound,",
            '                "{label}: softmax of a masked row is not a distribution (bound {sum_bound})"',
        ),
        "new": R(
            "                out3.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p))",
            "                    && (out3.iter().map(|p| f64::from(*p)).sum::<f64>() - 1.0).abs() > sum_bound,",
            '                "{label}: softmax of a masked row is not a distribution (bound {sum_bound})"',
        ),
    },
    {
        "id": "softmax-sum",
        "file": "tests/reductions.rs",
        "bin": "reductions",
        "test": "softmax_matches_a_f64_reference",
        "needle": "not 1 (bound",
        "old": R(
            "                assert!(",
            "                    (sum - 1.0).abs() <= sum_bound,",
            '                    "row {r} of {rows}x{cols} sums to {sum}, not 1 (bound {sum_bound})"',
        ),
        "new": R(
            "                assert!(",
            "                    (sum - 1.0).abs() > sum_bound,",
            '                    "row {r} of {rows}x{cols} sums to {sum}, not 1 (bound {sum_bound})"',
        ),
    },
    {
        "id": "sub-sum",
        "file": "tests/reductions.rs",
        "bin": "reductions",
        "test": "subnormal_inputs_are_summed_or_flushed_never_invented",
        "needle": "row_sum_f32 flushes a subnormal row",
        "old": R(
            "        assert_eq!(",
            "            s[0], 0.0,",
            '            "row_sum_f32 flushes a subnormal row: sum of {cols} x {sub:e} got {:e}",',
            "            s[0]",
        ),
        "new": R(
            "        assert_eq!(",
            "            s[0], sub,",
            '            "row_sum_f32 flushes a subnormal row: sum of {cols} x {sub:e} got {:e}",',
            "            s[0]",
        ),
    },
    {
        "id": "sub-max",
        "file": "tests/reductions.rs",
        "bin": "reductions",
        "test": "subnormal_inputs_are_summed_or_flushed_never_invented",
        "needle": "row_max_f32 keeps a subnormal",
        "old": R(
            "        assert_eq!(",
            "            m[0], sub,",
            '            "row_max_f32 keeps a subnormal: max got {:e}, want {sub:e}",',
            "            m[0]",
        ),
        "new": R(
            "        assert_eq!(",
            "            m[0], 0.0,",
            '            "row_max_f32 keeps a subnormal: max got {:e}, want {sub:e}",',
            "            m[0]",
        ),
    },
    {
        "id": "f16-tiled",
        "file": "tests/reductions.rs",
        "bin": "reductions",
        "test": "f16_operands_accumulate_in_f32",
        "needle": "tiled f16 accumulator",
        "old": R(
            "                    assert!(",
            "                        !got.is_finite() || (got - want).abs() > 100.0 * bound,",
            '                        "tile {tile}: tiled f16 accumulator is not off the f32 result ({got} vs {want} +- {bound})"',
        ),
        "new": R(
            "                    assert!(",
            "                        got.is_finite() && (got - want).abs() <= 100.0 * bound,",
            '                        "tile {tile}: tiled f16 accumulator is not off the f32 result ({got} vs {want} +- {bound})"',
        ),
    },
    {
        "id": "f16-gpu",
        "file": "tests/reductions.rs",
        "bin": "reductions",
        "test": "f16_operands_accumulate_in_f32",
        "needle": "f32-accumulated reference",
        "old": R(
            "                            assert!(",
            "                                (f64::from(*got) - want).abs() <= bound,",
            '                                "k={} {backend:?}: c[{i}] = {got}, f32-accumulated reference {want} +- {bound} (an f16 accumulator gives {f16_sum})",',
        ),
        "new": R(
            "                            assert!(",
            "                                (f64::from(*got) - want).abs() > bound,",
            '                                "k={} {backend:?}: c[{i}] = {got}, f32-accumulated reference {want} +- {bound} (an f16 accumulator gives {f16_sum})",',
        ),
    },
    {
        "id": "rms-nan",
        "file": "tests/nn_adversarial.rs",
        "bin": "nn_adversarial",
        "test": "rms_norm_planted_non_finite_and_subnormal_rows",
        "needle": "a NaN input must make its whole row NaN",
        "old": R(
            "        assert!(",
            "            row(&got, 1).iter().all(|v| v.is_nan()),",
            '            "a NaN input must make its whole row NaN"',
        ),
        "new": R(
            "        assert!(",
            "            row(&got, 1).iter().all(|v| !v.is_nan()),",
            '            "a NaN input must make its whole row NaN"',
        ),
    },
    {
        "id": "rms-pinf",
        "file": "tests/nn_adversarial.rs",
        "bin": "nn_adversarial",
        "test": "rms_norm_planted_non_finite_and_subnormal_rows",
        "needle": "planted +inf",
        "old": R(
            "        assert!(",
            "            r2[c_inf].is_nan(),",
            '            "inf * rsqrt(inf) = inf * 0 = NaN at the planted +inf"',
        ),
        "new": R(
            "        assert!(",
            "            !r2[c_inf].is_nan(),",
            '            "inf * rsqrt(inf) = inf * 0 = NaN at the planted +inf"',
        ),
    },
    {
        "id": "rms-ninf",
        "file": "tests/nn_adversarial.rs",
        "bin": "nn_adversarial",
        "test": "rms_norm_planted_non_finite_and_subnormal_rows",
        "needle": "planted -inf",
        "old": R(
            "        assert!(",
            "            r4[c_inf].is_nan(),",
            '            "inf * rsqrt(inf) = inf * 0 = NaN at the planted -inf"',
        ),
        "new": R(
            "        assert!(",
            "            !r4[c_inf].is_nan(),",
            '            "inf * rsqrt(inf) = inf * 0 = NaN at the planted -inf"',
        ),
    },
    {
        "id": "rms-sub",
        "file": "tests/nn_adversarial.rs",
        "bin": "nn_adversarial",
        "test": "rms_norm_planted_non_finite_and_subnormal_rows",
        "needle": "rms_norm flushes a subnormal row",
        "old": '            assert_eq!(*v, 0.0, "rms_norm flushes a subnormal row: [{c}] = {v:e}, want 0");',
        "new": '            assert_eq!(*v, 1.0, "rms_norm flushes a subnormal row: [{c}] = {v:e}, want 0");',
    },
]

def selected():
    if not arg:
        return list(ROWS)
    want = [p for p in arg.split(",") if p]
    by = {r["id"]: r for r in ROWS}
    missing = [i for i in want if i not in by]
    if missing:
        sys.stderr.write("unknown id: %s\n" % ",".join(missing))
        sys.exit(2)
    return [by[i] for i in want]

def count(data, old):
    if isinstance(data, str):
        data = data.encode()
    return data.count(old.encode())

def self_test():
    errors = []
    if len(ROWS) != 14:
        errors.append("expected 14 rows, got %d" % len(ROWS))
    ids = [r["id"] for r in ROWS]
    if len(ids) != len(set(ids)):
        errors.append("duplicate id")
    olds = [r["old"] for r in ROWS]
    if len(olds) != len(set(olds)):
        errors.append("duplicate old string")
    tiled = next(r for r in ROWS if r["id"] == "f16-tiled")
    if "!got.is_finite()" in tiled["new"]:
        errors.append("f16-tiled new still accepts a non-finite tile sum")
    if "got.is_finite() &&" not in tiled["new"]:
        errors.append("f16-tiled new does not require a finite tile sum")
    for r in ROWS:
        if not r["needle"] or len(r["needle"]) < 8:
            errors.append("%s needle too short" % r["id"])
        if r["needle"] not in r["old"] or r["needle"] not in r["new"]:
            errors.append("%s drops its assertion message" % r["id"])
        if r["old"] == r["new"]:
            errors.append("%s does not change the source" % r["id"])
    # Replace-once: a duplicate must not write, a miss must not write, one match writes.
    blob = "keep\nALPHA\nkeep\nALPHA\n"
    if count(blob, "ALPHA") != 2:
        errors.append("count failed")
    single = "keep\nALPHA\nkeep\n"
    raw = single.encode()
    n = count(single, "ALPHA")
    if n != 1:
        errors.append("single count")
    else:
        out = raw.replace("ALPHA".encode(), "BETA".encode(), 1)
        if out.count(b"BETA") != 1 or b"ALPHA" in out:
            errors.append("replace failed")
    if errors:
        sys.stderr.write("\n".join(errors) + "\n")
        sys.exit(1)
    print("self-test ok: 14 rows, messages kept, f16 tile must be finite, replace-once holds")

def check():
    bad = False
    for r in selected():
        path = root + "/" + r["file"]
        try:
            data = open(path, "rb").read()
        except OSError as e:
            print("%s MISSING FILE %s" % (r["id"], e))
            bad = True
            continue
        n = count(data, r["old"])
        print("%s %s" % (r["id"], n))
        if n != 1:
            bad = True
    if bad:
        sys.exit(1)

def plan():
    rows = selected()
    seen = []
    groups = {}
    for r in rows:
        key = (r["bin"], r["test"])
        if key not in groups:
            seen.append(key)
            groups[key] = []
        groups[key].append(r)
    for key in seen:
        bin_name, test = key
        sys.stdout.write("baseline\t%s\t%s\n" % (bin_name, test))
        for r in groups[key]:
            sys.stdout.write("row\t%s\n" % r["id"])

def apply_one():
    row = next((r for r in ROWS if r["id"] == arg), None)
    if row is None:
        sys.stderr.write("unknown id %s\n" % arg)
        sys.exit(2)
    path = root + "/" + row["file"]
    data = open(path, "rb").read()
    n = count(data, row["old"])
    if n != 1:
        sys.stderr.write("%s count %d\n" % (arg, n))
        sys.exit(1)
    open(path, "wb").write(data.replace(row["old"].encode(), row["new"].encode(), 1))

if cmd == "self":
    self_test()
elif cmd == "check":
    check()
elif cmd == "list":
    for r in ROWS:
        print("%s\t%s\t%s\t%s" % (r["id"], r["bin"], r["test"], r["needle"]))
elif cmd == "plan":
    plan()
elif cmd == "apply":
    apply_one()
else:
    sys.stderr.write("unknown py command %s\n" % cmd)
    sys.exit(2)
PY
}

if [ "$mode" = self ]; then
  py self
  exit $?
fi

if [ "$mode" = list ]; then
  py list
  exit $?
fi

if [ "$mode" = plan ]; then
  py plan "$only"
  exit $?
fi

dirty=$(git status --porcelain --untracked-files=all -- tests/reductions.rs tests/nn_adversarial.rs)
if [ -n "$dirty" ] && [ "${TESSL_GUARD_ALLOW_DIRTY:-}" != 1 ]; then
  echo "guard_break_it: tests/reductions.rs or tests/nn_adversarial.rs differs from HEAD." >&2
  echo "$dirty" >&2
  echo "guard_break_it: refusing. Set TESSL_GUARD_ALLOW_DIRTY=1 to run against this tree." >&2
  exit 2
fi

if [ "$mode" = check ]; then
  py check "$only"
  exit $?
fi

bak=$(mktemp -d /tmp/tessl-guard-bak.XXXXXX)
cp tests/reductions.rs tests/nn_adversarial.rs "$bak/"
before=$(shasum -a 256 tests/reductions.rs tests/nn_adversarial.rs)
restored=0
restore() {
  if [ "$restored" -eq 1 ]; then
    return
  fi
  restored=1
  cp "$bak/reductions.rs" tests/reductions.rs || exit 2
  cp "$bak/nn_adversarial.rs" tests/nn_adversarial.rs || exit 2
  after=$(shasum -a 256 tests/reductions.rs tests/nn_adversarial.rs)
  if [ "$before" != "$after" ]; then
    echo "guard_break_it: RESTORE FAILED" >&2
    echo "before: $before" >&2
    echo "after:  $after" >&2
    exit 2
  fi
  rm -rf "$bak"
}
trap restore EXIT INT TERM HUP

run_bounded() {
  perl -e '
    my ($secs, $log, @cmd) = @ARGV;
    my $pid = fork();
    die "fork: $!" unless defined $pid;
    if (!$pid) {
      setpgrp(0, 0);
      open(STDOUT, ">", $log) or exit 126;
      open(STDERR, ">&STDOUT");
      exec { $cmd[0] } @cmd or exit 127;
    }
    $SIG{ALRM} = sub {
      kill "TERM", -$pid; select(undef, undef, undef, 3); kill "KILL", -$pid;
      waitpid($pid, 0); exit 124;
    };
    alarm $secs;
    waitpid($pid, 0);
    my $st = $?;
    alarm 0;
    exit(($st & 127) ? 128 + ($st & 127) : ($st >> 8));
  ' "$@"
}

classify() {
  local log=$1 code=$2 needle=$3
  if [ "$code" -eq 124 ]; then echo "NOT RUN (timeout)"; return; fi
  if ! grep -q '^test result:' "$log"; then
    echo "NOT RUN (no test result, exit $code)"
    return
  fi
  if grep -q '^test result: ok\. 0 passed' "$log"; then
    echo "NOT RUN (test not found)"
    return
  fi
  if grep -q '^test result: FAILED' "$log"; then
    if grep -F -q -- "$needle" "$log"; then
      echo CAUGHT
    else
      echo "MISSED (failed without the assertion message)"
    fi
    return
  fi
  if grep -q '^test result: ok\. [1-9]' "$log" && [ "$code" -eq 0 ]; then
    echo PASS
    return
  fi
  echo "NOT RUN (exit $code)"
}

logdir=$(mktemp -d /tmp/tessl-guard-log.XXXXXX)
caught=0
missed=0
notrun=0
base_fail=0

note() { echo "guard_break_it: $*"; }

run_cargo() {
  local bin=$1 name=$2 log=$3
  run_bounded "$test_timeout" "$log" \
    cargo test --manifest-path "$root/Cargo.toml" --test "$bin" -- "$name" --exact --test-threads=1 --nocapture
}

planfile=$(mktemp /tmp/tessl-guard-plan.XXXXXX)
py plan "$only" > "$planfile" || die "plan failed"
current_bin=""
current_test=""
current_ok=0

while IFS= read -r line; do
  [ -n "$line" ] || continue
  kind=${line%%$'\t'*}
  rest=${line#*$'\t'}
  if [ "$kind" = baseline ]; then
    current_bin=${rest%%$'\t'*}
    current_test=${rest#*$'\t'}
    blog=$logdir/baseline-${current_bin}-${current_test}.log
    set +e
    run_cargo "$current_bin" "$current_test" "$blog"
    bcode=$?
    set -e
    bver=$(classify "$blog" "$bcode" "___baseline_has_no_needle___")
    if [ "$bver" = PASS ]; then
      current_ok=1
      note "BASELINE $current_bin $current_test PASS"
    else
      current_ok=0
      base_fail=$((base_fail + 1))
      note "BASELINE $current_bin $current_test $bver (log $blog)"
    fi
    after_row=$(shasum -a 256 tests/reductions.rs tests/nn_adversarial.rs)
    if [ "$before" != "$after_row" ]; then
      die "baseline changed a test file"
    fi
    continue
  fi
  if [ "$kind" != row ]; then
    die "bad plan line: $line"
  fi
  id=$rest
  if [ "$current_ok" -ne 1 ]; then
    note "$id NOT RUN (baseline red)"
    notrun=$((notrun + 1))
    continue
  fi
  if ! py apply "$id"; then
    note "$id NOT RUN (old string not unique)"
    notrun=$((notrun + 1))
    restore_mid=$(shasum -a 256 tests/reductions.rs tests/nn_adversarial.rs)
    if [ "$before" != "$restore_mid" ]; then
      cp "$bak/reductions.rs" tests/reductions.rs
      cp "$bak/nn_adversarial.rs" tests/nn_adversarial.rs
    fi
    continue
  fi
  rlog=$logdir/$id.log
  set +e
  run_cargo "$current_bin" "$current_test" "$rlog"
  rcode=$?
  set -e
  needle=$(py list | awk -F '\t' -v id="$id" '$1 == id { print $4; exit }')
  rver=$(classify "$rlog" "$rcode" "$needle")
  cp "$bak/reductions.rs" tests/reductions.rs
  cp "$bak/nn_adversarial.rs" tests/nn_adversarial.rs
  after_row=$(shasum -a 256 tests/reductions.rs tests/nn_adversarial.rs)
  if [ "$before" != "$after_row" ]; then
    die "restore after $id failed"
  fi
  note "$id $rver (exit $rcode, log $rlog)"
  case "$rver" in
    CAUGHT) caught=$((caught + 1)) ;;
    MISSED*) missed=$((missed + 1)) ;;
    *) notrun=$((notrun + 1)) ;;
  esac
done < "$planfile"
rm -f "$planfile"

note "summary caught=$caught missed=$missed not_run=$notrun baseline_failures=$base_fail"
if [ "$missed" -eq 0 ] && [ "$notrun" -eq 0 ] && [ "$base_fail" -eq 0 ] && [ "$caught" -gt 0 ]; then
  exit 0
fi
exit 1
