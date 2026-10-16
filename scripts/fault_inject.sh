#!/bin/bash
# GPU fault injection for tessl's own cooperative/TensorOps GEMM and attention
# kernels (docs/verification.md, "Fault injection").
#
#   scripts/fault_inject.sh [--out DIR] [--only ID[,ID...]]   run the set
#   scripts/fault_inject.sh --list                            print the manifest
#   scripts/fault_inject.sh --check                           do all patches still apply?
#   scripts/fault_inject.sh --self-test                       device-free proof of the driver's rules
#
# Each row of scripts/faults/manifest.tsv names a patch (scripts/faults/<id>.patch,
# a one-hunk edit of a kernel source), the kernel and the fault, and a list of
# `test_binary:test_name` candidates that should go red. For every fault the
# driver applies the patch to the working tree, builds each candidate's test binary
# (the metallib rebuilds because build.rs tracks kernels/), runs EVERY candidate,
# restores the touched files, rebuilds, and re-runs the row's candidates against the
# restored build (the post-restore baseline).
#
# Every build and every test run is bounded (TESSL_FAULT_BUILD_TIMEOUT, default
# 1800 s; TESSL_FAULT_TEST_TIMEOUT, default 300 s; TESSL_FAULT_WHOLE_TIMEOUT,
# default 1200 s for the whole-binary fallback). A timeout is never a catch.
#
# Verdicts. A check that could not run is never a pass:
#   CAUGHT                       a candidate failed on a value assertion (got/want/tolerance/NaN)
#   CAUGHT (non-assert)          a candidate failed, but not on a value assertion: a Metal runtime
#                                or dispatch error, a setup panic, a crash. Counted separately.
#   MISSED                       every candidate ran and passed (see "other" for the whole-binary
#                                run, which names what, if anything, did fail)
#   NOT RUN (patch stale)        `git apply --check` refused the patch
#   NOT RUN (build failed)       the mutated kernel or the test binary did not compile
#   NOT RUN (timeout)            a bounded build or run timed out and nothing else caught it
#   NOT RUN (baseline red or test not found)   no candidate passed on the unmutated tree
#   NOT RUN (post-restore baseline red)        a candidate failed again after the restore, so the
#                                row's evidence is not trusted
# Only CAUGHT counts toward "N of M caught"; non-assert catches and every other row are
# listed by name in the summary.
#
# Safety. The patched files are backed up first and restored by an EXIT/INT/TERM/HUP
# trap, even on failure, and the run ends by comparing every file under kernels/
# with its pre-run hash; an empty or failed hash is a failure, never a match. The
# driver refuses to start when kernels/ has any difference from HEAD, tracked or
# untracked, unless TESSL_FAULT_ALLOW_DIRTY=1 (a peer's uncommitted kernel edit: the
# patch is applied to that content and the same content is restored).
#
# Run it under the machine lock, once:
#   MAC_HEAVY_MAX_LOAD=60 bash <Lappi>/tools/mac_heavy.sh tessl-w10-faults \
#       bash scripts/fault_inject.sh --out <dir>

set -uo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root" || exit 2
manifest=scripts/faults/manifest.tsv
build_timeout=${TESSL_FAULT_BUILD_TIMEOUT:-1800}
test_timeout=${TESSL_FAULT_TEST_TIMEOUT:-300}
whole_timeout=${TESSL_FAULT_WHOLE_TIMEOUT:-1200}

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
die() { echo "fault_inject: $*" >&2; exit 2; }

# ---------------------------------------------------------------------------
# Bounded execution. run_bounded SECS LOG CMD... runs CMD in its own process group
# with stdout+stderr in LOG. Exit status: the command's, 128+signal if it was killed,
# or 124 when SECS elapsed (the group is then TERMed, KILLed 3 s later). Only the
# group this call started is signalled.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Pure helpers (exercised by --self-test).
# ---------------------------------------------------------------------------

# classify_log FILE EXIT -> PASS | FAIL | NOTFOUND | TIMEOUT | CRASH   (a test run's log)
classify_log() {
  local log=$1 code=$2
  if [ "$code" -eq 124 ]; then echo TIMEOUT; return; fi
  if grep -qE '^test result: FAILED|\.\.\. FAILED$' "$log"; then echo FAIL; return; fi
  if grep -qE '^test result: ok\. [1-9][0-9]* passed' "$log" && [ "$code" -eq 0 ]; then echo PASS; return; fi
  if grep -qE '^test result: ok\. 0 passed' "$log"; then echo NOTFOUND; return; fi
  # No recognisable result line: the run did not complete (crash, signal, abort). Not a pass.
  echo CRASH
}

patch_files() { sed -n 's#^+++ b/##p' "$1"; }

# panic_message FILE -> "location: message" of the first panic, or nothing.
panic_message() {
  awk '/panicked at/ { loc=$0; sub(/^.*panicked at /, "", loc); getline n; sub(/^[ \t]+/, "", n); print loc " " n; exit }' "$1"
}

# failure_kind FILE CLASS -> assert | non-assert
# A value assertion compares a number or a flag against an expectation. Everything
# else (an Err unwrapped from a dispatch, a Metal error, a panic before any
# comparison, a crash with no panic at all) is non-assert, and the default.
failure_kind() {
  local log=$1 class=$2 msg
  [ "$class" = CRASH ] && { echo non-assert; return; }
  msg=$(panic_message "$log")
  [ -n "$msg" ] || { echo non-assert; return; }
  if echo "$msg" | grep -qE 'Result::unwrap|Option::unwrap|on an `Err`|command buffer|MTLCommandBuffer|Metal |device lost|failed to|could not|timed out|alloc'; then
    echo non-assert; return
  fi
  if echo "$msg" | grep -qE 'got |want|error budget|non-finite|max err|max abs error|exceeds|worst|left == right|assertion|expected'; then
    echo assert; return
  fi
  echo non-assert
}

kernel_hashes() {
  local out
  out=$(find kernels -type f -print0 | sort -z | xargs -0 shasum -a 256) || return 1
  [ -n "$out" ] || return 1
  printf '%s\n' "$out"
}

# verify_restored PRE POST: fail closed. An empty or missing hash is a failure.
verify_restored() {
  [ -n "${1:-}" ] && [ -n "${2:-}" ] && [ "$1" = "$2" ]
}

dirty_kernels() { [ -n "$(git status --porcelain --untracked-files=all -- kernels)" ]; }

# ---------------------------------------------------------------------------
# Backup / restore, with a verification the restore really happened.
# ---------------------------------------------------------------------------
bakdir=""
marker=""
declare -a touched=()

backup_files() {
  local f
  for f in "$@"; do
    mkdir -p "$bakdir/$(dirname "$f")"
    cp -p "$f" "$bakdir/$f" || die "cannot back up $f"
    touched+=("$f")
  done
}

restore_files() {
  local f rc=0
  for f in "${touched[@]:-}"; do
    [ -n "$f" ] || continue
    if [ -f "$bakdir/$f" ]; then
      # Not cp -p: the restored file must look *newer* than the build output, or cargo
      # (which compares mtimes) keeps the mutated metallib it built from the patch.
      cp "$bakdir/$f" "$f" || rc=1
      cmp -s "$bakdir/$f" "$f" || { echo "fault_inject: RESTORE FAILED for $f" >&2; rc=1; }
    fi
  done
  touched=()
  return $rc
}

# apply_fault ID -> 0 applied; 1 stale (nothing modified)
apply_fault() {
  local id=$1 p=scripts/faults/$1.patch
  [ -f "$p" ] || return 1
  git apply --check "$p" 2>/dev/null || return 1
  # shellcheck disable=SC2046
  backup_files $(patch_files "$p")
  if ! git apply "$p" 2>/dev/null; then restore_files; return 1; fi
  return 0
}

cleanup() {
  local rc=0
  restore_files || rc=1
  [ -n "$bakdir" ] && rm -rf "$bakdir"
  [ -n "$marker" ] && rm -f "$marker"
  return $rc
}

install_traps() {
  trap 'cleanup' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM HUP
}

# ---------------------------------------------------------------------------
# Self-test: device-free, no cargo. Proves the rules the table relies on.
# ---------------------------------------------------------------------------
self_test() {
  local fails=0 t
  t=$(mktemp -d) || die "mktemp"
  bakdir=$(mktemp -d)
  marker=$(mktemp)
  install_traps
  trap 'cleanup; rm -rf "$t"' EXIT

  ok() { echo "self-test ok:   $1"; }
  bad() { echo "self-test FAIL: $1" >&2; fails=$((fails + 1)); }

  local before after f
  before=$(kernel_hashes)

  # 1. A stale patch is "patch stale", never applied, and leaves kernels/ untouched.
  if apply_fault selftest-stale; then
    bad "the deliberately stale patch applied"
    restore_files
  else
    ok "a stale patch is refused (verdict: NOT RUN (patch stale))"
  fi
  after=$(kernel_hashes)
  [ "$before" = "$after" ] && ok "a refused patch left kernels/ untouched" || bad "kernels/ changed after a refused patch"

  # 2. A real patch applies, changes the file, and the restore undoes it and bumps its mtime.
  local first
  first=$(grep -v '^#' "$manifest" | head -n1 | cut -d'|' -f1)
  sleep 1
  touch "$marker"
  sleep 1
  if apply_fault "$first"; then
    after=$(kernel_hashes)
    [ "$before" != "$after" ] && ok "applying $first changes a kernel" || bad "applying $first changed nothing"
    restore_files
    after=$(kernel_hashes)
    [ "$before" = "$after" ] && ok "restore_files returns kernels/ to its pre-run hash" || bad "restore left a difference"
    for f in $(patch_files "scripts/faults/$first.patch"); do
      if [ "$f" -nt "$marker" ]; then
        ok "restore bumps $f mtime past the build output (cargo rebuilds the clean kernel)"
      else
        bad "restored $f kept an old mtime: cargo would reuse the mutated metallib"
      fi
    done
  else
    bad "the first manifest patch ($first) does not apply to this tree"
  fi

  # 3. Log classification: a run that did not finish, found no test or timed out is not a pass.
  printf 'running 1 test\ntest a ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored\n' > "$t/pass"
  printf 'running 1 test\ntest a ... FAILED\n\ntest result: FAILED. 0 passed; 1 failed\n' > "$t/fail"
  printf 'running 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 filtered out\n' > "$t/none"
  printf 'Killed: 9\n' > "$t/killed"
  [ "$(classify_log "$t/pass" 0)" = PASS ] && ok "classify: pass" || bad "classify: pass"
  [ "$(classify_log "$t/fail" 101)" = FAIL ] && ok "classify: fail" || bad "classify: fail"
  [ "$(classify_log "$t/none" 0)" = NOTFOUND ] && ok "classify: a name that matches no test is NOT FOUND, not a pass" || bad "classify: none"
  [ "$(classify_log "$t/killed" 137)" = CRASH ] && ok "classify: a killed run is a crash, not a pass" || bad "classify: killed"
  [ "$(classify_log "$t/pass" 1)" != PASS ] && ok "classify: ok text with a nonzero exit is not a pass" || bad "classify: exit code ignored"
  [ "$(classify_log "$t/pass" 124)" = TIMEOUT ] && ok "classify: exit 124 is a timeout even with a pass line" || bad "classify: timeout"

  # 4. Failure kinds: only a value assertion is CAUGHT; everything else is non-assert.
  printf "thread 't' panicked at tests/x.rs:9:5:\nbf16 NN 4x4: element [3] is 9.0x its error budget (got 0, want 1)\n" > "$t/p1"
  printf "thread 't' panicked at tests/x.rs:9:5:\ncalled \`Result::unwrap()\` on an \`Err\` value: \"command buffer failed\"\n" > "$t/p2"
  printf "thread 't' panicked at tests/x.rs:9:5:\nsomething unrelated happened\n" > "$t/p3"
  printf 'test a ... FAILED\n' > "$t/p4"
  [ "$(failure_kind "$t/p1" FAIL)" = assert ] && ok "failure kind: a got/want message is a value assertion" || bad "failure kind: assert"
  [ "$(failure_kind "$t/p2" FAIL)" = non-assert ] && ok "failure kind: an unwrapped dispatch Err is non-assert" || bad "failure kind: unwrap"
  [ "$(failure_kind "$t/p3" FAIL)" = non-assert ] && ok "failure kind: an unrecognised panic defaults to non-assert" || bad "failure kind: default"
  [ "$(failure_kind "$t/p4" FAIL)" = non-assert ] && ok "failure kind: a failure with no panic line is non-assert" || bad "failure kind: no panic"
  [ "$(failure_kind "$t/p1" CRASH)" = non-assert ] && ok "failure kind: a crash is non-assert" || bad "failure kind: crash"

  # 5. Bounded execution: a command that outlives its bound is killed and reports 124;
  # the command's own status and output pass through otherwise.
  local t0 t1 rc
  t0=$(date +%s)
  run_bounded 1 "$t/to.log" sleep 30
  rc=$?
  t1=$(date +%s)
  if [ "$rc" -eq 124 ] && [ $((t1 - t0)) -lt 15 ]; then ok "a run past its bound is killed and exits 124 (after $((t1 - t0)) s)"; else bad "timeout: rc=$rc after $((t1 - t0)) s"; fi
  run_bounded 20 "$t/rc.log" bash -c 'echo hello; exit 3'
  rc=$?
  if [ "$rc" -eq 3 ] && grep -q hello "$t/rc.log"; then ok "run_bounded passes the exit status and output through"; else bad "run_bounded: rc=$rc"; fi

  # 6. The end-of-run hash check fails closed.
  if verify_restored "" ""; then bad "two empty hashes matched"; else ok "verify_restored: two empty hashes are a failure, not a match"; fi
  if verify_restored "a" ""; then bad "an empty post hash matched"; else ok "verify_restored: an empty post-run hash is a failure"; fi
  if verify_restored "a" "b"; then bad "different hashes matched"; else ok "verify_restored: different hashes are a failure"; fi
  if verify_restored "a" "a"; then ok "verify_restored: identical non-empty hashes pass"; else bad "identical hashes failed"; fi
  if (cd "$t" && kernel_hashes) > /dev/null 2>&1; then bad "kernel_hashes succeeded with no kernels/ directory"; else ok "kernel_hashes fails when there is nothing to hash"; fi

  # 7. An untracked file under kernels/ makes the tree dirty.
  : > kernels/.selftest_untracked
  if dirty_kernels; then ok "an untracked file in kernels/ counts as dirty"; else bad "untracked file ignored by the dirty check"; fi
  rm -f kernels/.selftest_untracked

  # 8. Manifest rows are well formed and every patch they name exists.
  local id kf kernel fault cands n=0
  while IFS='|' read -r id kf kernel fault cands; do
    case $id in ''|'#'*) continue ;; esac
    n=$((n + 1))
    [ -f "scripts/faults/$id.patch" ] || bad "$id: patch missing"
    [ -n "$kernel" ] && [ -n "$fault" ] && [ -n "$cands" ] || bad "$id: incomplete row"
  done < "$manifest"
  if [ "$n" -ge 12 ]; then ok "manifest has $n faults"; else bad "manifest has only $n faults"; fi

  if [ "$fails" -ne 0 ]; then echo "self-test: $fails failure(s)" >&2; exit 1; fi
  echo "self-test: all checks passed"
  exit 0
}

# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
mode=run; out=""; only=""
while [ $# -gt 0 ]; do
  case $1 in
    --self-test) mode=self ;;
    --list) mode=list ;;
    --check) mode=check ;;
    --out) out=${2:?--out needs a directory}; shift ;;
    --only) only=${2:?--only needs ids}; shift ;;
    *) die "unknown argument $1" ;;
  esac
  shift
done

[ -f "$manifest" ] || die "missing $manifest"

case $mode in
  self) self_test ;;
  list) grep -v '^#' "$manifest" | awk -F'|' 'NF { printf "%-40s %s\n", $1, $3 }'; exit 0 ;;
  check)
    stale=0
    while IFS='|' read -r id kf kernel fault cands; do
      case $id in ''|'#'*) continue ;; esac
      if git apply --check "scripts/faults/$id.patch" 2>/dev/null; then echo "applies  $id"
      else echo "STALE    $id"; stale=$((stale + 1)); fi
    done < "$manifest"
    [ "$stale" -eq 0 ] || { echo "$stale stale patch(es)" >&2; exit 1; }
    exit 0 ;;
esac

# ---- run ----
if [ -z "$out" ]; then out="target/fault-inject/$(date -u +%Y%m%dT%H%M%SZ)"; fi
mkdir -p "$out/logs" "$out/baseline" || die "cannot create $out"
bakdir=$(mktemp -d)
install_traps

if [ "${TESSL_FAULT_ALLOW_DIRTY:-0}" != 1 ] && dirty_kernels; then
  die "kernels/ differs from HEAD (tracked or untracked); refusing to run (TESSL_FAULT_ALLOW_DIRTY=1 to apply on top of it)"
fi
pre_hash=$(kernel_hashes) || die "cannot hash kernels/; refusing to run"
# A metallib left by an earlier run may carry a fault (an old-mtime restore, a killed run):
# make every kernel newer than the build output so the baseline is built from clean sources.
find kernels -type f -exec touch {} +

lock_dir=${MAC_HEAVY_LOCK:-/Users/bharath/Code/research/Lappi-decision/build/mac-heavy.lock.d}
{
  echo "date        $(stamp)"
  echo "commit      $(git rev-parse HEAD)  (branch $(git rev-parse --abbrev-ref HEAD))"
  echo "machine     $(sysctl -n machdep.cpu.brand_string 2>/dev/null), $(sysctl -n hw.model 2>/dev/null), $(sw_vers -productVersion 2>/dev/null)"
  echo "load (idle check at start)  $(sysctl -n vm.loadavg 2>/dev/null)"
  echo "lock holder $(if [ -n "${MAC_HEAVY_LABEL:-}" ]; then echo "this run (label $MAC_HEAVY_LABEL)"; else echo "none (driver not run under mac_heavy.sh)"; fi); lock dir present: $([ -d "$lock_dir" ] && echo yes || echo no)"
  echo "bounds      build ${build_timeout}s, test ${test_timeout}s, whole-binary ${whole_timeout}s"
} | tee "$out/env.txt"

want() { [ -z "$only" ] && return 0; case ",$only," in *",$1,"*) return 0 ;; esac; return 1; }
safe() { echo "$1" | tr -c 'A-Za-z0-9_\n' _; }

# build_bin BIN LOGPREFIX -> 0 built (path in $exe), 1 failed, 124 timed out
build_bin() {
  local bin=$1 log=$2.build.log rc
  run_bounded "$build_timeout" "$log" cargo test --release --test "$bin" --no-run
  rc=$?
  [ "$rc" -eq 0 ] || return "$rc"
  exe=$(sed -n 's/^ *Executable .*(\(.*\))$/\1/p' "$log" | tail -n1)
  [ -n "$exe" ] && [ -x "$exe" ] || return 1
}

# run_cand BIN NAME LOGPREFIX -> echoes PASS|FAIL|NOTFOUND|TIMEOUT|CRASH|BUILD|BUILDTIMEOUT
run_cand() {
  local bin=$1 name=$2 pre=$3 rc
  build_bin "$bin" "$pre"
  rc=$?
  if [ "$rc" -eq 124 ]; then echo BUILDTIMEOUT; return; fi
  if [ "$rc" -ne 0 ]; then echo BUILD; return; fi
  CARGO_MANIFEST_DIR=$root run_bounded "$test_timeout" "$pre.log" "$exe" --exact "$name" --test-threads=1
  rc=$?
  classify_log "$pre.log" "$rc"
}

# ---- baseline: every distinct candidate must pass on the unmutated tree ----
echo "== baseline (unmutated) =="
while IFS='|' read -r id kf kernel fault cands; do
  case $id in ''|'#'*) continue ;; esac
  want "$id" || continue
  for c in $cands; do
    f="$out/baseline/$(safe "$c")"
    [ -f "$f" ] && continue
    r=$(run_cand "${c%%:*}" "${c#*:}" "$out/logs/baseline-$(safe "$c")")
    echo "$r" > "$f"
    echo "baseline $r  $c"
  done
done < "$manifest"

printf 'id\tverdict\tfailure_kind\tkernel\tfault\tcatching_test\tobserved_failure\tall_catching_tests\tother_failing_tests\n' > "$out/results.tsv"
n_assert=0; n_nonassert=0; total=0; declare -a notcaught=()

while IFS='|' read -r id kf kernel fault cands; do
  case $id in ''|'#'*) continue ;; esac
  want "$id" || continue
  total=$((total + 1))
  verdict=""; kind="-"; catching="-"; observed="-"; allc=""; other="-"

  if ! apply_fault "$id"; then
    verdict="NOT RUN (patch stale)"
  else
    usable=0; timed_out=0; build_failed=0; assert_hit=0; nonassert_hit=0
    for c in $cands; do
      bin=${c%%:*}; name=${c#*:}
      [ "$(cat "$out/baseline/$(safe "$c")")" = PASS ] || continue
      usable=1
      pre="$out/logs/$id--$(safe "$c")"
      r=$(run_cand "$bin" "$name" "$pre")
      case $r in
        FAIL|CRASH)
          k=$(failure_kind "$pre.log" "$r")
          allc="$allc $c($k)"
          msg=$(panic_message "$pre.log")
          [ -n "$msg" ] || msg="no panic line (crash or abort, see $pre.log)"
          if [ "$k" = assert ]; then
            if [ "$assert_hit" -eq 0 ]; then catching=$c; observed=$msg; fi
            assert_hit=1
          else
            if [ "$assert_hit" -eq 0 ] && [ "$nonassert_hit" -eq 0 ]; then catching=$c; observed=$msg; fi
            nonassert_hit=1
          fi ;;
        BUILD) build_failed=1; observed="build failed, see $pre.build.log" ;;
        BUILDTIMEOUT|TIMEOUT) timed_out=1; observed="timeout, see $pre.log" ;;
        NOTFOUND|PASS) ;;
      esac
    done
    if [ "$assert_hit" -eq 1 ]; then verdict=CAUGHT; kind=assert
    elif [ "$nonassert_hit" -eq 1 ]; then verdict="CAUGHT (non-assert)"; kind=non-assert
    elif [ "$build_failed" -eq 1 ]; then verdict="NOT RUN (build failed)"
    elif [ "$timed_out" -eq 1 ]; then verdict="NOT RUN (timeout)"
    elif [ "$usable" -eq 0 ]; then verdict="NOT RUN (baseline red or test not found)"
    else verdict=MISSED; fi

    if [ "$verdict" = MISSED ]; then
      # Whole-binary run of each candidate binary, to name anything that does go red.
      failing=""
      for b in $(for c in $cands; do echo "${c%%:*}"; done | sort -u); do
        wlog="$out/logs/$id--whole-$b.log"
        if build_bin "$b" "$out/logs/$id--whole-$b"; then
          CARGO_MANIFEST_DIR=$root run_bounded "$whole_timeout" "$wlog" "$exe" --test-threads=1
          wrc=$?
          [ "$wrc" -eq 124 ] && failing="$failing $b:TIMEOUT"
          failing="$failing $(sed -n 's/^test \(.*\) \.\.\. FAILED$/'"$b"':\1/p' "$wlog" | tr '\n' ' ')"
        else
          failing="$failing $b:BUILD-FAILED"
        fi
      done
      other=$(echo "$failing" | xargs)
      [ -n "$other" ] || other="none"
    fi

    restore_files || { echo "FATAL: restore failed after $id" >&2; exit 3; }

    # Post-restore baseline: the restored kernel, rebuilt, must pass this row's candidates again.
    post_bad=""
    for c in $cands; do
      [ "$(cat "$out/baseline/$(safe "$c")")" = PASS ] || continue
      r=$(run_cand "${c%%:*}" "${c#*:}" "$out/logs/$id--post-restore-$(safe "$c")")
      [ "$r" = PASS ] || post_bad="$post_bad $c=$r"
    done
    if [ -n "$post_bad" ]; then
      echo "[$id] post-restore baseline RED:$post_bad" >&2
      verdict="NOT RUN (post-restore baseline red)"; kind="-"
      other="post-restore:$post_bad"
    fi
  fi

  echo "[$id] $verdict  ${catching}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$verdict" "$kind" "$kernel" "$fault" "$catching" "$observed" "$(echo "$allc" | xargs)" "$other" >> "$out/results.tsv"
  case $verdict in
    CAUGHT) n_assert=$((n_assert + 1)) ;;
    "CAUGHT (non-assert)") n_nonassert=$((n_nonassert + 1)); notcaught+=("$id: $verdict") ;;
    *) notcaught+=("$id: $verdict") ;;
  esac
done < "$manifest"

post_hash=$(kernel_hashes) || post_hash=""
if verify_restored "$pre_hash" "$post_hash"; then restored="yes"; else restored="NO (empty, failed or different hash)"; fi
{
  echo "kernels/ byte-identical to pre-run: $restored"
  echo "$n_assert of $total caught by a value assertion ($(date -u +%Y-%m-%d)); $n_nonassert more caught non-assert"
  for n in "${notcaught[@]:-}"; do [ -n "$n" ] && echo "  not counted: $n"; done
} | tee "$out/summary.txt"
[ "$restored" = yes ] || exit 3
[ "$n_assert" -eq "$total" ]
