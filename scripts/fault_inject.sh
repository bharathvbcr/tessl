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
# a one-hunk edit of a kernel source), the kernel and the fault, and an ordered list
# of `test_binary:test_name` candidates that should go red. For every fault the
# driver applies the patch to the working tree, runs the candidates with
# `cargo test --release --test BIN -- --exact NAME --test-threads=1` (the metallib
# rebuilds because build.rs tracks kernels/), then restores the touched files.
#
# Verdicts. A check that could not run is never a pass:
#   CAUGHT                       a candidate test failed on the mutated kernel
#   MISSED                       every candidate ran and passed (see "other" for the
#                                whole-binary run, which names what, if anything, did fail)
#   NOT RUN (patch stale)        `git apply --check` refused the patch
#   NOT RUN (build failed)       the mutated kernel or the test binary did not compile
#   NOT RUN (test not found)     a candidate name matched no test (0 passed, 0 failed)
#   NOT RUN (baseline red or test not found)   no candidate passed on the unmutated tree
# Only CAUGHT counts toward "N of M caught"; the summary names every other row.
#
# Safety. The patched files are backed up first and restored by an EXIT/INT/TERM/HUP
# trap, even on failure, and the run ends by comparing every file under kernels/
# with its pre-run hash. A mutated kernel is never committed: the driver refuses to
# start when kernels/ already differs from HEAD, unless TESSL_FAULT_ALLOW_DIRTY=1
# (a peer's uncommitted kernel edit: the patch is applied to that content and the
# same content is restored).
#
# Run it under the machine lock, once:
#   MAC_HEAVY_MAX_LOAD=60 bash <Lappi>/tools/mac_heavy.sh tessl-w10-faults \
#       bash scripts/fault_inject.sh --out <dir>

set -uo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root" || exit 2
manifest=scripts/faults/manifest.tsv

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
die() { echo "fault_inject: $*" >&2; exit 2; }

# ---------------------------------------------------------------------------
# Pure helpers (exercised by --self-test).
# ---------------------------------------------------------------------------

# classify_log FILE EXIT -> PASS | FAIL | BUILD | NOTFOUND
classify_log() {
  local log=$1 code=$2
  if grep -qE 'error: could not compile|error: failed to run custom build command|error\[E[0-9]+\]|error: .*metal' "$log"; then
    echo BUILD; return
  fi
  if grep -qE '^test result: FAILED|\.\.\. FAILED$' "$log"; then
    echo FAIL; return
  fi
  if grep -qE '^test result: ok\. [1-9][0-9]* passed' "$log" && [ "$code" -eq 0 ]; then
    echo PASS; return
  fi
  if grep -qE '^test result: ok\. 0 passed' "$log"; then
    echo NOTFOUND; return
  fi
  # No recognisable result line: the run did not complete. Not a pass.
  echo BUILD
}

patch_files() { sed -n 's#^+++ b/##p' "$1"; }

# First assertion line of a failing log, for the table.
first_failure() {
  awk '/panicked at/ { getline n; sub(/^[ \t]+/, "", n); print n; exit }' "$1"
}

# ---------------------------------------------------------------------------
# Backup / restore, with a verification the restore really happened.
# ---------------------------------------------------------------------------
bakdir=""
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

kernel_hashes() { find kernels -type f -print0 | sort -z | xargs -0 shasum -a 256; }

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
  return $rc
}

# ---------------------------------------------------------------------------
# Self-test: device-free, no cargo. Proves the rules the table relies on.
# ---------------------------------------------------------------------------
self_test() {
  local fails=0 t
  t=$(mktemp -d) || die "mktemp"
  bakdir=$(mktemp -d)
  trap 'cleanup; rm -rf "$t"' EXIT

  ok() { echo "self-test ok:   $1"; }
  bad() { echo "self-test FAIL: $1" >&2; fails=$((fails + 1)); }

  local before after
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

  # 2. A real patch applies, changes the file, and the trap-path restore undoes it.
  local first
  first=$(sed -n '1p' "$manifest" | cut -d'|' -f1)
  first=$(grep -v '^#' "$manifest" | head -n1 | cut -d'|' -f1)
  marker=$(mktemp); sleep 1
  if apply_fault "$first"; then
    after=$(kernel_hashes)
    [ "$before" != "$after" ] && ok "applying $first changes a kernel" || bad "applying $first changed nothing"
    restore_files
    after=$(kernel_hashes)
    [ "$before" = "$after" ] && ok "restore_files returns kernels/ to its pre-run hash" || bad "restore left a difference"
    local f
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

  # 3. Log classification: a run that did not finish, or found no test, is not a pass.
  printf 'running 1 test\ntest a ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored\n' > "$t/pass"
  printf 'running 1 test\ntest a ... FAILED\n\ntest result: FAILED. 0 passed; 1 failed\n' > "$t/fail"
  printf 'running 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 filtered out\n' > "$t/none"
  printf 'error: could not compile `tessl` (lib) due to 1 previous error\n' > "$t/build"
  printf 'Killed: 9\n' > "$t/killed"
  [ "$(classify_log "$t/pass" 0)" = PASS ] && ok "classify: pass" || bad "classify: pass"
  [ "$(classify_log "$t/fail" 101)" = FAIL ] && ok "classify: fail" || bad "classify: fail"
  [ "$(classify_log "$t/none" 0)" = NOTFOUND ] && ok "classify: a name that matches no test is NOT FOUND, not a pass" || bad "classify: none"
  [ "$(classify_log "$t/build" 101)" = BUILD ] && ok "classify: build failure" || bad "classify: build"
  [ "$(classify_log "$t/killed" 137)" = BUILD ] && ok "classify: a killed run is not a pass" || bad "classify: killed"
  [ "$(classify_log "$t/pass" 1)" != PASS ] && ok "classify: ok text with a nonzero exit is not a pass" || bad "classify: exit code ignored"

  # 4. Manifest rows are well formed and every patch they name exists.
  local id kf kernel fault cands n=0
  while IFS='|' read -r id kf kernel fault cands; do
    case $id in ''|'#'*) continue ;; esac
    n=$((n + 1))
    [ -f "scripts/faults/$id.patch" ] || bad "$id: patch missing"
    [ -n "$kernel" ] && [ -n "$fault" ] && [ -n "$cands" ] || bad "$id: incomplete row"
  done < "$manifest"
  [ "$n" -ge 12 ] && ok "manifest has $n faults" || bad "manifest has only $n faults"

  if [ "$fails" -ne 0 ]; then echo "self-test: $fails failure(s)" >&2; exit 1; fi
  echo "self-test: all checks passed"
  exit 0
}

# ---------------------------------------------------------------------------
# Main run.
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
mkdir -p "$out/logs" || die "cannot create $out"
bakdir=$(mktemp -d)
trap 'cleanup' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

if [ "${TESSL_FAULT_ALLOW_DIRTY:-0}" != 1 ] && ! git diff --quiet HEAD -- kernels; then
  die "kernels/ differs from HEAD; refusing to run (TESSL_FAULT_ALLOW_DIRTY=1 to apply on top of it)"
fi
pre_hash=$(kernel_hashes)
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
} | tee "$out/env.txt"

want() { [ -z "$only" ] && return 0; case ",$only," in *",$1,"*) return 0 ;; esac; return 1; }

# run_cand BIN NAME LOGFILE -> echoes classification
run_cand() {
  local bin=$1 name=$2 log=$3 code
  cargo test --release --test "$bin" -- --exact "$name" --test-threads=1 > "$log" 2>&1
  code=$?
  classify_log "$log" "$code"
}

# Baseline: every distinct candidate must pass on the unmutated tree.
bl() { printf "%s" "$out/baseline/$(echo "$1" | tr -c "A-Za-z0-9_\n" _)"; }
mkdir -p "$out/baseline"
echo "== baseline (unmutated) =="
while IFS='|' read -r id kf kernel fault cands; do
  case $id in ''|'#'*) continue ;; esac
  want "$id" || continue
  for c in $cands; do
    [ -f "$(bl "$c")" ] && continue
    r=$(run_cand "${c%%:*}" "${c#*:}" "$out/logs/baseline-${c//[:\/]/_}.log")
    echo "$r" > "$(bl "$c")"
    echo "baseline $r  $c"
  done
done < "$manifest"

printf 'id\tverdict\tkernel\tfault\tcatching_test\tobserved_failure\tother_failing_tests\n' > "$out/results.tsv"
caught=0; total=0; declare -a notcaught=()

while IFS='|' read -r id kf kernel fault cands; do
  case $id in ''|'#'*) continue ;; esac
  want "$id" || continue
  total=$((total + 1))
  verdict=""; catching="-"; observed="-"; other="-"

  if ! apply_fault "$id"; then
    verdict="NOT RUN (patch stale)"
  else
    usable=0
    for c in $cands; do
      bin=${c%%:*}; name=${c#*:}
      if [ "$(cat "$(bl "$c")")" != PASS ]; then continue; fi
      usable=1
      log="$out/logs/$id--${c//[:\/]/_}.log"
      r=$(run_cand "$bin" "$name" "$log")
      if [ "$r" = FAIL ]; then verdict=CAUGHT; catching=$c; observed=$(first_failure "$log"); break; fi
      if [ "$r" = BUILD ]; then verdict="NOT RUN (build failed)"; observed="see $log"; break; fi
      if [ "$r" = NOTFOUND ]; then verdict="NOT RUN (test not found)"; catching=$c; break; fi
    done
    if [ -z "$verdict" ]; then
      if [ "$usable" -eq 0 ]; then
        verdict="NOT RUN (baseline red or test not found)"
      else
        verdict=MISSED
        # Whole-binary run of each candidate binary, to name anything that does go red.
        failing=""
        for b in $(for c in $cands; do echo "${c%%:*}"; done | sort -u); do
          wlog="$out/logs/$id--whole-$b.log"
          cargo test --release --test "$b" -- --test-threads=1 > "$wlog" 2>&1
          failing="$failing $(sed -n 's/^test \(.*\) \.\.\. FAILED$/'"$b"':\1/p' "$wlog" | tr '\n' ' ')"
        done
        other=$(echo "$failing" | xargs)
        [ -n "$other" ] || other="none"
      fi
    fi
    restore_files || { echo "FATAL: restore failed after $id" >&2; exit 3; }
  fi

  echo "[$id] $verdict  ${catching}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$verdict" "$kernel" "$fault" "$catching" "$observed" "$other" >> "$out/results.tsv"
  if [ "$verdict" = CAUGHT ]; then caught=$((caught + 1)); else notcaught+=("$id: $verdict"); fi
done < "$manifest"

post_hash=$(kernel_hashes)
if [ "$pre_hash" = "$post_hash" ]; then restored="yes"; else restored="NO"; fi
{
  echo "kernels/ byte-identical to pre-run: $restored"
  echo "$caught of $total caught ($(date -u +%Y-%m-%d))"
  for n in "${notcaught[@]:-}"; do [ -n "$n" ] && echo "  not caught: $n"; done
} | tee "$out/summary.txt"
[ "$restored" = yes ] || exit 3
[ "$caught" -eq "$total" ]
