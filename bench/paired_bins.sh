#!/usr/bin/env bash
# Interleaved min-of-N across frozen binaries, for any bench that prints
# `METRIC name value` lines (bench_nn_kernels, bench_qwen35_layers --decode).
#
#   BENCH_BINS="before=PATH after=PATH" bench/paired_bins.sh ROUNDS OUT ARGS...
#
# Every round runs each binary once with ARGS in its own process, forward
# order on even rounds and reversed on odd ones, so drift lands on every
# binary alike (docs/benchmarking.md, pitfalls 1 and 3). Each process reports
# its own medians; the summary is the min of those over the rounds, with the
# max beside it as the spread, and — with exactly two binaries — the ratio of
# the second's min to the first's. OUT gets every process's full output.
# Lower is better for every metric these benches print.
set -euo pipefail

if [[ $# -lt 2 || -z ${BENCH_BINS:-} ]]; then
  echo "usage: BENCH_BINS=\"before=PATH after=PATH\" $0 ROUNDS OUT ARGS..." >&2
  exit 2
fi
rounds=$1
out=$2
shift 2
declare -a labels=() paths=()
for pair in $BENCH_BINS; do
  labels+=("${pair%%=*}")
  paths+=("${pair#*=}")
done
for p in "${paths[@]}"; do
  [[ -x $p ]] || { echo "$p is missing or not executable" >&2; exit 2; }
done

: >"$out"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
: >"$tmp/samples"

n=${#paths[@]}
for ((r = 0; r < rounds; r++)); do
  for ((i = 0; i < n; i++)); do
    if ((r % 2 == 0)); then b=$i; else b=$((n - 1 - i)); fi
    log="$tmp/run.$r.$b"
    {
      echo "# ---- round $r, ${labels[b]}: ${paths[b]} $* ----"
      "${paths[b]}" "$@" 2>&1
    } >"$log"
    cat "$log" >>"$out"
    if ! grep -q '^METRIC ' "$log"; then
      echo "${labels[b]} round $r printed no METRIC lines; see $out" >&2
      exit 1
    fi
    awk -v l="${labels[b]}" '/^METRIC / { print l, $2, $3 }' "$log" >>"$tmp/samples"
    echo "round $r: ${labels[b]} done" >&2
  done
done

{
  echo
  echo "# ---- summary: $rounds rounds; min (max) over rounds of each process's median ----"
  awk -v labels="${labels[*]}" '
    {
      k = $2 SUBSEP $1
      if (!(k in lo) || $3 < lo[k]) lo[k] = $3
      if (!(k in hi) || $3 > hi[k]) hi[k] = $3
      if (!($2 in seen)) { seen[$2] = 1; order[++m] = $2 }
    }
    END {
      nl = split(labels, L, " ")
      printf "%-44s", "metric"
      for (i = 1; i <= nl; i++) printf " %22s", L[i] " min (max)"
      if (nl == 2) printf " %9s", L[2] "/" L[1]
      printf "\n"
      for (j = 1; j <= m; j++) {
        name = order[j]
        printf "%-44s", name
        for (i = 1; i <= nl; i++) {
          k = name SUBSEP L[i]
          if (k in lo) printf " %10.3f (%9.3f)", lo[k], hi[k]
          else printf " %22s", "-"
        }
        if (nl == 2) {
          a = name SUBSEP L[1]; b = name SUBSEP L[2]
          if ((a in lo) && (b in lo) && lo[a] > 0) printf " %9.3f", lo[b] / lo[a]
        }
        printf "\n"
      }
    }' "$tmp/samples"
} | tee -a "$out"
