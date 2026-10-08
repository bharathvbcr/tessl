#!/usr/bin/env bash
# Interleaved min-of-N for `bench_qwen35_train --step` across configurations.
#
#   QWEN35_2B_SAFETENSORS=... bench/paired_qwen35_step.sh ROUNDS OUT CONFIG...
#
# A CONFIG is `T:MID:MODE[:EXTRA]`: T tokens, MID the TESSL_MID_COMMIT value
# (`-` leaves it unset), MODE `sync` or `async` (`--async`), and EXTRA an
# optional bench flag (e.g. `--bf16`). Each round runs every configuration in
# its own process under `/usr/bin/time -l` (TESSL_MID_COMMIT is read once
# per process), forward order on even rounds and reversed on odd ones, so
# drift lands on every configuration alike (docs/benchmarking.md, pitfalls 1
# and 3). Each process times the median of 3 steps; the summary is the min
# of those medians over the rounds, with the max beside it as the spread,
# and the largest peak memory footprint and device peak seen. OUT gets every
# process's full output. BENCH_BIN overrides the binary: a frozen build of an
# earlier commit, for a before/after pair.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: QWEN35_2B_SAFETENSORS=... $0 ROUNDS OUT CONFIG..." >&2
  exit 2
fi
rounds=$1
out=$2
shift 2
configs=("$@")
bin=${BENCH_BIN:-target/release/bench_qwen35_train}
[[ -x $bin ]] || { echo "$bin is missing: cargo build --release --bins first" >&2; exit 2; }
: "${QWEN35_2B_SAFETENSORS:?set QWEN35_2B_SAFETENSORS to the Qwen3.5-2B .safetensors}"

: >"$out"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

run_one() {
  local cfg=$1 round=$2
  local t mid mode extra
  IFS=: read -r t mid mode extra <<<"$cfg"
  local args=("$t" --step-only "--step=$t")
  [[ $mode == async ]] && args+=(--async)
  [[ -n ${extra:-} ]] && args+=("$extra")
  local log="$tmp/$(echo "$cfg" | tr ':/' '__').$round"
  {
    echo "# ---- round $round, config $cfg: ${args[*]} (TESSL_MID_COMMIT=$mid) ----"
    if [[ $mid == - ]]; then
      env -u TESSL_MID_COMMIT -u METAL_RUNTIME_MID_COMMIT /usr/bin/time -l "$bin" "${args[@]}" 2>&1
    else
      env -u METAL_RUNTIME_MID_COMMIT TESSL_MID_COMMIT="$mid" /usr/bin/time -l "$bin" "${args[@]}" 2>&1
    fi
  } >"$log"
  cat "$log" >>"$out"
  local secs peak dev
  secs=$(sed -n 's/^train_step, T = [0-9]*: \([0-9.]*\) s.*/\1/p' "$log")
  peak=$(awk '/peak memory footprint/ {print $1}' "$log")
  dev=$(sed -n 's/^  device peak over the steps: \([0-9.]*\) GiB.*/\1/p' "$log")
  if [[ -z $secs || -z $peak || -z $dev ]]; then
    echo "config $cfg round $round produced no timing; see $out" >&2
    exit 1
  fi
  echo "$cfg $secs $peak $dev" >>"$tmp/samples"
}

for ((r = 0; r < rounds; r++)); do
  if ((r % 2 == 0)); then
    order=("${configs[@]}")
  else
    order=()
    for ((i = ${#configs[@]} - 1; i >= 0; i--)); do order+=("${configs[i]}"); done
  fi
  for cfg in "${order[@]}"; do
    run_one "$cfg" "$r"
    echo "round $r: $cfg done" >&2
  done
done

{
  echo
  echo "# ---- summary: $rounds rounds, each a median of 3 steps; min (max) over rounds ----"
  printf '%-28s %10s %10s %16s %15s\n' config "min s" "max s" "peak footprint" "device peak"
  for cfg in "${configs[@]}"; do
    awk -v c="$cfg" '$1 == c {
        if (n == 0 || $2 < lo) lo = $2
        if (n == 0 || $2 > hi) hi = $2
        if ($3 > pk) pk = $3
        if ($4 > dv) dv = $4
        n++
      }
      END { printf "%-28s %10.3f %10.3f %13.2f GB %11.2f GiB\n", c, lo, hi, pk / 1e9, dv }' "$tmp/samples"
  done
} | tee -a "$out"
