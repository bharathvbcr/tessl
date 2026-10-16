#!/bin/bash
# One lane of bench/results/attn_tiled_rebench_2026-10-15_m5pro.json, run B.
#
#   attn_lane.sh native BIN T : BIN is a 70ae3bc build of bench_qwen35_layers (before --attn-tiles
#                               existed): its `--paired-attn T` PAIRED attn_prefill median_ms, printed
#                               as the METRIC line the new `--attn-tiles` prints for the default tile
#   attn_lane.sh tiles  BIN T : BIN --attn-tiles T, only the default tile's METRIC line
#
# bench_paired (`--format metrics`) needs both lanes to print the same METRIC name; the 70ae3bc binary
# has no --attn-tiles, so this wrapper renames its one comparable number.
#
# The exact command lines of runs A to D (the run order and the flags of
# bench/results/attn_tiled_rebench_2026-10-15_m5pro.json). R is the directory holding the worktrees,
# AFTER the gp/w9-attn harness commit, OV a detached worktree at the same commit with
# kernels/qwen35_attn_tiled.metal replaced by its 70ae3bc content, N70 a detached worktree at 70ae3bc,
# every one built with `cargo build --release --bin ...` and its own CARGO_TARGET_DIR; BP is
# $AFTER/target/release/bench_paired and SNAP an EmbeddingGemma 2 snapshot directory (only its
# config.json is read). Each run is under the machine lock (tessl-w9-attn).
#
#   run() { name=$1; rounds=$2; shift 2
#     "$BP" --format metrics --clean-env --rounds "$rounds" --lane-timeout 3000 --out "$OUT/$name.json" "$@"; }
#   A: run A_tiles 12 \
#        "before_70ae3bc_kernel=$OV/target/release/bench_qwen35_layers --attn-tiles 2048" \
#        "after_main=$AFTER/target/release/bench_qwen35_layers --attn-tiles 2048"
#   B: run B_default_tile 12 \
#        "native_70ae3bc=bash bench/attn_lane.sh native $N70/target/release/bench_qwen35_layers 2048" \
#        "overlay_70ae3bc_kernel=bash bench/attn_lane.sh tiles $OV/target/release/bench_qwen35_layers 2048" \
#        "after_main=bash bench/attn_lane.sh tiles $AFTER/target/release/bench_qwen35_layers 2048"
#   C: run C_train_lse 8 \
#        "before_70ae3bc_kernel=$OV/target/release/bench_qwen35_train 2048 --op=attn_train_forward" \
#        "after_main=$AFTER/target/release/bench_qwen35_train 2048 --op=attn_train_forward"
#   D: run D_encoder_kernel 12 \
#        "rows:EMBEDGEMMA2_SNAPSHOT=$SNAP=$AFTER/target/release/bench_embedgemma2 --attn-kernel=rows" \
#        "tiled:EMBEDGEMMA2_SNAPSHOT=$SNAP=$AFTER/target/release/bench_embedgemma2 --attn-kernel=tiled"
#
# (Run B's driver was this script without the usage check and `pipefail` (added since, so a failing
# bench cannot hide behind awk's exit status), at the scratch path recorded in the result's lane argv. The
# metric names in A to D are those of the harness commit; since the rework, bench_qwen35_train's is
# op_attn_train_forward_f32_ms_t2048 and bench_embedgemma2's tiled lane compares itself with the
# rows kernel before timing.)
set -euo pipefail
[ $# -eq 3 ] || { echo "usage: attn_lane.sh native|tiles BIN T" >&2; exit 2; }
mode=$1; bin=$2; t=$3
case "$mode" in
  native)
    "$bin" --paired-attn "$t" | awk -v t="$t" '/^PAIRED t=/ && /kernel=attn_prefill/ { for (i=1;i<=NF;i++) if ($i ~ /^median_ms=/) { sub("median_ms=","",$i); printf "METRIC attn_prefill_q32_k32_sg4_ms_t%s %s\n", t, $i } }'
    ;;
  tiles)
    "$bin" --attn-tiles "$t" | awk '/^METRIC attn_prefill_q32_k32_sg4_/'
    ;;
  *) echo "bad mode" >&2; exit 2;;
esac
