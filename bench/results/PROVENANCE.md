# Imported results

These 29 files were not measured on this tree. They came from the MLSystemsLab
nested copy of this crate (`Rust_MLKit/crates/tessl`): commit `6c62ab7` overlaid
that snapshot, and `ff0d954` merged its "nested-only files" into main. They
describe that copy's code, which differs from this tree (for example
`SPLITK_MIN_K` and `splitk_gate_opens_at_the_measured_depth` appear nowhere in
`src/`, `tests/` or `kernels/` here). Treat every number in them as unverified
for this tree until the owning bench is re-run here. JSON files cannot carry a
comment, so the label lives in this index.

The list is the set of files under `bench/results/` that `ff0d954` added
relative to its first parent (`ff0d954^1`):

- `arg_table_latch_m5pro.txt`
- `attn_dispatch_split_m5pro.json`
- `attn_parity_m5pro.json`
- `attn_speed_decode_kernel_m5pro.json`
- `attn_speed_decode_kernel_m5pro_b.json`
- `attn_speed_decode_kernel_m5pro_c.json`
- `attn_speed_decode_m5pro.json`
- `attn_speed_m5pro.json`
- `attn_speed_routed_m5pro.json`
- `attn_speed_routed_m5pro_b.json`
- `attn_speed_routed_m5pro_c.json`
- `attn_speed_rows_m5pro.json`
- `attn_tune_decode_kernel_m5pro.json`
- `attn_tune_decode_r_kernel_m5pro.json`
- `attn_tune_decode_sgs_m5pro.json`
- `attn_tune_reduce_w_m5pro.json`
- `attn_tune_rows_g_m5pro.json`
- `attn_tune_rows_m5pro.json`
- `bf16_smallm_coop_m5pro.txt`
- `coop_clear_m5pro.txt`
- `gemm_parity_grid_m5pro.json`
- `gemm_speed_bf16_confirm_m5pro.json`
- `gemm_speed_ladder_m5pro.json`
- `gemm_speed_ladder_m5pro_b.json`
- `gemv_q4_simd_m5pro.txt`
- `kernel_coverage_m5pro.json`
- `rope_simdgroup_m5pro.txt`
- `row_reduction_simd_m5pro.txt`
- `splitk_gate_m5pro.txt`

`bf16_smallm_coop_m5pro.txt` and `splitk_gate_m5pro.txt` also carry the label
in their own header, and were re-measured on this tree's kernels in
`bf16_smallm_coop_reconciled_m5pro.txt` and `splitk_par_bf16_m5pro.txt`.
