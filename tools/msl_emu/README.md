# msl_emu — run tessl's Metal kernels on a CPU

A way to execute kernel *source* where there is no Metal toolchain (Linux CI, a
cloud container), so indexing, masking, barrier placement and algebra can be
checked before a kernel ever reaches a GPU.

```sh
python3 tools/msl_emu/check_qwen35.py          # needs torch + transformers
python3 tools/msl_emu/check_qwen35.py -k chunk # a subset
```

## How it works

- `build.sh` strips `[[attribute]]` annotations from the `.metal` sources (C++
  has no meaning for them) and compiles them unmodified otherwise, as C++20,
  against `metal_stdlib` in this directory instead of Apple's.
- `metal_stdlib` supplies the MSL surface the kernels use. A threadgroup is
  real OS threads. `threadgroup_barrier` is a real barrier. Each `simd_*`
  collective is a real exchange through shared slots behind a 32-lane barrier.
  `simdgroup_float8x8` is held whole by each lane, and a store writes only the
  lane's own two elements, so a missing `simdgroup_barrier` shows up as a race
  here too. Threadgroup memory starts as NaN, so an unwritten read is visible.
- `harness.cpp` launches each kernel with exactly the grid, threadgroup size
  and threadgroup memory that `src/qwen35.rs` uses.
- `check_qwen35.py` generates inputs, runs the harness, and compares against
  transformers' `modeling_qwen3_5` functions and an f64 recurrence. Outputs are
  pre-filled with NaN, so a skipped element fails.

## What a pass does and does not mean

A pass means the kernel's arithmetic, index math, tail masking and
synchronisation structure compute what transformers computes. It does **not**
cover:

- the Metal compiler: MSL dialect errors, address-space mismatches, or
  register pressure;
- the GPU memory model beyond barrier placement, or real scheduling;
- fast-math precision (the emulator uses IEEE `exp`, `pow` and friends);
- threadgroup memory limits (the host wrappers check those).

Running `tests/qwen35_kernels.rs` on a Mac is still the real test. This tool
makes the first on-device run far more likely to be about the device than about
the kernel.
