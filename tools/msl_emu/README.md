# msl_emu — run tessl's Metal kernels on a CPU

A way to execute kernel *source* where there is no Metal toolchain (Linux CI, a
cloud container), so indexing, masking, barrier placement and algebra can be
checked before a kernel ever reaches a GPU.

```sh
pip install -r tools/msl_emu/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python3 tools/msl_emu/check_qwen35.py                   # every case, three threadgroup orders
python3 tools/msl_emu/check_qwen35.py -k chunk,score    # cases whose name contains any of these
python3 tools/msl_emu/check_qwen35.py --fast-math       # Metal-like transcendental error, on-device bounds
python3 tools/msl_emu/check_qwen35_model.py             # a whole Qwen3_5ForCausalLM vs its own logits
python3 tools/msl_emu/dialect_lint.py                   # MSL constructs no compiling tessl kernel uses
MSL_EMU_SANITIZE=address MSL_EMU_OUT=/tmp/asan python3 tools/msl_emu/check_qwen35.py
MSL_EMU_SANITIZE=thread  MSL_EMU_OUT=/tmp/tsan python3 tools/msl_emu/check_qwen35.py -k chunk_T65
```

CI runs all of these on Linux (the `kernel-emulator` job).

`check_qwen35.py` holds each kernel to the transformers *function* it replaces.
`check_qwen35_model.py` holds them to the *model*. It builds a small random
`Qwen3_5ForCausalLM` (Qwen3.5's head_dim 256 and rotary 64, a key head dim of
128, grouped value heads) and runs its forward pass with every Qwen3.5-specific
step on the kernels, using `src/qwen35.rs`'s weight packing and column layouts.
The attention runs on tessl's own `flash_attn_rows`, also emulated. It covers
three flows: prefill, cached decode, and several questions answered from one
shared snapshot, each compared against the model's own logits. A wrong column
offset, packing order, rotary width, norm convention or attention position
moves those logits by O(1). Each of those was injected and caught.

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
- Threadgroups run in grid, reverse or shuffled order (`MSL_EMU_TG_ORDER`), and
  the driver requires all three to agree bit for bit: a GPU promises no order,
  so a kernel whose threadgroups write each other's outputs must not pass on
  the luck of one.
- `MSL_EMU_ULP=N` perturbs each non-`precise::` `exp`/`log`/`rsqrt`/`pow`/`sin`/
  `cos` by a deterministic ±N ulps, a stand-in for Metal's fast math. `exp`
  gets ±(N + floor(2|x|)), following Metal's documented `3 + floor(2|x|)`-ulp
  bound, whose error grows with the argument.
- `MSL_EMU_SANITIZE=address|thread` builds under ASan or TSan. Every device
  buffer and threadgroup allocation is exactly sized, never grown, so an
  overrun can't hide in spare capacity.
- `harness.cpp` launches each kernel with exactly the grid, threadgroup size
  and threadgroup memory that `src/qwen35.rs` uses. `harness --constants`
  prints the kernels' shape constants, and the `host_contract` case holds
  `src/qwen35.rs`'s binds and constants to them.
- `check_qwen35.py` generates inputs, runs the harness, and compares against
  transformers' `modeling_qwen3_5` functions and an f64 recurrence. Outputs are
  pre-filled with NaN, so a skipped element fails.

`dialect_lint.py` is the stand-in for the one thing the emulator cannot see,
the Metal compiler. It lists every function, qualified name, attribute, cast
and language feature the Qwen3.5 kernels use that no other tessl kernel (all
of which compile on every release build) uses. It fails unless each one is in
its reviewed list, with the reason it is standard MSL.

## What a pass does and does not mean

A pass means the kernel's arithmetic, index math, tail masking and
synchronisation structure compute what transformers computes. It does **not**
cover:

- the Metal compiler: MSL dialect errors, address-space mismatches, or
  register pressure;
- the GPU memory model beyond barrier placement, or real scheduling;
- fast-math precision exactly (the ulp-noise mode is a model of it, not a copy,
  and it cannot perturb division, which is an operator);
- the device's threadgroup-memory limit (the kernels `static_assert` their
  budgets against 32 KB, and the host wrappers check the device's own limit).

Running `tests/qwen35_kernels.rs` on a Mac is still the real test. This tool
makes the first on-device run far more likely to be about the device than about
the kernel.
