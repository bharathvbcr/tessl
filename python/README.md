# tessl_torch

torch bindings for tessl's Metal kernels, loaded with `ctypes` from the C ABI
in `src/capi.rs` (`libtessl.dylib`). Nothing beyond torch and the standard
library is needed.

```python
import sys; sys.path.insert(0, "python")
import tessl_torch

# hidden: [B, T, H] on mps (f32 or bf16), weight: [V, H] (the tied embedding)
loss = tessl_torch.cross_entropy(hidden[:, :-1], weight, targets, mask)
loss.backward()
```

`cross_entropy` is `F.cross_entropy(hidden[mask] @ weight.T, targets[mask])`
without the logits: tessl gathers the supervised rows and walks the
vocabulary in chunks (`chunk=`, default 4096 columns), so memory is bounded
by the supervised rows, not by `T * V`. The weight gradient is accumulated in
f32 and cast to the weight's dtype once. Slices along the leading dimensions
(such as `hidden[:, :-1]`) and tensors at a storage offset are read in place;
other strided layouts are copied contiguous first. An empty mask is an error.

## Training a whole Qwen3.5 model

`Qwen35` runs tessl's `Qwen35Model::train_step`: the forward, the causal-LM
loss and every parameter's gradient happen in tessl. AdamW can run in tessl
too, on the model's own parameters, which is what fits the 2B on a 64 GB Mac
(params, gradients and both moments: 32 GB plus the step's scratch):

```python
model = tessl_torch.Qwen35("model.safetensors", "config.json")  # prefix "model.language_model."
model.adamw_init()                    # both moments, inside tessl
for step, ids in enumerate(batches):  # one sequence of token ids per step
    loss = model.train_step(ids, operands="bf16")
    coef = tessl_torch.clip_coef(model.grad_sq_norm() ** 0.5, max_norm=1.0)
    model.adamw_step(lr=schedule(step), weight_decay=0.1, grad_scale=coef)  # Trainer's exclusions take none
```

`adamw_step` is `torch.optim.AdamW`'s update (checked against it to 1e-6);
`weight_decay` is a float for every parameter transformers' Trainer decays
(not the norms or `linear_attn.dt_bias`), or a dict giving each name its own.
`adamw_step_count` is torch's `state["step"]`. `adamw_state()` returns the step
and both moments by name (torch's `exp_avg` and `exp_avg_sq`); saved with
`parameters()` and restored with `load_parameters()`, `adamw_init()` and
`load_adamw_state()`, the run resumes bit for bit. `grad_scale=clip_coef(...)`
is `clip_grad_norm_` before the step (checked against it); with a head of
your own, add its gradients' squares to `grad_sq_norm()` and scale them by
the same coefficient.

`precision="bf16"` keeps the matrices, their gradients and each layer's
saved input in bf16 (arithmetic, GDN state, softmax and log-sum-exp stay
f32), and `adamw_init` then names how the bf16 weights take their updates
and how the moments are stored. That is what fits the 4B's training in a
48 GiB working set (`docs/qwen35.md`, "Training on bf16 storage"):

```python
model = tessl_torch.Qwen35("model.safetensors", "config.json", precision="bf16")
model.adamw_init(update="bf16-kahan", moments="block8")  # or "f32-master", or "bf16-stochastic" with seed=
model.describe()  # "... matrices bf16 ...; AdamW update=bf16-kahan moments=block8/256 step=0"
```

Its steps take `operands="bf16"`. `adamw_state()` then also records the
configuration (`"config"`) and the f32 masters or Kahan compensations
(`"aux"`), and `load_adamw_state` refuses a checkpoint made under another
configuration; a run resumes bit for bit under every rule.

A padded batch with a loss on chosen positions, and a head of your own on
the final hidden states, runs row by row: each row trimmed to its length
(right padding is never attended, so this is exact), its gradients added
into tessl's with `accumulate=True`. A batch mean over `N` supervised
positions is `scale=1/N` on every row:

```python
n_letter = sum(1 for r in rows if not r.is_span)
n_span_decisions = 2 * sum(1 for r in rows if r.is_span)  # a start and an end per span row
for i, r in enumerate(rows):
    ids = r.tokens[: r.length]
    if r.is_span:                       # scored by your head, not tessl's
        model.train_forward(ids, positions=[], targets=[], operands="bf16")
        at = [r.query] + r.candidates
        h = model.hidden(at).requires_grad_(True)
        loss = head_loss(h[0], h[1:]) / n_span_decisions
        loss.backward()                 # your head's gradients land in torch
        model.train_backward(dh=h.grad, positions=at, accumulate=i > 0)
    else:                               # the tied LM head, in tessl
        model.train_forward(ids, positions=[r.target_index], targets=[r.answer],
                            scale=1.0 / n_letter, operands="bf16")
        model.train_backward(accumulate=i > 0)
```

`train_forward` returns its loss (the sum over its positions, unscaled).
`train_discard()` drops a step whose backward will not run; while a step is
pending, another step, the gradients and AdamW are refused.

With torch's own optimizer instead, which holds its own copy of every
parameter and gradient (about 55 GB on the 2B):

```python
params = model.parameters()           # an f32 copy on MPS, transformers' names and shapes
opt = torch.optim.AdamW(params.values(), lr=1e-5)
grads = None
for ids in batches:                   # one sequence of token ids per step
    loss = model.train_step(ids, operands="bf16")  # tessl: the loss and every gradient
    grads = model.grads(into=grads)   # the same tensors every step after the first
    for name, g in grads.items():
        params[name].grad = g
    opt.step()
    model.load_parameters(params)     # write the update back to tessl
```

`operands="bf16"` (also on `cross_entropy`) rounds every GEMM's operands to bf16 and accumulates in f32; the weights,
activations and gradients stay f32. The default, `"f32"`, is exact. On the
2B's cross-entropy at T = 2048 bf16 is 3.15x faster; its gradients differ
from transformers' f32 ones by up to 3.6e-2 of a parameter's peak on the 2B
(the exact step: 3.9e-3; `docs/qwen35.md`, "A training step"). Every
parity bound this project states is for the exact default.

`grads(into=...)` writes into tensors an earlier `grads()` returned instead
of allocating another 8 GB copy on the 2B while the previous one is still
attached to the parameters.

Names are transformers' below the text tower (`layers.3.mlp.gate_proj.weight`),
and values are the parameters' own (the zero-centred norms as `w`, as tessl
stores them). Linear weights come back as transposed views, because
tessl keeps them as `[in, out]`. `load_parameters` needs every parameter,
checks them all before writing any, and accepts any layout, dtype or device.
An f32 model runs entirely in f32, the tied embedding included, so a write
is exact and the parameters torch holds are the ones tessl differentiates; a
bf16 model rounds the matrices it is given to nearest.
`parameters()` and
`load_parameters()` are GPU copies of the whole model (about 8 GB of f32 each
way on the 2B), and torch holds its own copy of the parameters and gradients
beside tessl's. To read or set one parameter, use `read_parameter(name)` and
`write_parameter(name, value)`: they move only that parameter's bytes, on the
host (`tessl_qwen35_read_parameter_host` / `write_parameter_host`, which are
`Qwen35Model::read_parameter_host` / `write_parameter_host`), with no GPU copy
and no wait on torch's queue. `read_parameter` returns a CPU tensor unless you
pass `device=`; `write_parameter` is refused while a step is pending, and
refused for a bf16 matrix whose AdamW state keeps auxiliary state
(`update="f32-master"` or `"bf16-kahan"`): the next `adamw_step` rebuilds the
weight from the master or the compensation and would overwrite the write, and
there is no per-parameter write of that state. Restore the parameters and the
state together (`load_parameters` then `load_adamw_state`, whose `"aux"` carries
it), or write before `adamw_init`; `load_parameters` has the same caveat (the
state must be restored with it for the write to stick).

## Handle ownership, closing and errors

A `Qwen35` (and the per-thread runtime behind every call) owns a C handle
that tessl frees exactly once; the C ABI keeps no registry of live handles, so
a second free or a call on a freed handle is undefined behaviour there, not an
error. The binding therefore:

- refuses `copy.copy`, `copy.deepcopy` and pickling (`TypeError`): a copy
  would own the same pointer and free it twice;
- gives `Qwen35` an idempotent `close()` (also `with tessl_torch.Qwen35(...) as
  model:`, and called when the object is dropped). After it the object is
  `closed` and every use raises `TesslError` without reaching the library.
  A model's weights and optimizer state are released at `close()`, not at the
  garbage collector's whim;
- uses a handle from the thread that made it. Another thread's call is
  refused (`TesslError`), and a `close()` from another thread leaks the handle
  rather than drop a thread-affine runtime there;
- raises `TesslPanic` (a subclass of `TesslError`) when the library reports a
  bug in tessl caught at its boundary (status 2). The object that made the
  call is closed, and a panic in a runtime call (cross-entropy, GDN) drops that
  thread's runtime so the next call builds a new one. An ordinary refusal
  (status 1) stays a plain `TesslError` and the object stays usable;
- range-checks every integer headed for a C field (`chunk` below 2**32, the
  AdamW `step` and `seed` below 2**64, tensor dimensions) and raises
  `TesslError` naming it, before any tensor is touched. ctypes would wrap
  `-1` or `2**32` silently.
  Integer arguments must be integers (no floats, no `bool`; `accumulate`
  takes a bool or 0/1), and id tensors must have an integer dtype.
- reports a failed free: `close()` raises `TesslPanic` if the free hit a bug
  in tessl and `TesslError` if called from another thread (the handle is then
  leaked); either way the object is closed. A finalizer issues a
  `ResourceWarning` instead of raising.

A runtime whose GPU work failed is "poisoned": it latches and refuses every
later call (status 1, "recreate it"). That is not a status code of its own, so
the binding asks the library (`tessl_runtime_poisoned`, ABI 11) after any
failed call. If the runtime is poisoned it closes the runtime and every
`Qwen35` loaded on it, drops the thread's cached runtime, and raises
`TesslPoisoned` (a subclass of `TesslError`) naming the original error. The
next call on that thread builds a fresh runtime; a model must be loaded again.

## How the buffers cross

An MPS tensor's `untyped_storage().data_ptr()` is its `id<MTLBuffer>`, the
same pointer ATen's `getMTLBufferStorage` bit-casts, and
`storage_offset() * element_size()` is the byte offset into it. The binding
checks each view against the storage's size, because the MTLBuffer is larger
(torch rounds allocations up) and tessl can only see the buffer.

torch's stream and tessl's queue are separate, and each side's wait is
there for a reason:

- **torch to tessl.** A call that gives tessl a torch buffer (cross-entropy,
  GDN, `hidden`, `copy`, and `train_backward` with a `dh`) first runs
  `torch.mps.synchronize()`, once, so tessl never reads a buffer torch is
  still writing. Calls that pass only host arrays (`train_step`,
  `train_forward`, `adamw_step`, `train_backward` without `dh`) do not wait on
  torch at all. This stays a CPU wait: a shared-event handoff would need
  torch to signal an `MTLSharedEvent` tessl then waits on, and torch 2.14
  has no way to do that (`torch.mps.Event` owns an event Python cannot
  share, and nothing encodes a wait or signal for an external one). Signalling
  tessl's own event from torch is not an option either; tessl's lifetime
  decisions read a private event, and `GpuRuntime::shared_event()` is the
  separate handoff event, so a foreign signal cannot make tessl free memory
  early.
- **tessl to torch.** Every call returns only after tessl's queue is idle, on
  failure as well as on success. The runtime batches a call's kernels
  (async encode, switched on per runtime by
  `tessl_runtime_set_async_encode`, ABI 12) and waits **once**, where it used
  to wait after every kernel. The wait on the failure path is a memory-safety
  requirement: when a call raises, Python drops its temporaries and torch's
  MPS allocator may hand their storage to someone else, so tessl work still
  in flight for the refused call would write into memory that is no longer
  the caller's. A refused call therefore finishes what it already submitted
  before the error is raised.
- **A poisoned runtime is the exception.** It refuses the wait, so a call that
  poisons it can return with tessl work still running. The binding handles
  that: it frees the poisoned runtime (`TesslPoisoned`; the free waits for the
  GPU, bounded by the library's event-wait budget, 30 s by default) before the
  exception reaches your code, while the failed call's tensors are still held
  by the frames it unwinds through, so torch cannot recycle their storage
  first. A caller of the C ABI has no such guarantee: after
  `tessl_runtime_poisoned` reports 1 it must keep every external buffer it
  passed alive until the runtime is freed. A wait that merely times out on a
  healthy runtime is reported in the error message, and work may still be in
  flight then.

The library's own default is synchronous encode (one waited commit per
kernel); `tessl_torch` turns async encode on for the runtimes it creates.
Per-call and per-step costs of both settings are in
`bench/results/capi_*`.

## Build and test

```bash
cargo build --release
```

```bash
python3 -m unittest discover -s python/tests -v
```

`python/tests/test_binding_host.py` needs no GPU (a recording stand-in
replaces the library): `python3 -m unittest python/tests/test_binding_host.py`.

Where they run: `scripts/ci_local.sh` runs `test_binding_host.py` on every
invocation with the torch oracle python (and fails if there is none). It runs
the rest of `python/tests` (`test_cross_entropy.py`, `test_gdn.py`,
`test_qwen35.py`: GPU and MPS, transformers) only under the machine-wide heavy
lock (`MAC_HEAVY_LABEL` set by `tools/mac_heavy.sh`); otherwise it prints a
`SKIPPED` line. GitHub's CI does not run `python/tests`.

`TESSL_LIB` overrides where `libtessl.dylib` is loaded from. The library
reads its kernels from the metallib path baked in when it was built, so keep
that build's `target/` directory. The tests compare against
`torch.nn.functional.cross_entropy` in float64 on the CPU, transformers'
GDN fallback, and (`test_qwen35.py`) transformers' own `Qwen3_5ForCausalLM`
autograd on the committed tiny fixture, before and after an AdamW step
written back through `load_parameters`. Build the
reference with `.cpu().double()`: in torch 2.13, `.to("cpu", torch.float64)`
on a bf16 MPS tensor returns wrong values without an error.
