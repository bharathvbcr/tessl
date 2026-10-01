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
    model.adamw_step(lr=schedule(step), weight_decay=0.1)  # Trainer's exclusions take none
```

`adamw_step` is `torch.optim.AdamW`'s update (checked against it to 1e-6);
`weight_decay` is a float for every parameter transformers' Trainer decays
(not the norms or `linear_attn.dt_bias`), or a dict giving each name its own.
`adamw_step_count` is torch's `state["step"]`.

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
The model runs entirely in f32, the tied embedding included, so a write is
exact and the parameters torch holds are the ones tessl differentiates.
Every read and write
is a GPU copy of the whole model (about 8 GB of f32 each way on the 2B), and
torch holds its own copy of the parameters and gradients beside tessl's.

## How the buffers cross

An MPS tensor's `untyped_storage().data_ptr()` is its `id<MTLBuffer>`, the
same pointer ATen's `getMTLBufferStorage` bit-casts, and
`storage_offset() * element_size()` is the byte offset into it. The binding
checks each view against the storage's size, because the MTLBuffer is larger
(torch rounds allocations up) and tessl can only see the buffer.

torch's stream and tessl's queue are separate. Each call runs
`torch.mps.synchronize()` before tessl reads anything and returns only after
tessl's queue has finished, so the two never touch a buffer at once. That is
correct, not fast: two GPU waits per call. Sharing an `MTLSharedEvent`
instead would remove them.

## Build and test

```bash
cargo build --release
```

```bash
python3 -m unittest discover -s python/tests -v
```

`TESSL_LIB` overrides where `libtessl.dylib` is loaded from. The library
reads its kernels from the metallib path baked in when it was built, so keep
that build's `target/` directory. The tests compare against
`torch.nn.functional.cross_entropy` in float64 on the CPU, transformers'
GDN fallback, and (`test_qwen35.py`) transformers' own `Qwen3_5ForCausalLM`
autograd on the committed tiny fixture, before and after an AdamW step
written back through `load_parameters`. Build the
reference with `.cpu().double()`: in torch 2.13, `.to("cpu", torch.float64)`
on a bf16 MPS tensor returns wrong values without an error.
