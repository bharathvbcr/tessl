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
`torch.nn.functional.cross_entropy` in float64 on the CPU. Build the
reference with `.cpu().double()`: in torch 2.13, `.to("cpu", torch.float64)`
on a bf16 MPS tensor returns wrong values without an error.
