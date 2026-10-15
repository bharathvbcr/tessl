"""The two waits of a tessl_torch call, measured one at a time (a measurement
script: it prints ``METRIC name value`` lines for ``bench_paired --format
metrics`` and ships no behaviour).

    python bench/capi_torch_overhead.py [--async]

``torch_mps_sync_idle_us`` is ``torch.mps.synchronize()`` on an idle queue,
the wait before tessl reads a torch buffer. ``tessl_sync_idle_us`` is
``tessl_synchronize`` on an idle runtime, the wait every successful call ends
with. ``ce_call_us`` is one whole ``cross_entropy_rows`` call, both waits and
the kernels in between (batched or per-kernel, by ``--async``).
``ce_call_us`` is the median of 200 calls; the idle waits the median of 1000.
"""

import ctypes
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import torch  # noqa: E402
import tessl_torch  # noqa: E402


def median_us(fn, calls):
    fn()
    samples = []
    for _ in range(calls):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples)


def main():
    use_async = "--async" in sys.argv[1:]
    unknown = [a for a in sys.argv[1:] if a != "--async"]
    if unknown:
        sys.exit(f"unknown arguments {unknown}")
    if not torch.backends.mps.is_available():
        sys.exit("capi_torch_overhead: MPS is not available")
    rt = tessl_torch._runtime()
    err = ctypes.create_string_buffer(tessl_torch._ERR_LEN)
    status = rt.lib.tessl_runtime_set_async_encode(rt.handle, int(use_async), err, tessl_torch._ERR_LEN)
    if status:
        sys.exit(f"set_async_encode: {err.value.decode()}")

    n, h, v = 256, 128, 1024
    g = torch.Generator().manual_seed(0)
    hidden = torch.randn(n, h, generator=g).to("mps")
    weight = torch.randn(v, h, generator=g).to("mps")
    targets = torch.randint(0, v, (n,), generator=g).to("mps")

    torch_idle = median_us(torch.mps.synchronize, 1000)

    def tessl_sync():
        e = ctypes.create_string_buffer(tessl_torch._ERR_LEN)
        s = rt.lib.tessl_synchronize(rt.handle, e, tessl_torch._ERR_LEN)
        if s:
            raise RuntimeError(e.value.decode())

    tessl_idle = median_us(tessl_sync, 1000)

    def call():
        float(tessl_torch.cross_entropy(hidden, weight, targets))

    whole = median_us(call, 200)
    print(f"METRIC torch_mps_sync_idle_us {torch_idle:.3f}")
    print(f"METRIC tessl_sync_idle_us {tessl_idle:.3f}")
    print(f"METRIC ce_call_us {whole:.3f}")


if __name__ == "__main__":
    main()
