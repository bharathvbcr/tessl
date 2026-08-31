//! Runtime construction, buffer recycling, the bump arena, and a long
//! unsynchronized dispatch chain.
//!
//! A single GEMM proves almost nothing about the runtime: the pool never gets
//! a chance to hand a buffer back, the constant arena never advances past its
//! first few slots, and nothing is ever dropped while the GPU still holds a
//! reference to it. The interesting failures live in the second lap -- a
//! recycled buffer reused before the command buffer that reads it has
//! completed, a bump cursor reset out from under a live view, a constant slot
//! reused across two dispatches in the same encoder. So the work here is sized
//! to reach the second lap and then checked for the answer, not just for a
//! clean return code.

mod common;

use std::collections::HashSet;

use common::{assert_within_bound, random_f32, reference, tensor_f32, with_gpu, Layout};
use tessl::gemm::select_backend;
use tessl::tensor::gpu_copy;
use tessl::{gemm_f32, softcap_f32, BufferKind, DType, GemmBackend, GpuRuntime, Tensor};

/// Identity of the underlying `MTLBuffer`, which is how pool reuse is observed
/// from outside the crate: the freelist hands back the same object, not a copy.
fn buffer_id(buf: &tessl::GpuBuffer) -> usize {
    buf.metal() as *const _ as *const () as usize
}

#[test]
fn runtime_reports_a_usable_device_and_budget() {
    with_gpu(|rt| {
        assert!(!rt.device_name().is_empty(), "device has no name");
        assert!(
            std::path::Path::new(tessl::metallib_path()).exists(),
            "metallib_path() points at {} which does not exist",
            tessl::metallib_path()
        );

        // `select_backend` is what downstream crates call instead of hardcoding
        // a backend, so its answer has to track the metallib actually loaded --
        // claiming TensorOps on a build without those kernels would fail later
        // at pipeline lookup, far from the cause.
        let expected = if rt.has_tensorops() {
            GemmBackend::TensorOps
        } else {
            GemmBackend::Simdgroup
        };
        assert_eq!(select_backend(rt), expected);

        let info = rt.memory_info();
        assert!(info.recommended_working_set > 0, "no working set probed");
        assert!(
            info.wired_budget > 0 && info.wired_budget <= info.recommended_working_set,
            "wired budget {} is not inside the working set {}",
            info.wired_budget,
            info.recommended_working_set
        );
        assert!(info.pool_cache_cap > 0, "pool cache starts disabled");

        // `set_wired_fraction` clamps to [0.5, 0.95]; an unclamped 2.0 would
        // hand the caller a budget larger than the device reported.
        rt.set_wired_fraction(2.0);
        assert!(rt.memory_info().wired_budget <= info.recommended_working_set);
    });
}

#[test]
fn buffer_kind_survives_the_round_trip_to_the_holder() {
    with_gpu(|rt| {
        // Whoever ends up holding a buffer needs to know its recycling policy,
        // because Cold storage is reclaimed after the command buffer completes
        // and Hot storage is not. The kind is set at allocation and read back
        // somewhere else entirely, so the two have to agree.
        assert_eq!(rt.alloc_buffer(4096).unwrap().kind(), BufferKind::Cold);
        assert_eq!(rt.alloc_buffer_hot(4096).unwrap().kind(), BufferKind::Hot);
        assert_eq!(
            rt.alloc_buffer_kind(4096, BufferKind::Hot).unwrap().kind(),
            BufferKind::Hot
        );
        assert_eq!(
            rt.alloc_tensor_f32_hot(&[1024]).unwrap().buffer.kind(),
            BufferKind::Hot
        );
        assert_eq!(
            rt.alloc_tensor_f32(&[1024]).unwrap().buffer.kind(),
            BufferKind::Cold
        );
        rt.ensure_bump(1 << 16).unwrap();
        assert_eq!(
            rt.bump_alloc_f32(&[64]).unwrap().buffer.kind(),
            BufferKind::Bump
        );
    });
}

#[test]
fn cold_buffers_come_back_from_the_freelist_after_a_sync() {
    with_gpu(|rt| {
        // Pool reuse is observed through *contents*, not through the address of
        // the MTLBuffer. Releasing a buffer and immediately asking for the same
        // size hands back the same address whether or not tessl pooled it --
        // the system allocator reuses it either way -- so pointer identity
        // proves nothing. `alloc_buffer` does not zero, while a buffer Metal
        // has just created is zero-filled, which makes a sentinel written
        // before the drop a decisive signal: it survives a freelist round trip
        // and does not survive a real deallocation.
        const BYTES: usize = 96 * 1024;
        const SENTINEL: f32 = 1.0316e-9;

        let held: Vec<_> = (0..4).map(|_| rt.alloc_buffer(BYTES).unwrap()).collect();
        let ids: HashSet<usize> = held.iter().map(buffer_id).collect();
        assert_eq!(ids.len(), held.len(), "concurrently held buffers alias");
        drop(held);

        {
            let marked = rt.alloc_buffer(BYTES).unwrap();
            marked.write_f32_prefix(&[SENTINEL; 8]);
            // Dropping only queues the recycle; it lands after GPU work
            // completes, which is the point -- reclaiming earlier would hand a
            // live buffer to the next dispatch.
        }
        rt.synchronize().unwrap();
        let recycled = rt.alloc_buffer(BYTES).unwrap();
        assert_eq!(
            recycled.read_f32()[0],
            SENTINEL,
            "same-sized reallocation did not come from the freelist"
        );
        drop(recycled);
        rt.synchronize().unwrap();

        // With the cache disabled the recycled storage is released rather than
        // parked, so the sentinel must not survive the same round trip.
        rt.set_pool_cache_cap_bytes(0);
        assert_eq!(rt.memory_info().pool_cache_cap, 0);
        {
            let marked = rt.alloc_buffer(BYTES).unwrap();
            marked.write_f32_prefix(&[SENTINEL; 8]);
        }
        rt.synchronize().unwrap();
        let fresh = rt.alloc_buffer(BYTES).unwrap();
        assert_ne!(
            fresh.read_f32()[0],
            SENTINEL,
            "pool kept recycling with the cache cap set to zero"
        );
    });
}

#[test]
fn the_pool_keeps_serving_correct_results_under_churn() {
    with_gpu(|rt| {
        // Bucketing rounds each request to a power of two, so requests of very
        // different sizes share buckets and a recycled buffer can be handed to
        // a tensor with a different shape than the one that freed it. The
        // observable that matters is not which object comes back but that the
        // GEMM into it is still right.
        let (m, n, k) = (33, 45, 17);
        let a_host = random_f32(m * k, 21);
        let b_host = random_f32(k * n, 22);
        let expect = reference(Layout::Nn, &a_host, &b_host, m, n, k);
        let a = tensor_f32(rt, &[m, k], &a_host);
        let b = tensor_f32(rt, &[k, n], &b_host);

        for round in 0..64 {
            // Sizes that land in the same bucket as [m, n] some rounds and not
            // others, so the freelist is genuinely churning rather than cycling
            // one buffer.
            let filler = rt.alloc_tensor_f32(&[(round % 7) * 200 + 64]).unwrap();
            filler.buffer.write_f32(&vec![9.0; filler.numel()]);
            drop(filler);

            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            gemm_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            assert_within_bound(
                &format!("pool churn round {round}"),
                &c.buffer.read_f32(),
                &expect,
                k,
                0.0,
            );
        }
    });
}

#[test]
fn bump_arena_hands_out_zeroed_slices_and_reports_exhaustion() {
    with_gpu(|rt| {
        // Before `ensure_bump` there is no slab, and the error has to say so
        // rather than silently falling back to a pool allocation -- callers use
        // the bump path precisely because they do not want one.
        assert_eq!(
            rt.bump_alloc_f32(&[4]).map(|_| ()).unwrap_err(),
            "bump arena not initialized; call ensure_bump first"
        );
        assert!(!rt.bump_enabled());

        rt.ensure_bump(4096).unwrap();
        assert!(rt.bump_enabled());

        // Sub-allocations must arrive zeroed even though the slab is recycled
        // storage; a caller writing only part of a temp would otherwise read a
        // previous step's values out of the untouched remainder.
        let mut views = Vec::new();
        for i in 0..8 {
            let t = rt.bump_alloc_f32(&[64]).unwrap();
            assert!(
                t.buffer.read_f32()[t.byte_offset / 4..][..64]
                    .iter()
                    .all(|&x| x == 0.0),
                "bump slice {i} was not zeroed"
            );
            t.buffer.write_f32_prefix(&[i as f32 + 1.0]);
            views.push(t);
        }

        // Exhaustion is a returned error, not a panic or a silent overrun into
        // the next view's window.
        let err = rt.bump_alloc_f32(&[4096]).map(|_| ()).unwrap_err();
        assert!(
            err.starts_with("bump arena exhausted:"),
            "unexpected exhaustion message: {err}"
        );

        // Resetting with views still outstanding must move to a fresh slab
        // rather than alias them; the previously handed-out windows keep their
        // contents.
        let marks: Vec<f32> = views
            .iter()
            .map(|t| t.buffer.read_f32()[t.byte_offset / 4])
            .collect();
        rt.bump_reset();
        let after_reset = rt.bump_alloc_f32(&[512]).unwrap();
        after_reset
            .buffer
            .write_f32_prefix(&vec![-1.0f32; after_reset.numel()]);
        for (i, t) in views.iter().enumerate() {
            assert_eq!(
                t.buffer.read_f32()[t.byte_offset / 4],
                marks[i],
                "bump reset aliased a live view (slice {i})"
            );
        }

        // A capacity that cannot be rounded to a power of two is rejected
        // instead of wrapping to a tiny slab.
        assert_eq!(
            rt.ensure_bump(usize::MAX).unwrap_err(),
            "bump capacity overflow"
        );
    });
}

#[test]
fn a_long_unsynchronized_chain_keeps_every_result() {
    with_gpu(|rt| {
        // The steady-state shape of a real consumer: encode a long run of
        // dispatches into one command buffer, allocating and dropping cold
        // temporaries as it goes, and synchronize once at the end.
        //
        // This is where a premature recycle shows up. A temp dropped mid-chain
        // is still being read by the GPU, so the pool must not hand it to a
        // later dispatch in the same command buffer; if it does, an earlier
        // slot's result is overwritten and only a per-slot check catches it.
        // It is also the only test that advances the 16 MiB constant arena over
        // hundreds of dispatches instead of a handful.
        rt.set_async_encode(true).unwrap();
        assert!(rt.async_encode_enabled());
        rt.take_dispatch_count();

        const ROUNDS: usize = 256;
        const VARIANTS: usize = 8;
        let (m, n, k) = (32, 32, 48);

        let a = tensor_f32(rt, &[m, k], &random_f32(m * k, 31));
        // Distinct operands per slot: identical ones would make a stale or
        // swapped buffer indistinguishable from a correct one.
        let b_hosts: Vec<Vec<f32>> = (0..VARIANTS)
            .map(|v| random_f32(k * n, 40 + v as u64))
            .collect();
        let bs: Vec<Tensor> = b_hosts.iter().map(|h| tensor_f32(rt, &[k, n], h)).collect();
        let a_host = a.buffer.read_f32();

        let sink = rt.alloc_tensor_f32(&[ROUNDS * m * n]).unwrap();
        for round in 0..ROUNDS {
            let scratch = rt.alloc_tensor_f32(&[m, n]).unwrap();
            gemm_f32(&a, &bs[round % VARIANTS], &scratch, GemmBackend::TensorOps).unwrap();
            gpu_copy(&scratch, &sink.view(&[m, n], round * m * n)).unwrap();
            drop(scratch);
        }
        rt.synchronize().unwrap();

        assert_eq!(
            rt.take_dispatch_count(),
            2 * ROUNDS,
            "one GEMM and one copy per round should have been encoded"
        );

        let all = sink.buffer.read_f32();
        let expected: Vec<_> = b_hosts
            .iter()
            .map(|h| reference(Layout::Nn, &a_host, h, m, n, k))
            .collect();
        for round in 0..ROUNDS {
            assert_within_bound(
                &format!("chain slot {round}"),
                &all[round * m * n..(round + 1) * m * n],
                &expected[round % VARIANTS],
                k,
                0.0,
            );
        }
    });
}

#[test]
fn externally_allocated_storage_can_back_a_gemm_output() {
    with_gpu(|rt| {
        // How a consumer wires its own arena into tessl: allocate a `GpuBuffer`,
        // wrap sub-windows of it with `Tensor::from_buffer`, and dispatch. This
        // is the only public route that does not go through `alloc_tensor_*`,
        // so the offset it computes is never exercised by any other test here.
        let (m, n, k) = (48, 33, 21);
        let a_host = random_f32(m * k, 51);
        let b_host = random_f32(k * n, 52);
        let expect = reference(Layout::Nn, &a_host, &b_host, m, n, k);

        let bytes = (m * k + k * n + m * n) * DType::F32.size_of();
        let arena = rt.alloc_buffer(bytes).unwrap();
        let a = Tensor::from_buffer(rt, arena.clone(), &[m, k], DType::F32, 0).unwrap();
        let b = Tensor::from_buffer(
            rt,
            arena.clone(),
            &[k, n],
            DType::F32,
            m * k * DType::F32.size_of(),
        )
        .unwrap();
        let c = Tensor::from_buffer(
            rt,
            arena.clone(),
            &[m, n],
            DType::F32,
            (m * k + k * n) * DType::F32.size_of(),
        )
        .unwrap();

        {
            let mut host = arena.contents_f32();
            host[..m * k].copy_from_slice(&a_host);
            host[m * k..m * k + k * n].copy_from_slice(&b_host);
            host[m * k + k * n..].fill(-7.0);
        }
        gemm_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();

        let host = arena.read_f32();
        assert_within_bound(
            "gemm into caller-owned storage",
            &host[m * k + k * n..],
            &expect,
            k,
            0.0,
        );
        // The operands share the allocation; a kernel writing outside C's
        // window would have corrupted them.
        assert_eq!(&host[..m * k], &a_host[..], "A was modified");
        assert_eq!(&host[m * k..m * k + k * n], &b_host[..], "B was modified");
    });
}

#[test]
fn deep_copy_and_gpu_copy_reproduce_their_source() {
    with_gpu(|rt| {
        // `deep_copy` allocates and blits on the GPU; consumers use it to fork
        // a bank without a host round trip, so a bitwise match is the contract.
        let src = tensor_f32(rt, &[97], &random_f32(97, 61));
        let dup = src.deep_copy().unwrap();
        let dst = rt.alloc_tensor_f32(&[97]).unwrap();
        gpu_copy(&src, &dst).unwrap();
        rt.synchronize().unwrap();

        let original = src.buffer.read_f32();
        assert!(original.iter().any(|&x| x != 0.0), "source was all zero");
        for (i, (&want, (&got_dup, &got_dst))) in original
            .iter()
            .zip(
                dup.buffer
                    .read_f32()
                    .iter()
                    .zip(dst.buffer.read_f32().iter()),
            )
            .enumerate()
        {
            assert_eq!(
                got_dup.to_bits(),
                want.to_bits(),
                "deep_copy differs at [{i}]"
            );
            assert_eq!(
                got_dst.to_bits(),
                want.to_bits(),
                "gpu_copy differs at [{i}]"
            );
        }
    });
}

#[test]
fn softcap_matches_its_definition() {
    with_gpu(|rt| {
        // `softcap * tanh(x / softcap)` is applied to logits, where getting the
        // cap or the scaling wrong changes the distribution without producing
        // anything obviously broken. Checking the saturating tails as well as
        // the linear middle pins both the shape and the asymptote.
        let cap = 30.0f32;
        // +-1000 is 33 caps out, far enough that tanh has saturated to exactly
        // 1.0, and short of the ~+-1300 where the kernel's tanh overflows (see
        // `softcap_saturates_instead_of_overflowing_for_extreme_logits`).
        let pre_host: Vec<f32> = (0..1024)
            .map(|i| (i as f32 - 512.0) * 0.5)
            .chain([-1000.0, 1000.0, 0.0])
            .collect();
        let pre = tensor_f32(rt, &[pre_host.len()], &pre_host);
        let post = softcap_f32(rt, &pre, cap).unwrap();
        rt.synchronize().unwrap();

        let got = post.buffer.read_f32();
        assert_eq!(got.len(), pre_host.len());
        for (i, (&x, &g)) in pre_host.iter().zip(got.iter()).enumerate() {
            let want = cap * (x / cap).tanh();
            // Metal's `tanh` is specified to a handful of ULP and the division
            // rounds once more, so a few hundred ULP of slack is generous for
            // the implementation while still an order of magnitude tighter than
            // any real defect: a missing tanh, a dropped cap, or a reciprocal
            // in place of the divide all move the result by whole percent.
            let tol = 1e-5 * want.abs().max(1.0);
            assert!(
                (g - want).abs() <= tol,
                "softcap[{i}] pre={x}: got {g}, want {want}"
            );
        }
        // The asymptote is the cap itself, in both directions.
        assert!((got[got.len() - 2] - cap).abs() < 1e-4);
        assert!((got[got.len() - 3] + cap).abs() < 1e-4);
        assert_eq!(got[got.len() - 1], 0.0);
    });
}

/// Known defect, filed rather than fixed here: this suite owns `tests/` only.
///
/// Softcapping exists to bound unbounded logits, so the one input class it must
/// survive is the extreme one.
///
/// It did not. `kernels/utils.metal` evaluated `softcap * tanh(pre/softcap)`
/// with Metal's `tanh`, which is computed through `exp(2z)` and leaves float
/// range around |z| ~= 44: at cap = 30 the result went to `inf` near pre = 1300
/// and to NaN from pre = 1350. A NaN logit poisons its entire softmax row, not
/// just its own element.
///
/// The kernel now clamps `z` before `tanh`, which is free — `tanh` has already
/// rounded to exactly +/-1 in f32 by |z| ~= 8.7. This test asserted the correct
/// behaviour and failed while the defect stood; it guards against its return.
#[test]
fn softcap_saturates_instead_of_overflowing_for_extreme_logits() {
    with_gpu(|rt| {
        let cap = 30.0f32;
        let pre_host: Vec<f32> = vec![-1e6, -5000.0, -1400.0, 1400.0, 5000.0, 1e6];
        let pre = tensor_f32(rt, &[pre_host.len()], &pre_host);
        let post = softcap_f32(rt, &pre, cap).unwrap();
        rt.synchronize().unwrap();
        for (&x, &g) in pre_host.iter().zip(post.buffer.read_f32().iter()) {
            assert!(
                (g - cap * x.signum()).abs() < 1e-3,
                "softcap(pre={x}, cap={cap}) = {g}, expected saturation at {}",
                cap * x.signum()
            );
        }
    });
}

#[test]
fn a_runtime_that_outlives_its_tensors_still_works() {
    with_gpu(|rt| {
        // Tensors hold an `Arc<GpuRuntime>` and pooled buffers hold a `Weak`
        // back to it, which is a cycle waiting to be got wrong in either
        // direction: a dropped tensor must not take the runtime with it, and a
        // buffer freed after its runtime went away must not try to recycle into
        // it. This drops many generations of tensors and keeps using the runtime.
        for generation in 0..32 {
            let t = rt.alloc_tensor_f32(&[1 << 10]).unwrap();
            t.buffer.write_f32(&vec![generation as f32; 1 << 10]);
            let view = t.view(&[1 << 9], 1 << 9);
            drop(t);
            // The view keeps the storage alive on its own.
            assert!(view
                .buffer
                .read_f32()
                .iter()
                .all(|&x| x == generation as f32));
            drop(view);
            rt.synchronize().unwrap();
        }
        let probe = rt.alloc_tensor_f32(&[4]).unwrap();
        assert_eq!(probe.buffer.read_f32(), vec![0.0; 4]);

        // Buffers whose runtime handle has already been dropped elsewhere still
        // release cleanly; a `GpuRuntime` clone kept only by a tensor is enough.
        let orphan = {
            let scoped: std::sync::Arc<GpuRuntime> = std::sync::Arc::clone(rt);
            let t = scoped.alloc_tensor_f32(&[8]).unwrap();
            drop(scoped);
            t
        };
        assert_eq!(orphan.numel(), 8);
        rt.synchronize().unwrap();
    });
}
