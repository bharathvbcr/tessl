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

use objc2_metal::MTLSharedEvent;

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

/// Apple silicon GPUs share system memory with the CPU, so a consumer that
/// budgets GPU memory as a separate pool double-counts it. Only the `true`
/// side is reachable on this hardware.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[test]
fn apple_silicon_reports_unified_memory() {
    with_gpu(|rt| {
        assert!(
            rt.memory_info().has_unified_memory,
            "MTLDevice.hasUnifiedMemory read false on Apple silicon"
        );
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
        assert_eq!(rt.alloc_tensor_f32_hot(&[1024]).unwrap().buffer.kind(), BufferKind::Hot);
        assert_eq!(rt.alloc_tensor_f32(&[1024]).unwrap().buffer.kind(), BufferKind::Cold);
        rt.ensure_bump(1 << 16).unwrap();
        assert_eq!(rt.bump_alloc_f32(&[64]).unwrap().buffer.kind(), BufferKind::Bump);
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
                t.read_f32().unwrap().iter().all(|&x| x == 0.0),
                "bump slice {i} was not zeroed"
            );
            t.write_f32(&[i as f32 + 1.0; 64]).unwrap();
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
        let marks: Vec<f32> = views.iter().map(|t| t.read_f32().unwrap()[0]).collect();
        rt.bump_reset().unwrap();
        let after_reset = rt.bump_alloc_f32(&[512]).unwrap();
        after_reset.write_f32(&vec![-1.0f32; after_reset.numel()]).unwrap();
        for (i, t) in views.iter().enumerate() {
            assert_eq!(
                t.read_f32().unwrap()[0],
                marks[i],
                "bump reset aliased a live view (slice {i})"
            );
        }

        // A capacity that cannot be rounded to a power of two is rejected
        // instead of wrapping to a tiny slab.
        assert_eq!(rt.ensure_bump(usize::MAX).unwrap_err(), "bump capacity overflow");
    });
}

/// Views hand out their own window, not the slab: reading or writing through
/// the view reaches only its elements, and a wrong-length write is refused.
#[test]
fn bump_views_read_and_write_only_their_own_window() {
    with_gpu(|rt| {
        rt.ensure_bump(1 << 16).unwrap();
        let a = rt.bump_alloc_f32(&[64]).unwrap();
        let b = rt.bump_alloc_f32(&[64]).unwrap();
        assert_ne!(a.byte_offset(), b.byte_offset());
        b.write_f32(&[1.0; 64]).unwrap();
        assert!(
            a.read_f32().unwrap().iter().all(|&x| x == 0.0),
            "writing b landed on a's window"
        );
        assert!(b.read_f32().unwrap().iter().all(|&x| x == 1.0));
        assert!(
            b.write_f32(&[0.0; 63]).is_err(),
            "a short write must be refused, not silently partial"
        );
        // The whole-slab accessor still exists for callers who want it; the
        // view's window is where b's ones actually are.
        assert_eq!(b.buffer.read_f32()[b.byte_offset() / 4], 1.0);
    });
}

/// `bump_reset` reports the conditions `bump_alloc_f32` reports, instead of
/// panicking on them: a live host mapping makes the runtime busy.
#[test]
fn bump_reset_reports_a_busy_runtime_instead_of_panicking() {
    with_gpu(|rt| {
        rt.ensure_bump(4096).unwrap();
        let t = rt.bump_alloc_f32(&[4]).unwrap();
        let mapping = t.buffer.contents_f32();
        let err = rt.bump_reset().unwrap_err();
        assert!(err.contains("busy"), "{err}");
        drop(mapping);
        rt.bump_reset().unwrap();
    });
}

/// Only an exhausted arena falls through to the pool. A poisoned runtime is
/// an error, not a pool allocation that quietly bypasses the arena.
#[test]
fn alloc_temp_refuses_a_poisoned_runtime_instead_of_bypassing_the_bump_arena() {
    with_gpu(|rt| {
        rt.ensure_bump(1 << 16).unwrap();
        rt.set_async_encode(true).unwrap();
        assert!(rt.with_binder(|_| Err("injected".into())).is_err());
        let err = rt.alloc_temp_f32(&[64]).map(|_| ()).unwrap_err();
        assert!(err.contains("poisoned"), "{err}");
    });
}

/// A poisoned runtime refuses every later call, and only the call that
/// poisoned it saw why. The refusals have to carry the first cause, or a
/// fault surfaces far from where it happened as a bare "poisoned".
#[test]
fn a_poisoned_runtime_names_what_poisoned_it() {
    with_gpu(|rt| {
        rt.set_async_encode(true).unwrap();
        let first = rt.with_binder(|_| Err("injected encode failure".into())).unwrap_err();
        assert!(first.contains("injected encode failure"), "{first}");
        // A later, different failure does not replace the first cause.
        assert!(rt.with_binder(|_| Err("second failure".into())).is_err());
        for err in [
            rt.synchronize().unwrap_err(),
            rt.alloc_tensor_f32(&[4]).map(|_| ()).unwrap_err(),
            rt.with_binder(|_| Ok(())).unwrap_err(),
        ] {
            assert!(err.contains("poisoned"), "{err}");
            assert!(
                err.contains("injected encode failure"),
                "the refusal lost the cause: {err}"
            );
        }
    });

    with_gpu(|rt| {
        rt.poison_as_shared_event_timeout_for_test();
        let err = rt.synchronize().unwrap_err();
        assert!(err.contains("timed out"), "the refusal lost the cause: {err}");
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
        let b_hosts: Vec<Vec<f32>> = (0..VARIANTS).map(|v| random_f32(k * n, 40 + v as u64)).collect();
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
fn async_encode_past_the_constant_arena_without_a_synchronize_stays_healthy() {
    with_gpu(|rt| {
        // A decode or training loop that encodes asynchronously and never
        // waits. Every scalar bind takes a slot in the runtime's 16 MiB
        // constant arena, and only a waiting commit used to give those slots
        // back, so a long enough run exhausted the arena and the failed bind
        // poisoned the runtime for good. Separately, the 100k-dispatch hard cap
        // counted dispatches across non-waiting commits, so once a run passed
        // 100k every later dispatch committed a command buffer of its own.
        //
        // Each dispatch here binds 64 bytes of constants (a 48-byte source
        // payload and a 4-byte count, each in a 16-byte-aligned slot), so the
        // loop puts 19.2 MB through the arena -- past its 16 MiB -- and fills
        // more than one command buffer to the 100k cap, in a few seconds rather
        // than the ~1M single-scalar dispatches a real run would take.
        const DISPATCHES: usize = 300_000;
        const PAYLOAD_FLOATS: usize = 12;
        const DISPATCH_CAP: usize = 100_000;

        rt.set_async_encode(true).unwrap();
        rt.take_dispatch_count();
        let pipe = rt.pipeline("copy_f32").unwrap();
        let sink = rt.alloc_tensor_f32(&[DISPATCHES]).unwrap();
        sink.buffer.contents_f32().fill(-1.0);

        tessl::infer_trace::set_enabled(true);
        let before = tessl::infer_trace::snapshot();
        for i in 0..DISPATCHES {
            // The kernel reads its source straight out of the constant arena,
            // so a slot handed back while the GPU could still read it would
            // show up as another dispatch's value in this one's sink element.
            let mut payload = [0u8; PAYLOAD_FLOATS * 4];
            payload[..4].copy_from_slice(&((i + 1) as f32).to_ne_bytes());
            let dst = sink.view(&[1], i);
            let encoded = rt.with_binder(|bnd| {
                bnd.set_pipeline(&pipe);
                bnd.bind_bytes(&payload, 0);
                bnd.bind_tensor(&dst, 1);
                bnd.bind_u32(1, 2);
                bnd.dispatch(tessl::runtime::mtl_size(1, 1, 1), tessl::runtime::mtl_size(1, 1, 1));
                Ok(())
            });
            if let Err(err) = encoded {
                tessl::infer_trace::set_enabled(false);
                panic!(
                    "dispatch {i} of {DISPATCHES} failed without a synchronize (poisoned: {}): {err}",
                    rt.is_poisoned()
                );
            }
        }
        let encoded = tessl::infer_trace::snapshot().since(&before);
        tessl::infer_trace::set_enabled(false);
        assert!(!rt.is_poisoned(), "the runtime was poisoned by the run");
        rt.synchronize().unwrap();

        assert_eq!(rt.take_dispatch_count(), DISPATCHES);
        // One commit per full command buffer (the dispatch cap, or the
        // `TESSL_MID_COMMIT` threshold when a stress run sets one), plus the
        // arena reclaims -- a handful by default. Committing every dispatch
        // past the cap would be ~200k.
        let mid_commit = ["TESSL_MID_COMMIT", "METAL_RUNTIME_MID_COMMIT"]
            .iter()
            .find_map(|k| std::env::var(k).ok())
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0);
        let per_cb = mid_commit.map_or(DISPATCH_CAP, |n| n.min(DISPATCH_CAP));
        let laps = 19_200_000usize.div_ceil(16 * 1024 * 1024);
        let bound = (DISPATCHES / per_cb + laps + 1) as u64;
        assert!(
            encoded.commits <= bound,
            "{} commits for {DISPATCHES} dispatches; at most {bound} expected",
            encoded.commits
        );

        let out = sink.buffer.read_f32();
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, (i + 1) as f32, "dispatch {i} read another dispatch's constants");
        }
    });
}

/// Drop `t`, recording its buffer's release point: the next commit's event.
fn release(rt: &GpuRuntime, released: &mut std::collections::HashMap<usize, u64>, t: Tensor) {
    released.insert(buffer_id(&t.buffer), rt.last_signaled_value() + 1);
    drop(t);
}

#[test]
fn temporaries_recycle_across_unwaited_commits() {
    with_gpu(|rt| {
        // A run that commits without waiting and never synchronizes. A cold
        // temporary dropped mid-run cannot go back to the freelist while a
        // command buffer that may read it is in flight -- but once that command
        // buffer has completed it can, and has to: otherwise every temporary of
        // the run stays allocated and resident until a synchronize that never
        // comes. With two allocators, the third command buffer cannot open
        // until the first has completed, so the live set is bounded by about
        // three rounds of temporaries however many rounds run.
        //
        // A dropped temporary's release point is the shared-event value of the
        // next commit after its drop (`last_signaled_value() + 1` then): every
        // command buffer that could have read it is at or before that one.
        // After each round an allocation must not hand back a temporary whose
        // release point the GPU has not reached. The temporaries stay alive
        // (queued or pooled), so pointer identity is decisive. Each round opens
        // with a large GEMM, so its last temporaries are usually still in
        // flight at the probe -- the check has something to catch -- and most
        // rounds must be; under `TESSL_MID_COMMIT` earlier ones of the round
        // may legitimately have come back already.
        const ROUNDS: usize = 16;
        const TEMPS: usize = 8;
        const N: usize = 256 * 1024; // 1 MiB of f32: one pool bucket
        let round_bytes = (TEMPS * N * 4) as u64;

        rt.set_async_encode(true).unwrap();
        let srcs: Vec<Vec<f32>> = (0..ROUNDS).map(|r| random_f32(N, 700 + r as u64)).collect();
        let src: Vec<Tensor> = srcs.iter().map(|h| tensor_f32(rt, &[N], h)).collect();
        let sink = rt.alloc_tensor_f32(&[ROUNDS * N]).unwrap();
        const G: usize = 1536;
        let ga = tensor_f32(rt, &[G, G], &random_f32(G * G, 791));
        let gb = tensor_f32(rt, &[G, G], &random_f32(G * G, 792));
        let gc = rt.alloc_tensor_f32(&[G, G]).unwrap();
        rt.synchronize().unwrap();

        let base = rt.current_allocated_bytes();
        let mut peak_growth = 0u64;
        let mut checked = 0usize;
        // Buffer identity -> release point of its latest drop.
        let mut released = std::collections::HashMap::new();
        for (round, src) in src.iter().enumerate() {
            gemm_f32(&ga, &gb, &gc, GemmBackend::TensorOps).unwrap();
            let mut prev: Option<Tensor> = None;
            for _ in 0..TEMPS {
                let temp = rt.alloc_tensor_f32(&[N]).unwrap();
                gpu_copy(prev.as_ref().unwrap_or(src), &temp).unwrap();
                if let Some(done) = prev.replace(temp) {
                    release(rt, &mut released, done);
                }
            }
            let last = prev.expect("TEMPS > 0");
            gpu_copy(&last, &sink.view(&[N], round * N)).unwrap();
            release(rt, &mut released, last);
            rt.commit(false).unwrap();
            let probe = rt.alloc_tensor_f32(&[N]).unwrap();
            // Read after the probe: the event only rises, so a release point
            // above this value was above it at every drain before the probe.
            let reached = rt.shared_event().signaledValue();
            if released.values().any(|&after| after > reached) {
                checked += 1;
            }
            if let Some(&after) = released.get(&buffer_id(&probe.buffer)) {
                assert!(
                    after <= reached,
                    "round {round}: a temporary with release point {after} went back to the pool \
                     while the GPU had reached only {reached}"
                );
            }
            release(rt, &mut released, probe);
            peak_growth = peak_growth.max(rt.current_allocated_bytes().saturating_sub(base));
        }
        rt.synchronize().unwrap();

        assert!(
            checked >= ROUNDS / 2,
            "only {checked} of {ROUNDS} rounds had temporaries in flight at their probe; the GEMM no \
             longer outlasts a round's host work, so the recycle check is not being exercised"
        );
        assert!(
            peak_growth <= 4 * round_bytes,
            "allocations grew {peak_growth} bytes over {ROUNDS} unwaited rounds of {round_bytes}; \
             temporaries are not recycling once their command buffer completes"
        );
        let out = sink.buffer.read_f32();
        for (round, expected) in srcs.iter().enumerate() {
            assert_eq!(
                &out[round * N..(round + 1) * N],
                &expected[..],
                "round {round} read a temporary another round had reused"
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

        // Pad each matrix start to a 16-byte boundary so validate_gemm's
        // alignment gate is not what this wiring test exercises.
        let a_elems = m * k;
        let b_off_elems = a_elems.div_ceil(4) * 4;
        let c_off_elems = (b_off_elems + k * n).div_ceil(4) * 4;
        let total_elems = c_off_elems + m * n;
        let arena = rt.alloc_buffer(total_elems * DType::F32.size_of()).unwrap();
        let a = Tensor::from_buffer(rt, arena.clone(), &[m, k], DType::F32, 0).unwrap();
        let b = Tensor::from_buffer(
            rt,
            arena.clone(),
            &[k, n],
            DType::F32,
            b_off_elems * DType::F32.size_of(),
        )
        .unwrap();
        let c = Tensor::from_buffer(
            rt,
            arena.clone(),
            &[m, n],
            DType::F32,
            c_off_elems * DType::F32.size_of(),
        )
        .unwrap();

        {
            let mut host = arena.contents_f32();
            host[..a_elems].copy_from_slice(&a_host);
            host[b_off_elems..b_off_elems + k * n].copy_from_slice(&b_host);
            host[c_off_elems..].fill(-7.0);
        }
        gemm_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();

        let host = arena.read_f32();
        assert_within_bound(
            "external arena GEMM",
            &host[c_off_elems..c_off_elems + m * n],
            &expect,
            k,
            0.0,
        );
        // The operands share the allocation; a kernel writing outside C's
        // window would have corrupted them.
        assert_eq!(&host[..a_elems], &a_host[..], "A was modified");
        assert_eq!(&host[b_off_elems..b_off_elems + k * n], &b_host[..], "B was modified");
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
            .zip(dup.buffer.read_f32().iter().zip(dst.buffer.read_f32().iter()))
            .enumerate()
        {
            assert_eq!(got_dup.to_bits(), want.to_bits(), "deep_copy differs at [{i}]");
            assert_eq!(got_dst.to_bits(), want.to_bits(), "gpu_copy differs at [{i}]");
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
            assert!((g - want).abs() <= tol, "softcap[{i}] pre={x}: got {g}, want {want}");
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
            assert!(view.buffer.read_f32().iter().all(|&x| x == generation as f32));
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

/// A params slot is not rewritten under a dispatch that has been encoded but
/// not yet run. With async encode on, the dispatch below sits in an open
/// command buffer when the step resets the cursor and pushes the next value
/// into the same slot; a raw host write lands first and the kernel reads the
/// new value. The push has to take the host-access lease, which commits and
/// waits, as every other host write into a GPU buffer does.
#[test]
fn a_params_push_waits_for_encoded_work_reading_the_slot() {
    with_gpu(|rt| {
        rt.set_async_encode(true).expect("async encode");
        let out = rt.alloc_buffer(4).expect("alloc");
        out.write_f32(&[0.0]);
        let off = rt
            .with_params(|p| {
                p.reset();
                p.push_f32(7.0)
            })
            .expect("params")
            .expect("push 7");
        let pipe = rt.pipeline("add_inplace_f32").expect("pipeline");
        rt.with_params(|p| {
            tessl::dispatch::dispatch_1d(rt, &pipe, 1, |bnd| {
                tessl::dispatch::set_gpu_buf(bnd, &out, 0);
                tessl::dispatch::set_gpu_buf_offset(bnd, p.buffer(), off, 1);
                tessl::dispatch::set_u32(bnd, 1, 2);
            })
        })
        .expect("params")
        .expect("dispatch");
        let next = rt
            .with_params(|p| {
                p.reset();
                p.push_f32(9.0)
            })
            .expect("params")
            .expect("push 9");
        assert_eq!(next, off, "the reset should reuse the slot");
        rt.synchronize().expect("sync");
        rt.set_async_encode(false).expect("sync encode");
        assert_eq!(
            out.read_f32(),
            vec![7.0],
            "the encoded dispatch read the next step's value"
        );
    });
}
