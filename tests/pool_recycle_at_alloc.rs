//! The buffer pool recycles at allocation time.
//!
//! A buffer dropped after the GPU has finished with it used to return to the
//! freelist only at the next commit, which the next call reaches after it has
//! allocated its own buffers: two sets were live across consecutive calls. An
//! allocation now takes it when no command buffer is open and the lifetime
//! event has caught up with everything submitted, and not before.
//!
//! The device-bytes test runs in a child process (`common::in_child`):
//! `GpuRuntime::current_allocated_bytes` is the device's figure for the whole
//! process. The decision logic under an in-flight command buffer and under a
//! held `active_m4` lock is in `src/runtime.rs`'s `alloc_recycle_tests`, which
//! can set the lifetime event by hand; the GPU race here covers it end to end.

mod common;

use common::{in_child, with_gpu};
use objc2_metal::MTLBuffer;
use tessl::dispatch::{dispatch_1d, set_gpu_buf, set_u32};
use tessl::runtime::GpuRuntime;
use tessl::tensor::GpuBuffer;

/// Address of a buffer's storage: the identity of a pooled buffer.
fn addr(b: &GpuBuffer) -> usize {
    b.metal().contents().as_ptr() as usize
}

/// `dst[..n] = src[..n]`, one `copy_f32` dispatch.
fn copy(rt: &GpuRuntime, src: &GpuBuffer, dst: &GpuBuffer, n: usize) -> Result<(), String> {
    let pipeline = rt.pipeline("copy_f32")?;
    dispatch_1d(rt, &pipeline, n, |bnd| {
        set_gpu_buf(bnd, src, 0);
        set_gpu_buf(bnd, dst, 1);
        set_u32(bnd, n as u32, 2);
    })
}

/// Sizes in bytes that exercise both bucket rules: a power of two, a
/// page-rounded size above 1 MiB, and a small power of two.
const SIZES: [usize; 3] = [1 << 20, 3_000_000, 1 << 16];

fn set(rt: &GpuRuntime) -> Vec<GpuBuffer> {
    SIZES.iter().map(|&n| rt.alloc_buffer(n).unwrap()).collect()
}

/// A dispatch that binds every buffer of `bufs` as `copy_f32`'s src or dst.
fn touch(rt: &GpuRuntime, bufs: &[GpuBuffer]) {
    for pair in bufs.windows(2) {
        copy(rt, &pair[0], &pair[1], pair[1].nbytes() / 4).unwrap();
    }
}

/// Pipelines are compiled on first use and charged to the device outside the
/// pool, once per process; a runtime of its own pays that before a window is
/// measured.
fn warm() {
    with_gpu(|rt| {
        let bufs = set(rt);
        touch(rt, &bufs);
        rt.synchronize().unwrap();
    });
}

/// Allocate a set, encode, commit and wait, drop the set after the wait, and
/// allocate the same sizes again before any new commit: the pool serves the
/// same buffers and the device's bytes do not move.
#[test]
fn a_set_dropped_after_the_wait_serves_the_next_set_without_a_commit() {
    in_child(
        "a_set_dropped_after_the_wait_serves_the_next_set_without_a_commit",
        &[],
        || {
            warm();
            with_gpu(|rt| {
                let first = set(rt);
                touch(rt, &first);
                rt.synchronize().unwrap();
                let mut old: Vec<usize> = first.iter().map(addr).collect();
                old.sort_unstable();
                let base = rt.current_allocated_bytes();
                rt.reset_peak_allocated_bytes();

                drop(first);
                let second = set(rt);

                let mut got: Vec<usize> = second.iter().map(addr).collect();
                got.sort_unstable();
                assert_eq!(got, old, "the dropped buffers were not reused before the next commit");
                assert_eq!(
                    rt.current_allocated_bytes(),
                    base,
                    "the second set allocated fresh device memory"
                );
                assert_eq!(
                    rt.peak_allocated_bytes(),
                    base,
                    "the device peak rose above the first set"
                );

                // And they are usable: a dispatch over the reused set runs.
                touch(rt, &second);
                rt.synchronize().unwrap();
            });
        },
    );
}

/// The shape `EmbedGemma2Model::encode` has: a scope is open, the work is
/// encoded, `synchronize` waits inside it, and the buffers drop after the
/// wait with the scope still open (and a second one nested in it).
#[test]
fn a_set_dropped_after_a_wait_inside_open_scopes_is_reused() {
    warm();
    with_gpu(|rt| {
        let outer = rt.async_scope();
        let inner = rt.async_scope();
        assert_eq!(rt.async_scope_depth(), 2);

        let first = set(rt);
        touch(rt, &first);
        rt.synchronize().unwrap();
        let old: Vec<usize> = first.iter().map(addr).collect();
        drop(first);

        // No dispatch yet: the scope's command buffer is not open.
        let second = set(rt);
        let got: Vec<usize> = second.iter().map(addr).collect();
        assert_eq!(
            got, old,
            "buffers dropped after the wait were not reused inside the scope"
        );
        touch(rt, &second);
        rt.synchronize().unwrap();
        drop(inner);
        drop(outer);
    });
}

/// A buffer dropped while a command buffer that binds it is still open must
/// not be handed out: the allocation is a different buffer. After the wait it
/// is the one the pool serves.
#[test]
fn a_buffer_dropped_while_a_command_buffer_is_open_is_not_reused_until_it_completes() {
    warm();
    with_gpu(|rt| {
        let scope = rt.async_scope();
        let bufs = set(rt);
        touch(rt, &bufs); // queued, not committed: the command buffer is open
        let held: Vec<usize> = bufs.iter().map(addr).collect();
        drop(bufs);

        let during = set(rt);
        for (i, b) in during.iter().enumerate() {
            assert!(
                !held.contains(&addr(b)),
                "buffer {i} was handed out while an open command buffer still binds its previous owner"
            );
        }
        // Keep encoding into the same command buffer with the new set.
        touch(rt, &during);
        let mut known = held;
        known.extend(during.iter().map(addr));
        drop(during);

        rt.synchronize().unwrap();
        let after = set(rt);
        for (i, b) in after.iter().enumerate() {
            assert!(
                known.contains(&addr(b)),
                "buffer {i} after the wait is fresh memory, not one the pool already held"
            );
        }
        drop(after);
        drop(scope);
    });
}

/// A GPU write that races a reuse. A long chain of copies into `x` is
/// committed without a wait; `x` is dropped; a buffer of the same size is
/// then allocated and filled on the host with a different pattern, without a
/// wait. If the pool handed `x` back while the chain was running, the GPU's
/// pattern would land in the new buffer. It holds the host's pattern at the end.
#[test]
fn a_gpu_write_in_flight_does_not_land_in_a_reused_buffer() {
    warm();
    const ELEMS: usize = 8 << 20; // 32 MiB
    const COPIES: usize = 300;
    const HOST: u32 = 0x4000_0000; // 2.0f
    with_gpu(|rt| {
        let src = rt.alloc_buffer(ELEMS * 4).unwrap();
        src.try_write_u32(&vec![0x3f80_0000; ELEMS]).unwrap(); // 1.0f
        let x = rt.alloc_buffer(ELEMS * 4).unwrap();
        x.try_write_u32(&vec![HOST; ELEMS]).unwrap();
        rt.synchronize().unwrap();

        {
            let _scope = rt.async_scope();
            for _ in 0..COPIES {
                copy(rt, &src, &x, ELEMS).unwrap();
            }
            rt.commit(false).unwrap();
        }
        let submitted = rt.last_signaled_value();
        let running = rt.last_completed_value() < submitted;
        let x_addr = addr(&x);
        drop(x);

        // Same size, so the same bucket. Written on the host with no wait.
        let y = rt.alloc_buffer_from_u32(&vec![HOST; ELEMS]).unwrap();
        if running {
            assert_ne!(
                addr(&y),
                x_addr,
                "a buffer a running command buffer writes was handed out again"
            );
        }
        rt.synchronize().unwrap();
        let seen = y.try_contents_u32().unwrap();
        assert!(
            seen.iter().all(|&w| w == HOST),
            "the GPU's write landed in a buffer allocated after the drop (first word {:#x})",
            seen[0]
        );
    });
}
