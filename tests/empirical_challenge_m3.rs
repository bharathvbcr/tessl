//! Empirical Challenger Test Suite for Milestone M3:
//! Shared Memory Pools & Zero-Copy UMA Integration.
//!
//! Evaluates:
//! 1. `alloc_internal_buffer` behavior:
//!    - Verifies buffer usability and zero-copy mapped contents.
//!    - Verifies whether `retain_external` is invoked or bypassed (`external_wrap_total`).
//! 2. `SharedBuf::to_gpu_buffer`:
//!    - Verifies that `SharedAlloc` pooled buffers expose a valid `GpuBuffer`.
//!    - Verifies lifecycle independence: `SharedAlloc` dropped while `GpuBuffer` retained.
//!    - Verifies data integrity across shared host leases and GPU buffer views.

mod common;

use common::with_gpu;
use ojas_core::MappedBuffer;
use tessl::shared_buf::SharedAlloc;

#[test]
fn challenge_shared_buf_to_gpu_buffer_lifecycle() {
    with_gpu(|rt| {
        let nbytes = 4096;
        let alloc = SharedAlloc::new(rt, nbytes).expect("SharedAlloc allocation");
        let shared = alloc.shared_buf();

        // 1. Verify to_gpu_buffer returns Some(GpuBuffer)
        let gpu_buf = shared
            .to_gpu_buffer()
            .expect("to_gpu_buffer must return Some for pooled alloc");
        assert_eq!(gpu_buf.nbytes(), nbytes);

        // 2. Write data via host_write on MappedBuffer trait
        {
            let mut wlease = shared.host_write().expect("write lease");
            let f32s = wlease.as_f32_mut().expect("f32 lease");
            for (i, v) in f32s.iter_mut().enumerate() {
                *v = (i as f32) * 1.5 + 0.25;
            }
        }

        // 3. Read back via GpuBuffer contents
        let contents = gpu_buf.try_contents_f32().expect("contents f32");
        assert_eq!(contents.len(), nbytes / 4);
        for (i, &v) in contents.iter().enumerate() {
            assert_eq!(v, (i as f32) * 1.5 + 0.25);
        }
        drop(contents);

        // 4. Lifecycle decoupling: drop SharedAlloc, retain GpuBuffer
        drop(alloc);
        drop(shared);

        // GpuBuffer must still be completely valid and readable without dangling pointer
        let contents_after = gpu_buf.try_contents_f32().expect("contents f32 after SharedAlloc drop");
        for (i, &v) in contents_after.iter().enumerate() {
            assert_eq!(v, (i as f32) * 1.5 + 0.25);
        }
    });
}

#[test]
fn challenge_alloc_internal_buffer_wrap_bookkeeping() {
    with_gpu(|rt| {
        let before_wraps = rt.external_wrap_total();

        // Allocate internal buffer (as used by ojas-metal status slab and spare)
        let buf = rt.alloc_internal_buffer(1024).expect("alloc_internal_buffer");
        assert_eq!(buf.nbytes(), 1024);

        // Can write and read
        let mut map = buf.try_contents_u32().expect("contents u32");
        map[0] = 0xCAFEBABE;
        assert_eq!(map[0], 0xCAFEBABE);
        drop(map);

        let after_wraps = rt.external_wrap_total();

        // Observation on external wrap bookkeeping:
        // Does alloc_internal_buffer count toward external wraps?
        println!("external_wrap_total: before={before_wraps}, after={after_wraps}");

        // Drop buffer and synchronize
        drop(buf);
        rt.synchronize().expect("synchronize after dropping internal buffer");
    });
}
