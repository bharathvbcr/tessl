//! Opt-in per-dispatch GPU timestamps (profiling only, off by default).
//!
//! `TESSL_DISPATCH_STAMPS=1` (or [`crate::ab_flags::set_dispatch_stamps`]) makes
//! a runtime built with the flag on own a 1,024-slot timestamp heap. For every
//! command buffer opened while the flag is on, slot 0 holds an origin stamp
//! written when the encoder opens, and [`crate::dispatch::Binder::dispatch`] writes one Precise
//! timestamp into the next slot right after each dispatch. Each slot is tagged
//! with the pipeline that was bound, whether a barrier was emitted since the
//! previous dispatch, and the layer set through [`set_stamp_layer`]. At the
//! waiting commit the heap is resolved and the stamps are available through
//! [`crate::runtime::GpuRuntime::take_dispatch_stamps`].
//!
//! What a stamp measures. Metal's `writeTimestamp` "ensures that any prior work
//! finishes, but doesn't delay any subsequent work". Consecutive stamps
//! therefore bracket one dispatch *including* its launch, the drain of its
//! last threadgroups and the barrier before it. The API cannot report when a
//! kernel started, so kernel time and inter-dispatch gap are not separable
//! here. The cost of a stamp is part of every delta; the median delta of a
//! near-empty kernel is the practical floor. (Back-to-back stamps with no
//! dispatch between them are not usable as a calibration: measured on M-series,
//! they resolve to 0, i.e. the write is tied to a dispatch.)
//!
//! Stamps also serialise dispatches that hazard mode would otherwise leave
//! unordered, so profiled time is an upper bound on the unprofiled GPU time.
//! Compare stamps-on against stamps-off before reading anything off them.
//!
//! Off costs one relaxed atomic load per command buffer open and one per
//! `Binder` construction; nothing is allocated and no heap exists unless the
//! runtime was built with the flag on.

use crate::runtime::GpuRuntime;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTL4ComputeCommandEncoder, MTL4TimestampGranularity, MTLDevice};
use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, Ordering};

/// Slots in the per-dispatch heap: slot 0 is the origin stamp, the rest are one
/// per dispatch.
pub const STAMP_HEAP_SLOTS: usize = 1024;

static LAYER: AtomicI32 = AtomicI32::new(-1);

/// Tag the dispatches stamped from now on with a layer index. A no-op (one
/// relaxed load) when stamps are off. By convention `-1` is "before the
/// layers" and `n_layers` is the tail (final norm, lm_head, argmax).
pub fn set_stamp_layer(layer: i32) {
    if crate::ab_flags::dispatch_stamps() {
        LAYER.store(layer, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct StampMeta {
    pipeline: usize,
    barrier_before: bool,
    layer: i32,
}

/// Metadata for the open command buffer's stamps. `meta[i]` describes heap slot `i + 1`.
#[derive(Debug, Default)]
pub struct StampLog {
    armed: bool,
    meta: Vec<StampMeta>,
    barrier_pending: bool,
    overflowed: bool,
    dropped_batches: u32,
}

impl StampLog {
    /// Count one armed command buffer whose stamps were discarded.
    fn note_dropped(&mut self) {
        self.dropped_batches += 1;
    }
}

/// One dispatch, as seen by the stamps.
#[derive(Clone, Debug, PartialEq)]
pub struct DispatchStamp {
    /// Kernel the dispatch ran (`"execute_icb"` for an ICB range, `"?"` if the
    /// pipeline was not built by this runtime).
    pub name: String,
    /// Layer set by [`set_stamp_layer`] when the dispatch was encoded.
    pub layer: i32,
    /// A barrier was emitted between the previous dispatch and this one
    /// (always-on mode: the previous dispatch's trailing barrier counts).
    pub barrier_before: bool,
    /// GPU time at the end of the dispatch, ns.
    pub t_end_ns: u64,
    /// `t_end_ns` minus the previous stamp (the origin stamp for the
    /// first dispatch): launch + run + drain + barrier + the stamp itself.
    pub dt_ns: u64,
}

/// Everything one waiting commit resolved.
#[derive(Clone, Debug, PartialEq)]
pub struct DispatchStampBatch {
    /// Timestamp ticks per second, from `queryTimestampFrequency`.
    pub freq_hz: u64,
    /// Origin stamp (first command on the encoder), ns.
    pub t0_ns: u64,
    /// One entry per stamped dispatch, in encode order.
    pub stamps: Vec<DispatchStamp>,
    /// More dispatches than the heap holds: `stamps` is a prefix, not the token.
    pub overflowed: bool,
    /// Earlier armed command buffers that ended without a waiting commit and
    /// whose stamps were discarded.
    pub dropped_batches: u32,
    /// Waiting commits folded into this batch. 1 for a token encoded into one
    /// command buffer; more when host mappings commit mid-token. A later
    /// command buffer's first `dt_ns` is measured from its own origin stamp,
    /// so the time between command buffers is in `t_end_ns` but in no `dt_ns`.
    pub command_buffers: u32,
}

impl DispatchStampBatch {
    pub(crate) fn merge(&mut self, next: DispatchStampBatch) {
        self.stamps.extend(next.stamps);
        self.overflowed |= next.overflowed;
        self.dropped_batches += next.dropped_batches;
        self.command_buffers += next.command_buffers;
    }
}

impl GpuRuntime {
    /// Arm (or disarm) stamping for the command buffer that just began. Called
    /// with the command buffer open and before any encoder exists.
    pub(crate) fn arm_dispatch_stamps(&self) {
        let Ok(mut log) = self.stamp_log.lock() else { return };
        if log.armed && !log.meta.is_empty() {
            log.note_dropped();
        }
        log.meta.clear();
        log.barrier_pending = false;
        log.overflowed = false;
        log.armed = false;
        if !crate::ab_flags::dispatch_stamps() {
            return;
        }
        if self.metal4.stamp_heap.is_none() {
            return;
        }
        // Slot 0 is written by `stamp_origin` once the encoder exists. The
        // command-buffer-level begin stamp is not usable as an origin: measured
        // on M-series, it can land *after* the first dispatch's stamp.
        log.armed = true;
    }

    /// Write slot 0, the time origin, as the first command on a new encoder.
    pub(crate) fn stamp_origin(&self, enc: &ProtocolObject<dyn MTL4ComputeCommandEncoder>) {
        let Some(heap) = self.metal4.stamp_heap.as_ref() else {
            return;
        };
        let Ok(log) = self.stamp_log.lock() else { return };
        if !log.armed || !log.meta.is_empty() {
            return;
        }
        // SAFETY: slot 0 is inside the heap (`STAMP_HEAP_SLOTS >= 1`).
        unsafe {
            enc.writeTimestampWithGranularity_intoHeap_atIndex(MTL4TimestampGranularity::Precise, heap, 0);
        }
    }

    /// True when the command buffer being encoded is armed.
    pub(crate) fn dispatch_stamps_armed(&self) -> bool {
        crate::ab_flags::dispatch_stamps() && self.stamp_log.lock().map(|l| l.armed).unwrap_or(false)
    }

    pub(crate) fn stamp_note_barrier(&self) {
        if let Ok(mut log) = self.stamp_log.lock() {
            if log.armed {
                log.barrier_pending = true;
            }
        }
    }

    /// Write the stamp for the dispatch just encoded. `pipeline` is the bound
    /// pipeline's address (0 for an ICB execute).
    pub(crate) fn stamp_after_dispatch(&self, enc: &ProtocolObject<dyn MTL4ComputeCommandEncoder>, pipeline: usize) {
        let Some(heap) = self.metal4.stamp_heap.as_ref() else {
            return;
        };
        let Ok(mut log) = self.stamp_log.lock() else { return };
        if !log.armed {
            return;
        }
        let slot = log.meta.len() + 1;
        if slot >= STAMP_HEAP_SLOTS {
            log.overflowed = true;
            return;
        }
        // SAFETY: `slot < STAMP_HEAP_SLOTS` (checked above) and the heap has
        // exactly that many entries.
        unsafe {
            enc.writeTimestampWithGranularity_intoHeap_atIndex(MTL4TimestampGranularity::Precise, heap, slot as _);
        }
        let barrier_before = std::mem::take(&mut log.barrier_pending);
        log.meta.push(StampMeta {
            pipeline,
            barrier_before,
            layer: LAYER.load(Ordering::Relaxed),
        });
    }

    /// Read back an armed command buffer after its waiting commit completed.
    pub(crate) fn resolve_dispatch_stamps(&self) {
        let Ok(mut log) = self.stamp_log.lock() else { return };
        if !log.armed {
            return;
        }
        log.armed = false;
        let meta = std::mem::take(&mut log.meta);
        let overflowed = log.overflowed;
        let dropped = log.dropped_batches;
        log.dropped_batches = 0;
        drop(log);
        let Some(heap) = self.metal4.stamp_heap.as_ref() else {
            return;
        };
        let result = self.build_batch(heap, &meta, overflowed, dropped);
        self.store_dispatch_stamps(result);
    }

    fn build_batch(
        &self,
        heap: &ProtocolObject<dyn objc2_metal::MTL4CounterHeap>,
        meta: &[StampMeta],
        overflowed: bool,
        dropped_batches: u32,
    ) -> Result<DispatchStampBatch, String> {
        let freq_hz = self.device.queryTimestampFrequency();
        if freq_hz == 0 {
            return Err("queryTimestampFrequency returned 0".into());
        }
        let ticks = crate::runtime::resolve_timestamps(heap, meta.len() + 1)?;
        let to_ns = |t: u64| (u128::from(t) * 1_000_000_000u128 / u128::from(freq_hz)) as u64;
        let names: HashMap<usize, String> = self.pipeline_names_by_ptr()?;
        let t0_ns = to_ns(ticks[0]);
        let mut prev = ticks[0];
        let mut stamps = Vec::with_capacity(meta.len());
        for (i, (m, &t)) in meta.iter().zip(&ticks[1..]).enumerate() {
            if t == 0 {
                return Err(format!("stamp slot {} was never written (resolved 0)", i + 1));
            }
            if t < prev {
                return Err(format!(
                    "non-monotone timestamps at slot {} of {} (freq {freq_hz} Hz): {t} after {prev}",
                    i + 1,
                    meta.len()
                ));
            }
            let name = match m.pipeline {
                0 => "execute_icb".to_string(),
                p => names.get(&p).cloned().unwrap_or_else(|| "?".to_string()),
            };
            stamps.push(DispatchStamp {
                name,
                layer: m.layer,
                barrier_before: m.barrier_before,
                t_end_ns: to_ns(t),
                dt_ns: to_ns(t - prev),
            });
            prev = t;
        }
        Ok(DispatchStampBatch {
            freq_hz,
            t0_ns,
            stamps,
            overflowed,
            dropped_batches,
            command_buffers: 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::mtl_size;
    use objc2_metal::MTLComputePipelineState;

    fn run_copies(rt: &GpuRuntime, n_dispatches: usize) {
        let n = 64usize;
        let src = rt.alloc_buffer(n * 4).unwrap();
        let dst = rt.alloc_buffer(n * 4).unwrap();
        let pipe = rt.pipeline("copy_f32").unwrap();
        let tpt = pipe.threadExecutionWidth().min(n).max(1);
        let groups = n.div_ceil(tpt);
        for i in 0..n_dispatches {
            set_stamp_layer(i as i32 / 2);
            rt.with_binder(|b| {
                b.set_pipeline(&pipe);
                b.bind_gpu_buf(&src, 0);
                b.bind_gpu_buf(&dst, 1);
                b.bind_u32(n as u32, 2);
                b.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
                Ok(())
            })
            .unwrap();
        }
    }

    /// One test body: the flag is process-global, so the off and on halves must
    /// not run as parallel tests.
    #[test]
    fn stamps_off_record_nothing_and_on_match_the_dispatch_count() {
        // Off: no heap, nothing resolved, nothing armed.
        crate::ab_flags::set_dispatch_stamps(false);
        let rt = GpuRuntime::new().expect("runtime");
        assert!(!rt.dispatch_stamp_heap_available());
        rt.set_async_encode(true).unwrap();
        run_copies(&rt, 5);
        rt.synchronize().unwrap();
        assert!(rt.take_dispatch_stamps().is_none());

        // On: runtime built with the flag has a heap.
        crate::ab_flags::set_dispatch_stamps(true);
        let rt = GpuRuntime::new().expect("runtime");
        assert!(rt.dispatch_stamp_heap_available());
        rt.set_async_encode(true).unwrap();
        const N: usize = 40;
        run_copies(&rt, N);
        rt.synchronize().unwrap();
        let batch = rt.take_dispatch_stamps().expect("armed").expect("resolved");
        assert!(!batch.overflowed);
        assert_eq!(batch.dropped_batches, 0);
        assert_eq!(batch.stamps.len(), N, "one stamp per dispatch");
        assert!(batch.freq_hz > 0);
        let mut prev = batch.t0_ns;
        for (i, s) in batch.stamps.iter().enumerate() {
            assert_eq!(s.name, "copy_f32", "stamp {i}");
            assert_eq!(s.layer, i as i32 / 2, "layer tag of stamp {i}");
            assert!(s.t_end_ns >= prev, "stamp {i} went backwards");
            // dt and t_end are each rounded from ticks, so they agree to 1 ns.
            assert!(s.dt_ns.abs_diff(s.t_end_ns - prev) <= 1, "dt of stamp {i}");
            prev = s.t_end_ns;
        }
        // Each dispatch carries a barrier (always-on or hazard-lazy) except the first.
        assert!(batch.stamps.iter().skip(1).all(|s| s.barrier_before));
        // Sane scale: a 64-element copy plus a stamp is well under 10 ms.
        let span = prev - batch.t0_ns;
        assert!(span > 0 && span < 100_000_000, "span {span} ns");
        // Taken once.
        assert!(rt.take_dispatch_stamps().is_none());

        // Flag flipped off after construction: next command buffer is unarmed.
        crate::ab_flags::set_dispatch_stamps(false);
        run_copies(&rt, 3);
        rt.synchronize().unwrap();
        assert!(rt.take_dispatch_stamps().is_none());

        overflow_is_reported_not_hidden();
    }

    /// More dispatches than the heap holds: a prefix comes back, flagged.
    fn overflow_is_reported_not_hidden() {
        crate::ab_flags::set_dispatch_stamps(true);
        let rt = GpuRuntime::new().expect("runtime");
        rt.set_async_encode(true).unwrap();
        run_copies(&rt, STAMP_HEAP_SLOTS + 10);
        rt.synchronize().unwrap();
        let batch = rt.take_dispatch_stamps().expect("armed").expect("resolved");
        assert!(batch.overflowed);
        assert!(batch.stamps.len() < STAMP_HEAP_SLOTS);
        crate::ab_flags::set_dispatch_stamps(false);
    }
}

/// Device-free: the counters are plain integers, so these run in the hosted pass.
#[cfg(test)]
mod counter_tests {
    use super::*;

    fn batch(dropped_batches: u32, command_buffers: u32) -> DispatchStampBatch {
        DispatchStampBatch {
            freq_hz: 1,
            t0_ns: 0,
            stamps: Vec::new(),
            overflowed: false,
            dropped_batches,
            command_buffers,
        }
    }

    #[test]
    fn dropped_batches_saturate_at_u32_max() {
        let mut log = StampLog {
            dropped_batches: u32::MAX - 1,
            ..StampLog::default()
        };
        log.note_dropped();
        assert_eq!(log.dropped_batches, u32::MAX);
        log.note_dropped();
        log.note_dropped();
        assert_eq!(log.dropped_batches, u32::MAX, "a saturated count stays a floor");
    }

    #[test]
    fn merge_saturates_both_counters_at_u32_max() {
        let mut a = batch(u32::MAX - 1, u32::MAX - 1);
        a.merge(batch(5, 5));
        assert_eq!(a.dropped_batches, u32::MAX);
        assert_eq!(a.command_buffers, u32::MAX);
    }

    #[test]
    fn merge_below_the_limit_is_still_exact() {
        let mut a = batch(2, 1);
        a.merge(batch(3, 4));
        assert_eq!((a.dropped_batches, a.command_buffers), (5, 5));
    }
}
