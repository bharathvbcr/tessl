//! Ordering contracts for packed multi-dispatch NN operations.

mod common;

use std::sync::Arc;

use common::{empty, with_gpu};
use tessl::nn::{self, AttnDims};
use tessl::{GpuBuffer, GpuRuntime};

fn u32_buf(rt: &Arc<GpuRuntime>, value: u32) -> GpuBuffer {
    let buffer = rt.alloc_buffer(4).expect("allocate u32 device scalar");
    buffer.write_u32(&[value]);
    buffer
}

struct HazardModeRestore(bool);

impl Drop for HazardModeRestore {
    fn drop(&mut self) {
        tessl::ab_flags::set_hazard_barriers(self.0);
        tessl::end_decode_icb_capture();
    }
}

#[test]
fn decode_partial_is_barriered_before_reduce_when_auto_barriers_are_skipped() {
    with_gpu(|rt| {
        tessl::end_decode_icb_capture();
        let previous = tessl::ab_flags::hazard_barriers();
        let _restore = HazardModeRestore(previous);
        tessl::ab_flags::set_hazard_barriers(true);

        const D: usize = 128;
        let q = empty(rt, D);
        let k = empty(rt, D);
        let v = empty(rt, D);
        let out = empty(rt, D);
        let tkv = u32_buf(rt, 1);
        let zero = u32_buf(rt, 0);
        let dims = AttnDims {
            batch: 1,
            tq: 1,
            heads: 1,
            heads_kv: 1,
            window: 0,
            scale: 1.0 / (D as f32).sqrt(),
        };

        tessl::begin_decode_icb_capture();
        let result = nn::flash_attn_decode(
            rt, &q, &k, &v, &out, &tkv, &zero, &zero, dims, D as u32, 1, false,
        );
        let capture = tessl::take_decode_icb_capture().expect("decode capture");
        result.expect("encode decode partial and reduction");

        assert_eq!(capture.commands.len(), 2, "partial + reduce command pair");
        assert!(
            capture.commands[0].barrier_after,
            "partial scratch producer needs a captured RAW barrier before reduce"
        );
        assert!(
            !capture.commands[1].barrier_after,
            "skip-auto mode should not invent an unrelated trailing barrier"
        );
    });
}

fn f32_buf(rt: &Arc<GpuRuntime>, data: &[f32]) -> GpuBuffer {
    let buffer = rt
        .alloc_buffer(data.len() * 4)
        .expect("allocate f32 buffer");
    buffer.write_f32(data);
    buffer
}

/// Consecutive `with_binder` scopes share one Metal 4 encoder in async mode,
/// and in hazard mode nothing inside either scope barriers the edge between
/// them: op A's last dispatch writes a buffer that op B's first dispatch
/// reads. The runtime owns that edge, so it has to emit the barrier itself
/// whenever the previous scope ended with an unbarriered dispatch.
#[test]
fn hazard_mode_barriers_the_edge_between_consecutive_scopes() {
    with_gpu(|rt| {
        tessl::end_decode_icb_capture();
        let previous = tessl::ab_flags::hazard_barriers();
        let _restore = HazardModeRestore(previous);
        tessl::ab_flags::set_hazard_barriers(true);
        rt.set_async_encode(true).unwrap();

        let logits = f32_buf(rt, &[1.0, 2.0, 3.0, 4.0]);
        let cap = f32_buf(rt, &[30.0]);
        tessl::begin_decode_icb_capture();
        // Two single-dispatch ops, each its own scope, both rewriting `logits`.
        let first = nn::softcap_logits(rt, &logits, &cap, 4);
        let second = nn::softcap_logits(rt, &logits, &cap, 4);
        let capture = tessl::take_decode_icb_capture().expect("capture");
        first.expect("first scope");
        second.expect("second scope");
        rt.synchronize().unwrap();

        assert_eq!(capture.commands.len(), 2, "one dispatch per scope");
        assert!(
            capture.commands[0].barrier_after,
            "the write→read edge between two scopes needs a barrier in hazard mode"
        );
        assert!(
            !capture.commands[1].barrier_after,
            "no trailing barrier is invented after the last scope"
        );
    });
}

/// A scope that emits its own explicit barrier (what a caller managing its
/// own edges does) satisfies the pending edge, so the runtime adds nothing on
/// top of it — hazard mode keeps costing one barrier per edge, not two.
#[test]
fn an_explicit_barrier_scope_satisfies_the_pending_edge() {
    with_gpu(|rt| {
        tessl::end_decode_icb_capture();
        let previous = tessl::ab_flags::hazard_barriers();
        let _restore = HazardModeRestore(previous);
        tessl::ab_flags::set_hazard_barriers(true);
        rt.set_async_encode(true).unwrap();

        let logits = f32_buf(rt, &[1.0, 2.0, 3.0, 4.0]);
        let cap = f32_buf(rt, &[30.0]);
        tessl::infer_trace::set_enabled(true);
        // Opens the shared encoder and leaves an unbarriered dispatch behind.
        nn::softcap_logits(rt, &logits, &cap, 4).unwrap();
        tessl::infer_trace::reset_token_counters();
        // Pending edge from the warm-up: one runtime-owned barrier.
        nn::softcap_logits(rt, &logits, &cap, 4).unwrap();
        // Explicit barrier: counts once and clears the pending edge.
        rt.with_binder(|bnd| {
            bnd.barrier();
            Ok(())
        })
        .unwrap();
        // Nothing pending: no barrier at all.
        nn::softcap_logits(rt, &logits, &cap, 4).unwrap();
        let snap = tessl::infer_trace::snapshot();
        tessl::infer_trace::set_enabled(false);
        rt.synchronize().unwrap();

        assert_eq!(snap.dispatches, 2);
        assert_eq!(
            snap.barriers, 2,
            "one runtime-owned edge barrier plus the explicit one, never a doubled edge"
        );
    });
}
