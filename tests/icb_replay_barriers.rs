//! A tape's `barrier_after` markers are what replay emits, and the last
//! command's pending hazard still orders the runtime's next dispatch.
//!
//! The barrier counter ([`tessl::infer_trace`]) is process-wide, so every test
//! here holds [`LOCK`] while it runs and this file owns its own process.

use std::sync::Mutex;

use tessl::runtime::mtl_size;
use tessl::{infer_trace, pipeline_icb, DecodeIcb, DecodeIcbBind, DecodeIcbCommand, GpuRuntime};

static LOCK: Mutex<()> = Mutex::new(());

const N: usize = 32;
const CMDS: usize = 4;

/// What `f` counted.
fn traced<T>(f: impl FnOnce() -> T) -> (T, infer_trace::Snapshot) {
    infer_trace::set_enabled(true);
    let s0 = infer_trace::snapshot();
    let out = f();
    let s1 = infer_trace::snapshot();
    infer_trace::set_enabled(false);
    (out, s1.since(&s0))
}

/// `CMDS` independent `copy_f32` commands (disjoint buffers, no RAW edge), with
/// the given `barrier_after` markers.
fn tape(rt: &GpuRuntime, markers: [bool; CMDS]) -> DecodeIcb {
    let pipe = pipeline_icb(rt, "copy_f32").expect("copy_f32 icb pipeline");
    let n_buf = rt.alloc_buffer_hot(4).unwrap();
    n_buf.write_u32(&[N as u32]);
    // One thread per group: the kernel only needs every element covered.
    let (tg, tptg) = (mtl_size(N, 1, 1), mtl_size(1, 1, 1));
    let bind = |index, buf: &tessl::tensor::GpuBuffer| DecodeIcbBind::Buf {
        index,
        buf: buf.clone(),
        byte_offset: 0,
        gpu_addr: 0,
    };
    let cmds = markers
        .iter()
        .map(|&barrier_after| {
            let src = rt.alloc_buffer_hot(N * 4).unwrap();
            let dst = rt.alloc_buffer_hot(N * 4).unwrap();
            DecodeIcbCommand {
                pipeline: pipe.clone(),
                threadgroups: tg,
                threads_per_tg: tptg,
                binds: vec![bind(0, &src), bind(1, &dst), bind(2, &n_buf)],
                tg_mem: None,
                owned_immediates: Vec::new(),
                barrier_after,
                incomplete_binds: 0,
            }
        })
        .collect();
    DecodeIcb::from_commands_ex(rt, cmds, false).expect("tape")
}

fn replay_barriers(markers: [bool; CMDS]) -> u64 {
    let rt = GpuRuntime::new().expect("runtime");
    let mut dec = tape(&rt, markers);
    let (res, snap) = traced(|| dec.execute(&rt));
    res.expect("execute");
    rt.synchronize().unwrap();
    assert_eq!(snap.dispatches, CMDS as u64, "one dispatch per tape command");
    snap.barriers
}

/// Against the pre-fix code every scope's dispatch leaves `hazard_pending` set,
/// so replay emitted `CMDS - 1` barriers whatever the markers said.
#[test]
fn replay_emits_exactly_the_captured_barrier_markers() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(
        replay_barriers([false; CMDS]),
        0,
        "no marker, no barrier (before the fix: CMDS - 1)"
    );
    assert_eq!(replay_barriers([true, false, true, false]), 2);
    assert_eq!(replay_barriers([true, true, true, true]), 4);
}

/// The unmarked last command must still be ordered before whatever the runtime
/// encodes next: dropping its pending hazard would be a missing barrier.
#[test]
fn the_last_unmarked_command_still_orders_the_next_scope() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let live = |rt: &GpuRuntime| {
        let p = rt.pipeline("copy_f32").unwrap();
        let (res, snap) = traced(|| {
            let a = rt.alloc_buffer_hot(N * 4).unwrap();
            let b = rt.alloc_buffer_hot(N * 4).unwrap();
            let n = rt.alloc_buffer_hot(4).unwrap();
            n.write_u32(&[N as u32]);
            tessl::dispatch::dispatch_1d(rt, &p, N, |bnd| {
                tessl::dispatch::set_gpu_buf(bnd, &a, 0);
                tessl::dispatch::set_gpu_buf(bnd, &b, 1);
                tessl::dispatch::set_gpu_buf(bnd, &n, 2);
            })
        });
        res.unwrap();
        rt.synchronize().unwrap();
        snap.barriers
    };
    let baseline = live(&GpuRuntime::new().unwrap());

    let rt = GpuRuntime::new().unwrap();
    let mut dec = tape(&rt, [false; CMDS]);
    dec.execute(&rt).unwrap();
    let after_tape = live(&rt);
    assert_eq!(
        after_tape,
        baseline + 1,
        "the tape's unbarriered last command must be ordered before the next dispatch"
    );
}
