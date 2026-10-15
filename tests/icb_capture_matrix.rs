//! Capture-mode matrix: a tape captured under every combination of hazard mode
//! (on/off) and encode mode (sync/async) replays bit-identically to the live
//! run, 40 times, through both replay shapes (plain and freeze-binds +
//! range-batch), for two chains:
//!
//! * `softcap`: 24 in-place elementwise ops (each edge is RAW + WAR + WAW on one
//!   buffer, with a different cap each so order matters);
//! * `rms`: 24 ping-pong `rms_norm_f32` ops on one row, a cross-element RAW
//!   chain where an overlapped dispatch reads half-written data.
//!
//! Replay emits only the barriers the capture recorded, so a capture that lets
//! a scope boundary go unrecorded (sync encode: each scope is its own command
//! buffer and live needs no barrier) races on replay. Each cell is its own test.
//!
//! The process-global flags are shared, so every test holds [`LOCK`] and this
//! file owns its process.

use std::sync::{Arc, Mutex};

use tessl::decode_icb::set_icb_pipelines;
use tessl::nn;
use tessl::tensor::GpuBuffer;
use tessl::{DecodeIcb, GpuRuntime};

static LOCK: Mutex<()> = Mutex::new(());

const CHAIN: usize = 24;
const REPLAYS: usize = 40;
const SOFTCAP_N: usize = 1 << 22;
const RMS_DIM: usize = 1 << 18;

struct Flags;
impl Drop for Flags {
    fn drop(&mut self) {
        tessl::end_decode_icb_capture();
        set_icb_pipelines(false);
        tessl::set_icb_freeze_binds(false);
        tessl::set_icb_range_batch(false);
        tessl::set_icb_coarse_ranges(false);
        tessl::ab_flags::set_hazard_barriers(false);
    }
}

#[derive(Clone, Copy)]
struct Cell {
    hazard: bool,
    async_encode: bool,
    /// Replay through freeze-binds + range-batch (needs ICB-capable pipelines).
    freeze_range: bool,
}

/// Capture `ops` (the chain's encode calls), replay `REPLAYS` times, and count
/// the replays whose `out` differs bitwise from the live result. Returns the
/// markers too, so a failure shows which edge was dropped.
fn run(cell: Cell, rt_setup: impl FnOnce(&Arc<GpuRuntime>) -> Chain) -> (Vec<bool>, usize) {
    let _f = Flags;
    tessl::ab_flags::set_hazard_barriers(cell.hazard);
    set_icb_pipelines(cell.freeze_range);
    let rt = GpuRuntime::new().expect("runtime");
    rt.set_async_encode(cell.async_encode).unwrap();
    let chain = rt_setup(&rt);
    rt.synchronize().unwrap();
    tessl::end_decode_icb_capture();
    tessl::begin_decode_icb_capture();
    let results = (chain.encode)(&rt);
    let capture = tessl::take_decode_icb_capture().expect("capture");
    results.into_iter().for_each(|r| r.expect("live encode"));
    rt.synchronize().unwrap();
    let live = chain.out.read_f32();
    assert_eq!(capture.commands.len(), CHAIN);
    let marks: Vec<bool> = capture.commands.iter().map(|c| c.barrier_after).collect();
    tessl::set_icb_freeze_binds(cell.freeze_range);
    tessl::set_icb_range_batch(cell.freeze_range);
    tessl::set_icb_coarse_ranges(false);
    let mut icb = DecodeIcb::from_commands(&rt, capture.commands).expect("tape");
    let mut bad = 0;
    for _ in 0..REPLAYS {
        chain.out.write_f32(&chain.seed);
        rt.synchronize().unwrap();
        icb.execute(&rt).expect("execute");
        rt.synchronize().unwrap();
        if chain
            .out
            .read_f32()
            .iter()
            .zip(&live)
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            bad += 1;
        }
    }
    drop(chain.keep);
    (marks, bad)
}

type Encode = Box<dyn Fn(&Arc<GpuRuntime>) -> Vec<Result<(), String>>>;

struct Chain {
    out: GpuBuffer,
    seed: Vec<f32>,
    encode: Encode,
    keep: Vec<GpuBuffer>,
}

fn softcap(rt: &Arc<GpuRuntime>) -> Chain {
    let seed: Vec<f32> = (0..SOFTCAP_N)
        .map(|i| ((i * 7919) % 2003) as f32 * 0.05 - 50.0)
        .collect();
    let logits = rt.alloc_buffer_hot(SOFTCAP_N * 4).unwrap();
    logits.write_f32(&seed);
    let caps: Vec<GpuBuffer> = (0..CHAIN)
        .map(|i| {
            let c = rt.alloc_buffer_hot(4).unwrap();
            c.write_f32(&[10.0 + 3.0 * i as f32]);
            c
        })
        .collect();
    let (l, cs) = (logits.clone(), caps.clone());
    Chain {
        out: logits,
        seed,
        encode: Box::new(move |rt| {
            cs.iter()
                .map(|c| nn::softcap_logits(rt, &l, c, SOFTCAP_N as u32))
                .collect()
        }),
        keep: caps,
    }
}

fn rms(rt: &Arc<GpuRuntime>) -> Chain {
    let x = rt.alloc_buffer_hot(RMS_DIM * 4).unwrap();
    let y = rt.alloc_buffer_hot(RMS_DIM * 4).unwrap();
    let w: Vec<GpuBuffer> = (0..CHAIN)
        .map(|i| {
            let b = rt.alloc_buffer_hot(RMS_DIM * 4).unwrap();
            b.write_f32(&vec![0.5 + 0.05 * i as f32; RMS_DIM]);
            b
        })
        .collect();
    let seed: Vec<f32> = (0..RMS_DIM).map(|i| ((i * 7919) % 2003) as f32 * 0.01 - 10.0).collect();
    x.write_f32(&seed);
    let (xc, yc, wc) = (x.clone(), y.clone(), w.clone());
    Chain {
        out: x,
        seed,
        encode: Box::new(move |rt| {
            (0..CHAIN)
                .map(|i| {
                    let (src, dst) = if i % 2 == 0 { (&xc, &yc) } else { (&yc, &xc) };
                    nn::rms_norm_f32(rt, src, &wc[i], dst, 1, RMS_DIM as u32, 1e-6)
                })
                .collect()
        }),
        keep: [w, vec![y]].concat(),
    }
}

fn check(cell: Cell, chain: fn(&Arc<GpuRuntime>) -> Chain) {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (marks, bad) = run(cell, chain);
    let what = format!(
        "hazard={} async={} freeze_range={}",
        cell.hazard, cell.async_encode, cell.freeze_range
    );
    assert!(
        marks[..CHAIN - 1].iter().all(|&m| m),
        "{what}: capture dropped a RAW edge: {marks:?}"
    );
    assert_eq!(bad, 0, "{what}: {bad}/{REPLAYS} replays differ from live");
}

macro_rules! cell {
    ($name:ident, $chain:ident, $h:expr, $a:expr, $f:expr) => {
        #[test]
        fn $name() {
            check(
                Cell {
                    hazard: $h,
                    async_encode: $a,
                    freeze_range: $f,
                },
                $chain,
            );
        }
    };
}

cell!(softcap_plain_always_on_sync, softcap, false, false, false);
cell!(softcap_plain_always_on_async, softcap, false, true, false);
cell!(softcap_plain_hazard_sync, softcap, true, false, false);
cell!(softcap_plain_hazard_async, softcap, true, true, false);
cell!(softcap_freeze_always_on_sync, softcap, false, false, true);
cell!(softcap_freeze_always_on_async, softcap, false, true, true);
cell!(softcap_freeze_hazard_sync, softcap, true, false, true);
cell!(softcap_freeze_hazard_async, softcap, true, true, true);
cell!(rms_plain_always_on_sync, rms, false, false, false);
cell!(rms_plain_always_on_async, rms, false, true, false);
cell!(rms_plain_hazard_sync, rms, true, false, false);
cell!(rms_plain_hazard_async, rms, true, true, false);
cell!(rms_freeze_always_on_sync, rms, false, false, true);
cell!(rms_freeze_always_on_async, rms, false, true, true);
cell!(rms_freeze_hazard_sync, rms, true, false, true);
cell!(rms_freeze_hazard_async, rms, true, true, true);
