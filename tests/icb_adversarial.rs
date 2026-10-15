//! Adversarial checks of DecodeIcb tape capture and replay (wave 7, ft-9ac2a619).
//!
//! Replay emits only the barriers the capture recorded, so a capture that
//! under-records a hazard would race on replay. These tests capture a chain of
//! in-place ops (every edge is RAW + WAR + WAW on one buffer, so a missing
//! barrier changes the numbers), replay it many times, and compare each replay
//! to the live result bit for bit.
//!
//! The process-global flags (hazard mode, ICB pipelines, ICB execute)
//! are shared, so every test holds [`LOCK`] and this file owns its process.

use std::sync::{Arc, Mutex};

use tessl::decode_icb::set_icb_pipelines;
use tessl::nn;
use tessl::tensor::GpuBuffer;
use tessl::{DecodeIcb, DecodeIcbCapture, GpuRuntime};

static LOCK: Mutex<()> = Mutex::new(());

const N: usize = 1 << 22;
const CHAIN: usize = 24;
const REPLAYS: usize = 40;

struct Flags;
impl Drop for Flags {
    fn drop(&mut self) {
        tessl::end_decode_icb_capture();
        set_icb_pipelines(false);
        tessl::set_icb_execute(false);
    }
}

fn input() -> Vec<f32> {
    (0..N).map(|i| ((i * 7919) % 2003) as f32 * 0.05 - 50.0).collect()
}

struct Chain {
    rt: Arc<GpuRuntime>,
    logits: GpuBuffer,
    caps: Vec<GpuBuffer>,
    live: Vec<f32>,
    capture: DecodeIcbCapture,
}

/// Capture `CHAIN` in-place softcaps with different caps (order matters), and
/// record the live result.
fn capture_chain(hazard: bool, async_encode: bool, icb_pipelines: bool) -> Chain {
    tessl::ab_flags::set_hazard_barriers(hazard);
    set_icb_pipelines(icb_pipelines);
    let rt = GpuRuntime::new().expect("runtime");
    rt.set_async_encode(async_encode).unwrap();
    let logits = rt.alloc_buffer_hot(N * 4).unwrap();
    logits.write_f32(&input());
    let caps: Vec<GpuBuffer> = (0..CHAIN)
        .map(|i| {
            let c = rt.alloc_buffer_hot(4).unwrap();
            c.write_f32(&[10.0 + 3.0 * i as f32]);
            c
        })
        .collect();
    rt.synchronize().unwrap();
    tessl::end_decode_icb_capture();
    tessl::begin_decode_icb_capture();
    let results: Vec<_> = caps
        .iter()
        .map(|c| nn::softcap_logits(&rt, &logits, c, N as u32))
        .collect();
    let capture = tessl::take_decode_icb_capture().expect("capture");
    results.into_iter().for_each(|r| r.expect("live softcap"));
    rt.synchronize().unwrap();
    let live = logits.read_f32();
    assert_eq!(capture.commands.len(), CHAIN);
    Chain {
        rt,
        logits,
        caps,
        live,
        capture,
    }
}

/// Attack 2, capture side: every edge of an in-place chain must be marked.
#[test]
fn capture_marks_every_edge_of_an_inplace_chain() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _f = Flags;
    for hazard in [false, true] {
        for async_encode in [false, true] {
            let c = capture_chain(hazard, async_encode, false);
            let marks: Vec<bool> = c.capture.commands.iter().map(|x| x.barrier_after).collect();
            assert!(
                marks[..CHAIN - 1].iter().all(|&m| m),
                "hazard={hazard} async={async_encode}: an in-place RAW/WAW edge was not captured: {marks:?}"
            );
        }
    }
}

fn replay_matches(icb_exec: bool, hazard: bool, async_encode: bool) {
    let c = capture_chain(hazard, async_encode, icb_exec);
    let Chain {
        rt,
        logits,
        caps,
        live,
        capture,
    } = c;
    tessl::set_icb_execute(icb_exec);
    let mut icb = DecodeIcb::from_commands(&rt, capture.commands).expect("tape");
    let mut bad = 0;
    for r in 0..REPLAYS {
        logits.write_f32(&input());
        rt.synchronize().unwrap();
        icb.execute(&rt).expect("execute");
        rt.synchronize().unwrap();
        if logits
            .read_f32()
            .iter()
            .zip(&live)
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            bad += 1;
            eprintln!("replay {r} differs from the live result");
        }
    }
    drop(caps);
    assert_eq!(
        bad, 0,
        "icb_exec={icb_exec} hazard={hazard} async={async_encode}: {bad}/{REPLAYS} replays differ"
    );
}

/// Attack 1: replay is bit-identical to live, repeatedly, across modes.
#[test]
fn replay_is_bit_identical_across_modes_and_runs() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _f = Flags;
    for icb_exec in [false, true] {
        for hazard in [false, true] {
            for async_encode in [false, true] {
                replay_matches(icb_exec, hazard, async_encode);
            }
        }
    }
}

/// Attack 2, replay side: flipping a RAW marker to false is honoured, i.e. it
/// is the capture's job to be right. Informational: prints how often the
/// unmarked tape diverges; asserts only that the marked tape never does.
#[test]
fn an_unmarked_raw_chain_is_the_captures_responsibility() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _f = Flags;
    let c = capture_chain(true, true, false);
    let Chain {
        rt,
        logits,
        live,
        mut capture,
        ..
    } = c;
    for cmd in capture.commands.iter_mut() {
        cmd.barrier_after = false;
    }
    let mut icb = DecodeIcb::from_commands(&rt, capture.commands).expect("tape");
    let mut bad = 0;
    for _ in 0..REPLAYS {
        logits.write_f32(&input());
        rt.synchronize().unwrap();
        icb.execute(&rt).expect("execute");
        rt.synchronize().unwrap();
        if logits
            .read_f32()
            .iter()
            .zip(&live)
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            bad += 1;
        }
    }
    eprintln!("unmarked in-place chain: {bad}/{REPLAYS} replays diverged");
}

const DIM: usize = 1 << 18;

/// A chain with a cross-element dependence (one row, so each norm needs every
/// element the previous one wrote): ping-pong `x -> y -> x ...`. Unlike the
/// elementwise chain above, an overlapped dispatch reads half-written data.
fn rms_chain(hazard: bool, async_encode: bool) -> (Vec<bool>, usize, usize) {
    tessl::ab_flags::set_hazard_barriers(hazard);
    let rt = GpuRuntime::new().expect("runtime");
    rt.set_async_encode(async_encode).unwrap();
    let x = rt.alloc_buffer_hot(DIM * 4).unwrap();
    let y = rt.alloc_buffer_hot(DIM * 4).unwrap();
    let w: Vec<GpuBuffer> = (0..CHAIN)
        .map(|i| {
            let b = rt.alloc_buffer_hot(DIM * 4).unwrap();
            b.write_f32(&vec![0.5 + 0.05 * i as f32; DIM]);
            b
        })
        .collect();
    let seed: Vec<f32> = (0..DIM).map(|i| ((i * 7919) % 2003) as f32 * 0.01 - 10.0).collect();
    x.write_f32(&seed);
    rt.synchronize().unwrap();
    let run = |capture: bool| {
        let results: Vec<_> = (0..CHAIN)
            .map(|i| {
                let (src, dst) = if i % 2 == 0 { (&x, &y) } else { (&y, &x) };
                nn::rms_norm_f32(&rt, src, &w[i], dst, 1, DIM as u32, 1e-6)
            })
            .collect();
        let cap = capture.then(|| tessl::take_decode_icb_capture().expect("capture"));
        results.into_iter().for_each(|r| r.expect("live rms_norm"));
        cap
    };
    tessl::end_decode_icb_capture();
    tessl::begin_decode_icb_capture();
    let capture = run(true).unwrap();
    rt.synchronize().unwrap();
    let live = x.read_f32();
    let marks: Vec<bool> = capture.commands.iter().map(|c| c.barrier_after).collect();
    let mut icb = DecodeIcb::from_commands(&rt, capture.commands).expect("tape");
    let (mut bad, mut runs) = (0, 0);
    for _ in 0..REPLAYS {
        x.write_f32(&seed);
        rt.synchronize().unwrap();
        icb.execute(&rt).expect("execute");
        rt.synchronize().unwrap();
        runs += 1;
        if x.read_f32().iter().zip(&live).any(|(a, b)| a.to_bits() != b.to_bits()) {
            bad += 1;
        }
    }
    (marks, bad, runs)
}

/// The break: a tape captured with sync encode in hazard mode records NO
/// barrier for a true cross-element RAW chain (each sync scope is its own
/// command buffer, so live needs none), but replay packs the whole tape into
/// one encoder, where an unmarked edge is no ordering at all.
/// CURRENTLY FAILS on gp/w7-icb (marks all false); passed before f632bda,
/// where replay re-added a hazard barrier after every dispatch.
#[test]
fn a_sync_hazard_capture_of_a_raw_chain_replays_with_its_barriers() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _f = Flags;
    for (hazard, async_encode) in [(false, false), (true, false), (false, true), (true, true)] {
        let (marks, bad, runs) = rms_chain(hazard, async_encode);
        eprintln!("hazard={hazard} async={async_encode}: markers={marks:?} replays differing: {bad}/{runs}");
        assert!(
            marks[..CHAIN - 1].iter().all(|&m| m),
            "hazard={hazard} async={async_encode}: capture dropped a RAW edge: {marks:?}"
        );
        assert_eq!(
            bad, 0,
            "hazard={hazard} async={async_encode}: replay diverged from live"
        );
    }
}
