//! Library paths that write a small host-visible buffer report a busy runtime
//! (a live `HostMapping`) as an error instead of panicking, and the runtime is
//! usable once the mapping is dropped.

mod common;

use common::with_gpu;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use tessl::decode_icb::DecodeIcb;
use tessl::embedgemma2::upload_segments;
use tessl::infer_trace;
use tessl::qwen35_model::{LogitRows, Precision, Qwen35Config, Qwen35Model};
use tessl::safetensors::SafeTensors;
use tessl::tensor::GpuBuffer;
use tessl::GpuRuntime;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train")
}

fn ids(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 37 + 11) % 64).collect()
}

/// `Staged::logits`, `Staged::score_answers` and `forward_rows` (the
/// `read_rows` path) report a busy runtime as an error, and the same session
/// and model work once the mapping is dropped. (These already returned `Err`
/// before this lane: coverage, not a regression for a code change.)
#[test]
fn staged_readbacks_report_a_busy_runtime_instead_of_panicking() {
    let cfg = Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap();
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    let n = cfg.layers.len();
    with_gpu(|rt| {
        let model = Qwen35Model::load(rt, &st, "model.", cfg.clone(), Precision::F32).unwrap();
        let mut s = model.begin(&ids(4)).unwrap();
        s.advance_to(n).unwrap();
        let held = rt.alloc_buffer(64).unwrap();
        let mapping = held.contents_u8();
        let busy = |what: &str, r: std::thread::Result<Result<(), String>>| {
            let e = r
                .unwrap_or_else(|_| panic!("{what} panicked on a busy runtime"))
                .unwrap_err();
            assert!(e.contains("busy"), "{what}: {e}");
        };
        busy(
            "logits",
            catch_unwind(AssertUnwindSafe(|| s.logits(LogitRows::Last).map(|_| ()))),
        );
        busy(
            "score_answers",
            catch_unwind(AssertUnwindSafe(|| s.score_answers(&[0], &[1, 2]).map(|_| ()))),
        );
        busy(
            "forward_rows",
            catch_unwind(AssertUnwindSafe(|| {
                model.forward_rows(&ids(4), LogitRows::Last, false).map(|_| ())
            })),
        );
        drop(mapping);
        assert!(!s.logits(LogitRows::Last).unwrap().is_empty());
        assert!(s.score_answers(&[0], &[1, 2]).is_ok());
        assert!(model.forward_rows(&ids(4), LogitRows::Last, false).is_ok());
    });
}

#[test]
fn upload_segments_reports_a_busy_runtime_instead_of_panicking() {
    with_gpu(|rt| {
        let held = rt.alloc_buffer(64).unwrap();
        let mapping = held.contents_u8();
        let outcome = catch_unwind(AssertUnwindSafe(|| upload_segments(rt, &[(0, 2), (2, 5)], 5)));
        let err = outcome
            .expect("upload_segments panicked on a busy runtime")
            .unwrap_err();
        assert!(err.contains("busy") && err.contains("upload_segments"), "{err}");
        drop(mapping);
        let ok = upload_segments(rt, &[(0, 2), (2, 5)], 5).unwrap();
        assert_eq!(ok.read_u32(), vec![0, 2, 2, 5]);
    });
}

/// An `n` that cannot be a u32 is refused by name before anything is
/// allocated (a 16 GiB-per-buffer allocation would otherwise be attempted).
#[test]
fn mini_copy_chain_refuses_an_n_past_u32_before_allocating() {
    with_gpu(|rt| {
        let n = u32::MAX as usize + 1;
        let err = DecodeIcb::mini_copy_chain(rt, n).map(|_| ()).unwrap_err();
        assert!(err.contains("mini_copy_chain") && err.contains("u32"), "{err}");
        let err = DecodeIcb::mini_copy_chain(rt, usize::MAX).map(|_| ()).unwrap_err();
        assert!(err.contains("u32"), "{err}");
        assert!(DecodeIcb::mini_copy_chain(rt, 0).is_err());
    });
}

#[test]
fn mini_copy_chain_reports_a_busy_runtime_instead_of_panicking() {
    with_gpu(|rt| {
        let held = rt.alloc_buffer(64).unwrap();
        let mapping = held.contents_u8();
        let outcome = catch_unwind(AssertUnwindSafe(|| DecodeIcb::mini_copy_chain(rt, 8).map(|_| ())));
        let err = outcome
            .expect("mini_copy_chain panicked on a busy runtime")
            .unwrap_err();
        assert!(err.contains("busy"), "{err}");
        drop(mapping);
        let (mut dec, out) = DecodeIcb::mini_copy_chain(rt, 8).unwrap();
        dec.execute(rt).unwrap();
        rt.synchronize().unwrap();
        assert_eq!(out.read_f32(), (1..=8).map(|i| i as f32).collect::<Vec<_>>());
    });
}

#[test]
fn bert_upload_segments_reports_a_busy_runtime_instead_of_panicking() {
    with_gpu(|rt| {
        let held = rt.alloc_buffer(64).unwrap();
        let mapping = held.contents_u8();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            tessl::bert::upload_segments(rt, &[(0, 2), (2, 5)], 5)
        }));
        let err = outcome
            .expect("bert::upload_segments panicked on a busy runtime")
            .unwrap_err();
        assert!(err.contains("busy") && err.contains("upload_segments"), "{err}");
        drop(mapping);
        let ok = tessl::bert::upload_segments(rt, &[(0, 2), (2, 5)], 5).unwrap();
        assert_eq!(ok.read_u32(), vec![0, 2, 2, 5]);
        assert!(rt.synchronize().is_ok(), "the runtime is poisoned");
    });
}

/// What `f` counted. The snapshot is taken before `f`'s result is looked at,
/// and tracing is off again before any assertion can fail.
fn traced<T>(f: impl FnOnce() -> T) -> (T, infer_trace::Snapshot) {
    infer_trace::set_enabled(true);
    let s0 = infer_trace::snapshot();
    let out = f();
    let s1 = infer_trace::snapshot();
    infer_trace::set_enabled(false);
    (out, s1.since(&s0))
}

/// With async encode on, leaves an encoded, uncommitted batch that binds the
/// returned buffer: any host access to an ordinary buffer now waits on it.
fn open_batch(rt: &GpuRuntime) -> GpuBuffer {
    rt.set_async_encode(true).unwrap();
    let dst = rt.alloc_buffer(16).unwrap();
    rt.upload_u32(&dst, &[1, 2, 3, 4]).unwrap();
    dst
}

fn close_batch(rt: &GpuRuntime) {
    rt.synchronize().unwrap();
    rt.set_async_encode(false).unwrap();
}

/// A host write into a buffer the open batch binds waits: the counter this
/// test reads sees a waiting host write, so a zero elsewhere is not vacuous.
fn assert_counter_sees_a_waiting_write(rt: &GpuRuntime) {
    let bound = open_batch(rt);
    let (r, d) = traced(|| bound.try_write_u32(&[9, 9, 9, 9]));
    r.unwrap();
    assert!(
        d.sync_waits >= 1,
        "positive control: a bound host write counted no wait"
    );
    close_batch(rt);
}

/// The id/segment/count uploads of the library paths never wait for the GPU
/// under async encode: each is a fresh buffer written under an access lease,
/// not a host write into a buffer the open batch may bind. (Reverting any of
/// them to `alloc_buffer` + `write_u32` counts a wait here.)
#[test]
fn upload_paths_do_not_wait_under_async_encode() {
    assert!(
        std::env::var_os("TESSL_MID_COMMIT").is_none(),
        "TESSL_MID_COMMIT changes the commit and wait counts"
    );
    with_gpu(|rt| {
        assert_counter_sees_a_waiting_write(rt);

        let _batch = open_batch(rt);
        let (r, d) = traced(|| upload_segments(rt, &[(0, 2), (2, 5)], 5).map(|b| b.nbytes()));
        assert_eq!(
            (r.unwrap(), d.sync_waits, d.commits),
            (16, 0, 0),
            "embedgemma2::upload_segments: {d:?}"
        );

        let (r, d) = traced(|| tessl::bert::upload_segments(rt, &[(0, 2), (2, 5)], 5).map(|b| b.nbytes()));
        assert_eq!(
            (r.unwrap(), d.sync_waits, d.commits),
            (16, 0, 0),
            "bert::upload_segments: {d:?}"
        );

        let (r, d) = traced(|| DecodeIcb::mini_copy_chain(rt, 8).map(|_| ()));
        r.unwrap();
        assert_eq!((d.sync_waits, d.commits), (0, 0), "mini_copy_chain (n_buf): {d:?}");
        close_batch(rt);
    });
}

type Upload = fn(&std::sync::Arc<GpuRuntime>, &[(u32, u32)], u32) -> Result<GpuBuffer, String>;

fn both_uploads() -> [(&'static str, Upload); 2] {
    [
        ("embedgemma2", upload_segments as Upload),
        ("bert", tessl::bert::upload_segments as Upload),
    ]
}

/// Hostile segment lists are refused by name (never a panic), an empty list
/// still yields a readable buffer, and extreme u32 bounds round-trip.
#[test]
fn upload_segments_handles_hostile_segment_lists_as_values() {
    with_gpu(|rt| {
        for (name, up) in both_uploads() {
            for (segs, rows) in [
                (vec![(3u32, 3u32)], 10u32),
                (vec![(5, 2)], 10),
                (vec![(0, 11)], 10),
                (vec![(0, 2), (u32::MAX, u32::MAX)], u32::MAX),
                (vec![(0, 1)], 0),
            ] {
                let r = catch_unwind(AssertUnwindSafe(|| up(rt, &segs, rows)))
                    .unwrap_or_else(|_| panic!("{name} panicked on {segs:?}/{rows}"));
                let e = r.map(|_| ()).unwrap_err();
                assert!(e.contains("upload_segments"), "{name}: {e}");
            }
            let empty = up(rt, &[], 0).unwrap_or_else(|e| panic!("{name} empty: {e}"));
            assert!(empty.nbytes() >= 4, "{name}: empty buffer is {} bytes", empty.nbytes());
            let max = up(rt, &[(u32::MAX - 1, u32::MAX)], u32::MAX).unwrap();
            assert_eq!(max.read_u32(), vec![u32::MAX - 1, u32::MAX], "{name}");
            let many: Vec<(u32, u32)> = (0..50_000u32).map(|i| (i, i + 1)).collect();
            let big = up(rt, &many, 50_000).unwrap();
            let got = big.read_u32();
            assert_eq!(got.len(), 100_000, "{name}");
            assert_eq!(
                (got[0], got[1], got[99_998], got[99_999]),
                (0, 1, 49_999, 50_000),
                "{name}"
            );
        }
        assert!(rt.synchronize().is_ok(), "runtime poisoned by hostile inputs");
    });
}

/// A busy refusal while an async batch is open neither loses nor reorders
/// the queued work: the batch's upload still lands, and nothing is poisoned.
#[test]
fn busy_refusal_under_an_open_async_batch_keeps_the_batch_and_the_runtime() {
    with_gpu(|rt| {
        let dst = open_batch(rt);
        let held = rt.alloc_buffer(64).unwrap();
        let mapping = held.contents_u8();
        for (name, up) in both_uploads() {
            let (r, d) = traced(|| catch_unwind(AssertUnwindSafe(|| up(rt, &[(0, 2)], 2))));
            let e = r.unwrap_or_else(|_| panic!("{name} panicked")).map(|_| ()).unwrap_err();
            assert!(e.contains("busy"), "{name}: {e}");
            assert_eq!((d.sync_waits, d.commits), (0, 0), "{name} waited while refusing: {d:?}");
        }
        let r = catch_unwind(AssertUnwindSafe(|| DecodeIcb::mini_copy_chain(rt, 4).map(|_| ())));
        assert!(r.expect("mini_copy_chain panicked").unwrap_err().contains("busy"));
        drop(mapping);
        close_batch(rt);
        assert_eq!(dst.read_u32(), vec![1, 2, 3, 4], "queued upload was lost");
        assert_eq!(upload_segments(rt, &[(0, 2)], 2).unwrap().read_u32(), vec![0, 2]);
    });
}

/// Repeated busy/ok cycles (a mapping taken and dropped, uploads refused in
/// between) never leave the access flag set or poison the runtime.
#[test]
fn repeated_busy_cycles_recover_every_time() {
    with_gpu(|rt| {
        for round in 0..50 {
            let held = rt.alloc_buffer(64).unwrap();
            let mapping = held.contents_u8();
            for r in [
                upload_segments(rt, &[(0, 2)], 2).map(|_| ()),
                tessl::bert::upload_segments(rt, &[(0, 2)], 2).map(|_| ()),
                DecodeIcb::mini_copy_chain(rt, 2).map(|_| ()),
            ] {
                assert!(r.unwrap_err().contains("busy"), "round {round}");
            }
            drop(mapping);
            drop(held);
            assert!(upload_segments(rt, &[(0, 2)], 2).is_ok(), "round {round}");
        }
        assert!(rt.synchronize().is_ok());
    });
}

/// `n == 1` is the smallest chain: it still builds under an open async batch
/// without a wait and runs to the right value.
#[test]
fn mini_copy_chain_of_one_does_not_wait_and_runs() {
    with_gpu(|rt| {
        let _batch = open_batch(rt);
        let (r, d) = traced(|| DecodeIcb::mini_copy_chain(rt, 1));
        let (mut dec, out) = r.unwrap();
        assert_eq!((d.sync_waits, d.commits), (0, 0), "{d:?}");
        dec.execute(rt).unwrap();
        close_batch(rt);
        assert_eq!(out.read_f32(), vec![1.0]);
    });
}

/// A host read that fails after the work ran and the runtime synchronized is an
/// `Err` from every readback of the staged and decode entry points, never a
/// panic; a failed `Decode::step` leaves `position()` and the logits it would
/// have returned unconsumed (the session is poisoned, since its layers'
/// state moved on), and the read-only entry points work again afterwards.
#[test]
fn a_failed_host_read_is_an_error_and_a_failed_step_does_not_advance() {
    let cfg = Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap();
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    let layers = cfg.layers.len();
    with_gpu(|rt| {
        let model = Qwen35Model::load(rt, &st, "model.", cfg.clone(), Precision::F32).unwrap();
        let faulted = |what: &str, r: std::thread::Result<Result<(), String>>| {
            let e = r
                .unwrap_or_else(|_| panic!("{what} panicked on a failed host read"))
                .unwrap_err();
            assert!(e.contains("injected fault"), "{what}: {e}");
        };

        // Staged::logits and score_answers: read-only, so the session survives.
        let mut s = model.begin(&ids(4)).unwrap();
        s.advance_to(layers).unwrap();
        let want = s.logits(LogitRows::Last).unwrap();
        model.fail_host_read_after_for_test(0);
        faulted(
            "logits",
            catch_unwind(AssertUnwindSafe(|| s.logits(LogitRows::Last).map(|_| ()))),
        );
        assert_eq!(s.logits(LogitRows::Last).unwrap(), want, "the fault was one-shot");
        model.fail_host_read_after_for_test(0);
        faulted(
            "score_answers",
            catch_unwind(AssertUnwindSafe(|| s.score_answers(&[0], &[1, 2]).map(|_| ()))),
        );
        assert!(s.score_answers(&[0], &[1, 2]).is_ok());

        // Staged::advance_to's trace (reached through forward): the read after
        // the second layer fails.
        model.fail_host_read_after_for_test(1);
        faulted(
            "advance_to trace",
            catch_unwind(AssertUnwindSafe(|| model.forward(&ids(4), true).map(|_| ()))),
        );

        // forward's final-norm trace row (`read_rows`): after one read per
        // layer and the logits.
        model.fail_host_read_after_for_test(layers + 1);
        faulted(
            "forward trace row",
            catch_unwind(AssertUnwindSafe(|| model.forward(&ids(4), true).map(|_| ()))),
        );
        assert!(model.forward(&ids(4), true).is_ok(), "the fault was one-shot");

        // Decode::step reads before it advances.
        let (mut d, _) = model.prefill(&ids(4), 3).unwrap();
        let (pos, left) = (d.position(), d.remaining());
        model.fail_host_read_after_for_test(0);
        faulted("Decode::step", catch_unwind(AssertUnwindSafe(|| d.step(5).map(|_| ()))));
        assert_eq!(d.position(), pos, "a failed read advanced the session");
        assert_eq!(d.remaining(), left);
        let e = d.step(5).unwrap_err();
        assert!(e.contains("failed part way"), "the session must stay poisoned: {e}");
        // A fresh session on the same model decodes normally.
        let (mut d, _) = model.prefill(&ids(4), 3).unwrap();
        assert_eq!(d.step(5).unwrap().len(), cfg.vocab as usize);
        assert_eq!(d.position(), pos + 1);
    });
}

/// `Qwen35Model::prefill` and `Decode::step` under a live `HostMapping`
/// return a busy error rather than panicking, and a refused step leaves the
/// session usable: the refusal comes before any layer state moves, so it must
/// not poison it or advance its position.
#[test]
fn decode_prefill_and_step_report_a_busy_runtime_without_poisoning() {
    let cfg = Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap();
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    with_gpu(|rt| {
        let model = Qwen35Model::load(rt, &st, "model.", cfg.clone(), Precision::F32).unwrap();
        let (mut d, _) = model.prefill(&ids(4), 3).unwrap();
        let (pos, left) = (d.position(), d.remaining());
        let held = rt.alloc_buffer(64).unwrap();
        let mapping = held.contents_u8();
        let r = catch_unwind(AssertUnwindSafe(|| d.step(5).map(|_| ())));
        let e = r.expect("Decode::step panicked on a busy runtime").unwrap_err();
        assert!(e.contains("busy"), "{e}");
        let r = catch_unwind(AssertUnwindSafe(|| model.prefill(&ids(4), 3).map(|_| ())));
        let e = r.expect("prefill panicked on a busy runtime").unwrap_err();
        assert!(e.contains("busy"), "{e}");
        let r = catch_unwind(AssertUnwindSafe(|| model.forward(&ids(4), true).map(|_| ())));
        let e = r.expect("forward panicked on a busy runtime").unwrap_err();
        assert!(e.contains("busy"), "{e}");
        assert_eq!((d.position(), d.remaining()), (pos, left), "a refused step advanced");
        drop(mapping);
        let row = d.step(5).expect("a step refused as busy poisoned the session");
        assert_eq!(row.len(), cfg.vocab as usize);
        assert_eq!(d.position(), pos + 1);
        assert!(model.forward(&ids(4), true).is_ok());
    });
}

/// A decode that hits a read fault on step k, replayed on a fresh session,
/// gives bit-identical logits to an undisturbed run: the fault leaves nothing
/// behind in the model (shared weights, scratch) that a new session inherits.
#[test]
fn a_read_fault_leaves_no_state_in_the_model_for_later_sessions() {
    let cfg = Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap();
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    with_gpu(|rt| {
        let model = Qwen35Model::load(rt, &st, "model.", cfg.clone(), Precision::F32).unwrap();
        let run = || {
            let (mut d, first) = model.prefill(&ids(5), 3).unwrap();
            let a = d.step(7).unwrap();
            let b = d.step(9).unwrap();
            (first, a, b)
        };
        let want = run();
        let (mut d, _) = model.prefill(&ids(5), 3).unwrap();
        d.step(7).unwrap();
        model.fail_host_read_after_for_test(0);
        assert!(d.step(9).unwrap_err().contains("injected fault"));
        drop(d);
        let got = run();
        assert_eq!(got, want, "a failed session changed what a later one computes");
        // Disarming with usize::MAX leaves a clean run.
        model.fail_host_read_after_for_test(usize::MAX);
        assert_eq!(run(), want);
    });
}
