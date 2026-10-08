//! Numeric tests for the Qwen3.5 kernels (`tessl::qwen35`).
//!
//! Three layers of evidence, from the model down:
//!
//! 1. **The references are transformers.** `tests/fixtures/qwen35/` holds goldens
//!    that `scripts/gen_qwen35_fixtures.py` generated from
//!    `transformers.models.qwen3_5` itself. The first tests hold the f64
//!    references in `common::qwen35` to them, with no GPU involved.
//! 2. **The kernels match the references**, on the fixtures and on randomized
//!    shapes chosen for their edges: T = 1, T one either side of the 64-row
//!    chunk, grouped heads, a shared snapshot, strong decay, strided windows.
//! 3. **They compose**: a chunked prefill followed by recurrent decode steps
//!    equals the recurrence over the whole sequence, and likewise for the conv.
//!
//! The same kernels are also checked, off-device, by
//! `tools/msl_emu/check_qwen35.py`, which runs their source on a CPU emulator
//! against transformers directly.
//!
//! The error bound is relative to the output's magnitude: `1e-4 * max|y|`. The
//! kernels land around `1e-6` relative (the emulator measures them next to
//! transformers' own fp32 error), and every defect the mutation run injected
//! moved the output by more than `1e-2` relative, so the bound sits two orders
//! from each.

mod common;

use std::sync::Arc;

use common::qwen35::*;
use common::{buf, buf_u32, random_f32, seeded, with_gpu};
use tessl::qwen35::{
    self, AttnProjLayout, AttnShape, AttnTargets, Cols, GdnDims, GdnGateLogits, GdnParams, GdnProjLayout, GdnQkv,
    GdnScanSlice, GdnWorkspace, LmHead, OutCols, StateIn,
};
use tessl::tensor::{bf16_bits_to_f32, f32_slice_to_bf16, DType, GpuBuffer};
use tessl::{GemmBackend, GpuRuntime};

/// Written to every output before a kernel runs, so an element it never wrote
/// is visible as such.
const SENTINEL: f32 = -7.25e27;

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// `got` within `abs + rel * |want|` elementwise. The shape of error Metal's
/// fast-math `exp`, division and `rsqrt` produce (a few ulps of each result),
/// which a flat absolute bound misjudges at both ends of the range.
fn assert_close_rel(label: &str, got: &[f32], want: &[f64], rel: f64, abs: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g != SENTINEL, "{label}[{i}]: never written");
        assert!(g.is_finite(), "{label}[{i}]: non-finite {g}");
        let bound = abs + rel * w.abs();
        assert!(
            (f64::from(g) - w).abs() <= bound,
            "{label}[{i}]: got {g} want {w} (bound {bound:.3e})"
        );
    }
}

fn assert_close(label: &str, got: &[f32], want: &[f64], atol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut worst = (0.0f64, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g != SENTINEL, "{label}[{i}]: never written");
        assert!(g.is_finite(), "{label}[{i}]: non-finite {g}");
        let e = (f64::from(g) - w).abs();
        if e > worst.0 {
            worst = (e, i);
        }
    }
    assert!(
        worst.0 <= atol,
        "{label}: max err {:.3e} at {} (got {} want {}), bound {atol:.3e}",
        worst.0,
        worst.1,
        got[worst.1],
        want[worst.1]
    );
}

/// The magnitude-relative bound this file holds every GDN output to.
fn rel_bound(want: &[f64]) -> f64 {
    1e-4 * max_abs(want).max(1e-3)
}

fn read_bf16(b: &GpuBuffer, n: usize) -> Vec<f32> {
    b.contents_u16()[..n].iter().map(|&x| bf16_bits_to_f32(x)).collect()
}

// ------------------------------------------------ 1. references vs transformers ---

struct GdnData {
    s: GdnShape,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    state0: Option<Vec<f32>>,
    snapshot: bool,
}

impl GdnData {
    fn problem(&self) -> GdnProblem<'_> {
        GdnProblem {
            s: self.s,
            q: &self.q,
            k: &self.k,
            v: &self.v,
            a: &self.a,
            b: &self.b,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            state0: self.state0.as_deref(),
            snapshot: self.snapshot,
        }
    }

    fn fixture() -> Self {
        let (qs, q) = load_f32("gdn_q");
        let (vs, v) = load_f32("gdn_v");
        Self {
            s: GdnShape {
                b: qs[0],
                t: qs[1],
                hk: qs[2],
                hv: vs[2],
                dv: vs[3],
            },
            q,
            k: load_f32("gdn_k").1,
            v,
            a: load_f32("gdn_a").1,
            b: load_f32("gdn_b").1,
            a_log: load_f32("gdn_a_log").1,
            dt_bias: load_f32("gdn_dt_bias").1,
            state0: Some(load_f32("gdn_state0").1),
            snapshot: false,
        }
    }

    fn random(s: GdnShape, state: &str, seed: u64) -> Self {
        let scale = |v: Vec<f32>, k: f32| v.into_iter().map(|x| x * k).collect::<Vec<_>>();
        let state0 = match state {
            "batch" => Some(scale(random_f32(s.b * s.hv * DK * s.dv, seed + 7), 0.1)),
            "snapshot" => Some(scale(random_f32(s.hv * DK * s.dv, seed + 7), 0.1)),
            _ => None,
        };
        Self {
            s,
            q: random_f32(s.b * s.t * s.hk * DK, seed),
            k: random_f32(s.b * s.t * s.hk * DK, seed + 1),
            v: random_f32(s.b * s.t * s.hv * s.dv, seed + 2),
            a: scale(random_f32(s.b * s.t * s.hv, seed + 3), 2.0),
            b: scale(random_f32(s.b * s.t * s.hv, seed + 4), 2.0),
            // A_log in [-2, 1], dt_bias in [-1, 1].
            a_log: random_f32(s.hv, seed + 5).into_iter().map(|x| -0.5 + 1.5 * x).collect(),
            dt_bias: random_f32(s.hv, seed + 6),
            state0,
            snapshot: state == "snapshot",
        }
    }
}

#[test]
fn gdn_reference_matches_transformers() {
    let d = GdnData::fixture();
    let (y, st) = gdn_f64(&d.problem());
    let (_, y64) = load("gdn_y_f64");
    let (_, st64) = load("gdn_state_f64");
    let (_, yhf) = load("gdn_y_hf");
    let diff = |a: &[f64], b: &[f64]| max_abs(&a.iter().zip(b).map(|(x, y)| x - y).collect::<Vec<_>>());
    // Same arithmetic in the same precision: agreement to rounding.
    assert!(diff(&y, &y64) < 1e-12, "y vs f64 golden: {:e}", diff(&y, &y64));
    assert!(diff(&st, &st64) < 1e-12, "state vs f64 golden: {:e}", diff(&st, &st64));
    // transformers' own fp32 chunked run: measured 2.6e-8 at generation.
    assert!(diff(&y, &yhf) < 1e-6, "y vs transformers fp32: {:e}", diff(&y, &yhf));
    // ...and its final state, which transformers returns as output_final_state.
    let (_, sthf) = load("gdn_state_hf");
    assert!(
        diff(&st, &sthf) < 1e-5,
        "state vs transformers fp32: {:e}",
        diff(&st, &sthf)
    );
}

#[test]
fn rope_reference_matches_transformers() {
    let (qs, q) = load_f32("rope_q_in");
    let (_, k) = load_f32("rope_k_in");
    let (_, qw) = load_f32("rope_q_norm_w");
    let (_, kw) = load_f32("rope_k_norm_w");
    let (_, qo) = load("rope_q_out");
    let (_, ko) = load("rope_k_out");
    let (t, hq, d) = (qs[1], qs[2], qs[3]);
    for ti in 0..t {
        let pos = 20_000 + ti as u64;
        for h in 0..hq {
            let row = (ti * hq + h) * d;
            let got = norm_rope_row_f64(&q[row..row + d], &qw, 64, pos, 1e7, 1e-6);
            for i in 0..d {
                assert!((got[i] - qo[row + i]).abs() < 5e-6, "q t{ti} h{h} [{i}]");
            }
        }
        let got = norm_rope_row_f64(&k[ti * d..(ti + 1) * d], &kw, 64, pos, 1e7, 1e-6);
        for i in 0..d {
            assert!((got[i] - ko[ti * d + i]).abs() < 5e-6, "k t{ti} [{i}]");
        }
    }
}

// -------------------------------------------------------------- GDN on device ---

/// The operands packed as the kernels read them: q|k|v in one strided buffer
/// (as the conv writes them, plus padding columns), a|b in another.
struct Packed {
    qkv: GpuBuffer,
    ld: u32,
    gates: GpuBuffer,
    g_ld: u32,
    a_log: GpuBuffer,
    dt_bias: GpuBuffer,
    key_w: u32,
}

impl Packed {
    fn new(rt: &Arc<GpuRuntime>, d: &GdnData) -> Self {
        let GdnShape { b, t, hk, hv, dv } = d.s;
        let key_w = hk * DK;
        let ld = 1 + 2 * key_w + hv * dv + 4;
        let mut qkv = vec![SENTINEL; b * t * ld];
        for r in 0..b * t {
            let row = &mut qkv[r * ld..(r + 1) * ld];
            row[1..1 + key_w].copy_from_slice(&d.q[r * key_w..(r + 1) * key_w]);
            row[1 + key_w..1 + 2 * key_w].copy_from_slice(&d.k[r * key_w..(r + 1) * key_w]);
            row[1 + 2 * key_w..1 + 2 * key_w + hv * dv].copy_from_slice(&d.v[r * hv * dv..(r + 1) * hv * dv]);
        }
        let g_ld = 1 + 2 * hv + 2;
        let mut gates = vec![SENTINEL; b * t * g_ld];
        for r in 0..b * t {
            let row = &mut gates[r * g_ld..(r + 1) * g_ld];
            row[1..1 + hv].copy_from_slice(&d.a[r * hv..(r + 1) * hv]);
            row[1 + hv..1 + 2 * hv].copy_from_slice(&d.b[r * hv..(r + 1) * hv]);
        }
        Self {
            qkv: buf(rt, &qkv),
            ld: ld as u32,
            gates: buf(rt, &gates),
            g_ld: g_ld as u32,
            a_log: buf(rt, &d.a_log),
            dt_bias: buf(rt, &d.dt_bias),
            key_w: key_w as u32,
        }
    }

    fn qkv(&self) -> GdnQkv<'_> {
        GdnQkv {
            buf: &self.qkv,
            ld: self.ld,
            q_off: 1,
            k_off: 1 + self.key_w,
            v_off: 1 + 2 * self.key_w,
        }
    }

    fn gates(&self, hv: u32) -> GdnGateLogits<'_> {
        GdnGateLogits {
            buf: &self.gates,
            ld: self.g_ld,
            a_off: 1,
            b_off: 1 + hv,
        }
    }

    fn params(&self) -> GdnParams<'_> {
        GdnParams {
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
        }
    }
}

fn dims_of(s: GdnShape) -> GdnDims {
    GdnDims {
        batch: s.b as u32,
        seq: s.t as u32,
        k_heads: s.hk as u32,
        v_heads: s.hv as u32,
        v_dim: s.dv as u32,
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Path {
    Chunk,
    Recurrent,
}

/// Where a GDN call puts its final state.
#[derive(Clone, Copy, PartialEq, Debug)]
enum StateOut {
    /// A buffer of its own.
    Separate,
    /// Nowhere (`state_out = None`): the output must still be right, and the
    /// buffer bound as a placeholder in the state_out slot must be untouched.
    Discard,
    /// Back into the per-batch input state.
    InPlace,
}

/// Run one GDN call; returns (y [B*T, Hv*Dv], final state — empty for
/// [`StateOut::Discard`]).
fn run_gdn_with(rt: &Arc<GpuRuntime>, d: &GdnData, path: Path, mode: StateOut) -> (Vec<f32>, Vec<f32>) {
    run_gdn_lens(rt, d, path, mode, None)
}

/// [`run_gdn_with`], through the `_varlen` entry points when `lens` is given.
///
/// The chunked path runs with both scan slices (`GdnScanSlice`), which must
/// agree bit for bit, so every chunked test holds both kernels.
fn run_gdn_lens(
    rt: &Arc<GpuRuntime>,
    d: &GdnData,
    path: Path,
    mode: StateOut,
    lens: Option<&[u32]>,
) -> (Vec<f32>, Vec<f32>) {
    let base = run_gdn_slice(rt, d, path, mode, lens, GdnScanSlice::Cols32);
    if path == Path::Chunk {
        let narrow = run_gdn_slice(rt, d, path, mode, lens, GdnScanSlice::Cols16);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&narrow.0), bits(&base.0), "16-column scan output vs 32-column");
        assert_eq!(bits(&narrow.1), bits(&base.1), "16-column scan state vs 32-column");
    }
    base
}

fn run_gdn_slice(
    rt: &Arc<GpuRuntime>,
    d: &GdnData,
    path: Path,
    mode: StateOut,
    lens: Option<&[u32]>,
    slice: GdnScanSlice,
) -> (Vec<f32>, Vec<f32>) {
    let s = d.s;
    let dims = dims_of(s);
    let p = Packed::new(rt, d);
    let width = s.hv * s.dv;
    let (out_ld, out_off) = (width + 3, 2usize);
    let out = seeded(rt, s.b * s.t * out_ld, SENTINEL);
    let state_elems = s.b * s.hv * DK * s.dv;
    let separate = seeded(rt, state_elems, SENTINEL);
    let state_buf = d.state0.as_ref().map(|s0| buf(rt, s0));
    let state = match (&state_buf, d.snapshot) {
        (None, _) => StateIn::Zero,
        (Some(b), false) => StateIn::PerBatch(b),
        (Some(b), true) => StateIn::Snapshot(b),
    };
    let state_out = match mode {
        StateOut::Separate => Some(&separate),
        StateOut::Discard => None,
        StateOut::InPlace => Some(state_buf.as_ref().expect("in place needs a per-batch state")),
    };
    let out_cols = Cols {
        buf: &out,
        ld: out_ld as u32,
        off: out_off as u32,
    };
    let lens_buf = lens.map(|l| buf_u32(rt, l));
    let (qkv, gates, params) = (p.qkv(), p.gates(dims.v_heads), p.params());
    match (path, &lens_buf) {
        (Path::Chunk, _) => {
            let ws = GdnWorkspace::new(rt, &dims).unwrap().with_scan_slice(slice);
            if lens_buf.is_some() {
                // A workspace is reused across layers, so a ragged call must
                // not read what an earlier call left there. Fill it with live
                // chunks first: a fresh one is zero, and zero chunks are inert.
                let scratch = seeded(rt, s.b * s.t * width, 0.0);
                qwen35::gdn_chunk_forward(
                    rt,
                    &dims,
                    &qkv,
                    &gates,
                    &params,
                    StateIn::Zero,
                    &ws,
                    Cols::dense(&scratch, width as u32),
                    None,
                )
                .unwrap();
            }
            match &lens_buf {
                None => qwen35::gdn_chunk_forward(rt, &dims, &qkv, &gates, &params, state, &ws, out_cols, state_out),
                Some(l) => qwen35::gdn_chunk_forward_varlen(
                    rt, &dims, &qkv, &gates, &params, state, &ws, out_cols, state_out, l,
                ),
            }
            .unwrap();
        }
        (Path::Recurrent, None) => {
            qwen35::gdn_recurrent(rt, &dims, &qkv, &gates, &params, state, out_cols, state_out).unwrap()
        }
        (Path::Recurrent, Some(l)) => {
            qwen35::gdn_recurrent_varlen(rt, &dims, &qkv, &gates, &params, state, out_cols, state_out, l).unwrap()
        }
    }
    rt.synchronize().unwrap();

    let all = out.read_f32();
    let mut y = Vec::with_capacity(s.b * s.t * width);
    for r in 0..s.b * s.t {
        let row = &all[r * out_ld..(r + 1) * out_ld];
        // Only the window is the kernel's to write.
        assert!(
            row[..out_off]
                .iter()
                .chain(&row[out_off + width..])
                .all(|&x| x == SENTINEL),
            "row {r}: wrote outside the output window"
        );
        y.extend_from_slice(&row[out_off..out_off + width]);
    }
    if mode != StateOut::InPlace {
        if let (Some(b), Some(s0)) = (&state_buf, &d.state0) {
            assert_eq!(&b.read_f32()[..s0.len()], &s0[..], "the input state was written");
        }
    }
    let state = match mode {
        StateOut::Separate => separate.read_f32()[..state_elems].to_vec(),
        StateOut::Discard => {
            assert!(
                separate.read_f32()[..state_elems].iter().all(|&x| x == SENTINEL),
                "a state was written with state_out = None"
            );
            Vec::new()
        }
        StateOut::InPlace => state_buf.unwrap().read_f32()[..state_elems].to_vec(),
    };
    (y, state)
}

fn run_gdn(rt: &Arc<GpuRuntime>, d: &GdnData, path: Path) -> (Vec<f32>, Vec<f32>) {
    run_gdn_with(rt, d, path, StateOut::Separate)
}

fn check_gdn(label: &str, rt: &Arc<GpuRuntime>, d: &GdnData, path: Path) {
    let (want_y, want_s) = gdn_f64(&d.problem());
    let (y, st) = run_gdn(rt, d, path);
    assert_close(&format!("{label} y"), &y, &want_y, rel_bound(&want_y));
    assert_close(&format!("{label} state"), &st, &want_s, rel_bound(&want_s));
}

#[test]
fn gdn_chunk_matches_transformers_golden() {
    with_gpu(|rt| {
        let d = GdnData::fixture();
        let (y, st) = run_gdn(rt, &d, Path::Chunk);
        assert_close("chunk y", &y, &load("gdn_y_f64").1, rel_bound(&load("gdn_y_f64").1));
        assert_close(
            "chunk state",
            &st,
            &load("gdn_state_f64").1,
            rel_bound(&load("gdn_state_f64").1),
        );
    });
}

#[test]
fn gdn_recurrent_matches_transformers_golden() {
    with_gpu(|rt| {
        let d = GdnData::fixture();
        let (y, st) = run_gdn(rt, &d, Path::Recurrent);
        assert_close("rec y", &y, &load("gdn_y_f64").1, rel_bound(&load("gdn_y_f64").1));
        assert_close(
            "rec state",
            &st,
            &load("gdn_state_f64").1,
            rel_bound(&load("gdn_state_f64").1),
        );
    });
}

#[test]
fn gdn_chunk_phases_in_order_are_gdn_chunk_forward_bit_for_bit() {
    // The bench times the chunked rule's two dispatches apart through
    // `gdn_chunk_phase`; that is only an attribution of `gdn_chunk_forward` if
    // Prep then Scan is exactly it, and Prep alone writes no output or state.
    with_gpu(|rt| {
        let s = GdnShape {
            b: 2,
            t: 130,
            hk: 2,
            hv: 4,
            dv: 64,
        };
        let d = GdnData::random(s, "batch", 7300);
        let dims = dims_of(s);
        let p = Packed::new(rt, &d);
        let (qkv, gates, params) = (p.qkv(), p.gates(dims.v_heads), p.params());
        let width = s.hv * s.dv;
        let n_state = s.b * s.hv * DK * s.dv;
        let state_in = buf(rt, d.state0.as_ref().unwrap());
        let run = |phases: &[Option<qwen35::GdnChunkPhase>]| {
            let ws = GdnWorkspace::new(rt, &dims).unwrap();
            let out = seeded(rt, s.b * s.t * width, SENTINEL);
            let st = seeded(rt, n_state, SENTINEL);
            for phase in phases {
                let cols = Cols::dense(&out, width as u32);
                let si = StateIn::PerBatch(&state_in);
                match phase {
                    None => qwen35::gdn_chunk_forward(rt, &dims, &qkv, &gates, &params, si, &ws, cols, Some(&st)),
                    Some(ph) => {
                        qwen35::gdn_chunk_phase(rt, &dims, &qkv, &gates, &params, si, &ws, cols, Some(&st), *ph)
                    }
                }
                .unwrap();
            }
            rt.synchronize().unwrap();
            let bits =
                |b: &GpuBuffer, n: usize| -> Vec<u32> { b.read_f32()[..n].iter().map(|x| x.to_bits()).collect() };
            (bits(&out, s.b * s.t * width), bits(&st, n_state))
        };
        use qwen35::GdnChunkPhase::{Prep, Scan};
        let whole = run(&[None]);
        assert!(
            whole.0.iter().all(|&x| x != SENTINEL.to_bits()),
            "gdn_chunk_forward left outputs unwritten"
        );
        assert_eq!(run(&[Some(Prep), Some(Scan)]), whole, "Prep then Scan");
        let prep_only = run(&[Some(Prep)]);
        assert!(
            prep_only.0.iter().chain(&prep_only.1).all(|&x| x == SENTINEL.to_bits()),
            "Prep alone wrote an output or a state"
        );
    });
}

#[test]
fn gdn_chunk_edges_of_the_chunk_and_head_grouping() {
    with_gpu(|rt| {
        let cases = [
            (
                GdnShape {
                    b: 1,
                    t: 1,
                    hk: 1,
                    hv: 1,
                    dv: 32,
                },
                "none",
            ),
            (
                GdnShape {
                    b: 1,
                    t: 63,
                    hk: 1,
                    hv: 1,
                    dv: 64,
                },
                "none",
            ),
            (
                GdnShape {
                    b: 1,
                    t: 64,
                    hk: 1,
                    hv: 2,
                    dv: 32,
                },
                "batch",
            ),
            (
                GdnShape {
                    b: 2,
                    t: 65,
                    hk: 1,
                    hv: 2,
                    dv: 64,
                },
                "batch",
            ),
            (
                GdnShape {
                    b: 3,
                    t: 100,
                    hk: 1,
                    hv: 1,
                    dv: 128,
                },
                "snapshot",
            ),
            (
                GdnShape {
                    b: 1,
                    t: 200,
                    hk: 2,
                    hv: 4,
                    dv: 128,
                },
                "none",
            ),
        ];
        for (i, (s, state)) in cases.into_iter().enumerate() {
            let d = GdnData::random(s, state, 100 + i as u64);
            check_gdn(&format!("chunk {s:?} {state}"), rt, &d, Path::Chunk);
        }
    });
}

/// Qwen3.5-4B's gated delta net heads: 16 key heads over 32 value heads of
/// 128 (two value heads per key head), through the chunked prefill (past two
/// chunk edges) and the recurrent decode, against the f64 reference that
/// `gdn_reference_matches_transformers` holds to transformers.
#[test]
fn gdn_at_the_4b_head_counts() {
    with_gpu(|rt| {
        let s = GdnShape {
            b: 1,
            t: 130,
            hk: 16,
            hv: 32,
            dv: 128,
        };
        check_gdn("chunk 4B heads", rt, &GdnData::random(s, "batch", 4400), Path::Chunk);
        let s = GdnShape { t: 3, ..s };
        check_gdn(
            "recurrent 4B heads",
            rt,
            &GdnData::random(s, "batch", 4410),
            Path::Recurrent,
        );
    });
}

#[test]
fn gdn_recurrent_decode_shapes() {
    with_gpu(|rt| {
        let cases = [
            (
                GdnShape {
                    b: 4,
                    t: 1,
                    hk: 2,
                    hv: 4,
                    dv: 64,
                },
                "snapshot",
            ),
            (
                GdnShape {
                    b: 2,
                    t: 7,
                    hk: 1,
                    hv: 2,
                    dv: 128,
                },
                "batch",
            ),
            (
                GdnShape {
                    b: 1,
                    t: 20,
                    hk: 1,
                    hv: 1,
                    dv: 32,
                },
                "none",
            ),
        ];
        for (i, (s, state)) in cases.into_iter().enumerate() {
            let d = GdnData::random(s, state, 200 + i as u64);
            check_gdn(&format!("recurrent {s:?} {state}"), rt, &d, Path::Recurrent);
        }
    });
}

#[test]
fn gdn_strong_decay_stays_finite_and_exact() {
    // A_log up to 2.5 and |a| up to 4: g reaches about -50 per step, so every
    // decay product in a chunk underflows. No exponent in either kernel is
    // positive, so nothing may overflow on the way.
    with_gpu(|rt| {
        let s = GdnShape {
            b: 1,
            t: 150,
            hk: 1,
            hv: 1,
            dv: 32,
        };
        let mut d = GdnData::random(s, "none", 300);
        d.a_log = vec![2.5];
        d.a = d.a.iter().map(|x| x * 2.0).collect();
        check_gdn("chunk strong decay", rt, &d, Path::Chunk);
        check_gdn("recurrent strong decay", rt, &d, Path::Recurrent);
    });
}

/// Split rows `[t0, t1)` of every sequence out of a problem.
fn slice_time(d: &GdnData, t0: usize, t1: usize) -> GdnData {
    let GdnShape { b, t, hk, hv, dv } = d.s;
    let take = |v: &[f32], w: usize| -> Vec<f32> {
        (0..b)
            .flat_map(|bi| v[(bi * t + t0) * w..(bi * t + t1) * w].to_vec())
            .collect()
    };
    GdnData {
        s: GdnShape {
            b,
            t: t1 - t0,
            hk,
            hv,
            dv,
        },
        q: take(&d.q, hk * DK),
        k: take(&d.k, hk * DK),
        v: take(&d.v, hv * dv),
        a: take(&d.a, hv),
        b: take(&d.b, hv),
        a_log: d.a_log.clone(),
        dt_bias: d.dt_bias.clone(),
        state0: None,
        snapshot: false,
    }
}

/// Row `bi` of `d`, cut to its first `len` tokens, as a batch of one; its own
/// start state, or the shared snapshot.
fn row_of(d: &GdnData, bi: usize, len: usize) -> GdnData {
    let GdnShape { t, hk, hv, dv, .. } = d.s;
    let take = |v: &[f32], w: usize| v[bi * t * w..(bi * t + len) * w].to_vec();
    let per_state = hv * DK * dv;
    GdnData {
        s: GdnShape {
            b: 1,
            t: len,
            hk,
            hv,
            dv,
        },
        q: take(&d.q, hk * DK),
        k: take(&d.k, hk * DK),
        v: take(&d.v, hv * dv),
        a: take(&d.a, hv),
        b: take(&d.b, hv),
        a_log: d.a_log.clone(),
        dt_bias: d.dt_bias.clone(),
        state0: d.state0.as_ref().map(|s0| {
            if d.snapshot {
                s0.clone()
            } else {
                s0[bi * per_state..(bi + 1) * per_state].to_vec()
            }
        }),
        snapshot: d.snapshot,
    }
}

/// A ragged GDN batch through the `_varlen` path against each row alone at
/// its own length through the equal-length path: outputs, final state, and
/// the rows past each length left unwritten, all bit for bit.
fn gdn_varlen_case(rt: &Arc<GpuRuntime>, d: &GdnData, path: Path, lens: &[u32]) {
    let GdnShape { t, hv, dv, .. } = d.s;
    let width = hv * dv;
    let per_state = hv * DK * dv;
    let (y, st) = run_gdn_lens(rt, d, path, StateOut::Separate, Some(lens));
    for (bi, &len) in lens.iter().enumerate() {
        let live = (len as usize).min(t);
        let (y1, st1) = run_gdn(rt, &row_of(d, bi, live), path);
        let label = format!("{path:?} row {bi} (len {len} of {t})");
        let mine = &y[bi * t * width..(bi + 1) * t * width];
        for (i, (a, w)) in mine[..live * width].iter().zip(&y1).enumerate() {
            assert_eq!(a.to_bits(), w.to_bits(), "{label} y[{i}]: {a} vs {w} alone");
        }
        assert!(
            mine[live * width..].iter().all(|&x| x == SENTINEL),
            "{label}: wrote a row past its length"
        );
        let s_mine = &st[bi * per_state..(bi + 1) * per_state];
        assert!(
            s_mine.iter().zip(&st1).all(|(a, w)| a.to_bits() == w.to_bits()),
            "{label}: final state differs from the row alone"
        );
    }
}

#[test]
fn gdn_varlen_rows_equal_each_row_alone_bit_for_bit() {
    // Ragged continuations in one call, on both GDN paths, from per-row start
    // states and from one shared snapshot. Lengths either side of the 64-row
    // chunk, 0 (the state passes through), the full T, and one past it
    // (clamped).
    with_gpu(|rt| {
        let shape = GdnShape {
            b: 7,
            t: 130,
            hk: 1,
            hv: 2,
            dv: 32,
        };
        let lens = [0u32, 1, 63, 64, 65, 130, 200];
        for (i, state) in ["batch", "snapshot", "zero"].iter().enumerate() {
            let d = GdnData::random(shape, state, 8200 + 10 * i as u64);
            gdn_varlen_case(rt, &d, Path::Chunk, &lens);
        }
        let short = GdnShape {
            b: 5,
            t: 9,
            hk: 1,
            hv: 2,
            dv: 64,
        };
        let lens = [0u32, 1, 5, 9, 12];
        for (i, state) in ["batch", "snapshot"].iter().enumerate() {
            let d = GdnData::random(short, state, 8250 + 10 * i as u64);
            gdn_varlen_case(rt, &d, Path::Recurrent, &lens);
        }
    });
}

#[test]
fn conv1d_varlen_rows_equal_each_row_alone_bit_for_bit() {
    // The conv's ragged form: each row's outputs and its carried state (the
    // last KW - 1 inputs before its own length, reaching back into the input
    // state when the row is shorter than that) equal the row run alone.
    with_gpu(|rt| {
        let (t, c, kw) = (20usize, 70usize, 4usize);
        let hist = kw - 1;
        let lens = [0u32, 1, 2, 3, 4, 19, 20, 25];
        let b = lens.len();
        let x = random_f32(b * t * c, 8300);
        let w = random_f32(c * kw, 8301);
        let (xb, wb) = (buf(rt, &x), buf(rt, &w));
        for snapshot in [false, true] {
            let st0 = random_f32(if snapshot { c * hist } else { b * c * hist }, 8302);
            let sb = buf(rt, &st0);
            let state = if snapshot {
                StateIn::Snapshot(&sb)
            } else {
                StateIn::PerBatch(&sb)
            };
            let y = seeded(rt, b * t * c, SENTINEL);
            let so = seeded(rt, b * c * hist, SENTINEL);
            let lb = buf_u32(rt, &lens);
            qwen35::conv1d_silu_varlen(
                rt,
                Cols::dense(&xb, c as u32),
                &wb,
                kw as u32,
                state,
                &y,
                Some(&so),
                b as u32,
                t as u32,
                c as u32,
                &lb,
            )
            .unwrap();
            rt.synchronize().unwrap();
            let (yh, soh) = (y.read_f32(), so.read_f32());
            for (bi, &len) in lens.iter().enumerate() {
                let live = (len as usize).min(t);
                let x1 = buf(rt, &x[bi * t * c..(bi * t + live) * c]);
                let s1 = buf(
                    rt,
                    if snapshot {
                        &st0[..]
                    } else {
                        &st0[bi * c * hist..(bi + 1) * c * hist]
                    },
                );
                let y1 = seeded(rt, live * c, SENTINEL);
                let so1 = seeded(rt, c * hist, SENTINEL);
                qwen35::conv1d_silu(
                    rt,
                    Cols::dense(&x1, c as u32),
                    &wb,
                    kw as u32,
                    StateIn::PerBatch(&s1),
                    &y1,
                    Some(&so1),
                    1,
                    live as u32,
                    c as u32,
                )
                .unwrap();
                rt.synchronize().unwrap();
                let label = format!("conv row {bi} (len {len}, snapshot {snapshot})");
                let mine = &yh[bi * t * c..(bi + 1) * t * c];
                let alone = y1.read_f32();
                for (i, (a, w)) in mine[..live * c].iter().zip(&alone).enumerate() {
                    assert_eq!(a.to_bits(), w.to_bits(), "{label} y[{i}]");
                }
                assert!(
                    mine[live * c..].iter().all(|&v| v == SENTINEL),
                    "{label}: wrote past its length"
                );
                let sm = &soh[bi * c * hist..(bi + 1) * c * hist];
                assert!(
                    sm.iter().zip(&so1.read_f32()).all(|(a, w)| a.to_bits() == w.to_bits()),
                    "{label}: state differs from the row alone"
                );
            }
        }
    });
}

#[test]
fn gdn_prefill_then_decode_in_place_continues_the_sequence() {
    with_gpu(|rt| {
        let full = GdnData::random(
            GdnShape {
                b: 2,
                t: 103,
                hk: 1,
                hv: 2,
                dv: 64,
            },
            "none",
            400,
        );
        let (want, want_state) = gdn_f64(&full.problem());
        let pre = slice_time(&full, 0, 100);
        let dims = dims_of(pre.s);
        let p = Packed::new(rt, &pre);
        let width = pre.s.hv * pre.s.dv;
        let out = seeded(rt, 2 * 100 * width, SENTINEL);
        let state = seeded(rt, 2 * pre.s.hv * DK * pre.s.dv, SENTINEL);
        let ws = GdnWorkspace::new(rt, &dims).unwrap();
        qwen35::gdn_chunk_forward(
            rt,
            &dims,
            &p.qkv(),
            &p.gates(dims.v_heads),
            &p.params(),
            StateIn::Zero,
            &ws,
            Cols::dense(&out, width as u32),
            Some(&state),
        )
        .unwrap();
        // Three decode steps, one token each, advancing the state in place.
        let mut dec_out = Vec::new();
        for step in 0..3 {
            let tok = slice_time(&full, 100 + step, 101 + step);
            let dd = dims_of(tok.s);
            let tp = Packed::new(rt, &tok);
            let o = seeded(rt, 2 * width, SENTINEL);
            qwen35::gdn_recurrent(
                rt,
                &dd,
                &tp.qkv(),
                &tp.gates(dd.v_heads),
                &tp.params(),
                StateIn::PerBatch(&state),
                Cols::dense(&o, width as u32),
                Some(&state),
            )
            .unwrap();
            rt.synchronize().unwrap();
            dec_out.push(o.read_f32()[..2 * width].to_vec());
        }
        let prefill = out.read_f32();
        for bi in 0..2 {
            for t in 0..103 {
                let got = if t < 100 {
                    &prefill[(bi * 100 + t) * width..(bi * 100 + t + 1) * width]
                } else {
                    &dec_out[t - 100][bi * width..(bi + 1) * width]
                };
                let w = &want[(bi * 103 + t) * width..(bi * 103 + t + 1) * width];
                assert_close(&format!("b{bi} t{t}"), got, w, rel_bound(&want));
            }
        }
        let st = state.read_f32();
        assert_close(
            "final state",
            &st[..want_state.len()],
            &want_state,
            rel_bound(&want_state),
        );
    });
}

// ------------------------------------------------------------------ conv1d ---

#[test]
fn conv1d_prefill_then_decode_continues() {
    with_gpu(|rt| {
        let (b, t, c, kw) = (2usize, 38usize, 100usize, 4usize);
        let x = random_f32(b * t * c, 500);
        let w = random_f32(c * kw, 501);
        let (want, want_state) = conv1d_silu_f64(&x, &w, None, b, t, c, kw);
        // Prefill the first 37 steps from a strided window, then decode step 38.
        let (ld, off) = (c + 13, 5usize);
        let mut xs = vec![SENTINEL; b * t * ld];
        for r in 0..b * t {
            xs[r * ld + off..r * ld + off + c].copy_from_slice(&x[r * c..(r + 1) * c]);
        }
        let pre_rows: Vec<f32> = (0..b)
            .flat_map(|bi| xs[bi * t * ld..(bi * t + t - 1) * ld].to_vec())
            .collect();
        let xb = buf(rt, &pre_rows);
        let wb = buf(rt, &w);
        let y = seeded(rt, b * (t - 1) * c, SENTINEL);
        let st = seeded(rt, b * c * (kw - 1), SENTINEL);
        let x_cols = Cols {
            buf: &xb,
            ld: ld as u32,
            off: off as u32,
        };
        qwen35::conv1d_silu(
            rt,
            x_cols,
            &wb,
            kw as u32,
            StateIn::Zero,
            &y,
            Some(&st),
            b as u32,
            (t - 1) as u32,
            c as u32,
        )
        .unwrap();
        let last: Vec<f32> = (0..b)
            .flat_map(|bi| x[(bi * t + t - 1) * c..(bi * t + t) * c].to_vec())
            .collect();
        let lb = buf(rt, &last);
        let y1 = seeded(rt, b * c, SENTINEL);
        let st1 = seeded(rt, b * c * (kw - 1), SENTINEL);
        qwen35::conv1d_silu(
            rt,
            Cols::dense(&lb, c as u32),
            &wb,
            kw as u32,
            StateIn::PerBatch(&st),
            &y1,
            Some(&st1),
            b as u32,
            1,
            c as u32,
        )
        .unwrap();
        rt.synchronize().unwrap();
        let yp = y.read_f32();
        let yd = y1.read_f32();
        for bi in 0..b {
            for ti in 0..t {
                let got = if ti < t - 1 {
                    &yp[(bi * (t - 1) + ti) * c..(bi * (t - 1) + ti + 1) * c]
                } else {
                    &yd[bi * c..(bi + 1) * c]
                };
                assert_close_rel(
                    &format!("conv b{bi} t{ti}"),
                    got,
                    &want[(bi * t + ti) * c..(bi * t + ti + 1) * c],
                    1e-5,
                    1e-6,
                );
            }
        }
        assert_close("conv state", &st1.read_f32()[..want_state.len()], &want_state, 0.0);
    });
}

#[test]
fn conv1d_snapshot_state_is_shared_and_read_only() {
    with_gpu(|rt| {
        let (b, t, c, kw) = (3usize, 2usize, 64usize, 4usize);
        let x = random_f32(b * t * c, 510);
        let w = random_f32(c * kw, 511);
        let snap = random_f32(c * (kw - 1), 512);
        let (want, want_state) = conv1d_silu_f64(&x, &w, Some((&snap, true)), b, t, c, kw);
        let (xb, wb, sb) = (buf(rt, &x), buf(rt, &w), buf(rt, &snap));
        let y = seeded(rt, b * t * c, SENTINEL);
        let st = seeded(rt, b * c * (kw - 1), SENTINEL);
        qwen35::conv1d_silu(
            rt,
            Cols::dense(&xb, c as u32),
            &wb,
            kw as u32,
            StateIn::Snapshot(&sb),
            &y,
            Some(&st),
            b as u32,
            t as u32,
            c as u32,
        )
        .unwrap();
        rt.synchronize().unwrap();
        assert_close_rel("conv y", &y.read_f32()[..want.len()], &want, 1e-5, 1e-6);
        assert_close("conv state", &st.read_f32()[..want_state.len()], &want_state, 0.0);
        assert_eq!(sb.read_f32()[..snap.len()], snap[..], "snapshot was written");
    });
}

// -------------------------------------------------------------- gated norm ---

#[test]
fn gated_rms_norm_f32_and_bf16() {
    with_gpu(|rt| {
        let (rows, h, d) = (37usize, 4usize, 128usize);
        let width = h * d;
        let (x_ld, x_off, z_ld, z_off) = (width + 3, 2usize, width + 11, 7usize);
        let x = random_f32(rows * x_ld, 600);
        let z: Vec<f32> = random_f32(rows * z_ld, 601).iter().map(|v| v * 3.0).collect();
        let w: Vec<f32> = random_f32(d, 602).iter().map(|v| 1.0 + 0.2 * v).collect();
        let dense = |v: &[f32], ld: usize, off: usize| -> Vec<f32> {
            (0..rows)
                .flat_map(|r| v[r * ld + off..r * ld + off + width].to_vec())
                .collect()
        };
        let want = gated_rms_norm_f64(&dense(&x, x_ld, x_off), &dense(&z, z_ld, z_off), &w, d, 1e-6);
        let (xb, zb, wb) = (buf(rt, &x), buf(rt, &z), buf(rt, &w));
        let xc = Cols {
            buf: &xb,
            ld: x_ld as u32,
            off: x_off as u32,
        };
        let zc = Cols {
            buf: &zb,
            ld: z_ld as u32,
            off: z_off as u32,
        };
        let out = seeded(rt, rows * width, SENTINEL);
        let oc = OutCols {
            cols: Cols::dense(&out, width as u32),
            dtype: DType::F32,
        };
        qwen35::gated_rms_norm(rt, xc, zc, &wb, oc, rows as u32, h as u32, d as u32, 1e-6).unwrap();
        let outb = rt.alloc_buffer(rows * width * 2).unwrap();
        let ob = OutCols {
            cols: Cols::dense(&outb, width as u32),
            dtype: DType::BF16,
        };
        qwen35::gated_rms_norm(rt, xc, zc, &wb, ob, rows as u32, h as u32, d as u32, 1e-6).unwrap();
        rt.synchronize().unwrap();
        assert_close_rel("gated norm f32", &out.read_f32()[..want.len()], &want, 1e-5, 1e-6);
        let got = read_bf16(&outb, want.len());
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            // One bf16 rounding of the f32 result: half an ulp of 2^-8, plus slack.
            assert!(
                (f64::from(g) - w).abs() <= w.abs() * (2f64.powi(-8) + 1e-5) + 1e-6,
                "bf16 [{i}] {g} vs {w}"
            );
        }
    });
}

// ------------------------------------------------------------ residual norm ---

/// `qwen35::rms_norm` on the zero-centred `w`: the f32 output is tessl's
/// generic RMSNorm fed `1 + w` folded in f32, bit for bit (the same reduction
/// and the same `fl(1 + w)`), so storing `w` moved no forward; both outputs
/// are `x * rstd * (1 + w)` in f64 to their dtype's rounding. Weights reach
/// below -0.5, where `1 + w` rounds `w`. The 2B's hidden size spans several
/// simdgroups; 300 is not a power of two.
#[test]
fn qwen35_rms_norm_f32_and_bf16() {
    with_gpu(|rt| {
        for (rows, dim, seed) in [(37usize, 2048usize, 610u64), (5, 300, 620)] {
            let eps = 1e-6f32;
            let x = random_f32(rows * dim, seed);
            let w: Vec<f32> = random_f32(dim, seed + 1).iter().map(|v| 0.9 * v).collect();
            assert!(w.iter().any(|&v| v < -0.5), "the weights must reach below -0.5");
            let want: Vec<f64> = x
                .chunks(dim)
                .flat_map(|r| {
                    let ms = r.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / dim as f64;
                    let rstd = 1.0 / (ms + f64::from(eps)).sqrt();
                    r.iter()
                        .zip(&w)
                        .map(move |(&v, &wv)| f64::from(v) * rstd * (1.0 + f64::from(wv)))
                        .collect::<Vec<_>>()
                })
                .collect();
            let (xb, wb) = (buf(rt, &x), buf(rt, &w));
            let folded: Vec<f32> = w.iter().map(|v| 1.0 + v).collect();
            let fb = buf(rt, &folded);
            let out = seeded(rt, rows * dim, SENTINEL);
            let generic = seeded(rt, rows * dim, SENTINEL);
            let outb = rt.alloc_buffer(rows * dim * 2).unwrap();
            let (r, d) = (rows as u32, dim as u32);
            qwen35::rms_norm(rt, &xb, &wb, &out, DType::F32, r, d, eps).unwrap();
            qwen35::rms_norm(rt, &xb, &wb, &outb, DType::BF16, r, d, eps).unwrap();
            tessl::nn::rms_norm_f32(rt, &xb, &fb, &generic, r, d, eps).unwrap();
            rt.synchronize().unwrap();
            let got = out.read_f32()[..rows * dim].to_vec();
            let same = got
                .iter()
                .zip(&generic.read_f32()[..rows * dim])
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "{rows}x{dim}: not the generic RMSNorm on 1 + w bit for bit");
            assert_close_rel(&format!("qwen35 rms_norm f32 {rows}x{dim}"), &got, &want, 1e-5, 1e-6);
            for (i, (&g, &wv)) in read_bf16(&outb, want.len()).iter().zip(&want).enumerate() {
                // One bf16 rounding of the f32 result: half an ulp of 2^-8, plus slack.
                assert!(
                    (f64::from(g) - wv).abs() <= wv.abs() * (2f64.powi(-8) + 1e-5) + 1e-6,
                    "{rows}x{dim} bf16 [{i}] {g} vs {wv}"
                );
            }
        }
        // Refusals: no width, a non-positive eps, a bf16 output over x, a dtype
        // with no kernel.
        let (xb, wb) = (buf(rt, &[1.0; 64]), buf(rt, &[0.0; 64]));
        let out = seeded(rt, 64, SENTINEL);
        for (e, want) in [
            (
                qwen35::rms_norm(rt, &xb, &wb, &out, DType::F32, 1, 0, 1e-6),
                "dim must be non-zero",
            ),
            (
                qwen35::rms_norm(rt, &xb, &wb, &out, DType::F32, 1, 64, 0.0),
                "eps must be finite and positive",
            ),
            (
                qwen35::rms_norm(rt, &xb, &wb, &xb, DType::BF16, 1, 64, 1e-6),
                "overlaps read-only buffer x",
            ),
            (
                qwen35::rms_norm(rt, &xb, &wb, &wb, DType::F32, 1, 64, 1e-6),
                "overlaps read-only buffer w",
            ),
        ] {
            let e = e.unwrap_err();
            assert!(e.contains(want), "{e}");
        }
    });
}

/// Host-only. Widths whose sum overflows used to wrap (release) or panic
/// (debug); `in_features = 0` with a huge width used to spin through every
/// empty row. Both now finish at once.
#[test]
fn pack_linear_weights_rejects_overflowing_widths_and_skips_empty_rows() {
    let empty: [&[f32]; 2] = [&[], &[]];
    let e = qwen35::pack_linear_weights_f32(&empty, &[usize::MAX, 2], 0).unwrap_err();
    assert!(e.contains("overflow"), "{e}");
    let e = qwen35::pack_linear_weights_bf16(&[&[], &[]], &[usize::MAX, 2], 0).unwrap_err();
    assert!(e.contains("overflow"), "{e}");
    let t0 = std::time::Instant::now();
    let packed = qwen35::pack_linear_weights_f32(&empty, &[usize::MAX / 2, 3], 0).unwrap();
    assert!(packed.is_empty());
    assert!(t0.elapsed().as_secs() < 5, "took {:?}", t0.elapsed());
    // The ordinary case is unchanged: [out, in] parts side by side, transposed.
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let b = [7.0f32, 8.0, 9.0]; // [1, 3]
    let packed = qwen35::pack_linear_weights_f32(&[&a, &b], &[2, 1], 3).unwrap();
    assert_eq!(packed, vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
}

#[test]
fn residual_add_is_the_exact_f32_sum_inside_its_windows() {
    with_gpu(|rt| {
        // Windows of wider rows on both sides, a width off every 32-lane edge,
        // and values whose f32 sum rounds (so an f64 or fused path would show).
        let (rows, width) = (29usize, 301usize);
        let (y_ld, y_off) = (width + 7, 5usize);
        let (r_ld, r_off) = (width + 3, 2usize);
        let y: Vec<f32> = random_f32(rows * y_ld, 710).iter().map(|v| 1e3 * v).collect();
        let mut resid: Vec<f32> = random_f32(rows * r_ld, 711).iter().map(|v| 1e-3 * v).collect();
        // Padding columns carry a sentinel the kernel must not touch.
        for r in 0..rows {
            for c in (0..r_off).chain(r_off + width..r_ld) {
                resid[r * r_ld + c] = SENTINEL;
            }
        }
        let yb = buf(rt, &y);
        let rb = buf(rt, &resid);
        let yc = Cols {
            buf: &yb,
            ld: y_ld as u32,
            off: y_off as u32,
        };
        let rc = Cols {
            buf: &rb,
            ld: r_ld as u32,
            off: r_off as u32,
        };
        qwen35::residual_add(rt, yc, rc, rows as u32, width as u32).unwrap();
        rt.synchronize().unwrap();
        let got = rb.read_f32();
        for r in 0..rows {
            for c in 0..r_ld {
                let i = r * r_ld + c;
                let want = if (r_off..r_off + width).contains(&c) {
                    resid[i] + y[r * y_ld + y_off + (c - r_off)]
                } else {
                    SENTINEL
                };
                assert_eq!(got[i].to_bits(), want.to_bits(), "row {r} col {c}");
            }
        }

        // Rejections: aliasing, a window past the buffer, and zero work.
        expect_err(qwen35::residual_add(rt, rc, rc, rows as u32, width as u32), "resid");
        // One row too many: `y` is checked first.
        expect_err(
            qwen35::residual_add(rt, yc, rc, rows as u32 + 1, width as u32),
            "residual_add y",
        );
        expect_err(
            qwen35::residual_add(rt, yc, Cols { off: 4, ..rc }, rows as u32, width as u32),
            "residual_add resid",
        );
        qwen35::residual_add(rt, yc, rc, 0, width as u32).unwrap();
        qwen35::residual_add(rt, yc, rc, rows as u32, 0).unwrap();
    });
}

#[test]
fn swiglu_f32_and_bf16_from_a_fused_gate_up_buffer() {
    with_gpu(|rt| {
        // Gate and up as two windows of one [gate | up] row, as a fused GEMM
        // would write them; a width off every 32-lane edge; gates out to +-120,
        // where a textbook sigmoid's exp(-x) overflows under fast math.
        let (rows, width) = (37usize, 300usize);
        let ld = 2 * width + 5;
        let (g_off, u_off) = (1usize, 1 + width);
        let mut fused: Vec<f32> = random_f32(rows * ld, 700).iter().map(|v| 8.0 * v).collect();
        fused[g_off] = 120.0;
        fused[g_off + 1] = -120.0;
        fused[ld + g_off] = -89.0;
        let silu = |x: f64| x / (1.0 + (-x).exp());
        let want: Vec<f64> = (0..rows)
            .flat_map(|r| {
                let f = &fused;
                (0..width).map(move |c| silu(f[r * ld + g_off + c] as f64) * f[r * ld + u_off + c] as f64)
            })
            .collect();
        let fb = buf(rt, &fused);
        let gate = Cols {
            buf: &fb,
            ld: ld as u32,
            off: g_off as u32,
        };
        let up = Cols {
            buf: &fb,
            ld: ld as u32,
            off: u_off as u32,
        };
        // f32 into a window of a wider row: the padding must stay untouched.
        let (o_ld, o_off) = (width + 4, 3usize);
        let out = seeded(rt, rows * o_ld, SENTINEL);
        let oc = OutCols {
            cols: Cols {
                buf: &out,
                ld: o_ld as u32,
                off: o_off as u32,
            },
            dtype: DType::F32,
        };
        qwen35::swiglu(rt, gate, up, oc, rows as u32, width as u32).unwrap();
        let outb = rt.alloc_buffer(rows * width * 2).unwrap();
        let ob = OutCols {
            cols: Cols::dense(&outb, width as u32),
            dtype: DType::BF16,
        };
        qwen35::swiglu(rt, gate, up, ob, rows as u32, width as u32).unwrap();
        rt.synchronize().unwrap();
        let all = out.read_f32();
        let mut got = Vec::with_capacity(rows * width);
        for r in 0..rows {
            let row = &all[r * o_ld..(r + 1) * o_ld];
            assert!(
                row[..o_off].iter().chain(&row[o_off + width..]).all(|&x| x == SENTINEL),
                "row {r}: wrote outside the output window"
            );
            got.extend_from_slice(&row[o_off..o_off + width]);
        }
        assert!(got.iter().all(|x| x.is_finite()), "non-finite swiglu output");
        assert_close_rel("swiglu f32", &got, &want, 1e-6, 1e-7);
        // The bf16 store is the same f32 value rounded, element for element.
        // (Scoped: the host mapping must be released before the next encode.)
        {
            let rounded = f32_slice_to_bf16(&got);
            let got16 = &outb.contents_u16()[..rows * width];
            for (i, (&g, &w)) in got16.iter().zip(&rounded).enumerate() {
                assert_eq!(g, w, "swiglu bf16 [{i}]: {g:#x} vs rounded f32 {w:#x}");
            }
        }

        let dense_out = seeded(rt, rows * width, SENTINEL);
        let dense = OutCols {
            cols: Cols::dense(&dense_out, width as u32),
            dtype: DType::F32,
        };
        let onto_input = OutCols {
            cols: Cols {
                buf: &fb,
                ld: ld as u32,
                off: 0,
            },
            dtype: DType::F32,
        };
        expect_err(
            qwen35::swiglu(rt, gate, up, onto_input, rows as u32, width as u32),
            "qwen35::swiglu: writable buffer out overlaps read-only buffer gate",
        );
        let short = Cols {
            buf: &fb,
            ld: ld as u32,
            off: (ld - width + 1) as u32,
        };
        expect_err(
            qwen35::swiglu(rt, gate, short, dense, rows as u32, width as u32),
            "swiglu up",
        );
        let f16 = OutCols {
            cols: Cols::dense(&dense_out, width as u32),
            dtype: DType::F16,
        };
        expect_err(
            qwen35::swiglu(rt, gate, up, f16, rows as u32, width as u32),
            "qwen35::swiglu: dtype must be F32 or BF16",
        );
    });
}

// ------------------------------------------------------- attention extras ---

fn attn_proj_rows(layout: AttnProjLayout, rows: usize, q: &[f32], k: &[f32], v: &[f32], gate: &[f32]) -> Vec<f32> {
    let (hq, hkv, d) = (
        layout.q_heads() as usize,
        layout.kv_heads() as usize,
        layout.head_dim() as usize,
    );
    let w = layout.width() as usize;
    let mut p = vec![SENTINEL; rows * w];
    for r in 0..rows {
        for h in 0..hq {
            let dst = r * w + layout.q_off() as usize + h * 2 * d;
            p[dst..dst + d].copy_from_slice(&q[(r * hq + h) * d..(r * hq + h + 1) * d]);
            p[dst + d..dst + 2 * d].copy_from_slice(&gate[(r * hq + h) * d..(r * hq + h + 1) * d]);
        }
        for h in 0..hkv {
            let kd = r * w + layout.k_off() as usize + h * d;
            let vd = r * w + layout.v_off() as usize + h * d;
            p[kd..kd + d].copy_from_slice(&k[(r * hkv + h) * d..(r * hkv + h + 1) * d]);
            p[vd..vd + d].copy_from_slice(&v[(r * hkv + h) * d..(r * hkv + h + 1) * d]);
        }
    }
    p
}

/// Run the norm/RoPE/cache kernel; returns (q_out, k_cache, v_cache).
#[allow(clippy::too_many_arguments)]
fn run_qk_rope(
    rt: &Arc<GpuRuntime>,
    shape: AttnShape,
    p: &[f32],
    qw: &[f32],
    kw: &[f32],
    cap: u32,
    pos: u32,
    theta: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (b, t, hq, hkv, d) = (
        shape.batch as usize,
        shape.seq as usize,
        shape.q_heads as usize,
        shape.kv_heads as usize,
        shape.head_dim as usize,
    );
    let layout = AttnProjLayout::new(shape.q_heads, shape.kv_heads, shape.head_dim).unwrap();
    let (pb, qwb, kwb) = (buf(rt, p), buf(rt, qw), buf(rt, kw));
    let q_out = seeded(rt, b * t * hq * d, SENTINEL);
    let cache = b * cap as usize * hkv * d;
    let (kc, vc) = (seeded(rt, cache, SENTINEL), seeded(rt, cache, SENTINEL));
    let targets = AttnTargets {
        q_out: &q_out,
        k_cache: &kc,
        v_cache: &vc,
    };
    qwen35::attn_qk_norm_rope(
        rt,
        &shape,
        Cols::dense(&pb, layout.width()),
        &qwb,
        &kwb,
        &targets,
        pos,
        theta,
        1e-6,
    )
    .unwrap();
    rt.synchronize().unwrap();
    (
        q_out.read_f32()[..b * t * hq * d].to_vec(),
        kc.read_f32()[..cache].to_vec(),
        vc.read_f32()[..cache].to_vec(),
    )
}

#[test]
fn attn_qk_norm_rope_matches_transformers_golden() {
    with_gpu(|rt| {
        let (qs, q) = load_f32("rope_q_in");
        let (_, k) = load_f32("rope_k_in");
        let (t, hq, d) = (qs[1], qs[2], qs[3]);
        let shape = AttnShape {
            batch: 1,
            seq: t as u32,
            q_heads: hq as u32,
            kv_heads: 1,
            head_dim: d as u32,
            rotary_dim: 64,
        };
        let layout = AttnProjLayout::new(hq as u32, 1, d as u32).unwrap();
        let v = random_f32(t * d, 700);
        let gate = random_f32(t * hq * d, 701);
        let p = attn_proj_rows(layout, t, &q, &k, &v, &gate);
        let (cap, pos) = (20_008u32, 20_000u32);
        let (qo, kc, vc) = run_qk_rope(
            rt,
            shape,
            &p,
            &load_f32("rope_q_norm_w").1,
            &load_f32("rope_k_norm_w").1,
            cap,
            pos,
            1e7,
        );
        // At position 20000 the fp32 angle transformers forms carries ~1e-3 rad
        // of its own rounding (see `norm_rope_row_f64`), and one ulp of
        // difference in `pow` between Metal and the host moves it by as much.
        // The bound admits that and nothing structural: a wrong pairing or
        // frequency denominator is O(1).
        assert_close("q", &qo, &load("rope_q_out").1, 4e-3);
        let kslot = &kc[pos as usize * d..(pos as usize + t) * d];
        assert_close("k", kslot, &load("rope_k_out").1, 4e-3);
        assert_eq!(&vc[pos as usize * d..(pos as usize + t) * d], &v[..], "v cache");
        assert!(
            kc[..pos as usize * d].iter().all(|&x| x == SENTINEL),
            "k cache written below pos"
        );
        assert!(
            vc[..pos as usize * d].iter().all(|&x| x == SENTINEL),
            "v cache written below pos"
        );
    });
}

#[test]
fn attn_qk_norm_rope_small_positions_are_tight() {
    with_gpu(|rt| {
        let (b, t, hq, hkv, d, rot) = (2usize, 9usize, 4usize, 2usize, 256usize, 64usize);
        let shape = AttnShape {
            batch: b as u32,
            seq: t as u32,
            q_heads: hq as u32,
            kv_heads: hkv as u32,
            head_dim: d as u32,
            rotary_dim: rot as u32,
        };
        let layout = AttnProjLayout::new(hq as u32, hkv as u32, d as u32).unwrap();
        let q = random_f32(b * t * hq * d, 710);
        let k = random_f32(b * t * hkv * d, 711);
        let v = random_f32(b * t * hkv * d, 712);
        let gate = random_f32(b * t * hq * d, 713);
        let qw: Vec<f32> = random_f32(d, 714).iter().map(|x| 0.1 * x).collect();
        let kw: Vec<f32> = random_f32(d, 715).iter().map(|x| 0.1 * x).collect();
        let p = attn_proj_rows(layout, b * t, &q, &k, &v, &gate);
        let (cap, pos) = (24u32, 11u32);
        let (qo, kc, vc) = run_qk_rope(rt, shape, &p, &qw, &kw, cap, pos, 1e7);
        for bi in 0..b {
            for ti in 0..t {
                let r = bi * t + ti;
                let at = u64::from(pos) + ti as u64;
                for h in 0..hq {
                    let want = norm_rope_row_f64(&q[(r * hq + h) * d..(r * hq + h + 1) * d], &qw, rot, at, 1e7, 1e-6);
                    assert_close(
                        &format!("q b{bi} t{ti} h{h}"),
                        &qo[(r * hq + h) * d..(r * hq + h + 1) * d],
                        &want,
                        2e-5,
                    );
                }
                for h in 0..hkv {
                    let want = norm_rope_row_f64(&k[(r * hkv + h) * d..(r * hkv + h + 1) * d], &kw, rot, at, 1e7, 1e-6);
                    let slot = ((bi * cap as usize + pos as usize + ti) * hkv + h) * d;
                    assert_close(&format!("k b{bi} t{ti} h{h}"), &kc[slot..slot + d], &want, 2e-5);
                    assert_eq!(&vc[slot..slot + d], &v[(r * hkv + h) * d..(r * hkv + h + 1) * d]);
                }
            }
        }
    });
}

#[test]
fn attn_output_gate_f32_bf16_and_in_place() {
    with_gpu(|rt| {
        let (rows, hq, d) = (11usize, 3usize, 64usize);
        let width = hq * d;
        let layout = AttnProjLayout::new(hq as u32, 1, d as u32).unwrap();
        let gate: Vec<f32> = random_f32(rows * width, 800).iter().map(|x| 4.0 * x).collect();
        let zeros = vec![0.0f32; rows * width];
        let p = attn_proj_rows(layout, rows, &zeros, &zeros[..rows * d], &zeros[..rows * d], &gate);
        let attn = random_f32(rows * width, 801);
        let want: Vec<f64> = attn
            .iter()
            .zip(&gate)
            .map(|(&a, &g)| f64::from(a) * sigmoid(f64::from(g)))
            .collect();
        let pb = buf(rt, &p);
        let pc = Cols::dense(&pb, layout.width());
        let ab = buf(rt, &attn);
        let out = seeded(rt, rows * width, SENTINEL);
        qwen35::attn_output_gate(
            rt,
            &ab,
            pc,
            OutCols {
                cols: Cols::dense(&out, width as u32),
                dtype: DType::F32,
            },
            rows as u32,
            hq as u32,
            d as u32,
        )
        .unwrap();
        let outb = rt.alloc_buffer(rows * width * 2).unwrap();
        qwen35::attn_output_gate(
            rt,
            &ab,
            pc,
            OutCols {
                cols: Cols::dense(&outb, width as u32),
                dtype: DType::BF16,
            },
            rows as u32,
            hq as u32,
            d as u32,
        )
        .unwrap();
        rt.synchronize().unwrap();
        assert_close_rel("gate f32", &out.read_f32()[..want.len()], &want, 1e-5, 1e-7);
        let bf = read_bf16(&outb, want.len());
        for (i, (&g, &w)) in bf.iter().zip(&want).enumerate() {
            assert!(
                (f64::from(g) - w).abs() <= w.abs() * (2f64.powi(-8) + 1e-5) + 1e-7,
                "bf16 [{i}]"
            );
        }
        // In place over the attention output.
        qwen35::attn_output_gate(
            rt,
            &ab,
            pc,
            OutCols {
                cols: Cols::dense(&ab, width as u32),
                dtype: DType::F32,
            },
            rows as u32,
            hq as u32,
            d as u32,
        )
        .unwrap();
        rt.synchronize().unwrap();
        assert_close_rel("gate in place", &ab.read_f32()[..want.len()], &want, 1e-5, 1e-7);
    });
}

// ----------------------------------------------------------------- scoring ---

#[test]
fn score_answer_rows_f32_and_bf16() {
    with_gpu(|rt| {
        let (rows, hidden, vocab) = (20usize, 300usize, 50usize);
        let h = random_f32(rows * hidden, 900);
        let nw: Vec<f32> = random_f32(hidden, 901).iter().map(|x| 0.1 * x).collect();
        let emb: Vec<f32> = random_f32(vocab * hidden, 902).iter().map(|x| 0.05 * x).collect();
        let answers: Vec<u32> = (0..17).map(|i| (i * 7 + 3) % vocab as u32).collect();
        let slots = [3u32, 19, 0, 7];
        let (hb, nwb) = (buf(rt, &h), buf(rt, &nw));
        let (ab, sb) = (buf_u32(rt, &answers), buf_u32(rt, &slots));
        for dtype in [DType::F32, DType::BF16] {
            let (eb, emb_seen) = match dtype {
                DType::BF16 => {
                    let bits = f32_slice_to_bf16(&emb);
                    let b = rt.alloc_buffer(bits.len() * 2).unwrap();
                    b.write_bf16_bits(&bits);
                    (b, bits.into_iter().map(bf16_bits_to_f32).collect::<Vec<_>>())
                }
                _ => (buf(rt, &emb), emb.clone()),
            };
            let n = slots.len() * answers.len();
            let (lg, lp) = (seeded(rt, n, SENTINEL), seeded(rt, n, SENTINEL));
            let head = LmHead {
                weight: &eb,
                dtype,
                vocab: vocab as u32,
            };
            qwen35::score_answer_rows(
                rt,
                &hb,
                rows as u32,
                hidden as u32,
                &sb,
                slots.len() as u32,
                &nwb,
                1.0,
                1e-6,
                head,
                &ab,
                answers.len() as u32,
                &lg,
                &lp,
            )
            .unwrap();
            rt.synchronize().unwrap();
            let (gl, gp) = (lg.read_f32(), lp.read_f32());
            for (si, &slot) in slots.iter().enumerate() {
                let rows_e: Vec<&[f32]> = answers
                    .iter()
                    .map(|&a| &emb_seen[a as usize * hidden..(a as usize + 1) * hidden])
                    .collect();
                let (wl, wp) = score_row_f64(
                    &h[slot as usize * hidden..(slot as usize + 1) * hidden],
                    &nw,
                    1.0,
                    1e-6,
                    &rows_e,
                );
                let r = si * answers.len()..(si + 1) * answers.len();
                assert_close_rel(
                    &format!("{dtype:?} logits slot {slot}"),
                    &gl[r.clone()],
                    &wl,
                    1e-5,
                    1e-5,
                );
                assert_close_rel(&format!("{dtype:?} logprobs slot {slot}"), &gp[r], &wp, 1e-5, 1e-5);
            }
        }
    });
}

#[test]
fn score_out_of_range_indices_score_nan_and_spare_the_rest() {
    with_gpu(|rt| {
        let (rows, hidden, vocab) = (4usize, 64usize, 8usize);
        let h = random_f32(rows * hidden, 910);
        let nw = vec![0.0f32; hidden];
        let emb = random_f32(vocab * hidden, 911);
        let (hb, nwb, eb) = (buf(rt, &h), buf(rt, &nw), buf(rt, &emb));
        let ab = buf_u32(rt, &[1, vocab as u32, 2]);
        let sb = buf_u32(rt, &[rows as u32, 1]);
        let (lg, lp) = (seeded(rt, 6, SENTINEL), seeded(rt, 6, SENTINEL));
        let head = LmHead {
            weight: &eb,
            dtype: DType::F32,
            vocab: vocab as u32,
        };
        qwen35::score_answer_rows(
            rt,
            &hb,
            rows as u32,
            hidden as u32,
            &sb,
            2,
            &nwb,
            1.0,
            1e-6,
            head,
            &ab,
            3,
            &lg,
            &lp,
        )
        .unwrap();
        rt.synchronize().unwrap();
        let (l, p) = (lg.read_f32(), lp.read_f32());
        // Slot 0 names row 4 of 4: the whole row is NaN, in both outputs.
        assert!(
            l[..3].iter().chain(&p[..3]).all(|x| x.is_nan()),
            "bad slot: {:?} {:?}",
            &l[..3],
            &p[..3]
        );
        // Slot 1 is valid; its middle answer id is past the vocabulary. That
        // entry is NaN, and the other two are scored as a two-way softmax, not
        // poisoned by it.
        assert!(l[4].is_nan() && p[4].is_nan(), "bad answer: {} {}", l[4], p[4]);
        let rows_e: Vec<&[f32]> = [1usize, 2]
            .iter()
            .map(|&a| &emb[a * hidden..(a + 1) * hidden])
            .collect();
        let (wl, wp) = score_row_f64(&h[hidden..2 * hidden], &nw, 1.0, 1e-6, &rows_e);
        assert_close_rel("valid logits", &[l[3], l[5]], &wl, 1e-5, 1e-5);
        assert_close_rel("valid logprobs", &[p[3], p[5]], &wp, 1e-5, 1e-5);
    });
}

// -------------------------------------------------------- projection fusion ---

#[test]
fn fused_projection_equals_the_separate_linears() {
    with_gpu(|rt| {
        let (rows, hidden) = (16usize, 64usize);
        let layout = GdnProjLayout::new(1, 2, 32).unwrap();
        let widths = layout.part_widths();
        let parts: Vec<Vec<f32>> = widths
            .iter()
            .enumerate()
            .map(|(i, &o)| random_f32(o * hidden, 1000 + i as u64))
            .collect();
        let refs: Vec<&[f32]> = parts.iter().map(|p| p.as_slice()).collect();
        let packed = qwen35::pack_linear_weights_f32(&refs, &widths, hidden).unwrap();
        let x = random_f32(rows * hidden, 1010);
        let total = layout.width() as usize;
        let xt = common::tensor_f32(rt, &[rows, hidden], &x);
        let wt = common::tensor_f32(rt, &[hidden, total], &packed);
        let pt = rt.alloc_tensor_f32(&[rows, total]).unwrap();
        qwen35::fused_projection(&xt, &wt, &pt, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let got = pt.read_f32().unwrap();
        let mut col0 = 0;
        for (part, &o) in parts.iter().zip(&widths) {
            for r in 0..rows {
                for j in 0..o {
                    // nn.Linear: y = x @ W^T.
                    let want: f64 = (0..hidden)
                        .map(|k| f64::from(x[r * hidden + k]) * f64::from(part[j * hidden + k]))
                        .sum();
                    let g = f64::from(got[r * total + col0 + j]);
                    assert!((g - want).abs() < 1e-4, "part col {j} row {r}: {g} vs {want}");
                }
            }
            col0 += o;
        }
        assert_eq!(col0, total);
        assert_eq!(layout.a_off() as usize, total - 2);
    });
}

// ------------------------------------------------------------ host contract ---

#[test]
fn every_qwen35_kernel_is_in_the_metallib() {
    with_gpu(|rt| {
        for name in [
            "qwen35_conv1d_silu",
            "qwen35_gdn_chunk_prep",
            "qwen35_gdn_chunk_scan",
            "qwen35_gdn_chunk_scan_bv16",
            "qwen35_gdn_recurrent",
            "qwen35_gated_rms_norm_f32",
            "qwen35_gated_rms_norm_bf16",
            "qwen35_attn_qk_norm_rope",
            "qwen35_attn_qk_norm_rope_posbuf",
            "qwen35_attn_gate_f32",
            "qwen35_attn_gate_bf16",
            "qwen35_attn_tiled_h256_q32_k32_sg4",
            "qwen35_attn_tiled_h256_q32_k64_sg4",
            "qwen35_attn_tiled_h256_q64_k32_sg8",
            "qwen35_attn_tiled_h256_q64_k64_sg8",
            "qwen35_swiglu_f32",
            "qwen35_swiglu_bf16",
            "qwen35_residual_add_f32",
            "qwen35_score_rows_f32",
            "qwen35_score_rows_bf16",
        ] {
            rt.pipeline(name).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    });
}

#[test]
fn gdn_state_out_discarded_and_in_place_on_both_paths() {
    with_gpu(|rt| {
        let s = GdnShape {
            b: 2,
            t: 70,
            hk: 1,
            hv: 2,
            dv: 32,
        };
        let d = GdnData::random(s, "batch", 1200);
        let (want_y, want_s) = gdn_f64(&d.problem());
        for path in [Path::Chunk, Path::Recurrent] {
            let (y, _) = run_gdn_with(rt, &d, path, StateOut::Discard);
            assert_close(&format!("{path:?} discard y"), &y, &want_y, rel_bound(&want_y));
            let (y, st) = run_gdn_with(rt, &d, path, StateOut::InPlace);
            assert_close(&format!("{path:?} in-place y"), &y, &want_y, rel_bound(&want_y));
            assert_close(&format!("{path:?} in-place state"), &st, &want_s, rel_bound(&want_s));
        }
    });
}

#[test]
fn empty_sequence_copies_the_state_through() {
    // A decode loop that alternates two state buffers must not find a stale one
    // after a step with no tokens.
    with_gpu(|rt| {
        let s = GdnShape {
            b: 2,
            t: 0,
            hk: 1,
            hv: 2,
            dv: 32,
        };
        let mut d = GdnData::random(GdnShape { t: 1, ..s }, "batch", 1300);
        d.s = s;
        (d.q, d.k, d.v, d.a, d.b) = (vec![], vec![], vec![], vec![], vec![]);
        let want: Vec<f64> = d.state0.as_ref().unwrap().iter().map(|&x| f64::from(x)).collect();
        for path in [Path::Chunk, Path::Recurrent] {
            let (y, st) = run_gdn_with(rt, &d, path, StateOut::Separate);
            assert!(y.is_empty());
            assert_close(&format!("{path:?} T=0 state"), &st, &want, 0.0);
        }
        let (c, kw) = (64usize, 4usize);
        let snap = random_f32(c * (kw - 1), 1301);
        let (xb, wb, sb) = (seeded(rt, 1, 0.0), buf(rt, &random_f32(c * kw, 1302)), buf(rt, &snap));
        let (y, st) = (seeded(rt, 1, SENTINEL), seeded(rt, 3 * c * (kw - 1), SENTINEL));
        qwen35::conv1d_silu(
            rt,
            Cols::dense(&xb, c as u32),
            &wb,
            kw as u32,
            StateIn::Snapshot(&sb),
            &y,
            Some(&st),
            3,
            0,
            c as u32,
        )
        .unwrap();
        rt.synchronize().unwrap();
        let got = st.read_f32();
        for bi in 0..3 {
            assert_eq!(
                &got[bi * snap.len()..(bi + 1) * snap.len()],
                &snap[..],
                "conv T=0 row {bi}"
            );
        }
    });
}

#[test]
fn a_whole_gdn_layer_through_the_layout_helpers() {
    // Projection GEMM -> conv -> chunked delta rule -> gated norm, wired only
    // through `GdnProjLayout`, against the same chain in f64. This is what
    // holds the layout's column offsets to the kernels' expectations.
    with_gpu(|rt| {
        let (t, hidden, kw) = (70usize, 64usize, 4usize);
        let layout = GdnProjLayout::new(1, 2, 32).unwrap();
        let widths = layout.part_widths();
        let width = layout.width() as usize;
        let conv_dim = layout.conv_dim() as usize;
        let parts: Vec<Vec<f32>> = widths
            .iter()
            .enumerate()
            .map(|(i, &o)| {
                random_f32(o * hidden, 1400 + i as u64)
                    .iter()
                    .map(|v| v * 0.3)
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = parts.iter().map(|p| p.as_slice()).collect();
        let packed = qwen35::pack_linear_weights_f32(&refs, &widths, hidden).unwrap();
        let x = random_f32(t * hidden, 1410);
        let conv_w: Vec<f32> = random_f32(conv_dim * kw, 1411).iter().map(|v| v * 0.5).collect();
        let a_log = vec![-1.0f32, 0.5];
        let dt_bias = vec![-4.0f32, -2.0];
        let norm_w: Vec<f32> = random_f32(32, 1412).iter().map(|v| 1.0 + 0.1 * v).collect();

        // --- device ---
        let xt = common::tensor_f32(rt, &[t, hidden], &x);
        let wt = common::tensor_f32(rt, &[hidden, width], &packed);
        let pt = rt.alloc_tensor_f32(&[t, width]).unwrap();
        assert_eq!(
            pt.byte_offset(),
            0,
            "the kernels address the projection from its buffer's start"
        );
        qwen35::fused_projection(&xt, &wt, &pt, GemmBackend::TensorOps).unwrap();
        let proj = &pt.buffer;
        let qkv = seeded(rt, t * conv_dim, SENTINEL);
        let cwb = buf(rt, &conv_w);
        qwen35::conv1d_silu(
            rt,
            Cols::dense(proj, width as u32),
            &cwb,
            kw as u32,
            StateIn::Zero,
            &qkv,
            None,
            1,
            t as u32,
            conv_dim as u32,
        )
        .unwrap();
        let dims = layout.dims(1, t as u32);
        let ws = GdnWorkspace::new(rt, &dims).unwrap();
        let (alb, dtb) = (buf(rt, &a_log), buf(rt, &dt_bias));
        let o = seeded(rt, t * 64, SENTINEL);
        let params = GdnParams {
            a_log: &alb,
            dt_bias: &dtb,
        };
        qwen35::gdn_chunk_forward(
            rt,
            &dims,
            &layout.conv_qkv(&qkv),
            &layout.gates(proj),
            &params,
            StateIn::Zero,
            &ws,
            Cols::dense(&o, 64),
            None,
        )
        .unwrap();
        let nwb = buf(rt, &norm_w);
        let y = seeded(rt, t * 64, SENTINEL);
        let yc = OutCols {
            cols: Cols::dense(&y, 64),
            dtype: DType::F32,
        };
        qwen35::gated_rms_norm(rt, Cols::dense(&o, 64), layout.z(proj), &nwb, yc, t as u32, 2, 32, 1e-6).unwrap();
        rt.synchronize().unwrap();

        // --- f64 ---
        let p64: Vec<f64> = (0..t * width)
            .map(|i| {
                let (r, col) = (i / width, i % width);
                (0..hidden)
                    .map(|k| f64::from(x[r * hidden + k]) * f64::from(packed[k * width + col]))
                    .sum()
            })
            .collect();
        let pf: Vec<f32> = p64.iter().map(|&v| v as f32).collect();
        let cols = |off: usize, w: usize| -> Vec<f32> {
            (0..t)
                .flat_map(|r| pf[r * width + off..r * width + off + w].to_vec())
                .collect()
        };
        let (c64, _) = conv1d_silu_f64(&cols(0, conv_dim), &conv_w, None, 1, t, conv_dim, kw);
        let cf: Vec<f32> = c64.iter().map(|&v| v as f32).collect();
        let pick = |off: usize, w: usize| -> Vec<f32> {
            (0..t)
                .flat_map(|r| cf[r * conv_dim + off..r * conv_dim + off + w].to_vec())
                .collect()
        };
        let q = pick(0, 128);
        let k = pick(128, 128);
        let v = pick(256, 64);
        let a = cols(layout.a_off() as usize, 2);
        let b = cols(layout.b_off() as usize, 2);
        let problem = GdnProblem {
            s: GdnShape {
                b: 1,
                t,
                hk: 1,
                hv: 2,
                dv: 32,
            },
            q: &q,
            k: &k,
            v: &v,
            a: &a,
            b: &b,
            a_log: &a_log,
            dt_bias: &dt_bias,
            state0: None,
            snapshot: false,
        };
        let (o64, _) = gdn_f64(&problem);
        let of: Vec<f32> = o64.iter().map(|&v| v as f32).collect();
        let want = gated_rms_norm_f64(&of, &cols(layout.z_off() as usize, 64), &norm_w, 32, 1e-6);
        let got = y.read_f32();
        assert_close_rel("layer output", &got[..want.len()], &want, 1e-3, 1e-4 * max_abs(&want));
    });
}

/// Expect an error whose message names the check, so a rejection can never be
/// credited to some other validation that happened to fire first.
fn expect_err(r: Result<(), String>, needle: &str) {
    match r {
        Ok(()) => panic!("expected an error containing {needle:?}, got Ok"),
        Err(e) => assert!(e.contains(needle), "expected an error containing {needle:?}, got {e:?}"),
    }
}

#[test]
fn host_rejects_what_the_kernels_cannot_do() {
    with_gpu(|rt| {
        let s = GdnShape {
            b: 1,
            t: 10,
            hk: 1,
            hv: 2,
            dv: 32,
        };
        let d = GdnData::random(s, "batch", 1100);
        let p = Packed::new(rt, &d);
        // Big enough to pass as a state too, so aliasing is what fires.
        let out = seeded(rt, 2 * DK * 32, 0.0);
        let st = buf(rt, d.state0.as_ref().unwrap());
        let good = dims_of(s);
        let ws = GdnWorkspace::new(rt, &good).unwrap();
        let chunk = |dims: GdnDims,
                     qkv: GdnQkv<'_>,
                     state: StateIn<'_>,
                     out: Cols<'_>,
                     so: Option<&GpuBuffer>,
                     ws: &GdnWorkspace| {
            qwen35::gdn_chunk_forward(rt, &dims, &qkv, &p.gates(2), &p.params(), state, ws, out, so)
        };
        let o = Cols::dense(&out, 64);
        chunk(good, p.qkv(), StateIn::PerBatch(&st), o, Some(&st), &ws).expect("in-place per-batch state is allowed");

        expect_err(
            chunk(GdnDims { v_dim: 48, ..good }, p.qkv(), StateIn::Zero, o, None, &ws),
            "multiple of 32",
        );
        expect_err(
            chunk(
                GdnDims {
                    k_heads: 2,
                    v_heads: 3,
                    ..good
                },
                p.qkv(),
                StateIn::Zero,
                o,
                None,
                &ws,
            ),
            "multiple of k_heads",
        );
        let small = GdnWorkspace::new(rt, &GdnDims { seq: 1, ..good }).unwrap();
        expect_err(
            chunk(GdnDims { seq: 70, ..good }, p.qkv(), StateIn::Zero, o, None, &small),
            "GdnWorkspace k",
        );
        // Dims that pass `GdnDims::validate` but whose workspace size overflows
        // usize: an error naming the overflow, not a wrapped (release) or
        // panicking (debug) multiply.
        expect_err(
            chunk(
                GdnDims {
                    batch: 1 << 31,
                    seq: 256,
                    k_heads: 1,
                    v_heads: 1 << 19,
                    v_dim: 32,
                },
                p.qkv(),
                StateIn::Zero,
                o,
                None,
                &small,
            ),
            "GdnWorkspace: product overflows usize",
        );
        expect_err(
            chunk(good, p.qkv(), StateIn::Snapshot(&st), o, Some(&st), &ws),
            "state_out overlaps read-only buffer state_in",
        );
        expect_err(
            chunk(good, p.qkv(), StateIn::Zero, o, Some(&out), &ws),
            "out and state_out overlap",
        );
        let tiny = seeded(rt, 10, 0.0);
        expect_err(
            chunk(good, p.qkv(), StateIn::Zero, Cols::dense(&tiny, 64), None, &ws),
            "gdn out",
        );
        let qkv_as_out = GdnQkv { buf: &out, ..p.qkv() };
        expect_err(
            chunk(good, qkv_as_out, StateIn::Zero, o, None, &ws),
            "out overlaps read-only buffer qkv",
        );
        // In place, the state must still be disjoint from every input.
        let big = buf(rt, &random_f32(2 * DK * 32, 1104));
        let qkv_in_state = GdnQkv { buf: &big, ..p.qkv() };
        expect_err(
            chunk(good, qkv_in_state, StateIn::PerBatch(&big), o, Some(&big), &ws),
            "state (in place) overlaps read-only buffer qkv",
        );
        expect_err(
            qwen35::gdn_recurrent(
                rt,
                &good,
                &qkv_in_state,
                &p.gates(2),
                &p.params(),
                StateIn::PerBatch(&big),
                o,
                Some(&big),
            ),
            "state (in place) overlaps read-only buffer qkv",
        );

        let x = buf(rt, &random_f32(64 * 3, 1101));
        let w = buf(rt, &random_f32(64 * 4, 1102));
        let cs = buf(rt, &random_f32(64 * 3, 1103));
        let y = seeded(rt, 64 * 3, 0.0);
        expect_err(
            qwen35::conv1d_silu(
                rt,
                Cols::dense(&x, 64),
                &w,
                4,
                StateIn::PerBatch(&cs),
                &y,
                Some(&cs),
                1,
                3,
                64,
            ),
            "state_out overlaps read-only buffer state_in",
        );
        expect_err(
            qwen35::conv1d_silu(rt, Cols::dense(&x, 64), &w, 9, StateIn::Zero, &y, None, 1, 3, 64),
            "kernel_width must be 2..=8",
        );

        let shape = AttnShape {
            batch: 1,
            seq: 2,
            q_heads: 1,
            kv_heads: 1,
            head_dim: 64,
            rotary_dim: 63,
        };
        let pb = seeded(rt, 2 * 256, 0.0);
        let nw = seeded(rt, 64, 0.0);
        let qo = seeded(rt, 2 * 64, 0.0);
        // Caches of capacity 2 (batch 1, one KV head of 64).
        let (kc, vc) = (seeded(rt, 2 * 64, 0.0), seeded(rt, 2 * 64, 0.0));
        let targets = AttnTargets {
            q_out: &qo,
            k_cache: &kc,
            v_cache: &vc,
        };
        let rope = |shape: &AttnShape, t: &AttnTargets<'_>, pos: u32| {
            qwen35::attn_qk_norm_rope(rt, shape, Cols::dense(&pb, 256), &nw, &nw, t, pos, 1e4, 1e-6)
        };
        expect_err(rope(&shape, &targets, 0), "rotary_dim must be even");
        let shape = AttnShape {
            rotary_dim: 64,
            ..shape
        };
        rope(&shape, &targets, 0).expect("positions 0..2 fit capacity 2");
        expect_err(rope(&shape, &targets, 1), "exceed the caches' capacity 2");
        let vc3 = seeded(rt, 3 * 64, 0.0);
        let uneven = AttnTargets {
            v_cache: &vc3,
            ..targets
        };
        expect_err(rope(&shape, &uneven, 0), "K and V imply different fixed capacities");

        expect_err(
            qwen35::gated_rms_norm(
                rt,
                Cols::dense(&x, 64),
                Cols::dense(&x, 64),
                &nw,
                OutCols {
                    cols: Cols::dense(&y, 64),
                    dtype: DType::F16,
                },
                1,
                1,
                64,
                1e-6,
            ),
            "dtype must be F32 or BF16",
        );
        expect_err(
            qwen35::attn_output_gate(
                rt,
                &x,
                Cols::dense(&pb, 100),
                OutCols {
                    cols: Cols::dense(&y, 64),
                    dtype: DType::F32,
                },
                1,
                1,
                64,
            ),
            "does not fit a row",
        );

        let ans = buf_u32(rt, &[0, 1]);
        let slots = buf_u32(rt, &[0]);
        let (lg, lp) = (seeded(rt, 2, 0.0), seeded(rt, 2, 0.0));
        let head = LmHead {
            weight: &w,
            dtype: DType::F32,
            vocab: 4,
        };
        let score = |rows: u32, head: LmHead<'_>, n_ans: u32| {
            qwen35::score_answer_rows(rt, &x, rows, 64, &slots, 1, &nw, 1.0, 1e-6, head, &ans, n_ans, &lg, &lp)
        };
        score(3, head, 2).expect("a valid scoring call");
        expect_err(score(0, head, 2), "rows and vocab must be non-zero");
        expect_err(
            score(3, LmHead { vocab: 0, ..head }, 2),
            "rows and vocab must be non-zero",
        );
        expect_err(score(3, head, 0), "n_answers in 1..=");
        expect_err(
            score(
                3,
                LmHead {
                    dtype: DType::F16,
                    ..head
                },
                2,
            ),
            "lm_head: dtype must be F32 or BF16",
        );
        rt.synchronize().unwrap();
    });
}

#[test]
fn attention_layer_through_flash_attn_rows() {
    // The seam the Qwen3.5 attention kernels exist to feed: attn_qk_norm_rope
    // writes q and the caches, tessl's own flash_attn_rows reads them, and
    // attn_output_gate gates its output — continuing a cache that already
    // holds a prefix, at Qwen3.5's head_dim and rotary width.
    with_gpu(|rt| {
        let (b, t, hq, hkv, d, rot) = (2usize, 3usize, 2usize, 1usize, 256usize, 64usize);
        let (prefix, cap) = (5usize, 12usize);
        let layout = AttnProjLayout::new(hq as u32, hkv as u32, d as u32).unwrap();
        let q = random_f32(b * t * hq * d, 1500);
        let k = random_f32(b * t * hkv * d, 1501);
        let v = random_f32(b * t * hkv * d, 1502);
        let gate: Vec<f32> = random_f32(b * t * hq * d, 1503).iter().map(|x| 3.0 * x).collect();
        let qw: Vec<f32> = random_f32(d, 1504).iter().map(|x| 0.1 * x).collect();
        let kw: Vec<f32> = random_f32(d, 1505).iter().map(|x| 0.1 * x).collect();
        let p = attn_proj_rows(layout, b * t, &q, &k, &v, &gate);
        // A prefix already in the caches; the slots after it are the kernel's.
        let cache_len = b * cap * hkv * d;
        let mut kc_host = vec![SENTINEL; cache_len];
        let mut vc_host = vec![SENTINEL; cache_len];
        let pk = random_f32(cache_len, 1506);
        let pv = random_f32(cache_len, 1507);
        for bi in 0..b {
            let r = bi * cap * hkv * d..(bi * cap + prefix) * hkv * d;
            kc_host[r.clone()].copy_from_slice(&pk[r.clone()]);
            vc_host[r.clone()].copy_from_slice(&pv[r]);
        }
        let (pb, qwb, kwb) = (buf(rt, &p), buf(rt, &qw), buf(rt, &kw));
        let (kc, vc) = (buf(rt, &kc_host), buf(rt, &vc_host));
        let q_out = seeded(rt, b * t * hq * d, SENTINEL);
        let targets = AttnTargets {
            q_out: &q_out,
            k_cache: &kc,
            v_cache: &vc,
        };
        let shape = AttnShape {
            batch: b as u32,
            seq: t as u32,
            q_heads: hq as u32,
            kv_heads: hkv as u32,
            head_dim: d as u32,
            rotary_dim: rot as u32,
        };
        let pc = Cols::dense(&pb, layout.width());
        qwen35::attn_qk_norm_rope(rt, &shape, pc, &qwb, &kwb, &targets, prefix as u32, 1e7, 1e-6).unwrap();
        let o = seeded(rt, b * t * hq * d, SENTINEL);
        let scale = 1.0 / (d as f32).sqrt();
        let (tkv, qpos, kvpos) = (
            buf_u32(rt, &[(prefix + t) as u32]),
            buf_u32(rt, &[prefix as u32]),
            buf_u32(rt, &[0]),
        );
        let dims = tessl::nn::AttnDims {
            batch: b as u32,
            tq: t as u32,
            heads: hq as u32,
            heads_kv: hkv as u32,
            window: 0,
            scale,
        };
        tessl::nn::flash_attn_rows(rt, &q_out, &kc, &vc, &o, &tkv, &qpos, &kvpos, dims, d as u32, false).unwrap();
        let out = seeded(rt, b * t * hq * d, SENTINEL);
        let oc = OutCols {
            cols: Cols::dense(&out, (hq * d) as u32),
            dtype: DType::F32,
        };
        qwen35::attn_output_gate(rt, &o, pc, oc, (b * t) as u32, hq as u32, d as u32).unwrap();
        rt.synchronize().unwrap();

        // f64: keys and values per (batch, position), prefix then new tokens.
        let mut want = vec![0.0f64; b * t * hq * d];
        for bi in 0..b {
            let key = |s: usize| -> Vec<f64> {
                if s < prefix {
                    let at = ((bi * cap + s) * hkv) * d;
                    pk[at..at + d].iter().map(|&x| f64::from(x)).collect()
                } else {
                    let r = (bi * t + s - prefix) * hkv * d;
                    norm_rope_row_f64(&k[r..r + d], &kw, rot, s as u64, 1e7, 1e-6)
                }
            };
            let val = |s: usize| -> Vec<f64> {
                let src = if s < prefix {
                    &pv[((bi * cap + s) * hkv) * d..]
                } else {
                    &v[(bi * t + s - prefix) * hkv * d..]
                };
                src[..d].iter().map(|&x| f64::from(x)).collect()
            };
            for ti in 0..t {
                let at = prefix + ti;
                for h in 0..hq {
                    let r = (bi * t + ti) * hq + h;
                    let qv = norm_rope_row_f64(&q[r * d..(r + 1) * d], &qw, rot, at as u64, 1e7, 1e-6);
                    let scores: Vec<f64> = (0..=at)
                        .map(|s| key(s).iter().zip(&qv).map(|(a, b)| a * b).sum::<f64>() * f64::from(scale))
                        .collect();
                    let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let w: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
                    let z: f64 = w.iter().sum();
                    for (s, ws) in w.iter().enumerate() {
                        let vs = val(s);
                        for i in 0..d {
                            want[r * d + i] += ws / z * vs[i];
                        }
                    }
                    for i in 0..d {
                        want[r * d + i] *= sigmoid(f64::from(gate[r * d + i]));
                    }
                }
            }
        }
        assert_close_rel("gated attention", &out.read_f32()[..want.len()], &want, 1e-4, 1e-5);
    });
}

#[test]
fn attn_qk_norm_rope_posbuf_matches_the_scalar_variant_step_by_step() {
    // The decode-loop shape the posbuf variant exists for: one call per step,
    // the position advanced through a device buffer between them (as an ICB
    // replay would see it), against the scalar variant at the same positions.
    with_gpu(|rt| {
        let (b, hq, hkv, d, cap) = (2usize, 2usize, 1usize, 256usize, 8usize);
        let layout = AttnProjLayout::new(hq as u32, hkv as u32, d as u32).unwrap();
        let shape = AttnShape {
            batch: b as u32,
            seq: 1,
            q_heads: hq as u32,
            kv_heads: hkv as u32,
            head_dim: d as u32,
            rotary_dim: 64,
        };
        let (qw, kw) = (buf(rt, &random_f32(d, 1600)), buf(rt, &random_f32(d, 1601)));
        let cache = b * cap * hkv * d;
        let (kc_a, vc_a) = (seeded(rt, cache, SENTINEL), seeded(rt, cache, SENTINEL));
        let (kc_b, vc_b) = (seeded(rt, cache, SENTINEL), seeded(rt, cache, SENTINEL));
        let pos_buf = buf_u32(rt, &[0]);
        for step in 0..4u32 {
            let p = buf(rt, &random_f32(b * layout.width() as usize, 1610 + u64::from(step)));
            let pc = Cols::dense(&p, layout.width());
            let (qa, qb) = (seeded(rt, b * hq * d, SENTINEL), seeded(rt, b * hq * d, SENTINEL));
            let ta = AttnTargets {
                q_out: &qa,
                k_cache: &kc_a,
                v_cache: &vc_a,
            };
            let tb = AttnTargets {
                q_out: &qb,
                k_cache: &kc_b,
                v_cache: &vc_b,
            };
            pos_buf.write_u32(&[3 + step]);
            qwen35::attn_qk_norm_rope(rt, &shape, pc, &qw, &kw, &ta, 3 + step, 1e7, 1e-6).unwrap();
            qwen35::attn_qk_norm_rope_posbuf(rt, &shape, pc, &qw, &kw, &tb, &pos_buf, 1e7, 1e-6).unwrap();
            // The host rewrites the position next step: the GPU must be done.
            rt.synchronize().unwrap();
            assert_eq!(qa.read_f32(), qb.read_f32(), "q at step {step}");
        }
        assert_eq!(kc_a.read_f32(), kc_b.read_f32(), "k cache");
        assert_eq!(vc_a.read_f32(), vc_b.read_f32(), "v cache");
        // Past the capacity the posbuf variant writes nothing at all.
        let before = kc_b.read_f32();
        pos_buf.write_u32(&[u32::MAX - 1]);
        let p = buf(rt, &random_f32(b * layout.width() as usize, 1620));
        let q = seeded(rt, b * hq * d, SENTINEL);
        let t = AttnTargets {
            q_out: &q,
            k_cache: &kc_b,
            v_cache: &vc_b,
        };
        qwen35::attn_qk_norm_rope_posbuf(
            rt,
            &shape,
            Cols::dense(&p, layout.width()),
            &qw,
            &kw,
            &t,
            &pos_buf,
            1e7,
            1e-6,
        )
        .unwrap();
        rt.synchronize().unwrap();
        assert_eq!(kc_b.read_f32(), before, "an out-of-range position wrote the cache");
        assert!(
            q.read_f32().iter().all(|&x| x == SENTINEL),
            "an out-of-range position wrote q"
        );
    });
}

// ------------------------------------------------- shared-prefix attention ---

/// Qwen3.5's full-attention heads: 8 query heads over 2 KV heads of 256.
const PFX_HQ: usize = 8;
const PFX_HKV: usize = 2;
const PFX_D: usize = 256;

fn pfx_dims(batch: usize, tq: usize) -> tessl::nn::AttnDims {
    tessl::nn::AttnDims {
        batch: batch as u32,
        tq: tq as u32,
        heads: PFX_HQ as u32,
        heads_kv: PFX_HKV as u32,
        window: 0,
        scale: 1.0 / (PFX_D as f32).sqrt(),
    }
}

/// Output words as bits: `f32` words, or the bf16 halves when `bf16`.
fn out_bits(o: &GpuBuffer, n: usize, bf16: bool) -> Vec<u32> {
    if bf16 {
        o.contents_u16()[..n].iter().map(|&x| u32::from(x)).collect()
    } else {
        o.read_f32()[..n].iter().map(|x| x.to_bits()).collect()
    }
}

/// One shared-prefix case against its `nn` counterpart (`flash_attn_rows`,
/// or `flash_attn_decode` when `decode`) over the per-row copy
/// `prefix ‖ suffix_b`. Every K/V slot past the live prefix and suffix is NaN,
/// in both layouts, so a kernel that read one would print NaN into the output.
#[allow(clippy::too_many_arguments)]
fn prefix_case(
    rt: &Arc<GpuRuntime>,
    decode: bool,
    batch: usize,
    p: usize,
    s: usize,
    s_cap: usize,
    tq: usize,
    q_pos: usize,
    seed: u64,
) {
    let row = PFX_HKV * PFX_D;
    let p_cap = p + 3;
    let mut pk = random_f32(p_cap * row, seed);
    let mut pv = random_f32(p_cap * row, seed + 1);
    pk[p * row..].fill(f32::NAN);
    pv[p * row..].fill(f32::NAN);
    let mut sk = random_f32(batch * s_cap * row, seed + 2);
    let mut sv = random_f32(batch * s_cap * row, seed + 3);
    for bi in 0..batch {
        let dead = (bi * s_cap + s) * row..(bi + 1) * s_cap * row;
        sk[dead.clone()].fill(f32::NAN);
        sv[dead].fill(f32::NAN);
    }
    // The per-row copy the shared layout exists to avoid.
    let full_cap = p + s_cap;
    let (mut fk, mut fv) = (Vec::new(), Vec::new());
    for bi in 0..batch {
        fk.extend_from_slice(&pk[..p * row]);
        fv.extend_from_slice(&pv[..p * row]);
        fk.extend_from_slice(&sk[bi * s_cap * row..(bi + 1) * s_cap * row]);
        fv.extend_from_slice(&sv[bi * s_cap * row..(bi + 1) * s_cap * row]);
    }
    assert_eq!(fk.len(), batch * full_cap * row);
    let n = batch * tq * PFX_HQ * PFX_D;
    let q = buf(rt, &random_f32(n, seed + 4));
    let (pkb, pvb, skb, svb) = (buf(rt, &pk), buf(rt, &pv), buf(rt, &sk), buf(rt, &sv));
    let (fkb, fvb) = (buf(rt, &fk), buf(rt, &fv));
    let slen = buf_u32(rt, &[s as u32]);
    let qpos = buf_u32(rt, &[q_pos as u32]);
    let tkv = buf_u32(rt, &[(p + s) as u32]);
    let zero = buf_u32(rt, &[0]);
    let dims = pfx_dims(batch, tq);
    for bf16 in [false, true] {
        let got = seeded(rt, n, SENTINEL);
        let want = seeded(rt, n, SENTINEL);
        let prefix = qwen35::SharedPrefix {
            k: &pkb,
            v: &pvb,
            len: p as u32,
        };
        if decode {
            let scratch = tessl::nn::DecodeScratch::new(rt, dims.batch, dims.heads, full_cap, PFX_D as u32).unwrap();
            qwen35::attn_prefix_decode(rt, &q, prefix, &skb, &svb, &slen, &qpos, &got, &scratch, dims, bf16).unwrap();
            tessl::nn::flash_attn_decode(
                rt,
                &q,
                &fkb,
                &fvb,
                &want,
                &scratch,
                &tkv,
                &qpos,
                &zero,
                dims,
                PFX_D as u32,
                full_cap,
                bf16,
            )
            .unwrap();
        } else {
            qwen35::attn_prefix_rows(rt, &q, prefix, &skb, &svb, &slen, &qpos, &got, dims, bf16).unwrap();
            tessl::nn::flash_attn_rows(rt, &q, &fkb, &fvb, &want, &tkv, &qpos, &zero, dims, PFX_D as u32, bf16)
                .unwrap();
        }
        rt.synchronize().unwrap();
        let path = if decode { "decode" } else { "rows" };
        let label = format!("{path} B{batch} P{p} S{s}/{s_cap} Tq{tq} q@{q_pos} bf16={bf16}");
        let (g, w) = (out_bits(&got, n, bf16), out_bits(&want, n, bf16));
        for (i, (&gi, &wi)) in g.iter().zip(&w).enumerate() {
            // A half of SENTINEL can be a real bf16 value, so unwritten
            // elements are checked on the f32 pass, which writes the same set.
            assert!(bf16 || gi != SENTINEL.to_bits(), "{label}[{i}]: never written");
            assert_eq!(gi, wi, "{label}[{i}]: shared prefix {gi:#x} vs copied prefix {wi:#x}");
        }
        let finite = if bf16 {
            read_bf16(&got, n).iter().all(|x| x.is_finite())
        } else {
            got.read_f32()[..n].iter().all(|x| x.is_finite())
        };
        assert!(finite, "{label}: read a poisoned K/V slot past a live length");
    }
}

#[test]
fn attn_prefix_rows_equal_flash_attn_rows_on_a_copied_prefix_bit_for_bit() {
    // The gate for the shared-prefix kernel: the same bits as tessl's own
    // flash_attn_rows run on each row's copied `prefix ‖ suffix`, at Qwen3.5's
    // head shape. Prefix lengths at 0, 1, and either side of the 64-row
    // query tile and a KV chunk; a long prefix; queries at the end of the
    // suffix, inside the prefix (no suffix), and past every key.
    with_gpu(|rt| {
        #[rustfmt::skip]
        let cases: &[(usize, usize, usize, usize, usize, usize)] = &[
            // (batch, P, S, S capacity, Tq, q position)
            (1, 0, 5, 8, 5, 0),
            (3, 1, 4, 4, 4, 1),
            (2, 63, 1, 2, 1, 63),
            (4, 64, 2, 2, 1, 65),
            (5, 65, 7, 9, 7, 65),
            (16, 127, 3, 3, 2, 128),
            (7, 128, 65, 70, 65, 128),
            (2, 129, 64, 64, 64, 129),
            (3, 300, 0, 1, 2, 298),
            (2, 40, 3, 4, 3, 50),
            (16, 2100, 5, 6, 5, 2100),
            (9, 2049, 1, 1, 1, 2049),
        ];
        for (i, &(b, p, s, cap, tq, qp)) in cases.iter().enumerate() {
            prefix_case(rt, false, b, p, s, cap, tq, qp, 7000 + 10 * i as u64);
        }
        // Every batch size 1..=16 on one mid-sized shape.
        for b in 1..=16usize {
            prefix_case(rt, false, b, 70, 3, 5, 3, 70, 7500 + b as u64);
        }
    });
}

#[test]
fn attn_prefix_decode_equals_flash_attn_decode_on_a_copied_prefix_bit_for_bit() {
    // The split-KV decode against tessl's own flash_attn_decode on each row's
    // copied `prefix ‖ suffix`. Its chunks are 128 keys, so the prefix sits
    // at 0, 1 and either side of one and two chunks, and chunks straddle the
    // prefix/suffix boundary. Also a long prefix, a step whose query is
    // inside the prefix (no suffix), and one past every key.
    with_gpu(|rt| {
        #[rustfmt::skip]
        let cases: &[(usize, usize, usize, usize, usize)] = &[
            // (batch, P, S, S capacity, q position)
            (1, 0, 1, 4, 0),
            (3, 1, 1, 1, 1),
            (2, 127, 1, 2, 127),
            (4, 128, 1, 1, 128),
            (5, 129, 3, 130, 131),
            (16, 255, 2, 2, 256),
            (7, 256, 1, 1, 256),
            (2, 257, 200, 256, 456),
            (3, 300, 0, 1, 250),
            (2, 40, 3, 4, 60),
            (16, 2100, 5, 6, 2104),
            (9, 8200, 1, 1, 8200),
        ];
        for (i, &(b, p, s, cap, qp)) in cases.iter().enumerate() {
            prefix_case(rt, true, b, p, s, cap, 1, qp, 7800 + 10 * i as u64);
        }
        for b in 1..=16usize {
            prefix_case(rt, true, b, 190, 4, 6, 1, 193, 7950 + b as u64);
        }
    });
}

#[test]
fn attn_prefix_rows_continue_a_prefix_written_by_attn_qk_norm_rope_suffix() {
    // End to end: the prefix is prefilled once (batch 1) with
    // attn_qk_norm_rope, each row's continuation is cached relative to it by
    // attn_qk_norm_rope_suffix, and attention reads both. The reference writes
    // every row's full cache with attn_qk_norm_rope and runs flash_attn_rows.
    // The suffix must be rotated at its absolute position but stored at its
    // relative slot; either mistake changes the bits.
    with_gpu(|rt| {
        let (b, p, t, rot) = (4usize, 70usize, 5usize, 64usize);
        let s_cap = t + 2;
        let layout = AttnProjLayout::new(PFX_HQ as u32, PFX_HKV as u32, PFX_D as u32).unwrap();
        let w = layout.width() as usize;
        let qw: Vec<f32> = random_f32(PFX_D, 7601).iter().map(|x| 0.1 * x).collect();
        let kw: Vec<f32> = random_f32(PFX_D, 7602).iter().map(|x| 0.1 * x).collect();
        let (qwb, kwb) = (buf(rt, &qw), buf(rt, &kw));
        let proj_p = random_f32(p * w, 7603);
        let proj_s = random_f32(b * t * w, 7604);
        let shape = |batch: usize, seq: usize| AttnShape {
            batch: batch as u32,
            seq: seq as u32,
            q_heads: PFX_HQ as u32,
            kv_heads: PFX_HKV as u32,
            head_dim: PFX_D as u32,
            rotary_dim: rot as u32,
        };
        let row = PFX_HKV * PFX_D;

        // Shared: one prefix cache, per-row suffix caches.
        let (pp, ps) = (buf(rt, &proj_p), buf(rt, &proj_s));
        let pq = seeded(rt, p * PFX_HQ * PFX_D, SENTINEL);
        let (pk, pv) = (seeded(rt, p * row, SENTINEL), seeded(rt, p * row, SENTINEL));
        let prefill = AttnTargets {
            q_out: &pq,
            k_cache: &pk,
            v_cache: &pv,
        };
        qwen35::attn_qk_norm_rope(
            rt,
            &shape(1, p),
            Cols::dense(&pp, w as u32),
            &qwb,
            &kwb,
            &prefill,
            0,
            1e7,
            1e-6,
        )
        .unwrap();
        let sq = seeded(rt, b * t * PFX_HQ * PFX_D, SENTINEL);
        let sk = seeded(rt, b * s_cap * row, SENTINEL);
        let sv = seeded(rt, b * s_cap * row, SENTINEL);
        let suffix = AttnTargets {
            q_out: &sq,
            k_cache: &sk,
            v_cache: &sv,
        };
        qwen35::attn_qk_norm_rope_suffix(
            rt,
            &shape(b, t),
            Cols::dense(&ps, w as u32),
            &qwb,
            &kwb,
            &suffix,
            p as u32,
            0,
            1e7,
            1e-6,
        )
        .unwrap();
        let n = b * t * PFX_HQ * PFX_D;
        let got = seeded(rt, n, SENTINEL);
        let (slen, qpos) = (buf_u32(rt, &[t as u32]), buf_u32(rt, &[p as u32]));
        let prefix = qwen35::SharedPrefix {
            k: &pk,
            v: &pv,
            len: p as u32,
        };
        qwen35::attn_prefix_rows(rt, &sq, prefix, &sk, &sv, &slen, &qpos, &got, pfx_dims(b, t), false).unwrap();

        // Reference: every row's whole sequence through attn_qk_norm_rope.
        let mut full_proj = Vec::with_capacity(b * (p + t) * w);
        for bi in 0..b {
            full_proj.extend_from_slice(&proj_p);
            full_proj.extend_from_slice(&proj_s[bi * t * w..(bi + 1) * t * w]);
        }
        let fp = buf(rt, &full_proj);
        let fq = seeded(rt, b * (p + t) * PFX_HQ * PFX_D, SENTINEL);
        let fk = seeded(rt, b * (p + t) * row, SENTINEL);
        let fv = seeded(rt, b * (p + t) * row, SENTINEL);
        let full = AttnTargets {
            q_out: &fq,
            k_cache: &fk,
            v_cache: &fv,
        };
        qwen35::attn_qk_norm_rope(
            rt,
            &shape(b, p + t),
            Cols::dense(&fp, w as u32),
            &qwb,
            &kwb,
            &full,
            0,
            1e7,
            1e-6,
        )
        .unwrap();
        rt.synchronize().unwrap();
        // The suffix queries are the last t of each full row; flash_attn_rows
        // wants them dense, so gather them.
        let fq_host = fq.read_f32();
        let qrow = PFX_HQ * PFX_D;
        let mut tail = Vec::with_capacity(n);
        for bi in 0..b {
            let at = (bi * (p + t) + p) * qrow;
            tail.extend_from_slice(&fq_host[at..at + t * qrow]);
        }
        let sq_host = sq.read_f32();
        for (i, (g, w)) in sq_host[..n].iter().zip(&tail).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "suffix q[{i}]: {g} vs {w}");
        }
        let want = seeded(rt, n, SENTINEL);
        let tq_buf = buf(rt, &tail);
        let (tkv, zero) = (buf_u32(rt, &[(p + t) as u32]), buf_u32(rt, &[0]));
        tessl::nn::flash_attn_rows(
            rt,
            &tq_buf,
            &fk,
            &fv,
            &want,
            &tkv,
            &qpos,
            &zero,
            pfx_dims(b, t),
            PFX_D as u32,
            false,
        )
        .unwrap();
        rt.synchronize().unwrap();
        let (g, w) = (got.read_f32(), want.read_f32());
        for i in 0..n {
            assert!(g[i] != SENTINEL, "attention[{i}]: never written");
            assert_eq!(g[i].to_bits(), w[i].to_bits(), "attention[{i}]: {} vs {}", g[i], w[i]);
        }
        // The slots past the live suffix were never written.
        let sk_host = sk.read_f32();
        for bi in 0..b {
            let dead = &sk_host[(bi * s_cap + t) * row..(bi + 1) * s_cap * row];
            assert!(dead.iter().all(|&x| x == SENTINEL), "row {bi} wrote past its suffix");
        }
    });
}

#[test]
fn attn_qk_norm_rope_suffix_posbuf_matches_the_scalar_suffix_writer_step_by_step() {
    // The ICB decode shape: one token per step, the absolute position advanced
    // through a device buffer, against the scalar suffix writer at the same
    // positions. Then positions the kernel must skip, since the host cannot
    // see them: before the suffix's first slot, past its capacity, and an
    // offset that would wrap in 32 bits.
    with_gpu(|rt| {
        let (b, p, cap, rot) = (3usize, 20u32, 4usize, 64u32);
        let layout = AttnProjLayout::new(PFX_HQ as u32, PFX_HKV as u32, PFX_D as u32).unwrap();
        let w = layout.width();
        let row = PFX_HKV * PFX_D;
        let qw: Vec<f32> = random_f32(PFX_D, 7801).iter().map(|x| 0.1 * x).collect();
        let kw: Vec<f32> = random_f32(PFX_D, 7802).iter().map(|x| 0.1 * x).collect();
        let (qwb, kwb) = (buf(rt, &qw), buf(rt, &kw));
        let shape = AttnShape {
            batch: b as u32,
            seq: 1,
            q_heads: PFX_HQ as u32,
            kv_heads: PFX_HKV as u32,
            head_dim: PFX_D as u32,
            rotary_dim: rot,
        };
        let qn = b * PFX_HQ * PFX_D;
        let fresh = || {
            (
                seeded(rt, qn, SENTINEL),
                seeded(rt, b * cap * row, SENTINEL),
                seeded(rt, b * cap * row, SENTINEL),
            )
        };
        let (sq, sk, sv) = fresh();
        let (bq, bk, bv) = fresh();
        let pos = buf_u32(rt, &[0]);
        for step in 0..cap as u32 {
            let proj = buf(rt, &random_f32(b * w as usize, 7810 + u64::from(step)));
            let pc = Cols::dense(&proj, w);
            let scalar = AttnTargets {
                q_out: &sq,
                k_cache: &sk,
                v_cache: &sv,
            };
            qwen35::attn_qk_norm_rope_suffix(rt, &shape, pc, &qwb, &kwb, &scalar, p, step, 1e7, 1e-6).unwrap();
            pos.write_u32(&[p + step]);
            let buffered = AttnTargets {
                q_out: &bq,
                k_cache: &bk,
                v_cache: &bv,
            };
            qwen35::attn_qk_norm_rope_suffix_posbuf(rt, &shape, pc, &qwb, &kwb, &buffered, p, &pos, 1e7, 1e-6).unwrap();
            rt.synchronize().unwrap();
            for (name, x, y) in [("q", &sq, &bq), ("k", &sk, &bk), ("v", &sv, &bv)] {
                let (x, y) = (x.read_f32(), y.read_f32());
                assert!(
                    x.iter().zip(&y).all(|(a, c)| a.to_bits() == c.to_bits()),
                    "step {step}: {name} differs between the scalar and buffer positions"
                );
            }
        }
        // Positions whose slot is out of range write nothing at all.
        let proj = buf(rt, &random_f32(b * w as usize, 7820));
        for bad in [p - 1, p + cap as u32, u32::MAX] {
            let (q, k, v) = fresh();
            pos.write_u32(&[bad]);
            let t = AttnTargets {
                q_out: &q,
                k_cache: &k,
                v_cache: &v,
            };
            qwen35::attn_qk_norm_rope_suffix_posbuf(
                rt,
                &shape,
                Cols::dense(&proj, w),
                &qwb,
                &kwb,
                &t,
                p,
                &pos,
                1e7,
                1e-6,
            )
            .unwrap();
            rt.synchronize().unwrap();
            for (name, x) in [("q", &q), ("k", &k), ("v", &v)] {
                assert!(
                    x.read_f32().iter().all(|&e| e == SENTINEL),
                    "position {bad}: {name} was written"
                );
            }
        }
    });
}

/// One ragged shared-prefix batch against each of its rows run alone through
/// the equal-length entry point. `rows` holds each row's (live suffix
/// length, query position); a length past `s_cap` is clamped by the kernel,
/// and the reference is given the clamped value. Every K/V slot past a row's
/// own live length is NaN.
fn prefix_varlen_case(
    rt: &Arc<GpuRuntime>,
    decode: bool,
    p: usize,
    s_cap: usize,
    tq: usize,
    rows: &[(usize, usize)],
    seed: u64,
) {
    let batch = rows.len();
    let row = PFX_HKV * PFX_D;
    let mut pk = random_f32((p + 1) * row, seed);
    let mut pv = random_f32((p + 1) * row, seed + 1);
    pk[p * row..].fill(f32::NAN);
    pv[p * row..].fill(f32::NAN);
    let mut sk = random_f32(batch * s_cap * row, seed + 2);
    let mut sv = random_f32(batch * s_cap * row, seed + 3);
    for (bi, &(len, _)) in rows.iter().enumerate() {
        let dead = (bi * s_cap + len.min(s_cap)) * row..(bi + 1) * s_cap * row;
        sk[dead.clone()].fill(f32::NAN);
        sv[dead].fill(f32::NAN);
    }
    let per_q = tq * PFX_HQ * PFX_D;
    let q_host = random_f32(batch * per_q, seed + 4);
    let (pkb, pvb) = (buf(rt, &pk), buf(rt, &pv));
    let prefix = qwen35::SharedPrefix {
        k: &pkb,
        v: &pvb,
        len: p as u32,
    };
    let (q, skb, svb) = (buf(rt, &q_host), buf(rt, &sk), buf(rt, &sv));
    let lens: Vec<u32> = rows.iter().map(|r| r.0 as u32).collect();
    let qpos: Vec<u32> = rows.iter().map(|r| r.1 as u32).collect();
    let (lb, qb) = (buf_u32(rt, &lens), buf_u32(rt, &qpos));
    let n = batch * per_q;
    let got = seeded(rt, n, SENTINEL);
    let dims = pfx_dims(batch, tq);
    let scratch = tessl::nn::DecodeScratch::new(rt, batch as u32, PFX_HQ as u32, p + s_cap, PFX_D as u32).unwrap();
    if decode {
        qwen35::attn_prefix_decode_varlen(rt, &q, prefix, &skb, &svb, &lb, &qb, &got, &scratch, dims, false)
    } else {
        qwen35::attn_prefix_rows_varlen(rt, &q, prefix, &skb, &svb, &lb, &qb, &got, dims, false)
    }
    .unwrap();
    rt.synchronize().unwrap();
    let got = got.read_f32();
    for (bi, &(len, q_pos)) in rows.iter().enumerate() {
        let one = |v: &[f32], per: usize| buf(rt, &v[bi * per..(bi + 1) * per]);
        let (q1, sk1, sv1) = (one(&q_host, per_q), one(&sk, s_cap * row), one(&sv, s_cap * row));
        let (l1, p1) = (buf_u32(rt, &[len.min(s_cap) as u32]), buf_u32(rt, &[q_pos as u32]));
        let want = seeded(rt, per_q, SENTINEL);
        let d1 = pfx_dims(1, tq);
        if decode {
            qwen35::attn_prefix_decode(rt, &q1, prefix, &sk1, &sv1, &l1, &p1, &want, &scratch, d1, false)
        } else {
            qwen35::attn_prefix_rows(rt, &q1, prefix, &sk1, &sv1, &l1, &p1, &want, d1, false)
        }
        .unwrap();
        rt.synchronize().unwrap();
        let want = want.read_f32();
        let g = &got[bi * per_q..(bi + 1) * per_q];
        let path = if decode { "decode" } else { "rows" };
        for (i, (a, w)) in g.iter().zip(&want[..per_q]).enumerate() {
            assert!(*a != SENTINEL, "{path} row {bi}[{i}]: never written");
            assert_eq!(
                a.to_bits(),
                w.to_bits(),
                "{path} P{p} row {bi} (len {len}, q@{q_pos})[{i}]: {a} vs {w} alone"
            );
        }
    }
}

#[test]
fn attn_prefix_varlen_rows_equal_each_row_alone_bit_for_bit() {
    // Ragged continuations of one prefix, one call: each row must be exactly
    // what the equal-length path computes for it alone. Lengths 0, 1, either
    // side of the 64-row query tile and the 128-key decode chunk, full, and
    // one past the capacity (clamped).
    with_gpu(|rt| {
        let p = 130;
        let rows_prefill: Vec<(usize, usize)> = [0usize, 1, 63, 64, 65, 70, 90].iter().map(|&l| (l, p)).collect();
        prefix_varlen_case(rt, false, p, 70, 66, &rows_prefill, 8000);
        // Queries at different positions in one call (as after a ragged
        // prefill), including one inside the prefix and one past every key.
        prefix_varlen_case(rt, false, p, 8, 3, &[(5, 132), (2, 10), (8, 200), (0, 129)], 8020);
        // Decode: each row's one query at its own next position.
        let rows_decode: Vec<(usize, usize)> = [1usize, 2, 126, 127, 128, 129, 200, 250, 300]
            .iter()
            .map(|&l| (l, p + l.min(256) - 1))
            .collect();
        prefix_varlen_case(rt, true, p, 256, 1, &rows_decode, 8040);
        prefix_varlen_case(rt, true, 0, 5, 1, &[(1, 0), (5, 4), (3, 2)], 8060);
    });
}

#[test]
fn attn_qk_norm_rope_suffix_rows_equal_each_row_alone_bit_for_bit() {
    // The per-row suffix writer: row b's tokens rotated from its own absolute
    // position and cached at that position less the prefix, exactly as the
    // scalar suffix writer does for the row alone. A row whose slot is out of
    // range writes nothing.
    with_gpu(|rt| {
        let (p, cap, t, rot) = (40u32, 6usize, 2usize, 64u32);
        let positions = [40u32, 43, 44, 39, 46];
        let b = positions.len();
        let layout = AttnProjLayout::new(PFX_HQ as u32, PFX_HKV as u32, PFX_D as u32).unwrap();
        let w = layout.width() as usize;
        let row = PFX_HKV * PFX_D;
        let qw: Vec<f32> = random_f32(PFX_D, 8101).iter().map(|x| 0.1 * x).collect();
        let kw: Vec<f32> = random_f32(PFX_D, 8102).iter().map(|x| 0.1 * x).collect();
        let (qwb, kwb) = (buf(rt, &qw), buf(rt, &kw));
        let proj_host = random_f32(b * t * w, 8103);
        let shape = |batch: usize| AttnShape {
            batch: batch as u32,
            seq: t as u32,
            q_heads: PFX_HQ as u32,
            kv_heads: PFX_HKV as u32,
            head_dim: PFX_D as u32,
            rotary_dim: rot,
        };
        let per_q = t * PFX_HQ * PFX_D;
        let (q, k, v) = (
            seeded(rt, b * per_q, SENTINEL),
            seeded(rt, b * cap * row, SENTINEL),
            seeded(rt, b * cap * row, SENTINEL),
        );
        let proj = buf(rt, &proj_host);
        let pos = buf_u32(rt, &positions);
        let targets = AttnTargets {
            q_out: &q,
            k_cache: &k,
            v_cache: &v,
        };
        qwen35::attn_qk_norm_rope_suffix_rows(
            rt,
            &shape(b),
            Cols::dense(&proj, w as u32),
            &qwb,
            &kwb,
            &targets,
            p,
            &pos,
            1e7,
            1e-6,
        )
        .unwrap();
        rt.synchronize().unwrap();
        let (qh, kh, vh) = (q.read_f32(), k.read_f32(), v.read_f32());
        for (bi, &at) in positions.iter().enumerate() {
            let (q1, k1, v1) = (
                seeded(rt, per_q, SENTINEL),
                seeded(rt, cap * row, SENTINEL),
                seeded(rt, cap * row, SENTINEL),
            );
            let proj1 = buf(rt, &proj_host[bi * t * w..(bi + 1) * t * w]);
            let t1 = AttnTargets {
                q_out: &q1,
                k_cache: &k1,
                v_cache: &v1,
            };
            // Alone, through the per-row buffer writer at batch 1, which the
            // scalar and shared-buffer writers are already held equal to; it
            // also skips an out-of-range slot the way the kernel must.
            qwen35::attn_qk_norm_rope_suffix_posbuf(
                rt,
                &shape(1),
                Cols::dense(&proj1, w as u32),
                &qwb,
                &kwb,
                &t1,
                p,
                &buf_u32(rt, &[at]),
                1e7,
                1e-6,
            )
            .unwrap();
            rt.synchronize().unwrap();
            for (name, all, alone, per) in [
                ("q", &qh, q1.read_f32(), per_q),
                ("k", &kh, k1.read_f32(), cap * row),
                ("v", &vh, v1.read_f32(), cap * row),
            ] {
                let mine = &all[bi * per..(bi + 1) * per];
                assert!(
                    mine.iter().zip(&alone[..per]).all(|(a, c)| a.to_bits() == c.to_bits()),
                    "row {bi} at {at}: {name} differs from the row alone"
                );
            }
        }
        // The skip is per token. Row 3 starts at 39, before the prefix ends at
        // 40: its token 0 is skipped and token 1 lands in slot 0. Row 4's
        // slots 6 and 7 are past capacity 6: nothing. Row 2 fills slots 4, 5.
        let slot = |bi: usize, s: usize| &kh[(bi * cap + s) * row..(bi * cap + s + 1) * row];
        let written = |bi: usize, s: usize| slot(bi, s).iter().all(|&x| x != SENTINEL);
        let empty = |bi: usize, s: usize| slot(bi, s).iter().all(|&x| x == SENTINEL);
        assert!(written(3, 0) && (1..cap).all(|s| empty(3, s)), "row 3 slots");
        assert!((0..cap).all(|s| empty(4, s)), "row 4 slots");
        assert!(
            (0..4).all(|s| empty(2, s)) && written(2, 4) && written(2, 5),
            "row 2 slots"
        );
    });
}

#[test]
fn attn_prefix_rows_rejects_bad_shapes_and_aliases() {
    with_gpu(|rt| {
        let (b, tq, p, s_cap) = (2usize, 3usize, 4usize, 4usize);
        let row = PFX_HKV * PFX_D;
        let n = b * tq * PFX_HQ * PFX_D;
        let q = buf(rt, &random_f32(n, 7700));
        let (pk, pv) = (buf(rt, &random_f32(p * row, 7701)), buf(rt, &random_f32(p * row, 7702)));
        let (sk, sv) = (
            buf(rt, &random_f32(b * s_cap * row, 7703)),
            buf(rt, &random_f32(b * s_cap * row, 7704)),
        );
        let (slen, qpos) = (buf_u32(rt, &[3]), buf_u32(rt, &[p as u32]));
        let o = seeded(rt, n, SENTINEL);
        let prefix = qwen35::SharedPrefix {
            k: &pk,
            v: &pv,
            len: p as u32,
        };
        let run = |prefix: qwen35::SharedPrefix<'_>, o: &GpuBuffer, dims: tessl::nn::AttnDims| {
            qwen35::attn_prefix_rows(rt, &q, prefix, &sk, &sv, &slen, &qpos, o, dims, false)
        };
        let mut windowed = pfx_dims(b, tq);
        windowed.window = 16;
        expect_err(run(prefix, &o, windowed), "window must be 0");
        let too_long = qwen35::SharedPrefix {
            len: p as u32 + 1,
            ..prefix
        };
        expect_err(run(too_long, &o, pfx_dims(b, tq)), "exceeds the prefix K/V capacity");
        // Writing the output over the shared prefix would corrupt every row.
        // K and V of one capacity, so only the alias can be what fails.
        let pk_as_o = seeded(rt, n, SENTINEL);
        let pv_same = seeded(rt, n, 0.0);
        let aliased = qwen35::SharedPrefix {
            k: &pk_as_o,
            v: &pv_same,
            len: p as u32,
        };
        expect_err(
            run(aliased, &pk_as_o, pfx_dims(b, tq)),
            "overlaps read-only buffer prefix k",
        );
        let mut grouped = pfx_dims(b, tq);
        grouped.heads_kv = 3;
        expect_err(run(prefix, &o, grouped), "is not a multiple of heads_kv");
        // The decode path takes one query per row, and shares every other
        // check with the rows path.
        let scratch = tessl::nn::DecodeScratch::new(rt, b as u32, PFX_HQ as u32, p + 1 + s_cap, PFX_D as u32).unwrap();
        let decode = |prefix: qwen35::SharedPrefix<'_>, dims: tessl::nn::AttnDims| {
            qwen35::attn_prefix_decode(rt, &q, prefix, &sk, &sv, &slen, &qpos, &o, &scratch, dims, false)
        };
        expect_err(decode(prefix, pfx_dims(b, tq)), "one query per row (tq = 1)");
        expect_err(decode(too_long, pfx_dims(b, 1)), "exceeds the prefix K/V capacity");
        // The suffix writer: positions past the suffix cache, and a position
        // that overflows u32.
        let layout = AttnProjLayout::new(PFX_HQ as u32, PFX_HKV as u32, PFX_D as u32).unwrap();
        let w = layout.width();
        let proj = buf(rt, &random_f32(b * tq * w as usize, 7705));
        let qo = seeded(rt, n, SENTINEL);
        let targets = AttnTargets {
            q_out: &qo,
            k_cache: &sk,
            v_cache: &sv,
        };
        let shape = AttnShape {
            batch: b as u32,
            seq: tq as u32,
            q_heads: PFX_HQ as u32,
            kv_heads: PFX_HKV as u32,
            head_dim: PFX_D as u32,
            rotary_dim: 64,
        };
        let (qwb, kwb) = (buf(rt, &vec![0.0; PFX_D]), buf(rt, &vec![0.0; PFX_D]));
        let suffix = |prefix_len: u32, offset: u32| {
            qwen35::attn_qk_norm_rope_suffix(
                rt,
                &shape,
                Cols::dense(&proj, w),
                &qwb,
                &kwb,
                &targets,
                prefix_len,
                offset,
                1e7,
                1e-6,
            )
        };
        expect_err(suffix(1000, 2), "exceed the caches' capacity");
        expect_err(suffix(u32::MAX, 1), "exceeds u32 positions");
        // The first position fits u32 but the cache's last slot does not: the
        // kernel's u32 position would wrap for in-range tokens (rotated at 0,
        // cached at slot 2). Refused for every position mode, as the reader
        // (`validate_prefix_attn`) refuses such a cache.
        expect_err(suffix(u32::MAX - 1, 0), "exceed u32");
        let pos1 = buf_u32(rt, &[u32::MAX - 1]);
        expect_err(
            qwen35::attn_qk_norm_rope_suffix_posbuf(
                rt,
                &shape,
                Cols::dense(&proj, w),
                &qwb,
                &kwb,
                &targets,
                u32::MAX - 1,
                &pos1,
                1e7,
                1e-6,
            ),
            "exceed u32",
        );
        // Caches too small for one position: every token would be skipped, a
        // silent no-op. The host knows the capacity whatever the mode.
        let (tiny_k, tiny_v) = (buf(rt, &[0.0; 4]), buf(rt, &[0.0; 4]));
        let tiny = AttnTargets {
            q_out: &qo,
            k_cache: &tiny_k,
            v_cache: &tiny_v,
        };
        let pos0 = buf_u32(rt, &[0]);
        expect_err(
            qwen35::attn_qk_norm_rope_posbuf(rt, &shape, Cols::dense(&proj, w), &qwb, &kwb, &tiny, &pos0, 1e7, 1e-6),
            "capacity 0",
        );
        suffix(1000, 1).unwrap();
        rt.synchronize().unwrap();
    });
}

// ------------------------------------------------------------ embedding ---

#[test]
fn embed_rows_equals_the_host_gather_bit_for_bit() {
    // The device gather qd-metal needs to keep a forward in one command buffer,
    // against the host gather it replaces: bf16 bits widened to f32, which is
    // exact, so the bits must match. Qwen3.5's hidden size, a small vocab.
    with_gpu(|rt| {
        let (vocab, hidden) = (1000usize, 2048usize);
        let bits = f32_slice_to_bf16(&random_f32(vocab * hidden, 7900));
        let table = rt.alloc_buffer(bits.len() * 2).unwrap();
        table.write_bf16_bits(&bits);
        let head = LmHead {
            weight: &table,
            dtype: DType::BF16,
            vocab: vocab as u32,
        };
        let bad = [vocab as u32, u32::MAX];
        let ids: Vec<u32> = vec![0, 999, 17, 17, 523, bad[0], 1, bad[1], 998];
        let n = ids.len();
        let idb = buf_u32(rt, &ids);
        let out = seeded(rt, n * hidden, SENTINEL);
        qwen35::embed_rows(rt, &idb, n as u32, head, hidden as u32, &out).unwrap();
        rt.synchronize().unwrap();
        let got = out.read_f32();
        for (r, &id) in ids.iter().enumerate() {
            let row = &got[r * hidden..(r + 1) * hidden];
            if bad.contains(&id) {
                assert!(row.iter().all(|x| x.is_nan()), "row {r}: id {id} is not NaN");
                continue;
            }
            let want = &bits[id as usize * hidden..(id as usize + 1) * hidden];
            for (c, (g, &w)) in row.iter().zip(want).enumerate() {
                assert_eq!(g.to_bits(), bf16_bits_to_f32(w).to_bits(), "row {r} (id {id}) col {c}");
            }
        }
        // Nothing to gather is a clean no-op that writes nothing.
        let untouched = seeded(rt, hidden, SENTINEL);
        qwen35::embed_rows(rt, &idb, 0, head, hidden as u32, &untouched).unwrap();
        rt.synchronize().unwrap();
        assert!(untouched.read_f32().iter().all(|&x| x == SENTINEL));
        // An f32 table (the f32 model's) is gathered as bits: every value,
        // including a NaN payload, -0 and a subnormal, comes back unchanged.
        let mut wide = random_f32(vocab * hidden, 7901);
        wide[17 * hidden] = f32::from_bits(0x7fc0_1234);
        wide[17 * hidden + 1] = -0.0;
        wide[523 * hidden + 2] = f32::from_bits(1);
        let wide_table = rt.alloc_buffer(wide.len() * 4).unwrap();
        wide_table.write_f32(&wide);
        let f32_head = LmHead {
            weight: &wide_table,
            dtype: DType::F32,
            vocab: vocab as u32,
        };
        let out32 = seeded(rt, n * hidden, SENTINEL);
        qwen35::embed_rows(rt, &idb, n as u32, f32_head, hidden as u32, &out32).unwrap();
        rt.synchronize().unwrap();
        let got32 = out32.read_f32();
        for (r, &id) in ids.iter().enumerate() {
            let row = &got32[r * hidden..(r + 1) * hidden];
            if bad.contains(&id) {
                assert!(row.iter().all(|x| x.is_nan()), "f32 row {r}: id {id} is not NaN");
                continue;
            }
            let want = &wide[id as usize * hidden..(id as usize + 1) * hidden];
            for (c, (g, w)) in row.iter().zip(want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "f32 row {r} (id {id}) col {c}");
            }
        }
        // Rejections, each by its own message.
        let f16_head = LmHead {
            dtype: DType::F16,
            ..head
        };
        expect_err(
            qwen35::embed_rows(rt, &idb, n as u32, f16_head, hidden as u32, &out),
            "bf16 and f32 tables are compiled",
        );
        // The bf16 table's bytes are half an f32 table's.
        let f32_over_bf16 = LmHead {
            dtype: DType::F32,
            ..head
        };
        expect_err(
            qwen35::embed_rows(rt, &idb, n as u32, f32_over_bf16, hidden as u32, &out),
            "embed_rows table",
        );
        let short = LmHead {
            vocab: vocab as u32 + 1,
            ..head
        };
        expect_err(
            qwen35::embed_rows(rt, &idb, n as u32, short, hidden as u32, &out),
            "embed_rows table",
        );
        let big_out = rt.alloc_buffer(bits.len() * 4).unwrap();
        let aliased = LmHead {
            weight: &big_out,
            ..head
        };
        expect_err(
            qwen35::embed_rows(rt, &idb, n as u32, aliased, hidden as u32, &big_out),
            "overlaps read-only buffer table",
        );
    });
}

// ------------------------------------------------ matrix-unit prefill attention ---

/// Causal attention in f64 over `[batch, tq, heads, 256]` queries and
/// `[batch, cap, kv_heads, 256]` keys/values, `(heads, kv_heads)` given,
/// live `tkv`, query `t` at `q_off + t` and key `t` at `kv_off + t`. A row
/// that sees no key is zeros.
#[allow(clippy::too_many_arguments)]
fn causal_attn_f64(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    (h, hkv): (usize, usize),
    batch: usize,
    tq: usize,
    tkv: usize,
    cap: usize,
    q_off: usize,
    kv_off: usize,
) -> Vec<f64> {
    let d = PFX_D;
    let scale = 1.0 / (d as f64).sqrt();
    let mut out = vec![0.0f64; batch * tq * h * d];
    for b in 0..batch {
        for t in 0..tq {
            for hi in 0..h {
                let g = hi / (h / hkv);
                let qr = &q[((b * tq + t) * h + hi) * d..][..d];
                let keys: Vec<usize> = (0..tkv).filter(|&s| kv_off + s <= q_off + t).collect();
                if keys.is_empty() {
                    continue;
                }
                let scores: Vec<f64> = keys
                    .iter()
                    .map(|&s| {
                        let kr = &k[((b * cap + s) * hkv + g) * d..][..d];
                        scale * qr.iter().zip(kr).map(|(&a, &c)| a as f64 * c as f64).sum::<f64>()
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let l: f64 = w.iter().sum();
                let o = &mut out[((b * tq + t) * h + hi) * d..][..d];
                for (&s, &wi) in keys.iter().zip(&w) {
                    let vr = &v[((b * cap + s) * hkv + g) * d..][..d];
                    for (oi, &vi) in o.iter_mut().zip(vr) {
                        *oi += wi / l * vi as f64;
                    }
                }
            }
        }
    }
    out
}

fn max_err(got: &[f32], want: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (g as f64 - w).abs())
        .fold(0.0, f64::max)
}

/// One `attn_prefill_with_tile` call into a fresh sentinel-filled output.
#[allow(clippy::too_many_arguments)]
fn run_tiled(
    rt: &Arc<GpuRuntime>,
    (q, k, v): (&GpuBuffer, &GpuBuffer, &GpuBuffer),
    (tkv, qpos, kvpos): (&GpuBuffer, &GpuBuffer, &GpuBuffer),
    dims: tessl::nn::AttnDims,
    n: usize,
    bf16: bool,
    tile: qwen35::AttnTile,
) -> GpuBuffer {
    let o = seeded(rt, n, SENTINEL);
    qwen35::attn_prefill_with_tile(rt, q, k, v, &o, tkv, qpos, kvpos, dims, bf16, tile).unwrap();
    o
}

/// Qwen3.5-4B's full-attention heads (16 query over 4 KV heads of 256, the
/// shape rsi-jev-v6.0-vl-4b serves): the matrix-unit prefill at the default
/// tile against the f64 reference, held to the scalar kernel's own error as
/// the 2B-shaped test below holds it, across block edges and a ragged length.
#[test]
fn attn_prefill_at_the_4b_head_counts() {
    let (hq, hkv) = (16usize, 4usize);
    with_gpu(|rt| {
        for (ci, &(batch, tq)) in [(1usize, 1usize), (1, 65), (2, 300)].iter().enumerate() {
            let seed = 9700 + 10 * ci as u64;
            let (row, n) = (hkv * PFX_D, batch * tq * hq * PFX_D);
            let q: Vec<f32> = random_f32(n, seed).iter().map(|x| 4.0 * x).collect();
            let k = random_f32(batch * tq * row, seed + 1);
            let v = random_f32(batch * tq * row, seed + 2);
            let want = causal_attn_f64(&q, &k, &v, (hq, hkv), batch, tq, tq, tq, 0, 0);
            let (qb, kb, vb) = (buf(rt, &q), buf(rt, &k), buf(rt, &v));
            let tkvb = buf_u32(rt, &[tq as u32]);
            let zero = buf_u32(rt, &[0]);
            let dims = tessl::nn::AttnDims {
                batch: batch as u32,
                tq: tq as u32,
                heads: hq as u32,
                heads_kv: hkv as u32,
                window: 0,
                scale: 1.0 / (PFX_D as f32).sqrt(),
            };
            let scalar = seeded(rt, n, SENTINEL);
            tessl::nn::flash_attn_rows(
                rt,
                &qb,
                &kb,
                &vb,
                &scalar,
                &tkvb,
                &zero,
                &zero,
                dims,
                PFX_D as u32,
                false,
            )
            .unwrap();
            let tiled = seeded(rt, n, SENTINEL);
            qwen35::attn_prefill(rt, &qb, &kb, &vb, &tiled, &tkvb, &zero, &zero, dims, false).unwrap();
            rt.synchronize().unwrap();
            let (s, t) = (scalar.read_f32(), tiled.read_f32());
            let (es, et) = (max_err(&s[..n], &want), max_err(&t[..n], &want));
            eprintln!("4B heads B{batch} Tq{tq}: max |err| scalar {es:.2e}, tiled {et:.2e}");
            assert!(
                t[..n].iter().all(|x| x.is_finite()),
                "B{batch} Tq{tq}: non-finite (unwritten?)"
            );
            assert!(es <= 1e-5, "B{batch} Tq{tq}: scalar kernel error {es:.2e}");
            assert!(
                et <= 4.0 * es.max(1e-7),
                "B{batch} Tq{tq}: tiled {et:.2e} > 4x scalar {es:.2e}"
            );
        }
    });
}

#[test]
fn attn_prefill_matches_an_f64_reference_as_closely_as_flash_attn_rows() {
    // (batch, tq, live tkv, capacity, q_pos_offset, kv_pos_offset), run at
    // every tile (32 or 64 queries by 32 or 64 keys): one query; either side
    // of each block edge; a few hundred (several blocks on and below the
    // diagonal); a continuation whose queries start mid-cache; a key offset;
    // and leading queries that precede every key, which must come out zeros.
    let cases: [(usize, usize, usize, usize, usize, usize); 11] = [
        (1, 1, 1, 1, 0, 0),
        (1, 31, 31, 40, 0, 0),
        (1, 32, 32, 32, 0, 0),
        (2, 33, 33, 35, 0, 0),
        (1, 63, 63, 70, 0, 0),
        (1, 64, 64, 64, 0, 0),
        (2, 65, 65, 66, 0, 0),
        (2, 300, 300, 301, 0, 0),
        (2, 37, 100, 128, 63, 0),
        (1, 50, 50, 64, 40, 30),
        (1, 40, 40, 48, 0, 5),
    ];
    with_gpu(|rt| {
        for (ci, &(batch, tq, tkv, cap, q_off, kv_off)) in cases.iter().enumerate() {
            let seed = 9100 + 10 * ci as u64;
            let row = PFX_HKV * PFX_D;
            let n = batch * tq * PFX_HQ * PFX_D;
            // Sharper than a near-uniform softmax: scores of a few units.
            let q: Vec<f32> = random_f32(n, seed).iter().map(|x| 4.0 * x).collect();
            let mut k = random_f32(batch * cap * row, seed + 1);
            let mut v = random_f32(batch * cap * row, seed + 2);
            // Every slot past the live length is NaN, so a kernel that read
            // one would print NaN into the output.
            for b in 0..batch {
                let dead = (b * cap + tkv) * row..(b + 1) * cap * row;
                k[dead.clone()].fill(f32::NAN);
                v[dead].fill(f32::NAN);
            }
            let want = causal_attn_f64(&q, &k, &v, (PFX_HQ, PFX_HKV), batch, tq, tkv, cap, q_off, kv_off);
            let (qb, kb, vb) = (buf(rt, &q), buf(rt, &k), buf(rt, &v));
            let tkvb = buf_u32(rt, &[tkv as u32]);
            let qpos = buf_u32(rt, &[q_off as u32]);
            let kvpos = buf_u32(rt, &[kv_off as u32]);
            let dims = pfx_dims(batch, tq);
            let scalar = seeded(rt, n, SENTINEL);
            tessl::nn::flash_attn_rows(
                rt,
                &qb,
                &kb,
                &vb,
                &scalar,
                &tkvb,
                &qpos,
                &kvpos,
                dims,
                PFX_D as u32,
                false,
            )
            .unwrap();
            rt.synchronize().unwrap();
            let s = scalar.read_f32();
            let es = max_err(&s[..n], &want);
            for tile in qwen35::AttnTile::ALL {
                let label = format!("{} B{batch} Tq{tq} Tkv{tkv}/{cap} q@{q_off} k@{kv_off}", tile.label());
                let bufs = (&qb, &kb, &vb);
                let scalars = (&tkvb, &qpos, &kvpos);
                let tiled = run_tiled(rt, bufs, scalars, dims, n, false, tile);
                let tiled_bf16 = run_tiled(rt, bufs, scalars, dims, n, true, tile);
                rt.synchronize().unwrap();
                let t = tiled.read_f32();
                let t = &t[..n];
                assert!(
                    t.iter().all(|x| x.is_finite()),
                    "{label}: non-finite output (unwritten, or read a poisoned K/V slot)"
                );
                let et = max_err(t, &want);
                eprintln!("{label}: max |err| scalar {es:.2e}, tiled {et:.2e}");
                assert!(
                    et <= 4.0 * es.max(1e-7),
                    "{label}: tiled error {et:.2e} is more than 4x the scalar kernel's {es:.2e}"
                );
                // Rows that precede every key are exactly zero, as in flash_attn_rows.
                for (i, (&ti, &wi)) in t.iter().zip(&want).enumerate() {
                    if wi == 0.0 {
                        assert_eq!(ti, 0.0, "{label}[{i}]: a fully masked row must be zeros");
                    }
                }
                // The bf16 store is the f32 result rounded, element for element.
                let rounded = f32_slice_to_bf16(t);
                let got16 = &tiled_bf16.contents_u16()[..n];
                for (i, (&g, &w)) in got16.iter().zip(&rounded).enumerate() {
                    assert_eq!(g, w, "{label}[{i}]: bf16 output {g:#x} vs rounded f32 {w:#x}");
                }
            }
        }

        // Long enough for dozens of blocks per row, off every block edge, at
        // B = 2: against the scalar kernel, since the f64 reference is too slow
        // here. Each lands ~1e-6 from the truth at T = 300.
        let (batch, t) = (2usize, 1500usize);
        let n = batch * t * PFX_HQ * PFX_D;
        let q: Vec<f32> = random_f32(n, 9190).iter().map(|x| 4.0 * x).collect();
        let k = buf(rt, &random_f32(batch * t * PFX_HKV * PFX_D, 9191));
        let v = buf(rt, &random_f32(batch * t * PFX_HKV * PFX_D, 9192));
        let (qb, tkv, zero) = (buf(rt, &q), buf_u32(rt, &[t as u32]), buf_u32(rt, &[0]));
        let dims = pfx_dims(batch, t);
        let scalar = seeded(rt, n, SENTINEL);
        tessl::nn::flash_attn_rows(rt, &qb, &k, &v, &scalar, &tkv, &zero, &zero, dims, PFX_D as u32, false).unwrap();
        rt.synchronize().unwrap();
        let want: Vec<f64> = scalar.read_f32()[..n].iter().map(|&x| x as f64).collect();
        for tile in qwen35::AttnTile::ALL {
            let tiled = run_tiled(rt, (&qb, &k, &v), (&tkv, &zero, &zero), dims, n, false, tile);
            rt.synchronize().unwrap();
            let diff = max_err(&tiled.read_f32()[..n], &want);
            eprintln!("{} B{batch} T{t}: max |tiled - scalar| {diff:.2e}", tile.label());
            assert!(
                diff <= 1e-5,
                "{} B{batch} T{t}: tiled vs scalar {diff:.2e}",
                tile.label()
            );
        }
    });
}

#[test]
fn attn_prefix_rows_continuing_attn_prefill_matches_attn_prefill_on_the_whole_row() {
    // Lappi's shape of use: a context prefilled with `attn_prefill`, then
    // several questions attending to it through `attn_prefix_rows` without a
    // per-row copy. The two kernels round differently (matrix units vs scalar
    // f32), so each question's attention must agree with `attn_prefill` over
    // that row's own `prefix ‖ suffix` to rounding, not bit for bit.
    with_gpu(|rt| {
        let (batch, p, s) = (3usize, 200usize, 37usize);
        let t = p + s;
        let row = PFX_HKV * PFX_D;
        let qrow = PFX_HQ * PFX_D;
        let pk = random_f32(p * row, 9300);
        let pv = random_f32(p * row, 9301);
        let sk = random_f32(batch * s * row, 9302);
        let sv = random_f32(batch * s * row, 9303);
        let q: Vec<f32> = random_f32(batch * s * qrow, 9304).iter().map(|x| 4.0 * x).collect();
        let (pkb, pvb, skb, svb, qb) = (buf(rt, &pk), buf(rt, &pv), buf(rt, &sk), buf(rt, &sv), buf(rt, &q));
        let (slen, qpos) = (buf_u32(rt, &[s as u32]), buf_u32(rt, &[p as u32]));
        let got = seeded(rt, batch * s * qrow, SENTINEL);
        let prefix = qwen35::SharedPrefix {
            k: &pkb,
            v: &pvb,
            len: p as u32,
        };
        qwen35::attn_prefix_rows(
            rt,
            &qb,
            prefix,
            &skb,
            &svb,
            &slen,
            &qpos,
            &got,
            pfx_dims(batch, s),
            false,
        )
        .unwrap();
        rt.synchronize().unwrap();
        let got = got.read_f32();
        for b in 0..batch {
            // Row b alone, as one sequence: its whole cache, and queries for
            // every position with the question's at the end.
            let mut fk = pk.clone();
            fk.extend_from_slice(&sk[b * s * row..(b + 1) * s * row]);
            let mut fv = pv.clone();
            fv.extend_from_slice(&sv[b * s * row..(b + 1) * s * row]);
            let mut fq = random_f32(p * qrow, 9310 + b as u64);
            fq.extend_from_slice(&q[b * s * qrow..(b + 1) * s * qrow]);
            let full = seeded(rt, t * qrow, SENTINEL);
            let (tkv, zero) = (buf_u32(rt, &[t as u32]), buf_u32(rt, &[0]));
            let (fkb, fvb, fqb) = (buf(rt, &fk), buf(rt, &fv), buf(rt, &fq));
            qwen35::attn_prefill(rt, &fqb, &fkb, &fvb, &full, &tkv, &zero, &zero, pfx_dims(1, t), false).unwrap();
            rt.synchronize().unwrap();
            let want: Vec<f64> = full.read_f32()[p * qrow..t * qrow].iter().map(|&x| x as f64).collect();
            let mine = &got[b * s * qrow..(b + 1) * s * qrow];
            assert!(mine.iter().all(|x| x.is_finite()), "row {b}: non-finite");
            let diff = max_err(mine, &want);
            eprintln!("row {b}: max |prefix_rows - attn_prefill| {diff:.2e}");
            assert!(
                diff <= 1e-5,
                "row {b}: shared-prefix questions vs attn_prefill {diff:.2e}"
            );
        }
    });
}

#[test]
fn attn_prefill_rejects_a_window_and_bad_storage() {
    with_gpu(|rt| {
        let (batch, tq, cap) = (1usize, 4usize, 4usize);
        let n = batch * tq * PFX_HQ * PFX_D;
        let q = buf(rt, &random_f32(n, 9200));
        let k = buf(rt, &random_f32(batch * cap * PFX_HKV * PFX_D, 9201));
        let v = buf(rt, &random_f32(batch * cap * PFX_HKV * PFX_D, 9202));
        let o = seeded(rt, n, SENTINEL);
        let (tkv, zero) = (buf_u32(rt, &[tq as u32]), buf_u32(rt, &[0]));
        let mut dims = pfx_dims(batch, tq);
        dims.window = 16;
        expect_err(
            qwen35::attn_prefill(rt, &q, &k, &v, &o, &tkv, &zero, &zero, dims, false),
            "window must be 0",
        );
        let short = seeded(rt, n - 1, SENTINEL);
        expect_err(
            qwen35::attn_prefill(rt, &q, &k, &v, &short, &tkv, &zero, &zero, pfx_dims(batch, tq), false),
            "qwen35::attn_prefill o: buffer holds",
        );
        // K/V big enough to hold an output, so the alias check is what fires.
        let big_k = buf(rt, &random_f32(n, 9203));
        let big_v = buf(rt, &random_f32(n, 9204));
        expect_err(
            qwen35::attn_prefill(
                rt,
                &q,
                &big_k,
                &big_v,
                &big_k,
                &tkv,
                &zero,
                &zero,
                pfx_dims(batch, tq),
                false,
            ),
            "qwen35::attn_prefill: output must not alias read-only k input",
        );
    });
}
