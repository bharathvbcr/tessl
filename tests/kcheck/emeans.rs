//! The 23 `emeans_*` kernels, each dispatched alone through the raw dispatch
//! API against a CPU mirror built from the Metal source.
//!
//! Nothing here goes through `EmeansGpuRunner` or `assign_gpu_*`: those paths
//! request every pipeline up front, so the kernel trace would credit kernels
//! that never ran. Every input is built on the CPU, no kernel's output feeds
//! another kernel's credit, and every pure output buffer starts as a sentinel
//! so a missed or stray write shows. Comparisons are bit-exact except where the
//! comment beside a bound names the transcendental that forces it.
//!
//! Mismatches are collected in a [`Report`] and fail the sweep once, after all
//! 23 credits ran and before the ledger prints a single PASS line.

use std::sync::Arc;

use crate::common::{buf, buf_u32, random_f32};
use tessl::dispatch::{dispatch_1d, Binder};
use tessl::emeans::revival_gumbel_key;
use tessl::runtime::mtl_size;
use tessl::tensor::{bf16_bits_to_f32, f32_to_bf16_bits_gpu, GpuBuffer};
use tessl::GpuRuntime;

use super::Ledger;

const SENT_U16: u16 = 0xDEAD;
const SENT_U32: u32 = 0xA5A5_A5A5;
const SENT_F32_BITS: u32 = 0x7fa5_a5a5;
const SENT_U64: u64 = 0xA5A5_5A5A_DEAD_BEEF;
const INVALID: u32 = 0xffff_ffff;
const MAX_SENTINEL: f32 = f32::MAX;
const SAFE_RN_MAX: f32 = 7.0e37;

const FAULT_INPUT: u32 = 1;
const FAULT_BF16: u32 = 2;
const FAULT_ROW_NORM: u32 = 4;
const FAULT_CENT_NORM: u32 = 8;
const FAULT_DIST: u32 = 16;

// ---------------------------------------------------------------------------
// Failure collection
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Report {
    failures: Vec<String>,
}

fn ord32(bits: u32) -> i64 {
    if bits >> 31 == 1 {
        -i64::from(bits & 0x7fff_ffff)
    } else {
        i64::from(bits)
    }
}

fn ord16(bits: u16) -> i64 {
    if bits >> 15 == 1 {
        -i64::from(bits & 0x7fff)
    } else {
        i64::from(bits)
    }
}

impl Report {
    fn fail(&mut self, msg: String) {
        self.failures.push(msg);
    }

    /// Bit-exact comparison; the message names the tag, the buffer and the first mismatches.
    fn cmp<T: Copy + PartialEq + Into<u64>>(&mut self, tag: &str, what: &str, got: &[T], want: &[T]) {
        if got.len() != want.len() {
            self.fail(format!("{tag}: {what}: length {} != {}", got.len(), want.len()));
            return;
        }
        let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
        if bad.is_empty() {
            return;
        }
        let shown: Vec<String> = bad
            .iter()
            .take(3)
            .map(|&i| {
                let (g, w): (u64, u64) = (got[i].into(), want[i].into());
                format!("[{i}] got {g:#x} want {w:#x}")
            })
            .collect();
        self.fail(format!(
            "{tag}: {what}: {} of {} differ: {}",
            bad.len(),
            got.len(),
            shown.join("; ")
        ));
    }

    /// Comparison within `tol` ulps over an ordered-integer view; inexact-but-within-bound
    /// entries are printed when `note` is set.
    fn cmp_ulp(&mut self, tag: &str, what: &str, got: &[u32], want: &[u32], (tol, wide): (i64, bool), note: bool) {
        if got.len() != want.len() {
            self.fail(format!("{tag}: {what}: length {} != {}", got.len(), want.len()));
            return;
        }
        let dist = |g: u32, w: u32| -> i64 {
            if wide {
                (ord32(g) - ord32(w)).abs()
            } else {
                (ord16(g as u16) - ord16(w as u16)).abs()
            }
        };
        let beyond: Vec<usize> = (0..got.len()).filter(|&i| dist(got[i], want[i]) > tol).collect();
        if !beyond.is_empty() {
            let shown: Vec<String> = beyond
                .iter()
                .take(3)
                .map(|&i| {
                    format!(
                        "[{i}] got {:#x} want {:#x} ({} ulp)",
                        got[i],
                        want[i],
                        dist(got[i], want[i])
                    )
                })
                .collect();
            self.fail(format!(
                "{tag}: {what}: {} of {} beyond {tol} ulp: {}",
                beyond.len(),
                got.len(),
                shown.join("; ")
            ));
        } else if note {
            for i in (0..got.len()).filter(|&i| got[i] != want[i]).take(3) {
                println!(
                    "NOTE {tag}: {what}[{i}] got {:#x} want {:#x} ({} ulp, within {tol})",
                    got[i],
                    want[i],
                    dist(got[i], want[i])
                );
            }
        }
    }

    fn assert_clean(&self) {
        const SHOWN: usize = 80;
        assert!(
            self.failures.is_empty(),
            "emeans kcheck: {} mismatches (first {}):\n{}",
            self.failures.len(),
            SHOWN.min(self.failures.len()),
            self.failures.iter().take(SHOWN).cloned().collect::<Vec<_>>().join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// Buffers and launches
// ---------------------------------------------------------------------------

fn f32b(rt: &Arc<GpuRuntime>, data: &[f32]) -> GpuBuffer {
    buf(rt, data)
}

fn u32b(rt: &Arc<GpuRuntime>, data: &[u32]) -> GpuBuffer {
    buf_u32(rt, data)
}

fn u16b(rt: &Arc<GpuRuntime>, data: &[u16]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1) * 2).expect("alloc u16");
    if data.is_empty() {
        b.write_bf16_bits(&[0]);
    } else {
        b.write_bf16_bits(data);
    }
    b
}

fn u64b(rt: &Arc<GpuRuntime>, data: &[u64]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1) * 8).expect("alloc u64");
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
    b.write_bytes(&bytes);
    b
}

fn sent_f32(rt: &Arc<GpuRuntime>, n: usize) -> GpuBuffer {
    u32b(rt, &vec![SENT_F32_BITS; n.max(1)])
}

fn sent_u32(rt: &Arc<GpuRuntime>, n: usize) -> GpuBuffer {
    u32b(rt, &vec![SENT_U32; n.max(1)])
}

fn sent_u16(rt: &Arc<GpuRuntime>, n: usize) -> GpuBuffer {
    u16b(rt, &vec![SENT_U16; n.max(1)])
}

fn sent_u64(rt: &Arc<GpuRuntime>, n: usize) -> GpuBuffer {
    u64b(rt, &vec![SENT_U64; n.max(1)])
}

fn fault_buf(rt: &Arc<GpuRuntime>, preset: u32) -> GpuBuffer {
    u32b(rt, &[preset])
}

fn rd32(b: &GpuBuffer, n: usize) -> Vec<u32> {
    b.contents_u32()[..n].to_vec()
}

fn rd16(b: &GpuBuffer, n: usize) -> Vec<u16> {
    b.contents_u16()[..n].to_vec()
}

fn rd64(b: &GpuBuffer, n: usize) -> Vec<u64> {
    let bytes = b.contents_u8()[..n * 8].to_vec();
    (0..n)
        .map(|i| u64::from_ne_bytes(bytes[i * 8..i * 8 + 8].try_into().expect("8 bytes")))
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn launch(
    rt: &Arc<GpuRuntime>,
    name: &str,
    grid: (usize, usize, usize),
    tg: (usize, usize, usize),
    bind: impl FnOnce(&mut Binder<'_>),
) {
    let p = rt.pipeline(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&p);
        bind(bnd);
        bnd.dispatch(mtl_size(grid.0, grid.1, grid.2), mtl_size(tg.0, tg.1, tg.2));
        Ok(())
    })
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    rt.synchronize().unwrap_or_else(|e| panic!("{name}: {e}"));
}

fn launch_1d(rt: &Arc<GpuRuntime>, name: &str, n: usize, bind: impl FnOnce(&mut Binder<'_>)) {
    let p = rt.pipeline(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    dispatch_1d(rt, &p, n, bind).unwrap_or_else(|e| panic!("{name}: {e}"));
    rt.synchronize().unwrap_or_else(|e| panic!("{name}: {e}"));
}

// ---------------------------------------------------------------------------
// CPU mirrors shared by several kernels
// ---------------------------------------------------------------------------

fn nonfinite32(bits: u32) -> bool {
    (bits & 0x7fff_ffff) >= 0x7f80_0000
}

fn nonfinite16(bits: u16) -> bool {
    (bits & 0x7fff) >= 0x7f80
}

/// `tg_reduce_sum_256`: strides 128, 64, ..., 1 over 256 lane partials.
fn tree256(mut p: [f32; 256]) -> f32 {
    let mut stride = 128;
    while stride > 0 {
        for t in 0..stride {
            p[t] += p[t + stride];
        }
        stride >>= 1;
    }
    p[0]
}

/// Lane `t` sums columns `t, t + 256, ...` in order from 0.0, then the tree.
fn lane_sum(d: usize, f: impl Fn(usize) -> f32) -> f32 {
    let mut p = [0.0f32; 256];
    for col in 0..d {
        p[col % 256] += f(col);
    }
    tree256(p)
}

fn bf16f(b: u16) -> f32 {
    bf16_bits_to_f32(b)
}

/// Round a value to the operand the bf16 kernels read, or keep it for the f32 ones.
fn eff(v: f32, bf16: bool) -> f32 {
    if bf16 {
        bf16f(f32_to_bf16_bits_gpu(v))
    } else {
        v
    }
}

fn bf16_bits_of(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| f32_to_bf16_bits_gpu(x)).collect()
}

/// A deterministic 64-bit stream for key fixtures.
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

// ---------------------------------------------------------------------------
// prepare_centroids
// ---------------------------------------------------------------------------

struct PrepCase {
    name: &'static str,
    k: usize,
    d: usize,
    sph: bool,
    data: Vec<f32>,
    /// Fault word the case must raise (checked against the oracle and the kernel).
    fault: u32,
}

struct PrepWant {
    c_eff: Vec<u32>,
    norm2: Vec<f32>,
    fault: u32,
}

/// Norm of one effective row from its stored bits (`bf16` bits or `f32` bits).
fn prep_norm(row: &[u32], d: usize, bf16: bool) -> f32 {
    let cn = lane_sum(d, |c| {
        let q = if bf16 {
            bf16f(row[c] as u16)
        } else {
            f32::from_bits(row[c])
        };
        q * q
    });
    if nonfinite32(cn.to_bits()) {
        0.0
    } else {
        cn
    }
}

fn prep_oracle(case: &PrepCase, bf16: bool) -> PrepWant {
    let (k, d) = (case.k, case.d);
    let mut c_eff = Vec::with_capacity(k * d);
    let mut norm2 = Vec::with_capacity(k);
    let mut fault = 0u32;
    for j in 0..k {
        let row = &case.data[j * d..(j + 1) * d];
        if row.iter().any(|v| nonfinite32(v.to_bits())) {
            fault |= FAULT_INPUT;
        }
        let max_abs = row
            .iter()
            .filter(|v| !nonfinite32(v.to_bits()))
            .fold(0.0f32, |a, v| a.max(v.abs()));
        let mut inv_norm = 1.0f32;
        let mut ma = 1.0f32;
        if case.sph {
            ma = max_abs;
            let total = if ma > 0.0 {
                lane_sum(d, |c| {
                    let clean = if nonfinite32(row[c].to_bits()) {
                        0.0
                    } else {
                        row[c] / ma
                    };
                    clean * clean
                })
            } else {
                0.0
            };
            // precise::rsqrt is not pinned to an ulp bound, so spherical rows carry a tolerance.
            inv_norm = if ma > 0.0 && total > 0.0 {
                (1.0 / f64::from(total).sqrt()) as f32
            } else {
                0.0
            };
        }
        let mut stored = Vec::with_capacity(d);
        for &raw in row {
            let mut v = if nonfinite32(raw.to_bits()) { 0.0 } else { raw };
            if case.sph {
                v = if ma > 0.0 { (v / ma) * inv_norm } else { 0.0 };
            }
            if bf16 {
                let mut bf = f32_to_bf16_bits_gpu(v);
                if nonfinite16(bf) {
                    fault |= FAULT_BF16;
                    bf = 0;
                }
                stored.push(u32::from(bf));
            } else {
                stored.push(v.to_bits());
            }
        }
        let cn = lane_sum(d, |c| {
            let q = if bf16 {
                bf16f(stored[c] as u16)
            } else {
                f32::from_bits(stored[c])
            };
            q * q
        });
        if nonfinite32(cn.to_bits()) {
            fault |= FAULT_CENT_NORM;
            norm2.push(0.0);
        } else {
            norm2.push(cn);
        }
        c_eff.extend(stored);
    }
    PrepWant { c_eff, norm2, fault }
}

fn prep_cases() -> Vec<PrepCase> {
    let mut cases = Vec::new();
    for (k, d) in [(1usize, 67usize), (3, 67), (1, 300), (3, 300)] {
        cases.push(PrepCase {
            name: "random",
            k,
            d,
            sph: false,
            data: random_f32(k * d, 4100 + (k * 1000 + d) as u64),
            fault: 0,
        });
    }
    // Power-of-4 rows: the scaled row is 1.0 on 4 or 16 coordinates, so rsqrt is 0.5 or 0.25.
    let d = 67;
    let mut pow4 = vec![0.0f32; 4 * d];
    for c in 0..4 {
        pow4[c] = 2.0;
        pow4[d + c] = if c % 2 == 0 { 1.0 } else { -1.0 };
    }
    for c in 0..16 {
        pow4[2 * d + c] = 3.0;
    }
    cases.push(PrepCase {
        name: "pow4",
        k: 4,
        d,
        sph: true,
        data: pow4,
        fault: 0,
    });
    for d in [67usize, 300] {
        let mut data = random_f32(4 * d, 4200 + d as u64);
        for v in &mut data[d..2 * d] {
            *v *= 1000.0;
        }
        for v in &mut data[2 * d..3 * d] {
            *v *= 1.0e-3;
        }
        data[3 * d..].fill(0.0);
        cases.push(PrepCase {
            name: "sph-random",
            k: 4,
            d,
            sph: true,
            data,
            fault: 0,
        });
    }
    let d = 67;
    let mut nonfinite = random_f32(2 * d, 4300);
    nonfinite[0] = f32::INFINITY;
    nonfinite[7] = f32::NAN;
    nonfinite[d + 5] = f32::NEG_INFINITY;
    cases.push(PrepCase {
        name: "nonfinite",
        k: 2,
        d,
        sph: false,
        data: nonfinite,
        fault: FAULT_INPUT,
    });
    cases
}

fn prepare_centroids(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    for bf16 in [true, false] {
        let name: &'static str = if bf16 {
            "emeans_prepare_centroids_bf16"
        } else {
            "emeans_prepare_centroids_f32"
        };
        ledger.credit(rt, name, &[name], || {
            let mut cases = prep_cases();
            // f32::MAX squares to infinity, and rounds to bf16 infinity.
            let d = 67;
            let mut ovf = random_f32(2 * d, 4400);
            ovf[3] = f32::MAX;
            ovf[d + 9] = 3.0e38;
            cases.push(PrepCase {
                name: "overflow",
                k: 2,
                d,
                sph: false,
                data: ovf,
                fault: if bf16 {
                    FAULT_BF16 | FAULT_CENT_NORM
                } else {
                    FAULT_CENT_NORM
                },
            });
            for case in &cases {
                let tag = format!("{name}[{} k={} d={} sph={}]", case.name, case.k, case.d, case.sph);
                let want = prep_oracle(case, bf16);
                if want.fault != case.fault {
                    rep.fail(format!(
                        "{tag}: ORACLE derived fault {} but the case plans {}",
                        want.fault, case.fault
                    ));
                }
                let n = case.k * case.d;
                let cent = f32b(rt, &case.data);
                let c_eff = if bf16 { sent_u16(rt, n) } else { sent_f32(rt, n) };
                let norm2 = sent_f32(rt, case.k);
                let fault = fault_buf(rt, 0);
                launch(rt, name, (case.k, 1, 1), (256, 1, 1), |bnd| {
                    bnd.bind_gpu_buf(&cent, 0);
                    bnd.bind_gpu_buf(&c_eff, 1);
                    bnd.bind_gpu_buf(&norm2, 2);
                    bnd.bind_gpu_buf(&fault, 3);
                    bnd.bind_u32(case.k as u32, 4);
                    bnd.bind_u32(case.d as u32, 5);
                    bnd.bind_u32(u32::from(case.sph), 6);
                });
                let got: Vec<u32> = if bf16 {
                    rd16(&c_eff, n).into_iter().map(u32::from).collect()
                } else {
                    rd32(&c_eff, n)
                };
                let got_norm = rd32(&norm2, case.k);
                rep.cmp(&tag, "fault", &rd32(&fault, 1), &[want.fault]);
                if case.sph {
                    // precise::rsqrt feeds only the scaled coordinate, so only that output carries
                    // an ulp bound (1 bf16 ulp, 2 f32 ulp); every norm is recomputed exactly from
                    // the coordinates the kernel stored, and the planted rows' bookkeeping is exact.
                    let bound = if bf16 { (1, false) } else { (2, true) };
                    rep.cmp_ulp(&tag, "c_eff", &got, &want.c_eff, bound, case.name == "pow4");
                    let want_norm: Vec<u32> = (0..case.k)
                        .map(|j| prep_norm(&got[j * case.d..(j + 1) * case.d], case.d, bf16).to_bits())
                        .collect();
                    rep.cmp(&tag, "norm2", &got_norm, &want_norm);
                } else {
                    rep.cmp(&tag, "c_eff", &got, &want.c_eff);
                    rep.cmp(&tag, "norm2", &got_norm, &bits(&want.norm2));
                }
            }
        });
    }
}

// ---------------------------------------------------------------------------
// stage_rows
// ---------------------------------------------------------------------------

struct StageWant {
    bf16: Vec<u16>,
    f32s: Vec<u32>,
    norm2: Vec<f32>,
    fault: u32,
}

fn stage_bf16_data(d: usize, seed: u64) -> Vec<u16> {
    let n = 5;
    let mut src = bf16_bits_of(&random_f32(n * d, seed));
    let special = [0x0001u16, 0x8001, 0x007f, 0x0000, 0x8000, 0x0080, 0x8080, 0x00ff];
    src[..special.len()].copy_from_slice(&special);
    src
}

fn stage_bf16_oracle(src: &[u16], n: usize, d: usize, wb: bool, wf: bool) -> StageWant {
    let mut out_b = vec![SENT_U16; n * d];
    let mut out_f = vec![SENT_F32_BITS; n * d];
    let mut norm2 = Vec::new();
    let mut fault = 0;
    for r in 0..n {
        let mut vf = Vec::with_capacity(d);
        for c in 0..d {
            let raw = src[r * d + c];
            let safe = if nonfinite16(raw) {
                fault |= FAULT_INPUT;
                0
            } else if raw & 0x7f80 == 0 {
                raw & 0x8000
            } else {
                raw
            };
            let v = bf16f(safe);
            if wb {
                out_b[r * d + c] = safe;
            }
            if wf {
                out_f[r * d + c] = v.to_bits();
            }
            vf.push(v);
        }
        let rn = lane_sum(d, |c| vf[c] * vf[c]);
        if nonfinite32(rn.to_bits()) {
            fault |= FAULT_ROW_NORM;
            norm2.push(0.0);
        } else {
            norm2.push(rn);
        }
    }
    StageWant {
        bf16: out_b,
        f32s: out_f,
        norm2,
        fault,
    }
}

fn stage_f32_oracle(src: &[f32], n: usize, d: usize, (wb, wf, nb): (bool, bool, bool)) -> StageWant {
    let mut out_b = vec![SENT_U16; n * d];
    let mut out_f = vec![SENT_F32_BITS; n * d];
    let mut norm2 = Vec::new();
    let mut fault = 0;
    for r in 0..n {
        let mut nv = Vec::with_capacity(d);
        for c in 0..d {
            let raw = src[r * d + c];
            let mut vf = raw;
            if nonfinite32(raw.to_bits()) {
                fault |= FAULT_INPUT;
                vf = 0.0;
            }
            let mut safe = 0u16;
            if wb || nb {
                let bf = f32_to_bf16_bits_gpu(vf);
                if nonfinite16(bf) {
                    fault |= FAULT_BF16;
                } else {
                    safe = bf;
                }
                if wb {
                    out_b[r * d + c] = safe;
                }
            }
            if wf {
                out_f[r * d + c] = vf.to_bits();
            }
            nv.push(if nb { bf16f(safe) } else { vf });
        }
        let rn = lane_sum(d, |c| nv[c] * nv[c]);
        if nonfinite32(rn.to_bits()) {
            fault |= FAULT_ROW_NORM;
            norm2.push(0.0);
        } else {
            norm2.push(rn);
        }
    }
    StageWant {
        bf16: out_b,
        f32s: out_f,
        norm2,
        fault,
    }
}

fn stage_compare(
    rep: &mut Report,
    tag: &str,
    outs: (&GpuBuffer, &GpuBuffer, &GpuBuffer, &GpuBuffer),
    n: usize,
    d: usize,
    want: &StageWant,
    planned_fault: u32,
) {
    if want.fault != planned_fault {
        rep.fail(format!(
            "{tag}: ORACLE derived fault {} but the case plans {planned_fault}",
            want.fault
        ));
    }
    rep.cmp(tag, "dst_bf16", &rd16(outs.0, n * d), &want.bf16);
    rep.cmp(tag, "dst_f32", &rd32(outs.1, n * d), &want.f32s);
    rep.cmp(tag, "dst_norm2", &rd32(outs.2, n), &bits(&want.norm2));
    rep.cmp(tag, "fault", &rd32(outs.3, 1), &[planned_fault]);
}

/// (label, d, source bits, (write_bf16, write_f32), planned fault)
type StageBf16Case = (String, usize, Vec<u16>, (bool, bool), u32);

fn stage_rows_bf16(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    let name = "emeans_stage_rows_bf16";
    ledger.credit(rt, name, &[name], || {
        let n = 5usize;
        let mut cases: Vec<StageBf16Case> = Vec::new();
        for d in [67usize, 300] {
            for flags in [(true, false), (true, true), (false, true)] {
                cases.push(("base".into(), d, stage_bf16_data(d, 5100 + d as u64), flags, 0));
            }
        }
        let d = 67;
        let mut nonfinite = stage_bf16_data(d, 5200);
        nonfinite[d + 2] = 0x7f80;
        nonfinite[d + 3] = 0xff80;
        nonfinite[d + 4] = 0x7fc1;
        cases.push(("nonfinite".into(), d, nonfinite, (true, true), FAULT_INPUT));
        let mut huge = stage_bf16_data(d, 5300);
        huge[2 * d..2 * d + 8].fill(0x7f00);
        cases.push(("norm-overflow".into(), d, huge, (true, true), FAULT_ROW_NORM));
        for (case, d, src, (wb, wf), planned) in &cases {
            let tag = format!("{name}[{case} d={d} write_bf16={wb} write_f32={wf}]");
            let want = stage_bf16_oracle(src, n, *d, *wb, *wf);
            let sb = u16b(rt, src);
            let (ob, of, on, fl) = (
                sent_u16(rt, n * d),
                sent_f32(rt, n * d),
                sent_f32(rt, n),
                fault_buf(rt, 0),
            );
            launch(rt, name, (n, 1, 1), (256, 1, 1), |bnd| {
                bnd.bind_gpu_buf(&sb, 0);
                bnd.bind_gpu_buf(&ob, 1);
                bnd.bind_gpu_buf(&of, 2);
                bnd.bind_gpu_buf(&on, 3);
                bnd.bind_gpu_buf(&fl, 4);
                bnd.bind_u32(n as u32, 5);
                bnd.bind_u32(*d as u32, 6);
                bnd.bind_u32(u32::from(*wb), 7);
                bnd.bind_u32(u32::from(*wf), 8);
            });
            stage_compare(rep, &tag, (&ob, &of, &on, &fl), n, *d, &want, *planned);
        }
    });
}

fn stage_rows_f32(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    let name = "emeans_stage_rows_f32";
    ledger.credit(rt, name, &[name], || {
        let n = 5usize;
        let triples = [(true, false, true), (true, true, true), (false, true, false)];
        // (label, d, data, planned fault per triple)
        let mut cases: Vec<(&str, usize, Vec<f32>, [u32; 3])> = Vec::new();
        for d in [67usize, 300] {
            cases.push(("base", d, random_f32(n * d, 6100 + d as u64), [0, 0, 0]));
        }
        let d = 67;
        let mut nonfinite = random_f32(n * d, 6200);
        nonfinite[0] = f32::INFINITY;
        nonfinite[d + 1] = f32::NEG_INFINITY;
        nonfinite[2 * d + 2] = f32::NAN;
        cases.push(("nonfinite", d, nonfinite, [FAULT_INPUT; 3]));
        // f32::MAX rounds to bf16 infinity (fault 2, coordinate 0) under a bf16 triple; under the
        // f32 triple nothing converts and the squared norm overflows (fault 4).
        let mut max = random_f32(n * d, 6300);
        max[0] = f32::MAX;
        cases.push(("f32-max", d, max, [FAULT_BF16, FAULT_BF16, FAULT_ROW_NORM]));
        // 3.0e38 is a finite bf16 (0x7f62..), so no triple raises fault 2; its square overflows.
        let mut big = random_f32(n * d, 6400);
        big[d + 1] = 3.0e38;
        cases.push(("3e38", d, big, [FAULT_ROW_NORM; 3]));
        for (case, d, src, planned) in &cases {
            for (t, &flags) in triples.iter().enumerate() {
                let tag = format!(
                    "{name}[{case} d={d} write_bf16={} write_f32={} norm_from_bf16={}]",
                    flags.0, flags.1, flags.2
                );
                let want = stage_f32_oracle(src, n, *d, flags);
                let sb = f32b(rt, src);
                let (ob, of, on, fl) = (
                    sent_u16(rt, n * d),
                    sent_f32(rt, n * d),
                    sent_f32(rt, n),
                    fault_buf(rt, 0),
                );
                launch(rt, name, (n, 1, 1), (256, 1, 1), |bnd| {
                    bnd.bind_gpu_buf(&sb, 0);
                    bnd.bind_gpu_buf(&ob, 1);
                    bnd.bind_gpu_buf(&of, 2);
                    bnd.bind_gpu_buf(&on, 3);
                    bnd.bind_gpu_buf(&fl, 4);
                    bnd.bind_u32(n as u32, 5);
                    bnd.bind_u32(*d as u32, 6);
                    bnd.bind_u32(u32::from(flags.0), 7);
                    bnd.bind_u32(u32::from(flags.1), 8);
                    bnd.bind_u32(u32::from(flags.2), 9);
                });
                stage_compare(rep, &tag, (&ob, &of, &on, &fl), n, *d, &want, planned[t]);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// gather_rows
// ---------------------------------------------------------------------------

fn gather_rows(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    for bf16 in [true, false] {
        let name = if bf16 {
            "emeans_gather_rows_bf16"
        } else {
            "emeans_gather_rows_f32"
        };
        ledger.credit(rt, name, &[name], || {
            let (ds_n, row_offset, chunk_rows) = (40usize, 3usize, 21usize);
            let pad_rows = 11usize;
            let mut order: Vec<u32> = (0..30u32).map(|i| (i * 7 + 3) % 40).collect();
            order[row_offset] = 39;
            order[row_offset + 5] = 1000; // past ds_n: clamps to ds_n - 1
            order[row_offset + 9] = 40; // exactly ds_n: clamps too
            let norm = random_f32(ds_n, 7001);
            for d in [16usize, 67] {
                let data = random_f32(ds_n * d, 7100 + d as u64);
                for copy_norm2 in [1u32, 0] {
                    let tag = format!("{name}[d={d} copy_norm2={copy_norm2}]");
                    let rows_out = chunk_rows + pad_rows;
                    let (src, dst) = if bf16 {
                        (u16b(rt, &bf16_bits_of(&data)), sent_u16(rt, rows_out * d))
                    } else {
                        (f32b(rt, &data), sent_f32(rt, rows_out * d))
                    };
                    let (nb, ob, cn) = (f32b(rt, &norm), u32b(rt, &order), sent_f32(rt, rows_out));
                    launch(
                        rt,
                        name,
                        (d.div_ceil(16), chunk_rows.div_ceil(16), 1),
                        (16, 16, 1),
                        |bnd| {
                            bnd.bind_gpu_buf(&src, 0);
                            bnd.bind_gpu_buf(&nb, 1);
                            bnd.bind_gpu_buf(&ob, 2);
                            bnd.bind_gpu_buf(&dst, 3);
                            bnd.bind_gpu_buf(&cn, 4);
                            bnd.bind_u32(ds_n as u32, 5);
                            bnd.bind_u32(row_offset as u32, 6);
                            bnd.bind_u32(chunk_rows as u32, 7);
                            bnd.bind_u32(d as u32, 8);
                            bnd.bind_u32(copy_norm2, 9);
                        },
                    );
                    let mut want_norm = vec![SENT_F32_BITS; rows_out];
                    let src_row = |r: usize| (order[row_offset + r] as usize).min(ds_n - 1);
                    for r in 0..chunk_rows {
                        if copy_norm2 != 0 {
                            want_norm[r] = norm[src_row(r)].to_bits();
                        }
                    }
                    if bf16 {
                        let bits16 = bf16_bits_of(&data);
                        let mut want = vec![SENT_U16; rows_out * d];
                        for r in 0..chunk_rows {
                            let s = src_row(r);
                            want[r * d..(r + 1) * d].copy_from_slice(&bits16[s * d..(s + 1) * d]);
                        }
                        rep.cmp(&tag, "chunk", &rd16(&dst, rows_out * d), &want);
                    } else {
                        let mut want = vec![SENT_F32_BITS; rows_out * d];
                        for r in 0..chunk_rows {
                            let s = src_row(r);
                            for c in 0..d {
                                want[r * d + c] = data[s * d + c].to_bits();
                            }
                        }
                        rep.cmp(&tag, "chunk", &rd32(&dst, rows_out * d), &want);
                    }
                    rep.cmp(&tag, "chunk_norm2", &rd32(&cn, rows_out), &want_norm);
                }
            }
        });
    }
}

// ---------------------------------------------------------------------------
// assign (TensorOps and simdgroup, bf16 and f32)
// ---------------------------------------------------------------------------

const EPOCH_OFF: usize = 5;
const PAD: usize = 64;

/// Small-integer operands (|v| <= 9) make every product and partial sum exact in f32, so any
/// accumulation order the TensorOps or simdgroup matrix unit uses gives the same bits.
struct AssignFx {
    d: usize,
    x: Vec<f32>,
    c: Vec<f32>,
    rn: Vec<f32>,
    cn: Vec<f32>,
    self_row: Option<usize>,
}

fn seq_norm(v: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &a in v {
        s += a * a;
    }
    s
}

/// `exact_pair_dist2`: sequential f32 coordinate-wise squared distance, `None` when it overflows.
fn exact_dist2(x: &[f32], c: &[f32]) -> Option<f32> {
    let mut e = 0.0f32;
    for (a, b) in x.iter().zip(c) {
        let diff = a - b;
        e += diff * diff;
    }
    (!nonfinite32(e.to_bits())).then_some(e)
}

/// `finalize_row_dist_{bf16,f32}`: the distance and whether it raised FAULT_DIST_NONFINITE.
fn finalize_dist(x: &[f32], c: &[f32], rn: f32, best_s: f32, sph: bool) -> (f32, bool) {
    let exact = |x: &[f32], c: &[f32]| match exact_dist2(x, c) {
        Some(e) => (e, false),
        None => (0.0, true),
    };
    if sph {
        let v = 2.0f32 + best_s;
        return if nonfinite32(v.to_bits()) {
            (0.0, true)
        } else {
            (v, false)
        };
    }
    if rn > SAFE_RN_MAX {
        return exact(x, c);
    }
    let raw = rn + best_s;
    if nonfinite32(raw.to_bits()) {
        return exact(x, c);
    }
    let scale = rn.max(1.0e-30);
    if raw.abs() <= 1024.0f32 * 1.192_092_9e-7f32 * scale {
        return exact(x, c);
    }
    (raw.max(0.0), false)
}

fn assign_fx(m: usize, k: usize, d: usize, bf16: bool, special: bool) -> AssignFx {
    let mut x: Vec<f32> = (0..m * d)
        .map(|n| {
            let (i, t) = (n / d, n % d);
            (((i * 7 + t * 13 + i * t) % 17) as i32 - 8) as f32
        })
        .collect();
    let mut c: Vec<f32> = (0..k * d)
        .map(|n| {
            let (j, t) = (n / d, n % d);
            (((j * 5 + t * 11 + (j * t) % 7) % 17) as i32 - 8) as f32
        })
        .collect();
    if k > 67 {
        let row3 = c[3 * d..4 * d].to_vec();
        c[67 * d..68 * d].copy_from_slice(&row3);
    }
    // Rows on or next to a centroid, so labels spread over every cluster tile.
    for i in (0..m).step_by(3) {
        let j = (i * 13) % k;
        let row = c[j * d..(j + 1) * d].to_vec();
        x[i * d..(i + 1) * d].copy_from_slice(&row);
        if i % 2 == 0 {
            x[i * d + i % d] += 1.0;
        }
    }
    // Rows equal to the duplicated centroid: the lower index must win across tiles.
    if k > 67 && m > 66 {
        for i in [5usize, 66] {
            let row3 = c[3 * d..4 * d].to_vec();
            x[i * d..(i + 1) * d].copy_from_slice(&row3);
        }
    }
    let mut self_row = None;
    let mut invalid_row = None;
    if special && k > 50 && m > 50 {
        // A non-integer row equal to its centroid: the refine path must give +0.0.
        let s: Vec<f32> = random_f32(d, 8100 + d as u64).iter().map(|&v| eff(v, bf16)).collect();
        c[40 * d..41 * d].copy_from_slice(&s);
        x[33 * d..34 * d].copy_from_slice(&s);
        self_row = Some(33);
        // rn > 7.0e37 with a finite exact distance: row 20 and centroid 9 share a huge coordinate.
        let h = eff(9.0e18, bf16);
        x[20 * d] = h;
        c[9 * d] = h;
        // Every candidate overflows: all invalid.
        let g = eff(3.0e19, bf16);
        x[50 * d..51 * d].fill(g);
        invalid_row = Some(50);
    }
    let rn: Vec<f32> = (0..m)
        .map(|i| {
            if invalid_row == Some(i) {
                3.0e38
            } else {
                seq_norm(&x[i * d..(i + 1) * d])
            }
        })
        .collect();
    let cn: Vec<f32> = (0..k).map(|j| seq_norm(&c[j * d..(j + 1) * d])).collect();
    AssignFx {
        d,
        x,
        c,
        rn,
        cn,
        self_row,
    }
}

impl AssignFx {
    fn xr(&self, i: usize) -> &[f32] {
        &self.x[i * self.d..(i + 1) * self.d]
    }

    fn cr(&self, j: usize) -> &[f32] {
        &self.c[j * self.d..(j + 1) * self.d]
    }

    /// First strict minimum of the score over `c0..c1` (lower index wins ties), or `None`
    /// when every candidate is rejected.
    fn best(&self, i: usize, sph: bool, c0: usize, c1: usize) -> Option<(u32, f32)> {
        let mut best: Option<(u32, f64)> = None;
        for j in c0..c1 {
            let dot: f64 = self
                .xr(i)
                .iter()
                .zip(self.cr(j))
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let sc = if sph {
                -2.0 * dot
            } else if self.rn[i] > SAFE_RN_MAX {
                match exact_dist2(self.xr(i), self.cr(j)) {
                    Some(e) => f64::from(e),
                    None => continue,
                }
            } else {
                f64::from(self.cn[j]) - 2.0 * dot
            };
            if best.is_none_or(|(_, b)| sc < b) {
                best = Some((j as u32, sc));
            }
        }
        best.map(|(l, s)| (l, s as f32))
    }
}

struct AssignCase {
    name: &'static str,
    sph: bool,
    k: usize,
    split: usize,
    has_epoch: bool,
    special: bool,
}

fn assign_kernel(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report, bf16: bool, tensorops: bool) {
    let name = match (bf16, tensorops) {
        (true, true) => "emeans_assign_bf16_tensorops",
        (false, true) => "emeans_assign_f32_tensorops",
        (true, false) => "emeans_assign_bf16_simdgroup",
        (false, false) => "emeans_assign_f32_simdgroup",
    };
    let tile_m = if tensorops { 64 } else { 32 };
    let (m, d) = (70usize, 67usize);
    let cases = [
        AssignCase {
            name: "full",
            sph: false,
            k: 70,
            split: 1,
            has_epoch: true,
            special: true,
        },
        AssignCase {
            name: "no-epoch",
            sph: false,
            k: 70,
            split: 1,
            has_epoch: false,
            special: false,
        },
        AssignCase {
            name: "spherical",
            sph: true,
            k: 70,
            split: 1,
            has_epoch: true,
            special: false,
        },
        AssignCase {
            name: "k1",
            sph: false,
            k: 1,
            split: 1,
            has_epoch: false,
            special: false,
        },
        AssignCase {
            name: "split2",
            sph: false,
            k: 70,
            split: 2,
            has_epoch: false,
            special: true,
        },
        AssignCase {
            name: "split2-spherical",
            sph: true,
            k: 70,
            split: 2,
            has_epoch: false,
            special: false,
        },
    ];
    ledger.credit(rt, name, &[name], || {
        for case in &cases {
            let tag = format!(
                "{name}[{} M={m} k={} d={d} sph={} split_c={} epoch={}]",
                case.name, case.k, case.sph, case.split, case.has_epoch
            );
            let fx = assign_fx(m, case.k, d, bf16, case.special);
            let (xb, cb) = if bf16 {
                (u16b(rt, &bf16_bits_of(&fx.x)), u16b(rt, &bf16_bits_of(&fx.c)))
            } else {
                (f32b(rt, &fx.x), f32b(rt, &fx.c))
            };
            let (rnb, cnb) = (f32b(rt, &fx.rn), f32b(rt, &fx.cn));
            let chunk_labels = sent_u32(rt, m + PAD);
            let chunk_dists = sent_f32(rt, m + PAD);
            let epoch_labels = sent_u32(rt, EPOCH_OFF + m + PAD);
            let part_scores = sent_f32(rt, case.split * m + PAD);
            let part_labels = sent_u32(rt, case.split * m + PAD);
            let fault = fault_buf(rt, 0);
            let part_k = if case.split == 1 { case.k } else { 64 };
            for part_c in 0..case.split {
                let c_start = part_c * part_k;
                let c_end = (c_start + part_k).min(case.k);
                launch(rt, name, (m.div_ceil(tile_m), 1, 1), (128, 1, 1), |bnd| {
                    bnd.bind_gpu_buf(&xb, 0);
                    bnd.bind_gpu_buf(&cb, 1);
                    bnd.bind_gpu_buf(&rnb, 2);
                    bnd.bind_gpu_buf(&cnb, 3);
                    bnd.bind_gpu_buf(&chunk_labels, 4);
                    bnd.bind_gpu_buf(&chunk_dists, 5);
                    bnd.bind_gpu_buf(&epoch_labels, 6);
                    bnd.bind_gpu_buf(&part_scores, 7);
                    bnd.bind_gpu_buf(&part_labels, 8);
                    bnd.bind_u32(m as u32, 9);
                    bnd.bind_u32(d as u32, 10);
                    bnd.bind_u32(c_start as u32, 11);
                    bnd.bind_u32(c_end as u32, 12);
                    bnd.bind_u32(part_c as u32, 13);
                    bnd.bind_u32(case.split as u32, 14);
                    bnd.bind_u32(u32::from(case.sph), 15);
                    bnd.bind_u32(u32::from(case.has_epoch), 16);
                    bnd.bind_u32(EPOCH_OFF as u32, 17);
                    bnd.bind_gpu_buf(&fault, 18);
                });
            }
            let mut w_labels = vec![SENT_U32; m + PAD];
            let mut w_dists = vec![SENT_F32_BITS; m + PAD];
            let mut w_epoch = vec![SENT_U32; EPOCH_OFF + m + PAD];
            let mut w_ps = vec![SENT_F32_BITS; case.split * m + PAD];
            let mut w_pl = vec![SENT_U32; case.split * m + PAD];
            let mut w_fault = 0u32;
            for i in 0..m {
                if case.split == 1 {
                    match fx.best(i, case.sph, 0, case.k) {
                        Some((l, s)) => {
                            w_labels[i] = l;
                            let (dist, f) = finalize_dist(fx.xr(i), fx.cr(l as usize), fx.rn[i], s, case.sph);
                            w_dists[i] = dist.to_bits();
                            if f {
                                w_fault |= FAULT_DIST;
                            }
                            if case.has_epoch {
                                w_epoch[EPOCH_OFF + i] = l;
                            }
                        }
                        None => {
                            w_labels[i] = 0;
                            w_dists[i] = 0.0f32.to_bits();
                            w_fault |= FAULT_DIST;
                            if case.has_epoch {
                                w_epoch[EPOCH_OFF + i] = 0;
                            }
                        }
                    }
                } else {
                    for part_c in 0..case.split {
                        let c_start = part_c * part_k;
                        let c_end = (c_start + part_k).min(case.k);
                        let at = part_c * m + i;
                        match fx.best(i, case.sph, c_start, c_end) {
                            Some((l, s)) => {
                                w_ps[at] = s.to_bits();
                                w_pl[at] = l;
                            }
                            None => {
                                w_ps[at] = MAX_SENTINEL.to_bits();
                                w_pl[at] = INVALID;
                            }
                        }
                    }
                }
            }
            let got_ps = rd32(&part_scores, case.split * m + PAD);
            if let Some(i) = fx.self_row {
                // The self row's partition scores are matrix-unit noise around -|c|^2; its label
                // and (via the refine path) its distance are exact and stay compared.
                for part_c in 0..case.split {
                    let at = part_c * m + i;
                    w_ps[at] = got_ps[at];
                }
            }
            rep.cmp(&tag, "chunk_labels", &rd32(&chunk_labels, m + PAD), &w_labels);
            rep.cmp(&tag, "chunk_dists", &rd32(&chunk_dists, m + PAD), &w_dists);
            rep.cmp(
                &tag,
                "epoch_labels",
                &rd32(&epoch_labels, EPOCH_OFF + m + PAD),
                &w_epoch,
            );
            rep.cmp(&tag, "part_scores", &got_ps, &w_ps);
            rep.cmp(&tag, "part_labels", &rd32(&part_labels, case.split * m + PAD), &w_pl);
            rep.cmp(&tag, "fault", &rd32(&fault, 1), &[w_fault]);
        }
    });
}

// ---------------------------------------------------------------------------
// reduce_split_c
// ---------------------------------------------------------------------------

fn reduce_split_c(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    let name = "emeans_reduce_split_c";
    let (m, d, kc, split) = (40usize, 19usize, 12usize, 3usize);
    ledger.credit(rt, name, &[name], || {
        for bf16 in [false, true] {
            for (sph, has_epoch, bad) in [
                (false, true, false),
                (false, true, true),
                (false, false, false),
                (true, true, false),
            ] {
                let tag = format!("{name}[bf16_assign={bf16} sph={sph} epoch={has_epoch} bad_rows={bad}]");
                let mut x: Vec<f32> = (0..m * d)
                    .map(|n| (((n / d * 5 + n % d * 3 + (n / d) * (n % d)) % 9) as i32 - 4) as f32)
                    .collect();
                let c: Vec<f32> = (0..kc * d)
                    .map(|n| (((n / d * 7 + n % d * 2 + (n / d) * (n % d) % 5) % 9) as i32 - 4) as f32)
                    .collect();
                if bad {
                    x[7 * d..8 * d].fill(eff(3.0e19, bf16));
                }
                let mut rn: Vec<f32> = (0..m).map(|i| seq_norm(&x[i * d..(i + 1) * d])).collect();
                rn[5] = 9.0e37;
                rn[6] = 5.0e37;
                if bad {
                    rn[7] = 9.0e37;
                }
                let mut ps = vec![0.0f32; split * m];
                let mut pl = vec![0u32; split * m];
                for i in 0..m {
                    for p in 0..split {
                        let at = p * m + i;
                        pl[at] = ((i * 3 + p * 5) % kc) as u32;
                        ps[at] = ((i * 7 + p * 11) % 23) as f32 - 11.0;
                        if (i + p) % 7 == 0 {
                            pl[at] = INVALID;
                            ps[at] = -1000.0;
                        }
                    }
                }
                let mut set = |i: usize, parts: [(u32, f32); 3]| {
                    for (p, (l, s)) in parts.into_iter().enumerate() {
                        pl[p * m + i] = l;
                        ps[p * m + i] = s;
                    }
                };
                set(1, [(7, -5.0), (3, -5.0), (INVALID, -9.0)]);
                set(2, [(3, -5.0), (7, -5.0), (5, -5.0)]);
                if bad {
                    set(3, [(INVALID, -1.0), (INVALID, 2.0), (INVALID, -3.0)]);
                }
                set(4, [(4, 5.0), (2, -rn[4]), (9, 6.0)]);
                set(5, [(6, 1.0), (INVALID, 0.0), (INVALID, 0.0)]);
                set(6, [(1, 3.0e38), (INVALID, 0.0), (INVALID, 0.0)]);
                if bad {
                    set(7, [(0, 1.0), (INVALID, 0.0), (INVALID, 0.0)]);
                }
                // raw = 2^-9 inside the refine band, and raw = -0.5 outside it (clamps to zero).
                set(8, [(2, 0.001_953_125 - rn[8]), (INVALID, 0.0), (INVALID, 0.0)]);
                set(9, [(5, -0.5 - rn[9]), (INVALID, 0.0), (INVALID, 0.0)]);
                let mut w_labels = vec![SENT_U32; m + PAD];
                let mut w_dists = vec![SENT_F32_BITS; m + PAD];
                let mut w_epoch = vec![SENT_U32; EPOCH_OFF + m + PAD];
                let mut w_fault = 0u32;
                for i in 0..m {
                    let mut best_s = MAX_SENTINEL;
                    let mut best_l = 0u32;
                    let mut have = false;
                    for p in 0..split {
                        let (sc, ul) = (ps[p * m + i], pl[p * m + i]);
                        if ul == INVALID {
                            continue;
                        }
                        if !have || sc < best_s || (sc == best_s && ul < best_l) {
                            best_s = sc;
                            best_l = ul;
                            have = true;
                        }
                    }
                    if !have {
                        w_fault |= FAULT_DIST;
                        w_labels[i] = 0;
                        w_dists[i] = 0.0f32.to_bits();
                        if has_epoch {
                            w_epoch[EPOCH_OFF + i] = 0;
                        }
                        continue;
                    }
                    w_labels[i] = best_l;
                    let (dist, f) = finalize_dist(
                        &x[i * d..(i + 1) * d],
                        &c[best_l as usize * d..(best_l as usize + 1) * d],
                        rn[i],
                        best_s,
                        sph,
                    );
                    w_dists[i] = dist.to_bits();
                    if f {
                        w_fault |= FAULT_DIST;
                    }
                    if has_epoch {
                        w_epoch[EPOCH_OFF + i] = best_l;
                    }
                }
                if w_fault != if bad { FAULT_DIST } else { 0 } {
                    rep.fail(format!("{tag}: ORACLE derived fault {w_fault} for the planned rows"));
                }
                let (xb, cb) = if bf16 {
                    (u16b(rt, &bf16_bits_of(&x)), u16b(rt, &bf16_bits_of(&c)))
                } else {
                    (f32b(rt, &x), f32b(rt, &c))
                };
                let (psb, plb, rnb) = (f32b(rt, &ps), u32b(rt, &pl), f32b(rt, &rn));
                let chunk_labels = sent_u32(rt, m + PAD);
                let chunk_dists = sent_f32(rt, m + PAD);
                let epoch_labels = sent_u32(rt, EPOCH_OFF + m + PAD);
                let fault = fault_buf(rt, 0);
                launch_1d(rt, name, m, |bnd| {
                    bnd.bind_gpu_buf(&psb, 0);
                    bnd.bind_gpu_buf(&plb, 1);
                    bnd.bind_gpu_buf(&rnb, 2);
                    bnd.bind_gpu_buf(&xb, 3);
                    bnd.bind_gpu_buf(&cb, 4);
                    bnd.bind_gpu_buf(&chunk_labels, 5);
                    bnd.bind_gpu_buf(&chunk_dists, 6);
                    bnd.bind_gpu_buf(&epoch_labels, 7);
                    bnd.bind_u32(m as u32, 8);
                    bnd.bind_u32(d as u32, 9);
                    bnd.bind_u32(split as u32, 10);
                    bnd.bind_u32(u32::from(sph), 11);
                    bnd.bind_u32(u32::from(bf16), 12);
                    bnd.bind_u32(u32::from(has_epoch), 13);
                    bnd.bind_u32(EPOCH_OFF as u32, 14);
                    bnd.bind_gpu_buf(&fault, 15);
                });
                rep.cmp(&tag, "chunk_labels", &rd32(&chunk_labels, m + PAD), &w_labels);
                rep.cmp(&tag, "chunk_dists", &rd32(&chunk_dists, m + PAD), &w_dists);
                rep.cmp(
                    &tag,
                    "epoch_labels",
                    &rd32(&epoch_labels, EPOCH_OFF + m + PAD),
                    &w_epoch,
                );
                rep.cmp(&tag, "fault", &rd32(&fault, 1), &[w_fault]);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// radix sort and CSR bucket bounds
// ---------------------------------------------------------------------------

fn pack_label_keys(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    let name = "emeans_pack_label_keys";
    ledger.credit(rt, name, &[name], || {
        let m = 300usize;
        let mut labels: Vec<u32> = (0..m).map(|r| ((r * 37 + 5) % 70) as u32).collect();
        labels[17] = 0xffff_fff0; // needs the 64-bit shift
        let tag = format!("{name}[M={m}]");
        let lb = u32b(rt, &labels);
        let keys = sent_u64(rt, m + 16);
        let osc = u32b(rt, &[0xdead]);
        launch_1d(rt, name, m, |bnd| {
            bnd.bind_gpu_buf(&lb, 0);
            bnd.bind_gpu_buf(&keys, 1);
            bnd.bind_gpu_buf(&osc, 2);
            bnd.bind_u32(m as u32, 3);
        });
        let mut want = vec![SENT_U64; m + 16];
        for (r, &l) in labels.iter().enumerate() {
            want[r] = (u64::from(l) << 16) | r as u64;
        }
        rep.cmp(&tag, "keys", &rd64(&keys, m + 16), &want);
        rep.cmp(&tag, "overflow_slice_count", &rd32(&osc, 1), &[0]);
    });
}

fn rand_keys(n: usize, seed: u64) -> Vec<u64> {
    let mut mix = Mix(seed);
    (0..n).map(|_| mix.next()).collect()
}

fn digit(key: u64, shift: u32) -> usize {
    ((key >> shift) & 0xff) as usize
}

fn tile_hist(keys: &[u64], shift: u32, tiles: usize) -> Vec<u32> {
    let mut h = vec![0u32; 256 * tiles];
    for (i, &key) in keys.iter().enumerate() {
        h[digit(key, shift) * tiles + i / 256] += 1;
    }
    h
}

fn radix_hist(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    let name = "emeans_radix_hist";
    ledger.credit(rt, name, &[name], || {
        for (count, shift) in [(600usize, 16u32), (600, 24), (600, 32), (257, 24), (100, 16)] {
            let tiles = count.div_ceil(256);
            let tag = format!("{name}[count={count} shift={shift}]");
            let keys = rand_keys(count, 9000 + u64::from(shift) + count as u64);
            let kb = u64b(rt, &keys);
            let hb = sent_u32(rt, 256 * tiles);
            launch(rt, name, (tiles, 1, 1), (256, 1, 1), |bnd| {
                bnd.bind_gpu_buf(&kb, 0);
                bnd.bind_gpu_buf(&hb, 1);
                bnd.bind_u32(count as u32, 2);
                bnd.bind_u32(tiles as u32, 3);
                bnd.bind_u32(shift, 4);
            });
            rep.cmp(&tag, "hist", &rd32(&hb, 256 * tiles), &tile_hist(&keys, shift, tiles));
        }
    });
}

fn exclusive_scan(h: &[u32]) -> Vec<u32> {
    let mut run = 0u32;
    h.iter()
        .map(|&c| {
            let o = run;
            run += c;
            o
        })
        .collect()
}

fn radix_prefix(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    let name = "emeans_radix_prefix";
    ledger.credit(rt, name, &[name], || {
        for tiles in [1usize, 3] {
            for shape in ["mixed", "one-bin"] {
                let tag = format!("{name}[tiles={tiles} {shape}]");
                let hist: Vec<u32> = (0..256 * tiles)
                    .map(|n| {
                        let (b, t) = (n / tiles, n % tiles);
                        if shape == "one-bin" {
                            if b == 200 {
                                256
                            } else {
                                0
                            }
                        } else if (b * 7 + t) % 5 == 0 {
                            0
                        } else {
                            ((b * 13 + t * 29) % 19) as u32
                        }
                    })
                    .collect();
                let hb = u32b(rt, &hist);
                launch(rt, name, (1, 1, 1), (256, 1, 1), |bnd| {
                    bnd.bind_gpu_buf(&hb, 0);
                    bnd.bind_u32(tiles as u32, 1);
                });
                rep.cmp(&tag, "hist", &rd32(&hb, 256 * tiles), &exclusive_scan(&hist));
            }
        }
    });
}

fn radix_scatter(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    let name = "emeans_radix_scatter";
    ledger.credit(rt, name, &[name], || {
        for (count, shift) in [(600usize, 16u32), (600, 24), (600, 32), (257, 16)] {
            let tiles = count.div_ceil(256);
            let tag = format!("{name}[count={count} shift={shift}]");
            let keys = rand_keys(count, 9500 + u64::from(shift) + count as u64);
            let offsets = exclusive_scan(&tile_hist(&keys, shift, tiles));
            let (kb, hb) = (u64b(rt, &keys), u32b(rt, &offsets));
            let out = sent_u64(rt, count + 16);
            launch(rt, name, (tiles, 1, 1), (256, 1, 1), |bnd| {
                bnd.bind_gpu_buf(&kb, 0);
                bnd.bind_gpu_buf(&hb, 1);
                bnd.bind_gpu_buf(&out, 2);
                bnd.bind_u32(count as u32, 3);
                bnd.bind_u32(tiles as u32, 4);
                bnd.bind_u32(shift, 5);
            });
            // A stable partition by digit: equal digits keep input order.
            let mut idx: Vec<usize> = (0..count).collect();
            idx.sort_by_key(|&i| digit(keys[i], shift));
            let mut want = vec![SENT_U64; count + 16];
            for (pos, &i) in idx.iter().enumerate() {
                want[pos] = keys[i];
            }
            rep.cmp(&tag, "keys_out", &rd64(&out, count + 16), &want);
        }
    });
}

fn find_bucket_bounds(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    let name = "emeans_find_bucket_bounds";
    ledger.credit(rt, name, &[name], || {
        let cases: [(&str, Vec<usize>, u32, bool); 4] = [
            ("two-heavy", vec![300, 0, 270, 5, 25, 0], 0, true),
            ("two-heavy-base3", vec![300, 0, 270, 5, 25, 0], 3, false),
            ("k1-light", vec![100], 0, true),
            ("k1-heavy", vec![300], 0, false),
        ];
        for (case, sizes, b0, final_epoch) in cases {
            let k = sizes.len();
            let m: usize = sizes.iter().sum();
            let tag = format!("{name}[{case} M={m} k={k} base0={b0} final_epoch={final_epoch}]");
            let mut keys: Vec<u64> = Vec::with_capacity(m);
            for (j, &n) in sizes.iter().enumerate() {
                for _ in 0..n {
                    let r = (keys.len() * 37 + 11) % m;
                    keys.push(((j as u64) << 16) | r as u64);
                }
            }
            keys.sort_unstable();
            let heavy: Vec<usize> = (0..k).filter(|&j| sizes[j] > 256).collect();
            let total: usize = heavy.iter().map(|&j| sizes[j].div_ceil(256)).sum();
            let kb = u64b(rt, &keys);
            let info = sent_u32(rt, k * 4);
            let meta = sent_u32(rt, (b0 as usize + total + 4) * 2);
            let osc = u32b(rt, &[b0]);
            let (batch, epoch, pass) = (u64b(rt, &vec![7; k]), u64b(rt, &vec![11; k]), u64b(rt, &vec![13; k]));
            launch_1d(rt, name, k, |bnd| {
                bnd.bind_gpu_buf(&kb, 0);
                bnd.bind_gpu_buf(&info, 1);
                bnd.bind_gpu_buf(&meta, 2);
                bnd.bind_gpu_buf(&osc, 3);
                bnd.bind_gpu_buf(&batch, 4);
                bnd.bind_gpu_buf(&epoch, 5);
                bnd.bind_gpu_buf(&pass, 6);
                bnd.bind_u32(m as u32, 7);
                bnd.bind_u32(k as u32, 8);
                bnd.bind_u32(u32::from(final_epoch), 9);
            });
            let starts: Vec<usize> = sizes
                .iter()
                .scan(0usize, |s, &n| {
                    let st = *s;
                    *s += n;
                    Some(st)
                })
                .collect();
            let got_info = rd32(&info, k * 4);
            let mut want_info = Vec::new();
            let mut bases: Vec<(u32, usize, usize)> = Vec::new();
            for j in 0..k {
                let ns = if sizes[j] > 256 { sizes[j].div_ceil(256) } else { 0 };
                let base = if ns > 0 { got_info[j * 4 + 2] } else { 0 };
                if ns > 0 {
                    bases.push((base, ns, j));
                }
                want_info.extend([starts[j] as u32, (starts[j] + sizes[j]) as u32, base, ns as u32]);
            }
            rep.cmp(&tag, "bucket_info", &got_info, &want_info);
            // The slice base comes from an atomic add, so only the tiling is order-free.
            bases.sort_unstable();
            let mut next = b0;
            for &(base, ns, j) in &bases {
                if base != next {
                    rep.fail(format!(
                        "{tag}: bucket {j} slice base {base}, the tiling expects {next}"
                    ));
                }
                next = base + ns as u32;
            }
            let mut want_meta = vec![SENT_U32; (b0 as usize + total + 4) * 2];
            for &(base, ns, j) in &bases {
                for s in 0..ns {
                    let at = (base as usize + s) * 2;
                    if at + 1 < want_meta.len() {
                        let s_start = starts[j] + s * 256;
                        want_meta[at] = s_start as u32;
                        want_meta[at + 1] = (s_start + 256).min(starts[j] + sizes[j]) as u32;
                    }
                }
            }
            rep.cmp(&tag, "slice_meta", &rd32(&meta, want_meta.len()), &want_meta);
            rep.cmp(&tag, "overflow_slice_count", &rd32(&osc, 1), &[b0 + total as u32]);
            let plus = |base: u64, on: bool| -> Vec<u64> {
                sizes
                    .iter()
                    .map(|&n| if n > 0 && on { base + n as u64 } else { base })
                    .collect()
            };
            rep.cmp(&tag, "batch_counts", &rd64(&batch, k), &plus(7, true));
            rep.cmp(&tag, "epoch_counts", &rd64(&epoch, k), &plus(11, true));
            rep.cmp(&tag, "pass_counts", &rd64(&pass, k), &plus(13, final_epoch));
        }
    });
}

// ---------------------------------------------------------------------------
// slice gather and fold
// ---------------------------------------------------------------------------

/// `rows x d` statistics operand: its effective f32 values and the buffer the kernel reads.
fn stats_operand(rt: &Arc<GpuRuntime>, rows: usize, d: usize, bf16: bool, seed: u64) -> (Vec<f32>, GpuBuffer) {
    let raw = random_f32(rows * d, seed);
    let values: Vec<f32> = raw.iter().map(|&v| eff(v, bf16)).collect();
    let b = if bf16 {
        u16b(rt, &bf16_bits_of(&raw))
    } else {
        f32b(rt, &raw)
    };
    (values, b)
}

fn local_row(p: usize, rows: usize) -> usize {
    (p * 37 + 11) % rows
}

fn gather_slice_partials(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    for bf16 in [true, false] {
        let name = if bf16 {
            "emeans_gather_slice_partials_bf16"
        } else {
            "emeans_gather_slice_partials_f32"
        };
        ledger.credit(rt, name, &[name], || {
            let (d, rows, n_keys, grid_slices) = (67usize, 600usize, 556usize, 16usize);
            let tag = format!("{name}[d={d} slices=3 of {grid_slices}]");
            let (xv, xb) = stats_operand(rt, rows, d, bf16, 9800);
            let keys: Vec<u64> = (0..n_keys)
                .map(|p| (((p / 256) as u64) << 16) | local_row(p, rows) as u64)
                .collect();
            let mut meta: Vec<u32> = vec![0, 256, 256, 512, 512, 556];
            for _ in 3..grid_slices {
                meta.extend([0, 5]); // slices past the count must stay unread
            }
            let (kb, mb, osc) = (u64b(rt, &keys), u32b(rt, &meta), u32b(rt, &[3]));
            let out = sent_f32(rt, grid_slices * d);
            launch(
                rt,
                name,
                (d.div_ceil(16), grid_slices.div_ceil(16), 1),
                (16, 16, 1),
                |bnd| {
                    bnd.bind_gpu_buf(&kb, 0);
                    bnd.bind_gpu_buf(&xb, 1);
                    bnd.bind_gpu_buf(&mb, 2);
                    bnd.bind_gpu_buf(&osc, 3);
                    bnd.bind_gpu_buf(&out, 4);
                    bnd.bind_u32(d as u32, 5);
                },
            );
            let mut want = vec![SENT_F32_BITS; grid_slices * d];
            for s in 0..3 {
                for col in 0..d {
                    let mut acc = 0.0f32;
                    for &key in &keys[meta[2 * s] as usize..meta[2 * s + 1] as usize] {
                        acc += xv[(key & 0xffff) as usize * d + col];
                    }
                    want[s * d + col] = acc.to_bits();
                }
            }
            rep.cmp(&tag, "slice_partials", &rd32(&out, grid_slices * d), &want);
        });
    }
}

fn fold_slice_partials(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    let name = "emeans_fold_slice_partials";
    ledger.credit(rt, name, &[name], || {
        let (k, d, rows, n_keys) = (5usize, 67usize, 64usize, 200usize);
        // (start, end, base, num_slices): direct, sliced at base 1, empty, direct single row,
        // sliced at base 3.
        let info: [[u32; 4]; 5] = [
            [0, 40, 0, 0],
            [41, 141, 1, 2],
            [50, 50, 0, 0],
            [40, 41, 0, 0],
            [141, 190, 3, 3],
        ];
        let keys: Vec<u64> = (0..n_keys).map(|p| local_row(p, rows) as u64).collect();
        let partials: Vec<f32> = random_f32(6 * d, 9810).iter().map(|v| v * 10.0).collect();
        let batch0: Vec<f32> = random_f32(k * d, 9811).iter().map(|v| v + 2.0).collect();
        let pass0: Vec<f32> = random_f32(k * d, 9812).iter().map(|v| v * 3.0 + 5.0).collect();
        for bf16 in [true, false] {
            for final_epoch in [true, false] {
                let tag = format!("{name}[stats_bf16={bf16} final_epoch={final_epoch}]");
                let (xv, xb) = stats_operand(rt, rows, d, bf16, 9813);
                let kb = u64b(rt, &keys);
                let ib = u32b(rt, &info.concat());
                let (pb, bb, sb) = (f32b(rt, &partials), f32b(rt, &batch0), f32b(rt, &pass0));
                launch(rt, name, (d.div_ceil(16), k.div_ceil(16), 1), (16, 16, 1), |bnd| {
                    bnd.bind_gpu_buf(&kb, 0);
                    bnd.bind_gpu_buf(&xb, 1);
                    bnd.bind_gpu_buf(&ib, 2);
                    bnd.bind_gpu_buf(&pb, 3);
                    bnd.bind_gpu_buf(&bb, 4);
                    bnd.bind_gpu_buf(&sb, 5);
                    bnd.bind_u32(k as u32, 6);
                    bnd.bind_u32(d as u32, 7);
                    bnd.bind_u32(u32::from(bf16), 8);
                    bnd.bind_u32(u32::from(final_epoch), 9);
                });
                let (mut wb, mut ws) = (batch0.clone(), pass0.clone());
                for (j, [start, end, base, ns]) in info.iter().copied().enumerate() {
                    if start >= end {
                        continue;
                    }
                    for col in 0..d {
                        let at = j * d + col;
                        if ns == 0 {
                            let mut acc = 0.0f32;
                            for &key in &keys[start as usize..end as usize] {
                                acc += xv[(key & 0xffff) as usize * d + col];
                            }
                            wb[at] = batch0[at] + acc;
                            if final_epoch {
                                ws[at] = pass0[at] + acc;
                            }
                        } else {
                            let mut b_acc = batch0[at];
                            let mut p_acc = if final_epoch { pass0[at] } else { 0.0 };
                            for s in 0..ns {
                                let part = partials[(base + s) as usize * d + col];
                                b_acc += part;
                                if final_epoch {
                                    p_acc += part;
                                }
                            }
                            wb[at] = b_acc;
                            if final_epoch {
                                ws[at] = p_acc;
                            }
                        }
                    }
                }
                rep.cmp(&tag, "batch_sums", &rd32(&bb, k * d), &bits(&wb));
                rep.cmp(&tag, "pass_sums", &rd32(&sb, k * d), &bits(&ws));
            }
        }
    });
}

// ---------------------------------------------------------------------------
// accum_chunk_obj
// ---------------------------------------------------------------------------

fn accum_chunk_obj(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    let name = "emeans_accum_chunk_obj";
    ledger.credit(rt, name, &[name], || {
        for (m, prior, planted) in [(600usize, 1.5f64, true), (100, 0.0, false)] {
            let tag = format!("{name}[M={m} prior={prior} planted={planted}]");
            let mut dists: Vec<f32> = random_f32(m, 9900 + m as u64)
                .iter()
                .enumerate()
                .map(|(i, &v)| v * 10.0f32.powi((i * 7 % 11) as i32 - 5))
                .collect();
            if !planted {
                for v in &mut dists {
                    *v = v.abs();
                }
            } else {
                dists[100] = 1.0e-40; // f32 subnormal: only reinterpreted, so it stays exact
                dists[250] = f32::NAN;
                dists[400] = f32::INFINITY;
            }
            // Native f64 `+` in the kernel's order: lane sums, tree 128..1, then the prior.
            let mut lanes = [0.0f64; 256];
            for (r, &v) in dists.iter().enumerate() {
                if !nonfinite32(v.to_bits()) {
                    lanes[r % 256] += f64::from(v);
                }
            }
            let mut stride = 128;
            while stride > 0 {
                for t in 0..stride {
                    lanes[t] += lanes[t + stride];
                }
                stride >>= 1;
            }
            let want = prior + lanes[0];
            let (db, ob, fault) = (f32b(rt, &dists), u64b(rt, &[prior.to_bits()]), fault_buf(rt, 0));
            launch(rt, name, (1, 1, 1), (256, 1, 1), |bnd| {
                bnd.bind_gpu_buf(&db, 0);
                bnd.bind_gpu_buf(&ob, 1);
                bnd.bind_u32(m as u32, 2);
                bnd.bind_gpu_buf(&fault, 3);
            });
            rep.cmp(&tag, "epoch_obj_bits", &rd64(&ob, 1), &[want.to_bits()]);
            rep.cmp(&tag, "fault", &rd32(&fault, 1), &[if planted { FAULT_DIST } else { 0 }]);
        }
    });
}

// ---------------------------------------------------------------------------
// apply_ema
// ---------------------------------------------------------------------------

fn apply_ema(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    let name = "emeans_apply_ema";
    ledger.credit(rt, name, &[name], || {
        let (k, d) = (4usize, 300usize);
        let counts0: [u64; 4] = [7, 0, 1000, 13];
        let usage0: [f32; 4] = [5.0, 9.0, 0.0, 2.5];
        let cent0 = random_f32(k * d, 10_000);
        let sums0: Vec<f32> = random_f32(k * d, 10_001).iter().map(|v| v * 50.0).collect();
        for (alpha, preset) in [(1.0f32, 0u32), (0.25, 0), (0.0, 0), (0.25, 1)] {
            let tag = format!("{name}[alpha={alpha} fault_preset={preset}]");
            let (cb, ub, sb) = (f32b(rt, &cent0), f32b(rt, &usage0), f32b(rt, &sums0));
            let (nb, fault) = (u64b(rt, &counts0), fault_buf(rt, preset));
            launch(rt, name, (k, 1, 1), (256, 1, 1), |bnd| {
                bnd.bind_gpu_buf(&cb, 0);
                bnd.bind_gpu_buf(&ub, 1);
                bnd.bind_gpu_buf(&sb, 2);
                bnd.bind_gpu_buf(&nb, 3);
                bnd.bind_u32(k as u32, 4);
                bnd.bind_u32(d as u32, 5);
                bnd.bind_f32(alpha, 6);
                bnd.bind_gpu_buf(&fault, 7);
            });
            let (mut wc, mut wu) = (cent0.clone(), usage0.to_vec());
            let replace = alpha == 1.0;
            for j in 0..k {
                if preset == 0 {
                    let (n_j, n_old) = (counts0[j], usage0[j]);
                    if n_j > 0 {
                        let nf = n_j as f32;
                        let n_new = if replace {
                            alpha * nf
                        } else {
                            (1.0 - alpha) * n_old + alpha * nf
                        };
                        let w = if n_new > 0.0 { (alpha * nf) / n_new } else { 0.0 };
                        wu[j] = n_new;
                        for col in 0..d {
                            let at = j * d + col;
                            let mean = sums0[at] / nf;
                            let c_old = cent0[at];
                            wc[at] = if w == 1.0 {
                                mean
                            } else if w == 0.0 {
                                c_old
                            } else {
                                c_old + w * (mean - c_old)
                            };
                        }
                    } else {
                        wu[j] = if replace { 0.0 } else { (1.0 - alpha) * n_old };
                    }
                }
            }
            rep.cmp(&tag, "centroids", &rd32(&cb, k * d), &bits(&wc));
            rep.cmp(&tag, "usage", &rd32(&ub, k), &bits(&wu));
            rep.cmp(&tag, "batch_sums", &rd32(&sb, k * d), &vec![0u32; k * d]);
            rep.cmp(&tag, "batch_counts", &rd64(&nb, k), &[0u64; 4]);
            rep.cmp(&tag, "fault", &rd32(&fault, 1), &[preset]);
        }
    });
}

// ---------------------------------------------------------------------------
// revive_prepare and revive_splits
// ---------------------------------------------------------------------------

fn ordered_u32(f: f32) -> u32 {
    let clean = if f == 0.0 { 0.0 } else { f };
    let u = clean.to_bits();
    if u & 0x8000_0000 != 0 {
        !u
    } else {
        u ^ 0x8000_0000
    }
}

fn revive_prepare(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    let name = "emeans_revive_prepare";
    ledger.credit(rt, name, &[name], || {
        let threshold = 0.1f32;
        for (k, seed, step, preset) in [
            (6usize, 0x1234_5678_9abc_def0u64, 3u64, 0u32),
            (300, 0x0fed_cba9_8765_4321, 17, 0),
            (300, 0x0fed_cba9_8765_4321, 17, 1),
        ] {
            let tag = format!("{name}[k={k} step={step} fault_preset={preset}]");
            // Integer usages: the total is exact whatever the summation order, the unused
            // ones sit below the threshold, and every donor is >= 30 so ln(usage) dominates
            // the Gumbel term (no cancellation in the log difference).
            let usage: Vec<f32> = (0..k)
                .map(|j| match j % 5 {
                    0 => 0.0,
                    1 => 3.0,
                    _ => 30.0 + ((j * 37) % 400) as f32,
                })
                .collect();
            let total = lane_sum(k, |j| usage[j]);
            let thresh = threshold * (total / k as f32);
            let is_donor = |j: usize| {
                let below = usage[j] < thresh;
                !below && usage[j] > 0.0
            };
            let unused: Vec<u32> = (0..k).filter(|&j| usage[j] < thresh).map(|j| j as u32).collect();
            let n_donors = (0..k).filter(|&j| is_donor(j)).count();
            let nsplit = unused.len().min(n_donors) as u32;
            let (ub, kb) = (f32b(rt, &usage), sent_u64(rt, k));
            let (ib, mb, fault) = (sent_u32(rt, k), u32b(rt, &[9, 5]), fault_buf(rt, preset));
            launch(rt, name, (1, 1, 1), (256, 1, 1), |bnd| {
                bnd.bind_gpu_buf(&ub, 0);
                bnd.bind_gpu_buf(&kb, 1);
                bnd.bind_gpu_buf(&ib, 2);
                bnd.bind_gpu_buf(&mb, 3);
                bnd.bind_u32(k as u32, 4);
                bnd.bind_f32(threshold, 5);
                bnd.bind_bytes(&seed.to_ne_bytes(), 6);
                bnd.bind_bytes(&step.to_ne_bytes(), 7);
                bnd.bind_gpu_buf(&fault, 8);
            });
            let mut want_ids = vec![SENT_U32; k];
            let want_meta = if preset == 0 {
                want_ids[..unused.len()].copy_from_slice(&unused);
                [nsplit, 5 + nsplit]
            } else {
                [0, 5]
            };
            rep.cmp(&tag, "unused_ids", &rd32(&ib, k), &want_ids);
            rep.cmp(&tag, "revive_meta", &rd32(&mb, 2), &want_meta);
            rep.cmp(&tag, "fault", &rd32(&fault, 1), &[preset]);
            let got = rd64(&kb, k);
            let mut donors: Vec<(i64, u32, u32)> = Vec::new(); // (host ordered g, gpu high, host high)
            for (j, &key) in got.iter().enumerate() {
                if (key & 0xffff_ffff) as usize != j {
                    rep.fail(format!("{tag}: donor_keys[{j}] low word {:#x}", key & 0xffff_ffff));
                }
                let high = (key >> 32) as u32;
                if !is_donor(j) {
                    if high != 0xffff_ffff {
                        rep.fail(format!("{tag}: non-donor {j} high word {high:#x}, want 0xffffffff"));
                    }
                    continue;
                }
                let g = revival_gumbel_key(usage[j], seed, step, j as u32);
                if g < 0.3 {
                    rep.fail(format!("{tag}: ORACLE donor {j} has g {g}, cancellation not excluded"));
                }
                let host_ord = ordered_u32(g);
                // precise::log is not pinned to an ulp bound here: the ordered-integer distance
                // of the decoded key stays within 2 ulp of the host f32 logarithms.
                let dist = (i64::from(!high) - i64::from(host_ord)).abs();
                if dist > 2 {
                    rep.fail(format!(
                        "{tag}: donor {j} key ordered {:#x} vs host {host_ord:#x} ({dist} ulp)",
                        !high
                    ));
                }
                donors.push((i64::from(host_ord), high, !host_ord));
            }
            // The donor rank order must agree wherever the host keys differ by more than 4 ulp.
            for a in 0..donors.len() {
                for b in a + 1..donors.len() {
                    if (donors[a].0 - donors[b].0).abs() > 4
                        && donors[a].1.cmp(&donors[b].1) != donors[a].2.cmp(&donors[b].2)
                    {
                        rep.fail(format!("{tag}: donor rank order disagrees between donors {a} and {b}"));
                    }
                }
            }
        }
    });
}

fn revive_splits(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    let name = "emeans_revive_splits";
    ledger.credit(rt, name, &[name], || {
        let (k, d) = (16usize, 300usize);
        let mut cent0 = random_f32(k * d, 10_100);
        cent0[..d].fill(1.0); // collapses under a small epsilon
        cent0[d..2 * d].fill(0.0); // zero donor
        cent0[2 * d..3 * d].fill(-1.0); // negative collapsing donor
        let usage0: Vec<f32> = (0..k).map(|j| 2.0 + j as f32 * 0.75).collect();
        let mut dkeys: Vec<u64> = (0..8u64).map(|p| ((0x1234_0000 + p) << 32) | p).collect();
        dkeys[4] = (0xffff_ffffu64 << 32) | 4; // sentinel: skipped
        let unused: [u32; 8] = [8, 9, 10, 11, 12, 5, 14, 15]; // pair 5: donor == target
        let nsplit = 6u32; // pairs 6 and 7 stay untouched
        for (bf16, eps) in [(true, 1.0e-4f32), (true, 0.02), (false, 1.0e-8), (false, 1.0e-3)] {
            let tag = format!("{name}[bf16_repr={bf16} eps={eps}]");
            let (kb, ib, mb) = (u64b(rt, &dkeys), u32b(rt, &unused), u32b(rt, &[nsplit, 0]));
            let (cb, ub) = (f32b(rt, &cent0), f32b(rt, &usage0));
            launch(rt, name, (k, 1, 1), (256, 1, 1), |bnd| {
                bnd.bind_gpu_buf(&kb, 0);
                bnd.bind_gpu_buf(&ib, 1);
                bnd.bind_gpu_buf(&mb, 2);
                bnd.bind_gpu_buf(&cb, 3);
                bnd.bind_gpu_buf(&ub, 4);
                bnd.bind_u32(k as u32, 5);
                bnd.bind_u32(d as u32, 6);
                bnd.bind_f32(eps, 7);
                bnd.bind_u32(u32::from(bf16), 8);
            });
            let (mut wc, mut wu) = (cent0.clone(), usage0.clone());
            for pair in 0..nsplit as usize {
                let dkey = dkeys[pair];
                if (dkey >> 32) as u32 == 0xffff_ffff {
                    continue;
                }
                let donor = (dkey & 0xffff_ffff) as usize;
                let target = unused[pair] as usize;
                if donor == target {
                    continue;
                }
                let mut any_diff = false;
                let mut child0 = 0.0f32;
                for col in 0..d {
                    let c = cent0[donor * d + col];
                    let off = if col % 2 == 0 { eps } else { -eps };
                    let donor_c = c * (1.0f32 - off);
                    let child_c = c * (1.0f32 + off);
                    wc[donor * d + col] = donor_c;
                    wc[target * d + col] = child_c;
                    if col == 0 {
                        child0 = child_c;
                    }
                    if bf16 {
                        if bf16f(f32_to_bf16_bits_gpu(donor_c)) != bf16f(f32_to_bf16_bits_gpu(child_c)) {
                            any_diff = true;
                        }
                    } else {
                        let flush = |v: f32| {
                            if (v.to_bits() & 0x7fff_ffff) < 0x0080_0000 {
                                0.0
                            } else {
                                v
                            }
                        };
                        if flush(donor_c) != flush(child_c) {
                            any_diff = true;
                        }
                    }
                }
                if !any_diff {
                    child0 = nudge_child0(child0, bf16);
                    wc[target * d] = child0;
                }
                let half = usage0[donor] * 0.5;
                wu[donor] = half;
                wu[target] = half;
            }
            rep.cmp(&tag, "centroids", &rd32(&cb, k * d), &bits(&wc));
            rep.cmp(&tag, "usage", &rd32(&ub, k), &bits(&wu));
        }
    });
}

/// The collapse nudge of `emeans_revive_splits` (metal 1866-1889): one representable step.
fn nudge_child0(child0: f32, bf16: bool) -> f32 {
    if bf16 {
        let b = f32_to_bf16_bits_gpu(child0);
        let step = if b & 0x7fff < 0x0080 {
            return f32::from_bits(0x2080_0000);
        } else if b & 0x8000 == 0 {
            if b >= 0x7f7f {
                b - 1
            } else {
                b + 1
            }
        } else if b == 0x8080 {
            0
        } else {
            b - 1
        };
        f32::from_bits(u32::from(step) << 16)
    } else {
        let u = child0.to_bits();
        let step = if u & 0x7fff_ffff < 0x0080_0000 {
            return f32::from_bits(0x2080_0000);
        } else if u & 0x8000_0000 == 0 {
            if u >= 0x7f7f_ffff {
                u - 1
            } else {
                u + 1
            }
        } else if u == 0x8080_0000 {
            0
        } else {
            u - 1
        };
        f32::from_bits(step)
    }
}

pub fn run(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    assert!(rt.has_tensorops(), "emeans sweep needs TensorOps");
    let mut rep = Report::default();
    prepare_centroids(rt, ledger, &mut rep);
    stage_rows_bf16(rt, ledger, &mut rep);
    stage_rows_f32(rt, ledger, &mut rep);
    gather_rows(rt, ledger, &mut rep);
    for tensorops in [true, false] {
        for bf16 in [true, false] {
            assign_kernel(rt, ledger, &mut rep, bf16, tensorops);
        }
    }
    reduce_split_c(rt, ledger, &mut rep);
    pack_label_keys(rt, ledger, &mut rep);
    radix_hist(rt, ledger, &mut rep);
    radix_prefix(rt, ledger, &mut rep);
    radix_scatter(rt, ledger, &mut rep);
    find_bucket_bounds(rt, ledger, &mut rep);
    gather_slice_partials(rt, ledger, &mut rep);
    fold_slice_partials(rt, ledger, &mut rep);
    accum_chunk_obj(rt, ledger, &mut rep);
    apply_ema(rt, ledger, &mut rep);
    revive_prepare(rt, ledger, &mut rep);
    revive_splits(rt, ledger, &mut rep);
    // One failure for the whole sweep, raised before the ledger prints any PASS line.
    rep.assert_clean();
}
