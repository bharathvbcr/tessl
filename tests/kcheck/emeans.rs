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

use std::collections::BTreeMap;
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

    /// Comparison within `tol` ulps over an ordered-integer view.
    fn cmp_ulp(&mut self, tag: &str, what: &str, got: &[u32], want: &[u32], (tol, wide): (i64, bool)) {
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
        }
    }

    fn assert_clean(&self) {
        // Group by kernel so a long list for one kernel cannot hide another kernel's mismatches.
        const PER_KERNEL: usize = 4;
        let mut by_kernel: BTreeMap<&str, Vec<&String>> = BTreeMap::new();
        for f in &self.failures {
            by_kernel.entry(f.split('[').next().unwrap_or(f)).or_default().push(f);
        }
        let mut lines = Vec::new();
        for (kernel, list) in &by_kernel {
            lines.push(format!("{kernel}: {} mismatches", list.len()));
            lines.extend(list.iter().take(PER_KERNEL).map(|f| format!("  {f}")));
        }
        assert!(
            self.failures.is_empty(),
            "emeans kcheck: {} mismatches in {} kernels\n{}",
            self.failures.len(),
            by_kernel.len(),
            lines.join("\n")
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

/// One threadgroup of `threads` threads: more threads than the kernel has items, so its bounds
/// guard has to hold for the surplus.
fn launch_threads(rt: &Arc<GpuRuntime>, name: &str, threads: usize, bind: impl FnOnce(&mut Binder<'_>)) {
    launch(rt, name, (1, 1, 1), (threads, 1, 1), bind);
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
    /// Fault word the case must raise, for the bf16 and the f32 kernel.
    fault: (u32, u32),
}

struct PrepWant {
    c_eff: Vec<u32>,
    norm2: Vec<f32>,
    fault: u32,
    /// The scaled coordinate before it is rounded to bf16 (what `sph` rows are checked against).
    y: Vec<f32>,
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
    let mut ys = Vec::with_capacity(k * d);
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
            ys.push(v);
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
        norm2.push(prep_norm(&stored, d, bf16));
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
        }
        c_eff.extend(stored);
    }
    PrepWant {
        c_eff,
        norm2,
        fault,
        y: ys,
    }
}

fn prep_cases() -> Vec<PrepCase> {
    let mut cases = Vec::new();
    // Round-half-even ties (low 16 bits exactly 0x8000) sit at the front of the random rows: an even
    // bf16 neighbour must stay, an odd one must go up.
    let ties = [
        0x3f80_8000u32,
        0x3f81_8000,
        0xbf80_8000,
        0xbf81_8000,
        0x4040_8000,
        0x3e00_8000,
    ];
    for (k, d) in [(1usize, 67usize), (3, 67), (1, 300), (3, 300)] {
        let mut data = random_f32(k * d, 4100 + (k * 1000 + d) as u64);
        for (slot, bits) in data.iter_mut().zip(ties) {
            *slot = f32::from_bits(bits);
        }
        cases.push(PrepCase {
            name: "random",
            k,
            d,
            sph: false,
            data,
            fault: (0, 0),
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
        fault: (0, 0),
    });
    for d in [67usize, 300] {
        let mut data = random_f32(5 * d, 4200 + d as u64);
        for v in &mut data[d..2 * d] {
            *v *= 1000.0;
        }
        for v in &mut data[2 * d..3 * d] {
            *v *= 1.0e-3;
        }
        data[3 * d..4 * d].fill(0.0);
        // An all-negative row: a signed max-abs would read it as the zero row.
        for v in &mut data[4 * d..] {
            *v = -(v.abs() + 0.05);
        }
        cases.push(PrepCase {
            name: "sph-random",
            k: 5,
            d,
            sph: true,
            data,
            fault: (0, 0),
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
        fault: (FAULT_INPUT, FAULT_INPUT),
    });
    // f32::MAX rounds to bf16 infinity (fault 2, coordinate 0) and squares to f32 infinity (fault 8).
    let mut max = random_f32(2 * d, 4400);
    max[3] = f32::MAX;
    cases.push(PrepCase {
        name: "f32-max",
        k: 2,
        d,
        sph: false,
        data: max,
        fault: (FAULT_BF16, FAULT_CENT_NORM),
    });
    // 3.0e38 is a finite bf16, so only the squared norm overflows.
    let mut big = random_f32(2 * d, 4500);
    big[d + 9] = 3.0e38;
    cases.push(PrepCase {
        name: "3e38",
        k: 2,
        d,
        sph: false,
        data: big,
        fault: (FAULT_CENT_NORM, FAULT_CENT_NORM),
    });
    cases
}

/// bf16 spherical rows against the unrounded scaled value `y`. `precise::rsqrt` is not pinned to
/// an ulp bound, so a coordinate may land on either bf16 neighbour only when `y` is within 1e-5
/// (relative) of a rounding boundary; everywhere else the bits are exact. Estimate, not measured: a
/// 0.02% scale error moves a coordinate across a boundary with probability about 2e-4 divided by
/// the relative bf16 spacing (2^-8 to 2^-7), i.e. roughly 2.5% to 5% of the coordinates.
fn sph_bf16_tight(rep: &mut Report, tag: &str, got: &[u32], y: &[f32]) {
    let bad: Vec<String> = got
        .iter()
        .zip(y)
        .enumerate()
        .filter_map(|(i, (&g, &y))| {
            let lo = f32_to_bf16_bits_gpu(y * (1.0 - 1.0e-5));
            let hi = f32_to_bf16_bits_gpu(y * (1.0 + 1.0e-5));
            (g as u16 != lo && g as u16 != hi).then(|| format!("[{i}] got {g:#x} accepted {lo:#x}/{hi:#x} (y {y})"))
        })
        .collect();
    if !bad.is_empty() {
        rep.fail(format!(
            "{tag}: c_eff: {} of {} outside the 1e-5 rounding band: {}",
            bad.len(),
            got.len(),
            bad.iter().take(3).cloned().collect::<Vec<_>>().join("; ")
        ));
    }
}

fn prepare_centroids(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    for bf16 in [true, false] {
        let name: &'static str = if bf16 {
            "emeans_prepare_centroids_bf16"
        } else {
            "emeans_prepare_centroids_f32"
        };
        ledger.credit(rt, name, &[name], || {
            for case in &prep_cases() {
                let (k, d) = (case.k, case.d);
                let rows = k + 1; // one launched group past k: the `j >= k` guard must hold
                let tag = format!("{name}[{} k={k} d={d} sph={}]", case.name, case.sph);
                let want = prep_oracle(case, bf16);
                let planned = if bf16 { case.fault.0 } else { case.fault.1 };
                if want.fault != planned {
                    rep.fail(format!(
                        "{tag}: ORACLE derived fault {} but the case plans {planned}",
                        want.fault
                    ));
                }
                let mut padded = case.data.clone();
                padded.extend(random_f32(d, 4900 + d as u64));
                let cent = f32b(rt, &padded);
                let n = rows * d;
                let c_eff = if bf16 { sent_u16(rt, n) } else { sent_f32(rt, n) };
                let norm2 = sent_f32(rt, rows);
                let fault = fault_buf(rt, 0);
                launch(rt, name, (rows, 1, 1), (256, 1, 1), |bnd| {
                    bnd.bind_gpu_buf(&cent, 0);
                    bnd.bind_gpu_buf(&c_eff, 1);
                    bnd.bind_gpu_buf(&norm2, 2);
                    bnd.bind_gpu_buf(&fault, 3);
                    bnd.bind_u32(k as u32, 4);
                    bnd.bind_u32(d as u32, 5);
                    bnd.bind_u32(u32::from(case.sph), 6);
                });
                let got: Vec<u32> = if bf16 {
                    rd16(&c_eff, n).into_iter().map(u32::from).collect()
                } else {
                    rd32(&c_eff, n)
                };
                let pad = if bf16 { u32::from(SENT_U16) } else { SENT_F32_BITS };
                rep.cmp(&tag, "c_eff canary row", &got[k * d..], &vec![pad; d]);
                let got_norm = rd32(&norm2, rows);
                rep.cmp(&tag, "norm2 canary", &got_norm[k..], &[SENT_F32_BITS]);
                rep.cmp(&tag, "fault", &rd32(&fault, 1), &[want.fault]);
                let got = &got[..k * d];
                if case.sph {
                    if bf16 {
                        sph_bf16_tight(rep, &tag, got, &want.y);
                    } else if case.name == "pow4" {
                        // rsqrt of 4 and 16 is exact: the power-of-4 rows are compared bit for bit.
                        rep.cmp(&tag, "c_eff", got, &want.c_eff);
                    } else {
                        // precise::rsqrt feeds only the scaled coordinate: 2 f32 ulp.
                        rep.cmp_ulp(&tag, "c_eff", got, &want.c_eff, (2, true));
                    }
                    // Every norm is recomputed exactly from the coordinates the kernel stored.
                    let want_norm: Vec<u32> = (0..k)
                        .map(|j| prep_norm(&got[j * d..(j + 1) * d], d, bf16).to_bits())
                        .collect();
                    rep.cmp(&tag, "norm2", &got_norm[..k], &want_norm);
                } else {
                    rep.cmp(&tag, "c_eff", got, &want.c_eff);
                    rep.cmp(&tag, "norm2", &got_norm[..k], &bits(&want.norm2));
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

/// Compare the `n` staged rows and the canary row the launch had one group too many for.
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
    let (mut wb, mut wf, mut wn) = (want.bf16.clone(), want.f32s.clone(), bits(&want.norm2));
    wb.extend(vec![SENT_U16; d]);
    wf.extend(vec![SENT_F32_BITS; d]);
    wn.push(SENT_F32_BITS);
    rep.cmp(tag, "dst_bf16 (+canary row)", &rd16(outs.0, (n + 1) * d), &wb);
    rep.cmp(tag, "dst_f32 (+canary row)", &rd32(outs.1, (n + 1) * d), &wf);
    rep.cmp(tag, "dst_norm2 (+canary)", &rd32(outs.2, n + 1), &wn);
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
            // One launched group past n: the `row >= n` guard must hold.
            let mut padded = src.clone();
            padded.extend(bf16_bits_of(&random_f32(*d, 5900)));
            let sb = u16b(rt, &padded);
            let (ob, of, on, fl) = (
                sent_u16(rt, (n + 1) * d),
                sent_f32(rt, (n + 1) * d),
                sent_f32(rt, n + 1),
                fault_buf(rt, 0),
            );
            launch(rt, name, (n + 1, 1, 1), (256, 1, 1), |bnd| {
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

/// (label, d, data, planned fault per triple, triple enabled)
type StageF32Case = (&'static str, usize, Vec<f32>, [u32; 5], [bool; 5]);

fn stage_rows_f32(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    let name = "emeans_stage_rows_f32";
    ledger.credit(rt, name, &[name], || {
        let n = 5usize;
        // (write_bf16, write_f32, norm_from_bf16): Bf16, Bf16AssignF32Stats and F32 as the host
        // uses them, then norm_from_bf16 without write_bf16, which the kernel also defines.
        let triples = [
            (true, false, true),
            (true, true, true),
            (false, true, false),
            (false, false, true),
            (false, true, true),
        ];
        let all = [true; 5];
        let mut cases: Vec<StageF32Case> = Vec::new();
        // Ties (low 16 bits exactly 0x8000) and an f32 value just below the smallest normal, which
        // rounds to bf16 0x0080 unless the subnormal flush is taken.
        let ties = [
            0x3f80_8000u32,
            0x3f81_8000,
            0xbf80_8000,
            0xbf81_8000,
            0x4040_8000,
            0x3e00_8000,
        ];
        for d in [67usize, 300] {
            let mut data = random_f32(n * d, 6100 + d as u64);
            for (slot, b) in data.iter_mut().zip(ties) {
                *slot = f32::from_bits(b);
            }
            cases.push(("base", d, data, [0; 5], all));
        }
        let d = 67;
        let mut nonfinite = random_f32(n * d, 6200);
        nonfinite[0] = f32::INFINITY;
        nonfinite[d + 1] = f32::NEG_INFINITY;
        nonfinite[2 * d + 2] = f32::NAN;
        cases.push(("nonfinite", d, nonfinite, [FAULT_INPUT; 5], all));
        // f32::MAX rounds to bf16 infinity (fault 2, coordinate 0) wherever a bf16 conversion
        // happens; under the f32 triple nothing converts and the squared norm overflows (fault 4).
        let mut max = random_f32(n * d, 6300);
        max[0] = f32::MAX;
        cases.push((
            "f32-max",
            d,
            max,
            [FAULT_BF16, FAULT_BF16, FAULT_ROW_NORM, FAULT_BF16, FAULT_BF16],
            all,
        ));
        // 3.0e38 is a finite bf16 (0x7f62..), so no triple raises fault 2; its square overflows.
        let mut big = random_f32(n * d, 6400);
        big[d + 1] = 3.0e38;
        cases.push(("3e38", d, big, [FAULT_ROW_NORM; 5], all));
        // Subnormal inputs, only under triples that never store the f32 copy, so nothing depends on
        // how the device stores a subnormal: the flush to a signed zero decides the bf16 bits.
        let mut sub = random_f32(n * d, 6500);
        for (slot, b) in sub
            .iter_mut()
            .zip([0x007f_c000u32, 0x807f_c000, 0x0000_0001, 0x8000_0001])
        {
            *slot = f32::from_bits(b);
        }
        cases.push(("subnormal", d, sub, [0; 5], [true, false, false, true, false]));
        for (case, d, src, planned, enabled) in &cases {
            for (t, &flags) in triples.iter().enumerate() {
                if !enabled[t] {
                    continue;
                }
                let tag = format!(
                    "{name}[{case} d={d} write_bf16={} write_f32={} norm_from_bf16={}]",
                    flags.0, flags.1, flags.2
                );
                let want = stage_f32_oracle(src, n, *d, flags);
                // One launched group past n: the `row >= n` guard must hold.
                let mut padded = src.clone();
                padded.extend(random_f32(*d, 6900));
                let sb = f32b(rt, &padded);
                let (ob, of, on, fl) = (
                    sent_u16(rt, (n + 1) * d),
                    sent_f32(rt, (n + 1) * d),
                    sent_f32(rt, n + 1),
                    fault_buf(rt, 0),
                );
                launch(rt, name, (n + 1, 1, 1), (256, 1, 1), |bnd| {
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
            for d in [1usize, 16, 67] {
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

// Centroid roles in the special fixture (k = 70, M = 70, d = 67).
const SELF_C: usize = 40; // non-integer, but with 8 significant bits (see assign_fx)
const HUGE_C: usize = 9; // shares a 9.0e18 coordinate with row 20
const EDGE_C: usize = 68; // [90, 9, 3, 1]: |c|^2 = 8191, row 60 sits on it with rn = 8192
const OVF_C: usize = 69; // [2.5e19, 0, ...]: 2 * dot overflows for row 55

/// `x` and `c` are small integers (|v| <= 9), so every product and partial sum is exact in f32
/// and the matrix unit's accumulation order cannot change a bit. Centroid 40 is the one
/// non-integer: its 67 coordinates are bf16-representable (8 significant bits), which keeps every
/// product (<= 12 bits) and every partial sum (< 2^10 over a 2^-16 grid) exact as well, in the f32
/// arms too, so nothing here carries a tolerance.
struct AssignFx {
    d: usize,
    x: Vec<f32>,
    c: Vec<f32>,
    rn: Vec<f32>,
    cn: Vec<f32>,
}

#[derive(Clone, Copy)]
struct AssignOpts {
    special: bool,
    sph_huge: bool,
    periodic: bool,
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

fn int_x(i: usize, t: usize) -> f32 {
    (((i * 7 + t * 13 + i * t) % 17) as i32 - 8) as f32
}

fn int_c(j: usize, t: usize) -> f32 {
    (((j * 5 + t * 11 + (j * t) % 7) % 17) as i32 - 8) as f32
}

fn assign_fx(m: usize, k: usize, d: usize, bf16: bool, opts: AssignOpts, rep: &mut Report, tag: &str) -> AssignFx {
    let mut x: Vec<f32> = (0..m * d).map(|n| int_x(n / d, n % d)).collect();
    let mut c: Vec<f32> = (0..k * d)
        .map(|n| {
            // Periodic centroids repeat every 7 clusters, so every row ties across tile columns,
            // lanes and simdgroups and the lower index has to win each merge.
            let j = if opts.periodic { (n / d) % 7 } else { n / d };
            int_c(j, n % d)
        })
        .collect();
    if k > 67 && !opts.periodic {
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
    let special = opts.special && k == 70 && m == 70;
    let mut invalid_row = None;
    if special {
        let s: Vec<f32> = random_f32(d, 8100 + d as u64).iter().map(|&v| eff(v, true)).collect();
        c[SELF_C * d..(SELF_C + 1) * d].copy_from_slice(&s);
        x[33 * d..34 * d].copy_from_slice(&s);
        // rn > 7.0e37 with a finite exact distance: row 20 and centroid 9 share a huge coordinate.
        let h = eff(9.0e18, bf16);
        x[20 * d] = h;
        c[HUGE_C * d] = h;
        // Every candidate overflows: all invalid.
        let g = eff(3.0e19, bf16);
        x[50 * d..51 * d].fill(g);
        invalid_row = Some(50);
        // |c|^2 = 8191 = 8192 - 1: a row on this centroid with rn = 8192 has raw = 1 = the refine
        // threshold exactly.
        c[EDGE_C * d..(EDGE_C + 1) * d].fill(0.0);
        c[EDGE_C * d..EDGE_C * d + 4].copy_from_slice(&[90.0, 9.0, 3.0, 1.0]);
        let edge = c[EDGE_C * d..(EDGE_C + 1) * d].to_vec();
        x[60 * d..61 * d].copy_from_slice(&edge);
        // 2 * dot overflows (rn <= 7.0e37) while the exact distance stays finite.
        c[OVF_C * d..(OVF_C + 1) * d].fill(0.0);
        c[OVF_C * d] = eff(2.5e19, bf16);
        x[55 * d..56 * d].fill(0.0);
        x[55 * d] = eff(8.0e18, bf16);
    }
    if opts.sph_huge {
        // -2 * dot overflows for every centroid whose coordinate 0 is not zero.
        x[60 * d..61 * d].fill(0.0);
        x[60 * d] = eff(3.0e38, bf16);
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
    let mut fx = AssignFx { d, x, c, rn, cn };
    if special {
        // Supplied row norms that sit exactly on the guards of eval_assign_score / finalize.
        fx.rn[11] = SAFE_RN_MAX; // rn == 7.0e37: still the fast path
        fx.rn[60] = 8192.0;
        // Rows on a centroid whose supplied norm puts raw at rho * threshold: inside the refine
        // window, just outside, between 1x and 2x, far outside, and the negative mirror images.
        for (row, rho) in [
            (3usize, 0.5f64),
            (9, 0.95),
            (15, 1.05),
            (21, 1.9),
            (27, 2.5),
            (39, -0.95),
            (45, -1.05),
        ] {
            let Some((_, s)) = fx.best(row, false, 0, k) else {
                rep.fail(format!("{tag}: ORACLE row {row} has no candidate"));
                continue;
            };
            if s > -100.0 {
                rep.fail(format!(
                    "{tag}: ORACLE row {row} score {s} is too small for the window rows"
                ));
                continue;
            }
            // raw = rn + s = rho * rn / 8192  =>  rn = -s / (1 - rho / 8192)
            let r = (-f64::from(s) / (1.0 - rho / 8192.0)) as f32;
            fx.rn[row] = r;
            let ratio = f64::from(r + s) / f64::from(1024.0f32 * 1.192_092_9e-7f32 * r);
            if (ratio - rho).abs() > 0.1 * rho.abs() {
                rep.fail(format!(
                    "{tag}: ORACLE row {row} realised raw/threshold {ratio}, wanted {rho}"
                ));
            }
        }
    }
    fx
}

impl AssignFx {
    fn xr(&self, i: usize) -> &[f32] {
        &self.x[i * self.d..(i + 1) * self.d]
    }

    fn cr(&self, j: usize) -> &[f32] {
        &self.c[j * self.d..(j + 1) * self.d]
    }

    /// `eval_assign_score`: the candidate's score, or `None` when it is rejected. The dot is the
    /// correctly rounded exact dot product: the fixtures keep every product and partial sum exact,
    /// or leave a single nonzero product, so the matrix unit's order cannot show.
    fn cand(&self, i: usize, j: usize, sph: bool) -> Option<f32> {
        let dot: f64 = self
            .xr(i)
            .iter()
            .zip(self.cr(j))
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        let dot = dot as f32;
        if sph {
            let sc = -2.0f32 * dot;
            return (!nonfinite32(sc.to_bits())).then_some(sc);
        }
        let rn = self.rn[i];
        if rn > SAFE_RN_MAX {
            return exact_dist2(self.xr(i), self.cr(j));
        }
        let two_dot = 2.0f32 * dot;
        let sc = self.cn[j] - two_dot;
        if nonfinite32(two_dot.to_bits()) || nonfinite32(sc.to_bits()) {
            return exact_dist2(self.xr(i), self.cr(j)).map(|e| e - rn);
        }
        Some(sc)
    }

    /// First strict minimum of the score over `c0..c1` (lower index wins ties), or `None`
    /// when every candidate is rejected.
    fn best(&self, i: usize, sph: bool, c0: usize, c1: usize) -> Option<(u32, f32)> {
        let mut best: Option<(u32, f32)> = None;
        for j in c0..c1 {
            if let Some(sc) = self.cand(i, j, sph) {
                if best.is_none_or(|(_, b)| sc < b) {
                    best = Some((j as u32, sc));
                }
            }
        }
        best
    }
}

struct AssignCase {
    name: &'static str,
    sph: bool,
    k: usize,
    split: usize,
    /// (part_c, c_start, c_end) of every dispatch.
    ranges: Vec<(usize, usize, usize)>,
    has_epoch: bool,
    opts: AssignOpts,
}

fn assign_cases() -> Vec<AssignCase> {
    let plain = AssignOpts {
        special: false,
        sph_huge: false,
        periodic: false,
    };
    let special = AssignOpts { special: true, ..plain };
    let periodic = AssignOpts {
        periodic: true,
        ..plain
    };
    let parts = vec![(0, 0, 64), (1, 64, 70)];
    let case = |name, sph, k, split, ranges: Vec<(usize, usize, usize)>, has_epoch, opts| AssignCase {
        name,
        sph,
        k,
        split,
        ranges,
        has_epoch,
        opts,
    };
    vec![
        case("full", false, 70, 1, vec![(0, 0, 70)], true, special),
        case("no-epoch", false, 70, 1, vec![(0, 0, 70)], false, plain),
        case(
            "spherical",
            true,
            70,
            1,
            vec![(0, 0, 70)],
            true,
            AssignOpts {
                sph_huge: true,
                ..plain
            },
        ),
        case("k1", false, 1, 1, vec![(0, 0, 1)], false, plain),
        case("split2", false, 70, 2, parts.clone(), false, special),
        case("split2-spherical", true, 70, 2, parts.clone(), false, plain),
        case("periodic", false, 70, 1, vec![(0, 0, 70)], false, periodic),
        case("periodic-spherical", true, 70, 1, vec![(0, 0, 70)], false, periodic),
        case("periodic-split2", false, 70, 2, parts, false, periodic),
        // A nonzero c_start with split_c == 1: the all-invalid row takes c_start as its label.
        case("c-window-3", false, 70, 1, vec![(0, 3, 70)], true, special),
        // An empty range writes nothing at all.
        case("empty-range", false, 70, 1, vec![(0, 5, 5)], true, plain),
        // Only the last centroid, as the second part of a split: 2 * dot overflows for row 55,
        // so its score is exact_d - rn, and every other row is rejected.
        case("only-ovf-centroid", false, 70, 2, vec![(1, 69, 70)], false, special),
    ]
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
    ledger.credit(rt, name, &[name], || {
        for case in &assign_cases() {
            let tag = format!(
                "{name}[{} M={m} k={} d={d} sph={} split_c={} ranges={:?} epoch={}]",
                case.name, case.k, case.sph, case.split, case.ranges, case.has_epoch
            );
            let fx = assign_fx(m, case.k, d, bf16, case.opts, rep, &tag);
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
            for &(part_c, c_start, c_end) in &case.ranges {
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
            for &(part_c, c_start, c_end) in &case.ranges {
                if c_start >= c_end {
                    continue;
                }
                for i in 0..m {
                    let best = fx.best(i, case.sph, c_start, c_end);
                    if case.split == 1 {
                        match best {
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
                                w_labels[i] = c_start as u32;
                                w_dists[i] = 0.0f32.to_bits();
                                w_fault |= FAULT_DIST;
                                if case.has_epoch {
                                    w_epoch[EPOCH_OFF + i] = c_start as u32;
                                }
                            }
                        }
                    } else {
                        let at = part_c * m + i;
                        match best {
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
            rep.cmp(&tag, "chunk_labels", &rd32(&chunk_labels, m + PAD), &w_labels);
            rep.cmp(&tag, "chunk_dists", &rd32(&chunk_dists, m + PAD), &w_dists);
            rep.cmp(
                &tag,
                "epoch_labels",
                &rd32(&epoch_labels, EPOCH_OFF + m + PAD),
                &w_epoch,
            );
            rep.cmp(&tag, "part_scores", &rd32(&part_scores, case.split * m + PAD), &w_ps);
            rep.cmp(&tag, "part_labels", &rd32(&part_labels, case.split * m + PAD), &w_pl);
            rep.cmp(&tag, "fault", &rd32(&fault, 1), &[w_fault]);
        }
    });
}

// ---------------------------------------------------------------------------
// reduce_split_c
// ---------------------------------------------------------------------------

/// The one fault cause a reduce_split_c case plants (or none).
#[derive(Clone, Copy, PartialEq)]
enum Bad {
    Nothing,
    /// A row whose parts are all INVALID.
    AllInvalid,
    /// A row with rn > 7.0e37 whose exact distance overflows.
    ExactNonfinite,
    /// A spherical row whose best score is -inf, so `2 + best` is not finite.
    SphNonfinite,
}

fn reduce_split_c(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    let name = "emeans_reduce_split_c";
    let (m, d, kc, split) = (40usize, 19usize, 12usize, 3usize);
    // Rows 20..=26 sit at raw = 127, 128, 129, 200, -127, -128, -129 against rn = 2^20, whose
    // refine threshold is 1024 * 2^-23 * 2^20 = 128 exactly.
    let edge_targets = [127.0f32, 128.0, 129.0, 200.0, -127.0, -128.0, -129.0];
    ledger.credit(rt, name, &[name], || {
        for bf16 in [false, true] {
            for (sph, has_epoch, bad) in [
                (false, true, Bad::Nothing),
                (false, false, Bad::Nothing),
                (true, true, Bad::Nothing),
                (false, true, Bad::AllInvalid),
                (false, true, Bad::ExactNonfinite),
                (true, true, Bad::SphNonfinite),
            ] {
                let tag = format!(
                    "{name}[bf16_assign={bf16} sph={sph} epoch={has_epoch} bad={}]",
                    match bad {
                        Bad::Nothing => "none",
                        Bad::AllInvalid => "all-invalid",
                        Bad::ExactNonfinite => "exact-nonfinite",
                        Bad::SphNonfinite => "sph-nonfinite",
                    }
                );
                let mut x: Vec<f32> = (0..m * d)
                    .map(|n| (((n / d * 5 + n % d * 3 + (n / d) * (n % d)) % 9) as i32 - 4) as f32)
                    .collect();
                let c: Vec<f32> = (0..kc * d)
                    .map(|n| (((n / d * 7 + n % d * 2 + (n / d) * (n % d) % 5) % 9) as i32 - 4) as f32)
                    .collect();
                if bad == Bad::ExactNonfinite {
                    x[7 * d..8 * d].fill(eff(3.0e19, bf16));
                }
                let mut rn: Vec<f32> = (0..m).map(|i| seq_norm(&x[i * d..(i + 1) * d])).collect();
                rn[5] = 9.0e37;
                rn[6] = 5.0e37;
                if bad == Bad::ExactNonfinite {
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
                // Rows with one valid part and every other part INVALID.
                let only = |i: usize, label: u32, score: f32, pl: &mut [u32], ps: &mut [f32]| {
                    for p in 0..split {
                        pl[p * m + i] = if p == 0 { label } else { INVALID };
                        ps[p * m + i] = if p == 0 { score } else { 0.0 };
                    }
                };
                let mut set = |i: usize, parts: [(u32, f32); 3]| {
                    for (p, (l, s)) in parts.into_iter().enumerate() {
                        pl[p * m + i] = l;
                        ps[p * m + i] = s;
                    }
                };
                set(1, [(7, -5.0), (3, -5.0), (INVALID, -9.0)]);
                set(2, [(3, -5.0), (7, -5.0), (5, -5.0)]);
                if bad == Bad::AllInvalid {
                    set(3, [(INVALID, -1.0), (INVALID, 2.0), (INVALID, -3.0)]);
                }
                set(4, [(4, 5.0), (2, -rn[4]), (9, 6.0)]);
                set(5, [(6, 1.0), (INVALID, 0.0), (INVALID, 0.0)]);
                set(6, [(1, 3.0e38), (INVALID, 0.0), (INVALID, 0.0)]);
                if bad == Bad::ExactNonfinite {
                    set(7, [(0, 1.0), (INVALID, 0.0), (INVALID, 0.0)]);
                }
                // raw = 2^-9 inside the refine band, and raw = -0.5 outside it (clamps to zero).
                set(8, [(2, 0.001_953_125 - rn[8]), (INVALID, 0.0), (INVALID, 0.0)]);
                set(9, [(5, -0.5 - rn[9]), (INVALID, 0.0), (INVALID, 0.0)]);
                if bad == Bad::SphNonfinite {
                    set(12, [(3, f32::NEG_INFINITY), (INVALID, 0.0), (INVALID, 0.0)]);
                }
                // The only valid part holds f32::MAX: the first part must be taken unconditionally.
                only(28, 5, f32::MAX, &mut pl, &mut ps);
                // rn = 0 and raw = 1e-35: inside the floored threshold (1e-30 * 2^-13), outside 0.
                rn[27] = 0.0;
                only(27, 1, 1.0e-35, &mut pl, &mut ps);
                for (e, &target) in edge_targets.iter().enumerate() {
                    let i = 20 + e;
                    rn[i] = 1_048_576.0;
                    let label = ((i * 5) % kc) as u32;
                    only(i, label, target - 1_048_576.0, &mut pl, &mut ps);
                    // The exact distance must differ from raw, or the window cannot be seen.
                    while exact_dist2(&x[i * d..(i + 1) * d], &c[label as usize * d..(label as usize + 1) * d])
                        == Some(target)
                    {
                        x[i * d] += 1.0;
                    }
                }
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
                if w_fault != if bad == Bad::Nothing { 0 } else { FAULT_DIST } {
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
                launch_threads(rt, name, m + 5, |bnd| {
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
        // M = 1: only thread 0 exists, so only `r == 0` can clear the overflow slice count.
        for m in [300usize, 1] {
            let mut labels: Vec<u32> = (0..m).map(|r| ((r * 37 + 5) % 70) as u32).collect();
            if m > 17 {
                labels[17] = 0xffff_fff0; // needs the 64-bit shift
            }
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
        }
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
        // 5000 keys: 20 tiles, so the multi-tile path (many threadgroups, last one partial) is
        // covered. It cannot force stale threadgroup memory, so it does not reliably kill a
        // histogram that is never cleared (rh_local_hist_not_cleared survived).
        for (count, shift) in [
            (600usize, 16u32),
            (600, 24),
            (600, 32),
            (257, 24),
            (100, 16),
            (5000, 24),
        ] {
            let tiles = count.div_ceil(256);
            let tag = format!("{name}[count={count} shift={shift}]");
            let keys = rand_keys(count, 9000 + u64::from(shift) + count as u64);
            let kb = u64b(rt, &keys);
            // One launched group past num_tiles writes hist[tid * tiles + tiles] if the guard fails.
            let len = 256 * tiles + 256;
            let hb = sent_u32(rt, len);
            launch(rt, name, (tiles + 1, 1, 1), (256, 1, 1), |bnd| {
                bnd.bind_gpu_buf(&kb, 0);
                bnd.bind_gpu_buf(&hb, 1);
                bnd.bind_u32(count as u32, 2);
                bnd.bind_u32(tiles as u32, 3);
                bnd.bind_u32(shift, 4);
            });
            let mut want = tile_hist(&keys, shift, tiles);
            want.extend(vec![SENT_U32; 256]);
            rep.cmp(&tag, "hist (+canary)", &rd32(&hb, len), &want);
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
            // The slot after the last key holds digit 255: a thread that wrongly counts as valid
            // (idx == count) would scatter it to keys_out[count], which must stay a sentinel.
            let mut padded = keys.clone();
            padded.push((0xffu64 << shift) | 0x1234);
            let (kb, hb) = (u64b(rt, &padded), u32b(rt, &offsets));
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
            rep.cmp(&tag, "keys_out (+canary)", &rd64(&out, count + 16), &want);
        }
    });
}

fn find_bucket_bounds(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    let name = "emeans_find_bucket_bounds";
    ledger.credit(rt, name, &[name], || {
        // (label, sizes, slice base before the kernel, final_epoch)
        type Case = (&'static str, Vec<usize>, u32, bool);
        let cases: [Case; 6] = [
            ("two-heavy", vec![300, 0, 270, 5, 25, 0], 0, true),
            ("two-heavy-base3", vec![300, 0, 270, 5, 25, 0], 3, false),
            ("k1-light", vec![100], 0, true),
            ("k1-heavy", vec![300], 0, false),
            // The 256-row threshold (255 and 256 are light, 257 is heavy), slice rounding at an
            // exact multiple (512 is 2 slices), a bucket of exactly 1 row, an empty one.
            ("threshold", vec![255, 256, 257, 512, 1, 0, 2], 1, false),
            ("threshold-final", vec![255, 256, 257, 512, 1, 0, 2], 0, true),
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
            let extra = 5usize; // launched threads past k: the `j >= k` guard must hold
            let info = sent_u32(rt, (k + extra) * 4);
            let meta_len = (b0 as usize + total + 4) * 2;
            let meta = sent_u32(rt, meta_len);
            let osc = u32b(rt, &[b0]);
            let (batch, epoch, pass) = (u64b(rt, &vec![7; k]), u64b(rt, &vec![11; k]), u64b(rt, &vec![13; k]));
            launch_threads(rt, name, k + extra, |bnd| {
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
            let got_info = rd32(&info, (k + extra) * 4);
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
            want_info.extend(vec![SENT_U32; extra * 4]);
            rep.cmp(&tag, "bucket_info (+canary)", &got_info, &want_info);
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
            let mut want_meta = vec![SENT_U32; meta_len];
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
            rep.cmp(&tag, "slice_meta", &rd32(&meta, meta_len), &want_meta);
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
        let (k, d, rows, n_keys) = (5usize, 67usize, 320usize, 200usize);
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
        let mut batch0: Vec<f32> = random_f32(k * d, 9811).iter().map(|v| v + 2.0).collect();
        let mut pass0: Vec<f32> = random_f32(k * d, 9812).iter().map(|v| v * 3.0 + 5.0).collect();
        // The empty bucket (start == end) must not be touched: -0.0 + 0.0 would flip the sign.
        batch0[2 * d..3 * d].fill(-0.0);
        pass0[2 * d..3 * d].fill(-0.0);
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
        // (label, dists, prior, planned fault). The expected objective is native f64 `+` in the
        // kernel's order: lane sums, tree 128..1, then the prior.
        let mut cases: Vec<(String, Vec<f32>, f64, u32)> = Vec::new();
        let scaled = |m: usize, seed: u64, signed: bool| -> Vec<f32> {
            random_f32(m, seed)
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let v = if signed { v } else { v.abs() };
                    v * 2.0f32.powi(((i * 7 + seed as usize) % 41) as i32 - 20)
                })
                .collect()
        };
        let mut planted = scaled(600, 9900, true);
        planted[100] = 1.0e-40; // f32 subnormal: only reinterpreted, so it stays exact
        planted[250] = f32::NAN;
        planted[400] = f32::INFINITY;
        cases.push(("planted".into(), planted, 1.5, FAULT_DIST));
        cases.push(("positive".into(), scaled(100, 9901, false), 0.0, 0));
        // +inf alone: it must raise the fault (and contribute nothing), with no NaN to do it.
        let mut inf = scaled(300, 9902, true);
        inf[77] = f32::INFINITY;
        cases.push(("inf-alone".into(), inf, 0.25, FAULT_DIST));
        // Negative values must be summed, not mistaken for non-finite: all of them, no NaN.
        let negative: Vec<f32> = scaled(300, 9903, false).iter().map(|v| -v).collect();
        cases.push(("all-negative".into(), negative, 0.0, 0));
        // Subnormal dists only, one per lane, so none is swamped by a larger term: every f32
        // subnormal is a multiple of 2^-149, so the f64 sums are exact.
        // 16 distinct mantissas (sign, mantissa): small ones need many normalisation shifts (0x1,
        // 0x3, 0x5), 0x7f_ffff is the top of the subnormal range, three are negative. No two are
        // opposites, and the sum is asserted non-zero below, so every input moves the result.
        let sub_bits: [(u32, u32); 16] = [
            (0, 0x7f_ffff),
            (0, 0x40_0001),
            (0, 0x00_0001),
            (0, 0x55_aaaa),
            (0, 0x2a_5555),
            (0, 0x7f_0000),
            (0, 0x01_0000),
            (0, 0x6d_b6db),
            (0, 0x00_0003),
            (0, 0x00_0102),
            (0, 0x13_579b),
            (0, 0x3c_0f0f),
            (1, 0x00_0005),
            (1, 0x24_6801),
            (1, 0x00_8421),
            (0, 0x71_1717),
        ];
        let sub: Vec<f32> = sub_bits.iter().map(|&(s, m)| f32::from_bits((s << 31) | m)).collect();
        let sub_sum: f64 = sub.iter().map(|&v| f64::from(v)).sum();
        if sub_sum == 0.0 {
            rep.fail(format!("{name}[subnormal-only]: ORACLE the fixture sums to zero"));
        }
        cases.push(("subnormal-only".into(), sub, 0.0, 0));
        // Exact cancellation inside a lane (dists[t] and dists[t + 256] are opposites), and near
        // cancellation by 2^-20 and 2^-22 that renormalises the difference.
        let mut cancel = scaled(512, 9904, true);
        for t in 0..32 {
            cancel[t + 256] = -cancel[t];
        }
        for t in 32..64 {
            cancel[t + 256] = -cancel[t] * (1.0 - 2.0f32.powi(-20));
        }
        for t in 64..96 {
            cancel[t + 256] = -cancel[t] * (1.0 - 2.0f32.powi(-22));
        }
        cases.push(("cancel".into(), cancel, 0.0, 0));
        // Many random mixed-sign draws: sticky bits, rounding ties and renormalisation of the
        // binary64 adder all need many different operand pairs to show.
        for s in 0..160u64 {
            let m = 256 + (s as usize * 13) % 400;
            cases.push((
                format!("random-{s}"),
                scaled(m, 10_300 + s, true),
                (s as f64) * 0.37 - 20.0,
                0,
            ));
        }
        for (label, dists, prior, planned) in &cases {
            let m = dists.len();
            let tag = format!("{name}[{label} M={m}]");
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
            let (db, ob, fault) = (
                f32b(rt, dists),
                u64b(rt, &[prior.to_bits(), SENT_U64]),
                fault_buf(rt, 0),
            );
            launch(rt, name, (1, 1, 1), (256, 1, 1), |bnd| {
                bnd.bind_gpu_buf(&db, 0);
                bnd.bind_gpu_buf(&ob, 1);
                bnd.bind_u32(m as u32, 2);
                bnd.bind_gpu_buf(&fault, 3);
            });
            rep.cmp(
                &tag,
                "epoch_obj_bits (+canary)",
                &rd64(&ob, 2),
                &[want.to_bits(), SENT_U64],
            );
            rep.cmp(&tag, "fault", &rd32(&fault, 1), &[*planned]);
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
        let rows = k + 1; // one launched group past k, over canary rows
        let counts0: [u64; 5] = [7, 0, 1000, 13, 5];
        let usage0: [f32; 5] = [5.0, 9.0, 0.0, 2.5, 77.0];
        let cent_base = random_f32(rows * d, 10_000);
        let sums_base: Vec<f32> = random_f32(rows * d, 10_001).iter().map(|v| v * 50.0).collect();
        for (alpha, preset) in [(1.0f32, 0u32), (0.25, 0), (0.0, 0), (1.5, 0), (0.25, 1)] {
            let tag = format!("{name}[alpha={alpha} fault_preset={preset}]");
            let mut cent0 = cent_base.clone();
            let mut sums0 = sums_base.clone();
            if alpha == 0.0 {
                // w == 0 keeps the old centroid bit for bit: a -0.0 stays -0.0 (adding +0.0 would
                // flip it) and an infinite batch sum must not turn it into NaN.
                for j in [0usize, 2, 3] {
                    for col in 0..d {
                        if col % 7 == 0 {
                            cent0[j * d + col] = -0.0;
                        }
                        if col % 11 == 0 {
                            sums0[j * d + col] = f32::INFINITY;
                        }
                    }
                }
            }
            let (cb, ub, sb) = (f32b(rt, &cent0), f32b(rt, &usage0), f32b(rt, &sums0));
            let (nb, fault) = (u64b(rt, &counts0), fault_buf(rt, preset));
            launch(rt, name, (rows, 1, 1), (256, 1, 1), |bnd| {
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
            let mut ws = sums0.clone();
            let mut wn = counts0.to_vec();
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
                ws[j * d..(j + 1) * d].fill(0.0);
                wn[j] = 0;
            }
            // Row k is a canary: counts 5, usage 77, sums and centroids untouched.
            rep.cmp(&tag, "centroids (+canary row)", &rd32(&cb, rows * d), &bits(&wc));
            rep.cmp(&tag, "usage (+canary)", &rd32(&ub, rows), &bits(&wu));
            rep.cmp(&tag, "batch_sums (+canary row)", &rd32(&sb, rows * d), &bits(&ws));
            rep.cmp(&tag, "batch_counts (+canary)", &rd64(&nb, rows), &wn);
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

/// How close a decoded Gumbel key must be to the host's f32 logarithms.
#[derive(Clone, Copy)]
enum KeyBound {
    /// Within 2 ordered ulps: these cases only have donors with g >= 0.3 (enforced by the ORACLE
    /// check below), so there is no cancellation in the log difference.
    Ulp2,
    /// Within 4e-6 absolute: small usages give negative and near-zero g, where the
    /// logarithms' unspecified accuracy (not their relative size) sets the error.
    Abs,
}

struct ReviveCase {
    name: &'static str,
    usage: Vec<f32>,
    threshold: f32,
    seed: u64,
    step: u64,
    preset: u32,
    bound: KeyBound,
}

/// Integer usages whose classification flips if the mean or the threshold is off by 0.3%: the
/// entries at `j % 20 == 3` sit 0.2% below the threshold, those at `j % 20 == 13` 0.2% above.
fn near_threshold_usage(k: usize, threshold: f32) -> Vec<f32> {
    let mut usage: Vec<f32> = (0..k).map(|j| 1000.0 + ((j * 37) % 1500) as f32).collect();
    for _ in 0..12 {
        let total = lane_sum(k, |j| usage[j]);
        let thr = threshold * (total / k as f32);
        for (j, u) in usage.iter_mut().enumerate() {
            match j % 20 {
                3 => *u = (thr * 0.998).floor(),
                13 => *u = (thr * 1.002).ceil(),
                _ => {}
            }
        }
    }
    usage
}

fn revive_cases() -> Vec<ReviveCase> {
    let strong = |j: usize| -> f32 {
        match j % 5 {
            0 => 0.0,
            1 => 3.0,
            _ => 30.0 + ((j * 37) % 400) as f32,
        }
    };
    let mut cases = vec![
        ReviveCase {
            name: "k6",
            usage: (0..6).map(strong).collect(),
            threshold: 0.1,
            seed: 0x1234_5678_9abc_def0,
            step: 3,
            preset: 0,
            bound: KeyBound::Ulp2,
        },
        ReviveCase {
            name: "k300",
            usage: (0..300).map(strong).collect(),
            threshold: 0.1,
            seed: 0x0fed_cba9_8765_4321,
            step: 17,
            preset: 0,
            bound: KeyBound::Ulp2,
        },
        ReviveCase {
            name: "k300-fault",
            usage: (0..300).map(strong).collect(),
            threshold: 0.1,
            seed: 0x0fed_cba9_8765_4321,
            step: 17,
            preset: 1,
            bound: KeyBound::Ulp2,
        },
        // More unused clusters than donors: nsplit = n_donors, and meta.y adds nsplit, not n_unused.
        ReviveCase {
            name: "unused>donors",
            usage: vec![0.0, 0.0, 0.0, 0.0, 50.0, 60.0],
            threshold: 0.1,
            seed: 0x55aa_55aa_1234_4321,
            step: 9,
            preset: 0,
            bound: KeyBound::Ulp2,
        },
    ];
    cases.push(ReviveCase {
        name: "near-threshold",
        usage: near_threshold_usage(300, 0.5),
        threshold: 0.5,
        seed: 0x1111_2222_3333_4444,
        step: 5,
        preset: 0,
        bound: KeyBound::Ulp2,
    });
    // usage == threshold exactly: entry 7 equals 0.25 * mean, so it is a donor (nj < thresh is
    // false). The other 299 entries sum to 40 * (4 * 300 - 1), which makes mean 160 and thresh 40.
    let mut at_thresh: Vec<f32> = (0..300)
        .map(|j| {
            if j % 4 == 0 {
                5.0
            } else {
                120.0 + ((j * 37) % 80) as f32
            }
        })
        .collect();
    at_thresh[7] = 40.0;
    let others: f32 = at_thresh
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != 7)
        .map(|(_, &u)| u)
        .sum();
    at_thresh[299] += 40.0 * 1199.0 - others;
    cases.push(ReviveCase {
        name: "usage-at-threshold",
        usage: at_thresh,
        threshold: 0.25,
        seed: 0x7777_8888_9999_aaaa,
        step: 11,
        preset: 0,
        bound: KeyBound::Ulp2,
    });
    // threshold 0: unused are the negative usages; zeros are neither unused nor donors.
    cases.push(ReviveCase {
        name: "threshold-0",
        usage: vec![-5.0, -3.0, -1.0, -2.0, -4.0, -6.0, 0.0, 0.0, 0.0, 0.0, 7.0, 9.0],
        threshold: 0.0,
        seed: 0xdead_beef_0bad_f00d,
        step: 2,
        preset: 0,
        bound: KeyBound::Abs,
    });
    // Small usages: roughly a quarter of the donors get a negative Gumbel key.
    cases.push(ReviveCase {
        name: "small-usage-negative-keys",
        usage: (0..96).map(|j| 1.0 + (j % 6) as f32).collect(),
        threshold: 0.01,
        seed: 0x0123_4567_89ab_cdef,
        step: 4,
        preset: 0,
        bound: KeyBound::Abs,
    });
    cases
}

fn revive_prepare(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, rep: &mut Report) {
    let name = "emeans_revive_prepare";
    ledger.credit(rt, name, &[name], || {
        for case in &revive_cases() {
            let (usage, threshold, seed, step, preset) =
                (&case.usage, case.threshold, case.seed, case.step, case.preset);
            let k = usage.len();
            let tag = format!("{name}[{} k={k} step={step} fault_preset={preset}]", case.name);
            let total = lane_sum(k, |j| usage[j]);
            let thresh = threshold * (total / k as f32);
            let is_donor = |j: usize| {
                let below = usage[j] < thresh;
                !below && usage[j] > 0.0
            };
            let unused: Vec<u32> = (0..k).filter(|&j| usage[j] < thresh).map(|j| j as u32).collect();
            let n_donors = (0..k).filter(|&j| is_donor(j)).count();
            let nsplit = unused.len().min(n_donors) as u32;
            let (ub, kb) = (f32b(rt, usage), sent_u64(rt, k));
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
            // (host g, gpu high word)
            let mut donors: Vec<(f32, u32)> = Vec::new();
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
                // The high word is the bitwise complement of the ordered key of g.
                let ordered = !high;
                let gpu_g = f32::from_bits(if ordered & 0x8000_0000 != 0 {
                    ordered ^ 0x8000_0000
                } else {
                    !ordered
                });
                let ok = match case.bound {
                    KeyBound::Ulp2 => {
                        if g < 0.3 {
                            rep.fail(format!("{tag}: ORACLE donor {j} has g {g}, cancellation not excluded"));
                        }
                        // precise::log is not pinned to an ulp bound here: the decoded key stays
                        // within 2 ordered ulps of the host f32 logarithms.
                        (i64::from(ordered) - i64::from(ordered_u32(g))).abs() <= 2
                    }
                    // precise::log carries no stated bound; small usages put g near zero, where the
                    // error of the two logarithms (each up to a few ulp of values up to ~17) is
                    // absolute, so the bound is absolute: 4e-6, far below a wrong branch or a
                    // wrong sign, far above the observed 2-ulp agreement for the g >= 0.3 cases.
                    KeyBound::Abs => (gpu_g - g).abs() <= 4.0e-6,
                };
                if !ok {
                    rep.fail(format!(
                        "{tag}: donor {j} key decodes to {gpu_g}, the host f32 logarithms give {g}"
                    ));
                }
                donors.push((g, high));
            }
            // The donor rank order (ascending key = descending g) must agree wherever the host
            // values differ by more than twice the bound.
            let margin = 8.0e-6f32;
            for a in 0..donors.len() {
                for b in a + 1..donors.len() {
                    if (donors[a].0 - donors[b].0).abs() > margin
                        && (donors[a].1.cmp(&donors[b].1)) != donors[b].0.total_cmp(&donors[a].0)
                    {
                        rep.fail(format!("{tag}: donor rank order disagrees between donors {a} and {b}"));
                    }
                }
            }
        }
    });
}

/// One revive_splits dispatch: `pairs` are (donor key, target) in pair order, `nsplit` of them
/// active; the centroid and usage buffers carry two canary rows past `k`.
struct SplitCase {
    label: String,
    k: usize,
    d: usize,
    cent: Vec<f32>,
    usage: Vec<f32>,
    dkeys: Vec<u64>,
    unused: Vec<u32>,
    nsplit: u32,
    eps: f32,
    bf16: bool,
}

fn run_splits(rt: &Arc<GpuRuntime>, rep: &mut Report, case: &SplitCase) {
    let name = "emeans_revive_splits";
    let (k, d) = (case.k, case.d);
    let tag = format!("{name}[{} bf16_repr={} eps={}]", case.label, case.bf16, case.eps);
    let mut cent = case.cent.clone();
    cent.extend(random_f32(2 * d, 10_700)); // canary rows k and k + 1
    let mut usage = case.usage.clone();
    usage.extend([77.0, 78.0]);
    let (kb, ib, mb) = (
        u64b(rt, &case.dkeys),
        u32b(rt, &case.unused),
        u32b(rt, &[case.nsplit, 0]),
    );
    let (cb, ub) = (f32b(rt, &cent), f32b(rt, &usage));
    launch(rt, name, (k, 1, 1), (256, 1, 1), |bnd| {
        bnd.bind_gpu_buf(&kb, 0);
        bnd.bind_gpu_buf(&ib, 1);
        bnd.bind_gpu_buf(&mb, 2);
        bnd.bind_gpu_buf(&cb, 3);
        bnd.bind_gpu_buf(&ub, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(d as u32, 6);
        bnd.bind_f32(case.eps, 7);
        bnd.bind_u32(u32::from(case.bf16), 8);
    });
    let (mut wc, mut wu) = (cent.clone(), usage.clone());
    for pair in 0..case.nsplit as usize {
        let dkey = case.dkeys[pair];
        if (dkey >> 32) as u32 == 0xffff_ffff {
            continue;
        }
        let donor = (dkey & 0xffff_ffff) as usize;
        let target = case.unused[pair] as usize;
        if donor >= k || target >= k || donor == target {
            continue;
        }
        let mut any_diff = false;
        let mut child0 = 0.0f32;
        for col in 0..d {
            let c = cent[donor * d + col];
            let off = if col % 2 == 0 { case.eps } else { -case.eps };
            let donor_c = c * (1.0f32 - off);
            let child_c = c * (1.0f32 + off);
            wc[donor * d + col] = donor_c;
            wc[target * d + col] = child_c;
            if col == 0 {
                child0 = child_c;
            }
            if case.bf16 {
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
            child0 = nudge_child0(child0, case.bf16);
            wc[target * d] = child0;
        }
        let half = usage[donor] * 0.5;
        wu[donor] = half;
        wu[target] = half;
    }
    rep.cmp(&tag, "centroids (+canary rows)", &rd32(&cb, (k + 2) * d), &bits(&wc));
    rep.cmp(&tag, "usage (+canary)", &rd32(&ub, k + 2), &bits(&wu));
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
        // Pairs 0..=3 are ordinary, 4 is a sentinel key, 5 has donor == target, 6 a donor past k,
        // 7 a target past k; pairs 8 and 9 lie past nsplit and stay untouched.
        let donors = [0u64, 1, 2, 3, 4, 5, 16, 6, 7, 13];
        let mut dkeys: Vec<u64> = donors
            .iter()
            .enumerate()
            .map(|(p, &dn)| ((0x1234_0000 + p as u64) << 32) | dn)
            .collect();
        dkeys[4] = (0xffff_ffffu64 << 32) | 4;
        let unused: Vec<u32> = vec![8, 9, 10, 11, 12, 5, 14, 17, 15, 14];
        for (bf16, eps) in [(true, 1.0e-4f32), (true, 0.02), (false, 1.0e-8), (false, 1.0e-3)] {
            run_splits(
                rt,
                rep,
                &SplitCase {
                    label: "pairs".into(),
                    k,
                    d,
                    cent: cent0.clone(),
                    usage: usage0.clone(),
                    dkeys: dkeys.clone(),
                    unused: unused.clone(),
                    nsplit: 8,
                    eps,
                    bf16,
                },
            );
        }
        // Donors at the edges of the representable range, whose collapse nudge steps over a
        // boundary: bf16 {smallest normal, -smallest normal, largest finite} with eps 1e-4,
        // f32 {largest finite, -smallest normal} with eps 1e-8 (1 +- 1e-8 rounds to 1).
        let value_rows: [(bool, f32, [f32; 3]); 2] = [
            (
                true,
                1.0e-4,
                [
                    1.000_5 * f32::from_bits(0x0080_0000),
                    -1.000_5 * f32::from_bits(0x0080_0000),
                    f32::from_bits(0x7f7f_0000),
                ],
            ),
            (
                false,
                1.0e-8,
                [f32::MAX, f32::from_bits(0x8080_0000), f32::from_bits(0x0080_0000)],
            ),
        ];
        for (bf16, eps, vals) in value_rows {
            let (kk, dd) = (6usize, 40usize);
            let mut cent = random_f32(kk * dd, 10_200);
            for (r, v) in vals.iter().enumerate() {
                cent[r * dd..(r + 1) * dd].fill(*v);
            }
            run_splits(
                rt,
                rep,
                &SplitCase {
                    label: "edge-values".into(),
                    k: kk,
                    d: dd,
                    cent,
                    usage: (0..kk).map(|j| 2.0 + j as f32).collect(),
                    dkeys: (0..3u64).map(|p| ((0x2000 + p) << 32) | p).collect(),
                    unused: vec![3, 4, 5],
                    nsplit: 3,
                    eps,
                    bf16,
                },
            );
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
