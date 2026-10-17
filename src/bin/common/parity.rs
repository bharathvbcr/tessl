//! The numeric-parity scorer: every lane of a `bench_gemm_sweep --dump-parity`
//! directory, on every seed, against one float64 reference per seed, next to the
//! comparison runtimes (MLX, torch-MPS) scored on the same operands.
//!
//! This is the Rust owner of what `bench/gemm_sweep_mlx.py`'s `parity`,
//! `adjudicate`, `_score`, `_load_manifest` and `_check_external_lanes` did, and
//! of `bench/parity_ladder.py`'s `merge`. MLX and torch stay in Python: that
//! script runs the comparison matmuls and writes their outputs next to the dump
//! (`external_lanes.json` + `seed_XX/parity_c_<lane>.npy`), and this module reads
//! them back and scores them through the same code as the tessl lanes.
//!
//! The bar every check here defends is one: a check that could not run must
//! never report the same result as a check that ran and passed. Every
//! structural problem is an `Err`; the only numbers that come out are for lanes
//! that were scored in full.
//!
//! Included with `#[path]` by `gemm_parity` and by `tests/gemm_parity.rs`, next
//! to `crate::json` (the strict parser) and `crate::pyjson` (the writer).

#![allow(dead_code)] // The binary and the tests each use a subset.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::json::{self, Json, NumForm, Syntax};
use crate::pyjson::{float_str, fmt_e, median, obj, quote, quote_list, Doc};

// ------------------------------------------------------------------ precision

/// Unit roundoff of f32, which every lane accumulates in.
pub const U_F32: f64 = 1.0 / 16_777_216.0; // 2^-24
/// Unit roundoff of bf16 operands (8 explicit-or-implicit significand bits), also
/// a safe upper bound for tf32-class formats (2^-11), which is why the relaxed
/// path shares it.
pub const U_BF16: f64 = 1.0 / 256.0; // 2^-8

/// `lane -> (operand_u, out_u)`.
///
/// `operand_u` is nonzero when the kernel reads operands narrower than the
/// reference does. This differs from the Rust unit tests on purpose: those
/// round their operands to bf16 *before* upload, so the GPU sees exactly the
/// reference's values and operand width contributes nothing. This scorer reads
/// the f32 originals as the reference, so the bf16 lane really does narrow.
///
/// `out_u` is nonzero only for lanes that return bf16 (MLX and torch do; tessl
/// accumulates and returns f32), which is a rounding of the result itself.
///
/// MLX's GPU f32 matmul is tf32-class, measured on M5 Pro (MLX 0.32.3): worst
/// abs error 3.5e-3 at K=16 against 7e-7 on its CPU stream, on operands an f32
/// kernel scores at ~1e-7. So its lane takes the same operand class as
/// `tensorops-tf32` rather than the f32-exact `(0, 0)` it was given first; under
/// that budget every valid dump aborted wherever `mlx` was importable. The lane
/// keeps its label (the dtype it is asked for is f32); only the budget reflects
/// what it does. `torch-mps-f32` is unchanged and still f32-exact.
pub const LANE_PRECISION: &[(&str, (f64, f64))] = &[
    ("tensorops-f32", (0.0, 0.0)),
    ("simdgroup-f32", (0.0, 0.0)),
    ("tensorops-tf32", (U_BF16, 0.0)),
    ("tensorops-bf16", (U_BF16, 0.0)),
    ("mlx-f32", (U_BF16, 0.0)),
    ("mlx-bf16", (U_BF16, U_BF16)),
    ("torch-mps-f32", (0.0, 0.0)),
    ("torch-mps-bf16", (U_BF16, U_BF16)),
];

/// The table a scoring run consults. Passed in rather than read from the
/// constant so a test can hand a lane a budget other than the shipped one (the
/// check that MLX's tf32-class error is a breach under an f32-exact claim).
#[derive(Clone, Debug)]
pub struct Precision(pub Vec<(String, (f64, f64))>);

impl Precision {
    pub fn standard() -> Self {
        Precision(LANE_PRECISION.iter().map(|(l, p)| (l.to_string(), *p)).collect())
    }

    /// The table with `lane`'s entry replaced (or added).
    pub fn with(mut self, lane: &str, pair: (f64, f64)) -> Self {
        self.0.retain(|(l, _)| l != lane);
        self.0.push((lane.to_string(), pair));
        self
    }

    pub fn get(&self, lane: &str) -> Option<(f64, f64)> {
        self.0.iter().find(|(l, _)| l == lane).map(|(_, p)| *p)
    }

    pub fn known(&self) -> Vec<String> {
        let mut v: Vec<String> = self.0.iter().map(|(l, _)| l.clone()).collect();
        v.sort();
        v
    }
}

/// `gamma_n = n*u / (1 - n*u)`, the classical dot-product bound.
pub fn gamma(n: usize, u: f64) -> Result<f64, String> {
    let nu = n as f64 * u;
    if nu >= 1.0 {
        return Err(format!("error bound degenerate at K={n}: n*u = {} >= 1", float_str(nu)));
    }
    Ok(nu / (1.0 - nu))
}

// ------------------------------------------------------------------- manifest

/// The dump manifest: written last and only on success by `bench_gemm_sweep`,
/// so its absence means the dump was interrupted or never ran.
#[derive(Clone, Debug)]
pub struct Manifest {
    pub shape: Option<String>,
    pub dist: Option<String>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lanes: Vec<String>,
    pub seeds: Vec<String>,
}

const JSON_SYNTAX: Syntax = Syntax {
    what: "JSON",
    max_depth: 8,
    uints_only: false,
    literals: true,
};

/// A name that becomes a path component. A manifest is data from a directory
/// the caller chose; a lane or seed called `../x` must not walk out of it.
fn path_component(what: &str, name: &str) -> Result<(), String> {
    if name.is_empty() || name == "." || name == ".." || name.contains(['/', '\\', '\0']) {
        return Err(format!(
            "parity_manifest.json: {what} {} is not a plain name; refusing to build a path from it",
            quote(name)
        ));
    }
    Ok(())
}

fn string_list(man: &Json, key: &str) -> Result<Vec<String>, String> {
    let Some(Json::Array(items)) = man.get(key) else {
        return Err(format!("parity_manifest.json: '{key}' must be a non-empty list"));
    };
    if items.is_empty() {
        return Err(format!("parity_manifest.json: '{key}' must be a non-empty list"));
    }
    let mut out = Vec::new();
    for item in items {
        let Json::Str(s) = item else {
            return Err(format!("parity_manifest.json: '{key}' must contain only strings"));
        };
        out.push(s.clone());
    }
    Ok(out)
}

/// Read and structurally validate `dir/parity_manifest.json`.
pub fn load_manifest(dir: &Path) -> Result<Manifest, String> {
    let path = dir.join("parity_manifest.json");
    if !path.exists() {
        return Err(format!(
            "no parity_manifest.json in {}. It is written last and only on a clean run, so \
             this dump is absent or partial. Re-run `bench_gemm_sweep --dump-parity`.",
            dir.display()
        ));
    }
    let text = std::fs::read_to_string(&path).map_err(|e| format!("parity_manifest.json is unreadable: {e}"))?;
    let man = json::parse(&text, JSON_SYNTAX).map_err(|e| format!("parity_manifest.json is unreadable: {e}"))?;
    if !matches!(man, Json::Object(_)) {
        return Err("parity_manifest.json must be an object".to_string());
    }
    let lanes = string_list(&man, "lanes")?;
    let seeds = string_list(&man, "seeds")?;
    let mut dims = [0usize; 3];
    for (slot, key) in dims.iter_mut().zip(["m", "n", "k"]) {
        match man.get(key) {
            Some(Json::Num {
                int: v,
                form: NumForm::Uint,
                ..
            }) if *v > 0 && *v <= usize::MAX as u64 => *slot = *v as usize,
            _ => return Err(format!("parity_manifest.json: '{key}' must be a positive integer")),
        }
    }
    for lane in &lanes {
        path_component("lane", lane)?;
    }
    for (i, lane) in lanes.iter().enumerate() {
        if lanes[..i].contains(lane) {
            return Err(format!(
                "parity_manifest.json: lane {} is listed more than once; it would be scored as two",
                quote(lane)
            ));
        }
    }
    for seed in &seeds {
        path_component("seed", seed)?;
    }
    let text_of = |key: &str| match man.get(key) {
        Some(Json::Str(s)) => Some(s.clone()),
        _ => None,
    };
    Ok(Manifest {
        shape: text_of("shape"),
        dist: text_of("dist"),
        m: dims[0],
        n: dims[1],
        k: dims[2],
        lanes,
        seeds,
    })
}

// ------------------------------------------------------------- external lanes

/// What one comparison runtime produced on one seed.
#[derive(Clone, Debug, PartialEq)]
pub enum ExternalSeed {
    /// An output was written to `seed_XX/parity_c_<lane>.npy`; this is the
    /// dtype the runtime returned it in before it was widened to f32.
    Scored { out_dtype: String },
    /// The runtime could not run on this seed; the reason is surfaced.
    Skipped { reason: String },
}

#[derive(Clone, Debug)]
pub struct ExternalLane {
    pub lane: String,
    /// The operand dtype the runtime was asked for.
    pub dtype: String,
    pub seeds: Vec<(String, ExternalSeed)>,
}

/// Read `dir/external_lanes.json`, written by `bench/gemm_sweep_mlx.py
/// --external-lanes DIR`, and validate it before any scoring: a name outside the
/// precision table would be scored against a budget nobody chose for it, and a
/// duplicate or a collision with a dumped lane would merge two different
/// runtimes into one row. A missing file is an empty list (the caller says so).
pub fn load_externals(dir: &Path, man: &Manifest, precision: &Precision) -> Result<Vec<ExternalLane>, String> {
    let path = dir.join("external_lanes.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(&path).map_err(|e| format!("external_lanes.json is unreadable: {e}"))?;
    let doc = json::parse(&text, JSON_SYNTAX).map_err(|e| format!("external_lanes.json is unreadable: {e}"))?;
    let Some(Json::Array(entries)) = doc.get("lanes") else {
        return Err("external_lanes.json: 'lanes' must be a list of {lane, dtype, seeds} objects".to_string());
    };
    let mut out: Vec<ExternalLane> = Vec::new();
    for (i, ent) in entries.iter().enumerate() {
        let Json::Object(_) = ent else {
            return Err(format!(
                "external_lanes[{i}] must be a {{lane, dtype, seeds}} object, got {}",
                kind(ent)
            ));
        };
        let lane = match ent.get("lane") {
            Some(Json::Str(s)) => s.clone(),
            _ => return Err(format!("external_lanes[{i}]: 'lane' must be a string")),
        };
        if precision.get(&lane).is_none() {
            return Err(format!(
                "external_lanes[{i}]: lane {} has no entry in LANE_PRECISION; known: {}",
                quote(&lane),
                quote_list(&precision.known())
            ));
        }
        let dtype = match ent.get("dtype") {
            Some(Json::Str(s)) if s == "f32" || s == "bf16" => s.clone(),
            other => {
                let shown = match other {
                    Some(Json::Str(s)) => quote(s),
                    Some(v) => kind(v).to_string(),
                    None => "missing".to_string(),
                };
                return Err(format!(
                    "external_lanes[{i}] ({lane}): dtype {shown} must be 'f32' or 'bf16'"
                ));
            }
        };
        if out.iter().any(|e| e.lane == lane) {
            return Err(format!("external_lanes: lane {} listed more than once", quote(&lane)));
        }
        if man.lanes.contains(&lane) {
            return Err(format!(
                "external_lanes: lane {} collides with a lane in the dump; two runtimes would share one row",
                quote(&lane)
            ));
        }
        path_component("external lane", &lane)?;
        let Some(Json::Array(per_seed)) = ent.get("seeds") else {
            return Err(format!("external_lanes[{i}] ({lane}): 'seeds' must be a list"));
        };
        let mut seeds: Vec<(String, ExternalSeed)> = Vec::new();
        for s in per_seed {
            let name = match s.get("seed") {
                Some(Json::Str(n)) => n.clone(),
                _ => return Err(format!("external_lanes[{i}] ({lane}): a seed entry has no 'seed' name")),
            };
            if seeds.iter().any(|(n, _)| *n == name) {
                return Err(format!(
                    "external lane {} lists seed {} twice",
                    quote(&lane),
                    quote(&name)
                ));
            }
            let result = match (s.get("out_dtype"), s.get("skipped")) {
                (Some(Json::Str(d)), None) if d == "f32" || d == "bf16" => {
                    ExternalSeed::Scored { out_dtype: d.clone() }
                }
                (None, Some(Json::Str(reason))) => ExternalSeed::Skipped { reason: reason.clone() },
                _ => {
                    return Err(format!(
                    "external lane {} seed {}: needs exactly one of out_dtype ('f32' or 'bf16') or skipped (a reason)",
                    quote(&lane),
                    quote(&name)
                ))
                }
            };
            seeds.push((name, result));
        }
        // Every manifest seed must be accounted for, scored or skipped: a lane
        // the runtime never reported on is not the same as one it skipped.
        for want in &man.seeds {
            if !seeds.iter().any(|(n, _)| n == want) {
                return Err(format!(
                    "external lane {} has no result for seed {}; the writer did not finish",
                    quote(&lane),
                    quote(want)
                ));
            }
        }
        out.push(ExternalLane { lane, dtype, seeds });
    }
    Ok(out)
}

fn kind(v: &Json) -> &'static str {
    match v {
        Json::Object(_) => "object",
        Json::Array(_) => "list",
        Json::Str(_) => "str",
        Json::Num { .. } => "number",
        Json::Bool(_) => "bool",
        Json::Null => "null",
    }
}

// ----------------------------------------------------------------- reference

/// `C = A @ B` and `S = |A| @ |B|` in float64, row-major, for `A` m x k and `B`
/// k x n.
///
/// `S` is the conditioning of each output element's sum and the numerator of
/// the per-element budget. Each element accumulates its k terms in ascending
/// order with one fused multiply-add per term, whatever the thread count, so the
/// result is deterministic and a run is reproducible. It is not numpy's
/// Accelerate dgemm, whose blocking is not reproducible from here; the two agree
/// to a few units in the last place of float64, nine decades below the one
/// threshold (budget ratio 1.0) any verdict turns on.
pub fn reference_gemm(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![0.0f64; m * n];
    let mut s = vec![0.0f64; m * n];
    if m == 0 || n == 0 || k == 0 {
        return (c, s);
    }
    const ROWS: usize = 4;
    const NB: usize = 512;
    const KB: usize = 256;
    /// The first row of a band, and the band's rows of `C` and `S`.
    type Band<'a> = (usize, (&'a mut [f64], &'a mut [f64]));
    let bands: Vec<Band<'_>> = c
        .chunks_mut(ROWS * n)
        .zip(s.chunks_mut(ROWS * n))
        .enumerate()
        .map(|(i, cs)| (i * ROWS, cs))
        .collect();
    let queue = Mutex::new(bands.into_iter());
    let workers = std::thread::available_parallelism()
        .map_or(1, |p| p.get())
        .min(m.div_ceil(ROWS));
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let next = queue.lock().unwrap_or_else(|p| p.into_inner()).next();
                let Some((row0, (cb, sb))) = next else { break };
                let rows = cb.len() / n;
                for jb in (0..n).step_by(NB) {
                    let jn = NB.min(n - jb);
                    for kb in (0..k).step_by(KB) {
                        for kk in kb..(kb + KB).min(k) {
                            let brow = &b[kk * n + jb..kk * n + jb + jn];
                            for r in 0..rows {
                                let av = a[(row0 + r) * k + kk];
                                let aa = av.abs();
                                let cr = &mut cb[r * n + jb..r * n + jb + jn];
                                let sr = &mut sb[r * n + jb..r * n + jb + jn];
                                for j in 0..jn {
                                    let bv = brow[j];
                                    cr[j] = av.mul_add(bv, cr[j]);
                                    sr[j] = aa.mul_add(bv.abs(), sr[j]);
                                }
                            }
                        }
                    }
                }
            });
        }
    });
    (c, s)
}

/// numpy's pairwise summation (`DOUBLE_pairwise_sum`), which `ndarray.sum` and
/// `ndarray.mean` use over a contiguous float64 array. Blocks of eight lanes up
/// to 128 elements, halves (the left rounded down to a multiple of eight) above.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, rj) in r.iter_mut().enumerate() {
                *rj += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `ndarray.mean()` of a contiguous float64 array.
pub fn np_mean(a: &[f64]) -> f64 {
    pairwise_sum(a) / a.len() as f64
}

/// `math.fsum`: the correctly rounded sum (Shewchuk), which `statistics.fmean`
/// divides by the count.
pub fn fsum(xs: &[f64]) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for &x0 in xs {
        let mut x = x0;
        let mut i = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        partials.push(x);
    }
    let mut n = partials.len();
    let mut hi = 0.0;
    if n > 0 {
        n -= 1;
        hi = partials[n];
        let mut lo = 0.0;
        while n > 0 {
            let x = hi;
            n -= 1;
            let y = partials[n];
            hi = x + y;
            let yr = hi - x;
            lo = y - yr;
            if lo != 0.0 {
                break;
            }
        }
        // Half-even rounding across several partials.
        if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
            let y = lo * 2.0;
            let x = hi + y;
            let yr = x - hi;
            if y == yr {
                hi = x;
            }
        }
    }
    hi
}

// -------------------------------------------------------------------- scoring

/// One lane, one seed: the five numbers `_score` returned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score {
    pub abs: f64,
    pub rel: f64,
    pub mean: f64,
    pub elem: f64,
    pub higham: f64,
}

/// The float64 reference of one seed.
pub struct Reference {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub c: Vec<f64>,
    pub mag: Vec<f64>,
    pub scale: f64,
}

/// One lane, one seed, three metrics, and a bound rather than just numbers.
///
/// `rel` (max|err| / max|ref|) is what this harness used to report alone. It
/// cannot see a per-element failure: an output whose true value is near zero from
/// cancellation can be wrong by orders of magnitude in relative terms and still
/// move max|err| almost not at all. Measured on a plain uniform draw, a bf16
/// GEMM's worst element-wise relative error is ~6e+2 while its normwise error is
/// ~3e-3: five orders apart, both correct, describing different things.
///
/// `higham` is the one that decides pass or fail:
/// `err / ((gamma_{K+8} + 2*u_in) * sum|a.b| + u_out*|ref|)`, the same
/// per-element budget `tests/common/mod.rs` asserts against. Above 1.0 the
/// kernel is genuinely out of tolerance, and that is fatal rather than reported.
pub fn score(
    lane: &str,
    c: &[f64],
    c_shape: &[usize],
    r: &Reference,
    where_: &str,
    precision: &Precision,
) -> Result<Score, String> {
    if c_shape != [r.m, r.n] {
        return Err(format!(
            "{where_} lane {lane}: shape {} != reference {}",
            shape_str(c_shape),
            shape_str(&[r.m, r.n])
        ));
    }
    let bad = c.iter().filter(|v| !v.is_finite()).count();
    if bad > 0 {
        return Err(format!(
            "{where_} lane {lane}: {bad} non-finite element(s). That is a broken kernel, not an \
             accuracy number -- refusing to score it."
        ));
    }
    let Some((operand_u, out_u)) = precision.get(lane) else {
        // A permissive default for an unrecognised lane would let a new kernel
        // pass a budget nobody chose for it.
        return Err(format!(
            "{where_}: lane {} has no entry in LANE_PRECISION; add its operand/output unit \
             roundoff rather than scoring it against a budget picked by default. Known: {}",
            quote(lane),
            quote_list(&precision.known())
        ));
    };
    let g = gamma(r.k + 8, U_F32)?;
    let bound = g + 2.0 * operand_u;
    let live_floor = r.scale * 1e-9;
    let mut err_all = Vec::with_capacity(c.len());
    let (mut err_max, mut elem_rel, mut worst) = (0.0f64, 0.0f64, f64::NEG_INFINITY);
    for ((&cv, &rv), &mv) in c.iter().zip(&r.c).zip(&r.mag) {
        let err = (cv - rv).abs();
        err_all.push(err);
        err_max = err_max.max(err);
        let ref_abs = rv.abs();
        if ref_abs > live_floor {
            elem_rel = elem_rel.max(err / ref_abs);
        }
        let budget = bound * mv + out_u * ref_abs;
        // An exactly-zero budget means every product was exactly zero, so the
        // only acceptable output is exactly zero.
        let ratio = if budget > 0.0 {
            err / budget.max(f64::MIN_POSITIVE)
        } else if err == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        if ratio > worst || ratio.is_nan() {
            worst = ratio;
        }
    }
    Ok(Score {
        abs: err_max,
        rel: err_max / r.scale,
        mean: np_mean(&err_all),
        elem: elem_rel,
        higham: worst,
    })
}

fn shape_str(s: &[usize]) -> String {
    match s {
        [x] => format!("({x},)"),
        _ => format!("({})", s.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    }
}

/// Message for a set of `{lane: worst budget ratio}` that exceeded 1.0.
///
/// Which verdict applies decides the response, and they are opposite: a
/// tessl-only breach is a kernel to fix, an everyone-breached result is a bound
/// to re-derive.
///
/// `externals_scored == false` means no comparison runtime produced a score, so
/// "the others stayed inside" is not evidence anyone has: the verdict must not
/// claim corroboration it lacks.
pub fn adjudicate(over: &BTreeMap<String, f64>, externals_scored: bool) -> String {
    let is_tessl = |l: &str| l.starts_with("tensorops-") || l.starts_with("simdgroup-");
    let tessl = over.keys().any(|l| is_tessl(l));
    let other = over.keys().any(|l| !is_tessl(l));
    let detail = over
        .iter()
        .map(|(l, v)| format!("{l} {v:.3}x"))
        .collect::<Vec<_>>()
        .join(", ");
    let verdict = if tessl && !other && !externals_scored {
        "Only tessl lanes were scored (no comparison runtime was scored), so this cannot be told \
         apart from a bound that is too tight for these operands: unconfirmed, not a verified \
         kernel defect. Re-run with comparison runtimes."
    } else if tessl && !other {
        "Only tessl lanes exceeded it, and the comparison runtimes stayed inside on the same \
         operands: this is a kernel defect."
    } else if other && !tessl {
        "Only the comparison runtimes exceeded it. tessl is inside the bound; the other runtime \
         is the outlier."
    } else {
        "Every runtime exceeded it on the same operands, which points at the bound rather than at \
         any one kernel -- re-derive it for this operand distribution before reading it as a defect."
    };
    format!("per-element error budget exceeded: {detail}. {verdict}")
}

// ---------------------------------------------------------------- the run

/// One row of a scored dump: a lane that was scored on every seed, or one that
/// was skipped on all of them.
#[derive(Clone, Debug, PartialEq)]
pub enum LaneRow {
    Scored(Scored),
    Skipped { lane: String, reason: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Scored {
    pub lane: String,
    pub out_dtype: String,
    pub seeds: usize,
    pub max_rel_err: f64,
    pub median_rel_err: f64,
    pub min_rel_err: f64,
    pub max_abs_err: f64,
    pub mean_abs_err: f64,
    pub max_elem_rel_err: f64,
    pub max_higham_ratio: f64,
    pub per_seed_rel_err: Vec<f64>,
    pub per_seed_higham: Vec<f64>,
    /// How many comparison runtimes this run was scored alongside; 0 means the
    /// numbers above had no cross-runtime corroboration.
    pub external_lanes_scored: usize,
}

impl LaneRow {
    pub fn lane(&self) -> &str {
        match self {
            LaneRow::Scored(s) => &s.lane,
            LaneRow::Skipped { lane, .. } => lane,
        }
    }

    /// The row as the dict the Python scorer built, keys in its order.
    pub fn doc(&self) -> Doc {
        match self {
            LaneRow::Skipped { lane, reason } => obj([
                ("lane", Doc::str(lane)),
                ("skipped", Doc::str(reason)),
                ("seeds", Doc::Int(0)),
            ]),
            LaneRow::Scored(s) => obj([
                ("lane", Doc::str(&s.lane)),
                ("out_dtype", Doc::str(&s.out_dtype)),
                ("seeds", Doc::uint(s.seeds)),
                ("max_rel_err", Doc::Float(s.max_rel_err)),
                ("median_rel_err", Doc::Float(s.median_rel_err)),
                ("min_rel_err", Doc::Float(s.min_rel_err)),
                ("max_abs_err", Doc::Float(s.max_abs_err)),
                ("mean_abs_err", Doc::Float(s.mean_abs_err)),
                ("max_elem_rel_err", Doc::Float(s.max_elem_rel_err)),
                ("max_higham_ratio", Doc::Float(s.max_higham_ratio)),
                ("per_seed_rel_err", Doc::floats(&s.per_seed_rel_err)),
                ("per_seed_higham", Doc::floats(&s.per_seed_higham)),
                ("external_lanes_scored", Doc::uint(s.external_lanes_scored)),
            ]),
        }
    }
}

/// `json.dumps(rows, indent=2)` of a scored dump.
pub fn rows_json(rows: &[LaneRow]) -> Result<String, String> {
    Doc::Arr(rows.iter().map(LaneRow::doc).collect()).dumps()
}

#[derive(Default)]
struct Acc {
    out: String,
    abs: Vec<f64>,
    rel: Vec<f64>,
    mean: Vec<f64>,
    elem: Vec<f64>,
    higham: Vec<f64>,
}

fn read_operand(path: &Path) -> Result<(Vec<usize>, Vec<f64>), String> {
    let arr = tessl::npy::read_npy(path)?;
    let data = match (&arr.data_f32, &arr.data_f64) {
        (Some(f), _) => f.iter().map(|&v| v as f64).collect(),
        (None, Some(d)) => d.clone(),
        _ => return Err(format!("{}: expected a float32 or float64 array", path.display())),
    };
    Ok((arr.shape, data))
}

fn log(line: &str) {
    eprintln!("{line}");
}

/// Score every dumped lane on every seed against one float64 reference per seed,
/// alongside the comparison runtimes on those same operands.
///
/// A reduced-precision lane is *expected* to sit further from the reference (that
/// is what it trades for speed), so the question is never whether it matches
/// bit-for-bit (tf32 and bf16 never can, by construction). It is how far out it
/// lands next to what the runtime it is being compared against costs on the same
/// inputs.
///
/// Reported per lane is the **worst** relative error across all seeds, not the
/// last and not the mean: one draw cannot tell a bound from a lucky sample.
pub fn parity(dir: &Path, precision: &Precision) -> Result<Vec<LaneRow>, String> {
    let man = load_manifest(dir)?;
    let externals = load_externals(dir, &man, precision)?;
    if externals.is_empty() {
        log(
            "parity: NO COMPARISON RUNTIMES SCORED -- no external lanes were supplied, so tessl lanes \
             are checked against the float64 reference only and carry no cross-runtime corroboration.",
        );
    }
    let (m, n, k) = (man.m, man.n, man.k);
    log(&format!(
        "parity: {} lanes x {} seeds at {} (M={m} N={n} K={k})",
        man.lanes.len(),
        man.seeds.len(),
        man.shape.as_deref().unwrap_or("?")
    ));

    // Insertion-ordered, like the dict it replaces: dumped lanes first, then the
    // comparison runtimes in the order they were listed.
    let mut acc: Vec<(String, Acc)> = Vec::new();
    let mut skipped: Vec<(String, Vec<(String, String)>)> = Vec::new();

    fn record(acc: &mut Vec<(String, Acc)>, lane: &str, out_dtype: &str, s: Score) -> Result<(), String> {
        let at = match acc.iter().position(|(l, _)| l == lane) {
            Some(i) => i,
            None => {
                acc.push((
                    lane.to_string(),
                    Acc {
                        out: out_dtype.to_string(),
                        ..Acc::default()
                    },
                ));
                acc.len() - 1
            }
        };
        let e = &mut acc[at].1;
        if e.out != out_dtype {
            return Err(format!(
                "lane {lane}: output dtype changed between seeds ({} then {out_dtype})",
                e.out
            ));
        }
        e.abs.push(s.abs);
        e.rel.push(s.rel);
        e.mean.push(s.mean);
        e.elem.push(s.elem);
        e.higham.push(s.higham);
        Ok(())
    }

    for seed in &man.seeds {
        let sd = dir.join(seed);
        if !sd.is_dir() {
            return Err(format!(
                "manifest lists seed directory {}, which is absent",
                quote(seed)
            ));
        }
        let (a_shape, a) =
            read_operand(&sd.join("parity_a.npy")).map_err(|e| format!("{seed}: operands unreadable: {e}"))?;
        let (b_shape, b) =
            read_operand(&sd.join("parity_b.npy")).map_err(|e| format!("{seed}: operands unreadable: {e}"))?;
        if a_shape != [m, k] || b_shape != [k, n] {
            return Err(format!(
                "{seed}: operand shapes {} @ {} contradict the manifest ({m}x{k} @ {k}x{n})",
                shape_str(&a_shape),
                shape_str(&b_shape)
            ));
        }
        if !(a.iter().all(|v| v.is_finite()) && b.iter().all(|v| v.is_finite())) {
            return Err(format!("{seed}: operands contain non-finite values"));
        }

        let (rc, mag) = reference_gemm(&a, &b, m, k, n);
        drop((a, b));
        let scale = rc.iter().fold(0.0f64, |s, v| s.max(v.abs()));
        if !scale.is_finite() || scale == 0.0 {
            return Err(format!(
                "{seed}: reference peak magnitude is {}; every relative error would be a division \
                 by zero or a NaN",
                float_str(scale)
            ));
        }
        let reference = Reference {
            m,
            n,
            k,
            c: rc,
            mag,
            scale,
        };

        for lane in &man.lanes {
            let file = sd.join(format!("parity_c_{lane}.npy"));
            if !file.exists() {
                return Err(format!(
                    "{seed}: manifest lists lane {} but parity_c_{lane}.npy is missing. Re-run \
                     `bench_gemm_sweep --dump-parity`. Refusing to report a partial check as a \
                     complete one.",
                    quote(lane)
                ));
            }
            let (shape, c) = read_operand(&file)?;
            let s = score(lane, &c, &shape, &reference, seed, precision)?;
            record(&mut acc, lane, "f32", s)?;
        }

        for ext in &externals {
            let result = ext
                .seeds
                .iter()
                .find(|(name, _)| name == seed)
                .map(|(_, r)| r)
                .ok_or_else(|| {
                    format!(
                        "external lane {} has no result for seed {}",
                        quote(&ext.lane),
                        quote(seed)
                    )
                })?;
            match result {
                ExternalSeed::Skipped { reason } => match skipped.iter_mut().find(|(l, _)| *l == ext.lane) {
                    Some((_, v)) => v.push((seed.clone(), reason.clone())),
                    None => skipped.push((ext.lane.clone(), vec![(seed.clone(), reason.clone())])),
                },
                ExternalSeed::Scored { out_dtype } => {
                    let file = sd.join(format!("parity_c_{}.npy", ext.lane));
                    if !file.exists() {
                        return Err(format!(
                            "{seed}: external lane {} reports an output but parity_c_{}.npy is \
                             missing; refusing to report a partial check as a complete one.",
                            quote(&ext.lane),
                            ext.lane
                        ));
                    }
                    let (shape, c) = read_operand(&file)?;
                    let s = score(&ext.lane, &c, &shape, &reference, seed, precision)?;
                    record(&mut acc, &ext.lane, out_dtype, s)?;
                }
            }
        }
    }

    // A lane over budget is adjudicated only once every lane has been scored.
    // Aborting on the first one cannot tell "tessl is out of tolerance" from
    // "this bound is too tight for these operands", and those need opposite
    // responses. If MLX and torch blow the same budget on the same inputs, the
    // bound is the thing that is wrong.
    let mut over: BTreeMap<String, f64> = BTreeMap::new();
    for (lane, e) in &acc {
        let worst = e.higham.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if worst > 1.0 {
            over.insert(lane.clone(), worst);
        }
    }
    if !over.is_empty() {
        let scored_ext = externals.iter().any(|x| acc.iter().any(|(l, _)| *l == x.lane));
        return Err(adjudicate(&over, scored_ext));
    }

    let n_ext_scored = externals
        .iter()
        .filter(|x| acc.iter().any(|(l, _)| *l == x.lane))
        .count();
    let mut order: Vec<String> = acc.iter().map(|(l, _)| l.clone()).collect();
    for (l, _) in &skipped {
        if !order.contains(l) {
            order.push(l.clone());
        }
    }
    let mut rows = Vec::new();
    for lane in order {
        let miss: &[(String, String)] = skipped.iter().find(|(l, _)| *l == lane).map_or(&[], |(_, v)| v);
        let Some((_, e)) = acc.iter().find(|(l, _)| *l == lane) else {
            let mut reasons: Vec<&str> = miss.iter().map(|(_, r)| r.as_str()).collect();
            reasons.sort();
            reasons.dedup();
            log(&format!(
                "parity {lane:<22} SKIPPED on all {} seeds -- {}",
                man.seeds.len(),
                reasons[0]
            ));
            rows.push(LaneRow::Skipped {
                lane,
                reason: reasons.join("; "),
            });
            continue;
        };
        if !miss.is_empty() {
            // Averaging over a lane that ran on some seeds and not others would
            // quietly report a partial sample as a full one.
            let mut failed: Vec<&str> = miss.iter().map(|(s, _)| s.as_str()).collect();
            failed.sort();
            let mut reasons: Vec<&str> = miss.iter().map(|(_, r)| r.as_str()).collect();
            reasons.sort();
            return Err(format!(
                "lane {lane} scored on {} seeds but failed on {}: {}. A lane must cover every \
                 seed or none.",
                e.rel.len(),
                quote_list(&failed),
                reasons[0]
            ));
        }
        let max = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
        let row = Scored {
            lane: lane.clone(),
            out_dtype: e.out.clone(),
            seeds: e.rel.len(),
            max_rel_err: max(&e.rel),
            median_rel_err: median(&e.rel),
            min_rel_err: min(&e.rel),
            max_abs_err: max(&e.abs),
            mean_abs_err: fsum(&e.mean) / e.mean.len() as f64,
            // Worst single output element's relative error: orders of magnitude
            // above max_rel_err wherever cancellation makes an output near zero,
            // and the honest number for "how wrong can one result be".
            max_elem_rel_err: max(&e.elem),
            // Fraction of the per-element error budget used. <= 1.0 or the run
            // would already have aborted; this says how much headroom is left.
            max_higham_ratio: max(&e.higham),
            per_seed_rel_err: e.rel.clone(),
            per_seed_higham: e.higham.clone(),
            external_lanes_scored: n_ext_scored,
        };
        log(&format!(
            "parity {lane:<22} out={:<4} norm={}  elem={}  budget={:.3}x  ({} seeds)",
            row.out_dtype,
            fmt_e(row.max_rel_err, 2),
            fmt_e(row.max_elem_rel_err, 2),
            row.max_higham_ratio,
            row.seeds
        ));
        rows.push(LaneRow::Scored(row));
    }
    Ok(rows)
}

// ------------------------------------------------------------- ladder merge

/// A `(shape, distribution)` cell of the ladder grid.
pub type Cell = (String, String);

/// One lane's row for one cell, with the dimensions the cell was dumped at.
#[derive(Clone, Debug, PartialEq)]
pub struct CellRow {
    pub row: LaneRow,
    pub shape: String,
    pub dist: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl CellRow {
    /// The row as the Python ladder emitted it: the scorer's keys, then the cell's.
    pub fn doc(&self) -> Doc {
        let Doc::Obj(mut fields) = self.row.doc() else {
            unreachable!("LaneRow::doc is an object")
        };
        for (k, v) in [
            ("shape", Doc::str(&self.shape)),
            ("dist", Doc::str(&self.dist)),
            ("m", Doc::uint(self.m)),
            ("n", Doc::uint(self.n)),
            ("k", Doc::uint(self.k)),
        ] {
            fields.push((k.to_string(), v));
        }
        Doc::Obj(fields)
    }
}

/// A lane's worst across the whole grid. A handful exist per run, so the size
/// gap between the variants costs nothing worth an indirection.
#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum LaneMerge {
    Scored {
        lane: String,
        out_dtype: String,
        cells: usize,
        seeds_per_cell: usize,
        worst_rel_err: f64,
        worst_shape: String,
        worst_dist: String,
        best_rel_err: f64,
        median_cell_rel_err: f64,
        worst_higham_ratio: f64,
        worst_higham_shape: String,
        worst_higham_dist: String,
        worst_elem_rel_err: f64,
        worst_elem_shape: String,
        worst_elem_dist: String,
        per_cell_rel_err: Vec<(String, f64)>,
        per_cell_higham: Vec<(String, f64)>,
    },
    Skipped {
        lane: String,
        reason: String,
    },
}

impl LaneMerge {
    pub fn lane(&self) -> &str {
        match self {
            LaneMerge::Scored { lane, .. } | LaneMerge::Skipped { lane, .. } => lane,
        }
    }

    pub fn doc(&self) -> Doc {
        let cells = |v: &[(String, f64)]| Doc::Obj(v.iter().map(|(k, x)| (k.clone(), Doc::Float(*x))).collect());
        match self {
            LaneMerge::Skipped { lane, reason } => obj([
                ("lane", Doc::str(lane)),
                ("skipped", Doc::str(reason)),
                ("cells", Doc::Int(0)),
            ]),
            LaneMerge::Scored {
                lane,
                out_dtype,
                cells: n_cells,
                seeds_per_cell,
                worst_rel_err,
                worst_shape,
                worst_dist,
                best_rel_err,
                median_cell_rel_err,
                worst_higham_ratio,
                worst_higham_shape,
                worst_higham_dist,
                worst_elem_rel_err,
                worst_elem_shape,
                worst_elem_dist,
                per_cell_rel_err,
                per_cell_higham,
            } => obj([
                ("lane", Doc::str(lane)),
                ("out_dtype", Doc::str(out_dtype)),
                ("cells", Doc::uint(*n_cells)),
                ("seeds_per_cell", Doc::uint(*seeds_per_cell)),
                ("worst_rel_err", Doc::Float(*worst_rel_err)),
                ("worst_shape", Doc::str(worst_shape)),
                ("worst_dist", Doc::str(worst_dist)),
                ("best_rel_err", Doc::Float(*best_rel_err)),
                ("median_cell_rel_err", Doc::Float(*median_cell_rel_err)),
                ("worst_higham_ratio", Doc::Float(*worst_higham_ratio)),
                ("worst_higham_shape", Doc::str(worst_higham_shape)),
                ("worst_higham_dist", Doc::str(worst_higham_dist)),
                ("worst_elem_rel_err", Doc::Float(*worst_elem_rel_err)),
                ("worst_elem_shape", Doc::str(worst_elem_shape)),
                ("worst_elem_dist", Doc::str(worst_elem_dist)),
                ("per_cell_rel_err", cells(per_cell_rel_err)),
                ("per_cell_higham", cells(per_cell_higham)),
            ]),
        }
    }
}

/// Worst across every (shape, distribution) cell, per lane.
///
/// `cells` is the full grid that was requested. A lane must cover all of it or
/// none of it: the same rule the seed axis already enforces, one level up.
pub fn merge(per_shape: &[CellRow], cells: &[Cell]) -> Result<Vec<LaneMerge>, String> {
    let mut lanes: Vec<&str> = Vec::new();
    for r in per_shape {
        if !lanes.contains(&r.row.lane()) {
            lanes.push(r.row.lane());
        }
    }
    let want: std::collections::BTreeSet<&Cell> = cells.iter().collect();
    let mut out = Vec::new();
    for lane in lanes {
        let rows: Vec<&CellRow> = per_shape.iter().filter(|r| r.row.lane() == lane).collect();
        let scored: Vec<(&CellRow, &Scored)> = rows
            .iter()
            .filter_map(|r| match &r.row {
                LaneRow::Scored(s) => Some((*r, s)),
                LaneRow::Skipped { .. } => None,
            })
            .collect();
        let covered: std::collections::BTreeSet<Cell> =
            rows.iter().map(|r| (r.shape.clone(), r.dist.clone())).collect();
        if covered.iter().collect::<std::collections::BTreeSet<_>>() != want {
            let missing: Vec<String> = want
                .iter()
                .filter(|c| !covered.contains(**c))
                .map(|(s, d)| format!("('{s}', '{d}')"))
                .collect();
            return Err(format!(
                "lane {lane} is missing from [{}]; a lane must cover the whole grid or none of it",
                missing.join(", ")
            ));
        }
        if scored.is_empty() {
            let reason = match &rows[0].row {
                LaneRow::Skipped { reason, .. } => reason.clone(),
                LaneRow::Scored(_) => String::new(),
            };
            out.push(LaneMerge::Skipped {
                lane: lane.to_string(),
                reason,
            });
            continue;
        }
        if scored.len() != rows.len() {
            // Half a grid averaged into one number reads as a full sweep.
            let absent: Vec<String> = rows
                .iter()
                .filter(|r| matches!(r.row, LaneRow::Skipped { .. }))
                .map(|r| format!("('{}', '{}')", r.shape, r.dist))
                .collect();
            return Err(format!(
                "lane {lane} scored on {}/{} cells, absent on [{}]. A lane must cover the whole \
                 grid or none.",
                scored.len(),
                rows.len(),
                absent.join(", ")
            ));
        }
        let mut dtypes: Vec<&str> = scored.iter().map(|(_, s)| s.out_dtype.as_str()).collect();
        dtypes.sort();
        dtypes.dedup();
        if dtypes.len() != 1 {
            return Err(format!(
                "lane {lane}: output dtype varies across cells: {}",
                quote_list(&dtypes)
            ));
        }
        // First maximal, as Python's max(key=...) keeps the earliest of a tie.
        fn first_max<'a>(xs: &[(&'a CellRow, &'a Scored)], key: impl Fn(&Scored) -> f64) -> (&'a CellRow, &'a Scored) {
            let mut best = xs[0];
            for &x in &xs[1..] {
                if key(x.1) > key(best.1) {
                    best = x;
                }
            }
            best
        }
        let worst = first_max(&scored, |s| s.max_rel_err);
        // The budget ratio is the one that decides whether a lane is *correct*;
        // the others describe how wrong it is when it is still inside.
        let wb = first_max(&scored, |s| s.max_higham_ratio);
        let we = first_max(&scored, |s| s.max_elem_rel_err);
        let rels: Vec<f64> = scored.iter().map(|(_, s)| s.max_rel_err).collect();
        let cell_key = |r: &CellRow| format!("{}/{}", r.shape, r.dist);
        out.push(LaneMerge::Scored {
            lane: lane.to_string(),
            out_dtype: dtypes[0].to_string(),
            cells: scored.len(),
            seeds_per_cell: scored[0].1.seeds,
            worst_rel_err: worst.1.max_rel_err,
            worst_shape: worst.0.shape.clone(),
            worst_dist: worst.0.dist.clone(),
            best_rel_err: rels.iter().cloned().fold(f64::INFINITY, f64::min),
            median_cell_rel_err: median(&rels),
            worst_higham_ratio: wb.1.max_higham_ratio,
            worst_higham_shape: wb.0.shape.clone(),
            worst_higham_dist: wb.0.dist.clone(),
            worst_elem_rel_err: we.1.max_elem_rel_err,
            worst_elem_shape: we.0.shape.clone(),
            worst_elem_dist: we.0.dist.clone(),
            per_cell_rel_err: scored.iter().map(|(r, s)| (cell_key(r), s.max_rel_err)).collect(),
            per_cell_higham: scored.iter().map(|(r, s)| (cell_key(r), s.max_higham_ratio)).collect(),
        });
    }
    Ok(out)
}

/// A scratch directory under the system temp dir, removed on drop.
pub struct Scratch(pub PathBuf);

impl Scratch {
    pub fn new(tag: &str) -> Result<Scratch, String> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let d = std::env::temp_dir().join(format!(
            "{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&d).map_err(|e| format!("mkdir {}: {e}", d.display()))?;
        Ok(Scratch(d))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
