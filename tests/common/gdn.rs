//! The gated delta rule, as an f64 sequential reference.
//!
//! This is the CPU oracle the GDN kernels are judged against. It exists in Rust,
//! rather than only as the Python-generated `*_y_seq_f64.npy` fixtures, for two
//! reasons: a kernel test at a shape nobody generated a fixture for still needs
//! a reference, and an operator definition that lives only in another repo's
//! Python is not one this crate can be held to.
//!
//! # The two rules are different operators
//!
//! `MLSystemsLab/nanolab/mixers.py:767-773` gives both state updates:
//!
//! ```text
//! repo:      S <- a*S + b * k (v -     S^T k)^T     (undecayed read)
//! published: S <- a*S + b * k (v - a * S^T k)^T     (arXiv:2412.06464 eq. 8)
//! y = S^T q                                          (against the updated S)
//! ```
//!
//! They differ in one factor — whether the correction reads the state before or
//! after decay — and that is not a rounding difference. Measured on the fixture
//! corpus the two diverge by up to 31% of the output magnitude.
//!
//! They do coincide in one case, exactly: at `t = 0` the state is zero, so
//! `a * S^T k` and `S^T k` are both zero. A single-step problem therefore cannot
//! distinguish the rules, which is worth knowing before using one as evidence.
//!
//! nanolab's **default is `rule="repo"`**, and every committed nanolab GDN run
//! used it. The published rule is what this project targets, which is why repo
//! rule 9 requires every fixture and test to name its rule: a golden generated
//! at the default is silently the wrong operator, and nothing about its shape,
//! dtype or magnitude would reveal it. [`Rule::Repo`] is implemented here purely
//! so a test can demonstrate the gap rather than assert the absence of one.
//!
//! # The clamp is part of the operator
//!
//! `alpha` and `beta` are clamped before they enter the recurrence, and on the
//! `L65_tinyalpha` fixture that is not cosmetic: the stored `alpha.npy` holds
//! `1e-8` while the golden was generated with it clamped up to the `1e-4` floor.
//! Running the recurrence on the stored gate reproduces the golden to only
//! `8.4e-5` relative; clamping first reproduces it to `1.9e-16`. The clamp is
//! therefore applied *here*, not left to the caller, because a caller who forgets
//! it gets a plausible wrong answer rather than an error.

use std::path::{Path, PathBuf};

use tessl::npy::read_npy;

/// Which state the delta correction reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// `arXiv:2412.06464` eq. 8 — the correction reads the *decayed* state.
    /// The operator this project targets.
    Published,
    /// nanolab's default — the correction reads the undecayed state. Present for
    /// contrast only; a fixture generated under it is not a valid golden.
    Repo,
}

/// Inclusive bounds applied to each gate before it enters the recurrence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateClamp {
    pub alpha: (f64, f64),
    pub beta: (f64, f64),
}

impl GateClamp {
    /// The clamp every published fixture manifest declares. Held to that claim
    /// by a test, so a manifest that changes cannot leave this stale.
    pub const PUBLISHED: Self = Self {
        alpha: (1e-4, 1.0),
        beta: (0.0, 1.0),
    };
}

/// Shape of one GDN problem. `d` is both the key and the value dimension.
#[derive(Debug, Clone, Copy)]
pub struct Dims {
    pub b: usize,
    pub h: usize,
    pub l: usize,
    pub d: usize,
}

impl Dims {
    /// Elements in a `[B, H, L, D]` tensor.
    pub fn qkv_len(&self) -> usize {
        self.b * self.h * self.l * self.d
    }

    /// Elements in a `[B, H, L]` gate tensor.
    pub fn gate_len(&self) -> usize {
        self.b * self.h * self.l
    }
}

/// One problem's operands. `q`/`k`/`v` are `[B,H,L,D]` row-major, the gates are
/// `[B,H,L]`. For a grouped-query case `k` and `v` are expected already expanded
/// to `dims.h` heads, which is how the published fixtures store them.
#[derive(Debug, Clone, Copy)]
pub struct Problem<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub alpha: &'a [f32],
    pub beta: &'a [f32],
    pub dims: Dims,
}

/// Runs the gated delta rule sequentially in f64 and returns `y` as `[B,H,L,D]`.
///
/// Operands arrive as `f32` because that is what the fixtures store; widening
/// `f32` to `f64` is exact, so the reference is never handed a rounded input.
/// Everything after that accumulates in `f64`, which is the whole point — the
/// reference must not be a meaningful source of the error it is used to measure.
///
/// # Panics
///
/// If any slice length disagrees with `dims`. A reference that quietly consumed
/// a mis-shaped operand would return plausible numbers for a different problem.
pub fn sequential_f64(p: &Problem<'_>, rule: Rule, clamp: GateClamp) -> Vec<f64> {
    state_update_f64(p, rule, clamp, SumOrder::Sequential).y
}

/// How the two `d`-term dot products inside the recurrence are summed.
///
/// A Metal kernel does not sum in source order: a simdgroup reduces across lanes
/// by shuffling, which is a tree. Both orders are exact in exact arithmetic and
/// differ in floating point, so a reference offering only one order cannot
/// distinguish a lawful reassociation from a defect.
/// `AUDIT/tessl-integration.md:571-572` records this as the reason
/// `tests/qkv_rope.rs` compares against f64 rather than a matching f32
/// accumulation, and says K7/K3 need the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SumOrder {
    /// Ascending `p`, one accumulator. What this reference has always done, kept
    /// bit-identical so [`sequential_f64`] is unchanged by the refactor that
    /// introduced this enum.
    Sequential,
    /// Recursive halving (`buf[i] += buf[i + half]`), the shape a simdgroup
    /// shuffle-down reduction produces. An odd tail is carried, never dropped.
    PairwiseTree,
}

/// **K3.** The f64 recurrence, plus the magnitude terms an f32 error bound scales
/// with.
///
/// `AUDIT/tessl-integration.md:639` — *"a delta-rule state update is a K-term
/// accumulation"* — so the bound K3 is judged by has to be proportional to
/// `sum_p |products|`, the way `common::tolerance` is for the GEMMs, and not to
/// `|y|`. An element whose terms cancel has a small result and a large error
/// budget; judging it by its own `|y|` demands accuracy f32 cannot deliver, and
/// judging it by `max|y|` over the whole tensor — which is what
/// `gdn_fixtures.rs` does today — is a global fudge that happens to hold on this
/// corpus at `d = 8`. These fields are what make the bound derivable instead.
pub struct StateUpdate {
    /// `[B,H,L,D]`. Exactly the values [`sequential_f64`] returns.
    pub y: Vec<f64>,
    /// `[B,H,L,D]`: `sum_p |S[p][n] * q[p]|` over the state that produced `y`.
    pub y_mag: Vec<f64>,
    /// `[B,H,L,D]`: `sum_p |S[p][n] * k[p]|`, the *other* `d`-term accumulation.
    /// It feeds `delta` and therefore the state itself, so a reassociation
    /// reaches this one step before it reaches `y`.
    pub pred_mag: Vec<f64>,
}

/// Recursive halving, the shape a simdgroup shuffle-down reduction produces.
///
/// An odd `n` leaves the last element unconsumed; it is folded into the next
/// round rather than dropped, which is the bug this being one function avoids.
fn sum_tree(buf: &mut [f64]) -> f64 {
    let mut n = buf.len();
    if n == 0 {
        return 0.0;
    }
    while n > 1 {
        let half = n / 2;
        for i in 0..half {
            buf[i] += buf[i + half];
        }
        if n % 2 == 1 {
            buf[half] = buf[n - 1];
            n = half + 1;
        } else {
            n = half;
        }
    }
    buf[0]
}

/// `out[n] = sum_p s[p*d + n] * x[p]`, summed in `order`.
fn dot_state(out: &mut [f64], s: &[f64], x: &[f32], d: usize, order: SumOrder, scratch: &mut [f64]) {
    match order {
        SumOrder::Sequential => {
            out.fill(0.0);
            for pi in 0..d {
                let xp = f64::from(x[pi]);
                let row = &s[pi * d..pi * d + d];
                for (n, acc) in out.iter_mut().enumerate() {
                    *acc += row[n] * xp;
                }
            }
        }
        SumOrder::PairwiseTree => {
            for (n, cell) in out.iter_mut().enumerate() {
                for (pi, slot) in scratch[..d].iter_mut().enumerate() {
                    *slot = s[pi * d + n] * f64::from(x[pi]);
                }
                *cell = sum_tree(&mut scratch[..d]);
            }
        }
    }
}

/// `out[n] = sum_p |s[p*d + n] * x[p]|`. Always summed in ascending `p`: this is
/// an input to an error bound, not a result being bounded, and every term is
/// non-negative so there is no cancellation for an order to expose.
fn abs_dot_state(out: &mut [f64], s: &[f64], x: &[f32], d: usize) {
    out.fill(0.0);
    for pi in 0..d {
        let xp = f64::from(x[pi]);
        let row = &s[pi * d..pi * d + d];
        for (n, acc) in out.iter_mut().enumerate() {
            // The product's magnitude is the quantity the bound needs, and
            // `|a*b| == |a|*|b|`, so absing the operand first would be redundant
            // rather than safer. A mutation run found that out: pre-absing `xp`
            // and then absing the product made one of the two unkillable.
            *acc += (row[n] * xp).abs();
        }
    }
}

/// Runs the recurrence and reports the magnitudes alongside the result.
///
/// See [`StateUpdate`] for why the magnitudes are here, and [`SumOrder`] for why
/// the summation order is a parameter. [`sequential_f64`] is this function at
/// `SumOrder::Sequential`, so there is one recurrence in this file and not two.
///
/// # Panics
///
/// If any slice length disagrees with `dims`.
pub fn state_update_f64(p: &Problem<'_>, rule: Rule, clamp: GateClamp, order: SumOrder) -> StateUpdate {
    let Dims { b, h, l, d } = p.dims;
    for (name, got) in [("q", p.q.len()), ("k", p.k.len()), ("v", p.v.len())] {
        assert_eq!(
            got,
            p.dims.qkv_len(),
            "{name} has {got} elements, expected {} for [{b},{h},{l},{d}]",
            p.dims.qkv_len()
        );
    }
    for (name, got) in [("alpha", p.alpha.len()), ("beta", p.beta.len())] {
        assert_eq!(
            got,
            p.dims.gate_len(),
            "{name} has {got} elements, expected {} for [{b},{h},{l}]",
            p.dims.gate_len()
        );
    }

    let mut y = vec![0.0f64; p.dims.qkv_len()];
    let mut y_mag = vec![0.0f64; p.dims.qkv_len()];
    let mut pred_mag = vec![0.0f64; p.dims.qkv_len()];
    // s[p * d + n]: `p` indexes the key dimension, `n` the value dimension. This
    // is the transpose of the chunked kernel's state layout; outputs match.
    let mut s = vec![0.0f64; d * d];
    let mut pred = vec![0.0f64; d];
    let mut delta = vec![0.0f64; d];
    let mut scratch = vec![0.0f64; d];

    for bi in 0..b {
        for hi in 0..h {
            s.fill(0.0);
            for t in 0..l {
                let base = ((bi * h + hi) * l + t) * d;
                let gate = (bi * h + hi) * l + t;
                let a = f64::from(p.alpha[gate]).clamp(clamp.alpha.0, clamp.alpha.1);
                let be = f64::from(p.beta[gate]).clamp(clamp.beta.0, clamp.beta.1);
                let k_row = &p.k[base..base + d];

                // pred = S^T k, and the magnitude that accumulation's error
                // scales with. Both read the state *before* the update.
                dot_state(&mut pred, &s, k_row, d, order, &mut scratch);
                abs_dot_state(&mut pred_mag[base..base + d], &s, k_row, d);

                // delta = (v - read(S^T k)) * beta, where `read` is the rule.
                let decay = match rule {
                    Rule::Published => a,
                    Rule::Repo => 1.0,
                };
                for (n, cell) in delta.iter_mut().enumerate() {
                    *cell = (f64::from(p.v[base + n]) - decay * pred[n]) * be;
                }

                // S <- a*S + outer(k, delta)
                for pi in 0..d {
                    let kp = f64::from(p.k[base + pi]);
                    let row = &mut s[pi * d..pi * d + d];
                    for (n, cell) in row.iter_mut().enumerate() {
                        *cell = a * *cell + kp * delta[n];
                    }
                }

                // y_t = S^T q_t, against the state *after* the update.
                let q_row = &p.q[base..base + d];
                dot_state(&mut y[base..base + d], &s, q_row, d, order, &mut scratch);
                abs_dot_state(&mut y_mag[base..base + d], &s, q_row, d);
            }
        }
    }
    StateUpdate { y, y_mag, pred_mag }
}

// ------------------------------------------------- the published fixture corpus ---

/// Where the published GDN fixtures live.
pub fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gdn")
}

/// Every case name in the corpus, discovered from the chunked outputs on disk.
///
/// Discovered rather than hard-coded so that adding a case to the generator
/// cannot leave a suite quietly testing the old set.
pub fn case_names() -> Vec<String> {
    const PREFIX: &str = "gdn_published_";
    const SUFFIX: &str = "_y_chunked.npy";
    let mut names: Vec<String> = std::fs::read_dir(fixture_dir())
        .expect("fixture directory must exist")
        .map(|e| e.expect("readable dir entry").file_name())
        .filter_map(|n| {
            let n = n.to_str()?;
            Some(n.strip_prefix(PREFIX)?.strip_suffix(SUFFIX)?.to_string())
        })
        .collect();
    names.sort();
    assert!(
        !names.is_empty(),
        "no published GDN fixtures found in {}; a suite over them must not pass vacuously",
        fixture_dir().display()
    );
    names
}

/// One case's recurrence inputs and its f64 golden, loaded from disk.
pub struct Case {
    pub name: String,
    pub dims: Dims,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub alpha: Vec<f32>,
    pub beta: Vec<f32>,
    pub golden: Vec<f64>,
}

/// # Panics
///
/// If a file is missing, has the wrong rank, or disagrees with `q`'s shape. A
/// loader that shrugged at a shape mismatch would hand the recurrence a
/// different problem than the golden was generated for.
pub fn load_case(name: &str) -> Case {
    let load = |suffix: &str| {
        let path = fixture_dir().join(format!("gdn_published_{name}_{suffix}.npy"));
        read_npy(&path).unwrap_or_else(|e| panic!("load {name}_{suffix}: {e}"))
    };

    let q = load("q");
    // [B, H, L, D] for q/k/v, [B, H, L] for the gates.
    assert_eq!(q.shape.len(), 4, "{name}: q must be rank 4, got {:?}", q.shape);
    let dims = Dims {
        b: q.shape[0],
        h: q.shape[1],
        l: q.shape[2],
        d: q.shape[3],
    };

    let k = load("k");
    let v = load("v");
    let alpha = load("alpha");
    let beta = load("beta");
    let golden = load("y_seq_f64");
    assert_eq!(alpha.shape, vec![dims.b, dims.h, dims.l], "{name}: alpha shape");
    assert_eq!(beta.shape, vec![dims.b, dims.h, dims.l], "{name}: beta shape");
    assert_eq!(golden.shape, q.shape, "{name}: golden shape");

    Case {
        name: name.to_string(),
        dims,
        q: q.f32_slice().expect("q is f32").to_vec(),
        k: k.f32_slice().expect("k is f32").to_vec(),
        v: v.f32_slice().expect("v is f32").to_vec(),
        alpha: alpha.f32_slice().expect("alpha is f32").to_vec(),
        beta: beta.f32_slice().expect("beta is f32").to_vec(),
        golden: golden.f64_slice().expect("golden is f64").to_vec(),
    }
}

impl Case {
    pub fn problem(&self) -> Problem<'_> {
        Problem {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            alpha: &self.alpha,
            beta: &self.beta,
            dims: self.dims,
        }
    }

    pub fn run(&self, rule: Rule) -> Vec<f64> {
        sequential_f64(&self.problem(), rule, GateClamp::PUBLISHED)
    }
}

/// Per-head gate biases. nanolab zero-initialises both (`mixers.py:784-785`) but
/// they are trainable, so a reference that hard-coded zero would agree with an
/// untrained checkpoint and nothing else.
#[derive(Debug, Clone, Copy)]
pub struct GateBias<'a> {
    /// `decay_bias`, `[H]`. Added before the sigmoid that produces `alpha`.
    pub decay: &'a [f64],
    /// `update_bias`, `[H]`. Added before the sigmoid that produces `beta`.
    pub update: &'a [f64],
}

fn sigmoid(x: f64) -> f64 {
    // Branch on the sign so neither `exp` argument is ever large and positive.
    // `1/(1+exp(-x))` overflows for very negative `x`; this form does not.
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// **K7.** The published gate production: `sigmoid(gate + per-head bias)`, clamped.
///
/// `MLSystemsLab/nanolab/mixers.py:803-804`, as recorded in
/// `qwen-decision/AUDIT/gdn-reference-and-contracts.md` §4.2:
///
/// ```text
/// alpha = sigmoid(a_gate + decay_bias ).clamp(1e-4, 1.0)
/// beta  = sigmoid(b_gate + update_bias).clamp(0.0,  1.0)
/// ```
///
/// The plan specified something else entirely — see [`alpha_mamba2_refuted`].
///
/// # Two silent hazards this signature closes
///
/// **The layout changes.** `a_gate` and `b_gate` are the tail of a split over the
/// feature axis (`mixers.py:797-798`, sizes `[D, D, D, n_head, n_head]`), so they
/// arrive as `[B, T, H]` with the head index **last**. The recurrence wants
/// `[B, H, L]`. Both hold `B*H*L` elements, so a missed transpose changes every
/// value while changing no shape, and the result stays in range because it is
/// still a set of sigmoids. This function consumes `[B, T, H]` and returns
/// `[B, H, L]`, so the conversion cannot be forgotten by a caller.
///
/// **Three of the four clamp bounds can never change a value.** `sigmoid` has
/// range `(0, 1)` exactly and `[0, 1]` after rounding, so `alpha`'s ceiling of
/// 1.0 and both of `beta`'s bounds can be *attained* but never *exceeded* —
/// clamping at them is arithmetically a no-op. The one clamp that does work is
/// `alpha`'s floor of 1e-4, which engages below
/// `a_gate + decay_bias = ln(1e-4 / (1 - 1e-4)) = -9.21024`. That is why the
/// audit calls the floor load-bearing, and why `L65_tinyalpha` exists as a
/// fixture: it is the only clamp with anything to do.
///
/// **Where `alpha` saturates is arithmetic-dependent, and the gap compounds.**
/// f64 `sigmoid` returns exactly 1.0 once `exp(-x)` falls below `2^-53`, above
/// `x ~= 36.74`. An f32 kernel does so above `x ~= 16.64`, because its unit
/// roundoff is `2^-24`. Between those thresholds an f32 K7 kernel yields
/// `alpha == 1.0` while this reference yields `1.0 - eps`, a gap of at most one
/// `u_f32` per step — but `alpha` enters the recurrence *multiplicatively*, so
/// across `L` steps the gap on the decay product grows to roughly `L * u_f32`.
/// At `L = 8191` that is 4.9e-4, which is 8200 `u_f32` and four orders above the
/// `64 * u_f32` agreement `gdn_fixtures.rs` holds the chunked kernel to. An
/// early-saturating K7 kernel is therefore not automatically wrong, and equally
/// cannot be judged by the fixture bound. Both thresholds are pinned by test.
///
/// # Panics
///
/// If any slice length disagrees with `dims`, or a bias is not `[H]`.
pub fn gates_published(
    a_gate_bth: &[f32],
    b_gate_bth: &[f32],
    dims: Dims,
    bias: GateBias<'_>,
    clamp: GateClamp,
) -> (Vec<f64>, Vec<f64>) {
    let Dims { b, h, l, .. } = dims;
    for (name, got) in [("a_gate", a_gate_bth.len()), ("b_gate", b_gate_bth.len())] {
        assert_eq!(
            got,
            dims.gate_len(),
            "{name} has {got} elements, expected {} for [{b},{l},{h}]",
            dims.gate_len()
        );
    }
    for (name, got) in [("decay", bias.decay.len()), ("update", bias.update.len())] {
        assert_eq!(
            got, h,
            "{name} bias has {got} entries, expected one per head ({h}). A bias indexed by \
             position instead of head is the mistake this check exists for"
        );
    }
    assert_finite("decay bias", bias.decay);
    assert_finite("update bias", bias.update);

    let mut alpha = vec![0.0f64; dims.gate_len()];
    let mut beta = vec![0.0f64; dims.gate_len()];
    for bi in 0..b {
        for t in 0..l {
            for hi in 0..h {
                let src = (bi * l + t) * h + hi; // [B, T, H]
                let dst = (bi * h + hi) * l + t; // [B, H, L]
                let (a_raw, b_raw) = (f64::from(a_gate_bth[src]), f64::from(b_gate_bth[src]));
                // `f64::clamp` passes NaN straight through, so an upstream NaN
                // would survive the bounds, enter the recurrence, and poison
                // every later step — where it surfaces as a kernel mismatch
                // rather than as the bad input it is.
                assert!(
                    a_raw.is_finite() && b_raw.is_finite(),
                    "non-finite gate logit at [b{bi} t{t} h{hi}]: a_gate {a_raw}, b_gate {b_raw}"
                );
                alpha[dst] = sigmoid(a_raw + bias.decay[hi]).clamp(clamp.alpha.0, clamp.alpha.1);
                beta[dst] = sigmoid(b_raw + bias.update[hi]).clamp(clamp.beta.0, clamp.beta.1);
            }
        }
    }
    (alpha, beta)
}

fn assert_finite(what: &str, xs: &[f64]) {
    for (i, x) in xs.iter().enumerate() {
        assert!(x.is_finite(), "{what}[{i}] is not finite: {x}");
    }
}

/// **The plan's K7 alpha, which belongs to a different mixer.**
///
/// `alpha = exp(-exp(A_log) * softplus(a_gate + dt_bias))` is the Mamba2 / SSD
/// decay from `mixers.py:621-642`, not the gated delta rule's. `A_log` and
/// `dt_bias` are not GDN parameters at all; GDN's is `decay_bias`, and its gate is
/// a plain sigmoid.
///
/// Implemented here for the same reason [`Rule::Repo`] is: so a test can measure
/// how far wrong the plan was rather than assert that it was not. The audit calls
/// this the most dangerous single error it found, because K7 reads as the safest
/// kernel in the set — "a small elementwise pass" — and a kernel written from the
/// plan would have been wrong in a way no shape or dtype check could see.
///
/// Returns `[B, H, L]`, the same layout [`gates_published`] returns, so the two
/// are directly comparable.
pub fn alpha_mamba2_refuted(a_gate_bth: &[f32], dims: Dims, a_log: &[f64], dt_bias: &[f64]) -> Vec<f64> {
    let Dims { b, h, l, .. } = dims;
    assert_eq!(a_gate_bth.len(), dims.gate_len(), "a_gate length");
    assert_eq!(a_log.len(), h, "A_log is per-head");
    assert_eq!(dt_bias.len(), h, "dt_bias is per-head");
    assert_finite("A_log", a_log);
    assert_finite("dt_bias", dt_bias);

    // softplus in the shifted form. The textbook `ln(1 + exp(x))` overflows to
    // infinity above x = 709, and `exp(-A * inf)` is then 0 rather than a number,
    // so a large logit paired with a small decay rate would silently read as
    // total decay instead of the partial decay it is. A contrast reference that
    // returns 0 where it should return 0.449 makes the divergence it exists to
    // measure look larger than it is.
    let softplus = |x: f64| x.max(0.0) + (-x.abs()).exp().ln_1p();

    let mut out = vec![0.0f64; dims.gate_len()];
    for bi in 0..b {
        for t in 0..l {
            for hi in 0..h {
                let src = (bi * l + t) * h + hi;
                let dst = (bi * h + hi) * l + t;
                let raw = f64::from(a_gate_bth[src]);
                assert!(raw.is_finite(), "non-finite a_gate at [b{bi} t{t} h{hi}]: {raw}");
                let dt = softplus(raw + dt_bias[hi]);
                out[dst] = (-a_log[hi].exp() * dt).exp();
            }
        }
    }
    out
}
