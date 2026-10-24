//! The f64 oracle for one cross-entropy row, in the form the kernels compute.
//!
//! With `L` the log-sum-exp of the classes other than the target `t`, and
//! `margin = z_t - L`:
//!
//! ```text
//! loss = softplus(-margin) = max(-margin, 0) + log1p(exp(-|margin|))
//! q    = sigmoid(-margin)  = sum over j != t of exp(z_j - L) * q  (exactly)
//! g_j  = exp(z_j - L) * q        for j != t
//! g_t  = -q
//! ```
//!
//! which is `softmax - onehot` written without the `1 + exp(-margin)` that
//! rounds to 1 past the precision cliff (Barron, "A Fix for Softmax and
//! Cross-Entropy"). f64 reaches margins of 700 before `exp` leaves its range,
//! so the oracle is exact for everything the f32 tests probe. A row whose
//! other classes are all `-inf` has loss 0 and zero gradient.

/// `(loss, gradient of the loss)` for the logits `z` and target `t`.
pub fn ce_oracle(z: &[f64], t: usize) -> (f64, Vec<f64>) {
    let mo = z
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != t)
        .fold(f64::NEG_INFINITY, |m, (_, &l)| m.max(l));
    if mo == f64::NEG_INFINITY {
        return (0.0, vec![0.0; z.len()]);
    }
    let s: f64 = z
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != t)
        .map(|(_, &l)| (l - mo).exp())
        .sum();
    let lo = mo + s.ln();
    let margin = z[t] - lo;
    let x = -margin;
    let loss = if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    };
    let q = 1.0 / (1.0 + margin.exp());
    let g = z
        .iter()
        .enumerate()
        .map(|(j, &l)| if j == t { -q } else { (l - lo).exp() * q })
        .collect();
    (loss, g)
}

/// `log(sum_{j != t} exp(z_j))`, for sizing a row-sum tolerance.
pub fn other_classes_lse(z: &[f64], t: usize) -> f64 {
    let mo = z
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != t)
        .fold(f64::NEG_INFINITY, |m, (_, &l)| m.max(l));
    mo + z
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != t)
        .map(|(_, &l)| (l - mo).exp())
        .sum::<f64>()
        .ln()
}
