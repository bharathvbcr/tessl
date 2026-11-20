//! The audio tower's benchmark workloads, one owner for `bench_embedgemma2_audio`
//! and `profile_embedgemma2_audio`, so the two time the same inputs.
//!
//! * `clip_1.0s`, `clip_3.3s`: the golden clips `tools/embedgemma2_ref/
//!   make_mm_reference.py` writes (`audio0_*`, `audio1_*`: 99 frames, and 330
//!   frames of which 329 are valid), read from `EMBEDGEMMA2_MM_REF_DIR`. Their
//!   `*_soft.npy` is transformers' answer, so a timed run can be checked
//!   against it.
//! * `batch_8x30s`: eight clips of 3000 frames (30 s at the processor's 100
//!   frames per second), every frame valid. Synthetic: clip `i`'s row `r` is
//!   the golden 3.3 s clip's row `(r + 37 i) mod 330` plus a deterministic
//!   perturbation of at most 0.05, so the values have a real log-mel
//!   distribution and no two clips are equal. Timing does not depend on the
//!   values (it is shapes: 24,000 frames, two forwards of 15,000 and 9,000).
//! * `batch_4x5s`, `batch_8x10s`, `clip_max`: the same synthetic construction at
//!   4 clips of 500 frames, 8 of 1000 and one of `MAX_FRAMES_PER_FORWARD`
//!   (16,384 frames, 163.84 s: the longest clip the tower accepts), named
//!   only; they and `clip_1.0s` (1 s) are the paired speed benchmark's four
//!   workloads (`docs/embedgemma2.md`, the audio tower bf16 and speed section).

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use tessl::embedgemma2_audio::{AudioInput, MAX_FRAMES_PER_FORWARD};
use tessl::npy::read_npy;

/// The workloads a run without a list covers.
pub const WORKLOAD_NAMES: [&str; 3] = ["clip_1.0s", "clip_3.3s", "batch_8x30s"];
/// Named only: the batches of the paired speed benchmark (`docs/embedgemma2.md`, the audio tower bf16 and speed
/// section), each `clips x frames` synthetic like
/// `batch_8x30s`: `batch_4x5s` (4 clips of 5 s), `batch_8x10s` (8 of 10 s) and
/// `clip_max` (one clip of `MAX_FRAMES_PER_FORWARD` frames, the most a forward
/// holds: 163.84 s).
pub const BENCH_NAMES: [&str; 3] = ["batch_4x5s", "batch_8x10s", "clip_max"];
const MELS: usize = 128;

pub struct Workload {
    pub name: &'static str,
    pub clips: Vec<AudioInput>,
    /// Per clip, transformers' soft tokens `[S, 512]` (empty: no reference).
    pub reference: Vec<Vec<f32>>,
}

impl Workload {
    pub fn frames(&self) -> usize {
        self.clips.iter().map(|c| c.mask.len()).sum()
    }
}

pub fn ref_dir() -> Result<PathBuf, String> {
    std::env::var("EMBEDGEMMA2_MM_REF_DIR")
        .map(PathBuf::from)
        .map_err(|_| "EMBEDGEMMA2_MM_REF_DIR must name make_mm_reference.py's (sdpa) output".to_string())
}

fn golden(dir: &Path, prefix: &str) -> Result<(AudioInput, Vec<f32>), String> {
    let feats = read_npy(&dir.join(format!("{prefix}input_features.npy")))?;
    let mask = read_npy(&dir.join(format!("{prefix}input_features_mask.npy")))?;
    let soft = read_npy(&dir.join(format!("{prefix}soft.npy")))?;
    let mask: Vec<bool> = mask.i64_slice()?.iter().map(|&m| m != 0).collect();
    let input_features = feats.f32_slice()?.to_vec();
    if input_features.len() != mask.len() * MELS {
        return Err(format!(
            "{prefix}: {} feature values for {} frames of {MELS} mel bins",
            input_features.len(),
            mask.len()
        ));
    }
    Ok((AudioInput { input_features, mask }, soft.f32_slice()?.to_vec()))
}

/// `clips` synthetic clips of `frames` frames, every frame valid: clip `i`'s row
/// `r` is the golden 3.3 s clip's row `(r + 37 i) mod 330` plus a deterministic
/// perturbation of at most 0.05 (the module docs).
fn synthetic(dir: &Path, clips: usize, frames: usize) -> Result<Vec<AudioInput>, String> {
    let (base, _) = golden(dir, "audio1_")?;
    let base_frames = base.mask.len();
    Ok((0..clips)
        .map(|i| {
            let mut input_features = Vec::with_capacity(frames * MELS);
            let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (i as u64 + 1);
            for r in 0..frames {
                let src = (r + 37 * i) % base_frames;
                for m in 0..MELS {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = ((state >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0;
                    input_features.push(base.input_features[src * MELS + m] + 0.05 * u);
                }
            }
            AudioInput {
                input_features,
                mask: vec![true; frames],
            }
        })
        .collect())
}

/// The named workload; the golden files come from `EMBEDGEMMA2_MM_REF_DIR`.
pub fn workload(name: &str) -> Result<Workload, String> {
    let dir = ref_dir()?;
    let (name, clips, reference) = match name {
        "clip_1.0s" => {
            let (c, s) = golden(&dir, "audio0_")?;
            ("clip_1.0s", vec![c], vec![s])
        }
        "clip_3.3s" => {
            let (c, s) = golden(&dir, "audio1_")?;
            ("clip_3.3s", vec![c], vec![s])
        }
        "batch_8x30s" => ("batch_8x30s", synthetic(&dir, 8, 3000)?, Vec::new()),
        "batch_4x5s" => ("batch_4x5s", synthetic(&dir, 4, 500)?, Vec::new()),
        "batch_8x10s" => ("batch_8x10s", synthetic(&dir, 8, 1000)?, Vec::new()),
        "clip_max" => ("clip_max", synthetic(&dir, 1, MAX_FRAMES_PER_FORWARD)?, Vec::new()),
        other => {
            return Err(format!(
                "unknown workload {other:?}; expected one of {WORKLOAD_NAMES:?} or {BENCH_NAMES:?}"
            ))
        }
    };
    Ok(Workload { name, clips, reference })
}

/// The workloads named by `var` (a comma list), all of them when unset.
pub fn selected(var: &str) -> Result<Vec<Workload>, String> {
    let spec = std::env::var(var).unwrap_or_else(|_| WORKLOAD_NAMES.join(","));
    spec.split(',').map(|n| workload(n.trim())).collect()
}

/// Largest `|got - want|` over the largest `|want|`: the measure the model
/// tests bound (`1e-4` for the soft tokens).
pub fn rel_max_err(got: &[f32], want: &[f32]) -> Result<f64, String> {
    crate::common::max_rel_diff("soft tokens", got, want)
}

/// An observed error as JSON: the number, or `null` where there was no reference.
pub fn json_opt(e: Option<f64>) -> String {
    e.map_or_else(|| "null".to_string(), |e| format!("{e:e}"))
}

/// Cosine of two vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    // Vectors of different length (`zip` would stop at the shorter) or of
    // no length have no cosine: NaN, which every caller's floor test refuses.
    if a.len() != b.len() || a.is_empty() {
        return f64::NAN;
    }
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

/// A workload with a reference must be answered with exactly one set of soft
/// tokens per reference clip: a short answer would leave the unmatched
/// references unchecked (the loops below index references by clip), and an
/// answer with no clips has nothing to check at all. A workload with no
/// reference (the synthetic batches) is held to finiteness only.
fn check_answer_covers_reference(w: &Workload, soft_tokens: &[Vec<f32>]) -> Result<(), String> {
    if soft_tokens.is_empty() {
        return Err(format!("{}: the forward returned no clips", w.name));
    }
    if !w.reference.is_empty() && soft_tokens.len() != w.reference.len() {
        return Err(format!(
            "{}: {} clips answered against {} reference clips",
            w.name,
            soft_tokens.len(),
            w.reference.len()
        ));
    }
    Ok(())
}

/// [`check`] for a bf16 forward, whose soft tokens are not within `1e-4` of
/// transformers' fp32 ones: fails unless every output is finite and, where
/// the workload has a reference, at cosine `floor` or better with it. Returns
/// the worst `(largest |got - want| over largest |want|, cosine)` over the
/// referenced clips, `None` for a workload with no reference.
pub fn check_cosine(w: &Workload, soft_tokens: &[Vec<f32>], floor: f64) -> Result<Option<(f64, f64)>, String> {
    check_answer_covers_reference(w, soft_tokens)?;
    let mut worst: Option<(f64, f64)> = None;
    for (i, s) in soft_tokens.iter().enumerate() {
        if s.is_empty() || !s.iter().all(|v| v.is_finite()) {
            return Err(format!("{}: clip {i} has no soft tokens or a non-finite one", w.name));
        }
        if let Some(r) = w.reference.get(i) {
            let (e, c) = (rel_max_err(s, r)?, cosine(s, r));
            if c.is_nan() || c < floor {
                return Err(format!(
                    "{}: clip {i}'s soft tokens are at cosine {c:.8} from transformers' (floor {floor})",
                    w.name
                ));
            }
            worst = Some(worst.map_or((e, c), |(we, wc)| (we.max(e), wc.min(c))));
        }
    }
    Ok(worst)
}

/// Fails unless every output is finite and, where the workload has a
/// reference, within `bound` of it (relative to its largest magnitude).
pub fn check(w: &Workload, soft_tokens: &[Vec<f32>], bound: f64) -> Result<Option<f64>, String> {
    check_answer_covers_reference(w, soft_tokens)?;
    let mut worst: Option<f64> = None;
    for (i, s) in soft_tokens.iter().enumerate() {
        if s.is_empty() || !s.iter().all(|v| v.is_finite()) {
            return Err(format!("{}: clip {i} has no soft tokens or a non-finite one", w.name));
        }
        if let Some(r) = w.reference.get(i) {
            let e = rel_max_err(s, r)?;
            if e.is_nan() || e > bound {
                return Err(format!(
                    "{}: clip {i}'s soft tokens are {e:.3e} of max from transformers (bound {bound:.0e})",
                    w.name
                ));
            }
            worst = Some(worst.map_or(e, |w| w.max(e)));
        }
    }
    Ok(worst)
}

#[cfg(test)]
mod tests {
    use super::{check, check_cosine, cosine, rel_max_err, Workload};

    fn workload(reference: Vec<Vec<f32>>) -> Workload {
        Workload {
            name: "t",
            clips: Vec::new(),
            reference,
        }
    }

    /// The expected set must cover the answer: a short or empty set, an empty
    /// answer, NaN, a length mismatch and a scale-less reference are refused.
    #[test]
    fn short_empty_and_non_finite_expected_sets_are_refused() {
        let want = vec![vec![1.0f32, -2.0], vec![0.5, 0.5]];
        let w = workload(want.clone());
        assert!(check(&w, &want, 1e-4).is_ok());
        assert!(check_cosine(&w, &want, 0.999).is_ok());
        // Fewer clips answered than the reference holds.
        assert!(check(&w, &want[..1], 1e-4).unwrap_err().contains("1 clips answered against 2"));
        assert!(check_cosine(&w, &want[..1], 0.999).is_err());
        // More clips answered than referenced.
        let mut more = want.clone();
        more.push(vec![1.0, 1.0]);
        assert!(check(&w, &more, 1e-4).is_err());
        // Nothing answered, with or without a reference.
        assert!(check(&w, &[], 1e-4).unwrap_err().contains("no clips"));
        assert!(check(&workload(Vec::new()), &[], 1e-4).is_err());
        assert!(check_cosine(&workload(Vec::new()), &[], 0.9).is_err());
        // An empty reference clip, a NaN answer, a wrong value.
        assert!(check(&workload(vec![Vec::new()]), &[vec![1.0]], 1e-4).is_err());
        assert!(check(&w, &[vec![f32::NAN, 1.0], want[1].clone()], 1e-4).is_err());
        assert!(check(&w, &[vec![1.0, -2.5], want[1].clone()], 1e-4).is_err());
        // No reference: finiteness only, and still at least one clip.
        assert_eq!(check(&workload(Vec::new()), &[vec![3.0]], 1e-4).unwrap(), None);
        // The measure itself.
        assert!(rel_max_err(&[0.0], &[0.0]).unwrap_err().contains("all zero"));
        assert!(rel_max_err(&[], &[]).unwrap_err().contains("nothing to compare"));
        assert!(rel_max_err(&[1.0], &[1.0, 2.0]).is_err());
        // A NaN anywhere, in the reference too, is not skipped by the max.
        assert!(rel_max_err(&[1.0, 1.0], &[1.0, f32::NAN]).is_err());
        assert!(cosine(&[1.0, 0.0], &[1.0]).is_nan());
        assert!(cosine(&[], &[]).is_nan());
    }
}
