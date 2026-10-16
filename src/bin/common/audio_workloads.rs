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

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use tessl::embedgemma2_audio::AudioInput;
use tessl::npy::read_npy;

pub const WORKLOAD_NAMES: [&str; 3] = ["clip_1.0s", "clip_3.3s", "batch_8x30s"];
const MELS: usize = 128;
const LONG_CLIPS: usize = 8;
const LONG_FRAMES: usize = 3000;

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
        "batch_8x30s" => {
            let (base, _) = golden(&dir, "audio1_")?;
            let base_frames = base.mask.len();
            let clips = (0..LONG_CLIPS)
                .map(|i| {
                    let mut input_features = Vec::with_capacity(LONG_FRAMES * MELS);
                    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (i as u64 + 1);
                    for r in 0..LONG_FRAMES {
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
                        mask: vec![true; LONG_FRAMES],
                    }
                })
                .collect();
            ("batch_8x30s", clips, Vec::new())
        }
        other => {
            return Err(format!(
                "unknown workload {other:?}; expected one of {WORKLOAD_NAMES:?}"
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
    if got.len() != want.len() {
        return Err(format!("{} values against a reference of {}", got.len(), want.len()));
    }
    let scale = want.iter().map(|v| f64::from(v.abs())).fold(0.0, f64::max);
    let err = got
        .iter()
        .zip(want)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0.0, f64::max);
    Ok(err / scale)
}

/// An observed error as JSON: the number, or `null` where there was no reference.
pub fn json_opt(e: Option<f64>) -> String {
    e.map_or_else(|| "null".to_string(), |e| format!("{e:e}"))
}

/// Fails unless every output is finite and, where the workload has a
/// reference, within `bound` of it (relative to its largest magnitude).
pub fn check(w: &Workload, soft_tokens: &[Vec<f32>], bound: f64) -> Result<Option<f64>, String> {
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
