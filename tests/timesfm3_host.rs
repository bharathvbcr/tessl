//! TimesFM 3's host steps (`tessl::timesfm3_host`) against upstream's own
//! PyTorch decode() on the tiny fixtures: device-free.
//!
//! Stages whose fp32 error is set by conditioning (detrending a steep series,
//! then dividing by a small sigma) are held to upstream's f64 run of the same
//! model: no further from it than upstream's fp32 run is.
//!
//! Every stage is held separately so a mismatch names its step: the patched
//! forward() input, the running RevIN statistics, the residual-block input,
//! the leading-masked counts, and the forecast from the recorded raw head
//! output (CPM refinement, de-normalisation, stitching, trend).

#[path = "common/timesfm3.rs"]
mod tfm;

use tessl::timesfm3::Timesfm3Config;
use tessl::timesfm3_host::{postprocess, prepare, preprocess, update_running_stats, DecodeInput};

fn config() -> Timesfm3Config {
    Timesfm3Config::from_config_file(&tfm::tiny_dir().join("config.json")).expect("tiny config.json")
}

#[test]
fn host_steps_match_upstream_on_every_tiny_case() {
    let cfg = config();
    let hc = &cfg.host;
    let dir = tfm::tiny_dir();
    for case in tfm::TINY_CASES {
        let inp = tfm::decode_input(&dir, case);
        let pr = prepare(hc, &inp).unwrap_or_else(|e| panic!("{case}: {e}"));
        let fv = tfm::load(&dir, case, "fwd_values");
        assert_eq!(
            fv.shape,
            [pr.batch, pr.variates, pr.patches, hc.input_patch_len],
            "{case}: patched shape"
        );
        tfm::assert_no_worse_than_upstream(
            &format!("{case} fwd_values"),
            &pr.values,
            &fv.data,
            &tfm::load64(&dir, case, "fwd_values"),
        );
        assert_eq!(
            pr.masks,
            tfm::mask(&tfm::load(&dir, case, "fwd_masks")),
            "{case}: masks"
        );
        assert_eq!(
            pr.patch_is_target,
            tfm::mask(&tfm::load(&dir, case, "fwd_patch_is_target")),
            "{case}: targets"
        );
        assert_eq!(pr.cpm, tfm::mask(&tfm::load(&dir, case, "fwd_cpm")), "{case}: cpm");

        let pre = preprocess(hc, &pr).unwrap_or_else(|e| panic!("{case}: {e}"));
        for (name, got) in [
            ("revin_mean", &pre.running_mean),
            ("revin_std", &pre.running_std),
            ("resblock_input", &pre.resblock_input),
        ] {
            let want = tfm::load(&dir, case, name);
            tfm::assert_no_worse_than_upstream(
                &format!("{case} {name}"),
                got,
                &want.data,
                &tfm::load64(&dir, case, name),
            );
        }
        let eff = tfm::mask(&tfm::load(&dir, case, "effective_patch_mask"));
        let n = pr.patches;
        let lead: Vec<u32> = eff
            .chunks(n)
            .map(|row| row.iter().take_while(|&&m| m == 1).count() as u32)
            .collect();
        for (r, row) in eff.chunks(n).enumerate() {
            // The effective mask is a prefix by construction (cumprod).
            assert!(
                row.iter().skip(lead[r] as usize).all(|&m| m == 0),
                "{case}: row {r} mask is not a prefix"
            );
        }
        assert_eq!(pre.leading_masked, lead, "{case}: leading masked patches");

        // Post-processing alone: upstream's own raw head output in, so the
        // only differences are the host steps after the head.
        let raw = tfm::load(&dir, case, "raw_logits");
        let out = postprocess(hc, &pr, &pre, &raw.data).unwrap_or_else(|e| panic!("{case}: {e}"));
        tfm::assert_close(
            &format!("{case} forecast from raw logits"),
            &out,
            &tfm::load(&dir, case, "out").data,
            1e-5,
        );
    }
}

#[test]
fn case_b_exercises_the_paths_it_was_built_for() {
    // Guards the fixture, not the code: if a regeneration stopped exercising
    // these, the parity test above would pass vacuously on them.
    let cfg = config();
    let dir = tfm::tiny_dir();
    let inp = tfm::decode_input(&dir, "b_covariates");
    let pr = prepare(&cfg.host, &inp).unwrap();
    assert!(pr.detrended.contains(&1), "no variate detrended");
    assert!(pr.detrended.contains(&0), "every variate detrended");
    let pre = preprocess(&cfg.host, &pr).unwrap();
    assert!(
        pre.leading_masked.iter().any(|&l| l > 0),
        "no leading fully-masked patch"
    );
    assert!(pr.variates > 1 && pr.horizon > cfg.host.output_patch_len);
}

#[test]
fn running_stats_ignore_masked_points_and_merge_exactly() {
    // Folding two halves equals the statistics of the whole.
    let x = [1.0f32, 2.0, 4.0, 8.0, -3.0, 0.5];
    let none = [0u8; 6];
    let (n1, m1, s1) = update_running_stats(0.0, 0.0, 0.0, &x[..3], &none[..3]);
    let (n2, m2, s2) = update_running_stats(n1, m1, s1, &x[3..], &none[3..]);
    let (na, ma, sa) = update_running_stats(0.0, 0.0, 0.0, &x, &none);
    assert_eq!(n2, na);
    assert!((m2 - ma).abs() < 1e-6 && (s2 - sa).abs() < 1e-5, "{m2} {ma} {s2} {sa}");
    // A masked point changes nothing, whatever its value.
    let masked = [0u8, 0, 1, 0, 0, 0];
    let mut y = x;
    y[2] = f32::MAX;
    let a = update_running_stats(0.0, 0.0, 0.0, &x, &masked);
    let b = update_running_stats(0.0, 0.0, 0.0, &y, &masked);
    assert_eq!(a, b);
    assert_eq!(a.0, 5.0);
}

#[test]
fn prepare_refuses_malformed_input_by_name() {
    let cfg = config();
    let ok = DecodeInput {
        batch: 1,
        context: 40,
        horizon: 8,
        num_target: 1,
        target: vec![0.0; 40],
        ..Default::default()
    };
    assert!(prepare(&cfg.host, &ok).is_ok());
    let cases: Vec<(DecodeInput, &str)> = vec![
        (
            DecodeInput {
                horizon: 0,
                ..ok.clone()
            },
            "horizon",
        ),
        (
            DecodeInput {
                target: vec![0.0; 39],
                ..ok.clone()
            },
            "target holds 39",
        ),
        (
            DecodeInput {
                target_mask: Some(vec![0; 3]),
                ..ok.clone()
            },
            "target_mask",
        ),
        (
            DecodeInput {
                past_only: Some(vec![0.0; 40]),
                num_past_only: 0,
                ..ok.clone()
            },
            "zero channels",
        ),
        (
            DecodeInput {
                past_future: Some(vec![0.0; 40]),
                num_past_future: 1,
                ..ok.clone()
            },
            "past_future holds 40",
        ),
        (
            DecodeInput {
                past_only_mask: Some(vec![0; 40]),
                ..ok.clone()
            },
            "past_only_mask without",
        ),
        (DecodeInput { batch: 0, ..ok.clone() }, "non-zero"),
    ];
    for (inp, want) in cases {
        let e = prepare(&cfg.host, &inp).expect_err(want);
        assert!(e.contains(want), "expected {want:?} in {e:?}");
    }
}

#[test]
fn config_refuses_what_the_port_does_not_compute() {
    let text = std::fs::read_to_string(tfm::tiny_dir().join("config.json")).unwrap();
    assert!(Timesfm3Config::from_config_json(&text).is_ok());
    for (from, to, want) in [
        (
            "\"use_frozen_running_stats\": false",
            "\"use_frozen_running_stats\": true",
            "use_frozen_running_stats",
        ),
        (
            "\"ff_activation\": \"relu\"",
            "\"ff_activation\": \"swish\"",
            "ff_activation",
        ),
        ("\"use_rope_var\": false", "\"use_rope_var\": true", "use_rope_var"),
        ("\"num_heads\": 2", "\"num_heads\": 4", "head dim"),
        ("\"v_norm\": \"none\"", "\"v_norm\": \"rms\"", "v_norm"),
    ] {
        assert!(text.contains(from), "tiny config.json lacks {from}");
        let e = Timesfm3Config::from_config_json(&text.replacen(from, to, 1)).expect_err(want);
        assert!(e.contains(want), "expected {want:?} in {e:?}");
    }
}
