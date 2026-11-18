//! `Timesfm3Model` on the real `google/timesfm-3.0-pytorch` checkpoint
//! against upstream's PyTorch decode(). Both live outside the repository (the
//! weights are under the TimesFM Non-Commercial License v1.0, so neither they
//! nor anything computed from them is committed), so the test is ignored by
//! default and fails, never passes, when asked to run without them:
//!
//!   ~/.venvs/ml/bin/python tools/timesfm3_ref/make_reference.py \
//!       --model-dir $TIMESFM3_DIR --out target/timesfm3_ref_real
//!   TIMESFM3_DIR=<snapshot dir> TIMESFM3_REF_DIR=target/timesfm3_ref_real \
//!       cargo test --release --test timesfm3_real -- --ignored --test-threads=1

mod common;
#[path = "common/timesfm3.rs"]
mod tfm;

use std::path::PathBuf;

use common::with_gpu;
use tessl::safetensors::SafeTensors;
use tessl::timesfm3::{Timesfm3Config, Timesfm3Model};
use tessl::timesfm3_host::{prepare, preprocess};

fn env_dir(k: &str) -> PathBuf {
    let p = PathBuf::from(std::env::var_os(k).unwrap_or_else(|| panic!("set {k} (see the module docs)")));
    assert!(p.is_dir(), "{k}={} is not a directory", p.display());
    p
}

#[test]
#[ignore]
fn real_checkpoint_matches_upstream() {
    let (wdir, rdir) = (env_dir("TIMESFM3_DIR"), env_dir("TIMESFM3_REF_DIR"));
    with_gpu(|rt| {
        let cfg = Timesfm3Config::from_config_file(&wdir.join("config.json")).unwrap();
        let st = SafeTensors::open(&wdir.join("model.safetensors")).unwrap();
        let m = Timesfm3Model::load(rt, &st, cfg).unwrap();
        for case in tfm::REAL_CASES {
            let inp = tfm::decode_input(&rdir, case);
            let pr = prepare(&m.config().host, &inp).unwrap();
            let pre = preprocess(&m.config().host, &pr).unwrap();
            let rin = tfm::load(&rdir, case, "resblock_input");
            let out = m.forward(pr.batch, pr.variates, pr.patches, &rin.data, &pre.leading_masked, true).unwrap();
            for l in [0usize, 10, 19] {
                let want = tfm::load(&rdir, case, &format!("layer_{l}"));
                tfm::assert_close(&format!("{case} layer {l}"), &out.trace[l + 1], &want.data, 1e-3);
            }
            tfm::assert_close(&format!("{case} raw logits"), &out.raw_logits, &tfm::load(&rdir, case, "raw_logits").data, 1e-3);
            let f = m.decode(&inp).unwrap();
            tfm::assert_close(&format!("{case} forecast"), &f.values, &tfm::load(&rdir, case, "out").data, 1e-3);
        }
    });
}
