//! Empirical Challenger Test Suite for Milestone M2:
//! Code Deduplication Across Ojas & Tessl.
//!
//! Evaluates:
//! 1. Numerical bit-identity of AdamW scalar formation between ojas-core and tessl
//!    across boundary steps (1, 10_000, u64::MAX), extreme learning rates, weight decay 0.0,
//!    and edge beta1/beta2.
//! 2. Memory layout correspondence with Metal kernel `kernels/qwen35_adamw_math.h`.
//! 3. Fail-closed rejection hardening for deduplicated safetensors and npy parsers
//!    under adversarial, truncated, oversized, and corrupted inputs.

mod common;

use common::with_gpu;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use tessl::qwen35_adamw::{adamw_step, AdamWHyper};
use tessl::{npy, safetensors};

static SEQ: AtomicUsize = AtomicUsize::new(0);

fn tmp_file(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("tessl-challenger-m2-{}-{tag}-{n}.npy", std::process::id()))
}

/// The legacy (pre-M2) host-side scalar computation in tessl/src/qwen35_adamw.rs.
fn legacy_adamw_scalars(hyper: &AdamWHyper, step: u64, weight_decay: f64) -> [f32; 8] {
    let AdamWHyper {
        lr,
        beta1,
        beta2,
        eps,
        grad_scale,
    } = *hyper;
    let t = step as f64;
    let bc1 = 1.0 - beta1.powf(t);
    let bc2 = 1.0 - beta2.powf(t);
    let step_size = lr / bc1;
    let bc2_sqrt = bc2.powf(0.5);
    [
        (1.0 - lr * weight_decay) as f32,
        (1.0 - beta1) as f32,
        beta2 as f32,
        (1.0 - beta2) as f32,
        step_size as f32,
        bc2_sqrt as f32,
        eps as f32,
        grad_scale as f32,
    ]
}

/// The unified post-M2 host-side scalar computation delegating to ojas_core::AdamWScalars.
fn unified_adamw_scalars(hyper: &AdamWHyper, step: u64, weight_decay: f64) -> Result<[f32; 8], String> {
    let config = ojas_core::AdamWConfig {
        lr: hyper.lr,
        beta1: hyper.beta1,
        beta2: hyper.beta2,
        eps: hyper.eps,
        weight_decay,
    };
    let step_before = step.saturating_sub(1);
    let s = ojas_core::AdamWScalars::new(config, step_before).map_err(|e| format!("AdamWScalars::new error: {e:?}"))?;
    let w = s
        .f32_words("adamw_step")
        .map_err(|e| format!("f32_words error: {e:?}"))?;
    Ok([w[0], w[1], w[2], w[3], w[4], w[5], w[6], hyper.grad_scale as f32])
}

#[test]
fn challenge_adamw_scalar_bit_identity_across_boundary_parameters() {
    let steps: Vec<u64> = vec![
        1,
        2,
        3,
        5,
        10,
        50,
        100,
        500,
        1_000,
        10_000,
        50_000,
        100_000,
        1_000_000,
        10_000_000,
        1_000_000_000,
        (1u64 << 32) - 1,
        (1u64 << 32),
        (1u64 << 48),
        u64::MAX - 1,
        u64::MAX,
    ];
    let lrs: Vec<f64> = vec![0.0, 1e-12, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 10.0, 1e4];
    let wds: Vec<f64> = vec![0.0, 1e-8, 1e-4, 0.01, 0.1, 1.0, 10.0];
    let beta1s: Vec<f64> = vec![0.0, 0.1, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999];
    let beta2s: Vec<f64> = vec![0.0, 0.1, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999];
    let epss: Vec<f64> = vec![1e-16, 1e-12, 1e-8, 1e-5, 1e-3, 1.0];
    let grad_scales: Vec<f64> = vec![0.0, 0.1, 0.5, 1.0, 2.0, 10.0];

    let mut tested_cases = 0usize;
    let mut bit_identical_cases = 0usize;
    let mut max_ulp_gap = 0u32;

    for &step in &steps {
        for &lr in &lrs {
            for &wd in &wds {
                let hyper = AdamWHyper {
                    lr,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    grad_scale: 1.0,
                };
                let legacy = legacy_adamw_scalars(&hyper, step, wd);
                match unified_adamw_scalars(&hyper, step, wd) {
                    Ok(unified) => {
                        tested_cases += 1;
                        let mut match_all = true;
                        for i in 0..8 {
                            let b_leg = legacy[i].to_bits();
                            let b_uni = unified[i].to_bits();
                            let ulp_gap = (b_leg as i64 - b_uni as i64).unsigned_abs() as u32;
                            if ulp_gap > max_ulp_gap {
                                max_ulp_gap = ulp_gap;
                            }
                            if b_leg != b_uni {
                                match_all = false;
                                // At most 1 ULP gap is allowed for libm powf vs binary pow_u64
                                assert!(
                                    ulp_gap <= 1,
                                    "Word [{i}] divergent: legacy={:?} (bits {:#010x}) vs unified={:?} (bits {:#010x}) step={} lr={} wd={} gap={}",
                                    legacy[i], b_leg, unified[i], b_uni, step, lr, wd, ulp_gap
                                );
                            }
                        }
                        if match_all {
                            bit_identical_cases += 1;
                        }
                    }
                    Err(e) => {
                        // If unified refuses, ensure legacy had non-finite
                        assert!(
                            legacy.iter().any(|x| !x.is_finite()) || step == 0,
                            "Unified rejected valid case: {e} step={step} lr={lr} wd={wd}"
                        );
                    }
                }
            }
        }
    }

    // Also sweep beta1, beta2, eps, grad_scale
    for &b1 in &beta1s {
        for &b2 in &beta2s {
            for &eps in &epss {
                for &scale in &grad_scales {
                    let hyper = AdamWHyper {
                        lr: 1e-3,
                        beta1: b1,
                        beta2: b2,
                        eps,
                        grad_scale: scale,
                    };
                    for &step in &[1u64, 10, 1_000, 10_000, u64::MAX] {
                        let legacy = legacy_adamw_scalars(&hyper, step, 0.01);
                        if let Ok(unified) = unified_adamw_scalars(&hyper, step, 0.01) {
                            tested_cases += 1;
                            let mut match_all = true;
                            for i in 0..8 {
                                let b_leg = legacy[i].to_bits();
                                let b_uni = unified[i].to_bits();
                                let ulp_gap = (b_leg as i64 - b_uni as i64).unsigned_abs() as u32;
                                if ulp_gap > max_ulp_gap {
                                    max_ulp_gap = ulp_gap;
                                }
                                if b_leg != b_uni {
                                    match_all = false;
                                    assert!(
                                        ulp_gap <= 1,
                                        "Word [{i}] divergent for b1={} b2={} eps={} scale={} step={}: legacy={} unified={}",
                                        b1, b2, eps, scale, step, legacy[i], unified[i]
                                    );
                                }
                            }
                            if match_all {
                                bit_identical_cases += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    eprintln!(
        "AdamW scalar grid tested: {} configurations. Bit-identical: {} ({:.2}%), Max ULP gap: {}",
        tested_cases,
        bit_identical_cases,
        (bit_identical_cases as f64 / tested_cases as f64) * 100.0,
        max_ulp_gap
    );
    assert!(
        max_ulp_gap <= 1,
        "Max ULP gap between pow_u64 and libm powf exceeds 1 ULP"
    );
}

#[test]
fn challenge_adamw_scalars_memory_layout_alignment_with_metal_header() {
    // Verifies that [f32; 8] matches `struct Qwen35AdamW` in kernels/qwen35_adamw_math.h:
    // struct Qwen35AdamW {
    //     float decay_mul;       // 0: offset 0
    //     float lerp_w;          // 1: offset 4
    //     float beta2;           // 2: offset 8
    //     float one_minus_beta2; // 3: offset 12
    //     float step_size;       // 4: offset 16
    //     float bc2_sqrt;        // 5: offset 20
    //     float eps;             // 6: offset 24
    //     float grad_scale;      // 7: offset 28
    // };
    let hyper = AdamWHyper {
        lr: 0.002,
        beta1: 0.85,
        beta2: 0.98,
        eps: 1e-7,
        grad_scale: 0.75,
    };
    let step = 100u64;
    let wd = 0.05f64;

    let words = unified_adamw_scalars(&hyper, step, wd).expect("valid scalars");
    assert_eq!(words.len(), 8);

    // Byte size must be exactly 32 bytes (8 * 4)
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    assert_eq!(bytes.len(), 32);

    // Verify word 0: decay_mul = 1 - lr * weight_decay
    let want_decay = (1.0 - hyper.lr * wd) as f32;
    assert_eq!(words[0], want_decay);

    // Verify word 1: lerp_w = 1 - beta1
    let want_lerp_w = (1.0 - hyper.beta1) as f32;
    assert_eq!(words[1], want_lerp_w);

    // Verify word 2: beta2
    let want_beta2 = hyper.beta2 as f32;
    assert_eq!(words[2], want_beta2);

    // Verify word 3: one_minus_beta2 = 1 - beta2
    let want_one_minus_beta2 = (1.0 - hyper.beta2) as f32;
    assert_eq!(words[3], want_one_minus_beta2);

    // Verify word 4: step_size = lr / (1 - beta1^step)
    let bc1 = 1.0 - ojas_core::pow_u64(hyper.beta1, step);
    let want_step_size = (hyper.lr / bc1) as f32;
    assert_eq!(words[4], want_step_size);

    // Verify word 5: bc2_sqrt = sqrt(1 - beta2^step)
    let bc2 = 1.0 - ojas_core::pow_u64(hyper.beta2, step);
    let want_bc2_sqrt = (bc2.sqrt()) as f32;
    assert_eq!(words[5], want_bc2_sqrt);

    // Verify word 6: eps
    let want_eps = hyper.eps as f32;
    assert_eq!(words[6], want_eps);

    // Verify word 7: grad_scale
    let want_grad_scale = hyper.grad_scale as f32;
    assert_eq!(words[7], want_grad_scale);
}

#[test]
fn challenge_adamw_on_gpu_across_boundary_steps() {
    with_gpu(|rt| {
        let n = 256;
        let mut p = rt.alloc_tensor_f32(&[n]).unwrap();
        let g = rt.alloc_tensor_f32(&[n]).unwrap();
        let mut m = rt.alloc_tensor_f32(&[n]).unwrap();
        let mut v = rt.alloc_tensor_f32(&[n]).unwrap();

        let init_p = vec![1.5f32; n];
        let init_g = vec![0.02f32; n];
        let init_m = vec![0.0f32; n];
        let init_v = vec![0.0f32; n];

        // 1. Step = 1 (first step, bias correction is largest)
        p.write_f32(&init_p).unwrap();
        g.write_f32(&init_g).unwrap();
        m.write_f32(&init_m).unwrap();
        v.write_f32(&init_v).unwrap();
        let hyper = AdamWHyper::default();
        adamw_step(rt, &mut p, &g, &mut m, &mut v, &hyper, 1, 0.1).unwrap();
        rt.synchronize().unwrap();
        let p_step1 = p.read_f32().unwrap();
        assert!(p_step1.iter().all(|x| x.is_finite()));
        assert!(
            p_step1[0] < 1.5,
            "parameter should decrease after positive gradient step"
        );

        // 2. Step = 10,000 (deep into training)
        adamw_step(rt, &mut p, &g, &mut m, &mut v, &hyper, 10_000, 0.1).unwrap();
        rt.synchronize().unwrap();
        let p_step10k = p.read_f32().unwrap();
        assert!(p_step10k.iter().all(|x| x.is_finite()));

        // 3. Step = u64::MAX (counter upper bound)
        adamw_step(rt, &mut p, &g, &mut m, &mut v, &hyper, u64::MAX, 0.1).unwrap();
        rt.synchronize().unwrap();
        let p_max = p.read_f32().unwrap();
        assert!(p_max.iter().all(|x| x.is_finite()));

        // 4. Weight decay = 0.0 (no decay, only gradient update)
        p.write_f32(&init_p).unwrap();
        adamw_step(rt, &mut p, &g, &mut m, &mut v, &hyper, 2, 0.0).unwrap();
        rt.synchronize().unwrap();
        let p_no_wd = p.read_f32().unwrap();
        assert!(p_no_wd.iter().all(|x| x.is_finite()));

        // 5. Invalid step = 0 must fail-closed immediately without GPU invocation
        let err = adamw_step(rt, &mut p, &g, &mut m, &mut v, &hyper, 0, 0.1).unwrap_err();
        assert!(err.contains("step 0 is refused"), "err: {err}");

        // 6. Invalid non-finite hyperparameter must fail closed
        let bad_hyper = AdamWHyper {
            lr: f64::NAN,
            ..AdamWHyper::default()
        };
        let err2 = adamw_step(rt, &mut p, &g, &mut m, &mut v, &bad_hyper, 1, 0.1).unwrap_err();
        assert!(err2.contains("lr NaN must be finite"), "err: {err2}");
    });
}

#[test]
fn challenge_safetensors_parser_bounds_and_adversarial_rejection() {
    // 1. Zero-byte input
    assert!(safetensors::SafeTensors::from_bytes(vec![]).is_err());

    // 2. Truncated header size (less than 8 bytes)
    assert!(safetensors::SafeTensors::from_bytes(vec![1, 2, 3, 4]).is_err());

    // 3. Huge declared header size (e.g. 1 GiB header declared in a 16-byte file)
    let mut bad_header_len = vec![0u8; 16];
    bad_header_len[..8].copy_from_slice(&(1u64 << 30).to_le_bytes());
    let err = safetensors::SafeTensors::from_bytes(bad_header_len).unwrap_err();
    assert!(
        err.to_string().contains("header") || err.to_string().contains("bounds") || err.to_string().contains("corrupt")
    );

    // 4. Malformed JSON header
    let invalid_json = b"{\"tensor\": { bad json";
    let mut file = vec![0u8; 8 + invalid_json.len()];
    file[..8].copy_from_slice(&(invalid_json.len() as u64).to_le_bytes());
    file[8..].copy_from_slice(invalid_json);
    assert!(safetensors::SafeTensors::from_bytes(file).is_err());

    // 5. Out-of-bounds tensor offset
    // Header claims tensor starts at byte 1000 and ends at 2000, but file ends at 100
    let oob_json = br#"{"t":{"dtype":"F32","shape":[250],"data_offsets":[1000,2000]}}"#;
    let mut oob_file = vec![0u8; 8 + oob_json.len() + 10];
    oob_file[..8].copy_from_slice(&(oob_json.len() as u64).to_le_bytes());
    oob_file[8..8 + oob_json.len()].copy_from_slice(oob_json);
    assert!(safetensors::SafeTensors::from_bytes(oob_file).is_err());

    // 6. Begin offset > End offset
    let inverted_json = br#"{"t":{"dtype":"F32","shape":[10],"data_offsets":[40,0]}}"#;
    let mut inv_file = vec![0u8; 8 + inverted_json.len() + 100];
    inv_file[..8].copy_from_slice(&(inverted_json.len() as u64).to_le_bytes());
    inv_file[8..8 + inverted_json.len()].copy_from_slice(inverted_json);
    assert!(safetensors::SafeTensors::from_bytes(inv_file).is_err());

    // 7. Shape overflow in dimensions: shape [u64::MAX, u64::MAX]
    let overflow_json =
        br#"{"t":{"dtype":"F32","shape":[18446744073709551615,18446744073709551615],"data_offsets":[0,4]}}"#;
    let mut ovf_file = vec![0u8; 8 + overflow_json.len() + 4];
    ovf_file[..8].copy_from_slice(&(overflow_json.len() as u64).to_le_bytes());
    ovf_file[8..8 + overflow_json.len()].copy_from_slice(overflow_json);
    assert!(safetensors::SafeTensors::from_bytes(ovf_file).is_err());
}

#[test]
fn challenge_npy_parser_bounds_and_framing_rejection() {
    let check_err = |bytes: &[u8], tag: &str| -> String {
        let p = tmp_file(tag);
        std::fs::write(&p, bytes).unwrap();
        let r = npy::read_npy(&p);
        let _ = std::fs::remove_file(&p);
        r.expect_err(&format!("expected failure for {tag}"))
    };

    // 1. Truncated magic string
    assert!(
        check_err(b"\x93NUM", "trunc_magic").contains("read header")
            || check_err(b"\x93NUM", "trunc_magic").contains("magic")
    );

    // 2. Invalid magic string
    assert!(check_err(b"BAD_MAGIC_HEADER", "bad_magic").contains("bad magic"));

    // 3. Header len declares more bytes than available
    let mut bad_npy = vec![0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0];
    bad_npy.extend_from_slice(&1000u16.to_le_bytes()); // Header len = 1000
    bad_npy.extend_from_slice(b"{'descr': '<f4', 'fortran_order': False, 'shape': (2,)}"); // Only ~50 bytes
    assert!(check_err(&bad_npy, "hdr_len_exceeds").contains("header"));

    // 4. Header exceeding MAX_NPY_HEADER_BYTES (1 MiB)
    let mut huge_header_npy = vec![0x93, b'N', b'U', b'M', b'P', b'Y', 2, 0]; // Version 2.0 uses u32 header length
    huge_header_npy.extend_from_slice(&(2u32 * 1024 * 1024).to_le_bytes()); // 2 MiB declared header
    assert!(check_err(&huge_header_npy, "huge_header").contains("header length"));

    // 5. Version 1.0 header contains non-ASCII characters (must fail closed)
    let non_ascii_dict = "{'descr': '<f4', 'shape': (1,), 'descr_ext': 'café'}";
    let mut padded = non_ascii_dict.as_bytes().to_vec();
    while (10 + padded.len()) % 16 != 15 {
        padded.push(b' ');
    }
    padded.push(b'\n');
    let mut non_ascii_npy = vec![0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0];
    non_ascii_npy.extend_from_slice(&(padded.len() as u16).to_le_bytes());
    non_ascii_npy.extend_from_slice(&padded);
    non_ascii_npy.extend_from_slice(&[0u8; 4]); // payload
    assert!(check_err(&non_ascii_npy, "non_ascii").contains("ASCII"));

    // 6. Valid header but payload smaller than declared shape (2 f32 elements = 8 bytes, but only 4 bytes supplied)
    let valid_dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,)}";
    let mut v_padded = valid_dict.as_bytes().to_vec();
    while (10 + v_padded.len()) % 16 != 15 {
        v_padded.push(b' ');
    }
    v_padded.push(b'\n');
    let mut truncated_payload = vec![0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0];
    truncated_payload.extend_from_slice(&(v_padded.len() as u16).to_le_bytes());
    truncated_payload.extend_from_slice(&v_padded);
    truncated_payload.extend_from_slice(&[0u8; 4]); // only 4 bytes instead of 8
    assert!(check_err(&truncated_payload, "trunc_payload").contains("follow"));
}
