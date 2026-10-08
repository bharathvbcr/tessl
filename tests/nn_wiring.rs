//! Behavioural tests for the kernels wired into [`tessl::nn`] after the
//! promotion out of `gemma-metal`.
//!
//! Each of these entry points previously existed only as a raw string passed to
//! `pipeline()`, with correctness established end-to-end by whole-model golden
//! parity. That cannot say *which* kernel drifted. These check them singly,
//! against references derived from the kernel sources.

mod common;

use std::sync::Arc;

use common::{random_f32, with_gpu};
use tessl::nn::{self, Q4Bank, Q4MlxBank, Q4MlxLayout, Q4MlxRowVariant, QuantShape};
use tessl::tensor::{f32_slice_to_bf16, GpuBuffer};
use tessl::GpuRuntime;

fn buf(rt: &Arc<GpuRuntime>, data: &[f32]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1) * 4).expect("alloc");
    b.write_f32(data);
    b
}

fn buf_u32(rt: &Arc<GpuRuntime>, data: &[u32]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1) * 4).expect("alloc");
    b.write_u32(data);
    b
}

fn buf_bf16(rt: &Arc<GpuRuntime>, data: &[f32]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1) * 2).expect("alloc");
    b.write_bf16_bits(&f32_slice_to_bf16(data));
    b
}

fn empty(rt: &Arc<GpuRuntime>, elems: usize) -> GpuBuffer {
    let b = rt.alloc_buffer(elems.max(1) * 4).expect("alloc");
    b.zero();
    b
}

/// Pack 4-bit values two to a byte, low nibble first — the layout every Q4
/// kernel here indexes as `packed[i / 2]`, low nibble for even `i`.
fn pack_nibbles(nibbles: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; nibbles.len().div_ceil(2)];
    for (i, n) in nibbles.iter().enumerate() {
        let n = n & 0x0f;
        if i % 2 == 0 {
            out[i / 2] |= n;
        } else {
            out[i / 2] |= n << 4;
        }
    }
    out
}

/// Sign-extend a 4-bit value to the signed range the Q4 kernels use.
///
/// `gemv_q4` and `embed_lookup_q4` both do `int q = (int)(nibble << 28) >> 28`,
/// so a stored nibble of 8 is -8, not 8. The MLX kernels do **not** — they read
/// the nibble unsigned and add a bias. Getting this backwards flips the sign of
/// half the weights, which is exactly what this reference is here to catch.
fn i4(nibble: u8) -> f32 {
    ((((nibble & 0x0f) as i8) << 4) >> 4) as f32
}

fn close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}]: got {g} want {w} (tol {tol})"
        );
    }
}

// -------------------------------------------------------------- Sampling ---

#[test]
fn softcap_logits_matches_the_tanh_reference() {
    with_gpu(|rt| {
        let n = 512usize;
        // Span the range where a fast tanh would misbehave.
        let mut logits: Vec<f32> = (0..n).map(|i| (i as f32 - 256.0) * 0.5).collect();
        // Softcapping exists specifically to tame unbounded logits. Metal's
        // tanh implementation overflows internally without an explicit clamp
        // around this range, turning an otherwise finite row into NaNs.
        logits[0] = -1.0e6;
        logits[n - 1] = 1.0e6;
        let lb = buf(rt, &logits);
        let cap = buf(rt, &[30.0]);

        nn::softcap_logits(rt, &lb, &cap, n as u32).unwrap();
        rt.synchronize().unwrap();

        let want: Vec<f32> = logits.iter().map(|v| 30.0 * (v / 30.0).tanh()).collect();
        let got = lb.read_f32();
        assert!(
            got[..n].iter().all(|v| v.is_finite()),
            "softcap produced non-finite"
        );
        close("softcap_logits", &got[..n], &want, 1e-4);
    });
}

#[test]
fn invalid_device_softcap_is_a_noop_instead_of_poisoning_logits() {
    with_gpu(|rt| {
        let logits = [-1.0e6, -1.0, 0.0, 1.0, 1.0e6];
        for (label, invalid_cap) in [
            ("zero", 0.0),
            ("negative", -30.0),
            ("infinite", f32::INFINITY),
            ("nan", f32::NAN),
        ] {
            let lb = buf(rt, &logits);
            let cap = buf(rt, &[invalid_cap]);

            nn::softcap_logits(rt, &lb, &cap, logits.len() as u32).unwrap();
            rt.synchronize().unwrap();

            let got = lb.read_f32();
            assert_eq!(
                &got[..logits.len()],
                &logits,
                "invalid {label} softcap must leave finite logits unchanged"
            );
        }
    });
}

#[test]
fn softcap_argmax_one_pass_finds_the_max_over_a_large_vocab() {
    with_gpu(|rt| {
        // Larger than any single threadgroup, so the strided scan is exercised.
        let n = 40_000usize;
        let mut logits = random_f32(n, 0x91);
        let winner = 31_337usize;
        logits[winner] = 99.0;
        let lb = buf(rt, &logits);
        let out = buf_u32(rt, &[0]);
        let cap = buf(rt, &[30.0]);

        nn::softcap_argmax_one_pass(rt, &lb, &out, &cap, n as u32).unwrap();
        rt.synchronize().unwrap();
        assert_eq!(out.read_u32()[0] as usize, winner);

        // And it must not have rewritten the logits — that is the documented
        // difference from softcap_sample.
        close("logits untouched", &lb.read_f32()[..n], &logits, 0.0);
    });
}

#[test]
fn softcap_sample_writes_the_argmax_and_rewrites_logits() {
    with_gpu(|rt| {
        let n = 100usize;
        let mut logits = random_f32(n, 0xA1);
        logits[57] = 50.0;
        let lb = buf(rt, &logits);
        let out = buf_u32(rt, &[0]);
        let cap = buf(rt, &[30.0]);

        nn::softcap_sample(rt, &lb, &out, &cap, n as u32).unwrap();
        rt.synchronize().unwrap();

        assert_eq!(out.read_u32()[0], 57);
        let got = lb.read_f32();
        let want: Vec<f32> = logits.iter().map(|v| 30.0 * (v / 30.0).tanh()).collect();
        close("softcap_sample rewrote logits", &got[..n], &want, 1e-4);
    });
}

#[test]
fn softcap_sample_refuses_more_logits_than_its_threadgroup_reduces() {
    with_gpu(|rt| {
        let n = 4096u32;
        let lb = empty(rt, n as usize);
        let out = buf_u32(rt, &[0]);
        let cap = buf(rt, &[30.0]);
        // Silently reducing only the first 256 would return a plausible token.
        let err = nn::softcap_sample(rt, &lb, &out, &cap, n).expect_err("n too large");
        assert!(err.contains("exceeds"), "unexpected error: {err:?}");
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}

#[test]
fn argmax_f32_pass_reduces_a_full_vocab_across_two_passes() {
    with_gpu(|rt| {
        let n = 50_000u32;
        let mut logits = random_f32(n as usize, 0xB1);
        let winner = 44_444usize;
        logits[winner] = 77.0;
        let lb = buf(rt, &logits);
        // Softcap of 0 disables capping in the kernel's `softcap > 0` guard.
        let cap = buf(rt, &[0.0]);

        let g1 = nn::argmax_pass_groups(n);
        let idx1 = buf_u32(rt, &vec![0u32; g1]);
        let val1 = empty(rt, g1);
        nn::argmax_f32_pass(rt, &lb, &idx1, &val1, None, &cap, n).unwrap();

        // Second pass folds the partials, carrying original indices through.
        let g2 = nn::argmax_pass_groups(g1 as u32);
        let idx2 = buf_u32(rt, &vec![0u32; g2]);
        let val2 = empty(rt, g2);
        nn::argmax_f32_pass(rt, &val1, &idx2, &val2, Some(&idx1), &cap, g1 as u32).unwrap();
        rt.synchronize().unwrap();

        assert_eq!(g2, 1, "test assumes the second pass collapses to one group");
        assert_eq!(idx2.read_u32()[0] as usize, winner);
        assert!((val2.read_f32()[0] - 77.0).abs() < 1e-4);
    });
}

// ---------------------------------------------------- Embedding lookup ---

#[test]
fn embed_lookup_q4_gathers_dequantized_rows_and_zeroes_out_of_range_tokens() {
    with_gpu(|rt| {
        let (vocab, hidden, group) = (64usize, 32usize, 16usize);
        let groups = vocab * (hidden / group);
        let nibbles: Vec<u8> = (0..vocab * hidden).map(|i| (i % 16) as u8).collect();
        let packed_bytes = pack_nibbles(&nibbles);
        let scales: Vec<f32> = (0..groups).map(|i| 0.01 + (i % 5) as f32 * 0.002).collect();
        let zeros: Vec<f32> = (0..groups).map(|i| (i % 3) as f32).collect();

        let pb = rt.alloc_buffer(packed_bytes.len()).unwrap();
        pb.write_bytes(&packed_bytes);
        let sb = buf(rt, &scales);
        let zb = buf(rt, &zeros);
        // Last id is out of range and must yield a zero row, not a stray read.
        let tokens = [3u32, 17, 63, vocab as u32 + 5];
        let tb = buf_u32(rt, &tokens);
        let out = empty(rt, tokens.len() * hidden);

        nn::embed_lookup_q4(
            rt,
            Q4Bank {
                packed: &pb,
                scales: &sb,
                zeros: &zb,
            },
            &tb,
            &out,
            vocab as u32,
            hidden as u32,
            group as u32,
            tokens.len() as u32,
        )
        .unwrap();
        rt.synchronize().unwrap();

        let mut want = vec![0.0f32; tokens.len() * hidden];
        for (m, &tid) in tokens.iter().enumerate() {
            if tid as usize >= vocab {
                continue; // stays zero
            }
            for d in 0..hidden {
                let gi = tid as usize * (hidden / group) + d / group;
                let idx = tid as usize * hidden + d;
                want[m * hidden + d] = scales[gi] * (i4(nibbles[idx]) - zeros[gi]);
            }
        }
        close(
            "embed_lookup_q4",
            &out.read_f32()[..want.len()],
            &want,
            1e-5,
        );
    });
}

// ------------------------------------------------------------- Q4 GEMV ---

/// Build a Q4 matrix and its f32 dequantization, sharing one source of truth.
fn q4_matrix(rows: usize, cols: usize, group: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let nibbles: Vec<u8> = (0..rows * cols).map(|i| ((i * 7) % 16) as u8).collect();
    let groups = rows * (cols / group);
    let scales: Vec<f32> = (0..groups).map(|i| 0.02 + (i % 9) as f32 * 0.001).collect();
    let zeros: Vec<f32> = (0..groups).map(|i| (i % 4) as f32 - 1.0).collect();
    let mut dense = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let gi = r * (cols / group) + c / group;
            dense[r * cols + c] = scales[gi] * (i4(nibbles[r * cols + c]) - zeros[gi]);
        }
    }
    (pack_nibbles(&nibbles), scales, zeros, dense)
}

#[test]
fn gemv_q4_matches_a_dense_dequantized_reference() {
    with_gpu(|rt| {
        let (rows, cols, group) = (96usize, 128usize, 32usize);
        let (packed, scales, zeros, dense) = q4_matrix(rows, cols, group);
        let x = random_f32(cols, 0xC1);

        let pb = rt.alloc_buffer(packed.len()).unwrap();
        pb.write_bytes(&packed);
        let sb = buf(rt, &scales);
        let zb = buf(rt, &zeros);
        let xb = buf(rt, &x);
        let yb = empty(rt, rows);

        nn::gemv_q4(
            rt,
            Q4Bank {
                packed: &pb,
                scales: &sb,
                zeros: &zb,
            },
            &xb,
            &yb,
            QuantShape {
                rows: rows as u32,
                cols: cols as u32,
                group_size: group as u32,
            },
            false,
        )
        .unwrap();
        rt.synchronize().unwrap();

        let want: Vec<f32> = (0..rows)
            .map(|r| (0..cols).map(|c| dense[r * cols + c] * x[c]).sum())
            .collect();
        close("gemv_q4", &yb.read_f32()[..rows], &want, 2e-3);
    });
}

/// MLX banks store `(scale, bias)` adjacent as a bfloat2 and dequantize with an
/// **add**: `w = scale * nibble + bias`.
fn q4_mlx_matrix(rows: usize, cols: usize, group: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let nibbles: Vec<u8> = (0..rows * cols).map(|i| ((i * 5) % 16) as u8).collect();
    let groups = rows * (cols / group);
    // Round scales and biases through bf16 so the reference sees exactly the
    // values the kernel loads, not their f32 originals.
    let sb_f32: Vec<f32> = (0..groups * 2)
        .map(|i| {
            if i % 2 == 0 {
                0.03 + (i % 7) as f32 * 0.002
            } else {
                (i % 5) as f32 * 0.1 - 0.2
            }
        })
        .collect();
    let sb_bits = f32_slice_to_bf16(&sb_f32);
    let sb_round: Vec<f32> = sb_bits
        .iter()
        .map(|b| tessl::tensor::bf16_bits_to_f32(*b))
        .collect();
    let mut dense = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let gi = r * (cols / group) + c / group;
            dense[r * cols + c] =
                sb_round[gi * 2] * nibbles[r * cols + c] as f32 + sb_round[gi * 2 + 1];
        }
    }
    (pack_nibbles(&nibbles), sb_f32, dense)
}

#[test]
fn gemv_q4_mlx_matches_a_dense_dequantized_reference() {
    with_gpu(|rt| {
        let (rows, cols, group) = (64usize, 128usize, 32usize);
        let (packed, sb_f32, dense) = q4_mlx_matrix(rows, cols, group);
        let x = random_f32(cols, 0xD1);

        let pb = rt.alloc_buffer(packed.len()).unwrap();
        pb.write_bytes(&packed);
        let sbb = buf_bf16(rt, &sb_f32);
        let xb = buf(rt, &x);
        let yb = empty(rt, rows);

        nn::gemv_q4_mlx(
            rt,
            Q4MlxBank {
                packed: &pb,
                scales_biases: &sbb,
            },
            &xb,
            &yb,
            QuantShape {
                rows: rows as u32,
                cols: cols as u32,
                group_size: group as u32,
            },
            Q4MlxRowVariant::Standard,
        )
        .unwrap();
        rt.synchronize().unwrap();

        let want: Vec<f32> = (0..rows)
            .map(|r| (0..cols).map(|c| dense[r * cols + c] * x[c]).sum())
            .collect();
        close("gemv_q4_mlx", &yb.read_f32()[..rows], &want, 2e-3);
    });
}

#[test]
fn gemv_q4_mlx_simd_matches_the_same_reference_on_a_bf16_activation() {
    with_gpu(|rt| {
        let (rows, cols, group) = (64usize, 256usize, 32usize);
        let (packed, sb_f32, dense) = q4_mlx_matrix(rows, cols, group);
        let x = random_f32(cols, 0xE1);
        // The kernel reads bf16 activations; the reference must see the same.
        let x_round: Vec<f32> = f32_slice_to_bf16(&x)
            .iter()
            .map(|b| tessl::tensor::bf16_bits_to_f32(*b))
            .collect();

        let pb = rt.alloc_buffer(packed.len()).unwrap();
        pb.write_bytes(&packed);
        let sbb = buf_bf16(rt, &sb_f32);
        let xb = buf_bf16(rt, &x);
        let yb = empty(rt, rows);

        nn::gemv_q4_mlx_simd(
            rt,
            Q4MlxBank {
                packed: &pb,
                scales_biases: &sbb,
            },
            &xb,
            &yb,
            QuantShape {
                rows: rows as u32,
                cols: cols as u32,
                group_size: group as u32,
            },
            Q4MlxLayout::RowMajor,
            None,
        )
        .unwrap();
        rt.synchronize().unwrap();

        let want: Vec<f32> = (0..rows)
            .map(|r| (0..cols).map(|c| dense[r * cols + c] * x_round[c]).sum())
            .collect();
        close("gemv_q4_mlx_simd", &yb.read_f32()[..rows], &want, 5e-3);
    });
}

#[test]
fn gemv_q4_mlx_simd_add_folds_the_residual() {
    with_gpu(|rt| {
        let (rows, cols, group) = (32usize, 256usize, 32usize);
        let (packed, sb_f32, dense) = q4_mlx_matrix(rows, cols, group);
        let x = random_f32(cols, 0xF1);
        let x_round: Vec<f32> = f32_slice_to_bf16(&x)
            .iter()
            .map(|b| tessl::tensor::bf16_bits_to_f32(*b))
            .collect();
        let resid = random_f32(rows, 0xF2);

        let pb = rt.alloc_buffer(packed.len()).unwrap();
        pb.write_bytes(&packed);
        let sbb = buf_bf16(rt, &sb_f32);
        let xb = buf_bf16(rt, &x);
        let yb = empty(rt, rows);
        let rb = buf(rt, &resid);

        nn::gemv_q4_mlx_simd(
            rt,
            Q4MlxBank {
                packed: &pb,
                scales_biases: &sbb,
            },
            &xb,
            &yb,
            QuantShape {
                rows: rows as u32,
                cols: cols as u32,
                group_size: group as u32,
            },
            Q4MlxLayout::RowMajor,
            Some(&rb),
        )
        .unwrap();
        rt.synchronize().unwrap();

        let want: Vec<f32> = (0..rows)
            .map(|r| {
                resid[r]
                    + (0..cols)
                        .map(|c| dense[r * cols + c] * x_round[c])
                        .sum::<f32>()
            })
            .collect();
        close("gemv_q4_mlx_simd_add", &yb.read_f32()[..rows], &want, 5e-3);
    });
}

// ------------------------------------------------------------- Guards ---

#[test]
fn quant_shape_refuses_a_ragged_final_group() {
    with_gpu(|rt| {
        let b = empty(rt, 4096);
        let bad = QuantShape {
            rows: 32,
            cols: 100,
            group_size: 32,
        };
        let err = nn::gemv_q4(
            rt,
            Q4Bank {
                packed: &b,
                scales: &b,
                zeros: &b,
            },
            &b,
            &b,
            bad,
            false,
        )
        .expect_err("ragged group");
        assert!(err.contains("not a multiple"), "unexpected: {err:?}");

        let zero_group = QuantShape {
            rows: 32,
            cols: 128,
            group_size: 0,
        };
        let err = nn::gemv_q4(
            rt,
            Q4Bank {
                packed: &b,
                scales: &b,
                zeros: &b,
            },
            &b,
            &b,
            zero_group,
            false,
        )
        .expect_err("zero group_size");
        assert!(err.contains("non-zero"), "unexpected: {err:?}");
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}

#[test]
fn gemm_q4_mlx_refuses_an_m_the_kernel_would_silently_truncate() {
    with_gpu(|rt| {
        let b = empty(rt, 8192);
        let shape = QuantShape {
            rows: 64,
            cols: 128,
            group_size: 32,
        };
        // GEMM_MAX_M is 8; the kernel takes min(M, 8) and leaves rows 8.. alone.
        let err = nn::gemm_q4_mlx(
            rt,
            Q4MlxBank {
                packed: &b,
                scales_biases: &b,
            },
            &b,
            &b,
            shape,
            16,
            Q4MlxLayout::RowMajor,
            None,
        )
        .expect_err("M past GEMM_MAX_M");
        assert!(err.contains("GEMM_MAX_M"), "unexpected: {err:?}");
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}

#[test]
fn attention_refuses_head_counts_that_do_not_group() {
    with_gpu(|rt| {
        let b = empty(rt, 1 << 16);
        let u = buf_u32(rt, &[0]);
        let dims = nn::AttnDims {
            batch: 1,
            tq: 1,
            heads: 7,
            heads_kv: 2, // 7 is not a multiple of 2
            window: 64,
            scale: 0.088,
        };
        let err = nn::flash_attn_swa(rt, nn::AttnHeadDim::D128, &b, &b, &b, &b, &u, &u, &u, dims)
            .expect_err("ragged GQA grouping");
        assert!(err.contains("multiple of heads_kv"), "unexpected: {err:?}");

        let bad_scale = nn::AttnDims {
            heads: 8,
            scale: f32::NAN,
            ..dims
        };
        let err = nn::flash_attn_swa(
            rt,
            nn::AttnHeadDim::D128,
            &b,
            &b,
            &b,
            &b,
            &u,
            &u,
            &u,
            bad_scale,
        )
        .expect_err("non-finite scale");
        assert!(err.contains("scale must be finite"), "unexpected: {err:?}");
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}

#[test]
fn qkv_rope_refuses_a_variant_operand_mismatch() {
    with_gpu(|rt| {
        let b = empty(rt, 4096);
        let u = buf_u32(rt, &[0]);
        let qkv = nn::QkvBuffers {
            q: &b,
            k: &b,
            v: &b,
            q_weight: &b,
            k_weight: &b,
            v_weight: &b,
        };
        let dims = nn::QkvRopeDims {
            t: 1,
            heads_q: 4,
            heads_kv: 2,
            head_dim: 64,
            rotary_dim: 64,
            theta: 10_000.0,
            eps: 1e-6,
        };

        // A PosBuffer variant with no buffer would read a stale position for a
        // whole session if it were quietly accepted.
        let err = nn::rms_qkv_rope(
            rt,
            nn::QkvRopeVariant::PosBuffer,
            qkv,
            dims,
            0,
            None,
            None,
            false,
        )
        .expect_err("missing pos buffer");
        assert!(
            err.contains("require pos_offset_buf"),
            "unexpected: {err:?}"
        );

        // And the reverse: a constant-offset variant handed a buffer.
        let err = nn::rms_qkv_rope(
            rt,
            nn::QkvRopeVariant::PosConst,
            qkv,
            dims,
            0,
            Some(&u),
            None,
            false,
        )
        .expect_err("buffer on the const variant");
        assert!(err.contains("not a buffer"), "unexpected: {err:?}");

        // rotary_dim past head_dim rotates off the end of every head.
        let long_rope = nn::QkvRopeDims {
            rotary_dim: 128,
            ..dims
        };
        let err = nn::rms_qkv_rope(
            rt,
            nn::QkvRopeVariant::PosConst,
            qkv,
            long_rope,
            0,
            None,
            None,
            false,
        )
        .expect_err("rotary_dim > head_dim");
        assert!(err.contains("exceeds head_dim"), "unexpected: {err:?}");

        // RoPE rotates pairs, so an odd span is always a caller mistake.
        let odd_rope = nn::QkvRopeDims {
            rotary_dim: 63,
            ..dims
        };
        let err = nn::rms_qkv_rope(
            rt,
            nn::QkvRopeVariant::PosConst,
            qkv,
            odd_rope,
            0,
            None,
            None,
            false,
        )
        .expect_err("odd rotary_dim");
        assert!(err.contains("is odd"), "unexpected: {err:?}");
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}

/// `gemv_q4` has no `cols` ceiling.
///
/// Until 2026-09-05 the row kernel staged all of `x` in dynamic threadgroup
/// memory, so `cols * 4` bytes above the device limit were refused before
/// encoding (and a `cols` that was not a multiple of four failed later with an
/// alignment message naming neither the kernel nor `cols`, audit N15). The
/// simdgroup kernel reads `x` from device memory, so a width the old kernel
/// could not launch now validates, dispatches, and computes the right numbers.
#[test]
fn gemv_q4_has_no_cols_ceiling() {
    with_gpu(|rt| {
        let limit = rt.max_threadgroup_memory();
        let group_size = 64usize;
        // Wider than the old cache could hold, rounded up to a whole group.
        let cols = (limit / std::mem::size_of::<f32>() + 1).div_ceil(group_size) * group_size;
        let rows = 4usize;
        assert!(cols * 4 > limit, "test setup must exceed the old ceiling");
        let groups = rows * (cols / group_size);
        let q: Vec<i8> = (0..rows * cols)
            .map(|i| ((i * 13 + 5) % 16) as i8 - 8)
            .collect();
        let packed_bytes: Vec<u8> = q
            .chunks(2)
            .map(|p| ((p[0] as u8) & 0x0f) | (((p[1] as u8) & 0x0f) << 4))
            .collect();
        let scales: Vec<f32> = (0..groups).map(|i| 0.01 + (i % 3) as f32 * 0.005).collect();
        let zeros: Vec<f32> = (0..groups).map(|i| (i % 4) as f32 * 0.5 - 1.0).collect();
        let x: Vec<f32> = (0..cols)
            .map(|c| ((c * 7 % 23) as f32 - 11.0) * 0.1)
            .collect();
        let want: Vec<f32> = (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| {
                        let g = r * (cols / group_size) + c / group_size;
                        scales[g] as f64 * (q[r * cols + c] as f64 - zeros[g] as f64) * x[c] as f64
                    })
                    .sum::<f64>() as f32
            })
            .collect();

        let packed = rt
            .alloc_buffer(packed_bytes.len())
            .expect("packed allocation");
        packed.write_bytes(&packed_bytes);
        let sb = empty(rt, groups);
        sb.write_f32(&scales);
        let zb = empty(rt, groups);
        zb.write_f32(&zeros);
        let xb = empty(rt, cols);
        xb.write_f32(&x);
        let yb = empty(rt, rows);
        let shape = QuantShape {
            rows: rows as u32,
            cols: cols as u32,
            group_size: group_size as u32,
        };
        nn::gemv_q4(
            rt,
            Q4Bank {
                packed: &packed,
                scales: &sb,
                zeros: &zb,
            },
            &xb,
            &yb,
            shape,
            false,
        )
        .expect("a wide row is an ordinary dispatch now");
        rt.synchronize().unwrap();
        let got = yb.read_f32();
        for r in 0..rows {
            let tol = 2e-4 * want[r].abs().max(1.0);
            assert!(
                (got[r] - want[r]).abs() <= tol,
                "gemv_q4 cols={cols} row {r}: got {} want {}",
                got[r],
                want[r]
            );
        }
    });
}

/// The softcap belongs to the first pass only. Later passes reduce values that
/// are already capped, and `tanh` is not idempotent: capping a partial maximum
/// again shrinks it (`30·tanh(29.65/30) ≈ 22.7` instead of `29.65`), so a
/// two-pass argmax returned a wrong value whenever the cap was on.
#[test]
fn argmax_f32_pass_applies_the_softcap_once_across_passes() {
    with_gpu(|rt| {
        let n = 50_000u32;
        let mut logits = random_f32(n as usize, 0xB2);
        let winner = 12_345usize;
        logits[winner] = 77.0;
        let lb = buf(rt, &logits);
        let softcap = 30.0f32;
        let cap = buf(rt, &[softcap]);

        let g1 = nn::argmax_pass_groups(n);
        let idx1 = buf_u32(rt, &vec![0u32; g1]);
        let val1 = empty(rt, g1);
        nn::argmax_f32_pass(rt, &lb, &idx1, &val1, None, &cap, n).unwrap();

        let g2 = nn::argmax_pass_groups(g1 as u32);
        assert_eq!(g2, 1, "test assumes the second pass collapses to one group");
        let idx2 = buf_u32(rt, &vec![0u32; g2]);
        let val2 = empty(rt, g2);
        nn::argmax_f32_pass(rt, &val1, &idx2, &val2, Some(&idx1), &cap, g1 as u32).unwrap();
        rt.synchronize().unwrap();

        let want = softcap * (77.0f32 / softcap).tanh();
        assert_eq!(idx2.read_u32()[0] as usize, winner);
        let got = val2.read_f32()[0];
        assert!(
            (got - want).abs() < 1e-3,
            "two-pass softcapped max: got {got}, want {want} (capping twice gives {})",
            softcap * (want / softcap).tanh()
        );
    });
}
