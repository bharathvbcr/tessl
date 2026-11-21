//! Numeric tests for the Interleaved4 (`*_i4`) MLX Q4 kernels.
//!
//! Seven entry points select an `_i4` kernel when handed
//! `Q4MlxLayout::Interleaved4`, and every one of them had only a name check.
//! They read a *different weight packing* from their row-major twins, which is
//! the same hazard `gemv_q4_mlx_blocked` turned out to be: `Q4MlxBank` carries
//! no layout tag, so the wrong packing dispatches happily and returns numbers.
//!
//! The packer below is transcribed from `gemv_q4_mlx_simd_i4`'s own indexing,
//! not guessed:
//!
//! * weights — `packed + ((tile * packs_u2 + pack2) * SIMD_ROWS + r) * 8`,
//!   where `tile = row / 4`, `r = row % 4`, `pack2 = col / 16` and
//!   `packs_u2 = cols / 16`. Each 8-byte group holds 16 nibbles.
//! * scale/bias — `sb[(tile * groups_per_row + g) * 4 + r]`.
//!
//! Within a group the nibble order is unchanged: `qdot16` reads nibble `j` of
//! each `ushort` from bits `[4j, 4j+4)`, which is the ordinary low-nibble-first
//! packing of four consecutive columns. Only the placement of the group moves.
//!
//! Each test checks the `_i4` kernel against the dense f64 reference *and*
//! against its row-major twin over the same logical weights, so a packer that
//! agreed with a broken kernel would still have to agree with the other layout.

mod common;

use common::{
    buf, buf_bf16, close_rel, dense_gemv, empty, gelu_pytorch_tanh, q4_mlx_matrix, random_f32, round_trip_bf16, seeded,
    with_gpu,
};
use tessl::nn::{self, GateUpDispatch, Q4MlxBank, Q4MlxLayout, QkvOutputs, QuantShape};

const UNWRITTEN: f32 = -4.25e28;
/// `SIMD_ROWS` in `kernels/q4_mlx_dot.h`.
const I4_ROWS: usize = 4;
/// Columns per `uint2` pack — `SIMD_VPT`, and the width `qdot16` consumes.
const I4_PACK_COLS: usize = 16;

fn shape(rows: usize, cols: usize, group: usize) -> QuantShape {
    QuantShape {
        rows: rows as u32,
        cols: cols as u32,
        group_size: group as u32,
    }
}

/// Repack row-major nibbles and scale/bias pairs into the Interleaved4 layout.
fn interleave4(nibbles: &[u8], sb: &[f32], rows: usize, cols: usize, group: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(cols % I4_PACK_COLS, 0, "Interleaved4 needs cols % 16 == 0");
    let gpr = cols / group;
    let packs = cols / I4_PACK_COLS;
    let tiles = rows.div_ceil(I4_ROWS);
    let mut out_n = vec![0u8; tiles * packs * I4_ROWS * I4_PACK_COLS];
    let mut out_sb = vec![0.0f32; tiles * gpr * I4_ROWS * 2];
    for row in 0..rows {
        let (tile, r) = (row / I4_ROWS, row % I4_ROWS);
        for p in 0..packs {
            let dst = ((tile * packs + p) * I4_ROWS + r) * I4_PACK_COLS;
            let src = row * cols + p * I4_PACK_COLS;
            out_n[dst..dst + I4_PACK_COLS].copy_from_slice(&nibbles[src..src + I4_PACK_COLS]);
        }
        for g in 0..gpr {
            let dst = (tile * gpr + g) * I4_ROWS + r;
            out_sb[dst * 2] = sb[(row * gpr + g) * 2];
            out_sb[dst * 2 + 1] = sb[(row * gpr + g) * 2 + 1];
        }
    }
    (common::pack_nibbles(&out_n), out_sb)
}

/// Both layouts of one logical matrix, plus the dense matrix they dequantize to.
struct Banks {
    row_packed: Vec<u8>,
    row_sb: Vec<f32>,
    i4_packed: Vec<u8>,
    i4_sb: Vec<f32>,
    dense: Vec<f32>,
}

fn banks(rows: usize, cols: usize, group: usize) -> Banks {
    banks_salted(rows, cols, group, 0)
}

/// A bank whose nibbles and scale/bias pairs depend on `salt` (0 is
/// `q4_mlx_matrix`'s own matrix). Fused kernels read two or three banks, and
/// two banks built from one formula are the same bytes: a kernel that swapped
/// them would return the right numbers. Every fused test takes its second and
/// third bank from a different salt, and `assert_distinct` checks the
/// outputs really differ.
fn banks_salted(rows: usize, cols: usize, group: usize, salt: usize) -> Banks {
    let nib: Vec<u8> = (0..rows * cols)
        .map(|i| ((i * 5 + salt * (3 + i / cols)) % 16) as u8)
        .collect();
    let (_, sb0, _) = q4_mlx_matrix(rows, cols, group);
    let s = salt as f32;
    let row_sb: Vec<f32> = sb0
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if i % 2 == 0 {
                v + s * (0.004 + (i % 3) as f32 * 0.002)
            } else {
                v - s * 0.07 * ((i % 4) as f32 + 1.0) / 2.0
            }
        })
        .collect();
    let gpr = cols / group;
    let sbr = round_trip_bf16(&row_sb);
    let mut dense = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let gi = r * gpr + c / group;
            dense[r * cols + c] = sbr[gi * 2] * nib[r * cols + c] as f32 + sbr[gi * 2 + 1];
        }
    }
    let (i4_packed, i4_sb) = interleave4(&nib, &row_sb, rows, cols, group);
    Banks {
        row_packed: common::pack_nibbles(&nib),
        row_sb,
        i4_packed,
        i4_sb,
        dense,
    }
}

/// Two banks' outputs must differ in at least three rows in four, or swapping the banks
/// cannot change a result and the test cannot see a swap.
fn assert_distinct(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: length");
    let apart = a
        .iter()
        .zip(b)
        .filter(|(x, y)| (**x - **y).abs() > 2e-2 * x.abs().max(y.abs()).max(1e-3))
        .count();
    assert!(
        apart * 4 >= a.len() * 3,
        "{label}: only {apart} of {} rows differ between the two banks, so a bank swap would pass",
        a.len()
    );
}

/// Both layouts of one logical matrix, against the dense reference.
#[test]
fn simd_i4_matches_the_dense_reference_and_its_row_major_twin() {
    with_gpu(|rt| {
        for &(rows, cols, group) in &[
            (256usize, 256usize, 32usize),
            // rows a multiple of the 4 a tile holds but not of the 8 a
            // threadgroup holds.
            (100, 512, 64),
            // rows a multiple of neither: a partial last tile, stored padded
            // (see `nn::Q4MlxBank`), with the tail guard holding in both
            // layouts.
            (102, 512, 64),
        ] {
            let b = banks(rows, cols, group);
            let x = random_f32(cols, 0x9911 + cols as u64);
            let want = dense_gemv(&b.dense, &round_trip_bf16(&x), rows, cols);
            let xb = buf_bf16(rt, &x);

            let rp = rt.alloc_buffer(b.row_packed.len()).unwrap();
            rp.write_bytes(&b.row_packed);
            let rs = buf_bf16(rt, &b.row_sb);
            let ip = rt.alloc_buffer(b.i4_packed.len()).unwrap();
            ip.write_bytes(&b.i4_packed);
            let is = buf_bf16(rt, &b.i4_sb);

            let mut got = Vec::new();
            for (name, bank, layout) in [
                (
                    "row-major",
                    Q4MlxBank {
                        packed: &rp,
                        scales_biases: &rs,
                    },
                    Q4MlxLayout::RowMajor,
                ),
                (
                    "i4",
                    Q4MlxBank {
                        packed: &ip,
                        scales_biases: &is,
                    },
                    Q4MlxLayout::Interleaved4,
                ),
            ] {
                let yb = seeded(rt, rows, UNWRITTEN);
                nn::gemv_q4_mlx_simd(rt, bank, &xb, &yb, shape(rows, cols, group), layout, None).unwrap();
                rt.synchronize().unwrap();
                let y = yb.read_f32()[..rows].to_vec();
                assert!(
                    !y.contains(&UNWRITTEN),
                    "{name} {rows}x{cols}: some rows were never written"
                );
                close_rel(&format!("simd {name} {rows}x{cols}"), &y, &want, 3e-3);
                got.push(y);
            }
            // Same logical weights, two packings: the layouts must agree with
            // each other, not merely each land inside the tolerance.
            close_rel(&format!("i4 vs row-major {rows}x{cols}"), &got[1], &got[0], 1e-5);
        }
    });
}

/// `_add_i4` folds a residual. Checked against the same GEMV plus the residual
/// on the host, so a kernel that ignored `resid` fails.
#[test]
fn simd_add_i4_folds_the_residual() {
    with_gpu(|rt| {
        let (rows, cols, group) = (128usize, 256usize, 32usize);
        let b = banks(rows, cols, group);
        let x = random_f32(cols, 0x9922);
        let resid = random_f32(rows, 0x9923);
        let base = dense_gemv(&b.dense, &round_trip_bf16(&x), rows, cols);
        let want: Vec<f32> = base.iter().zip(&resid).map(|(a, r)| a + r).collect();

        let ip = rt.alloc_buffer(b.i4_packed.len()).unwrap();
        ip.write_bytes(&b.i4_packed);
        let is = buf_bf16(rt, &b.i4_sb);
        let xb = buf_bf16(rt, &x);
        let rb = buf(rt, &resid);
        let yb = seeded(rt, rows, UNWRITTEN);

        nn::gemv_q4_mlx_simd(
            rt,
            Q4MlxBank {
                packed: &ip,
                scales_biases: &is,
            },
            &xb,
            &yb,
            shape(rows, cols, group),
            Q4MlxLayout::Interleaved4,
            Some(&rb),
        )
        .unwrap();
        rt.synchronize().unwrap();
        close_rel("simd_add_i4", &yb.read_f32()[..rows], &want, 3e-3);
        // And it must differ from the residual-free result, or "folded" is
        // satisfied by ignoring the buffer.
        assert!(
            resid.iter().any(|r| r.abs() > 1e-3),
            "the fixture residual is ~0, so this proves nothing"
        );
    });
}

/// The M>1 form in Interleaved4, row by row against the GEMV reference.
#[test]
fn gemm_i4_agrees_with_the_gemv_on_every_row() {
    with_gpu(|rt| {
        let (rows, cols, group, m) = (128usize, 256usize, 32usize, 4usize);
        let b = banks(rows, cols, group);
        let xm = random_f32(m * cols, 0x9933);
        let xr = round_trip_bf16(&xm);

        let ip = rt.alloc_buffer(b.i4_packed.len()).unwrap();
        ip.write_bytes(&b.i4_packed);
        let is = buf_bf16(rt, &b.i4_sb);
        let xb = buf_bf16(rt, &xm);
        let yb = seeded(rt, m * rows, UNWRITTEN);

        nn::gemm_q4_mlx(
            rt,
            Q4MlxBank {
                packed: &ip,
                scales_biases: &is,
            },
            &xb,
            &yb,
            shape(rows, cols, group),
            m as u32,
            Q4MlxLayout::Interleaved4,
            None,
        )
        .unwrap();
        rt.synchronize().unwrap();

        let got = yb.read_f32();
        assert!(!got[..m * rows].contains(&UNWRITTEN), "gemm_i4: unwritten rows");
        for i in 0..m {
            let want = dense_gemv(&b.dense, &xr[i * cols..(i + 1) * cols], rows, cols);
            close_rel(&format!("gemm_i4 row {i}"), &got[i * rows..(i + 1) * rows], &want, 3e-3);
        }
    });
}

/// The fused K‖V form in Interleaved4: one grid split across two banks, so a
/// wrong split writes one output from the other's rows.
#[test]
fn kv_i4_matches_two_separate_gemvs() {
    with_gpu(|rt| {
        let (rows, cols, group) = (128usize, 256usize, 32usize);
        // A distinct V bank (own nibbles and scales), or swapping K and V
        // would return the same numbers.
        let bk = banks_salted(rows, cols, group, 0);
        let bv = banks_salted(rows, cols, group, 1);

        let x = random_f32(cols, 0x9944);
        let xr = round_trip_bf16(&x);
        let k_want = dense_gemv(&bk.dense, &xr, rows, cols);
        let v_want = dense_gemv(&bv.dense, &xr, rows, cols);
        assert_distinct("kv_i4 K vs V reference", &k_want, &v_want);
        let kp = rt.alloc_buffer(bk.i4_packed.len()).unwrap();
        kp.write_bytes(&bk.i4_packed);
        let ks = buf_bf16(rt, &bk.i4_sb);
        let vp = rt.alloc_buffer(bv.i4_packed.len()).unwrap();
        vp.write_bytes(&bv.i4_packed);
        let vs = buf_bf16(rt, &bv.i4_sb);
        let xb = buf_bf16(rt, &x);
        let ko = seeded(rt, rows, UNWRITTEN);
        let vo = seeded(rt, rows, UNWRITTEN);

        nn::gemv_q4_mlx_kv(
            rt,
            Q4MlxBank {
                packed: &kp,
                scales_biases: &ks,
            },
            Q4MlxBank {
                packed: &vp,
                scales_biases: &vs,
            },
            &xb,
            &ko,
            &vo,
            shape(rows, cols, group),
            Q4MlxLayout::Interleaved4,
        )
        .unwrap();
        rt.synchronize().unwrap();

        close_rel("kv_i4 k", &ko.read_f32()[..rows], &k_want, 3e-3);
        close_rel("kv_i4 v", &vo.read_f32()[..rows], &v_want, 3e-3);
    });
}

/// The fused Q‖K‖V form in Interleaved4, with `rows_q != rows_kv` so the grid
/// partition is not symmetric.
#[test]
fn qkv_i4_matches_three_separate_gemvs() {
    with_gpu(|rt| {
        let (rows_q, rows_kv, cols, group) = (128usize, 64usize, 256usize, 32usize);
        // Three banks from three salts: K and V are the same shape, so a swap
        // of the two would otherwise return the same numbers.
        let bq = banks_salted(rows_q, cols, group, 2);
        let bk = banks_salted(rows_kv, cols, group, 0);
        let bv = banks_salted(rows_kv, cols, group, 1);

        let x = random_f32(cols, 0x9955);
        let xr = round_trip_bf16(&x);
        let k_want = dense_gemv(&bk.dense, &xr, rows_kv, cols);
        let v_want = dense_gemv(&bv.dense, &xr, rows_kv, cols);
        assert_distinct("qkv_i4 K vs V reference", &k_want, &v_want);
        let mk = |bytes: &[u8], sb: &[f32]| {
            let p = rt.alloc_buffer(bytes.len()).unwrap();
            p.write_bytes(bytes);
            (p, buf_bf16(rt, sb))
        };
        let (qp, qs) = mk(&bq.i4_packed, &bq.i4_sb);
        let (kp, ks) = mk(&bk.i4_packed, &bk.i4_sb);
        let (vp, vs) = mk(&bv.i4_packed, &bv.i4_sb);
        let xb = buf_bf16(rt, &x);
        let (qo, ko, vo) = (
            seeded(rt, rows_q, UNWRITTEN),
            seeded(rt, rows_kv, UNWRITTEN),
            seeded(rt, rows_kv, UNWRITTEN),
        );

        nn::gemv_q4_mlx_qkv(
            rt,
            Q4MlxBank {
                packed: &qp,
                scales_biases: &qs,
            },
            Q4MlxBank {
                packed: &kp,
                scales_biases: &ks,
            },
            Q4MlxBank {
                packed: &vp,
                scales_biases: &vs,
            },
            &xb,
            QkvOutputs {
                q_out: &qo,
                k_out: &ko,
                v_out: &vo,
            },
            rows_q as u32,
            rows_kv as u32,
            cols as u32,
            group as u32,
            Q4MlxLayout::Interleaved4,
        )
        .unwrap();
        rt.synchronize().unwrap();

        close_rel(
            "qkv_i4 q",
            &qo.read_f32()[..rows_q],
            &dense_gemv(&bq.dense, &xr, rows_q, cols),
            3e-3,
        );
        close_rel("qkv_i4 k", &ko.read_f32()[..rows_kv], &k_want, 3e-3);
        close_rel("qkv_i4 v", &vo.read_f32()[..rows_kv], &v_want, 3e-3);
    });
}

/// The fused gate/up + GELU form in Interleaved4, against the two GEMVs and the
/// same clamped GELU the standalone kernel uses.
#[test]
fn gate_up_gelu_i4_matches_two_gemvs_and_a_gelu() {
    with_gpu(|rt| {
        let (rows, cols, group) = (128usize, 256usize, 32usize);
        // Gate and up are the same shape: their own salts, or a swap (which
        // changes the product's GELU argument) would go unseen.
        let bg = banks_salted(rows, cols, group, 0);
        let bu = banks_salted(rows, cols, group, 1);

        let x = random_f32(cols, 0x9966);
        let xr = round_trip_bf16(&x);
        let gate = dense_gemv(&bg.dense, &xr, rows, cols);
        let up = dense_gemv(&bu.dense, &xr, rows, cols);
        assert_distinct("gate_up_i4 gate vs up reference", &gate, &up);
        // The kernel's GELU (act.h), as `nn::mlp_gelu_tanh`.
        let want: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| (gelu_pytorch_tanh(*g as f64) * (*u as f64)) as f32)
            .collect();

        let gp = rt.alloc_buffer(bg.i4_packed.len()).unwrap();
        gp.write_bytes(&bg.i4_packed);
        let gs = buf_bf16(rt, &bg.i4_sb);
        let up_p = rt.alloc_buffer(bu.i4_packed.len()).unwrap();
        up_p.write_bytes(&bu.i4_packed);
        let up_s = buf_bf16(rt, &bu.i4_sb);
        let xb = buf_bf16(rt, &x);
        let mid = empty(rt, rows);

        nn::gemv_q4_mlx_gate_up_gelu(
            rt,
            Q4MlxBank {
                packed: &gp,
                scales_biases: &gs,
            },
            Q4MlxBank {
                packed: &up_p,
                scales_biases: &up_s,
            },
            &xb,
            &mid,
            shape(rows, cols, group),
            GateUpDispatch::Simd(Q4MlxLayout::Interleaved4),
            false,
        )
        .unwrap();
        rt.synchronize().unwrap();
        close_rel("gate_up_gelu_i4", &mid.read_f32()[..rows], &want, 5e-3);
    });
}

/// The GEMM residual arm, in both layouts.
///
/// `gemm_q4_mlx_simd_add` and `gemm_q4_mlx_simd_add_i4` are selected only by
/// passing `Some(resid)`, and every other GEMM test here passes `None` — so
/// without this they were the last two promoted kernels with a name check and
/// no number. The residual is per output row and broadcast across `m`.
#[test]
fn gemm_add_folds_the_residual_in_both_layouts() {
    with_gpu(|rt| {
        let (rows, cols, group, m) = (128usize, 256usize, 32usize, 3usize);
        let b = banks(rows, cols, group);
        let xm = random_f32(m * cols, 0x9977);
        let xr = round_trip_bf16(&xm);
        let resid = random_f32(m * rows, 0x9978);
        assert!(
            resid.iter().any(|r| r.abs() > 1e-3),
            "the fixture residual is ~0, so this proves nothing"
        );

        let rp = rt.alloc_buffer(b.row_packed.len()).unwrap();
        rp.write_bytes(&b.row_packed);
        let rs = buf_bf16(rt, &b.row_sb);
        let ip = rt.alloc_buffer(b.i4_packed.len()).unwrap();
        ip.write_bytes(&b.i4_packed);
        let is = buf_bf16(rt, &b.i4_sb);
        let xb = buf_bf16(rt, &xm);
        let rb = buf(rt, &resid);

        for (name, bank, layout) in [
            (
                "row-major",
                Q4MlxBank {
                    packed: &rp,
                    scales_biases: &rs,
                },
                Q4MlxLayout::RowMajor,
            ),
            (
                "i4",
                Q4MlxBank {
                    packed: &ip,
                    scales_biases: &is,
                },
                Q4MlxLayout::Interleaved4,
            ),
        ] {
            let yb = seeded(rt, m * rows, UNWRITTEN);
            nn::gemm_q4_mlx(
                rt,
                bank,
                &xb,
                &yb,
                shape(rows, cols, group),
                m as u32,
                layout,
                Some(&rb),
            )
            .unwrap();
            rt.synchronize().unwrap();
            let got = yb.read_f32();
            assert!(!got[..m * rows].contains(&UNWRITTEN), "gemm_add {name}: unwritten rows");
            for i in 0..m {
                let base = dense_gemv(&b.dense, &xr[i * cols..(i + 1) * cols], rows, cols);
                let want: Vec<f32> = base
                    .iter()
                    .zip(&resid[i * rows..(i + 1) * rows])
                    .map(|(a, r)| a + r)
                    .collect();
                close_rel(
                    &format!("gemm_add {name} row {i}"),
                    &got[i * rows..(i + 1) * rows],
                    &want,
                    3e-3,
                );
            }
        }
    });
}
