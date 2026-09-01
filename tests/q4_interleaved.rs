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
    buf, buf_bf16, close_rel, dense_gemv, empty, q4_mlx_matrix, random_f32, round_trip_bf16,
    seeded, with_gpu,
};
use tessl::nn::{self, GateUpDispatch, Q4MlxBank, Q4MlxLayout, QkvOutputs, QuantShape};

const UNWRITTEN: f32 = -4.25e28;
/// `SIMD_ROWS` in `kernels/gemv_q4_mlx.metal`.
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

/// Raw nibbles for the matrix `q4_mlx_matrix` describes, so both layouts
/// dequantize to the same dense values.
fn nibbles_for(rows: usize, cols: usize) -> Vec<u8> {
    (0..rows * cols).map(|i| ((i * 5) % 16) as u8).collect()
}

/// Repack row-major nibbles and scale/bias pairs into the Interleaved4 layout.
fn interleave4(
    nibbles: &[u8],
    sb: &[f32],
    rows: usize,
    cols: usize,
    group: usize,
) -> (Vec<u8>, Vec<f32>) {
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

struct Banks {
    row_packed: Vec<u8>,
    row_sb: Vec<f32>,
    i4_packed: Vec<u8>,
    i4_sb: Vec<f32>,
    dense: Vec<f32>,
}

fn banks(rows: usize, cols: usize, group: usize) -> Banks {
    let nib = nibbles_for(rows, cols);
    let (row_packed, row_sb, dense) = q4_mlx_matrix(rows, cols, group);
    let (i4_packed, i4_sb) = interleave4(&nib, &row_sb, rows, cols, group);
    Banks {
        row_packed,
        row_sb,
        i4_packed,
        i4_sb,
        dense,
    }
}

/// Both layouts of one logical matrix, against the dense reference.
#[test]
fn simd_i4_matches_the_dense_reference_and_its_row_major_twin() {
    with_gpu(|rt| {
        for &(rows, cols, group) in &[
            (256usize, 256usize, 32usize),
            // rows not a multiple of the 4 a tile holds, and of the 8 a
            // threadgroup holds: the tail guard has to hold in both layouts.
            (100, 512, 64),
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
                nn::gemv_q4_mlx_simd(rt, bank, &xb, &yb, shape(rows, cols, group), layout, None)
                    .unwrap();
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
            close_rel(
                &format!("i4 vs row-major {rows}x{cols}"),
                &got[1],
                &got[0],
                1e-5,
            );
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
        assert!(
            !got[..m * rows].contains(&UNWRITTEN),
            "gemm_i4: unwritten rows"
        );
        for i in 0..m {
            let want = dense_gemv(&b.dense, &xr[i * cols..(i + 1) * cols], rows, cols);
            close_rel(
                &format!("gemm_i4 row {i}"),
                &got[i * rows..(i + 1) * rows],
                &want,
                3e-3,
            );
        }
    });
}

/// The fused K‖V form in Interleaved4: one grid split across two banks, so a
/// wrong split writes one output from the other's rows.
#[test]
fn kv_i4_matches_two_separate_gemvs() {
    with_gpu(|rt| {
        let (rows, cols, group) = (128usize, 256usize, 32usize);
        let bk = banks(rows, cols, group);
        // A distinct V bank, or the two outputs cannot be told apart.
        let nv = nibbles_for(rows + 4, cols);
        let (_, sv, dv) = q4_mlx_matrix(rows + 4, cols, group);
        let (ivp, ivs) = interleave4(
            &nv[..rows * cols],
            &sv[..rows * (cols / group) * 2],
            rows,
            cols,
            group,
        );
        let dv = dv[..rows * cols].to_vec();

        let x = random_f32(cols, 0x9944);
        let xr = round_trip_bf16(&x);
        let kp = rt.alloc_buffer(bk.i4_packed.len()).unwrap();
        kp.write_bytes(&bk.i4_packed);
        let ks = buf_bf16(rt, &bk.i4_sb);
        let vp = rt.alloc_buffer(ivp.len()).unwrap();
        vp.write_bytes(&ivp);
        let vs = buf_bf16(rt, &ivs);
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

        close_rel(
            "kv_i4 k",
            &ko.read_f32()[..rows],
            &dense_gemv(&bk.dense, &xr, rows, cols),
            3e-3,
        );
        close_rel(
            "kv_i4 v",
            &vo.read_f32()[..rows],
            &dense_gemv(&dv, &xr, rows, cols),
            3e-3,
        );
    });
}

/// The fused Q‖K‖V form in Interleaved4, with `rows_q != rows_kv` so the grid
/// partition is not symmetric.
#[test]
fn qkv_i4_matches_three_separate_gemvs() {
    with_gpu(|rt| {
        let (rows_q, rows_kv, cols, group) = (128usize, 64usize, 256usize, 32usize);
        let bq = banks(rows_q, cols, group);
        let bk = banks(rows_kv, cols, group);
        // Perturb V's scales so it is not a copy of K.
        let nv = nibbles_for(rows_kv, cols);
        let (_, sv0, _) = q4_mlx_matrix(rows_kv, cols, group);
        let sv: Vec<f32> = sv0
            .iter()
            .enumerate()
            .map(|(i, v)| v + (i % 3) as f32 * 0.01)
            .collect();
        let gpr = cols / group;
        let mut dv = vec![0.0f32; rows_kv * cols];
        let svr = round_trip_bf16(&sv);
        for r in 0..rows_kv {
            for c in 0..cols {
                let gi = r * gpr + c / group;
                dv[r * cols + c] = svr[gi * 2] * nv[r * cols + c] as f32 + svr[gi * 2 + 1];
            }
        }
        let (ivp, ivs) = interleave4(&nv, &sv, rows_kv, cols, group);

        let x = random_f32(cols, 0x9955);
        let xr = round_trip_bf16(&x);
        let mk = |bytes: &[u8], sb: &[f32]| {
            let p = rt.alloc_buffer(bytes.len()).unwrap();
            p.write_bytes(bytes);
            (p, buf_bf16(rt, sb))
        };
        let (qp, qs) = mk(&bq.i4_packed, &bq.i4_sb);
        let (kp, ks) = mk(&bk.i4_packed, &bk.i4_sb);
        let (vp, vs) = mk(&ivp, &ivs);
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
        close_rel(
            "qkv_i4 k",
            &ko.read_f32()[..rows_kv],
            &dense_gemv(&bk.dense, &xr, rows_kv, cols),
            3e-3,
        );
        close_rel(
            "qkv_i4 v",
            &vo.read_f32()[..rows_kv],
            &dense_gemv(&dv, &xr, rows_kv, cols),
            3e-3,
        );
    });
}

/// The fused gate/up + GELU form in Interleaved4, against the two GEMVs and the
/// same clamped GELU the standalone kernel uses.
#[test]
fn gate_up_gelu_i4_matches_two_gemvs_and_a_gelu() {
    with_gpu(|rt| {
        let (rows, cols, group) = (128usize, 256usize, 32usize);
        let bg = banks(rows, cols, group);
        let nu = nibbles_for(rows + 2, cols);
        let (_, su, _) = q4_mlx_matrix(rows + 2, cols, group);
        let gpr = cols / group;
        let sur = round_trip_bf16(&su);
        let mut du = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let gi = r * gpr + c / group;
                du[r * cols + c] = sur[gi * 2] * nu[r * cols + c] as f32 + sur[gi * 2 + 1];
            }
        }
        let (iup, ius) = interleave4(&nu[..rows * cols], &su[..rows * gpr * 2], rows, cols, group);

        let x = random_f32(cols, 0x9966);
        let xr = round_trip_bf16(&x);
        let gate = dense_gemv(&bg.dense, &xr, rows, cols);
        let up = dense_gemv(&du, &xr, rows, cols);
        // The kernel's GELU: clamp, tanh formulation, as `nn::mlp_gelu_tanh`.
        let want: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| {
                let xc = (*g as f64).clamp(-20.0, 20.0);
                let inner = 0.797_884_560_802_865_4 * (xc + 0.044715 * xc * xc * xc);
                (0.5 * xc * (1.0 + inner.clamp(-10.0, 10.0).tanh()) * (*u as f64)) as f32
            })
            .collect();

        let gp = rt.alloc_buffer(bg.i4_packed.len()).unwrap();
        gp.write_bytes(&bg.i4_packed);
        let gs = buf_bf16(rt, &bg.i4_sb);
        let up_p = rt.alloc_buffer(iup.len()).unwrap();
        up_p.write_bytes(&iup);
        let up_s = buf_bf16(rt, &ius);
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
            assert!(
                !got[..m * rows].contains(&UNWRITTEN),
                "gemm_add {name}: unwritten rows"
            );
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
