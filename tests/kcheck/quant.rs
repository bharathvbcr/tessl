//! Quantized GEMV/GEMM, int8, W4A8, and embedding lookups versus a CPU reference.

use std::sync::Arc;

use crate::common::{
    buf, buf_bf16, buf_u32, close_rel, dense_gemv, gelu_pytorch_tanh, pack_nibbles, random_f32, round_trip_bf16, silu,
};
use tessl::nn::{
    self, GateUpDispatch, I8AScale, I8BLayout, I8BScale, I8Gemm, I8Out, Q4Bank, Q4MlxBank, Q4MlxLayout,
    Q4MlxRowVariant, QkvOutputs, QuantShape, W4Arm, W4Gemm,
};
use tessl::tensor::GpuBuffer;
use tessl::GpuRuntime;

use super::Ledger;

fn bits_eq(label: &str, got: &[f32], want: &[f32]) {
    assert!(got.len() >= want.len(), "{label}: got {} < {}", got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{label}[{i}] got {g} want {w}");
    }
}

fn shape(rows: usize, cols: usize, group: usize) -> QuantShape {
    QuantShape {
        rows: rows as u32,
        cols: cols as u32,
        group_size: group as u32,
    }
}

fn pack_signed(q: &[i8]) -> Vec<u8> {
    q.chunks(2)
        .map(|pair| {
            let lo = (pair[0] as u8) & 0x0f;
            let hi = pair.get(1).copied().unwrap_or(0) as u8 & 0x0f;
            lo | (hi << 4)
        })
        .collect()
}

fn q4_rows(q: &[i8], scales: &[f32], zeros: &[f32], x: &[f32], rows: usize, cols: usize, group: usize) -> Vec<f32> {
    let gpr = cols / group;
    (0..rows)
        .map(|r| {
            let mut acc = 0.0f64;
            for c in 0..cols {
                let g = r * gpr + c / group;
                let w = scales[g] as f64 * (q[r * cols + c] as f64 - zeros[g] as f64);
                acc += w * x[c] as f64;
            }
            acc as f32
        })
        .collect()
}

fn classic(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let (rows, cols, group) = (16usize, 64usize, 32usize);
    let q: Vec<i8> = (0..rows * cols).map(|i| ((i * 37 + 11) % 16) as i8 - 8).collect();
    let groups = rows * (cols / group);
    let scales: Vec<f32> = (0..groups).map(|i| 0.01 + (i % 7) as f32 * 0.003).collect();
    let zeros: Vec<f32> = (0..groups).map(|i| -0.5 + (i % 5) as f32 * 0.25).collect();
    let x = random_f32(cols, 0xA4);
    let want = q4_rows(&q, &scales, &zeros, &x, rows, cols, group);
    let packed = rt.alloc_buffer(q.len() / 2).unwrap();
    packed.write_bytes(&pack_signed(&q));
    let sb = buf(rt, &scales);
    let zb = buf(rt, &zeros);
    let xb = buf(rt, &x);
    let sh = shape(rows, cols, group);
    for (tiled, name) in [(false, "gemv_q4"), (true, "gemv_q4_tiled")] {
        let yb = buf(rt, &vec![f32::NAN; rows]);
        ledger.credit(rt, name, &[name], || {
            let bank = Q4Bank {
                packed: &packed,
                scales: &sb,
                zeros: &zb,
            };
            nn::gemv_q4(rt, bank, &xb, &yb, sh, tiled).unwrap_or_else(|e| panic!("{name}: {e}"));
            rt.synchronize().unwrap();
            close_rel(name, &yb.read_f32()[..rows], &want, 2e-4);
        });
    }
}

fn q8_vec(q: &[i8], scales: &[f32], zeros: &[f32], x: &[f32], rows: usize, cols: usize, group: usize) -> Vec<f32> {
    q4_rows(q, scales, zeros, x, rows, cols, group)
}

#[allow(clippy::too_many_arguments)]
fn q8_mat(
    q: &[i8],
    scales: &[f32],
    zeros: &[f32],
    x: &[f32],
    m: usize,
    rows: usize,
    cols: usize,
    group: usize,
) -> Vec<f32> {
    let gpr = cols / group;
    let mut y = vec![0.0f32; m * rows];
    for i in 0..m {
        for r in 0..rows {
            let mut acc = 0.0f64;
            for c in 0..cols {
                let g = r * gpr + c / group;
                let w = scales[g] as f64 * (q[r * cols + c] as f64 - zeros[g] as f64);
                acc += w * x[i * cols + c] as f64;
            }
            y[i * rows + r] = acc as f32;
        }
    }
    y
}

fn q8(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let (rows, cols, group) = (16usize, 64usize, 32usize);
    let q: Vec<i8> = (0..rows * cols).map(|i| ((i * 53 + 7) % 251) as i32 as i8).collect();
    let groups = rows * (cols / group);
    let scales: Vec<f32> = (0..groups).map(|i| 0.004 + (i % 9) as f32 * 0.001).collect();
    let zeros: Vec<f32> = (0..groups).map(|i| -3.0 + (i % 4) as f32).collect();
    let x = random_f32(cols, 0xB8);
    let want = q8_vec(&q, &scales, &zeros, &x, rows, cols, group);
    let packed = rt.alloc_buffer(q.len()).unwrap();
    packed.write_bytes(&q.iter().map(|v| *v as u8).collect::<Vec<_>>());
    let sb = buf(rt, &scales);
    let zb = buf(rt, &zeros);
    let xb = buf(rt, &x);
    let yb = buf(rt, &vec![f32::NAN; rows]);
    ledger.credit(rt, "gemv_q8", &["gemv_q8"], || {
        nn::gemv_q8(rt, &packed, &sb, &zb, &xb, &yb, rows as u32, cols as u32, group as u32)
            .unwrap_or_else(|e| panic!("gemv_q8: {e}"));
        rt.synchronize().unwrap();
        close_rel("gemv_q8", &yb.read_f32()[..rows], &want, 2e-4);
    });

    let (rows, cols, m) = (32usize, 128usize, 2usize);
    for group in [32usize, 64, 128] {
        let name = match group {
            32 => "gemm_q8_tensorops_g32",
            64 => "gemm_q8_tensorops_g64",
            128 => "gemm_q8_tensorops_g128",
            _ => unreachable!(),
        };
        let q: Vec<i8> = (0..rows * cols)
            .map(|i| ((i * 3 + group) % 17) as i8 - 8)
            .collect();
        let ng = rows * (cols / group);
        let scales: Vec<f32> = (0..ng).map(|i| 0.02 + (i % 5) as f32 * 0.001).collect();
        let zeros: Vec<f32> = (0..ng).map(|i| (i % 3) as f32 - 1.0).collect();
        let x = random_f32(m * cols, 0xC0 + group as u64);
        let want = q8_mat(&q, &scales, &zeros, &x, m, rows, cols, group);
        let packed = rt.alloc_buffer(q.len()).unwrap();
        packed.write_bytes(&q.iter().map(|v| *v as u8).collect::<Vec<_>>());
        let sb = buf(rt, &scales);
        let zb = buf(rt, &zeros);
        let xb = buf(rt, &x);
        let yb = buf(rt, &vec![f32::NAN; m * rows]);
        ledger.credit(rt, name, &[name], || {
            nn::gemm_q8_tensorops(
                rt,
                &packed,
                &sb,
                &zb,
                &xb,
                &yb,
                rows as u32,
                cols as u32,
                group as u32,
                m as u32,
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
            rt.synchronize().unwrap();
            close_rel(name, &yb.read_f32()[..want.len()], &want, 2e-4);
        });
    }
}

struct Mlx {
    nibbles: Vec<u8>,
    packed: Vec<u8>,
    sb: Vec<f32>,
    dense: Vec<f32>,
}

fn mlx_matrix(rows: usize, cols: usize, group: usize, salt: usize) -> Mlx {
    let nibbles: Vec<u8> = (0..rows * cols).map(|i| ((i * 5 + salt) % 16) as u8).collect();
    let groups = rows * (cols / group);
    let sb: Vec<f32> = (0..groups * 2)
        .map(|i| {
            if i % 2 == 0 {
                0.03 + ((i + salt) % 7) as f32 * 0.002
            } else {
                ((i + salt) % 5) as f32 * 0.1 - 0.2
            }
        })
        .collect();
    let rounded = round_trip_bf16(&sb);
    let mut dense = vec![0.0f32; rows * cols];
    let gpr = cols / group;
    for r in 0..rows {
        for c in 0..cols {
            let gi = r * gpr + c / group;
            dense[r * cols + c] = rounded[gi * 2] * nibbles[r * cols + c] as f32 + rounded[gi * 2 + 1];
        }
    }
    Mlx {
        packed: pack_nibbles(&nibbles),
        sb,
        dense,
        nibbles,
    }
}

fn interleave4(nibbles: &[u8], sb: &[f32], rows: usize, cols: usize, group: usize) -> (Vec<u8>, Vec<f32>) {
    let gpr = cols / group;
    let packs = cols / 16;
    let tiles = rows.div_ceil(4);
    let mut out_n = vec![0u8; tiles * packs * 4 * 16];
    let mut out_sb = vec![0.0f32; tiles * gpr * 4 * 2];
    for row in 0..rows {
        let (tile, r) = (row / 4, row % 4);
        for p in 0..packs {
            let dst = ((tile * packs + p) * 4 + r) * 16;
            let src = row * cols + p * 16;
            out_n[dst..dst + 16].copy_from_slice(&nibbles[src..src + 16]);
        }
        for g in 0..gpr {
            let dst = (tile * gpr + g) * 4 + r;
            out_sb[dst * 2] = sb[(row * gpr + g) * 2];
            out_sb[dst * 2 + 1] = sb[(row * gpr + g) * 2 + 1];
        }
    }
    (pack_nibbles(&out_n), out_sb)
}

fn blocked_bank(nibbles: &[u8], sb: &[f32], rows: usize, cols: usize, group: usize) -> (Vec<u8>, Vec<f32>) {
    let gpr = cols / group;
    let stored = rows.div_ceil(16) * 16;
    let mut nb = vec![0u8; stored * cols];
    let mut out_sb = vec![0.0f32; stored * gpr * 2];
    for b in 0..rows.div_ceil(16) {
        for r_local in 0..16 {
            let r = b * 16 + r_local;
            if r >= rows {
                continue;
            }
            for g in 0..gpr {
                let dst = b * gpr * 16 + g * 16 + r_local;
                out_sb[dst * 2] = sb[(r * gpr + g) * 2];
                out_sb[dst * 2 + 1] = sb[(r * gpr + g) * 2 + 1];
                nb[dst * group..(dst + 1) * group]
                    .copy_from_slice(&nibbles[r * cols + g * group..r * cols + (g + 1) * group]);
            }
        }
    }
    (pack_nibbles(&nb), out_sb)
}

fn hold(rt: &Arc<GpuRuntime>, packed: &[u8], sb: &[f32]) -> (GpuBuffer, GpuBuffer) {
    let p = rt.alloc_buffer(packed.len()).unwrap();
    p.write_bytes(packed);
    (p, buf_bf16(rt, sb))
}

fn gemm_ref(dense: &[f32], x: &[f32], m: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m * rows];
    for i in 0..m {
        for r in 0..rows {
            let mut acc = 0.0f64;
            for c in 0..cols {
                acc += dense[r * cols + c] as f64 * x[i * cols + c] as f64;
            }
            y[i * rows + r] = acc as f32;
        }
    }
    y
}

fn mlx_rows(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let (rows, cols, group) = (32usize, 64usize, 32usize);
    let w = mlx_matrix(rows, cols, group, 1);
    let x = random_f32(cols, 0xF101);
    let want = dense_gemv(&w.dense, &x, rows, cols);
    let (pb, sb) = hold(rt, &w.packed, &w.sb);
    let xb = buf(rt, &x);
    let sh = shape(rows, cols, group);
    for (name, variant) in [
        ("gemv_q4_mlx", Q4MlxRowVariant::Standard),
        ("gemv_q4_mlx_wide", Q4MlxRowVariant::Wide),
        ("gemv_q4_mlx_tiled", Q4MlxRowVariant::Tiled),
    ] {
        let yb = buf(rt, &vec![f32::NAN; rows]);
        ledger.credit(rt, name, &[name], || {
            let bank = Q4MlxBank {
                packed: &pb,
                scales_biases: &sb,
            };
            nn::gemv_q4_mlx(rt, bank, &xb, &yb, sh, variant).unwrap_or_else(|e| panic!("{name}: {e}"));
            rt.synchronize().unwrap();
            close_rel(name, &yb.read_f32()[..rows], &want, 3e-3);
        });
    }
    let (bp, bs) = blocked_bank(&w.nibbles, &w.sb, rows, cols, group);
    let (pb, sb) = hold(rt, &bp, &bs);
    let yb = buf(rt, &vec![f32::NAN; rows]);
    ledger.credit(rt, "gemv_q4_mlx_blocked", &["gemv_q4_mlx_blocked"], || {
        let bank = Q4MlxBank {
            packed: &pb,
            scales_biases: &sb,
        };
        nn::gemv_q4_mlx_blocked(rt, bank, &xb, &yb, sh).unwrap_or_else(|e| panic!("blocked: {e}"));
        rt.synchronize().unwrap();
        close_rel("gemv_q4_mlx_blocked", &yb.read_f32()[..rows], &want, 3e-3);
    });
}

#[allow(clippy::too_many_arguments)]
fn mlx_simd_one(
    rt: &Arc<GpuRuntime>,
    ledger: &mut Ledger,
    name: &str,
    layout: Q4MlxLayout,
    with_resid: bool,
    w: &Mlx,
    rows: usize,
    cols: usize,
    group: usize,
) {
    let x = random_f32(cols, 0x1100 + cols as u64);
    let x_ref = round_trip_bf16(&x);
    let mut want = dense_gemv(&w.dense, &x_ref, rows, cols);
    let resid = random_f32(rows, 0x1200);
    if with_resid {
        for (y, r) in want.iter_mut().zip(&resid) {
            *y += *r;
        }
    }
    let (packed, sb) = match layout {
        Q4MlxLayout::RowMajor => (w.packed.clone(), w.sb.clone()),
        Q4MlxLayout::Interleaved4 => interleave4(&w.nibbles, &w.sb, rows, cols, group),
    };
    let (pb, sb) = hold(rt, &packed, &sb);
    let xb = buf_bf16(rt, &x);
    let rb = buf(rt, &resid);
    let yb = buf(rt, &vec![f32::NAN; rows]);
    let sh = shape(rows, cols, group);
    ledger.credit(rt, name, &[name], || {
        let bank = Q4MlxBank {
            packed: &pb,
            scales_biases: &sb,
        };
        let resid_arg = if with_resid { Some(&rb) } else { None };
        nn::gemv_q4_mlx_simd(rt, bank, &xb, &yb, sh, layout, resid_arg).unwrap_or_else(|e| panic!("{name}: {e}"));
        rt.synchronize().unwrap();
        close_rel(name, &yb.read_f32()[..rows], &want, 3e-3);
    });
}

#[allow(clippy::too_many_arguments)]
fn mlx_gemm_one(
    rt: &Arc<GpuRuntime>,
    ledger: &mut Ledger,
    name: &str,
    layout: Q4MlxLayout,
    with_resid: bool,
    w: &Mlx,
    rows: usize,
    cols: usize,
    group: usize,
) {
    let m = 2usize;
    let x = random_f32(m * cols, 0x2100);
    let x_ref = round_trip_bf16(&x);
    let mut want = gemm_ref(&w.dense, &x_ref, m, rows, cols);
    let resid = random_f32(m * rows, 0x2200);
    if with_resid {
        for (y, r) in want.iter_mut().zip(&resid) {
            *y += *r;
        }
    }
    let (packed, sb) = match layout {
        Q4MlxLayout::RowMajor => (w.packed.clone(), w.sb.clone()),
        Q4MlxLayout::Interleaved4 => interleave4(&w.nibbles, &w.sb, rows, cols, group),
    };
    let (pb, sb) = hold(rt, &packed, &sb);
    let xb = buf_bf16(rt, &x);
    let rb = buf(rt, &resid);
    let yb = buf(rt, &vec![f32::NAN; m * rows]);
    let sh = shape(rows, cols, group);
    ledger.credit(rt, name, &[name], || {
        let bank = Q4MlxBank {
            packed: &pb,
            scales_biases: &sb,
        };
        let resid_arg = if with_resid { Some(&rb) } else { None };
        nn::gemm_q4_mlx(rt, bank, &xb, &yb, sh, m as u32, layout, resid_arg).unwrap_or_else(|e| panic!("{name}: {e}"));
        rt.synchronize().unwrap();
        close_rel(name, &yb.read_f32()[..want.len()], &want, 3e-3);
    });
}

fn mlx_products(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let (rows, cols, group) = (32usize, 64usize, 32usize);
    let w = mlx_matrix(rows, cols, group, 3);
    for (name, layout, resid) in [
        ("gemv_q4_mlx_simd", Q4MlxLayout::RowMajor, false),
        ("gemv_q4_mlx_simd_i4", Q4MlxLayout::Interleaved4, false),
        ("gemv_q4_mlx_simd_add", Q4MlxLayout::RowMajor, true),
        ("gemv_q4_mlx_simd_add_i4", Q4MlxLayout::Interleaved4, true),
    ] {
        mlx_simd_one(rt, ledger, name, layout, resid, &w, rows, cols, group);
    }
    for (name, layout, resid) in [
        ("gemm_q4_mlx_simd", Q4MlxLayout::RowMajor, false),
        ("gemm_q4_mlx_simd_i4", Q4MlxLayout::Interleaved4, false),
        ("gemm_q4_mlx_simd_add", Q4MlxLayout::RowMajor, true),
        ("gemm_q4_mlx_simd_add_i4", Q4MlxLayout::Interleaved4, true),
    ] {
        mlx_gemm_one(rt, ledger, name, layout, resid, &w, rows, cols, group);
    }
    for group in [32usize, 64, 128] {
        let name = match group {
            32 => "gemm_q4_mlx_tensorops_g32",
            64 => "gemm_q4_mlx_tensorops_g64",
            128 => "gemm_q4_mlx_tensorops_g128",
            _ => unreachable!(),
        };
        let (rows, cols, m) = (32usize, 128usize, 2usize);
        let w = mlx_matrix(rows, cols, group, group);
        let x = random_f32(m * cols, 0x3100 + group as u64);
        let x_ref = round_trip_bf16(&x);
        let want = gemm_ref(&w.dense, &x_ref, m, rows, cols);
        let (pb, sb) = hold(rt, &w.packed, &w.sb);
        let xb = buf_bf16(rt, &x);
        let yb = buf(rt, &vec![f32::NAN; m * rows]);
        let sh = shape(rows, cols, group);
        ledger.credit(rt, name, &[name], || {
            let bank = Q4MlxBank {
                packed: &pb,
                scales_biases: &sb,
            };
            nn::gemm_q4_mlx_tensorops(rt, bank, &xb, &yb, sh, m as u32).unwrap_or_else(|e| panic!("{name}: {e}"));
            rt.synchronize().unwrap();
            close_rel(name, &yb.read_f32()[..want.len()], &want, 3e-3);
        });
    }
}

fn two_outs(gate: &[f32], up: &[f32], gelu: bool) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(g, u)| {
            let h = if gelu {
                gelu_pytorch_tanh(*g as f64)
            } else {
                silu(*g as f64)
            };
            (h * *u as f64) as f32
        })
        .collect()
}

fn mlx_fused(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let (rows, cols, group) = (32usize, 64usize, 32usize);
    let g = mlx_matrix(rows, cols, group, 4);
    let u = mlx_matrix(rows, cols, group, 9);
    let x = random_f32(cols, 0x4400);
    let x_bf = round_trip_bf16(&x);
    let sh = shape(rows, cols, group);
    let xb16 = buf_bf16(rt, &x);
    let xb32 = buf(rt, &x);

    let launch = |layout: Q4MlxLayout, gelu: bool, blocked: bool| {
        let xref = if blocked { &x } else { &x_bf };
        let want = two_outs(
            &dense_gemv(&g.dense, xref, rows, cols),
            &dense_gemv(&u.dense, xref, rows, cols),
            gelu,
        );
        let (gp, gs) = if layout == Q4MlxLayout::Interleaved4 {
            interleave4(&g.nibbles, &g.sb, rows, cols, group)
        } else if blocked {
            blocked_bank(&g.nibbles, &g.sb, rows, cols, group)
        } else {
            (g.packed.clone(), g.sb.clone())
        };
        let (up, us) = if layout == Q4MlxLayout::Interleaved4 {
            interleave4(&u.nibbles, &u.sb, rows, cols, group)
        } else if blocked {
            blocked_bank(&u.nibbles, &u.sb, rows, cols, group)
        } else {
            (u.packed.clone(), u.sb.clone())
        };
        (want, gp, gs, up, us)
    };

    for (name, layout, blocked) in [
        ("gemv_q4_mlx_simd_gate_up_gelu", Q4MlxLayout::RowMajor, false),
        ("gemv_q4_mlx_simd_gate_up_gelu_i4", Q4MlxLayout::Interleaved4, false),
        ("gemv_q4_mlx_blocked_gate_up_gelu", Q4MlxLayout::RowMajor, true),
    ] {
        let (want, gp, gs, up, us) = launch(layout, true, blocked);
        let (gpb, gsb) = hold(rt, &gp, &gs);
        let (upb, usb) = hold(rt, &up, &us);
        let mid = buf(rt, &vec![f32::NAN; rows]);
        let xbuf = if blocked { &xb32 } else { &xb16 };
        let dispatch = if blocked {
            GateUpDispatch::Blocked
        } else {
            GateUpDispatch::Simd(layout)
        };
        ledger.credit(rt, name, &[name], || {
            nn::gemv_q4_mlx_gate_up_gelu(
                rt,
                Q4MlxBank {
                    packed: &gpb,
                    scales_biases: &gsb,
                },
                Q4MlxBank {
                    packed: &upb,
                    scales_biases: &usb,
                },
                xbuf,
                &mid,
                sh,
                dispatch,
                false,
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
            rt.synchronize().unwrap();
            close_rel(name, &mid.read_f32()[..rows], &want, 3e-3);
        });
    }
    let (want, gp, gs, up, us) = launch(Q4MlxLayout::RowMajor, false, false);
    let (gpb, gsb) = hold(rt, &gp, &gs);
    let (upb, usb) = hold(rt, &up, &us);
    let mid = buf(rt, &vec![f32::NAN; rows]);
    ledger.credit(
        rt,
        "gemv_q4_mlx_simd_gate_up_silu",
        &["gemv_q4_mlx_simd_gate_up_silu"],
        || {
            nn::gemv_q4_mlx_gate_up_silu(
                rt,
                Q4MlxBank {
                    packed: &gpb,
                    scales_biases: &gsb,
                },
                Q4MlxBank {
                    packed: &upb,
                    scales_biases: &usb,
                },
                &xb16,
                &mid,
                sh,
                false,
            )
            .unwrap_or_else(|e| panic!("silu: {e}"));
            rt.synchronize().unwrap();
            close_rel("gemv_q4_mlx_simd_gate_up_silu", &mid.read_f32()[..rows], &want, 3e-3);
        },
    );
}

fn mlx_kv_qkv(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let (rows, cols, group) = (32usize, 64usize, 32usize);
    let k = mlx_matrix(rows, cols, group, 5);
    let v = mlx_matrix(rows, cols, group, 6);
    let q = mlx_matrix(rows, cols, group, 7);
    let x = random_f32(cols, 0x5500);
    let xref = round_trip_bf16(&x);
    let xb = buf_bf16(rt, &x);
    let sh = shape(rows, cols, group);
    for (name, layout) in [
        ("gemv_q4_mlx_simd_kv", Q4MlxLayout::RowMajor),
        ("gemv_q4_mlx_simd_kv_i4", Q4MlxLayout::Interleaved4),
    ] {
        let (kp, ks) = match layout {
            Q4MlxLayout::RowMajor => (k.packed.clone(), k.sb.clone()),
            Q4MlxLayout::Interleaved4 => interleave4(&k.nibbles, &k.sb, rows, cols, group),
        };
        let (vp, vs) = match layout {
            Q4MlxLayout::RowMajor => (v.packed.clone(), v.sb.clone()),
            Q4MlxLayout::Interleaved4 => interleave4(&v.nibbles, &v.sb, rows, cols, group),
        };
        let (kpb, ksb) = hold(rt, &kp, &ks);
        let (vpb, vsb) = hold(rt, &vp, &vs);
        let ko = buf(rt, &vec![f32::NAN; rows]);
        let vo = buf(rt, &vec![f32::NAN; rows]);
        let want_k = dense_gemv(&k.dense, &xref, rows, cols);
        let want_v = dense_gemv(&v.dense, &xref, rows, cols);
        ledger.credit(rt, name, &[name], || {
            nn::gemv_q4_mlx_kv(
                rt,
                Q4MlxBank {
                    packed: &kpb,
                    scales_biases: &ksb,
                },
                Q4MlxBank {
                    packed: &vpb,
                    scales_biases: &vsb,
                },
                &xb,
                &ko,
                &vo,
                sh,
                layout,
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
            rt.synchronize().unwrap();
            close_rel(name, &ko.read_f32()[..rows], &want_k, 3e-3);
            close_rel(name, &vo.read_f32()[..rows], &want_v, 3e-3);
        });
    }
    for (name, layout) in [
        ("gemv_q4_mlx_simd_qkv", Q4MlxLayout::RowMajor),
        ("gemv_q4_mlx_simd_qkv_i4", Q4MlxLayout::Interleaved4),
    ] {
        let pack = |w: &Mlx| match layout {
            Q4MlxLayout::RowMajor => (w.packed.clone(), w.sb.clone()),
            Q4MlxLayout::Interleaved4 => interleave4(&w.nibbles, &w.sb, rows, cols, group),
        };
        let (qp, qs) = pack(&q);
        let (kp, ks) = pack(&k);
        let (vp, vs) = pack(&v);
        let (qpb, qsb) = hold(rt, &qp, &qs);
        let (kpb, ksb) = hold(rt, &kp, &ks);
        let (vpb, vsb) = hold(rt, &vp, &vs);
        let qo = buf(rt, &vec![f32::NAN; rows]);
        let ko = buf(rt, &vec![f32::NAN; rows]);
        let vo = buf(rt, &vec![f32::NAN; rows]);
        ledger.credit(rt, name, &[name], || {
            nn::gemv_q4_mlx_qkv(
                rt,
                Q4MlxBank {
                    packed: &qpb,
                    scales_biases: &qsb,
                },
                Q4MlxBank {
                    packed: &kpb,
                    scales_biases: &ksb,
                },
                Q4MlxBank {
                    packed: &vpb,
                    scales_biases: &vsb,
                },
                &xb,
                QkvOutputs {
                    q_out: &qo,
                    k_out: &ko,
                    v_out: &vo,
                },
                rows as u32,
                rows as u32,
                cols as u32,
                group as u32,
                layout,
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
            rt.synchronize().unwrap();
            close_rel(
                name,
                &qo.read_f32()[..rows],
                &dense_gemv(&q.dense, &xref, rows, cols),
                3e-3,
            );
            close_rel(
                name,
                &ko.read_f32()[..rows],
                &dense_gemv(&k.dense, &xref, rows, cols),
                3e-3,
            );
            close_rel(
                name,
                &vo.read_f32()[..rows],
                &dense_gemv(&v.dense, &xref, rows, cols),
                3e-3,
            );
        });
    }
}

fn embed_row_q4(
    q: &[i8],
    scales: &[f32],
    zeros: &[f32],
    id: u32,
    vocab: usize,
    hidden: usize,
    group: usize,
) -> Vec<f32> {
    if id as usize >= vocab {
        return vec![0.0; hidden];
    }
    let r = id as usize;
    let gpr = hidden / group;
    (0..hidden)
        .map(|c| {
            let g = r * gpr + c / group;
            (scales[g] as f64 * (q[r * hidden + c] as f64 - zeros[g] as f64)) as f32
        })
        .collect()
}

fn embeds(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let (vocab, hidden, group, n_tok) = (8usize, 32usize, 8usize, 3usize);
    let q: Vec<i8> = (0..vocab * hidden).map(|i| ((i * 3) % 16) as i8 - 8).collect();
    let groups = vocab * (hidden / group);
    let scales: Vec<f32> = (0..groups).map(|i| 0.05 + (i % 4) as f32 * 0.01).collect();
    let zeros: Vec<f32> = (0..groups).map(|i| (i % 3) as f32).collect();
    let ids = [0u32, 2, 100];
    let mut want = Vec::new();
    for id in ids {
        want.extend(embed_row_q4(&q, &scales, &zeros, id, vocab, hidden, group));
    }
    let packed = rt.alloc_buffer(q.len() / 2).unwrap();
    packed.write_bytes(&pack_signed(&q));
    let sb = buf(rt, &scales);
    let zb = buf(rt, &zeros);
    let ib = buf_u32(rt, &ids);
    let out = buf(rt, &vec![f32::NAN; n_tok * hidden]);
    ledger.credit(rt, "embed_lookup_q4", &["embed_lookup_q4"], || {
        nn::embed_lookup_q4(
            rt,
            Q4Bank {
                packed: &packed,
                scales: &sb,
                zeros: &zb,
            },
            &ib,
            &out,
            vocab as u32,
            hidden as u32,
            group as u32,
            n_tok as u32,
        )
        .unwrap_or_else(|e| panic!("embed_lookup_q4: {e}"));
        rt.synchronize().unwrap();
        close_rel("embed_lookup_q4", &out.read_f32()[..want.len()], &want, 2e-4);
    });

    let (vocab, hidden, group) = (8usize, 32usize, 32usize);
    let w = mlx_matrix(vocab, hidden, group, 2);
    let ids = [1u32, 3, 99];
    let mut want = Vec::new();
    for id in ids {
        if (id as usize) >= vocab {
            want.extend(vec![0.0f32; hidden]);
        } else {
            let r = id as usize;
            want.extend_from_slice(&w.dense[r * hidden..(r + 1) * hidden]);
        }
    }
    let (pb, sb) = hold(rt, &w.packed, &w.sb);
    let ib = buf_u32(rt, &ids);
    let out = buf(rt, &vec![f32::NAN; ids.len() * hidden]);
    ledger.credit(rt, "embed_lookup_q4_mlx", &["embed_lookup_q4_mlx"], || {
        nn::embed_lookup_q4_mlx(
            rt,
            Q4MlxBank {
                packed: &pb,
                scales_biases: &sb,
            },
            &ib,
            &out,
            vocab as u32,
            hidden as u32,
            group as u32,
            ids.len() as u32,
        )
        .unwrap_or_else(|e| panic!("embed_lookup_q4_mlx: {e}"));
        rt.synchronize().unwrap();
        close_rel("embed_lookup_q4_mlx", &out.read_f32()[..want.len()], &want, 3e-3);
    });
}

fn seq_i8(n: usize, seed: i32) -> Vec<i8> {
    (0..n)
        .map(|i| ((i as i32).wrapping_mul(13).wrapping_add(seed) % 17 - 8) as i8)
        .collect()
}

/// Signed nibbles, the same span `tests/w4a8.rs` packs. A value of 8 does not
/// fit in four bits and would change the neighbouring nibble.
fn seq_nib(n: usize, seed: i32) -> Vec<i8> {
    (0..n)
        .map(|i| ((i as i32).wrapping_mul(5).wrapping_add(seed).rem_euclid(16) - 8) as i8)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn i8_oracle(a: &[i8], b: &[i8], m: usize, n: usize, k: usize, nt: bool, group: usize, scales: &[f32]) -> Vec<f32> {
    let ngroups = k.checked_div(group).unwrap_or(0);
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let v = if group == 0 {
                let mut s: i32 = 0;
                for p in 0..k {
                    let bv = if nt { b[j * k + p] } else { b[p * n + j] };
                    s += a[i * k + p] as i32 * bv as i32;
                }
                s as f32
            } else {
                let mut acc = 0.0f32;
                for g in 0..ngroups {
                    let mut s: i32 = 0;
                    for p in 0..group {
                        let kk = g * group + p;
                        let bv = if nt { b[j * k + kk] } else { b[kk * n + j] };
                        s += a[i * k + kk] as i32 * bv as i32;
                    }
                    acc += s as f32 * scales[g * n + j];
                }
                acc
            };
            out[i * n + j] = v;
        }
    }
    out
}

fn i8_bytes(q: &[i8]) -> Vec<u8> {
    q.iter().map(|v| *v as u8).collect()
}

fn i8_all(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let mut cases: Vec<(u32, I8BLayout, u32)> = vec![(1, I8BLayout::RowMajorKN, 0)];
    for wide in [false, true] {
        for nt in [false, true] {
            for group in [0u32, 64, 128, 256] {
                let m = if wide { 65 } else { 32 };
                cases.push((
                    m,
                    if nt {
                        I8BLayout::TransposedNK
                    } else {
                        I8BLayout::RowMajorKN
                    },
                    group,
                ));
            }
        }
    }
    for (m, layout, group) in cases {
        let name = nn::i8_gemm_pipeline_name(m, layout, group).unwrap_or_else(|e| panic!("{e}"));
        let k = if group == 0 { 32 } else { group };
        let n = 8u32;
        let nt = layout == I8BLayout::TransposedNK;
        let a = seq_i8((m * k) as usize, 2);
        let b = seq_i8((n * k) as usize, 5);
        let groups = k.checked_div(group).unwrap_or(0) as usize;
        let scales: Vec<f32> = (0..groups * n as usize)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.5 })
            .collect();
        let want = i8_oracle(&a, &b, m as usize, n as usize, k as usize, nt, group as usize, &scales);
        let ab = rt.alloc_buffer(a.len()).unwrap();
        ab.write_bytes(&i8_bytes(&a));
        let bb = rt.alloc_buffer(b.len()).unwrap();
        bb.write_bytes(&i8_bytes(&b));
        let sb = buf(rt, if scales.is_empty() { &[1.0f32] } else { &scales });
        let cb = rt.alloc_buffer((m * n * 4) as usize).unwrap();
        cb.write_bytes(&vec![0u8; (m * n * 4) as usize]);
        let scale = if group == 0 {
            I8BScale::None
        } else {
            I8BScale::PerGroup { scales: &sb, group }
        };
        ledger.credit(rt, name, &[name], || {
            nn::gemm_i8(
                rt,
                I8Gemm {
                    a: &ab,
                    b: &bb,
                    c: &cb,
                    m,
                    n,
                    k,
                    a_scale: I8AScale::Scalar(1.0),
                    b_scale: scale,
                    bias: None,
                    b_layout: layout,
                    out: I8Out::F32,
                },
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
            rt.synchronize().unwrap();
            bits_eq(name, &cb.read_f32(), &want);
        });
    }
}

fn quantize(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let rows = [vec![0.0f32, 0.0, 0.0, 0.0], vec![127.0, -63.5, 63.5, 1.0]];
    let cols = 4usize;
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let x = buf(rt, &flat);
    let q = rt.alloc_buffer(flat.len()).unwrap();
    q.write_bytes(&vec![0u8; flat.len()]);
    let scales = buf(rt, &[99.0, 99.0]);
    ledger.credit(rt, "i8_quantize_rows", &["i8_quantize_rows"], || {
        nn::quantize_i8_rows(rt, &x, &q, &scales, rows.len() as u32, cols as u32).unwrap_or_else(|e| panic!("{e}"));
        rt.synchronize().unwrap();
        let got_q: Vec<u8> = q.contents_u8()[..flat.len()].to_vec();
        let got_s = scales.read_f32();
        for (r, row) in rows.iter().enumerate() {
            let mut absmax = 0.0f32;
            for v in row {
                absmax = absmax.max(v.abs());
            }
            let scale = if absmax == 0.0 { 1.0 } else { absmax / 127.0 };
            assert_eq!(got_s[r].to_bits(), scale.to_bits(), "scale row {r}");
            for (c, v) in row.iter().enumerate() {
                let rq = (*v / scale).round_ties_even().clamp(-127.0, 127.0) as i8;
                assert_eq!(got_q[r * cols + c] as i8, rq, "q row {r} col {c}");
            }
        }
    });
}

fn i8_dequant(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let (m, n, k) = (16u32, 16u32, 16u32);
    let a = seq_i8((m * k) as usize, 3);
    let b = seq_i8((k * n) as usize, 4);
    let want = i8_oracle(&a, &b, m as usize, n as usize, k as usize, false, 0, &[]);
    let ab = rt.alloc_buffer(a.len()).unwrap();
    ab.write_bytes(&i8_bytes(&a));
    let bb = rt.alloc_buffer(b.len()).unwrap();
    bb.write_bytes(&i8_bytes(&b));
    let cb = rt.alloc_buffer((m * n * 4) as usize).unwrap();
    cb.write_bytes(&vec![0u8; (m * n * 4) as usize]);
    ledger.credit(rt, "matmul2d_tensorops_i8_f32", &["matmul2d_tensorops_i8_f32"], || {
        nn::gemm_i8_dequant(rt, &ab, &bb, &cb, m, n, k, 1.0, None).unwrap_or_else(|e| panic!("i8 dequant: {e}"));
        rt.synchronize().unwrap();
        bits_eq("matmul2d_tensorops_i8_f32", &cb.read_f32(), &want);
    });
}

fn w4_pack(logical: &[i8], n: usize, k: usize, zp8: bool) -> Vec<u8> {
    let mut out = vec![0u8; n * (k / 2)];
    for ni in 0..n {
        for kk in (0..k).step_by(2) {
            let nib = |v: i8| -> u8 {
                if zp8 {
                    (v + 8) as u8
                } else if v < 0 {
                    (v + 16) as u8
                } else {
                    v as u8
                }
            };
            let lo = nib(logical[ni * k + kk]);
            let hi = nib(logical[ni * k + kk + 1]);
            out[ni * (k / 2) + kk / 2] = lo | (hi << 4);
        }
    }
    out
}

fn w4_oracle(a: &[i8], b: &[i8], m: usize, n: usize, k: usize, group: usize, scales: &[f32]) -> Vec<f32> {
    let ngroups = k.checked_div(group).unwrap_or(0);
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let v = if group == 0 {
                let mut s: i32 = 0;
                for p in 0..k {
                    s += a[i * k + p] as i32 * b[j * k + p] as i32;
                }
                s as f32
            } else {
                let mut acc = 0.0f32;
                for g in 0..ngroups {
                    let mut s: i32 = 0;
                    for p in 0..group {
                        let kk = g * group + p;
                        s += a[i * k + kk] as i32 * b[j * k + kk] as i32;
                    }
                    acc += s as f32 * scales[g * n + j];
                }
                acc
            };
            out[i * n + j] = v;
        }
    }
    out
}

fn w4(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let (m, n, k) = (4u32, 8u32, 32u32);
    let a = seq_nib((m * k) as usize, 2);
    let b = seq_nib((n * k) as usize, 5);
    let want = w4_oracle(&a, &b, m as usize, n as usize, k as usize, 0, &[]);
    let packed = w4_pack(&b, n as usize, k as usize, true);
    let ab = rt.alloc_buffer(a.len()).unwrap();
    ab.write_bytes(&i8_bytes(&a));
    let pb = rt.alloc_buffer(packed.len()).unwrap();
    pb.write_bytes(&packed);
    let cb = rt.alloc_buffer((m * n * 4) as usize).unwrap();
    cb.write_bytes(&vec![0u8; (m * n * 4) as usize]);
    ledger.credit(rt, "w4a8_unpack_zp8", &["w4a8_unpack_zp8"], || {
        nn::gemm_w4a8(
            rt,
            W4Gemm {
                a: &ab,
                packed: &pb,
                c: &cb,
                m,
                n,
                k,
                a_scale: I8AScale::Scalar(1.0),
                b_scale: I8BScale::None,
                bias: None,
                out: I8Out::F32,
                arm: W4Arm::UnpackZp8,
            },
        )
        .unwrap_or_else(|e| panic!("w4 unpack: {e}"));
        rt.synchronize().unwrap();
        bits_eq("w4a8_unpack_zp8", &cb.read_f32(), &want);
    });

    for (group, name) in [
        (0u32, "w4a8_i4_32"),
        (64, "w4a8_i4_g64"),
        (128, "w4a8_i4_g128"),
        (256, "w4a8_i4_g256"),
    ] {
        let k = if group == 0 { 32 } else { group };
        let (m, n) = (32u32, 32u32);
        let a = seq_nib((m * k) as usize, 2);
        let b = seq_nib((n * k) as usize, 5);
        let groups = k.checked_div(group).unwrap_or(0) as usize;
        let scales: Vec<f32> = vec![1.0; groups * n as usize];
        let want = w4_oracle(&a, &b, m as usize, n as usize, k as usize, group as usize, &scales);
        let packed = w4_pack(&b, n as usize, k as usize, false);
        let ab = rt.alloc_buffer(a.len()).unwrap();
        ab.write_bytes(&i8_bytes(&a));
        let pb = rt.alloc_buffer(packed.len()).unwrap();
        pb.write_bytes(&packed);
        let sb = buf(rt, if scales.is_empty() { &[1.0f32] } else { &scales });
        let cb = rt.alloc_buffer((m * n * 4) as usize).unwrap();
        cb.write_bytes(&vec![0u8; (m * n * 4) as usize]);
        let scale = if group == 0 {
            I8BScale::None
        } else {
            I8BScale::PerGroup { scales: &sb, group }
        };
        ledger.credit(rt, name, &[name], || {
            nn::gemm_w4a8(
                rt,
                W4Gemm {
                    a: &ab,
                    packed: &pb,
                    c: &cb,
                    m,
                    n,
                    k,
                    a_scale: I8AScale::Scalar(1.0),
                    b_scale: scale,
                    bias: None,
                    out: I8Out::F32,
                    arm: W4Arm::Int4b,
                },
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
            rt.synchronize().unwrap();
            bits_eq(name, &cb.read_f32(), &want);
        });
    }
}

pub fn run(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    assert!(rt.has_tensorops(), "quant sweep needs TensorOps");
    classic(rt, ledger);
    q8(rt, ledger);
    mlx_rows(rt, ledger);
    mlx_products(rt, ledger);
    mlx_fused(rt, ledger);
    mlx_kv_qkv(rt, ledger);
    embeds(rt, ledger);
    i8_all(rt, ledger);
    quantize(rt, ledger);
    i8_dequant(rt, ledger);
    w4(rt, ledger);
}
