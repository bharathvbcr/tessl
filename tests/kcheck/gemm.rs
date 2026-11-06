//! GEMM, cast, copy, reduce, transpose, and the exact fixed-order kernels.

use std::sync::Arc;

use crate::common::{
    assert_within_bound, random_f32, reference, round_trip_bf16, tensor_bf16, tensor_f32, Layout, Reference, U_BF16,
};
use tessl::dispatch::{dispatch_1d, set_gpu_buf, set_u32};
use tessl::gemm::{
    gemm, gemm_epilogue_tiled, gemm_f32, gemm_f32_cpu, gemm_fixed_order_nn, gemm_fixed_order_nt, gemm_fixed_order_tn,
    gemm_nn_splitk_par_f32, gemm_nt_bf16, gemm_nt_bf16_bias, gemm_nt_f32, gemm_tiled, gemm_tn_bf16, gemm_tn_f32,
    gemm_tn_splitk_par_bf16, gemm_tn_splitk_par_f32, transpose_f32_into, Activation, BatchStrides, BatchedGemm,
    EpiTile, Epilogue, GemmBackend, GemmOperands,
};
use tessl::tensor::{bf16_bits_to_f32, f16_bits_to_f32, f32_slice_to_f16, f32_to_bf16_bits_gpu};
use tessl::{DType, GpuRuntime, PrecisionMode, Tensor};

use super::Ledger;

fn sync(rt: &GpuRuntime) {
    rt.synchronize().expect("sync");
}

fn round_f16(data: &[f32]) -> Vec<f32> {
    f32_slice_to_f16(data).into_iter().map(f16_bits_to_f32).collect()
}

fn tensor_f16(rt: &Arc<GpuRuntime>, shape: &[usize], data: &[f32]) -> Tensor {
    let t = rt.alloc_tensor_f16(shape).expect("alloc_tensor_f16");
    t.buffer.write_f16_bits(&f32_slice_to_f16(data));
    t
}

struct Mats {
    a: Tensor,
    b: Tensor,
    c: Tensor,
    expect: Reference,
    k: usize,
}

fn mats(rt: &Arc<GpuRuntime>, layout: Layout, m: usize, n: usize, k: usize, seed: u64, dtype: DType) -> Mats {
    let (a_len, b_len) = match layout {
        Layout::Nn => (m * k, k * n),
        Layout::Tn => (k * m, k * n),
        Layout::Nt => (m * k, n * k),
    };
    let (a_shape, b_shape) = match layout {
        Layout::Nn => (vec![m, k], vec![k, n]),
        Layout::Tn => (vec![k, m], vec![k, n]),
        Layout::Nt => (vec![m, k], vec![n, k]),
    };
    let mut a_host = random_f32(a_len, seed);
    let mut b_host = random_f32(b_len, seed ^ 0x9e37);
    if dtype == DType::BF16 {
        a_host = round_trip_bf16(&a_host);
        b_host = round_trip_bf16(&b_host);
    } else if dtype == DType::F16 {
        a_host = round_f16(&a_host);
        b_host = round_f16(&b_host);
    }
    let expect = reference(layout, &a_host, &b_host, m, n, k);
    let (a, b) = match dtype {
        DType::F32 => (tensor_f32(rt, &a_shape, &a_host), tensor_f32(rt, &b_shape, &b_host)),
        DType::BF16 => (tensor_bf16(rt, &a_shape, &a_host), tensor_bf16(rt, &b_shape, &b_host)),
        DType::F16 => (tensor_f16(rt, &a_shape, &a_host), tensor_f16(rt, &b_shape, &b_host)),
        _ => panic!("gemm sweep dtype {dtype:?}"),
    };
    let c = rt.alloc_tensor_f32(&[m, n]).expect("c");
    Mats { a, b, c, expect, k }
}

fn read_c(c: &Tensor) -> Vec<f32> {
    c.read_f32().expect("read c")
}

fn check_bound(label: &str, got: &[f32], expect: &Reference, k: usize, operand_u: f64) {
    assert_within_bound(label, got, expect, k, operand_u);
}

fn launch_1d(rt: &GpuRuntime, name: &str, n: usize, encode: impl FnOnce(&mut tessl::dispatch::Binder<'_>)) {
    let p = rt.pipeline(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    dispatch_1d(rt, &p, n, encode).unwrap_or_else(|e| panic!("{name}: {e}"));
}

fn bits_eq(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label} length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{label} [{i}] got {g} want {w}");
    }
}

fn utils(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let n = 16usize;
    let src_f = random_f32(n, 3);
    ledger.credit(rt, "copy_f32", &["copy_f32"], || {
        let src = rt.alloc_buffer(n * 4).unwrap();
        let dst = rt.alloc_buffer(n * 4).unwrap();
        src.write_f32(&src_f);
        launch_1d(rt, "copy_f32", n, |bnd| {
            set_gpu_buf(bnd, &src, 0);
            set_gpu_buf(bnd, &dst, 1);
            set_u32(bnd, n as u32, 2);
        });
        sync(rt);
        bits_eq("copy_f32", &dst.read_f32()[..n], &src_f);
    });

    let words: Vec<u32> = (0..n).map(|i| 0x1000_0001 + i as u32).collect();
    ledger.credit(rt, "copy_u32", &["copy_u32"], || {
        let src = rt.alloc_buffer(n * 4).unwrap();
        let dst = rt.alloc_buffer(n * 4).unwrap();
        src.write_u32(&words);
        launch_1d(rt, "copy_u32", n, |bnd| {
            set_gpu_buf(bnd, &src, 0);
            set_gpu_buf(bnd, &dst, 1);
            set_u32(bnd, n as u32, 2);
        });
        sync(rt);
        let got = dst.contents_u32();
        assert_eq!(&got[..n], &words[..]);
    });

    for (name, bits) in [
        ("copy_f16", f32_slice_to_f16(&src_f)),
        ("copy_bf16", src_f.iter().copied().map(f32_to_bf16_bits_gpu).collect()),
    ] {
        ledger.credit(rt, name, &[name], || {
            let src = rt.alloc_buffer(n * 2).unwrap();
            let dst = rt.alloc_buffer(n * 2).unwrap();
            src.write_f16_bits(&bits);
            launch_1d(rt, name, n, |bnd| {
                set_gpu_buf(bnd, &src, 0);
                set_gpu_buf(bnd, &dst, 1);
                set_u32(bnd, n as u32, 2);
            });
            sync(rt);
            let got = dst.contents_u16();
            assert_eq!(&got[..n], &bits[..], "{name}");
        });
    }

    ledger.credit(rt, "cast_f32_to_f16", &["cast_f32_to_f16"], || {
        let src = rt.alloc_buffer(n * 4).unwrap();
        let dst = rt.alloc_buffer(n * 2).unwrap();
        src.write_f32(&src_f);
        launch_1d(rt, "cast_f32_to_f16", n, |bnd| {
            set_gpu_buf(bnd, &src, 0);
            set_gpu_buf(bnd, &dst, 1);
            set_u32(bnd, n as u32, 2);
        });
        sync(rt);
        let want = f32_slice_to_f16(&src_f);
        assert_eq!(&dst.contents_u16()[..n], &want[..], "cast_f32_to_f16");
    });
    ledger.credit(rt, "cast_f16_to_f32", &["cast_f16_to_f32"], || {
        let bits = f32_slice_to_f16(&src_f);
        let src = rt.alloc_buffer(n * 2).unwrap();
        let dst = rt.alloc_buffer(n * 4).unwrap();
        src.write_f16_bits(&bits);
        launch_1d(rt, "cast_f16_to_f32", n, |bnd| {
            set_gpu_buf(bnd, &src, 0);
            set_gpu_buf(bnd, &dst, 1);
            set_u32(bnd, n as u32, 2);
        });
        sync(rt);
        let want: Vec<f32> = bits.iter().copied().map(f16_bits_to_f32).collect();
        bits_eq("cast_f16_to_f32", &dst.read_f32()[..n], &want);
    });

    let mut mixed = src_f.clone();
    mixed[0] = f32::NAN;
    mixed[1] = 1.0e-40;
    let gpu_bf = mixed.iter().copied().map(f32_to_bf16_bits_gpu).collect::<Vec<_>>();
    ledger.credit(rt, "cast_f32_to_bf16", &["cast_f32_to_bf16"], || {
        let src = rt.alloc_buffer(n * 4).unwrap();
        let dst = rt.alloc_buffer(n * 2).unwrap();
        src.write_f32(&mixed);
        launch_1d(rt, "cast_f32_to_bf16", n, |bnd| {
            set_gpu_buf(bnd, &src, 0);
            set_gpu_buf(bnd, &dst, 1);
            set_u32(bnd, n as u32, 2);
        });
        sync(rt);
        assert_eq!(&dst.contents_u16()[..n], &gpu_bf[..], "cast_f32_to_bf16");
    });
    ledger.credit(rt, "cast_bf16_to_f32", &["cast_bf16_to_f32"], || {
        let src = rt.alloc_buffer(n * 2).unwrap();
        let dst = rt.alloc_buffer(n * 4).unwrap();
        src.write_bf16_bits(&gpu_bf);
        launch_1d(rt, "cast_bf16_to_f32", n, |bnd| {
            set_gpu_buf(bnd, &src, 0);
            set_gpu_buf(bnd, &dst, 1);
            set_u32(bnd, n as u32, 2);
        });
        sync(rt);
        let want: Vec<f32> = gpu_bf.iter().copied().map(bf16_bits_to_f32).collect();
        bits_eq("cast_bf16_to_f32", &dst.read_f32()[..n], &want);
    });

    ledger.credit(rt, "zero_f32", &["zero_f32"], || {
        let x = rt.alloc_buffer(n * 4).unwrap();
        x.write_f32(&src_f);
        launch_1d(rt, "zero_f32", n, |bnd| {
            set_gpu_buf(bnd, &x, 0);
            set_u32(bnd, n as u32, 1);
        });
        sync(rt);
        bits_eq("zero_f32", &x.read_f32()[..n], &vec![0.0; n]);
    });

    ledger.credit(rt, "add_inplace_f32", &["add_inplace_f32"], || {
        let dst_h = random_f32(n, 11);
        let src_h = random_f32(n, 13);
        let dst = rt.alloc_buffer(n * 4).unwrap();
        let src = rt.alloc_buffer(n * 4).unwrap();
        dst.write_f32(&dst_h);
        src.write_f32(&src_h);
        launch_1d(rt, "add_inplace_f32", n, |bnd| {
            set_gpu_buf(bnd, &dst, 0);
            set_gpu_buf(bnd, &src, 1);
            set_u32(bnd, n as u32, 2);
        });
        sync(rt);
        let want: Vec<f32> = dst_h.iter().zip(&src_h).map(|(a, b)| a + b).collect();
        bits_eq("add_inplace_f32", &dst.read_f32()[..n], &want);
    });

    let parts = 3usize;
    let slice = n;
    let mut s = vec![0.0f32; parts * slice];
    for p in 0..parts {
        for i in 0..n {
            s[p * slice + i] = (p + 1) as f32 * 0.25 + i as f32 * 0.01;
        }
    }
    ledger.credit(rt, "reduce_partitions_f32", &["reduce_partitions_f32"], || {
        let sb = rt.alloc_buffer(s.len() * 4).unwrap();
        let out = rt.alloc_buffer(n * 4).unwrap();
        sb.write_f32(&s);
        launch_1d(rt, "reduce_partitions_f32", n, |bnd| {
            set_gpu_buf(bnd, &sb, 0);
            set_gpu_buf(bnd, &out, 1);
            set_u32(bnd, n as u32, 2);
            set_u32(bnd, parts as u32, 3);
            set_u32(bnd, slice as u32, 4);
        });
        sync(rt);
        let mut want = vec![0.0f32; n];
        for i in 0..n {
            let mut acc = s[i];
            for p in 1..parts {
                acc += s[p * slice + i];
            }
            want[i] = acc;
        }
        bits_eq("reduce_partitions_f32", &out.read_f32()[..n], &want);
    });
    ledger.credit(rt, "reduce_partitions_add_f32", &["reduce_partitions_add_f32"], || {
        let base = random_f32(n, 19);
        let sb = rt.alloc_buffer(s.len() * 4).unwrap();
        let out = rt.alloc_buffer(n * 4).unwrap();
        sb.write_f32(&s);
        out.write_f32(&base);
        launch_1d(rt, "reduce_partitions_add_f32", n, |bnd| {
            set_gpu_buf(bnd, &sb, 0);
            set_gpu_buf(bnd, &out, 1);
            set_u32(bnd, n as u32, 2);
            set_u32(bnd, parts as u32, 3);
            set_u32(bnd, slice as u32, 4);
        });
        sync(rt);
        let mut want = base.clone();
        for p in 0..parts {
            for i in 0..n {
                want[i] += s[p * slice + i];
            }
        }
        bits_eq("reduce_partitions_add_f32", &out.read_f32()[..n], &want);
    });

    ledger.credit(rt, "transpose2d_f32", &["transpose2d_f32"], || {
        let rows = 5usize;
        let cols = 7usize;
        let host = random_f32(rows * cols, 23);
        let src = tensor_f32(rt, &[rows, cols], &host);
        let dst = rt.alloc_tensor_f32(&[cols, rows]).unwrap();
        transpose_f32_into(&src, &dst).unwrap();
        sync(rt);
        let got = read_c(&dst);
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(
                    got[c * rows + r].to_bits(),
                    host[r * cols + c].to_bits(),
                    "transpose {r},{c}"
                );
            }
        }
    });
}

fn fixed_order(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let (m, n, k) = (4usize, 8usize, 9usize);
    let mut a_nn = vec![0.0f32; m * k];
    let mut b_nn = vec![0.0f32; k * n];
    for i in 0..m {
        for p in 0..k {
            a_nn[i * k + p] = match p {
                0 => 1.0,
                1 => -1.0,
                _ => (p as f32) * 1.0e-6,
            };
        }
    }
    for p in 0..k {
        for j in 0..n {
            b_nn[p * n + j] = 1.0 + (j as f32) * 1.0e-3;
        }
    }
    let want = gemm_f32_cpu(&a_nn, &b_nn, m, n, k);
    ledger.credit(rt, "matmul_fixed_order_nn_f32", &["matmul_fixed_order_nn_f32"], || {
        let a = tensor_f32(rt, &[m, k], &a_nn);
        let b = tensor_f32(rt, &[k, n], &b_nn);
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        gemm_fixed_order_nn(&a, &b, &c).unwrap();
        sync(rt);
        bits_eq("fixed nn", &read_c(&c), &want);
    });
    ledger.credit(rt, "matmul_fixed_order_nt_f32", &["matmul_fixed_order_nt_f32"], || {
        let mut b_nt = vec![0.0f32; n * k];
        for j in 0..n {
            for p in 0..k {
                b_nt[j * k + p] = b_nn[p * n + j];
            }
        }
        let a = tensor_f32(rt, &[m, k], &a_nn);
        let b = tensor_f32(rt, &[n, k], &b_nt);
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        gemm_fixed_order_nt(&a, &b, &c).unwrap();
        sync(rt);
        bits_eq("fixed nt", &read_c(&c), &want);
    });
    ledger.credit(rt, "matmul_fixed_order_tn_f32", &["matmul_fixed_order_tn_f32"], || {
        let mut a_tn = vec![0.0f32; k * m];
        for p in 0..k {
            for i in 0..m {
                a_tn[p * m + i] = a_nn[i * k + p];
            }
        }
        let a = tensor_f32(rt, &[k, m], &a_tn);
        let b = tensor_f32(rt, &[k, n], &b_nn);
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        gemm_fixed_order_tn(&a, &b, &c).unwrap();
        sync(rt);
        bits_eq("fixed tn", &read_c(&c), &want);
    });
}

fn nn_force(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, dtype: DType, tile: EpiTile, kernel: &str, operand_u: f64) {
    let (m, n, k) = (32usize, 64usize, 16usize);
    let case = mats(rt, Layout::Nn, m, n, k, 0x51, dtype);
    let expect = case.expect;
    let kk = case.k;
    ledger.credit(rt, kernel, &[kernel], || {
        gemm_tiled(&case.a, &case.b, &case.c, GemmBackend::TensorOps, tile).unwrap_or_else(|e| panic!("{kernel}: {e}"));
        sync(rt);
        check_bound(kernel, &read_c(&case.c), &expect, kk, operand_u);
    });
}

fn epi_force(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, dtype: DType, tile: EpiTile, kernel: &str, operand_u: f64) {
    let (m, n, k) = (32usize, 64usize, 16usize);
    let case = mats(rt, Layout::Nn, m, n, k, 0x52, dtype);
    let alpha = 2.0f64;
    let mut expect = case.expect;
    for (c, mag) in expect.c.iter_mut().zip(expect.mag.iter_mut()) {
        *c = (alpha * *c).max(0.0);
        *mag = alpha.abs() * *mag;
    }
    let kk = case.k;
    let epi = Epilogue {
        alpha: alpha as f32,
        beta: 0.0,
        bias: None,
        activation: Activation::Relu,
    };
    ledger.credit(rt, kernel, &[kernel], || {
        gemm_epilogue_tiled(&case.a, &case.b, &case.c, GemmBackend::TensorOps, epi, tile)
            .unwrap_or_else(|e| panic!("{kernel}: {e}"));
        sync(rt);
        check_bound(kernel, &read_c(&case.c), &expect, kk, operand_u);
    });
}

fn batched(
    rt: &Arc<GpuRuntime>,
    ledger: &mut Ledger,
    dtype: DType,
    m: usize,
    n: usize,
    k: usize,
    kernel: &str,
    operand_u: f64,
) {
    let batch = 2usize;
    let one = mats(rt, Layout::Nn, m, n, k, 0x53 ^ n as u64, dtype);
    let two = mats(rt, Layout::Nn, m, n, k, 0x54 ^ n as u64, dtype);
    let mut c_ref = one.expect.c.clone();
    let mut mag = one.expect.mag.clone();
    c_ref.extend_from_slice(&two.expect.c);
    mag.extend_from_slice(&two.expect.mag);
    let expect = Reference { c: c_ref, mag };
    let a = match dtype {
        DType::F32 => {
            let mut h = one.a.read_f32().unwrap();
            h.extend(two.a.read_f32().unwrap());
            tensor_f32(rt, &[batch * m, k], &h)
        }
        DType::BF16 => {
            let h = cat_f32(&one.a, &two.a);
            tensor_bf16(rt, &[batch * m, k], &h)
        }
        DType::F16 => {
            let h = cat_f32(&one.a, &two.a);
            tensor_f16(rt, &[batch * m, k], &h)
        }
        _ => unreachable!(),
    };
    let b = match dtype {
        DType::F32 => {
            let mut h = one.b.read_f32().unwrap();
            h.extend(two.b.read_f32().unwrap());
            tensor_f32(rt, &[batch * k, n], &h)
        }
        DType::BF16 => tensor_bf16(rt, &[batch * k, n], &cat_f32(&one.b, &two.b)),
        DType::F16 => tensor_f16(rt, &[batch * k, n], &cat_f32(&one.b, &two.b)),
        _ => unreachable!(),
    };
    let c = rt.alloc_tensor_f32(&[batch * m, n]).unwrap();
    let spec = BatchedGemm {
        m,
        n,
        k,
        batch,
        strides: BatchStrides::contiguous(m, n, k),
    };
    ledger.credit(rt, kernel, &[kernel], || {
        tessl::gemm::gemm_batched(&a, &b, &c, GemmBackend::TensorOps, spec).unwrap_or_else(|e| panic!("{kernel}: {e}"));
        sync(rt);
        check_bound(kernel, &read_c(&c), &expect, k, operand_u);
    });
}

fn cat_f32(a: &Tensor, b: &Tensor) -> Vec<f32> {
    let mut h = a.read_f32().unwrap_or_else(|_| widen(a));
    h.extend(b.read_f32().unwrap_or_else(|_| widen(b)));
    h
}

fn widen(t: &Tensor) -> Vec<f32> {
    let bits = t.buffer.contents_u16();
    let n = t.numel();
    match t.dtype {
        DType::BF16 => bits[..n].iter().copied().map(bf16_bits_to_f32).collect(),
        DType::F16 => bits[..n].iter().copied().map(f16_bits_to_f32).collect(),
        _ => panic!("widen {:?}", t.dtype),
    }
}

fn accum_nn(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let (m, n, k) = (32usize, 32usize, 32usize);
    let case = mats(rt, Layout::Nn, m, n, k, 0x61, DType::F32);
    let prev = random_f32(m * n, 0x62);
    case.c.write_f32(&prev).unwrap();
    let mut expect = case.expect;
    for (i, p) in prev.iter().enumerate() {
        expect.c[i] += f64::from(*p);
        expect.mag[i] += f64::from(p.abs());
    }
    let kk = case.k;
    ledger.credit(
        rt,
        "matmul2d_tensorops_nn_accum_f32",
        &["matmul2d_tensorops_nn_accum_f32"],
        || {
            GemmOperands::ExactF32
                .nn_acc(&case.a, &case.b, &case.c)
                .unwrap_or_else(|e| panic!("nn_accum: {e}"));
            sync(rt);
            check_bound("nn_accum", &read_c(&case.c), &expect, kk, 0.0);
        },
    );
}

fn split_nn(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    // Sequential split: m > 32, k*n >= 2^23, k >= 4 * k_tile, k_tile = (2^21/n)/256*256.
    let (m, n, k) = (33usize, 4096usize, 2048usize);
    let case = mats(rt, Layout::Nn, m, n, k, 0x71, DType::F32);
    let expect = case.expect;
    ledger.credit(
        rt,
        "matmul2d_tensorops_nn_splitk_f32",
        &["matmul2d_tensorops_nn_splitk_f32"],
        || {
            gemm(&case.a, &case.b, &case.c, GemmBackend::TensorOps).unwrap_or_else(|e| panic!("nn splitk: {e}"));
            sync(rt);
            check_bound("nn splitk", &read_c(&case.c), &expect, k, 0.0);
        },
    );
}

fn par_nn(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let (m, n, k) = (32usize, 32usize, 64usize);
    let case = mats(rt, Layout::Nn, m, n, k, 0x72, DType::F32);
    let expect = case.expect;
    ledger.credit(
        rt,
        "matmul2d_tensorops_nn_splitk_par_f32",
        &["matmul2d_tensorops_nn_splitk_par_f32"],
        || {
            gemm_nn_splitk_par_f32(&case.a, &case.b, &case.c, 32).unwrap_or_else(|e| panic!("nn par: {e}"));
            sync(rt);
            check_bound("nn par", &read_c(&case.c), &expect, k, 0.0);
        },
    );
}

fn side(
    rt: &Arc<GpuRuntime>,
    ledger: &mut Ledger,
    layout: Layout,
    dtype: DType,
    m: usize,
    n: usize,
    k: usize,
    kernel: &str,
    call: impl FnOnce(&Tensor, &Tensor, &Tensor),
    operand_u: f64,
) {
    let case = mats(rt, layout, m, n, k, 0x80 ^ k as u64, dtype);
    let expect = case.expect;
    ledger.credit(rt, kernel, &[kernel], || {
        call(&case.a, &case.b, &case.c);
        sync(rt);
        check_bound(kernel, &read_c(&case.c), &expect, k, operand_u);
    });
}

fn accum_side(
    rt: &Arc<GpuRuntime>,
    ledger: &mut Ledger,
    layout: Layout,
    dtype: DType,
    m: usize,
    n: usize,
    k: usize,
    kernel: &str,
    call: impl FnOnce(&Tensor, &Tensor, &Tensor),
    operand_u: f64,
) {
    let case = mats(rt, layout, m, n, k, 0x90 ^ m as u64, dtype);
    let prev = random_f32(m * n, 0x91);
    if dtype == DType::BF16 {
        // C stays f32. Previous contents are exact f32.
    }
    case.c.write_f32(&prev).unwrap();
    let mut expect = case.expect;
    for (i, p) in prev.iter().enumerate() {
        expect.c[i] += f64::from(*p);
        expect.mag[i] += f64::from(p.abs());
    }
    ledger.credit(rt, kernel, &[kernel], || {
        call(&case.a, &case.b, &case.c);
        sync(rt);
        check_bound(kernel, &read_c(&case.c), &expect, k, operand_u);
    });
}

pub fn run(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    assert!(
        rt.has_tensorops(),
        "kernel sweep needs TensorOps to launch the cooperative GEMMs"
    );
    rt.set_precision(PrecisionMode::F32);
    rt.set_relaxed_precision(false);

    utils(rt, ledger);
    fixed_order(rt, ledger);

    ledger.credit(rt, "matmul_simdgroup_f32", &["matmul_simdgroup_f32"], || {
        let case = mats(rt, Layout::Nn, 32, 32, 32, 1, DType::F32);
        gemm_f32(&case.a, &case.b, &case.c, GemmBackend::Simdgroup).unwrap();
        sync(rt);
        check_bound("simd", &read_c(&case.c), &case.expect, case.k, 0.0);
    });
    ledger.credit(
        rt,
        "matmul_simdgroup_edges_f32",
        &["matmul_simdgroup_edges_f32"],
        || {
            let case = mats(rt, Layout::Nn, 17, 19, 13, 2, DType::F32);
            gemm_f32(&case.a, &case.b, &case.c, GemmBackend::Simdgroup).unwrap();
            sync(rt);
            check_bound("simd edges", &read_c(&case.c), &case.expect, case.k, 0.0);
        },
    );

    ledger.credit(rt, "matmul2d_tensorops_f32", &["matmul2d_tensorops_f32"], || {
        let case = mats(rt, Layout::Nn, 32, 32, 32, 4, DType::F32);
        gemm(&case.a, &case.b, &case.c, GemmBackend::TensorOps).unwrap();
        sync(rt);
        check_bound("f32", &read_c(&case.c), &case.expect, case.k, 0.0);
    });

    nn_force(
        rt,
        ledger,
        DType::BF16,
        EpiTile::Wide,
        "matmul2d_tensorops_bf16_f32",
        0.0,
    );
    nn_force(
        rt,
        ledger,
        DType::BF16,
        EpiTile::Narrow,
        "matmul2d_tensorops_bf16_f32_64x64_sg4",
        0.0,
    );
    nn_force(rt, ledger, DType::F16, EpiTile::Wide, "matmul2d_tensorops_f16_f32", 0.0);
    nn_force(
        rt,
        ledger,
        DType::F16,
        EpiTile::Narrow,
        "matmul2d_tensorops_f16_f32_64x64_sg4",
        0.0,
    );

    rt.set_relaxed_precision(true);
    nn_force(
        rt,
        ledger,
        DType::F32,
        EpiTile::Wide,
        "matmul2d_tensorops_f32_relaxed",
        U_BF16,
    );
    nn_force(
        rt,
        ledger,
        DType::F32,
        EpiTile::Narrow,
        "matmul2d_tensorops_f32_relaxed_64x64_sg4",
        U_BF16,
    );
    epi_force(
        rt,
        ledger,
        DType::F32,
        EpiTile::Wide,
        "matmul2d_tensorops_f32_relaxed_epi",
        U_BF16,
    );
    epi_force(
        rt,
        ledger,
        DType::F32,
        EpiTile::Narrow,
        "matmul2d_tensorops_f32_relaxed_epi_64x64_sg4",
        U_BF16,
    );
    batched(
        rt,
        ledger,
        DType::F32,
        32,
        64,
        16,
        "matmul2d_tensorops_f32_relaxed_batched_64x64_sg4",
        U_BF16,
    );
    batched(
        rt,
        ledger,
        DType::F32,
        32,
        520,
        16,
        "matmul2d_tensorops_f32_relaxed_batched",
        U_BF16,
    );
    rt.set_relaxed_precision(false);

    epi_force(
        rt,
        ledger,
        DType::BF16,
        EpiTile::Wide,
        "matmul2d_tensorops_bf16_f32_epi",
        0.0,
    );
    epi_force(
        rt,
        ledger,
        DType::BF16,
        EpiTile::Narrow,
        "matmul2d_tensorops_bf16_f32_epi_64x64_sg4",
        0.0,
    );
    epi_force(
        rt,
        ledger,
        DType::F16,
        EpiTile::Wide,
        "matmul2d_tensorops_f16_f32_epi",
        0.0,
    );
    epi_force(
        rt,
        ledger,
        DType::F16,
        EpiTile::Narrow,
        "matmul2d_tensorops_f16_f32_epi_64x64_sg4",
        0.0,
    );
    epi_force(
        rt,
        ledger,
        DType::F32,
        EpiTile::Wide,
        "matmul2d_tensorops_f32_exact_epi",
        0.0,
    );
    epi_force(
        rt,
        ledger,
        DType::F32,
        EpiTile::Narrow,
        "matmul2d_tensorops_f32_exact_epi_64x64_sg4",
        0.0,
    );

    batched(
        rt,
        ledger,
        DType::BF16,
        32,
        64,
        16,
        "matmul2d_tensorops_bf16_f32_batched_64x64_sg4",
        0.0,
    );
    batched(
        rt,
        ledger,
        DType::BF16,
        128,
        4096,
        8,
        "matmul2d_tensorops_bf16_f32_batched",
        0.0,
    );
    batched(
        rt,
        ledger,
        DType::F16,
        32,
        64,
        16,
        "matmul2d_tensorops_f16_f32_batched_64x64_sg4",
        0.0,
    );
    batched(
        rt,
        ledger,
        DType::F16,
        32,
        520,
        16,
        "matmul2d_tensorops_f16_f32_batched",
        0.0,
    );

    accum_nn(rt, ledger);
    split_nn(rt, ledger);
    par_nn(rt, ledger);

    side(
        rt,
        ledger,
        Layout::Tn,
        DType::F32,
        256,
        512,
        32,
        "matmul2d_tensorops_tn_f32",
        |a, b, c| {
            gemm_tn_f32(a, b, c, GemmBackend::TensorOps).unwrap();
        },
        0.0,
    );
    side(
        rt,
        ledger,
        Layout::Nt,
        DType::F32,
        32,
        32,
        32,
        "matmul2d_tensorops_nt_f32",
        |a, b, c| {
            gemm_nt_f32(a, b, c, GemmBackend::TensorOps).unwrap();
        },
        0.0,
    );
    side(
        rt,
        ledger,
        Layout::Tn,
        DType::BF16,
        64,
        64,
        32,
        "matmul2d_tensorops_tn_bf16_f32",
        |a, b, c| {
            gemm_tn_bf16(a, b, c).unwrap();
        },
        0.0,
    );
    side(
        rt,
        ledger,
        Layout::Nt,
        DType::BF16,
        32,
        32,
        32,
        "matmul2d_tensorops_nt_bf16_f32",
        |a, b, c| {
            gemm_nt_bf16(a, b, c).unwrap();
        },
        0.0,
    );

    side(
        rt,
        ledger,
        Layout::Tn,
        DType::F32,
        32,
        32,
        64,
        "matmul2d_tensorops_tn_splitk_par_f32",
        |a, b, c| {
            gemm_tn_splitk_par_f32(a, b, c, 32).unwrap();
        },
        0.0,
    );
    side(
        rt,
        ledger,
        Layout::Tn,
        DType::BF16,
        32,
        32,
        64,
        "matmul2d_tensorops_tn_splitk_par_bf16_f32",
        |a, b, c| {
            gemm_tn_splitk_par_bf16(a, b, c, 32).unwrap();
        },
        0.0,
    );

    accum_side(
        rt,
        ledger,
        Layout::Tn,
        DType::F32,
        32,
        32,
        32,
        "matmul2d_tensorops_tn_accum_f32",
        |a, b, c| {
            GemmOperands::ExactF32.tn_acc(a, b, c).unwrap();
        },
        0.0,
    );
    accum_side(
        rt,
        ledger,
        Layout::Nt,
        DType::F32,
        32,
        32,
        32,
        "matmul2d_tensorops_nt_accum_f32",
        |a, b, c| {
            GemmOperands::ExactF32.nt_acc(a, b, c).unwrap();
        },
        0.0,
    );
    accum_side(
        rt,
        ledger,
        Layout::Tn,
        DType::BF16,
        64,
        64,
        32,
        "matmul2d_tensorops_tn_accum_bf16_f32",
        |a, b, c| {
            GemmOperands::Bf16.tn_acc(a, b, c).unwrap();
        },
        0.0,
    );
    accum_side(
        rt,
        ledger,
        Layout::Nt,
        DType::BF16,
        32,
        32,
        32,
        "matmul2d_tensorops_nt_accum_bf16_f32",
        |a, b, c| {
            GemmOperands::Bf16.nt_acc(a, b, c).unwrap();
        },
        0.0,
    );
    // Sequential TN split-K is the exact accumulate path at a tall K.
    accum_side(
        rt,
        ledger,
        Layout::Tn,
        DType::F32,
        32,
        32,
        2048,
        "matmul2d_tensorops_tn_splitk_f32",
        |a, b, c| {
            GemmOperands::ExactF32.tn_acc(a, b, c).unwrap();
        },
        0.0,
    );

    let (m, n, k) = (32usize, 32usize, 32usize);
    let case = mats(rt, Layout::Nt, m, n, k, 0xa1, DType::BF16);
    let bias_h = random_f32(n, 0xa2);
    let bias = tensor_f32(rt, &[n], &bias_h);
    let mut expect = case.expect;
    for i in 0..m {
        for j in 0..n {
            expect.c[i * n + j] += f64::from(bias_h[j]);
            expect.mag[i * n + j] += f64::from(bias_h[j].abs());
        }
    }
    ledger.credit(
        rt,
        "matmul2d_tensorops_nt_bf16_f32_bias",
        &["matmul2d_tensorops_nt_bf16_f32_bias"],
        || {
            gemm_nt_bf16_bias(&case.a, &case.b, &case.c, &bias, false).unwrap();
            sync(rt);
            check_bound("nt bias", &read_c(&case.c), &expect, k, 0.0);
        },
    );

    rt.set_precision(PrecisionMode::F32);
    rt.set_relaxed_precision(false);
}
