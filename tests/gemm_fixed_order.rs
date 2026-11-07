//! Fixed-order f32 GEMM against `gemm_f32_cpu`'s ascending-k left fold.
//!
//! The TensorOps 32×32 tile and the simdgroup tile match that fold on short K
//! and on integer products. They do not match it on order-sensitive products
//! whose K fills a hardware chunk (8×16×16). These kernels are the ones
//! `Numerics::Exact` dispatches, and the comparison is `to_bits`.

mod common;

use common::{tensor_f32, with_gpu};
use tessl::gemm::{gemm_f32_cpu, gemm_fixed_order_nn, gemm_fixed_order_nt, gemm_fixed_order_tn};

fn order_sensitive(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; k * n];
    for i in 0..m {
        for p in 0..k {
            a[i * k + p] = match p {
                0 => 1.0,
                1 => -1.0,
                _ => (p as f32) * 1.0e-6,
            };
        }
    }
    for p in 0..k {
        for j in 0..n {
            b[p * n + j] = 1.0 + (j as f32) * 1.0e-3;
        }
    }
    (a, b)
}

fn integers(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; k * n];
    for (i, x) in a.iter_mut().enumerate() {
        *x = ((i % 7) as f32) - 3.0;
    }
    for (i, x) in b.iter_mut().enumerate() {
        *x = ((i % 5) as f32) - 2.0;
    }
    (a, b)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn nt_stored(b_nn: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut b = vec![0.0f32; n * k];
    for j in 0..n {
        for p in 0..k {
            b[j * k + p] = b_nn[p * n + j];
        }
    }
    b
}

fn tn_stored(a_nn: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; k * m];
    for p in 0..k {
        for i in 0..m {
            a[p * m + i] = a_nn[i * k + p];
        }
    }
    a
}

#[test]
fn fixed_order_matches_the_cpu_left_fold_on_order_sensitive_products() {
    with_gpu(|rt| {
        rt.set_relaxed_precision(false);
        let shapes = [(3usize, 5usize, 7usize), (8, 16, 16), (4, 8, 9), (2, 2, 2)];
        for &(m, n, k) in &shapes {
            for (label, (a_nn, b_nn)) in [("order", order_sensitive(m, n, k)), ("integer", integers(m, n, k))] {
                let want = gemm_f32_cpu(&a_nn, &b_nn, m, n, k);

                let a = tensor_f32(rt, &[m, k], &a_nn);
                let b = tensor_f32(rt, &[k, n], &b_nn);
                let c = rt.alloc_tensor_f32(&[m, n]).expect("c");
                gemm_fixed_order_nn(&a, &b, &c).expect("nn");
                rt.synchronize().expect("sync");
                let got = c.buffer.read_f32()[..m * n].to_vec();
                assert_eq!(bits(&got), bits(&want), "nn {label} {m}x{n}x{k}");

                let b_nt = nt_stored(&b_nn, n, k);
                let bn = tensor_f32(rt, &[n, k], &b_nt);
                let c_nt = rt.alloc_tensor_f32(&[m, n]).expect("c nt");
                gemm_fixed_order_nt(&a, &bn, &c_nt).expect("nt");
                rt.synchronize().expect("sync");
                let got_nt = c_nt.buffer.read_f32()[..m * n].to_vec();
                assert_eq!(bits(&got_nt), bits(&want), "nt {label} {m}x{n}x{k}");

                let a_tn = tn_stored(&a_nn, m, k);
                let at = tensor_f32(rt, &[k, m], &a_tn);
                let c_tn = rt.alloc_tensor_f32(&[m, n]).expect("c tn");
                gemm_fixed_order_tn(&at, &b, &c_tn).expect("tn");
                rt.synchronize().expect("sync");
                let got_tn = c_tn.buffer.read_f32()[..m * n].to_vec();
                assert_eq!(bits(&got_tn), bits(&want), "tn {label} {m}x{n}x{k}");
            }
        }
    });
}
