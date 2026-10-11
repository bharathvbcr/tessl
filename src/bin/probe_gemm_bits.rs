//! A bit snapshot of every production GEMM entry point: the FNV-1a hash of
//! each output over a fixed shape grid and fixed inputs, one line per
//! (entry point, shape). Two builds whose snapshots are equal produced the
//! same bits everywhere the grid reaches, so a refactor that must not change
//! numerics (a shader moved into a shared header, a new tile walk) diffs two
//! runs instead of trusting tolerance tests.
//!
//! ```text
//! git checkout BEFORE && cargo run --release --bin probe_gemm_bits > before.txt
//! git checkout AFTER  && cargo run --release --bin probe_gemm_bits > after.txt
//! diff before.txt after.txt
//! ```
//!
//! Covered: NN exact f32, bf16, f16 and relaxed f32 (production tile and
//! both explicit tiles), their fused epilogues, `GemmOperands` NN/TN/NT
//! accumulates, strided-batched NN, TN/NT f32 and bf16, the parallel f32
//! split-K and the int8 dequant kernel with and without `b_scale`. A call the
//! API refuses prints `ERR` and its message, so a refusal that appears or
//! disappears shows in the diff too. bench/results/gemm_coop_header_bits_m5pro.txt
//! records the runs that held the shared-header and walk commits to their
//! parents' bits.
use std::sync::Arc;
use tessl::gemm::*;
use tessl::nn::gemm_i8_dequant;
use tessl::runtime::{GpuRuntime, PrecisionMode};
use tessl::tensor::Tensor;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((((s >> 32) as u32) as f64 / u32::MAX as f64) * 2.0 - 1.0) as f32
        })
        .collect()
}
fn f32t(rt: &Arc<GpuRuntime>, shape: &[usize], seed: u64) -> Tensor {
    let t = rt.alloc_tensor_f32(shape).unwrap();
    t.buffer.write_f32(&fill(shape.iter().product(), seed));
    t
}
fn hash(c: &Tensor, n: usize) -> u64 {
    let v = c.buffer.read_f32();
    let mut h: u64 = 0xcbf29ce484222325;
    for x in &v[..n] {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}
fn line(rt: &Arc<GpuRuntime>, name: &str, (m, n, k): (usize, usize, usize), c: &Tensor, r: Result<(), String>) {
    match r {
        Ok(()) => {
            rt.synchronize().unwrap();
            println!("{name:<28} {m}x{n}x{k} {:016x}", hash(c, m * n));
        }
        Err(e) => println!("{name:<28} {m}x{n}x{k} ERR {e}"),
    }
}

fn main() {
    let rt = GpuRuntime::new().unwrap();
    let to = GemmBackend::TensorOps;
    let nn_shapes = [
        (256, 256, 256),
        (200, 72, 96),
        (61, 2048, 2048),
        (128, 512, 128),
        (130, 97, 70),
        (512, 16384, 512),
        (4096, 4096, 256),
        (2048, 1024, 64),
        (1024, 1024, 4096),
        (61, 640, 256),
        (2048, 6144, 2048),
        (512, 768, 3072),
    ];
    let tnt_shapes = [
        (256, 256, 256),
        (200, 72, 96),
        (130, 97, 70),
        (128, 384, 4096),
        (128, 128, 4096),
        (12, 768, 4096),
        (512, 8192, 1024),
        (256, 8192, 1024),
        (2048, 2048, 2048),
        (64, 64, 8192),
    ];
    let seed = |m: usize, n: usize, k: usize, s: u64| (m * 1_000_003 + n * 1009 + k) as u64 ^ s;
    rt.set_precision(PrecisionMode::F32);
    for &(m, n, k) in &nn_shapes {
        let a = f32t(&rt, &[m, k], seed(m, n, k, 1));
        let b = f32t(&rt, &[k, n], seed(m, n, k, 2));
        let a16 = cast_f32_to_bf16(&a).unwrap();
        let b16 = cast_f32_to_bf16(&b).unwrap();
        let ah = cast_f32_to_f16(&a).unwrap();
        let bh = cast_f32_to_f16(&b).unwrap();
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        rt.set_relaxed_precision(false);
        line(&rt, "nn_f32_exact", (m, n, k), &c, gemm(&a, &b, &c, to));
        line(&rt, "nn_bf16", (m, n, k), &c, gemm(&a16, &b16, &c, to));
        line(
            &rt,
            "nn_bf16_wide",
            (m, n, k),
            &c,
            gemm_tiled(&a16, &b16, &c, to, EpiTile::Wide),
        );
        line(
            &rt,
            "nn_bf16_narrow",
            (m, n, k),
            &c,
            gemm_tiled(&a16, &b16, &c, to, EpiTile::Narrow),
        );
        line(&rt, "nn_f16", (m, n, k), &c, gemm(&ah, &bh, &c, to));
        line(
            &rt,
            "nn_f16_narrow",
            (m, n, k),
            &c,
            gemm_tiled(&ah, &bh, &c, to, EpiTile::Narrow),
        );
        let c0 = f32t(&rt, &[m, n], seed(m, n, k, 3));
        let c0v = c0.buffer.read_f32();
        let reset = |c: &Tensor| c.buffer.write_f32(&c0v);
        reset(&c);
        line(
            &rt,
            "nn_acc_f32",
            (m, n, k),
            &c,
            GemmOperands::ExactF32.nn_acc(&a, &b, &c),
        );
        reset(&c);
        line(&rt, "nn_acc_bf16", (m, n, k), &c, GemmOperands::Bf16.nn_acc(&a, &b, &c));
        let bias = f32t(&rt, &[n], seed(m, n, k, 4));
        for (ename, elem_a, elem_b) in [("bf16", &a16, &b16), ("f16", &ah, &bh)] {
            for (tname, tile) in [
                ("wide", Some(EpiTile::Wide)),
                ("narrow", Some(EpiTile::Narrow)),
                ("auto", None),
            ] {
                for (k_epi, epi) in [
                    (
                        "beta1",
                        Epilogue {
                            beta: 1.0,
                            ..Default::default()
                        },
                    ),
                    (
                        "full",
                        Epilogue {
                            alpha: 0.5,
                            beta: 0.25,
                            bias: Some(&bias),
                            activation: Activation::GeluTanh,
                        },
                    ),
                    (
                        "silu",
                        Epilogue {
                            bias: Some(&bias),
                            activation: Activation::Silu,
                            ..Default::default()
                        },
                    ),
                    (
                        "relu_a",
                        Epilogue {
                            alpha: 2.0,
                            activation: Activation::Relu,
                            ..Default::default()
                        },
                    ),
                ] {
                    reset(&c);
                    let r = match tile {
                        Some(t) => gemm_epilogue_tiled(elem_a, elem_b, &c, to, epi, t),
                        None => gemm_epilogue(elem_a, elem_b, &c, to, epi),
                    };
                    line(&rt, &format!("epi_{ename}_{tname}_{k_epi}"), (m, n, k), &c, r);
                }
            }
        }
        rt.set_relaxed_precision(true);
        line(&rt, "nn_relaxed", (m, n, k), &c, gemm(&a, &b, &c, to));
        line(
            &rt,
            "nn_relaxed_narrow",
            (m, n, k),
            &c,
            gemm_tiled(&a, &b, &c, to, EpiTile::Narrow),
        );
        reset(&c);
        line(
            &rt,
            "epi_relaxed_full",
            (m, n, k),
            &c,
            gemm_epilogue(
                &a,
                &b,
                &c,
                to,
                Epilogue {
                    alpha: 0.5,
                    beta: 0.25,
                    bias: Some(&bias),
                    activation: Activation::GeluTanh,
                },
            ),
        );
        rt.set_relaxed_precision(false);
    }
    // Batched.
    for &(m, n, k, batch) in &[
        (64usize, 64usize, 64usize, 3usize),
        (200, 72, 96, 3),
        (61, 640, 256, 2),
        (256, 1024, 128, 4),
    ] {
        let a = f32t(&rt, &[batch * m, k], seed(m, n, k, 5));
        let b = f32t(&rt, &[batch * k, n], seed(m, n, k, 6));
        let c = rt.alloc_tensor_f32(&[batch * m, n]).unwrap();
        for (sname, strides) in [
            ("contig", BatchStrides::contiguous(m, n, k)),
            ("shared_b", BatchStrides::shared_b(m, n, k)),
        ] {
            let spec = BatchedGemm {
                m,
                n,
                k,
                batch,
                strides,
            };
            let (a16, b16) = (cast_f32_to_bf16(&a).unwrap(), cast_f32_to_bf16(&b).unwrap());
            let (ah, bh) = (cast_f32_to_f16(&a).unwrap(), cast_f32_to_f16(&b).unwrap());
            line(
                &rt,
                &format!("batched_bf16_{sname}"),
                (m, n, k * batch),
                &c,
                gemm_batched(&a16, &b16, &c, to, spec),
            );
            line(
                &rt,
                &format!("batched_f16_{sname}"),
                (m, n, k * batch),
                &c,
                gemm_batched(&ah, &bh, &c, to, spec),
            );
            rt.set_relaxed_precision(true);
            line(
                &rt,
                &format!("batched_relaxed_{sname}"),
                (m, n, k * batch),
                &c,
                gemm_batched(&a, &b, &c, to, spec),
            );
            rt.set_relaxed_precision(false);
        }
        // Each line above hashes the first matrix; this one hashes all of C
        // after the last call.
        rt.synchronize().unwrap();
        println!(
            "batched_last_full            {m}x{n}x{k}x{batch} {:016x}",
            hash(&c, batch * m * n)
        );
    }
    // TN / NT.
    for &(m, n, k) in &tnt_shapes {
        let a_km = f32t(&rt, &[k, m], seed(m, n, k, 7));
        let b_kn = f32t(&rt, &[k, n], seed(m, n, k, 8));
        let a_mk = f32t(&rt, &[m, k], seed(m, n, k, 9));
        let b_nk = f32t(&rt, &[n, k], seed(m, n, k, 10));
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        line(&rt, "tn_f32", (m, n, k), &c, gemm_tn_f32(&a_km, &b_kn, &c, to));
        line(&rt, "nt_f32", (m, n, k), &c, gemm_nt_f32(&a_mk, &b_nk, &c, to));
        line(&rt, "tn_bf16", (m, n, k), &c, gemm_tn_bf16(&a_km, &b_kn, &c));
        line(&rt, "nt_bf16", (m, n, k), &c, gemm_nt_bf16(&a_mk, &b_nk, &c));
        let c0 = f32t(&rt, &[m, n], seed(m, n, k, 11));
        let c0v = c0.buffer.read_f32();
        for (name, lane) in [("f32", GemmOperands::ExactF32), ("bf16", GemmOperands::Bf16)] {
            c.buffer.write_f32(&c0v);
            line(
                &rt,
                &format!("tn_acc_{name}"),
                (m, n, k),
                &c,
                lane.tn_acc(&a_km, &b_kn, &c),
            );
            c.buffer.write_f32(&c0v);
            line(
                &rt,
                &format!("nt_acc_{name}"),
                (m, n, k),
                &c,
                lane.nt_acc(&a_mk, &b_nk, &c),
            );
        }
        if m * n <= 1 << 20 {
            line(
                &rt,
                "tn_splitk_par_f32_512",
                (m, n, k),
                &c,
                gemm_tn_splitk_par_f32(&a_km, &b_kn, &c, 512),
            );
        }
    }
    // int8.
    for &(m, n, k) in &[
        (200usize, 72usize, 96usize),
        (256, 256, 256),
        (512, 8192, 1024),
        (130, 97, 70),
    ] {
        let i8v = |len: usize, s: u64| -> Vec<u8> {
            fill(len, s)
                .iter()
                .map(|&x| ((x * 127.0).round() as i8) as u8)
                .collect()
        };
        let a = rt.alloc_buffer(m * k).unwrap();
        a.write_bytes(&i8v(m * k, seed(m, n, k, 12)));
        let b = rt.alloc_buffer(k * n).unwrap();
        b.write_bytes(&i8v(k * n, seed(m, n, k, 13)));
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        let scale = f32t(&rt, &[n], seed(m, n, k, 14));
        let (mu, nu, ku) = (m as u32, n as u32, k as u32);
        line(
            &rt,
            "i8",
            (m, n, k),
            &c,
            gemm_i8_dequant(&rt, &a, &b, &c.buffer, mu, nu, ku, 0.01, None),
        );
        line(
            &rt,
            "i8_bscale",
            (m, n, k),
            &c,
            gemm_i8_dequant(&rt, &a, &b, &c.buffer, mu, nu, ku, 0.01, Some(&scale.buffer)),
        );
    }
}
