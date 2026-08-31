//! Decisive check: do the TensorOps and simdgroup GEMM lanes agree bit-for-bit?
//! Separate C buffers (no reuse), compared in-process against a CPU f64 reference.
use tessl::gemm::{cast_f32_to_bf16, gemm, gemm_f32_cpu, GemmBackend};
use tessl::runtime::GpuRuntime;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((((s >> 32) as u32) as f64 / u32::MAX as f64) * 2.0 - 1.0) as f32
    }).collect()
}

fn main() -> Result<(), String> {
    let rt = GpuRuntime::new()?;
    let (m, n, k) = (256usize, 256usize, 1024usize);
    let a = rt.alloc_tensor_f32(&[m, k])?;
    let b = rt.alloc_tensor_f32(&[k, n])?;
    let c_t = rt.alloc_tensor_f32(&[m, n])?;
    let c_s = rt.alloc_tensor_f32(&[m, n])?;
    let ah = fill(m * k, 1);
    let bh = fill(k * n, 2);
    a.buffer.write_f32(&ah);
    b.buffer.write_f32(&bh);

    gemm(&a, &b, &c_t, GemmBackend::TensorOps)?;
    rt.synchronize()?;
    let ct = c_t.buffer.read_f32()[..m * n].to_vec();

    gemm(&a, &b, &c_s, GemmBackend::Simdgroup)?;
    rt.synchronize()?;
    let cs = c_s.buffer.read_f32()[..m * n].to_vec();

    let cpu = gemm_f32_cpu(&ah, &bh, m, n, k);

    let mut ref64 = vec![0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f64;
            for kk in 0..k { acc += ah[i * k + kk] as f64 * bh[kk * n + j] as f64; }
            ref64[i * n + j] = acc;
        }
    }

    let a_bf = cast_f32_to_bf16(&a)?;
    let b_bf = cast_f32_to_bf16(&b)?;
    let c_bf = rt.alloc_tensor_f32(&[m, n])?;
    gemm(&a_bf, &b_bf, &c_bf, GemmBackend::TensorOps)?;
    rt.synchronize()?;
    let cb = c_bf.buffer.read_f32()[..m * n].to_vec();

    let diff_ts = ct.iter().zip(&cs).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    let diff_tc = ct.iter().zip(&cpu).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    let err = |v: &[f32]| v.iter().zip(&ref64).map(|(x, r)| (*x as f64 - r).abs()).fold(0.0, f64::max);
    println!("M={m} N={n} K={k}  ({} elems)", m * n);
    println!("  tensorops vs simdgroup: {diff_ts} differing floats");
    println!("  tensorops vs cpu-f32  : {diff_tc} differing floats");
    println!("  max|err vs f64 ref| tensorops={:.3e} simdgroup={:.3e} cpu={:.3e}",
             err(&ct), err(&cs), err(&cpu));
    println!("  bf16 lane: max|err vs f64 ref|={:.3e}  rel={:.3e}",
             err(&cb), err(&cb) / ref64.iter().fold(0.0f64, |a, r| a.max(r.abs())));
    println!("  samples t/s/cpu: {:?} {:?} {:?}", &ct[..3], &cs[..3], &cpu[..3]);
    Ok(())
}
