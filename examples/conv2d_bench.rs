//! `tessl::conv::conv2d_nchw` at SpikeTrack's SAM 3x3 shapes, wall time per call (encode, GPU and
//! the wait), f16 and f32 interleaved in rounds. Usage: `conv2d_bench [rounds] [reps]`.

use std::time::Instant;

use tessl::conv::{conv2d_nchw, ConvPrecision};
use tessl::runtime::GpuRuntime;

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let (rounds, reps) = (args.first().copied().unwrap_or(9), args.get(1).copied().unwrap_or(10));
    let rt = GpuRuntime::new().expect("runtime");
    // (name, n, c_in, c_out, h, w)
    let shapes = [
        ("288^2 256->256", 1, 256, 256, 288, 288),
        ("144^2 256->256", 1, 256, 256, 144, 144),
        ("72^2 256->256", 1, 256, 256, 72, 72),
        ("1152^2 32->16", 1, 32, 16, 1152, 1152),
        ("144^2 256->1024", 1, 256, 1024, 144, 144),
    ];
    let arms = [("f16", ConvPrecision::F16), ("f32", ConvPrecision::F32)];
    println!("shape\tarm\tmin_ms\tmedian_ms\ttflops_at_min");
    for (name, n, ci, co, h, w) in shapes {
        let x = rt.alloc_tensor_f32(&[n, ci, h, w]).unwrap();
        x.write_f32(
            &(0..n * ci * h * w)
                .map(|i| ((i % 97) as f32 - 48.0) / 48.0)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let wt = rt.alloc_tensor_f32(&[co, ci, 3, 3]).unwrap();
        wt.write_f32(
            &(0..co * ci * 9)
                .map(|i| ((i % 89) as f32 - 44.0) / 880.0)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let b = rt.alloc_tensor_f32(&[co]).unwrap();
        let y = rt.alloc_tensor_f32(&[n, co, h, w]).unwrap();
        let flops = 2.0 * (n * co * h * w) as f64 * (ci * 9) as f64;
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for round in 0..rounds + 2 {
            for (i, (_, precision)) in arms.iter().enumerate() {
                rt.synchronize().unwrap();
                let t0 = Instant::now();
                for _ in 0..reps {
                    conv2d_nchw(&x, &wt, Some(&b), &y, 1, *precision).unwrap();
                }
                rt.synchronize().unwrap();
                if round >= 2 {
                    times[i].push(t0.elapsed().as_secs_f64() * 1e3 / reps as f64);
                }
            }
        }
        for (i, (arm, _)) in arms.iter().enumerate() {
            let mut t = times[i].clone();
            t.sort_by(f64::total_cmp);
            println!(
                "{name}\t{arm}\t{:.3}\t{:.3}\t{:.2}",
                t[0],
                t[t.len() / 2],
                flops / t[0] / 1e9
            );
        }
    }
}
