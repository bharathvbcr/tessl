//! Numeric tests for the quantized GEMVs against a CPU reference.
//!
//! `gemv_q4` (signed nibble) had only a row-versus-tiled agreement test and
//! `gemv_q8` had none: both were checked by the benchmark's `verify` and by
//! error-path tests, and nothing in the suite ran them for a number. Each is
//! now compared against an f64 reference at the group sizes and edge counts
//! that exercise every lane assignment of the simdgroup kernels: a chunk that
//! ends exactly on the 256-column simdgroup step, one that spills a single
//! chunk past it, the smallest group the host admits, a group wider than a
//! step, and a row count that leaves a simdgroup half used.

mod common;

use common::{buf, close_rel, random_f32, with_gpu};
use tessl::nn;

/// Two's-complement nibbles packed low-first, as the kernel reads them.
fn pack_signed_nibbles(q: &[i8]) -> Vec<u8> {
    q.chunks(2)
        .map(|pair| {
            let lo = (pair[0] as u8) & 0x0f;
            let hi = (pair.get(1).copied().unwrap_or(0) as u8) & 0x0f;
            lo | (hi << 4)
        })
        .collect()
}

fn q4_reference(
    q: &[i8],
    scales: &[f32],
    zeros: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    group: usize,
) -> Vec<f32> {
    let groups_per_row = cols / group;
    (0..rows)
        .map(|r| {
            let mut acc = 0.0f64;
            for c in 0..cols {
                let g = r * groups_per_row + c / group;
                let w = scales[g] as f64 * (q[r * cols + c] as f64 - zeros[g] as f64);
                acc += w * x[c] as f64;
            }
            acc as f32
        })
        .collect()
}

#[test]
fn gemv_q4_matches_an_f64_reference_at_every_lane_boundary() {
    with_gpu(|rt| {
        for &(rows, cols, group) in &[
            // One simdgroup step exactly, and one chunk past it.
            (16usize, 256usize, 32usize),
            (16, 264, 8),
            // The smallest admitted group; a group wider than a step.
            (24, 512, 8),
            (24, 512, 256),
            // Ragged rows: 13 rows leave a simdgroup half used; 300 rows leave
            // a threadgroup half used.
            (13, 768, 64),
            (300, 512, 32),
            // Model-shaped.
            (64, 4096, 128),
        ] {
            let q: Vec<i8> = (0..rows * cols)
                .map(|i| ((i * 37 + 11) % 16) as i8 - 8)
                .collect();
            let groups = rows * (cols / group);
            let scales: Vec<f32> = (0..groups).map(|i| 0.01 + (i % 7) as f32 * 0.003).collect();
            let zeros: Vec<f32> = (0..groups).map(|i| -0.5 + (i % 5) as f32 * 0.25).collect();
            let x = random_f32(cols, 0xA4 + cols as u64);
            let want = q4_reference(&q, &scales, &zeros, &x, rows, cols, group);

            let packed = rt.alloc_buffer(rows * cols / 2).unwrap();
            packed.write_bytes(&pack_signed_nibbles(&q));
            let sb = buf(rt, &scales);
            let zb = buf(rt, &zeros);
            let xb = buf(rt, &x);
            let yb = buf(rt, &vec![f32::NAN; rows]);
            let shape = nn::QuantShape {
                rows: rows as u32,
                cols: cols as u32,
                group_size: group as u32,
            };
            let bank = nn::Q4Bank {
                packed: &packed,
                scales: &sb,
                zeros: &zb,
            };
            nn::gemv_q4(rt, bank, &xb, &yb, shape, false).unwrap();
            rt.synchronize().unwrap();
            close_rel(
                &format!("gemv_q4 {rows}x{cols} g{group}"),
                &yb.read_f32()[..rows],
                &want,
                2e-4,
            );
        }
    });
}

fn q8_reference(
    q: &[i8],
    scales: &[f32],
    zeros: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    group: usize,
) -> Vec<f32> {
    let groups_per_row = cols / group;
    (0..rows)
        .map(|r| {
            let mut acc = 0.0f64;
            for c in 0..cols {
                let g = r * groups_per_row + c / group;
                let w = scales[g] as f64 * (q[r * cols + c] as f64 - zeros[g] as f64);
                acc += w * x[c] as f64;
            }
            acc as f32
        })
        .collect()
}

#[test]
fn gemv_q8_matches_an_f64_reference_on_both_load_paths() {
    with_gpu(|rt| {
        for &(rows, cols, group) in &[
            // vec4 path: group and cols multiples of four.
            (16usize, 256usize, 64usize),
            (13, 768, 32),
            (300, 512, 128),
            // Scalar path: a group that is not a multiple of four.
            (16, 96, 6),
            (9, 90, 10),
            // Model-shaped.
            (64, 4096, 64),
        ] {
            let q: Vec<i8> = (0..rows * cols)
                .map(|i| ((i * 53 + 7) % 251) as i32 as i8)
                .collect();
            let groups = rows * (cols / group);
            let scales: Vec<f32> = (0..groups)
                .map(|i| 0.004 + (i % 9) as f32 * 0.001)
                .collect();
            let zeros: Vec<f32> = (0..groups).map(|i| -3.0 + (i % 4) as f32).collect();
            let x = random_f32(cols, 0xB8 + cols as u64);
            let want = q8_reference(&q, &scales, &zeros, &x, rows, cols, group);

            let packed = rt.alloc_buffer(rows * cols).unwrap();
            packed.write_bytes(&q.iter().map(|v| *v as u8).collect::<Vec<u8>>());
            let sb = buf(rt, &scales);
            let zb = buf(rt, &zeros);
            let xb = buf(rt, &x);
            let yb = buf(rt, &vec![f32::NAN; rows]);
            nn::gemv_q8(
                rt,
                &packed,
                &sb,
                &zb,
                &xb,
                &yb,
                rows as u32,
                cols as u32,
                group as u32,
            )
            .unwrap();
            rt.synchronize().unwrap();
            close_rel(
                &format!("gemv_q8 {rows}x{cols} g{group}"),
                &yb.read_f32()[..rows],
                &want,
                2e-4,
            );
        }
    });
}
