//! Shape domains the Q4 kernels can actually address.
//!
//! `QuantShape` only demanded `cols % group_size == 0`. The kernels assume
//! more. `gemv_q4` peels each group through `uint` loads at
//! `packed + row * cols / 2 + g * group_size / 2`, which is 4-byte aligned for
//! every row and group only when `group_size % 8 == 0`. The MLX family peels
//! through `uint4` loads and its simdgroup kernels walk K in 512-wide blocks
//! of 16-column packs with a `512 / group_size` scale stride, which needs
//! `group_size` to be one of 32, 64, 128, 256 or 512 (MLX itself quantizes
//! with 32, 64 or 128). Before these checks an unaligned group produced a
//! misaligned load or a silently wrong number rather than an error.

mod common;

use common::{buf, buf_bf16, empty, with_gpu};
use tessl::nn::{self, Q4Bank, Q4MlxBank, Q4MlxLayout, Q4MlxRowVariant, QuantShape};

fn shape(rows: usize, cols: usize, group: usize) -> QuantShape {
    QuantShape {
        rows: rows as u32,
        cols: cols as u32,
        group_size: group as u32,
    }
}

#[test]
fn q4_banks_refuse_group_sizes_the_uint_peel_cannot_address() {
    with_gpu(|rt| {
        let (rows, cols) = (16usize, 32usize);
        let packed = rt.alloc_buffer(rows * cols / 2).unwrap();
        packed.write_bytes(&vec![0x21u8; rows * cols / 2]);
        let x = buf(rt, &vec![1.0f32; cols]);
        let y = empty(rt, rows);

        for group in [2usize, 4] {
            let groups = rows * (cols / group);
            let scales = buf(rt, &vec![0.5f32; groups]);
            let zeros = buf(rt, &vec![0.0f32; groups]);
            let bank = Q4Bank {
                packed: &packed,
                scales: &scales,
                zeros: &zeros,
            };
            for tiled in [false, true] {
                let err = nn::gemv_q4(rt, bank, &x, &y, shape(rows, cols, group), tiled)
                    .expect_err(&format!(
                        "group_size {group} (tiled={tiled}) must be refused"
                    ));
                assert!(
                    err.contains("group_size") && err.contains(&group.to_string()),
                    "tiled={tiled}: {err}"
                );
            }
        }

        // Eight is the boundary and still runs.
        let groups = rows * (cols / 8);
        let scales = buf(rt, &vec![0.5f32; groups]);
        let zeros = buf(rt, &vec![0.0f32; groups]);
        let bank = Q4Bank {
            packed: &packed,
            scales: &scales,
            zeros: &zeros,
        };
        nn::gemv_q4(rt, bank, &x, &y, shape(rows, cols, 8), false).unwrap();
        rt.synchronize().unwrap();
    });
}

#[test]
fn mlx_banks_refuse_group_sizes_outside_the_simd_block_domain() {
    with_gpu(|rt| {
        // (cols, group): every pair divides cleanly, so only the alignment
        // rule can refuse it. 96 is a multiple of 32 that does not divide the
        // 512-wide simdgroup K block.
        for (cols, group) in [(64usize, 8usize), (64, 16), (192, 96)] {
            let rows = 16usize;
            let packed = rt.alloc_buffer(rows * cols / 2).unwrap();
            packed.write_bytes(&vec![0x21u8; rows * cols / 2]);
            let groups = rows * (cols / group);
            let sb = buf_bf16(rt, &vec![1.0f32; groups * 2]);
            let bank = Q4MlxBank {
                packed: &packed,
                scales_biases: &sb,
            };
            let x32 = buf(rt, &vec![1.0f32; cols]);
            let x16 = buf_bf16(rt, &vec![1.0f32; cols]);
            let y = empty(rt, rows);
            let sh = shape(rows, cols, group);

            let attempts: Vec<(&str, Result<(), String>)> = vec![
                (
                    "gemv_q4_mlx",
                    nn::gemv_q4_mlx(rt, bank, &x32, &y, sh, Q4MlxRowVariant::Standard),
                ),
                (
                    "gemv_q4_mlx_blocked",
                    nn::gemv_q4_mlx_blocked(rt, bank, &x32, &y, sh),
                ),
                (
                    "gemv_q4_mlx_simd",
                    nn::gemv_q4_mlx_simd(rt, bank, &x16, &y, sh, Q4MlxLayout::RowMajor, None),
                ),
                (
                    "gemm_q4_mlx",
                    nn::gemm_q4_mlx(rt, bank, &x16, &y, sh, 1, Q4MlxLayout::RowMajor, None),
                ),
            ];
            for (entry, result) in attempts {
                let err = result.expect_err(&format!(
                    "{entry}: group_size {group} of {cols} must be refused"
                ));
                assert!(
                    err.contains("group_size") && err.contains(&group.to_string()),
                    "{entry}: {err}"
                );
            }
        }
    });
}
