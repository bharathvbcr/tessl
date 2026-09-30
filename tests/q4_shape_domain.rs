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
use tessl::nn::{self, GateUpDispatch, Q4Bank, Q4MlxBank, Q4MlxLayout, Q4MlxRowVariant, QkvOutputs, QuantShape};

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
                    .expect_err(&format!("group_size {group} (tiled={tiled}) must be refused"));
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
                ("gemv_q4_mlx_blocked", nn::gemv_q4_mlx_blocked(rt, bank, &x32, &y, sh)),
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
                let err = result.expect_err(&format!("{entry}: group_size {group} of {cols} must be refused"));
                assert!(
                    err.contains("group_size") && err.contains(&group.to_string()),
                    "{entry}: {err}"
                );
            }
        }
    });
}

/// The tiled MLX layouts store a partial last tile at full height: the `_i4`
/// kernels address 4-row tiles and `gemv_q4_mlx_blocked` 16-row blocks as if
/// every row existed. A bank sized for exactly `rows` rows used to pass
/// validation, and the kernels then read up to a tile's worth of nibbles and
/// scale/bias pairs past it (17 rows x 2 groups blocked: scale entry 48 read,
/// 34 validated). Each tiled entry must now demand the padded rows; the
/// row-major kernels still take exactly `rows`.
#[test]
fn tiled_mlx_banks_must_hold_their_padded_last_tile() {
    with_gpu(|rt| {
        let (rows, cols, group) = (17usize, 128usize, 64usize);
        let gpr = cols / group;
        let bank_for = |stored_rows: usize, sb_rows: usize| {
            let packed = rt.alloc_buffer(stored_rows * cols / 2).unwrap();
            packed.write_bytes(&vec![0x21u8; stored_rows * cols / 2]);
            let sb = buf_bf16(rt, &vec![0.5f32; sb_rows * gpr * 2]);
            (packed, sb)
        };
        let x32 = buf(rt, &vec![1.0f32; cols]);
        let x16 = buf_bf16(rt, &vec![1.0f32; cols]);
        let (y, y2, y3) = (empty(rt, rows), empty(rt, rows), empty(rt, rows));
        let sh = shape(rows, cols, group);

        // (entry, tile rows, call over one bank used for every matrix operand)
        type Call<'a> = Box<dyn Fn(Q4MlxBank<'_>) -> Result<(), String> + 'a>;
        let entries: Vec<(&str, usize, Call)> = vec![
            (
                "gemv_q4_mlx_blocked",
                16,
                Box::new(|b| nn::gemv_q4_mlx_blocked(rt, b, &x32, &y, sh)),
            ),
            (
                "gemv_q4_mlx_blocked_gate_up_gelu",
                16,
                Box::new(|b| nn::gemv_q4_mlx_gate_up_gelu(rt, b, b, &x32, &y, sh, GateUpDispatch::Blocked, false)),
            ),
            (
                "gemv_q4_mlx_simd_i4",
                4,
                Box::new(|b| nn::gemv_q4_mlx_simd(rt, b, &x16, &y, sh, Q4MlxLayout::Interleaved4, None)),
            ),
            (
                "gemv_q4_mlx_simd_add_i4",
                4,
                Box::new(|b| nn::gemv_q4_mlx_simd(rt, b, &x16, &y, sh, Q4MlxLayout::Interleaved4, Some(&y2))),
            ),
            (
                "gemv_q4_mlx_simd_gate_up_gelu_i4",
                4,
                Box::new(|b| {
                    nn::gemv_q4_mlx_gate_up_gelu(
                        rt,
                        b,
                        b,
                        &x16,
                        &y,
                        sh,
                        GateUpDispatch::Simd(Q4MlxLayout::Interleaved4),
                        false,
                    )
                }),
            ),
            (
                "gemv_q4_mlx_simd_kv_i4",
                4,
                Box::new(|b| nn::gemv_q4_mlx_kv(rt, b, b, &x16, &y, &y2, sh, Q4MlxLayout::Interleaved4)),
            ),
            (
                "gemv_q4_mlx_simd_qkv_i4",
                4,
                Box::new(|b| {
                    nn::gemv_q4_mlx_qkv(
                        rt,
                        b,
                        b,
                        b,
                        &x16,
                        QkvOutputs {
                            q_out: &y,
                            k_out: &y2,
                            v_out: &y3,
                        },
                        rows as u32,
                        rows as u32,
                        cols as u32,
                        group as u32,
                        Q4MlxLayout::Interleaved4,
                    )
                }),
            ),
            (
                "gemm_q4_mlx_simd_i4",
                4,
                Box::new(|b| nn::gemm_q4_mlx(rt, b, &x16, &y, sh, 1, Q4MlxLayout::Interleaved4, None)),
            ),
        ];
        for (entry, tile, call) in &entries {
            let padded = rows.div_ceil(*tile) * tile;
            // Exactly `rows` rows of nibbles: refused on the nibbles.
            let (p, s) = bank_for(rows, padded);
            let err = call(Q4MlxBank {
                packed: &p,
                scales_biases: &s,
            })
            .expect_err(&format!("{entry}: an unpadded nibble buffer must be refused"));
            assert!(err.contains("packed"), "{entry}: {err}");
            // Padded nibbles but exactly `rows` rows of scale/bias pairs.
            let (p, s) = bank_for(padded, rows);
            let err = call(Q4MlxBank {
                packed: &p,
                scales_biases: &s,
            })
            .expect_err(&format!("{entry}: unpadded scale/bias pairs must be refused"));
            assert!(err.contains("scales_biases"), "{entry}: {err}");
            // Padded both: accepted.
            let (p, s) = bank_for(padded, padded);
            call(Q4MlxBank {
                packed: &p,
                scales_biases: &s,
            })
            .unwrap_or_else(|e| panic!("{entry}: a padded bank must be accepted: {e}"));
        }
        rt.synchronize().unwrap();

        // Row-major kernels address rows directly: exactly `rows` is enough.
        let (p, s) = bank_for(rows, rows);
        let bank = Q4MlxBank {
            packed: &p,
            scales_biases: &s,
        };
        nn::gemv_q4_mlx_simd(rt, bank, &x16, &y, sh, Q4MlxLayout::RowMajor, None).unwrap();
        nn::gemv_q4_mlx(rt, bank, &x32, &y, sh, Q4MlxRowVariant::Standard).unwrap();
        rt.synchronize().unwrap();
    });
}
