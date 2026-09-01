//! Numeric tests for promoted kernels that had only a name check.
//!
//! `promoted_kernels.rs` asserts each of the 44 promoted entry points resolves
//! out of tessl's own metallib. That is a real gate on the move, and it is not
//! a correctness test: a kernel can resolve, dispatch, and return wrong numbers.
//!
//! `gemv_q4_tiled` proved the gap was not theoretical. It resolved, it had an
//! adversarial test covering its error paths, and it wrote 4 rows of 512 —
//! because the host handed it the *other* Q4 kernel's grid. Nothing ran it for
//! a number, so nothing noticed.
//!
//! The families here are the ones where that shape of bug hides: several
//! kernels behind one entry point, selected by an enum or a bool, each wanting
//! a different grid. Every test uses `rows` above the 128 that the
//! one-thread-per-row kernels group by, because at smaller sizes the competing
//! grids coincide and a mismatch is invisible.

mod common;

use common::{
    buf, buf_bf16, close_rel, dense_gemv, empty, q4_mlx_matrix, random_f32, round_trip_bf16,
    seeded, with_gpu,
};
use tessl::nn::{self, Q4MlxBank, Q4MlxLayout, Q4MlxRowVariant, QuantShape};

/// Sentinel for outputs, to separate "wrote a wrong value" from "never wrote".
const UNWRITTEN: f32 = -7.7e28;

fn shape(rows: usize, cols: usize, group: usize) -> QuantShape {
    QuantShape {
        rows: rows as u32,
        cols: cols as u32,
        group_size: group as u32,
    }
}

fn assert_all_written(what: &str, got: &[f32]) {
    let n = got.iter().filter(|v| **v == UNWRITTEN).count();
    assert_eq!(
        n,
        0,
        "{what}: {n} of {} outputs were never written",
        got.len()
    );
}

// ------------------------------------------------- MLX Q4 GEMV variants ---

/// All three row variants against one dense reference.
///
/// `Standard`, `Wide` and `Tiled` are one enum away from each other and only
/// `Standard` was tested. `Tiled` is the one that takes a threadgroup-per-row
/// grid, which is exactly the arrangement that was wrong in `gemv_q4`.
#[test]
fn every_q4_mlx_row_variant_matches_the_dense_reference() {
    with_gpu(|rt| {
        let (rows, cols, group) = (512usize, 256usize, 32usize);
        let (packed, sb, dense) = q4_mlx_matrix(rows, cols, group);
        let x = random_f32(cols, 0xF101);
        let want = dense_gemv(&dense, &x, rows, cols);

        let pb = rt.alloc_buffer(packed.len()).unwrap();
        pb.write_bytes(&packed);
        let sbb = buf_bf16(rt, &sb);
        let xb = buf(rt, &x);
        let bank = Q4MlxBank {
            packed: &pb,
            scales_biases: &sbb,
        };

        for (name, variant) in [
            ("Standard", Q4MlxRowVariant::Standard),
            ("Wide", Q4MlxRowVariant::Wide),
            ("Tiled", Q4MlxRowVariant::Tiled),
        ] {
            let yb = seeded(rt, rows, UNWRITTEN);
            nn::gemv_q4_mlx(rt, bank, &xb, &yb, shape(rows, cols, group), variant).unwrap();
            rt.synchronize().unwrap();
            let got = yb.read_f32();
            assert_all_written(&format!("gemv_q4_mlx[{name}]"), &got[..rows]);
            close_rel(&format!("gemv_q4_mlx[{name}]"), &got[..rows], &want, 3e-3);
        }
    });
}

/// Rows per block in `gemv_q4_mlx_blocked`'s weight layout (`GEMV_BN`).
const BLOCKED_BN: usize = 16;

/// Repack a row-major MLX Q4 bank into the block-interleaved layout
/// `gemv_q4_mlx_blocked` indexes.
///
/// Within block `b`, group `g` and row `r` of that block, the kernel reads
/// scale/bias and nibbles at `b * groups_per_row * 16 + g * 16 + r`, where the
/// row-major kernels read `row * groups_per_row + g`. This is the reference
/// implementation `nn::gemv_q4_mlx_blocked`'s documentation points at.
fn to_blocked_bank(
    nibbles: &[u8],
    sb: &[f32],
    rows: usize,
    cols: usize,
    group: usize,
) -> (Vec<u8>, Vec<f32>) {
    let gpr = cols / group;
    let mut nb = vec![0u8; rows * cols];
    let mut out_sb = vec![0.0f32; rows * gpr * 2];
    for b in 0..rows.div_ceil(BLOCKED_BN) {
        for r_local in 0..BLOCKED_BN {
            let r = b * BLOCKED_BN + r_local;
            if r >= rows {
                continue;
            }
            for g in 0..gpr {
                let dst = b * gpr * BLOCKED_BN + g * BLOCKED_BN + r_local;
                out_sb[dst * 2] = sb[(r * gpr + g) * 2];
                out_sb[dst * 2 + 1] = sb[(r * gpr + g) * 2 + 1];
                nb[dst * group..(dst + 1) * group]
                    .copy_from_slice(&nibbles[r * cols + g * group..r * cols + (g + 1) * group]);
            }
        }
    }
    (common::pack_nibbles(&nb), out_sb)
}

/// `gemv_q4_mlx_blocked` against the dense reference, with the bank in the
/// block-interleaved layout it requires.
///
/// This entry point takes the same `Q4MlxBank` type as its row-major siblings
/// and silently returns wrong numbers when handed a row-major bank — the type
/// carries no layout tag, so nothing catches it. Measured at 64x256 with
/// `group_size` 64: 63 of 64 rows wrong row-major, 0 of 64 repacked. The two
/// layouts coincide only when `groups_per_row == 1`, which is exactly the shape
/// a small smoke test would have picked.
#[test]
fn q4_mlx_blocked_matches_the_dense_reference_in_its_own_layout() {
    with_gpu(|rt| {
        for &(rows, cols, group) in &[(512usize, 256usize, 32usize), (304, 512, 64)] {
            // Same nibble stream `q4_mlx_matrix` builds, so `dense` describes
            // these weights.
            let nibbles: Vec<u8> = (0..rows * cols).map(|i| ((i * 5) % 16) as u8).collect();
            let (_row_major, sb, dense) = q4_mlx_matrix(rows, cols, group);
            let x = random_f32(cols, 0xF202 + cols as u64);
            let want = dense_gemv(&dense, &x, rows, cols);

            let (packed, sb_blocked) = to_blocked_bank(&nibbles, &sb, rows, cols, group);
            let pb = rt.alloc_buffer(packed.len()).unwrap();
            pb.write_bytes(&packed);
            let sbb = buf_bf16(rt, &sb_blocked);
            let xb = buf(rt, &x);
            let yb = seeded(rt, rows, UNWRITTEN);

            nn::gemv_q4_mlx_blocked(
                rt,
                Q4MlxBank {
                    packed: &pb,
                    scales_biases: &sbb,
                },
                &xb,
                &yb,
                shape(rows, cols, group),
            )
            .unwrap();
            rt.synchronize().unwrap();
            let got = yb.read_f32();
            assert_all_written("gemv_q4_mlx_blocked", &got[..rows]);
            close_rel("gemv_q4_mlx_blocked", &got[..rows], &want, 3e-3);
        }
    });
}

/// The fused K‖V GEMV against two separate ones over the same banks.
///
/// It partitions a single grid across two matrices, so an off-by-one in the
/// split writes one output correctly and the other from the wrong rows.
#[test]
fn q4_mlx_kv_matches_two_separate_gemvs() {
    with_gpu(|rt| {
        let (rows, cols, group) = (256usize, 256usize, 32usize);
        let (pk, sk, dk) = q4_mlx_matrix(rows, cols, group);
        // A different bank for V, or the test cannot tell the two apart.
        let (pv, sv, dv) = q4_mlx_matrix(rows + 8, cols, group);
        let dv = dv[..rows * cols].to_vec();
        let x = random_f32(cols, 0xF303);
        let xr = round_trip_bf16(&x);

        let pkb = rt.alloc_buffer(pk.len()).unwrap();
        pkb.write_bytes(&pk);
        let pvb = rt.alloc_buffer(pv.len()).unwrap();
        pvb.write_bytes(&pv);
        let skb = buf_bf16(rt, &sk);
        let svb = buf_bf16(rt, &sv);
        let xb = buf_bf16(rt, &x);
        let kb = seeded(rt, rows, UNWRITTEN);
        let vb = seeded(rt, rows, UNWRITTEN);

        nn::gemv_q4_mlx_kv(
            rt,
            Q4MlxBank {
                packed: &pkb,
                scales_biases: &skb,
            },
            Q4MlxBank {
                packed: &pvb,
                scales_biases: &svb,
            },
            &xb,
            &kb,
            &vb,
            shape(rows, cols, group),
            Q4MlxLayout::RowMajor,
        )
        .unwrap();
        rt.synchronize().unwrap();

        let (gk, gv) = (kb.read_f32(), vb.read_f32());
        assert_all_written("gemv_q4_mlx_kv k", &gk[..rows]);
        assert_all_written("gemv_q4_mlx_kv v", &gv[..rows]);
        close_rel(
            "gemv_q4_mlx_kv k",
            &gk[..rows],
            &dense_gemv(&dk, &xr, rows, cols),
            3e-3,
        );
        close_rel(
            "gemv_q4_mlx_kv v",
            &gv[..rows],
            &dense_gemv(&dv, &xr, rows, cols),
            3e-3,
        );
    });
}

/// `gemm_q4_mlx` is the M>1 form of the simd GEMV. Row `i` of its output must
/// equal the GEMV of row `i` of the activation, which is a reference the suite
/// already trusts.
#[test]
fn gemm_q4_mlx_agrees_with_the_gemv_on_every_row() {
    with_gpu(|rt| {
        let (rows, cols, group, m) = (256usize, 256usize, 32usize, 5usize);
        let (packed, sb, dense) = q4_mlx_matrix(rows, cols, group);
        let xm = random_f32(m * cols, 0xF404);
        let xr = round_trip_bf16(&xm);

        let pb = rt.alloc_buffer(packed.len()).unwrap();
        pb.write_bytes(&packed);
        let sbb = buf_bf16(rt, &sb);
        let xb = buf_bf16(rt, &xm);
        let yb = seeded(rt, m * rows, UNWRITTEN);
        let bank = Q4MlxBank {
            packed: &pb,
            scales_biases: &sbb,
        };

        nn::gemm_q4_mlx(
            rt,
            bank,
            &xb,
            &yb,
            shape(rows, cols, group),
            m as u32,
            Q4MlxLayout::RowMajor,
            None,
        )
        .unwrap();
        rt.synchronize().unwrap();

        let got = yb.read_f32();
        assert_all_written("gemm_q4_mlx", &got[..m * rows]);
        for i in 0..m {
            let want = dense_gemv(&dense, &xr[i * cols..(i + 1) * cols], rows, cols);
            close_rel(
                &format!("gemm_q4_mlx row {i}"),
                &got[i * rows..(i + 1) * rows],
                &want,
                3e-3,
            );
        }
    });
}

// ------------------------------------------------------------ MLP bf16 ---

/// The bf16 gating kernel against its f32 sibling, which is tested.
#[test]
fn mlp_gelu_tanh_bf16_matches_the_f32_kernel_within_bf16_resolution() {
    with_gpu(|rt| {
        // Spans the range where a fast tanh would NaN, as the f32 test does.
        let gate: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.05).collect();
        let up: Vec<f32> = (0..1024).map(|i| ((i % 17) as f32 - 8.0) * 0.25).collect();
        let n = gate.len();

        let gb = buf(rt, &gate);
        let ub = buf(rt, &up);
        let f32_out = empty(rt, n);
        nn::mlp_gelu_tanh(rt, &gb, &ub, &f32_out, n as u32).unwrap();

        let bf_out = rt.alloc_buffer(n * 2).unwrap();
        bf_out.zero();
        nn::mlp_gelu_tanh_bf16(rt, &gb, &ub, &bf_out, n as u32).unwrap();
        rt.synchronize().unwrap();

        let want = f32_out.read_f32();
        let got: Vec<f32> = bf_out.read_u32()[..n / 2]
            .iter()
            .flat_map(|p| {
                [
                    tessl::tensor::bf16_bits_to_f32((*p & 0xffff) as u16),
                    tessl::tensor::bf16_bits_to_f32((*p >> 16) as u16),
                ]
            })
            .collect();
        assert!(
            got.iter().all(|v| v.is_finite()),
            "mlp_gelu_tanh_bf16 produced a non-finite value"
        );
        // bf16 keeps 8 significand bits: ~2^-8 relative.
        close_rel("mlp_gelu_tanh_bf16", &got, &want[..n], 1e-2);
    });
}

// ------------------------------------------------------------- KV cache ---

/// The paired store must write both halves at the device-side offset, and
/// nothing else. A single-buffer version of this exists; the pair does not.
#[test]
fn kv_store_timestep_pair_writes_both_halves_and_nothing_else() {
    with_gpu(|rt| {
        let n = 96usize;
        let k = random_f32(n, 0xF501);
        let v = random_f32(n, 0xF502);
        let kb = buf(rt, &k);
        let vb = buf(rt, &v);
        let dk = seeded(rt, 5 * n, UNWRITTEN);
        let dv = seeded(rt, 5 * n, UNWRITTEN);
        let off = rt.alloc_buffer(4).unwrap();
        off.write_u32(&[(3 * n) as u32]);

        nn::kv_store_timestep_pair(rt, &kb, &vb, &dk, &dv, &off, n as u32).unwrap();
        rt.synchronize().unwrap();

        for (name, src, dst) in [("k", &k, dk.read_f32()), ("v", &v, dv.read_f32())] {
            close_rel(
                &format!("kv pair {name} slot"),
                &dst[3 * n..4 * n],
                src,
                0.0,
            );
            let outside = dst[..3 * n]
                .iter()
                .chain(&dst[4 * n..5 * n])
                .filter(|p| **p != UNWRITTEN)
                .count();
            assert_eq!(
                outside, 0,
                "kv pair {name}: wrote {outside} slots it does not own"
            );
        }
    });
}

/// `kv_ring_densify` rotates a ring buffer into chronological order. Only its
/// zero-capacity rejection was tested; the rotation itself was not.
#[test]
fn kv_ring_densify_puts_the_ring_in_chronological_order() {
    with_gpu(|rt| {
        let (n_slot, capacity) = (4usize, 6usize);
        // Slot values are distinguishable so a wrong rotation is visible.
        let src: Vec<f32> = (0..capacity * n_slot).map(|i| i as f32).collect();
        for (filled, start) in [(6usize, 2usize), (6, 0), (3, 0)] {
            let sb = buf(rt, &src);
            let db = seeded(rt, capacity * n_slot, UNWRITTEN);
            let fb = rt.alloc_buffer(4).unwrap();
            fb.write_u32(&[filled as u32]);
            let stb = rt.alloc_buffer(4).unwrap();
            stb.write_u32(&[start as u32]);

            nn::kv_ring_densify(rt, &sb, &db, &fb, &stb, n_slot as u32, capacity as u32).unwrap();
            rt.synchronize().unwrap();

            let got = db.read_f32();
            for i in 0..filled {
                let from = (start + i) % capacity;
                for j in 0..n_slot {
                    assert_eq!(
                        got[i * n_slot + j],
                        src[from * n_slot + j],
                        "ring densify filled={filled} start={start}: dst slot {i} elem {j}"
                    );
                }
            }
        }
    });
}

// -------------------------------------------------------- Embedding MLX ---

/// The MLX embedding gather against dense rows, including an out-of-range id.
#[test]
fn embed_lookup_q4_mlx_gathers_rows_and_zeroes_out_of_range_tokens() {
    with_gpu(|rt| {
        let (vocab, hidden, group) = (64usize, 128usize, 32usize);
        let (packed, sb, dense) = q4_mlx_matrix(vocab, hidden, group);
        let ids: Vec<u32> = vec![0, 7, 63, 64, 1000];
        let n_tokens = ids.len();

        let pb = rt.alloc_buffer(packed.len()).unwrap();
        pb.write_bytes(&packed);
        let sbb = buf_bf16(rt, &sb);
        let idb = rt.alloc_buffer(n_tokens * 4).unwrap();
        idb.write_u32(&ids);
        let ob = seeded(rt, n_tokens * hidden, UNWRITTEN);

        nn::embed_lookup_q4_mlx(
            rt,
            Q4MlxBank {
                packed: &pb,
                scales_biases: &sbb,
            },
            &idb,
            &ob,
            vocab as u32,
            hidden as u32,
            group as u32,
            n_tokens as u32,
        )
        .unwrap();
        rt.synchronize().unwrap();

        let got = ob.read_f32();
        assert_all_written("embed_lookup_q4_mlx", &got[..n_tokens * hidden]);
        for (t, id) in ids.iter().enumerate() {
            let row = &got[t * hidden..(t + 1) * hidden];
            if (*id as usize) < vocab {
                let want = &dense[*id as usize * hidden..(*id as usize + 1) * hidden];
                close_rel(&format!("embed row {t}"), row, want, 3e-3);
            } else {
                // Out of range must be zeroed, not left as whatever was there:
                // a gather that silently reads past the table is a disclosure
                // bug as much as a numeric one.
                assert!(
                    row.iter().all(|v| *v == 0.0),
                    "embed token {id} is out of range and must gather zeros"
                );
            }
        }
    });
}
