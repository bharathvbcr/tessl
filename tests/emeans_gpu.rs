mod common;

use std::sync::Arc;

use ojas_core::{Budget, OjasError};
use tessl::emeans::{
    assign_gpu_bf16, assign_gpu_f32, revival_gumbel_key, EmeansGpuBackend, EmeansGpuConfig, EmeansGpuPrecision,
    EmeansGpuRunner, EpochBatchPlan, MAX_GPU_CHUNK_ROWS, NONFINITE_ERR_PREFIX,
};
use tessl::pool_ledger::OjasBudgetLedger;
use tessl::tensor::{bf16_bits_to_f32, f32_slice_to_bf16, f32_to_bf16_bits};
use tessl::GpuRuntime;

fn prescale_normalize_row(row: &[f32]) -> Vec<f32> {
    let max_abs = row.iter().fold(0.0f64, |acc, &v| acc.max((v as f64).abs()));
    if max_abs == 0.0 {
        return vec![0.0f32; row.len()];
    }
    let mut norm2 = 0.0f64;
    for &v in row {
        let w = (v as f64) / max_abs;
        norm2 += w * w;
    }
    let inv = 1.0 / norm2.sqrt();
    row.iter().map(|&v| (((v as f64) / max_abs) * inv) as f32).collect()
}

fn effective_vec(row: &[f32], precision: EmeansGpuPrecision) -> Vec<f32> {
    match precision {
        EmeansGpuPrecision::Bf16 | EmeansGpuPrecision::Bf16AssignF32Stats => common::round_trip_bf16(row),
        EmeansGpuPrecision::F32 => row.to_vec(),
    }
}

fn cpu_assign_oracle(
    rows: &[f32],
    n: usize,
    d: usize,
    centroids: &[f32],
    k: usize,
    precision: EmeansGpuPrecision,
    spherical: bool,
) -> (Vec<u32>, f64, Vec<f64>) {
    let mut c_eff = Vec::with_capacity(k * d);
    for j in 0..k {
        let base = &centroids[j * d..(j + 1) * d];
        let unit = if spherical {
            prescale_normalize_row(base)
        } else {
            base.to_vec()
        };
        c_eff.extend_from_slice(&effective_vec(&unit, precision));
    }

    let mut labels = vec![0u32; n];
    let mut total_obj = 0.0f64;
    let mut gap_to_runner_up = vec![0.0f64; n];

    for i in 0..n {
        let x_eff = effective_vec(&rows[i * d..(i + 1) * d], precision);
        let mut best_j = 0usize;
        let mut best_dist = f64::INFINITY;
        let mut second_dist = f64::INFINITY;
        for j in 0..k {
            let c_row = &c_eff[j * d..(j + 1) * d];
            let dist = if spherical {
                let dot: f64 = x_eff
                    .iter()
                    .zip(c_row.iter())
                    .map(|(&a, &b)| (a as f64) * (b as f64))
                    .sum();
                2.0 - 2.0 * dot
            } else {
                x_eff
                    .iter()
                    .zip(c_row.iter())
                    .map(|(&a, &b)| {
                        let diff = (a as f64) - (b as f64);
                        diff * diff
                    })
                    .sum()
            };
            if dist < best_dist {
                second_dist = best_dist;
                best_dist = dist;
                best_j = j;
            } else if dist < second_dist {
                second_dist = dist;
            }
        }
        labels[i] = best_j as u32;
        total_obj += best_dist;
        gap_to_runner_up[i] = second_dist - best_dist;
    }

    (labels, total_obj, gap_to_runner_up)
}

#[test]
fn test_assign_arms_match_oracle_and_overwrite_sentinels() {
    common::with_gpu(|rt| {
        const { assert!(MAX_GPU_CHUNK_ROWS == 65_536, "MAX_GPU_CHUNK_ROWS must stay 65_536") };

        let precisions = [
            EmeansGpuPrecision::Bf16,
            EmeansGpuPrecision::Bf16AssignF32Stats,
            EmeansGpuPrecision::F32,
        ];
        let mut backends = vec![EmeansGpuBackend::Auto, EmeansGpuBackend::Simdgroup];
        if rt.has_tensorops() {
            backends.push(EmeansGpuBackend::TensorOps);
        }

        // 1. Non-aligned prime shape exercising split_c > 1 (n=131, k=1031, d=67)
        //    plus 64x64 cooperative multi-tile shape (n=193, k=130, d=128).
        for &(n, k, d, chunk_rows) in &[(131usize, 1031usize, 67usize, 50u32), (193, 130, 128, 128)] {
            let rows = common::random_f32(n * d, 0x1234_5678 ^ (n as u64));
            let centroids = common::random_f32(k * d, 0x9abc_def0 ^ (k as u64));
            let rows_bf16 = f32_slice_to_bf16(&rows);
            let rows_bf16_as_f32: Vec<f32> = rows_bf16.iter().copied().map(bf16_bits_to_f32).collect();

            for &precision in &precisions {
                let (want_labels, want_obj, gaps) = cpu_assign_oracle(&rows, n, d, &centroids, k, precision, false);
                let (want_bf16_labels, want_bf16_obj, bf16_gaps) =
                    cpu_assign_oracle(&rows_bf16_as_f32, n, d, &centroids, k, precision, false);

                for &backend in &backends {
                    let (got_labels, got_obj) = assign_gpu_f32(
                        rt, &rows, n as u32, d as u32, &centroids, k as u32, chunk_rows, precision, backend, false,
                    )
                    .expect("assign_gpu_f32");

                    assert_eq!(got_labels.len(), n);
                    for i in 0..n {
                        assert!(
                            (got_labels[i] as usize) < k,
                            "label out of range at row {i}: {}",
                            got_labels[i]
                        );
                        if gaps[i] > 1e-4 {
                            assert_eq!(
                                got_labels[i], want_labels[i],
                                "label mismatch at row {i} ({precision:?}, {backend:?}, n={n}, k={k}, d={d})"
                            );
                        }
                    }
                    let rel_err = (got_obj - want_obj).abs() / want_obj.max(1.0);
                    assert!(
                        rel_err < 1e-5,
                        "obj mismatch ({precision:?}, {backend:?}): got {got_obj}, want {want_obj} (rel {rel_err})"
                    );

                    // bf16 input entry point
                    let (got_b_labels, got_b_obj) = assign_gpu_bf16(
                        rt, &rows_bf16, n as u32, d as u32, &centroids, k as u32, chunk_rows, precision, backend, false,
                    )
                    .expect("assign_gpu_bf16");
                    for i in 0..n {
                        if bf16_gaps[i] > 1e-4 {
                            assert_eq!(got_b_labels[i], want_bf16_labels[i]);
                        }
                    }
                    let b_rel = (got_b_obj - want_bf16_obj).abs() / want_bf16_obj.max(1.0);
                    assert!(
                        b_rel < 1e-5,
                        "bf16 obj mismatch ({precision:?}, {backend:?}): got {got_b_obj}, want {want_bf16_obj}"
                    );

                    // Sentinel overwrite check on resident + streaming paths
                    let cfg = EmeansGpuConfig {
                        k: k as u32,
                        d: d as u32,
                        max_chunk_rows: chunk_rows,
                        precision,
                        backend,
                        spherical: false,
                        unused_threshold: 0.0,
                        split_epsilon: 1e-4,
                        init_usage: 1.0,
                        seed: 42,
                    };
                    let mut runner = EmeansGpuRunner::new(rt, cfg, &centroids).expect("EmeansGpuRunner::new");
                    let ds = runner.upload_dataset_f32(&rows, n as u32).expect("upload_dataset_f32");
                    let order: Vec<u32> = (0..n as u32).collect();
                    let batches = [EpochBatchPlan {
                        row_offset: 0,
                        row_count: n as u32,
                        alpha: 0.25,
                        step: 0,
                        check_revive: false,
                    }];
                    let mut resident_labels = vec![0xdead_beefu32; n];
                    let (ep_obj, revivals, counts) = runner
                        .run_resident_epoch(&ds, &order, &batches, true, Some(&mut resident_labels))
                        .expect("run_resident_epoch");
                    assert_eq!(revivals, 0);
                    assert_eq!(counts.iter().sum::<u64>(), n as u64);
                    assert!(
                        resident_labels.iter().all(|&l| l != 0xdead_beef && (l as usize) < k),
                        "sentinel survived in resident_labels"
                    );
                    assert_eq!(resident_labels, got_labels);
                    assert!((ep_obj - got_obj).abs() <= 1e-5 * got_obj.max(1.0));
                }
            }
        }

        // 2. Self-distance == 0.0 AND strict `<` ascending cluster-ID tie-breaking
        //    across 64-cluster tile and split_c boundaries!
        let n_self = 16usize;
        let k_tie = 1031usize;
        let d_tie = 67usize;
        let x_self = common::random_f32(n_self * d_tie, 0x7777);
        let mut c_tie = common::random_f32(k_tie * d_tie, 0x8888);
        // Scale background centroids far away so exact copies clearly win.
        for v in &mut c_tie {
            *v += 10.0;
        }
        // Place exact copies of row i at BOTH cluster (i + 3) and cluster (i + 600)
        // (spanning across 64-wide tiles and split_c partitions). Strict `<` tie-breaking
        // MUST pick (i + 3), and self-distance MUST be exact 0.0!
        for i in 0..n_self {
            let row = &x_self[i * d_tie..(i + 1) * d_tie];
            let lo_j = i + 3;
            let hi_j = i + 600;
            c_tie[lo_j * d_tie..(lo_j + 1) * d_tie].copy_from_slice(row);
            c_tie[hi_j * d_tie..(hi_j + 1) * d_tie].copy_from_slice(row);
        }

        for &precision in &precisions {
            for &backend in &backends {
                let (labels, obj) = assign_gpu_f32(
                    rt,
                    &x_self,
                    n_self as u32,
                    d_tie as u32,
                    &c_tie,
                    k_tie as u32,
                    64,
                    precision,
                    backend,
                    false,
                )
                .expect("self-distance & tie-break assign");
                for (i, &lbl) in labels.iter().enumerate().take(n_self) {
                    assert_eq!(
                        lbl,
                        (i + 3) as u32,
                        "strict `<` tie-break failed for row {i} ({precision:?}, {backend:?})"
                    );
                }
                assert_eq!(
                    obj, 0.0,
                    "self-distance must be exact 0.0 ({precision:?}, {backend:?}), got {obj}"
                );
            }
        }
    });
}

#[test]
fn test_radix_csr_gather_zipf_skew_is_bit_identical_to_ascending_sum() {
    common::with_gpu(|rt| {
        let m = 4096usize;
        let k = 64usize;
        let d = 96usize;

        // Build centroids well separated along col 0: cluster 0 sits near 0.0,
        // clusters 1..64 sit near +50.0 * j so > 3,500 rows deterministically map to cluster 0.
        let mut init_centroids = vec![0.0f32; k * d];
        for j in 1..k {
            init_centroids[j * d] = (j as f32) * 50.0;
        }

        // Case A: Arbitrary random f32 values where 3,650 rows (> 14 slices of 256)
        // land in cluster 0 and the remaining 446 rows land across clusters 1..64.
        let mut rows_rand = common::random_f32(m * d, 0x51f0_cafe);
        for r in 0..m {
            if r < 3650 {
                // Near origin -> cluster 0
                rows_rand[r * d] *= 0.25;
            } else {
                let target_j = 1 + (r % (k - 1));
                rows_rand[r * d] = (target_j as f32) * 50.0 + rows_rand[r * d] * 0.1;
            }
        }

        for precision in [
            EmeansGpuPrecision::F32,
            EmeansGpuPrecision::Bf16AssignF32Stats,
            EmeansGpuPrecision::Bf16,
        ] {
            let cfg = EmeansGpuConfig {
                k: k as u32,
                d: d as u32,
                max_chunk_rows: m as u32,
                precision,
                backend: EmeansGpuBackend::Auto,
                spherical: false,
                unused_threshold: 0.0,
                split_epsilon: 1e-4,
                init_usage: 1.0 / 65536.0,
                seed: 99,
            };

            // Repeat 1: resident epoch
            let mut runner_a = EmeansGpuRunner::new(rt, cfg.clone(), &init_centroids).expect("runner_a");
            let ds_a = runner_a
                .upload_dataset_f32(&rows_rand, m as u32)
                .expect("upload_dataset_f32");
            let order: Vec<u32> = (0..m as u32).collect();
            let batches = [EpochBatchPlan {
                row_offset: 0,
                row_count: m as u32,
                alpha: 1.0,
                step: 0,
                check_revive: false,
            }];
            let mut labels_a = vec![u32::MAX; m];
            let (obj_a, rev_a, counts_a) = runner_a
                .run_resident_epoch(&ds_a, &order, &batches, true, Some(&mut labels_a))
                .expect("resident_epoch a");
            assert_eq!(rev_a, 0);
            assert!(
                counts_a[0] > 3500,
                "expected > 3,500 rows in cluster 0, got {}",
                counts_a[0]
            );
            let cents_a = runner_a.read_centroids().expect("read_centroids a");
            let usage_a = runner_a.read_usage().expect("read_usage a");
            let closing_a = runner_a
                .finalize_closing_centroids()
                .expect("finalize_closing_centroids a");

            // Repeat 2: resident epoch via same-runtime GpuBuffer upload
            let rows_buf = common::buf(rt, &rows_rand);
            let mut runner_b = EmeansGpuRunner::new(rt, cfg.clone(), &init_centroids).expect("runner_b");
            let ds_b = runner_b
                .upload_dataset_f32_buffer(&rows_buf, m as u32)
                .expect("upload_dataset_f32_buffer");
            let mut labels_b = vec![u32::MAX; m];
            let (obj_b, _, counts_b) = runner_b
                .run_resident_epoch(&ds_b, &order, &batches, true, Some(&mut labels_b))
                .expect("resident_epoch b");
            let cents_b = runner_b.read_centroids().expect("read_centroids b");
            let closing_b = runner_b
                .finalize_closing_centroids()
                .expect("finalize_closing_centroids b");

            assert_eq!(labels_a, labels_b);
            assert_eq!(counts_a, counts_b);
            assert_eq!(obj_a.to_bits(), obj_b.to_bits(), "repeat objective bit-identity");
            for idx in 0..(k * d) {
                assert_eq!(
                    cents_a[idx].to_bits(),
                    cents_b[idx].to_bits(),
                    "repeat centroid bit mismatch at {idx}"
                );
                assert_eq!(
                    closing_a[idx].to_bits(),
                    closing_b[idx].to_bits(),
                    "repeat closing bit mismatch at {idx}"
                );
            }

            // Streaming path with deferred sync (labels_out = None) must be bit-identical
            let mut runner_stream = EmeansGpuRunner::new(rt, cfg.clone(), &init_centroids).expect("runner_stream");
            runner_stream
                .reset_pass_accumulators(true)
                .expect("reset_pass_accumulators");
            runner_stream
                .accumulate_f32_chunk(&rows_rand, m as u32, None)
                .expect("accumulate_f32_chunk");
            let rev_s = runner_stream
                .apply_batch_update(1.0, 0, false)
                .expect("apply_batch_update");
            assert_eq!(rev_s, 0);
            let (obj_s, counts_s) = runner_stream.finish_streaming_epoch().expect("finish_streaming_epoch");
            let cents_s = runner_stream.read_centroids().expect("read_centroids stream");
            let usage_s = runner_stream.read_usage().expect("read_usage stream");
            let closing_s = runner_stream
                .finalize_closing_centroids()
                .expect("finalize_closing_centroids stream");

            assert_eq!(counts_a, counts_s);
            assert_eq!(obj_a.to_bits(), obj_s.to_bits(), "resident vs stream obj bits");
            for j in 0..k {
                assert_eq!(usage_a[j].to_bits(), usage_s[j].to_bits());
            }
            for idx in 0..(k * d) {
                assert_eq!(
                    cents_a[idx].to_bits(),
                    cents_s[idx].to_bits(),
                    "resident vs stream centroid bit mismatch at {idx} ({precision:?})"
                );
                assert_eq!(
                    closing_a[idx].to_bits(),
                    closing_s[idx].to_bits(),
                    "resident vs stream closing bit mismatch at {idx} ({precision:?})"
                );
            }

            // Sequential ascending-row 256-slice f32 reference (`weight == 1.0` exact override)
            let stats_rows = match precision {
                EmeansGpuPrecision::Bf16 => common::round_trip_bf16(&rows_rand),
                EmeansGpuPrecision::Bf16AssignF32Stats | EmeansGpuPrecision::F32 => rows_rand.clone(),
            };
            let mut bucket_rows = vec![Vec::<usize>::new(); k];
            for (r, &lbl) in labels_a.iter().enumerate() {
                bucket_rows[lbl as usize].push(r);
            }
            for j in 0..k {
                let rlist = &bucket_rows[j];
                if rlist.is_empty() {
                    for col in 0..d {
                        assert_eq!(cents_a[j * d + col].to_bits(), init_centroids[j * d + col].to_bits());
                    }
                    assert_eq!(usage_a[j].to_bits(), 0.0f32.to_bits());
                } else {
                    let count_f32 = rlist.len() as f32;
                    assert_eq!(usage_a[j].to_bits(), count_f32.to_bits());
                    for col in 0..d {
                        let mut total = 0.0f32;
                        for slice in rlist.chunks(256) {
                            let mut s_acc = 0.0f32;
                            for &r in slice {
                                s_acc += stats_rows[r * d + col];
                            }
                            total += s_acc;
                        }
                        let want_mean = total / count_f32;
                        assert_eq!(
                            cents_a[j * d + col].to_bits(),
                            want_mean.to_bits(),
                            "segmented ascending sum bit mismatch at cluster {j} col {col} ({precision:?})"
                        );
                    }
                }
            }
        }

        // Case B: Exact-mantissa Zipf rows (> 3,500 rows in cluster 0) compared against
        // flat single-loop `0..M` sequential ascending f32 accumulation (`to_bits()`).
        let mut rng = common::SplitMix::new(0x2222_3333);
        let mut rows_dyadic = vec![0.0f32; m * d];
        for r in 0..m {
            for col in 0..d {
                let step_int = (rng.range(0, 256) as i32) - 128;
                rows_dyadic[r * d + col] = (step_int as f32) * 0.0625;
            }
            if r >= 3650 {
                let target_j = 1 + (r % (k - 1));
                rows_dyadic[r * d] = (target_j as f32) * 50.0;
            }
        }
        let cfg_f32 = EmeansGpuConfig {
            k: k as u32,
            d: d as u32,
            max_chunk_rows: m as u32,
            precision: EmeansGpuPrecision::F32,
            backend: EmeansGpuBackend::Auto,
            spherical: false,
            unused_threshold: 0.0,
            split_epsilon: 1e-4,
            init_usage: 1.0,
            seed: 7,
        };
        let mut runner_flat = EmeansGpuRunner::new(rt, cfg_f32, &init_centroids).expect("runner_flat");
        runner_flat.reset_pass_accumulators(false).expect("reset flat");
        let mut labels_flat = vec![0u32; m];
        runner_flat
            .accumulate_f32_chunk(&rows_dyadic, m as u32, Some(&mut labels_flat))
            .expect("accum flat");
        runner_flat.apply_batch_update(1.0, 0, false).expect("update flat");
        let cents_flat = runner_flat.read_centroids().expect("read flat");
        let mut flat_sums = vec![0.0f32; k * d];
        let mut flat_counts = vec![0u64; k];
        for r in 0..m {
            let j = labels_flat[r] as usize;
            flat_counts[j] += 1;
            for col in 0..d {
                flat_sums[j * d + col] += rows_dyadic[r * d + col];
            }
        }
        assert!(flat_counts[0] > 3500);
        for j in 0..k {
            if flat_counts[j] > 0 {
                let inv_n = flat_counts[j] as f32;
                for col in 0..d {
                    let want = flat_sums[j * d + col] / inv_n;
                    assert_eq!(
                        cents_flat[j * d + col].to_bits(),
                        want.to_bits(),
                        "flat sequential f32 bit mismatch at ({j}, {col})"
                    );
                }
            }
        }

        // Also verify bf16 streaming + resident buffer parity
        let bf16_bits = f32_slice_to_bf16(&rows_dyadic);
        let bf16_buf = rt.alloc_buffer(bf16_bits.len() * 2).expect("alloc bf16");
        bf16_buf.write_bf16_bits(&bf16_bits);
        let cfg_bf16 = EmeansGpuConfig {
            k: k as u32,
            d: d as u32,
            max_chunk_rows: 1024,
            precision: EmeansGpuPrecision::Bf16,
            backend: EmeansGpuBackend::Auto,
            spherical: false,
            unused_threshold: 0.0,
            split_epsilon: 1e-4,
            init_usage: 1.0,
            seed: 7,
        };
        let mut r_res = EmeansGpuRunner::new(rt, cfg_bf16.clone(), &init_centroids).unwrap();
        let ds_bf16 = r_res
            .upload_dataset_bf16_buffer(&bf16_buf, m as u32)
            .expect("upload_dataset_bf16_buffer");
        let order_all: Vec<u32> = (0..m as u32).collect();
        let plan = [EpochBatchPlan {
            row_offset: 0,
            row_count: m as u32,
            alpha: 0.5,
            step: 0,
            check_revive: false,
        }];
        let (o1, _, c1) = r_res
            .run_resident_epoch(&ds_bf16, &order_all, &plan, true, None)
            .unwrap();
        let mut r_str = EmeansGpuRunner::new(rt, cfg_bf16, &init_centroids).unwrap();
        r_str.reset_pass_accumulators(true).unwrap();
        for ch in bf16_bits.chunks(1024 * d) {
            r_str.accumulate_bf16_chunk(ch, (ch.len() / d) as u32, None).unwrap();
        }
        r_str.apply_batch_update(0.5, 0, false).unwrap();
        let (o2, c2) = r_str.finish_streaming_epoch().unwrap();
        assert_eq!(c1, c2);
        assert_eq!(o1.to_bits(), o2.to_bits());
        assert_eq!(r_res.read_centroids().unwrap(), r_str.read_centroids().unwrap());
    });
}

#[test]
fn test_revive_splits_separates_bf16_neg_zero_and_ftz_donors() {
    common::with_gpu(|rt| {
        let d = 8usize;
        let k = 2usize;

        // Build unit-norm rows with |x_0| ~ 0.1 on opposite sides of col 0 so any
        // non-FTZ separation along col 0 strictly splits the dataset on the next step!
        let tail_val = ((1.0f64 - 0.01f64) / 7.0f64).sqrt() as f32;
        let mut rows_zero = Vec::with_capacity(64 * d);
        for i in 0..64 {
            let x0 = if i < 32 { -0.1f32 } else { 0.1f32 };
            rows_zero.push(x0);
            for _ in 1..d {
                rows_zero.push(tail_val);
            }
        }

        // 1. Test [-0.0f32; 8] and [0.0f32; 8] across Bf16 and F32:
        //    Cluster 0 starts at donor_val (receives all 64 rows when cluster 1 is far away
        //    at +100.0 so cluster 1 has 0 rows and usage < thresh, triggering revival!).
        //    Wait: use alpha = 0.0001? No: if cluster 0 starts at [-0.0; 8] and we want
        //    cluster 0's coordinates at revival time to be EXACTLY [-0.0; 8] or [0.0; 8],
        //    set rows to exact pairs (+v, -v) summing to 0.0 OR pass 0 rows into cluster 1
        //    while cluster 0 retains [-0.0; 8] via init_usage > 0 and alpha tiny, OR
        //    have balanced rows summing to exact 0.0 with alpha = 1.0!
        //    Wait: for exact [-0.0f32; 8] donor preservation: if init_centroids[0] = [-0.0; 8],
        //    init_usage = 10.0, and batch has 1 row at [+1000.0; 8] landing on a 3rd cluster
        //    (k = 3: cluster 0 at [-0.0; 8] has N_0 = 10*(1-alpha), cluster 1 at [500; 8]
        //    Wait: both 0 and 1 would have identical N!).
        //    Even better: k = 3 where cluster 2 gets 100 rows at [50.0; 8] (so mean_N is ~33),
        //    Wait — then cluster 2 is the donor! If cluster 0 is the donor with [-0.0; 8]:
        //    give cluster 0 balanced rows `[+x, -x]`? In IEEE 754 `(+x) + (-x) = +0.0`.
        //    How does cluster 0 keep EXACT `[-0.0f32; 8]` at `emeans_revive_splits`?
        //    When `n_0 == 0` in the batch (`c_new = c_old = [-0.0f32; 8]`!), while `init_usage = 100.0`
        //    and `alpha = 0.5`:
        //      - Suppose k = 2: cluster 0 at [-0.0; 8], cluster 1 at [100.0; 8], init_usage = 100.0?
        //        No, both start with init_usage. What if batch 0 gives cluster 0 rows summing to 0?
        //        Wait: in IEEE-754, `(-0.0f32) + (-0.0f32) == -0.0f32`, and `(-0.0f32) / 1.0f32 == -0.0f32`!
        //        YES!! A batch of `[-0.0f32; 8]` rows assigned to cluster 0 with `alpha = 1.0`
        //        computes `s_0 = -0.0f32` (or `+0.0` if seeded at `+0.0`, wait: `0.0 + (-0.0) = +0.0`,
        //        whereas with `w < 1.0` or `n_0 == 0` after a prior batch:
        //        Batch 0 (check_revive=false, alpha=0.5): 10 rows of `[-0.0f32; 8]` hit cluster 0 ->
        //        N_0 = 0.5*init + 5.0, N_1 = 0.5*init. Wait: `mean = +0.0 / 10 = +0.0`, `c_old = -0.0`,
        //        `-0.0 + w * (+0.0 - (-0.0)) = +0.0`.
        //        Whereas if `init_centroids[0] = [-0.0f32; 8]` with `init_usage = 10.0`, and Batch 0
        //        sends 1 row to cluster 2 (`k = 3`, `alpha = 1.0` -> `N_0 = 0`... wait: with `alpha = 0.01`,
        //        if cluster 1 had `N_1 = 0`? All clusters share `init_usage`!).
        //        Wait: how can cluster 0 have `c_0 = [-0.0f32; 8]` AND `N_0 > thresh`?
        //        Send rows of `[-1.0e-40f32; 8]` or `[-0.0f32; 8]`? Wait: `(-1.0) + (-1.0)` is `-2.0`;
        //        if `alpha = 1.0` and rows are `[-0.0f32; 8]`, `0.0f + (-0.0f) = +0.0f`, UNLESS in batch 1
        //        cluster 0 gets rows `[ -1.0; 8 ]` and `[ +1.0; 8 ]`? Still `+0.0f`!
        //        Wait: how CAN `-0.0f32` arise in `c_0`? `(-0.0f32)` from `-1.0 * 0.0 = -0.0f32` or
        //        when `c_old = [-1.0; 8]` and `mean = [0.0; 8]` with `w = 1.0`? `w == 1.0` sets `c_new = mean`:
        //        if `s_j = -0.0f` (e.g. if `mean` is negative tiny subnormal that flushes to `-0.0f` on Apple GPU!!)!
        //        YES!! On Apple GPU FTZ, `s_j = -1.0e-38f` (`-f32::MIN_POSITIVE`) divided by `n_j = 4.0f`
        //        flushes under FTZ directly to **`-0.0f32` (`0x8000_0000`)**!!
        let neg_ftz_row = [-f32::MIN_POSITIVE; 8];
        for precision in [EmeansGpuPrecision::Bf16, EmeansGpuPrecision::F32] {
            for case_rows in [
                vec![neg_ftz_row, [0.0f32; 8], [0.0f32; 8], [0.0f32; 8]],
                vec![[0.0f32; 8]; 4],
            ] {
                let flat_case: Vec<f32> = case_rows.into_iter().flatten().collect();
                let init_c = [
                    -0.0f32, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, // cluster 0
                    100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, // cluster 1 (unused)
                ];
                let cfg = EmeansGpuConfig {
                    k: k as u32,
                    d: d as u32,
                    max_chunk_rows: 64,
                    precision,
                    backend: EmeansGpuBackend::Auto,
                    spherical: false,
                    unused_threshold: 0.5,
                    split_epsilon: 1e-4,
                    init_usage: 1.0e-5,
                    seed: 123,
                };
                let mut runner = EmeansGpuRunner::new(rt, cfg, &init_c).expect("runner");
                runner.reset_pass_accumulators(false).unwrap();
                runner.accumulate_f32_chunk(&flat_case, 4, None).unwrap();
                let revived = runner.apply_batch_update(1.0, 4, true).expect("revive zero/neg-zero");
                assert_eq!(revived, 1, "dead cluster 1 must be revived from donor 0");

                let cents = runner.read_centroids().unwrap();
                let usage = runner.read_usage().unwrap();
                assert!(cents.iter().all(|v| v.is_finite()), "zero NaNs allowed: {cents:?}");
                assert_eq!(usage[0], 2.0);
                assert_eq!(usage[1], 2.0);
                assert_eq!(cents[0], 0.0);
                assert_eq!(
                    cents[d].to_bits(),
                    0x2080_0000u32,
                    "collapsed zero/-0.0 split must place child[0] at 2^-62 (0x2080_0000), got {:#x}",
                    cents[d].to_bits()
                );

                // Verify strict symmetry-breaking on the very next assignment step with unit-norm
                // rows having |x_0| = 0.1 (surviving Apple GPU FTZ on both Bf16 and F32!):
                runner.reset_pass_accumulators(false).unwrap();
                let mut next_labels = vec![u32::MAX; 64];
                runner
                    .accumulate_f32_chunk(&rows_zero, 64, Some(&mut next_labels))
                    .unwrap();
                let c0_cnt = next_labels.iter().filter(|&&l| l == 0).count();
                let c1_cnt = next_labels.iter().filter(|&&l| l == 1).count();
                assert_eq!(
                    (c0_cnt, c1_cnt),
                    (32, 32),
                    "2^-62 child[0] must survive FTZ and split 64 rows 32/32 ({precision:?})"
                );
            }
        }

        // 2. Test [1.0; 8] donor on Bf16 (where 1.0 * (1 +/- 1e-4) collapses to 1.0 in bf16!)
        //    AND verify revival_gumbel_key donor ranking parity across multiple donors!
        let k_multi = 6usize;
        let mut init_multi = vec![100.0f32; k_multi * d];
        for j in 0..4 {
            for col in 0..d {
                init_multi[j * d + col] = (j as f32) + 1.0; // 1.0, 2.0, 3.0, 4.0
            }
        }
        // Build batch: clusters 0..4 each get (j + 2) copies of their exact vector;
        // clusters 4 and 5 get 0 rows -> 2 unused clusters, 4 candidate donors!
        let mut multi_rows = Vec::new();
        for j in 0..4 {
            let reps = (j + 2) * 4;
            for _ in 0..reps {
                for _col in 0..d {
                    multi_rows.push((j as f32) + 1.0);
                }
            }
        }
        let m_multi = (multi_rows.len() / d) as u32;
        let seed = 0xfeed_face_cafe_babeu64;
        let step = 9u64;
        let cfg_bf16 = EmeansGpuConfig {
            k: k_multi as u32,
            d: d as u32,
            max_chunk_rows: 256,
            precision: EmeansGpuPrecision::Bf16,
            backend: EmeansGpuBackend::Auto,
            spherical: false,
            unused_threshold: 0.2,
            split_epsilon: 1e-4,
            init_usage: 1.0e-5,
            seed,
        };
        let mut runner_m = EmeansGpuRunner::new(rt, cfg_bf16, &init_multi).unwrap();
        runner_m.reset_pass_accumulators(false).unwrap();
        runner_m.accumulate_f32_chunk(&multi_rows, m_multi, None).unwrap();
        let revived_m = runner_m.apply_batch_update(1.0, step, true).unwrap();
        assert_eq!(revived_m, 2);

        // Compute expected top-2 donors via `revival_gumbel_key`
        let donor_usages = [8.0f32, 12.0, 16.0, 20.0];
        let mut keyed: Vec<(f32, u32)> = (0..4u32)
            .map(|j| (revival_gumbel_key(donor_usages[j as usize], seed, step, j), j))
            .collect();
        keyed.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let top0 = keyed[0].1 as usize;
        let top1 = keyed[1].1 as usize;

        let cents_m = runner_m.read_centroids().unwrap();
        let usage_m = runner_m.read_usage().unwrap();
        assert_eq!(usage_m[4], donor_usages[top0] * 0.5);
        assert_eq!(usage_m[5], donor_usages[top1] * 0.5);
        assert_eq!(usage_m[top0], donor_usages[top0] * 0.5);
        assert_eq!(usage_m[top1], donor_usages[top1] * 0.5);

        // Both split pairs had integer power/small coordinates where 1 +/- 1e-4 collapses in bf16!
        for &(donor_idx, child_idx) in &[(top0, 4usize), (top1, 5usize)] {
            let donor_q0 = f32_to_bf16_bits(cents_m[donor_idx * d]);
            let child_q0 = f32_to_bf16_bits(cents_m[child_idx * d]);
            assert_ne!(
                donor_q0, child_q0,
                "bf16 donor {donor_idx} and child {child_idx} must separate on col 0"
            );
            assert_eq!(child_q0, donor_q0 + 1, "positive bf16 donor steps +1 bf16 ULP");
        }
    });
}

#[test]
fn test_spherical_prescaling_and_zero_vectors_on_gpu() {
    common::with_gpu(|rt| {
        let d = 16usize;
        let k = 3usize;

        // Extreme centroids that would underflow (1e-25^2 = 1e-50 -> 0 in f32!) or overflow
        // (1e25^2 = 1e50 -> Inf in f32!) without max_abs prescaling, plus an exact zero centroid!
        let mut init_centroids = vec![0.0f32; k * d];
        for col in 0..d {
            init_centroids[col] = if col % 2 == 0 { 1.0e-25 } else { -1.0e-25 };
            init_centroids[d + col] = 1.0e25;
            init_centroids[2 * d + col] = 0.0;
        }

        // Dataset: 32 rows aligned with cluster 0, 32 rows aligned with cluster 1, and 2 exact zero rows.
        let mut rows = Vec::with_capacity(66 * d);
        for _ in 0..32 {
            for col in 0..d {
                rows.push(if col % 2 == 0 { 0.25 } else { -0.25 }); // norm = sqrt(16 * 1/16) = 1.0
            }
        }
        rows.extend(std::iter::repeat_n(0.25, 32 * d)); // norm = 1.0
        rows.extend(std::iter::repeat_n(0.0, 2 * d)); // zero input rows
        let n = 66u32;

        for precision in [
            EmeansGpuPrecision::Bf16,
            EmeansGpuPrecision::Bf16AssignF32Stats,
            EmeansGpuPrecision::F32,
        ] {
            let (labels, obj) = assign_gpu_f32(
                rt,
                &rows,
                n,
                d as u32,
                &init_centroids,
                k as u32,
                32,
                precision,
                EmeansGpuBackend::Auto,
                true,
            )
            .expect("spherical assign_gpu_f32");
            assert!(obj.is_finite(), "spherical obj must be finite, got {obj}");
            for &lbl in &labels[..32] {
                assert_eq!(lbl, 0, "1e-25 centroid must normalize to +/-0.25");
            }
            for &lbl in &labels[32..64] {
                assert_eq!(lbl, 1, "1e25 centroid must normalize to +0.25");
            }
            // Zero rows have dot = 0 with all 3 clusters -> score = 2.0 on all 3 -> tie-break picks cluster 0!
            assert_eq!(labels[64], 0);
            assert_eq!(labels[65], 0);
            // Total objective: 64 unit matches (dist ~ 0.0) + 2 zero rows (dist = 2.0 each) = ~4.0
            assert!((obj - 4.0).abs() < 1e-3, "expected spherical objective ~4.0, got {obj}");

            // Run resident final epoch + finalize_closing_centroids (keeping cluster 2 unassigned zero!)
            let cfg = EmeansGpuConfig {
                k: k as u32,
                d: d as u32,
                max_chunk_rows: 32,
                precision,
                backend: EmeansGpuBackend::Auto,
                spherical: true,
                unused_threshold: 0.0, // keep cluster 2 unassigned at [0.0; 16]
                split_epsilon: 1e-4,
                init_usage: 1.0,
                seed: 5,
            };
            let mut runner = EmeansGpuRunner::new(rt, cfg, &init_centroids).unwrap();
            let ds = runner.upload_dataset_f32(&rows, n).unwrap();
            let order: Vec<u32> = (0..n).collect();
            let batches = [EpochBatchPlan {
                row_offset: 0,
                row_count: n,
                alpha: 0.5,
                step: 0,
                check_revive: false,
            }];
            let (ep_obj, _, counts) = runner.run_resident_epoch(&ds, &order, &batches, true, None).unwrap();
            assert!(ep_obj.is_finite());
            assert_eq!(counts[2], 0);

            let closing = runner.finalize_closing_centroids().unwrap();
            assert!(closing.iter().all(|v| v.is_finite()), "closing has NaN/Inf");
            for j in 0..2 {
                let norm: f64 = closing[j * d..(j + 1) * d]
                    .iter()
                    .map(|&v| (v as f64) * (v as f64))
                    .sum::<f64>()
                    .sqrt();
                assert!(
                    (norm - 1.0).abs() < 1e-5,
                    "non-zero spherical closing centroid {j} must have unit norm, got {norm}"
                );
            }
            assert!(
                closing[2 * d..3 * d].iter().all(|&v| v == 0.0),
                "zero spherical centroid must stay 0.0 without NaN"
            );
        }
    });
}

#[test]
fn test_gpu_refuses_norm_overflow_and_bf16_inf_without_poisoning() {
    common::with_two_gpus(|rt, foreign_rt| {
        let d = 16usize;
        let k = 4usize;
        let good_centroids = common::random_f32(k * d, 0x1111);
        let good_rows = common::random_f32(32 * d, 0x2222);

        let make_cfg = |precision: EmeansGpuPrecision| EmeansGpuConfig {
            k: k as u32,
            d: d as u32,
            max_chunk_rows: 16,
            precision,
            backend: EmeansGpuBackend::Auto,
            spherical: false,
            unused_threshold: 0.1,
            split_epsilon: 1e-4,
            init_usage: 1.0,
            seed: 1,
        };

        // 1. NaN in init_centroids and input rows
        let mut nan_c = good_centroids.clone();
        nan_c[5] = f32::NAN;
        let err = EmeansGpuRunner::new(rt, make_cfg(EmeansGpuPrecision::F32), &nan_c)
            .err()
            .expect("NaN init_centroids must fail");
        assert!(err.starts_with(NONFINITE_ERR_PREFIX), "got: {err}");
        assert!(!rt.is_poisoned());

        let mut runner = EmeansGpuRunner::new(rt, make_cfg(EmeansGpuPrecision::F32), &good_centroids).unwrap();
        let mut nan_rows = good_rows.clone();
        nan_rows[17] = f32::NAN;
        let err = runner
            .upload_dataset_f32(&nan_rows, 32)
            .err()
            .expect("NaN row must fail");
        assert!(err.starts_with(NONFINITE_ERR_PREFIX), "got: {err}");
        assert!(!rt.is_poisoned());

        // 2. 2.0e19f32 (finite f32 whose squared norm 4.0e38 overflows f32::MAX!)
        let mut norm_ovf_rows = good_rows.clone();
        norm_ovf_rows[3] = 2.0e19f32;
        let err = runner
            .upload_dataset_f32(&norm_ovf_rows, 32)
            .err()
            .expect("2.0e19 norm overflow must fail");
        assert!(err.starts_with(NONFINITE_ERR_PREFIX), "got: {err}");
        assert!(!rt.is_poisoned());

        // Also test streaming chunk norm overflow refusal
        runner.reset_pass_accumulators(false).unwrap();
        runner.accumulate_f32_chunk(&norm_ovf_rows[..16 * d], 16, None).unwrap();
        let err = runner
            .apply_batch_update(0.5, 0, false)
            .expect_err("streaming 2.0e19 must fail on sync");
        assert!(err.starts_with(NONFINITE_ERR_PREFIX), "got: {err}");
        assert!(!rt.is_poisoned());

        // 3. 3.40e38f32 on Bf16 (finite in f32, but rounds above max bf16 3.38953139e38 -> bf16 Inf!)
        let runner_bf16 = EmeansGpuRunner::new(rt, make_cfg(EmeansGpuPrecision::Bf16), &good_centroids).unwrap();
        let mut bf16_ovf_rows = good_rows.clone();
        bf16_ovf_rows[0] = 3.40e38f32;
        let err = runner_bf16
            .upload_dataset_f32(&bf16_ovf_rows, 32)
            .err()
            .expect("3.40e38 bf16 overflow must fail");
        assert!(err.starts_with(NONFINITE_ERR_PREFIX), "got: {err}");
        assert!(!rt.is_poisoned());

        // Raw bf16 Inf bits (0x7f80) must also fail closed without poisoning
        let mut bad_bf16_bits = f32_slice_to_bf16(&good_rows);
        bad_bf16_bits[9] = 0x7f80;
        let err = runner_bf16
            .upload_dataset_bf16(&bad_bf16_bits, 32)
            .err()
            .expect("bf16 Inf bits must fail");
        assert!(err.starts_with(NONFINITE_ERR_PREFIX), "got: {err}");
        assert!(!rt.is_poisoned());

        // 4. Foreign-runtime GpuBuffer passed to upload_dataset_{f32,bf16}_buffer
        let foreign_f32_buf = common::buf(foreign_rt, &good_rows);
        let err = runner
            .upload_dataset_f32_buffer(&foreign_f32_buf, 32)
            .err()
            .expect("foreign f32 buffer must be refused before Binder");
        assert!(err.contains("runtime"), "got: {err}");
        assert!(!rt.is_poisoned());
        assert!(!foreign_rt.is_poisoned());

        let foreign_bf16_buf = common::buf_bf16(foreign_rt, &good_rows);
        let err = runner_bf16
            .upload_dataset_bf16_buffer(&foreign_bf16_buf, 32)
            .err()
            .expect("foreign bf16 buffer must be refused before Binder");
        assert!(err.contains("runtime"), "got: {err}");
        assert!(!rt.is_poisoned());

        // 5. Verify `rt` is completely healthy afterward AND verify OjasBudgetLedger end-to-end
        let ds_ok = runner
            .upload_dataset_f32(&good_rows, 32)
            .expect("rt must remain usable after clean refusals");
        let order: Vec<u32> = (0..32).collect();
        let batches = [EpochBatchPlan {
            row_offset: 0,
            row_count: 32,
            alpha: 0.5,
            step: 0,
            check_revive: true,
        }];
        let (obj, _, counts) = runner
            .run_resident_epoch(&ds_ok, &order, &batches, true, None)
            .expect("healthy epoch after refusals");
        assert!(obj.is_finite());
        assert_eq!(counts.iter().sum::<u64>(), 32);

        // Also verify OjasBudgetLedger with a fresh runtime on the same device
        let budget = Budget::new(64 * 1024); // 64 KiB cap
        let ledger = OjasBudgetLedger::new(&budget, 32 * 1024);
        let capped_rt = GpuRuntime::with_shared(Arc::clone(rt.shared())).expect("capped_rt");
        capped_rt
            .set_pool_ledger(Some(ledger.as_pool_ledger()))
            .expect("set_pool_ledger");
        let small = capped_rt.alloc_buffer(16 * 1024).expect("fits in 64 KiB");
        assert_eq!(ledger.live_bytes(), 16 * 1024);
        let too_big_err = capped_rt.alloc_buffer(128 * 1024).expect_err("must exceed 64 KiB cap");
        match OjasBudgetLedger::capacity_error(&too_big_err) {
            Some(OjasError::CapacityExceeded { cap, .. }) => assert_eq!(cap, 64 * 1024),
            other => panic!("expected CapacityExceeded, got {other:?} for {too_big_err}"),
        }
        drop(small);
        capped_rt.synchronize().unwrap();
        capped_rt.reclaim(u64::MAX).unwrap();
        assert_eq!(ledger.live_bytes(), 0);
        assert_eq!(ledger.cached_bytes(), 16 * 1024);
        capped_rt.trim_free_list();
        assert_eq!(ledger.cached_bytes(), 0);
        assert_eq!(ledger.unmatched_releases(), 0);
    });
}

#[test]
fn test_streaming_and_chunked_assign_bound_staging_under_tight_budget() {
    common::with_gpu(|rt| {
        let k = 16usize;
        let d = 256usize;
        let chunk_cap = 256u32;
        let num_chunks = 16usize;
        let n = num_chunks * (chunk_cap as usize); // 4096 rows -> 4 MiB f32 (strictly > 2.5 MiB cap)

        let rows_f32 = common::random_f32(n * d, 9101);
        let rows_bf16_n = n * 2; // 8192 rows -> 4 MiB bf16 (strictly > 2.5 MiB cap)
        let rows_bf16 = f32_slice_to_bf16(&common::random_f32(rows_bf16_n * d, 9102));
        let init_centroids = common::random_f32(k * d, 9103);

        // Reference uncapped streaming & assign results
        let ref_cfg = EmeansGpuConfig {
            k: k as u32,
            d: d as u32,
            max_chunk_rows: chunk_cap,
            precision: EmeansGpuPrecision::F32,
            backend: EmeansGpuBackend::Auto,
            spherical: false,
            unused_threshold: 0.0,
            split_epsilon: 1e-4,
            init_usage: 1.0,
            seed: 42,
        };
        let mut ref_runner = EmeansGpuRunner::new(rt, ref_cfg.clone(), &init_centroids).expect("ref_runner");
        ref_runner.reset_pass_accumulators(true).unwrap();
        for c in 0..num_chunks {
            let off = c * (chunk_cap as usize) * d;
            ref_runner
                .accumulate_f32_chunk(&rows_f32[off..off + (chunk_cap as usize) * d], chunk_cap, None)
                .unwrap();
        }
        ref_runner.apply_batch_update(0.5, 0, false).unwrap();
        let (ref_obj, ref_counts) = ref_runner.finish_streaming_epoch().unwrap();
        let ref_closing = ref_runner.finalize_closing_centroids().unwrap();
        let (ref_assign_labels, ref_assign_obj) = assign_gpu_f32(
            rt,
            &rows_f32,
            n as u32,
            d as u32,
            &init_centroids,
            k as u32,
            chunk_cap,
            EmeansGpuPrecision::F32,
            EmeansGpuBackend::Auto,
            false,
        )
        .expect("ref assign_gpu_f32");

        // Tight budget cap: 2.5 MiB (< 4 MiB dataset size), proving O(chunk_cap * d) live memory!
        let cap_bytes: u64 = 2_500_000;
        let budget = Budget::new(cap_bytes);
        let ledger = OjasBudgetLedger::new(&budget, 512 * 1024);
        let capped_rt = GpuRuntime::with_shared(Arc::clone(rt.shared())).expect("capped_rt");
        capped_rt
            .set_pool_ledger(Some(ledger.as_pool_ledger()))
            .expect("set_pool_ledger");

        // 1. Multi-chunk f32 streaming (16 separate chunks + single oversized 4096-row sub-windowed call)
        {
            let mut runner =
                EmeansGpuRunner::new(&capped_rt, ref_cfg.clone(), &init_centroids).expect("capped f32 runner");
            runner.reset_pass_accumulators(true).unwrap();
            for c in 0..num_chunks {
                let off = c * (chunk_cap as usize) * d;
                runner
                    .accumulate_f32_chunk(&rows_f32[off..off + (chunk_cap as usize) * d], chunk_cap, None)
                    .expect("bounded f32 chunk under 2.5 MiB cap");
                assert!(ledger.live_bytes() <= cap_bytes);
            }
            runner.apply_batch_update(0.5, 0, false).unwrap();
            let (obj, counts) = runner.finish_streaming_epoch().unwrap();
            let closing = runner.finalize_closing_centroids().unwrap();
            assert_eq!(obj.to_bits(), ref_obj.to_bits());
            assert_eq!(counts, ref_counts);
            assert_eq!(closing, ref_closing);

            // Single 4096-row call sub-windowed into 16 x 256-row windows must also stay bounded
            let mut runner2 = EmeansGpuRunner::new(&capped_rt, ref_cfg, &init_centroids).expect("capped f32 runner2");
            drop(runner);
            capped_rt.synchronize().unwrap();
            capped_rt.reclaim(u64::MAX).unwrap();
            runner2.reset_pass_accumulators(false).unwrap();
            runner2
                .accumulate_f32_chunk(&rows_f32, n as u32, None)
                .expect("oversized sub-windowed f32 chunk under 2.5 MiB cap");
            runner2.apply_batch_update(0.5, 0, false).unwrap();
            let (obj2, counts2) = runner2.finish_streaming_epoch().unwrap();
            assert_eq!(obj2.to_bits(), ref_obj.to_bits());
            assert_eq!(counts2, ref_counts);
        }
        capped_rt.synchronize().unwrap();
        capped_rt.reclaim(u64::MAX).unwrap();
        capped_rt.trim_free_list();

        // 2. Multi-chunk bf16 streaming (32 chunks = 8192 rows = 4 MiB bf16 under 2.5 MiB cap)
        {
            let bf16_cfg = EmeansGpuConfig {
                k: k as u32,
                d: d as u32,
                max_chunk_rows: chunk_cap,
                precision: EmeansGpuPrecision::Bf16,
                backend: EmeansGpuBackend::Auto,
                spherical: false,
                unused_threshold: 0.0,
                split_epsilon: 1e-4,
                init_usage: 1.0,
                seed: 42,
            };
            let mut bf16_runner = EmeansGpuRunner::new(&capped_rt, bf16_cfg, &init_centroids).expect("bf16 runner");
            bf16_runner.reset_pass_accumulators(false).unwrap();
            bf16_runner
                .accumulate_bf16_chunk(&rows_bf16, rows_bf16_n as u32, None)
                .expect("oversized sub-windowed bf16 chunk under 2.5 MiB cap");
            bf16_runner.apply_batch_update(0.5, 0, false).unwrap();
            let (bf16_obj, bf16_counts) = bf16_runner.finish_streaming_epoch().unwrap();
            assert!(bf16_obj.is_finite());
            assert_eq!(bf16_counts.iter().sum::<u64>(), rows_bf16_n as u64);
        }
        capped_rt.synchronize().unwrap();
        capped_rt.reclaim(u64::MAX).unwrap();
        capped_rt.trim_free_list();

        // 3. Standalone assign_gpu_f32 & assign_gpu_bf16 under the same 2.5 MiB cap
        let (got_labels, got_obj) = assign_gpu_f32(
            &capped_rt,
            &rows_f32,
            n as u32,
            d as u32,
            &init_centroids,
            k as u32,
            chunk_cap,
            EmeansGpuPrecision::F32,
            EmeansGpuBackend::Auto,
            false,
        )
        .expect("assign_gpu_f32 must stay O(chunk_cap * d) under 2.5 MiB cap");
        assert_eq!(got_labels, ref_assign_labels);
        assert_eq!(got_obj.to_bits(), ref_assign_obj.to_bits());

        capped_rt.synchronize().unwrap();
        capped_rt.reclaim(u64::MAX).unwrap();
        capped_rt.trim_free_list();

        let (bf16_labels, bf16_assign_obj) = assign_gpu_bf16(
            &capped_rt,
            &rows_bf16,
            rows_bf16_n as u32,
            d as u32,
            &init_centroids,
            k as u32,
            chunk_cap,
            EmeansGpuPrecision::Bf16,
            EmeansGpuBackend::Auto,
            false,
        )
        .expect("assign_gpu_bf16 must stay O(chunk_cap * d) under 2.5 MiB cap");
        assert_eq!(bf16_labels.len(), rows_bf16_n);
        assert!(bf16_assign_obj.is_finite());

        capped_rt.synchronize().unwrap();
        capped_rt.reclaim(u64::MAX).unwrap();
        capped_rt.trim_free_list();
        assert_eq!(ledger.live_bytes(), 0);
        assert_eq!(ledger.cached_bytes(), 0);
        assert_eq!(ledger.unmatched_releases(), 0);
    });
}

#[test]
fn test_apply_ema_zero_usage_and_zero_alpha_produces_finite_centroids() {
    common::with_gpu(|rt| {
        let k = 4usize;
        let d = 16usize;
        let init_centroids: Vec<f32> = (0..k * d).map(|i| ((i % 7) as f32) * 0.5 - 1.0).collect();
        let rows = common::random_f32(32 * d, 9201);

        for precision in [
            EmeansGpuPrecision::Bf16,
            EmeansGpuPrecision::Bf16AssignF32Stats,
            EmeansGpuPrecision::F32,
        ] {
            // Case 1: init_usage = 0.0 + alpha = 0.0 on both resident and streaming paths
            let cfg = EmeansGpuConfig {
                k: k as u32,
                d: d as u32,
                max_chunk_rows: 32,
                precision,
                backend: EmeansGpuBackend::Auto,
                spherical: false,
                unused_threshold: 0.5,
                split_epsilon: 1e-4,
                init_usage: 0.0,
                seed: 77,
            };
            let mut runner = EmeansGpuRunner::new(rt, cfg, &init_centroids).expect("runner init_usage=0");
            let ds = runner.upload_dataset_f32(&rows, 32).expect("upload");
            let order: Vec<u32> = (0..32).collect();
            let batches = [EpochBatchPlan {
                row_offset: 0,
                row_count: 32,
                alpha: 0.0,
                step: 0,
                check_revive: true,
            }];
            let (obj, revivals, counts) = runner
                .run_resident_epoch(&ds, &order, &batches, false, None)
                .expect("resident alpha=0 with usage=0 must not produce NaN");
            assert!(obj.is_finite());
            assert_eq!(revivals, 0);
            assert_eq!(counts.iter().sum::<u64>(), 32);
            let cents = runner.read_centroids().expect("read_centroids");
            let usage = runner.read_usage().expect("read_usage");
            assert!(cents.iter().all(|v| v.is_finite()), "NaN/Inf in centroids: {cents:?}");
            assert_eq!(cents, init_centroids, "alpha=0 must preserve c_old bit-for-bit");
            assert_eq!(usage, vec![0.0f32; k]);

            // Streaming alpha=0.0 and subnormal alpha (FTZs to 0.0 on GPU) after alpha=1.0 empties unoccupied clusters
            runner.reset_pass_accumulators(false).unwrap();
            let single_cluster_rows = init_centroids[..d].repeat(16);
            runner.accumulate_f32_chunk(&single_cluster_rows, 16, None).unwrap();
            runner
                .apply_batch_update(1.0, 1, false)
                .expect("alpha=1 empties k=1..3");
            let u_after_replace = runner.read_usage().unwrap();
            assert_eq!(u_after_replace[1], 0.0);

            // Now send rows landing on cluster 1 with alpha = 0.0 and subnormal alpha
            let c1_rows = init_centroids[d..2 * d].repeat(8);
            for &tiny_alpha in &[0.0f32, f32::from_bits(1)] {
                runner.accumulate_f32_chunk(&c1_rows, 8, None).unwrap();
                runner
                    .apply_batch_update(tiny_alpha, 2, false)
                    .expect("zero/subnormal alpha on usage=0 cluster must succeed");
                let cents_after = runner.read_centroids().unwrap();
                assert!(
                    cents_after.iter().all(|v| v.is_finite()),
                    "NaN/Inf for tiny_alpha={tiny_alpha:e}: {cents_after:?}"
                );
                assert_eq!(&cents_after[d..2 * d], &init_centroids[d..2 * d]);
            }
        }
    });
}

#[test]
fn test_streaming_faults_latch_and_resident_last_batch_prep_catches_overflow() {
    common::with_gpu(|rt| {
        let k = 4usize;
        let d = 16usize;
        let good_centroids: Vec<f32> = (0..k * d).map(|i| ((i % 9) as f32) * 0.25).collect();
        let good_rows = common::random_f32(32 * d, 9301);
        let mut bad_rows = good_rows[..16 * d].to_vec();
        bad_rows[5] = 2.0e19f32; // overflows row_norm2

        let cfg = EmeansGpuConfig {
            k: k as u32,
            d: d as u32,
            max_chunk_rows: 16,
            precision: EmeansGpuPrecision::F32,
            backend: EmeansGpuBackend::Auto,
            spherical: false,
            unused_threshold: 0.0,
            split_epsilon: 1e-4,
            init_usage: 1.0,
            seed: 55,
        };
        let mut runner = EmeansGpuRunner::new(rt, cfg, &good_centroids).expect("runner");

        // 1. Start a final streaming epoch, feed 1 good chunk + 1 bad chunk
        runner.reset_pass_accumulators(true).unwrap();
        runner.accumulate_f32_chunk(&good_rows[..16 * d], 16, None).unwrap();
        runner.accumulate_f32_chunk(&bad_rows, 16, None).unwrap();

        // read_centroids and read_usage must observe AND keep the fault latched!
        let err_c = runner.read_centroids().expect_err("read_centroids must report fault");
        assert!(err_c.starts_with(NONFINITE_ERR_PREFIX), "got: {err_c}");
        let err_u = runner
            .read_usage()
            .expect_err("read_usage must still report latched fault");
        assert!(err_u.starts_with(NONFINITE_ERR_PREFIX), "got: {err_u}");

        // upload_dataset_f32 / upload_dataset_f32_buffer must use local fault_flags and NOT clear latched fault!
        let valid_ds = runner
            .upload_dataset_f32(&good_rows, 32)
            .expect("upload_dataset_f32 on valid data must succeed independently");
        assert_eq!(valid_ds.rows(), 32);

        // apply_batch_update, accumulate_f32_chunk, finish_streaming_epoch, finalize_closing_centroids ALL fail closed!
        let err_upd = runner
            .apply_batch_update(1.0, 0, false)
            .expect_err("apply_batch_update must refuse latched fault");
        assert!(err_upd.starts_with(NONFINITE_ERR_PREFIX), "got: {err_upd}");
        let err_acc = runner
            .accumulate_f32_chunk(&good_rows[..16 * d], 16, None)
            .expect_err("accumulate_f32_chunk must refuse while faulted");
        assert!(err_acc.starts_with(NONFINITE_ERR_PREFIX), "got: {err_acc}");
        let err_fin = runner
            .finish_streaming_epoch()
            .expect_err("finish_streaming_epoch must refuse dirty epoch accumulators");
        assert!(err_fin.starts_with(NONFINITE_ERR_PREFIX), "got: {err_fin}");
        let err_close = runner
            .finalize_closing_centroids()
            .expect_err("finalize_closing_centroids must refuse dirty pass_sums");
        assert!(err_close.starts_with(NONFINITE_ERR_PREFIX), "got: {err_close}");

        // 2. Also verify synchronous `labels_out = Some(...)` fault latches across epoch end & closing
        runner.reset_pass_accumulators(true).unwrap();
        assert_eq!(
            runner.read_centroids().expect("reset_pass_accumulators clears fault"),
            good_centroids
        );
        let mut lbl_out = vec![0u32; 16];
        let err_sync = runner
            .accumulate_f32_chunk(&bad_rows, 16, Some(&mut lbl_out))
            .expect_err("sync chunk must report fault");
        assert!(err_sync.starts_with(NONFINITE_ERR_PREFIX), "got: {err_sync}");
        assert!(runner.finish_streaming_epoch().is_err());
        assert!(runner.finalize_closing_centroids().is_err());

        // Recover via reset_pass_accumulators(false) — dirty pass_sums from the aborted final_epoch=true
        // must also have been wiped so finalize_closing_centroids falls back to untouched good_centroids!
        runner.reset_pass_accumulators(false).unwrap();
        runner.accumulate_f32_chunk(&good_rows[..16 * d], 16, None).unwrap();
        let (clean_obj, clean_counts) = runner.finish_streaming_epoch().expect("clean epoch after reset");
        assert!(clean_obj.is_finite());
        assert_eq!(clean_counts.iter().sum::<u64>(), 16);
        let closed_after_abort = runner
            .finalize_closing_centroids()
            .expect("dirty pass_counts from aborted final epoch must be zeroed");
        assert_eq!(closed_after_abort, good_centroids);

        // Also test recovery from latched fault via run_resident_epoch
        runner.reset_pass_accumulators(false).unwrap();
        runner.accumulate_f32_chunk(&bad_rows, 16, None).unwrap();
        assert!(runner.apply_batch_update(0.5, 0, false).is_err());
        let order: Vec<u32> = (0..32).collect();
        let batches = [EpochBatchPlan {
            row_offset: 0,
            row_count: 32,
            alpha: 0.5,
            step: 0,
            check_revive: false,
        }];
        let (res_obj, _, res_counts) = runner
            .run_resident_epoch(&valid_ds, &order, &batches, false, None)
            .expect("run_resident_epoch resets all epoch accumulators and recovers");
        assert!(res_obj.is_finite());
        assert_eq!(res_counts.iter().sum::<u64>(), 32);

        // 3. Resident epoch last-batch revival overflow when final_epoch == false must fail closed!
        // Row [4.0e18; 16] has ||x||_2^2 = 16 * 1.6e37 = 2.56e38 < FLT_MAX (valid!),
        // but splitting with split_epsilon = 0.9 produces 8 cols at 7.6e18 -> ||c||_2^2 = 4.62e38 > FLT_MAX!
        let mut init_large = vec![0.0f32; 2 * d];
        for v in &mut init_large[..d] {
            *v = 4.0e18f32;
        }
        let large_rows = vec![4.0e18f32; 16 * d];
        let ovf_cfg = EmeansGpuConfig {
            k: 2,
            d: d as u32,
            max_chunk_rows: 16,
            precision: EmeansGpuPrecision::F32,
            backend: EmeansGpuBackend::Auto,
            spherical: false,
            unused_threshold: 0.5,
            split_epsilon: 0.9,
            init_usage: 1.0e-5,
            seed: 99,
        };
        let mut ovf_runner = EmeansGpuRunner::new(rt, ovf_cfg, &init_large).expect("ovf_runner init valid");
        let ovf_ds = ovf_runner
            .upload_dataset_f32(&large_rows, 16)
            .expect("large_rows valid");
        let ovf_order: Vec<u32> = (0..16).collect();
        let ovf_batches = [EpochBatchPlan {
            row_offset: 0,
            row_count: 16,
            alpha: 1.0,
            step: 0,
            check_revive: true,
        }];
        let err_res_ovf = ovf_runner
            .run_resident_epoch(&ovf_ds, &ovf_order, &ovf_batches, false, None)
            .expect_err("last-batch revival norm2 overflow with final_epoch=false must fail closed");
        assert!(err_res_ovf.starts_with(NONFINITE_ERR_PREFIX), "got: {err_res_ovf}");
        assert!(
            ovf_runner.read_centroids().is_err(),
            "read_centroids must stay latched after revival overflow"
        );
    });
}

fn build_k1024_zipf_fixture(n: usize, d: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    fn mix(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn next_u64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix(*state)
    }
    fn next_bounded_u64(state: &mut u64, bound: u64) -> u64 {
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let r = next_u64(state);
            let m = u128::from(r) * u128::from(bound);
            if (m as u64) >= threshold {
                return (m >> 64) as u64;
            }
        }
    }
    fn normal_pair(state: &mut u64) -> (f64, f64) {
        let u1 = ((next_u64(state) >> 11) | 1) as f64 * (1.0 / ((1_u64 << 53) as f64));
        let u2 = (next_u64(state) >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64));
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }
    fn fill_normal_f32(state: &mut u64, scale: f64, dst: &mut [f32]) {
        let mut i = 0;
        while i + 1 < dst.len() {
            let (z0, z1) = normal_pair(state);
            dst[i] = (z0 * scale) as f32;
            dst[i + 1] = (z1 * scale) as f32;
            i += 2;
        }
        if i < dst.len() {
            let (z0, _) = normal_pair(state);
            dst[i] = (z0 * scale) as f32;
        }
    }

    let mut state = seed;
    let inv_sqrt_d = 1.0 / (d as f64).sqrt();
    let mut true_centers = vec![0.0_f32; k * d];
    fill_normal_f32(&mut state, inv_sqrt_d, &mut true_centers);

    // Power-law Zipf component weights w_j ~ (j + 16)^(-0.75) so uniform Forgy initialization
    // seeds multiple centroids inside dense modes while leaving hundreds of tail clusters dead.
    let mut cdf = Vec::with_capacity(k);
    let mut weight_sum = 0.0_f64;
    for j in 0..k {
        weight_sum += ((j + 16) as f64).powf(-0.75);
        cdf.push(weight_sum);
    }
    for val in &mut cdf {
        *val /= weight_sum;
    }
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }

    let mut x_rows = vec![0.0_f32; n * d];
    let mut noise_buf = vec![0.0_f32; d];
    for r in 0..n {
        let u = (next_u64(&mut state) >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64));
        let comp = cdf.partition_point(|&p| p < u).min(k - 1);
        fill_normal_f32(&mut state, 0.25 * inv_sqrt_d, &mut noise_buf);
        let c_row = &true_centers[comp * d..(comp + 1) * d];
        let dst = &mut x_rows[r * d..(r + 1) * d];
        for col in 0..d {
            dst[col] = c_row[col] + noise_buf[col];
        }
    }

    let mut indices: Vec<u32> = (0..(n as u32)).collect();
    let mut init_state = mix(seed ^ 0x494e_4954_4345_4e54);
    for i in 0..k {
        let pick = i + (next_bounded_u64(&mut init_state, (n - i) as u64) as usize);
        indices.swap(i, pick);
    }
    let mut init_centroids = vec![0.0_f32; k * d];
    for (j, &row_idx) in indices[..k].iter().enumerate() {
        let src = (row_idx as usize) * d;
        init_centroids[j * d..(j + 1) * d].copy_from_slice(&x_rows[src..src + d]);
    }
    (x_rows, init_centroids)
}

fn shuffled_epoch_order(n: u32, seed: u64, epoch: u64) -> Vec<u32> {
    let mut order: Vec<u32> = (0..n).collect();
    let mut state = seed ^ epoch.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    for i in (1..order.len()).rev() {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        let j = (z as usize) % (i + 1);
        order.swap(i, j);
    }
    order
}

#[test]
fn test_k1024_multi_epoch_repeat_and_streaming_bit_identity() {
    common::with_gpu(|rt| {
        // Exercises multi-pass CSR radix (&[16, 24]), 4-tile donor radix, part_k=1024,
        // 16 tiles of 64x64 TensorOps, partial tail batches, init_usage=1/65536, and pool reclaim.
        let n = 36_000usize;
        let d = 128usize;
        let k = 1_024usize;
        let batch_size = 4_096u32;
        let alpha0 = 0.314537f32;
        let seed = 0x2026_1110_0128_1024u64;
        let repeats = 8usize;

        let (rows, init_centroids) = build_k1024_zipf_fixture(n, d, k, seed);
        let order_ep0 = shuffled_epoch_order(n as u32, seed, 0);
        let order_ep1 = shuffled_epoch_order(n as u32, seed, 1);

        let batches_per_epoch = (n as u32).div_ceil(batch_size);
        let total_steps = (batches_per_epoch * 2) as usize;
        let build_batches = |step_base: u64| -> Vec<EpochBatchPlan> {
            (0..batches_per_epoch)
                .map(|b| {
                    let row_offset = b * batch_size;
                    let row_count = (n as u32 - row_offset).min(batch_size);
                    let step = step_base + u64::from(b);
                    let frac = ((step as usize).min(total_steps) as f64) / (total_steps as f64);
                    let cos = 0.5 * (1.0 + (std::f64::consts::PI * frac).cos());
                    let alpha = ((alpha0 as f64) * (0.1 + 0.9 * cos)) as f32;
                    EpochBatchPlan {
                        row_offset,
                        row_count,
                        alpha,
                        step,
                        check_revive: (step + 1).is_multiple_of(5) || b + 1 == batches_per_epoch,
                    }
                })
                .collect()
        };
        let batches_ep0 = build_batches(0);
        let batches_ep1 = build_batches(batches_ep0.len() as u64);

        let mut backends = vec![EmeansGpuBackend::Simdgroup];
        if rt.has_tensorops() {
            backends.insert(0, EmeansGpuBackend::TensorOps);
        }

        for backend in backends {
            for precision in [
                EmeansGpuPrecision::F32,
                EmeansGpuPrecision::Bf16AssignF32Stats,
                EmeansGpuPrecision::Bf16,
            ] {
                #[derive(Debug, PartialEq, Eq)]
                struct FitBits {
                    ep0_obj_bits: u64,
                    ep0_revivals: usize,
                    ep0_counts: Vec<u64>,
                    ep1_obj_bits: u64,
                    ep1_revivals: usize,
                    ep1_counts: Vec<u64>,
                    labels: Vec<u32>,
                    usage_bits: Vec<u32>,
                    closing_bits: Vec<u32>,
                }

                let mut ref_fit: Option<FitBits> = None;
                for rep in 0..repeats {
                    let cfg = EmeansGpuConfig {
                        k: k as u32,
                        d: d as u32,
                        max_chunk_rows: batch_size,
                        precision,
                        backend,
                        spherical: false,
                        unused_threshold: 0.1,
                        split_epsilon: 1.0e-4,
                        init_usage: 1.0 / 65_536.0,
                        seed,
                    };
                    let mut runner = EmeansGpuRunner::new(rt, cfg, &init_centroids).expect("runner");
                    let ds = runner.upload_dataset_f32(&rows, n as u32).expect("upload");
                    let (ep0_obj, ep0_rev, ep0_counts) = runner
                        .run_resident_epoch(&ds, &order_ep0, &batches_ep0, false, None)
                        .expect("ep0");
                    let mut labels = vec![0u32; n];
                    let (ep1_obj, ep1_rev, ep1_counts) = runner
                        .run_resident_epoch(&ds, &order_ep1, &batches_ep1, true, Some(&mut labels))
                        .expect("ep1");
                    let closing = runner.finalize_closing_centroids().expect("closing");
                    let usage = runner.read_usage().expect("usage");
                    drop(ds);
                    drop(runner);
                    rt.synchronize().expect("sync");
                    let _ = rt.reclaim(u64::MAX);

                    assert!(
                        ep0_rev + ep1_rev >= 100,
                        "fixture must exercise heavy dead-cluster revival at k=1024 (got {} + {})",
                        ep0_rev,
                        ep1_rev
                    );

                    let cur = FitBits {
                        ep0_obj_bits: ep0_obj.to_bits(),
                        ep0_revivals: ep0_rev,
                        ep0_counts,
                        ep1_obj_bits: ep1_obj.to_bits(),
                        ep1_revivals: ep1_rev,
                        ep1_counts,
                        labels,
                        usage_bits: usage.iter().map(|v| v.to_bits()).collect(),
                        closing_bits: closing.iter().map(|v| v.to_bits()).collect(),
                    };

                    if let Some(ref want) = ref_fit {
                        assert_eq!(
                            cur.ep0_obj_bits, want.ep0_obj_bits,
                            "k=1024 repeat {rep} ep0_obj diverged ({backend:?}, {precision:?}): got {:#x} ({}) vs {:#x} ({}); rev {} vs {}",
                            cur.ep0_obj_bits, ep0_obj, want.ep0_obj_bits, f64::from_bits(want.ep0_obj_bits),
                            cur.ep0_revivals, want.ep0_revivals
                        );
                        assert_eq!(
                            cur.ep0_revivals, want.ep0_revivals,
                            "k=1024 repeat {rep} ep0_revivals diverged ({backend:?}, {precision:?})"
                        );
                        assert_eq!(
                            cur.ep0_counts, want.ep0_counts,
                            "k=1024 repeat {rep} ep0_counts diverged ({backend:?}, {precision:?})"
                        );
                        assert_eq!(
                            cur.ep1_obj_bits, want.ep1_obj_bits,
                            "k=1024 repeat {rep} ep1_obj diverged ({backend:?}, {precision:?}): got {:#x} ({}) vs {:#x} ({}); rev {} vs {}",
                            cur.ep1_obj_bits, ep1_obj, want.ep1_obj_bits, f64::from_bits(want.ep1_obj_bits),
                            cur.ep1_revivals, want.ep1_revivals
                        );
                        assert_eq!(
                            cur.ep1_revivals, want.ep1_revivals,
                            "k=1024 repeat {rep} ep1_revivals diverged ({backend:?}, {precision:?})"
                        );
                        assert_eq!(
                            cur.ep1_counts, want.ep1_counts,
                            "k=1024 repeat {rep} ep1_counts diverged ({backend:?}, {precision:?})"
                        );
                        assert_eq!(
                            cur.labels, want.labels,
                            "k=1024 repeat {rep} labels diverged ({backend:?}, {precision:?})"
                        );
                        assert_eq!(
                            cur.usage_bits, want.usage_bits,
                            "k=1024 repeat {rep} usage bits diverged ({backend:?}, {precision:?})"
                        );
                        assert_eq!(
                            cur.closing_bits, want.closing_bits,
                            "k=1024 repeat {rep} closing centroid bits diverged ({backend:?}, {precision:?})"
                        );
                    } else {
                        ref_fit = Some(cur);
                    }
                }

                // Streaming path with matching chunk boundaries must match `run_resident_epoch`
                // bit-for-bit (0-ULP) at k = 1024, and sub-chunked streaming must also be
                // bit-identical across repeats.
                let want = ref_fit.expect("ref_fit");
                let stream_cfg = EmeansGpuConfig {
                    k: k as u32,
                    d: d as u32,
                    max_chunk_rows: batch_size,
                    precision,
                    backend,
                    spherical: false,
                    unused_threshold: 0.1,
                    split_epsilon: 1.0e-4,
                    init_usage: 1.0 / 65_536.0,
                    seed,
                };
                let mut s_runner = EmeansGpuRunner::new(rt, stream_cfg, &init_centroids).expect("s_runner");

                // Epoch 0 (streaming async chunks matching resident chunk size)
                s_runner.reset_pass_accumulators(false).expect("reset ep0");
                let mut s_rev0 = 0usize;
                for b in &batches_ep0 {
                    let start = b.row_offset as usize;
                    let end = start + (b.row_count as usize);
                    let mut batch_rows = Vec::with_capacity((b.row_count as usize) * d);
                    for &rid in &order_ep0[start..end] {
                        let r = rid as usize;
                        batch_rows.extend_from_slice(&rows[r * d..(r + 1) * d]);
                    }
                    s_runner
                        .accumulate_f32_chunk(&batch_rows, b.row_count, None)
                        .expect("acc ep0");
                    s_rev0 += s_runner
                        .apply_batch_update(b.alpha, b.step, b.check_revive)
                        .expect("upd ep0");
                }
                let (s_obj0, s_counts0) = s_runner.finish_streaming_epoch().expect("fin ep0");

                // Epoch 1 (final epoch with label capture)
                s_runner.reset_pass_accumulators(true).expect("reset ep1");
                let mut s_rev1 = 0usize;
                let mut s_labels = vec![0u32; n];
                for b in &batches_ep1 {
                    let start = b.row_offset as usize;
                    let end = start + (b.row_count as usize);
                    let mut batch_rows = Vec::with_capacity((b.row_count as usize) * d);
                    for &rid in &order_ep1[start..end] {
                        let r = rid as usize;
                        batch_rows.extend_from_slice(&rows[r * d..(r + 1) * d]);
                    }
                    let mut batch_lbls = vec![0u32; b.row_count as usize];
                    s_runner
                        .accumulate_f32_chunk(&batch_rows, b.row_count, Some(&mut batch_lbls))
                        .expect("acc ep1");
                    for (local_i, &rid) in order_ep1[start..end].iter().enumerate() {
                        s_labels[rid as usize] = batch_lbls[local_i];
                    }
                    s_rev1 += s_runner
                        .apply_batch_update(b.alpha, b.step, b.check_revive)
                        .expect("upd ep1");
                }
                let (s_obj1, s_counts1) = s_runner.finish_streaming_epoch().expect("fin ep1");
                let s_closing = s_runner.finalize_closing_centroids().expect("s_closing");
                let s_usage = s_runner.read_usage().expect("s_usage");
                drop(s_runner);
                rt.synchronize().expect("sync");
                let _ = rt.reclaim(u64::MAX);

                assert_eq!(
                    s_obj0.to_bits(),
                    want.ep0_obj_bits,
                    "stream ep0_obj ({backend:?}, {precision:?})"
                );
                assert_eq!(s_rev0, want.ep0_revivals, "stream ep0_rev ({backend:?}, {precision:?})");
                assert_eq!(
                    s_counts0, want.ep0_counts,
                    "stream ep0_counts ({backend:?}, {precision:?})"
                );
                assert_eq!(
                    s_obj1.to_bits(),
                    want.ep1_obj_bits,
                    "stream ep1_obj ({backend:?}, {precision:?})"
                );
                assert_eq!(s_rev1, want.ep1_revivals, "stream ep1_rev ({backend:?}, {precision:?})");
                assert_eq!(
                    s_counts1, want.ep1_counts,
                    "stream ep1_counts ({backend:?}, {precision:?})"
                );
                assert_eq!(s_labels, want.labels, "stream labels ({backend:?}, {precision:?})");
                assert_eq!(
                    s_usage.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want.usage_bits,
                    "stream usage ({backend:?}, {precision:?})"
                );
                assert_eq!(
                    s_closing.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want.closing_bits,
                    "stream closing ({backend:?}, {precision:?})"
                );
            }
        }
    });
}

#[test]
fn test_d512_split_c_repeat_bit_identity() {
    common::with_gpu(|rt| {
        // Shape-B feature width (`d = 512`, multi-iteration `col += 256` loop) combined with
        // prime `k = 1_031` (`part_k = 1_024, split_c = 2` partition reduction) and sub-batch
        // chunking (`max_chunk_rows = 1_024` < `batch_size = 2_048`).
        let n = 8_192usize;
        let d = 512usize;
        let k = 1_031usize;
        let batch_size = 2_048u32;
        let max_chunk_rows = 1_024u32;
        let alpha0 = 0.203799f32;
        let seed = 0x2026_1110_0512_1031u64;
        let repeats = 8usize;

        let (rows, init_centroids) = build_k1024_zipf_fixture(n, d, k, seed);
        let order_ep0 = shuffled_epoch_order(n as u32, seed, 0);
        let batches_per_epoch = (n as u32) / batch_size;
        let batches_ep0: Vec<EpochBatchPlan> = (0..batches_per_epoch)
            .map(|b| {
                let step = u64::from(b);
                let frac = (b as f64) / (batches_per_epoch as f64);
                let cos = 0.5 * (1.0 + (std::f64::consts::PI * frac).cos());
                let alpha = ((alpha0 as f64) * (0.1 + 0.9 * cos)) as f32;
                EpochBatchPlan {
                    row_offset: b * batch_size,
                    row_count: batch_size,
                    alpha,
                    step,
                    check_revive: (b + 1) % 2 == 0,
                }
            })
            .collect();

        let mut backends = vec![EmeansGpuBackend::Simdgroup];
        if rt.has_tensorops() {
            backends.insert(0, EmeansGpuBackend::TensorOps);
        }

        for backend in backends {
            for precision in [
                EmeansGpuPrecision::F32,
                EmeansGpuPrecision::Bf16AssignF32Stats,
                EmeansGpuPrecision::Bf16,
            ] {
                #[derive(Debug, PartialEq, Eq)]
                struct SplitCFitBits {
                    obj_bits: u64,
                    rev: usize,
                    counts: Vec<u64>,
                    labels: Vec<u32>,
                    usage_bits: Vec<u32>,
                    closing_bits: Vec<u32>,
                }

                let mut ref_state: Option<SplitCFitBits> = None;
                for rep in 0..repeats {
                    let cfg = EmeansGpuConfig {
                        k: k as u32,
                        d: d as u32,
                        max_chunk_rows,
                        precision,
                        backend,
                        spherical: false,
                        unused_threshold: 0.1,
                        split_epsilon: 1.0e-4,
                        init_usage: 1.0 / 65_536.0,
                        seed,
                    };
                    let mut runner = EmeansGpuRunner::new(rt, cfg, &init_centroids).expect("runner");
                    let ds = runner.upload_dataset_f32(&rows, n as u32).expect("upload");
                    let mut labels = vec![0u32; n];
                    let (obj, rev, counts) = runner
                        .run_resident_epoch(&ds, &order_ep0, &batches_ep0, true, Some(&mut labels))
                        .expect("epoch");
                    let closing_bits: Vec<u32> = runner
                        .finalize_closing_centroids()
                        .expect("closing")
                        .into_iter()
                        .map(f32::to_bits)
                        .collect();
                    let usage_bits: Vec<u32> = runner
                        .read_usage()
                        .expect("usage")
                        .into_iter()
                        .map(f32::to_bits)
                        .collect();
                    drop(ds);
                    drop(runner);
                    rt.synchronize().expect("sync");
                    let _ = rt.reclaim(u64::MAX);

                    assert!(rev >= 50, "d=512 split_c=2 fixture must trigger revivals (got {rev})");
                    let cur = SplitCFitBits {
                        obj_bits: obj.to_bits(),
                        rev,
                        counts,
                        labels,
                        usage_bits,
                        closing_bits,
                    };
                    if let Some(ref want) = ref_state {
                        assert_eq!(
                            &cur, want,
                            "d=512 k=1031 split_c=2 repeat {rep} diverged ({backend:?}, {precision:?})"
                        );
                    } else {
                        ref_state = Some(cur);
                    }
                }
            }
        }
    });
}

#[test]
fn test_near_flt_max_euclidean_overflow_and_recovery() {
    common::with_gpu(|rt| {
        let mut backends = vec![EmeansGpuBackend::Simdgroup];
        if rt.has_tensorops() {
            backends.insert(0, EmeansGpuBackend::TensorOps);
        }

        fn assert_nonfinite<T: std::fmt::Debug>(res: Result<T, String>, ctx: &str) {
            match res {
                Ok(val) => panic!("expected NONFINITE_ERR_PREFIX for {ctx}, got Ok({val:?})"),
                Err(err) => assert!(
                    err.starts_with(NONFINITE_ERR_PREFIX),
                    "expected error starting with {NONFINITE_ERR_PREFIX:?} for {ctx}, got {err:?}"
                ),
            }
        }

        for backend in backends {
            for precision in [
                EmeansGpuPrecision::F32,
                EmeansGpuPrecision::Bf16AssignF32Stats,
                EmeansGpuPrecision::Bf16,
            ] {
                for d in [1usize, 32usize] {
                    let pad = |lead: f32| -> Vec<f32> {
                        let mut v = vec![0.0f32; d];
                        v[0] = lead;
                        v
                    };

                    // 1. Sub-case 1 (`rn + best_s` overflow to `+Inf`: `x = [-1.0e19]`, `c = [+1.0e19]`,
                    //    plus `x = [-8.0e18]`, `c = [+1.05e19]` where `rn <= 7e37` and `sc < FLT_MAX`
                    //    yet `rn + sc > FLT_MAX`) and Sub-case 2 (`2*dot` overflow to `-Inf`:
                    //    `x = [-1.5e19]`, `c = [+1.5e19]`).
                    for (case_name, x_lead, c_lead) in [
                        ("opp_rn_plus_sc_ovf", -1.0e19f32, 1.0e19f32),
                        ("opp_low_rn_sum_ovf", -8.0e18f32, 1.05e19f32),
                        ("opp_two_dot_neg_inf", -1.5e19f32, 1.5e19f32),
                    ] {
                        let x_row = pad(x_lead);
                        let x_bf16 = f32_slice_to_bf16(&x_row);
                        for k in [1usize, 1_026usize] {
                            let mut cents = Vec::with_capacity(k * d);
                            for _ in 0..k {
                                cents.extend_from_slice(&pad(c_lead));
                            }
                            let ctx = format!("{case_name} ({backend:?}, {precision:?}, d={d}, k={k})");

                            assert_nonfinite(
                                assign_gpu_f32(
                                    rt, &x_row, 1, d as u32, &cents, k as u32, 64, precision, backend, false,
                                ),
                                &format!("assign_gpu_f32 {ctx}"),
                            );
                            assert_nonfinite(
                                assign_gpu_bf16(
                                    rt, &x_bf16, 1, d as u32, &cents, k as u32, 64, precision, backend, false,
                                ),
                                &format!("assign_gpu_bf16 {ctx}"),
                            );

                            // Resident epoch (`alpha = 0.5`, `check_revive = true`) must refuse with
                            // non-finite WITHOUT mutating `centroids` or `usage`, then recover cleanly.
                            let cfg = EmeansGpuConfig {
                                k: k as u32,
                                d: d as u32,
                                max_chunk_rows: 64,
                                precision,
                                backend,
                                spherical: false,
                                unused_threshold: 0.1,
                                split_epsilon: 1.0e-4,
                                init_usage: 1.0,
                                seed: 42,
                            };
                            let mut runner = EmeansGpuRunner::new(rt, cfg, &cents).expect("runner");
                            let bad_ds = runner.upload_dataset_f32(&x_row, 1).expect("upload bad_ds");
                            let plan = [EpochBatchPlan {
                                row_offset: 0,
                                row_count: 1,
                                alpha: 0.5,
                                step: 0,
                                check_revive: true,
                            }];
                            let mut lbl_out = [0u32; 1];
                            assert_nonfinite(
                                runner.run_resident_epoch(&bad_ds, &[0], &plan, true, Some(&mut lbl_out)),
                                &format!("resident {ctx}"),
                            );
                            assert_nonfinite(
                                runner.read_centroids(),
                                &format!("read_centroids while fault-latched {ctx}"),
                            );

                            // Recover on the same `runner` with valid in-range row (`+0.5 * c_lead`).
                            let ok_row = pad(0.5 * c_lead);
                            let ok_ds = runner.upload_dataset_f32(&ok_row, 1).expect("upload ok_ds");
                            let ok_plan = [EpochBatchPlan {
                                row_offset: 0,
                                row_count: 1,
                                alpha: 0.0,
                                step: 0,
                                check_revive: false,
                            }];
                            let (ok_obj, _, ok_counts) = runner
                                .run_resident_epoch(&ok_ds, &[0], &ok_plan, true, Some(&mut lbl_out))
                                .expect("resident recovery after non-finite");
                            assert_eq!(lbl_out, [0]);
                            assert!(ok_obj.is_finite() && ok_obj > 0.0);
                            assert_eq!(ok_counts[0], 1);
                            let after_recover_cents = runner.read_centroids().expect("read centroids after recovery");
                            assert_eq!(
                                after_recover_cents[0], cents[0],
                                "faulted resident epoch (alpha=0.5) must not mutate centroids ({ctx})"
                            );

                            // Streaming sync (`Some(&mut lbls)`) + async (`None` -> `apply_batch_update`) + recovery.
                            runner.reset_pass_accumulators(true).expect("reset sync");
                            assert_nonfinite(
                                runner.accumulate_f32_chunk(&x_row, 1, Some(&mut lbl_out)),
                                &format!("stream_sync_f32 {ctx}"),
                            );
                            runner.reset_pass_accumulators(false).expect("reset async");
                            runner.accumulate_bf16_chunk(&x_bf16, 1, None).expect("async enqueue");
                            assert_nonfinite(
                                runner.apply_batch_update(0.0, 0, false),
                                &format!("stream_async_bf16 {ctx}"),
                            );
                            runner.reset_pass_accumulators(true).expect("reset recover");
                            runner
                                .accumulate_f32_chunk(&ok_row, 1, Some(&mut lbl_out))
                                .expect("stream recovery accum");
                            let (rec_obj, rec_counts) =
                                runner.finish_streaming_epoch().expect("stream recovery finish");
                            assert_eq!(lbl_out, [0]);
                            assert!(rec_obj.is_finite() && rec_obj > 0.0);
                            assert_eq!(rec_counts[0], 1);
                        }
                    }

                    // 2. Partial overflow across clusters (including `split_c = 2` where ALL 1024 clusters
                    //    in partition 0 overflow `f32` while cluster `k - 1` in partition 1 has finite
                    //    distance `(-1.5e19 - 0.2e19)^2 = 2.89e38 < FLT_MAX` with `best_s > 0`!).
                    let x_part = pad(-1.5e19f32);
                    let x_part_eff = effective_vec(&x_part, precision)[0];
                    for k in [2usize, 1_026usize] {
                        let mut cents = Vec::with_capacity(k * d);
                        for _ in 0..(k - 1) {
                            cents.extend_from_slice(&pad(1.5e19f32));
                        }
                        cents.extend_from_slice(&pad(0.2e19f32));
                        let c_win_eff = effective_vec(&pad(0.2e19f32), precision)[0];
                        let want_d2 = f64::from((x_part_eff - c_win_eff) * (x_part_eff - c_win_eff));

                        let (lbls, obj) = assign_gpu_f32(
                            rt, &x_part, 1, d as u32, &cents, k as u32, 64, precision, backend, false,
                        )
                        .expect("partial overflow assign_gpu_f32");
                        assert_eq!(
                            lbls,
                            vec![(k - 1) as u32],
                            "partial overflow must select sole finite cluster ({backend:?}, {precision:?}, d={d}, k={k})"
                        );
                        let rel = (obj - want_d2).abs() / want_d2;
                        assert!(
                            rel <= 1.0e-5,
                            "partial overflow obj {obj:e} != want {want_d2:e} (rel={rel:e})"
                        );
                    }

                    // 3. Sub-case 3: Same-sign `2.0f * dot` overflow (`x = [0.93e19]`,
                    //    `c_0 = [1.84e19]` vs `c_{k-1} = [0.5e19]` across `k = 2` and `k = 1026`).
                    let x_same = pad(0.93e19f32);
                    let x_same_bf16 = f32_slice_to_bf16(&x_same);
                    let x_same_eff = effective_vec(&x_same, precision)[0];
                    let c_win_eff = effective_vec(&pad(0.5e19f32), precision)[0];
                    let want_same_d2 = f64::from((x_same_eff - c_win_eff) * (x_same_eff - c_win_eff));

                    for k in [2usize, 1_026usize] {
                        let mut cents = Vec::with_capacity(k * d);
                        for _ in 0..(k - 1) {
                            cents.extend_from_slice(&pad(1.84e19f32));
                        }
                        cents.extend_from_slice(&pad(0.5e19f32));

                        let (lbl_f32, obj_f32) = assign_gpu_f32(
                            rt, &x_same, 1, d as u32, &cents, k as u32, 64, precision, backend, false,
                        )
                        .expect("same-sign assign_gpu_f32");
                        assert_eq!(
                            lbl_f32,
                            vec![(k - 1) as u32],
                            "same-sign 2*dot overflow hijacked argmin ({backend:?}, {precision:?}, d={d}, k={k})"
                        );
                        assert!(
                            ((obj_f32 - want_same_d2).abs() / want_same_d2) <= 1.0e-5,
                            "same-sign f32 obj {obj_f32:e} != want {want_same_d2:e}"
                        );

                        if matches!(
                            precision,
                            EmeansGpuPrecision::Bf16 | EmeansGpuPrecision::Bf16AssignF32Stats
                        ) {
                            let (lbl_bf16, obj_bf16) = assign_gpu_bf16(
                                rt,
                                &x_same_bf16,
                                1,
                                d as u32,
                                &cents,
                                k as u32,
                                64,
                                precision,
                                backend,
                                false,
                            )
                            .expect("same-sign assign_gpu_bf16");
                            assert_eq!(lbl_bf16, vec![(k - 1) as u32]);
                            assert!(((obj_bf16 - want_same_d2).abs() / want_same_d2) <= 1.0e-5);
                        }
                    }
                }
            }
        }
    });
}
