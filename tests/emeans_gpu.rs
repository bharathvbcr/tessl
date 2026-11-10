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
