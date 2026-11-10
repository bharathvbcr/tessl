//! Metal 4 GPU substrate for exact-deterministic e-means clustering.
//!
//! Implements:
//! - Three explicit numerical precision lanes ([`EmeansGpuPrecision::Bf16`],
//!   [`EmeansGpuPrecision::Bf16AssignF32Stats`], and [`EmeansGpuPrecision::F32`]).
//! - `64x64` (`execution_simdgroups<4>`, 128 threads) cooperative TensorOps
//!   in-register argmin epilogues (`2 KiB` threadgroup SRAM) with `32x32`
//!   `simdgroup_float8x8` fallback and `split_c` cluster partitioning.
//! - Deterministic 8-bit `ulong` GPU radix sort (`(label << 16) | local_row`)
//!   plus `256`-row overflow slice gather and ascending slice fold (zero `f32`
//!   atomics).
//! - Pure-integer IEEE-754 `binary64` objective reduction, `weight == 1.0f`
//!   exact EMA override, and deterministic `SplitMix64` Gumbel split revival.

use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;

use crate::dispatch::dispatch_1d;
use crate::gemm::require_i32_extent;
use crate::nn::validate_buffer_byte_range;
use crate::runtime::{mtl_size, GpuRuntime};
use crate::tensor::GpuBuffer;

/// Canonical error prefix returned whenever a GPU staging, centroid-prep, or
/// norm reduction kernel detects `NaN`, `Inf`, `bf16` range overflow, or
/// squared-L2 `f32` overflow without poisoning [`GpuRuntime`].
pub const NONFINITE_ERR_PREFIX: &str = "emeans: non-finite";

/// Maximum chunk rows packed into `(ulong(label) << 16) | ulong(local_row)`
/// (`u16` local row index in `0..65_536`).
pub const MAX_GPU_CHUNK_ROWS: u32 = 65_536;

const FAULT_INPUT_NONFINITE: u32 = 1;
const FAULT_BF16_OVERFLOW: u32 = 2;
const FAULT_ROW_NORM_OVF: u32 = 4;
const FAULT_CENT_NORM_OVF: u32 = 8;

const MAX_OVERFLOW_SLICES: usize = 512;

/// Numerical precision lane for GPU e-means execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmeansGpuPrecision {
    /// Rounds rows once to `bf16` (`\hat{x}_i`); row norms, centroid norms,
    /// dot products, batch sums, and closing pass sums all read `\hat{x}_i`.
    Bf16,
    /// Uses `bf16`-rounded `\hat{x}_i` and `\hat{c}_j` for norms and argmin
    /// assignment while gathering unrounded `f32` `x_i` into batch and pass sums.
    Bf16AssignF32Stats,
    /// Unrounded `f32` operands, norms, TensorOps (`RELAXED = false`) /
    /// simdgroup dot products, and gather sums everywhere.
    F32,
}

impl EmeansGpuPrecision {
    #[inline]
    fn assign_is_bf16(self) -> bool {
        matches!(self, Self::Bf16 | Self::Bf16AssignF32Stats)
    }

    #[inline]
    fn stats_is_bf16(self) -> bool {
        matches!(self, Self::Bf16)
    }

    #[inline]
    fn needs_f32_rows(self) -> bool {
        matches!(self, Self::Bf16AssignF32Stats | Self::F32)
    }
}

/// Matrix engine backend selection for GPU e-means assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmeansGpuBackend {
    /// Selects `TensorOps` when the Metal 4 device supports it, else `Simdgroup`.
    Auto,
    /// Requires Metal 4 `64x64` `execution_simdgroups<4>` cooperative TensorOps;
    /// refuses cleanly on devices without TensorOps support.
    TensorOps,
    /// Portable `32x32` `simdgroup_float8x8` path.
    Simdgroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedBackend {
    TensorOps,
    Simdgroup,
}

/// Configuration for [`EmeansGpuRunner`].
#[derive(Debug, Clone)]
pub struct EmeansGpuConfig {
    pub k: u32,
    pub d: u32,
    pub max_chunk_rows: u32,
    pub precision: EmeansGpuPrecision,
    pub backend: EmeansGpuBackend,
    pub spherical: bool,
    pub unused_threshold: f32,
    pub split_epsilon: f32,
    pub init_usage: f32,
    pub seed: u64,
}

/// One mini-batch (or full-batch) step within [`EmeansGpuRunner::run_resident_epoch`].
#[derive(Debug, Clone, Copy)]
pub struct EpochBatchPlan {
    pub row_offset: u32,
    pub row_count: u32,
    pub alpha: f32,
    pub step: u64,
    pub check_revive: bool,
}

/// Pre-staged GPU-resident dataset (`X` + precomputed `row_norm2`) bound to a
/// specific [`GpuRuntime`].
pub struct ResidentDataset {
    runtime_addr: usize,
    n: u32,
    d: u32,
    precision: EmeansGpuPrecision,
    x_bf16: Option<GpuBuffer>,
    x_f32: Option<GpuBuffer>,
    row_norm2: GpuBuffer,
}

impl ResidentDataset {
    /// Number of rows staged in this resident dataset.
    pub fn rows(&self) -> u32 {
        self.n
    }

    /// Feature dimension of each staged row.
    pub fn cols(&self) -> u32 {
        self.d
    }
}

struct EmeansPipelines {
    zero_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    prepare_centroids_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    prepare_centroids_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    stage_rows_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    stage_rows_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    gather_rows_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    gather_rows_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    assign_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    assign_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    reduce_split_c: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pack_label_keys: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    radix_hist: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    radix_prefix: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    radix_scatter: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    find_bucket_bounds: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    gather_slice_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    gather_slice_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    fold_slice_partials: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    accum_chunk_obj: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    apply_ema: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    revive_prepare: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    revive_splits: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl EmeansPipelines {
    fn resolve(rt: &GpuRuntime, backend: ResolvedBackend) -> Result<Self, String> {
        let (assign_bf16_name, assign_f32_name) = match backend {
            ResolvedBackend::TensorOps => ("emeans_assign_bf16_tensorops", "emeans_assign_f32_tensorops"),
            ResolvedBackend::Simdgroup => ("emeans_assign_bf16_simdgroup", "emeans_assign_f32_simdgroup"),
        };
        Ok(Self {
            zero_f32: rt.pipeline("zero_f32")?,
            prepare_centroids_bf16: rt.pipeline("emeans_prepare_centroids_bf16")?,
            prepare_centroids_f32: rt.pipeline("emeans_prepare_centroids_f32")?,
            stage_rows_bf16: rt.pipeline("emeans_stage_rows_bf16")?,
            stage_rows_f32: rt.pipeline("emeans_stage_rows_f32")?,
            gather_rows_bf16: rt.pipeline("emeans_gather_rows_bf16")?,
            gather_rows_f32: rt.pipeline("emeans_gather_rows_f32")?,
            assign_bf16: rt.pipeline(assign_bf16_name)?,
            assign_f32: rt.pipeline(assign_f32_name)?,
            reduce_split_c: rt.pipeline("emeans_reduce_split_c")?,
            pack_label_keys: rt.pipeline("emeans_pack_label_keys")?,
            radix_hist: rt.pipeline("emeans_radix_hist")?,
            radix_prefix: rt.pipeline("emeans_radix_prefix")?,
            radix_scatter: rt.pipeline("emeans_radix_scatter")?,
            find_bucket_bounds: rt.pipeline("emeans_find_bucket_bounds")?,
            gather_slice_bf16: rt.pipeline("emeans_gather_slice_partials_bf16")?,
            gather_slice_f32: rt.pipeline("emeans_gather_slice_partials_f32")?,
            fold_slice_partials: rt.pipeline("emeans_fold_slice_partials")?,
            accum_chunk_obj: rt.pipeline("emeans_accum_chunk_obj")?,
            apply_ema: rt.pipeline("emeans_apply_ema")?,
            revive_prepare: rt.pipeline("emeans_revive_prepare")?,
            revive_splits: rt.pipeline("emeans_revive_splits")?,
        })
    }
}

/// Stateful Metal 4 runner owning GPU-resident centroids, EMA usage, CSR
/// radix-sort workspaces, and closing-pass accumulators.
pub struct EmeansGpuRunner {
    rt: Arc<GpuRuntime>,
    cfg: EmeansGpuConfig,
    backend: ResolvedBackend,
    pipes: EmeansPipelines,
    chunk_cap: u32,
    part_k: u32,
    split_c: u32,

    dummy_buf: GpuBuffer,
    fault_flags: GpuBuffer,

    centroids: GpuBuffer,
    usage: GpuBuffer,
    c_eff_bf16: Option<GpuBuffer>,
    c_eff_f32: Option<GpuBuffer>,
    norm2_c: GpuBuffer,

    chunk_x_bf16: Option<GpuBuffer>,
    chunk_x_f32: Option<GpuBuffer>,
    chunk_norm2: GpuBuffer,
    chunk_labels: GpuBuffer,
    chunk_dists: GpuBuffer,
    part_scores: Option<GpuBuffer>,
    part_labels: Option<GpuBuffer>,

    keys_a: GpuBuffer,
    keys_b: GpuBuffer,
    radix_hist: GpuBuffer,
    overflow_slice_count: GpuBuffer,
    bucket_info: GpuBuffer,
    slice_meta: GpuBuffer,
    slice_partials: GpuBuffer,

    batch_sums: GpuBuffer,
    batch_counts: GpuBuffer,
    epoch_counts: GpuBuffer,
    pass_sums: GpuBuffer,
    pass_counts: GpuBuffer,
    epoch_obj_bits: GpuBuffer,
    unused_ids: GpuBuffer,
    revive_meta: GpuBuffer,

    streaming_final_epoch: bool,
    in_flight_staging: Vec<GpuBuffer>,
}

#[inline]
fn splitmix64_mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Deterministic `SplitMix64` Gumbel donor ranking score matching
/// `emeans_revive_prepare` in `kernels/emeans.metal`:
/// `g_j = ln(usage) - ln(-ln(u_j))` where `u_j = (((h >> 41) << 1) | 1) * 2^-24`.
pub fn revival_gumbel_key(usage: f32, seed: u64, step: u64, cluster_id: u32) -> f32 {
    let s0 = splitmix64_mix(seed ^ 0xd1b5_4a32_d192_ed03);
    let s1 = splitmix64_mix(s0 ^ step.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let h = splitmix64_mix(s1 ^ u64::from(cluster_id).wrapping_mul(0xbf58_476d_1ce4_e5b9));
    let odd24 = (((h >> 41) as u32) << 1) | 1;
    let u = (odd24 as f32) * f32::from_bits(0x3380_0000);
    usage.ln() - (-u.ln()).ln()
}

fn resolve_backend(rt: &GpuRuntime, requested: EmeansGpuBackend) -> Result<ResolvedBackend, String> {
    match requested {
        EmeansGpuBackend::Auto => {
            if rt.has_tensorops() {
                Ok(ResolvedBackend::TensorOps)
            } else {
                Ok(ResolvedBackend::Simdgroup)
            }
        }
        EmeansGpuBackend::TensorOps => {
            if rt.has_tensorops() {
                Ok(ResolvedBackend::TensorOps)
            } else {
                Err("emeans: TensorOps backend requested, but this Metal device does not support TensorOps".to_string())
            }
        }
        EmeansGpuBackend::Simdgroup => Ok(ResolvedBackend::Simdgroup),
    }
}

fn compute_partition_plan(k: u32, d: u32, chunk_cap: u32) -> Result<(u32, u32), String> {
    if k == 0 || d == 0 {
        return Err(format!("emeans: k ({k}) and d ({d}) must both be >= 1"));
    }
    let d_usize = d as usize;
    require_i32_extent(d_usize, "emeans feature dimension")?;
    let chunk_elems = (chunk_cap as usize)
        .checked_mul(d_usize)
        .ok_or_else(|| "emeans: chunk_rows * d overflows usize".to_string())?;
    require_i32_extent(chunk_elems, "emeans chunk matrix")?;

    let max_by_i32 = (i32::MAX as usize) / d_usize;
    if max_by_i32 == 0 {
        return Err(format!("emeans: feature dimension {d} exceeds TensorOps i32 bound"));
    }
    let target_cap = if k <= 8_192 { 1_024usize } else { 4_096usize };
    let aligned_cap = if max_by_i32 < 64 {
        max_by_i32
    } else {
        max_by_i32.min(target_cap) & !63usize
    };
    let part_k = if (k as usize) <= aligned_cap {
        k
    } else {
        aligned_cap as u32
    };
    let part_elems = (part_k as usize)
        .checked_mul(d_usize)
        .ok_or_else(|| "emeans: partition elements overflow usize".to_string())?;
    require_i32_extent(part_elems, "emeans cluster partition")?;
    let split_c = k.div_ceil(part_k);
    Ok((part_k, split_c))
}

fn validate_config(cfg: &EmeansGpuConfig, init_centroids_len: usize) -> Result<usize, String> {
    if cfg.k == 0 || cfg.d == 0 {
        return Err(format!("emeans: k ({}) and d ({}) must both be >= 1", cfg.k, cfg.d));
    }
    let kd = (cfg.k as usize)
        .checked_mul(cfg.d as usize)
        .ok_or_else(|| "emeans: k * d overflows usize".to_string())?;
    if kd > u32::MAX as usize {
        return Err(format!("emeans: k * d ({kd}) exceeds Metal uint indexing"));
    }
    if init_centroids_len != kd {
        return Err(format!(
            "emeans: init_centroids length {init_centroids_len} != k * d ({kd})"
        ));
    }
    if !cfg.unused_threshold.is_finite() || cfg.unused_threshold < 0.0 {
        return Err(format!(
            "emeans: unused_threshold must be finite and >= 0, got {}",
            cfg.unused_threshold
        ));
    }
    if !cfg.split_epsilon.is_finite() || cfg.split_epsilon < 0.0 {
        return Err(format!(
            "emeans: split_epsilon must be finite and >= 0, got {}",
            cfg.split_epsilon
        ));
    }
    if !cfg.init_usage.is_finite() || cfg.init_usage < 0.0 {
        return Err(format!(
            "emeans: init_usage must be finite and >= 0, got {}",
            cfg.init_usage
        ));
    }
    Ok(kd)
}

fn prescale_normalize_rows_f64(src: &[f32], rows: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * d];
    for r in 0..rows {
        let row_in = &src[r * d..(r + 1) * d];
        let row_out = &mut out[r * d..(r + 1) * d];
        let mut max_abs = 0.0f64;
        for &v in row_in {
            let a = (v as f64).abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        if max_abs > 0.0 {
            let mut sum_sq = 0.0f64;
            for &v in row_in {
                let w = (v as f64) / max_abs;
                sum_sq += w * w;
            }
            let norm = sum_sq.sqrt();
            if norm > 0.0 {
                for (dst, &v) in row_out.iter_mut().zip(row_in.iter()) {
                    *dst = (((v as f64) / max_abs) / norm) as f32;
                }
            }
        }
    }
    out
}

fn format_fault_error(flags: u32) -> String {
    if (flags & FAULT_INPUT_NONFINITE) != 0 {
        format!("{NONFINITE_ERR_PREFIX}: input row or centroid contains NaN or Inf (fault={flags:#x})")
    } else if (flags & FAULT_BF16_OVERFLOW) != 0 {
        format!("{NONFINITE_ERR_PREFIX}: operand overflows bf16 finite range (fault={flags:#x})")
    } else if (flags & FAULT_ROW_NORM_OVF) != 0 {
        format!("{NONFINITE_ERR_PREFIX}: squared row L2 norm overflows f32 (fault={flags:#x})")
    } else if (flags & FAULT_CENT_NORM_OVF) != 0 {
        format!("{NONFINITE_ERR_PREFIX}: squared centroid L2 norm overflows f32 (fault={flags:#x})")
    } else {
        format!("{NONFINITE_ERR_PREFIX}: GPU fault flag raised (fault={flags:#x})")
    }
}

fn stage_f32_unwaited(rt: &GpuRuntime, data: &[f32]) -> Result<GpuBuffer, String> {
    let words: Vec<u32> = data.iter().map(|&v| v.to_bits()).collect();
    rt.alloc_buffer_from_u32(&words)
}

fn stage_u16_unwaited(rt: &GpuRuntime, data: &[u16]) -> Result<GpuBuffer, String> {
    let mut words = Vec::with_capacity(data.len().div_ceil(2).max(1));
    let (pairs, rem) = data.as_chunks::<2>();
    for &[lo, hi] in pairs {
        words.push(u32::from(lo) | (u32::from(hi) << 16));
    }
    if let &[tail] = rem {
        words.push(u32::from(tail));
    }
    rt.alloc_buffer_from_u32(&words)
}

fn try_read_u64_vec(buf: &GpuBuffer, count: usize) -> Result<Vec<u64>, String> {
    let bytes = buf.try_contents_u8()?;
    let need = count
        .checked_mul(8)
        .ok_or_else(|| "emeans: u64 read byte count overflow".to_string())?;
    if bytes.len() < need {
        return Err(format!("emeans: u64 read buffer too short ({} < {need})", bytes.len()));
    }
    let mut out = Vec::with_capacity(count);
    for &chunk in bytes[..need].as_chunks::<8>().0 {
        out.push(u64::from_ne_bytes(chunk));
    }
    Ok(out)
}

fn csr_radix_shifts(k: u32) -> &'static [u32] {
    if k <= 1 {
        &[]
    } else if k <= 256 {
        &[16]
    } else if k <= 65_536 {
        &[16, 24]
    } else if k <= 16_777_216 {
        &[16, 24, 32]
    } else {
        &[16, 24, 32, 40]
    }
}

fn encode_zero_buf(
    rt: &GpuRuntime,
    zero_pipe: &ProtocolObject<dyn MTLComputePipelineState>,
    buf: &GpuBuffer,
) -> Result<(), String> {
    let nwords = buf.nbytes() / 4;
    if nwords == 0 {
        return Ok(());
    }
    let n32 = u32::try_from(nwords).map_err(|_| format!("emeans: zero_f32 word count {nwords} exceeds u32"))?;
    dispatch_1d(rt, zero_pipe, nwords, |bnd| {
        bnd.bind_gpu_buf(buf, 0);
        bnd.bind_u32(n32, 1);
    })
}

#[allow(clippy::too_many_arguments)]
fn encode_prepare_centroids(
    rt: &GpuRuntime,
    pipes: &EmeansPipelines,
    precision: EmeansGpuPrecision,
    spherical: bool,
    k: u32,
    d: u32,
    centroids: &GpuBuffer,
    c_eff_bf16: Option<&GpuBuffer>,
    c_eff_f32: Option<&GpuBuffer>,
    norm2_c: &GpuBuffer,
    fault_flags: &GpuBuffer,
) -> Result<(), String> {
    let sph = u32::from(spherical);
    rt.with_binder(|bnd| {
        if precision.assign_is_bf16() {
            let dst = c_eff_bf16.ok_or_else(|| "emeans: missing bf16 centroid buffer".to_string())?;
            bnd.set_pipeline(&pipes.prepare_centroids_bf16);
            bnd.bind_gpu_buf(centroids, 0);
            bnd.bind_gpu_buf(dst, 1);
        } else {
            let dst = c_eff_f32.ok_or_else(|| "emeans: missing f32 centroid buffer".to_string())?;
            bnd.set_pipeline(&pipes.prepare_centroids_f32);
            bnd.bind_gpu_buf(centroids, 0);
            bnd.bind_gpu_buf(dst, 1);
        }
        bnd.bind_gpu_buf(norm2_c, 2);
        bnd.bind_gpu_buf(fault_flags, 3);
        bnd.bind_u32(k, 4);
        bnd.bind_u32(d, 5);
        bnd.bind_u32(sph, 6);
        bnd.dispatch(mtl_size(k as usize, 1, 1), mtl_size(256, 1, 1));
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
fn encode_stage_rows_f32(
    rt: &GpuRuntime,
    pipes: &EmeansPipelines,
    src_f32: &GpuBuffer,
    dst_bf16: &GpuBuffer,
    dst_f32: &GpuBuffer,
    dst_norm2: &GpuBuffer,
    fault_flags: &GpuBuffer,
    n: u32,
    d: u32,
    precision: EmeansGpuPrecision,
) -> Result<(), String> {
    if n == 0 {
        return Ok(());
    }
    let write_bf16 = u32::from(precision.assign_is_bf16());
    let write_f32 = u32::from(precision.needs_f32_rows());
    let norm_from_bf16 = u32::from(precision.assign_is_bf16());
    // Stage in windows of at most 65_536 threadgroups if n is huge, or 1D grid of n threadgroups
    // (Metal 4 supports up to u32::MAX threadgroups in X, validated by Binder::dispatch).
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&pipes.stage_rows_f32);
        bnd.bind_gpu_buf(src_f32, 0);
        bnd.bind_gpu_buf(dst_bf16, 1);
        bnd.bind_gpu_buf(dst_f32, 2);
        bnd.bind_gpu_buf(dst_norm2, 3);
        bnd.bind_gpu_buf(fault_flags, 4);
        bnd.bind_u32(n, 5);
        bnd.bind_u32(d, 6);
        bnd.bind_u32(write_bf16, 7);
        bnd.bind_u32(write_f32, 8);
        bnd.bind_u32(norm_from_bf16, 9);
        bnd.dispatch(mtl_size(n as usize, 1, 1), mtl_size(256, 1, 1));
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
fn encode_stage_rows_bf16(
    rt: &GpuRuntime,
    pipes: &EmeansPipelines,
    src_bf16: &GpuBuffer,
    dst_bf16: &GpuBuffer,
    dst_f32: &GpuBuffer,
    dst_norm2: &GpuBuffer,
    fault_flags: &GpuBuffer,
    n: u32,
    d: u32,
    precision: EmeansGpuPrecision,
) -> Result<(), String> {
    if n == 0 {
        return Ok(());
    }
    let write_bf16 = u32::from(precision.assign_is_bf16());
    let write_f32 = u32::from(precision.needs_f32_rows());
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&pipes.stage_rows_bf16);
        bnd.bind_gpu_buf(src_bf16, 0);
        bnd.bind_gpu_buf(dst_bf16, 1);
        bnd.bind_gpu_buf(dst_f32, 2);
        bnd.bind_gpu_buf(dst_norm2, 3);
        bnd.bind_gpu_buf(fault_flags, 4);
        bnd.bind_u32(n, 5);
        bnd.bind_u32(d, 6);
        bnd.bind_u32(write_bf16, 7);
        bnd.bind_u32(write_f32, 8);
        bnd.dispatch(mtl_size(n as usize, 1, 1), mtl_size(256, 1, 1));
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
fn encode_assign_chunk(
    rt: &GpuRuntime,
    pipes: &EmeansPipelines,
    backend: ResolvedBackend,
    precision: EmeansGpuPrecision,
    spherical: bool,
    m: u32,
    k: u32,
    d: u32,
    part_k: u32,
    split_c: u32,
    x_assign: &GpuBuffer,
    c_eff: &GpuBuffer,
    row_norm2: &GpuBuffer,
    norm2_c: &GpuBuffer,
    chunk_labels: &GpuBuffer,
    chunk_dists: &GpuBuffer,
    epoch_labels: &GpuBuffer,
    has_epoch_labels: bool,
    epoch_off: u32,
    part_scores: &GpuBuffer,
    part_labels: &GpuBuffer,
) -> Result<(), String> {
    if m == 0 {
        return Ok(());
    }
    let x_elems = (m as usize)
        .checked_mul(d as usize)
        .ok_or_else(|| "emeans: chunk M * d overflows usize".to_string())?;
    require_i32_extent(x_elems, "emeans assign X")?;

    let tile_m = match backend {
        ResolvedBackend::TensorOps => 64usize,
        ResolvedBackend::Simdgroup => 32usize,
    };
    let groups_x = (m as usize).div_ceil(tile_m);
    let sph = u32::from(spherical);
    let has_ep = u32::from(has_epoch_labels);
    let assign_pipe = if precision.assign_is_bf16() {
        &pipes.assign_bf16
    } else {
        &pipes.assign_f32
    };

    for part_c in 0..split_c {
        let c_start = part_c * part_k;
        let c_end = (c_start + part_k).min(k);
        let c_len = c_end - c_start;
        let c_elems = (c_len as usize)
            .checked_mul(d as usize)
            .ok_or_else(|| "emeans: cluster partition c_len * d overflows usize".to_string())?;
        require_i32_extent(c_elems, "emeans assign C")?;

        rt.with_binder(|bnd| {
            bnd.set_pipeline(assign_pipe);
            bnd.bind_gpu_buf(x_assign, 0);
            bnd.bind_gpu_buf(c_eff, 1);
            bnd.bind_gpu_buf(row_norm2, 2);
            bnd.bind_gpu_buf(norm2_c, 3);
            bnd.bind_gpu_buf(chunk_labels, 4);
            bnd.bind_gpu_buf(chunk_dists, 5);
            bnd.bind_gpu_buf(epoch_labels, 6);
            bnd.bind_gpu_buf(part_scores, 7);
            bnd.bind_gpu_buf(part_labels, 8);
            bnd.bind_u32(m, 9);
            bnd.bind_u32(d, 10);
            bnd.bind_u32(c_start, 11);
            bnd.bind_u32(c_end, 12);
            bnd.bind_u32(part_c, 13);
            bnd.bind_u32(split_c, 14);
            bnd.bind_u32(sph, 15);
            bnd.bind_u32(has_ep, 16);
            bnd.bind_u32(epoch_off, 17);
            bnd.dispatch(mtl_size(groups_x, 1, 1), mtl_size(128, 1, 1));
            Ok(())
        })?;
    }

    if split_c > 1 {
        let use_bf16 = u32::from(precision.assign_is_bf16());
        dispatch_1d(rt, &pipes.reduce_split_c, m as usize, |bnd| {
            bnd.bind_gpu_buf(part_scores, 0);
            bnd.bind_gpu_buf(part_labels, 1);
            bnd.bind_gpu_buf(row_norm2, 2);
            bnd.bind_gpu_buf(x_assign, 3);
            bnd.bind_gpu_buf(c_eff, 4);
            bnd.bind_gpu_buf(chunk_labels, 5);
            bnd.bind_gpu_buf(chunk_dists, 6);
            bnd.bind_gpu_buf(epoch_labels, 7);
            bnd.bind_u32(m, 8);
            bnd.bind_u32(d, 9);
            bnd.bind_u32(split_c, 10);
            bnd.bind_u32(sph, 11);
            bnd.bind_u32(use_bf16, 12);
            bnd.bind_u32(has_ep, 13);
            bnd.bind_u32(epoch_off, 14);
        })?;
    }
    Ok(())
}

fn encode_radix_passes<'a>(
    rt: &GpuRuntime,
    pipes: &EmeansPipelines,
    keys_a: &'a GpuBuffer,
    keys_b: &'a GpuBuffer,
    radix_hist: &GpuBuffer,
    count: u32,
    shifts: &[u32],
) -> Result<&'a GpuBuffer, String> {
    if count == 0 || shifts.is_empty() {
        return Ok(keys_a);
    }
    let num_tiles = count.div_ceil(256).max(1);
    let mut cur_in = keys_a;
    let mut cur_out = keys_b;
    for &shift in shifts {
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipes.radix_hist);
            bnd.bind_gpu_buf(cur_in, 0);
            bnd.bind_gpu_buf(radix_hist, 1);
            bnd.bind_u32(count, 2);
            bnd.bind_u32(num_tiles, 3);
            bnd.bind_u32(shift, 4);
            bnd.dispatch(mtl_size(num_tiles as usize, 1, 1), mtl_size(256, 1, 1));
            Ok(())
        })?;
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipes.radix_prefix);
            bnd.bind_gpu_buf(radix_hist, 0);
            bnd.bind_u32(num_tiles, 1);
            bnd.dispatch(mtl_size(1, 1, 1), mtl_size(256, 1, 1));
            Ok(())
        })?;
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipes.radix_scatter);
            bnd.bind_gpu_buf(cur_in, 0);
            bnd.bind_gpu_buf(radix_hist, 1);
            bnd.bind_gpu_buf(cur_out, 2);
            bnd.bind_u32(count, 3);
            bnd.bind_u32(num_tiles, 4);
            bnd.bind_u32(shift, 5);
            bnd.dispatch(mtl_size(num_tiles as usize, 1, 1), mtl_size(256, 1, 1));
            Ok(())
        })?;
        std::mem::swap(&mut cur_in, &mut cur_out);
    }
    Ok(cur_in)
}

impl EmeansGpuRunner {
    /// Allocate GPU workspaces and upload initial centroids and `init_usage` on `rt`.
    pub fn new(rt: &Arc<GpuRuntime>, cfg: EmeansGpuConfig, init_centroids: &[f32]) -> Result<Self, String> {
        let kd = validate_config(&cfg, init_centroids.len())?;
        if init_centroids.iter().any(|v| !v.is_finite()) {
            return Err(format!("{NONFINITE_ERR_PREFIX}: initial centroids contain NaN or Inf"));
        }
        let backend = resolve_backend(rt, cfg.backend)?;
        let chunk_cap = cfg.max_chunk_rows.clamp(1, MAX_GPU_CHUNK_ROWS);
        let (part_k, split_c) = compute_partition_plan(cfg.k, cfg.d, chunk_cap)?;
        let pipes = EmeansPipelines::resolve(rt, backend)?;

        let k_usize = cfg.k as usize;
        let d_usize = cfg.d as usize;
        let chunk_usize = chunk_cap as usize;
        let chunk_elems = chunk_usize * d_usize;

        let dummy_buf = rt.alloc_buffer_hot(16)?;
        let fault_flags = rt.alloc_buffer_hot(4)?;
        fault_flags.try_write_u32(&[0])?;

        let centroids = rt.alloc_buffer_hot(kd * 4)?;
        if cfg.spherical {
            let norm_init = prescale_normalize_rows_f64(init_centroids, k_usize, d_usize);
            centroids.try_write_f32(&norm_init)?;
        } else {
            centroids.try_write_f32(init_centroids)?;
        }

        let usage = rt.alloc_buffer_hot(k_usize * 4)?;
        let init_u = vec![cfg.init_usage; k_usize];
        usage.try_write_f32(&init_u)?;

        let c_eff_bf16 = if cfg.precision.assign_is_bf16() {
            Some(rt.alloc_buffer_hot((kd * 2).max(4))?)
        } else {
            None
        };
        let c_eff_f32 = if !cfg.precision.assign_is_bf16() {
            Some(rt.alloc_buffer_hot(kd * 4)?)
        } else {
            None
        };
        let norm2_c = rt.alloc_buffer_hot(k_usize * 4)?;

        let chunk_x_bf16 = if cfg.precision.assign_is_bf16() {
            Some(rt.alloc_buffer_hot((chunk_elems * 2).max(4))?)
        } else {
            None
        };
        let chunk_x_f32 = if cfg.precision.needs_f32_rows() {
            Some(rt.alloc_buffer_hot(chunk_elems * 4)?)
        } else {
            None
        };
        let chunk_norm2 = rt.alloc_buffer_hot(chunk_usize * 4)?;
        let chunk_labels = rt.alloc_buffer_hot(chunk_usize * 4)?;
        let chunk_dists = rt.alloc_buffer_hot(chunk_usize * 4)?;

        let (part_scores, part_labels) = if split_c > 1 {
            let p_elems = (split_c as usize)
                .checked_mul(chunk_usize)
                .ok_or_else(|| "emeans: split_c * chunk_cap overflows usize".to_string())?;
            (
                Some(rt.alloc_buffer_hot(p_elems * 4)?),
                Some(rt.alloc_buffer_hot(p_elems * 4)?),
            )
        } else {
            (None, None)
        };

        let radix_cap = if cfg.unused_threshold > 0.0 {
            chunk_usize.max(k_usize)
        } else {
            chunk_usize
        };
        let radix_tiles_max = radix_cap.div_ceil(256).max(1);
        let keys_a = rt.alloc_buffer_hot((radix_cap * 8).max(8))?;
        let keys_b = rt.alloc_buffer_hot((radix_cap * 8).max(8))?;
        let radix_hist = rt.alloc_buffer_hot(256 * radix_tiles_max * 4)?;

        let overflow_slice_count = rt.alloc_buffer_hot(4)?;
        let bucket_info = rt.alloc_buffer_hot(k_usize * 16)?;
        let slice_meta = rt.alloc_buffer_hot(MAX_OVERFLOW_SLICES * 8)?;
        let slice_partials = rt.alloc_buffer_hot(MAX_OVERFLOW_SLICES * d_usize * 4)?;

        let batch_sums = rt.alloc_buffer_hot(kd * 4)?;
        batch_sums.try_zero()?;
        let batch_counts = rt.alloc_buffer_hot(k_usize * 8)?;
        batch_counts.try_zero()?;
        let epoch_counts = rt.alloc_buffer_hot(k_usize * 8)?;
        epoch_counts.try_zero()?;
        let pass_sums = rt.alloc_buffer_hot(kd * 4)?;
        pass_sums.try_zero()?;
        let pass_counts = rt.alloc_buffer_hot(k_usize * 8)?;
        pass_counts.try_zero()?;
        let epoch_obj_bits = rt.alloc_buffer_hot(8)?;
        epoch_obj_bits.try_zero()?;
        let unused_ids = rt.alloc_buffer_hot(k_usize * 4)?;
        let revive_meta = rt.alloc_buffer_hot(8)?;
        revive_meta.try_zero()?;

        let runner = Self {
            rt: Arc::clone(rt),
            cfg,
            backend,
            pipes,
            chunk_cap,
            part_k,
            split_c,
            dummy_buf,
            fault_flags,
            centroids,
            usage,
            c_eff_bf16,
            c_eff_f32,
            norm2_c,
            chunk_x_bf16,
            chunk_x_f32,
            chunk_norm2,
            chunk_labels,
            chunk_dists,
            part_scores,
            part_labels,
            keys_a,
            keys_b,
            radix_hist,
            overflow_slice_count,
            bucket_info,
            slice_meta,
            slice_partials,
            batch_sums,
            batch_counts,
            epoch_counts,
            pass_sums,
            pass_counts,
            epoch_obj_bits,
            unused_ids,
            revive_meta,
            streaming_final_epoch: false,
            in_flight_staging: Vec::new(),
        };

        // Pre-compute effective centroids and validate norm/bf16 bounds immediately.
        {
            let _scope = runner.rt.async_scope();
            runner.encode_prepare_centroids_self()?;
            runner.rt.synchronize()?;
        }
        runner.check_and_clear_faults()?;
        Ok(runner)
    }

    fn encode_prepare_centroids_self(&self) -> Result<(), String> {
        encode_prepare_centroids(
            &self.rt,
            &self.pipes,
            self.cfg.precision,
            self.cfg.spherical,
            self.cfg.k,
            self.cfg.d,
            &self.centroids,
            self.c_eff_bf16.as_ref(),
            self.c_eff_f32.as_ref(),
            &self.norm2_c,
            &self.fault_flags,
        )
    }

    fn check_and_clear_faults(&self) -> Result<(), String> {
        let flags = self.fault_flags.try_read_u32()?[0];
        if flags != 0 {
            self.fault_flags.try_write_u32(&[0])?;
            return Err(format_fault_error(flags));
        }
        Ok(())
    }

    /// Stage an `f32` host dataset `[n, d]` into GPU-resident storage (`X` + `row_norm2`).
    pub fn upload_dataset_f32(&self, rows_f32: &[f32], n: u32) -> Result<ResidentDataset, String> {
        let d_usize = self.cfg.d as usize;
        let elems = (n as usize)
            .checked_mul(d_usize)
            .ok_or_else(|| "emeans::upload_dataset_f32: n * d overflows usize".to_string())?;
        if rows_f32.len() != elems {
            return Err(format!(
                "emeans::upload_dataset_f32: slice len {} != n * d ({elems})",
                rows_f32.len()
            ));
        }
        if n == 0 {
            return self.empty_resident_dataset();
        }
        let staged = stage_f32_unwaited(&self.rt, rows_f32)?;
        self.upload_dataset_f32_buffer(&staged, n)
    }

    /// Stage a `bf16`-bits host dataset `[n, d]` into GPU-resident storage (`X` + `row_norm2`).
    pub fn upload_dataset_bf16(&self, rows_bf16: &[u16], n: u32) -> Result<ResidentDataset, String> {
        let d_usize = self.cfg.d as usize;
        let elems = (n as usize)
            .checked_mul(d_usize)
            .ok_or_else(|| "emeans::upload_dataset_bf16: n * d overflows usize".to_string())?;
        if rows_bf16.len() != elems {
            return Err(format!(
                "emeans::upload_dataset_bf16: slice len {} != n * d ({elems})",
                rows_bf16.len()
            ));
        }
        if n == 0 {
            return self.empty_resident_dataset();
        }
        let staged = stage_u16_unwaited(&self.rt, rows_bf16)?;
        self.upload_dataset_bf16_buffer(&staged, n)
    }

    /// Stage an already-resident `f32` [`GpuBuffer`] `[n, d]` on `self.rt` into a
    /// [`ResidentDataset`], validating runtime ownership before opening any `Binder`.
    pub fn upload_dataset_f32_buffer(&self, rows_buf: &GpuBuffer, n: u32) -> Result<ResidentDataset, String> {
        let elems = (n as usize)
            .checked_mul(self.cfg.d as usize)
            .ok_or_else(|| "emeans::upload_dataset_f32_buffer: n * d overflows usize".to_string())?;
        let byte_len = elems
            .checked_mul(4)
            .ok_or_else(|| "emeans::upload_dataset_f32_buffer: byte length overflows usize".to_string())?;
        validate_buffer_byte_range(&self.rt, rows_buf, 0, byte_len, "emeans::upload_dataset_f32_buffer")?;
        if n == 0 {
            return self.empty_resident_dataset();
        }

        let x_bf16 = if self.cfg.precision.assign_is_bf16() {
            Some(self.rt.alloc_buffer_hot((elems * 2).max(4))?)
        } else {
            None
        };
        let x_f32 = if self.cfg.precision.needs_f32_rows() {
            Some(self.rt.alloc_buffer_hot(elems * 4)?)
        } else {
            None
        };
        let row_norm2 = self.rt.alloc_buffer_hot((n as usize) * 4)?;

        {
            let _scope = self.rt.async_scope();
            encode_zero_buf(&self.rt, &self.pipes.zero_f32, &self.fault_flags)?;
            encode_stage_rows_f32(
                &self.rt,
                &self.pipes,
                rows_buf,
                x_bf16.as_ref().unwrap_or(&self.dummy_buf),
                x_f32.as_ref().unwrap_or(&self.dummy_buf),
                &row_norm2,
                &self.fault_flags,
                n,
                self.cfg.d,
                self.cfg.precision,
            )?;
            self.rt.synchronize()?;
        }
        self.check_and_clear_faults()?;

        Ok(ResidentDataset {
            runtime_addr: Arc::as_ptr(&self.rt) as usize,
            n,
            d: self.cfg.d,
            precision: self.cfg.precision,
            x_bf16,
            x_f32,
            row_norm2,
        })
    }

    /// Stage an already-resident `bf16` [`GpuBuffer`] `[n, d]` on `self.rt` into a
    /// [`ResidentDataset`], validating runtime ownership before opening any `Binder`.
    pub fn upload_dataset_bf16_buffer(&self, rows_buf: &GpuBuffer, n: u32) -> Result<ResidentDataset, String> {
        let elems = (n as usize)
            .checked_mul(self.cfg.d as usize)
            .ok_or_else(|| "emeans::upload_dataset_bf16_buffer: n * d overflows usize".to_string())?;
        let byte_len = elems
            .checked_mul(2)
            .ok_or_else(|| "emeans::upload_dataset_bf16_buffer: byte length overflows usize".to_string())?;
        validate_buffer_byte_range(&self.rt, rows_buf, 0, byte_len, "emeans::upload_dataset_bf16_buffer")?;
        if n == 0 {
            return self.empty_resident_dataset();
        }

        let x_bf16 = if self.cfg.precision.assign_is_bf16() {
            Some(self.rt.alloc_buffer_hot((elems * 2).max(4))?)
        } else {
            None
        };
        let x_f32 = if self.cfg.precision.needs_f32_rows() {
            Some(self.rt.alloc_buffer_hot(elems * 4)?)
        } else {
            None
        };
        let row_norm2 = self.rt.alloc_buffer_hot((n as usize) * 4)?;

        {
            let _scope = self.rt.async_scope();
            encode_zero_buf(&self.rt, &self.pipes.zero_f32, &self.fault_flags)?;
            encode_stage_rows_bf16(
                &self.rt,
                &self.pipes,
                rows_buf,
                x_bf16.as_ref().unwrap_or(&self.dummy_buf),
                x_f32.as_ref().unwrap_or(&self.dummy_buf),
                &row_norm2,
                &self.fault_flags,
                n,
                self.cfg.d,
                self.cfg.precision,
            )?;
            self.rt.synchronize()?;
        }
        self.check_and_clear_faults()?;

        Ok(ResidentDataset {
            runtime_addr: Arc::as_ptr(&self.rt) as usize,
            n,
            d: self.cfg.d,
            precision: self.cfg.precision,
            x_bf16,
            x_f32,
            row_norm2,
        })
    }

    fn empty_resident_dataset(&self) -> Result<ResidentDataset, String> {
        Ok(ResidentDataset {
            runtime_addr: Arc::as_ptr(&self.rt) as usize,
            n: 0,
            d: self.cfg.d,
            precision: self.cfg.precision,
            x_bf16: None,
            x_f32: None,
            row_norm2: self.rt.alloc_buffer(16)?,
        })
    }

    fn encode_gather_resident_chunk(
        &self,
        ds: &ResidentDataset,
        order_buf: &GpuBuffer,
        row_offset: u32,
        chunk_rows: u32,
    ) -> Result<(), String> {
        if chunk_rows == 0 {
            return Ok(());
        }
        let groups_x = (self.cfg.d as usize).div_ceil(16);
        let groups_y = (chunk_rows as usize).div_ceil(16);
        let mut copied_norm2 = false;

        if self.cfg.precision.assign_is_bf16() {
            let src = ds
                .x_bf16
                .as_ref()
                .ok_or_else(|| "emeans: ResidentDataset missing bf16 storage".to_string())?;
            let dst = self
                .chunk_x_bf16
                .as_ref()
                .ok_or_else(|| "emeans: runner missing chunk_x_bf16".to_string())?;
            self.rt.with_binder(|bnd| {
                bnd.set_pipeline(&self.pipes.gather_rows_bf16);
                bnd.bind_gpu_buf(src, 0);
                bnd.bind_gpu_buf(&ds.row_norm2, 1);
                bnd.bind_gpu_buf(order_buf, 2);
                bnd.bind_gpu_buf(dst, 3);
                bnd.bind_gpu_buf(&self.chunk_norm2, 4);
                bnd.bind_u32(ds.n, 5);
                bnd.bind_u32(row_offset, 6);
                bnd.bind_u32(chunk_rows, 7);
                bnd.bind_u32(self.cfg.d, 8);
                bnd.bind_u32(1, 9);
                bnd.dispatch(mtl_size(groups_x, groups_y, 1), mtl_size(16, 16, 1));
                Ok(())
            })?;
            copied_norm2 = true;
        }

        if self.cfg.precision.needs_f32_rows() {
            let src = ds
                .x_f32
                .as_ref()
                .ok_or_else(|| "emeans: ResidentDataset missing f32 storage".to_string())?;
            let dst = self
                .chunk_x_f32
                .as_ref()
                .ok_or_else(|| "emeans: runner missing chunk_x_f32".to_string())?;
            let copy_norm2 = u32::from(!copied_norm2);
            self.rt.with_binder(|bnd| {
                bnd.set_pipeline(&self.pipes.gather_rows_f32);
                bnd.bind_gpu_buf(src, 0);
                bnd.bind_gpu_buf(&ds.row_norm2, 1);
                bnd.bind_gpu_buf(order_buf, 2);
                bnd.bind_gpu_buf(dst, 3);
                bnd.bind_gpu_buf(&self.chunk_norm2, 4);
                bnd.bind_u32(ds.n, 5);
                bnd.bind_u32(row_offset, 6);
                bnd.bind_u32(chunk_rows, 7);
                bnd.bind_u32(self.cfg.d, 8);
                bnd.bind_u32(copy_norm2, 9);
                bnd.dispatch(mtl_size(groups_x, groups_y, 1), mtl_size(16, 16, 1));
                Ok(())
            })?;
        }
        Ok(())
    }

    fn encode_assign_csr_and_obj(
        &self,
        m: u32,
        final_epoch: bool,
        epoch_labels_buf: Option<&GpuBuffer>,
        epoch_off: u32,
    ) -> Result<(), String> {
        if m == 0 {
            return Ok(());
        }
        let x_assign = if self.cfg.precision.assign_is_bf16() {
            self.chunk_x_bf16
                .as_ref()
                .ok_or_else(|| "emeans: missing chunk_x_bf16".to_string())?
        } else {
            self.chunk_x_f32
                .as_ref()
                .ok_or_else(|| "emeans: missing chunk_x_f32".to_string())?
        };
        let c_eff = if self.cfg.precision.assign_is_bf16() {
            self.c_eff_bf16
                .as_ref()
                .ok_or_else(|| "emeans: missing c_eff_bf16".to_string())?
        } else {
            self.c_eff_f32
                .as_ref()
                .ok_or_else(|| "emeans: missing c_eff_f32".to_string())?
        };

        encode_assign_chunk(
            &self.rt,
            &self.pipes,
            self.backend,
            self.cfg.precision,
            self.cfg.spherical,
            m,
            self.cfg.k,
            self.cfg.d,
            self.part_k,
            self.split_c,
            x_assign,
            c_eff,
            &self.chunk_norm2,
            &self.norm2_c,
            &self.chunk_labels,
            &self.chunk_dists,
            epoch_labels_buf.unwrap_or(&self.dummy_buf),
            epoch_labels_buf.is_some(),
            epoch_off,
            self.part_scores.as_ref().unwrap_or(&self.dummy_buf),
            self.part_labels.as_ref().unwrap_or(&self.dummy_buf),
        )?;

        // Pack 64-bit `(ulong(label) << 16) | ulong(local_row)` keys and clear overflow slice count.
        dispatch_1d(&self.rt, &self.pipes.pack_label_keys, m as usize, |bnd| {
            bnd.bind_gpu_buf(&self.chunk_labels, 0);
            bnd.bind_gpu_buf(&self.keys_a, 1);
            bnd.bind_gpu_buf(&self.overflow_slice_count, 2);
            bnd.bind_u32(m, 3);
        })?;

        let sorted_keys = encode_radix_passes(
            &self.rt,
            &self.pipes,
            &self.keys_a,
            &self.keys_b,
            &self.radix_hist,
            m,
            csr_radix_shifts(self.cfg.k),
        )?;

        let final_ep_u32 = u32::from(final_epoch);
        dispatch_1d(&self.rt, &self.pipes.find_bucket_bounds, self.cfg.k as usize, |bnd| {
            bnd.bind_gpu_buf(sorted_keys, 0);
            bnd.bind_gpu_buf(&self.bucket_info, 1);
            bnd.bind_gpu_buf(&self.slice_meta, 2);
            bnd.bind_gpu_buf(&self.overflow_slice_count, 3);
            bnd.bind_gpu_buf(&self.batch_counts, 4);
            bnd.bind_gpu_buf(&self.epoch_counts, 5);
            bnd.bind_gpu_buf(
                if final_epoch {
                    &self.pass_counts
                } else {
                    &self.dummy_buf
                },
                6,
            );
            bnd.bind_u32(m, 7);
            bnd.bind_u32(self.cfg.k, 8);
            bnd.bind_u32(final_ep_u32, 9);
        })?;

        let x_stats = if self.cfg.precision.stats_is_bf16() {
            self.chunk_x_bf16
                .as_ref()
                .ok_or_else(|| "emeans: missing chunk_x_bf16 for stats".to_string())?
        } else {
            self.chunk_x_f32
                .as_ref()
                .ok_or_else(|| "emeans: missing chunk_x_f32 for stats".to_string())?
        };

        // Overflow slices exist only when at least one cluster has > 256 rows (`m > 256`).
        if m > 256 {
            let max_heavy = (m as usize) / 257;
            let max_slices = ((m as usize) + max_heavy * 255)
                .div_ceil(256)
                .clamp(1, MAX_OVERFLOW_SLICES);
            let groups_x = (self.cfg.d as usize).div_ceil(16);
            let groups_y = max_slices.div_ceil(16);
            let gather_pipe = if self.cfg.precision.stats_is_bf16() {
                &self.pipes.gather_slice_bf16
            } else {
                &self.pipes.gather_slice_f32
            };
            self.rt.with_binder(|bnd| {
                bnd.set_pipeline(gather_pipe);
                bnd.bind_gpu_buf(sorted_keys, 0);
                bnd.bind_gpu_buf(x_stats, 1);
                bnd.bind_gpu_buf(&self.slice_meta, 2);
                bnd.bind_gpu_buf(&self.overflow_slice_count, 3);
                bnd.bind_gpu_buf(&self.slice_partials, 4);
                bnd.bind_u32(self.cfg.d, 5);
                bnd.dispatch(mtl_size(groups_x, groups_y, 1), mtl_size(16, 16, 1));
                Ok(())
            })?;
        }

        let fold_gx = (self.cfg.d as usize).div_ceil(16);
        let fold_gy = (self.cfg.k as usize).div_ceil(16);
        let stats_is_bf16 = u32::from(self.cfg.precision.stats_is_bf16());
        self.rt.with_binder(|bnd| {
            bnd.set_pipeline(&self.pipes.fold_slice_partials);
            bnd.bind_gpu_buf(sorted_keys, 0);
            bnd.bind_gpu_buf(x_stats, 1);
            bnd.bind_gpu_buf(&self.bucket_info, 2);
            bnd.bind_gpu_buf(&self.slice_partials, 3);
            bnd.bind_gpu_buf(&self.batch_sums, 4);
            bnd.bind_gpu_buf(if final_epoch { &self.pass_sums } else { &self.dummy_buf }, 5);
            bnd.bind_u32(self.cfg.k, 6);
            bnd.bind_u32(self.cfg.d, 7);
            bnd.bind_u32(stats_is_bf16, 8);
            bnd.bind_u32(final_ep_u32, 9);
            bnd.dispatch(mtl_size(fold_gx, fold_gy, 1), mtl_size(16, 16, 1));
            Ok(())
        })?;

        // Pure-integer IEEE-754 binary64 objective reduction.
        self.rt.with_binder(|bnd| {
            bnd.set_pipeline(&self.pipes.accum_chunk_obj);
            bnd.bind_gpu_buf(&self.chunk_dists, 0);
            bnd.bind_gpu_buf(&self.epoch_obj_bits, 1);
            bnd.bind_u32(m, 2);
            bnd.dispatch(mtl_size(1, 1, 1), mtl_size(256, 1, 1));
            Ok(())
        })
    }

    fn encode_ema_and_revive(
        &self,
        alpha: f32,
        step: u64,
        check_revive: bool,
        prepare_after: bool,
    ) -> Result<(), String> {
        self.rt.with_binder(|bnd| {
            bnd.set_pipeline(&self.pipes.apply_ema);
            bnd.bind_gpu_buf(&self.centroids, 0);
            bnd.bind_gpu_buf(&self.usage, 1);
            bnd.bind_gpu_buf(&self.batch_sums, 2);
            bnd.bind_gpu_buf(&self.batch_counts, 3);
            bnd.bind_u32(self.cfg.k, 4);
            bnd.bind_u32(self.cfg.d, 5);
            bnd.bind_f32(alpha, 6);
            bnd.dispatch(mtl_size(self.cfg.k as usize, 1, 1), mtl_size(256, 1, 1));
            Ok(())
        })?;

        if check_revive && self.cfg.unused_threshold > 0.0 {
            self.rt.with_binder(|bnd| {
                bnd.set_pipeline(&self.pipes.revive_prepare);
                bnd.bind_gpu_buf(&self.usage, 0);
                bnd.bind_gpu_buf(&self.keys_a, 1);
                bnd.bind_gpu_buf(&self.unused_ids, 2);
                bnd.bind_gpu_buf(&self.revive_meta, 3);
                bnd.bind_u32(self.cfg.k, 4);
                bnd.bind_f32(self.cfg.unused_threshold, 5);
                bnd.bind_bytes(&self.cfg.seed.to_ne_bytes(), 6);
                bnd.bind_bytes(&step.to_ne_bytes(), 7);
                bnd.dispatch(mtl_size(1, 1, 1), mtl_size(256, 1, 1));
                Ok(())
            })?;

            let sorted_donors = encode_radix_passes(
                &self.rt,
                &self.pipes,
                &self.keys_a,
                &self.keys_b,
                &self.radix_hist,
                self.cfg.k,
                &[32, 40, 48, 56],
            )?;

            let use_bf16_repr = u32::from(self.cfg.precision.assign_is_bf16());
            self.rt.with_binder(|bnd| {
                bnd.set_pipeline(&self.pipes.revive_splits);
                bnd.bind_gpu_buf(sorted_donors, 0);
                bnd.bind_gpu_buf(&self.unused_ids, 1);
                bnd.bind_gpu_buf(&self.revive_meta, 2);
                bnd.bind_gpu_buf(&self.centroids, 3);
                bnd.bind_gpu_buf(&self.usage, 4);
                bnd.bind_u32(self.cfg.k, 5);
                bnd.bind_u32(self.cfg.d, 6);
                bnd.bind_f32(self.cfg.split_epsilon, 7);
                bnd.bind_u32(use_bf16_repr, 8);
                bnd.dispatch(mtl_size(self.cfg.k as usize, 1, 1), mtl_size(256, 1, 1));
                Ok(())
            })?;
        }

        if prepare_after {
            self.encode_prepare_centroids_self()?;
        }
        Ok(())
    }

    /// Execute a full epoch over a GPU-resident dataset inside a single
    /// [`GpuRuntime::async_scope`] with one end-of-epoch `synchronize()`.
    ///
    /// When `labels_out` is `Some(dst)`:
    /// - `dst.len()` must equal `ds.rows() as usize`.
    /// - Each visited row `row_id = order[pos]` across `batches` receives its
    ///   assigned cluster index at `dst[row_id as usize]` (so `labels_out` is
    ///   always indexed in dataset row order `0..ds.rows()`, regardless of
    ///   whether `order` is identity `0..n` or a shuffled permutation).
    pub fn run_resident_epoch(
        &mut self,
        ds: &ResidentDataset,
        order: &[u32],
        batches: &[EpochBatchPlan],
        final_epoch: bool,
        labels_out: Option<&mut [u32]>,
    ) -> Result<(f64, usize, Vec<u64>), String> {
        if ds.runtime_addr != Arc::as_ptr(&self.rt) as usize {
            return Err("emeans::run_resident_epoch: ResidentDataset belongs to another GpuRuntime".to_string());
        }
        if ds.d != self.cfg.d || ds.precision != self.cfg.precision {
            return Err(format!(
                "emeans::run_resident_epoch: ResidentDataset (d={}, {:?}) mismatches runner (d={}, {:?})",
                ds.d, ds.precision, self.cfg.d, self.cfg.precision
            ));
        }
        if let Some(ref out) = labels_out {
            if out.len() != ds.n as usize {
                return Err(format!(
                    "emeans::run_resident_epoch: labels_out len {} != dataset rows {}",
                    out.len(),
                    ds.n
                ));
            }
        }

        let mut total_epoch_rows = 0usize;
        for (bi, b) in batches.iter().enumerate() {
            if !b.alpha.is_finite() || !(0.0..=1.0).contains(&b.alpha) {
                return Err(format!(
                    "emeans::run_resident_epoch: batch {bi} alpha {} must be finite in [0, 1]",
                    b.alpha
                ));
            }
            let start = b.row_offset as usize;
            let end = start
                .checked_add(b.row_count as usize)
                .ok_or_else(|| format!("emeans::run_resident_epoch: batch {bi} row range overflow"))?;
            if end > order.len() {
                return Err(format!(
                    "emeans::run_resident_epoch: batch {bi} range {start}..{end} exceeds order len {}",
                    order.len()
                ));
            }
            for &row_id in &order[start..end] {
                if row_id >= ds.n {
                    return Err(format!(
                        "emeans::run_resident_epoch: order row index {row_id} >= dataset rows {}",
                        ds.n
                    ));
                }
            }
            total_epoch_rows = total_epoch_rows
                .checked_add(b.row_count as usize)
                .ok_or_else(|| "emeans::run_resident_epoch: total epoch rows overflow usize".to_string())?;
        }
        if total_epoch_rows > u32::MAX as usize {
            return Err(format!(
                "emeans::run_resident_epoch: total epoch rows {total_epoch_rows} exceed u32"
            ));
        }
        if batches.is_empty() {
            return Ok((0.0, 0, vec![0u64; self.cfg.k as usize]));
        }

        self.in_flight_staging.clear();
        let order_buf = self.rt.alloc_buffer_from_u32(order)?;
        let epoch_labels_buf = if labels_out.is_some() && total_epoch_rows > 0 {
            Some(self.rt.alloc_buffer(total_epoch_rows * 4)?)
        } else {
            None
        };

        {
            let _scope = self.rt.async_scope();
            encode_zero_buf(&self.rt, &self.pipes.zero_f32, &self.fault_flags)?;
            encode_zero_buf(&self.rt, &self.pipes.zero_f32, &self.batch_sums)?;
            encode_zero_buf(&self.rt, &self.pipes.zero_f32, &self.batch_counts)?;
            encode_zero_buf(&self.rt, &self.pipes.zero_f32, &self.epoch_counts)?;
            encode_zero_buf(&self.rt, &self.pipes.zero_f32, &self.epoch_obj_bits)?;
            encode_zero_buf(&self.rt, &self.pipes.zero_f32, &self.revive_meta)?;
            if final_epoch {
                encode_zero_buf(&self.rt, &self.pipes.zero_f32, &self.pass_sums)?;
                encode_zero_buf(&self.rt, &self.pipes.zero_f32, &self.pass_counts)?;
            }
            self.encode_prepare_centroids_self()?;

            let mut epoch_cursor = 0u32;
            for (bi, b) in batches.iter().enumerate() {
                let mut rem = b.row_count;
                let mut cur_off = b.row_offset;
                while rem > 0 {
                    let m = rem.min(self.chunk_cap);
                    self.encode_gather_resident_chunk(ds, &order_buf, cur_off, m)?;
                    self.encode_assign_csr_and_obj(m, final_epoch, epoch_labels_buf.as_ref(), epoch_cursor)?;
                    cur_off += m;
                    epoch_cursor += m;
                    rem -= m;
                }
                let is_last = bi + 1 == batches.len();
                self.encode_ema_and_revive(b.alpha, b.step, b.check_revive, !is_last || final_epoch)?;
            }

            self.rt.synchronize()?;
        }

        // Keep temporary buffers alive until after synchronize() (`order_buf`, `epoch_labels_buf`).
        drop(order_buf);
        self.check_and_clear_faults()?;

        if let (Some(dst), Some(ep_buf)) = (labels_out, epoch_labels_buf.as_ref()) {
            let raw = ep_buf.try_contents_u32()?;
            let mut cursor = 0usize;
            for b in batches {
                let start = b.row_offset as usize;
                let end = start + (b.row_count as usize);
                for &row_id in &order[start..end] {
                    dst[row_id as usize] = raw[cursor];
                    cursor += 1;
                }
            }
        }

        let obj_bits = try_read_u64_vec(&self.epoch_obj_bits, 1)?[0];
        let revive_words = self.revive_meta.try_read_u32()?;
        let total_revivals = revive_words[1] as usize;
        let counts = try_read_u64_vec(&self.epoch_counts, self.cfg.k as usize)?;
        Ok((f64::from_bits(obj_bits), total_revivals, counts))
    }

    /// Reset streaming epoch accumulators (`epoch_counts`, `epoch_obj_bits`,
    /// `revive_meta`, `batch_sums`, `batch_counts`, plus `pass_sums`/`pass_counts`
    /// when `final_epoch == true`) and refresh effective centroids asynchronously.
    pub fn reset_pass_accumulators(&mut self, final_epoch: bool) -> Result<(), String> {
        self.streaming_final_epoch = final_epoch;
        {
            let _scope = self.rt.async_scope();
            encode_zero_buf(&self.rt, &self.pipes.zero_f32, &self.fault_flags)?;
            encode_zero_buf(&self.rt, &self.pipes.zero_f32, &self.batch_sums)?;
            encode_zero_buf(&self.rt, &self.pipes.zero_f32, &self.batch_counts)?;
            encode_zero_buf(&self.rt, &self.pipes.zero_f32, &self.epoch_counts)?;
            encode_zero_buf(&self.rt, &self.pipes.zero_f32, &self.epoch_obj_bits)?;
            encode_zero_buf(&self.rt, &self.pipes.zero_f32, &self.revive_meta)?;
            if final_epoch {
                encode_zero_buf(&self.rt, &self.pipes.zero_f32, &self.pass_sums)?;
                encode_zero_buf(&self.rt, &self.pipes.zero_f32, &self.pass_counts)?;
            }
            self.encode_prepare_centroids_self()?;
            self.rt.commit(false)?;
        }
        Ok(())
    }

    /// Stage and accumulate a streaming `f32` chunk (`chunk_rows <= MAX_GPU_CHUNK_ROWS`).
    ///
    /// When `labels_out` is `None`, submits asynchronously (`commit(false)`) with
    /// zero host wait. When `labels_out` is `Some(dst)`, synchronizes to copy
    /// the chunk's assigned labels into `dst[..chunk_rows]`.
    pub fn accumulate_f32_chunk(
        &mut self,
        rows_f32: &[f32],
        chunk_rows: u32,
        labels_out: Option<&mut [u32]>,
    ) -> Result<(), String> {
        let elems = (chunk_rows as usize)
            .checked_mul(self.cfg.d as usize)
            .ok_or_else(|| "emeans::accumulate_f32_chunk: chunk_rows * d overflows usize".to_string())?;
        if rows_f32.len() != elems {
            return Err(format!(
                "emeans::accumulate_f32_chunk: slice len {} != chunk_rows * d ({elems})",
                rows_f32.len()
            ));
        }
        if let Some(ref dst) = labels_out {
            if dst.len() != chunk_rows as usize {
                return Err(format!(
                    "emeans::accumulate_f32_chunk: labels_out len {} != chunk_rows {chunk_rows}",
                    dst.len()
                ));
            }
        }
        if chunk_rows == 0 {
            return Ok(());
        }

        if chunk_rows > self.chunk_cap {
            // Sub-chunk seamlessly into `<= self.chunk_cap` windows so arbitrary caller
            // chunk sizes never exceed the `u16` radix local-row invariant.
            let d = self.cfg.d as usize;
            let mut offset_rows = 0usize;
            let mut opt_dst = labels_out;
            while offset_rows < chunk_rows as usize {
                let take = ((chunk_rows as usize) - offset_rows).min(self.chunk_cap as usize);
                let sub_rows = &rows_f32[offset_rows * d..(offset_rows + take) * d];
                let sub_labels = match opt_dst.take() {
                    Some(slice) => {
                        let (head, tail) = slice.split_at_mut(take);
                        opt_dst = Some(tail);
                        Some(head)
                    }
                    None => None,
                };
                self.accumulate_f32_window(sub_rows, take as u32, sub_labels)?;
                offset_rows += take;
            }
            return Ok(());
        }

        self.accumulate_f32_window(rows_f32, chunk_rows, labels_out)
    }

    fn accumulate_f32_window(
        &mut self,
        rows_f32: &[f32],
        m: u32,
        labels_out: Option<&mut [u32]>,
    ) -> Result<(), String> {
        let staged = stage_f32_unwaited(&self.rt, rows_f32)?;
        {
            let _scope = self.rt.async_scope();
            encode_stage_rows_f32(
                &self.rt,
                &self.pipes,
                &staged,
                self.chunk_x_bf16.as_ref().unwrap_or(&self.dummy_buf),
                self.chunk_x_f32.as_ref().unwrap_or(&self.dummy_buf),
                &self.chunk_norm2,
                &self.fault_flags,
                m,
                self.cfg.d,
                self.cfg.precision,
            )?;
            self.encode_assign_csr_and_obj(m, self.streaming_final_epoch, None, 0)?;
            if labels_out.is_some() {
                self.rt.synchronize()?;
            } else {
                self.rt.commit(false)?;
            }
        }
        if let Some(dst) = labels_out {
            drop(staged);
            self.in_flight_staging.clear();
            if let Err(err) = self.check_and_clear_faults() {
                self.batch_sums.try_zero()?;
                self.batch_counts.try_zero()?;
                return Err(err);
            }
            let raw = self.chunk_labels.try_contents_u32()?;
            dst.copy_from_slice(&raw[..m as usize]);
        } else {
            self.in_flight_staging.push(staged);
        }
        Ok(())
    }

    /// Stage and accumulate a streaming `bf16`-bits chunk (`chunk_rows <= MAX_GPU_CHUNK_ROWS`).
    pub fn accumulate_bf16_chunk(
        &mut self,
        rows_bf16_bits: &[u16],
        chunk_rows: u32,
        labels_out: Option<&mut [u32]>,
    ) -> Result<(), String> {
        let elems = (chunk_rows as usize)
            .checked_mul(self.cfg.d as usize)
            .ok_or_else(|| "emeans::accumulate_bf16_chunk: chunk_rows * d overflows usize".to_string())?;
        if rows_bf16_bits.len() != elems {
            return Err(format!(
                "emeans::accumulate_bf16_chunk: slice len {} != chunk_rows * d ({elems})",
                rows_bf16_bits.len()
            ));
        }
        if let Some(ref dst) = labels_out {
            if dst.len() != chunk_rows as usize {
                return Err(format!(
                    "emeans::accumulate_bf16_chunk: labels_out len {} != chunk_rows {chunk_rows}",
                    dst.len()
                ));
            }
        }
        if chunk_rows == 0 {
            return Ok(());
        }

        if chunk_rows > self.chunk_cap {
            let d = self.cfg.d as usize;
            let mut offset_rows = 0usize;
            let mut opt_dst = labels_out;
            while offset_rows < chunk_rows as usize {
                let take = ((chunk_rows as usize) - offset_rows).min(self.chunk_cap as usize);
                let sub_rows = &rows_bf16_bits[offset_rows * d..(offset_rows + take) * d];
                let sub_labels = match opt_dst.take() {
                    Some(slice) => {
                        let (head, tail) = slice.split_at_mut(take);
                        opt_dst = Some(tail);
                        Some(head)
                    }
                    None => None,
                };
                self.accumulate_bf16_window(sub_rows, take as u32, sub_labels)?;
                offset_rows += take;
            }
            return Ok(());
        }

        self.accumulate_bf16_window(rows_bf16_bits, chunk_rows, labels_out)
    }

    fn accumulate_bf16_window(
        &mut self,
        rows_bf16_bits: &[u16],
        m: u32,
        labels_out: Option<&mut [u32]>,
    ) -> Result<(), String> {
        let staged = stage_u16_unwaited(&self.rt, rows_bf16_bits)?;
        {
            let _scope = self.rt.async_scope();
            encode_stage_rows_bf16(
                &self.rt,
                &self.pipes,
                &staged,
                self.chunk_x_bf16.as_ref().unwrap_or(&self.dummy_buf),
                self.chunk_x_f32.as_ref().unwrap_or(&self.dummy_buf),
                &self.chunk_norm2,
                &self.fault_flags,
                m,
                self.cfg.d,
                self.cfg.precision,
            )?;
            self.encode_assign_csr_and_obj(m, self.streaming_final_epoch, None, 0)?;
            if labels_out.is_some() {
                self.rt.synchronize()?;
            } else {
                self.rt.commit(false)?;
            }
        }
        if let Some(dst) = labels_out {
            drop(staged);
            self.in_flight_staging.clear();
            if let Err(err) = self.check_and_clear_faults() {
                self.batch_sums.try_zero()?;
                self.batch_counts.try_zero()?;
                return Err(err);
            }
            let raw = self.chunk_labels.try_contents_u32()?;
            dst.copy_from_slice(&raw[..m as usize]);
        } else {
            self.in_flight_staging.push(staged);
        }
        Ok(())
    }

    /// Apply EMA centroid update (`weight == 1.0` exact override) and optional
    /// `SplitMix64` Gumbel dead-cluster split revival for the accumulated batch,
    /// returning the number of clusters revived in this batch.
    pub fn apply_batch_update(&mut self, alpha: f32, step: u64, check_revive: bool) -> Result<usize, String> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(format!(
                "emeans::apply_batch_update: alpha {alpha} must be finite in [0, 1]"
            ));
        }

        // Wait on queued chunk accumulations first so any non-finite / overflow fault
        // refuses cleanly BEFORE modifying `centroids` or `usage`.
        self.rt.synchronize()?;
        self.in_flight_staging.clear();
        if let Err(err) = self.check_and_clear_faults() {
            self.batch_sums.try_zero()?;
            self.batch_counts.try_zero()?;
            return Err(err);
        }

        {
            let _scope = self.rt.async_scope();
            self.encode_ema_and_revive(alpha, step, check_revive, true)?;
            self.rt.synchronize()?;
        }
        self.check_and_clear_faults()?;

        if check_revive && self.cfg.unused_threshold > 0.0 {
            let meta = self.revive_meta.try_read_u32()?;
            Ok(meta[0] as usize)
        } else {
            Ok(0)
        }
    }

    /// Synchronize and read back `(epoch_objective, epoch_cluster_counts)` at the
    /// end of a streaming epoch.
    pub fn finish_streaming_epoch(&mut self) -> Result<(f64, Vec<u64>), String> {
        self.rt.synchronize()?;
        self.in_flight_staging.clear();
        self.check_and_clear_faults()?;
        let obj_bits = try_read_u64_vec(&self.epoch_obj_bits, 1)?[0];
        let counts = try_read_u64_vec(&self.epoch_counts, self.cfg.k as usize)?;
        Ok((f64::from_bits(obj_bits), counts))
    }

    /// Synchronize and read the current internal `[k * d]` `f32` centroids.
    pub fn read_centroids(&self) -> Result<Vec<f32>, String> {
        self.rt.synchronize()?;
        self.check_and_clear_faults()?;
        self.centroids.try_read_f32()
    }

    /// Synchronize and read the current `[k]` `f32` EMA usage vector.
    pub fn read_usage(&self) -> Result<Vec<f32>, String> {
        self.rt.synchronize()?;
        self.check_and_clear_faults()?;
        self.usage.try_read_f32()
    }

    /// Compute exact closing centroids in `f64` from `pass_sums` and `pass_counts`
    /// (falling back to current `centroids` where `pass_counts[j] == 0`, and
    /// applying `f64` `max_abs`-prescaled L2 normalization when `spherical == true`),
    /// writing the result back into `self.centroids` and returning `[k * d]`.
    pub fn finalize_closing_centroids(&mut self) -> Result<Vec<f32>, String> {
        self.rt.synchronize()?;
        self.in_flight_staging.clear();
        self.check_and_clear_faults()?;

        let k = self.cfg.k as usize;
        let d = self.cfg.d as usize;
        let sums = self.pass_sums.try_read_f32()?;
        let counts = try_read_u64_vec(&self.pass_counts, k)?;
        let cur_centroids = self.centroids.try_read_f32()?;

        let mut out = vec![0.0f32; k * d];
        let mut row_f64 = vec![0.0f64; d];
        for j in 0..k {
            let cnt = counts[j];
            if cnt > 0 {
                let inv = 1.0 / (cnt as f64);
                for col in 0..d {
                    row_f64[col] = (sums[j * d + col] as f64) * inv;
                }
            } else {
                for col in 0..d {
                    row_f64[col] = cur_centroids[j * d + col] as f64;
                }
            }

            if self.cfg.spherical {
                let mut max_abs = 0.0f64;
                for &v in &row_f64 {
                    let a = v.abs();
                    if a > max_abs {
                        max_abs = a;
                    }
                }
                if max_abs > 0.0 {
                    let mut sum_sq = 0.0f64;
                    for &v in &row_f64 {
                        let w = v / max_abs;
                        sum_sq += w * w;
                    }
                    let norm = sum_sq.sqrt();
                    if norm > 0.0 {
                        for col in 0..d {
                            out[j * d + col] = ((row_f64[col] / max_abs) / norm) as f32;
                        }
                    }
                }
            } else {
                for col in 0..d {
                    out[j * d + col] = row_f64[col] as f32;
                }
            }
        }

        self.centroids.try_write_f32(&out)?;
        {
            let _scope = self.rt.async_scope();
            self.encode_prepare_centroids_self()?;
            self.rt.synchronize()?;
        }
        self.check_and_clear_faults()?;
        Ok(out)
    }
}

/// Standalone GPU assignment of `f32` rows `[n, d]` against `centroids_f32` `[k, d]`.
///
/// Returns `(labels, total_objective)` where `total_objective` is:
/// - Euclidean (`spherical == false`): $\sum_i \|\tilde{x}_i - \tilde{c}_{a_i}\|_2^2$.
/// - Spherical (`spherical == true`): $\sum_i (2.0 - 2.0 \langle \tilde{x}_i, \hat{c}_{a_i} \rangle)$.
#[allow(clippy::too_many_arguments)]
pub fn assign_gpu_f32(
    rt: &Arc<GpuRuntime>,
    rows_f32: &[f32],
    n: u32,
    d: u32,
    centroids_f32: &[f32],
    k: u32,
    max_chunk_rows: u32,
    precision: EmeansGpuPrecision,
    backend: EmeansGpuBackend,
    spherical: bool,
) -> Result<(Vec<u32>, f64), String> {
    assign_gpu_impl(
        rt,
        AssignInput::F32(rows_f32),
        n,
        d,
        centroids_f32,
        k,
        max_chunk_rows,
        precision,
        backend,
        spherical,
    )
}

/// Standalone GPU assignment of `bf16`-bits rows `[n, d]` against `centroids_f32` `[k, d]`.
#[allow(clippy::too_many_arguments)]
pub fn assign_gpu_bf16(
    rt: &Arc<GpuRuntime>,
    rows_bf16_bits: &[u16],
    n: u32,
    d: u32,
    centroids_f32: &[f32],
    k: u32,
    max_chunk_rows: u32,
    precision: EmeansGpuPrecision,
    backend: EmeansGpuBackend,
    spherical: bool,
) -> Result<(Vec<u32>, f64), String> {
    assign_gpu_impl(
        rt,
        AssignInput::Bf16(rows_bf16_bits),
        n,
        d,
        centroids_f32,
        k,
        max_chunk_rows,
        precision,
        backend,
        spherical,
    )
}

enum AssignInput<'a> {
    F32(&'a [f32]),
    Bf16(&'a [u16]),
}

#[allow(clippy::too_many_arguments)]
fn assign_gpu_impl(
    rt: &Arc<GpuRuntime>,
    input: AssignInput<'_>,
    n: u32,
    d: u32,
    centroids_f32: &[f32],
    k: u32,
    max_chunk_rows: u32,
    precision: EmeansGpuPrecision,
    backend: EmeansGpuBackend,
    spherical: bool,
) -> Result<(Vec<u32>, f64), String> {
    if k == 0 || d == 0 {
        return Err(format!("emeans::assign_gpu: k ({k}) and d ({d}) must both be >= 1"));
    }
    let kd = (k as usize)
        .checked_mul(d as usize)
        .ok_or_else(|| "emeans::assign_gpu: k * d overflows usize".to_string())?;
    if kd > u32::MAX as usize {
        return Err(format!("emeans::assign_gpu: k * d ({kd}) exceeds Metal uint indexing"));
    }
    if centroids_f32.len() != kd {
        return Err(format!(
            "emeans::assign_gpu: centroids len {} != k * d ({kd})",
            centroids_f32.len()
        ));
    }
    let n_elems = (n as usize)
        .checked_mul(d as usize)
        .ok_or_else(|| "emeans::assign_gpu: n * d overflows usize".to_string())?;
    match input {
        AssignInput::F32(s) if s.len() != n_elems => {
            return Err(format!(
                "emeans::assign_gpu_f32: rows len {} != n * d ({n_elems})",
                s.len()
            ));
        }
        AssignInput::Bf16(s) if s.len() != n_elems => {
            return Err(format!(
                "emeans::assign_gpu_bf16: rows len {} != n * d ({n_elems})",
                s.len()
            ));
        }
        _ => {}
    }
    if centroids_f32.iter().any(|v| !v.is_finite()) {
        return Err(format!("{NONFINITE_ERR_PREFIX}: centroids contain NaN or Inf"));
    }

    let resolved_backend = resolve_backend(rt, backend)?;
    let chunk_cap = max_chunk_rows.clamp(1, MAX_GPU_CHUNK_ROWS).min(n.max(1));
    let (part_k, split_c) = compute_partition_plan(k, d, chunk_cap)?;
    let pipes = EmeansPipelines::resolve(rt, resolved_backend)?;

    let dummy_buf = rt.alloc_buffer(16)?;
    let fault_flags = rt.alloc_buffer(4)?;
    fault_flags.try_write_u32(&[0])?;

    let c_buf = stage_f32_unwaited(rt, centroids_f32)?;
    let c_eff_bf16 = if precision.assign_is_bf16() {
        Some(rt.alloc_buffer((kd * 2).max(4))?)
    } else {
        None
    };
    let c_eff_f32 = if !precision.assign_is_bf16() {
        Some(rt.alloc_buffer(kd * 4)?)
    } else {
        None
    };
    let norm2_c = rt.alloc_buffer((k as usize) * 4)?;

    if n == 0 {
        {
            let _scope = rt.async_scope();
            encode_prepare_centroids(
                rt,
                &pipes,
                precision,
                spherical,
                k,
                d,
                &c_buf,
                c_eff_bf16.as_ref(),
                c_eff_f32.as_ref(),
                &norm2_c,
                &fault_flags,
            )?;
            rt.synchronize()?;
        }
        drop(c_buf);
        let flags = fault_flags.try_read_u32()?[0];
        if flags != 0 {
            return Err(format_fault_error(flags));
        }
        return Ok((Vec::new(), 0.0));
    }

    let chunk_usize = chunk_cap as usize;
    let chunk_elems = chunk_usize * (d as usize);
    let chunk_x_bf16 = if precision.assign_is_bf16() {
        Some(rt.alloc_buffer((chunk_elems * 2).max(4))?)
    } else {
        None
    };
    let chunk_x_f32 = if !precision.assign_is_bf16() {
        Some(rt.alloc_buffer(chunk_elems * 4)?)
    } else {
        None
    };
    let chunk_norm2 = rt.alloc_buffer(chunk_usize * 4)?;
    let chunk_labels = rt.alloc_buffer(chunk_usize * 4)?;
    let chunk_dists = rt.alloc_buffer(chunk_usize * 4)?;
    let labels_buf = rt.alloc_buffer((n as usize) * 4)?;
    let obj_bits_buf = rt.alloc_buffer(8)?;
    obj_bits_buf.try_zero()?;

    let (part_scores, part_labels) = if split_c > 1 {
        let p_elems = (split_c as usize) * chunk_usize;
        (Some(rt.alloc_buffer(p_elems * 4)?), Some(rt.alloc_buffer(p_elems * 4)?))
    } else {
        (None, None)
    };

    let mut staged_chunks = Vec::new();
    {
        let _scope = rt.async_scope();
        encode_prepare_centroids(
            rt,
            &pipes,
            precision,
            spherical,
            k,
            d,
            &c_buf,
            c_eff_bf16.as_ref(),
            c_eff_f32.as_ref(),
            &norm2_c,
            &fault_flags,
        )?;

        let d_usize = d as usize;
        let mut cursor = 0u32;
        while cursor < n {
            let m = (n - cursor).min(chunk_cap);
            let start_elem = (cursor as usize) * d_usize;
            let end_elem = start_elem + (m as usize) * d_usize;
            match input {
                AssignInput::F32(rows) => {
                    let staged = stage_f32_unwaited(rt, &rows[start_elem..end_elem])?;
                    let write_bf16 = u32::from(precision.assign_is_bf16());
                    let write_f32 = u32::from(!precision.assign_is_bf16());
                    rt.with_binder(|bnd| {
                        bnd.set_pipeline(&pipes.stage_rows_f32);
                        bnd.bind_gpu_buf(&staged, 0);
                        bnd.bind_gpu_buf(chunk_x_bf16.as_ref().unwrap_or(&dummy_buf), 1);
                        bnd.bind_gpu_buf(chunk_x_f32.as_ref().unwrap_or(&dummy_buf), 2);
                        bnd.bind_gpu_buf(&chunk_norm2, 3);
                        bnd.bind_gpu_buf(&fault_flags, 4);
                        bnd.bind_u32(m, 5);
                        bnd.bind_u32(d, 6);
                        bnd.bind_u32(write_bf16, 7);
                        bnd.bind_u32(write_f32, 8);
                        bnd.bind_u32(write_bf16, 9);
                        bnd.dispatch(mtl_size(m as usize, 1, 1), mtl_size(256, 1, 1));
                        Ok(())
                    })?;
                    staged_chunks.push(staged);
                }
                AssignInput::Bf16(rows) => {
                    let staged = stage_u16_unwaited(rt, &rows[start_elem..end_elem])?;
                    let write_bf16 = u32::from(precision.assign_is_bf16());
                    let write_f32 = u32::from(!precision.assign_is_bf16());
                    rt.with_binder(|bnd| {
                        bnd.set_pipeline(&pipes.stage_rows_bf16);
                        bnd.bind_gpu_buf(&staged, 0);
                        bnd.bind_gpu_buf(chunk_x_bf16.as_ref().unwrap_or(&dummy_buf), 1);
                        bnd.bind_gpu_buf(chunk_x_f32.as_ref().unwrap_or(&dummy_buf), 2);
                        bnd.bind_gpu_buf(&chunk_norm2, 3);
                        bnd.bind_gpu_buf(&fault_flags, 4);
                        bnd.bind_u32(m, 5);
                        bnd.bind_u32(d, 6);
                        bnd.bind_u32(write_bf16, 7);
                        bnd.bind_u32(write_f32, 8);
                        bnd.dispatch(mtl_size(m as usize, 1, 1), mtl_size(256, 1, 1));
                        Ok(())
                    })?;
                    staged_chunks.push(staged);
                }
            }

            let x_assign = if precision.assign_is_bf16() {
                chunk_x_bf16.as_ref().unwrap_or(&dummy_buf)
            } else {
                chunk_x_f32.as_ref().unwrap_or(&dummy_buf)
            };
            let c_eff = if precision.assign_is_bf16() {
                c_eff_bf16.as_ref().unwrap_or(&dummy_buf)
            } else {
                c_eff_f32.as_ref().unwrap_or(&dummy_buf)
            };

            encode_assign_chunk(
                rt,
                &pipes,
                resolved_backend,
                precision,
                spherical,
                m,
                k,
                d,
                part_k,
                split_c,
                x_assign,
                c_eff,
                &chunk_norm2,
                &norm2_c,
                &chunk_labels,
                &chunk_dists,
                &labels_buf,
                true,
                cursor,
                part_scores.as_ref().unwrap_or(&dummy_buf),
                part_labels.as_ref().unwrap_or(&dummy_buf),
            )?;

            rt.with_binder(|bnd| {
                bnd.set_pipeline(&pipes.accum_chunk_obj);
                bnd.bind_gpu_buf(&chunk_dists, 0);
                bnd.bind_gpu_buf(&obj_bits_buf, 1);
                bnd.bind_u32(m, 2);
                bnd.dispatch(mtl_size(1, 1, 1), mtl_size(256, 1, 1));
                Ok(())
            })?;

            cursor += m;
        }
        rt.synchronize()?;
    }
    drop(c_buf);
    drop(staged_chunks);

    let flags = fault_flags.try_read_u32()?[0];
    if flags != 0 {
        return Err(format_fault_error(flags));
    }
    let labels = labels_buf.try_read_u32()?;
    let obj_bits = try_read_u64_vec(&obj_bits_buf, 1)?[0];
    Ok((labels, f64::from_bits(obj_bits)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revival_gumbel_key_is_finite_deterministic_and_monotonic_in_usage() {
        let g1 = revival_gumbel_key(4.0, 0x1234_5678_9abc_def0, 7, 3);
        let g2 = revival_gumbel_key(4.0, 0x1234_5678_9abc_def0, 7, 3);
        assert_eq!(g1.to_bits(), g2.to_bits());
        assert!(g1.is_finite());

        // Same (seed, step, cluster_id) => identical Gumbel noise term `-ln(-ln(u))`,
        // so doubling `usage` increases `g_j` by `+ln(2)`.
        let g_hi = revival_gumbel_key(8.0, 0x1234_5678_9abc_def0, 7, 3);
        assert!((g_hi - g1 - std::f32::consts::LN_2).abs() < 1e-6);

        // Distinct cluster ids / steps / seeds yield distinct noise draws.
        let g_other_cluster = revival_gumbel_key(4.0, 0x1234_5678_9abc_def0, 7, 4);
        let g_other_step = revival_gumbel_key(4.0, 0x1234_5678_9abc_def0, 8, 3);
        assert_ne!(g1.to_bits(), g_other_cluster.to_bits());
        assert_ne!(g1.to_bits(), g_other_step.to_bits());
    }

    #[test]
    fn partition_plan_splits_prime_1031_and_refuses_i32_overflow() {
        let (part_k, split_c) = compute_partition_plan(1_031, 67, 64).expect("1031 x 67");
        assert_eq!(part_k, 1_024);
        assert_eq!(split_c, 2);

        let (part_small, split_small) = compute_partition_plan(512, 128, 1_024).expect("512 x 128");
        assert_eq!(part_small, 512);
        assert_eq!(split_small, 1);

        assert!(compute_partition_plan(0, 16, 64).is_err());
        assert!(compute_partition_plan(16, 0, 64).is_err());
        assert!(compute_partition_plan(16, (i32::MAX as u32) + 1, 64).is_err());
        assert!(compute_partition_plan(16, 40_000, MAX_GPU_CHUNK_ROWS).is_err());
    }

    #[test]
    fn config_validator_rejects_bad_extents_and_hyperparameters() {
        let base = EmeansGpuConfig {
            k: 4,
            d: 8,
            max_chunk_rows: 128,
            precision: EmeansGpuPrecision::Bf16,
            backend: EmeansGpuBackend::Auto,
            spherical: false,
            unused_threshold: 0.1,
            split_epsilon: 1e-4,
            init_usage: 1.0,
            seed: 42,
        };
        assert_eq!(validate_config(&base, 32).unwrap(), 32);
        assert!(validate_config(&base, 31).is_err());

        let mut bad = base.clone();
        bad.unused_threshold = -0.01;
        assert!(validate_config(&bad, 32).is_err());

        let mut nan_eps = base.clone();
        nan_eps.split_epsilon = f32::NAN;
        assert!(validate_config(&nan_eps, 32).is_err());

        let mut inf_u = base;
        inf_u.init_usage = f32::INFINITY;
        assert!(validate_config(&inf_u, 32).is_err());
    }

    #[test]
    fn prescale_normalize_handles_extreme_scales_and_zero_rows() {
        let rows = [1.0e-30f32, -1.0e-30, 1.0e30, 1.0e30, 0.0, 0.0];
        let out = prescale_normalize_rows_f64(&rows, 3, 2);
        let s = std::f32::consts::FRAC_1_SQRT_2;
        assert!((out[0] - s).abs() < 1e-6);
        assert!((out[1] + s).abs() < 1e-6);
        assert!((out[2] - s).abs() < 1e-6);
        assert!((out[3] - s).abs() < 1e-6);
        assert_eq!(out[4], 0.0);
        assert_eq!(out[5], 0.0);
    }

    #[test]
    fn csr_radix_shifts_cover_cluster_bit_widths() {
        assert_eq!(csr_radix_shifts(1), &[] as &[u32]);
        assert_eq!(csr_radix_shifts(256), &[16]);
        assert_eq!(csr_radix_shifts(257), &[16, 24]);
        assert_eq!(csr_radix_shifts(65_536), &[16, 24]);
        assert_eq!(csr_radix_shifts(65_537), &[16, 24, 32]);
    }
}
