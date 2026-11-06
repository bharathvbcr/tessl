//! Int8 and int4 TensorOps GEMM, row quantize, and decode-only GQA.
//!
//! Apple GEMM that needs `matmul2d` lives here. The legacy
//! [`crate::nn::gemm_i8_dequant`] kernel is unchanged. Nothing in this module
//! computes a refused call on the host.

use std::sync::{Arc, OnceLock};

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;

use crate::dispatch::{self, set_f32, set_gpu_buf, set_u32, Binder};
use crate::runtime::GpuRuntime;
use crate::tensor::GpuBuffer;

const MAX_INT8_PRODUCT: u32 = 128 * 128;
/// Largest K whose full-range int8 dot still fits in int32.
pub(crate) const MAX_K_EXACT: u32 = i32::MAX as u32 / MAX_INT8_PRODUCT;

/// Head dimensions [`gqa_decode`] accepts. Any other value is an error that
/// names the rejected dimension and does not dispatch.
pub const ATTENTION_HEAD_DIMS: [u32; 5] = [32, 64, 128, 256, 512];

/// Store type of a quantized GEMM. The int32 product is scaled in f32 first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum I8Out {
    F32,
    F16,
    Bf16,
}

/// Activation scale. `Rows` is one finite f32 per row of A.
#[derive(Clone, Copy)]
pub enum I8AScale<'a> {
    Scalar(f32),
    Rows(&'a GpuBuffer),
}

/// Weight scale. Per-column and per-group are different kernels; a call cannot
/// ask for both.
#[derive(Clone, Copy)]
pub enum I8BScale<'a> {
    None,
    PerColumn(&'a GpuBuffer),
    /// `scales` is `[K / group, N]`, group in {64, 128, 256}. Groups flush in
    /// ascending order: `acc += (float)sum_g * scales[g, n]`, then `*= a_scale`.
    PerGroup {
        scales: &'a GpuBuffer,
        group: u32,
    },
}

/// `RowMajorKN` is B `[K, N]`. `TransposedNK` is B `[N, K]` (safetensors / MLX).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum I8BLayout {
    RowMajorKN,
    TransposedNK,
}

/// One int8×int8→int32 GEMM with a fused dequant epilogue.
pub struct I8Gemm<'a> {
    pub a: &'a GpuBuffer,
    pub b: &'a GpuBuffer,
    pub c: &'a GpuBuffer,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_scale: I8AScale<'a>,
    pub b_scale: I8BScale<'a>,
    pub bias: Option<&'a GpuBuffer>,
    pub b_layout: I8BLayout,
    pub out: I8Out,
}

/// Which W4A8 weight decode to run. The arms do not reference each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W4Arm {
    /// `int4b_format` on the right operand. Two's complement, low nibble first.
    /// An explicit unsupported error when the device probe fails. No host GEMM.
    Int4b,
    /// Software unpack, unsigned nibble minus 8, then the int8 GEMM.
    UnpackZp8,
}

/// int8 activations times packed int4 weights `[N, K]`.
pub struct W4Gemm<'a> {
    pub a: &'a GpuBuffer,
    pub packed: &'a GpuBuffer,
    pub c: &'a GpuBuffer,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_scale: I8AScale<'a>,
    pub b_scale: I8BScale<'a>,
    pub bias: Option<&'a GpuBuffer>,
    pub out: I8Out,
    pub arm: W4Arm,
}

/// Pipeline the int8 GEMM dispatches. `group == 0` is the full-K kernel.
/// M == 1 is always the int8 GEMV, including grouped scales.
pub fn i8_gemm_pipeline_name(m: u32, layout: I8BLayout, group: u32) -> Result<&'static str, String> {
    if m == 0 {
        return Err("gemm_i8: m, n and k must be non-zero".into());
    }
    if group != 0 && !matches!(group, 64 | 128 | 256) {
        return Err(format!("gemm_i8: group {group} is not 64, 128 or 256"));
    }
    if m == 1 {
        return Ok("i8_gemv_dequant");
    }
    let wide = m > 64;
    let nt = matches!(layout, I8BLayout::TransposedNK);
    Ok(match (wide, nt, group) {
        (false, false, 0) => "i8_w8a8_64",
        (false, true, 0) => "i8_w8a8_64_nt",
        (true, false, 0) => "i8_w8a8_128",
        (true, true, 0) => "i8_w8a8_128_nt",
        (false, false, 64) => "i8_w8a8_g64_64",
        (false, true, 64) => "i8_w8a8_g64_64_nt",
        (true, false, 64) => "i8_w8a8_g64_128",
        (true, true, 64) => "i8_w8a8_g64_128_nt",
        (false, false, 128) => "i8_w8a8_g128_64",
        (false, true, 128) => "i8_w8a8_g128_64_nt",
        (true, false, 128) => "i8_w8a8_g128_128",
        (true, true, 128) => "i8_w8a8_g128_128_nt",
        (false, false, 256) => "i8_w8a8_g256_64",
        (false, true, 256) => "i8_w8a8_g256_64_nt",
        (true, false, 256) => "i8_w8a8_g256_128",
        (true, true, 256) => "i8_w8a8_g256_128_nt",
        _ => return Err(format!("gemm_i8: group {group} is not 64, 128 or 256")),
    })
}

/// Per-row symmetric int8 quantize. `scale = absmax / 127`, or 1 for an
/// all-zero row. `q = clamp(rint(x / scale), -127, 127)`. A non-finite input
/// returns an error and does not run a matmul.
pub fn quantize_i8_rows(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    q: &GpuBuffer,
    scales: &GpuBuffer,
    rows: u32,
    cols: u32,
) -> Result<(), String> {
    if rows == 0 || cols == 0 {
        return Err("quantize_i8_rows: rows and cols must be non-zero".into());
    }
    let n = numel(&[rows, cols], "quantize_i8_rows")?;
    require_bytes(rt, x, n * 4, "quantize_i8_rows x")?;
    require_bytes(rt, q, n, "quantize_i8_rows q")?;
    require_bytes(rt, scales, rows as usize * 4, "quantize_i8_rows scales")?;
    reject_write_alias("quantize_i8_rows", &[("q", q), ("scales", scales)], &[("x", x)])?;
    let bad = rt.alloc_buffer(rows as usize * 4)?;
    bad.try_write_u32(&vec![0u32; rows as usize])?;
    let p = rt.pipeline("i8_quantize_rows")?;
    dispatch_groups(rt, &p, rows as usize, 256, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_gpu_buf(bnd, q, 1);
        set_gpu_buf(bnd, scales, 2);
        set_gpu_buf(bnd, &bad, 3);
        set_u32(bnd, cols, 4);
    })?;
    let flags = bad.try_contents_u32()?;
    if flags.iter().take(rows as usize).any(|v| *v != 0) {
        return Err("quantize_i8_rows: non-finite input".into());
    }
    Ok(())
}

/// `C = dequant(A @ B)` on the GPU. M == 1 runs the int8 GEMV. `2 <= M <= 64`
/// runs the 64×64 tile. Larger M runs 128×64. A malformed call returns before
/// dispatch.
pub fn gemm_i8(rt: &Arc<GpuRuntime>, spec: I8Gemm<'_>) -> Result<(), String> {
    let I8Gemm {
        a,
        b,
        c,
        m,
        n,
        k,
        a_scale,
        b_scale,
        bias,
        b_layout,
        out,
    } = spec;
    check_mnk(m, n, k, "gemm_i8")?;
    let group = scale_group(b_scale)?;
    if group == 0 && k > MAX_K_EXACT {
        return Err(format!(
            "gemm_i8: k = {k} exceeds {MAX_K_EXACT}, past which an int32 accumulator can overflow \
             at full int8 range"
        ));
    }
    if group != 0 && k % group != 0 {
        return Err(format!("gemm_i8: k = {k} is not divisible by group {group}"));
    }
    let (a_scalar, a_mode, a_rows) = split_a(a_scale)?;
    let (b_mode, b_buf, group_k) = split_b(b_scale)?;
    let b_elems = numel(&[k, n], "gemm_i8")?;
    let a_elems = numel(&[m, k], "gemm_i8")?;
    let c_elems = numel(&[m, n], "gemm_i8")?;
    for (numel, what) in [(a_elems, "gemm_i8 a"), (b_elems, "gemm_i8 b"), (c_elems, "gemm_i8 c")] {
        crate::gemm::require_i32_extent(numel, what)?;
    }
    require_bytes(rt, a, a_elems, "gemm_i8 a")?;
    require_bytes(rt, b, b_elems, "gemm_i8 b")?;
    require_bytes(rt, c, c_elems * out_bytes(out), "gemm_i8 c")?;
    if let Some(rows) = a_rows {
        require_bytes(rt, rows, m as usize * 4, "gemm_i8 a_scale")?;
        require_finite(rows, m as usize, "gemm_i8 a_scale")?;
    }
    if let Some(sc) = b_buf {
        let need = if group == 0 {
            n as usize
        } else {
            (k / group) as usize * n as usize
        };
        require_bytes(rt, sc, need * 4, "gemm_i8 b_scale")?;
        require_finite(sc, need, "gemm_i8 b_scale")?;
    }
    if let Some(bias) = bias {
        require_bytes(rt, bias, n as usize * 4, "gemm_i8 bias")?;
        require_finite(bias, n as usize, "gemm_i8 bias")?;
    }
    let mut reads = vec![("a", a), ("b", b)];
    if let Some(rows) = a_rows {
        reads.push(("a_scale", rows));
    }
    if let Some(sc) = b_buf {
        reads.push(("b_scale", sc));
    }
    if let Some(bias) = bias {
        reads.push(("bias", bias));
    }
    reject_write_alias("gemm_i8", &[("c", c)], &reads)?;

    let name = i8_gemm_pipeline_name(m, b_layout, group)?;
    let p = rt.pipeline(name)?;
    let a_bind = a_rows.unwrap_or(c);
    let b_bind = b_buf.unwrap_or(c);
    let bias_bind = bias.unwrap_or(c);
    let has_bias = u32::from(bias.is_some());
    let out_kind = out_kind(out);
    let trans = u32::from(matches!(b_layout, I8BLayout::TransposedNK));
    if m == 1 {
        return dispatch_1d(rt, &p, n as usize, |bnd| {
            set_gpu_buf(bnd, a, 0);
            set_gpu_buf(bnd, b, 1);
            set_gpu_buf(bnd, c, 2);
            set_u32(bnd, n, 3);
            set_u32(bnd, k, 4);
            set_gpu_buf(bnd, a_bind, 5);
            set_f32(bnd, a_scalar, 6);
            set_u32(bnd, a_mode, 7);
            set_gpu_buf(bnd, b_bind, 8);
            set_u32(bnd, b_mode, 9);
            set_u32(bnd, group_k.max(1), 10);
            set_gpu_buf(bnd, bias_bind, 11);
            set_u32(bnd, has_bias, 12);
            set_u32(bnd, out_kind, 13);
            set_u32(bnd, trans, 14);
        });
    }
    let (sm, sn) = if m <= 64 {
        (64usize, 64usize)
    } else {
        (128usize, 64usize)
    };
    let tiles_n = (n as usize).div_ceil(sn);
    let tiles_m = (m as usize).div_ceil(sm);
    let tptg = p.threadExecutionWidth() * 4;
    dispatch_groups(rt, &p, tiles_n * tiles_m, tptg, |bnd| {
        set_gpu_buf(bnd, a, 0);
        set_gpu_buf(bnd, b, 1);
        set_gpu_buf(bnd, c, 2);
        set_u32(bnd, m, 3);
        set_u32(bnd, n, 4);
        set_u32(bnd, k, 5);
        set_u32(bnd, tiles_n as u32, 6);
        set_u32(bnd, tiles_m as u32, 7);
        set_gpu_buf(bnd, a_bind, 8);
        set_f32(bnd, a_scalar, 9);
        set_u32(bnd, a_mode, 10);
        set_gpu_buf(bnd, b_bind, 11);
        set_u32(bnd, b_mode, 12);
        set_gpu_buf(bnd, bias_bind, 13);
        set_u32(bnd, has_bias, 14);
        set_u32(bnd, out_kind, 15);
    })
}

/// W4A8 against one caller-supplied logical matrix, decoded by `arm`.
///
/// `Int4b` requires `k % 32 == 0`. A failed device probe returns an error
/// containing `unsupported` and does not fall back to the unpack arm or the host.
/// `UnpackZp8` always uses the int8 GEMM after a GPU nibble unpack.
pub fn gemm_w4a8(rt: &Arc<GpuRuntime>, spec: W4Gemm<'_>) -> Result<(), String> {
    let W4Gemm {
        a,
        packed,
        c,
        m,
        n,
        k,
        a_scale,
        b_scale,
        bias,
        out,
        arm,
    } = spec;
    check_mnk(m, n, k, "gemm_w4a8")?;
    if k % 2 != 0 {
        return Err(format!("gemm_w4a8: k = {k} must be even"));
    }
    let group = scale_group(b_scale)?;
    if matches!(b_scale, I8BScale::PerColumn(_)) {
        return Err("gemm_w4a8: per-column scales are not a W4A8 mode".into());
    }
    if group != 0 && k % group != 0 {
        return Err(format!("gemm_w4a8: k = {k} is not divisible by group {group}"));
    }
    if matches!(arm, W4Arm::Int4b) && k % 32 != 0 {
        return Err(format!("gemm_w4a8: k = {k} must be a multiple of 32 for int4b"));
    }
    let a_elems = numel(&[m, k], "gemm_w4a8")?;
    let c_elems = numel(&[m, n], "gemm_w4a8")?;
    let packed_bytes = numel(&[n, k / 2], "gemm_w4a8")?;
    require_bytes(rt, a, a_elems, "gemm_w4a8 a")?;
    require_bytes(rt, packed, packed_bytes, "gemm_w4a8 packed")?;
    require_bytes(rt, c, c_elems * out_bytes(out), "gemm_w4a8 c")?;
    let (a_scalar, a_mode, a_rows) = split_a(a_scale)?;
    let (_b_mode, b_buf, _) = split_b(b_scale)?;
    if let Some(rows) = a_rows {
        require_bytes(rt, rows, m as usize * 4, "gemm_w4a8 a_scale")?;
        require_finite(rows, m as usize, "gemm_w4a8 a_scale")?;
    }
    if let Some(sc) = b_buf {
        let need = (k / group) as usize * n as usize;
        require_bytes(rt, sc, need * 4, "gemm_w4a8 b_scale")?;
        require_finite(sc, need, "gemm_w4a8 b_scale")?;
    }
    if let Some(bias) = bias {
        require_bytes(rt, bias, n as usize * 4, "gemm_w4a8 bias")?;
        require_finite(bias, n as usize, "gemm_w4a8 bias")?;
    }
    match arm {
        W4Arm::Int4b => {
            probe_int4b(rt)?;
            dispatch_w4(
                rt, a, packed, c, m, n, k, a_scalar, a_mode, a_rows, b_buf, group, bias, out,
            )
        }
        W4Arm::UnpackZp8 => {
            let unpacked = rt.alloc_buffer(numel(&[n, k], "gemm_w4a8")?)?;
            let p = rt.pipeline("w4a8_unpack_zp8")?;
            let count = numel(&[n, k], "gemm_w4a8")?;
            dispatch_1d(rt, &p, count, |bnd| {
                set_gpu_buf(bnd, packed, 0);
                set_gpu_buf(bnd, &unpacked, 1);
                set_u32(bnd, n, 2);
                set_u32(bnd, k, 3);
            })?;
            gemm_i8(
                rt,
                I8Gemm {
                    a,
                    b: &unpacked,
                    c,
                    m,
                    n,
                    k,
                    a_scale,
                    b_scale,
                    bias,
                    b_layout: I8BLayout::TransposedNK,
                    out,
                },
            )
        }
    }
}

/// Decode-only grouped-query attention. Query length is 1.
///
/// Q is `[batch, q_heads, head_dim]`, K and V are
/// `[batch, kv_len, kv_heads, head_dim]`, O matches Q. `q_heads` must be
/// divisible by `kv_heads`. A head dimension outside [`ATTENTION_HEAD_DIMS`]
/// returns an error that names it and does not dispatch. Prefill attention is
/// not this function.
pub fn gqa_decode(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    batch: u32,
    q_heads: u32,
    kv_heads: u32,
    kv_len: u32,
    head_dim: u32,
) -> Result<(), String> {
    if !ATTENTION_HEAD_DIMS.contains(&head_dim) {
        return Err(format!(
            "gqa_decode: head dim {head_dim} has no decode kernel (32, 64, 128, 256 or 512)"
        ));
    }
    if batch == 0 || q_heads == 0 || kv_heads == 0 || kv_len == 0 {
        return Err("gqa_decode: batch, q_heads, kv_heads and kv_len must be non-zero".into());
    }
    if q_heads % kv_heads != 0 {
        return Err(format!(
            "gqa_decode: q_heads {q_heads} is not divisible by kv_heads {kv_heads}"
        ));
    }
    let q_elems = numel(&[batch, q_heads, head_dim], "gqa_decode")?;
    let kv_elems = numel(&[batch, kv_len, kv_heads, head_dim], "gqa_decode")?;
    require_bytes(rt, q, q_elems * 4, "gqa_decode q")?;
    require_bytes(rt, k, kv_elems * 4, "gqa_decode k")?;
    require_bytes(rt, v, kv_elems * 4, "gqa_decode v")?;
    require_bytes(rt, o, q_elems * 4, "gqa_decode o")?;
    reject_write_alias("gqa_decode", &[("o", o)], &[("q", q), ("k", k), ("v", v)])?;
    let name = match head_dim {
        32 => "gqa_decode_h32",
        64 => "gqa_decode_h64",
        128 => "gqa_decode_h128",
        256 => "gqa_decode_h256",
        512 => "gqa_decode_h512",
        _ => {
            return Err(format!(
                "gqa_decode: head dim {head_dim} has no decode kernel (32, 64, 128, 256 or 512)"
            ))
        }
    };
    let p = rt.pipeline(name)?;
    let threads = numel(&[batch, q_heads], "gqa_decode")?;
    dispatch_1d(rt, &p, threads, |bnd| {
        set_gpu_buf(bnd, q, 0);
        set_gpu_buf(bnd, k, 1);
        set_gpu_buf(bnd, v, 2);
        set_gpu_buf(bnd, o, 3);
        set_u32(bnd, batch, 4);
        set_u32(bnd, q_heads, 5);
        set_u32(bnd, kv_heads, 6);
        set_u32(bnd, kv_len, 7);
    })
}

fn dispatch_w4(
    rt: &Arc<GpuRuntime>,
    a: &GpuBuffer,
    packed: &GpuBuffer,
    c: &GpuBuffer,
    m: u32,
    n: u32,
    k: u32,
    a_scalar: f32,
    a_mode: u32,
    a_rows: Option<&GpuBuffer>,
    b_scale: Option<&GpuBuffer>,
    group: u32,
    bias: Option<&GpuBuffer>,
    out: I8Out,
) -> Result<(), String> {
    let name = match group {
        0 => "w4a8_i4_32",
        64 => "w4a8_i4_g64",
        128 => "w4a8_i4_g128",
        256 => "w4a8_i4_g256",
        other => return Err(format!("gemm_w4a8: group {other} is not 64, 128 or 256")),
    };
    let p = rt.pipeline(name)?;
    let tiles_n = (n as usize).div_ceil(32);
    let tiles_m = (m as usize).div_ceil(32);
    let a_bind = a_rows.unwrap_or(c);
    let b_bind = b_scale.unwrap_or(c);
    let bias_bind = bias.unwrap_or(c);
    dispatch_groups(rt, &p, tiles_n * tiles_m, p.threadExecutionWidth() * 4, |bnd| {
        set_gpu_buf(bnd, a, 0);
        set_gpu_buf(bnd, packed, 1);
        set_gpu_buf(bnd, c, 2);
        set_u32(bnd, m, 3);
        set_u32(bnd, n, 4);
        set_u32(bnd, k, 5);
        set_u32(bnd, tiles_n as u32, 6);
        set_u32(bnd, tiles_m as u32, 7);
        set_gpu_buf(bnd, a_bind, 8);
        set_f32(bnd, a_scalar, 9);
        set_u32(bnd, a_mode, 10);
        set_gpu_buf(bnd, b_bind, 11);
        set_gpu_buf(bnd, bias_bind, 12);
        set_u32(bnd, u32::from(bias.is_some()), 13);
        set_u32(bnd, out_kind(out), 14);
    })
}

fn probe_int4b(rt: &Arc<GpuRuntime>) -> Result<(), String> {
    static CACHED: OnceLock<Result<(), String>> = OnceLock::new();
    if let Some(hit) = CACHED.get() {
        return hit.clone();
    }
    let probed = (|| -> Result<(), String> {
        if !rt.has_tensorops() {
            return Err("w4a8 int4b: unsupported (device has no TensorOps)".into());
        }
        let m = 32u32;
        let a = rt.alloc_buffer((m * m) as usize)?;
        a.try_write_bytes(&vec![0u8; (m * m) as usize])?;
        let w = rt.alloc_buffer((m * m / 2) as usize)?;
        w.try_write_bytes(&vec![0u8; (m * m / 2) as usize])?;
        let c = rt.alloc_buffer((m * m * 4) as usize)?;
        c.try_write_bytes(&vec![0u8; (m * m * 4) as usize])?;
        dispatch_w4(rt, &a, &w, &c, m, m, m, 1.0, 0, None, None, 0, None, I8Out::F32)
            .map_err(|e| format!("w4a8 int4b: unsupported ({e})"))?;
        rt.synchronize().map_err(|e| format!("w4a8 int4b: unsupported ({e})"))
    })();
    let _ = CACHED.set(probed.clone());
    CACHED.get().cloned().unwrap_or(probed)
}

fn check_mnk(m: u32, n: u32, k: u32, entry: &str) -> Result<(), String> {
    if m == 0 || n == 0 || k == 0 {
        return Err(format!("{entry}: m, n and k must be non-zero"));
    }
    for (v, name) in [(m, "m"), (n, "n"), (k, "k")] {
        if v > i32::MAX as u32 {
            return Err(format!("{entry}: {name} = {v} exceeds signed 32-bit kernel indexing"));
        }
    }
    Ok(())
}

fn scale_group(scale: I8BScale<'_>) -> Result<u32, String> {
    match scale {
        I8BScale::PerGroup { group, .. } => {
            if !matches!(group, 64 | 128 | 256) {
                return Err(format!("gemm_i8: group {group} is not 64, 128 or 256"));
            }
            Ok(group)
        }
        _ => Ok(0),
    }
}

fn split_a(scale: I8AScale<'_>) -> Result<(f32, u32, Option<&GpuBuffer>), String> {
    match scale {
        I8AScale::Scalar(v) => {
            if !v.is_finite() {
                return Err(format!("gemm_i8: a_scale must be finite, got {v}"));
            }
            Ok((v, 0, None))
        }
        I8AScale::Rows(buf) => Ok((1.0, 1, Some(buf))),
    }
}

fn split_b(scale: I8BScale<'_>) -> Result<(u32, Option<&GpuBuffer>, u32), String> {
    match scale {
        I8BScale::None => Ok((0, None, 1)),
        I8BScale::PerColumn(buf) => Ok((1, Some(buf), 1)),
        I8BScale::PerGroup { scales, group } => Ok((2, Some(scales), group)),
    }
}

fn out_kind(out: I8Out) -> u32 {
    match out {
        I8Out::F32 => 0,
        I8Out::F16 => 1,
        I8Out::Bf16 => 2,
    }
}

fn out_bytes(out: I8Out) -> usize {
    match out {
        I8Out::F32 => 4,
        I8Out::F16 | I8Out::Bf16 => 2,
    }
}

fn numel(dims: &[u32], what: &str) -> Result<usize, String> {
    dims.iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim as usize))
        .ok_or_else(|| format!("{what}: dimension product overflows usize"))
}

fn require_bytes(rt: &GpuRuntime, buf: &GpuBuffer, need: usize, what: &str) -> Result<(), String> {
    if !buf.belongs_to(rt) {
        return Err(format!("{what}: buffer belongs to another runtime"));
    }
    if buf.nbytes() < need {
        return Err(format!(
            "{what}: buffer holds {} bytes, kernel needs {need}",
            buf.nbytes()
        ));
    }
    Ok(())
}

fn require_finite(buf: &GpuBuffer, n: usize, what: &str) -> Result<(), String> {
    let values = buf.try_read_f32()?;
    if values.len() < n {
        return Err(format!("{what}: buffer holds {} floats, need {n}", values.len()));
    }
    for (i, v) in values.iter().take(n).enumerate() {
        if !v.is_finite() {
            return Err(format!("{what}: non-finite value at {i}"));
        }
    }
    Ok(())
}

fn reject_write_alias(entry: &str, writes: &[(&str, &GpuBuffer)], reads: &[(&str, &GpuBuffer)]) -> Result<(), String> {
    for (i, (left_name, left)) in writes.iter().enumerate() {
        for (right_name, right) in writes.iter().skip(i + 1) {
            if left.aliases(right) {
                return Err(format!(
                    "{entry}: writable buffers {left_name} and {right_name} overlap"
                ));
            }
        }
        for (read_name, read) in reads {
            if left.aliases(read) {
                return Err(format!(
                    "{entry}: writable buffer {left_name} overlaps read-only buffer {read_name}"
                ));
            }
        }
    }
    Ok(())
}

fn dispatch_groups(
    rt: &Arc<GpuRuntime>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    groups: usize,
    threads_per_tg: usize,
    encode: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    crate::nn::dispatch_tg_1d(rt, pipeline, groups, threads_per_tg, None, encode)
}

fn dispatch_1d(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    n: usize,
    encode: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    dispatch::dispatch_1d(rt, pipeline, n, encode)
}
