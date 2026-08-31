//! GEMM dispatch: TensorOps `matmul2d` (preferred) or `simdgroup_matrix` fallback.
//!
//! GEMM v2: Morton 1D TG walk, packed zero+matmul (one binder), MLP/bf16 split-K,
//! `execution_simdgroups<4>` on bf16/relaxed kernels (see matmul_tensorops.metal).
//!
//! Phase H: `PrecisionMode::Bf16` uses bf16 TensorOps GEMMs (f32 accumulate).
//! Callers may keep persistent bf16 activation/weight buffers; `ensure_bf16`
//! is a no-op when the operand is already bf16. Residual/RMSNorm/CE stay f32.
//! Optional `relaxed_precision` (tf32-class) on f32 GEMMs as a bridge; off by
//! default for golden parity.

// A GEMM dispatch takes A, B, C, four extents and a layout flag. Bundling
// them into a parameter struct would add a type whose only purpose is to
// satisfy a lint, and every call site would immediately destructure it.
#![allow(clippy::too_many_arguments)]

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;

use crate::runtime::{mtl_size, GpuRuntime, PrecisionMode};
use crate::tensor::{DType, Tensor};

#[derive(Clone, Copy)]
enum Layout {
    NN,
    TN,
    NT,
}

/// All public GEMM paths validate before casting, allocating scratch, or encoding.
/// MPP uses signed 32-bit extents/offset arithmetic; reject larger matrices.
fn validate_gemm(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    layout: Layout,
    allow_bf16: bool,
) -> Result<(usize, usize, usize), String> {
    for t in [a, b, c] {
        t.validate()?;
        if t.shape.len() != 2 || t.shape.contains(&0) {
            return Err("GEMM requires nonempty rank-2 tensors".into());
        }
        if t.numel() > i32::MAX as usize {
            return Err("GEMM exceeds signed 32-bit kernel indexing".into());
        }
    }
    if !std::sync::Arc::ptr_eq(a.runtime(), b.runtime())
        || !std::sync::Arc::ptr_eq(a.runtime(), c.runtime())
    {
        return Err("GEMM tensors must belong to the same runtime".into());
    }
    if c.dtype != DType::F32 || (!allow_bf16 && (a.dtype != DType::F32 || b.dtype != DType::F32)) {
        return Err("GEMM operand dtype does not match the selected precision path".into());
    }
    let (m, k, k2, n) = match layout {
        Layout::NN => (a.shape[0], a.shape[1], b.shape[0], b.shape[1]),
        Layout::TN => (a.shape[1], a.shape[0], b.shape[0], b.shape[1]),
        Layout::NT => (a.shape[0], a.shape[1], b.shape[1], b.shape[0]),
    };
    if k != k2 || c.shape != [m, n] {
        return Err("GEMM inner dimensions or output shape do not match".into());
    }
    if a.overlaps(c) || b.overlaps(c) {
        return Err("GEMM output must not overlap either input".into());
    }
    Ok((m, n, k))
}

/// Tall-K / small-MN → split-K accumulate.
/// Attn dW: M=N=128, K=BT=4096. MLP dW: one side = mlp_dim=384.
fn prefer_tn_splitk(m: usize, n: usize, k: usize) -> bool {
    k >= 2048 && m <= 384 && n <= 384 && m.min(n) <= 128
}

/// Tile sizes for TensorOps kernels (must match matmul_tensorops.metal).
#[derive(Clone, Copy)]
struct TileGeom {
    sm: usize,
    sn: usize,
    /// Simdgroups per TG (`execution_simdgroups<N>`). Exact f32 uses 1.
    simdgroups: usize,
}

const TILE_F32: TileGeom = TileGeom {
    sm: 32,
    sn: 32,
    simdgroups: 1,
};
/// Split-K bf16 kernels only; the plain bf16/relaxed NN/TN/NT lanes use the
/// cooperative-destination geometries below.
const TILE_V2: TileGeom = TileGeom {
    sm: 64,
    sn: 32,
    simdgroups: 4,
};
/// Coop TN/NT descriptor kernels (128×64 sg4; bench/results/
/// bf16_tnnt_coop_m5pro.txt: 1.5–2.0× over the multiply single-run kernels).
const TILE_COOP_TN_NT: TileGeom = TileGeom {
    sm: 128,
    sn: 64,
    simdgroups: 4,
};
/// Coop accumulate kernels (64×64 sg4; load-add-store, 1.4–1.5× over
/// multiply_accumulate at bandwidth-bound shapes).
const TILE_COOP_ACCUM: TileGeom = TileGeom {
    sm: 64,
    sn: 64,
    simdgroups: 4,
};

/// Exact 1D TG count for a `tiles_n × tiles_m` rectangle (no power-of-two pad —
/// padding blew up tall NN shapes like BT×C and erased the binder win).
fn morton_tg_count(tiles_n: usize, tiles_m: usize) -> usize {
    tiles_n.saturating_mul(tiles_m).max(1)
}

/// Live TN/NT TensorOps descriptors (transpose_left/right). Fixed multi-tile
/// slice axes: TN slices A's M on dim0; NT slices B's N on dim1.
const USE_TN_NT_DESCRIPTORS: bool = true;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmBackend {
    /// MPP TensorOps `matmul2d` (Metal 4 / macOS 26+, M5 accelerators).
    TensorOps,
    /// Hand-tiled `simdgroup_matrix` portable path.
    Simdgroup,
}

impl GemmBackend {
    pub fn kernel_name_f32(self) -> &'static str {
        match self {
            GemmBackend::TensorOps => "matmul2d_tensorops_f32",
            GemmBackend::Simdgroup => "matmul_simdgroup_f32",
        }
    }
}

/// Pick TensorOps when the metallib contains it; else simdgroup.
pub fn select_backend(rt: &GpuRuntime) -> GemmBackend {
    if rt.has_tensorops() {
        GemmBackend::TensorOps
    } else {
        GemmBackend::Simdgroup
    }
}

fn validate_cast_input(src: &Tensor, dtype: DType) -> Result<(), String> {
    src.validate()?;
    if src.dtype != dtype || src.numel() == 0 || src.numel() > u32::MAX as usize {
        return Err("cast requires the declared dtype and 1..=u32::MAX elements".into());
    }
    Ok(())
}

/// Cast f32 tensor → bf16 (GPU). Used at GEMM boundaries under `PrecisionMode::Bf16`.
pub fn cast_f32_to_bf16(src: &Tensor) -> Result<Tensor, String> {
    validate_cast_input(src, DType::F32)?;
    let rt = src.runtime();
    let dst = rt.alloc_tensor_bf16(&src.shape)?;
    cast_f32_to_bf16_into(src, &dst)?;
    Ok(dst)
}

/// Cast into an existing bf16 buffer (persistent weight banks).
pub fn cast_f32_to_bf16_into(src: &Tensor, dst: &Tensor) -> Result<(), String> {
    validate_cast_input(src, DType::F32)?;
    dst.validate()?;
    if dst.dtype != DType::BF16
        || src.shape != dst.shape
        || !std::sync::Arc::ptr_eq(src.runtime(), dst.runtime())
        || src.overlaps(dst)
    {
        return Err(
            "cast destination must match shape/runtime, be bf16, and not overlap source".into(),
        );
    }
    let rt = src.runtime();
    let p = rt.pipeline("cast_f32_to_bf16")?;
    let n = src.numel();
    crate::dispatch::dispatch_1d(rt, &p, n, |bnd| {
        crate::dispatch::set_tensor(bnd, src, 0);
        crate::dispatch::set_tensor(bnd, dst, 1);
        crate::dispatch::set_u32(bnd, n as u32, 2);
    })?;
    Ok(())
}

/// Hot-resident bf16 clone of an f32 master (weights / EMA banks).
pub fn cast_f32_to_bf16_hot(src: &Tensor) -> Result<Tensor, String> {
    validate_cast_input(src, DType::F32)?;
    let rt = src.runtime();
    let dst = rt.alloc_tensor_bf16_hot(&src.shape)?;
    cast_f32_to_bf16_into(src, &dst)?;
    Ok(dst)
}

/// Cast bf16 tensor → f32 (GPU).
pub fn cast_bf16_to_f32(src: &Tensor) -> Result<Tensor, String> {
    validate_cast_input(src, DType::BF16)?;
    let rt = src.runtime();
    let dst = rt.alloc_tensor_f32(&src.shape)?;
    let p = rt.pipeline("cast_bf16_to_f32")?;
    let n = src.numel();
    crate::dispatch::dispatch_1d(rt, &p, n, |bnd| {
        crate::dispatch::set_tensor(bnd, src, 0);
        crate::dispatch::set_tensor(bnd, &dst, 1);
        crate::dispatch::set_u32(bnd, n as u32, 2);
    })?;
    Ok(dst)
}

/// `f32 -> f16`, allocating the destination.
pub fn cast_f32_to_f16(src: &Tensor) -> Result<Tensor, String> {
    src.validate()?;
    if src.dtype != DType::F32 {
        return Err("cast_f32_to_f16 expects an f32 source".into());
    }
    let rt = src.runtime();
    let dst = rt.alloc_tensor_f16(&src.shape)?;
    cast_between(src, &dst, "cast_f32_to_f16")?;
    Ok(dst)
}

/// `f16 -> f32`, allocating the destination.
pub fn cast_f16_to_f32(src: &Tensor) -> Result<Tensor, String> {
    src.validate()?;
    if src.dtype != DType::F16 {
        return Err("cast_f16_to_f32 expects an f16 source".into());
    }
    let rt = src.runtime();
    let dst = rt.alloc_tensor_f32(&src.shape)?;
    cast_between(src, &dst, "cast_f16_to_f32")?;
    Ok(dst)
}

/// Shared elementwise cast dispatch.
fn cast_between(src: &Tensor, dst: &Tensor, kernel: &str) -> Result<(), String> {
    let n = src.numel();
    if n > u32::MAX as usize {
        return Err(format!("{kernel}: element count exceeds uint indexing"));
    }
    let rt = src.runtime();
    let p = rt.pipeline(kernel)?;
    crate::dispatch::dispatch_1d(rt, &p, n, |bnd| {
        crate::dispatch::set_tensor(bnd, src, 0);
        crate::dispatch::set_tensor(bnd, dst, 1);
        crate::dispatch::set_u32(bnd, n as u32, 2);
    })
}

fn ensure_bf16(t: &Tensor) -> Result<Tensor, String> {
    match t.dtype {
        DType::BF16 => Ok(t.clone()),
        DType::F32 => cast_f32_to_bf16(t),
        // Deliberately not a conversion. f16 -> bf16 loses three mantissa bits
        // *and* changes the exponent range, so a silent one would degrade the
        // caller's operands to buy a code path they did not ask for. An f16
        // operand belongs on the f16 GEMM.
        DType::F16 => Err(
            "bf16 GEMM was asked for f16 operands; convert explicitly, or use \
             the f16 kernels, which accumulate in f32 just as bf16 does"
                .into(),
        ),
    }
}

fn use_bf16_gemm(rt: &GpuRuntime, backend: GemmBackend) -> bool {
    rt.precision() == PrecisionMode::Bf16 && backend == GemmBackend::TensorOps && rt.has_tensorops()
}

fn use_relaxed_f32(rt: &GpuRuntime, backend: GemmBackend) -> bool {
    rt.relaxed_precision()
        && rt.precision() == PrecisionMode::F32
        && backend == GemmBackend::TensorOps
        && rt.has_tensorops()
}

/// `C[M,N] = A[M,K] @ B[K,N]`. Overwrites C.
///
/// - f32×f32→f32 always supported (exact or relaxed via runtime flag)
/// - bf16×bf16→f32 accum (C must be f32) via TensorOps when available
pub fn gemm(a: &Tensor, b: &Tensor, c: &Tensor, backend: GemmBackend) -> Result<(), String> {
    let (m, n, k) = validate_gemm(a, b, c, Layout::NN, true)?;

    let use_bf16 = a.dtype == DType::BF16 && b.dtype == DType::BF16;
    let use_f16 = a.dtype == DType::F16 && b.dtype == DType::F16;
    let narrow = use_bf16 || use_f16;
    if a.dtype != b.dtype || (narrow && backend != GemmBackend::TensorOps) {
        return Err("GEMM requires matching operand dtypes; bf16 and f16 require TensorOps".into());
    }

    let rt = a.runtime();
    let elem = if use_bf16 {
        CoopElem::Bf16
    } else if use_f16 {
        CoopElem::F16
    } else {
        CoopElem::RelaxedF32
    };
    match backend {
        // Cooperative-destination NN kernels (bf16, f16 and relaxed f32):
        // register accumulator, C written exactly once — no zero pre-pass.
        GemmBackend::TensorOps if narrow || use_relaxed_f32(rt, backend) => {
            let (kernel, tile) = nn_coop_kernel(m, n, k, elem);
            let pipeline = rt.pipeline(kernel)?;
            dispatch_tensorops_nn_coop(rt, &pipeline, a, b, c, m, n, k, tile)?;
        }
        GemmBackend::TensorOps => {
            let pipeline = rt.pipeline(backend.kernel_name_f32())?;
            // Zero-tax: pack C-zero + matmul into one binder (~−1 binder/GEMM).
            dispatch_tensorops_nn(rt, &pipeline, a, b, c, m, n, k, TILE_F32)?;
        }
        GemmBackend::Simdgroup => {
            let kernel = if m % 16 != 0 || n % 16 != 0 || k % 8 != 0 {
                "matmul_simdgroup_edges_f32"
            } else {
                backend.kernel_name_f32()
            };
            let pipeline = rt.pipeline(kernel)?;
            // Both simdgroup kernels overwrite every logical output element.
            // No pre-zero dispatch or barrier is needed (including offset views).
            let m_u = m as u32;
            let n_u = n as u32;
            let k_u = k as u32;
            let (tg_w, tg_h, tpt) = threadgroup_geometry_simdgroup(&pipeline, m, n);
            rt.with_binder(|bnd| {
                bnd.set_pipeline(&pipeline);
                bnd.bind_buf(a.buffer.metal(), a.byte_offset, 0);
                bnd.bind_buf(b.buffer.metal(), b.byte_offset, 1);
                bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
                bnd.bind_u32(m_u, 3);
                bnd.bind_u32(n_u, 4);
                bnd.bind_u32(k_u, 5);
                bnd.dispatch(mtl_size(tg_w, tg_h, 1), mtl_size(tpt, 1, 1));
                Ok(())
            })?;
        }
    }

    Ok(())
}

/// Element strides between consecutive batch elements.
///
/// A **zero** stride is the point of having three of these rather than one: it
/// broadcasts that operand across the batch. Batched activations against a
/// single shared weight matrix — the common shape — is `stride_b: 0`, which
/// needs no copies of B at all.
#[derive(Clone, Copy, Debug)]
pub struct BatchStrides {
    /// Elements between consecutive A matrices. Usually `m * k`.
    pub a: usize,
    /// Elements between consecutive B matrices. `0` broadcasts one B.
    pub b: usize,
    /// Elements between consecutive C matrices. Usually `m * n`.
    pub c: usize,
}

impl BatchStrides {
    /// Contiguous batches of all three operands.
    pub fn contiguous(m: usize, n: usize, k: usize) -> Self {
        Self {
            a: m * k,
            b: k * n,
            c: m * n,
        }
    }

    /// Contiguous A and C against one shared B.
    pub fn shared_b(m: usize, n: usize, k: usize) -> Self {
        Self {
            a: m * k,
            b: 0,
            c: m * n,
        }
    }
}

/// A batched GEMM's shape: the dimensions of one matrix, how many there are,
/// and how to step between them.
///
/// The per-matrix dimensions are given explicitly rather than read off the
/// tensors' shapes. A rank-2 shape cannot express a batch — `[batch * m, k]`
/// and `[m, k]` are the same tensor to a shape check — and a zero stride makes
/// it worse, since a broadcast B is genuinely `[k, n]` while A is not. Stating
/// the dimensions removes the guess.
#[derive(Clone, Copy, Debug)]
pub struct BatchedGemm {
    /// Rows of one A, and of one C.
    pub m: usize,
    /// Columns of one B, and of one C.
    pub n: usize,
    /// The contracted dimension.
    pub k: usize,
    /// How many matrices.
    pub batch: usize,
    /// Element steps between consecutive matrices.
    pub strides: BatchStrides,
}

/// `C[i] = A[i] @ B[i]` for `spec.batch` matrices, in one dispatch.
///
/// The batch is the grid's second dimension, so batching costs a pointer offset
/// per threadgroup and nothing else — the tile geometry, the register
/// accumulator and the swizzle are the single-matrix path's, and each element
/// is bit-identical to the [`gemm`] that would have produced it.
///
/// Requires the cooperative-destination path (bf16, f16, or f32 with relaxed
/// precision on TensorOps), for the same reason [`gemm_epilogue`] does.
///
/// `a`, `b` and `c` point at the *first* matrix; [`BatchStrides`] reaches the
/// rest. Every operand's last element is bounds checked against its buffer,
/// because an over-long batch reads past the end of device memory rather than
/// failing.
pub fn gemm_batched(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
    spec: BatchedGemm,
) -> Result<(), String> {
    let BatchedGemm {
        m,
        n,
        k,
        batch,
        strides,
    } = spec;
    if batch == 0 {
        return Ok(());
    }
    if m == 0 || n == 0 || k == 0 {
        return Err("batched GEMM requires nonzero m, n and k".into());
    }
    for t in [a, b, c] {
        t.validate()?;
        if t.numel() > i32::MAX as usize {
            return Err("batched GEMM exceeds signed 32-bit kernel indexing".into());
        }
    }
    if !std::sync::Arc::ptr_eq(a.runtime(), b.runtime())
        || !std::sync::Arc::ptr_eq(a.runtime(), c.runtime())
    {
        return Err("batched GEMM tensors must belong to the same runtime".into());
    }
    if c.dtype != DType::F32 {
        return Err("batched GEMM writes f32 output".into());
    }
    if a.dtype != b.dtype {
        return Err("GEMM requires matching operand dtypes".into());
    }
    let rt = a.runtime();
    let use_bf16 = a.dtype == DType::BF16 && b.dtype == DType::BF16;
    let use_f16 = a.dtype == DType::F16 && b.dtype == DType::F16;
    if backend != GemmBackend::TensorOps || !(use_bf16 || use_f16 || use_relaxed_f32(rt, backend)) {
        return Err(
            "batched GEMM needs the cooperative-destination path: bf16 or f16 operands, \
             or f32 with relaxed precision, on the TensorOps backend"
                .into(),
        );
    }

    // The last batch element must fit. Without this the kernel walks off the
    // end of whichever operand was sized for a smaller batch and reads whatever
    // happens to be resident.
    for (t, first, stride, what) in [
        (a, m * k, strides.a, "A"),
        (b, k * n, strides.b, "B"),
        (c, m * n, strides.c, "C"),
    ] {
        let need = stride
            .checked_mul(batch - 1)
            .and_then(|off| off.checked_add(first))
            .ok_or_else(|| format!("batched GEMM: {what} extent overflows usize"))?;
        if t.numel() < need {
            return Err(format!(
                "batched GEMM: {what} holds {} elements but batch {batch} at stride \
                 {stride} reaches {need}",
                t.numel()
            ));
        }
        if stride > u32::MAX as usize {
            return Err(format!("batched GEMM: {what} stride exceeds uint indexing"));
        }
    }

    if batch == 1 {
        // Nothing to batch; the single-matrix kernel is already the best
        // implementation and keeps the documented bit-identity trivially true.
        let elem = coop_elem(use_bf16, use_f16);
        let (kernel, tile) = nn_coop_kernel(m, n, k, elem);
        let pipeline = rt.pipeline(kernel)?;
        return dispatch_tensorops_nn_coop(rt, &pipeline, a, b, c, m, n, k, tile);
    }

    let kernel = match coop_elem(use_bf16, use_f16) {
        CoopElem::Bf16 => "matmul2d_tensorops_bf16_f32_batched",
        CoopElem::F16 => "matmul2d_tensorops_f16_f32_batched",
        CoopElem::RelaxedF32 => "matmul2d_tensorops_f32_relaxed_batched",
    };
    let pipeline = rt.pipeline(kernel)?;
    // Only the 128x64 geometry is instantiated batched; a narrow variant is a
    // tuning question left until measured, as for the epilogue.
    let tile = TILE_COOP_DEFAULT;
    let tiles_n = n.div_ceil(tile.sn);
    let tiles_m = m.div_ceil(tile.sm);
    let tg = morton_tg_count(tiles_n, tiles_m);
    let tpt = threads_per_tg(&pipeline, tile);
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&pipeline);
        bnd.bind_buf(a.buffer.metal(), a.byte_offset, 0);
        bnd.bind_buf(b.buffer.metal(), b.byte_offset, 1);
        bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.bind_u32(tiles_m as u32, 7);
        bnd.bind_u32(strides.a as u32, 8);
        bnd.bind_u32(strides.b as u32, 9);
        bnd.bind_u32(strides.c as u32, 10);
        bnd.dispatch(mtl_size(tg, batch, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

fn coop_elem(use_bf16: bool, use_f16: bool) -> CoopElem {
    if use_bf16 {
        CoopElem::Bf16
    } else if use_f16 {
        CoopElem::F16
    } else {
        CoopElem::RelaxedF32
    }
}

/// Activation fused into a GEMM epilogue.
///
/// The discriminants are ABI: they cross to `GemmActivation` in
/// `matmul_tensorops.metal` as a `uint`, so reordering them changes what every
/// caller computes without changing any Rust that reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Activation {
    /// No activation; the epilogue is scale, accumulate and bias only.
    #[default]
    None = 0,
    /// `max(x, 0)`.
    Relu = 1,
    /// `gelu_pytorch_tanh`, in the same clamped `precise::tanh` formulation as
    /// `nn::mlp_gelu_tanh`. Deliberately not a second derivation: at `-O2` MSL
    /// lowers plain `tanh` to `air.fast_tanh`, which NaNs past roughly |10|.
    GeluTanh = 2,
    /// `x * sigmoid(x)`, matching `nn::mlp_silu`.
    Silu = 3,
}

/// What a fused GEMM epilogue applies to the accumulator before it is stored.
///
/// `C = activation(alpha * (A @ B) + beta * C_prev + bias)`
///
/// # Why fuse
///
/// Every term here is otherwise a separate dispatch that reads all of `C` and
/// writes all of `C`. A bias-plus-GELU costs two extra full round-trips through
/// device memory, which on a bandwidth-bound machine is most of what the GEMM
/// saved. Applied here the accumulator is still in registers, so `C` is written
/// exactly once — and read at most once, only when `beta != 0`.
///
/// `bias` is per-column, length `N`, broadcast across every row. It is read
/// through a row-stride-0 tensor view, so the same cooperative `load` path that
/// fetches `C_prev` fetches the bias with no separate indexing.
#[derive(Clone, Copy, Debug)]
pub struct Epilogue<'a> {
    /// Scale on the product. `1.0` is the identity.
    pub alpha: f32,
    /// Scale on `C`'s prior contents. `0.0` skips reading `C` entirely, which
    /// is a bandwidth saving and not merely an arithmetic one.
    pub beta: f32,
    /// Per-column bias of length `N`, or `None`.
    pub bias: Option<&'a Tensor>,
    /// Activation applied last, after scale, accumulate and bias.
    pub activation: Activation,
}

impl Default for Epilogue<'_> {
    /// The identity epilogue: `C = A @ B`.
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.0,
            bias: None,
            activation: Activation::None,
        }
    }
}

impl Epilogue<'_> {
    /// Whether this epilogue would change the result at all.
    ///
    /// A caller passing the identity is dispatched to the plain kernel rather
    /// than paying for an epilogue that computes `C = 1.0 * C + 0.0`.
    pub fn is_identity(&self) -> bool {
        self.alpha == 1.0
            && self.beta == 0.0
            && self.bias.is_none()
            && self.activation == Activation::None
    }
}

/// `C = activation(alpha * (A @ B) + beta * C + bias)`, in one dispatch.
///
/// The fused form of a GEMM followed by a scale, an accumulate, a bias add and
/// an activation. See [`Epilogue`] for why that matters.
///
/// Requires the cooperative-destination path — bf16 operands, or f32 with
/// [`GpuRuntime::set_relaxed_precision`] on — because that is the path that holds the
/// accumulator in registers. An f32 exact GEMM has nowhere to fuse into and is
/// refused rather than silently falling back to separate dispatches, which
/// would make the call quietly slower than the unfused code it replaced.
///
/// An identity epilogue dispatches to the plain [`gemm`].
pub fn gemm_epilogue(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
    epi: Epilogue<'_>,
) -> Result<(), String> {
    if epi.is_identity() {
        return gemm(a, b, c, backend);
    }
    let (m, n, k) = validate_gemm(a, b, c, Layout::NN, true)?;
    if !epi.alpha.is_finite() || !epi.beta.is_finite() {
        return Err(format!(
            "GEMM epilogue: alpha and beta must be finite, got alpha={} beta={}",
            epi.alpha, epi.beta
        ));
    }

    let use_bf16 = a.dtype == DType::BF16 && b.dtype == DType::BF16;
    let use_f16 = a.dtype == DType::F16 && b.dtype == DType::F16;
    if a.dtype != b.dtype {
        return Err("GEMM requires matching operand dtypes".into());
    }
    let rt = a.runtime();
    if !(use_bf16 || use_f16 || use_relaxed_f32(rt, backend)) || backend != GemmBackend::TensorOps {
        return Err(
            "GEMM epilogue needs the cooperative-destination path: bf16 operands, or f32              with PrecisionMode::Relaxed, on the TensorOps backend. The exact-f32 and              simdgroup kernels write C straight from the matmul with no register              accumulator to fuse into, so there is nothing here to make faster"
                .into(),
        );
    }

    if let Some(bias) = epi.bias {
        bias.validate()?;
        if bias.dtype != DType::F32 {
            return Err("GEMM epilogue: bias must be f32".into());
        }
        if !std::sync::Arc::ptr_eq(rt, bias.runtime()) {
            return Err("GEMM epilogue: bias belongs to a different runtime".into());
        }
        if bias.numel() < n {
            return Err(format!(
                "GEMM epilogue: bias is per-column and must hold at least N = {n}                  elements, got {}",
                bias.numel()
            ));
        }
    }

    let kernel = if use_bf16 {
        "matmul2d_tensorops_bf16_f32_epi"
    } else if use_f16 {
        "matmul2d_tensorops_f16_f32_epi"
    } else {
        "matmul2d_tensorops_f32_relaxed_epi"
    };
    let pipeline = rt.pipeline(kernel)?;
    // Only the 128x64 sg4 geometry has an epilogue instantiation. The narrow
    // 64x64 variant exists for shapes the tile tune found it better on; adding
    // an epilogue copy of it is a tuning question, not a correctness one, and
    // is deliberately left until measured.
    let tile = TILE_COOP_DEFAULT;
    let tiles_n = n.div_ceil(tile.sn);
    let tiles_m = m.div_ceil(tile.sm);
    let tg = morton_tg_count(tiles_n, tiles_m);
    let tpt = threads_per_tg(&pipeline, tile);
    let (alpha, beta, act) = (epi.alpha, epi.beta, epi.activation as u32);
    let has_bias = u32::from(epi.bias.is_some());
    // Buffer 8 is read unconditionally by the kernel binding, so it must be
    // bound even when unused: Metal faults on a declared-but-unbound buffer.
    // `has_bias` is what decides whether it is dereferenced.
    let bias_buf = epi.bias.unwrap_or(c);
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&pipeline);
        bnd.bind_buf(a.buffer.metal(), a.byte_offset, 0);
        bnd.bind_buf(b.buffer.metal(), b.byte_offset, 1);
        bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.bind_u32(tiles_m as u32, 7);
        bnd.bind_buf(bias_buf.buffer.metal(), bias_buf.byte_offset, 8);
        bnd.bind_f32(alpha, 9);
        bnd.bind_f32(beta, 10);
        bnd.bind_u32(act, 11);
        bnd.bind_u32(has_bias, 12);
        bnd.dispatch(mtl_size(tg, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

/// Cooperative-destination NN tile geometries (must match the NN_COOP_KERNEL
/// instantiations in matmul_tensorops.metal).
const TILE_COOP_DEFAULT: TileGeom = TileGeom {
    sm: 128,
    sn: 64,
    simdgroups: 4,
};
const TILE_COOP_NARROW: TileGeom = TileGeom {
    sm: 64,
    sn: 64,
    simdgroups: 4,
};

/// Shape → coop NN kernel, from the 2026-08-30 M5 Pro tile tunes
/// (bench/results/bf16_tile_tune_m5pro_coop.txt, bf16_tnnt_coop_m5pro.txt):
/// 64×64 sg4 wins narrow-N shapes by ~6%; 128×64 sg4 everything else. The
/// in-kernel column-panel swizzle covers the huge-square case (+11% at
/// 4096³), which retired the earlier 256×64 sg8 wide entry it outran.
/// Operand element type for the cooperative-destination NN kernels.
///
/// A three-way choice rather than the boolean this used to be: f16 and bf16
/// are both two bytes and both accumulate in f32, but their bit layouts differ,
/// so picking the wrong kernel is silently wrong rather than merely slower.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CoopElem {
    RelaxedF32,
    Bf16,
    F16,
}

fn nn_coop_kernel(_m: usize, n: usize, _k: usize, elem: CoopElem) -> (&'static str, TileGeom) {
    if n <= 512 {
        (
            match elem {
                CoopElem::Bf16 => "matmul2d_tensorops_bf16_f32_64x64_sg4",
                CoopElem::F16 => "matmul2d_tensorops_f16_f32_64x64_sg4",
                CoopElem::RelaxedF32 => "matmul2d_tensorops_f32_relaxed_64x64_sg4",
            },
            TILE_COOP_NARROW,
        )
    } else {
        (
            match elem {
                CoopElem::Bf16 => "matmul2d_tensorops_bf16_f32",
                CoopElem::F16 => "matmul2d_tensorops_f16_f32",
                CoopElem::RelaxedF32 => "matmul2d_tensorops_f32_relaxed",
            },
            TILE_COOP_DEFAULT,
        )
    }
}

/// Single-dispatch NN matmul for the cooperative-destination kernels: the
/// kernel overwrites every in-bounds C element (register accumulator plus a
/// bounds-checked store), so there is no zero pre-pass to pack.
fn dispatch_tensorops_nn_coop(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    n: usize,
    k: usize,
    tile: TileGeom,
) -> Result<(), String> {
    let tiles_n = n.div_ceil(tile.sn);
    let tiles_m = m.div_ceil(tile.sm);
    let tg = morton_tg_count(tiles_n, tiles_m);
    let tpt = threads_per_tg(pipeline, tile);
    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        bnd.bind_buf(a.buffer.metal(), a.byte_offset, 0);
        bnd.bind_buf(b.buffer.metal(), b.byte_offset, 1);
        bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.bind_u32(tiles_m as u32, 7);
        bnd.dispatch(mtl_size(tg, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

/// Pack `zero_f32(C)` + TensorOps NN matmul into a single Metal 4 binder.
fn dispatch_tensorops_nn(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    n: usize,
    k: usize,
    tile: TileGeom,
) -> Result<(), String> {
    let zero_p = rt.pipeline("zero_f32")?;
    let numel = c.numel();
    let tiles_n = n.div_ceil(tile.sn);
    let tiles_m = m.div_ceil(tile.sm);
    let tg = morton_tg_count(tiles_n, tiles_m);
    let tpt = threads_per_tg(pipeline, tile);
    let z_width = zero_p.threadExecutionWidth();
    let z_tpt = z_width.min(numel).max(1);
    let z_groups = numel.div_ceil(z_tpt);

    rt.with_binder(|bnd| {
        bnd.set_pipeline(&zero_p);
        bnd.bind_tensor(c, 0);
        bnd.bind_u32(numel as u32, 1);
        bnd.dispatch(mtl_size(z_groups, 1, 1), mtl_size(z_tpt, 1, 1));
        // Explicit barrier only when auto per-dispatch barriers are off.
        // Ask the binder, not the global flag — the binder's latched mode is
        // what decided whether the zero dispatch already got a barrier.
        if bnd.needs_explicit_barriers() {
            bnd.barrier();
        }

        bnd.set_pipeline(pipeline);
        bnd.bind_buf(a.buffer.metal(), a.byte_offset, 0);
        bnd.bind_buf(b.buffer.metal(), b.byte_offset, 1);
        bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.bind_u32(tiles_m as u32, 7);
        // f32 exact NN/TN/NT read buffer(8); bf16/relaxed ignore extra bind.
        bnd.bind_u32(
            if crate::ab_flags::gemm_interior_offsets() {
                1
            } else {
                0
            },
            8,
        );
        bnd.dispatch(mtl_size(tg, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

fn dispatch_tensorops_tn_nt(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    n: usize,
    k: usize,
    tile: TileGeom,
) -> Result<(), String> {
    // Same binder packing as NN.
    dispatch_tensorops_nn(rt, pipeline, a, b, c, m, n, k, tile)
}

/// TensorOps matmul with `mode::multiply_accumulate` — no C zero (1 binder).
fn dispatch_tensorops_accum(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    n: usize,
    k: usize,
    tile: TileGeom,
    bind_interior: bool,
) -> Result<(), String> {
    let tiles_n = n.div_ceil(tile.sn);
    let tiles_m = m.div_ceil(tile.sm);
    let tg = morton_tg_count(tiles_n, tiles_m);
    let tpt = threads_per_tg(pipeline, tile);

    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        bnd.bind_buf(a.buffer.metal(), a.byte_offset, 0);
        bnd.bind_buf(b.buffer.metal(), b.byte_offset, 1);
        bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.bind_u32(tiles_m as u32, 7);
        if bind_interior {
            bnd.bind_u32(
                if crate::ab_flags::gemm_interior_offsets() {
                    1
                } else {
                    0
                },
                8,
            );
        }
        bnd.dispatch(mtl_size(tg, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

/// Convenience: f32 GEMM (parity path). Honors `relaxed_precision` when set.
pub fn gemm_f32(a: &Tensor, b: &Tensor, c: &Tensor, backend: GemmBackend) -> Result<(), String> {
    gemm(a, b, c, backend)
}

/// Training GEMM: under `PrecisionMode::Bf16` uses bf16 TensorOps (f32 accum into
/// `c`). Already-bf16 operands skip cast (persistent bf16 activations/weights).
/// Falls back to f32 GEMM when TensorOps is absent.
pub fn gemm_train(a: &Tensor, b: &Tensor, c: &Tensor, backend: GemmBackend) -> Result<(), String> {
    validate_gemm(a, b, c, Layout::NN, true)?;
    let rt = a.runtime();
    if use_bf16_gemm(rt, backend) {
        let a_bf = ensure_bf16(a)?;
        let b_bf = ensure_bf16(b)?;
        assert_eq!(c.dtype, DType::F32);
        return gemm(&a_bf, &b_bf, c, backend);
    }
    gemm_f32(a, b, c, backend)
}

/// `C[M,N] = A[K,M]^T @ B[K,N]` (TN). A is stored `[K,M]`, B `[K,N]`.
pub fn gemm_tn_f32(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    let (m, n, k) = validate_gemm(a_km, b_kn, c, Layout::TN, false)?;

    if USE_TN_NT_DESCRIPTORS && backend == GemmBackend::TensorOps && a_km.runtime().has_tensorops()
    {
        if prefer_tn_splitk(m, n, k) {
            return gemm_tn_splitk_f32(a_km, b_kn, c, k);
        }
        let rt = a_km.runtime();
        let pipeline = rt.pipeline("matmul2d_tensorops_tn_f32")?;
        return dispatch_tensorops_tn_nt(rt, &pipeline, a_km, b_kn, c, m, n, k, TILE_F32);
    }

    // Default: explicit transpose + NN (golden-safe).
    let at = {
        let rt = a_km.runtime();
        let out = rt.alloc_temp_f32(&[m, k])?;
        let p = rt.pipeline("transpose2d_f32")?;
        crate::dispatch::dispatch_1d(rt, &p, m * k, |bnd| {
            crate::dispatch::set_tensor(bnd, a_km, 0);
            crate::dispatch::set_tensor(bnd, &out, 1);
            crate::dispatch::set_u32(bnd, k as u32, 2);
            crate::dispatch::set_u32(bnd, m as u32, 3);
        })?;
        out
    };
    gemm_f32(&at, b_kn, c, backend)
}

/// Training TN GEMM — bf16 TensorOps descriptor when `PrecisionMode::Bf16`.
pub fn gemm_tn_train(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    validate_gemm(
        a_km,
        b_kn,
        c,
        Layout::TN,
        use_bf16_gemm(a_km.runtime(), backend),
    )?;
    let rt = a_km.runtime();
    if use_bf16_gemm(rt, backend) {
        assert_eq!(c.dtype, DType::F32);
        let a_bf = ensure_bf16(a_km)?;
        let b_bf = ensure_bf16(b_kn)?;
        let k = a_bf.shape[0];
        let m = a_bf.shape[1];
        let n = b_bf.shape[1];
        assert_eq!(c.shape, &[m, n]);
        if prefer_tn_splitk(m, n, k) {
            return gemm_tn_splitk_bf16(&a_bf, &b_bf, c, k);
        }
        // Coop kernel: register accumulator, C written once, no zero pre-pass.
        let pipeline = rt.pipeline("matmul2d_tensorops_tn_bf16_f32")?;
        return dispatch_tensorops_nn_coop(
            rt,
            &pipeline,
            &a_bf,
            &b_bf,
            c,
            m,
            n,
            k,
            TILE_COOP_TN_NT,
        );
    }
    gemm_tn_f32(a_km, b_kn, c, backend)
}

fn gemm_tn_splitk_f32(a_km: &Tensor, b_kn: &Tensor, c: &Tensor, k: usize) -> Result<(), String> {
    gemm_tn_splitk_f32_opts(a_km, b_kn, c, k, /*zero_first=*/ true)
}

fn gemm_tn_splitk_f32_opts(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    k: usize,
    zero_first: bool,
) -> Result<(), String> {
    let m = a_km.shape[1];
    let n = b_kn.shape[1];
    let rt = a_km.runtime();
    let pipeline = rt.pipeline("matmul2d_tensorops_tn_splitk_f32")?;
    let zero_p = rt.pipeline("zero_f32")?;
    let tile = TILE_F32;
    let tiles_n = n.div_ceil(tile.sn);
    let tiles_m = m.div_ceil(tile.sm);
    let tg = morton_tg_count(tiles_n, tiles_m);
    let tpt = threads_per_tg(&pipeline, tile);
    let numel = c.numel();
    let z_width = zero_p.threadExecutionWidth();
    let z_tpt = z_width.min(numel).max(1);
    let z_groups = numel.div_ceil(z_tpt);
    let k_tile = 256u32;
    let partitions: Vec<u32> = (0..k as u32).step_by(k_tile as usize).collect();

    // Zero once (optional) + all K-partitions in one binder.
    rt.with_binder(|bnd| {
        let need_explicit = bnd.needs_explicit_barriers();
        if zero_first {
            bnd.set_pipeline(&zero_p);
            bnd.bind_tensor(c, 0);
            bnd.bind_u32(numel as u32, 1);
            bnd.dispatch(mtl_size(z_groups, 1, 1), mtl_size(z_tpt, 1, 1));
            if need_explicit {
                bnd.barrier();
            }
        }

        bnd.set_pipeline(&pipeline);
        for (pi, &k0) in partitions.iter().enumerate() {
            if pi > 0 && need_explicit {
                bnd.barrier();
            }
            bnd.bind_buf(a_km.buffer.metal(), a_km.byte_offset, 0);
            bnd.bind_buf(b_kn.buffer.metal(), b_kn.byte_offset, 1);
            bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
            bnd.bind_u32(m as u32, 3);
            bnd.bind_u32(n as u32, 4);
            bnd.bind_u32(k as u32, 5);
            bnd.bind_u32(k0, 6);
            bnd.bind_u32(k_tile, 7);
            bnd.bind_u32(tiles_n as u32, 8);
            bnd.bind_u32(tiles_m as u32, 9);
            bnd.dispatch(mtl_size(tg, 1, 1), mtl_size(tpt, 1, 1));
        }
        Ok(())
    })?;
    Ok(())
}

fn gemm_tn_splitk_bf16(a_km: &Tensor, b_kn: &Tensor, c: &Tensor, k: usize) -> Result<(), String> {
    gemm_tn_splitk_bf16_opts(a_km, b_kn, c, k, /*zero_first=*/ true)
}

fn gemm_tn_splitk_bf16_opts(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    k: usize,
    zero_first: bool,
) -> Result<(), String> {
    let m = a_km.shape[1];
    let n = b_kn.shape[1];
    let rt = a_km.runtime();
    let pipeline = rt.pipeline("matmul2d_tensorops_tn_splitk_bf16_f32")?;
    let zero_p = rt.pipeline("zero_f32")?;
    let tile = TILE_V2;
    let tiles_n = n.div_ceil(tile.sn);
    let tiles_m = m.div_ceil(tile.sm);
    let tg = morton_tg_count(tiles_n, tiles_m);
    let tpt = threads_per_tg(&pipeline, tile);
    let numel = c.numel();
    let z_width = zero_p.threadExecutionWidth();
    let z_tpt = z_width.min(numel).max(1);
    let z_groups = numel.div_ceil(z_tpt);
    let k_tile = 256u32;
    let partitions: Vec<u32> = (0..k as u32).step_by(k_tile as usize).collect();

    rt.with_binder(|bnd| {
        let need_explicit = bnd.needs_explicit_barriers();
        if zero_first {
            bnd.set_pipeline(&zero_p);
            bnd.bind_tensor(c, 0);
            bnd.bind_u32(numel as u32, 1);
            bnd.dispatch(mtl_size(z_groups, 1, 1), mtl_size(z_tpt, 1, 1));
            if need_explicit {
                bnd.barrier();
            }
        }

        bnd.set_pipeline(&pipeline);
        for (pi, &k0) in partitions.iter().enumerate() {
            if pi > 0 && need_explicit {
                bnd.barrier();
            }
            bnd.bind_buf(a_km.buffer.metal(), a_km.byte_offset, 0);
            bnd.bind_buf(b_kn.buffer.metal(), b_kn.byte_offset, 1);
            bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
            bnd.bind_u32(m as u32, 3);
            bnd.bind_u32(n as u32, 4);
            bnd.bind_u32(k as u32, 5);
            bnd.bind_u32(k0, 6);
            bnd.bind_u32(k_tile, 7);
            bnd.bind_u32(tiles_n as u32, 8);
            bnd.bind_u32(tiles_m as u32, 9);
            bnd.dispatch(mtl_size(tg, 1, 1), mtl_size(tpt, 1, 1));
        }
        Ok(())
    })?;
    Ok(())
}

/// `C[M,N] = A[M,K] @ B[N,K]^T` (NT). B is stored `[N,K]` (e.g. `W[in,out]`).
pub fn gemm_nt_f32(
    a_mk: &Tensor,
    b_nk: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    let (m, n, k) = validate_gemm(a_mk, b_nk, c, Layout::NT, false)?;

    if USE_TN_NT_DESCRIPTORS && backend == GemmBackend::TensorOps && a_mk.runtime().has_tensorops()
    {
        let rt = a_mk.runtime();
        let pipeline = rt.pipeline("matmul2d_tensorops_nt_f32")?;
        return dispatch_tensorops_tn_nt(rt, &pipeline, a_mk, b_nk, c, m, n, k, TILE_F32);
    }

    let bt = {
        let rt = b_nk.runtime();
        let out = rt.alloc_temp_f32(&[k, n])?;
        let p = rt.pipeline("transpose2d_f32")?;
        crate::dispatch::dispatch_1d(rt, &p, n * k, |bnd| {
            crate::dispatch::set_tensor(bnd, b_nk, 0);
            crate::dispatch::set_tensor(bnd, &out, 1);
            crate::dispatch::set_u32(bnd, n as u32, 2);
            crate::dispatch::set_u32(bnd, k as u32, 3);
        })?;
        out
    };
    gemm_f32(a_mk, &bt, c, backend)
}

/// Training NT GEMM — bf16 TensorOps descriptor when `PrecisionMode::Bf16`.
pub fn gemm_nt_train(
    a_mk: &Tensor,
    b_nk: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    validate_gemm(
        a_mk,
        b_nk,
        c,
        Layout::NT,
        use_bf16_gemm(a_mk.runtime(), backend),
    )?;
    let rt = a_mk.runtime();
    if use_bf16_gemm(rt, backend) {
        assert_eq!(c.dtype, DType::F32);
        let a_bf = ensure_bf16(a_mk)?;
        let b_bf = ensure_bf16(b_nk)?;
        let m = a_bf.shape[0];
        let k = a_bf.shape[1];
        let n = b_bf.shape[0];
        assert_eq!(c.shape, &[m, n]);
        // Coop kernel: register accumulator, C written once, no zero pre-pass.
        let pipeline = rt.pipeline("matmul2d_tensorops_nt_bf16_f32")?;
        return dispatch_tensorops_nn_coop(
            rt,
            &pipeline,
            &a_bf,
            &b_bf,
            c,
            m,
            n,
            k,
            TILE_COOP_TN_NT,
        );
    }
    gemm_nt_f32(a_mk, b_nk, c, backend)
}

/// `C += A[K,M]^T @ B[K,N]` (TN accumulate). No C zero — for dW into grad banks
/// and dx accumulate into a pre-zeroed buffer.
pub fn gemm_tn_accum_train(
    a_km: &Tensor,
    b_kn: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    let (m, n, k) = validate_gemm(
        a_km,
        b_kn,
        c,
        Layout::TN,
        use_bf16_gemm(a_km.runtime(), backend),
    )?;

    let rt = a_km.runtime();
    let use_accum = crate::ab_flags::gemm_accum();
    if use_accum && use_bf16_gemm(rt, backend) {
        let a_bf = ensure_bf16(a_km)?;
        let b_bf = ensure_bf16(b_kn)?;
        if prefer_tn_splitk(m, n, k) {
            return gemm_tn_splitk_bf16_opts(&a_bf, &b_bf, c, k, /*zero_first=*/ false);
        }
        let pipeline = rt.pipeline("matmul2d_tensorops_tn_accum_bf16_f32")?;
        return dispatch_tensorops_accum(
            rt,
            &pipeline,
            &a_bf,
            &b_bf,
            c,
            m,
            n,
            k,
            TILE_COOP_ACCUM,
            /*bind_interior=*/ false,
        );
    }

    if use_accum && USE_TN_NT_DESCRIPTORS && backend == GemmBackend::TensorOps && rt.has_tensorops()
    {
        if prefer_tn_splitk(m, n, k) {
            return gemm_tn_splitk_f32_opts(a_km, b_kn, c, k, /*zero_first=*/ false);
        }
        let pipeline = rt.pipeline("matmul2d_tensorops_tn_accum_f32")?;
        return dispatch_tensorops_accum(
            rt, &pipeline, a_km, b_kn, c, m, n, k, TILE_F32, /*bind_interior=*/ true,
        );
    }

    // Fallback / Soft-bisect: temp + add (pre–Audit 6 P1a/P1a2 numerics).
    let tmp = rt.alloc_temp_f32(&[m, n])?;
    gemm_tn_train(a_km, b_kn, &tmp, backend)?;
    let p = rt.pipeline("add_inplace_f32")?;
    crate::dispatch::dispatch_1d(rt, &p, c.numel(), |bnd| {
        crate::dispatch::set_tensor(bnd, c, 0);
        crate::dispatch::set_tensor(bnd, &tmp, 1);
        crate::dispatch::set_u32(bnd, c.numel() as u32, 2);
    })?;
    Ok(())
}

/// `C += A[M,K] @ B[N,K]^T` (NT accumulate). No C zero.
///
/// All call sites are **dX-class** accumulations into fresh pre-zeroed
/// activation-grad buffers (never weight banks), so this path additionally
/// honors `METAL_NATIVE_GEMM_ACCUM_DX` — accumulate-mode dX with dW kept on
/// the Soft-safe temp+add path (Audit 7).
pub fn gemm_nt_accum_train(
    a_mk: &Tensor,
    b_nk: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<(), String> {
    let (m, n, k) = validate_gemm(
        a_mk,
        b_nk,
        c,
        Layout::NT,
        use_bf16_gemm(a_mk.runtime(), backend),
    )?;

    let rt = a_mk.runtime();
    let use_accum = crate::ab_flags::gemm_accum() || crate::ab_flags::gemm_accum_dx();
    if use_accum && use_bf16_gemm(rt, backend) {
        let a_bf = ensure_bf16(a_mk)?;
        let b_bf = ensure_bf16(b_nk)?;
        let pipeline = rt.pipeline("matmul2d_tensorops_nt_accum_bf16_f32")?;
        return dispatch_tensorops_accum(
            rt,
            &pipeline,
            &a_bf,
            &b_bf,
            c,
            m,
            n,
            k,
            TILE_COOP_ACCUM,
            /*bind_interior=*/ false,
        );
    }

    if use_accum && USE_TN_NT_DESCRIPTORS && backend == GemmBackend::TensorOps && rt.has_tensorops()
    {
        let pipeline = rt.pipeline("matmul2d_tensorops_nt_accum_f32")?;
        return dispatch_tensorops_accum(
            rt, &pipeline, a_mk, b_nk, c, m, n, k, TILE_F32, /*bind_interior=*/ true,
        );
    }

    let tmp = rt.alloc_temp_f32(&[m, n])?;
    gemm_nt_train(a_mk, b_nk, &tmp, backend)?;
    let p = rt.pipeline("add_inplace_f32")?;
    crate::dispatch::dispatch_1d(rt, &p, c.numel(), |bnd| {
        crate::dispatch::set_tensor(bnd, c, 0);
        crate::dispatch::set_tensor(bnd, &tmp, 1);
        crate::dispatch::set_u32(bnd, c.numel() as u32, 2);
    })?;
    Ok(())
}

/// Prefer bf16 / relaxed GEMM per runtime precision policy.
pub fn gemm_auto(a: &Tensor, b: &Tensor, c: &Tensor, backend: GemmBackend) -> Result<(), String> {
    gemm_train(a, b, c, backend)
}

fn threads_per_tg(pipeline: &ProtocolObject<dyn MTLComputePipelineState>, tile: TileGeom) -> usize {
    let width = pipeline.threadExecutionWidth();
    width * tile.simdgroups
}

fn threadgroup_geometry_simdgroup(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    m: usize,
    n: usize,
) -> (usize, usize, usize) {
    let width = pipeline.threadExecutionWidth();
    let tg_w = n.div_ceil(16);
    let tg_h = m.div_ceil(16);
    (tg_w, tg_h, width * 4)
}

/// CPU reference GEMM for tests.
pub fn gemm_f32_cpu(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuRuntime;
    use std::sync::Arc;

    /// TensorOps is a hard requirement, not an optional extra: tessl is
    /// Apple-silicon-only and its README requires Neural Accelerators, so a
    /// metallib without `matmul2d_tensorops_f32` is a broken build, not a
    /// configuration to tolerate. These tests used to `return` silently when the
    /// probe came back false, which made "skipped" and "passed" print the same
    /// `ok` — the entire TensorOps half of the suite could stop running without
    /// a single red line. Assert instead, the way `stress_tests` already does.
    fn tensorops_runtime() -> Arc<GpuRuntime> {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        assert!(
            rt.has_tensorops(),
            "matmul2d_tensorops_f32 missing from the metallib on device {}: \
             tessl requires Neural Accelerators, so this is a broken build \
             (rebuild kernels via build.rs), not a testable configuration",
            rt.device_name()
        );
        rt
    }

    /// Same rule one level down: a metallib that loaded but lacks the specific
    /// kernel a test drives means build.rs emitted a stale or partial kernel
    /// set. That must fail, not vacuously pass.
    fn require_pipeline(rt: &GpuRuntime, name: &str) {
        assert!(
            rt.pipeline(name).is_ok(),
            "kernel {name} missing from the metallib; rebuild it rather than \
             letting the test that covers it report `ok` without running"
        );
    }

    fn max_abs_err(got: &[f32], exp: &[f32]) -> f32 {
        assert_eq!(got.len(), exp.len(), "parity length mismatch");
        assert!(
            got.iter().chain(exp).all(|x| x.is_finite()),
            "nonfinite parity input"
        );
        got.iter()
            .zip(exp.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn parity_metric_rejects_nonfinite_and_length_mismatch() {
        for (got, expected) in [
            (vec![f32::NAN], vec![0.0]),
            (vec![f32::INFINITY], vec![f32::INFINITY]),
            (vec![0.0], vec![0.0, 1.0]),
        ] {
            assert!(std::panic::catch_unwind(|| max_abs_err(&got, &expected)).is_err());
        }
    }

    fn run_case(m: usize, n: usize, k: usize, backend: GemmBackend) {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        eprintln!(
            "device={} encode=Metal4 tensorops={} backend={:?}",
            rt.device_name(),
            rt.has_tensorops(),
            backend
        );

        let mut a_host = vec![0.0f32; m * k];
        let mut b_host = vec![0.0f32; k * n];
        for (i, slot) in a_host.iter_mut().enumerate() {
            *slot = ((i % 17) as f32) * 0.1 - 0.8;
        }
        for (i, slot) in b_host.iter_mut().enumerate() {
            *slot = ((i % 13) as f32) * 0.07 - 0.4;
        }
        let expected = gemm_f32_cpu(&a_host, &b_host, m, n, k);

        let a = rt.alloc_tensor_f32(&[m, k]).unwrap();
        let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer.write_f32(&a_host);
        b.buffer.write_f32(&b_host);

        gemm_f32(&a, &b, &c, backend).unwrap();
        rt.synchronize().unwrap();
        let got = c.buffer.read_f32();
        let err = max_abs_err(&got, &expected);
        assert!(
            err < 1e-4,
            "GEMM {m}x{k}@{k}x{n} backend={backend:?} max_abs_err={err}"
        );
    }

    #[test]
    fn gemm_simdgroup_16() {
        run_case(16, 16, 16, GemmBackend::Simdgroup);
    }

    #[test]
    fn gemm_simdgroup_32() {
        run_case(32, 32, 32, GemmBackend::Simdgroup);
    }

    #[test]
    fn gemm_auto_small() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        let backend = select_backend(&rt);
        let dim = if backend == GemmBackend::TensorOps {
            32
        } else {
            16
        };
        run_case(dim, dim, dim, backend);
    }

    #[test]
    fn gemm_tensorops_32() {
        tensorops_runtime();
        run_case(32, 32, 64, GemmBackend::TensorOps);
        run_case(64, 32, 32, GemmBackend::TensorOps);
    }

    #[test]
    fn gemm_bf16_tensorops() {
        let rt = tensorops_runtime();
        rt.set_precision(crate::runtime::PrecisionMode::Bf16);
        let m = 32usize;
        let n = 32usize;
        let k = 64usize;
        let mut a_f = vec![0.0f32; m * k];
        let mut b_f = vec![0.0f32; k * n];
        for (i, slot) in a_f.iter_mut().enumerate() {
            *slot = ((i % 17) as f32) * 0.1 - 0.8;
        }
        for (i, slot) in b_f.iter_mut().enumerate() {
            *slot = ((i % 13) as f32) * 0.07 - 0.4;
        }
        let expected = gemm_f32_cpu(&a_f, &b_f, m, n, k);
        let a = rt.alloc_tensor_bf16(&[m, k]).unwrap();
        let b = rt.alloc_tensor_bf16(&[k, n]).unwrap();
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer
            .write_bf16_bits(&crate::tensor::f32_slice_to_bf16(&a_f));
        b.buffer
            .write_bf16_bits(&crate::tensor::f32_slice_to_bf16(&b_f));
        gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let got = c.buffer.read_f32();
        let err = max_abs_err(&got, &expected);
        // bf16 rounding — looser than f32
        assert!(err < 2e-2, "bf16 GEMM max_abs_err={err}");
    }

    /// Phase H: `gemm_train` under Bf16 casts f32 masters → bf16 TensorOps.
    #[test]
    fn gemm_train_bf16_casts_f32_operands() {
        let rt = tensorops_runtime();
        rt.set_precision(PrecisionMode::Bf16);
        let m = 32usize;
        let n = 32usize;
        let k = 64usize;
        let mut a_f = vec![0.0f32; m * k];
        let mut b_f = vec![0.0f32; k * n];
        for (i, slot) in a_f.iter_mut().enumerate() {
            *slot = ((i % 17) as f32) * 0.1 - 0.8;
        }
        for (i, slot) in b_f.iter_mut().enumerate() {
            *slot = ((i % 13) as f32) * 0.07 - 0.4;
        }
        let expected = gemm_f32_cpu(&a_f, &b_f, m, n, k);
        let a = rt.alloc_tensor_f32(&[m, k]).unwrap();
        let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer.write_f32(&a_f);
        b.buffer.write_f32(&b_f);
        gemm_train(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let got = c.buffer.read_f32();
        let err = max_abs_err(&got, &expected);
        assert!(err < 2e-2, "gemm_train bf16 max_abs_err={err}");
    }

    /// Phase H bridge: `relaxed_precision` numerics vs exact f32 / CPU.
    /// Kept behind a flag for train; documents whether 1e-5 goldens survive.
    #[test]
    fn gemm_relaxed_precision_numerics() {
        let rt = tensorops_runtime();
        // Phase H kernel: present in every metallib build.rs currently emits.
        require_pipeline(&rt, "matmul2d_tensorops_f32_relaxed");
        let m = 64usize;
        let n = 64usize;
        let k = 128usize;
        let mut a_f = vec![0.0f32; m * k];
        let mut b_f = vec![0.0f32; k * n];
        for (i, slot) in a_f.iter_mut().enumerate() {
            *slot = ((i % 17) as f32) * 0.1 - 0.8;
        }
        for (i, slot) in b_f.iter_mut().enumerate() {
            *slot = ((i % 13) as f32) * 0.07 - 0.4;
        }
        let expected = gemm_f32_cpu(&a_f, &b_f, m, n, k);

        let a = rt.alloc_tensor_f32(&[m, k]).unwrap();
        let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c_exact = rt.alloc_tensor_f32(&[m, n]).unwrap();
        let c_relax = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer.write_f32(&a_f);
        b.buffer.write_f32(&b_f);

        rt.set_precision(PrecisionMode::F32);
        rt.set_relaxed_precision(false);
        gemm_f32(&a, &b, &c_exact, GemmBackend::TensorOps).unwrap();
        rt.set_relaxed_precision(true);
        gemm_f32(&a, &b, &c_relax, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();

        let got_exact = c_exact.buffer.read_f32();
        let got_relax = c_relax.buffer.read_f32();
        let err_exact = max_abs_err(&got_exact, &expected);
        let err_relax = max_abs_err(&got_relax, &expected);
        let err_vs_exact = max_abs_err(&got_relax, &got_exact);
        eprintln!(
            "relaxed_precision: err_vs_cpu_exact={err_exact:.3e} err_vs_cpu_relax={err_relax:.3e} \
             err_relax_vs_exact={err_vs_exact:.3e}"
        );
        assert!(err_exact < 1e-4, "exact f32 GEMM drifted: {err_exact}");
        // Smoke: relaxed must be finite and within a generous bound (tf32-class).
        assert!(
            err_relax < 5e-2,
            "relaxed GEMM too far from CPU: {err_relax}"
        );
        // Document 1e-5 golden gate: if this fails, keep --tf32 off for parity.
        if err_relax >= 1e-5 {
            eprintln!(
                "NOTE: relaxed_precision breaks 1e-5 golden atol (err={err_relax:.3e}); \
                 leave flag off for f32 parity / enable only for throughput experiments"
            );
        } else {
            eprintln!("relaxed_precision within 1e-5 of CPU on this shape");
        }
        rt.set_relaxed_precision(false);
    }

    #[test]
    fn gemm_train_bf16_awkward_k() {
        // sota shapes: bigram_dim=48, ve_dim=24 — must not NaN under bf16 TensorOps.
        let rt = tensorops_runtime();
        rt.set_precision(PrecisionMode::Bf16);
        for (m, n, k) in [(64usize, 128usize, 48usize), (64, 128, 24), (4096, 128, 48)] {
            let mut a_f = vec![0.0f32; m * k];
            let mut b_f = vec![0.0f32; k * n];
            for (i, slot) in a_f.iter_mut().enumerate() {
                *slot = ((i % 17) as f32) * 0.01 - 0.08;
            }
            for (i, slot) in b_f.iter_mut().enumerate() {
                *slot = ((i % 13) as f32) * 0.007 - 0.04;
            }
            let expected = gemm_f32_cpu(&a_f, &b_f, m, n, k);
            let a = rt.alloc_tensor_f32(&[m, k]).unwrap();
            let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            a.buffer.write_f32(&a_f);
            b.buffer.write_f32(&b_f);
            gemm_train(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let got = c.buffer.read_f32();
            let n_bad = got.iter().filter(|x| !x.is_finite()).count();
            let err = max_abs_err(&got, &expected);
            eprintln!("bf16 awkward {m}x{k}@{k}x{n}: nonfinite={n_bad} err={err:.3e}");
            assert_eq!(n_bad, 0, "NaN/Inf in bf16 GEMM {m}x{k}@{k}x{n}");
            assert!(err < 5e-2, "bf16 awkward K err={err}");
        }
    }

    #[test]
    fn gemm_tn_nt_bf16_train_smoke() {
        let rt = tensorops_runtime();
        require_pipeline(&rt, "matmul2d_tensorops_tn_bf16_f32");
        rt.set_precision(PrecisionMode::Bf16);
        let m = 32usize;
        let n = 32usize;
        let k = 64usize;
        // TN: A[K,M], B[K,N] → C[M,N]
        let mut a_km = vec![0.0f32; k * m];
        let mut b_kn = vec![0.0f32; k * n];
        for (i, slot) in a_km.iter_mut().enumerate() {
            *slot = ((i % 11) as f32) * 0.05 - 0.2;
        }
        for (i, slot) in b_kn.iter_mut().enumerate() {
            *slot = ((i % 7) as f32) * 0.04 - 0.1;
        }
        // CPU: C = A^T @ B
        let mut a_mk = vec![0.0f32; m * k];
        for i in 0..k {
            for j in 0..m {
                a_mk[j * k + i] = a_km[i * m + j];
            }
        }
        let exp_tn = gemm_f32_cpu(&a_mk, &b_kn, m, n, k);
        let a_t = rt.alloc_tensor_f32(&[k, m]).unwrap();
        let b_t = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c_tn = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a_t.buffer.write_f32(&a_km);
        b_t.buffer.write_f32(&b_kn);
        gemm_tn_train(&a_t, &b_t, &c_tn, GemmBackend::TensorOps).unwrap();

        // NT: A[M,K], B[N,K] → C[M,N]
        let mut b_nk = vec![0.0f32; n * k];
        for i in 0..n {
            for j in 0..k {
                b_nk[i * k + j] = b_kn[j * n + i];
            }
        }
        let mut b_kn_from_nk = vec![0.0f32; k * n];
        for i in 0..n {
            for j in 0..k {
                b_kn_from_nk[j * n + i] = b_nk[i * k + j];
            }
        }
        let exp_nt = gemm_f32_cpu(&a_mk, &b_kn_from_nk, m, n, k);
        let a_n = rt.alloc_tensor_f32(&[m, k]).unwrap();
        let b_n = rt.alloc_tensor_f32(&[n, k]).unwrap();
        let c_nt = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a_n.buffer.write_f32(&a_mk);
        b_n.buffer.write_f32(&b_nk);
        gemm_nt_train(&a_n, &b_n, &c_nt, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();

        let err_tn = max_abs_err(&c_tn.buffer.read_f32(), &exp_tn);
        let err_nt = max_abs_err(&c_nt.buffer.read_f32(), &exp_nt);
        assert!(err_tn < 2e-2, "tn bf16 err={err_tn}");
        assert!(err_nt < 2e-2, "nt bf16 err={err_nt}");
    }

    fn gemm_tn_cpu(a_km: &[f32], b_kn: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut a_mk = vec![0.0f32; m * k];
        for i in 0..k {
            for j in 0..m {
                a_mk[j * k + i] = a_km[i * m + j];
            }
        }
        gemm_f32_cpu(&a_mk, b_kn, m, n, k)
    }

    fn gemm_nt_cpu(a_mk: &[f32], b_nk: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut b_kn = vec![0.0f32; k * n];
        for i in 0..n {
            for j in 0..k {
                b_kn[j * n + i] = b_nk[i * k + j];
            }
        }
        gemm_f32_cpu(a_mk, &b_kn, m, n, k)
    }

    #[test]
    fn gemm_tn_nt_tensorops_descriptors() {
        let rt = tensorops_runtime();
        for (m, n, k) in [(32usize, 32, 64), (64, 128, 128), (128, 128, 256)] {
            let mut a_km = vec![0.0f32; k * m];
            let mut b_kn = vec![0.0f32; k * n];
            for (i, slot) in a_km.iter_mut().enumerate() {
                *slot = ((i % 11) as f32) * 0.05 - 0.2;
            }
            for (i, slot) in b_kn.iter_mut().enumerate() {
                *slot = ((i % 7) as f32) * 0.04 - 0.1;
            }
            let exp = gemm_tn_cpu(&a_km, &b_kn, m, n, k);
            let a = rt.alloc_tensor_f32(&[k, m]).unwrap();
            let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            a.buffer.write_f32(&a_km);
            b.buffer.write_f32(&b_kn);
            gemm_tn_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let err = max_abs_err(&c.buffer.read_f32(), &exp);
            assert!(err < 1e-4, "TN desc {m}x{k}^T@{k}x{n} err={err}");

            let mut a_mk = vec![0.0f32; m * k];
            let mut b_nk = vec![0.0f32; n * k];
            for i in 0..m {
                for j in 0..k {
                    a_mk[i * k + j] = ((i * k + j) % 13) as f32 * 0.03 - 0.15;
                }
            }
            for i in 0..n {
                for j in 0..k {
                    b_nk[i * k + j] = ((i * k + j) % 17) as f32 * 0.02 - 0.1;
                }
            }
            let exp_nt = gemm_nt_cpu(&a_mk, &b_nk, m, n, k);
            let a2 = rt.alloc_tensor_f32(&[m, k]).unwrap();
            let b2 = rt.alloc_tensor_f32(&[n, k]).unwrap();
            let c2 = rt.alloc_tensor_f32(&[m, n]).unwrap();
            a2.buffer.write_f32(&a_mk);
            b2.buffer.write_f32(&b_nk);
            gemm_nt_f32(&a2, &b2, &c2, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let err_nt = max_abs_err(&c2.buffer.read_f32(), &exp_nt);
            assert!(err_nt < 1e-4, "NT desc {m}x{k}@{n}x{k}^T err={err_nt}");
        }
    }

    #[test]
    fn gemm_tn_splitk_tall_dw_shape() {
        let rt = tensorops_runtime();
        // dW-shaped: M=N=128, K=4096 (BT).
        let m = 128usize;
        let n = 128usize;
        let k = 4096usize;
        let mut a_km = vec![0.0f32; k * m];
        let mut b_kn = vec![0.0f32; k * n];
        for (i, slot) in a_km.iter_mut().enumerate() {
            *slot = ((i % 19) as f32) * 0.01 - 0.08;
        }
        for (i, slot) in b_kn.iter_mut().enumerate() {
            *slot = ((i % 23) as f32) * 0.008 - 0.05;
        }
        let exp = gemm_tn_cpu(&a_km, &b_kn, m, n, k);
        let a = rt.alloc_tensor_f32(&[k, m]).unwrap();
        let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer.write_f32(&a_km);
        b.buffer.write_f32(&b_kn);
        assert!(prefer_tn_splitk(m, n, k));
        gemm_tn_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let err = max_abs_err(&c.buffer.read_f32(), &exp);
        assert!(err < 1e-3, "split-K TN dW shape err={err}");
    }

    #[test]
    fn gemm_tn_splitk_mlp_dw_shape() {
        let rt = tensorops_runtime();
        // MLP-up dW: M=128, N=384, K=4096
        let m = 128usize;
        let n = 384usize;
        let k = 4096usize;
        assert!(prefer_tn_splitk(m, n, k));
        let mut a_km = vec![0.0f32; k * m];
        let mut b_kn = vec![0.0f32; k * n];
        for (i, slot) in a_km.iter_mut().enumerate() {
            *slot = ((i % 19) as f32) * 0.01 - 0.08;
        }
        for (i, slot) in b_kn.iter_mut().enumerate() {
            *slot = ((i % 23) as f32) * 0.008 - 0.05;
        }
        let exp = gemm_tn_cpu(&a_km, &b_kn, m, n, k);
        let a = rt.alloc_tensor_f32(&[k, m]).unwrap();
        let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer.write_f32(&a_km);
        b.buffer.write_f32(&b_kn);
        gemm_tn_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let err = max_abs_err(&c.buffer.read_f32(), &exp);
        assert!(err < 1e-3, "MLP-up split-K TN err={err}");
    }
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    type Launch = fn(&Tensor, &Tensor, &Tensor, GemmBackend) -> Result<(), String>;
    const LAUNCHES: &[Launch] = &[
        gemm,
        gemm_train,
        gemm_tn_f32,
        gemm_nt_f32,
        gemm_tn_train,
        gemm_nt_train,
        gemm_tn_accum_train,
        gemm_nt_accum_train,
    ];

    #[test]
    fn rejects_invalid_metadata_and_mixed_runtimes() {
        let rt = GpuRuntime::new().unwrap();
        let other = GpuRuntime::new().unwrap();
        let a = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        let b = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        let c = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        let foreign = other.alloc_tensor_f32(&[16, 16]).unwrap();
        let mut cases = Vec::new();
        for (shape, offset, dtype) in [
            (vec![0, 16], 0, DType::F32),
            (vec![16, 16], 1, DType::F32),
            (vec![16, 16], 4, DType::F32),
            (vec![usize::MAX, usize::MAX], 0, DType::F32),
            (vec![16, 16], 0, DType::BF16),
            (vec![16, 15], 0, DType::F32),
        ] {
            let mut bad = c.clone();
            bad.shape = shape;
            bad.byte_offset = offset;
            bad.dtype = dtype;
            cases.push(bad);
        }
        for precision in [PrecisionMode::F32, PrecisionMode::Bf16] {
            rt.set_precision(precision);
            for launch in LAUNCHES {
                for bad in &cases {
                    assert!(launch(&a, &b, bad, GemmBackend::TensorOps).is_err());
                }
                assert!(launch(&a, &foreign, &c, GemmBackend::TensorOps).is_err());
                // A mismatched inner dimension must fail before any bf16 cast.
                let bad_b = b.view(&[16, 15], 0);
                assert!(launch(&a, &bad_b, &c, GemmBackend::TensorOps).is_err());
            }
        }
        assert_eq!(rt.take_dispatch_count(), 0);
    }

    #[test]
    fn disjoint_bank_views_work_and_overlap_is_rejected() {
        let rt = GpuRuntime::new().unwrap();
        let bank = rt.alloc_tensor_f32(&[3 * 256]).unwrap();
        bank.buffer.write_f32(&vec![1.0; 3 * 256]);
        let a = bank.view(&[16, 16], 0);
        let b = bank.view(&[16, 16], 256);
        let c = bank.view(&[16, 16], 512);
        for launch in LAUNCHES {
            let overlap = bank.view(&[16, 16], 128);
            assert!(launch(&a, &b, &overlap, GemmBackend::TensorOps).is_err());
        }
        rt.take_dispatch_count();
        gemm(&a, &b, &c, GemmBackend::Simdgroup).unwrap();
        rt.synchronize().unwrap();
        assert_eq!(rt.take_dispatch_count(), 1, "simdgroup must not pre-zero C");
        let got = bank.buffer.read_f32();
        assert!(got[..512].iter().all(|&x| x == 1.0));
        assert!(got[512..].iter().all(|&x| x == 16.0));
    }

    #[test]
    fn transpose_edges_precision_and_accumulation() {
        let rt = GpuRuntime::new().unwrap();
        assert!(
            rt.has_tensorops(),
            "TensorOps coverage requires the actual metallib"
        );
        for (m, n, k) in [(1, 3, 1), (17, 31, 9), (33, 65, 129), (17, 31, 2049)] {
            for backend in [GemmBackend::Simdgroup, GemmBackend::TensorOps] {
                for precision in [PrecisionMode::F32, PrecisionMode::Bf16] {
                    rt.set_precision(precision);
                    for (tn, accum) in [(true, false), (false, false), (true, true), (false, true)]
                    {
                        let ashape = if tn { [k, m] } else { [m, k] };
                        let bshape = if tn { [k, n] } else { [n, k] };
                        let av: Vec<f32> =
                            (0..m * k).map(|i| (i % 13) as f32 / 16.0 - 0.25).collect();
                        let bv: Vec<f32> =
                            (0..n * k).map(|i| (i % 7) as f32 / 16.0 - 0.125).collect();
                        let a = rt.alloc_tensor_f32(&ashape).unwrap();
                        let b = rt.alloc_tensor_f32(&bshape).unwrap();
                        let bank = rt.alloc_tensor_f32(&[m * n + 8]).unwrap();
                        a.buffer.write_f32(&av);
                        b.buffer.write_f32(&bv);
                        bank.buffer.write_f32(&vec![2.0; m * n + 8]);
                        let c = bank.view(&[m, n], 4);
                        let launch: Launch = match (tn, accum) {
                            (true, false) => gemm_tn_train,
                            (false, false) => gemm_nt_train,
                            (true, true) => gemm_tn_accum_train,
                            (false, true) => gemm_nt_accum_train,
                        };
                        launch(&a, &b, &c, backend).unwrap();
                        rt.synchronize().unwrap();
                        let got = bank.buffer.read_f32();
                        assert_eq!(&got[..4], &[2.0; 4]);
                        assert_eq!(&got[m * n + 4..], &[2.0; 4]);
                        for row in 0..m {
                            for col in 0..n {
                                let mut expected = if accum { 2.0 } else { 0.0 };
                                for p in 0..k {
                                    expected += av[if tn { p * m + row } else { row * k + p }]
                                        * bv[if tn { p * n + col } else { col * k + p }];
                                }
                                let x = got[4 + row * n + col];
                                assert!(x.is_finite() && (x-expected).abs()<1e-4,
                                "{m}x{n}x{k} {backend:?} {precision:?} TN={tn} accum={accum}: {x} vs {expected}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn casts_reject_dtype_and_shape_before_encoding() {
        let rt = GpuRuntime::new().unwrap();
        let a = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        let b = rt.alloc_tensor_bf16(&[256]).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cast_f32_to_bf16_into(&a, &b)
        }));
        assert!(result.is_ok(), "cast Result API panicked");
        assert!(result.unwrap().is_err());
        assert!(cast_bf16_to_f32(&a).is_err());
        assert!(cast_f32_to_bf16(&b).is_err());
        assert_eq!(rt.take_dispatch_count(), 0);
    }

    #[test]
    fn rejects_bad_rank_without_panicking_or_encoding() {
        let rt = GpuRuntime::new().unwrap();
        let a = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        let b = a.deep_copy().unwrap();
        let c = a.deep_copy().unwrap();
        rt.synchronize().unwrap();
        let bad = a.view(&[256], 0);
        for launch in LAUNCHES {
            rt.take_dispatch_count();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                launch(&bad, &b, &c, GemmBackend::TensorOps)
            }));
            assert!(result.is_ok(), "public Result API panicked");
            assert!(result.unwrap().is_err(), "invalid rank accepted");
            assert_eq!(rt.take_dispatch_count(), 0);
        }
    }

    #[test]
    fn rejects_output_alias_before_encoding() {
        let rt = GpuRuntime::new().unwrap();
        let a = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        let b = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        for launch in LAUNCHES {
            rt.take_dispatch_count();
            assert!(launch(&a, &b, &a, GemmBackend::TensorOps).is_err());
            assert_eq!(rt.take_dispatch_count(), 0);
        }
    }

    #[test]
    fn rejects_wrong_dtype_on_transpose_paths() {
        let rt = GpuRuntime::new().unwrap();
        let mut a = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        let b = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        let c = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        a.dtype = DType::BF16; // backing allocation remains large enough for old buggy path
        assert!(gemm_tn_f32(&a, &b, &c, GemmBackend::TensorOps).is_err());
        assert!(gemm_nt_f32(&a, &b, &c, GemmBackend::TensorOps).is_err());
    }

    #[test]
    fn simdgroup_edges_and_offset_guards() {
        let rt = GpuRuntime::new().unwrap();
        for (m, n, k) in [
            (1, 1, 1),
            (7, 9, 3),
            (16, 16, 16),
            (17, 31, 9),
            (33, 65, 129),
        ] {
            let av: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 / 16.0 - 0.25).collect();
            let bv: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 / 16.0 - 0.125).collect();
            let a = rt.alloc_tensor_f32(&[m, k]).unwrap();
            let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
            let bank = rt.alloc_tensor_f32(&[m * n + 8]).unwrap();
            a.buffer.write_f32(&av);
            b.buffer.write_f32(&bv);
            let mut poisoned = vec![f32::NAN; m * n + 8];
            poisoned[..4].fill(123.0);
            poisoned[m * n + 4..].fill(123.0);
            bank.buffer.write_f32(&poisoned);
            let c = bank.view(&[m, n], 4);
            gemm(&a, &b, &c, GemmBackend::Simdgroup).unwrap();
            rt.synchronize().unwrap();
            let got = bank.buffer.read_f32();
            assert_eq!(&got[..4], &[123.0; 4]);
            assert_eq!(&got[m * n + 4..], &[123.0; 4]);
            let expected = gemm_f32_cpu(&av, &bv, m, n, k);
            for (x, y) in got[4..m * n + 4].iter().zip(expected) {
                assert!(
                    x.is_finite() && (x - y).abs() < 1e-4,
                    "{m}x{n}x{k}: {x} vs {y}"
                );
            }
        }
    }
}

#[cfg(test)]
mod stress_tests {
    //! Randomized + adversarial GEMM stress: every public launch family against
    //! a CPU reference, boundary-biased shapes, poisoned guard zones around C,
    //! bitwise determinism, sampled large-shape parity, and concurrent runtimes.
    //!
    //! Fast versions run in the default suite; `cargo test -- --ignored` runs
    //! the deep fuzz. `STRESS_SEED=<u64>` reruns a failing seed.

    use super::*;
    use crate::runtime::PrecisionMode;
    use crate::tensor::{bf16_bits_to_f32, f32_to_bf16_bits};
    use crate::GpuRuntime;

    /// xorshift64* — deterministic, dependency-free.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed.max(1))
        }
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        /// Uniform in [-0.5, 0.5).
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        }
    }

    /// Tile-boundary-biased dimensions (SM/SN 32/64, simdgroup 16, ±1 edges).
    const EDGE_DIMS: &[usize] = &[
        1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 95, 96, 127, 128, 129, 191, 192, 193,
    ];
    /// K values around BK=128, the split-K k_tile=256, and the split-K gate (2048).
    const EDGE_KS: &[usize] = &[
        1, 3, 16, 31, 63, 96, 127, 128, 129, 255, 256, 257, 383, 511, 2047, 2048, 2049,
    ];

    fn sample_dim(rng: &mut Rng) -> usize {
        if rng.below(2) == 0 {
            EDGE_DIMS[rng.below(EDGE_DIMS.len())]
        } else {
            1 + rng.below(160)
        }
    }

    fn sample_k(rng: &mut Rng) -> usize {
        if rng.below(2) == 0 {
            EDGE_KS[rng.below(EDGE_KS.len())]
        } else {
            1 + rng.below(320)
        }
    }

    fn round_bf16(v: &[f32]) -> Vec<f32> {
        v.iter()
            .map(|&x| bf16_bits_to_f32(f32_to_bf16_bits(x)))
            .collect()
    }

    #[derive(Clone, Copy, Debug)]
    enum Family {
        Nn,
        NnRawBf16,
        Tn,
        Nt,
        TnAccum,
        NtAccum,
        NnSimdgroup,
    }
    const FAMILIES: &[Family] = &[
        Family::Nn,
        Family::NnRawBf16,
        Family::Tn,
        Family::Nt,
        Family::TnAccum,
        Family::NtAccum,
        Family::NnSimdgroup,
    ];

    /// Upload `data` (or its bf16 rounding) into a fresh tensor, optionally as
    /// an offset view into a larger bank (exercises byte_offset binding).
    fn upload(
        rt: &std::sync::Arc<GpuRuntime>,
        rng: &mut Rng,
        shape: &[usize],
        data: &[f32],
        bf16: bool,
    ) -> Tensor {
        let numel: usize = shape.iter().product();
        let off = if rng.below(3) == 0 { 8 } else { 0 };
        if bf16 {
            let bank = rt.alloc_tensor_bf16(&[numel + off]).unwrap();
            let mut bits = vec![0u16; numel + off];
            bits[off..].copy_from_slice(&crate::tensor::f32_slice_to_bf16(data));
            bank.buffer.write_bf16_bits(&bits);
            bank.view(shape, off)
        } else {
            let bank = rt.alloc_tensor_f32(&[numel + off]).unwrap();
            let mut host = vec![0.0f32; numel + off];
            host[off..].copy_from_slice(data);
            bank.buffer.write_f32(&host);
            bank.view(shape, off)
        }
    }

    /// One randomized case: run the family on GPU, compare against the CPU
    /// reference inside a NaN-poisoned guard bank. Returns observed max error.
    fn run_case(
        rt: &std::sync::Arc<GpuRuntime>,
        rng: &mut Rng,
        family: Family,
        seed_note: u64,
    ) -> f32 {
        let m = sample_dim(rng);
        let n = sample_dim(rng);
        let mut k = sample_k(rng);
        // Bound the CPU reference cost; split-K shapes are small-MN anyway.
        if m * n * k > 24_000_000 {
            k = (24_000_000 / (m * n)).max(1);
        }
        let bf16 = match family {
            Family::NnRawBf16 => true,
            Family::NnSimdgroup => false,
            _ => rng.below(2) == 0,
        };
        rt.set_precision(if bf16 {
            PrecisionMode::Bf16
        } else {
            PrecisionMode::F32
        });

        let a_host: Vec<f32> = (0..m * k).map(|_| rng.unit()).collect();
        let b_host: Vec<f32> = (0..n * k).map(|_| rng.unit()).collect();
        // bf16 paths: reference on the same RNE-rounded values the GPU consumes,
        // so the only remaining divergence is f32 accumulation order.
        let (a_ref, b_ref) = if bf16 {
            (round_bf16(&a_host), round_bf16(&b_host))
        } else {
            (a_host.clone(), b_host.clone())
        };
        let accum = matches!(family, Family::TnAccum | Family::NtAccum);
        let prefill = if accum { 0.25f32 } else { 0.0 };

        let (a_shape, b_shape): (Vec<usize>, Vec<usize>) = match family {
            Family::Nn | Family::NnRawBf16 | Family::NnSimdgroup => (vec![m, k], vec![k, n]),
            Family::Tn | Family::TnAccum => (vec![k, m], vec![k, n]),
            Family::Nt | Family::NtAccum => (vec![m, k], vec![n, k]),
        };
        // Host data laid out to match the tensor shape.
        let a_data: Vec<f32> = match family {
            Family::Tn | Family::TnAccum => {
                let mut t = vec![0.0f32; k * m];
                for i in 0..m {
                    for p in 0..k {
                        t[p * m + i] = a_host[i * k + p];
                    }
                }
                t
            }
            _ => a_host.clone(),
        };
        let b_data: Vec<f32> = match family {
            Family::Nt | Family::NtAccum => {
                let mut t = vec![0.0f32; n * k];
                for j in 0..n {
                    for p in 0..k {
                        t[j * k + p] = b_host[p * n + j];
                    }
                }
                t
            }
            _ => b_host.clone(),
        };

        let raw_bf16 = matches!(family, Family::NnRawBf16);
        let a = upload(rt, rng, &a_shape, &a_data, raw_bf16);
        let b = upload(rt, rng, &b_shape, &b_data, raw_bf16);

        let bank = rt.alloc_tensor_f32(&[m * n + 8]).unwrap();
        let mut poisoned = vec![f32::NAN; m * n + 8];
        poisoned[..4].fill(777.0);
        poisoned[m * n + 4..].fill(777.0);
        if accum {
            for v in poisoned[4..m * n + 4].iter_mut() {
                *v = prefill;
            }
        }
        bank.buffer.write_f32(&poisoned);
        let c = bank.view(&[m, n], 4);

        match family {
            Family::Nn => gemm_train(&a, &b, &c, GemmBackend::TensorOps).unwrap(),
            Family::NnRawBf16 => gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap(),
            Family::NnSimdgroup => gemm(&a, &b, &c, GemmBackend::Simdgroup).unwrap(),
            Family::Tn => gemm_tn_train(&a, &b, &c, GemmBackend::TensorOps).unwrap(),
            Family::Nt => gemm_nt_train(&a, &b, &c, GemmBackend::TensorOps).unwrap(),
            Family::TnAccum => gemm_tn_accum_train(&a, &b, &c, GemmBackend::TensorOps).unwrap(),
            Family::NtAccum => gemm_nt_accum_train(&a, &b, &c, GemmBackend::TensorOps).unwrap(),
        }
        rt.synchronize().unwrap();

        let got = bank.buffer.read_f32();
        assert!(
            got[..4] == [777.0; 4] && got[m * n + 4..] == [777.0; 4],
            "guard zone clobbered: {family:?} {m}x{n}x{k} seed={seed_note}"
        );
        let expected = gemm_f32_cpu(&a_ref, &b_ref, m, n, k);
        // f32 exact tracks the suite's 1e-4 gate; bf16 rounds inputs identically
        // on both sides, so only f32 reassociation remains (grows with K).
        let atol = if bf16 {
            2e-3f32
        } else {
            1e-4 + 1e-7 * k as f32
        };
        let mut max_err = 0.0f32;
        for (i, (&x, &e)) in got[4..m * n + 4].iter().zip(expected.iter()).enumerate() {
            let want = e + prefill;
            let err = (x - want).abs();
            assert!(
                x.is_finite() && err < atol,
                "{family:?} {m}x{n}x{k} bf16={bf16} seed={seed_note} idx={i}: got {x} want {want} (atol {atol})"
            );
            max_err = max_err.max(err);
        }
        max_err
    }

    fn fuzz(cases: usize) {
        let seed = std::env::var("STRESS_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0x5EED_2026_0830u64);
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        assert!(
            rt.has_tensorops(),
            "stress fuzz requires the TensorOps metallib"
        );
        let mut rng = Rng::new(seed);
        let mut worst = 0.0f32;
        for i in 0..cases {
            let family = FAMILIES[rng.below(FAMILIES.len())];
            let err = run_case(&rt, &mut rng, family, seed);
            worst = worst.max(err);
            if i % 50 == 0 {
                eprintln!("fuzz case {i}/{cases} worst_err={worst:.3e}");
            }
        }
        rt.set_precision(PrecisionMode::F32);
        eprintln!("gemm fuzz: {cases} cases seed={seed:#x} worst_err={worst:.3e}");
    }

    #[test]
    fn gemm_fuzz_quick() {
        fuzz(160);
    }

    /// Deep soak — `cargo test --release -- --ignored gemm_fuzz_deep`.
    #[test]
    #[ignore]
    fn gemm_fuzz_deep() {
        fuzz(2500);
    }

    /// The bf16 NN kernel's ragged-edge slice path (M % 64 != 0, N % 32 != 0)
    /// had no direct coverage: awkward-K tests kept M and N tile-aligned.
    #[test]
    fn gemm_bf16_nn_ragged_mn() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        assert!(rt.has_tensorops(), "requires TensorOps metallib");
        for (m, n, k) in [
            (65usize, 33usize, 128usize),
            (63, 31, 129),
            (1, 1, 130),
            (130, 70, 260),
            (127, 95, 2049),
        ] {
            let a_f: Vec<f32> = (0..m * k)
                .map(|i| ((i % 251) as f32) / 256.0 - 0.49)
                .collect();
            let b_f: Vec<f32> = (0..k * n)
                .map(|i| ((i % 241) as f32) / 256.0 - 0.47)
                .collect();
            let a_r = round_bf16(&a_f);
            let b_r = round_bf16(&b_f);
            let expected = gemm_f32_cpu(&a_r, &b_r, m, n, k);
            let a = rt.alloc_tensor_bf16(&[m, k]).unwrap();
            let b = rt.alloc_tensor_bf16(&[k, n]).unwrap();
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            a.buffer
                .write_bf16_bits(&crate::tensor::f32_slice_to_bf16(&a_f));
            b.buffer
                .write_bf16_bits(&crate::tensor::f32_slice_to_bf16(&b_f));
            gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let got = c.buffer.read_f32();
            for (i, (&x, &e)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    x.is_finite() && (x - e).abs() < 2e-3,
                    "bf16 ragged NN {m}x{n}x{k} idx={i}: {x} vs {e}"
                );
            }
        }
    }

    /// Missing-barrier bugs show up as run-to-run nondeterminism, not as a
    /// stable wrong answer. Every family must be bitwise-identical across reps.
    #[test]
    fn gemm_determinism_soak() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        assert!(rt.has_tensorops(), "requires TensorOps metallib");
        let mut rng = Rng::new(0xD57E_2026u64);
        // Ragged bf16 tile shape, split-K dW shape, plain f32.
        for (family, m, n, k) in [
            (Family::NnRawBf16, 130usize, 70usize, 260usize),
            (Family::Tn, 96, 96, 4096),
            (Family::Nn, 128, 96, 384),
            (Family::NtAccum, 64, 48, 2048),
        ] {
            let mut baseline: Option<Vec<u32>> = None;
            for rep in 0..25 {
                let mut case_rng = Rng::new(0xBA5E_11E5u64); // same data every rep
                let bits = {
                    let m_ = m;
                    let n_ = n;
                    let k_ = k;
                    let a_host: Vec<f32> = (0..m_ * k_).map(|_| case_rng.unit()).collect();
                    let b_host: Vec<f32> = (0..n_ * k_).map(|_| case_rng.unit()).collect();
                    let bf16 = matches!(family, Family::NnRawBf16);
                    rt.set_precision(if bf16 {
                        PrecisionMode::Bf16
                    } else {
                        PrecisionMode::F32
                    });
                    let (a_shape, b_shape): (Vec<usize>, Vec<usize>) = match family {
                        Family::Tn => (vec![k_, m_], vec![k_, n_]),
                        Family::NtAccum => (vec![m_, k_], vec![n_, k_]),
                        _ => (vec![m_, k_], vec![k_, n_]),
                    };
                    let a = upload(&rt, &mut rng, &a_shape, &a_host, bf16);
                    let b = upload(&rt, &mut rng, &b_shape, &b_host, bf16);
                    let c = rt.alloc_tensor_f32(&[m_, n_]).unwrap();
                    c.buffer.write_f32(&vec![0.5f32; m_ * n_]);
                    match family {
                        Family::NnRawBf16 => gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap(),
                        Family::Tn => gemm_tn_train(&a, &b, &c, GemmBackend::TensorOps).unwrap(),
                        Family::Nn => gemm_train(&a, &b, &c, GemmBackend::TensorOps).unwrap(),
                        Family::NtAccum => {
                            gemm_nt_accum_train(&a, &b, &c, GemmBackend::TensorOps).unwrap()
                        }
                        _ => unreachable!(),
                    }
                    rt.synchronize().unwrap();
                    c.buffer
                        .read_f32()
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<u32>>()
                };
                match &baseline {
                    None => baseline = Some(bits),
                    Some(base) => assert_eq!(
                        base, &bits,
                        "{family:?} {m}x{n}x{k} diverged bitwise at rep {rep} — missing barrier?"
                    ),
                }
            }
        }
        rt.set_precision(PrecisionMode::F32);
    }

    /// Large-shape parity by sampling: full CPU reference is too slow at this
    /// size, so verify random output entries with f64 dot products and label
    /// coverage honestly (sampled, not exhaustive).
    #[test]
    fn gemm_large_sampled_parity() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        assert!(rt.has_tensorops(), "requires TensorOps metallib");
        let mut rng = Rng::new(0x1A26_E5A3u64);
        for (m, n, k, bf16) in [
            (1024usize, 1024usize, 1024usize, false),
            (1024, 1024, 1024, true),
            (1000, 520, 1030, true),
        ] {
            rt.set_precision(if bf16 {
                PrecisionMode::Bf16
            } else {
                PrecisionMode::F32
            });
            let a_host: Vec<f32> = (0..m * k).map(|_| rng.unit()).collect();
            let b_host: Vec<f32> = (0..k * n).map(|_| rng.unit()).collect();
            let (a_ref, b_ref) = if bf16 {
                (round_bf16(&a_host), round_bf16(&b_host))
            } else {
                (a_host.clone(), b_host.clone())
            };
            let a = upload(&rt, &mut rng, &[m, k], &a_host, bf16);
            let b = upload(&rt, &mut rng, &[k, n], &b_host, bf16);
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            if bf16 {
                gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            } else {
                gemm_train(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            }
            rt.synchronize().unwrap();
            let got = c.buffer.read_f32();
            let n_bad = got.iter().filter(|x| !x.is_finite()).count();
            assert_eq!(n_bad, 0, "nonfinite outputs at {m}x{n}x{k} bf16={bf16}");
            let samples = 1500usize;
            let mut max_err = 0.0f64;
            for _ in 0..samples {
                let i = rng.below(m);
                let j = rng.below(n);
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += a_ref[i * k + p] as f64 * b_ref[p * n + j] as f64;
                }
                let err = (got[i * n + j] as f64 - acc).abs();
                assert!(
                    err < 1e-2,
                    "{m}x{n}x{k} bf16={bf16} C[{i},{j}] = {} vs f64 {acc}",
                    got[i * n + j]
                );
                max_err = max_err.max(err);
            }
            eprintln!(
                "large sampled parity {m}x{n}x{k} bf16={bf16}: {samples} samples max_err={max_err:.3e} (sampled coverage, not exhaustive)"
            );
        }
        rt.set_precision(PrecisionMode::F32);
    }

    /// The NN kernels switch to the column-panel swizzle only when
    /// tiles_n*tiles_m >= 2048, a scale no other test reaches — cover the
    /// swizzle mapping (bijective tile coverage) and its edge interaction
    /// with sampled f64 parity at small K so the shapes stay cheap.
    #[test]
    fn gemm_swizzle_grid_sampled_parity() {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        assert!(rt.has_tensorops(), "requires TensorOps metallib");
        rt.set_precision(PrecisionMode::Bf16);
        let mut rng = Rng::new(0x5117_2026u64);
        // 4096x4096 with 128x64 tiles = 64*32 = 2048 tiles: swizzle ON.
        // The ragged twin keeps the same grid with edge tiles in play.
        for (m, n, k) in [(4096usize, 4096usize, 64usize), (4095, 4033, 65)] {
            let a_host: Vec<f32> = (0..m * k).map(|_| rng.unit()).collect();
            let b_host: Vec<f32> = (0..k * n).map(|_| rng.unit()).collect();
            let a_ref = round_bf16(&a_host);
            let b_ref = round_bf16(&b_host);
            let a = upload(&rt, &mut rng, &[m, k], &a_host, true);
            let b = upload(&rt, &mut rng, &[k, n], &b_host, true);
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            c.buffer.write_f32(&vec![f32::NAN; m * n]);
            gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            let got = c.buffer.read_f32();
            // Every element overwritten: a tile skipped by a broken swizzle
            // mapping would leave NaN poison behind.
            let n_bad = got.iter().filter(|x| !x.is_finite()).count();
            assert_eq!(n_bad, 0, "unwritten/nonfinite C at {m}x{n}x{k}");
            for _ in 0..1500 {
                let i = rng.below(m);
                let j = rng.below(n);
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += a_ref[i * k + p] as f64 * b_ref[p * n + j] as f64;
                }
                let err = (got[i * n + j] as f64 - acc).abs();
                assert!(
                    err < 1e-2,
                    "{m}x{n}x{k} C[{i},{j}] = {} vs f64 {acc}",
                    got[i * n + j]
                );
            }
        }
        rt.set_precision(PrecisionMode::F32);
    }

    /// Separate runtimes on separate threads must not corrupt each other
    /// (buffer pools, dispatch counters, pipeline caches are per-runtime).
    #[test]
    fn gemm_concurrent_runtimes() {
        let handles: Vec<_> = (0..3u64)
            .map(|t| {
                std::thread::spawn(move || {
                    let rt = GpuRuntime::new().expect("GpuRuntime::new");
                    let mut rng = Rng::new(0xC0C0_0000u64 + t);
                    for _ in 0..12 {
                        let family = FAMILIES[rng.below(FAMILIES.len())];
                        run_case(&rt, &mut rng, family, t);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("stress thread panicked");
        }
    }
}
