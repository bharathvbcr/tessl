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

use crate::runtime::{mtl_size, BufferKind, GpuRuntime, PrecisionMode};
use crate::tensor::{DType, Tensor};

#[derive(Clone, Copy)]
enum Layout {
    NN,
    TN,
    NT,
}

/// MPP uses signed 32-bit extents/offset arithmetic, and the TensorOps kernels
/// cast their extents to `int`. Every public GEMM operand — tensor or raw
/// buffer — passes this before encoding, which also keeps every logical element
/// address below `u32::MAX` for the kernels' `uint` tile arithmetic.
pub(crate) fn require_i32_extent(numel: usize, what: &str) -> Result<(), String> {
    if numel > i32::MAX as usize {
        return Err(format!("{what} exceeds signed 32-bit kernel indexing"));
    }
    Ok(())
}

/// All public GEMM paths validate before casting, allocating scratch, or encoding.
/// MPP uses signed 32-bit extents/offset arithmetic; reject larger matrices.
/// `entry` is the public function, named in the alignment error.
fn validate_gemm(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    layout: Layout,
    allow_bf16: bool,
    entry: &str,
) -> Result<(usize, usize, usize), String> {
    for (t, operand) in [(a, "A"), (b, "B"), (c, "C")] {
        t.validate()?;
        if t.shape.len() != 2 || t.shape.contains(&0) {
            return Err("GEMM requires nonempty rank-2 tensors".into());
        }
        require_i32_extent(t.numel(), "GEMM")?;
        require_view_alignment(t, entry, operand)?;
    }
    if !std::sync::Arc::ptr_eq(a.runtime(), b.runtime()) || !std::sync::Arc::ptr_eq(a.runtime(), c.runtime()) {
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

/// Byte alignment every GEMM operand view must start on — one rule for every
/// kernel family:
///
/// | family | kernels | rule |
/// |---|---|---|
/// | plain | exact f32 NN/TN/NT, split-K, simdgroup | 16 B |
/// | exact accumulate | `tn_accum_f32`, `nt_accum_f32` | 16 B |
/// | cooperative | bf16/f16/relaxed-f32 NN, bf16 TN/NT | 16 B |
/// | cooperative accumulate | `tn_accum_bf16_f32`, `nt_accum_bf16_f32` | 16 B |
/// | epilogue | `*_epi*` | 16 B |
/// | batched | `*_batched`, at every batch's start | 16 B |
///
/// Every kernel builds inline `tensor(ptr, extents, strides)` views over a
/// raw device pointer — not `MTLTensor` objects, whose alignment
/// `crate::mtl_tensor` checks separately — and each interior tile already
/// rebases that pointer to an arbitrary element (`A + ty * K`). The 64-byte
/// rule the cooperative, accumulate, epilogue and batched entries used to
/// enforce was not a hardware requirement. On the M5 Pro every family is
/// bit-identical to its 0-offset run at 4, 8, 16, 32 and 48 bytes, under Metal
/// API and shader validation (`bench/results/gemm_align_probe_m5pro.txt`, from
/// `alignment_probe::alignment_probe_report`). 16 bytes is the documented
/// contract, not a measured floor; nothing below it is relied on.
pub(crate) const GEMM_VIEW_ALIGN: usize = 16;

/// Refuse an operand view that does not start on [`GEMM_VIEW_ALIGN`]. The
/// error names the public entry point and the operand.
fn require_view_alignment(t: &Tensor, entry: &str, operand: &str) -> Result<(), String> {
    if t.byte_offset % GEMM_VIEW_ALIGN != 0 {
        return Err(format!(
            "{entry}: operand {operand} byte_offset {} is not {GEMM_VIEW_ALIGN}-byte aligned \
             (every GEMM operand view must start on a {GEMM_VIEW_ALIGN}-byte boundary)",
            t.byte_offset
        ));
    }
    Ok(())
}

/// Tall-K / small-MN → split-K accumulate.
/// Attn dW: M=N=128, K=BT=4096. MLP dW: one side = mlp_dim=384.
fn prefer_tn_splitk(m: usize, n: usize, k: usize) -> bool {
    k >= 2048 && m <= 384 && n <= 384 && m.min(n) <= 128
}

/// Tile sizes for TensorOps kernels (must match matmul_tensorops.metal).
///
/// `scripts/audit_gemm_tiles.py` checks each `TILE_*` against the kernel whose
/// `pipeline("...")` call it follows, in any file under `src/`, so a dispatch
/// outside this module takes its geometry from here, not from local constants.
#[derive(Clone, Copy)]
pub(crate) struct TileGeom {
    pub(crate) sm: usize,
    pub(crate) sn: usize,
    /// Simdgroups per TG (`execution_simdgroups<N>`). Exact f32 uses 1.
    pub(crate) simdgroups: usize,
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
        return Err("cast destination must match shape/runtime, be bf16, and not overlap source".into());
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

/// `dst = src^T` for f32 matrices `[rows, cols]` and `[cols, rows]` (GPU).
pub fn transpose_f32_into(src: &Tensor, dst: &Tensor) -> Result<(), String> {
    validate_cast_input(src, DType::F32)?;
    dst.validate()?;
    let (s, d) = (src.shape(), dst.shape());
    if s.len() != 2
        || d != [s[1], s[0]]
        || dst.dtype != DType::F32
        || !std::sync::Arc::ptr_eq(src.runtime(), dst.runtime())
        || src.overlaps(dst)
    {
        return Err(format!(
            "transpose: destination {d:?} must be f32 [cols, rows] of the 2-D source {s:?}, on its runtime, not overlapping it"
        ));
    }
    let rt = src.runtime();
    let p = rt.pipeline("transpose2d_f32")?;
    crate::dispatch::dispatch_1d(rt, &p, src.numel(), |bnd| {
        crate::dispatch::set_tensor(bnd, src, 0);
        crate::dispatch::set_tensor(bnd, dst, 1);
        crate::dispatch::set_u32(bnd, s[0] as u32, 2);
        crate::dispatch::set_u32(bnd, s[1] as u32, 3);
    })?;
    Ok(())
}

/// Cast bf16 tensor → f32 (GPU).
pub fn cast_bf16_to_f32(src: &Tensor) -> Result<Tensor, String> {
    validate_cast_input(src, DType::BF16)?;
    let rt = src.runtime();
    let dst = rt.alloc_tensor_f32(&src.shape)?;
    cast_bf16_to_f32_into(src, &dst)?;
    Ok(dst)
}

/// Widen bf16 into an existing f32 tensor (exact: every bf16 is an f32).
pub fn cast_bf16_to_f32_into(src: &Tensor, dst: &Tensor) -> Result<(), String> {
    validate_cast_input(src, DType::BF16)?;
    dst.validate()?;
    if dst.dtype != DType::F32
        || src.shape != dst.shape
        || !std::sync::Arc::ptr_eq(src.runtime(), dst.runtime())
        || src.overlaps(dst)
    {
        return Err("cast destination must match shape/runtime, be f32, and not overlap source".into());
    }
    let rt = src.runtime();
    let p = rt.pipeline("cast_bf16_to_f32")?;
    let n = src.numel();
    crate::dispatch::dispatch_1d(rt, &p, n, |bnd| {
        crate::dispatch::set_tensor(bnd, src, 0);
        crate::dispatch::set_tensor(bnd, dst, 1);
        crate::dispatch::set_u32(bnd, n as u32, 2);
    })?;
    Ok(())
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
        DType::F16 => Err("bf16 GEMM was asked for f16 operands; convert explicitly, or use \
             the f16 kernels, which accumulate in f32 just as bf16 does"
            .into()),
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
///
/// Bf16 operands whose M does not fill a 128-row tile dispatch
/// `matmul2d_tensorops_bf16_f32_64x64_sg4` even when N > 512. That
/// instantiation is the same cooperative helper as the 128×64 kernel and
/// takes N as a runtime extent. M ≥ 128 keeps the previous rule: 64×64 only
/// when N ≤ 512.
pub fn gemm(a: &Tensor, b: &Tensor, c: &Tensor, backend: GemmBackend) -> Result<(), String> {
    gemm_dispatch(a, b, c, backend, None)
}

/// [`gemm`] on an explicit cooperative tile.
///
/// [`gemm`] picks the tile. This forces one so a numeric check or a paired
/// timing can run both geometries on the same buffers. Exact f32 and the
/// simdgroup backend have no cooperative tile and are refused. The call
/// validates with the same checks as [`gemm`] before a kernel is chosen.
pub fn gemm_tiled(a: &Tensor, b: &Tensor, c: &Tensor, backend: GemmBackend, tile: EpiTile) -> Result<(), String> {
    gemm_dispatch(a, b, c, backend, Some(tile))
}

fn gemm_dispatch(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
    tile: Option<EpiTile>,
) -> Result<(), String> {
    let (m, n, k) = validate_gemm(
        a,
        b,
        c,
        Layout::NN,
        true,
        if tile.is_some() { "gemm_tiled" } else { "gemm" },
    )?;

    let use_bf16 = a.dtype == DType::BF16 && b.dtype == DType::BF16;
    let use_f16 = a.dtype == DType::F16 && b.dtype == DType::F16;
    let narrow_dtype = use_bf16 || use_f16;
    if a.dtype != b.dtype || (narrow_dtype && backend != GemmBackend::TensorOps) {
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
    let coop = narrow_dtype || use_relaxed_f32(rt, backend);
    if tile.is_some() && !(backend == GemmBackend::TensorOps && coop) {
        return Err(
            "GEMM tile override needs the cooperative-destination path: bf16 or f16 operands, \
             or f32 with relaxed precision, on the TensorOps backend"
                .into(),
        );
    }
    match backend {
        // Cooperative-destination NN kernels (bf16, f16 and relaxed f32):
        // register accumulator, C written exactly once — no zero pre-pass.
        GemmBackend::TensorOps if coop => {
            let (kernel, geom) = match tile {
                Some(forced) => nn_coop_kernel_for(elem, forced),
                None => nn_coop_kernel(m, n, k, elem),
            };
            let pipeline = rt.pipeline(kernel)?;
            dispatch_tensorops_nn_coop(rt, &pipeline, a, b, c, m, n, k, geom)?;
        }
        GemmBackend::TensorOps => match nn_splitk_k_tile(m, n, k) {
            Some(k_tile) => gemm_nn_splitk_f32(a, b, c, (m, n, k), k_tile)?,
            None => {
                let pipeline = rt.pipeline(backend.kernel_name_f32())?;
                // Zero-tax: pack C-zero + matmul into one binder (~−1 binder/GEMM).
                dispatch_tensorops_nn(rt, &pipeline, a, b, c, m, n, k, TILE_F32)?;
            }
        },
        GemmBackend::Simdgroup => dispatch_simdgroup(rt, a, b, c, m, n, k)?,
    }

    Ok(())
}

/// Encode one simdgroup-backend f32 NN GEMM. Validation, alignment included,
/// is the caller's.
fn dispatch_simdgroup(
    rt: &GpuRuntime,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), String> {
    let kernel = if m % 16 != 0 || n % 16 != 0 || k % 8 != 0 {
        "matmul_simdgroup_edges_f32"
    } else {
        GemmBackend::Simdgroup.kernel_name_f32()
    };
    let pipeline = rt.pipeline(kernel)?;
    // Both simdgroup kernels overwrite every logical output element.
    // No pre-zero dispatch or barrier is needed (including offset views).
    let (tg_w, tg_h, tpt) = threadgroup_geometry_simdgroup(&pipeline, m, n);
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&pipeline);
        bnd.bind_tensor(a, 0);
        bnd.bind_tensor(b, 1);
        bnd.bind_tensor(c, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.dispatch(mtl_size(tg_w, tg_h, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
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
            a: m.saturating_mul(k),
            b: k.saturating_mul(n),
            c: m.saturating_mul(n),
        }
    }

    /// Contiguous A and C against one shared B.
    pub fn shared_b(m: usize, n: usize, k: usize) -> Self {
        Self {
            a: m.saturating_mul(k),
            b: 0,
            c: m.saturating_mul(n),
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
///
/// Every batch's view must start on a 16-byte boundary, as every GEMM operand
/// view must: each base, and each nonzero stride times the element size.
pub fn gemm_batched(a: &Tensor, b: &Tensor, c: &Tensor, backend: GemmBackend, spec: BatchedGemm) -> Result<(), String> {
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
    for (name, value) in [("m", m), ("n", n), ("k", k), ("batch", batch)] {
        if value > i32::MAX as usize {
            return Err(format!("batched GEMM {name} exceeds signed 32-bit kernel indexing"));
        }
    }
    let matrix_a = m
        .checked_mul(k)
        .ok_or_else(|| "batched GEMM: A matrix extent overflows usize".to_string())?;
    let matrix_b = k
        .checked_mul(n)
        .ok_or_else(|| "batched GEMM: B matrix extent overflows usize".to_string())?;
    let matrix_c = m
        .checked_mul(n)
        .ok_or_else(|| "batched GEMM: C matrix extent overflows usize".to_string())?;
    for (t, operand) in [(a, "A"), (b, "B"), (c, "C")] {
        t.validate()?;
        require_i32_extent(t.numel(), "batched GEMM")?;
        require_view_alignment(t, "gemm_batched", operand)?;
    }
    if !std::sync::Arc::ptr_eq(a.runtime(), b.runtime()) || !std::sync::Arc::ptr_eq(a.runtime(), c.runtime()) {
        return Err("batched GEMM tensors must belong to the same runtime".into());
    }
    if a.overlaps(c) || b.overlaps(c) {
        return Err("batched GEMM output must not overlap either input".into());
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

    if batch > 1 && strides.c < matrix_c {
        return Err(format!(
            "batched GEMM output batches overlap: C stride {} is smaller than one matrix ({matrix_c} elements)",
            strides.c
        ));
    }

    // The last batch element must fit. Without this the kernel walks off the
    // end of whichever operand was sized for a smaller batch and reads whatever
    // happens to be resident.
    for (t, first, stride, what) in [
        (a, matrix_a, strides.a, "A"),
        (b, matrix_b, strides.b, "B"),
        (c, matrix_c, strides.c, "C"),
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
        // The kernel starts batch `i` at `base + i * stride`, so the alignment
        // rule binds every batch, not only the first. The base is aligned
        // above; an aligned stride in bytes keeps every later start aligned.
        let stride_bytes = stride * t.dtype.size_of();
        if batch > 1 && stride_bytes % GEMM_VIEW_ALIGN != 0 {
            return Err(format!(
                "gemm_batched: operand {what} stride {stride} elements ({stride_bytes} bytes) puts batch 1 at \
                 byte_offset {}, not {GEMM_VIEW_ALIGN}-byte aligned (every batch's operand view must start on a \
                 {GEMM_VIEW_ALIGN}-byte boundary)",
                t.byte_offset + stride_bytes
            ));
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
    dispatch_tensorops_batched(rt, &pipeline, a, b, c, spec)
}

/// Encode one batched cooperative GEMM. Validation, alignment included, is
/// the caller's.
fn dispatch_tensorops_batched(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    spec: BatchedGemm,
) -> Result<(), String> {
    let BatchedGemm {
        m,
        n,
        k,
        batch,
        strides,
    } = spec;
    // Only the 128x64 geometry is instantiated batched; a narrow variant is a
    // tuning question left until measured, as for the epilogue.
    let tile = TILE_COOP_DEFAULT;
    let tiles_n = n.div_ceil(tile.sn);
    let tiles_m = m.div_ceil(tile.sm);
    let tg = morton_tg_count(tiles_n, tiles_m);
    let tpt = threads_per_tg(pipeline, tile);
    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        bnd.bind_tensor(a, 0);
        bnd.bind_tensor(b, 1);
        bnd.bind_tensor(c, 2);
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
    /// `gelu_pytorch_tanh`, computed by `kernels/gelu.h` — the same function
    /// `nn::mlp_gelu_tanh` calls, not a second derivation. At `-O2` MSL lowers
    /// plain `tanh` to `air.fast_tanh`, which NaNs past roughly |10|.
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
        self.alpha == 1.0 && self.beta == 0.0 && self.bias.is_none() && self.activation == Activation::None
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
///
/// Bf16 operands whose M does not fill a 128-row tile use the 64×64 epilogue
/// instantiation. Every other shape stays on 128×64.
pub fn gemm_epilogue(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
    epi: Epilogue<'_>,
) -> Result<(), String> {
    run_gemm_epilogue(a, b, c, backend, epi, None)
}

/// Cooperative epilogue geometry.
///
/// [`gemm_epilogue`] picks [`EpiTile::Narrow`] for bf16 only when M does not
/// fill a 128-row tile. [`gemm_epilogue_tiled`] forces a geometry so a numeric
/// check or a paired timing can run both on the same shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpiTile {
    /// 128×64 simdgroup-4. Instantiated for bf16, f16, and relaxed f32.
    Wide,
    /// 64×64 simdgroup-4. Instantiated for bf16 only.
    Narrow,
}

/// [`gemm_epilogue`] on an explicit tile.
///
/// An identity epilogue is refused. That path is the plain GEMM, which selects
/// its own tile from N and would ignore `tile`.
pub fn gemm_epilogue_tiled(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
    epi: Epilogue<'_>,
    tile: EpiTile,
) -> Result<(), String> {
    run_gemm_epilogue(a, b, c, backend, epi, Some(tile))
}

fn run_gemm_epilogue(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
    epi: Epilogue<'_>,
    tile: Option<EpiTile>,
) -> Result<(), String> {
    if epi.is_identity() {
        if tile.is_some() {
            return Err("GEMM epilogue: an explicit tile needs a non-identity epilogue; \
                 the identity path is the plain GEMM, which picks its own tile"
                .into());
        }
        // Validated here too, so a refusal names this entry, not `gemm`.
        validate_gemm(a, b, c, Layout::NN, true, "gemm_epilogue")?;
        return gemm(a, b, c, backend);
    }
    let entry = if tile.is_some() {
        "gemm_epilogue_tiled"
    } else {
        "gemm_epilogue"
    };
    let (m, n, k) = validate_gemm(a, b, c, Layout::NN, true, entry)?;
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
        // Bias is loaded while C is stored. Separate tile rows are separate
        // threadgroups, so a bias that aliases C races. M = 127 on the 64-row
        // tile is two rows; the 128-row tile is one.
        if bias.overlaps(c) {
            return Err("GEMM epilogue: bias must not overlap the output".into());
        }
    }

    let elem = if use_bf16 {
        CoopElem::Bf16
    } else if use_f16 {
        CoopElem::F16
    } else {
        CoopElem::RelaxedF32
    };
    let chosen = tile.unwrap_or_else(|| epi_tile_auto(m, elem));
    let (kernel, tile) = epi_kernel(elem, chosen)?;
    let pipeline = rt.pipeline(kernel)?;
    dispatch_tensorops_epi(rt, &pipeline, a, b, c, m, n, k, tile, epi)
}

/// Encode one cooperative epilogue GEMM. Validation, alignment included, is
/// the caller's.
fn dispatch_tensorops_epi(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    n: usize,
    k: usize,
    tile: TileGeom,
    epi: Epilogue<'_>,
) -> Result<(), String> {
    let tiles_n = n.div_ceil(tile.sn);
    let tiles_m = m.div_ceil(tile.sm);
    let tg = morton_tg_count(tiles_n, tiles_m);
    let tpt = threads_per_tg(pipeline, tile);
    let (alpha, beta, act) = (epi.alpha, epi.beta, epi.activation as u32);
    let has_bias = u32::from(epi.bias.is_some());
    // Buffer 8 is read unconditionally by the kernel binding, so it must be
    // bound even when unused: Metal faults on a declared-but-unbound buffer.
    // `has_bias` is what decides whether it is dereferenced.
    let bias_buf = epi.bias.unwrap_or(c);
    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        bnd.bind_tensor(a, 0);
        bnd.bind_tensor(b, 1);
        bnd.bind_tensor(c, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.bind_u32(tiles_m as u32, 7);
        bnd.bind_tensor(bias_buf, 8);
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
pub(crate) const TILE_COOP_DEFAULT: TileGeom = TileGeom {
    sm: 128,
    sn: 64,
    simdgroups: 4,
};
const TILE_COOP_NARROW: TileGeom = TileGeom {
    sm: 64,
    sn: 64,
    simdgroups: 4,
};

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

/// 64×64 epilogue only when a 128-row tile would not be filled, and only for
/// bf16, which is the dtype that instantiation exists for.
fn epi_tile_auto(m: usize, elem: CoopElem) -> EpiTile {
    if elem == CoopElem::Bf16 && m < TILE_COOP_DEFAULT.sm {
        EpiTile::Narrow
    } else {
        EpiTile::Wide
    }
}

fn epi_kernel(elem: CoopElem, tile: EpiTile) -> Result<(&'static str, TileGeom), String> {
    match (elem, tile) {
        (CoopElem::Bf16, EpiTile::Narrow) => Ok(("matmul2d_tensorops_bf16_f32_epi_64x64_sg4", TILE_COOP_NARROW)),
        (CoopElem::Bf16, EpiTile::Wide) => Ok(("matmul2d_tensorops_bf16_f32_epi", TILE_COOP_DEFAULT)),
        (CoopElem::F16, EpiTile::Wide) => Ok(("matmul2d_tensorops_f16_f32_epi", TILE_COOP_DEFAULT)),
        (CoopElem::RelaxedF32, EpiTile::Wide) => Ok(("matmul2d_tensorops_f32_relaxed_epi", TILE_COOP_DEFAULT)),
        (_, EpiTile::Narrow) => Err("GEMM epilogue: the 64x64 tile is instantiated for bf16 operands only".into()),
    }
}

/// Shape → coop NN kernel.
///
/// N ≤ 512 uses 64×64 from the 2026-08-30 M5 Pro tile tunes. Bf16 with
/// M < 128 also uses `matmul2d_tensorops_bf16_f32_64x64_sg4`, whose N is a
/// runtime extent. M ≥ 128 with N > 512 stays on 128×64: paired timing at
/// the GDN in-projection (K=2048, N=8224) had non-overlapping ranges, with
/// 64×64 faster at M=61 and 128×64 faster at M=200.
fn nn_coop_kernel(m: usize, n: usize, _k: usize, elem: CoopElem) -> (&'static str, TileGeom) {
    // N ≤ 512 is the 2026-08-30 tune. The short-M clause is bf16 only: f16
    // and relaxed f32 keep the N rule. M ≥ 128 with N > 512 stays on 128×64.
    let short_m = elem == CoopElem::Bf16 && m < TILE_COOP_DEFAULT.sm;
    coop_kernel(elem, short_m || n <= 512)
}

fn nn_coop_kernel_for(elem: CoopElem, tile: EpiTile) -> (&'static str, TileGeom) {
    coop_kernel(elem, tile == EpiTile::Narrow)
}

fn coop_kernel(elem: CoopElem, narrow: bool) -> (&'static str, TileGeom) {
    if narrow {
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
        bnd.bind_tensor(a, 0);
        bnd.bind_tensor(b, 1);
        bnd.bind_tensor(c, 2);
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
        bnd.bind_tensor(a, 0);
        bnd.bind_tensor(b, 1);
        bnd.bind_tensor(c, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.bind_u32(tiles_m as u32, 7);
        // f32 exact NN/TN/NT read buffer(8); bf16/relaxed ignore extra bind.
        bnd.bind_u32(if crate::ab_flags::gemm_interior_offsets() { 1 } else { 0 }, 8);
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
        bnd.bind_tensor(a, 0);
        bnd.bind_tensor(b, 1);
        bnd.bind_tensor(c, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.bind_u32(tiles_m as u32, 7);
        if bind_interior {
            bnd.bind_u32(if crate::ab_flags::gemm_interior_offsets() { 1 } else { 0 }, 8);
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
    validate_gemm(a, b, c, Layout::NN, true, "gemm_train")?;
    if use_bf16_gemm(a.runtime(), backend) {
        return gemm_bf16(a, b, c);
    }
    gemm_f32(a, b, c, backend)
}

/// The operand precision of a caller that chooses per call rather than
/// through the runtime's [`PrecisionMode`]. Both accumulate in f32 into an f32
/// C on the TensorOps backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmOperands {
    /// f32 operands, exact (not `relaxed_precision`) products.
    ExactF32,
    /// Operands rounded to bf16 ([`gemm_bf16`], [`gemm_tn_bf16`], [`gemm_nt_bf16`]).
    Bf16,
}

impl GemmOperands {
    fn exact(rt: &GpuRuntime, what: &str) -> Result<(), String> {
        if rt.relaxed_precision() {
            return Err(format!(
                "{what}: exact-f32 operands asked for, but the runtime's relaxed precision is on"
            ));
        }
        Ok(())
    }

    /// `C = A @ B`.
    pub fn nn(self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<(), String> {
        match self {
            Self::ExactF32 => {
                Self::exact(a.runtime(), "GemmOperands::nn")?;
                gemm(a, b, c, GemmBackend::TensorOps)
            }
            Self::Bf16 => gemm_bf16(a, b, c),
        }
    }

    /// `C = A^T @ B`, A stored `[K, M]`.
    pub fn tn(self, a_km: &Tensor, b_kn: &Tensor, c: &Tensor) -> Result<(), String> {
        match self {
            Self::ExactF32 => {
                Self::exact(a_km.runtime(), "GemmOperands::tn")?;
                gemm_tn_f32(a_km, b_kn, c, GemmBackend::TensorOps)
            }
            Self::Bf16 => gemm_tn_bf16(a_km, b_kn, c),
        }
    }

    /// `C = A @ B^T`, B stored `[N, K]`.
    pub fn nt(self, a_mk: &Tensor, b_nk: &Tensor, c: &Tensor) -> Result<(), String> {
        match self {
            Self::ExactF32 => {
                Self::exact(a_mk.runtime(), "GemmOperands::nt")?;
                gemm_nt_f32(a_mk, b_nk, c, GemmBackend::TensorOps)
            }
            Self::Bf16 => gemm_nt_bf16(a_mk, b_nk, c),
        }
    }

    /// Device bytes [`Self::nn`] allocates for itself on an f32 `A [m, k]`
    /// and a `B [k, n]` of dtype `b`, at allocated sizes
    /// ([`GpuRuntime::allocated_bytes_for`]): a bf16 copy of each f32 operand
    /// under [`Self::Bf16`] (a bf16 `B`, such as a bf16-stored weight, is
    /// used as it is), nothing under [`Self::ExactF32`]. Like every
    /// temporary, they stay allocated until the next waited commit.
    pub(crate) fn nn_scratch_bytes(self, m: usize, n: usize, k: usize, b: DType) -> u64 {
        match self {
            Self::ExactF32 => 0,
            Self::Bf16 => bf16_copies(m.saturating_mul(k), narrowed(k.saturating_mul(n), b)),
        }
    }

    /// As [`Self::nn_scratch_bytes`], for [`Self::tn`] (`C [m, n]`, shared
    /// dimension `k`): under [`Self::ExactF32`], the parallel split-K's
    /// partition scratch ([`tn_par_k_tile`]), or the transposed `A` a device
    /// without TensorOps takes.
    pub(crate) fn tn_scratch_bytes(self, has_tensorops: bool, m: usize, n: usize, k: usize) -> u64 {
        match self {
            Self::Bf16 => bf16_copies(k.saturating_mul(m), k.saturating_mul(n)),
            Self::ExactF32 if !(USE_TN_NT_DESCRIPTORS && has_tensorops) => f32_temp(m.saturating_mul(k)),
            Self::ExactF32 => tn_par_k_tile(m, n, k).map_or(0, |k_tile| {
                // Each partition's slice is padded to 16 bytes.
                let slice = m.saturating_mul(n).div_ceil(4).saturating_mul(4);
                f32_temp(slice.saturating_mul(k.div_ceil(k_tile)))
            }),
        }
    }

    /// As [`Self::nn_scratch_bytes`], for [`Self::nt`] (`C [m, n]`, shared
    /// dimension `k`, `B` of dtype `b`): under [`Self::ExactF32`], the
    /// transposed `B` a device without TensorOps takes.
    pub(crate) fn nt_scratch_bytes(self, has_tensorops: bool, m: usize, n: usize, k: usize, b: DType) -> u64 {
        match self {
            Self::Bf16 => bf16_copies(m.saturating_mul(k), narrowed(n.saturating_mul(k), b)),
            Self::ExactF32 if !(USE_TN_NT_DESCRIPTORS && has_tensorops) => f32_temp(k.saturating_mul(n)),
            Self::ExactF32 => 0,
        }
    }
}

/// Allocated bytes of the bf16 operand copies [`ensure_bf16`] makes, of
/// `a` and `b` elements (0: the operand is already bf16, no copy).
fn bf16_copies(a: usize, b: usize) -> u64 {
    [a, b]
        .iter()
        .filter(|&&n| n > 0)
        .map(|&n| GpuRuntime::allocated_bytes_for(n.saturating_mul(2), BufferKind::Cold))
        .fold(0, u64::saturating_add)
}

/// Elements [`ensure_bf16`] copies for an operand of `n` elements of `dtype`.
fn narrowed(n: usize, dtype: DType) -> usize {
    if dtype == DType::BF16 {
        0
    } else {
        n
    }
}

/// Allocated bytes of one f32 temporary (`alloc_temp_f32`, from the pool).
fn f32_temp(n: usize) -> u64 {
    GpuRuntime::allocated_bytes_for(n.saturating_mul(4), BufferKind::Cold)
}

/// The bf16 lane's preconditions: TensorOps, and an f32 destination.
fn check_bf16_lane(rt: &GpuRuntime, c: &Tensor, what: &str) -> Result<(), String> {
    if !rt.has_tensorops() {
        return Err(format!("{what}: bf16 operands need TensorOps, which this device lacks"));
    }
    if c.dtype != DType::F32 {
        return Err(format!(
            "{what}: bf16 operands accumulate into an f32 C, got {:?}",
            c.dtype
        ));
    }
    Ok(())
}

/// `C[M,N] = A[M,K] @ B[K,N]` with bf16 operands and f32 accumulation,
/// whatever the runtime's [`PrecisionMode`]: f32 operands are rounded to bf16
/// first (into temporaries), bf16 ones are used as they are. The lane
/// [`gemm_train`] takes under `PrecisionMode::Bf16`, for callers that choose
/// it per call.
pub fn gemm_bf16(a: &Tensor, b: &Tensor, c: &Tensor) -> Result<(), String> {
    validate_gemm(a, b, c, Layout::NN, true, "gemm_bf16")?;
    check_bf16_lane(a.runtime(), c, "gemm_bf16")?;
    gemm(&ensure_bf16(a)?, &ensure_bf16(b)?, c, GemmBackend::TensorOps)
}

/// `C[M,N] = A[K,M]^T @ B[K,N]` (TN). A is stored `[K,M]`, B `[K,N]`.
pub fn gemm_tn_f32(a_km: &Tensor, b_kn: &Tensor, c: &Tensor, backend: GemmBackend) -> Result<(), String> {
    let (m, n, k) = validate_gemm(a_km, b_kn, c, Layout::TN, false, "gemm_tn_f32")?;

    if USE_TN_NT_DESCRIPTORS && backend == GemmBackend::TensorOps && a_km.runtime().has_tensorops() {
        if let Some(k_tile) = tn_par_k_tile(m, n, k) {
            return gemm_tn_splitk_par_f32(a_km, b_kn, c, k_tile);
        }
        let rt = a_km.runtime();
        let pipeline = rt.pipeline("matmul2d_tensorops_tn_f32")?;
        return dispatch_tensorops_tn_nt(rt, &pipeline, a_km, b_kn, c, m, n, k, TILE_F32);
    }

    // Default: explicit transpose + NN (golden-safe).
    let at = a_km.runtime().alloc_temp_f32(&[m, k])?;
    transpose_f32_into(a_km, &at)?;
    gemm_f32(&at, b_kn, c, backend)
}

/// Training TN GEMM — bf16 TensorOps descriptor when `PrecisionMode::Bf16`.
pub fn gemm_tn_train(a_km: &Tensor, b_kn: &Tensor, c: &Tensor, backend: GemmBackend) -> Result<(), String> {
    validate_gemm(
        a_km,
        b_kn,
        c,
        Layout::TN,
        use_bf16_gemm(a_km.runtime(), backend),
        "gemm_tn_train",
    )?;
    if use_bf16_gemm(a_km.runtime(), backend) {
        return gemm_tn_bf16(a_km, b_kn, c);
    }
    gemm_tn_f32(a_km, b_kn, c, backend)
}

/// `C[M,N] = A[K,M]^T @ B[K,N]` with bf16 operands and f32 accumulation,
/// whatever the runtime's [`PrecisionMode`] (see [`gemm_bf16`]).
pub fn gemm_tn_bf16(a_km: &Tensor, b_kn: &Tensor, c: &Tensor) -> Result<(), String> {
    let (m, n, k) = validate_gemm(a_km, b_kn, c, Layout::TN, true, "gemm_tn_bf16")?;
    let rt = a_km.runtime();
    check_bf16_lane(rt, c, "gemm_tn_bf16")?;
    let a_bf = ensure_bf16(a_km)?;
    let b_bf = ensure_bf16(b_kn)?;
    if prefer_tn_splitk(m, n, k) {
        return gemm_tn_splitk_bf16(&a_bf, &b_bf, c, k);
    }
    // Coop kernel: register accumulator, C written once, no zero pre-pass.
    let pipeline = rt.pipeline("matmul2d_tensorops_tn_bf16_f32")?;
    dispatch_tensorops_nn_coop(rt, &pipeline, &a_bf, &b_bf, c, m, n, k, TILE_COOP_TN_NT)
}

/// K-partition width of the TN split-K lanes (small M·N, tall K).
const TN_SPLITK_K_TILE: usize = 256;

/// The sequential exact-f32 TN split-K, now only for `C +=`
/// ([`gemm_tn_accum_train`]): its partitions add into C one after another.
/// An overwriting TN takes the parallel lane ([`tn_par_k_tile`]) instead.
fn gemm_tn_splitk_f32_opts(a_km: &Tensor, b_kn: &Tensor, c: &Tensor, k: usize, zero_first: bool) -> Result<(), String> {
    let (m, n) = (a_km.shape[1], b_kn.shape[1]);
    let rt = a_km.runtime();
    let pipeline = rt.pipeline("matmul2d_tensorops_tn_splitk_f32")?;
    dispatch_k_partitions(
        rt,
        &pipeline,
        a_km,
        b_kn,
        c,
        (m, n, k),
        TILE_F32,
        TN_SPLITK_K_TILE,
        zero_first,
    )
}

/// Exact-f32 NN with a long K whose B does not fit in cache: the K-partition
/// width, or `None` for the single-dispatch kernel.
///
/// One dispatch re-reads all of B (K×N) for every 32-row tile row, and since
/// each tile's A slab is K long too, a column-panel walk cannot keep either
/// operand resident. Partitions of `NN_SPLITK_B_ELEMS / N` rows of B (a
/// multiple of 256) keep each partition's slice of B in cache; C is read and
/// written once per partition, so they are only worth it when K is many
/// partitions long and there is more than one tile row to share them.
/// The LM head's input gradient (M 4096, N 768, K 50304) is the shape this
/// is for.
fn nn_splitk_k_tile(m: usize, n: usize, k: usize) -> Option<usize> {
    const NN_SPLITK_B_ELEMS: usize = 1 << 21; // 8 MiB of f32 per partition
    const MIN_PARTITIONS: usize = 4;
    if m <= TILE_F32.sm || n == 0 || k.checked_mul(n)? < (1 << 23) {
        return None;
    }
    let k_tile = (NN_SPLITK_B_ELEMS / n) / 256 * 256;
    (k_tile >= 256 && k >= MIN_PARTITIONS * k_tile).then_some(k_tile)
}

/// Largest scratch [`gemm_tn_splitk_par_f32`] allocates, in f32 elements
/// (16 MiB); a width that would need more is refused.
const TN_PAR_MAX_SCRATCH: usize = 1 << 22;

/// K-partition width for the parallel TN split-K, or `None` when the shape
/// is not one it is for: an exact-f32 TN whose C has fewer than
/// `TN_PAR_MAX_TILES` tiles over a K at least two partitions long. Such a C
/// gives one dispatch only a few threadgroups, each walking all of K: the
/// gate weight gradient `d_pre^T · x` (12 × 768 over 4096 rows) is 24 tiles
/// and ran at ~0.4 TFLOP/s. The width gives about `TN_PAR_TARGET_TGS`
/// threadgroups in all, in multiples of 256 (M5 Pro sweep, `metal_bench tn`:
/// within ~10% of the best width at every routed shape but 12 × 768 over
/// 1024, where a width of 128 was ~15% faster at the median).
/// It also takes the shapes [`prefer_tn_splitk`] picks, where the sequential
/// split-K ran 2.6–9.6× slower than this lane and slower than one dispatch
/// (attention dW, 128 × 128 over 4096: 186 µs sequential, 31 µs here).
/// The scratch stays under `TN_PAR_MAX_SCRATCH` by construction: at most
/// `TN_PAR_TARGET_TGS / tiles` partitions of `tiles · 1024` floats each.
fn tn_par_k_tile(m: usize, n: usize, k: usize) -> Option<usize> {
    const TN_PAR_MAX_TILES: usize = 128;
    const TN_PAR_TARGET_TGS: usize = 768;
    const STEP: usize = 256;
    if m == 0 || n == 0 {
        return None;
    }
    let tiles = m.div_ceil(TILE_F32.sm) * n.div_ceil(TILE_F32.sn);
    if tiles >= TN_PAR_MAX_TILES {
        return None;
    }
    let want = (TN_PAR_TARGET_TGS / tiles).max(2);
    let k_tile = k.div_ceil(want).div_ceil(STEP).max(1) * STEP;
    (k.div_ceil(k_tile) >= 2).then_some(k_tile)
}

/// `C [M, N] = A^T · B` for `A [K, M]` and `B [K, N]`, exact f32, as
/// `ceil(K / k_tile)` K-partitions computed in one dispatch into a scratch
/// and added in partition order (`matmul2d_tensorops_tn_splitk_par_f32`,
/// then `reduce_partitions_f32`). Deterministic. The rounding differs from
/// one dispatch over all of K, within the same f32 bound. [`gemm_tn_f32`]
/// routes here by `tn_par_k_tile`; this entry takes the width, so a bench
/// can sweep it.
pub fn gemm_tn_splitk_par_f32(a_km: &Tensor, b_kn: &Tensor, c: &Tensor, k_tile: usize) -> Result<(), String> {
    let (m, n, k) = validate_gemm(a_km, b_kn, c, Layout::TN, false, "gemm_tn_splitk_par_f32")?;
    let rt = a_km.runtime();
    if !rt.has_tensorops() {
        return Err("gemm_tn_splitk_par_f32: TensorOps is unavailable on this device".into());
    }
    // A multiple of 4 keeps every partition's start in A and B 16-byte
    // aligned, as the scratch slices are.
    if k_tile == 0 || k_tile % 4 != 0 {
        return Err(format!(
            "gemm_tn_splitk_par_f32: a K partition of {k_tile}, not a positive multiple of 4"
        ));
    }
    let partitions = k.div_ceil(k_tile);
    let as_u32 = |v: usize, what: &str| -> Result<u32, String> {
        u32::try_from(v).map_err(|_| format!("gemm_tn_splitk_par_f32: {what} {v} does not fit the kernel's u32"))
    };
    let tiles_n = n.div_ceil(TILE_F32.sn);
    let tiles_m = m.div_ceil(TILE_F32.sm);
    let groups = tiles_n
        .checked_mul(tiles_m)
        .and_then(|t| t.checked_mul(partitions))
        .ok_or("gemm_tn_splitk_par_f32: threadgroup count overflows")?;
    let numel = m.checked_mul(n).ok_or("gemm_tn_splitk_par_f32: M·N overflows")?;
    // Each slice starts 16-byte aligned, as tessl asks of every GEMM operand.
    // The M5 Pro also computes correctly from 4-byte-aligned slices, so no
    // test observes this padding.
    let slice = numel.div_ceil(4) * 4;
    let scratch_len = slice
        .checked_mul(partitions)
        .ok_or("gemm_tn_splitk_par_f32: scratch length overflows")?;
    if scratch_len > TN_PAR_MAX_SCRATCH {
        return Err(format!(
            "gemm_tn_splitk_par_f32: {partitions} partitions of {m}x{n} need {scratch_len} scratch floats, over the cap of {TN_PAR_MAX_SCRATCH}"
        ));
    }
    as_u32(groups, "threadgroup count")?;
    // The kernel offsets A and B by k0·M and k0·N.
    as_u32(k.saturating_mul(m.max(n)), "K·max(M, N)")?;
    let (m_u, n_u, k_u) = (as_u32(m, "M")?, as_u32(n, "N")?, as_u32(k, "K")?);
    let (k_tile_u, parts_u) = (as_u32(k_tile, "k_tile")?, as_u32(partitions, "partitions")?);
    let (numel_u, slice_u) = (as_u32(numel, "M·N")?, as_u32(slice, "slice")?);
    // Zeroed by allocation (bump and pool both), as `mode::multiply` needs.
    let scratch = rt.alloc_temp_f32(&[scratch_len])?;
    let gemm_p = rt.pipeline("matmul2d_tensorops_tn_splitk_par_f32")?;
    let reduce_p = rt.pipeline("reduce_partitions_f32")?;
    let tpt = threads_per_tg(&gemm_p, TILE_F32);
    let r_tpt = reduce_p.threadExecutionWidth().min(numel).max(1);
    let r_groups = numel.div_ceil(r_tpt);
    rt.with_binder(|bnd| {
        let need_explicit = bnd.needs_explicit_barriers();
        bnd.set_pipeline(&gemm_p);
        bnd.bind_tensor(a_km, 0);
        bnd.bind_tensor(b_kn, 1);
        bnd.bind_tensor(&scratch, 2);
        bnd.bind_u32(m_u, 3);
        bnd.bind_u32(n_u, 4);
        bnd.bind_u32(k_u, 5);
        bnd.bind_u32(k_tile_u, 6);
        bnd.bind_u32(tiles_n as u32, 7);
        bnd.bind_u32(tiles_m as u32, 8);
        bnd.bind_u32(parts_u, 9);
        bnd.bind_u32(slice_u, 10);
        bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
        if need_explicit {
            bnd.barrier();
        }
        bnd.set_pipeline(&reduce_p);
        bnd.bind_tensor(&scratch, 0);
        bnd.bind_tensor(c, 1);
        bnd.bind_u32(numel_u, 2);
        bnd.bind_u32(parts_u, 3);
        bnd.bind_u32(slice_u, 4);
        bnd.dispatch(mtl_size(r_groups, 1, 1), mtl_size(r_tpt, 1, 1));
        Ok(())
    })
}

fn gemm_nn_splitk_f32(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    (m, n, k): (usize, usize, usize),
    k_tile: usize,
) -> Result<(), String> {
    let rt = a.runtime();
    let pipeline = rt.pipeline("matmul2d_tensorops_nn_splitk_f32")?;
    dispatch_k_partitions(
        rt,
        &pipeline,
        a,
        b,
        c,
        (m, n, k),
        TILE_F32,
        k_tile,
        /*zero_first=*/ true,
    )
}

/// The split-K lanes' binder: zero C (when asked), then one dispatch of
/// `pipeline` per `k_tile`-wide partition of K, in order, each accumulating
/// into C. The partitions write the same C, so each runs after the last; the
/// order is fixed, so the result is deterministic.
#[allow(clippy::too_many_arguments)]
fn dispatch_k_partitions(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    (m, n, k): (usize, usize, usize),
    tile: TileGeom,
    k_tile: usize,
    zero_first: bool,
) -> Result<(), String> {
    let as_u32 = |v: usize, what: &str| -> Result<u32, String> {
        u32::try_from(v).map_err(|_| format!("split-K GEMM: {what} {v} does not fit the kernel's u32"))
    };
    if k_tile == 0 {
        return Err("split-K GEMM: a K partition of 0".into());
    }
    let (m_u, n_u, k_u, k_tile_u) = (
        as_u32(m, "M")?,
        as_u32(n, "N")?,
        as_u32(k, "K")?,
        as_u32(k_tile, "k_tile")?,
    );
    // The kernels offset B by k0·N (NN) or A and B by k0·M and k0·N (TN) in u32.
    as_u32(k.saturating_mul(m.max(n)), "K·max(M, N)")?;
    let zero_p = rt.pipeline("zero_f32")?;
    let tiles_n = n.div_ceil(tile.sn);
    let tiles_m = m.div_ceil(tile.sm);
    let tg = morton_tg_count(tiles_n, tiles_m);
    let tpt = threads_per_tg(pipeline, tile);
    let numel = c.numel();
    let z_width = zero_p.threadExecutionWidth();
    let z_tpt = z_width.min(numel).max(1);
    let z_groups = numel.div_ceil(z_tpt);
    let partitions: Vec<u32> = (0..k_u).step_by(k_tile).collect();
    let (m, n, k, k_tile) = (m_u, n_u, k_u, k_tile_u);

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

        bnd.set_pipeline(pipeline);
        for (pi, &k0) in partitions.iter().enumerate() {
            if pi > 0 && need_explicit {
                bnd.barrier();
            }
            bnd.bind_tensor(a, 0);
            bnd.bind_tensor(b, 1);
            bnd.bind_tensor(c, 2);
            bnd.bind_u32(m, 3);
            bnd.bind_u32(n, 4);
            bnd.bind_u32(k, 5);
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
    let (m, n) = (a_km.shape[1], b_kn.shape[1]);
    let rt = a_km.runtime();
    let pipeline = rt.pipeline("matmul2d_tensorops_tn_splitk_bf16_f32")?;
    dispatch_k_partitions(
        rt,
        &pipeline,
        a_km,
        b_kn,
        c,
        (m, n, k),
        TILE_V2,
        TN_SPLITK_K_TILE,
        zero_first,
    )
}

/// `C[M,N] = A[M,K] @ B[N,K]^T` (NT). B is stored `[N,K]` (e.g. `W[in,out]`).
pub fn gemm_nt_f32(a_mk: &Tensor, b_nk: &Tensor, c: &Tensor, backend: GemmBackend) -> Result<(), String> {
    let (m, n, k) = validate_gemm(a_mk, b_nk, c, Layout::NT, false, "gemm_nt_f32")?;

    if USE_TN_NT_DESCRIPTORS && backend == GemmBackend::TensorOps && a_mk.runtime().has_tensorops() {
        let rt = a_mk.runtime();
        let pipeline = rt.pipeline("matmul2d_tensorops_nt_f32")?;
        return dispatch_tensorops_tn_nt(rt, &pipeline, a_mk, b_nk, c, m, n, k, TILE_F32);
    }

    let bt = b_nk.runtime().alloc_temp_f32(&[k, n])?;
    transpose_f32_into(b_nk, &bt)?;
    gemm_f32(a_mk, &bt, c, backend)
}

/// Training NT GEMM — bf16 TensorOps descriptor when `PrecisionMode::Bf16`.
pub fn gemm_nt_train(a_mk: &Tensor, b_nk: &Tensor, c: &Tensor, backend: GemmBackend) -> Result<(), String> {
    validate_gemm(
        a_mk,
        b_nk,
        c,
        Layout::NT,
        use_bf16_gemm(a_mk.runtime(), backend),
        "gemm_nt_train",
    )?;
    if use_bf16_gemm(a_mk.runtime(), backend) {
        return gemm_nt_bf16(a_mk, b_nk, c);
    }
    gemm_nt_f32(a_mk, b_nk, c, backend)
}

/// `C[M,N] = A[M,K] @ B[N,K]^T` with bf16 operands and f32 accumulation,
/// whatever the runtime's [`PrecisionMode`] (see [`gemm_bf16`]).
pub fn gemm_nt_bf16(a_mk: &Tensor, b_nk: &Tensor, c: &Tensor) -> Result<(), String> {
    let (m, n, k) = validate_gemm(a_mk, b_nk, c, Layout::NT, true, "gemm_nt_bf16")?;
    let rt = a_mk.runtime();
    check_bf16_lane(rt, c, "gemm_nt_bf16")?;
    let a_bf = ensure_bf16(a_mk)?;
    let b_bf = ensure_bf16(b_nk)?;
    // Coop kernel: register accumulator, C written once, no zero pre-pass.
    let pipeline = rt.pipeline("matmul2d_tensorops_nt_bf16_f32")?;
    dispatch_tensorops_nn_coop(rt, &pipeline, &a_bf, &b_bf, c, m, n, k, TILE_COOP_TN_NT)
}

/// `C += A[K,M]^T @ B[K,N]` (TN accumulate). No C zero — for dW into grad banks
/// and dx accumulate into a pre-zeroed buffer.
pub fn gemm_tn_accum_train(a_km: &Tensor, b_kn: &Tensor, c: &Tensor, backend: GemmBackend) -> Result<(), String> {
    let (m, n, k) = validate_gemm(
        a_km,
        b_kn,
        c,
        Layout::TN,
        use_bf16_gemm(a_km.runtime(), backend),
        "gemm_tn_accum_train",
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

    if use_accum && USE_TN_NT_DESCRIPTORS && backend == GemmBackend::TensorOps && rt.has_tensorops() {
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
/// the safer temp-plus-add path.
pub fn gemm_nt_accum_train(a_mk: &Tensor, b_nk: &Tensor, c: &Tensor, backend: GemmBackend) -> Result<(), String> {
    let (m, n, k) = validate_gemm(
        a_mk,
        b_nk,
        c,
        Layout::NT,
        use_bf16_gemm(a_mk.runtime(), backend),
        "gemm_nt_accum_train",
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

    if use_accum && USE_TN_NT_DESCRIPTORS && backend == GemmBackend::TensorOps && rt.has_tensorops() {
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

pub(crate) fn threads_per_tg(pipeline: &ProtocolObject<dyn MTLComputePipelineState>, tile: TileGeom) -> usize {
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
        assert!(got.iter().chain(exp).all(|x| x.is_finite()), "nonfinite parity input");
        got.iter()
            .zip(exp.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f32, f32::max)
    }

    /// Which exact-f32 NN shapes take K partitions. The two shapes
    /// `tests/gemm_ragged_shapes.rs` checks on the GPU must route here, or that
    /// test covers the single-dispatch kernel instead.
    #[test]
    fn long_k_nn_with_a_large_b_takes_k_partitions() {
        // LM-head input gradient at nanolab and at Lappi's vocabulary.
        assert_eq!(nn_splitk_k_tile(4096, 768, 50_304), Some(2560));
        assert_eq!(nn_splitk_k_tile(4096, 768, 248_320), Some(2560));
        // The GPU test's shapes: a short last partition each.
        assert_eq!(nn_splitk_k_tile(40, 520, 16_200), Some(3840));
        assert_eq!(nn_splitk_k_tile(33, 768, 11_008), Some(2560));
        // Transformer widths, a B that fits in cache, one tile row, or K
        // shorter than four partitions keep the single dispatch.
        for (m, n, k) in [
            (4096, 768, 2048),
            (4096, 2304, 768),
            (4096, 768, 8192),
            (32, 768, 50_304),
            (4096, 768, 10_239),
            (4096, 0, 50_304),
            (4096, 1 << 22, 4),
        ] {
            assert_eq!(nn_splitk_k_tile(m, n, k), None, "{m}x{n}x{k}");
        }
    }

    /// A TN whose C has few tiles over a long K takes the parallel
    /// K-partitions; the shapes `tests/gemm_ragged_shapes.rs` checks on the
    /// GPU through the router must route here.
    #[test]
    fn few_tile_long_k_tn_takes_parallel_partitions() {
        // The per-head gate weight gradient (12 heads, d 768, 4096 rows), at
        // shorter and longer K, and the swept shapes up to 96 tiles.
        assert_eq!(tn_par_k_tile(12, 768, 4096), Some(256));
        assert_eq!(tn_par_k_tile(12, 768, 1024), Some(256));
        assert_eq!(tn_par_k_tile(12, 768, 16_384), Some(512));
        assert_eq!(tn_par_k_tile(40, 520, 9000), Some(512));
        assert_eq!(tn_par_k_tile(64, 1024, 4096), Some(512));
        assert_eq!(tn_par_k_tile(128, 768, 4096), Some(512));
        // The GPU test's ragged shape: 12 partitions, the last 184 long.
        assert_eq!(tn_par_k_tile(13, 520, 3000), Some(256));
        // Two partitions is the least that routes.
        assert_eq!(tn_par_k_tile(12, 768, 257), Some(256));
        // Shapes the sequential split-K took before (`prefer_tn_splitk`):
        // attention dW, MLP dW, and the shorter K.
        assert_eq!(tn_par_k_tile(128, 128, 4096), Some(256));
        assert_eq!(tn_par_k_tile(128, 384, 4096), Some(256));
        assert_eq!(tn_par_k_tile(64, 64, 4096), Some(256));
        assert_eq!(tn_par_k_tile(128, 128, 2048), Some(256));
        // Every routed width keeps the scratch under the cap, including
        // all of the sequential lane's old domain.
        for m in (1..=384).step_by(31) {
            for n in (1..=384).step_by(29) {
                for k in [2, 257, 2048, 4096, 50_304, 1 << 20] {
                    if let Some(k_tile) = tn_par_k_tile(m, n, k) {
                        let scratch = k.div_ceil(k_tile) * (m * n).div_ceil(4) * 4;
                        assert!(scratch <= TN_PAR_MAX_SCRATCH, "({m}, {n}, {k}) at {k_tile}: {scratch}");
                    }
                }
            }
        }
        for (m, n, k) in [
            // 128 tiles or more keep the single dispatch.
            (96, 2048, 4096),
            (128, 1024, 4096),
            // One partition.
            (12, 768, 256),
            // Empty.
            (0, 768, 4096),
            (12, 0, 4096),
        ] {
            assert_eq!(tn_par_k_tile(m, n, k), None, "{m}x{n}x{k}");
        }
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
        assert!(err < 1e-4, "GEMM {m}x{k}@{k}x{n} backend={backend:?} max_abs_err={err}");
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
        let dim = if backend == GemmBackend::TensorOps { 32 } else { 16 };
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
        a.buffer.write_bf16_bits(&crate::tensor::f32_slice_to_bf16(&a_f));
        b.buffer.write_bf16_bits(&crate::tensor::f32_slice_to_bf16(&b_f));
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
        assert!(err_relax < 5e-2, "relaxed GEMM too far from CPU: {err_relax}");
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
    fn in_bounds_four_byte_offset_is_refused_as_misaligned() {
        let rt = GpuRuntime::new().unwrap();
        let a = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        let b = rt.alloc_tensor_f32(&[16, 16]).unwrap();
        // One f32 of headroom so elem_offset=1 stays inside the bank — this is
        // an alignment probe, not the OOB case already covered above.
        let bank = rt.alloc_tensor_f32(&[16 * 16 + 1]).unwrap();
        let c = bank.view(&[16, 16], 1);
        assert_eq!(c.byte_offset, 4);
        let err = gemm(&a, &b, &c, GemmBackend::Simdgroup).expect_err("offset 4 must fail");
        assert!(
            err.contains("16-byte aligned"),
            "expected 16-byte alignment refusal, got: {err}"
        );
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
        assert!(rt.has_tensorops(), "TensorOps coverage requires the actual metallib");
        for (m, n, k) in [(1, 3, 1), (17, 31, 9), (33, 65, 129), (17, 31, 2049)] {
            for backend in [GemmBackend::Simdgroup, GemmBackend::TensorOps] {
                for precision in [PrecisionMode::F32, PrecisionMode::Bf16] {
                    rt.set_precision(precision);
                    for (tn, accum) in [(true, false), (false, false), (true, true), (false, true)] {
                        let ashape = if tn { [k, m] } else { [m, k] };
                        let bshape = if tn { [k, n] } else { [n, k] };
                        let av: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 / 16.0 - 0.25).collect();
                        let bv: Vec<f32> = (0..n * k).map(|i| (i % 7) as f32 / 16.0 - 0.125).collect();
                        let a = rt.alloc_tensor_f32(&ashape).unwrap();
                        let b = rt.alloc_tensor_f32(&bshape).unwrap();
                        let bank = rt.alloc_tensor_f32(&[m * n + 32]).unwrap();
                        a.buffer.write_f32(&av);
                        b.buffer.write_f32(&bv);
                        bank.buffer.write_f32(&vec![2.0; m * n + 32]);
                        // 16 elems = 64 bytes, a multiple of the 16-byte
                        // rule every GEMM family shares.
                        let c = bank.view(&[m, n], 16);
                        let launch: Launch = match (tn, accum) {
                            (true, false) => gemm_tn_train,
                            (false, false) => gemm_nt_train,
                            (true, true) => gemm_tn_accum_train,
                            (false, true) => gemm_nt_accum_train,
                        };
                        launch(&a, &b, &c, backend).unwrap();
                        rt.synchronize().unwrap();
                        let got = bank.buffer.read_f32();
                        assert_eq!(&got[..16], &[2.0; 16]);
                        assert_eq!(&got[m * n + 16..], &[2.0; 16]);
                        for row in 0..m {
                            for col in 0..n {
                                let mut expected = if accum { 2.0 } else { 0.0 };
                                for p in 0..k {
                                    expected += av[if tn { p * m + row } else { row * k + p }]
                                        * bv[if tn { p * n + col } else { col * k + p }];
                                }
                                let x = got[16 + row * n + col];
                                assert!(
                                    x.is_finite() && (x - expected).abs() < 1e-4,
                                    "{m}x{n}x{k} {backend:?} {precision:?} TN={tn} accum={accum}: {x} vs {expected}"
                                );
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
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cast_f32_to_bf16_into(&a, &b)));
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
        for (m, n, k) in [(1, 1, 1), (7, 9, 3), (16, 16, 16), (17, 31, 9), (33, 65, 129)] {
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
            // 4 elems = 16 bytes: the smallest offset the one GEMM
            // alignment rule accepts, for every family.
            let c = bank.view(&[m, n], 4);
            gemm(&a, &b, &c, GemmBackend::Simdgroup).unwrap();
            rt.synchronize().unwrap();
            let got = bank.buffer.read_f32();
            assert_eq!(&got[..4], &[123.0; 4]);
            assert_eq!(&got[m * n + 4..], &[123.0; 4]);
            let expected = gemm_f32_cpu(&av, &bv, m, n, k);
            for (x, y) in got[4..m * n + 4].iter().zip(expected) {
                assert!(x.is_finite() && (x - y).abs() < 1e-4, "{m}x{n}x{k}: {x} vs {y}");
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
        v.iter().map(|&x| bf16_bits_to_f32(f32_to_bf16_bits(x))).collect()
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
    fn upload(rt: &std::sync::Arc<GpuRuntime>, rng: &mut Rng, shape: &[usize], data: &[f32], bf16: bool) -> Tensor {
        let numel: usize = shape.iter().product();
        // Offset in elements so the resulting byte_offset sits on the
        // GEMM_VIEW_ALIGN boundary and no further, for f32 and bf16/f16.
        let align_elems = GEMM_VIEW_ALIGN / if bf16 { 2 } else { 4 };
        let off = if rng.below(3) == 0 { align_elems } else { 0 };
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
    fn run_case(rt: &std::sync::Arc<GpuRuntime>, rng: &mut Rng, family: Family, seed_note: u64) -> f32 {
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
        rt.set_precision(if bf16 { PrecisionMode::Bf16 } else { PrecisionMode::F32 });

        let a_host: Vec<f32> = (0..m * k).map(|_| Rng::unit(rng)).collect();
        let b_host: Vec<f32> = (0..n * k).map(|_| Rng::unit(rng)).collect();
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

        let bank = rt.alloc_tensor_f32(&[m * n + 32]).unwrap();
        let mut poisoned = vec![f32::NAN; m * n + 32];
        poisoned[..16].fill(777.0);
        poisoned[m * n + 16..].fill(777.0);
        if accum {
            for v in poisoned[16..m * n + 16].iter_mut() {
                *v = prefill;
            }
        }
        bank.buffer.write_f32(&poisoned);
        let c = bank.view(&[m, n], 16);

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
            got[..16] == [777.0; 16] && got[m * n + 16..] == [777.0; 16],
            "guard zone clobbered: {family:?} {m}x{n}x{k} seed={seed_note}"
        );
        let expected = gemm_f32_cpu(&a_ref, &b_ref, m, n, k);
        // f32 exact tracks the suite's 1e-4 gate; bf16 rounds inputs identically
        // on both sides, so only f32 reassociation remains (grows with K).
        let atol = if bf16 { 2e-3f32 } else { 1e-4 + 1e-7 * k as f32 };
        let mut max_err = 0.0f32;
        for (i, (&x, &e)) in got[16..m * n + 16].iter().zip(expected.iter()).enumerate() {
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
        assert!(rt.has_tensorops(), "stress fuzz requires the TensorOps metallib");
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
            let a_f: Vec<f32> = (0..m * k).map(|i| ((i % 251) as f32) / 256.0 - 0.49).collect();
            let b_f: Vec<f32> = (0..k * n).map(|i| ((i % 241) as f32) / 256.0 - 0.47).collect();
            let a_r = round_bf16(&a_f);
            let b_r = round_bf16(&b_f);
            let expected = gemm_f32_cpu(&a_r, &b_r, m, n, k);
            let a = rt.alloc_tensor_bf16(&[m, k]).unwrap();
            let b = rt.alloc_tensor_bf16(&[k, n]).unwrap();
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            a.buffer.write_bf16_bits(&crate::tensor::f32_slice_to_bf16(&a_f));
            b.buffer.write_bf16_bits(&crate::tensor::f32_slice_to_bf16(&b_f));
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
                    let a_host: Vec<f32> = (0..m_ * k_).map(|_| Rng::unit(&mut case_rng)).collect();
                    let b_host: Vec<f32> = (0..n_ * k_).map(|_| Rng::unit(&mut case_rng)).collect();
                    let bf16 = matches!(family, Family::NnRawBf16);
                    rt.set_precision(if bf16 { PrecisionMode::Bf16 } else { PrecisionMode::F32 });
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
                        Family::NtAccum => gemm_nt_accum_train(&a, &b, &c, GemmBackend::TensorOps).unwrap(),
                        _ => unreachable!(),
                    }
                    rt.synchronize().unwrap();
                    c.buffer.read_f32().iter().map(|x| x.to_bits()).collect::<Vec<u32>>()
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
            rt.set_precision(if bf16 { PrecisionMode::Bf16 } else { PrecisionMode::F32 });
            let a_host: Vec<f32> = (0..m * k).map(|_| Rng::unit(&mut rng)).collect();
            let b_host: Vec<f32> = (0..k * n).map(|_| Rng::unit(&mut rng)).collect();
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
            let a_host: Vec<f32> = (0..m * k).map(|_| Rng::unit(&mut rng)).collect();
            let b_host: Vec<f32> = (0..k * n).map(|_| Rng::unit(&mut rng)).collect();
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
                assert!(err < 1e-2, "{m}x{n}x{k} C[{i},{j}] = {} vs f64 {acc}", got[i * n + j]);
            }
        }
        rt.set_precision(PrecisionMode::F32);
    }

    /// Which production kernel a [`panel_walk_matches_row_major_chunks_bit_for_bit`]
    /// case runs, with the operand element it reads.
    #[derive(Clone, Copy, Debug)]
    enum PanelLane {
        NtBf16,
        TnBf16,
        NtAccumBf16,
        TnAccumBf16,
        NnF32,
        NtF32,
        TnF32,
        NtAccumF32,
        TnAccumF32,
    }

    impl PanelLane {
        fn bf16(self) -> bool {
            matches!(
                self,
                Self::NtBf16 | Self::TnBf16 | Self::NtAccumBf16 | Self::TnAccumBf16
            )
        }

        fn accum(self) -> bool {
            matches!(
                self,
                Self::NtAccumBf16 | Self::TnAccumBf16 | Self::NtAccumF32 | Self::TnAccumF32
            )
        }

        fn layout(self) -> Layout {
            match self {
                Self::NnF32 => Layout::NN,
                Self::TnBf16 | Self::TnAccumBf16 | Self::TnF32 | Self::TnAccumF32 => Layout::TN,
                _ => Layout::NT,
            }
        }

        /// Whether `run` at this shape is the one tiled kernel, rather than a
        /// split-K route that would sum K in a different order.
        fn single_dispatch(self, m: usize, n: usize, k: usize) -> bool {
            match self {
                Self::TnBf16 => !prefer_tn_splitk(m, n, k),
                Self::NnF32 => nn_splitk_k_tile(m, n, k).is_none(),
                Self::TnF32 => tn_par_k_tile(m, n, k).is_none(),
                _ => true,
            }
        }

        fn tile(self) -> TileGeom {
            match self {
                Self::NtBf16 | Self::TnBf16 => TILE_COOP_TN_NT,
                Self::NtAccumBf16 | Self::TnAccumBf16 => TILE_COOP_ACCUM,
                _ => TILE_F32,
            }
        }

        /// `C (+)= op(A) op(B)` through the production entry point, or the
        /// production accumulate pipeline (reachable from the public API only
        /// under an A/B flag).
        fn run(self, rt: &std::sync::Arc<GpuRuntime>, a: &Tensor, b: &Tensor, c: &Tensor) {
            let (m, n) = (c.shape[0], c.shape[1]);
            let k = match self.layout() {
                Layout::TN => a.shape[0],
                _ => a.shape[1],
            };
            let accum = |kernel: &str, interior: bool| {
                let p = rt.pipeline(kernel).unwrap();
                dispatch_tensorops_accum(rt, &p, a, b, c, m, n, k, self.tile(), interior).unwrap();
            };
            match self {
                Self::NtBf16 => gemm_nt_bf16(a, b, c).unwrap(),
                Self::TnBf16 => gemm_tn_bf16(a, b, c).unwrap(),
                Self::NtAccumBf16 => accum("matmul2d_tensorops_nt_accum_bf16_f32", false),
                Self::TnAccumBf16 => accum("matmul2d_tensorops_tn_accum_bf16_f32", false),
                Self::NnF32 => gemm_f32(a, b, c, GemmBackend::TensorOps).unwrap(),
                Self::NtF32 => gemm_nt_f32(a, b, c, GemmBackend::TensorOps).unwrap(),
                Self::TnF32 => gemm_tn_f32(a, b, c, GemmBackend::TensorOps).unwrap(),
                Self::NtAccumF32 => accum("matmul2d_tensorops_nt_accum_f32", true),
                Self::TnAccumF32 => accum("matmul2d_tensorops_tn_accum_f32", true),
            }
        }
    }

    /// The shader's gate for the column-panel walk, in elements of B; held to
    /// the source so the shapes below keep straddling it.
    const PANEL_MIN_B_ELEMS: usize = 1 << 23;

    /// The column-panel walk changes only the order threadgroups run in, so a
    /// product that takes it has the bits of the same product cut into
    /// column chunks small enough to keep the row-major walk. Every kernel
    /// that walks by B's size is run past the gate, with ragged M and N (a
    /// partial last band, tile row and tile column), and its chunks under it.
    /// Two more cases are square power-of-two grids past the gate, which keep
    /// Morton order. Plain kernels start from NaN, so a skipped tile shows;
    /// accumulate kernels start from the same finite C, so a skipped or
    /// doubled tile shows.
    #[test]
    fn panel_walk_matches_row_major_chunks_bit_for_bit() {
        assert!(
            include_str!("../kernels/matmul_tensorops.metal")
                .contains("constexpr constant ulong PANEL_MIN_B_ELEMS = 1ul << 23;"),
            "the shader's panel gate moved; update PANEL_MIN_B_ELEMS and these shapes"
        );
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        assert!(rt.has_tensorops(), "requires TensorOps metallib");
        rt.set_precision(PrecisionMode::F32);
        let mut rng = Rng::new(0x9A2E_2026u64);
        let lanes = [
            PanelLane::NtBf16,
            PanelLane::TnBf16,
            PanelLane::NtAccumBf16,
            PanelLane::TnAccumBf16,
            PanelLane::NnF32,
            PanelLane::NtF32,
            PanelLane::TnF32,
            PanelLane::NtAccumF32,
            PanelLane::TnAccumF32,
        ];
        // (lane, m, n, k, chunk, morton). M = 2348 is 19 tile rows of 128
        // (bands of 4: the last holds 3), 37 of 64 (bands of 8: 5) and 74 of
        // 32 (bands of 16: 10). N leaves a partial tile column, and a last
        // chunk wide enough (48) that f32 TN keeps its tiled kernel.
        let mut cases: Vec<(PanelLane, usize, usize, usize, usize, bool)> = lanes
            .iter()
            .map(|&lane| {
                let n = if lane.bf16() { 16432 } else { 8240 };
                (lane, 2348, n, 1056, 2048, false)
            })
            .collect();
        cases.push((PanelLane::NtBf16, 4096, 2048, 4096, 1024, true));
        cases.push((PanelLane::NtF32, 2048, 2048, 4096, 1024, true));
        for (lane, m, n, k, chunk, morton) in cases {
            let t = lane.tile();
            let (tiles_n, tiles_m) = (n.div_ceil(t.sn), m.div_ceil(t.sm));
            assert_eq!(
                tiles_n == tiles_m && tiles_n.is_power_of_two(),
                morton,
                "{lane:?} {m}x{n}"
            );
            assert!(n * k >= PANEL_MIN_B_ELEMS && chunk * k < PANEL_MIN_B_ELEMS);
            for w in [n, chunk, n % chunk].into_iter().filter(|&w| w > 0) {
                assert!(lane.single_dispatch(m, w, k), "{lane:?} {m}x{w}x{k} takes split-K");
            }
            let a_host: Vec<f32> = (0..m * k).map(|_| Rng::unit(&mut rng)).collect();
            let b_host: Vec<f32> = (0..k * n).map(|_| Rng::unit(&mut rng)).collect();
            // Stored layouts: A [M,K] or [K,M]; B [K,N] (NN, TN) or [N,K] (NT).
            let a_shape = match lane.layout() {
                Layout::TN => [k, m],
                _ => [m, k],
            };
            let b_rows_are_n = matches!(lane.layout(), Layout::NT);
            let b_cols = |j0: usize, w: usize| -> (Vec<f32>, [usize; 2]) {
                if b_rows_are_n {
                    (b_host[j0 * k..(j0 + w) * k].to_vec(), [w, k])
                } else {
                    let mut out = Vec::with_capacity(k * w);
                    for p in 0..k {
                        out.extend_from_slice(&b_host[p * n + j0..p * n + j0 + w]);
                    }
                    (out, [k, w])
                }
            };
            let start = |i: usize, j: usize| -> f32 {
                if lane.accum() {
                    0.25 + ((i * 31 + j * 7) % 13) as f32 * 0.125
                } else {
                    f32::NAN
                }
            };
            let a = upload(&rt, &mut rng, &a_shape, &a_host, lane.bf16());
            let (b_all, b_shape) = b_cols(0, n);
            let b = upload(&rt, &mut rng, &b_shape, &b_all, lane.bf16());
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            let c0: Vec<f32> = (0..m * n).map(|x| start(x / n, x % n)).collect();
            c.buffer.write_f32(&c0);
            lane.run(&rt, &a, &b, &c);
            rt.synchronize().unwrap();
            let whole = c.buffer.read_f32();
            let mut mismatches = 0usize;
            let mut first = None;
            for j0 in (0..n).step_by(chunk) {
                let w = chunk.min(n - j0);
                let (b_part, b_part_shape) = b_cols(j0, w);
                let bp = upload(&rt, &mut rng, &b_part_shape, &b_part, lane.bf16());
                let cp = rt.alloc_tensor_f32(&[m, w]).unwrap();
                let cp0: Vec<f32> = (0..m * w).map(|x| start(x / w, j0 + x % w)).collect();
                cp.buffer.write_f32(&cp0);
                lane.run(&rt, &a, &bp, &cp);
                rt.synchronize().unwrap();
                let part = cp.buffer.read_f32();
                for i in 0..m {
                    for j in 0..w {
                        let (got, want) = (whole[i * n + j0 + j], part[i * w + j]);
                        if got.to_bits() != want.to_bits() || !got.is_finite() {
                            mismatches += 1;
                            first.get_or_insert((i, j0 + j, got, want));
                        }
                    }
                }
            }
            assert_eq!(
                mismatches, 0,
                "{lane:?} {m}x{n}x{k}: {mismatches} outputs differ from the row-major chunks, first {first:?}"
            );
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

#[cfg(test)]
mod epi_tile_select {
    use super::*;

    #[test]
    fn narrow_only_when_bf16_m_does_not_fill_a_128_row_tile() {
        assert_eq!(epi_tile_auto(1, CoopElem::Bf16), EpiTile::Narrow);
        assert_eq!(epi_tile_auto(61, CoopElem::Bf16), EpiTile::Narrow);
        assert_eq!(epi_tile_auto(127, CoopElem::Bf16), EpiTile::Narrow);
        assert_eq!(epi_tile_auto(128, CoopElem::Bf16), EpiTile::Wide);
        assert_eq!(epi_tile_auto(200, CoopElem::Bf16), EpiTile::Wide);
        assert_eq!(epi_tile_auto(61, CoopElem::F16), EpiTile::Wide);
        assert_eq!(epi_tile_auto(61, CoopElem::RelaxedF32), EpiTile::Wide);
    }
}

#[cfg(test)]
mod coop_tile_select {
    use super::*;

    #[test]
    fn bf16_short_m_uses_the_existing_64x64_kernel_when_n_exceeds_512() {
        let narrow = "matmul2d_tensorops_bf16_f32_64x64_sg4";
        let wide = "matmul2d_tensorops_bf16_f32";
        for m in [1usize, 61, 127] {
            let (name, tile) = nn_coop_kernel(m, 8224, 2048, CoopElem::Bf16);
            assert_eq!(name, narrow, "M={m}");
            assert_eq!((tile.sm, tile.sn, tile.simdgroups), (64, 64, 4));
        }
        for m in [128usize, 200] {
            let (name, tile) = nn_coop_kernel(m, 8224, 2048, CoopElem::Bf16);
            assert_eq!(name, wide, "M={m}");
            assert_eq!((tile.sm, tile.sn, tile.simdgroups), (128, 64, 4));
        }
        // N ≤ 512 is unchanged, including when M fills a 128-row tile.
        let (name, _) = nn_coop_kernel(200, 512, 2048, CoopElem::Bf16);
        assert_eq!(name, narrow);
        // f16 and relaxed f32 do not take the short-M exception.
        assert_eq!(
            nn_coop_kernel(61, 8224, 2048, CoopElem::F16).0,
            "matmul2d_tensorops_f16_f32"
        );
        assert_eq!(
            nn_coop_kernel(61, 8224, 2048, CoopElem::RelaxedF32).0,
            "matmul2d_tensorops_f32_relaxed"
        );
        assert_eq!(
            nn_coop_kernel(61, 512, 64, CoopElem::F16).0,
            "matmul2d_tensorops_f16_f32_64x64_sg4"
        );
    }

    /// Ragged K and N must not move the tile. `nn_coop_kernel` ignores K, and
    /// N enters only as `n <= 512`. N = 520 is past that cut and not a multiple
    /// of 64; K = 7 is not a multiple of 8. f16 and relaxed f32 stay on their
    /// own 128×64 kernels, never the bf16 64×64 name.
    #[test]
    fn ragged_k_and_n_do_not_hand_f16_or_f32_the_bf16_short_m_kernel() {
        let n = 520usize;
        let k = 7usize;
        assert!(
            n > 512 && n % 64 != 0,
            "attack shape must keep a partial N tile past the 512 cut"
        );
        assert_ne!(k % 8, 0, "attack shape must keep a partial K");
        let bf16_narrow = "matmul2d_tensorops_bf16_f32_64x64_sg4";
        let bf16_wide = "matmul2d_tensorops_bf16_f32";
        for m in [1usize, 127] {
            let (name, tile) = nn_coop_kernel(m, n, k, CoopElem::Bf16);
            assert_eq!(name, bf16_narrow, "M={m}");
            assert_eq!((tile.sm, tile.sn, tile.simdgroups), (64, 64, 4));
            assert_eq!(epi_tile_auto(m, CoopElem::Bf16), EpiTile::Narrow, "M={m}");
            for elem in [CoopElem::F16, CoopElem::RelaxedF32] {
                let (other, wide) = nn_coop_kernel(m, n, k, elem);
                assert_ne!(other, bf16_narrow, "{elem:?} M={m} took the bf16 short-M kernel");
                assert_ne!(other, bf16_wide, "{elem:?} M={m} took the bf16 wide kernel");
                assert_eq!((wide.sm, wide.sn), (128, 64), "{elem:?} M={m}");
                assert_eq!(epi_tile_auto(m, elem), EpiTile::Wide, "{elem:?} M={m}");
                let err = match epi_kernel(elem, EpiTile::Narrow) {
                    Err(err) => err,
                    Ok((name, _)) => panic!("{elem:?}: narrow epilogue was accepted as {name}"),
                };
                assert!(err.contains("bf16"), "{elem:?}: {err}");
            }
        }
        let (name, tile) = nn_coop_kernel(128, n, k, CoopElem::Bf16);
        assert_eq!(name, bf16_wide);
        assert_eq!((tile.sm, tile.sn), (128, 64));
        assert_eq!(epi_tile_auto(128, CoopElem::Bf16), EpiTile::Wide);
        // K is not an input. Branching on K % 8 later would split these.
        assert_eq!(
            nn_coop_kernel(127, n, k, CoopElem::Bf16).0,
            nn_coop_kernel(127, n, k + 1, CoopElem::Bf16).0
        );
    }
}

/// What the hardware does with an operand view that starts off a 64-byte
/// boundary. Every kernel family is encoded through its raw dispatch helper —
/// below the host's alignment gate — on views at each byte offset, and the
/// output is compared bit for bit with the same kernel on 0-offset views. A
/// kernel that needs an alignment it is not given faults, writes nothing (C
/// starts as NaN), or computes from the wrong addresses; all three show as a
/// mismatch.
///
/// `alignment_probe_report` prints the table recorded in
/// `bench/results/gemm_align_probe_m5pro.txt`. `every_rule_offset_matches_the_aligned_run`
/// asserts the part of it that [`GEMM_VIEW_ALIGN`] relies on.
#[cfg(test)]
mod alignment_probe {
    use super::*;
    use crate::tensor::{bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits, f32_to_f16_bits};
    use std::sync::Arc;

    /// Byte offsets probed. 0 is the reference run; 4 and 8 sit below the
    /// 16-byte floor and are recorded, not relied on.
    const OFFSETS: [usize; 7] = [0, 4, 8, 16, 32, 48, 64];

    /// One shape with 64-byte row strides and one with odd strides, both
    /// ragged against every tile so edge and interior tiles both run.
    const SHAPES: [(usize, usize, usize); 2] = [(144, 96, 80), (130, 97, 70)];

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Probe {
        ExactNn,
        ExactTn,
        ExactNt,
        ExactTnAccum,
        ExactNtAccum,
        Simdgroup,
        CoopNn(CoopElem, EpiTile),
        CoopTnBf16,
        CoopNtBf16,
        CoopTnAccumBf16,
        CoopNtAccumBf16,
        Epilogue(CoopElem),
        Batched(CoopElem),
    }

    const PROBES: &[Probe] = &[
        Probe::ExactNn,
        Probe::ExactTn,
        Probe::ExactNt,
        Probe::ExactTnAccum,
        Probe::ExactNtAccum,
        Probe::Simdgroup,
        Probe::CoopNn(CoopElem::RelaxedF32, EpiTile::Wide),
        Probe::CoopNn(CoopElem::RelaxedF32, EpiTile::Narrow),
        Probe::CoopNn(CoopElem::Bf16, EpiTile::Wide),
        Probe::CoopNn(CoopElem::Bf16, EpiTile::Narrow),
        Probe::CoopNn(CoopElem::F16, EpiTile::Wide),
        Probe::CoopNn(CoopElem::F16, EpiTile::Narrow),
        Probe::CoopTnBf16,
        Probe::CoopNtBf16,
        Probe::CoopTnAccumBf16,
        Probe::CoopNtAccumBf16,
        Probe::Epilogue(CoopElem::RelaxedF32),
        Probe::Epilogue(CoopElem::Bf16),
        Probe::Epilogue(CoopElem::F16),
        Probe::Batched(CoopElem::RelaxedF32),
        Probe::Batched(CoopElem::Bf16),
        Probe::Batched(CoopElem::F16),
    ];

    impl Probe {
        fn elem(self) -> DType {
            match self {
                Self::CoopNn(e, _) | Self::Epilogue(e) | Self::Batched(e) => match e {
                    CoopElem::RelaxedF32 => DType::F32,
                    CoopElem::Bf16 => DType::BF16,
                    CoopElem::F16 => DType::F16,
                },
                Self::CoopTnBf16 | Self::CoopNtBf16 | Self::CoopTnAccumBf16 | Self::CoopNtAccumBf16 => DType::BF16,
                _ => DType::F32,
            }
        }

        fn layout(self) -> Layout {
            match self {
                Self::ExactTn | Self::ExactTnAccum | Self::CoopTnBf16 | Self::CoopTnAccumBf16 => Layout::TN,
                Self::ExactNt | Self::ExactNtAccum | Self::CoopNtBf16 | Self::CoopNtAccumBf16 => Layout::NT,
                _ => Layout::NN,
            }
        }

        fn accumulates(self) -> bool {
            matches!(
                self,
                Self::ExactTnAccum
                    | Self::ExactNtAccum
                    | Self::CoopTnAccumBf16
                    | Self::CoopNtAccumBf16
                    | Self::Epilogue(_)
            )
        }

        fn kernel(self) -> &'static str {
            match self {
                Self::ExactNn => "matmul2d_tensorops_f32",
                Self::ExactTn => "matmul2d_tensorops_tn_f32",
                Self::ExactNt => "matmul2d_tensorops_nt_f32",
                Self::ExactTnAccum => "matmul2d_tensorops_tn_accum_f32",
                Self::ExactNtAccum => "matmul2d_tensorops_nt_accum_f32",
                Self::Simdgroup => "matmul_simdgroup_edges_f32",
                Self::CoopNn(e, t) => nn_coop_kernel_for(e, t).0,
                Self::CoopTnBf16 => "matmul2d_tensorops_tn_bf16_f32",
                Self::CoopNtBf16 => "matmul2d_tensorops_nt_bf16_f32",
                Self::CoopTnAccumBf16 => "matmul2d_tensorops_tn_accum_bf16_f32",
                Self::CoopNtAccumBf16 => "matmul2d_tensorops_nt_accum_bf16_f32",
                Self::Epilogue(CoopElem::Bf16) => "matmul2d_tensorops_bf16_f32_epi",
                Self::Epilogue(CoopElem::F16) => "matmul2d_tensorops_f16_f32_epi",
                Self::Epilogue(CoopElem::RelaxedF32) => "matmul2d_tensorops_f32_relaxed_epi",
                Self::Batched(CoopElem::Bf16) => "matmul2d_tensorops_bf16_f32_batched",
                Self::Batched(CoopElem::F16) => "matmul2d_tensorops_f16_f32_batched",
                Self::Batched(CoopElem::RelaxedF32) => "matmul2d_tensorops_f32_relaxed_batched",
            }
        }
    }

    fn seq(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// A buffer holding `pad_elems` of zeros and then `host` in `dtype`,
    /// viewed at the start of `host`.
    fn upload(rt: &Arc<GpuRuntime>, dtype: DType, shape: &[usize], host: &[f32], pad_elems: usize) -> Tensor {
        let total = pad_elems + host.len();
        let bank = match dtype {
            DType::F32 => rt.alloc_tensor_f32(&[total]),
            DType::BF16 => rt.alloc_tensor_bf16(&[total]),
            DType::F16 => rt.alloc_tensor_f16(&[total]),
        }
        .expect("probe alloc");
        let mut full = vec![0.0f32; pad_elems];
        full.extend_from_slice(host);
        match dtype {
            DType::F32 => bank.buffer.write_f32(&full),
            DType::BF16 => bank
                .buffer
                .write_bf16_bits(&full.iter().map(|&x| f32_to_bf16_bits(x)).collect::<Vec<_>>()),
            DType::F16 => bank
                .buffer
                .write_f16_bits(&full.iter().map(|&x| f32_to_f16_bits(x)).collect::<Vec<_>>()),
        }
        bank.view(shape, pad_elems)
    }

    /// Elements between consecutive batch matrices: the matrix rounded up to
    /// 64 bytes, plus `off` bytes, so batch `i` starts `i * off` bytes off a
    /// 64-byte boundary while batch 0 stays aligned.
    fn batch_stride(matrix: usize, elem: usize, off: usize) -> usize {
        (matrix * elem).div_ceil(64) * 64 / elem + off / elem
    }

    /// `batch` matrices of `matrix` elements, `stride` apart, zeros between.
    /// Matrix `i` is the same whatever the stride, so runs at different
    /// strides compare bit for bit.
    fn batch_bank(matrix: usize, stride: usize, batch: usize, seed: u64) -> Vec<f32> {
        let mut bank = vec![0.0f32; stride * (batch - 1) + matrix];
        for i in 0..batch {
            bank[i * stride..i * stride + matrix].copy_from_slice(&seq(matrix, seed * 16 + i as u64));
        }
        bank
    }

    enum Outcome {
        /// Bit-identical to the 0-offset run.
        Match,
        /// `count` of `total` output elements differ from the 0-offset run.
        Mismatch { count: usize, total: usize },
        /// The dispatch or the read-back failed.
        Error(String),
    }

    /// Run `probe` on `(m, n, k)` with every operand view starting `off`
    /// bytes past a 64-byte boundary (for batched, every batch past the first).
    fn run(
        rt: &Arc<GpuRuntime>,
        probe: Probe,
        (m, n, k): (usize, usize, usize),
        off: usize,
    ) -> Result<Vec<f32>, String> {
        let dt = probe.elem();
        let es = dt.size_of();
        let (a_shape, b_shape) = match probe.layout() {
            Layout::NN => ([m, k], [k, n]),
            Layout::TN => ([k, m], [k, n]),
            Layout::NT => ([m, k], [n, k]),
        };
        let c0 = seq(m * n, 3);
        if let Probe::Batched(_) = probe {
            const BATCH: usize = 3;
            let strides = BatchStrides {
                a: batch_stride(m * k, es, off),
                b: batch_stride(k * n, es, off),
                c: batch_stride(m * n, 4, off),
            };
            let a_host = batch_bank(m * k, strides.a, BATCH, 1);
            let b_host = batch_bank(k * n, strides.b, BATCH, 2);
            let a = upload(rt, dt, &[a_host.len()], &a_host, 0);
            let b = upload(rt, dt, &[b_host.len()], &b_host, 0);
            let c_len = strides.c * (BATCH - 1) + m * n;
            let c = upload(rt, DType::F32, &[c_len], &vec![f32::NAN; c_len], 0);
            let p = rt.pipeline(probe.kernel())?;
            let spec = BatchedGemm {
                m,
                n,
                k,
                batch: BATCH,
                strides,
            };
            dispatch_tensorops_batched(rt, &p, &a, &b, &c, spec)?;
            rt.synchronize()?;
            let all = c.read_f32()?;
            return Ok((0..BATCH)
                .flat_map(|i| all[i * strides.c..i * strides.c + m * n].to_vec())
                .collect());
        }

        let pad = off / es;
        let a = upload(rt, dt, &a_shape, &seq(m * k, 1), pad);
        let b = upload(rt, dt, &b_shape, &seq(k * n, 2), pad);
        let c_init = if probe.accumulates() { c0 } else { vec![f32::NAN; m * n] };
        let c = upload(rt, DType::F32, &[m, n], &c_init, off / 4);
        let bias = upload(rt, DType::F32, &[n], &seq(n, 4), off / 4);
        let p = rt.pipeline(probe.kernel())?;
        match probe {
            Probe::ExactNn => dispatch_tensorops_nn(rt, &p, &a, &b, &c, m, n, k, TILE_F32)?,
            Probe::ExactTn | Probe::ExactNt => dispatch_tensorops_tn_nt(rt, &p, &a, &b, &c, m, n, k, TILE_F32)?,
            Probe::ExactTnAccum | Probe::ExactNtAccum => {
                dispatch_tensorops_accum(rt, &p, &a, &b, &c, m, n, k, TILE_F32, true)?
            }
            Probe::Simdgroup => dispatch_simdgroup(rt, &a, &b, &c, m, n, k)?,
            Probe::CoopNn(e, t) => dispatch_tensorops_nn_coop(rt, &p, &a, &b, &c, m, n, k, nn_coop_kernel_for(e, t).1)?,
            Probe::CoopTnBf16 | Probe::CoopNtBf16 => {
                dispatch_tensorops_nn_coop(rt, &p, &a, &b, &c, m, n, k, TILE_COOP_TN_NT)?
            }
            Probe::CoopTnAccumBf16 | Probe::CoopNtAccumBf16 => {
                dispatch_tensorops_accum(rt, &p, &a, &b, &c, m, n, k, TILE_COOP_ACCUM, false)?
            }
            Probe::Epilogue(_) => {
                let epi = Epilogue {
                    alpha: 0.5,
                    beta: 0.25,
                    bias: Some(&bias),
                    activation: Activation::None,
                };
                dispatch_tensorops_epi(rt, &p, &a, &b, &c, m, n, k, TILE_COOP_DEFAULT, epi)?
            }
            Probe::Batched(_) => unreachable!("handled above"),
        }
        rt.synchronize()?;
        c.read_f32()
    }

    /// The CPU value the 0-offset run must approximate, so a bit-identical
    /// but wrong baseline cannot pass.
    fn reference(probe: Probe, (m, n, k): (usize, usize, usize)) -> Vec<f32> {
        let round = |v: Vec<f32>| -> Vec<f32> {
            match probe.elem() {
                DType::F32 => v,
                DType::BF16 => v.into_iter().map(|x| bf16_bits_to_f32(f32_to_bf16_bits(x))).collect(),
                DType::F16 => v.into_iter().map(|x| f16_bits_to_f32(f32_to_f16_bits(x))).collect(),
            }
        };
        let product = |a: &[f32], b: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for kk in 0..k {
                        let av = match probe.layout() {
                            Layout::TN => a[kk * m + i],
                            _ => a[i * k + kk],
                        };
                        let bv = match probe.layout() {
                            Layout::NT => b[j * k + kk],
                            _ => b[kk * n + j],
                        };
                        acc += av as f64 * bv as f64;
                    }
                    out[i * n + j] = acc as f32;
                }
            }
            out
        };
        let es = probe.elem().size_of();
        if let Probe::Batched(_) = probe {
            let sa = batch_stride(m * k, es, 0);
            let sb = batch_stride(k * n, es, 0);
            let a = round(batch_bank(m * k, sa, 3, 1));
            let b = round(batch_bank(k * n, sb, 3, 2));
            return (0..3)
                .flat_map(|i| product(&a[i * sa..i * sa + m * k], &b[i * sb..i * sb + k * n]))
                .collect();
        }
        let prod = product(&round(seq(m * k, 1)), &round(seq(k * n, 2)));
        let c0 = seq(m * n, 3);
        let bias = seq(n, 4);
        match probe {
            Probe::Epilogue(_) => (0..m * n).map(|e| 0.5 * prod[e] + 0.25 * c0[e] + bias[e % n]).collect(),
            p if p.accumulates() => prod.iter().zip(&c0).map(|(x, y)| x + y).collect(),
            _ => prod,
        }
    }

    /// One probe run: a kernel family on a shape at a byte offset.
    struct Row {
        probe: Probe,
        shape: (usize, usize, usize),
        off: usize,
        outcome: Outcome,
    }

    /// Every probe at every offset on every shape, with the 0-offset run
    /// checked against the CPU first.
    fn sweep(rt: &Arc<GpuRuntime>) -> Vec<Row> {
        let mut rows = Vec::new();
        for &probe in PROBES {
            for &shape in &SHAPES {
                let base = run(rt, probe, shape, 0).expect("0-offset probe run");
                let expect = reference(probe, shape);
                let scale = expect.iter().fold(1.0f32, |m, x| m.max(x.abs()));
                let worst = base
                    .iter()
                    .zip(&expect)
                    .map(|(g, e)| if g.is_finite() { (g - e).abs() } else { f32::INFINITY })
                    .fold(0.0f32, f32::max);
                // bf16 operands are rounded identically on both sides, so the
                // remaining gap is f32-vs-f64 accumulation (and tf32-class
                // products on the relaxed kernels).
                assert!(
                    worst <= 2e-2 * scale,
                    "{probe:?} {shape:?}: the aligned run is off the CPU reference by {worst} (scale {scale})"
                );
                for &off in &OFFSETS[1..] {
                    let outcome = match run(rt, probe, shape, off) {
                        Ok(got) => {
                            let count = got
                                .iter()
                                .zip(&base)
                                .filter(|(g, b)| g.to_bits() != b.to_bits())
                                .count();
                            if count == 0 {
                                Outcome::Match
                            } else {
                                Outcome::Mismatch {
                                    count,
                                    total: got.len(),
                                }
                            }
                        }
                        Err(e) => Outcome::Error(e),
                    };
                    rows.push(Row {
                        probe,
                        shape,
                        off,
                        outcome,
                    });
                }
            }
        }
        rows
    }

    fn tensorops_runtime() -> Option<Arc<GpuRuntime>> {
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        rt.has_tensorops().then_some(rt)
    }

    /// Every offset the documented rule accepts produces the aligned bits, for
    /// every kernel family the rule covers.
    #[test]
    fn every_rule_offset_matches_the_aligned_run() {
        let Some(rt) = tensorops_runtime() else {
            panic!("requires the TensorOps metallib");
        };
        for Row {
            probe,
            shape,
            off,
            outcome,
        } in sweep(&rt)
        {
            if off % GEMM_VIEW_ALIGN != 0 {
                continue;
            }
            match outcome {
                Outcome::Match => {}
                Outcome::Mismatch { count, total } => panic!(
                    "{probe:?} {shape:?} at a {off}-byte offset, which the {GEMM_VIEW_ALIGN}-byte rule accepts: \
                     {count}/{total} elements differ from the aligned run"
                ),
                Outcome::Error(e) => panic!("{probe:?} {shape:?} at a {off}-byte offset: {e}"),
            }
        }
    }

    /// The table in `bench/results/gemm_align_probe_m5pro.txt`:
    /// `cargo test --release --lib alignment_probe_report -- --ignored --nocapture`
    #[test]
    #[ignore = "prints the probe table; run by hand to refresh the committed result"]
    fn alignment_probe_report() {
        let Some(rt) = tensorops_runtime() else {
            panic!("requires the TensorOps metallib");
        };
        let rows = sweep(&rt);
        println!("probe\tshape(m,n,k)\toffset_bytes\toutcome");
        for Row {
            probe,
            shape,
            off,
            outcome,
        } in &rows
        {
            let text = match outcome {
                Outcome::Match => "bit-identical".to_string(),
                Outcome::Mismatch { count, total } => format!("MISMATCH {count}/{total}"),
                Outcome::Error(e) => format!("ERROR {e}"),
            };
            println!("{probe:?}\t{shape:?}\t{off}\t{text}");
        }
        let bad = rows.iter().filter(|r| !matches!(r.outcome, Outcome::Match)).count();
        println!("summary: {} runs, {} not bit-identical", rows.len(), bad);
    }
}
