//! MLX affine 4-bit weights for the Qwen3.5 tower: the checkpoint layout
//! `mlx_lm.convert -q --q-bits 4 --q-group-size 64` writes, read straight into
//! the Q4 kernels' banks with no bf16 (or any other dense) intermediate.
//!
//! # The layout (checked against mlx-lm 0.32.0's source, `docs/qwen35.md`)
//!
//! Every `nn.Linear` and `nn.Embedding` whose last dimension is a multiple of
//! the group size is replaced by three tensors, named after the weight:
//!
//! | tensor | dtype | shape |
//! |---|---|---|
//! | `X.weight` | `U32`, eight 4-bit values each, element `i` at bits `4i..4i+4` | `[rows, cols / 8]` |
//! | `X.scales` | `BF16` (the model dtype) | `[rows, cols / group]` |
//! | `X.biases` | `BF16` | `[rows, cols / group]` |
//!
//! `w = scale * q + bias`, the form [`crate::nn::Q4MlxBank`] dequantizes. The
//! packed words are little-endian, so their bytes are already the bank's
//! "two nibbles per byte, low first" and are read into the device buffer
//! unchanged. The bank wants the group's scale and bias adjacent (one
//! `bfloat2`), which the checkpoint stores as two arrays; they are interleaved
//! as they are read.
//!
//! What the converter leaves dense: the norms, `conv1d.weight`, `A_log`,
//! `dt_bias` and the GDN gated norm (none is an `nn.Linear` or `nn.Embedding`).
//! What `Qwen3_5.sanitize` does to a checkpoint it converts, which this loader
//! therefore undoes or accepts ([`NormConvention`]):
//!
//! * `conv1d.weight` is moved from `[conv_dim, 1, k]` to `[conv_dim, k, 1]`;
//! * `input_layernorm`, `post_attention_layernorm`, the final `norm` and the
//!   attention `q_norm` / `k_norm` weights have 1 added (MLX's RMSNorm
//!   multiplies by `w`; transformers' Qwen3.5 by `1 + w`), and the kernels
//!   here add the 1 themselves, so the loader subtracts it again (exactly in
//!   f32: the stored value is a bf16, the difference is representable);
//! * the keys become `language_model.model.*` and the head
//!   `language_model.lm_head.*` ([`MlxNames`]); `mtp.*` and the vision tower
//!   are dropped by mlx-lm, and are the two groups of tensors this loader
//!   skips by name when a checkpoint still carries them ([`OutsideTensors`]).
//!
//! # Refusals
//!
//! Anything but one uniform affine group-64 4-bit quantization is refused with
//! a [`MlxQ4Error`] naming the key or tensor: another `bits`, `group_size` or
//! `mode`, per-layer overrides in the config's `quantization` (a mixed
//! checkpoint), a projection that has no `.scales` / `.biases` or whose weight
//! is dense, a wrong dtype or shape.

use std::fmt;
use std::sync::Arc;

use crate::config_json;
use crate::nn::{self, Q4MlxBank, Q4MlxLayout, QuantShape};
use crate::runtime::GpuRuntime;
use crate::safetensors::{Dtype, TensorInfo};
use crate::tensor::{DType, GpuBuffer, Tensor};
use ojas_io::json::JsonValue;

// The peak-footprint reader the bench bins and tests share.
#[path = "bin/common/footprint.rs"]
mod footprint;

/// Bits per weight of the one supported quantization.
pub const MLX_Q4_BITS: u32 = 4;
/// Weights per scale/bias group of the one supported quantization.
pub const MLX_Q4_GROUP: u32 = 64;
/// 4-bit values per packed `U32` word.
const NIBBLES_PER_WORD: usize = 8;

/// Why an MLX 4-bit checkpoint or its `config.json` was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MlxQ4Error {
    /// The config has no `quantization` (or `quantization_config`) object.
    NotQuantized,
    /// The config's `quantization` is not what mlx-lm writes: the key and why.
    MalformedConfig { key: String, why: String },
    /// `bits` is not 4.
    Bits { bits: u32 },
    /// `group_size` is not 64.
    GroupSize { group_size: u32 },
    /// `mode` is not `"affine"` (`mxfp4`, `nvfp4`, `mxfp8` are other formats).
    Mode { mode: String },
    /// `quantization` carries per-layer entries: a mixed checkpoint.
    Mixed { overrides: Vec<String> },
    /// A projection or table whose weight is not packed (it is a dense tensor
    /// in a quantized checkpoint).
    DenseWeight { tensor: String, dtype: String },
    /// A packed weight with no `.scales` or no `.biases`.
    MissingSidecar { tensor: String, missing: &'static str },
    /// A tensor of the wrong dtype.
    Dtype {
        tensor: String,
        found: String,
        want: &'static str,
    },
    /// A tensor of the wrong shape (including a `cols` the group does not divide).
    Shape {
        tensor: String,
        found: Vec<usize>,
        want: Vec<usize>,
    },
    /// The device has no TensorOps, which every prefill of more than one row
    /// needs ([`crate::nn::gemm_q4_mlx_tensorops`]); refused at load rather
    /// than mid-forward.
    NoTensorOps,
    /// A matrix with no rows or no columns: nothing to load, so a config that
    /// asks for one is wrong.
    Empty { tensor: String, rows: usize, cols: usize },
    /// `cols` is not a multiple of the group size, so no group layout exists.
    Cols { tensor: String, cols: usize },
    /// The checkpoint's conv and norm conventions disagree between layers, or
    /// cannot be told (no GDN layer to read them from).
    Convention { why: String },
    /// The checkpoint has no tower under either known key prefix.
    NoTower,
    /// Tensors outside the tower's prefix that are neither the head nor a
    /// group this loader knows to skip.
    Unexpected { count: usize, first: Vec<String> },
}

impl fmt::Display for MlxQ4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotQuantized => write!(
                f,
                "config.json: no \"quantization\" object; this is not an MLX-quantized checkpoint"
            ),
            Self::MalformedConfig { key, why } => write!(f, "config.json: quantization.{key}: {why}"),
            Self::Bits { bits } => write!(
                f,
                "config.json: quantization.bits is {bits}; only {MLX_Q4_BITS}-bit weights are supported"
            ),
            Self::GroupSize { group_size } => write!(
                f,
                "config.json: quantization.group_size is {group_size}; only {MLX_Q4_GROUP} is supported"
            ),
            Self::Mode { mode } => write!(
                f,
                "config.json: quantization.mode is {mode:?}; only \"affine\" is supported"
            ),
            Self::Mixed { overrides } => write!(
                f,
                "config.json: quantization has per-layer entries {overrides:?}: a mixed-precision checkpoint \
                 (other bits, or layers left dense) is not supported; only one uniform affine group-64 4-bit \
                 quantization"
            ),
            Self::DenseWeight { tensor, dtype } => write!(
                f,
                "{tensor}: a {dtype} tensor in a quantized checkpoint (a layer the converter left dense); a \
                 mixed checkpoint is not supported"
            ),
            Self::MissingSidecar { tensor, missing } => {
                write!(f, "{tensor}: packed 4-bit weight with no {missing} tensor beside it")
            }
            Self::Dtype { tensor, found, want } => write!(f, "{tensor}: dtype {found}, expected {want}"),
            Self::Shape { tensor, found, want } => write!(
                f,
                "{tensor}: shape {found:?}, expected {want:?} (4-bit packs eight values per U32 word; a \
                 different width means other bits)"
            ),
            Self::NoTensorOps => write!(
                f,
                "this device has no TensorOps, which the Q4 prefill GEMM (more than one row) needs; an MLX 4-bit \
                 model is refused at load rather than failing mid-forward"
            ),
            Self::Empty { tensor, rows, cols } => {
                write!(f, "{tensor}: a [{rows}, {cols}] matrix has nothing to load")
            }
            Self::Cols { tensor, cols } => write!(
                f,
                "{tensor}: {cols} columns is not a multiple of the group size {MLX_Q4_GROUP}"
            ),
            Self::Convention { why } => write!(f, "checkpoint conventions: {why}"),
            Self::NoTower => write!(
                f,
                "no tensor under \"language_model.model.\" (mlx-lm) or \"model.language_model.\" \
                 (Hugging Face): not a Qwen3.5 checkpoint"
            ),
            Self::Unexpected { count, first } => write!(
                f,
                "{count} tensor(s) outside the tower that are neither its head nor mtp.* / the vision tower: {first:?}"
            ),
        }
    }
}

impl std::error::Error for MlxQ4Error {}

impl From<MlxQ4Error> for String {
    fn from(e: MlxQ4Error) -> String {
        e.to_string()
    }
}

/// A checkpoint's quantization, once it is the one supported kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlxQuant {
    pub bits: u32,
    pub group_size: u32,
}

impl MlxQuant {
    /// Read `config.json`'s `quantization` (and `quantization_config`, which
    /// mlx-lm writes as an identical copy; if both are present they must
    /// agree). Refuses everything but affine, 4 bits, group 64, uniform.
    pub fn from_config_json(text: &str) -> Result<Self, MlxQ4Error> {
        let root = config_json::parse_object(text).map_err(|why| MlxQ4Error::MalformedConfig {
            key: "config.json".into(),
            why,
        })?;
        let a = root.get("quantization");
        let b = root.get("quantization_config");
        let q = match (a, b) {
            (None, None) => return Err(MlxQ4Error::NotQuantized),
            (Some(a), Some(b)) if a != b => {
                return Err(MlxQ4Error::MalformedConfig {
                    key: "quantization_config".into(),
                    why: "differs from quantization".into(),
                })
            }
            (Some(q), _) | (None, Some(q)) => q,
        };
        let JsonValue::Object(entries) = q else {
            return Err(MlxQ4Error::MalformedConfig {
                key: "quantization".into(),
                why: format!("must be an object, got {q:?}"),
            });
        };
        let overrides: Vec<String> = entries
            .iter()
            .map(|(k, _)| k.clone())
            .filter(|k| !matches!(k.as_str(), "group_size" | "bits" | "mode"))
            .collect();
        if !overrides.is_empty() {
            return Err(MlxQ4Error::Mixed { overrides });
        }
        let uint = |k: &str| -> Result<u32, MlxQ4Error> {
            config_json::uint(q, k).map_err(|why| MlxQ4Error::MalformedConfig { key: k.into(), why })
        };
        let (bits, group_size) = (uint("bits")?, uint("group_size")?);
        let mode = config_json::string(q, "mode").map_err(|why| MlxQ4Error::MalformedConfig {
            key: "mode".into(),
            why,
        })?;
        // mlx-lm's own default when the key is absent.
        let mode = mode.unwrap_or("affine");
        if mode != "affine" {
            return Err(MlxQ4Error::Mode { mode: mode.into() });
        }
        if bits != MLX_Q4_BITS {
            return Err(MlxQ4Error::Bits { bits });
        }
        if group_size != MLX_Q4_GROUP {
            return Err(MlxQ4Error::GroupSize { group_size });
        }
        Ok(Self { bits, group_size })
    }
}

/// How a checkpoint spells its tower: the prefix of every layer tensor and the
/// name of an untied head (without its `.weight` / `.scales` / `.biases`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlxNames {
    /// `"language_model.model."` (mlx-lm, mlx-vlm) or `"model.language_model."`.
    pub prefix: &'static str,
    /// `"language_model.lm_head"` or `"lm_head"`.
    pub head: &'static str,
}

impl MlxNames {
    /// Pick the spelling from the tensor names present. Both present is
    /// refused (which one is the model?).
    pub fn detect<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<Self, MlxQ4Error> {
        const MLX: MlxNames = MlxNames {
            prefix: "language_model.model.",
            head: "language_model.lm_head",
        };
        const HF: MlxNames = MlxNames {
            prefix: "model.language_model.",
            head: "lm_head",
        };
        let (mut mlx, mut hf) = (false, false);
        for n in names {
            mlx |= n.starts_with(MLX.prefix);
            hf |= n.starts_with(HF.prefix);
        }
        match (mlx, hf) {
            (true, false) => Ok(MLX),
            (false, true) => Ok(HF),
            (false, false) => Err(MlxQ4Error::NoTower),
            (true, true) => Err(MlxQ4Error::Convention {
                why: "tensors under both \"language_model.model.\" and \"model.language_model.\"".into(),
            }),
        }
    }
}

/// Whether the norm weights are stored as transformers has them (`w`, the
/// kernels add 1) or as `mlx_lm.convert` writes them (`1 + w`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormConvention {
    /// `conv1d.weight` is `[conv_dim, 1, k]`: an unsanitized checkpoint.
    Hf,
    /// `conv1d.weight` is `[conv_dim, k, 1]`: `Qwen3_5.sanitize` ran, and the
    /// zero-centred norms hold `1 + w`.
    MlxShifted,
}

impl NormConvention {
    /// The value to hand a kernel that adds 1 itself, from a stored norm weight.
    pub fn unshift(self, stored: f32) -> f32 {
        match self {
            Self::Hf => stored,
            Self::MlxShifted => stored - 1.0,
        }
    }

    /// The shape of a GDN layer's `conv1d.weight` in this convention.
    pub fn conv_shape(self, conv_dim: usize, kernel: usize) -> [usize; 3] {
        match self {
            Self::Hf => [conv_dim, 1, kernel],
            Self::MlxShifted => [conv_dim, kernel, 1],
        }
    }

    /// Which convention a `conv1d.weight` of `shape` is in.
    pub fn from_conv_shape(tensor: &str, shape: &[usize], conv_dim: usize, kernel: usize) -> Result<Self, MlxQ4Error> {
        for c in [Self::Hf, Self::MlxShifted] {
            if shape == c.conv_shape(conv_dim, kernel) {
                return Ok(c);
            }
        }
        Err(MlxQ4Error::Shape {
            tensor: tensor.into(),
            found: shape.to_vec(),
            want: Self::Hf.conv_shape(conv_dim, kernel).to_vec(),
        })
    }
}

/// The tensors outside the tower's prefix that a quantized load skipped, by
/// name: never silently. See [`classify_outside`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OutsideTensors {
    /// `mtp.*` (the multi-token-prediction head: a drafter, not this forward),
    /// by name.
    pub mtp: Vec<String>,
    /// Vision-tower tensors, by name.
    pub vision: Vec<String>,
}

/// Sort `unread`, the checkpoint's tensors outside the tower's prefix that no
/// read touched, into what is skipped on purpose (`mtp.*`, the vision tower)
/// and what is not: anything else is refused.
pub fn classify_outside<'a>(unread: impl IntoIterator<Item = &'a str>) -> Result<OutsideTensors, MlxQ4Error> {
    const VISION: [&str; 4] = ["vision_tower.", "model.visual.", "visual.", "model.vision_tower."];
    let mut out = OutsideTensors::default();
    let mut other = Vec::new();
    for n in unread {
        if n.starts_with("mtp.") {
            out.mtp.push(n.to_string());
        } else if VISION.iter().any(|p| n.starts_with(p)) {
            out.vision.push(n.to_string());
        } else {
            other.push(n.to_string());
        }
    }
    if other.is_empty() {
        return Ok(out);
    }
    Err(MlxQ4Error::Unexpected {
        count: other.len(),
        first: other.into_iter().take(8).collect(),
    })
}

/// The `.scales` / `.biases` names of the weight `X.weight`.
pub fn sidecar_names(weight: &str) -> Option<(String, String)> {
    let base = weight.strip_suffix(".weight")?;
    Some((format!("{base}.scales"), format!("{base}.biases")))
}

/// Check one quantized matrix `[rows, cols]` against its three tensors'
/// headers; `scales` / `biases` are `None` when absent from the checkpoint.
/// Pure: no tensor data, no device.
pub fn check_part(
    weight_name: &str,
    weight: &TensorInfo,
    scales: Option<&TensorInfo>,
    biases: Option<&TensorInfo>,
    rows: usize,
    cols: usize,
) -> Result<(), MlxQ4Error> {
    let group = MLX_Q4_GROUP as usize;
    if rows == 0 || cols == 0 {
        return Err(MlxQ4Error::Empty {
            tensor: weight_name.into(),
            rows,
            cols,
        });
    }
    if !cols.is_multiple_of(group) || !cols.is_multiple_of(NIBBLES_PER_WORD) {
        return Err(MlxQ4Error::Cols {
            tensor: weight_name.into(),
            cols,
        });
    }
    if weight.dtype != Dtype::U32 {
        return Err(MlxQ4Error::DenseWeight {
            tensor: weight_name.into(),
            dtype: format!("{:?}", weight.dtype),
        });
    }
    let want = vec![rows, cols / NIBBLES_PER_WORD];
    let weight_shape = weight.shape_usize();
    if weight_shape != want {
        return Err(MlxQ4Error::Shape {
            tensor: weight_name.into(),
            found: weight_shape,
            want,
        });
    }
    let (sn, bn) = sidecar_names(weight_name).ok_or_else(|| MlxQ4Error::Dtype {
        tensor: weight_name.into(),
        found: "a name that does not end in .weight".into(),
        want: "X.weight",
    })?;
    for (name, info, missing) in [(sn, scales, "scales"), (bn, biases, "biases")] {
        let info = info.ok_or_else(|| MlxQ4Error::MissingSidecar {
            tensor: weight_name.into(),
            missing,
        })?;
        if info.dtype != Dtype::BF16 {
            return Err(MlxQ4Error::Dtype {
                tensor: name,
                found: format!("{:?}", info.dtype),
                want: "BF16 (a bf16 model's scales; float16 checkpoints are not supported)",
            });
        }
        let want = vec![rows, cols / group];
        let info_shape = info.shape_usize();
        if info_shape != want {
            return Err(MlxQ4Error::Shape {
                tensor: name,
                found: info_shape,
                want,
            });
        }
    }
    Ok(())
}

/// A quantized `[rows, cols]` matrix on the device in [`Q4MlxBank`] form,
/// row-major: the rows of several checkpoint tensors concatenated, which is
/// what a packed projection window is for a bank (a dense window concatenates
/// output columns, a bank's output rows).
pub(crate) struct Q4Weight {
    packed: GpuBuffer,
    scales_biases: GpuBuffer,
    rows: u32,
    cols: u32,
}

impl Q4Weight {
    /// Unwritten hot buffers for `rows x cols` (the loader writes every byte).
    pub(crate) fn alloc(rt: &Arc<GpuRuntime>, rows: usize, cols: usize) -> Result<Self, String> {
        let group = MLX_Q4_GROUP as usize;
        let (rows32, cols32) = match (u32::try_from(rows), u32::try_from(cols)) {
            (Ok(r), Ok(c)) if (c as usize).is_multiple_of(group) => (r, c),
            _ => return Err(format!("Q4Weight: {rows} x {cols} does not fit a group-{group} bank")),
        };
        let weights = rows
            .checked_mul(cols)
            .ok_or_else(|| format!("Q4Weight: {rows} x {cols} overflows"))?;
        let packed = rt.alloc_buffer_hot(weights / 2)?;
        let scales_biases = rt.alloc_buffer_hot(rows * (cols / group) * 4)?;
        Ok(Self {
            packed,
            scales_biases,
            rows: rows32,
            cols: cols32,
        })
    }

    /// Device bytes of the bank.
    pub(crate) fn nbytes(&self) -> u64 {
        (self.packed.nbytes() + self.scales_biases.nbytes()) as u64
    }

    /// The packed nibbles (two per byte, low first).
    pub(crate) fn packed(&self) -> &GpuBuffer {
        &self.packed
    }

    pub(crate) fn scales_biases(&self) -> &GpuBuffer {
        &self.scales_biases
    }

    pub(crate) fn bank(&self) -> Q4MlxBank<'_> {
        Q4MlxBank {
            packed: &self.packed,
            scales_biases: &self.scales_biases,
        }
    }

    fn shape(&self) -> QuantShape {
        QuantShape {
            rows: self.rows,
            cols: self.cols,
            group_size: MLX_Q4_GROUP,
        }
    }

    /// `y[m, rows] = x[m, cols] @ Wᵀ`. `x` is bf16, `y` f32, both whole tensors.
    /// One row runs the decode GEMV ([`nn::gemv_q4_mlx_simd`]), and fuses
    /// `resid` when it is set. More than one row runs
    /// [`nn::gemm_q4_mlx_tensorops`], which dequantizes inside the matmul and
    /// has no fused residual: `resid` there is an error, and the caller adds
    /// it afterwards. Both read the same bank. A group, device, or shape the
    /// cooperative kernel refuses is that kernel's error, returned before it
    /// encodes.
    pub(crate) fn matmul(
        &self,
        rt: &Arc<GpuRuntime>,
        x: &Tensor,
        y: &Tensor,
        resid: Option<&GpuBuffer>,
    ) -> Result<(), String> {
        const WHAT: &str = "Q4Weight::matmul";
        let (m, cols) = match x.shape() {
            &[m, c] => (m, c),
            s => return Err(format!("{WHAT}: x must be [rows, cols], got {s:?}")),
        };
        if x.dtype != DType::BF16 || y.dtype != DType::F32 {
            return Err(format!(
                "{WHAT}: x must be bf16 and y f32, got {:?} and {:?}",
                x.dtype, y.dtype
            ));
        }
        if x.byte_offset() != 0 || y.byte_offset() != 0 {
            return Err(format!("{WHAT}: x and y must be whole tensors, not views"));
        }
        if cols != self.cols as usize || y.shape() != [m, self.rows as usize] {
            return Err(format!(
                "{WHAT}: x {:?} and y {:?} do not fit a [{}, {}] weight",
                x.shape(),
                y.shape(),
                self.rows,
                self.cols
            ));
        }
        let m = u32::try_from(m).map_err(|_| format!("{WHAT}: {m} rows exceed u32"))?;
        let (bank, shape) = (self.bank(), self.shape());
        match m {
            0 => Ok(()),
            1 => nn::gemv_q4_mlx_simd(rt, bank, &x.buffer, &y.buffer, shape, Q4MlxLayout::RowMajor, resid),
            _ if resid.is_some() => Err(format!(
                "{WHAT}: {m} rows run on the TensorOps GEMM, which has no fused residual add"
            )),
            _ => nn::gemm_q4_mlx_tensorops(rt, bank, &x.buffer, &y.buffer, shape, m),
        }
    }

    /// `mid[1, rows] = silu(x @ self.Wᵀ) * (x @ up.Wᵀ)` for one row, in one
    /// dispatch ([`nn::gemv_q4_mlx_gate_up_silu`]): `self` is the gate and `up`
    /// the up projection, both `[rows, cols]`. `x` is bf16 `[1, cols]`; `mid` is
    /// bf16 or f32 `[1, rows]`, both whole tensors. The same bits as
    /// [`Self::matmul`] twice into f32 and `qwen35::swiglu` into `mid`'s dtype.
    pub(crate) fn gate_up_silu(
        &self,
        up: &Q4Weight,
        rt: &Arc<GpuRuntime>,
        x: &Tensor,
        mid: &Tensor,
    ) -> Result<(), String> {
        const WHAT: &str = "Q4Weight::gate_up_silu";
        if (self.rows, self.cols) != (up.rows, up.cols) {
            return Err(format!(
                "{WHAT}: gate is [{}, {}] but up is [{}, {}]",
                self.rows, self.cols, up.rows, up.cols
            ));
        }
        let mid_bf16 = match mid.dtype {
            DType::BF16 => true,
            DType::F32 => false,
            other => return Err(format!("{WHAT}: mid must be bf16 or f32, got {other:?}")),
        };
        if x.dtype != DType::BF16 {
            return Err(format!("{WHAT}: x must be bf16, got {:?}", x.dtype));
        }
        if x.byte_offset() != 0 || mid.byte_offset() != 0 {
            return Err(format!("{WHAT}: x and mid must be whole tensors, not views"));
        }
        if x.shape() != [1, self.cols as usize] || mid.shape() != [1, self.rows as usize] {
            return Err(format!(
                "{WHAT}: x {:?} and mid {:?} do not fit a [{}, {}] weight at one row",
                x.shape(),
                mid.shape(),
                self.rows,
                self.cols
            ));
        }
        nn::gemv_q4_mlx_gate_up_silu(
            rt,
            self.bank(),
            up.bank(),
            &x.buffer,
            &mid.buffer,
            self.shape(),
            mid_bf16,
        )
    }

    /// `out[n, cols] = W[ids[i], :]` in f32 (the embedding gather).
    pub(crate) fn gather(&self, rt: &Arc<GpuRuntime>, ids: &GpuBuffer, n: u32, out: &GpuBuffer) -> Result<(), String> {
        nn::embed_lookup_q4_mlx(rt, self.bank(), ids, out, self.rows, self.cols, MLX_Q4_GROUP, n)
    }
}

/// One weight of the model: dense in the forward's precision, or MLX 4-bit.
///
/// Training, the parameter table and the optimizer read dense tensors only.
/// They reach them through [`Weight::dense`], which returns a typed error for a
/// quantized weight, so a path that forgets the up-front
/// [`crate::qwen35_model::Qwen35Model::require_dense`] refusal still fails with
/// a message, never a crash.
pub(crate) enum Weight {
    Dense(Tensor),
    Q4(Q4Weight),
}

impl Weight {
    pub(crate) fn q4(&self) -> Option<&Q4Weight> {
        match self {
            Self::Q4(q) => Some(q),
            Self::Dense(_) => None,
        }
    }

    /// The dense tensor, or an error for a quantized weight.
    pub(crate) fn dense(&self) -> Result<&Tensor, String> {
        match self {
            Self::Dense(t) => Ok(t),
            Self::Q4(q) => Err(format!(
                "a [{}, {}] MLX 4-bit weight has no dense tensor: training, parameter and optimizer paths run on \
                 dense models only",
                q.rows, q.cols
            )),
        }
    }
}

/// What a quantized load measured about itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlxQ4LoadReport {
    /// Device bytes of the quantized banks (packed nibbles and scale/bias pairs).
    pub q4_bytes: u64,
    /// The PROCESS LIFETIME peak physical footprint (`ri_lifetime_max_phys_footprint`,
    /// what `/usr/bin/time -l` prints) read before the load began, in bytes. It
    /// is a maximum over everything the process has ever done, not a measure of
    /// the load.
    pub peak_footprint_before: u64,
    /// The process lifetime peak read after the weights were on the device and
    /// the load had synchronized. It measures the load only when it exceeds
    /// [`Self::peak_footprint_before`]; equal means the load stayed under what
    /// the process had already reached and says nothing about its own peak. A
    /// meaningful figure needs a process with nothing large allocated before
    /// the load.
    pub peak_footprint_after: u64,
    /// `mtp.*` and vision tensors present and skipped, every one by name.
    pub skipped: OutsideTensors,
    /// The conventions the checkpoint was read under.
    pub convention: NormConvention,
    /// The key spelling the checkpoint was read under.
    pub names: MlxNames,
}

/// The process's peak footprint now; refuses rather than report 0.
pub(crate) fn peak_footprint_now() -> Result<u64, String> {
    footprint::peak_footprint()
}

#[cfg(test)]
mod tests {
    use super::Q4Weight;
    use crate::qwen35::{self, Cols};
    use crate::runtime::{kernel_trace_enabled, traced_kernel_count};
    use crate::tensor::{bf16_bits_to_f32, f32_to_bf16_bits};
    use crate::GpuRuntime;
    use std::sync::Arc;

    const N: usize = 32;
    const K: usize = 256;
    const GROUP: usize = 64;
    const SENTINEL: f32 = -7.7e28;

    fn in_traced_child(test_name: &str, body: impl FnOnce()) -> Option<String> {
        const CHILD: &str = "TESSL_TRACED_CHILD";
        if std::env::var_os(CHILD).is_some() {
            body();
            return None;
        }
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args(["--exact", "--test-threads=1", "--nocapture", test_name])
            .env(CHILD, "1")
            .env("TESSL_KERNEL_TRACE", "1")
            .output()
            .expect("spawn child test");
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(out.status.success(), "child {test_name} failed:\n{stdout}\n{stderr}");
        assert!(
            stdout.contains("1 passed"),
            "child ran no test (filter out of sync with {test_name}?):\n{stdout}"
        );
        Some(stdout)
    }

    fn round_bf16(x: f32) -> f32 {
        bf16_bits_to_f32(f32_to_bf16_bits(x))
    }

    fn gamma(terms: usize) -> f64 {
        let tu = terms as f64 * 2f64.powi(-24);
        tu / (1.0 - tu)
    }

    /// f64 product of the loaded operands: `s * q * x + b * xsum` per group.
    fn reference(x: &[f32], nibbles: &[u8], sb: &[f32], m: usize) -> (Vec<f64>, Vec<f64>) {
        let groups = K / GROUP;
        let mut want = vec![0.0f64; m * N];
        let mut mag = vec![0.0f64; m * N];
        for mi in 0..m {
            for ni in 0..N {
                let (mut acc, mut s_mag) = (0.0, 0.0);
                for g in 0..groups {
                    let (s, b) = (sb[(ni * groups + g) * 2] as f64, sb[(ni * groups + g) * 2 + 1] as f64);
                    let (mut qx, mut aqx, mut xsum, mut ax) = (0.0, 0.0, 0.0, 0.0);
                    for j in 0..GROUP {
                        let xv = x[mi * K + g * GROUP + j] as f64;
                        let q = nibbles[ni * K + g * GROUP + j] as f64;
                        qx += q * xv;
                        aqx += q * xv.abs();
                        xsum += xv;
                        ax += xv.abs();
                    }
                    acc += s * qx + b * xsum;
                    s_mag += s.abs() * aqx + b.abs() * ax;
                }
                want[mi * N + ni] = acc;
                mag[mi * N + ni] = s_mag;
            }
        }
        (want, mag)
    }

    fn le_bytes(bits: &[u16]) -> Vec<u8> {
        let mut out = Vec::with_capacity(bits.len() * 2);
        for b in bits {
            out.extend_from_slice(&b.to_le_bytes());
        }
        out
    }

    struct Bank {
        w: Q4Weight,
        nibbles: Vec<u8>,
        sb: Vec<f32>,
        packed: Vec<u8>,
    }

    fn bank(rt: &Arc<GpuRuntime>) -> Bank {
        let mut nibbles = vec![0u8; N * K];
        let mut sb = vec![0f32; N * (K / GROUP) * 2];
        for (i, q) in nibbles.iter_mut().enumerate() {
            *q = ((i * 3) % 16) as u8;
        }
        for (i, v) in sb.iter_mut().enumerate() {
            let u = ((i * 17) % 64) as f32 / 64.0;
            *v = round_bf16(if i % 2 == 0 {
                0.03125 + 0.0625 * u
            } else {
                -0.25 + 0.5 * u
            });
        }
        let mut packed = vec![0u8; N * K / 2];
        for (i, &q) in nibbles.iter().enumerate() {
            if i % 2 == 0 {
                packed[i / 2] |= q;
            } else {
                packed[i / 2] |= q << 4;
            }
        }
        let mut sb_bits = Vec::with_capacity(sb.len());
        for &v in &sb {
            sb_bits.push(f32_to_bf16_bits(v));
        }
        let w = Q4Weight::alloc(rt, N, K).unwrap();
        w.packed().write_bytes(&packed);
        w.scales_biases().write_bytes(&le_bytes(&sb_bits));
        Bank { w, nibbles, sb, packed }
    }

    fn activation(rt: &Arc<GpuRuntime>, m: usize, seed: u32) -> (crate::Tensor, Vec<f32>) {
        let x: Vec<f32> = (0..m * K)
            .map(|i| round_bf16((((i as u32).wrapping_mul(seed)) % 128) as f32 / 64.0 - 1.0))
            .collect();
        let bits: Vec<u16> = x.iter().copied().map(f32_to_bf16_bits).collect();
        let t = rt.alloc_tensor_bf16(&[m, K]).unwrap();
        t.buffer.write_bytes(&le_bytes(&bits));
        (t, x)
    }

    fn assert_close(got: &[f32], want: &[f64], mag: &[f64], what: &str) {
        let bound = gamma(K + 2 * (K / GROUP) + GROUP);
        let mut worst = 0.0f64;
        for i in 0..got.len() {
            assert!(got[i].is_finite() && got[i] != SENTINEL, "{what}: unwritten {i}");
            let err = (got[i] as f64 - want[i]).abs();
            let allow = bound * mag[i] + 1e-6;
            let frac = if allow == 0.0 { err } else { err / allow };
            worst = worst.max(frac);
            assert!(
                err <= allow,
                "{what} [{i}]: |{err}| > {allow} (got {} want {})",
                got[i],
                want[i]
            );
        }
        println!("{what}: worst fraction {worst:.4}");
    }

    fn counts() -> (u64, u64, u64, u64) {
        (
            traced_kernel_count("gemv_q4_mlx_simd"),
            traced_kernel_count("gemv_q4_mlx_simd_add"),
            traced_kernel_count("gemm_q4_mlx_simd"),
            traced_kernel_count("gemm_q4_mlx_tensorops_g64"),
        )
    }

    /// Shipped `Q4Weight::matmul`: M=1 is the GEMV, M>1 (including 2..=8) is
    /// the cooperative TensorOps matmul, and a fused residual for M>1 is the
    /// existing error. The model adds that residual afterwards.
    #[test]
    fn q4_weight_matmul_routes_cooperative() {
        const NAME: &str = "qwen35_q4::tests::q4_weight_matmul_routes_cooperative";
        let Some(out) = in_traced_child(NAME, run) else {
            return;
        };
        println!("{out}");
    }

    fn run() {
        assert!(kernel_trace_enabled(), "TESSL_KERNEL_TRACE did not arm the trace");
        let rt = GpuRuntime::new().expect("Metal device");
        let bank = bank(&rt);
        let packed_before = bank.w.packed().contents_u8()[..bank.packed.len()].to_vec();

        // Refused shape: no encode.
        let (x_bad, _) = activation(&rt, 2, 3);
        let y_bad = rt.alloc_tensor_f32(&[2, N + 1]).unwrap();
        let before = counts();
        let err = bank.w.matmul(&rt, &x_bad, &y_bad, None).unwrap_err();
        assert!(err.contains("do not fit"), "{err}");
        assert_eq!(counts(), before, "a refused shape encoded a kernel");

        // M>1 fused residual: the explicit error, and the add is not dropped
        // when the caller does what `project_residual` does.
        for m in [2usize, 8, 16] {
            let (x, xv) = activation(&rt, m, 9 + m as u32);
            let y = rt.alloc_tensor_f32(&[m, N]).unwrap();
            y.write_f32(&vec![SENTINEL; m * N]).unwrap();
            let before = counts();
            let err = bank.w.matmul(&rt, &x, &y, Some(&y.buffer)).unwrap_err();
            assert!(err.contains("no fused residual add"), "{err}");
            assert_eq!(counts(), before, "M={m} residual refusal encoded a kernel");
            assert!(
                y.read_f32().unwrap().iter().all(|v| *v == SENTINEL),
                "M={m} residual refusal wrote y"
            );

            y.write_f32(&vec![SENTINEL; m * N]).unwrap();
            bank.w.matmul(&rt, &x, &y, None).unwrap();
            rt.synchronize().unwrap();
            let (want, mag) = reference(&xv, &bank.nibbles, &bank.sb, m);
            let got = y.read_f32().unwrap();
            assert_close(&got, &want, &mag, &format!("Q4Weight M={m}"));
            let after = counts();
            assert_eq!(after.3, before.3 + 1, "M={m} did not request gemm_q4_mlx_tensorops_g64");
            assert_eq!(after.0, before.0, "M={m} requested the GEMV");
            assert_eq!(after.2, before.2, "M={m} requested gemm_q4_mlx_simd");

            let resid = rt.alloc_buffer(m * N * 4).unwrap();
            let resid0 = vec![0.25f32; m * N];
            resid.write_f32(&resid0);
            qwen35::residual_add(
                &rt,
                Cols::dense(&y.buffer, N as u32),
                Cols::dense(&resid, N as u32),
                m as u32,
                N as u32,
            )
            .unwrap();
            rt.synchronize().unwrap();
            let summed = resid.read_f32();
            for i in 0..m * N {
                let expect = want[i] + resid0[i] as f64;
                let err = (summed[i] as f64 - expect).abs();
                let allow = gamma(K + 2 * (K / GROUP) + GROUP) * mag[i] + 1e-5;
                assert!(err <= allow, "residual add M={m} [{i}]: {err} > {allow}");
            }
        }

        // M=1 stays on the GEMV, with and without the fused residual.
        let (x, xv) = activation(&rt, 1, 11);
        let y = rt.alloc_tensor_f32(&[1, N]).unwrap();
        y.write_f32(&[SENTINEL; N]).unwrap();
        let before = counts();
        bank.w.matmul(&rt, &x, &y, None).unwrap();
        rt.synchronize().unwrap();
        let (want, mag) = reference(&xv, &bank.nibbles, &bank.sb, 1);
        assert_close(&y.read_f32().unwrap(), &want, &mag, "Q4Weight M=1");
        let after = counts();
        assert_eq!(after.0, before.0 + 1, "M=1 did not request gemv_q4_mlx_simd");
        assert_eq!(after.3, before.3, "M=1 requested the cooperative GEMM");

        let resid0 = vec![0.25f32; N];
        y.write_f32(&resid0).unwrap();
        let before = counts();
        bank.w.matmul(&rt, &x, &y, Some(&y.buffer)).unwrap();
        rt.synchronize().unwrap();
        let got = y.read_f32().unwrap();
        for i in 0..N {
            let expect = want[i] + resid0[i] as f64;
            let err = (got[i] as f64 - expect).abs();
            let allow = gamma(K + 2 * (K / GROUP) + GROUP) * mag[i] + 1e-5;
            assert!(err <= allow, "M=1 fused residual [{i}]: {err} > {allow}");
        }
        let after = counts();
        assert_eq!(
            after.1,
            before.1 + 1,
            "M=1 residual did not request gemv_q4_mlx_simd_add"
        );
        assert_eq!(after.3, before.3, "M=1 residual requested the cooperative GEMM");

        let packed_after = bank.w.packed().contents_u8()[..bank.packed.len()].to_vec();
        assert_eq!(packed_before, packed_after, "host dequant rewrote the packed weights");
        assert_eq!(packed_before, bank.packed);
        println!("q4 weight router: M=1 GEMV, M=2/8/16 cooperative, residual kept");
    }
}
