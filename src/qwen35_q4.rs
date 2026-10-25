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
use crate::json::Json;
use crate::nn::{self, Q4MlxBank, Q4MlxLayout, QuantShape, GEMM_Q4_MLX_MAX_M};
use crate::runtime::GpuRuntime;
use crate::safetensors::{Dtype, TensorInfo};
use crate::tensor::{DType, GpuBuffer, Tensor};

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
    /// The device has no TensorOps, which every prefill of more than 8 rows
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
                "this device has no TensorOps, which the Q4 prefill GEMM (more than 8 rows) needs; an MLX 4-bit \
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
        let Json::Object(entries) = q else {
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
    if cols % group != 0 || cols % NIBBLES_PER_WORD != 0 {
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
    if weight.shape != want {
        return Err(MlxQ4Error::Shape {
            tensor: weight_name.into(),
            found: weight.shape.clone(),
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
        if info.shape != want {
            return Err(MlxQ4Error::Shape {
                tensor: name,
                found: info.shape.clone(),
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
            (Ok(r), Ok(c)) if c as usize % group == 0 => (r, c),
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

    /// `y[m, rows] = x[m, cols] @ Wᵀ` (+ `resid`, elementwise, for `m` up to
    /// [`GEMM_Q4_MLX_MAX_M`]). `x` is bf16, `y` f32, both whole tensors. One
    /// row runs the decode GEMV, two to eight rows the small-batch kernel (one
    /// read of the weights for all of them), and more the TensorOps Q4 GEMM
    /// ([`nn::gemm_q4_mlx_tensorops`]; no fused residual, so `resid` there is
    /// refused: the caller adds it). All three read the same bank.
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
            2..=GEMM_Q4_MLX_MAX_M => {
                nn::gemm_q4_mlx(rt, bank, &x.buffer, &y.buffer, shape, m, Q4MlxLayout::RowMajor, resid)
            }
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
