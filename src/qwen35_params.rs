//! A Qwen3.5 model's parameters and gradients as transformers names them,
//! copied between the model and caller tensors on the GPU, so a training loop
//! outside tessl (torch, through [`crate::capi`]) can run the optimizer.
//!
//! Each entry is one `Qwen3_5ForCausalLM` parameter, named as transformers
//! names it below the text tower (`layers.3.mlp.gate_proj.weight`, without
//! the checkpoint's `model.language_model.` prefix), and its value is the
//! parameter's own value, whatever layout tessl stores it in:
//!
//! - The zero-centred norms (`input_layernorm`, `post_attention_layernorm`,
//!   the final `norm`) are stored as `w`, as transformers holds them; their
//!   kernels add the 1, so a write and a read are exact.
//! - The linear layers live packed side by side as `[in, sum(out)]` right
//!   operands; each reads and writes as its own `[in, out]` window, the
//!   transpose of transformers' `[out, in]` ([`ParamInfo::transposed`]).
//! - The tied embedding is one `[vocab, hidden]` table, read by the gather,
//!   the training step's cross-entropy and the inference forward's head
//!   alike, so a write moves all three.
//!
//! The matrices (the embedding and the packed projections) are stored in the
//! model's [`Precision`]; everything else is f32 at either precision. The
//! caller's tensors are always f32: a bf16 matrix reads out widened (exact)
//! and a write rounds to nearest. Gradients come in the same layouts and the
//! same rule (a bf16 bank's matrices are bf16). Every copy is a GPU dispatch,
//! since a caller's buffers may be GPU-private.

use std::sync::Arc;

use crate::dispatch::{dispatch_2d, set_gpu_buf_offset, set_u32};
use crate::nn::require_runtime;
use crate::qwen35_model::{Mixer, Precision, Qwen35Model};
use crate::qwen35_train::{MixerGrads, Qwen35Grads};
use crate::runtime::GpuRuntime;
use crate::tensor::{DType, GpuBuffer, Tensor};

/// One parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamInfo {
    /// transformers' name below the text tower.
    pub name: String,
    /// transformers' shape.
    pub shape: Vec<usize>,
    /// The caller's tensor holds the transpose, `[in, out]`, of the 2-D
    /// `[out, in]` `shape`.
    pub transposed: bool,
}

impl ParamInfo {
    /// The shape of the dense f32 tensor a copy reads or writes.
    pub fn storage_shape(&self) -> Vec<usize> {
        if self.transposed {
            vec![self.shape[1], self.shape[0]]
        } else {
            self.shape.clone()
        }
    }
}

/// Where a value lives in tessl.
#[derive(Clone, Copy)]
pub(crate) enum Src<'a> {
    /// An f32 buffer holding exactly the value.
    Raw(&'a GpuBuffer),
    /// Columns `[off, off + width)` of a packed `[in, total]` matrix (f32 or
    /// bf16).
    Packed(&'a Tensor, usize),
    /// A dense tensor (f32 or bf16) holding exactly the value.
    Dense(&'a Tensor),
}

impl Src<'_> {
    fn buffer(&self) -> &GpuBuffer {
        match self {
            Src::Raw(b) => b,
            Src::Packed(t, _) | Src::Dense(t) => &t.buffer,
        }
    }
}

/// One value's storage as a window: a buffer, the byte offset of its tensor,
/// its element type, and `rows` x `width` at leading dimension `ld` from
/// element `off`. Window element `i` (row `i / width`, column `i % width`) is
/// element `i` of the caller's dense [`ParamInfo::storage_shape`] tensor.
#[derive(Clone, Copy)]
pub(crate) struct Window<'a> {
    pub(crate) buf: &'a GpuBuffer,
    pub(crate) byte_off: usize,
    pub(crate) dtype: DType,
    pub(crate) rows: usize,
    pub(crate) width: usize,
    pub(crate) ld: usize,
    pub(crate) off: usize,
}

impl<'a> Window<'a> {
    pub(crate) fn of(src: Src<'a>, shape: &[usize]) -> Self {
        let numel = shape.iter().product::<usize>();
        match src {
            Src::Raw(b) => Self::dense(b, 0, DType::F32, numel),
            Src::Dense(t) => Self::dense(&t.buffer, t.byte_offset(), t.dtype, numel),
            // transformers' [out, in]: `in` rows of `out` columns in the packed [in, total].
            Src::Packed(t, off) => Window {
                buf: &t.buffer,
                byte_off: t.byte_offset(),
                dtype: t.dtype,
                rows: shape[1],
                width: shape[0],
                ld: t.shape()[1],
                off,
            },
        }
    }

    /// `n` contiguous elements as one row.
    pub(crate) fn dense(buf: &'a GpuBuffer, byte_off: usize, dtype: DType, n: usize) -> Self {
        Window {
            buf,
            byte_off,
            dtype,
            rows: 1,
            width: n,
            ld: n,
            off: 0,
        }
    }

    /// A caller's dense tensor, read as `like`'s rows and columns so window
    /// row `r` of `like` meets row `r` here.
    pub(crate) fn tensor_as(t: &'a Tensor, like: &Window<'_>) -> Result<Self, String> {
        if t.numel() != like.numel() {
            return Err(format!("{} elements for a window of {}", t.numel(), like.numel()));
        }
        Ok(Window {
            buf: &t.buffer,
            byte_off: t.byte_offset(),
            dtype: t.dtype,
            rows: like.rows,
            width: like.width,
            ld: like.width,
            off: 0,
        })
    }

    /// The window lies inside its buffer, is f32 or bf16, and its geometry
    /// fits the kernels' `uint`s.
    pub(crate) fn check(&self, what: &str) -> Result<(), String> {
        if !matches!(self.dtype, DType::F32 | DType::BF16) {
            return Err(format!("{what}: {:?} storage is not f32 or bf16", self.dtype));
        }
        let size = self.dtype.size_of();
        if self.rows == 0 || self.width == 0 {
            return Err(format!("{what}: empty window"));
        }
        let last = (self.rows - 1)
            .checked_mul(self.ld)
            .and_then(|x| x.checked_add(self.off + self.width))
            .ok_or_else(|| format!("{what}: window overflows"))?;
        let have = self.buf.nbytes().saturating_sub(self.byte_off) / size;
        if self.byte_off % size != 0 || last > have || self.width > self.ld {
            return Err(format!(
                "{what}: window of {last} elements does not fit its buffer's {have}"
            ));
        }
        for n in [self.rows, self.width, self.ld, self.off, self.numel()] {
            u32::try_from(n).map_err(|_| format!("{what}: {n} exceeds u32"))?;
        }
        Ok(())
    }

    pub(crate) fn numel(&self) -> usize {
        self.rows * self.width
    }

    /// The same rows, columns and leading dimension (dtypes may differ).
    pub(crate) fn same_layout(&self, o: &Window<'_>) -> bool {
        (self.rows, self.width, self.ld, self.off) == (o.rows, o.width, o.ld, o.off)
    }
}

/// `dst = src` element for element between two checked windows of the same
/// `rows` x `width`: f32 to bf16 rounds to nearest, bf16 to f32 is exact,
/// same-type is exact. Encoded, not waited for.
pub(crate) fn window_copy(rt: &GpuRuntime, src: &Window<'_>, dst: &Window<'_>) -> Result<(), String> {
    const WHAT: &str = "window_copy";
    src.check(&format!("{WHAT} src"))?;
    dst.check(&format!("{WHAT} dst"))?;
    let (rows, width) = (src.rows, src.width);
    if (dst.rows, dst.width) != (rows, width) {
        return Err(format!(
            "{WHAT}: a {rows}x{width} source into a {}x{} destination",
            dst.rows, dst.width
        ));
    }
    let name = match (src.dtype, dst.dtype) {
        (DType::F32, DType::F32) => "qwen35_window_copy_f32_f32",
        (DType::BF16, DType::F32) => "qwen35_window_copy_bf16_f32",
        (DType::F32, DType::BF16) => "qwen35_window_copy_f32_bf16",
        (s, d) => return Err(format!("{WHAT}: no {s:?} -> {d:?} copy")),
    };
    let p = rt.pipeline(name)?;
    dispatch_2d(rt, &p, width, rows, |bnd| {
        set_gpu_buf_offset(bnd, src.buf, src.byte_off, 0);
        set_gpu_buf_offset(bnd, dst.buf, dst.byte_off, 1);
        set_u32(bnd, rows as u32, 2);
        set_u32(bnd, width as u32, 3);
        set_u32(bnd, src.ld as u32, 4);
        set_u32(bnd, src.off as u32, 5);
        set_u32(bnd, dst.ld as u32, 6);
        set_u32(bnd, dst.off as u32, 7);
    })
}

pub(crate) struct Slot<'a> {
    pub(crate) info: ParamInfo,
    pub(crate) param: Src<'a>,
    pub(crate) grad: Option<Src<'a>>,
}

impl Slot<'_> {
    pub(crate) fn param_window(&self) -> Window<'_> {
        Window::of(self.param, &self.info.shape)
    }

    pub(crate) fn grad_window(&self) -> Option<Window<'_>> {
        self.grad.map(|g| Window::of(g, &self.info.shape))
    }
}

fn info(name: String, shape: &[usize], transposed: bool) -> ParamInfo {
    ParamInfo {
        name,
        shape: shape.to_vec(),
        transposed,
    }
}

/// Offsets of each part in a packed projection, from its widths.
fn offsets<const N: usize>(widths: [usize; N]) -> [usize; N] {
    let mut out = [0; N];
    for i in 1..N {
        out[i] = out[i - 1] + widths[i - 1];
    }
    out
}

/// Every parameter in a fixed order, with its gradient in `grads` if given.
pub(crate) fn slots<'a>(m: &'a Qwen35Model, grads: Option<&'a Qwen35Grads>) -> Result<Vec<Slot<'a>>, String> {
    let cfg = &m.cfg;
    let (h, inter, vocab) = (cfg.hidden as usize, cfg.intermediate as usize, cfg.vocab as usize);
    if let Some(g) = grads {
        if g.layers.len() != m.layers.len() {
            return Err(format!(
                "gradients for {} layers, the model has {}",
                g.layers.len(),
                m.layers.len()
            ));
        }
    }
    let mut out = vec![Slot {
        info: info("embed_tokens.weight".into(), &[vocab, h], false),
        param: Src::Dense(&m.embed),
        grad: grads.map(|g| Src::Dense(&g.embed)),
    }];
    for (l, layer) in m.layers.iter().enumerate() {
        let lg = grads.map(|g| &g.layers[l]);
        let p = |s: &str| format!("layers.{l}.{s}");
        match &layer.mixer {
            Mixer::Gdn(w) => {
                let gg = match lg.map(|x| &x.mixer) {
                    None => None,
                    Some(MixerGrads::Gdn(gg)) => Some(gg),
                    Some(MixerGrads::Attn(_)) => return Err(format!("layer {l}: attention gradients for a GDN layer")),
                };
                let gd = cfg.gdn;
                let widths = gd.part_widths();
                let offs = offsets(widths);
                for (i, part) in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]
                    .iter()
                    .enumerate()
                {
                    out.push(Slot {
                        info: info(p(&format!("linear_attn.{part}.weight")), &[widths[i], h], true),
                        param: Src::Packed(&w.w_in, offs[i]),
                        grad: gg.map(|g| Src::Packed(&g.w_in, offs[i])),
                    });
                }
                let (vh, kw) = (gd.v_heads() as usize, cfg.conv_kernel as usize);
                out.push(Slot {
                    info: info(p("linear_attn.out_proj.weight"), &[h, gd.value_dim() as usize], true),
                    param: Src::Packed(&w.w_out, 0),
                    grad: gg.map(|g| Src::Packed(&g.w_out, 0)),
                });
                out.push(Slot {
                    info: info(p("linear_attn.conv1d.weight"), &[gd.conv_dim() as usize, 1, kw], false),
                    param: Src::Raw(&w.conv_w),
                    grad: gg.map(|g| Src::Raw(&g.conv_w)),
                });
                out.push(Slot {
                    info: info(p("linear_attn.A_log"), &[vh], false),
                    param: Src::Raw(&w.a_log),
                    grad: gg.map(|g| Src::Raw(&g.a_log)),
                });
                out.push(Slot {
                    info: info(p("linear_attn.dt_bias"), &[vh], false),
                    param: Src::Raw(&w.dt_bias),
                    grad: gg.map(|g| Src::Raw(&g.dt_bias)),
                });
                out.push(Slot {
                    info: info(p("linear_attn.norm.weight"), &[gd.v_dim() as usize], false),
                    param: Src::Raw(&w.norm_w),
                    grad: gg.map(|g| Src::Raw(&g.norm_w)),
                });
            }
            Mixer::Attn(w) => {
                let ag = match lg.map(|x| &x.mixer) {
                    None => None,
                    Some(MixerGrads::Attn(ag)) => Some(ag),
                    Some(MixerGrads::Gdn(_)) => return Err(format!("layer {l}: GDN gradients for an attention layer")),
                };
                let at = cfg.attn;
                let widths = at.part_widths();
                let offs = offsets(widths);
                for (i, part) in ["q_proj", "k_proj", "v_proj"].iter().enumerate() {
                    out.push(Slot {
                        info: info(p(&format!("self_attn.{part}.weight")), &[widths[i], h], true),
                        param: Src::Packed(&w.w_in, offs[i]),
                        grad: ag.map(|g| Src::Packed(&g.w_in, offs[i])),
                    });
                }
                let d = at.head_dim() as usize;
                out.push(Slot {
                    info: info(p("self_attn.o_proj.weight"), &[h, at.q_heads() as usize * d], true),
                    param: Src::Packed(&w.w_out, 0),
                    grad: ag.map(|g| Src::Packed(&g.w_out, 0)),
                });
                out.push(Slot {
                    info: info(p("self_attn.q_norm.weight"), &[d], false),
                    param: Src::Raw(&w.q_norm),
                    grad: ag.map(|g| Src::Raw(&g.q_norm)),
                });
                out.push(Slot {
                    info: info(p("self_attn.k_norm.weight"), &[d], false),
                    param: Src::Raw(&w.k_norm),
                    grad: ag.map(|g| Src::Raw(&g.k_norm)),
                });
            }
        }
        out.push(Slot {
            info: info(p("mlp.gate_proj.weight"), &[inter, h], true),
            param: Src::Packed(&layer.gate, 0),
            grad: lg.map(|g| Src::Packed(&g.gate, 0)),
        });
        out.push(Slot {
            info: info(p("mlp.up_proj.weight"), &[inter, h], true),
            param: Src::Packed(&layer.up, 0),
            grad: lg.map(|g| Src::Packed(&g.up, 0)),
        });
        out.push(Slot {
            info: info(p("mlp.down_proj.weight"), &[h, inter], true),
            param: Src::Packed(&layer.down, 0),
            grad: lg.map(|g| Src::Packed(&g.down, 0)),
        });
        out.push(Slot {
            info: info(p("input_layernorm.weight"), &[h], false),
            param: Src::Raw(&layer.input_norm),
            grad: lg.map(|g| Src::Raw(&g.input_norm)),
        });
        out.push(Slot {
            info: info(p("post_attention_layernorm.weight"), &[h], false),
            param: Src::Raw(&layer.post_norm),
            grad: lg.map(|g| Src::Raw(&g.post_norm)),
        });
    }
    out.push(Slot {
        info: info("norm.weight".into(), &[h], false),
        param: Src::Raw(&m.final_norm),
        grad: grads.map(|g| Src::Raw(&g.final_norm)),
    });
    // `Qwen35Grads`' fields are public, and an `AdamW` made for another model
    // carries that model's runtime: refuse either before anything reads or
    // writes through them.
    for s in &out {
        if let Some(g) = s.grad {
            require_runtime(&m.rt, g.buffer(), format_args!("{}'s gradient", s.info.name))?;
        }
    }
    Ok(out)
}

/// Check the caller's tensors against the table before touching anything.
pub(crate) fn check(what: &str, rt: &Arc<GpuRuntime>, slots: &[Slot<'_>], ts: &[Tensor]) -> Result<(), String> {
    if ts.len() != slots.len() {
        return Err(format!("{what}: {} tensors for {} parameters", ts.len(), slots.len()));
    }
    for (s, t) in slots.iter().zip(ts) {
        if !Arc::ptr_eq(t.runtime(), rt) {
            return Err(format!("{what}: {} belongs to a different runtime", s.info.name));
        }
        require_runtime(rt, &t.buffer, format_args!("{what}: {}", s.info.name))?;
        let want = s.info.storage_shape();
        if t.dtype != DType::F32 || t.shape() != want.as_slice() {
            return Err(format!(
                "{what}: {} must be f32 {want:?}, got {:?} {:?}",
                s.info.name,
                t.dtype,
                t.shape()
            ));
        }
        t.validate().map_err(|e| format!("{what}: {}: {e}", s.info.name))?;
        if t.byte_offset() % 4 != 0 {
            return Err(format!(
                "{what}: {}: byte offset {} is not a multiple of 4",
                s.info.name,
                t.byte_offset()
            ));
        }
    }
    Ok(())
}

impl Qwen35Model {
    /// The model can be trained and written: f32, or bf16 without the
    /// inference head's packed copy of the embedding, which a write or an
    /// optimizer step would leave behind ([`Self::load_tower`] loads without
    /// it).
    pub(crate) fn require_trainable(&self, what: &str) -> Result<(), String> {
        if self.precision == Precision::Bf16 && self.lm_head_bf16.is_some() {
            return Err(format!(
                "{what}: a bf16 model loaded with its packed LM head cannot change its weights (the head \
                 copy would go stale); load it with Qwen35Model::load_tower to train"
            ));
        }
        Ok(())
    }

    /// Every parameter, in the order the copies take their tensors.
    pub fn parameter_table(&self) -> Result<Vec<ParamInfo>, String> {
        Ok(slots(self, None)?.into_iter().map(|s| s.info).collect())
    }

    /// Copy every parameter's value into `dst` (one dense f32 tensor per
    /// [`Self::parameter_table`] entry, of its [`ParamInfo::storage_shape`]);
    /// a bf16 matrix is widened exactly.
    pub fn read_parameters(&self, dst: &[Tensor]) -> Result<(), String> {
        const WHAT: &str = "Qwen35Model::read_parameters";
        let slots = slots(self, None)?;
        check(WHAT, &self.rt, &slots, dst)?;
        for (s, t) in slots.iter().zip(dst) {
            self.read_one(s.param_window(), t)
                .map_err(|e| format!("{WHAT}: {}: {e}", s.info.name))?;
        }
        self.rt.synchronize()
    }

    /// Copy `grads` (from [`Self::train_step`] on this model, or a bank) into
    /// `dst`, laid out as [`Self::read_parameters`] lays out the values.
    pub fn read_gradients(&self, grads: &Qwen35Grads, dst: &[Tensor]) -> Result<(), String> {
        const WHAT: &str = "Qwen35Model::read_gradients";
        let slots = slots(self, Some(grads)).map_err(|e| format!("{WHAT}: {e}"))?;
        check(WHAT, &self.rt, &slots, dst)?;
        for (s, t) in slots.iter().zip(dst) {
            let g = s
                .grad_window()
                .ok_or_else(|| format!("{WHAT}: {} has no gradient", s.info.name))?;
            self.read_one(g, t)
                .map_err(|e| format!("{WHAT}: {}: {e}", s.info.name))?;
        }
        self.rt.synchronize()
    }

    /// Set every parameter from `src`, laid out as [`Self::read_parameters`]
    /// writes them; a bf16 matrix takes each value rounded to nearest. Every
    /// tensor is checked before anything is written. A
    /// [`crate::qwen35_train::PendingStep`] from before the write can no
    /// longer be backpropagated. An optimizer's f32 master
    /// ([`crate::qwen35_adamw::UpdateRule::F32Master`]) is its own state and
    /// is not moved by this; restore it with [`Self::write_adamw_aux`].
    pub fn write_parameters(&self, src: &[Tensor]) -> Result<(), String> {
        const WHAT: &str = "Qwen35Model::write_parameters";
        self.require_trainable(WHAT)?;
        let slots = slots(self, None)?;
        check(WHAT, &self.rt, &slots, src)?;
        for s in &slots {
            s.param_window().check(&format!("{WHAT}: {}", s.info.name))?;
        }
        self.bump_param_generation();
        for (s, t) in slots.iter().zip(src) {
            let to = s.param_window();
            window_copy(&self.rt, &Window::tensor_as(t, &to)?, &to)
                .map_err(|e| format!("{WHAT}: {}: {e}", s.info.name))?;
        }
        self.rt.synchronize()
    }

    fn read_one(&self, from: Window<'_>, dst: &Tensor) -> Result<(), String> {
        window_copy(&self.rt, &from, &Window::tensor_as(dst, &from)?)
    }
}
