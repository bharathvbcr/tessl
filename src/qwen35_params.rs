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
//! - The tied embedding is one f32 `[vocab, hidden]` table, read by the
//!   gather, the training step's cross-entropy and the inference forward's
//!   head alike, so a write is exact and moves all three.
//!
//! Gradients come in the same layouts. Every copy is a GPU dispatch, since a
//! caller's buffers may be GPU-private.

use crate::qwen35::Cols;
use crate::qwen35_bwd::copy_cols;
use crate::qwen35_model::{Mixer, Precision, Qwen35Model};
use crate::qwen35_train::{MixerGrads, Qwen35Grads};
use crate::tensor::{gpu_copy, DType, GpuBuffer, Tensor};

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
    /// Columns `[off, off + width)` of a packed `[in, total]` f32 matrix.
    Packed(&'a Tensor, usize),
    /// A dense f32 tensor holding exactly the value.
    Dense(&'a Tensor),
}

pub(crate) struct Slot<'a> {
    pub(crate) info: ParamInfo,
    pub(crate) param: Src<'a>,
    pub(crate) grad: Option<Src<'a>>,
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
    Ok(out)
}

fn u32_of(n: usize) -> Result<u32, String> {
    u32::try_from(n).map_err(|_| format!("{n} exceeds u32"))
}

/// Check the caller's tensors against the table before touching anything.
fn check(what: &str, slots: &[Slot<'_>], ts: &[Tensor]) -> Result<(), String> {
    if ts.len() != slots.len() {
        return Err(format!("{what}: {} tensors for {} parameters", ts.len(), slots.len()));
    }
    for (s, t) in slots.iter().zip(ts) {
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
    pub(crate) fn require_f32(&self, what: &str) -> Result<(), String> {
        if self.precision != Precision::F32 {
            return Err(format!(
                "{what}: the parameter table needs a model loaded with Precision::F32"
            ));
        }
        Ok(())
    }

    /// Every parameter, in the order the copies take their tensors.
    pub fn parameter_table(&self) -> Result<Vec<ParamInfo>, String> {
        Ok(slots(self, None)?.into_iter().map(|s| s.info).collect())
    }

    /// Copy every parameter's value into `dst` (one dense f32 tensor per
    /// [`Self::parameter_table`] entry, of its [`ParamInfo::storage_shape`]).
    pub fn read_parameters(&self, dst: &[Tensor]) -> Result<(), String> {
        const WHAT: &str = "Qwen35Model::read_parameters";
        self.require_f32(WHAT)?;
        let slots = slots(self, None)?;
        check(WHAT, &slots, dst)?;
        for (s, t) in slots.iter().zip(dst) {
            self.read_one(s.param, &s.info, t)
                .map_err(|e| format!("{WHAT}: {}: {e}", s.info.name))?;
        }
        self.rt.synchronize()
    }

    /// Copy `grads` (from [`Self::train_step`] on this model) into `dst`, laid
    /// out as [`Self::read_parameters`] lays out the values.
    pub fn read_gradients(&self, grads: &Qwen35Grads, dst: &[Tensor]) -> Result<(), String> {
        const WHAT: &str = "Qwen35Model::read_gradients";
        self.require_f32(WHAT)?;
        let slots = slots(self, Some(grads)).map_err(|e| format!("{WHAT}: {e}"))?;
        check(WHAT, &slots, dst)?;
        for (s, t) in slots.iter().zip(dst) {
            let g = s
                .grad
                .ok_or_else(|| format!("{WHAT}: {} has no gradient", s.info.name))?;
            self.read_one(g, &s.info, t)
                .map_err(|e| format!("{WHAT}: {}: {e}", s.info.name))?;
        }
        self.rt.synchronize()
    }

    /// Set every parameter from `src`, laid out as [`Self::read_parameters`]
    /// writes them. Every tensor is checked before anything is written.
    pub fn write_parameters(&self, src: &[Tensor]) -> Result<(), String> {
        const WHAT: &str = "Qwen35Model::write_parameters";
        self.require_f32(WHAT)?;
        let slots = slots(self, None)?;
        check(WHAT, &slots, src)?;
        for (s, t) in slots.iter().zip(src) {
            self.write_one(s.param, &s.info, t)
                .map_err(|e| format!("{WHAT}: {}: {e}", s.info.name))?;
        }
        self.rt.synchronize()
    }

    /// Columns `[off, off + width)` of the packed `[rows, total]` matrix `t`
    /// as a dense `[rows, width]` window of `dense`, in either direction.
    /// `copy_cols` windows start at a column, so a caller tensor at a byte
    /// offset goes through a dense temporary and [`gpu_copy`], which honours
    /// offsets.
    fn packed_copy(
        &self,
        t: &Tensor,
        off: usize,
        rows: usize,
        width: usize,
        dense: &Tensor,
        read: bool,
    ) -> Result<(), String> {
        let rt = &self.rt;
        let packed = Cols {
            buf: &t.buffer,
            ld: u32_of(t.shape()[1])?,
            off: u32_of(off)?,
        };
        let (r, w) = (u32_of(rows)?, u32_of(width)?);
        if dense.byte_offset() == 0 {
            let d = Cols::dense(&dense.buffer, w);
            return if read {
                copy_cols(rt, packed, d, r, w)
            } else {
                copy_cols(rt, d, packed, r, w)
            };
        }
        let tmp = rt.alloc_tensor_f32(&[rows, width])?;
        if read {
            copy_cols(rt, packed, Cols::dense(&tmp.buffer, w), r, w)?;
            gpu_copy(&tmp, dense)
        } else {
            gpu_copy(dense, &tmp)?;
            copy_cols(rt, Cols::dense(&tmp.buffer, w), packed, r, w)
        }
    }

    /// A tessl f32 buffer as a tensor of the caller's storage shape.
    fn as_tensor(&self, b: &GpuBuffer, info: &ParamInfo) -> Result<Tensor, String> {
        Tensor::from_buffer(&self.rt, b.clone(), &info.storage_shape(), DType::F32, 0)
    }

    fn read_one(&self, from: Src<'_>, info: &ParamInfo, dst: &Tensor) -> Result<(), String> {
        match from {
            Src::Raw(b) => gpu_copy(&self.as_tensor(b, info)?, dst),
            Src::Dense(t) => gpu_copy(t, dst),
            Src::Packed(t, off) => self.packed_copy(t, off, info.shape[1], info.shape[0], dst, true),
        }
    }

    fn write_one(&self, to: Src<'_>, info: &ParamInfo, src: &Tensor) -> Result<(), String> {
        match to {
            Src::Dense(t) => gpu_copy(src, t),
            Src::Raw(b) => gpu_copy(src, &self.as_tensor(b, info)?),
            Src::Packed(t, off) => self.packed_copy(t, off, info.shape[1], info.shape[0], src, false),
        }
    }
}
