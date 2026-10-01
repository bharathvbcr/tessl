//! AdamW over a Qwen3.5 model's own parameters, on the GPU, in place.
//!
//! A training loop that runs [`Qwen35Model::train_step`] and this keeps one
//! copy of each parameter and gradient: torch's optimizer would need its own
//! copy of both (16 GB more on the 2B, which does not fit beside tessl's on a
//! 64 GB Mac). [`AdamW`] holds the two moments as mirrors of the parameters'
//! own tensors (packed projections, `[vocab, hidden]` embedding, the norms'
//! buffers), so one window addresses a parameter, its gradient and both
//! moments.
//!
//! The update is `torch.optim.AdamW`'s single-tensor path with `amsgrad` and
//! `maximize` off: the step count increments first, the bias corrections and
//! step size are formed in f64 (`lr / (1 - beta1^t)`, `(1 - beta2^t)^0.5`) and
//! passed as f32, and the kernel (`kernels/qwen35_adamw.metal`) applies
//! decoupled weight decay, then the moments (the first through torch's
//! `lerp`), then `p += -step_size * m / (sqrt(v) / sqrt(bc2) + eps)`. Weight
//! decay is per parameter-table entry, so parameter groups map onto it. Every
//! parameter is updated as stored, which is transformers' value (the
//! zero-centred norms' `w` included).

use crate::dispatch::{dispatch_2d, set_gpu_buf_offset, set_u32};
use crate::qwen35_model::{Mixer, Qwen35Model};
use crate::qwen35_params::{slots, Src};
use crate::qwen35_train::{AttnGrads, GdnGrads, LayerGrads, MixerGrads, Qwen35Grads};
use crate::tensor::{GpuBuffer, Tensor};

/// AdamW's hyperparameters other than weight decay, as torch names them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWHyper {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl Default for AdamWHyper {
    /// torch's defaults, with transformers' fine-tuning learning rate.
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

/// The optimizer's state: both moments, zero until the first step, and the
/// step count.
pub struct AdamW {
    m: Qwen35Grads,
    v: Qwen35Grads,
    step: u64,
}

fn zeros_tensor(t: &Tensor) -> Result<Tensor, String> {
    // Hot: optimizer state stays resident. Allocations come back zeroed.
    t.runtime().alloc_tensor_f32_hot(t.shape())
}

fn zeros_buf(model: &Qwen35Model, b: &GpuBuffer) -> Result<GpuBuffer, String> {
    let out = model.rt.alloc_buffer_hot(b.nbytes())?;
    out.zero();
    Ok(out)
}

/// Zeroed tensors shaped like every parameter tensor of `model`.
fn zeros_like(model: &Qwen35Model) -> Result<Qwen35Grads, String> {
    let mut layers = Vec::with_capacity(model.layers.len());
    for layer in &model.layers {
        let mixer = match &layer.mixer {
            Mixer::Gdn(w) => MixerGrads::Gdn(GdnGrads {
                w_in: zeros_tensor(&w.w_in)?,
                w_out: zeros_tensor(&w.w_out)?,
                conv_w: zeros_buf(model, &w.conv_w)?,
                a_log: zeros_buf(model, &w.a_log)?,
                dt_bias: zeros_buf(model, &w.dt_bias)?,
                norm_w: zeros_buf(model, &w.norm_w)?,
            }),
            Mixer::Attn(w) => MixerGrads::Attn(AttnGrads {
                w_in: zeros_tensor(&w.w_in)?,
                w_out: zeros_tensor(&w.w_out)?,
                q_norm: zeros_buf(model, &w.q_norm)?,
                k_norm: zeros_buf(model, &w.k_norm)?,
            }),
        };
        layers.push(LayerGrads {
            input_norm: zeros_buf(model, &layer.input_norm)?,
            post_norm: zeros_buf(model, &layer.post_norm)?,
            mixer,
            gate: zeros_tensor(&layer.gate)?,
            up: zeros_tensor(&layer.up)?,
            down: zeros_tensor(&layer.down)?,
        });
    }
    Ok(Qwen35Grads {
        embed: zeros_tensor(&model.embed)?,
        final_norm: zeros_buf(model, &model.final_norm)?,
        layers,
    })
}

impl AdamW {
    /// Zeroed moments for `model` (twice its parameters' memory).
    pub fn new(model: &Qwen35Model) -> Result<Self, String> {
        model.require_f32("AdamW::new")?;
        Ok(Self {
            m: zeros_like(model)?,
            v: zeros_like(model)?,
            step: 0,
        })
    }

    /// Steps taken so far (torch's `state["step"]`).
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Set the step count, restoring a checkpoint with
    /// [`Qwen35Model::write_adamw_moment`].
    pub fn set_step_count(&mut self, step: u64) {
        self.step = step;
    }

    fn moment(&self, which: Moment) -> &Qwen35Grads {
        match which {
            Moment::First => &self.m,
            Moment::Second => &self.v,
        }
    }
}

/// One of AdamW's two moments: torch's `exp_avg` and `exp_avg_sq`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Moment {
    First,
    Second,
}

/// transformers' `Trainer.get_decay_parameter_names` exclusions, by name: a
/// parameter whose name matches `bias`, `layernorm`, `rmsnorm`,
/// `(^|\.)norm($|\.)` or `_norm($|\.)` takes no weight decay. On Qwen3.5 that
/// is every norm and `linear_attn.dt_bias`.
pub fn excluded_from_weight_decay(name: &str) -> bool {
    if name.contains("bias") || name.contains("layernorm") || name.contains("rmsnorm") {
        return true;
    }
    name.split('.').any(|seg| seg == "norm" || seg.ends_with("_norm"))
}

/// One window: a buffer, the byte offset of its tensor, and `rows` x `width`
/// at leading dimension `ld` from element `off`.
struct Window<'a> {
    buf: &'a GpuBuffer,
    byte_off: usize,
    rows: usize,
    width: usize,
    ld: usize,
    off: usize,
}

fn window<'a>(src: Src<'a>, shape: &[usize]) -> Window<'a> {
    let numel = shape.iter().product::<usize>();
    match src {
        Src::Raw(b) => Window {
            buf: b,
            byte_off: 0,
            rows: 1,
            width: numel,
            ld: numel,
            off: 0,
        },
        Src::Dense(t) => {
            let s = t.shape();
            Window {
                buf: &t.buffer,
                byte_off: t.byte_offset(),
                rows: s[0],
                width: s[1],
                ld: s[1],
                off: 0,
            }
        }
        // transformers' [out, in]: `in` rows of `out` columns in the packed [in, total].
        Src::Packed(t, off) => Window {
            buf: &t.buffer,
            byte_off: t.byte_offset(),
            rows: shape[1],
            width: shape[0],
            ld: t.shape()[1],
            off,
        },
    }
}

impl Window<'_> {
    /// The window lies inside its buffer.
    fn check(&self, what: &str) -> Result<(), String> {
        let last = (self.rows - 1)
            .checked_mul(self.ld)
            .and_then(|x| x.checked_add(self.off + self.width))
            .ok_or_else(|| format!("{what}: window overflows"))?;
        let have = self.buf.nbytes().saturating_sub(self.byte_off) / 4;
        if self.byte_off % 4 != 0 || last > have || self.width > self.ld {
            return Err(format!(
                "{what}: window of {last} elements does not fit its buffer's {have}"
            ));
        }
        for n in [self.rows, self.width, self.ld, self.off] {
            u32::try_from(n).map_err(|_| format!("{what}: {n} exceeds u32"))?;
        }
        Ok(())
    }
}

impl Qwen35Model {
    /// One AdamW step on every parameter from `grads` (this model's
    /// [`Qwen35Model::train_step`]), with `weight_decay[i]` for
    /// [`Qwen35Model::parameter_table`] entry `i`. Everything is checked before
    /// anything moves; the step count advances only when the step runs.
    pub fn adamw_step(
        &self,
        grads: &Qwen35Grads,
        state: &mut AdamW,
        hyper: &AdamWHyper,
        weight_decay: &[f32],
    ) -> Result<(), String> {
        const WHAT: &str = "Qwen35Model::adamw_step";
        self.require_f32(WHAT)?;
        let AdamWHyper { lr, beta1, beta2, eps } = *hyper;
        if !(lr.is_finite() && lr >= 0.0) {
            return Err(format!("{WHAT}: lr {lr} must be finite and >= 0"));
        }
        for (name, b) in [("beta1", beta1), ("beta2", beta2)] {
            if !(0.0..1.0).contains(&b) {
                return Err(format!("{WHAT}: {name} {b} must lie in [0, 1)"));
            }
        }
        if !(eps.is_finite() && eps > 0.0) {
            return Err(format!("{WHAT}: eps {eps} must be finite and > 0"));
        }
        let ps = slots(self, Some(grads)).map_err(|e| format!("{WHAT}: {e}"))?;
        let ms = slots(self, Some(&state.m)).map_err(|e| format!("{WHAT}: moments: {e}"))?;
        let vs = slots(self, Some(&state.v)).map_err(|e| format!("{WHAT}: moments: {e}"))?;
        if weight_decay.len() != ps.len() {
            return Err(format!(
                "{WHAT}: {} weight decays for {} parameters",
                weight_decay.len(),
                ps.len()
            ));
        }
        let mut plan = Vec::with_capacity(ps.len());
        for (((s, m), v), &wd) in ps.iter().zip(&ms).zip(&vs).zip(weight_decay) {
            let name = &s.info.name;
            if !(wd.is_finite() && wd >= 0.0) {
                return Err(format!("{WHAT}: {name}: weight decay {wd} must be finite and >= 0"));
            }
            let (g, mm, vv) = match (s.grad, m.grad, v.grad) {
                (Some(g), Some(mm), Some(vv)) => (g, mm, vv),
                _ => return Err(format!("{WHAT}: {name} has no gradient")),
            };
            let shape = &s.info.shape;
            let w = [
                window(s.param, shape),
                window(g, shape),
                window(mm, shape),
                window(vv, shape),
            ];
            for (x, part) in w.iter().zip(["parameter", "gradient", "first moment", "second moment"]) {
                x.check(&format!("{WHAT}: {name} {part}"))?;
            }
            if w[1..]
                .iter()
                .any(|x| (x.rows, x.width, x.ld, x.off) != (w[0].rows, w[0].width, w[0].ld, w[0].off))
            {
                return Err(format!(
                    "{WHAT}: {name}: gradient or moments are not laid out as the parameter"
                ));
            }
            plan.push((w, f64::from(wd)));
        }

        // torch: the step count increments, then the scalars are formed in f64.
        let t = (state.step + 1) as f64;
        let bc1 = 1.0 - beta1.powf(t);
        let bc2 = 1.0 - beta2.powf(t);
        let step_size = lr / bc1;
        let bc2_sqrt = bc2.powf(0.5);
        let p = self.rt.pipeline("qwen35_adamw_f32")?;
        for (w, wd) in &plan {
            let scalars = [
                (1.0 - lr * wd) as f32,
                (1.0 - beta1) as f32,
                beta2 as f32,
                (1.0 - beta2) as f32,
                step_size as f32,
                bc2_sqrt as f32,
                eps as f32,
            ];
            let bytes: Vec<u8> = scalars.iter().flat_map(|x| x.to_le_bytes()).collect();
            let [wp, wg, wm, wv] = w;
            dispatch_2d(&self.rt, &p, wp.width, wp.rows, |bnd| {
                set_gpu_buf_offset(bnd, wp.buf, wp.byte_off, 0);
                set_gpu_buf_offset(bnd, wg.buf, wg.byte_off, 1);
                set_gpu_buf_offset(bnd, wm.buf, wm.byte_off, 2);
                set_gpu_buf_offset(bnd, wv.buf, wv.byte_off, 3);
                bnd.bind_bytes(&bytes, 4);
                set_u32(bnd, wp.rows as u32, 5);
                set_u32(bnd, wp.width as u32, 6);
                set_u32(bnd, wp.ld as u32, 7);
                set_u32(bnd, wp.off as u32, 8);
            })?;
        }
        self.rt.synchronize()?;
        state.step += 1;
        Ok(())
    }

    /// Copy one of `state`'s moments into `dst`, one dense f32 tensor per
    /// [`Qwen35Model::parameter_table`] entry laid out as
    /// [`Qwen35Model::read_parameters`] lays out the values.
    pub fn read_adamw_moment(&self, state: &AdamW, which: Moment, dst: &[Tensor]) -> Result<(), String> {
        self.read_gradients(state.moment(which), dst)
            .map_err(|e| format!("Qwen35Model::read_adamw_moment: {e}"))
    }

    /// Set one of `state`'s moments from `src`, laid out as
    /// [`Self::read_adamw_moment`] reads it; with
    /// [`AdamW::set_step_count`] this restores a checkpoint. Every tensor is
    /// checked before anything is written.
    pub fn write_adamw_moment(&self, state: &mut AdamW, which: Moment, src: &[Tensor]) -> Result<(), String> {
        self.write_gradient_layout("Qwen35Model::write_adamw_moment", state.moment(which), src)
    }

    /// Weight decay `wd` for every parameter-table entry except those
    /// [`excluded_from_weight_decay`], which take 0.
    pub fn default_weight_decay(&self, wd: f32) -> Result<Vec<f32>, String> {
        Ok(self
            .parameter_table()?
            .iter()
            .map(|p| if excluded_from_weight_decay(&p.name) { 0.0 } else { wd })
            .collect())
    }
}
