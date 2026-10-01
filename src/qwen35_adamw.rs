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

use crate::dispatch::{dispatch_2d, dispatch_2d_tg, set_gpu_buf_offset, set_u32};
use crate::qwen35_model::Qwen35Model;
use crate::qwen35_params::{slots, Src};
use crate::qwen35_train::Qwen35Grads;
use crate::runtime::GpuRuntime;
use crate::tensor::{DType, GpuBuffer, Tensor};
use std::sync::Arc;

/// AdamW's hyperparameters other than weight decay, as torch names them,
/// and the step's gradient scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWHyper {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    /// Every gradient is multiplied by this (as f32) before the update:
    /// `torch.nn.utils.clip_grad_norm_`'s clip coefficient, formed by the
    /// caller from [`Qwen35Model::grad_sq_norm`] (and any gradients outside
    /// the model), or 1. The gradients themselves are left as they are.
    pub grad_scale: f64,
}

impl Default for AdamWHyper {
    /// torch's defaults, with transformers' fine-tuning learning rate, and
    /// no clipping.
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            grad_scale: 1.0,
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

impl AdamW {
    /// Zeroed moments for `model` (twice its parameters' memory).
    pub fn new(model: &Qwen35Model) -> Result<Self, String> {
        model.require_f32("AdamW::new")?;
        Ok(Self {
            m: Qwen35Grads::zeros_like(model)?,
            v: Qwen35Grads::zeros_like(model)?,
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

fn dense_window<'a>(t: &'a Tensor) -> Result<Window<'a>, String> {
    let n = t.try_numel()?;
    let w = Window {
        buf: &t.buffer,
        byte_off: t.byte_offset(),
        rows: 1,
        width: n,
        ld: n,
        off: 0,
    };
    w.check("adamw_step")?;
    Ok(w)
}

fn check_hyper(what: &str, hyper: &AdamWHyper) -> Result<(), String> {
    let AdamWHyper {
        lr,
        beta1,
        beta2,
        eps,
        grad_scale,
    } = *hyper;
    if !(lr.is_finite() && lr >= 0.0) {
        return Err(format!("{what}: lr {lr} must be finite and >= 0"));
    }
    // torch would carry a non-finite norm's coefficient into every
    // parameter; refused here instead, before anything moves.
    if !(grad_scale.is_finite() && grad_scale >= 0.0) {
        return Err(format!("{what}: grad_scale {grad_scale} must be finite and >= 0"));
    }
    for (name, b) in [("beta1", beta1), ("beta2", beta2)] {
        if !(0.0..1.0).contains(&b) {
            return Err(format!("{what}: {name} {b} must lie in [0, 1)"));
        }
    }
    if !(eps.is_finite() && eps > 0.0) {
        return Err(format!("{what}: eps {eps} must be finite and > 0"));
    }
    Ok(())
}

/// Host-side scalars, in the order `kernels/qwen35_adamw.metal` reads them.
///
/// `step` is torch's count after it has incremented (1-based). The formation
/// is f64, then each value is stored as f32. `weight_decay` is one tensor's.
fn adamw_scalars(hyper: &AdamWHyper, step: u64, weight_decay: f64) -> [f32; 8] {
    let AdamWHyper {
        lr,
        beta1,
        beta2,
        eps,
        grad_scale,
    } = *hyper;
    let t = step as f64;
    let bc1 = 1.0 - beta1.powf(t);
    let bc2 = 1.0 - beta2.powf(t);
    let step_size = lr / bc1;
    let bc2_sqrt = bc2.powf(0.5);
    [
        (1.0 - lr * weight_decay) as f32,
        (1.0 - beta1) as f32,
        beta2 as f32,
        (1.0 - beta2) as f32,
        step_size as f32,
        bc2_sqrt as f32,
        eps as f32,
        grad_scale as f32,
    ]
}

fn dispatch_adamw(rt: &GpuRuntime, w: &[Window<'_>; 4], scalars: &[f32; 8]) -> Result<(), String> {
    let bytes: Vec<u8> = scalars.iter().flat_map(|x| x.to_le_bytes()).collect();
    let [wp, wg, wm, wv] = w;
    let p = rt.pipeline("qwen35_adamw_f32")?;
    dispatch_2d(rt, &p, wp.width, wp.rows, |bnd| {
        set_gpu_buf_offset(bnd, wp.buf, wp.byte_off, 0);
        set_gpu_buf_offset(bnd, wg.buf, wg.byte_off, 1);
        set_gpu_buf_offset(bnd, wm.buf, wm.byte_off, 2);
        set_gpu_buf_offset(bnd, wv.buf, wv.byte_off, 3);
        bnd.bind_bytes(&bytes, 4);
        set_u32(bnd, wp.rows as u32, 5);
        set_u32(bnd, wp.width as u32, 6);
        set_u32(bnd, wp.ld as u32, 7);
        set_u32(bnd, wp.off as u32, 8);
    })
}

fn on_runtime(rt: &GpuRuntime, t: &Tensor) -> bool {
    std::ptr::eq(Arc::as_ptr(t.runtime()), rt)
}

/// One AdamW step on device-resident f32 tensors, in place.
///
/// `param`, `grad`, `m` and `v` must share `rt`, the same shape and
/// [`DType::F32`]. Storage may be a view: [`Tensor::byte_offset`] is the
/// start of the window, and the elements are contiguous in that shape. `m`
/// and `v` are torch's `exp_avg` and `exp_avg_sq`; they start at zero on the
/// first step. `step` is the 1-based count torch stores after incrementing.
/// Step 0 is refused: it is what `u64::MAX + 1` wraps to, and it zeroes the
/// bias correction. The caller increments with `checked_add` and refuses
/// `u64::MAX` before calling, which is what [`Qwen35Model::adamw_step`] does
/// for its own counter.
///
/// The update is encoded and not waited on. Call [`GpuRuntime::synchronize`]
/// before reading the tensors on the host. A later kernel on `rt` is ordered
/// after this one by the runtime's encoder barrier.
///
/// # Passing a gradient that already lives on the device
///
/// `grad` is not copied. A buffer tessl allocated (`cross_entropy`'s `dW`,
/// `rt.alloc_tensor_f32`, or [`Tensor::try_view`] into one of those) is passed
/// as `&grad`. A foreign `MTLBuffer` on the same device is wrapped once with
/// [`Tensor::from_mtl_buffer`] and then passed the same way. `param`, `m` and
/// `v` are written through the pointers they already hold.
#[allow(clippy::too_many_arguments)]
pub fn adamw_step(
    rt: &GpuRuntime,
    param: &mut Tensor,
    grad: &Tensor,
    m: &mut Tensor,
    v: &mut Tensor,
    hyper: &AdamWHyper,
    step: u64,
    weight_decay: f32,
) -> Result<(), String> {
    const WHAT: &str = "adamw_step";
    check_hyper(WHAT, hyper)?;
    if step == 0 {
        return Err(format!(
            "{WHAT}: step 0 is refused; a count of u64::MAX cannot advance, and 0 zeroes the bias correction"
        ));
    }
    if !(weight_decay.is_finite() && weight_decay >= 0.0) {
        return Err(format!("{WHAT}: weight decay {weight_decay} must be finite and >= 0"));
    }
    let tensors = [
        ("param", param as &Tensor),
        ("grad", grad),
        ("m", m as &Tensor),
        ("v", v as &Tensor),
    ];
    for (name, t) in tensors {
        t.validate().map_err(|e| format!("{WHAT}: {name}: {e}"))?;
        if t.dtype != DType::F32 {
            return Err(format!("{WHAT}: {name} is {:?}, not f32", t.dtype));
        }
        if !on_runtime(rt, t) {
            return Err(format!("{WHAT}: {name} belongs to a different runtime"));
        }
    }
    if param.shape() != grad.shape() || param.shape() != m.shape() || param.shape() != v.shape() {
        return Err(format!(
            "{WHAT}: shapes differ: param {:?}, grad {:?}, m {:?}, v {:?}",
            param.shape(),
            grad.shape(),
            m.shape(),
            v.shape()
        ));
    }
    let names = ["param", "grad", "m", "v"];
    let views = [param as &Tensor, grad, m as &Tensor, v as &Tensor];
    for i in 0..4 {
        for j in (i + 1)..4 {
            if views[i].overlaps(views[j]) {
                return Err(format!("{WHAT}: {} overlaps {}", names[i], names[j]));
            }
        }
    }
    if param.try_numel()? == 0 {
        return Ok(());
    }
    let windows = [
        dense_window(param)?,
        dense_window(grad)?,
        dense_window(m)?,
        dense_window(v)?,
    ];
    dispatch_adamw(rt, &windows, &adamw_scalars(hyper, step, f64::from(weight_decay)))
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
        check_hyper(WHAT, hyper)?;
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
        // `u64` wraps only at `u64::MAX`. A release build without overflow
        // checks used to turn that into t = 0, which zeroes the bias correction
        // and sends the parameter update to infinity.
        let next = state
            .step
            .checked_add(1)
            .ok_or_else(|| format!("{WHAT}: step count {} + 1 does not fit in u64", state.step))?;
        // Packed projections are a strided window (`ld`, `off`), not a
        // contiguous `Tensor`, so this loop calls the same encoder as
        // [`adamw_step`] rather than copying each parameter out and back.
        for (w, wd) in &plan {
            dispatch_adamw(&self.rt, w, &adamw_scalars(hyper, next, *wd))?;
        }
        self.rt.synchronize()?;
        state.step = next;
        Ok(())
    }

    /// The sum of squares of every gradient in `grads` (this model's
    /// [`Qwen35Model::train_step`]), over the parameters of
    /// [`Qwen35Model::parameter_table`]: the square of the global L2 norm
    /// `torch.nn.utils.clip_grad_norm_` takes. Each row of each gradient is
    /// summed in f32 on the GPU in a fixed order and the rows are added in
    /// f64, so it is deterministic; torch's own reduction order differs, so
    /// the two agree to rounding, not bits.
    pub fn grad_sq_norm(&self, grads: &Qwen35Grads) -> Result<f64, String> {
        const WHAT: &str = "Qwen35Model::grad_sq_norm";
        self.require_f32(WHAT)?;
        let ps = slots(self, Some(grads)).map_err(|e| format!("{WHAT}: {e}"))?;
        let mut plan = Vec::with_capacity(ps.len());
        let mut rows = 0usize;
        for s in &ps {
            let name = &s.info.name;
            let g = s.grad.ok_or_else(|| format!("{WHAT}: {name} has no gradient"))?;
            let w = window(g, &s.info.shape);
            w.check(&format!("{WHAT}: {name} gradient"))?;
            let n = w.rows;
            plan.push((w, rows));
            rows += n;
        }
        u32::try_from(rows).map_err(|_| format!("{WHAT}: {rows} rows exceed u32"))?;
        let out = self.rt.alloc_tensor_f32(&[rows])?;
        let p = self.rt.pipeline("qwen35_sq_sum_rows_f32")?;
        for (w, at) in &plan {
            dispatch_2d_tg(&self.rt, &p, w.rows, 1, 256, |bnd| {
                set_gpu_buf_offset(bnd, w.buf, w.byte_off, 0);
                set_gpu_buf_offset(bnd, &out.buffer, out.byte_offset(), 1);
                set_u32(bnd, w.width as u32, 2);
                set_u32(bnd, w.ld as u32, 3);
                set_u32(bnd, w.off as u32, 4);
                set_u32(bnd, *at as u32, 5);
            })?;
        }
        self.rt.synchronize()?;
        Ok(out.read_f32()?.iter().map(|&x| f64::from(x)).sum())
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
