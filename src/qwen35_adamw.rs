//! One AdamW step on device-resident f32 tensors, in place: the launcher of
//! `kernels/qwen35_adamw.metal`'s plain f32 kernel (`qwen35_adamw_f32`).
//!
//! This is the only part of tessl's old Qwen3.5 AdamW that remains. The
//! optimizer object (its state per parameter-table entry), its
//! stored-precision update rules and moment storage, the fused step and the
//! checkpoint reads of the moments retired with the Qwen3.5
//! training step (ABI 14): ojas owns Metal training through the Tape and
//! `Backend::adamw_step`, and tessl keeps the kernels and this launcher
//! (`ojas-metal`'s legacy tiny step calls it).
//!
//! The update is `torch.optim.AdamW`'s single-tensor path with `amsgrad` and
//! `maximize` off: the step count increments first, the bias corrections and
//! step size are formed in f64 (`lr / (1 - beta1^t)`, `(1 - beta2^t)^0.5`) and
//! passed as f32, and the kernel (`kernels/qwen35_adamw_math.h`) applies
//! decoupled weight decay, then the moments (the first through torch's
//! `lerp`), then `p += -step_size * m / (sqrt(v) / sqrt(bc2) + eps)`.

use std::sync::Arc;

use crate::dispatch::{dispatch_2d, set_gpu_buf_offset, set_u32};
use crate::qwen35_params::Window;
use crate::runtime::GpuRuntime;
use crate::tensor::{DType, Tensor};

/// AdamW's hyperparameters other than weight decay, as torch names them,
/// and the step's gradient scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWHyper {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    /// Every gradient is multiplied by this (as f32) before the update:
    /// `torch.nn.utils.clip_grad_norm_`'s clip coefficient (see
    /// [`crate::capi::tessl_clip_coef`]), or 1. The gradients themselves are
    /// left as they are.
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

/// Host-side scalars, in the order `kernels/qwen35_adamw_math.h` reads them.
///
/// `step` is torch's count after it has incremented (1-based).
/// Delegates canonical scalar formation to `ojas_core::AdamWScalars`.
fn adamw_scalars(hyper: &AdamWHyper, step: u64, weight_decay: f64) -> [f32; 8] {
    let config = ojas_core::AdamWConfig {
        lr: hyper.lr,
        beta1: hyper.beta1,
        beta2: hyper.beta2,
        eps: hyper.eps,
        weight_decay,
    };
    let step_before = step.saturating_sub(1);
    let s = ojas_core::AdamWScalars::new(config, step_before).expect("validated hyperparameters");
    let w = s.f32_words("adamw_step").expect("finite f32 words");
    [w[0], w[1], w[2], w[3], w[4], w[5], w[6], hyper.grad_scale as f32]
}

fn le_bytes<const N: usize, T: Copy>(xs: [T; N], f: impl Fn(T) -> [u8; 4]) -> Vec<u8> {
    xs.iter().flat_map(|&x| f(x)).collect()
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
/// `u64::MAX` before calling.
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
    let n = param.try_numel()?;
    if n == 0 {
        return Ok(());
    }
    let w: Vec<Window<'_>> = views
        .iter()
        .map(|t| Window::dense(&t.buffer, t.byte_offset(), DType::F32, n))
        .collect();
    for (x, name) in w.iter().zip(names) {
        x.check(format_args!("{WHAT}: {name}"))?;
    }
    let bytes = le_bytes(adamw_scalars(hyper, step, f64::from(weight_decay)), f32::to_le_bytes);
    let p = rt.pipeline("qwen35_adamw_f32")?;
    dispatch_2d(rt, &p, n, 1, |bnd| {
        for (i, x) in w.iter().enumerate() {
            set_gpu_buf_offset(bnd, x.buf, x.byte_off, i);
        }
        bnd.bind_bytes(&bytes, 4);
        set_u32(bnd, 1, 5);
        set_u32(bnd, n as u32, 6);
        set_u32(bnd, n as u32, 7);
        set_u32(bnd, 0, 8);
    })
}
