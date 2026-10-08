//! AdamW over a Qwen3.5 model's own parameters, on the GPU, in place.
//!
//! A training loop that runs [`Qwen35Model::train_step`] and this keeps one
//! copy of each parameter and gradient: torch's optimizer would need its own
//! copy of both (16 GB more on the 2B, which does not fit beside tessl's on a
//! 64 GB Mac). [`AdamW`] holds its state per [`Qwen35Model::parameter_table`]
//! entry, dense in the order [`Qwen35Model::read_parameters`] writes that
//! entry, so a moment reads out as a plain copy.
//!
//! The update is `torch.optim.AdamW`'s single-tensor path with `amsgrad` and
//! `maximize` off: the step count increments first, the bias corrections and
//! step size are formed in f64 (`lr / (1 - beta1^t)`, `(1 - beta2^t)^0.5`) and
//! passed as f32, and the kernel (`kernels/qwen35_adamw_math.h`) applies
//! decoupled weight decay, then the moments (the first through torch's
//! `lerp`), then `p += -step_size * m / (sqrt(v) / sqrt(bc2) + eps)`. Weight
//! decay is per parameter-table entry, so parameter groups map onto it. Every
//! parameter is updated as stored, which is transformers' value (the
//! zero-centred norms' `w` included).
//!
//! # Stored precision ([`AdamWConfig`])
//!
//! The arithmetic is f32 whatever is stored; what a configuration chooses is
//! what each value is rounded to between steps, and so the memory it holds.
//!
//! - [`UpdateRule`]: how a bf16-stored parameter (a [`Precision::Bf16`]
//!   model's matrices) takes its update. An update far below a bf16 ulp
//!   rounds away to nothing if the weight is simply rounded to nearest, so
//!   the rules are an f32 master copy ([`UpdateRule::F32Master`], 4 bytes per
//!   weight more), a bf16 Kahan compensation ([`UpdateRule::Bf16Kahan`], 2
//!   bytes), or stochastic rounding ([`UpdateRule::Bf16Stochastic`],
//!   nothing). Parameters stored in f32 (an f32 model's, and a bf16 model's
//!   norms, conv, `A_log` and `dt_bias`) are updated in f32 under any rule.
//! - [`MomentStorage`]: both moments in f32 (8 bytes per weight), bf16 (4),
//!   or 8-bit codes in blocks of [`MOMENT_BLOCK`] with one f32 scale each
//!   (about 2.03): `kernels/qwen35_train_storage.metal` documents the codes.
//!
//! [`AdamW::config`] and [`AdamW::describe`] record the choice with the
//! state, and the auxiliary state (master or compensation) reads and writes
//! with [`Qwen35Model::read_adamw_aux`] / [`Qwen35Model::write_adamw_aux`], so
//! a checkpoint restores the run bit for bit.

use std::fmt;
use std::sync::Arc;

use objc2_metal::MTLBuffer;

use crate::dispatch::{dispatch_2d, dispatch_2d_tg, set_gpu_buf, set_gpu_buf_offset, set_u32};
use crate::nn::require_runtime;
use crate::qwen35_model::{Precision, Qwen35Model};
use crate::qwen35_params::{check, slots, window_copy, Window};
use crate::qwen35_train::Qwen35Grads;
use crate::runtime::GpuRuntime;
use crate::tensor::{DType, GpuBuffer, Tensor};

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

/// How a parameter takes its update (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateRule {
    /// An f32 model: every parameter updated in f32, torch's bits.
    F32,
    /// A bf16 model: an f32 master of each bf16 weight is updated, and the
    /// weight is the master rounded to nearest.
    F32Master,
    /// A bf16 model: each bf16 weight keeps a bf16 compensation of what its
    /// rounding dropped, added back into the next update.
    Bf16Kahan,
    /// A bf16 model: each new weight is rounded up or down with probability
    /// equal to the distance, from bits hashed from `seed`, the step count,
    /// the parameter-table index and the element, so a run is reproducible.
    Bf16Stochastic { seed: u64 },
}

/// What both moments are stored as (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MomentStorage {
    F32,
    Bf16,
    /// 8-bit codes, one f32 scale per [`MOMENT_BLOCK`] elements.
    Block8,
}

/// Elements per 8-bit moment block (one threadgroup of the step kernel).
pub const MOMENT_BLOCK: usize = 256;

/// Bytes of one entry of a step kernel's slot table (`Qwen35AdamWSlot` in
/// `kernels/qwen35_train_storage.metal`, which asserts the same size): seven
/// GPU addresses, six `u32`s, the eight scalars, the five-word key and a pad.
const SLOT_BYTES: usize = 136;

/// An [`AdamW`]'s stored precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdamWConfig {
    pub update: UpdateRule,
    pub moments: MomentStorage,
}

impl AdamWConfig {
    /// torch's AdamW in f32: what [`AdamW::new`] makes.
    pub const F32: Self = Self {
        update: UpdateRule::F32,
        moments: MomentStorage::F32,
    };

    fn rule_name(self) -> &'static str {
        match self.update {
            UpdateRule::F32 => "f32",
            UpdateRule::F32Master => "f32-master",
            UpdateRule::Bf16Kahan => "bf16-kahan",
            UpdateRule::Bf16Stochastic { .. } => "bf16-stochastic",
        }
    }

    fn moments_name(self) -> &'static str {
        match self.moments {
            MomentStorage::F32 => "f32",
            MomentStorage::Bf16 => "bf16",
            MomentStorage::Block8 => "block8",
        }
    }
}

impl fmt::Display for AdamWConfig {
    /// `update=bf16-kahan moments=block8/256`, with the seed of a stochastic
    /// rule.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "update={}", self.rule_name())?;
        if let UpdateRule::Bf16Stochastic { seed } = self.update {
            write!(f, "(seed={seed})")?;
        }
        write!(f, " moments={}", self.moments_name())?;
        if self.moments == MomentStorage::Block8 {
            write!(f, "/{MOMENT_BLOCK}")?;
        }
        Ok(())
    }
}

/// One parameter-table entry's state, dense in its read order.
struct SlotState {
    n: usize,
    m: GpuBuffer,
    v: GpuBuffer,
    /// Block8: the first and second moments' per-block scales.
    scales: Option<(GpuBuffer, GpuBuffer)>,
    /// A bf16 parameter's f32 master (F32Master) or bf16 compensation
    /// (Bf16Kahan).
    aux: Option<GpuBuffer>,
}

/// The optimizer's state: both moments, zero until the first step, any
/// auxiliary state, the step count, and the configuration it was made with.
pub struct AdamW {
    config: AdamWConfig,
    /// The precision of the model it was made for.
    precision: Precision,
    slots: Vec<SlotState>,
    /// Steps encoded so far: [`Qwen35Model::adamw_step_unwaited`] advances it
    /// before the GPU has run the step.
    step: u64,
    /// Steps a wait of this state's has seen complete: what
    /// [`Self::step_count`] reports once the runtime is poisoned.
    confirmed: std::cell::Cell<u64>,
    /// The runtime the state lives on.
    rt: Arc<GpuRuntime>,
    /// Bound to the kernel's unused buffer slots.
    dummy: GpuBuffer,
}

/// One parameter-table entry's state buffers, in bytes, for `n` elements
/// stored as `dtype` under `config`: each moment, each moment's block
/// scales (0: none), the auxiliary state (0: none).
fn slot_bytes(n: usize, dtype: DType, config: AdamWConfig) -> (usize, usize, usize) {
    let moment = n * match config.moments {
        MomentStorage::F32 => 4,
        MomentStorage::Bf16 => 2,
        MomentStorage::Block8 => 1,
    };
    let scales = if config.moments == MomentStorage::Block8 {
        n.div_ceil(MOMENT_BLOCK) * 4
    } else {
        0
    };
    let aux = match (dtype, config.update) {
        (DType::BF16, UpdateRule::F32Master) => n * 4,
        (DType::BF16, UpdateRule::Bf16Kahan) => n * 2,
        _ => 0,
    };
    (moment, scales, aux)
}

fn zeroed(rt: &GpuRuntime, nbytes: usize) -> Result<GpuBuffer, String> {
    // Hot: optimizer state stays resident.
    let b = rt.alloc_buffer_hot(nbytes.max(4))?;
    b.try_zero()?;
    Ok(b)
}

impl AdamW {
    /// torch's AdamW in f32 ([`AdamWConfig::F32`]) for an f32 `model`:
    /// zeroed moments, twice its parameters' memory. A bf16 model chooses
    /// its stored precision with [`Self::with_config`].
    pub fn new(model: &Qwen35Model) -> Result<Self, String> {
        if model.precision() != Precision::F32 {
            return Err(
                "AdamW::new: the model is stored in bf16; choose an update rule and moment storage with \
                 AdamW::with_config"
                    .into(),
            );
        }
        Self::with_config(model, AdamWConfig::F32)
    }

    /// Zeroed state for `model` in `config`'s stored precision. An f32
    /// model takes [`UpdateRule::F32`] and a bf16 model one of the bf16
    /// rules; any [`MomentStorage`] goes with either. An f32 master starts
    /// as the model's current weights.
    pub fn with_config(model: &Qwen35Model, config: AdamWConfig) -> Result<Self, String> {
        const WHAT: &str = "AdamW::with_config";
        model.require_trainable(WHAT)?;
        match (model.precision(), config.update) {
            (Precision::F32, UpdateRule::F32) => {}
            (Precision::Bf16, UpdateRule::F32Master | UpdateRule::Bf16Kahan | UpdateRule::Bf16Stochastic { .. }) => {}
            (p, u) => {
                return Err(format!(
                    "{WHAT}: update rule {u:?} does not apply to a {p:?} model (f32 models take UpdateRule::F32, \
                     bf16 models F32Master, Bf16Kahan or Bf16Stochastic)"
                ))
            }
        }
        let rt = &model.rt;
        let mut out = Vec::new();
        for s in slots(model, None)? {
            let pw = s.param_window();
            pw.check(&format!("{WHAT}: {}", s.info.name))?;
            let n = pw.numel();
            let (moment, scale_bytes, aux_bytes) = slot_bytes(n, pw.dtype, config);
            let scales = if scale_bytes > 0 {
                Some((zeroed(rt, scale_bytes)?, zeroed(rt, scale_bytes)?))
            } else {
                None
            };
            let aux = match (pw.dtype, config.update) {
                (DType::BF16, UpdateRule::F32Master) => {
                    let b = zeroed(rt, aux_bytes)?;
                    let dense = Window {
                        rows: pw.rows,
                        width: pw.width,
                        ld: pw.width,
                        off: 0,
                        ..Window::dense(&b, 0, DType::F32, n)
                    };
                    window_copy(rt, &pw, &dense)?;
                    Some(b)
                }
                (DType::BF16, UpdateRule::Bf16Kahan) => Some(zeroed(rt, aux_bytes)?),
                _ => None,
            };
            out.push(SlotState {
                n,
                m: zeroed(rt, moment)?,
                v: zeroed(rt, moment)?,
                scales,
                aux,
            });
        }
        rt.synchronize()?;
        Ok(Self {
            config,
            precision: model.precision(),
            slots: out,
            step: 0,
            confirmed: std::cell::Cell::new(0),
            rt: Arc::clone(rt),
            dummy: zeroed(rt, 32)?,
        })
    }

    /// The stored precision this state was made with.
    pub fn config(&self) -> AdamWConfig {
        self.config
    }

    /// `AdamW update=... moments=... step=N`: the configuration and the
    /// step count, for a run's log or checkpoint metadata.
    pub fn describe(&self) -> String {
        format!("AdamW {} step={}", self.config, self.step)
    }

    /// Device bytes [`Self::with_config`] would allocate for `model` under
    /// `config`, at the sizes the allocator makes resident buffers
    /// ([`GpuRuntime::allocated_bytes_for`]), without allocating: what a
    /// configuration costs at shapes that do not fit.
    pub fn allocated_bytes_for(model: &Qwen35Model, config: AdamWConfig) -> Result<u64, String> {
        let hot = |b: usize| {
            if b == 0 {
                0
            } else {
                GpuRuntime::allocated_bytes_for(b.max(4), crate::runtime::BufferKind::Hot)
            }
        };
        let mut total = hot(32);
        for s in slots(model, None)? {
            let pw = s.param_window();
            let (moment, scales, aux) = slot_bytes(pw.numel(), pw.dtype, config);
            total += 2 * hot(moment) + 2 * hot(scales) + hot(aux);
        }
        Ok(total)
    }

    /// Device bytes the state holds (logical, before the allocator's
    /// rounding).
    pub fn state_bytes(&self) -> u64 {
        self.slots
            .iter()
            .map(|s| {
                let mut b = s.m.nbytes() + s.v.nbytes();
                if let Some((a, c)) = &s.scales {
                    b += a.nbytes() + c.nbytes();
                }
                if let Some(a) = &s.aux {
                    b += a.nbytes();
                }
                b as u64
            })
            .sum()
    }

    /// Steps taken so far (torch's `state["step"]`). A step
    /// [`Qwen35Model::adamw_step_unwaited`] encoded counts at once; if the
    /// runtime has since been poisoned (a GPU fault, which leaves the moments
    /// unknown), the count rolls back to the last one a wait of this state's
    /// saw complete ([`Qwen35Model::adamw_step`], or a moment or auxiliary
    /// read or write).
    pub fn step_count(&self) -> u64 {
        if self.rt.is_poisoned() {
            self.confirmed.get()
        } else {
            self.step
        }
    }

    /// Set the step count, restoring a checkpoint with
    /// [`Qwen35Model::write_adamw_moment`].
    pub fn set_step_count(&mut self, step: u64) {
        self.step = step;
        self.confirmed.set(step);
    }

    fn moment(&self, s: &SlotState, which: Moment) -> (GpuBuffer, Option<GpuBuffer>) {
        let first = which == Moment::First;
        let buf = if first { &s.m } else { &s.v };
        let scale = s.scales.as_ref().map(|(a, b)| if first { a } else { b }).cloned();
        (buf.clone(), scale)
    }

    /// State for `model`: one entry per parameter of the same size.
    fn check_model(&self, what: &str, model: &Qwen35Model) -> Result<(), String> {
        let ps = slots(model, None)?;
        if model.precision() != self.precision
            || ps.len() != self.slots.len()
            || ps.iter().zip(&self.slots).any(|(p, s)| p.param_window().numel() != s.n)
        {
            return Err(format!("{what}: the AdamW state was made for another model"));
        }
        // An `AdamW` made for a model loaded on another runtime matches in
        // shape; its buffers are outside this runtime's residency set.
        let rt = &model.rt;
        require_runtime(rt, &self.dummy, &format!("{what}: the AdamW state"))?;
        for (p, s) in ps.iter().zip(&self.slots) {
            let name = &p.info.name;
            for b in [Some(&s.m), Some(&s.v), s.aux.as_ref()]
                .into_iter()
                .flatten()
                .chain(s.scales.iter().flat_map(|(a, b)| [a, b]))
            {
                require_runtime(rt, b, &format!("{what}: {name}'s AdamW state"))?;
            }
        }
        Ok(())
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
    let n = param.try_numel()?;
    if n == 0 {
        return Ok(());
    }
    let w: Vec<Window<'_>> = views
        .iter()
        .map(|t| Window::dense(&t.buffer, t.byte_offset(), DType::F32, n))
        .collect();
    for (x, name) in w.iter().zip(names) {
        x.check(&format!("{WHAT}: {name}"))?;
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

/// The stored-precision step kernel for a parameter window of `p`, its
/// gradient's dtype `g`, and `config`.
fn step_kernel(p: DType, g: DType, config: AdamWConfig) -> Result<String, String> {
    let mom = match config.moments {
        MomentStorage::F32 => "f32",
        MomentStorage::Bf16 => "bf16",
        MomentStorage::Block8 => "q8",
    };
    let gname = match g {
        DType::F32 => "f32",
        DType::BF16 => "bf16",
        d => return Err(format!("a {d:?} gradient")),
    };
    match p {
        DType::F32 if g == DType::F32 => Ok(format!("qwen35_adamw_f32_plain_gf32_m{mom}")),
        DType::F32 => Err("an f32 parameter with a bf16 gradient".into()),
        DType::BF16 => {
            let rule = match config.update {
                UpdateRule::F32Master => "master",
                UpdateRule::Bf16Kahan => "kahan",
                UpdateRule::Bf16Stochastic { .. } => "sr",
                UpdateRule::F32 => return Err("a bf16 parameter under UpdateRule::F32".into()),
            };
            Ok(format!("qwen35_adamw_bf16_{rule}_g{gname}_m{mom}"))
        }
        d => Err(format!("a {d:?} parameter")),
    }
}

/// Encode one 8-bit moment block kernel (`qwen35_moment_q8_encode` /
/// `_decode`) over `n` dense elements.
fn dispatch_q8(
    rt: &GpuRuntime,
    encode: bool,
    q: &GpuBuffer,
    scale: &GpuBuffer,
    x: &Tensor,
    n: usize,
    first: bool,
) -> Result<(), String> {
    let p = rt.pipeline(if encode {
        "qwen35_moment_q8_encode"
    } else {
        "qwen35_moment_q8_decode"
    })?;
    dispatch_2d_tg(rt, &p, n.div_ceil(MOMENT_BLOCK), 1, MOMENT_BLOCK, |bnd| {
        if encode {
            set_gpu_buf_offset(bnd, &x.buffer, x.byte_offset(), 0);
            set_gpu_buf_offset(bnd, q, 0, 1);
            set_gpu_buf_offset(bnd, scale, 0, 2);
        } else {
            set_gpu_buf_offset(bnd, q, 0, 0);
            set_gpu_buf_offset(bnd, scale, 0, 1);
            set_gpu_buf_offset(bnd, &x.buffer, x.byte_offset(), 2);
        }
        set_u32(bnd, n as u32, 3);
        set_u32(bnd, u32::from(first), 4);
    })
}

/// A slot's dense state buffer as a window shaped like the parameter's.
fn state_window<'a>(buf: &'a GpuBuffer, dtype: DType, like: &Window<'_>) -> Window<'a> {
    Window {
        rows: like.rows,
        width: like.width,
        ld: like.width,
        off: 0,
        ..Window::dense(buf, 0, dtype, like.numel())
    }
}

impl Qwen35Model {
    /// One AdamW step on every parameter from `grads` (this model's
    /// [`Qwen35Model::train_step`] or a bank), with `weight_decay[i]` for
    /// [`Qwen35Model::parameter_table`] entry `i`, stored as `state`'s
    /// [`AdamWConfig`] says. Everything is checked before anything moves; the
    /// step count advances only when the step runs. Gradients or state from
    /// another runtime are refused, and a
    /// [`crate::qwen35_train::PendingStep`] from before the update can no
    /// longer be backpropagated. Every entry takes `hyper.lr`; see
    /// [`Qwen35Model::adamw_step_scaled`] for a learning rate per entry.
    pub fn adamw_step(
        &self,
        grads: &Qwen35Grads,
        state: &mut AdamW,
        hyper: &AdamWHyper,
        weight_decay: &[f32],
    ) -> Result<(), String> {
        self.encode_adamw("Qwen35Model::adamw_step", grads, state, hyper, weight_decay, None)?;
        self.rt.synchronize()?;
        state.confirmed.set(state.step);
        Ok(())
    }

    /// [`Self::adamw_step`] encoded and not waited for: the next wait (the
    /// next step's, or [`GpuRuntime::synchronize`]) runs it. The step count
    /// advances now; if the GPU then fails, the runtime is poisoned and
    /// [`AdamW::step_count`] rolls back to the count before every step this
    /// state has not waited for. A clipped iteration's waits are then the
    /// step's own and `grad_sq_norm`'s.
    pub fn adamw_step_unwaited(
        &self,
        grads: &Qwen35Grads,
        state: &mut AdamW,
        hyper: &AdamWHyper,
        weight_decay: &[f32],
    ) -> Result<(), String> {
        self.encode_adamw("Qwen35Model::adamw_step_unwaited", grads, state, hyper, weight_decay, None)
    }

    /// [`Qwen35Model::adamw_step`] with entry `i` at learning rate
    /// `hyper.lr * lr_scale[i]`: torch's AdamW with one param group per
    /// entry, so the scaled lr forms both the decoupled decay factor
    /// `1 - lr * wd` and the step size. A scale of 0 freezes the entry as a
    /// torch group at lr 0 does: the parameter keeps its bits while its
    /// moments still update. `lr_scale` has one value per
    /// [`Qwen35Model::parameter_table`] entry, each finite and >= 0, and the
    /// scaled lr must be finite; all of it is checked before anything moves.
    pub fn adamw_step_scaled(
        &self,
        grads: &Qwen35Grads,
        state: &mut AdamW,
        hyper: &AdamWHyper,
        weight_decay: &[f32],
        lr_scale: &[f64],
    ) -> Result<(), String> {
        self.encode_adamw(
            "Qwen35Model::adamw_step_scaled",
            grads,
            state,
            hyper,
            weight_decay,
            Some(lr_scale),
        )?;
        self.rt.synchronize()?;
        state.confirmed.set(state.step);
        Ok(())
    }

    /// Check everything, then encode the step: one table-driven dispatch per
    /// step kernel the parameters need (one on an f32 model; a bf16 model's
    /// f32 vectors take their own), where a dispatch per parameter (~300 on
    /// the 2B) used to be. `lr_scale` `None` is every entry at 1. Advances
    /// `state.step`.
    fn encode_adamw(
        &self,
        what: &str,
        grads: &Qwen35Grads,
        state: &mut AdamW,
        hyper: &AdamWHyper,
        weight_decay: &[f32],
        lr_scale: Option<&[f64]>,
    ) -> Result<(), String> {
        self.require_trainable(what)?;
        check_hyper(what, hyper)?;
        state.check_model(what, self)?;
        let ps = slots(self, Some(grads)).map_err(|e| format!("{what}: {e}"))?;
        if weight_decay.len() != ps.len() {
            return Err(format!(
                "{what}: {} weight decays for {} parameters",
                weight_decay.len(),
                ps.len()
            ));
        }
        if let Some(scale) = lr_scale {
            if scale.len() != ps.len() {
                return Err(format!("{what}: {} lr scales for {} parameters", scale.len(), ps.len()));
            }
        }
        let mut plan = Vec::with_capacity(ps.len());
        for (i, ((s, st), &wd)) in ps.iter().zip(&state.slots).zip(weight_decay).enumerate() {
            let name = &s.info.name;
            if !(wd.is_finite() && wd >= 0.0) {
                return Err(format!("{what}: {name}: weight decay {wd} must be finite and >= 0"));
            }
            // torch param-group semantics: the group's lr forms both the
            // decay factor and the step size, so the whole lr is scaled.
            let entry_hyper = match lr_scale {
                None => *hyper,
                Some(scale) => {
                    let k = scale[i];
                    if !(k.is_finite() && k >= 0.0) {
                        return Err(format!("{what}: {name}: lr scale {k} must be finite and >= 0"));
                    }
                    let h = AdamWHyper {
                        lr: hyper.lr * k,
                        ..*hyper
                    };
                    check_hyper(&format!("{what}: {name}"), &h)?;
                    h
                }
            };
            let pw = s.param_window();
            let gw = s
                .grad_window()
                .ok_or_else(|| format!("{what}: {name} has no gradient"))?;
            pw.check(&format!("{what}: {name} parameter"))?;
            gw.check(&format!("{what}: {name} gradient"))?;
            if !gw.same_layout(&pw) {
                return Err(format!("{what}: {name}: the gradient is not laid out as the parameter"));
            }
            let kernel = step_kernel(pw.dtype, gw.dtype, state.config).map_err(|e| format!("{what}: {name}: {e}"))?;
            plan.push((pw, gw, st, kernel, entry_hyper, f64::from(wd)));
        }

        // torch: the step count increments, then the scalars are formed in f64.
        // `u64` wraps only at `u64::MAX`. A release build without overflow
        // checks used to turn that into t = 0, which zeroes the bias correction
        // and sends the parameter update to infinity.
        let next = state
            .step
            .checked_add(1)
            .ok_or_else(|| format!("{what}: step count {} + 1 does not fit in u64", state.step))?;
        let seed = match state.config.update {
            UpdateRule::Bf16Stochastic { seed } => seed,
            _ => 0,
        };
        // The slot tables, one per kernel, each entry's salt its
        // parameter-table index (as the stochastic rounding keys it).
        let mut tables: Vec<(&str, Vec<u32>, u32, u32)> = Vec::new();
        for (salt, (pw, gw, st, kernel, entry_hyper, wd)) in plan.iter().enumerate() {
            let at = match tables.iter().position(|(k, ..)| k == kernel) {
                Some(at) => at,
                None => {
                    tables.push((kernel.as_str(), Vec::new(), 0, 0));
                    tables.len() - 1
                }
            };
            let (_, words, n_slots, blocks) = &mut tables[at];
            let addr = |b: &GpuBuffer, off: usize| b.metal().gpuAddress().wrapping_add(off as u64);
            let dummy = &state.dummy;
            let (ms, vs) = match &st.scales {
                Some((a, b)) => (a, b),
                None => (dummy, dummy),
            };
            let aux = st.aux.as_ref().unwrap_or(dummy);
            let n = u32::try_from(st.n).map_err(|_| format!("{what}: {} elements exceed u32", st.n))?;
            let slot_blocks =
                u32::try_from(st.n.div_ceil(MOMENT_BLOCK)).map_err(|_| format!("{what}: too many blocks"))?;
            for a in [
                addr(pw.buf, pw.byte_off),
                addr(gw.buf, gw.byte_off),
                addr(&st.m, 0),
                addr(&st.v, 0),
                addr(aux, 0),
                addr(ms, 0),
                addr(vs, 0),
            ] {
                words.extend([a as u32, (a >> 32) as u32]);
            }
            words.extend([n, pw.width as u32, pw.ld as u32, pw.off as u32, *blocks, 0]);
            words.extend(adamw_scalars(entry_hyper, next, *wd).map(f32::to_bits));
            words.extend([
                seed as u32,
                (seed >> 32) as u32,
                next as u32,
                (next >> 32) as u32,
                salt as u32,
                0,
            ]);
            debug_assert_eq!(words.len() % (SLOT_BYTES / 4), 0);
            *n_slots += 1;
            *blocks = blocks
                .checked_add(slot_blocks)
                .ok_or_else(|| format!("{what}: the step's threadgroups exceed u32"))?;
        }
        let mut encoded = Vec::with_capacity(tables.len());
        for (kernel, words, n_slots, blocks) in &tables {
            // Fresh, so written without waiting for the GPU.
            encoded.push((
                self.rt.pipeline(kernel)?,
                self.rt.alloc_buffer_from_u32(words)?,
                *n_slots,
                *blocks,
            ));
        }
        self.bump_param_generation();
        for (p, table, n_slots, blocks) in &encoded {
            dispatch_2d_tg(&self.rt, p, *blocks as usize, 1, MOMENT_BLOCK, |bnd| {
                set_gpu_buf(bnd, table, 0);
                set_u32(bnd, *n_slots, 1);
            })?;
        }
        state.step = next;
        Ok(())
    }

    /// The sum of squares of every gradient in `grads` (this model's
    /// [`Qwen35Model::train_step`], or a bank, f32 or bf16), over the
    /// parameters of [`Qwen35Model::parameter_table`]: the square of the
    /// global L2 norm `torch.nn.utils.clip_grad_norm_` takes. Each row of
    /// each gradient is summed in f32 on the GPU in a fixed order and the
    /// rows are added in f64, so it is deterministic; torch's own reduction
    /// order differs, so the two agree to rounding, not bits.
    pub fn grad_sq_norm(&self, grads: &Qwen35Grads) -> Result<f64, String> {
        const WHAT: &str = "Qwen35Model::grad_sq_norm";
        let ps = slots(self, Some(grads)).map_err(|e| format!("{WHAT}: {e}"))?;
        let mut plan = Vec::with_capacity(ps.len());
        let mut rows = 0usize;
        for s in &ps {
            let name = &s.info.name;
            let w = s
                .grad_window()
                .ok_or_else(|| format!("{WHAT}: {name} has no gradient"))?;
            w.check(&format!("{WHAT}: {name} gradient"))?;
            let n = w.rows;
            plan.push((w, rows));
            rows += n;
        }
        u32::try_from(rows).map_err(|_| format!("{WHAT}: {rows} rows exceed u32"))?;
        let out = self.rt.alloc_tensor_f32(&[rows])?;
        for (w, at) in &plan {
            let p = self.rt.pipeline(if w.dtype == DType::BF16 {
                "qwen35_sq_sum_rows_bf16"
            } else {
                "qwen35_sq_sum_rows_f32"
            })?;
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
    /// [`Qwen35Model::read_parameters`] lays out the values: bf16 moments
    /// widened, 8-bit ones decoded.
    pub fn read_adamw_moment(&self, state: &AdamW, which: Moment, dst: &[Tensor]) -> Result<(), String> {
        const WHAT: &str = "Qwen35Model::read_adamw_moment";
        self.moment_copy(WHAT, state, which, dst, false)
    }

    /// Set one of `state`'s moments from `src`, laid out as
    /// [`Self::read_adamw_moment`] reads it; with [`AdamW::set_step_count`]
    /// and [`Self::write_adamw_aux`] this restores a checkpoint. bf16 moments
    /// round to nearest and 8-bit ones encode per block, so a write of what a
    /// read gave is the same state. Every tensor is checked before anything
    /// is written.
    pub fn write_adamw_moment(&self, state: &mut AdamW, which: Moment, src: &[Tensor]) -> Result<(), String> {
        const WHAT: &str = "Qwen35Model::write_adamw_moment";
        self.moment_copy(WHAT, state, which, src, true)
    }

    fn moment_copy(&self, what: &str, state: &AdamW, which: Moment, ts: &[Tensor], write: bool) -> Result<(), String> {
        state.check_model(what, self)?;
        let ps = slots(self, None)?;
        check(what, &self.rt, &ps, ts)?;
        let dtype = match state.config.moments {
            MomentStorage::F32 => DType::F32,
            MomentStorage::Bf16 => DType::BF16,
            MomentStorage::Block8 => DType::F32,
        };
        for ((s, st), t) in ps.iter().zip(&state.slots).zip(ts) {
            let pw = s.param_window();
            let (buf, scale) = state.moment(st, which);
            let e = |e: String| format!("{what}: {}: {e}", s.info.name);
            match scale {
                Some(scale) => {
                    dispatch_q8(&self.rt, write, &buf, &scale, t, st.n, which == Moment::First).map_err(e)?
                }
                None => {
                    let mine = state_window(&buf, dtype, &pw);
                    let theirs = Window::tensor_as(t, &pw).map_err(e)?;
                    let (from, to) = if write { (theirs, mine) } else { (mine, theirs) };
                    window_copy(&self.rt, &from, &to).map_err(e)?;
                }
            }
        }
        self.rt.synchronize()?;
        state.confirmed.set(state.step);
        Ok(())
    }

    /// Copy `state`'s auxiliary state into `dst` (one dense f32 tensor per
    /// [`Qwen35Model::parameter_table`] entry, as
    /// [`Qwen35Model::read_parameters`] lays them out): under
    /// [`UpdateRule::F32Master`] the f32 masters, under
    /// [`UpdateRule::Bf16Kahan`] the compensations, widened. An entry stored
    /// in f32 has none, and reads as its parameter's value (master) or zero
    /// (Kahan). Other rules keep no auxiliary state and are refused.
    pub fn read_adamw_aux(&self, state: &AdamW, dst: &[Tensor]) -> Result<(), String> {
        self.aux_copy("Qwen35Model::read_adamw_aux", state, dst, false)
    }

    /// Set `state`'s auxiliary state from `src`, laid out as
    /// [`Self::read_adamw_aux`] reads it (a compensation rounds to nearest).
    /// Entries stored in f32 have none and are skipped. Every tensor is
    /// checked before anything is written.
    pub fn write_adamw_aux(&self, state: &mut AdamW, src: &[Tensor]) -> Result<(), String> {
        self.aux_copy("Qwen35Model::write_adamw_aux", state, src, true)
    }

    fn aux_copy(&self, what: &str, state: &AdamW, ts: &[Tensor], write: bool) -> Result<(), String> {
        let dtype = match state.config.update {
            UpdateRule::F32Master => DType::F32,
            UpdateRule::Bf16Kahan => DType::BF16,
            _ => {
                return Err(format!(
                    "{what}: update rule {} keeps no auxiliary state",
                    state.config.rule_name()
                ))
            }
        };
        state.check_model(what, self)?;
        let ps = slots(self, None)?;
        check(what, &self.rt, &ps, ts)?;
        for ((s, st), t) in ps.iter().zip(&state.slots).zip(ts) {
            let pw = s.param_window();
            let e = |e: String| format!("{what}: {}: {e}", s.info.name);
            let theirs = Window::tensor_as(t, &pw).map_err(e)?;
            match (&st.aux, write) {
                (Some(aux), _) => {
                    let mine = state_window(aux, dtype, &pw);
                    let (from, to) = if write { (theirs, mine) } else { (mine, theirs) };
                    window_copy(&self.rt, &from, &to).map_err(e)?;
                }
                (None, true) => {}
                (None, false) if dtype == DType::F32 => window_copy(&self.rt, &pw, &theirs).map_err(e)?,
                (None, false) => {
                    let p = self.rt.pipeline("zero_f32")?;
                    crate::dispatch::dispatch_1d(&self.rt, &p, st.n, |bnd| {
                        set_gpu_buf_offset(bnd, &t.buffer, t.byte_offset(), 0);
                        set_u32(bnd, st.n as u32, 1);
                    })?;
                }
            }
        }
        self.rt.synchronize()?;
        state.confirmed.set(state.step);
        Ok(())
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
