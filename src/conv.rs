//! 2D convolution on the GPU's matrix units: Metal Performance Primitives' `convolution2d` tensor
//! op, driven from NCHW f32 tensors (kernels/conv2d_tensorops.metal).
//!
//! One call is three dispatches in one binder: the weights to HWIO in the compute precision, the
//! input to a padded NHWC copy, and the op, which stores its tile straight to NCHW with the bias
//! added. The compute precision is f16 (f32 accumulate and output) or f32. At SAM's 3x3,
//! 256 -> 256 conv at 288^2 the op alone took 3.26 ms in f16 and 14.1 ms in f32, where im2col plus
//! the f32 GEMM took 53.7 ms (bench/results/conv2d_20261104/README.md).
//!
//! Scope: 3x3 kernels, stride 1, dilation 1, one group, any padding, and an output channel count
//! that is a multiple of 16. [`conv2d_tensorops_supports`] says whether a shape is covered, so a
//! caller can route only those here and keep its general path for the rest.

use crate::runtime::mtl_size;
use crate::tensor::{DType, Tensor};

/// The arithmetic of [`conv2d_nchw`]'s matrix-unit op. Both accumulate in f32 and write f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvPrecision {
    /// Inputs and weights rounded to f16: relative L2 error about 3e-4 at K = 2304.
    F16,
    /// f32 inputs and weights: relative L2 error about 1e-6 at K = 2304.
    F32,
}

/// The kernel size the op is instantiated for.
pub const CONV2D_TOPS_KERNEL: usize = 3;
/// Destination tile width; the packed source is padded to a whole number of these, because a
/// partial tile computes wrong values.
const TILE_W: usize = 32;

/// Whether [`conv2d_nchw`] covers this convolution. Shapes are PyTorch's: `x [n, c_in, h, w]`,
/// `w [c_out, c_in / groups, kh, kw]`.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_tensorops_supports(
    c_in: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> bool {
    kh == CONV2D_TOPS_KERNEL
        && kw == CONV2D_TOPS_KERNEL
        && stride == (1, 1)
        && dilation == (1, 1)
        && groups == 1
        && c_in > 0
        && c_out > 0
        && c_out.is_multiple_of(16)
}

/// The output tile: 64 channels when they divide `c_out`, else 16.
fn channel_tile(c_out: usize) -> usize {
    if c_out.is_multiple_of(64) {
        64
    } else {
        16
    }
}

/// The scratch [`conv2d_nchw`] allocates for `n` images of `c_in` channels giving an `oh x ow`
/// output of `c_out` channels: the padded NHWC input `[n, oh + 2, ow_p + 2, c_in]` (`ow_p` is `ow`
/// rounded up to whole tiles) and the HWIO weights `[3, 3, c_in, c_out]`. `None` on overflow.
fn scratch_shapes(n: usize, c_in: usize, c_out: usize, oh: usize, ow: usize) -> Option<([usize; 4], [usize; 4])> {
    let k = CONV2D_TOPS_KERNEL;
    let ow_p = ow.div_ceil(TILE_W).checked_mul(TILE_W)?;
    Some((
        [n, oh.checked_add(k - 1)?, ow_p.checked_add(k - 1)?, c_in],
        [k, k, c_in, c_out],
    ))
}

/// The output extent `(oh, ow)` of the 3x3, stride-1 convolution of an `h x w` image padded by `pad`
/// on every side, or `None` when the padded image is smaller than the kernel or the sum overflows.
fn output_extent(h: usize, wd: usize, pad: usize) -> Option<(usize, usize)> {
    let reach = CONV2D_TOPS_KERNEL - 1;
    let padded = |d: usize| pad.checked_mul(2).and_then(|p| p.checked_add(d));
    let (hp, wp) = (padded(h)?, padded(wd)?);
    (hp > reach && wp > reach).then(|| (hp - reach, wp - reach))
}

/// The scratch shapes `(xp, wh)` of a convolution the op can index, or `None` when any tensor it
/// spans (or the scratch) holds more than `i32::MAX` elements.
fn int_scratch(
    n: usize,
    c_in: usize,
    c_out: usize,
    oh: usize,
    ow: usize,
    h: usize,
    wd: usize,
) -> Option<([usize; 4], [usize; 4])> {
    // The op indexes in int, and its slices span the whole tensors.
    let int = |s: &[usize]| {
        s.iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .filter(|&e| e <= i32::MAX as usize)
    };
    scratch_shapes(n, c_in, c_out, oh, ow).filter(|(xp, wh)| {
        // `wh` is the weights kernel's `uint total = CO * CI * K * K` and the op's HWIO strides.
        int(xp).is_some() && int(wh).is_some() && int(&[n, c_out, oh, ow]).is_some() && int(&[n, c_in, h, wd]).is_some()
    })
}

/// Elements of [`conv2d_nchw`]'s scratch for this convolution, in the compute precision (so at most
/// this many f32s), or `None` when the count overflows. Callers that budget memory charge this.
pub fn conv2d_nchw_scratch_elems(n: usize, c_in: usize, c_out: usize, oh: usize, ow: usize) -> Option<usize> {
    let (xp, wh) = scratch_shapes(n, c_in, c_out, oh, ow)?;
    let numel = |s: [usize; 4]| s.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
    numel(xp)?.checked_add(numel(wh)?)
}

/// `y = conv2d(x, w, padding = pad) + bias`, stride 1, for f32 NCHW `x [n, c_in, h, w]`, OIHW
/// `w [c_out, c_in, 3, 3]`, optional `bias [c_out]`, and `y [n, c_out, h + 2 pad - 2, w + 2 pad - 2]`.
/// Encoded on `x`'s runtime; nothing waits.
pub fn conv2d_nchw(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    y: &Tensor,
    pad: usize,
    precision: ConvPrecision,
) -> Result<(), String> {
    let what = "conv2d_nchw";
    x.validate()?;
    w.validate()?;
    y.validate()?;
    if let Some(b) = bias {
        b.validate()?;
    }
    let rt = x.runtime();
    for (name, t) in [("x", Some(x)), ("w", Some(w)), ("y", Some(y)), ("bias", bias)] {
        let Some(t) = t else { continue };
        if t.dtype != DType::F32 {
            return Err(format!("{what}: {name} must be f32, got {:?}", t.dtype));
        }
        if !std::sync::Arc::ptr_eq(t.runtime(), rt) {
            return Err(format!("{what}: {name} is on another runtime"));
        }
    }
    let [n, c_in, h, wd] = x.shape() else {
        return Err(format!("{what}: x must be [n, c_in, h, w], got {:?}", x.shape()));
    };
    let (n, c_in, h, wd) = (*n, *c_in, *h, *wd);
    let [c_out, wc, kh, kw] = w.shape() else {
        return Err(format!("{what}: w must be [c_out, c_in, kh, kw], got {:?}", w.shape()));
    };
    let (c_out, kh, kw) = (*c_out, *kh, *kw);
    if *wc != c_in || !conv2d_tensorops_supports(c_in, c_out, kh, kw, (1, 1), (1, 1), 1) {
        return Err(format!(
            "{what}: w {:?} against x {:?} is outside the op's scope (3x3, c_in matching x, c_out a multiple of 16)",
            w.shape(),
            x.shape()
        ));
    }
    let k = CONV2D_TOPS_KERNEL;
    let Some((oh, ow)) = output_extent(h, wd, pad) else {
        return Err(format!(
            "{what}: x {:?} padded by {pad} is smaller than the 3x3 kernel, or the padded size overflows",
            x.shape()
        ));
    };
    if y.shape() != [n, c_out, oh, ow] {
        return Err(format!(
            "{what}: y must be {:?}, got {:?}",
            [n, c_out, oh, ow],
            y.shape()
        ));
    }
    if let Some(b) = bias {
        if b.shape() != [c_out] {
            return Err(format!("{what}: bias must be [{c_out}], got {:?}", b.shape()));
        }
    }
    for (name, t) in [("x", Some(x)), ("w", Some(w)), ("bias", bias)] {
        if t.is_some_and(|t| t.overlaps(y)) {
            return Err(format!("{what}: y overlaps {name}"));
        }
    }
    if n == 0 {
        return Ok(());
    }

    let Some((xp_shape, wh_shape)) = int_scratch(n, c_in, c_out, oh, ow, h, wd) else {
        return Err(format!(
            "{what}: x {:?} -> y {:?} exceeds int indexing",
            x.shape(),
            y.shape()
        ));
    };
    let [_, hp, wp, _] = xp_shape;
    let ow_p = wp - (k - 1);

    let (elem, pack_k, weights_k) = match precision {
        ConvPrecision::F16 => (DType::F16, "conv2d_pack_nhwc_f16", "conv2d_weights_hwio_f16"),
        ConvPrecision::F32 => (DType::F32, "conv2d_pack_nhwc_f32", "conv2d_weights_hwio_f32"),
    };
    let ot = channel_tile(c_out);
    let tops_k = match (precision, ot) {
        (ConvPrecision::F16, 64) => "conv2d_tops_nchw_k3_o64_w32_f16",
        (ConvPrecision::F16, _) => "conv2d_tops_nchw_k3_o16_w32_f16",
        (ConvPrecision::F32, 64) => "conv2d_tops_nchw_k3_o64_w32_f32",
        (ConvPrecision::F32, _) => "conv2d_tops_nchw_k3_o16_w32_f32",
    };
    let pack_p = rt.pipeline(pack_k)?;
    let weights_p = rt.pipeline(weights_k)?;
    let tops_p = rt.pipeline(tops_k)?;

    // Every element of each scratch is written before it is read: the pack covers all of
    // [n, hp, wp, c_in] (halo and width padding as zeros), and the weights all of [k, k, c_in, c_out].
    let xp = rt.alloc_tensor_unzeroed(&xp_shape, elem)?;
    let wh = rt.alloc_tensor_unzeroed(&wh_shape, elem)?;
    let wn = c_out * c_in * k * k;
    let u = |v: usize| v as u32;

    rt.with_binder(|bnd| {
        bnd.set_pipeline(&weights_p);
        bnd.bind_tensor(w, 0);
        bnd.bind_tensor(&wh, 1);
        bnd.bind_u32(u(c_out), 2);
        bnd.bind_u32(u(c_in), 3);
        bnd.bind_u32(u(k), 4);
        bnd.dispatch(mtl_size(wn.div_ceil(256), 1, 1), mtl_size(256, 1, 1));

        bnd.set_pipeline(&pack_p);
        bnd.bind_tensor(x, 0);
        bnd.bind_tensor(&xp, 1);
        bnd.bind_u32(u(c_in), 2);
        bnd.bind_u32(u(h), 3);
        bnd.bind_u32(u(wd), 4);
        bnd.bind_u32(u(pad), 5);
        bnd.bind_u32(u(hp), 6);
        bnd.bind_u32(u(wp), 7);
        bnd.dispatch(mtl_size(wp.div_ceil(32), c_in.div_ceil(32), n * hp), mtl_size(32, 8, 1));

        bnd.set_pipeline(&tops_p);
        bnd.bind_tensor(&xp, 0);
        bnd.bind_tensor(&wh, 1);
        bnd.bind_tensor(y, 2);
        // Unread without a bias; any live buffer satisfies the binding.
        bnd.bind_tensor(bias.unwrap_or(&wh), 3);
        bnd.bind_u32(u32::from(bias.is_some()), 4);
        bnd.bind_u32(u(c_in), 5);
        bnd.bind_u32(u(c_out), 6);
        bnd.bind_u32(u(oh), 7);
        bnd.bind_u32(u(ow), 8);
        bnd.bind_u32(u(ow_p), 9);
        bnd.bind_u32(u(n), 10);
        bnd.dispatch(mtl_size(c_out / ot, ow_p / TILE_W, n * oh), mtl_size(128, 1, 1));
        Ok(())
    })
}

#[cfg(test)]
mod extent_tests {
    use super::*;

    #[test]
    fn output_extent_refuses_a_pad_whose_double_wraps() {
        assert_eq!(output_extent(5, 7, 1), Some((5, 7)));
        assert_eq!(output_extent(1, 1, 0), None);
        assert_eq!(output_extent(5, 5, usize::MAX / 2 + 1), None);
        assert_eq!(output_extent(5, 5, usize::MAX), None);
    }

    #[test]
    fn weights_past_i32_are_refused_before_any_dispatch() {
        // x [1, 65536, 1, 1] and y [1, 65536, 1, 1] are tiny; the weights are 9 * 2^32 elements,
        // which the weights kernel's `uint total = CO * CI * K * K` would wrap.
        assert!(int_scratch(1, 1 << 16, 1 << 16, 1, 1, 1, 1).is_none());
        assert!(int_scratch(1, 1 << 12, 1 << 12, 1, 1, 1, 1).is_some());
    }
}
