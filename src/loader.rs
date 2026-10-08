//! Reading a checkpoint's tensors onto the device: the one loader
//! [`crate::qwen35_model::Qwen35Model`] and
//! [`crate::embedgemma2::EmbedGemma2Model`] share.
//!
//! Each tensor is named `{prefix}{rest}` and its shape is checked against
//! what the model expects. Nothing is held on the host longer than it takes
//! to reach the device: a projection's parts are placed into the packed
//! operand one at a time as they are read ([`Loader::linear`]), and a table
//! is read straight into its device tensor ([`Loader::f32_into`],
//! [`Loader::bf16_into`]). Every name read is recorded, so a model that
//! implements everything under its prefix can refuse a checkpoint carrying a
//! tensor it would silently ignore ([`Loader::refuse_unread`]).
//!
//! The two models kept near-copies of this (`name`/`f32`/`norm`/`linear`)
//! until EmbedGemma2's copy fell behind: it held every part of a projection
//! and a packed host copy at once, and read its embedding through a `Vec`.
//! One owner keeps that from recurring. `bert`'s loader stays separate for
//! now: it has no prefix, refuses non-finite weights, and allocates its
//! weights `Cold` (ft-afb1c42 is the convergence card).

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::sync::Arc;

use crate::qwen35;
use crate::qwen35_model::Precision;
use crate::runtime::GpuRuntime;
use crate::safetensors::SafeTensors;
use crate::tensor::{GpuBuffer, Tensor};

pub(crate) struct Loader<'a> {
    st: &'a SafeTensors,
    prefix: &'a str,
    rt: &'a Arc<GpuRuntime>,
    /// Every tensor name read so far, for [`Self::refuse_unread`].
    read: RefCell<BTreeSet<String>>,
}

impl<'a> Loader<'a> {
    pub(crate) fn new(rt: &'a Arc<GpuRuntime>, st: &'a SafeTensors, prefix: &'a str) -> Self {
        Self {
            st,
            prefix,
            rt,
            read: RefCell::new(BTreeSet::new()),
        }
    }

    /// `rest`'s full name, recorded as read.
    fn take(&self, rest: &str) -> String {
        let name = format!("{}{rest}", self.prefix);
        self.read.borrow_mut().insert(name.clone());
        name
    }

    /// A tensor widened to f32, with its shape checked.
    pub(crate) fn f32(&self, rest: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let name = self.take(rest);
        let (got, data) = self.st.read_f32(&name)?;
        if got != shape {
            return Err(format!("{name}: shape {got:?}, expected {shape:?}"));
        }
        Ok(data)
    }

    /// A bf16 tensor's bit patterns, with its shape checked.
    pub(crate) fn bf16(&self, rest: &str, shape: &[usize]) -> Result<Vec<u16>, String> {
        let name = self.take(rest);
        let (got, bits) = self.st.read_bf16_bits(&name)?;
        if got != shape {
            return Err(format!("{name}: shape {got:?}, expected {shape:?}"));
        }
        Ok(bits)
    }

    /// A tensor widened to f32 straight into `t`'s shared storage, with no
    /// host copy. `t` must be an f32 tensor of exactly `shape`'s elements.
    pub(crate) fn f32_into(&self, rest: &str, shape: &[usize], t: &Tensor) -> Result<(), String> {
        let name = self.take(rest);
        self.st.read_f32_into(&name, shape, &mut t.buffer.try_contents_f32()?)
    }

    /// A bf16 tensor's bits straight into `t`'s shared storage, with no host
    /// copy. `t` must be a bf16 tensor of exactly `shape`'s elements.
    pub(crate) fn bf16_into(&self, rest: &str, shape: &[usize], t: &Tensor) -> Result<(), String> {
        let name = self.take(rest);
        self.st
            .read_bf16_bits_into(&name, shape, &mut t.buffer.try_contents_u16()?)
    }

    /// One finite value (a `[1]` or scalar tensor).
    pub(crate) fn scalar(&self, rest: &str) -> Result<f32, String> {
        let name = self.take(rest);
        let (_, data) = self.st.read_f32(&name)?;
        match data.as_slice() {
            [s] if s.is_finite() => Ok(*s),
            other => Err(format!("{name}: expected one finite value, got {other:?}")),
        }
    }

    /// `data` in a weight buffer (`BufferKind::Hot`: it lives as long as the
    /// model, not as a step temporary).
    pub(crate) fn buf(&self, data: &[f32]) -> Result<GpuBuffer, String> {
        let b = self.rt.alloc_buffer_hot(data.len().max(1) * 4)?;
        b.write_f32(data);
        Ok(b)
    }

    /// A `[dim]` weight (a norm's, say) as the checkpoint holds it, in f32.
    pub(crate) fn norm(&self, rest: &str, dim: usize) -> Result<GpuBuffer, String> {
        self.buf(&self.f32(rest, &[dim])?)
    }

    /// `nn.Linear` weights `[out_i, in]` packed side by side into the right
    /// operand `[in, sum(out_i)]` of one GEMM, in `precision`. Each part is
    /// placed straight into the tensor's shared storage as it is read and
    /// dropped before the next is read, so the host holds one part at a time.
    pub(crate) fn linear(
        &self,
        parts: &[(&str, usize)],
        in_features: usize,
        precision: Precision,
    ) -> Result<Tensor, String> {
        let total = parts
            .iter()
            .try_fold(0usize, |acc, &(_, o)| acc.checked_add(o))
            .ok_or("linear: output widths overflow usize")?;
        let mut col0 = 0;
        match precision {
            Precision::Bf16 => {
                let t = self.rt.alloc_tensor_bf16_hot(&[in_features, total])?;
                for &(rest, out) in parts {
                    let part = self.bf16(rest, &[out, in_features])?;
                    qwen35::place_linear_part(&mut t.buffer.try_contents_u16()?, total, col0, &part, out, in_features)?;
                    col0 += out;
                }
                Ok(t)
            }
            Precision::F32 => {
                let t = self.rt.alloc_tensor_f32_hot(&[in_features, total])?;
                for &(rest, out) in parts {
                    let part = self.f32(rest, &[out, in_features])?;
                    qwen35::place_linear_part(&mut t.buffer.try_contents_f32()?, total, col0, &part, out, in_features)?;
                    col0 += out;
                }
                Ok(t)
            }
        }
    }

    /// A tensor under the prefix that the forward never read would be a
    /// parameter it silently ignores (a bias, an extra norm): refuse it.
    pub(crate) fn refuse_unread(&self) -> Result<(), String> {
        let read = self.read.borrow();
        let unread: Vec<&str> = self
            .st
            .names()
            .filter(|n| n.starts_with(self.prefix) && !read.contains(*n))
            .collect();
        if unread.is_empty() {
            return Ok(());
        }
        let shown: Vec<&str> = unread.iter().copied().take(8).collect();
        Err(format!(
            "{} tensor(s) under {:?} are not part of the model this loader implements: {shown:?}{}",
            unread.len(),
            self.prefix,
            if unread.len() > shown.len() { " ..." } else { "" }
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// safetensors bytes for `(name, dtype, shape, values)`; bf16 values are
    /// taken as given (they are chosen exactly representable).
    fn checkpoint(tensors: &[(&str, &str, Vec<usize>, Vec<f32>)]) -> SafeTensors {
        let (mut header, mut data) = (String::from("{"), Vec::new());
        for (i, (name, dtype, shape, vals)) in tensors.iter().enumerate() {
            let bytes: Vec<u8> = match *dtype {
                "BF16" => vals
                    .iter()
                    .flat_map(|&v| crate::tensor::f32_to_bf16_bits(v).to_le_bytes())
                    .collect(),
                _ => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
            };
            let shape: Vec<String> = shape.iter().map(usize::to_string).collect();
            header += &format!(
                "{}\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                if i > 0 { "," } else { "" },
                shape.join(","),
                data.len(),
                data.len() + bytes.len()
            );
            data.extend(bytes);
        }
        header.push('}');
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend(header.into_bytes());
        out.extend(data);
        SafeTensors::from_bytes("loader test", out).unwrap()
    }

    /// Part `p`'s element `[r, k]`: distinct everywhere, exact in bf16.
    fn val(p: usize, r: usize, k: usize) -> f32 {
        (100 * p + 10 * r + k) as f32
    }

    /// Each part lands in its own columns, transposed: packed `[k, col0_p +
    /// r]` is part `p`'s `[r, k]`. The expectation is built here by index, not
    /// by `place_linear_part`, so it also checks that helper. Parts of mixed
    /// dtype, in both precisions.
    #[test]
    fn linear_places_each_part_in_its_columns() {
        let rt = GpuRuntime::new().expect("runtime");
        let in_f = 3;
        let outs = [5usize, 2, 4];
        let total: usize = outs.iter().sum();
        let names = ["m.q.weight", "m.k.weight", "m.v.weight"];
        let tensor = |p: usize, dtype: &'static str| {
            let vals = (0..outs[p] * in_f).map(|i| val(p, i / in_f, i % in_f)).collect();
            (names[p], dtype, vec![outs[p], in_f], vals)
        };
        let mut want = vec![0f32; in_f * total];
        let mut col0 = 0;
        for (p, &o) in outs.iter().enumerate() {
            for r in 0..o {
                for k in 0..in_f {
                    want[k * total + col0 + r] = val(p, r, k);
                }
            }
            col0 += o;
        }
        let parts: Vec<(&str, usize)> = names.iter().map(|n| &n[2..]).zip(outs).collect();

        let st = checkpoint(&[tensor(0, "F32"), tensor(1, "BF16"), tensor(2, "F32")]);
        let ld = Loader::new(&rt, &st, "m.");
        let t = ld.linear(&parts, in_f, Precision::F32).unwrap();
        assert_eq!(t.shape(), &[in_f, total]);
        assert_eq!(t.buffer.read_f32()[..in_f * total], want[..]);
        ld.refuse_unread().unwrap();

        let st = checkpoint(&[tensor(0, "BF16"), tensor(1, "BF16"), tensor(2, "BF16")]);
        let ld = Loader::new(&rt, &st, "m.");
        let t = ld.linear(&parts, in_f, Precision::Bf16).unwrap();
        let got: Vec<f32> = t.buffer.try_contents_u16().unwrap()[..in_f * total]
            .iter()
            .map(|&b| crate::tensor::bf16_bits_to_f32(b))
            .collect();
        assert_eq!(got, want);
    }

    /// The `*_into` reads fill the device tensor, and are recorded as read; a
    /// tensor under the prefix nobody read is refused by name, and one outside
    /// it is not.
    #[test]
    fn into_reads_and_unread_tensors() {
        let rt = GpuRuntime::new().expect("runtime");
        let vals: Vec<f32> = (0..6).map(|i| i as f32 - 2.5).collect();
        let st = checkpoint(&[
            ("m.table", "BF16", vec![2, 3], vals.clone()),
            ("m.wide", "BF16", vec![3, 2], vals.clone()),
            ("m.extra.bias", "F32", vec![2], vec![0.0, 0.0]),
            ("other.weight", "F32", vec![1], vec![1.0]),
        ]);
        let ld = Loader::new(&rt, &st, "m.");
        let b = rt.alloc_tensor_bf16(&[2, 3]).unwrap();
        ld.bf16_into("table", &[2, 3], &b).unwrap();
        let got: Vec<f32> = b.buffer.try_contents_u16().unwrap()[..6]
            .iter()
            .map(|&x| crate::tensor::bf16_bits_to_f32(x))
            .collect();
        assert_eq!(got, vals);
        let f = rt.alloc_tensor_f32(&[3, 2]).unwrap();
        ld.f32_into("wide", &[3, 2], &f).unwrap();
        assert_eq!(f.buffer.read_f32()[..6], vals[..]);
        assert!(ld.f32_into("wide", &[2, 3], &f).unwrap_err().contains("shape"));

        let err = ld.refuse_unread().unwrap_err();
        assert!(err.contains("m.extra.bias") && !err.contains("other.weight"), "{err}");
        assert!(ld.scalar("extra.bias").unwrap_err().contains("one finite value"));
        ld.refuse_unread().unwrap();
    }
}
