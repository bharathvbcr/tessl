//! Reading a checkpoint's tensors onto the device: the one loader
//! [`crate::qwen35_model::Qwen35Model`] and
//! [`crate::embedgemma2::EmbedGemma2Model`] share.
//!
//! Each tensor is named `{prefix}{rest}` and its shape is checked against
//! what the model expects. Nothing is held on the host longer than it takes
//! to reach the device: a projection's parts are placed into the packed
//! operand one at a time as they are read ([`Loader::linear`], through one
//! scratch buffer the size of the largest part, reused for every part rather
//! than faulting in fresh pages for each), and a table
//! is read straight into its device tensor ([`Loader::f32_into`],
//! [`Loader::bf16_into`]). Every name read is recorded, so a model that
//! implements everything under its prefix can refuse a checkpoint carrying a
//! tensor it would silently ignore ([`Loader::refuse_unread`]).
//!
//! The two models kept near-copies of this (`name`/`f32`/`norm`/`linear`)
//! until EmbedGemma2's copy fell behind: it held every part of a projection
//! and a packed host copy at once, and read its embedding through a `Vec`.
//! One owner keeps that from recurring. `bert` reads through this loader too
//! (with an empty prefix, so every tensor in the file is its own), adding only
//! what is BERT's: [`Loader::f32_finite`] and `Cold` weight buffers.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::sync::Arc;

use crate::qwen35;
use crate::qwen35_model::Precision;
use crate::qwen35_q4::{self, NormConvention, Q4Weight};
use crate::runtime::GpuRuntime;
use crate::safetensors::TensorSource;
use crate::tensor::{GpuBuffer, Tensor};

/// A whole-tensor read's `(shape, elements)` with the shape checked against
/// the one the model expects: the single owner of the loader's shape error.
fn shape_checked<T>(name: &str, shape: &[usize], read: Result<(Vec<usize>, Vec<T>), String>) -> Result<Vec<T>, String> {
    let (got, data) = read?;
    if got != shape {
        return Err(format!("{name}: shape {got:?}, expected {shape:?}"));
    }
    Ok(data)
}

pub(crate) struct Loader<'a> {
    st: &'a dyn TensorSource,
    prefix: &'a str,
    rt: &'a Arc<GpuRuntime>,
    /// Every tensor name read so far, for [`Self::refuse_unread`].
    read: RefCell<BTreeSet<String>>,
    /// [`Self::linear`]'s staging for one part, grown to the largest part
    /// and reused: a fresh `Vec` per part paid page faults and kernel zeroing
    /// for every one (about 4 GB over a 2B bf16 load).
    scratch_bf16: RefCell<Vec<u16>>,
    scratch_f32: RefCell<Vec<f32>>,
}

impl<'a> Loader<'a> {
    pub(crate) fn new(rt: &'a Arc<GpuRuntime>, st: &'a dyn TensorSource, prefix: &'a str) -> Self {
        Self {
            st,
            prefix,
            rt,
            read: RefCell::new(BTreeSet::new()),
            scratch_bf16: RefCell::new(Vec::new()),
            scratch_f32: RefCell::new(Vec::new()),
        }
    }

    /// `rest`'s full name, recorded as read.
    fn take(&self, rest: &str) -> String {
        self.take_root(&self.prefixed(rest))
    }

    /// `rest` under this loader's prefix.
    pub(crate) fn prefixed(&self, rest: &str) -> String {
        format!("{}{rest}", self.prefix)
    }

    /// `name` as the checkpoint spells it (no prefix), recorded as read.
    fn take_root(&self, name: &str) -> String {
        self.read.borrow_mut().insert(name.to_string());
        name.to_string()
    }

    /// Whether the checkpoint holds a tensor named `name` exactly (no
    /// prefix): for a tensor that lives at the root, beside the tower, such
    /// as an untied model's `lm_head.weight`.
    pub(crate) fn has_root(&self, name: &str) -> bool {
        self.st.names().any(|n| n == name)
    }

    /// [`Self::bf16`] for a tensor named `name` as the checkpoint spells it.
    pub(crate) fn bf16_root(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>, String> {
        let name = self.take_root(name);
        shape_checked(&name, shape, self.st.read_bf16_bits(&name))
    }

    /// [`Self::f32_into`] for a tensor named `name` as the checkpoint spells it.
    pub(crate) fn f32_into_root(&self, name: &str, shape: &[usize], t: &Tensor) -> Result<(), String> {
        let name = self.take_root(name);
        self.st.read_f32_into(&name, shape, &mut t.buffer.try_contents_f32()?)
    }

    /// [`Self::bf16_into`] for a tensor named `name` as the checkpoint spells it.
    pub(crate) fn bf16_into_root(&self, name: &str, shape: &[usize], t: &Tensor) -> Result<(), String> {
        let name = self.take_root(name);
        self.st
            .read_bf16_bits_into(&name, shape, &mut t.buffer.try_contents_u16()?)
    }

    /// Check tensor `name` (as the checkpoint spells it) has `shape`, from the
    /// header alone: no data is read and nothing is recorded as read. Every
    /// loader that allocates a tensor the *config* sizes calls this first, so a
    /// hostile config over a small file is refused before the allocation, not
    /// by the device running out of memory. The message is the one the reads
    /// give for the same mismatch.
    pub(crate) fn expect_shape(&self, name: &str, shape: &[usize]) -> Result<(), String> {
        let got = self.st.tensor_info(name)?.shape;
        if got != shape {
            return Err(format!("{name}: shape {got:?}, expected {shape:?}"));
        }
        Ok(())
    }

    /// A tensor widened to f32, with its shape checked.
    pub(crate) fn f32(&self, rest: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let name = self.take(rest);
        shape_checked(&name, shape, TensorSource::read_f32(self.st, &name))
    }

    /// [`Self::f32`], refusing a non-finite element by index.
    pub(crate) fn f32_finite(&self, rest: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let data = self.f32(rest, shape)?;
        if let Some(i) = data.iter().position(|v| !v.is_finite()) {
            return Err(format!("{}{rest}: element {i} is {}, not finite", self.prefix, data[i]));
        }
        Ok(data)
    }

    /// A bf16 tensor's bit patterns, with its shape checked.
    pub(crate) fn bf16(&self, rest: &str, shape: &[usize]) -> Result<Vec<u16>, String> {
        self.bf16_root(&self.prefixed(rest), shape)
    }

    /// A tensor widened to f32 straight into `t`'s shared storage, with no
    /// host copy. `t` must be an f32 tensor of exactly `shape`'s elements.
    pub(crate) fn f32_into(&self, rest: &str, shape: &[usize], t: &Tensor) -> Result<(), String> {
        self.f32_into_root(&self.prefixed(rest), shape, t)
    }

    /// A bf16 tensor's bits straight into `t`'s shared storage, with no host
    /// copy. `t` must be a bf16 tensor of exactly `shape`'s elements.
    pub(crate) fn bf16_into(&self, rest: &str, shape: &[usize], t: &Tensor) -> Result<(), String> {
        self.bf16_into_root(&self.prefixed(rest), shape, t)
    }

    /// One finite value (a `[1]` or scalar tensor).
    pub(crate) fn scalar(&self, rest: &str) -> Result<f32, String> {
        let name = self.take(rest);
        let (_, data) = TensorSource::read_f32(self.st, &name)?;
        match data.as_slice() {
            [s] if s.is_finite() => Ok(*s),
            other => Err(format!("{name}: expected one finite value, got {other:?}")),
        }
    }

    /// `data` in a weight buffer (`BufferKind::Hot`: it lives as long as the
    /// model, not as a step temporary).
    pub(crate) fn buf(&self, data: &[f32]) -> Result<GpuBuffer, String> {
        let b = self.rt.alloc_buffer_hot(data.len().max(1) * 4)?;
        b.try_write_f32(data).map_err(|e| format!("Loader::buf: {e}"))?;
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
        // Headers first: the operand below is sized by the config.
        for &(rest, out) in parts {
            self.expect_shape(&self.prefixed(rest), &[out, in_features])?;
        }
        let mut col0 = 0;
        match precision {
            Precision::Bf16 => {
                let t = self.rt.alloc_tensor_bf16_hot(&[in_features, total])?;
                let mut scratch = self.scratch_bf16.borrow_mut();
                for &(rest, out) in parts {
                    let part = staged(&mut scratch, out, in_features)?;
                    let name = self.take(rest);
                    self.st.read_bf16_bits_into(&name, &[out, in_features], part)?;
                    qwen35::place_linear_part(&mut t.buffer.try_contents_u16()?, total, col0, part, out, in_features)?;
                    col0 += out;
                }
                Ok(t)
            }
            Precision::F32 => {
                let t = self.rt.alloc_tensor_f32_hot(&[in_features, total])?;
                let mut scratch = self.scratch_f32.borrow_mut();
                for &(rest, out) in parts {
                    let part = staged(&mut scratch, out, in_features)?;
                    let name = self.take(rest);
                    self.st.read_f32_into(&name, &[out, in_features], part)?;
                    qwen35::place_linear_part(&mut t.buffer.try_contents_f32()?, total, col0, part, out, in_features)?;
                    col0 += out;
                }
                Ok(t)
            }
        }
    }

    /// Quantized `nn.Linear` weights, MLX affine 4-bit: each part `rest`
    /// (`X.weight`, with `X.scales` and `X.biases` beside it) is
    /// `[out_i, in_features]`; the parts' rows are concatenated into one
    /// [`Q4Weight`] of `sum(out_i)` rows, the quantized form of the window
    /// [`Self::linear`] packs. See [`Self::mlx_q4_named`].
    pub(crate) fn mlx_q4(&self, parts: &[(&str, usize)], in_features: usize) -> Result<Q4Weight, String> {
        let named: Vec<(String, usize)> = parts.iter().map(|&(r, o)| (self.prefixed(r), o)).collect();
        self.mlx_q4_named(&named, in_features)
    }

    /// [`Self::mlx_q4`] for one matrix named as the checkpoint spells it
    /// (no prefix): a table or head that lives beside the tower.
    pub(crate) fn mlx_q4_root(&self, name: &str, rows: usize, cols: usize) -> Result<Q4Weight, String> {
        self.mlx_q4_named(&[(name.to_string(), rows)], cols)
    }

    /// Read `parts` (full names, rows) into one bank. Each part's three
    /// tensors are checked against their headers ([`qwen35_q4::check_part`])
    /// before anything is read; the packed words go straight into the bank's
    /// device storage, and the scales and biases (one part's at a time on the
    /// host, `rows * cols / 64` bf16 each) are interleaved into the bank's
    /// `bfloat2` pairs. All three names are recorded as read.
    fn mlx_q4_named(&self, parts: &[(String, usize)], cols: usize) -> Result<Q4Weight, String> {
        let total = parts
            .iter()
            .try_fold(0usize, |acc, (_, o)| acc.checked_add(*o))
            .ok_or("mlx_q4: output widths overflow usize")?;
        let group = qwen35_q4::MLX_Q4_GROUP as usize;
        let groups = cols / group;
        // Every part's three headers are checked against the config's shape
        // before anything is allocated: the config sizes the bank, and a
        // hostile one over a tiny file must be refused here, not by the device
        // running out of memory.
        let mut names = Vec::with_capacity(parts.len());
        for (name, rows) in parts {
            let (sn, bn) = qwen35_q4::sidecar_names(name)
                .ok_or_else(|| format!("{name}: a quantized weight is named X.weight"))?;
            let info = self.st.tensor_info(name)?;
            let (s_info, b_info) = (self.st.tensor_info(&sn).ok(), self.st.tensor_info(&bn).ok());
            qwen35_q4::check_part(name, &info, s_info.as_ref(), b_info.as_ref(), *rows, cols)?;
            names.push((sn, bn));
        }
        let w = Q4Weight::alloc(self.rt, total, cols)?;
        let words_per_row = cols / 8;
        let mut row0 = 0usize;
        for ((name, rows), (sn, bn)) in parts.iter().zip(names) {
            self.take_root(name);
            let (sn, bn) = (self.take_root(&sn), self.take_root(&bn));
            {
                let mut words = w.packed().try_contents_u32()?;
                self.st.read_u32_into(
                    name,
                    &[*rows, words_per_row],
                    &mut words[row0 * words_per_row..(row0 + rows) * words_per_row],
                )?;
            }
            let n = rows * groups;
            let (mut scales, mut biases) = (vec![0u16; n], vec![0u16; n]);
            self.st.read_bf16_bits_into(&sn, &[*rows, groups], &mut scales)?;
            self.st.read_bf16_bits_into(&bn, &[*rows, groups], &mut biases)?;
            // A bf16 with every exponent bit set is an infinity or a NaN. One
            // in a scale or bias would turn every logit it touches into NaN.
            let nonfinite = |bits: u16| bits & 0x7f80 == 0x7f80;
            for (tensor, v) in [(&sn, &scales), (&bn, &biases)] {
                if let Some(i) = v.iter().position(|&b| nonfinite(b)) {
                    return Err(format!(
                        "{tensor}: element {i} (row {}, group {}) is {}, not finite",
                        i / groups,
                        i % groups,
                        crate::tensor::bf16_bits_to_f32(v[i])
                    ));
                }
            }
            let mut pairs = w.scales_biases().try_contents_u16()?;
            let dst = &mut pairs[2 * row0 * groups..2 * (row0 + rows) * groups];
            for (i, (s, b)) in scales.iter().zip(&biases).enumerate() {
                dst[2 * i] = *s;
                dst[2 * i + 1] = *b;
            }
            row0 += rows;
        }
        Ok(w)
    }

    /// A `[dim]` weight as a kernel that adds 1 itself wants it, from a
    /// checkpoint stored in `convention` (see [`NormConvention`]).
    pub(crate) fn unit_norm(&self, rest: &str, dim: usize, convention: NormConvention) -> Result<GpuBuffer, String> {
        let stored = self.f32_finite(rest, &[dim])?;
        let w: Vec<f32> = stored.iter().map(|&v| convention.unshift(v)).collect();
        self.buf(&w)
    }

    /// Tensors outside the prefix that no read touched, sorted.
    pub(crate) fn unread_outside(&self) -> Vec<&str> {
        let read = self.read.borrow();
        self.st
            .names()
            .filter(|n| !n.starts_with(self.prefix) && !read.contains(*n))
            .collect()
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

/// The first `out * in_features` elements of `scratch`, grown if needed.
/// Only growth is zero-filled; the read overwrites the whole slice.
fn staged<T: Copy + Default>(scratch: &mut Vec<T>, out: usize, in_features: usize) -> Result<&mut [T], String> {
    let n = out
        .checked_mul(in_features)
        .ok_or("linear: part element count overflows usize")?;
    if scratch.len() < n {
        scratch.resize(n, T::default());
    }
    Ok(&mut scratch[..n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::SafeTensors;

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

    /// The f32/bf16 readers (host `Vec` and straight-into-tensor, prefixed and
    /// root) each return the data when the shape matches and the same shape
    /// error, naming the full tensor name, when it does not.
    #[test]
    fn readers_agree_on_data_and_shape_errors() {
        let rt = GpuRuntime::new().expect("runtime");
        let st = checkpoint(&[
            ("m.a", "F32", vec![2], vec![1.0, 2.0]),
            ("m.b", "BF16", vec![2], vec![3.0, 4.0]),
        ]);
        let ld = Loader::new(&rt, &st, "m.");
        assert_eq!(ld.f32("a", &[2]).unwrap(), vec![1.0, 2.0]);
        assert_eq!(ld.bf16("b", &[2]).unwrap(), vec![0x4040, 0x4080]);
        assert_eq!(ld.bf16_root("m.b", &[2]).unwrap(), vec![0x4040, 0x4080]);
        assert_eq!(ld.f32("a", &[3]).unwrap_err(), "m.a: shape [2], expected [3]");
        assert_eq!(ld.bf16("b", &[3]).unwrap_err(), "m.b: shape [2], expected [3]");
        assert_eq!(ld.bf16_root("m.b", &[3]).unwrap_err(), "m.b: shape [2], expected [3]");
        let ta = rt.alloc_tensor_f32(&[2]).unwrap();
        ld.f32_into("a", &[2], &ta).unwrap();
        assert_eq!(ta.buffer.read_f32(), vec![1.0, 2.0]);
        assert_eq!(ld.f32_into("a", &[3], &ta).unwrap_err(), "m.a: shape [2], expected [3]");
        assert_eq!(
            ld.f32_into_root("m.a", &[3], &ta).unwrap_err(),
            "m.a: shape [2], expected [3]"
        );
        let tb = rt.alloc_tensor_bf16(&[2]).unwrap();
        ld.bf16_into("b", &[2], &tb).unwrap();
        assert_eq!(tb.buffer.contents_u16().to_vec(), vec![0x4040, 0x4080]);
        assert_eq!(
            ld.bf16_into("b", &[3], &tb).unwrap_err(),
            "m.b: shape [2], expected [3]"
        );
        assert_eq!(
            ld.bf16_into_root("m.b", &[3], &tb).unwrap_err(),
            "m.b: shape [2], expected [3]"
        );
        // Every name read, success or not, is recorded.
        assert!(ld.refuse_unread().is_ok());
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
