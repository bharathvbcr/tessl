//! From decoded blocks to RGB: the inverse transform of a coefficient store,
//! the per-component upsamplers (`jdsample.c`) and the colour conversion
//! (`jdcolor.c`, plus the CMYK step torchvision adds).

use super::color;
use super::idct;
use super::{invalid, Decoder, JpegColorSpace, JpegError, JpegImage, Store, R};

/// How one component is brought to full resolution (`jinit_upsampler`).
#[derive(Clone, Copy, Debug)]
enum Upsample {
    /// `fullsize_upsample`.
    Full,
    /// `h2v1_fancy_upsample`.
    H2V1Fancy,
    /// `h2v1_upsample`.
    H2V1,
    /// `h1v2_fancy_upsample`.
    H1V2Fancy,
    /// `h2v2_fancy_upsample`.
    H2V2Fancy,
    /// `h2v2_upsample`.
    H2V2,
    /// `int_upsample` with the horizontal and vertical replication factors.
    Int(usize, usize),
}

impl Decoder<'_> {
    pub(crate) fn finish(&mut self) -> R<JpegImage> {
        let store = self
            .store
            .take()
            .ok_or_else(|| JpegError::Invalid("internal: no block store".into()))?;
        let planes = match store {
            Store::Planes(p) => p,
            Store::Coefs(coefs) => self.transform_coefficients(coefs)?,
        };
        let rgb = self.render(&planes)?;
        Ok(JpegImage {
            width: self.width as usize,
            height: self.height as usize,
            rgb,
            info: self.info(),
        })
    }

    /// Inverse-transform every component of a coefficient store to a plane.
    fn transform_coefficients(&mut self, mut coefs: Vec<Vec<i16>>) -> R<Vec<Vec<u8>>> {
        let smooth = self.smoothing_ok();
        self.smoothed = smooth;
        let mut planes = Vec::with_capacity(self.comps.len());
        for ci in 0..self.comps.len() {
            let c = &self.comps[ci];
            let stride = c.blocks_w * 8;
            let mut plane = vec![128u8; stride * c.blocks_h * 8];
            if let Some(quant) = c.quant.as_ref() {
                if smooth {
                    self.smooth_component(ci, &coefs[ci], &mut plane)?;
                } else {
                    for by in 0..c.height_in_blocks {
                        for bx in 0..c.width_in_blocks {
                            let at = (by * c.blocks_w + bx) * 64;
                            let block: &[i16; 64] = coefs[ci][at..at + 64].try_into().expect("64 coefficients");
                            idct::islow(block, quant, &mut plane[by * 8 * stride + bx * 8..], stride);
                        }
                    }
                }
            }
            // The coefficients are not needed past this component.
            coefs[ci] = Vec::new();
            planes.push(plane);
        }
        Ok(planes)
    }

    fn upsampler(&self, ci: usize) -> R<Upsample> {
        let c = &self.comps[ci];
        let (h_in, v_in) = (c.h, c.v);
        let (h_out, v_out) = (self.max_h, self.max_v);
        Ok(if h_in == h_out && v_in == v_out {
            Upsample::Full
        } else if h_in * 2 == h_out && v_in == v_out {
            if c.downsampled_width > 2 {
                Upsample::H2V1Fancy
            } else {
                Upsample::H2V1
            }
        } else if h_in == h_out && v_in * 2 == v_out {
            Upsample::H1V2Fancy
        } else if h_in * 2 == h_out && v_in * 2 == v_out {
            if c.downsampled_width > 2 {
                Upsample::H2V2Fancy
            } else {
                Upsample::H2V2
            }
        } else if h_out % h_in == 0 && v_out % v_in == 0 {
            Upsample::Int(h_out / h_in, v_out / v_in)
        } else {
            return invalid("fractional sampling ratios are not implemented (JERR_FRACT_SAMPLE_NOTIMPL)");
        })
    }

    /// Upsample every component and convert to RGB.
    fn render(&self, planes: &[Vec<u8>]) -> R<Vec<u8>> {
        let (w, h) = (self.width as usize, self.height as usize);
        let space = self.guess_color_space();
        let n = self.comps.len();
        let methods: Vec<Upsample> = (0..n).map(|ci| self.upsampler(ci)).collect::<R<_>>()?;
        let mut rgb = vec![0u8; w * h * 3];
        // One full-width row per component; h2v2 needs two input rows to make one.
        let mut rows: Vec<Vec<u8>> = (0..n).map(|_| vec![0u8; w + 2]).collect();
        let mut tmp = vec![0u16; self.comps.iter().map(|c| c.downsampled_width).max().unwrap_or(0) + 2];
        let conv = color::Tables::new();
        for y in 0..h {
            for ci in 0..n {
                let c = &self.comps[ci];
                let stride = c.blocks_w * 8;
                self.upsample_row(methods[ci], ci, &planes[ci], stride, y, &mut rows[ci], &mut tmp, w);
            }
            let out = &mut rgb[y * w * 3..(y + 1) * w * 3];
            match space {
                JpegColorSpace::Grayscale => color::gray_to_rgb(&rows[0][..w], out),
                JpegColorSpace::Rgb => color::interleave_rgb(&rows[0][..w], &rows[1][..w], &rows[2][..w], out),
                JpegColorSpace::YCbCr => {
                    conv.ycc_to_rgb(&rows[0][..w], &rows[1][..w], &rows[2][..w], out);
                }
                JpegColorSpace::Cmyk => {
                    color::cmyk_to_rgb(&rows[0][..w], &rows[1][..w], &rows[2][..w], &rows[3][..w], out);
                }
                JpegColorSpace::Ycck => {
                    conv.ycck_to_rgb(&rows[0][..w], &rows[1][..w], &rows[2][..w], &rows[3][..w], out);
                }
                JpegColorSpace::Unknown => return invalid("no conversion to RGB for this component count"),
            }
        }
        Ok(rgb)
    }

    #[allow(clippy::too_many_arguments)]
    fn upsample_row(
        &self,
        method: Upsample,
        ci: usize,
        plane: &[u8],
        stride: usize,
        y: usize,
        out: &mut [u8],
        tmp: &mut [u16],
        w: usize,
    ) {
        let c = &self.comps[ci];
        let (dw, dh) = (c.downsampled_width, c.downsampled_height);
        let row = |r: usize| &plane[r * stride..r * stride + dw];
        match method {
            Upsample::Full => out[..w].copy_from_slice(&row(y)[..w]),
            Upsample::H2V1 => {
                let r = row(y);
                for (x, o) in out[..w].iter_mut().enumerate() {
                    *o = r[x / 2];
                }
            }
            Upsample::H2V1Fancy => {
                let r = row(y);
                let mut full = |i: usize, v: u8| {
                    if i < out.len() {
                        out[i] = v;
                    }
                };
                full(0, r[0]);
                full(1, ((u32::from(r[0]) * 3 + u32::from(r[1]) + 2) >> 2) as u8);
                for i in 1..dw - 1 {
                    let v = u32::from(r[i]) * 3;
                    full(2 * i, ((v + u32::from(r[i - 1]) + 1) >> 2) as u8);
                    full(2 * i + 1, ((v + u32::from(r[i + 1]) + 2) >> 2) as u8);
                }
                let l = dw - 1;
                full(2 * l, ((u32::from(r[l]) * 3 + u32::from(r[l - 1]) + 1) >> 2) as u8);
                full(2 * l + 1, r[l]);
            }
            Upsample::H1V2Fancy => {
                let inrow = y / 2;
                let (other, bias) = if y % 2 == 0 {
                    (inrow.saturating_sub(1), 1u32)
                } else {
                    ((inrow + 1).min(dh - 1), 2u32)
                };
                let (a, b) = (row(inrow), row(other));
                for x in 0..w {
                    out[x] = ((u32::from(a[x]) * 3 + u32::from(b[x]) + bias) >> 2) as u8;
                }
            }
            Upsample::H2V2 => {
                let r = row(y / 2);
                for (x, o) in out[..w].iter_mut().enumerate() {
                    *o = r[x / 2];
                }
            }
            Upsample::H2V2Fancy => {
                let inrow = y / 2;
                let other = if y % 2 == 0 {
                    inrow.saturating_sub(1)
                } else {
                    (inrow + 1).min(dh - 1)
                };
                let (a, b) = (row(inrow), row(other));
                for i in 0..dw {
                    tmp[i] = u16::from(a[i]) * 3 + u16::from(b[i]);
                }
                let cs = |i: usize| u32::from(tmp[i]);
                let mut put = |i: usize, v: u32| {
                    if i < out.len() {
                        out[i] = v as u8;
                    }
                };
                put(0, (cs(0) * 4 + 8) >> 4);
                put(1, (cs(0) * 3 + cs(1) + 7) >> 4);
                for i in 1..dw - 1 {
                    put(2 * i, (cs(i) * 3 + cs(i - 1) + 8) >> 4);
                    put(2 * i + 1, (cs(i) * 3 + cs(i + 1) + 7) >> 4);
                }
                let l = dw - 1;
                put(2 * l, (cs(l) * 3 + cs(l - 1) + 8) >> 4);
                put(2 * l + 1, (cs(l) * 4 + 7) >> 4);
            }
            Upsample::Int(hx, vx) => {
                let r = row(y / vx);
                for (x, o) in out[..w].iter_mut().enumerate() {
                    *o = r[x / hx];
                }
            }
        }
    }
}
