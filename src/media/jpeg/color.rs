//! Colour conversion: libjpeg's `ycc_rgb_convert` and `ycck_cmyk_convert`
//! (`jdcolor.c`) with their 16-bit fixed-point tables, and the CMYK to RGB step
//! torchvision applies on top.

const SCALEBITS: i32 = 16;
const ONE_HALF: i32 = 1 << (SCALEBITS - 1);

/// `FIX(x)`: `x` as a 16-bit fixed-point constant, rounded.
const fn fix(x: f64) -> i32 {
    (x * (1i64 << SCALEBITS) as f64 + 0.5) as i32
}

/// The `Cr_r_tab`, `Cb_b_tab`, `Cr_g_tab` and `Cb_g_tab` of `build_ycc_rgb_table`.
pub(crate) struct Tables {
    cr_r: [i32; 256],
    cb_b: [i32; 256],
    cr_g: [i32; 256],
    cb_g: [i32; 256],
}

impl Tables {
    pub fn new() -> Tables {
        let mut t = Tables {
            cr_r: [0; 256],
            cb_b: [0; 256],
            cr_g: [0; 256],
            cb_g: [0; 256],
        };
        for i in 0..256usize {
            let x = i as i32 - 128;
            t.cr_r[i] = (fix(1.40200) * x + ONE_HALF) >> SCALEBITS;
            t.cb_b[i] = (fix(1.77200) * x + ONE_HALF) >> SCALEBITS;
            t.cr_g[i] = -fix(0.71414) * x;
            t.cb_g[i] = -fix(0.34414) * x + ONE_HALF;
        }
        t
    }

    /// The R, G and B of one YCbCr triple.
    #[inline]
    fn rgb(&self, y: u8, cb: u8, cr: u8) -> [u8; 3] {
        let y = i32::from(y);
        let (cb, cr) = (usize::from(cb), usize::from(cr));
        [
            clamp(y + self.cr_r[cr]),
            clamp(y + ((self.cb_g[cb] + self.cr_g[cr]) >> SCALEBITS)),
            clamp(y + self.cb_b[cb]),
        ]
    }

    pub fn ycc_to_rgb(&self, y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8]) {
        for (px, ((&y, &cb), &cr)) in out.chunks_exact_mut(3).zip(y.iter().zip(cb).zip(cr)) {
            px.copy_from_slice(&self.rgb(y, cb, cr));
        }
    }

    /// YCCK to CMYK (`ycck_cmyk_convert`), then to RGB.
    pub fn ycck_to_rgb(&self, y: &[u8], cb: &[u8], cr: &[u8], k: &[u8], out: &mut [u8]) {
        for (i, px) in out.chunks_exact_mut(3).enumerate() {
            let [r, g, b] = self.rgb(y[i], cb[i], cr[i]);
            let (c, m, ye) = (255 - r, 255 - g, 255 - b);
            let kk = u32::from(k[i]);
            px[0] = cmy_k(kk, c);
            px[1] = cmy_k(kk, m);
            px[2] = cmy_k(kk, ye);
        }
    }
}

#[inline]
fn clamp(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// torchvision's `clamped_cmyk_rgb_convert`: `k * cmy / 255`, rounded the way
/// Pillow does.
#[inline]
fn cmy_k(k: u32, cmy: u8) -> u8 {
    let v = k * u32::from(cmy) + 128;
    (((v >> 8) + v) >> 8).min(255) as u8
}

pub(crate) fn gray_to_rgb(y: &[u8], out: &mut [u8]) {
    for (px, &v) in out.chunks_exact_mut(3).zip(y) {
        px.fill(v);
    }
}

pub(crate) fn interleave_rgb(r: &[u8], g: &[u8], b: &[u8], out: &mut [u8]) {
    for (i, px) in out.chunks_exact_mut(3).enumerate() {
        px[0] = r[i];
        px[1] = g[i];
        px[2] = b[i];
    }
}

pub(crate) fn cmyk_to_rgb(c: &[u8], m: &[u8], y: &[u8], k: &[u8], out: &mut [u8]) {
    for (i, px) in out.chunks_exact_mut(3).enumerate() {
        let kk = u32::from(k[i]);
        px[0] = cmy_k(kk, c[i]);
        px[1] = cmy_k(kk, m[i]);
        px[2] = cmy_k(kk, y[i]);
    }
}
