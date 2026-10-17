//! Sample-rate conversion and channel downmix that reproduce the `ffmpeg`
//! binary's defaults (`ffmpeg -i in.wav -ac 1 -ar R -f f32le out.raw`) bit for
//! bit on AArch64.
//!
//! This module was written without access to FFmpeg's source. Every rule below
//! was derived from FFmpeg's published option documentation, from the
//! windowed-sinc literature, and from black-box experiments on the `ffmpeg`
//! binary; `docs/media_resample_cleanroom.md` records which experiment fixed
//! which rule. In short:
//!
//! * **Filter bank.** A Kaiser-windowed sinc (`beta` 9). The cutoff is
//!   `min(1, 0.97 * out / in)` of the input Nyquist rate, the window is
//!   `taps / 2` samples wide on each side, and `taps` is
//!   `ceil(32 / cutoff)` rounded up to even. One phase per reduced output
//!   rate numerator, at most 1024; every phase is scaled by the one factor that
//!   makes phase 0 sum to 1. All coefficients are rounded to `f32`.
//! * **Positions.** Output `m` sits at input position `m * in / out`, kept as
//!   exact integers: the integer part picks the window (which starts
//!   `taps / 2 - 1` samples before it) and the remainder picks the phase.
//! * **Interpolation.** When the reduced output rate exceeds 1024 phases the
//!   bank has 1024 of them and each output blends two neighbouring phases.
//! * **Edges.** Before the first sample the signal is mirrored without
//!   repeating the edge sample; after the last it is mirrored about the
//!   half-sample past it, for as many samples as half of what the resampler
//!   still holds (see [`SwrResampler::finish`]).
//! * **Summation.** The float sums are not associative, so the order is part
//!   of the contract: four lanes of fused multiply-adds reduced pairwise for
//!   the exact banks, a plain fused chain for the interpolating ones.
//! * **Channels.** Each channel is resampled, then the channels are mixed with
//!   the default matrix for that channel count ([`SwrMix`]).
//!
//! The bounds `MAX_RATE`, `MAX_SWR_CHANNELS`, `MAX_TAPS` and `MAX_BANK_FLOATS`
//! are tessl's own limits, chosen to refuse inputs that would need absurd
//! memory; they say nothing about FFmpeg.

use std::f64::consts::PI;
use std::sync::Arc;

use super::MediaError;

/// The highest sample rate accepted (tessl's bound).
pub const MAX_RATE: u32 = 1_000_000;
/// The most channels [`SwrMix`] and [`SwrConverter`] take: FFmpeg's default
/// layouts exist for 1 to 8 channels here.
pub const MAX_SWR_CHANNELS: usize = 8;
/// The longest filter built (tessl's bound; 33 million taps would be needed for
/// 1 MHz down to 1 Hz).
const MAX_TAPS: usize = 1 << 15;
/// The largest filter bank built, in `f32` coefficients (tessl's bound).
const MAX_BANK_FLOATS: usize = 1 << 24;

/// The documented default of the `filter_size` option.
const FILTER_SIZE: f64 = 32.0;
/// The documented default of the `cutoff` option for the native resampler.
const CUTOFF: f64 = 0.97;
/// The documented default of the `kaiser_beta` option.
const KAISER_BETA: f64 = 9.0;
/// `1 << phase_shift` with the documented default `phase_shift` of 10.
const MAX_PHASES: usize = 1 << 10;

/// What the converter does with the samples it is given.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    /// Equal rates: nothing is filtered, samples pass through.
    Identity,
    /// One bank row per reduced output-rate numerator (at most 1024).
    Exact,
    /// 1024 rows and a linear blend between neighbouring rows.
    Interpolated,
}

/// A streaming single-channel sample-rate converter.
///
/// The output does not depend on how the input is split across [`push`]
/// calls. Call [`finish`] once the input is complete: with the reflection at
/// the end (and the rule that nothing is produced until more than `taps()`
/// samples have arrived) the output of a clip is a function of the whole clip.
///
/// [`push`]: SwrResampler::push
/// [`finish`]: SwrResampler::finish
pub struct SwrResampler {
    mode: Mode,
    taps: usize,
    phases: usize,
    /// Reduced input rate: output `m` is at input position `m * num / den`.
    num: u64,
    /// Reduced output rate.
    den: u64,
    /// `rows * taps` coefficients, shared by the channels of a converter.
    bank: Arc<[f32]>,
    /// `buf[k]` is input sample `base + k`.
    buf: Vec<f32>,
    base: u64,
    /// Input samples received since the last [`finish`](Self::finish).
    received: u64,
    /// Index of the next output.
    next: u64,
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// The modified Bessel function of the first kind and order 0, by its power
/// series (`sum_k ((x/2)^k / k!)^2`), which converges for every `x` used here
/// to the last bit of an `f64` within 40 terms.
fn bessel_i0(x: f64) -> f64 {
    let q = x * x / 4.0;
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    let mut k = 1.0f64;
    while term > sum * 1e-17 {
        term *= q / (k * k);
        sum += term;
        k += 1.0;
    }
    sum
}

/// The filter length for a pair of rates: `ceil(32 / cutoff)` where the cutoff
/// is `min(1, 0.97 * out / in)`, rounded up to an even number.
fn tap_count(in_rate: u32, out_rate: u32) -> f64 {
    let cutoff = (CUTOFF * f64::from(out_rate) / f64::from(in_rate)).min(1.0);
    let t = (FILTER_SIZE / cutoff).ceil();
    t + (t % 2.0)
}

impl SwrResampler {
    /// A converter from `in_rate` to `out_rate` (both in `1..=MAX_RATE`).
    ///
    /// Building the filter bank costs `taps * phases` coefficient
    /// evaluations; it is refused (`MediaError::Invalid`) when the bank would
    /// exceed tessl's size bounds.
    pub fn new(in_rate: u32, out_rate: u32) -> Result<Self, MediaError> {
        if in_rate == 0 || in_rate > MAX_RATE || out_rate == 0 || out_rate > MAX_RATE {
            return Err(MediaError::Invalid {
                what: "resample",
                detail: format!("rates must be in 1..={MAX_RATE} Hz, got {in_rate} -> {out_rate}"),
            });
        }
        let g = gcd(u64::from(in_rate), u64::from(out_rate));
        let (num, den) = (u64::from(in_rate) / g, u64::from(out_rate) / g);
        let taps_f = tap_count(in_rate, out_rate);
        if taps_f > MAX_TAPS as f64 {
            return Err(MediaError::Invalid {
                what: "resample",
                detail: format!("{in_rate} -> {out_rate} Hz needs a {taps_f}-tap filter, over the {MAX_TAPS} bound"),
            });
        }
        let taps = taps_f as usize;
        let phases = den.min(MAX_PHASES as u64) as usize;
        let mode = if in_rate == out_rate {
            Mode::Identity
        } else if den as usize <= MAX_PHASES {
            Mode::Exact
        } else {
            Mode::Interpolated
        };
        let rows = phases + usize::from(mode == Mode::Interpolated);
        if mode != Mode::Identity && rows.saturating_mul(taps) > MAX_BANK_FLOATS {
            return Err(MediaError::Invalid {
                what: "resample",
                detail: format!("{in_rate} -> {out_rate} Hz needs a {rows} x {taps} filter bank, over the size bound"),
            });
        }
        let bank: Arc<[f32]> = if mode == Mode::Identity {
            Arc::from(Vec::new())
        } else {
            let cutoff = (CUTOFF * f64::from(out_rate) / f64::from(in_rate)).min(1.0);
            Arc::from(design_bank(taps, phases, cutoff, mode == Mode::Interpolated))
        };
        Ok(Self {
            mode,
            taps,
            phases,
            num,
            den,
            bank,
            buf: Vec::new(),
            base: 0,
            received: 0,
            next: 0,
        })
    }

    /// A converter with the same rates and filter bank (shared, not rebuilt)
    /// and no input yet.
    fn fresh(&self) -> Self {
        Self {
            mode: self.mode,
            taps: self.taps,
            phases: self.phases,
            num: self.num,
            den: self.den,
            bank: Arc::clone(&self.bank),
            buf: Vec::new(),
            base: 0,
            received: 0,
            next: 0,
        }
    }

    /// The filter length in taps.
    pub fn taps(&self) -> usize {
        self.taps
    }

    /// The number of filter phases: the reduced output rate, at most 1024.
    pub fn phases(&self) -> usize {
        self.phases
    }

    /// Input position of output `m` as `(whole samples, remainder over den)`.
    fn position(&self, m: u64) -> (u64, u64) {
        let t = m * self.num;
        (t / self.den, t % self.den)
    }

    /// Append the output of every window that is complete once `limit` input
    /// samples (real or reflected) exist, starting at output `self.next`.
    fn emit_through(&mut self, limit: u64, out: &mut Vec<f32>) {
        let half = (self.taps / 2) as u64;
        let centre = half - 1;
        let n = self.received;
        let mut window = vec![0f32; self.taps];
        loop {
            let (idx0, rem) = self.position(self.next);
            if idx0 + half + 1 > limit {
                break;
            }
            let start = idx0 as i64 - centre as i64;
            let real = start >= self.base as i64 && (start as u64 + self.taps as u64) <= n;
            let samples: &[f32] = if real {
                let at = (start as u64 - self.base) as usize;
                &self.buf[at..at + self.taps]
            } else {
                for (k, w) in window.iter_mut().enumerate() {
                    *w = self.sample_ext(start + k as i64);
                }
                &window
            };
            out.push(self.output(samples, rem));
            self.next += 1;
        }
    }

    /// Input sample `i` of the clip as the converter sees it: before the start
    /// the clip is mirrored without repeating sample 0, after the end it is
    /// mirrored with the last sample repeated.
    fn sample_ext(&self, mut i: i64) -> f32 {
        let n = self.received as i64;
        while i < 0 || i >= n {
            i = if i < 0 { -i } else { 2 * n - 1 - i };
        }
        self.buf[(i as u64 - self.base) as usize]
    }

    /// One output from the window `x` (exactly `taps` samples) at remainder
    /// `rem` (over `den`) of the position.
    fn output(&self, x: &[f32], rem: u64) -> f32 {
        let taps = self.taps;
        match self.mode {
            Mode::Exact => dot(&self.bank[rem as usize * taps..][..taps], x),
            Mode::Interpolated => {
                let scaled = rem * MAX_PHASES as u64;
                let row = (scaled / self.den) as usize;
                let frac = (scaled % self.den) as f64 / self.den as f64;
                let v0 = dot_chain(&self.bank[row * taps..][..taps], x);
                let v1 = dot_chain(&self.bank[(row + 1) * taps..][..taps], x);
                let diff = v1 - v0;
                (f64::from(v0) + frac * f64::from(diff)) as f32
            }
            Mode::Identity => unreachable!("identity converters never filter"),
        }
    }

    /// Feed input samples; append every output that is now determined.
    ///
    /// Nothing is produced until more than `taps()` samples have arrived in
    /// all: shorter clips are handled entirely by [`finish`](Self::finish).
    pub fn push(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if self.mode == Mode::Identity {
            out.extend_from_slice(input);
            return;
        }
        self.buf.extend_from_slice(input);
        self.received += input.len() as u64;
        if self.received > self.taps as u64 {
            self.emit_through(self.received, out);
            // Drop what no later window reaches.
            let (idx0, _) = self.position(self.next);
            let keep_from = idx0.saturating_sub(self.taps as u64 / 2 - 1);
            if keep_from > self.base {
                self.buf.drain(..(keep_from - self.base) as usize);
                self.base = keep_from;
            }
        }
    }

    /// Flush the tail and start a new clip.
    ///
    /// The resampler holds `B` input samples that no emitted window has left
    /// behind (all of them for a clip of `taps()` samples or fewer). The clip
    /// is extended by `ceil(B / 2)` samples mirrored about its end, the
    /// windows that fit in the extended clip are emitted, and nothing is
    /// emitted at all unless the extended clip is longer than `taps()`.
    pub fn finish(&mut self, out: &mut Vec<f32>) {
        if self.mode != Mode::Identity && self.received > 0 {
            let n = self.received;
            let main = n > self.taps as u64;
            let consumed = if main { self.base } else { 0 };
            let held = n - consumed;
            let total = n + held.div_ceil(2);
            if total > self.taps as u64 {
                self.emit_through(total, out);
            }
        }
        self.buf.clear();
        self.base = 0;
        self.received = 0;
        self.next = 0;
    }
}

/// The coefficients of every phase, `rows * taps` of them, with `rows` equal
/// to `phases` plus one for an interpolating bank.
///
/// Phase `p` is the sinc of cutoff `cutoff` sampled at the distances
/// `d = (j - (taps / 2 - 1)) - p / phases` from the output position, times a
/// Kaiser window of half-width `taps / 2`; all phases share the scale that
/// makes phase 0 sum to 1. For a cutoff of exactly 1 the sinc at a nonzero
/// whole distance is exactly 0. The extra row of an interpolating bank (phase
/// `phases`, one whole sample on) is phase 0 moved one tap along, with a zero
/// entering.
fn design_bank(taps: usize, phases: usize, cutoff: f64, extra_row: bool) -> Vec<f32> {
    let centre = (taps / 2 - 1) as f64;
    let half = taps as f64 / 2.0;
    let i0_beta = bessel_i0(KAISER_BETA);
    let mut rows = vec![0f64; phases * taps];
    for p in 0..phases {
        for j in 0..taps {
            let d = (j as f64 - centre) - p as f64 / phases as f64;
            let sinc = if d == 0.0 {
                cutoff
            } else if cutoff == 1.0 && d == d.trunc() {
                0.0
            } else {
                (PI * cutoff * d).sin() / (PI * d)
            };
            let arg = (1.0 - (d / half).powi(2)).max(0.0);
            let window = bessel_i0(KAISER_BETA * arg.sqrt()) / i0_beta;
            rows[p * taps + j] = sinc * window;
        }
    }
    let scale: f64 = rows[..taps].iter().sum();
    let mut bank: Vec<f32> = rows.iter().map(|v| (v / scale) as f32).collect();
    if extra_row {
        bank.push(0.0);
        for j in 1..taps {
            bank.push(bank[j - 1]);
        }
    }
    bank
}

/// The dot product of a filter row and a window, in the order the reference
/// sums: four lanes (tap index mod 4) accumulate by fused multiply-add over the
/// first `8 * floor(taps / 8)` taps in order, the lanes are added as
/// `(l0 + l1) + (l2 + l3)`, and each remaining tap is fused onto that sum.
fn dot(c: &[f32], x: &[f32]) -> f32 {
    let main = c.len() & !7;
    let mut acc = [0f32; 4];
    for (cb, xb) in c[..main].chunks_exact(8).zip(x[..main].chunks_exact(8)) {
        let ((c0, c1), (x0, x1)) = (cb.split_at(4), xb.split_at(4));
        for ((a, cj), xj) in acc.iter_mut().zip(c0).zip(x0) {
            *a = cj.mul_add(*xj, *a);
        }
        for ((a, cj), xj) in acc.iter_mut().zip(c1).zip(x1) {
            *a = cj.mul_add(*xj, *a);
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (cj, xj) in c[main..].iter().zip(&x[main..]) {
        s = cj.mul_add(*xj, s);
    }
    s
}

/// The dot product as one sequential chain of fused multiply-adds, used for the
/// two neighbouring phases of an interpolating bank.
fn dot_chain(c: &[f32], x: &[f32]) -> f32 {
    let mut s = 0f32;
    for (cj, xj) in c.iter().zip(x) {
        s = cj.mul_add(*xj, s);
    }
    s
}

/// `resample` for one channel: the whole clip at once.
pub fn resample(input: &[f32], in_rate: u32, out_rate: u32) -> Result<Vec<f32>, MediaError> {
    let mut r = SwrResampler::new(in_rate, out_rate)?;
    let mut out = Vec::with_capacity((input.len() as u64 * u64::from(out_rate) / u64::from(in_rate)) as usize + 1);
    r.push(input, &mut out);
    r.finish(&mut out);
    Ok(out)
}

/// The default downmix of `channels` channels (FFmpeg's default layout for that
/// count) to mono.
///
/// Weights, found by feeding a unit impulse into each channel: front left and
/// right `sqrt(1/2)` (as `f32`), front centre 1, low-frequency effects 0,
/// every back or surround channel 0.5. The weights by channel count, in
/// channel order (L R C/LFE/back...), as measured:
///
/// | channels | weights |
/// |---|---|
/// | 1 | 1 |
/// | 2 | s s |
/// | 3 | s s 0 (2.1: the third channel is the LFE) |
/// | 4 | s s 1 0.5 (4.0) |
/// | 5 | s s 1 0.5 0.5 (5.0) |
/// | 6 | s s 1 0 0.5 0.5 (5.1) |
/// | 7 | s s 1 0 0.5 0.5 0.5 (6.1) |
/// | 8 | s s 1 0 0.5 0.5 0.5 0.5 (7.1) |
///
/// with `s = sqrt(1/2)` as an `f32`.
///
/// The sum is the one the reference computes: with exactly two weighted
/// channels `fma(w0, x0, w1 * x1)`; with more, a chain of fused multiply-adds
/// from the first weighted channel to the last. Channels of weight 0 do not
/// take part.
#[derive(Clone, Debug)]
pub struct SwrMix {
    channels: usize,
    /// `(channel, weight)` of each channel with a nonzero weight, in order.
    active: Vec<(usize, f32)>,
}

impl SwrMix {
    /// The mix from `channels` channels (1 to [`MAX_SWR_CHANNELS`]).
    pub fn new(channels: usize) -> Result<Self, MediaError> {
        if channels == 0 || channels > MAX_SWR_CHANNELS {
            return Err(MediaError::Unsupported {
                what: "audio",
                detail: format!("no default downmix is modelled for {channels} channels (1 to {MAX_SWR_CHANNELS})"),
            });
        }
        let active = Self::weights(channels)
            .iter()
            .enumerate()
            .filter(|(_, w)| **w != 0.0)
            .map(|(i, w)| (i, *w))
            .collect();
        Ok(Self { channels, active })
    }

    /// The weight of each channel for a channel count in `1..=8`.
    fn weights(channels: usize) -> Vec<f32> {
        const S: f32 = std::f32::consts::FRAC_1_SQRT_2;
        match channels {
            1 => vec![1.0],
            2 => vec![S, S],
            3 => vec![S, S, 0.0],
            4 => vec![S, S, 1.0, 0.5],
            5 => vec![S, S, 1.0, 0.5, 0.5],
            6 => vec![S, S, 1.0, 0.0, 0.5, 0.5],
            7 => vec![S, S, 1.0, 0.0, 0.5, 0.5, 0.5],
            _ => vec![S, S, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5],
        }
    }

    /// The number of input channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Mix one frame, given a way to read channel `i`.
    #[inline]
    fn mix_frame(&self, at: impl Fn(usize) -> f32) -> f32 {
        match self.active.as_slice() {
            [(i, w)] => *w * at(*i),
            [(i0, w0), (i1, w1)] => w0.mul_add(at(*i0), *w1 * at(*i1)),
            [(i0, w0), rest @ ..] => {
                let mut s = *w0 * at(*i0);
                for (i, w) in rest {
                    s = w.mul_add(at(*i), s);
                }
                s
            }
            [] => 0.0,
        }
    }

    /// Mix planar channels (`chans.len()` must equal [`channels`](Self::channels)
    /// and the channels must have equal length); append the mono samples.
    ///
    /// # Panics
    /// If the channel count or lengths do not match.
    pub fn mix_planar(&self, chans: &[Vec<f32>], out: &mut Vec<f32>) {
        assert_eq!(chans.len(), self.channels, "mix_planar: channel count");
        let frames = chans[0].len();
        assert!(
            chans.iter().all(|c| c.len() == frames),
            "mix_planar: channel lengths differ"
        );
        out.extend((0..frames).map(|f| self.mix_frame(|i| chans[i][f])));
    }

    /// Mix interleaved frames; append the mono samples. A trailing partial
    /// frame is ignored.
    pub fn mix_interleaved(&self, frames: &[f32], out: &mut Vec<f32>) {
        out.reserve(frames.len() / self.channels);
        for frame in frames.chunks_exact(self.channels) {
            out.push(self.mix_frame(|i| frame[i]));
        }
    }
}

/// `channels` interleaved channels at `in_rate` to mono at `out_rate`: each
/// channel is resampled, then the channels are mixed by [`SwrMix`]. Equal rates
/// only mix.
pub struct SwrConverter {
    channels: usize,
    mix: SwrMix,
    /// One resampler per channel; empty when the rates are equal.
    resamplers: Vec<SwrResampler>,
    planar_in: Vec<Vec<f32>>,
    planar_out: Vec<Vec<f32>>,
    /// Samples of a frame the last `push` ended in the middle of.
    pending: Vec<f32>,
}

impl SwrConverter {
    /// A converter for `channels` channels (1 to [`MAX_SWR_CHANNELS`]).
    pub fn new(channels: usize, in_rate: u32, out_rate: u32) -> Result<Self, MediaError> {
        let mix = SwrMix::new(channels)?;
        let first = SwrResampler::new(in_rate, out_rate)?;
        let resamplers = if in_rate == out_rate {
            Vec::new()
        } else {
            let mut v: Vec<SwrResampler> = (1..channels).map(|_| first.fresh()).collect();
            v.push(first);
            v
        };
        let planes = if channels > 1 && !resamplers.is_empty() {
            channels
        } else {
            0
        };
        Ok(Self {
            channels,
            mix,
            resamplers,
            planar_in: vec![Vec::new(); planes],
            planar_out: vec![Vec::new(); planes],
            pending: Vec::new(),
        })
    }

    /// Feed interleaved frames; append the mono samples now determined. A
    /// trailing partial frame is held until the next call.
    pub fn push(&mut self, frames: &[f32], out: &mut Vec<f32>) {
        if self.pending.is_empty() {
            let whole = frames.len() - frames.len() % self.channels;
            self.push_whole(&frames[..whole], out);
            self.pending.extend_from_slice(&frames[whole..]);
        } else {
            self.pending.extend_from_slice(frames);
            let whole = self.pending.len() - self.pending.len() % self.channels;
            let rest = self.pending.split_off(whole);
            let data = std::mem::replace(&mut self.pending, rest);
            self.push_whole(&data, out);
        }
    }

    fn push_whole(&mut self, frames: &[f32], out: &mut Vec<f32>) {
        if self.resamplers.is_empty() {
            self.mix.mix_interleaved(frames, out);
        } else if self.channels == 1 {
            self.resamplers[0].push(frames, out);
        } else {
            for (c, plane) in self.planar_in.iter_mut().enumerate() {
                plane.clear();
                plane.extend(frames.iter().skip(c).step_by(self.channels).copied());
            }
            for ((r, input), output) in self
                .resamplers
                .iter_mut()
                .zip(&self.planar_in)
                .zip(&mut self.planar_out)
            {
                r.push(input, output);
            }
            self.mix_planes(out);
        }
    }

    /// Mix and empty the per-channel outputs.
    fn mix_planes(&mut self, out: &mut Vec<f32>) {
        self.mix.mix_planar(&self.planar_out, out);
        for plane in &mut self.planar_out {
            plane.clear();
        }
    }

    /// Flush the tail and start a new clip. A partial frame still held is
    /// dropped.
    pub fn finish(&mut self, out: &mut Vec<f32>) {
        self.pending.clear();
        if self.resamplers.is_empty() {
            return;
        }
        if self.channels == 1 {
            self.resamplers[0].finish(out);
            return;
        }
        for (r, output) in self.resamplers.iter_mut().zip(&mut self.planar_out) {
            r.finish(output);
        }
        self.mix_planes(out);
    }
}

/// `channels` interleaved channels at `in_rate` to mono at `out_rate`: the
/// whole clip at once.
pub fn convert_interleaved(
    frames: &[f32],
    channels: usize,
    in_rate: u32,
    out_rate: u32,
) -> Result<Vec<f32>, MediaError> {
    if channels == 0 {
        return Err(MediaError::Invalid {
            what: "resample",
            detail: "zero channels".into(),
        });
    }
    if frames.len() % channels != 0 {
        return Err(MediaError::Invalid {
            what: "resample",
            detail: format!(
                "{} samples are not a whole number of {channels}-channel frames",
                frames.len()
            ),
        });
    }
    let mut c = SwrConverter::new(channels, in_rate, out_rate)?;
    let mut out = Vec::new();
    c.push(frames, &mut out);
    c.finish(&mut out);
    Ok(out)
}
