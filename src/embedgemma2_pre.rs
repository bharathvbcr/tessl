//! Native preprocessing for EmbeddingGemma 2 (`google/embeddinggemma-2`): raw
//! images, raw 16 kHz audio and decoded video frames to the inputs of the
//! vision and audio towers ([`ImageInput`], [`AudioInput`]), with no Python
//! processor.
//!
//! Everything here reproduces transformers 5.19's `EmbeddingGemma2Processor`
//! as `AutoProcessor.from_pretrained(snapshot)` builds it, which is what
//! sentence-transformers calls. Its parameters are read from the snapshot's
//! `processor_config.json` ([`PreConfig`]); a setting this module does not
//! implement is refused by name rather than ignored.
//!
//! # Image (`Gemma4ImageProcessor`, the torchvision backend)
//!
//! `AutoProcessor` loads the torchvision ("fast") image processor, not the PIL
//! one; the two differ in the resize (PIL's own, against ATen's). The torchvision
//! path, per image:
//!
//! 1. `do_convert_rgb` converts PIL images only: gray and gray+alpha become RGB
//!    by replication and RGBA drops its alpha (no compositing) — PIL's
//!    `convert("RGB")`. [`RawImage`] takes 1, 2, 3 or 4 interleaved channels
//!    and does that.
//! 2. The target size is [`aspect_ratio_preserving_size`]: the largest
//!    multiple of `patch_size * pooling_kernel` per side whose area stays within
//!    `max_soft_tokens * pooling_kernel² * patch_size²`.
//! 3. `tvF.resize(uint8, bicubic, antialias=True)`. On a uint8 CPU tensor
//!    ATen's kernel is a separable fixed-point resample: the weights are
//!    computed in f64 (the antialiased bicubic filter, `a = -0.5`, support
//!    scaled by the downscale factor, normalised to sum to one), converted to
//!    integers at a precision chosen so the largest weight times `2^p` stays
//!    under `2^15`, the width pass runs first and rounds to uint8, then the
//!    height pass does the same. [`resize_rgb_u8`] is that kernel; the result
//!    is bit for bit the one transformers computes (tests/embedgemma2_pre.rs).
//!    It is close to, not equal to, PIL's `Image.resize(BICUBIC)` (PIL fixes the
//!    precision at 22 bits): at most 1 to 2 grey levels apart.
//! 4. Rescale by `rescale_factor` as `f32(u8) * f32(1/255)`, patchify into
//!    `[patches, patch_size² * 3]` with the elements of a patch in (row,
//!    column, channel) order, positions `(x, y)` row-major, and pad to
//!    `max_soft_tokens * pooling_kernel²` patches with zeros and `(-1, -1)`.
//!
//! Only the arm64 generic ATen kernel was checked here. ATen's x86 AVX2 kernel
//! is a different implementation of the same scheme; it is not verified against
//! this one.
//!
//! # Audio (`Gemma4AudioFeatureExtractor`)
//!
//! Mono f32 PCM at the extractor's sampling rate (a different rate is refused,
//! never resampled; transformers does not check it). The clip is cut to
//! `max_samples` (480,000: 30 s, the extractor's `max_length`, rounded up to a
//! multiple of 128 as its `_truncate` does), padded with zeros to a multiple of
//! 128 samples,
//! given `frame_length / 2` zeros in front (semicausal), cut into 320-sample
//! frames every 160 samples, multiplied by the periodic Hann window (f32),
//! transformed by a 512-point real FFT, reduced to the magnitude, projected on
//! the HTK mel filterbank (f64, triangles in Hz, no normalisation), and
//! `ln(mel + mel_floor)`. A frame is real only when its last sample (index
//! `i * hop + frame_length` of the left-padded clip) is a real, unpadded sample;
//! the features of the others are zeroed. The FFT is this module's own
//! (radix-2, f64); transformers' is numpy's f32 pocketfft, so the features
//! agree to f32 rounding, not bit for bit (tests/embedgemma2_pre.rs states the
//! measured bound).
//!
//! The soft-token count of a clip is the number of true entries of the tower's
//! mask ([`crate::embedgemma2_audio::out_mask`]). The processor's
//! `audio_seq_length` caps only `_get_num_multimodal_tokens`, not the
//! placeholders `replace_audio_token` writes, so a 30 s clip has 750
//! placeholders (and soft tokens) whatever `audio_seq_length` says.
//!
//! # Video (`EmbeddingGemma2VideoProcessor`)
//!
//! Frames are chosen by [`sample_frame_indices`] (fps sampling, then a uniform
//! or truncating cap at `max_frames`) and each goes through the image path
//! above at the video's `max_soft_tokens`. Fps sampling needs a duration;
//! [`preprocess_video`] gives the processor `duration = frames / fps`, as
//! transformers' own loaders do.
//!
//! # Bounds
//!
//! Every input is checked before work starts: an image has at most
//! [`MAX_IMAGE_PIXELS`] pixels and its width-pass intermediate at most
//! [`MAX_INTERMEDIATE_BYTES`]; a video at most [`MAX_VIDEO_FRAMES`] decoded
//! frames, of which at most `max_frames` are processed; audio is cut to
//! `max_samples` before any frame is built, and refused when it holds a
//! non-finite sample (the tower refuses them too).

use std::path::Path;

use crate::embedgemma2_audio::{out_mask, AudioInput};
use crate::embedgemma2_mm::read_config_text;
use crate::embedgemma2_vision::ImageInput;
use crate::json::{self, Json, NumForm, Syntax};

/// Most pixels (`width * height`) of one input image.
pub const MAX_IMAGE_PIXELS: usize = 1 << 28;

/// Most bytes of the width pass's output (`height * target_width * 3`).
pub const MAX_INTERMEDIATE_BYTES: usize = 1 << 31;

/// Most decoded frames one [`preprocess_video`] call takes.
pub const MAX_VIDEO_FRAMES: usize = 1 << 20;

/// The soft-token budgets the processors accept.
pub const SUPPORTED_SOFT_TOKENS: [u32; 5] = [70, 140, 280, 560, 1120];

/// `Gemma4AudioFeatureExtractor.__call__`'s `max_length` default: 30 s at 16 kHz.
pub const DEFAULT_MAX_AUDIO_SAMPLES: usize = 480_000;

/// `Gemma4AudioFeatureExtractor.__call__`'s `pad_to_multiple_of` default.
pub const AUDIO_PAD_MULTIPLE: usize = 128;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// The image processor's parameters (`image_processor` of
/// `processor_config.json`, or `video_processor`'s per-frame part).
#[derive(Clone, Debug, PartialEq)]
pub struct ImageConfig {
    pub patch_size: u32,
    pub pooling_kernel: u32,
    /// One of [`SUPPORTED_SOFT_TOKENS`].
    pub max_soft_tokens: u32,
    /// `rescale_factor`, held as the f64 the file spells (applied as its f32).
    pub rescale_factor: f64,
}

impl ImageConfig {
    /// Patches after padding: `max_soft_tokens * pooling_kernel²`.
    pub fn max_patches(&self) -> usize {
        self.max_soft_tokens as usize * (self.pooling_kernel as usize).pow(2)
    }

    /// Columns of one patch: `3 * patch_size²`.
    pub fn patch_dim(&self) -> usize {
        3 * (self.patch_size as usize).pow(2)
    }

    fn validate(&self, what: &str) -> Result<(), String> {
        if self.patch_size == 0 || self.pooling_kernel == 0 {
            return Err(format!(
                "{what}: patch_size {} and pooling_kernel_size {} must be positive",
                self.patch_size, self.pooling_kernel
            ));
        }
        if !SUPPORTED_SOFT_TOKENS.contains(&self.max_soft_tokens) {
            return Err(format!(
                "{what}: max_soft_tokens must be one of {SUPPORTED_SOFT_TOKENS:?}, got {}",
                self.max_soft_tokens
            ));
        }
        if !(self.rescale_factor.is_finite() && self.rescale_factor > 0.0) {
            return Err(format!(
                "{what}: rescale_factor must be positive, got {}",
                self.rescale_factor
            ));
        }
        Ok(())
    }
}

/// The audio feature extractor's parameters (`feature_extractor`).
#[derive(Clone, Debug, PartialEq)]
pub struct AudioFeatureConfig {
    pub sampling_rate: u32,
    pub frame_length: usize,
    pub hop_length: usize,
    pub fft_length: usize,
    pub mel_bins: usize,
    pub min_frequency: f64,
    pub max_frequency: f64,
    pub mel_floor: f64,
    /// The extractor's `max_length` ([`DEFAULT_MAX_AUDIO_SAMPLES`]). Longer clips
    /// are cut to it rounded up to a multiple of [`AUDIO_PAD_MULTIPLE`] (the
    /// extractor's `_truncate` does that rounding, so the cut samples are real).
    pub max_samples: usize,
}

impl AudioFeatureConfig {
    fn validate(&self, what: &str) -> Result<(), String> {
        if self.sampling_rate == 0 {
            return Err(format!("{what}: sampling_rate is zero"));
        }
        if self.frame_length < 2 || self.hop_length == 0 || self.mel_bins == 0 {
            return Err(format!(
                "{what}: frame_length {}, hop_length {} and feature_size {} must be positive (frame at least 2)",
                self.frame_length, self.hop_length, self.mel_bins
            ));
        }
        if !self.fft_length.is_power_of_two() || self.fft_length < self.frame_length || self.fft_length < 4 {
            return Err(format!(
                "{what}: fft_length {} must be a power of two of at least 4 and frame_length {}",
                self.fft_length, self.frame_length
            ));
        }
        if self.fft_length > 1 << 16 {
            return Err(format!("{what}: fft_length {} is above 65536", self.fft_length));
        }
        let nyquist = f64::from(self.sampling_rate / 2);
        if !(self.min_frequency.is_finite()
            && self.max_frequency.is_finite()
            && 0.0 <= self.min_frequency
            && self.min_frequency <= self.max_frequency
            && self.max_frequency <= nyquist)
        {
            return Err(format!(
                "{what}: need 0 <= min_frequency ({}) <= max_frequency ({}) <= {nyquist}",
                self.min_frequency, self.max_frequency
            ));
        }
        if !(self.mel_floor.is_finite() && self.mel_floor > 0.0) {
            return Err(format!("{what}: mel_floor must be positive, got {}", self.mel_floor));
        }
        if self.max_samples == 0 {
            return Err(format!("{what}: max_samples is zero"));
        }
        Ok(())
    }
}

/// How the video processor shortens a video that has more than `max_frames`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overflow {
    /// `np.linspace(0, n - 1, max_frames, dtype=int)` over the sampled frames.
    Uniform,
    /// The first `max_frames`.
    Truncate,
}

/// The video processor's parameters (`video_processor`).
#[derive(Clone, Debug, PartialEq)]
pub struct VideoConfig {
    /// Per-frame processing.
    pub image: ImageConfig,
    /// Target frames per second; `None` keeps every frame before the cap.
    pub fps: Option<f64>,
    pub max_frames: usize,
    pub overflow: Overflow,
    /// `do_sample_frames`: when false every given frame is processed.
    pub do_sample_frames: bool,
    /// `add_timestamps`: the processor writes each frame's `mm:ss` time in
    /// front of its block. Preprocessing is the same either way; the setting
    /// is the sequence layout's, read by the caller (or by a facade item that
    /// carries the timestamps: `MmItem::TimestampedVideo`) and applied by
    /// [`crate::embedgemma2::EmbedGemma2Config::video_sequence_with_timestamps`].
    pub add_timestamps: bool,
}

/// Everything `processor_config.json` configures.
#[derive(Clone, Debug, PartialEq)]
pub struct PreConfig {
    pub image: ImageConfig,
    pub audio: AudioFeatureConfig,
    pub video: VideoConfig,
}

const SYNTAX: Syntax = Syntax {
    what: "processor_config.json",
    max_depth: 8,
    uints_only: false,
    literals: true,
};

/// A section of the config with typed, defaulted reads.
struct Section<'a> {
    name: &'a str,
    obj: &'a Json,
}

impl Section<'_> {
    fn err(&self, key: &str, want: &str, got: &Json) -> String {
        format!("processor_config.json: {}.{key} must be {want}, got {got:?}", self.name)
    }

    fn num(&self, key: &str, default: f64) -> Result<f64, String> {
        match self.obj.get(key) {
            None => Ok(default),
            Some(Json::Num { value, .. }) => Ok(*value),
            Some(v) => Err(self.err(key, "a number", v)),
        }
    }

    fn uint(&self, key: &str, default: u64) -> Result<u64, String> {
        match self.obj.get(key) {
            None => Ok(default),
            Some(Json::Num {
                form: NumForm::Uint,
                int: n,
                ..
            }) => Ok(*n),
            Some(v) => Err(self.err(key, "a non-negative integer", v)),
        }
    }

    fn flag(&self, key: &str, default: bool) -> Result<bool, String> {
        match self.obj.get(key) {
            None => Ok(default),
            Some(Json::Bool(b)) => Ok(*b),
            Some(v) => Err(self.err(key, "a boolean", v)),
        }
    }

    fn text(&self, key: &str, default: &str) -> Result<String, String> {
        match self.obj.get(key) {
            None => Ok(default.to_string()),
            Some(Json::Str(s)) => Ok(s.clone()),
            Some(v) => Err(self.err(key, "a string", v)),
        }
    }

    fn is_null(&self, key: &str) -> Result<bool, String> {
        match self.obj.get(key) {
            None | Some(Json::Null) => Ok(true),
            Some(v) => Err(format!(
                "processor_config.json: {}.{key} = {v:?} is not supported (only null)",
                self.name
            )),
        }
    }

    fn u32(&self, key: &str, default: u32) -> Result<u32, String> {
        let n = self.uint(key, u64::from(default))?;
        u32::try_from(n).map_err(|_| format!("processor_config.json: {}.{key} = {n} exceeds u32", self.name))
    }

    fn usize(&self, key: &str, default: usize) -> Result<usize, String> {
        let n = self.uint(key, default as u64)?;
        usize::try_from(n).map_err(|_| format!("processor_config.json: {}.{key} = {n} exceeds usize", self.name))
    }

    /// Refuse a setting that is on (or off, for `want`) and not implemented.
    fn require_flag(&self, key: &str, default: bool, want: bool) -> Result<(), String> {
        if self.flag(key, default)? != want {
            return Err(format!(
                "processor_config.json: {}.{key} = {} is not supported (only {want})",
                self.name, !want
            ));
        }
        Ok(())
    }

    fn require_num(&self, key: &str, default: f64, want: f64) -> Result<(), String> {
        let v = self.num(key, default)?;
        if v != want {
            return Err(format!(
                "processor_config.json: {}.{key} = {v} is not supported (only {want})",
                self.name
            ));
        }
        Ok(())
    }
}

fn section<'a>(root: &'a Json, name: &'a str) -> Result<Section<'a>, String> {
    match root.get(name) {
        Some(obj @ Json::Object(_)) => Ok(Section { name, obj }),
        Some(v) => Err(format!("processor_config.json: {name} must be an object, got {v:?}")),
        None => Err(format!("processor_config.json: missing {name:?}")),
    }
}

/// PIL's `BICUBIC`, the only resample the resize implements.
const PIL_BICUBIC: u64 = 3;

/// The image-processing keys the two processors share, defaults as the HF
/// classes have them (`default_max_soft_tokens`: 280 for images, 70 for video).
fn image_config(s: &Section<'_>, default_max_soft_tokens: u32) -> Result<ImageConfig, String> {
    s.require_flag("do_convert_rgb", true, true)?;
    s.require_flag("do_resize", true, true)?;
    s.require_flag("do_rescale", true, true)?;
    // The video class defaults do_normalize to true; the snapshot turns it off.
    // A normalisation (even mean 0, std 1) is the fused `(x - m) / s` in torch,
    // which is not `x * rescale_factor`; it is not implemented.
    s.require_flag("do_normalize", false, false)?;
    let resample = s.uint("resample", PIL_BICUBIC)?;
    if resample != PIL_BICUBIC {
        return Err(format!(
            "processor_config.json: {}.resample = {resample} is not supported (only {PIL_BICUBIC}, bicubic)",
            s.name
        ));
    }
    let cfg = ImageConfig {
        patch_size: s.u32("patch_size", 16)?,
        pooling_kernel: s.u32("pooling_kernel_size", 3)?,
        max_soft_tokens: s.u32("max_soft_tokens", default_max_soft_tokens)?,
        rescale_factor: s.num("rescale_factor", 1.0 / 255.0)?,
    };
    cfg.validate(&format!("processor_config.json: {}", s.name))?;
    Ok(cfg)
}

fn audio_config(s: &Section<'_>) -> Result<AudioFeatureConfig, String> {
    // Unimplemented stages: refused when on.
    s.require_num("preemphasis", 0.0, 0.0)?;
    s.require_num("dither", 0.0, 0.0)?;
    s.require_num("input_scale_factor", 1.0, 1.0)?;
    s.require_flag("fft_overdrive", false, false)?;
    s.require_flag("return_attention_mask", true, true)?;
    s.require_num("padding_value", 0.0, 0.0)?;
    if s.text("padding_side", "right")? != "right" {
        return Err("processor_config.json: feature_extractor.padding_side must be \"right\"".into());
    }
    if !s.is_null("per_bin_mean")? || !s.is_null("per_bin_stddev")? {
        return Err("processor_config.json: per_bin_mean and per_bin_stddev are not supported".into());
    }
    let sampling_rate = s.u32("sampling_rate", 16_000)?;
    // Gemma4AudioFeatureExtractor.__init__ derives the frame and hop from
    // `frame_length_ms` / `hop_length_ms` (20 and 10) and the FFT from the
    // frame; the snapshot's sample-count keys restate them and transformers
    // ignores them. Use them when they agree, refuse when they do not.
    let ms = |key: &str, default: f64| -> Result<usize, String> {
        let v = s.num(key, default)?;
        let samples = (f64::from(sampling_rate) * v / 1000.0).round();
        if !(samples.is_finite() && (1.0..=1e9).contains(&samples)) {
            return Err(format!(
                "processor_config.json: feature_extractor.{key} = {v} is out of range"
            ));
        }
        Ok(samples as usize)
    };
    let frame_length = ms("frame_length_ms", 20.0)?;
    let hop_length = ms("hop_length_ms", 10.0)?;
    let fft_length = frame_length.next_power_of_two();
    for (key, derived) in [
        ("frame_length", frame_length),
        ("hop_length", hop_length),
        ("fft_length", fft_length),
    ] {
        let stated = s.usize(key, derived)?;
        if stated != derived {
            return Err(format!(
                "processor_config.json: feature_extractor.{key} = {stated}, but transformers derives {derived} \
                 from frame_length_ms / hop_length_ms and the sampling rate"
            ));
        }
    }
    let cfg = AudioFeatureConfig {
        sampling_rate,
        frame_length,
        hop_length,
        fft_length,
        mel_bins: s.usize("feature_size", 128)?,
        min_frequency: s.num("min_frequency", 0.0)?,
        max_frequency: s.num("max_frequency", 8000.0)?,
        mel_floor: s.num("mel_floor", 1e-3)?,
        max_samples: DEFAULT_MAX_AUDIO_SAMPLES,
    };
    cfg.validate("processor_config.json: feature_extractor")?;
    Ok(cfg)
}

fn video_config(s: &Section<'_>) -> Result<VideoConfig, String> {
    let image = image_config(s, 70)?;
    // The base class's `fps` is None-able; the EmbeddingGemma2 class defaults 1.
    let fps = match s.obj.get("fps") {
        None => Some(1.0),
        Some(Json::Null) => None,
        Some(Json::Num { value, .. }) if value.is_finite() && *value > 0.0 => Some(*value),
        Some(v) => return Err(s.err("fps", "a positive number or null", v)),
    };
    let max_frames = s.usize("max_frames", 32)?;
    if max_frames == 0 {
        return Err("processor_config.json: video_processor.max_frames is zero".into());
    }
    let overflow = match s.text("overflow_strategy", "uniform")?.as_str() {
        "uniform" => Overflow::Uniform,
        "truncate" => Overflow::Truncate,
        other => {
            return Err(format!(
                "processor_config.json: video_processor.overflow_strategy {other:?} is not \"uniform\" or \"truncate\""
            ))
        }
    };
    Ok(VideoConfig {
        image,
        fps,
        max_frames,
        overflow,
        do_sample_frames: s.flag("do_sample_frames", true)?,
        // `False` in the class (video processor:194) and in the snapshot.
        add_timestamps: s.flag("add_timestamps", false)?,
    })
}

impl PreConfig {
    /// Parse the text of a snapshot's `processor_config.json`.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let root = json::parse(text, SYNTAX)?;
        if !matches!(root, Json::Object(_)) {
            return Err("processor_config.json: the root is not an object".into());
        }
        Ok(Self {
            image: image_config(&section(&root, "image_processor")?, 280)?,
            audio: audio_config(&section(&root, "feature_extractor")?)?,
            video: video_config(&section(&root, "video_processor")?)?,
        })
    }

    /// Read `processor_config.json` from `path` (bounded, a regular file).
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let text = read_config_text(path)?;
        Self::from_json(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Read `processor_config.json` from a Hugging Face snapshot directory.
    pub fn from_snapshot(dir: &Path) -> Result<Self, String> {
        Self::from_file(&dir.join("processor_config.json"))
    }
}

// ---------------------------------------------------------------------------
// Image: target size
// ---------------------------------------------------------------------------

/// `get_aspect_ratio_preserving_size`: the `(height, width)` an image of
/// `height x width` is resized to, each a multiple of `patch_size *
/// pooling_kernel`, with at most `max_patches` patches. Python's float
/// arithmetic, step for step.
pub fn aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel: usize,
) -> Result<(usize, usize), String> {
    const WHAT: &str = "aspect_ratio_preserving_size";
    if height == 0 || width == 0 {
        return Err(format!("{WHAT}: a {height}x{width} image has no pixels"));
    }
    let total_px = (height as f64) * (width as f64);
    let target_px = (max_patches * patch_size * patch_size) as f64;
    let factor = (target_px / total_px).sqrt();
    let ideal_height = factor * height as f64;
    let ideal_width = factor * width as f64;
    let side_mult = pooling_kernel * patch_size;
    let side = side_mult as f64;

    let mut target_height = (ideal_height / side).floor() as usize * side_mult;
    let mut target_width = (ideal_width / side).floor() as usize * side_mult;
    if target_height == 0 && target_width == 0 {
        return Err(format!(
            "{WHAT}: attempting to resize {height}x{width} to 0x0; sides must be multiples of \
             pooling_kernel_size * patch_size = {side_mult}"
        ));
    }
    let max_side_length = (max_patches / (pooling_kernel * pooling_kernel)) * side_mult;
    if target_height == 0 {
        target_height = side_mult;
        target_width = ((width as f64 / height as f64).floor() as usize * side_mult).min(max_side_length);
    } else if target_width == 0 {
        target_width = side_mult;
        target_height = ((height as f64 / width as f64).floor() as usize * side_mult).min(max_side_length);
    }
    if target_height == 0 || target_width == 0 {
        return Err(format!(
            "{WHAT}: {height}x{width} resolves to {target_height}x{target_width}, an empty image"
        ));
    }
    if (target_height * target_width) as f64 > target_px {
        return Err(format!(
            "{WHAT}: resizing {height}x{width} to {target_height}x{target_width} exceeds {max_patches} patches \
             with patch_size {patch_size}"
        ));
    }
    Ok((target_height, target_width))
}

/// The soft tokens (the placeholders of the processor's sequence) an image of
/// `height x width` pixels makes: the patches of its [`aspect_ratio_preserving_size`]
/// divided by the pooling cell, which is `PreparedImage::soft_tokens` of
/// [`preprocess_image`] without resizing anything. The same count for a video
/// frame under the video's `ImageConfig`.
pub fn image_soft_tokens(cfg: &ImageConfig, height: usize, width: usize) -> Result<usize, String> {
    cfg.validate("image_soft_tokens")?;
    let (th, tw) = aspect_ratio_preserving_size(
        height,
        width,
        cfg.patch_size as usize,
        cfg.max_patches(),
        cfg.pooling_kernel as usize,
    )?;
    let p = cfg.patch_size as usize;
    Ok((th / p) * (tw / p) / (cfg.pooling_kernel as usize).pow(2))
}

// ---------------------------------------------------------------------------
// Image: the resize
// ---------------------------------------------------------------------------

/// Taps of the bicubic filter (`interp_size`): its support is two pixels each way.
const BICUBIC_INTERP: f64 = 4.0;

/// `HelperInterpCubic::aa_filter`: Keys' cubic convolution with `a = -0.5`.
fn bicubic_filter(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

/// The integer taps of one axis' resample: for output `i`, the input pixels
/// `start[i]..start[i] + len[i]` with weights `w[off[i]..off[i] + len[i]]`
/// (each a multiple of `2^-precision`).
struct AxisWeights {
    start: Vec<usize>,
    len: Vec<usize>,
    off: Vec<usize>,
    w: Vec<i32>,
    precision: u32,
}

impl AxisWeights {
    /// ATen's `_compute_indices_min_size_weights_aa` for the bicubic filter
    /// (`align_corners = false`), then the int16 conversion of
    /// `compute_index_ranges_int16_weights`.
    fn new(in_size: usize, out_size: usize) -> Self {
        let scale = in_size as f64 / out_size as f64;
        let support = if scale >= 1.0 {
            BICUBIC_INTERP * 0.5 * scale
        } else {
            BICUBIC_INTERP * 0.5
        };
        let max_interp = support.ceil() as usize * 2 + 1;
        let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
        let mut start = Vec::with_capacity(out_size);
        let mut len = Vec::with_capacity(out_size);
        let mut off = Vec::with_capacity(out_size);
        let mut weights: Vec<f64> = Vec::with_capacity(out_size * max_interp);
        for i in 0..out_size {
            let center = scale * (i as f64 + 0.5);
            let xmin = ((center - support + 0.5) as i64).max(0);
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64);
            let xsize = ((xmax - xmin).max(0) as usize).min(max_interp);
            start.push(xmin as usize);
            len.push(xsize);
            off.push(weights.len());
            let first = weights.len();
            let mut total = 0.0f64;
            for j in 0..xsize {
                let w = bicubic_filter(((j as i64 + xmin) as f64 - center + 0.5) * invscale);
                weights.push(w);
                total += w;
            }
            if total != 0.0 {
                for w in &mut weights[first..] {
                    *w /= total;
                }
            }
        }
        // The largest weight sets the fixed-point precision: the most bits for
        // which every weight times 2^(p + 1) stays under 2^15.
        let max_w = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut precision = 0u32;
        while precision < 22 {
            let next = (0.5 + max_w * f64::from(1u32 << (precision + 1))) as i32;
            if next >= 1 << 15 {
                break;
            }
            precision += 1;
        }
        let one = f64::from(1u32 << precision);
        let w = weights
            .iter()
            .map(|&w| (w * one + if w < 0.0 { -0.5 } else { 0.5 }) as i32)
            .collect();
        Self {
            start,
            len,
            off,
            w,
            precision,
        }
    }

    fn out_size(&self) -> usize {
        self.start.len()
    }
}

/// Threads for a pass over `rows` output rows of `row_bytes` each: one below
/// roughly a quarter megabyte, else up to `available_parallelism` (at most 8).
fn pass_threads(rows: usize, row_bytes: usize) -> usize {
    if rows * row_bytes < 1 << 18 || rows < 2 {
        return 1;
    }
    std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(8)
        .min(rows)
}

/// Run `f(first_row, rows_slice)` over `out` split into contiguous row blocks,
/// on up to `threads` scoped threads. Rows are independent, so the result does
/// not depend on the split.
fn for_row_blocks<F>(out: &mut [u8], row_bytes: usize, threads: usize, f: F)
where
    F: Fn(usize, &mut [u8]) + Sync,
{
    let rows = out.len() / row_bytes;
    if threads <= 1 || rows == 0 {
        f(0, out);
        return;
    }
    let per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (b, block) in out.chunks_mut(per * row_bytes).enumerate() {
            let f = &f;
            scope.spawn(move || f(b * per, block));
        }
    });
}

/// `(acc >> precision)` clamped to a byte; `acc` already holds the rounding half.
#[inline(always)]
fn to_u8(acc: i32, precision: u32) -> u8 {
    (acc >> precision).clamp(0, 255) as u8
}

/// Resize an interleaved RGB image `[h, w, 3]` to `[th, tw, 3]` the way
/// `tvF.resize(uint8_tensor, bicubic, antialias=True)` does on CPU: the width
/// pass, rounded to uint8, then the height pass. Bit for bit the torchvision
/// result (see the module docs for what that is and what it is not).
pub fn resize_rgb_u8(src: &[u8], h: usize, w: usize, th: usize, tw: usize) -> Result<Vec<u8>, String> {
    const WHAT: &str = "resize_rgb_u8";
    if h == 0 || w == 0 || th == 0 || tw == 0 {
        return Err(format!("{WHAT}: {h}x{w} to {th}x{tw} has an empty side"));
    }
    if src.len() != h * w * 3 {
        return Err(format!("{WHAT}: {} bytes for a {h}x{w}x3 image", src.len()));
    }
    let across = AxisWeights::new(w, tw);
    let down = AxisWeights::new(h, th);
    resize_with(src, h, w, th, tw, &across, &down)
}

fn resize_with(
    src: &[u8],
    h: usize,
    w: usize,
    th: usize,
    tw: usize,
    across: &AxisWeights,
    down: &AxisWeights,
) -> Result<Vec<u8>, String> {
    let mid_bytes = h
        .checked_mul(tw)
        .and_then(|n| n.checked_mul(3))
        .filter(|&n| n <= MAX_INTERMEDIATE_BYTES)
        .ok_or_else(|| {
            format!("resize: the width pass of a {h}x{w} image to width {tw} exceeds {MAX_INTERMEDIATE_BYTES} bytes")
        })?;
    // Width pass: every source row to `tw` pixels.
    let mut mid = vec![0u8; mid_bytes];
    let row_in = w * 3;
    let row_mid = tw * 3;
    let half = 1i32 << (across.precision - 1);
    for_row_blocks(&mut mid, row_mid, pass_threads(h, row_mid), |first, block| {
        for (r, out_row) in block.chunks_exact_mut(row_mid).enumerate() {
            let in_row = &src[(first + r) * row_in..][..row_in];
            for (x, px) in out_row.chunks_exact_mut(3).enumerate() {
                let (start, n, off) = (across.start[x], across.len[x], across.off[x]);
                let taps = &across.w[off..off + n];
                let pixels = &in_row[start * 3..(start + n) * 3];
                let (mut a0, mut a1, mut a2) = (half, half, half);
                for (&wt, p) in taps.iter().zip(pixels.chunks_exact(3)) {
                    a0 += wt * i32::from(p[0]);
                    a1 += wt * i32::from(p[1]);
                    a2 += wt * i32::from(p[2]);
                }
                px[0] = to_u8(a0, across.precision);
                px[1] = to_u8(a1, across.precision);
                px[2] = to_u8(a2, across.precision);
            }
        }
    });
    // Height pass: `th` rows, each a weighted sum of source rows.
    let mut out = vec![0u8; th * row_mid];
    let half = 1i32 << (down.precision - 1);
    for_row_blocks(&mut out, row_mid, pass_threads(th, row_mid), |first, block| {
        let mut acc = vec![0i32; row_mid];
        for (r, out_row) in block.chunks_exact_mut(row_mid).enumerate() {
            let y = first + r;
            let (start, n, off) = (down.start[y], down.len[y], down.off[y]);
            acc.fill(half);
            for (&wt, in_row) in down.w[off..off + n]
                .iter()
                .zip(mid[start * row_mid..(start + n) * row_mid].chunks_exact(row_mid))
            {
                for (a, &p) in acc.iter_mut().zip(in_row) {
                    *a += wt * i32::from(p);
                }
            }
            for (o, &a) in out_row.iter_mut().zip(&acc) {
                *o = to_u8(a, down.precision);
            }
        }
    });
    debug_assert_eq!(down.out_size(), th);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Image: input, patches
// ---------------------------------------------------------------------------

/// A decoded image: `height * width` pixels of `channels` interleaved u8
/// samples (row-major HWC).
#[derive(Clone, Copy, Debug)]
pub struct RawImage<'a> {
    pub width: usize,
    pub height: usize,
    /// 1 (gray), 2 (gray, alpha), 3 (RGB) or 4 (RGBA).
    pub channels: usize,
    pub data: &'a [u8],
}

impl<'a> RawImage<'a> {
    /// An RGB image.
    pub fn rgb(width: usize, height: usize, data: &'a [u8]) -> Self {
        Self {
            width,
            height,
            channels: 3,
            data,
        }
    }

    fn check(&self, what: &str) -> Result<(), String> {
        if self.width == 0 || self.height == 0 {
            return Err(format!("{what}: a {}x{} image has no pixels", self.height, self.width));
        }
        if !(1..=4).contains(&self.channels) {
            return Err(format!(
                "{what}: {} channels; 1 (gray), 2 (gray+alpha), 3 (RGB) or 4 (RGBA) expected",
                self.channels
            ));
        }
        let pixels = self.width.checked_mul(self.height).filter(|&p| p <= MAX_IMAGE_PIXELS);
        let Some(pixels) = pixels else {
            return Err(format!(
                "{what}: a {}x{} image exceeds {MAX_IMAGE_PIXELS} pixels",
                self.height, self.width
            ));
        };
        if self.data.len() != pixels * self.channels {
            return Err(format!(
                "{what}: {} bytes for a {}x{}x{} image ({} expected)",
                self.data.len(),
                self.height,
                self.width,
                self.channels,
                pixels * self.channels
            ));
        }
        Ok(())
    }

    /// PIL's `convert("RGB")` of the image: gray replicated, alpha dropped.
    fn into_rgb(self) -> std::borrow::Cow<'a, [u8]> {
        use std::borrow::Cow;
        match self.channels {
            3 => Cow::Borrowed(self.data),
            4 => Cow::Owned(self.data.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect()),
            1 => Cow::Owned(self.data.iter().flat_map(|&g| [g, g, g]).collect()),
            _ => Cow::Owned(self.data.chunks_exact(2).flat_map(|p| [p[0], p[0], p[0]]).collect()),
        }
    }
}

/// One preprocessed image (or video frame): what the vision tower takes.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedImage {
    /// `[max_patches, patch_dim]`, f32 in `[0, 1]`, padding patches zero.
    pub pixel_values: Vec<f32>,
    /// `[max_patches, 2]` `(x, y)`; padding patches are `(-1, -1)`.
    pub position_ids: Vec<i32>,
    /// Real patches (the rest is padding).
    pub patches: usize,
    /// `patches / pooling_kernel²`: the soft tokens the tower returns.
    pub soft_tokens: usize,
    /// The resized image's `(height, width)`.
    pub resized: (usize, usize),
}

impl PreparedImage {
    /// The tower's view of this image.
    pub fn input(&self) -> ImageInput<'_> {
        ImageInput {
            pixel_values: &self.pixel_values,
            position_ids: &self.position_ids,
        }
    }
}

/// A resized RGB image to patches and positions, padded to `max_patches`.
fn patchify(rgb: &[u8], th: usize, tw: usize, cfg: &ImageConfig) -> PreparedImage {
    let p = cfg.patch_size as usize;
    let (rows, cols) = (th / p, tw / p);
    let patches = rows * cols;
    let dim = cfg.patch_dim();
    let max_patches = cfg.max_patches();
    let scale = cfg.rescale_factor as f32;
    let mut pixel_values = vec![0.0f32; max_patches * dim];
    let mut position_ids = vec![-1i32; max_patches * 2];
    for py in 0..rows {
        for px in 0..cols {
            let idx = py * cols + px;
            position_ids[2 * idx] = px as i32;
            position_ids[2 * idx + 1] = py as i32;
            let patch = &mut pixel_values[idx * dim..(idx + 1) * dim];
            for dy in 0..p {
                let row = ((py * p + dy) * tw + px * p) * 3;
                for (o, &v) in patch[dy * p * 3..(dy + 1) * p * 3]
                    .iter_mut()
                    .zip(&rgb[row..row + p * 3])
                {
                    *o = f32::from(v) * scale;
                }
            }
        }
    }
    PreparedImage {
        pixel_values,
        position_ids,
        patches,
        soft_tokens: patches / (cfg.pooling_kernel as usize).pow(2),
        resized: (th, tw),
    }
}

/// The image path for one image: convert, resize, rescale, patchify.
fn prepare_image(
    cfg: &ImageConfig,
    rgb: &[u8],
    h: usize,
    w: usize,
    weights: Option<&(AxisWeights, AxisWeights)>,
) -> Result<PreparedImage, String> {
    let (th, tw) = aspect_ratio_preserving_size(
        h,
        w,
        cfg.patch_size as usize,
        cfg.max_patches(),
        cfg.pooling_kernel as usize,
    )?;
    if (th, tw) == (h, w) {
        return Ok(patchify(rgb, th, tw, cfg));
    }
    let resized = match weights {
        Some((across, down)) => resize_with(rgb, h, w, th, tw, across, down)?,
        None => resize_rgb_u8(rgb, h, w, th, tw)?,
    };
    Ok(patchify(&resized, th, tw, cfg))
}

/// Preprocess one image the way `Gemma4ImageProcessor` does.
pub fn preprocess_image(cfg: &ImageConfig, image: &RawImage<'_>) -> Result<PreparedImage, String> {
    const WHAT: &str = "preprocess_image";
    cfg.validate(WHAT)?;
    image.check(WHAT)?;
    let rgb = image.into_rgb();
    prepare_image(cfg, &rgb, image.height, image.width, None)
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

/// A real FFT of a power-of-two length `n`, by a radix-2 complex FFT of `n / 2`
/// and the usual even/odd split. f64 throughout.
struct RealFft {
    n: usize,
    /// Bit reversal of `0..n/2`.
    rev: Vec<usize>,
    /// `e^{-2 pi i k / (n/2)}` for the half-size FFT, `k < n/4`.
    half_tw: Vec<(f64, f64)>,
    /// `e^{-2 pi i k / n}` for the split, `k <= n/2`.
    split_tw: Vec<(f64, f64)>,
}

impl RealFft {
    fn new(n: usize) -> Self {
        debug_assert!(n.is_power_of_two() && n >= 4);
        let m = n / 2;
        let bits = m.trailing_zeros();
        let rev = (0..m).map(|i| i.reverse_bits() >> (usize::BITS - bits)).collect();
        let tw = |k: usize, of: usize| {
            let a = -2.0 * std::f64::consts::PI * k as f64 / of as f64;
            (a.cos(), a.sin())
        };
        Self {
            n,
            rev,
            half_tw: (0..m / 2).map(|k| tw(k, m)).collect(),
            split_tw: (0..=m).map(|k| tw(k, n)).collect(),
        }
    }

    /// The magnitudes `|X[k]|`, `k = 0..=n/2`, of the real signal `x` (zero-padded
    /// to `n`), into `out`; `re` and `im` are `n / 2` of scratch each.
    fn magnitudes(&self, x: &[f64], re: &mut [f64], im: &mut [f64], out: &mut [f64]) {
        let m = self.n / 2;
        for (i, &r) in self.rev.iter().enumerate() {
            re[r] = x.get(2 * i).copied().unwrap_or(0.0);
            im[r] = x.get(2 * i + 1).copied().unwrap_or(0.0);
        }
        let mut size = 2;
        while size <= m {
            let step = m / size;
            for block in (0..m).step_by(size) {
                for j in 0..size / 2 {
                    let (wr, wi) = self.half_tw[j * step];
                    let (a, b) = (block + j, block + j + size / 2);
                    let (tr, ti) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            size *= 2;
        }
        // X[k] = E[k] + w^k O[k] with E, O the transforms of the even and odd
        // samples, recovered from Z = E + iO.
        for (k, o) in out.iter_mut().enumerate().take(m + 1) {
            let (zr, zi) = (re[k % m], im[k % m]);
            let (cr, ci) = (re[(m - k) % m], -im[(m - k) % m]);
            let (er, ei) = (0.5 * (zr + cr), 0.5 * (zi + ci));
            let (dr, di) = (0.5 * (zr - cr), 0.5 * (zi - ci));
            // O = (Z - conj(Z_rev)) / (2i) = (di, -dr)
            let (or_, oi) = (di, -dr);
            let (wr, wi) = self.split_tw[k];
            let xr = er + wr * or_ - wi * oi;
            let xi = ei + wr * oi + wi * or_;
            *o = xr.hypot(xi);
        }
    }
}

/// `np.hanning(m)[:-1]`, the periodic Hann window of length `m - 1`, as f32:
/// numpy's own formula, `0.5 + 0.5 cos(pi n / (M - 1))` over `n = 1 - M, 3 - M, ..`.
fn periodic_hann(length: usize) -> Vec<f32> {
    let m = length + 1;
    (0..length)
        .map(|i| {
            let n = (1.0 - m as f64) + 2.0 * i as f64;
            (0.5 + 0.5 * (std::f64::consts::PI * n / (m as f64 - 1.0)).cos()) as f32
        })
        .collect()
}

/// `mel_filter_bank(norm=None, mel_scale="htk")` as `[fft_bins, mel_bins]`
/// (row-major), f64, built with the same operations transformers uses.
fn htk_mel_filters(cfg: &AudioFeatureConfig) -> Vec<f64> {
    let bins = cfg.fft_length / 2 + 1;
    let n_mel = cfg.mel_bins;
    let to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let to_hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let (mel_min, mel_max) = (to_mel(cfg.min_frequency), to_mel(cfg.max_frequency));
    // np.linspace(mel_min, mel_max, n_mel + 2): i * step + start, the last forced to stop.
    let points = n_mel + 2;
    let step = (mel_max - mel_min) / (points - 1) as f64;
    let filter_freqs: Vec<f64> = (0..points)
        .map(|i| {
            let m = if i == points - 1 {
                mel_max
            } else {
                i as f64 * step + mel_min
            };
            to_hz(m)
        })
        .collect();
    // np.linspace(0, sampling_rate // 2, bins)
    let top = f64::from(cfg.sampling_rate / 2);
    let fstep = (top - 0.0) / (bins - 1) as f64;
    let fft_freqs: Vec<f64> = (0..bins)
        .map(|i| if i == bins - 1 { top } else { i as f64 * fstep })
        .collect();
    let diff: Vec<f64> = filter_freqs.windows(2).map(|w| w[1] - w[0]).collect();
    let mut out = vec![0.0f64; bins * n_mel];
    for (f, &freq) in fft_freqs.iter().enumerate() {
        for m in 0..n_mel {
            let slope_lo = filter_freqs[m] - freq;
            let slope_hi = filter_freqs[m + 2] - freq;
            let down = -slope_lo / diff[m];
            let up = slope_hi / diff[m + 1];
            out[f * n_mel + m] = down.min(up).max(0.0);
        }
    }
    out
}

/// Frames of a clip of `samples` samples (after truncation) and how many of
/// them are real: `(frames, valid)`.
fn audio_frame_counts(cfg: &AudioFeatureConfig, samples: usize) -> (usize, usize) {
    let padded = samples.div_ceil(AUDIO_PAD_MULTIPLE) * AUDIO_PAD_MULTIPLE;
    let left = cfg.frame_length / 2;
    let size = cfg.frame_length + 1;
    let total = padded + left;
    let frames = if total >= size {
        (total - size) / cfg.hop_length + 1
    } else {
        0
    };
    // Frame i is real iff the left-padded sample at i * hop + frame_length
    // is a real one: i * hop + frame_length - left < samples.
    let end = cfg.frame_length - left;
    let valid = if samples > end {
        ((samples - end - 1) / cfg.hop_length + 1).min(frames)
    } else {
        0
    };
    (frames, valid)
}

/// The soft tokens (the placeholders of the processor's sequence) a clip of
/// `samples` mono samples at the extractor's rate makes: the true entries of
/// the tower's mask for the frames [`AudioFrontEnd::extract`] would build, after
/// the clip is cut to `max_samples`. It is that function's frame arithmetic
/// and [`crate::embedgemma2_audio::out_mask`], not a second formula, and no
/// feature is computed. 0 for a clip too short for one frame (the tower refuses
/// it).
pub fn audio_soft_tokens(cfg: &AudioFeatureConfig, samples: usize) -> usize {
    let kept = samples.min(cfg.max_samples.div_ceil(AUDIO_PAD_MULTIPLE) * AUDIO_PAD_MULTIPLE);
    let (frames, valid) = audio_frame_counts(cfg, kept);
    let mask: Vec<bool> = (0..frames).map(|i| i < valid).collect();
    out_mask(&mask).into_iter().filter(|&k| k).count()
}

/// The log-mel front end of `Gemma4AudioFeatureExtractor`, with its window,
/// filterbank and FFT built once.
pub struct AudioFrontEnd {
    cfg: AudioFeatureConfig,
    window: Vec<f32>,
    fft: RealFft,
    /// Per mel bin, the first FFT bin with a non-zero weight and the weights from there.
    filters: Vec<(usize, Vec<f64>)>,
}

impl AudioFrontEnd {
    pub fn new(cfg: &AudioFeatureConfig) -> Result<Self, String> {
        cfg.validate("AudioFrontEnd::new")?;
        let dense = htk_mel_filters(cfg);
        let bins = cfg.fft_length / 2 + 1;
        let filters = (0..cfg.mel_bins)
            .map(|m| {
                let column: Vec<f64> = (0..bins).map(|f| dense[f * cfg.mel_bins + m]).collect();
                match column.iter().position(|&w| w != 0.0) {
                    None => (0, Vec::new()),
                    Some(lo) => {
                        let hi = column.iter().rposition(|&w| w != 0.0).unwrap_or(lo);
                        (lo, column[lo..=hi].to_vec())
                    }
                }
            })
            .collect();
        Ok(Self {
            cfg: cfg.clone(),
            window: periodic_hann(cfg.frame_length),
            fft: RealFft::new(cfg.fft_length),
            filters,
        })
    }

    pub fn config(&self) -> &AudioFeatureConfig {
        &self.cfg
    }

    /// Frames of a clip of `samples` samples (after truncation) and how many
    /// of them are real: `(frames, valid)`.
    fn frame_counts(&self, samples: usize) -> (usize, usize) {
        audio_frame_counts(&self.cfg, samples)
    }

    /// The tower's input for one clip of mono f32 PCM at `sampling_rate`. The
    /// clip is cut to `max_samples`. A clip too short for one frame has none
    /// (the tower refuses a clip without frames).
    pub fn extract(&self, pcm: &[f32], sampling_rate: u32) -> Result<AudioInput, String> {
        const WHAT: &str = "AudioFrontEnd::extract";
        if sampling_rate != self.cfg.sampling_rate {
            return Err(format!(
                "{WHAT}: {sampling_rate} Hz audio; the extractor takes {} Hz and does not resample",
                self.cfg.sampling_rate
            ));
        }
        let kept = &pcm[..pcm
            .len()
            .min(self.cfg.max_samples.div_ceil(AUDIO_PAD_MULTIPLE) * AUDIO_PAD_MULTIPLE)];
        if let Some(at) = kept.iter().position(|v| !v.is_finite()) {
            return Err(format!("{WHAT}: sample {at} is {}", kept[at]));
        }
        let (frames, valid) = self.frame_counts(kept.len());
        let (hop, flen, left) = (self.cfg.hop_length, self.cfg.frame_length, self.cfg.frame_length / 2);
        let bins = self.cfg.fft_length / 2 + 1;
        let mels = self.cfg.mel_bins;
        let mut features = vec![0.0f32; frames * mels];
        let mask: Vec<bool> = (0..frames).map(|i| i < valid).collect();

        let work = |first: usize, block: &mut [f32]| {
            let mut x = vec![0.0f64; flen];
            let mut re = vec![0.0f64; self.cfg.fft_length / 2];
            let mut im = vec![0.0f64; self.cfg.fft_length / 2];
            let mut mag = vec![0.0f64; bins];
            for (r, row) in block.chunks_exact_mut(mels).enumerate() {
                let i = first + r;
                if i >= valid {
                    continue;
                }
                // Left-padded sample j is pcm[j - left]; padding beyond the clip is zero.
                for (k, v) in x.iter_mut().enumerate() {
                    let j = i * hop + k;
                    let s = if j >= left {
                        kept.get(j - left).copied().unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    *v = f64::from(s * self.window[k]);
                }
                self.fft.magnitudes(&x, &mut re, &mut im, &mut mag);
                for (o, (lo, taps)) in row.iter_mut().zip(&self.filters) {
                    let mel: f64 = taps.iter().zip(&mag[*lo..]).map(|(w, m)| w * m).sum();
                    *o = (mel + self.cfg.mel_floor).ln() as f32;
                }
            }
        };
        // About 4 microseconds a frame against tens for a thread: one thread per 24 frames.
        let threads = (frames / 24).clamp(
            1,
            std::thread::available_parallelism()
                .map_or(1, std::num::NonZeroUsize::get)
                .min(8),
        );
        if threads <= 1 || frames == 0 {
            work(0, &mut features);
        } else {
            let per = frames.div_ceil(threads);
            std::thread::scope(|scope| {
                for (b, block) in features.chunks_mut(per * mels).enumerate() {
                    let work = &work;
                    scope.spawn(move || work(b * per, block));
                }
            });
        }
        Ok(AudioInput {
            input_features: features,
            mask,
        })
    }
}

/// [`AudioFrontEnd::new`] and [`AudioFrontEnd::extract`] in one call. Build an
/// [`AudioFrontEnd`] once to preprocess many clips.
pub fn preprocess_audio(cfg: &AudioFeatureConfig, pcm: &[f32], sampling_rate: u32) -> Result<AudioInput, String> {
    AudioFrontEnd::new(cfg)?.extract(pcm, sampling_rate)
}

// ---------------------------------------------------------------------------
// Video
// ---------------------------------------------------------------------------

/// `np.linspace(0, n - 1, count, dtype=int)` for `count >= 2`: numpy's
/// `i * ((n - 1) / (count - 1))`, the last element forced to `n - 1`, floored.
fn linspace_indices(n: usize, count: usize) -> Vec<usize> {
    let stop = (n - 1) as f64;
    let step = stop / (count - 1) as f64;
    (0..count)
        .map(|i| if i == count - 1 { stop } else { i as f64 * step })
        .map(|v| v.floor() as usize)
        .collect()
}

/// `EmbeddingGemma2VideoProcessor.sample_frames` for a decoded video of
/// `total_frames` frames at `source_fps`, whose duration is `total_frames /
/// source_fps`: sample at the target fps (`min(total - 1, int(i * step))`),
/// then cap at `max_frames` by the overflow strategy.
pub fn sample_frame_indices(cfg: &VideoConfig, total_frames: usize, source_fps: f64) -> Result<Vec<usize>, String> {
    const WHAT: &str = "sample_frame_indices";
    if total_frames == 0 {
        return Err(format!("{WHAT}: the video has no frames"));
    }
    if !(source_fps.is_finite() && source_fps > 0.0) {
        return Err(format!("{WHAT}: source fps must be positive, got {source_fps}"));
    }
    let mut indices: Vec<usize> = match cfg.fps {
        None => (0..total_frames).collect(),
        Some(fps) => {
            let duration = total_frames as f64 / source_fps;
            let step = source_fps / fps;
            let sampled = ((duration * fps) as usize).max(1);
            (0..sampled)
                .map(|i| ((i as f64 * step) as usize).min(total_frames - 1))
                .collect()
        }
    };
    if indices.len() > cfg.max_frames {
        match cfg.overflow {
            Overflow::Truncate => indices.truncate(cfg.max_frames),
            Overflow::Uniform => {
                let n = indices.len();
                indices = if cfg.max_frames == 1 {
                    vec![indices[0]]
                } else {
                    linspace_indices(n, cfg.max_frames)
                        .into_iter()
                        .map(|i| indices[i])
                        .collect()
                };
            }
        }
    }
    Ok(indices)
}

/// `"mm:ss"` of a timestamp in seconds, as the processor formats it:
/// `int(s // 60)` and `int(s % 60)`.
pub fn timestamp_label(seconds: f64) -> String {
    let rest = seconds % 60.0;
    let minutes = ((seconds - rest) / 60.0).round();
    format!("{:02}:{:02}", minutes as i64, rest as i64)
}

/// What a video of `total_frames` frames of `height x width` at `source_fps`
/// becomes, without decoding or resizing anything: `(sampled frames, soft
/// tokens a frame)`. The frames are [`sample_frame_indices`]'s (every frame
/// when `do_sample_frames` is off, up to [`MAX_VIDEO_FRAMES`]) and the count a
/// frame makes is [`image_soft_tokens`] under the video's image settings, so the
/// video's placeholders number their product.
pub fn video_soft_tokens(
    cfg: &VideoConfig,
    height: usize,
    width: usize,
    total_frames: usize,
    source_fps: f64,
) -> Result<(usize, usize), String> {
    const WHAT: &str = "video_soft_tokens";
    if total_frames > MAX_VIDEO_FRAMES {
        return Err(format!(
            "{WHAT}: {total_frames} frames exceeds the limit of {MAX_VIDEO_FRAMES}"
        ));
    }
    let frames = if cfg.do_sample_frames {
        sample_frame_indices(cfg, total_frames, source_fps)?.len()
    } else {
        if total_frames == 0 {
            return Err(format!("{WHAT}: the video has no frames"));
        }
        if !(source_fps.is_finite() && source_fps > 0.0) {
            return Err(format!("{WHAT}: source fps must be positive, got {source_fps}"));
        }
        total_frames
    };
    Ok((frames, image_soft_tokens(&cfg.image, height, width)?))
}

/// A preprocessed video: the sampled frames and where they came from.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedVideo {
    /// One per sampled frame, in order.
    pub frames: Vec<PreparedImage>,
    /// Index of each sampled frame in the decoded video.
    pub indices: Vec<usize>,
    /// `index / source_fps`, seconds (`VideoMetadata.timestamps`).
    pub timestamps: Vec<f64>,
    /// Soft tokens per frame (the same for every frame).
    pub soft_tokens_per_frame: usize,
}

/// Preprocess a decoded video: sample frames ([`sample_frame_indices`]), then
/// resize, rescale and patchify each as an image at the video's budget. The
/// frames must be RGB and all the same size.
pub fn preprocess_video(cfg: &VideoConfig, frames: &[RawImage<'_>], source_fps: f64) -> Result<PreparedVideo, String> {
    const WHAT: &str = "preprocess_video";
    cfg.image.validate(WHAT)?;
    if frames.len() > MAX_VIDEO_FRAMES {
        return Err(format!(
            "{WHAT}: {} frames exceeds the limit of {MAX_VIDEO_FRAMES}",
            frames.len()
        ));
    }
    let indices = if cfg.do_sample_frames {
        sample_frame_indices(cfg, frames.len(), source_fps)?
    } else {
        if frames.is_empty() {
            return Err(format!("{WHAT}: the video has no frames"));
        }
        if !(source_fps.is_finite() && source_fps > 0.0) {
            return Err(format!("{WHAT}: source fps must be positive, got {source_fps}"));
        }
        (0..frames.len()).collect()
    };
    let first = frames[0];
    for (i, f) in frames.iter().enumerate() {
        if f.channels != 3 {
            return Err(format!(
                "{WHAT}: frame {i} has {} channels; decoded RGB frames expected",
                f.channels
            ));
        }
        if (f.height, f.width) != (first.height, first.width) {
            return Err(format!(
                "{WHAT}: frame {i} is {}x{}, frame 0 is {}x{}",
                f.height, f.width, first.height, first.width
            ));
        }
    }
    let (th, tw) = aspect_ratio_preserving_size(
        first.height,
        first.width,
        cfg.image.patch_size as usize,
        cfg.image.max_patches(),
        cfg.image.pooling_kernel as usize,
    )?;
    let weights = ((th, tw) != (first.height, first.width))
        .then(|| (AxisWeights::new(first.width, tw), AxisWeights::new(first.height, th)));
    let mut out = Vec::with_capacity(indices.len());
    for &i in &indices {
        let f = &frames[i];
        f.check(WHAT)?;
        out.push(prepare_image(&cfg.image, f.data, f.height, f.width, weights.as_ref())?);
    }
    let soft_tokens_per_frame = out[0].soft_tokens;
    Ok(PreparedVideo {
        timestamps: indices.iter().map(|&i| i as f64 / source_fps).collect(),
        frames: out,
        indices,
        soft_tokens_per_frame,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SNAPSHOT_CONFIG: &str = r#"{
      "audio_ms_per_token": 40, "audio_seq_length": 280,
      "feature_extractor": {"dither": 0.0, "feature_extractor_type": "Gemma4AudioFeatureExtractor",
        "feature_size": 128, "fft_length": 512, "fft_overdrive": false, "frame_length": 320,
        "hop_length": 160, "input_scale_factor": 1.0, "max_frequency": 8000.0, "mel_floor": 0.001,
        "min_frequency": 0.0, "padding_side": "right", "padding_value": 0.0, "per_bin_mean": null,
        "per_bin_stddev": null, "preemphasis": 0.0, "preemphasis_htk_flavor": true,
        "return_attention_mask": true, "sampling_rate": 16000},
      "image_processor": {"do_convert_rgb": true, "do_normalize": false, "do_rescale": true,
        "do_resize": true, "image_mean": [0.0, 0.0, 0.0], "image_processor_type": "Gemma4ImageProcessor",
        "image_seq_length": 280, "image_std": [1.0, 1.0, 1.0], "max_soft_tokens": 280, "patch_size": 16,
        "pooling_kernel_size": 3, "resample": 3, "rescale_factor": 0.00392156862745098},
      "video_processor": {"add_timestamps": false, "do_convert_rgb": true, "do_normalize": false,
        "do_rescale": true, "do_resize": true, "do_sample_frames": true, "fps": 1,
        "image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0], "max_frames": 32,
        "max_soft_tokens": 140, "overflow_strategy": "uniform", "patch_size": 16,
        "pooling_kernel_size": 3, "resample": 3, "rescale_factor": 0.00392156862745098,
        "return_metadata": false, "video_processor_type": "EmbeddingGemma2VideoProcessor"}
    }"#;

    #[test]
    fn the_snapshot_config_reads() {
        let c = PreConfig::from_json(SNAPSHOT_CONFIG).unwrap();
        assert_eq!(
            (c.image.patch_size, c.image.pooling_kernel, c.image.max_soft_tokens),
            (16, 3, 280)
        );
        assert_eq!(c.image.max_patches(), 2520);
        assert_eq!(c.image.patch_dim(), 768);
        assert_eq!(c.image.rescale_factor, 0.00392156862745098);
        assert_eq!(c.video.image.max_soft_tokens, 140);
        assert_eq!(
            (c.video.fps, c.video.max_frames, c.video.overflow),
            (Some(1.0), 32, Overflow::Uniform)
        );
        let a = &c.audio;
        assert_eq!(
            (a.sampling_rate, a.frame_length, a.hop_length, a.fft_length, a.mel_bins),
            (16000, 320, 160, 512, 128)
        );
        assert_eq!(
            (a.min_frequency, a.max_frequency, a.mel_floor, a.max_samples),
            (0.0, 8000.0, 1e-3, 480_000)
        );
    }

    #[test]
    fn config_refuses_what_it_does_not_implement() {
        for (from, to, why) in [
            (
                "\"do_normalize\": false, \"do_rescale\": true,\n        \"do_resize\": true, \"image_mean\"",
                "\"do_normalize\": true, \"do_rescale\": true,\n        \"do_resize\": true, \"image_mean\"",
                "normalize",
            ),
            (
                "\"resample\": 3, \"rescale_factor\": 0.00392156862745098},\n      \"video_processor\"",
                "\"resample\": 2, \"rescale_factor\": 0.00392156862745098},\n      \"video_processor\"",
                "resample",
            ),
            ("\"preemphasis\": 0.0", "\"preemphasis\": 0.97", "preemphasis"),
            ("\"dither\": 0.0", "\"dither\": 1e-4", "dither"),
            ("\"fft_overdrive\": false", "\"fft_overdrive\": true", "overdrive"),
            ("\"per_bin_mean\": null", "\"per_bin_mean\": [1.0]", "per_bin_mean"),
            (
                "\"padding_side\": \"right\"",
                "\"padding_side\": \"left\"",
                "padding_side",
            ),
            ("\"fft_length\": 512", "\"fft_length\": 1024", "fft_length"),
            ("\"hop_length\": 160", "\"hop_length\": 128", "hop_length"),
            (
                "\"overflow_strategy\": \"uniform\"",
                "\"overflow_strategy\": \"random\"",
                "overflow",
            ),
            ("\"max_soft_tokens\": 280", "\"max_soft_tokens\": 100", "soft tokens"),
            ("\"max_frames\": 32", "\"max_frames\": 0", "max_frames"),
        ] {
            assert!(SNAPSHOT_CONFIG.contains(from), "{why}: the edit target is missing");
            let bad = SNAPSHOT_CONFIG.replace(from, to);
            let err = PreConfig::from_json(&bad).expect_err(why);
            assert!(err.contains("processor_config.json"), "{why}: {err}");
        }
        assert!(PreConfig::from_json("[]").is_err());
        assert!(PreConfig::from_json("{}").unwrap_err().contains("image_processor"));
        // A trailing byte, a duplicate key and a non-integer count are the parser's refusals.
        assert!(PreConfig::from_json(&format!("{SNAPSHOT_CONFIG} x")).is_err());
        assert!(PreConfig::from_json(&SNAPSHOT_CONFIG.replace("\"patch_size\": 16,\n        \"pooling_kernel_size\": 3, \"resample\": 3, \"rescale_factor\": 0.00392156862745098},\n      \"video", "\"patch_size\": 16.5,\n        \"pooling_kernel_size\": 3, \"resample\": 3, \"rescale_factor\": 0.00392156862745098},\n      \"video")).is_err());
    }

    #[test]
    fn target_sizes_follow_the_processor() {
        // 224x224 and 640x360 are the reference directory's two images.
        assert_eq!(aspect_ratio_preserving_size(224, 224, 16, 2520, 3).unwrap(), (768, 768));
        assert_eq!(
            aspect_ratio_preserving_size(360, 640, 16, 2520, 3).unwrap(),
            (576, 1056)
        );
        // An already exact multiple that fits keeps its size only when it is the largest.
        assert_eq!(aspect_ratio_preserving_size(768, 768, 16, 2520, 3).unwrap(), (768, 768));
        // 1x1 grows to the largest square.
        assert_eq!(aspect_ratio_preserving_size(1, 1, 16, 2520, 3).unwrap(), (768, 768));
        // Extreme aspect ratios clamp the long side to max_side_length = 280 * 48.
        assert_eq!(aspect_ratio_preserving_size(1, 2000, 16, 2520, 3).unwrap(), (48, 13440));
        assert_eq!(aspect_ratio_preserving_size(2000, 1, 16, 2520, 3).unwrap(), (13440, 48));
        assert!(aspect_ratio_preserving_size(0, 5, 16, 2520, 3).is_err());
    }

    #[test]
    fn weights_are_a_partition_of_unity_at_the_chosen_precision() {
        for (n_in, n_out) in [(500, 288), (64, 48), (37, 96), (3000, 336), (1, 48), (768, 768)] {
            let a = AxisWeights::new(n_in, n_out);
            assert!(
                (1..=22).contains(&a.precision),
                "{n_in}->{n_out}: precision {}",
                a.precision
            );
            for i in 0..n_out {
                assert!(
                    a.start[i] + a.len[i] <= n_in,
                    "{n_in}->{n_out} tap {i} leaves the image"
                );
                let sum: i32 = a.w[a.off[i]..a.off[i] + a.len[i]].iter().sum();
                let one = 1i32 << a.precision;
                // Rounding each tap to the grid moves the sum by at most half a step per tap.
                assert!(
                    (sum - one).abs() <= a.len[i] as i32 / 2 + 1,
                    "{n_in}->{n_out} output {i}: sum {sum} of {one}"
                );
            }
        }
    }

    #[test]
    fn identity_resize_is_exact_and_constants_stay_constant() {
        let img: Vec<u8> = (0..7 * 5 * 3).map(|i| (i * 37 % 251) as u8).collect();
        assert_eq!(resize_rgb_u8(&img, 7, 5, 7, 5).unwrap(), img);
        for (th, tw) in [(3, 2), (14, 11), (1, 1), (50, 3)] {
            let flat = vec![123u8; 7 * 5 * 3];
            assert!(
                resize_rgb_u8(&flat, 7, 5, th, tw).unwrap().iter().all(|&v| v == 123),
                "{th}x{tw}"
            );
        }
        assert!(resize_rgb_u8(&img, 7, 5, 0, 5).is_err());
        assert!(resize_rgb_u8(&img[1..], 7, 5, 3, 3).is_err());
    }

    /// Inputs past the bounds are refused before any buffer is sized from them.
    #[test]
    fn oversized_inputs_are_refused_before_work() {
        let cfg = PreConfig::from_json(SNAPSHOT_CONFIG).unwrap();
        // An image with more than MAX_IMAGE_PIXELS pixels, whatever its data.
        let too_big = RawImage {
            width: 1 << 20,
            height: (MAX_IMAGE_PIXELS >> 20) + 1,
            channels: 3,
            data: &[],
        };
        assert!(preprocess_image(&cfg.image, &too_big).unwrap_err().contains("exceeds"));
        // A width pass that would need more than MAX_INTERMEDIATE_BYTES: 2^19 rows of a
        // 1-pixel-wide image grown to the extreme 13440-wide target.
        let across = AxisWeights::new(1, 13440);
        let down = AxisWeights::new(1 << 19, 48);
        let err = resize_with(&[], 1 << 19, 1, 48, 13440, &across, &down).unwrap_err();
        assert!(err.contains("width pass"), "{err}");
        // More decoded frames than MAX_VIDEO_FRAMES.
        let px = [0u8; 3];
        let frames = vec![RawImage::rgb(1, 1, &px); MAX_VIDEO_FRAMES + 1];
        assert!(preprocess_video(&cfg.video, &frames, 30.0)
            .unwrap_err()
            .contains("frames exceeds"));
        // Degenerate sizes.
        assert!(AudioFrontEnd::new(&AudioFeatureConfig {
            fft_length: 300,
            ..cfg.audio.clone()
        })
        .is_err());
        assert!(AudioFrontEnd::new(&AudioFeatureConfig {
            max_frequency: 9000.0,
            ..cfg.audio.clone()
        })
        .is_err());
        assert!(AudioFrontEnd::new(&AudioFeatureConfig {
            mel_floor: 0.0,
            ..cfg.audio.clone()
        })
        .is_err());
    }

    #[test]
    fn rgb_conversion_follows_pil() {
        let gray = RawImage {
            width: 2,
            height: 1,
            channels: 1,
            data: &[10, 20],
        };
        assert_eq!(&*gray.into_rgb(), &[10, 10, 10, 20, 20, 20]);
        let la = RawImage {
            width: 2,
            height: 1,
            channels: 2,
            data: &[10, 99, 20, 1],
        };
        assert_eq!(&*la.into_rgb(), &[10, 10, 10, 20, 20, 20]);
        let rgba = RawImage {
            width: 2,
            height: 1,
            channels: 4,
            data: &[1, 2, 3, 0, 4, 5, 6, 255],
        };
        assert_eq!(&*rgba.into_rgb(), &[1, 2, 3, 4, 5, 6]);
        assert!(RawImage {
            width: 2,
            height: 1,
            channels: 5,
            data: &[0; 10]
        }
        .check("t")
        .is_err());
        assert!(RawImage {
            width: 2,
            height: 1,
            channels: 3,
            data: &[0; 5]
        }
        .check("t")
        .is_err());
        assert!(RawImage {
            width: 0,
            height: 1,
            channels: 3,
            data: &[]
        }
        .check("t")
        .is_err());
    }

    #[test]
    fn patches_carry_positions_and_zero_padding() {
        let cfg = ImageConfig {
            patch_size: 2,
            pooling_kernel: 2,
            max_soft_tokens: 70,
            rescale_factor: 1.0 / 255.0,
        };
        // A 4x8 (HxW) ramp is exact: 2x4 patches, 2 soft tokens.
        let data: Vec<u8> = (0..4 * 8 * 3).map(|i| i as u8).collect();
        // 4x8 resizes to the target of its own area, so build the patchify input directly.
        let p = patchify(&data, 4, 8, &cfg);
        assert_eq!((p.patches, p.soft_tokens, p.resized), (8, 2, (4, 8)));
        assert_eq!(p.pixel_values.len(), 280 * 12);
        assert_eq!(&p.position_ids[..4], &[0, 0, 1, 0]);
        assert_eq!(&p.position_ids[2 * 4..2 * 4 + 2], &[0, 1]);
        assert_eq!(&p.position_ids[2 * 8..2 * 8 + 2], &[-1, -1]);
        // Patch 1 (x = 1, y = 0): rows 0 and 1, columns 2 and 3, (row, col, channel) order.
        let want: Vec<f32> = [6u8, 7, 8, 9, 10, 11, 30, 31, 32, 33, 34, 35]
            .iter()
            .map(|&v| f32::from(v) * (1.0f32 / 255.0))
            .collect();
        assert_eq!(&p.pixel_values[12..24], &want[..]);
        assert!(p.pixel_values[8 * 12..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn real_fft_matches_the_dft() {
        let n = 64;
        let x: Vec<f64> = (0..n).map(|i| ((i * i + 3 * i) % 17) as f64 / 8.0 - 1.0).collect();
        let fft = RealFft::new(n);
        let (mut re, mut im, mut out) = (vec![0.0; n / 2], vec![0.0; n / 2], vec![0.0; n / 2 + 1]);
        fft.magnitudes(&x, &mut re, &mut im, &mut out);
        for (k, &got) in out.iter().enumerate() {
            let (mut sr, mut si) = (0.0, 0.0);
            for (t, &v) in x.iter().enumerate() {
                let a = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                sr += v * a.cos();
                si += v * a.sin();
            }
            assert!((got - sr.hypot(si)).abs() < 1e-12, "bin {k}: {got} vs {}", sr.hypot(si));
        }
        // A short input is zero-padded.
        fft.magnitudes(&x[..10], &mut re, &mut im, &mut out);
        let mut padded = x[..10].to_vec();
        padded.resize(n, 0.0);
        let mut want = vec![0.0; n / 2 + 1];
        fft.magnitudes(&padded, &mut re, &mut im, &mut want);
        assert_eq!(out, want);
    }

    #[test]
    fn frame_counts_follow_the_extractor() {
        let cfg = PreConfig::from_json(SNAPSHOT_CONFIG).unwrap().audio;
        let fe = AudioFrontEnd::new(&cfg).unwrap();
        // (samples, frames, valid): the reference clips 16000 -> 99, 52800 -> 330 (329 real).
        // 129..=160 samples pad to 256: one frame, whose last sample is padding, so not real.
        for (n, frames, valid) in [
            (0, 0, 0),
            (1, 0, 0),
            (100, 0, 0),
            (128, 0, 0),
            (129, 1, 0),
            (160, 1, 0),
            (161, 1, 1),
            (192, 1, 1),
            (16000, 99, 99),
            (52800, 330, 329),
            (480000, 2999, 2999),
        ] {
            assert_eq!(fe.frame_counts(n), (frames, valid), "{n} samples");
        }
        let one = fe.extract(&vec![0.25; 192], 16000).unwrap();
        assert_eq!((one.mask.len(), one.input_features.len()), (1, 128));
        assert!(fe.extract(&[0.0; 10], 8000).unwrap_err().contains("does not resample"));
        assert!(fe.extract(&[0.0, f32::NAN], 16000).unwrap_err().contains("NaN"));
        let long = fe.extract(&vec![0.1; 480_000 + 5000], 16000).unwrap();
        assert_eq!(long.mask.len(), 2999, "a clip past max_samples is cut");
    }

    /// The extractor's `_truncate` rounds `max_length` up to a multiple of 128 before it
    /// cuts, so a 4000-sample limit keeps 4096 real samples: 25 frames, all real.
    #[test]
    fn the_sample_limit_rounds_up_to_the_pad_multiple() {
        let mut cfg = PreConfig::from_json(SNAPSHOT_CONFIG).unwrap().audio;
        cfg.max_samples = 4000;
        let a = preprocess_audio(&cfg, &vec![0.2; 6000], 16000).unwrap();
        assert_eq!((a.mask.len(), a.mask.iter().filter(|&&m| m).count()), (25, 25));
        let b = preprocess_audio(&cfg, &vec![0.2; 4001], 16000).unwrap();
        assert_eq!((b.mask.len(), b.mask.iter().filter(|&&m| m).count()), (25, 25));
        let c = preprocess_audio(&cfg, &vec![0.2; 4000], 16000).unwrap();
        assert_eq!((c.mask.len(), c.mask.iter().filter(|&&m| m).count()), (25, 24));
    }

    #[test]
    fn silence_is_the_log_of_the_floor_and_padded_frames_are_zero() {
        let cfg = PreConfig::from_json(SNAPSHOT_CONFIG).unwrap().audio;
        let fe = AudioFrontEnd::new(&cfg).unwrap();
        // 900 samples pad to 1024: six frames, the last of which ends in padding.
        let a = fe.extract(&vec![0.0; 900], 16000).unwrap();
        let floor = (1e-3f64).ln() as f32;
        let real = a.mask.iter().filter(|&&m| m).count();
        assert!(real > 0 && real < a.mask.len());
        for (i, row) in a.input_features.chunks_exact(128).enumerate() {
            let want = if a.mask[i] { floor } else { 0.0 };
            assert!(row.iter().all(|&v| v == want), "frame {i}");
        }
    }

    #[test]
    fn mel_filters_are_triangles_with_the_documented_zero_first_filter() {
        let cfg = PreConfig::from_json(SNAPSHOT_CONFIG).unwrap().audio;
        let dense = htk_mel_filters(&cfg);
        let col = |m: usize| (0..257).map(|f| dense[f * 128 + m]).collect::<Vec<f64>>();
        assert!(dense.iter().all(|&w| (0.0..=1.0).contains(&w)));
        // 128 filters over 257 bins: the lowest falls between two bins and is all zero
        // (transformers warns about it and keeps it); the next few have a single bin.
        assert!(col(0).iter().all(|&w| w == 0.0));
        let nonzero = |m: usize| col(m).iter().filter(|&&w| w > 0.0).count();
        assert_eq!((nonzero(1), nonzero(2)), (1, 1));
        assert!(nonzero(127) > 4 && nonzero(64) > 1);
    }

    fn video_cfg() -> VideoConfig {
        PreConfig::from_json(SNAPSHOT_CONFIG).unwrap().video
    }

    #[test]
    fn frame_sampling_follows_the_processor() {
        let cfg = video_cfg();
        // 10 s at 30 fps: one frame per second, the first of each.
        let idx = sample_frame_indices(&cfg, 300, 30.0).unwrap();
        assert_eq!(idx, (0..10).map(|s| s * 30).collect::<Vec<_>>());
        // Under one second still yields a frame.
        assert_eq!(sample_frame_indices(&cfg, 5, 30.0).unwrap(), vec![0]);
        // 100 s: 100 sampled, capped to 32 uniformly, first and last kept.
        let idx = sample_frame_indices(&cfg, 3000, 30.0).unwrap();
        assert_eq!((idx.len(), idx[0], idx[31]), (32, 0, 99 * 30));
        assert!(idx.windows(2).all(|w| w[0] < w[1]));
        // Truncate keeps the first 32.
        let t = VideoConfig {
            overflow: Overflow::Truncate,
            ..cfg.clone()
        };
        assert_eq!(
            sample_frame_indices(&t, 3000, 30.0).unwrap(),
            (0..32).map(|s| s * 30).collect::<Vec<_>>()
        );
        // No fps: every frame, then the cap.
        let all = VideoConfig {
            fps: None,
            ..cfg.clone()
        };
        assert_eq!(sample_frame_indices(&all, 7, 25.0).unwrap(), (0..7).collect::<Vec<_>>());
        assert_eq!(sample_frame_indices(&all, 33, 25.0).unwrap().len(), 32);
        assert!(sample_frame_indices(&cfg, 0, 30.0).is_err());
        assert!(sample_frame_indices(&cfg, 10, 0.0).is_err());
        assert!(sample_frame_indices(&cfg, 10, f64::NAN).is_err());
    }

    #[test]
    fn timestamps_are_formatted_as_mm_ss() {
        assert_eq!(timestamp_label(0.0), "00:00");
        assert_eq!(timestamp_label(59.9), "00:59");
        assert_eq!(timestamp_label(60.0), "01:00");
        assert_eq!(timestamp_label(3725.5), "62:05");
    }

    #[test]
    fn a_video_is_sampled_resized_and_timestamped() {
        let cfg = VideoConfig {
            image: ImageConfig {
                patch_size: 2,
                pooling_kernel: 2,
                max_soft_tokens: 70,
                rescale_factor: 1.0 / 255.0,
            },
            ..video_cfg()
        };
        let frames: Vec<Vec<u8>> = (0..9)
            .map(|f| (0..6 * 8 * 3).map(|i| (i * 7 + f * 13) as u8).collect())
            .collect();
        let raw: Vec<RawImage<'_>> = frames.iter().map(|d| RawImage::rgb(8, 6, d)).collect();
        // 9 frames at 3 fps = 3 s: frames 0, 3, 6.
        let v = preprocess_video(&cfg, &raw, 3.0).unwrap();
        assert_eq!(v.indices, vec![0, 3, 6]);
        assert_eq!(v.timestamps, vec![0.0, 1.0, 2.0]);
        assert_eq!(v.frames.len(), 3);
        assert_eq!(v.soft_tokens_per_frame, v.frames[0].soft_tokens);
        // Each frame is what the image path gives that frame.
        for (f, &i) in v.frames.iter().zip(&v.indices) {
            assert_eq!(*f, preprocess_image(&cfg.image, &raw[i]).unwrap());
        }
        // Mixed sizes and non-RGB frames are refused.
        let small = vec![0u8; 4 * 4 * 3];
        let mut mixed = raw.clone();
        mixed[4] = RawImage::rgb(4, 4, &small);
        assert!(preprocess_video(&cfg, &mixed, 3.0).unwrap_err().contains("frame 4"));
        let gray = vec![0u8; 6 * 8];
        let mut g = raw.clone();
        g[1] = RawImage {
            width: 8,
            height: 6,
            channels: 1,
            data: &gray,
        };
        assert!(preprocess_video(&cfg, &g, 3.0).unwrap_err().contains("channels"));
        // do_sample_frames off processes every frame.
        let every = VideoConfig {
            do_sample_frames: false,
            ..cfg
        };
        assert_eq!(preprocess_video(&every, &raw, 3.0).unwrap().frames.len(), 9);
    }
}
