//! Files and bytes to the towers' inputs: [`crate::media`]'s decoding followed by
//! [`crate::embedgemma2_pre`]'s preprocessing, in one call per medium.
//!
//! ```no_run
//! use std::path::Path;
//! use tessl::embedgemma2_media::{prepare_image, MediaSource};
//! use tessl::embedgemma2_pre::PreConfig;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = PreConfig::from_snapshot(Path::new("/path/to/snapshot"))?;
//! let image = prepare_image(&cfg.image, MediaSource::Path(Path::new("cat.png")))?;
//! # let _ = image.input();
//! # Ok(())
//! # }
//! ```
//!
//! Each function decodes with the defaults the reference does (an image's RGB
//! samples as stored with the EXIF orientation ignored, as torchvision reads a
//! path; audio at the extractor's rate as the channel mean; a video's upright
//! frames), preprocesses at once and drops the decoded pixels, so a caller that
//! prepares many files holds one decoded medium at a time and the prepared
//! inputs (bounded by the vision tower's per-call limits). Every failure is a
//! [`MediaError`]: a decoder's own, or [`MediaError::Preprocess`] when the
//! decoded media is refused by preprocessing.

use std::path::Path;

use crate::embedgemma2_audio::AudioInput;
use crate::embedgemma2_pre::{
    preprocess_image, sample_frame_indices, AudioFrontEnd, ImageConfig, PreparedImage, PreparedVideo, VideoConfig,
};
use crate::media::audio::{decode_audio_bytes, decode_audio_file, AudioContainer, AudioOptions};
use crate::media::image::{decode_image, decode_image_file, ImageOptions};
use crate::media::video::{preprocess_decoded_video, VideoOptions, VideoSource};
use crate::media::MediaError;

/// Where a medium comes from.
#[derive(Clone, Copy, Debug)]
pub enum MediaSource<'a> {
    /// A file.
    Path(&'a Path),
    /// A whole encoded file in memory.
    Bytes(&'a [u8]),
}

/// An image file or its bytes, decoded ([`ImageOptions::default`]) and
/// preprocessed ([`preprocess_image`]).
pub fn prepare_image(cfg: &ImageConfig, source: MediaSource<'_>) -> Result<PreparedImage, MediaError> {
    let options = ImageOptions::default();
    let decoded = match source {
        MediaSource::Path(path) => decode_image_file(path, &options)?,
        MediaSource::Bytes(bytes) => decode_image(bytes, &options)?,
    };
    preprocess_image(cfg, &decoded.raw()).map_err(MediaError::Preprocess)
}

/// An audio file or its bytes, decoded to mono at the extractor's rate
/// ([`AudioOptions::default`] at `front_end`'s sampling rate: the channel
/// mean, at most 30 minutes) and turned into the audio tower's input
/// ([`AudioFrontEnd::extract`], which cuts the clip to the extractor's 30 s).
pub fn prepare_audio(front_end: &AudioFrontEnd, source: MediaSource<'_>) -> Result<AudioInput, MediaError> {
    let rate = front_end.config().sampling_rate;
    let options = AudioOptions {
        sample_rate: rate,
        ..AudioOptions::default()
    };
    let decoded = match source {
        MediaSource::Path(path) => decode_audio_file(path, &options)?,
        MediaSource::Bytes(bytes) => decode_audio_bytes(bytes, AudioContainer::Auto, &options)?,
    };
    front_end
        .extract(&decoded.samples, rate)
        .map_err(MediaError::Preprocess)
}

/// A video file whose first pass has run (frame count, rate, timestamps) and
/// whose pixels have not been decoded: the point where a caller learns how many
/// pictures the video would make and can refuse before decoding any of them.
pub struct OpenedVideo {
    source: VideoSource,
}

impl OpenedVideo {
    /// Open a video file, or video bytes (written to a private temporary file,
    /// removed on drop; `extension`, such as `"mp4"` or `"mov"`, is the type
    /// AVFoundation sniffs it as and is ignored for a path).
    pub fn open(source: MediaSource<'_>, extension: &str) -> Result<Self, MediaError> {
        let options = VideoOptions::default();
        let source = match source {
            MediaSource::Path(path) => VideoSource::open(path, &options)?,
            MediaSource::Bytes(bytes) => VideoSource::open_bytes(bytes, extension, &options)?,
        };
        Ok(Self { source })
    }

    /// The video's frames in all and its source rate.
    pub fn frames_and_rate(&self) -> (usize, f64) {
        let info = self.source.info();
        (info.total_frames, info.fps)
    }

    /// How many frames [`Self::prepare`] would decode and preprocess under
    /// `cfg`: the sampler's ([`sample_frame_indices`]), or every frame when the
    /// video processor does not sample.
    pub fn sampled_frames(&self, cfg: &VideoConfig) -> Result<usize, MediaError> {
        let (total, fps) = self.frames_and_rate();
        if cfg.do_sample_frames {
            Ok(sample_frame_indices(cfg, total, fps)
                .map_err(MediaError::Preprocess)?
                .len())
        } else {
            Ok(total)
        }
    }

    /// Decode the sampled frames and preprocess them, with the true source
    /// indices and `index / fps` timestamps ([`preprocess_decoded_video`]).
    pub fn prepare(&self, cfg: &VideoConfig) -> Result<PreparedVideo, MediaError> {
        let decoded = self.source.decode_sampled(cfg)?;
        preprocess_decoded_video(cfg, &decoded)
    }
}
