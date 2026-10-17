//! EmbeddingGemma 2 as the checkpoint ships it: the text encoder and both
//! towers behind one object, one snapshot and one call.
//!
//! [`EmbedGemma2Multimodal`] is a composition, not a fourth implementation: it
//! owns an [`EmbedGemma2Model`], an [`EmbedGemma2VisionModel`] and an
//! [`EmbedGemma2Audio`] loaded from one `config.json` and one
//! `model.safetensors`, and [`EmbedGemma2Multimodal::encode`] runs what the
//! three already do, in the order the checkpoint's reference does:
//!
//! 1. every image and video frame of the call through the vision tower and
//!    every clip through the audio tower (the soft tokens that replace an
//!    item's placeholder ids);
//! 2. the items' ids, with those soft tokens in place of their placeholders,
//!    through the text encoder ([`EmbedGemma2Model::encode_multimodal`]):
//!    mean pool and normalize, one row per item, in the order given.
//!
//! It lives in its own module beside [`crate::embedgemma2`] (which stays
//! text-only, so the C ABI's text entries do not load 1.2 GB of towers) and
//! named as the tests are (`embedgemma2_mm_*`).
//!
//! # Items
//!
//! An [`MmItem`] is one row of the output:
//!
//! * [`MmItem::Text`]: token ids with the tokenizer's `<bos>`/`<eos>` and any
//!   prompt already in them. A placeholder id (`image_token_id`,
//!   `audio_token_id`, `video_token_id`) in a text item is refused: in a
//!   multimodal call it would be replaced by a soft token it does not have.
//! * [`MmItem::Image`] and [`MmItem::Audio`]: one image (the processor's
//!   `pixel_values [P, 768]` and `image_position_ids [P, 2]`) or one clip
//!   (`input_features [F, 128]` and its mask). The sequence is the processor's:
//!   [`crate::embedgemma2::EmbedGemma2Config::image_sequence`] or
//!   [`audio_sequence`](crate::embedgemma2::EmbedGemma2Config::audio_sequence)
//!   of the tower's soft-token count.
//! * [`MmItem::Video`]: one video as its sampled frames, each the processor's
//!   patch tensors (what [`crate::embedgemma2_pre::preprocess_video`] makes).
//!   Every frame goes through the vision tower, in the same batch as the call's
//!   images, and its soft tokens are the rows of one `<boi> <video> x n <eoi>`
//!   block; the sequence is
//!   [`crate::embedgemma2::EmbedGemma2Config::video_sequence`] of the frames'
//!   soft-token counts, which must be equal, as the processor's single
//!   `num_soft_tokens_per_video` has it.
//! * [`MmItem::TimestampedVideo`]: the same frames laid out under the
//!   processor's `add_timestamps = true`: each block is preceded by its
//!   `mm:ss` time, tokenized by the [`Tokenizer`] the item carries
//!   ([`crate::embedgemma2::EmbedGemma2Config::video_sequence_with_timestamps`]).
//!   `PreConfig`'s `video.add_timestamps` says which of the two items a
//!   processor configured like the snapshot's wants.
//! * [`MmItem::Media`]: ids a caller built (a prompt around an image, an
//!   image and a clip in one sequence, timestamps between a video's frames)
//!   with the media their placeholders stand for. The `j`-th image placeholder
//!   takes the `j`-th soft row of the item's images laid end to end, the video
//!   placeholders likewise from the frames of its videos and the audio
//!   placeholders from its clips, as transformers' `masked_scatter` does per
//!   modality. The counts must match exactly.
//!
//! # Refusals
//!
//! Everything a call can be refused for is checked before the first
//! allocation or dispatch: the towers' input checks
//! ([`EmbedGemma2VisionModel::check_images`], [`EmbedGemma2Audio::check_clips`]),
//! the text side's (`EmbedGemma2Model::check_sequences`: lengths, ids,
//! `truncate_dim`, the batch size) and the facade's own (placeholder counts
//! against soft-token counts). A refused call allocates nothing and waits for
//! nothing. Media are numbered across the call in item order in the towers'
//! messages (`image 2` is the third image of the call, not of its item), and a
//! video's frames are images to the vision tower: they are numbered with the
//! images, in the order of the items, and count toward its per-call bounds
//! ([`crate::embedgemma2_vision::MAX_CALL_IMAGES`] pictures and
//! [`crate::embedgemma2_vision::MAX_CALL_PATCHES`] padded patches; the clips
//! likewise toward [`crate::embedgemma2_audio::MAX_CALL_CLIPS`] and
//! [`crate::embedgemma2_audio::MAX_CALL_FRAMES`]). A call of more than one
//! tower forward's worth of pictures (256, [`crate::embedgemma2_vision::MAX_IMAGES`])
//! or clips ([`crate::embedgemma2_audio::MAX_CLIPS`]) is cut into several
//! forwards inside the same call, behind the same one wait, with the soft rows
//! still read from the device by the text encoder's scatter.
//!
//! # Waiting
//!
//! A call waits for the GPU **once**, with or without media. The towers'
//! forwards ([`EmbedGemma2VisionModel::queue_images`],
//! [`EmbedGemma2Audio::queue_audio`]) and then the text encoder's are queued
//! under one [`crate::runtime::GpuRuntime::async_scope`] behind one wait. The
//! towers' soft tokens stay on the device: each item's soft rows are described
//! as runs of rows of the towers' output buffers (a vision image's rows are one
//! run; an audio clip's are one run per maximal stretch its output mask keeps)
//! and the text encoder's scatter copies them over the placeholder rows where
//! the towers wrote them. Nothing is compacted: the same runs a host caller's
//! rows make are what the scatter takes. The placeholder counts are known before
//! any tower runs ([`EmbedGemma2VisionModel::check_images`],
//! [`EmbedGemma2Audio::check_clips`]), so the layout is checked first and every
//! refusal still happens before any GPU work.
//!
//! A host caller's soft rows are checked finite before any GPU work; the
//! towers' are not read back before the text encoder runs, so a media item's
//! embedding is checked finite after the wait instead (an overflowing tower is
//! an error naming the item, not a NaN row).
//!
//! # Precision
//!
//! One [`GemmOperands`] ([`EmbedGemma2Multimodal::load`]'s `operands`)
//! configures the **text encoder, the vision tower and the audio tower** alike.
//! Under `Bf16` all three keep their projections as the checkpoint stores them
//! and run every GEMM with bf16 operands and f32 accumulation; the text
//! encoder's and the vision tower's attention runs on bf16 tiles, the audio
//! tower's (chunked, with relative-position logits and a softcap, which the
//! tile kernel does not have) stays its f32 kernel and stores its output bf16;
//! the residual streams, the norms, the convolutions, the pooling and the soft
//! tokens stay f32, and the soft tokens enter the text encoder as its
//! embeddings do and are rounded where its GEMMs round them. It is one flag
//! because the models make one precision claim to the caller, and a mixed set
//! (a bf16 tower into an exact text model, or the reverse) is a different
//! forward that a caller composing the models by hand
//! ([`EmbedGemma2Multimodal::from_parts`]) can still build.
//!
//! Under `Bf16` no GEMM of any model is an f32 one, so a call accepts a
//! runtime whose relaxed f32 GEMMs are on; under `ExactF32` every call that
//! runs a model refuses it.

use std::borrow::Cow;
use std::io::Read;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use crate::embedgemma2::{
    EmbedGemma2Config, EmbedGemma2Model, EncodeOutput, MmTokens, Seq, SoftSegment, SoftSource, MAX_BATCH,
};
use crate::embedgemma2_audio::{AudioConfig, AudioInput, EmbedGemma2Audio, PendingAudio};
use crate::embedgemma2_tok::Tokenizer;
use crate::embedgemma2_vision::{EmbedGemma2VisionModel, ImageInput, PendingImages, VisionConfig};
use crate::gemm::GemmOperands;
use crate::runtime::GpuRuntime;
use crate::safetensors::SafeTensors;

/// The tensor-name prefix of the text encoder in `google/embeddinggemma-2`'s
/// `model.safetensors`. (The towers' prefixes, `vision_tower.`, `embed_vision.`,
/// `audio_tower.` and `embed_audio.`, are theirs.)
pub const TEXT_PREFIX: &str = "language_model.";

/// Largest `config.json` [`read_config_text`] reads, in bytes (the
/// checkpoint's is a few KiB).
pub const MAX_CONFIG_BYTES: u64 = 1 << 20;

/// `config.json` as text, after a bounded read of a regular file (not a FIFO,
/// not a directory); a file past [`MAX_CONFIG_BYTES`] is refused rather than
/// read whole.
pub fn read_config_text(path: &Path) -> Result<String, String> {
    let (file, len) = crate::plain::open_regular_file(path)?;
    if len > MAX_CONFIG_BYTES {
        return Err(format!(
            "{}: {len} bytes, above the {MAX_CONFIG_BYTES}-byte config cap",
            path.display()
        ));
    }
    let mut text = String::new();
    file.take(MAX_CONFIG_BYTES)
        .read_to_string(&mut text)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(text)
}

/// One item of an [`EmbedGemma2Multimodal::encode`] call: one output row.
#[derive(Clone, Copy, Debug)]
pub enum MmItem<'a> {
    /// Token ids with no placeholders.
    Text(&'a [u32]),
    /// One image, in the processor's sequence layout.
    Image(ImageInput<'a>),
    /// One clip, in the processor's sequence layout.
    Audio(&'a AudioInput),
    /// One video, in the processor's sequence layout
    /// ([`crate::embedgemma2::EmbedGemma2Config::video_sequence`]): its sampled
    /// frames, each as the processor's patch tensors, in order. Every frame
    /// must make the same number of soft tokens, as the processor's single
    /// `num_soft_tokens_per_video` has it.
    Video(&'a [ImageInput<'a>]),
    /// One video laid out as the processor does under `add_timestamps = true`
    /// ([`crate::embedgemma2::EmbedGemma2Config::video_sequence_with_timestamps`]):
    /// every frame's block is preceded by its `mm:ss` time, tokenized by
    /// `tokenizer`, the blocks joined by a space. `timestamps[f]` is frame
    /// `f`'s time in seconds (`PreparedVideo::timestamps`); there is one per
    /// frame. The frames go through the vision tower as [`MmItem::Video`]'s
    /// do, and must make the same number of soft tokens.
    TimestampedVideo {
        /// The sampled frames, in order.
        frames: &'a [ImageInput<'a>],
        /// Each frame's time in seconds.
        timestamps: &'a [f64],
        /// The snapshot's tokenizer, for the labels' ids.
        tokenizer: &'a Tokenizer,
    },
    /// Caller-built ids and the media their placeholders stand for.
    Media {
        /// The ids, placeholders included.
        ids: &'a [u32],
        /// The images whose soft tokens fill the image placeholders, in order.
        images: &'a [ImageInput<'a>],
        /// The clips whose soft tokens fill the audio placeholders, in order.
        audio: &'a [AudioInput],
        /// The videos, each as its frames, whose soft tokens fill the video
        /// placeholders, the frames of all of them laid end to end in order
        /// (what a caller needs to place timestamps between frames: the ids
        /// are theirs).
        videos: &'a [&'a [ImageInput<'a>]],
    },
}

/// Where an item's media sit in the call's flat picture and clip lists. The
/// pictures are the vision tower's inputs: an item's images, then the frames
/// of its videos.
struct Span {
    images: Range<usize>,
    frames: Range<usize>,
    clips: Range<usize>,
}

/// The text encoder and both towers of one snapshot. See the module docs.
pub struct EmbedGemma2Multimodal {
    rt: Arc<GpuRuntime>,
    text: EmbedGemma2Model,
    vision: EmbedGemma2VisionModel,
    audio: EmbedGemma2Audio,
    tokens: MmTokens,
}

impl EmbedGemma2Multimodal {
    /// Load from a Hugging Face snapshot directory: its `config.json` (read
    /// once, bounded) and its single-file `model.safetensors` (opened once),
    /// all three models from them. `operands` is the text encoder's
    /// ([`EmbedGemma2Model::load_with`]), the vision tower's
    /// ([`EmbedGemma2VisionModel::load_with`]) and the audio tower's
    /// ([`EmbedGemma2Audio::load_with`]).
    pub fn load(rt: &Arc<GpuRuntime>, snapshot_dir: &Path, operands: GemmOperands) -> Result<Self, String> {
        let config = read_config_text(&snapshot_dir.join("config.json"))?;
        let st = SafeTensors::open(&snapshot_dir.join("model.safetensors"))?;
        Self::load_from(rt, &st, &config, operands)
    }

    /// [`Self::load`] from an open checkpoint and the text of its
    /// `config.json`. The three configs are parsed from the same text, the
    /// text encoder reads `language_model.*` ([`TEXT_PREFIX`]), the towers
    /// theirs, and each refuses a tensor of its own that its forward does not
    /// read.
    pub fn load_from(
        rt: &Arc<GpuRuntime>,
        st: &SafeTensors,
        config_json: &str,
        operands: GemmOperands,
    ) -> Result<Self, String> {
        let text_cfg = EmbedGemma2Config::from_config_json(config_json)?;
        let vision_cfg = VisionConfig::from_config_json(config_json)?;
        let audio_cfg = AudioConfig::from_config_json(config_json)?;
        // Refuse a config the composition cannot serve before reading 1.2 GB.
        text_cfg.special_tokens.multimodal("EmbedGemma2Multimodal::load")?;
        let text = EmbedGemma2Model::load_with(rt, st, TEXT_PREFIX, text_cfg, operands)?;
        let vision = EmbedGemma2VisionModel::load_with(rt, st, vision_cfg, operands)?;
        let audio = EmbedGemma2Audio::load_with(rt, st, "", audio_cfg, operands)?;
        Self::from_parts(text, vision, audio)
    }

    /// Compose already-loaded models. The towers' soft-token width must be the
    /// text encoder's hidden size, and the text config must carry the
    /// multimodal token ids ([`crate::embedgemma2::SpecialTokens::multimodal`]);
    /// all three must share one runtime.
    pub fn from_parts(
        text: EmbedGemma2Model,
        vision: EmbedGemma2VisionModel,
        audio: EmbedGemma2Audio,
    ) -> Result<Self, String> {
        const WHAT: &str = "EmbedGemma2Multimodal::from_parts";
        let hidden = text.config().hidden;
        if vision.config().text_hidden != hidden {
            return Err(format!(
                "{WHAT}: the vision tower's soft tokens are {} wide, the text encoder's hidden size is {hidden}",
                vision.config().text_hidden
            ));
        }
        if audio.config().embed_dim != hidden {
            return Err(format!(
                "{WHAT}: the audio tower's soft tokens are {} wide, the text encoder's hidden size is {hidden}",
                audio.config().embed_dim
            ));
        }
        let rt = text.runtime();
        if !Arc::ptr_eq(rt, vision.runtime()) || !Arc::ptr_eq(rt, audio.runtime()) {
            return Err(format!("{WHAT}: the three models were loaded on different runtimes"));
        }
        let tokens = text.config().special_tokens.multimodal(WHAT)?;
        Ok(Self {
            rt: Arc::clone(rt),
            text,
            vision,
            audio,
            tokens,
        })
    }

    /// The text encoder (its config, [`EmbedGemma2Model::operands`], limits).
    pub fn text(&self) -> &EmbedGemma2Model {
        &self.text
    }

    /// The text encoder, for [`EmbedGemma2Model::set_max_tokens`] and
    /// [`EmbedGemma2Model::set_ple_run_bytes`].
    pub fn text_mut(&mut self) -> &mut EmbedGemma2Model {
        &mut self.text
    }

    pub fn vision(&self) -> &EmbedGemma2VisionModel {
        &self.vision
    }

    pub fn audio(&self) -> &EmbedGemma2Audio {
        &self.audio
    }

    /// Embed `items`, one L2-normalized row each (`[items.len(),
    /// truncate_dim.unwrap_or(embedding_dim)]`, in the order given), the
    /// Matryoshka prefix as [`EmbedGemma2Model::encode`] has it. See the
    /// module docs for the items, what is refused before any GPU work, and the
    /// waits.
    ///
    /// An item's embedding does not depend on what else is in the call beyond
    /// GEMM tiling: the towers are bit for bit batch-independent, and the text
    /// encoder's rows agree to its batch bound (bit for bit when the batch
    /// packs into the same forwards).
    pub fn encode(&self, items: &[MmItem<'_>], truncate_dim: Option<u32>) -> Result<EncodeOutput, String> {
        const WHAT: &str = "EmbedGemma2Multimodal::encode";
        if items.is_empty() {
            return Err(format!("{WHAT}: empty batch"));
        }
        if items.len() > MAX_BATCH {
            return Err(format!(
                "{WHAT}: {} items exceeds the batch limit {MAX_BATCH}",
                items.len()
            ));
        }
        let cfg = self.text.config();
        let tokens = self.tokens;

        // The call's media, flat, with where each item's sit. The vision
        // tower's inputs are one list: every image, and every frame of every
        // video, so a call queues one tower batch whatever it holds.
        let mut images: Vec<ImageInput<'_>> = Vec::new();
        let mut clips: Vec<&AudioInput> = Vec::new();
        let mut spans: Vec<Span> = Vec::with_capacity(items.len());
        for item in items {
            let (i0, c0) = (images.len(), clips.len());
            // Where the item's images end and its video frames begin.
            let frames_from = match item {
                MmItem::Text(_) => i0,
                MmItem::Image(img) => {
                    images.push(*img);
                    images.len()
                }
                MmItem::Audio(clip) => {
                    clips.push(*clip);
                    i0
                }
                MmItem::Video(frames) | MmItem::TimestampedVideo { frames, .. } => {
                    images.extend_from_slice(frames);
                    i0
                }
                MmItem::Media {
                    images: im,
                    audio: au,
                    videos,
                    ..
                } => {
                    images.extend_from_slice(im);
                    let frames_from = images.len();
                    for frames in *videos {
                        images.extend_from_slice(frames);
                    }
                    clips.extend(au.iter());
                    frames_from
                }
            };
            spans.push(Span {
                images: i0..frames_from,
                frames: frames_from..images.len(),
                clips: c0..clips.len(),
            });
        }
        let image_tokens = if images.is_empty() {
            Vec::new()
        } else {
            self.vision.check_images(&images).map_err(|e| format!("{WHAT}: {e}"))?
        };
        let clip_tokens = if clips.is_empty() {
            Vec::new()
        } else {
            self.audio.check_clips(&clips).map_err(|e| format!("{WHAT}: {e}"))?
        };

        // Each item's ids.
        let mut ids: Vec<Cow<'_, [u32]>> = Vec::with_capacity(items.len());
        for (b, (item, span)) in items.iter().zip(&spans).enumerate() {
            let at = |e: String| format!("{WHAT}: item {b}: {e}");
            ids.push(match item {
                MmItem::Text(t) => Cow::Borrowed(*t),
                MmItem::Media { ids, .. } => Cow::Borrowed(*ids),
                MmItem::Image(_) => Cow::Owned(cfg.image_sequence(image_tokens[span.images.start]).map_err(at)?),
                MmItem::Audio(_) => Cow::Owned(cfg.audio_sequence(clip_tokens[span.clips.start]).map_err(at)?),
                MmItem::Video(_) | MmItem::TimestampedVideo { .. } => {
                    let per_frame = &image_tokens[span.frames.clone()];
                    // The processor has one soft-token count per video.
                    if let Some(j) = per_frame.iter().position(|&n| n != per_frame[0]) {
                        return Err(format!(
                            "{WHAT}: item {b}: frame {j} makes {} soft tokens but frame 0 makes {}; the \
                             processor lays out one count for a video (MmItem::Media takes frames of \
                             different sizes with ids of your own)",
                            per_frame[j], per_frame[0]
                        ));
                    }
                    match item {
                        MmItem::TimestampedVideo {
                            timestamps, tokenizer, ..
                        } => Cow::Owned(
                            cfg.video_sequence_with_timestamps(tokenizer, timestamps, per_frame)
                                .map_err(at)?,
                        ),
                        _ => Cow::Owned(cfg.video_sequence(per_frame).map_err(at)?),
                    }
                }
            });
        }
        let id_refs: Vec<&[u32]> = ids.iter().map(AsRef::as_ref).collect();
        self.text
            .check_sequences(WHAT, &id_refs, Some(tokens), truncate_dim, false)?;
        for (b, (item, span)) in items.iter().zip(&spans).enumerate() {
            let count = |id: u32| id_refs[b].iter().filter(|&&t| t == id).count();
            let (n_image, n_audio, n_video) = (count(tokens.image), count(tokens.audio), count(tokens.video));
            let have_image: usize = image_tokens[span.images.clone()].iter().sum();
            let have_video: usize = image_tokens[span.frames.clone()].iter().sum();
            let have_audio: usize = clip_tokens[span.clips.clone()].iter().sum();
            if matches!(item, MmItem::Text(_)) && n_image + n_audio + n_video > 0 {
                return Err(format!(
                    "{WHAT}: item {b} is text but its ids hold {} placeholder ids (image {}, audio {}, video {}); \
                     give its media with MmItem::Media",
                    n_image + n_audio + n_video,
                    tokens.image,
                    tokens.audio,
                    tokens.video
                ));
            }
            if n_video != have_video {
                return Err(format!(
                    "{WHAT}: item {b} has {n_video} video placeholder ids (id {}) but its {} video frame(s) make \
                     {have_video} soft tokens",
                    tokens.video,
                    span.frames.len()
                ));
            }
            if n_image != have_image {
                return Err(format!(
                    "{WHAT}: item {b} has {n_image} image placeholder ids (id {}) but its {} image(s) make {have_image} \
                     soft tokens",
                    tokens.image,
                    span.images.len()
                ));
            }
            if n_audio != have_audio {
                return Err(format!(
                    "{WHAT}: item {b} has {n_audio} audio placeholder ids (id {}) but its {} clip(s) make {have_audio} \
                     soft tokens",
                    tokens.audio,
                    span.clips.len()
                ));
            }
        }

        // Every forward of the call, towers then text encoder, queued under one
        // scope behind one wait. The towers' soft rows are not read back: the
        // text encoder's scatter reads them where the towers wrote them.
        let has_media = !images.is_empty() || !clips.is_empty();
        let scope = self.rt.async_scope();
        // A tower queued before a failure is dropped (and so waited for) when
        // this returns.
        let queued = (|| {
            let v = if images.is_empty() {
                None
            } else {
                Some(self.vision.queue_images(&images, false)?)
            };
            let a = if clips.is_empty() {
                None
            } else {
                Some(self.audio.queue_audio(&clips, false)?)
            };
            Ok::<_, String>((v, a))
        })();
        let (v, a) = queued?;
        let image_soft: Vec<SoftSegment> = v.map(PendingImages::into_device).unwrap_or_default();
        let clip_soft: Vec<Vec<SoftSegment>> = a.map(PendingAudio::into_device).unwrap_or_default();
        // From here the towers' work is queued and nothing waits for it but
        // the wait below, which is made whatever follows: the runtime holds
        // their buffers until it, and a text-encoder call that fails while
        // queuing holds its own the same way.
        let text = (|| {
            // Each item's soft rows, in the order of its placeholder ids.
            let softs = items
                .iter()
                .zip(&spans)
                .zip(&id_refs)
                .enumerate()
                .map(|(b, ((item, span), ids))| match item {
                    MmItem::Text(_) => Ok(Vec::new()),
                    _ => item_segments(
                        ids,
                        &tokens,
                        &image_soft[span.images.clone()],
                        &image_soft[span.frames.clone()],
                        &clip_soft[span.clips.clone()],
                    )
                    .map_err(|e| format!("{WHAT}: item {b}: {e}")),
                })
                .collect::<Result<Vec<_>, String>>()?;
            let seqs: Vec<Seq<'_>> = id_refs
                .iter()
                .zip(&softs)
                .map(|(ids, soft)| Seq {
                    ids,
                    soft: SoftSource::Device(soft),
                })
                .collect();
            self.text.queue_multimodal(WHAT, &seqs, truncate_dim)
        })();
        // The device rows are the text encoder's now; release ours with the
        // rest before the wait.
        drop((image_soft, clip_soft));
        drop(scope);
        let text = match text {
            Err(e) if !has_media => return Err(e),
            other => other,
        };
        let waited = self.rt.synchronize();
        let out = text?.finish(waited)?;

        // A host caller's soft rows are checked finite before any GPU work;
        // the towers' are on the device, so the rows they produced are checked
        // through what they made of the embeddings: a non-finite soft token
        // makes its item's embedding non-finite.
        let dim = out.embeddings.len() / items.len();
        for (b, item) in items.iter().enumerate() {
            let media = !matches!(item, MmItem::Text(_));
            if media && out.embeddings[b * dim..(b + 1) * dim].iter().any(|v| !v.is_finite()) {
                return Err(format!(
                    "{WHAT}: item {b}: the embedding is not finite; a tower's soft tokens were not"
                ));
            }
        }
        Ok(out)
    }
}

/// An item's soft rows as device segments, in the order of its placeholder
/// ids: the `j`-th image placeholder takes the `j`-th row of the item's
/// images laid end to end, the video placeholders likewise from its video
/// frames (the vision tower's rows, one segment per frame) and the audio
/// placeholders from its clips (`masked_scatter` per modality). Consecutive
/// rows of one buffer are one segment, so an item with one image or clip is
/// one segment per run the tower wrote, however its placeholders interleave
/// with text, and the frames of a video that one tower forward made are one
/// segment per placeholder run.
fn item_segments(
    ids: &[u32],
    tokens: &MmTokens,
    images: &[SoftSegment],
    frames: &[SoftSegment],
    clips: &[Vec<SoftSegment>],
) -> Result<Vec<SoftSegment>, String> {
    let mut img = images
        .iter()
        .flat_map(|s| (s.row..s.row + s.rows).map(move |r| (&s.buf, r)));
    let mut vid = frames
        .iter()
        .flat_map(|s| (s.row..s.row + s.rows).map(move |r| (&s.buf, r)));
    let mut aud = clips
        .iter()
        .flatten()
        .flat_map(|s| (s.row..s.row + s.rows).map(move |r| (&s.buf, r)));
    let mut out: Vec<SoftSegment> = Vec::new();
    for &id in ids {
        let next = if id == tokens.image {
            img.next()
        } else if id == tokens.video {
            vid.next()
        } else if id == tokens.audio {
            aud.next()
        } else {
            continue;
        };
        let (buf, row) = next.ok_or_else(|| "more placeholder ids than soft rows".to_string())?;
        match out.last_mut() {
            Some(last) if last.buf.aliases(buf) && last.row + last.rows == row => last.rows += 1,
            _ => out.push(SoftSegment {
                buf: buf.clone(),
                row,
                rows: 1,
            }),
        }
    }
    if img.next().is_some() || vid.next().is_some() || aud.next().is_some() {
        return Err("more soft rows than placeholder ids".to_string());
    }
    Ok(out)
}
