//! EmbeddingGemma 2 as the checkpoint ships it: the text encoder and both
//! towers behind one object, one snapshot and one call.
//!
//! [`EmbedGemma2Multimodal`] is a composition, not a fourth implementation: it
//! owns an [`EmbedGemma2Model`], an [`EmbedGemma2VisionModel`] and an
//! [`EmbedGemma2Audio`] loaded from one `config.json` and one
//! `model.safetensors`, and [`EmbedGemma2Multimodal::encode`] runs what the
//! three already do, in the order the checkpoint's reference does:
//!
//! 1. every image of the call through the vision tower and every clip through
//!    the audio tower (the soft tokens that replace an item's placeholder ids);
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
//! * [`MmItem::Media`]: ids a caller built (a prompt around an image, an
//!   image and a clip in one sequence) with the media their placeholders
//!   stand for. The `j`-th image placeholder takes the `j`-th soft row of the
//!   item's images laid end to end, and the audio placeholders likewise from
//!   the item's clips, as transformers' `masked_scatter` does per modality. The
//!   counts must match exactly. A video placeholder is refused: there is no
//!   video tower.
//!
//! # Refusals
//!
//! Everything a call can be refused for is checked before the first
//! allocation or dispatch: the towers' input checks
//! ([`EmbedGemma2VisionModel::check_images`], [`EmbedGemma2Audio::check_clips`]),
//! the text side's ([`EmbedGemma2Model::check_sequences`]: lengths, ids,
//! `truncate_dim`, the batch size) and the facade's own (placeholder counts
//! against soft-token counts). A refused call allocates nothing and waits for
//! nothing. Media are numbered across the call in item order in the towers'
//! messages (`image 2` is the third image of the call, not of its item).
//!
//! # Waiting
//!
//! A call with media waits for the GPU **twice**, a call of text items once.
//! The towers' forwards are queued together ([`EmbedGemma2VisionModel::queue_images`],
//! [`EmbedGemma2Audio::queue_audio`]) under one [`crate::runtime::GpuRuntime::async_scope`]
//! behind one wait; the soft tokens are then read back and handed to the text
//! encoder as host rows, whose own forwards are queued behind its one wait. The
//! second wait is the cost of the host round trip: the text encoder takes its
//! soft rows as host slices, so the towers' output must reach the host first.
//! Handing the towers' device buffers to the text encoder directly would make
//! it one wait; that is not done here (see `docs/embedgemma2.md`).
//!
//! # Precision
//!
//! [`GemmOperands`] applies to the **text encoder** only
//! ([`EmbedGemma2Multimodal::load`]'s `operands`). The towers are exact f32 in
//! either mode: they have no bf16 operand path, and the call refuses a runtime
//! whose relaxed f32 GEMMs are on whatever the mode. Under `Bf16` the text
//! encoder's projections are bf16, so the soft tokens (computed in f32) enter
//! it as the embeddings do and are rounded where its GEMMs round them.

use std::borrow::Cow;
use std::io::Read;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use crate::embedgemma2::{EmbedGemma2Config, EmbedGemma2Model, EncodeOutput, MmSequence, MmTokens, MAX_BATCH};
use crate::embedgemma2_audio::{AudioConfig, AudioInput, EmbedGemma2Audio};
use crate::embedgemma2_vision::{EmbedGemma2VisionModel, ImageInput, VisionConfig};
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
    /// Caller-built ids and the media their placeholders stand for.
    Media {
        /// The ids, placeholders included.
        ids: &'a [u32],
        /// The images whose soft tokens fill the image placeholders, in order.
        images: &'a [ImageInput<'a>],
        /// The clips whose soft tokens fill the audio placeholders, in order.
        audio: &'a [AudioInput],
    },
}

/// Where an item's media sit in the call's flat image and clip lists.
struct Span {
    images: Range<usize>,
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
    /// ([`EmbedGemma2Model::load_with`]); the towers are exact f32.
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
        let vision = EmbedGemma2VisionModel::load(rt, st, vision_cfg)?;
        let audio = EmbedGemma2Audio::load(rt, st, "", audio_cfg)?;
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
        let hidden = cfg.hidden as usize;
        let tokens = self.tokens;

        // The call's media, flat, with where each item's sit.
        let mut images: Vec<ImageInput<'_>> = Vec::new();
        let mut clips: Vec<&AudioInput> = Vec::new();
        let mut spans: Vec<Span> = Vec::with_capacity(items.len());
        for item in items {
            let (i0, c0) = (images.len(), clips.len());
            match item {
                MmItem::Text(_) => {}
                MmItem::Image(img) => images.push(*img),
                MmItem::Audio(clip) => clips.push(*clip),
                MmItem::Media {
                    images: im, audio: au, ..
                } => {
                    images.extend_from_slice(im);
                    clips.extend(au.iter());
                }
            }
            spans.push(Span {
                images: i0..images.len(),
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
            });
        }
        let id_refs: Vec<&[u32]> = ids.iter().map(AsRef::as_ref).collect();
        self.text
            .check_sequences(WHAT, &id_refs, Some(tokens), truncate_dim, false)?;
        for (b, (item, span)) in items.iter().zip(&spans).enumerate() {
            let count = |id: u32| id_refs[b].iter().filter(|&&t| t == id).count();
            let (n_image, n_audio, n_video) = (count(tokens.image), count(tokens.audio), count(tokens.video));
            let have_image: usize = image_tokens[span.images.clone()].iter().sum();
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
            if n_video > 0 {
                return Err(format!(
                    "{WHAT}: item {b} has {n_video} video placeholder ids (id {}); there is no video tower",
                    tokens.video
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

        // The towers, queued together behind one wait, then read back.
        let (mut image_soft, mut clip_soft): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (Vec::new(), Vec::new());
        if !images.is_empty() || !clips.is_empty() {
            let scope = self.rt.async_scope();
            // A tower queued before a failure is dropped (and so waited for)
            // when this returns.
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
            self.rt.synchronize()?;
            drop(scope);
            image_soft = v.map(|p| p.collect()).transpose()?.map(|o| o.soft).unwrap_or_default();
            clip_soft = a
                .map(|p| p.collect())
                .transpose()?
                .map(|o| o.soft_tokens)
                .unwrap_or_default();
        }

        // Each item's soft rows, in the order of its placeholder ids.
        let mut softs: Vec<Vec<f32>> = Vec::with_capacity(items.len());
        for (b, (item, span)) in items.iter().zip(&spans).enumerate() {
            softs.push(match item {
                MmItem::Text(_) => Vec::new(),
                MmItem::Image(_) => std::mem::take(&mut image_soft[span.images.start]),
                MmItem::Audio(_) => std::mem::take(&mut clip_soft[span.clips.start]),
                MmItem::Media { .. } => {
                    let img: Vec<f32> = image_soft[span.images.clone()].concat();
                    let aud: Vec<f32> = clip_soft[span.clips.clone()].concat();
                    let (mut ic, mut ac) = (0usize, 0usize);
                    let mut out = Vec::with_capacity(img.len() + aud.len());
                    for &id in id_refs[b] {
                        if id == tokens.image {
                            out.extend_from_slice(&img[ic * hidden..(ic + 1) * hidden]);
                            ic += 1;
                        } else if id == tokens.audio {
                            out.extend_from_slice(&aud[ac * hidden..(ac + 1) * hidden]);
                            ac += 1;
                        }
                    }
                    debug_assert_eq!((ic * hidden, ac * hidden), (img.len(), aud.len()));
                    out
                }
            });
        }
        let seqs: Vec<MmSequence<'_>> = id_refs
            .iter()
            .zip(&softs)
            .map(|(ids, soft)| MmSequence { ids, soft })
            .collect();
        self.text.encode_multimodal(&seqs, truncate_dim, false)
    }
}
