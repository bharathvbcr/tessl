//! The C ABI's EmbeddingGemma 2 entries, so a caller with no Rust toolchain
//! (Python) can embed text, images, audio and video: [`tessl_embedgemma2_load`]
//! reads a snapshot directory and [`tessl_embedgemma2_encode`] turns token ids
//! into L2-normalized embeddings ([`EmbedGemma2Model::encode`]);
//! [`tessl_embedgemma2_load_multimodal`] loads the towers too, and
//! [`tessl_embedgemma2_encode_images`] and [`tessl_embedgemma2_encode_audio`]
//! turn a processor's `pixel_values` / `input_features` into embeddings
//! ([`EmbedGemma2Multimodal::encode`]).
//!
//! The entries that need no transformers processor and no Python tokenizer run
//! the snapshot's own front ends in Rust:
//!
//! | Entry | Takes | Front end |
//! |---|---|---|
//! | [`tessl_embedgemma2_embed_texts`] | UTF-8 texts, an optional prompt name | [`crate::embedgemma2_tok::Tokenizer`] |
//! | [`tessl_embedgemma2_tokenize`], [`tessl_embedgemma2_decode`] | texts / ids | the tokenizer |
//! | [`tessl_embedgemma2_prompt_count`], [`tessl_embedgemma2_prompt_name`], [`tessl_embedgemma2_prompt_get`], [`tessl_embedgemma2_default_prompt`] | prompt index or name | the snapshot's named prompts |
//! | [`tessl_embedgemma2_embed_images`] | raw `u8` HWC images, 1 to 4 channels | [`crate::embedgemma2_pre::preprocess_image`] |
//! | [`tessl_embedgemma2_embed_audio`] | mono `f32` PCM at the extractor's rate | [`crate::embedgemma2_pre::AudioFrontEnd`] |
//! | [`tessl_embedgemma2_embed_videos`] | decoded RGB frames, source fps, a timestamps flag | [`crate::embedgemma2_pre::preprocess_video`] |
//! | [`tessl_embedgemma2_embed_media`] | caller ids and raw images, clips and videos ([`MmItem::Media`]) | all of the above |
//!
//! The tokenizer and the processor settings are read from the snapshot
//! directory the handle was loaded from, on the first call that needs them
//! (`tokenizer.json`, `tokenizer_config.json`, `config_sentence_transformers.json`
//! when present; `processor_config.json`), and cached on the handle. A snapshot
//! without them loads and encodes ids as before; a call that needs a missing
//! file is refused naming it, and the next call tries again.
//!
//! The handle, thread-affinity, status-code and error-message conventions are
//! `src/capi.rs`'s (see its module docs): every entry runs through the same
//! `guarded` (null and foreign-thread handles refused before any reference is
//! formed, panics caught, the runtime idle on return) and `free_handle`. They
//! live in their own module so the EmbeddingGemma 2 surface does not share a
//! file with the training entries; nothing here is a second implementation of
//! the handle machinery. The entries are additive: no existing signature or
//! `#[repr(C)]` layout changed, so [`crate::capi::TESSL_ABI_VERSION`] did not.
//!
//! # What a caller passes
//!
//! Token ids travel as one flat `u32` array plus CSR offsets: sequence `i` is
//! `ids[offsets[i]..offsets[i + 1]]`, so `offsets` has `n_seqs + 1` entries,
//! starts at 0, increases strictly (a sequence is never empty) and ends at
//! `n_ids`. The ids carry the tokenizer's `<bos>`/`<eos>` and any task prompt
//! already. The output is `f32` `[n_seqs, dim]` row-major, where `dim` is
//! `truncate_dim`, or [`tessl_embedgemma2_embedding_dim`] when that is 0.
//!
//! Images and clips use the same CSR shape over their rows. Image `i` is the
//! processor's `pixel_values [P_i, patch_dim]` (f32) and `image_position_ids
//! [P_i, 2]` (i32), padding patches `(-1, -1)` included, so the arrays hold
//! `offsets[n_images]` patch rows in all; clip `i` is `input_features [F_i,
//! mel_bins]` (f32) and `input_features_mask [F_i]` (one byte per frame, 0 or
//! 1). The text sequence of each item is the processor's
//! ([`crate::embedgemma2::EmbedGemma2Config::image_sequence`] /
//! `audio_sequence`) and each row of the output is that item's embedding.
//!
//! # What is checked, and when
//!
//! Everything about the call is checked before anything is allocated, mutated
//! or sent to the GPU, and every caller-supplied length is bounded before it
//! sizes a slice: `n_seqs` by [`MAX_BATCH`], each sequence by the model's
//! `max_tokens`, so `n_ids` is bounded by their product and never taken on
//! trust; `n_images` and `n_clips` by [`MAX_BATCH`] (one output row each) and
//! the patch rows of all the images by [`MAX_CALL_PATCHES`] and the frames of
//! all the clips by [`MAX_CALL_FRAMES`] (the calls' bounds, which the tower
//! checks again after the copy this avoids); each image's `P` by
//! [`MAX_BATCH_PATCHES`] and each clip's `F` by [`MAX_FRAMES_PER_FORWARD`];
//! every array length must equal exactly what its
//! offsets imply, and `out_len` must equal `n * dim`. The output buffer is
//! written only when the whole batch has succeeded, so a refused or failed call
//! leaves it as it was. Token ids beyond the vocabulary are refused by
//! [`EmbedGemma2Model::encode`] before it touches the device; the pixel and
//! feature checks are the towers' own ([`EmbedGemma2Multimodal::encode`]). A
//! clip's features are copied into the host memory the tower takes them in
//! (`AudioInput` owns its rows) after its length was bounded.
//!
//! The raw entries lay their inputs out the same way. Texts are one UTF-8 byte
//! array and `n_texts + 1` offsets (starting at 0, never decreasing, ending at
//! the byte count: an empty text is the two-token sequence `<bos> <eos>`).
//! Raw images are one `u8` array of the images laid end to end, each row-major
//! HWC, and a `[n_images, 3]` `u32` table of `(height, width, channels)` whose
//! products must sum to exactly the array's length. Raw clips are one `f32`
//! array and `n_clips + 1` offsets over samples. Videos are one `u8` array of
//! RGB frames laid end to end, a `[n_videos, 2]` `u32` table of `(height,
//! width)`, `n_videos + 1` offsets over frames and one `f64` source rate each.
//!
//! # Bounds
//!
//! The output rows of every embed entry are the batch the text encoder takes
//! at once, so the number of items (texts, images, clips, videos, media items)
//! is bounded by [`MAX_BATCH`] (256). What one item may hold is bounded
//! separately by the tower calls' own limits, which are per call and not per
//! forward since the facade cuts a call into forwards behind its one wait:
//! pictures (images and video frames) by [`MAX_CALL_IMAGES`] and padded patches
//! by [`MAX_CALL_PATCHES`], clips by [`MAX_CALL_CLIPS`] and frames by
//! [`MAX_CALL_FRAMES`]. [`crate::embedgemma2_vision::MAX_IMAGES`] and
//! [`crate::embedgemma2_audio::MAX_CLIPS`] are the size of one forward and
//! bound nothing here. A raw entry checks those against what the
//! preprocessing *would* build before it builds anything (a raw image becomes
//! `max_soft_tokens * pooling_kernel²` patches of 3 KiB whatever its size), so
//! the host memory a call asks for is bounded by them and not by the caller's
//! counts. A clip is at most [`TESSL_EMBEDGEMMA2_MAX_CLIP_SAMPLES`] samples
//! (the extractor cuts it to 30 s after that), a text less than
//! [`MAX_TEXT_BYTES`] bytes, a name [`TESSL_EMBEDGEMMA2_MAX_NAME_BYTES`].
//!
//! # Precision
//!
//! The runtime must not have relaxed precision switched on for an `operands`
//! exact-f32 handle (the entries that run a model refuse it by name). A handle
//! holding `operands` bf16 accepts it, calls with clips and videos included:
//! the text encoder, the vision tower and the audio tower then run no f32 GEMM
//! (`tests/capi_embedgemma2_raw.rs` pins both). Nothing in the C ABI turns
//! relaxed precision on. The handle's `operands` applies to the text encoder
//! and both towers.

use std::ffi::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::ThreadId;

use crate::capi::{
    checked_slice, checked_slice_mut, free_handle, guarded, parse_operands, Handle, TesslRuntime, TESSL_OK,
};
use crate::embedgemma2::{EmbedGemma2Config, EmbedGemma2Model, EncodeOutput, MAX_BATCH};
use crate::embedgemma2_audio::{AudioConfig, AudioInput, MAX_CALL_CLIPS, MAX_CALL_FRAMES, MAX_FRAMES_PER_FORWARD};
use crate::embedgemma2_media::{prepare_audio, prepare_image, MediaSource, OpenedVideo};
use crate::embedgemma2_mm::{read_config_text, EmbedGemma2Multimodal, MmItem, MmSegment, TEXT_PREFIX};
use crate::embedgemma2_pre::{
    audio_soft_tokens, image_soft_tokens, preprocess_image, preprocess_video, sample_frame_indices, video_soft_tokens,
    AudioFeatureConfig, AudioFrontEnd, ImageConfig, PreConfig, PreparedImage, PreparedVideo, RawImage, VideoConfig,
    AUDIO_PAD_MULTIPLE, MAX_IMAGE_PIXELS, MAX_VIDEO_FRAMES,
};
use crate::embedgemma2_tok::{Tokenizer, MAX_TEXT_BYTES};
use crate::embedgemma2_vision::{ImageInput, VisionConfig, MAX_BATCH_PATCHES, MAX_CALL_IMAGES, MAX_CALL_PATCHES};
use crate::runtime::GpuRuntime;
use crate::safetensors::SafeTensors;

/// Longest snapshot-directory path [`tessl_embedgemma2_load`] accepts, in
/// bytes. macOS's `PATH_MAX` is 1024; this leaves room for a longer
/// canonicalized form without letting a caller's length size a copy.
pub const TESSL_EMBEDGEMMA2_MAX_PATH_BYTES: u64 = 4096;

/// What a [`tessl_embedgemma2_load`] pointer holds: the text encoder alone, or
/// it with both towers.
enum Engine {
    Text(Box<EmbedGemma2Model>),
    Multimodal(Box<EmbedGemma2Multimodal>),
}

impl Engine {
    fn text(&self) -> &EmbedGemma2Model {
        match self {
            Engine::Text(m) => m,
            Engine::Multimodal(m) => m.text(),
        }
    }

    fn text_mut(&mut self) -> &mut EmbedGemma2Model {
        match self {
            Engine::Text(m) => m,
            Engine::Multimodal(m) => m.text_mut(),
        }
    }

    /// The multimodal model, or the refusal for a text-only handle.
    fn multimodal(&self, what: &str) -> Result<&EmbedGemma2Multimodal, String> {
        match self {
            Engine::Multimodal(m) => Ok(m),
            Engine::Text(_) => Err(format!(
                "{what}: this handle was loaded with tessl_embedgemma2_load (text only); load it with \
                 tessl_embedgemma2_load_multimodal"
            )),
        }
    }
}

/// What a [`tessl_embedgemma2_load`] pointer owns.
pub struct TesslEmbedGemma2 {
    engine: Engine,
    /// The model's own clone of the runtime (the models keep their copy
    /// private); the model stays valid however the caller treats the runtime
    /// handle it was loaded on.
    rt: Arc<GpuRuntime>,
    owner: ThreadId,
    /// The snapshot directory the model was loaded from, where the front ends
    /// read their files on first use.
    snapshot: PathBuf,
    front: Front,
}

/// The snapshot's own front ends, each read on the first call that needs it
/// and kept on the handle. A read that fails leaves its slot empty, so the
/// next call tries again (the file may have been fixed or put there since).
#[derive(Default)]
struct Front {
    tokenizer: Option<Tokenizer>,
    pre: Option<PreConfig>,
    audio: Option<AudioFrontEnd>,
}

impl Front {
    /// `tokenizer.json` and `tokenizer_config.json`, and the prompts when the
    /// snapshot has `config_sentence_transformers.json`.
    fn tokenizer(&mut self, snapshot: &Path, what: &str) -> Result<&Tokenizer, String> {
        if self.tokenizer.is_none() {
            self.tokenizer = Some(
                Tokenizer::from_snapshot_optional_prompts(snapshot)
                    .map_err(|e| format!("{what}: the snapshot's tokenizer: {e}"))?,
            );
        }
        self.tokenizer
            .as_ref()
            .ok_or_else(|| format!("{what}: the tokenizer is not loaded"))
    }

    /// `processor_config.json`.
    fn pre(&mut self, snapshot: &Path, what: &str) -> Result<&PreConfig, String> {
        if self.pre.is_none() {
            self.pre =
                Some(PreConfig::from_snapshot(snapshot).map_err(|e| format!("{what}: the snapshot's processor: {e}"))?);
        }
        self.pre
            .as_ref()
            .ok_or_else(|| format!("{what}: the processor config is not loaded"))
    }

    /// The log-mel front end built from the processor config's extractor.
    fn audio(&mut self, snapshot: &Path, what: &str) -> Result<&AudioFrontEnd, String> {
        if self.audio.is_none() {
            let cfg = self.pre(snapshot, what)?.audio.clone();
            self.audio = Some(AudioFrontEnd::new(&cfg).map_err(|e| format!("{what}: {e}"))?);
        }
        self.audio
            .as_ref()
            .ok_or_else(|| format!("{what}: the audio front end is not built"))
    }
}

impl Handle for TesslEmbedGemma2 {
    const KIND: &'static str = "embedgemma2 model";
    unsafe fn owner_of(this: *const Self) -> ThreadId {
        // SAFETY: by the trait contract; a field place read, no reference.
        unsafe { std::ptr::addr_of!((*this).owner).read() }
    }
    fn runtime(&self) -> &Arc<GpuRuntime> {
        &self.rt
    }
}

/// The text encoder from a snapshot directory: `config.json` (a bounded read
/// of a regular file) parsed, and `model.safetensors` opened.
fn load_text(rt: &Arc<GpuRuntime>, dir: &Path) -> Result<Engine, String> {
    let config = dir.join("config.json");
    let text = read_config_text(&config)?;
    let cfg = EmbedGemma2Config::from_config_json(&text).map_err(|e| format!("{}: {e}", config.display()))?;
    let st = SafeTensors::open(&dir.join("model.safetensors"))?;
    Ok(Engine::Text(Box::new(EmbedGemma2Model::load(
        rt,
        &st,
        TEXT_PREFIX,
        cfg,
    )?)))
}

/// The body of both load entries: the arguments checked, the snapshot
/// directory decoded, `build` run on it, and the handle written to `*out` only
/// after the call's final synchronize succeeded.
///
/// # Safety
/// As [`tessl_embedgemma2_load`].
#[allow(clippy::too_many_arguments)]
unsafe fn load_entry(
    what: &'static str,
    runtime: *mut TesslRuntime,
    snapshot_dir: *const c_char,
    snapshot_dir_len: u64,
    out: *mut *mut TesslEmbedGemma2,
    err: *mut c_char,
    err_len: usize,
    build: impl FnOnce(&Arc<GpuRuntime>, &Path) -> Result<Engine, String>,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        if !out.is_null() {
            *out = std::ptr::null_mut();
        }
        let mut built: Option<Box<TesslEmbedGemma2>> = None;
        let status = guarded(runtime, err, err_len, |h| {
            if out.is_null() {
                return Err(format!("{what}: null out"));
            }
            if snapshot_dir_len == 0 || snapshot_dir_len > TESSL_EMBEDGEMMA2_MAX_PATH_BYTES {
                return Err(format!(
                    "{what}: snapshot_dir is {snapshot_dir_len} bytes, expected 1..={TESSL_EMBEDGEMMA2_MAX_PATH_BYTES}"
                ));
            }
            let n =
                usize::try_from(snapshot_dir_len).map_err(|_| format!("{what}: snapshot_dir_len overflows usize"))?;
            let bytes = checked_slice(snapshot_dir.cast::<u8>(), n, what, "snapshot_dir")?;
            let dir = std::str::from_utf8(bytes).map_err(|_| format!("{what}: snapshot_dir is not UTF-8"))?;
            if dir.contains('\0') {
                return Err(format!("{what}: snapshot_dir contains a NUL byte"));
            }
            let rt = Arc::clone(h.runtime());
            let engine = build(&rt, Path::new(dir))?;
            built = Some(Box::new(TesslEmbedGemma2 {
                engine,
                rt,
                owner: std::thread::current().id(),
                snapshot: PathBuf::from(dir),
                front: Front::default(),
            }));
            Ok(())
        });
        match (status, built) {
            (TESSL_OK, Some(model)) => *out = Box::into_raw(model),
            // A failed sync or a panic: the model never reaches the caller.
            // Dropping waits on the GPU, which must not unwind into C.
            (_, model) => {
                let _ = catch_unwind(AssertUnwindSafe(move || drop(model)));
            }
        }
        status
    }
}

/// Load the text encoder from a Hugging Face snapshot directory (the
/// `config.json` and the single-file `model.safetensors` of
/// `google/embeddinggemma-2`, tensors under `language_model.`) on `runtime`'s
/// device, into a new handle written to `*out`. Free it with
/// [`tessl_embedgemma2_free`], on this thread.
///
/// `snapshot_dir` is `snapshot_dir_len` bytes of UTF-8, not NUL-terminated;
/// at most [`TESSL_EMBEDGEMMA2_MAX_PATH_BYTES`], no NUL byte.
///
/// `*out` is null whenever the status is not [`TESSL_OK`]: the handle is
/// written only after the call's final synchronize succeeds, and a model built
/// by a call that then fails is freed here, not leaked into `*out`.
///
/// # Safety
/// As [`crate::capi::tessl_cross_entropy_rows`] for `runtime`, `err` and
/// `err_len`; `snapshot_dir` is null (refused) or points to
/// `snapshot_dir_len` readable bytes; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_load(
    runtime: *mut TesslRuntime,
    snapshot_dir: *const c_char,
    snapshot_dir_len: u64,
    out: *mut *mut TesslEmbedGemma2,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        load_entry(
            "tessl_embedgemma2_load",
            runtime,
            snapshot_dir,
            snapshot_dir_len,
            out,
            err,
            err_len,
            load_text,
        )
    }
}

/// [`tessl_embedgemma2_load`] with the vision and audio towers as well
/// ([`EmbedGemma2Multimodal::load`]): the handle takes every entry of this
/// module, and [`tessl_embedgemma2_encode_images`] and
/// [`tessl_embedgemma2_encode_audio`] too. One read of `config.json`, one open
/// of `model.safetensors`; about 1.7 GB of weights on the device where the text
/// encoder alone is 0.8 GB.
///
/// `operands` is the GEMM operands of the text encoder, the vision tower and
/// the audio tower ([`crate::capi::TESSL_OPERANDS_EXACT_F32`] or
/// [`crate::capi::TESSL_OPERANDS_BF16`]; anything else is refused).
///
/// # Safety
/// As [`tessl_embedgemma2_load`].
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_load_multimodal(
    runtime: *mut TesslRuntime,
    snapshot_dir: *const c_char,
    snapshot_dir_len: u64,
    operands: u32,
    out: *mut *mut TesslEmbedGemma2,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    const WHAT: &str = "tessl_embedgemma2_load_multimodal";
    // SAFETY: forwarded from this function's contract.
    unsafe {
        load_entry(
            WHAT,
            runtime,
            snapshot_dir,
            snapshot_dir_len,
            out,
            err,
            err_len,
            |rt, dir| {
                let operands = parse_operands(operands, WHAT)?;
                Ok(Engine::Multimodal(Box::new(EmbedGemma2Multimodal::load(
                    rt, dir, operands,
                )?)))
            },
        )
    }
}

/// Free a handle from [`tessl_embedgemma2_load`] or
/// [`tessl_embedgemma2_load_multimodal`]. Null is ignored ([`TESSL_OK`]); from
/// a thread other than the creator's the handle is leaked and `TESSL_ERR`
/// returned, as [`crate::capi::tessl_runtime_free`] does.
///
/// # Safety
/// As [`crate::capi::tessl_runtime_free`], for a [`tessl_embedgemma2_load`]
/// handle.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_free(model: *mut TesslEmbedGemma2) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe { free_handle(model) }
}

/// The model's `embedding_dim`, written to `*out`: the width of an untruncated
/// embedding, and the largest `truncate_dim`.
///
/// # Safety
/// As [`crate::capi::tessl_cross_entropy_rows`], for a
/// [`tessl_embedgemma2_load`] handle; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_embedding_dim(
    model: *mut TesslEmbedGemma2,
    out: *mut u32,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            if out.is_null() {
                return Err("tessl_embedgemma2_embedding_dim: null out".into());
            }
            *out = h.engine.text().config().embedding_dim;
            Ok(())
        })
    }
}

/// The per-sequence token limit [`tessl_embedgemma2_encode`] enforces,
/// written to `*out` (8192 until [`tessl_embedgemma2_set_max_tokens`]).
///
/// # Safety
/// As [`tessl_embedgemma2_embedding_dim`].
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_max_tokens(
    model: *mut TesslEmbedGemma2,
    out: *mut u32,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            if out.is_null() {
                return Err("tessl_embedgemma2_max_tokens: null out".into());
            }
            *out = h.engine.text().max_tokens();
            Ok(())
        })
    }
}

/// Raise or lower the per-sequence token limit
/// ([`EmbedGemma2Model::set_max_tokens`]): `max_tokens` in
/// `1..=`[`crate::embedgemma2::MAX_BATCH_TOKENS`], else refused and the limit
/// unchanged. It bounds the image and audio sequences of the multimodal
/// entries too.
///
/// # Safety
/// As [`crate::capi::tessl_cross_entropy_rows`], for a
/// [`tessl_embedgemma2_load`] handle.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_set_max_tokens(
    model: *mut TesslEmbedGemma2,
    max_tokens: u32,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe { guarded(model, err, err_len, |h| h.engine.text_mut().set_max_tokens(max_tokens)) }
}

/// The columns of the output (`truncate_dim` 0 is the full width, else it must
/// be in `1..=embedding_dim`) and the exact element count `out_len` must have
/// for `n` rows.
fn output_shape(what: &str, cfg: &EmbedGemma2Config, n: usize, requested: u32, out_len: u64) -> Result<usize, String> {
    let dim_out = if requested == 0 { cfg.embedding_dim } else { requested };
    if dim_out > cfg.embedding_dim {
        return Err(format!(
            "{what}: truncate_dim {requested} must be 0 (full) or in 1..={}",
            cfg.embedding_dim
        ));
    }
    let want_out = n
        .checked_mul(dim_out as usize)
        .ok_or_else(|| format!("{what}: rows * dim overflows usize"))?;
    if out_len != want_out as u64 {
        return Err(format!(
            "{what}: out_len is {out_len}, expected exactly {want_out} (= {n} rows x {dim_out})"
        ));
    }
    Ok(want_out)
}

/// `enc`'s rows into `out`, which holds exactly as many (the caller checked
/// `out_len` against the rows it asked for).
fn copy_embeddings(what: &str, enc: &EncodeOutput, out: &mut [f32]) -> Result<(), String> {
    if enc.embeddings.len() != out.len() {
        return Err(format!(
            "{what}: the model returned {} values for {} expected",
            enc.embeddings.len(),
            out.len()
        ));
    }
    out.copy_from_slice(&enc.embeddings);
    Ok(())
}

/// `n + 1` CSR offsets read from `offsets`, validated: they start at 0,
/// increase strictly, no row has more than `max_rows` entries. Returns them
/// with the last one (the total row count), which is bounded by `n * max_rows`
/// before it sizes anything.
///
/// # Safety
/// `offsets` points to `n + 1` readable `u64`s or is refused as `checked_slice` does.
unsafe fn csr_offsets<'a>(
    what: &str,
    offsets: *const u64,
    n: usize,
    noun: &str,
    unit: &str,
    max_rows: u64,
    limit_hint: &str,
) -> Result<&'a [u64], String> {
    // SAFETY: forwarded from this function's contract.
    let offsets = unsafe { checked_slice(offsets, n + 1, what, "offsets") }?;
    if offsets[0] != 0 {
        return Err(format!("{what}: offsets[0] is {}, expected 0", offsets[0]));
    }
    for (i, w) in offsets.windows(2).enumerate() {
        if w[1] <= w[0] {
            return Err(format!(
                "{what}: offsets[{}] = {} does not exceed offsets[{i}] = {} (a {noun} is never empty, and offsets \
                 only increase)",
                i + 1,
                w[1],
                w[0]
            ));
        }
        if w[1] - w[0] > max_rows {
            return Err(format!(
                "{what}: {noun} {i} has {} {unit}; the limit is {max_rows}{limit_hint}",
                w[1] - w[0]
            ));
        }
    }
    Ok(offsets)
}

/// Embed `n_seqs` token sequences into `out`, `[n_seqs, dim]` row-major f32,
/// each row L2-normalized, in the order given (see the module docs for the CSR
/// layout of `ids` and `offsets`).
///
/// `truncate_dim` 0 returns the full `embedding_dim` columns; any other value
/// must be in `1..=embedding_dim` and returns that Matryoshka prefix,
/// normalized over those columns ([`EmbedGemma2Model::encode`]). `out_len` is
/// the element count of `out` and must be exactly `n_seqs * dim`.
///
/// # Safety
/// As [`crate::capi::tessl_cross_entropy_rows`], for a
/// [`tessl_embedgemma2_load`] handle; `ids` points to `n_ids` readable `u32`s,
/// `offsets` to `n_seqs + 1` readable `u64`s, `out` to `out_len` writable
/// `f32`s, none of them overlapping (each is null-checked and, when `n` is
/// non-zero, alignment- and extent-checked before it is read).
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_encode(
    model: *mut TesslEmbedGemma2,
    ids: *const u32,
    n_ids: u64,
    offsets: *const u64,
    n_seqs: u64,
    truncate_dim: u32,
    out: *mut f32,
    out_len: u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_encode";
            let text = h.engine.text();
            let n_seqs = usize::try_from(n_seqs).map_err(|_| format!("{WHAT}: n_seqs {n_seqs} overflows usize"))?;
            if n_seqs == 0 || n_seqs > MAX_BATCH {
                return Err(format!("{WHAT}: n_seqs is {n_seqs}, expected 1..={MAX_BATCH}"));
            }
            let want_out = output_shape(WHAT, text.config(), n_seqs, truncate_dim, out_len)?;
            if out.is_null() || ids.is_null() || offsets.is_null() {
                return Err(format!("{WHAT}: null ids, offsets or out"));
            }
            let max_tokens = u64::from(text.max_tokens());
            let offsets = csr_offsets(
                WHAT,
                offsets,
                n_seqs,
                "sequence",
                "tokens",
                max_tokens,
                " (truncate it, or raise it with tessl_embedgemma2_set_max_tokens)",
            )?;
            // Every gap is at most max_tokens (at most MAX_BATCH_TOKENS) and there are
            // at most MAX_BATCH of them, so the last offset is bounded before it
            // sizes a slice.
            if offsets[n_seqs] != n_ids {
                return Err(format!(
                    "{WHAT}: offsets end at {} but n_ids is {n_ids}",
                    offsets[n_seqs]
                ));
            }
            let n_ids = usize::try_from(n_ids).map_err(|_| format!("{WHAT}: n_ids overflows usize"))?;
            let ids = checked_slice(ids, n_ids, WHAT, "ids")?;
            let out = checked_slice_mut(out, want_out, WHAT, "out")?;

            let batch: Vec<&[u32]> = offsets.windows(2).map(|w| &ids[w[0] as usize..w[1] as usize]).collect();
            let enc = text.encode(&batch, (truncate_dim != 0).then_some(truncate_dim), false)?;
            copy_embeddings(WHAT, &enc, out)
        })
    }
}

/// Embed `n_images` images into `out`, `[n_images, dim]` row-major f32, each
/// row the L2-normalized embedding of the processor's image sequence
/// (`<bos> <boi> <image> x S <eoi> <eos>`) with the vision tower's `S` soft
/// tokens in place of the placeholders. Needs a handle from
/// [`tessl_embedgemma2_load_multimodal`].
///
/// Image `i` is rows `offsets[i]..offsets[i + 1]` of `pixel_values`
/// (`[offsets[n_images], patch_dim]` f32, `pixel_values_len` elements) and of
/// `position_ids` (`[offsets[n_images], 2]` i32, `position_ids_len` elements),
/// the processor's `pixel_values` and `image_position_ids` with the images laid
/// end to end: padding patches `(-1, -1)` included, as a suffix. `offsets` has
/// `n_images + 1` entries, starts at 0 and increases strictly; an image has at
/// most [`MAX_BATCH_PATCHES`] patch rows, a call at most [`MAX_BATCH`] images
/// (one output row each) and [`MAX_CALL_PATCHES`] patch rows in all. Both array lengths must be exactly what the offsets imply
/// (`patch_dim` is the vision config's `3 * patch_size^2`, 768 in the
/// checkpoint); `truncate_dim` and `out_len` are as for
/// [`tessl_embedgemma2_encode`].
///
/// # Safety
/// As [`tessl_embedgemma2_encode`]; `pixel_values` points to
/// `pixel_values_len` readable `f32`s, `position_ids` to `position_ids_len`
/// readable `i32`s, `offsets` to `n_images + 1` readable `u64`s.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_encode_images(
    model: *mut TesslEmbedGemma2,
    pixel_values: *const f32,
    pixel_values_len: u64,
    position_ids: *const i32,
    position_ids_len: u64,
    offsets: *const u64,
    n_images: u64,
    truncate_dim: u32,
    out: *mut f32,
    out_len: u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_encode_images";
            let mm = h.engine.multimodal(WHAT)?;
            let n = usize::try_from(n_images).map_err(|_| format!("{WHAT}: n_images {n_images} overflows usize"))?;
            if n == 0 || n > MAX_BATCH {
                return Err(format!("{WHAT}: n_images is {n}, expected 1..={MAX_BATCH}"));
            }
            let want_out = output_shape(WHAT, mm.text().config(), n, truncate_dim, out_len)?;
            if out.is_null() || pixel_values.is_null() || position_ids.is_null() || offsets.is_null() {
                return Err(format!("{WHAT}: null pixel_values, position_ids, offsets or out"));
            }
            let offsets = csr_offsets(WHAT, offsets, n, "image", "patch rows", MAX_BATCH_PATCHES as u64, "")?;
            // At most `n * MAX_BATCH_PATCHES` rows, so the products below fit.
            let patches = offsets[n];
            // The call's memory bound, checked before the copy it avoids (the
            // tower checks it again on the padded count).
            if patches > MAX_CALL_PATCHES as u64 {
                return Err(format!(
                    "{WHAT}: the images make {patches} patch rows; the per-call bound MAX_CALL_PATCHES is \
                     {MAX_CALL_PATCHES} (split the call)"
                ));
            }
            let pd = u64::from(mm.vision().config().patch_dim());
            if pixel_values_len != patches * pd {
                return Err(format!(
                    "{WHAT}: pixel_values_len is {pixel_values_len}, expected {} (= {patches} patch rows x {pd})",
                    patches * pd
                ));
            }
            if position_ids_len != patches * 2 {
                return Err(format!(
                    "{WHAT}: position_ids_len is {position_ids_len}, expected {} (= {patches} patch rows x 2)",
                    patches * 2
                ));
            }
            let pixels = checked_slice(pixel_values, pixel_values_len as usize, WHAT, "pixel_values")?;
            let positions = checked_slice(position_ids, position_ids_len as usize, WHAT, "position_ids")?;
            let out = checked_slice_mut(out, want_out, WHAT, "out")?;

            let items: Vec<MmItem<'_>> = offsets
                .windows(2)
                .map(|w| {
                    let (a, b) = (w[0] as usize, w[1] as usize);
                    MmItem::Image(ImageInput {
                        pixel_values: &pixels[a * pd as usize..b * pd as usize],
                        position_ids: &positions[2 * a..2 * b],
                    })
                })
                .collect();
            let enc = mm.encode(&items, (truncate_dim != 0).then_some(truncate_dim))?;
            copy_embeddings(WHAT, &enc, out)
        })
    }
}

/// Embed `n_clips` audio clips into `out`, `[n_clips, dim]` row-major f32, each
/// row the L2-normalized embedding of the processor's audio sequence
/// (`<bos> <boa> <audio> x S <eoa> <eos>`) with the audio tower's `S` soft
/// tokens in place of the placeholders. Needs a handle from
/// [`tessl_embedgemma2_load_multimodal`].
///
/// Clip `i` is rows `offsets[i]..offsets[i + 1]` of `input_features`
/// (`[offsets[n_clips], mel_bins]` f32, `input_features_len` elements) and
/// entries of `input_features_mask` (`[offsets[n_clips]]` bytes, 0 for a padded
/// frame and 1 for a real one, any other byte refused, `mask_len` elements):
/// the processor's `input_features` and `input_features_mask` with the clips
/// laid end to end. A clip has at most [`MAX_FRAMES_PER_FORWARD`] frames and a
/// call at most [`MAX_BATCH`] clips (one output row each) and
/// [`MAX_CALL_FRAMES`] frames in all; `mel_bins` is the audio config's (128 in
/// the checkpoint). A clip whose mask leaves no soft token has no sequence and
/// is refused. `truncate_dim` and `out_len` are as for
/// [`tessl_embedgemma2_encode`].
///
/// # Safety
/// As [`tessl_embedgemma2_encode`]; `input_features` points to
/// `input_features_len` readable `f32`s, `input_features_mask` to `mask_len`
/// readable bytes, `offsets` to `n_clips + 1` readable `u64`s.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_encode_audio(
    model: *mut TesslEmbedGemma2,
    input_features: *const f32,
    input_features_len: u64,
    input_features_mask: *const u8,
    mask_len: u64,
    offsets: *const u64,
    n_clips: u64,
    truncate_dim: u32,
    out: *mut f32,
    out_len: u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_encode_audio";
            let mm = h.engine.multimodal(WHAT)?;
            let n = usize::try_from(n_clips).map_err(|_| format!("{WHAT}: n_clips {n_clips} overflows usize"))?;
            if n == 0 || n > MAX_BATCH {
                return Err(format!("{WHAT}: n_clips is {n}, expected 1..={MAX_BATCH}"));
            }
            let want_out = output_shape(WHAT, mm.text().config(), n, truncate_dim, out_len)?;
            if out.is_null() || input_features.is_null() || input_features_mask.is_null() || offsets.is_null() {
                return Err(format!(
                    "{WHAT}: null input_features, input_features_mask, offsets or out"
                ));
            }
            let offsets = csr_offsets(WHAT, offsets, n, "clip", "frames", MAX_FRAMES_PER_FORWARD as u64, "")?;
            // At most `n * MAX_FRAMES_PER_FORWARD` frames, so the products below fit.
            let frames = offsets[n];
            if frames > MAX_CALL_FRAMES as u64 {
                return Err(format!(
                    "{WHAT}: the clips make {frames} frames; the per-call bound MAX_CALL_FRAMES is \
                     {MAX_CALL_FRAMES} (split the call)"
                ));
            }
            let mels = u64::from(mm.audio().config().mel_bins());
            if input_features_len != frames * mels {
                return Err(format!(
                    "{WHAT}: input_features_len is {input_features_len}, expected {} (= {frames} frames x {mels})",
                    frames * mels
                ));
            }
            if mask_len != frames {
                return Err(format!(
                    "{WHAT}: mask_len is {mask_len}, expected {frames} (one byte per frame)"
                ));
            }
            let feats = checked_slice(input_features, input_features_len as usize, WHAT, "input_features")?;
            let mask = checked_slice(input_features_mask, mask_len as usize, WHAT, "input_features_mask")?;
            if let Some(at) = mask.iter().position(|&b| b > 1) {
                return Err(format!(
                    "{WHAT}: input_features_mask[{at}] is {}, expected 0 or 1",
                    mask[at]
                ));
            }
            let out = checked_slice_mut(out, want_out, WHAT, "out")?;

            // The lengths are bounded above; the clips are copied into the
            // owned form the tower takes.
            let clips: Vec<AudioInput> = offsets
                .windows(2)
                .map(|w| {
                    let (a, b) = (w[0] as usize, w[1] as usize);
                    AudioInput {
                        input_features: feats[a * mels as usize..b * mels as usize].to_vec(),
                        mask: mask[a..b].iter().map(|&m| m == 1).collect(),
                    }
                })
                .collect();
            let items: Vec<MmItem<'_>> = clips.iter().map(MmItem::Audio).collect();
            let enc = mm.encode(&items, (truncate_dim != 0).then_some(truncate_dim))?;
            copy_embeddings(WHAT, &enc, out)
        })
    }
}

// ---------------------------------------------------------------------------
// The entries that run the snapshot's own front ends
// ---------------------------------------------------------------------------

/// Longest prompt name a call takes, in bytes.
pub const TESSL_EMBEDGEMMA2_MAX_NAME_BYTES: u64 = 256;

/// Most samples one raw clip may hold, in total across the call's clips no
/// more than the caller's own array. The extractor cuts a clip to its
/// `max_length` (30 s at the checkpoint) after this, so the bound is a sanity
/// limit on what the caller may hand over, not on what is processed.
pub const TESSL_EMBEDGEMMA2_MAX_CLIP_SAMPLES: u64 = 1 << 28;

/// Most ids [`tessl_embedgemma2_decode`] takes.
pub const TESSL_EMBEDGEMMA2_MAX_DECODE_IDS: u64 = 1 << 24;

/// Most texts [`tessl_embedgemma2_tokenize`] takes in one call.
pub const TESSL_EMBEDGEMMA2_MAX_TOKENIZE_TEXTS: u64 = 1 << 16;

/// `v` as a count in `1..=max`.
fn count_arg(what: &str, name: &str, v: u64, max: usize) -> Result<usize, String> {
    match usize::try_from(v) {
        Ok(n) if (1..=max).contains(&n) => Ok(n),
        _ => Err(format!("{what}: {name} is {v}, expected 1..={max}")),
    }
}

/// The one `T` a caller's out-pointer addresses, null and misalignment refused.
///
/// # Safety
/// As [`checked_slice_mut`] for one element.
unsafe fn out_cell<'a, T>(what: &str, name: &str, p: *mut T) -> Result<&'a mut T, String> {
    // SAFETY: forwarded from this function's contract.
    let cell = unsafe { checked_slice_mut(p, 1, what, name) }?;
    cell.first_mut().ok_or_else(|| format!("{what}: {name} is empty"))
}

/// Hand `bytes` to the caller: their length is always written to `*needed`,
/// and the bytes to `buf` when it holds them. A null `buf` with `buf_len` 0 is
/// the size query (nothing else is written); a buffer too short for `bytes` is
/// refused naming both sizes, with `*needed` already written so the caller can
/// retry with a buffer that fits.
///
/// # Safety
/// `buf` is null or points to `buf_len` writable bytes (checked for extent and
/// null as [`checked_slice_mut`] does); `needed` is the cell from [`out_cell`].
unsafe fn write_bytes(
    what: &str,
    field: &str,
    bytes: &[u8],
    buf: *mut u8,
    buf_len: u64,
    needed: &mut u64,
) -> Result<(), String> {
    *needed = bytes.len() as u64;
    if buf.is_null() {
        if buf_len != 0 {
            return Err(format!("{what}: null {field} with {field}_len {buf_len}"));
        }
        return Ok(());
    }
    if buf_len < *needed {
        return Err(format!(
            "{what}: {field} holds {buf_len} bytes, {} are needed (pass a null {field} and 0 to ask for the size)",
            *needed
        ));
    }
    // SAFETY: forwarded from this function's contract; `bytes.len() <= buf_len`.
    let dst = unsafe { checked_slice_mut(buf, bytes.len(), what, field) }?;
    dst.copy_from_slice(bytes);
    Ok(())
}

/// A name argument (`len` bytes of UTF-8, 1 to [`TESSL_EMBEDGEMMA2_MAX_NAME_BYTES`]).
///
/// # Safety
/// As [`checked_slice`] over `len` bytes.
unsafe fn name_arg<'a>(what: &str, field: &str, p: *const c_char, len: u64) -> Result<&'a str, String> {
    if len == 0 || len > TESSL_EMBEDGEMMA2_MAX_NAME_BYTES {
        return Err(format!(
            "{what}: {field}_len is {len}, expected 1..={TESSL_EMBEDGEMMA2_MAX_NAME_BYTES}"
        ));
    }
    let n = usize::try_from(len).map_err(|_| format!("{what}: {field}_len overflows usize"))?;
    // SAFETY: forwarded from this function's contract.
    let bytes = unsafe { checked_slice(p.cast::<u8>(), n, what, field) }?;
    std::str::from_utf8(bytes).map_err(|_| format!("{what}: {field} is not UTF-8"))
}

/// An optional prompt name: null (with length 0) is none.
///
/// # Safety
/// As [`name_arg`].
unsafe fn prompt_arg<'a>(what: &str, p: *const c_char, len: u64) -> Result<Option<&'a str>, String> {
    if p.is_null() {
        if len != 0 {
            return Err(format!("{what}: null prompt_name with prompt_name_len {len}"));
        }
        return Ok(None);
    }
    // SAFETY: forwarded from this function's contract.
    unsafe { name_arg(what, "prompt_name", p, len) }.map(Some)
}

/// `n` texts laid end to end: `n + 1` offsets starting at 0, never decreasing
/// (an empty text is two offsets alike) and ending at `bytes_len`, which is
/// below [`MAX_TEXT_BYTES`]; each text valid UTF-8. The offsets are checked
/// before any byte is read.
///
/// # Safety
/// As [`checked_slice`] for `bytes_len` bytes and `n + 1` offsets.
unsafe fn text_layout<'a>(
    what: &str,
    bytes: *const u8,
    bytes_len: u64,
    offsets: *const u64,
    n: usize,
) -> Result<Vec<&'a str>, String> {
    if bytes_len >= MAX_TEXT_BYTES as u64 {
        return Err(format!(
            "{what}: text_bytes_len is {bytes_len}; the texts of a call hold less than {MAX_TEXT_BYTES} bytes"
        ));
    }
    // SAFETY: forwarded from this function's contract.
    let offsets = unsafe { checked_slice(offsets, n + 1, what, "text_offsets") }?;
    if offsets[0] != 0 {
        return Err(format!("{what}: text_offsets[0] is {}, expected 0", offsets[0]));
    }
    for (i, w) in offsets.windows(2).enumerate() {
        if w[1] < w[0] {
            return Err(format!(
                "{what}: text_offsets[{}] = {} is below text_offsets[{i}] = {}",
                i + 1,
                w[1],
                w[0]
            ));
        }
    }
    if offsets[n] != bytes_len {
        return Err(format!(
            "{what}: text_offsets end at {} but text_bytes_len is {bytes_len}",
            offsets[n]
        ));
    }
    let len = usize::try_from(bytes_len).map_err(|_| format!("{what}: text_bytes_len overflows usize"))?;
    // SAFETY: forwarded from this function's contract.
    let bytes = unsafe { checked_slice(bytes, len, what, "text_bytes") }?;
    offsets
        .windows(2)
        .enumerate()
        .map(|(i, w)| {
            std::str::from_utf8(&bytes[w[0] as usize..w[1] as usize])
                .map_err(|e| format!("{what}: text {i} is not UTF-8 (invalid after byte {})", e.valid_up_to()))
        })
        .collect()
}

/// The ids sentence-transformers feeds the model for each of `texts` with the
/// prompt `prompt` (else the snapshot's default one), one text at a time. With
/// `max_tokens` a text longer than it is refused as soon as it is tokenized,
/// so the ids held at once are the batch the encoder would accept.
fn tokenize_texts(
    what: &str,
    tok: &Tokenizer,
    prompt: Option<&str>,
    texts: &[&str],
    max_tokens: Option<u32>,
) -> Result<Vec<Vec<u32>>, String> {
    let prompt = prompt.or_else(|| tok.default_prompt_name());
    let mut out = Vec::with_capacity(texts.len());
    for (i, text) in texts.iter().enumerate() {
        let ids = tok
            .encode_with_prompt(prompt, text)
            .map_err(|e| format!("{what}: text {i}: {e}"))?;
        if let Some(max) = max_tokens {
            if ids.len() as u64 > u64::from(max) {
                return Err(format!(
                    "{what}: text {i} has {} tokens; the limit is {max} (shorten it, or raise it with \
                     tessl_embedgemma2_set_max_tokens)",
                    ids.len()
                ));
            }
        }
        out.push(ids);
    }
    Ok(out)
}

/// The number of named prompts in the snapshot's `config_sentence_transformers.json`
/// (0 when it has none), written to `*out`. Reads `tokenizer.json` on the first
/// call that needs the tokenizer (see the module docs).
///
/// # Safety
/// As [`tessl_embedgemma2_embedding_dim`].
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_prompt_count(
    model: *mut TesslEmbedGemma2,
    out: *mut u32,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_prompt_count";
            let TesslEmbedGemma2 { snapshot, front, .. } = h;
            let out = out_cell(WHAT, "out", out)?;
            let n = front.tokenizer(snapshot, WHAT)?.prompt_names().count();
            *out = u32::try_from(n).map_err(|_| format!("{WHAT}: {n} prompts overflow u32"))?;
            Ok(())
        })
    }
}

/// The name of prompt `index` (in the file's order, `0..prompt_count`), as
/// UTF-8 bytes with no terminator, copied to `buf` (`buf_len` bytes) with its
/// length written to `*out_len`. A null `buf` with `buf_len` 0 only asks for
/// the length; a buffer too short is refused with `*out_len` already set.
///
/// # Safety
/// As [`tessl_embedgemma2_embedding_dim`]; `buf` is null or points to `buf_len`
/// writable bytes; `out_len` is writable.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_prompt_name(
    model: *mut TesslEmbedGemma2,
    index: u32,
    buf: *mut u8,
    buf_len: u64,
    out_len: *mut u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_prompt_name";
            let TesslEmbedGemma2 { snapshot, front, .. } = h;
            let needed = out_cell(WHAT, "out_len", out_len)?;
            let tok = front.tokenizer(snapshot, WHAT)?;
            let name = tok.prompt_names().nth(index as usize).ok_or_else(|| {
                format!(
                    "{WHAT}: index {index} is past the snapshot's {} prompts",
                    tok.prompt_names().count()
                )
            })?;
            write_bytes(WHAT, "buf", name.as_bytes(), buf, buf_len, needed)
        })
    }
}

/// The text of the prompt called `name` (`name_len` bytes of UTF-8, at most
/// [`TESSL_EMBEDGEMMA2_MAX_NAME_BYTES`]), copied out as
/// [`tessl_embedgemma2_prompt_name`] copies a name. An unknown name is refused
/// listing the snapshot's.
///
/// # Safety
/// As [`tessl_embedgemma2_prompt_name`]; `name` points to `name_len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_prompt_get(
    model: *mut TesslEmbedGemma2,
    name: *const c_char,
    name_len: u64,
    buf: *mut u8,
    buf_len: u64,
    out_len: *mut u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_prompt_get";
            let TesslEmbedGemma2 { snapshot, front, .. } = h;
            let needed = out_cell(WHAT, "out_len", out_len)?;
            let name = name_arg(WHAT, "name", name, name_len)?;
            let tok = front.tokenizer(snapshot, WHAT)?;
            let prompt = tok.prompt(name).ok_or_else(|| {
                let names: Vec<&str> = tok.prompt_names().collect();
                format!("{WHAT}: no prompt named {name:?} (the snapshot has {names:?})")
            })?;
            write_bytes(WHAT, "buf", prompt.as_bytes(), buf, buf_len, needed)
        })
    }
}

/// The index (as for [`tessl_embedgemma2_prompt_name`]) of the snapshot's
/// `default_prompt_name`, written to `*out`, or -1 when it names none (the
/// checkpoint's is null). A call that takes no prompt name applies this prompt.
///
/// # Safety
/// As [`tessl_embedgemma2_embedding_dim`], with `out` an `i32`.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_default_prompt(
    model: *mut TesslEmbedGemma2,
    out: *mut i32,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_default_prompt";
            let TesslEmbedGemma2 { snapshot, front, .. } = h;
            let out = out_cell(WHAT, "out", out)?;
            let tok = front.tokenizer(snapshot, WHAT)?;
            *out = match tok.default_prompt_name() {
                None => -1,
                Some(name) => {
                    let at = tok.prompt_names().position(|n| n == name);
                    at.and_then(|i| i32::try_from(i).ok())
                        .ok_or_else(|| format!("{WHAT}: the default prompt {name:?} is not in the prompt list"))?
                }
            };
            Ok(())
        })
    }
}

/// Tokenize `n_texts` UTF-8 texts into the ids the model is given
/// (`<bos> ... <eos>` around the prompt and the text, as
/// sentence-transformers feeds them: [`Tokenizer::encode_with_prompt`]). The
/// texts are `text_bytes` (`text_bytes_len` bytes) cut by `text_offsets`
/// (`n_texts + 1` entries, see the module docs); `prompt_name` (`prompt_name_len`
/// bytes) names the prompt written in front of each, null for the snapshot's
/// default one (none for the checkpoint).
///
/// Two calls make a tokenization: with a null `ids_out` and `ids_cap` 0 the
/// call writes only `*n_ids_out`, the total number of ids; then with `ids_out`
/// of at least that many `u32`s and `id_offsets_out` of `n_texts + 1` `u64`s it
/// writes the ids end to end and the offsets that cut them (the first 0, the
/// last the total). A buffer too short is refused with `*n_ids_out` set.
/// No limit of the encoder applies: a text may be longer than `max_tokens`.
///
/// # Safety
/// As [`tessl_embedgemma2_embedding_dim`]; `text_bytes` points to
/// `text_bytes_len` readable bytes, `text_offsets` to `n_texts + 1` readable
/// `u64`s, `prompt_name` is null or points to `prompt_name_len` readable bytes,
/// `ids_out` is null or points to `ids_cap` writable `u32`s, `id_offsets_out`
/// is null (in the size query) or points to `n_texts + 1` writable `u64`s,
/// `n_ids_out` is writable; none overlap.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_tokenize(
    model: *mut TesslEmbedGemma2,
    text_bytes: *const u8,
    text_bytes_len: u64,
    text_offsets: *const u64,
    n_texts: u64,
    prompt_name: *const c_char,
    prompt_name_len: u64,
    ids_out: *mut u32,
    ids_cap: u64,
    id_offsets_out: *mut u64,
    n_ids_out: *mut u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_tokenize";
            let TesslEmbedGemma2 { snapshot, front, .. } = h;
            let n = count_arg(WHAT, "n_texts", n_texts, TESSL_EMBEDGEMMA2_MAX_TOKENIZE_TEXTS as usize)?;
            let total = out_cell(WHAT, "n_ids_out", n_ids_out)?;
            let name = prompt_arg(WHAT, prompt_name, prompt_name_len)?;
            if ids_out.is_null() && (ids_cap != 0 || !id_offsets_out.is_null()) {
                return Err(format!(
                    "{WHAT}: a null ids_out takes ids_cap 0 and a null id_offsets_out (the size query)"
                ));
            }
            let texts = text_layout(WHAT, text_bytes, text_bytes_len, text_offsets, n)?;
            let tok = front.tokenizer(snapshot, WHAT)?;
            let ids = tokenize_texts(WHAT, tok, name, &texts, None)?;
            let all: u64 = ids.iter().map(|s| s.len() as u64).sum();
            *total = all;
            if ids_out.is_null() {
                return Ok(());
            }
            if all > ids_cap {
                return Err(format!(
                    "{WHAT}: ids_out holds {ids_cap} ids, {all} are needed (pass a null ids_out and 0 to ask for the count)"
                ));
            }
            let out = checked_slice_mut(ids_out, all as usize, WHAT, "ids_out")?;
            let cuts = checked_slice_mut(id_offsets_out, n + 1, WHAT, "id_offsets_out")?;
            let mut at = 0usize;
            cuts[0] = 0;
            for (i, seq) in ids.iter().enumerate() {
                out[at..at + seq.len()].copy_from_slice(seq);
                at += seq.len();
                cuts[i + 1] = at as u64;
            }
            Ok(())
        })
    }
}

/// Decode ids to text as the tokenizer library's `decode` does (an id with no
/// token is skipped): `n_ids` ids (at most [`TESSL_EMBEDGEMMA2_MAX_DECODE_IDS`],
/// possibly none) to UTF-8 bytes with no terminator, copied out as
/// [`tessl_embedgemma2_prompt_name`] copies a name. `skip_special_tokens` is 1
/// to leave `<bos>`, `<eos>` and the other special tokens out (the library's
/// default) and 0 to keep them; any other value is refused.
///
/// # Safety
/// As [`tessl_embedgemma2_prompt_name`]; `ids` points to `n_ids` readable `u32`s.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_decode(
    model: *mut TesslEmbedGemma2,
    ids: *const u32,
    n_ids: u64,
    skip_special_tokens: u32,
    buf: *mut u8,
    buf_len: u64,
    out_len: *mut u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_decode";
            let TesslEmbedGemma2 { snapshot, front, .. } = h;
            let needed = out_cell(WHAT, "out_len", out_len)?;
            if n_ids > TESSL_EMBEDGEMMA2_MAX_DECODE_IDS {
                return Err(format!(
                    "{WHAT}: n_ids is {n_ids}, expected at most {TESSL_EMBEDGEMMA2_MAX_DECODE_IDS}"
                ));
            }
            if skip_special_tokens > 1 {
                return Err(format!(
                    "{WHAT}: skip_special_tokens is {skip_special_tokens}, expected 0 or 1"
                ));
            }
            let ids = checked_slice(ids, n_ids as usize, WHAT, "ids")?;
            let tok = front.tokenizer(snapshot, WHAT)?;
            let text = tok.decode_with(ids, skip_special_tokens == 1);
            write_bytes(WHAT, "buf", text.as_bytes(), buf, buf_len, needed)
        })
    }
}

/// Embed `n_texts` UTF-8 texts into `out`, `[n_texts, dim]` row-major f32, each
/// row L2-normalized: [`tessl_embedgemma2_tokenize`]'s ids through
/// [`tessl_embedgemma2_encode`]. Texts and prompt name are laid out as for
/// tokenize (an empty text is the sequence `<bos> <eos>`); `n_texts` is
/// `1..=`[`MAX_BATCH`] and a text that tokenizes to more than `max_tokens`
/// is refused naming its index. `truncate_dim` and `out_len` are as for
/// [`tessl_embedgemma2_encode`]. Works on a handle from either load entry.
///
/// # Safety
/// As [`tessl_embedgemma2_tokenize`], with `out` pointing to `out_len` writable `f32`s.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_embed_texts(
    model: *mut TesslEmbedGemma2,
    text_bytes: *const u8,
    text_bytes_len: u64,
    text_offsets: *const u64,
    n_texts: u64,
    prompt_name: *const c_char,
    prompt_name_len: u64,
    truncate_dim: u32,
    out: *mut f32,
    out_len: u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_embed_texts";
            let TesslEmbedGemma2 {
                engine,
                snapshot,
                front,
                ..
            } = h;
            let text = engine.text();
            let n = count_arg(WHAT, "n_texts", n_texts, MAX_BATCH)?;
            let want_out = output_shape(WHAT, text.config(), n, truncate_dim, out_len)?;
            let name = prompt_arg(WHAT, prompt_name, prompt_name_len)?;
            let texts = text_layout(WHAT, text_bytes, text_bytes_len, text_offsets, n)?;
            let out = checked_slice_mut(out, want_out, WHAT, "out")?;
            let tok = front.tokenizer(snapshot, WHAT)?;
            let ids = tokenize_texts(WHAT, tok, name, &texts, Some(text.max_tokens()))?;
            let batch: Vec<&[u32]> = ids.iter().map(Vec::as_slice).collect();
            let enc = text.encode(&batch, (truncate_dim != 0).then_some(truncate_dim), false)?;
            copy_embeddings(WHAT, &enc, out)
        })
    }
}

/// `n` images laid end to end as raw `u8` HWC samples: `dims` is `[n, 3]`
/// `u32` `(height, width, channels)`, channels 1 to 4, no dimension zero, at
/// most [`MAX_IMAGE_PIXELS`] pixels each; the products must sum to exactly
/// `pixels_len`.
///
/// # Safety
/// As [`checked_slice`] for `pixels_len` bytes and `3 * n` `u32`s.
unsafe fn raw_images<'a>(
    what: &str,
    pixels: *const u8,
    pixels_len: u64,
    dims: *const u32,
    n: usize,
) -> Result<Vec<RawImage<'a>>, String> {
    // SAFETY: forwarded from this function's contract.
    let dims = unsafe { checked_slice(dims, 3 * n, what, "image_dims") }?;
    let mut total = 0u64;
    for (i, d) in dims.chunks_exact(3).enumerate() {
        let (h, w, c) = (d[0], d[1], d[2]);
        if h == 0 || w == 0 {
            return Err(format!("{what}: image {i} is {h}x{w}, which has no pixels"));
        }
        if !(1..=4).contains(&c) {
            return Err(format!(
                "{what}: image {i} has {c} channels; 1 (gray), 2 (gray+alpha), 3 (RGB) or 4 (RGBA) expected"
            ));
        }
        let px = u64::from(h) * u64::from(w);
        if px > MAX_IMAGE_PIXELS as u64 {
            return Err(format!(
                "{what}: image {i} is {h}x{w}, above the {MAX_IMAGE_PIXELS}-pixel limit"
            ));
        }
        // At most 2^30 a picture and at most MAX_CALL_IMAGES pictures.
        total += px * u64::from(c);
    }
    if total != pixels_len {
        return Err(format!(
            "{what}: pixels_len is {pixels_len}, expected {total} (the sum of height x width x channels)"
        ));
    }
    let len = usize::try_from(pixels_len).map_err(|_| format!("{what}: pixels_len overflows usize"))?;
    // SAFETY: forwarded from this function's contract.
    let mut rest = unsafe { checked_slice(pixels, len, what, "pixels") }?;
    Ok(dims
        .chunks_exact(3)
        .map(|d| {
            let (height, width, channels) = (d[0] as usize, d[1] as usize, d[2] as usize);
            let (data, tail) = rest.split_at(height * width * channels);
            rest = tail;
            RawImage {
                width,
                height,
                channels,
                data,
            }
        })
        .collect())
}

/// One raw video: its decoded RGB frames and the source frame rate.
struct RawVideo<'a> {
    frames: Vec<RawImage<'a>>,
    fps: f64,
}

/// `n` videos laid end to end: `frames` holds all their decoded RGB `u8` frames
/// (`frames_len` bytes), `dims` is `[n, 2]` `u32` `(height, width)` (one size
/// per video), `frame_offsets` `n + 1` frame counts cut (each video has at
/// least one frame, and the whole call at most [`MAX_VIDEO_FRAMES`]), `fps`
/// one finite positive source rate per video. The byte counts must sum to
/// exactly `frames_len`.
///
/// # Safety
/// As [`checked_slice`] for `frames_len` bytes, `2 * n` `u32`s, `n + 1` `u64`s and `n` `f64`s.
unsafe fn raw_videos<'a>(
    what: &str,
    frames: *const u8,
    frames_len: u64,
    dims: *const u32,
    frame_offsets: *const u64,
    fps: *const f64,
    n: usize,
) -> Result<Vec<RawVideo<'a>>, String> {
    // SAFETY: forwarded from this function's contract.
    let (dims, fps) = unsafe {
        (
            checked_slice(dims, 2 * n, what, "video_dims")?,
            checked_slice(fps, n, what, "video_fps")?,
        )
    };
    // SAFETY: forwarded from this function's contract.
    let offsets = unsafe { csr_offsets(what, frame_offsets, n, "video", "frames", MAX_VIDEO_FRAMES as u64, "") }?;
    if offsets[n] > MAX_VIDEO_FRAMES as u64 {
        return Err(format!(
            "{what}: the videos hold {} frames; a call takes at most {MAX_VIDEO_FRAMES}",
            offsets[n]
        ));
    }
    let mut total = 0u64;
    for (i, (d, w)) in dims.chunks_exact(2).zip(offsets.windows(2)).enumerate() {
        let (h, wd) = (d[0], d[1]);
        if h == 0 || wd == 0 {
            return Err(format!("{what}: video {i} is {h}x{wd}, which has no pixels"));
        }
        let px = u64::from(h) * u64::from(wd);
        if px > MAX_IMAGE_PIXELS as u64 {
            return Err(format!(
                "{what}: video {i} has {h}x{wd} frames, above the {MAX_IMAGE_PIXELS}-pixel limit"
            ));
        }
        if !(fps[i].is_finite() && fps[i] > 0.0) {
            return Err(format!(
                "{what}: video {i}: the source fps must be positive, got {}",
                fps[i]
            ));
        }
        // At most 2^30 bytes a frame and 2^20 frames in all: below 2^50.
        total += (w[1] - w[0]) * px * 3;
    }
    if total != frames_len {
        return Err(format!(
            "{what}: frames_len is {frames_len}, expected {total} (the sum of frames x height x width x 3)"
        ));
    }
    let len = usize::try_from(frames_len).map_err(|_| format!("{what}: frames_len overflows usize"))?;
    // SAFETY: forwarded from this function's contract.
    let mut rest = unsafe { checked_slice(frames, len, what, "frames") }?;
    Ok(dims
        .chunks_exact(2)
        .zip(offsets.windows(2))
        .zip(fps)
        .map(|((d, w), &fps)| {
            let (height, width) = (d[0] as usize, d[1] as usize);
            let frames = (w[0]..w[1])
                .map(|_| {
                    let (data, tail) = rest.split_at(height * width * 3);
                    rest = tail;
                    RawImage {
                        width,
                        height,
                        channels: 3,
                        data,
                    }
                })
                .collect();
            RawVideo { frames, fps }
        })
        .collect())
}

/// `n` mono clips laid end to end: `samples` (`samples_len` f32s) cut by `n + 1`
/// strictly increasing `offsets` (a clip has at least one sample and at most
/// [`TESSL_EMBEDGEMMA2_MAX_CLIP_SAMPLES`]) ending at `samples_len`.
///
/// # Safety
/// As [`checked_slice`] for `samples_len` f32s and `n + 1` `u64`s.
unsafe fn raw_clips<'a>(
    what: &str,
    samples: *const f32,
    samples_len: u64,
    offsets: *const u64,
    n: usize,
) -> Result<Vec<&'a [f32]>, String> {
    // SAFETY: forwarded from this function's contract.
    let offsets = unsafe {
        csr_offsets(
            what,
            offsets,
            n,
            "clip",
            "samples",
            TESSL_EMBEDGEMMA2_MAX_CLIP_SAMPLES,
            "",
        )
    }?;
    if offsets[n] != samples_len {
        return Err(format!(
            "{what}: sample_offsets end at {} but samples_len is {samples_len}",
            offsets[n]
        ));
    }
    let len = usize::try_from(samples_len).map_err(|_| format!("{what}: samples_len overflows usize"))?;
    // SAFETY: forwarded from this function's contract.
    let all = unsafe { checked_slice(samples, len, what, "samples") }?;
    Ok(offsets.windows(2).map(|w| &all[w[0] as usize..w[1] as usize]).collect())
}

/// The processor config's picture settings against the vision tower's: a
/// mismatch would feed the tower patches it cannot read.
fn check_image_config(what: &str, kind: &str, cfg: &ImageConfig, vision: &VisionConfig) -> Result<(), String> {
    if cfg.patch_size != vision.patch_size || cfg.pooling_kernel != vision.pooling_kernel {
        return Err(format!(
            "{what}: processor_config.json's {kind} has patch_size {} and pooling_kernel_size {}, the vision tower \
             has {} and {}",
            cfg.patch_size, cfg.pooling_kernel, vision.patch_size, vision.pooling_kernel
        ));
    }
    Ok(())
}

/// The processor config's mel bins against the audio tower's.
fn check_audio_config(what: &str, cfg: &AudioFeatureConfig, audio: &AudioConfig) -> Result<(), String> {
    if cfg.mel_bins != audio.mel_bins() as usize {
        return Err(format!(
            "{what}: processor_config.json's feature_extractor has feature_size {}, the audio tower takes {}",
            cfg.mel_bins,
            audio.mel_bins()
        ));
    }
    Ok(())
}

/// The call's picture bounds against what the preprocessing *would* build,
/// before it builds anything: `pictures` images and video frames of
/// `patches_each` padded patches (a raw image becomes `max_patches` patches
/// whatever its size).
fn check_pictures(what: &str, pictures: usize, patches: usize) -> Result<(), String> {
    if pictures > MAX_CALL_IMAGES {
        return Err(format!(
            "{what}: {pictures} images and video frames exceed the per-call limit MAX_CALL_IMAGES = {MAX_CALL_IMAGES} \
             (split the call)"
        ));
    }
    if patches > MAX_CALL_PATCHES {
        return Err(format!(
            "{what}: the images and video frames would make {patches} padded patches; the per-call bound \
             MAX_CALL_PATCHES is {MAX_CALL_PATCHES} (split the call)"
        ));
    }
    Ok(())
}

/// The frames of the call's clips, estimated from their lengths after the
/// extractor's cut, against [`MAX_CALL_FRAMES`] (the tower checks the exact
/// count; this one bounds the features built).
fn check_clip_frames(what: &str, cfg: &AudioFeatureConfig, clips: &[&[f32]]) -> Result<(), String> {
    let frames: usize = clips.iter().map(|c| clip_frames(cfg, c.len())).sum();
    check_frames(what, frames)
}

/// The frames a clip of `samples` samples makes, estimated from its length
/// after the extractor's cut (a bound, not the count).
fn clip_frames(cfg: &AudioFeatureConfig, samples: usize) -> usize {
    let kept_max = cfg.max_samples.div_ceil(AUDIO_PAD_MULTIPLE) * AUDIO_PAD_MULTIPLE;
    samples.min(kept_max).div_ceil(AUDIO_PAD_MULTIPLE) * AUDIO_PAD_MULTIPLE / cfg.hop_length + 2
}

/// `frames` against [`MAX_CALL_FRAMES`].
fn check_frames(what: &str, frames: usize) -> Result<(), String> {
    if frames > MAX_CALL_FRAMES {
        return Err(format!(
            "{what}: the clips would make about {frames} frames; the per-call bound MAX_CALL_FRAMES is \
             {MAX_CALL_FRAMES} (split the call)"
        ));
    }
    Ok(())
}

/// How many pictures a video becomes: its sampled frames, or all of them when
/// the processor does not sample.
fn video_pictures(cfg: &VideoConfig, video: &RawVideo<'_>) -> Result<usize, String> {
    if cfg.do_sample_frames {
        Ok(sample_frame_indices(cfg, video.frames.len(), video.fps)?.len())
    } else {
        Ok(video.frames.len())
    }
}

fn prepare_images(what: &str, cfg: &ImageConfig, raws: &[RawImage<'_>]) -> Result<Vec<PreparedImage>, String> {
    raws.iter()
        .enumerate()
        .map(|(i, r)| preprocess_image(cfg, r).map_err(|e| format!("{what}: image {i}: {e}")))
        .collect()
}

fn prepare_clips(
    what: &str,
    front_end: &AudioFrontEnd,
    clips: &[&[f32]],
    sampling_rate: u32,
) -> Result<Vec<AudioInput>, String> {
    clips
        .iter()
        .enumerate()
        .map(|(i, pcm)| {
            front_end
                .extract(pcm, sampling_rate)
                .map_err(|e| format!("{what}: clip {i}: {e}"))
        })
        .collect()
}

fn prepare_videos(what: &str, cfg: &VideoConfig, videos: &[RawVideo<'_>]) -> Result<Vec<PreparedVideo>, String> {
    videos
        .iter()
        .enumerate()
        .map(|(i, v)| preprocess_video(cfg, &v.frames, v.fps).map_err(|e| format!("{what}: video {i}: {e}")))
        .collect()
}

/// Embed `n_images` raw images into `out`, `[n_images, dim]` row-major f32, each
/// row L2-normalized: the processor's resize, rescale and patchify done in Rust
/// ([`preprocess_image`], the settings of the snapshot's
/// `processor_config.json`), then the vision tower, as
/// [`tessl_embedgemma2_encode_images`] does with a processor's tensors. Needs a
/// handle from [`tessl_embedgemma2_load_multimodal`].
///
/// `pixels` is the images laid end to end, each row-major HWC `u8`;
/// `image_dims` is `[n_images, 3]` `u32` `(height, width, channels)`, channels
/// 1 (gray), 2 (gray, alpha), 3 (RGB) or 4 (RGBA; alpha is dropped, as PIL's
/// `convert("RGB")` does). The products must sum to exactly `pixels_len`.
/// `n_images` is `1..=`[`MAX_BATCH`], and the call's padded patches
/// (`n_images * max_soft_tokens * pooling_kernel²`) at most [`MAX_CALL_PATCHES`],
/// checked before the first image is resized. `truncate_dim` and `out_len` are
/// as for [`tessl_embedgemma2_encode`].
///
/// # Safety
/// As [`tessl_embedgemma2_encode`]; `pixels` points to `pixels_len` readable
/// bytes and `image_dims` to `3 * n_images` readable `u32`s.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_embed_images(
    model: *mut TesslEmbedGemma2,
    pixels: *const u8,
    pixels_len: u64,
    image_dims: *const u32,
    n_images: u64,
    truncate_dim: u32,
    out: *mut f32,
    out_len: u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_embed_images";
            let TesslEmbedGemma2 {
                engine,
                snapshot,
                front,
                ..
            } = h;
            let mm = engine.multimodal(WHAT)?;
            let n = count_arg(WHAT, "n_images", n_images, MAX_BATCH)?;
            let want_out = output_shape(WHAT, mm.text().config(), n, truncate_dim, out_len)?;
            let raws = raw_images(WHAT, pixels, pixels_len, image_dims, n)?;
            let out = checked_slice_mut(out, want_out, WHAT, "out")?;
            let pre = front.pre(snapshot, WHAT)?;
            check_image_config(WHAT, "image_processor", &pre.image, mm.vision().config())?;
            check_pictures(WHAT, n, n.saturating_mul(pre.image.max_patches()))?;
            let prepared = prepare_images(WHAT, &pre.image, &raws)?;
            let items: Vec<MmItem<'_>> = prepared.iter().map(|p| MmItem::Image(p.input())).collect();
            let enc = mm.encode(&items, (truncate_dim != 0).then_some(truncate_dim))?;
            copy_embeddings(WHAT, &enc, out)
        })
    }
}

/// Embed `n_clips` raw mono clips into `out`, `[n_clips, dim]` row-major f32,
/// each row L2-normalized: the log-mel features computed in Rust
/// ([`AudioFrontEnd`], the extractor of `processor_config.json`), then the audio
/// tower, as [`tessl_embedgemma2_encode_audio`] does with a processor's
/// features. Needs a handle from [`tessl_embedgemma2_load_multimodal`].
///
/// `samples` is the clips laid end to end (`samples_len` f32 PCM samples in
/// `[-1, 1]`, finite) cut by `sample_offsets` (`n_clips + 1` strictly increasing
/// `u64`s from 0 to `samples_len`; a clip has at least one and at most
/// [`TESSL_EMBEDGEMMA2_MAX_CLIP_SAMPLES`] samples, and is cut to the
/// extractor's 30 s). `sampling_rate` must be the extractor's (16000 for the
/// checkpoint): audio at another rate is refused, never resampled. A clip too
/// short for one frame is refused by the tower. `n_clips` is `1..=`[`MAX_BATCH`]
/// and the frames at most [`MAX_CALL_FRAMES`]. `truncate_dim` and `out_len` are
/// as for [`tessl_embedgemma2_encode`].
///
/// # Safety
/// As [`tessl_embedgemma2_encode`]; `samples` points to `samples_len` readable
/// `f32`s and `sample_offsets` to `n_clips + 1` readable `u64`s.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_embed_audio(
    model: *mut TesslEmbedGemma2,
    samples: *const f32,
    samples_len: u64,
    sample_offsets: *const u64,
    n_clips: u64,
    sampling_rate: u32,
    truncate_dim: u32,
    out: *mut f32,
    out_len: u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_embed_audio";
            let TesslEmbedGemma2 {
                engine,
                snapshot,
                front,
                ..
            } = h;
            let mm = engine.multimodal(WHAT)?;
            let n = count_arg(WHAT, "n_clips", n_clips, MAX_BATCH)?;
            let want_out = output_shape(WHAT, mm.text().config(), n, truncate_dim, out_len)?;
            let clips = raw_clips(WHAT, samples, samples_len, sample_offsets, n)?;
            let out = checked_slice_mut(out, want_out, WHAT, "out")?;
            let front_end = front.audio(snapshot, WHAT)?;
            check_audio_config(WHAT, front_end.config(), mm.audio().config())?;
            check_clip_frames(WHAT, front_end.config(), &clips)?;
            let features = prepare_clips(WHAT, front_end, &clips, sampling_rate)?;
            let items: Vec<MmItem<'_>> = features.iter().map(MmItem::Audio).collect();
            let enc = mm.encode(&items, (truncate_dim != 0).then_some(truncate_dim))?;
            copy_embeddings(WHAT, &enc, out)
        })
    }
}

/// Embed `n_videos` raw videos into `out`, `[n_videos, dim]` row-major f32, each
/// row L2-normalized: the processor's frame sampling (the snapshot's `fps` and
/// `max_frames`) and per-frame resize in Rust ([`preprocess_video`]), every
/// sampled frame through the vision tower, and the sequence laid out as the
/// processor lays it out. Needs a handle from [`tessl_embedgemma2_load_multimodal`].
///
/// `frames` is the decoded RGB `u8` frames of all the videos end to end,
/// row-major HWC; `video_dims` is `[n_videos, 2]` `u32` `(height, width)`, one
/// size per video (its frames are all that size); `frame_offsets` has
/// `n_videos + 1` strictly increasing `u64`s from 0 cutting the videos in
/// frames (a call takes at most [`MAX_VIDEO_FRAMES`] frames in all);
/// `video_fps` is each video's source frame rate (finite, positive). The byte
/// counts must sum to exactly `frames_len`. `timestamps` 1 writes each frame's
/// `mm:ss` time in front of its block (the processor's `add_timestamps = true`;
/// it needs `tokenizer.json`), 0 does not, any other value is refused.
/// `n_videos` is `1..=`[`MAX_BATCH`]; the sampled pictures count toward
/// [`MAX_CALL_IMAGES`] and [`MAX_CALL_PATCHES`], checked before the first is
/// resized. `truncate_dim` and `out_len` are as for [`tessl_embedgemma2_encode`].
///
/// # Safety
/// As [`tessl_embedgemma2_encode`]; `frames` points to `frames_len` readable
/// bytes, `video_dims` to `2 * n_videos` readable `u32`s, `frame_offsets` to
/// `n_videos + 1` readable `u64`s and `video_fps` to `n_videos` readable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_embed_videos(
    model: *mut TesslEmbedGemma2,
    frames: *const u8,
    frames_len: u64,
    video_dims: *const u32,
    frame_offsets: *const u64,
    video_fps: *const f64,
    n_videos: u64,
    timestamps: u32,
    truncate_dim: u32,
    out: *mut f32,
    out_len: u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_embed_videos";
            let TesslEmbedGemma2 {
                engine,
                snapshot,
                front,
                ..
            } = h;
            let mm = engine.multimodal(WHAT)?;
            let n = count_arg(WHAT, "n_videos", n_videos, MAX_BATCH)?;
            if timestamps > 1 {
                return Err(format!("{WHAT}: timestamps is {timestamps}, expected 0 or 1"));
            }
            let want_out = output_shape(WHAT, mm.text().config(), n, truncate_dim, out_len)?;
            let videos = raw_videos(WHAT, frames, frames_len, video_dims, frame_offsets, video_fps, n)?;
            let out = checked_slice_mut(out, want_out, WHAT, "out")?;
            let video_cfg = front.pre(snapshot, WHAT)?.video.clone();
            check_image_config(WHAT, "video_processor", &video_cfg.image, mm.vision().config())?;
            let mut pictures = 0usize;
            for (i, v) in videos.iter().enumerate() {
                pictures += video_pictures(&video_cfg, v).map_err(|e| format!("{WHAT}: video {i}: {e}"))?;
            }
            check_pictures(WHAT, pictures, pictures.saturating_mul(video_cfg.image.max_patches()))?;
            if timestamps == 1 {
                front.tokenizer(snapshot, WHAT)?;
            }
            let prepared = prepare_videos(WHAT, &video_cfg, &videos)?;
            let inputs: Vec<Vec<ImageInput<'_>>> = prepared
                .iter()
                .map(|p| p.frames.iter().map(PreparedImage::input).collect())
                .collect();
            let items: Vec<MmItem<'_>> = match (timestamps, front.tokenizer.as_ref()) {
                (1, Some(tokenizer)) => prepared
                    .iter()
                    .zip(&inputs)
                    .map(|(p, frames)| MmItem::TimestampedVideo {
                        frames,
                        timestamps: &p.timestamps,
                        tokenizer,
                    })
                    .collect(),
                (1, None) => return Err(format!("{WHAT}: the tokenizer is not loaded")),
                _ => inputs.iter().map(|frames| MmItem::Video(frames)).collect(),
            };
            let enc = mm.encode(&items, (truncate_dim != 0).then_some(truncate_dim))?;
            copy_embeddings(WHAT, &enc, out)
        })
    }
}

/// `n_items + 1` per-item cut points over a modality's `total` entries: they
/// start at 0, never decrease (an item may have none) and end at `total`.
///
/// # Safety
/// As [`checked_slice`] for `n_items + 1` `u64`s.
unsafe fn item_cuts<'a>(
    what: &str,
    name: &str,
    cuts: *const u64,
    n_items: usize,
    total: usize,
) -> Result<&'a [u64], String> {
    // SAFETY: forwarded from this function's contract.
    let cuts = unsafe { checked_slice(cuts, n_items + 1, what, name) }?;
    if cuts[0] != 0 {
        return Err(format!("{what}: {name}[0] is {}, expected 0", cuts[0]));
    }
    if let Some(i) = cuts.windows(2).position(|w| w[1] < w[0]) {
        return Err(format!(
            "{what}: {name}[{}] = {} is below {name}[{i}] = {}",
            i + 1,
            cuts[i + 1],
            cuts[i]
        ));
    }
    if cuts[n_items] != total as u64 {
        return Err(format!(
            "{what}: {name} end at {} but the call passes {total}",
            cuts[n_items]
        ));
    }
    Ok(cuts)
}

/// Embed `n_items` items of caller-built ids and the raw media their
/// placeholders stand for ([`MmItem::Media`]) into `out`, `[n_items, dim]`
/// row-major f32, each row L2-normalized. Item `i` is the ids
/// `ids[id_offsets[i]..id_offsets[i + 1]]` (placeholders included, as
/// [`tessl_embedgemma2_encode`] takes ids, at most `max_tokens` each), the
/// images `image_offsets[i]..image_offsets[i + 1]` of the call's images, its
/// clips `clip_offsets[i]..clip_offsets[i + 1]` and its videos
/// `video_offsets[i]..video_offsets[i + 1]`: an item may have none of a kind, and
/// the `j`-th image, audio and video placeholder id of its ids takes the `j`-th
/// soft row of its images, clips and video frames (the sampled frames of its
/// videos, end to end), the counts matching exactly.
///
/// The media are laid out as in [`tessl_embedgemma2_embed_images`] (`pixels`,
/// `image_dims`), [`tessl_embedgemma2_embed_audio`] (`samples`,
/// `sample_offsets`, `sampling_rate`) and [`tessl_embedgemma2_embed_videos`]
/// (`frames`, `video_dims`, `frame_offsets`, `video_fps`, always without
/// timestamps: the ids are the caller's). `n_images`, `n_clips` and `n_videos`
/// are the totals the per-item cuts end at; a modality with none may pass null
/// pointers and zero lengths (its per-item cuts are still `n_items + 1` zeros).
/// `n_items` is `1..=`[`MAX_BATCH`]; the pictures (images and sampled video
/// frames) count toward [`MAX_CALL_IMAGES`] and [`MAX_CALL_PATCHES`], the clips
/// toward [`MAX_CALL_CLIPS`] and [`MAX_CALL_FRAMES`], all checked before any is
/// preprocessed. `truncate_dim` and `out_len` are as for
/// [`tessl_embedgemma2_encode`]. Needs a handle from [`tessl_embedgemma2_load_multimodal`].
///
/// # Safety
/// As [`tessl_embedgemma2_encode`] for `ids` and `id_offsets`, and as
/// [`tessl_embedgemma2_embed_images`], [`tessl_embedgemma2_embed_audio`] and
/// [`tessl_embedgemma2_embed_videos`] for each modality's arrays; every
/// per-item cut array points to `n_items + 1` readable `u64`s.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_embed_media(
    model: *mut TesslEmbedGemma2,
    ids: *const u32,
    n_ids: u64,
    id_offsets: *const u64,
    n_items: u64,
    pixels: *const u8,
    pixels_len: u64,
    image_dims: *const u32,
    n_images: u64,
    image_offsets: *const u64,
    samples: *const f32,
    samples_len: u64,
    sample_offsets: *const u64,
    n_clips: u64,
    clip_offsets: *const u64,
    sampling_rate: u32,
    frames: *const u8,
    frames_len: u64,
    video_dims: *const u32,
    frame_offsets: *const u64,
    video_fps: *const f64,
    n_videos: u64,
    video_offsets: *const u64,
    truncate_dim: u32,
    out: *mut f32,
    out_len: u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_embed_media";
            let TesslEmbedGemma2 {
                engine,
                snapshot,
                front,
                ..
            } = h;
            let mm = engine.multimodal(WHAT)?;
            let n = count_arg(WHAT, "n_items", n_items, MAX_BATCH)?;
            let want_out = output_shape(WHAT, mm.text().config(), n, truncate_dim, out_len)?;
            // The counts bound every array before any is sized or read.
            let n_img = usize::try_from(n_images)
                .ok()
                .filter(|&c| c <= MAX_CALL_IMAGES)
                .ok_or_else(|| format!("{WHAT}: n_images is {n_images}, expected at most {MAX_CALL_IMAGES}"))?;
            let n_clip = usize::try_from(n_clips)
                .ok()
                .filter(|&c| c <= MAX_CALL_CLIPS)
                .ok_or_else(|| format!("{WHAT}: n_clips is {n_clips}, expected at most {MAX_CALL_CLIPS}"))?;
            let n_vid = usize::try_from(n_videos)
                .ok()
                .filter(|&c| c <= MAX_VIDEO_FRAMES)
                .ok_or_else(|| format!("{WHAT}: n_videos is {n_videos}, expected at most {MAX_VIDEO_FRAMES}"))?;
            if out.is_null() || ids.is_null() || id_offsets.is_null() {
                return Err(format!("{WHAT}: null ids, id_offsets or out"));
            }
            let max_tokens = u64::from(mm.text().max_tokens());
            let id_cuts = csr_offsets(
                WHAT,
                id_offsets,
                n,
                "item",
                "tokens",
                max_tokens,
                " (truncate it, or raise it with tessl_embedgemma2_set_max_tokens)",
            )?;
            if id_cuts[n] != n_ids {
                return Err(format!("{WHAT}: id_offsets end at {} but n_ids is {n_ids}", id_cuts[n]));
            }
            let image_cuts = item_cuts(WHAT, "image_offsets", image_offsets, n, n_img)?;
            let clip_cuts = item_cuts(WHAT, "clip_offsets", clip_offsets, n, n_clip)?;
            let video_cuts = item_cuts(WHAT, "video_offsets", video_offsets, n, n_vid)?;
            let all_ids = checked_slice(ids, n_ids as usize, WHAT, "ids")?;
            let imgs_raw = raw_images(WHAT, pixels, pixels_len, image_dims, n_img)?;
            let clips_raw = raw_clips_or_none(WHAT, samples, samples_len, sample_offsets, n_clip)?;
            let vids_raw = raw_videos_or_none(WHAT, frames, frames_len, video_dims, frame_offsets, video_fps, n_vid)?;
            let out = checked_slice_mut(out, want_out, WHAT, "out")?;

            // Everything the media would build, bounded before anything is built.
            let pre = front.pre(snapshot, WHAT)?.clone();
            if n_img > 0 {
                check_image_config(WHAT, "image_processor", &pre.image, mm.vision().config())?;
            }
            let mut vid_pictures = 0usize;
            if n_vid > 0 {
                check_image_config(WHAT, "video_processor", &pre.video.image, mm.vision().config())?;
                for (i, v) in vids_raw.iter().enumerate() {
                    vid_pictures += video_pictures(&pre.video, v).map_err(|e| format!("{WHAT}: video {i}: {e}"))?;
                }
            }
            check_pictures(
                WHAT,
                n_img + vid_pictures,
                n_img
                    .saturating_mul(pre.image.max_patches())
                    .saturating_add(vid_pictures.saturating_mul(pre.video.image.max_patches())),
            )?;
            let audio_front = if n_clip > 0 {
                let fe = front.audio(snapshot, WHAT)?;
                check_audio_config(WHAT, fe.config(), mm.audio().config())?;
                check_clip_frames(WHAT, fe.config(), &clips_raw)?;
                Some(fe)
            } else {
                None
            };

            let images = prepare_images(WHAT, &pre.image, &imgs_raw)?;
            let clips = match audio_front {
                Some(fe) => prepare_clips(WHAT, fe, &clips_raw, sampling_rate)?,
                None => Vec::new(),
            };
            let videos = prepare_videos(WHAT, &pre.video, &vids_raw)?;
            let image_inputs: Vec<ImageInput<'_>> = images.iter().map(PreparedImage::input).collect();
            let frame_inputs: Vec<Vec<ImageInput<'_>>> = videos
                .iter()
                .map(|v| v.frames.iter().map(PreparedImage::input).collect())
                .collect();
            let video_slices: Vec<&[ImageInput<'_>]> = frame_inputs.iter().map(Vec::as_slice).collect();
            let items: Vec<MmItem<'_>> = (0..n)
                .map(|i| {
                    let span = |cuts: &[u64]| cuts[i] as usize..cuts[i + 1] as usize;
                    MmItem::Media {
                        ids: &all_ids[span(id_cuts)],
                        images: &image_inputs[span(image_cuts)],
                        audio: &clips[span(clip_cuts)],
                        videos: &video_slices[span(video_cuts)],
                    }
                })
                .collect();
            let enc = mm.encode(&items, (truncate_dim != 0).then_some(truncate_dim))?;
            copy_embeddings(WHAT, &enc, out)
        })
    }
}

/// [`raw_clips`] for a modality that may be absent: no clips need no arrays.
///
/// # Safety
/// As [`raw_clips`].
unsafe fn raw_clips_or_none<'a>(
    what: &str,
    samples: *const f32,
    samples_len: u64,
    offsets: *const u64,
    n: usize,
) -> Result<Vec<&'a [f32]>, String> {
    if n == 0 {
        if samples_len != 0 {
            return Err(format!("{what}: samples_len is {samples_len} but n_clips is 0"));
        }
        return Ok(Vec::new());
    }
    // SAFETY: forwarded from this function's contract.
    unsafe { raw_clips(what, samples, samples_len, offsets, n) }
}

/// [`raw_videos`] for a modality that may be absent.
///
/// # Safety
/// As [`raw_videos`].
unsafe fn raw_videos_or_none<'a>(
    what: &str,
    frames: *const u8,
    frames_len: u64,
    dims: *const u32,
    frame_offsets: *const u64,
    fps: *const f64,
    n: usize,
) -> Result<Vec<RawVideo<'a>>, String> {
    if n == 0 {
        if frames_len != 0 {
            return Err(format!("{what}: frames_len is {frames_len} but n_videos is 0"));
        }
        return Ok(Vec::new());
    }
    // SAFETY: forwarded from this function's contract.
    unsafe { raw_videos(what, frames, frames_len, dims, frame_offsets, fps, n) }
}

// ---------------------------------------------------------------------------
// Files, bytes, and text interleaved with media
// ---------------------------------------------------------------------------

/// A modality's media passed as raw decoded arrays: pixels (`u8` HWC with an
/// `[n, 3]` dims table), 32-bit float PCM (native endianness, as bytes) or RGB
/// video frames (an `[n, 2]` dims table and one rate each).
pub const TESSL_EMBEDGEMMA2_SOURCE_RAW: u32 = 0;
/// A modality's media passed as UTF-8 file paths, laid end to end.
pub const TESSL_EMBEDGEMMA2_SOURCE_PATHS: u32 = 1;
/// A modality's media passed as whole encoded files in memory, laid end to end.
pub const TESSL_EMBEDGEMMA2_SOURCE_BYTES: u32 = 2;

/// A segment of [`tessl_embedgemma2_embed_segments`] that is a text: `refs`
/// indexes the call's texts.
pub const TESSL_EMBEDGEMMA2_SEGMENT_TEXT: u32 = 0;
/// A segment that is an image: `refs` indexes the call's images.
pub const TESSL_EMBEDGEMMA2_SEGMENT_IMAGE: u32 = 1;
/// A segment that is an audio clip: `refs` indexes the call's clips.
pub const TESSL_EMBEDGEMMA2_SEGMENT_AUDIO: u32 = 2;
/// A segment that is a video: `refs` indexes the call's videos.
pub const TESSL_EMBEDGEMMA2_SEGMENT_VIDEO: u32 = 3;
/// A segment that is a video laid out with the `mm:ss` timestamp in front of
/// each frame's block (the processor's `add_timestamps = true`).
pub const TESSL_EMBEDGEMMA2_SEGMENT_VIDEO_TIMESTAMPED: u32 = 4;

/// Most segments one [`tessl_embedgemma2_embed_segments`] call takes.
pub const TESSL_EMBEDGEMMA2_MAX_SEGMENTS: u64 = 1 << 16;

/// Most bytes one encoded video may hold (a video is written to a temporary
/// file for AVFoundation).
pub const TESSL_EMBEDGEMMA2_MAX_VIDEO_BYTES: u64 = 1 << 32;

/// Longest file extension hint for video bytes.
pub const TESSL_EMBEDGEMMA2_MAX_EXTENSION_BYTES: u64 = 8;

/// One encoded medium: a file to read or its bytes.
#[derive(Clone, Copy)]
enum Piece<'a> {
    Path(&'a Path),
    Bytes(&'a [u8]),
}

impl<'a> Piece<'a> {
    fn source(self) -> MediaSource<'a> {
        match self {
            Piece::Path(p) => MediaSource::Path(p),
            Piece::Bytes(b) => MediaSource::Bytes(b),
        }
    }

    /// How the medium is named in a message: the index in its table, and whether
    /// it is a file or bytes.
    fn label(self, index: usize) -> String {
        match self {
            Piece::Path(_) => format!("file {index}"),
            Piece::Bytes(_) => format!("bytes {index}"),
        }
    }
}

/// One modality's media of a call: raw decoded arrays, or encoded files.
enum Table<'a, R> {
    Raw(Vec<R>),
    Encoded(Vec<Piece<'a>>),
}

/// `n` encoded media (`source` [`TESSL_EMBEDGEMMA2_SOURCE_PATHS`] or `_BYTES`)
/// laid end to end in `blob` and cut by `n + 1` strictly increasing `offsets`
/// from 0 to `blob_len` (a medium is never empty). Each path is 1 to
/// [`TESSL_EMBEDGEMMA2_MAX_PATH_BYTES`] bytes of UTF-8 with no NUL; each
/// encoded medium at most `max_bytes`. Every offset is checked before the blob is
/// read, and nothing is read from the files here.
///
/// # Safety
/// As [`checked_slice`] for `blob_len` bytes and `n + 1` `u64`s.
#[allow(clippy::too_many_arguments)]
unsafe fn encoded_pieces<'a>(
    what: &str,
    noun: &str,
    source: u32,
    blob: *const u8,
    blob_len: u64,
    offsets: *const u64,
    n: usize,
    max_bytes: u64,
) -> Result<Vec<Piece<'a>>, String> {
    let paths = match source {
        TESSL_EMBEDGEMMA2_SOURCE_PATHS => true,
        TESSL_EMBEDGEMMA2_SOURCE_BYTES => false,
        other => {
            return Err(format!(
                "{what}: {noun}_source is {other}, expected {TESSL_EMBEDGEMMA2_SOURCE_PATHS} (UTF-8 file paths) or \
                 {TESSL_EMBEDGEMMA2_SOURCE_BYTES} (encoded bytes); raw arrays go through the entries that take \
                 them (embed_images, embed_audio, embed_videos, embed_segments)"
            ))
        }
    };
    let (unit, limit) = if paths {
        ("path bytes", TESSL_EMBEDGEMMA2_MAX_PATH_BYTES)
    } else {
        ("bytes", max_bytes)
    };
    // SAFETY: forwarded from this function's contract.
    let cuts = unsafe { csr_offsets(what, offsets, n, noun, unit, limit, "") }?;
    if cuts[n] != blob_len {
        return Err(format!(
            "{what}: {noun}_offsets end at {} but {noun}_blob_len is {blob_len}",
            cuts[n]
        ));
    }
    let len = usize::try_from(blob_len).map_err(|_| format!("{what}: {noun}_blob_len overflows usize"))?;
    // SAFETY: forwarded from this function's contract.
    let all = unsafe { checked_slice(blob, len, what, &format!("{noun}_blob")) }?;
    cuts.windows(2)
        .enumerate()
        .map(|(i, w)| {
            let part = &all[w[0] as usize..w[1] as usize];
            if !paths {
                return Ok(Piece::Bytes(part));
            }
            let text = std::str::from_utf8(part).map_err(|_| format!("{what}: {noun} {i}: the path is not UTF-8"))?;
            if text.contains('\0') {
                return Err(format!("{what}: {noun} {i}: the path contains a NUL byte"));
            }
            Ok(Piece::Path(Path::new(text)))
        })
        .collect()
}

/// A video's extension hint: for bytes, 1 to
/// [`TESSL_EMBEDGEMMA2_MAX_EXTENSION_BYTES`] bytes of UTF-8 (`mp4` when null),
/// for paths null with length 0 (a file's own name says what it is).
///
/// # Safety
/// As [`checked_slice`] for `len` bytes.
unsafe fn extension_arg<'a>(what: &str, source: u32, ext: *const c_char, len: u64) -> Result<&'a str, String> {
    if source != TESSL_EMBEDGEMMA2_SOURCE_BYTES {
        if !ext.is_null() || len != 0 {
            return Err(format!(
                "{what}: an extension hint goes with encoded bytes; pass a null extension and 0 with file paths"
            ));
        }
        return Ok("mp4");
    }
    if ext.is_null() {
        if len != 0 {
            return Err(format!("{what}: null extension with extension_len {len}"));
        }
        return Ok("mp4");
    }
    if len == 0 || len > TESSL_EMBEDGEMMA2_MAX_EXTENSION_BYTES {
        return Err(format!(
            "{what}: extension_len is {len}, expected 1..={TESSL_EMBEDGEMMA2_MAX_EXTENSION_BYTES}"
        ));
    }
    let n = usize::try_from(len).map_err(|_| format!("{what}: extension_len overflows usize"))?;
    // SAFETY: forwarded from this function's contract.
    let bytes = unsafe { checked_slice(ext.cast::<u8>(), n, what, "extension") }?;
    std::str::from_utf8(bytes).map_err(|_| format!("{what}: extension is not UTF-8"))
}

fn prepare_encoded_images(what: &str, cfg: &ImageConfig, pieces: &[Piece<'_>]) -> Result<Vec<PreparedImage>, String> {
    pieces
        .iter()
        .enumerate()
        .map(|(i, p)| prepare_image(cfg, p.source()).map_err(|e| format!("{what}: {}: {e}", p.label(i))))
        .collect()
}

/// Decode and extract each clip in turn. `refs[k]` is how many times clip `k`
/// reaches the tower (1 each when every clip is one item); the frames of what
/// has been prepared so far are checked against [`MAX_CALL_FRAMES`] as they are
/// built, so a call refuses before it has built much more than the bound.
fn prepare_encoded_clips(
    what: &str,
    front_end: &AudioFrontEnd,
    pieces: &[Piece<'_>],
    refs: &[usize],
) -> Result<Vec<AudioInput>, String> {
    let mut frames = 0usize;
    pieces
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let clip = prepare_audio(front_end, p.source()).map_err(|e| format!("{what}: {}: {e}", p.label(i)))?;
            frames = frames.saturating_add(clip.mask.len().saturating_mul(refs[i]));
            check_frames(what, frames)?;
            Ok(clip)
        })
        .collect()
}

/// Open every video's first pass (no pixels), so the call's picture bounds are
/// checked on the frames the sampler will keep before one is decoded. Returns
/// the opened videos and the pictures each makes.
fn open_videos(
    what: &str,
    cfg: &VideoConfig,
    pieces: &[Piece<'_>],
    extension: &str,
) -> Result<(Vec<OpenedVideo>, Vec<usize>), String> {
    let mut opened = Vec::with_capacity(pieces.len());
    let mut pictures = Vec::with_capacity(pieces.len());
    for (i, p) in pieces.iter().enumerate() {
        let at = |e: crate::media::MediaError| format!("{what}: {}: {e}", p.label(i));
        let video = OpenedVideo::open(p.source(), extension).map_err(at)?;
        pictures.push(video.sampled_frames(cfg).map_err(at)?);
        opened.push(video);
    }
    Ok((opened, pictures))
}

fn prepare_opened_videos(
    what: &str,
    cfg: &VideoConfig,
    pieces: &[Piece<'_>],
    opened: &[OpenedVideo],
) -> Result<Vec<PreparedVideo>, String> {
    opened
        .iter()
        .zip(pieces)
        .enumerate()
        .map(|(i, (v, p))| v.prepare(cfg).map_err(|e| format!("{what}: {}: {e}", p.label(i))))
        .collect()
}

/// Embed `n_images` images read from files or given as encoded bytes into `out`,
/// `[n_images, dim]` row-major f32, each row L2-normalized: [`crate::media`]'s
/// decoder (ImageIO: PNG, JPEG, GIF, HEIC, TIFF, BMP, ... the first image of the
/// file; RGB samples as stored, the EXIF orientation not applied, as
/// torchvision reads a path), then [`tessl_embedgemma2_embed_images`]'s
/// preprocessing and the vision tower, in one call and one wait. Needs a handle
/// from [`tessl_embedgemma2_load_multimodal`].
///
/// `source` is [`TESSL_EMBEDGEMMA2_SOURCE_PATHS`] (`blob` holds UTF-8 paths) or
/// [`TESSL_EMBEDGEMMA2_SOURCE_BYTES`] (`blob` holds encoded files); `blob`
/// (`blob_len` bytes) is the media laid end to end and `offsets` its
/// `n_images + 1` strictly increasing cuts from 0 to `blob_len`. A path is at
/// most [`TESSL_EMBEDGEMMA2_MAX_PATH_BYTES`] bytes, an encoded image at most
/// [`crate::media::image::MAX_IMAGE_FILE_BYTES`]; all are checked, with the
/// count (`1..=`[`MAX_BATCH`]) and the call's padded patches
/// ([`MAX_CALL_PATCHES`], before the first file is opened), before anything is
/// read. One image at a time is decoded and preprocessed. A file that cannot
/// be decoded is an error naming its index and the decoder's reason
/// (`file 2: cannot read ...`, `bytes 1: image: cannot be decoded ...`), and
/// the output is untouched. `truncate_dim` and `out_len` are as for
/// [`tessl_embedgemma2_encode`].
///
/// # Safety
/// As [`tessl_embedgemma2_encode`]; `blob` points to `blob_len` readable bytes and
/// `offsets` to `n_images + 1` readable `u64`s.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_embed_image_sources(
    model: *mut TesslEmbedGemma2,
    source: u32,
    blob: *const u8,
    blob_len: u64,
    offsets: *const u64,
    n_images: u64,
    truncate_dim: u32,
    out: *mut f32,
    out_len: u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_embed_image_sources";
            let TesslEmbedGemma2 {
                engine,
                snapshot,
                front,
                ..
            } = h;
            let mm = engine.multimodal(WHAT)?;
            let n = count_arg(WHAT, "n_images", n_images, MAX_BATCH)?;
            let want_out = output_shape(WHAT, mm.text().config(), n, truncate_dim, out_len)?;
            let pieces = encoded_pieces(
                WHAT,
                "image",
                source,
                blob,
                blob_len,
                offsets,
                n,
                crate::media::image::MAX_IMAGE_FILE_BYTES,
            )?;
            let out = checked_slice_mut(out, want_out, WHAT, "out")?;
            let pre = front.pre(snapshot, WHAT)?;
            check_image_config(WHAT, "image_processor", &pre.image, mm.vision().config())?;
            check_pictures(WHAT, n, n.saturating_mul(pre.image.max_patches()))?;
            let prepared = prepare_encoded_images(WHAT, &pre.image, &pieces)?;
            let items: Vec<MmItem<'_>> = prepared.iter().map(|p| MmItem::Image(p.input())).collect();
            let enc = mm.encode(&items, (truncate_dim != 0).then_some(truncate_dim))?;
            copy_embeddings(WHAT, &enc, out)
        })
    }
}

/// Embed `n_clips` audio files or encoded clips into `out`, `[n_clips, dim]`
/// row-major f32, each row L2-normalized: [`crate::media`]'s decoder
/// (AudioToolbox: WAV, AIFF, CAF, MP3, M4A/AAC, FLAC, ...; mono by the mean of
/// the channels, resampled to the extractor's 16 kHz), then
/// [`tessl_embedgemma2_embed_audio`]'s feature extraction (a clip is cut to the
/// extractor's 30 s) and the audio tower, in one call and one wait. Needs a handle
/// from [`tessl_embedgemma2_load_multimodal`].
///
/// `source`, `blob`, `blob_len` and `offsets` are as for
/// [`tessl_embedgemma2_embed_image_sources`], with an encoded clip at most
/// [`crate::media::audio::MAX_AUDIO_FILE_BYTES`]; a clip longer than 30 minutes
/// is refused by the decoder. The frames built so far are checked against
/// [`MAX_CALL_FRAMES`] after each clip. Errors name the clip's index and the
/// decoder's reason.
///
/// # Safety
/// As [`tessl_embedgemma2_embed_image_sources`], with `n_clips + 1` offsets.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_embed_audio_sources(
    model: *mut TesslEmbedGemma2,
    source: u32,
    blob: *const u8,
    blob_len: u64,
    offsets: *const u64,
    n_clips: u64,
    truncate_dim: u32,
    out: *mut f32,
    out_len: u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_embed_audio_sources";
            let TesslEmbedGemma2 {
                engine,
                snapshot,
                front,
                ..
            } = h;
            let mm = engine.multimodal(WHAT)?;
            let n = count_arg(WHAT, "n_clips", n_clips, MAX_BATCH)?;
            let want_out = output_shape(WHAT, mm.text().config(), n, truncate_dim, out_len)?;
            let pieces = encoded_pieces(
                WHAT,
                "audio",
                source,
                blob,
                blob_len,
                offsets,
                n,
                crate::media::audio::MAX_AUDIO_FILE_BYTES,
            )?;
            let out = checked_slice_mut(out, want_out, WHAT, "out")?;
            let front_end = front.audio(snapshot, WHAT)?;
            check_audio_config(WHAT, front_end.config(), mm.audio().config())?;
            let features = prepare_encoded_clips(WHAT, front_end, &pieces, &vec![1; n])?;
            let items: Vec<MmItem<'_>> = features.iter().map(MmItem::Audio).collect();
            let enc = mm.encode(&items, (truncate_dim != 0).then_some(truncate_dim))?;
            copy_embeddings(WHAT, &enc, out)
        })
    }
}

/// Embed `n_videos` video files or encoded videos into `out`, `[n_videos, dim]`
/// row-major f32, each row L2-normalized: [`crate::media`]'s decoder
/// (AVFoundation: MP4, MOV, ... H.264, HEVC; upright frames), the processor's
/// frame sampling (the snapshot's `fps` and `max_frames`, over the exact frame
/// count and source rate the file reports) decoding only the sampled frames,
/// [`tessl_embedgemma2_embed_videos`]'s per-frame resize, and the vision tower,
/// in one call and one wait. Needs a handle from
/// [`tessl_embedgemma2_load_multimodal`].
///
/// `source`, `blob`, `blob_len` and `offsets` are as for
/// [`tessl_embedgemma2_embed_image_sources`], with an encoded video at most
/// [`TESSL_EMBEDGEMMA2_MAX_VIDEO_BYTES`]. `extension` (`extension_len` bytes,
/// 1 to [`TESSL_EMBEDGEMMA2_MAX_EXTENSION_BYTES`] of letters and digits: `mp4`,
/// `mov`) is the type AVFoundation sniffs encoded bytes as, since it reads
/// files and the bytes are written to a private temporary file (removed
/// before the call returns); null with 0 means `mp4`, and paths take null with 0.
/// `timestamps` 1 writes each frame's `mm:ss` time (`index / fps`, the
/// processor's) in front of its block (it needs `tokenizer.json`), 0 does not,
/// anything else is refused. Every video's first pass (no pixels decoded) runs
/// and the sampled pictures are checked against [`MAX_CALL_IMAGES`] and
/// [`MAX_CALL_PATCHES`] before any frame is decoded. Errors name the video's
/// index and the decoder's reason.
///
/// # Safety
/// As [`tessl_embedgemma2_embed_image_sources`], with `n_videos + 1` offsets and
/// `extension` null or pointing to `extension_len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_embed_video_sources(
    model: *mut TesslEmbedGemma2,
    source: u32,
    blob: *const u8,
    blob_len: u64,
    offsets: *const u64,
    n_videos: u64,
    extension: *const c_char,
    extension_len: u64,
    timestamps: u32,
    truncate_dim: u32,
    out: *mut f32,
    out_len: u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_embed_video_sources";
            let TesslEmbedGemma2 {
                engine,
                snapshot,
                front,
                ..
            } = h;
            let mm = engine.multimodal(WHAT)?;
            let n = count_arg(WHAT, "n_videos", n_videos, MAX_BATCH)?;
            if timestamps > 1 {
                return Err(format!("{WHAT}: timestamps is {timestamps}, expected 0 or 1"));
            }
            let want_out = output_shape(WHAT, mm.text().config(), n, truncate_dim, out_len)?;
            let ext = extension_arg(WHAT, source, extension, extension_len)?;
            let pieces = encoded_pieces(
                WHAT,
                "video",
                source,
                blob,
                blob_len,
                offsets,
                n,
                TESSL_EMBEDGEMMA2_MAX_VIDEO_BYTES,
            )?;
            let out = checked_slice_mut(out, want_out, WHAT, "out")?;
            let video_cfg = front.pre(snapshot, WHAT)?.video.clone();
            check_image_config(WHAT, "video_processor", &video_cfg.image, mm.vision().config())?;
            let (opened, sampled) = open_videos(WHAT, &video_cfg, &pieces, ext)?;
            let pictures: usize = sampled.iter().sum();
            check_pictures(WHAT, pictures, pictures.saturating_mul(video_cfg.image.max_patches()))?;
            if timestamps == 1 {
                front.tokenizer(snapshot, WHAT)?;
            }
            let prepared = prepare_opened_videos(WHAT, &video_cfg, &pieces, &opened)?;
            drop(opened);
            let inputs: Vec<Vec<ImageInput<'_>>> = prepared
                .iter()
                .map(|p| p.frames.iter().map(PreparedImage::input).collect())
                .collect();
            let items: Vec<MmItem<'_>> = match (timestamps, front.tokenizer.as_ref()) {
                (1, Some(tokenizer)) => prepared
                    .iter()
                    .zip(&inputs)
                    .map(|(p, frames)| MmItem::TimestampedVideo {
                        frames,
                        timestamps: &p.timestamps,
                        tokenizer,
                    })
                    .collect(),
                (1, None) => return Err(format!("{WHAT}: the tokenizer is not loaded")),
                _ => inputs.iter().map(|frames| MmItem::Video(frames)).collect(),
            };
            let enc = mm.encode(&items, (truncate_dim != 0).then_some(truncate_dim))?;
            copy_embeddings(WHAT, &enc, out)
        })
    }
}

/// How many soft tokens (placeholders in the processor's sequence) each of
/// `n_images` images makes, written to `out` (`n_images` `u64`s, `out_len`
/// exactly that), so a caller who lays out ids of their own
/// ([`tessl_embedgemma2_embed_media`]) need not guess. `image_dims` is
/// `[n_images, 2]` `u32` `(height, width)`, no dimension zero and at most
/// [`MAX_IMAGE_PIXELS`] pixels each; the count is
/// [`crate::embedgemma2_pre::image_soft_tokens`], the processor's resize target
/// divided by the pooling cell, and no pixel is touched. `n_images` is
/// `1..=`[`MAX_CALL_IMAGES`]. Works on a handle from either load entry (it reads
/// the snapshot's `processor_config.json`).
///
/// # Safety
/// As [`tessl_embedgemma2_embedding_dim`]; `image_dims` points to `2 * n_images`
/// readable `u32`s and `out` to `out_len` writable `u64`s.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_image_soft_token_count(
    model: *mut TesslEmbedGemma2,
    image_dims: *const u32,
    n_images: u64,
    out: *mut u64,
    out_len: u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_image_soft_token_count";
            let TesslEmbedGemma2 { snapshot, front, .. } = h;
            let n = count_arg(WHAT, "n_images", n_images, MAX_CALL_IMAGES)?;
            if out_len != n as u64 {
                return Err(format!(
                    "{WHAT}: out_len is {out_len}, expected exactly {n} (one count per image)"
                ));
            }
            let dims = checked_slice(image_dims, 2 * n, WHAT, "image_dims")?;
            let out = checked_slice_mut(out, n, WHAT, "out")?;
            let pre = front.pre(snapshot, WHAT)?;
            let mut counts = Vec::with_capacity(n);
            for (i, d) in dims.chunks_exact(2).enumerate() {
                let (height, width) = (d[0], d[1]);
                if height == 0 || width == 0 {
                    return Err(format!("{WHAT}: image {i} is {height}x{width}, which has no pixels"));
                }
                if u64::from(height) * u64::from(width) > MAX_IMAGE_PIXELS as u64 {
                    return Err(format!(
                        "{WHAT}: image {i} is {height}x{width}, above the {MAX_IMAGE_PIXELS}-pixel limit"
                    ));
                }
                counts.push(
                    image_soft_tokens(&pre.image, height as usize, width as usize)
                        .map_err(|e| format!("{WHAT}: image {i}: {e}"))? as u64,
                );
            }
            out.copy_from_slice(&counts);
            Ok(())
        })
    }
}

/// How many soft tokens each of `n_clips` audio clips makes, written to `out`
/// (`n_clips` `u64`s, `out_len` exactly that): the true entries of the audio
/// tower's mask for the frames the extractor would build from a clip of that
/// many samples ([`crate::embedgemma2_pre::audio_soft_tokens`]; a clip is cut to the
/// extractor's 30 s first, so 750 at most), with no feature computed.
/// `sample_counts` is `n_clips` `u64`s, each `1..=`[`TESSL_EMBEDGEMMA2_MAX_CLIP_SAMPLES`];
/// `sampling_rate` must be the extractor's (16000: what the file decoders
/// resample to), as for [`tessl_embedgemma2_embed_audio`]. A clip too short for
/// one frame is refused by index, as the tower would. `n_clips` is
/// `1..=`[`MAX_CALL_CLIPS`]. Works on either kind of handle.
///
/// # Safety
/// As [`tessl_embedgemma2_embedding_dim`]; `sample_counts` points to `n_clips`
/// readable `u64`s and `out` to `out_len` writable `u64`s.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_audio_soft_token_count(
    model: *mut TesslEmbedGemma2,
    sample_counts: *const u64,
    n_clips: u64,
    sampling_rate: u32,
    out: *mut u64,
    out_len: u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_audio_soft_token_count";
            let TesslEmbedGemma2 { snapshot, front, .. } = h;
            let n = count_arg(WHAT, "n_clips", n_clips, MAX_CALL_CLIPS)?;
            if out_len != n as u64 {
                return Err(format!(
                    "{WHAT}: out_len is {out_len}, expected exactly {n} (one count per clip)"
                ));
            }
            let counts = checked_slice(sample_counts, n, WHAT, "sample_counts")?;
            let out = checked_slice_mut(out, n, WHAT, "out")?;
            let cfg = &front.pre(snapshot, WHAT)?.audio;
            if sampling_rate != cfg.sampling_rate {
                return Err(format!(
                    "{WHAT}: {sampling_rate} Hz audio; the extractor takes {} Hz and does not resample",
                    cfg.sampling_rate
                ));
            }
            let mut softs = Vec::with_capacity(n);
            for (i, &samples) in counts.iter().enumerate() {
                if samples == 0 || samples > TESSL_EMBEDGEMMA2_MAX_CLIP_SAMPLES {
                    return Err(format!(
                        "{WHAT}: clip {i} has {samples} samples, expected 1..={TESSL_EMBEDGEMMA2_MAX_CLIP_SAMPLES}"
                    ));
                }
                let soft = audio_soft_tokens(cfg, samples as usize);
                if soft == 0 {
                    return Err(format!(
                        "{WHAT}: clip {i} ({samples} samples) is too short for one frame"
                    ));
                }
                softs.push(soft as u64);
            }
            out.copy_from_slice(&softs);
            Ok(())
        })
    }
}

/// What each of `n_videos` videos becomes, written to `out` as `[n_videos, 2]`
/// `u64` (`out_len` exactly `2 * n_videos`): the frames the processor's
/// sampler keeps and the soft tokens each makes, so a video's placeholders
/// number their product ([`crate::embedgemma2_pre::video_soft_tokens`]); nothing
/// is decoded. `video_dims` is `[n_videos, 2]` `u32` `(height, width)` of the
/// frames (after any rotation), `frame_counts` the frames in each video (`1..=`
/// [`MAX_VIDEO_FRAMES`]) and `video_fps` each source rate (finite, positive).
/// `n_videos` is `1..=`[`MAX_CALL_IMAGES`]. Works on either kind of handle.
///
/// # Safety
/// As [`tessl_embedgemma2_embedding_dim`]; `video_dims` points to `2 * n_videos`
/// readable `u32`s, `frame_counts` to `n_videos` readable `u64`s, `video_fps` to
/// `n_videos` readable `f64`s and `out` to `out_len` writable `u64`s.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_video_soft_token_count(
    model: *mut TesslEmbedGemma2,
    video_dims: *const u32,
    frame_counts: *const u64,
    video_fps: *const f64,
    n_videos: u64,
    out: *mut u64,
    out_len: u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_video_soft_token_count";
            let TesslEmbedGemma2 { snapshot, front, .. } = h;
            let n = count_arg(WHAT, "n_videos", n_videos, MAX_CALL_IMAGES)?;
            if out_len != 2 * n as u64 {
                return Err(format!(
                    "{WHAT}: out_len is {out_len}, expected exactly {} (= {n} videos x 2: sampled frames, soft tokens a frame)",
                    2 * n
                ));
            }
            let dims = checked_slice(video_dims, 2 * n, WHAT, "video_dims")?;
            let frames = checked_slice(frame_counts, n, WHAT, "frame_counts")?;
            let fps = checked_slice(video_fps, n, WHAT, "video_fps")?;
            let out = checked_slice_mut(out, 2 * n, WHAT, "out")?;
            let cfg = &front.pre(snapshot, WHAT)?.video;
            let mut pairs = Vec::with_capacity(2 * n);
            for i in 0..n {
                let (height, width) = (dims[2 * i], dims[2 * i + 1]);
                if height == 0 || width == 0 {
                    return Err(format!("{WHAT}: video {i} is {height}x{width}, which has no pixels"));
                }
                if u64::from(height) * u64::from(width) > MAX_IMAGE_PIXELS as u64 {
                    return Err(format!(
                        "{WHAT}: video {i} has {height}x{width} frames, above the {MAX_IMAGE_PIXELS}-pixel limit"
                    ));
                }
                if frames[i] == 0 || frames[i] > MAX_VIDEO_FRAMES as u64 {
                    return Err(format!(
                        "{WHAT}: video {i} has {} frames, expected 1..={MAX_VIDEO_FRAMES}",
                        frames[i]
                    ));
                }
                let (sampled, per_frame) =
                    video_soft_tokens(cfg, height as usize, width as usize, frames[i] as usize, fps[i])
                        .map_err(|e| format!("{WHAT}: video {i}: {e}"))?;
                pairs.extend([sampled as u64, per_frame as u64]);
            }
            out.copy_from_slice(&pairs);
            Ok(())
        })
    }
}

/// One segment of an item, resolved: which of the call's texts or media it names.
#[derive(Clone, Copy)]
enum SegRef {
    Text(usize),
    Image(usize),
    Audio(usize),
    Video { index: usize, timestamps: bool },
}

/// The segment table of a [`tessl_embedgemma2_embed_segments`] call, checked:
/// each item's segments, and how many times each text and medium is named.
struct SegPlan {
    items: Vec<Vec<SegRef>>,
    image_refs: Vec<usize>,
    clip_refs: Vec<usize>,
    video_refs: Vec<usize>,
}

/// `n_items` items' segments from `kinds`/`refs`, cut by `n_items + 1`
/// strictly increasing `item_offsets` (an item has at least one segment) that
/// end at `n_segments`; `lens` the byte lengths of the texts and
/// `n_images`/`n_clips`/`n_videos` the media counts the references are checked
/// against. A reference past its table, a kind that is not one of the five, a
/// text or medium no segment names, and more text than a call may tokenize are
/// refused by index.
///
/// # Safety
/// As [`checked_slice`] for `n_segments` `u32`s and `u64`s and `n_items + 1` `u64`s.
#[allow(clippy::too_many_arguments)]
unsafe fn segment_plan(
    what: &str,
    kinds: *const u32,
    refs: *const u64,
    n_segments: u64,
    item_offsets: *const u64,
    n_items: usize,
    lens: &[usize],
    counts: [usize; 3],
) -> Result<SegPlan, String> {
    let n_seg = count_arg(what, "n_segments", n_segments, TESSL_EMBEDGEMMA2_MAX_SEGMENTS as usize)?;
    // SAFETY: forwarded from this function's contract.
    let cuts = unsafe {
        csr_offsets(
            what,
            item_offsets,
            n_items,
            "item",
            "segments",
            TESSL_EMBEDGEMMA2_MAX_SEGMENTS,
            "",
        )
    }?;
    if cuts[n_items] != n_segments {
        return Err(format!(
            "{what}: item_offsets end at {} but n_segments is {n_segments}",
            cuts[n_items]
        ));
    }
    // SAFETY: forwarded from this function's contract.
    let (kinds, refs) = unsafe {
        (
            checked_slice(kinds, n_seg, what, "segment_kinds")?,
            checked_slice(refs, n_seg, what, "segment_refs")?,
        )
    };
    let [n_images, n_clips, n_videos] = counts;
    let mut plan = SegPlan {
        items: Vec::with_capacity(n_items),
        image_refs: vec![0; n_images],
        clip_refs: vec![0; n_clips],
        video_refs: vec![0; n_videos],
    };
    let mut text_refs = vec![0usize; lens.len()];
    let mut text_bytes = 0u64;
    for w in cuts.windows(2) {
        let mut item = Vec::with_capacity((w[1] - w[0]) as usize);
        for s in w[0] as usize..w[1] as usize {
            let (kind, r) = (kinds[s], refs[s]);
            let bad = |noun: &str, len: usize| format!("{what}: segment {s} names {noun} {r}, but the call has {len}");
            let at = |len: usize| usize::try_from(r).ok().filter(|&i| i < len);
            item.push(match kind {
                TESSL_EMBEDGEMMA2_SEGMENT_TEXT => {
                    let i = at(lens.len()).ok_or_else(|| bad("text", lens.len()))?;
                    text_refs[i] += 1;
                    text_bytes += lens[i] as u64;
                    SegRef::Text(i)
                }
                TESSL_EMBEDGEMMA2_SEGMENT_IMAGE => {
                    let i = at(n_images).ok_or_else(|| bad("image", n_images))?;
                    plan.image_refs[i] += 1;
                    SegRef::Image(i)
                }
                TESSL_EMBEDGEMMA2_SEGMENT_AUDIO => {
                    let i = at(n_clips).ok_or_else(|| bad("clip", n_clips))?;
                    plan.clip_refs[i] += 1;
                    SegRef::Audio(i)
                }
                TESSL_EMBEDGEMMA2_SEGMENT_VIDEO | TESSL_EMBEDGEMMA2_SEGMENT_VIDEO_TIMESTAMPED => {
                    let i = at(n_videos).ok_or_else(|| bad("video", n_videos))?;
                    plan.video_refs[i] += 1;
                    SegRef::Video {
                        index: i,
                        timestamps: kind == TESSL_EMBEDGEMMA2_SEGMENT_VIDEO_TIMESTAMPED,
                    }
                }
                other => {
                    return Err(format!(
                        "{what}: segment {s} has kind {other}, expected {TESSL_EMBEDGEMMA2_SEGMENT_TEXT} (text), \
                         {TESSL_EMBEDGEMMA2_SEGMENT_IMAGE} (image), {TESSL_EMBEDGEMMA2_SEGMENT_AUDIO} (clip), \
                         {TESSL_EMBEDGEMMA2_SEGMENT_VIDEO} (video) or {TESSL_EMBEDGEMMA2_SEGMENT_VIDEO_TIMESTAMPED} \
                         (video with timestamps)"
                    ))
                }
            });
        }
        plan.items.push(item);
    }
    if text_bytes >= MAX_TEXT_BYTES as u64 {
        return Err(format!(
            "{what}: the segments name {text_bytes} bytes of text; a call tokenizes less than {MAX_TEXT_BYTES}"
        ));
    }
    for (noun, refs) in [
        ("text", &text_refs),
        ("image", &plan.image_refs),
        ("clip", &plan.clip_refs),
        ("video", &plan.video_refs),
    ] {
        if let Some(i) = refs.iter().position(|&c| c == 0) {
            return Err(format!("{what}: {noun} {i} is not named by any segment"));
        }
    }
    Ok(plan)
}

/// The images of a [`tessl_embedgemma2_embed_segments`] call: `n` raw
/// images (`blob` the pixels, `dims` the `[n, 3]` table, `offsets` the byte
/// cuts, which must be the sizes `dims` imply) or encoded ones.
///
/// # Safety
/// As [`raw_images`] or [`encoded_pieces`].
#[allow(clippy::too_many_arguments)]
unsafe fn image_table<'a>(
    what: &str,
    source: u32,
    blob: *const u8,
    blob_len: u64,
    offsets: *const u64,
    dims: *const u32,
    n: usize,
) -> Result<Table<'a, RawImage<'a>>, String> {
    if n == 0 {
        return empty_table(what, "image", blob_len);
    }
    if source != TESSL_EMBEDGEMMA2_SOURCE_RAW {
        // SAFETY: forwarded from this function's contract.
        let pieces = unsafe {
            encoded_pieces(
                what,
                "image",
                source,
                blob,
                blob_len,
                offsets,
                n,
                crate::media::image::MAX_IMAGE_FILE_BYTES,
            )
        }?;
        return Ok(Table::Encoded(pieces));
    }
    // SAFETY: forwarded from this function's contract.
    let raws = unsafe { raw_images(what, blob, blob_len, dims, n) }?;
    // SAFETY: forwarded from this function's contract.
    let cuts = unsafe { checked_slice(offsets, n + 1, what, "image_offsets") }?;
    let mut at = 0u64;
    for (i, r) in raws.iter().enumerate() {
        if cuts[i] != at {
            return Err(format!(
                "{what}: image_offsets[{i}] is {}, expected {at} (where the sizes in image_dims put image {i})",
                cuts[i]
            ));
        }
        at += (r.height * r.width * r.channels) as u64;
    }
    if cuts[n] != at {
        return Err(format!(
            "{what}: image_offsets end at {} but the images end at {at}",
            cuts[n]
        ));
    }
    Ok(Table::Raw(raws))
}

/// The clips of a call: `n` raw clips (`blob` holds native-endian f32 PCM,
/// `offsets` the byte cuts, each a multiple of 4) or encoded ones.
///
/// # Safety
/// As [`encoded_pieces`], and `blob` readable for `blob_len` bytes.
#[allow(clippy::too_many_arguments)]
unsafe fn clip_table<'a>(
    what: &str,
    source: u32,
    blob: *const u8,
    blob_len: u64,
    offsets: *const u64,
    sampling_rate: u32,
    n: usize,
) -> Result<Table<'a, &'a [u8]>, String> {
    if n == 0 {
        return empty_table(what, "audio", blob_len);
    }
    if source != TESSL_EMBEDGEMMA2_SOURCE_RAW {
        if sampling_rate != 0 {
            return Err(format!(
                "{what}: sampling_rate is {sampling_rate}; it describes raw PCM, so pass 0 with files or encoded bytes \
                 (the decoder resamples to the extractor's rate)"
            ));
        }
        // SAFETY: forwarded from this function's contract.
        let pieces = unsafe {
            encoded_pieces(
                what,
                "audio",
                source,
                blob,
                blob_len,
                offsets,
                n,
                crate::media::audio::MAX_AUDIO_FILE_BYTES,
            )
        }?;
        return Ok(Table::Encoded(pieces));
    }
    // SAFETY: forwarded from this function's contract.
    let cuts = unsafe {
        csr_offsets(
            what,
            offsets,
            n,
            "clip",
            "bytes",
            4 * TESSL_EMBEDGEMMA2_MAX_CLIP_SAMPLES,
            "",
        )
    }?;
    if cuts[n] != blob_len {
        return Err(format!(
            "{what}: audio_offsets end at {} but audio_blob_len is {blob_len}",
            cuts[n]
        ));
    }
    if let Some(i) = cuts.windows(2).position(|w| (w[1] - w[0]) % 4 != 0) {
        return Err(format!(
            "{what}: clip {i} is {} bytes, not a whole number of 4-byte f32 samples",
            cuts[i + 1] - cuts[i]
        ));
    }
    let len = usize::try_from(blob_len).map_err(|_| format!("{what}: audio_blob_len overflows usize"))?;
    // SAFETY: forwarded from this function's contract.
    let all = unsafe { checked_slice(blob, len, what, "audio_blob") }?;
    Ok(Table::Raw(
        cuts.windows(2).map(|w| &all[w[0] as usize..w[1] as usize]).collect(),
    ))
}

/// The videos of a call: `n` raw videos (`blob` the RGB frames, `dims` the
/// `[n, 2]` table, `fps` each rate, `offsets` the byte cuts, each a whole number
/// of frames) or encoded ones (`dims` and `fps` are not read).
///
/// # Safety
/// As [`raw_videos`] or [`encoded_pieces`].
#[allow(clippy::too_many_arguments)]
unsafe fn video_table<'a>(
    what: &str,
    source: u32,
    blob: *const u8,
    blob_len: u64,
    offsets: *const u64,
    dims: *const u32,
    fps: *const f64,
    n: usize,
) -> Result<Table<'a, RawVideo<'a>>, String> {
    if n == 0 {
        return empty_table(what, "video", blob_len);
    }
    if source != TESSL_EMBEDGEMMA2_SOURCE_RAW {
        // SAFETY: forwarded from this function's contract.
        let pieces = unsafe {
            encoded_pieces(
                what,
                "video",
                source,
                blob,
                blob_len,
                offsets,
                n,
                TESSL_EMBEDGEMMA2_MAX_VIDEO_BYTES,
            )
        }?;
        return Ok(Table::Encoded(pieces));
    }
    // SAFETY: forwarded from this function's contract.
    let (cuts, dims_s) = unsafe {
        (
            csr_offsets(what, offsets, n, "video", "bytes", u64::MAX, "")?,
            checked_slice(dims, 2 * n, what, "video_dims")?,
        )
    };
    // Byte cuts to the frame cuts `raw_videos` takes: each video a whole number
    // of its frames.
    let mut frame_cuts = vec![0u64];
    for (i, (w, d)) in cuts.windows(2).zip(dims_s.chunks_exact(2)).enumerate() {
        let frame = u64::from(d[0]) * u64::from(d[1]) * 3;
        if frame == 0 || (w[1] - w[0]) % frame != 0 {
            return Err(format!(
                "{what}: video {i} is {} bytes, not a whole number of {}x{} RGB frames",
                w[1] - w[0],
                d[0],
                d[1]
            ));
        }
        frame_cuts.push(frame_cuts[i] + (w[1] - w[0]) / frame);
    }
    // SAFETY: `frame_cuts` is a live vector of `n + 1` entries; the rest is forwarded.
    let videos = unsafe { raw_videos(what, blob, blob_len, dims, frame_cuts.as_ptr(), fps, n) }?;
    Ok(Table::Raw(videos))
}

/// A modality with none: no bytes, and its pointers are not read.
fn empty_table<'a, R>(what: &str, noun: &str, blob_len: u64) -> Result<Table<'a, R>, String> {
    if blob_len != 0 {
        return Err(format!(
            "{what}: {noun}_blob_len is {blob_len} but the call has no {noun} media"
        ));
    }
    Ok(Table::Encoded(Vec::new()))
}

/// The first `keep` samples of native-endian f32 PCM in `bytes`.
fn pcm_from_bytes(bytes: &[u8], keep: usize) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .take(keep)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Embed `n_items` items that interleave text with media into `out`,
/// `[n_items, dim]` row-major f32, each row L2-normalized, **without the caller
/// counting a placeholder**: item `i` is a list of segments (text, image,
/// clip, video), and the library lays out the processor's sequence for it
/// ([`MmItem::Segments`], [`crate::embedgemma2::EmbedGemma2Config::mixed_sequence`]):
/// the text pieces joined in order, each medium written as its `<boi> <image>
/// x n <eoi>` (or audio, or one block per video frame) with `n` from the
/// tower, the whole wrapped in `<bos>` and `<eos>`, which is what
/// sentence-transformers' structured messages produce through the processor.
/// `[text "a photo of", image 0, text "on a table"]` is one item. Needs a
/// handle from [`tessl_embedgemma2_load_multimodal`] and `tokenizer.json`.
///
/// **Texts.** `text_bytes` (`text_bytes_len` bytes) is the UTF-8 texts laid
/// end to end and cut by `text_offsets` (`n_texts + 1` entries, as for
/// [`tessl_embedgemma2_embed_texts`]); no prompt is added (a prompt is text:
/// write it in the first segment). `n_texts` may be 0, with null pointers and 0.
///
/// **Segments.** `segment_kinds[s]` is one of the `TESSL_EMBEDGEMMA2_SEGMENT_*`
/// codes and `segment_refs[s]` indexes the call's texts, images, clips or videos
/// by it; `item_offsets` (`n_items + 1` strictly increasing cuts from 0 to
/// `n_segments`) cuts the `n_segments` (at most
/// [`TESSL_EMBEDGEMMA2_MAX_SEGMENTS`]) segments into items. A text or medium
/// may be named by several segments (it is read, decoded and preprocessed once,
/// and goes through its tower once for each time it is named); one that no
/// segment names is refused, as is a reference past a table. An item's text is
/// tokenized as the processor does: the text between two media is one stretch.
///
/// **Media.** Each modality is its own table, with its own `*_source`:
/// [`TESSL_EMBEDGEMMA2_SOURCE_RAW`] (decoded arrays), `_PATHS` (UTF-8 paths) or
/// `_BYTES` (encoded files); a modality with no media passes `n = 0`, null
/// pointers and 0. In every case `*_blob` (`*_blob_len` bytes) is the media laid end
/// to end and `*_offsets` has `n + 1` strictly increasing byte cuts from 0 to
/// the blob's length. Raw images are `u8` HWC with `image_dims` `[n, 3]` `u32`
/// `(height, width, channels)` as for [`tessl_embedgemma2_embed_images`] (the
/// cuts must be the sizes the dims imply); raw clips are native-endian `f32`
/// PCM at `sampling_rate` (the extractor's, as for
/// [`tessl_embedgemma2_embed_audio`]; 0 when the clips are files or bytes) with
/// each cut a multiple of 4; raw videos are RGB `u8` frames with `video_dims`
/// `[n, 2]` `u32` `(height, width)` and `video_fps` one source rate each (null
/// for files and bytes, whose rates the decoder reports), each cut a whole number of
/// frames. The image and clip entries
/// ([`tessl_embedgemma2_embed_image_sources`] and its siblings) say what the
/// decoders accept and do; `video_extension` is as there (null with 0 for `mp4`,
/// bytes only). Every count and cut is checked before anything is read; the
/// call's pictures (images and the sampled frames of videos, once for each
/// name) are checked against [`MAX_CALL_IMAGES`] and [`MAX_CALL_PATCHES`]
/// before any file is decoded and any frame resized, and its clips' frames
/// against [`MAX_CALL_FRAMES`] (before any is built for raw clips, as each is
/// built for files). Errors name the file or the segment.
///
/// `truncate_dim` and `out_len` are as for [`tessl_embedgemma2_encode`]; one wait.
///
/// # Safety
/// As [`tessl_embedgemma2_encode`]; `text_bytes` points to `text_bytes_len`
/// readable bytes and `text_offsets` to `n_texts + 1` `u64`s (both may be null
/// with `n_texts` 0); `segment_kinds` to `n_segments` `u32`s; `segment_refs` to
/// `n_segments` `u64`s; `item_offsets` to `n_items + 1` `u64`s; each modality's
/// `*_blob` to its `*_blob_len` readable bytes, `*_offsets` to its `n + 1`
/// `u64`s, `image_dims` to `3 * n_images` `u32`s, `video_dims` to
/// `2 * n_videos` `u32`s and `video_fps` to `n_videos` `f64`s when the source is raw
/// (a modality with `n = 0` reads none of them); `video_extension` is null or
/// points to `video_extension_len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn tessl_embedgemma2_embed_segments(
    model: *mut TesslEmbedGemma2,
    text_bytes: *const u8,
    text_bytes_len: u64,
    text_offsets: *const u64,
    n_texts: u64,
    segment_kinds: *const u32,
    segment_refs: *const u64,
    n_segments: u64,
    item_offsets: *const u64,
    n_items: u64,
    image_source: u32,
    image_blob: *const u8,
    image_blob_len: u64,
    image_offsets: *const u64,
    image_dims: *const u32,
    n_images: u64,
    audio_source: u32,
    audio_blob: *const u8,
    audio_blob_len: u64,
    audio_offsets: *const u64,
    sampling_rate: u32,
    n_clips: u64,
    video_source: u32,
    video_blob: *const u8,
    video_blob_len: u64,
    video_offsets: *const u64,
    video_dims: *const u32,
    video_fps: *const f64,
    n_videos: u64,
    video_extension: *const c_char,
    video_extension_len: u64,
    truncate_dim: u32,
    out: *mut f32,
    out_len: u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_embed_segments";
            let TesslEmbedGemma2 {
                engine,
                snapshot,
                front,
                ..
            } = h;
            let mm = engine.multimodal(WHAT)?;
            let n = count_arg(WHAT, "n_items", n_items, MAX_BATCH)?;
            let want_out = output_shape(WHAT, mm.text().config(), n, truncate_dim, out_len)?;
            // The counts bound every array before any is sized or read.
            let bounded = |name: &str, v: u64, max: usize| {
                usize::try_from(v)
                    .ok()
                    .filter(|&c| c <= max)
                    .ok_or_else(|| format!("{WHAT}: {name} is {v}, expected at most {max}"))
            };
            let n_text = bounded("n_texts", n_texts, TESSL_EMBEDGEMMA2_MAX_SEGMENTS as usize)?;
            let n_img = bounded("n_images", n_images, MAX_CALL_IMAGES)?;
            let n_clip = bounded("n_clips", n_clips, MAX_CALL_CLIPS)?;
            let n_vid = bounded("n_videos", n_videos, MAX_CALL_IMAGES)?;
            if out.is_null() {
                return Err(format!("{WHAT}: null out"));
            }
            let texts: Vec<&str> = if n_text == 0 {
                if text_bytes_len != 0 {
                    return Err(format!("{WHAT}: text_bytes_len is {text_bytes_len} but n_texts is 0"));
                }
                Vec::new()
            } else {
                text_layout(WHAT, text_bytes, text_bytes_len, text_offsets, n_text)?
            };
            let lens: Vec<usize> = texts.iter().map(|t| t.len()).collect();
            let plan = segment_plan(
                WHAT,
                segment_kinds,
                segment_refs,
                n_segments,
                item_offsets,
                n,
                &lens,
                [n_img, n_clip, n_vid],
            )?;
            let image_tbl = image_table(
                WHAT,
                image_source,
                image_blob,
                image_blob_len,
                image_offsets,
                image_dims,
                n_img,
            )?;
            let clip_tbl = clip_table(
                WHAT,
                audio_source,
                audio_blob,
                audio_blob_len,
                audio_offsets,
                sampling_rate,
                n_clip,
            )?;
            let video_tbl = video_table(
                WHAT,
                video_source,
                video_blob,
                video_blob_len,
                video_offsets,
                video_dims,
                video_fps,
                n_vid,
            )?;
            let video_ext = if n_vid > 0 {
                extension_arg(WHAT, video_source, video_extension, video_extension_len)?
            } else if video_extension.is_null() && video_extension_len == 0 {
                "mp4"
            } else {
                return Err(format!(
                    "{WHAT}: an extension hint goes with encoded videos; the call has none"
                ));
            };
            let out = checked_slice_mut(out, want_out, WHAT, "out")?;

            // Everything the media would build, bounded before anything is built.
            let pre = front.pre(snapshot, WHAT)?.clone();
            if n_img > 0 {
                check_image_config(WHAT, "image_processor", &pre.image, mm.vision().config())?;
            }
            let mut opened: Vec<OpenedVideo> = Vec::new();
            let mut video_pics = vec![0usize; n_vid];
            if n_vid > 0 {
                check_image_config(WHAT, "video_processor", &pre.video.image, mm.vision().config())?;
                match &video_tbl {
                    Table::Raw(videos) => {
                        for (i, v) in videos.iter().enumerate() {
                            video_pics[i] =
                                video_pictures(&pre.video, v).map_err(|e| format!("{WHAT}: video {i}: {e}"))?;
                        }
                    }
                    Table::Encoded(pieces) => {
                        let (o, pics) = open_videos(WHAT, &pre.video, pieces, video_ext)?;
                        opened = o;
                        video_pics = pics;
                    }
                }
            }
            let image_pictures: usize = plan.image_refs.iter().sum();
            let video_pictures_total: usize = video_pics.iter().zip(&plan.video_refs).map(|(p, r)| p * r).sum();
            check_pictures(
                WHAT,
                image_pictures + video_pictures_total,
                image_pictures
                    .saturating_mul(pre.image.max_patches())
                    .saturating_add(video_pictures_total.saturating_mul(pre.video.image.max_patches())),
            )?;
            if n_clip > 0 {
                if let Table::Raw(pcms) = &clip_tbl {
                    let frames: usize = pcms
                        .iter()
                        .zip(&plan.clip_refs)
                        .map(|(p, r)| clip_frames(&pre.audio, p.len() / 4).saturating_mul(*r))
                        .fold(0, usize::saturating_add);
                    check_frames(WHAT, frames)?;
                }
            }
            // The tokenizer is loaded now (a mutable use of the handle's front ends)
            // and borrowed once the audio front end is done with.
            front.tokenizer(snapshot, WHAT)?;

            let images: Vec<PreparedImage> = match &image_tbl {
                Table::Raw(raws) => prepare_images(WHAT, &pre.image, raws)?,
                Table::Encoded(pieces) => prepare_encoded_images(WHAT, &pre.image, pieces)?,
            };
            let clips: Vec<AudioInput> = if n_clip == 0 {
                Vec::new()
            } else {
                let fe = front.audio(snapshot, WHAT)?;
                check_audio_config(WHAT, fe.config(), mm.audio().config())?;
                match &clip_tbl {
                    Table::Raw(pcms) => {
                        let keep = pre.audio.max_samples.div_ceil(AUDIO_PAD_MULTIPLE) * AUDIO_PAD_MULTIPLE;
                        pcms.iter()
                            .enumerate()
                            .map(|(i, p)| {
                                fe.extract(&pcm_from_bytes(p, keep), sampling_rate)
                                    .map_err(|e| format!("{WHAT}: clip {i}: {e}"))
                            })
                            .collect::<Result<Vec<_>, _>>()?
                    }
                    Table::Encoded(pieces) => prepare_encoded_clips(WHAT, fe, pieces, &plan.clip_refs)?,
                }
            };
            let videos: Vec<PreparedVideo> = match &video_tbl {
                Table::Raw(raws) => prepare_videos(WHAT, &pre.video, raws)?,
                Table::Encoded(pieces) => prepare_opened_videos(WHAT, &pre.video, pieces, &opened)?,
            };
            drop(opened);
            let tokenizer = front
                .tokenizer
                .as_ref()
                .ok_or_else(|| format!("{WHAT}: the tokenizer is not loaded"))?;

            let image_inputs: Vec<ImageInput<'_>> = images.iter().map(PreparedImage::input).collect();
            let frame_inputs: Vec<Vec<ImageInput<'_>>> = videos
                .iter()
                .map(|v| v.frames.iter().map(PreparedImage::input).collect())
                .collect();
            let segments: Vec<Vec<MmSegment<'_>>> = plan
                .items
                .iter()
                .map(|item| {
                    item.iter()
                        .map(|seg| match *seg {
                            SegRef::Text(i) => MmSegment::Text(texts[i]),
                            SegRef::Image(i) => MmSegment::Image(image_inputs[i]),
                            SegRef::Audio(i) => MmSegment::Audio(&clips[i]),
                            SegRef::Video {
                                index,
                                timestamps: false,
                            } => MmSegment::Video(&frame_inputs[index]),
                            SegRef::Video {
                                index,
                                timestamps: true,
                            } => MmSegment::TimestampedVideo {
                                frames: &frame_inputs[index],
                                timestamps: &videos[index].timestamps,
                            },
                        })
                        .collect()
                })
                .collect();
            let items: Vec<MmItem<'_>> = segments
                .iter()
                .map(|segments| MmItem::Segments { segments, tokenizer })
                .collect();
            let enc = mm.encode(&items, (truncate_dim != 0).then_some(truncate_dim))?;
            copy_embeddings(WHAT, &enc, out)
        })
    }
}

/// Test hook: the text encoder behind a handle, so a test can compare the C
/// entry's output with a direct [`EmbedGemma2Model::encode`] on the very same
/// weights.
///
/// # Safety
/// `model` is a live handle from [`tessl_embedgemma2_load`], not freed while
/// the returned reference is used.
#[doc(hidden)]
pub unsafe fn test_embedgemma2_model<'a>(model: *const TesslEmbedGemma2) -> &'a EmbedGemma2Model {
    // SAFETY: live by the contract.
    unsafe { (*model).engine.text() }
}

/// Test hook: the multimodal model behind a handle from
/// [`tessl_embedgemma2_load_multimodal`], `None` for a text-only one.
///
/// # Safety
/// As [`test_embedgemma2_model`].
#[doc(hidden)]
pub unsafe fn test_embedgemma2_multimodal<'a>(model: *const TesslEmbedGemma2) -> Option<&'a EmbedGemma2Multimodal> {
    // SAFETY: live by the contract.
    match unsafe { &(*model).engine } {
        Engine::Multimodal(m) => Some(m),
        Engine::Text(_) => None,
    }
}

/// Test hook: the runtime the model behind a handle runs on, so a test can
/// switch the runtime's relaxed f32 GEMMs on and see which handles accept it.
///
/// # Safety
/// As [`test_embedgemma2_model`].
#[doc(hidden)]
pub unsafe fn test_embedgemma2_runtime<'a>(model: *const TesslEmbedGemma2) -> &'a Arc<GpuRuntime> {
    // SAFETY: live by the contract.
    unsafe { &(*model).rt }
}
