//! The C ABI's EmbeddingGemma 2 entries, so a caller that holds the
//! tokenizer or the processor (Python) can embed text, images and audio
//! without a Rust toolchain: [`tessl_embedgemma2_load`] reads a snapshot
//! directory and [`tessl_embedgemma2_encode`] turns token ids into
//! L2-normalized embeddings ([`EmbedGemma2Model::encode`]);
//! [`tessl_embedgemma2_load_multimodal`] loads the towers too, and
//! [`tessl_embedgemma2_encode_images`] and [`tessl_embedgemma2_encode_audio`]
//! turn a processor's `pixel_values` / `input_features` into embeddings
//! ([`EmbedGemma2Multimodal::encode`]).
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
//! trust; `n_images` by [`MAX_IMAGES`] and each image's `P` by
//! [`MAX_BATCH_PATCHES`]; `n_clips` by [`MAX_CLIPS`] and each clip's `F` by
//! [`MAX_FRAMES_PER_FORWARD`]; every array length must equal exactly what its
//! offsets imply, and `out_len` must equal `n * dim`. The output buffer is
//! written only when the whole batch has succeeded, so a refused or failed call
//! leaves it as it was. Token ids beyond the vocabulary are refused by
//! [`EmbedGemma2Model::encode`] before it touches the device; the pixel and
//! feature checks are the towers' own ([`EmbedGemma2Multimodal::encode`]). A
//! clip's features are copied into the host memory the tower takes them in
//! (`AudioInput` owns its rows) after its length was bounded.
//!
//! The runtime must not have relaxed precision switched on (`encode` refuses
//! it by name, unless the handle holds `operands` bf16: the text encoder and
//! both towers then run no f32 GEMM); nothing in the C ABI turns it on. The
//! handle's `operands` applies to the text encoder and both towers.

use std::ffi::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::Arc;
use std::thread::ThreadId;

use crate::capi::{
    checked_slice, checked_slice_mut, free_handle, guarded, parse_operands, Handle, TesslRuntime, TESSL_OK,
};
use crate::embedgemma2::{EmbedGemma2Config, EmbedGemma2Model, MAX_BATCH};
use crate::embedgemma2_audio::{AudioInput, MAX_CLIPS, MAX_FRAMES_PER_FORWARD};
use crate::embedgemma2_mm::{read_config_text, EmbedGemma2Multimodal, MmItem, TEXT_PREFIX};
use crate::embedgemma2_vision::{ImageInput, MAX_BATCH_PATCHES, MAX_IMAGES};
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
            let trunc = (truncate_dim != 0).then_some(truncate_dim);
            let enc = text.encode(&batch, trunc, false)?;
            if enc.embeddings.len() != out.len() {
                return Err(format!(
                    "{WHAT}: the model returned {} values for {} expected",
                    enc.embeddings.len(),
                    out.len()
                ));
            }
            out.copy_from_slice(&enc.embeddings);
            Ok(())
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
/// most [`MAX_BATCH_PATCHES`] patch rows and a call at most [`MAX_IMAGES`]
/// images. Both array lengths must be exactly what the offsets imply
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
            if n == 0 || n > MAX_IMAGES {
                return Err(format!("{WHAT}: n_images is {n}, expected 1..={MAX_IMAGES}"));
            }
            let want_out = output_shape(WHAT, mm.text().config(), n, truncate_dim, out_len)?;
            if out.is_null() || pixel_values.is_null() || position_ids.is_null() || offsets.is_null() {
                return Err(format!("{WHAT}: null pixel_values, position_ids, offsets or out"));
            }
            let offsets = csr_offsets(WHAT, offsets, n, "image", "patch rows", MAX_BATCH_PATCHES as u64, "")?;
            // At most `n * MAX_BATCH_PATCHES` rows, so the products below fit.
            let patches = offsets[n];
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
            let trunc = (truncate_dim != 0).then_some(truncate_dim);
            let enc = mm.encode(&items, trunc)?;
            if enc.embeddings.len() != out.len() {
                return Err(format!(
                    "{WHAT}: the model returned {} values for {} expected",
                    enc.embeddings.len(),
                    out.len()
                ));
            }
            out.copy_from_slice(&enc.embeddings);
            Ok(())
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
/// call at most [`MAX_CLIPS`] clips; `mel_bins` is the audio config's (128 in
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
            if n == 0 || n > MAX_CLIPS {
                return Err(format!("{WHAT}: n_clips is {n}, expected 1..={MAX_CLIPS}"));
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
            let trunc = (truncate_dim != 0).then_some(truncate_dim);
            let enc = mm.encode(&items, trunc)?;
            if enc.embeddings.len() != out.len() {
                return Err(format!(
                    "{WHAT}: the model returned {} values for {} expected",
                    enc.embeddings.len(),
                    out.len()
                ));
            }
            out.copy_from_slice(&enc.embeddings);
            Ok(())
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
