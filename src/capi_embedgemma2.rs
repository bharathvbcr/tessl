//! The C ABI's EmbeddingGemma 2 entries, so a caller that holds the
//! tokenizer (Python) can embed text without a Rust toolchain:
//! [`tessl_embedgemma2_load`] reads a snapshot directory, and
//! [`tessl_embedgemma2_encode`] turns token ids into L2-normalized
//! embeddings ([`EmbedGemma2Model::encode`]).
//!
//! The handle, thread-affinity, status-code and error-message conventions are
//! `src/capi.rs`'s (see its module docs): every entry runs through the same
//! `guarded` (null and foreign-thread handles refused before any reference is
//! formed, panics caught, the runtime idle on return) and `free_handle`. They
//! live in their own module so the EmbeddingGemma 2 surface does not share a
//! file with the training entries; nothing here is a second implementation of
//! the handle machinery.
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
//! # What is checked, and when
//!
//! Everything about the call is checked before anything is allocated, mutated
//! or sent to the GPU, and every caller-supplied length is bounded before it
//! sizes a slice: `n_seqs` by [`MAX_BATCH`], each sequence by the model's
//! `max_tokens`, so `n_ids` is bounded by their product and never taken on
//! trust; `out_len` must equal `n_seqs * dim` exactly. The output buffer is
//! written only when the whole batch has succeeded, so a refused or failed call
//! leaves it as it was. Token ids beyond the vocabulary are refused by
//! [`EmbedGemma2Model::encode`] before it touches the device.
//!
//! The runtime must not have relaxed precision switched on (`encode` refuses
//! it by name); nothing in the C ABI turns it on.

use std::ffi::c_char;
use std::io::Read;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::Arc;
use std::thread::ThreadId;

use crate::capi::{checked_slice, checked_slice_mut, free_handle, guarded, Handle, TesslRuntime, TESSL_OK};
use crate::embedgemma2::{EmbedGemma2Config, EmbedGemma2Model, MAX_BATCH};
use crate::runtime::GpuRuntime;
use crate::safetensors::SafeTensors;

/// The tensor-name prefix of `google/embeddinggemma-2`'s `model.safetensors`.
const PREFIX: &str = "language_model.";

/// Longest snapshot-directory path [`tessl_embedgemma2_load`] accepts, in
/// bytes. macOS's `PATH_MAX` is 1024; this leaves room for a longer
/// canonicalized form without letting a caller's length size a copy.
pub const TESSL_EMBEDGEMMA2_MAX_PATH_BYTES: u64 = 4096;

/// Largest `config.json` read, in bytes (the checkpoint's is a few KiB). A
/// file past it is refused rather than read whole into memory.
const MAX_CONFIG_BYTES: u64 = 1 << 20;

/// What a [`tessl_embedgemma2_load`] pointer owns.
pub struct TesslEmbedGemma2 {
    model: EmbedGemma2Model,
    /// The model's own clone of the runtime ([`EmbedGemma2Model`] keeps its
    /// copy private); the model stays valid however the caller treats the
    /// runtime handle it was loaded on.
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

/// `config.json` parsed, after a bounded read of a regular file (not a FIFO,
/// not a directory), which [`EmbedGemma2Config::from_config_file`] does not
/// ask for.
fn read_config(path: &Path) -> Result<EmbedGemma2Config, String> {
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
    EmbedGemma2Config::from_config_json(&text).map_err(|e| format!("{}: {e}", path.display()))
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
        if !out.is_null() {
            *out = std::ptr::null_mut();
        }
        let mut built: Option<Box<TesslEmbedGemma2>> = None;
        let status = guarded(runtime, err, err_len, |h| {
            const WHAT: &str = "tessl_embedgemma2_load";
            if out.is_null() {
                return Err(format!("{WHAT}: null out"));
            }
            if snapshot_dir_len == 0 || snapshot_dir_len > TESSL_EMBEDGEMMA2_MAX_PATH_BYTES {
                return Err(format!(
                    "{WHAT}: snapshot_dir is {snapshot_dir_len} bytes, expected 1..={TESSL_EMBEDGEMMA2_MAX_PATH_BYTES}"
                ));
            }
            let n =
                usize::try_from(snapshot_dir_len).map_err(|_| format!("{WHAT}: snapshot_dir_len overflows usize"))?;
            let bytes = checked_slice(snapshot_dir.cast::<u8>(), n, WHAT, "snapshot_dir")?;
            let dir = std::str::from_utf8(bytes).map_err(|_| format!("{WHAT}: snapshot_dir is not UTF-8"))?;
            if dir.contains('\0') {
                return Err(format!("{WHAT}: snapshot_dir contains a NUL byte"));
            }
            let dir = Path::new(dir);
            let cfg = read_config(&dir.join("config.json"))?;
            let st = SafeTensors::open(&dir.join("model.safetensors"))?;
            let rt = Arc::clone(h.runtime());
            let model = EmbedGemma2Model::load(&rt, &st, PREFIX, cfg)?;
            built = Some(Box::new(TesslEmbedGemma2 {
                model,
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

/// Free a handle from [`tessl_embedgemma2_load`]. Null is ignored
/// ([`TESSL_OK`]); from a thread other than the creator's the handle is leaked
/// and `TESSL_ERR` returned, as [`crate::capi::tessl_runtime_free`] does.
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
            *out = h.model.config().embedding_dim;
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
            *out = h.model.max_tokens();
            Ok(())
        })
    }
}

/// Raise or lower the per-sequence token limit
/// ([`EmbedGemma2Model::set_max_tokens`]): `max_tokens` in
/// `1..=`[`crate::embedgemma2::MAX_BATCH_TOKENS`], else refused and the limit
/// unchanged.
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
    unsafe { guarded(model, err, err_len, |h| h.model.set_max_tokens(max_tokens)) }
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
            let cfg = h.model.config();
            let n_seqs = usize::try_from(n_seqs).map_err(|_| format!("{WHAT}: n_seqs {n_seqs} overflows usize"))?;
            if n_seqs == 0 || n_seqs > MAX_BATCH {
                return Err(format!("{WHAT}: n_seqs is {n_seqs}, expected 1..={MAX_BATCH}"));
            }
            let dim_out = if truncate_dim == 0 {
                cfg.embedding_dim
            } else {
                truncate_dim
            };
            if dim_out > cfg.embedding_dim {
                return Err(format!(
                    "{WHAT}: truncate_dim {truncate_dim} must be 0 (full) or in 1..={}",
                    cfg.embedding_dim
                ));
            }
            let want_out = n_seqs
                .checked_mul(dim_out as usize)
                .ok_or_else(|| format!("{WHAT}: n_seqs * dim overflows usize"))?;
            if out_len != want_out as u64 {
                return Err(format!(
                    "{WHAT}: out_len is {out_len}, expected exactly {want_out} (= {n_seqs} sequences x {dim_out})"
                ));
            }
            if out.is_null() || ids.is_null() || offsets.is_null() {
                return Err(format!("{WHAT}: null ids, offsets or out"));
            }
            let offsets = checked_slice(offsets, n_seqs + 1, WHAT, "offsets")?;
            if offsets[0] != 0 {
                return Err(format!("{WHAT}: offsets[0] is {}, expected 0", offsets[0]));
            }
            let max_tokens = u64::from(h.model.max_tokens());
            for (i, w) in offsets.windows(2).enumerate() {
                if w[1] <= w[0] {
                    return Err(format!(
                        "{WHAT}: offsets[{}] = {} does not exceed offsets[{i}] = {} (a sequence is never empty, \
                         and offsets only increase)",
                        i + 1,
                        w[1],
                        w[0]
                    ));
                }
                if w[1] - w[0] > max_tokens {
                    return Err(format!(
                        "{WHAT}: sequence {i} has {} tokens; the limit is {max_tokens} (truncate it, or raise it \
                         with tessl_embedgemma2_set_max_tokens)",
                        w[1] - w[0]
                    ));
                }
            }
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
            let enc = h.model.encode(&batch, trunc, false)?;
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

/// Test hook: the model behind a handle, so a test can compare the C entry's
/// output with a direct [`EmbedGemma2Model::encode`] on the very same weights.
///
/// # Safety
/// `model` is a live handle from [`tessl_embedgemma2_load`], not freed while
/// the returned reference is used.
#[doc(hidden)]
pub unsafe fn test_embedgemma2_model<'a>(model: *const TesslEmbedGemma2) -> &'a EmbedGemma2Model {
    // SAFETY: live by the contract.
    &unsafe { &*model }.model
}
