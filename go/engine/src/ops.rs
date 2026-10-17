//! The operations: each decodes its request frame, calls the matching
//! `tessl_embedgemma2_*` entry of tessl's C ABI, and encodes the reply.
//!
//! # Why the model lives in a thread-local
//!
//! tessl's handles are thread-affine: `guarded` refuses a call on any thread
//! but the one that made the handle, and freeing from another thread leaks it.
//! Gusset runs handlers on its own worker threads, so the model cannot be made
//! on the Go side and passed in. The Go package opens one gusset handle with a
//! pool of exactly one worker per loaded model; that worker's thread makes the
//! runtime and the model in [`load`] and keeps them here, in this thread's
//! slot, until [`free`] (or the thread's end) drops them. Every later call on
//! that handle reaches the same worker, so the slot is the model.
//!
//! One worker is also the honest concurrency of the C ABI: `guarded` takes the
//! handle by `&mut`, so two calls on one model could not overlap anyway. Many
//! goroutines calling one model queue on gusset's semaphore in Go (cancellable,
//! no thread each), not on an OS thread in cgo.
//!
//! # What is validated here
//!
//! Only what keeps this crate memory-safe: a frame is read with bounds checks,
//! and every pointer handed to tessl comes with the length of the vector it
//! points into. Whether the values make sense (ranges, limits, UTF-8 where it
//! matters, dimensions that fit the data) is tessl's own check, and its message
//! is what the caller gets. Output buffers are sized from counts bounded by
//! tessl's own batch limits before they are allocated, never from a count read
//! off the frame unchecked.

use std::cell::RefCell;
use std::ffi::{c_char, CStr};
use std::ptr;
use std::time::{Duration, Instant};

use tessl::capi::{tessl_runtime_free, tessl_runtime_new, tessl_runtime_poisoned, TesslRuntime, TESSL_OK, TESSL_PANIC};
use tessl::capi_embedgemma2::{self as c, TesslEmbedGemma2};
use tessl::embedgemma2::MAX_BATCH;
use tessl::embedgemma2_audio::MAX_CALL_CLIPS;
use tessl::embedgemma2_vision::MAX_CALL_IMAGES;

use crate::wire::{Reader, Reply, FLAG_POISONED, STATUS_ERR, STATUS_NO_MODEL, STATUS_OK};

/// Bytes of the buffer tessl writes an error message into. A longer message is
/// cut at a UTF-8 boundary by tessl.
const ERR_LEN: usize = 4096;

/// The most prompts a snapshot may list before this crate stops asking for
/// them one by one.
const MAX_PROMPTS: u32 = 1 << 16;

/// How an operation failed.
pub enum Fail {
    /// The frame could not be read: a bug in the caller of this crate. Goes
    /// out through gusset's error path, not as a reply.
    Frame(String),
    /// tessl's refusal, with its status and message.
    Tessl { status: i32, message: String },
}

impl From<String> for Fail {
    fn from(message: String) -> Self {
        Fail::Frame(message)
    }
}

type Done = Result<Reply, Fail>;

/// A loaded model and the runtime it was made on, both owned by the worker
/// thread that made them.
struct Model {
    rt: *mut TesslRuntime,
    handle: *mut TesslEmbedGemma2,
    dim: u32,
    multimodal: bool,
    /// A call panicked or the runtime is poisoned: every later call is refused
    /// here without reaching tessl.
    dead: bool,
}

impl Model {
    /// Free the model, then its runtime, on this thread. Returns the first
    /// non-OK status (tessl leaks a handle freed from the wrong thread).
    fn release(&mut self) -> i32 {
        let mut status = TESSL_OK;
        if !self.handle.is_null() {
            // SAFETY: the handle came from a successful load on this thread
            // and is freed exactly once (the pointer is nulled below).
            let s = unsafe { c::tessl_embedgemma2_free(self.handle) };
            if status == TESSL_OK {
                status = s;
            }
            self.handle = ptr::null_mut();
        }
        if !self.rt.is_null() {
            // SAFETY: as above, for the runtime handle.
            let s = unsafe { tessl_runtime_free(self.rt) };
            if status == TESSL_OK {
                status = s;
            }
            self.rt = ptr::null_mut();
        }
        status
    }
}

impl Drop for Model {
    /// A worker that ends without a `free` (the gusset handle dropped by the
    /// garbage collector, say) still frees its model, on its own thread.
    fn drop(&mut self) {
        self.release();
    }
}

thread_local! {
    static MODEL: RefCell<Option<Model>> = const { RefCell::new(None) };
}

fn message(buf: &[c_char]) -> String {
    // SAFETY: tessl writes a NUL-terminated string into the buffer, and the
    // buffer is zero-filled before the call, so a terminator always exists.
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
}

fn nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// `p` for a possibly empty slice: null when empty, as a C caller passes none.
fn p<T>(s: &[T]) -> *const T {
    if s.is_empty() {
        ptr::null()
    } else {
        s.as_ptr()
    }
}

/// The model, and the time spent inside tessl on this call.
struct Call<'m> {
    m: &'m mut Model,
    work: Duration,
}

impl Call<'_> {
    /// Run one entry of the C ABI and turn a non-OK status into [`Fail`].
    fn run(&mut self, f: impl FnOnce(*mut TesslEmbedGemma2, *mut c_char, usize) -> i32) -> Result<(), Fail> {
        let mut err = [0 as c_char; ERR_LEN];
        let start = Instant::now();
        let status = f(self.m.handle, err.as_mut_ptr(), ERR_LEN);
        self.work += start.elapsed();
        if status == TESSL_OK {
            Ok(())
        } else {
            Err(Fail::Tessl {
                status,
                message: message(&err),
            })
        }
    }

    /// The `size query, then fill` pattern the string entries share: a null
    /// buffer asks for the length, a second call with a buffer of that length
    /// fetches the bytes.
    fn fetch(
        &mut self,
        f: impl Fn(*mut TesslEmbedGemma2, *mut u8, u64, *mut u64, *mut c_char, usize) -> i32,
    ) -> Result<Vec<u8>, Fail> {
        let mut needed = 0u64;
        self.run(|h, e, el| f(h, ptr::null_mut(), 0, &mut needed, e, el))?;
        let len = usize::try_from(needed).map_err(|_| Fail::Frame(format!("a string of {needed} bytes")))?;
        let mut buf = vec![0u8; len];
        let mut written = 0u64;
        self.run(|h, e, el| f(h, p_mut(&mut buf), needed, &mut written, e, el))?;
        if written != needed {
            return Err(Fail::Frame(format!(
                "a string changed length between calls: {needed} then {written}"
            )));
        }
        Ok(buf)
    }
}

fn p_mut<T>(s: &mut [T]) -> *mut T {
    if s.is_empty() {
        ptr::null_mut()
    } else {
        s.as_mut_ptr()
    }
}

/// An output of `rows` embeddings: zeroed, sized from at most [`MAX_BATCH`]
/// rows so a count read off the frame cannot size an allocation (tessl refuses
/// a larger count before it writes), and the width of each row.
fn rows(m: &Model, rows: usize, truncate: u32) -> (Vec<f32>, usize) {
    let dim = if truncate == 0 || truncate > m.dim {
        m.dim as usize
    } else {
        truncate as usize
    };
    (vec![0.0; rows.min(MAX_BATCH) * dim], dim)
}

fn embeddings(out: &[f32]) -> Reply {
    let mut r = Reply::with_capacity(8 + 4 * out.len());
    r.put_f32s(out);
    r
}

fn cut_count(offsets: &[u64]) -> usize {
    offsets.len().saturating_sub(1)
}

/// `n_items` of a table of `width`-wide rows, or the frame error.
fn table_count(what: &str, len: usize, width: usize) -> Result<usize, Fail> {
    if !len.is_multiple_of(width) {
        return Err(Fail::Frame(format!(
            "{what}: {len} values is not a whole number of {width}-wide rows"
        )));
    }
    Ok(len / width)
}

// ---------------------------------------------------------------------------
// Load and free
// ---------------------------------------------------------------------------

/// `operands u32 | multimodal u32 | snapshot blob` -> `dim u32 | max_tokens u32`.
fn load(slot: &mut Option<Model>, input: &[u8]) -> Result<Vec<u8>, String> {
    let mut r = Reader::new(input);
    let operands = r.u32("operands")?;
    let multimodal = r.u32("multimodal")?;
    let dir = r.text("snapshot_dir")?;
    r.finish("load")?;
    if slot.is_some() {
        return Ok(Reply::failure(
            STATUS_NO_MODEL,
            0,
            0,
            "this worker already hosts a model",
        ));
    }
    let start = Instant::now();
    let mut err = [0 as c_char; ERR_LEN];
    // SAFETY: `err` is ERR_LEN writable bytes.
    let rt = unsafe { tessl_runtime_new(err.as_mut_ptr(), ERR_LEN) };
    if rt.is_null() {
        return Ok(Reply::failure(STATUS_ERR, 0, nanos(start.elapsed()), &message(&err)));
    }
    let mut handle: *mut TesslEmbedGemma2 = ptr::null_mut();
    // SAFETY: `dir` is valid UTF-8 bytes alive for the call, `handle` and
    // `err` are writable; `rt` was made on this thread just above.
    let status = unsafe {
        if multimodal == 1 {
            c::tessl_embedgemma2_load_multimodal(
                rt,
                dir.as_ptr().cast(),
                dir.len() as u64,
                operands,
                &mut handle,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        } else {
            c::tessl_embedgemma2_load(
                rt,
                dir.as_ptr().cast(),
                dir.len() as u64,
                &mut handle,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        }
    };
    if status != TESSL_OK {
        let work = nanos(start.elapsed());
        // SAFETY: made above on this thread, freed once.
        unsafe { tessl_runtime_free(rt) };
        return Ok(Reply::failure(status as u32, 0, work, &message(&err)));
    }
    let mut model = Model {
        rt,
        handle,
        dim: 0,
        multimodal: multimodal == 1,
        dead: false,
    };
    let (mut dim, mut max_tokens) = (0u32, 0u32);
    // SAFETY: the handle is live on this thread; the outs are writable.
    let status = unsafe {
        let s = c::tessl_embedgemma2_embedding_dim(handle, &mut dim, err.as_mut_ptr(), ERR_LEN);
        if s == TESSL_OK {
            c::tessl_embedgemma2_max_tokens(handle, &mut max_tokens, err.as_mut_ptr(), ERR_LEN)
        } else {
            s
        }
    };
    if status != TESSL_OK {
        let work = nanos(start.elapsed());
        model.release();
        return Ok(Reply::failure(status as u32, 0, work, &message(&err)));
    }
    model.dim = dim;
    let mut reply = Reply::new();
    reply.put_u32(dim);
    reply.put_u32(max_tokens);
    *slot = Some(model);
    Ok(reply.finish(STATUS_OK, 0, nanos(start.elapsed())))
}

/// (empty) -> (empty). Frees the model and its runtime on this thread.
fn free(slot: &mut Option<Model>, input: &[u8]) -> Result<Vec<u8>, String> {
    Reader::new(input).finish("free")?;
    let start = Instant::now();
    match slot.take() {
        None => Ok(Reply::failure(STATUS_NO_MODEL, 0, 0, "this worker hosts no model")),
        Some(mut model) => {
            let status = model.release();
            let work = nanos(start.elapsed());
            if status == TESSL_OK {
                Ok(Reply::new().finish(STATUS_OK, 0, work))
            } else {
                Ok(Reply::failure(
                    status as u32,
                    0,
                    work,
                    "freeing the model failed (tessl leaks a handle freed from a thread other than its creator's)",
                ))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Model operations
// ---------------------------------------------------------------------------

/// Run `op` on `model`, and turn what happened into a reply: a result, a
/// refusal tessl made (with whether the runtime is now poisoned), or the
/// frame error for gusset's error path.
fn run_model_op(opcode: u32, model: &mut Model, input: &[u8]) -> Result<Vec<u8>, String> {
    let mut call = Call {
        m: model,
        work: Duration::ZERO,
    };
    let done = dispatch(opcode, &mut call, input);
    let work = call.work;
    match done {
        Ok(reply) => Ok(reply.finish(STATUS_OK, 0, nanos(work))),
        Err(Fail::Frame(m)) => Err(m),
        Err(Fail::Tessl { status, message }) => {
            let model = call.m;
            let mut flags = 0;
            if status == TESSL_PANIC {
                model.dead = true;
            }
            if runtime_poisoned(model) {
                model.dead = true;
                flags |= FLAG_POISONED;
            }
            Ok(Reply::failure(status as u32, flags, nanos(work), &message))
        }
    }
}

fn runtime_poisoned(m: &Model) -> bool {
    let mut out = 0i32;
    let mut err = [0 as c_char; ERR_LEN];
    // SAFETY: the runtime handle is live on this thread; the outs are writable.
    let status = unsafe { tessl_runtime_poisoned(m.rt, &mut out, err.as_mut_ptr(), ERR_LEN) };
    status == TESSL_OK && out != 0
}

fn dispatch(opcode: u32, call: &mut Call<'_>, input: &[u8]) -> Done {
    use crate::{
        OP_AUDIO_SOFT_TOKENS, OP_DECODE, OP_EMBED_AUDIO, OP_EMBED_AUDIO_SOURCES, OP_EMBED_IMAGES,
        OP_EMBED_IMAGE_SOURCES, OP_EMBED_SEGMENTS, OP_EMBED_TEXTS, OP_EMBED_VIDEOS, OP_EMBED_VIDEO_SOURCES, OP_ENCODE,
        OP_IMAGE_SOFT_TOKENS, OP_INFO, OP_PROMPTS, OP_SET_MAX_TOKENS, OP_TOKENIZE, OP_VIDEO_SOFT_TOKENS,
    };
    let mut r = Reader::new(input);
    let reply = match opcode {
        OP_INFO => info(call, &mut r)?,
        OP_SET_MAX_TOKENS => set_max_tokens(call, &mut r)?,
        OP_PROMPTS => prompts(call, &mut r)?,
        OP_TOKENIZE => tokenize(call, &mut r)?,
        OP_DECODE => decode(call, &mut r)?,
        OP_ENCODE => encode(call, &mut r)?,
        OP_EMBED_TEXTS => embed_texts(call, &mut r)?,
        OP_EMBED_IMAGES => embed_images(call, &mut r)?,
        OP_EMBED_AUDIO => embed_audio(call, &mut r)?,
        OP_EMBED_VIDEOS => embed_videos(call, &mut r)?,
        OP_EMBED_IMAGE_SOURCES => embed_sources(call, &mut r, Medium::Image)?,
        OP_EMBED_AUDIO_SOURCES => embed_sources(call, &mut r, Medium::Audio)?,
        OP_EMBED_VIDEO_SOURCES => embed_sources(call, &mut r, Medium::Video)?,
        OP_IMAGE_SOFT_TOKENS => image_soft_tokens(call, &mut r)?,
        OP_AUDIO_SOFT_TOKENS => audio_soft_tokens(call, &mut r)?,
        OP_VIDEO_SOFT_TOKENS => video_soft_tokens(call, &mut r)?,
        OP_EMBED_SEGMENTS => embed_segments(call, &mut r)?,
        other => {
            return Err(Fail::Frame(format!(
                "opcode {other} is not an operation of this engine"
            )))
        }
    };
    r.finish("request")?;
    Ok(reply)
}

/// (empty) -> `dim u32 | max_tokens u32 | multimodal u32`.
fn info(call: &mut Call<'_>, _r: &mut Reader<'_>) -> Done {
    let mut max_tokens = 0u32;
    // SAFETY: the handle is live on this thread; `max_tokens` is writable.
    call.run(|h, e, el| unsafe { c::tessl_embedgemma2_max_tokens(h, &mut max_tokens, e, el) })?;
    let mut reply = Reply::new();
    reply.put_u32(call.m.dim);
    reply.put_u32(max_tokens);
    reply.put_u32(u32::from(call.m.multimodal));
    Ok(reply)
}

/// `max_tokens u32` -> (empty).
fn set_max_tokens(call: &mut Call<'_>, r: &mut Reader<'_>) -> Done {
    let max_tokens = r.u32("max_tokens")?;
    // SAFETY: the handle is live on this thread.
    call.run(|h, e, el| unsafe { c::tessl_embedgemma2_set_max_tokens(h, max_tokens, e, el) })?;
    Ok(Reply::new())
}

/// (empty) -> `default_index i32 | count u32 | (name blob, text blob) * count`.
fn prompts(call: &mut Call<'_>, _r: &mut Reader<'_>) -> Done {
    let mut count = 0u32;
    // SAFETY: the handle is live on this thread; `count` is writable.
    call.run(|h, e, el| unsafe { c::tessl_embedgemma2_prompt_count(h, &mut count, e, el) })?;
    if count > MAX_PROMPTS {
        return Err(Fail::Frame(format!(
            "the snapshot lists {count} prompts; the limit is {MAX_PROMPTS}"
        )));
    }
    let mut default = 0i32;
    // SAFETY: as above.
    call.run(|h, e, el| unsafe { c::tessl_embedgemma2_default_prompt(h, &mut default, e, el) })?;
    let mut reply = Reply::new();
    reply.put_i32(default);
    reply.put_u32(count);
    for index in 0..count {
        // SAFETY (both fetches): the handle is live on this thread; `buf` is
        // null with length 0 or `buf_len` writable bytes, `out_len` writable.
        let name = call.fetch(|h, buf, buf_len, out_len, e, el| unsafe {
            c::tessl_embedgemma2_prompt_name(h, index, buf, buf_len, out_len, e, el)
        })?;
        let text = call.fetch(|h, buf, buf_len, out_len, e, el| unsafe {
            c::tessl_embedgemma2_prompt_get(h, name.as_ptr().cast(), name.len() as u64, buf, buf_len, out_len, e, el)
        })?;
        reply.put_blob(&name);
        reply.put_blob(&text);
    }
    Ok(reply)
}

/// `text_bytes blob | text_offsets u64s | prompt blob` ->
/// `ids u32s | id_offsets u64s`.
fn tokenize(call: &mut Call<'_>, r: &mut Reader<'_>) -> Done {
    let bytes = r.blob("text_bytes")?;
    let offsets = r.u64s("text_offsets")?;
    let prompt = r.blob("prompt")?;
    let n = cut_count(&offsets);
    // SAFETY (both calls): the pointers address `bytes`, `offsets` (n + 1
    // entries, or null when empty), `prompt`, and the outs below, all alive for
    // the call with the lengths passed.
    let mut total = 0u64;
    call.run(|h, e, el| unsafe {
        c::tessl_embedgemma2_tokenize(
            h,
            bytes.as_ptr(),
            bytes.len() as u64,
            p(&offsets),
            n as u64,
            p(prompt).cast(),
            prompt.len() as u64,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            &mut total,
            e,
            el,
        )
    })?;
    let count = usize::try_from(total).map_err(|_| Fail::Frame(format!("{total} ids")))?;
    let mut ids = vec![0u32; count];
    let mut cuts = vec![0u64; n + 1];
    let mut filled = 0u64;
    call.run(|h, e, el| unsafe {
        c::tessl_embedgemma2_tokenize(
            h,
            bytes.as_ptr(),
            bytes.len() as u64,
            p(&offsets),
            n as u64,
            p(prompt).cast(),
            prompt.len() as u64,
            p_mut(&mut ids),
            total,
            cuts.as_mut_ptr(),
            &mut filled,
            e,
            el,
        )
    })?;
    let mut reply = Reply::with_capacity(16 + 4 * ids.len() + 8 * cuts.len());
    reply.put_u32s(&ids);
    reply.put_u64s(&cuts);
    Ok(reply)
}

/// `skip_special u32 | ids u32s` -> `text blob`.
fn decode(call: &mut Call<'_>, r: &mut Reader<'_>) -> Done {
    let skip = r.u32("skip_special_tokens")?;
    let ids = r.u32s("ids")?;
    // SAFETY: `ids` is alive with the length passed; buf/out_len as in `fetch`.
    let text = call.fetch(|h, buf, buf_len, out_len, e, el| unsafe {
        c::tessl_embedgemma2_decode(h, p(&ids), ids.len() as u64, skip, buf, buf_len, out_len, e, el)
    })?;
    let mut reply = Reply::new();
    reply.put_blob(&text);
    Ok(reply)
}

/// `truncate u32 | ids u32s | offsets u64s` -> `embeddings f32s`.
fn encode(call: &mut Call<'_>, r: &mut Reader<'_>) -> Done {
    let truncate = r.u32("truncate_dim")?;
    let ids = r.u32s("ids")?;
    let offsets = r.u64s("offsets")?;
    let n = cut_count(&offsets);
    let (mut out, _) = rows(call.m, n, truncate);
    // SAFETY: `ids`, `offsets` (n + 1 entries) and `out` are alive with the
    // lengths passed.
    call.run(|h, e, el| unsafe {
        c::tessl_embedgemma2_encode(
            h,
            p(&ids),
            ids.len() as u64,
            p(&offsets),
            n as u64,
            truncate,
            p_mut(&mut out),
            out.len() as u64,
            e,
            el,
        )
    })?;
    Ok(embeddings(&out))
}

/// `truncate u32 | text_bytes blob | text_offsets u64s | prompt blob` ->
/// `embeddings f32s`.
fn embed_texts(call: &mut Call<'_>, r: &mut Reader<'_>) -> Done {
    let truncate = r.u32("truncate_dim")?;
    let bytes = r.blob("text_bytes")?;
    let offsets = r.u64s("text_offsets")?;
    let prompt = r.blob("prompt")?;
    let n = cut_count(&offsets);
    let (mut out, _) = rows(call.m, n, truncate);
    // SAFETY: the pointers address `bytes`, `offsets` (n + 1 entries), `prompt`
    // and `out`, alive with the lengths passed.
    call.run(|h, e, el| unsafe {
        c::tessl_embedgemma2_embed_texts(
            h,
            bytes.as_ptr(),
            bytes.len() as u64,
            p(&offsets),
            n as u64,
            p(prompt).cast(),
            prompt.len() as u64,
            truncate,
            p_mut(&mut out),
            out.len() as u64,
            e,
            el,
        )
    })?;
    Ok(embeddings(&out))
}

/// `truncate u32 | pixels blob | dims u32s (n x 3)` -> `embeddings f32s`.
fn embed_images(call: &mut Call<'_>, r: &mut Reader<'_>) -> Done {
    let truncate = r.u32("truncate_dim")?;
    let pixels = r.blob("pixels")?;
    let dims = r.u32s("image_dims")?;
    let n = table_count("image_dims", dims.len(), 3)?;
    let (mut out, _) = rows(call.m, n, truncate);
    // SAFETY: `pixels`, `dims` (3 * n entries) and `out` are alive with the
    // lengths passed.
    call.run(|h, e, el| unsafe {
        c::tessl_embedgemma2_embed_images(
            h,
            pixels.as_ptr(),
            pixels.len() as u64,
            p(&dims),
            n as u64,
            truncate,
            p_mut(&mut out),
            out.len() as u64,
            e,
            el,
        )
    })?;
    Ok(embeddings(&out))
}

/// `truncate u32 | rate u32 | samples f32s | offsets u64s` -> `embeddings f32s`.
fn embed_audio(call: &mut Call<'_>, r: &mut Reader<'_>) -> Done {
    let truncate = r.u32("truncate_dim")?;
    let rate = r.u32("sampling_rate")?;
    let samples = r.f32s("samples")?;
    let offsets = r.u64s("sample_offsets")?;
    let n = cut_count(&offsets);
    let (mut out, _) = rows(call.m, n, truncate);
    // SAFETY: `samples`, `offsets` (n + 1 entries) and `out` are alive with
    // the lengths passed.
    call.run(|h, e, el| unsafe {
        c::tessl_embedgemma2_embed_audio(
            h,
            p(&samples),
            samples.len() as u64,
            p(&offsets),
            n as u64,
            rate,
            truncate,
            p_mut(&mut out),
            out.len() as u64,
            e,
            el,
        )
    })?;
    Ok(embeddings(&out))
}

/// `truncate u32 | timestamps u32 | frames blob | dims u32s (n x 2) |
/// frame_offsets u64s (n + 1) | fps f64s (n)` -> `embeddings f32s`.
fn embed_videos(call: &mut Call<'_>, r: &mut Reader<'_>) -> Done {
    let truncate = r.u32("truncate_dim")?;
    let timestamps = r.u32("timestamps")?;
    let frames = r.blob("frames")?;
    let dims = r.u32s("video_dims")?;
    let offsets = r.u64s("frame_offsets")?;
    let fps = r.f64s("video_fps")?;
    let n = fps.len();
    if dims.len() != 2 * n || offsets.len() != n + 1 {
        return Err(Fail::Frame(format!(
            "{n} videos need {} dims and {} frame offsets, got {} and {}",
            2 * n,
            n + 1,
            dims.len(),
            offsets.len()
        )));
    }
    let (mut out, _) = rows(call.m, n, truncate);
    // SAFETY: `frames`, `dims` (2n), `offsets` (n + 1), `fps` (n) and `out` are
    // alive with the lengths checked above.
    call.run(|h, e, el| unsafe {
        c::tessl_embedgemma2_embed_videos(
            h,
            frames.as_ptr(),
            frames.len() as u64,
            p(&dims),
            p(&offsets),
            p(&fps),
            n as u64,
            timestamps,
            truncate,
            p_mut(&mut out),
            out.len() as u64,
            e,
            el,
        )
    })?;
    Ok(embeddings(&out))
}

#[derive(Clone, Copy)]
enum Medium {
    Image,
    Audio,
    Video,
}

/// `truncate u32 | source u32 | [video: timestamps u32 | extension blob] |
/// blob blob | offsets u64s (n + 1)` -> `embeddings f32s`.
fn embed_sources(call: &mut Call<'_>, r: &mut Reader<'_>, medium: Medium) -> Done {
    let truncate = r.u32("truncate_dim")?;
    let source = r.u32("source")?;
    let (timestamps, extension) = match medium {
        Medium::Video => (r.u32("timestamps")?, r.blob("extension")?),
        _ => (0, &[][..]),
    };
    let blob = r.blob("blob")?;
    let offsets = r.u64s("offsets")?;
    let n = cut_count(&offsets);
    let (mut out, _) = rows(call.m, n, truncate);
    // SAFETY (all three): `blob`, `offsets` (n + 1 entries), `extension` and
    // `out` are alive with the lengths passed.
    call.run(|h, e, el| unsafe {
        match medium {
            Medium::Image => c::tessl_embedgemma2_embed_image_sources(
                h,
                source,
                blob.as_ptr(),
                blob.len() as u64,
                p(&offsets),
                n as u64,
                truncate,
                p_mut(&mut out),
                out.len() as u64,
                e,
                el,
            ),
            Medium::Audio => c::tessl_embedgemma2_embed_audio_sources(
                h,
                source,
                blob.as_ptr(),
                blob.len() as u64,
                p(&offsets),
                n as u64,
                truncate,
                p_mut(&mut out),
                out.len() as u64,
                e,
                el,
            ),
            Medium::Video => c::tessl_embedgemma2_embed_video_sources(
                h,
                source,
                blob.as_ptr(),
                blob.len() as u64,
                p(&offsets),
                n as u64,
                p(extension).cast(),
                extension.len() as u64,
                timestamps,
                truncate,
                p_mut(&mut out),
                out.len() as u64,
                e,
                el,
            ),
        }
    })?;
    Ok(embeddings(&out))
}

/// `dims u32s (n x 2)` -> `counts u64s (n)`.
fn image_soft_tokens(call: &mut Call<'_>, r: &mut Reader<'_>) -> Done {
    let dims = r.u32s("image_dims")?;
    let n = table_count("image_dims", dims.len(), 2)?;
    let mut out = vec![0u64; n.min(MAX_CALL_IMAGES)];
    // SAFETY: `dims` (2n entries) and `out` are alive with the lengths passed.
    call.run(|h, e, el| unsafe {
        c::tessl_embedgemma2_image_soft_token_count(h, p(&dims), n as u64, p_mut(&mut out), n as u64, e, el)
    })?;
    let mut reply = Reply::new();
    reply.put_u64s(&out);
    Ok(reply)
}

/// `rate u32 | sample_counts u64s (n)` -> `counts u64s (n)`.
fn audio_soft_tokens(call: &mut Call<'_>, r: &mut Reader<'_>) -> Done {
    let rate = r.u32("sampling_rate")?;
    let counts = r.u64s("sample_counts")?;
    let n = counts.len();
    let mut out = vec![0u64; n.min(MAX_CALL_CLIPS)];
    // SAFETY: `counts` (n entries) and `out` are alive with the lengths passed.
    call.run(|h, e, el| unsafe {
        c::tessl_embedgemma2_audio_soft_token_count(h, p(&counts), n as u64, rate, p_mut(&mut out), n as u64, e, el)
    })?;
    let mut reply = Reply::new();
    reply.put_u64s(&out);
    Ok(reply)
}

/// `dims u32s (n x 2) | frame_counts u64s (n) | fps f64s (n)` ->
/// `pairs u64s (n x 2)`.
fn video_soft_tokens(call: &mut Call<'_>, r: &mut Reader<'_>) -> Done {
    let dims = r.u32s("video_dims")?;
    let frames = r.u64s("frame_counts")?;
    let fps = r.f64s("video_fps")?;
    let n = frames.len();
    if dims.len() != 2 * n || fps.len() != n {
        return Err(Fail::Frame(format!(
            "{n} videos need {} dims and {n} rates, got {} and {}",
            2 * n,
            dims.len(),
            fps.len()
        )));
    }
    let mut out = vec![0u64; 2 * n.min(MAX_CALL_IMAGES)];
    // SAFETY: `dims` (2n), `frames` (n), `fps` (n) and `out` are alive with the
    // lengths checked above.
    call.run(|h, e, el| unsafe {
        c::tessl_embedgemma2_video_soft_token_count(
            h,
            p(&dims),
            p(&frames),
            p(&fps),
            n as u64,
            p_mut(&mut out),
            2 * n as u64,
            e,
            el,
        )
    })?;
    let mut reply = Reply::new();
    reply.put_u64s(&out);
    Ok(reply)
}

/// The text and the three media tables of an `embed_segments` call, as the
/// frame carries them.
///
/// `truncate u32 | text_bytes blob | text_offsets u64s | kinds u32s |
/// refs u64s | item_offsets u64s |
/// image_source u32 | image_blob blob | image_offsets u64s | image_dims u32s |
/// audio_source u32 | audio_blob blob | audio_offsets u64s | rate u32 |
/// video_source u32 | video_blob blob | video_offsets u64s | video_dims u32s |
/// video_fps f64s | video_extension blob` -> `embeddings f32s`.
fn embed_segments(call: &mut Call<'_>, r: &mut Reader<'_>) -> Done {
    let truncate = r.u32("truncate_dim")?;
    let text_bytes = r.blob("text_bytes")?;
    let text_offsets = r.u64s("text_offsets")?;
    let kinds = r.u32s("segment_kinds")?;
    let refs = r.u64s("segment_refs")?;
    let item_offsets = r.u64s("item_offsets")?;
    let image_source = r.u32("image_source")?;
    let image_blob = r.blob("image_blob")?;
    let image_offsets = r.u64s("image_offsets")?;
    let image_dims = r.u32s("image_dims")?;
    let audio_source = r.u32("audio_source")?;
    let audio_blob = r.blob("audio_blob")?;
    let audio_offsets = r.u64s("audio_offsets")?;
    let rate = r.u32("sampling_rate")?;
    let video_source = r.u32("video_source")?;
    let video_blob = r.blob("video_blob")?;
    let video_offsets = r.u64s("video_offsets")?;
    let video_dims = r.u32s("video_dims")?;
    let video_fps = r.f64s("video_fps")?;
    let video_ext = r.blob("video_extension")?;

    if refs.len() != kinds.len() {
        return Err(Fail::Frame(format!(
            "{} segment kinds but {} segment references",
            kinds.len(),
            refs.len()
        )));
    }
    let n_items = cut_count(&item_offsets);
    let n_texts = cut_count(&text_offsets);
    let n_images = cut_count(&image_offsets);
    let n_clips = cut_count(&audio_offsets);
    let n_videos = cut_count(&video_offsets);
    // tessl reads `3 * n_images` dims, `2 * n_videos` dims and `n_videos` rates
    // from raw tables; the vectors must hold at least that many.
    if image_source == c::TESSL_EMBEDGEMMA2_SOURCE_RAW && image_dims.len() != 3 * n_images {
        return Err(Fail::Frame(format!(
            "{n_images} raw images need {} dims, got {}",
            3 * n_images,
            image_dims.len()
        )));
    }
    if video_source == c::TESSL_EMBEDGEMMA2_SOURCE_RAW
        && (video_dims.len() != 2 * n_videos || video_fps.len() != n_videos)
    {
        return Err(Fail::Frame(format!(
            "{n_videos} raw videos need {} dims and {n_videos} rates, got {} and {}",
            2 * n_videos,
            video_dims.len(),
            video_fps.len()
        )));
    }
    let (mut out, _) = rows(call.m, n_items, truncate);
    // SAFETY: every pointer addresses a vector or blob alive for the call with
    // the length passed (offsets: n + 1 entries, or null when empty; dims and
    // rates: the lengths checked above, or not read by a non-raw source).
    call.run(|h, e, el| unsafe {
        c::tessl_embedgemma2_embed_segments(
            h,
            text_bytes.as_ptr(),
            text_bytes.len() as u64,
            p(&text_offsets),
            n_texts as u64,
            p(&kinds),
            p(&refs),
            kinds.len() as u64,
            p(&item_offsets),
            n_items as u64,
            image_source,
            image_blob.as_ptr(),
            image_blob.len() as u64,
            p(&image_offsets),
            p(&image_dims),
            n_images as u64,
            audio_source,
            audio_blob.as_ptr(),
            audio_blob.len() as u64,
            p(&audio_offsets),
            rate,
            n_clips as u64,
            video_source,
            video_blob.as_ptr(),
            video_blob.len() as u64,
            p(&video_offsets),
            p(&video_dims),
            p(&video_fps),
            n_videos as u64,
            p(video_ext).cast(),
            video_ext.len() as u64,
            truncate,
            p_mut(&mut out),
            out.len() as u64,
            e,
            el,
        )
    })?;
    Ok(embeddings(&out))
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

/// Run opcode `opcode` on `input` against this thread's model.
///
/// `Err` is gusset's error path (an unreadable frame); everything tessl said
/// is an `Ok` reply with a status.
pub fn execute(opcode: u32, input: &[u8]) -> Result<Vec<u8>, String> {
    use crate::{OP_FREE, OP_LOAD};
    MODEL.with(|cell| {
        let mut slot = cell.borrow_mut();
        match opcode {
            OP_LOAD => load(&mut slot, input),
            OP_FREE => free(&mut slot, input),
            _ => match slot.as_mut() {
                None => Ok(Reply::failure(
                    STATUS_NO_MODEL,
                    0,
                    0,
                    "this worker hosts no model: load one first, or the worker was replaced after a panic",
                )),
                Some(m) if m.dead => Ok(Reply::failure(
                    STATUS_NO_MODEL,
                    0,
                    0,
                    "the model is unusable after a panic or a poisoned runtime; close it and load it again",
                )),
                Some(m) => run_model_op(opcode, m, input),
            },
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OP_EMBED_TEXTS, OP_FREE, OP_INFO, OP_LOAD};

    fn status(reply: &[u8]) -> u32 {
        u32::from_le_bytes([reply[0], reply[1], reply[2], reply[3]])
    }

    fn body(reply: &[u8]) -> String {
        String::from_utf8_lossy(&reply[crate::wire::REPLY_HEADER..]).into_owned()
    }

    #[test]
    fn a_model_op_with_no_model_is_a_reply_not_a_panic() {
        let reply = execute(OP_INFO, &[]).unwrap();
        assert_eq!(status(&reply), STATUS_NO_MODEL);
        assert!(body(&reply).contains("hosts no model"), "{}", body(&reply));
        let reply = execute(OP_EMBED_TEXTS, &[1, 2, 3]).unwrap();
        assert_eq!(
            status(&reply),
            STATUS_NO_MODEL,
            "a bad frame is not read before the model is known"
        );
    }

    #[test]
    fn free_with_no_model_is_refused_by_name() {
        let reply = execute(OP_FREE, &[]).unwrap();
        assert_eq!(status(&reply), STATUS_NO_MODEL);
    }

    #[test]
    fn a_truncated_load_frame_is_a_frame_error() {
        let e = execute(OP_LOAD, &[1, 0, 0]).unwrap_err();
        assert!(e.contains("truncated frame"), "{e}");
    }

    #[test]
    fn a_load_frame_with_trailing_bytes_is_refused() {
        let mut w = Reply::new();
        w.put_u32(0);
        w.put_u32(0);
        w.put_blob(b"/nonexistent");
        w.put_u64(1);
        let frame = w.finish(0, 0, 0)[crate::wire::REPLY_HEADER..].to_vec();
        let e = execute(OP_LOAD, &frame).unwrap_err();
        assert!(e.contains("unread bytes"), "{e}");
    }
}
