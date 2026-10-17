//! Gusset umbrella for tessl's EmbeddingGemma 2: the one Rust staticlib a Go
//! binary links (R14).
//!
//! cgo links `-lgusset`, so this crate's archive is `libgusset.a` (`[lib] name`,
//! not the package name). It holds gusset, tessl (its Metal shaders are
//! embedded in the rlib), and the operations of [`ops`], reached through
//! gusset's opcode dispatch. A second Rust staticlib in the same Go binary
//! would define `rust_eh_personality` twice.
//!
//! The Go package `go/embedgemma2` is the only caller. It owns the wire format
//! ([`wire`]) from its side and the opcode numbers below.

use gusset_core::{register_engine, set_engine_handler, Counting, JobContext};

pub mod ops;
pub mod wire;

/// Counts every Rust allocation in the host process.
///
/// Gusset declares no global allocator, so without this `gusset.Stats()` reads
/// zero and `AdviseMemoryLimit` subtracts nothing: the Go side of a binary that
/// links this archive would budget as if Rust used no memory. The umbrella is
/// the one archive per binary, so this is the one place it can be declared.
#[global_allocator]
static ALLOCATOR: Counting<std::alloc::System> = Counting::new(std::alloc::System);

/// Version of the request and reply frames. The Go package refuses an archive
/// whose version differs, so a stale `libgusset.a` is a named failure at
/// start-up rather than a misread frame.
pub const WIRE_VERSION: u32 = 1;

/// Load a snapshot on this worker: runtime, then model.
pub const OP_LOAD: u32 = 1;
/// Free the model and its runtime on the worker that made them.
pub const OP_FREE: u32 = 2;
/// Embedding width, token limit, and whether the towers are loaded.
pub const OP_INFO: u32 = 3;
/// Set the per-sequence token limit.
pub const OP_SET_MAX_TOKENS: u32 = 4;
/// The snapshot's named prompts and its default.
pub const OP_PROMPTS: u32 = 5;
/// Texts to token ids.
pub const OP_TOKENIZE: u32 = 6;
/// Token ids to text.
pub const OP_DECODE: u32 = 7;
/// Token ids to embeddings.
pub const OP_ENCODE: u32 = 8;
/// Texts to embeddings.
pub const OP_EMBED_TEXTS: u32 = 9;
/// Raw images to embeddings.
pub const OP_EMBED_IMAGES: u32 = 10;
/// Raw audio clips to embeddings.
pub const OP_EMBED_AUDIO: u32 = 11;
/// Raw videos to embeddings.
pub const OP_EMBED_VIDEOS: u32 = 12;
/// Image files or encoded images to embeddings.
pub const OP_EMBED_IMAGE_SOURCES: u32 = 13;
/// Audio files or encoded clips to embeddings.
pub const OP_EMBED_AUDIO_SOURCES: u32 = 14;
/// Video files or encoded videos to embeddings.
pub const OP_EMBED_VIDEO_SOURCES: u32 = 15;
/// Soft tokens an image of a given size makes.
pub const OP_IMAGE_SOFT_TOKENS: u32 = 16;
/// Soft tokens an audio clip of a given length makes.
pub const OP_AUDIO_SOFT_TOKENS: u32 = 17;
/// Frames kept and soft tokens a frame for a video of a given size.
pub const OP_VIDEO_SOFT_TOKENS: u32 = 18;
/// Items that interleave text with media, to embeddings.
pub const OP_EMBED_SEGMENTS: u32 = 19;
/// Does nothing and says so: the round trip with no work in it, which the
/// overhead benchmark measures.
pub const OP_PING: u32 = 20;
/// Panics on purpose, so the host can prove the panic firewall (I2) against
/// this archive. Reachable only through the call header Go sets; no payload
/// byte selects it.
pub const OP_SELF_TEST_PANIC: u32 = 0x7fff_0001;

/// Every operation that reaches tessl, registered the same way.
const MODEL_OPS: &[u32] = &[
    OP_LOAD,
    OP_FREE,
    OP_INFO,
    OP_SET_MAX_TOKENS,
    OP_PROMPTS,
    OP_TOKENIZE,
    OP_DECODE,
    OP_ENCODE,
    OP_EMBED_TEXTS,
    OP_EMBED_IMAGES,
    OP_EMBED_AUDIO,
    OP_EMBED_VIDEOS,
    OP_EMBED_IMAGE_SOURCES,
    OP_EMBED_AUDIO_SOURCES,
    OP_EMBED_VIDEO_SOURCES,
    OP_IMAGE_SOFT_TOKENS,
    OP_AUDIO_SOFT_TOKENS,
    OP_VIDEO_SOFT_TOKENS,
    OP_EMBED_SEGMENTS,
];

/// The cancellation check, before any work: a job cancelled or past its
/// deadline while it queued never reaches tessl. A running tessl call cannot be
/// interrupted (one synchronous GPU wait), so this is the only check.
fn cancelled(ctx: &JobContext) -> Result<(), String> {
    ctx.check().map_err(String::from)
}

/// Registers the operations on gusset's opcode dispatch.
///
/// Call once, before the first submission. Registration is not a gusset
/// export; the umbrella owns it. Opcode 0, gusset's global handler, is
/// installed to refuse, so a submission that forgot its opcode fails instead of
/// running anything, and the diagnostic engine stays unreachable.
///
/// Returns 0 when every opcode registered, otherwise the number that did not;
/// the Go package refuses to open a handle on a non-zero return. The Go process
/// that linked this archive calls it once, before `gusset_submit`.
#[no_mangle]
pub extern "C" fn tessl_go_engine_init() -> i32 {
    set_engine_handler(|_ctx: &JobContext, _input: &[u8]| -> Result<Vec<u8>, String> {
        Err("tessl go engine: a submission must name an opcode (gusset opcode 0 is not an operation)".to_string())
    });
    let mut failed = 0;
    for &op in MODEL_OPS {
        let registered = register_engine(op, move |ctx: &JobContext, input: &[u8]| -> Result<Vec<u8>, String> {
            cancelled(ctx)?;
            ops::execute(op, input)
        });
        failed += i32::from(registered.is_err());
    }
    let ping = register_engine(OP_PING, |ctx: &JobContext, input: &[u8]| -> Result<Vec<u8>, String> {
        cancelled(ctx)?;
        wire::Reader::new(input).finish("ping")?;
        Ok(wire::Reply::new().finish(wire::STATUS_OK, 0, 0))
    });
    let panic = register_engine(
        OP_SELF_TEST_PANIC,
        |_ctx: &JobContext, _input: &[u8]| -> Result<Vec<u8>, String> {
            panic!("tessl go engine self-test panic");
        },
    );
    failed + i32::from(ping.is_err()) + i32::from(panic.is_err())
}

/// The frame version this archive speaks ([`WIRE_VERSION`]).
#[no_mangle]
pub extern "C" fn tessl_go_engine_wire_version() -> u32 {
    WIRE_VERSION
}
