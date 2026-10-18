//! Audio decoding with AudioToolbox (`ExtAudioFile`), then the reference's
//! channel mix and sample-rate conversion in [`super::resample`].
//!
//! WAV, AIFF, CAF, MP3, M4A/AAC, FLAC, Ogg Opus and whatever else the system
//! decodes become mono `f32` PCM at the sampling rate the preprocessing wants
//! (16 kHz for EmbeddingGemma 2, [`crate::embedgemma2_pre::preprocess_audio`]).
//!
//! # What the reference does, and what this does
//!
//! `sentence-transformers` hands an audio path to the processor's
//! `fetch_audio`, which calls `transformers.audio_utils.load_audio(path,
//! sampling_rate=16000)` (`feature_extraction_sequence_utils.py:385`). Its
//! `backend="auto"` uses torchcodec's `AudioDecoder(source, sample_rate=16000,
//! num_channels=1)` when torchcodec is installed (`audio_utils.py:230-233,
//! 267`) and `librosa.load(path, sr=16000)` otherwise (`:279`); the
//! `backend="torchaudio"` branch (`:269-276`) is opt-in. torchcodec resamples
//! and mixes with a default `SwrContext` (libswresample), created in
//! `FFMPEGCommon.cpp` by `swr_alloc_set_opts2` with no options set. This module
//! reproduces that, bit for bit on the machines FFmpeg's AArch64 code runs on:
//!
//! * **Decode** at the file's own rate and channel count, to packed `f32`
//!   (`ExtAudioFile` with a client format equal to the file's rate). Integer
//!   PCM becomes float as `v / 2^(bits-1)`, as soundfile and FFmpeg do, so a
//!   PCM16, PCM24, PCM32, PCM8 or float file already at the target rate and
//!   mono decodes exactly.
//! * **Channels and resampling**: [`Resampler::Swresample`] with
//!   [`Downmix::Swresample`] (the defaults) is [`super::resample::SwrConverter`]:
//!   each channel resampled, then mixed with libswresample's matrix, which is
//!   the order `swr_init` picks for a downmix. [`Downmix::Mean`] (what librosa
//!   and the torchaudio backend do) mixes first and resamples the one channel.
//! * **[`Resampler::AudioToolbox`]** keeps the system `AudioConverter`
//!   (`kAudioConverterQuality_Max`, `Mastering` complexity) as a second stage,
//!   for comparison. Its filter is not FFmpeg's, soxr's or torchaudio's;
//!   `docs/embedgemma2.md` records how far apart they are, measured. It is run
//!   as a second stage, not as the `ExtAudioFile` client rate, because a codec
//!   followed by the file's own rate conversion came out 0.26 samples late for
//!   AAC and Ogg Opus.
//! * **Gapless**: AAC (the edit list or `iTunSMPB`), MPEG-1 MP3 and the other
//!   codecs are trimmed by `ExtAudioFile`; MPEG-2/2.5 MP3 carries its encoder
//!   delay in a tag it does not read, so `media::mp3` reads it and the
//!   delay and padding are trimmed here.
//! * **Cut-short files** are errors: `media::containers` compares the
//!   sizes WAV, AIFF, CAF, MP4/M4A, FLAC and tagged MP3 files declare with the
//!   bytes present, because AudioToolbox returns the part it has as if it were
//!   the whole.

use std::ffi::c_void;
use std::path::Path;

use super::cf;
use super::containers::{self, FileSource, Source};
use super::resample::{self, SwrConverter, SwrMix};
use super::{os_status_error, MediaError};

// ---------------------------------------------------------------------------
// FFI: AudioToolbox (ExtendedAudioFile.h, AudioFile.h, AudioConverter.h) and
// the structs of CoreAudioTypes (CoreAudioBaseTypes.h), macOS SDK.
// ---------------------------------------------------------------------------

/// `AudioStreamBasicDescription` (`CoreAudioBaseTypes.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Asbd {
    sample_rate: f64,
    format_id: u32,
    format_flags: u32,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    bytes_per_frame: u32,
    channels_per_frame: u32,
    bits_per_channel: u32,
    reserved: u32,
}

/// `AudioBuffer`.
#[repr(C)]
struct AudioBuffer {
    number_channels: u32,
    data_byte_size: u32,
    data: *mut c_void,
}

/// `AudioBufferList` with one buffer (interleaved client format).
#[repr(C)]
struct AudioBufferList1 {
    number_buffers: u32,
    buffers: [AudioBuffer; 1],
}

type ExtAudioFileRef = *mut c_void;
type AudioFileId = *mut c_void;
type AudioConverterRef = *mut c_void;
type OsStatus = i32;

/// `kAudioFormatLinearPCM` ('lpcm'), `kAudioFormatMPEGLayer3` ('.mp3') and
/// `kAudioFormatFLAC` ('flac').
const FORMAT_LINEAR_PCM: u32 = u32::from_be_bytes(*b"lpcm");
const FORMAT_MPEG_LAYER3: u32 = u32::from_be_bytes(*b".mp3");
const FORMAT_FLAC: u32 = u32::from_be_bytes(*b"flac");
/// `kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked`.
const FLAGS_FLOAT_PACKED: u32 = 0x1 | 0x8;
/// `kExtAudioFileProperty_*` (`ExtendedAudioFile.h`).
const PROP_FILE_DATA_FORMAT: u32 = u32::from_be_bytes(*b"ffmt");
const PROP_CLIENT_DATA_FORMAT: u32 = u32::from_be_bytes(*b"cfmt");
const PROP_FILE_LENGTH_FRAMES: u32 = u32::from_be_bytes(*b"#frm");
const PROP_PACKET_TABLE: u32 = u32::from_be_bytes(*b"xpti");
/// `kAudioConverterSampleRateConverterComplexity` ('srca') and `...Quality`
/// ('srcq') (`AudioConverter.h`).
const CONV_SRC_COMPLEXITY: u32 = u32::from_be_bytes(*b"srca");
const CONV_SRC_QUALITY: u32 = u32::from_be_bytes(*b"srcq");
/// `kAudioConverterSampleRateConverterComplexity_Mastering` ('bats').
const SRC_COMPLEXITY_MASTERING: u32 = u32::from_be_bytes(*b"bats");
/// `kAudioConverterQuality_Max`.
const QUALITY_MAX: u32 = 0x7F;

/// `AudioFilePacketTableInfo` (`AudioFile.h`): the valid frame count and the
/// priming and remainder frames a decoder trims.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct PacketTableInfo {
    number_valid_frames: i64,
    priming_frames: i32,
    remainder_frames: i32,
}

type ReadProc = unsafe extern "C" fn(*mut c_void, i64, u32, *mut c_void, *mut u32) -> OsStatus;
type GetSizeProc = unsafe extern "C" fn(*mut c_void) -> i64;
/// `AudioConverterComplexInputDataProc`.
type ConverterInputProc =
    unsafe extern "C" fn(AudioConverterRef, *mut u32, *mut AudioBufferList1, *mut *mut c_void, *mut c_void) -> OsStatus;

#[link(name = "AudioToolbox", kind = "framework")]
extern "C" {
    fn ExtAudioFileOpenURL(url: cf::CfTypeRef, out: *mut ExtAudioFileRef) -> OsStatus;
    fn ExtAudioFileWrapAudioFileID(file: AudioFileId, for_writing: u8, out: *mut ExtAudioFileRef) -> OsStatus;
    fn ExtAudioFileDispose(ext: ExtAudioFileRef) -> OsStatus;
    fn ExtAudioFileRead(ext: ExtAudioFileRef, frames: *mut u32, data: *mut AudioBufferList1) -> OsStatus;
    fn ExtAudioFileGetProperty(ext: ExtAudioFileRef, id: u32, size: *mut u32, out: *mut c_void) -> OsStatus;
    fn ExtAudioFileSetProperty(ext: ExtAudioFileRef, id: u32, size: u32, data: *const c_void) -> OsStatus;
    fn AudioFileOpenWithCallbacks(
        client: *mut c_void,
        read: ReadProc,
        write: Option<unsafe extern "C" fn()>,
        get_size: GetSizeProc,
        set_size: Option<unsafe extern "C" fn()>,
        type_hint: u32,
        out: *mut AudioFileId,
    ) -> OsStatus;
    fn AudioFileClose(file: AudioFileId) -> OsStatus;
    fn AudioConverterNew(src: *const Asbd, dst: *const Asbd, out: *mut AudioConverterRef) -> OsStatus;
    fn AudioConverterDispose(conv: AudioConverterRef) -> OsStatus;
    fn AudioConverterSetProperty(conv: AudioConverterRef, id: u32, size: u32, data: *const c_void) -> OsStatus;
    fn AudioConverterFillComplexBuffer(
        conv: AudioConverterRef,
        input: ConverterInputProc,
        user_data: *mut c_void,
        io_packets: *mut u32,
        out: *mut AudioBufferList1,
        packet_descriptions: *mut c_void,
    ) -> OsStatus;
}

// ---------------------------------------------------------------------------
// Options and results
// ---------------------------------------------------------------------------

/// A container to try first when decoding from bytes (an `AudioFileTypeID`
/// hint, `AudioFile.h`). [`AudioContainer::Auto`] lets AudioToolbox sniff.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioContainer {
    Auto,
    Wave,
    Aiff,
    Caf,
    Mp3,
    M4a,
    Flac,
    AacAdts,
}

impl AudioContainer {
    fn type_id(self) -> u32 {
        let code: &[u8; 4] = match self {
            AudioContainer::Auto => return 0,
            AudioContainer::Wave => b"WAVE",
            AudioContainer::Aiff => b"AIFF",
            AudioContainer::Caf => b"caff",
            AudioContainer::Mp3 => b"MPG3",
            AudioContainer::M4a => b"m4af",
            AudioContainer::Flac => b"flac",
            AudioContainer::AacAdts => b"adts",
        };
        u32::from_be_bytes(*code)
    }
}

/// How several channels become one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Downmix {
    /// The mean of the channels, taken before resampling: `librosa.load`,
    /// soundfile-based loaders and the `torchaudio` backend of
    /// `transformers.load_audio`.
    Mean,
    /// libswresample's default mix to mono, which torchcodec's
    /// `AudioDecoder(num_channels=1)` uses ([`super::resample::SwrMix`]): each
    /// channel of a stereo pair times `1/sqrt(2)` (no normalisation for float
    /// output), so a stereo clip is `sqrt(2)` louder than under
    /// [`Downmix::Mean`] (measured: FFmpeg's output is 7.7 dB below the
    /// mean's for uncorrelated channels). Defined for FFmpeg's default
    /// layouts of 1 to 8 channels; more are refused.
    Swresample,
}

/// What converts the file's sampling rate to the requested one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resampler {
    /// libswresample's default resampler, reproduced in Rust
    /// ([`super::resample`]): torchcodec's.
    Swresample,
    /// The system `AudioConverter` at its best setting, as a second stage.
    AudioToolbox,
}

/// Options for [`decode_audio_file`] and [`decode_audio_bytes`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioOptions {
    /// The output sampling rate in Hz.
    pub sample_rate: u32,
    /// The longest clip to decode, in seconds of output.
    pub max_seconds: f64,
    /// What a clip longer than `max_seconds` does: `false` returns
    /// [`MediaError::Limit`], `true` returns the first `max_seconds` of it.
    pub truncate: bool,
    /// How multi-channel audio is mixed down.
    pub downmix: Downmix,
    /// What changes the sampling rate.
    pub resampler: Resampler,
}

impl Default for AudioOptions {
    /// 16 kHz (EmbeddingGemma 2's front end), a limit of 30 minutes, and
    /// torchcodec's mix and resampler (libswresample's).
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            max_seconds: 1800.0,
            truncate: false,
            downmix: Downmix::Swresample,
            resampler: Resampler::Swresample,
        }
    }
}

/// A decoded clip: mono `f32` samples at [`AudioOptions::sample_rate`].
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// The file's own sampling rate.
    pub source_rate: f64,
    /// The file's channel count (the mix of these is `samples`).
    pub source_channels: u32,
    /// The file's length in its own frames as `ExtAudioFile` reports it, before
    /// any MP3 delay and padding are trimmed here.
    pub source_frames: Option<u64>,
    /// `true` when the clip was cut at `max_seconds` ([`AudioOptions::truncate`]).
    pub truncated: bool,
}

/// The most channels a file may have.
pub const MAX_AUDIO_CHANNELS: u32 = 64;

/// The largest audio file [`decode_audio_bytes`] is given, and
/// [`decode_audio_file`] opens.
pub const MAX_AUDIO_FILE_BYTES: u64 = 1 << 32;

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Decode an audio file to mono `f32` at `options.sample_rate`.
///
/// # Errors
/// [`MediaError::Io`] when the file cannot be read, [`MediaError::Empty`] for
/// an empty file, [`MediaError::Unrecognized`] when AudioToolbox does not know
/// the format, [`MediaError::Decode`] for damaged, cut-short or sample-less
/// data, [`MediaError::Limit`] for a clip longer than `options.max_seconds`
/// (unless it truncates).
pub fn decode_audio_file(path: &Path, options: &AudioOptions) -> Result<DecodedAudio, MediaError> {
    validate_options(options)?;
    let meta = std::fs::metadata(path).map_err(|source| MediaError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if !meta.is_file() {
        return Err(MediaError::Invalid {
            what: "audio",
            detail: format!("{} is not a file", path.display()),
        });
    }
    if meta.len() == 0 {
        return Err(MediaError::Empty { what: "audio" });
    }
    if meta.len() > MAX_AUDIO_FILE_BYTES {
        return Err(MediaError::Limit {
            what: "audio",
            detail: format!("{} bytes exceeds {MAX_AUDIO_FILE_BYTES}", meta.len()),
        });
    }
    let source = FileSource {
        file: std::fs::File::open(path).map_err(|source| MediaError::Io {
            path: path.to_path_buf(),
            source,
        })?,
        len: meta.len(),
    };
    containers::check_audio(&source).map_err(|detail| MediaError::Decode { what: "audio", detail })?;
    let url = cf::file_url(path).ok_or_else(|| MediaError::Framework {
        api: "CFURLCreateFromFileSystemRepresentation",
        detail: format!("no URL for {}", path.display()),
    })?;
    let mut ext: ExtAudioFileRef = std::ptr::null_mut();
    // SAFETY: `url` is a live CFURL and `ext` a valid out pointer.
    let status = unsafe { ExtAudioFileOpenURL(url.as_ptr(), &mut ext) };
    if status != 0 || ext.is_null() {
        return Err(os_status_error("audio", "ExtAudioFileOpenURL", status));
    }
    let file = ExtFile { ext, audio_file: None };
    read_all(&file, options, &source)
}

/// Decode encoded audio bytes (a whole WAV, AIFF, CAF, MP3, M4A, FLAC, ...
/// file) to mono `f32` at `options.sample_rate`. `container` is a hint for
/// formats AudioToolbox cannot recognize from the bytes alone
/// ([`AudioContainer::Auto`] usually suffices).
///
/// # Errors
/// As [`decode_audio_file`].
pub fn decode_audio_bytes(
    bytes: &[u8],
    container: AudioContainer,
    options: &AudioOptions,
) -> Result<DecodedAudio, MediaError> {
    validate_options(options)?;
    if bytes.is_empty() {
        return Err(MediaError::Empty { what: "audio" });
    }
    if bytes.len() as u64 > MAX_AUDIO_FILE_BYTES {
        return Err(MediaError::Limit {
            what: "audio",
            detail: format!("{} bytes exceeds {MAX_AUDIO_FILE_BYTES}", bytes.len()),
        });
    }
    let source: &[u8] = bytes;
    containers::check_audio(&source).map_err(|detail| MediaError::Decode { what: "audio", detail })?;
    let mem = MemSource { bytes };
    let mut file_id: AudioFileId = std::ptr::null_mut();
    // SAFETY: `mem` outlives the AudioFile (it is closed by `ExtFile::drop`
    // before `mem` goes out of scope, below); the callbacks only read through
    // the pointer; the write and set-size callbacks are NULL, so the file is
    // read-only.
    let status = unsafe {
        AudioFileOpenWithCallbacks(
            (&mem as *const MemSource).cast_mut().cast(),
            mem_read,
            None,
            mem_size,
            None,
            container.type_id(),
            &mut file_id,
        )
    };
    if status != 0 || file_id.is_null() {
        return Err(os_status_error("audio", "AudioFileOpenWithCallbacks", status));
    }
    let mut ext: ExtAudioFileRef = std::ptr::null_mut();
    // SAFETY: `file_id` is an open AudioFile; `ext` a valid out pointer.
    let status = unsafe { ExtAudioFileWrapAudioFileID(file_id, 0, &mut ext) };
    if status != 0 || ext.is_null() {
        // SAFETY: `file_id` is open and not used again.
        unsafe { AudioFileClose(file_id) };
        return Err(os_status_error("audio", "ExtAudioFileWrapAudioFileID", status));
    }
    let file = ExtFile {
        ext,
        audio_file: Some(file_id),
    };
    let out = read_all(&file, options, &source);
    drop(file); // closes the wrapped AudioFile while `mem` is still alive
    let _ = &mem; // `mem` must outlive the AudioFile: keep it to here
    out
}

fn validate_options(options: &AudioOptions) -> Result<(), MediaError> {
    if options.sample_rate == 0 || options.sample_rate > 384_000 {
        return Err(MediaError::Invalid {
            what: "audio",
            detail: format!("sample_rate {} Hz is not usable", options.sample_rate),
        });
    }
    if !(options.max_seconds.is_finite() && options.max_seconds > 0.0) {
        return Err(MediaError::Invalid {
            what: "audio",
            detail: format!("max_seconds {} must be positive", options.max_seconds),
        });
    }
    Ok(())
}

/// The bytes an AudioFile reads through its callbacks.
struct MemSource<'a> {
    bytes: &'a [u8],
}

/// `AudioFile_ReadProc`: copy up to `request` bytes at `position`.
///
/// # Safety
/// `client` must point at a live [`MemSource`]; `buffer` at `request` writable
/// bytes; `actual` at a `u32`. AudioToolbox guarantees these for a callback
/// registered with that client pointer.
unsafe extern "C" fn mem_read(
    client: *mut c_void,
    position: i64,
    request: u32,
    buffer: *mut c_void,
    actual: *mut u32,
) -> OsStatus {
    let src = &*(client as *const MemSource);
    let len = src.bytes.len() as u64;
    let pos = u64::try_from(position).unwrap_or(u64::MAX);
    let n = if pos >= len {
        0
    } else {
        (len - pos).min(u64::from(request)) as usize
    };
    if n > 0 {
        std::ptr::copy_nonoverlapping(src.bytes.as_ptr().add(pos as usize), buffer.cast::<u8>(), n);
    }
    *actual = n as u32;
    0
}

/// `AudioFile_GetSizeProc`.
///
/// # Safety
/// `client` must point at a live [`MemSource`].
unsafe extern "C" fn mem_size(client: *mut c_void) -> i64 {
    let src = &*(client as *const MemSource);
    src.bytes.len() as i64
}

/// An open `ExtAudioFileRef`, disposed (and its wrapped `AudioFileID`, if any,
/// closed) on drop.
struct ExtFile {
    ext: ExtAudioFileRef,
    audio_file: Option<AudioFileId>,
}

impl Drop for ExtFile {
    fn drop(&mut self) {
        // SAFETY: `ext` is the live reference this struct owns; the wrapped
        // AudioFile must be closed after the ExtAudioFile (the header says the
        // client closes it), and neither is used again.
        unsafe {
            ExtAudioFileDispose(self.ext);
            if let Some(id) = self.audio_file {
                AudioFileClose(id);
            }
        }
    }
}

/// An `AudioConverter`, disposed on drop.
struct Converter(AudioConverterRef);

impl Drop for Converter {
    fn drop(&mut self) {
        // SAFETY: the converter was made by `AudioConverterNew` and is released
        // exactly once.
        unsafe { AudioConverterDispose(self.0) };
    }
}

fn get_prop<T: Default + Copy>(file: &ExtFile, id: u32, api: &'static str) -> Result<T, MediaError> {
    let mut value = T::default();
    let mut size = std::mem::size_of::<T>() as u32;
    // SAFETY: `value` is a `T` of `size` bytes, a plain-data struct or integer
    // the property fills.
    let status = unsafe { ExtAudioFileGetProperty(file.ext, id, &mut size, (&mut value as *mut T).cast()) };
    if status != 0 {
        return Err(os_status_error("audio", api, status));
    }
    Ok(value)
}

/// The encoder delay and padding, in source frames, that the decode of this MP3
/// still contains, if the file's tag gives them.
///
/// MPEG-2 and 2.5 Layer III files (16, 22.05, 24 kHz ...) carry their gapless
/// numbers in a tag AudioToolbox does not read (and refuses to take as a
/// `kExtAudioFileProperty_PacketTable` override), so its decode keeps both. When
/// `ExtAudioFile` reports no priming or remainder for the file, the tag's are
/// returned for the caller to trim.
fn mp3_untrimmed_gapless(file: &ExtFile, source: &dyn Source) -> Result<Option<(u32, u32)>, MediaError> {
    if let Ok(info) = get_prop::<PacketTableInfo>(file, PROP_PACKET_TABLE, "ExtAudioFileGetProperty(PacketTable)") {
        if info.priming_frames != 0 || info.remainder_frames != 0 {
            return Ok(None); // AudioToolbox already trims this file
        }
    }
    Ok(containers::mp3_prefix(source).and_then(|p| super::mp3::gapless(&p)))
}

/// Source-rate frames of a file, read in chunks and trimmed.
struct Feed<'a> {
    file: &'a ExtFile,
    channels: usize,
    downmix: Downmix,
    /// The libswresample mix, when `downmix` is [`Downmix::Swresample`].
    swr_mix: Option<SwrMix>,
    /// Interleaved frames as read.
    raw: Vec<f32>,
    /// The latest chunk's range of `raw`, in frames.
    chunk: std::ops::Range<usize>,
    /// The latest chunk mixed to mono, once [`Feed::mix_chunk`] has run.
    mono: Vec<f32>,
    /// Source frames still to drop from the front.
    skip: usize,
    /// Source frames still to deliver.
    keep: usize,
    /// The first `ExtAudioFileRead` error, if any.
    error: OsStatus,
    done: bool,
}

impl Feed<'_> {
    /// The latest chunk, interleaved.
    fn chunk_frames(&self) -> &[f32] {
        &self.raw[self.chunk.start * self.channels..self.chunk.end * self.channels]
    }

    /// Mix the latest chunk to mono into `self.mono`.
    fn mix_chunk(&mut self) {
        let ch = self.channels;
        let frames = &self.raw[self.chunk.start * ch..self.chunk.end * ch];
        self.mono.clear();
        match (&self.swr_mix, self.downmix) {
            (Some(mix), Downmix::Swresample) => mix.mix_interleaved(frames, &mut self.mono),
            _ => append_mean(&mut self.mono, frames, ch),
        }
    }

    /// Read the next chunk of at most `max` frames into `self.raw`
    /// ([`Feed::chunk_frames`]). `false` at the end of the data (or the first
    /// error, left in `self.error`).
    fn next_chunk(&mut self, max: usize) -> bool {
        const CHUNK_FRAMES: usize = 16_384;
        while !self.done && self.keep > 0 {
            let want = max.clamp(1, CHUNK_FRAMES);
            let mut frames = want as u32;
            let mut list = AudioBufferList1 {
                number_buffers: 1,
                buffers: [AudioBuffer {
                    number_channels: self.channels as u32,
                    data_byte_size: (want * self.channels * 4) as u32,
                    data: self.raw.as_mut_ptr().cast(),
                }],
            };
            // SAFETY: `list` describes one interleaved buffer of at least
            // `want * channels` f32 (`raw` is allocated for CHUNK_FRAMES
            // frames), live for the call; `frames` is in/out.
            let status = unsafe { ExtAudioFileRead(self.file.ext, &mut frames, &mut list) };
            if status != 0 {
                self.error = status;
                self.done = true;
                return false;
            }
            let frames = (frames as usize).min(want);
            if frames == 0 {
                self.done = true;
                return false;
            }
            let dropped = self.skip.min(frames);
            self.skip -= dropped;
            let take = (frames - dropped).min(self.keep);
            if take == 0 {
                continue;
            }
            self.keep -= take;
            self.chunk = dropped..dropped + take;
            return true;
        }
        false
    }
}

/// `AudioConverterComplexInputDataProc`: hand the converter the next mono chunk.
///
/// # Safety
/// `user` must point at a live [`Feed`]; `packets` and `data` are the
/// converter's own, valid for the call.
unsafe extern "C" fn feed_input(
    _conv: AudioConverterRef,
    packets: *mut u32,
    data: *mut AudioBufferList1,
    _descriptions: *mut *mut c_void,
    user: *mut c_void,
) -> OsStatus {
    let feed = &mut *(user as *mut Feed<'_>);
    let list = &mut *data;
    if feed.next_chunk((*packets) as usize) {
        feed.mix_chunk();
        list.number_buffers = 1;
        list.buffers[0] = AudioBuffer {
            number_channels: 1,
            data_byte_size: (feed.mono.len() * 4) as u32,
            data: feed.mono.as_mut_ptr().cast(),
        };
        *packets = feed.mono.len() as u32;
        0
    } else {
        // End of data, or an error already recorded in `feed.error`: zero
        // packets, as the header requires.
        list.number_buffers = 1;
        list.buffers[0].data_byte_size = 0;
        *packets = 0;
        0
    }
}

fn read_all(file: &ExtFile, options: &AudioOptions, source: &dyn Source) -> Result<DecodedAudio, MediaError> {
    const OUT_FRAMES: usize = 16_384;
    let src: Asbd = get_prop(file, PROP_FILE_DATA_FORMAT, "ExtAudioFileGetProperty(FileDataFormat)")?;
    let channels = src.channels_per_frame;
    if !(src.sample_rate.is_finite() && src.sample_rate > 0.0) || channels == 0 {
        return Err(MediaError::Decode {
            what: "audio",
            detail: format!("the file reports {} Hz and {channels} channels", src.sample_rate),
        });
    }
    if channels > MAX_AUDIO_CHANNELS {
        return Err(MediaError::Limit {
            what: "audio",
            detail: format!("{channels} channels exceeds {MAX_AUDIO_CHANNELS}"),
        });
    }
    // The libswresample mix is modelled for FFmpeg's default layouts only.
    let swr_mix = match options.downmix {
        Downmix::Swresample => Some(SwrMix::new(channels as usize)?),
        Downmix::Mean => None,
    };
    let source_frames = get_prop::<i64>(
        file,
        PROP_FILE_LENGTH_FRAMES,
        "ExtAudioFileGetProperty(FileLengthFrames)",
    )
    .ok()
    .and_then(|n| u64::try_from(n).ok());

    // The client format: packed native-endian f32 at the file's own rate and
    // channel count. Rate conversion is the second stage, below.
    let client = Asbd {
        sample_rate: src.sample_rate,
        format_id: FORMAT_LINEAR_PCM,
        format_flags: FLAGS_FLOAT_PACKED,
        bytes_per_packet: 4 * channels,
        frames_per_packet: 1,
        bytes_per_frame: 4 * channels,
        channels_per_frame: channels,
        bits_per_channel: 32,
        reserved: 0,
    };
    // SAFETY: `client` is an ASBD and the size passed is its size.
    let status = unsafe {
        ExtAudioFileSetProperty(
            file.ext,
            PROP_CLIENT_DATA_FORMAT,
            std::mem::size_of::<Asbd>() as u32,
            (&client as *const Asbd).cast(),
        )
    };
    if status != 0 {
        return Err(os_status_error(
            "audio",
            "ExtAudioFileSetProperty(ClientDataFormat)",
            status,
        ));
    }

    // Frames to drop from the front and to deliver, in source frames, when
    // AudioToolbox does not apply the file's own gapless numbers.
    let (mut skip, mut keep) = (0usize, usize::MAX);
    if src.format_id == FORMAT_MPEG_LAYER3 {
        if let (Some((delay, padding)), Some(total)) = (mp3_untrimmed_gapless(file, source)?, source_frames) {
            let valid = total
                .checked_sub(u64::from(delay) + u64::from(padding))
                .filter(|&v| v > 0)
                .ok_or_else(|| MediaError::Decode {
                    what: "audio",
                    detail: format!("the MP3 tag trims {delay} + {padding} frames of {total}"),
                })?;
            skip = delay as usize;
            keep = usize::try_from(valid).unwrap_or(usize::MAX);
        }
    }

    let ratio = f64::from(options.sample_rate) / src.sample_rate;
    let max_samples = (options.max_seconds * f64::from(options.sample_rate)).ceil() as usize;
    let expected = source_frames.map(|n| (n as f64 * ratio).ceil() as usize).unwrap_or(0);
    let mut mono: Vec<f32> = Vec::with_capacity(expected.min(max_samples).min(1 << 28));
    let mut feed = Feed {
        file,
        channels: channels as usize,
        downmix: options.downmix,
        swr_mix,
        raw: vec![0f32; 16_384 * channels as usize],
        chunk: 0..0,
        mono: Vec::new(),
        skip,
        keep,
        error: 0,
        done: false,
    };
    let mut truncated = false;
    let same_rate = (src.sample_rate - f64::from(options.sample_rate)).abs() <= f64::EPSILON;
    if same_rate || options.resampler == Resampler::Swresample {
        // The mix and the rate change are libswresample's: per-channel
        // resampling then the matrix (`Downmix::Swresample`), or the mean
        // first and one channel resampled (`Downmix::Mean`). Equal rates only
        // mix.
        let in_rate = if same_rate {
            options.sample_rate
        } else if src.sample_rate.fract() == 0.0 && src.sample_rate <= f64::from(resample::MAX_RATE) {
            src.sample_rate as u32
        } else {
            return Err(MediaError::Unsupported {
                what: "audio",
                detail: format!(
                    "a source rate of {} Hz is not a whole number of Hz within {} Hz; Resampler::AudioToolbox takes it",
                    src.sample_rate,
                    resample::MAX_RATE
                ),
            });
        };
        let mean_first = options.downmix == Downmix::Mean;
        let mut conv = SwrConverter::new(
            if mean_first { 1 } else { channels as usize },
            in_rate,
            options.sample_rate,
        )?;
        let mut produced: Vec<f32> = Vec::new();
        'read: while feed.next_chunk(usize::MAX) {
            produced.clear();
            if mean_first {
                feed.mix_chunk();
                conv.push(&feed.mono, &mut produced);
            } else {
                conv.push(feed.chunk_frames(), &mut produced);
            }
            if append_capped(&mut mono, &produced, max_samples) {
                truncated = true;
                break 'read;
            }
        }
        if !truncated && feed.error == 0 {
            produced.clear();
            conv.finish(&mut produced);
            truncated = append_capped(&mut mono, &produced, max_samples);
        }
    } else {
        let converter = new_resampler(src.sample_rate, options.sample_rate)?;
        let mut out = vec![0f32; OUT_FRAMES];
        loop {
            let mut packets = OUT_FRAMES as u32;
            let mut list = AudioBufferList1 {
                number_buffers: 1,
                buffers: [AudioBuffer {
                    number_channels: 1,
                    data_byte_size: (OUT_FRAMES * 4) as u32,
                    data: out.as_mut_ptr().cast(),
                }],
            };
            // SAFETY: the converter is live; `feed_input` is the documented
            // callback and `feed` outlives the call; `list` describes `out`,
            // which has room for `packets` mono f32 frames.
            let status = unsafe {
                AudioConverterFillComplexBuffer(
                    converter.0,
                    feed_input,
                    (&mut feed as *mut Feed<'_>).cast(),
                    &mut packets,
                    &mut list,
                    std::ptr::null_mut(),
                )
            };
            if feed.error != 0 {
                return Err(os_status_error("audio", "ExtAudioFileRead", feed.error));
            }
            let produced = (packets as usize).min(OUT_FRAMES);
            if status != 0 && produced == 0 {
                return Err(os_status_error("audio", "AudioConverterFillComplexBuffer", status));
            }
            if produced == 0 {
                break;
            }
            let room = max_samples - mono.len().min(max_samples);
            if produced > room {
                mono.extend_from_slice(&out[..room]);
                truncated = true;
                break;
            }
            mono.extend_from_slice(&out[..produced]);
        }
    }
    if feed.error != 0 {
        return Err(os_status_error("audio", "ExtAudioFileRead", feed.error));
    }
    if src.format_id == FORMAT_FLAC && !truncated {
        // FLAC declares its sample count; a cut file decodes the frames it has.
        if let Some(total) = containers::flac_total_samples(source) {
            let expected = (total as f64 * ratio).round() as usize;
            if mono.len() + 2 < expected {
                return Err(MediaError::Decode {
                    what: "audio",
                    detail: format!(
                        "FLAC declares {total} samples ({expected} at {} Hz) but only {} were decoded",
                        options.sample_rate,
                        mono.len()
                    ),
                });
            }
        }
    }
    if truncated && !options.truncate {
        return Err(MediaError::Limit {
            what: "audio",
            detail: format!("the clip is longer than {} s", options.max_seconds),
        });
    }
    if mono.is_empty() {
        return Err(MediaError::Decode {
            what: "audio",
            detail: "the file decoded to no samples".into(),
        });
    }
    Ok(DecodedAudio {
        samples: mono,
        sample_rate: options.sample_rate,
        source_rate: src.sample_rate,
        source_channels: channels,
        source_frames,
        truncated,
    })
}

/// Append one mono sample per interleaved frame of `frames`: the mean of the
/// channels, summed in order in `f32` then divided.
fn append_mean(out: &mut Vec<f32>, frames: &[f32], channels: usize) {
    if channels == 1 {
        out.extend_from_slice(frames);
        return;
    }
    let n = channels as f32;
    out.extend(frames.chunks_exact(channels).map(|f| f.iter().sum::<f32>() / n));
}

/// Append `chunk` to `out` without letting `out` pass `max` samples. `true`
/// when the chunk had more than fit (the extra is dropped).
fn append_capped(out: &mut Vec<f32>, chunk: &[f32], max: usize) -> bool {
    let room = max.saturating_sub(out.len());
    if chunk.len() > room {
        out.extend_from_slice(&chunk[..room]);
        true
    } else {
        out.extend_from_slice(chunk);
        false
    }
}

/// A mono `f32` sample-rate converter from `from` Hz to `to` Hz at the best
/// setting: `kAudioConverterQuality_Max`, `Mastering` complexity.
fn new_resampler(from: f64, to: u32) -> Result<Converter, MediaError> {
    let format = |rate: f64| Asbd {
        sample_rate: rate,
        format_id: FORMAT_LINEAR_PCM,
        format_flags: FLAGS_FLOAT_PACKED,
        bytes_per_packet: 4,
        frames_per_packet: 1,
        bytes_per_frame: 4,
        channels_per_frame: 1,
        bits_per_channel: 32,
        reserved: 0,
    };
    let (src, dst) = (format(from), format(f64::from(to)));
    let mut conv: AudioConverterRef = std::ptr::null_mut();
    // SAFETY: both ASBDs are valid for the call; `conv` is a valid out pointer.
    let status = unsafe { AudioConverterNew(&src, &dst, &mut conv) };
    if status != 0 || conv.is_null() {
        return Err(os_status_error("audio", "AudioConverterNew", status));
    }
    let converter = Converter(conv);
    // SAFETY: `conv` is live; the values are OSType/UInt32 as the header documents.
    let (a, b) = unsafe {
        (
            AudioConverterSetProperty(
                conv,
                CONV_SRC_COMPLEXITY,
                4,
                (&SRC_COMPLEXITY_MASTERING as *const u32).cast(),
            ),
            AudioConverterSetProperty(conv, CONV_SRC_QUALITY, 4, (&QUALITY_MAX as *const u32).cast()),
        )
    };
    if a != 0 {
        return Err(os_status_error(
            "audio",
            "AudioConverterSetProperty(SampleRateConverterComplexity)",
            a,
        ));
    }
    if b != 0 {
        return Err(os_status_error(
            "audio",
            "AudioConverterSetProperty(SampleRateConverterQuality)",
            b,
        ));
    }
    Ok(converter)
}
