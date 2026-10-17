//! The frames that cross the Go/Rust boundary.
//!
//! Gusset moves bytes and an opcode; the schema is the adopter's. A request is
//! a sequence of fields, all little-endian:
//!
//! * a scalar is one 64-bit word (a `u32` is zero-extended, an `i32`
//!   sign-extended, an `f64` its bit pattern);
//! * a blob is `u64` length, that many bytes, then zero bytes up to the next
//!   multiple of 8, so every field starts 8-aligned in the frame whatever the
//!   blob lengths were.
//!
//! The typed arrays (`u32`, `u64`, `f32`, `f64`) are blobs of their elements in
//! little-endian order. They are decoded element by element into owned vectors,
//! never reinterpreted in place: the frame's base address is whatever Go (or
//! gusset's buffer registry) gave, and tessl's C ABI refuses a misaligned
//! pointer.
//!
//! A reply is a 16-byte header (`status u32`, `flags u32`, `work_ns u64`)
//! followed by the operation's payload, or by tessl's error message when the
//! status is not [`STATUS_OK`]. Failures of tessl travel in the reply so the
//! Go side never parses text to learn what happened; gusset's own error path is
//! kept for frames this crate cannot read and for cancellation.
//!
//! The reader only has to be memory-safe: every length it reads is checked
//! against what is left of the frame before it sizes anything. What the values
//! mean is checked again by tessl's C ABI, which this crate calls.

/// The operation completed; the payload is its result.
pub const STATUS_OK: u32 = 0;
/// tessl refused or failed the call (`TESSL_ERR`); the payload is its message.
pub const STATUS_ERR: u32 = 1;
/// tessl caught a panic of its own (`TESSL_PANIC`); the model is dead.
pub const STATUS_PANIC: u32 = 2;
/// The worker holds no usable model: none was loaded, one already is, or the
/// last call panicked.
pub const STATUS_NO_MODEL: u32 = 3;

/// Reply flag: the model's runtime reports itself poisoned. Every later call
/// on it is refused; the model must be closed and loaded again.
pub const FLAG_POISONED: u32 = 1;

/// Bytes of a reply header.
pub const REPLY_HEADER: usize = 16;

/// A cursor over a request frame.
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.buf.len())
            .ok_or_else(|| {
                format!(
                    "truncated frame: {what} needs {n} bytes at offset {}, {} remain",
                    self.pos,
                    self.buf.len() - self.pos
                )
            })?;
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    /// A scalar: every one is 8 bytes on the wire, so the next field starts
    /// 8-aligned.
    pub fn u64(&mut self, what: &str) -> Result<u64, String> {
        let b = self.take(8, what)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    /// A scalar that must fit 32 bits.
    pub fn u32(&mut self, what: &str) -> Result<u32, String> {
        let v = self.u64(what)?;
        u32::try_from(v).map_err(|_| format!("{what}: {v} does not fit 32 bits"))
    }

    /// A blob's bytes, borrowed from the frame.
    pub fn blob(&mut self, what: &str) -> Result<&'a [u8], String> {
        let len = self.u64(what)?;
        let len = usize::try_from(len).map_err(|_| format!("{what}: length {len} overflows usize"))?;
        let bytes = self.take(len, what)?;
        let pad = (8 - len % 8) % 8;
        self.take(pad, what)?;
        Ok(bytes)
    }

    fn array<const N: usize, T>(&mut self, what: &str, decode: impl Fn([u8; N]) -> T) -> Result<Vec<T>, String> {
        let bytes = self.blob(what)?;
        let (elements, rest) = bytes.as_chunks::<N>();
        if !rest.is_empty() {
            return Err(format!(
                "{what}: {} bytes is not a whole number of {N}-byte elements",
                bytes.len()
            ));
        }
        Ok(elements.iter().map(|&a| decode(a)).collect())
    }

    pub fn u32s(&mut self, what: &str) -> Result<Vec<u32>, String> {
        self.array::<4, u32>(what, u32::from_le_bytes)
    }

    pub fn u64s(&mut self, what: &str) -> Result<Vec<u64>, String> {
        self.array::<8, u64>(what, u64::from_le_bytes)
    }

    pub fn f32s(&mut self, what: &str) -> Result<Vec<f32>, String> {
        self.array::<4, f32>(what, f32::from_le_bytes)
    }

    pub fn f64s(&mut self, what: &str) -> Result<Vec<f64>, String> {
        self.array::<8, f64>(what, f64::from_le_bytes)
    }

    /// A UTF-8 blob.
    pub fn text(&mut self, what: &str) -> Result<&'a str, String> {
        let b = self.blob(what)?;
        std::str::from_utf8(b).map_err(|e| format!("{what} is not UTF-8 (invalid after byte {})", e.valid_up_to()))
    }

    /// The frame must be consumed exactly: a trailing field is a caller that
    /// and this crate disagree about the operation.
    pub fn finish(self, what: &str) -> Result<(), String> {
        if self.pos != self.buf.len() {
            return Err(format!(
                "{what}: {} unread bytes after the last field",
                self.buf.len() - self.pos
            ));
        }
        Ok(())
    }
}

/// A reply under construction. The payload is written after a zeroed header,
/// which [`Reply::finish`] fills in once the status and the time are known, so
/// the embedding rows are never copied to make room for it.
pub struct Reply {
    out: Vec<u8>,
}

impl Reply {
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// A reply with a payload sized up front, for the embedding rows.
    pub fn with_capacity(payload: usize) -> Self {
        let mut out = Vec::with_capacity(REPLY_HEADER + payload);
        out.resize(REPLY_HEADER, 0);
        Self { out }
    }

    /// The frame, with its header written.
    pub fn finish(mut self, status: u32, flags: u32, work_ns: u64) -> Vec<u8> {
        self.out[0..4].copy_from_slice(&status.to_le_bytes());
        self.out[4..8].copy_from_slice(&flags.to_le_bytes());
        self.out[8..16].copy_from_slice(&work_ns.to_le_bytes());
        self.out
    }

    /// A reply whose payload is tessl's (or this crate's) message.
    pub fn failure(status: u32, flags: u32, work_ns: u64, message: &str) -> Vec<u8> {
        let mut r = Self::with_capacity(message.len());
        r.out.extend_from_slice(message.as_bytes());
        r.finish(status, flags, work_ns)
    }

    pub fn put_u32(&mut self, v: u32) {
        self.put_u64(u64::from(v));
    }

    /// A signed scalar, sign-extended to 64 bits.
    pub fn put_i32(&mut self, v: i32) {
        self.put_u64(i64::from(v) as u64);
    }

    pub fn put_u64(&mut self, v: u64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    pub fn put_blob(&mut self, bytes: &[u8]) {
        self.put_u64(bytes.len() as u64);
        self.out.extend_from_slice(bytes);
        self.out.resize(self.out.len() + (8 - bytes.len() % 8) % 8, 0);
    }

    pub fn put_u32s(&mut self, v: &[u32]) {
        self.put_u64(4 * v.len() as u64);
        for x in v {
            self.out.extend_from_slice(&x.to_le_bytes());
        }
        self.out.resize(self.out.len() + (8 - (4 * v.len()) % 8) % 8, 0);
    }

    pub fn put_u64s(&mut self, v: &[u64]) {
        self.put_u64(8 * v.len() as u64);
        for x in v {
            self.out.extend_from_slice(&x.to_le_bytes());
        }
    }

    pub fn put_f32s(&mut self, v: &[f32]) {
        self.put_u64(4 * v.len() as u64);
        for x in v {
            self.out.extend_from_slice(&x.to_le_bytes());
        }
        self.out.resize(self.out.len() + (8 - (4 * v.len()) % 8) % 8, 0);
    }
}

impl Default for Reply {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The body of a frame holding `parts` as blobs, written by the reply
    /// writer (the request and reply encodings are one).
    fn frame(parts: &[&[u8]]) -> Vec<u8> {
        let mut r = Reply::new();
        for p in parts {
            r.put_blob(p);
        }
        r.finish(0, 0, 0)[REPLY_HEADER..].to_vec()
    }

    #[test]
    fn blobs_round_trip_and_pad_to_eight() {
        let f = frame(&[b"abc", b"", b"12345678", b"x"]);
        assert_eq!(f.len() % 8, 0);
        let mut r = Reader::new(&f);
        assert_eq!(r.blob("a").unwrap(), b"abc");
        assert_eq!(r.blob("b").unwrap(), b"");
        assert_eq!(r.blob("c").unwrap(), b"12345678");
        assert_eq!(r.blob("d").unwrap(), b"x");
        r.finish("test").unwrap();
    }

    #[test]
    fn scalars_are_one_word_each_and_a_u32_must_fit() {
        let mut w = Reply::new();
        w.put_u32(7);
        w.put_u64(9);
        w.put_i32(-3);
        w.put_u64(1 << 33);
        let bytes = w.finish(0, 0, 0);
        assert_eq!(bytes.len(), REPLY_HEADER + 32);
        let mut r = Reader::new(&bytes[REPLY_HEADER..]);
        assert_eq!(r.u32("a").unwrap(), 7);
        assert_eq!(r.u64("b").unwrap(), 9);
        assert_eq!(r.u64("c").unwrap() as i64, -3);
        assert!(r.u32("d").unwrap_err().contains("does not fit 32 bits"));
        r.finish("test").unwrap();
    }

    #[test]
    fn a_length_past_the_frame_is_refused_before_it_sizes_anything() {
        let mut huge = u64::MAX.to_le_bytes().to_vec();
        huge.extend_from_slice(&[0; 8]);
        let e = Reader::new(&huge).blob("field").unwrap_err();
        assert!(
            e.contains("field") && (e.contains("truncated") || e.contains("overflows")),
            "{e}"
        );
        // A length that fits usize but not the frame.
        let mut big = (1u64 << 40).to_le_bytes().to_vec();
        big.extend_from_slice(&[0; 16]);
        let e = Reader::new(&big).blob("field").unwrap_err();
        assert!(e.contains("truncated frame"), "{e}");
    }

    #[test]
    fn a_truncated_scalar_and_a_missing_pad_are_refused() {
        assert!(Reader::new(&[1, 2, 3]).u32("n").unwrap_err().contains("truncated"));
        assert!(Reader::new(&[1; 7]).u64("n").unwrap_err().contains("truncated"));
        // Length 3, three bytes, but the five pad bytes are missing.
        let mut f = 3u64.to_le_bytes().to_vec();
        f.extend_from_slice(b"abc");
        assert!(Reader::new(&f).blob("b").unwrap_err().contains("truncated"));
    }

    #[test]
    fn typed_arrays_decode_little_endian_and_refuse_a_ragged_length() {
        let mut w = Reply::new();
        w.put_u32s(&[1, 2, 3]);
        w.put_u64s(&[u64::MAX, 5]);
        w.put_f32s(&[1.5, -2.0]);
        let bytes = w.finish(0, 0, 0);
        let mut r = Reader::new(&bytes[REPLY_HEADER..]);
        assert_eq!(r.u32s("a").unwrap(), vec![1, 2, 3]);
        assert_eq!(r.u64s("b").unwrap(), vec![u64::MAX, 5]);
        assert_eq!(r.f32s("c").unwrap(), vec![1.5, -2.0]);
        r.finish("test").unwrap();

        let ragged = frame(&[b"12345"]);
        assert!(Reader::new(&ragged).u32s("ids").unwrap_err().contains("whole number"));
        assert!(Reader::new(&ragged).f64s("fps").unwrap_err().contains("whole number"));
    }

    #[test]
    fn text_must_be_utf8_and_trailing_bytes_are_refused() {
        let f = frame(&[&[0xff, 0xfe]]);
        assert!(Reader::new(&f).text("path").unwrap_err().contains("not UTF-8"));
        let f = frame(&[b"ok", b"extra"]);
        let mut r = Reader::new(&f);
        assert_eq!(r.text("a").unwrap(), "ok");
        assert!(r.finish("op").unwrap_err().contains("unread bytes"));
    }

    #[test]
    fn the_reply_header_is_sixteen_bytes_and_a_failure_carries_its_message() {
        let bytes = Reply::new().finish(STATUS_ERR, FLAG_POISONED, 99);
        assert_eq!(bytes.len(), REPLY_HEADER);
        assert_eq!(&bytes[0..4], &1u32.to_le_bytes());
        assert_eq!(&bytes[4..8], &1u32.to_le_bytes());
        assert_eq!(&bytes[8..16], &99u64.to_le_bytes());
        let bytes = Reply::failure(STATUS_PANIC, 0, 5, "boom");
        assert_eq!(&bytes[0..4], &2u32.to_le_bytes());
        assert_eq!(&bytes[REPLY_HEADER..], b"boom");
    }

    const GOLDEN_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../embedgemma2/internal/wire/testdata");

    /// The request the Go side wrote (wire_test.go, `goldenRequest`) decodes here
    /// field by field. The file is the Go package's; this test is the other half
    /// of the check that the two agree on the format.
    #[test]
    fn golden_request_decodes_as_the_go_side_wrote_it() {
        let bytes = std::fs::read(format!("{GOLDEN_DIR}/golden_request.bin")).expect("golden_request.bin");
        let mut r = Reader::new(&bytes);
        assert_eq!(r.u32("scalar").unwrap(), 7);
        assert_eq!(r.text("strings").unwrap(), "h\u{e9}llow\u{f6}rld");
        assert_eq!(r.u64s("offsets").unwrap(), vec![0, 6, 12]);
        assert_eq!(r.f32s("f32s").unwrap(), vec![1.5, -2.0, 0.25]);
        let f64s = r.f64s("f64s").unwrap();
        assert_eq!(f64s[0], 3.5);
        assert_eq!(f64s[1], f64::NEG_INFINITY);
        assert_eq!(r.blob("bytes").unwrap(), &[1, 2, 3, 4]);
        assert_eq!(r.text("empty").unwrap(), "");
        assert_eq!(r.u32s("u32s").unwrap(), vec![1, 2, 3]);
        assert_eq!(r.u64("max").unwrap(), u64::MAX);
        r.finish("golden").unwrap();
    }

    /// The reply this side writes is byte for byte the file the Go side parses
    /// (wire_test.go, `TestUnitGoldenReplyFromRustParses`). Set
    /// `TESSL_WIRE_UPDATE=1` to rewrite the file after a deliberate format change
    /// (and bump `wire.Version`).
    #[test]
    fn golden_reply_is_the_file_the_go_side_parses() {
        let mut w = Reply::new();
        w.put_u32(42);
        w.put_i32(-1);
        w.put_u64(1 << 40);
        w.put_blob(b"hello");
        w.put_u32s(&[9, 10, 11]);
        w.put_u64s(&[5, 1 << 33]);
        w.put_f32s(&[0.5, -1.25, 3.0]);
        let bytes = w.finish(STATUS_OK, FLAG_POISONED, 123_456_789);
        let path = format!("{GOLDEN_DIR}/golden_reply.bin");
        if std::env::var_os("TESSL_WIRE_UPDATE").is_some() {
            std::fs::write(&path, &bytes).expect("rewrite golden_reply.bin");
        }
        let golden = std::fs::read(&path).expect("golden_reply.bin");
        assert_eq!(bytes, golden, "the reply format changed: see this test's doc comment");
    }
}
