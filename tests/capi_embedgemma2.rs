//! The C ABI's EmbeddingGemma 2 entries (`tessl::capi_embedgemma2`), called the
//! way a foreign caller calls them.
//!
//! The model is the tiny random checkpoint `tests/embedgemma2_tiny.rs` holds to
//! an f64 forward, written to a directory under `target/` so `tessl_embedgemma2_load`
//! reads it the way it reads a Hugging Face snapshot. What this file pins is the
//! boundary: every refusal (null pointers, offsets, lengths, `truncate_dim`, a
//! bad snapshot directory) is a status and a message that leaves the caller's
//! output untouched and the handle usable, and a successful call is
//! `EmbedGemma2Model::encode` bit for bit on the same weights. The real
//! checkpoint is `python/tests/test_embedgemma2.py`.
//!
//! The tests with a null handle need no device (they are in the hosted
//! selection, `scripts/hosted_tests.txt`); the rest create a runtime.

mod common;

use std::ffi::{c_char, CStr};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::OnceLock;

use common::embedgemma2::{random_weights, safetensors_bytes, CONFIG};
use common::gpu_lock;
use tessl::capi::{tessl_runtime_free, tessl_runtime_new, TesslRuntime, TESSL_ERR, TESSL_OK};
use tessl::capi_embedgemma2::{
    tessl_embedgemma2_embedding_dim, tessl_embedgemma2_encode, tessl_embedgemma2_free, tessl_embedgemma2_load,
    tessl_embedgemma2_max_tokens, tessl_embedgemma2_set_max_tokens, test_embedgemma2_model, TesslEmbedGemma2,
    TESSL_EMBEDGEMMA2_MAX_PATH_BYTES,
};
use tessl::embedgemma2::{EmbedGemma2Config, MAX_BATCH, MAX_BATCH_TOKENS};

const ERR_LEN: usize = 512;
/// The tiny checkpoint's `embedding_dim` and vocabulary.
const DIM: usize = 96;
const VOCAB: u32 = 48;
/// Written to the output of every call that must leave it alone.
const SENTINEL: f32 = -7.25;

fn msg(err: &[c_char]) -> String {
    unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned()
}

/// One encode call with raw arguments; the status and the library's message.
#[allow(clippy::too_many_arguments)]
fn encode_raw(
    model: *mut TesslEmbedGemma2,
    ids: *const u32,
    n_ids: u64,
    offsets: *const u64,
    n_seqs: u64,
    truncate_dim: u32,
    out: *mut f32,
    out_len: u64,
) -> (i32, String) {
    let mut err = [0 as c_char; ERR_LEN];
    let s = unsafe {
        tessl_embedgemma2_encode(
            model,
            ids,
            n_ids,
            offsets,
            n_seqs,
            truncate_dim,
            out,
            out_len,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    (s, msg(&err))
}

/// CSR form of `seqs`: the flat ids and the `n + 1` offsets.
fn csr(seqs: &[Vec<u32>]) -> (Vec<u32>, Vec<u64>) {
    let mut offsets = vec![0u64];
    let mut ids = Vec::new();
    for s in seqs {
        ids.extend_from_slice(s);
        offsets.push(ids.len() as u64);
    }
    (ids, offsets)
}

/// `seqs` through the C entry, well formed; the output and the status.
fn encode_seqs(model: *mut TesslEmbedGemma2, seqs: &[Vec<u32>], truncate_dim: u32) -> (i32, String, Vec<f32>) {
    let (ids, offsets) = csr(seqs);
    let dim = if truncate_dim == 0 { DIM } else { truncate_dim as usize };
    let mut out = vec![SENTINEL; seqs.len() * dim];
    let (s, m) = encode_raw(
        model,
        ids.as_ptr(),
        ids.len() as u64,
        offsets.as_ptr(),
        seqs.len() as u64,
        truncate_dim,
        out.as_mut_ptr(),
        out.len() as u64,
    );
    (s, m, out)
}

// ---------------------------------------------------------------------------
// No device: a null handle, a null free.
// ---------------------------------------------------------------------------

#[test]
fn a_null_model_handle_is_refused_by_every_entry() {
    let null: *mut TesslEmbedGemma2 = ptr::null_mut();
    let mut err = [0 as c_char; ERR_LEN];
    let mut out_u32 = 0u32;
    let ids = [1u32, 2];
    let offsets = [0u64, 2];
    let mut out = [SENTINEL; DIM];
    let mut results = vec![
        (
            "dim",
            unsafe { tessl_embedgemma2_embedding_dim(null, &mut out_u32, err.as_mut_ptr(), ERR_LEN) },
            msg(&err),
        ),
        (
            "max_tokens",
            unsafe { tessl_embedgemma2_max_tokens(null, &mut out_u32, err.as_mut_ptr(), ERR_LEN) },
            msg(&err),
        ),
        (
            "set_max_tokens",
            unsafe { tessl_embedgemma2_set_max_tokens(null, 16, err.as_mut_ptr(), ERR_LEN) },
            msg(&err),
        ),
    ];
    let (s, m) = encode_raw(
        null,
        ids.as_ptr(),
        2,
        offsets.as_ptr(),
        1,
        0,
        out.as_mut_ptr(),
        DIM as u64,
    );
    results.push(("encode", s, m));
    for (name, status, message) in results {
        assert_eq!(status, TESSL_ERR, "{name}");
        assert!(message.contains("null embedgemma2 model handle"), "{name}: {message}");
    }
    assert_eq!(out_u32, 0, "a refused query wrote its output");
    assert!(out.iter().all(|&v| v == SENTINEL), "a refused encode wrote its output");
}

#[test]
fn freeing_a_null_handle_is_a_no_op() {
    assert_eq!(unsafe { tessl_embedgemma2_free(ptr::null_mut()) }, TESSL_OK);
}

#[test]
fn loading_on_a_null_runtime_is_refused_and_leaves_out_null() {
    let dir = b"/nonexistent";
    let mut err = [0 as c_char; ERR_LEN];
    // A stale non-null value must not survive a refusal.
    let mut out = 0x10 as *mut TesslEmbedGemma2;
    let s = unsafe {
        tessl_embedgemma2_load(
            ptr::null_mut(),
            dir.as_ptr().cast(),
            dir.len() as u64,
            &mut out,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    assert_eq!(s, TESSL_ERR);
    assert!(out.is_null(), "*out is non-null after a refused load");
    assert!(msg(&err).contains("null runtime handle"), "{}", msg(&err));
}

// ---------------------------------------------------------------------------
// The tiny checkpoint on disk.
// ---------------------------------------------------------------------------

/// A snapshot directory holding the tiny checkpoint, written once per process.
fn snapshot() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/capi_embedgemma2_tiny");
        std::fs::create_dir_all(&dir).expect("create the snapshot directory");
        let cfg = EmbedGemma2Config::from_config_json(CONFIG).expect("tiny config");
        let w = random_weights(&cfg, 2026);
        std::fs::write(dir.join("config.json"), CONFIG).expect("write config.json");
        std::fs::write(dir.join("model.safetensors"), safetensors_bytes(&w, &[])).expect("write model.safetensors");
        dir
    })
}

/// Load `dir` on `rt`; the status, the message and `*out`.
fn load_raw(rt: *mut TesslRuntime, dir: &[u8]) -> (i32, String, *mut TesslEmbedGemma2) {
    let mut err = [0 as c_char; ERR_LEN];
    let mut out = 0x10 as *mut TesslEmbedGemma2;
    let s = unsafe {
        tessl_embedgemma2_load(
            rt,
            dir.as_ptr().cast(),
            dir.len() as u64,
            &mut out,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    (s, msg(&err), out)
}

/// A runtime handle and the model loaded on it, both freed on drop.
struct Loaded {
    rt: *mut TesslRuntime,
    model: *mut TesslEmbedGemma2,
}

impl Loaded {
    fn new() -> Self {
        let mut err = [0 as c_char; ERR_LEN];
        let rt = unsafe { tessl_runtime_new(err.as_mut_ptr(), ERR_LEN) };
        assert!(!rt.is_null(), "tessl_runtime_new: {}", msg(&err));
        let (s, m, model) = load_raw(rt, snapshot().to_str().unwrap().as_bytes());
        assert_eq!(s, TESSL_OK, "tessl_embedgemma2_load: {m}");
        assert!(!model.is_null());
        Self { rt, model }
    }

    /// Free the runtime handle now: the model must not depend on it.
    fn drop_runtime(&mut self) {
        assert_eq!(unsafe { tessl_runtime_free(self.rt) }, TESSL_OK);
        self.rt = ptr::null_mut();
    }
}

impl Drop for Loaded {
    fn drop(&mut self) {
        assert_eq!(unsafe { tessl_embedgemma2_free(self.model) }, TESSL_OK);
        assert_eq!(unsafe { tessl_runtime_free(self.rt) }, TESSL_OK);
    }
}

fn sequences() -> Vec<Vec<u32>> {
    // Ragged, a length-1 sequence, a repeat, and ids across the vocabulary.
    let seq = |n: usize, salt: u32| (0..n as u32).map(|i| (i * 7 + salt) % VOCAB).collect::<Vec<u32>>();
    vec![seq(5, 1), seq(9, 4), seq(1, 3), seq(7, 11), seq(5, 1), seq(23, 2)]
}

#[test]
fn the_c_entry_is_encode_bit_for_bit() {
    let _g = gpu_lock();
    let mut l = Loaded::new();
    let model = unsafe { test_embedgemma2_model(l.model) };
    let seqs = sequences();
    let batch: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
    // The model outlives the runtime handle it was loaded on.
    l.drop_runtime();
    for trunc in [0u32, 64, 96, 1] {
        let (s, m, got) = encode_seqs(l.model, &seqs, trunc);
        assert_eq!(s, TESSL_OK, "truncate_dim {trunc}: {m}");
        let want = model.encode(&batch, (trunc != 0).then_some(trunc), false).unwrap();
        assert_eq!(got.len(), want.embeddings.len(), "truncate_dim {trunc}");
        let differs = got
            .iter()
            .zip(&want.embeddings)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        assert_eq!(differs, None, "truncate_dim {trunc}: first differing element");
        assert!(got.iter().all(|v| v.is_finite()));
    }
    // One sequence alone, and the whole batch in one call's order.
    let (s, _, one) = encode_seqs(l.model, &seqs[3..4], 0);
    assert_eq!(s, TESSL_OK);
    let want = model.encode(&batch[3..4], None, false).unwrap();
    assert_eq!(one, want.embeddings);
    // The unit-norm contract holds through the boundary.
    let (_, _, all) = encode_seqs(l.model, &seqs, 0);
    for row in all.chunks(DIM) {
        let norm = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm {norm}");
    }
}

#[test]
fn queries_and_the_token_limit_round_trip() {
    let _g = gpu_lock();
    let l = Loaded::new();
    let mut err = [0 as c_char; ERR_LEN];
    let (mut dim, mut max) = (0u32, 0u32);
    assert_eq!(
        unsafe { tessl_embedgemma2_embedding_dim(l.model, &mut dim, err.as_mut_ptr(), ERR_LEN) },
        TESSL_OK
    );
    assert_eq!(dim as usize, DIM);
    assert_eq!(
        unsafe { tessl_embedgemma2_max_tokens(l.model, &mut max, err.as_mut_ptr(), ERR_LEN) },
        TESSL_OK
    );
    assert_eq!(max, tessl::embedgemma2::DEFAULT_MAX_TOKENS);
    // A null out is a refusal, not a write through null.
    for f in [tessl_embedgemma2_embedding_dim, tessl_embedgemma2_max_tokens] {
        let s = unsafe { f(l.model, ptr::null_mut(), err.as_mut_ptr(), ERR_LEN) };
        assert_eq!(s, TESSL_ERR);
        assert!(msg(&err).contains("null out"), "{}", msg(&err));
    }

    // Lower the limit: a 5-token sequence passes at 5 and is refused at 4.
    let seqs = vec![vec![1u32, 2, 3, 4, 5]];
    assert_eq!(
        unsafe { tessl_embedgemma2_set_max_tokens(l.model, 5, err.as_mut_ptr(), ERR_LEN) },
        TESSL_OK
    );
    assert_eq!(encode_seqs(l.model, &seqs, 0).0, TESSL_OK);
    assert_eq!(
        unsafe { tessl_embedgemma2_set_max_tokens(l.model, 4, err.as_mut_ptr(), ERR_LEN) },
        TESSL_OK
    );
    let (s, m, out) = encode_seqs(l.model, &seqs, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("limit is 4"), "{m}");
    assert!(out.iter().all(|&v| v == SENTINEL));
    // Out-of-range limits are refused and leave the limit as it was.
    for bad in [0u32, MAX_BATCH_TOKENS as u32 + 1, u32::MAX] {
        let s = unsafe { tessl_embedgemma2_set_max_tokens(l.model, bad, err.as_mut_ptr(), ERR_LEN) };
        assert_eq!(s, TESSL_ERR, "{bad}");
        assert!(msg(&err).contains("max_tokens"), "{}", msg(&err));
    }
    unsafe { tessl_embedgemma2_max_tokens(l.model, &mut max, err.as_mut_ptr(), ERR_LEN) };
    assert_eq!(max, 4);
}

#[test]
fn encode_refuses_bad_arguments_and_stays_usable() {
    let _g = gpu_lock();
    let l = Loaded::new();
    let seqs = vec![vec![1u32, 2, 3], vec![4, 5]];
    let (ids, offsets) = csr(&seqs);
    let (n_ids, n_seqs) = (ids.len() as u64, seqs.len() as u64);
    let mut out = vec![SENTINEL; 2 * DIM];
    let out_len = out.len() as u64;

    // (case, offsets, n_ids, n_seqs, truncate_dim, out_len, expected message)
    #[allow(clippy::type_complexity)]
    let cases: Vec<(&str, Vec<u64>, u64, u64, u32, u64, &str)> = vec![
        (
            "offsets[0] not 0",
            vec![1, 3, 5],
            n_ids,
            n_seqs,
            0,
            out_len,
            "offsets[0]",
        ),
        (
            "offsets decrease",
            vec![0, 3, 2],
            n_ids,
            n_seqs,
            0,
            out_len,
            "does not exceed",
        ),
        (
            "an empty sequence",
            vec![0, 3, 3],
            n_ids,
            n_seqs,
            0,
            out_len,
            "does not exceed",
        ),
        (
            "offsets end short",
            vec![0, 2, 4],
            n_ids,
            n_seqs,
            0,
            out_len,
            "n_ids is 5",
        ),
        (
            "offsets end past n_ids",
            vec![0, 3, 6],
            n_ids,
            n_seqs,
            0,
            out_len,
            "n_ids is 5",
        ),
        (
            "n_ids larger than offsets",
            offsets.clone(),
            n_ids + 1,
            n_seqs,
            0,
            out_len,
            "n_ids is 6",
        ),
        (
            "n_ids u64::MAX",
            offsets.clone(),
            u64::MAX,
            n_seqs,
            0,
            out_len,
            "n_ids is",
        ),
        (
            "a sequence past the limit",
            vec![0, 3, 3 + 9000],
            n_ids,
            n_seqs,
            0,
            out_len,
            "limit is 8192",
        ),
        (
            "a huge gap",
            vec![0, 3, u64::MAX / 2],
            n_ids,
            n_seqs,
            0,
            out_len,
            "limit is 8192",
        ),
        (
            "offsets at u64::MAX",
            vec![0, u64::MAX, u64::MAX],
            n_ids,
            n_seqs,
            0,
            out_len,
            "limit is 8192",
        ),
        ("n_seqs zero", offsets.clone(), n_ids, 0, 0, 0, "n_seqs is 0"),
        (
            "n_seqs past MAX_BATCH",
            offsets.clone(),
            n_ids,
            MAX_BATCH as u64 + 1,
            0,
            out_len,
            "n_seqs is",
        ),
        (
            "n_seqs u64::MAX",
            offsets.clone(),
            n_ids,
            u64::MAX,
            0,
            out_len,
            "n_seqs",
        ),
        (
            "n_seqs fewer than offsets",
            offsets.clone(),
            n_ids,
            1,
            0,
            DIM as u64,
            "n_ids is 5",
        ),
        (
            "truncate_dim past the width",
            offsets.clone(),
            n_ids,
            n_seqs,
            DIM as u32 + 1,
            out_len,
            "truncate_dim",
        ),
        (
            "truncate_dim u32::MAX",
            offsets.clone(),
            n_ids,
            n_seqs,
            u32::MAX,
            out_len,
            "truncate_dim",
        ),
        (
            "out_len short by one",
            offsets.clone(),
            n_ids,
            n_seqs,
            0,
            out_len - 1,
            "out_len",
        ),
        (
            "out_len long by one",
            offsets.clone(),
            n_ids,
            n_seqs,
            0,
            out_len + 1,
            "out_len",
        ),
        ("out_len zero", offsets.clone(), n_ids, n_seqs, 0, 0, "out_len"),
        (
            "out_len u64::MAX",
            offsets.clone(),
            n_ids,
            n_seqs,
            0,
            u64::MAX,
            "out_len",
        ),
        (
            "out_len for the full width under a prefix",
            offsets.clone(),
            n_ids,
            n_seqs,
            64,
            out_len,
            "out_len",
        ),
    ];
    for (name, offs, n_ids, n_seqs, trunc, out_len, want) in &cases {
        let (s, m) = encode_raw(
            l.model,
            ids.as_ptr(),
            *n_ids,
            offs.as_ptr(),
            *n_seqs,
            *trunc,
            out.as_mut_ptr(),
            *out_len,
        );
        assert_eq!(s, TESSL_ERR, "{name}: {m}");
        assert!(m.contains(want), "{name}: {m:?} lacks {want:?}");
        assert!(out.iter().all(|&v| v == SENTINEL), "{name} wrote the output");
    }

    // Null pointers, each alone.
    let (s, m) = encode_raw(
        l.model,
        ptr::null(),
        n_ids,
        offsets.as_ptr(),
        n_seqs,
        0,
        out.as_mut_ptr(),
        out_len,
    );
    assert_eq!((s, m.contains("null")), (TESSL_ERR, true), "null ids: {m}");
    let (s, m) = encode_raw(
        l.model,
        ids.as_ptr(),
        n_ids,
        ptr::null(),
        n_seqs,
        0,
        out.as_mut_ptr(),
        out_len,
    );
    assert_eq!((s, m.contains("null")), (TESSL_ERR, true), "null offsets: {m}");
    let (s, m) = encode_raw(
        l.model,
        ids.as_ptr(),
        n_ids,
        offsets.as_ptr(),
        n_seqs,
        0,
        ptr::null_mut(),
        out_len,
    );
    assert_eq!((s, m.contains("null")), (TESSL_ERR, true), "null out: {m}");

    // A misaligned offsets pointer is a refusal, not a misaligned read.
    let mut bytes = vec![0u8; (offsets.len() + 1) * 8 + 8];
    let aligned = bytes.as_ptr().align_offset(8);
    let skew = unsafe { bytes.as_mut_ptr().add(aligned + 1) };
    unsafe { ptr::copy_nonoverlapping(offsets.as_ptr().cast::<u8>(), skew, offsets.len() * 8) };
    let (s, m) = encode_raw(
        l.model,
        ids.as_ptr(),
        n_ids,
        skew.cast::<u64>(),
        n_seqs,
        0,
        out.as_mut_ptr(),
        out_len,
    );
    assert_eq!(s, TESSL_ERR, "misaligned offsets: {m}");
    assert!(m.contains("not aligned"), "{m}");

    // A token id past the vocabulary is refused before any GPU work.
    let bad = vec![vec![1u32, VOCAB], vec![2]];
    let (s, m, o) = encode_seqs(l.model, &bad, 0);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("token id 48"), "{m}");
    assert!(o.iter().all(|&v| v == SENTINEL));

    // After all of that the handle still embeds, and embeds what it did before.
    let (s, m, first) = encode_seqs(l.model, &seqs, 0);
    assert_eq!(s, TESSL_OK, "{m}");
    let (_, _, again) = encode_seqs(l.model, &seqs, 0);
    assert_eq!(first, again);
}

#[test]
fn a_batch_at_the_limits_is_accepted() {
    let _g = gpu_lock();
    let l = Loaded::new();
    // MAX_BATCH one-token sequences: the largest n_seqs, offsets 0..=256.
    let seqs: Vec<Vec<u32>> = (0..MAX_BATCH as u32).map(|i| vec![i % VOCAB]).collect();
    let (s, m, out) = encode_seqs(l.model, &seqs, 0);
    assert_eq!(s, TESSL_OK, "{m}");
    assert_eq!(out.len(), MAX_BATCH * DIM);
    assert!(out.iter().all(|v| v.is_finite()));
    // The same token twice is the same row, up to GEMM tiling.
    let (a, b) = (&out[..DIM], &out[VOCAB as usize * DIM..(VOCAB as usize + 1) * DIM]);
    let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(worst < 1e-5, "rows 0 and {VOCAB} (the same token) differ by {worst}");
}

#[test]
fn a_bad_snapshot_is_refused_with_out_null_and_the_runtime_usable() {
    let _g = gpu_lock();
    let mut err = [0 as c_char; ERR_LEN];
    let rt = unsafe { tessl_runtime_new(err.as_mut_ptr(), ERR_LEN) };
    assert!(!rt.is_null(), "{}", msg(&err));
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/capi_embedgemma2_bad");
    let _ = std::fs::remove_dir_all(&base);
    let mk = |name: &str| {
        let d = base.join(name);
        std::fs::create_dir_all(&d).unwrap();
        d
    };
    let good = snapshot();
    let weights = std::fs::read(good.join("model.safetensors")).unwrap();

    let no_weights = mk("no_weights");
    std::fs::write(no_weights.join("config.json"), CONFIG).unwrap();
    let no_config = mk("no_config");
    std::fs::write(no_config.join("model.safetensors"), &weights).unwrap();
    let config_is_dir = mk("config_is_dir");
    std::fs::create_dir_all(config_is_dir.join("config.json")).unwrap();
    std::fs::write(config_is_dir.join("model.safetensors"), &weights).unwrap();
    let junk_config = mk("junk_config");
    std::fs::write(junk_config.join("config.json"), "{not json").unwrap();
    std::fs::write(junk_config.join("model.safetensors"), &weights).unwrap();
    let huge_config = mk("huge_config");
    let f = std::fs::File::create(huge_config.join("config.json")).unwrap();
    f.set_len(64 << 20).unwrap();
    std::fs::write(huge_config.join("model.safetensors"), &weights).unwrap();
    let junk_weights = mk("junk_weights");
    std::fs::write(junk_weights.join("config.json"), CONFIG).unwrap();
    std::fs::write(junk_weights.join("model.safetensors"), b"not safetensors").unwrap();

    let path = |p: &Path| p.to_str().unwrap().as_bytes().to_vec();
    let mut nul = path(good);
    nul.push(0);
    let too_long = vec![b'a'; TESSL_EMBEDGEMMA2_MAX_PATH_BYTES as usize + 1];
    let cases: Vec<(&str, Vec<u8>, &str)> = vec![
        ("a missing directory", path(&base.join("absent")), "config.json"),
        ("no model.safetensors", path(&no_weights), "model.safetensors"),
        ("no config.json", path(&no_config), "config.json"),
        ("config.json a directory", path(&config_is_dir), "not a regular file"),
        ("config.json not JSON", path(&junk_config), "config.json"),
        ("config.json past the cap", path(&huge_config), "config cap"),
        (
            "model.safetensors not safetensors",
            path(&junk_weights),
            "model.safetensors",
        ),
        ("not UTF-8", vec![0xff, 0xfe, b'/'], "not UTF-8"),
        ("a NUL byte", nul, "NUL"),
        ("a path past the cap", too_long, "snapshot_dir is"),
        ("an empty path", Vec::new(), "snapshot_dir is 0 bytes"),
    ];
    for (name, dir, want) in &cases {
        let (s, m, out) = load_raw(rt, dir);
        assert_eq!(s, TESSL_ERR, "{name}: {m}");
        assert!(m.contains(want), "{name}: {m:?} lacks {want:?}");
        assert!(out.is_null(), "{name}: *out is not null after a refused load");
    }

    // A null dir with a non-zero length, a null out, and a length that does not
    // fit a path.
    let mut out = 0x10 as *mut TesslEmbedGemma2;
    let s = unsafe { tessl_embedgemma2_load(rt, ptr::null(), 8, &mut out, err.as_mut_ptr(), ERR_LEN) };
    assert_eq!(s, TESSL_ERR);
    assert!(out.is_null());
    assert!(msg(&err).contains("null snapshot_dir"), "{}", msg(&err));
    let dir = path(good);
    let s = unsafe {
        tessl_embedgemma2_load(
            rt,
            dir.as_ptr().cast(),
            dir.len() as u64,
            ptr::null_mut(),
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    assert_eq!(s, TESSL_ERR);
    assert!(msg(&err).contains("null out"), "{}", msg(&err));
    let s = unsafe { tessl_embedgemma2_load(rt, dir.as_ptr().cast(), u64::MAX, &mut out, err.as_mut_ptr(), ERR_LEN) };
    assert_eq!(s, TESSL_ERR);
    assert!(out.is_null());

    // After every refusal the runtime loads the good snapshot.
    let (s, m, model) = load_raw(rt, &dir);
    assert_eq!(s, TESSL_OK, "{m}");
    assert_eq!(unsafe { tessl_embedgemma2_free(model) }, TESSL_OK);
    assert_eq!(unsafe { tessl_runtime_free(rt) }, TESSL_OK);
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn a_model_used_from_another_thread_is_refused() {
    let _g = gpu_lock();
    let l = Loaded::new();
    let model = l.model as usize;
    let seqs = vec![vec![1u32, 2, 3]];
    let (s, m, out) = std::thread::spawn(move || encode_seqs(model as *mut TesslEmbedGemma2, &seqs, 0))
        .join()
        .unwrap();
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("thread other than the one that created it"), "{m}");
    assert!(out.iter().all(|&v| v == SENTINEL));
    // A free from the wrong thread leaks rather than drops, and says so.
    let model = l.model as usize;
    let freed = std::thread::spawn(move || unsafe { tessl_embedgemma2_free(model as *mut TesslEmbedGemma2) })
        .join()
        .unwrap();
    assert_eq!(freed, TESSL_ERR);
    // The owner's handle is untouched and still works.
    assert_eq!(encode_seqs(l.model, &[vec![1u32, 2, 3]], 0).0, TESSL_OK);
}
