//! Checkpoint-reader bounds (card ft-51395822): the payload budget, the npy
//! header framing, order-independent shape overflow, in-place-rewrite
//! detection and `CheckpointView::tensor_info`. CPU only, and no test here
//! allocates a payload it declares: the oversized files are sparse and are
//! only ever opened, never read.

use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use tessl::npy::{read_npy, read_npy_with_limit};
use tessl::safetensors::{CheckpointView, Dtype, SafeTensors};

static SEQ: AtomicUsize = AtomicUsize::new(0);

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("tessl-reader-bounds-{}-{tag}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const GIB: u64 = 1 << 30;

/// A sparse `.safetensors` file declaring one U8 tensor of `bytes` bytes: the
/// header is real, the data buffer is a hole.
fn sparse_st(path: &Path, name: &str, bytes: u64) {
    let header = format!("{{\"{name}\":{{\"dtype\":\"U8\",\"shape\":[{bytes}],\"data_offsets\":[0,{bytes}]}}}}");
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.set_len(8 + header.len() as u64 + bytes).unwrap();
}

/// A small real file: tensors `a` and `b`, two F32 each.
fn small_st(path: &Path) {
    let header = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#;
    let mut b = (header.len() as u64).to_le_bytes().to_vec();
    b.extend_from_slice(header.as_bytes());
    for v in [1.0f32, 2.0, 3.0, 4.0] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, b).unwrap();
}

#[test]
fn a_512_gib_declaration_is_refused_at_open_by_name() {
    let s = Scratch::new("sparse");
    let p = s.path("huge.safetensors");
    sparse_st(&p, "t", 512 * GIB);
    // Default budget: this machine's physical memory. Only `open` runs; on
    // the unfixed code it succeeds, which is the gate that let a read go on
    // to fill memory.
    let e = SafeTensors::open(&p).unwrap_err();
    assert!(e.contains("payload budget exceeded"), "{e}");
    assert!(e.contains("physical memory"), "{e}");
    assert!(e.contains(&(512 * GIB).to_string()), "{e}");
    // The view and the explicit limit go through the same owner.
    let e = CheckpointView::open(&p).unwrap_err();
    assert!(e.contains("payload budget exceeded"), "{e}");
    let e = SafeTensors::open_with_limit(&p, Some(GIB)).unwrap_err();
    assert!(e.contains("the caller's limit of 1073741824"), "{e}");
}

#[test]
fn the_budget_is_the_callers_limit_when_given_and_boundary_is_inclusive() {
    let s = Scratch::new("limit");
    let p = s.path("small.safetensors");
    small_st(&p);
    assert!(SafeTensors::open_with_limit(&p, Some(16)).is_ok());
    let e = SafeTensors::open_with_limit(&p, Some(15)).unwrap_err();
    assert!(
        e.contains("declares 16 tensor bytes") && e.contains("limit of 15"),
        "{e}"
    );
    // A caller's limit above physical memory is the caller's call.
    assert!(SafeTensors::open_with_limit(&p, Some(u64::MAX)).is_ok());
    assert!(SafeTensors::open(&p).is_ok());
}

#[test]
fn a_sharded_checkpoint_is_budgeted_as_a_whole() {
    let s = Scratch::new("shards");
    for (f, n) in [("s1.safetensors", "x"), ("s2.safetensors", "y")] {
        sparse_st(&s.path(f), n, 300 * GIB);
    }
    std::fs::write(
        s.path("model.safetensors.index.json"),
        r#"{"weight_map":{"x":"s1.safetensors","y":"s2.safetensors"}}"#,
    )
    .unwrap();
    // Each shard alone fits 400 GiB; together they do not.
    assert!(CheckpointView::open_with_limit(&s.path("s1.safetensors"), Some(400 * GIB)).is_ok());
    let e = CheckpointView::open_with_limit(&s.0, Some(400 * GIB)).unwrap_err();
    assert!(
        e.contains("payload budget exceeded") && e.contains(&(600 * GIB).to_string()),
        "{e}"
    );
    assert!(CheckpointView::open_with_limit(&s.0, Some(600 * GIB)).is_ok());
}

/// An npy file: `version`, a `dict` padded by `pad_to` (0 = no padding at all)
/// and `newline`, then `payload`.
fn npy(version: u8, dict: &str, pad_to: usize, newline: bool, payload: &[u8]) -> Vec<u8> {
    let prefix = if version == 1 { 10 } else { 12 };
    let mut h = dict.as_bytes().to_vec();
    if pad_to > 0 {
        while (prefix + h.len() + usize::from(newline)) % pad_to != 0 {
            h.push(b' ');
        }
    }
    if newline {
        h.push(b'\n');
    }
    let mut b = b"\x93NUMPY".to_vec();
    b.extend_from_slice(&[version, 0]);
    if version == 1 {
        b.extend_from_slice(&(h.len() as u16).to_le_bytes());
    } else {
        b.extend_from_slice(&(h.len() as u32).to_le_bytes());
    }
    b.extend_from_slice(&h);
    b.extend_from_slice(payload);
    b
}

fn npy_result(s: &Scratch, bytes: &[u8]) -> Result<Vec<usize>, String> {
    let p = s.path("t.npy");
    std::fs::write(&p, bytes).unwrap();
    read_npy(&p).map(|a| a.shape)
}

const DICT: &str = "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }";
const PAYLOAD: [u8; 4] = [0, 0, 128, 63];

#[test]
fn npy_headers_follow_the_format_spec_in_every_version() {
    let s = Scratch::new("npy");
    for v in [1u8, 2, 3] {
        assert_eq!(
            npy_result(&s, &npy(v, DICT, 64, true, &PAYLOAD)).unwrap(),
            vec![1],
            "v{v}"
        );
        // Aligned to 16 only (NumPy before 1.14 wrote that; the spec says 64).
        let b16 = npy(v, DICT, 16, true, &PAYLOAD);
        if (b16.len() - PAYLOAD.len()) % 64 != 0 {
            assert!(npy_result(&s, &b16).unwrap_err().contains("multiple of 64"), "v{v}");
        }
        // No padding, no newline.
        let e = npy_result(&s, &npy(v, DICT, 0, false, &PAYLOAD)).unwrap_err();
        assert!(e.contains("multiple of 64") || e.contains("newline"), "v{v}: {e}");
        // Aligned but without the terminator: the last pad byte is a space.
        let mut h = DICT.to_string();
        let prefix = if v == 1 { 10 } else { 12 };
        while (prefix + h.len()) % 64 != 0 {
            h.push(' ');
        }
        let e = npy_result(&s, &npy(v, &h, 0, false, &PAYLOAD)).unwrap_err();
        assert!(e.contains("does not end with a newline"), "v{v}: {e}");
        // Padding that is not spaces (a NUL before the newline).
        let mut h = DICT.to_string();
        while (prefix + h.len() + 1) % 64 != 0 {
            h.push('\0');
        }
        let e = npy_result(&s, &npy(v, &h, 0, true, &PAYLOAD)).unwrap_err();
        assert!(e.contains("closed by '}'"), "v{v}: {e}");
    }
}

#[test]
fn npy_header_text_is_ascii_before_v3_and_utf8_in_v3() {
    let s = Scratch::new("npy-enc");
    let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), 'note': '\u{e9}', }";
    for v in [1u8, 2] {
        let e = npy_result(&s, &npy(v, dict, 64, true, &PAYLOAD)).unwrap_err();
        assert!(e.contains("must be ASCII"), "v{v}: {e}");
    }
    assert_eq!(npy_result(&s, &npy(3, dict, 64, true, &PAYLOAD)).unwrap(), vec![1]);
}

#[test]
fn npy_payload_budget_refuses_before_allocating() {
    let s = Scratch::new("npy-budget");
    let p = s.path("t.npy");
    std::fs::write(&p, npy(1, DICT, 64, true, &PAYLOAD)).unwrap();
    assert!(read_npy_with_limit(&p, Some(4)).is_ok());
    let e = read_npy_with_limit(&p, Some(3)).unwrap_err();
    assert!(e.contains("payload budget exceeded") && e.contains("declares 4"), "{e}");
    // A sparse file declaring 512 GiB of f4: refused by the default budget
    // without a read (the file's length matches the shape, so it reaches the
    // budget check, the last gate before the allocation).
    let dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
        128 * GIB
    );
    let bytes = npy(1, &dict, 64, true, &[]);
    let hdr_len = bytes.len() as u64;
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(&bytes).unwrap();
    f.seek(SeekFrom::Start(0)).unwrap();
    f.set_len(hdr_len + 512 * GIB).unwrap();
    let e = read_npy(&p).unwrap_err();
    assert!(e.contains("payload budget exceeded"), "{e}");
}

/// Safetensors files whose one tensor has `shape` (U8, `data_offsets` [0, 0]).
fn shape_verdict(s: &Scratch, shape: [u64; 3]) -> Result<(), String> {
    let p = s.path("shape.safetensors");
    let header = format!(
        "{{\"t\":{{\"dtype\":\"U8\",\"shape\":[{},{},{}],\"data_offsets\":[0,0]}}}}",
        shape[0], shape[1], shape[2]
    );
    let mut b = (header.len() as u64).to_le_bytes().to_vec();
    b.extend_from_slice(header.as_bytes());
    std::fs::write(&p, b).unwrap();
    SafeTensors::open(&p)
        .map(|_| ())
        .map_err(|e| e.replace(&p.display().to_string(), "<f>"))
}

#[test]
fn shape_overflow_verdict_does_not_depend_on_dimension_order() {
    let s = Scratch::new("perm");
    let dims = [0u64, 1 << 32, 1 << 32];
    let perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    let verdicts: Vec<_> = perms
        .iter()
        .map(|p| shape_verdict(&s, [dims[p[0]], dims[p[1]], dims[p[2]]]))
        .collect();
    assert!(verdicts.iter().all(|v| *v == verdicts[0]), "{verdicts:?}");
    // The empty tensor is a valid empty tensor.
    assert!(verdicts[0].is_ok(), "{verdicts:?}");
    // Without the zero, the same dims overflow in every order.
    let e = shape_verdict(&s, [1 << 32, 1 << 32, 1]).unwrap_err();
    assert!(e.contains("overflows"), "{e}");
}

#[test]
fn a_file_rewritten_in_place_after_open_is_refused_at_the_next_read() {
    let s = Scratch::new("rewrite");
    let p = s.path("w.safetensors");
    small_st(&p);
    let st = SafeTensors::open(&p).unwrap();
    assert_eq!(st.read_f32("a").unwrap().1, vec![1.0, 2.0]);
    assert_eq!(
        st.read_f32("a").unwrap().1,
        vec![1.0, 2.0],
        "an untouched file keeps reading"
    );

    // Same length, new bytes, same inode.
    std::thread::sleep(std::time::Duration::from_millis(20));
    let mut f = std::fs::OpenOptions::new().write(true).open(&p).unwrap();
    f.seek(SeekFrom::End(-16)).unwrap();
    f.write_all(&[0u8; 16]).unwrap();
    drop(f);
    let e = st.read_f32("a").unwrap_err();
    assert!(e.contains("file changed since it was opened"), "{e}");
    // Growth is seen too (it used to be silent; only shrinking failed).
    let s2 = Scratch::new("grow");
    let p2 = s2.path("g.safetensors");
    small_st(&p2);
    let st2 = SafeTensors::open(&p2).unwrap();
    let mut f = std::fs::OpenOptions::new().append(true).open(&p2).unwrap();
    f.write_all(&[7u8; 8]).unwrap();
    drop(f);
    let e = st2.read_f32("b").unwrap_err();
    assert!(e.contains("file changed since it was opened"), "{e}");
}

#[test]
fn a_file_replaced_at_the_path_is_not_read_at_all() {
    let s = Scratch::new("replace");
    let p = s.path("r.safetensors");
    small_st(&p);
    let st = SafeTensors::open(&p).unwrap();
    std::fs::remove_file(&p).unwrap();
    std::fs::write(&p, b"not safetensors").unwrap();
    // The open handle still names the original, unchanged inode.
    assert_eq!(st.read_f32("b").unwrap().1, vec![3.0, 4.0]);
}

#[test]
fn tensor_info_gives_dtype_and_shape_across_shards() {
    let s = Scratch::new("info");
    small_st(&s.path("s1.safetensors"));
    let header = r#"{"c":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
    let mut b = (header.len() as u64).to_le_bytes().to_vec();
    b.extend_from_slice(header.as_bytes());
    b.extend_from_slice(&[0u8; 12]);
    std::fs::write(s.path("s2.safetensors"), b).unwrap();
    std::fs::write(
        s.path("model.safetensors.index.json"),
        r#"{"weight_map":{"a":"s1.safetensors","b":"s1.safetensors","c":"s2.safetensors"}}"#,
    )
    .unwrap();
    let v = CheckpointView::open(&s.0).unwrap();
    let a = v.tensor_info("a").unwrap();
    assert_eq!((a.dtype, a.shape.as_slice()), (Dtype::F32, &[2usize][..]));
    let c = v.tensor_info("c").unwrap();
    assert_eq!((c.dtype, c.shape.as_slice()), (Dtype::BF16, &[2usize, 3][..]));
    assert!(v.tensor_info("nope").is_none());
}
