//! `tessl::npy` against malformed files. CPU only.
//!
//! The reader used to trust its header: an unchecked shape product (wrapping
//! in release, so `(2^62, 4)` read as an empty array), a declared shape
//! allocated before anything was compared with the file (a tiny file could
//! ask for terabytes and abort the process), any version byte other than 1
//! parsed as 2.0, `fortran_order` found by substring, and trailing bytes
//! ignored. Each is now an error.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use tessl::npy::{read_npy, transpose_last2, write_npy_f32, MAX_NPY_HEADER_BYTES};

static SEQ: AtomicUsize = AtomicUsize::new(0);

fn tmp(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("tessl-npy-h-{}-{tag}-{n}.npy", std::process::id()))
}

/// A v1.0 file with `header` (padded as NumPy pads it) and `payload`.
fn v1(tag: &str, header: &str, payload: &[u8]) -> PathBuf {
    let mut h = header.to_string();
    let pad = (64 - (10 + h.len() + 1) % 64) % 64;
    h.push_str(&" ".repeat(pad));
    h.push('\n');
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend_from_slice(&(h.len() as u16).to_le_bytes());
    b.extend_from_slice(h.as_bytes());
    b.extend_from_slice(payload);
    let p = tmp(tag);
    std::fs::write(&p, b).unwrap();
    p
}

fn err(p: PathBuf) -> String {
    let r = read_npy(&p);
    let _ = std::fs::remove_file(&p);
    match r {
        Ok(a) => panic!("{} read as shape {:?}, expected an error", p.display(), a.shape),
        Err(e) => e,
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn well_formed_files_still_read() {
    let p = v1(
        "ok",
        "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }",
        &f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );
    let a = read_npy(&p).unwrap();
    std::fs::remove_file(&p).unwrap();
    assert_eq!(a.shape, vec![2, 3]);
    assert_eq!(a.f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // A scalar and an empty array.
    let p = v1("scalar", "{'descr': '<f4', 'fortran_order': False, 'shape': (), }", &f32_bytes(&[7.0]));
    assert_eq!(read_npy(&p).unwrap().f32_slice().unwrap(), &[7.0]);
    std::fs::remove_file(&p).unwrap();
    let p = v1("empty", "{'descr': '<f4', 'fortran_order': False, 'shape': (0, 5), }", &[]);
    assert!(read_npy(&p).unwrap().f32_slice().unwrap().is_empty());
    std::fs::remove_file(&p).unwrap();
    // The writer's own output round-trips.
    let p = tmp("rt");
    write_npy_f32(&p, &[3], &[1.5, -2.0, 0.25]).unwrap();
    assert_eq!(read_npy(&p).unwrap().f32_slice().unwrap(), &[1.5, -2.0, 0.25]);
    std::fs::remove_file(&p).unwrap();
}

#[test]
fn the_payload_must_be_exactly_the_rest_of_the_file() {
    let h = "{'descr': '<f4', 'fortran_order': False, 'shape': (4,), }";
    assert!(err(v1("short", h, &f32_bytes(&[1.0, 2.0, 3.0]))).contains("but 12 follow"));
    assert!(err(v1("long", h, &f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0]))).contains("but 20 follow"));
    // A tiny file declaring a terabyte array: an error, not an allocation.
    let huge = "{'descr': '<f8', 'fortran_order': False, 'shape': (1000000000000,), }";
    assert!(err(v1("huge", huge, &[0u8; 8])).contains("follow the header"));
    // A shape whose product overflows (it used to wrap to 0 in release).
    let wrap = "{'descr': '<f4', 'fortran_order': False, 'shape': (4611686018427387904, 4), }";
    assert!(err(v1("wrap", wrap, &[])).contains("overflows"));
}

#[test]
fn headers_outside_the_supported_format_are_refused() {
    let payload = f32_bytes(&[1.0]);
    // Version bytes other than 1, 2 or 3.
    let mut b = b"\x93NUMPY\x04\x00".to_vec();
    b.extend_from_slice(&[0u8; 8]);
    let p = tmp("ver");
    std::fs::write(&p, b).unwrap();
    assert!(err(p).contains("version 4.0"));
    // A v2 header length past the file (and past the cap).
    for hl in [1000u32, (MAX_NPY_HEADER_BYTES + 1) as u32, u32::MAX] {
        let mut b = b"\x93NUMPY\x02\x00".to_vec();
        b.extend_from_slice(&hl.to_le_bytes());
        b.extend_from_slice(b"{}");
        let p = tmp("hlen");
        std::fs::write(&p, b).unwrap();
        assert!(err(p).contains("header length"), "hl {hl}");
    }
    // fortran_order missing, garbled, or True.
    let missing = "{'descr': '<f4', 'shape': (1,), }";
    assert!(err(v1("fo_missing", missing, &payload)).contains("missing fortran_order"));
    let garbled = "{'descr': '<f4', 'fortran_order': 0, 'shape': (1,), }";
    assert!(err(v1("fo_garbled", garbled, &payload)).contains("neither True nor False"));
    let fortran = "{'descr': '<f4', 'fortran_order': True, 'shape': (1,), }";
    assert!(err(v1("fo_true", fortran, &payload)).contains("fortran-order"));
    // A spaced-out `True` used to slip past the substring match.
    let spaced = "{'descr': '<f4', 'fortran_order':   True, 'shape': (1,), }";
    assert!(err(v1("fo_spaced", spaced, &payload)).contains("fortran-order"));
    // Non-UTF-8 header bytes.
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    let h = b"{'descr': '<f4', \xff}";
    b.extend_from_slice(&(h.len() as u16).to_le_bytes());
    b.extend_from_slice(h);
    let p = tmp("utf8");
    std::fs::write(&p, b).unwrap();
    assert!(err(p).contains("not UTF-8"));
    // Unsupported dtype.
    let c16 = "{'descr': '<c16', 'fortran_order': False, 'shape': (1,), }";
    assert!(err(v1("c16", c16, &[0u8; 16])).contains("unsupported dtype"));
}

#[test]
fn transpose_last2_checks_the_data_against_the_shape() {
    let mut shape = [2usize, 3];
    let mut short = vec![0.0f32; 5];
    assert!(transpose_last2(&mut short, &mut shape).unwrap_err().contains("data has 5"));
    let mut long = vec![0.0f32; 7];
    assert!(transpose_last2(&mut long, &mut shape).unwrap_err().contains("data has 7"));
    let mut ok = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    transpose_last2(&mut ok, &mut shape).unwrap();
    assert_eq!(shape, [3, 2]);
    assert_eq!(ok, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}
