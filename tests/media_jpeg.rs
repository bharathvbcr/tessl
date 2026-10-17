//! The native JPEG decoder (`tessl::media::jpeg`) against the reference the
//! EmbeddingGemma 2 path uses: `torchvision.io.decode_image(path, mode=RGB)`, whose
//! JPEG decoder is libjpeg-turbo 3.1.3 (8-bit, `JDCT_ISLOW`, fancy upsampling).
//!
//! `tests/fixtures/jpeg/` is written by `scripts/gen_jpeg_fixtures.py` (see its
//! docstring): `NAME.jpg` is an input, `NAME.jpg.rgb` the oracle's decoded bytes,
//! `manifest.txt` says per case whether the oracle decodes it (`expect=ok`) or
//! refuses it (`expect=error`), and whether this decoder is expected to match
//! (`tessl=equal`) or to refuse by name a stream the oracle does decode
//! (`tessl=refused`: lossless SOF3 and libjpeg-turbo's block smoothing).
//! `fuzz_corpus.bin` holds mutated inputs with the oracle's verdict and pixels.
//!
//! Three kinds of test: byte-for-byte equality wherever the oracle decodes;
//! agreement on refusal wherever it does not; and no panic, ever, on mutated,
//! truncated or oversized input (every bound is checked before it allocates).
//! Everything runs without a device.
//!
//! `live_fuzz_against_the_oracle` is ignored: it needs the Python environment.
//! Generate a directory with
//! `python scripts/gen_jpeg_fixtures.py --live-fuzz DIR COUNT SEED` and run it with
//! `JPEG_LIVE_FUZZ_DIR=DIR cargo test --release --test media_jpeg -- --ignored`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use tessl::media::jpeg::{self, JpegCoding, JpegColorSpace, JpegError, MAX_SCANS};

/// A bound on `width * height` for the tests that decode untrusted bytes.
const FUZZ_MAX_PIXELS: usize = 1 << 24;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/jpeg")
}

fn read(name: &str) -> Vec<u8> {
    std::fs::read(dir().join(name)).unwrap_or_else(|e| panic!("tests/fixtures/jpeg/{name}: {e}"))
}

struct Case {
    name: String,
    ok: bool,
    refused: bool,
    class: String,
    w: usize,
    h: usize,
    pil: String,
}

fn cases() -> Vec<Case> {
    let text = String::from_utf8(read("manifest.txt")).expect("manifest.txt is UTF-8");
    text.lines()
        .map(|line| {
            let mut it = line.split_whitespace();
            assert_eq!(it.next(), Some("case"), "bad manifest line: {line}");
            let name = it.next().expect("case name").to_string();
            let kv: BTreeMap<&str, &str> = it.filter_map(|t| t.split_once('=')).collect();
            let num = |k: &str| kv.get(k).map_or(0, |v| v.parse().expect("number"));
            Case {
                name,
                ok: kv["expect"] == "ok",
                refused: kv.get("tessl") == Some(&"refused"),
                class: kv["class"].to_string(),
                w: num("w"),
                h: num("h"),
                pil: kv.get("pil").map_or(String::new(), |s| (*s).to_string()),
            }
        })
        .collect()
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<(usize, u8, u8)> {
    a.iter().zip(b).position(|(x, y)| x != y).map(|i| (i, a[i], b[i]))
}

#[test]
fn the_manifest_lists_every_input_and_every_golden() {
    let cases = cases();
    let mut on_disk: Vec<String> = std::fs::read_dir(dir())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".jpg") || n.ends_with(".jpg.rgb"))
        .collect();
    on_disk.sort();
    let mut listed: Vec<String> = Vec::new();
    for c in &cases {
        listed.push(format!("{}.jpg", c.name));
        if c.ok {
            listed.push(format!("{}.jpg.rgb", c.name));
        }
    }
    listed.sort();
    assert_eq!(on_disk, listed, "manifest.txt and the directory disagree");
    let mut names: Vec<&str> = cases.iter().map(|c| c.name.as_str()).collect();
    names.dedup();
    assert_eq!(names.len(), cases.len(), "duplicate case names");
}

/// The classes the brief names, each with at least the cases listed. A fixture
/// class that went missing would otherwise pass vacuously.
#[test]
fn the_fixture_grid_covers_every_class_the_decoder_claims() {
    let cases = cases();
    let has = |n: &str| cases.iter().any(|c| c.name == n);
    for n in [
        // sizes
        "size_1x1_420",
        "size_1x1_444",
        "size_7x13_420",
        "size_13x7_444",
        "size_4097x3_420",
        "size_3x4097_420",
        "size_65x129_prog",
        // every subsampling, baseline and progressive
        "sub_444_baseline",
        "sub_422_baseline",
        "sub_420_baseline",
        "sub_411_baseline",
        "sub_440_baseline",
        "sub_444_progressive",
        "sub_422_progressive",
        "sub_420_progressive",
        "sub_411_progressive",
        "sub_440_progressive",
        // restart markers
        "rst_1_420",
        "rst_progressive",
        "rst_arith",
        // grayscale, CMYK, YCCK, RGB
        "gray_baseline",
        "gray_progressive",
        "cmyk_pil_adobe",
        "cmyk4_adobe_0",
        "cmyk4_adobe_2",
        "ycck_flat",
        "rgb3_adobe_0_jfif0",
        "rgb3_ids_RGB",
        // qualities
        "quality_1_420",
        "quality_50_420",
        "quality_100_420",
        // arithmetic
        "arith_sequential_420",
        "arith_progressive",
        // the IDCT's extremes
        "idct_dense32767_q255",
        "idct_q65535",
    ] {
        assert!(has(n), "fixture {n} is missing from manifest.txt");
    }
    let count = |class: &str| cases.iter().filter(|c| c.class == class && c.ok && !c.refused).count();
    for (class, at_least) in [
        ("subsampling", 5),
        ("progressive", 4),
        ("sampling", 15),
        ("size", 40),
        ("restart", 8),
        ("grayscale", 3),
        ("quality", 9),
        ("colour", 20),
        ("arithmetic", 5),
        ("idct", 5),
        ("markers", 6),
        ("layout", 6),
    ] {
        assert!(
            count(class) >= at_least,
            "class {class}: {} cases, expected at least {at_least}",
            count(class)
        );
    }
}

#[test]
fn fixtures_equal_the_oracle_byte_for_byte() {
    let cases = cases();
    let mut per_class: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut failures = Vec::new();
    for c in &cases {
        let input = read(&format!("{}.jpg", c.name));
        let got = jpeg::decode(&input, 1 << 28);
        let entry = per_class.entry(c.class.clone()).or_default();
        entry.1 += 1;
        match (c.ok, c.refused, got) {
            (true, false, Ok(img)) => {
                let golden = read(&format!("{}.jpg.rgb", c.name));
                if (img.width, img.height) != (c.w, c.h) {
                    failures.push(format!(
                        "{}: size {}x{}, oracle {}x{}",
                        c.name, img.width, img.height, c.w, c.h
                    ));
                } else if img.rgb.len() != golden.len() {
                    failures.push(format!("{}: {} bytes, oracle {}", c.name, img.rgb.len(), golden.len()));
                } else if let Some((i, a, b)) = first_diff(&img.rgb, &golden) {
                    failures.push(format!(
                        "{}: first difference at byte {i}: {a} against the oracle's {b}",
                        c.name
                    ));
                } else {
                    entry.0 += 1;
                }
            }
            (true, false, Err(e)) => failures.push(format!("{}: the oracle decodes it, this does not: {e}", c.name)),
            (true, true, Err(JpegError::Unsupported(msg))) => {
                let named = match c.class.as_str() {
                    "smoothing" => msg.contains("block smoothing"),
                    "lossless" => msg.contains("lossless"),
                    _ => false,
                };
                if named {
                    entry.0 += 1;
                } else {
                    failures.push(format!(
                        "{}: refused, but not by the name its class promises: {msg}",
                        c.name
                    ));
                }
            }
            (true, true, other) => failures.push(format!(
                "{}: expected a named refusal, got {:?}",
                c.name,
                other.map(|i| i.width)
            )),
            (false, _, Err(_)) => entry.0 += 1,
            (false, _, Ok(img)) => failures.push(format!(
                "{}: the oracle refuses it, this decodes {}x{}",
                c.name, img.width, img.height
            )),
        }
    }
    println!("{:<14} {:>8} {:>8}", "class", "agree", "cases");
    for (class, (agree, total)) in &per_class {
        println!("{class:<14} {agree:>8} {total:>8}");
    }
    let (agree, total) = per_class.values().fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    println!("{:<14} {agree:>8} {total:>8}", "all");
    assert!(
        failures.is_empty(),
        "{} of {} cases disagree:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

/// PIL (libjpeg-turbo 3.1.4.1) and torchvision (3.1.3) agree on every case both
/// decode; the manifest records it, so the choice of oracle is not load-bearing.
#[test]
fn the_two_candidate_oracles_agree_on_every_fixture() {
    let cases = cases();
    let both: Vec<&Case> = cases.iter().filter(|c| c.ok).collect();
    assert!(both.len() > 100, "{} decoded cases", both.len());
    for c in both {
        assert!(c.pil == "equal" || c.pil == "error", "{}: PIL {}", c.name, c.pil);
    }
}

#[test]
fn the_decoder_reports_what_the_file_said() {
    let get = |n: &str| jpeg::decode(&read(&format!("{n}.jpg")), 1 << 28).unwrap().info;
    let i = get("sub_420_progressive");
    assert_eq!(
        (i.components, i.coding, i.color_space),
        (3, JpegCoding::Progressive, JpegColorSpace::YCbCr)
    );
    assert_eq!(i.sampling, vec![(2, 2), (1, 1), (1, 1)]);
    assert!(i.jfif);
    assert!(i.scans > 3);
    let i = get("arith_sequential_420");
    assert_eq!(i.coding, JpegCoding::SequentialArithmetic);
    let i = get("arith_progressive");
    assert_eq!(i.coding, JpegCoding::ProgressiveArithmetic);
    let i = get("gray_baseline");
    assert_eq!((i.components, i.color_space), (1, JpegColorSpace::Grayscale));
    let i = get("cmyk4_adobe_2");
    assert_eq!(
        (i.components, i.color_space, i.adobe_transform),
        (4, JpegColorSpace::Ycck, Some(2))
    );
    let i = get("cmyk_pil_adobe");
    assert_eq!(
        (i.components, i.color_space, i.adobe_transform),
        (4, JpegColorSpace::Cmyk, Some(0))
    );
    let i = get("rgb3_ids_RGB");
    assert_eq!(i.color_space, JpegColorSpace::Rgb);
}

// ---------------------------------------------------------------------------
// The fuzz corpus
// ---------------------------------------------------------------------------

struct Entry {
    data: Vec<u8>,
    /// `Some((width, height, rgb))` when the oracle decoded it.
    pixels: Option<(usize, usize, Vec<u8>)>,
}

fn le_u32(b: &[u8], at: &mut usize) -> usize {
    let v = u32::from_le_bytes(b[*at..*at + 4].try_into().unwrap()) as usize;
    *at += 4;
    v
}

/// `JFZ1`, u32 count, then per entry: u32 length, the JPEG, a flag byte; when the
/// flag is 1: u32 width, u32 height and width*height*3 oracle bytes.
fn corpus() -> Vec<Entry> {
    let b = read("fuzz_corpus.bin");
    assert_eq!(&b[..4], b"JFZ1");
    let mut at = 4;
    let n = le_u32(&b, &mut at);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let len = le_u32(&b, &mut at);
        let data = b[at..at + len].to_vec();
        at += len;
        let flag = b[at];
        at += 1;
        let pixels = if flag == 1 {
            let w = le_u32(&b, &mut at);
            let h = le_u32(&b, &mut at);
            let rgb = b[at..at + w * h * 3].to_vec();
            at += w * h * 3;
            Some((w, h, rgb))
        } else {
            None
        };
        out.push(Entry { data, pixels });
    }
    assert_eq!(at, b.len(), "trailing bytes in fuzz_corpus.bin");
    out
}

#[test]
fn the_fuzz_corpus_equals_the_oracle_wherever_it_decodes() {
    let entries = corpus();
    let (mut equal, mut refused_both, mut named, mut failures) = (0usize, 0usize, 0usize, Vec::new());
    for (i, e) in entries.iter().enumerate() {
        let got = jpeg::decode(&e.data, FUZZ_MAX_PIXELS);
        match (&e.pixels, got) {
            // A mutant that cuts a progressive file short enough for libjpeg-turbo to smooth it:
            // the one documented case where the oracle decodes and this decoder says why it won't.
            (Some(_), Err(JpegError::Unsupported(msg))) if msg.contains("block smoothing") => named += 1,
            (Some((w, h, rgb)), Ok(img)) => {
                if (img.width, img.height) == (*w, *h) && img.rgb == *rgb {
                    equal += 1;
                } else {
                    failures.push(format!("entry {i}: decodes, but not to the oracle's {w}x{h} bytes"));
                }
            }
            (Some((w, h, _)), Err(err)) => {
                failures.push(format!("entry {i}: the oracle decodes {w}x{h}, this fails: {err}"))
            }
            (None, Err(_)) => refused_both += 1,
            (None, Ok(img)) => failures.push(format!(
                "entry {i}: the oracle refuses it, this decodes {}x{}",
                img.width, img.height
            )),
        }
    }
    println!(
        "fuzz corpus: {} entries, {equal} byte-equal, {refused_both} refused by both, {named} refused by name (block smoothing)",
        entries.len()
    );
    assert!(entries.len() >= 400, "{} entries", entries.len());
    assert!(
        named <= 5,
        "{named} mutants hit the block-smoothing refusal: the corpus stopped testing equality"
    );
    assert!(
        equal >= 250 && refused_both >= 100,
        "{equal} equal, {refused_both} refused: the corpus lost its balance"
    );
    assert!(
        failures.is_empty(),
        "{} disagreements:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// No panic on hostile input
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const MARKERS: [u8; 16] = [
    0x00, 0x01, 0xC0, 0xC2, 0xC4, 0xD0, 0xD7, 0xD8, 0xD9, 0xDA, 0xDB, 0xDD, 0xE1, 0xEE, 0xFE, 0xFF,
];

fn mutate(rng: &mut Rng, base: &[u8]) -> Vec<u8> {
    let mut b = base.to_vec();
    for _ in 0..1 + rng.below(6) {
        let n = b.len();
        match rng.below(8) {
            0 => b[rng.below(n)] = rng.next() as u8,
            1 => b[rng.below(n)] ^= 1 << rng.below(8),
            2 => b[rng.below(n)] = 0xFF,
            3 => b.insert(rng.below(n), rng.next() as u8),
            4 if n > 4 => {
                b.remove(rng.below(n));
            }
            5 => {
                let at = rng.below(n);
                b.splice(at..at, [0xFF, MARKERS[rng.below(MARKERS.len())]]);
            }
            6 => {
                let (i, j) = (rng.below(n), rng.below(n));
                b.swap(i, j);
            }
            _ => b.truncate(2 + rng.below(n - 1)),
        }
        if b.len() < 3 {
            break;
        }
    }
    b
}

/// Small inputs of every class: baseline, progressive, arithmetic, restart, CMYK,
/// non-interleaved, odd sampling.
fn mutation_bases() -> Vec<(String, Vec<u8>)> {
    cases()
        .into_iter()
        .filter(|c| c.ok && !c.refused)
        .map(|c| (c.name.clone(), read(&format!("{}.jpg", c.name))))
        .filter(|(_, b)| b.len() < 12_000)
        .collect()
}

#[test]
fn mutated_inputs_never_panic() {
    let bases = mutation_bases();
    assert!(bases.len() > 100, "{} bases", bases.len());
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut ok, mut err) = (0usize, 0usize);
    for (_, base) in &bases {
        for _ in 0..40 {
            let m = mutate(&mut rng, base);
            match jpeg::decode(&m, FUZZ_MAX_PIXELS) {
                Ok(img) => {
                    assert_eq!(img.rgb.len(), img.width * img.height * 3);
                    ok += 1;
                }
                Err(_) => err += 1,
            }
        }
    }
    println!(
        "{} mutants of {} bases: {ok} decoded, {err} refused",
        ok + err,
        bases.len()
    );
    assert!(ok > 100 && err > 100, "the mutator stopped exercising both outcomes");
}

#[test]
fn every_strict_prefix_of_a_valid_file_is_an_error() {
    // The reference's source manager fails at the end of the data, and a stream is
    // complete only with its EOI; so the only prefix that decodes is the whole file.
    for name in [
        "sub_420_baseline",
        "sub_420_progressive",
        "arith_progressive",
        "rst_2_420",
        "cmyk4_adobe_2",
        "gray_baseline",
        "size_1x1_420",
    ] {
        let full = read(&format!("{name}.jpg"));
        assert!(jpeg::decode(&full, 1 << 28).is_ok(), "{name}");
        for n in 0..full.len() {
            assert!(
                jpeg::decode(&full[..n], 1 << 28).is_err(),
                "{name}: the {n}-byte prefix of {} decoded",
                full.len()
            );
        }
    }
}

#[test]
fn hostile_inputs_are_named_errors() {
    for bytes in [
        &b""[..],
        &b"\xff"[..],
        &b"\xff\xd8"[..],
        &b"\xff\xd8\xff"[..],
        &b"\xff\xd8\xff\xd9"[..],
        &b"\xff\xd8\xff\xc0\x00\x00"[..],
        &b"\xff\xd8\xff\xc0\xff\xff"[..],
        &[0xFF; 4096][..],
        &[0u8; 4096][..],
    ] {
        assert!(jpeg::decode(bytes, 1 << 24).is_err());
    }
    assert_eq!(jpeg::decode(b"\x89PNG....", 1 << 24), Err(JpegError::NotJpeg));
    assert!(!jpeg::has_jpeg_signature(b"\xff\xd8\x00"));
}

/// Offset of the `(height, width)` fields of the first SOF0 segment.
fn sof_dims(b: &[u8]) -> usize {
    b.windows(2).position(|w| w == [0xFF, 0xC0]).expect("SOF0") + 5
}

#[test]
fn an_oversized_image_is_refused_from_the_header_before_anything_is_allocated() {
    let base = read("size_1x1_444.jpg");
    let at = sof_dims(&base);
    let with_dims = |h: u16, w: u16| {
        let mut b = base.clone();
        b[at..at + 2].copy_from_slice(&h.to_be_bytes());
        b[at + 2..at + 4].copy_from_slice(&w.to_be_bytes());
        b
    };
    // The caller's bound.
    assert!(matches!(
        jpeg::decode(&with_dims(4096, 4096), 1 << 20),
        Err(JpegError::Limit(_))
    ));
    // No caller bound: the decoder's own working-memory bound.
    let t = std::time::Instant::now();
    assert!(matches!(
        jpeg::decode(&with_dims(65500, 65500), usize::MAX),
        Err(JpegError::Limit(_))
    ));
    assert!(t.elapsed().as_secs() < 5, "refusal took {:?}", t.elapsed());
    // libjpeg's own dimension limit is a format error, not a limit of this decoder.
    assert!(matches!(
        jpeg::decode(&with_dims(65501, 1), usize::MAX),
        Err(JpegError::Invalid(_))
    ));
    assert!(matches!(
        jpeg::decode(&with_dims(0, 8), usize::MAX),
        Err(JpegError::Invalid(_))
    ));
    // A big but permitted image whose entropy data ends at the EOI: libjpeg zero-fills the rest of
    // the scan (a marker inside entropy data ends it), so it decodes, in bounded memory.
    let r = jpeg::decode(&with_dims(8000, 8000), 1 << 26).expect("a marker ends the data; the rest is zeros");
    assert_eq!((r.width, r.height, r.rgb.len()), (8000, 8000, 8000 * 8000 * 3));
}

#[test]
fn the_scan_count_is_bounded() {
    // Three non-interleaved scans, then the last one repeated: libjpeg takes any
    // number of them, this decoder stops at MAX_SCANS with a named limit.
    let base = read("noninterleaved_scans.jpg");
    let sos = base.windows(2).rposition(|w| w == [0xFF, 0xDA]).expect("SOS");
    let eoi = base.len() - 2;
    assert_eq!(&base[eoi..], &[0xFF, 0xD9]);
    let scan = base[sos..eoi].to_vec();
    let mut b = base[..eoi].to_vec();
    for _ in 0..MAX_SCANS + 10 {
        b.extend_from_slice(&scan);
    }
    b.extend_from_slice(&[0xFF, 0xD9]);
    match jpeg::decode(&b, 1 << 24) {
        Err(JpegError::Limit(msg)) => assert!(msg.contains("scans"), "{msg}"),
        other => panic!("expected the scan limit, got {:?}", other.map(|i| i.width)),
    }
    // Just under the bound decodes.
    let mut b = base[..eoi].to_vec();
    for _ in 0..MAX_SCANS - 4 {
        b.extend_from_slice(&scan);
    }
    b.extend_from_slice(&[0xFF, 0xD9]);
    assert!(jpeg::decode(&b, 1 << 24).is_ok());
}

// ---------------------------------------------------------------------------
// Live fuzzing against the oracle (ignored: needs the Python environment)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "needs JPEG_LIVE_FUZZ_DIR from scripts/gen_jpeg_fixtures.py --live-fuzz"]
fn live_fuzz_against_the_oracle() {
    let d = PathBuf::from(std::env::var("JPEG_LIVE_FUZZ_DIR").expect("JPEG_LIVE_FUZZ_DIR"));
    let mut names: Vec<String> = std::fs::read_dir(&d)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".jpg"))
        .collect();
    names.sort();
    assert!(!names.is_empty(), "no mutants in {}", d.display());
    let (mut equal, mut refused, mut named, mut bad) = (0usize, 0usize, 0usize, Vec::new());
    for n in &names {
        let stem = n.trim_end_matches(".jpg");
        let data = std::fs::read(d.join(n)).unwrap();
        let got = jpeg::decode(&data, FUZZ_MAX_PIXELS);
        let rgb = d.join(format!("{stem}.rgb"));
        if rgb.exists() {
            let want = std::fs::read(&rgb).unwrap();
            match got {
                Ok(img) if img.rgb == want => equal += 1,
                Ok(img) => bad.push(format!("{n}: decodes {}x{} to different bytes", img.width, img.height)),
                Err(JpegError::Unsupported(msg)) if msg.contains("block smoothing") => named += 1,
                Err(e) => bad.push(format!("{n}: the oracle decodes it, this fails: {e}")),
            }
        } else if got.is_err() {
            refused += 1;
        } else {
            bad.push(format!("{n}: the oracle refuses it, this decodes"));
        }
    }
    println!("{} mutants: {equal} byte-equal, {refused} refused by both, {named} refused by name (block smoothing), {} disagree", names.len(), bad.len());
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
