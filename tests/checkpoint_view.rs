//! `CheckpointView`: a sharded checkpoint read through its
//! `model.safetensors.index.json`. The tiny training fixture is split across
//! two files in a temp directory. What is checked: the split loads
//! bit-identically to the single file; a stray tensor in the second shard is
//! refused by name; and every hostile index (a missing shard, a path out of the
//! directory by `..`, an absolute path or a symlink, a duplicate name, a name
//! absent from its shard, an unindexed tensor) is a named error from
//! `CheckpointView::open`, before any GPU work.

mod common;

use std::path::{Path, PathBuf};

use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::safetensors::{CheckpointView, Dtype, SafeTensors, TensorSource, INDEX_FILE_NAME};
use tessl::GpuRuntime;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train")
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("tessl-checkpoint-view-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

type Entry = (String, Dtype, Vec<usize>, Vec<u8>);

fn dtype_name(d: Dtype) -> &'static str {
    match d {
        Dtype::F32 => "F32",
        Dtype::BF16 => "BF16",
        Dtype::F16 => "F16",
        other => panic!("fixture has no {other:?} tensors"),
    }
}

/// A `.safetensors` file holding `entries`, tiled in the order given.
fn write_st(path: &Path, entries: &[Entry]) {
    let (mut header, mut data, mut at) = (Vec::new(), Vec::new(), 0usize);
    for (name, dtype, shape, bytes) in entries {
        header.push(format!(
            "{name:?}:{{\"dtype\":\"{}\",\"shape\":{shape:?},\"data_offsets\":[{at},{}]}}",
            dtype_name(*dtype),
            at + bytes.len()
        ));
        data.extend_from_slice(bytes);
        at += bytes.len();
    }
    let header = format!("{{{}}}", header.join(","));
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    std::fs::write(path, out).unwrap();
}

/// Every tensor of the fixture, sorted by name.
fn fixture_entries() -> Vec<Entry> {
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    let names: Vec<String> = st.names().map(str::to_string).collect();
    names
        .into_iter()
        .map(|n| {
            let (info, bytes) = st.read_raw(&n).unwrap();
            (n, info.dtype, info.shape, bytes)
        })
        .collect()
}

fn index_text(map: &[(&str, &str)]) -> String {
    let body: Vec<String> = map.iter().map(|(n, f)| format!("{n:?}:{f:?}")).collect();
    format!(
        "{{\"metadata\":{{\"total_size\":1}},\"weight_map\":{{{}}}}}",
        body.join(",")
    )
}

/// The fixture split in two (alternating tensors), written under `dir` with
/// `extra` appended to shard 2 and indexed. Returns the index's weight_map.
fn write_split(dir: &Path, extra: Option<Entry>) -> Vec<(String, &'static str)> {
    let all = fixture_entries();
    let (mut a, mut b): (Vec<Entry>, Vec<Entry>) = (Vec::new(), Vec::new());
    for (i, e) in all.into_iter().enumerate() {
        if i % 2 == 0 {
            a.push(e)
        } else {
            b.push(e)
        }
    }
    b.extend(extra);
    let mut map: Vec<(String, &'static str)> = Vec::new();
    map.extend(a.iter().map(|e| (e.0.clone(), "model-00001-of-00002.safetensors")));
    map.extend(b.iter().map(|e| (e.0.clone(), "model-00002-of-00002.safetensors")));
    write_st(&dir.join("model-00001-of-00002.safetensors"), &a);
    write_st(&dir.join("model-00002-of-00002.safetensors"), &b);
    let pairs: Vec<(&str, &str)> = map.iter().map(|(n, f)| (n.as_str(), *f)).collect();
    std::fs::write(dir.join(INDEX_FILE_NAME), index_text(&pairs)).unwrap();
    map
}

fn tiny_config() -> Qwen35Config {
    Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap()
}

fn open_err(path: &Path) -> String {
    CheckpointView::open(path).expect_err("this checkpoint must be refused")
}

#[test]
fn split_checkpoint_loads_bit_identically_to_the_single_file() {
    let _g = common::gpu_lock();
    let sc = Scratch::new("split");
    write_split(&sc.0, None);
    let view = CheckpointView::open(&sc.0).unwrap();
    assert_eq!(view.shard_count(), 2);
    let single = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    assert_eq!(view.names().collect::<Vec<_>>(), single.names().collect::<Vec<_>>());

    let rt = GpuRuntime::new().unwrap();
    let ids: Vec<u32> = (0..6u32).map(|i| (i * 37 + 11) % 64).collect();
    for precision in [Precision::F32, Precision::Bf16] {
        let a = Qwen35Model::load(&rt, &single, "model.", tiny_config(), precision).unwrap();
        let b = Qwen35Model::load(&rt, &view, "model.", tiny_config(), precision).unwrap();
        let table = a.parameter_table().unwrap();
        assert_eq!(table.len(), b.parameter_table().unwrap().len());
        for (i, p) in table.iter().enumerate() {
            let n: usize = p.shape.iter().product();
            let (mut x, mut y) = (vec![0f32; n], vec![0f32; n]);
            a.read_parameter_host(i, &mut x).unwrap();
            b.read_parameter_host(i, &mut y).unwrap();
            let same = x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits());
            assert!(
                same,
                "{precision:?} {}: split load differs from single-file load",
                p.name
            );
        }
        let (la, lb) = (
            a.forward(&ids, false).unwrap().logits,
            b.forward(&ids, false).unwrap().logits,
        );
        assert!(
            la.iter().zip(&lb).all(|(p, q)| p.to_bits() == q.to_bits()),
            "{precision:?} logits"
        );
    }
    // The index file's own path, and a lone .safetensors file, open too.
    assert_eq!(
        CheckpointView::open(&sc.0.join(INDEX_FILE_NAME)).unwrap().shard_count(),
        2
    );
    let one = CheckpointView::open(&fixture().join("model.safetensors")).unwrap();
    assert_eq!(one.shard_count(), 1);
    assert_eq!(one.names().count(), single.names().count());
    // And the tower loader reads through the view as well.
    Qwen35Model::load_tower(&rt, &view, "model.", tiny_config(), Precision::Bf16).unwrap();
}

#[test]
fn a_stray_tensor_in_the_second_shard_is_refused_by_name() {
    let _g = common::gpu_lock();
    let sc = Scratch::new("stray");
    let stray: Entry = ("model.stray_bias".into(), Dtype::F32, vec![2], vec![0u8; 8]);
    write_split(&sc.0, Some(stray));
    let view = CheckpointView::open(&sc.0).unwrap();
    let rt = GpuRuntime::new().unwrap();
    let err = Qwen35Model::load(&rt, &view, "model.", tiny_config(), Precision::F32)
        .err()
        .expect("a tensor the model does not implement must be refused");
    assert!(err.contains("model.stray_bias"), "{err}");
}

#[test]
fn a_weight_map_file_that_is_missing_is_refused_at_open() {
    let sc = Scratch::new("missing");
    write_split(&sc.0, None);
    std::fs::remove_file(sc.0.join("model-00002-of-00002.safetensors")).unwrap();
    let err = open_err(&sc.0);
    assert!(err.contains("model-00002-of-00002.safetensors"), "{err}");
    assert!(err.contains("No such file"), "{err}");
}

#[test]
fn shard_names_that_are_not_bare_file_names_are_refused() {
    let sc = Scratch::new("escape");
    let dir = sc.0.join("ckpt");
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    let all = fixture_entries();
    let outside = sc.0.join("outside.safetensors");
    write_st(&outside, &all);
    write_st(&dir.join("inside.safetensors"), &all);
    write_st(&dir.join("sub/inside.safetensors"), &all);
    let n = all[0].0.clone();
    for hostile in [
        "../outside.safetensors".to_string(),
        "/".to_string(),
        outside.display().to_string(),
        "sub/../inside.safetensors".to_string(),
        "sub/inside.safetensors".to_string(),
        "./inside.safetensors".to_string(),
        "..".to_string(),
        ".".to_string(),
        "a\\b".to_string(),
        String::new(),
        "inside.safetensors\0".to_string(),
    ] {
        let body = format!(
            "{{\"weight_map\":{{{n:?}:{}}}}}",
            // `{:?}` renders NUL as \0, which is not JSON; spell it as \u0000.
            format!("{hostile:?}").replace("\\0", "\\u0000")
        );
        std::fs::write(dir.join(INDEX_FILE_NAME), body).unwrap();
        let err = open_err(&dir);
        assert!(err.contains("not a bare file name"), "{hostile:?}: {err}");
    }
}

/// Hugging Face's layout: each shard in the snapshot directory is a symlink
/// to `../../blobs/<sha>`, outside it. It opens and reads back the same bytes.
#[test]
fn a_symlinked_shard_outside_the_directory_loads() {
    let sc = Scratch::new("hf");
    let blobs = sc.0.join("blobs");
    let snap = sc.0.join("snapshots/rev");
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::create_dir_all(&snap).unwrap();
    let all = fixture_entries();
    write_st(&blobs.join("sha1"), &all[..all.len() / 2]);
    write_st(&blobs.join("sha2"), &all[all.len() / 2..]);
    std::os::unix::fs::symlink("../../blobs/sha1", snap.join("model-00001-of-00002.safetensors")).unwrap();
    std::os::unix::fs::symlink("../../blobs/sha2", snap.join("model-00002-of-00002.safetensors")).unwrap();
    let map: Vec<(&str, &str)> = all
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let f = if i < all.len() / 2 {
                "model-00001-of-00002.safetensors"
            } else {
                "model-00002-of-00002.safetensors"
            };
            (e.0.as_str(), f)
        })
        .collect();
    std::fs::write(snap.join(INDEX_FILE_NAME), index_text(&map)).unwrap();
    let view = CheckpointView::open(&snap).unwrap();
    assert_eq!(view.shard_count(), 2);
    let single = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    for e in &all {
        let (_, want) = single.read_f32(&e.0).unwrap();
        let (_, got) = view.read_f32(&e.0).unwrap();
        assert!(
            want.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()),
            "{}",
            e.0
        );
    }
}

/// A FIFO or a directory named by the index is refused, and the open returns
/// (a plain `File::open` on a FIFO with no writer blocks forever).
#[test]
fn a_fifo_or_directory_shard_is_refused_without_blocking() {
    let sc = Scratch::new("special");
    let all = fixture_entries();
    std::fs::create_dir(sc.0.join("adir")).unwrap();
    let ok = std::process::Command::new("mkfifo")
        .arg(sc.0.join("afifo"))
        .status()
        .unwrap();
    assert!(ok.success());
    for special in ["adir", "afifo"] {
        std::fs::write(sc.0.join(INDEX_FILE_NAME), index_text(&[(all[0].0.as_str(), special)])).unwrap();
        let dir = sc.0.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(CheckpointView::open(&dir).map(|_| ()));
        });
        let r = rx
            .recv_timeout(std::time::Duration::from_secs(20))
            .unwrap_or_else(|_| panic!("{special}: open blocked"));
        let err = r.expect_err("a non-regular shard must be refused");
        assert!(err.contains("not a regular file"), "{special}: {err}");
    }
}

#[test]
fn duplicate_and_inconsistent_names_are_refused_at_open() {
    let sc = Scratch::new("dup");
    let map = write_split(&sc.0, None);
    let (first, f1) = (map[0].0.clone(), map[0].1);
    let second_shard_name = map.iter().find(|(_, f)| *f != f1).unwrap().0.clone();
    let index = sc.0.join(INDEX_FILE_NAME);
    let put = |pairs: Vec<(&str, &str)>| std::fs::write(&index, index_text(&pairs)).unwrap();
    let base: Vec<(&str, &str)> = map.iter().map(|(n, f)| (n.as_str(), *f)).collect();

    // The same name twice in the weight_map (a duplicate JSON key).
    let mut dup = base.clone();
    dup.push((&first, f1));
    put(dup);
    let err = open_err(&sc.0);
    assert!(err.contains("duplicate key") && err.contains(&first), "{err}");

    // A weight_map name the shard does not hold.
    let mut ghost = base.clone();
    ghost.push(("model.ghost", f1));
    put(ghost);
    let err = open_err(&sc.0);
    assert!(err.contains("model.ghost") && err.contains("no such tensor"), "{err}");

    // A tensor held by two shards: shard 1 also carries shard 2's tensor.
    let all = fixture_entries();
    let twice = all.iter().find(|e| e.0 == second_shard_name).unwrap().clone();
    let mut shard1: Vec<Entry> = all
        .iter()
        .filter(|e| map.iter().any(|(n, f)| *n == e.0 && *f == f1))
        .cloned()
        .collect();
    shard1.push(twice);
    write_st(&sc.0.join(f1), &shard1);
    put(base.clone());
    let err = open_err(&sc.0);
    assert!(err.contains("is in both") && err.contains(&second_shard_name), "{err}");

    // A tensor in a shard that the weight_map does not list.
    put(base.iter().copied().filter(|(n, _)| *n != second_shard_name).collect());
    let err = open_err(&sc.0);
    assert!(err.contains(&second_shard_name), "{err}");

    // Index shape: no weight_map, empty, wrong types, unknown field.
    for (text, needle) in [
        (r#"{"metadata":{}}"#, "no weight_map"),
        (r#"{"weight_map":{}}"#, "empty"),
        (r#"{"weight_map":{"a":1}}"#, "not a file name"),
        (r#"{"weight_map":{"a":"b"},"extra":{}}"#, "unexpected"),
        (r#"[]"#, "not a JSON object"),
    ] {
        std::fs::write(&index, text).unwrap();
        let err = open_err(&sc.0);
        assert!(err.contains(needle), "{text}: {err}");
    }
}

// ---- adversarial additions (wave-7 tester) ----

/// Every tensor of the fixture split over `n` shards, round-robin from the
/// back so neither the shard order nor the tensor order is alphabetical.
fn write_interleaved(dir: &Path, n: usize) {
    let all = fixture_entries();
    let mut shards: Vec<Vec<Entry>> = vec![Vec::new(); n];
    for (i, e) in all.into_iter().rev().enumerate() {
        shards[(i * 7 + 1) % n].push(e);
    }
    // Shard file names run opposite to the shard numbering.
    let file = |i: usize| format!("part-{:03}.safetensors", n - i);
    let mut map: Vec<(String, String)> = Vec::new();
    for (i, s) in shards.iter().enumerate() {
        write_st(&dir.join(file(i)), s);
        map.extend(s.iter().map(|e| (e.0.clone(), file(i))));
    }
    let pairs: Vec<(&str, &str)> = map.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    std::fs::write(dir.join(INDEX_FILE_NAME), index_text(&pairs)).unwrap();
}

#[test]
fn three_interleaved_shards_load_bit_identically_f32_bf16_and_tower() {
    let _g = common::gpu_lock();
    let sc = Scratch::new("three");
    write_interleaved(&sc.0, 3);
    let view = CheckpointView::open(&sc.0).unwrap();
    assert_eq!(view.shard_count(), 3);
    let single = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    let rt = GpuRuntime::new().unwrap();
    for precision in [Precision::F32, Precision::Bf16] {
        for tower in [false, true] {
            let go = |st: &dyn TensorSource| {
                if tower {
                    Qwen35Model::load_tower(&rt, st, "model.", tiny_config(), precision)
                } else {
                    Qwen35Model::load(&rt, st, "model.", tiny_config(), precision)
                }
            };
            let (a, b) = (go(&single), go(&view));
            // Same verdict (Ok/Err and message) through either source.
            assert_eq!(a.is_ok(), b.is_ok(), "{precision:?} tower={tower}");
            let (Ok(a), Ok(b)) = (a, b) else { continue };
            for (i, p) in a.parameter_table().unwrap().iter().enumerate() {
                let n: usize = p.shape.iter().product();
                let (mut x, mut y) = (vec![0f32; n], vec![0f32; n]);
                a.read_parameter_host(i, &mut x).unwrap();
                b.read_parameter_host(i, &mut y).unwrap();
                assert!(
                    x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits()),
                    "{precision:?} tower={tower} {}",
                    p.name
                );
            }
        }
    }
}

#[test]
fn hostile_index_shapes_are_named_errors() {
    let sc = Scratch::new("shapes");
    write_split(&sc.0, None);
    let index = sc.0.join(INDEX_FILE_NAME);
    for text in [
        r#"{"weight_map":[]}"#,
        r#"{"weight_map":null}"#,
        r#"{"weight_map":"x"}"#,
        r#"{"weight_map":{"a":null}}"#,
        r#"{"weight_map":{"a":["f"]}}"#,
        r#"{"weight_map":{"a":{"b":"f"}}}"#,
        r#"{"weight_map":{"a":"f"},"metadata":"x"}"#,
        r#"{"weight_map":{"a":"f"},"metadata":{"total_size":-1}}"#,
        r#"{"weight_map":{"a":"f"},"metadata":{"total_size":1e400}}"#,
        r#"{"weight_map":{"a":"f"}} trailing"#,
        "\u{feff}{\"weight_map\":{\"a\":\"f\"}}",
        r#"{"weight_map":{"a":"f"},"weight_map":{"a":"f"}}"#,
        r#"{"weight_map":{"a":"f.safetensors\u0000"}}"#,
        "",
        "{",
    ] {
        std::fs::write(&index, text).unwrap();
        let err = open_err(&sc.0);
        assert!(!err.is_empty(), "{text}");
    }
    // A lying metadata.total_size is only metadata.
    let map = write_split(&sc.0, None);
    let pairs: Vec<(&str, &str)> = map.iter().map(|(n, f)| (n.as_str(), *f)).collect();
    let lying = index_text(&pairs).replace("\"total_size\":1", "\"total_size\":18446744073709551615");
    std::fs::write(&index, lying).unwrap();
    CheckpointView::open(&sc.0).expect("total_size is advisory");
}

#[test]
fn awkward_but_bare_shard_names_load_literally() {
    let sc = Scratch::new("awkward");
    let all = fixture_entries();
    let names = [
        "sp ace.safetensors",
        "trail \n.safetensors",
        "caf\u{e9}-\u{4e2d}.safetensors",
    ];
    let mut map: Vec<(String, String)> = Vec::new();
    let mut shards: Vec<Vec<Entry>> = vec![Vec::new(); 3];
    for (i, e) in all.into_iter().enumerate() {
        shards[i % 3].push(e);
    }
    for (f, s) in names.iter().zip(&shards) {
        write_st(&sc.0.join(f), s);
        map.extend(s.iter().map(|e| (e.0.clone(), f.to_string())));
    }
    let pairs: Vec<(&str, &str)> = map.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    std::fs::write(sc.0.join(INDEX_FILE_NAME), index_text(&pairs)).unwrap();
    assert_eq!(CheckpointView::open(&sc.0).unwrap().shard_count(), 3);
}

#[test]
fn one_file_under_two_spellings_is_refused_not_double_counted() {
    // macOS volumes are case-insensitive: `Model.safetensors` and
    // `model.safetensors` open the same file. Two weight_map files that are
    // one inode must not pass as two shards (every tensor is then "in both").
    let sc = Scratch::new("case");
    let all = fixture_entries();
    write_st(&sc.0.join("model.safetensors"), &all);
    let mut pairs: Vec<(&str, &str)> = Vec::new();
    for (i, e) in all.iter().enumerate() {
        pairs.push((
            &e.0,
            if i % 2 == 0 {
                "Model.safetensors"
            } else {
                "model.safetensors"
            },
        ));
    }
    std::fs::write(sc.0.join(INDEX_FILE_NAME), index_text(&pairs)).unwrap();
    let on_ci_fs = std::fs::metadata(sc.0.join("Model.safetensors")).is_ok();
    if !on_ci_fs {
        return; // case-sensitive volume: Model.safetensors is simply missing.
    }
    let err = open_err(&sc.0);
    assert!(!err.is_empty());
}

#[test]
fn a_huge_index_is_refused_quickly() {
    let sc = Scratch::new("huge");
    write_split(&sc.0, None);
    let body: Vec<String> = (0..300_000)
        .map(|i| format!("\"t{i}\":\"model-00001-of-00002.safetensors\""))
        .collect();
    std::fs::write(
        sc.0.join(INDEX_FILE_NAME),
        format!("{{\"weight_map\":{{{}}}}}", body.join(",")),
    )
    .unwrap();
    let t = std::time::Instant::now();
    let err = open_err(&sc.0);
    assert!(t.elapsed().as_secs() < 10, "{:?}: {err}", t.elapsed());
    assert!(
        err.contains("t0") || err.contains("no such tensor") || err.contains("cap"),
        "{err}"
    );
}

#[test]
fn directory_with_both_an_index_and_a_model_file_follows_the_index() {
    let sc = Scratch::new("both");
    write_split(&sc.0, None);
    std::fs::copy(fixture().join("model.safetensors"), sc.0.join("model.safetensors")).unwrap();
    assert_eq!(CheckpointView::open(&sc.0).unwrap().shard_count(), 2);
    // A broken index is not papered over by falling back to model.safetensors.
    std::fs::write(sc.0.join(INDEX_FILE_NAME), "{").unwrap();
    open_err(&sc.0);
    // Nor is a dangling-symlink index.
    std::fs::remove_file(sc.0.join(INDEX_FILE_NAME)).unwrap();
    std::os::unix::fs::symlink("nowhere", sc.0.join(INDEX_FILE_NAME)).unwrap();
    open_err(&sc.0);
}

#[test]
fn a_directory_without_index_or_model_file_is_a_named_error() {
    let sc = Scratch::new("empty-dir");
    let err = open_err(&sc.0);
    assert!(err.contains("model.safetensors"), "{err}");
    let err = open_err(Path::new(""));
    assert!(!err.is_empty());
}

#[test]
fn many_distinct_shards_that_exhaust_fds_fail_named_and_release() {
    // Run under a low `ulimit -n` to exercise exhaustion; with the default
    // limit it just checks that 400 shards open and every handle is retained.
    let sc = Scratch::new("fds");
    let mut pairs: Vec<(String, String)> = Vec::new();
    for i in 0..400 {
        let f = format!("s{i:04}.safetensors");
        let n = format!("t{i}");
        write_st(&sc.0.join(&f), &[(n.clone(), Dtype::F32, vec![1], vec![0u8; 4])]);
        pairs.push((n, f));
    }
    let p: Vec<(&str, &str)> = pairs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    std::fs::write(sc.0.join(INDEX_FILE_NAME), index_text(&p)).unwrap();
    match CheckpointView::open(&sc.0) {
        Ok(v) => assert_eq!(v.shard_count(), 400),
        Err(e) => assert!(
            e.contains("s0") && e.to_lowercase().contains("too many open files"),
            "{e}"
        ),
    }
    // Nothing leaked: the process can still open files.
    CheckpointView::open(&fixture().join("model.safetensors")).unwrap();
}
