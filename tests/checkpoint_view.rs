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
    assert!(err.contains("cannot be resolved"), "{err}");
}

#[test]
fn shard_paths_that_leave_the_checkpoint_directory_are_refused() {
    let sc = Scratch::new("escape");
    let dir = sc.0.join("ckpt");
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    let all = fixture_entries();
    let outside = sc.0.join("outside.safetensors");
    write_st(&outside, &all);
    write_st(&dir.join("inside.safetensors"), &all);
    let n = all[0].0.as_str();
    let write_index = |file: &str| std::fs::write(dir.join(INDEX_FILE_NAME), index_text(&[(n, file)])).unwrap();

    for hostile in [
        "../outside.safetensors".to_string(),
        outside.display().to_string(),
        "sub/../inside.safetensors".to_string(),
        "sub/inside.safetensors".to_string(),
        "./inside.safetensors".to_string(),
        String::new(),
    ] {
        write_index(&hostile);
        let err = open_err(&dir);
        assert!(err.contains("not a plain file name"), "{hostile:?}: {err}");
    }

    // A symlink inside the directory whose target is outside it.
    std::os::unix::fs::symlink(&outside, dir.join("link.safetensors")).unwrap();
    write_index("link.safetensors");
    let err = open_err(&dir);
    assert!(err.contains("outside the checkpoint directory"), "{err}");

    // A symlink to a file inside the directory is fine only if it is whole:
    // the index below names every tensor, so it opens.
    std::os::unix::fs::symlink(dir.join("inside.safetensors"), dir.join("alias.safetensors")).unwrap();
    let all_names: Vec<(&str, &str)> = all.iter().map(|e| (e.0.as_str(), "alias.safetensors")).collect();
    std::fs::write(dir.join(INDEX_FILE_NAME), index_text(&all_names)).unwrap();
    assert_eq!(CheckpointView::open(&dir).unwrap().shard_count(), 1);
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
