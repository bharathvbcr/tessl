//! Source-text contract (no GPU): the runtime's relaxed-precision flag is read
//! in exactly one library place, `use_relaxed_f32`, which also demands
//! `PrecisionMode::F32` and TensorOps. Every "needs exact f32" refusal asks
//! `GpuRuntime::f32_gemm_runs_relaxed` instead of the raw flag, which is
//! ignored under `PrecisionMode::Bf16` and would over-refuse.

use std::fs;
use std::path::Path;

fn rust_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n != "bin") {
                rust_files(&path, out);
            }
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn relaxed_flag_is_read_only_by_the_shared_predicate() {
    let mut files = Vec::new();
    rust_files(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut files);
    let mut hits = Vec::new();
    for f in &files {
        let text = fs::read_to_string(f).unwrap();
        let name = f.file_name().unwrap().to_string_lossy().into_owned();
        for (i, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            if code.contains("relaxed_precision()")
                && !(name == "runtime.rs" && code.contains("pub fn relaxed_precision"))
            {
                hits.push(format!("{name}:{}: {}", i + 1, line.trim()));
            }
        }
    }
    // Only `use_relaxed_f32`'s own read in gemm.rs.
    assert!(
        hits.len() == 1 && hits[0].starts_with("gemm.rs:") && hits[0].contains("rt.relaxed_precision()"),
        "relaxed_precision() must be called only inside use_relaxed_f32, found: {hits:#?}"
    );
}
