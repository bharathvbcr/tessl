//! The shipped library is the metallib's bytes, not a path into `OUT_DIR`.
//!
//! `build.rs` still writes the artifact and publishes its path (tooling and
//! `DEP_TESSL_METALLIB` read it). Opening a runtime must not. This copies the
//! test binary aside, renames the build artifact, and opens `GpuRuntime` from
//! the copy.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[test]
fn embedded_metallib_opens_after_the_build_artifact_is_renamed() {
    if std::env::var_os("TESSL_EMBED_PROBE").is_some() {
        let rt = tessl::GpuRuntime::new().expect("embedded metallib must open with the build artifact gone");
        rt.pipeline("qwen35_adamw_f32")
            .expect("the embedded library must contain qwen35_adamw_f32");
        return;
    }

    let path = PathBuf::from(tessl::metallib_path());
    assert!(
        path.is_file(),
        "build artifact missing at {} — the probe needs a real file to rename",
        path.display()
    );
    let hidden = hidden_path(&path);
    fs::rename(&path, &hidden).unwrap_or_else(|e| panic!("rename {} -> {}: {e}", path.display(), hidden.display()));
    let _restore = Restore {
        from: hidden.clone(),
        to: path.clone(),
    };

    let exe = std::env::current_exe().expect("current test binary");
    let scratch = std::env::temp_dir().join(format!("tessl-embed-probe-{}", std::process::id()));
    let _ = fs::remove_dir_all(&scratch);
    fs::create_dir_all(&scratch).expect("scratch dir");
    let copied = scratch.join("probe");
    fs::copy(&exe, &copied).unwrap_or_else(|e| panic!("copy {} -> {}: {e}", exe.display(), copied.display()));

    let out = Command::new(&copied)
        .arg("embedded_metallib_opens_after_the_build_artifact_is_renamed")
        .arg("--exact")
        .env("TESSL_EMBED_PROBE", "1")
        .output()
        .unwrap_or_else(|e| panic!("spawn probe: {e}"));
    assert!(
        out.status.success(),
        "runtime did not open after {} was renamed away\nstatus: {:?}\nstdout:\n{}\nstderr:\n{}",
        path.display(),
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = fs::remove_dir_all(&scratch);
}

fn hidden_path(path: &Path) -> PathBuf {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("metallib");
    path.with_file_name(format!("{name}.hidden-{}", std::process::id()))
}

struct Restore {
    from: PathBuf,
    to: PathBuf,
}

impl Drop for Restore {
    fn drop(&mut self) {
        if self.from.exists() {
            let _ = fs::rename(&self.from, &self.to);
        }
    }
}
