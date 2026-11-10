//! A scratch directory (or scratch file) under the system temp dir that is
//! removed when the guard drops, on a pass and on a panic alike.
//!
//! Every name is unique per call (pid plus a per-process counter), so tests
//! running in parallel in one binary cannot collide and no stale same-pid
//! leftover can be inherited. Hold the guard for as long as anything (a child
//! process, a stand-in script) still uses the path.
//!
//! Used by the test binaries through `mod scratch_support;`. Not every binary
//! uses every item.
#![allow(dead_code)]

use std::ffi::OsStr;
use std::io::ErrorKind;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

pub struct Scratch(PathBuf);

impl Scratch {
    fn unique(prefix: &str, suffix: &str) -> PathBuf {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("{prefix}-{}-{n}{suffix}", std::process::id()))
    }

    /// Create `$TMPDIR/{prefix}-{pid}-{n}` (and any missing parents of it).
    pub fn dir(prefix: &str) -> Scratch {
        let d = Self::unique(prefix, "");
        std::fs::create_dir_all(&d).unwrap_or_else(|e| panic!("create scratch dir {}: {e}", d.display()));
        Scratch(d)
    }

    /// Reserve `$TMPDIR/{prefix}-{pid}-{n}{suffix}` without creating it; the
    /// caller writes it. Whatever is there at drop (file or directory) goes.
    pub fn file(prefix: &str, suffix: &str) -> Scratch {
        Scratch(Self::unique(prefix, suffix))
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Deref for Scratch {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for Scratch {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<OsStr> for Scratch {
    fn as_ref(&self) -> &OsStr {
        self.0.as_os_str()
    }
}

/// Remove `p` whether it is a directory tree, a file or a symlink. A path that
/// is already gone is fine. Any other failure is reported, never panicked on:
/// this runs inside `Drop`, possibly during an unwind.
pub fn remove_path(p: &Path) {
    let r = match std::fs::symlink_metadata(p) {
        Ok(m) if m.is_dir() => std::fs::remove_dir_all(p),
        Ok(_) => std::fs::remove_file(p),
        Err(e) => Err(e),
    };
    if let Err(e) = r {
        if e.kind() != ErrorKind::NotFound {
            eprintln!("scratch_support: could not remove {}: {e}", p.display());
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        remove_path(&self.0);
    }
}
