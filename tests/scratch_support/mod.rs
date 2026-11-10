//! A scratch directory (or scratch file) under the system temp dir that is
//! removed when the guard drops, on a pass and on a panic alike.
//!
//! Every name is unique per call (pid plus a per-process counter), so tests
//! running in parallel in one binary cannot collide. A path left by an earlier
//! killed run that had the same pid is removed before it is handed out, so a
//! stale leftover is never inherited. Hold the guard for as long as anything (a
//! child process, a stand-in script) still uses the path.
//!
//! Used by the test binaries through `mod scratch_support;`. Not every binary
//! uses every item.
#![allow(dead_code)]

use std::ffi::OsStr;
use std::io::ErrorKind;
use std::ops::Deref;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

pub struct Scratch(PathBuf);

impl Scratch {
    fn unique(prefix: &str, suffix: &str) -> PathBuf {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("{prefix}-{}-{n}{suffix}", std::process::id()))
    }

    /// Create an empty `$TMPDIR/{prefix}-{pid}-{n}` (and any missing parents of it).
    pub fn dir(prefix: &str) -> Scratch {
        Self::dir_at(Self::unique(prefix, ""))
    }

    /// Reserve `$TMPDIR/{prefix}-{pid}-{n}{suffix}` without creating it; the
    /// caller writes it. Whatever is there at drop (file or directory) goes.
    pub fn file(prefix: &str, suffix: &str) -> Scratch {
        Self::file_at(Self::unique(prefix, suffix))
    }

    /// An empty directory at exactly `d`: anything already there is removed first.
    pub fn dir_at(d: PathBuf) -> Scratch {
        remove_path(&d);
        std::fs::create_dir_all(&d).unwrap_or_else(|e| panic!("create scratch dir {}: {e}", d.display()));
        Scratch(d)
    }

    /// A free path at exactly `p`, not created: anything already there is removed first.
    pub fn file_at(p: PathBuf) -> Scratch {
        remove_path(&p);
        Scratch(p)
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

fn remove_once(p: &Path) -> std::io::Result<()> {
    match std::fs::symlink_metadata(p) {
        Ok(m) if m.is_dir() => std::fs::remove_dir_all(p),
        Ok(_) => std::fs::remove_file(p),
        Err(e) => Err(e),
    }
}

/// Give every directory under `p` owner rwx so it can be read and emptied.
/// Symlinks are never followed (`symlink_metadata`), so nothing outside the
/// tree is touched. Failures are ignored here: the retry reports what is left.
fn make_traversable(p: &Path) {
    match std::fs::symlink_metadata(p) {
        Ok(m) if m.is_dir() => {
            let _ = std::fs::set_permissions(p, std::fs::Permissions::from_mode(m.permissions().mode() | 0o700));
            if let Ok(rd) = std::fs::read_dir(p) {
                for e in rd.flatten() {
                    make_traversable(&e.path());
                }
            }
        }
        _ => {}
    }
}

/// Remove `p` whether it is a directory tree, a file or a symlink. A path that
/// is already gone is fine. A tree holding a directory the owner cannot enter
/// (mode 000) is made traversable and removed once more. Any other failure is
/// reported, never panicked on: this runs inside `Drop`, possibly during an
/// unwind.
pub fn remove_path(p: &Path) {
    let mut r = remove_once(p);
    if matches!(&r, Err(e) if e.kind() == ErrorKind::PermissionDenied) {
        make_traversable(p);
        r = remove_once(p);
    }
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
