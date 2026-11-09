//! Wave-5 adversary attacks on the media file readers' open path (card
//! ft-2f75f232). Device-free. The former `defect_*` pins (error kind, ENOTDIR text) are
//! ordinary tests now that the readers keep the OS error.

use std::ffi::OsStr;
use std::fs::File;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};

use tessl::media::image::{decode_image_file, ImageOptions};
use tessl::media::MediaError;

/// A fresh directory under the system temp dir, removed when dropped (a test's
/// sparse files and FIFOs must not outlive it). Derefs to the directory's path.
struct Scratch(PathBuf);

impl std::ops::Deref for Scratch {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch(tag: &str) -> Scratch {
    let dir = std::env::temp_dir().join(format!("tessl-adv-readers-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    Scratch(dir)
}

fn decode(p: &Path) -> Result<(), MediaError> {
    decode_image_file(p, &ImageOptions::default()).map(|_| ())
}

fn running_as_root() -> bool {
    // Root reads a mode-000 file, so the permission attacks cannot run.
    std::fs::File::open("/etc/master.passwd").is_ok() || std::fs::read_dir("/private/var/root").is_ok()
}

fn unreadable(dir: &Path) -> Option<PathBuf> {
    let p = dir.join("locked.png");
    std::fs::write(&p, b"not an image").unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o000)).unwrap();
    if File::open(&p).is_ok() {
        return None;
    }
    Some(p)
}

#[test]
fn enotdir_eloop_enametoolong_and_nul_paths_are_io_errors_that_name_the_cause() {
    let dir = scratch("errno");
    let file = dir.join("f");
    std::fs::write(&file, b"x").unwrap();
    let under_file = file.join("child");
    match decode(&under_file) {
        Err(MediaError::Io { source, .. }) => assert_eq!(source.raw_os_error(), Some(20), "ENOTDIR: {source}"),
        other => panic!("{other:?}"),
    }
    let a = dir.join("a");
    let b = dir.join("b");
    symlink(&b, &a).unwrap();
    symlink(&a, &b).unwrap();
    match decode(&a) {
        Err(MediaError::Io { source, .. }) => assert_eq!(source.raw_os_error(), Some(62), "ELOOP: {source}"),
        other => panic!("{other:?}"),
    }
    let long = dir.join("x".repeat(300));
    match decode(&long) {
        Err(MediaError::Io { source, .. }) => assert_eq!(source.raw_os_error(), Some(63), "ENAMETOOLONG: {source}"),
        other => panic!("{other:?}"),
    }
    let nul = Path::new(OsStr::from_bytes(b"/tmp/a\0b"));
    match decode(nul) {
        Err(MediaError::Io { source, .. }) => assert_eq!(source.kind(), std::io::ErrorKind::InvalidInput),
        other => panic!("{other:?}"),
    }
    let non_utf8 = dir.join(OsStr::from_bytes(b"\xff\xfe"));
    assert!(matches!(decode(&non_utf8), Err(MediaError::Io { .. })));
}

#[test]
fn a_regular_file_behind_a_symlink_chain_and_a_dot_dot_path_still_reads() {
    let dir = scratch("chain");
    let real = dir.join("real.bin");
    std::fs::write(&real, b"not an image").unwrap();
    let l1 = dir.join("l1");
    let l2 = dir.join("l2");
    symlink(&real, &l1).unwrap();
    symlink(&l1, &l2).unwrap();
    // It opens and reads; the decode (not the open) then rejects the bytes.
    let r = decode(&l2);
    assert!(
        !matches!(r, Err(MediaError::Io { .. } | MediaError::Empty { .. })),
        "{r:?}"
    );
}

#[test]
fn an_unreadable_file_is_an_io_error_naming_the_path() {
    if running_as_root() {
        return;
    }
    let dir = scratch("eacces-text");
    let Some(p) = unreadable(&dir) else { return };
    match decode(&p) {
        Err(MediaError::Io { path, .. }) => assert_eq!(path, p),
        other => panic!("{other:?}"),
    }
}

/// `ojas_io::open_regular_file` reports every failure as text, and
/// `bounded_file::open_regular` rebuilds a final-component open failure as
/// `io::Error::other(text)` when `metadata` then succeeds. EACCES on the open
/// therefore arrives as `ErrorKind::Other`, where `std::fs::File::open` (the
/// pre-fix code) gave `PermissionDenied`, and the source text repeats the path.
#[test]
fn permission_denied_keeps_its_error_kind() {
    if running_as_root() {
        return;
    }
    let dir = scratch("eacces-kind");
    let Some(p) = unreadable(&dir) else { return };
    match decode(&p) {
        Err(MediaError::Io { source, .. }) => {
            assert_eq!(source.kind(), std::io::ErrorKind::PermissionDenied, "{source}")
        }
        other => panic!("{other:?}"),
    }
}
