//! The one way this crate opens a file named by a caller and reads it whole:
//! the file is opened once through [`ojas_io::open_regular_file`], judged by
//! `fstat` on the opened handle, and read through a byte cap.
//!
//! Deciding on a path by name (`exists()`, `metadata().is_file()`) and then
//! opening it by name again leaves two holes. A FIFO swapped in between blocks
//! the `open` forever, and a character device such as `/dev/zero` (or a symlink
//! to it) passes `exists()` and reads without end. [`open_regular`] refuses both
//! without blocking, and [`read_capped`] bounds the read at `cap + 1` bytes so a
//! file that grows after the open is caught rather than truncated.

use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Why a path was not opened or read.
#[derive(Debug)]
pub(crate) enum OpenFail {
    /// The path could not be opened or read (missing, permission, I/O).
    Io(std::io::Error),
    /// The path resolves to something that is not a regular file: a directory,
    /// FIFO, socket or device.
    NotRegular,
    /// The file is longer than the cap; `at_least` is set when the length came
    /// from the bounded read (the file grew after the open) rather than `fstat`.
    TooLarge { len: u64, at_least: bool },
}

/// Open `path` once and return the handle with its length from `fstat` on that
/// handle. A symlink is followed; a FIFO, directory or device is refused without
/// blocking or reading.
///
/// The failure is classified from the [`ojas_io::IoError`] alone: the path is
/// never stat-ed or opened a second time by name, so there is no window in which
/// it can be swapped between the refusal and its classification.
///
/// * `ojas_io` kept the OS `std::io::Error` as the error's source (the `stat`,
///   the open or the `fstat` failed: missing, ENOTDIR, EACCES, EMFILE, ...):
///   `OpenFail::Io` carries an `io::Error` rebuilt from that error's raw OS code
///   (`from_raw_os_error`), so the kind and the OS message text are exactly the
///   original's. A source without a raw code keeps its kind and text.
/// * `ojas_io` refused the file itself ("not a regular file": a directory,
///   FIFO, socket or device, on the `stat` or on the opened handle) and has no
///   source: `OpenFail::NotRegular`.
pub(crate) fn open_regular(path: &Path) -> Result<(File, u64), OpenFail> {
    ojas_io::open_regular_file(path).map_err(|e| match std::error::Error::source(&e) {
        Some(source) => OpenFail::Io(
            match source
                .downcast_ref::<std::io::Error>()
                .and_then(std::io::Error::raw_os_error)
            {
                Some(code) => std::io::Error::from_raw_os_error(code),
                None => std::io::Error::new(e.io_kind().unwrap_or(std::io::ErrorKind::Other), source.to_string()),
            },
        ),
        None => OpenFail::NotRegular,
    })
}

/// Read the regular file at `path`, refusing one longer than `cap` bytes before
/// any read (by the opened handle's length) and again after (by the bounded
/// read). An empty file returns an empty `Vec`; the caller decides what that
/// means.
pub(crate) fn read_capped(path: &Path, cap: u64) -> Result<Vec<u8>, OpenFail> {
    let (file, len) = open_regular(path)?;
    read_opened(file, len, cap)
}

/// [`read_capped`] after the open: `len` is what `fstat` said, `cap` the limit.
fn read_opened(file: File, len: u64, cap: u64) -> Result<Vec<u8>, OpenFail> {
    if len > cap {
        return Err(OpenFail::TooLarge { len, at_least: false });
    }
    // `len <= cap`, so this allocation is bounded by the cap.
    let mut out = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
    file.take(cap.saturating_add(1))
        .read_to_end(&mut out)
        .map_err(OpenFail::Io)?;
    if out.len() as u64 > cap {
        return Err(OpenFail::TooLarge {
            len: out.len() as u64,
            at_least: true,
        });
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::fs::symlink;
    use std::path::PathBuf;
    use std::sync::mpsc;
    use std::time::Duration;

    /// A fresh directory under the system temp dir, removed when dropped (a
    /// test's sparse files and FIFOs must not outlive it). Derefs to its path.
    pub(crate) struct Scratch(PathBuf);

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

    pub(crate) fn scratch(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("tessl-bounded-file-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    pub(crate) fn mkfifo(path: &Path) {
        let status = std::process::Command::new("mkfifo").arg(path).status().unwrap();
        assert!(status.success(), "mkfifo {}", path.display());
    }

    /// Run `f` on a thread and fail, rather than hang the suite, if it has not
    /// returned in five seconds. A hung thread is left behind (it is blocked in
    /// `open`, which only the pre-fix code does).
    pub(crate) fn within_5s<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx.recv_timeout(Duration::from_secs(5))
            .unwrap_or_else(|_| panic!("{what} did not return within 5 s (blocked on a non-regular file?)"))
    }

    #[test]
    fn a_fifo_direct_or_linked_is_refused_without_blocking() {
        let dir = scratch("fifo");
        let fifo = dir.join("f");
        mkfifo(&fifo);
        let link = dir.join("l");
        symlink(&fifo, &link).unwrap();
        for p in [fifo, link] {
            let shown = p.display().to_string();
            let r = within_5s(&shown, move || read_capped(&p, 1024));
            assert!(matches!(r, Err(OpenFail::NotRegular)), "{shown}: {r:?}");
        }
    }

    #[test]
    fn a_symlink_to_dev_zero_is_refused_and_never_read() {
        let dir = scratch("zero");
        let link = dir.join("zero");
        symlink("/dev/zero", &link).unwrap();
        let r = within_5s("dev/zero", move || read_capped(&link, u64::MAX / 2));
        assert!(matches!(r, Err(OpenFail::NotRegular)), "{r:?}");
    }

    #[test]
    fn a_directory_is_not_regular_and_a_missing_or_dangling_path_is_io_not_found() {
        let dir = scratch("dir");
        assert!(matches!(read_capped(&dir, 8), Err(OpenFail::NotRegular)));
        let dangling = dir.join("d");
        symlink(dir.join("nowhere"), &dangling).unwrap();
        for p in [dir.join("absent"), dangling] {
            match read_capped(&p, 8) {
                Err(OpenFail::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "{}", p.display()),
                other => panic!("{}: {other:?}", p.display()),
            }
        }
    }

    #[test]
    fn the_cap_is_judged_from_the_handle_before_any_read_and_is_inclusive() {
        let dir = scratch("cap");
        let sparse = dir.join("sparse");
        let f = File::create(&sparse).unwrap();
        f.set_len(1 << 40).unwrap(); // a terabyte of holes: reading it would not finish
        let r = within_5s("sparse", move || read_capped(&sparse, 1 << 20));
        assert!(
            matches!(r, Err(OpenFail::TooLarge { len, at_least: false }) if len == 1 << 40),
            "{r:?}"
        );

        let exact = dir.join("exact");
        std::fs::write(&exact, vec![7u8; 64]).unwrap();
        assert_eq!(read_capped(&exact, 64).unwrap().len(), 64);
        assert!(matches!(
            read_capped(&exact, 63),
            Err(OpenFail::TooLarge {
                len: 64,
                at_least: false
            })
        ));
    }

    #[test]
    fn an_empty_file_reads_as_empty_and_an_unreadable_one_is_io() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("empty");
        let empty = dir.join("e");
        File::create(&empty).unwrap();
        assert!(read_capped(&empty, 8).unwrap().is_empty());

        let locked = dir.join("locked");
        std::fs::write(&locked, b"x").unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        // Root reads anything, so only a non-root run can see the refusal.
        if std::fs::File::open(&locked).is_err() {
            assert!(matches!(read_capped(&locked, 8), Err(OpenFail::Io(_))));
        }
    }

    /// The read is bounded at `cap + 1` independently of the length `fstat`
    /// reported, so a file that grows after the open is refused, not truncated.
    #[test]
    fn a_file_that_grows_after_the_open_is_caught_by_the_bounded_read() {
        let dir = scratch("grow");
        let path = dir.join("g");
        let mut w = File::create(&path).unwrap();
        w.write_all(&[1u8; 8]).unwrap();
        let (file, len) = open_regular(&path).unwrap();
        assert_eq!(len, 8);
        w.write_all(&[1u8; 1024]).unwrap();
        let r = read_opened(file, len, 8);
        assert!(matches!(r, Err(OpenFail::TooLarge { len: 9, at_least: true })), "{r:?}");
    }

    #[test]
    fn cap_zero_and_cap_max_behave_at_the_edges() {
        let dir = scratch("capedge");
        let empty = dir.join("e");
        File::create(&empty).unwrap();
        assert!(read_capped(&empty, 0).unwrap().is_empty());
        let one = dir.join("o");
        std::fs::write(&one, b"x").unwrap();
        assert!(matches!(
            read_capped(&one, 0),
            Err(OpenFail::TooLarge {
                len: 1,
                at_least: false
            })
        ));
        assert_eq!(read_capped(&one, u64::MAX).unwrap(), b"x");
        // A file one past the cap whose fstat length was stale-small: the bounded read still stops at cap + 1.
        let (file, _) = open_regular(&one).unwrap();
        assert!(matches!(
            read_opened(file, 0, 0),
            Err(OpenFail::TooLarge { len: 1, at_least: true })
        ));
    }

    #[test]
    fn a_file_that_shrinks_after_the_open_reads_short_without_error() {
        let dir = scratch("shrink");
        let path = dir.join("s");
        std::fs::write(&path, [3u8; 64]).unwrap();
        let (file, len) = open_regular(&path).unwrap();
        File::options().write(true).open(&path).unwrap().set_len(10).unwrap();
        assert_eq!(read_opened(file, len, 64).unwrap().len(), 10);
    }

    #[test]
    fn an_unreadable_file_is_io_and_the_open_error_text_names_the_path_once() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("eacces-once");
        let locked = dir.join("locked");
        std::fs::write(&locked, b"x").unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        if File::open(&locked).is_ok() {
            return;
        }
        let Err(OpenFail::Io(e)) = read_capped(&locked, 8) else {
            panic!("not Io")
        };
        assert!(e.to_string().contains("ermission denied"), "{e}");
    }

    #[test]
    fn open_regular_keeps_permission_denied_kind() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("eacces-kind");
        let locked = dir.join("locked");
        std::fs::write(&locked, b"x").unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        if File::open(&locked).is_ok() {
            return;
        }
        let Err(OpenFail::Io(e)) = open_regular(&locked) else {
            panic!("not Io")
        };
        assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied, "{e}");
    }

    /// The production half of this file (before its tests) classifies from the
    /// ojas error alone: no second open or stat by name, no `O_NONBLOCK` probe.
    #[test]
    fn open_regular_never_re_probes_the_path() {
        let src = include_str!("bounded_file.rs");
        let prod = &src[..src
            .find(concat!("#[cfg(test)]", "\npub(crate) mod tests"))
            .expect("tests marker")];
        for banned in [
            "probe_open_error",
            "O_NONBLOCK",
            "OpenOptions",
            "File::open(",
            "fs::metadata(",
            "libc::",
        ] {
            assert!(
                !prod.contains(banned),
                "bounded_file.rs production code mentions `{banned}`"
            );
        }
        assert_eq!(prod.matches("ojas_io::open_regular_file(").count(), 1);
    }

    fn io_err(p: &Path) -> std::io::Error {
        match open_regular(p) {
            Err(OpenFail::Io(e)) => e,
            other => panic!("{}: expected Io, got {other:?}", p.display()),
        }
    }

    /// The kind and the exact OS text per failure, as `std` itself gives them.
    #[test]
    fn the_io_error_kind_and_text_are_the_operating_systems() {
        use std::io::ErrorKind::{NotADirectory, NotFound};
        let dir = scratch("kinds");
        let file = dir.join("f.txt");
        std::fs::write(&file, b"x").unwrap();
        let dangling = dir.join("d");
        symlink(dir.join("nowhere"), &dangling).unwrap();
        let cases = [
            (dir.join("absent"), NotFound),
            (dir.join("no-parent").join("x"), NotFound),
            (dangling, NotFound),
            (file.join("x"), NotADirectory),
        ];
        for (p, kind) in cases {
            let e = io_err(&p);
            assert_eq!(e.kind(), kind, "{}: {e}", p.display());
            let std_err = File::open(&p).unwrap_err();
            assert_eq!(e.raw_os_error(), std_err.raw_os_error(), "{}", p.display());
            assert_eq!(e.to_string(), std_err.to_string(), "{}", p.display());
        }
    }

    #[test]
    fn an_unsearchable_parent_is_permission_denied() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("eacces-parent");
        let sub = dir.join("sub");
        std::fs::create_dir(&sub).unwrap();
        let file = sub.join("f");
        std::fs::write(&file, b"x").unwrap();
        std::fs::set_permissions(&sub, std::fs::Permissions::from_mode(0o000)).unwrap();
        let skip = File::open(&file).is_ok(); // root searches anything
        let r = open_regular(&file);
        std::fs::set_permissions(&sub, std::fs::Permissions::from_mode(0o755)).unwrap();
        if skip {
            return;
        }
        let Err(OpenFail::Io(e)) = r else {
            panic!("not Io: {r:?}")
        };
        assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied, "{e}");
    }

    #[test]
    fn a_non_utf8_name_and_an_over_long_name_keep_their_os_errors() {
        use std::os::unix::ffi::OsStrExt;
        let dir = scratch("names");
        let odd = dir.join(std::ffi::OsStr::from_bytes(b"\xff\xfe-absent"));
        assert_eq!(io_err(&odd).kind(), std::io::ErrorKind::NotFound);
        let long = dir.join("n".repeat(300));
        let e = io_err(&long);
        assert_eq!(e.raw_os_error(), File::open(&long).unwrap_err().raw_os_error());
        assert_eq!(e.to_string(), File::open(&long).unwrap_err().to_string());
    }
}
