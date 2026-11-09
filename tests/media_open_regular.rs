//! The media file readers open a path once and judge the opened handle, so a FIFO,
//! a device (or a symlink to one), a directory or an oversize file is refused
//! before any read, without blocking. No device: every case fails before a
//! framework is called, so this runs in the hosted job's sandbox.
//!
//! Each call runs on a thread behind a five second guard, so a regression that
//! blocks in `open` (a FIFO) fails the test instead of hanging the suite.

use std::fs::File;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use tessl::media::audio::{decode_audio_file, AudioOptions, MAX_AUDIO_FILE_BYTES};
use tessl::media::image::{decode_image_file, ImageOptions, MAX_IMAGE_FILE_BYTES};
use tessl::media::video::{VideoOptions, VideoSource};
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
    let dir = std::env::temp_dir().join(format!("tessl-media-open-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    Scratch(dir)
}

fn mkfifo(path: &Path) {
    let status = std::process::Command::new("mkfifo").arg(path).status().unwrap();
    assert!(status.success(), "mkfifo {}", path.display());
}

fn within_5s<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(5))
        .unwrap_or_else(|_| panic!("{what} did not return within 5 s (blocked on a non-regular file?)"))
}

/// The three readers behind one name each: image (`read_file_bounded`), audio
/// and video, each reduced to the error it returned.
type Reader = fn(PathBuf) -> Result<(), MediaError>;

fn readers() -> [(&'static str, Reader); 3] {
    [
        ("image", |p| decode_image_file(&p, &ImageOptions::default()).map(|_| ())),
        ("audio", |p| decode_audio_file(&p, &AudioOptions::default()).map(|_| ())),
        ("video", |p| VideoSource::open(&p, &VideoOptions::default()).map(|_| ())),
    ]
}

fn is_invalid(r: &Result<(), MediaError>) -> bool {
    matches!(r, Err(MediaError::Invalid { detail, .. }) if detail.ends_with("is not a file"))
}

#[test]
fn a_fifo_direct_or_through_a_symlink_is_refused_without_blocking() {
    let dir = scratch("fifo");
    let fifo = dir.join("f");
    mkfifo(&fifo);
    let link = dir.join("l");
    symlink(&fifo, &link).unwrap();
    for (name, read) in readers() {
        for p in [fifo.clone(), link.clone()] {
            let shown = format!("{name} {}", p.display());
            let r = within_5s(&shown, move || read(p));
            assert!(is_invalid(&r), "{shown}: {r:?}");
        }
    }
}

#[test]
fn a_symlink_to_dev_zero_is_refused_as_not_a_file() {
    let dir = scratch("zero");
    let link = dir.join("zero");
    symlink("/dev/zero", &link).unwrap();
    for (name, read) in readers() {
        let p = link.clone();
        let r = within_5s(name, move || read(p));
        assert!(is_invalid(&r), "{name}: {r:?}");
    }
}

#[test]
fn a_directory_is_refused_as_not_a_file() {
    let dir = scratch("dir");
    for (name, read) in readers() {
        let r = within_5s(name, {
            let d = dir.to_path_buf();
            move || read(d)
        });
        assert!(is_invalid(&r), "{name}: {r:?}");
    }
}

#[test]
fn an_empty_file_is_empty_and_a_missing_or_dangling_path_is_io() {
    let dir = scratch("empty");
    let empty = dir.join("e");
    File::create(&empty).unwrap();
    let dangling = dir.join("d");
    symlink(dir.join("nowhere"), &dangling).unwrap();
    for (name, read) in readers() {
        let r = within_5s(name, {
            let p = empty.clone();
            move || read(p)
        });
        assert!(matches!(r, Err(MediaError::Empty { .. })), "{name}: {r:?}");
        for p in [dir.join("absent"), dangling.clone()] {
            let shown = format!("{name} {}", p.display());
            let r = within_5s(&shown, move || read(p));
            assert!(
                matches!(&r, Err(MediaError::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound),
                "{shown}: {r:?}"
            );
        }
    }
}

#[test]
fn a_sparse_file_just_over_the_cap_is_refused_from_its_length_before_any_read() {
    let dir = scratch("sparse");
    let image = dir.join("big.png");
    File::create(&image).unwrap().set_len(MAX_IMAGE_FILE_BYTES + 1).unwrap();
    let r = within_5s("image", move || decode_image_file(&image, &ImageOptions::default()));
    assert!(
        matches!(&r, Err(MediaError::Limit { detail, .. }) if *detail == format!("{} bytes exceeds {MAX_IMAGE_FILE_BYTES}", MAX_IMAGE_FILE_BYTES + 1)),
        "{r:?}"
    );

    let audio = dir.join("big.wav");
    File::create(&audio).unwrap().set_len(MAX_AUDIO_FILE_BYTES + 1).unwrap();
    let r = within_5s("audio", move || decode_audio_file(&audio, &AudioOptions::default()));
    assert!(
        matches!(&r, Err(MediaError::Limit { detail, .. }) if *detail == format!("{} bytes exceeds {MAX_AUDIO_FILE_BYTES}", MAX_AUDIO_FILE_BYTES + 1)),
        "{r:?}"
    );
}
