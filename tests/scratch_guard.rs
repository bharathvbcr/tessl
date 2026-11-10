//! The scratch guard of `tests/scratch_support`: a stale path is never
//! inherited, a drop removes a tree it cannot enter, a symlink is unlinked and
//! not followed. Device-free.

mod scratch_support;
use scratch_support::Scratch;
use std::os::unix::fs::PermissionsExt;

/// A killed earlier run with the same pid and counter left this directory
/// behind; the guard must hand out an empty one.
#[test]
fn a_stale_dir_at_the_scratch_path_is_not_inherited() {
    let stale = std::env::temp_dir().join(format!("tessl-scratch-guard-stale-{}", std::process::id()));
    std::fs::create_dir_all(stale.join("sub")).unwrap();
    std::fs::write(stale.join("leftover"), "from a dead run").unwrap();
    let s = Scratch::dir_at(stale.clone());
    assert_eq!(s.path(), stale.as_path());
    assert_eq!(std::fs::read_dir(&s).unwrap().count(), 0, "the old contents were inherited");
}

/// `Scratch::file` promises a path nothing is at.
#[test]
fn a_stale_file_at_a_reserved_path_is_removed() {
    let stale = std::env::temp_dir().join(format!("tessl-scratch-guard-stale-file-{}.bin", std::process::id()));
    std::fs::write(&stale, "stale").unwrap();
    let s = Scratch::file_at(stale.clone());
    assert!(!s.exists(), "the reserved path already existed");
}

/// Running as root ignores mode bits, so the case cannot fail there.
fn is_root() -> bool {
    std::process::Command::new("id")
        .arg("-u")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "0")
        .unwrap_or(false)
}

#[test]
fn a_tree_with_a_mode_000_subdir_is_gone_after_drop() {
    if is_root() {
        eprintln!("skipped: running as root, mode 000 does not stop root");
        return;
    }
    let p;
    {
        let s = Scratch::dir("tessl-scratch-guard-locked");
        p = s.path().to_path_buf();
        let sub = s.join("locked");
        std::fs::create_dir_all(sub.join("inner")).unwrap();
        std::fs::write(sub.join("inner/f"), "x").unwrap();
        std::fs::set_permissions(sub.join("inner"), std::fs::Permissions::from_mode(0o000)).unwrap();
        std::fs::set_permissions(&sub, std::fs::Permissions::from_mode(0o000)).unwrap();
    }
    if p.exists() {
        // Let the harness clean up before failing.
        let _ = std::fs::set_permissions(p.join("locked"), std::fs::Permissions::from_mode(0o755));
        let _ = std::fs::set_permissions(p.join("locked/inner"), std::fs::Permissions::from_mode(0o755));
        let _ = std::fs::remove_dir_all(&p);
        panic!("a mode-000 subdir left the scratch tree behind");
    }
}

/// The permission repair walks with `symlink_metadata`: a link out of the tree
/// is unlinked and its target is untouched.
#[test]
fn a_symlink_out_of_the_tree_is_unlinked_not_followed() {
    let outside = Scratch::dir("tessl-scratch-guard-outside");
    std::fs::write(outside.join("keep"), "x").unwrap();
    std::fs::set_permissions(&outside, std::fs::Permissions::from_mode(0o700)).unwrap();
    let p;
    {
        let s = Scratch::dir("tessl-scratch-guard-link");
        p = s.path().to_path_buf();
        std::os::unix::fs::symlink(outside.path(), s.join("link")).unwrap();
    }
    assert!(!p.exists());
    assert!(outside.join("keep").exists(), "the drop followed the symlink");
}

/// A scratch file path that turns into a directory tree is removed whole.
#[test]
fn a_reserved_path_that_became_a_tree_is_removed() {
    let p;
    {
        let s = Scratch::file("tessl-scratch-guard-tree", "");
        p = s.path().to_path_buf();
        std::fs::create_dir_all(p.join("a/b")).unwrap();
        std::fs::write(p.join("a/b/c"), "x").unwrap();
    }
    assert!(!p.exists());
}
