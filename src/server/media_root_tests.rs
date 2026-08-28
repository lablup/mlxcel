// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Containment tests for the `--media-path` root (issue #1451).
//!
//! Every case here is an escape attempt or the legitimate access it must not
//! break. The resolver is exercised through [`resolve_media_file_in`], which
//! takes the root explicitly, so no test installs the process-wide `OnceLock`
//! and the cases can run in any order.

use super::*;

/// A media root holding `ok.png` and a `sub/` directory with `nested.png`.
struct Fixture {
    _dir: tempfile::TempDir,
    root: PathBuf,
    outside: PathBuf,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    // The tempdir itself is the parent; the root is a directory inside it so
    // there is somewhere outside the root but on the same filesystem to escape
    // to.
    let root = dir.path().join("media");
    std::fs::create_dir(&root).expect("create root");
    std::fs::write(root.join("ok.png"), b"PNG-BYTES").expect("write ok.png");
    std::fs::create_dir(root.join("sub")).expect("create sub");
    std::fs::write(root.join("sub/nested.png"), b"NESTED").expect("write nested");
    let outside = dir.path().join("secret.txt");
    std::fs::write(&outside, b"SECRET").expect("write secret");
    Fixture {
        root: std::fs::canonicalize(&root).expect("canonical root"),
        outside: std::fs::canonicalize(&outside).expect("canonical secret"),
        _dir: dir,
    }
}

#[tokio::test]
async fn a_relative_file_url_resolves_inside_the_root() {
    let f = fixture();
    let resolved = resolve_media_file_in(&f.root, "file://ok.png")
        .await
        .expect("a plain relative reference resolves");
    assert_eq!(resolved, f.root.join("ok.png"));

    let nested = resolve_media_file_in(&f.root, "file://sub/nested.png")
        .await
        .expect("subdirectories are allowed, as upstream allows them");
    assert_eq!(nested, f.root.join("sub/nested.png"));
}

#[tokio::test]
async fn a_bare_relative_path_resolves_the_same_way() {
    let f = fixture();
    let resolved = resolve_media_file_in(&f.root, "ok.png")
        .await
        .expect("a bare relative path resolves");
    assert_eq!(resolved, f.root.join("ok.png"));
}

#[tokio::test]
async fn an_absolute_looking_path_is_concatenated_not_joined() {
    // b10621 does `media_path + file_path`, so `file:///etc/passwd` lands at
    // `<root>//etc/passwd` and reads nothing. A `Path::join` would have
    // replaced the root with `/etc/passwd`, which is the arbitrary-file read
    // this test exists to keep out.
    let f = fixture();
    let err = resolve_media_file_in(&f.root, "file:///etc/passwd")
        .await
        .expect_err("an absolute path may not escape");
    assert!(
        matches!(err, MediaPathError::Unresolvable { .. }),
        "expected an unresolvable path under the root, got {err:?}"
    );

    // The same absolute-looking form *inside* the root still resolves, which is
    // what proves the leading separator was stripped rather than refused.
    let resolved = resolve_media_file_in(&f.root, "file:///ok.png")
        .await
        .expect("a leading separator is stripped, not rejected");
    assert_eq!(resolved, f.root.join("ok.png"));
}

#[tokio::test]
async fn dot_dot_traversal_is_refused() {
    let f = fixture();
    for attempt in [
        "file://../secret.txt",
        "file://sub/../../secret.txt",
        "file://..%2fsecret.txt",
        "file://....//secret.txt",
        "../secret.txt",
    ] {
        assert!(
            matches!(
                resolve_media_file_in(&f.root, attempt).await,
                Err(MediaPathError::NotAllowed { .. })
            ),
            "{attempt} must be refused by the name validation"
        );
    }
}

#[tokio::test]
async fn percent_encoded_traversal_is_refused() {
    let f = fixture();
    for attempt in [
        "file://%2e%2e/secret.txt",
        "file://%2E%2E%2Fsecret.txt",
        "file://sub%2f%2e%2e%2fsecret.txt",
        "file://ok%00.png",
        "file://sub%5c..%5csecret.txt",
    ] {
        assert!(
            matches!(
                resolve_media_file_in(&f.root, attempt).await,
                Err(MediaPathError::NotAllowed { .. })
            ),
            "{attempt} must be refused"
        );
    }
}

#[cfg(unix)]
#[tokio::test]
async fn a_symlink_pointing_outside_the_root_is_refused() {
    let f = fixture();
    std::os::unix::fs::symlink(&f.outside, f.root.join("escape.png")).expect("symlink");
    let err = resolve_media_file_in(&f.root, "file://escape.png")
        .await
        .expect_err("a symlink target outside the root is refused");
    assert!(
        matches!(err, MediaPathError::Escape { .. }),
        "expected an escape, got {err:?}"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn a_symlinked_directory_component_cannot_escape_either() {
    let f = fixture();
    let parent = f.root.parent().expect("root has a parent").to_path_buf();
    std::os::unix::fs::symlink(&parent, f.root.join("up")).expect("symlink");
    let err = resolve_media_file_in(&f.root, "file://up/secret.txt")
        .await
        .expect_err("an intermediate symlink is followed and then caught");
    assert!(
        matches!(err, MediaPathError::Escape { .. }),
        "expected an escape, got {err:?}"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn a_symlink_that_stays_inside_the_root_still_resolves() {
    // Confinement is about where the target lands, not about symlinks as such;
    // an operator who organises the media root with links keeps working.
    let f = fixture();
    std::os::unix::fs::symlink(f.root.join("ok.png"), f.root.join("alias.png")).expect("symlink");
    let resolved = resolve_media_file_in(&f.root, "file://alias.png")
        .await
        .expect("an internal symlink resolves");
    assert_eq!(resolved, f.root.join("ok.png"));
}

#[cfg(unix)]
#[tokio::test]
async fn open_confined_refuses_a_symlink_swapped_in_after_resolution() {
    // `resolve_media_file_in` canonicalizes; `open_confined` then opens with
    // `O_NOFOLLOW`, so a last-component swap between the two fails loudly
    // instead of reading the new target.
    let f = fixture();
    let victim = f.root.join("swapped.png");
    std::fs::write(&victim, b"REAL").expect("write");
    let resolved = resolve_media_file_in(&f.root, "file://swapped.png")
        .await
        .expect("resolves before the swap");
    std::fs::remove_file(&victim).expect("remove");
    std::os::unix::fs::symlink(&f.outside, &victim).expect("symlink swap");
    assert!(
        open_confined(&resolved).await.is_err(),
        "O_NOFOLLOW must refuse the swapped-in symlink"
    );
}

#[tokio::test]
async fn a_directory_is_not_a_media_file() {
    let f = fixture();
    let err = resolve_media_file_in(&f.root, "file://sub")
        .await
        .expect_err("a directory is refused");
    assert!(
        matches!(err, MediaPathError::NotRegular { .. }),
        "expected a non-regular file, got {err:?}"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn a_fifo_inside_the_root_is_refused_before_it_can_block() {
    // Reading a FIFO would hang the request task until a writer appears.
    let f = fixture();
    let fifo = f.root.join("pipe");
    let c_path = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).expect("cstring");
    // SAFETY: `c_path` is a valid NUL-terminated path inside a fresh tempdir,
    // and `mkfifo` only creates a filesystem entry at it.
    let rc = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
    assert_eq!(rc, 0, "mkfifo failed: {}", std::io::Error::last_os_error());
    let err = resolve_media_file_in(&f.root, "file://pipe")
        .await
        .expect_err("a FIFO is refused");
    assert!(
        matches!(err, MediaPathError::NotRegular { .. }),
        "expected a non-regular file, got {err:?}"
    );
}

#[tokio::test]
async fn a_missing_file_reports_upstreams_wording() {
    let f = fixture();
    let err = resolve_media_file_in(&f.root, "file://absent.png")
        .await
        .expect_err("a missing file is refused");
    assert!(
        err.to_string()
            .contains("file does not exist or cannot be opened"),
        "{err}"
    );
}

#[test]
fn with_no_root_configured_local_files_report_upstreams_own_sentence() {
    // The wrapper's `NoRoot` arm is what an operator who never passed
    // `--media-path` sees, and clients may match on the wording, so it is
    // pinned here character for character. The arm is asserted through the
    // error value rather than by calling the wrapper, because other tests in
    // this binary install the process-wide root.
    assert_eq!(MediaPathError::NoRoot.to_string(), NO_MEDIA_ROOT_MESSAGE);
    assert_eq!(
        NO_MEDIA_ROOT_MESSAGE, "file:// URLs are not allowed unless --media-path is specified",
        "the wording is b10621's own and clients may match on it"
    );
}

#[tokio::test]
async fn the_installed_root_is_what_the_wrapper_resolves_against() {
    let root = install_test_root_once();
    std::fs::write(root.join("installed.png"), b"INSTALLED").expect("write");
    let resolved = resolve_media_file("file://installed.png")
        .await
        .expect("the installed root resolves");
    assert_eq!(resolved, root.join("installed.png"));
    assert!(
        resolve_media_file("file://../escape.png").await.is_err(),
        "the wrapper applies the same containment rules as the explicit-root form"
    );
}

#[test]
fn filename_validation_matches_upstreams_rule_set() {
    // Accepted by b10621's `fs_validate_filename(path, allow_subdirs=true)`.
    for ok in [
        "a.png",
        "sub/dir/a.png",
        "a b.png",
        "ünïcödé.png",
        "a%20b.png",
    ] {
        assert!(validate_media_filename(ok).is_ok(), "{ok} must be accepted");
    }
    // Refused by it.
    for bad in [
        "",
        ".",
        "..",
        "a/../b",
        "a.png ",
        " a.png",
        "a.png.",
        "a:b.png",
        "a*b.png",
        "a?b.png",
        "a\"b.png",
        "a<b.png",
        "a>b.png",
        "a|b.png",
        "a\u{7F}.png",
        "a\u{FF0E}\u{FF0E}/b.png",
        "a\u{2215}b.png",
        "a\u{2216}b.png",
        "a\u{FFFD}.png",
        "a\u{FEFF}.png",
    ] {
        assert!(
            validate_media_filename(bad).is_err(),
            "{bad:?} must be refused"
        );
    }
    assert!(validate_media_filename("a\n.png").is_err());
    assert!(validate_media_filename(&"a".repeat(256)).is_err());
    assert!(validate_media_filename(&"a".repeat(255)).is_ok());
}
