// Copyright 2025-2026 Lablup Inc.
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

//! What `bind_unix` replaces at its path, and what it leaves alone.

use std::os::unix::fs::{FileTypeExt, MetadataExt, symlink};
use std::os::unix::net::{UnixDatagram, UnixListener as StdUnixListener, UnixStream};
use std::path::Path;

use super::bind_unix;

fn directory() -> tempfile::TempDir {
    // Keep paths below sun_path's platform limit, including on macOS.
    tempfile::Builder::new()
        .prefix("mlxcel-bind-")
        .tempdir_in("/tmp")
        .unwrap()
}

fn identity(path: &Path) -> (u64, u64, u32) {
    let metadata = std::fs::symlink_metadata(path).unwrap();
    (metadata.dev(), metadata.ino(), metadata.mode())
}

/// `bind_unix` must refuse `path` with `reason` in the error and leave the
/// entry exactly as it was.
fn assert_left_alone(path: &Path, reason: &str) {
    let before = identity(path);
    let error = bind_unix(path).expect_err("the entry must not be replaced");
    assert!(error.to_string().contains(reason), "{error}");
    assert_eq!(identity(path), before);
}

fn is_socket(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .unwrap()
        .file_type()
        .is_socket()
}

#[tokio::test]
async fn unix_bind_replaces_the_socket_an_earlier_run_left_behind() {
    // The server does not remove its socket on shutdown, so the next start on
    // the same path has to.
    let directory = directory();
    let path = directory.path().join("server.sock");
    drop(bind_unix(&path).unwrap());
    assert!(
        is_socket(&path),
        "a dropped listener leaves its socket file"
    );

    let listener = bind_unix(&path).expect("a restart on the same path");
    let _client = UnixStream::connect(&path).unwrap();
    let (_accepted, _) = listener.accept().await.unwrap();
}

#[tokio::test]
async fn unix_bind_replaces_a_socket_whose_owner_is_gone() {
    let directory = directory();
    let path = directory.path().join("stale.sock");
    drop(StdUnixListener::bind(&path).unwrap());
    assert!(is_socket(&path));

    // Reachability shows the replacement; an inode number may be reused.
    let listener = bind_unix(&path).unwrap();
    let _client = UnixStream::connect(&path).unwrap();
    let (_accepted, _) = listener.accept().await.unwrap();
}

#[tokio::test]
async fn unix_bind_leaves_a_live_listener_its_path() {
    let directory = directory();
    let path = directory.path().join("live.sock");
    let owner = StdUnixListener::bind(&path).unwrap();
    owner.set_nonblocking(true).unwrap();

    assert_left_alone(&path, "in use by a running listener");

    // The probe's own connection, then a client's: the owner still gets both.
    let _probe = owner.accept().expect("the probe reached the owner");
    let _client = UnixStream::connect(&path).unwrap();
    owner.accept().expect("the owner still owns the path");
}

// Linux answers "would block" for a full accept queue. macOS answers
// "connection refused", the same as for a stale socket, so a saturated
// listener cannot be recognised there and this case is Linux-only.
#[cfg(target_os = "linux")]
#[tokio::test]
async fn unix_bind_leaves_a_listener_with_a_full_backlog_its_path() {
    let directory = directory();
    let path = directory.path().join("busy.sock");
    let address = socket2::SockAddr::unix(&path).unwrap();
    let stream = || socket2::Socket::new(socket2::Domain::UNIX, socket2::Type::STREAM, None);
    let owner = stream().unwrap();
    owner.bind(&address).unwrap();
    owner.listen(1).unwrap();
    // Fill the accept queue: nothing is accepted while these stay open.
    let mut pending = Vec::new();
    for _ in 0..64 {
        let socket = stream().unwrap();
        socket.set_nonblocking(true).unwrap();
        match socket.connect(&address) {
            Ok(()) => pending.push(socket),
            Err(_) => break,
        }
    }
    assert!(pending.len() < 64, "the backlog never filled");

    assert_left_alone(&path, "in use by a running listener");
    drop(owner);
}

#[tokio::test]
async fn unix_bind_leaves_a_regular_file() {
    let directory = directory();
    let path = directory.path().join("file.sock");
    std::fs::write(&path, b"sentinel").unwrap();
    assert_left_alone(&path, "is not a socket");
    assert_eq!(std::fs::read(path).unwrap(), b"sentinel");
}

#[tokio::test]
async fn unix_bind_leaves_both_symlink_kinds() {
    let directory = directory();
    let target = directory.path().join("target");
    std::fs::write(&target, b"sentinel").unwrap();
    for (name, target) in [
        ("live.sock", target.clone()),
        ("dangling.sock", directory.path().join("absent")),
    ] {
        let path = directory.path().join(name);
        symlink(&target, &path).unwrap();
        assert_left_alone(&path, "is not a socket");
        assert_eq!(std::fs::read_link(path).unwrap(), target);
    }
    assert_eq!(std::fs::read(target).unwrap(), b"sentinel");
}

#[tokio::test]
async fn unix_bind_leaves_a_bound_datagram_socket() {
    // Another program's socket of another type: a stream connect is not
    // refused with "connection refused", so it is not taken for stale.
    let directory = directory();
    let path = directory.path().join("datagram.sock");
    let owner = UnixDatagram::bind(&path).unwrap();
    assert_left_alone(&path, "cannot tell whether the socket is still in use");
    drop(owner);
}

#[tokio::test]
async fn unix_bind_leaves_a_socket_it_may_not_connect_to() {
    use std::os::unix::fs::PermissionsExt;

    // A leftover socket this user has no write permission on: the probe is
    // denied, so nothing is known about its owner and it is not removed.
    let directory = directory();
    let path = directory.path().join("denied.sock");
    drop(StdUnixListener::bind(&path).unwrap());
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();
    match UnixStream::connect(&path) {
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {}
        // A privileged user is not denied; there is nothing to test then.
        _ => return,
    }
    assert_left_alone(&path, "cannot tell whether the socket is still in use");
}

#[tokio::test]
async fn unix_bind_creates_an_unused_nested_path() {
    let directory = directory();
    let path = directory.path().join("a/b/new.sock");
    let listener = bind_unix(&path).unwrap();
    let _client = UnixStream::connect(&path).unwrap();
    let (_accepted, _) = listener.accept().await.unwrap();
}
