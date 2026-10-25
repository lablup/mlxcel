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

use std::fs::OpenOptions;
use std::io::{self, Write as _};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

#[cfg(unix)]
use super::write_video_temp_file;
use super::write_video_temp_file_for_test;

#[cfg(unix)]
#[tokio::test]
async fn native_video_temp_file_is_private_and_owned_by_directory_guard() {
    use std::os::unix::fs::PermissionsExt as _;

    let (path, guard) = write_video_temp_file(b"private video".to_vec(), "mp4")
        .await
        .expect("temp materialization succeeds");
    let directory = guard.path().to_path_buf();

    assert_eq!(
        std::fs::metadata(&directory).unwrap().permissions().mode() & 0o777,
        0o700
    );
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );

    drop(guard);
    assert!(
        !directory.exists(),
        "dropping the guard removes the directory"
    );
}

#[tokio::test]
async fn forced_write_error_removes_partial_private_directory() {
    let observed = Arc::new(Mutex::new(None::<PathBuf>));
    let observed_by_writer = Arc::clone(&observed);

    let result = write_video_temp_file_for_test(
        b"partial video".to_vec(),
        "mp4",
        move |dir, name, bytes| {
            *observed_by_writer.lock().unwrap() = Some(dir.path().to_path_buf());
            dir.write_file(name, &bytes[..4])?;
            Err(io::Error::other("forced write failure"))
        },
    )
    .await;

    assert!(result.is_none());
    let directory = observed.lock().unwrap().clone().unwrap();
    assert!(
        !directory.exists(),
        "a failed writer must drop the guard and remove partial output"
    );
}

#[tokio::test]
async fn dropped_resolver_future_cleans_up_after_blocking_writer_finishes() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();

    let resolver = tokio::spawn(async move {
        write_video_temp_file_for_test(
            b"video bytes written in two pieces".to_vec(),
            "mp4",
            move |dir, name, bytes| {
                let path = dir.path().join(name);
                let mut options = OpenOptions::new();
                options.write(true).create_new(true);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::OpenOptionsExt as _;
                    options.mode(0o600);
                }
                let mut file = options.open(&path)?;
                file.write_all(&bytes[..4])?;
                entered_tx.send(dir.path().to_path_buf()).unwrap();
                release_rx.recv().unwrap();
                file.write_all(&bytes[4..])?;
                finished_tx.send(()).unwrap();
                Ok(path)
            },
        )
        .await
    });

    let directory = tokio::task::spawn_blocking(move || entered_rx.recv().unwrap())
        .await
        .unwrap();
    assert!(
        directory.exists(),
        "writer must be paused with partial output"
    );

    resolver.abort();
    release_tx.send(()).unwrap();
    tokio::task::spawn_blocking(move || finished_rx.recv().unwrap())
        .await
        .unwrap();
    assert!(resolver.await.unwrap_err().is_cancelled());

    tokio::time::timeout(Duration::from_secs(2), async {
        while directory.exists() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("blocking writer output must be dropped after request cancellation");
    assert!(!directory.exists());
}
