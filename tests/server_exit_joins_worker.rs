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

//! `mlxcel-server` must join its model workers before `main` returns
//! (issue #2738).
//!
//! A worker left to free its model while the process runs its exit handlers
//! calls `allocator().free()` on an MLX singleton whose destructor already
//! ran, and the CPU backend aborts with `pure virtual method called`. Both
//! tests here drive a real server process into the exit path that used to
//! leave a worker behind: the listener fails to bind (the test holds the port)
//! right after the model began loading, so the process has to exit through
//! the join. A clean run exits with code 1 (the bind error), not a signal,
//! and logs the join.
//!
//! The tests skip when the `qwen3-0.6b-4bit` checkpoint is absent, like the
//! other real-model integration tests.

#![cfg(unix)]

mod common;

use std::io::Read;
use std::net::TcpListener;
use std::os::unix::process::ExitStatusExt;
use std::path::Path;
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use common::{repo_binary_path, repo_model_dir};

const MODEL: &str = "qwen3-0.6b-4bit";

/// Generous bound for "load a 0.6B checkpoint, fail to bind, join the
/// worker": the join itself is bounded by `WORKER_EXIT_TIMEOUT` (120 s).
const EXIT_DEADLINE: Duration = Duration::from_secs(300);

struct ChildOutcome {
    status: ExitStatus,
    stdout: String,
    stderr: String,
}

/// Run `mlxcel-server` with `args` to completion, collecting both streams.
fn run_server_until_exit(args: &[&str]) -> ChildOutcome {
    let mut child = Command::new(repo_binary_path("mlxcel-server"))
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn mlxcel-server");
    let mut stdout = child.stdout.take().expect("piped stdout");
    let mut stderr = child.stderr.take().expect("piped stderr");
    let stdout_reader = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = stdout.read_to_string(&mut text);
        text
    });
    let stderr_reader = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = stderr.read_to_string(&mut text);
        text
    });
    let deadline = Instant::now() + EXIT_DEADLINE;
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll mlxcel-server") {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("mlxcel-server did not exit within {EXIT_DEADLINE:?}");
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    ChildOutcome {
        status,
        stdout: stdout_reader.join().expect("stdout reader"),
        stderr: stderr_reader.join().expect("stderr reader"),
    }
}

/// The process left through the bind error (exit code 1) after joining its
/// workers: no signal, no C++ runtime abort text, and the join's log line.
fn assert_exited_through_the_join(outcome: &ChildOutcome, join_log_line: &str) {
    let logs = format!(
        "--- stdout ---\n{}\n--- stderr ---\n{}",
        outcome.stdout, outcome.stderr
    );
    assert!(
        outcome.status.signal().is_none(),
        "mlxcel-server died of signal {:?} instead of exiting\n{logs}",
        outcome.status.signal()
    );
    assert_eq!(
        outcome.status.code(),
        Some(1),
        "mlxcel-server should exit with the bind error\n{logs}"
    );
    assert!(
        !logs.contains("pure virtual method called") && !logs.contains("terminate called"),
        "C++ runtime abort text in the server output\n{logs}"
    );
    assert!(
        logs.contains(join_log_line),
        "missing {join_log_line:?} in the server output\n{logs}"
    );
}

fn skip_without_model() -> Option<std::path::PathBuf> {
    let model_dir = repo_model_dir(MODEL);
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return None;
    }
    Some(model_dir)
}

/// Hold a loopback port so the server's own bind fails after the model
/// started loading. The listener stays alive for the server's lifetime.
fn held_port() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind a loopback port to hold");
    let port = listener.local_addr().expect("local addr").port();
    (listener, port.to_string())
}

fn path_arg(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[test]
fn single_model_server_joins_its_worker_when_the_listener_fails() {
    let Some(model_dir) = skip_without_model() else {
        return;
    };
    let (_held, port) = held_port();
    let model_arg = path_arg(&model_dir);
    let outcome = run_server_until_exit(&[
        "-m",
        &model_arg,
        "--host",
        "127.0.0.1",
        "--port",
        &port,
        "--no-warmup",
    ]);
    assert_exited_through_the_join(&outcome, "model worker exited before process exit");
}

#[test]
fn router_server_drains_a_loading_model_when_the_listener_fails() {
    let Some(model_dir) = skip_without_model() else {
        return;
    };
    let preset_dir = tempfile::tempdir().expect("temp dir for the preset");
    let preset_path = preset_dir.path().join("models.ini");
    // `load-on-startup` begins the load before the listener binds, so the
    // bind error finds a worker that is still loading.
    std::fs::write(
        &preset_path,
        format!(
            "[eager]\nmodel = {}\nload-on-startup = 1\n",
            path_arg(&model_dir)
        ),
    )
    .expect("write preset");
    let (_held, port) = held_port();
    let preset_arg = path_arg(&preset_path);
    let outcome = run_server_until_exit(&[
        "--models-preset",
        &preset_arg,
        "--host",
        "127.0.0.1",
        "--port",
        &port,
        "--no-warmup",
    ]);
    assert_exited_through_the_join(&outcome, "router: lifecycle shutdown completed");
}
