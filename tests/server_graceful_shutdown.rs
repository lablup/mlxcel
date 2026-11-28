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

//! SIGTERM and SIGINT stop the server through a bounded graceful drain
//! (issue #3483).
//!
//! Each test drives a real `mlxcel-server` (or `mlxcel serve`) process: it
//! waits for `/health`, sends the signal, and checks the exit status, the
//! timing, and the log lines. Before #3483 no handler existed, so outside a
//! container the signal killed the process (`signal()` is `Some`) and as PID 1
//! it was ignored (the process outlived the deadline); both fail here.
//!
//! The tests skip when neither `qwen3-0.6b-4bit` nor `qwen3-1.7b-4bit` is
//! present, like the other real-model integration tests.

#![cfg(unix)]

mod common;

use std::io::{Read, Write};
use std::net::TcpStream;
use std::os::unix::net::UnixStream;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use common::{repo_binary_path, repo_model_dir_any};

/// Any small Qwen3 text checkpoint; the first one found is used.
const MODELS: &[&str] = &["qwen3-0.6b-4bit", "qwen3-1.7b-4bit"];
/// Bound for the model load before `/health` answers 200.
const READY_DEADLINE: Duration = Duration::from_secs(240);
/// Bound for any exit after a signal. The assertions on timing are tighter;
/// this only keeps a hung process from hanging the test.
const EXIT_DEADLINE: Duration = Duration::from_secs(60);

/// Where a test server listens.
enum Endpoint {
    Tcp(String),
    Unix(PathBuf),
}

impl Endpoint {
    fn connect(&self) -> std::io::Result<Box<dyn ReadWrite + Send>> {
        let timeout = Some(Duration::from_secs(30));
        match self {
            Self::Tcp(addr) => {
                let stream = TcpStream::connect(addr)?;
                stream.set_read_timeout(timeout)?;
                Ok(Box::new(stream))
            }
            Self::Unix(path) => {
                let stream = UnixStream::connect(path)?;
                stream.set_read_timeout(timeout)?;
                Ok(Box::new(stream))
            }
        }
    }
}

trait ReadWrite: Read + Write {}
impl<T: Read + Write> ReadWrite for T {}

/// A running server process; killed on drop so a failed assertion never
/// leaves it behind.
struct Server {
    child: Child,
    endpoint: Endpoint,
    stdout: Option<JoinHandle<String>>,
    stderr: Option<JoinHandle<String>>,
}

struct Exited {
    status: ExitStatus,
    /// Time from the signal to the exit.
    elapsed: Duration,
    logs: String,
}

impl Server {
    fn spawn(binary: &str, args: &[String], endpoint: Endpoint) -> Self {
        let mut child = Command::new(repo_binary_path(binary))
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|err| panic!("spawn {binary}: {err}"));
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let mut server = Self {
            child,
            endpoint,
            stdout: Some(read_all(stdout)),
            stderr: Some(read_all(stderr)),
        };
        server.wait_ready();
        server
    }

    fn wait_ready(&mut self) {
        let deadline = Instant::now() + READY_DEADLINE;
        loop {
            if let Some(status) = self.child.try_wait().expect("poll the server") {
                let logs = self.collect_logs();
                panic!("the server exited with {status} before it was ready\n{logs}");
            }
            if self
                .request("GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
                .is_some_and(|reply| reply.starts_with("HTTP/1.1 200"))
            {
                return;
            }
            if Instant::now() > deadline {
                let _ = self.child.kill();
                let _ = self.child.wait();
                let logs = self.collect_logs();
                panic!("the server was not ready within {READY_DEADLINE:?}\n{logs}");
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    /// One request read to the end, or `None` when the server is unreachable.
    fn request(&self, raw: &str) -> Option<String> {
        let mut stream = self.endpoint.connect().ok()?;
        stream.write_all(raw.as_bytes()).ok()?;
        let mut reply = Vec::new();
        let _ = stream.read_to_end(&mut reply);
        Some(String::from_utf8_lossy(&reply).into_owned())
    }

    /// Start a streaming chat completion and return once its first event is
    /// on the wire; the returned thread reads the rest of the stream.
    fn start_stream(&self, max_tokens: u32) -> JoinHandle<String> {
        let body = format!(
            r#"{{"model":"m","stream":true,"max_tokens":{max_tokens},"messages":[{{"role":"user","content":"Count from one to one thousand in words."}}]}}"#
        );
        let raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let mut stream = self.endpoint.connect().expect("connect for the stream");
        stream.write_all(raw.as_bytes()).expect("send the request");
        let mut seen = Vec::new();
        let mut chunk = [0u8; 4096];
        while !String::from_utf8_lossy(&seen).contains("data: {") {
            let n = stream.read(&mut chunk).expect("read the stream head");
            assert!(n > 0, "the stream ended before its first event");
            seen.extend_from_slice(&chunk[..n]);
        }
        std::thread::spawn(move || {
            let _ = stream.read_to_end(&mut seen);
            String::from_utf8_lossy(&seen).into_owned()
        })
    }

    fn signal(&self, signal: libc::c_int) {
        let pid = libc::pid_t::try_from(self.child.id()).expect("pid fits");
        // SAFETY: `kill` has no memory-safety preconditions; the pid is our
        // own live child.
        let rc = unsafe { libc::kill(pid, signal) };
        assert_eq!(rc, 0, "kill({pid}, {signal}) failed");
    }

    /// Send `signal` and wait (bounded) for the process to exit.
    fn stop_with(mut self, signal: libc::c_int) -> Exited {
        let started = Instant::now();
        self.signal(signal);
        self.wait_exit(started)
    }

    fn wait_exit(&mut self, started: Instant) -> Exited {
        let deadline = started + EXIT_DEADLINE;
        let status = loop {
            if let Some(status) = self.child.try_wait().expect("poll the server") {
                break status;
            }
            if Instant::now() > deadline {
                let _ = self.child.kill();
                let _ = self.child.wait();
                let logs = self.collect_logs();
                panic!("the server did not exit within {EXIT_DEADLINE:?} of the signal\n{logs}");
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let elapsed = started.elapsed();
        Exited {
            status,
            elapsed,
            logs: self.collect_logs(),
        }
    }

    fn collect_logs(&mut self) -> String {
        let stdout = self.stdout.take().map(|h| h.join().unwrap_or_default());
        let stderr = self.stderr.take().map(|h| h.join().unwrap_or_default());
        format!(
            "--- stdout ---\n{}\n--- stderr ---\n{}",
            stdout.unwrap_or_default(),
            stderr.unwrap_or_default()
        )
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn read_all(mut pipe: impl Read + Send + 'static) -> JoinHandle<String> {
    std::thread::spawn(move || {
        let mut text = String::new();
        let _ = pipe.read_to_string(&mut text);
        text
    })
}

fn skip_without_model() -> Option<PathBuf> {
    let model_dir = repo_model_dir_any(MODELS);
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return None;
    }
    Some(model_dir)
}

fn free_port() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a probe port");
    listener
        .local_addr()
        .expect("local addr")
        .port()
        .to_string()
}

fn path_arg(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// Start a single-model `mlxcel-server` on a loopback port.
fn single_model_server(model_dir: &Path, extra: &[&str]) -> Server {
    let port = free_port();
    let mut args = vec![
        "-m".to_string(),
        path_arg(model_dir),
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        port.clone(),
        "--no-warmup".to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    Server::spawn(
        "mlxcel-server",
        &args,
        Endpoint::Tcp(format!("127.0.0.1:{port}")),
    )
}

/// Exited on its own with `code`: no signal death and no C++ runtime abort.
fn assert_exit_code(exited: &Exited, code: i32) {
    let logs = &exited.logs;
    assert!(
        exited.status.signal().is_none(),
        "the server died of signal {:?} instead of exiting\n{logs}",
        exited.status.signal()
    );
    assert_eq!(exited.status.code(), Some(code), "exit code\n{logs}");
    assert!(
        !logs.contains("pure virtual method called") && !logs.contains("terminate called"),
        "C++ runtime abort text in the server output\n{logs}"
    );
}

fn assert_logged(exited: &Exited, line: &str) {
    assert!(
        exited.logs.contains(line),
        "missing {line:?} in the server output\n{}",
        exited.logs
    );
}

#[test]
fn sigterm_stops_an_idle_server_with_exit_code_zero() {
    let Some(model_dir) = skip_without_model() else {
        return;
    };
    let exited = single_model_server(&model_dir, &[]).stop_with(libc::SIGTERM);
    assert_exit_code(&exited, 0);
    assert_logged(&exited, "shutdown signal received: SIGTERM");
    assert_logged(&exited, "model worker exited before process exit");
    assert!(
        exited.elapsed < Duration::from_secs(2),
        "an idle server must exit within 2 s of SIGTERM: {:?}",
        exited.elapsed
    );
}

#[test]
fn sigterm_lets_an_in_flight_stream_finish_before_exiting() {
    let Some(model_dir) = skip_without_model() else {
        return;
    };
    let server = single_model_server(&model_dir, &[]);
    let stream = server.start_stream(64);
    let exited = server.stop_with(libc::SIGTERM);
    let body = stream.join().expect("stream reader");
    assert!(
        body.contains("data: [DONE]"),
        "the client must receive the final [DONE] event\n{body}\n{}",
        exited.logs
    );
    assert_exit_code(&exited, 0);
    assert!(
        exited.elapsed < Duration::from_secs(30),
        "exit within the default --shutdown-timeout: {:?}",
        exited.elapsed
    );
}

#[test]
fn the_drain_deadline_aborts_a_long_generation_and_still_exits_zero() {
    let Some(model_dir) = skip_without_model() else {
        return;
    };
    let server = single_model_server(&model_dir, &["--shutdown-timeout", "2", "--ignore-eos"]);
    let stream = server.start_stream(30_000);
    let exited = server.stop_with(libc::SIGTERM);
    let _ = stream.join();
    assert_exit_code(&exited, 0);
    assert_logged(
        &exited,
        "shutdown: drain timeout after 2 s, aborting 1 connection(s)",
    );
    assert!(
        exited.elapsed >= Duration::from_secs(2) && exited.elapsed < Duration::from_secs(10),
        "exit about 2 s after the signal: {:?}",
        exited.elapsed
    );
}

#[test]
fn sigint_stops_an_idle_server_with_exit_code_zero() {
    let Some(model_dir) = skip_without_model() else {
        return;
    };
    let exited = single_model_server(&model_dir, &[]).stop_with(libc::SIGINT);
    assert_exit_code(&exited, 0);
    assert_logged(&exited, "shutdown signal received: SIGINT");
    assert_logged(&exited, "model worker exited before process exit");
}

#[test]
fn a_second_signal_during_the_drain_exits_one_at_once() {
    let Some(model_dir) = skip_without_model() else {
        return;
    };
    let mut server = single_model_server(&model_dir, &["--ignore-eos"]);
    let stream = server.start_stream(30_000);
    server.signal(libc::SIGTERM);
    std::thread::sleep(Duration::from_millis(500));
    let started = Instant::now();
    server.signal(libc::SIGINT);
    let exited = server.wait_exit(started);
    let _ = stream.join();
    assert_exit_code(&exited, 1);
    assert_logged(
        &exited,
        "second shutdown signal (SIGINT); exiting without drain",
    );
    assert!(
        exited.elapsed < Duration::from_secs(2),
        "the second signal skips the 30 s drain: {:?}",
        exited.elapsed
    );
}

#[test]
fn router_mode_stops_on_sigterm_with_exit_code_zero() {
    let Some(model_dir) = skip_without_model() else {
        return;
    };
    let preset_dir = tempfile::tempdir().expect("temp dir for the preset");
    let preset_path = preset_dir.path().join("models.ini");
    std::fs::write(
        &preset_path,
        format!(
            "[eager]\nmodel = {}\nload-on-startup = 1\n",
            path_arg(&model_dir)
        ),
    )
    .expect("write preset");
    let port = free_port();
    let args = [
        "--models-preset".to_string(),
        path_arg(&preset_path),
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        port.clone(),
        "--no-warmup".to_string(),
    ];
    let server = Server::spawn(
        "mlxcel-server",
        &args,
        Endpoint::Tcp(format!("127.0.0.1:{port}")),
    );
    let exited = server.stop_with(libc::SIGTERM);
    assert_exit_code(&exited, 0);
    assert_logged(&exited, "shutdown signal received: SIGTERM");
    assert_logged(&exited, "router: lifecycle shutdown completed");
}

#[test]
fn mlxcel_serve_stops_on_sigterm_with_exit_code_zero() {
    let Some(model_dir) = skip_without_model() else {
        return;
    };
    let port = free_port();
    let args = [
        "serve".to_string(),
        "-m".to_string(),
        path_arg(&model_dir),
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        port.clone(),
        "--no-warmup".to_string(),
    ];
    let server = Server::spawn("mlxcel", &args, Endpoint::Tcp(format!("127.0.0.1:{port}")));
    let exited = server.stop_with(libc::SIGTERM);
    assert_exit_code(&exited, 0);
    assert_logged(&exited, "shutdown signal received: SIGTERM");
    assert_logged(&exited, "model worker exited before process exit");
}

#[test]
fn a_unix_socket_listener_removes_its_socket_file_on_sigterm() {
    let Some(model_dir) = skip_without_model() else {
        return;
    };
    // A short path: `sun_path` is limited to about 104 bytes on macOS.
    let dir = tempfile::Builder::new()
        .prefix("mlx")
        .tempdir_in("/tmp")
        .expect("temp dir for the socket");
    let socket = dir.path().join("s.sock");
    let args = [
        "-m".to_string(),
        path_arg(&model_dir),
        "--host".to_string(),
        path_arg(&socket),
        "--no-warmup".to_string(),
    ];
    let server = Server::spawn("mlxcel-server", &args, Endpoint::Unix(socket.clone()));
    assert!(socket.exists(), "the server bound its socket");
    let exited = server.stop_with(libc::SIGTERM);
    assert_exit_code(&exited, 0);
    assert!(
        std::fs::symlink_metadata(&socket).is_err(),
        "the socket file must be removed on exit\n{}",
        exited.logs
    );
}
