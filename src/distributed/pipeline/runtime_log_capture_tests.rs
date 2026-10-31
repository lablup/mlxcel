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

//! A per-thread `tracing` capture for the coordinator tests (#2782,
//! #2790): the tests read back the lines the runtime logged.

use std::sync::Arc;

/// A `tracing` writer that appends to a shared buffer, so a test can read
/// back what the coordinator logged on this thread.
#[derive(Clone)]
pub(super) struct LogBuf(pub(super) Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for LogBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogBuf {
    type Writer = LogBuf;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// A log capture for the calling thread, at debug and above.
pub(super) fn capture_logs() -> (LogBuf, tracing::subscriber::DefaultGuard) {
    let log = LogBuf(Arc::new(std::sync::Mutex::new(Vec::new())));
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_ansi(false)
        .with_writer(log.clone())
        .finish();
    let guard = tracing::subscriber::set_default(subscriber);
    (log, guard)
}

/// The gate's closed-to-open lines in `log`.
pub(super) fn gate_open_lines(log: &LogBuf) -> Vec<String> {
    let captured = String::from_utf8_lossy(&log.0.lock().unwrap()).into_owned();
    captured
        .lines()
        .filter(|line| line.contains(" WARN ") && line.contains("refusing new pipeline requests"))
        .map(str::to_string)
        .collect()
}
