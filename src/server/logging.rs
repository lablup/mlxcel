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

//! Log destination, format, verbosity precedence, and secret redaction for
//! the two server binaries (issue #1448, epic #1431).
//!
//! b10621 exposes six knobs over one logger: `--log-disable`, `--log-file`,
//! `--log-colors`, `--log-prefix` / `--no-log-prefix`, `--log-timestamps` /
//! `--no-log-timestamps`, and the verbosity pair `-v` / `--verbose` /
//! `--log-verbose` and `-lv` / `--verbosity` / `--log-verbosity`. Before
//! #1448 mlxcel accepted only the first two, built its filter from
//! `RUST_LOG` in a way that *overrode* `--verbose`, and opened the log file
//! at whatever the process umask happened to allow.
//!
//! This module owns all of it, so the precedence lives in one readable place
//! instead of being spread across `initialize_server_logging` and the clap
//! attributes on two binaries.
//!
//! # Destination precedence
//!
//! 1. `--log-disable` wins over everything and installs no subscriber at all.
//! 2. `--log-file PATH` on the command line.
//! 3. `LLAMA_ARG_LOG_FILE` (b10621's canonical binding, bound through clap).
//! 4. `LLAMA_LOG_FILE` (mlxcel's pre-#1448 spelling, kept as a fallback).
//! 5. Standard output.
//!
//! An unusable destination is a **startup failure**, never a silent fallback
//! to the terminal: a deployment that asked for a log file and got stdout
//! believes it has an audit trail it does not have. See [`open_log_file`] for
//! the exact refusals.
//!
//! # Verbosity precedence
//!
//! 1. `--verbose` / `--log-verbose` on the command line: every mlxcel message.
//! 2. `--verbosity N` / `--log-verbosity N` on the command line.
//! 3. `RUST_LOG`, mlxcel's native per-target filter.
//! 4. `LLAMA_ARG_LOG_VERBOSITY`.
//! 5. The compiled-in default, b10621's threshold `3` (info).
//!
//! A command-line flag always beats the environment, which is what b10621
//! does and what mlxcel did **not** do before #1448: `EnvFilter::try_from_default_env`
//! ran first, so `RUST_LOG=warn mlxcel-server -v` silently ignored `-v`.
//!
//! # Secrets
//!
//! [`register_log_secret`] records a value that must never reach a log sink;
//! every line written through [`LogSinkWriter`] is scanned and any occurrence
//! replaced with [`REDACTED`]. API keys (`--api-key`, `--api-key-file`,
//! `LLAMA_API_KEY`) and repository tokens (`--hf-token`, `HF_TOKEN`) are
//! registered by the binaries before the subscriber is installed, so a future
//! `tracing::debug!("config: {args:?}")` cannot turn a log file into a
//! credential store. `tests/llama_logging_presets.rs` proves it with canary
//! values against a real process.
//!
//! Used by: mlxcel serve, mlxcel-server.
//!
//! Upstream reference: <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp>

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Context, Result};
use tracing_subscriber::fmt::MakeWriter;

/// b10621's default verbosity threshold (`common_params::verbosity`).
pub const DEFAULT_VERBOSITY: i32 = 3;

/// Replacement text substituted for a registered secret.
pub const REDACTED: &str = "[redacted]";

/// Shortest value [`register_log_secret`] will register.
///
/// A one or two character "secret" would redact ordinary prose everywhere it
/// occurred and make the log useless; anything that short is not a credential
/// worth protecting either.
const MIN_SECRET_LEN: usize = 8;

// ── colors ──────────────────────────────────────────────────────────────

/// b10621 `--log-colors` value domain (`on` / `off` / `auto`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogColors {
    /// Color when the sink is a terminal. b10621's default.
    #[default]
    Auto,
    /// Always color.
    On,
    /// Never color.
    Off,
}

impl LogColors {
    /// Parse a `--log-colors` value using b10621's own vocabulary.
    ///
    /// `common_arg_utils::is_truthy` / `is_falsey` / `is_autoy` compare
    /// case-sensitively against fixed word lists, and anything outside them
    /// throws; this reproduces that rather than accepting clap's wider
    /// boolish set.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "on" | "enabled" | "true" | "1" => Ok(Self::On),
            "off" | "disabled" | "false" | "0" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            other => Err(format!(
                "--log-colors: unknown value {other:?}; expected 'on', 'off' or 'auto'"
            )),
        }
    }

    /// Whether ANSI escapes should be emitted.
    ///
    /// `to_file` is true when the sink is a log file rather than the
    /// terminal; `Auto` never colors a file, matching b10621's
    /// `LOG_COLORS_AUTO`.
    #[must_use]
    pub fn resolve(self, to_file: bool, sink_is_terminal: bool) -> bool {
        match self {
            Self::On => true,
            Self::Off => false,
            Self::Auto => !to_file && sink_is_terminal,
        }
    }
}

// ── verbosity ───────────────────────────────────────────────────────────

/// Where the effective verbosity came from, highest precedence first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerbositySource {
    /// `--verbose` / `--log-verbose` on the command line.
    VerboseFlag,
    /// `--verbosity N` / `--log-verbosity N` on the command line.
    VerbosityFlag,
    /// `RUST_LOG`.
    RustLog,
    /// `LLAMA_ARG_LOG_VERBOSITY`.
    LlamaEnv,
    /// The compiled-in default.
    Default,
}

/// Map a b10621 verbosity threshold onto a `tracing` `EnvFilter` directive.
///
/// b10621 documents the scale as 0 generic, 1 error, 2 warning, 3 info,
/// 4 trace, 5 debug, and ignores every message *above* the threshold, so a
/// larger number always means more output. The mlxcel tiers below preserve
/// that monotonicity. The top tier raises only mlxcel's own targets to
/// `trace`, leaving dependencies at `debug`: a global `trace` directive turns
/// on hyper and tokio internals, which buries the mlxcel messages the
/// operator asked to see.
#[must_use]
pub fn filter_directive_for_verbosity(threshold: i32) -> &'static str {
    match threshold {
        i32::MIN..=1 => "error",
        2 => "warn",
        3 => "info",
        4 => "debug",
        _ => "debug,mlxcel=trace,mlxcel_core=trace",
    }
}

/// Resolve the effective `EnvFilter` directive and the source that produced
/// it, applying the module-level precedence.
///
/// Pure: every input is passed in, so the unit tests need no process
/// environment. `rust_log` is `RUST_LOG`'s value, `env_verbosity` is a parsed
/// `LLAMA_ARG_LOG_VERBOSITY`.
#[must_use]
pub fn resolve_log_filter(
    verbose_flag: bool,
    cli_verbosity: Option<i32>,
    rust_log: Option<&str>,
    env_verbosity: Option<i32>,
) -> (String, VerbositySource) {
    if verbose_flag {
        return (
            filter_directive_for_verbosity(i32::MAX).to_owned(),
            VerbositySource::VerboseFlag,
        );
    }
    if let Some(threshold) = cli_verbosity {
        return (
            filter_directive_for_verbosity(threshold).to_owned(),
            VerbositySource::VerbosityFlag,
        );
    }
    if let Some(directive) = rust_log.filter(|d| !d.trim().is_empty()) {
        return (directive.to_owned(), VerbositySource::RustLog);
    }
    if let Some(threshold) = env_verbosity {
        return (
            filter_directive_for_verbosity(threshold).to_owned(),
            VerbositySource::LlamaEnv,
        );
    }
    (
        filter_directive_for_verbosity(DEFAULT_VERBOSITY).to_owned(),
        VerbositySource::Default,
    )
}

/// Resolved b10621 log-format state carried from the CLI into `start_server`.
///
/// One field on `ServerStartupInput` / `ServerStartupConfig` rather than six,
/// so the two binaries and the startup path grow a single line each.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogFormatOptions {
    /// `--log-colors`.
    pub colors: LogColors,
    /// `--log-prefix` / `--no-log-prefix`: the per-line level tag.
    pub prefix: bool,
    /// `--log-timestamps` / `--no-log-timestamps`.
    pub timestamps: bool,
    /// `--verbosity N` when it came from the command line.
    pub cli_verbosity: Option<i32>,
    /// `LLAMA_ARG_LOG_VERBOSITY` when it came from the environment.
    pub env_verbosity: Option<i32>,
}

impl Default for LogFormatOptions {
    fn default() -> Self {
        Self {
            colors: LogColors::Auto,
            // b10621's llama-server prints both by default, verified against
            // the pinned macOS arm64 binary.
            prefix: true,
            timestamps: true,
            cli_verbosity: None,
            env_verbosity: None,
        }
    }
}

// ── secret redaction ────────────────────────────────────────────────────

fn secrets() -> &'static RwLock<Vec<String>> {
    static SECRETS: RwLock<Vec<String>> = RwLock::new(Vec::new());
    &SECRETS
}

/// Register a value that must never appear in any log sink.
///
/// Ignores empty, whitespace-only, and values shorter than
/// [`MIN_SECRET_LEN`]. Idempotent: registering the same value twice is a
/// no-op. Call before [`install`]; registering later still works, but lines
/// already written are gone.
pub fn register_log_secret(value: &str) {
    let value = value.trim();
    if value.len() < MIN_SECRET_LEN {
        return;
    }
    let Ok(mut guard) = secrets().write() else {
        // A poisoned registry means a previous writer panicked. Failing to
        // register would silently weaken redaction, so treat it as fatal for
        // this call only and leave the process alone.
        return;
    };
    if !guard.iter().any(|existing| existing == value) {
        guard.push(value.to_owned());
    }
}

/// Number of registered secrets. Test and diagnostic helper.
#[must_use]
pub fn registered_secret_count() -> usize {
    secrets().read().map(|g| g.len()).unwrap_or(0)
}

/// Replace every registered secret in `line` with [`REDACTED`].
///
/// Returns `line` unchanged when nothing matched, so the common case
/// allocates nothing.
#[must_use]
pub fn redact(line: &str) -> std::borrow::Cow<'_, str> {
    let Ok(guard) = secrets().read() else {
        return std::borrow::Cow::Borrowed(line);
    };
    if guard.is_empty() {
        return std::borrow::Cow::Borrowed(line);
    }
    let mut out: Option<String> = None;
    for secret in guard.iter() {
        let current = out.as_deref().unwrap_or(line);
        if current.contains(secret.as_str()) {
            out = Some(current.replace(secret.as_str(), REDACTED));
        }
    }
    match out {
        Some(replaced) => std::borrow::Cow::Owned(replaced),
        None => std::borrow::Cow::Borrowed(line),
    }
}

// ── sink ────────────────────────────────────────────────────────────────

/// The concrete destination a log line ends up in.
///
/// One enum rather than two generic subscriber branches: the subscriber
/// builder's type changes with `without_time()`, and multiplying that by a
/// writer type parameter produces four nearly identical `init()` arms.
#[derive(Debug)]
enum LogSink {
    Stdout(io::Stdout),
    File(File),
}

impl Write for LogSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::Stdout(w) => w.write(buf),
            Self::File(w) => w.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Stdout(w) => w.flush(),
            Self::File(w) => w.flush(),
        }
    }
}

/// Line-buffered, redacting wrapper around a [`LogSink`].
///
/// Redaction has to happen on whole lines: `tracing_subscriber`'s formatter
/// writes an event in several `write` calls (timestamp, level, target,
/// fields), so a secret that straddles two of them would slip through a
/// per-call scan. Bytes accumulate here until a newline, and only complete
/// lines reach the sink.
#[derive(Debug)]
struct LineRedactor {
    sink: LogSink,
    pending: Vec<u8>,
}

impl LineRedactor {
    fn emit_line(&mut self, line: &[u8]) -> io::Result<()> {
        let text = String::from_utf8_lossy(line);
        self.sink.write_all(redact(&text).as_bytes())?;
        self.sink.write_all(b"\n")
    }

    fn drain_complete_lines(&mut self) -> io::Result<()> {
        while let Some(index) = self.pending.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = self.pending.drain(..=index).collect();
            let without_newline = &line[..line.len() - 1];
            self.emit_line(without_newline)?;
        }
        Ok(())
    }

    fn flush_pending(&mut self) -> io::Result<()> {
        self.drain_complete_lines()?;
        if !self.pending.is_empty() {
            let rest = std::mem::take(&mut self.pending);
            self.emit_line(&rest)?;
        }
        self.sink.flush()
    }
}

/// `MakeWriter` handing every event to one shared [`LineRedactor`].
#[derive(Debug, Clone)]
pub struct RedactingWriter {
    shared: Arc<Mutex<LineRedactor>>,
}

/// The per-event writer. Holds no borrow of the `MakeWriter`, so the
/// `MakeWriter` impl needs no lifetime gymnastics.
#[derive(Debug)]
pub struct LogSinkWriter {
    shared: Arc<Mutex<LineRedactor>>,
}

impl Write for LogSinkWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let Ok(mut guard) = self.shared.lock() else {
            // A poisoned log mutex must not take the server down; drop the
            // line rather than panicking inside a tracing event.
            return Ok(buf.len());
        };
        guard.pending.extend_from_slice(buf);
        guard.drain_complete_lines()?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        let Ok(mut guard) = self.shared.lock() else {
            return Ok(());
        };
        guard.flush_pending()
    }
}

impl<'a> MakeWriter<'a> for RedactingWriter {
    type Writer = LogSinkWriter;

    fn make_writer(&'a self) -> Self::Writer {
        LogSinkWriter {
            shared: Arc::clone(&self.shared),
        }
    }
}

// ── log file ────────────────────────────────────────────────────────────

/// Open (or create) the `--log-file` destination with restrictive
/// permissions, refusing every shape that would make the log untrustworthy.
///
/// Refusals, all at startup and all before a weight is read:
///
/// - The path is a symlink. A pre-planted symlink is how an unprivileged
///   local user turns a log file into an append primitive against a file the
///   server can write but they cannot.
/// - The path is a directory, or its parent directory does not exist.
/// - The file cannot be opened for append.
///
/// On unix the file is created `0600` and an existing file is tightened to
/// `0600`, so a log containing prompts, model paths, and request metadata is
/// not readable by every account on the host. The default umask would have
/// left it `0644`.
pub fn open_log_file(path: &Path) -> Result<File> {
    if let Ok(metadata) = std::fs::symlink_metadata(path)
        && metadata.file_type().is_symlink()
    {
        anyhow::bail!(
            "--log-file: {} is a symbolic link; refusing to write logs through it. \
             Name the destination file directly.",
            path.display()
        );
    }
    if path.is_dir() {
        anyhow::bail!("--log-file: {} is a directory, not a file", path.display());
    }
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
        && !parent.is_dir()
    {
        anyhow::bail!(
            "--log-file: directory {} does not exist; create it or name a different path",
            parent.display()
        );
    }

    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options
        .open(path)
        .with_context(|| format!("--log-file: cannot open {} for writing", path.display()))?;

    // `mode()` above only applies when this call created the file. An
    // existing world-readable log is tightened here so re-running the server
    // fixes the permissions instead of inheriting them.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions = std::fs::Permissions::from_mode(0o600);
        std::fs::set_permissions(path, permissions).with_context(|| {
            format!(
                "--log-file: cannot restrict permissions on {} to 0600",
                path.display()
            )
        })?;
    }
    Ok(file)
}

/// Validate the `--log-file` destination before the model reference is
/// resolved, creating (and tightening) the file the way [`install`] later
/// will.
///
/// `initialize_server_logging` runs inside `start_server`, which both server
/// binaries reach only after the model reference has resolved, which can mean
/// after a multi-gigabyte download. A log destination that cannot be written
/// must be reported before that, not after it, so both binaries call this at
/// the same point they classify the rest of the b10621 compatibility surface.
///
/// Creating the file here rather than only checking it matches b10621, whose
/// `--log-file` handler opens the file from inside the argument parser. The
/// handle is dropped immediately; [`install`] reopens the path in append mode.
///
/// `log_disable` short-circuits: with no subscriber there is no destination to
/// validate. `log_file` is whatever the caller has resolved so far, so a
/// destination that arrives only through the mlxcel-native `LLAMA_LOG_FILE`
/// fallback is validated by [`install`] instead, which is still before the
/// server listens.
pub fn precheck_log_destination(log_disable: bool, log_file: Option<&Path>) -> Result<()> {
    if log_disable {
        return Ok(());
    }
    let Some(path) = log_file else {
        return Ok(());
    };
    open_log_file(path).map(|_| ())
}

// ── installation ────────────────────────────────────────────────────────

/// Install the process-wide tracing subscriber for the server binaries.
///
/// `log_file` is the already-resolved destination (see the module-level
/// precedence); `None` means standard output. Returns the one-line summary
/// the caller logs once the subscriber exists, so the effective destination,
/// filter, and filter source are stated in the log itself.
pub fn install(
    format: &LogFormatOptions,
    log_file: Option<&Path>,
    verbose_flag: bool,
) -> Result<String> {
    let (directive, source) = resolve_log_filter(
        verbose_flag,
        format.cli_verbosity,
        std::env::var("RUST_LOG").ok().as_deref(),
        format.env_verbosity,
    );
    let env_filter = tracing_subscriber::EnvFilter::try_new(&directive).unwrap_or_else(|_| {
        tracing_subscriber::EnvFilter::new(filter_directive_for_verbosity(DEFAULT_VERBOSITY))
    });

    let (sink, to_file, destination) = match log_file {
        Some(path) => (
            LogSink::File(open_log_file(path)?),
            true,
            path.display().to_string(),
        ),
        None => (LogSink::Stdout(io::stdout()), false, "stdout".to_owned()),
    };
    let sink_is_terminal = !to_file && std::io::IsTerminal::is_terminal(&io::stdout());
    let ansi = format.colors.resolve(to_file, sink_is_terminal);

    let writer = RedactingWriter {
        shared: Arc::new(Mutex::new(LineRedactor {
            sink,
            pending: Vec::new(),
        })),
    };

    let builder = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_ansi(ansi)
        .with_level(format.prefix)
        .with_writer(writer);
    if format.timestamps {
        builder
            .try_init()
            .map_err(|e| anyhow::anyhow!("failed to install the log subscriber: {e}"))?;
    } else {
        builder
            .without_time()
            .try_init()
            .map_err(|e| anyhow::anyhow!("failed to install the log subscriber: {e}"))?;
    }

    Ok(format!(
        "logging to {destination} (filter {directive:?} from {source:?}, colors={ansi}, \
         prefix={}, timestamps={})",
        format.prefix, format.timestamps
    ))
}

#[cfg(test)]
#[path = "logging_tests.rs"]
mod tests;
