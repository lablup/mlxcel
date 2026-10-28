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

//! Operator-facing lines raised before the server's tracing subscriber
//! exists (#2697, #2722).
//!
//! The binaries resolve their command line and environment
//! (`build_startup_input`, the `env_fallback_*` helpers,
//! `ServerStartupInput::into_startup_config`) before `start_server` calls
//! `initialize_server_logging`, so a `tracing::` macro there writes to
//! nothing. Each line is collected as a [`StartupNotice`] instead and
//! `start_server` logs it at its own level once the subscriber exists.

use std::fmt;

/// The level a [`StartupNotice`] is logged at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartupNoticeLevel {
    /// Informational, for example "the CLI flag takes precedence over the
    /// environment variable".
    Info,
    /// Something the operator configured was invalid, ignored, or
    /// substituted.
    Warn,
    /// The server starts, but a configured feature will fail at request
    /// time (for example an unparseable `--grammar`).
    Error,
}

/// One line collected before the tracing subscriber exists, logged by
/// `start_server` at [`Self::level`] right after it installs logging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartupNotice {
    /// The level the line is logged at.
    pub level: StartupNoticeLevel,
    /// The line itself.
    pub message: String,
}

impl StartupNotice {
    /// An informational notice.
    pub fn info(message: impl Into<String>) -> Self {
        Self {
            level: StartupNoticeLevel::Info,
            message: message.into(),
        }
    }

    /// A warning notice.
    pub fn warn(message: impl Into<String>) -> Self {
        Self {
            level: StartupNoticeLevel::Warn,
            message: message.into(),
        }
    }

    /// An error notice.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            level: StartupNoticeLevel::Error,
            message: message.into(),
        }
    }

    /// Whether the message contains `pattern`.
    pub fn contains(&self, pattern: &str) -> bool {
        self.message.contains(pattern)
    }
}

impl fmt::Display for StartupNotice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

#[cfg(test)]
#[path = "startup_notice_tests.rs"]
mod tests;
