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

//! Progress bar helpers for the mlxcel downloader.
//!
//! Wraps `indicatif::MultiProgress` with TTY detection and env-var opt-out so
//! that CI logs and piped output remain clean plain-text while interactive
//! terminals get per-file and aggregate progress bars.

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::io::IsTerminal;

/// Check whether progress bars should be shown.
///
/// Bars are suppressed when ANY of the following is true (checked in order):
/// 1. `MLXCEL_NO_PROGRESS` is set to any non-empty value (mlxcel-specific opt-out).
/// 2. `NO_COLOR` is set to any non-empty value (de-facto terminal color suppression standard).
/// 3. `CI` is set to `true` (GitHub Actions and similar CI environments).
/// 4. `stderr` is not a TTY (piped output, redirected logs, etc.).
///
/// When bars are suppressed, the caller falls back to the original line-per-file
/// `println!` output so CI log captures stay golden-text-stable.
pub fn should_show_progress() -> bool {
    // 1. mlxcel-specific opt-out: MLXCEL_NO_PROGRESS=<any non-empty value>
    if matches!(std::env::var("MLXCEL_NO_PROGRESS"), Ok(val) if !val.trim().is_empty()) {
        return false;
    }
    // 2. Standard no-color flag: NO_COLOR=<any non-empty value>
    if matches!(std::env::var("NO_COLOR"), Ok(val) if !val.trim().is_empty()) {
        return false;
    }
    // 3. CI environment: CI=true (GitHub Actions sets this automatically)
    if matches!(std::env::var("CI"), Ok(val) if val.trim().eq_ignore_ascii_case("true") || val.trim() == "1")
    {
        return false;
    }
    // 4. Terminal detection: bars go to stderr; suppress when stderr is not a tty
    if !std::io::stderr().is_terminal() {
        return false;
    }
    true
}

/// Create a `MultiProgress` with the appropriate draw target.
///
/// When progress bars are disabled (`should_show_progress()` returns false),
/// returns a hidden `MultiProgress` so all added bars are suppressed silently.
/// When enabled, bars render to stderr so stdout is kept free for structured output.
pub fn create_multi_progress() -> MultiProgress {
    if should_show_progress() {
        MultiProgress::with_draw_target(ProgressDrawTarget::stderr())
    } else {
        MultiProgress::with_draw_target(ProgressDrawTarget::hidden())
    }
}

/// Build the per-file progress bar style.
///
/// Style: `{spinner} {wide_msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, ETA {eta})`
pub fn file_progress_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.cyan} {wide_msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, ETA {eta})",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=>-")
}

/// Build the aggregate progress bar style.
///
/// Style: `Total: [{bar:40.green/yellow}] {bytes}/{total_bytes} ({bytes_per_sec})`
pub fn aggregate_progress_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "Total: [{bar:40.green/yellow}] {bytes}/{total_bytes} ({bytes_per_sec})",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=>-")
}

/// Create a per-file progress bar attached to the given `MultiProgress`.
///
/// `filename` is set as the initial bar message. `total_bytes` sets the bar
/// length; pass `0` for indeterminate (spinner-only).
pub fn add_file_bar(mp: &MultiProgress, filename: &str, total_bytes: u64) -> ProgressBar {
    let pb = if total_bytes > 0 {
        ProgressBar::new(total_bytes)
    } else {
        ProgressBar::new_spinner()
    };
    pb.set_style(file_progress_style());
    pb.set_message(filename.to_string());
    mp.add(pb)
}

/// Create the aggregate progress bar attached to the given `MultiProgress`.
///
/// The aggregate bar is inserted at the bottom of the bar stack, below
/// per-file bars. `total_bytes` is the sum of all file sizes.
pub fn add_aggregate_bar(mp: &MultiProgress, total_bytes: u64) -> ProgressBar {
    let pb = if total_bytes > 0 {
        ProgressBar::new(total_bytes)
    } else {
        ProgressBar::new_spinner()
    };
    pb.set_style(aggregate_progress_style());
    mp.add(pb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::env_lock::env_lock;

    #[test]
    fn mlxcel_no_progress_suppresses_bars() {
        let _guard = env_lock();
        unsafe {
            std::env::set_var("MLXCEL_NO_PROGRESS", "1");
            std::env::remove_var("NO_COLOR");
            std::env::remove_var("CI");
        }
        assert!(!should_show_progress());
        unsafe {
            std::env::remove_var("MLXCEL_NO_PROGRESS");
        }
    }

    #[test]
    fn no_color_suppresses_bars() {
        let _guard = env_lock();
        unsafe {
            std::env::remove_var("MLXCEL_NO_PROGRESS");
            std::env::set_var("NO_COLOR", "1");
            std::env::remove_var("CI");
        }
        assert!(!should_show_progress());
        unsafe {
            std::env::remove_var("NO_COLOR");
        }
    }

    #[test]
    fn ci_true_suppresses_bars() {
        let _guard = env_lock();
        unsafe {
            std::env::remove_var("MLXCEL_NO_PROGRESS");
            std::env::remove_var("NO_COLOR");
            std::env::set_var("CI", "true");
        }
        assert!(!should_show_progress());
        unsafe {
            std::env::remove_var("CI");
        }
    }

    #[test]
    fn ci_one_suppresses_bars() {
        let _guard = env_lock();
        unsafe {
            std::env::remove_var("MLXCEL_NO_PROGRESS");
            std::env::remove_var("NO_COLOR");
            std::env::set_var("CI", "1");
        }
        assert!(!should_show_progress());
        unsafe {
            std::env::remove_var("CI");
        }
    }

    #[test]
    fn mlxcel_no_progress_takes_precedence_over_other_vars() {
        // All three are set; MLXCEL_NO_PROGRESS wins regardless of others.
        let _guard = env_lock();
        unsafe {
            std::env::set_var("MLXCEL_NO_PROGRESS", "yes");
            std::env::set_var("NO_COLOR", "1");
            std::env::set_var("CI", "true");
        }
        assert!(!should_show_progress());
        unsafe {
            std::env::remove_var("MLXCEL_NO_PROGRESS");
            std::env::remove_var("NO_COLOR");
            std::env::remove_var("CI");
        }
    }

    #[test]
    fn hidden_multi_progress_when_suppressed() {
        let _guard = env_lock();
        unsafe {
            std::env::set_var("MLXCEL_NO_PROGRESS", "1");
        }
        // When bars are suppressed, adding a bar to the hidden MultiProgress
        // should produce a hidden ProgressBar (is_hidden() == true).
        let mp = create_multi_progress();
        let pb = add_file_bar(&mp, "test.safetensors", 1024);
        assert!(pb.is_hidden());
        unsafe {
            std::env::remove_var("MLXCEL_NO_PROGRESS");
        }
    }
}
