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

use std::ffi::OsString;
use std::sync::{Arc, mpsc};

use super::install_media_startup_inputs;
use crate::server::media::VIDEO_DIR_ALLOWLIST_ENV;
use crate::server::{
    AppState, ChatTemplateProcessor, ModelMediaSupport, ModelProvider, ServerConfig,
};
use crate::tokenizer::MlxcelTokenizer;

struct EnvGuard {
    previous: Option<OsString>,
}

impl EnvGuard {
    fn set(value: &std::path::Path) -> Self {
        let previous = std::env::var_os(VIDEO_DIR_ALLOWLIST_ENV);
        // SAFETY: the test holds the crate-wide environment lock until this
        // guard restores the previous value.
        unsafe { std::env::set_var(VIDEO_DIR_ALLOWLIST_ENV, value) };
        Self { previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: the crate-wide environment lock still outlives this guard.
        unsafe {
            match &self.previous {
                Some(value) => std::env::set_var(VIDEO_DIR_ALLOWLIST_ENV, value),
                None => std::env::remove_var(VIDEO_DIR_ALLOWLIST_ENV),
            }
        }
    }
}

fn test_state() -> AppState {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        "in-process-startup-state-test-model".into(),
        batch_metrics,
    )
}

#[test]
fn installs_video_allowlist_with_media_support() {
    let _env_lock = crate::test_support::env_lock::env_lock();
    let allowlist_dir = tempfile::tempdir().unwrap();
    let canonical = std::fs::canonicalize(allowlist_dir.path()).unwrap();
    let _env = EnvGuard::set(&canonical);
    let support = ModelMediaSupport {
        video_native: true,
        ..ModelMediaSupport::default()
    };

    let state = install_media_startup_inputs(test_state(), support);

    assert_eq!(state.media_support, support);
    assert_eq!(state.video_dir_allowlist.as_slice(), &[canonical]);
}
