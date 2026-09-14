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

//! Observation-only policy shared by bootstrap and mutation admission.
//! Cache presence means configured authority, not a filesystem permission probe.

use super::api::{ActionAvailability, WebUiServerMode};

pub(crate) fn availability(
    mode: WebUiServerMode,
    cache_available: bool,
    download: bool,
) -> ActionAvailability {
    project(
        mode,
        cache_available,
        download,
        crate::downloader::offline_mode(),
        crate::server::router_cache::managed_mutations_supported(),
    )
}

fn project(
    mode: WebUiServerMode,
    cache_available: bool,
    download: bool,
    offline: bool,
    platform_supported: bool,
) -> ActionAvailability {
    let (state, reason, instructions) = if mode == WebUiServerMode::SingleModel {
        (
            "read_only",
            Some("single-model mode does not own the router managed cache"),
            Some("Restart without -m/--model to manage cached models"),
        )
    } else if !cache_available {
        (
            "disabled",
            Some("no managed model cache is configured"),
            Some("Configure --model-store-root and restart to enable library mutations"),
        )
    } else if !platform_supported {
        (
            "disabled",
            Some("managed cache mutations require supported directory-descriptor operations"),
            Some("Use a supported macOS or Linux host"),
        )
    } else if download && offline {
        (
            "disabled",
            Some("offline mode is enabled"),
            Some("Restart without --offline / LLAMA_ARG_OFFLINE to download public models"),
        )
    } else {
        ("enabled", None, None)
    };
    ActionAvailability {
        state,
        reason,
        instructions,
    }
}

#[cfg(test)]
#[path = "library_policy_tests.rs"]
mod tests;
