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

//! Test-only guards that flip the process-wide private-address opt-in
//! (issue #1451), moved out of `media_net.rs` to keep it under the file-size
//! limit (issue #2644).

use super::{configure_private_media_urls, private_addresses_allowed};

/// Allow private media addresses for the duration of a test, restoring the
/// previous setting on drop.
///
/// Serialized against every other holder, so a test that stands up a loopback
/// origin cannot overlap one that asserts the refusal.
#[must_use]
pub(crate) fn allow_private_media_urls_in_tests() -> PrivateAddressTestGuard {
    private_media_urls_in_tests(true)
}

/// Refuse private media addresses for the duration of a test.
///
/// A test that asserts the refusal has to hold the same lock as one that opts
/// in, or the two overlap and the refusal never fires.
#[must_use]
pub(crate) fn deny_private_media_urls_in_tests() -> PrivateAddressTestGuard {
    private_media_urls_in_tests(false)
}

fn private_media_urls_in_tests(allowed: bool) -> PrivateAddressTestGuard {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let guard = LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let previous = private_addresses_allowed();
    configure_private_media_urls(allowed);
    PrivateAddressTestGuard {
        _lock: guard,
        previous,
    }
}

/// Restores the process-wide opt-in when dropped.
pub(crate) struct PrivateAddressTestGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    previous: bool,
}

impl Drop for PrivateAddressTestGuard {
    fn drop(&mut self) {
        configure_private_media_urls(self.previous);
    }
}
