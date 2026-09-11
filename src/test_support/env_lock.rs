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

//! Crate-wide environment-variable lock for unit tests.
//!
//! `std::env::set_var` and `std::env::remove_var` mutate the process-global
//! environment block. On Rust 2024 they became `unsafe` because libc's
//! `setenv`/`getenv` have no internal synchronization: a concurrent reader
//! on any thread (including async runtimes, file-IO crates, and
//! `mlxcel-core`'s startup paths that read `MLXCEL_*` env vars) is
//! undefined behavior.
//!
//! Cargo's default test runner runs every `#[test]` in the same binary
//! across a thread pool, and `cargo test --workspace` runs multiple test
//! binaries in parallel processes. Per-module `OnceLock<Mutex<()>>` locks
//! only protect tests within the same module — two modules each holding
//! their own lock can still call `set_var`/`remove_var` simultaneously
//! on different keys from different threads of the same test binary.
//!
//! This module exposes a single [`env_lock`] function that returns a guard
//! on a shared `Mutex<()>`. Every test that touches the process environment
//! in this crate must acquire this guard for the full duration of its
//! `set_var`/`remove_var` window. Tests targeting unrelated keys will
//! serialize, but the cost is small (env mutation tests are rare and fast)
//! and the alternative — reasoning about which keys are safe to interleave
//! — is fragile.
//!
//! `mlxcel-core` carries its own copy of this lock under
//! `mlxcel_core::test_support::env_lock` because tests in a separate crate
//! compile to a separate test binary and cannot share a `static`.
//!
//! Used by: every `#[cfg(test)]` module in this crate that calls
//! `std::env::set_var` or `std::env::remove_var`.

use std::sync::{Mutex, MutexGuard, OnceLock};

/// Process-wide singleton serializing every env-var mutation in this
/// crate's test binary. Lazily initialized on first acquire.
static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// Acquire the crate-wide env-var lock for the lifetime of the returned
/// guard. Recovers from a poisoned lock by taking the inner guard, since a
/// previous test panicking does not corrupt the env-block guard's logical
/// state — the next test will simply re-overwrite the keys it cares about.
///
/// Hold the guard for the full window in which env mutations are visible.
/// Acquire it BEFORE any RAII guard whose `Drop` calls `remove_var` so the
/// outer env_lock guard outlives the env mutation.
pub(crate) fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}
