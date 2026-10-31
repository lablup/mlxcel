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

//! Environment writes that native code can see, on every platform (#2803).
//!
//! MLX, its overlays and the bridge read their settings with `std::getenv`. On
//! Windows the C runtime keeps its own copies of the environment, one narrow
//! and one wide, and `std::env::set_var` only calls `SetEnvironmentVariableW`,
//! so a value set from Rust never reaches `getenv`. [`set_native_var`] also
//! calls `_wputenv_s`, which updates both copies. Elsewhere `getenv` and
//! `std::env` share one table and these functions are the `std::env` calls.
//!
//! This relies on Rust, the bridge and MLX sharing one C runtime. They do: the
//! bridge, MLX and the `nvcc` host objects are all built `/MD`, the Rust target
//! does not set `crt-static`, and a binary imports `getenv` and `_wputenv_s`
//! from one UCRT. `native_env_tests.rs` reads back through both the bridge and
//! `mlx::core::env::get_var` so that a build where this stops holding fails.
//!
//! Production code writes the environment only through this module, which
//! `scripts/ci/check_native_env_writes.py` enforces. Tests may use `std::env`.
//!
//! What a native reader sees on Windows:
//!
//! - `getenv` returns the value in the active code page, so a character outside
//!   it comes back lossy (`?`, or a best fit). `std::env` and
//!   `GetEnvironmentVariableW` keep the exact value. The only value here that
//!   can hold such characters is `MLX_PTX_CACHE_DIR`, a path, and its reader in
//!   `ptx_cache_dir_win.h` uses `GetEnvironmentVariableW` for this reason.
//! - The C runtime cannot hold an empty value, so an empty value reads as unset
//!   through `getenv`, while `std::env` still returns the empty string.

use std::ffi::OsStr;

/// The variables written in production that native code reads, in the order
/// the startup report prints them.
pub const AUDITED_NATIVE_VARS: &[&str] = &[
    "MLX_USE_CUDA_GRAPHS",
    "MLX_CUDA_GRAPH_CACHE_SIZE",
    "MLX_CUDA_SDPA_CACHE_SIZE",
    "MLX_MAX_OPS_PER_BUFFER",
    "MLX_MAX_MB_PER_BUFFER",
    "MLX_PTX_CACHE_DIR",
    "MLXCEL_QMM_TILE_M",
    "MLXCEL_QMV_MULTIROW_MAX_ROWS",
];

/// Set `name` for Rust and for native code.
///
/// # Panics
///
/// Where `std::env::set_var` panics: an empty name, a name holding `=` or NUL,
/// a value holding NUL, or a value the OS refuses.
///
/// # Safety
///
/// The contract of `std::env::set_var`: no other thread may read or write the
/// environment at the same time, native reads through `getenv` included. On
/// Windows, replacing a value frees the string an earlier `getenv` returned.
pub unsafe fn set_native_var(name: &str, value: impl AsRef<OsStr>) {
    let value = value.as_ref();
    // The C runtime first. `_wputenv_s` also writes the OS block, and with an
    // empty value it removes the variable there, so `std::env` must run last
    // for the value it reads back to be the value set.
    #[cfg(windows)]
    crt::set(name, value);
    // SAFETY: the caller upholds the contract documented above.
    unsafe { std::env::set_var(name, value) };
}

/// Unset `name` for Rust and for native code.
///
/// # Panics
///
/// Where `std::env::remove_var` panics: an empty name, or a name holding `=`
/// or NUL.
///
/// # Safety
///
/// The same contract as [`set_native_var`].
pub unsafe fn remove_native_var(name: &str) {
    // An empty value is how `_wputenv_s` removes a variable.
    #[cfg(windows)]
    crt::set(name, OsStr::new(""));
    // SAFETY: the caller upholds the contract documented above.
    unsafe { std::env::remove_var(name) };
}

/// `name` as native code reads it: `std::getenv` in the bridge, raw bytes, in
/// the active code page on Windows. `None` when unset.
#[must_use]
pub fn native_getenv(name: &str) -> Option<Vec<u8>> {
    let mut value = Vec::new();
    crate::ffi::native_env_read(name, &mut value).then_some(value)
}

/// `name` as MLX reads an integer setting: `mlx::core::env::get_var`, inside
/// the MLX library, which is `atoi` over `getenv`.
#[must_use]
pub fn mlx_env_int(name: &str, default_value: i32) -> i32 {
    crate::ffi::mlx_env_get_var_int(name, default_value)
}

/// Log [`AUDITED_NATIVE_VARS`] at debug level as native code reads them.
///
/// This reads the C runtime, not `std::env`, so the line shows what MLX gets.
/// Call it after the writes it reports; it reads the environment, so it has
/// the same threading contract as any other `getenv`.
pub fn log_effective_native_env() {
    for name in AUDITED_NATIVE_VARS {
        let native = native_getenv(name);
        let shown = native
            .as_deref()
            .map_or("<unset>".into(), String::from_utf8_lossy);
        tracing::debug!(variable = name, native_getenv = %shown, "native environment");
    }
}

#[cfg(windows)]
mod crt {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;

    /// `_MAX_ENV`. `_wputenv_s` ends the process on a value this long or
    /// longer rather than returning an error: measured at 32767 on UCRT.
    const MAX_ENV: usize = 32767;

    unsafe extern "C" {
        fn _wputenv_s(name: *const u16, value: *const u16) -> i32;
    }

    pub(super) fn set(name: &str, value: &OsStr) {
        let mut wide_name: Vec<u16> = OsStr::new(name).encode_wide().collect();
        let mut wide_value: Vec<u16> = value.encode_wide().collect();
        // What `std::env` panics on. Leave the report to it.
        let malformed = wide_name.is_empty()
            || wide_name.iter().any(|&c| c == 0 || c == u16::from(b'='))
            || wide_value.contains(&0);
        if malformed {
            return;
        }
        if wide_name.len() >= MAX_ENV || wide_value.len() >= MAX_ENV {
            eprintln!("mlxcel: {name} is too long for the C runtime; native code will not see it");
            return;
        }
        wide_name.push(0);
        wide_value.push(0);
        // SAFETY: both pointers are NUL-terminated and live for the call. The
        // caller of the public function upholds the threading contract.
        let error = unsafe { _wputenv_s(wide_name.as_ptr(), wide_value.as_ptr()) };
        if error != 0 {
            eprintln!(
                "mlxcel: could not set {name} in the C runtime (errno {error}); \
                 native code will not see it"
            );
        }
    }
}
