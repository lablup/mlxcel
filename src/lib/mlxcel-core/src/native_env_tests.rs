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

//! Reads go through the bridge and through MLX, so these fail if the Rust
//! binary's C runtime is not the one MLX reads (#2803).

use std::ffi::{OsStr, OsString};

use crate::native_env::{mlx_env_int, native_getenv, remove_native_var, set_native_var};
use crate::test_support::env_lock::env_lock;

const ABSENT: i32 = -7;

/// Unsets the variable a test wrote, for the next run in this process.
struct Unset(&'static str);

impl Drop for Unset {
    fn drop(&mut self) {
        // SAFETY: dropped before the test's `env_lock` guard, declared first.
        unsafe { remove_native_var(self.0) };
    }
}

#[test]
fn a_set_value_reaches_rust_the_bridge_and_mlx() {
    let _guard = env_lock();
    const NAME: &str = "NATIVE_ENV_TEST_SET";
    let _unset = Unset(NAME);
    // SAFETY: serialized by `env_lock`.
    unsafe { set_native_var(NAME, "2000") };
    assert_eq!(std::env::var_os(NAME), Some(OsString::from("2000")));
    assert_eq!(native_getenv(NAME).as_deref(), Some(&b"2000"[..]));
    assert_eq!(mlx_env_int(NAME, ABSENT), 2000);
}

#[test]
fn overwriting_replaces_the_value_everywhere() {
    let _guard = env_lock();
    const NAME: &str = "NATIVE_ENV_TEST_OVERWRITE";
    let _unset = Unset(NAME);
    // SAFETY: serialized by `env_lock`.
    unsafe {
        set_native_var(NAME, "16");
        set_native_var(NAME, "128");
    }
    assert_eq!(std::env::var_os(NAME), Some(OsString::from("128")));
    assert_eq!(mlx_env_int(NAME, ABSENT), 128);
}

#[test]
fn removing_unsets_everywhere_and_an_unset_name_is_fine() {
    let _guard = env_lock();
    const NAME: &str = "NATIVE_ENV_TEST_REMOVE";
    // SAFETY: serialized by `env_lock`.
    unsafe {
        remove_native_var(NAME);
        set_native_var(NAME, "1");
        remove_native_var(NAME);
    }
    assert_eq!(std::env::var_os(NAME), None);
    assert_eq!(native_getenv(NAME), None);
    assert_eq!(mlx_env_int(NAME, ABSENT), ABSENT);
}

#[test]
fn an_empty_value_keeps_the_std_env_contract() {
    let _guard = env_lock();
    const NAME: &str = "NATIVE_ENV_TEST_EMPTY";
    let _unset = Unset(NAME);
    // SAFETY: serialized by `env_lock`.
    unsafe {
        set_native_var(NAME, "x");
        set_native_var(NAME, "");
    }
    assert_eq!(std::env::var_os(NAME), Some(OsString::new()));
    // The C runtime on Windows has no empty value, and the old one must be gone.
    let expected: Option<&[u8]> = if cfg!(windows) { None } else { Some(b"") };
    assert_eq!(native_getenv(NAME).as_deref(), expected);
}

#[test]
fn a_non_ascii_value_stays_exact_for_rust() {
    let _guard = env_lock();
    const NAME: &str = "NATIVE_ENV_TEST_NON_ASCII";
    let _unset = Unset(NAME);
    // Korean, a Latin letter and an emoji: no single code page holds all three.
    let value = "캐시-é-😀";
    // SAFETY: serialized by `env_lock`.
    unsafe { set_native_var(NAME, value) };
    assert_eq!(std::env::var_os(NAME).as_deref(), Some(OsStr::new(value)));
    assert_eq!(native_getenv(NAME), Some(expected_narrow(value)));
}

#[cfg(not(windows))]
fn expected_narrow(value: &str) -> Vec<u8> {
    value.as_bytes().to_vec()
}

/// `value` in the active code page, as the C runtime converts it.
#[cfg(windows)]
fn expected_narrow(value: &str) -> Vec<u8> {
    use std::os::windows::ffi::OsStrExt;
    unsafe extern "system" {
        fn WideCharToMultiByte(
            code_page: u32,
            flags: u32,
            wide: *const u16,
            wide_len: i32,
            narrow: *mut u8,
            narrow_len: i32,
            default_char: *const u8,
            used_default: *mut i32,
        ) -> i32;
    }
    let wide: Vec<u16> = OsStr::new(value).encode_wide().collect();
    let mut narrow = vec![0u8; wide.len() * 4];
    // SAFETY: the pointers and lengths describe the two live buffers; code page
    // 0 is CP_ACP, and the two optional pointers may be null.
    let written = unsafe {
        WideCharToMultiByte(
            0,
            0,
            wide.as_ptr(),
            wide.len() as i32,
            narrow.as_mut_ptr(),
            narrow.len() as i32,
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };
    assert!(written > 0, "WideCharToMultiByte failed");
    narrow.truncate(written as usize);
    narrow
}

/// The cause of #2803. If this ever fails, std started updating the C runtime
/// and the Windows half of the helper can go.
#[cfg(windows)]
#[test]
fn std_set_var_alone_does_not_reach_the_c_runtime() {
    let _guard = env_lock();
    const NAME: &str = "NATIVE_ENV_TEST_STD_ONLY";
    let _unset = Unset(NAME);
    // SAFETY: serialized by `env_lock`.
    unsafe { std::env::set_var(NAME, "1") };
    assert_eq!(std::env::var_os(NAME), Some(OsString::from("1")));
    assert_eq!(native_getenv(NAME), None);
    assert_eq!(mlx_env_int(NAME, ABSENT), ABSENT);
}

/// `_wputenv_s` ends the process on a value of 32767 or more. The test binary
/// surviving this call is the assertion, along with the old value staying.
#[cfg(windows)]
#[test]
fn an_oversized_value_does_not_end_the_process() {
    let _guard = env_lock();
    const NAME: &str = "NATIVE_ENV_TEST_OVERSIZED";
    let _unset = Unset(NAME);
    // SAFETY: serialized by `env_lock`.
    unsafe { set_native_var(NAME, "kept") };
    let oversized = "x".repeat(32767);
    // `std::env` may refuse this length by panicking, which is its report.
    // SAFETY: serialized by `env_lock`.
    let _ = std::panic::catch_unwind(|| unsafe { set_native_var(NAME, &oversized) });
    assert_eq!(native_getenv(NAME).as_deref(), Some(&b"kept"[..]));
}

/// The two startup defaults #2803 was filed for, through the real appliers.
#[cfg(feature = "cuda")]
#[test]
fn the_cuda_cache_defaults_reach_mlx() {
    let _guard = env_lock();
    for (name, apply) in [
        (
            "MLX_CUDA_GRAPH_CACHE_SIZE",
            crate::hardware::apply_cuda_graph_cache_default as fn(),
        ),
        (
            "MLX_CUDA_SDPA_CACHE_SIZE",
            crate::hardware::apply_cuda_sdpa_cache_default as fn(),
        ),
    ] {
        let before = std::env::var_os(name);
        // SAFETY: serialized by `env_lock`.
        unsafe { remove_native_var(name) };
        apply();
        let read = mlx_env_int(name, ABSENT);
        // SAFETY: serialized by `env_lock`.
        unsafe {
            match &before {
                Some(value) => set_native_var(name, value),
                None => remove_native_var(name),
            }
        }
        assert_eq!(read, 2000, "{name}");
    }
}
