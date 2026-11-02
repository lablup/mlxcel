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

//! The current directory is out of the DLL search once a GPU build has
//! initialized (issue #2641).
//!
//! The CUDA libraries and the HIP runtime open some DLLs by plain name through
//! the standard search. The overlays' `delayload.cpp` calls
//! `SetDllDirectoryW(L"")` before the first GPU DLL loads. No path these tests
//! can drive requests those names, so the check is direct: after init, in a
//! child process whose current directory holds a valid DLL under each name,
//! `LoadLibraryW` must not take it. Restoring the default search then loads the
//! same decoys from that directory, which shows the setting is what kept them
//! out and that the copies are loadable.
//!
//! Needs the GPU runtime the build uses, so it runs by hand on a GPU host:
//!
//! ```sh
//! cargo test --release --features cuda --test windows_dll_search_cwd
//! ```

#![cfg(all(windows, any(feature = "cuda", feature = "rocm")))]

use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::Command;

const CHILD_TEST: &str = "dll_search_child";
const CHILD_ENV: &str = "MLXCEL_TEST_DLL_SEARCH_CHILD";

/// Named by the HIP runtime; neither the ROCm SDK nor the driver ships them.
const ROCM_DECOYS: &[&str] = &[
    "hsa-amd-aqlprofile64.dll",
    "hsa-ext-finalize64.dll",
    "dtif64a.dll",
];
/// Named by the CUDA libraries, with the two optional cuDNN sub-libraries that
/// a bundle need not ship. `nvcuda.dll` is checked apart, since it must load.
const CUDA_DECOYS: &[&str] = &[
    "nvcudart_hybrid64.dll",
    "nvcudart_hybrida64.dll",
    "zlibwapi.dll",
    "cask_profile_interface.dll",
    "nvtileiras64.dll",
    "cudnn_ext64_9.dll",
    "cudnn_engines_tensor_ir64_9.dll",
];
const CONTROL_NAME: &str = "mlxcel_dll_search_control.dll";
const NVCUDA: &str = "nvcuda.dll";

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut std::ffi::c_void;
    fn FreeLibrary(module: *mut std::ffi::c_void) -> i32;
    fn SetDllDirectoryW(path: *const u16) -> i32;
    fn GetModuleFileNameW(module: *mut std::ffi::c_void, buf: *mut u16, size: u32) -> u32;
}

fn wide(text: &str) -> Vec<u16> {
    OsStr::new(text).encode_wide().chain([0]).collect()
}

/// `LoadLibraryW(name)` as the path it loaded from, or `None` when it failed.
/// The module stays loaded until `release`.
fn load(name: &str) -> Option<(*mut std::ffi::c_void, PathBuf)> {
    let name = wide(name);
    // SAFETY: `name` is NUL-terminated and outlives the call. The DLLs loaded
    // here are this test's copies of a system DLL or the driver's own.
    let module = unsafe { LoadLibraryW(name.as_ptr()) };
    if module.is_null() {
        return None;
    }
    let mut buf = vec![0u16; 4096];
    // SAFETY: `buf` holds `buf.len()` elements and `module` is a live module.
    let len = unsafe { GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32) };
    let path = String::from_utf16_lossy(&buf[..len as usize]);
    Some((module, PathBuf::from(path)))
}

fn release(module: *mut std::ffi::c_void) {
    // SAFETY: `module` came from `LoadLibraryW` and nothing uses it afterwards.
    unsafe { FreeLibrary(module) };
}

fn in_dir(path: &Path, dir: &Path) -> bool {
    // canonicalize() returns a verbatim path and GetModuleFileNameW a plain one.
    let lower = |p: &Path| {
        let text = p.to_string_lossy().to_lowercase();
        text.strip_prefix(r"\\?\")
            .map(str::to_owned)
            .unwrap_or(text)
    };
    lower(path).starts_with(&lower(dir))
}

/// The child: runs the real init, then asks for every decoy by name.
#[test]
fn dll_search_child() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }
    let cwd = std::env::current_dir()
        .expect("current directory")
        .canonicalize()
        .expect("canonical current directory");
    // Goes through the first GPU DLL load on both backends.
    let backend = mlxcel_core::hardware::gpu_backend_kind();
    eprintln!("backend: {backend:?}");

    let mut decoys: Vec<&str> = ROCM_DECOYS.to_vec();
    decoys.extend_from_slice(CUDA_DECOYS);
    for name in decoys {
        if let Some((module, path)) = load(name) {
            assert!(
                !in_dir(&path, &cwd),
                "{name} was loaded from the current directory: {}",
                path.display()
            );
            // A real copy elsewhere on PATH is allowed: PATH stays in the search.
            eprintln!("{name}: not a decoy, loaded from {}", path.display());
            release(module);
        }
    }

    if let Some((module, path)) = load(NVCUDA) {
        assert!(
            !in_dir(&path, &cwd),
            "{NVCUDA} was loaded from the current directory: {}",
            path.display()
        );
        if cfg!(feature = "cuda") {
            let system_root = std::env::var("SystemRoot").expect("SystemRoot");
            assert!(
                in_dir(&path, Path::new(&system_root)),
                "{NVCUDA} must stay on the driver's copy in System32: {}",
                path.display()
            );
        }
        release(module);
    } else {
        assert!(
            !cfg!(feature = "cuda") || backend != mlxcel_core::hardware::GpuBackendKind::Cuda,
            "{NVCUDA} did not load although CUDA is the running backend"
        );
    }

    // Control: with the default search restored the decoy loads from here, so
    // the failures above come from the setting, not from a bad copy.
    // SAFETY: a null pointer restores the default search order.
    let restored = unsafe { SetDllDirectoryW(std::ptr::null()) };
    assert_ne!(restored, 0, "SetDllDirectoryW(NULL) failed");
    let (module, path) = load(CONTROL_NAME)
        .expect("the control decoy must load from the current directory under the default search");
    assert!(
        in_dir(&path, &cwd),
        "the control decoy came from {}",
        path.display()
    );
    release(module);
}

#[test]
fn current_directory_is_not_searched_after_init() {
    if std::env::var_os(CHILD_ENV).is_some() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let system_root = std::env::var("SystemRoot").expect("SystemRoot");
    // Any small DLL that imports nothing unusual works as a valid decoy.
    let source = Path::new(&system_root).join("System32").join("version.dll");
    let mut names: Vec<&str> = ROCM_DECOYS.to_vec();
    names.extend_from_slice(CUDA_DECOYS);
    names.push(NVCUDA);
    names.push(CONTROL_NAME);
    for name in names {
        std::fs::copy(&source, dir.path().join(name)).expect("copy decoy");
    }
    let output = Command::new(std::env::current_exe().expect("test binary"))
        .args([CHILD_TEST, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, "1")
        .current_dir(dir.path())
        .output()
        .expect("run the child");
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success(), "child failed:\n{text}");
    assert!(
        text.contains("1 passed"),
        "the child test did not run:\n{text}"
    );
}
