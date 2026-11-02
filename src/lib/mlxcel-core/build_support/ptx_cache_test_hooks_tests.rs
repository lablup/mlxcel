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

use super::*;

fn out_dir(profile_path: &str) -> std::path::PathBuf {
    Path::new(profile_path).join("build/mlxcel-core-0123456789abcdef/out")
}

#[test]
fn release_builds_leave_the_hooks_out() {
    for target in [
        "/src/target/release",
        "/src/target/aarch64-unknown-linux-gnu/release",
    ] {
        let dir = out_dir(target);
        assert_eq!(profile_dir_name(&dir), Some("release"));
        assert!(!test_hooks_enabled(&dir), "{}", dir.display());
    }
}

#[test]
fn test_profiles_compile_the_hooks_in() {
    for target in [
        "/src/target/test-fast",
        "/src/target/debug",
        "/src/target/x86_64-unknown-linux-gnu/test-fast",
        "relative/target/debug",
    ] {
        let dir = out_dir(target);
        assert!(test_hooks_enabled(&dir), "{}", dir.display());
    }
}

#[test]
fn other_profiles_leave_the_hooks_out() {
    for profile in [
        "bench",
        "dist",
        "profiling",
        "release-with-debug",
        "Debug",
        "test_fast",
    ] {
        let dir = out_dir(&format!("/src/target/{profile}"));
        assert_eq!(profile_dir_name(&dir), Some(profile));
        assert!(!test_hooks_enabled(&dir), "{profile}");
    }
}

#[test]
fn an_unrecognized_layout_leaves_the_hooks_out() {
    for dir in [
        "/src/target/debug/build/mlxcel-core-0123/output",
        "/src/target/debug/builds/mlxcel-core-0123/out",
        "/src/target/test-fast/out",
        "debug/out",
        "out",
        "",
        "/",
    ] {
        let dir = Path::new(dir);
        assert!(!test_hooks_enabled(dir), "{}", dir.display());
    }
}
