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

use super::{gpu_backend_enabled, resolve_metal_backend};

#[test]
fn macos_default_enables_metal_without_a_cargo_feature() {
    assert_eq!(resolve_metal_backend(true, None), Ok(true));
}

#[test]
fn explicit_macos_backend_override_controls_both_consumers() {
    for value in ["1", "on", "true", "yes", " ON ", "True"] {
        assert_eq!(resolve_metal_backend(true, Some(value)), Ok(true));
    }
    for value in ["0", "off", "false", "no", " OFF ", "False"] {
        assert_eq!(resolve_metal_backend(true, Some(value)), Ok(false));
    }
}

#[test]
fn non_macos_never_enables_metal_even_with_an_override() {
    for value in [None, Some("ON"), Some("OFF"), Some("invalid")] {
        assert_eq!(resolve_metal_backend(false, value), Ok(false));
    }
}

#[test]
fn invalid_macos_override_retains_the_named_build_error() {
    let error = resolve_metal_backend(true, Some("maybe")).unwrap_err();
    assert!(error.contains(r#"Invalid MLXCEL_BUILD_METAL value "maybe""#));
    assert!(error.contains("1/0, on/off, true/false, yes/no"));
}

#[test]
fn cpu_only_build_has_no_gpu_backend() {
    // A Linux build with no GPU feature: the bridge must not reference
    // `copy_gpu_inplace` (#2108).
    assert!(!gpu_backend_enabled(false, false, false));
}

#[test]
fn any_single_gpu_backend_enables_the_gpu_bridge_paths() {
    assert!(gpu_backend_enabled(true, false, false));
    assert!(gpu_backend_enabled(false, true, false));
    assert!(gpu_backend_enabled(false, false, true));
}

#[test]
fn macos_with_metal_disabled_and_no_other_backend_is_cpu_only() {
    let metal = resolve_metal_backend(true, Some("OFF")).unwrap();
    assert!(!gpu_backend_enabled(metal, false, false));
}
