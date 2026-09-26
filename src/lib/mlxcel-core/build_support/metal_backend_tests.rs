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

use super::resolve_metal_backend;

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
