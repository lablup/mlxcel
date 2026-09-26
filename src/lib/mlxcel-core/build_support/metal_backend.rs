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

//! Resolve the backend actually compiled into MLX, independently of Cargo features.

/// Used by: mlxcel-core's CMake configuration and C++ runtime-control bridge.
/// macOS enables Metal by default; other platforms do not build that backend.
pub fn resolve_metal_backend(is_macos: bool, override_value: Option<&str>) -> Result<bool, String> {
    if !is_macos {
        return Ok(false);
    }
    match override_value {
        None => Ok(true),
        Some(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "on" | "true" | "yes" => Ok(true),
            "0" | "off" | "false" | "no" => Ok(false),
            _ => Err(format!(
                "Invalid MLXCEL_BUILD_METAL value {value:?}. Expected one of: 1/0, on/off, true/false, yes/no."
            )),
        },
    }
}

#[cfg(test)]
#[path = "metal_backend_tests.rs"]
mod tests;
