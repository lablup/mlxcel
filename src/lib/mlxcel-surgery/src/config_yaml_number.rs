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

//! Preserve non-finite values through Serde's internally tagged enum buffer.
//!
//! serde-saphyr represents typeless non-finite floats as canonical strings.
//! Recover only those spellings so the existing operation validators can emit
//! their specific finite-value errors; ordinary quoted numbers remain invalid.

use serde::Deserialize;

pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Number {
        Float(f32),
        Special(String),
    }
    match Number::deserialize(deserializer)? {
        Number::Float(value) => Ok(value),
        Number::Special(value) => match value.as_str() {
            ".nan" => Ok(f32::NAN),
            ".inf" => Ok(f32::INFINITY),
            "-.inf" => Ok(f32::NEG_INFINITY),
            _ => Err(serde::de::Error::custom("expected a number")),
        },
    }
}
