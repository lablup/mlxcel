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

//! The gated-MLP activation a Nemotron-NAS checkpoint declares in `hidden_act`.
//!
//! Parsed once at load. Before #2548 the MLP matched the string per call and
//! sent every name it did not know, including the tanh GELU names `gelu_new`
//! and `gelu_fast`, to SiLU. GELU names now go through
//! [`GeluActivation::from_hf_name`] and any other unknown name is a load error.

use mlxcel_core::utils::GeluActivation;
use mlxcel_core::{MlxArray, UniquePtr, gelu, relu, silu};

/// Used by: `NASMLP` in `nemotron_nas.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NasActivation {
    Silu,
    Relu,
    Gelu(GeluActivation),
}

impl NasActivation {
    /// Map a Hugging Face `hidden_act` onto the activation `ACT2FN` computes,
    /// or reject it.
    pub(super) fn from_hf_name(name: &str) -> Result<Self, String> {
        match name.trim().to_ascii_lowercase().as_str() {
            "silu" | "swish" => Ok(Self::Silu),
            "relu" => Ok(Self::Relu),
            other => GeluActivation::from_hf_name(other)
                // The tanh names run as the compiled f32 kernel the other
                // families switched to tanh in #2423 / #2548 use.
                .map(|act| Self::Gelu(act.in_f32_compiled()))
                .ok_or_else(|| {
                    format!(
                        "Nemotron-NAS: unsupported hidden_act {name:?}; expected silu, relu, \
                         or a GELU variant (gelu, gelu_new, gelu_fast, gelu_pytorch_tanh, \
                         quick_gelu)"
                    )
                }),
        }
    }

    pub(super) fn apply(self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Silu => silu(x),
            Self::Relu => relu(x),
            // Keep the erf entry point this MLP always used for `gelu`.
            Self::Gelu(GeluActivation::Erf) => gelu(x),
            Self::Gelu(act) => act.apply(x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_gelu_names_through_the_shared_selector() {
        assert_eq!(
            NasActivation::from_hf_name("gelu"),
            Ok(NasActivation::Gelu(GeluActivation::Erf))
        );
        for name in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"] {
            assert_eq!(
                NasActivation::from_hf_name(name),
                Ok(NasActivation::Gelu(GeluActivation::TanhF32Compiled)),
                "{name} is the tanh GELU, not SiLU"
            );
        }
        assert_eq!(
            NasActivation::from_hf_name("quick_gelu"),
            Ok(NasActivation::Gelu(GeluActivation::Fast))
        );
    }

    #[test]
    fn keeps_silu_and_relu() {
        assert_eq!(NasActivation::from_hf_name("silu"), Ok(NasActivation::Silu));
        assert_eq!(NasActivation::from_hf_name("relu"), Ok(NasActivation::Relu));
    }

    #[test]
    fn rejects_unknown_names_instead_of_using_silu() {
        let err = NasActivation::from_hf_name("tanh").unwrap_err();
        assert!(err.contains("unsupported hidden_act"), "{err}");
        assert!(NasActivation::from_hf_name("").is_err());
    }
}
