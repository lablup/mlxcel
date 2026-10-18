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

//! Lift a transformers 5 `rope_parameters` block into the flat RoPE keys.
//!
//! transformers 5 saves `rope_theta`, `rope_scaling` and
//! `partial_rotary_factor` only under a nested `rope_parameters` block (for
//! example `{"rope_theta": 1000000, "rope_type": "default"}`) and drops the flat
//! keys. A config struct that reads the flat `rope_theta` with a serde default
//! would then silently run at the default base, decoding fluently at short
//! lengths and drifting at long ones (#2384, #2443).
//!
//! The lift lives here rather than in the model crate so that drafter configs
//! in this crate share it. The model crate re-exports it from
//! `models::rope_utils`.

use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value};

/// Copy a nested `rope_parameters` block into the flat `rope_theta`,
/// `partial_rotary_factor` and `rope_scaling` keys, when the flat keys are
/// absent.
///
/// A flat key that is present and non-null always wins over the nested one.
///
/// `rope_scaling` is filled (unless the config already has a non-null one)
/// when the block names a non-default scheme or carries keys beyond the base
/// and the scheme name, such as the MRoPE `mrope_section` that transformers 4
/// kept in `rope_scaling` and transformers 5 moved here. It is the block
/// without `rope_theta`, with the scheme under `type` only. The scheme is read
/// `type` first, then `rope_type`, the order `RopeScalingSpec` and upstream
/// use. `type` is the key every reader in the tree accepts, while carrying
/// both spellings would turn a `#[serde(alias = "type", alias = "rope_type")]`
/// field into a `duplicate field` error.
///
/// The per-layer-type form (every value an object) is read too. For Gemma 3,
/// Gemma 3n and OLMo 3 (`{"full_attention": {...}, "sliding_attention":
/// {...}}`) the `full_attention` block supplies the keys above and the
/// `sliding_attention` base becomes `rope_local_base_freq`. For DeepSeek-V4
/// (`{"main": {...}, "compress": {...}}`) `main` supplies the base, the
/// `compress` base becomes `compress_rope_theta`, and the `compress` YaRN block
/// becomes `rope_scaling`, which is where transformers 4 kept it.
///
/// Upstream: ml-explore/mlx-lm#1898 (qwen3) and ml-explore/mlx-lm#1605
/// (granitemoehybrid).
pub fn lift_rope_parameters(config: &mut Value) {
    let Some(obj) = config.as_object_mut() else {
        return;
    };
    let Some(params) = obj.get("rope_parameters").and_then(Value::as_object) else {
        return;
    };
    let block_at = |key: &str| params.get(key).and_then(Value::as_object).cloned();
    let per_layer_type =
        !params.is_empty() && params.values().all(|v| v.is_object() || v.is_null());
    let (main, sliding, compress) = if per_layer_type {
        (
            block_at("full_attention")
                .or_else(|| block_at("main"))
                .unwrap_or_default(),
            block_at("sliding_attention"),
            block_at("compress"),
        )
    } else {
        (params.clone(), None, None)
    };

    copy_number_if_absent(obj, "rope_theta", main.get("rope_theta"));
    copy_number_if_absent(
        obj,
        "partial_rotary_factor",
        main.get("partial_rotary_factor"),
    );
    if let Some(sliding) = &sliding {
        copy_number_if_absent(obj, "rope_local_base_freq", sliding.get("rope_theta"));
    }
    if let Some(compress) = &compress {
        copy_number_if_absent(obj, "compress_rope_theta", compress.get("rope_theta"));
    }

    if obj.get("rope_scaling").is_some_and(|v| !v.is_null()) {
        return;
    }
    if let Some(scaling) = scaling_block(main).or_else(|| compress.and_then(scaling_block)) {
        obj.insert("rope_scaling".to_string(), Value::Object(scaling));
    }
}

/// The `rope_scaling` a nested block stands for, or `None` when it names the
/// default scheme and carries nothing beyond the base.
fn scaling_block(mut block: Map<String, Value>) -> Option<Map<String, Value>> {
    let kind = block
        .get("type")
        .and_then(Value::as_str)
        .or_else(|| block.get("rope_type").and_then(Value::as_str))
        .map(str::to_string);
    let has_extra_keys = block.keys().any(|key| {
        !matches!(
            key.as_str(),
            "rope_theta" | "rope_type" | "type" | "partial_rotary_factor"
        )
    });
    if !(kind.as_deref().is_some_and(|k| k != "default") || has_extra_keys) {
        return None;
    }
    block.remove("rope_theta");
    block.remove("rope_type");
    match kind {
        Some(kind) => block.insert("type".to_string(), Value::String(kind)),
        None => block.remove("type"),
    };
    Some(block)
}

fn copy_number_if_absent(obj: &mut Map<String, Value>, key: &str, nested: Option<&Value>) {
    if obj.get(key).is_none_or(Value::is_null)
        && let Some(value) = nested.filter(|v| v.is_number())
    {
        obj.insert(key.to_string(), value.clone());
    }
}

/// Deserialize a config through [`lift_rope_parameters`].
///
/// A config struct whose `Deserialize` derive carries `#[serde(remote =
/// "Self")]` (which turns the derive into an inherent `Type::deserialize`
/// function instead of the trait impl) gets the trait impl from
/// [`impl_rope_lifting_deserialize!`](crate::impl_rope_lifting_deserialize),
/// which calls this. Every `serde_json::from_value` / `from_str` /
/// nested-field parse of that type then sees the lifted keys, so a new parse
/// site cannot miss the lift.
pub fn deserialize_lifting_rope_parameters<'de, D, T>(
    deserializer: D,
    parse: impl FnOnce(Value) -> Result<T, serde_json::Error>,
) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
{
    let mut value = Value::deserialize(deserializer)?;
    lift_rope_parameters(&mut value);
    parse(value).map_err(serde::de::Error::custom)
}

/// Implement `Deserialize` for config structs through
/// [`deserialize_lifting_rope_parameters`].
///
/// Each type must derive `Deserialize` with `#[serde(remote = "Self")]`:
///
/// ```ignore
/// #[derive(Debug, Clone, Deserialize)]
/// #[serde(remote = "Self")]
/// pub struct ModelArgs {
///     #[serde(default = "default_rope_theta")]
///     pub rope_theta: f32,
/// }
/// mlxcel_core::impl_rope_lifting_deserialize!(ModelArgs);
/// ```
#[macro_export]
macro_rules! impl_rope_lifting_deserialize {
    ($($ty:ty),+ $(,)?) => {$(
        impl<'de> ::serde::Deserialize<'de> for $ty {
            fn deserialize<D: ::serde::Deserializer<'de>>(
                deserializer: D,
            ) -> ::core::result::Result<Self, D::Error> {
                $crate::rope_parameters::deserialize_lifting_rope_parameters(
                    deserializer,
                    <$ty>::deserialize,
                )
            }
        }
    )+};
}

#[cfg(test)]
#[path = "rope_parameters_tests.rs"]
mod tests;
