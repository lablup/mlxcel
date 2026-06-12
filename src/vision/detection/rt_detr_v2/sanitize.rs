// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! HuggingFace -> MLX weight-key translation for RT-DETRv2.
//!
//! Mirrors the rename pipeline in
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/rt_detr_v2/convert.py (the single
//! source of truth upstream). Two transformations are applied per key:
//!
//!   1. Name rewrite: strip the `model.` prefix, then apply submodule-specific
//!      substring/prefix rules in order.
//!   2. Conv2d weight layout transpose: PyTorch `(out, in, kH, kW)` ->
//!      MLX NHWC `(out, kH, kW, in)` for any `*.conv.weight` 4D tensor.
//!
//! Keys matching `DROP_PATTERNS` (`*.num_batches_tracked`) are dropped — MLX
//! BatchNorm has no slot for that counter.
//!
//! Pre-converted mlx-community checkpoints already carry MLX-layout keys (and
//! NHWC conv weights), so [`needs_sanitize`] returns `false` for them and the
//! pipeline is skipped — re-running it would double-transpose conv weights.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const HF_PREFIX: &str = "model.";

/// Decide whether a freshly-loaded weight map is in raw HuggingFace layout and
/// therefore needs the rename + transpose pipeline.
///
/// MLX-layout checkpoints (produced by `convert.py` or shipped as
/// `mlx-community/*-mlx-*`) surface keys under `vision.backbone.` /
/// `decoder.layers.`; raw HF checkpoints surface `model.backbone.` /
/// `backbone.model.` / `model.decoder.` and the `*.convolution.` /
/// `*.normalization.` field names. We sniff a few unambiguous markers.
pub fn needs_sanitize(weights: &WeightMap) -> bool {
    let mut has_mlx_marker = false;
    let mut has_hf_marker = false;
    for key in weights.keys() {
        if key.starts_with("vision.backbone.") || key.starts_with("vision.hybrid_encoder.") {
            has_mlx_marker = true;
        }
        if key.starts_with("model.backbone.")
            || key.starts_with("backbone.model.")
            || key.contains(".convolution.")
            || key.contains(".normalization.")
            || key.starts_with("model.decoder.")
            || key.starts_with("model.encoder.")
        {
            has_hf_marker = true;
        }
        // Early out once both decisions are unambiguous.
        if has_mlx_marker {
            return false;
        }
        if has_hf_marker {
            return true;
        }
    }
    // No markers at all: treat as already-MLX (no-op) to avoid corrupting an
    // unexpected layout.
    has_hf_marker
}

/// True when `key` (already stripped of the `model.` prefix) should be dropped.
fn should_drop(stripped: &str) -> bool {
    stripped.ends_with(".num_batches_tracked")
}

/// Strip the leading `model.` prefix HF uses on the whole module tree.
fn strip_prefix(key: &str) -> &str {
    key.strip_prefix(HF_PREFIX).unwrap_or(key)
}

/// Apply the ordered rename rules to a prefix-stripped key.
///
/// Order matters: later rules can rely on earlier ones having fired (e.g. the
/// generic `encoder.` -> `vision.hybrid_encoder.` rule runs *after* the more
/// specific `encoder.encoder.` -> AIFI rule and the `encoder_input_proj`
/// rules). This is a 1:1 port of `convert.RENAMES`.
fn rename_stripped(key: &str) -> String {
    let mut out = key.to_string();

    // Backbone: HF wraps the ResNet body in `backbone.model.X`.
    out = replace_prefix(&out, "backbone.model.", "vision.backbone.");

    // vd downsampling shortcut: Sequential[AvgPool, ShortCut] indexes the
    // inner ShortCut at `.1.` (AvgPool at `.0.` has no params).
    out = out.replace(".shortcut.1.", ".shortcut.proj.");

    // RTDetrResNetConvLayer field names.
    out = out.replace(".convolution.", ".conv.");
    out = out.replace(".normalization.", ".bn.");

    // AIFI keys on disk are prefixed `encoder.encoder.` (older HF naming).
    out = replace_prefix(&out, "encoder.encoder.", "vision.hybrid_encoder.aifi.");

    // encoder_input_proj is Sequential[Conv, BN]: `.{N}.0.X` / `.{N}.1.X`.
    out = rename_indexed_seq(&out, "encoder_input_proj.", "vision.encoder_input_proj.");

    // Hybrid encoder body (FPN/PAN/laterals/downsamples): all under `encoder.*`.
    out = replace_prefix(&out, "encoder.", "vision.hybrid_encoder.");

    // RTDetrV2ConvNormLayer field rename (`.norm.` -> `.bn.`).
    out = out.replace(".norm.", ".bn.");

    // decoder_input_proj is also Sequential[Conv, BN].
    out = rename_indexed_seq(&out, "decoder_input_proj.", "decoder_input_proj.");

    // enc_output is Sequential[Linear, LayerNorm].
    out = replace_prefix(&out, "enc_output.0.", "enc_output.fc.");
    out = replace_prefix(&out, "enc_output.1.", "enc_output.ln.");

    out
}

/// `replace_prefix(s, from, to)` rewrites `s` only if it starts with `from`.
fn replace_prefix(s: &str, from: &str, to: &str) -> String {
    match s.strip_prefix(from) {
        Some(rest) => format!("{to}{rest}"),
        None => s.to_string(),
    }
}

/// Rewrite a `Sequential[Conv, BN]` block stored as `{base}{N}.0.X` /
/// `{base}{N}.1.X` into `{dest}{N}.conv.X` / `{dest}{N}.bn.X`.
///
/// Mirrors the two `^{base}(\d+)\.0\.` / `\.1\.` regex rules in `convert.py`,
/// but without a regex engine: it only fires when the key starts with `base`
/// followed by an integer index and then `.0.` or `.1.`.
fn rename_indexed_seq(key: &str, base: &str, dest: &str) -> String {
    let Some(rest) = key.strip_prefix(base) else {
        return key.to_string();
    };
    // rest = "{N}.0.X" or "{N}.1.X" (or something else, left untouched).
    let Some(dot) = rest.find('.') else {
        return key.to_string();
    };
    let (idx, tail) = rest.split_at(dot);
    if idx.is_empty() || !idx.bytes().all(|b| b.is_ascii_digit()) {
        return key.to_string();
    }
    if let Some(after) = tail.strip_prefix(".0.") {
        format!("{dest}{idx}.conv.{after}")
    } else if let Some(after) = tail.strip_prefix(".1.") {
        format!("{dest}{idx}.bn.{after}")
    } else {
        key.to_string()
    }
}

/// Translate one HF key to its MLX form (prefix-strip + rename rules).
pub fn rename_key(key: &str) -> String {
    rename_stripped(strip_prefix(key))
}

/// Run the full sanitize pipeline over `weights`, returning a new map with
/// MLX-layout keys, conv weights transposed to NHWC, and `num_batches_tracked`
/// counters dropped. Consumes the input map.
pub fn sanitize(weights: WeightMap) -> WeightMap {
    let mut out: WeightMap = WeightMap::with_capacity(weights.len());
    for (k, v) in weights {
        let stripped = strip_prefix(&k);
        if should_drop(stripped) {
            continue;
        }
        let new_key = rename_stripped(stripped);
        let value = maybe_transpose_conv(&new_key, v);
        out.insert(new_key, value);
    }
    out
}

/// PyTorch conv weights are `(out, in, kH, kW)`; MLX wants NHWC
/// `(out, kH, kW, in)`. Transpose any 4D `*.conv.weight`.
fn maybe_transpose_conv(key: &str, value: UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    if key.ends_with(".conv.weight") {
        let shape = mlxcel_core::array_shape(&value);
        if shape.len() == 4 {
            return mlxcel_core::transpose_axes(&value, &[0, 2, 3, 1]);
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_model_prefix() {
        assert_eq!(strip_prefix("model.foo.bar"), "foo.bar");
        assert_eq!(strip_prefix("foo.bar"), "foo.bar");
    }

    #[test]
    fn backbone_rename() {
        assert_eq!(
            rename_key("model.backbone.model.embedder.embedder.0.convolution.weight"),
            "vision.backbone.embedder.embedder.0.conv.weight"
        );
        assert_eq!(
            rename_key("model.backbone.model.embedder.embedder.0.normalization.running_mean"),
            "vision.backbone.embedder.embedder.0.bn.running_mean"
        );
    }

    #[test]
    fn vd_shortcut_rename() {
        assert_eq!(
            rename_key(
                "model.backbone.model.encoder.stages.1.layers.0.shortcut.1.convolution.weight"
            ),
            "vision.backbone.encoder.stages.1.layers.0.shortcut.proj.conv.weight"
        );
    }

    #[test]
    fn aifi_rename() {
        assert_eq!(
            rename_key("model.encoder.encoder.0.layers.0.self_attn.q_proj.weight"),
            "vision.hybrid_encoder.aifi.0.layers.0.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn encoder_input_proj_seq_rename() {
        assert_eq!(
            rename_key("model.encoder_input_proj.0.0.weight"),
            "vision.encoder_input_proj.0.conv.weight"
        );
        assert_eq!(
            rename_key("model.encoder_input_proj.2.1.running_var"),
            "vision.encoder_input_proj.2.bn.running_var"
        );
    }

    #[test]
    fn hybrid_encoder_body_and_norm_rename() {
        assert_eq!(
            rename_key("model.encoder.lateral_convs.0.norm.weight"),
            "vision.hybrid_encoder.lateral_convs.0.bn.weight"
        );
    }

    #[test]
    fn decoder_input_proj_and_enc_output_rename() {
        assert_eq!(
            rename_key("model.decoder_input_proj.1.0.weight"),
            "decoder_input_proj.1.conv.weight"
        );
        assert_eq!(
            rename_key("model.decoder_input_proj.1.1.bias"),
            "decoder_input_proj.1.bn.bias"
        );
        assert_eq!(
            rename_key("model.enc_output.0.weight"),
            "enc_output.fc.weight"
        );
        assert_eq!(rename_key("model.enc_output.1.bias"), "enc_output.ln.bias");
    }

    #[test]
    fn decoder_layers_passthrough() {
        // Decoder layer keys have no `model.` content rules beyond prefix strip.
        assert_eq!(
            rename_key("model.decoder.layers.0.self_attn.q_proj.weight"),
            "decoder.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            rename_key("model.decoder.bbox_embed.0.layers.2.weight"),
            "decoder.bbox_embed.0.layers.2.weight"
        );
    }

    #[test]
    fn drop_num_batches_tracked() {
        assert!(should_drop(
            "vision.backbone.embedder.embedder.0.bn.num_batches_tracked"
        ));
        assert!(!should_drop(
            "vision.backbone.embedder.embedder.0.bn.running_mean"
        ));
    }

    #[test]
    fn needs_sanitize_detects_layout() {
        // We can't construct MlxArrays cheaply here; test the key-marker logic
        // through a small synthetic map keyed by name only is enough because
        // needs_sanitize never dereferences the values.
        // (Values are required by the type, so build trivial scalars.)
        let mut mlx_map: WeightMap = WeightMap::new();
        mlx_map.insert(
            "vision.backbone.embedder.embedder.0.conv.weight".to_string(),
            mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32),
        );
        assert!(!needs_sanitize(&mlx_map));

        let mut hf_map: WeightMap = WeightMap::new();
        hf_map.insert(
            "model.backbone.model.embedder.embedder.0.convolution.weight".to_string(),
            mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32),
        );
        assert!(needs_sanitize(&hf_map));
    }
}
