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

//! Image-generation checkpoint family probe.
//!
//! A port of `_local_image_model_types` and its helpers in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/generate/image.py>
//! (`#L274-L369`). Diffusers-style checkpoints are directories of components
//! (`transformer/`, `vae/`, `text_encoder/`, ...) with a `model_index.json`
//! and usually no top-level `config.json` `model_type`, so they cannot be
//! classified the way text checkpoints are. The probe reads, in order:
//!
//! 1. `manifest.json`, which identifies Bonsai before any class name does,
//!    because Bonsai ships a `Flux2KleinPipeline` `model_index.json`;
//! 2. `model_index.json` and `config.json` (`model_type`, `_class_name` and
//!    the component class names);
//! 3. the transformer `weight_map` marker sets;
//! 4. each `<dir>/config.json`.
//!
//! The first candidate that names a family in [`IMAGE_FAMILIES`] wins.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Result;
use serde_json::Value;

/// Families the probe can classify. A model issue that ports a new family adds
/// its probe name here and registers a constructor in
/// [`super::registry::FAMILIES`].
pub const IMAGE_FAMILIES: &[&str] = &[
    "bonsai",
    "flux2",
    "z_image",
    "ming_image",
    "ernie_image",
    "qwen_image",
    "mage_flow",
    "ideogram4",
];

const FLUX2_MARKERS: &[&str] = &[
    "time_guidance_embed.linear_1.weight",
    "double_stream_modulation_img.linear.weight",
    "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
];
const Z_IMAGE_MARKERS: &[&str] = &[
    "layers.0.feed_forward.w1.weight",
    "context_refiner.0.attention.to_q.weight",
    "noise_refiner.0.adaLN_modulation.0.weight",
];
const ERNIE_IMAGE_MARKERS: &[&str] = &[
    "adaln_modulation.weight",
    "final_norm.linear.weight",
    "layers.0.adaLN_sa_ln.weight",
];
const QWEN_IMAGE_MARKERS: &[&str] = &[
    "transformer_blocks.0.img_mlp.gate_layer.weight",
    "txt_in.text_norm.weight",
    "txt_in.in_layer.weight",
];

const BONSAI_MANIFEST_PATHS: &[&str] = &[
    "transformer-packed-mflux/diffusion_pytorch_model.safetensors",
    "text_encoder-mlx-4bit/model.safetensors",
    "tokenizer/tokenizer.json",
];

/// Classify the checkpoint at `path`.
///
/// Returns `Ok(None)` for a path that does not exist, is not a directory, or
/// carries no recognizable image-generation metadata. Unparseable JSON files
/// are skipped, as upstream does, so a text checkpoint never errors here.
pub fn detect_image_family(path: &Path) -> Result<Option<&'static str>> {
    if !path.is_dir() {
        return Ok(None);
    }
    let mut candidates: Vec<String> = Vec::new();

    if let Some(manifest) = load_json_object(&path.join("manifest.json")) {
        add_model_type(
            &mut candidates,
            model_type_from_manifest(&manifest).map(str::to_string),
        );
    }

    for filename in ["model_index.json", "config.json"] {
        if let Some(metadata) = load_json_object(&path.join(filename)) {
            for model_type in model_types_from_metadata(&metadata) {
                add_model_type(&mut candidates, Some(model_type));
            }
        }
    }

    add_model_type(
        &mut candidates,
        model_type_from_component_indexes(path).map(str::to_string),
    );

    for config_path in sorted_component_configs(path)? {
        if let Some(metadata) = load_json_object(&config_path) {
            for model_type in model_types_from_metadata(&metadata) {
                add_model_type(&mut candidates, Some(model_type));
            }
        }
    }

    Ok(candidates
        .iter()
        .find_map(|candidate| IMAGE_FAMILIES.iter().find(|f| **f == candidate).copied()))
}

/// The error `generate` and `serve` return for an image checkpoint instead of
/// the misleading `model_type not found`.
///
/// `None` when `path` is not a recognized image-generation checkpoint.
#[must_use]
pub fn image_checkpoint_redirect_error(path: &Path) -> Option<anyhow::Error> {
    match detect_image_family(path) {
        Ok(Some(family)) => Some(anyhow::anyhow!(
            "image generation checkpoint ({family}): use mlxcel generate-image"
        )),
        _ => None,
    }
}

fn load_json_object(path: &Path) -> Option<Value> {
    let text = std::fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    value.is_object().then_some(value)
}

/// Python `str.strip().lower().replace("-", "_")` followed by `isidentifier()`.
fn normalize_model_type(value: &str) -> Option<String> {
    let normalized = value.trim().to_lowercase().replace('-', "_");
    let mut chars = normalized.chars();
    let first = chars.next()?;
    if !(first.is_alphabetic() || first == '_') {
        return None;
    }
    chars
        .all(|c| c.is_alphanumeric() || c == '_')
        .then_some(normalized)
}

fn add_model_type(candidates: &mut Vec<String>, model_type: Option<String>) {
    if let Some(normalized) = model_type.as_deref().and_then(normalize_model_type)
        && !candidates.contains(&normalized)
    {
        candidates.push(normalized);
    }
}

/// Split a class name into the tokens upstream's
/// `[A-Z][a-z0-9]*|[A-Z]+(?=[A-Z]|$)` regex yields.
///
/// The first alternative matches at every ASCII capital, so the second never
/// fires; a token is one capital followed by its lowercase letters and digits.
fn class_name_tokens(class_name: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current: Option<String> = None;
    for c in class_name.chars() {
        if c.is_ascii_uppercase() {
            tokens.extend(current.take());
            current = Some(c.to_string());
        } else if c.is_ascii_lowercase() || c.is_ascii_digit() {
            // Lowercase or digits before the first capital match nothing.
            if let Some(token) = current.as_mut() {
                token.push(c);
            }
        } else {
            // A separator ends the current token.
            tokens.extend(current.take());
        }
    }
    tokens.extend(current.take());
    tokens
}

fn model_types_from_class_name(class_name: &str) -> Vec<String> {
    let tokens = class_name_tokens(class_name);
    let mut candidates: Vec<String> = Vec::new();
    for end in 1..=tokens.len() {
        add_model_type(&mut candidates, Some(tokens[..end].join("_")));
        if end > 1 {
            // Preserve mixed-case acronyms such as LLaDA in LLaDAImagePipeline.
            add_model_type(
                &mut candidates,
                Some(format!(
                    "{}_{}",
                    tokens[..end - 1].join(""),
                    tokens[end - 1]
                )),
            );
        }
    }
    for token in &tokens {
        add_model_type(&mut candidates, Some(token.clone()));
    }
    candidates
}

fn model_types_from_metadata(metadata: &Value) -> Vec<String> {
    let mut candidates: Vec<String> = Vec::new();
    add_model_type(
        &mut candidates,
        metadata
            .get("model_type")
            .and_then(Value::as_str)
            .map(str::to_string),
    );
    let class_name = metadata
        .get("_class_name")
        .and_then(Value::as_str)
        .unwrap_or("");
    for candidate in model_types_from_class_name(class_name) {
        add_model_type(&mut candidates, Some(candidate));
    }
    for component in ["transformer", "vae", "scheduler", "text_encoder"] {
        let component_class = match metadata.get(component) {
            Some(Value::Array(items)) => items
                .last()
                .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_string)),
            Some(Value::Object(map)) => Some(
                map.get("_class_name")
                    .or_else(|| map.get("class"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            ),
            _ => None,
        }
        .unwrap_or_default();
        for candidate in model_types_from_class_name(&component_class) {
            add_model_type(&mut candidates, Some(candidate));
        }
    }
    candidates
}

fn manifest_paths(manifest: &Value) -> BTreeSet<String> {
    let mut paths = BTreeSet::new();
    let Some(files) = manifest.get("files").and_then(Value::as_array) else {
        return paths;
    };
    for entry in files {
        match entry {
            Value::String(s) => {
                paths.insert(s.clone());
            }
            Value::Object(map) => {
                let path = ["remote_path", "path", "filename", "name"]
                    .iter()
                    .find_map(|key| {
                        map.get(*key)
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                    });
                if let Some(path) = path {
                    paths.insert(path.to_string());
                }
            }
            _ => {}
        }
    }
    paths
}

fn model_type_from_manifest(manifest: &Value) -> Option<&'static str> {
    let paths = manifest_paths(manifest);
    BONSAI_MANIFEST_PATHS
        .iter()
        .all(|p| paths.contains(*p))
        .then_some("bonsai")
}

fn contains_all(keys: &BTreeSet<String>, markers: &[&str]) -> bool {
    markers.iter().all(|m| keys.contains(*m))
}

fn model_type_from_component_indexes(root: &Path) -> Option<&'static str> {
    let transformer = root.join("transformer");
    let index =
        load_json_object(&transformer.join("model.safetensors.index.json")).or_else(|| {
            load_json_object(&transformer.join("diffusion_pytorch_model.safetensors.index.json"))
        })?;
    let keys: BTreeSet<String> = index
        .get("weight_map")?
        .as_object()?
        .keys()
        .cloned()
        .collect();
    if contains_all(&keys, FLUX2_MARKERS) {
        return Some("flux2");
    }
    if contains_all(&keys, Z_IMAGE_MARKERS) {
        // Ming-Image reuses the Z-Image transformer and adds an `mllm/` tower.
        return Some(if root.join("mllm").join("config.json").exists() {
            "ming_image"
        } else {
            "z_image"
        });
    }
    if contains_all(&keys, ERNIE_IMAGE_MARKERS) {
        return Some("ernie_image");
    }
    if contains_all(&keys, QWEN_IMAGE_MARKERS) {
        return Some("qwen_image");
    }
    None
}

/// `<root>/*/config.json`, sorted by directory name.
fn sorted_component_configs(root: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut dirs: Vec<std::path::PathBuf> = std::fs::read_dir(root)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    Ok(dirs
        .into_iter()
        .map(|d| d.join("config.json"))
        .filter(|p| p.is_file())
        .collect())
}

#[cfg(test)]
#[path = "probe_tests.rs"]
mod tests;
