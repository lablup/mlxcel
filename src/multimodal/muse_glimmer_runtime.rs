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

//! Muse Glimmer request-time multimodal preparation.

use anyhow::Result;
use image::DynamicImage;
use mlxcel_core::MlxArray;

use crate::vision::MuseGlimmerVlmModel;
use crate::vision::merge::InputEmbeddings;

use super::muse_glimmer_prompt::{
    MuseGlimmerPromptTokens, expand_muse_glimmer_image_placeholders, image_placeholder_tokens,
};
use super::vlm_runtime::{PreparedVlmEmbeddings, VlmPreparationSummary};

pub(crate) fn reject_muse_glimmer_text_fallback(
    model: &MuseGlimmerVlmModel,
    prompt_tokens: &[i32],
    image_count: usize,
) -> Result<()> {
    let tokens = prompt_tokens_for(model);
    if prompt_tokens.iter().any(|&id| id == model.video_token_id()) {
        anyhow::bail!("Muse Glimmer VLM does not support video inputs yet");
    }
    if image_count == 0
        && prompt_tokens.iter().any(|&id| {
            id == tokens.image_placeholder_token_id
                || id == tokens.image_start_token_id
                || id == tokens.image_token_id
                || id == tokens.image_end_token_id
        })
    {
        anyhow::bail!(
            "Muse Glimmer prompt contains image marker tokens but no images were provided; refusing text-only fallback"
        );
    }
    Ok(())
}

pub(crate) fn prepare_muse_glimmer_vlm_embeddings(
    model: &MuseGlimmerVlmModel,
    prompt_tokens: &mut Vec<i32>,
    images: &[DynamicImage],
) -> Result<PreparedVlmEmbeddings> {
    reject_muse_glimmer_text_fallback(model, prompt_tokens, images.len())?;

    let (pixel_values, image_grid_thw) = model.preprocess_images(images);
    validate_preprocessed_pixels(model, &pixel_values, &image_grid_thw, images.len())?;

    let stats = expand_muse_glimmer_image_placeholders(
        prompt_tokens,
        &image_grid_thw,
        model.image_processor.merge_size,
        prompt_tokens_for(model),
    )
    .map_err(anyhow::Error::msg)?;

    let pixel_values = pixel_values
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Muse Glimmer image preprocessing produced null pixels"))?;
    let image_features = model
        .encode_and_fuse_images(pixel_values, &image_grid_thw)
        .map_err(anyhow::Error::msg)?;
    let image_features = image_features.as_ref().ok_or_else(|| {
        anyhow::anyhow!("Muse Glimmer image encoder produced null projected features")
    })?;

    let input_ids = prompt_ids_array(prompt_tokens);
    let inputs_embeds = model
        .text_embeddings(&input_ids)
        .map_err(anyhow::Error::msg)?;
    let inputs_embeds = inputs_embeds.as_ref().ok_or_else(|| {
        anyhow::anyhow!("Muse Glimmer text decoder produced null token embeddings")
    })?;
    let embeddings = merge_muse_glimmer_features(
        prompt_tokens_for(model).image_token_id,
        prompt_tokens,
        &image_grid_thw,
        model.image_processor.merge_size,
        image_features,
        inputs_embeds,
    )?;

    Ok(PreparedVlmEmbeddings {
        embeddings,
        preparation: Some(VlmPreparationSummary::MuseGlimmer {
            image_blocks: stats.image_blocks,
            image_tokens: stats.image_tokens,
            total_tokens: stats.total_tokens,
        }),
    })
}

pub(crate) fn merge_muse_glimmer_features(
    image_token_id: i32,
    prompt_tokens: &[i32],
    image_grid_thw: &[(i32, i32, i32)],
    merge_size: usize,
    image_features: &MlxArray,
    inputs_embeds: &MlxArray,
) -> Result<InputEmbeddings> {
    let expected_rows = expected_feature_rows(image_grid_thw, merge_size)?;
    let feature_shape = mlxcel_core::array_shape(image_features);
    if feature_shape.len() != 2 {
        anyhow::bail!(
            "Muse Glimmer projected image features must be [visual_tokens, hidden], got {feature_shape:?}"
        );
    }
    let feature_rows = positive_dim_to_usize(feature_shape[0], "projected feature rows")?;
    if feature_rows != expected_rows {
        anyhow::bail!(
            "Muse Glimmer projected feature rows {feature_rows} do not match merged-grid visual token count {expected_rows}"
        );
    }

    let embed_shape = mlxcel_core::array_shape(inputs_embeds);
    if embed_shape.len() != 3 || embed_shape[0] != 1 {
        anyhow::bail!(
            "Muse Glimmer text embeddings must be [1, tokens, hidden], got {embed_shape:?}"
        );
    }
    if feature_shape[1] != embed_shape[2] {
        anyhow::bail!(
            "Muse Glimmer projected hidden size {} does not match text embedding hidden size {}",
            feature_shape[1],
            embed_shape[2]
        );
    }

    let patch_tokens = prompt_tokens
        .iter()
        .filter(|&&id| id == image_token_id)
        .count();
    if patch_tokens != expected_rows {
        anyhow::bail!(
            "Muse Glimmer expanded prompt contains {patch_tokens} patch token(s), but image grids require {expected_rows}"
        );
    }

    let input_ids = prompt_ids_array(prompt_tokens);
    Ok(crate::vision::merge::merge_llava(
        image_token_id,
        image_features,
        inputs_embeds,
        &input_ids,
    ))
}

fn validate_preprocessed_pixels(
    model: &MuseGlimmerVlmModel,
    pixel_values: &mlxcel_core::UniquePtr<MlxArray>,
    image_grid_thw: &[(i32, i32, i32)],
    image_count: usize,
) -> Result<()> {
    if image_grid_thw.len() != image_count {
        anyhow::bail!(
            "Muse Glimmer image preprocessing returned {} grids for {image_count} image(s)",
            image_grid_thw.len()
        );
    }
    let pixel_values = pixel_values
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Muse Glimmer image preprocessing produced null pixels"))?;
    let shape = mlxcel_core::array_shape(pixel_values);
    let expected_patch_dim = model.image_processor.temporal_patch_size
        * 3
        * model.image_processor.patch_size
        * model.image_processor.patch_size;
    if shape.len() != 2 || shape[1] as usize != expected_patch_dim {
        anyhow::bail!(
            "Muse Glimmer image preprocessing produced shape {shape:?}, expected [patches, {expected_patch_dim}]"
        );
    }
    let expected_patch_rows = image_grid_thw.iter().try_fold(0usize, |acc, &(t, h, w)| {
        if t <= 0 || h <= 0 || w <= 0 {
            anyhow::bail!("Muse Glimmer grid must be positive, got {:?}", (t, h, w));
        }
        acc.checked_add(t as usize * h as usize * w as usize)
            .ok_or_else(|| anyhow::anyhow!("Muse Glimmer preprocessed patch row count overflowed"))
    })?;
    let actual_patch_rows = positive_dim_to_usize(shape[0], "preprocessed patch rows")?;
    if actual_patch_rows != expected_patch_rows {
        anyhow::bail!(
            "Muse Glimmer preprocessed patch rows {actual_patch_rows} do not match image_grid_thw patch rows {expected_patch_rows}"
        );
    }
    Ok(())
}

fn expected_feature_rows(image_grid_thw: &[(i32, i32, i32)], merge_size: usize) -> Result<usize> {
    image_grid_thw.iter().try_fold(0usize, |acc, &grid| {
        let rows = image_placeholder_tokens(grid, merge_size).map_err(anyhow::Error::msg)?;
        acc.checked_add(rows)
            .ok_or_else(|| anyhow::anyhow!("Muse Glimmer visual token count overflowed"))
    })
}

fn prompt_tokens_for(model: &MuseGlimmerVlmModel) -> MuseGlimmerPromptTokens {
    MuseGlimmerPromptTokens {
        image_token_id: model.image_token_id(),
        ..MuseGlimmerPromptTokens::default()
    }
}

fn prompt_ids_array(prompt_tokens: &[i32]) -> mlxcel_core::UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32])
}

fn positive_dim_to_usize(dim: i32, label: &str) -> Result<usize> {
    if dim < 0 {
        anyhow::bail!("Muse Glimmer {label} must be non-negative, got {dim}");
    }
    Ok(dim as usize)
}

#[cfg(test)]
#[path = "muse_glimmer_runtime_tests.rs"]
mod tests;
