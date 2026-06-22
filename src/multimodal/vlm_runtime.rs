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

//! Shared VLM runtime preparation used by CLI and server flows.
//!
//! This module answers two questions for the control plane:
//! - should a request prepare multimodal embeddings at all?
//! - if so, which model-family-specific embedding path should be used?
//!
//! It owns request-time image preprocessing and prepared-embedding validation,
//! while model loading and vision math stay in `src/loading/` and `src/vision/`.

use anyhow::Result;
use image::DynamicImage;
use mlxcel_core::MlxArray;

use crate::internvl_prompt::insert_internvl_image_tokens;
use crate::minicpmo_prompt::{
    prepare_minicpmo_prompt_tokens, prepare_minicpmo_prompt_tokens_with_image_feature_sizes,
};
use crate::moondream3_prompt::{Moondream3PromptMode, prepare_moondream3_prompt_tokens};
use crate::phi3v_prompt::prepare_phi3v_prompt_tokens;
use crate::phi4_siglip_prompt::prepare_phi4_siglip_prompt_tokens;
use crate::phi4mm_prompt::prepare_phi4mm_prompt_tokens;
use crate::qwen_vl::insert_qwen_vl_image_tokens;
use crate::vision::feature_cache::{CacheKey, ModelVisionCaches, image_hash_from_pixels};
use crate::vision::merge::InputEmbeddings;
use crate::vision::processors::ImageProcessor;
use crate::vlm_prompt::{ImageTokenBlockStats, apply_image_token_blocks};
use crate::youtu_vl_prompt::insert_youtu_vl_image_tokens;
use crate::{LoadedModel, VlmRuntimeRef};

const MOLMO_V1_BOS_TOKEN_ID: i32 = 151643;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VlmPreparationSummary {
    QwenVlm {
        image_blocks: usize,
        total_image_tokens: i32,
    },
    MiniCPMO {
        image_slots: usize,
        total_tokens: usize,
    },
    Moondream3 {
        mode: Moondream3PromptMode,
        total_tokens: usize,
        prefix_tokens: usize,
    },
    Gemma4 {
        image_slots: usize,
        total_tokens: usize,
    },
    Gemma4Audio {
        audio_tokens: usize,
        total_tokens: usize,
    },
    /// Gemma 4 expanded `<|video|>` placeholders for one or
    /// more decoded videos. `frame_slots` is the total number of frames
    /// across all videos (each frame consumes
    /// `boi + image_token * num_soft_tokens_per_frame + eoi`).
    Gemma4Video {
        video_count: usize,
        frame_slots: usize,
        total_tokens: usize,
    },
    Phi4MM {
        image_slots: usize,
        total_tokens: usize,
    },
    Molmo {
        total_tokens: usize,
    },
    Molmo2 {
        total_tokens: usize,
    },
    MolmoPoint {
        total_tokens: usize,
    },
    Phi4SigLip {
        image_slots: usize,
        total_tokens: usize,
    },
    Phi3V {
        image_slots: usize,
        total_tokens: usize,
    },
    NemotronHNanoOmni {
        image_slots: usize,
        total_tokens: usize,
    },
    /// Nemotron H Nano Omni expanded one or more audio
    /// clip placeholders. `audio_clips` is the number of clips
    /// processed; `audio_tokens` is the post-subsampling encoder
    /// output token count fed into the merge step.
    NemotronHNanoOmniAudio {
        audio_clips: usize,
        audio_tokens: usize,
        total_tokens: usize,
    },
    YoutuVL {
        image_blocks: usize,
        total_image_tokens: i32,
    },
    /// InternVL expanded each `<image>`/`<IMG_CONTEXT>` placeholder
    /// into `<img> + <IMG_CONTEXT> * (num_image_token * tiles) + </img>`.
    InternVL {
        image_blocks: usize,
        total_image_tokens: usize,
    },
    ImageBlocks(ImageTokenBlockStats),
}

pub struct PreparedVlmEmbeddings {
    pub embeddings: InputEmbeddings,
    pub preparation: Option<VlmPreparationSummary>,
}

pub fn prepared_embedding_refs(
    embeddings: &InputEmbeddings,
) -> Result<(&MlxArray, Option<&MlxArray>)> {
    let input_embeds = embeddings
        .inputs_embeds
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Prepared VLM embeddings are missing input embeddings"))?;
    let attention_mask = match embeddings.attention_mask_4d.as_ref() {
        Some(mask) => Some(mask.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Prepared VLM embeddings contain a null 4D attention mask")
        })?),
        None => None,
    };
    Ok((input_embeds, attention_mask))
}

fn should_prepare_vlm_embeddings(image_count: usize, is_vlm: bool) -> Result<bool> {
    if image_count == 0 {
        Ok(false)
    } else if is_vlm {
        Ok(true)
    } else {
        Err(anyhow::anyhow!(
            "Images provided but model is not a vision-language model"
        ))
    }
}

fn prompt_ids_array(prompt_tokens: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32])
}

/// Pick the first explicit (non-None) cache key from a caller-supplied slice.
///
/// Used by Qwen-VL families which pack every image into a single pixel tensor
/// per request: a request-scoped cache key is sufficient and we prefer the
/// caller's key when they provided one (e.g. a filesystem path).
fn first_explicit_key(keys: Option<&[Option<CacheKey>]>) -> Option<CacheKey> {
    keys.and_then(|slice| slice.iter().find_map(|k| k.clone()))
}

/// Build a per-image cache key list, preferring explicit caller-supplied keys
/// and falling back to pixel-byte hashing for entries the caller left `None`.
///
/// This is used by Gemma 4 VLM which runs the vision tower once per image and
/// therefore benefits from per-image cache lookup.
fn resolve_per_image_keys<F>(
    explicit: Option<&[Option<CacheKey>]>,
    count: usize,
    mut fallback: F,
) -> Vec<Option<CacheKey>>
where
    F: FnMut(usize) -> Option<[u8; 32]>,
{
    (0..count)
        .map(|idx| {
            if let Some(key) = explicit.and_then(|slice| slice.get(idx).cloned()).flatten() {
                Some(key)
            } else {
                fallback(idx).map(CacheKey::from_hash)
            }
        })
        .collect()
}

pub fn prepare_and_compute_vlm_embeddings<E>(
    model: &LoadedModel,
    prompt_tokens: &mut Vec<i32>,
    prompt: &str,
    images: &[DynamicImage],
    encode: E,
) -> Result<Option<PreparedVlmEmbeddings>>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    prepare_and_compute_vlm_embeddings_with_cache(
        model,
        prompt_tokens,
        prompt,
        images,
        None,
        None,
        encode,
    )
}

/// Cache-aware wrapper for [`prepare_and_compute_vlm_embeddings`].
///
/// When `caches` is `Some`, the VLM runtime is invoked through its
/// cache-aware variant. Cache keys (one per image for per-image VLM families
/// like Gemma 4, or a single request-scoped key for batch-style VLM families
/// like Qwen2.5/3-VL) are supplied via `image_cache_keys`. When no keys are
/// provided, keys are derived on the fly from the pixel tensor bytes.
///
/// Passing `caches == None` or `caches.enabled() == false` falls through to
/// the un-cached path with zero additional cost.
pub fn prepare_and_compute_vlm_embeddings_with_cache<E>(
    model: &LoadedModel,
    prompt_tokens: &mut Vec<i32>,
    prompt: &str,
    images: &[DynamicImage],
    image_cache_keys: Option<&[Option<CacheKey>]>,
    caches: Option<&ModelVisionCaches>,
    mut encode: E,
) -> Result<Option<PreparedVlmEmbeddings>>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    if !should_prepare_vlm_embeddings(images.len(), model.is_vlm())? {
        return Ok(None);
    }

    let runtime = model
        .vlm_runtime()
        .ok_or_else(|| anyhow::anyhow!("Images provided but model has no VLM runtime"))?;

    // Only activate the cache when both the caches are present AND enabled.
    let active_caches = caches.filter(|c| c.enabled());

    match runtime {
        VlmRuntimeRef::Qwen(qwen) => {
            let info = qwen.prompt_info();
            let (pixel_values, grid_thw) = info.processor.preprocess_with_grid(images);
            let preparation = insert_qwen_vl_image_tokens(
                prompt_tokens,
                &grid_thw,
                info.spatial_merge_size,
                info.vision_start_token_id,
                info.image_token_id,
            )
            .map(|stats| VlmPreparationSummary::QwenVlm {
                image_blocks: stats.image_blocks,
                total_image_tokens: stats.total_image_tokens,
            });

            // Qwen-VL families take one concatenated pixel tensor per request.
            // Use the first explicit cache key when provided; otherwise derive
            // one from the pixel bytes when caching is enabled.
            let qwen_cache_key = if active_caches.is_some() {
                first_explicit_key(image_cache_keys)
                    .or_else(|| Some(CacheKey::from_hash(image_hash_from_pixels(&pixel_values))))
            } else {
                None
            };

            let input_ids_arr = prompt_ids_array(prompt_tokens);
            let embeddings = qwen.input_embeddings_with_cache(
                &input_ids_arr,
                &pixel_values,
                &grid_thw,
                qwen_cache_key.as_ref(),
                active_caches,
            );

            Ok(Some(PreparedVlmEmbeddings {
                embeddings,
                preparation,
            }))
        }
        VlmRuntimeRef::MiniCPMO(minicpmo) => {
            let prepared = prepare_minicpmo_prompt_tokens(
                prompt,
                images.len(),
                minicpmo.processor.image_feature_size,
                &mut encode,
            )
            .map_err(|err| anyhow::anyhow!("{}", err))?;
            *prompt_tokens = prepared.tokens;

            let processed_images = minicpmo.processor.preprocess(images);
            let input_ids_arr = prompt_ids_array(prompt_tokens);
            let embeddings = minicpmo.get_input_embeddings(
                &input_ids_arr,
                &processed_images,
                &prepared.image_bounds,
            );

            Ok(Some(PreparedVlmEmbeddings {
                embeddings,
                preparation: Some(VlmPreparationSummary::MiniCPMO {
                    image_slots: prepared.image_slots,
                    total_tokens: prompt_tokens.len(),
                }),
            }))
        }
        // MiniCPM-V 4.6 shares the same prompt token format as MiniCPM-O
        // (`<image><unk>...<unk></image>`) but uses Qwen3.5 text backbone +
        // the VitMerger+Merger vision pipeline instead of the resampler.
        VlmRuntimeRef::MiniCPMV46(minicpmv46) => {
            let processed_images = minicpmv46.processor.preprocess(images);
            let image_feature_sizes: Vec<usize> = processed_images
                .iter()
                .map(|processed| {
                    minicpmv46
                        .image_feature_size_for_processed(processed)
                        .map_err(|err| anyhow::anyhow!("{}", err))
                })
                .collect::<Result<Vec<_>>>()?;

            let prepared = prepare_minicpmo_prompt_tokens_with_image_feature_sizes(
                prompt,
                &image_feature_sizes,
                &mut encode,
            )
            .map_err(|err| anyhow::anyhow!("{}", err))?;
            *prompt_tokens = prepared.tokens;

            let input_ids_arr = prompt_ids_array(prompt_tokens);
            let embeddings = minicpmv46.get_input_embeddings(
                &input_ids_arr,
                &processed_images,
                &prepared.image_bounds,
            );

            Ok(Some(PreparedVlmEmbeddings {
                embeddings,
                preparation: Some(VlmPreparationSummary::MiniCPMO {
                    image_slots: prepared.image_slots,
                    total_tokens: prompt_tokens.len(),
                }),
            }))
        }
        VlmRuntimeRef::Moondream3(moondream3) => {
            let prepared = prepare_moondream3_prompt_tokens(prompt, images.len(), &mut encode)
                .map_err(|err| anyhow::anyhow!("{}", err))?;
            *prompt_tokens = prepared.tokens;

            let processed_image = moondream3.processor.preprocess_image(&images[0]);
            let input_ids_arr = prompt_ids_array(prompt_tokens);
            let embeddings = moondream3.get_input_embeddings(&input_ids_arr, &processed_image);

            Ok(Some(PreparedVlmEmbeddings {
                embeddings,
                preparation: Some(VlmPreparationSummary::Moondream3 {
                    mode: prepared.mode,
                    total_tokens: prompt_tokens.len(),
                    prefix_tokens: moondream3.prefix_token_count(),
                }),
            }))
        }
        VlmRuntimeRef::Gemma3n(gemma3n_vl) => {
            let preparation = model
                .image_token_block_info()
                .and_then(|info| apply_image_token_blocks(prompt_tokens, info, images.len()))
                .map(VlmPreparationSummary::ImageBlocks);

            let pixel_values = gemma3n_vl.processor.preprocess(images);
            let input_ids_arr = prompt_ids_array(prompt_tokens);
            let embeddings = gemma3n_vl.get_input_embeddings(&input_ids_arr, &pixel_values);

            Ok(Some(PreparedVlmEmbeddings {
                embeddings,
                preparation,
            }))
        }
        VlmRuntimeRef::Gemma4(gemma4_vl) => {
            let processed_images = gemma4_vl.processor.preprocess(images);
            let num_soft_tokens: Vec<usize> = processed_images
                .iter()
                .map(|image| image.num_soft_tokens)
                .collect();
            expand_gemma4_image_tokens(
                prompt_tokens,
                gemma4_vl.image_token_id,
                gemma4_vl.boi_token_id,
                gemma4_vl.eoi_token_id,
                &num_soft_tokens,
            )?;

            // Build per-image cache keys when caching is enabled. Explicit
            // keys supplied by the caller take precedence over hashing the
            // pixel tensor, which matches the "path > hash" policy.
            let gemma4_keys: Option<Vec<Option<CacheKey>>> = if active_caches.is_some() {
                Some(resolve_per_image_keys(
                    image_cache_keys,
                    processed_images.len(),
                    |i| {
                        processed_images[i]
                            .pixel_values
                            .as_ref()
                            .map(image_hash_from_pixels)
                    },
                ))
            } else {
                None
            };

            let input_ids_arr = prompt_ids_array(prompt_tokens);
            let embeddings = gemma4_vl.get_input_embeddings_with_audio_and_cache(
                &input_ids_arr,
                &processed_images,
                None,
                None,
                gemma4_keys.as_deref(),
                active_caches.map(|c| &c.single),
            );

            Ok(Some(PreparedVlmEmbeddings {
                embeddings,
                preparation: Some(VlmPreparationSummary::Gemma4 {
                    image_slots: processed_images.len(),
                    total_tokens: prompt_tokens.len(),
                }),
            }))
        }
        VlmRuntimeRef::Gemma4Unified(unified) => {
            // Encoder-free patch projector: preprocess to flat patch matrices +
            // 2-D positions, expand BOI/IMAGE/EOI placeholders, then merge.
            let processed_images = unified.processor.preprocess(images);
            let num_soft_tokens: Vec<usize> = processed_images
                .iter()
                .map(|image| image.num_soft_tokens)
                .collect();
            expand_gemma4_image_tokens(
                prompt_tokens,
                unified.image_token_id,
                unified.boi_token_id,
                unified.eoi_token_id,
                &num_soft_tokens,
            )?;

            let input_ids_arr = prompt_ids_array(prompt_tokens);
            let embeddings = unified.get_input_embeddings(&input_ids_arr, &processed_images);

            Ok(Some(PreparedVlmEmbeddings {
                embeddings,
                preparation: Some(VlmPreparationSummary::Gemma4 {
                    image_slots: processed_images.len(),
                    total_tokens: prompt_tokens.len(),
                }),
            }))
        }
        VlmRuntimeRef::Phi4MM(phi4mm) => {
            // 1. Preprocess images first to get num_img_tokens per image
            let processed_images = phi4mm.processor.preprocess(images);

            // 2. Tokenize prompt (gets 1x -200 per image placeholder)
            let prepared = prepare_phi4mm_prompt_tokens(prompt, images.len(), &mut encode)
                .map_err(|err| anyhow::anyhow!("{}", err))?;
            let tokens = prepared.tokens;

            // 3. Expand each -200 sentinel to match num_img_tokens from HD transform
            let mut img_idx = 0;
            let mut expanded = Vec::with_capacity(tokens.len());
            for &tok in &tokens {
                if tok == crate::phi4_siglip_prompt::PHI4_SIGLIP_IMAGE_TOKEN_INDEX {
                    if let Some(processed) = processed_images.get(img_idx) {
                        expanded.extend(std::iter::repeat_n(
                            crate::phi4_siglip_prompt::PHI4_SIGLIP_IMAGE_TOKEN_INDEX,
                            processed.num_img_tokens,
                        ));
                    } else {
                        expanded.push(tok);
                    }
                    img_idx += 1;
                } else {
                    expanded.push(tok);
                }
            }
            *prompt_tokens = expanded;

            let input_ids_arr = prompt_ids_array(prompt_tokens);
            let embeddings = phi4mm.get_input_embeddings(&input_ids_arr, &processed_images);

            Ok(Some(PreparedVlmEmbeddings {
                embeddings,
                preparation: Some(VlmPreparationSummary::Phi4MM {
                    image_slots: prepared.image_slots,
                    total_tokens: prompt_tokens.len(),
                }),
            }))
        }
        VlmRuntimeRef::Phi4SigLip(phi4_siglip) => {
            let prepared = prepare_phi4_siglip_prompt_tokens(prompt, images.len(), &mut encode)
                .map_err(|err| anyhow::anyhow!("{}", err))?;
            *prompt_tokens = prepared.tokens;

            let processed_images = phi4_siglip.processor.preprocess(images);
            let input_ids_arr = prompt_ids_array(prompt_tokens);
            let embeddings = phi4_siglip.get_input_embeddings(&input_ids_arr, &processed_images);

            Ok(Some(PreparedVlmEmbeddings {
                embeddings,
                preparation: Some(VlmPreparationSummary::Phi4SigLip {
                    image_slots: prepared.image_slots,
                    total_tokens: prompt_tokens.len(),
                }),
            }))
        }
        VlmRuntimeRef::Molmo(molmo) => {
            // Molmo v1 builds the `<im_start>…<im_end>` token block as explicit
            // IDs, then prepends an EOS-as-BOS token to the whole sequence. The
            // model-local processor also wraps text as ` User: ... Assistant:`
            // and shifts `image_input_idx` by one after inserting BOS.
            let proc_out = molmo.processor.preprocess_image(&images[0]);

            let molmo_prompt = format_molmo_v1_prompt_for_processor(prompt);
            let prompt_text_ids = encode(&molmo_prompt, false);
            let mut combined =
                Vec::with_capacity(1 + proc_out.image_token_ids.len() + prompt_text_ids.len());
            combined.push(MOLMO_V1_BOS_TOKEN_ID);
            combined.extend(proc_out.image_token_ids.clone());
            combined.extend(prompt_text_ids);
            *prompt_tokens = combined;

            let image_input_idx_shifted =
                shift_molmo_v1_image_input_idx_for_bos(&proc_out.image_input_idx);
            let pixel_values =
                mlxcel_core::from_slice_f32(&proc_out.pixel_values, &proc_out.pixel_values_shape);
            let image_input_idx = mlxcel_core::from_slice_i32(
                &image_input_idx_shifted,
                &[proc_out.image_input_idx_len],
            );
            let image_masks =
                mlxcel_core::from_slice_f32(&proc_out.image_masks, &proc_out.image_masks_shape);
            let input_ids_arr = prompt_ids_array(prompt_tokens);
            let embeddings = molmo.get_input_embeddings(
                &input_ids_arr,
                &pixel_values,
                &image_input_idx,
                &image_masks,
            );

            Ok(Some(PreparedVlmEmbeddings {
                embeddings,
                preparation: Some(VlmPreparationSummary::Molmo {
                    total_tokens: prompt_tokens.len(),
                }),
            }))
        }
        VlmRuntimeRef::Molmo2(molmo2) => {
            let proc_out = molmo2.processor.preprocess_image(&images[0]);
            let image_token_str = molmo2.processor.get_image_tokens(&proc_out.image_grid);
            let text = if prompt.contains("<|image|>") {
                prompt.replace("<|image|>", &image_token_str)
            } else {
                format!("{}{}", image_token_str, prompt)
            };
            *prompt_tokens = encode(&text, true);

            let pixel_values =
                mlxcel_core::from_slice_f32(&proc_out.pixel_values, &proc_out.pixel_values_shape);
            let image_token_pooling = mlxcel_core::from_slice_i32(
                &proc_out.image_token_pooling,
                &proc_out.image_token_pooling_shape,
            );
            let image_grids = mlxcel_core::from_slice_i32(&proc_out.image_grid, &[4]);
            let image_num_crops = mlxcel_core::from_slice_i32(&[proc_out.image_num_crops], &[1]);
            let input_ids_arr = prompt_ids_array(prompt_tokens);
            let embeddings = molmo2.get_input_embeddings(
                &input_ids_arr,
                &pixel_values,
                &image_token_pooling,
                &image_grids,
                &image_num_crops,
            );

            Ok(Some(PreparedVlmEmbeddings {
                embeddings,
                preparation: Some(VlmPreparationSummary::Molmo2 {
                    total_tokens: prompt_tokens.len(),
                }),
            }))
        }
        VlmRuntimeRef::MolmoPoint(molmo_point) => {
            let proc_out = molmo_point.processor.preprocess_image(&images[0]);
            let image_token_str = molmo_point.processor.get_image_tokens(&proc_out.image_grid);
            let text = if prompt.contains("<|image|>") {
                prompt.replace("<|image|>", &image_token_str)
            } else {
                format!("{image_token_str}{prompt}")
            };
            *prompt_tokens = encode(&text, true);

            let pixel_values =
                mlxcel_core::from_slice_f32(&proc_out.pixel_values, &proc_out.pixel_values_shape);
            let image_token_pooling = mlxcel_core::from_slice_i32(
                &proc_out.image_token_pooling,
                &proc_out.image_token_pooling_shape,
            );
            let image_grids = mlxcel_core::from_slice_i32(&proc_out.image_grid, &[4]);
            let image_num_crops = mlxcel_core::from_slice_i32(&[proc_out.image_num_crops], &[1]);
            let input_ids_arr = prompt_ids_array(prompt_tokens);
            let embeddings = molmo_point.get_input_embeddings(
                &input_ids_arr,
                &pixel_values,
                &image_token_pooling,
                &image_grids,
                &image_num_crops,
            );

            Ok(Some(PreparedVlmEmbeddings {
                embeddings,
                preparation: Some(VlmPreparationSummary::MolmoPoint {
                    total_tokens: prompt_tokens.len(),
                }),
            }))
        }
        VlmRuntimeRef::Phi3V(phi3v) => {
            let preparation = if let Some(prepared) =
                prepare_phi3v_prompt_tokens(prompt, images.len(), &mut encode, |image_num| {
                    let image = &images[image_num - 1];
                    phi3v
                        .processor
                        .calc_num_image_tokens(image.width(), image.height())
                }) {
                *prompt_tokens = prepared.tokens;
                Some(VlmPreparationSummary::Phi3V {
                    image_slots: prepared.image_slots,
                    total_tokens: prompt_tokens.len(),
                })
            } else {
                None
            };

            let (pixel_values, image_sizes) = phi3v.processor.preprocess(images);
            let input_ids_arr = prompt_ids_array(prompt_tokens);
            let embeddings =
                phi3v.get_input_embeddings(&input_ids_arr, &pixel_values, &image_sizes);

            Ok(Some(PreparedVlmEmbeddings {
                embeddings,
                preparation,
            }))
        }
        VlmRuntimeRef::NemotronHNanoOmni(model) => {
            // Per-image dynamic-resolution preprocessing produces a
            // distinct `num_tokens` per image. We expand each
            // `<image>`-style placeholder in the existing prompt token
            // stream into `image_start + img_context * n + image_end`
            // (mirrors upstream `processing_nemotron_h_nano_omni`).
            let processed_images = model.processor.preprocess_batch(images);
            let placeholder = model.config.img_context_token_id;
            let image_start = model.config.image_start_token_id;
            let image_end = model.config.image_end_token_id;

            // Count placeholder occurrences. Each occurrence consumes
            // one preprocessed image; if the prompt has no placeholder
            // (e.g., legacy callers passing only an image), we prepend
            // one block per image so the runtime always sees the
            // image-context tokens before any text tokens.
            let placeholder_positions: Vec<usize> = prompt_tokens
                .iter()
                .enumerate()
                .filter_map(|(idx, &tok)| if tok == placeholder { Some(idx) } else { None })
                .collect();

            let mut expanded: Vec<i32> = Vec::with_capacity(
                prompt_tokens.len()
                    + processed_images
                        .iter()
                        .map(|img| img.num_tokens + 2)
                        .sum::<usize>(),
            );

            if placeholder_positions.is_empty() {
                for image in processed_images.iter() {
                    if image_start != 0 {
                        expanded.push(image_start);
                    }
                    for _ in 0..image.num_tokens {
                        expanded.push(placeholder);
                    }
                    if image_end != 0 {
                        expanded.push(image_end);
                    }
                }
                expanded.extend_from_slice(prompt_tokens);
            } else {
                let mut image_idx = 0usize;
                for &token in prompt_tokens.iter() {
                    if token == placeholder && image_idx < processed_images.len() {
                        let image = &processed_images[image_idx];
                        if image_start != 0 {
                            expanded.push(image_start);
                        }
                        for _ in 0..image.num_tokens {
                            expanded.push(placeholder);
                        }
                        if image_end != 0 {
                            expanded.push(image_end);
                        }
                        image_idx += 1;
                    } else {
                        expanded.push(token);
                    }
                }
            }
            *prompt_tokens = expanded;

            let input_ids_arr = prompt_ids_array(prompt_tokens);
            let embeddings = model.get_input_embeddings(&input_ids_arr, &processed_images);

            Ok(Some(PreparedVlmEmbeddings {
                embeddings,
                preparation: Some(VlmPreparationSummary::NemotronHNanoOmni {
                    image_slots: processed_images.len(),
                    total_tokens: prompt_tokens.len(),
                }),
            }))
        }
        VlmRuntimeRef::YoutuVL(youtu) => {
            // Preprocess images into the flattened-patch + spatial_shapes
            // contract that Youtu-VL's vision tower expects.
            let (pixel_values, spatial_shapes) = youtu
                .processor
                .try_preprocess_with_spatial(images)
                .map_err(|err| anyhow::anyhow!("Youtu-VL image preprocessing failed: {err}"))?;

            // Splice image-token runs into the prompt when they aren't already
            // present. We mirror the upstream `<vision_start> + image*N +
            // <vision_end>` framing.
            let preparation = insert_youtu_vl_image_tokens(
                prompt_tokens,
                &spatial_shapes,
                youtu.spatial_merge_size,
                youtu.vision_start_token_id,
                youtu.vision_end_token_id,
                youtu.image_token_id,
            )
            .map(|stats| VlmPreparationSummary::YoutuVL {
                image_blocks: stats.image_blocks,
                total_image_tokens: stats.total_image_tokens,
            });

            // Drop any opportunistic cache wiring for now — Youtu-VL processes
            // every image's pixels in one tower call, but we can revisit later
            // to avoid recomputation across multi-turn requests sharing the
            // same image set. Skipping the cache here keeps the first
            // integration small and easy to validate.
            let _ = active_caches;
            let _ = image_cache_keys;

            let input_ids_arr = prompt_ids_array(prompt_tokens);
            let embeddings =
                youtu.get_input_embeddings(&input_ids_arr, &pixel_values, &spatial_shapes);

            Ok(Some(PreparedVlmEmbeddings {
                embeddings,
                preparation,
            }))
        }
        VlmRuntimeRef::InternVL(internvl) => {
            // Dynamic tiling: each image becomes `tiles` 448x448 tiles
            // (plus a thumbnail when split). Each tile contributes
            // `num_image_token` (256) image-feature tokens.
            let (pixel_values, tiles_per_image) = internvl.processor.preprocess_with_tiles(images);

            let preparation = insert_internvl_image_tokens(
                prompt_tokens,
                &tiles_per_image,
                internvl.num_image_token,
                internvl.img_start_token_id,
                internvl.image_context_token_id,
                internvl.img_end_token_id,
            )
            .map(|stats| VlmPreparationSummary::InternVL {
                image_blocks: stats.image_blocks,
                total_image_tokens: stats.total_image_tokens,
            });

            // InternVL processes all tiles for the request in one tower call;
            // skip the opportunistic vision cache for the first integration
            // (mirrors the Youtu-VL decision).
            let _ = active_caches;
            let _ = image_cache_keys;

            let input_ids_arr = prompt_ids_array(prompt_tokens);
            let embeddings = internvl.get_input_embeddings(&input_ids_arr, &pixel_values);

            Ok(Some(PreparedVlmEmbeddings {
                embeddings,
                preparation,
            }))
        }
        VlmRuntimeRef::Standard(vision_module) => {
            let preparation = model
                .image_token_block_info()
                .and_then(|info| apply_image_token_blocks(prompt_tokens, info, images.len()))
                .map(VlmPreparationSummary::ImageBlocks);

            let pixel_values = vision_module.processor.preprocess(images);
            let mask =
                mlxcel_core::ones(&[1, prompt_tokens.len() as i32], mlxcel_core::dtype::INT32);
            let input_ids_arr = prompt_ids_array(prompt_tokens);
            let embeddings = vision_module.get_input_embeddings(
                model,
                &input_ids_arr,
                Some(&pixel_values),
                &mask,
            )?;

            Ok(Some(PreparedVlmEmbeddings {
                embeddings,
                preparation,
            }))
        }
    }
}

fn format_molmo_v1_prompt_for_processor(prompt: &str) -> String {
    let without_image_placeholder = prompt.replace("<|image|>", "");
    let prompt = without_image_placeholder.trim();
    if prompt.starts_with("User:") && prompt.ends_with("Assistant:") {
        format!(" {prompt}")
    } else {
        format!(" User: {prompt} Assistant:")
    }
}

fn shift_molmo_v1_image_input_idx_for_bos(image_input_idx: &[i32]) -> Vec<i32> {
    image_input_idx
        .iter()
        .map(|&idx| if idx < 0 { idx } else { idx + 1 })
        .collect()
}

fn expand_gemma4_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    image_token_id: i32,
    boi_token_id: i32,
    eoi_token_id: i32,
    num_soft_tokens: &[usize],
) -> Result<()> {
    if prompt_tokens.is_empty() || num_soft_tokens.is_empty() {
        return Ok(());
    }

    let placeholder_count = prompt_tokens
        .iter()
        .filter(|&&token| token == image_token_id || token == boi_token_id)
        .count();

    if placeholder_count > 0 {
        if placeholder_count != num_soft_tokens.len() {
            return Err(anyhow::anyhow!(
                "Gemma4 prompt has {} image placeholder(s) but {} image(s) were provided",
                placeholder_count,
                num_soft_tokens.len()
            ));
        }

        let mut expanded = Vec::new();
        let mut soft_tokens = num_soft_tokens.iter();
        for &token in prompt_tokens.iter() {
            if token == image_token_id || token == boi_token_id {
                let count = *soft_tokens.next().ok_or_else(|| {
                    anyhow::anyhow!("Gemma4 soft-token expansion ran out of images")
                })?;
                expanded.push(boi_token_id);
                expanded.extend(std::iter::repeat_n(image_token_id, count));
                expanded.push(eoi_token_id);
            } else {
                expanded.push(token);
            }
        }
        *prompt_tokens = expanded;
        return Ok(());
    }

    let mut image_tokens = Vec::new();
    for &count in num_soft_tokens {
        image_tokens.push(boi_token_id);
        image_tokens.extend(std::iter::repeat_n(image_token_id, count));
        image_tokens.push(eoi_token_id);
    }

    let bos = prompt_tokens[0];
    let rest = prompt_tokens[1..].to_vec();
    *prompt_tokens = vec![bos];
    prompt_tokens.extend(image_tokens);
    prompt_tokens.extend(rest);
    Ok(())
}

/// Public wrapper for Gemma4 image token expansion.
/// Used by generate_vlm when combining image + audio inputs.
pub fn expand_gemma4_image_tokens_pub(
    prompt_tokens: &mut Vec<i32>,
    image_token_id: i32,
    boi_token_id: i32,
    eoi_token_id: i32,
    num_soft_tokens: &[usize],
) -> Result<()> {
    expand_gemma4_image_tokens(
        prompt_tokens,
        image_token_id,
        boi_token_id,
        eoi_token_id,
        num_soft_tokens,
    )
}

/// Expand a Gemma 4 video token placeholder into per-frame
/// `<boi> image_token * N <eoi>` runs.
///
/// Mirrors the upstream Python `Gemma4Processor.__call__(videos=...)`
/// expansion: every `video_token_id` in `prompt_tokens` is replaced by
/// the concatenation of its corresponding video's frames, where each
/// frame contributes `<boi>` + `image_token_id × num_soft_tokens_per_frame`
/// + `<eoi>`.
///
/// When the prompt does not contain a `video_token_id` (e.g. the chat
/// template did not insert one — common for the basic CLI path that
/// just appends `--video` files), the expansion runs are inserted
/// after the BOS token, in the same position the image expansion path
/// uses for fallback insertion.
///
/// # Errors
/// Returns `Err` if the prompt has more `video_token_id` placeholders
/// than the number of supplied videos.
pub fn expand_gemma4_video_tokens(
    prompt_tokens: &mut Vec<i32>,
    video_token_id: i32,
    image_token_id: i32,
    boi_token_id: i32,
    eoi_token_id: i32,
    soft_tokens_per_frame_per_video: &[Vec<usize>],
) -> Result<()> {
    if prompt_tokens.is_empty() || soft_tokens_per_frame_per_video.is_empty() {
        return Ok(());
    }

    let placeholder_count = prompt_tokens
        .iter()
        .filter(|&&t| t == video_token_id)
        .count();

    let build_video_run = |frames: &[usize]| -> Vec<i32> {
        let total_len: usize = frames.iter().map(|n| n + 2).sum();
        let mut run = Vec::with_capacity(total_len);
        for &n in frames {
            run.push(boi_token_id);
            run.extend(std::iter::repeat_n(image_token_id, n));
            run.push(eoi_token_id);
        }
        run
    };

    if placeholder_count > 0 {
        if placeholder_count != soft_tokens_per_frame_per_video.len() {
            return Err(anyhow::anyhow!(
                "Gemma4 prompt has {} video placeholder(s) but {} video(s) were provided",
                placeholder_count,
                soft_tokens_per_frame_per_video.len()
            ));
        }
        let mut runs = soft_tokens_per_frame_per_video.iter();
        let mut expanded = Vec::with_capacity(prompt_tokens.len());
        for &token in prompt_tokens.iter() {
            if token == video_token_id {
                let frames = runs
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("video run iterator drained early"))?;
                expanded.extend(build_video_run(frames));
            } else {
                expanded.push(token);
            }
        }
        *prompt_tokens = expanded;
        return Ok(());
    }

    // No placeholder in prompt — splice all video runs in after BOS.
    let bos = prompt_tokens[0];
    let rest: Vec<i32> = prompt_tokens[1..].to_vec();
    let mut expanded = vec![bos];
    for frames in soft_tokens_per_frame_per_video {
        expanded.extend(build_video_run(frames));
    }
    expanded.extend(rest);
    *prompt_tokens = expanded;
    Ok(())
}

/// Expand a Gemma 4 **Unified** video placeholder into per-frame
/// `<boi> video_token * N <eoi>` runs.
///
/// Differs from [`expand_gemma4_video_tokens`] (the ViT-backed `gemma4` VLM
/// path) in that each frame's soft tokens are `video_token_id`, not
/// `image_token_id`: the encoder-free unified model scatters per-frame video
/// features into `video_token_id` placeholder spans (issue #164). The boi/eoi
/// framing per frame matches the upstream Gemma 4 processor, which emits
/// `{boi}{video_token * n}{eoi}` per frame (see
/// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/processing_gemma4.py).
///
/// When the prompt already carries `video_token_id` placeholders (one per
/// video, inserted by the chat template's `<|video|>`), each is replaced in
/// prompt order by its video's per-frame runs. The replacement scan iterates
/// the original token stream, so the `video_token_id` tokens introduced by the
/// runs are not re-scanned. When the prompt has no placeholder (the basic CLI
/// path), every video's runs are spliced in after the BOS token, mirroring the
/// image-token fallback.
///
/// # Errors
/// Returns `Err` if the prompt has a different number of `video_token_id`
/// placeholders than the number of supplied videos.
pub fn expand_gemma4_unified_video_tokens(
    prompt_tokens: &mut Vec<i32>,
    video_token_id: i32,
    boi_token_id: i32,
    eoi_token_id: i32,
    soft_tokens_per_frame_per_video: &[Vec<usize>],
) -> Result<()> {
    if prompt_tokens.is_empty() || soft_tokens_per_frame_per_video.is_empty() {
        return Ok(());
    }

    let placeholder_count = prompt_tokens
        .iter()
        .filter(|&&t| t == video_token_id)
        .count();

    let build_video_run = |frames: &[usize]| -> Vec<i32> {
        let total_len: usize = frames.iter().map(|n| n + 2).sum();
        let mut run = Vec::with_capacity(total_len);
        for &n in frames {
            run.push(boi_token_id);
            run.extend(std::iter::repeat_n(video_token_id, n));
            run.push(eoi_token_id);
        }
        run
    };

    if placeholder_count > 0 {
        if placeholder_count != soft_tokens_per_frame_per_video.len() {
            return Err(anyhow::anyhow!(
                "Gemma4 Unified prompt has {} video placeholder(s) but {} video(s) were provided",
                placeholder_count,
                soft_tokens_per_frame_per_video.len()
            ));
        }
        let mut runs = soft_tokens_per_frame_per_video.iter();
        let mut expanded = Vec::with_capacity(prompt_tokens.len());
        for &token in prompt_tokens.iter() {
            if token == video_token_id {
                let frames = runs
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("video run iterator drained early"))?;
                expanded.extend(build_video_run(frames));
            } else {
                expanded.push(token);
            }
        }
        *prompt_tokens = expanded;
        return Ok(());
    }

    // No placeholder in prompt: splice all video runs in after BOS.
    let bos = prompt_tokens[0];
    let rest: Vec<i32> = prompt_tokens[1..].to_vec();
    let mut expanded = vec![bos];
    for frames in soft_tokens_per_frame_per_video {
        expanded.extend(build_video_run(frames));
    }
    expanded.extend(rest);
    *prompt_tokens = expanded;
    Ok(())
}

/// Expand audio token placeholder in prompt tokens for server requests.
///
/// Replaces the first `audio_token_id` with `boa + audio_token*N + eoa`.
/// If no audio placeholder is found, inserts before the last token.
pub fn expand_gemma4_audio_tokens_for_server(
    prompt_tokens: &mut Vec<i32>,
    audio_token_id: i32,
    boa_token_id: i32,
    eoa_token_id: i32,
    num_audio_tokens: usize,
) {
    let mut expanded = Vec::with_capacity(prompt_tokens.len() + num_audio_tokens + 2);
    let mut found = false;
    for &token in prompt_tokens.iter() {
        if token == audio_token_id && !found {
            found = true;
            expanded.push(boa_token_id);
            expanded.extend(std::iter::repeat_n(audio_token_id, num_audio_tokens));
            expanded.push(eoa_token_id);
        } else {
            expanded.push(token);
        }
    }
    // If no placeholder found, insert before last token
    if !found {
        let last = expanded.pop();
        expanded.push(boa_token_id);
        expanded.extend(std::iter::repeat_n(audio_token_id, num_audio_tokens));
        expanded.push(eoa_token_id);
        if let Some(tok) = last {
            expanded.push(tok);
        }
    }
    *prompt_tokens = expanded;
}

#[cfg(test)]
#[path = "vlm_runtime_tests.rs"]
mod tests;
