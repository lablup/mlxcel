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

//! CLI driver for the two-stage IndicOCR page pipeline.
//!
//! `mlxcel generate -m <indic-ocr-root> --image page.png` runs IndicDocLayout
//! (a PP-DocLayoutV3 fine-tune with mlx-vlm decode semantics), cleans the
//! layout, crops each transcribable region into the area band, OCRs it with
//! the stage's Qwen3.5 model and one instruction per block type, then prints
//! reading-ordered Markdown. `--layout-detections FILE` replaces stage 1.
//!
//! The pure routing, cleanup and reconstruction rules live in
//! [`mlxcel::vision::indic_ocr`]; this file owns model I/O only.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, ensure};
use image::DynamicImage;
use mlxcel::vision::detection::pp_doclayout_v3::{
    MLX_VLM_DEFAULT_CONF, MLX_VLM_DEFAULT_IMG_SIZE, PpDocLayoutDecode, PpDocLayoutV3Predictor,
};
use mlxcel::vision::indic_ocr::{
    Block, MAX_CROP_PIXELS, MIN_CROP_PIXELS, NestMode, TableFormat, blocks_from_detections,
    build_ocr_requests, is_transcribed, prepare_blocks, reconstruct, resolve_nested_equations,
};
use mlxcel::{SamplingConfig, select_backend};
use mlxcel_core::sampling::TokenBiasMap;

use super::generate::{CliPromptMedia, load_cli_prompt, run_generation_mode};
use crate::GenerateArgs;

/// Default per-block token budget, matching `RecognizerOptions.max_tokens`.
const DEFAULT_MAX_TOKENS: usize = 2048;

/// Is this directory a two-stage IndicOCR repository?
pub(crate) fn is_indic_ocr_root(directory: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(directory.join("config.json")) else {
        return false;
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    config.get("model_type").and_then(|value| value.as_str()) == Some("indic_ocr")
        && config.get("stages").is_some()
}

/// Resolve the layout and OCR stage directories from the root config.
fn stage_directories(root: &Path, config: &serde_json::Value) -> Result<(PathBuf, PathBuf)> {
    let layout = stage_config_path(root, config, "layout", "weights/layout/config.json")?;
    let ocr_relative = config
        .get("ocr_model_path")
        .and_then(|value| value.as_str())
        .unwrap_or("weights/ocr");
    let ocr = {
        let candidate = root.join(ocr_relative);
        if candidate.is_dir() {
            candidate
        } else {
            stage_config_path(root, config, "ocr", "weights/ocr/config.json")?
        }
    };
    ensure!(
        ocr.join("config.json").is_file(),
        "IndicOCR OCR stage config not found at {}",
        ocr.join("config.json").display()
    );
    Ok((layout, ocr))
}

fn stage_config_path(
    root: &Path,
    config: &serde_json::Value,
    stage: &str,
    default: &str,
) -> Result<PathBuf> {
    let relative = config
        .pointer(&format!("/stages/{stage}/config"))
        .and_then(|value| value.as_str())
        .unwrap_or(default);
    let candidate = root.join(relative);
    let directory = if candidate.is_dir() {
        candidate
    } else {
        candidate
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| root.to_path_buf())
    };
    ensure!(
        directory.join("config.json").is_file(),
        "IndicOCR {stage} stage config not found at {}",
        directory.join("config.json").display()
    );
    Ok(directory)
}

/// Run the whole page pipeline for a two-stage IndicOCR repository.
pub(crate) fn run_indic_ocr_pipeline(args: &GenerateArgs) -> Result<()> {
    let generation = &args.generation;
    ensure!(
        generation.image.len() == 1,
        "IndicOCR processes one page at a time: pass exactly one --image, got {}",
        generation.image.len()
    );
    ensure!(
        generation.audio.is_none() && generation.video.is_empty(),
        "IndicOCR is an image-only pipeline; drop --audio / --video"
    );
    ensure!(
        args.model.draft_model.is_none(),
        "IndicOCR does not support speculative decoding; drop --draft-model"
    );
    ensure!(
        args.tensor_parallel.tp_size <= 1,
        "IndicOCR does not support tensor parallelism"
    );
    if generation
        .prompt
        .as_deref()
        .is_some_and(|prompt| !prompt.is_empty())
    {
        eprintln!(
            "NOTE: IndicOCR derives each region's instruction from its layout class, so \
             -p/--prompt is not used."
        );
    }
    let table_format =
        TableFormat::parse(&generation.table_format).map_err(|error| anyhow!(error))?;
    let max_tokens = if generation.max_tokens == mlxcel::cli::max_tokens::UNLIMITED_MAX_TOKENS {
        DEFAULT_MAX_TOKENS
    } else {
        generation.max_tokens
    };

    let root = &args.model.model;
    let config_text = std::fs::read_to_string(root.join("config.json"))
        .with_context(|| format!("failed to read {}", root.join("config.json").display()))?;
    let config: serde_json::Value = serde_json::from_str(&config_text)
        .map_err(|error| anyhow!("invalid IndicOCR root config: {error}"))?;
    let (layout_dir, ocr_dir) = stage_directories(root, &config)?;

    // Parse an explicit layout before any weight is loaded.
    let replays_layout = generation.layout_detections.is_some();
    let explicit_detections = match generation.layout_detections.as_deref() {
        Some(path) => Some(super::generate_falcon_ocr::load_layout_detections(path)?),
        None => None,
    };

    let image_path = &generation.image[0];
    let bytes = std::fs::read(image_path)
        .with_context(|| format!("Failed to read image {image_path:?}"))?;
    let mut images =
        mlxcel::decode_image_payloads_with_limits(&[bytes], mlxcel::current_image_input_limits())
            .with_context(|| format!("Failed to decode image {image_path:?}"))?;
    let page: DynamicImage = images
        .pop()
        .ok_or_else(|| anyhow!("Image decoding returned no image for {image_path:?}"))?;
    let (width, height) = (page.width(), page.height());

    let _runtime = mlxcel::initialize_runtime_checked()?;

    let started = Instant::now();
    let mut blocks = match explicit_detections {
        Some(detections) => prepare_blocks(blocks_from_detections(&detections, width, height)),
        None => {
            let predictor = PpDocLayoutV3Predictor::from_pretrained_with_decode(
                &layout_dir,
                MLX_VLM_DEFAULT_CONF,
                PpDocLayoutDecode::MlxVlmDetect {
                    conf: MLX_VLM_DEFAULT_CONF,
                    img_size: MLX_VLM_DEFAULT_IMG_SIZE,
                },
            )
            .map_err(|error| anyhow!("failed to load IndicOCR layout stage: {error}"))?;
            let result = predictor
                .detect_mlx_vlm_image(&page, MLX_VLM_DEFAULT_CONF, MLX_VLM_DEFAULT_IMG_SIZE)
                .map_err(|error| anyhow!("IndicOCR layout detection failed: {error}"))?;
            prepare_blocks(blocks_from_detections(&result.detections, width, height))
        }
    };
    if replays_layout {
        println!(
            "IndicOCR: replayed {} layout block(s) from --layout-detections",
            blocks.len()
        );
    }

    // Nested equations are excluded from OCR but stay in the output with empty
    // text, exactly as the reference does.
    let by_order: HashMap<usize, usize> = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.order, index))
        .collect();
    let eligible: Vec<usize> = blocks
        .iter()
        .enumerate()
        .filter(|(_, block)| is_transcribed(&block.label))
        .map(|(index, _)| index)
        .collect();
    let eligible_blocks: Vec<Block> = eligible
        .iter()
        .map(|&index| blocks[index].clone())
        .collect();
    let surviving = resolve_nested_equations(&eligible_blocks, true, NestMode::Both, 0.70);
    let transcribable: Vec<usize> = surviving
        .iter()
        .filter_map(|block| by_order.get(&block.order).copied())
        .collect();

    let requests = build_ocr_requests(
        &blocks,
        &page,
        table_format,
        0,
        MIN_CROP_PIXELS,
        MAX_CROP_PIXELS,
    );
    let requests: Vec<_> = requests
        .into_iter()
        .filter(|request| transcribable.contains(&request.block))
        .collect();

    let backend = select_backend();
    let (ocr_model, tokenizer) = backend
        .load_model(&ocr_dir)
        .map_err(|error| anyhow!("failed to load IndicOCR OCR stage: {error}"))?;
    let kv_cache_mode = mlxcel::cli::turbo_args::resolve_and_announce_kv_cache_mode(
        mlxcel::cli::turbo_args::resolve_kv_cache_mode(
            generation.turbo.cache_type_k.as_deref(),
            generation.turbo.cache_type_v.as_deref(),
            generation.turbo.kv_cache_mode.as_deref(),
        )
        .map_err(|error| anyhow!("{error}"))?,
        &ocr_dir,
    );

    let sampling = SamplingConfig::greedy();
    let mut generated_tokens = 0usize;
    let total_blocks = blocks.len();
    for request in &requests {
        let prompt = load_cli_prompt(
            &ocr_dir,
            &tokenizer,
            request.prompt,
            generation.no_chat_template,
            &CliPromptMedia {
                images: 1,
                video_frame_groups: Vec::new(),
                videos: 0,
                audios: 0,
            },
        )?;
        let mut prompt_tokens: Vec<i32> = tokenizer
            .encode(&prompt, true)
            .map_err(|error| anyhow!("Tokenization failed: {error}"))?
            .iter()
            .map(|&token| token as i32)
            .collect();
        let prepared = mlxcel::vlm_runtime::prepare_vlm_embeddings_with_budget(
            &ocr_model,
            &mut prompt_tokens,
            &prompt,
            std::slice::from_ref(&request.crop),
            None,
            |text, add_special| {
                tokenizer
                    .encode(text, add_special)
                    .unwrap_or_default()
                    .iter()
                    .map(|&token| token as i32)
                    .collect()
            },
        )?;
        let embeddings = prepared.as_ref().map(|prepared| &prepared.embeddings);
        let (tokens, _stats) = run_generation_mode(
            &ocr_model,
            args,
            &prompt_tokens,
            max_tokens,
            &sampling,
            embeddings,
            kv_cache_mode,
            TokenBiasMap::new(),
        )?;
        generated_tokens += tokens.len();
        let answer = tokenizer
            .decode(
                &tokens.iter().map(|&token| token as u32).collect::<Vec<_>>(),
                true,
            )
            .unwrap_or_default();
        blocks[request.block].text = Some(clean_answer(&answer));
    }
    for block in &mut blocks {
        if block.text.is_none() {
            block.text = Some(String::new());
        }
    }

    let markdown = reconstruct(&blocks, true);
    println!("{markdown}");

    if let Some(path) = generation.output_json.as_deref() {
        write_page_json(path, image_path, width, height, &blocks)?;
    }

    let elapsed = started.elapsed().as_secs_f64();
    let rate = if elapsed > 0.0 {
        generated_tokens as f64 / elapsed
    } else {
        0.0
    };
    eprintln!(
        "[IndicOCR: {} block(s), {} transcribed, {} tokens in {:.2}s = {:.2} tok/s]",
        total_blocks,
        requests.len(),
        generated_tokens,
        elapsed,
        rate
    );
    mlxcel_core::clear_memory_cache();
    Ok(())
}

/// Keep only the answer channel: strip a Qwen ` thinking...</think>` prefix and
/// trim. The OCR stage occasionally reasons before transcribing, and those
/// tokens are not part of the region's text.
fn clean_answer(text: &str) -> String {
    let content = match text.rsplit_once("</think>") {
        Some((_, answer)) => answer,
        None if text.trim_start().starts_with(" thinking") => "",
        None => text,
    };
    content.trim().to_string()
}

fn write_page_json(
    path: &Path,
    image_path: &Path,
    width: u32,
    height: u32,
    blocks: &[Block],
) -> Result<()> {
    let records: Vec<serde_json::Value> = blocks
        .iter()
        .map(|block| {
            serde_json::json!({
                "order": block.order,
                "label": block.label,
                "type": block.block_type,
                "bbox_xyxy": [
                    block.bbox[0].round(),
                    block.bbox[1].round(),
                    block.bbox[2].round(),
                    block.bbox[3].round(),
                ],
                "conf": block.conf,
                "text": block.text.clone().unwrap_or_default(),
            })
        })
        .collect();
    let document = serde_json::json!({
        "image": image_path.display().to_string(),
        "width": width,
        "height": height,
        "blocks": records,
    });
    let contents = serde_json::to_string_pretty(&document)?;
    std::fs::write(path, contents)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_config(directory: &Path, value: &serde_json::Value) {
        std::fs::create_dir_all(directory).unwrap();
        std::fs::write(directory.join("config.json"), value.to_string()).unwrap();
    }

    #[test]
    fn root_detection_requires_both_model_type_and_stages() {
        let root =
            std::env::temp_dir().join(format!("mlxcel-indic-root-detect-{}", std::process::id()));
        write_config(
            &root,
            &serde_json::json!({"model_type": "indic_ocr", "stages": {"layout": {}}}),
        );
        assert!(is_indic_ocr_root(&root));
        write_config(&root, &serde_json::json!({"model_type": "indic_ocr"}));
        assert!(!is_indic_ocr_root(&root));
        write_config(&root, &serde_json::json!({"model_type": "qwen3_5"}));
        assert!(!is_indic_ocr_root(&root));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stage_resolution_reads_stages_block_and_ocr_model_path() {
        let root =
            std::env::temp_dir().join(format!("mlxcel-indic-stage-dirs-{}", std::process::id()));
        write_config(&root, &serde_json::json!({"model_type": "indic_ocr"}));
        write_config(
            &root.join("weights/layout"),
            &serde_json::json!({"model_type": "pp_doclayout_v3"}),
        );
        write_config(
            &root.join("weights/ocr"),
            &serde_json::json!({"model_type": "indic_ocr", "text_config": {}}),
        );
        let config = serde_json::json!({
            "stages": {
                "layout": {"config": "weights/layout/config.json"},
                "ocr": {"config": "weights/ocr/config.json"}
            },
            "ocr_model_path": "weights/ocr"
        });
        let (layout, ocr) = stage_directories(&root, &config).unwrap();
        assert_eq!(layout, root.join("weights/layout"));
        assert_eq!(ocr, root.join("weights/ocr"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_missing_stage_directory_names_the_path() {
        let root =
            std::env::temp_dir().join(format!("mlxcel-indic-missing-stage-{}", std::process::id()));
        write_config(&root, &serde_json::json!({"model_type": "indic_ocr"}));
        let config = serde_json::json!({
            "stages": {"layout": {"config": "weights/layout/config.json"}}
        });
        let error = stage_directories(&root, &config).unwrap_err().to_string();
        assert!(error.contains("layout"), "{error}");
        std::fs::remove_dir_all(root).unwrap();
    }
}
