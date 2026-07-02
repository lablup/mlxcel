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

use anyhow::Result;
use std::path::{Path, PathBuf};

use mlxcel::LoadedModel;
use mlxcel::video;
use mlxcel::vision::merge::InputEmbeddings;
use mlxcel::vlm_prompt::ImageTokenBlockAction;
use mlxcel::vlm_runtime::{VlmPreparationSummary, prepare_and_compute_vlm_embeddings};

use crate::MlxcelTokenizer;

fn print_preparation_summary(summary: VlmPreparationSummary) {
    match summary {
        VlmPreparationSummary::QwenVlm {
            image_blocks,
            total_image_tokens,
        } => {
            println!(
                "Inserted {} Qwen VL image token blocks ({} total image tokens)",
                image_blocks, total_image_tokens
            );
        }
        VlmPreparationSummary::MiniCPMO {
            image_slots,
            total_tokens,
        } => {
            println!(
                "MiniCPM-o: tokenized with {} image slots ({} total tokens)",
                image_slots, total_tokens
            );
        }
        VlmPreparationSummary::Moondream3 {
            mode,
            total_tokens,
            prefix_tokens,
        } => {
            println!(
                "Moondream3: prepared {:?} prompt ({} text tokens, {} image-prefix tokens)",
                mode, total_tokens, prefix_tokens
            );
        }
        VlmPreparationSummary::Moondream2 {
            mode,
            total_tokens,
            prefix_tokens,
        } => {
            println!(
                "Moondream2: prepared {:?} prompt ({} text tokens, {} image-prefix tokens)",
                mode, total_tokens, prefix_tokens
            );
        }
        VlmPreparationSummary::Gemma4 {
            image_slots,
            total_tokens,
        } => {
            println!(
                "Gemma4: expanded {} image slot(s) into dynamic soft tokens ({} total tokens)",
                image_slots, total_tokens
            );
        }
        VlmPreparationSummary::Gemma4Audio {
            audio_tokens,
            total_tokens,
        } => {
            println!(
                "Gemma4: expanded audio into {} soft tokens ({} total tokens)",
                audio_tokens, total_tokens
            );
        }
        VlmPreparationSummary::Gemma4Video {
            video_count,
            frame_slots,
            total_tokens,
        } => {
            println!(
                "Gemma4: expanded {} video(s) into {} frame slot(s) ({} total tokens)",
                video_count, frame_slots, total_tokens
            );
        }
        VlmPreparationSummary::Phi4MM {
            image_slots,
            total_tokens,
        } => {
            println!(
                "Phi4MM: tokenized with {} image slots ({} total tokens)",
                image_slots, total_tokens
            );
        }
        VlmPreparationSummary::Molmo { total_tokens } => {
            println!(
                "Molmo: expanded prompt with image tokens ({} total tokens)",
                total_tokens
            );
        }
        VlmPreparationSummary::Molmo2 { total_tokens } => {
            println!(
                "Molmo2: expanded prompt with image tokens ({} total tokens)",
                total_tokens
            );
        }
        VlmPreparationSummary::MolmoPoint { total_tokens } => {
            println!(
                "Molmo-Point: expanded prompt with image tokens ({} total tokens)",
                total_tokens
            );
        }
        VlmPreparationSummary::Phi3V {
            image_slots,
            total_tokens,
        } => {
            println!(
                "Phi3V: tokenized with {} image slots ({} total tokens)",
                image_slots, total_tokens
            );
        }
        VlmPreparationSummary::Phi4SigLip {
            image_slots,
            total_tokens,
        } => {
            println!(
                "Phi4-SigLIP: tokenized with {} image slots ({} total tokens)",
                image_slots, total_tokens
            );
        }
        VlmPreparationSummary::NemotronHNanoOmni {
            image_slots,
            total_tokens,
        } => {
            println!(
                "Nemotron H Nano Omni: tokenized with {} image slot(s) ({} total tokens)",
                image_slots, total_tokens
            );
        }
        VlmPreparationSummary::NemotronHNanoOmniAudio {
            audio_clips,
            audio_tokens,
            total_tokens,
        } => {
            println!(
                "Nemotron H Nano Omni audio: tokenized {} clip(s) into {} audio token(s) ({} total tokens)",
                audio_clips, audio_tokens, total_tokens
            );
        }
        VlmPreparationSummary::YoutuVL {
            image_blocks,
            total_image_tokens,
        } => {
            println!(
                "Youtu-VL: inserted {} image block(s) ({} total image tokens)",
                image_blocks, total_image_tokens
            );
        }
        VlmPreparationSummary::InternVL {
            image_blocks,
            total_image_tokens,
        } => {
            println!(
                "InternVL: inserted {} image block(s) ({} total image tokens)",
                image_blocks, total_image_tokens
            );
        }
        VlmPreparationSummary::SmolVLM {
            image_blocks,
            total_image_tokens,
        } => {
            println!(
                "SmolVLM: inserted {} image block(s) ({} total image tokens)",
                image_blocks, total_image_tokens
            );
        }
        VlmPreparationSummary::KimiVL {
            image_blocks,
            total_image_tokens,
        } => {
            println!(
                "Kimi-VL: expanded {} media placeholder(s) ({} total image tokens)",
                image_blocks, total_image_tokens
            );
        }
        VlmPreparationSummary::ImageBlocks(stats) => match stats.action {
            ImageTokenBlockAction::Expanded {
                existing_image_count,
            } => {
                println!(
                    "Expanded {} <image> token(s) to {} tokens each",
                    existing_image_count, stats.tokens_per_image
                );
            }
            ImageTokenBlockAction::Inserted { image_blocks } => {
                println!(
                    "Inserted {} image token blocks ({} tokens each)",
                    image_blocks, stats.tokens_per_image
                );
            }
        },
    }
}

pub(crate) fn compute_vlm_embeddings(
    model: &LoadedModel,
    prompt_tokens: &mut Vec<i32>,
    prompt: &str,
    image_paths: &[PathBuf],
    audio_path: Option<&Path>,
    video_paths: &[PathBuf],
    target_fps: f64,
    tokenizer: &MlxcelTokenizer,
) -> Result<Option<InputEmbeddings>> {
    // Handle video-only or video + image mode for Gemma4.
    // Video and audio cannot coexist in this CLI surface yet, surface a
    // clean error rather than silently accept one.
    if !video_paths.is_empty() {
        if audio_path.is_some() {
            return Err(anyhow::anyhow!(
                "Combined --video and --audio inputs are not supported yet"
            ));
        }
        if let LoadedModel::Gemma4VLM(gemma4_vl) = model {
            return compute_gemma4_video_embeddings(
                gemma4_vl,
                prompt_tokens,
                image_paths,
                video_paths,
                target_fps,
            );
        }
        if let LoadedModel::Gemma4Unified(unified) = model {
            return compute_gemma4_unified_video_embeddings(
                unified,
                prompt_tokens,
                image_paths,
                video_paths,
                target_fps,
            );
        }
        return Err(anyhow::anyhow!(
            "--video input is currently only supported by Gemma4 VLMs"
        ));
    }

    // Handle audio-only mode for Gemma4
    if image_paths.is_empty()
        && let Some(audio) = audio_path
        && let LoadedModel::Gemma4VLM(gemma4_vl) = model
    {
        return compute_gemma4_audio_embeddings(gemma4_vl, prompt_tokens, audio);
    }

    // Handle combined image + audio for Gemma4
    if !image_paths.is_empty()
        && let Some(audio) = audio_path
        && let LoadedModel::Gemma4VLM(gemma4_vl) = model
    {
        return compute_gemma4_multimodal_embeddings(gemma4_vl, prompt_tokens, image_paths, audio);
    }

    // Handle audio-only and image+audio for Gemma 4 Unified (encoder-free).
    if let Some(audio) = audio_path
        && let LoadedModel::Gemma4Unified(unified) = model
    {
        return compute_gemma4_unified_multimodal_embeddings(
            unified,
            prompt_tokens,
            image_paths,
            audio,
        );
    }

    // Nemotron H Nano Omni: audio-only or combined image + audio
    // Mirrors the Gemma 4 dispatch above; the helper
    // handles both branches so a single match arm covers both modes.
    if let Some(audio) = audio_path
        && let LoadedModel::NemotronHNanoOmniVLM(nemotron_vl) = model
    {
        return compute_nemotron_h_nano_omni_audio_embeddings(
            nemotron_vl,
            prompt_tokens,
            image_paths,
            audio,
        );
    }

    // Reject `--audio` for any remaining VLM that does not have a
    // dedicated dispatch above. Without this guard, `--audio` would be
    // silently dropped and the runtime would emit text-only output,
    // which is worse than a clear error since the user supplied data
    // they expect the model to consume.
    if audio_path.is_some() {
        return Err(anyhow::anyhow!(
            "--audio input is not supported for this model family. Currently audio is wired \
             through Gemma 4 and Nemotron H Nano Omni VLMs only."
        ));
    }

    if image_paths.is_empty() {
        // Moondream3 needs special prompt formatting even for text-only
        if matches!(model, LoadedModel::Moondream3VLM(_)) {
            let prepared = mlxcel::moondream3_prompt::prepare_moondream3_prompt_tokens(
                prompt,
                0,
                |text, add_special| {
                    tokenizer
                        .encode(text, add_special)
                        .unwrap_or_default()
                        .iter()
                        .map(|&t| t as i32)
                        .collect()
                },
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?;
            *prompt_tokens = prepared.tokens;
        } else if let LoadedModel::Moondream2VLM(moondream2) = model {
            // Moondream2 also needs its own text-only framing.
            let prepared = mlxcel::moondream2_prompt::prepare_moondream2_prompt_tokens(
                prompt,
                0,
                moondream2.prompt_style,
                |text, add_special| {
                    tokenizer
                        .encode(text, add_special)
                        .unwrap_or_default()
                        .iter()
                        .map(|&t| t as i32)
                        .collect()
                },
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?;
            *prompt_tokens = prepared.tokens;
        }
        return Ok(None);
    }

    let images: Vec<image::DynamicImage> = image_paths
        .iter()
        .map(|path| {
            image::open(path).map_err(|e| anyhow::anyhow!("Failed to load image {:?}: {}", path, e))
        })
        .collect::<Result<Vec<_>>>()?;
    println!("Loaded {} image(s).", images.len());

    let prepared = prepare_and_compute_vlm_embeddings(
        model,
        prompt_tokens,
        prompt,
        &images,
        |text, add_special| {
            tokenizer
                .encode(text, add_special)
                .unwrap_or_default()
                .iter()
                .map(|&t| t as i32)
                .collect()
        },
    )?;

    if let Some(prepared) = prepared {
        if let Some(summary) = prepared.preparation {
            print_preparation_summary(summary);
        }
        Ok(Some(prepared.embeddings))
    } else {
        Ok(None)
    }
}

/// Compute audio-only embeddings for Gemma4 VLM.
fn compute_gemma4_audio_embeddings(
    gemma4_vl: &mlxcel::vision::Gemma4VLModel,
    prompt_tokens: &mut Vec<i32>,
    audio_path: &Path,
) -> Result<Option<InputEmbeddings>> {
    use mlxcel::audio;

    if gemma4_vl.audio_tower.is_none() {
        return Err(anyhow::anyhow!(
            "This model does not have an audio encoder. Audio input is not supported."
        ));
    }

    // Load and process audio
    let (samples, sample_rate) =
        audio::load_wav_file(audio_path).map_err(|e| anyhow::anyhow!("{}", e))?;
    println!(
        "Loaded audio: {} samples at {} Hz ({:.1}s)",
        samples.len(),
        sample_rate,
        samples.len() as f64 / sample_rate as f64
    );

    // Compute number of audio tokens
    let num_audio_tokens = audio::compute_audio_num_tokens(
        samples.len(),
        sample_rate,
        40,  // ms_per_token
        750, // max_tokens
    );

    // Expand audio token placeholder in prompt: BOA + AUDIO*N + EOA
    expand_gemma4_audio_tokens(
        prompt_tokens,
        gemma4_vl.audio_token_id,
        gemma4_vl.boa_token_id,
        gemma4_vl.eoa_token_id,
        num_audio_tokens,
    );

    // `AudioFeatureExtractor::extract` assumes a 16 kHz waveform (160-sample
    // hop = 10 ms). `load_wav_file` returns native-rate samples (the reference
    // clips are 24 kHz), so without resampling the Conformer encoder emits
    // ~1.5x too many frames and desyncs from the duration-based placeholder
    // count above, garbling the audio embeddings and forcing an immediate EOS
    // (issue #436). Resample to 16 kHz before mel extraction; duration (and
    // thus `num_audio_tokens`) is rate-invariant, so the placeholder count
    // stays correct and now matches the encoder output length.
    let samples = if sample_rate != 16_000 {
        audio::whisper_mel::resample_to_16k(&samples, sample_rate)
    } else {
        samples
    };

    // Extract mel spectrogram
    let extractor =
        audio::AudioFeatureExtractor::new(audio::AudioFeatureExtractorConfig::default());
    let (features, mask) = extractor.extract(&samples, None);
    let num_frames = mask.len();

    // Convert to MlxArray: [1, T, 128]
    let audio_features = mlxcel_core::from_slice_f32(
        &features,
        &[1, num_frames as i32, extractor.feature_size() as i32],
    );
    // Convert mask to MlxArray: [1, T] (true = invalid)
    let mask_i32: Vec<i32> = mask.iter().map(|&b| if b { 1 } else { 0 }).collect();
    let audio_mask = mlxcel_core::from_slice_i32(&mask_i32, &[1, num_frames as i32]);
    let audio_mask = mlxcel_core::astype(&audio_mask, mlxcel_core::dtype::BOOL);

    // Compute embeddings
    let input_ids_arr =
        mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let embeddings = gemma4_vl
        .get_input_embeddings_with_audio(
            &input_ids_arr,
            &[], // no images
            Some(&audio_features),
            Some(&audio_mask),
        )
        .map_err(|e| anyhow::anyhow!(e))?;

    print_preparation_summary(VlmPreparationSummary::Gemma4Audio {
        audio_tokens: num_audio_tokens,
        total_tokens: prompt_tokens.len(),
    });

    Ok(Some(embeddings))
}

/// Compute combined image + audio embeddings for Gemma4 VLM.
fn compute_gemma4_multimodal_embeddings(
    gemma4_vl: &mlxcel::vision::Gemma4VLModel,
    prompt_tokens: &mut Vec<i32>,
    image_paths: &[PathBuf],
    audio_path: &Path,
) -> Result<Option<InputEmbeddings>> {
    use mlxcel::audio;

    if gemma4_vl.audio_tower.is_none() {
        return Err(anyhow::anyhow!(
            "This model does not have an audio encoder. Audio input is not supported."
        ));
    }

    // Process images
    let images: Vec<image::DynamicImage> = image_paths
        .iter()
        .map(|path| {
            image::open(path).map_err(|e| anyhow::anyhow!("Failed to load image {:?}: {}", path, e))
        })
        .collect::<Result<Vec<_>>>()?;
    println!("Loaded {} image(s).", images.len());

    let processed_images = gemma4_vl.processor.preprocess(&images);
    let num_soft_tokens: Vec<usize> = processed_images.iter().map(|i| i.num_soft_tokens).collect();

    // Expand image tokens
    mlxcel::vlm_runtime::expand_gemma4_image_tokens_pub(
        prompt_tokens,
        gemma4_vl.image_token_id,
        gemma4_vl.boi_token_id,
        gemma4_vl.eoi_token_id,
        &num_soft_tokens,
    )?;

    // Process audio
    let (samples, sample_rate) =
        audio::load_wav_file(audio_path).map_err(|e| anyhow::anyhow!("{}", e))?;
    println!(
        "Loaded audio: {} samples at {} Hz ({:.1}s)",
        samples.len(),
        sample_rate,
        samples.len() as f64 / sample_rate as f64
    );

    let num_audio_tokens = audio::compute_audio_num_tokens(samples.len(), sample_rate, 40, 750);
    expand_gemma4_audio_tokens(
        prompt_tokens,
        gemma4_vl.audio_token_id,
        gemma4_vl.boa_token_id,
        gemma4_vl.eoa_token_id,
        num_audio_tokens,
    );

    // Resample to 16 kHz before mel extraction (the extractor assumes 16 kHz;
    // see `compute_gemma4_audio_embeddings` and issue #436). Duration, and thus
    // the placeholder count computed above, is rate-invariant.
    let samples = if sample_rate != 16_000 {
        audio::whisper_mel::resample_to_16k(&samples, sample_rate)
    } else {
        samples
    };

    // Extract mel spectrogram
    let extractor =
        audio::AudioFeatureExtractor::new(audio::AudioFeatureExtractorConfig::default());
    let (features, mask) = extractor.extract(&samples, None);
    let num_frames = mask.len();
    let audio_features = mlxcel_core::from_slice_f32(
        &features,
        &[1, num_frames as i32, extractor.feature_size() as i32],
    );
    let mask_i32: Vec<i32> = mask.iter().map(|&b| if b { 1 } else { 0 }).collect();
    let audio_mask = mlxcel_core::from_slice_i32(&mask_i32, &[1, num_frames as i32]);
    let audio_mask = mlxcel_core::astype(&audio_mask, mlxcel_core::dtype::BOOL);

    // Compute embeddings with both images and audio
    let input_ids_arr =
        mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let embeddings = gemma4_vl
        .get_input_embeddings_with_audio(
            &input_ids_arr,
            &processed_images,
            Some(&audio_features),
            Some(&audio_mask),
        )
        .map_err(|e| anyhow::anyhow!(e))?;

    Ok(Some(embeddings))
}

/// Compute audio-only or combined image + audio embeddings for Gemma 4
/// Unified (encoder-free: waveform chunking, no mel spectrogram / Conformer).
fn compute_gemma4_unified_multimodal_embeddings(
    unified: &mlxcel::vision::Gemma4UnifiedModel,
    prompt_tokens: &mut Vec<i32>,
    image_paths: &[PathBuf],
    audio_path: &Path,
) -> Result<Option<InputEmbeddings>> {
    use mlxcel::audio;

    if unified.embed_audio.is_none() {
        return Err(anyhow::anyhow!(
            "This Gemma 4 Unified model has no audio embedder. Audio input is not supported."
        ));
    }

    // Process images (optional) through the encoder-free patch projector.
    let processed_images = if image_paths.is_empty() {
        Vec::new()
    } else {
        let images: Vec<image::DynamicImage> = image_paths
            .iter()
            .map(|path| {
                image::open(path)
                    .map_err(|e| anyhow::anyhow!("Failed to load image {:?}: {}", path, e))
            })
            .collect::<Result<Vec<_>>>()?;
        println!("Loaded {} image(s).", images.len());
        let processed = unified.processor.preprocess(&images);
        let num_soft_tokens: Vec<usize> = processed.iter().map(|i| i.num_soft_tokens).collect();
        mlxcel::vlm_runtime::expand_gemma4_image_tokens_pub(
            prompt_tokens,
            unified.image_token_id,
            unified.boi_token_id,
            unified.eoi_token_id,
            &num_soft_tokens,
        )?;
        processed
    };

    // Process audio: raw waveform chunked into audio_samples_per_token frames.
    let (samples, sample_rate) =
        audio::load_wav_file(audio_path).map_err(|e| anyhow::anyhow!("{}", e))?;
    println!(
        "Loaded audio: {} samples at {} Hz ({:.1}s)",
        samples.len(),
        sample_rate,
        samples.len() as f64 / sample_rate.max(1) as f64
    );

    let audio_input = unified.processor.process_audio(&samples);
    let num_audio_tokens = audio_input.num_frames;
    expand_gemma4_audio_tokens(
        prompt_tokens,
        unified.audio_token_id,
        unified.boa_token_id,
        unified.eoa_token_id,
        num_audio_tokens,
    );

    let input_ids_arr =
        mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let embeddings = unified.get_input_embeddings_with_audio(
        &input_ids_arr,
        &processed_images,
        Some(&audio_input.features),
        Some(&audio_input.mask),
    );

    print_preparation_summary(VlmPreparationSummary::Gemma4Audio {
        audio_tokens: num_audio_tokens,
        total_tokens: prompt_tokens.len(),
    });

    Ok(Some(embeddings))
}

/// Compute video embeddings (and optional preceding image embeddings)
/// for the Gemma 4 VLM.
///
/// Decodes each video via `mlxcel::video::load_video` (subprocess
/// `ffmpeg`), processes frames through
/// [`mlxcel::vision::processors::gemma4::Gemma4Processor::process_videos`],
/// expands `<|video|>` placeholders in the prompt into per-frame
/// `<boi> image_token*N <eoi>` runs, and dispatches the combined
/// (images + video frames) tensor through the same vision tower /
/// multimodal projector path that powers static image inputs.
fn compute_gemma4_video_embeddings(
    gemma4_vl: &mlxcel::vision::Gemma4VLModel,
    prompt_tokens: &mut Vec<i32>,
    image_paths: &[PathBuf],
    video_paths: &[PathBuf],
    target_fps: f64,
) -> Result<Option<InputEmbeddings>> {
    if !video::ffmpeg_available() {
        return Err(anyhow::anyhow!(
            "Video input requires `ffmpeg` on PATH. Install ffmpeg (e.g. `brew install ffmpeg` \
             on macOS or `apt install ffmpeg` on Linux) and retry."
        ));
    }

    // Decode the videos. `target_fps == 0` is rejected by `smart_nframes`,
    // so guard here with a clean error.
    let videos = video::load_videos(video_paths, Some(target_fps), None)
        .map_err(|err| anyhow::anyhow!("Failed to load video(s): {}", err))?;
    println!(
        "Loaded {} video(s) ({} total frames after sampling).",
        videos.len(),
        videos.iter().map(Vec::len).sum::<usize>()
    );

    // Optional companion images (e.g. user passes both --image and --video).
    let images: Vec<image::DynamicImage> = image_paths
        .iter()
        .map(|path| {
            image::open(path).map_err(|e| anyhow::anyhow!("Failed to load image {:?}: {}", path, e))
        })
        .collect::<Result<Vec<_>>>()?;
    if !images.is_empty() {
        println!("Loaded {} image(s).", images.len());
    }

    let processed_images = gemma4_vl.processor.preprocess(&images);
    let image_soft_tokens: Vec<usize> = processed_images
        .iter()
        .map(|img| img.num_soft_tokens)
        .collect();

    let fps_per_video = vec![target_fps; video_paths.len()];
    let processed_videos = gemma4_vl
        .processor
        .process_videos(&videos, Some(&fps_per_video));

    // Build the per-video soft-token-per-frame matrix expected by
    // `expand_gemma4_video_tokens`.
    let video_frame_tokens: Vec<Vec<usize>> = processed_videos
        .iter()
        .map(|v| vec![v.num_soft_tokens_per_frame; v.num_frames()])
        .collect();

    // The CLI path does not insert a dedicated `<|video|>` marker into
    // the prompt, the chat template handles `type == "image"` blocks
    // and we splice video frames in after BOS using the existing image-
    // token expansion. `i32::MIN` is a sentinel that cannot appear in a
    // tokenised prompt, so the placeholder-replace branch of
    // `expand_gemma4_video_tokens` is bypassed and the function takes
    // its "splice after BOS" fallback path. Server / chat-template
    // callers that *do* emit a real video token id can pass the proper
    // value through `mlxcel::vlm_runtime::expand_gemma4_video_tokens`
    // directly.
    let video_token_sentinel = i32::MIN;

    if image_paths.is_empty() {
        // Pure-video path: place `boi/image/eoi` runs after BOS.
        mlxcel::vlm_runtime::expand_gemma4_video_tokens(
            prompt_tokens,
            video_token_sentinel,
            gemma4_vl.image_token_id,
            gemma4_vl.boi_token_id,
            gemma4_vl.eoi_token_id,
            &video_frame_tokens,
        )?;
    } else {
        // Mixed path: expand image placeholders first, then videos.
        mlxcel::vlm_runtime::expand_gemma4_image_tokens_pub(
            prompt_tokens,
            gemma4_vl.image_token_id,
            gemma4_vl.boi_token_id,
            gemma4_vl.eoi_token_id,
            &image_soft_tokens,
        )?;
        mlxcel::vlm_runtime::expand_gemma4_video_tokens(
            prompt_tokens,
            video_token_sentinel,
            gemma4_vl.image_token_id,
            gemma4_vl.boi_token_id,
            gemma4_vl.eoi_token_id,
            &video_frame_tokens,
        )?;
    }

    let total_frames: usize = processed_videos.iter().map(|v| v.num_frames()).sum();
    let input_ids_arr =
        mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let embeddings = gemma4_vl.get_input_embeddings_with_videos(
        &input_ids_arr,
        &processed_images,
        &processed_videos,
    );

    print_preparation_summary(VlmPreparationSummary::Gemma4Video {
        video_count: processed_videos.len(),
        frame_slots: total_frames,
        total_tokens: prompt_tokens.len(),
    });

    Ok(Some(embeddings))
}

/// Compute video embeddings (and optional preceding image embeddings)
/// for the Gemma 4 Unified (encoder-free) model.
///
/// Decodes each video via `mlxcel::video::load_videos` (subprocess `ffmpeg`,
/// default 2.0 fps), patchifies each frame through the encoder-free vision
/// embedder with the per-frame `vision_soft_tokens_per_video_frame` budget,
/// expands the prompt into per-frame `<boi> video_token*N <eoi>` runs, and
/// scatters the per-frame soft tokens into `video_token_id` placeholders.
/// Mirrors [`compute_gemma4_video_embeddings`] but routes through the unified
/// model's `video_token_id` scatter instead of the ViT image path.
fn compute_gemma4_unified_video_embeddings(
    unified: &mlxcel::vision::Gemma4UnifiedModel,
    prompt_tokens: &mut Vec<i32>,
    image_paths: &[PathBuf],
    video_paths: &[PathBuf],
    target_fps: f64,
) -> Result<Option<InputEmbeddings>> {
    if !video::ffmpeg_available() {
        return Err(anyhow::anyhow!(
            "Video input requires `ffmpeg` on PATH. Install ffmpeg (e.g. `brew install ffmpeg` \
             on macOS or `apt install ffmpeg` on Linux) and retry."
        ));
    }

    // Decode the videos. `target_fps == 0` is rejected by `smart_nframes`,
    // so guard here with a clean error.
    let videos = video::load_videos(video_paths, Some(target_fps), None)
        .map_err(|err| anyhow::anyhow!("Failed to load video(s): {}", err))?;
    println!(
        "Loaded {} video(s) ({} total frames after sampling).",
        videos.len(),
        videos.iter().map(Vec::len).sum::<usize>()
    );

    // Optional companion images (e.g. user passes both --image and --video).
    let processed_images = if image_paths.is_empty() {
        Vec::new()
    } else {
        let images: Vec<image::DynamicImage> = image_paths
            .iter()
            .map(|path| {
                image::open(path)
                    .map_err(|e| anyhow::anyhow!("Failed to load image {:?}: {}", path, e))
            })
            .collect::<Result<Vec<_>>>()?;
        println!("Loaded {} image(s).", images.len());
        let processed = unified.processor.preprocess(&images);
        let num_soft_tokens: Vec<usize> = processed.iter().map(|i| i.num_soft_tokens).collect();
        mlxcel::vlm_runtime::expand_gemma4_image_tokens_pub(
            prompt_tokens,
            unified.image_token_id,
            unified.boi_token_id,
            unified.eoi_token_id,
            &num_soft_tokens,
        )?;
        processed
    };

    // Patchify every frame of every video through the encoder-free embedder.
    // Frames are kept flat (in video, then frame order) so the scatter sees
    // them in the same order as the expanded video_token_id placeholders.
    let mut video_frames: Vec<mlxcel::vision::processors::gemma4_unified::Gemma4UnifiedImageInput> =
        Vec::new();
    let mut video_frame_tokens: Vec<Vec<usize>> = Vec::with_capacity(videos.len());
    for frames in &videos {
        let processed = unified.processor.preprocess_video_frames(frames);
        video_frame_tokens.push(processed.iter().map(|f| f.num_soft_tokens).collect());
        video_frames.extend(processed);
    }

    mlxcel::vlm_runtime::expand_gemma4_unified_video_tokens(
        prompt_tokens,
        unified.video_token_id,
        unified.boi_token_id,
        unified.eoi_token_id,
        &video_frame_tokens,
    )?;

    let total_frames: usize = video_frame_tokens.iter().map(Vec::len).sum();
    let input_ids_arr =
        mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let embeddings =
        unified.get_input_embeddings_with_video(&input_ids_arr, &processed_images, &video_frames);

    print_preparation_summary(VlmPreparationSummary::Gemma4Video {
        video_count: videos.len(),
        frame_slots: total_frames,
        total_tokens: prompt_tokens.len(),
    });

    Ok(Some(embeddings))
}

/// Expand audio token placeholder: single audio_token -> BOA + AUDIO*N + EOA
fn expand_gemma4_audio_tokens(
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
    // If no audio placeholder found, insert BOA + AUDIO*N + EOA before last token
    if !found && !prompt_tokens.is_empty() {
        let last = expanded.pop().unwrap();
        expanded.push(boa_token_id);
        expanded.extend(std::iter::repeat_n(audio_token_id, num_audio_tokens));
        expanded.push(eoa_token_id);
        expanded.push(last);
    }
    *prompt_tokens = expanded;
}

/// Expand the Nemotron H Nano Omni sound placeholder block.
///
/// Mirrors upstream `processing_nemotron_h_nano_omni`'s text rewrite:
/// each `sound_context_token_id` occurrence in the prompt is wrapped
/// into `sound_start + sound_context * num_audio_tokens + sound_end`.
/// If `sound_start_token_id` or `sound_end_token_id` is `0`, the
/// framing token is omitted (matches the model surface contract that
/// `0` means "no framing token configured").
///
/// If no placeholder is found in the prompt, common when the user
/// runs `mlxcel generate --audio file.wav -p "..."` without manually
/// inserting a sound token, the block is prepended before the first
/// non-special token, mirroring the Nemotron image-token-expansion
/// path that does the equivalent prepend when the prompt has no
/// `<image>` placeholder.
///
/// Returns the number of audio tokens (post-subsampling) inserted, so
/// the caller can pass it into the runtime summary.
fn expand_nemotron_h_nano_omni_audio_tokens(
    prompt_tokens: &mut Vec<i32>,
    sound_context_token_id: i32,
    sound_start_token_id: i32,
    sound_end_token_id: i32,
    num_audio_tokens: usize,
) -> usize {
    let block_len = num_audio_tokens
        + if sound_start_token_id != 0 { 1 } else { 0 }
        + if sound_end_token_id != 0 { 1 } else { 0 };
    let mut expanded = Vec::with_capacity(prompt_tokens.len() + block_len);

    let mut placed = false;
    for &token in prompt_tokens.iter() {
        if token == sound_context_token_id && !placed {
            placed = true;
            if sound_start_token_id != 0 {
                expanded.push(sound_start_token_id);
            }
            for _ in 0..num_audio_tokens {
                expanded.push(sound_context_token_id);
            }
            if sound_end_token_id != 0 {
                expanded.push(sound_end_token_id);
            }
        } else {
            expanded.push(token);
        }
    }

    if !placed {
        // No placeholder. Prepend the audio block (matches the image-
        // token-expansion fallback in `vlm_runtime` for this model).
        let mut prepended = Vec::with_capacity(prompt_tokens.len() + block_len);
        if sound_start_token_id != 0 {
            prepended.push(sound_start_token_id);
        }
        for _ in 0..num_audio_tokens {
            prepended.push(sound_context_token_id);
        }
        if sound_end_token_id != 0 {
            prepended.push(sound_end_token_id);
        }
        prepended.extend(expanded);
        *prompt_tokens = prepended;
    } else {
        *prompt_tokens = expanded;
    }
    num_audio_tokens
}

/// Compute audio embeddings (with optional preceding images) for the
/// Nemotron H Nano Omni VLM.
///
/// Mirrors upstream `_extract_sound_features` + `_merge_features`:
/// 1. Loads the WAV via the shared `audio::load_wav_file` helper.
/// 2. Validates that the model exposes `sound_context_token_id` and
///    that the WAV's sample rate matches the configured
///    `sampling_rate` (mlxcel does not yet ship a resampler in core).
/// 3. Runs the Parakeet feature extractor to produce mel features +
///    attention mask + per-clip frame counts.
/// 4. Computes the post-subsampling audio token count via
///    `bundle.config.subsampling_output_length(num_frames)` and
///    expands the sound-context placeholder in `prompt_tokens` into
///    `sound_start + sound_context * num_audio_tokens + sound_end`.
/// 5. (Image+audio) preprocesses the images and expands the image
///    placeholder block. Image expansion uses the same per-image
///    `num_tokens` and start/end framing as the image-only runtime
///    path so the combined token stream is identical to what the
///    runtime would emit for image-only or audio-only inputs.
/// 6. Calls `model.extract_audio_features(...)` to obtain the
///    `[total_audio_tokens, hidden_size]` audio embedding flattened
///    across the batch.
/// 7. Calls `model.get_input_embeddings_full(input_ids, &images,
///    Some(&audio_features))` which scatters image features at
///    `img_context_token_id` slots and audio features at
///    `sound_context_token_id` slots.
/// 8. Emits a `VlmPreparationSummary::NemotronHNanoOmniAudio`
///    summary so the runtime CLI surfaces the audio path.
fn compute_nemotron_h_nano_omni_audio_embeddings(
    model: &mlxcel::vision::NemotronHNanoOmniVlModel,
    prompt_tokens: &mut Vec<i32>,
    image_paths: &[PathBuf],
    audio_path: &Path,
) -> Result<Option<InputEmbeddings>> {
    use mlxcel::audio;
    use mlxcel::audio::nemotron_h_nano_omni::NemotronOmniFeatureExtractor;

    let bundle = model.audio().ok_or_else(|| {
        anyhow::anyhow!(
            "This Nemotron H Nano Omni checkpoint was loaded without audio support. \
             The released model must ship a `sound_config` block in `config.json` for audio inputs."
        )
    })?;

    let sound_context_token_id = model.config.sound_context_token_id.ok_or_else(|| {
        anyhow::anyhow!(
            "Audio path requires `sound_context_token_id` in the model config but it is missing."
        )
    })?;

    // Load WAV. Same helper Gemma 4's audio path uses; no duplication.
    let (samples, sample_rate) =
        audio::load_wav_file(audio_path).map_err(|e| anyhow::anyhow!("{}", e))?;
    println!(
        "Loaded audio: {} samples at {} Hz ({:.1}s)",
        samples.len(),
        sample_rate,
        samples.len() as f64 / sample_rate as f64
    );

    if sample_rate != bundle.config.sampling_rate {
        return Err(anyhow::anyhow!(
            "Audio sample rate {} Hz does not match the model's expected {} Hz. \
             Resample the WAV (e.g. via `ffmpeg -i in.wav -ar {} out.wav`) before passing it.",
            sample_rate,
            bundle.config.sampling_rate,
            bundle.config.sampling_rate
        ));
    }

    // Run the feature extractor. The output is row-major f32 with
    // shape `[1, num_frames, num_mel_bins]` plus an int32 attention
    // mask of shape `[1, num_frames]` and a `[1]` lengths vector.
    let extractor = NemotronOmniFeatureExtractor::new(&bundle.config);
    let extracted = extractor.extract_batch(&[&samples[..]]);
    let num_frames = extracted.features_shape[1] as usize;

    // Post-subsampling token count = encoder output length, which
    // becomes the number of `sound_context_token_id` placeholders in
    // the expanded prompt. Single-clip CLI input, so feature_lengths
    // has length 1.
    let total_frames = extracted
        .feature_lengths
        .first()
        .copied()
        .unwrap_or(num_frames as i32) as usize;
    let num_audio_tokens = bundle.config.subsampling_output_length(total_frames).max(1);

    expand_nemotron_h_nano_omni_audio_tokens(
        prompt_tokens,
        sound_context_token_id,
        model.config.sound_start_token_id,
        model.config.sound_end_token_id,
        num_audio_tokens,
    );

    // Optional image branch: preprocess and expand image tokens with
    // the same per-image token count + framing that the image-only
    // runtime path uses, so the combined stream matches.
    let processed_images = if !image_paths.is_empty() {
        let images: Vec<image::DynamicImage> = image_paths
            .iter()
            .map(|path| {
                image::open(path)
                    .map_err(|e| anyhow::anyhow!("Failed to load image {:?}: {}", path, e))
            })
            .collect::<Result<Vec<_>>>()?;
        println!("Loaded {} image(s).", images.len());
        let processed = model.processor.preprocess_batch(&images);
        expand_nemotron_h_nano_omni_image_tokens(
            prompt_tokens,
            model.config.img_context_token_id,
            model.config.image_start_token_id,
            model.config.image_end_token_id,
            &processed,
        );
        processed
    } else {
        Vec::new()
    };

    // Build MLX tensors for the encoder. Features in row-major f32,
    // attention mask in int32 (the encoder broadcasts it via `less`).
    let audio_features_in = mlxcel_core::from_slice_f32(
        &extracted.features,
        &[
            extracted.features_shape[0],
            extracted.features_shape[1],
            extracted.features_shape[2],
        ],
    );
    let audio_attention_mask = mlxcel_core::from_slice_i32(
        &extracted.attention_mask,
        &[
            extracted.attention_mask_shape[0],
            extracted.attention_mask_shape[1],
        ],
    );
    let feature_lengths = mlxcel_core::from_slice_i32(
        &extracted.feature_lengths,
        &[extracted.feature_lengths.len() as i32],
    );

    // Run the encoder + projector and trim to per-clip valid lengths.
    let audio_features = model
        .extract_audio_features(
            &audio_features_in,
            Some(&audio_attention_mask),
            Some(&feature_lengths),
        )
        .map_err(|e| anyhow::anyhow!("Audio feature extraction failed: {}", e))?;

    // Compose final input embeddings: image placeholders get image
    // features, audio placeholders get audio features, in upstream
    // order (images first, then audio).
    let input_ids_arr =
        mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let embeddings =
        model.get_input_embeddings_full(&input_ids_arr, &processed_images, Some(&audio_features));

    print_preparation_summary(VlmPreparationSummary::NemotronHNanoOmniAudio {
        audio_clips: 1,
        audio_tokens: num_audio_tokens,
        total_tokens: prompt_tokens.len(),
    });

    Ok(Some(embeddings))
}

/// Expand each `img_context_token_id` placeholder in `prompt_tokens`
/// into `image_start + img_context * num_tokens + image_end`. Mirrors
/// the matching block in `vlm_runtime` for this model so the audio +
/// image CLI path produces the same token stream as the image-only
/// runtime path.
fn expand_nemotron_h_nano_omni_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    img_context_token_id: i32,
    image_start_token_id: i32,
    image_end_token_id: i32,
    images: &[mlxcel::vision::processors::nemotron_h_nano_omni::NemotronHNanoOmniImageInput],
) {
    let mut expanded = Vec::with_capacity(
        prompt_tokens.len() + images.iter().map(|img| img.num_tokens + 2).sum::<usize>(),
    );
    let placeholder_positions: Vec<usize> = prompt_tokens
        .iter()
        .enumerate()
        .filter_map(|(idx, &tok)| {
            if tok == img_context_token_id {
                Some(idx)
            } else {
                None
            }
        })
        .collect();

    if placeholder_positions.is_empty() {
        // Prepend one block per image. Matches the runtime fallback.
        for image in images.iter() {
            if image_start_token_id != 0 {
                expanded.push(image_start_token_id);
            }
            for _ in 0..image.num_tokens {
                expanded.push(img_context_token_id);
            }
            if image_end_token_id != 0 {
                expanded.push(image_end_token_id);
            }
        }
        expanded.extend_from_slice(prompt_tokens);
    } else {
        let mut image_idx = 0usize;
        for &token in prompt_tokens.iter() {
            if token == img_context_token_id && image_idx < images.len() {
                let image = &images[image_idx];
                if image_start_token_id != 0 {
                    expanded.push(image_start_token_id);
                }
                for _ in 0..image.num_tokens {
                    expanded.push(img_context_token_id);
                }
                if image_end_token_id != 0 {
                    expanded.push(image_end_token_id);
                }
                image_idx += 1;
            } else {
                expanded.push(token);
            }
        }
    }
    *prompt_tokens = expanded;
}

#[cfg(test)]
mod tests {
    use super::{expand_gemma4_audio_tokens, expand_nemotron_h_nano_omni_audio_tokens};

    #[test]
    fn gemma4_audio_expands_placeholder_in_place() {
        // Prompt with the `<|audio|>` marker (id 9) already rendered into the
        // user turn (Fix A): expand it in place into BOA + AUDIO*N + EOA so the
        // audio block stays inside the user turn (issue #436).
        // [BOS, <sot>, audio_token, EOS]
        let mut tokens = vec![1i32, 5, 9, 2];
        expand_gemma4_audio_tokens(&mut tokens, 9, 7, 8, 3);
        // [BOS, <sot>, BOA, AUDIO, AUDIO, AUDIO, EOA, EOS]
        assert_eq!(tokens, vec![1i32, 5, 7, 9, 9, 9, 8, 2]);
    }

    #[test]
    fn gemma4_audio_fallback_inserts_before_last_token() {
        // No `<|audio|>` placeholder present: the last-resort fallback inserts
        // the audio block before the final token. (Fix A renders the marker so
        // this path is not taken for Gemma 4, but it must stay correct.)
        let mut tokens = vec![1i32, 42, 2];
        expand_gemma4_audio_tokens(&mut tokens, 9, 7, 8, 2);
        assert_eq!(tokens, vec![1i32, 42, 7, 9, 9, 8, 2]);
    }

    #[test]
    fn audio_token_expansion_replaces_first_placeholder() {
        // Prompt: [BOS, sound_ctx, "hello", EOS]
        let mut tokens = vec![1i32, 99, 42, 2];
        let inserted = expand_nemotron_h_nano_omni_audio_tokens(&mut tokens, 99, 7, 8, 3);
        assert_eq!(inserted, 3);
        // Expected: [BOS, sound_start, sound_ctx, sound_ctx, sound_ctx, sound_end, "hello", EOS]
        assert_eq!(tokens, vec![1i32, 7, 99, 99, 99, 8, 42, 2]);
    }

    #[test]
    fn audio_token_expansion_omits_zero_framing_tokens() {
        // sound_start=0, sound_end=0 → no framing tokens emitted.
        let mut tokens = vec![1i32, 99, 2];
        let inserted = expand_nemotron_h_nano_omni_audio_tokens(&mut tokens, 99, 0, 0, 2);
        assert_eq!(inserted, 2);
        assert_eq!(tokens, vec![1i32, 99, 99, 2]);
    }

    #[test]
    fn audio_token_expansion_prepends_when_no_placeholder_present() {
        // Prompt has no sound_ctx token, block prepended before the
        // existing tokens.
        let mut tokens = vec![1i32, 42, 2];
        let inserted = expand_nemotron_h_nano_omni_audio_tokens(&mut tokens, 99, 7, 8, 2);
        assert_eq!(inserted, 2);
        assert_eq!(tokens, vec![7i32, 99, 99, 8, 1, 42, 2]);
    }

    #[test]
    fn audio_token_expansion_replaces_only_first_occurrence() {
        // Two sound_ctx tokens: only the first is expanded so a multi-
        // clip prompt would land each clip in its own marker (the
        // single-clip CLI surface uses N=1 here).
        let mut tokens = vec![99i32, 42, 99, 2];
        expand_nemotron_h_nano_omni_audio_tokens(&mut tokens, 99, 7, 8, 1);
        assert_eq!(tokens, vec![7i32, 99, 8, 42, 99, 2]);
    }
}
