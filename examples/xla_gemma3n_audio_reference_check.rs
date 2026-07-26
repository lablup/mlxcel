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

//! Audio-only MLX↔IREE intermediate gate for the pinned Gemma3n #875 fixture.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use mlxcel::audio::gemma3n::{Gemma3nAudioFeatureExtractor, run_gemma3n_audio_mlx_diagnostics};
use mlxcel::audio::{AudioFamilyPolicy, preprocess_wav_file};
use mlxcel::tokenizer::load_tokenizer;
use mlxcel_xla::{
    GEMMA3N_AUDIO_MEL_BINS, GEMMA3N_AUDIO_SOFT_TOKENS, Gemma3nAudioInput, Gemma3nAudioIreeRuntime,
    select_gemma3n_audio_frame_bucket,
};

const PROMPT: &str = "Transcribe the following speech segment in English:";
const AUDIO_TOKEN_ID: i32 = 262_273;
const BOA_TOKEN_ID: i32 = 256_000;
const EOA_TOKEN_ID: i32 = 262_272;
const CONTEXT_CAPACITY: usize = 256;
const MAX_BF16_OUTLIER_PROBES: usize = 32;

fn path_env(name: &str, default: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

fn expanded_prompt(model_dir: &Path) -> Result<Vec<i32>, String> {
    let tokenizer = load_tokenizer(model_dir).map_err(|error| error.to_string())?;
    let rendered = format!(
        "<bos><start_of_turn>user\n{PROMPT}<audio_soft_token><end_of_turn>\n\
         <start_of_turn>model\n"
    );
    let mut tokens = tokenizer
        .encode(&rendered, false)
        .map_err(|error| error.to_string())?
        .into_iter()
        .map(|token| token as i32)
        .collect::<Vec<_>>();
    let wrapper = tokenizer
        .encode("\n\n", false)
        .map_err(|error| error.to_string())?
        .into_iter()
        .map(|token| token as i32)
        .collect::<Vec<_>>();
    mlxcel::vlm_runtime::expand_gemma3n_audio_tokens(
        &mut tokens,
        AUDIO_TOKEN_ID,
        BOA_TOKEN_ID,
        EOA_TOKEN_ID,
        1,
        GEMMA3N_AUDIO_SOFT_TOKENS,
        &wrapper,
        None,
    )
    .map_err(|error| error.to_string())?;
    Ok(tokens)
}

fn audio_input(audio_path: &Path) -> Result<Gemma3nAudioInput, String> {
    let waveforms = preprocess_wav_file(
        audio_path,
        AudioFamilyPolicy::gemma3n(),
        &AtomicBool::new(false),
    )
    .map_err(|error| error.to_string())?;
    let clips = waveforms
        .clips
        .into_iter()
        .map(|clip| clip.samples)
        .collect::<Vec<_>>();
    let features = Gemma3nAudioFeatureExtractor::new().extract_batch(&clips)?;
    let bucket =
        select_gemma3n_audio_frame_bucket(features.frames).map_err(|error| error.to_string())?;
    let mut mel = vec![0.0; features.batch_size * bucket * GEMMA3N_AUDIO_MEL_BINS];
    let mut valid_mask = vec![0; features.batch_size * bucket];
    let mut frame_lengths = Vec::with_capacity(features.batch_size);
    for clip in 0..features.batch_size {
        let mut valid_frames = 0;
        for frame in 0..features.frames {
            let source_row = clip * features.frames + frame;
            let target_row = clip * bucket + frame;
            valid_mask[target_row] = u8::from(features.valid_mask[source_row]);
            valid_frames += usize::from(features.valid_mask[source_row]);
            let source = source_row * GEMMA3N_AUDIO_MEL_BINS;
            let target = target_row * GEMMA3N_AUDIO_MEL_BINS;
            mel[target..target + GEMMA3N_AUDIO_MEL_BINS]
                .copy_from_slice(&features.features[source..source + GEMMA3N_AUDIO_MEL_BINS]);
        }
        frame_lengths.push(valid_frames);
    }
    Gemma3nAudioInput::new(mel, valid_mask, frame_lengths, bucket)
        .map_err(|error| error.to_string())
}

fn audio_rows(tokens: &[i32]) -> Result<Vec<i32>, String> {
    let mut rank = 0i32;
    let rows = tokens
        .iter()
        .map(|token| {
            if *token == AUDIO_TOKEN_ID {
                let row = rank;
                rank += 1;
                row
            } else {
                -1
            }
        })
        .collect::<Vec<_>>();
    if rank as usize != GEMMA3N_AUDIO_SOFT_TOKENS {
        return Err(format!(
            "expanded prompt has {rank} audio rows, expected {GEMMA3N_AUDIO_SOFT_TOKENS}"
        ));
    }
    Ok(rows)
}

struct Diff {
    max_abs: f64,
    rms: f64,
    max_index: usize,
    max_bf16_ulp: u32,
    non_bf16_values: usize,
    over_one_bf16_ulp: usize,
    bf16_outliers: Vec<Bf16Outlier>,
}

struct Bf16Outlier {
    index: usize,
    reference: f32,
    actual: f32,
    distance: u32,
}

fn bf16_bits(value: f32) -> Option<u16> {
    let bits = value.to_bits();
    (bits & 0xffff == 0).then_some((bits >> 16) as u16)
}

fn ordered_bf16(bits: u16) -> i32 {
    if bits & 0x8000 == 0 {
        0x8000 + i32::from(bits)
    } else {
        0x8000 - i32::from(bits & 0x7fff)
    }
}

fn bf16_ulp_distance(left: f32, right: f32) -> Option<u32> {
    let left = ordered_bf16(bf16_bits(left)?);
    let right = ordered_bf16(bf16_bits(right)?);
    Some(left.abs_diff(right))
}

fn display_bf16_bits(value: f32) -> String {
    bf16_bits(value)
        .map(|bits| format!("0x{bits:04x}"))
        .unwrap_or_else(|| "not-bf16".to_string())
}

fn compare(name: &str, reference: &[f32], actual: &[f32]) -> Result<Diff, String> {
    if reference.len() != actual.len() {
        return Err(format!(
            "{name} length mismatch: MLX={} IREE={}",
            reference.len(),
            actual.len()
        ));
    }
    let mut squared = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut max_index = 0usize;
    let mut max_bf16_ulp = 0u32;
    let mut non_bf16_values = 0usize;
    let mut over_one_bf16_ulp = 0usize;
    let mut bf16_outliers = Vec::new();
    for (index, (reference, actual)) in reference.iter().zip(actual).enumerate() {
        let difference = f64::from(*actual) - f64::from(*reference);
        squared += difference * difference;
        if difference.abs() > max_abs {
            max_abs = difference.abs();
            max_index = index;
        }
        match bf16_ulp_distance(*reference, *actual) {
            Some(distance) => {
                max_bf16_ulp = max_bf16_ulp.max(distance);
                if distance > 1 {
                    over_one_bf16_ulp += 1;
                    if bf16_outliers.len() < MAX_BF16_OUTLIER_PROBES {
                        bf16_outliers.push(Bf16Outlier {
                            index,
                            reference: *reference,
                            actual: *actual,
                            distance,
                        });
                    }
                }
            }
            None => non_bf16_values += 1,
        }
    }
    let rms = (squared / reference.len().max(1) as f64).sqrt();
    let probes = [
        0usize,
        1.min(reference.len().saturating_sub(1)),
        reference.len() / 3,
        reference.len() / 2,
        reference.len().saturating_sub(1),
        max_index,
    ];
    let max_reference_bits = display_bf16_bits(reference[max_index]);
    let max_actual_bits = display_bf16_bits(actual[max_index]);
    let max_pair_ulp = bf16_ulp_distance(reference[max_index], actual[max_index]);
    eprintln!(
        "gemma3n-audio-reference: stage={name} max_abs={max_abs:.9e} rms={rms:.9e} \
         max_index={max_index} max_pair_mlx={:.9e} max_pair_iree={:.9e} \
         max_pair_mlx_bf16={max_reference_bits} max_pair_iree_bf16={max_actual_bits} \
         max_pair_bf16_ulp={max_pair_ulp:?} max_bf16_ulp={max_bf16_ulp} \
         non_bf16_values={non_bf16_values} over_one_bf16_ulp={over_one_bf16_ulp}",
        reference[max_index], actual[max_index]
    );
    for index in probes {
        eprintln!(
            "gemma3n-audio-reference: probe stage={name} index={index} mlx={:.9e} iree={:.9e} abs={:.9e}",
            reference[index],
            actual[index],
            (actual[index] - reference[index]).abs()
        );
    }
    for outlier in &bf16_outliers {
        eprintln!(
            "gemma3n-audio-reference: bf16-outlier stage={name} index={} mlx={:.9e} \
             iree={:.9e} abs={:.9e} mlx_bf16={} iree_bf16={} bf16_ulp={}",
            outlier.index,
            outlier.reference,
            outlier.actual,
            (outlier.actual - outlier.reference).abs(),
            display_bf16_bits(outlier.reference),
            display_bf16_bits(outlier.actual),
            outlier.distance,
        );
    }
    if bf16_outliers.len() < over_one_bf16_ulp {
        eprintln!(
            "gemma3n-audio-reference: bf16-outlier stage={name} captured={} total={} truncated=true",
            bf16_outliers.len(),
            over_one_bf16_ulp,
        );
    }
    Ok(Diff {
        max_abs,
        rms,
        max_index,
        max_bf16_ulp,
        non_bf16_values,
        over_one_bf16_ulp,
        bf16_outliers,
    })
}

fn main() -> Result<(), String> {
    let model_dir = path_env(
        "MLXCEL_GEMMA3N_AUDIO_MODEL",
        "/home/inureyes/models/gemma3n-e4b-4bit",
    );
    let audio_path = path_env("MLXCEL_GEMMA3N_AUDIO_WAV", "/tmp/gemma3n-103-1240-0000.wav");
    let device = std::env::var("MLXCEL_XLA_DEVICE")
        .map_err(|_| "MLXCEL_XLA_DEVICE must be set explicitly for this gate".to_string())?;
    let tokens = expanded_prompt(&model_dir)?;
    let rows = audio_rows(&tokens)?;
    let input = audio_input(&audio_path)?;
    eprintln!(
        "gemma3n-audio-reference: frontend bucket={} clips={} valid={:?} tokens={}",
        input.frame_bucket(),
        input.clips(),
        input.frame_lengths(),
        tokens.len()
    );

    let mlx = run_gemma3n_audio_mlx_diagnostics(
        &model_dir,
        input.mel(),
        input.valid_mask(),
        input.frame_bucket(),
        input.clips(),
        &tokens,
        AUDIO_TOKEN_ID,
        CONTEXT_CAPACITY,
    )?;
    mlxcel_core::memory::clear_cache();
    eprintln!(
        "gemma3n-audio-reference: MLX audio-only oracle complete projected_lengths={:?}",
        mlx.projected_lengths
    );

    let mut iree = Gemma3nAudioIreeRuntime::load_audio_only_diagnostic(
        &model_dir,
        &device,
        CONTEXT_CAPACITY,
        input.frame_bucket(),
        input.clips(),
    )?;
    let iree = iree.invoke(&input, &tokens, &rows)?;
    eprintln!(
        "gemma3n-audio-reference: IREE audio-only path complete projected_lengths={:?}",
        iree.projected_lengths()
    );
    if mlx.projected_lengths != iree.projected_lengths() {
        return Err(format!(
            "projected length mismatch: MLX={:?} IREE={:?}",
            mlx.projected_lengths,
            iree.projected_lengths()
        ));
    }

    let stage_values = [
        (
            "sscp_conv_0_convolution",
            mlx.sscp_conv_0_convolution.as_slice(),
        ),
        ("sscp_conv_0_norm", mlx.sscp_conv_0_norm.as_slice()),
        ("sscp_conv_0", mlx.sscp_conv_0.as_slice()),
        (
            "sscp_conv_1_convolution",
            mlx.sscp_conv_1_convolution.as_slice(),
        ),
        ("sscp_conv_1_norm", mlx.sscp_conv_1_norm.as_slice()),
        ("sscp_conv_1", mlx.sscp_conv_1.as_slice()),
        ("input_projection", mlx.input_projection.as_slice()),
        (
            "conformer.0.feed_forward_start",
            mlx.conformer_0_feed_forward_start.as_slice(),
        ),
        ("encoded_reduced", mlx.encoded_reduced.as_slice()),
        ("soft_norm", mlx.soft_norm.as_slice()),
        ("soft_linear", mlx.soft_linear.as_slice()),
        ("soft_post_norm", mlx.soft_post_norm.as_slice()),
        ("hard_embedding", mlx.hard_embedding.as_slice()),
        ("hard_norm", mlx.hard_norm.as_slice()),
        ("hard_linear", mlx.hard_linear.as_slice()),
        ("hard_post_norm", mlx.hard_post_norm.as_slice()),
    ];
    let mut names = Vec::with_capacity(stage_values.len() + 4);
    let mut stages = Vec::with_capacity(stage_values.len() + 4);
    for (name, reference) in stage_values {
        let actual = iree
            .diagnostic_stage(name)
            .ok_or_else(|| format!("IREE diagnostic output is missing stage {name}"))?;
        names.push(name);
        stages.push(compare(name, reference, actual)?);
    }
    names.extend([
        "projected_audio",
        "hard_audio",
        "merged_embeddings",
        "dense_ple",
    ]);
    stages.extend([
        compare(
            "projected_audio",
            &mlx.projected_audio,
            iree.projected_audio(),
        )?,
        compare("hard_audio", &mlx.hard_audio, iree.hard_audio())?,
        compare("merged_embeddings", &mlx.embeddings, iree.embeddings())?,
        compare("dense_ple", &mlx.dense_ple, iree.dense_ple())?,
    ]);
    let first = stages
        .iter()
        .enumerate()
        .find(|(_, diff)| diff.max_abs > 5e-3 || diff.rms > 5e-4);
    match first {
        Some((index, diff)) => Err(format!(
            "first divergent stage={} max_abs={:.9e} rms={:.9e} max_index={} \
             max_bf16_ulp={} non_bf16_values={} over_one_bf16_ulp={} \
             captured_bf16_outliers={}",
            names[index],
            diff.max_abs,
            diff.rms,
            diff.max_index,
            diff.max_bf16_ulp,
            diff.non_bf16_values,
            diff.over_one_bf16_ulp,
            diff.bf16_outliers.len(),
        )),
        None => {
            println!("Gemma3n audio-only MLX↔IREE intermediate gate PASS");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_ulp_distance_handles_sign_zero_and_one_unit_steps() {
        assert_eq!(bf16_ulp_distance(0.0, -0.0), Some(0));
        assert_eq!(bf16_ulp_distance(128.0, 129.0), Some(1));
        assert_eq!(bf16_ulp_distance(-128.0, -129.0), Some(1));
        assert_eq!(bf16_ulp_distance(128.0, 130.0), Some(2));
        assert_eq!(bf16_ulp_distance(1.0, 1.003), None);
    }

    #[test]
    fn compare_keeps_absolute_failure_data_and_bf16_drift_data_separate() {
        let diff = compare("fixture", &[128.0, -128.0], &[129.0, -130.0]).unwrap();
        assert_eq!(diff.max_abs, 2.0);
        assert_eq!(diff.max_index, 1);
        assert_eq!(diff.max_bf16_ulp, 2);
        assert_eq!(diff.non_bf16_values, 0);
        assert_eq!(diff.over_one_bf16_ulp, 1);
        assert_eq!(diff.bf16_outliers.len(), 1);
        assert_eq!(diff.bf16_outliers[0].index, 1);
        assert_eq!(diff.bf16_outliers[0].distance, 2);
    }

    #[test]
    fn compare_bounds_detailed_bf16_outlier_reporting() {
        let reference = vec![128.0; MAX_BF16_OUTLIER_PROBES + 8];
        let actual = vec![130.0; reference.len()];
        let diff = compare("bounded-outliers", &reference, &actual).unwrap();
        assert_eq!(diff.over_one_bf16_ulp, reference.len());
        assert_eq!(diff.bf16_outliers.len(), MAX_BF16_OUTLIER_PROBES);
        assert_eq!(
            diff.bf16_outliers.last().map(|outlier| outlier.index),
            Some(MAX_BF16_OUTLIER_PROBES - 1),
        );
    }
}
