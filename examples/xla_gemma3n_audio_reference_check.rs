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
    for (index, (reference, actual)) in reference.iter().zip(actual).enumerate() {
        let difference = f64::from(*actual) - f64::from(*reference);
        squared += difference * difference;
        if difference.abs() > max_abs {
            max_abs = difference.abs();
            max_index = index;
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
    eprintln!(
        "gemma3n-audio-reference: stage={name} max_abs={max_abs:.9e} rms={rms:.9e} max_index={max_index}"
    );
    for index in probes {
        eprintln!(
            "gemma3n-audio-reference: probe stage={name} index={index} mlx={:.9e} iree={:.9e} abs={:.9e}",
            reference[index],
            actual[index],
            (actual[index] - reference[index]).abs()
        );
    }
    Ok(Diff {
        max_abs,
        rms,
        max_index,
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
            "first divergent stage={} max_abs={:.9e} rms={:.9e} max_index={}",
            names[index], diff.max_abs, diff.rms, diff.max_index
        )),
        None => {
            println!("Gemma3n audio-only MLX↔IREE intermediate gate PASS");
            Ok(())
        }
    }
}
