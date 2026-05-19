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

//! Same-process decode benchmark harness.
//!
//! `scripts/bench_decode.sh` compares mlxcel against Python `mlx-lm` /
//! `mlx-vlm` baselines collected by `scripts/bench_mlxlm.py`.  The Python
//! harness loads the model once, performs warmup, and then measures
//! `stream_generate()` in the same process.  Shelling out to
//! `mlxcel generate` twice makes the measured prefill path cold again, which
//! disproportionately penalizes one-shot prefill timings while leaving the
//! multi-token decode loop mostly amortized.
//!
//! This binary keeps the CLI-facing prompt/VLM preparation semantics but runs
//! warmup and measured generation against one loaded model in one process.

use std::io::{self, Write as _};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use mlxcel::cli::turbo_args::{TurboKvCacheArgs, resolve_kv_cache_mode};
use mlxcel::sampling::{ResolvedSamplingParams, build_sampling_config};
use mlxcel::server::chat_template::{ChatMessage, ChatTemplateProcessor};
use mlxcel::tokenizer::{MlxcelTokenizer, load_tokenizer};
use mlxcel::vision::merge::InputEmbeddings;
use mlxcel::{CxxGenerator, LanguageModel, LoadedModel, SamplingConfig};
use mlxcel_core::cache::KVCacheMode;

/// Same-process benchmark for `scripts/bench_decode.sh`.
#[derive(Parser, Debug)]
#[command(name = "mlxcel-bench-decode")]
struct Args {
    /// Path to the model directory.
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Prompt text.
    #[arg(short = 'p', long)]
    prompt: String,

    /// Maximum generated tokens in the measured pass.
    #[arg(short = 'n', long, default_value_t = 100)]
    max_tokens: usize,

    /// Generated tokens in the warmup pass.
    #[arg(long, default_value_t = 20)]
    warmup_tokens: usize,

    /// Image path(s) for VLM benchmark mode.
    #[arg(long, value_name = "PATH", num_args = 1..)]
    image: Vec<PathBuf>,

    /// Disable automatic chat-template application.
    #[arg(long, default_value_t = false)]
    no_chat_template: bool,

    /// Shared KV-cache mode flags, matching `mlxcel generate`.
    #[command(flatten)]
    turbo: TurboKvCacheArgs,
}

struct PreparedPrompt {
    tokens: Vec<i32>,
    embeddings: Option<InputEmbeddings>,
}

fn apply_user_chat_template(processor: &ChatTemplateProcessor, user_prompt: &str) -> String {
    let messages = [ChatMessage {
        role: "user".to_string(),
        content: user_prompt.to_string(),
    }];
    processor
        .apply(&messages, None)
        .unwrap_or_else(|_| user_prompt.to_string())
}

fn apply_vlm_chat_template(
    processor: &ChatTemplateProcessor,
    user_prompt: &str,
    num_images: usize,
) -> String {
    if !processor.supports_image_content() {
        return apply_user_chat_template(processor, user_prompt);
    }

    let mut content_parts: Vec<serde_json::Value> = Vec::new();
    for _ in 0..num_images {
        content_parts.push(serde_json::json!({"type": "image"}));
    }
    content_parts.push(serde_json::json!({"type": "text", "text": user_prompt}));

    let messages = serde_json::json!([{
        "role": "user",
        "content": content_parts,
    }]);

    processor
        .apply_raw(&messages, None)
        .unwrap_or_else(|_| apply_user_chat_template(processor, user_prompt))
}

fn load_cli_prompt(
    model_path: &Path,
    user_prompt: &str,
    no_chat_template: bool,
    num_images: usize,
) -> String {
    if no_chat_template {
        return user_prompt.to_string();
    }

    let processor = ChatTemplateProcessor::from_model_path(model_path)
        .ok()
        .flatten();
    processor.map_or_else(
        || user_prompt.to_string(),
        |processor| {
            if num_images > 0 {
                apply_vlm_chat_template(&processor, user_prompt, num_images)
            } else {
                apply_user_chat_template(&processor, user_prompt)
            }
        },
    )
}

fn tokenize_prompt(tokenizer: &MlxcelTokenizer, prompt: &str) -> Result<Vec<i32>> {
    // Matches the CLI generate path: chat templates that render a BOS token
    // should not receive a duplicate special token from the tokenizer.
    let add_special = !prompt.starts_with("<bos>") && !prompt.starts_with("<s>");
    let ids = tokenizer
        .encode(prompt, add_special)
        .map_err(|err| anyhow::anyhow!("tokenization failed: {err}"))?;
    Ok(ids.into_iter().map(|id| id as i32).collect())
}

fn prepare_prompt(
    model: &LoadedModel,
    model_path: &Path,
    tokenizer: &MlxcelTokenizer,
    user_prompt: &str,
    no_chat_template: bool,
    image_paths: &[PathBuf],
) -> Result<PreparedPrompt> {
    let prompt = load_cli_prompt(model_path, user_prompt, no_chat_template, image_paths.len());
    let mut tokens = tokenize_prompt(tokenizer, &prompt)?;

    if image_paths.is_empty() {
        return Ok(PreparedPrompt {
            tokens,
            embeddings: None,
        });
    }

    let images = image_paths
        .iter()
        .map(|path| {
            image::open(path).with_context(|| format!("failed to load image {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;

    let prepared = mlxcel::vlm_runtime::prepare_and_compute_vlm_embeddings(
        model,
        &mut tokens,
        &prompt,
        &images,
        |text, add_special| {
            tokenizer
                .encode(text, add_special)
                .unwrap_or_default()
                .into_iter()
                .map(|id| id as i32)
                .collect()
        },
    )?;

    Ok(PreparedPrompt {
        tokens,
        embeddings: prepared.map(|prepared| prepared.embeddings),
    })
}

fn sampling_config(model_path: &Path) -> SamplingConfig {
    build_sampling_config(ResolvedSamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: None,
        repetition_penalty: 1.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_penalty_last_n: 0,
        dry_sequence_breakers: Vec::new(),
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        stop_token_ids: mlxcel::read_eos_token_ids(model_path),
    })
}

fn warmup(
    model: &LoadedModel,
    prepared: &PreparedPrompt,
    max_tokens: usize,
    sampling: &SamplingConfig,
    kv_cache_mode: KVCacheMode,
) -> Result<()> {
    if max_tokens == 0 {
        return Ok(());
    }

    let mut generator = CxxGenerator::new_with_kv_mode(model.num_layers(), kv_cache_mode);
    if let Some(embeddings) = prepared.embeddings.as_ref() {
        let (input_embeds, mask) = mlxcel::vlm_runtime::prepared_embedding_refs(embeddings)?;
        let _ = generator.generate_streaming_with_embeddings(
            model,
            &prepared.tokens,
            Some(input_embeds),
            mask,
            max_tokens,
            sampling,
            |_| true,
        );
    } else {
        let _ = generator.generate(model, &prepared.tokens, max_tokens, sampling);
    }

    // Reset any model-owned cache state (hybrid/recurrent models) before the
    // measured pass.  Keeping the process alive preserves MLX/Metal warm state
    // while avoiding semantic leakage from the warmup generation.
    generator.reset_with_model(model);
    mlxcel_core::synchronize_default();
    Ok(())
}

fn measured(
    model: &LoadedModel,
    prepared: &PreparedPrompt,
    max_tokens: usize,
    sampling: &SamplingConfig,
    kv_cache_mode: KVCacheMode,
) -> Result<mlxcel::GenerationStats> {
    let mut generator = CxxGenerator::new_with_kv_mode(model.num_layers(), kv_cache_mode);
    let (_tokens, stats) = if let Some(embeddings) = prepared.embeddings.as_ref() {
        let (input_embeds, mask) = mlxcel::vlm_runtime::prepared_embedding_refs(embeddings)?;
        generator.generate_with_stats_and_embeddings(
            model,
            &prepared.tokens,
            Some(input_embeds),
            mask,
            max_tokens,
            sampling,
        )
    } else {
        generator.generate_with_stats(model, &prepared.tokens, max_tokens, sampling)
    };
    mlxcel_core::synchronize_default();
    Ok(stats)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let kv_cache_mode = resolve_kv_cache_mode(
        args.turbo.cache_type_k.as_deref(),
        args.turbo.cache_type_v.as_deref(),
        args.turbo.kv_cache_mode.as_deref(),
    )
    .map_err(|err| anyhow::anyhow!("{err}"))?;

    // Keep `--turbo-boundary-v` semantics identical to `mlxcel generate`.
    // This must happen before any generator/cache construction.
    args.turbo.apply_to_environment();

    let _runtime = mlxcel::initialize_runtime();
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    let (model, loaded_tokenizer) =
        mlxcel::load_model(&args.model).context("failed to load model")?;
    let tokenizer = load_tokenizer(&args.model).unwrap_or(loaded_tokenizer);
    let sampling = sampling_config(&args.model);

    let prepared = prepare_prompt(
        &model,
        &args.model,
        &tokenizer,
        &args.prompt,
        args.no_chat_template,
        &args.image,
    )?;
    warmup(
        &model,
        &prepared,
        args.warmup_tokens,
        &sampling,
        kv_cache_mode,
    )?;
    drop(prepared);

    // Some VLM models store single-use state on the model during prepare_prompt
    // (e.g., Gemma 3n caches per_layer_inputs that the first prefill takes()
    // and Gemma 3 carries an attention-mask shape that only matches the first
    // call). The warmup pass consumes that state, so regenerate the prepared
    // prompt before the measured pass. For text-only models this is cheap; for
    // VLM it re-runs the vision encoder against now-warm MLX/Metal state.
    let prepared = prepare_prompt(
        &model,
        &args.model,
        &tokenizer,
        &args.prompt,
        args.no_chat_template,
        &args.image,
    )?;
    let stats = measured(&model, &prepared, args.max_tokens, &sampling, kv_cache_mode)?;

    println!("[Profile Results]");
    stats.print();
    io::stdout().flush()?;

    mlxcel_core::clear_memory_cache();
    Ok(())
}
