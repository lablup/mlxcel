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

//! CLI driver for LLaDA-2 MoE block-unmasking generation (issue #546).
//!
//! Thin bridge between the parsed [`crate::GenerateArgs`] flag surface and
//! [`Llada2MoeModel::generate_llada2_streaming`]: resolves the generation
//! options, seeds the RNG, streams decoded text through the shared incremental
//! detokenizer, and prints the generation stats. LLaDA-2 is text-only, so
//! image / audio / video input and speculative decoding are rejected.

use anyhow::{Result, anyhow};
use std::io::{self, Write as IoWrite};

use mlxcel::models::Llada2MoeModel;
use mlxcel::models::llada2_moe::{Llada2GenerateOptions, Llada2GenerationStats};
use mlxcel::server::model_provider::model_worker::StreamingDecodeState;
use mlxcel::tokenizer::MlxcelTokenizer;

use super::generate::print_generation_preamble;
use crate::GenerateArgs;

fn llada2_options_from_args(args: &GenerateArgs) -> Llada2GenerateOptions {
    Llada2GenerateOptions {
        max_new_tokens: args.generation.max_tokens,
        temperature: args.sampling.temp,
        top_k: args.sampling.top_k,
        top_p: args.sampling.top_p,
        // `--max-denoising-steps` doubles as the LLaDA-2 per-block step count
        // override; the sampler/threshold diffusion flags are gemma-only.
        steps: args
            .generation
            .diffusion
            .max_denoising_steps
            .unwrap_or_else(|| Llada2GenerateOptions::default().steps),
        // Stop ids from generation_config.json (when present) join the
        // checkpoint's embedded EOS set inside the engine.
        extra_eos_token_ids: mlxcel::read_eos_token_ids(&args.model.model),
        ..Llada2GenerateOptions::default()
    }
}

fn print_llada2_stats(stats: &Llada2GenerationStats, profile: bool) {
    println!();
    println!();
    println!(
        "[Generated {} tokens in {:.2}s = {:.2} tok/s]",
        stats.generated_tokens, stats.generation_time_s, stats.generation_tps
    );
    if profile {
        println!("[LLaDA-2 Profile]");
        println!(
            "  Prompt: {} tokens in {:.3}s = {:.2} tok/s",
            stats.prompt_tokens, stats.prompt_time_s, stats.prompt_tps
        );
        println!(
            "  Blocks: {}, denoising steps: {}",
            stats.blocks, stats.denoising_steps
        );
        println!("  Finish reason: {:?}", stats.finish_reason);
    }
}

/// Run one LLaDA-2 MoE block-unmasking generation for the CLI `generate` flow.
pub(crate) fn run_llada2_generation(
    model: &Llada2MoeModel,
    args: &GenerateArgs,
    tokenizer: &MlxcelTokenizer,
    prompt_tokens: &[i32],
    user_prompt: &str,
) -> Result<()> {
    if args.generation.audio.is_some()
        || !args.generation.video.is_empty()
        || !args.generation.image.is_empty()
    {
        return Err(anyhow!(
            "LLaDA-2 MoE is text-only; run with a text prompt (no --image/--audio/--video)"
        ));
    }
    if args.model.draft_model.is_some() {
        return Err(anyhow!(
            "LLaDA-2 MoE does not support speculative decoding (--draft-model); block \
             unmasking already denoises a whole block per step"
        ));
    }

    let options = llada2_options_from_args(args);
    if let Some(seed) = args.sampling.seed {
        mlxcel_core::random_seed(seed);
    }

    print_generation_preamble(user_prompt)?;

    // Stream through the shared incremental detokenizer; raw ids are collected
    // in parallel so any held-back tail bytes can be flushed afterward.
    let mut decode_state = StreamingDecodeState::new(tokenizer, prompt_tokens);
    let mut generated_ids: Vec<u32> = Vec::with_capacity(options.max_new_tokens);
    let mut streamed = String::new();
    let mut stdout = io::stdout();

    let stats = model
        .generate_llada2_streaming(prompt_tokens, &options, |token_id| {
            generated_ids.push(token_id as u32);
            if let Some(text) = decode_state.on_token(token_id, tokenizer) {
                print!("{text}");
                let _ = stdout.flush();
                streamed.push_str(&text);
            }
            true
        })
        .map_err(|e| anyhow!("{e}"))?;

    let full_text = tokenizer.decode(&generated_ids, true).unwrap_or_default();
    if let Some(tail) = full_text.strip_prefix(&streamed)
        && !tail.is_empty()
    {
        print!("{tail}");
        let _ = stdout.flush();
    }

    print_llada2_stats(&stats, args.generation.profile);

    mlxcel_core::clear_memory_cache();
    Ok(())
}
