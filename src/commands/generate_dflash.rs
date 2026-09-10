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

//! Offline `mlxcel generate --draft-kind dflash` driver (#1351).
//!
//! Drives [`DFlashGenerator`] against a target that implements
//! [`SpeculativeTarget`] the same way the server burst does
//! (`src/server/batch/speculative_burst.rs`): prefill through the verify
//! hook with the drafter's `target_layer_ids` captured, sample the first
//! bonus from the last prompt logits with the same penalty context the
//! classic path seeds, hand the captured prompt hidden states to the
//! drafter as its initial context, then run the round loop. At temperature
//! 0 the emitted tokens are the target's own greedy continuation, so the
//! output is token-identical to `mlxcel generate` without `--draft-model`.
//!
//! Targets: Laguna ([`LoadedModel::Laguna`]). Qwen 3.5 DFlash keeps its
//! server-only status (its offline arm was never wired).

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use anyhow::{Result, anyhow};
use mlxcel::LoadedModel;
use mlxcel_core::drafter::dflash::{DFlashGenerator, SpeculativeTarget};
use mlxcel_core::drafter::{DrafterKind, load_drafter};
use mlxcel_core::generate::{GenerationStats, LanguageModel, SamplingConfig};
use mlxcel_core::generation_policy::{initial_token_history, merged_eos_token_ids};
use mlxcel_core::sampling::{LogprobsConfig, TokenBiasMap, sample_token_optimized};

/// Whether the offline DFlash path serves `model`.
pub(super) fn offline_dflash_target_supported(model: &LoadedModel) -> bool {
    matches!(model, LoadedModel::Laguna(_))
}

/// Run the DFlash round loop offline. Returns the emitted tokens (first
/// bonus included, terminal EOS stripped) and decode statistics.
pub(super) fn run_offline_dflash(
    model: &LoadedModel,
    draft_model_path: &Path,
    prompt_tokens: &[i32],
    max_tokens: usize,
    sampling_config: &SamplingConfig,
    block_size: usize,
    token_bias: TokenBiasMap,
) -> Result<(Vec<i32>, GenerationStats)> {
    if block_size < 2 {
        return Err(anyhow!(
            "--draft-kind dflash with block_size={block_size} produces no draft proposals \
             (need >= 2); pass --draft-block-size with a value >= 2"
        ));
    }
    if prompt_tokens.is_empty() {
        return Err(anyhow!(
            "DFlash speculative decoding needs a non-empty prompt"
        ));
    }
    let LoadedModel::Laguna(wrapper) = model else {
        return Err(anyhow!(
            "--draft-kind dflash is only wired offline for Laguna targets (the Qwen 3.5 \
             DFlash pairing runs through mlxcel-server); omit --draft-model to run classic \
             decode"
        ));
    };
    // The same measured block-versus-chain gate the server burst runs
    // (`DFlashTargetModel::exactness_allows`): a host where a `block_size`-row
    // verify block is not byte-identical to the single-token chain would
    // silently differ from `mlxcel generate` without --draft-model at
    // temperature 0. Fails closed; `MLXCEL_MTP_ALLOW_INEXACT=1` engages anyway.
    if !wrapper.model.dflash_exactness_allows(block_size) {
        return Err(anyhow!(
            "Laguna DFlash speculative decoding declined: at --draft-block-size {block_size} \
             this GPU's multi-token verify block is not byte-identical to the single-token \
             decode chain, so temperature-0 output would silently differ from `mlxcel \
             generate` without --draft-model (see the probe verdict logged above). Try a \
             smaller --draft-block-size, or set MLXCEL_MTP_ALLOW_INEXACT=1 to engage anyway \
             and forfeit the byte-identity contract."
        ));
    }

    println!("Loading DFlash drafter from {draft_model_path:?}...");
    let (mut drafter, kind) = load_drafter(draft_model_path, Some(DrafterKind::Dflash))
        .map_err(|e| anyhow!("DFlash drafter load failed: {e}"))?;
    if kind != DrafterKind::Dflash {
        return Err(anyhow!(
            "drafter at {draft_model_path:?} did not resolve to a DFlash drafter (got {kind})"
        ));
    }
    // Family half of the pairing gate the server burst runs
    // (`required_family_pairing_error`): a plain Qwen 3.5 DFlash drafter
    // passes `validate_target_compat` on any target, and the mismatch would
    // surface as an MLX shape throw inside the drafter forward, which crosses
    // the cxx bridge as a process abort.
    if !drafter.is_laguna_dflash() {
        return Err(anyhow!(
            "the target is a Laguna checkpoint and the drafter at {draft_model_path:?} is not \
             a Laguna DFlash one. A Laguna target can only be paired with the Poolside DFlash \
             speculator published for its release (Laguna-XS-2.1 with Laguna-XS-2.1-DFlash, \
             and so on): another drafter's fc projection reads the residual streams it was \
             trained on, at a width this target does not produce. Point --draft-model at a \
             Laguna DFlash drafter, or drop it to run classic decode."
        ));
    }
    // The DFlash round loop verifies with a per-position argmax and has no
    // stochastic acceptance rule, so a sampling request would silently come
    // back greedy. Say so instead of drafting.
    if sampling_config.temperature > 0.0 && sampling_config.top_k != 1 {
        return Err(anyhow!(
            "--draft-kind dflash runs a greedy-only round loop; the request samples with \
             temperature {} / top_k {}. Pass --temp 0 (or --top-k 1), or drop --draft-model \
             to sample with classic decode.",
            sampling_config.temperature,
            sampling_config.top_k
        ));
    }
    let target_lm: &dyn LanguageModel = wrapper;
    drafter
        .validate_target_compat(target_lm)
        .map_err(|e| anyhow!("DFlash drafter incompatible with target: {e}"))?;
    // `DFlashGenerator::run` binds and resets the drafter itself; the bind
    // here only fails fast on a target that cannot lend its embedding.
    drafter
        .bind(target_lm)
        .map_err(|e| anyhow!("DFlash drafter bind failed: {e}"))?;
    let capture_layer_ids: Vec<usize> = drafter
        .dflash_target_layer_ids()
        .map(<[usize]>::to_vec)
        .unwrap_or_default();
    if capture_layer_ids.is_empty() {
        return Err(anyhow!(
            "DFlash drafter declares no target_layer_ids to capture"
        ));
    }
    println!("DFlash drafter loaded and bound (block_size = {block_size}).");

    let mut sampling = sampling_config.clone();
    sampling.token_bias = token_bias;
    let token_history = initial_token_history(prompt_tokens, sampling.needs_token_history());
    let eos_token_ids = merged_eos_token_ids(target_lm.eos_token_ids(), &sampling.stop_token_ids);

    let prefill_start = Instant::now();
    let mut caches = wrapper.model.make_speculative_caches(block_size);
    let prompt_arr = mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let verify_out =
        wrapper.verify_forward_with_capture_layers(&prompt_arr, &mut caches, &capture_layer_ids);
    let last_pos = prompt_tokens.len() as i32 - 1;
    let logits = wrapper.verify_logits(&verify_out);
    let logits_shape = mlxcel_core::array_shape(logits);
    let last_logits = mlxcel_core::slice(
        logits,
        &[0, last_pos, 0],
        &[logits_shape[0], last_pos + 1, logits_shape[2]],
    );
    let (first_bonus_arr, _adjusted) =
        sample_token_optimized(&last_logits, &sampling, &token_history);
    mlxcel_core::eval(&first_bonus_arr);
    let first_bonus = mlxcel_core::item_i32(&first_bonus_arr);
    // The whole prompt's captured hidden states seed the drafter context;
    // the drafter keeps the newest window of them.
    let first_hidden = wrapper.concat_hidden_for_drafter(&verify_out);
    mlxcel_core::eval(&first_hidden);
    let prefill_time_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let cancel = AtomicBool::new(false);
    let logprobs = LogprobsConfig::default();
    let mut generator = DFlashGenerator::new(
        drafter,
        sampling,
        block_size as u32,
        mlxcel_core::drafter::dflash::round_loop::DEFAULT_MASK_TOKEN_ID,
    );
    let output = generator
        .run(
            wrapper,
            target_lm,
            &mut caches,
            first_bonus,
            first_hidden,
            &eos_token_ids,
            max_tokens,
            &cancel,
            &logprobs,
        )
        .map_err(|e| anyhow!("DFlash round loop failed: {e}"))?;

    let d = &output.diagnostics;
    let mean_accepted = if d.rounds > 0 {
        d.accepted_tokens as f64 / d.rounds as f64
    } else {
        0.0
    };
    println!(
        "DFlash: rounds={} proposed={} accepted={} mean_accepted_length={mean_accepted:.2} \
         acceptance_rate={:.3} emitted_per_verify={:.2} draft_ms={:.1} verify_ms={:.1} \
         rollback_ms={:.1}",
        d.rounds,
        d.proposed_tokens,
        d.accepted_tokens,
        d.acceptance_rate(),
        d.emitted_per_verify(),
        d.draft_time_ms,
        d.verify_time_ms + d.target_argmax_time_ms,
        d.rollback_time_ms,
    );

    let mut tokens = Vec::with_capacity(output.tokens.len() + 1);
    tokens.push(first_bonus);
    tokens.extend(output.tokens);
    let tokens = super::generate::strip_trailing_eos(tokens, &eos_token_ids);

    let decode_time_ms = output.stats.decode_time_ms;
    let stats = GenerationStats {
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: tokens.len(),
        prefill_time_ms,
        decode_time_ms,
        prefill_tok_per_sec: if prefill_time_ms > 0.0 {
            prompt_tokens.len() as f64 / (prefill_time_ms / 1000.0)
        } else {
            0.0
        },
        decode_tok_per_sec: if decode_time_ms > 0.0 {
            tokens.len() as f64 / (decode_time_ms / 1000.0)
        } else {
            0.0
        },
    };
    Ok((tokens, stats))
}
