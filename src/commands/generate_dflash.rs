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
//! Targets: Laguna ([`LoadedModel::Laguna`]), Inkling 2
//! ([`LoadedModel::Inkling2VLM`], its bundled DSpark drafter, issue #3374;
//! an image prompt is prefilled from its merged embeddings), Raon-1 mini
//! ([`LoadedModel::Raon1Mini`], its DFlash2 drafter, issue #3436) and dense
//! Qwen3 and Llama ([`LoadedModel::Qwen3`], [`LoadedModel::Llama`], a Red Hat
//! Speculators or OpenBMB DSpark drafter, issue #3261). Qwen 3.5 DFlash
//! keeps its server-only status (its offline arm was never wired).

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use anyhow::{Result, anyhow};
use mlxcel::LoadedModel;
use mlxcel::vision::merge::InputEmbeddings;
use mlxcel_core::drafter::dflash::{DFlashGenerator, SpeculativeTarget};
use mlxcel_core::drafter::{Drafter, DrafterKind, load_drafter};
use mlxcel_core::generate::{GenerationStats, LanguageModel, SamplingConfig};
use mlxcel_core::generation_policy::{initial_token_history, merged_eos_token_ids};
use mlxcel_core::sampling::{LogprobsConfig, TokenBiasMap, sample_token_optimized};

/// Whether the offline DFlash path serves `model`.
pub(super) fn offline_dflash_target_supported(model: &LoadedModel) -> bool {
    matches!(
        model,
        LoadedModel::Laguna(_)
            | LoadedModel::Inkling2VLM(_)
            | LoadedModel::Raon1Mini(_)
            | LoadedModel::Qwen3(_)
            | LoadedModel::Llama(_)
    )
}

/// Run the DFlash round loop offline. Returns the emitted tokens (first
/// bonus included, terminal EOS stripped) and decode statistics.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_offline_dflash(
    model: &LoadedModel,
    draft_model_path: &Path,
    prompt_tokens: &[i32],
    max_tokens: usize,
    sampling_config: &SamplingConfig,
    block_size: usize,
    token_bias: TokenBiasMap,
    vlm_embeddings: Option<&InputEmbeddings>,
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
    match model {
        LoadedModel::Laguna(wrapper) => {
            if vlm_embeddings.is_some() {
                return Err(anyhow!(
                    "--draft-kind dflash does not support multimodal input in the offline \
                     `mlxcel generate` path; rerun with a text-only prompt, or omit \
                     --draft-model"
                ));
            }
            run_laguna(
                wrapper,
                draft_model_path,
                prompt_tokens,
                max_tokens,
                sampling_config,
                block_size,
                token_bias,
            )
        }
        LoadedModel::Inkling2VLM(vlm) => run_inkling2(
            vlm,
            draft_model_path,
            prompt_tokens,
            max_tokens,
            sampling_config,
            block_size,
            token_bias,
            vlm_embeddings,
        ),
        LoadedModel::Raon1Mini(model) => {
            if vlm_embeddings.is_some() {
                return Err(anyhow!(
                    "Raon-1 mini is a text model; --draft-kind dflash takes a text prompt"
                ));
            }
            run_raon1_mini(
                model,
                draft_model_path,
                prompt_tokens,
                max_tokens,
                sampling_config,
                block_size,
                token_bias,
            )
        }
        LoadedModel::Qwen3(target) => {
            if vlm_embeddings.is_some() {
                return Err(anyhow!(
                    "Qwen3 is a text model; --draft-kind dflash takes a text prompt"
                ));
            }
            run_dense(
                target,
                "Qwen3",
                draft_model_path,
                prompt_tokens,
                max_tokens,
                sampling_config,
                block_size,
                token_bias,
            )
        }
        LoadedModel::Llama(target) => {
            if vlm_embeddings.is_some() {
                return Err(anyhow!(
                    "this Llama checkpoint is a text model; --draft-kind dflash takes a text \
                     prompt"
                ));
            }
            run_dense(
                target,
                "Llama",
                draft_model_path,
                prompt_tokens,
                max_tokens,
                sampling_config,
                block_size,
                token_bias,
            )
        }
        _ => Err(anyhow!(
            "--draft-kind dflash is only wired offline for Laguna, Inkling 2, Raon-1 mini, \
             Qwen3 and Llama targets (the Qwen 3.5 DFlash pairing runs through \
             mlxcel-server); omit --draft-model to run classic decode"
        )),
    }
}

/// The DFlash round loop accepts proposals by a per-position argmax, so a
/// sampling request would silently come back greedy.
fn refuse_sampling(sampling_config: &SamplingConfig) -> Result<()> {
    if sampling_config.temperature > 0.0 && sampling_config.top_k != 1 {
        return Err(anyhow!(
            "--draft-kind dflash runs a greedy-only round loop; the request samples with \
             temperature {} / top_k {}. Pass --temp 0 (or --top-k 1), or drop --draft-model \
             to sample with classic decode.",
            sampling_config.temperature,
            sampling_config.top_k
        ));
    }
    Ok(())
}

/// Load the DFlash-kind drafter at `draft_model_path`, refuse it unless
/// `family_ok` accepts it (`family_error` says why), then run the drafter's
/// compatibility gate and a fail-fast bind against `target_lm`.
fn load_paired_drafter(
    draft_model_path: &Path,
    target_lm: &dyn LanguageModel,
    family_ok: fn(&dyn Drafter) -> bool,
    family_error: impl FnOnce() -> String,
) -> Result<(Box<dyn Drafter>, Vec<usize>)> {
    println!("Loading DFlash drafter from {draft_model_path:?}...");
    let (mut drafter, kind) = load_drafter(draft_model_path, Some(DrafterKind::Dflash))
        .map_err(|e| anyhow!("DFlash drafter load failed: {e}"))?;
    if kind != DrafterKind::Dflash {
        return Err(anyhow!(
            "drafter at {draft_model_path:?} did not resolve to a DFlash drafter (got {kind})"
        ));
    }
    // Family half of the pairing gate the server burst runs
    // (`required_family_pairing_error`): a mismatched drafter would surface
    // as an MLX shape throw inside the drafter forward, which crosses the cxx
    // bridge as a process abort.
    if !family_ok(drafter.as_ref()) {
        return Err(anyhow!(family_error()));
    }
    drafter
        .validate_target_compat(target_lm)
        .map_err(|e| anyhow!("DFlash drafter incompatible with target: {e}"))?;
    // `DFlashGenerator::run` binds and resets the drafter itself; the bind
    // here only fails fast on a target that cannot lend its parts.
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
    Ok((drafter, capture_layer_ids))
}

fn inexact_error(family: &str, block_size: usize) -> anyhow::Error {
    anyhow!(
        "{family} DFlash speculative decoding declined: at --draft-block-size {block_size} \
         this GPU's multi-token verify block is not byte-identical to the single-token \
         decode chain, so temperature-0 output would silently differ from `mlxcel \
         generate` without --draft-model (see the probe verdict logged above). Try a \
         smaller --draft-block-size, or set MLXCEL_MTP_ALLOW_INEXACT=1 to engage anyway \
         and forfeit the byte-identity contract."
    )
}

fn run_laguna(
    wrapper: &mlxcel::models::LagunaWrapper,
    draft_model_path: &Path,
    prompt_tokens: &[i32],
    max_tokens: usize,
    sampling_config: &SamplingConfig,
    block_size: usize,
    token_bias: TokenBiasMap,
) -> Result<(Vec<i32>, GenerationStats)> {
    // The same measured block-versus-chain gate the server burst runs
    // (`DFlashTargetModel::exactness_allows`). Fails closed;
    // `MLXCEL_MTP_ALLOW_INEXACT=1` engages anyway.
    if !wrapper.model.dflash_exactness_allows(block_size) {
        return Err(inexact_error("Laguna", block_size));
    }
    let target_lm: &dyn LanguageModel = wrapper;
    let (drafter, capture_layer_ids) = load_paired_drafter(
        draft_model_path,
        target_lm,
        |d| d.is_laguna_dflash(),
        || {
            format!(
                "the target is a Laguna checkpoint and the drafter at {draft_model_path:?} is \
                 not a Laguna DFlash one. A Laguna target can only be paired with the Poolside \
                 DFlash speculator published for its release (Laguna-XS-2.1 with \
                 Laguna-XS-2.1-DFlash, and so on): another drafter's fc projection reads the \
                 residual streams it was trained on, at a width this target does not produce. \
                 Point --draft-model at a Laguna DFlash drafter, or drop it to run classic \
                 decode."
            )
        },
    )?;
    refuse_sampling(sampling_config)?;
    println!("DFlash drafter loaded and bound (block_size = {block_size}).");
    // The round loop's per-position verify argmax applies no token bias
    // (only the first bonus goes through `sample_token_optimized`), so a
    // bias would be honoured for one token and dropped for the rest.
    if !token_bias.is_empty() {
        return Err(anyhow!(
            "--draft-kind dflash cannot honour a token bias (--lang-bias or a model's \
             suppressed-token set): the verify argmax applies none. Drop --draft-model to \
             decode classically with the bias."
        ));
    }
    let prefill_start = Instant::now();
    let mut caches = wrapper.model.make_speculative_caches(block_size);
    let prompt_arr = mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let verify_out =
        wrapper.verify_forward_with_capture_layers(&prompt_arr, &mut caches, &capture_layer_ids);
    drive_round_loop(
        wrapper,
        drafter,
        &mut caches,
        verify_out,
        prompt_tokens,
        max_tokens,
        sampling_config,
        block_size,
        token_bias,
        prefill_start,
    )
}

/// A dense Qwen3 or Llama target drafting with a DSpark drafter (issue
/// #3261): `RedHatAI/Qwen3-4B-speculator.dspark` on `Qwen/Qwen3-4B`,
/// `openbmb/MiniCPM5-2B-DSpark` on `openbmb/MiniCPM5-2B`.
///
/// The gates are the server burst's, in its order: the measured
/// block-versus-chain probe before any drafter IO, then the drafter's own
/// pairing gate (`validate_target_compat`, inside [`load_paired_drafter`]),
/// then the target's tap and vocabulary bounds. The prefill forwards the
/// pieces classic `mlxcel generate` forwards, so the caches every later
/// token is computed from are the classic ones.
#[allow(clippy::too_many_arguments)]
fn run_dense<T>(
    target: &T,
    family: &'static str,
    draft_model_path: &Path,
    prompt_tokens: &[i32],
    max_tokens: usize,
    sampling_config: &SamplingConfig,
    block_size: usize,
    token_bias: TokenBiasMap,
) -> Result<(Vec<i32>, GenerationStats)>
where
    T: mlxcel::models::dense_speculative::DenseSpeculativeTarget,
{
    // Fails closed; `MLXCEL_MTP_ALLOW_INEXACT=1` engages anyway.
    if !target.dflash_exactness_allows(block_size) {
        return Err(inexact_error(family, block_size));
    }
    let target_lm: &dyn LanguageModel = target;
    // These targets name no required drafter family (as Qwen 3.5 does not),
    // so the family half of the gate admits every DFlash-kind drafter and
    // the numbers below tell a mismatched one apart.
    let (drafter, capture_layer_ids) =
        load_paired_drafter(draft_model_path, target_lm, |_| true, String::new)?;
    if let Some(reason) = target.dflash_drafter_pairing_error(drafter.as_ref()) {
        return Err(anyhow!("DFlash drafter incompatible with target: {reason}"));
    }
    // A greedy-only drafter (DSpark) has no stochastic acceptance rule, and
    // the round loop's verify argmax would return a sampling request greedy.
    refuse_sampling(sampling_config)?;
    // The round loop's per-position verify argmax applies no token bias
    // (only the first bonus goes through `sample_token_optimized`), so a
    // bias would be honoured for one token and dropped for the rest.
    if !token_bias.is_empty() {
        return Err(anyhow!(
            "--draft-kind dflash cannot honour a token bias (--lang-bias or a model's \
             suppressed-token set): the verify argmax applies none. Drop --draft-model to \
             decode classically with the bias."
        ));
    }
    println!("DFlash drafter loaded and bound (block_size = {block_size}).");
    let prefill_start = Instant::now();
    let mut caches = LanguageModel::make_caches(target);
    let prompt_arr = mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let verify_out = target.dflash_prefill_planned(
        &prompt_arr,
        &mut caches,
        &capture_layer_ids,
        mlxcel_core::prefill_plan::prefill_chunk_len(),
    );
    drive_round_loop(
        target,
        drafter,
        &mut caches,
        verify_out,
        prompt_tokens,
        max_tokens,
        sampling_config,
        block_size,
        token_bias,
        prefill_start,
    )
}

/// Raon-1 mini drafting with KRAFTON's DFlash2 speculator (issue #3436). The
/// burst runs at the model-default reasoning effort (`--reasoning-effort`),
/// which `generate` set on the model before this call.
fn run_raon1_mini(
    model: &mlxcel::models::Raon1MiniModel,
    draft_model_path: &Path,
    prompt_tokens: &[i32],
    max_tokens: usize,
    sampling_config: &SamplingConfig,
    block_size: usize,
    token_bias: TokenBiasMap,
) -> Result<(Vec<i32>, GenerationStats)> {
    let target_lm: &dyn LanguageModel = model;
    let (drafter, capture_layer_ids) = load_paired_drafter(
        draft_model_path,
        target_lm,
        |d| d.is_raon_dflash2(),
        || {
            format!(
                "the target is a Raon-1 mini checkpoint and the drafter at {draft_model_path:?} \
                 is not its DFlash2 speculator; point --draft-model at \
                 KRAFTON/Raon-1-mini-dflash2, or drop it to run classic decode."
            )
        },
    )?;
    let trained = drafter.configured_block_size().unwrap_or(block_size);
    if !(2..=trained).contains(&block_size) {
        return Err(anyhow!(
            "the Raon DFlash2 drafter serves verify blocks of 2..={trained} rows (its trained \
             block); got --draft-block-size {block_size}"
        ));
    }
    // The draft is deterministic, so sampling (and penalties, and a token
    // bias) runs through the sampled verify rows; only a feedback sampler
    // (Mirostat, adaptive-p) cannot (issue #3436).
    if !mlxcel_core::drafter::dflash::sampled_verify::sampled_verify_serves(sampling_config) {
        return Err(anyhow!(
            "--draft-kind dflash on Raon-1 mini samples each verify row independently and \
             cannot carry a feedback sampler's state (Mirostat, adaptive-p); drop that \
             sampler or --draft-model."
        ));
    }
    if !model.dflash_exactness_allows(block_size) {
        return Err(inexact_error("Raon-1 mini", block_size));
    }
    println!("Raon DFlash2 drafter loaded and bound (block_size = {block_size}).");
    let prefill_start = Instant::now();
    let mut caches = model.make_speculative_caches(None);
    // The drafter keeps a window of the newest positions; older prompt rows
    // are not captured (issue #3436).
    if let Some(window) = drafter.context_window() {
        model.limit_prefill_capture(&mut caches, window);
    }
    let prompt_arr = mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let verify_out =
        model.prefill_forward_with_capture_layers(&prompt_arr, &mut caches, &capture_layer_ids);
    drive_round_loop(
        model,
        drafter,
        &mut caches,
        verify_out,
        prompt_tokens,
        max_tokens,
        sampling_config,
        block_size,
        token_bias,
        prefill_start,
    )
}

/// Inkling 2 drafting with the DSpark drafter bundled in its own checkpoint.
#[allow(clippy::too_many_arguments)]
fn run_inkling2(
    vlm: &mlxcel::vision::Inkling2VlModel,
    draft_model_path: &Path,
    prompt_tokens: &[i32],
    max_tokens: usize,
    sampling_config: &SamplingConfig,
    block_size: usize,
    token_bias: TokenBiasMap,
    vlm_embeddings: Option<&InputEmbeddings>,
) -> Result<(Vec<i32>, GenerationStats)> {
    let target_lm: &dyn LanguageModel = vlm;
    let (drafter, capture_layer_ids) = load_paired_drafter(
        draft_model_path,
        target_lm,
        |d| d.is_inkling2_dspark(),
        || {
            format!(
                "the target is an Inkling 2 checkpoint and the drafter at {draft_model_path:?} \
                 is not its bundled DSpark drafter; pass the same checkpoint (the directory \
                 holding mtp.safetensors) to --draft-model, or drop it to run classic decode."
            )
        },
    )?;
    let trained = drafter.configured_block_size().unwrap_or(block_size);
    if !(2..=trained).contains(&block_size) {
        return Err(anyhow!(
            "Inkling 2 DSpark drafter serves verify blocks of 2..={trained} rows (one anchor and \
             up to {} proposals, its trained block); got --draft-block-size {block_size}",
            trained - 1
        ));
    }
    refuse_sampling(sampling_config)?;
    // The verify forward applies the model's own suppressed ids (the image
    // placeholder) to every row, which is exactly the bias classic decode
    // composes for this model. Anything beyond it (`--lang-bias`) the verify
    // argmax would drop after the first token.
    let suppressed = LanguageModel::output_suppressed_token_ids(vlm);
    let only_suppression = token_bias.len() == suppressed.len()
        && suppressed
            .iter()
            .all(|id| token_bias.get(id).is_some_and(|b| *b == f32::NEG_INFINITY));
    if !only_suppression {
        return Err(anyhow!(
            "--draft-kind dflash cannot honour --lang-bias on Inkling 2: the verify argmax \
             applies only the model's suppressed-token set. Drop --draft-model to decode \
             classically with the bias."
        ));
    }
    // Same gate as the server burst. Fails closed; `MLXCEL_MTP_ALLOW_INEXACT=1`
    // engages anyway.
    if !vlm.text.dflash_exactness_allows(block_size) {
        return Err(inexact_error("Inkling 2", block_size));
    }
    println!("Inkling 2 DSpark drafter loaded and bound (block_size = {block_size}).");

    let prefill_start = Instant::now();
    let mut caches = vlm.text.make_speculative_caches();
    let verify_out = match vlm_embeddings {
        // Image prompt: the merged, already-normalized embeddings. The
        // drafter's context then starts with the image rows' captures, and
        // every later row is text.
        Some(embeddings) => {
            let map = mlxcel::models::inkling2::speculative::suppression_map(&suppressed);
            vlm.text.forward_speculative_embeddings(
                &embeddings.inputs_embeds,
                &mut caches,
                &capture_layer_ids,
                false,
                &map,
            )
        }
        None => {
            let prompt_arr =
                mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);
            vlm.prefill_forward_with_capture_layers(&prompt_arr, &mut caches, &capture_layer_ids)
        }
    };
    drive_round_loop(
        vlm,
        drafter,
        &mut caches,
        verify_out,
        prompt_tokens,
        max_tokens,
        sampling_config,
        block_size,
        token_bias,
        prefill_start,
    )
}

/// The shared tail: first bonus from the prefill's last logits, the whole
/// prompt's captures as the drafter's initial context, then the round loop.
#[allow(clippy::too_many_arguments)]
fn drive_round_loop<T>(
    target: &T,
    drafter: Box<dyn Drafter>,
    caches: &mut [T::Cache],
    verify_out: T::VerifyOut,
    prompt_tokens: &[i32],
    max_tokens: usize,
    sampling_config: &SamplingConfig,
    block_size: usize,
    token_bias: TokenBiasMap,
    prefill_start: Instant,
) -> Result<(Vec<i32>, GenerationStats)>
where
    T: SpeculativeTarget + LanguageModel,
{
    let target_lm: &dyn LanguageModel = target;
    let mut sampling = sampling_config.clone();
    sampling.token_bias = token_bias;
    let token_history = initial_token_history(prompt_tokens, sampling.needs_token_history());
    let eos_token_ids = merged_eos_token_ids(target_lm.eos_token_ids(), &sampling.stop_token_ids);

    let logits = target.verify_logits(&verify_out);
    let logits_shape = mlxcel_core::array_shape(logits);
    let last_pos = logits_shape[1] - 1;
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
    let first_hidden = target.concat_hidden_for_drafter(&verify_out);
    mlxcel_core::eval(&first_hidden);
    // A target that captured only the newest rows (Raon-1 mini, issue
    // #3436) leaves the older prompt rows to the drafter's position count.
    let captured_rows = mlxcel_core::array_shape(&first_hidden)
        .get(1)
        .map_or(0, |&n| usize::try_from(n).unwrap_or(0));
    let first_context_start = prompt_tokens.len().saturating_sub(captured_rows);
    drop(verify_out);
    let prefill_time_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let cancel = AtomicBool::new(false);
    let logprobs = LogprobsConfig::default();
    let mut generator = DFlashGenerator::new(
        drafter,
        sampling,
        block_size as u32,
        mlxcel_core::drafter::dflash::round_loop::DEFAULT_MASK_TOKEN_ID,
    );
    generator.set_token_history(token_history);
    generator.set_first_context_start(first_context_start);
    let output = generator
        .run(
            target,
            target_lm,
            caches,
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
    // `draft_ms` and `verify_graph_ms` are host-side graph construction
    // (MLX evaluates lazily); the device work of a round is synchronized in
    // `verify_sync_ms`, so the three together are where a round's time goes.
    println!(
        "DFlash: rounds={} classic_steps={} proposed={} accepted={} mean_accepted_length={mean_accepted:.2} \
         acceptance_rate={:.3} emitted_per_verify={:.2} draft_ms={:.1} verify_ms={:.1} \
         verify_graph_ms={:.1} verify_sync_ms={:.1} hidden_concat_ms={:.1} walk_ms={:.1} \
         rollback_ms={:.1} decode_ms={:.1}",
        d.rounds,
        d.classic_steps,
        d.proposed_tokens,
        d.accepted_tokens,
        d.acceptance_rate(),
        d.emitted_per_verify(),
        d.draft_time_ms,
        d.verify_time_ms + d.target_argmax_time_ms,
        d.verify_time_ms,
        d.target_argmax_time_ms,
        d.hidden_concat_time_ms,
        d.walk_time_ms,
        d.rollback_time_ms,
        d.total_decode_time_ms,
    );

    let mut tokens = Vec::with_capacity(output.tokens.len() + 1);
    tokens.push(first_bonus);
    tokens.extend(output.tokens);
    let tokens = super::generate::strip_trailing_eos(tokens, &eos_token_ids);

    let decode_time_ms = output.stats.decode_time_ms;
    // `tokens` starts with the first bonus sampled in prefill, outside the
    // round loop's decode window.
    let decode_count = tokens.len().saturating_sub(1);
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
        decode_tok_per_sec: mlxcel_core::generate::decode_tok_per_sec(decode_count, decode_time_ms),
    };
    Ok((tokens, stats))
}
