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

//! Shared request-to-generation option adapters for server routes.
//!
//! Chat and native completion requests expose slightly different field names,
//! but once their overrides are resolved they should map onto the same
//! `ServerGenerateOptions` policy.

use super::{LiveSettings, ServerConfig, ServerGenerateOptions};
use crate::sampling::{ResolvedSamplingParams, build_sampling_config};
use crate::server::batch::RequestPriority;
use crate::server::config::ReasoningBudgetOverride;
use mlxcel_core::LoopDetectionConfig;
use mlxcel_core::sampling::{LogprobSource, LogprobsConfig};

/// Conservative built-in loop-detection threshold (issue #432, recalibrated by
/// issue #967): scan single-token through 20-token tail patterns and end
/// generation once any repeats twelve times. This is the engine-level default
/// applied to the Gemma 4 family (with no per-user configuration), and also the
/// "force-enable" configuration for the `MLXCEL_LOOP_DETECTION` global override.
///
/// The original `min_count` of 4 was low enough to fire on ordinary prose. A
/// markdown table alignment row (`| :--- | :--- | :--- | :--- |`) tokenizes as a
/// 3-token block repeated once per column, so any table with 4 or more columns
/// was truncated mid-row. The asymmetry favors the higher count: a false
/// positive silently truncates a correct answer and reports a normal stop, while
/// a genuine collapse repeats until the token budget is exhausted.
///
/// The higher count costs two things, both accepted deliberately. A pattern of
/// size `p` is now cut after `12p` tokens instead of `4p`, so a genuine collapse
/// runs `8p` tokens longer (8 for the single-token collapse in the #967
/// reproduction, 160 at the `p = 20` ceiling). And because firing at size `p`
/// needs `p * 12` tail tokens rather than `p * 4`, a pattern is undetectable
/// whenever the remaining budget is shorter than `p * 12`: under
/// `thinking_budget_tokens = 128`, patterns of `p >= 11` can no longer be caught
/// where `p * 4 = 44` tokens previously sufficed. Short-budget requests
/// therefore lose coverage of long patterns, not of the short ones that dominate
/// real collapses.
///
/// Raising the count does not remove the false-positive class, only shrink it.
/// Grammar-constrained decoding is therefore excluded from the automatic
/// activation signal: a `json_schema` request for an integer array of 30 zeros
/// legitimately repeats the same token beyond any fixed `min_count`. Tool-shaped
/// prompts remain covered, while per-request and global overrides can still
/// explicitly enable detection for any request.
pub(crate) const LOOP_DETECTION_RECOMMENDED: LoopDetectionConfig =
    LoopDetectionConfig::new(1, 20, 12);

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct RequestOptionOverrides {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub dry_multiplier: Option<f32>,
    pub dry_base: Option<f32>,
    pub dry_allowed_length: Option<usize>,
    pub dry_penalty_last_n: Option<usize>,
    pub dry_sequence_breakers: Option<Vec<i32>>,
    /// b10621 breaker STRINGS from the native `dry_sequence_breakers` field
    /// (#1485). `Some` replaces both the server default set and any id-based
    /// override; the head map is derived against the vocabulary at enqueue
    /// time.
    pub dry_sequence_breaker_strings: Option<Vec<String>>,
    /// Mirostat mode override (#1485), validated to `{0, 1, 2}` at the
    /// request layer.
    pub mirostat: Option<i32>,
    /// Mirostat tau override (#1485).
    pub mirostat_tau: Option<f32>,
    /// Mirostat eta override (#1485).
    pub mirostat_eta: Option<f32>,
    /// Dynamic-temperature range override (#1485).
    pub dynatemp_range: Option<f32>,
    /// Dynamic-temperature exponent override (#1485).
    pub dynatemp_exponent: Option<f32>,
    /// Adaptive-p target override (#1485; soft-clamped to `<= 1.0` at the
    /// request layer, upstream's soft schema limit).
    pub adaptive_target: Option<f32>,
    /// Adaptive-p decay override (#1485; hard limits `0.0..=0.99` enforced
    /// with a 400 at the request layer).
    pub adaptive_decay: Option<f32>,
    /// Whether the request's `samplers` field named the `adaptive_p` stage
    /// (#1485). `Some` (the field was present) replaces the server-wide
    /// sampler list wholesale, as b10621 replaces `params.samplers`.
    pub adaptive_p_named: Option<bool>,
    /// b10621 `min_keep` override (#1485; hard limits `0..=2147483647`
    /// enforced at the request layer).
    pub min_keep: Option<usize>,
    /// Resolved numeric logit biases from the request (#1485): `Some`
    /// replaces the server-wide `--logit-bias` set wholesale, as b10621's
    /// field handler clears and rebuilds the list.
    pub logit_bias: Option<Vec<(i32, f32)>>,
    /// Text-keyed logit biases from the request (#1485): each string is
    /// tokenized (special parsing off) at enqueue time and the bias applies
    /// to every resulting token, upstream's string-key form.
    pub logit_bias_texts: Vec<(String, f32)>,
    /// b10621 `n_probs` / `logprobs` (#1485): per-token top-N probability
    /// reporting on the native route.
    pub n_probs: Option<usize>,
    /// b10621 `post_sampling_probs` (#1485).
    pub post_sampling_probs: Option<bool>,
    /// XTC (Exclude Top Choices) per-step probability override. Validated at
    /// the request layer (`0.0..=1.0`) before it reaches here.
    pub xtc_probability: Option<f32>,
    /// XTC probability threshold override. Validated at the request layer
    /// (`0.0..=0.5`) before it reaches here.
    pub xtc_threshold: Option<f32>,
    /// Top-n-sigma logit-filter override. Validated at the request layer
    /// (finite, `>= 0.0`) before it reaches here.
    pub top_n_sigma: Option<f32>,
    /// Locally typical sampling override. Validated at the request layer
    /// (finite, in `(0.0, 1.0]`) on the OpenAI-shaped endpoints; the native
    /// `/completion` route sanitizes out-of-domain values itself.
    pub typical_p: Option<f32>,
    /// Repetition / frequency / presence penalty window override (b10621
    /// `repeat_last_n`, #1436): `-1` = full history, `0` = disables the
    /// stage, `N > 0` = last N tokens. Absent falls back to the server-wide
    /// `--repeat-last-n` default (64, matching b10621).
    pub penalty_last_n: Option<i32>,
    /// b10621 `ignore_eos` override (#1436). Absent falls back to the
    /// server-wide `--ignore-eos` default.
    pub ignore_eos: Option<bool>,
    pub stop_sequences: Option<Vec<String>>,
    pub priority: RequestPriority,
    /// per-request thinking-token budget override.
    pub reasoning_budget: ReasoningBudgetOverride,
    /// whether the rendered prompt primes `<think>\n` (true for
    /// chat endpoints) vs. takes a free-form prompt (false for raw text
    /// completion endpoints).
    pub thinking_enter_block_on_start: bool,
    /// Explicit per-request loop-detection override (issue #432). `Some` when
    /// the request set any of the vLLM `max_pattern_size` / `min_pattern_size`
    /// / `min_count` fields; it then wins over the global override and the
    /// family default-on. `None` lets the global override / family policy
    /// decide.
    pub loop_detection_request: Option<LoopDetectionConfig>,
    /// Whether this request carries an amplifier of the Gemma 4 repetition
    /// collapse: tool declarations that reach the rendered prompt or tool-shaped
    /// message content (`tool_calls` / `tool_call_id`). Only such requests get
    /// the family loop-detection default-on; plain and grammar-only requests
    /// stay on the disabled baseline. Chat-shaped routes compute it with
    /// [`chat_carries_loop_amplifier`], which cannot be handed the wrong tools
    /// slice. The `Default` of `false` is the safe (detection off) baseline for
    /// any construction site that does not set it.
    pub request_carries_loop_amplifier: bool,
}

/// Whether a request carries one of the amplifiers that justify turning the
/// Gemma 4 family loop-detection default-on for it (issue #967).
///
/// `tools` is the set the model will actually see. Chat-shaped callers should
/// use [`chat_carries_loop_amplifier`] instead of calling this directly, so the
/// slice can only come from [`crate::server::chat_request::effective_tools`].
/// Absent or empty is not a declaration. A non-empty `tools` array therefore
/// counts only when it reaches the rendered prompt, so `tool_choice: "none"`
/// does not count: `effective_tools` drops the declarations in that case,
/// leaving a prompt identical to plain chat. The gate and the template read the
/// same helper on purpose, so the policy cannot drift from what the model sees.
/// Tool-shaped *message* content is the separate disjunct that
/// [`chat_carries_loop_amplifier`] adds.
pub(crate) fn carries_loop_amplifier(
    tools: Option<&[crate::server::types::request::Tool]>,
) -> bool {
    tools.is_some_and(|tools| !tools.is_empty())
}

/// [`carries_loop_amplifier`] for a chat-shaped request, reading the tools half
/// from the request itself.
///
/// Two disjuncts, either of which makes the request amplified:
///
/// 1. Tool declarations that reach the rendered prompt, per
///    [`crate::server::chat_request::effective_tools`]. `tool_choice: "none"`
///    drops these, so on its own it does not amplify.
/// 2. Tool-shaped message content, per
///    [`crate::server::chat_request::has_tool_fields`]. Messages carrying
///    `tool_calls` or `tool_call_id` take the raw-JSON render path, which writes
///    those fields into the prompt regardless of `effective_tools`. An agent loop
///    replaying prior tool calls on a follow-up turn that omits the top-level
///    `tools` array is the common case, and it is thoroughly tool-shaped. Issue
///    #432's unconditional default-on covered these turns; the #967 narrowing was
///    meant to exclude plain chat only, so they must stay covered.
///
/// Every chat-shaped route goes through this rather than assembling the signal
/// at the call site. Taking the whole request removes the failure mode: there is
/// no slice parameter a caller could fill with the raw `request.tools`, which
/// would quietly restore `tool_choice: "none"` to the amplified set.
pub(crate) fn chat_carries_loop_amplifier(
    request: &crate::server::types::request::ChatCompletionRequest,
) -> bool {
    carries_loop_amplifier(crate::server::chat_request::effective_tools(request))
        || crate::server::chat_request::has_tool_fields(request)
}

/// Build a [`LoopDetectionConfig`] from the raw vLLM-style request fields.
///
/// Returns `None` when the request set none of the three fields (so the global
/// override / family policy decides). Returns `Some` when any field is present,
/// treating the request as authoritative, including an explicit disable
/// (`max_pattern_size = 0`). Unset companions fall back to vLLM defaults (`0`),
/// which the detector normalizes (`min_pattern_size 0 -> 1`, disabled when
/// `max_pattern_size == 0` or `min_count < 2`).
pub(crate) fn loop_detection_from_request(
    max_pattern_size: Option<usize>,
    min_pattern_size: Option<usize>,
    min_count: Option<usize>,
) -> Option<LoopDetectionConfig> {
    if max_pattern_size.is_none() && min_pattern_size.is_none() && min_count.is_none() {
        return None;
    }
    Some(LoopDetectionConfig::new(
        min_pattern_size.unwrap_or(0),
        max_pattern_size.unwrap_or(0),
        min_count.unwrap_or(0),
    ))
}

/// Resolve the effective loop-detection config for one request.
///
/// Precedence (highest first):
///
/// 1. Explicit per-request override, which wins over everything including the
///    amplifier gate below (it may force-enable or force-disable).
/// 2. Global operator override (`MLXCEL_LOOP_DETECTION`), which applies to every
///    request unconditionally in both its `off` and its `on` / triple forms and
///    is deliberately not subject to the gate.
/// 3. The Gemma 4 family engine-level default-on, gated on this request
///    carrying an amplifier (`request_carries_loop_amplifier`).
/// 4. Otherwise disabled.
///
/// Issue #967 added the gate at step 3. #432 turned the family default on for
/// every request, but running the detector outside tool-shaped traffic costs a
/// class of false positives that truncate correct answers while reporting a
/// normal stop. Issue #977 removes grammar constraints from the gate because a
/// schema-valid uniform array is indistinguishable from a repetition collapse
/// at the token level. Tool-shaped requests keep the default-on, so a downstream
/// serving app still needs no configuration for the protected traffic.
pub(crate) fn resolve_loop_detection(
    request_override: Option<LoopDetectionConfig>,
    global_override: Option<LoopDetectionConfig>,
    family_default_on: bool,
    request_carries_loop_amplifier: bool,
) -> LoopDetectionConfig {
    if let Some(req) = request_override {
        return req;
    }
    if let Some(global) = global_override {
        return global;
    }
    if family_default_on && request_carries_loop_amplifier {
        return LOOP_DETECTION_RECOMMENDED;
    }
    LoopDetectionConfig::disabled()
}

#[allow(dead_code)]
pub(crate) fn build_server_generate_options(
    config: &ServerConfig,
    overrides: RequestOptionOverrides,
) -> ServerGenerateOptions {
    build_server_generate_options_with_live(config, &config.live_settings(), overrides)
}

/// Build request options from one immutable live-settings snapshot.
pub(crate) fn build_server_generate_options_with_live(
    config: &ServerConfig,
    live: &LiveSettings,
    overrides: RequestOptionOverrides,
) -> ServerGenerateOptions {
    let mut sampling = build_sampling_config(ResolvedSamplingParams {
        temperature: overrides.temperature.unwrap_or(live.default_temperature),
        top_k: overrides.top_k.unwrap_or(live.default_top_k),
        top_p: overrides.top_p.unwrap_or(live.default_top_p),
        min_p: overrides.min_p.unwrap_or(live.default_min_p),
        seed: overrides.seed.or(live.default_seed),
        repetition_penalty: overrides
            .repetition_penalty
            .unwrap_or(live.default_repetition_penalty),
        dry_multiplier: overrides
            .dry_multiplier
            .unwrap_or(live.default_dry_multiplier),
        // b10621's schema handler replaces a dry_base below 1.0 with the
        // server default rather than erroring (#1436).
        dry_base: {
            let v = overrides.dry_base.unwrap_or(live.default_dry_base);
            if v < 1.0 { live.default_dry_base } else { v }
        },
        dry_allowed_length: overrides
            .dry_allowed_length
            .unwrap_or(live.default_dry_allowed_length),
        // Clamped to upstream's INT32_MAX schema ceiling so a wire value can
        // never collide with the internal DRY_FULL_HISTORY sentinel (#1436).
        dry_penalty_last_n: overrides
            .dry_penalty_last_n
            .unwrap_or(live.default_dry_penalty_last_n)
            .min(i32::MAX as usize),
        // b10621 value domain (#1485): the sampler's exact-id breaker list
        // serves only the OpenAI-shaped id override now; string breakers
        // (the native field, or the server-wide default set) travel on
        // `ServerGenerateOptions::dry_breaker_strings` and are derived into
        // the head map at enqueue time.
        dry_sequence_breakers: overrides.dry_sequence_breakers.clone().unwrap_or_default(),
        frequency_penalty: overrides
            .frequency_penalty
            .unwrap_or(live.default_frequency_penalty),
        presence_penalty: overrides
            .presence_penalty
            .unwrap_or(live.default_presence_penalty),
        // #1436: XTC gained the b10621 server-wide defaults
        // (--xtc-probability / --xtc-threshold, defaults 0.0 / 0.1, the
        // disabled baseline); an absent request field resolves to them.
        xtc_probability: overrides
            .xtc_probability
            .unwrap_or(config.default_xtc_probability),
        xtc_threshold: overrides
            .xtc_threshold
            .unwrap_or(config.default_xtc_threshold),
        // #1436: top-n-sigma gained a server-wide default (--top-nsigma),
        // sanitized at startup so it is either a valid enabled value or the
        // 0.0 disabled baseline; an absent request field resolves to it.
        top_n_sigma: overrides.top_n_sigma.unwrap_or(config.default_top_n_sigma),
        // Unlike XTC and top-n-sigma, typical_p follows the classic sampler
        // knobs (--temp / --top-p / --min-p): the server-wide --typical /
        // --typical-p default applies when the request omits the field,
        // matching llama-server's sampling-params surface.
        typical_p: overrides.typical_p.unwrap_or(config.default_typical_p),
        // b10621 window semantics (#1436): the server-wide --repeat-last-n
        // default (64) bounds the repetition/frequency/presence penalties for
        // every request that does not carry its own repeat_last_n. Before
        // #1436 the flag was parsed and echoed by /props but never reached
        // the sampler, which scanned the full history.
        penalty_last_n: overrides.penalty_last_n.unwrap_or_else(|| {
            i32::try_from(live.default_repetition_context_size).unwrap_or(i32::MAX)
        }),
        stop_token_ids: Vec::new(),
        // #1485 sampling remainder.
        mirostat: overrides.mirostat.unwrap_or(config.default_mirostat),
        mirostat_tau: overrides
            .mirostat_tau
            .unwrap_or(config.default_mirostat_tau),
        mirostat_eta: overrides
            .mirostat_eta
            .unwrap_or(config.default_mirostat_eta),
        dynatemp_range: overrides
            .dynatemp_range
            .unwrap_or(config.default_dynatemp_range),
        dynatemp_exponent: overrides
            .dynatemp_exponent
            .unwrap_or(config.default_dynatemp_exponent),
        // Adaptive-p runs only when the effective sampler list names the
        // stage (b10621 activates it solely through `params.samplers`): a
        // request `samplers` field replaces the server-wide list wholesale,
        // so its named flag wins outright when present.
        adaptive_target: {
            let named = overrides
                .adaptive_p_named
                .unwrap_or(config.default_adaptive_p_named);
            if named {
                overrides
                    .adaptive_target
                    .unwrap_or(config.default_adaptive_target)
            } else {
                -1.0
            }
        },
        adaptive_decay: overrides
            .adaptive_decay
            .unwrap_or(config.default_adaptive_decay),
        min_keep: overrides.min_keep.unwrap_or(0),
    });

    // Loop-detection policy (issue #432) is resolved here, in the server
    // control plane, where the loaded model family is visible. The Gemma 4
    // family is default-on (engine-level, no per-user configuration) for
    // requests that carry an amplifier (issue #967); the shared
    // `build_sampling_config` leaves the field disabled for everything else,
    // preserving the bit-exact baseline.
    sampling.loop_detection = resolve_loop_detection(
        overrides.loop_detection_request,
        live.loop_detection,
        config.model_is_gemma4_family,
        overrides.request_carries_loop_amplifier,
    );
    sampling.token_bias = live.resolved_token_bias.clone();

    // b10621 -r / --reverse-prompt (#1436): upstream seeds the task's stop
    // set from the CLI antiprompt list and then REPLACES it wholesale when
    // the request carries any effective stop string, falling back to the CLI
    // list only when the request supplied none. A request's non-empty `stop`
    // therefore discards the server-wide strings, exactly as b10621 does.
    let stop_sequences = match overrides.stop_sequences {
        Some(request_stops) if !request_stops.is_empty() => Some(request_stops),
        _ if !config.default_stop_sequences.is_empty() => {
            Some(config.default_stop_sequences.clone())
        }
        other => other,
    };

    // b10621 breaker precedence (#1485): native strings replace everything;
    // an id-based (OpenAI-shaped) override runs with exactly those ids and
    // no string set; otherwise the server-wide default strings apply.
    let dry_breaker_strings = match (
        overrides.dry_sequence_breaker_strings,
        overrides.dry_sequence_breakers.is_some(),
    ) {
        (Some(strings), _) => Some(strings),
        (None, true) => None,
        (None, false) => Some(live.default_dry_sequence_breakers.clone()),
    };

    // b10621 logit-bias precedence (#1485): a request field replaces the
    // server-wide `--logit-bias` set wholesale (upstream's handler clears
    // and rebuilds the list); absent falls back to it.
    let logit_bias = overrides
        .logit_bias
        .unwrap_or_else(|| config.default_logit_bias.clone());

    // b10621 n_probs (#1485): per-token probability reporting through the
    // shared logprobs machinery, on the source the request selected.
    let logprobs = match overrides.n_probs {
        Some(n) if n > 0 => LogprobsConfig {
            enabled: true,
            top_k: n,
            source: if overrides.post_sampling_probs.unwrap_or(false) {
                LogprobSource::PostSampling
            } else {
                LogprobSource::RawModel
            },
        },
        _ => LogprobsConfig::default(),
    };

    ServerGenerateOptions {
        retention: Default::default(),
        max_tokens: resolve_server_max_tokens_with_live(config, live, overrides.max_tokens),
        sampling,
        stop_sequences,
        dry_breaker_strings,
        logit_bias,
        logit_bias_texts: overrides.logit_bias_texts,
        post_sampling_probs: overrides.post_sampling_probs.unwrap_or(false),
        // b10621 --ignore-eos (#1436): a per-request ignore_eos overrides
        // the server-wide default; absent falls back to it.
        ignore_eos: overrides.ignore_eos.unwrap_or(config.default_ignore_eos),
        priority: overrides.priority,
        logprobs,
        reasoning_budget: overrides.reasoning_budget,
        thinking_enter_block_on_start: overrides.thinking_enter_block_on_start,
        // Default unarmed; routes that accept the b10621 `reasoning_control`
        // field replace this with the shared force flag they registered in
        // the completion-control registry (#1444).
        reasoning_control: None,
        // Default unset; chat routes populate this when the prompt cache
        // store is installed (see `src/server/routes/chat.rs`).
        prompt_cache_ctx: None,
        // defaults to `None` (unconstrained generation). Routes
        // that handle `response_format` populate this after the helper
        // returns — see `chat.rs` and `completions.rs`.
        structured: None,
        grammar: None,
        // Default unset (use the checkpoint's configured Gemma 4 budget). The
        // chat routes populate this from the resolved `image_url` content
        // parts after the helper returns; the raw text-completion endpoints
        // carry no image parts and always leave it `None`.
        image_soft_tokens: None,
        // Snapshot the server-default adapter scales at admission (#1439);
        // the native-completion route replaces this when the request carries
        // its own `lora` field.
        lora_scales: config
            .lora_runtime
            .as_ref()
            .map(|set| std::sync::Arc::new(set.server_scales())),
    }
}

/// Resolve the generation budget shared by every server route.
///
/// Explicit client budgets are silently clamped to the effective per-slot
/// context window when `--ctx-size` supplied one. When the context size is
/// model-derived (`config.context_size == 0`), the startup-resolved server
/// default is the cap; that default comes from the checkpoint context window
/// for the `-n -1` sentinel, with 4096 as the final fallback. An omitted
/// request budget keeps the configured server default unchanged.
///
/// Under `--context-shift` (#1472) the per-slot window no longer caps an
/// explicit budget: shifting exists for generation past the window
/// (b10621's "context shift on infinite text generation"), and the KV bound
/// is enforced by the shift itself rather than by shrinking the request.
#[allow(dead_code)]
pub(crate) fn resolve_server_max_tokens(config: &ServerConfig, requested: Option<usize>) -> usize {
    resolve_server_max_tokens_with_live(config, &config.live_settings(), requested)
}

/// Resolve one request's generation budget from its captured live snapshot.
pub(crate) fn resolve_server_max_tokens_with_live(
    config: &ServerConfig,
    live: &LiveSettings,
    requested: Option<usize>,
) -> usize {
    let Some(requested) = requested else {
        return live.default_max_tokens;
    };
    if config.context_shift {
        return requested;
    }
    let cap = if config.context_size > 0 {
        config.context_size
    } else {
        live.default_max_tokens
    };
    requested.min(cap)
}

#[cfg(test)]
#[path = "request_options_tests.rs"]
mod tests;
