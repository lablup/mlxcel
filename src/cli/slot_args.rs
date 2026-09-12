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

//! Slot-state and context-checkpoint flag group (llama-server b10621, #1473).
//!
//! One canonical clap definition of the five b10621 options that configure
//! per-slot retained prompts and the per-slot context-checkpoint ring, so both
//! server binaries declare them identically. mlxcel has neither structure, so
//! most of them are `not_applicable`: the inert value is accepted, and a
//! request for the behavior is refused at startup with a diagnostic that names
//! what is missing rather than an unknown-argument error from clap. `--kv-unified`
//! is implemented as shared logical context-budget policy over mlxcel's existing
//! per-sequence KV allocation.
//!
//! The upstream definitions are in
//! [`common/arg.cpp`](https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp)
//! and the state they configure lives in
//! [`tools/server/server-context.cpp`](https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/server/server-context.cpp).
//!
//! # Why these five are one group
//!
//! They all address state mlxcel does not keep. Slot reuse in mlxcel goes
//! through a process-wide radix trie over token prefixes
//! (`src/server/prompt_cache/`) that is consulted for every request regardless
//! of which sequence last held the tokens, so there is no per-slot retained
//! prompt for `--cache-idle-slots` to save or for
//! `--slot-prompt-similarity` to compare against. KV is allocated per
//! sequence through the scheduler's cache pool, so `--kv-unified` is a shared
//! logical budget rather than a physical KV layout switch. And
//! `capture_history_boundary_snapshot` takes at most one snapshot per sequence,
//! at the prompt/generation boundary, so there is no ring for `--ctx-checkpoints` to size or for
//! `--checkpoint-min-step` to space.
//!
//! Used by: mlxcel serve, mlxcel-server.

use clap::Args;

/// Shared slot-state and checkpoint flag group.
#[derive(Args, Debug, Default, Clone)]
#[command(next_help_heading = "Slot State Options (llama-server compatibility)")]
pub struct SlotCompatArgs {
    /// Save a slot's retained prompt when it goes idle (refused: mlxcel keeps
    /// no per-slot prompt).
    #[arg(
        long = "cache-idle-slots",
        env = "LLAMA_ARG_CACHE_IDLE_SLOTS",
        overrides_with = "no_cache_idle_slots",
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        value_parser = clap::builder::BoolishValueParser::new(),
        action = clap::ArgAction::Set
    )]
    pub cache_idle_slots: Option<bool>,

    /// Do not save idle slots (llama-server `--no-cache-idle-slots`).
    #[arg(
        long = "no-cache-idle-slots",
        overrides_with = "cache_idle_slots",
        action = clap::ArgAction::SetTrue
    )]
    pub no_cache_idle_slots: bool,

    /// Similarity threshold for picking the slot whose prompt best matches a
    /// request (refused above 0: slots hold no prompt to compare).
    #[arg(long = "slot-prompt-similarity", value_name = "SIM")]
    pub slot_prompt_similarity: Option<f32>,

    /// Give every slot the whole context window and enforce one shared live
    /// token budget across all slots.
    #[arg(
        long = "kv-unified",
        env = "LLAMA_ARG_KV_UNIFIED",
        overrides_with = "no_kv_unified",
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        value_parser = clap::builder::BoolishValueParser::new(),
        action = clap::ArgAction::Set
    )]
    pub kv_unified: Option<bool>,

    /// Keep split per-slot context windows (llama-server `--no-kv-unified`).
    #[arg(
        long = "no-kv-unified",
        overrides_with = "kv_unified",
        action = clap::ArgAction::SetTrue
    )]
    pub no_kv_unified: bool,

    /// Size of the per-slot context-checkpoint ring (refused above 0: mlxcel
    /// captures at most one history-boundary snapshot per sequence).
    #[arg(
        long = "ctx-checkpoints",
        visible_alias = "swa-checkpoints",
        env = "LLAMA_ARG_CTX_CHECKPOINTS",
        value_name = "N"
    )]
    pub ctx_checkpoints: Option<i64>,

    /// Minimum token spacing between successive context checkpoints (refused
    /// above 0: there are no successive checkpoints to space).
    #[arg(
        long = "checkpoint-min-step",
        env = "LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT",
        value_name = "N"
    )]
    pub checkpoint_min_step: Option<i64>,
}

impl SlotCompatArgs {
    /// Refuse every value that asks for state mlxcel does not keep.
    ///
    /// Each inert value is accepted so a deployment script written against
    /// b10621 that turns the feature OFF keeps working; only a request for
    /// the behavior itself fails, and it fails with a diagnostic naming the
    /// missing structure.
    pub fn resolve(&self) -> Result<(), String> {
        if self.cache_idle_slots == Some(true) && !self.no_cache_idle_slots {
            return Err(
                "--cache-idle-slots asks the server to save a slot's retained prompt when the \
                 slot goes idle and restore it on the slot's next task. mlxcel keeps no \
                 per-slot prompt: reuse goes through a process-wide radix trie over token \
                 prefixes (src/server/prompt_cache/) that every request consults regardless of \
                 which sequence last held the tokens, so there is nothing slot-local to save or \
                 clear. The prompt cache already survives a slot going idle, which is the effect \
                 the flag buys upstream. Pass --no-cache-idle-slots, or drop the flag."
                    .to_string(),
            );
        }

        if let Some(similarity) = self.slot_prompt_similarity
            && similarity != 0.0
        {
            return Err(format!(
                "--slot-prompt-similarity {similarity} asks the server to pick the slot whose \
                 retained prompt is most similar to the request, above the given threshold. \
                 mlxcel's slots hold no prompt to compare: they are a per-request registry over \
                 a scheduler whose reuse is a process-wide prefix trie, consulted identically \
                 whichever slot serves the request. Upstream's own default is 0.10, so this \
                 refuses a script that passes that default rather than pretending a \
                 slot-selection policy exists. Pass --slot-prompt-similarity 0 to disable it \
                 explicitly, or drop the flag."
            ));
        }

        if let Some(checkpoints) = self.ctx_checkpoints
            && checkpoints != 0
        {
            return Err(format!(
                "--ctx-checkpoints {checkpoints} asks for a ring of per-slot context \
                 checkpoints to roll back to. mlxcel captures at most one snapshot per \
                 sequence, at the prompt/generation boundary \
                 (capture_history_boundary_snapshot), and has no ring to size. Pass \
                 --ctx-checkpoints 0, or drop the flag."
            ));
        }

        if let Some(spacing) = self.checkpoint_min_step
            && spacing != 0
        {
            return Err(format!(
                "--checkpoint-min-step {spacing} sets the minimum token spacing between \
                 successive context checkpoints. mlxcel has no checkpoint ring (see \
                 --ctx-checkpoints), so there are no successive checkpoints for a spacing to \
                 separate. Pass --checkpoint-min-step 0, or drop the flag."
            ));
        }

        Ok(())
    }

    /// Resolve llama-server b10621's unified context-budget default.
    ///
    /// Upstream enables `kv_unified` when `--parallel` is left on auto. mlxcel
    /// matches the visible contract by giving each slot the whole context
    /// window and enforcing a shared live-token budget in the scheduler. An
    /// explicit `--parallel N` keeps split windows unless `--kv-unified` is
    /// also explicit. An explicit `--max-batch-size` keeps split windows too,
    /// because it is an operator-supplied concurrency divisor.
    pub fn resolve_kv_unified(&self, parallel_auto: bool, max_batch_size_was_set: bool) -> bool {
        if self.no_kv_unified {
            return false;
        }

        if let Some(kv_unified) = self.kv_unified {
            return kv_unified;
        }

        parallel_auto && !max_batch_size_was_set
    }
}

#[cfg(test)]
#[path = "slot_args_tests.rs"]
mod tests;
