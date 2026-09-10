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

//! Muse Glimmer assistant drafter (`model_type: muse_glimmer_assistant`,
//! issue #1343): the DFlash-style block drafter Meta publishes for
//! `Muse-Glimmer-30B`, run on the existing DFlash round loop.
//!
//! What differs from the Qwen 3.5 DFlash drafter in this crate:
//!
//! - five sliding-attention layers with per-head `q_norm` / `k_norm`, under
//!   a bidirectional sliding-window mask over absolute positions
//!   ([`attention`]);
//! - the context K/V lives in a temporal window rather than a plain
//!   growing cache ([`cache`]), because the mask reads absolute positions;
//! - the encoder is `encoder.fc` plus `encoder.output_norm_enc`, the norms
//!   are plain RMSNorm, and the checkpoint keys carry no `model.` prefix
//!   ([`model`]);
//! - the target's RAW embedding table and untied `lm_head` are borrowed at
//!   bind time; the Muse target's own `embed_norm` and logit softcap are
//!   never applied to the drafter ([`drafter`]);
//! - the config is flat and carries neither `num_target_layers` nor
//!   `vocab_size`, both checked against the bound target instead
//!   ([`config`]).
//!
//! The first draft round consumes EVERY prompt row (the drafter's context
//! window holds the prompt), so the server pairs it with
//! `FirstHiddenRows::EveryPromptRow`, like the LFM2 DSpark drafter.

pub mod attention;
pub mod cache;
pub mod config;
pub mod drafter;
pub mod model;

pub use attention::{
    MuseAssistantAttention, bidirectional_sliding_mask, bidirectional_sliding_mask_bool,
};
pub use cache::MuseAssistantContextCache;
pub use config::{
    MUSE_ASSISTANT_ARCHITECTURE, MUSE_ASSISTANT_INITIAL_BLOCK_SIZE, MUSE_ASSISTANT_MODEL_TYPE,
    MuseAssistantConfig, is_muse_assistant_config, is_muse_assistant_dir,
    peek_muse_assistant_configured_block_size,
};
pub use drafter::MuseAssistantDrafter;
pub use model::{MuseAssistantDecoderLayer, MuseAssistantModel};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
