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

//! Stage-local execution for pipeline-parallel inference.
//!
//! This module bridges the existing pipeline partition and partial-loading
//! helpers to a real execution seam:
//! - build a stage-local model from a [`StageAssignment`]
//! - run only the assigned layer range
//! - accept either token IDs (entry stage) or hidden states (middle/last stage)
//! - return either hidden states (non-final stage) or logits (final stage)
//!
//! The runtime and wire protocol stay backend-agnostic. Family-specific stage
//! loaders plug into [`LoadedStageExecutor`] via [`FamilyStageExecutor`].
//!
//! Used by: pipeline worker loop, CLI pipeline runtime, server pipeline runtime

mod common;
mod deepseek_v3;
mod gemma3;
mod gemma4;
mod glm4;
mod glm_moe_dsa;
mod gpt_oss;
mod jamba;
mod llama;
mod llama4;
mod mistral;
mod mixtral;
mod nemotron_h;
mod qwen3;
mod qwen35;
mod qwen3_next;

#[cfg(test)]
#[path = "family_registry_tests.rs"]
mod family_registry_tests;

use std::path::Path;

use anyhow::{Result, bail, ensure};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::{ModelType, get_model_type, sanitize_config_json};

use super::partial_loading::LayerFilter;
use super::partition::StageAssignment;
use deepseek_v3::DeepSeekV3StageExecutor;
use gemma3::Gemma3StageExecutor;
use gemma4::Gemma4StageExecutor;
use glm_moe_dsa::GlmMoeDsaStageExecutor;
use glm4::{Glm4MoeLiteStageExecutor, Glm4MoeStageExecutor, Glm4StageExecutor};
use gpt_oss::GptOssStageExecutor;
use jamba::JambaStageExecutor;
use llama::LlamaStageExecutor;
use llama4::Llama4StageExecutor;
use mistral::MistralStageExecutor;
use mixtral::MixtralStageExecutor;
use nemotron_h::NemotronHStageExecutor;
use qwen3::Qwen3StageExecutor;
use qwen3_next::Qwen3NextStageExecutor;
use qwen35::Qwen35StageExecutor;

/// Input payload for a single stage-local forward.
pub enum StageExecutionInput<'a> {
    /// Entry-stage execution from token IDs.
    TokenIds(&'a MlxArray),
    /// Middle/last-stage execution from upstream hidden states.
    HiddenStates(&'a MlxArray),
}

/// Output payload from a stage-local forward.
pub enum StageExecutionOutput {
    /// Hidden states to be forwarded to the next stage.
    HiddenStates(UniquePtr<MlxArray>),
    /// Final logits produced by the last stage.
    Logits(UniquePtr<MlxArray>),
}

impl StageExecutionOutput {
    pub fn into_hidden_states(self) -> Result<UniquePtr<MlxArray>> {
        match self {
            Self::HiddenStates(hidden) => Ok(hidden),
            Self::Logits(_) => bail!("expected hidden states but stage produced logits"),
        }
    }

    pub fn into_logits(self) -> Result<UniquePtr<MlxArray>> {
        match self {
            Self::Logits(logits) => Ok(logits),
            Self::HiddenStates(_) => bail!("expected logits but stage produced hidden states"),
        }
    }
}

/// Common execution interface for one pipeline stage.
pub trait StageExecutor {
    fn stage_assignment(&self) -> &StageAssignment;
    fn layer_filter(&self) -> &LayerFilter;
    fn make_caches(&self) -> Vec<KVCache>;
    fn release_caches(&self, caches: &[KVCache]);
    fn execute(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput>;
}

pub trait FamilyStageExecutor {
    fn make_caches(&self) -> Vec<KVCache>;
    fn release_caches(&self, _caches: &[KVCache]) {}
    fn execute(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput>;
}

/// A stage-local executor loaded from model weights.
pub struct LoadedStageExecutor {
    stage: StageAssignment,
    filter: LayerFilter,
    backend: Box<dyn FamilyStageExecutor>,
}

impl LoadedStageExecutor {
    /// Load a stage-local executor from a model directory and assignment.
    ///
    /// This constructor deliberately routes through the existing partial-load
    /// filter so later phases can replace the "load then filter" path with
    /// shard-selective I/O without changing the runtime API.
    pub fn load(model_dir: &Path, stage: &StageAssignment) -> Result<Self> {
        Self::load_with_adapter(model_dir, stage, None)
    }

    /// Load a stage-local executor optionally composing a LoRA adapter.
    ///
    /// When `adapter_path` is `Some`, the family stage executor loads its
    /// base weights, applies only the adapter tensors inside the stage's
    /// layer range, and then constructs the stage model from the fused
    /// weights. This is the pipeline-parallel counterpart of the non-PP
    /// `load_model_with_adapter` entry point — both paths share the same
    /// adapter file format (`adapter_config.json` plus
    /// `adapters.safetensors` / `adapter_model.safetensors`) and the same
    /// rank/scaling semantics.
    pub fn load_with_adapter(
        model_dir: &Path,
        stage: &StageAssignment,
        adapter_path: Option<&Path>,
    ) -> Result<Self> {
        ensure!(
            stage.layer_range.end > stage.layer_range.start,
            "stage {} has an empty layer range",
            stage.stage_index
        );

        // Issue #688 (M2 hardening), extended to DeepSeek-V2 by issue #824:
        // defense-in-depth CUDA-graph disable for a hazard-family pipeline stage,
        // mirroring the non-PP backend-seam load sites.
        // Both the in-process and the remote-process pipeline loaders reach this
        // common stage-load chokepoint before any stage weights are realised. The
        // authoritative env write already happens on the main startup thread in
        // `start_server`; this idempotent call (guarded by `var_os`) only takes
        // effect if a future pipeline entry ever bypasses that hoist, so the env is
        // set before the stage's first GPU eval regardless.
        crate::loading::maybe_disable_cuda_graphs_for_model_for_path(model_dir);

        let filter = LayerFilter::from_stage(stage);
        let family = resolve_stage_family(model_dir)?;
        let backend =
            load_family_backend(model_dir, &filter, stage.stage_index, family, adapter_path)?;

        Ok(Self {
            stage: stage.clone(),
            filter,
            backend,
        })
    }
}

impl StageExecutor for LoadedStageExecutor {
    fn stage_assignment(&self) -> &StageAssignment {
        &self.stage
    }

    fn layer_filter(&self) -> &LayerFilter {
        &self.filter
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.backend.make_caches()
    }

    fn release_caches(&self, caches: &[KVCache]) {
        self.backend.release_caches(caches);
    }

    fn execute(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput> {
        self.backend.execute(input, caches, mask)
    }
}

/// Pipeline-parallel family identifier used by the stage executor registry
/// and by the server's capability negotiation.
///
/// A single [`ModelType`] may map to more than one [`StageFamily`] (today
/// `ModelType::Llama` covers both the base Llama branch and the `mistral`
/// config variant). The reverse is also allowed — a family may accept
/// several model types if they share a stage loader.
///
/// Every family carries a stable textual [`Self::name`] that is the
/// on-the-wire identifier during cross-version cluster handshakes.
/// Operators will see mismatches as `pipeline capability mismatch` errors
/// rather than silent activation corruption.
///
/// Used by: `LoadedStageExecutor`, server capability negotiation, CLI
/// `mlxcel generate --pp-size N` runtime startup
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StageFamily {
    Llama,
    Mistral,
    Mixtral,
    DeepSeekV3,
    Llama4,
    GptOss,
    Gemma3,
    Gemma4,
    Gemma4Vlm,
    Glm4,
    Glm4Moe,
    Glm4MoeLite,
    GlmMoeDsa,
    Qwen3,
    Qwen3Next,
    Qwen35,
    Qwen35Vlm,
    Qwen35Moe,
    Qwen35MoeVlm,
    Jamba,
    NemotronH,
}

impl StageFamily {
    /// Stable textual name used for cluster-side capability negotiation.
    /// Do not change these strings without bumping the pipeline capability
    /// protocol version; clusters mix stages across mlxcel revisions and
    /// rely on these strings staying byte-identical.
    pub fn name(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::Mixtral => "mixtral",
            Self::DeepSeekV3 => "deepseek_v3",
            Self::Llama4 => "llama4",
            Self::GptOss => "gpt_oss",
            Self::Gemma3 => "gemma3",
            Self::Gemma4 => "gemma4",
            Self::Gemma4Vlm => "gemma4_vlm",
            Self::Glm4 => "glm4",
            Self::Glm4Moe => "glm4_moe",
            Self::Glm4MoeLite => "glm4_moe_lite",
            Self::GlmMoeDsa => "glm_moe_dsa",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Next => "qwen3_next",
            Self::Qwen35 => "qwen3_5",
            Self::Qwen35Vlm => "qwen3_5_vlm",
            Self::Qwen35Moe => "qwen3_5_moe",
            Self::Qwen35MoeVlm => "qwen3_5_moe_vlm",
            Self::Jamba => "jamba",
            Self::NemotronH => "nemotron_h",
        }
    }
}

/// The full list of families supported by this binary's stage-executor
/// registry.
///
/// `mlxcel-server` advertises this list during cluster handshake so a
/// coordinator and its stage workers can detect version skew before
/// serving traffic. Clusters whose union of family support is smaller
/// than the coordinator's declared model family will refuse to start,
/// rather than silently routing an unsupported model to a worker that
/// would `bail!` at load time.
///
/// The returned slice is sorted by the textual [`StageFamily::name`] so
/// handshake payloads are byte-identical across reruns with the same
/// build.
///
/// Used by: `mlxcel-server` capability negotiation, integration tests
pub fn supported_families() -> &'static [StageFamily] {
    // Compile-time constant avoids heap allocations on every handshake.
    // The list must stay sorted by `StageFamily::name` — verified by the
    // `supported_families_is_sorted_by_name` test in
    // `family_registry_tests.rs`.
    const FAMILIES: &[StageFamily] = &[
        StageFamily::DeepSeekV3,
        StageFamily::Gemma3,
        StageFamily::Gemma4,
        StageFamily::Gemma4Vlm,
        StageFamily::Glm4,
        StageFamily::Glm4Moe,
        StageFamily::Glm4MoeLite,
        StageFamily::GlmMoeDsa,
        StageFamily::GptOss,
        StageFamily::Jamba,
        StageFamily::Llama,
        StageFamily::Llama4,
        StageFamily::Mistral,
        StageFamily::Mixtral,
        StageFamily::NemotronH,
        StageFamily::Qwen3,
        StageFamily::Qwen35,
        StageFamily::Qwen35Moe,
        StageFamily::Qwen35MoeVlm,
        StageFamily::Qwen35Vlm,
        StageFamily::Qwen3Next,
    ];
    FAMILIES
}

/// Decide which [`StageFamily`] should execute a given on-disk model. Most
/// [`ModelType`] values map one-to-one; the exception is `ModelType::Llama`,
/// which covers both the Llama stack and the base Mistral config — we peek
/// at the raw `config.json` to disambiguate.
fn resolve_stage_family(model_dir: &Path) -> Result<StageFamily> {
    let model_type = get_model_type(model_dir)?;
    let family = match model_type {
        ModelType::Llama => {
            if read_raw_config_model_type(model_dir)? == Some("mistral".to_string()) {
                StageFamily::Mistral
            } else {
                StageFamily::Llama
            }
        }
        ModelType::Mixtral => StageFamily::Mixtral,
        ModelType::DeepSeekV3 => StageFamily::DeepSeekV3,
        ModelType::Llama4 | ModelType::Llama4VLM => StageFamily::Llama4,
        ModelType::GptOss => StageFamily::GptOss,
        ModelType::Gemma3 => StageFamily::Gemma3,
        ModelType::Gemma4 => StageFamily::Gemma4,
        ModelType::Gemma4VLM => StageFamily::Gemma4Vlm,
        ModelType::Glm4 => StageFamily::Glm4,
        ModelType::Glm4Moe => StageFamily::Glm4Moe,
        ModelType::Glm4MoeLite => StageFamily::Glm4MoeLite,
        ModelType::GlmMoeDsa => StageFamily::GlmMoeDsa,
        ModelType::Qwen3 => StageFamily::Qwen3,
        ModelType::Qwen3Next => StageFamily::Qwen3Next,
        ModelType::Qwen35 => StageFamily::Qwen35,
        ModelType::Qwen35VLM => StageFamily::Qwen35Vlm,
        ModelType::Qwen35Moe => StageFamily::Qwen35Moe,
        ModelType::Qwen35MoeVLM => StageFamily::Qwen35MoeVlm,
        ModelType::Jamba => StageFamily::Jamba,
        ModelType::NemotronH => StageFamily::NemotronH,
        other => bail!(
            "pipeline stage executor is not implemented for model type {:?} yet",
            other
        ),
    };
    Ok(family)
}

fn read_raw_config_model_type(model_dir: &Path) -> Result<Option<String>> {
    let config_path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&config_path)?;
    let raw = sanitize_config_json(&raw);
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    Ok(value
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string()))
}

fn load_family_backend(
    model_dir: &Path,
    filter: &LayerFilter,
    stage_index: usize,
    family: StageFamily,
    adapter_path: Option<&Path>,
) -> Result<Box<dyn FamilyStageExecutor>> {
    match family {
        StageFamily::Llama => Ok(Box::new(LlamaStageExecutor::load(
            model_dir,
            filter,
            stage_index,
            adapter_path,
        )?)),
        StageFamily::Mistral => {
            ensure_no_adapter(adapter_path, family)?;
            Ok(Box::new(MistralStageExecutor::load(
                model_dir,
                filter,
                stage_index,
            )?))
        }
        StageFamily::Mixtral => {
            ensure_no_adapter(adapter_path, family)?;
            Ok(Box::new(MixtralStageExecutor::load(
                model_dir,
                filter,
                stage_index,
            )?))
        }
        StageFamily::DeepSeekV3 => {
            ensure_no_adapter(adapter_path, family)?;
            Ok(Box::new(DeepSeekV3StageExecutor::load(
                model_dir,
                filter,
                stage_index,
            )?))
        }
        StageFamily::Llama4 => {
            ensure_no_adapter(adapter_path, family)?;
            Ok(Box::new(Llama4StageExecutor::load(
                model_dir,
                filter,
                stage_index,
            )?))
        }
        StageFamily::GptOss => {
            ensure_no_adapter(adapter_path, family)?;
            Ok(Box::new(GptOssStageExecutor::load(
                model_dir,
                filter,
                stage_index,
            )?))
        }
        StageFamily::Gemma3 => {
            ensure_no_adapter(adapter_path, family)?;
            Ok(Box::new(Gemma3StageExecutor::load(
                model_dir,
                filter,
                stage_index,
            )?))
        }
        StageFamily::Gemma4 | StageFamily::Gemma4Vlm => {
            ensure_no_adapter(adapter_path, family)?;
            Ok(Box::new(Gemma4StageExecutor::load(
                model_dir,
                filter,
                stage_index,
            )?))
        }
        StageFamily::Glm4 => {
            ensure_no_adapter(adapter_path, family)?;
            Ok(Box::new(Glm4StageExecutor::load(
                model_dir,
                filter,
                stage_index,
            )?))
        }
        StageFamily::Glm4Moe => {
            ensure_no_adapter(adapter_path, family)?;
            Ok(Box::new(Glm4MoeStageExecutor::load(
                model_dir,
                filter,
                stage_index,
            )?))
        }
        StageFamily::Glm4MoeLite => {
            ensure_no_adapter(adapter_path, family)?;
            Ok(Box::new(Glm4MoeLiteStageExecutor::load(
                model_dir,
                filter,
                stage_index,
            )?))
        }
        StageFamily::GlmMoeDsa => {
            ensure_no_adapter(adapter_path, family)?;
            Ok(Box::new(GlmMoeDsaStageExecutor::load(
                model_dir,
                filter,
                stage_index,
            )?))
        }
        StageFamily::Qwen3 => {
            ensure_no_adapter(adapter_path, family)?;
            Ok(Box::new(Qwen3StageExecutor::load(
                model_dir,
                filter,
                stage_index,
            )?))
        }
        StageFamily::Qwen3Next => {
            ensure_no_adapter(adapter_path, family)?;
            Ok(Box::new(Qwen3NextStageExecutor::load(
                model_dir,
                filter,
                stage_index,
            )?))
        }
        StageFamily::Qwen35
        | StageFamily::Qwen35Vlm
        | StageFamily::Qwen35Moe
        | StageFamily::Qwen35MoeVlm => {
            ensure_no_adapter(adapter_path, family)?;
            Ok(Box::new(Qwen35StageExecutor::load(
                model_dir,
                filter,
                stage_index,
            )?))
        }
        StageFamily::Jamba => {
            ensure_no_adapter(adapter_path, family)?;
            Ok(Box::new(JambaStageExecutor::load(
                model_dir,
                filter,
                stage_index,
            )?))
        }
        StageFamily::NemotronH => {
            ensure_no_adapter(adapter_path, family)?;
            Ok(Box::new(NemotronHStageExecutor::load(
                model_dir,
                filter,
                stage_index,
            )?))
        }
    }
}

/// Guard that rejects an adapter path for stage families whose PP stage
/// executors have not yet been wired for LoRA composition.
///
/// Families opt into PP+LoRA by:
///   1. Accepting `adapter_path: Option<&Path>` in their `load` signature.
///   2. Calling `crate::lora::apply_stage_lora_adapter` between the weight
///      load and the `filter_weight_map` call.
///   3. Being removed from this guard.
///
/// Keeping this guard explicit means that adding a new family without
/// implementing the adapter path surfaces a loud error instead of silently
/// producing base-only outputs.
fn ensure_no_adapter(adapter_path: Option<&Path>, family: StageFamily) -> Result<()> {
    if let Some(path) = adapter_path {
        bail!(
            "pipeline-parallel LoRA composition is not yet implemented for stage family \
             '{}'; adapter path {} was provided but only the Llama family currently \
             supports PP+LoRA composition (v1 scope)",
            family.name(),
            path.display(),
        );
    }
    Ok(())
}
