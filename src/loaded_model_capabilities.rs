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

//! Capability-facing helpers for `LoadedModel`.
//!
//! `src/loaded_model.rs` keeps explicit model storage and `LanguageModel`
//! forwarding. This module owns the narrower multimodal surfaces consumed by
//! CLI/server control-plane code so those callers do not need to know concrete
//! model variant names.

use crate::loaded_model::LoadedModel;
use crate::{qwen_vl, vision, vlm_prompt};

/// Capability-oriented references used by CLI/server multimodal preparation.
///
/// The control plane should ask for the narrowest VLM runtime capability it
/// needs instead of depending on concrete `LoadedModel` variants. Add new
/// variants here only when a family truly needs distinct preparation logic.
pub enum VlmRuntimeRef<'a> {
    Qwen(&'a dyn qwen_vl::QwenVlRuntime),
    MiniCPMO(&'a vision::MiniCPMOVLModel),
    MiniCPMV46(&'a vision::MiniCPMV46VLModel),
    Moondream3(&'a vision::Moondream3VLModel),
    Moondream2(&'a vision::Moondream2VLModel),
    Gemma3n(&'a vision::Gemma3nVLModel),
    Gemma4(&'a vision::Gemma4VLModel),
    Gemma4Unified(&'a vision::Gemma4UnifiedModel),
    Phi4MM(&'a vision::Phi4MMVLModel),
    Phi4SigLip(&'a vision::Phi4SigLipVLModel),
    Phi3V(&'a vision::Phi3VLModel),
    Molmo(&'a vision::MolmoVLModel),
    Molmo2(&'a vision::Molmo2VLModel),
    MolmoPoint(&'a vision::MolmoPointVLModel),
    /// Nemotron H Nano Omni vision runtime (vision-only scope).
    NemotronHNanoOmni(&'a vision::NemotronHNanoOmniVlModel),
    /// PaddleOCR-VL runtime (NaViT vision + ERNIE-4.5 MRoPE text).
    PaddleOcr(&'a vision::PaddleOcrVlModel),
    /// Youtu-VL runtime.
    YoutuVL(&'a vision::YoutuVLModel),
    /// InternVL (internvl_chat) runtime.
    InternVL(&'a vision::InternVLChatVLM),
    /// Kimi-VL / Kimi-VL 2.5 (MoonViT) runtime.
    KimiVL(&'a vision::KimiVLModel),
    /// Llama 3.2 Vision (mllama) cross-attention runtime.
    Mllama(&'a vision::MllamaVLModel),
    /// SmolVLM / SmolVLM2 (smolvlm) runtime.
    SmolVLM(&'a vision::SmolVLMModel),
    /// Idefics2 (idefics2) runtime.
    Idefics2(&'a vision::Idefics2Model),
    /// LFM2-VL (lfm2_vl) runtime.
    Lfm2Vl(&'a vision::Lfm2VlModel),
    GraniteVision(&'a vision::GraniteVisionVLModel),
    Granite4Vision(&'a vision::Granite4VisionVLModel),
    DeepSeekOcr(&'a vision::deepseekocr::DeepSeekOcrVlModel),
    Standard(&'a vision::VisionModule),
}

pub(crate) fn standard_image_token_block_info(
    vm: &vision::VisionModule,
) -> vlm_prompt::ImageTokenBlockInfo {
    vlm_prompt::ImageTokenBlockInfo {
        use_boi_eoi: vm.boi_token_id != 0,
        image_token_id: vm.image_token_id,
        mm_tokens_per_image: vm.mm_tokens_per_image,
        boi_token_id: vm.boi_token_id,
        eoi_token_id: vm.eoi_token_id,
        has_bos: vm.has_bos,
        separator_token_id: vm.separator_token_id,
        suffix_tokens: vm.suffix_tokens.clone(),
        block_prefix_tokens: vm.block_prefix_tokens.clone(),
        block_suffix_tokens: vm.block_suffix_tokens.clone(),
    }
}

fn gemma3n_image_token_block_info(
    model: &vision::Gemma3nVLModel,
) -> vlm_prompt::ImageTokenBlockInfo {
    vlm_prompt::ImageTokenBlockInfo {
        use_boi_eoi: true,
        image_token_id: model.image_token_id,
        mm_tokens_per_image: 256,
        boi_token_id: model.boi_token_id,
        eoi_token_id: model.eoi_token_id,
        has_bos: true,
        separator_token_id: None,
        suffix_tokens: Vec::new(),
        block_prefix_tokens: Vec::new(),
        block_suffix_tokens: Vec::new(),
    }
}

fn qwen_runtime(runtime: VlmRuntimeRef<'_>) -> Option<&dyn qwen_vl::QwenVlRuntime> {
    match runtime {
        VlmRuntimeRef::Qwen(runtime) => Some(runtime),
        _ => None,
    }
}

pub(crate) fn vision_module_from_runtime(
    runtime: VlmRuntimeRef<'_>,
) -> Option<&vision::VisionModule> {
    match runtime {
        VlmRuntimeRef::Standard(vision) => Some(vision),
        _ => None,
    }
}

pub(crate) fn image_token_block_info_from_runtime(
    runtime: VlmRuntimeRef<'_>,
) -> Option<vlm_prompt::ImageTokenBlockInfo> {
    match runtime {
        VlmRuntimeRef::Gemma3n(model) => Some(gemma3n_image_token_block_info(model)),
        VlmRuntimeRef::Standard(vision) => Some(standard_image_token_block_info(vision)),
        _ => None,
    }
}

impl LoadedModel {
    /// Check if this model is a vision-language model.
    pub fn is_vlm(&self) -> bool {
        self.vlm_runtime().is_some()
    }

    /// Get the vision module if this is a standard `VisionModule`-backed VLM.
    pub fn vision_module(&self) -> Option<&vision::VisionModule> {
        vision_module_from_runtime(self.vlm_runtime()?)
    }

    /// Return the multimodal runtime capability needed by prompt/image prep.
    ///
    /// Keep this as the single VLM switchboard so new variants do not require
    /// ad hoc getter methods throughout CLI or server code.
    pub fn vlm_runtime(&self) -> Option<VlmRuntimeRef<'_>> {
        match self {
            Self::Qwen2VL(model) => Some(VlmRuntimeRef::Qwen(model)),
            Self::Qwen25VL(model) => Some(VlmRuntimeRef::Qwen(model)),
            Self::Qwen3VL(model) => Some(VlmRuntimeRef::Qwen(model)),
            Self::Qwen3VLMoe(model) => Some(VlmRuntimeRef::Qwen(model)),
            Self::Glm4v(model) => Some(VlmRuntimeRef::Qwen(model)),
            Self::Glm4vMoe(model) => Some(VlmRuntimeRef::Qwen(model)),
            Self::GlmOcr(model) => Some(VlmRuntimeRef::Qwen(model)),
            Self::Qwen35VLM(model) | Self::Qwen35MoeVLM(model) => Some(VlmRuntimeRef::Qwen(model)),
            Self::PaddleOcrVL(model) => Some(VlmRuntimeRef::PaddleOcr(model)),
            Self::MiniCPMOVLM(model) => Some(VlmRuntimeRef::MiniCPMO(model)),
            Self::MiniCPMV46VLM(model) => Some(VlmRuntimeRef::MiniCPMV46(model)),
            Self::Moondream3VLM(model) => Some(VlmRuntimeRef::Moondream3(model)),
            Self::Moondream2VLM(model) => Some(VlmRuntimeRef::Moondream2(model)),
            Self::Gemma3nVLM(model) => Some(VlmRuntimeRef::Gemma3n(model)),
            Self::Gemma4VLM(model) => Some(VlmRuntimeRef::Gemma4(model)),
            Self::Gemma4Unified(model) => Some(VlmRuntimeRef::Gemma4Unified(model)),
            Self::Phi4MMVLM(model) => Some(VlmRuntimeRef::Phi4MM(model)),
            Self::Phi4SigLipVLM(model) => Some(VlmRuntimeRef::Phi4SigLip(model)),
            Self::Phi3VLM(model) => Some(VlmRuntimeRef::Phi3V(model)),
            Self::MolmoVLM(model) => Some(VlmRuntimeRef::Molmo(model)),
            Self::Molmo2VLM(model) => Some(VlmRuntimeRef::Molmo2(model)),
            Self::MolmoPointVLM(model) => Some(VlmRuntimeRef::MolmoPoint(model)),
            Self::NemotronHNanoOmniVLM(model) => Some(VlmRuntimeRef::NemotronHNanoOmni(model)),
            Self::YoutuVL(model) => Some(VlmRuntimeRef::YoutuVL(model)),
            Self::InternVLChatVLM(model) => Some(VlmRuntimeRef::InternVL(model)),
            Self::KimiVL(model) => Some(VlmRuntimeRef::KimiVL(model)),
            Self::MllamaVLM(model) => Some(VlmRuntimeRef::Mllama(model)),
            Self::SmolVLM(model) => Some(VlmRuntimeRef::SmolVLM(model)),
            Self::Idefics2(model) => Some(VlmRuntimeRef::Idefics2(model)),
            Self::Lfm2VL(model) => Some(VlmRuntimeRef::Lfm2Vl(model)),
            Self::Gemma3VLM(vlm) => Some(VlmRuntimeRef::Standard(&vlm.vision)),
            Self::Llama4VLM(vlm) => Some(VlmRuntimeRef::Standard(&vlm.vision)),
            Self::LlavaVLM(vlm) => Some(VlmRuntimeRef::Standard(&vlm.vision)),
            Self::GraniteVisionVLM(model) => Some(VlmRuntimeRef::GraniteVision(model)),
            Self::Granite4VisionVLM(model) => Some(VlmRuntimeRef::Granite4Vision(model)),
            Self::DeepSeekOcrVLM(model) => Some(VlmRuntimeRef::DeepSeekOcr(model)),
            _ => None,
        }
    }

    pub fn qwen_vl_prompt_info(&self) -> Option<qwen_vl::QwenVlmPromptInfo<'_>> {
        Some(qwen_runtime(self.vlm_runtime()?)?.prompt_info())
    }

    pub fn qwen_vl_input_embeddings(
        &self,
        input_ids: &mlxcel_core::MlxArray,
        pixel_values: &mlxcel_core::MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> Option<vision::merge::InputEmbeddings> {
        Some(qwen_runtime(self.vlm_runtime()?)?.input_embeddings(input_ids, pixel_values, grid_thw))
    }

    /// Bind any pending MRoPE state (the per-row deltas just set by a Qwen
    /// VL `get_input_embeddings` pass) to a specific server sequence id so
    /// the cached scalar can no longer leak across requests.
    ///
    /// This is a no-op for non-Qwen-VL models. The caller passes the id
    /// the scheduler already allocated for this request — the VLM runtime
    /// then transfers the fallback slot's MRoPE state into its
    /// per-`SequenceId` map.
    pub fn bind_qwen_vl_mrope_state_to_sequence(&self, seq_id: mlxcel_core::cache::SequenceId) {
        if let Some(runtime) = self.vlm_runtime()
            && let Some(qwen) = qwen_runtime(runtime)
        {
            qwen.bind_mrope_state_to_sequence(seq_id);
        }
    }

    /// Take the per-sequence MRoPE entry under `seq_id` out of the
    /// underlying Qwen VL text model. Used by the server preemption
    /// path so the entry survives the eviction (which releases the old
    /// sequence id) and can be reinstalled under the freshly allocated
    /// id (follow-up).
    ///
    /// Returns an empty snapshot for non-Qwen-VL models or when no
    /// entry exists for `seq_id`.
    pub fn take_qwen_vl_mrope_entry(
        &self,
        seq_id: mlxcel_core::cache::SequenceId,
    ) -> qwen_vl::QwenVlMRopeSnapshot {
        if let Some(runtime) = self.vlm_runtime()
            && let Some(qwen) = qwen_runtime(runtime)
        {
            return qwen.take_mrope_entry_for_sequence(seq_id);
        }
        qwen_vl::QwenVlMRopeSnapshot(None)
    }

    /// Re-install a previously taken Qwen VL MRoPE entry under
    /// `seq_id`. No-op for non-Qwen-VL models or when the snapshot is
    /// empty.
    pub fn install_qwen_vl_mrope_entry(
        &self,
        seq_id: mlxcel_core::cache::SequenceId,
        snapshot: qwen_vl::QwenVlMRopeSnapshot,
    ) {
        if snapshot.is_empty() {
            return;
        }
        if let Some(runtime) = self.vlm_runtime()
            && let Some(qwen) = qwen_runtime(runtime)
        {
            qwen.install_mrope_entry_for_sequence(seq_id, snapshot);
        }
    }

    /// Bind any pending Gemma 4 VLM `per_layer_inputs` tensor (just
    /// projected by `get_input_embeddings_with_audio_and_cache`) to a
    /// specific server sequence id so a burst of Gemma 4 VLM requests
    /// in a single drain tick cannot have one row's prefill consume
    /// another row's tensor.
    ///
    /// This is a no-op for non-Gemma-4 models. The caller passes the
    /// id the scheduler already allocated for the request; the VLM
    /// runtime transfers the fallback slot's tensor into its
    /// per-`SequenceId` map. When the fallback is empty (E1B variant
    /// or text-only request after a Gemma 4 VLM model load), the
    /// binding leaves the map without an entry — the prefill consumer
    /// then surfaces `None` for `per_layer_inputs`, matching the
    /// no-VLM-prefill semantics.
    pub fn bind_gemma4_per_layer_inputs_to_sequence(&self, seq_id: mlxcel_core::cache::SequenceId) {
        match self.vlm_runtime() {
            Some(VlmRuntimeRef::Gemma4(gemma4)) => {
                gemma4.bind_per_layer_inputs_to_sequence(seq_id);
            }
            Some(VlmRuntimeRef::Gemma4Unified(unified)) => {
                unified.bind_per_layer_inputs_to_sequence(seq_id);
            }
            _ => {}
        }
    }

    /// Take the per-sequence Gemma 4 `per_layer_inputs` tensor out of
    /// the underlying VL model. Used by the scheduler's preemption
    /// path so the tensor survives the eviction (which releases the
    /// old sequence id) and can be reinstalled under the freshly
    /// allocated id — `prepare_request_vlm_embeddings` does not run
    /// again on re-prefill, so without the take/install round trip
    /// the re-prefill would observe `per_layer_inputs == None` and
    /// produce wrong logits for E2B/E4B variants.
    ///
    /// Returns `None` for non-Gemma-4 models, the E1B variant, or
    /// when no entry exists for `seq_id`.
    pub fn take_gemma4_per_layer_inputs_entry(
        &self,
        seq_id: mlxcel_core::cache::SequenceId,
    ) -> Option<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>> {
        match self.vlm_runtime() {
            Some(VlmRuntimeRef::Gemma4(gemma4)) => {
                gemma4.take_per_layer_inputs_for_sequence(seq_id)
            }
            Some(VlmRuntimeRef::Gemma4Unified(unified)) => {
                unified.take_per_layer_inputs_for_sequence(seq_id)
            }
            _ => None,
        }
    }

    /// Re-install a previously taken Gemma 4 `per_layer_inputs`
    /// tensor under `seq_id`. No-op for non-Gemma-4 models or when
    /// the snapshot is `None`.
    pub fn install_gemma4_per_layer_inputs_entry(
        &self,
        seq_id: mlxcel_core::cache::SequenceId,
        snapshot: Option<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>>,
    ) {
        if snapshot.is_none() {
            return;
        }
        match self.vlm_runtime() {
            Some(VlmRuntimeRef::Gemma4(gemma4)) => {
                gemma4.install_per_layer_inputs_for_sequence(seq_id, snapshot);
            }
            Some(VlmRuntimeRef::Gemma4Unified(unified)) => {
                unified.install_per_layer_inputs_for_sequence(seq_id, snapshot);
            }
            _ => {}
        }
    }

    /// Drain the freshly written Gemma 3n VLM `per_layer_inputs` fallback
    /// slot into the per-`SequenceId` map under `seq_id` (issue #85).
    ///
    /// This is a no-op for non-Gemma-3n models. The caller passes the id
    /// the scheduler already allocated for the request; the VLM runtime
    /// transfers the fallback slot's tensor into its per-`SequenceId`
    /// map. When the fallback is empty (text-only request after a
    /// Gemma 3n VLM model load), the binding leaves the map without an
    /// entry — the prefill consumer then surfaces `None` for
    /// `per_layer_inputs`, matching the no-VLM-prefill semantics.
    pub fn bind_gemma3n_per_layer_inputs_to_sequence(
        &self,
        seq_id: mlxcel_core::cache::SequenceId,
    ) {
        if let Some(VlmRuntimeRef::Gemma3n(gemma3n)) = self.vlm_runtime() {
            gemma3n.bind_per_layer_inputs_to_sequence(seq_id);
        }
    }

    /// Take the per-sequence Gemma 3n `per_layer_inputs` tensor out of
    /// the underlying VL model. Used by the scheduler's preemption path
    /// so the tensor survives the eviction (which releases the old
    /// sequence id) and can be reinstalled under the freshly allocated
    /// id — `prepare_request_vlm_embeddings` does not run again on
    /// re-prefill, so without the take/install round trip the re-prefill
    /// would observe `per_layer_inputs == None` and panic in
    /// `Gemma3nVLModel::forward_with_embeddings_and_sequence_id`.
    ///
    /// Returns `None` for non-Gemma-3n models or when no entry exists
    /// for `seq_id`.
    pub fn take_gemma3n_per_layer_inputs_entry(
        &self,
        seq_id: mlxcel_core::cache::SequenceId,
    ) -> Option<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>> {
        if let Some(VlmRuntimeRef::Gemma3n(gemma3n)) = self.vlm_runtime() {
            return gemma3n.take_per_layer_inputs_for_sequence(seq_id);
        }
        None
    }

    /// Re-install a previously taken Gemma 3n `per_layer_inputs` tensor
    /// under `seq_id`. No-op for non-Gemma-3n models or when the
    /// snapshot is `None`.
    pub fn install_gemma3n_per_layer_inputs_entry(
        &self,
        seq_id: mlxcel_core::cache::SequenceId,
        snapshot: Option<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>>,
    ) {
        if snapshot.is_none() {
            return;
        }
        if let Some(VlmRuntimeRef::Gemma3n(gemma3n)) = self.vlm_runtime() {
            gemma3n.install_per_layer_inputs_for_sequence(seq_id, snapshot);
        }
    }

    pub fn image_token_block_info(&self) -> Option<vlm_prompt::ImageTokenBlockInfo> {
        image_token_block_info_from_runtime(self.vlm_runtime()?)
    }
}

#[cfg(test)]
#[path = "loaded_model_tests.rs"]
mod tests;
