use std::path::Path;

use crate::UniquePtr;
use crate::drafter::{Drafter, DrafterError, DrafterKind, SharedKv};
use crate::ffi::{self, MlxArray};
use crate::generate::{LanguageModel, ModelStateSnapshot, SamplingConfig};
use crate::inkling_layer::{
    InklingDecoderLayer, InklingDenseMlp, InklingLayerCache, InklingLayerSpec,
};
use crate::layers::{RMSNorm, UnifiedEmbedding, UnifiedLinear};
use crate::weights::{WeightMap, dir_has_tensor_name, load_weights_from_dir_index_filtered};

use super::config::InklingMtpConfig;
use super::sanitize::{is_inkling_mtp_tensor_name, sanitize_weights};

enum InklingLmHead {
    Linear(UnifiedLinear),
    Tied(UnifiedEmbedding),
}

struct InklingMtpBlock {
    embed_norm: RMSNorm,
    hidden_norm: RMSNorm,
    input_proj: UnifiedLinear,
    layer: InklingDecoderLayer<InklingDenseMlp>,
}

impl InklingMtpBlock {
    fn from_weights(
        weights: &WeightMap,
        index: usize,
        spec: &InklingLayerSpec,
    ) -> Result<Self, String> {
        let prefix = format!("blocks.{index}");
        let transformer = format!("{prefix}.transformer_block");
        let norm = |name: &str| {
            weights
                .get(&format!("{prefix}.{name}.weight"))
                .map(|value| ffi::copy(value))
                .ok_or_else(|| format!("Weight not found: {prefix}.{name}.weight"))
        };
        let mlp = InklingDenseMlp::from_weights(weights, &format!("{transformer}.mlp"), spec)?;
        Ok(Self {
            embed_norm: RMSNorm::new(norm("embed_norm")?, spec.rms_norm_eps),
            hidden_norm: RMSNorm::new(norm("hidden_norm")?, spec.rms_norm_eps),
            input_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.input_proj"),
                spec.quantization_group_size,
                spec.quantization_bits,
            )?,
            layer: InklingDecoderLayer::from_weights(weights, &transformer, spec, mlp)?,
        })
    }
}

/// Native Inkling chained-MTP drafter (B = 1).
pub struct InklingMtpDraftModel {
    config: InklingMtpConfig,
    blocks: Vec<InklingMtpBlock>,
    caches: Vec<InklingLayerCache>,
    target_embed: Option<UnifiedEmbedding>,
    target_norm: Option<RMSNorm>,
    lm_head: Option<InklingLmHead>,
    seed_token: Option<i32>,
    seed_hidden: Option<UniquePtr<MlxArray>>,
    round_snapshot: Option<ModelStateSnapshot>,
    round_appended: usize,
}

impl InklingMtpDraftModel {
    pub fn from_path(path: &Path) -> Result<Self, DrafterError> {
        let config = InklingMtpConfig::from_dir(path)?;
        let mut weights = load_weights_from_dir_index_filtered(path, |name| {
            is_inkling_mtp_tensor_name(name) || name.starts_with("blocks.")
        })
        .map_err(|reason| DrafterError::WeightLoad { reason })?;
        if weights.is_empty() {
            return Err(DrafterError::WeightLoad {
                reason: format!(
                    "{} has no model.mtp.layers.* tensors; an Inkling config flag alone is not a drafter",
                    path.display()
                ),
            });
        }
        sanitize_weights(&mut weights).map_err(|reason| DrafterError::WeightLoad { reason })?;
        crate::drafter::dflash::drafter::convert_bf16_to_f16_non_quantized(&mut weights);
        Self::from_weights(config, &weights)
    }

    pub fn from_weights(
        config: InklingMtpConfig,
        weights: &WeightMap,
    ) -> Result<Self, DrafterError> {
        let mut blocks = Vec::with_capacity(config.num_mtp_layers());
        for index in 0..config.num_mtp_layers() {
            let spec = config.layer_spec(index).map_err(DrafterError::Config)?;
            blocks.push(
                InklingMtpBlock::from_weights(weights, index, &spec)
                    .map_err(|reason| DrafterError::WeightLoad { reason })?,
            );
        }
        let caches = (0..blocks.len())
            .map(|_| InklingLayerCache::new())
            .collect();
        Ok(Self {
            config,
            blocks,
            caches,
            target_embed: None,
            target_norm: None,
            lm_head: None,
            seed_token: None,
            seed_hidden: None,
            round_snapshot: None,
            round_appended: 0,
        })
    }

    fn clear_runtime_state(&mut self) {
        self.caches = (0..self.blocks.len())
            .map(|_| InklingLayerCache::new())
            .collect();
        self.seed_token = None;
        self.seed_hidden = None;
        self.round_snapshot = None;
        self.round_appended = 0;
    }

    fn require_bound(&self) -> Result<(), DrafterError> {
        if self.target_embed.is_none() || self.target_norm.is_none() || self.lm_head.is_none() {
            Err(DrafterError::BindNotCalled)
        } else {
            Ok(())
        }
    }

    fn snapshot_caches(&self) -> ModelStateSnapshot {
        let token_len = self
            .caches
            .first()
            .map_or(0, |cache| cache.kv.offset.max(0) as usize);
        let mut snapshot = ModelStateSnapshot::new("inkling-mtp", token_len);
        for (index, cache) in self.caches.iter().enumerate() {
            cache.snapshot_into(&mut snapshot, &format!("block{index}"));
        }
        snapshot
    }

    fn restore_caches(&mut self, snapshot: &ModelStateSnapshot) -> Result<(), DrafterError> {
        if snapshot.family() != "inkling-mtp" {
            return Err(DrafterError::DraftFailed {
                reason: format!("cannot restore {} into Inkling MTP", snapshot.family()),
            });
        }
        let mut restored = (0..self.blocks.len())
            .map(|_| InklingLayerCache::new())
            .collect::<Vec<_>>();
        for (index, cache) in restored.iter_mut().enumerate() {
            cache
                .restore_from(snapshot, &format!("block{index}"))
                .map_err(|reason| DrafterError::DraftFailed { reason })?;
        }
        self.caches = restored;
        Ok(())
    }

    fn forward_block(
        &mut self,
        tokens: &[i32],
        hidden: &MlxArray,
        block_index: usize,
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        self.require_bound()?;
        let shape = ffi::array_shape(hidden);
        if shape.len() != 3 || shape[0] != 1 || shape[1] != tokens.len() as i32 {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "Inkling MTP block requires hidden [1, {}, H], got {shape:?}",
                    tokens.len()
                ),
            });
        }
        if shape[2] != self.config.hidden_size() as i32 {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "Inkling MTP hidden width {} does not match config {}",
                    shape[2],
                    self.config.hidden_size()
                ),
            });
        }
        let token_array = ffi::from_slice_i32(tokens, &[1, tokens.len() as i32]);
        let embed = self
            .target_embed
            .as_ref()
            .ok_or(DrafterError::BindNotCalled)?
            .forward(&token_array);
        let block = self
            .blocks
            .get(block_index)
            .ok_or_else(|| DrafterError::DraftFailed {
                reason: format!("Inkling MTP block {block_index} is out of range"),
            })?;
        let hidden = block.hidden_norm.forward(hidden);
        let embed = block.embed_norm.forward(&embed);
        let fused = crate::concatenate(&hidden, &embed, -1);
        let projected = block.input_proj.forward(&fused);
        let cache = self
            .caches
            .get_mut(block_index)
            .ok_or_else(|| DrafterError::DraftFailed {
                reason: format!("Inkling MTP cache {block_index} is out of range"),
            })?;
        Ok(block.layer.forward(&projected, cache))
    }

    fn project_logits(&self, hidden: &MlxArray) -> Result<UniquePtr<MlxArray>, DrafterError> {
        let normed = self
            .target_norm
            .as_ref()
            .ok_or(DrafterError::BindNotCalled)?
            .forward(hidden);
        let scaled = crate::divide_scalar(&normed, self.config.logits_mup_width_multiplier());
        let logits = match self.lm_head.as_ref().ok_or(DrafterError::BindNotCalled)? {
            InklingLmHead::Linear(head) => head.forward(&scaled),
            InklingLmHead::Tied(embed) => embed.as_linear(&scaled),
        };
        let shape = ffi::array_shape(&logits);
        let limit = self.config.unpadded_vocab_size() as i32;
        if shape.last().copied().unwrap_or(0) > limit {
            Ok(crate::utils::slice_axis(&logits, -1, 0, limit))
        } else {
            Ok(logits)
        }
    }

    fn sample_one(logits: &MlxArray, sampler: &SamplingConfig) -> i32 {
        let last = ffi::slice_last_logits(logits);
        let token = ffi::fused_sample(
            &last,
            sampler.temperature,
            sampler.top_k,
            sampler.top_p,
            sampler.min_p,
        );
        ffi::eval(&token);
        ffi::item_i32(&token)
    }

    fn set_seed_from_hidden(
        &mut self,
        hidden: &MlxArray,
        sampler: &SamplingConfig,
    ) -> Result<(), DrafterError> {
        let shape = ffi::array_shape(hidden);
        let last = shape[1] - 1;
        let hidden = ffi::slice(hidden, &[0, last, 0], &[shape[0], last + 1, shape[2]]);
        let logits = self.project_logits(&hidden)?;
        self.seed_token = Some(Self::sample_one(&logits, sampler));
        self.seed_hidden = Some(hidden);
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn forward_logits_for_test(
        &mut self,
        tokens: &[i32],
        hidden: &MlxArray,
        block_index: usize,
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        let output = self.forward_block(tokens, hidden, block_index)?;
        let logits = self.project_logits(&output)?;
        let shape = ffi::array_shape(&logits);
        Ok(ffi::slice(
            &logits,
            &[0, shape[1] - 1, 0],
            &[shape[0], shape[1], shape[2]],
        ))
    }
}

impl Drafter for InklingMtpDraftModel {
    fn bind(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        let embed = target
            .embed_tokens_module()
            .ok_or(DrafterError::TargetMissingFeature {
                feature: "embed_tokens_module",
            })?;
        let norm = target
            .final_norm_module()
            .ok_or(DrafterError::TargetMissingFeature {
                feature: "final_norm_module",
            })?;
        self.lm_head = Some(match target.lm_head_module() {
            Some(head) => InklingLmHead::Linear(head),
            None => InklingLmHead::Tied(embed.clone_shared()),
        });
        self.target_embed = Some(embed);
        self.target_norm = Some(norm);
        Ok(())
    }

    fn validate_target_compat(&self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        let sentinel = ffi::from_slice_i32(&[0], &[1, 1]);
        let embedded =
            target
                .embed_tokens(&sentinel)
                .ok_or(DrafterError::TargetMissingFeature {
                    feature: "embed_tokens",
                })?;
        let hidden = ffi::array_shape(&embedded).last().copied().unwrap_or(0);
        if hidden != self.config.hidden_size() as i32 {
            return Err(DrafterError::BindFailed {
                reason: format!(
                    "Inkling MTP hidden size {} does not match target hidden size {hidden}",
                    self.config.hidden_size()
                ),
            });
        }
        let zero = ffi::zeros(&[1, 1, hidden], crate::dtype::FLOAT32);
        let logits = match target.lm_head_module() {
            Some(head) => head.forward(&zero),
            None => target
                .embed_tokens_module()
                .map(|embed| embed.as_linear(&zero))
                .ok_or(DrafterError::TargetMissingFeature {
                    feature: "embed_tokens_module",
                })?,
        };
        let vocab = ffi::array_shape(&logits).last().copied().unwrap_or(0);
        if vocab != self.config.vocab_size() as i32 {
            return Err(DrafterError::BindFailed {
                reason: format!(
                    "Inkling MTP vocabulary {} does not match target vocabulary {vocab}",
                    self.config.vocab_size()
                ),
            });
        }
        Ok(())
    }

    fn reset(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        self.clear_runtime_state();
        self.bind(target)
    }

    fn set_shared_kv(
        &mut self,
        _shared_kv: SharedKv<'_>,
        _kv_offset: usize,
        _position: usize,
        _left_padding: usize,
    ) -> Result<(), DrafterError> {
        Ok(())
    }

    fn configured_block_size(&self) -> Option<usize> {
        Some(self.config.block_size())
    }

    fn prefer_requested_block_size(&self) -> bool {
        true
    }

    fn prefill_from_target_hidden(
        &mut self,
        prompt_tokens: &[i32],
        hidden: &MlxArray,
        first_bonus: i32,
        sampler: &SamplingConfig,
    ) -> Result<(), DrafterError> {
        self.require_bound()?;
        if prompt_tokens.is_empty() {
            return Ok(());
        }
        let shape = ffi::array_shape(hidden);
        if shape
            != [
                1,
                prompt_tokens.len() as i32,
                self.config.hidden_size() as i32,
            ]
        {
            return Err(DrafterError::DraftFailed {
                reason: format!("Inkling MTP prefill hidden must cover the prompt, got {shape:?}"),
            });
        }
        self.clear_runtime_state();
        let mut shifted = prompt_tokens[1..].to_vec();
        shifted.push(first_bonus);
        let output = self.forward_block(&shifted, hidden, 0)?;
        self.set_seed_from_hidden(&output, sampler)
    }

    fn draft_block(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
        sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        self.require_bound()?;
        if block_size <= 1 {
            return Ok(Vec::new());
        }
        self.round_snapshot = Some(self.snapshot_caches());
        self.round_appended = 0;
        let mut drafts = Vec::with_capacity(block_size - 1);
        let (mut token, mut previous) = match (self.seed_token.take(), self.seed_hidden.take()) {
            (Some(token), Some(hidden)) => {
                drafts.push(token);
                (token, hidden)
            }
            _ => (
                last_bonus,
                ffi::copy(hidden.ok_or(DrafterError::DraftBlockMissingHidden)?),
            ),
        };
        while drafts.len() < block_size - 1 {
            let index = self.round_appended.min(self.blocks.len() - 1);
            previous = self.forward_block(&[token], &previous, index)?;
            self.round_appended += 1;
            let logits = self.project_logits(&previous)?;
            token = Self::sample_one(&logits, sampler);
            drafts.push(token);
        }
        Ok(drafts)
    }

    fn accept_verified_tokens(
        &mut self,
        verify_hidden: &MlxArray,
        draft_tokens: &[i32],
        accepted: usize,
        new_tokens: &[i32],
        sampler: &SamplingConfig,
    ) -> Result<(), DrafterError> {
        self.require_bound()?;
        if accepted > draft_tokens.len() {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "Inkling MTP accepted {accepted} exceeds {} drafts",
                    draft_tokens.len()
                ),
            });
        }
        let shape = ffi::array_shape(verify_hidden);
        if shape.len() != 3 || shape[0] != 1 || shape[1] < accepted as i32 + 1 {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "Inkling MTP verify hidden {shape:?} does not cover accepted={accepted}"
                ),
            });
        }
        let snapshot = self
            .round_snapshot
            .take()
            .ok_or_else(|| DrafterError::DraftFailed {
                reason: "Inkling MTP accept called without a round snapshot".into(),
            })?;
        self.restore_caches(&snapshot)?;
        self.round_appended = 0;

        let mut tokens = draft_tokens[..accepted].to_vec();
        if let Some(token) = new_tokens.last().copied() {
            tokens.push(token);
        }
        if !tokens.is_empty() {
            let hidden = ffi::slice(
                verify_hidden,
                &[0, 0, 0],
                &[shape[0], tokens.len() as i32, shape[2]],
            );
            let output = self.forward_block(&tokens, &hidden, 0)?;
            self.set_seed_from_hidden(&output, sampler)?;
        }
        Ok(())
    }

    fn sanitize(&mut self, weights: &mut WeightMap) -> Result<(), DrafterError> {
        sanitize_weights(weights).map_err(|reason| DrafterError::WeightLoad { reason })
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Mtp
    }
}

pub fn has_inkling_mtp_tensors(path: &Path) -> Result<bool, DrafterError> {
    dir_has_tensor_name(path, is_inkling_mtp_tensor_name)
        .map_err(|reason| DrafterError::WeightLoad { reason })
}
