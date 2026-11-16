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

use std::path::Path;

use anyhow::{Result, anyhow, ensure};
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, copy};
use serde::de::DeserializeOwned;

use crate::distributed::pipeline::partial_loading::filter_weight_map;
use crate::models;
use crate::models::sanitize_config_json;

use super::common::{TransformerStageLayer, TransformerStageModel};
use super::{FamilyStageExecutor, LayerFilter, StageExecutionInput, StageExecutionOutput};

pub struct Glm4StageExecutor {
    model: TransformerStageModel<Glm4StageLayer>,
}

pub struct Glm4MoeStageExecutor {
    model: TransformerStageModel<Glm4MoeStageLayer>,
}

/// Stage-local executor for the MLA variant of the GLM4 family.
///
/// Unlike `Glm4StageExecutor` and `Glm4MoeStageExecutor`, this one has to
/// sanitize the weight map before loading: every published `glm4_moe_lite`
/// checkpoint stores the MLA up-projection as a single `self_attn.kv_b_proj`
/// rather than the decomposed `embed_q` / `unembed_out` pair that
/// `models::glm4_moe_lite::TransformerBlock::from_weights` reads
/// unconditionally (issue #1029). The split runs on the full weight map,
/// before the stage filter trims it, the same way `DeepSeekV3StageExecutor`
/// and `GlmMoeDsaStageExecutor` order it.
pub struct Glm4MoeLiteStageExecutor {
    model: TransformerStageModel<Glm4MoeLiteStageLayer>,
}

impl Glm4StageExecutor {
    pub fn load(model_dir: &Path, filter: &LayerFilter, stage_index: usize) -> Result<Self> {
        Ok(Self {
            model: load_glm4_family_stage_model(
                model_dir,
                filter,
                stage_index,
                no_sanitize,
                models::glm4::TransformerBlock::from_weights,
                |args: &models::glm4::ModelArgs| args.group_size(),
                |args: &models::glm4::ModelArgs| args.bits(),
                |args: &models::glm4::ModelArgs| args.rms_norm_eps,
                |_| false,
                Glm4StageLayer,
            )?,
        })
    }
}

impl Glm4MoeStageExecutor {
    pub fn load(model_dir: &Path, filter: &LayerFilter, stage_index: usize) -> Result<Self> {
        Ok(Self {
            model: load_glm4_family_stage_model(
                model_dir,
                filter,
                stage_index,
                no_sanitize,
                models::glm4_moe::TransformerBlock::from_weights,
                |args: &models::glm4_moe::ModelArgs| args.group_size(),
                |args: &models::glm4_moe::ModelArgs| args.bits(),
                |args: &models::glm4_moe::ModelArgs| args.rms_norm_eps,
                |args: &models::glm4_moe::ModelArgs| args.tie_word_embeddings,
                Glm4MoeStageLayer,
            )?,
        })
    }
}

impl Glm4MoeLiteStageExecutor {
    pub fn load(model_dir: &Path, filter: &LayerFilter, stage_index: usize) -> Result<Self> {
        Ok(Self {
            model: load_glm4_family_stage_model(
                model_dir,
                filter,
                stage_index,
                models::glm4_moe_lite::sanitize_weights,
                models::glm4_moe_lite::TransformerBlock::from_weights,
                |args: &models::glm4_moe_lite::ModelArgs| args.group_size(),
                |args: &models::glm4_moe_lite::ModelArgs| args.bits(),
                |args: &models::glm4_moe_lite::ModelArgs| args.rms_norm_eps,
                |args: &models::glm4_moe_lite::ModelArgs| args.tie_word_embeddings,
                Glm4MoeLiteStageLayer,
            )?,
        })
    }
}

impl FamilyStageExecutor for Glm4StageExecutor {
    fn make_caches(&self) -> Vec<KVCache> {
        self.model.make_caches()
    }

    fn execute(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput> {
        self.model.execute(input, caches, mask)
    }

    fn execute_last_row(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: Option<usize>,
    ) -> Result<StageExecutionOutput> {
        self.model.execute_last_row(input, caches, mask, last_pos)
    }
}

impl FamilyStageExecutor for Glm4MoeStageExecutor {
    fn make_caches(&self) -> Vec<KVCache> {
        self.model.make_caches()
    }

    fn execute(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput> {
        self.model.execute(input, caches, mask)
    }

    fn execute_last_row(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: Option<usize>,
    ) -> Result<StageExecutionOutput> {
        self.model.execute_last_row(input, caches, mask, last_pos)
    }
}

impl FamilyStageExecutor for Glm4MoeLiteStageExecutor {
    fn make_caches(&self) -> Vec<KVCache> {
        self.model.make_caches()
    }

    fn execute(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput> {
        self.execute_last_row(input, caches, mask, None)
    }

    fn execute_last_row(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: Option<usize>,
    ) -> Result<StageExecutionOutput> {
        // Same contract as `Glm4MoeLiteModel::hidden_pre_norm`: no mask
        // array, and the MLA attention runs a multi-token step through MLX's
        // causal mode over the live window (#3099). The incoming mask is
        // ignored, as the single-process model ignores its `_mask`.
        self.model.execute_last_row(input, caches, None, last_pos)
    }
}

struct Glm4StageLayer(models::glm4::TransformerBlock);
struct Glm4MoeStageLayer(models::glm4_moe::TransformerBlock);
struct Glm4MoeLiteStageLayer(models::glm4_moe_lite::TransformerBlock);

impl TransformerStageLayer for Glm4StageLayer {
    fn forward(
        &self,
        hidden: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.0.forward(hidden, cache, mask)
    }
}

impl TransformerStageLayer for Glm4MoeStageLayer {
    fn forward(
        &self,
        hidden: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.0.forward(hidden, cache, mask)
    }
}

impl TransformerStageLayer for Glm4MoeLiteStageLayer {
    fn forward(
        &self,
        hidden: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.0.forward(hidden, cache, mask)
    }
}

/// Glm4 and Glm4Moe store their attention projections in the layout
/// `TransformerBlock::from_weights` already reads, so their stage load needs
/// no weight surgery. Only the MLA variant does.
fn no_sanitize<A>(weights: WeightMap, _args: &A) -> Result<WeightMap, String> {
    Ok(weights)
}

fn load_glm4_family_stage_model<
    A,
    B,
    L,
    Sanitize,
    BuildLayer,
    GroupSize,
    Bits,
    RmsEps,
    TieEmbeddings,
    Wrap,
>(
    model_dir: &Path,
    filter: &LayerFilter,
    stage_index: usize,
    sanitize: Sanitize,
    build_layer: BuildLayer,
    group_size: GroupSize,
    bits: Bits,
    rms_eps: RmsEps,
    tie_embeddings: TieEmbeddings,
    wrap_layer: Wrap,
) -> Result<TransformerStageModel<L>>
where
    A: DeserializeOwned,
    L: TransformerStageLayer,
    Sanitize: Fn(WeightMap, &A) -> Result<WeightMap, String>,
    BuildLayer: Fn(&WeightMap, &A, usize) -> Result<B, String>,
    GroupSize: Fn(&A) -> i32,
    Bits: Fn(&A) -> i32,
    RmsEps: Fn(&A) -> f32,
    TieEmbeddings: Fn(&A) -> bool,
    Wrap: Fn(B) -> L,
{
    let args: A = parse_model_args(model_dir)?;
    // Sanitation runs on the full weight map, before `filter_weight_map` trims
    // it. Only the MLA variant of this family needs it, and it needs this
    // ordering: `glm4_moe_lite` checkpoints ship `self_attn.kv_b_proj` instead
    // of the `embed_q` / `unembed_out` pair the block loads (issue #1029), and
    // the stage filter would drop the `kv_b_proj` keys of every out-of-range
    // layer before the split could consume them.
    let weights = models::load_text_weights(model_dir, None).map_err(anyhow::Error::msg)?;
    let mut weights = sanitize(weights, &args).map_err(anyhow::Error::msg)?;

    let mut effective_filter = filter.clone();
    if tie_embeddings(&args) && filter.has_lm_head {
        effective_filter.has_embedding = true;
    }
    filter_weight_map(&mut weights, &effective_filter);

    let group_size = group_size(&args);
    let bits = bits(&args);
    let load_embeddings = filter.has_embedding || (tie_embeddings(&args) && filter.has_lm_head);

    let embed_tokens = if load_embeddings {
        Some(
            UnifiedEmbedding::from_weights(&weights, "model.embed_tokens", group_size, bits)
                .map_err(anyhow::Error::msg)?,
        )
    } else {
        None
    };

    let mut layers = Vec::with_capacity(filter.num_layers());
    for layer_idx in filter.layer_range.clone() {
        let layer = build_layer(&weights, &args, layer_idx).map_err(anyhow::Error::msg)?;
        layers.push(wrap_layer(layer));
    }

    let (norm, lm_head) = if filter.has_lm_head {
        let norm_weight = get_weight_copy(&weights, "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, rms_eps(&args));
        let lm_head = if tie_embeddings(&args) {
            UnifiedLinear::from_weights(&weights, "model.embed_tokens", group_size, bits)
                .map_err(anyhow::Error::msg)?
        } else {
            UnifiedLinear::from_weights(&weights, "lm_head", group_size, bits)
                .map_err(anyhow::Error::msg)?
        };
        (Some(norm), Some(lm_head))
    } else {
        (None, None)
    };

    ensure!(
        !layers.is_empty(),
        "stage {} did not load any layers from range {}..{}",
        stage_index,
        filter.layer_range.start,
        filter.layer_range.end
    );

    TransformerStageModel::new(filter.clone(), embed_tokens, layers, norm, lm_head)
}

fn parse_model_args<A: DeserializeOwned>(model_dir: &Path) -> Result<A> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|err| anyhow!("failed to read {}: {}", config_path.display(), err))?;
    let config_str = sanitize_config_json(&config_str);
    serde_json::from_str(&config_str)
        .map_err(|err| anyhow!("failed to parse {}: {}", config_path.display(), err))
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>> {
    weights
        .get(name)
        .map(|weight| copy(weight))
        .ok_or_else(|| anyhow!("weight not found: {}", name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::pipeline::stage_executor::last_row_test_support::assert_last_row_matches_full;
    use crate::models::glm4_moe_lite::tests::tiny_random_weights_and_args;
    use crate::models::glm4_moe_lite::{Glm4MoeLiteModel, ModelArgs};

    /// The stage executor over in-memory fixture weights, built the way
    /// `load_glm4_family_stage_model` builds it from disk.
    fn stage(
        weights: &WeightMap,
        args: &ModelArgs,
        filter: &LayerFilter,
    ) -> Glm4MoeLiteStageExecutor {
        let (gs, bits) = (args.group_size(), args.bits());
        let embed = filter.has_embedding.then(|| {
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", gs, bits)
                .expect("fixture embedding loads")
        });
        let layers = filter
            .layer_range
            .clone()
            .map(|i| {
                Glm4MoeLiteStageLayer(
                    models::glm4_moe_lite::TransformerBlock::from_weights(weights, args, i)
                        .expect("fixture layer loads"),
                )
            })
            .collect();
        let (norm, lm_head) = if filter.has_lm_head {
            (
                Some(RMSNorm::new(
                    get_weight_copy(weights, "model.norm.weight").expect("fixture norm"),
                    args.rms_norm_eps,
                )),
                Some(
                    UnifiedLinear::from_weights(weights, "lm_head", gs, bits)
                        .expect("fixture head loads"),
                ),
            )
        } else {
            (None, None)
        };
        Glm4MoeLiteStageExecutor {
            model: TransformerStageModel::new(filter.clone(), embed, layers, norm, lm_head)
                .expect("fixture stage builds"),
        }
    }

    fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
        assert_eq!(mlxcel_core::array_shape(a), mlxcel_core::array_shape(b));
        let d = mlxcel_core::max_all(&mlxcel_core::abs(&mlxcel_core::subtract(a, b)));
        mlxcel_core::eval(&d);
        mlxcel_core::item_f32(&d)
    }

    /// A two-stage split over the rope-folded causal-mode prefill (#3099)
    /// reproduces the single-process model's logits for a prompt, a warm
    /// multi-token chunk and a decode step.
    #[test]
    fn two_stage_split_matches_the_single_process_model() {
        let _guard = crate::models::embedding_test_support::mlx_test_guard();
        let (weights, args) = tiny_random_weights_and_args(2);
        let model = Glm4MoeLiteModel::from_weights(&weights, &args).expect("tiny model loads");
        let first = stage(
            &weights,
            &args,
            &LayerFilter {
                layer_range: 0..1,
                has_embedding: true,
                has_lm_head: false,
            },
        );
        let last = stage(
            &weights,
            &args,
            &LayerFilter {
                layer_range: 1..2,
                has_embedding: false,
                has_lm_head: true,
            },
        );

        let mut model_caches = model.make_caches();
        let mut first_caches = first.make_caches();
        let mut last_caches = last.make_caches();
        for piece in [&[3u32, 7, 11, 5, 1][..], &[9, 14, 2], &[6]] {
            let ids = mlxcel_core::from_slice_u32(piece, &[1, piece.len() as i32]);
            let want = model.forward(&ids, &mut model_caches, None);
            let hidden = first
                .execute(StageExecutionInput::TokenIds(&ids), &mut first_caches, None)
                .expect("first stage runs")
                .into_hidden_states()
                .expect("first stage returns hidden states");
            let got = last
                .execute(
                    StageExecutionInput::HiddenStates(&hidden),
                    &mut last_caches,
                    None,
                )
                .expect("last stage runs")
                .into_logits()
                .expect("last stage returns logits");
            let diff = max_abs_diff(&got, &want);
            assert!(
                diff < 1e-5,
                "{}-token piece: stage split vs model {diff}",
                piece.len()
            );
        }
    }

    /// The final stage slices the hidden row before the norm and head and
    /// returns the full forward's row (#3081).
    #[test]
    fn last_row_matches_the_full_forward_row() {
        let (weights, args) = tiny_random_weights_and_args(2);
        assert_last_row_matches_full(
            |filter| Box::new(stage(&weights, &args, filter)),
            2,
            16,
            1e-4,
        );
    }
}
