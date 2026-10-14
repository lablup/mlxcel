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

//! The EAR-TTS decoder: backbone, conditioning, and RVQ code generation.
//!
//! Ports `RVQEARTTSModel` from `mlx_vlm/models/nemotron_voicechat/tts.py`
//! (key tree `tts_model.tts_model`). Classifier-free guidance is always on,
//! as in `VoiceChatSession`: every backbone call runs a batch of two, the
//! conditional row with the subword condition and the unconditional row
//! with `null_emb`.
//!
//! Runtime dtypes follow the reference rather than the checkpoint: the code
//! embeddings are summed in `f32`, which promotes the fused inputs, the whole
//! backbone, the KV caches and the MoG head to `f32` against bf16 weights;
//! only the subword condition (with `null_emb`) stays bf16.
//!
//! On CUDA builds bf16 meeting f32 resolves to bf16 (see
//! [`crate::audio::f32_weights`]), so every stored tensor that joins the f32
//! stream is held or cast as f32 here (issue #2109): `bos_emb` and
//! `audio_prompt_projection_W` are promoted at load, the backbone norms hold
//! their `1 + w` widened to f32 ([`Gemma3Backbone::widen_norms_to_f32`]),
//! and the gated fusion casts its text branch. Each is an exact widening, so
//! builds with upstream promotion compute the same values.

use std::collections::HashMap;

use mlxcel_core::dtype;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::TtsConfig;
use super::fusion::GatedFusion;
use super::mog_head::MogHead;
use super::norm_mlp::{scalar, weight_with_shape};
use super::rvq::RvqCodebooks;
use super::subword::CharAwareSubwordEncoder;
use crate::audio::stage_probe as probe;
use crate::models::gemma3_backbone::{Gemma3Backbone, Gemma3BackboneCaches};

/// Backbone caches of one TTS stream (CFG batch of two).
pub type TtsCaches = Gemma3BackboneCaches;

/// Result of one [`RvqEarTtsModel::step`].
pub struct TtsStepOutput {
    /// Sampled codes, int32 `[1, 1, num_quantizers]`.
    pub codes: UniquePtr<MlxArray>,
    /// Backbone output, `[2, 1, hidden]` (conditional, unconditional).
    pub hidden_states: UniquePtr<MlxArray>,
}

/// `RVQEARTTSModel`.
pub struct RvqEarTtsModel {
    config: TtsConfig,
    backbone: Gemma3Backbone,
    bos_emb: UniquePtr<MlxArray>,
    null_emb: UniquePtr<MlxArray>,
    embed_code: UnifiedLinear,
    embed_subword: CharAwareSubwordEncoder,
    fusion: GatedFusion,
    audio_prompt_projection_w: UniquePtr<MlxArray>,
    mog_head: MogHead,
    rvq: RvqCodebooks,
    schedule: Vec<usize>,
}

/// Whether the EAR-TTS weight `key` (relative to the model prefix) only
/// ever meets f32 activations through a promoting matmul, so it can be held
/// as f32 without changing any output (see [`crate::audio::f32_weights`]):
/// the backbone and MoG-head projections, the code embedding and the audio
/// branch of the gated fusion, plus `bos_emb` (masked, then added to the f32
/// code embeddings) and `audio_prompt_projection_W` (matmul against them).
/// Norm weights (`1 + w` is built in the stored dtype; the backbone widens
/// the built `1 + w` instead), the bf16 subword path (`embed_subword`,
/// `text_proj`, `null_emb`) and the gathered MoG tables (`proj_mus`,
/// `low_mat`) stay as stored.
pub(crate) fn promotes_to_f32(key: &str) -> bool {
    let projection = key.ends_with("_proj.weight") || key.ends_with("_proj.bias");
    let mog_output = ["proj_logits", "proj_logs", "proj_else"]
        .iter()
        .any(|name| {
            key == format!("mog_head.{name}.weight") || key == format!("mog_head.{name}.bias")
        });
    (key.starts_with("backbone.layers.") && projection)
        || (key.starts_with("mog_head.mlp_stack.") && projection)
        || mog_output
        || key.starts_with("embed_code.")
        || key.starts_with("gated_fusion_audio_text.audio_proj.")
        || key == "bos_emb"
        || key == "audio_prompt_projection_W"
}

fn bool_column(values: &[bool]) -> UniquePtr<MlxArray> {
    let ints: Vec<i32> = values.iter().map(|&v| i32::from(v)).collect();
    let arr = mlxcel_core::from_slice_i32(&ints, &[1, values.len() as i32, 1]);
    mlxcel_core::astype(&arr, dtype::BOOL)
}

impl RvqEarTtsModel {
    /// Load from `{prefix}.*` (the checkpoint uses `tts_model.tts_model`).
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &TtsConfig,
    ) -> Result<Self, String> {
        let (gs, bits) = (config.group_size(), config.bits());
        let h = config.hidden_size as i32;
        let key_prefix = format!("{prefix}.");
        let promoted = crate::audio::f32_weights::promoted_subset(weights, &key_prefix, |key| {
            promotes_to_f32(&key[key_prefix.len()..])
        });
        let weights = &promoted;
        let schedule = config.mask_schedule();
        if schedule.iter().sum::<usize>() != config.num_quantizers {
            return Err(format!(
                "TTS mask schedule {schedule:?} does not cover {} codebooks",
                config.num_quantizers
            ));
        }
        let mut backbone = Gemma3Backbone::from_weights(
            weights,
            &format!("{prefix}.backbone"),
            &config.gemma_args(),
        )?;
        backbone.widen_norms_to_f32();
        Ok(Self {
            backbone,
            bos_emb: weight_with_shape(weights, &format!("{prefix}.bos_emb"), &[h])?,
            null_emb: weight_with_shape(weights, &format!("{prefix}.null_emb"), &[h])?,
            embed_code: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.embed_code"),
                gs,
                bits,
            )?,
            embed_subword: CharAwareSubwordEncoder::from_weights(
                weights,
                &format!("{prefix}.embed_subword"),
                &config.character_encoder,
                config.hidden_size,
                gs,
                bits,
            )?,
            fusion: GatedFusion::from_weights(
                weights,
                &format!("{prefix}.gated_fusion_audio_text"),
                config.hidden_size,
                config.num_quantizers,
                config.rms_norm_eps,
                gs,
                bits,
            )?,
            audio_prompt_projection_w: weight_with_shape(
                weights,
                &format!("{prefix}.audio_prompt_projection_W"),
                &[h, h],
            )?,
            mog_head: MogHead::from_weights(
                weights,
                &format!("{prefix}.mog_head"),
                config.hidden_size,
                config.latent_size,
                &config.mog_head,
                gs,
                bits,
            )?,
            rvq: RvqCodebooks::from_weights(
                weights,
                &format!("{prefix}.rvq_embs"),
                config.num_quantizers,
                config.codebook_size,
                config.latent_size,
            )?,
            schedule,
            config: config.clone(),
        })
    }

    pub fn config(&self) -> &TtsConfig {
        &self.config
    }

    /// See [`CharAwareSubwordEncoder::set_vocabulary`].
    pub fn set_vocabulary(&mut self, vocabulary: &HashMap<String, u32>) -> Result<(), String> {
        self.embed_subword.set_vocabulary(vocabulary)
    }

    /// Whether [`Self::set_vocabulary`] has run (steps fail until it has).
    pub fn has_vocabulary(&self) -> bool {
        self.embed_subword.has_vocabulary()
    }

    /// Fresh backbone caches for one stream.
    pub fn make_caches(&self) -> TtsCaches {
        self.backbone.make_caches()
    }

    /// `depthsum_embedding` (`f32`, mask index contributes zero).
    pub fn depthsum_embedding(&self, code: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        self.rvq.depthsum_embedding(code)
    }

    /// Char-aware subword embedding of `ids` (`[batch, time]`, row-major)
    /// where `mask` is true: the conditional channel before guidance.
    pub fn subword_embedding(
        &self,
        ids: &[i32],
        mask: &[bool],
        batch: usize,
        time: usize,
    ) -> Result<UniquePtr<MlxArray>, String> {
        self.embed_subword.forward(ids, mask, batch, time)
    }

    fn embed_codes(&self, code: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let summed = self.rvq.depthsum_embedding(code)?;
        Ok(self.embed_code.forward(&summed))
    }

    /// Conditional subword embedding followed by the unconditional
    /// `null_emb` row: `[2, T, hidden]`.
    pub(crate) fn guided_condition(
        &self,
        ids: &[i32],
        mask: &[bool],
    ) -> Result<UniquePtr<MlxArray>, String> {
        let cond = self.embed_subword.forward(ids, mask, 1, ids.len())?;
        let null = mlxcel_core::broadcast_to(&self.null_emb, &mlxcel_core::array_shape(&cond));
        Ok(mlxcel_core::concatenate(&cond, &null, 0))
    }

    /// Gated fusion of the code embeddings (duplicated for the CFG pair)
    /// with the guided condition: the backbone's input embeddings.
    pub(crate) fn backbone_inputs(
        &self,
        code_embed: &MlxArray,
        cond: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let doubled = mlxcel_core::concatenate(code_embed, code_embed, 0);
        self.fusion.forward(&doubled, cond)
    }

    fn run_backbone(
        &self,
        code_embed: &MlxArray,
        cond: &MlxArray,
        caches: &mut TtsCaches,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let inputs = self.backbone_inputs(code_embed, cond);
        self.backbone.forward_embeds(&inputs, caches)
    }

    /// Prime `caches` with the speaker prompt; returns the backbone output
    /// `[2, T, hidden]`.
    ///
    /// `code`: int `[1, T, Q]` prompt codes (see
    /// [`super::TtsPrompt::from_codec_codes`]); `subword_ids`,
    /// `subword_mask`, `audio_mask`: length `T`; `audio_prompt_latent`:
    /// `[1, T, hidden]` speaker latent that replaces every pre-BOS frame (when
    /// `None`, the frozen `audio_prompt_projection_W` projection is used).
    pub fn warmup(
        &self,
        code: &MlxArray,
        subword_ids: &[i32],
        subword_mask: &[bool],
        audio_mask: &[bool],
        audio_prompt_latent: Option<&MlxArray>,
        caches: &mut TtsCaches,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let t = subword_ids.len();
        let shape = mlxcel_core::array_shape(code);
        if t == 0
            || shape != [1, t as i32, self.config.num_quantizers as i32]
            || subword_mask.len() != t
            || audio_mask.len() != t
        {
            return Err(format!(
                "TTS warmup expects codes [1, T, {}] with T-length subword ids, subword mask \
                 and audio mask; got codes {shape:?}, {} ids, {} subword mask, {} audio mask",
                self.config.num_quantizers,
                t,
                subword_mask.len(),
                audio_mask.len()
            ));
        }
        if let Some(latent) = audio_prompt_latent {
            let ls = mlxcel_core::array_shape(latent);
            if ls != [1, t as i32, self.config.hidden_size as i32] {
                return Err(format!(
                    "audio prompt latent must be [1, {t}, {}], got {ls:?}",
                    self.config.hidden_size
                ));
            }
        }
        let code_embed = self.prompt_embeds(code, audio_mask, audio_prompt_latent)?;
        let cond = self.guided_condition(subword_ids, subword_mask)?;
        self.run_backbone(&code_embed, &cond, caches)
    }

    /// Code-embedding stream of a warmup prompt (`[1, T, hidden]`): the
    /// shifted prompt codes embedded, every pre-BOS frame replaced by the
    /// speaker latent (or its frozen projection), plus `bos_emb` on each BOS
    /// frame. Shapes are checked by [`Self::warmup`].
    pub(crate) fn prompt_embeds(
        &self,
        code: &MlxArray,
        audio_mask: &[bool],
        audio_prompt_latent: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let t = audio_mask.len();
        let (tt, q) = (t as i32, self.config.num_quantizers as i32);
        let head = mlxcel_core::zeros(&[1, 1, q], mlxcel_core::array_dtype(code));
        let body = mlxcel_core::slice(code, &[0, 0, 0], &[1, tt - 1, q]);
        let shifted = mlxcel_core::concatenate(&head, &body, 1);
        let code_embed = self.embed_codes(&shifted)?;
        let act = mlxcel_core::array_dtype(&code_embed);

        let mut bos = Vec::with_capacity(t);
        let mut pre_bos = Vec::with_capacity(t);
        let mut seen_bos = false;
        for (idx, &audio) in audio_mask.iter().enumerate() {
            let is_bos = audio && !(idx > 0 && audio_mask[idx - 1]);
            seen_bos |= is_bos;
            bos.push(is_bos);
            pre_bos.push(!seen_bos);
        }
        let projected = match audio_prompt_latent {
            Some(latent) => mlxcel_core::astype(latent, act),
            None => mlxcel_core::matmul(&code_embed, &self.audio_prompt_projection_w),
        };
        let code_embed = mlxcel_core::where_cond(&bool_column(&pre_bos), &projected, &code_embed);
        let bos_term = mlxcel_core::multiply(&bool_column(&bos), &self.bos_emb);
        Ok(mlxcel_core::add(&code_embed, &bos_term))
    }

    /// `generate_codes`: MaskGIT-style RVQ sampling from a guided backbone
    /// state `[2, 1, hidden]`; returns int32 `[1, 1, Q]`.
    ///
    /// Each refinement pass with a non-zero count draws, in order, one
    /// uniform `[1, 1, num_predictions]` (component choice) and one normal
    /// `[1, 1, latent]` (latent noise) from MLX's global RNG.
    pub fn generate_codes(&self, hidden_states: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let s = mlxcel_core::array_shape(hidden_states);
        if s.len() != 3 || s[0] % 2 != 0 {
            return Err(format!(
                "guided generation expects conditional/unconditional pairs, got {s:?}"
            ));
        }
        let half = s[0] / 2;
        let conditional = mlxcel_core::slice(hidden_states, &[0, 0, 0], &[half, s[1], s[2]]);
        let unconditional = mlxcel_core::slice(hidden_states, &[half, 0, 0], &[s[0], s[1], s[2]]);
        let q = self.config.num_quantizers as i32;
        let mut code = mlxcel_core::full_f32(
            &[half, s[1], q],
            self.config.codebook_size as f32,
            dtype::INT32,
        );
        let mut completed = 0usize;
        for &count in &self.schedule {
            if count == 0 {
                continue;
            }
            let embedded = self.embed_codes(&code)?;
            probe::mark("tts.pass_embed_codes", &[&embedded]);
            let mog_input = mlxcel_core::concatenate(
                &mlxcel_core::add(&embedded, &conditional),
                &mlxcel_core::add(&embedded, &unconditional),
                0,
            );
            let (mu, logs) =
                self.mog_head
                    .infer(&mog_input, self.config.guidance_scale, self.config.top_p)?;
            probe::mark("tts.mog_head", &[&mu, &logs]);
            let noise = unsafe {
                // SAFETY: a null key selects MLX's global key sequence.
                mlxcel_core::random_normal(
                    &mlxcel_core::array_shape(&mu),
                    dtype::FLOAT32,
                    std::ptr::null(),
                )
            };
            let act = mlxcel_core::array_dtype(&mu);
            let jitter = mlxcel_core::multiply(&mlxcel_core::exp(&logs), &noise);
            let jitter = mlxcel_core::multiply(&jitter, &scalar(self.config.noise_scale, act));
            let residual = mlxcel_core::add(&mu, &jitter);
            code = self.rvq.encode_step(&residual, &code, completed, count)?;
            probe::mark("tts.rvq_encode", &[&code]);
            completed += count;
        }
        Ok(code)
    }

    /// One 80 ms frame: condition on the previous frame's codes and the
    /// current LLM text token, advance the caches, and sample new codes.
    pub fn step(
        &self,
        previous_code: &MlxArray,
        text_id: i32,
        caches: &mut TtsCaches,
    ) -> Result<TtsStepOutput, String> {
        let s = mlxcel_core::array_shape(previous_code);
        if s != [1, 1, self.config.num_quantizers as i32] {
            return Err(format!(
                "TTS step expects previous codes [1, 1, {}], got {s:?}",
                self.config.num_quantizers
            ));
        }
        let code_embed = self.embed_codes(previous_code)?;
        probe::mark("tts.embed_codes", &[&code_embed]);
        let cond = self.guided_condition(&[text_id], &[true])?;
        probe::mark("tts.condition", &[&cond]);
        let hidden = self.run_backbone(&code_embed, &cond, caches)?;
        probe::mark("tts.backbone", &[&hidden]);
        let codes = self.generate_codes(&hidden)?;
        Ok(TtsStepOutput {
            codes,
            hidden_states: hidden,
        })
    }
}
