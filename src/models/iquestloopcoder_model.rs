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

//! IQuest-Coder Loop model and its `LanguageModel` wrapper.
//!
//! See [`crate::models::iquestloopcoder`] for the architecture and for the
//! layer primitives this assembles.

use std::path::Path;

use mlxcel_core::cache::{KVCacheMode, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::iquestloopcoder::{
    LayerCaches, LoopGate, ModelArgs, TransformerBlock, get_weight_copy,
};
use crate::models::model_owned::ModelOwnedSequenceState;

/// Identity weight sanitizer.
///
/// Kept as an explicit, named step rather than omitted, because "this family
/// needs no renaming" is a fact worth pinning: every key the loader asks for
/// (`model.layers.{i}.*`, `model.gate_projections.{i}.{weight,bias}`,
/// `model.norm.weight`, `lm_head.weight`) is already spelled that way in the
/// published checkpoints, and `model.gate_projections.*` in particular must
/// **not** be routed through any quantization-aware rewrite: it ships as a
/// plain `[H, D]` array with no `.scales` sibling, and
/// [`crate::models::load_text_weights`] already widens it with the other dense
/// weights at the load boundary.
///
/// Idempotent by construction; `sanitize_weights_is_identity_and_idempotent`
/// pins that.
pub fn sanitize_weights(weights: WeightMap) -> WeightMap {
    weights
}

/// The two-pass looped decoder.
///
/// # Divergences from the checkpoint's `modeling_iquestloopcoder.py`
///
/// That file has two forward paths. `_forward_loop` is the training and prefill
/// path and is what this module implements. `_forward_with_cache` is its
/// single-token decode path, and it differs from `_forward_loop` in two ways
/// that this module does **not** reproduce, because reproducing them would make
/// decode disagree with the semantics the model was trained under:
///
/// 1. Its prefill writes the pass-2 (local) cache from K/V recomputed from the
///    hidden state **after** the layer has already run, not from the K/V the
///    layer's own pass-2 attention used. This module stores the K/V that
///    actually attended, which is what `_forward_loop`'s in-layer window sees.
/// 2. Its local cache never shrinks to `loop_window_size` after a prompt longer
///    than the window: it is seeded with every prompt token and then evicts one
///    per decode step, so the "window" stays as wide as the prompt. This module
///    uses a [`mlxcel_core::layers::RotatingKVCache`] bounded at
///    `loop_window_size`, matching the explicit window mask `_forward_loop`
///    applies during training and prefill.
///
/// Prefill logits are unaffected by either point (neither touches what pass 2
/// reads during prefill), so a prefill-only comparison against the reference is
/// exact; decode continuations will diverge from that file after the first
/// token on a prompt longer than the window.
pub struct IQuestLoopCoderModel {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<TransformerBlock>,
    /// `model.gate_projections.{i}`, one per layer, used only by pass 2.
    gates: Vec<LoopGate>,
    norm: RMSNorm,
    lm_head: Option<UnifiedLinear>,
    loop_window_size: i32,
    eos_token_ids: Vec<i32>,
}

impl IQuestLoopCoderModel {
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn eos_token_ids(&self) -> &[i32] {
        &self.eos_token_ids
    }

    /// One [`LayerCaches`] per decoder layer: a dense pass-1 cache paired with a
    /// `loop_window_size`-bounded rotating pass-2 cache.
    pub fn make_caches(&self) -> Vec<LayerCaches> {
        (0..self.layers.len())
            .map(|_| LayerCaches::new(self.loop_window_size))
            .collect()
    }

    /// Run both passes and return the final hidden states, `[B, L, hidden]`.
    ///
    /// The RoPE offset for **both** passes is the pass-1 cache offset captured
    /// before anything is written this call, so the pass-2 query is rotated at
    /// the same absolute positions the pass-1 query was.
    fn hidden_states(
        &self,
        input_ids: &MlxArray,
        caches: &mut [LayerCaches],
    ) -> UniquePtr<MlxArray> {
        debug_assert_eq!(caches.len(), self.layers.len());
        // One sequence per call. `supports_batching` is false and
        // `supports_padded_prefill` is false, so nothing hands this a stacked
        // batch; a batch would silently share one cache pair across rows.
        debug_assert_eq!(mlxcel_core::array_shape(input_ids)[0], 1);
        // Both caches take the same tokens on every call, so they cannot drift.
        // Pass 2 rotating at the pass-1 offset depends on that, so state it.
        debug_assert_eq!(
            caches[0].pass1.offset, caches[0].pass2.offset,
            "the two caches must advance in lockstep; the shared RoPE offset assumes it"
        );
        let offset = caches[0].pass1.offset;

        let mut h = self.embed_tokens.forward(input_ids);

        // Pass 1. Each layer's returned K/V is past history plus the tokens
        // written this call; pass 2 attends exactly these arrays. In `Fp16`,
        // `update_and_fetch` returns `slice` views, which MLX implements as
        // `copy_shared_buffer`, so holding all of them costs no extra device
        // memory beyond the caches themselves. That is *only* true in `Fp16`:
        // the `Int8` and `Turbo*` arms dequantize on read and hand back
        // materialized tensors, so this loop would then pin one full K/V window
        // per layer. See `set_kv_cache_layer_modes`, which is why those modes
        // are refused rather than wired.
        let mut pass1_kv = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            let (out, keys, values) = layer.forward_pass1(&h, &mut caches[i].pass1, offset);
            h = out;
            pass1_kv.push((keys, values));
        }

        // Pass 2. Same weights, same positions, gated global/local mix.
        for (i, layer) in self.layers.iter().enumerate() {
            let (keys, values) = &pass1_kv[i];
            h = layer.forward_pass2(
                &h,
                keys,
                values,
                &self.gates[i],
                &mut caches[i].pass2,
                offset,
                self.loop_window_size,
            );
        }

        h
    }

    fn logits(&self, h: &MlxArray) -> UniquePtr<MlxArray> {
        let normed = self.norm.forward(h);
        match self.lm_head.as_ref() {
            Some(head) => head.forward(&normed),
            None => self.embed_tokens.as_linear(&normed),
        }
    }

    pub fn forward_with_caches(
        &self,
        input_ids: &MlxArray,
        caches: &mut [LayerCaches],
    ) -> UniquePtr<MlxArray> {
        let h = self.hidden_states(input_ids, caches);
        self.logits(&h)
    }

    /// `forward_with_caches` that projects only one position through `lm_head`.
    ///
    /// The vocabulary is 76 800 wide, so projecting a whole 2048-token prefill
    /// chunk allocates a `[1, 2048, 76800]` logits tensor purely to throw all
    /// but one row away. Slicing the hidden state first keeps that at
    /// `[1, 1, 76800]`.
    pub fn forward_last_logits_with_caches(
        &self,
        input_ids: &MlxArray,
        caches: &mut [LayerCaches],
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        let h = self.hidden_states(input_ids, caches);
        let shape = mlxcel_core::array_shape(&h);
        let pos = last_pos as i32;
        let last = mlxcel_core::slice(&h, &[0, pos, 0], &[shape[0], pos + 1, shape[2]]);
        self.logits(&last)
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let mut args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        args.set_checkpoint_label(model_dir);
        // Refuse an unsupported `loop_num` before reading 20+ GB of weights.
        args.validate()?;

        let weights = sanitize_weights(crate::models::load_text_weights(model_dir, None)?);
        let model = Self::from_weights(&weights, &args)?;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        args.validate()?;

        let embed_tokens = UnifiedEmbedding::from_weights(
            weights,
            "model.embed_tokens",
            args.group_size(),
            args.bits(),
        )?;

        let rope = args.rope_scaling_kind();
        let num_heads = args.num_attention_heads as i32;
        let head_dim = args.head_dim() as i32;

        let mut layers = Vec::with_capacity(args.num_hidden_layers);
        let mut gates = Vec::with_capacity(args.num_hidden_layers);
        for i in 0..args.num_hidden_layers {
            layers.push(TransformerBlock::from_weights(weights, args, i, &rope)?);
            gates.push(LoopGate::from_weights(
                weights,
                &format!("model.gate_projections.{i}"),
                num_heads,
                head_dim,
            )?);
        }

        let norm = RMSNorm::new(
            get_weight_copy(weights, "model.norm.weight")?,
            args.rms_norm_eps,
        );

        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights_with_mode(
                weights,
                "lm_head",
                args.group_size(),
                args.bits(),
                &args.quant_mode(),
            )?)
        };

        Ok(Self {
            embed_tokens,
            layers,
            gates,
            norm,
            lm_head,
            loop_window_size: args.loop_window_size as i32,
            eos_token_ids: args.eos_token_ids(),
        })
    }

    #[cfg(test)]
    pub(crate) fn layer(&self, idx: usize) -> &TransformerBlock {
        &self.layers[idx]
    }

    #[cfg(test)]
    pub(crate) fn gate(&self, idx: usize) -> &LoopGate {
        &self.gates[idx]
    }

    #[cfg(test)]
    pub(crate) fn attention(&self, idx: usize) -> &crate::models::iquestloopcoder::Attention {
        &self.layers[idx].self_attn
    }
}

/// `LanguageModel` wrapper owning the per-layer cache pairs.
///
/// # Why the caches are model-owned
///
/// The generator's external cache surface is a flat `Vec<KVCache>`, one dense
/// cache per layer. This family needs two caches per layer and one of them is a
/// [`mlxcel_core::layers::RotatingKVCache`], so it cannot be expressed there.
/// The state therefore lives in a
/// [`ModelOwnedSequenceState`], the same mechanism Gemma 3, Llama 4 and
/// Qwen3-Next use, keyed by [`SequenceId`] so concurrent server requests get
/// isolated cache pairs instead of sharing one `RefCell`.
///
/// # Prefix caching is excluded, deliberately
///
/// Declaring [`SequenceStateLayout::model_owned`] is what excludes this family
/// from the server's prompt/prefix cache, and the exclusion is explicit rather
/// than incidental: `BatchScheduler::donate_finished_sequence_cache` and its
/// adopt half both bail on `SequenceStateBackend::ModelOwned`, recording
/// `PromptCacheRejectReason::ModelOwnedState`, so a request never records a
/// misleading cache miss for a path it structurally cannot use. Snapshot reuse
/// is likewise off ([`LanguageModel::supports_snapshot_reuse`] defaults to
/// `false`). Wiring either on would require a detach/adopt representation for a
/// rotating cache alongside a dense one, which is out of scope here; see
/// `docs/supported-models.md`.
pub struct IQuestLoopCoderWrapper {
    model: IQuestLoopCoderModel,
    sequence_state: ModelOwnedSequenceState<LayerCaches>,
    /// Number of layers, cached so the hot accessors do not walk the model.
    num_layers: usize,
}

impl IQuestLoopCoderWrapper {
    pub fn new(model: IQuestLoopCoderModel) -> Self {
        let caches = model.make_caches();
        let num_layers = model.num_layers();
        Self {
            model,
            sequence_state: ModelOwnedSequenceState::new(caches),
            num_layers,
        }
    }

    /// Drop the fallback (no-`SequenceId`) cache pair and start a fresh one.
    ///
    /// The CLI and benchmark paths generate without sequence ids, so they all
    /// share the fallback slot; resetting it between runs is what stops a
    /// second generation from continuing the first one's offsets.
    fn reset_caches(&self) {
        self.sequence_state
            .replace_internal(self.model.make_caches());
    }

    #[cfg(test)]
    pub(crate) fn model(&self) -> &IQuestLoopCoderModel {
        &self.model
    }

    #[cfg(test)]
    pub(crate) fn with_fallback_caches<R>(&self, f: impl FnOnce(&mut [LayerCaches]) -> R) -> R {
        self.sequence_state.with_sequence_state(None, f)
    }
}

impl LanguageModel for IQuestLoopCoderWrapper {
    /// The `caches` and `mask` arguments are ignored on purpose.
    ///
    /// `caches` is the generator's dense per-layer slice, which cannot hold this
    /// family's cache pairs (see the type docs). `mask` is ignored because the
    /// two passes need two *different* masks (full causal for pass 1 and the
    /// global branch, windowed for the pass-2 local branch), so a single
    /// caller-supplied mask cannot serve both; both are derived from the cache
    /// offsets instead. `supports_padded_prefill` is `false` for the same
    /// reason a caller-supplied padded mask would be wrong here.
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_sequence_id(input_ids, None, caches, mask)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.model.make_caches(),
            |caches| self.model.forward_with_caches(input_ids, caches),
        )
    }

    fn forward_last_logits(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        self.forward_last_logits_with_sequence_id(input_ids, None, caches, mask, last_pos)
    }

    fn forward_last_logits_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.model.make_caches(),
            |caches| {
                self.model
                    .forward_last_logits_with_caches(input_ids, caches, last_pos)
            },
        )
    }

    /// Empty: the model owns its state. Matches Gemma 3's model-owned surface.
    /// Also the point at which the CLI path resets the fallback cache pair.
    fn make_caches(&self) -> Vec<KVCache> {
        self.reset_caches();
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.model.eos_token_ids().to_vec()
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(self.num_layers)
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, self.model.make_caches());
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.sequence_state.release_sequence_state(seq_id);
    }

    fn reset_runtime_state(&self) {
        self.reset_caches();
    }

    /// FP16 KV only, stated rather than silently ignored.
    ///
    /// The scheduler injects a per-layer mode table through here for every
    /// model-owned family, and the trait default drops it on the floor. It also
    /// logs `kv_cache_mode_applied_layers` from the *configured* table rather
    /// than from anything the model did, so an operator passing
    /// `--kv-cache-mode int8` would otherwise get no memory saving and a log
    /// line claiming the mode reached all 80 layers.
    ///
    /// Wiring the quantized modes is not a small change and is deliberately not
    /// attempted: `KVCache::update_and_fetch` returns dequantized *copies* in
    /// `Int8` and `Turbo*`, not views, and `hidden_states` holds every layer's
    /// pass-1 return alive across the whole of pass 2, so a quantized pass-1
    /// cache would multiply peak device memory by the layer count instead of
    /// reducing it. `supports_turbo_kv` already omits this family, so
    /// `mlxcel arch` reports FP16 only; this makes the runtime agree.
    fn set_kv_cache_layer_modes(&self, modes: Vec<KVCacheMode>) {
        if modes.iter().any(|mode| *mode != KVCacheMode::Fp16) {
            tracing::warn!(
                "iquestloopcoder serves FP16 KV only; the requested per-layer KV cache modes \
                 are ignored"
            );
        }
    }

    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        Some(vec![KVCacheMode::Fp16; self.num_layers])
    }

    /// Chunked prefill is safe here, and this states it rather than leaving it
    /// to the default.
    ///
    /// It is not obvious: a continuation chunk finds the pass-1 cache holding
    /// every prior key while the pass-2 ring holds at most `loop_window_size -
    /// 1` of them, so the two caches return different key counts for the same
    /// chunk and the windowed mask has to match the ring's. Verified on the real
    /// 40B checkpoint at chunk sizes landing on, inside and across the window
    /// boundary (218 single-pass against 128, 97, 64 and 32): identical top-5
    /// token ids, logits within the 0.25 the test asserts. See
    /// `check_chunked_prefill` in `tests/iquestloopcoder_parity.rs`.
    fn supports_chunked_prefill(&self) -> bool {
        true
    }

    /// No batched path: a batched forward would have to interleave two passes
    /// across sequences whose pass-2 rotating caches sit at different ring
    /// positions. Each request is served on its own cache pair instead.
    fn supports_batching(&self) -> bool {
        false
    }

    /// Padded prefill would append the pad tokens into the pass-2 rotating
    /// cache. A dense cache can be trimmed afterwards, but a ring that has
    /// already wrapped cannot: the pad keys would have overwritten real ones.
    fn supports_padded_prefill(&self) -> bool {
        false
    }
}
