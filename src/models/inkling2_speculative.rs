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

//! Inkling 2 as the target of its bundled DSpark drafter (issue #3374).
//!
//! - A verify forward over `[bonus, proposals]` on burst-owned caches that
//!   returns full-block logits and the states upstream taps
//!   (`language.py`: the partial block entering each listed visit, and the
//!   final normalized hidden for id `visit_count`). It runs inside the
//!   speculative append scope, as upstream's `speculative_verify=True`: the
//!   grouped expert kernel (#3359) engages and the convolution histories keep
//!   a whole block of margin.
//! - The rollback after a partial accept, through `trim_caches`, which checks
//!   every visit before it changes anything.
//! - The block-versus-chain exactness probe that gates the temperature-0
//!   contract, as LFM2 does.
//! - The parts the drafter binds to ([`Inkling2DsparkBinding`]).
//!
//! The scheduler-owned per-sequence state is never touched: a burst builds
//! its own visit caches.

use mlxcel_core::cache::DecodeLookaheadAppendScope;
use mlxcel_core::drafter::dflash::SpeculativeTarget;
use mlxcel_core::drafter::inkling2_dspark::{Inkling2DsparkBinding, Inkling2TargetShape};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::inkling_layer::InklingLayerCache;
use mlxcel_core::layers::{Linear, RMSNorm, UnifiedLinear};
use mlxcel_core::sampling::{TokenBiasMap, apply_token_bias};
use mlxcel_core::{MlxArray, UniquePtr, concatenate_many};

use super::{Head, Inkling2Model};
use crate::models::speculative_exactness::{
    BlockChainExactness, ProbeKey, compare_block_against_chain, mtp_exactness_gate,
};

/// Independent synthetic draws the exactness probe compares.
const PROBE_DRAWS: usize = 3;

/// Prompt length of each probe draw.
const PROBE_PROMPT_LEN: usize = 8;

/// Verify-pass output: `[1, L, vocab]` logits (suppressed ids at `-inf`) and
/// one `[1, L, hidden]` capture per requested id, in request order.
///
/// Used by: the DFlash round loop and the server DFlash burst.
pub struct Inkling2VerifyOutput {
    pub logits: UniquePtr<MlxArray>,
    pub hidden_states: Vec<UniquePtr<MlxArray>>,
}

/// The output-suppressed ids as the `-inf` map classic decode applies.
pub fn suppression_map(ids: &[i32]) -> TokenBiasMap {
    let mut map = TokenBiasMap::new();
    map.suppress_tokens(ids);
    map
}

impl Inkling2Model {
    /// Fresh per-visit caches for one speculative burst.
    ///
    /// Used by: `DFlashTargetModel::make_dflash_caches`, `mlxcel generate`.
    pub fn make_speculative_caches(&self) -> Vec<InklingLayerCache> {
        self.make_internal_caches()
    }

    /// One forward over already-normalized embeddings `[1, L, hidden]` that
    /// returns every row's logits and the captures at `capture_ids`.
    ///
    /// `verify` enters the speculative append scope for a block that may be
    /// rewound; a prompt prefill and the probe's single-token chain pass
    /// `false`, which is the plain decode path.
    pub fn forward_speculative_embeddings(
        &self,
        embeddings: &MlxArray,
        caches: &mut [InklingLayerCache],
        capture_ids: &[usize],
        verify: bool,
        suppressed: &TokenBiasMap,
    ) -> Inkling2VerifyOutput {
        let _scope = verify.then(DecodeLookaheadAppendScope::enter);
        let (hidden, hidden_states) = self.hidden_with_captures(embeddings, caches, capture_ids);
        let logits = self.project(&hidden);
        let logits = if suppressed.is_empty() {
            logits
        } else {
            apply_token_bias(&logits, suppressed)
        };
        Inkling2VerifyOutput {
            logits,
            hidden_states,
        }
    }

    /// [`Self::forward_speculative_embeddings`] over token ids `[1, L]`.
    pub fn forward_speculative(
        &self,
        input_ids: &MlxArray,
        caches: &mut [InklingLayerCache],
        capture_ids: &[usize],
        verify: bool,
        suppressed: &TokenBiasMap,
    ) -> Inkling2VerifyOutput {
        let embeddings = self.normalized_input_embeddings(input_ids);
        self.forward_speculative_embeddings(&embeddings, caches, capture_ids, verify, suppressed)
    }

    /// Keep `accepted + 1` of a `block_size`-row verify block. A full accept
    /// changes nothing. The margins of #3369 cover any block up to 16 rows,
    /// and the drafter's widest block is 8, so a refusal here is a bug.
    pub fn rollback_speculative_cache(
        &self,
        caches: &mut [InklingLayerCache],
        accepted: i32,
        block_size: i32,
    ) {
        let trim = block_size - (accepted + 1);
        if trim <= 0 {
            return;
        }
        if let Err(reason) = self.trim_caches(caches, trim) {
            panic!("Inkling 2 speculative rollback of {trim} rows failed: {reason}");
        }
    }

    /// What the drafter must match: `model_type`, sizes and the visit list.
    pub fn dspark_target_shape(&self) -> Inkling2TargetShape {
        let text = &self.config.text_config;
        Inkling2TargetShape {
            model_type: self.config.model_type.clone(),
            hidden_size: text.hidden_size,
            vocab_size: text.vocab_size,
            num_hidden_layers: text.num_hidden_layers,
            visit_count: self.layer_indices.len(),
            loop_config: self
                .loop_config
                .map(|lc| (lc.core_start, lc.core_end, lc.num_passes)),
        }
    }

    /// The embedding, `embed_norm` and LM head the drafter binds to. A dense
    /// head is handed over in the embedding dtype: it was stored in BF16 and
    /// is kept in f32 only for the target's own product, so the cast back is
    /// exact.
    pub fn dspark_binding(&self) -> Inkling2DsparkBinding {
        let lm_head = match &self.head {
            Head::Dense(weight) => UnifiedLinear::Regular(Linear::new(
                mlxcel_core::astype(weight, mlxcel_core::array_dtype(&self.embed_norm.weight)),
                None,
            )),
            Head::Quantized(head) => head.clone_shared(),
        };
        Inkling2DsparkBinding {
            embed_tokens: self.embed_tokens.clone_shared(),
            embed_norm: RMSNorm::new(
                mlxcel_core::copy(&self.embed_norm.weight),
                self.embed_norm.eps,
            ),
            lm_head,
            shape: self.dspark_target_shape(),
        }
    }

    /// Whether the DSpark burst may engage at `block_size` verify rows on
    /// this host: the block-versus-chain probe, memoized per (model, width)
    /// through [`mtp_exactness_gate`], which owns the decline log line and the
    /// `MLXCEL_MTP_ALLOW_INEXACT` override.
    ///
    /// Used by: the server DFlash burst gate and `mlxcel generate`.
    pub fn dflash_exactness_allows(&self, block_size: usize) -> bool {
        let text = &self.config.text_config;
        let key = ProbeKey {
            block_size: block_size as u32,
            hidden_size: text.hidden_size as u32,
            num_hidden_layers: self.layer_indices.len() as u32,
        };
        mtp_exactness_gate(key, || self.probe_block_chain_exactness(block_size))
    }

    /// Whether a `block_size`-row verify block (run as the round loop runs
    /// it, inside the speculative scope) produces logits byte-identical to
    /// `block_size` single-token decode steps from the same state, over
    /// [`PROBE_DRAWS`] synthetic inputs on fresh caches.
    pub fn probe_block_chain_exactness(&self, block_size: usize) -> BlockChainExactness {
        if block_size < 2 {
            return BlockChainExactness::NotRun("block width below 2 drafts nothing");
        }
        let vocab = self.config.text_config.logits_width();
        if vocab < 2 {
            return BlockChainExactness::NotRun("degenerate vocabulary");
        }
        for draw in 0..PROBE_DRAWS {
            let verdict = self.probe_one_draw(block_size, vocab, draw);
            if !verdict.is_equal() {
                return verdict;
            }
        }
        BlockChainExactness::Equal
    }

    fn probe_one_draw(&self, block_size: usize, vocab: usize, draw: usize) -> BlockChainExactness {
        let salt = draw * 977 + 1;
        let wrap =
            |i: usize, stride: usize, offset: usize| ((i * stride + offset + salt) % vocab) as i32;
        let prompt: Vec<i32> = (0..PROBE_PROMPT_LEN).map(|i| wrap(i, 7, 1)).collect();
        let block: Vec<i32> = (0..block_size).map(|i| wrap(i, 13, 3)).collect();
        let none = TokenBiasMap::new();
        let input = |tokens: &[i32]| mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
        let position_bytes = |logits: &MlxArray, index: i32| -> Vec<u8> {
            let shape = mlxcel_core::array_shape(logits);
            let row = mlxcel_core::slice(logits, &[0, index, 0], &[shape[0], index + 1, shape[2]]);
            mlxcel_core::array_to_raw_bytes(&row)
        };

        let mut chain = self.make_internal_caches();
        let _ = self.forward_speculative(&input(&prompt), &mut chain, &[], false, &none);
        let chain_positions: Vec<Vec<u8>> = block
            .iter()
            .map(|&token| {
                let out = self.forward_speculative(&input(&[token]), &mut chain, &[], false, &none);
                position_bytes(&out.logits, 0)
            })
            .collect();

        let mut caches = self.make_internal_caches();
        let _ = self.forward_speculative(&input(&prompt), &mut caches, &[], false, &none);
        let out = self.forward_speculative(&input(&block), &mut caches, &[], true, &none);
        let block_positions: Vec<Vec<u8>> = (0..block_size)
            .map(|i| position_bytes(&out.logits, i as i32))
            .collect();
        compare_block_against_chain(&block_positions, &chain_positions)
    }
}

/// `[1, L, taps * hidden]`: the captures concatenated in fusion order.
pub(crate) fn concat_captures(verify_out: &Inkling2VerifyOutput) -> UniquePtr<MlxArray> {
    let slabs: Vec<&MlxArray> = verify_out
        .hidden_states
        .iter()
        .map(|slab| slab.as_ref().expect("captured hidden must be non-null"))
        .collect();
    concatenate_many(&slabs, -1)
}

/// The text model as a DFlash target; it suppresses whatever its own
/// `output_suppressed_token_ids` names (nothing today).
impl Inkling2Model {
    /// The prompt goes through in the chunks classic decode prefills with
    /// (`MLXCEL_PREFILL_CHUNK`, default 2048), so a long prompt leaves the
    /// same caches, and the drafter-on run the same text, as classic decode
    /// (issue #3280); one forward over a longer prompt sums its attention in
    /// another order. Only the last chunk's logits are kept (the round loop
    /// samples the last row); the captures are concatenated.
    pub(crate) fn prefill_speculative(
        &self,
        verify_input: &MlxArray,
        caches: &mut [InklingLayerCache],
        capture_layer_ids: &[usize],
        suppressed: &TokenBiasMap,
    ) -> Inkling2VerifyOutput {
        let suppressed = suppressed;
        let length = mlxcel_core::array_shape(verify_input)[1];
        let chunk = mlxcel_core::prefill_plan::prefill_chunk_len() as i32;
        if chunk == 0 || length <= chunk {
            return self.forward_speculative(
                verify_input,
                caches,
                capture_layer_ids,
                false,
                &suppressed,
            );
        }
        let mut parts: Vec<Vec<UniquePtr<MlxArray>>> = Vec::new();
        let mut logits = None;
        let mut start = 0;
        while start < length {
            let end = (start + chunk).min(length);
            let piece = mlxcel_core::utils::slice_axis(verify_input, 1, start, end);
            let out =
                self.forward_speculative(&piece, caches, capture_layer_ids, false, &suppressed);
            // Materialize the chunk before the next one builds on its caches,
            // as the classic chunked prefill does.
            let mut live: Vec<*const MlxArray> = out
                .hidden_states
                .iter()
                .map(|h| &**h as *const MlxArray)
                .collect();
            live.push(&*out.logits as *const MlxArray);
            // SAFETY: every pointer refers to an array `out` owns until the
            // end of this iteration.
            unsafe { mlxcel_core::eval_all(&live) };
            parts.push(out.hidden_states);
            logits = Some(out.logits);
            start = end;
        }
        let captures = parts[0].len();
        let hidden_states = (0..captures)
            .map(|c| {
                let refs: Vec<&MlxArray> = parts.iter().map(|p| &*p[c]).collect();
                mlxcel_core::concatenate_many(&refs, 1)
            })
            .collect();
        Inkling2VerifyOutput {
            logits: logits.expect("a prompt longer than one chunk has a chunk"),
            hidden_states,
        }
    }
}

impl SpeculativeTarget for Inkling2Model {
    type Cache = InklingLayerCache;
    type VerifyOut = Inkling2VerifyOutput;

    fn capture_layer_ids(&self) -> &[usize] {
        // The drafter owns the list (`mtp_config.target_layer_ids`).
        &[]
    }

    fn verify_forward(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
    ) -> Self::VerifyOut {
        self.verify_forward_with_capture_layers(verify_input, caches, &[])
    }

    fn verify_forward_with_capture_layers(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
        capture_layer_ids: &[usize],
    ) -> Self::VerifyOut {
        let suppressed = suppression_map(&LanguageModel::output_suppressed_token_ids(self));
        self.forward_speculative(verify_input, caches, capture_layer_ids, true, &suppressed)
    }

    fn prefill_forward_with_capture_layers(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
        capture_layer_ids: &[usize],
    ) -> Self::VerifyOut {
        let suppressed = suppression_map(&LanguageModel::output_suppressed_token_ids(self));
        self.prefill_speculative(verify_input, caches, capture_layer_ids, &suppressed)
    }

    fn rollback_partial(
        &self,
        caches: &mut [Self::Cache],
        _verify_out: &Self::VerifyOut,
        accepted: i32,
        block_size: i32,
    ) {
        self.rollback_speculative_cache(caches, accepted, block_size);
    }

    fn concat_hidden_for_drafter(&self, verify_out: &Self::VerifyOut) -> UniquePtr<MlxArray> {
        concat_captures(verify_out)
    }

    fn verify_logits<'a>(&self, verify_out: &'a Self::VerifyOut) -> &'a MlxArray {
        verify_out
            .logits
            .as_ref()
            .expect("Inkling 2 verify output must carry logits")
    }
}

/// The image-language wrapper as the burst's target: the text model's
/// forward, with the wrapper's suppressed ids (the image placeholder) at
/// `-inf` in every verify row, as classic decode applies them to every token.
/// A burst serves text-only requests; image prompts are prefilled by the
/// caller through [`Inkling2Model::forward_speculative_embeddings`].
impl SpeculativeTarget for crate::vision::Inkling2VlModel {
    type Cache = InklingLayerCache;
    type VerifyOut = Inkling2VerifyOutput;

    fn capture_layer_ids(&self) -> &[usize] {
        &[]
    }

    fn verify_forward(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
    ) -> Self::VerifyOut {
        self.verify_forward_with_capture_layers(verify_input, caches, &[])
    }

    fn verify_forward_with_capture_layers(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
        capture_layer_ids: &[usize],
    ) -> Self::VerifyOut {
        let suppressed = suppression_map(&LanguageModel::output_suppressed_token_ids(self));
        self.text
            .forward_speculative(verify_input, caches, capture_layer_ids, true, &suppressed)
    }

    fn prefill_forward_with_capture_layers(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
        capture_layer_ids: &[usize],
    ) -> Self::VerifyOut {
        let suppressed = suppression_map(&LanguageModel::output_suppressed_token_ids(self));
        self.text
            .prefill_speculative(verify_input, caches, capture_layer_ids, &suppressed)
    }

    fn rollback_partial(
        &self,
        caches: &mut [Self::Cache],
        _verify_out: &Self::VerifyOut,
        accepted: i32,
        block_size: i32,
    ) {
        self.text
            .rollback_speculative_cache(caches, accepted, block_size);
    }

    fn concat_hidden_for_drafter(&self, verify_out: &Self::VerifyOut) -> UniquePtr<MlxArray> {
        concat_captures(verify_out)
    }

    fn verify_logits<'a>(&self, verify_out: &'a Self::VerifyOut) -> &'a MlxArray {
        verify_out
            .logits
            .as_ref()
            .expect("Inkling 2 verify output must carry logits")
    }
}

#[cfg(test)]
#[path = "inkling2_speculative_tests.rs"]
mod tests;
