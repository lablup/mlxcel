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

//! DFlash2 [`SpeculativeTarget`] for Raon-1 mini (issue #3436).
//!
//! The verify forward runs a block of up to 16 rows so that every row equals
//! the one-token decode step it replaces, bit for bit:
//!
//! - **RDA**: the projections go through `row_invariant` products, and the
//!   short conv and the delta-rule recurrence run one position at a time in
//!   their decode form ([`RaonDeltaAttention::forward_recording`]), which
//!   also records the state after every position for the rollback.
//! - **MLA**: one decode-shaped attention step per position against the
//!   latent cache, so each query sees the keys its decode step sees.
//! - **mHC, router, SiTU MLPs**: `row_invariant` products (per row, or the
//!   row-invariant matvec kernel when `MLXCEL_RAON_ROW_INVARIANT` is on).
//! - **experts and head** (MXFP8): MLX's quantized products are per row at
//!   these widths unless `qmv_wide` is on, which the exactness gate turns
//!   off when it is the only difference.
//! - **steering**: the burst's alpha (`[1, 1, 1]`) is added to the steered
//!   MoE outputs exactly as plain decode adds it.
//!
//! The drafter reads the four mHC streams after each tap layer
//! (`target_layer_ids`, layer then stream order), i.e. the rows entering
//! the next layer. Rollback trims the MLA latent caches and restores each
//! RDA layer's conv window and recurrent state from the recorded step of
//! the last accepted position.
//!
//! [`RaonDeltaAttention::forward_recording`]: super::super::raon_rda::RaonDeltaAttention::forward_recording

use super::super::raon_rda::RdaStep;
use super::{Mixer, Raon1MiniLayerCache, Raon1MiniModel, StepMasks};
use crate::models::speculative_exactness::{
    BlockChainExactness, ProbeKey, compare_block_against_chain, mtp_exactness_gate,
};
use crate::models::speculative_exactness_draws as draws;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::drafter::dflash::SpeculativeTarget;
use mlxcel_core::row_invariant;
use mlxcel_core::{MlxArray, UniquePtr};

/// Independent synthetic inputs the exactness probe runs.
const PROBE_DRAWS: usize = 3;
/// Prompt length of each probe draw.
const PROBE_PROMPT_LEN: usize = 8;

/// Whether every Raon forward forms its dense products with the
/// row-invariant matvec kernel (`MLXCEL_RAON_ROW_INVARIANT=1`). Off, a
/// verify block forms them one row at a time instead, which is exact but
/// reads those weights once per row.
pub(crate) fn row_kernel_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MLXCEL_RAON_ROW_INVARIANT").is_ok_and(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
    })
}

/// The row-invariant scope of a Raon forward: a verify block always opens
/// one; any other forward opens one only with the row kernel on, so decode
/// and verify form their products the same way.
pub(crate) fn row_scope(verify_block: bool) -> Option<row_invariant::Scope> {
    let kernel = row_kernel_enabled();
    (kernel || verify_block).then(|| {
        row_invariant::Scope::enter(row_invariant::Mode {
            kernel,
            verify_block,
        })
    })
}

/// One layer's state in a speculative burst.
pub struct Raon1MiniSpecCache {
    pub layer: Raon1MiniLayerCache,
    /// RDA layers: the state after each row of the last verify block.
    steps: Vec<RdaStep>,
    /// The burst's steering alpha (the same in every layer).
    alpha: f32,
}

/// Output of one Raon verify (or prefill) forward.
pub struct Raon1MiniVerifyOutput {
    /// `[1, L, vocab]` logits (`[1, 1, vocab]` for a prefill: the last row).
    pub logits: UniquePtr<MlxArray>,
    /// The streams after every captured layer, `[1, L, streams * hidden]`
    /// each, in capture order.
    pub hidden_states: Vec<UniquePtr<MlxArray>>,
}

impl Raon1MiniModel {
    /// Fresh burst caches steered at `seq`'s alpha (`None`: the model
    /// default).
    pub fn make_speculative_caches(&self, seq: Option<SequenceId>) -> Vec<Raon1MiniSpecCache> {
        let alpha = self.reasoning_effort_alpha(seq);
        self.make_layer_caches()
            .into_iter()
            .map(|layer| Raon1MiniSpecCache {
                layer,
                steps: Vec::new(),
                alpha,
            })
            .collect()
    }

    /// The forward of a burst. `verify` marks a verify block (2 to 16 rows:
    /// row-invariant products, per-position mixers, recorded RDA steps);
    /// `last_row` computes logits for the last row only, as plain prefill
    /// does.
    pub fn forward_speculative(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Raon1MiniSpecCache],
        capture_layer_ids: &[usize],
        verify: bool,
        last_row: bool,
    ) -> Raon1MiniVerifyOutput {
        assert_eq!(
            caches.len(),
            self.layers.len(),
            "raon1_mini: one cache per layer"
        );
        let shape = mlxcel_core::array_shape(input_ids);
        assert_eq!(shape[0], 1, "raon1_mini speculative forward is B = 1");
        let l = shape[1];
        let verify = verify && (2..=row_invariant::MAX_ROWS).contains(&l);
        let _rows = row_scope(verify);
        let steer = self.steer_rows(&[caches[0].alpha]);
        let h = self.embed_tokens.forward(input_ids);
        let mut streams = super::raon_mhc::expand_streams(&h, self.streams);
        let mut hidden_states = Vec::with_capacity(capture_layer_ids.len());
        let masks = StepMasks::default();
        for (idx, (layer, cache)) in self.layers.iter().zip(caches.iter_mut()).enumerate() {
            let Raon1MiniSpecCache {
                layer: state,
                steps,
                ..
            } = cache;
            steps.clear();
            streams = layer.forward_with_mixer(&streams, steer.as_deref(), |mixer, x| {
                match (mixer, state) {
                    (Mixer::Rda(attn), Raon1MiniLayerCache::Delta(c)) => {
                        attn.forward_recording(x, c, None, verify.then_some(steps))
                    }
                    (Mixer::Mla(attn), Raon1MiniLayerCache::Attn(kv)) if verify => {
                        let rows: Vec<_> = (0..l)
                            .map(|t| {
                                let row = mlxcel_core::utils::slice_axis(x, 1, t, t + 1);
                                attn.forward(&row, masks.mla_mask.as_deref(), kv)
                            })
                            .collect();
                        mlxcel_core::concatenate_many(
                            &rows.iter().map(|r| &**r).collect::<Vec<_>>(),
                            1,
                        )
                    }
                    (Mixer::Mla(attn), Raon1MiniLayerCache::Attn(kv)) => {
                        attn.forward(x, masks.mla_mask.as_deref(), kv)
                    }
                    _ => panic!("raon1_mini: cache kind mismatch at layer {idx}"),
                }
            });
            if capture_layer_ids.contains(&idx) {
                let s = mlxcel_core::array_shape(&streams);
                hidden_states.push(mlxcel_core::reshape(&streams, &[s[0], s[1], s[2] * s[3]]));
            }
        }
        let row = last_row.then_some(l as usize - 1);
        Raon1MiniVerifyOutput {
            logits: self.head(&streams, row),
            hidden_states,
        }
    }

    /// Rewind every layer to the accepted prefix: the verify block appended
    /// `block_size` rows and `accepted + 1` of them stay. MLA latent caches
    /// are trimmed; RDA layers take the state recorded after the last kept
    /// row. Returns the number of rows dropped.
    pub fn rollback_speculative(
        &self,
        caches: &mut [Raon1MiniSpecCache],
        accepted: i32,
        block_size: i32,
    ) -> i32 {
        let trim = block_size - (accepted + 1);
        if trim <= 0 {
            return 0;
        }
        for (layer_idx, cache) in caches.iter_mut().enumerate() {
            match &mut cache.layer {
                Raon1MiniLayerCache::Attn(kv) => {
                    let trimmed = kv.trim(trim);
                    assert_eq!(trimmed, trim, "raon1_mini MLA layer {layer_idx} rollback");
                }
                Raon1MiniLayerCache::Delta(state) => {
                    let step = cache.steps.get(accepted as usize).unwrap_or_else(|| {
                        panic!(
                            "raon1_mini RDA layer {layer_idx}: no recorded step {accepted} of \
                             {} (a verify block wider than the row-invariant limit cannot be \
                             rolled back)",
                            cache.steps.len()
                        )
                    });
                    state.conv = Some(mlxcel_core::array_handle_clone(&step.conv));
                    state.ssm = step.ssm.as_deref().map(mlxcel_core::array_handle_clone);
                    state.offset -= trim;
                }
            }
            cache.steps.clear();
        }
        trim
    }

    fn hidden_size(&self) -> usize {
        mlxcel_core::array_shape(&self.norm.weight)[0] as usize
    }

    fn vocab_size(&self) -> usize {
        mlxcel_core::array_shape(self.embed_tokens.weight())[0] as usize
    }

    /// Whether a `block_size`-row verify block is byte-identical to
    /// `block_size` decode steps on this host, memoized per (model, width)
    /// through [`mtp_exactness_gate`] (which owns the decline log line, the
    /// `qmv_wide` retry and `MLXCEL_MTP_ALLOW_INEXACT`).
    pub fn dflash_exactness_allows(&self, block_size: usize) -> bool {
        let key = ProbeKey {
            block_size: block_size as u32,
            hidden_size: self.hidden_size() as u32,
            num_hidden_layers: self.layers.len() as u32,
        };
        mtp_exactness_gate(key, || {
            let lengths = draws::prompt_lengths(PROBE_PROMPT_LEN, PROBE_DRAWS, None);
            self.probe_block_chain_exactness_at(block_size, &lengths)
        })
    }

    /// Compare a `block_size`-row verify block against `block_size` decode
    /// steps from the same state, one draw per prompt length, at the model
    /// default effort.
    pub fn probe_block_chain_exactness_at(
        &self,
        block_size: usize,
        prompt_lengths: &[usize],
    ) -> BlockChainExactness {
        if block_size < 2 {
            return BlockChainExactness::NotRun("block width below 2 drafts nothing");
        }
        if block_size > row_invariant::MAX_ROWS as usize {
            return BlockChainExactness::NotRun(
                "block wider than the row-invariant verify limit (16)",
            );
        }
        let vocab = self.vocab_size();
        draws::run_draws(prompt_lengths, |draw, prompt_len| {
            let prompt = draws::prompt_tokens(draw, prompt_len, vocab);
            let block = draws::block_tokens(draw, block_size, vocab);
            let as_input =
                |tokens: &[i32]| mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
            let row_bytes = |logits: &MlxArray, index: i32| -> Vec<u8> {
                let s = mlxcel_core::array_shape(logits);
                let row = mlxcel_core::slice(logits, &[0, index, 0], &[s[0], index + 1, s[2]]);
                mlxcel_core::array_to_raw_bytes(&row)
            };

            let mut chain_caches = self.make_layer_caches();
            let _ = self.forward_with_caches(&as_input(&prompt), &mut chain_caches, None);
            let chain: Vec<Vec<u8>> = block
                .iter()
                .map(|t| {
                    row_bytes(
                        &self.forward_with_caches(&as_input(&[*t]), &mut chain_caches, None),
                        0,
                    )
                })
                .collect();

            let mut spec = self.make_speculative_caches(None);
            let _ = self.forward_speculative(&as_input(&prompt), &mut spec, &[], false, true);
            let out = self.forward_speculative(&as_input(&block), &mut spec, &[], true, false);
            let blocked: Vec<Vec<u8>> = (0..block_size)
                .map(|i| row_bytes(&out.logits, i as i32))
                .collect();
            compare_block_against_chain(&blocked, &chain)
        })
    }
}

impl SpeculativeTarget for Raon1MiniModel {
    type Cache = Raon1MiniSpecCache;
    type VerifyOut = Raon1MiniVerifyOutput;

    fn capture_layer_ids(&self) -> &[usize] {
        // The drafter owns the list (`RaonDFlash2Config::target_layer_ids`).
        &[]
    }

    fn verify_forward(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
    ) -> Self::VerifyOut {
        self.forward_speculative(verify_input, caches, &[], true, false)
    }

    fn verify_forward_with_capture_layers(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
        capture_layer_ids: &[usize],
    ) -> Self::VerifyOut {
        self.forward_speculative(verify_input, caches, capture_layer_ids, true, false)
    }

    /// The prompt runs as plain prefill does (one pass, logits for the last
    /// row only), with the tap layers captured.
    fn prefill_forward_with_capture_layers(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
        capture_layer_ids: &[usize],
    ) -> Self::VerifyOut {
        self.forward_speculative(verify_input, caches, capture_layer_ids, false, true)
    }

    fn rollback_partial(
        &self,
        caches: &mut [Self::Cache],
        _verify_out: &Self::VerifyOut,
        accepted: i32,
        block_size: i32,
    ) {
        self.rollback_speculative(caches, accepted, block_size);
    }

    fn concat_hidden_for_drafter(&self, verify_out: &Self::VerifyOut) -> UniquePtr<MlxArray> {
        assert!(
            !verify_out.hidden_states.is_empty(),
            "Raon DFlash2 verify output must carry the captured tap layers"
        );
        let refs: Vec<&MlxArray> = verify_out.hidden_states.iter().map(|h| &**h).collect();
        mlxcel_core::concatenate_many(&refs, -1)
    }

    fn verify_logits<'a>(&self, verify_out: &'a Self::VerifyOut) -> &'a MlxArray {
        &verify_out.logits
    }
}

#[cfg(test)]
#[path = "raon1_mini_speculative_tests.rs"]
mod tests;
