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

//! Dense Qwen3 and Llama as DSpark speculative-decoding targets (issue
//! #3261).
//!
//! The official DSpark drafters for these families
//! (`RedHatAI/Qwen3-4B-speculator.dspark` for `Qwen/Qwen3-4B`,
//! `openbmb/MiniCPM5-2B-DSpark` for `openbmb/MiniCPM5-2B`, a `llama` target)
//! read the target's residual stream after each tapped layer and have their
//! proposals verified by one causal target forward over the block. Both
//! targets are attention-only, so what the generation forward lacks is small:
//!
//! - a forward that also returns the post-layer residual streams at the
//!   drafter's `target_layer_ids`, captured before the final norm
//!   ([`forward_speculative`]). It runs the model's own generation forward
//!   (`Qwen3Model::forward_hidden_capturing`,
//!   `Llama3Model::hidden_states_capturing`), so a verify block builds the
//!   graph a decode step builds;
//! - a rollback to the committed prefix after a partial accept, which is a
//!   KV trim and nothing else: there is no recurrent state ([`rollback`]);
//! - the block-versus-chain exactness probe that decides whether a verify
//!   block of `bs` rows produces the logits of `bs` single-token decode steps
//!   on this host ([`exactness_allows`]). The temperature-0 contract rests on
//!   it, as on every other DFlash-family target.
//!
//! Mirrors `lfm2_speculative.rs`; upstream is the `capture_layer_ids` /
//! `speculative_verify` / `rollback_speculative_cache` trio on
//! `mlx_vlm/models/qwen3/language.py` and `models/llama/language.py`
//! (Neywa-Labs/mlx-vlm#2676).

use std::ops::Range;

use mlxcel_core::cache::DecodeLookaheadAppendScope;
use mlxcel_core::concatenate_many;
use mlxcel_core::drafter::Drafter;
use mlxcel_core::drafter::dflash::SpeculativeTarget;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::prefill_plan::{PrefillCaps, PrefillPlan, PrefillSplits};
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::speculative_exactness::{
    BlockChainExactness, ProbeKey, compare_block_against_chain, mtp_exactness_gate,
};
use crate::models::speculative_exactness_draws as draws;
use crate::models::{Llama3Model, Qwen3Model};

/// Independent synthetic inputs the exactness probe compares before it
/// reports equality; see `models::qwen3_5::PROBE_DRAWS` for why one is not
/// enough.
const PROBE_DRAWS: usize = 3;

/// Prompt length of each probe draw.
const PROBE_PROMPT_LEN: usize = 8;

/// Verify-pass output of a dense target.
///
/// `hidden_states` is ordered to match the `capture_layer_ids` argument so
/// the drafter can concatenate it along the feature axis. There are no
/// rollback snapshots: a partial accept is undone by trimming the KV caches.
///
/// Used by: the DFlash round loop, the server and offline DFlash drivers.
pub struct DenseVerifyOutput {
    /// `[1, L, vocab]` logits for every row of a verify block; `[1, 1,
    /// vocab]`, the last row's, for a prefill (see [`Rows`]).
    pub logits: UniquePtr<MlxArray>,
    /// One `[1, L, hidden]` post-layer residual stream per requested layer.
    pub hidden_states: Vec<UniquePtr<MlxArray>>,
}

/// Record the residual stream `h` leaving layer `layer` in every slot whose
/// requested id is `layer`.
///
/// Called from inside the two generation forwards, once per layer. With no
/// capture requested (every generation step) the loop body never runs and
/// the forward is untouched.
///
/// Used by: `Qwen3Model::forward_hidden_capturing`,
/// `Llama3Model::hidden_states_capturing`.
pub(crate) fn capture_layer(
    layer: usize,
    h: &MlxArray,
    capture_layer_ids: &[usize],
    captured: &mut [Option<UniquePtr<MlxArray>>],
) {
    for (slot, &want) in captured.iter_mut().zip(capture_layer_ids) {
        if want == layer {
            *slot = Some(mlxcel_core::copy(h));
        }
    }
}

/// What the generic drivers below need from a dense target: its own
/// generation forward with a capture list, and the two sizes the probe
/// builds its synthetic inputs from.
///
/// Implemented for [`Qwen3Model`] and [`Llama3Model`].
pub(crate) trait DenseSpeculativeBackbone: LanguageModel {
    /// Family name for messages.
    const FAMILY: &'static str;

    /// The generation forward over `input_ids` (`[1, L]`, mask `None`, so a
    /// multi-row forward is causal through the cache's own helper), returning
    /// the normalized hidden states and filling `captured`.
    fn hidden_capturing(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        capture_layer_ids: &[usize],
        captured: &mut [Option<UniquePtr<MlxArray>>],
    ) -> UniquePtr<MlxArray>;

    /// The output head over normalized hidden states, as generation applies
    /// it.
    fn head(&self, hidden: &MlxArray) -> UniquePtr<MlxArray>;
}

impl DenseSpeculativeBackbone for Qwen3Model {
    const FAMILY: &'static str = "Qwen3";

    fn hidden_capturing(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        capture_layer_ids: &[usize],
        captured: &mut [Option<UniquePtr<MlxArray>>],
    ) -> UniquePtr<MlxArray> {
        self.forward_hidden_capturing(input_ids, None, caches, None, capture_layer_ids, captured)
    }

    fn head(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        self.logits_from_hidden(hidden)
    }
}

impl DenseSpeculativeBackbone for Llama3Model {
    const FAMILY: &'static str = "Llama";

    fn hidden_capturing(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        capture_layer_ids: &[usize],
        captured: &mut [Option<UniquePtr<MlxArray>>],
    ) -> UniquePtr<MlxArray> {
        self.hidden_states_capturing(input_ids, None, caches, None, capture_layer_ids, captured)
    }

    fn head(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        self.logits_from_hidden(hidden)
    }
}

/// Which rows of a speculative forward get logits, and which scope the
/// forward is built in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rows {
    /// A prompt prefill piece or a single decode step, built exactly as
    /// classic decode builds it (`forward_last_logits`): outside the verify
    /// scope, with the hidden state sliced to its last row BEFORE the head,
    /// so the logits are `[1, 1, vocab]`. A prefill therefore leaves the
    /// caches, and samples the first token from the logits, that classic
    /// decode would, and does not project every prompt row through the
    /// vocabulary.
    Last,
    /// A verify block: every row's logits, built inside the speculative
    /// append scope, as the engine's token-only verify round is
    /// (`Engine::verify`). The scope is what lets
    /// `MLXCEL_MLA_ROW_INVARIANT=1` make a Llama block row-invariant (#3421).
    VerifyBlock,
}

/// One forward over `input_ids` (`[1, L]`) that also captures the post-layer
/// residual streams at `capture_layer_ids`, for every row whichever `rows`
/// is asked for: the drafter consumes the captures of the whole prompt.
///
/// A capture id at or past the depth yields a zero slab so the result stays
/// length-aligned with the request; the pairing gate
/// ([`dense_target_pairing_error`], `DFlashDrafter::validate_target_compat`)
/// refuses such a drafter before any forward.
///
/// Used by: the `SpeculativeTarget` impls below, [`prefill_planned`],
/// [`probe_block_chain_exactness_at`].
pub(crate) fn forward_speculative<M: DenseSpeculativeBackbone>(
    model: &M,
    input_ids: &MlxArray,
    caches: &mut [KVCache],
    capture_layer_ids: &[usize],
    rows: Rows,
) -> DenseVerifyOutput {
    let _scope = (rows == Rows::VerifyBlock).then(DecodeLookaheadAppendScope::enter);
    let mut captured: Vec<Option<UniquePtr<MlxArray>>> =
        (0..capture_layer_ids.len()).map(|_| None).collect();
    let hidden = model.hidden_capturing(input_ids, caches, capture_layer_ids, &mut captured);
    let shape = mlxcel_core::array_shape(&hidden);
    let dtype = mlxcel_core::array_dtype(&hidden);
    let hidden_states = captured
        .into_iter()
        .map(|slot| slot.unwrap_or_else(|| mlxcel_core::zeros(&shape, dtype)))
        .collect();
    let logits = match rows {
        Rows::VerifyBlock => model.head(&hidden),
        Rows::Last => {
            let last = (shape[1].max(1) - 1) as usize;
            model.head(&mlxcel_core::generate::logits_at_position(&hidden, last))
        }
    };
    DenseVerifyOutput {
        logits,
        hidden_states,
    }
}

/// The burst's prompt prefill over the pieces classic decode forwards: the
/// `chunk`-sized chunks and the prompt-cache snapshot splits in `boundary`.
///
/// One forward over a prompt longer than a chunk sums its attention in
/// another order than the chunked prefill classic decode runs, and the
/// caches it leaves are the state every later token is computed from. Each
/// piece is evaluated before the next builds on its caches, as the classic
/// chunked prefill does. Each piece is a [`Rows::Last`] forward, the graph
/// classic decode builds for it; only the last piece's logits are kept (one
/// row, which the first bonus is sampled from) and the captures are
/// concatenated back to one row per prompt token.
///
/// A prompt of one piece, the common case, is a single forward. No tile
/// padding: the speculative forward pads nothing.
///
/// Used by: the server DFlash burst (`DFlashTargetModel::prefill_burst`) and
/// the offline driver (`commands::generate_dflash`).
pub(crate) fn prefill_planned<M: DenseSpeculativeBackbone>(
    model: &M,
    prompt: &MlxArray,
    caches: &mut [KVCache],
    capture_layer_ids: &[usize],
    chunk: usize,
    boundary: PrefillSplits,
) -> DenseVerifyOutput {
    let length = mlxcel_core::array_shape(prompt)[1].max(0) as usize;
    let caps = PrefillCaps::for_model(model, false);
    let ranges: Vec<Range<usize>> =
        PrefillPlan::with_prefix(length, 0, boundary, chunk, caps).ranges();
    if ranges.len() <= 1 {
        return forward_speculative(model, prompt, caches, capture_layer_ids, Rows::Last);
    }
    let mut parts: Vec<Vec<UniquePtr<MlxArray>>> = Vec::with_capacity(ranges.len());
    let mut logits = None;
    for range in &ranges {
        let piece = mlxcel_core::utils::slice_axis(prompt, 1, range.start as i32, range.end as i32);
        let out = forward_speculative(model, &piece, caches, capture_layer_ids, Rows::Last);
        let mut live: Vec<*const MlxArray> = out
            .hidden_states
            .iter()
            .map(|h| &**h as *const MlxArray)
            .collect();
        live.push(&*out.logits as *const MlxArray);
        // SAFETY: every pointer refers to an array `out` owns until the end
        // of this iteration.
        unsafe { mlxcel_core::eval_all(&live) };
        parts.push(out.hidden_states);
        logits = Some(out.logits);
    }
    let captures = parts[0].len();
    let hidden_states = (0..captures)
        .map(|c| {
            let refs: Vec<&MlxArray> = parts.iter().map(|p| &*p[c]).collect();
            concatenate_many(&refs, 1)
        })
        .collect();
    DenseVerifyOutput {
        logits: logits.expect("a prompt of several pieces has a piece"),
        hidden_states,
    }
}

/// Rewind `caches` to the committed prefix after a partial accept of a
/// `block_size`-row verify block: `accepted + 1` rows stay, the rest are
/// trimmed off every layer's KV cache. A full accept trims nothing.
///
/// Used by: `SpeculativeTarget::rollback_partial` on both targets.
pub(crate) fn rollback(caches: &mut [KVCache], accepted: i32, block_size: i32) {
    let trim = block_size - (accepted + 1);
    if trim <= 0 {
        return;
    }
    for cache in caches.iter_mut() {
        cache.trim(trim);
    }
}

/// Why a DFlash-family drafter cannot drive a dense target, from the two
/// numbers the target knows and the two the drafter declares, or `None`.
///
/// The drafter's own gate (`DFlashDrafter::validate_target_compat`) measures
/// the same things for a DSpark drafter. This is the target's half, run for
/// every drafter family these targets admit (they name no required family,
/// as Qwen 3.5 does not), so a plain DFlash drafter whose taps or vocabulary
/// do not fit is refused by name too rather than reaching MLX as a zero slab
/// or an out-of-range gather. `drafter_vocab` is `None` for a drafter that
/// does not report one.
///
/// Used by: [`DenseSpeculativeTarget::dflash_drafter_pairing_error`].
pub(crate) fn dense_target_pairing_error(
    family: &str,
    target_layers: usize,
    target_vocab: Option<usize>,
    target_layer_ids: &[usize],
    drafter_vocab: Option<usize>,
) -> Option<String> {
    if let Some(&tap) = target_layer_ids.iter().find(|&&tap| tap >= target_layers) {
        return Some(format!(
            "the drafter taps target layer index {tap} (target_layer_ids \
             {target_layer_ids:?}) but this {family} target has {target_layers} layers \
             (valid indices 0..={}); pair the target with the drafter published for it, or \
             drop --model-draft to serve it with classic decode",
            target_layers.saturating_sub(1)
        ));
    }
    if let (Some(drafter), Some(target)) = (drafter_vocab, target_vocab)
        && drafter != target
    {
        return Some(format!(
            "the drafter's vocab_size is {drafter} but this {family} target's vocabulary is \
             {target}; a drafter proposes target token ids, so the two must match. Pair the \
             target with the drafter published for it, or drop --model-draft to serve it with \
             classic decode"
        ));
    }
    None
}

/// The target vocabulary, measured on the embedding table the way the DSpark
/// gate measures it; `None` when the model hands out no embedding module.
///
/// Used by: [`DenseSpeculativeTarget::dflash_drafter_pairing_error`].
pub(crate) fn target_vocab_size(model: &dyn LanguageModel) -> Option<usize> {
    let sentinel = mlxcel_core::from_slice_i32(&[0_i32], &[1, 1]);
    let embedded = model.embed_tokens(&sentinel)?;
    let hidden = *mlxcel_core::array_shape(&embedded).last()?;
    let embed = model.embed_tokens_module()?;
    let zero = mlxcel_core::zeros(&[1, 1, hidden], mlxcel_core::dtype::FLOAT32);
    let vocab = *mlxcel_core::array_shape(&embed.as_linear(&zero)).last()?;
    usize::try_from(vocab).ok()
}

/// Hidden size of `model`, read off the shape of one embedded token. The
/// gather is never evaluated: MLX knows the shape of a lazy array.
fn hidden_size(model: &dyn LanguageModel) -> usize {
    let sentinel = mlxcel_core::from_slice_i32(&[0_i32], &[1, 1]);
    model
        .embed_tokens(&sentinel)
        .and_then(|embedded| mlxcel_core::array_shape(&embedded).last().copied())
        .map_or(0, |hidden| hidden.max(0) as usize)
}

/// Width of the logits `model`'s head emits, read off the shape of the head
/// applied to one zero row (never evaluated either).
fn logits_width<M: DenseSpeculativeBackbone>(model: &M) -> usize {
    let hidden = hidden_size(model) as i32;
    let zero = mlxcel_core::zeros(&[1, 1, hidden], mlxcel_core::dtype::FLOAT32);
    mlxcel_core::array_shape(&model.head(&zero))
        .last()
        .map_or(0, |&vocab| vocab.max(0) as usize)
}

/// Whether the DSpark burst may engage at `block_size` verify rows on this
/// host: the block-versus-chain probe, run at most once per (model, width)
/// per process through [`mtp_exactness_gate`], which also owns the decline
/// log line, the `qmv_wide` retry and the `MLXCEL_MTP_ALLOW_INEXACT`
/// override.
///
/// Used by: [`DenseSpeculativeTarget::dflash_exactness_allows`].
pub(crate) fn exactness_allows<M: DenseSpeculativeBackbone>(model: &M, block_size: usize) -> bool {
    let key = ProbeKey {
        block_size: block_size as u32,
        hidden_size: hidden_size(model) as u32,
        num_hidden_layers: model.num_layers() as u32,
    };
    mtp_exactness_gate(key, || probe_block_chain_exactness(model, block_size))
}

/// Measure whether a `block_size`-row verify block produces logits
/// byte-identical to `block_size` single-token decode steps from the same
/// state, over [`PROBE_DRAWS`] synthetic inputs. Both arms run on fresh
/// caches; no per-sequence state is touched.
pub(crate) fn probe_block_chain_exactness<M: DenseSpeculativeBackbone>(
    model: &M,
    block_size: usize,
) -> BlockChainExactness {
    let lengths = draws::prompt_lengths(PROBE_PROMPT_LEN, PROBE_DRAWS, None);
    probe_block_chain_exactness_at(model, block_size, &lengths)
}

/// [`probe_block_chain_exactness`] over draws of the given prompt lengths,
/// one draw per entry.
///
/// Used by: [`probe_block_chain_exactness`] and the unit tests.
pub(crate) fn probe_block_chain_exactness_at<M: DenseSpeculativeBackbone>(
    model: &M,
    block_size: usize,
    prompt_lengths: &[usize],
) -> BlockChainExactness {
    if block_size < 2 {
        return BlockChainExactness::NotRun("block width below 2 drafts nothing");
    }
    let vocab = logits_width(model);
    if vocab < 2 {
        return BlockChainExactness::NotRun("degenerate vocabulary");
    }
    draws::run_draws(prompt_lengths, |draw, prompt_len| {
        let prompt = draws::prompt_tokens(draw, prompt_len, vocab);
        let block = draws::block_tokens(draw, block_size, vocab);

        // The single-token arm is the graph classic decode builds, so it is
        // the contract's reference.
        let mut chain_caches = model.make_caches();
        let _ = forward_speculative(
            model,
            &draws::as_input(&prompt),
            &mut chain_caches,
            &[],
            Rows::Last,
        );
        let mut chain_positions: Vec<Vec<u8>> = Vec::with_capacity(block_size);
        for token in &block {
            let out = forward_speculative(
                model,
                &draws::as_input(&[*token]),
                &mut chain_caches,
                &[],
                Rows::Last,
            );
            chain_positions.push(draws::position_bytes(&out.logits, 0));
        }

        let mut block_caches = model.make_caches();
        let _ = forward_speculative(
            model,
            &draws::as_input(&prompt),
            &mut block_caches,
            &[],
            Rows::Last,
        );
        let out = forward_speculative(
            model,
            &draws::as_input(&block),
            &mut block_caches,
            &[],
            Rows::VerifyBlock,
        );
        let block_positions: Vec<Vec<u8>> = (0..block_size)
            .map(|i| draws::position_bytes(&out.logits, i as i32))
            .collect();

        compare_block_against_chain(&block_positions, &chain_positions)
    })
}

/// A dense target as the DFlash drivers see it: the round-loop contract plus
/// the three decisions a driver makes around it. The server burst
/// (`server::batch::dflash_target`) and the offline `mlxcel generate` driver
/// (`commands::generate_dflash`) both go through this, so the two front ends
/// cannot gate or prefill a pairing differently.
///
/// Implemented for [`Qwen3Model`] and [`Llama3Model`].
pub trait DenseSpeculativeTarget:
    LanguageModel + SpeculativeTarget<Cache = KVCache, VerifyOut = DenseVerifyOutput>
{
    /// Whether a `block_size`-row verify block reproduces the single-token
    /// decode chain on this host ([`exactness_allows`]).
    fn dflash_exactness_allows(&self, block_size: usize) -> bool;

    /// Why `drafter` cannot drive this target, from the target's depth and
    /// vocabulary ([`dense_target_pairing_error`]), or `None`.
    fn dflash_drafter_pairing_error(&self, drafter: &dyn Drafter) -> Option<String>;

    /// The prompt prefill over classic decode's pieces ([`prefill_planned`]),
    /// with no prompt-cache snapshot splits.
    fn dflash_prefill_planned(
        &self,
        prompt: &MlxArray,
        caches: &mut [KVCache],
        capture_layer_ids: &[usize],
        chunk: usize,
    ) -> DenseVerifyOutput;
}

/// `SpeculativeTarget` and [`DenseSpeculativeTarget`] for a dense backbone:
/// every hook is one of the generic drivers above.
macro_rules! impl_dense_speculative_target {
    ($model:ty) => {
        impl DenseSpeculativeTarget for $model {
            fn dflash_exactness_allows(&self, block_size: usize) -> bool {
                exactness_allows(self, block_size)
            }

            fn dflash_drafter_pairing_error(&self, drafter: &dyn Drafter) -> Option<String> {
                dense_target_pairing_error(
                    <$model as DenseSpeculativeBackbone>::FAMILY,
                    self.num_layers(),
                    target_vocab_size(self),
                    drafter.dflash_target_layer_ids().unwrap_or(&[]),
                    drafter.dflash_vocab_size(),
                )
            }

            fn dflash_prefill_planned(
                &self,
                prompt: &MlxArray,
                caches: &mut [KVCache],
                capture_layer_ids: &[usize],
                chunk: usize,
            ) -> DenseVerifyOutput {
                prefill_planned(
                    self,
                    prompt,
                    caches,
                    capture_layer_ids,
                    chunk,
                    PrefillSplits::default(),
                )
            }
        }

        impl SpeculativeTarget for $model {
            type Cache = KVCache;
            type VerifyOut = DenseVerifyOutput;

            fn capture_layer_ids(&self) -> &[usize] {
                // The drafter owns the capture list
                // (`DFlashConfig::target_layer_ids`); the round loop passes it
                // through `verify_forward_with_capture_layers`.
                &[]
            }

            fn verify_forward(
                &self,
                verify_input: &MlxArray,
                caches: &mut [Self::Cache],
            ) -> Self::VerifyOut {
                forward_speculative(self, verify_input, caches, &[], Rows::VerifyBlock)
            }

            fn verify_forward_with_capture_layers(
                &self,
                verify_input: &MlxArray,
                caches: &mut [Self::Cache],
                capture_layer_ids: &[usize],
            ) -> Self::VerifyOut {
                forward_speculative(
                    self,
                    verify_input,
                    caches,
                    capture_layer_ids,
                    Rows::VerifyBlock,
                )
            }

            /// The prompt prefill in one piece: classic decode's own graph,
            /// with logits for the last row only.
            fn prefill_forward_with_capture_layers(
                &self,
                verify_input: &MlxArray,
                caches: &mut [Self::Cache],
                capture_layer_ids: &[usize],
            ) -> Self::VerifyOut {
                forward_speculative(self, verify_input, caches, capture_layer_ids, Rows::Last)
            }

            fn rollback_partial(
                &self,
                caches: &mut [Self::Cache],
                _verify_out: &Self::VerifyOut,
                accepted: i32,
                block_size: i32,
            ) {
                rollback(caches, accepted, block_size);
            }

            fn concat_hidden_for_drafter(
                &self,
                verify_out: &Self::VerifyOut,
            ) -> UniquePtr<MlxArray> {
                debug_assert!(
                    !verify_out.hidden_states.is_empty(),
                    "DSpark verify output must carry at least one captured hidden layer"
                );
                let slabs: Vec<&MlxArray> = verify_out
                    .hidden_states
                    .iter()
                    .map(|slab| slab.as_ref().expect("captured hidden must be non-null"))
                    .collect();
                concatenate_many(&slabs, -1)
            }

            fn verify_logits<'a>(&self, verify_out: &'a Self::VerifyOut) -> &'a MlxArray {
                verify_out
                    .logits
                    .as_ref()
                    .expect("DSpark verify output must carry logits")
            }
        }
    };
}

impl_dense_speculative_target!(Qwen3Model);
impl_dense_speculative_target!(Llama3Model);

#[cfg(test)]
#[path = "dense_speculative_tests.rs"]
pub(crate) mod tests;

/// The two published pairings, on the real checkpoints when they are on
/// disk.
#[cfg(test)]
#[path = "dense_speculative_real_checkpoint_tests.rs"]
mod real_checkpoint_tests;
