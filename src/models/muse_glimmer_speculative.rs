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

//! Muse Glimmer as a DFlash speculative-decoding target (issue #1343).
//!
//! The Muse Glimmer assistant drafter is fed five of the target's residual
//! streams and its proposals are verified by one causal target forward over
//! the block, so the target needs:
//!
//! - a verify forward returning full-block softcapped logits plus the
//!   post-layer hidden states at the drafter's `target_layer_ids`, captured
//!   before the final norm ([`MuseGlimmerTextModel::forward_speculative`]);
//! - a rollback to the committed prefix after a partial accept. Every Muse
//!   cache is a KV cache, so the rollback is a pure trim: `KVCache::trim`
//!   on the full layers, `RotatingKVCache::trim` on the sliding ones
//!   ([`MuseGlimmerTextModel::rollback_speculative_cache`]);
//! - the sliding caches armed with a speculative buffer before the first
//!   round, so a verify block that crosses the ring boundary can be
//!   appended and rewound without overwriting still-visible window entries
//!   ([`MuseGlimmerTextModel::enable_speculative_buffers`]);
//! - the block-versus-chain exactness probe that decides whether a `bs`-row
//!   verify block reproduces `bs` single-token decode steps on this host
//!   ([`MuseGlimmerTextModel::dflash_exactness_allows`]), run at every width
//!   the adaptive round loop can settle on
//!   ([`MuseGlimmerTextModel::dflash_exactness_allows_every_width`]).
//!
//! `prefill_forward_with_capture_layers` keeps the trait default on
//! purpose. LFM2 overrides it to skip prompt-sized short-conv snapshots; a
//! Muse verify output carries no rollback state at all (the trim reads only
//! the cache offsets), so there is nothing prompt-sized to skip.
//!
//! A child of `muse_glimmer.rs` so it can reach the private model fields;
//! the server reaches it through
//! `mlxcel_core::drafter::dflash::SpeculativeTarget`.

use mlxcel_core::concatenate_many;
use mlxcel_core::drafter::dflash::SpeculativeTarget;
use mlxcel_core::drafter::dflash::muse::MUSE_ASSISTANT_INITIAL_BLOCK_SIZE;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{MuseCache, MuseGlimmerTextModel, MuseGlimmerTextWrapper};
use crate::models::speculative_exactness::{
    BlockChainExactness, ProbeKey, compare_block_against_chain, mtp_exactness_gate,
};

/// Independent synthetic inputs the exactness probe compares before it
/// reports equality; see `models::qwen3_5::PROBE_DRAWS` for why one is not
/// enough.
const PROBE_DRAWS: usize = 3;

/// Prompt length of each probe draw.
const PROBE_PROMPT_LEN: usize = 8;

/// Verify-pass output for the speculative path.
///
/// `hidden_states` is ordered to match the `capture_layer_ids` argument so
/// the drafter can concatenate it along the feature axis. There is no
/// rollback snapshot: every Muse layer holds a KV cache and the rollback is
/// a trim.
///
/// Used by: the DFlash round loop, the server DFlash burst.
pub struct VerifyOutput {
    /// `[B, bs, vocab]` softcapped logits for every verify row.
    pub logits: UniquePtr<MlxArray>,
    /// One `[B, bs, hidden]` post-layer residual stream per requested layer.
    pub hidden_states: Vec<UniquePtr<MlxArray>>,
}

/// The verify widths a run with `block_size` as its ceiling can settle on,
/// narrowest first, and therefore the widths the exactness gate has to
/// clear before the burst engages.
///
/// Two of them since issue #1343: the drafter's declared depth
/// ([`MUSE_ASSISTANT_INITIAL_BLOCK_SIZE`]), which the round loop warms up
/// at, and the requested ceiling the throughput comparator widens to. One
/// when the ceiling is at or below the depth, which is what
/// `--draft-block-size 4` (or narrower) asks for. Rounds the emission
/// budget forces narrower still are not listed: that clamp predates the
/// adaptive width and applies to every DFlash family.
///
/// Used by: [`MuseGlimmerTextModel::dflash_exactness_allows_every_width`].
pub(crate) fn probed_verify_widths(block_size: usize) -> Vec<usize> {
    let depth = MUSE_ASSISTANT_INITIAL_BLOCK_SIZE.min(block_size);
    if depth > 1 && depth != block_size {
        vec![depth, block_size]
    } else {
        vec![block_size]
    }
}

/// Speculative buffer the sliding caches are armed with for a `block_size`
/// verify: `clamp(block_size * 8, 32, 128)` rows, the Gemma 4 MTP rule,
/// floored at `block_size` itself.
///
/// The floor is the difference from `gemma4_mtp_target::mtp_rotating_buffer_size`,
/// and it is what makes the rule safe for an operator-chosen width.
/// `--draft-block-size` is not bounded above anywhere on the way in
/// (`resolve_draft_block_size` returns the override verbatim), and the
/// buffer is the slack past `max_size` that lets a verify block be
/// appended across the ring boundary and then rewound. A block wider than
/// the buffer overwrites window rows that are still visible, and
/// `RotatingKVCache::trim` rewinds the offsets without restoring them: the
/// sequence then continues from a window with holes in it, silently, with
/// no error anywhere. The Gemma rule's 128-row cap makes that reachable at
/// `--draft-block-size 129` and above; the floor keeps
/// `buffer >= block_size` at every width.
///
/// Used by: `DFlashTargetModel::enable_speculative_buffers` on the Muse
/// targets, the Muse exactness probe.
pub fn speculative_buffer_size(block_size: usize) -> i32 {
    (block_size * 8)
        .clamp(32, 128)
        .max(block_size)
        .min(i32::MAX as usize) as i32
}

impl MuseGlimmerTextModel {
    /// The mixed sliding / full cache vector a speculative burst runs on.
    /// Independent of the scheduler's per-sequence slots.
    ///
    /// Used by: the server DFlash burst (`DFlashTargetModel::make_dflash_caches`).
    pub fn make_speculative_caches(&self) -> Vec<MuseCache> {
        self.make_muse_caches()
    }

    /// Arm every sliding cache with a `buffer_size`-row speculative buffer
    /// so a verify block appended past the ring boundary can be rewound.
    /// Runs on fresh caches before the prefill; the full-layer caches need
    /// nothing.
    ///
    /// Used by: `DFlashTargetModel::enable_speculative_buffers`,
    /// [`Self::probe_block_chain_exactness`].
    pub fn enable_speculative_buffers(&self, caches: &mut [MuseCache], buffer_size: i32) {
        for (idx, cache) in caches.iter_mut().enumerate() {
            if let Err(error) = cache.enable_speculative_buffer(buffer_size) {
                tracing::warn!(
                    error,
                    layer_idx = idx,
                    "Muse Glimmer sliding cache could not arm its speculative buffer"
                );
            }
        }
    }

    /// One forward over `input_ids` (`[1, L]`) that also captures the
    /// post-layer residual streams at `capture_layer_ids`, before the final
    /// norm and the head.
    ///
    /// The arithmetic is the generation forward's: `mask = None` so every
    /// layer builds its own causal plus sliding mask, the same embedding
    /// norm, the same softcapped head. That is the exactness premise, and
    /// the probe below measures whether the kernels honour it at this block
    /// width.
    ///
    /// Used by: `SpeculativeTarget::verify_forward_with_capture_layers`,
    /// `SpeculativeTarget::prefill_forward_with_capture_layers`,
    /// [`Self::probe_block_chain_exactness`].
    pub fn forward_speculative(
        &self,
        input_ids: &MlxArray,
        caches: &mut [MuseCache],
        capture_layer_ids: &[usize],
    ) -> VerifyOutput {
        let mut h = self.embed_input_ids(input_ids);
        let shape = mlxcel_core::array_shape(&h);
        let hidden_dtype = mlxcel_core::array_dtype(&h);
        let mut hidden_slots: Vec<Option<UniquePtr<MlxArray>>> =
            (0..capture_layer_ids.len()).map(|_| None).collect();

        for (idx, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut caches[idx], None);
            for (slot, &want) in capture_layer_ids.iter().enumerate() {
                if want == idx {
                    hidden_slots[slot] = Some(mlxcel_core::copy(&h));
                }
            }
        }

        // Keep the vector length-aligned with the request; the drafter's
        // compatibility gate rejects out-of-range ids before this runs, so
        // a miss here means that gate was bypassed. Say so rather than
        // handing the drafter a zero slab that reads as a residual stream.
        let hidden_states: Vec<UniquePtr<MlxArray>> = hidden_slots
            .into_iter()
            .enumerate()
            .map(|(slot, captured)| {
                captured.unwrap_or_else(|| {
                    tracing::error!(
                        slot,
                        requested_layer =
                            capture_layer_ids.get(slot).copied().unwrap_or(usize::MAX),
                        num_layers = self.layers.len(),
                        "Muse Glimmer verify captured no residual stream for this drafter slot; \
                         the pairing check should have refused this drafter, and the drafter \
                         will now read zeros for that layer"
                    );
                    mlxcel_core::zeros(&[shape[0], shape[1], shape[2]], hidden_dtype)
                })
            })
            .collect();

        let h = self.norm.forward(&h);
        let logits = self.lm_head.forward(&h);
        let logits = Self::softcap_logits(
            &logits,
            self.output_multiplier,
            self.final_logit_softcapping,
        );
        VerifyOutput {
            logits,
            hidden_states,
        }
    }

    /// Rewind `caches` to the committed prefix after a partial accept of a
    /// `block_size`-row verify block: `n = accepted + 1` rows stay, every
    /// cache trims `block_size - n` positions. No data moves; the next
    /// append overwrites the rejected rows.
    ///
    /// A full accept never calls this.
    ///
    /// Used by: `SpeculativeTarget::rollback_partial`.
    pub fn rollback_speculative_cache(
        &self,
        caches: &mut [MuseCache],
        accepted: i32,
        block_size: i32,
    ) {
        let trim = block_size - (accepted + 1);
        if trim <= 0 {
            return;
        }
        for (layer_idx, cache) in caches.iter_mut().enumerate() {
            let trimmed = cache.trim(trim);
            if trimmed != trim {
                // Reachable only if a cache was armed with less speculative
                // slack than the block it verified, which
                // `speculative_buffer_size` now prevents. A short rewind
                // leaves the window holding rejected rows, so the
                // continuation is not the committed prefix any more: name it
                // rather than let the stream drift silently.
                tracing::error!(
                    layer_idx,
                    requested = trim,
                    trimmed,
                    "Muse Glimmer cache could not rewind the whole rejected tail; this \
                     sequence's continuation is no longer exact"
                );
            }
            debug_assert_eq!(
                trimmed, trim,
                "a Muse cache must rewind the whole rejected tail"
            );
        }
    }

    /// Whether the DFlash burst may engage at `block_size` verify rows on
    /// this host: the block-versus-chain probe, run at most once per
    /// (model, width) per process through [`mtp_exactness_gate`], which
    /// also owns the decline log line, the `qmv_wide` retry and the
    /// `MLXCEL_MTP_ALLOW_INEXACT` override.
    ///
    /// Used by: [`Self::dflash_exactness_allows_every_width`].
    pub fn dflash_exactness_allows(&self, block_size: usize) -> bool {
        let key = ProbeKey {
            block_size: block_size as u32,
            hidden_size: self.hidden_size as u32,
            num_hidden_layers: self.layers.len() as u32,
        };
        mtp_exactness_gate(key, || self.probe_block_chain_exactness(block_size))
    }

    /// Whether the DFlash burst may engage with `block_size` as its verify
    /// ceiling: the probe has to pass at EVERY width the round loop runs at
    /// systematically, not only at the ceiling.
    ///
    /// Since issue #1343 the loop no longer holds one width. It starts at
    /// the drafter's declared depth (`MUSE_ASSISTANT_INITIAL_BLOCK_SIZE`,
    /// four rows) and widens to the requested ceiling only when a
    /// measurement window emits more tokens per millisecond there, so a
    /// gate that probed the ceiling alone would admit a width it never
    /// measured. The forward width selects which quantized-matmul kernel
    /// MLX dispatches, so a pass at one width is not evidence about
    /// another: `docs/benchmarks.md` records the same comparison reading
    /// 20.6 percent disagreement at width 8 and 0.0 percent at width 32.
    ///
    /// Both probes are memoised per (model, width) by [`mtp_exactness_gate`],
    /// so this costs one extra probe per process, not per request.
    ///
    /// Used by: the server DFlash burst gate.
    pub fn dflash_exactness_allows_every_width(&self, block_size: usize) -> bool {
        probed_verify_widths(block_size)
            .into_iter()
            .all(|width| self.dflash_exactness_allows(width))
    }

    /// Measure whether a `block_size`-row verify block produces logits
    /// byte-identical to `block_size` single-token decode steps from the
    /// same state, over [`PROBE_DRAWS`] synthetic inputs. The chain arm runs
    /// on plain caches, as classic decode does; the block arm runs on caches
    /// armed the way the burst arms them.
    pub fn probe_block_chain_exactness(&self, block_size: usize) -> BlockChainExactness {
        if block_size < 2 {
            return BlockChainExactness::NotRun("block width below 2 drafts nothing");
        }
        let vocab = self.vocab_size;
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

        let as_input =
            |tokens: &[i32]| mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
        let position_bytes = |logits: &MlxArray, index: i32| -> Vec<u8> {
            let shape = mlxcel_core::array_shape(logits);
            let row = mlxcel_core::slice(logits, &[0, index, 0], &[shape[0], index + 1, shape[2]]);
            mlxcel_core::array_to_raw_bytes(&row)
        };

        let mut chain_caches = self.make_muse_caches();
        let _ = self.forward_speculative(&as_input(&prompt), &mut chain_caches, &[]);
        let mut chain_positions: Vec<Vec<u8>> = Vec::with_capacity(block_size);
        for token in &block {
            let out = self.forward_speculative(&as_input(&[*token]), &mut chain_caches, &[]);
            chain_positions.push(position_bytes(&out.logits, 0));
        }

        let mut block_caches = self.make_muse_caches();
        self.enable_speculative_buffers(&mut block_caches, speculative_buffer_size(block_size));
        let _ = self.forward_speculative(&as_input(&prompt), &mut block_caches, &[]);
        let out = self.forward_speculative(&as_input(&block), &mut block_caches, &[]);
        let block_positions: Vec<Vec<u8>> = (0..block_size)
            .map(|i| position_bytes(&out.logits, i as i32))
            .collect();

        compare_block_against_chain(&block_positions, &chain_positions)
    }
}

/// The server-side burst reaches the decoder through the wrapper; these
/// forward the three hooks `DFlashTargetModel` needs.
///
/// Used by: `DFlashTargetModel` on `MuseGlimmerTextWrapper` and
/// `MuseGlimmerVlmModel`.
impl MuseGlimmerTextWrapper {
    pub fn make_speculative_caches(&self) -> Vec<MuseCache> {
        self.model.make_speculative_caches()
    }

    pub fn enable_speculative_buffers(&self, caches: &mut [MuseCache], buffer_size: i32) {
        self.model.enable_speculative_buffers(caches, buffer_size);
    }

    pub fn dflash_exactness_allows_every_width(&self, block_size: usize) -> bool {
        self.model.dflash_exactness_allows_every_width(block_size)
    }
}

/// DFlash target adapter for the Muse Glimmer text decoder. The VLM wrapper
/// delegates to the same methods through its `text`.
impl SpeculativeTarget for MuseGlimmerTextWrapper {
    type Cache = MuseCache;
    type VerifyOut = VerifyOutput;

    fn capture_layer_ids(&self) -> &[usize] {
        // The drafter owns the capture list (`MuseAssistantConfig::target_layer_ids`);
        // the round loop passes it through `verify_forward_with_capture_layers`.
        &[]
    }

    fn verify_forward(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
    ) -> Self::VerifyOut {
        self.model.forward_speculative(verify_input, caches, &[])
    }

    fn verify_forward_with_capture_layers(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
        capture_layer_ids: &[usize],
    ) -> Self::VerifyOut {
        self.model
            .forward_speculative(verify_input, caches, capture_layer_ids)
    }

    fn rollback_partial(
        &self,
        caches: &mut [Self::Cache],
        _verify_out: &Self::VerifyOut,
        accepted: i32,
        block_size: i32,
    ) {
        self.model
            .rollback_speculative_cache(caches, accepted, block_size);
    }

    fn concat_hidden_for_drafter(&self, verify_out: &Self::VerifyOut) -> UniquePtr<MlxArray> {
        debug_assert!(
            !verify_out.hidden_states.is_empty(),
            "Muse verify output must carry at least one captured hidden layer"
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
            .expect("Muse verify output must carry logits")
    }
}

/// Test seams for `muse_glimmer_speculative_tests` (the captured slab must
/// be the residual stream before the final norm, which only the private
/// norm, head and softcap can demonstrate).
#[cfg(test)]
impl MuseGlimmerTextModel {
    pub(crate) fn head_for_test(&self, h: &MlxArray) -> UniquePtr<MlxArray> {
        let normed = self.norm.forward(h);
        let logits = self.lm_head.forward(&normed);
        Self::softcap_logits(
            &logits,
            self.output_multiplier,
            self.final_logit_softcapping,
        )
    }
}
