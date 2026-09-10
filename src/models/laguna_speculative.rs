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

//! DFlash [`SpeculativeTarget`] for the Laguna family (#1351).
//!
//! The verify forward is [`LagunaModel::forward_with_capture`] with the
//! drafter's `target_layer_ids`; rollback trims every layer cache by the
//! rejected tail. Sliding layers run a [`RotatingKVCache`] with
//! `block_size` positions of speculative slack
//! (`enable_speculative_buffer`), so a verify block append followed by its
//! rollback stays inside the temporal region and never wraps the ring.
//! [`LagunaWrapper`] forwards the same hooks so one object can serve both as
//! the `LanguageModel` the drafter binds to and as the target the round loop
//! drives.
//!
//! [`RotatingKVCache`]: mlxcel_core::layers::RotatingKVCache

use crate::models::laguna::{LagunaModel, LagunaWrapper};
use crate::models::laguna_layers::LagunaCache;
use mlxcel_core::drafter::dflash::SpeculativeTarget;
use mlxcel_core::{MlxArray, UniquePtr};

/// Output of one Laguna verify forward.
pub struct LagunaVerifyOutput {
    /// `[1, block_size, vocab]` logits.
    pub logits: UniquePtr<MlxArray>,
    /// Post-block residual stream of every captured layer, in
    /// `capture_layer_ids` order, each `[1, block_size, hidden]`.
    pub hidden_states: Vec<UniquePtr<MlxArray>>,
}

impl LagunaModel {
    /// Fresh per-layer caches for a speculative run. Rotating caches take
    /// `block_size` positions of rollback slack so a partial accept can trim
    /// the verify tail without the window having wrapped over it.
    pub fn make_speculative_caches(&self, block_size: usize) -> Vec<LagunaCache> {
        let mut caches = self.make_caches();
        for cache in &mut caches {
            if let LagunaCache::Rotating(rotating) = cache {
                rotating
                    .enable_speculative_buffer(block_size as i32)
                    .expect("fresh FP16 rotating cache accepts a speculative buffer");
            }
        }
        caches
    }

    /// Verify forward with hidden capture. `capture_layer_ids` are 0-based
    /// target layers whose post-block residual stream is returned.
    pub fn forward_speculative(
        &self,
        input_ids: &MlxArray,
        caches: &mut [LagunaCache],
        capture_layer_ids: &[usize],
    ) -> LagunaVerifyOutput {
        let (logits, hidden_states) =
            self.forward_with_capture(input_ids, caches, capture_layer_ids);
        LagunaVerifyOutput {
            logits,
            hidden_states,
        }
    }

    /// Rewind every layer cache to the accepted prefix: the verify block
    /// appended `block_size` positions, `accepted + 1` of them stay. Returns
    /// the number of positions trimmed.
    pub fn rollback_speculative_cache(
        &self,
        caches: &mut [LagunaCache],
        accepted: i32,
        block_size: i32,
    ) -> i32 {
        let trim = block_size - (accepted + 1);
        if trim <= 0 {
            return 0;
        }
        for cache in caches.iter_mut() {
            let trimmed = cache.trim(trim);
            debug_assert_eq!(trimmed, trim, "every Laguna cache must trim the full tail");
        }
        trim
    }
}

impl SpeculativeTarget for LagunaModel {
    type Cache = LagunaCache;
    type VerifyOut = LagunaVerifyOutput;

    fn capture_layer_ids(&self) -> &[usize] {
        // The drafter owns the list (`LagunaDFlashConfig::target_layer_ids`);
        // the round loop passes it through `verify_forward_with_capture_layers`.
        &[]
    }

    fn verify_forward(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
    ) -> Self::VerifyOut {
        self.forward_speculative(verify_input, caches, &[])
    }

    fn verify_forward_with_capture_layers(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
        capture_layer_ids: &[usize],
    ) -> Self::VerifyOut {
        self.forward_speculative(verify_input, caches, capture_layer_ids)
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
        assert!(
            !verify_out.hidden_states.is_empty(),
            "Laguna DFlash verify output must carry at least one captured layer"
        );
        let mut acc = mlxcel_core::copy(&verify_out.hidden_states[0]);
        for slab in verify_out.hidden_states.iter().skip(1) {
            acc = mlxcel_core::concatenate(&acc, slab, -1);
        }
        acc
    }

    fn verify_logits<'a>(&self, verify_out: &'a Self::VerifyOut) -> &'a MlxArray {
        &verify_out.logits
    }
}

impl SpeculativeTarget for LagunaWrapper {
    type Cache = LagunaCache;
    type VerifyOut = LagunaVerifyOutput;

    fn capture_layer_ids(&self) -> &[usize] {
        self.model.capture_layer_ids()
    }

    fn verify_forward(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
    ) -> Self::VerifyOut {
        self.model.verify_forward(verify_input, caches)
    }

    fn verify_forward_with_capture_layers(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
        capture_layer_ids: &[usize],
    ) -> Self::VerifyOut {
        self.model
            .verify_forward_with_capture_layers(verify_input, caches, capture_layer_ids)
    }

    fn rollback_partial(
        &self,
        caches: &mut [Self::Cache],
        verify_out: &Self::VerifyOut,
        accepted: i32,
        block_size: i32,
    ) {
        self.model
            .rollback_partial(caches, verify_out, accepted, block_size);
    }

    fn concat_hidden_for_drafter(&self, verify_out: &Self::VerifyOut) -> UniquePtr<MlxArray> {
        self.model.concat_hidden_for_drafter(verify_out)
    }

    fn verify_logits<'a>(&self, verify_out: &'a Self::VerifyOut) -> &'a MlxArray {
        self.model.verify_logits(verify_out)
    }
}
