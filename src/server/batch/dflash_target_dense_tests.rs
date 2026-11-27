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

//! Dense Qwen3 and Llama as server-side DFlash targets (issue #3261): the
//! target's half of the pairing gate, and the burst policy the two targets
//! declare. Run on real (tiny) targets, so the depth and the vocabulary are
//! the loaded model's own.

use mlxcel_core::MlxArray;
use mlxcel_core::drafter::{Drafter, DrafterKind};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};

use super::{
    DFlashDrafterFamily, DFlashTargetModel, FirstHiddenRows, required_family_pairing_error,
};

/// A drafter that reports only what the target's gate reads: a family,
/// a tap list and a vocabulary. Draft and bind are never reached.
struct GateDrafter {
    dspark: bool,
    taps: Vec<usize>,
    vocab: Option<usize>,
}

impl Drafter for GateDrafter {
    fn bind(
        &mut self,
        _target: &dyn LanguageModel,
    ) -> Result<(), mlxcel_core::drafter::DrafterError> {
        Ok(())
    }
    fn draft_block(
        &mut self,
        _last_bonus: i32,
        _hidden: Option<&MlxArray>,
        _block_size: usize,
        _sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, mlxcel_core::drafter::DrafterError> {
        Ok(Vec::new())
    }
    fn sanitize(
        &mut self,
        _weights: &mut mlxcel_core::weights::WeightMap,
    ) -> Result<(), mlxcel_core::drafter::DrafterError> {
        Ok(())
    }
    fn kind(&self) -> DrafterKind {
        DrafterKind::Dflash
    }
    fn is_dspark(&self) -> bool {
        self.dspark
    }
    fn dflash_target_layer_ids(&self) -> Option<&[usize]> {
        Some(&self.taps)
    }
    fn dflash_vocab_size(&self) -> Option<usize> {
        self.vocab
    }
}

/// Issue #3261: dense Qwen3 and Llama targets admit a DSpark drafter and
/// refuse a tap index at or above their depth with an error naming the
/// index. Run on real (tiny) targets, so the depth and the vocabulary
/// are the loaded model's own.
#[test]
fn qwen3_and_llama_targets_admit_dspark_and_refuse_a_tap_at_or_above_their_depth() {
    use crate::models::dense_speculative::tests::{LAYERS, VOCAB, tiny_llama, tiny_qwen3};
    let _runtime = crate::initialize_runtime();

    // No required family, as on Qwen 3.5, so a DSpark drafter passes the
    // family half of the gate (and so does every other family: these
    // targets rely on the two bounds below instead).
    assert_eq!(
        <crate::models::Qwen3Model as DFlashTargetModel>::required_drafter_family(),
        None
    );
    assert_eq!(
        <crate::models::Llama3Model as DFlashTargetModel>::required_drafter_family(),
        None
    );
    for required in [
        <crate::models::Qwen3Model as DFlashTargetModel>::required_drafter_family(),
        <crate::models::Llama3Model as DFlashTargetModel>::required_drafter_family(),
    ] {
        assert!(required_family_pairing_error(required, DFlashDrafterFamily::DSpark).is_none());
    }

    let qwen3 = tiny_qwen3();
    let llama = tiny_llama();
    let vocab = Some(VOCAB as usize);
    let drafter = |taps: Vec<usize>, vocab: Option<usize>| GateDrafter {
        dspark: true,
        taps,
        vocab,
    };
    assert_eq!(
        DFlashDrafterFamily::of(&drafter(vec![1], vocab)),
        DFlashDrafterFamily::DSpark
    );

    // Admitted: taps below the depth, the last layer included.
    let admitted = drafter(vec![1, LAYERS - 1], vocab);
    assert!(qwen3.drafter_pairing_error(&admitted).is_none());
    assert!(llama.drafter_pairing_error(&admitted).is_none());

    // Refused: a tap AT the depth, and one above it. The message names
    // the offending index and the depth.
    for tap in [LAYERS, LAYERS + 1, usize::MAX] {
        let refused = drafter(vec![1, tap], vocab);
        for (family, reason) in [
            ("Qwen3", qwen3.drafter_pairing_error(&refused)),
            ("Llama", llama.drafter_pairing_error(&refused)),
        ] {
            let reason = reason.unwrap_or_else(|| panic!("{family} must refuse tap {tap}"));
            assert!(reason.contains(&format!("layer index {tap}")), "{reason}");
            assert!(reason.contains(&format!("{LAYERS} layers")), "{reason}");
            assert!(reason.contains(family), "{reason}");
            assert!(reason.contains("--model-draft"), "{reason}");
        }
    }

    // Refused: a drafter for another vocabulary, naming both numbers.
    let other_vocab = drafter(vec![1], Some(248_320));
    for reason in [
        qwen3.drafter_pairing_error(&other_vocab),
        llama.drafter_pairing_error(&other_vocab),
    ] {
        let reason = reason.expect("a vocabulary mismatch must be refused");
        assert!(reason.contains("248320"), "{reason}");
        assert!(reason.contains(&VOCAB.to_string()), "{reason}");
    }
    // A drafter that reports no vocabulary is not guessed at.
    assert!(
        qwen3
            .drafter_pairing_error(&drafter(vec![1], None))
            .is_none()
    );

    // The same two bounds hold for a plain DFlash drafter, which the
    // drafter-side gate (`validate_target_compat`) lets through.
    let plain = GateDrafter {
        dspark: false,
        taps: vec![1, LAYERS],
        vocab,
    };
    assert_eq!(DFlashDrafterFamily::of(&plain), DFlashDrafterFamily::Dflash);
    assert!(qwen3.drafter_pairing_error(&plain).is_some());
    assert!(llama.drafter_pairing_error(&plain).is_some());
}

/// The two dense targets take the DSpark burst policy: every prompt row
/// as the first draft context, B = 1 only, and fresh FP16 caches, one
/// per layer.
#[test]
fn qwen3_and_llama_targets_run_the_dspark_burst_policy() {
    use crate::models::dense_speculative::tests::{LAYERS, tiny_llama, tiny_qwen3};
    let _runtime = crate::initialize_runtime();
    assert_eq!(
        FirstHiddenRows::EveryPromptRow,
        <crate::models::Qwen3Model as DFlashTargetModel>::first_hidden_rows()
    );
    assert_eq!(
        FirstHiddenRows::EveryPromptRow,
        <crate::models::Llama3Model as DFlashTargetModel>::first_hidden_rows()
    );
    assert!(!<crate::models::Qwen3Model as DFlashTargetModel>::supports_batched());
    assert!(!<crate::models::Llama3Model as DFlashTargetModel>::supports_batched());

    let qwen3 = tiny_qwen3();
    let llama = tiny_llama();
    let caches = qwen3.make_dflash_caches();
    assert_eq!(caches.len(), LAYERS);
    assert!(caches.iter().all(|c| c.is_empty() && c.is_dense_fp16()));
    assert_eq!(llama.make_dflash_caches().len(), LAYERS);

    // Every other target's default answer is unchanged: no objection.
    let any = GateDrafter {
        dspark: true,
        taps: vec![usize::MAX],
        vocab: Some(1),
    };
    assert!(<crate::models::Qwen35Model as DFlashTargetModel>::required_drafter_family().is_none());
    // The decline text no longer calls every DSpark drafter an LFM2 one.
    let reason = required_family_pairing_error(
        Some(DFlashDrafterFamily::MuseAssistant),
        DFlashDrafterFamily::of(&any),
    )
    .expect("a Muse target with a DSpark drafter must decline");
    assert!(reason.contains("a DSpark drafter"), "{reason}");
    assert!(!reason.contains("LFM2 DSpark drafter"), "{reason}");
}
