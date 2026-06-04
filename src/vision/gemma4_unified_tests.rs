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

//! Unit tests for Gemma 4 Unified mask/token-type derivation.

use super::{UnifiedTokenIds, compute_vision_block_ids, derive_mm_token_type_ids, token_type};

const IDS: UnifiedTokenIds = UnifiedTokenIds {
    image: 258_880,
    video: 258_884,
    audio: 258_881,
};

#[test]
fn mm_token_type_ids_classifies_each_modality() {
    // text BOI image image EOI text audio video
    let input = vec![1, 255_999, 258_880, 258_880, 258_882, 7, 258_881, 258_884];
    let types = derive_mm_token_type_ids(&input, IDS);
    assert_eq!(
        types,
        vec![
            token_type::TEXT,  // 1
            token_type::TEXT,  // BOI (255999) is not a soft token
            token_type::IMAGE, // 258880
            token_type::IMAGE, // 258880
            token_type::TEXT,  // EOI (258882)
            token_type::TEXT,  // 7
            token_type::AUDIO, // 258881
            token_type::VIDEO, // 258884
        ]
    );
}

#[test]
fn block_ids_image_only_are_bidirectional_intra_block() {
    // Two separate image spans → two distinct block ids; text gets -1.
    // text img img text img img img text
    let input = vec![1, 258_880, 258_880, 5, 258_880, 258_880, 258_880, 9];
    let block = compute_vision_block_ids(&input, IDS, true).expect("vision present, no audio");
    assert_eq!(block, vec![-1, 0, 0, -1, 1, 1, 1, -1]);
}

#[test]
fn block_ids_video_counts_as_vision() {
    // A contiguous video span forms one bidirectional block.
    let input = vec![1, 258_884, 258_884, 258_884, 9];
    let block = compute_vision_block_ids(&input, IDS, true).expect("video present");
    assert_eq!(block, vec![-1, 0, 0, 0, -1]);
}

#[test]
fn block_ids_disabled_when_audio_present() {
    // image + audio → fully causal (overlay disabled per issue §6).
    let input = vec![1, 258_880, 258_880, 258_881, 9];
    assert!(
        compute_vision_block_ids(&input, IDS, true).is_none(),
        "audio token present must force fully-causal (None) masks",
    );
}

#[test]
fn block_ids_none_for_text_only() {
    let input = vec![1, 2, 3, 4, 5];
    assert!(compute_vision_block_ids(&input, IDS, true).is_none());
}

#[test]
fn block_ids_none_on_decode_single_token() {
    // seq_len == 1 (decode) → no overlay.
    let input = vec![258_880];
    assert!(compute_vision_block_ids(&input, IDS, true).is_none());
}

#[test]
fn block_ids_none_when_bidirectional_disabled() {
    // A checkpoint without use_bidirectional_attention == "vision" never builds
    // the overlay even with vision tokens present.
    let input = vec![1, 258_880, 258_880, 9];
    assert!(compute_vision_block_ids(&input, IDS, false).is_none());
}

#[test]
fn block_ids_adjacent_spans_separated_by_eoi_boi_are_distinct() {
    // ...img img EOI BOI img img... — the EOI/BOI between spans break the run
    // so the two image groups get separate block ids.
    let input = vec![258_880, 258_880, 258_882, 255_999, 258_880, 258_880];
    let block = compute_vision_block_ids(&input, IDS, true).unwrap();
    assert_eq!(block, vec![0, 0, -1, -1, 1, 1]);
}
