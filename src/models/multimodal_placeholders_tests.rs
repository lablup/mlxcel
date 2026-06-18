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

use super::MultimodalPlaceholderTokens;

#[test]
fn empty_struct_yields_no_suppressed_ids() {
    assert!(
        MultimodalPlaceholderTokens::default()
            .suppressed_ids()
            .is_empty()
    );
}

#[test]
fn gemma4_unified_12b_config_values_produce_all_seven_placeholders() {
    // The exact ids from models/gemma-4-12b-it-4bit/config.json (issue #350).
    let tokens = MultimodalPlaceholderTokens {
        audio_token_id: Some(258_881),
        image_token_id: Some(258_880),
        video_token_id: Some(258_884),
        boa_token_id: Some(256_000),
        boi_token_id: Some(255_999),
        eoa_token_id: Some(258_883),
        eoi_token_id: Some(258_882),
    };
    // Sorted, deduplicated.
    assert_eq!(
        tokens.suppressed_ids(),
        vec![
            255_999, 256_000, 258_880, 258_881, 258_882, 258_883, 258_884
        ]
    );
}

#[test]
fn real_eos_ids_are_never_in_the_suppressed_set() {
    // The gemma-4-12b config also carries eos_token_id = [1, 106, 50]; those
    // must stay generatable. The placeholder struct intentionally has no EOS
    // field, so the derived set can never contain them.
    let tokens = MultimodalPlaceholderTokens {
        audio_token_id: Some(258_881),
        image_token_id: Some(258_880),
        video_token_id: Some(258_884),
        boa_token_id: Some(256_000),
        boi_token_id: Some(255_999),
        eoa_token_id: Some(258_883),
        eoi_token_id: Some(258_882),
    };
    let suppressed = tokens.suppressed_ids();
    for eos in [1, 106, 50] {
        assert!(
            !suppressed.contains(&eos),
            "real EOS id {eos} must not be suppressed"
        );
    }
}

#[test]
fn unset_fields_are_skipped() {
    // A vision-only model leaves the audio markers unset.
    let tokens = MultimodalPlaceholderTokens {
        image_token_id: Some(258_880),
        boi_token_id: Some(255_999),
        eoi_token_id: Some(258_882),
        ..Default::default()
    };
    assert_eq!(tokens.suppressed_ids(), vec![255_999, 258_880, 258_882]);
}

#[test]
fn negative_sentinel_ids_are_dropped() {
    let tokens = MultimodalPlaceholderTokens {
        image_token_id: Some(258_880),
        audio_token_id: Some(-1),
        ..Default::default()
    };
    assert_eq!(tokens.suppressed_ids(), vec![258_880]);
}

#[test]
fn duplicate_ids_are_deduplicated() {
    // Some configs alias two markers to the same id; the set stays unique.
    let tokens = MultimodalPlaceholderTokens {
        image_token_id: Some(258_880),
        video_token_id: Some(258_880),
        ..Default::default()
    };
    assert_eq!(tokens.suppressed_ids(), vec![258_880]);
}
