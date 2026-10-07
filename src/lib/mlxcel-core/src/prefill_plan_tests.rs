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

use super::*;

fn tokens(align: bool) -> PrefillCaps {
    PrefillCaps {
        supports_chunked_prefill: true,
        supports_padded_prefill: true,
        supports_maskless_padded_prefill: false,
        align_prefill: align,
        input: PrefillInput::Tokens,
    }
}

fn ranges(plan: &PrefillPlan) -> Vec<Range<usize>> {
    plan.ranges()
}

/// The chunk gate applies only when configured, supported, and useful.
#[test]
fn chunking_applies_only_when_configured_supported_and_useful() {
    assert_eq!(
        PrefillPlan::new(8192, 2048, tokens(false)).chunk(),
        Some(2048)
    );
    // Prompt fits in one chunk: single pass.
    assert_eq!(PrefillPlan::new(2048, 2048, tokens(false)).chunk(), None);
    assert_eq!(PrefillPlan::new(1, 2048, tokens(false)).chunk(), None);
    // MLXCEL_PREFILL_CHUNK=0 forces single pass.
    assert_eq!(PrefillPlan::new(8192, 0, tokens(false)).chunk(), None);
    // Model opt-out wins regardless of configuration.
    let opt_out = PrefillCaps {
        supports_chunked_prefill: false,
        ..tokens(false)
    };
    assert_eq!(PrefillPlan::new(8192, 2048, opt_out).chunk(), None);
    assert!(PrefillPlan::new(8192, 2048, opt_out).is_single_pass());
}

#[test]
fn short_prompt_is_one_piece_and_one_token_gets_no_padding() {
    let plan = PrefillPlan::new(10, 2048, tokens(false));
    assert_eq!(ranges(&plan), vec![0..10]);
    assert!(plan.is_single_pass());
    assert_eq!(plan.trim_after(), None);

    let one = PrefillPlan::new(1, 2048, tokens(true));
    assert_eq!(
        one.pieces(),
        &[PrefillPiece {
            range: 0..1,
            padded_len: 32
        }]
    );
    assert_eq!(one.trim_after(), Some(31));
    // A single position pads to the tile like any other length; the executor
    // trims it. What never happens is a zero-length or negative piece.
    assert_eq!(PrefillPlan::new(0, 2048, tokens(true)).pieces(), &[]);
}

#[test]
fn chunks_cover_the_prompt_in_order_without_overlap() {
    let plan = PrefillPlan::new(1100, 512, tokens(false));
    assert_eq!(ranges(&plan), vec![0..512, 512..1024, 1024..1100]);
    assert!(!plan.is_single_pass());
    assert_eq!(plan.forwarded_len(), 1100);
    let last = plan.pieces().last().unwrap();
    assert!(plan.is_terminal(last));
    assert!(!plan.is_terminal(&plan.pieces()[0]));
}

/// The server partition the Gemma 4 MTP burst mirrors: the boundary segment is
/// one forward even past the chunk size, and the suffix chunks from the
/// boundary (`mtp_prefill_ranges_mirror_classic_history_boundary_split`).
#[test]
fn boundary_segment_is_one_unpadded_piece_and_chunks_restart_there() {
    let caps = tokens(false);
    assert_eq!(
        ranges(&PrefillPlan::with_prefix(40, 0, None, 512, caps)),
        vec![0..40]
    );
    assert_eq!(
        ranges(&PrefillPlan::with_prefix(40, 0, Some(34), 512, caps)),
        vec![0..34, 34..40]
    );
    assert_eq!(
        ranges(&PrefillPlan::with_prefix(1300, 0, Some(700), 512, caps)),
        vec![0..700, 700..1212, 1212..1300]
    );
    // An adopted prefix already past the boundary, or a boundary covering the
    // whole prompt, splits nothing.
    assert_eq!(
        ranges(&PrefillPlan::with_prefix(40, 36, Some(34), 512, caps)),
        vec![36..40]
    );
    assert_eq!(
        ranges(&PrefillPlan::with_prefix(40, 0, Some(40), 512, caps)),
        vec![0..40]
    );
    assert_eq!(
        ranges(&PrefillPlan::with_prefix(40, 10, Some(20), 0, caps)),
        vec![10..20, 20..40]
    );

    // With alignment on, the boundary piece stays unpadded while the suffix
    // piece pads: the snapshot must describe exactly the keyed tokens.
    let aligned = PrefillPlan::with_prefix(40, 0, Some(34), 512, tokens(true));
    assert_eq!(
        aligned.pieces(),
        &[
            PrefillPiece {
                range: 0..34,
                padded_len: 34
            },
            PrefillPiece {
                range: 34..40,
                padded_len: 32
            },
        ]
    );
    assert!(aligned.ends_at_boundary(&aligned.pieces()[0]));
    assert!(!aligned.ends_at_boundary(&aligned.pieces()[1]));
    assert_eq!(aligned.boundary(), Some(34));
}

#[test]
fn adopted_prefix_is_never_forwarded_and_chunks_start_after_it() {
    // Issue #179 turn 3: the full prompt clears the chunk threshold but a hit
    // adopted all but a short suffix. The one piece reaches the prompt end.
    let hit = PrefillPlan::with_prefix(593, 560, None, 512, tokens(false));
    assert_eq!(ranges(&hit), vec![560..593]);
    assert!(hit.is_single_pass());
    assert!(hit.is_terminal(&hit.pieces()[0]));
    assert_eq!(hit.adopted(), 560);
    assert_eq!(hit.forwarded_len(), 33);

    let cold = PrefillPlan::with_prefix(593, 0, None, 512, tokens(false));
    assert_eq!(ranges(&cold), vec![0..512, 512..593]);

    // Nothing to forward when the whole prompt is adopted.
    assert!(
        PrefillPlan::with_prefix(593, 593, None, 512, tokens(false))
            .pieces()
            .is_empty()
    );
    // An adopted length past the prompt clamps.
    assert_eq!(
        PrefillPlan::with_prefix(10, 50, None, 0, tokens(false)).adopted(),
        10
    );
}

#[test]
fn padding_follows_alignment_model_support_and_input_kind() {
    // Aligned token input pads every piece that is not tile-sized.
    let plan = PrefillPlan::new(70, 64, tokens(true));
    assert_eq!(
        plan.pieces(),
        &[
            PrefillPiece {
                range: 0..64,
                padded_len: 64
            },
            PrefillPiece {
                range: 64..70,
                padded_len: 32
            },
        ]
    );
    assert_eq!(plan.trim_after(), Some(26));
    assert_eq!(plan.pieces()[1].pad_excess(), 26);
    assert!(plan.pieces()[1].is_padded());
    assert!(!plan.pieces()[0].is_padded());
    assert_eq!(plan.pieces()[1].last_real_pos(), 5);
    assert!(plan.pad_mask_required());

    // Alignment off: no padding anywhere.
    assert_eq!(PrefillPlan::new(70, 64, tokens(false)).trim_after(), None);
    // Model opt-out: no padding.
    let no_pad = PrefillCaps {
        supports_padded_prefill: false,
        ..tokens(true)
    };
    assert_eq!(PrefillPlan::new(70, 64, no_pad).trim_after(), None);
    // Maskless opt-in: padded, but no array mask.
    let maskless = PrefillCaps {
        supports_maskless_padded_prefill: true,
        ..tokens(true)
    };
    let plan = PrefillPlan::new(5, 0, maskless);
    assert_eq!(plan.trim_after(), Some(27));
    assert!(!plan.pad_mask_required());
}

#[test]
fn embedding_input_is_never_chunked_and_padded_only_where_the_executor_pads() {
    let cli = tokens(true).with_input(PrefillInput::Embeddings {
        executor_pads: true,
    });
    let plan = PrefillPlan::new(300, 64, cli);
    assert!(cli.is_embedding_input());
    assert_eq!(plan.chunk(), None);
    assert_eq!(
        plan.pieces(),
        &[PrefillPiece {
            range: 0..300,
            padded_len: 320
        }]
    );

    let server = tokens(true).with_input(PrefillInput::Embeddings {
        executor_pads: false,
    });
    let plan = PrefillPlan::new(300, 64, server);
    assert_eq!(
        plan.pieces(),
        &[PrefillPiece {
            range: 0..300,
            padded_len: 300
        }]
    );
    // An embedding row never splits at the history boundary either: the
    // scheduler passes no boundary for it, and a boundary given anyway still
    // yields one segment plus the suffix, never chunks.
    let plan = PrefillPlan::with_prefix(300, 0, Some(100), 64, server);
    assert_eq!(ranges(&plan), vec![0..100, 100..300]);
}

#[test]
fn piece_lookup_by_cursor_drives_a_stepping_scheduler() {
    let plan = PrefillPlan::with_prefix(1300, 0, Some(700), 512, tokens(false));
    let mut cursor = 0;
    let mut seen = Vec::new();
    while let Some(piece) = plan.piece_starting_at(cursor) {
        seen.push(piece.range.clone());
        cursor = piece.range.end;
    }
    assert_eq!(seen, ranges(&plan));
    assert_eq!(cursor, 1300);
    assert!(plan.piece_starting_at(1300).is_none());
    assert!(
        plan.piece_starting_at(5).is_none(),
        "cursors land only on piece starts"
    );
    assert_eq!(plan.split_points(), vec![0, 700, 1212]);
}

/// The prompt-cache invariant of issue #2170: a hit forwards the same pieces
/// as the miss, and so reproduces it, exactly when its adopted prefix ends on
/// one of the miss's split points.
#[test]
fn a_hit_reproduces_the_miss_only_from_a_split_point() {
    let caps = tokens(false);
    // Snapshot family: the miss splits at the boundary, the hit adopts it.
    let miss = PrefillPlan::with_prefix(51, 0, Some(48), 2048, caps);
    let hit = PrefillPlan::with_prefix(51, 48, Some(48), 2048, caps);
    assert_eq!(ranges(&miss), vec![0..48, 48..51]);
    assert_eq!(ranges(&hit), vec![48..51]);
    assert!(hit.reproduces(&miss));

    // Dense-KV family: the miss is one forward, the hit adopts 45 of 52 and
    // forwards 7; the 7-row forward is not a piece of the miss.
    let miss = PrefillPlan::with_prefix(52, 0, None, 2048, caps);
    let hit = PrefillPlan::with_prefix(52, 45, None, 2048, caps);
    assert_eq!(ranges(&miss), vec![0..52]);
    assert_eq!(ranges(&hit), vec![45..52]);
    assert!(!hit.reproduces(&miss));

    // The same hit against a miss chunked at 45 does reproduce it (the
    // `--server-prefill-chunk 45` measurement on qwen3-1.7b-4bit).
    let miss = PrefillPlan::with_prefix(52, 0, None, 45, caps);
    assert_eq!(ranges(&miss), vec![0..45, 45..52]);
    assert!(hit.reproduces(&miss));

    // A hit inside a chunk does not; one on a chunk edge does.
    let miss = PrefillPlan::with_prefix(1100, 0, None, 512, caps);
    assert!(!PrefillPlan::with_prefix(1100, 600, None, 512, caps).reproduces(&miss));
    assert!(PrefillPlan::with_prefix(1100, 512, None, 512, caps).reproduces(&miss));
    // Different prompts never reproduce each other.
    assert!(!PrefillPlan::with_prefix(1101, 512, None, 512, caps).reproduces(&miss));
}

#[test]
fn chunk_policy_default_is_the_adr_0007_value() {
    assert_eq!(DEFAULT_PREFILL_CHUNK, 2048);
    // The env override is read once per process, so only the default can be
    // asserted here without disturbing sibling tests.
    assert!(
        prefill_chunk_len() == DEFAULT_PREFILL_CHUNK
            || std::env::var_os("MLXCEL_PREFILL_CHUNK").is_some()
    );
}
