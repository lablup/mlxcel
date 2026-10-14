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
    // An embedding row never splits at the history boundary either: a
    // boundary handed in anyway is ignored, so the cursor an executor advances
    // piece by piece cannot be left mid-embedding.
    for caps in [server, cli] {
        let plan = PrefillPlan::with_prefix(300, 0, Some(100), 64, caps);
        assert_eq!(plan.boundary(), None);
        assert_eq!(plan.ranges(), vec![0..300]);
        assert!(plan.is_single_pass());
        assert_eq!(plan.forwarded_len(), 300);
        assert!(!plan.ends_at_boundary(&plan.pieces()[0]));
    }
    // The same boundary still splits token input.
    let plan = PrefillPlan::with_prefix(300, 0, Some(100), 0, tokens(false));
    assert_eq!(plan.boundary(), Some(100));
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
fn resume_at_matches_the_fixed_chunk_suffix_at_every_split_point() {
    for chunk in [512, 2048] {
        for boundary in [None, Some(777)] {
            for align in [false, true] {
                let caps = tokens(align);
                let base = PrefillPlan::with_prefix(5000, 123, boundary, chunk, caps);
                let mut cursors = base.split_points();
                cursors.push(base.prompt_len());
                for cursor in cursors {
                    let resumed = PrefillPlan::resume_at(5000, 123, boundary, cursor, chunk, caps);
                    let expected: Vec<_> = base
                        .pieces()
                        .iter()
                        .filter(|piece| piece.range.start >= cursor)
                        .cloned()
                        .collect();
                    assert_eq!(
                        resumed.pieces(),
                        expected,
                        "chunk={chunk}, boundary={boundary:?}, align={align}, cursor={cursor}"
                    );
                    assert_eq!(resumed.adopted(), base.adopted());
                    assert_eq!(resumed.boundary(), base.boundary());
                    assert_eq!(resumed.forwarded_len(), base.forwarded_len());
                }
            }
        }
    }
}

#[test]
fn resume_at_repartitions_from_the_cursor_when_the_chunk_changes() {
    let plan = PrefillPlan::resume_at(8192, 0, None, 1024, 2048, tokens(false));
    assert_eq!(
        ranges(&plan),
        vec![1024..3072, 3072..5120, 5120..7168, 7168..8192]
    );
    assert_eq!(plan.adopted(), 0);
    assert_eq!(plan.forwarded_len(), 8192);
}

#[test]
fn resumed_last_piece_still_reports_the_active_chunk() {
    let plan = PrefillPlan::resume_at(1537, 0, None, 1536, 512, tokens(false));
    assert_eq!(ranges(&plan), vec![1536..1537]);
    assert_eq!(plan.chunk(), Some(512));
}

#[test]
fn next_piece_matches_resumed_plan_across_boundaries_cursors_chunks_and_padding() {
    let caps = [
        tokens(false),
        tokens(true),
        PrefillCaps {
            supports_padded_prefill: false,
            ..tokens(true)
        },
        PrefillCaps {
            supports_chunked_prefill: false,
            ..tokens(true)
        },
        tokens(true).with_input(PrefillInput::Embeddings {
            executor_pads: false,
        }),
        tokens(true).with_input(PrefillInput::Embeddings {
            executor_pads: true,
        }),
    ];
    for (prompt_len, adopted, boundary) in [
        (0_usize, 0_usize, None),
        (70, 0, None),
        (1300, 0, Some(700)),
        (5000, 123, Some(777)),
    ] {
        let mut cursors = vec![
            adopted,
            boundary.unwrap_or(adopted),
            1024,
            prompt_len.saturating_sub(1),
            prompt_len,
        ];
        cursors.sort_unstable();
        cursors.dedup();
        for caps in caps {
            for chunk in [0, 64, 512, 2048] {
                for &cursor in &cursors {
                    if cursor < adopted
                        || cursor > prompt_len
                        || (cursor > adopted && boundary.is_some_and(|boundary| cursor < boundary))
                    {
                        continue;
                    }
                    let plan =
                        PrefillPlan::resume_at(prompt_len, adopted, boundary, cursor, chunk, caps);
                    let expected = plan.piece_starting_at(cursor).cloned();
                    assert_eq!(
                        PrefillPlan::next_piece_at(
                            prompt_len, adopted, boundary, cursor, chunk, caps,
                        ),
                        expected,
                        "prompt_len={prompt_len}, adopted={adopted}, boundary={boundary:?}, cursor={cursor}, chunk={chunk}, caps={caps:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn chunk_policy_resolves_adaptive_defaults_and_explicit_overrides() {
    assert_eq!(
        PrefillChunkPolicy::resolve(None),
        PrefillChunkPolicy {
            alone: DEFAULT_PREFILL_CHUNK,
            contended: CONTENDED_PREFILL_CHUNK,
        }
    );
    for chunk in [1024, 0] {
        let policy = PrefillChunkPolicy::resolve(Some(chunk));
        assert_eq!(policy.alone, chunk);
        assert_eq!(policy.contended, chunk);
        assert_eq!(policy.chunk(false), chunk);
        assert_eq!(policy.chunk(true), chunk);
    }
}

#[test]
fn chunk_policy_default_is_the_adr_0007_value() {
    assert_eq!(DEFAULT_PREFILL_CHUNK, 2048);
    assert_eq!(CONTENDED_PREFILL_CHUNK, 512);
    // The env override is read once per process, so only the default can be
    // asserted here without disturbing sibling tests.
    assert!(
        prefill_chunk_len() == DEFAULT_PREFILL_CHUNK
            || std::env::var_os("MLXCEL_PREFILL_CHUNK").is_some()
    );
}

#[test]
fn prefill_chunk_override_is_read_once_in_a_fresh_process() {
    const CHILD: &str = "MLXCEL_PREFILL_CHUNK_TEST_CHILD";
    if std::env::var_os(CHILD).is_some() {
        assert_eq!(prefill_chunk_override(), Some(1024));
        // SAFETY: this child process runs only this exact test, with one test
        // thread, and no other thread reads or writes this environment key.
        unsafe { std::env::set_var("MLXCEL_PREFILL_CHUNK", "512") };
        assert_eq!(prefill_chunk_override(), Some(1024));
        assert_eq!(prefill_chunk_len(), 1024);
        return;
    }

    let output = std::process::Command::new(std::env::current_exe().expect("current test binary"))
        .args([
            "--exact",
            "prefill_plan::tests::prefill_chunk_override_is_read_once_in_a_fresh_process",
            "--test-threads=1",
        ])
        .env(CHILD, "1")
        .env("MLXCEL_PREFILL_CHUNK", "1024")
        .output()
        .expect("spawn isolated prefill override test");
    assert!(
        output.status.success(),
        "isolated test failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("1 passed"),
        "isolated test did not run the exact case:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

/// A `tracing` subscriber that records every WARN-or-worse event message.
struct WarnCollector(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

impl tracing::Subscriber for WarnCollector {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Message(String);
        impl tracing::field::Visit for Message {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0 = format!("{value:?}");
                }
            }
        }
        if *event.metadata().level() <= tracing::Level::WARN {
            let mut message = Message(String::new());
            event.record(&mut message);
            self.0.lock().unwrap().push(message.0);
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// `MLXCEL_PREFILL_CHUNK` parses as a token count (`0` disables chunking),
/// and an unparseable value is treated as unset with exactly one warning that
/// names the variable and the value, instead of silently.
#[test]
fn unparseable_chunk_env_warns_once_and_falls_back_to_the_default() {
    let warnings = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let (unset, zero, padded, bad_text, negative) =
        tracing::subscriber::with_default(WarnCollector(warnings.clone()), || {
            (
                parse_prefill_chunk(None),
                parse_prefill_chunk(Some("0")),
                parse_prefill_chunk(Some(" 512 ")),
                parse_prefill_chunk(Some("2k")),
                parse_prefill_chunk(Some("-1")),
            )
        });
    assert_eq!(unset, None);
    assert_eq!(zero, Some(0));
    assert_eq!(padded, Some(512));
    assert_eq!(bad_text, None);
    assert_eq!(negative, None);
    let warnings = warnings.lock().unwrap();
    assert_eq!(
        warnings.len(),
        2,
        "one warning per bad value, none for good ones: {warnings:?}"
    );
    assert!(warnings[0].contains("MLXCEL_PREFILL_CHUNK") && warnings[0].contains("2k"));
    assert!(warnings[1].contains("MLXCEL_PREFILL_CHUNK") && warnings[1].contains("-1"));
}
