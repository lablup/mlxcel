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

//! Every scheduler function that forwards a model is classified here as prefill
//! or decode, and prefill functions must announce the sequence's total position
//! span (#1358).
//!
//! This is a source-level guard, and it exists because the natural bug is an
//! omission rather than a wrong line. A model that picks its RoPE frequency
//! table from the whole prompt length (Phi-3 / Phi-4 LongRoPE) reads
//! [`mlxcel_core::prefill_span`]; a prefill forward that does not announce falls
//! back to its own `(cache_offset, seq_len)`, and for a prompt that crosses the
//! threshold that silently mixes two frequency tables into one KV cache. The
//! first fix for #1358 announced in the two chunked-prefill functions and missed
//! the prompt-cache ones, which is exactly how the server regressed while the
//! CLI passed every gate: `capture_history_boundary_snapshot` forwards
//! `prompt_tokens[start..boundary]`, a strict prefix that can sit below the
//! threshold while the prompt does not.
//!
//! No unit test on the scheduler catches that, because reaching those forwards
//! needs a live `BatchScheduler` and a real model. Reading the source does catch
//! it, and it also catches the next one: a forward added to a function this
//! table does not name fails the test until someone classifies it.

use std::collections::BTreeMap;

/// How a scheduler function that forwards the model relates to the prefill-span
/// announcement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpanDuty {
    /// Runs prefill work for one sequence, so it must announce.
    Announces,
    /// Decode. The pass's own offset is the position, and an announcement left
    /// live here would hand this sequence another one's prompt length.
    DecodeMustNotAnnounce,
    /// Prefill that deliberately does not announce, with the reason recorded at
    /// the call site.
    PrefillOptedOut,
}

use SpanDuty::{Announces, DecodeMustNotAnnounce, PrefillOptedOut};

/// The classification of every model forward under `src/server/batch/scheduler`.
///
/// Keyed by `(file, function)`. Adding a forward to a function that is not here
/// fails [`every_scheduler_model_forward_is_classified`].
const DUTIES: &[(&str, &str, SpanDuty)] = &[
    // Prefill: the sequence's own prefill work, all of which can reach the
    // model with a strict prefix or suffix of the prompt.
    ("prefill.rs", "execute_full_prefill", Announces),
    ("prefill.rs", "start_chunked_prefill", Announces),
    ("prefill.rs", "continue_chunked_prefill", Announces),
    (
        "prompt_cache.rs",
        "capture_history_boundary_snapshot",
        Announces,
    ),
    ("prompt_cache.rs", "run_next_prompt_cache_warmup", Announces),
    // Prefill, opted out: one batched pass from offset 0 over the padded cohort,
    // where the pass span already equals the longest row's prompt.
    ("prefill.rs", "run_padded_batched_prefill", PrefillOptedOut),
    // Decode.
    ("decode_tick.rs", "lookahead_forward", DecodeMustNotAnnounce),
    (
        "decode_tick.rs",
        "execute_batched_decode",
        DecodeMustNotAnnounce,
    ),
    (
        "decode_tick.rs",
        "decode_single_step",
        DecodeMustNotAnnounce,
    ),
];

/// The scheduler sources this guard reads.
fn sources() -> Vec<(&'static str, &'static str)> {
    vec![
        ("prefill.rs", include_str!("prefill.rs")),
        ("prompt_cache.rs", include_str!("prompt_cache.rs")),
        ("decode_tick.rs", include_str!("decode_tick.rs")),
    ]
}

/// Split one source file into its top-level `impl`-method bodies.
///
/// Methods are recognised by a `fn` at four-space indentation, which is what
/// every method in these files is; the body runs to the next such line. Test
/// modules sit deeper or in their own files, so they do not appear.
fn methods(source: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for line in source.lines() {
        if let Some(name) = method_name(line) {
            out.push((name, String::new()));
        }
        if let Some((_, body)) = out.last_mut() {
            body.push_str(line);
            body.push('\n');
        }
    }
    out
}

/// The method name declared by `line`, if it declares one at method indentation.
fn method_name(line: &str) -> Option<String> {
    let rest = line.strip_prefix("    ")?;
    if rest.starts_with(' ') {
        return None;
    }
    let rest = rest.strip_prefix("pub ").unwrap_or(rest);
    let rest = rest.strip_prefix("pub(crate) ").unwrap_or(rest);
    let rest = rest.strip_prefix("pub(super) ").unwrap_or(rest);
    let rest = rest.strip_prefix("async ").unwrap_or(rest);
    let rest = rest.strip_prefix("fn ")?;
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// The body with all whitespace removed, so a call rustfmt broke across lines
/// (`self\n.model\n.forward_with_sequence_id(`) reads the same as a one-line one.
fn collapsed(body: &str) -> String {
    body.split_whitespace().collect()
}

/// Whether a method body forwards the model.
fn forwards_the_model(collapsed_body: &str) -> bool {
    collapsed_body.contains("self.model.forward")
}

/// Whether a method body announces a prefill span.
fn announces(collapsed_body: &str) -> bool {
    collapsed_body.contains("self.announce_prefill_span(")
        || collapsed_body.contains("prefill_span::announce(")
}

/// Every function in these files that forwards the model, and whether it
/// announces.
fn observed() -> BTreeMap<(String, String), bool> {
    let mut found = BTreeMap::new();
    for (file, source) in sources() {
        for (name, body) in methods(source) {
            // The helper itself names the function without forwarding.
            if name == "announce_prefill_span" {
                continue;
            }
            let body = collapsed(&body);
            if forwards_the_model(&body) {
                found.insert((file.to_string(), name), announces(&body));
            }
        }
    }
    found
}

#[test]
fn every_scheduler_model_forward_is_classified() {
    let observed = observed();
    assert!(
        !observed.is_empty(),
        "the source scan found no model forwards at all, so the parser stopped matching this file's shape"
    );

    let classified: BTreeMap<(String, String), SpanDuty> = DUTIES
        .iter()
        .map(|(file, name, duty)| ((file.to_string(), name.to_string()), *duty))
        .collect();

    let unclassified: Vec<_> = observed
        .keys()
        .filter(|key| !classified.contains_key(*key))
        .collect();
    assert!(
        unclassified.is_empty(),
        "these scheduler functions forward the model and are not in DUTIES: {unclassified:?}. Decide whether each one runs prefill (announce the sequence span with `self.announce_prefill_span`) or decode (must not announce), then add it."
    );

    let vanished: Vec<_> = classified
        .keys()
        .filter(|key| !observed.contains_key(*key))
        .collect();
    assert!(
        vanished.is_empty(),
        "DUTIES names functions that no longer forward the model: {vanished:?}. Remove the stale entries so the table keeps describing the code."
    );
}

#[test]
fn every_prefill_forward_announces_the_sequence_span() {
    let observed = observed();
    for (file, name, duty) in DUTIES {
        let key = (file.to_string(), name.to_string());
        let Some(&announces) = observed.get(&key) else {
            continue;
        };
        match duty {
            Announces => assert!(
                announces,
                "{file}::{name} runs prefill work but never announces the sequence's total position span. A prompt that crosses a whole-prompt RoPE threshold will be rotated with two different tables into one KV cache (#1358). Call `self.announce_prefill_span(&seq)` before the forward."
            ),
            DecodeMustNotAnnounce => assert!(
                !announces,
                "{file}::{name} is decode and must not announce a prefill span: the announcement would outlive this call only if it leaked, and a decode step's position is its own cache offset (#1358)."
            ),
            PrefillOptedOut => assert!(
                !announces,
                "{file}::{name} is recorded as an announced-exempt prefill but now announces. Either move it to Announces or drop the announcement."
            ),
        }
    }
}

#[test]
fn the_prompt_cache_prefill_forwards_are_covered() {
    // Named explicitly because these two are the ones the first fix missed, and
    // the server regression they caused (a 5136-token prompt returning
    // repetition through POST /v1/completions while the CLI was correct) is not
    // visible from any unit test that does not read this file.
    let observed = observed();
    for name in [
        "capture_history_boundary_snapshot",
        "run_next_prompt_cache_warmup",
    ] {
        let key = ("prompt_cache.rs".to_string(), name.to_string());
        assert_eq!(
            observed.get(&key),
            Some(&true),
            "prompt_cache.rs::{name} must forward the model and announce the prefill span"
        );
    }
}
