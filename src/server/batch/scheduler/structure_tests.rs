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

#[test]
fn scheduler_modules_stay_below_documented_anti_pattern_threshold() {
    const MAX_LINES_WITHOUT_JUSTIFICATION: usize = 2_000;

    let batch_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/server/batch");
    let scheduler_dir = batch_dir.join("scheduler");
    let mut paths = vec![batch_dir.join("scheduler.rs")];
    let entries = std::fs::read_dir(&scheduler_dir)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", scheduler_dir.display()));
    let mut oversized = Vec::new();
    for entry in entries {
        let path = entry.expect("directory entry").path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        paths.push(path);
    }

    for path in paths {
        let source =
            std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {path:?}: {err}"));
        let line_count = source.lines().count();
        if line_count > MAX_LINES_WITHOUT_JUSTIFICATION {
            oversized.push(format!("{} has {line_count} lines", path.display()));
        }
    }

    assert!(
        oversized.is_empty(),
        "scheduler module files must stay at or below {MAX_LINES_WITHOUT_JUSTIFICATION} lines unless docs/code-guidelines.md records a specific exception: {}",
        oversized.join(", ")
    );
}

fn collapsed(source: &str) -> String {
    source.split_whitespace().collect()
}

#[test]
fn shared_budget_enforcement_stays_wired_into_scheduler_paths() {
    let admission = collapsed(include_str!("admission.rs"));
    let prefill = collapsed(include_str!("prefill.rs"));
    let decode = collapsed(include_str!("decode_tick.rs"));
    let speculative = collapsed(include_str!("speculative_finalize.rs"));
    let worker = collapsed(include_str!("../../model_worker.rs"));

    assert!(
        admission.contains("!self.shared_budget_admits_prompt(prompt_tokens.len())")
            && admission.contains("Self::send_shared_budget_rejection("),
        "request admission must reject prompts that do not fit the unified shared-token budget"
    );
    assert!(
        prefill.contains("!self.shared_budget_admits_prompt(seq.prompt_tokens.len())")
            && prefill.contains(
                "!self.shared_budget_has_prefill_first_token_room(seq.prompt_tokens.len())"
            ),
        "prefill must re-check unified budget at dequeue and before committing the first sampled token"
    );
    assert!(
        decode.contains("execute_batched_decode(&mutself,seq_ids:&[SequenceId]){ifseq_ids.is_empty()")
            && decode.contains("!self.shared_budget_has_decode_room(seq_ids.len())")
            && decode.contains("decode_single_step(&mutself,seq_id:SequenceId){if!self.shared_budget_has_decode_room(1)"),
        "batched and single-row decode must enforce unified budget before appending the next token"
    );
    assert!(
        speculative.contains("ifself.shared_kv_budget().is_some(){returnSome(seq);}"),
        "speculative burst loops must route to classic scheduler paths while unified budget is active"
    );
    assert!(
        worker.contains(
            ".with_shared_kv_budget(ifsched_config.kv_unified{effective_max_kv_size}else{None})"
        ),
        "model worker must install the resolved unified context window as the shared scheduler budget"
    );
}
