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

//! Throughput comparison for the per-row DFlash accept port (issue #2442).
//!
//! Kept in its own module so the parity gate's command
//! (`--test speculative_parity qwen35_dflash_batched`) does not pull a
//! multi-minute benchmark into the correctness run.
use super::qwen35_dflash_harness::{TEXTS, mean_accepts, run_burst, run_burst_with_policy};
use super::{REACHABLE_PAIRINGS, pairing_present};

/// Issue #2442 gate 2: aggregate decode throughput of the per-row accept port
/// against the uniform clamp, at B = 4, three runs per arm, alternating.
///
/// Both arms run in one process on the same prompts and the same `[B, L]`
/// prefill shape; only the accept policy differs. The reported number is the
/// round-loop wall clock's tokens/sec (`DFlashBatchedRunOutput::stats`), so a
/// change in either the number of rounds or the cost per round shows up.
/// Prints the walk and committed accept counts the #2442 report needs.
///
/// Run on a quiet box (`caffeinate -i` plus
/// `scripts/with_indexers_paused.sh`); this test fails only when an arm
/// cannot run. The ship/keep verdict is made from the printed numbers.
#[test]
#[ignore = "real-model heavy (Qwen3.5-4B target + DFlash drafter, B=4 clamp vs per-row throughput); quiet-box manual run"]
fn throughput_dflash_qwen35_batched_clamp_vs_per_row() {
    use mlxcel::{LoadedModel, initialize_runtime, load_model};
    use mlxcel_core::drafter::{DrafterKind, load_drafter};

    fn median(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    }

    let pairing = &REACHABLE_PAIRINGS[0];
    let (target_path, draft_path, present) = pairing_present(pairing);
    if !present {
        eprintln!(
            "Skipping {} (throughput): target={:?} draft={:?}",
            pairing.name, target_path, draft_path,
        );
        return;
    }

    super::common::apply_server_mlx_cache_defaults();
    let _runtime = initialize_runtime();
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    let (loaded, tokenizer) = load_model(&target_path).expect("target model must load");
    let model: &mlxcel::models::Qwen35Model = match &loaded {
        LoadedModel::Qwen35(m) | LoadedModel::Qwen35Moe(m) => m,
        LoadedModel::Qwen35VLM(vlm) | LoadedModel::Qwen35MoeVLM(vlm) => &vlm.text_model,
        _ => panic!("DFlash pairing requires a Qwen 3.5 family target"),
    };
    let encoded: Vec<Vec<i32>> = TEXTS
        .iter()
        .map(|t| {
            tokenizer
                .encode(t, true)
                .expect("prompt must tokenize")
                .into_iter()
                .map(|id| id as i32)
                .collect()
        })
        .collect();
    let len = encoded.iter().map(Vec::len).min().expect("four prompts");
    let prompts: Vec<Vec<i32>> = encoded.into_iter().map(|p| p[..len].to_vec()).collect();

    let (mut drafter, kind) =
        load_drafter(&draft_path, Some(DrafterKind::Dflash)).expect("DFlash drafter must load");
    assert_eq!(kind, DrafterKind::Dflash);
    let block_size = pairing.block_size;

    // Warm up both arms so kernel selection and allocation are not measured.
    for per_row in [false, true] {
        let (_run, d) = run_burst_with_policy(model, drafter, &prompts, block_size, Some(per_row));
        drafter = d;
    }

    // B = 1 reference, for context only.
    {
        let (run, d) = run_burst(
            model,
            drafter,
            std::slice::from_ref(&prompts[0]),
            block_size,
        );
        drafter = d;
        let (walk, committed) = mean_accepts(&run);
        eprintln!(
            "[tp B=1] {:.2} tok/s, mean accept walk {walk:.3} committed {committed:.3}",
            run.aggregate_tok_per_sec()
        );
    }

    let mut clamp_tps = Vec::new();
    let mut port_tps = Vec::new();
    let mut clamp_accepts = (0.0, 0.0);
    let mut port_accepts = (0.0, 0.0);
    for round in 0..3 {
        for (label, per_row) in [("clamp", false), ("port", true)] {
            let (run, d) =
                run_burst_with_policy(model, drafter, &prompts, block_size, Some(per_row));
            drafter = d;
            let (walk, committed) = mean_accepts(&run);
            let tps = run.aggregate_tok_per_sec();
            eprintln!(
                "[tp B=4 {label} run {round}] {tps:.2} tok/s, rounds {}, clamped {}, \
                 mean accept walk {walk:.3} committed {committed:.3}",
                run.accept_lens[0].len(),
                run.clamped_rounds,
            );
            if per_row {
                port_tps.push(tps);
                port_accepts = (walk, committed);
            } else {
                clamp_tps.push(tps);
                clamp_accepts = (walk, committed);
            }
        }
    }
    let clamp = median(&clamp_tps);
    let port = median(&port_tps);
    eprintln!(
        "[tp B=4 VERDICT] clamp median {clamp:.2} tok/s (runs {clamp_tps:?}), \
         port median {port:.2} tok/s (runs {port_tps:?}), ratio {:.3}x",
        port / clamp,
    );
    eprintln!(
        "[tp B=4 VERDICT] committed accepts per row-round: clamp {:.3} (walk {:.3}), \
         port {:.3} (walk {:.3})",
        clamp_accepts.1, clamp_accepts.0, port_accepts.1, port_accepts.0,
    );
}
