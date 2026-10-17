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

//! Logit parity for a `model.embed_tokens` LoRA pair across the three load
//! paths (issue #2421).
//!
//! mlx-lm's `LoRAEmbedding` adapts the lookup and, on a tied checkpoint, the
//! output head through the same matrix. The fused path (`--adapter`), the
//! unfused runtime path (`--lora`) and the pipeline stage loader (`--pp-size N
//! --adapter`) must therefore produce the same logits for one adapter, on a
//! tied checkpoint (both ends adapted) and on an untied one (the lookup only).
//! Before #2421 every one of these loads failed: the tied checkpoint refused
//! the pair, the untied one fused it on the fused path and failed as unclaimed
//! on the runtime path, and the mlx-lm layout failed a shape check.
//!
//! The tiny-fixture tests run in the normal suite. The real-checkpoint gate at
//! the bottom is `#[ignore]`d and driven by environment variables; it is how
//! the PR's Llama, Qwen2 and Qwen3 numbers were taken.

use std::path::{Path, PathBuf};

use mlxcel_core::UniquePtr;
use mlxcel_core::generate::LanguageModel;

use super::*;
use crate::distributed::pipeline::{
    LoadedStageExecutor, StageAssignment, StageExecutionInput, StageExecutor,
};
use crate::lora::test_support::{
    TINY_HIDDEN, TINY_VOCAB, TinyLlama, reseed_tiny_checkpoint, seeded_tensor, seeded_values,
    temp_dir, tensor_from_values, write_adapter_dir, write_tiny_llama_checkpoint,
    write_tiny_tokenizer,
};
use crate::lora::{LoraAdapterSpec, RuntimeLoraSet};

const H: usize = TINY_HIDDEN;
const V: usize = TINY_VOCAB;
const RANK: usize = 2;
const PROMPT: [i32; 4] = [3, 17, 5, 30];

fn checkpoint(name: &str, tied: bool) -> PathBuf {
    let dir = temp_dir(name);
    write_tiny_llama_checkpoint(
        &dir,
        &TinyLlama {
            tie_word_embeddings: Some(tied),
            stores_lm_head: !tied,
            unread: Vec::new(),
        },
    );
    reseed_tiny_checkpoint(&dir);
    write_tiny_tokenizer(&dir);
    dir
}

/// The three on-disk layouts an embedding pair is read in, each written so
/// that its delta is the same `[vocab, hidden]` matrix `A @ B`.
#[derive(Debug, Clone, Copy)]
enum Layout {
    /// mlx-lm `LoRAEmbedding`: `a = A [vocab, rank]`, `b = B [rank, hidden]`.
    MlxLmEmbedding,
    /// The `[hidden] -> [vocab]` linear in mlx-lm orientation:
    /// `a = Bᵀ [hidden, rank]`, `b = Aᵀ [rank, vocab]`.
    LinearMlxLm,
    /// The same linear in PEFT orientation: `a = B [rank, hidden]`,
    /// `b = A [vocab, rank]`.
    LinearPeft,
}

const LAYOUTS: [Layout; 3] = [
    Layout::MlxLmEmbedding,
    Layout::LinearMlxLm,
    Layout::LinearPeft,
];

fn transpose(rows: usize, cols: usize, values: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; values.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = values[r * cols + c];
        }
    }
    out
}

/// An adapter with a seeded `model.embed_tokens` pair in `layout` and a seeded
/// `q_proj` pair, so the embedding term is not the only one.
fn embedding_adapter(name: &str, layout: Layout) -> PathBuf {
    let a = seeded_values(V, RANK, 11); // A [vocab, rank]
    let b = seeded_values(RANK, H, 12); // B [rank, hidden]
    let (lora_a, lora_b) = match layout {
        Layout::MlxLmEmbedding => (
            tensor_from_values(V, RANK, &a),
            tensor_from_values(RANK, H, &b),
        ),
        Layout::LinearMlxLm => (
            tensor_from_values(H, RANK, &transpose(RANK, H, &b)),
            tensor_from_values(RANK, V, &transpose(V, RANK, &a)),
        ),
        Layout::LinearPeft => (
            tensor_from_values(RANK, H, &b),
            tensor_from_values(V, RANK, &a),
        ),
    };
    let mut tensors = std::collections::HashMap::new();
    tensors.insert("model.embed_tokens.lora_a".to_string(), lora_a);
    tensors.insert("model.embed_tokens.lora_b".to_string(), lora_b);
    tensors.insert(
        "model.layers.0.self_attn.q_proj.lora_a".to_string(),
        seeded_tensor(H, RANK, 13),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.lora_b".to_string(),
        seeded_tensor(RANK, H, 14),
    );
    let dir = temp_dir(name);
    write_adapter_dir(&dir, "lora", RANK, tensors);
    dir
}

fn spec(path: &Path) -> LoraAdapterSpec {
    LoraAdapterSpec {
        path: path.to_path_buf(),
        scale: 1.0,
        apply: true,
    }
}

fn prompt_ids(ids: &[i32]) -> UniquePtr<mlxcel_core::MlxArray> {
    mlxcel_core::from_slice_i32(ids, &[1, ids.len() as i32])
}

/// The last position's logits for `ids`, evaluated.
fn last_logits(model: &LoadedModel, ids: &[i32]) -> UniquePtr<mlxcel_core::MlxArray> {
    let mut caches = model.make_caches();
    let logits = model.forward(&prompt_ids(ids), &mut caches, None);
    let last = mlxcel_core::generate::logits_at_position(&logits, ids.len() - 1);
    mlxcel_core::eval(&last);
    last
}

fn stage_last_logits(
    model_dir: &Path,
    adapter: Option<&Path>,
    num_layers: usize,
    stages: usize,
    ids: &[i32],
) -> UniquePtr<mlxcel_core::MlxArray> {
    let per_stage = num_layers.div_ceil(stages);
    let mut hidden: Option<UniquePtr<mlxcel_core::MlxArray>> = None;
    let input = prompt_ids(ids);
    for stage_index in 0..stages {
        let start = stage_index * per_stage;
        let end = ((stage_index + 1) * per_stage).min(num_layers);
        let assignment = StageAssignment {
            stage_index,
            device_id: format!("test-{stage_index}"),
            layer_range: start..end,
            has_embedding: stage_index == 0,
            has_lm_head: stage_index + 1 == stages,
            estimated_memory_bytes: 0,
        };
        let stage = LoadedStageExecutor::load_with_adapter(model_dir, &assignment, adapter)
            .expect("the pipeline stage loads the adapter");
        let mut caches = stage.make_caches();
        let stage_input = match &hidden {
            None => StageExecutionInput::TokenIds(&input),
            Some(h) => StageExecutionInput::HiddenStates(h),
        };
        let output = stage
            .execute(stage_input, &mut caches, None)
            .expect("the stage executes");
        if stage_index + 1 == stages {
            let logits = output.into_logits().expect("the last stage returns logits");
            let last = mlxcel_core::generate::logits_at_position(&logits, ids.len() - 1);
            mlxcel_core::eval(&last);
            return last;
        }
        let h = output.into_hidden_states().expect("hidden states");
        mlxcel_core::eval(&h);
        hidden = Some(h);
    }
    unreachable!("at least one stage runs")
}

fn values(array: &mlxcel_core::MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(array)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn top_k(logits: &[f32], k: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..logits.len()).collect();
    ids.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    ids.truncate(k);
    ids
}

/// Whether two logit vectors agree on the top 5 up to near-ties: every id in
/// either top 5 scores, in the other vector, within `tolerance` of that
/// vector's fifth logit. With bf16 logits two paths that differ by one ulp can
/// swap ids whose logits tie to the ulp, which is not a disagreement.
fn top5_agree(a: &[f32], b: &[f32], tolerance: f32) -> bool {
    let covers = |x: &[f32], y: &[f32]| {
        let fifth = x[top_k(x, 5)[4]];
        top_k(y, 5).iter().all(|&id| x[id] >= fifth - tolerance)
    };
    covers(a, b) && covers(b, a)
}

fn fused_logits(model: &Path, adapter: &Path) -> Vec<f32> {
    let (fused, _) = load_model_with_adapter(model, adapter).expect("the fused path loads");
    values(&last_logits(&fused, &PROMPT))
}

fn runtime_model(model: &Path, adapter: &Path) -> (LoadedModel, RuntimeLoraSet) {
    let set = RuntimeLoraSet::from_specs(&[spec(adapter)]).expect("a lora adapter config");
    let (runtime, _) = load_model_with_adapter_specs(model, &[spec(adapter)], Some(&set))
        .expect("the runtime path loads");
    assert!(!mlxcel_core::runtime_lora::has_pending());
    (runtime, set)
}

fn cleanup(dirs: &[&Path]) {
    for dir in dirs {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// Fused, runtime and stage logits for one layout, with the base logits.
fn three_paths(model: &Path, adapter: &Path) -> [Vec<f32>; 4] {
    let (base, _) = load_model(model).expect("the base checkpoint loads");
    let base = values(&last_logits(&base, &PROMPT));
    let fused = fused_logits(model, adapter);
    let (runtime, _set) = runtime_model(model, adapter);
    let runtime = values(&last_logits(&runtime, &PROMPT));
    let stage = values(&stage_last_logits(model, Some(adapter), 1, 1, &PROMPT));
    [base, fused, runtime, stage]
}

/// The tolerance between paths on the f32 fixture: the fused table and the
/// runtime terms are the same arithmetic in a different association.
const F32_TOLERANCE: f32 = 1e-4;

fn assert_paths_agree(case: &str, [base, fused, runtime, stage]: &[Vec<f32>; 4]) {
    assert!(
        max_abs_diff(base, fused) > 1e-2,
        "{case}: the adapter must change the logits"
    );
    let runtime_diff = max_abs_diff(fused, runtime);
    assert!(
        runtime_diff < F32_TOLERANCE,
        "{case}: runtime differs from fused by {runtime_diff}\n fused   {fused:?}\n runtime {runtime:?}"
    );
    let stage_diff = max_abs_diff(fused, stage);
    assert!(
        stage_diff < F32_TOLERANCE,
        "{case}: the pipeline stage differs from fused by {stage_diff}"
    );
}

#[test]
fn every_load_path_applies_a_tied_embedding_pair_at_both_ends() {
    // On a tied checkpoint the fused path folds the pair into the one matrix
    // the lookup and the head read. The runtime path matches it only if the
    // lookup term and the tied-head term are both applied, and every layout
    // canonicalizes to the same delta.
    let model = checkpoint("embed_parity_tied", true);
    let mut reference: Option<Vec<f32>> = None;
    for layout in LAYOUTS {
        let adapter = embedding_adapter("embed_parity_tied_adapter", layout);
        let paths = three_paths(&model, &adapter);
        assert_paths_agree(&format!("tied {layout:?}"), &paths);
        let fused = paths[1].clone();
        match &reference {
            None => reference = Some(fused),
            Some(reference) => assert!(
                max_abs_diff(reference, &fused) < F32_TOLERANCE,
                "{layout:?} must apply the same delta as the mlx-lm layout"
            ),
        }
        cleanup(&[&adapter]);
    }
    cleanup(&[&model]);
}

#[test]
fn every_load_path_adapts_only_the_lookup_on_an_untied_checkpoint() {
    // The untied head is its own matrix, so the fused path adapts the lookup
    // alone; a runtime path that also applied the head term would differ.
    let model = checkpoint("embed_parity_untied", false);
    for layout in [Layout::MlxLmEmbedding, Layout::LinearMlxLm] {
        let adapter = embedding_adapter("embed_parity_untied_adapter", layout);
        let paths = three_paths(&model, &adapter);
        assert_paths_agree(&format!("untied {layout:?}"), &paths);
        cleanup(&[&adapter]);
    }
    cleanup(&[&model]);
}

#[test]
fn a_tied_runtime_embedding_term_at_scale_zero_is_byte_identical_to_base() {
    // `POST /lora-adapters` writes the shared handle both ends read; at zero
    // neither the lookup nor the head adds an operation.
    let model = checkpoint("embed_scale_zero", true);
    let adapter = embedding_adapter("embed_scale_zero_adapter", Layout::MlxLmEmbedding);
    let (base, _) = load_model(&model).expect("the base checkpoint loads");
    let base = last_logits(&base, &PROMPT);
    let (runtime, set) = runtime_model(&model, &adapter);

    let adapted = values(&last_logits(&runtime, &PROMPT));
    assert!(max_abs_diff(&values(&base), &adapted) > 1e-2);

    set.apply_scales(&[0.0]);
    let zero = last_logits(&runtime, &PROMPT);
    assert_eq!(
        mlxcel_core::array_to_raw_bytes(&zero),
        mlxcel_core::array_to_raw_bytes(&base),
        "scale zero must serve the base logits exactly"
    );

    set.apply_scales(&[1.0]);
    let restored = values(&last_logits(&runtime, &PROMPT));
    assert_eq!(
        adapted, restored,
        "scaling back restores the adapted logits"
    );

    cleanup(&[&model, &adapter]);
}

// Real-checkpoint gate (issue #2421). `#[ignore]`d: it loads one real
// checkpoint four times. Inputs:
//
// - `MLXCEL_TEST_EMBED_LORA_MODEL`: the model directory.
// - `MLXCEL_TEST_EMBED_LORA_ADAPTER`: an mlx-lm adapter directory with a
//   `model.embed_tokens` pair. Without it only the base logits are taken
//   (for a checkpoint `mlx_lm.fuse` wrote).
// - `MLXCEL_TEST_EMBED_LORA_PROMPT`: the prompt (default "The capital of
//   France is"), encoded with the checkpoint's tokenizer and special tokens.
// - `MLXCEL_TEST_EMBED_LORA_PATHS`: comma-separated subset of
//   `fused,runtime,stage` (default all three; a quantized base cannot fuse).
// - `MLXCEL_TEST_EMBED_LORA_DUMP`: optional directory; the prompt ids and each
//   path's last-position logits (raw little-endian f32) are written there for
//   an external reference comparison.
// - `MLXCEL_TEST_EMBED_LORA_TOLERANCE`: max abs logit difference between
//   paths (default 0.25, two bf16 ulps at a logit of 16 to 32). The paths must
//   agree on the argmax and on the top 5 up to ties within this tolerance.

fn gate_env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|value| !value.is_empty())
}

fn dump(dir: Option<&Path>, name: &str, logits: &[f32]) {
    let Some(dir) = dir else { return };
    let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join(format!("{name}.f32")), bytes).expect("write logits dump");
}

#[test]
#[ignore = "real-checkpoint gate; set MLXCEL_TEST_EMBED_LORA_MODEL"]
fn real_checkpoint_embedding_adapter_parity() {
    let Some(model) = gate_env("MLXCEL_TEST_EMBED_LORA_MODEL") else {
        eprintln!("skipping: MLXCEL_TEST_EMBED_LORA_MODEL not set");
        return;
    };
    let model = PathBuf::from(model);
    let adapter = gate_env("MLXCEL_TEST_EMBED_LORA_ADAPTER").map(PathBuf::from);
    let prompt = gate_env("MLXCEL_TEST_EMBED_LORA_PROMPT")
        .unwrap_or_else(|| "The capital of France is".to_string());
    let paths = gate_env("MLXCEL_TEST_EMBED_LORA_PATHS")
        .unwrap_or_else(|| "fused,runtime,stage".to_string());
    let tolerance: f32 = gate_env("MLXCEL_TEST_EMBED_LORA_TOLERANCE")
        .map(|value| value.parse().expect("a float tolerance"))
        .unwrap_or(0.25);
    let dump_dir = gate_env("MLXCEL_TEST_EMBED_LORA_DUMP").map(PathBuf::from);
    if let Some(dir) = &dump_dir {
        std::fs::create_dir_all(dir).expect("create dump dir");
    }

    let (base_model, tokenizer) = load_model(&model).expect("the base checkpoint loads");
    let ids: Vec<i32> = tokenizer
        .encode(&prompt, true)
        .expect("the prompt tokenizes")
        .iter()
        .map(|&id| id as i32)
        .collect();
    let num_layers = base_model.num_layers();
    let base_array = last_logits(&base_model, &ids);
    let base = values(&base_array);
    drop(base_model);
    if let Some(dir) = &dump_dir {
        std::fs::write(
            dir.join("ids.json"),
            serde_json::to_string(&ids).expect("ids json"),
        )
        .expect("write ids");
    }
    dump(dump_dir.as_deref(), "base", &base);
    eprintln!("prompt ids {ids:?}; base top-5 {:?}", top_k(&base, 5));
    let Some(adapter) = adapter else {
        eprintln!("no MLXCEL_TEST_EMBED_LORA_ADAPTER: base logits only");
        return;
    };

    let mut results: Vec<(&str, Vec<f32>)> = Vec::new();
    if paths.contains("fused") {
        let (fused, _) = load_model_with_adapter(&model, &adapter).expect("the fused path loads");
        results.push(("fused", values(&last_logits(&fused, &ids))));
    }
    if paths.contains("runtime") {
        let (runtime, set) = runtime_model(&model, &adapter);
        results.push(("runtime", values(&last_logits(&runtime, &ids))));
        set.apply_scales(&[0.0]);
        let zero = last_logits(&runtime, &ids);
        assert_eq!(
            mlxcel_core::array_to_raw_bytes(&zero),
            mlxcel_core::array_to_raw_bytes(&base_array),
            "runtime scale zero must reproduce the base logits exactly"
        );
        eprintln!("runtime scale 0.0: byte-identical to base");
    }
    if paths.contains("stage") {
        let stage = stage_last_logits(&model, Some(&adapter), num_layers, 2, &ids);
        results.push(("stage", values(&stage)));
    }

    for (name, logits) in &results {
        dump(dump_dir.as_deref(), name, logits);
        eprintln!(
            "{name}: top-5 {:?}, max |diff| vs base {:.4}",
            top_k(logits, 5),
            max_abs_diff(&base, logits)
        );
    }
    let (first_name, first) = &results[0];
    for (name, logits) in &results[1..] {
        let diff = max_abs_diff(first, logits);
        eprintln!("{name} vs {first_name}: max |diff| {diff:.4}");
        assert_eq!(
            top_k(first, 1),
            top_k(logits, 1),
            "{name} and {first_name} disagree on the argmax"
        );
        assert!(
            top5_agree(first, logits, tolerance),
            "{name} and {first_name} disagree on the top-5 ids beyond bf16 near-ties: {:?} vs {:?}",
            top_k(first, 5),
            top_k(logits, 5)
        );
        assert!(
            diff <= tolerance,
            "{name} vs {first_name}: {diff} > {tolerance}"
        );
    }
}
