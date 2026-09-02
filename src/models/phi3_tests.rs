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

use super::{ModelArgs, Phi3Attention, SuRope, SuRopeTable};
use mlxcel_core::weights::WeightMap;

fn test_args() -> ModelArgs {
    serde_json::from_value(serde_json::json!({
        "model_type": "phi3",
        "hidden_size": 3072,
        "num_hidden_layers": 32,
        "intermediate_size": 8192,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "vocab_size": 1000
    }))
    .unwrap()
}

#[test]
fn phi3_model_args_default_to_full_rotary_dims() {
    let args = test_args();
    assert_eq!(args.head_dim(), 128);
    assert_eq!(args.rope_dims(), 128);
}

#[test]
fn phi3_model_args_support_partial_rotary_factor() {
    let args: ModelArgs = serde_json::from_value(serde_json::json!({
        "model_type": "phi4mm",
        "hidden_size": 3072,
        "num_hidden_layers": 32,
        "intermediate_size": 8192,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "vocab_size": 1000,
        "partial_rotary_factor": 0.75
    }))
    .unwrap();

    assert_eq!(args.head_dim(), 128);
    assert_eq!(args.rope_dims(), 96);
}

// LongRoPE table construction and selection (#1358).

/// Read a float32 array back into a `Vec<f32>` in row-major order.
fn to_vec(a: &mlxcel_core::MlxArray) -> Vec<f32> {
    let n = mlxcel_core::array_size(a);
    // Flatten first so the same helper reads a `[d/2]` frequency table and a
    // `[B, H, T, D]` projection.
    let flat = mlxcel_core::reshape(a, &[n as i32]);
    mlxcel_core::eval(&flat);
    (0..n)
        .map(|i| {
            let element = mlxcel_core::slice(&flat, &[i as i32], &[i as i32 + 1]);
            mlxcel_core::item_f32(&element)
        })
        .collect()
}

/// A `phi3` config with `head_dim = 8` (so `rope_dims / 2 = 4`) plus whatever
/// the caller merges into it.
fn args_with(overrides: serde_json::Value) -> ModelArgs {
    let mut config = serde_json::json!({
        "model_type": "phi3",
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "vocab_size": 32,
        "rope_theta": 10000.0
    });
    let object = config.as_object_mut().expect("config is an object");
    for (key, value) in overrides.as_object().expect("overrides is an object") {
        object.insert(key.clone(), value.clone());
    }
    serde_json::from_value(config).expect("config parses")
}

/// A `longrope` config whose short table is flat and whose long table is not,
/// with `original_max_position_embeddings = L` at the top level.
fn su_rope_with(original_max: usize, max_position: usize) -> SuRope {
    let args = args_with(serde_json::json!({
        "max_position_embeddings": max_position,
        "original_max_position_embeddings": original_max,
        "rope_scaling": {
            "type": "longrope",
            "short_factor": [1.0, 1.0, 1.0, 1.0],
            "long_factor": [2.0, 4.0, 8.0, 16.0]
        }
    }));
    SuRope::from_args(&args).expect("a longrope block builds both tables")
}

/// Whether the pass at `(offset, seq_len)` selects the long table.
fn picks_long(su: &SuRope, offset: i32, seq_len: i32) -> bool {
    std::ptr::eq(su.table_for(offset, seq_len), &su.long)
}

/// Relative RMS difference between two arrays of the same shape.
fn relative_rms(actual: &mlxcel_core::MlxArray, expected: &mlxcel_core::MlxArray) -> f32 {
    let actual = to_vec(actual);
    let expected = to_vec(expected);
    assert_eq!(actual.len(), expected.len(), "shapes must match to compare");
    let error: f32 = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    let magnitude: f32 = expected.iter().map(|b| b * b).sum();
    (error / magnitude.max(f32::MIN_POSITIVE)).sqrt()
}

#[test]
fn su_rope_builds_short_and_long_tables() {
    let su = su_rope_with(16, 128);

    // `freq_i = rope_theta ^ (2i / rope_dims)` before either factor list.
    let unscaled: Vec<f32> = (0..4)
        .map(|i| 10000f64.powf((2 * i) as f64 / 8.0) as f32)
        .collect();

    let short = to_vec(&su.short.freqs);
    let long = to_vec(&su.long.freqs);
    assert_eq!(short.len(), 4, "one entry per rotated pair");
    assert_eq!(long.len(), 4);

    for (i, want) in unscaled.iter().enumerate() {
        assert!(
            (short[i] - want).abs() <= want * 1e-5,
            "short[{i}] = {} but a flat short_factor leaves the base frequency {want}",
            short[i]
        );
    }
    for (i, factor) in [2.0f32, 4.0, 8.0, 16.0].iter().enumerate() {
        let want = unscaled[i] * factor;
        assert!(
            (long[i] - want).abs() <= want * 1e-5,
            "long[{i}] = {} but long_factor[{i}] scales the base frequency to {want}",
            long[i]
        );
    }
}

#[test]
fn su_table_switches_at_original_max() {
    let su = su_rope_with(16, 128);

    // A prefill that ends exactly at L still fits the trained context.
    assert!(!picks_long(&su, 0, 16));
    assert!(picks_long(&su, 0, 17));
    // Decode flips one step after the last trained position.
    assert!(!picks_long(&su, 15, 1));
    assert!(picks_long(&su, 16, 1));
}

#[test]
fn a_chunked_prefill_pins_one_table_for_the_whole_prompt() {
    // The shipped Phi-3.5-mini geometry, at the CLI's default prefill chunk.
    const CHUNK: i32 = mlxcel_core::generate::DEFAULT_PREFILL_CHUNK as i32;
    let su = su_rope_with(4096, 131072);

    // Without an announcement the chunk geometry alone splits one prompt across
    // both tables: chunks 1 and 2 of a 5136-token prompt end at 2048 and 4096,
    // inside the trained context, and only chunk 3 crosses it. That mixes keys
    // rotated with two tables into one KV cache, which is the defect the
    // announcement exists to prevent.
    assert!(!picks_long(&su, 0, CHUNK));
    assert!(!picks_long(&su, CHUNK, CHUNK));
    assert!(picks_long(&su, 2 * CHUNK, 1040));

    for (prompt_len, expect_long) in [(5136, true), (3000, false)] {
        let _span = mlxcel_core::prefill_span::announce(prompt_len);
        let mut offset = 0;
        while offset < prompt_len {
            let chunk_len = CHUNK.min(prompt_len - offset);
            assert_eq!(
                picks_long(&su, offset, chunk_len),
                expect_long,
                "prompt of {prompt_len}: the chunk at offset {offset} must use the same table as every other chunk"
            );
            offset += chunk_len;
        }
    }
}

#[test]
fn a_server_prefill_pins_one_table_across_the_boundary_segment_and_every_chunk() {
    // The shape the server actually runs, and the one that regressed after the
    // first fix for #1358 covered only the two chunked-prefill functions. With
    // the prompt cache on, one 5136-token prompt reaches the model as a
    // history-boundary segment (`prompt_tokens[0..boundary]`, forwarded by
    // `capture_history_boundary_snapshot`) followed by chunks of
    // `--prefill-chunk-size`, whose default is 512. Both the segment and the
    // early chunks end below the trained context, so without the announcement
    // they take the short table while the tail takes the long one, and the
    // greedy output degenerates into repetition.
    const SERVER_CHUNK: i32 = 512;
    let su = su_rope_with(4096, 131072);
    let prompt_len = 5136;
    let boundary = 3000;

    // The un-announced geometry really does split this prompt, so the
    // assertions below are not vacuous.
    assert!(!picks_long(&su, 0, boundary));
    assert!(!picks_long(&su, boundary, SERVER_CHUNK));
    assert!(picks_long(&su, 5120, 16));

    let _span = mlxcel_core::prefill_span::announce(prompt_len);
    assert!(
        picks_long(&su, 0, boundary),
        "the history-boundary segment must resolve the prompt's table, not its own"
    );
    let mut offset = boundary;
    while offset < prompt_len {
        let chunk_len = SERVER_CHUNK.min(prompt_len - offset);
        assert!(
            picks_long(&su, offset, chunk_len),
            "the 512-token chunk at offset {offset} must use the same table as the boundary segment"
        );
        offset += chunk_len;
    }
}

#[test]
fn mscale_override_wins_over_the_default_scale() {
    let args = args_with(serde_json::json!({
        "max_position_embeddings": 128,
        "original_max_position_embeddings": 16,
        "rope_scaling": {
            "type": "longrope",
            "short_factor": [1.0, 1.0, 1.0, 1.0],
            "long_factor": [2.0, 4.0, 8.0, 16.0],
            "short_mscale": 1.5
        }
    }));
    let su = SuRope::from_args(&args).expect("a longrope block builds both tables");

    // M / L = 8, so the default is sqrt(1 + ln(8) / ln(16)).
    let default_scale = (1.0f64 + 8.0f64.ln() / 16.0f64.ln()).sqrt() as f32;
    assert!(
        (su.short.scale - 1.5).abs() < 1e-6,
        "short_mscale overrides"
    );
    assert!(
        (su.long.scale - default_scale).abs() < 1e-6,
        "an absent long_mscale leaves the default {default_scale}, got {}",
        su.long.scale
    );
}

#[test]
fn a_scale_of_one_allocates_no_scale_array() {
    // M == L, so the default scale is exactly 1.0 and the graph path can skip
    // the rotary-prefix multiply entirely.
    let su = su_rope_with(4096, 4096);
    assert!((su.short.scale - 1.0).abs() < 1e-6);
    assert!(su.short.scale_arr.is_none());
    assert!(su.long.scale_arr.is_none());
}

#[test]
fn top_level_original_max_position_embeddings_is_honored() {
    // What `models/phi-3.5-mini-4bit/config.json` actually ships: the key is at
    // the top level and the `rope_scaling` block does not carry it.
    let args = args_with(serde_json::json!({
        "max_position_embeddings": 131072,
        "original_max_position_embeddings": 8192,
        "rope_scaling": {
            "type": "longrope",
            "short_factor": [1.0, 1.0, 1.0, 1.0],
            "long_factor": [2.0, 4.0, 8.0, 16.0]
        }
    }));
    assert_eq!(args.original_max_position_embeddings(), 8192);

    // The block wins when it carries the key too.
    let both = args_with(serde_json::json!({
        "max_position_embeddings": 131072,
        "original_max_position_embeddings": 8192,
        "rope_scaling": {
            "type": "longrope",
            "original_max_position_embeddings": 2048,
            "short_factor": [1.0, 1.0, 1.0, 1.0],
            "long_factor": [2.0, 4.0, 8.0, 16.0]
        }
    }));
    assert_eq!(both.original_max_position_embeddings(), 2048);

    // Neither spelling present falls back to the trained Phi-3 context.
    assert_eq!(test_args().original_max_position_embeddings(), 4096);
}

#[test]
fn a_config_without_rope_scaling_builds_no_su_tables() {
    // `models/phi-3-mini-4bit/config.json` has no `rope_scaling` at all, so it
    // must keep taking the plain RoPE path.
    let args = args_with(serde_json::json!({ "max_position_embeddings": 4096 }));
    assert!(SuRope::from_args(&args).is_none());

    // A block naming another scheme is not a LongRoPE block either.
    let linear = args_with(serde_json::json!({
        "rope_scaling": { "type": "linear", "factor": 4.0 }
    }));
    assert!(SuRope::from_args(&linear).is_none());

    // A `longrope` block whose factor list is too short for `rope_dims / 2`
    // cannot build a table and must not build a half-filled one.
    let truncated = args_with(serde_json::json!({
        "rope_scaling": { "type": "longrope", "long_factor": [1.0, 2.0] }
    }));
    assert!(SuRope::from_args(&truncated).is_none());
}

#[test]
fn a_long_factor_only_config_uses_it_for_both_tables() {
    // Nothing in the tree ships this, but a block that omits `short_factor`
    // must keep behaving the way it did before the short table existed.
    let args = args_with(serde_json::json!({
        "max_position_embeddings": 128,
        "original_max_position_embeddings": 16,
        "rope_scaling": {
            "type": "longrope",
            "long_factor": [2.0, 4.0, 8.0, 16.0]
        }
    }));
    let su = SuRope::from_args(&args).expect("long_factor alone still builds tables");
    assert_eq!(to_vec(&su.short.freqs), to_vec(&su.long.freqs));
}

/// Insert a quantized `{name}.weight` / `.scales` / `.biases` triple built from
/// a deterministic dense matrix.
fn insert_quantized_linear(
    weights: &mut WeightMap,
    name: &str,
    out_dim: i32,
    in_dim: i32,
    group_size: i32,
    bits: i32,
) {
    let count = (out_dim * in_dim) as usize;
    let dense: Vec<f32> = (0..count)
        .map(|i| ((i % 23) as f32 - 11.0) / 32.0)
        .collect();
    let dense = mlxcel_core::from_slice_f32(&dense, &[out_dim, in_dim]);
    weights.insert(
        format!("{name}.weight"),
        mlxcel_core::quantize_weights_w(&dense, group_size, bits),
    );
    weights.insert(
        format!("{name}.scales"),
        mlxcel_core::quantize_weights_scales(&dense, group_size, bits),
    );
    weights.insert(
        format!("{name}.biases"),
        mlxcel_core::quantize_weights_biases(&dense, group_size, bits),
    );
}

#[test]
fn fused_and_graph_su_paths_agree_per_table() {
    // Partial rotary (rope_dims 6 of head_dim 8) so the rotary-prefix slicing
    // is exercised rather than degenerating to the whole head.
    let args = args_with(serde_json::json!({
        "partial_rotary_factor": 0.75,
        "max_position_embeddings": 128,
        "original_max_position_embeddings": 16,
        "quantization": { "group_size": 32, "bits": 4 },
        "rope_scaling": {
            "type": "longrope",
            "short_factor": [1.0, 1.0, 1.0],
            "long_factor": [2.0, 8.0, 32.0],
            "short_mscale": 1.25,
            "long_mscale": 1.75
        }
    }));
    assert_eq!(args.rope_dims(), 6);

    let hidden = args.hidden_size as i32;
    let qkv_out =
        (args.num_attention_heads + 2 * args.num_kv_heads()) as i32 * args.head_dim() as i32;
    let mut weights = WeightMap::new();
    insert_quantized_linear(&mut weights, "attn.qkv_proj", qkv_out, hidden, 32, 4);
    insert_quantized_linear(&mut weights, "attn.o_proj", hidden, hidden, 32, 4);

    let attn = Phi3Attention::from_weights(&weights, &args, "attn").expect("attention loads");
    let su = attn.su_rope.as_ref().expect("longrope tables built");

    let seq_len = 3;
    let x: Vec<f32> = (0..(seq_len * hidden) as usize)
        .map(|i| ((i % 13) as f32 - 6.0) / 16.0)
        .collect();
    let x = mlxcel_core::from_slice_f32(&x, &[1, seq_len, hidden]);

    let mut fused_q_per_table: Vec<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>> = Vec::new();
    for (label, table) in [("short", &su.short), ("long", &su.long)] {
        let table: &SuRopeTable = table;
        let (fused_q, fused_k, fused_v) = attn
            .qkv_proj
            .forward_fused_qkv_split_su_scaled_rope(
                &x,
                attn.num_heads,
                attn.num_kv_heads,
                attn.head_dim,
                attn.rope_dims,
                &table.freqs,
                table.scale,
                0,
            )
            .expect("a quantized fused QKV weight takes the fused path");
        let (graph_q, graph_k, graph_v) =
            attn.prepare_qkv_with_rope(&x, 1, seq_len, 0, Some(table));

        for (name, fused, graph) in [
            ("q", &fused_q, &graph_q),
            ("k", &fused_k, &graph_k),
            ("v", &fused_v, &graph_v),
        ] {
            let rms = relative_rms(fused, graph);
            assert!(
                rms < 1e-3,
                "{label} table: fused and graph {name} differ by relative RMS {rms}"
            );
        }
        fused_q_per_table.push(fused_q);
    }

    // Guard against a vacuous comparison: the two tables must actually rotate
    // differently, or agreeing on both proves nothing.
    let across_tables = relative_rms(&fused_q_per_table[0], &fused_q_per_table[1]);
    assert!(
        across_tables > 1e-2,
        "the short and long tables produced nearly identical Q (relative RMS {across_tables}), so the per-table check is vacuous"
    );
}
