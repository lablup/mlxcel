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

//! Regression test for the bound on declared quantization params that the
//! Llama 4 `SwitchLinear` loader applies before it stores them on a quantized
//! expert plane (issue #958).

use mlxcel_core::weights::WeightMap;

use super::{SwitchLinear, TextArgs};
use crate::models::switch_layers::{HOSTILE_QUANT_PARAMS, insert_stacked_quantized_expert_plane};

/// Honest 4-bit expert geometry: `packed_in * 32 == bits * num_groups *
/// group_size` (8 * 32 == 4 * 1 * 64), so the positive control below is a plane
/// MLX can actually describe.
const EXPERTS: i32 = 3;
const OUT: i32 = 4;
const PACKED_IN: i32 = 8;
const NUM_GROUPS: i32 = 1;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

const PREFIX: &str = "language_model.model.layers.0.feed_forward.experts.gate_proj";

/// Smallest Llama 4 text config that parses, varying only the declared
/// quantization pair. Llama 4 spells it as flat top-level `group_size` / `bits`
/// keys rather than a nested `quantization` block. Built through serde rather
/// than a struct literal so every `#[serde(default)]` field fills itself in.
fn args_with(group_size: i32, bits: i32) -> TextArgs {
    serde_json::from_value(serde_json::json!({
        "model_type": "llama4",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "intermediate_size_mlp": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-5,
        "vocab_size": 32,
        "head_dim": 4,
        "max_position_embeddings": 16,
        "attention_chunk_size": 8,
        "interleave_moe_layer_step": 1,
        "num_local_experts": 3,
        "num_experts_per_tok": 1,
        "group_size": group_size,
        "bits": bits,
    }))
    .expect("test config must parse")
}

/// The pair this loader stores is handed straight to `gather_qmm`, which
/// crosses the cxx bridge as `UniquePtr<MlxArray>` rather than `Result`. A C++
/// throw there is an uncatchable `std::terminate`,
/// so losing the bound turns a rejected load into an uncatchable abort at the
/// first routed forward pass in production. This test asserts on the load
/// result rather than running a forward pass, so a regression fails cleanly
/// here instead of aborting the test binary.
#[test]
fn llama4_switch_linear_rejects_quantization_params_that_would_abort_gather_qmm() {
    let mut weights = WeightMap::new();
    insert_stacked_quantized_expert_plane(
        &mut weights,
        PREFIX,
        EXPERTS,
        OUT,
        PACKED_IN,
        NUM_GROUPS,
    );

    // Positive control first, so a guard that rejected every quantized plane
    // could not pass this test.
    SwitchLinear::from_weights(&weights, &args_with(GROUP_SIZE, BITS), PREFIX)
        .expect("honest 4-bit expert plane must load");

    for (group_size, bits, field) in HOSTILE_QUANT_PARAMS {
        let err = match SwitchLinear::from_weights(&weights, &args_with(group_size, bits), PREFIX) {
            Ok(_) => panic!(
                "(group_size {group_size}, bits {bits}) must be refused at load, \
                 not stored for gather_qmm"
            ),
            Err(e) => e,
        };
        assert!(
            err.contains(field),
            "(group_size {group_size}, bits {bits}) must be blamed on {field}, got: {err}"
        );
    }

    // A bf16 expert plane carries no packing at all, so the declared pair is
    // irrelevant there and must not gate the non-quantized fallback.
    let mut regular = WeightMap::new();
    regular.insert(
        format!("{PREFIX}.weight"),
        mlxcel_core::ones(&[EXPERTS, OUT, PACKED_IN], mlxcel_core::dtype::BFLOAT16),
    );
    SwitchLinear::from_weights(&regular, &args_with(0, 0), PREFIX)
        .expect("a bf16 expert plane must not be gated on quantization params");
}

// -----------------------------------------------------------------
// Exact-prefix snapshot prompt-cache support (issue #1335).
//
// Llama 4 is the only family holding a `ChunkedKVCache`, and it holds it
// beside an ordinary `KVCache` on every fourth layer. The fixture below is
// four layers under `attention_chunk_size = 8`, so layers 0 to 2 carry the
// chunked cache and layer 3 the dense one, and a prompt past eight tokens has
// already trimmed the chunked front by the time the snapshot is taken.
mod snapshot_prompt_cache {
    /// Sequence-id base for this module, so the ids stay readable as
    /// "issue 1335, sequence N" without tripping the inconsistent-digit-
    /// grouping lint that `1335_01` does.
    const SEQ_BASE: u64 = 1_335_000;

    use super::super::{Llama4Cache, Llama4CxxModel, Llama4Wrapper, TextArgs};
    use mlxcel_core::cache::SequenceId;
    use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot};
    use mlxcel_core::layers::{ChunkedKVCache, KVCache};
    use mlxcel_core::weights::WeightMap;
    use mlxcel_core::{MlxArray, UniquePtr};

    const HIDDEN: i32 = 8;
    const VOCAB: i32 = 16;
    const INTER: i32 = 8;
    const HEAD_DIM: i32 = 4;
    const HEADS: i32 = 2;
    const KV_HEADS: i32 = 1;
    const EXPERTS: i32 = 2;
    const LAYERS: usize = 4;
    const CHUNK: i32 = 8;

    /// Deterministic small-magnitude weights. Constant fills would collapse
    /// every logit to the same value and make the comparisons below vacuous.
    fn tensor(shape: &[i32], seed: u64) -> UniquePtr<MlxArray> {
        let len = shape.iter().product::<i32>() as usize;
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let data: Vec<f32> = (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = ((state >> 40) as f32) / ((1u32 << 24) as f32);
                unit * 0.4 - 0.2
            })
            .collect();
        mlxcel_core::from_slice_f32(&data, shape)
    }

    fn ones(shape: &[i32]) -> UniquePtr<MlxArray> {
        let len = shape.iter().product::<i32>() as usize;
        mlxcel_core::from_slice_f32(&vec![1.0_f32; len], shape)
    }

    fn synthetic_args() -> TextArgs {
        serde_json::from_value(serde_json::json!({
            "model_type": "llama4",
            "hidden_size": HIDDEN,
            "num_hidden_layers": LAYERS,
            "intermediate_size": INTER,
            "intermediate_size_mlp": INTER,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "rms_norm_eps": 1e-5,
            "vocab_size": VOCAB,
            "head_dim": HEAD_DIM,
            "max_position_embeddings": 64,
            "attention_chunk_size": CHUNK,
            "interleave_moe_layer_step": 1,
            "num_local_experts": EXPERTS,
            "num_experts_per_tok": 1,
        }))
        .expect("synthetic Llama 4 config must parse")
    }

    fn synthetic_weights() -> WeightMap {
        let mut w = WeightMap::new();
        w.insert(
            "language_model.model.embed_tokens.weight".into(),
            tensor(&[VOCAB, HIDDEN], 1),
        );
        w.insert(
            "language_model.lm_head.weight".into(),
            tensor(&[VOCAB, HIDDEN], 2),
        );
        w.insert("language_model.model.norm.weight".into(), ones(&[HIDDEN]));
        for layer in 0..LAYERS {
            let seed = 200 + layer as u64 * 20;
            let prefix = format!("language_model.model.layers.{layer}");
            let attn = format!("{prefix}.self_attn");
            w.insert(
                format!("{attn}.q_proj.weight"),
                tensor(&[HEADS * HEAD_DIM, HIDDEN], seed),
            );
            w.insert(
                format!("{attn}.k_proj.weight"),
                tensor(&[KV_HEADS * HEAD_DIM, HIDDEN], seed + 1),
            );
            w.insert(
                format!("{attn}.v_proj.weight"),
                tensor(&[KV_HEADS * HEAD_DIM, HIDDEN], seed + 2),
            );
            w.insert(
                format!("{attn}.o_proj.weight"),
                tensor(&[HIDDEN, HEADS * HEAD_DIM], seed + 3),
            );

            let moe = format!("{prefix}.feed_forward");
            w.insert(
                format!("{moe}.router.weight"),
                tensor(&[EXPERTS, HIDDEN], seed + 4),
            );
            w.insert(
                format!("{moe}.experts.gate_proj.weight"),
                tensor(&[EXPERTS, INTER, HIDDEN], seed + 5),
            );
            w.insert(
                format!("{moe}.experts.up_proj.weight"),
                tensor(&[EXPERTS, INTER, HIDDEN], seed + 6),
            );
            w.insert(
                format!("{moe}.experts.down_proj.weight"),
                tensor(&[EXPERTS, HIDDEN, INTER], seed + 7),
            );
            w.insert(
                format!("{moe}.shared_expert.gate_proj.weight"),
                tensor(&[INTER, HIDDEN], seed + 8),
            );
            w.insert(
                format!("{moe}.shared_expert.up_proj.weight"),
                tensor(&[INTER, HIDDEN], seed + 9),
            );
            w.insert(
                format!("{moe}.shared_expert.down_proj.weight"),
                tensor(&[HIDDEN, INTER], seed + 10),
            );

            w.insert(format!("{prefix}.input_layernorm.weight"), ones(&[HIDDEN]));
            w.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                ones(&[HIDDEN]),
            );
        }
        w
    }

    fn build_wrapper() -> Llama4Wrapper {
        let args = synthetic_args();
        let weights = synthetic_weights();
        Llama4Wrapper::new(
            Llama4CxxModel::from_weights(&weights, &args).expect("synthetic Llama 4 must load"),
        )
    }

    fn kv(start: i32, n: i32, base: f32) -> UniquePtr<MlxArray> {
        let mut data = Vec::with_capacity((KV_HEADS * n * HEAD_DIM) as usize);
        for h in 0..KV_HEADS {
            for t in 0..n {
                for d in 0..HEAD_DIM {
                    data.push(base + (h as f32) * 100.0 + ((start + t) as f32) * 10.0 + (d as f32));
                }
            }
        }
        mlxcel_core::from_slice_f32(&data, &[1, KV_HEADS, n, HEAD_DIM])
    }

    fn to_vec_f32(arr: &MlxArray) -> Vec<f32> {
        let arr_f32 = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::eval(&arr_f32);
        mlxcel_core::array_to_raw_bytes(&arr_f32)
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    fn ids(range: std::ops::Range<i32>) -> Vec<i32> {
        range.map(|i| i.rem_euclid(VOCAB - 1) + 1).collect()
    }

    fn prefill(wrapper: &Llama4Wrapper, seq: SequenceId, tokens: &[i32]) {
        wrapper.prepare_sequence_state(seq);
        let prompt = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
        let _ = wrapper.forward_with_sequence_id(&prompt, Some(seq), &mut [], None);
    }

    fn decode(wrapper: &Llama4Wrapper, seq: SequenceId, token: i32) -> Vec<f32> {
        let input = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
        let logits = wrapper.forward_with_sequence_id(&input, Some(seq), &mut [], None);
        to_vec_f32(logits.as_ref().expect("logits"))
    }

    #[test]
    fn llama4_chunked_cache_round_trips_through_a_snapshot() {
        let mut source = Llama4Cache::Chunked(ChunkedKVCache::new(CHUNK));
        if let Llama4Cache::Chunked(cache) = &mut source {
            let _ = cache.update_and_fetch(kv(0, 20, 0.0), kv(0, 20, 10_000.0));
            cache.maybe_trim_front();
            assert_eq!(cache.start_position, 12);
            assert_eq!(cache.offset, 20);
        }

        let mut snapshot = ModelStateSnapshot::new("llama4", 20);
        source
            .snapshot_into(&mut snapshot, "layer0")
            .expect("chunked snapshot must succeed");
        assert!(snapshot.tensor("layer0.chunked.keys").is_some());

        let mut restored = Llama4Cache::Chunked(ChunkedKVCache::new(CHUNK));
        restored
            .restore_from(&snapshot, "layer0")
            .expect("chunked restore must succeed");
        assert_eq!(restored.offset(), 20);
        assert_eq!(restored.start_position(), 12);

        let (source_k, source_v) = source.update_and_fetch(kv(20, 1, 0.0), kv(20, 1, 10_000.0));
        let (restored_k, restored_v) =
            restored.update_and_fetch(kv(20, 1, 0.0), kv(20, 1, 10_000.0));
        assert!(mlxcel_core::item_bool(&mlxcel_core::array_equal(
            &restored_k,
            &source_k,
            false
        )));
        assert!(mlxcel_core::item_bool(&mlxcel_core::array_equal(
            &restored_v,
            &source_v,
            false
        )));
    }

    #[test]
    fn llama4_regular_cache_round_trips_through_a_snapshot() {
        let mut source = Llama4Cache::Regular(KVCache::new());
        source.update_and_fetch(kv(0, 12, 0.0), kv(0, 12, 10_000.0));

        let mut snapshot = ModelStateSnapshot::new("llama4", 12);
        source
            .snapshot_into(&mut snapshot, "layer3")
            .expect("standard snapshot must succeed");
        assert!(snapshot.tensor("layer3.standard.keys").is_some());

        let mut restored = Llama4Cache::Regular(KVCache::new());
        restored
            .restore_from(&snapshot, "layer3")
            .expect("standard restore must succeed");
        assert_eq!(restored.offset(), 12);

        let (source_k, _) = source.update_and_fetch(kv(12, 1, 0.0), kv(12, 1, 10_000.0));
        let (restored_k, _) = restored.update_and_fetch(kv(12, 1, 0.0), kv(12, 1, 10_000.0));
        assert!(mlxcel_core::item_bool(&mlxcel_core::array_equal(
            &restored_k,
            &source_k,
            false
        )));
    }

    #[test]
    fn llama4_truncation_declines_once_the_chunked_front_is_trimmed() {
        let mut untrimmed = Llama4Cache::Chunked(ChunkedKVCache::new(CHUNK));
        untrimmed.update_and_fetch(kv(0, 6, 0.0), kv(0, 6, 10_000.0));
        untrimmed.maybe_trim_front();
        let mut before = ModelStateSnapshot::new("llama4", 6);
        untrimmed
            .snapshot_into(&mut before, "layer0")
            .expect("snapshot");
        assert!(Llama4Cache::snapshot_truncatable_to(&before, "layer0", 4));

        let mut trimmed = Llama4Cache::Chunked(ChunkedKVCache::new(CHUNK));
        trimmed.update_and_fetch(kv(0, 20, 0.0), kv(0, 20, 10_000.0));
        trimmed.maybe_trim_front();
        let mut after = ModelStateSnapshot::new("llama4", 20);
        trimmed
            .snapshot_into(&mut after, "layer0")
            .expect("snapshot");
        assert!(!Llama4Cache::snapshot_truncatable_to(&after, "layer0", 4));
    }

    #[test]
    fn llama4_declares_snapshot_reuse() {
        assert!(build_wrapper().supports_snapshot_reuse());
    }

    #[test]
    #[ignore = "requires serial MLX execution"]
    fn snapshot_restore_matches_cold_decode() {
        // 20 prefill tokens past an 8-token chunk means the chunked layers
        // have already trimmed their front when the snapshot is taken, which
        // is the state an exact restore has to reproduce.
        let cold = build_wrapper();
        let seq_cold = SequenceId::from_raw(SEQ_BASE + 11);
        prefill(&cold, seq_cold, &ids(0..20));
        for token in ids(20..24) {
            let _ = decode(&cold, seq_cold, token);
        }

        let snapshot = cold
            .snapshot_sequence_state(seq_cold, 24)
            .expect("Llama 4 must donate a non-empty snapshot");
        assert_eq!(snapshot.family(), "llama4");

        let restored = build_wrapper();
        let seq_restored = SequenceId::from_raw(SEQ_BASE + 12);
        restored.prepare_sequence_state(seq_restored);
        restored
            .restore_sequence_state(seq_restored, &snapshot)
            .expect("Llama 4 must restore its own snapshot");

        for token in ids(24..28) {
            let reference = decode(&cold, seq_cold, token);
            let got = decode(&restored, seq_restored, token);
            assert_eq!(got.len(), reference.len());
            for (i, (&g, &w)) in got.iter().zip(reference.iter()).enumerate() {
                let abs = (g - w).abs();
                let rel = abs / w.abs().max(1.0);
                assert!(
                    abs < 1e-3 || rel < 1e-3,
                    "logit[{i}] differs after Llama 4 snapshot restore: restored={g}, reference={w}, abs={abs}, rel={rel}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires serial MLX execution"]
    fn a_foreign_family_snapshot_is_refused() {
        let wrapper = build_wrapper();
        let foreign = ModelStateSnapshot::new("gemma3", 4);
        let err = wrapper
            .restore_sequence_state(SequenceId::from_raw(SEQ_BASE + 13), &foreign)
            .expect_err("a Gemma 3 snapshot must not land in Llama 4");
        assert!(err.contains("gemma3"), "unexpected error: {err}");
        assert!(!wrapper.snapshot_truncatable_to(&foreign, 2));
    }
}
