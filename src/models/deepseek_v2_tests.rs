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

//! Regression test for the bound on declared quantization params that
//! `deepseek_v2::load_switch_linear` applies before it stores them on a
//! quantized expert plane (issue #958).

use super::load_switch_linear;
use crate::models::switch_layers::{HOSTILE_QUANT_PARAMS, insert_stacked_quantized_expert_plane};
use mlxcel_core::weights::WeightMap;

/// Honest 4-bit expert geometry: `packed_in * 32 == bits * num_groups *
/// group_size` (8 * 32 == 4 * 1 * 64), so the positive control below is a plane
/// MLX can actually describe.
const EXPERTS: i32 = 3;
const OUT: i32 = 4;
const PACKED_IN: i32 = 8;
const NUM_GROUPS: i32 = 1;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

/// This loader builds its keys as `{prefix}.{weight_name}.weight`, so the
/// fixture is inserted under the joined path.
const PREFIX: &str = "model.layers.0.mlp.switch_mlp";
const WEIGHT_NAME: &str = "gate_proj";

/// The pair this loader stores is handed straight to `gather_qmm`, which
/// crosses the cxx bridge as `UniquePtr<MlxArray>` rather than `Result`. A C++
/// throw there is an uncatchable `std::terminate`,
/// so losing the bound turns a rejected load into an uncatchable abort at the
/// first routed forward pass in production. This test asserts on the load
/// result rather than running a forward pass, so a regression fails cleanly
/// here instead of aborting the test binary.
#[test]
fn deepseek_v2_switch_linear_rejects_quantization_params_that_would_abort_gather_qmm() {
    let plane_prefix = format!("{PREFIX}.{WEIGHT_NAME}");
    let mut weights = WeightMap::new();
    insert_stacked_quantized_expert_plane(
        &mut weights,
        &plane_prefix,
        EXPERTS,
        OUT,
        PACKED_IN,
        NUM_GROUPS,
    );

    // Positive control first, so a guard that rejected every quantized plane
    // could not pass this test.
    let experts = EXPERTS as usize;
    match load_switch_linear(&weights, experts, PREFIX, WEIGHT_NAME, GROUP_SIZE, BITS) {
        Ok(_) => {}
        Err(e) => panic!("honest 4-bit expert plane must load: {e}"),
    }

    for (group_size, bits, field) in HOSTILE_QUANT_PARAMS {
        let result = load_switch_linear(&weights, experts, PREFIX, WEIGHT_NAME, group_size, bits);
        let err = match result {
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
        assert!(
            err.contains(&plane_prefix),
            "the load error must name the offending tensor {plane_prefix}, got: {err}"
        );
    }

    // A bf16 expert plane carries no packing and no `.scales`, so the declared
    // pair is inert on the `gather_mm` path and must not gate the load.
    let mut dense = WeightMap::new();
    let n = (EXPERTS * OUT * PACKED_IN) as usize;
    dense.insert(
        format!("{plane_prefix}.weight"),
        mlxcel_core::from_slice_f32(&vec![0.0f32; n], &[EXPERTS, OUT, PACKED_IN]),
    );
    match load_switch_linear(&dense, experts, PREFIX, WEIGHT_NAME, 0, 0) {
        Ok(_) => {}
        Err(e) => panic!("a non-quantized expert plane must load with an unset pair: {e}"),
    }
}

// Absorbed MLA decode wiring (issue #907).
//
// The parity gate the issue asks for is that `MLXCEL_MLA_ABSORBED=1` changes
// the cache layout and the decode graph but not the numbers. Proving that on a
// real DeepSeek checkpoint is not possible here (no MLA checkpoint fits the
// development host), so this drives the family's own `MLAAttention::forward`
// on synthetic weights: same weights, same inputs, one attention with the fold
// and one without, compared through the full block including `o_proj`.
//
// The flag itself cannot be toggled inside one process (`absorbed_enabled` is a
// `OnceLock`), so the test sets the folded field directly. That is the same
// state the loader produces when the flag is on, and it is reachable because
// this test module is compiled inside `deepseek_v2`.

use super::{KVCache, MLAAttention, ModelArgs, load_mla_attention};
use mlxcel_core::mla::{MlaAbsorbedProjections, MlaGeometry};
use mlxcel_core::{MlxArray, UniquePtr};

const HIDDEN: usize = 64;
const ATTN_PREFIX: &str = "model.layers.0.self_attn";

fn mla_test_args() -> ModelArgs {
    // Small but with every head dimension distinct, so a swapped operand cannot
    // pass by coincidence.
    serde_json::from_str(
        r#"{
            "model_type": "deepseek_v2",
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_hidden_layers": 1,
            "kv_lora_rank": 32,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8,
            "v_head_dim": 12,
            "rope_theta": 10000.0
        }"#,
    )
    .expect("mla test args")
}

fn xorshift(state: &mut u64, n: usize, amplitude: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            let unit = ((*state >> 40) as f32) / (1u32 << 24) as f32;
            (unit * 2.0 - 1.0) * amplitude
        })
        .collect()
}

fn mla_test_weights(args: &ModelArgs, seed: u64) -> WeightMap {
    let mut state = seed | 1;
    let mut weights = WeightMap::new();
    let h = args.num_attention_heads as i32;
    let q_head_dim = args.q_head_dim() as i32;
    let hidden = HIDDEN as i32;
    let mut put = |name: &str, shape: &[i32], amplitude: f32| {
        let n: usize = shape.iter().map(|d| *d as usize).product();
        weights.insert(
            format!("{ATTN_PREFIX}.{name}"),
            mlxcel_core::from_slice_f32(&xorshift(&mut state, n, amplitude), shape),
        );
    };
    put("q_proj.weight", &[h * q_head_dim, hidden], 0.2);
    put(
        "kv_a_proj_with_mqa.weight",
        &[(args.kv_lora_rank + args.qk_rope_head_dim) as i32, hidden],
        0.2,
    );
    put("kv_a_layernorm.weight", &[args.kv_lora_rank as i32], 1.0);
    put(
        "kv_b_proj.weight",
        &[
            h * (args.qk_nope_head_dim + args.v_head_dim) as i32,
            args.kv_lora_rank as i32,
        ],
        0.2,
    );
    put("o_proj.weight", &[hidden, h * args.v_head_dim as i32], 0.2);
    weights
}

fn to_vec_f32(a: &MlxArray) -> Vec<f32> {
    let f = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f);
    mlxcel_core::array_to_raw_bytes(&f)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

/// Prefill `prompt` tokens then decode `steps` single tokens, returning every
/// step's block output so a divergence at any step is caught, not just the last.
fn run_block(attn: &MLAAttention, hidden_states: &[UniquePtr<MlxArray>]) -> Vec<Vec<f32>> {
    let mut cache = KVCache::new();
    hidden_states
        .iter()
        .map(|x| to_vec_f32(&attn.forward(x, None, &mut cache)))
        .collect()
}

#[test]
fn absorbed_mla_attention_matches_the_decompressed_block_step_for_step() {
    let args = mla_test_args();
    let weights = mla_test_weights(&args, 0x907);
    let geometry = MlaGeometry {
        num_heads: args.num_attention_heads,
        kv_lora_rank: args.kv_lora_rank,
        qk_nope_head_dim: args.qk_nope_head_dim,
        qk_rope_head_dim: args.qk_rope_head_dim,
        v_head_dim: args.v_head_dim,
    };

    let baseline = load_mla_attention(&weights, &args, ATTN_PREFIX).expect("load baseline");
    assert!(
        baseline.absorbed.is_none(),
        "the loader must not fold when MLXCEL_MLA_ABSORBED is unset"
    );

    let mut absorbed = load_mla_attention(&weights, &args, ATTN_PREFIX).expect("load absorbed");
    absorbed.absorbed =
        Some(MlaAbsorbedProjections::from_kv_b_proj(&absorbed.kv_b_proj, geometry).expect("fold"));

    // A 6-token prefill followed by 4 decode steps, so the test covers the
    // cache-append boundary and the rope offsets, not just one isolated call.
    let mut state = 0xABCDu64;
    let mut inputs = Vec::new();
    inputs.push(mlxcel_core::from_slice_f32(
        &xorshift(&mut state, 6 * HIDDEN, 0.5),
        &[1, 6, HIDDEN as i32],
    ));
    for _ in 0..4 {
        inputs.push(mlxcel_core::from_slice_f32(
            &xorshift(&mut state, HIDDEN, 0.5),
            &[1, 1, HIDDEN as i32],
        ));
    }

    let want = run_block(&baseline, &inputs);
    let got = run_block(&absorbed, &inputs);
    assert_eq!(want.len(), got.len());

    for (step, (w, g)) in want.iter().zip(&got).enumerate() {
        let scale = w.iter().fold(0.0f32, |acc, v| acc.max(v.abs())).max(1e-6);
        let err = w
            .iter()
            .zip(g)
            .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs() / scale));
        // f32 fixtures, so the only difference is the accumulation order of the
        // two mathematically identical contractions.
        assert!(err < 1e-4, "step {step} drifted by {err}");
    }
}

#[test]
fn absorbed_attention_leaves_a_much_smaller_cache_behind() {
    // The point of the change, measured on the family's own cache rather than
    // asserted from the arithmetic in `mla::cache`.
    let args = mla_test_args();
    let weights = mla_test_weights(&args, 0x908);
    let geometry = MlaGeometry {
        num_heads: args.num_attention_heads,
        kv_lora_rank: args.kv_lora_rank,
        qk_nope_head_dim: args.qk_nope_head_dim,
        qk_rope_head_dim: args.qk_rope_head_dim,
        v_head_dim: args.v_head_dim,
    };
    let baseline = load_mla_attention(&weights, &args, ATTN_PREFIX).unwrap();
    let mut absorbed = load_mla_attention(&weights, &args, ATTN_PREFIX).unwrap();
    absorbed.absorbed =
        Some(MlaAbsorbedProjections::from_kv_b_proj(&absorbed.kv_b_proj, geometry).unwrap());

    let x = mlxcel_core::from_slice_f32(&vec![0.1f32; 32 * HIDDEN], &[1, 32, HIDDEN as i32]);

    let mut base_cache = KVCache::new();
    let _ = baseline.forward(&x, None, &mut base_cache);
    base_cache.eval_state();

    let mut latent_cache = KVCache::new();
    let _ = absorbed.forward(&x, None, &mut latent_cache);
    latent_cache.eval_state();

    // 4 heads * (24 key + 12 value) = 144 elements vs 32 + 8 = 40.
    assert!(
        base_cache.nbytes() > latent_cache.nbytes() * 3,
        "decompressed cache {} bytes vs latent {} bytes; absorption did not take",
        base_cache.nbytes(),
        latent_cache.nbytes()
    );
}
