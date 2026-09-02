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

//! Regression tests for Mamba conv_state contiguous fix.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{dtype, generate::ModelStateSnapshot, utils::slice_axis};

/// Simulate 50 decode steps of conv-state update and assert the stored shape
/// stays at [B=1, k-1=3, channels=4] regardless of how many steps we run.
///
/// Before the fix, each step would store a slice that aliased the growing
/// `padded_input` (= concat(prev_state, new_token)) accumulation graph, so the
/// effective live allocation grew linearly with step count.  After the fix,
/// contiguous() materializes a compact [1, 3, 4] buffer each time.
#[test]
#[ignore = "requires serial MLX execution"]
fn mamba_conv_state_shape_plateaus_after_50_steps() {
    let batch = 1i32;
    let channels = 4i32;
    let k = 4usize; // conv_kernel_size
    let n_keep = (k - 1) as i32; // = 3
    let expected_shape = vec![batch, n_keep, channels];

    // Initial conv_state: zeros [1, k-1, channels]
    let mut conv_state =
        mlxcel_core::zeros(&[batch, n_keep, channels], mlxcel_core::dtype::FLOAT32);

    for _step in 0..50 {
        // Simulate: new single-token input [1, 1, channels]
        let new_token = mlxcel_core::zeros(&[batch, 1, channels], mlxcel_core::dtype::FLOAT32);

        // Build padded_input = concat(conv_state, new_token, axis=1) -> [1, k, channels]
        let padded_input = mlxcel_core::concatenate(&conv_state, &new_token, 1);

        let padded_shape = mlxcel_core::array_shape(&padded_input);
        let len = padded_shape[1] as usize;

        // Apply the fixed conv-state update: slice then contiguous
        let tail = slice_axis(&padded_input, 1, (len - (k - 1)) as i32, len as i32);
        conv_state = mlxcel_core::contiguous(&tail, false);

        // Materialize so shape reflects the actual allocation
        mlxcel_core::eval(&conv_state);

        let shape = mlxcel_core::array_shape(&conv_state);
        assert_eq!(
            shape, expected_shape,
            "step {_step}: conv_state shape {shape:?} != expected {expected_shape:?}"
        );
    }
}

/// Verify that MambaCache::new() starts with no conv_state.
#[test]
fn mamba_cache_new_has_no_state() {
    let cache = super::MambaCache::new();
    assert!(cache.conv_state.is_none());
    assert!(cache.ssm_state.is_none());
}

/// Verify that MambaCache::default() starts with no conv_state.
#[test]
fn mamba_cache_default_has_no_state() {
    let cache = super::MambaCache::default();
    assert!(cache.conv_state.is_none());
    assert!(cache.ssm_state.is_none());
}

#[test]
fn mamba_cache_snapshot_restore_round_trips_state_shapes() {
    let mut cache = super::MambaCache::new();
    cache.conv_state = Some(mlxcel_core::zeros(&[1, 3, 4], dtype::FLOAT32));
    cache.ssm_state = Some(mlxcel_core::zeros(&[1, 2, 4, 8], dtype::FLOAT32));

    let mut snapshot = ModelStateSnapshot::new("mamba", 7);
    cache.snapshot_into(&mut snapshot, "layer0");

    let mut restored = super::MambaCache::new();
    restored.restore_from(&snapshot, "layer0");

    let conv = restored
        .conv_state
        .as_ref()
        .and_then(|a| a.as_ref())
        .expect("conv_state restored");
    let ssm = restored
        .ssm_state
        .as_ref()
        .and_then(|a| a.as_ref())
        .expect("ssm_state restored");
    assert_eq!(mlxcel_core::array_shape(conv), vec![1, 3, 4]);
    assert_eq!(mlxcel_core::array_shape(ssm), vec![1, 2, 4, 8]);
}

/// The config every embedding-load test below shares.
///
/// `num_hidden_layers: 0` plus tied embeddings reduces the load to the
/// embedding table and the final norm, so an honest case loads all the way
/// through rather than stopping at an unrelated missing weight. `vocab_size`
/// and `hidden_size` are both 8, which is the width every fixture geometry
/// below is sized against.
fn embedding_config(group_size: i32, bits: i32) -> super::MambaConfig {
    serde_json::from_value(serde_json::json!({
        "model_type": "mamba",
        "vocab_size": 8,
        "hidden_size": 8,
        "intermediate_size": 16,
        "state_size": 4,
        "num_hidden_layers": 0,
        "conv_kernel": 4,
        "time_step_rank": 1,
        "tie_word_embeddings": true,
        "quantization": { "group_size": group_size, "bits": bits },
    }))
    .expect("test config must parse")
}

/// The final norm, the only non-embedding weight a `num_hidden_layers: 0` load
/// still asks for.
fn embedding_test_base_weights() -> WeightMap {
    let mut weights = WeightMap::new();
    weights.insert(
        "backbone.norm_f.weight".to_string(),
        mlxcel_core::from_slice_f32(&[1.0; 8], &[8]),
    );
    weights
}

/// Affine 4-bit table under `prefix`, honest at `embedding_config(8, 4)`:
/// packed_in * 32 == bits * groups * gs, i.e. 1 * 32 == 4 * 1 * 8, and the
/// dequantized width scales.shape(-1) * group_size == 8 == hidden_size. The
/// table is [vocab, packed_in].
fn affine_embedding_weights(prefix: &str) -> WeightMap {
    let mut weights = embedding_test_base_weights();
    weights.insert(
        format!("{prefix}.weight"),
        mlxcel_core::from_slice_f32(&[0.0; 8], &[8, 1]),
    );
    weights.insert(
        format!("{prefix}.scales"),
        mlxcel_core::from_slice_f32(&[1.0; 8], &[8, 1]),
    );
    weights.insert(
        format!("{prefix}.biases"),
        mlxcel_core::from_slice_f32(&[0.0; 8], &[8, 1]),
    );
    weights
}

/// Block-float table under `prefix`: `.scales` and no `.biases`, which is what
/// an mxfp4 / nvfp4 / mxfp8 export ships. `packed_in` is the stored width, so
/// the dequantized width is packed_in * 32 / bits and must come out at
/// hidden_size (8) for the declared bits.
fn block_float_embedding_weights(prefix: &str, packed_in: i32) -> WeightMap {
    let mut weights = embedding_test_base_weights();
    let elems = 8 * packed_in as usize;
    weights.insert(
        format!("{prefix}.weight"),
        mlxcel_core::from_slice_f32(&vec![0.0; elems], &[8, packed_in]),
    );
    weights.insert(
        format!("{prefix}.scales"),
        mlxcel_core::from_slice_f32(&[1.0; 8], &[8, 1]),
    );
    weights
}

/// Mamba reaches `QuantizedEmbedding` through `UnifiedEmbedding::from_weights`,
/// and therefore through `reconcile_quantization_layout`. Before issue #958 it
/// hand-built the layer and stored the declared pair verbatim; the pair was then
/// divided by inside `quantized_embedding` at the first token, which crosses the
/// cxx bridge as `UniquePtr<MlxArray>` rather than `Result` and therefore aborts
/// the process instead of failing the load.
///
/// This drives the real `MambaModel::from_weights` rather than the pure bounds
/// helper, so the guard is exercised where a checkpoint actually reaches it. The
/// assertions are on the load result and no forward pass is run, so a regression
/// fails cleanly here rather than aborting the test binary.
#[test]
fn mamba_rejects_quantization_params_that_would_abort_the_embedding_lookup() {
    use super::MambaModel;

    // Positive control first, so a guard that rejects everything quantized
    // cannot pass this test.
    MambaModel::from_weights(
        embedding_config(8, 4),
        affine_embedding_weights("backbone.embeddings"),
    )
    .expect("an honest affine pair must still load a quantized Mamba embedding");

    for (group_size, bits, field) in crate::models::switch_layers::HOSTILE_QUANT_PARAMS {
        let err = MambaModel::from_weights(
            embedding_config(group_size, bits),
            affine_embedding_weights("backbone.embeddings"),
        )
        .err()
        .unwrap_or_else(|| {
            panic!("Mamba must reject group_size {group_size} / bits {bits} at load")
        })
        .to_string();
        assert!(err.contains(field), "unhelpful error: {err}");
    }

    // A float embedding table carries no packing, so the params are irrelevant
    // there and must not be enforced. It has to be a genuine [vocab, hidden]
    // table rather than the packed one with its scales stripped, because the
    // shared table guard now checks the width of a non-quantized table against
    // hidden_size (see `mamba_rejects_an_embedding_table_narrower_than_hidden_size`).
    let mut regular = embedding_test_base_weights();
    regular.insert(
        "backbone.embeddings.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0; 64], &[8, 8]),
    );
    MambaModel::from_weights(embedding_config(0, 0), regular)
        .expect("a float embedding table must not be gated on quantization params");
}

/// Issue #976: a block-float embedding carries `.scales` and no `.biases`, and
/// must still load as quantized.
///
/// Affine stores zero points; mxfp4 / nvfp4 / mxfp8 do not, which is what
/// `infer_quantization_mode` keys on. The old gate required both planes, so a
/// block-float table fell to `Embedding::new` over the raw packed uint32 tensor,
/// whose lookup is a bare `take` with no dequantization: the result is a uint32
/// array whose last axis is the packed width rather than hidden_size.
///
/// The assertion is that the load succeeds, not that the variant is `Quantized`:
/// `MambaModel::embeddings` is private and this is a sibling module, and a
/// forward pass on a regressed load aborts the test binary rather than failing
/// it. The shared table guard supplies the observable instead, because a
/// regression stores a [8, packed_in] table as non-quantized and the guard
/// rejects a non-quantized width that is not hidden_size.
#[test]
fn mamba_loads_a_block_float_embedding_as_quantized() {
    use super::MambaModel;

    // mxfp4: infer_quantization_mode(false, 8, 4) == "mxfp4", dequantized width
    // packed_in * 32 / bits == 1 * 8 == 8 == hidden_size.
    MambaModel::from_weights(
        embedding_config(8, 4),
        block_float_embedding_weights("backbone.embeddings", 1),
    )
    .expect("an mxfp4 embedding (scales, no biases) must load as quantized");

    // mxfp8: bits 8 picks the mode, and 8-bit packing doubles the stored width
    // for the same dequantized 8, so packed_in is 2 here.
    MambaModel::from_weights(
        embedding_config(8, 8),
        block_float_embedding_weights("backbone.embeddings", 2),
    )
    .expect("an mxfp8 embedding (scales, no biases) must load as quantized");
}

/// The embedding prefix is checkpoint-dependent: `backbone.embeddings` in the
/// original mamba export, `model.embed_tokens` in the transformers conversion.
/// Resolving it is the only reason this cannot pass a fixed prefix to the shared
/// loader, so both spellings must reach the same code path, and a checkpoint
/// carrying neither must say so naming both.
#[test]
fn mamba_resolves_both_embedding_prefix_spellings() {
    use super::MambaModel;

    MambaModel::from_weights(
        embedding_config(8, 4),
        affine_embedding_weights("model.embed_tokens"),
    )
    .expect("the transformers spelling must load an affine embedding");
    MambaModel::from_weights(
        embedding_config(8, 4),
        block_float_embedding_weights("model.embed_tokens", 1),
    )
    .expect("the transformers spelling must load a block-float embedding");

    let err = MambaModel::from_weights(embedding_config(8, 4), embedding_test_base_weights())
        .err()
        .expect("a checkpoint with no embedding table at all must fail the load")
        .to_string();
    assert!(err.contains("backbone.embeddings.weight"), "{err}");
    assert!(err.contains("model.embed_tokens.weight"), "{err}");
}

/// The shared `validate_embedding_table` guard is new for this family. A
/// non-quantized table whose width is not the model width feeds a wrong-width
/// hidden state into the first norm, which throws inside MLX and crosses the cxx
/// bridge as an uncatchable abort rather than a load error, so it has to be
/// caught here. The message names the config field the reader will find in their
/// `config.json`.
#[test]
fn mamba_rejects_an_embedding_table_narrower_than_hidden_size() {
    use super::MambaModel;

    let mut narrow = embedding_test_base_weights();
    narrow.insert(
        "backbone.embeddings.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0; 8], &[8, 1]),
    );
    let err = MambaModel::from_weights(embedding_config(0, 0), narrow)
        .err()
        .expect("a float embedding table narrower than hidden_size must fail the load")
        .to_string();
    assert!(err.contains("hidden_size"), "unhelpful error: {err}");
}

/// The test above exercises the guard's width clause; this one exercises its
/// row-count clause, `claimed_rows > rows`. Token ids are bounded by
/// `vocab_size`, not by the rows actually present in the table, and MLX's
/// embedding gather wraps a negative index but performs no range check on a
/// positive one, so a config that overstates `vocab_size` turns an ordinary
/// prompt into an out-of-bounds read whose result reaches the logits rather
/// than faulting. It is also the one place a caller can silently mis-wire the
/// shared guard by passing the wrong config field into the `claimed_rows`
/// slot, so this pins Mamba to `config.vocab_size` specifically, not merely
/// to some `usize` that happens to be lying around.
#[test]
fn mamba_rejects_a_config_that_overstates_vocab_size() {
    use super::MambaModel;

    let mut narrow = embedding_test_base_weights();
    narrow.insert(
        "backbone.embeddings.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0; 24], &[3, 8]),
    );
    let err = MambaModel::from_weights(embedding_config(0, 0), narrow)
        .err()
        .expect("a config that overstates the embeddings table must fail the load")
        .to_string();
    assert!(err.contains("vocab_size"), "unhelpful error: {err}");

    // The other direction is safe: a table padded with more rows than
    // vocab_size keeps the bound inside it.
    let mut padded = embedding_test_base_weights();
    padded.insert(
        "backbone.embeddings.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0; 512], &[64, 8]),
    );
    MambaModel::from_weights(embedding_config(0, 0), padded)
        .expect("a table with more rows than vocab_size must still load");
}

/// Build a `MambaBlock` with minimal, functionally-inert projections so tests
/// can drive `mixer_norm` directly without a full checkpoint. Only
/// `use_bcdt_rms` and `mixer_rms_eps` matter for the assertions below; the
/// remaining fields exist only to satisfy `MambaBlock`'s shape and type
/// contract and are never read by `mixer_norm`.
fn tiny_mamba_block(use_bcdt_rms: bool, mixer_rms_eps: f32) -> super::MambaBlock {
    use mlxcel_core::layers::{Linear, UnifiedLinear};

    let hidden_size = 4usize;
    let intermediate_size = 8usize;
    let state_size = 2usize;
    let time_step_rank = 2usize;
    let conv_kernel_size = 4usize;

    let conv_weight = mlxcel_core::zeros(
        &[intermediate_size as i32, 1, conv_kernel_size as i32],
        dtype::FLOAT32,
    );
    let in_proj = UnifiedLinear::Regular(Linear::new(
        mlxcel_core::zeros(
            &[(2 * intermediate_size) as i32, hidden_size as i32],
            dtype::FLOAT32,
        ),
        None,
    ));
    let x_proj_out = (time_step_rank + 2 * state_size) as i32;
    let x_proj = UnifiedLinear::Regular(Linear::new(
        mlxcel_core::zeros(&[x_proj_out, intermediate_size as i32], dtype::FLOAT32),
        None,
    ));
    let dt_proj = UnifiedLinear::Regular(Linear::new(
        mlxcel_core::zeros(
            &[intermediate_size as i32, time_step_rank as i32],
            dtype::FLOAT32,
        ),
        Some(mlxcel_core::zeros(
            &[intermediate_size as i32],
            dtype::FLOAT32,
        )),
    ));
    let out_proj = UnifiedLinear::Regular(Linear::new(
        mlxcel_core::zeros(
            &[hidden_size as i32, intermediate_size as i32],
            dtype::FLOAT32,
        ),
        None,
    ));
    let a_log = mlxcel_core::zeros(
        &[intermediate_size as i32, state_size as i32],
        dtype::FLOAT32,
    );
    let d_param = mlxcel_core::zeros(&[intermediate_size as i32], dtype::FLOAT32);

    super::MambaBlock {
        hidden_size,
        intermediate_size,
        state_size,
        conv_kernel_size,
        time_step_rank,
        use_bcdt_rms,
        mixer_rms_eps,
        conv_weight,
        conv_bias: None,
        in_proj,
        x_proj,
        dt_proj,
        out_proj,
        a_log,
        d_param,
    }
}

/// Issue #1333: `ssm_step` used to run `mixer_norm(mixer_norm(x))`, applying
/// the weight-less RMS norm twice per tensor. This pins the fix at the
/// `mixer_norm` boundary: the block must return exactly what a single call to
/// the bridge's `fast_rms_norm_no_weight` produces, and that value must be
/// bit-distinct from what a second application on top of it would give. The
/// double application is numerically close to a no-op (a `1/sqrt(1+eps)`
/// rescale), so `array_equal` (exact, not tolerance-based) is what actually
/// distinguishes "once" from "twice" here; a tolerance check would be too
/// loose to catch a regression back to the double call.
#[test]
fn mamba_mixer_norm_applies_the_bridge_norm_exactly_once() {
    let eps = 1e-6f32;
    let block = tiny_mamba_block(true, eps);
    let x = mlxcel_core::from_slice_f32(&[3.0, 4.0, -2.0, 1.0], &[1, 4]);

    let once = block.mixer_norm(&x);
    let single_bridge_call = mlxcel_core::fast_rms_norm_no_weight(&x, eps);
    let twice_bridge_call = mlxcel_core::fast_rms_norm_no_weight(&single_bridge_call, eps);

    let matches_single = mlxcel_core::array_equal(&once, &single_bridge_call, false);
    mlxcel_core::eval(&matches_single);
    assert!(
        mlxcel_core::item_bool(&matches_single),
        "mixer_norm must equal exactly one fast_rms_norm_no_weight call"
    );

    let matches_double = mlxcel_core::array_equal(&once, &twice_bridge_call, false);
    mlxcel_core::eval(&matches_double);
    assert!(
        !mlxcel_core::item_bool(&matches_double),
        "mixer_norm must NOT equal a second norm applied on top of the first \
         (this is the regression #1333 fixed: applying the norm twice)"
    );
}

/// With `use_bcdt_rms == false` (every non-falcon_mamba checkpoint),
/// `mixer_norm` must be a pass-through: `delta`, `B` and `C` are used
/// unnormalized, matching the Python reference's `if self.use_bcdt_rms:` gate.
#[test]
fn mamba_mixer_norm_is_identity_when_bcdt_rms_disabled() {
    let block = tiny_mamba_block(false, 1e-6);
    let x = mlxcel_core::from_slice_f32(&[3.0, 4.0, -2.0, 1.0], &[1, 4]);

    let out = block.mixer_norm(&x);
    let equal = mlxcel_core::array_equal(&out, &x, false);
    mlxcel_core::eval(&equal);
    assert!(
        mlxcel_core::item_bool(&equal),
        "mixer_norm must pass x through unchanged when use_bcdt_rms is false"
    );
}
