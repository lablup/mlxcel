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

//! The affine int8 MLA latent on `DeepSeekV3Attention` (issue #3160).
//!
//! The quality reference is the fp16 latent of the same weights, held to the
//! #3155 codec bound (`mla::quant_latent_tests::INT8_TOL`, a relative error of
//! 2e-2 on the latent attention output), widened for a full layer's output. Paths that read the same stored rows (the
//! batched and sequential decode, a restored and a never-parked cache) are
//! held to float accuracy.

use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::mla::MlaDecodePath;
use mlxcel_core::weights::WeightMap;

use super::tests::{insert_dense_layer_weights, ramp, test_config_dense_direct_q};
use super::{DeepSeekV3Attention, DeepSeekV3Config, DeepSeekV3Model};

/// The #3155 int8 codec bound against the unquantized latent, measured on the
/// latent attention output.
const INT8_TOL: f32 = 2e-2;
/// The same bound after the layer's `unembed_out` and `o_proj`: the ramp
/// fixture's projections cancel enough of the output that the codec error
/// reaches 2.9% of its largest element (1.5% on the first step).
const INT8_LAYER_TOL: f32 = 5e-2;

/// A direct-q backbone whose `kv_lora_rank` is one affine group of 64.
fn quantizable_config() -> DeepSeekV3Config {
    DeepSeekV3Config {
        hidden_size: 16,
        intermediate_size: 16,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        kv_lora_rank: 64,
        qk_rope_head_dim: 8,
        qk_nope_head_dim: 8,
        v_head_dim: 8,
        ..test_config_dense_direct_q()
    }
}

fn attention(cfg: &DeepSeekV3Config) -> DeepSeekV3Attention {
    let mut weights = WeightMap::new();
    insert_dense_layer_weights(&mut weights, cfg, 0);
    let weights = DeepSeekV3Model::sanitize_weights(weights, cfg).expect("sanitize must succeed");
    DeepSeekV3Attention::from_weights(&weights, cfg, "model.layers.0.self_attn")
        .expect("tiny attention must load")
}

fn model(cfg: &DeepSeekV3Config) -> DeepSeekV3Model {
    let weights = super::tests::tiny_dense_model_weights(cfg);
    let weights = DeepSeekV3Model::sanitize_weights(weights, cfg).expect("sanitize must succeed");
    DeepSeekV3Model::from_weights(&weights, cfg).expect("tiny model must load")
}

/// Token `t`'s hidden state, varied per token so every latent row differs.
fn hidden_rows(
    cfg: &DeepSeekV3Config,
    from: usize,
    to: usize,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let hidden = cfg.hidden_size;
    let all = ramp((to + 3) * hidden * 3);
    let rows: Vec<f32> = (from..to)
        .flat_map(|t| all[t * 3 * hidden..t * 3 * hidden + hidden].iter().copied())
        .collect();
    mlxcel_core::from_slice_f32(&rows, &[1, (to - from) as i32, hidden as i32])
}

fn values(a: &mlxcel_core::MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(a)
}

fn max_rel_error(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()))
        .max(1e-6);
    got.iter()
        .zip(want)
        .fold(0.0f32, |acc, (g, w)| acc.max((g - w).abs() / scale))
}

/// Prefill `prefill` tokens, then decode one token at a time up to `total`.
/// Returns each decode step's attention output.
fn run_layer(
    attn: &DeepSeekV3Attention,
    cfg: &DeepSeekV3Config,
    cache: &mut KVCache,
    prefill: usize,
    total: usize,
) -> Vec<Vec<f32>> {
    let out = attn.forward(&hidden_rows(cfg, 0, prefill), cache, None);
    mlxcel_core::eval(&out);
    (prefill..total)
        .map(|t| values(&attn.forward(&hidden_rows(cfg, t, t + 1), cache, None)))
        .collect()
}

/// An int8 cache stores the quantized latent, its decode steps take the
/// composed quantized path, and the output tracks the fp16 latent within the
/// codec bound.
#[test]
fn int8_latent_decode_tracks_the_fp16_latent_on_the_composed_path() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let cfg = quantizable_config();
    let attn = attention(&cfg);

    let mut fp16 = KVCache::new();
    let want = run_layer(&attn, &cfg, &mut fp16, 5, 9);
    assert!(!fp16.holds_quantized_latent());

    let mut int8 = KVCache::new_with_mode(KVCacheMode::Int8);
    let (got, counts) = mlxcel_core::mla::stats::scoped(|| run_layer(&attn, &cfg, &mut int8, 5, 9));
    assert!(int8.holds_quantized_latent());
    assert_eq!(int8.seq_len(), 9);
    assert!(
        counts.get(MlaDecodePath::AbsorbedComposed) >= 4,
        "every int8 decode step must take the composed quantized path"
    );

    for (step, (got, want)) in got.iter().zip(&want).enumerate() {
        assert!(got.iter().all(|v| v.is_finite()), "step {step}");
        let err = max_rel_error(got, want);
        assert!(
            err > 0.0,
            "step {step}: the int8 arm must not be the fp16 arm"
        );
        assert!(err < INT8_LAYER_TOL, "step {step}: relative error {err}");
    }
}

/// A multi-token step behind a quantized cache attends over the dequantized
/// window and stays within the codec bound of the fp16 latent.
#[test]
fn a_chunked_prefill_over_an_int8_latent_tracks_the_fp16_latent() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let cfg = quantizable_config();
    let attn = attention(&cfg);
    let chunked = |cache: &mut KVCache| {
        let first = attn.forward(&hidden_rows(&cfg, 0, 4), cache, None);
        mlxcel_core::eval(&first);
        values(&attn.forward(&hidden_rows(&cfg, 4, 7), cache, None))
    };
    let want = chunked(&mut KVCache::new());
    let got = chunked(&mut KVCache::new_with_mode(KVCacheMode::Int8));
    let err = max_rel_error(&got, &want);
    assert!(err < INT8_LAYER_TOL, "relative error {err}");
}

/// The batched decode over quantized rows equals the sequential decode over
/// the same rows, and tracks the fp16-latent batch within the codec bound.
#[test]
fn batched_int8_rows_match_sequential_and_track_the_fp16_latent() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let cfg = quantizable_config();
    let model = model(&cfg);
    let prompts: [&[i32]; 3] = [&[1], &[2, 3], &[4, 5, 6, 7]];
    let next = [7, 8, 9];

    let prefilled = |mode: KVCacheMode| -> Vec<Vec<KVCache>> {
        prompts
            .iter()
            .map(|prompt| {
                let mut caches: Vec<KVCache> = (0..model.num_layers())
                    .map(|_| KVCache::new_with_mode(mode))
                    .collect();
                let ids = mlxcel_core::from_slice_i32(prompt, &[1, prompt.len() as i32]);
                mlxcel_core::eval(&model.forward(&ids, &mut caches, None));
                caches
            })
            .collect()
    };
    let batched = |mode: KVCacheMode| -> Vec<f32> {
        let mut rows = prefilled(mode);
        let mut refs: Vec<&mut [KVCache]> = rows.iter_mut().map(Vec::as_mut_slice).collect();
        let ids = mlxcel_core::from_slice_i32(&next, &[next.len() as i32, 1]);
        let logits = model.forward_batched(&ids, &mut refs, None);
        assert!(
            rows.iter()
                .flatten()
                .all(|cache| { cache.holds_quantized_latent() == (mode == KVCacheMode::Int8) })
        );
        values(&logits)
    };

    let int8_batched = batched(KVCacheMode::Int8);
    let mut int8_rows = prefilled(KVCacheMode::Int8);
    let int8_sequential: Vec<f32> = int8_rows
        .iter_mut()
        .zip(next)
        .flat_map(|(caches, token)| {
            let ids = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
            values(&model.forward(&ids, caches, None))
        })
        .collect();
    let err = max_rel_error(&int8_batched, &int8_sequential);
    assert!(err < 1e-5, "batched vs sequential over int8 rows: {err}");

    let fp16_batched = batched(KVCacheMode::Fp16);
    let err = max_rel_error(&int8_batched, &fp16_batched);
    assert!(err < INT8_TOL, "int8 vs fp16 latent batch: {err}");
}

/// The shared batched helper hands a key selector the row's window as it is
/// stored and attends over what the selector returns (#3163): a selector that
/// keeps one of two quantized rows gives the decode over that row alone.
#[test]
fn a_key_selector_is_applied_to_a_quantized_row_before_it_is_attended() {
    use mlxcel_core::mla::LatentWindow;
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let geometry = crate::models::mla_batched::mla_geometry(2, 64, 8, 8, 8);
    let q_latent = mlxcel_core::from_slice_f32(&ramp(2 * 64), &[1, 2, 1, 64]);
    let q_pe = mlxcel_core::from_slice_f32(&ramp(2 * 8), &[1, 2, 1, 8]);
    let row = |salt: usize, width: usize| {
        let all = ramp(salt + width);
        mlxcel_core::from_slice_f32(&all[salt..], &[1, 1, 1, width as i32])
    };
    let decode = |steps: &[usize], select: Option<&crate::models::mla_batched::MlaRowSelector>| {
        let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
        let mut out = Vec::new();
        for &salt in steps {
            let mut caches = [&mut cache];
            out = values(&crate::models::mla_batched::mla_latent_decode_rows(
                "DeepSeekV3Attention",
                true,
                geometry,
                &q_latent,
                &q_pe,
                &row(salt, 64),
                &row(salt + 5, 8),
                &mut caches,
                1.0,
                select,
            ));
        }
        out
    };
    let seen_quantized = std::cell::Cell::new(false);
    let keep_last = |_row: usize, window: LatentWindow| {
        seen_quantized.set(window.is_quantized());
        let last = window.seq_len() - 1;
        window.gather_rows(&mlxcel_core::from_slice_i32(&[last], &[1, 1, 1, 1]))
    };
    let selected = decode(&[3, 40], Some(&keep_last));
    assert!(
        seen_quantized.get(),
        "the selector must see the stored form"
    );
    let alone = decode(&[40], None);
    let both = decode(&[3, 40], None);
    assert!(max_rel_error(&selected, &alone) < 1e-5);
    assert!(
        max_rel_error(&selected, &both) > 1e-3,
        "the selection must matter"
    );
}

/// An int8 cache of a family that has not opted in (`glm4_moe_lite`,
/// `deepseek_v32`) keeps the per-token path: no latent
/// view, a dense window, and rows stored as the cache's own int8 per-token
/// layout. The opted-in caller gets the quantized window from the same cache
/// mode, so the flag alone separates the two.
#[test]
fn an_int8_cache_of_a_family_that_did_not_opt_in_keeps_the_per_token_path() {
    use mlxcel_core::mla::LatentWindow;
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let geometry = crate::models::mla_batched::mla_geometry(2, 64, 8, 8, 8);
    let step = |opt_in: bool| {
        let kv = mlxcel_core::from_slice_f32(&ramp(64), &[1, 1, 1, 64]);
        let k_pe = mlxcel_core::from_slice_f32(&ramp(8), &[1, 1, 1, 8]);
        let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
        crate::models::mla_batched::store_latent_step(
            &mut cache,
            geometry,
            kv,
            k_pe,
            "glm4_moe_lite",
            opt_in,
        )
    };
    assert!(matches!(step(false), LatentWindow::Dense { .. }));
    assert!(matches!(step(true), LatentWindow::Quantized { .. }));
}

/// A parked and restored quantized latent (the prompt-cache store and the
/// scheduler handoff) decodes exactly like the cache that was never parked.
#[test]
fn a_restored_int8_latent_decodes_identically() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let cfg = quantizable_config();
    let attn = attention(&cfg);
    let mut original = KVCache::new_with_mode(KVCacheMode::Int8);
    let mut twin = KVCache::new_with_mode(KVCacheMode::Int8);
    for cache in [&mut original, &mut twin] {
        mlxcel_core::eval(&attn.forward(&hidden_rows(&cfg, 0, 6), cache, None));
    }
    let mut restored = KVCache::new_with_mode(KVCacheMode::Int8);
    restored
        .install_detached(original.clone_handle())
        .expect("a quantized latent handle installs");
    assert!(restored.holds_quantized_latent());
    let step = hidden_rows(&cfg, 6, 7);
    assert_eq!(
        values(&attn.forward(&step, &mut restored, None)),
        values(&attn.forward(&step, &mut twin, None))
    );
}

/// At the Moonlight / Kimi-VL latent geometry (rank 512, rope 64) the runtime
/// cache allocates 672 bytes per reserved row, the rate `kv_arch` and
/// `mlxcel inspect` report.
#[test]
fn runtime_bytes_per_token_match_the_inspect_rate_at_rank_512() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let cfg = DeepSeekV3Config {
        kv_lora_rank: 512,
        qk_rope_head_dim: 64,
        ..quantizable_config()
    };
    let attn = attention(&cfg);
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    mlxcel_core::eval(&attn.forward(&hidden_rows(&cfg, 0, 10), &mut cache, None));
    assert!(cache.holds_quantized_latent());
    assert_eq!(cache.bytes_per_reserved_token(), 672);
    assert_eq!(
        cache.bytes_per_reserved_token(),
        mlxcel_core::mla::quantized_latent_bytes_per_token(512, 64, 8, 64)
    );
}

/// The latent view refuses a paged-backed cache. `DeepSeekV3Model` never gets
/// one: it does not opt into the paged decode backend and builds plain caches.
#[test]
fn deepseek_v3_latent_caches_are_never_paged_backed() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = model(&quantizable_config());
    assert!(!model.supports_paged_decode_backend());
    assert!(model.make_caches().iter().all(|c| !c.is_paged_backed()));
}

/// A rank the affine codec cannot group is declined by the view; the cache
/// then keeps its own per-token rows and the step still decodes.
#[test]
fn an_ungroupable_rank_falls_back_to_per_token_rows() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let cfg = test_config_dense_direct_q();
    assert_eq!(cfg.kv_lora_rank % 64, 8);
    let attn = attention(&cfg);
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    let steps = run_layer(&attn, &cfg, &mut cache, 3, 5);
    assert!(!cache.holds_quantized_latent());
    assert!(steps.iter().flatten().all(|v| v.is_finite()));
}

/// `--mla-latent-quant int4` on a `DeepSeekV3Attention` family (issue #3179):
/// the `int8` storage packs the latent at 4 bits, the rows read back within
/// the 4-bit codec step of the fp16 latent, every decode step takes the
/// composed path (the fused kernel reads 8-bit codes only), and a decode step
/// matches the dense step over the same rows dequantized.
#[test]
fn a_forced_int4_latent_stores_four_bit_rows_and_decodes_composed() {
    use crate::models::mla_latent_test_support as support;
    use mlxcel_core::mla::{MlaLatentQuant, with_mla_latent_quant};

    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let cfg = quantizable_config();
    let attn = attention(&cfg);
    let mut fp16 = KVCache::new();
    let _ = run_layer(&attn, &cfg, &mut fp16, 5, 9);

    with_mla_latent_quant(MlaLatentQuant::Int4, || {
        let mut int4 = KVCache::new_with_mode(KVCacheMode::Int8);
        mlxcel_core::eval(&attn.forward(&hidden_rows(&cfg, 0, 5), &mut int4, None));
        let (got, counts) = mlxcel_core::mla::stats::scoped(|| {
            (5..9)
                .map(|t| values(&attn.forward(&hidden_rows(&cfg, t, t + 1), &mut int4, None)))
                .collect::<Vec<Vec<f32>>>()
        });
        let delta = |path| counts.get(path);
        assert_eq!(delta(MlaDecodePath::AbsorbedComposed), 4);
        assert_eq!(delta(MlaDecodePath::AbsorbedFused), 0);

        assert_eq!(int4.stored_latent_bits(), Some(4));
        assert_eq!(
            int4.bytes_per_reserved_token(),
            mlxcel_core::mla::quantized_latent_bytes_per_token(64, 8, 4, 64)
        );
        support::assert_stored_columns(&int4, &fp16, 0, 64, support::INT4_ROW_TOL);

        // The dequantized reference: a two-token step dequantizes the window
        // and runs the dense code, and its last position reads the same nine
        // stored rows the last single-token step read packed.
        let mut twin = KVCache::new_with_mode(KVCacheMode::Int8);
        mlxcel_core::eval(&attn.forward(&hidden_rows(&cfg, 0, 7), &mut twin, None));
        let pair = values(&attn.forward(&hidden_rows(&cfg, 7, 9), &mut twin, None));
        let want = support::last_position(&pair, cfg.hidden_size);
        let err = support::max_rel_error(&got[3], &want);
        assert!(
            err < support::SAME_ROWS_TOL,
            "int4 decode off its stored rows by {err}"
        );
    });
    // The option is scoped: the same cache mode stores 8-bit rows outside it.
    let mut int8 = KVCache::new_with_mode(KVCacheMode::Int8);
    let _ = run_layer(&attn, &cfg, &mut int8, 5, 9);
    assert_eq!(int8.stored_latent_bits(), Some(8));
}

/// A bf16 layer over an int8 latent keeps bf16 activations (#3177). The
/// window dequantizes to FP16, and bf16 against FP16 promotes to float32 in
/// MLX, so without the cast back a multi-token step returned float32 and
/// carried it through `o_proj` into the residual stream.
#[test]
fn a_bf16_layer_keeps_bf16_over_an_int8_latent() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let bf16 = mlxcel_core::dtype::BFLOAT16;
    let cfg = quantizable_config();
    let mut weights = WeightMap::new();
    insert_dense_layer_weights(&mut weights, &cfg, 0);
    let weights = DeepSeekV3Model::sanitize_weights(weights, &cfg).expect("sanitize must succeed");
    let weights: WeightMap = weights
        .into_iter()
        .map(|(name, value)| (name, mlxcel_core::astype(&value, bf16)))
        .collect();
    let attn = DeepSeekV3Attention::from_weights(&weights, &cfg, "model.layers.0.self_attn")
        .expect("tiny bf16 attention must load");

    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    let prefill = attn.forward(
        &mlxcel_core::astype(&hidden_rows(&cfg, 0, 5), bf16),
        &mut cache,
        None,
    );
    assert!(cache.holds_quantized_latent());
    assert_eq!(
        mlxcel_core::array_dtype(&prefill),
        bf16,
        "prefill output dtype"
    );
    let decode = attn.forward(
        &mlxcel_core::astype(&hidden_rows(&cfg, 5, 6), bf16),
        &mut cache,
        None,
    );
    assert_eq!(
        mlxcel_core::array_dtype(&decode),
        bf16,
        "decode output dtype"
    );

    // The fp16 latent of the same layer was never promoted.
    let mut fp16 = KVCache::new();
    let prefill = attn.forward(
        &mlxcel_core::astype(&hidden_rows(&cfg, 0, 5), bf16),
        &mut fp16,
        None,
    );
    assert_eq!(mlxcel_core::array_dtype(&prefill), bf16, "fp16 latent");
}
