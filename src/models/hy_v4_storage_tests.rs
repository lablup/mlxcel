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

//! Hy4 storage and dtype tests over the synthetic two-layer checkpoint of
//! `hy_v4_tests.rs` (issue #2354): the quantized MLA latent and bf16
//! activations.

use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::mla::{LatentWindow, MlaLatentQuant, with_mla_latent_quant};

use super::hy_v4_tests::{
    PROMPT, Tiny, ids, last_logits, max_abs_diff, tiny_config, tiny_model_with, tiny_weights,
    values,
};
use super::{HyV4Model, ParityArms};

/// `--mla-latent-quant int8` / `int4`: the caches hold the packed latent
/// (the full layer with its indexer key, the shared layer without), a decode
/// step reads it, and int8 stays close to the fp16 latent.
#[test]
fn quantized_latent_caches_are_read_by_prefill_and_decode() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let tiny = Tiny {
        kv_lora: 64,
        index_head_dim: 64,
        ..Tiny::new(2)
    };
    let model = tiny_model_with(tiny, ParityArms::default());
    let fp16 = last_logits(&model, &mut model.make_caches(), &[PROMPT, &[2], &[8]]);

    for (option, bits) in [(MlaLatentQuant::Int8, 8), (MlaLatentQuant::Int4, 4)] {
        with_mla_latent_quant(option, || {
            let mut caches: Vec<KVCache> = (0..model.num_layers())
                .map(|_| KVCache::new_with_mode(KVCacheMode::Int8))
                .collect();
            let logits = last_logits(&model, &mut caches, &[PROMPT, &[2], &[8]]);

            let widths: Vec<i32> = caches
                .iter()
                .map(|cache| {
                    assert!(cache.holds_quantized_latent(), "int{bits}: fp16 rows");
                    assert_eq!(cache.stored_latent_bits(), Some(bits));
                    LatentWindow::stored(cache)
                        .expect("quantized window")
                        .key_width()
                })
                .collect();
            assert_eq!(widths, [128, 64], "int{bits}: [full, shared] key widths");

            let scale = fp16.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let diff = max_abs_diff(&fp16, &logits);
            if bits == 8 {
                assert!(
                    diff < 0.05 * scale,
                    "int8 moved logits by {diff} of {scale}"
                );
            }
        });
    }
}

/// The shipped checkpoint is bf16 with float32 `hc_*`, sink and router-bias
/// tensors. None of those may promote the residual stream: a float32 leak
/// would run every later layer of the real model in float32. Checked on a
/// sparse prefill, a dense prefill and a decode step of each.
#[test]
fn bf16_activations_stay_bf16_beside_the_float32_tensors() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let bf16 = mlxcel_core::dtype::BFLOAT16;
    for index_topk in [2, 2048] {
        let tiny = Tiny::new(index_topk);
        let mut weights = tiny_weights(tiny);
        for (key, value) in weights.iter_mut() {
            let keeps_f32 = key.contains(".hc_")
                || key.contains("learnable_sink_param")
                || key.contains("e_score_correction_bias");
            if !keeps_f32 {
                *value = mlxcel_core::astype(value, bf16);
            }
        }
        let model =
            HyV4Model::from_weights_with_arms(&weights, &tiny_config(tiny), ParityArms::default())
                .expect("bf16 tiny model loads");
        let reference = tiny_model_with(tiny, ParityArms::default());

        let mut caches = model.make_caches();
        let mut reference_caches = reference.make_caches();
        for chunk in [PROMPT, &[2]] {
            let logits = model.forward(&ids(chunk), &mut caches, None);
            assert_eq!(
                mlxcel_core::array_dtype(&logits),
                bf16,
                "top-k {index_topk}, {} token(s): logits were promoted",
                chunk.len()
            );
            let expected = values(&reference.forward(&ids(chunk), &mut reference_caches, None));
            let got = values(&logits);
            assert!(got.iter().all(|v| v.is_finite()));
            let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let diff = max_abs_diff(&got, &expected);
            assert!(
                diff < 0.1 * scale,
                "bf16 differs from f32 by {diff} of {scale}"
            );
        }
    }
}
