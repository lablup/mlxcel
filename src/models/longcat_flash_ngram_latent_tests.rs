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

//! The affine int8 MLA latent on LongCat-Flash and LongCat-Flash-Ngram
//! (issue #3161).
//!
//! Both types load `LongcatFlashNgramModel`, whose caches are model-owned, so
//! the mode table the engine injects is the only way a quantized latent
//! reaches them. The quality reference is the fp16 latent of the same
//! weights, held to the #3155 codec bound widened for full-model logits.
//!
//! The port is in place but off in production: LongCat is not in
//! `mla::QUANTIZED_LATENT_FAMILIES` until a real checkpoint passes the trace
//! gate, so these tests opt the model in (`quantized_latent`), and
//! [`production_longcat_resolves_int8_to_fp16`] pins the default.

use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::engine::DirectEngine;
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::mla::MlaDecodePath;

use super::LongcatFlashNgramModel;
use super::engine_tests::{TOKENS, ids, row, tiny_model_with_rank, tiny_model_with_rank_in};

/// One affine group: the smallest rank the int8 latent stores.
const QUANTIZABLE_RANK: i32 = 64;
/// The #3155 codec bound on full-model logits, relative to the largest fp16
/// logit of the step.
const INT8_LOGIT_TOL: f32 = 5e-2;

const FAMILIES: [&str; 2] = ["longcat_flash", "longcat_flash_ngram"];

/// [`tiny_model_with_rank`] opted in to the quantized latent.
fn opted_in(family: &str, rank: i32) -> LongcatFlashNgramModel {
    let model = tiny_model_with_rank(family, rank);
    model.quantized_latent.set(true);
    model
}

fn holds_quantized_latent(model: &LongcatFlashNgramModel) -> bool {
    let state = model.state.borrow();
    !state.caches.is_empty()
        && state
            .caches
            .iter()
            .flatten()
            .all(|cache| cache.holds_quantized_latent())
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

/// Prefill `TOKENS[..prefill]`, then teacher-force the rest one token at a
/// time; returns each decode step's logits.
fn teacher_forced(model: &LongcatFlashNgramModel, prefill: usize) -> Vec<Vec<f32>> {
    let _ = model.make_caches();
    mlxcel_core::eval(&model.forward(&ids(&TOKENS[..prefill]), &mut [], None));
    TOKENS[prefill..]
        .iter()
        .map(|token| row(&model.forward(&ids(&[*token]), &mut [], None), 0))
        .collect()
}

/// The production model follows the allowlist: LongCat is not a quantized
/// latent family, so an injected `int8` table is reported and built as fp16,
/// and an `int8` engine run is the fp16 run.
#[test]
fn production_longcat_resolves_int8_to_fp16() {
    let greedy = SamplingConfig::greedy();
    for family in FAMILIES {
        assert!(!mlxcel_core::mla::stores_quantized_latent_pair(family));
        let model = tiny_model_with_rank(family, QUANTIZABLE_RANK);
        assert!(!model.quantized_latent.get(), "{family}");
        let n = model.num_layers();
        model.set_kv_cache_layer_modes(vec![KVCacheMode::Int8; n]);
        assert_eq!(
            model.kv_cache_layer_modes(),
            Some(vec![KVCacheMode::Fp16; n])
        );

        let fp16_model = tiny_model_with_rank(family, QUANTIZABLE_RANK);
        let fp16 = DirectEngine::new(&fp16_model, 0)
            .run(&TOKENS[..6], 6, &greedy)
            .expect("fp16 run");
        let int8_model = tiny_model_with_rank(family, QUANTIZABLE_RANK);
        let int8 = DirectEngine::new(&int8_model, 0)
            .with_kv_cache_mode(KVCacheMode::Int8)
            .run(&TOKENS[..6], 6, &greedy)
            .expect("int8 run");
        assert!(!holds_quantized_latent(&int8_model), "{family}");
        assert_eq!(int8, fp16, "{family}");
    }
}

/// An opted-in model keeps an int8 table, builds both sub-layer caches of
/// every layer in it, and reports it (the pre-#3161 model refused it and
/// reported fp16).
#[test]
fn an_int8_table_is_kept_and_reported() {
    for family in FAMILIES {
        let model = opted_in(family, QUANTIZABLE_RANK);
        let n = model.num_layers();
        assert_eq!(
            model.kv_cache_layer_modes(),
            Some(vec![KVCacheMode::Fp16; n])
        );

        model.set_kv_cache_layer_modes(vec![KVCacheMode::Int8; n]);
        assert_eq!(
            model.kv_cache_layer_modes(),
            Some(vec![KVCacheMode::Int8; n]),
            "{family} must report the int8 latent it stores"
        );
        let _ = model.make_caches();
        assert!(
            model
                .state
                .borrow()
                .caches
                .iter()
                .flatten()
                .all(|cache| cache.mode == KVCacheMode::Int8),
            "{family}: both sub-layer caches of every layer follow the table"
        );
        mlxcel_core::eval(&model.forward(&ids(&TOKENS[..4]), &mut [], None));
        assert!(holds_quantized_latent(&model), "{family}");
    }
}

/// A Turbo mode the latent pair cannot hold is replaced by fp16 per layer,
/// and an int8 layer beside it is kept.
#[test]
fn a_turbo_layer_falls_back_to_fp16_and_int8_is_kept() {
    let model = opted_in("longcat_flash_ngram", QUANTIZABLE_RANK);
    let n = model.num_layers();
    assert!(n >= 2);
    let mut modes = vec![KVCacheMode::Int8; n];
    modes[0] = KVCacheMode::Turbo4Asym;
    model.set_kv_cache_layer_modes(modes);
    let mut want = vec![KVCacheMode::Int8; n];
    want[0] = KVCacheMode::Fp16;
    assert_eq!(model.kv_cache_layer_modes(), Some(want));
}

/// Int8 decode steps take the absorbed quantized path, never the decompressed
/// one, and the logits track the fp16 latent within the codec bound.
#[test]
fn int8_decode_tracks_the_fp16_latent_on_the_absorbed_path() {
    for family in FAMILIES {
        let fp16 = opted_in(family, QUANTIZABLE_RANK);
        let want = teacher_forced(&fp16, 4);

        let int8 = opted_in(family, QUANTIZABLE_RANK);
        int8.set_kv_cache_layer_modes(vec![KVCacheMode::Int8; int8.num_layers()]);
        let _ = int8.make_caches();
        mlxcel_core::eval(&int8.forward(&ids(&TOKENS[..4]), &mut [], None));
        let before = mlxcel_core::mla::stats::snapshot();
        let got: Vec<Vec<f32>> = TOKENS[4..]
            .iter()
            .map(|token| row(&int8.forward(&ids(&[*token]), &mut [], None), 0))
            .collect();
        let after = mlxcel_core::mla::stats::snapshot();
        assert!(holds_quantized_latent(&int8), "{family}");

        let delta = |path| after.get(path).saturating_sub(before.get(path));
        let steps = (TOKENS.len() - 4) as u64;
        let sublayers = 2 * int8.num_layers() as u64;
        assert!(
            delta(MlaDecodePath::AbsorbedComposed) + delta(MlaDecodePath::AbsorbedFused)
                >= steps * sublayers,
            "{family}: every int8 decode step must read the quantized latent absorbed"
        );
        assert_eq!(delta(MlaDecodePath::Decompressed), 0, "{family}");

        for (step, (got, want)) in got.iter().zip(&want).enumerate() {
            let err = max_rel_error(got, want);
            assert!(
                err > 0.0,
                "{family} step {step}: int8 must not be the fp16 arm"
            );
            assert!(
                err < INT8_LOGIT_TOL,
                "{family} step {step}: relative error {err}"
            );
        }
    }
}

/// The engine path: `--kv-cache-mode int8` reaches the model-owned caches
/// through `DirectEngine`, and the greedy run matches the fp16 greedy run on
/// this fixture (every step's top two are far apart at this codec error).
#[test]
fn an_int8_engine_run_stores_the_latent_and_matches_fp16_greedy() {
    let greedy = SamplingConfig::greedy();
    for family in FAMILIES {
        let fp16_model = opted_in(family, QUANTIZABLE_RANK);
        let fp16 = DirectEngine::new(&fp16_model, 0)
            .run(&TOKENS[..6], 6, &greedy)
            .expect("fp16 run");

        let int8_model = opted_in(family, QUANTIZABLE_RANK);
        let int8 = DirectEngine::new(&int8_model, 0)
            .with_kv_cache_mode(KVCacheMode::Int8)
            .run(&TOKENS[..6], 6, &greedy)
            .expect("int8 run");
        assert_eq!(
            int8_model.kv_cache_layer_modes(),
            Some(vec![KVCacheMode::Int8; int8_model.num_layers()])
        );
        assert!(holds_quantized_latent(&int8_model), "{family}");
        assert!(!holds_quantized_latent(&fp16_model), "{family}");
        assert_eq!(int8, fp16, "{family}");
    }
}

/// At the LongCat-Flash latent geometry (rank 512) each sub-layer cache of an
/// opted-in model allocates the affine rate per reserved row, so the model's
/// per-token footprint is two such caches per layer: the count `kv_arch`
/// uses (`longcat_counts_two_latent_caches_per_layer`).
#[test]
fn runtime_bytes_per_token_are_two_affine_caches_per_layer() {
    let model = opted_in("longcat_flash", 512);
    model.set_kv_cache_layer_modes(vec![KVCacheMode::Int8; model.num_layers()]);
    let _ = model.make_caches();
    mlxcel_core::eval(&model.forward(&ids(&TOKENS[..5]), &mut [], None));
    let per_cache = mlxcel_core::mla::quantized_latent_bytes_per_token(512, 4, 8, 64);
    let state = model.state.borrow();
    assert_eq!(
        state.caches.iter().flatten().count(),
        2 * model.num_layers()
    );
    for cache in state.caches.iter().flatten() {
        assert!(cache.holds_quantized_latent());
        assert_eq!(cache.bytes_per_reserved_token(), per_cache);
    }
}

/// A bf16 model keeps bf16 activations on the int8 latent: a multi-token step
/// reads the dequantized window in the queries' dtype, so the prefill and a
/// multi-token continuation return logits in the fp16 arm's dtype instead of
/// promoting the rest of the network to f32 (#3161).
#[test]
fn a_bf16_model_stays_bf16_over_the_int8_latent() {
    for family in FAMILIES {
        let model = tiny_model_with_rank_in(family, QUANTIZABLE_RANK, mlxcel_core::dtype::BFLOAT16);
        model.quantized_latent.set(true);
        let n = model.num_layers();
        for mode in [KVCacheMode::Fp16, KVCacheMode::Int8] {
            model.set_kv_cache_layer_modes(vec![mode; n]);
            let prefill = model.forward(&ids(&TOKENS[..4]), &mut [], None);
            let decode = model.forward(&ids(&TOKENS[4..5]), &mut [], None);
            let continuation = model.forward(&ids(&TOKENS[5..7]), &mut [], None);
            assert_eq!(
                holds_quantized_latent(&model),
                mode == KVCacheMode::Int8,
                "{family} {mode}"
            );
            for (what, logits) in [
                ("prefill", &prefill),
                ("decode", &decode),
                ("continuation", &continuation),
            ] {
                assert_eq!(
                    mlxcel_core::array_dtype(logits),
                    mlxcel_core::dtype::BFLOAT16,
                    "{family} {mode}: {what} logits left bf16"
                );
            }
        }
    }
}
