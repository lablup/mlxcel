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

//! `--mla-latent-quant` on GLM-4.7-Flash (issue #3179), the family the
//! default gate keeps at fp16.

use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::mla::{MlaDecodePath, MlaLatentQuant, with_mla_latent_quant};

use super::{DECODE, LAYERS, PROMPT, RANK, ROPE, caches, ids, max_rel_error, model_with, row};
use crate::models::mla_latent_test_support as support;

/// `--mla-latent-quant int8` and `int4` on GLM-4.7-Flash (issue #3179). The
/// production model reads the option at load, so a model built under it
/// stores the affine latent the default gate keeps off: 8-bit rows under
/// `int8`, 4-bit rows under `int4`, read back within the codec step of the
/// fp16 latent. A decode step matches the dense step over the same rows
/// dequantized, and `fp16` keeps the per-token path of the default.
#[test]
fn a_forced_latent_precision_switches_the_production_model_onto_the_view() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let reference = model_with(RANK, ROPE);
    let mut fp16 = caches(KVCacheMode::Fp16);
    mlxcel_core::eval(&reference.forward(&ids(&PROMPT), &mut fp16, None));

    for (option, bits, tol) in [
        (MlaLatentQuant::Int8, 8, support::INT8_ROW_TOL),
        (MlaLatentQuant::Int4, 4, support::INT4_ROW_TOL),
    ] {
        with_mla_latent_quant(option, || {
            let model = model_with(RANK, ROPE);
            assert!(model.layers.iter().all(|l| l.self_attn.quantized_latent));
            let mut forced = caches(KVCacheMode::Int8);
            mlxcel_core::eval(&model.forward(&ids(&PROMPT), &mut forced, None));
            for cache in &forced {
                assert_eq!(cache.stored_latent_bits(), Some(bits), "{option}");
                assert_eq!(
                    cache.bytes_per_reserved_token(),
                    mlxcel_core::mla::quantized_latent_bytes_per_token(
                        RANK as usize,
                        ROPE as usize,
                        bits as usize,
                        64
                    ),
                    "{option}"
                );
            }
            // Layer 0 saw the same inputs as the fp16 model's layer 0.
            support::assert_stored_columns(&forced[0], &fp16[0], 0, RANK, tol);

            // One step read packed against the same step read dequantized
            // (position 0 of a two-token step over an identical history).
            let mut twin = caches(KVCacheMode::Int8);
            mlxcel_core::eval(&model.forward(&ids(&PROMPT), &mut twin, None));
            let before = mlxcel_core::mla::stats::snapshot();
            let got = row(&model.forward(&ids(&[DECODE[0]]), &mut forced, None), 0);
            let after = mlxcel_core::mla::stats::snapshot();
            let delta = |path| after.get(path).saturating_sub(before.get(path));
            if bits == 4 {
                assert_eq!(delta(MlaDecodePath::AbsorbedComposed), LAYERS as u64);
                assert_eq!(delta(MlaDecodePath::AbsorbedFused), 0);
            }
            assert_eq!(delta(MlaDecodePath::Decompressed), 0);
            let want = row(&model.forward(&ids(&DECODE[..2]), &mut twin, None), 0);
            let err = max_rel_error(&got, &want);
            assert!(
                err < support::SAME_ROWS_TOL,
                "{option}: decode off its stored rows by {err}"
            );
        });
    }

    with_mla_latent_quant(MlaLatentQuant::Fp16, || {
        let model = model_with(RANK, ROPE);
        assert!(model.layers.iter().all(|l| !l.self_attn.quantized_latent));
    });
}
