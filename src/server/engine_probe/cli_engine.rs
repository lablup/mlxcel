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

//! The CLI decode path, called the way its two real callers call it.
//!
//! [`generate_like_cli`] is `mlxcel generate`'s text path without
//! `--profile` (`generate_standard` in `src/commands/generate.rs`): an
//! inference session from [`select_backend`], a one-token warmup, a reset,
//! then [`Session::generate`] on `CxxGenerator`. [`bench_like_bench_decode`]
//! is `mlxcel-bench-decode`'s warmup and measured pass, which time
//! `CxxGenerator::generate_with_stats`, so the numbers it returns are the ones
//! `scripts/bench_decode.sh` records.
//!
//! [`Session::generate`]: crate::backend::Session::generate

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Result;
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::generate::{GenerationStats, LanguageModel, SamplingConfig};
use mlxcel_core::sampling::TokenBiasMap;

use crate::backend::select_backend;
use crate::{CxxGenerator, LoadedModel};

/// Generate through the `mlxcel generate` text path and return the tokens.
pub fn generate_like_cli(
    model: &LoadedModel,
    model_path: &Path,
    prompt_tokens: &[i32],
    max_tokens: usize,
    sampling: &SamplingConfig,
    kv_cache_mode: KVCacheMode,
) -> Result<Vec<i32>> {
    let mut session = select_backend().create_session(
        model_path,
        model.num_layers(),
        kv_cache_mode,
        TokenBiasMap::default(),
    )?;
    // Same one-token warmup and reset `generate_standard` runs before the
    // generation it reports.
    let _ = session.generate(model, prompt_tokens, 1, sampling);
    session.reset_with_model(model);
    Ok(session.generate(model, prompt_tokens, max_tokens, sampling))
}

/// Run `mlxcel-bench-decode`'s warmup pass and measured pass and return the
/// measured tokens and statistics.
pub fn bench_like_bench_decode(
    model: &LoadedModel,
    prompt_tokens: &[i32],
    max_tokens: usize,
    warmup_tokens: usize,
    sampling: &SamplingConfig,
    kv_cache_mode: KVCacheMode,
) -> (Vec<i32>, GenerationStats) {
    if warmup_tokens > 0 {
        let mut warm = CxxGenerator::new_with_kv_mode(model.num_layers(), kv_cache_mode);
        let _ = warm.generate(model, prompt_tokens, warmup_tokens, sampling);
        // Reset model-owned state so the measured pass starts clean while the
        // process keeps its warm allocator and kernel caches.
        warm.reset_with_model(model);
        mlxcel_core::synchronize_default();
    }
    let mut generator = CxxGenerator::new_with_kv_mode(model.num_layers(), kv_cache_mode);
    let measured = generator.generate_with_stats(model, prompt_tokens, max_tokens, sampling);
    mlxcel_core::synchronize_default();
    measured
}

/// Suppress every end-of-generation token on the CLI path the way
/// `mlxcel-bench-decode --ignore-eos` does: a `-inf` bias on the union of the
/// model's built-in EOS ids and the configured stop ids, which is the set the
/// generator stops on.
pub fn suppress_eos_for_cli(sampling: &mut SamplingConfig, model: &LoadedModel) {
    let ids: BTreeSet<i32> = model
        .eos_token_ids()
        .into_iter()
        .chain(sampling.stop_token_ids.iter().copied())
        .collect();
    for id in ids {
        sampling.token_bias.insert(id, f32::NEG_INFINITY);
    }
}
