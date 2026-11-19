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

//! Opt-in KV cache storage arm (#1793).
//!
//! `MLXCEL_TRACE_KV_CACHE_MODE` (for example `fp16+turbo4`) stores the KV
//! cache the way `--kv-cache-mode` does on the server: the nominal mode is
//! resolved per layer with the same boundary rule
//! (`MLXCEL_KV_BOUNDARY_V_LAYERS`) and injected into the model's own caches, so it
//! reaches the model-owned families (Gemma 3, Gemma 4) whose caches the
//! harness's `make_caches()` does not hold. A family that reads the caches the
//! harness hands it (DeepSeek V2, whose `int8` mode stores the MLA
//! latent quantized, issue #3155) gets the same table through
//! [`KvArm::make_caches`], the way the engine sets it on its pool caches.
//!
//! `MLXCEL_TRACE_PREFILL_CHUNK=N` runs the context pass as consecutive
//! forwards of at most `N` tokens, the way the generator prefills a long
//! prompt (`prefill_plan`), instead of one forward over the whole context. A
//! single forward scores `heads * context^2` pairs, which at 16384 tokens on
//! a 128-head model (DeepSeek V3.2) is tens of gigabytes per temporary; both
//! arms of a comparison must use the same value (#3163).
//!
//! `MLXCEL_TRACE_DECODE_STEP=1` feeds the last context token through its own
//! single-token forward. A decode step is what grows a sliding
//! `RotatingKVCache` ahead of `offset`, the state the next multi-token append
//! must read through `visible_len()`; a context pass alone never reaches it.
use anyhow::{Result, ensure};
use mlxcel::{LanguageModel, LoadedModel};
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::layers::KVCache;

/// The storage arm selected by the environment.
pub struct KvArm {
    pub mode: Option<KVCacheMode>,
    pub decode_step: bool,
    /// Largest context forward, in tokens; `None` is one forward.
    context_chunk: Option<usize>,
    /// Per-layer modes for the harness's own caches, set by [`Self::install`]
    /// when the model does not own its KV caches.
    external_modes: Option<Vec<KVCacheMode>>,
}

impl KvArm {
    pub fn from_env() -> Result<Self> {
        let mode = std::env::var("MLXCEL_TRACE_KV_CACHE_MODE")
            .ok()
            .map(|value| {
                value
                    .parse::<KVCacheMode>()
                    .map_err(|err| anyhow::anyhow!("MLXCEL_TRACE_KV_CACHE_MODE={value}: {err}"))
            })
            .transpose()?;
        let decode_step = match std::env::var("MLXCEL_TRACE_DECODE_STEP").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("0") => false,
            Ok("1") => true,
            _ => anyhow::bail!("MLXCEL_TRACE_DECODE_STEP must be 0 or 1"),
        };
        let context_chunk = match std::env::var("MLXCEL_TRACE_PREFILL_CHUNK") {
            Err(_) => None,
            Ok(value) => match value.parse::<usize>() {
                Ok(tokens) if tokens > 0 => Some(tokens),
                _ => anyhow::bail!("MLXCEL_TRACE_PREFILL_CHUNK must be a positive integer"),
            },
        };
        Ok(Self {
            mode,
            decode_step,
            context_chunk,
            external_modes: None,
        })
    }

    /// Inject the resolved per-layer modes into the model's own caches.
    pub fn install(&mut self, model: &LoadedModel) -> Result<()> {
        let Some(mode) = self.mode else {
            return Ok(());
        };
        let n_layers = model.num_layers();
        ensure!(n_layers > 0, "model reports no layers to store");
        let requested = mlxcel_core::cache::turbo::boundary_v_layers_from_env();
        let modes = mlxcel_core::cache::turbo::resolve_layer_modes(mode, n_layers, requested);
        model.set_kv_cache_layer_modes(modes.clone());
        let Some(applied) = model.kv_cache_layer_modes() else {
            // The model reads the caches it is handed, so the harness builds
            // them in the resolved modes instead.
            self.external_modes = Some(modes);
            return Ok(());
        };
        ensure!(
            applied == modes,
            "the model kept KV cache modes {applied:?}, not the requested {modes:?}"
        );
        // Sliding-window layers store FP16 for every mode but an aligned
        // fp16+turbo4 (issue #2783), mode-blind caches store FP16, and
        // linear, recurrent or convolution layers store no KV at all (issue
        // #2790); say so instead of implying they quantize.
        if let Some(storage) = model.kv_cache_storage_modes()
            && storage
                .iter()
                .zip(&applied)
                .any(|(stored, table)| *stored != Some(*table))
        {
            let stored = storage
                .iter()
                .filter(|stored| **stored == Some(mode))
                .count();
            let with_kv = storage.iter().filter(|stored| stored.is_some()).count();
            let fallback = storage
                .iter()
                .zip(&applied)
                .filter(|(stored, table)| stored.is_some_and(|stored| stored != **table))
                .count();
            let stateless = storage.len() - with_kv;
            eprintln!(
                "kv cache storage: {stored}/{with_kv} layers {mode}, {fallback} layers fp16, \
                 {stateless} stateless layers"
            );
        }
        Ok(())
    }

    /// Fresh caches for one sequence, in the installed storage modes when the
    /// model reads external caches.
    pub fn make_caches(&self, model: &LoadedModel) -> Vec<KVCache> {
        let mut caches = model.make_caches();
        if let Some(modes) = &self.external_modes {
            for (cache, mode) in caches.iter_mut().zip(modes) {
                cache.mode = *mode;
            }
        }
        caches
    }

    /// The context pass's multi-token part as the forwards to run, in order:
    /// one piece, or `MLXCEL_TRACE_PREFILL_CHUNK`-token pieces.
    pub fn context_pieces<'a>(&self, context: &'a [i32]) -> std::slice::Chunks<'a, i32> {
        context.chunks(self.context_chunk.unwrap_or(context.len()).max(1))
    }

    /// Where the context pass splits: everything before is one forward, the
    /// rest is the single-token decode step.
    pub fn context_split(&self, context_len: usize) -> usize {
        if self.decode_step && context_len > 1 {
            context_len - 1
        } else {
            context_len
        }
    }
}
