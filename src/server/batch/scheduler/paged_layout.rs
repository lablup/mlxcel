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

use super::*;

impl BatchScheduler {
    pub(super) fn allocate_sequence_state(&mut self) -> Result<SequenceId, String> {
        let layout_override = self.sequence_state_layout_override();
        let seq_id = self
            .cache_pool
            .allocate_with_layout(&self.model, layout_override)?;
        // apply the configured KV cache mode (with
        // Boundary-V policy for Turbo4 modes) to the freshly allocated
        // per-layer caches. `model.make_caches()` always returns Fp16
        // caches; this step upgrades them to the requested mode while
        // keeping boundary layers at FP16 quality. No-op when the
        // configured mode is `Fp16`.
        self.apply_kv_cache_mode_to(seq_id);
        self.model.prepare_sequence_state(seq_id);
        Ok(seq_id)
    }

    /// Apply the configured `kv_cache_mode` to every per-layer cache of
    /// `seq_id` (with the Boundary-V upgrade for Turbo4 modes).
    ///
    /// No-op for `KVCacheMode::Fp16` — the cache pool already returns
    /// FP16 caches from `model.make_caches()`. For Turbo4 modes the
    /// per-layer mode is resolved via
    /// [`mlxcel_core::cache::turbo::resolve_layer_modes`] so boundary
    /// layers stay FP16 for quality.
    ///
    /// when [`BatchKvQuantConfig::is_enabled`] returns
    /// `true`, the per-layer mode table is sourced from
    /// [`BatchKvQuantConfig::resolve_layer_modes`] instead — that table
    /// honours the `skip_last_layer` policy, which is distinct from
    /// (and composes with) the existing Boundary-V mechanism.
    ///
    /// Used by: [`Self::allocate_sequence_state`].
    pub(super) fn apply_kv_cache_mode_to(&mut self, seq_id: SequenceId) {
        let batch_kv_quant = self.batch_kv_quant;
        let kv_cache_mode = self.kv_cache_mode;
        let requested_boundary_layers = mlxcel_core::cache::turbo::boundary_v_layers_from_env();
        let Some(caches) = self.cache_pool.get_caches_mut(seq_id) else {
            return;
        };
        if caches.is_empty() {
            // Model-owned / paged sequences without dense placeholder
            // caches — nothing to upgrade. The model's own decode path
            // is responsible for honoring the configured mode.
            return;
        }
        let layer_modes = if batch_kv_quant.is_enabled() {
            batch_kv_quant.resolve_layer_modes(caches.len())
        } else {
            mlxcel_core::cache::turbo::resolve_layer_modes(
                kv_cache_mode,
                caches.len(),
                requested_boundary_layers,
            )
        };
        for (cache, mode) in caches.iter_mut().zip(layer_modes) {
            cache.mode = mode;
        }
    }

    pub(super) fn resolved_kv_cache_layer_modes(&self, n_layers: usize) -> Vec<KVCacheMode> {
        if self.batch_kv_quant.is_enabled() {
            return self.batch_kv_quant.resolve_layer_modes(n_layers);
        }
        let requested = mlxcel_core::cache::turbo::boundary_v_layers_from_env();
        mlxcel_core::cache::turbo::resolve_layer_modes(self.kv_cache_mode, n_layers, requested)
    }

    pub(super) fn configured_model_kv_cache_layer_modes(&self) -> Vec<KVCacheMode> {
        self.resolved_kv_cache_layer_modes(self.model.num_layers())
    }

    pub(super) fn inject_model_owned_kv_cache_modes(&self) {
        self.model
            .set_kv_cache_layer_modes(self.configured_model_kv_cache_layer_modes());
    }

    pub(super) fn log_effective_kv_cache_application(&self) {
        let modes = self.configured_model_kv_cache_layer_modes();
        let effective_mode = if self.batch_kv_quant.is_enabled() {
            self.batch_kv_quant.base_mode()
        } else {
            self.kv_cache_mode
        };
        let applied = modes.iter().filter(|mode| **mode == effective_mode).count();
        tracing::info!(
            kv_cache_mode_effective = %effective_mode,
            kv_cache_mode_applied_layers = applied,
            kv_cache_mode_total_layers = modes.len(),
            "resolved KV cache mode applied to model caches"
        );
    }

    pub(super) fn validate_dense_detached_kv_modes(
        &self,
        dense: &mlxcel_core::cache::DetachedCacheSet,
    ) -> Result<(), String> {
        let expected = self.resolved_kv_cache_layer_modes(dense.num_layers());
        validate_dense_detached_kv_modes_against_table(dense, &expected)
    }

    /// Prepare Turbo4Delegated cache state before a sequence enters decode.
    ///
    /// `finish_prefill` has already emitted the first token; when that token
    /// also satisfies `max_tokens` or EOS, the sequence never decodes and this
    /// helper is intentionally not called.
    pub(super) fn prepare_turbo4_delegated_for_sequence_decode(&mut self, seq_id: SequenceId) {
        let Some(caches) = self.cache_pool.get_caches_mut(seq_id) else {
            return;
        };
        for cache in caches {
            cache.prepare_turbo4_delegated_for_decode();
        }
    }

    /// Enforce the `--max-kv-size` cap on a sequence's KV caches.
    ///
    /// Trims `live_len(cache) - max_kv_size` tokens from every plain
    /// `KVCache` layer whose live window exceeds the configured bound,
    /// pinning a small leading attention-sink prefix
    /// (`MAX_KV_SIZE_SINK_KEEP`) and dropping the excess tokens that
    /// follow it rather than the oldest tokens overall (issue #718). Turbo-mode
    /// caches return `0` from `KVCache::trim_front_keep_sink` (safe no-op,
    /// see [`KVCache::trim_front_keep_sink`] for the per-mode support
    /// matrix). Sliding-window models manage their own internal
    /// `RotatingKVCache` and are never stored in the pool's
    /// `Vec<KVCache>`, so they are unaffected.
    ///
    /// ** H1**: `max_kv_size` has already been validated to fit
    /// in `i32` by [`crate::server::cli_input::resolve_max_kv_size`], so
    /// the `i32::try_from` here is a defensive belt-and-suspenders fallback
    /// — it returns silently rather than panicking, because the validation
    /// at startup ensures we never reach the failure branch in practice.
    /// We compare against `cache.live_len()` (not `cache.offset`) so the
    /// cap is enforced on the **live window** — the monotonic `offset`
    /// keeps growing past the cap by design.
    ///
    /// ** H2**: called from every cache-mutating path:
    /// [`Self::execute_full_prefill`], [`Self::start_chunked_prefill`],
    /// [`Self::continue_chunked_prefill`], [`Self::decode_single_step`],
    /// and [`Self::execute_batched_decode`].
    pub(super) fn enforce_max_kv_size_for(&mut self, seq_id: SequenceId) {
        let Some(max) = self.max_kv_size else {
            return;
        };
        // Defensive: even though `resolve_max_kv_size` already clamps this
        // to `i32::MAX` at startup, a future caller that bypasses the CLI
        // validation could still construct an out-of-range scheduler. Use
        // `checked` arithmetic so the worst case is a no-op trim rather
        // than a wraparound that corrupts every cache.
        let Ok(max_i32) = i32::try_from(max) else {
            tracing::error!(
                "--max-kv-size value {max} does not fit in i32; skipping trim. \
                 This should have been rejected by ServerStartupInput::into_startup_config — \
                 please file a bug if you see this in production."
            );
            return;
        };
        let Some(caches) = self.cache_pool.get_caches_mut(seq_id) else {
            return;
        };
        // Pin a small attention-sink prefix so the trimmed window keeps the
        // leading tokens the model attends to (issue #718). Never large enough
        // to leave no room for the recent window under the configured cap.
        let sink_keep = MAX_KV_SIZE_SINK_KEEP.min(max_i32 - 1).max(0);
        for cache in caches {
            // `live_len() = offset - live_start`. We trim against the live
            // window length (what attention sees), not the monotonic
            // `offset` (which keeps growing by design — RoPE invariant in `KVCache::trim_front`).
            let live_len = cache.live_len();
            // `checked_sub` so a future arithmetic regression cannot
            // silently wrap into a negative trim depth that produces a
            // 4-billion-element slice and crashes Metal.
            if let Some(excess) = live_len.checked_sub(max_i32)
                && excess > 0
            {
                cache.trim_front_keep_sink(excess, sink_keep);
            }
        }
    }

    pub(super) fn sequence_state_layout_override(&self) -> Option<SequenceStateLayout> {
        if self.decode_storage_backend != DecodeStorageBackend::Paged {
            return None;
        }

        let num_layers = self.model.num_layers();
        // prefer the batched KV quant config when active so
        // its `base_mode()` drives paged-layout selection (Turbo-aware
        // when scheme is TurboQuant, otherwise the legacy uniform path).
        let effective_mode = if self.batch_kv_quant.is_enabled() {
            self.batch_kv_quant.base_mode()
        } else {
            self.kv_cache_mode
        };
        // when a Turbo4 cache mode is configured, build a
        // packed-aware paged layout so per-page
        // sidecar accounting and detach/adopt round-trip work correctly.
        // Fp16/Int8 keep the historical `PagedKvLayout::uniform` path —
        // bit-identical to earlier.
        let paged_layout = if Self::is_turbo_mode(effective_mode) {
            // The actual per-token packed sidecar size depends on the
            // model's V head_dim, which is not known to the scheduler
            // at construction time (the dense `KVCache::update_turbo4_*`
            // path lazily allocates the right shape on first write).
            // We charge a per-block budget equal to `DEFAULT_PAGED_BLOCK_SIZE`
            // as a placeholder so the layout passes the
            // `bytes % block_size == 0` validation in
            // [`PagedKvLayout::new_with_mode`]; the runtime
            // `turbo_sidecars.nbytes()` reports the true footprint via
            // `CachePool::memory_usage_bytes`.
            let sidecar_bytes_per_block = DEFAULT_PAGED_BLOCK_SIZE;
            PagedKvLayout::uniform_with_mode(
                num_layers,
                DEFAULT_PAGED_BLOCK_SIZE,
                DEFAULT_PAGED_BLOCK_SIZE,
                effective_mode,
                sidecar_bytes_per_block,
            )
            .expect("valid paged Turbo4 decode layout")
        } else {
            // Carry the actual cache mode so the pool-backing gate in
            // `CachePool::allocate_with_layout` (`paged_layout.cache_mode ==
            // Fp16`) pool-backs ONLY genuine Fp16 sequences. Int8 (and
            // Turbo3Asym) keep their dense KV path — memory saving preserved —
            // until the pool gains native quantized storage; these modes carry
            // no per-page sidecars, so the sidecar budget is 0 (uniform_with_mode
            // treats non-Turbo4 modes as sidecar-free).
            PagedKvLayout::uniform_with_mode(
                num_layers,
                DEFAULT_PAGED_BLOCK_SIZE,
                DEFAULT_PAGED_BLOCK_SIZE,
                effective_mode,
                0,
            )
            .expect("valid paged decode layout")
        };
        Some(SequenceStateLayout::paged_kv_cache(paged_layout))
    }

    /// Whether the supplied KV cache mode requires Turbo*-aware paged
    /// layout (per-page sidecar storage on `PagedBlockPool`) and is
    /// **incompatible** with the `--max-kv-size` cap.
    ///
    /// All Turbo modes carry per-token rotation state in their sidecars
    /// (`turbo_params` / `turbo3_params` / `v_packed` / `v_norms` /
    /// `cold_offset`) that `KVCache::trim_front` cannot safely truncate
    /// from the head. H3: `Turbo3Asym` belongs in this set
    /// the 3-bit V sidecars (`v_packed` with 24-bit groups + `v_norms`)
    /// have the same per-token contract as `Turbo4*`. Omitting it from
    /// this match silently allowed `--max-kv-size` + `fp16+turbo3` to ship
    /// without the operator-facing warning that the cap will not be
    /// honoured on the V side.
    ///
    /// Used by: scheduler dispatch for sequence allocation, the
    /// `--max-kv-size` + Turbo combination warning.
    #[inline]
    pub(super) fn is_turbo_mode(mode: KVCacheMode) -> bool {
        matches!(
            mode,
            KVCacheMode::Turbo4Asym
                | KVCacheMode::Turbo4
                | KVCacheMode::Turbo4Delegated
                | KVCacheMode::Turbo3Asym
        )
    }

    pub(super) fn sync_sequence_storage(&mut self, seq_id: SequenceId) {
        if let Err(err) = self
            .model
            .sync_sequence_storage(seq_id, &mut self.cache_pool)
        {
            tracing::warn!("Failed to sync paged state for {seq_id}: {err}");
        }
    }

    // ------------------------------------------------------------------
    // Request ingestion
    // ------------------------------------------------------------------
}
