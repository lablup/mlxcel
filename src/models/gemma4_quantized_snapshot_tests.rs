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

//! Gemma 4's per-layer cache arms under the quantized KV modes (issue #3250).
//!
//! Gemma 4 keeps a sliding ring on most layers, a full-attention cache on
//! every sixth and nothing on a KV-shared layer. `--cache-type-k int8
//! --cache-type-v int8` therefore leaves the rings holding plain rows behind
//! an `Int8` label and the full layers holding real int8 rows, and that pair
//! is what a snapshot has to carry. Each test compares against layers that
//! were filled cold, byte for byte.

use crate::models::gemma4::Cache;
use crate::models::kv_snapshot::test_support::{keys, values};
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::generate::ModelStateSnapshot;
use mlxcel_core::layers::{KVCache, RotatingKVCache};

const WINDOW: i32 = 64;

/// Sliding, full, sliding, and a KV-shared layer that never fills its cache.
fn layers(mode: KVCacheMode) -> Vec<Cache> {
    vec![
        Cache::Rotating(RotatingKVCache::new_with_mode(WINDOW, mode)),
        Cache::Standard(KVCache::new_with_mode(mode)),
        Cache::Rotating(RotatingKVCache::new_with_mode(WINDOW, mode)),
        Cache::Standard(KVCache::new_with_mode(mode)),
    ]
}

/// Feed tokens `[start, start + n)` to every layer but the KV-shared last
/// one, as one chunk, and return what each returned.
fn feed(layers: &mut [Cache], start: i32, n: i32) -> Vec<Vec<u8>> {
    let populated = layers.len() - 1;
    layers[..populated]
        .iter_mut()
        .flat_map(|layer| {
            let (k, v) = match layer {
                Cache::Standard(cache) => cache.update_and_fetch(keys(start, n), values(start, n)),
                Cache::Rotating(cache) => cache.update_and_fetch(keys(start, n), values(start, n)),
            };
            mlxcel_core::eval(&k);
            mlxcel_core::eval(&v);
            [
                mlxcel_core::array_to_raw_bytes(&k),
                mlxcel_core::array_to_raw_bytes(&v),
            ]
        })
        .collect()
}

fn snapshot_of(layers: &[Cache], tokens: i32) -> ModelStateSnapshot {
    let mut snapshot = ModelStateSnapshot::new("gemma4", tokens as usize);
    for (idx, layer) in layers.iter().enumerate() {
        layer
            .snapshot_into(&mut snapshot, &format!("layer{idx}"))
            .unwrap_or_else(|err| panic!("layer{idx} snapshot: {err}"));
    }
    snapshot
}

fn restored_from(mode: KVCacheMode, snapshot: &ModelStateSnapshot) -> Vec<Cache> {
    let mut restored = layers(mode);
    for (idx, layer) in restored.iter_mut().enumerate() {
        layer
            .restore_from(snapshot, &format!("layer{idx}"))
            .unwrap_or_else(|err| panic!("layer{idx} restore: {err}"));
    }
    restored
}

const MODES: [KVCacheMode; 3] = [
    KVCacheMode::Int8,
    KVCacheMode::Turbo4Asym,
    KVCacheMode::Turbo4,
];

/// A whole-state round trip, from before the ring wraps and from after.
#[test]
fn gemma4_layers_round_trip_under_a_quantized_table() {
    for mode in MODES {
        for tokens in [40, WINDOW + 30] {
            let mut source = layers(mode);
            let _ = feed(&mut source, 0, tokens);
            let snapshot = snapshot_of(&source, tokens);
            assert!(
                snapshot
                    .tensor_names()
                    .all(|name| !name.starts_with("layer3.")),
                "{mode:?}: the KV-shared layer stores nothing"
            );
            let mut restored = restored_from(mode, &snapshot);
            assert!(
                !restored[3].is_populated(),
                "{mode:?}: the KV-shared layer stays fresh"
            );
            for t in tokens..tokens + WINDOW + 3 {
                assert!(
                    feed(&mut restored, t, 1) == feed(&mut source, t, 1),
                    "{mode:?} restored at {tokens}: token {t} differs from the uninterrupted layers"
                );
            }
        }
    }
}

/// A truncated restore cuts every populated layer back, the full layer of a
/// symmetric `Turbo4` table included. That layer holds K packed and has no
/// `keys` buffer, so a populated check that looked at `keys` alone would skip
/// it and leave it covering the whole snapshot.
#[test]
fn a_truncated_gemma4_restore_cuts_every_populated_layer() {
    for mode in MODES {
        for (stored, target) in [(40, 25), (40, 32), (40, 1), (WINDOW, WINDOW - 1)] {
            let mut source = layers(mode);
            let _ = feed(&mut source, 0, stored);
            let snapshot = snapshot_of(&source, stored);
            let what = format!("{mode:?}, {stored} tokens cut to {target}");
            for idx in 0..4 {
                assert!(
                    Cache::snapshot_truncatable_to(&snapshot, &format!("layer{idx}"), target),
                    "{what}: layer{idx} truncates while its ring is unwrapped"
                );
            }

            let mut restored = restored_from(mode, &snapshot);
            for (idx, layer) in restored.iter_mut().enumerate() {
                assert_eq!(layer.is_populated(), idx != 3, "{what}: layer{idx}");
                layer
                    .truncate_to(target)
                    .unwrap_or_else(|err| panic!("{what}: layer{idx} truncate: {err}"));
                let offset = match layer {
                    Cache::Standard(cache) => cache.offset,
                    Cache::Rotating(cache) => cache.offset,
                };
                assert_eq!(
                    offset,
                    if idx == 3 { 0 } else { target },
                    "{what}: layer{idx} holds the target length"
                );
            }

            let mut cold = layers(mode);
            let _ = feed(&mut cold, 0, target);
            // The rest of the prompt as one chunk, then decode steps.
            assert!(
                feed(&mut restored, target, 7) == feed(&mut cold, target, 7),
                "{what}: the next prefill chunk differs from a cold fill of the prefix"
            );
            for t in target + 7..target + 12 {
                assert!(
                    feed(&mut restored, t, 1) == feed(&mut cold, t, 1),
                    "{what}: token {t} differs from a cold fill of the prefix"
                );
            }
        }
    }
}

/// Once a ring has wrapped, the state does not truncate, under any table.
#[test]
fn a_wrapped_gemma4_ring_is_not_truncatable_under_a_quantized_table() {
    for mode in MODES {
        let mut source = layers(mode);
        for t in 0..WINDOW + 30 {
            let _ = feed(&mut source, t, 1);
        }
        let snapshot = snapshot_of(&source, WINDOW + 30);
        assert!(
            !Cache::snapshot_truncatable_to(&snapshot, "layer0", WINDOW + 10),
            "{mode:?}: a wrapped ring"
        );
        assert!(
            Cache::snapshot_truncatable_to(&snapshot, "layer1", WINDOW + 10),
            "{mode:?}: the full layer still would, on its own"
        );
        let mut restored = restored_from(mode, &snapshot);
        assert!(
            restored[0].truncate_to(WINDOW + 10).is_err(),
            "{mode:?}: a wrapped ring cannot be trimmed"
        );
    }
}
