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

//! Vision feature cache for multi-turn VLM conversations.
//!
//! Stores post-projection image features keyed by image identity so the server
//! can skip the vision tower + multimodal embedder on subsequent turns that
//! reference the same image. Mirrors the upstream `VisionFeatureCache` that
//! landed in mlx-vlm..
//!
//! The cache is generic over a [`CloneableFeatures`] implementor so each VLM
//! family can store the feature shape it actually needs (a single projected
//! array for Gemma 4 / Qwen2.5-VL, or hidden states + DeepStack features for
//! Qwen3-VL). Features are deep-copied on both insertion and retrieval so the
//! cached MLX arrays are never aliased into the active computation graph.
//!
//! Keys are `PathBuf` (cheap, preferred when the image arrives as a filesystem
//! path) or a 32-byte SHA-256 digest of the pixel tensor's raw bytes (required
//! when the request carries the image inline). The digest is computed once per
//! cache-miss prefill. See [`image_hash_from_bytes`] and [`image_hash_from_pixels`].

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use mlxcel_core::{MlxArray, UniquePtr};
use sha2::{Digest, Sha256};

/// Default cache capacity when the CLI flag is not provided.
///
/// Matches the upstream mlx-vlm default. A 20-image cache is enough to cover
/// typical multi-turn conversations that revisit a handful of images.
pub const DEFAULT_VISION_CACHE_SIZE: usize = 20;

/// Identity of a cached image.
///
/// When the image arrives as a filesystem path we use the path directly — this
/// is both cheaper than hashing and stable across repeated references. When the
/// image arrives inline (base64, raw bytes, or an already-decoded pixel
/// tensor), we fall back to a 32-byte SHA-256 digest of the bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CacheKey {
    /// Filesystem path. Preferred when available because it avoids hashing the
    /// pixel buffer on every request.
    Path(PathBuf),
    /// SHA-256 digest of the pixel tensor's raw bytes (or the original encoded
    /// image bytes before decoding, depending on the call site).
    Hash([u8; 32]),
}

impl CacheKey {
    /// Construct a `Path` key.
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self::Path(path.into())
    }

    /// Construct a `Hash` key from a raw 32-byte digest.
    pub fn from_hash(digest: [u8; 32]) -> Self {
        Self::Hash(digest)
    }
}

impl From<&Path> for CacheKey {
    fn from(path: &Path) -> Self {
        Self::Path(path.to_path_buf())
    }
}

/// SHA-256 digest of a raw byte stream (encoded image payload).
///
/// Call this with the bytes you received over the wire. Hashing the encoded
/// bytes rather than the decoded pixel tensor is both ~100x cheaper and
/// sufficient for cache identity — two requests that carry bit-identical PNG
/// payloads will hash to the same key.
pub fn image_hash_from_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let out = hasher.finalize();
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&out);
    digest
}

/// SHA-256 digest of an encoded image payload, domain-separated by a
/// preprocessing soft-token budget.
///
/// The cached value is the *vision tower output*, which for Gemma 4 depends on
/// both the image bytes and the soft-token budget the preprocessor resized
/// under: the same PNG at budget 70 and budget 1120 yields different patch
/// grids and a different number of feature rows. Keying on bytes alone would
/// let the first request's features be served to the second, desyncing the
/// features from the prompt's placeholder expansion.
///
/// The image payload is length-prefixed and the budget field is tagged, so an
/// attacker cannot craft an image whose trailing bytes reproduce the budget
/// field of a different request and collide the two keys. The `None` budget
/// (no per-request override, the common case) is a distinct sentinel from any
/// concrete budget. `ModelVisionCaches` is a per-process in-memory cache with
/// no on-disk form, so this framing costs nothing: the cache is empty at
/// startup regardless of the key layout.
pub fn image_hash_from_bytes_with_soft_tokens(
    bytes: &[u8],
    max_soft_tokens: Option<usize>,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"mlxcel:image-soft-tokens:v2");
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    match max_soft_tokens {
        Some(budget) => {
            hasher.update([1u8]);
            hasher.update((budget as u64).to_le_bytes());
        }
        None => hasher.update([0u8]),
    }
    let out = hasher.finalize();
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&out);
    digest
}

/// SHA-256 digest of an MLX pixel tensor's raw bytes.
///
/// Used when the image already lives on the MLX side (e.g. the request handler
/// decoded PNG → pixel tensor before prefill). The cost is proportional to the
/// pixel buffer size — roughly ~8 ms for a 896×896×3 f32 tensor — which is
/// still ~25x cheaper than the 230 ms vision tower forward pass on Gemma 4, so
/// a cache hit is always a net win.
///
/// This variant needs no budget domain-separation: the pixel tensor it hashes
/// is the *already resized* one, so a different budget produces different
/// bytes and therefore a different digest on its own.
pub fn image_hash_from_pixels(array: &MlxArray) -> [u8; 32] {
    // Fully materialize before extracting bytes; without eval() the contents
    // may reference lazy graph nodes that have not been realized yet.
    mlxcel_core::eval(array);
    let bytes = mlxcel_core::array_to_raw_bytes(array);
    image_hash_from_bytes(&bytes)
}

/// Features stored in the cache must be deep-cloneable so cache reads never
/// alias live MLX arrays into the active computation graph.
///
/// The blanket rules for implementors:
/// 1. `deep_clone` must allocate a new MLX array for every field; the returned
///    value must be safe to mutate or drop without touching `self`.
/// 2. Implementors should stay cheap to construct; the cache calls `deep_clone`
///    on both `put` (to freeze the entry against later modification) and `get`
///    (to hand out an owned copy to the caller).
pub trait CloneableFeatures {
    /// Deep-copy every MLX array the entry owns.
    fn deep_clone(&self) -> Self;
}

/// Single-tensor vision features.
///
/// Used by: Gemma 4 VLM, Qwen2.5-VL.
pub struct SingleArrayFeatures {
    /// Post-projection image features ready to be merged at image token
    /// positions by the VLM.
    pub features: UniquePtr<MlxArray>,
}

impl SingleArrayFeatures {
    pub fn new(features: UniquePtr<MlxArray>) -> Self {
        Self { features }
    }
}

impl CloneableFeatures for SingleArrayFeatures {
    fn deep_clone(&self) -> Self {
        let src = self
            .features
            .as_ref()
            .expect("cached single-array features must not be null");
        Self {
            features: mlxcel_core::copy(src),
        }
    }
}

/// Vision features plus DeepStack side-branch outputs.
///
/// Used by: Qwen3-VL (DeepStack emits per-layer features that the language
/// model injects at selected transformer layers).
pub struct DeepStackFeatures {
    /// Post-projection hidden states, ready to merge at image token positions.
    pub hidden_states: UniquePtr<MlxArray>,
    /// One feature tensor per DeepStack injection layer.
    pub deepstack: Vec<UniquePtr<MlxArray>>,
}

impl DeepStackFeatures {
    pub fn new(hidden_states: UniquePtr<MlxArray>, deepstack: Vec<UniquePtr<MlxArray>>) -> Self {
        Self {
            hidden_states,
            deepstack,
        }
    }
}

impl CloneableFeatures for DeepStackFeatures {
    fn deep_clone(&self) -> Self {
        let hs = self
            .hidden_states
            .as_ref()
            .expect("cached DeepStack hidden_states must not be null");
        let deepstack = self
            .deepstack
            .iter()
            .map(|arr| {
                let src = arr
                    .as_ref()
                    .expect("cached DeepStack side-branch tensor must not be null");
                mlxcel_core::copy(src)
            })
            .collect();
        Self {
            hidden_states: mlxcel_core::copy(hs),
            deepstack,
        }
    }
}

/// LRU-ordered per-image vision feature cache.
///
/// The implementation is intentionally simple: a `VecDeque` tracks insertion /
/// access order, a parallel `Vec` of entries holds the keys and values. This
/// keeps the crate free of additional map-ordering dependencies while staying
/// O(max_size) per operation — for the default capacity of 20 the constant
/// factor is negligible compared to even a single vision tower forward pass.
///
/// Eviction happens at the front of the deque (least recently used); cache
/// hits bump the entry to the back.
pub struct VisionFeatureCache<V: CloneableFeatures> {
    entries: VecDeque<(CacheKey, V)>,
    max_size: usize,
}

impl<V: CloneableFeatures> VisionFeatureCache<V> {
    /// Construct a cache that holds up to `max_size` entries.
    ///
    /// `max_size == 0` disables caching entirely — `get` always returns `None`
    /// and `put` is a no-op. This matches the semantics expected by the
    /// `--vision-cache-size 0` CLI setting.
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_size.max(1)),
            max_size,
        }
    }

    /// Maximum number of entries the cache can hold.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Number of entries currently resident.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is disabled (`max_size == 0`) or currently empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a cached entry.
    ///
    /// On hit the entry is bumped to the back of the LRU queue and a deep copy
    /// of the cached features is returned. On miss or when the cache is
    /// disabled, returns `None`.
    pub fn get(&mut self, key: &CacheKey) -> Option<V> {
        if self.max_size == 0 {
            return None;
        }
        let position = self.entries.iter().position(|(k, _)| k == key)?;
        // Remove-then-push keeps LRU order without reshuffling the middle.
        let entry = self.entries.remove(position)?;
        let clone = entry.1.deep_clone();
        self.entries.push_back(entry);
        Some(clone)
    }

    /// Insert a new entry.
    ///
    /// If `key` is already present the existing entry is replaced and moved to
    /// the back of the LRU queue. When inserting past `max_size`, the least
    /// recently used entry is evicted. A `max_size == 0` cache discards the
    /// insertion.
    ///
    /// The stored value is a deep copy of `features` so the caller may
    /// continue to use the original without risking cache aliasing.
    pub fn put(&mut self, key: CacheKey, features: &V) {
        if self.max_size == 0 {
            return;
        }
        if let Some(position) = self.entries.iter().position(|(k, _)| k == &key) {
            self.entries.remove(position);
        }
        if self.entries.len() >= self.max_size {
            self.entries.pop_front();
        }
        self.entries.push_back((key, features.deep_clone()));
    }

    /// Drop every cached entry.
    ///
    /// Called on model unload so the `UniquePtr<MlxArray>` references release
    /// their GPU-side memory before the new model begins loading.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl<V: CloneableFeatures> Default for VisionFeatureCache<V> {
    /// The default cache uses [`DEFAULT_VISION_CACHE_SIZE`].
    fn default() -> Self {
        Self::new(DEFAULT_VISION_CACHE_SIZE)
    }
}

/// Bundle of per-feature-shape caches owned by a single model instance.
///
/// Different VLM families emit different feature shapes: most VLMs produce a
/// single post-projection tensor ([`SingleArrayFeatures`]), while Qwen3-VL
/// additionally emits per-layer DeepStack side-branch features
/// ([`DeepStackFeatures`]). The server creates one [`ModelVisionCaches`] per
/// loaded model so both families can be served without allocating caches for
/// shapes the model will never use.
///
/// Cache lifetime is tied to the model registry entry; `clear_all` must be
/// called on model unload so GPU-side memory is released before a new model
/// begins loading.
pub struct ModelVisionCaches {
    /// Shared vision-feature cache for single-tensor families (Gemma 4 VLM,
    /// Qwen2.5-VL). Wrapped in a `Mutex` so concurrent request tasks can share
    /// safely; contention on this lock is limited to the prefill phase which
    /// is already single-threaded per model worker.
    pub single: std::sync::Mutex<VisionFeatureCache<SingleArrayFeatures>>,
    /// Shared vision-feature cache for Qwen3-VL's DeepStack output shape.
    pub deepstack: std::sync::Mutex<VisionFeatureCache<DeepStackFeatures>>,
}

impl ModelVisionCaches {
    /// Construct a bundle where every cache shares the same `max_size`.
    ///
    /// Passing `max_size == 0` disables caching for this model instance.
    pub fn new(max_size: usize) -> Self {
        Self {
            single: std::sync::Mutex::new(VisionFeatureCache::new(max_size)),
            deepstack: std::sync::Mutex::new(VisionFeatureCache::new(max_size)),
        }
    }

    /// Drop every cached entry across every shape-specific cache.
    ///
    /// Intended for model unload. Errors from a poisoned mutex are silently
    /// swallowed — the mutex will be dropped along with the bundle shortly
    /// after this call.
    pub fn clear_all(&self) {
        if let Ok(mut c) = self.single.lock() {
            c.clear();
        }
        if let Ok(mut c) = self.deepstack.lock() {
            c.clear();
        }
    }

    /// Whether caching is enabled for this bundle (both caches share
    /// `max_size`, so checking one suffices).
    pub fn enabled(&self) -> bool {
        self.single
            .lock()
            .map(|c| c.max_size() > 0)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple test fixture: wraps an integer so we can track deep_clone calls
    /// without depending on MLX arrays in the unit test path.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct DummyFeatures {
        value: u32,
    }

    impl CloneableFeatures for DummyFeatures {
        fn deep_clone(&self) -> Self {
            self.clone()
        }
    }

    fn path_key(name: &str) -> CacheKey {
        CacheKey::from_path(name)
    }

    #[test]
    fn get_miss_returns_none() {
        let mut cache: VisionFeatureCache<DummyFeatures> = VisionFeatureCache::new(4);
        assert!(cache.get(&path_key("a")).is_none());
    }

    #[test]
    fn put_then_get_hits() {
        let mut cache = VisionFeatureCache::new(4);
        cache.put(path_key("a"), &DummyFeatures { value: 1 });
        let hit = cache.get(&path_key("a")).expect("expected cache hit");
        assert_eq!(hit.value, 1);
    }

    #[test]
    fn lru_eviction_removes_least_recently_used() {
        let mut cache = VisionFeatureCache::new(3);
        for i in 0..3 {
            cache.put(
                path_key(&format!("img_{i}")),
                &DummyFeatures { value: i as u32 },
            );
        }
        // Inserting a 4th entry evicts the oldest ("img_0").
        cache.put(path_key("img_3"), &DummyFeatures { value: 3 });

        assert!(cache.get(&path_key("img_0")).is_none());
        assert!(cache.get(&path_key("img_1")).is_some());
        assert!(cache.get(&path_key("img_2")).is_some());
        assert!(cache.get(&path_key("img_3")).is_some());
    }

    #[test]
    fn hit_bumps_position_to_most_recent() {
        let mut cache = VisionFeatureCache::new(3);
        cache.put(path_key("a"), &DummyFeatures { value: 1 });
        cache.put(path_key("b"), &DummyFeatures { value: 2 });
        cache.put(path_key("c"), &DummyFeatures { value: 3 });

        // Touching "a" should make "b" the oldest entry.
        assert!(cache.get(&path_key("a")).is_some());

        // Inserting "d" must now evict "b", not "a".
        cache.put(path_key("d"), &DummyFeatures { value: 4 });

        assert!(cache.get(&path_key("b")).is_none());
        assert!(cache.get(&path_key("a")).is_some());
        assert!(cache.get(&path_key("c")).is_some());
        assert!(cache.get(&path_key("d")).is_some());
    }

    #[test]
    fn put_replaces_existing_entry() {
        let mut cache = VisionFeatureCache::new(4);
        cache.put(path_key("a"), &DummyFeatures { value: 1 });
        cache.put(path_key("a"), &DummyFeatures { value: 99 });
        assert_eq!(cache.len(), 1);
        let hit = cache.get(&path_key("a")).expect("expected cache hit");
        assert_eq!(hit.value, 99);
    }

    #[test]
    fn clear_drops_all_entries() {
        let mut cache = VisionFeatureCache::new(4);
        cache.put(path_key("a"), &DummyFeatures { value: 1 });
        cache.put(path_key("b"), &DummyFeatures { value: 2 });
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert!(cache.is_empty());
        assert!(cache.get(&path_key("a")).is_none());
    }

    #[test]
    fn disabled_cache_returns_no_hits() {
        let mut cache: VisionFeatureCache<DummyFeatures> = VisionFeatureCache::new(0);
        cache.put(path_key("a"), &DummyFeatures { value: 1 });
        assert!(cache.get(&path_key("a")).is_none());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn hash_key_is_stable_for_identical_bytes() {
        let bytes = b"\x89PNG\r\n\x1a\n fake image payload";
        let a = image_hash_from_bytes(bytes);
        let b = image_hash_from_bytes(bytes);
        assert_eq!(a, b);

        let different = image_hash_from_bytes(b"different payload");
        assert_ne!(a, different);
    }

    #[test]
    fn hash_and_path_keys_do_not_collide() {
        let hash_key = CacheKey::from_hash([0u8; 32]);
        let path_key = CacheKey::from_path("");
        assert_ne!(hash_key, path_key);
    }

    #[test]
    fn soft_token_budget_partitions_the_cache_key() {
        let bytes = b"\x89PNG\r\n\x1a\n fake image payload";
        // No override, a concrete budget, and a different budget are all
        // distinct keys, so the same image at two budgets can never be served
        // the other budget's vision features.
        let none = image_hash_from_bytes_with_soft_tokens(bytes, None);
        let b70 = image_hash_from_bytes_with_soft_tokens(bytes, Some(70));
        let b1120 = image_hash_from_bytes_with_soft_tokens(bytes, Some(1120));
        assert_ne!(none, b70);
        assert_ne!(none, b1120);
        assert_ne!(b70, b1120);
        // The key is deterministic for a given (bytes, budget).
        assert_eq!(b70, image_hash_from_bytes_with_soft_tokens(bytes, Some(70)));
    }

    #[test]
    fn soft_token_budget_key_resists_byte_boundary_collision() {
        // The security concern: without length-prefixing, an attacker could
        // craft an image whose trailing bytes reproduce the budget field of a
        // different request. Construct exactly that: image A carries a payload
        // that ends in what could look like image B's budget suffix, sent with
        // no override; image B is the shorter prefix sent at that budget. The
        // two must not collide.
        let prefix = b"shared image prefix bytes";
        let mut crafted = prefix.to_vec();
        crafted.extend_from_slice(&[1u8]);
        crafted.extend_from_slice(&(70u64).to_le_bytes());

        let crafted_no_budget = image_hash_from_bytes_with_soft_tokens(&crafted, None);
        let prefix_at_70 = image_hash_from_bytes_with_soft_tokens(prefix, Some(70));
        assert_ne!(crafted_no_budget, prefix_at_70);
    }

    #[test]
    fn many_distinct_inserts_cap_at_max_size() {
        let mut cache = VisionFeatureCache::new(5);
        for i in 0..20 {
            cache.put(
                path_key(&format!("img_{i}")),
                &DummyFeatures { value: i as u32 },
            );
            assert!(cache.len() <= 5);
        }
        // Only the last 5 keys must survive.
        for i in 0..15 {
            assert!(cache.get(&path_key(&format!("img_{i}"))).is_none());
        }
        for i in 15..20 {
            assert!(cache.get(&path_key(&format!("img_{i}"))).is_some());
        }
    }
}
