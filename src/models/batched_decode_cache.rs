// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Context-rich cache validation for native model batched decode.

use mlxcel_core::cache::BatchedAttentionMetadata;
use mlxcel_core::layers::KVCache;

/// Reborrow one cache layer from every batch row with actionable diagnostics.
pub(crate) fn collect_layer_caches<'a>(
    batch_caches: &'a mut [&mut [KVCache]],
    family: &str,
    layer_idx: usize,
) -> Vec<&'a mut KVCache> {
    batch_caches
        .iter_mut()
        .enumerate()
        .map(|(row_idx, caches)| {
            let available = caches.len();
            assert!(
                layer_idx < available,
                "{family} batched decode cache mismatch at row {row_idx}, layer {layer_idx}: row has {available} cache layers"
            );
            &mut caches[layer_idx]
        })
        .collect()
}

/// Build uniform attention metadata while preserving model and layer context.
pub(crate) fn uniform_layer_metadata(
    caches: &[&mut KVCache],
    query_len: i32,
    window_size: i32,
    family: &str,
    layer_idx: usize,
) -> BatchedAttentionMetadata {
    BatchedAttentionMetadata::uniform_kv_caches(caches, query_len, window_size).unwrap_or_else(
        |error| panic!("{family} batched decode metadata mismatch at layer {layer_idx}: {error}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    fn panic_message(payload: Box<dyn Any + Send>) -> String {
        if let Some(message) = payload.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = payload.downcast_ref::<&str>() {
            (*message).to_string()
        } else {
            "non-string panic".to_string()
        }
    }

    #[test]
    fn missing_row_cache_names_family_row_and_layer() {
        let mut complete = vec![KVCache::new(), KVCache::new()];
        let mut short = vec![KVCache::new()];
        let mut rows = [complete.as_mut_slice(), short.as_mut_slice()];

        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _ = collect_layer_caches(&mut rows, "Granite", 1);
        }))
        .expect_err("a short cache row must fail before attention");
        let message = panic_message(panic);
        assert!(message.contains("Granite"), "missing family: {message}");
        assert!(message.contains("row 1"), "missing row: {message}");
        assert!(message.contains("layer 1"), "missing layer: {message}");
        assert!(
            message.contains("1 cache layers"),
            "missing available cache count: {message}"
        );
    }

    #[test]
    fn invalid_metadata_names_family_and_layer() {
        let mut cache = KVCache::new();
        let caches = vec![&mut cache];

        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _ = uniform_layer_metadata(&caches, -1, 0, "Phi3", 7);
        }))
        .expect_err("negative query length must fail metadata construction");
        let message = panic_message(panic);
        assert!(message.contains("Phi3"), "missing family: {message}");
        assert!(message.contains("layer 7"), "missing layer: {message}");
        assert!(
            message.contains("metadata mismatch"),
            "missing context: {message}"
        );
        assert!(
            message.contains("query length must be non-negative"),
            "missing root cause: {message}"
        );
    }
}
