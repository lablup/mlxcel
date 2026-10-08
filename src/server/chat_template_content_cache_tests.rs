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

use std::cell::Cell;

use super::*;

fn typed_only(context: Value, calls: &Cell<usize>) -> Result<String> {
    calls.set(calls.get() + 1);
    let mut output = String::new();
    for message in context.get_attr("messages")?.try_iter()? {
        let content = message.get_attr("content")?;
        if content.as_str().is_none() {
            for part in content.try_iter()? {
                if let Some(text) = part.get_attr("text")?.as_str() {
                    output.push_str(text);
                }
            }
        }
    }
    Ok(output)
}

#[test]
fn cloned_cache_reuses_decisions_without_retaining_messages_or_metadata() {
    let normalizer = ContentNormalizer::default();
    let context = minijinja::context! { messages => Value::UNDEFINED };
    let calls = Cell::new(0);
    let first = json!([{"role": "user", "content": "first"}]);
    assert!(
        normalizer.normalize(&first, &context, |ctx| typed_only(ctx, &calls))[0]["content"]
            .is_array()
    );
    assert_eq!(calls.get(), 4);
    let second = json!([{"role": "user", "content": "different private text"}]);
    assert!(
        normalizer
            .clone()
            .normalize(&second, &context, |ctx| typed_only(ctx, &calls))[0]["content"]
            .is_array()
    );
    assert_eq!(calls.get(), 4, "a warm clone must run no probes");

    let metadata = json!([{
        "role": "user", "content": "question", "reasoning_content": "private metadata"
    }]);
    for expected in [12, 20] {
        assert!(
            normalizer.normalize(&metadata, &context, |ctx| typed_only(ctx, &calls))[0]["content"]
                .is_array()
        );
        assert_eq!(calls.get(), expected, "metadata is checked without caching");
    }
    let cache = normalizer.cache.lock().unwrap();
    assert_eq!(cache.len(), 1);
    let key = String::from_utf8_lossy(&cache[0].0);
    assert!(!key.contains("different private text"));
    assert!(!key.contains("private metadata"));
}

#[test]
fn poisoned_cache_is_bypassed_without_changing_normalization() {
    let normalizer = ContentNormalizer::default();
    let cache = normalizer.cache.clone();
    let worker = std::thread::spawn(move || {
        let _guard = cache.lock().unwrap();
        panic!("deliberately poison the decision cache");
    });
    assert!(worker.join().is_err());
    assert!(normalizer.cache.is_poisoned());

    let messages = json!([{"role": "user", "content": "still normalize"}]);
    let calls = Cell::new(0);
    let normalized = normalizer.normalize(
        &messages,
        &minijinja::context! { messages => Value::UNDEFINED },
        |ctx| typed_only(ctx, &calls),
    );
    assert_eq!(normalized[0]["content"][0]["text"], "still normalize");
    assert_eq!(calls.get(), 4);
}

#[test]
fn cache_bounds_entries_and_key_allocations_but_still_checks_large_contexts() {
    let normalizer = ContentNormalizer::default();
    let messages = json!([{"role": "user", "content": "bounded"}]);
    let calls = Cell::new(0);
    for mode in 0..CACHE_ENTRIES + 4 {
        let context = minijinja::context! { messages => Value::UNDEFINED, mode };
        assert!(
            normalizer.normalize(&messages, &context, |ctx| typed_only(ctx, &calls))[0]["content"]
                .is_array()
        );
    }
    let cache = normalizer.cache.lock().unwrap();
    assert_eq!(cache.len(), CACHE_ENTRIES);
    assert!(
        cache
            .iter()
            .all(|(key, _)| key.len() <= MAX_CACHE_KEY_BYTES)
    );
    drop(cache);

    let large = minijinja::context! {
        messages => Value::UNDEFINED,
        hint => "x".repeat(MAX_CACHE_KEY_BYTES + 1)
    };
    let roles = string_roles(&messages).unwrap();
    assert!(cache_key(&large, &roles).is_none());
    let before = calls.get();
    for _ in 0..2 {
        assert!(
            normalizer.normalize(&messages, &large, |ctx| typed_only(ctx, &calls))[0]["content"]
                .is_array()
        );
    }
    assert_eq!(calls.get(), before + 8, "oversized contexts run uncached");
    assert_eq!(normalizer.cache.lock().unwrap().len(), CACHE_ENTRIES);
    let mut writer = KeyWriter(Vec::new());
    assert!(writer.write(&[0; MAX_CACHE_KEY_BYTES + 1]).is_err());
    assert!(
        writer.0.is_empty(),
        "oversized write must not allocate its payload"
    );
}
