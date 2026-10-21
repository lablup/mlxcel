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

//! Unit coverage for the bounded-store helpers (#1662).
//!
//! `serialized_json_len_saturating` feeds the byte budget and `LruKey`'s
//! `Ord` is the eviction order, so both are pinned directly rather than
//! only through the response and conversation stores.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::Serialize;
use serde::ser::{Error as _, Serializer};
use serde_json::json;

use crate::server::store_budget::{LruKey, serialized_json_len_saturating};

fn key(base: Instant, offset_ms: u64, sequence: u64, id: &str) -> LruKey {
    LruKey {
        last_accessed: base + Duration::from_millis(offset_ms),
        sequence,
        id: id.to_string(),
    }
}

/// A value whose `Serialize` impl always fails.
struct AlwaysFails;

impl Serialize for AlwaysFails {
    fn serialize<S: Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
        Err(S::Error::custom("refused"))
    }
}

#[test]
fn json_len_matches_serialized_bytes_for_representative_values() {
    let values = [
        json!(null),
        json!(true),
        json!(-12345),
        json!(1.5),
        json!("plain"),
        json!("quote \" backslash \\ newline \n tab \t"),
        json!("non-ascii: h\u{e9}llo \u{d55c}\u{ae00} \u{1f600}"),
        json!([]),
        json!({}),
        json!({"a": [1, 2, {"b": null}], "c": {"d": "e", "f": [true, false]}}),
    ];
    for value in &values {
        let expected = serde_json::to_vec(value).unwrap().len();
        assert_eq!(serialized_json_len_saturating(value), expected, "{value}");
    }
}

#[test]
fn json_len_accepts_unsized_values() {
    let text: &str = "str slice \u{d55c}";
    assert_eq!(
        serialized_json_len_saturating(text),
        serde_json::to_vec(text).unwrap().len()
    );
}

#[test]
fn json_len_saturates_when_a_custom_serialize_errors() {
    assert_eq!(serialized_json_len_saturating(&AlwaysFails), usize::MAX);
}

#[test]
fn json_len_saturates_on_non_string_map_keys() {
    let mut map = HashMap::new();
    map.insert((1_u8, 2_u8), "v");
    assert!(serde_json::to_vec(&map).is_err());
    assert_eq!(serialized_json_len_saturating(&map), usize::MAX);
}

#[test]
fn lru_key_orders_by_last_accessed_first() {
    let base = Instant::now();
    // The older access wins even though its sequence and id sort later.
    let older = key(base, 0, 99, "z");
    let newer = key(base, 5, 0, "a");
    assert!(older < newer);
    assert!(newer > older);
}

#[test]
fn lru_key_breaks_last_accessed_ties_by_sequence() {
    let base = Instant::now();
    let early = key(base, 5, 1, "z");
    let late = key(base, 5, 2, "a");
    assert!(early < late);
}

#[test]
fn lru_key_breaks_sequence_ties_by_id() {
    let base = Instant::now();
    let a = key(base, 5, 3, "a");
    let b = key(base, 5, 3, "b");
    assert!(a < b);
}

#[test]
fn lru_key_sort_yields_eviction_order() {
    let base = Instant::now();
    let mut keys = [
        key(base, 10, 1, "a"),
        key(base, 5, 2, "b"),
        key(base, 5, 1, "d"),
        key(base, 5, 1, "c"),
        key(base, 0, 9, "z"),
    ];
    keys.sort();
    let ids: Vec<&str> = keys.iter().map(|k| k.id.as_str()).collect();
    assert_eq!(ids, ["z", "c", "d", "b", "a"]);
}

#[test]
fn lru_key_partial_cmp_agrees_with_cmp_and_equal_keys_are_equal() {
    let base = Instant::now();
    let keys = [
        key(base, 0, 0, "a"),
        key(base, 0, 0, "b"),
        key(base, 0, 1, "a"),
        key(base, 1, 0, "a"),
    ];
    for a in &keys {
        for b in &keys {
            assert_eq!(a.partial_cmp(b), Some(a.cmp(b)));
        }
    }
    let x = key(base, 3, 4, "same");
    let y = key(base, 3, 4, "same");
    assert_eq!(x.cmp(&y), std::cmp::Ordering::Equal);
    assert_eq!(x, y);
}
