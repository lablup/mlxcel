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

//! `--embedding` / `--rerank` / `--pooling` / `--embd-normalize` parsing (#1452).

use super::*;

fn args() -> EmbeddingCompatArgs {
    EmbeddingCompatArgs::default()
}

fn resolve(a: EmbeddingCompatArgs) -> EmbeddingCompatResolution {
    a.resolve().expect("resolves")
}

fn err(a: EmbeddingCompatArgs) -> String {
    a.resolve().expect_err("must be refused")
}

/// Run `body` with the two mode variables forced to a known state, so an
/// inherited `LLAMA_ARG_EMBEDDINGS` in the developer's shell cannot flip a
/// test that is about the flags.
fn with_env<T>(pairs: &[(&str, Option<&str>)], body: impl FnOnce() -> T) -> T {
    let _guard = crate::test_support::env_lock::env_lock();
    let keys = ["LLAMA_ARG_EMBEDDINGS", "LLAMA_ARG_RERANKING"];
    let saved: Vec<(String, Option<String>)> = keys
        .iter()
        .chain(pairs.iter().map(|(k, _)| k))
        .map(|k| ((*k).to_owned(), std::env::var(k).ok()))
        .collect();
    unsafe {
        for key in keys {
            std::env::remove_var(key);
        }
        for (key, value) in pairs {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
    let out = body();
    unsafe {
        for (key, value) in &saved {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
    out
}

#[test]
fn nothing_supplied_resolves_to_nothing_requested() {
    let resolved = with_env(&[], || resolve(args()));
    assert_eq!(resolved, EmbeddingCompatResolution::default());
    assert!(!resolved.embedding_only);
    assert!(!resolved.rerank_only);
    assert_eq!(resolved.pooling, None);
    assert_eq!(resolved.embd_normalize, None);
}

#[test]
fn the_embedding_flag_restricts_the_server() {
    let resolved = with_env(&[], || {
        resolve(EmbeddingCompatArgs {
            embedding: true,
            ..args()
        })
    });
    assert!(resolved.embedding_only);
    assert!(!resolved.rerank_only);
}

#[test]
fn the_rerank_flag_implies_the_embedding_restriction_as_upstream_does() {
    // b10621's `--reranking` handler sets `params.embedding = true` as well as
    // the rank pooling type, so generation is off in both modes.
    let resolved = with_env(&[], || {
        resolve(EmbeddingCompatArgs {
            rerank: true,
            ..args()
        })
    });
    assert!(resolved.rerank_only);
    assert!(resolved.embedding_only);
}

#[test]
fn asking_for_both_modes_is_refused() {
    let message = with_env(&[], || {
        err(EmbeddingCompatArgs {
            embedding: true,
            rerank: true,
            ..args()
        })
    });
    assert!(
        message.contains("--embeddings and --reranking"),
        "{message}"
    );
    assert!(message.contains("--embedding-model"), "{message}");
}

#[test]
fn the_three_poolable_values_map_onto_mlxcel_kernels() {
    for (raw, expected) in [
        ("mean", PoolingMode::Mean),
        ("cls", PoolingMode::Cls),
        ("last", PoolingMode::LastToken),
        ("MEAN", PoolingMode::Mean),
        (" cls ", PoolingMode::Cls),
    ] {
        let resolved = with_env(&[], || {
            resolve(EmbeddingCompatArgs {
                pooling: Some(raw.to_string()),
                ..args()
            })
        });
        assert_eq!(resolved.pooling, Some(expected), "--pooling {raw}");
        assert!(!resolved.rerank_only, "--pooling {raw}");
    }
}

#[test]
fn pooling_rank_selects_reranking_rather_than_a_kernel() {
    let resolved = with_env(&[], || {
        resolve(EmbeddingCompatArgs {
            pooling: Some("rank".to_string()),
            ..args()
        })
    });
    assert_eq!(
        resolved.pooling, None,
        "rank is not a pooling kernel; it must not become one"
    );
    assert!(resolved.rerank_only);
    assert!(resolved.embedding_only);
}

#[test]
fn pooling_none_is_refused_with_the_reason() {
    let message = with_env(&[], || {
        err(EmbeddingCompatArgs {
            pooling: Some("none".to_string()),
            ..args()
        })
    });
    assert!(message.contains("--pooling none"), "{message}");
    assert!(
        message.contains("one embedding vector per token"),
        "{message}"
    );
    assert!(message.contains("mean, cls or last"), "{message}");
}

#[test]
fn an_unknown_pooling_value_names_the_domain() {
    let message = with_env(&[], || {
        err(EmbeddingCompatArgs {
            pooling: Some("max".to_string()),
            ..args()
        })
    });
    assert!(
        message.contains("none, mean, cls, last or rank"),
        "mlxcel's own `max` is not a b10621 spelling: {message}"
    );
}

#[test]
fn the_whole_embd_normalize_domain_is_accepted() {
    for value in [-1, 0, 1, 2, 3, 9] {
        let resolved = with_env(&[], || {
            resolve(EmbeddingCompatArgs {
                embd_normalize: Some(value),
                ..args()
            })
        });
        assert_eq!(
            resolved.embd_normalize.map(EmbdNormalize::value),
            Some(value)
        );
    }
    let message = with_env(&[], || {
        err(EmbeddingCompatArgs {
            embd_normalize: Some(-2),
            ..args()
        })
    });
    assert!(message.contains("--embd-normalize -2"), "{message}");
}

#[test]
fn the_mode_variables_follow_b10621s_truthy_set() {
    // b10621 fires a value-less option from the environment only for
    // on/enabled/true/1, so `0` and an empty value leave the flag alone.
    for (value, expected) in [
        ("1", true),
        ("true", true),
        ("on", true),
        ("enabled", true),
        ("0", false),
        ("false", false),
        ("", false),
        ("yes", false),
    ] {
        let resolved = with_env(&[("LLAMA_ARG_EMBEDDINGS", Some(value))], || resolve(args()));
        assert_eq!(
            resolved.embedding_only, expected,
            "LLAMA_ARG_EMBEDDINGS={value:?}"
        );
        let resolved = with_env(&[("LLAMA_ARG_RERANKING", Some(value))], || resolve(args()));
        assert_eq!(
            resolved.rerank_only, expected,
            "LLAMA_ARG_RERANKING={value:?}"
        );
        assert_eq!(resolved.embedding_only, expected);
    }
}

#[test]
fn the_environment_alone_resolves_the_group() {
    let resolved = with_env(
        &[
            ("LLAMA_ARG_EMBEDDINGS", Some("1")),
            ("LLAMA_ARG_POOLING", Some("cls")),
        ],
        || from_env().expect("resolves"),
    );
    assert!(resolved.embedding_only);
    assert_eq!(resolved.pooling, Some(PoolingMode::Cls));
}

#[test]
fn a_bad_pooling_value_in_the_environment_is_refused_the_same_way() {
    let message = with_env(&[("LLAMA_ARG_POOLING", Some("none"))], || {
        from_env().expect_err("must be refused")
    });
    assert!(message.contains("--pooling none"), "{message}");
}
