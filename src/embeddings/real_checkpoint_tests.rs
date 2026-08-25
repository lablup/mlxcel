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

//! Real-checkpoint gates for everything below the forward pass: detection,
//! pooling config, `max_length`, subfolder weight loading, batch
//! tokenization and the generation-loader rejection.
//!
//! Each test soft-skips when its checkpoint is absent, following the
//! convention of `tests/*_parity.rs`. Fetch with:
//!
//! ```sh
//! mlxcel download sentence-transformers/all-MiniLM-L6-v2
//! mlxcel download Qwen/Qwen3-Embedding-0.6B
//! ```

use std::path::PathBuf;

use mlxcel_core::weights::load_weights_from_dir_with_subfolders;

use super::limits::{EMBEDDING_MAX_LENGTH_CAP, derive_max_length, resolve_pad_token_id};
use super::loader::load_embedding_model;
use super::loader_tests::{err_string, temp_dir, write_f32_safetensors};
use super::pooling::{PoolingConfig, PoolingMode};
use super::tokenize::{EncodeOptions, encode_batch, strip_padding_and_truncation};
use crate::models::{ModelType, get_model_type};
use crate::tokenizer::load_tokenizer;

const MINILM: &str = "sentence-transformers/all-MiniLM-L6-v2";
const QWEN3_EMBEDDING: &str = "Qwen/Qwen3-Embedding-0.6B";

/// Locate a downloaded checkpoint: the mlxcel store, then the HuggingFace
/// cache, then `<repo>/models/<name>`. `None` skips the test.
fn local_checkpoint(repo_id: &str) -> Option<PathBuf> {
    let candidates = [
        crate::downloader::model_dir(repo_id),
        crate::downloader::hf_cache_snapshot(repo_id, None),
        Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("models")
                .join(crate::downloader::repo_basename(repo_id)),
        ),
    ];
    let found = candidates
        .into_iter()
        .flatten()
        .find(|dir| dir.join("config.json").is_file());
    if found.is_none() {
        eprintln!(
            "skipping real-checkpoint gate: {repo_id} not present (mlxcel download {repo_id})"
        );
    }
    found
}

#[test]
fn minilm_is_detected_as_bert_embedding_with_mean_pooling() {
    let Some(dir) = local_checkpoint(MINILM) else {
        return;
    };
    assert_eq!(get_model_type(&dir).unwrap(), ModelType::Bert);
    assert_eq!(PoolingConfig::read(&dir).unwrap(), Some(PoolingMode::Mean));
    // sentence_bert_config.json max_seq_length 256 beats tokenizer 512 and
    // max_position_embeddings 512.
    assert_eq!(derive_max_length(&dir, true, None), 256);
    assert_eq!(derive_max_length(&dir, true, Some(64)), 64);
}

#[test]
fn qwen3_embedding_is_detected_by_pooling_layout_with_lasttoken() {
    let Some(dir) = local_checkpoint(QWEN3_EMBEDDING) else {
        return;
    };
    // `model_type: qwen3` with `Qwen3ForCausalLM`: only the 1_Pooling /
    // modules.json layout routes it away from the causal generator.
    assert_eq!(get_model_type(&dir).unwrap(), ModelType::Qwen3Embedding);
    assert_eq!(
        PoolingConfig::read(&dir).unwrap(),
        Some(PoolingMode::LastToken)
    );
    // No sentence_bert_config.json; tokenizer model_max_length 131072 and
    // max_position_embeddings 32768 (rotary, so not consulted) both exceed
    // the hard cap.
    assert_eq!(
        derive_max_length(&dir, false, None),
        EMBEDDING_MAX_LENGTH_CAP
    );
}

#[test]
fn minilm_subfolder_loader_reads_top_level_shard_and_prefixes_dense_folder() {
    let Some(dir) = local_checkpoint(MINILM) else {
        return;
    };
    let weights = load_weights_from_dir_with_subfolders(&dir).unwrap();
    assert!(
        weights.contains_key("embeddings.word_embeddings.weight"),
        "top-level shard tensor missing; keys: {:?}",
        weights.keys().take(5).collect::<Vec<_>>()
    );
    assert_eq!(
        mlxcel_core::array_shape(&weights["embeddings.word_embeddings.weight"]),
        vec![30522, 384]
    );
    assert!(weights.keys().all(|k| !k.contains("2_Dense.")));

    // A synthetic 2_Dense module next to the real shard gets its prefix.
    let synthetic = temp_dir("minilm_dense");
    std::os::unix::fs::symlink(
        dir.join("model.safetensors"),
        synthetic.join("model.safetensors"),
    )
    .unwrap();
    std::fs::create_dir_all(synthetic.join("2_Dense")).unwrap();
    write_f32_safetensors(
        &synthetic.join("2_Dense").join("model.safetensors"),
        "linear.weight",
        &[1.0, 2.0, 3.0],
    );
    let weights = load_weights_from_dir_with_subfolders(&synthetic).unwrap();
    assert!(weights.contains_key("embeddings.word_embeddings.weight"));
    assert!(weights.contains_key("2_Dense.linear.weight"));
    std::fs::remove_dir_all(synthetic).unwrap();
}

#[test]
fn minilm_tokenizer_batch_is_right_padded_with_special_tokens() {
    let Some(dir) = local_checkpoint(MINILM) else {
        return;
    };
    let tokenizer = strip_padding_and_truncation(load_tokenizer(&dir).unwrap());
    let pad_id = resolve_pad_token_id(&dir, &tokenizer);
    assert_eq!(pad_id, 0, "[PAD] is id 0 in the BERT vocab");

    let opts = EncodeOptions {
        add_special_tokens: true,
        max_length: 256,
        with_token_type_ids: true,
    };
    let batch = encode_batch(
        &tokenizer,
        &["The weather is lovely today.", "Hi"],
        opts,
        pad_id,
        None,
    )
    .unwrap();
    assert_eq!(batch.batch, 2);
    let width = batch.width;
    let row0 = &batch.input_ids[..width];
    let row1 = &batch.input_ids[width..];
    let n0 = batch.token_counts[0];
    let n1 = batch.token_counts[1];
    assert!(n0 > n1, "the longer text has more tokens");
    assert_eq!(width, n0, "padded to the longest member, not to 128");
    assert_eq!(row0[0], 101, "[CLS]");
    assert_eq!(row0[n0 - 1], 102, "[SEP]");
    assert_eq!(row1[0], 101);
    assert_eq!(row1[n1 - 1], 102);
    assert!(
        row1[n1..].iter().all(|&id| id == pad_id as i32),
        "right padding"
    );
    let mask1 = &batch.attention_mask[width..];
    assert!(mask1[..n1].iter().all(|&m| m == 1));
    assert!(mask1[n1..].iter().all(|&m| m == 0));
    assert!(
        batch
            .token_type_ids
            .as_ref()
            .unwrap()
            .iter()
            .all(|&t| t == 0)
    );

    // Truncation keeps the trailing [SEP].
    let cut = encode_batch(
        &tokenizer,
        &["The weather is lovely today and the sky is blue."],
        EncodeOptions {
            max_length: 5,
            ..opts
        },
        pad_id,
        None,
    )
    .unwrap();
    assert_eq!(cut.token_counts, vec![5]);
    assert_eq!(cut.input_ids[0], 101);
    assert_eq!(cut.input_ids[4], 102);
}

#[test]
fn qwen3_embedding_tokenizer_batch_keeps_trailing_endoftext() {
    let Some(dir) = local_checkpoint(QWEN3_EMBEDDING) else {
        return;
    };
    let tokenizer = strip_padding_and_truncation(load_tokenizer(&dir).unwrap());
    let pad_id = resolve_pad_token_id(&dir, &tokenizer);
    const ENDOFTEXT: u32 = 151_643;
    assert_eq!(pad_id, ENDOFTEXT, "pad_token is <|endoftext|>");

    let opts = EncodeOptions {
        add_special_tokens: true,
        max_length: 8192,
        with_token_type_ids: false,
    };
    let batch = encode_batch(
        &tokenizer,
        &["Query: what is the capital of France?", "Paris"],
        opts,
        pad_id,
        None,
    )
    .unwrap();
    let width = batch.width;
    for (b, &count) in batch.token_counts.iter().enumerate() {
        let row = &batch.input_ids[b * width..(b + 1) * width];
        assert_eq!(
            row[count - 1],
            ENDOFTEXT as i32,
            "row {b} ends with <|endoftext|>"
        );
        assert!(row[count..].iter().all(|&id| id == pad_id as i32));
    }

    let cut = encode_batch(
        &tokenizer,
        &["Query: what is the capital of France? Tell me everything about it."],
        EncodeOptions {
            max_length: 4,
            ..opts
        },
        pad_id,
        None,
    )
    .unwrap();
    assert_eq!(cut.token_counts, vec![4]);
    assert_eq!(
        cut.input_ids[3], ENDOFTEXT as i32,
        "truncation keeps <|endoftext|>"
    );
}

#[test]
fn generation_loader_rejects_real_embedding_checkpoints() {
    for repo in [MINILM, QWEN3_EMBEDDING] {
        let Some(dir) = local_checkpoint(repo) else {
            continue;
        };
        let err = err_string(crate::load_model(&dir));
        assert!(err.contains("/v1/embeddings"), "{repo}: {err}");
    }
}

#[test]
fn embedding_loader_reports_unported_families_on_real_checkpoints() {
    // For a family whose forward pass has not landed yet, the dispatcher must
    // name the family and the route rather than fail on a missing tensor.
    // Ported families move out of this list: Gemma3Embedding and
    // Qwen3Embedding did so in #1329, BERT in #1321.
    for (repo, family) in [
        (
            "LiquidAI/LFM2.5-Embedding-350M",
            "LFM2 bidirectional embedder",
        ),
        ("vidore/colqwen2.5-base", "ColQwen2.5"),
    ] {
        let Some(dir) = local_checkpoint(repo) else {
            continue;
        };
        let err = err_string(load_embedding_model(&dir));
        assert!(err.contains("not yet supported"), "{repo}: {err}");
        assert!(err.contains(family), "{repo}: {err}");
    }
}

/// Every embedding and reranker checkpoint the epic's family issues use,
/// with the variant detection must return. Rerankers (`ForSequenceClassification`,
/// or a `modules.json` whose only extra module is `1_LogitScore`) must not
/// route to an embedding variant. Each entry skips when absent.
#[test]
fn local_embedding_checkpoints_detect_to_their_families() {
    let expected: &[(&str, Result<ModelType, &str>)] = &[
        (
            "sentence-transformers/all-MiniLM-L6-v2",
            Ok(ModelType::Bert),
        ),
        ("intfloat/multilingual-e5-small", Ok(ModelType::Bert)),
        ("BAAI/bge-m3", Ok(ModelType::XlmRoberta)),
        ("nomic-ai/modernbert-embed-base", Ok(ModelType::ModernBert)),
        ("google/siglip-base-patch16-224", Ok(ModelType::SiglipText)),
        (
            "mlx-community/embeddinggemma-300m-4bit",
            Ok(ModelType::Gemma3Embedding),
        ),
        ("Qwen/Qwen3-Embedding-0.6B", Ok(ModelType::Qwen3Embedding)),
        (
            "Qwen/Qwen3-VL-Embedding-2B",
            Ok(ModelType::Qwen3VLEmbedding),
        ),
        (
            "LiquidAI/LFM2.5-Embedding-350M",
            Ok(ModelType::Lfm2Embedding),
        ),
        (
            "nvidia/Nemotron-3-Embed-1B-BF16",
            Ok(ModelType::Ministral3Embedding),
        ),
        (
            "mlx-community/Nemotron-3-Embed-1B-BF16-8bit",
            Ok(ModelType::Ministral3Embedding),
        ),
        (
            "nvidia/llama-nemotron-embed-1b-v2",
            Ok(ModelType::LlamaBidirec),
        ),
        (
            "nvidia/llama-nemotron-embed-vl-1b-v2",
            Ok(ModelType::LlamaNemotronVLEmbedding),
        ),
        (
            "vidore/ColSmolVLM-Instruct-256M-base",
            Ok(ModelType::ColIdefics3),
        ),
        ("vidore/colqwen2.5-base", Ok(ModelType::ColQwen25)),
        // Rerankers: never an embedding variant.
        ("Qwen/Qwen3-VL-Reranker-2B", Ok(ModelType::Qwen3VL)),
        (
            "mlx-community/Qwen3-Reranker-0.6B-4bit",
            Ok(ModelType::Qwen3),
        ),
        (
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            Err("Unsupported model type"),
        ),
        ("BAAI/bge-reranker-v2-m3", Err("Unsupported model type")),
        (
            "Alibaba-NLP/gte-reranker-modernbert-base",
            Err("Unsupported model type"),
        ),
    ];
    let mut checked = 0;
    for (repo, want) in expected {
        let Some(dir) = local_checkpoint(repo) else {
            continue;
        };
        checked += 1;
        match (get_model_type(&dir), want) {
            (Ok(got), Ok(expected)) => assert_eq!(got, *expected, "{repo}"),
            (Err(err), Err(fragment)) => {
                assert!(err.to_string().contains(fragment), "{repo}: {err}")
            }
            (Ok(got), Err(_)) => panic!("{repo}: expected a detection error, got {got:?}"),
            (Err(err), Ok(expected)) => panic!("{repo}: expected {expected:?}, got error {err}"),
        }
    }
    eprintln!("checked {checked} local checkpoints");
}
