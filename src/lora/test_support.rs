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

//! On-disk adapter fixtures shared by the LoRA unit tests.
//!
//! The loader's entry points (`apply_lora_adapters`, `stage_runtime_adapters`)
//! read an adapter *directory*, not a [`WeightMap`], so the tests that cover
//! adapter acceptance have to write real `adapter_config.json` and
//! `adapters.safetensors` files. This module holds that writer once so the
//! fused-path tests and the runtime-path tests do not each carry a copy.
//!
//! Used by: `loader_tests.rs`, `runtime_tests.rs`, `tied_head_tests.rs`,
//! `loading/adapter_load_tests.rs`, the pipeline Llama stage adapter tests

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A safetensors view over an owned little-endian f32 buffer.
///
/// The impl is on `&OwnedTensor` because `serialize_to_file` is handed a
/// `&HashMap`, which yields borrowed values.
pub(crate) struct OwnedTensor {
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl safetensors::View for &OwnedTensor {
    fn dtype(&self) -> safetensors::tensor::Dtype {
        safetensors::tensor::Dtype::F32
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        self.data.as_slice().into()
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// An all-ones `[rows, cols]` f32 tensor, so a fused sum stays predictable.
pub(crate) fn ones_tensor(rows: usize, cols: usize) -> OwnedTensor {
    let mut data = Vec::with_capacity(rows * cols * 4);
    for _ in 0..rows * cols {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }
    OwnedTensor {
        shape: vec![rows, cols],
        data,
    }
}

/// `rows * cols` deterministic pseudo-random values in `[-0.5, 0.5)`, row
/// major, so a delta varies across rows and columns and a term applied at the
/// wrong end of a tie changes the logits (issue #2421).
pub(crate) fn seeded_values(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..rows * cols)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // The top 24 bits give an exactly representable f32 in [0, 1).
            (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

/// A `[rows, cols]` f32 tensor holding `values` (row major).
pub(crate) fn tensor_from_values(rows: usize, cols: usize, values: &[f32]) -> OwnedTensor {
    assert_eq!(values.len(), rows * cols, "values fill [{rows}, {cols}]");
    OwnedTensor {
        shape: vec![rows, cols],
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

/// A `[rows, cols]` f32 tensor of [`seeded_values`].
pub(crate) fn seeded_tensor(rows: usize, cols: usize, seed: u64) -> OwnedTensor {
    tensor_from_values(rows, cols, &seeded_values(rows, cols, seed))
}

/// An all-ones 1-D f32 tensor of length `n`.
///
/// A real DoRA magnitude vector is 1-D, so the fixtures that stand one in use
/// this rather than a degenerate `[n, 1]`.
pub(crate) fn ones_vector(n: usize) -> OwnedTensor {
    let mut data = Vec::with_capacity(n * 4);
    for _ in 0..n {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }
    OwnedTensor {
        shape: vec![n],
        data,
    }
}

/// An all-ones `[experts, rows, cols]` f32 tensor, the per-expert layout
/// mlx-lm's `LoRASwitchLinear` writes for a mixture-of-experts adapter.
pub(crate) fn ones_tensor_3d(experts: usize, rows: usize, cols: usize) -> OwnedTensor {
    let mut data = Vec::with_capacity(experts * rows * cols * 4);
    for _ in 0..experts * rows * cols {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }
    OwnedTensor {
        shape: vec![experts, rows, cols],
        data,
    }
}

/// A unique temporary directory for one test case.
///
/// The PID disambiguates parallel `cargo test` processes and the counter
/// disambiguates cases inside one process, so two tests cannot collide on the
/// same nanosecond timestamp.
pub(crate) fn temp_dir(name: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is after the unix epoch")
        .as_nanos();
    let pid = std::process::id();
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("mlxcel_lora_{name}_{pid}_{nanos}_{seq}"))
}

/// Write an adapter directory holding `tensors` and an `adapter_config.json`
/// that declares `fine_tune_type`.
///
/// The tensor map is passed in rather than derived from a layer name so a test
/// can write exactly the malformed shape it is about: an unpaired half, an
/// unknown leaf, or a pair whose layer path no base weight answers to.
pub(crate) fn write_adapter_dir(
    dir: &Path,
    fine_tune_type: &str,
    rank: usize,
    tensors: HashMap<String, OwnedTensor>,
) {
    std::fs::create_dir_all(dir).expect("create adapter dir");
    std::fs::write(
        dir.join("adapter_config.json"),
        format!(
            r#"{{"fine_tune_type": "{fine_tune_type}", "lora_parameters": {{"rank": {rank}, "scale": 1.0}}}}"#
        ),
    )
    .expect("write adapter config");
    safetensors::serialize_to_file(&tensors, None, &dir.join("adapters.safetensors"))
        .expect("write adapter safetensors");
}

/// An mlx-lm-convention pair for `layer`: `lora_a` is `[in, rank]`, `lora_b` is
/// `[rank, out]`, both all ones.
pub(crate) fn adapter_pair_tensors(
    layer: &str,
    in_features: usize,
    rank: usize,
    out_features: usize,
) -> HashMap<String, OwnedTensor> {
    let mut tensors = HashMap::new();
    tensors.insert(format!("{layer}.lora_a"), ones_tensor(in_features, rank));
    tensors.insert(format!("{layer}.lora_b"), ones_tensor(rank, out_features));
    tensors
}

/// Hidden width of the [`write_tiny_llama_checkpoint`] model.
pub(crate) const TINY_HIDDEN: usize = 8;
/// Vocabulary size of the [`write_tiny_llama_checkpoint`] model.
pub(crate) const TINY_VOCAB: usize = 32;

/// How [`write_tiny_llama_checkpoint`] lays out the output head.
pub(crate) struct TinyLlama {
    /// The `tie_word_embeddings` value written to `config.json`; `None` leaves
    /// the key out.
    pub(crate) tie_word_embeddings: Option<bool>,
    /// Whether the checkpoint stores its own `lm_head.weight`.
    pub(crate) stores_lm_head: bool,
    /// Extra `[rows, cols]` tensors the Llama constructors never read, for a
    /// fixture that needs a base weight no layer constructor claims.
    pub(crate) unread: Vec<(&'static str, usize, usize)>,
}

/// Write a complete one-layer Llama checkpoint (`config.json` plus a single
/// `model.safetensors`) that the real Llama constructors load.
///
/// Every LoRA path checks adapters against what a family's constructors do,
/// which a synthetic `WeightMap` cannot stand in for, so these fixtures go
/// through the real loaders. Dimensions: hidden 8, two query heads and one
/// KV head of width 4, MLP 16, vocabulary 32.
pub(crate) fn write_tiny_llama_checkpoint(dir: &Path, spec: &TinyLlama) {
    std::fs::create_dir_all(dir).expect("create model dir");
    let mut config = serde_json::json!({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": TINY_HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rms_norm_eps": 1e-5,
        "vocab_size": TINY_VOCAB,
    });
    if let Some(tie) = spec.tie_word_embeddings {
        config["tie_word_embeddings"] = serde_json::Value::Bool(tie);
    }
    std::fs::write(dir.join("config.json"), config.to_string()).expect("write model config");

    let h = TINY_HIDDEN;
    let mut tensors: HashMap<String, OwnedTensor> = HashMap::new();
    let mut matrix = |name: &str, rows: usize, cols: usize| {
        tensors.insert(name.to_string(), ones_tensor(rows, cols));
    };
    matrix("model.embed_tokens.weight", TINY_VOCAB, h);
    matrix("model.layers.0.self_attn.q_proj.weight", h, h);
    matrix("model.layers.0.self_attn.k_proj.weight", 4, h);
    matrix("model.layers.0.self_attn.v_proj.weight", 4, h);
    matrix("model.layers.0.self_attn.o_proj.weight", h, h);
    matrix("model.layers.0.mlp.gate_proj.weight", 16, h);
    matrix("model.layers.0.mlp.up_proj.weight", 16, h);
    matrix("model.layers.0.mlp.down_proj.weight", h, 16);
    if spec.stores_lm_head {
        matrix("lm_head.weight", TINY_VOCAB, h);
    }
    for (name, rows, cols) in &spec.unread {
        matrix(name, *rows, *cols);
    }
    for norm in [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
    ] {
        tensors.insert(norm.to_string(), ones_vector(h));
    }
    safetensors::serialize_to_file(&tensors, None, &dir.join("model.safetensors"))
        .expect("write model safetensors");
}

/// Rewrite a [`write_tiny_llama_checkpoint`] directory's `config.json` to
/// declare another family, for verdict tests that need a tied family outside
/// the ones that apply an embedding pair at both ends (issue #2421). The
/// tensors stay Llama-shaped, so only checks that run before construction can
/// use it.
pub(crate) fn relabel_checkpoint_family(dir: &Path, model_type: &str, architecture: &str) {
    let path = dir.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read model config"))
            .expect("parse model config");
    config["model_type"] = serde_json::Value::String(model_type.to_string());
    config["architectures"] = serde_json::json!([architecture]);
    std::fs::write(&path, config.to_string()).expect("write model config");
}

/// Write a WordLevel `tokenizer.json` covering the tiny vocabulary (`t0` ..
/// `t31`) into a [`write_tiny_llama_checkpoint`] directory, so the public load
/// entry points, which load the tokenizer last, can return the model.
pub(crate) fn write_tiny_tokenizer(dir: &Path) {
    let vocab: serde_json::Map<String, serde_json::Value> = (0..TINY_VOCAB)
        .map(|id| (format!("t{id}"), serde_json::Value::from(id)))
        .collect();
    let tokenizer = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "t0"},
    });
    std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).expect("write tokenizer");
}

/// Replace every 2-D tensor of a [`write_tiny_llama_checkpoint`] directory with
/// [`seeded_values`], one seed per tensor, so the tiny model's logits depend on
/// its input and on every projection (the all-ones fixture gives every token
/// the same logit). Norm vectors keep their ones.
pub(crate) fn reseed_tiny_checkpoint(dir: &Path) {
    let path = dir.join("model.safetensors");
    let bytes = std::fs::read(&path).expect("read model safetensors");
    let parsed = safetensors::SafeTensors::deserialize(&bytes).expect("parse model safetensors");
    let mut names: Vec<String> = parsed.names().into_iter().map(str::to_string).collect();
    names.sort();
    let mut tensors: HashMap<String, OwnedTensor> = HashMap::new();
    for (index, name) in names.iter().enumerate() {
        let view = parsed.tensor(name).expect("tensor");
        let shape = view.shape().to_vec();
        let tensor = if let [rows, cols] = shape[..] {
            seeded_tensor(rows, cols, 1000 + index as u64)
        } else {
            OwnedTensor {
                shape,
                data: view.data().to_vec(),
            }
        };
        tensors.insert(name.clone(), tensor);
    }
    safetensors::serialize_to_file(&tensors, None, &path).expect("write model safetensors");
}
