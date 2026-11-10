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

//! Real-IREE gates for the ragged continuous-batching hardening in issue #2923.
//!
//! These belong to the root crate because its `xla-iree` build script owns the
//! whole-archive IREE runtime link recipe; they are ignored in ordinary CI and
//! need the pinned ROCm distribution and compiler plus the `test-utils` feature.

use std::collections::HashMap;

use mlxcel_xla::{
    SampleParams, XlaBatchEngine, batch_kv_allocation_counters, reset_batch_kv_allocation_counters,
};
use safetensors::{Dtype, tensor::TensorView};
use tempfile::TempDir;

const CAPACITY: usize = 8;
const HIDDEN: usize = 8;
const VOCAB: usize = 32;

struct TinyModel {
    dir: TempDir,
}

impl TinyModel {
    fn path(&self) -> &std::path::Path {
        self.dir.path()
    }
}

fn deterministic_values(elements: usize, seed: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| (((index + seed * 17) % 29) as f32 - 14.0) * 0.003)
        .collect()
}

fn add_tensor(
    tensors: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
    name: impl Into<String>,
    shape: &[usize],
    values: Vec<f32>,
) {
    tensors.push((
        name.into(),
        shape.to_vec(),
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    ));
}

fn create_tiny_model() -> TinyModel {
    let dir = tempfile::tempdir().expect("temporary model directory");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"llama","hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"vocab_size":32,"rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true}"#,
    )
    .expect("model config");

    let embeddings = deterministic_values(VOCAB * HIDDEN, 1);
    let mut tensors = Vec::new();
    add_tensor(
        &mut tensors,
        "model.embed_tokens.weight",
        &[VOCAB, HIDDEN],
        embeddings,
    );
    add_tensor(
        &mut tensors,
        "model.norm.weight",
        &[HIDDEN],
        vec![1.0; HIDDEN],
    );
    for layer in 0..2 {
        let prefix = format!("model.layers.{layer}");
        add_tensor(
            &mut tensors,
            format!("{prefix}.input_layernorm.weight"),
            &[HIDDEN],
            vec![1.0; HIDDEN],
        );
        add_tensor(
            &mut tensors,
            format!("{prefix}.post_attention_layernorm.weight"),
            &[HIDDEN],
            vec![1.0; HIDDEN],
        );
        for (offset, suffix, shape) in [
            (2, "self_attn.q_proj.weight", vec![8, 8]),
            (3, "self_attn.k_proj.weight", vec![4, 8]),
            (4, "self_attn.v_proj.weight", vec![4, 8]),
            (5, "self_attn.o_proj.weight", vec![8, 8]),
            (6, "mlp.gate_proj.weight", vec![16, 8]),
            (7, "mlp.up_proj.weight", vec![16, 8]),
            (8, "mlp.down_proj.weight", vec![8, 16]),
        ] {
            add_tensor(
                &mut tensors,
                format!("{prefix}.{suffix}"),
                &shape,
                deterministic_values(shape.iter().product(), layer * 10 + offset),
            );
        }
    }
    let views = tensors
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).expect("valid tensor view"),
            )
        })
        .collect::<HashMap<_, _>>();
    safetensors::serialize_to_file(&views, None, &dir.path().join("model.safetensors"))
        .expect("model weights");
    TinyModel { dir }
}

fn device() -> String {
    std::env::var("MLXCEL_XLA_DEVICE").unwrap_or_else(|_| "local-task".to_string())
}

/// Item 1: the lazy rank-5 batch KV is zeroed on the device, with no host
/// mirror and no host-to-device copy of the cache on first admit.
#[test]
#[ignore = "requires the pinned IREE runtime and compiler"]
fn first_batch_admit_does_not_upload_a_host_kv_mirror() {
    let fixture = create_tiny_model();
    reset_batch_kv_allocation_counters();
    let mut engine =
        XlaBatchEngine::load_with_context_capacity(fixture.path(), 4, &device(), CAPACITY)
            .expect("batch engine");
    engine
        .submit(&[1, 2, 3], 2, SampleParams::greedy())
        .expect("submit request");
    let mut steps = 0;
    while !engine.is_idle() {
        steps += 1;
        assert!(steps < 64, "engine exceeded the safety step cap");
        engine.pump().expect("pump");
    }
    let (host_bytes, h2d_bytes) = batch_kv_allocation_counters();
    assert_eq!(
        host_bytes, 0,
        "the batch KV path must not allocate a host mirror of the rank-5 cache"
    );
    assert_eq!(
        h2d_bytes, 0,
        "the batch KV path must not upload the rank-5 cache from host memory"
    );
}
