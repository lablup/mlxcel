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
use crate::distributed::pipeline::partial_loading::{LayerFilter, filter_weight_map};

// ---------------------------------------------------------------------------
// Stage-local LoRA adapter filter (layer-range inclusion)
// ---------------------------------------------------------------------------

#[test]
fn should_load_adapter_key_layer_in_range() {
    let filter = LayerFilter {
        layer_range: 8..16,
        has_embedding: false,
        has_lm_head: false,
    };
    assert!(should_load_adapter_key(
        "model.layers.8.self_attn.q_proj.lora_a",
        &filter
    ));
    assert!(should_load_adapter_key(
        "model.layers.15.mlp.up_proj.lora_b",
        &filter
    ));
}

#[test]
fn should_load_adapter_key_layer_out_of_range() {
    let filter = LayerFilter {
        layer_range: 8..16,
        has_embedding: false,
        has_lm_head: false,
    };
    assert!(!should_load_adapter_key(
        "model.layers.0.self_attn.q_proj.lora_a",
        &filter
    ));
    assert!(!should_load_adapter_key(
        "model.layers.7.self_attn.q_proj.lora_b",
        &filter
    ));
    assert!(!should_load_adapter_key(
        "model.layers.16.mlp.up_proj.lora_a",
        &filter
    ));
    assert!(!should_load_adapter_key(
        "model.layers.31.mlp.down_proj.lora_b",
        &filter
    ));
}

#[test]
fn should_load_adapter_key_first_stage_embedding() {
    // First stage only: embedding-scoped adapter tensors included.
    let first = LayerFilter {
        layer_range: 0..8,
        has_embedding: true,
        has_lm_head: false,
    };
    assert!(should_load_adapter_key("model.embed_tokens.lora_a", &first));

    // Middle stage: embedding-scoped adapter tensors excluded.
    let middle = LayerFilter {
        layer_range: 8..16,
        has_embedding: false,
        has_lm_head: false,
    };
    assert!(!should_load_adapter_key(
        "model.embed_tokens.lora_a",
        &middle
    ));
}

#[test]
fn should_load_adapter_key_last_stage_lm_head() {
    // Last stage: lm_head-scoped adapter tensors included.
    let last = LayerFilter {
        layer_range: 16..32,
        has_embedding: false,
        has_lm_head: true,
    };
    assert!(should_load_adapter_key("lm_head.lora_a", &last));

    // First stage: lm_head-scoped adapter tensors excluded.
    let first = LayerFilter {
        layer_range: 0..16,
        has_embedding: true,
        has_lm_head: false,
    };
    assert!(!should_load_adapter_key("lm_head.lora_b", &first));
}

#[test]
fn should_load_adapter_key_language_model_prefix() {
    // VLM-style adapter keys must resolve to the same stage policy as the
    // base weights under the language_model.model.layers. prefix.
    let filter = LayerFilter {
        layer_range: 4..8,
        has_embedding: false,
        has_lm_head: false,
    };
    assert!(should_load_adapter_key(
        "language_model.model.layers.5.self_attn.q_proj.lora_a",
        &filter
    ));
    assert!(!should_load_adapter_key(
        "language_model.model.layers.3.self_attn.q_proj.lora_b",
        &filter
    ));
}

#[test]
fn filter_adapter_weights_drops_out_of_range_tensors() {
    use mlxcel_core::weights::WeightMap;

    let mut adapter = WeightMap::new();
    let dummy = || mlxcel_core::ones(&[2, 2], mlxcel_core::dtype::FLOAT32);

    adapter.insert("model.layers.0.self_attn.q_proj.lora_a".into(), dummy());
    adapter.insert("model.layers.0.self_attn.q_proj.lora_b".into(), dummy());
    adapter.insert("model.layers.1.self_attn.q_proj.lora_a".into(), dummy());
    adapter.insert("model.layers.1.self_attn.q_proj.lora_b".into(), dummy());
    adapter.insert("model.layers.2.self_attn.q_proj.lora_a".into(), dummy());
    adapter.insert("model.layers.2.self_attn.q_proj.lora_b".into(), dummy());

    let stage_last = LayerFilter {
        layer_range: 2..3,
        has_embedding: false,
        has_lm_head: true,
    };
    let removed = filter_adapter_weights(&mut adapter, &stage_last);
    assert_eq!(removed, 4);
    assert_eq!(adapter.len(), 2);
    assert!(adapter.contains_key("model.layers.2.self_attn.q_proj.lora_a"));
    assert!(adapter.contains_key("model.layers.2.self_attn.q_proj.lora_b"));
}

#[test]
fn filter_adapter_weights_full_range_keeps_everything() {
    use mlxcel_core::weights::WeightMap;

    let mut adapter = WeightMap::new();
    let dummy = || mlxcel_core::ones(&[2, 2], mlxcel_core::dtype::FLOAT32);

    adapter.insert("model.layers.0.mlp.up_proj.lora_a".into(), dummy());
    adapter.insert("model.layers.0.mlp.up_proj.lora_b".into(), dummy());
    adapter.insert("model.embed_tokens.lora_a".into(), dummy());
    adapter.insert("lm_head.lora_a".into(), dummy());

    let full = LayerFilter {
        layer_range: 0..1,
        has_embedding: true,
        has_lm_head: true,
    };
    let removed = filter_adapter_weights(&mut adapter, &full);
    assert_eq!(removed, 0);
    assert_eq!(adapter.len(), 4);
}

// ---------------------------------------------------------------------------
// Stage-local base+adapter composition (fuse_lora_weights_into semantics)
// ---------------------------------------------------------------------------

#[test]
fn stage_filtered_fusion_matches_full_fusion_on_stage_subset() {
    // Build a synthetic two-stage setup with 4 layers. Run two fusions:
    //   (a) Apply the adapter to a full weight map, then trim to stage 1's layers.
    //   (b) Stage-filter the adapter to stage 1's range first, fuse in place.
    // Both paths must produce byte-identical fused weights on the stage's keys.
    use mlxcel_core::weights::WeightMap;

    // Base weights — 4 layers worth of one projection each, plus a non-layer key.
    let base_tensor = |seed: f32| {
        let data = vec![seed, seed + 0.1, seed + 0.2, seed + 0.3];
        mlxcel_core::from_slice_f32(&data, &[2, 2])
    };
    let mut base_full = WeightMap::new();
    base_full.insert(
        "model.layers.0.self_attn.q_proj.weight".into(),
        base_tensor(1.0),
    );
    base_full.insert(
        "model.layers.1.self_attn.q_proj.weight".into(),
        base_tensor(2.0),
    );
    base_full.insert(
        "model.layers.2.self_attn.q_proj.weight".into(),
        base_tensor(3.0),
    );
    base_full.insert(
        "model.layers.3.self_attn.q_proj.weight".into(),
        base_tensor(4.0),
    );

    // Adapter tensors for all 4 layers (mlx-lm convention:
    // lora_a shape (in, rank), lora_b shape (rank, out)).
    let mut adapter_full = WeightMap::new();
    for layer in 0..4 {
        let a = mlxcel_core::from_slice_f32(&[0.1 * (layer as f32 + 1.0); 4], &[2, 2]);
        let b = mlxcel_core::from_slice_f32(&[0.2 * (layer as f32 + 1.0); 4], &[2, 2]);
        adapter_full.insert(format!("model.layers.{layer}.self_attn.q_proj.lora_a"), a);
        adapter_full.insert(format!("model.layers.{layer}.self_attn.q_proj.lora_b"), b);
    }

    let stage_filter = LayerFilter {
        layer_range: 2..4,
        has_embedding: false,
        has_lm_head: true,
    };
    let scale = 0.5;

    // Path (a): full fusion then stage filter.
    let mut path_a = crate::lora::fuse_lora_weights(&base_full, &adapter_full, scale).unwrap();
    filter_weight_map(&mut path_a, &stage_filter);

    // Path (b): stage-filter the adapter, then fuse into a stage-filtered base
    // weight map in place.
    let mut path_b: WeightMap = base_full
        .iter()
        .map(|(k, v)| (k.clone(), mlxcel_core::copy(v)))
        .collect();
    filter_weight_map(&mut path_b, &stage_filter);

    let mut stage_adapter: WeightMap = adapter_full
        .iter()
        .map(|(k, v)| (k.clone(), mlxcel_core::copy(v)))
        .collect();
    filter_adapter_weights(&mut stage_adapter, &stage_filter);
    crate::lora::fuse_lora_weights_into(&mut path_b, &stage_adapter, scale).unwrap();

    // Both paths must have the same set of keys (stage-2 & stage-3 only).
    assert_eq!(path_a.len(), path_b.len());
    let mut a_keys: Vec<&String> = path_a.keys().collect();
    let mut b_keys: Vec<&String> = path_b.keys().collect();
    a_keys.sort();
    b_keys.sort();
    assert_eq!(a_keys, b_keys);

    for key in path_a.keys() {
        let a = path_a.get(key).unwrap();
        let b = path_b.get(key).unwrap();
        mlxcel_core::eval(a);
        mlxcel_core::eval(b);
        let a_sum = mlxcel_core::sum_all(a);
        let b_sum = mlxcel_core::sum_all(b);
        mlxcel_core::eval(&a_sum);
        mlxcel_core::eval(&b_sum);
        let a_val = mlxcel_core::item_f32(&a_sum);
        let b_val = mlxcel_core::item_f32(&b_sum);
        assert!(
            (a_val - b_val).abs() < 1e-4,
            "stage-filtered fusion diverges for key {key}: full={a_val}, stage={b_val}",
        );
    }

    // Sanity: path_b must not contain out-of-range adapter fingerprint.
    // The layer-0 key is dropped by the stage filter, so it is absent from
    // the final fused map — verifying we never fuse out-of-range tensors.
    assert!(!path_b.contains_key("model.layers.0.self_attn.q_proj.weight"));
    assert!(!path_b.contains_key("model.layers.1.self_attn.q_proj.weight"));
}

// ---------------------------------------------------------------------------
// load_stage_adapter_weights — end-to-end over a real safetensors fixture
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct OwnedTensor {
    dtype: safetensors::tensor::Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl safetensors::View for &OwnedTensor {
    fn dtype(&self) -> safetensors::tensor::Dtype {
        self.dtype
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

impl safetensors::View for OwnedTensor {
    fn dtype(&self) -> safetensors::tensor::Dtype {
        self.dtype
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

fn make_tensor(rows: usize, cols: usize, seed: f32) -> OwnedTensor {
    let n = rows * cols;
    let mut data = Vec::with_capacity(n * 4);
    for i in 0..n {
        data.extend_from_slice(&(seed + i as f32 * 0.01).to_le_bytes());
    }
    OwnedTensor {
        dtype: safetensors::tensor::Dtype::F32,
        shape: vec![rows, cols],
        data,
    }
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    // Include PID to disambiguate across parallel cargo test runs, and a
    // process-wide atomic counter to disambiguate across test cases in the
    // same process. This prevents `std::fs::create_dir_all` races when two
    // tests hit the same nanosecond timestamp.
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let pid = std::process::id();
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("mlxcel_pp_lora_{name}_{pid}_{nanos}_{seq}"))
}

#[test]
fn load_stage_adapter_weights_skips_out_of_range_tensors() {
    use std::collections::HashMap;

    let tmp_dir = temp_dir("test");
    std::fs::create_dir_all(&tmp_dir).unwrap();

    let mut tensors: HashMap<String, OwnedTensor> = HashMap::new();
    // Adapter entries for 4 layers — two of them belong to the stage
    // (layers 2..4), two do not (layers 0..2).
    for layer in 0..4 {
        tensors.insert(
            format!("model.layers.{layer}.self_attn.q_proj.lora_a"),
            make_tensor(2, 4, 0.1 * (layer + 1) as f32),
        );
        tensors.insert(
            format!("model.layers.{layer}.self_attn.q_proj.lora_b"),
            make_tensor(4, 2, 0.2 * (layer + 1) as f32),
        );
    }
    let adapter_path = tmp_dir.join("adapters.safetensors");
    safetensors::serialize_to_file(&tensors, None, &adapter_path).unwrap();

    let filter = LayerFilter {
        layer_range: 2..4,
        has_embedding: false,
        has_lm_head: true,
    };

    let loaded = load_stage_adapter_weights(&tmp_dir, &filter).unwrap();

    // Must include only the stage's layer-range adapter tensors.
    assert_eq!(loaded.len(), 4, "loaded keys: {:?}", loaded.keys());
    assert!(loaded.contains_key("model.layers.2.self_attn.q_proj.lora_a"));
    assert!(loaded.contains_key("model.layers.2.self_attn.q_proj.lora_b"));
    assert!(loaded.contains_key("model.layers.3.self_attn.q_proj.lora_a"));
    assert!(loaded.contains_key("model.layers.3.self_attn.q_proj.lora_b"));

    // Out-of-range tensors must be skipped, not buffered-and-discarded.
    assert!(!loaded.contains_key("model.layers.0.self_attn.q_proj.lora_a"));
    assert!(!loaded.contains_key("model.layers.1.self_attn.q_proj.lora_b"));

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn load_stage_adapter_weights_falls_back_to_adapter_model_safetensors() {
    use std::collections::HashMap;

    let tmp_dir = temp_dir("peft");
    std::fs::create_dir_all(&tmp_dir).unwrap();

    let mut tensors: HashMap<String, OwnedTensor> = HashMap::new();
    tensors.insert(
        "model.layers.0.self_attn.q_proj.lora_a".to_string(),
        OwnedTensor {
            dtype: safetensors::tensor::Dtype::F32,
            shape: vec![2, 4],
            data: vec![0u8; 8 * 4],
        },
    );
    // Write only the HuggingFace PEFT-style filename.
    let adapter_path = tmp_dir.join("adapter_model.safetensors");
    safetensors::serialize_to_file(&tensors, None, &adapter_path).unwrap();

    let filter = LayerFilter {
        layer_range: 0..1,
        has_embedding: true,
        has_lm_head: false,
    };
    let loaded = load_stage_adapter_weights(&tmp_dir, &filter).unwrap();
    assert_eq!(loaded.len(), 1);
    assert!(loaded.contains_key("model.layers.0.self_attn.q_proj.lora_a"));

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn resolve_adapter_weights_path_errors_on_missing_files() {
    let tmp_dir = temp_dir("missing");
    std::fs::create_dir_all(&tmp_dir).unwrap();

    let err = resolve_adapter_weights_path(&tmp_dir).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("adapters.safetensors") && msg.contains("adapter_model.safetensors"),
        "unexpected error: {msg}"
    );

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn stage_filtered_fusion_is_noop_on_empty_stage_adapter() {
    // If the stage-filtered adapter has no matching layers, base weights
    // must remain unchanged (bit-identical) after fuse_lora_weights_into.
    use mlxcel_core::weights::WeightMap;

    let mut base: WeightMap = WeightMap::new();
    base.insert(
        "model.layers.4.self_attn.q_proj.weight".into(),
        mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
    );

    // An adapter-like map that contains no matching layer in the stage range.
    let empty_stage_adapter = WeightMap::new();

    crate::lora::fuse_lora_weights_into(&mut base, &empty_stage_adapter, 0.5).unwrap();

    let fused = base.get("model.layers.4.self_attn.q_proj.weight").unwrap();
    mlxcel_core::eval(fused);
    let sum = mlxcel_core::sum_all(fused);
    mlxcel_core::eval(&sum);
    let val = mlxcel_core::item_f32(&sum);
    assert!((val - 10.0).abs() < 1e-5, "base drifted: {val}");
}

// ---------------------------------------------------------------------------
// apply_stage_lora_adapter: the pipeline-parallel entry point
//
// This path never goes through `load_model_with_adapter`, so it needs its own
// coverage of the base-weight layout guard rather than inheriting the non-PP
// path's.
// ---------------------------------------------------------------------------

fn ones_tensor(rows: usize, cols: usize) -> OwnedTensor {
    let mut data = Vec::with_capacity(rows * cols * 4);
    for _ in 0..rows * cols {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }
    OwnedTensor {
        dtype: safetensors::tensor::Dtype::F32,
        shape: vec![rows, cols],
        data,
    }
}

/// Write an mlx-lm-style adapter directory: `lora_a` is `[in, rank]` and
/// `lora_b` is `[rank, out]`, both all ones, so every delta entry is
/// `scale * rank`.
fn write_adapter_dir(dir: &std::path::Path, layer: &str, in_f: usize, rank: usize, out_f: usize) {
    use std::collections::HashMap;

    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("adapter_config.json"),
        format!(
            r#"{{"fine_tune_type": "lora", "lora_parameters": {{"rank": {rank}, "scale": 1.0}}}}"#
        ),
    )
    .unwrap();

    let mut tensors: HashMap<String, OwnedTensor> = HashMap::new();
    tensors.insert(format!("{layer}.lora_a"), ones_tensor(in_f, rank));
    tensors.insert(format!("{layer}.lora_b"), ones_tensor(rank, out_f));
    safetensors::serialize_to_file(&tensors, None, &dir.join("adapters.safetensors")).unwrap();
}

fn sum_of(weights: &mlxcel_core::weights::WeightMap, key: &str) -> f32 {
    let w = weights.get(key).unwrap();
    mlxcel_core::eval(w);
    let sum = mlxcel_core::sum_all(w);
    mlxcel_core::eval(&sum);
    mlxcel_core::item_f32(&sum)
}

#[test]
fn stage_local_fusion_rejects_a_conv1d_layout_stage_weight_map() {
    // A pipeline stage holding a raw HuggingFace GPT-2 export: `c_proj` is
    // square, so the delta broadcasts and nothing downstream would notice the
    // transposed accumulation. The stage path must refuse it just like the
    // single-process path does.
    let args = crate::models::gpt2::tests::tiny_args();
    let mut base = crate::models::gpt2::tests::raw_hf_weights(&args, "transformer.");
    let h = args.n_embd;

    let tmp_dir = temp_dir("conv1d");
    write_adapter_dir(&tmp_dir, "transformer.h.0.attn.c_proj", h, 2, h);

    let filter = LayerFilter {
        layer_range: 0..1,
        has_embedding: true,
        has_lm_head: true,
    };

    let key = "transformer.h.0.attn.c_proj.weight";
    let before = sum_of(&base, key);

    let err = match crate::lora::apply_stage_lora_adapter(&mut base, &tmp_dir, &filter) {
        Ok(()) => panic!("a Conv1D-layout stage weight map must not fuse"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("Conv1D"), "{err}");
    assert!(err.contains(key), "{err}");

    let after = sum_of(&base, key);
    assert!(
        (before - after).abs() < 1e-6,
        "stage weights must be left untouched: {before} -> {after}"
    );

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn stage_local_fusion_still_applies_to_an_out_in_stage_weight_map() {
    // Positive control: the layout guard must not disturb the layout every
    // other family (and every MLX conversion) actually ships.
    use mlxcel_core::weights::WeightMap;

    let mut base: WeightMap = WeightMap::new();
    base.insert(
        "model.layers.0.self_attn.q_proj.weight".into(),
        mlxcel_core::ones(&[4, 4], mlxcel_core::dtype::FLOAT32),
    );

    let tmp_dir = temp_dir("outin");
    write_adapter_dir(&tmp_dir, "model.layers.0.self_attn.q_proj", 4, 2, 4);

    let filter = LayerFilter {
        layer_range: 0..1,
        has_embedding: false,
        has_lm_head: false,
    };

    crate::lora::apply_stage_lora_adapter(&mut base, &tmp_dir, &filter)
        .expect("stage fusion succeeds on an [out, in] weight map");

    // 16 entries of 1.0, plus a delta of scale (1.0) * rank (2) everywhere.
    let fused = sum_of(&base, "model.layers.0.self_attn.q_proj.weight");
    assert!((fused - 48.0).abs() < 1e-4, "unexpected fused sum: {fused}");

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
