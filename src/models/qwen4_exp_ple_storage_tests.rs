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

//! External table lookups against the resident table, bit for bit, over a
//! tiny checkpoint in both layouts and both quantization modes, through the
//! real conversion helper.

use std::path::Path;
use std::sync::Arc;

use mlxcel_core::weights::WeightMap;
use serde_json::{Value, json};
use tempfile::TempDir;

use super::super::config::Qwen4ExpConfig;
use super::super::ple_cache::AlwaysGrow;
use super::super::ple_footprint::external_ple_footprint;
use super::super::ple_manifest::PleManifest;
use super::super::ple_prepare::{
    MANIFEST_NAME, PrepareLayout, PrepareOptions, prepare_external_ple_model,
};
use super::*;

const ROW_WIDTH: usize = 32;
const SHARDS: u64 = 2;
const ROWS_PER_SHARD: u64 = 4;
const ROWS: u64 = SHARDS * ROWS_PER_SHARD;
const IDS: [i64; 7] = [7, 0, 3, 3, 5, 0, 1];

#[derive(Clone, Copy, Debug)]
enum Quant {
    Affine,
    Nvfp4,
}

const QUANTS: [Quant; 2] = [Quant::Affine, Quant::Nvfp4];
const LAYOUTS: [PrepareLayout; 2] = [PrepareLayout::SafetensorsRanges, PrepareLayout::Interleaved];

impl Quant {
    fn group(self) -> usize {
        match self {
            Self::Affine => 32,
            Self::Nvfp4 => 16,
        }
    }

    fn mode(self) -> &'static str {
        match self {
            Self::Affine => "affine",
            Self::Nvfp4 => "nvfp4",
        }
    }

    /// Tensors of one shard as `(suffix, dtype, shape, bytes)`.
    fn tensors(self, shard: u64) -> Vec<(&'static str, &'static str, Vec<u64>, Vec<u8>)> {
        let rows = ROWS_PER_SHARD;
        let words = (ROW_WIDTH / 8) as u64;
        let groups = (ROW_WIDTH / self.group()) as u64;
        let mix = |seed: u64| -> u64 {
            (seed.wrapping_add(shard).wrapping_mul(0x9E37_79B9) >> 7) & 0xFFFF_FFFF
        };
        let weight: Vec<u8> = (0..rows * words)
            .flat_map(|i| (mix(i) as u32).to_le_bytes())
            .collect();
        let scale_bf16 = |i: u64| -> [u8; 2] { (0x3F00u16 + (mix(i) % 64) as u16).to_le_bytes() };
        let bias_bf16 =
            |i: u64| -> [u8; 2] { (0x3E00u16 + (mix(i + 9) % 64) as u16).to_le_bytes() };
        match self {
            Self::Affine => vec![
                ("weight", "U32", vec![rows, words], weight),
                (
                    "scales",
                    "BF16",
                    vec![rows, groups],
                    (0..rows * groups).flat_map(scale_bf16).collect(),
                ),
                (
                    "biases",
                    "BF16",
                    vec![rows, groups],
                    (0..rows * groups).flat_map(bias_bf16).collect(),
                ),
            ],
            Self::Nvfp4 => vec![
                ("weight", "U32", vec![rows, words], weight),
                (
                    "scales",
                    "U8",
                    vec![rows, groups],
                    (0..rows * groups)
                        .map(|i| (mix(i) % 200 + 20) as u8)
                        .collect(),
                ),
            ],
        }
    }
}

const PREFIX: &str = "model.layers.1.ple.ple_embedding.ngram_embedding";

/// Write a one-layer checkpoint holding the table; returns its directory.
fn write_checkpoint(dir: &Path, quant: Quant) {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    let mut weight_map = serde_json::Map::new();
    let mut add = |key: String, dtype: &str, shape: Value, bytes: &[u8]| {
        let start = payload.len();
        payload.extend_from_slice(bytes);
        header.insert(
            key.clone(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]}),
        );
        weight_map.insert(key, json!("model-00001.safetensors"));
    };
    for shard in 0..SHARDS {
        for (name, dtype, shape, bytes) in quant.tensors(shard) {
            add(
                format!("{PREFIX}.shards.{shard}.{name}"),
                dtype,
                json!(shape),
                &bytes,
            );
        }
    }
    add(
        "model.other.weight".to_string(),
        "U8",
        json!([4]),
        &[1, 2, 3, 4],
    );
    let header = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&payload);
    std::fs::write(dir.join("model-00001.safetensors"), file).unwrap();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"metadata": {}, "weight_map": weight_map})).unwrap(),
    )
    .unwrap();
    let quantization = json!({"group_size": quant.group(), "bits": 4, "mode": quant.mode()});
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(
            &json!({"text_config": {"model_type": "qwen4_exp_text"}, "quantization": quantization}),
        )
        .unwrap(),
    )
    .unwrap();
}

fn resident_weights(quant: Quant) -> WeightMap {
    let mut weights = WeightMap::new();
    for shard in 0..SHARDS {
        for (name, dtype, shape, bytes) in quant.tensors(shard) {
            let shape: Vec<i32> = shape.iter().map(|&dim| dim as i32).collect();
            let array = match dtype {
                "U32" => mlxcel_core::from_bytes(&bytes, &shape, mlxcel_core::dtype::UINT32),
                "BF16" => mlxcel_core::from_bytes_f16(&bytes, &shape, true),
                _ => mlxcel_core::from_bytes(&bytes, &shape, mlxcel_core::dtype::UINT8),
            };
            weights.insert(format!("{PREFIX}.shards.{shard}.{name}"), array);
        }
    }
    weights
}

fn resident_table(quant: Quant) -> ShardedEmbedding {
    let fixture: Value = json!({
        "model_type": "qwen4_exp_text", "hidden_size": 40, "num_hidden_layers": 4, "ple_embed_dim": 32,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "linear_num_value_heads": 1, "linear_num_key_heads": 1, "linear_key_head_dim": 4,
        "linear_value_head_dim": 4, "linear_conv_kernel_dim": 3, "num_experts": 2,
        "num_experts_per_tok": 1, "shared_expert_intermediate_size": 8, "moe_intermediate_size": 8,
        "rms_norm_eps": 1e-6, "vocab_size": 100, "max_position_embeddings": 128,
        "split_ngram_parts": SHARDS, "eos_token_id": 3,
        "rope_parameters": {"type": "default", "mrope_section": [1, 1, 0], "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        "quantization": {"group_size": quant.group(), "bits": 4, "mode": quant.mode()},
    });
    let config = serde_json::from_value::<Qwen4ExpConfig>(fixture)
        .unwrap()
        .resolve()
        .unwrap();
    ShardedEmbedding::from_weights(&resident_weights(quant), &config, PREFIX, ROWS, ROW_WIDTH)
        .unwrap()
}

fn external_table(view: &Path, mode: PleIoMode, budget: u64) -> ExternalPleTable {
    let manifest = PleManifest::load(&view.join(MANIFEST_NAME), None).unwrap();
    ExternalPleTable::open(&manifest, mode, budget, Arc::new(AlwaysGrow)).unwrap()
}

struct Prepared {
    _source: TempDir,
    _view: TempDir,
    table: ExternalPleTable,
}

fn prepared(quant: Quant, layout: PrepareLayout, mode: PleIoMode, budget: u64) -> Prepared {
    let source = TempDir::new().unwrap();
    write_checkpoint(source.path(), quant);
    let view = TempDir::new().unwrap();
    prepare_external_ple_model(
        source.path(),
        view.path(),
        &PrepareOptions {
            layout,
            cache_rows: 0,
        },
    )
    .unwrap();
    let table = external_table(view.path(), mode, budget);
    Prepared {
        _source: source,
        _view: view,
        table,
    }
}

fn evaluated(array: UniquePtr<MlxArray>) -> (Vec<i32>, i32, Vec<u8>) {
    mlxcel_core::eval(&array);
    (
        mlxcel_core::array_shape(&array),
        mlxcel_core::array_dtype(&array),
        mlxcel_core::array_to_raw_bytes(&array),
    )
}

#[track_caller]
fn assert_same(resident: &ShardedEmbedding, external: &ExternalPleTable, ids: &[i64]) {
    let want = evaluated(PleRowStore::lookup(resident, ids).unwrap());
    let got = evaluated(external.lookup(ids).unwrap());
    assert_eq!(got, want, "ids {ids:?}");
}

#[test]
fn external_equals_resident_for_every_layout_and_quantization() {
    for quant in QUANTS {
        let resident = resident_table(quant);
        for layout in LAYOUTS {
            let fixture = prepared(quant, layout, PleIoMode::Mmap, 1 << 20);
            // Duplicates, ids in both shards, caller order.
            assert_same(&resident, &fixture.table, &IDS);
            // A second lookup serves the cache and must not change the bits.
            assert_same(&resident, &fixture.table, &IDS);
            let stats = fixture.table.stats();
            assert_eq!(stats.rows, IDS.len() as u64 * 2);
            assert!(stats.hits > 0, "{quant:?} {layout:?}: {stats:?}");
            // The second lookup is cached rows only.
            assert_eq!(stats.misses, 5, "{quant:?} {layout:?}: {stats:?}");
            let empty: &[i64] = &[];
            assert_eq!(
                evaluated(fixture.table.lookup(empty).unwrap()).0,
                vec![0, ROW_WIDTH as i32]
            );
            for bad in [ROWS as i64, -1] {
                assert!(
                    matches!(fixture.table.lookup(&[0, bad]), Err(PleStoreError::IdOutOfRange { id, .. }) if id == bad)
                );
            }
        }
    }
}

#[test]
fn every_io_mode_returns_the_same_bits() {
    for quant in QUANTS {
        let resident = resident_table(quant);
        for layout in LAYOUTS {
            for mode in [PleIoMode::Mmap, PleIoMode::Direct, PleIoMode::Ram] {
                // A budget below one row forces every read through the source.
                assert_same(&resident, &prepared(quant, layout, mode, 16).table, &IDS);
            }
        }
    }
}

#[test]
fn gather_ahead_fills_the_cache_without_changing_lookups() {
    let quant = Quant::Affine;
    let resident = resident_table(quant);
    for layout in LAYOUTS {
        let fixture = prepared(quant, layout, PleIoMode::Direct, 1 << 20);
        fixture.table.prefetch(&IDS);
        // A lookup waits for the gather-ahead and finds everything cached.
        assert_same(&resident, &fixture.table, &IDS);
        let stats = fixture.table.stats();
        assert_eq!(stats.prefetched_rows, 5);
        assert_eq!(stats.misses, 0, "{stats:?}");
    }
}

#[test]
fn a_cache_smaller_than_a_lookup_still_returns_every_row() {
    let quant = Quant::Nvfp4;
    let resident = resident_table(quant);
    // Room for one record: duplicates still resolve and eviction stays bounded.
    let stride = (ROW_WIDTH / 2 + ROW_WIDTH / 16) as u64;
    let fixture = prepared(
        quant,
        PrepareLayout::Interleaved,
        PleIoMode::Mmap,
        stride + 56,
    );
    assert_same(&resident, &fixture.table, &IDS);
    assert_same(&resident, &fixture.table, &IDS);
}

#[test]
fn a_view_drops_the_table_keys_from_its_index_and_accounts_for_only_the_cache() {
    let source = TempDir::new().unwrap();
    write_checkpoint(source.path(), Quant::Affine);
    let view = TempDir::new().unwrap();
    prepare_external_ple_model(
        source.path(),
        view.path(),
        &PrepareOptions {
            layout: PrepareLayout::Interleaved,
            cache_rows: 0,
        },
    )
    .unwrap();
    let index: Value = serde_json::from_slice(
        &std::fs::read(view.path().join("model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    assert!(
        !index["weight_map"]
            .as_object()
            .unwrap()
            .keys()
            .any(|key| key.contains("ple_embedding"))
    );
    assert_eq!(index["metadata"]["external_ple_bytes"], json!("160"));
    let config: Value =
        serde_json::from_slice(&std::fs::read(view.path().join("config.json")).unwrap()).unwrap();
    assert_eq!(
        config["text_config"]["ple_storage"]["manifest"],
        json!(MANIFEST_NAME)
    );

    let footprint = external_ple_footprint(view.path()).unwrap();
    assert_eq!(
        footprint.weight_bytes, 4,
        "only the non-table tensor is resident"
    );
    assert_eq!(footprint.row_memory_bytes, DEFAULT_PLE_CACHE_BYTES);
}

#[test]
fn the_helper_refuses_a_non_empty_target_and_a_checkpoint_without_a_table() {
    let source = TempDir::new().unwrap();
    write_checkpoint(source.path(), Quant::Affine);
    let occupied = TempDir::new().unwrap();
    std::fs::write(occupied.path().join("x"), b"x").unwrap();
    assert!(
        prepare_external_ple_model(source.path(), occupied.path(), &PrepareOptions::default())
            .is_err()
    );

    let plain = TempDir::new().unwrap();
    std::fs::write(
        plain.path().join("model.safetensors.index.json"),
        b"{\"weight_map\": {}}",
    )
    .unwrap();
    std::fs::write(plain.path().join("config.json"), b"{}").unwrap();
    let target = TempDir::new().unwrap();
    let error = prepare_external_ple_model(plain.path(), target.path(), &PrepareOptions::default())
        .unwrap_err();
    assert!(
        error.to_string().contains("no Qwen4-Exp PLE shards"),
        "{error}"
    );
}

/// Cross-checks views written by mlx-vlm's own `prepare_external_ple_model`
/// and `materialize_interleaved_ple_store`: point `PLE_INTEROP_DIR` at a
/// directory with `src`, `upstream-ranges` and `upstream-interleaved`. Without
/// the variable this is a no-op, so the suite stays hermetic.
#[test]
fn upstream_prepared_views_read_the_same_rows_as_the_helper() {
    let Ok(root) = std::env::var("PLE_INTEROP_DIR") else {
        return;
    };
    let root = Path::new(&root);
    let helper_view = TempDir::new().unwrap();
    prepare_external_ple_model(
        &root.join("src"),
        helper_view.path(),
        &PrepareOptions {
            layout: PrepareLayout::SafetensorsRanges,
            cache_rows: 0,
        },
    )
    .unwrap();
    let want = evaluated(
        external_table(helper_view.path(), PleIoMode::Mmap, 1 << 20)
            .lookup(&IDS)
            .unwrap(),
    );
    for view in ["upstream-ranges", "upstream-interleaved"] {
        let manifest = PleManifest::load(&root.join(view).join(MANIFEST_NAME), None).unwrap();
        let table =
            ExternalPleTable::open(&manifest, PleIoMode::Mmap, 1 << 20, Arc::new(AlwaysGrow))
                .unwrap();
        assert_eq!(evaluated(table.lookup(&IDS).unwrap()), want, "{view}");
    }
}

/// Rewrite the index of a checkpoint so `key` names `file`.
fn point_index_at(dir: &Path, key: &str, file: &str) {
    let path = dir.join("model.safetensors.index.json");
    let mut index: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    index["weight_map"][key] = json!(file);
    std::fs::write(path, serde_json::to_vec(&index).unwrap()).unwrap();
}

fn prepare_error(source: &Path) -> String {
    let view = TempDir::new().unwrap();
    format!(
        "{:#}",
        prepare_external_ple_model(source, view.path(), &PrepareOptions::default())
            .expect_err("the helper must refuse this checkpoint")
    )
}

#[test]
fn the_helper_refuses_a_shard_path_leaving_the_checkpoint() {
    let root = TempDir::new().unwrap();
    let source = root.path().join("source");
    std::fs::create_dir(&source).unwrap();
    write_checkpoint(&source, Quant::Affine);
    std::fs::copy(
        source.join("model-00001.safetensors"),
        root.path().join("outside.safetensors"),
    )
    .unwrap();
    point_index_at(&source, "model.other.weight", "../outside.safetensors");
    let error = prepare_error(&source);
    assert!(error.contains("refusing to link"), "{error}");
}

#[test]
fn the_helper_refuses_to_link_a_shard_over_a_file_it_rewrites() {
    let source = TempDir::new().unwrap();
    write_checkpoint(source.path(), Quant::Affine);
    point_index_at(source.path(), "model.other.weight", "config.json");
    let error = prepare_error(source.path());
    assert!(error.contains("not a .safetensors file"), "{error}");
}

#[cfg(unix)]
#[test]
fn the_helper_refuses_to_copy_a_symlink_leaving_the_checkpoint() {
    let source = TempDir::new().unwrap();
    write_checkpoint(source.path(), Quant::Affine);
    let outside = TempDir::new().unwrap();
    std::fs::write(outside.path().join("secret"), b"secret").unwrap();
    std::os::unix::fs::symlink(
        outside.path().join("secret"),
        source.path().join("notes.txt"),
    )
    .unwrap();
    let error = prepare_error(source.path());
    assert!(error.contains("refusing to copy"), "{error}");
}

#[test]
fn the_helper_does_not_let_a_source_file_replace_the_manifest() {
    let source = TempDir::new().unwrap();
    write_checkpoint(source.path(), Quant::Affine);
    std::fs::write(source.path().join(MANIFEST_NAME), b"planted").unwrap();
    let view = TempDir::new().unwrap();
    prepare_external_ple_model(source.path(), view.path(), &PrepareOptions::default()).unwrap();
    assert_ne!(
        std::fs::read(view.path().join(MANIFEST_NAME)).unwrap(),
        b"planted"
    );
}

#[test]
fn the_helper_refuses_a_header_size_past_the_end_of_the_file() {
    let source = TempDir::new().unwrap();
    write_checkpoint(source.path(), Quant::Affine);
    let path = source.path().join("model-00001.safetensors");
    let mut file = std::fs::read(&path).unwrap();
    file[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    std::fs::write(path, file).unwrap();
    let error = prepare_error(source.path());
    assert!(error.contains("header size"), "{error}");
}

#[test]
fn the_helper_names_shards_that_disagree_on_row_width() {
    let source = TempDir::new().unwrap();
    let mut header = serde_json::Map::new();
    let mut weight_map = serde_json::Map::new();
    let mut end = 0u64;
    for (shard, words, groups) in [(0, 4u64, 1u64), (1, 8, 2)] {
        for (name, dtype, width, item) in [
            ("weight", "U32", words, 4u64),
            ("scales", "BF16", groups, 2),
            ("biases", "BF16", groups, 2),
        ] {
            let key = format!("{PREFIX}.shards.{shard}.{name}");
            let next = end + width * item;
            header.insert(
                key.clone(),
                json!({"dtype": dtype, "shape": [1, width], "data_offsets": [end, next]}),
            );
            weight_map.insert(key, json!("model-00001.safetensors"));
            end = next;
        }
    }
    let header = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.resize(file.len() + end as usize, 0);
    std::fs::write(source.path().join("model-00001.safetensors"), file).unwrap();
    std::fs::write(
        source.path().join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"metadata": {}, "weight_map": weight_map})).unwrap(),
    )
    .unwrap();
    let quantization = json!({"group_size": 32, "bits": 4, "mode": "affine"});
    std::fs::write(
        source.path().join("config.json"),
        serde_json::to_vec(&json!({"text_config": {}, "quantization": quantization})).unwrap(),
    )
    .unwrap();
    let error = prepare_error(source.path());
    assert!(error.contains("row_width"), "{error}");
}

#[test]
fn the_manifest_path_must_stay_in_the_model_directory() {
    let dir = TempDir::new().unwrap();
    let storage = |manifest: &str| PleStorageConfig {
        manifest: manifest.to_string(),
        cache_rows: None,
    };
    std::fs::write(dir.path().join("ple-store.json"), b"{}").unwrap();
    assert!(resolve_manifest_path(&storage("ple-store.json"), Some(dir.path())).is_ok());
    let outside = dir.path().join("ple-store.json");
    for manifest in ["../ple-store.json", outside.to_str().unwrap()] {
        let error = resolve_manifest_path(&storage(manifest), Some(dir.path())).unwrap_err();
        assert_eq!(error.field, "ple_storage.manifest");
    }
}

#[test]
fn a_panicking_gather_ahead_clears_the_in_flight_flag() {
    let prepared = prepared(
        Quant::Affine,
        PrepareLayout::Interleaved,
        PleIoMode::Mmap,
        1 << 20,
    );
    let shared = &prepared.table.shared;
    *shared.in_flight.0.lock().unwrap() = true;
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _finish = FinishGatherAhead(shared);
        panic!("simulated gather-ahead failure");
    }));
    assert!(outcome.is_err());
    assert!(!*shared.in_flight.0.lock().unwrap());
}
