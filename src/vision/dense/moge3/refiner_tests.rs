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

use super::budget::{self, Limits, Request};
use super::*;

fn request(workspace: usize) -> Request {
    Request {
        phase: "test",
        level: 2,
        rows: 0,
        channels: 128,
        skips: 0,
        current_map: 0,
        workspace,
        protected: None,
    }
}

#[test]
fn moge3_refiner_cache_planner_reclaims_recorded_peak_without_raising_limits() {
    // The failed run reported only its aggregate peak. Its finest map's
    // size is known from the 784x1168 grid; no later occupancy is assumed.
    let finest = neighbor_map_bytes(784 * 1168).unwrap();
    assert_eq!(finest, 98_896_896);
    let plan = budget::plan(&[finest], &request(565_224_328 - finest), Limits::default()).unwrap();
    assert_eq!(plan.evicted, [0]);
    assert_eq!(plan.peak, 466_327_432);
    assert_eq!(plan.neighbors, 0);
    let kept = budget::plan(&[finest], &request(100), Limits::default()).unwrap();
    assert!(kept.evicted.is_empty());
    assert_eq!(kept.peak, finest + 100);
}

#[test]
fn moge3_refiner_cache_planner_rejects_mandatory_state_and_overflow() {
    let rows = 2_000_000;
    let mut req = request(neighbor_construction_peak_bytes(rows, 32, 0).unwrap());
    req.rows = rows;
    req.channels = 32;
    req.current_map = neighbor_map_bytes(rows).unwrap();
    let error = budget::plan(&[1024], &req, Limits::default())
        .err()
        .unwrap()
        .to_string();
    for detail in [
        "level=2",
        "rows=2000000",
        "channels=32",
        "skips=0",
        "workspace=",
        "peak=",
        "working_limit=536870912",
    ] {
        assert!(error.contains(detail), "{error}");
    }
    let safe_rows = 921_600;
    req.rows = safe_rows;
    req.current_map = neighbor_map_bytes(safe_rows).unwrap();
    req.workspace = neighbor_construction_peak_bytes(safe_rows, 32, 0).unwrap();
    assert!(budget::plan(&[], &req, Limits::default()).is_ok());
    assert!(budget::plan(&[usize::MAX], &request(1), Limits::default()).is_err());
    assert!(budget::feature_bytes(usize::MAX, 2).is_err());
    assert!(budget::skip_bytes(usize::MAX, 0).is_err());
    let mut protected = request(0);
    protected.protected = Some(0);
    assert!(
        budget::plan(
            &[11],
            &protected,
            Limits {
                working: 100,
                neighbors: 10
            }
        )
        .is_err()
    );
    let evicted = budget::plan(
        &[11],
        &request(0),
        Limits {
            working: 100,
            neighbors: 10,
        },
    )
    .unwrap();
    assert_eq!(evicted.evicted, [0]);
}

fn insert(weights: &mut WeightMap, key: String, shape: &[i32], scale: f32) {
    let count: usize = shape.iter().map(|&n| n as usize).product();
    let values: Vec<f32> = (0..count)
        .map(|i| ((i % 17) as f32 - 8.0) * scale)
        .collect();
    weights.insert(key, mlxcel_core::from_slice_f32(&values, shape));
}
fn linear(weights: &mut WeightMap, prefix: &str, input: usize, output: usize) {
    insert(
        weights,
        format!("{prefix}.weight"),
        &[output as i32, input as i32],
        0.01,
    );
    insert(weights, format!("{prefix}.bias"), &[output as i32], 0.005);
}
fn residual(weights: &mut WeightMap, prefix: &str, channels: usize) {
    weights.insert(
        format!("{prefix}.norm1.weight"),
        mlxcel_core::ones(&[channels as i32], mlxcel_core::dtype::FLOAT32),
    );
    insert(
        weights,
        format!("{prefix}.norm1.bias"),
        &[channels as i32],
        0.01,
    );
    for conv in ["conv1", "conv2"] {
        insert(
            weights,
            format!("{prefix}.{conv}.weight"),
            &[channels as i32, 3, 3, 3, channels as i32],
            0.002,
        );
        insert(
            weights,
            format!("{prefix}.{conv}.bias"),
            &[channels as i32],
            0.003,
        );
    }
}

#[test]
fn moge3_refiner_forced_cache_pressure_preserves_full_unet_output() {
    let config = RefinerConfig {
        model_channels: vec![2, 4, 8],
        encoder_channels: 2,
        encoder_downsample: 4,
        downsample_factors: vec![2, 2],
        ..Default::default()
    };
    let mut weights = WeightMap::new();
    for (level, &channels) in config.model_channels.iter().enumerate() {
        residual(
            &mut weights,
            &format!("refiner.down_stages.{level}.0"),
            channels,
        );
    }
    for i in 0..2 {
        linear(
            &mut weights,
            &format!("refiner.downsample_blocks.{i}.linear"),
            config.model_channels[i],
            config.model_channels[i + 1],
        );
        let level = 2 - i;
        linear(
            &mut weights,
            &format!("refiner.upsample_blocks.{i}.linear"),
            config.model_channels[level],
            config.model_channels[level - 1],
        );
        residual(
            &mut weights,
            &format!("refiner.up_stages.{i}.0"),
            config.model_channels[level - 1],
        );
    }
    residual(&mut weights, "refiner.bottleneck_stage.0", 8);
    for (prefix, input, output) in [
        ("input_proj", 3, 2),
        ("encoder_fuse", 2, 8),
        ("fuse_proj.0", 16, 8),
        ("fuse_proj.2", 8, 8),
        ("out_proj", 2, 1),
    ] {
        linear(&mut weights, &format!("refiner.{prefix}"), input, output);
    }
    let model = Sparse3dUnet::from_weights(&weights, &config).unwrap();
    let coords: Vec<i32> = (0..4)
        .flat_map(|row| (0..4).flat_map(move |col| [0, row, col, (row + col) % 4]))
        .collect();
    let coords = mlxcel_core::from_slice_i32(&coords, &[16, 4]);
    let data: Vec<f32> = (0..48).map(|i| (i as f32 - 23.0) / 24.0).collect();
    let features = mlxcel_core::from_slice_f32(&data, &[16, 3]);
    let encoder = mlxcel_core::from_slice_f32(&[0.25, -0.125], &[1, 1, 1, 2]);
    let shape = SparseShape::new(1, 4, 4, 4).unwrap();
    let (cached, evicted, rebuilt) = model
        .forward_with_limits(&features, &coords, shape, &encoder, Limits::default())
        .unwrap();
    mlxcel_core::eval(&cached);
    assert_eq!((evicted, rebuilt), (0, 0));
    let limits = Limits {
        neighbors: neighbor_map_bytes(16).unwrap(),
        ..Limits::default()
    };
    let (recomputed, evicted, rebuilt) = model
        .forward_with_limits(&features, &coords, shape, &encoder, limits)
        .unwrap();
    assert!(
        evicted > 0 && rebuilt > 0,
        "test must exercise eviction and up-stage reconstruction"
    );
    assert!(mlxcel_core::item_bool(&mlxcel_core::array_equal(
        &cached,
        &recomputed,
        false
    )));
    assert!(mlxcel_core::item_bool(&mlxcel_core::all_all(
        &mlxcel_core::isfinite(&recomputed)
    )));
}

#[test]
fn moge3_refiner_complete_sparse_phases_fit_supported_grids() {
    use super::super::sparse_budget as phases;
    for (rows, expected_residual) in [(915_712, 481_958_912), (921_600, 484_950_016)] {
        let map = neighbor_map_bytes(rows).unwrap();
        let residual = phases::residual(rows, 32).unwrap();
        assert_eq!(residual + map, expected_residual);
        assert!(phases::public_conv(rows, 32, 32, 4).unwrap() <= MAX_SPARSE_WORKING_BYTES);
        let req = Request {
            phase: "residual",
            level: 0,
            rows,
            channels: 32,
            skips: 0,
            current_map: 0,
            workspace: residual,
            protected: Some(0),
        };
        let admitted = budget::plan(&[map], &req, Limits::default()).unwrap();
        assert!(admitted.evicted.is_empty());
        assert!(admitted.peak <= MAX_SPARSE_WORKING_BYTES);
        let pool = Request {
            phase: "pool",
            protected: None,
            workspace: phases::pool(rows, 32).unwrap(),
            ..req
        };
        assert!(budget::plan(&[map], &pool, Limits::default()).is_ok());
        // Worst-case occupancy S=R is admitted before allocating projection.
        // Its map cache must be evicted; accepting only pool would miss this.
        let project = Request {
            phase: "pool-projection",
            workspace: phases::projection(rows, rows, 32, 64).unwrap(),
            ..pool
        };
        let admitted = budget::plan(&[map], &project, Limits::default()).unwrap();
        assert_eq!(admitted.evicted, [0]);
        assert!(admitted.peak <= MAX_SPARSE_WORKING_BYTES);
        if rows == 921_600 {
            assert_eq!(admitted.peak, 505_036_800);
        }
        // Mandatory retained skips are included even when the map is cached.
        let blocked = Request {
            skips: MAX_SPARSE_WORKING_BYTES,
            ..req
        };
        assert!(budget::plan(&[map], &blocked, Limits::default()).is_err());
    }
    let rows = 2_000_000;
    let map = neighbor_map_bytes(rows).unwrap();
    for workspace in [
        phases::residual(rows, 32).unwrap(),
        phases::pool(rows, 32).unwrap(),
        phases::projection(rows, rows, 32, 64).unwrap(),
    ] {
        assert!(
            budget::plan(
                &[map],
                &Request {
                    phase: "oversized",
                    level: 0,
                    rows,
                    channels: 32,
                    skips: 0,
                    current_map: 0,
                    workspace,
                    protected: Some(0)
                },
                Limits::default()
            )
            .is_err()
        );
    }
    assert!(phases::residual(usize::MAX, 32).is_err());
    assert!(phases::pool(usize::MAX, 32).is_err());
    assert!(phases::projection(1, usize::MAX, 32, 64).is_err());
    assert!(phases::chunk_rows(usize::MAX, 32, 4, 27).is_err());
    assert!(phases::chunk_rows(1 << 24, 32, 4, 27).is_err());
}

#[test]
fn moge3_refiner_conservative_published_pyramid_admission() {
    let shape = SparseShape::new(1, 784, 1168, 256).unwrap();
    let channels = [32, 64, 128, 256, 512];
    let admission = super::policy::admission(915_712, shape, &channels, 1026).unwrap();
    // Conservative grid-volume bounds, not measured model occupancies.
    assert_eq!(admission.rows, [915_712, 915_712, 915_712, 457_856, 57_232]);
    assert_eq!(admission.working, 2_538_353_664);
    assert!(admission.working as u64 <= super::policy::SUPPORTED_PHASE_BYTES);
    let retained = budget::skip_bytes(915_712, 32).unwrap();
    assert_eq!(retained, 135_525_376);
    let map = neighbor_map_bytes(915_712).unwrap();
    let req = Request {
        phase: "residual",
        level: 1,
        rows: 915_712,
        channels: 64,
        skips: retained,
        current_map: 0,
        workspace: super::super::sparse_budget::residual(915_712, 64).unwrap(),
        protected: Some(1),
    };
    assert!(budget::plan(&[map, map], &req, Limits::default()).is_err());
    let admitted = budget::plan(
        &[map, map],
        &req,
        Limits {
            working: admission.working,
            ..Limits::default()
        },
    )
    .unwrap();
    assert_eq!(admitted.evicted, [0]); // still evicts toward the 512MiB target
    assert_eq!(admitted.peak, 969_117_696);
    assert!(super::policy::admission(2_000_000, shape, &channels, 1026).is_err());
    assert!(super::policy::admission(usize::MAX, shape, &channels, 1026).is_err());
    let error = super::policy::admission(915_712, shape, &[32, 4096], 1026)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("supported phase budget"), "{error}");
    assert!(super::policy::admission(0, shape, &channels, 1026).is_err());
}
