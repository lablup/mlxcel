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

//! Variant resolution, sizes and request options against upstream's
//! `config.py` and `pipeline.py`.

use serde_json::json;

use super::*;

fn dir(name: &str) -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(root.path().join(name)).unwrap();
    root
}

#[test]
fn the_six_checkpoint_directories_resolve_to_their_variants() {
    let cases = [
        ("mage-flow", "mage-flow", ImageTask::Generate, 20, 5.0),
        (
            "mage-flow-base",
            "mage-flow-base",
            ImageTask::Generate,
            30,
            5.0,
        ),
        (
            "mage-flow-turbo",
            "mage-flow-turbo",
            ImageTask::Generate,
            4,
            1.0,
        ),
        ("mage-flow-edit", "mage-flow-edit", ImageTask::Edit, 30, 5.0),
        (
            "mage-flow-edit-base",
            "mage-flow-edit-base",
            ImageTask::Edit,
            30,
            5.0,
        ),
        (
            "mage-flow-edit-turbo",
            "mage-flow-edit-turbo",
            ImageTask::Edit,
            4,
            1.0,
        ),
        // Hugging Face repo spelling and the cache layout.
        (
            "Mage-Flow-Edit-Turbo",
            "mage-flow-edit-turbo",
            ImageTask::Edit,
            4,
            1.0,
        ),
        (
            "models--mage-flow-community--Mage-Flow-Base/snapshots/0123abcd",
            "mage-flow-base",
            ImageTask::Generate,
            30,
            5.0,
        ),
    ];
    for (name, variant, task, steps, guidance) in cases {
        let root = dir(name);
        let path = root.path().join(name);
        if name == "mage-flow" {
            // Only the model index names the plain variant.
            std::fs::write(
                path.join("model_index.json"),
                r#"{"_class_name": "MageFlowPipeline"}"#,
            )
            .unwrap();
        }
        let got = variant_from_local_path(&path).unwrap();
        assert_eq!(got.name, variant, "{name}");
        assert_eq!(got.task, task, "{name}");
        assert_eq!(got.defaults().steps, steps, "{name}");
        assert_eq!(got.defaults().guidance, guidance, "{name}");
    }
}

#[test]
fn a_parent_directory_name_does_not_pick_the_variant() {
    // Upstream matches the whole path, so `database` would read as `-base`.
    let root = dir("database/mage-flow-turbo");
    let got = variant_from_local_path(&root.path().join("database/mage-flow-turbo")).unwrap();
    assert_eq!(got.name, "mage-flow-turbo");
    // The converter's metadata wins over the directory name.
    let root = dir("mage-flow-turbo");
    let path = root.path().join("mage-flow-turbo");
    std::fs::write(
        path.join("mlx_mage_flow.json"),
        r#"{"variant": "mage-flow-edit"}"#,
    )
    .unwrap();
    assert_eq!(
        variant_from_local_path(&path).unwrap().name,
        "mage-flow-edit"
    );
    // Nothing to go on.
    let root = dir("checkpoint");
    let err = variant_from_local_path(&root.path().join("checkpoint"))
        .unwrap_err()
        .to_string();
    assert!(err.contains("Could not infer a Mage-Flow variant"), "{err}");
}

#[test]
fn get_variant_accepts_names_repo_ids_and_aliases() {
    assert_eq!(
        get_variant("Mage-Flow-Turbo").unwrap().name,
        "mage-flow-turbo"
    );
    assert_eq!(
        get_variant("mage-flow-community/Mage-Flow-Edit")
            .unwrap()
            .name,
        "mage-flow-edit"
    );
    assert_eq!(
        get_variant("microsoft/Mage-Flow-Edit-Base").unwrap().name,
        "mage-flow-edit-base"
    );
    assert!(get_variant("mage-flow-xl").is_err());
}

#[test]
fn wrong_task_errors_name_the_variant() {
    let edit = get_variant("mage-flow-edit-turbo").unwrap();
    let err = edit.require(ImageTask::Generate).unwrap_err().to_string();
    assert!(
        err.contains("Mage-Flow-Edit-Turbo is an image-edit checkpoint"),
        "{err}"
    );
    assert!(edit.require(ImageTask::Edit).is_ok());
    let gen_variant = get_variant("mage-flow").unwrap();
    let err = gen_variant
        .require(ImageTask::Edit)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("Mage-Flow is a text-to-image checkpoint"),
        "{err}"
    );
}

#[test]
fn dimensions_follow_upstream_bounds() {
    assert!(validate_dimensions(512, 512).is_ok());
    assert!(validate_dimensions(2048, 512).is_ok());
    assert!(validate_dimensions(496, 512).is_err());
    assert!(validate_dimensions(2064, 1024).is_err());
    let err = validate_dimensions(520, 1024).unwrap_err().to_string();
    assert!(err.contains("multiple of 16"), "{err}");
}

#[test]
fn edit_dimensions_scale_the_first_reference() {
    // Long edge kept, short edge scaled and rounded down to 16.
    assert_eq!(edit_dimensions(1000, 750, None), (992, 736));
    assert_eq!(edit_dimensions(750, 1000, None), (736, 992));
    // `max_size` sets the long edge.
    assert_eq!(edit_dimensions(1000, 750, Some(1024)), (1024, 768));
    // 600 * 1000 / 1600 = 375; both sides round down to multiples of 16.
    assert_eq!(edit_dimensions(1600, 600, Some(1000)), (992, 368));
    assert_eq!(default_edit_size(512, 512), (512, 512));
}

#[test]
fn transformer_config_reads_the_published_file() {
    let config = TransformerConfig::from_json(&json!({
        "in_channels": 128, "out_channels": 128, "context_in_dim": 2560,
        "hidden_size": 3072, "num_heads": 24, "depth": 12,
        "axes_dim": [16, 56, 56], "theta": 10000, "_class_name": "MageFlow"
    }))
    .unwrap();
    assert_eq!(config, TransformerConfig::default());
    assert_eq!(config.head_dim(), 128);
    let err = TransformerConfig::from_json(&json!({"axes_dim": [16, 56, 54]}))
        .unwrap_err()
        .to_string();
    assert!(err.contains("axes_dim must sum to head_dim"), "{err}");
    assert!(TransformerConfig::from_json(&json!({"num_heads": 0})).is_err());
}

fn extra(v: serde_json::Value) -> Map<String, Value> {
    v.as_object().unwrap().clone()
}

#[test]
fn options_default_and_parse_like_upstream() {
    let opts = MageFlowOptions::from_extra(&Map::new(), ImageTask::Edit).unwrap();
    assert_eq!(opts.static_shift, 6.0);
    assert!(!opts.renormalization);
    assert_eq!(opts.max_size, None);
    assert_eq!(opts.vl_cond_long_edge, Some(384));
    assert!(opts.sample_posterior);
    assert_eq!(opts.negative_prompt, None);

    let opts = MageFlowOptions::from_extra(
        &extra(json!({
            "static_shift": 3.0, "renormalization": true, "max_size": 1024,
            "vl_cond_long_edge": 512, "sample_posterior": false, "negative_prompt": "blurry"
        })),
        ImageTask::Edit,
    )
    .unwrap();
    assert_eq!(opts.static_shift, 3.0);
    assert!(opts.renormalization);
    assert_eq!(opts.max_size, Some(1024));
    assert_eq!(opts.vl_cond_long_edge, Some(512));
    assert!(!opts.sample_posterior);
    assert_eq!(opts.negative_prompt.as_deref(), Some("blurry"));

    // `null` and `<= 0` keep the references at their own size.
    for v in [json!(null), json!(0), json!(-1)] {
        let opts =
            MageFlowOptions::from_extra(&extra(json!({"vl_cond_long_edge": v})), ImageTask::Edit)
                .unwrap();
        assert_eq!(opts.vl_cond_long_edge, None);
    }
}

#[test]
fn bad_options_are_rejected() {
    for bad in [
        json!({"static_shift": 0}),
        json!({"static_shift": "6"}),
        json!({"renormalization": 1}),
        json!({"negative_prompt": 3}),
        json!({"max_size": 0}),
        json!({"vl_cond_long_edge": 1.5}),
        json!({"vl_cond_long_edge": 2049}),
    ] {
        assert!(
            MageFlowOptions::from_extra(&extra(bad.clone()), ImageTask::Edit).is_err(),
            "{bad}"
        );
    }
    // Edit-only keys on a generate request.
    for key in ["max_size", "vl_cond_long_edge", "sample_posterior"] {
        let err = MageFlowOptions::from_extra(&extra(json!({key: 512})), ImageTask::Generate)
            .unwrap_err()
            .to_string();
        assert!(err.contains("edits only"), "{err}");
    }
}

#[test]
fn vl_cond_long_edge_is_capped_at_the_output_ceiling() {
    assert_eq!(MAX_VL_COND_LONG_EDGE, MAX_SIDE);
    assert_eq!(MAX_VL_COND_PIXELS, 2048 * 2048);
    let opts = MageFlowOptions::from_extra(
        &extra(json!({"vl_cond_long_edge": MAX_VL_COND_LONG_EDGE})),
        ImageTask::Edit,
    )
    .unwrap();
    assert_eq!(opts.vl_cond_long_edge, Some(MAX_VL_COND_LONG_EDGE));
    let err =
        MageFlowOptions::from_extra(&extra(json!({"vl_cond_long_edge": 8192})), ImageTask::Edit)
            .unwrap_err()
            .to_string();
    assert!(err.contains("at most 2048"), "{err}");
}
