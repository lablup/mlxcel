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

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;

use mlxcel_core::cache::{BatchKvQuantConfig, KVCacheMode, KvQuantScheme};

use super::{
    ServerStartupConfig, detect_model_media_support, validate_muse_glimmer_unsupported_startup,
};

fn muse_model_dir() -> tempfile::TempDir {
    let dir = match tempfile::tempdir() {
        Ok(dir) => dir,
        Err(err) => panic!("failed to create temp model dir: {err}"),
    };
    if let Err(err) = std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"muse_glimmer"}"#,
    ) {
        panic!("failed to write Muse Glimmer config: {err}");
    }
    dir
}

/// A drafter directory whose `config.json` declares `model_type`.
fn drafter_dir(model_type: &str) -> tempfile::TempDir {
    let dir = match tempfile::tempdir() {
        Ok(dir) => dir,
        Err(err) => panic!("failed to create temp drafter dir: {err}"),
    };
    if let Err(err) = std::fs::write(
        dir.path().join("config.json"),
        format!(
            r#"{{"model_type":"{model_type}","block_size":16,"mask_token_id":201818,"target_layer_ids":[1,13,25,37,49]}}"#
        ),
    ) {
        panic!("failed to write drafter config: {err}");
    }
    dir
}

fn startup_for(path: &Path) -> ServerStartupConfig {
    ServerStartupConfig {
        model_path: path.to_path_buf(),
        ..ServerStartupConfig::default()
    }
}

fn assert_rejected<F>(mutate: F, expected: &str)
where
    F: FnOnce(&mut ServerStartupConfig),
{
    let model_dir = muse_model_dir();
    let mut startup = startup_for(model_dir.path());
    mutate(&mut startup);
    let err = match validate_muse_glimmer_unsupported_startup(&startup) {
        Ok(()) => panic!("Muse Glimmer startup unexpectedly accepted unsupported option"),
        Err(err) => err.to_string(),
    };
    assert!(
        err.contains(expected),
        "expected error to contain {expected:?}, got {err:?}"
    );
}

#[test]
fn muse_glimmer_startup_allows_baseline_and_keeps_video_disabled() {
    // `validate_muse_glimmer_unsupported_startup` reads `MLXCEL_BACKEND`, which
    // `muse_glimmer_startup_rejects_xla_backend_selection` in this module
    // mutates process-wide. Acquire the crate-wide env lock so this test
    // never observes that sibling's transient `MLXCEL_BACKEND=xla` value
    // under default parallel test execution.
    let _env_guard = crate::test_support::env_lock::env_lock();

    let model_dir = muse_model_dir();
    let startup = startup_for(model_dir.path());

    if let Err(err) = validate_muse_glimmer_unsupported_startup(&startup) {
        panic!("baseline Muse Glimmer startup should be accepted: {err}");
    }

    let media = detect_model_media_support(model_dir.path());
    assert!(
        !media.video(),
        "Muse Glimmer must not advertise video support"
    );
}

#[test]
fn muse_glimmer_startup_rejects_adapters_and_orphan_draft_flags() {
    assert_rejected(
        |startup| startup.adapter_path = Some("adapter.safetensors".into()),
        "LoRA/adapters",
    );
    // A draft kind or block size without a drafter is an operator mistake
    // the old blanket rejection also caught; it stays rejected.
    assert_rejected(
        |startup| startup.draft_kind = Some("dflash".to_string()),
        "only together with --draft-model",
    );
    assert_rejected(
        |startup| startup.draft_block_size = Some(8),
        "only together with --draft-model",
    );
    // A drafter directory with no config at all is not the assistant.
    assert_rejected(
        |startup| startup.draft_model_path = Some("draft".into()),
        "muse_glimmer_assistant",
    );
}

/// The one speculative pairing the family supports (issue #1343): the
/// Muse Glimmer assistant drafter on the DFlash round loop, with or without
/// an explicit `--draft-kind dflash` and `--draft-block-size`.
#[test]
fn muse_glimmer_accepts_assistant_drafter() {
    let _env_guard = crate::test_support::env_lock::env_lock();
    let model_dir = muse_model_dir();
    let drafter = drafter_dir("muse_glimmer_assistant");

    let mut startup = startup_for(model_dir.path());
    startup.draft_model_path = Some(drafter.path().to_path_buf());
    if let Err(err) = validate_muse_glimmer_unsupported_startup(&startup) {
        panic!("the Muse Glimmer assistant drafter must be accepted: {err}");
    }

    startup.draft_kind = Some("dflash".to_string());
    startup.draft_block_size = Some(8);
    if let Err(err) = validate_muse_glimmer_unsupported_startup(&startup) {
        panic!("an explicit dflash kind and block size must be accepted: {err}");
    }
}

/// Any other drafter is refused by name before the server loads anything:
/// an MTP assistant, a Qwen 3.5 DFlash drafter, an LFM2 DSpark drafter, or
/// the assistant under a kind the family cannot run.
#[test]
fn muse_glimmer_rejects_non_assistant_drafter() {
    let gemma = drafter_dir("gemma4_assistant");
    assert_rejected(
        |startup| startup.draft_model_path = Some(gemma.path().to_path_buf()),
        "muse_glimmer_assistant",
    );
    let qwen_dflash = drafter_dir("qwen3");
    assert_rejected(
        |startup| startup.draft_model_path = Some(qwen_dflash.path().to_path_buf()),
        "muse_glimmer_assistant",
    );
    let dspark = drafter_dir("lfm2");
    assert_rejected(
        |startup| startup.draft_model_path = Some(dspark.path().to_path_buf()),
        "muse_glimmer_assistant",
    );
    let assistant = drafter_dir("muse_glimmer_assistant");
    assert_rejected(
        |startup| {
            startup.draft_model_path = Some(assistant.path().to_path_buf());
            startup.draft_kind = Some("mtp".to_string());
        },
        "DFlash round loop only",
    );
}

#[test]
fn muse_glimmer_startup_rejects_quantized_kv_modes() {
    assert_rejected(
        |startup| startup.kv_cache_mode = KVCacheMode::Int8,
        "INT8/Turbo KV",
    );
    assert_rejected(
        |startup| startup.kv_cache_mode = KVCacheMode::Turbo4Asym,
        "INT8/Turbo KV",
    );
    assert_rejected(
        |startup| {
            startup.batch_kv_quant = BatchKvQuantConfig {
                scheme: KvQuantScheme::TurboQuant,
                bits: 4,
                group_size: 64,
                skip_last_layer: true,
            };
        },
        "batch KV quantization",
    );
}

#[test]
fn muse_glimmer_startup_rejects_parallel_and_distributed_modes() {
    assert_rejected(|startup| startup.tp_size = 2, "tensor-parallel");
    assert_rejected(
        |startup| startup.pp_layers = Some("0-24,25-49".to_string()),
        "pipeline-parallel",
    );
    assert_rejected(|startup| startup.pp_auto = Some(2), "pipeline-parallel");
    assert_rejected(
        |startup| startup.enable_elastic_pp = true,
        "pipeline-parallel",
    );
    assert_rejected(
        |startup| startup.node_role = Some("prefill".to_string()),
        "distributed or disaggregated",
    );
    assert_rejected(
        |startup| {
            startup.prefill_peers = vec![SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9000)];
        },
        "distributed or disaggregated",
    );
}

#[test]
fn muse_glimmer_startup_rejects_xla_backend_selection() {
    let _env_guard = crate::test_support::env_lock::env_lock();
    let previous = std::env::var_os("MLXCEL_BACKEND");
    // SAFETY: the crate-wide env lock serializes this test with other tests
    // that mutate process environment variables.
    unsafe {
        std::env::set_var("MLXCEL_BACKEND", "xla");
    }

    assert_rejected(|_| {}, "XLA/IREE/OpenXLA");

    // SAFETY: still protected by the env lock above.
    unsafe {
        match previous {
            Some(value) => std::env::set_var("MLXCEL_BACKEND", value),
            None => std::env::remove_var("MLXCEL_BACKEND"),
        }
    }
}
