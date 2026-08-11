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
    let model_dir = muse_model_dir();
    let startup = startup_for(model_dir.path());

    if let Err(err) = validate_muse_glimmer_unsupported_startup(&startup) {
        panic!("baseline Muse Glimmer startup should be accepted: {err}");
    }

    let media = detect_model_media_support(model_dir.path());
    assert!(
        !media.video,
        "Muse Glimmer must not advertise video support"
    );
}

#[test]
fn muse_glimmer_startup_rejects_adapters_and_speculative() {
    assert_rejected(
        |startup| startup.adapter_path = Some("adapter.safetensors".into()),
        "LoRA/adapters",
    );
    assert_rejected(
        |startup| startup.draft_model_path = Some("draft".into()),
        "speculative decoding or DFlash",
    );
    assert_rejected(
        |startup| startup.draft_kind = Some("dflash".to_string()),
        "speculative decoding or DFlash",
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
