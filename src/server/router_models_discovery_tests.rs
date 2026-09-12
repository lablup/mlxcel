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

use std::path::PathBuf;

use super::discover_models;

fn temp_models_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mlxcel-router-discovery-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create models dir");
    dir
}

#[cfg(unix)]
#[test]
fn discovery_rejects_model_whose_config_is_symlink_evidence() {
    let root = temp_models_dir("config-symlink-root");
    let outside = temp_models_dir("config-symlink-outside");
    let escaped_config = outside.join("config.json");
    std::fs::write(&escaped_config, r#"{"model_type":"qwen3"}"#).unwrap();
    let candidate = root.join("escape-config");
    std::fs::create_dir_all(&candidate).unwrap();
    std::os::unix::fs::symlink(&escaped_config, candidate.join("config.json")).unwrap();

    let legitimate = root.join("legit");
    std::fs::create_dir_all(&legitimate).unwrap();
    std::fs::write(legitimate.join("config.json"), "{}").unwrap();

    let found = discover_models(&root).expect("scan");
    assert_eq!(found.keys().cloned().collect::<Vec<_>>(), vec!["legit"]);
}

#[cfg(unix)]
#[test]
fn router_sources_reject_symlinked_config_evidence_consistently() {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use crate::downloader::DownloadHooks;
    use crate::server::ServerStartupConfig;
    use crate::server::router_cache::{CacheSource, RouterDownloader};
    use crate::server::router_presets::{PresetCliOverrides, PresetSection, RouterPresets};

    struct NoopDownloader;
    impl RouterDownloader for NoopDownloader {
        fn validate(&self, _repo_id: &str) -> anyhow::Result<()> {
            Ok(())
        }

        fn download(
            &self,
            _repo_id: &str,
            _dest_root: &std::path::Path,
            _hooks: DownloadHooks,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn symlink_config_snapshot(root: &std::path::Path, name: &str) -> PathBuf {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let outside = root.join(format!("{name}-outside-config.json"));
        std::fs::write(&outside, r#"{"model_type":"qwen3"}"#).unwrap();
        std::os::unix::fs::symlink(&outside, dir.join("config.json")).unwrap();
        dir
    }

    let cache_root = temp_models_dir("source-cache-symlink");
    symlink_config_snapshot(&cache_root, "cached");
    let cache = CacheSource::new(cache_root.clone(), Arc::new(NoopDownloader));
    assert!(
        cache.list().is_empty(),
        "cache source accepted symlinked config"
    );

    let direct = symlink_config_snapshot(&temp_models_dir("source-preset-direct"), "direct");
    let repo_path = cache.snapshot_dir("owner/model");
    std::fs::create_dir_all(&repo_path).unwrap();
    let outside = cache_root.join("repo-outside-config.json");
    std::fs::write(&outside, r#"{"model_type":"qwen3"}"#).unwrap();
    std::os::unix::fs::symlink(&outside, repo_path.join("config.json")).unwrap();

    let mut models = BTreeMap::new();
    models.insert(
        "direct".to_string(),
        PresetSection {
            model_path: Some(direct),
            ..Default::default()
        },
    );
    models.insert(
        "repo".to_string(),
        PresetSection {
            hf_repo: Some("owner/model".to_string()),
            ..Default::default()
        },
    );
    let pool = super::RouterPool::new(
        super::RouterSources {
            models_dir: None,
            cache: Some(cache),
            presets: RouterPresets {
                global: PresetSection::default(),
                models,
            },
        },
        ServerStartupConfig::default(),
        Default::default(),
        PresetCliOverrides::default(),
        4,
        false,
    )
    .expect("pool");
    assert!(
        pool.catalog_snapshot().is_empty(),
        "preset source accepted symlinked config evidence"
    );
}
