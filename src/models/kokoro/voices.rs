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

//! Voice-pack loading for Kokoro.
//!
//! Each voice is a `voices/<name>.safetensors` pack with a single `voice`
//! tensor of shape `(N, 1, 256)`. The row selected for a request is indexed by
//! the unpadded phoneme count minus one (`pack[len(tokens) - 1]`), giving a
//! `(1, 256)` style row split into a 128-d predictor half and a 128-d decoder
//! half. The redundant `.pt` copies are ignored.

use std::path::{Path, PathBuf};

use mlxcel_core::{MlxArray, UniquePtr};

use super::ops;

/// The default voice when a request omits `voice` or names an unknown one.
pub(crate) const DEFAULT_VOICE: &str = "af_heart";

/// Resolve the `voices` directory inside a Kokoro checkpoint.
pub(crate) fn voices_dir(model_path: &Path) -> PathBuf {
    model_path.join("voices")
}

/// List the available voice names (the `.safetensors` stems), sorted.
pub(crate) fn list_voices(model_path: &Path) -> Vec<String> {
    let dir = voices_dir(model_path);
    let mut names = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("safetensors")
                && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
            {
                names.push(stem.to_string());
            }
        }
    }
    names.sort();
    names
}

/// Validate a requested voice name against the available packs, falling back to
/// [`DEFAULT_VOICE`] when the request is empty or names an unknown voice.
///
/// Returns the resolved name. Names are restricted to a safe character set so a
/// request can never escape the `voices` directory.
pub(crate) fn resolve_voice(model_path: &Path, requested: Option<&str>) -> String {
    let available = list_voices(model_path);
    let pick = requested
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter(|s| is_safe_name(s))
        .map(str::to_string);

    match pick {
        Some(name) if available.iter().any(|v| v == &name) => name,
        _ => DEFAULT_VOICE.to_string(),
    }
}

/// Whether a voice name contains only `[A-Za-z0-9_-]` (no path separators).
fn is_safe_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// A loaded voice pack: the `(N, 1, 256)` style table.
pub(crate) struct VoicePack {
    table: UniquePtr<MlxArray>,
    rows: i32,
}

impl VoicePack {
    /// Load `voices/<name>.safetensors` from a checkpoint directory.
    pub(crate) fn load(model_path: &Path, name: &str) -> Result<Self, String> {
        if !is_safe_name(name) {
            return Err(format!("kokoro: unsafe voice name '{name}'"));
        }
        let path = voices_dir(model_path).join(format!("{name}.safetensors"));
        let map = mlxcel_core::weights::load_safetensors(&path)
            .map_err(|e| format!("kokoro: failed to load voice '{name}': {e}"))?;
        let table = map
            .get("voice")
            .map(|t| mlxcel_core::copy(ops::r(t)))
            .ok_or_else(|| format!("kokoro: voice pack '{name}' missing 'voice' tensor"))?;
        let shape = ops::shape(&table);
        let rows = *shape.first().unwrap_or(&0);
        Ok(Self { table, rows })
    }

    /// Style row for a phoneme sequence of unpadded length `n_tokens`, indexed
    /// by `n_tokens - 1` and clamped to the table bounds. Returns `(1, 256)`.
    pub(crate) fn row(&self, n_tokens: usize) -> UniquePtr<MlxArray> {
        let idx = (n_tokens.saturating_sub(1) as i32).clamp(0, self.rows.saturating_sub(1));
        // table is (N, 1, 256); slice row idx -> (1,1,256) -> reshape (1,256).
        let sliced = ops::slice(&self.table, &[idx, 0, 0], &[idx + 1, i32::MAX, i32::MAX]);
        ops::reshape(&sliced, &[1, 256])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_name_rejects_traversal() {
        assert!(is_safe_name("af_heart"));
        assert!(is_safe_name("zm_yunyang"));
        assert!(!is_safe_name("../secret"));
        assert!(!is_safe_name("a/b"));
        assert!(!is_safe_name(""));
        assert!(!is_safe_name("a.b"));
    }

    #[test]
    fn resolve_voice_falls_back_to_default_when_no_voices() {
        // With no voices directory the list is empty; any request falls back.
        let dir = std::env::temp_dir().join("kokoro_test_empty");
        let _ = std::fs::create_dir_all(&dir);
        assert_eq!(
            resolve_voice(&dir, Some("af_heart")),
            DEFAULT_VOICE,
            "falls back when voices/ dir is absent or empty"
        );
        assert_eq!(
            resolve_voice(&dir, None),
            DEFAULT_VOICE,
            "falls back when voice is None"
        );
        assert_eq!(
            resolve_voice(&dir, Some("../escape")),
            DEFAULT_VOICE,
            "unsafe name falls back"
        );
    }

    #[test]
    fn resolve_voice_picks_known_voice_from_tmpdir() {
        // Create a minimal voices/ directory with a fake safetensors file.
        let root = std::env::temp_dir().join("kokoro_test_voices");
        let voices = root.join("voices");
        let _ = std::fs::create_dir_all(&voices);
        std::fs::write(voices.join("bf_emma.safetensors"), b"fake")
            .expect("create test voice file");

        // bf_emma should resolve to itself; unknown names fall back.
        assert_eq!(resolve_voice(&root, Some("bf_emma")), "bf_emma");
        assert_eq!(resolve_voice(&root, Some("unknown")), DEFAULT_VOICE);
        assert_eq!(resolve_voice(&root, Some("")), DEFAULT_VOICE);
        assert_eq!(resolve_voice(&root, Some("   ")), DEFAULT_VOICE);

        // Clean up (best-effort).
        let _ = std::fs::remove_file(voices.join("bf_emma.safetensors"));
    }
}
