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

//! Shard file names read from an untrusted `model.safetensors.index.json`.
//!
//! Shared by the IREE loader and the emitter-side recurrent checkpoint
//! cross-check (issue #3338), which is built without the `iree` feature in
//! unit tests, so every reader of an index resolves shard names one way.

use std::path::{Path, PathBuf};

/// Resolve one shard file name from `model.safetensors.index.json` inside
/// `model_dir`. The index comes from an untrusted checkpoint, so a name must be
/// a single plain file name: an absolute path, a `..` component or a nested
/// path would let it point the loader at a file outside the model directory.
/// This matches `validate_index_shards` in `mlxcel-core`.
pub(crate) fn index_shard_path(
    model_dir: &Path,
    index: &Path,
    file: &str,
) -> Result<PathBuf, String> {
    let relative = Path::new(file);
    let plain = !file.contains('\0')
        && matches!(
            relative.components().collect::<Vec<_>>().as_slice(),
            [std::path::Component::Normal(_)]
        );
    if !plain {
        return Err(format!(
            "{} names shard {file:?}; shard names must be plain file names inside the model \
             directory",
            index.display()
        ));
    }
    Ok(model_dir.join(relative))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_plain_file_name_resolves() {
        let dir = Path::new("/models/m");
        let index = dir.join("model.safetensors.index.json");
        let ok = index_shard_path(dir, &index, "model-00001.safetensors").expect("plain");
        assert_eq!(ok, dir.join("model-00001.safetensors"));
        for bad in ["", "..", "/", "/etc/passwd", "a/b", "../x", "x\0y"] {
            assert!(index_shard_path(dir, &index, bad).is_err(), "{bad:?}");
        }
    }
}
