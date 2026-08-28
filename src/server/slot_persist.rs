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

//! Slot save/restore persistence for `POST /slots/:id_slot` (issue #1440).
//!
//! b10621 persists a slot's cached tokens plus its KV state through
//! `llama_state_seq_save_file`. mlxcel persists the token stream and the
//! identity needed to validate it (model id, a tokenizer fingerprint, the
//! per-slot context bound) in a versioned JSON envelope; the KV itself is
//! recomputed by the next request's prefill (or adopted from the prompt
//! cache), which is the recorded divergence on the route's manifest entry.
//!
//! Two properties are load-bearing here:
//!
//! - **Confinement.** A save file lives directly under `--slot-save-path`.
//!   The filename is validated with the same rules b10621's
//!   `fs_validate_filename` applies (no separators, no `..`, no control or
//!   lookalike codepoints), and the resolved path is canonicalized and
//!   required to stay inside the canonicalized root, which also refuses a
//!   symlink planted inside the root that points outside it.
//! - **Atomicity.** A save writes to a hidden temporary sibling and renames
//!   it into place, so a crash mid-write can never leave a torn file that a
//!   later restore would half-trust.

use std::path::{Path, PathBuf};

/// Envelope version; bump on any incompatible layout change.
const SLOT_SAVE_VERSION: u32 = 1;
/// Envelope marker distinguishing a slot save from arbitrary JSON.
const SLOT_SAVE_MAGIC: &str = "mlxcel-slot-save";

/// Persisted slot state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SlotSaveFile {
    magic: String,
    version: u32,
    /// Served model id at save time; a restore under a different model is
    /// refused.
    pub model_id: String,
    /// Fingerprint of the tokenizer's behavior (see
    /// [`tokenizer_fingerprint`]); a restore under a tokenizer that encodes
    /// differently is refused even when the model id happens to match.
    pub tokenizer_fingerprint: String,
    /// Cached token stream (prompt + generation), like b10621's
    /// `slot.prompt.tokens`.
    pub tokens: Vec<i32>,
}

/// Why a save/restore file operation was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotPersistError {
    /// Filename failed validation or escaped the storage root: b10621's
    /// "Invalid filename" (400).
    InvalidFilename,
    /// File missing or unreadable on restore (400, like upstream's failed
    /// `llama_state_seq_load_file`).
    Unreadable(String),
    /// Envelope failed validation on restore (400).
    Invalid(String),
    /// Filesystem failure on save (500).
    Io(String),
}

/// b10621 `fs_validate_filename` (subdirectories disallowed), ported rule for
/// rule so the same filenames are accepted and refused on both servers.
pub fn validate_filename(filename: &str) -> bool {
    if filename.is_empty() || filename.len() > 255 {
        return false;
    }
    for c in filename.chars() {
        let code = c as u32;
        if code <= 0x1F
            || code == 0x7F
            || (0x80..=0x9F).contains(&code)
            || code == 0xFF0E // Fullwidth Full Stop
            || code == 0x2215 // Division Slash
            || code == 0x2216 // Set Minus
            || code == 0xFFFD // Replacement Character
            || code == 0xFEFF // Byte Order Mark
            || matches!(c, ':' | '*' | '?' | '"' | '<' | '>' | '|' | '/' | '\\')
        {
            return false;
        }
    }
    if filename.starts_with(' ') || filename.ends_with(' ') || filename.ends_with('.') {
        return false;
    }
    if filename.contains("..") || filename == "." {
        return false;
    }
    true
}

/// Resolve `filename` inside `root`, refusing anything that escapes it.
///
/// `for_read` additionally canonicalizes the full path (following symlinks)
/// and requires the result to stay under the canonical root, so a symlink
/// planted inside the root cannot leak a file from outside it.
fn resolve_in_root(
    root: &Path,
    filename: &str,
    for_read: bool,
) -> Result<PathBuf, SlotPersistError> {
    if !validate_filename(filename) {
        return Err(SlotPersistError::InvalidFilename);
    }
    let canonical_root = root
        .canonicalize()
        .map_err(|e| SlotPersistError::Io(format!("slot save path unavailable: {e}")))?;
    let candidate = canonical_root.join(filename);
    if for_read {
        let resolved = candidate
            .canonicalize()
            .map_err(|e| SlotPersistError::Unreadable(format!("cannot open slot file: {e}")))?;
        if !resolved.starts_with(&canonical_root) {
            return Err(SlotPersistError::InvalidFilename);
        }
        return Ok(resolved);
    }
    // For writes the file may not exist yet; the validated filename has no
    // separators, so joining cannot leave the root. Refuse replacing a
    // symlink in place, which would follow it out of the root on write.
    if candidate
        .symlink_metadata()
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
    {
        return Err(SlotPersistError::InvalidFilename);
    }
    Ok(candidate)
}

/// A stable fingerprint of tokenizer behavior: the token ids of a fixed probe
/// string, hashed. Two tokenizers that encode the probe identically are
/// treated as compatible for restoring a token stream.
pub fn tokenizer_fingerprint(tokenizer: &crate::tokenizer::MlxcelTokenizer) -> String {
    use std::hash::{Hash, Hasher};
    let probe = "mlxcel slot-save probe: The quick brown fox jumps over 13 lazy dogs. \
                 다람쥐 헌 쳇바퀴에 타고파 1234567890";
    let ids = tokenizer.encode(probe, false).unwrap_or_default();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    ids.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Atomically persist `tokens` as `filename` under `root`.
///
/// Returns the number of bytes written (b10621 `n_written`).
pub fn save(
    root: &Path,
    filename: &str,
    model_id: &str,
    tokenizer_fingerprint: &str,
    tokens: &[i32],
) -> Result<usize, SlotPersistError> {
    let target = resolve_in_root(root, filename, false)?;
    let envelope = SlotSaveFile {
        magic: SLOT_SAVE_MAGIC.to_string(),
        version: SLOT_SAVE_VERSION,
        model_id: model_id.to_string(),
        tokenizer_fingerprint: tokenizer_fingerprint.to_string(),
        tokens: tokens.to_vec(),
    };
    let bytes = serde_json::to_vec(&envelope)
        .map_err(|e| SlotPersistError::Io(format!("cannot serialize slot state: {e}")))?;
    let tmp = target.with_file_name(format!(
        ".{}.tmp-{}",
        target
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "slot".to_string()),
        std::process::id()
    ));
    let write_result = (|| -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        std::fs::rename(&tmp, &target)
    })();
    if let Err(e) = write_result {
        let _ = std::fs::remove_file(&tmp);
        return Err(SlotPersistError::Io(format!("unable to save slot: {e}")));
    }
    Ok(bytes.len())
}

/// Load and validate a slot save file.
///
/// Returns the envelope and the number of bytes read (b10621 `n_read`). The
/// caller still applies the context-window bound, which is config-dependent.
pub fn load(
    root: &Path,
    filename: &str,
    model_id: &str,
    tokenizer_fingerprint: &str,
) -> Result<(SlotSaveFile, usize), SlotPersistError> {
    let path = resolve_in_root(root, filename, true)?;
    let bytes = std::fs::read(&path)
        .map_err(|e| SlotPersistError::Unreadable(format!("cannot read slot file: {e}")))?;
    let n_read = bytes.len();
    let envelope: SlotSaveFile = serde_json::from_slice(&bytes)
        .map_err(|e| SlotPersistError::Invalid(format!("invalid slot save file: {e}")))?;
    if envelope.magic != SLOT_SAVE_MAGIC {
        return Err(SlotPersistError::Invalid(
            "invalid slot save file: wrong magic".to_string(),
        ));
    }
    if envelope.version != SLOT_SAVE_VERSION {
        return Err(SlotPersistError::Invalid(format!(
            "unsupported slot save version {}",
            envelope.version
        )));
    }
    if envelope.model_id != model_id {
        return Err(SlotPersistError::Invalid(format!(
            "slot save file was created by model '{}', this server serves '{}'",
            envelope.model_id, model_id
        )));
    }
    if envelope.tokenizer_fingerprint != tokenizer_fingerprint {
        return Err(SlotPersistError::Invalid(
            "slot save file was created with an incompatible tokenizer".to_string(),
        ));
    }
    Ok((envelope, n_read))
}

#[cfg(test)]
#[path = "slot_persist_tests.rs"]
mod slot_persist_tests;
