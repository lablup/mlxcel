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

//! Prompt-cache snapshots for `qwen4_exp` (issue #2885).
//!
//! One [`ModelStateSnapshot`] tagged [`SNAPSHOT_FAMILY`] holds a sequence's
//! whole per-sequence state, every tensor at its stored dtype:
//!
//! | Key under `layer{i}.` | State |
//! |---|---|
//! | `kind` | linear or attention, checked against the model's layer |
//! | `conv_state`, `state_cache` | `GatedDeltaCache` |
//! | `ple.conv_state` | the PLE dilated convolution window |
//! | `ple.contexts` | the host-side n-gram contexts, see [`encode_contexts`] |
//! | `kv.keys`, `kv.values`, `kv.offset`, `kv.mode` | the QSA layer's `KVCache` |
//! | `qsa.raw_keys`, `qsa.positions`, `qsa.block_keys` | `QsaIndexCache` |
//!
//! The n-gram row table is never captured: a snapshot is per-sequence state,
//! so a restore works the same over the resident table and the external
//! (`ple_storage`) one.
//!
//! ## Speculative state
//!
//! A snapshot describes the state the sequence holds for exactly `token_len`
//! tokens. A verify block that has run but not yet been rewound leaves
//! positions past that length, so a capture whose attention offset differs
//! from `token_len` is refused with a named reason rather than stored under a
//! prefix it does not describe. The rollback log is deliberately not part of
//! a snapshot: the log unwinds into the forward that just ran on this
//! sequence, and a restored sequence has run none, so
//! [`Qwen4ExpModel::restore_sequence_snapshot`] installs an empty log and a
//! rewind past the restore point is refused instead of replayed against a
//! capture of some other state.
//!
//! ## Image and video requests
//!
//! Reuse is declined for them: [`Qwen4ExpModel::sequence_snapshot`] returns
//! `None` with a named reason while the sequence carries image or video MRoPE
//! state. Text-only sequences have no such state, and the whole conversation
//! suffix after a text-only restore takes the ordinary text positions. A
//! request with media therefore always prefills cold, the default behavior
//! without `--enable-vlm-prefix-cache`, and with the flag it finds no stored
//! entry to adopt.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::ModelStateSnapshot;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{Qwen4ExpCache, Qwen4ExpModel};
use crate::models::gated_delta::GatedDeltaCache;
use crate::models::kv_snapshot::{KvSnapshotNames, restore_standard, snapshot_standard};
use crate::models::recurrent_snapshot::{push_i32, push_optional, restore_i32, restore_optional};

/// Family tag of every snapshot this module writes.
pub(crate) const SNAPSHOT_FAMILY: &str = "qwen4_exp";

/// Tensor-name vocabulary for the attention layers' KV cache. This family has
/// no rotating KV cache, so the second name is unused.
const KV_NAMES: KvSnapshotNames = KvSnapshotNames::new("Qwen4-Exp", "kv", "rotating");

const LAYER_KIND_LINEAR: i32 = 0;
const LAYER_KIND_ATTENTION: i32 = 1;

/// Which pieces of state a capture stores.
///
/// Production always stores all of them ([`SnapshotParts::ALL`]). The
/// snapshot tests switch one piece off. Without the n-gram contexts the
/// cold-prefill comparison fails: the text stays fluent and differs. Without
/// the block keys the logits still match, because the next selection
/// recomputes them from the stored raw keys and positions; they are stored
/// anyway so a restore holds exactly the state of a cold run and does not pay
/// for that recomputation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SnapshotParts {
    /// The PLE n-gram contexts; when off the capture stores none.
    pub(crate) ple_contexts: bool,
    /// The QSA pooled block keys; when off the capture leaves them out.
    pub(crate) qsa_block_keys: bool,
}

impl SnapshotParts {
    pub(crate) const ALL: Self = Self {
        ple_contexts: true,
        qsa_block_keys: true,
    };
}

/// The contexts as one int32 tensor `[rows, len_0, tokens_0.., len_1, ..]`.
///
/// A single flat tensor keeps rows of different lengths and the empty state
/// representable without a zero-sized array, and the container already copes
/// with it. An empty cache encodes as `[0]`.
fn encode_contexts(contexts: &[Vec<i64>]) -> Result<UniquePtr<MlxArray>, String> {
    let mut encoded =
        Vec::with_capacity(1 + contexts.len() + contexts.iter().map(Vec::len).sum::<usize>());
    encoded.push(contexts.len() as i32);
    for row in contexts {
        encoded.push(row.len() as i32);
        for &token in row {
            let token = i32::try_from(token)
                .map_err(|_| format!("n-gram context token {token} does not fit int32"))?;
            encoded.push(token);
        }
    }
    Ok(mlxcel_core::from_slice_i32(
        &encoded,
        &[encoded.len() as i32],
    ))
}

/// Inverse of [`encode_contexts`]; `Err` names what is malformed.
fn decode_contexts(array: &MlxArray) -> Result<Vec<Vec<i64>>, String> {
    let array = mlxcel_core::astype(array, mlxcel_core::dtype::INT32);
    mlxcel_core::eval(&array);
    let flat: Vec<i32> = mlxcel_core::array_to_raw_bytes(&array)
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("4-byte chunk")))
        .collect();
    let mut values = flat.iter().copied();
    let rows = values
        .next()
        .ok_or_else(|| "empty n-gram context tensor".to_string())?;
    let mut contexts = Vec::with_capacity(rows.max(0) as usize);
    for _ in 0..rows {
        let len = values
            .next()
            .ok_or_else(|| "truncated n-gram context tensor".to_string())?;
        let mut row = Vec::with_capacity(len.max(0) as usize);
        for _ in 0..len {
            let token = values
                .next()
                .ok_or_else(|| "truncated n-gram context tensor".to_string())?;
            row.push(i64::from(token));
        }
        contexts.push(row);
    }
    if values.next().is_some() {
        return Err("n-gram context tensor has trailing values".to_string());
    }
    Ok(contexts)
}

fn check_shape(
    layer: &str,
    what: &str,
    array: &MlxArray,
    expected: &[Option<i32>],
) -> Result<(), String> {
    let shape = mlxcel_core::array_shape(array);
    let matches = shape.len() == expected.len()
        && shape
            .iter()
            .zip(expected)
            .all(|(actual, wanted)| wanted.is_none_or(|wanted| wanted == *actual));
    if matches {
        Ok(())
    } else {
        Err(format!(
            "Qwen4-Exp snapshot restore, {layer}: {what} has shape {shape:?}, expected {expected:?}"
        ))
    }
}

impl Qwen4ExpModel {
    /// Copy `caches` into a snapshot of the state for `token_len` tokens.
    ///
    /// Used by: [`Self::sequence_snapshot`], the snapshot tests.
    pub(crate) fn capture_snapshot(
        &self,
        caches: &[Qwen4ExpCache],
        token_len: usize,
        parts: SnapshotParts,
    ) -> Result<ModelStateSnapshot, String> {
        let offset = Qwen4ExpCache::offset(caches);
        if usize::try_from(offset).ok() != Some(token_len) {
            return Err(format!(
                "attention offset is {offset} but the snapshot would describe {token_len} tokens (an unrewound speculative block or a prefix mismatch)"
            ));
        }
        let mut snapshot = ModelStateSnapshot::new(SNAPSHOT_FAMILY, token_len);
        push_i32(&mut snapshot, "layers", caches.len() as i32);
        for (idx, cache) in caches.iter().enumerate() {
            let prefix = format!("layer{idx}");
            match cache {
                Qwen4ExpCache::Linear { delta, ple } => {
                    push_i32(&mut snapshot, format!("{prefix}.kind"), LAYER_KIND_LINEAR);
                    delta.snapshot_into(&mut snapshot, &prefix);
                    if let Some(ple) = ple {
                        push_optional(
                            &mut snapshot,
                            format!("{prefix}.ple.conv_state"),
                            &ple.conv_state,
                        );
                        let contexts: &[Vec<i64>] = if parts.ple_contexts {
                            &ple.contexts
                        } else {
                            &[]
                        };
                        let encoded =
                            encode_contexts(contexts).map_err(|err| format!("{prefix}: {err}"))?;
                        snapshot.push_tensor(
                            format!("{prefix}.ple.contexts"),
                            encoded.as_ref().expect("from_slice_i32 returns an array"),
                        );
                    }
                }
                Qwen4ExpCache::Attention { kv, index } => {
                    push_i32(
                        &mut snapshot,
                        format!("{prefix}.kind"),
                        LAYER_KIND_ATTENTION,
                    );
                    snapshot_standard(kv, &mut snapshot, &prefix, KV_NAMES)?;
                    push_optional(
                        &mut snapshot,
                        format!("{prefix}.qsa.raw_keys"),
                        &index.raw_keys,
                    );
                    push_optional(
                        &mut snapshot,
                        format!("{prefix}.qsa.positions"),
                        &index.positions,
                    );
                    if parts.qsa_block_keys {
                        push_optional(
                            &mut snapshot,
                            format!("{prefix}.qsa.block_keys"),
                            &index.block_keys,
                        );
                    }
                }
            }
        }
        Ok(snapshot)
    }

    /// The caches `snapshot` describes, validated against this model. Fails
    /// with a named error on a foreign family, a layer-count or layer-kind
    /// mismatch, a shape that does not fit this configuration, or a KV mode
    /// other than the live one; the caller then runs a cold prefill.
    ///
    /// Used by: [`Self::restore_sequence_snapshot`], the snapshot tests.
    pub(crate) fn restore_snapshot_caches(
        &self,
        snapshot: &ModelStateSnapshot,
    ) -> Result<Vec<Qwen4ExpCache>, String> {
        if snapshot.family() != SNAPSHOT_FAMILY {
            return Err(format!(
                "cannot restore a {} snapshot into Qwen4-Exp",
                snapshot.family()
            ));
        }
        let token_len = i32::try_from(snapshot.token_len()).map_err(|_| {
            format!(
                "Qwen4-Exp snapshot restore: token length {} does not fit int32",
                snapshot.token_len()
            )
        })?;
        let layers = restore_i32(snapshot, "layers")
            .ok_or_else(|| "Qwen4-Exp snapshot restore: missing layer count".to_string())?;
        let mut caches = self.make_internal_caches();
        if layers as usize != caches.len() {
            return Err(format!(
                "Qwen4-Exp snapshot restore: layer count {layers} does not match the model's {}",
                caches.len()
            ));
        }
        let raw = &self.config.raw;
        let as_dim = |value: usize| Some(value as i32);
        for (idx, cache) in caches.iter_mut().enumerate() {
            let prefix = format!("layer{idx}");
            let kind = restore_i32(snapshot, format!("{prefix}.kind")).ok_or_else(|| {
                format!("Qwen4-Exp snapshot restore, {prefix}: missing layer kind")
            })?;
            match cache {
                Qwen4ExpCache::Linear { delta, ple } => {
                    if kind != LAYER_KIND_LINEAR {
                        return Err(format!(
                            "Qwen4-Exp snapshot restore, {prefix}: layer kind {kind} does not match a linear layer"
                        ));
                    }
                    if let Some(state) = snapshot.tensor(&format!("{prefix}.state_cache")) {
                        check_shape(
                            &prefix,
                            "state_cache",
                            state,
                            &[
                                Some(1),
                                as_dim(raw.linear_num_value_heads),
                                as_dim(raw.linear_value_head_dim),
                                as_dim(raw.linear_key_head_dim),
                            ],
                        )?;
                    }
                    if let Some(window) = snapshot.tensor(&format!("{prefix}.conv_state")) {
                        let conv_dim = 2 * raw.linear_key_head_dim * raw.linear_num_key_heads
                            + raw.linear_value_head_dim * raw.linear_num_value_heads;
                        check_shape(
                            &prefix,
                            "conv_state",
                            window,
                            &[
                                Some(1),
                                as_dim(raw.linear_conv_kernel_dim - 1),
                                as_dim(conv_dim),
                            ],
                        )?;
                    }
                    *delta = GatedDeltaCache::new();
                    delta.restore_from(snapshot, &prefix);
                    if let Some(ple) = ple {
                        let window_rows =
                            (raw.ple_conv_kernel_size as i32 - 1) * raw.ngram_size as i32;
                        if let Some(window) = snapshot.tensor(&format!("{prefix}.ple.conv_state")) {
                            check_shape(
                                &prefix,
                                "ple.conv_state",
                                window,
                                &[Some(1), Some(window_rows), None],
                            )?;
                        }
                        ple.conv_state =
                            restore_optional(snapshot, format!("{prefix}.ple.conv_state"));
                        let contexts = snapshot
                            .tensor(&format!("{prefix}.ple.contexts"))
                            .ok_or_else(|| {
                                format!(
                                    "Qwen4-Exp snapshot restore, {prefix}: missing n-gram contexts"
                                )
                            })?;
                        ple.contexts = decode_contexts(contexts).map_err(|err| {
                            format!("Qwen4-Exp snapshot restore, {prefix}: {err}")
                        })?;
                        let context_len = raw.ngram_size - 1;
                        if ple.contexts.iter().any(|row| row.len() != context_len) {
                            return Err(format!(
                                "Qwen4-Exp snapshot restore, {prefix}: n-gram contexts are not {context_len} tokens long"
                            ));
                        }
                    }
                }
                Qwen4ExpCache::Attention { kv, index } => {
                    if kind != LAYER_KIND_ATTENTION {
                        return Err(format!(
                            "Qwen4-Exp snapshot restore, {prefix}: layer kind {kind} does not match an attention layer"
                        ));
                    }
                    for (name, array) in [
                        ("kv.keys", snapshot.tensor(&format!("{prefix}.kv.keys"))),
                        ("kv.values", snapshot.tensor(&format!("{prefix}.kv.values"))),
                    ] {
                        if let Some(array) = array {
                            check_shape(
                                &prefix,
                                name,
                                array,
                                &[
                                    Some(1),
                                    as_dim(raw.num_key_value_heads),
                                    None,
                                    as_dim(
                                        raw.head_dim
                                            .unwrap_or(raw.hidden_size / raw.num_attention_heads),
                                    ),
                                ],
                            )?;
                        }
                    }
                    restore_standard(kv, snapshot, &prefix, KV_NAMES)?;
                    if kv.offset != token_len {
                        return Err(format!(
                            "Qwen4-Exp snapshot restore, {prefix}: KV offset {} does not match the snapshot's {token_len} tokens",
                            kv.offset
                        ));
                    }
                    let head_dim = raw.indexer_head_dim as i32;
                    let raw_keys = snapshot
                        .tensor(&format!("{prefix}.qsa.raw_keys"))
                        .ok_or_else(|| {
                            format!("Qwen4-Exp snapshot restore, {prefix}: missing QSA raw keys")
                        })?;
                    check_shape(
                        &prefix,
                        "qsa.raw_keys",
                        raw_keys,
                        &[Some(1), Some(token_len), Some(head_dim)],
                    )?;
                    let positions = snapshot
                        .tensor(&format!("{prefix}.qsa.positions"))
                        .ok_or_else(|| {
                            format!("Qwen4-Exp snapshot restore, {prefix}: missing QSA positions")
                        })?;
                    check_shape(
                        &prefix,
                        "qsa.positions",
                        positions,
                        &[Some(3), Some(1), Some(token_len)],
                    )?;
                    if let Some(block_keys) = snapshot.tensor(&format!("{prefix}.qsa.block_keys")) {
                        check_shape(
                            &prefix,
                            "qsa.block_keys",
                            block_keys,
                            &[Some(1), Some(1), None, Some(head_dim)],
                        )?;
                        let blocks = mlxcel_core::array_shape(block_keys)[2];
                        let complete = token_len / raw.indexer_compress_ratio as i32;
                        if blocks > complete {
                            return Err(format!(
                                "Qwen4-Exp snapshot restore, {prefix}: {blocks} block keys for {complete} complete blocks"
                            ));
                        }
                    }
                    index.raw_keys = restore_optional(snapshot, format!("{prefix}.qsa.raw_keys"));
                    index.positions = restore_optional(snapshot, format!("{prefix}.qsa.positions"));
                    index.block_keys =
                        restore_optional(snapshot, format!("{prefix}.qsa.block_keys"));
                }
            }
        }
        Ok(caches)
    }

    /// `LanguageModel::snapshot_sequence_state`: `None`, with the reason
    /// logged, when the sequence cannot be snapshotted.
    pub(super) fn sequence_snapshot(
        &self,
        seq_id: SequenceId,
        token_len: usize,
    ) -> Option<ModelStateSnapshot> {
        let has_media_positions = self.mrope_state.with_entry(Some(seq_id), |entry| {
            entry.position_ids.is_some() || entry.rope_deltas.is_some_and(|delta| delta != 0)
        });
        if has_media_positions {
            tracing::debug!(
                seq_id = %seq_id,
                "Qwen4-Exp prompt-cache snapshot declined: the sequence carries image or video MRoPE state"
            );
            return None;
        }
        self.sequence_state
            .with_sequence_state_ref(seq_id, |caches| {
                match self.capture_snapshot(caches, token_len, SnapshotParts::ALL) {
                    Ok(snapshot) => Some(snapshot),
                    Err(err) => {
                        tracing::warn!("Qwen4-Exp prompt-cache snapshot skipped: {err}");
                        None
                    }
                }
            })
            .flatten()
    }

    /// `LanguageModel::restore_sequence_state`. Installs the state and an
    /// empty rollback log; `Err` leaves the sequence untouched.
    pub(super) fn restore_sequence_snapshot(
        &self,
        seq_id: SequenceId,
        snapshot: &ModelStateSnapshot,
    ) -> Result<(), String> {
        let caches = self.restore_snapshot_caches(snapshot)?;
        self.sequence_state.replace_sequence_state(seq_id, caches);
        self.rollback_log
            .replace_sequence_state(seq_id, vec![super::RollbackLog::default()]);
        Ok(())
    }
}
