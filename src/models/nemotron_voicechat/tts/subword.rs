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

//! Char-aware subword embedding: each LLM subword is spelled out as
//! characters, encoded by a small T5Gemma encoder, mean-pooled and projected.
//!
//! Ports `CharAwareSubwordEncoder`, `SubwordFlagEmbedding` and
//! `BOSEOSEmbedding` from `mlx_vlm/models/nemotron_voicechat/tts.py`.
//!
//! The dense character vocabulary is derived from the LLM tokenizer's own
//! vocabulary exactly as NeMo does: every single-code-point token, sorted by
//! token id, gets consecutive char ids, and the last row of the char
//! embedding (`char_vocab_size - 1`) is the padding id.
//! [`CharAwareSubwordEncoder::set_vocabulary`] fails when the derived table
//! does not have `char_vocab_size - 1` entries, because a different
//! tokenizer would silently spell every word with the wrong characters.
//!
//! The per-token flag buffers (`is_continuation`, `special_flags`) are read
//! once into host vectors: they are pure lookup tables and the subword ids
//! are host values anyway (the reference calls `.tolist()` on them).

use std::collections::HashMap;

use mlxcel_core::dtype;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::char_encoder::CharEncoder;
use super::config::CharEncoderConfig;
use super::norm_mlp::{weight, weight_with_shape};

/// Read an integer tensor into host `i32`s.
pub(crate) fn to_host_i32(arr: &MlxArray) -> Vec<i32> {
    let as_i32 = mlxcel_core::astype(arr, dtype::INT32);
    mlxcel_core::eval(&as_i32);
    mlxcel_core::array_to_raw_bytes(&as_i32)
        .chunks_exact(4)
        .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One flag lookup: `table[where(id >= limit, pad, id)]` then an embedding row.
struct FlagEmbedding {
    table: Vec<i32>,
    limit: i64,
    pad: i32,
    emb: UniquePtr<MlxArray>,
}

impl FlagEmbedding {
    fn load(
        weights: &WeightMap,
        prefix: &str,
        table_key: &str,
        emb_key: &str,
        limit_from_len: impl Fn(usize) -> usize,
    ) -> Result<Self, String> {
        let table_arr = weight(weights, &format!("{prefix}.{table_key}"))?;
        let table = to_host_i32(&table_arr);
        let pad_arr = weight(weights, &format!("{prefix}.pad_tensor"))?;
        let pad = to_host_i32(&pad_arr)
            .first()
            .copied()
            .ok_or_else(|| format!("{prefix}.pad_tensor is empty"))?;
        let emb = weight(weights, &format!("{prefix}.{emb_key}.weight"))?;
        let rows = mlxcel_core::array_shape(&emb)[0];
        let limit = limit_from_len(table.len());
        let pad_ok = usize::try_from(pad).is_ok_and(|p| p < table.len());
        if !pad_ok || table.iter().any(|&f| f < 0 || f >= rows) {
            return Err(format!(
                "{prefix}: flag table or pad id out of range for {rows} embedding rows"
            ));
        }
        Ok(Self {
            table,
            limit: limit as i64,
            pad,
            emb,
        })
    }

    fn flags(&self, ids: &[i32]) -> Vec<i32> {
        ids.iter()
            .map(|&id| {
                let safe = if i64::from(id) >= self.limit {
                    self.pad
                } else {
                    id
                };
                self.table[safe as usize]
            })
            .collect()
    }

    fn add_to(&self, embeds: &MlxArray, ids: &[i32], shape: &[i32]) -> UniquePtr<MlxArray> {
        let flags = mlxcel_core::from_slice_i32(&self.flags(ids), shape);
        mlxcel_core::add(embeds, &mlxcel_core::take(&self.emb, &flags, 0))
    }
}

/// `CharAwareSubwordEncoder` (key tree `embed_subword`).
pub struct CharAwareSubwordEncoder {
    encoder: CharEncoder,
    embed_tokens: UniquePtr<MlxArray>,
    proj_embedding: UnifiedLinear,
    subword_flag: FlagEmbedding,
    bos_eos: FlagEmbedding,
    char_padding_idx: i32,
    char_vocab_size: usize,
    out_size: usize,
    subword_to_chars: Option<HashMap<u32, Vec<i32>>>,
}

impl CharAwareSubwordEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &CharEncoderConfig,
        out_size: usize,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let embed_tokens = weight_with_shape(
            weights,
            &format!("{prefix}.embed_tokens.weight"),
            &[cfg.char_vocab_size as i32, cfg.hidden_size as i32],
        )?;
        let subword_flag = FlagEmbedding::load(
            weights,
            &format!("{prefix}.subword_flag_emb"),
            "is_continuation",
            "cont_emb",
            // `token_ids >= is_continuation.shape[0] - 1`
            |len| len.saturating_sub(1),
        )?;
        let bos_eos = FlagEmbedding::load(
            weights,
            &format!("{prefix}.bos_eos_emb"),
            "special_flags",
            "special_emb",
            // `token_ids >= special_flags.shape[0]`
            |len| len,
        )?;
        Ok(Self {
            encoder: CharEncoder::from_weights(
                weights,
                &format!("{prefix}.backbone.encoder"),
                cfg,
                group_size,
                bits,
            )?,
            embed_tokens,
            proj_embedding: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.proj_embedding"),
                group_size,
                bits,
            )?,
            subword_flag,
            bos_eos,
            char_padding_idx: cfg.char_vocab_size as i32 - 1,
            char_vocab_size: cfg.char_vocab_size,
            out_size,
            subword_to_chars: None,
        })
    }

    /// Build the dense character vocabulary from the LLM tokenizer
    /// vocabulary (`tokenizer.get_vocab()`, added tokens included).
    pub fn set_vocabulary(&mut self, vocabulary: &HashMap<String, u32>) -> Result<(), String> {
        let mut single: Vec<(u32, char)> = vocabulary
            .iter()
            .filter_map(|(token, &id)| {
                let mut chars = token.chars();
                match (chars.next(), chars.next()) {
                    (Some(c), None) => Some((id, c)),
                    _ => None,
                }
            })
            .collect();
        single.sort_unstable();
        let char_to_id: HashMap<char, i32> = single
            .iter()
            .enumerate()
            .map(|(idx, &(_, c))| (c, idx as i32))
            .collect();
        if char_to_id.len() + 1 != self.char_vocab_size {
            return Err(format!(
                "tokenizer-derived character vocabulary has {} entries, expected {}",
                char_to_id.len() + 1,
                self.char_vocab_size
            ));
        }
        let table = vocabulary
            .iter()
            .map(|(token, &id)| {
                let chars = token.chars().filter_map(|c| char_to_id.get(&c).copied());
                (id, chars.collect())
            })
            .collect();
        self.subword_to_chars = Some(table);
        Ok(())
    }

    /// Whether [`Self::set_vocabulary`] has run.
    pub fn has_vocabulary(&self) -> bool {
        self.subword_to_chars.is_some()
    }

    /// Embed `ids` (`[batch, time]`, row-major) whose `mask` entries are true;
    /// masked-out positions get only the flag embeddings. Returns
    /// `[batch, time, out_size]` in the char-embedding dtype.
    pub fn forward(
        &self,
        ids: &[i32],
        mask: &[bool],
        batch: usize,
        time: usize,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let table = self.subword_to_chars.as_ref().ok_or_else(|| {
            "set_vocabulary(tokenizer vocab) must be called before TTS".to_string()
        })?;
        if ids.len() != batch * time || mask.len() != ids.len() {
            return Err(format!(
                "subword ids ({}) and mask ({}) must both hold batch*time = {} entries",
                ids.len(),
                mask.len(),
                batch * time
            ));
        }
        if let Some(&bad) = ids.iter().find(|&&id| id < 0) {
            return Err(format!("negative subword id {bad}"));
        }

        let empty: Vec<i32> = Vec::new();
        let mut sequences: Vec<&Vec<i32>> = Vec::new();
        let mut row_of_position = vec![-1i32; ids.len()];
        for (pos, (&id, &keep)) in ids.iter().zip(mask).enumerate() {
            if keep {
                row_of_position[pos] = sequences.len() as i32;
                sequences.push(table.get(&(id as u32)).unwrap_or(&empty));
            }
        }

        let dtype_id = mlxcel_core::array_dtype(&self.embed_tokens);
        let shape = [batch as i32, time as i32];
        let out_shape = [batch as i32, time as i32, self.out_size as i32];
        let out = if sequences.is_empty() {
            mlxcel_core::zeros(&out_shape, dtype_id)
        } else {
            let projected = self.encode_sequences(&sequences, dtype_id);
            // `out.at[b, t].add(projected)` on a zero tensor with unique
            // positions: gather each position's row, or a zero row.
            let n = sequences.len() as i32;
            let zero_row = mlxcel_core::zeros(&[1, self.out_size as i32], dtype_id);
            let rows = mlxcel_core::concatenate(&projected, &zero_row, 0);
            let index: Vec<i32> = row_of_position
                .iter()
                .map(|&r| if r < 0 { n } else { r })
                .collect();
            let index = mlxcel_core::from_slice_i32(&index, &shape);
            mlxcel_core::take(&rows, &index, 0)
        };
        let out = self.subword_flag.add_to(&out, ids, &shape);
        Ok(self.bos_eos.add_to(&out, ids, &shape))
    }

    /// Char-encode, mean-pool and project `sequences` → `[n, out_size]`.
    fn encode_sequences(&self, sequences: &[&Vec<i32>], dtype_id: i32) -> UniquePtr<MlxArray> {
        let n = sequences.len();
        let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        // All-empty spellings still run the encoder on one padding column,
        // exactly like the reference, which yields a zero pooled vector.
        let width = max_len.max(1);
        let mut char_ids = vec![self.char_padding_idx; n * width];
        let mut valid = vec![0i32; n * width];
        let mut divisor = Vec::with_capacity(n);
        for (row, seq) in sequences.iter().enumerate() {
            char_ids[row * width..row * width + seq.len()].copy_from_slice(seq);
            valid[row * width..row * width + seq.len()].fill(1);
            divisor.push((seq.len() as i32).max(1));
        }
        let (n, width) = (n as i32, width as i32);
        let char_ids = mlxcel_core::from_slice_i32(&char_ids, &[n, width]);
        let mask = mlxcel_core::astype(
            &mlxcel_core::from_slice_i32(&valid, &[n, width]),
            dtype::BOOL,
        );
        let embeds = mlxcel_core::take(&self.embed_tokens, &char_ids, 0);
        let hidden = self.encoder.forward(&embeds, &mask);
        let masked = mlxcel_core::multiply(&hidden, &mlxcel_core::expand_dims(&mask, -1));
        let summed = mlxcel_core::sum_axis(&masked, 1, false);
        let divisor = mlxcel_core::from_slice_i32(&divisor, &[n, 1]);
        let pooled = mlxcel_core::divide(&summed, &divisor);
        let projected = self.proj_embedding.forward(&pooled);
        if mlxcel_core::array_dtype(&projected) == dtype_id {
            projected
        } else {
            mlxcel_core::astype(&projected, dtype_id)
        }
    }
}
