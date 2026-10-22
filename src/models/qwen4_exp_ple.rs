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

//! Qwen4-Exp hashed n-gram "PLE" embedding (`Qwen4ExpNGramEmbedding`,
//! `Qwen4ExpPLELayer`).
//!
//! Each token looks up `(ngram_size - 1) * heads_per_ngram` rows of one
//! large table: for every n-gram order `n` ending at the token, the previous
//! tokens are hashed with per-layer odd multipliers and reduced modulo one
//! prime per head. The context never crosses an EOS token. The hash runs on
//! the host, as it does upstream, because the table is kept in
//! checkpoint-sized row shards and each id has to be routed to its shard.

use mlxcel_core::layers::{UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::silu;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};

use super::config::ResolvedQwen4ExpConfig;
use super::norm::ZeroCenteredRmsNorm;
use super::shapes;

const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
const LAYER_SEED_PRIME: u64 = 10007;

pub(crate) fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(SPLITMIX_GAMMA);
    z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_M1);
    z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_M2);
    z ^ (z >> 31)
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3;
    while divisor <= value / divisor {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

/// The first prime strictly greater than `value`, or `None` past `limit`.
fn next_prime_within(value: u64, limit: u64) -> Option<u64> {
    let mut candidate = value.checked_add(1)?;
    while candidate <= limit {
        if is_prime(candidate) {
            return Some(candidate);
        }
        candidate = candidate.checked_add(1)?;
    }
    None
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NgramHasher {
    pub(crate) ngram_size: usize,
    pub(crate) heads_per_ngram: usize,
    pub(crate) eos_token_id: i64,
    pub(crate) multipliers: Vec<i64>,
    pub(crate) head_vocab_sizes: Vec<i64>,
    pub(crate) head_offsets: Vec<i64>,
    /// Table rows: the summed head vocabularies rounded up to
    /// `make_ngram_vocab_size_divisible_by`.
    pub(crate) padded_vocab_size: u64,
}

impl NgramHasher {
    /// `table_rows` is the size of the n-gram table the checkpoint stores.
    /// Every head vocabulary is a row range of it, so the prime search stops
    /// as soon as the heads can no longer fit: the search is otherwise driven
    /// by config values alone.
    pub(crate) fn new(
        config: &ResolvedQwen4ExpConfig,
        ple_layer_index: usize,
        table_rows: u64,
    ) -> Result<Self, String> {
        let raw = &config.raw;
        let eos_token_id = config
            .eos_token_id
            .ok_or_else(|| "Qwen4-Exp PLE needs eos_token_id".to_string())?;
        let ngram_heads = config.ngram_heads();
        let too_large = || {
            format!(
                "Qwen4-Exp PLE layer {ple_layer_index}: ngram_vocab_size_base {} with {ngram_heads} \
                 n-gram heads per layer derives a table larger than the {table_rows} rows the \
                 checkpoint stores",
                raw.ngram_vocab_size_base
            )
        };

        let first_head = ple_layer_index
            .checked_mul(ngram_heads)
            .ok_or_else(too_large)?;
        let mut head_vocab_sizes = Vec::new();
        let mut head_offsets = Vec::new();
        let mut total: u64 = 0;
        let mut prime = raw.ngram_vocab_size_base - 1;
        for global_head in 0..first_head + ngram_heads {
            prime = next_prime_within(prime, table_rows).ok_or_else(too_large)?;
            if global_head < first_head {
                continue;
            }
            head_vocab_sizes.push(prime as i64);
            head_offsets.push(total as i64);
            total = total
                .checked_add(prime)
                .filter(|&total| total <= table_rows)
                .ok_or_else(too_large)?;
        }
        let divisor = raw.make_ngram_vocab_size_divisible_by;
        let padded_vocab_size = total
            .div_ceil(divisor)
            .checked_mul(divisor)
            .ok_or_else(too_large)?;

        let multiplier_max = (i64::MAX as u64) / (raw.vocab_size as u64).max(1);
        let half_bound = (multiplier_max / 2).max(1);
        let base_seed = raw
            .seed
            .wrapping_add(LAYER_SEED_PRIME.wrapping_mul(ple_layer_index as u64));
        let multipliers = (0..raw.ngram_size as u64)
            .map(|index| {
                let value = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(index + 1));
                (2 * (splitmix64(value) % half_bound) + 1) as i64
            })
            .collect();

        Ok(Self {
            ngram_size: raw.ngram_size,
            heads_per_ngram: raw.heads_per_ngram,
            eos_token_id,
            multipliers,
            head_vocab_sizes,
            head_offsets,
            padded_vocab_size,
        })
    }

    pub(crate) fn context_len(&self) -> usize {
        self.ngram_size - 1
    }

    pub(crate) fn ngram_heads(&self) -> usize {
        self.context_len() * self.heads_per_ngram
    }

    pub(crate) fn initial_context(&self) -> Vec<i64> {
        vec![self.eos_token_id; self.context_len()]
    }

    /// Table ids `[tokens.len() * ngram_heads]` for `tokens` following
    /// `context`, and the context to carry into the next call.
    ///
    /// A token sees the previous `s` tokens only when no EOS sits between
    /// them; the EOS itself still belongs to the segment it closes.
    pub(crate) fn ids(&self, context: &[i64], tokens: &[i64]) -> (Vec<i64>, Vec<i64>) {
        let history: Vec<i64> = context.iter().chain(tokens).copied().collect();
        let mut previous_eos = Vec::with_capacity(history.len());
        let mut last_eos: i64 = -1;
        for (position, &token) in history.iter().enumerate() {
            previous_eos.push(last_eos);
            if token == self.eos_token_id {
                last_eos = position as i64;
            }
        }

        let mut ids = Vec::with_capacity(tokens.len() * self.ngram_heads());
        for position in context.len()..history.len() {
            let position_in_segment = position as i64 - (previous_eos[position] + 1);
            let shifted = |shift: usize| {
                if position_in_segment >= shift as i64 {
                    history[position - shift]
                } else {
                    self.eos_token_id
                }
            };
            for order in 2..=self.ngram_size {
                let mut mixed = shifted(0).wrapping_mul(self.multipliers[0]);
                for (offset, multiplier) in self.multipliers.iter().enumerate().take(order).skip(1)
                {
                    mixed ^= shifted(offset).wrapping_mul(*multiplier);
                }
                let first_head = (order - 2) * self.heads_per_ngram;
                for head in first_head..first_head + self.heads_per_ngram {
                    ids.push(
                        mixed.rem_euclid(self.head_vocab_sizes[head]) + self.head_offsets[head],
                    );
                }
            }
        }
        let keep = history.len() - self.context_len();
        (ids, history[keep..].to_vec())
    }
}

/// The n-gram table, kept as the checkpoint's row shards.
pub(crate) struct ShardedEmbedding {
    shards: Vec<UnifiedEmbedding>,
    /// Row offset of each shard, plus the total row count.
    offsets: Vec<i64>,
}

impl ShardedEmbedding {
    pub(crate) fn stored_rows(
        weights: &WeightMap,
        prefix: &str,
        num_shards: usize,
    ) -> Result<u64, String> {
        let mut rows: u64 = 0;
        for index in 0..num_shards {
            let key = format!("{prefix}.shards.{index}.weight");
            let weight = weights
                .get(&key)
                .ok_or_else(|| format!("Qwen4-Exp checkpoint is missing {key}"))?;
            let shape = mlxcel_core::array_shape(weight);
            if shape.len() != 2 {
                return Err(format!("{key} has shape {shape:?}, expected a 2-D table"));
            }
            rows += shape[0] as u64;
        }
        Ok(rows)
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &ResolvedQwen4ExpConfig,
        prefix: &str,
        rows: u64,
        row_width: usize,
    ) -> Result<Self, String> {
        let num_shards = config.raw.split_ngram_parts as u64;
        if num_shards == 0 || num_shards > rows {
            return Err(format!(
                "Qwen4-Exp split_ngram_parts ({num_shards}) must be in [1, {rows}]"
            ));
        }
        let (base, remainder) = (rows / num_shards, rows % num_shards);
        let mut shards = Vec::with_capacity(num_shards as usize);
        let mut offsets = vec![0i64];
        for index in 0..num_shards {
            let expected_rows = base + u64::from(index < remainder);
            let shard_prefix = format!("{prefix}.shards.{index}");
            let weight_key = format!("{shard_prefix}.weight");
            let weight = weights
                .get(&weight_key)
                .ok_or_else(|| format!("Qwen4-Exp checkpoint is missing {weight_key}"))?;
            let shape = mlxcel_core::array_shape(weight);
            if shape.len() != 2 || shape[0] as u64 != expected_rows {
                return Err(format!(
                    "{weight_key} has shape {shape:?}, expected {expected_rows} rows"
                ));
            }
            let (group_size, bits) = config.quantization.for_module(&shard_prefix);
            let quantized = weights.contains_key(&format!("{shard_prefix}.scales"));
            let width = if quantized {
                shape[1] as usize * 32 / bits.max(1) as usize
            } else {
                shape[1] as usize
            };
            if width != row_width {
                return Err(format!(
                    "{weight_key} stores rows of width {width}, expected ple_embed_dim / n-gram heads = {row_width}"
                ));
            }
            // The packed width alone does not bound the sidecar gathers.
            // Check scale/bias rows before the first embedding lookup.
            shapes::expect_matrix(
                weights,
                &shard_prefix,
                (group_size, bits),
                &[expected_rows as usize, row_width],
            )?;
            shards.push(UnifiedEmbedding::from_weights(
                weights,
                &shard_prefix,
                group_size,
                bits,
            )?);
            offsets.push(offsets[index as usize] + expected_rows as i64);
        }
        Ok(Self { shards, offsets })
    }

    /// Gather `ids` (all in `[0, rows)`) into `[ids.len(), row_width]`.
    pub(crate) fn lookup(&self, ids: &[i64]) -> UniquePtr<MlxArray> {
        let mut per_shard: Vec<Vec<(usize, i32)>> = vec![Vec::new(); self.shards.len()];
        for (position, &id) in ids.iter().enumerate() {
            let shard = self.offsets.partition_point(|&offset| offset <= id) - 1;
            per_shard[shard].push((position, (id - self.offsets[shard]) as i32));
        }
        let mut parts: Vec<UniquePtr<MlxArray>> = Vec::new();
        let mut source_row = vec![0i32; ids.len()];
        let mut next_row = 0i32;
        for (shard, entries) in self.shards.iter().zip(&per_shard) {
            if entries.is_empty() {
                continue;
            }
            let local: Vec<i32> = entries.iter().map(|&(_, local)| local).collect();
            for &(position, _) in entries {
                source_row[position] = next_row;
                next_row += 1;
            }
            parts.push(shard.forward(&mlxcel_core::from_slice_i32(&local, &[local.len() as i32])));
        }
        let parts: Vec<&MlxArray> = parts.iter().map(|part| &**part).collect();
        let gathered = mlxcel_core::concatenate_many(&parts, 0);
        let order = mlxcel_core::from_slice_i32(&source_row, &[source_row.len() as i32]);
        mlxcel_core::take(&gathered, &order, 0)
    }
}

/// Per-sequence PLE state: the short-convolution window and, per batch row,
/// the last `ngram_size - 1` tokens.
#[derive(Default)]
pub(crate) struct PleCache {
    pub(crate) conv_state: Option<UniquePtr<MlxArray>>,
    pub(crate) contexts: Vec<Vec<i64>>,
}

pub(crate) struct PleLayer {
    hasher: NgramHasher,
    table: ShardedEmbedding,
    key_proj: UnifiedLinear,
    value_proj: UnifiedLinear,
    norm_key: ZeroCenteredRmsNorm,
    norm_query: ZeroCenteredRmsNorm,
    norm_conv: ZeroCenteredRmsNorm,
    conv1d_weight: UniquePtr<MlxArray>,
    conv_dilation: i32,
    conv_state_len: i32,
    hc_count: i32,
    hidden_size: i32,
}

impl PleLayer {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &ResolvedQwen4ExpConfig,
        prefix: &str,
        ple_layer_index: usize,
    ) -> Result<Self, String> {
        let raw = &config.raw;
        let table_prefix = format!("{prefix}.ple_embedding.ngram_embedding");
        let table_rows =
            ShardedEmbedding::stored_rows(weights, &table_prefix, raw.split_ngram_parts)?;
        let hasher = NgramHasher::new(config, ple_layer_index, table_rows)?;
        let row_width = config.ple_embed_dim / hasher.ngram_heads();
        let table = ShardedEmbedding::from_weights(
            weights,
            config,
            &table_prefix,
            hasher.padded_vocab_size,
            row_width,
        )?;
        verify_stored_hash_state(weights, prefix, &hasher)?;

        let streams = raw.hc_count * raw.hidden_size;
        let linear = |name: &str, rows: usize| {
            let module = format!("{prefix}.{name}");
            let (group_size, bits) = config.quantization.for_module(&module);
            shapes::expect_matrix(
                weights,
                &module,
                (group_size, bits),
                &[rows, config.ple_embed_dim],
            )?;
            UnifiedLinear::from_weights(weights, &module, group_size, bits)
        };
        let grouped_norm = |name: &str| {
            ZeroCenteredRmsNorm::from_weights(
                weights,
                &format!("{prefix}.{name}"),
                streams,
                Some(raw.hidden_size),
                raw.rms_norm_eps,
            )
        };
        let channels = streams as i32;
        let conv_key = format!("{prefix}.conv1d.weight");
        let conv1d_weight = weights
            .get(&conv_key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Qwen4-Exp checkpoint is missing {conv_key}"))?;
        let conv_shape = mlxcel_core::array_shape(&conv1d_weight);
        let expected = [channels, raw.ple_conv_kernel_size as i32, 1];
        if conv_shape != expected {
            return Err(format!(
                "{conv_key} has shape {conv_shape:?}, expected {expected:?}"
            ));
        }
        let conv_dilation = raw.ngram_size as i32;
        Ok(Self {
            conv_state_len: (raw.ple_conv_kernel_size as i32 - 1) * conv_dilation,
            hasher,
            table,
            key_proj: linear("key_proj", streams)?,
            value_proj: linear("value_proj", raw.hidden_size)?,
            norm_key: grouped_norm("norm_key")?,
            norm_query: grouped_norm("norm_query")?,
            norm_conv: grouped_norm("norm_conv")?,
            conv1d_weight,
            conv_dilation,
            hc_count: raw.hc_count as i32,
            hidden_size: raw.hidden_size as i32,
        })
    }

    /// Embeddings `[B, L, ple_embed_dim]` for `tokens` (one row per batch
    /// entry), advancing each row's n-gram context in `cache`.
    fn embed(&self, tokens: &[Vec<i64>], cache: &mut PleCache) -> UniquePtr<MlxArray> {
        if cache.contexts.len() != tokens.len() {
            cache.contexts = vec![self.hasher.initial_context(); tokens.len()];
        }
        let mut ids = Vec::with_capacity(
            tokens.iter().map(Vec::len).sum::<usize>() * self.hasher.ngram_heads(),
        );
        for (row, context) in tokens.iter().zip(cache.contexts.iter_mut()) {
            let (row_ids, next) = self.hasher.ids(context, row);
            ids.extend(row_ids);
            *context = next;
        }
        let rows = self.table.lookup(&ids);
        let batch = tokens.len() as i32;
        let length = tokens.first().map_or(0, Vec::len) as i32;
        mlxcel_core::reshape(&rows, &[batch, length, -1])
    }

    fn short_conv(&self, x: &MlxArray, cache: &mut PleCache) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let state = cache
            .conv_state
            .as_ref()
            .filter(|state| mlxcel_core::array_shape(state)[0] == shape[0])
            .map(|state| mlxcel_core::copy(state))
            .unwrap_or_else(|| {
                mlxcel_core::zeros(
                    &[shape[0], self.conv_state_len, shape[2]],
                    mlxcel_core::array_dtype(x),
                )
            });
        let conv_input = concatenate(&state, x, 1);
        let total = mlxcel_core::array_shape(&conv_input)[1];
        let tail = mlxcel_core::slice(
            &conv_input,
            &[0, total - self.conv_state_len, 0],
            &[shape[0], total, shape[2]],
        );
        cache.conv_state = Some(mlxcel_core::contiguous(&tail, false));
        let out = mlxcel_core::conv1d(
            &conv_input,
            &self.conv1d_weight,
            1,
            0,
            self.conv_dilation,
            shape[2],
        );
        silu(&out)
    }

    /// The residual update for the `[B, L, hc_count * hidden]` stream.
    pub(crate) fn forward(
        &self,
        hidden: &MlxArray,
        tokens: &[Vec<i64>],
        cache: &mut PleCache,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden);
        let mut streams = shape[..shape.len() - 1].to_vec();
        streams.extend([self.hc_count, self.hidden_size]);

        let embeddings = self.embed(tokens, cache);
        let keys = mlxcel_core::reshape(
            &self.norm_key.forward(&self.key_proj.forward(&embeddings)),
            &streams,
        );
        let values = self.value_proj.forward(&embeddings);
        let queries = mlxcel_core::reshape(&self.norm_query.forward(hidden), &streams);

        let gate =
            mlxcel_core::sum_axis_native_dtype(&mlxcel_core::multiply(&keys, &queries), -1, true);
        let gate = mlxcel_core::divide_scalar(&gate, (self.hidden_size as f32).sqrt());
        let floor = mlxcel_core::full_f32(&[1], 1e-6, mlxcel_core::array_dtype(&gate));
        let magnitude = mlxcel_core::sqrt(&mlxcel_core::maximum(&mlxcel_core::abs(&gate), &floor));
        let gate = mlxcel_core::multiply(&mlxcel_core::sign(&gate), &magnitude);
        let gated = mlxcel_core::multiply(
            &mlxcel_core::sigmoid(&gate),
            &mlxcel_core::expand_dims(&values, -2),
        );
        let gated = mlxcel_core::reshape(&gated, &shape);
        let normed = self.norm_conv.forward(&gated);
        mlxcel_core::add(&gated, &self.short_conv(&normed, cache))
    }
}

/// Converted checkpoints store the hash state as tensors. When they do, it
/// has to agree with what the config derives, or the table rows would be read
/// at the wrong ids.
fn verify_stored_hash_state(
    weights: &WeightMap,
    prefix: &str,
    hasher: &NgramHasher,
) -> Result<(), String> {
    let checks: [(&str, &[i64]); 3] = [
        ("layer_multipliers", &hasher.multipliers),
        ("ngram_heads_vocab_sizes", &hasher.head_vocab_sizes),
        ("ngram_heads_offsets", &hasher.head_offsets),
    ];
    for (name, expected) in checks {
        let key = format!("{prefix}.ple_embedding.{name}");
        let Some(stored) = weights.get(&key) else {
            continue;
        };
        let shape = mlxcel_core::array_shape(stored);
        if shape != [expected.len() as i32] {
            return Err(format!(
                "{key} has shape {shape:?}, expected [{}]",
                expected.len()
            ));
        }
        let stored = mlxcel_core::astype(stored, mlxcel_core::dtype::INT64);
        mlxcel_core::eval(&stored);
        let bytes = mlxcel_core::array_to_raw_bytes(&stored);
        let values: Vec<i64> = bytes
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("8-byte chunk")))
            .collect();
        if values != expected {
            return Err(format!(
                "{key} = {values:?} disagrees with the value derived from the config ({expected:?})"
            ));
        }
    }
    Ok(())
}
