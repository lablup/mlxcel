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

// Mamba (v1): SSM-based architecture for mlxcel-core
// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mamba.py

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{silu, slice_axis};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, concatenate};
use serde::Deserialize;
use std::path::Path;

use super::model_owned::ModelOwnedSequenceState;
use super::recurrent_snapshot::{push_optional, restore_optional};

/// Parse eos_token_id from config.json. Can be a single int, an array of ints, or absent.
/// Used by: Mamba, Mamba2
pub fn parse_eos_token_ids(value: &Option<serde_json::Value>, default: i32) -> Vec<i32> {
    match value {
        Some(serde_json::Value::Number(n)) => vec![n.as_i64().unwrap_or(default as i64) as i32],
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_i64().map(|n| n as i32))
            .collect(),
        _ => vec![default],
    }
}

// Configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MambaConfig {
    pub model_type: String,
    pub vocab_size: usize,
    #[serde(alias = "d_model")]
    pub hidden_size: usize,
    #[serde(alias = "d_inner")]
    pub intermediate_size: usize,
    #[serde(alias = "d_state")]
    pub state_size: usize,
    #[serde(alias = "n_layer", alias = "n_layers")]
    pub num_hidden_layers: usize,
    #[serde(alias = "d_conv")]
    pub conv_kernel: usize,
    #[serde(alias = "bias", default)]
    pub use_bias: bool,
    #[serde(alias = "conv_bias", default = "default_true")]
    pub use_conv_bias: bool,
    #[serde(deserialize_with = "deserialize_time_step_rank")]
    pub time_step_rank: usize,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub use_bcdt_rms: bool,
    #[serde(default = "default_mixer_rms_eps")]
    pub mixer_rms_eps: f32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub quantization: Option<Quantization>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub bos_token_id: Option<serde_json::Value>,
}

fn default_true() -> bool {
    true
}

fn default_mixer_rms_eps() -> f32 {
    1e-6
}

fn default_rms_norm_eps() -> f32 {
    1e-5
}

fn deserialize_time_step_rank<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum TimeStepRank {
        Number(usize),
        String(String),
    }

    match TimeStepRank::deserialize(deserializer)? {
        TimeStepRank::Number(n) => Ok(n),
        TimeStepRank::String(s) if s == "auto" => Ok(0), // Will be computed later
        TimeStepRank::String(s) => Err(D::Error::custom(format!("invalid time_step_rank: {}", s))),
    }
}

impl MambaConfig {
    pub fn compute_time_step_rank(&mut self) {
        if self.time_step_rank == 0 {
            self.time_step_rank = self.hidden_size.div_ceil(16);
        }
        if self.model_type == "falcon_mamba" {
            self.use_bcdt_rms = true;
        }
    }

    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }
}

// SSM Cache for Mamba.
pub struct MambaCache {
    pub conv_state: Option<UniquePtr<MlxArray>>,
    pub ssm_state: Option<UniquePtr<MlxArray>>,
}

impl MambaCache {
    pub fn new() -> Self {
        Self {
            conv_state: None,
            ssm_state: None,
        }
    }

    pub fn snapshot_into(
        &self,
        snapshot: &mut mlxcel_core::generate::ModelStateSnapshot,
        prefix: &str,
    ) {
        push_optional(snapshot, format!("{prefix}.conv_state"), &self.conv_state);
        push_optional(snapshot, format!("{prefix}.ssm_state"), &self.ssm_state);
    }

    pub fn restore_from(
        &mut self,
        snapshot: &mlxcel_core::generate::ModelStateSnapshot,
        prefix: &str,
    ) {
        self.conv_state = restore_optional(snapshot, format!("{prefix}.conv_state"));
        self.ssm_state = restore_optional(snapshot, format!("{prefix}.ssm_state"));
    }
}

impl Default for MambaCache {
    fn default() -> Self {
        Self::new()
    }
}

// Helper Functions.
/// RMS normalization without learnable scale (for mixer_norm in falcon_mamba)
/// RMS norm without learned scale — matches Python's
/// `mx.fast.rms_norm(x, mx.ones(x.shape[-1], x.dtype), eps)`.
/// Uses MLX fast kernel for float32-precision internal computation.
fn rms_norm_no_scale(x: &MlxArray, eps: f32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let last_dim = shape[shape.len() - 1];
    let ones = mlxcel_core::ones(&[last_dim], mlxcel_core::array_dtype(x));
    mlxcel_core::fast_rms_norm(x, &ones, eps)
}

// Model Components.
/// Mamba SSM Block
#[allow(dead_code)]
pub struct MambaBlock {
    hidden_size: usize,
    intermediate_size: usize,
    state_size: usize,
    conv_kernel_size: usize,
    time_step_rank: usize,
    use_bcdt_rms: bool,
    mixer_rms_eps: f32,

    // Conv1d weights (depthwise)
    conv_weight: UniquePtr<MlxArray>,
    conv_bias: Option<UniquePtr<MlxArray>>,

    // Projections
    in_proj: UnifiedLinear,
    x_proj: UnifiedLinear,
    dt_proj: UnifiedLinear,
    out_proj: UnifiedLinear,

    // SSM parameters
    a_log: UniquePtr<MlxArray>,
    d_param: UniquePtr<MlxArray>,
}

impl MambaBlock {
    fn mixer_norm(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        if self.use_bcdt_rms {
            rms_norm_no_scale(x, self.mixer_rms_eps)
        } else {
            mlxcel_core::copy(x)
        }
    }

    fn conv(&self, conv_input: &MlxArray, cache: Option<&mut MambaCache>) -> UniquePtr<MlxArray> {
        let k = self.conv_kernel_size;
        let shape = mlxcel_core::array_shape(conv_input);

        // Get or create conv state
        let padded_input = if let Some(c) = cache.as_ref() {
            if let Some(ref conv_state) = c.conv_state {
                concatenate(conv_state.as_ref().unwrap(), conv_input, 1)
            } else {
                let pad_arr = mlxcel_core::zeros(
                    &[shape[0], (k - 1) as i32, shape[2]],
                    mlxcel_core::array_dtype(conv_input),
                );
                concatenate(&pad_arr, conv_input, 1)
            }
        } else {
            let pad_arr = mlxcel_core::zeros(
                &[shape[0], (k - 1) as i32, shape[2]],
                mlxcel_core::array_dtype(conv_input),
            );
            concatenate(&pad_arr, conv_input, 1)
        };

        // Update conv cache.
        // Wrap slice in contiguous() to force MLX to materialize a fresh,
        // independent buffer. Without this, the slice is a lazy view that
        // retains a reference to the full padded_input allocation, causing a
        // memory leak proportional to the sequence length.
        if let Some(c) = cache {
            let n_keep = k - 1;
            let padded_shape = mlxcel_core::array_shape(&padded_input);
            let len = padded_shape[1] as usize;
            let tail = slice_axis(&padded_input, 1, (len - n_keep) as i32, len as i32);
            c.conv_state = Some(mlxcel_core::contiguous(&tail, false));
        }

        // Depthwise conv1d
        let conv_out = mlxcel_core::conv1d(
            &padded_input,
            &self.conv_weight,
            1,
            0,
            1,
            self.intermediate_size as i32,
        );

        let conv_out = if let Some(ref b) = self.conv_bias {
            // Reshape bias for broadcasting
            let b_reshaped = mlxcel_core::reshape(b, &[1, 1, -1]);
            mlxcel_core::add(&conv_out, &b_reshaped)
        } else {
            conv_out
        };

        silu(&conv_out)
    }

    fn ssm_step(
        &self,
        x: &MlxArray,
        a: &MlxArray,
        state: Option<&MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let d = &self.d_param;

        // delta_bc = x_proj(x)
        let delta_bc = self.x_proj.forward(x);

        // Split into delta, B, C
        let delta_raw = slice_axis(&delta_bc, -1, 0, self.time_step_rank as i32);
        let b_raw = slice_axis(
            &delta_bc,
            -1,
            self.time_step_rank as i32,
            (self.time_step_rank + self.state_size) as i32,
        );
        let c_raw = slice_axis(
            &delta_bc,
            -1,
            (self.time_step_rank + self.state_size) as i32,
            -1,
        );

        // Apply mixer_norm if needed.
        // Python reference applies mixer_norm TWICE when use_bcdt_rms is true:
        // 1. During the map() over split outputs
        // 2. Again in the `if self.use_bcdt_rms:` block
        let delta_normed = self.mixer_norm(&self.mixer_norm(&delta_raw));
        let b_normed = self.mixer_norm(&self.mixer_norm(&b_raw));
        let c_normed = self.mixer_norm(&self.mixer_norm(&c_raw));

        // delta = softplus(dt_proj(delta))
        let dt_out = self.dt_proj.forward(&delta_normed);
        let delta = mlxcel_core::softplus(&dt_out);

        // new_state = (delta * x).unsqueeze(-1) * B.unsqueeze(1)
        let delta_x = mlxcel_core::multiply(&delta, x);
        let delta_x_exp = mlxcel_core::expand_dims(&delta_x, -1);
        let b_exp = mlxcel_core::expand_dims(&b_normed, 1);
        let mut new_state = mlxcel_core::multiply(&delta_x_exp, &b_exp);

        // If we have previous state: new_state += state * exp(delta.unsqueeze(-1) * A)
        if let Some(prev_state) = state {
            let delta_exp = mlxcel_core::expand_dims(&delta, -1);
            let delta_a = mlxcel_core::multiply(&delta_exp, a);
            let decay = mlxcel_core::exp(&delta_a);
            let state_contrib = mlxcel_core::multiply(prev_state, &decay);
            new_state = mlxcel_core::add(&new_state, &state_contrib);
        }

        // y = (new_state @ C.unsqueeze(-1)).squeeze(-1)
        let c_exp = mlxcel_core::expand_dims(&c_normed, -1);
        let y = mlxcel_core::matmul(&new_state, &c_exp);
        let y = mlxcel_core::squeeze_axis(&y, -1);

        // y = y + D * x
        let d_contrib = mlxcel_core::multiply(d, x);
        let y = mlxcel_core::add(&y, &d_contrib);

        (y, new_state)
    }

    pub fn forward(&self, x: &MlxArray, mut cache: Option<&mut MambaCache>) -> UniquePtr<MlxArray> {
        // x: [B, T, hidden_size]
        let shape = mlxcel_core::array_shape(x);
        let t = shape[1] as usize;

        // in_proj: [B, T, hidden_size] -> [B, T, intermediate_size * 2]
        let xz = self.in_proj.forward(x);
        let x_part = slice_axis(&xz, -1, 0, self.intermediate_size as i32);
        let z = slice_axis(&xz, -1, self.intermediate_size as i32, -1);

        // Conv1d with caching
        let x_conv = self.conv(&x_part, cache.as_deref_mut());

        // A = -exp(A_log)
        let a = mlxcel_core::negative(&mlxcel_core::exp(&self.a_log));

        // Process sequence step by step
        let state_cache = cache
            .as_ref()
            .and_then(|c| c.ssm_state.as_ref())
            .and_then(|s| s.as_ref());

        // For single-token decode (t=1), avoid copy + stack overhead
        let (y, current_state) = if t == 1 {
            let x_t = mlxcel_core::squeeze_axis(&x_conv, 1);
            let state_ref = state_cache;
            let (y_t, new_state) = self.ssm_step(&x_t, &a, state_ref);
            let y = mlxcel_core::expand_dims(&y_t, 1); // restore seq dim
            (y, Some(new_state))
        } else {
            let mut current_state = state_cache.map(mlxcel_core::copy);
            let mut y_steps = Vec::with_capacity(t);
            for t_idx in 0..t {
                let x_t = slice_axis(&x_conv, 1, t_idx as i32, (t_idx + 1) as i32);
                let x_t = mlxcel_core::squeeze_axis(&x_t, 1);
                let state_ref = current_state.as_ref().and_then(|s| s.as_ref());
                let (y_t, new_state) = self.ssm_step(&x_t, &a, state_ref);
                y_steps.push(y_t);
                current_state = Some(new_state);
            }
            let y_ptrs: Vec<*const MlxArray> = y_steps
                .iter()
                .map(|arr| arr.as_ref().unwrap() as *const MlxArray)
                .collect();
            (mlxcel_core::stack(&y_ptrs, 1), current_state)
        };

        // z = out_proj(silu(z) * y)
        let z_activated = silu(&z);
        let z_y = mlxcel_core::multiply(&z_activated, &y);
        let output = self.out_proj.forward(&z_y);

        // Update cache
        if let Some(c) = cache {
            c.ssm_state = current_state;
        }

        output
    }
}

/// Residual block wrapping MambaBlock
pub struct ResidualBlock {
    mixer: MambaBlock,
    norm: RMSNorm,
}

impl ResidualBlock {
    pub fn forward(&self, x: &MlxArray, cache: Option<&mut MambaCache>) -> UniquePtr<MlxArray> {
        let normed = self.norm.forward(x);
        let out = self.mixer.forward(&normed, cache);
        mlxcel_core::add(x, &out)
    }
}

// Full Mamba Model.

#[allow(dead_code)]
pub struct MambaModel {
    config: MambaConfig,
    embeddings: UnifiedEmbedding,
    layers: Vec<ResidualBlock>,
    norm_f: RMSNorm,
    lm_head: Option<UnifiedLinear>,
    /// Internal caches for LanguageModel trait compatibility
    sequence_state: ModelOwnedSequenceState<MambaCache>,
}

impl MambaModel {
    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    pub fn make_caches(&self) -> Vec<MambaCache> {
        (0..self.config.num_hidden_layers)
            .map(|_| MambaCache::new())
            .collect()
    }

    pub fn forward_with_caches(
        &self,
        x: &MlxArray,
        caches: &mut [MambaCache],
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embeddings.forward(x);

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, Some(cache));
        }

        let h = self.norm_f.forward(&h);

        if let Some(ref head) = self.lm_head {
            head.forward(&h)
        } else {
            self.embeddings.as_linear(&h)
        }
    }

    /// Load model from safetensors files
    pub fn load(model_path: &str) -> Result<(Self, MambaConfig), Box<dyn std::error::Error>> {
        let path = Path::new(model_path);

        // Load config
        println!("[Mamba] Loading config...");
        let config_path = path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config_str = super::sanitize_config_json(&config_str);
        let mut config: MambaConfig = serde_json::from_str(&config_str)?;
        config.compute_time_step_rank();
        println!(
            "[Mamba] Config loaded: {} layers, hidden_size={}, state_size={}",
            config.num_hidden_layers, config.hidden_size, config.state_size
        );

        // Load weights
        println!("[Mamba] Loading weights from safetensors...");
        let weights = crate::models::load_text_weights(path, None)?;

        // Process weights (handle conv1d weight transpose)
        let weights = Self::sanitize_weights(weights, &config);

        // Build model
        println!("[Mamba] Building model...");
        let model = Self::from_weights(config.clone(), weights)?;

        println!("[Mamba] Model loaded successfully");
        Ok((model, config))
    }

    fn sanitize_weights(mut weights: WeightMap, config: &MambaConfig) -> WeightMap {
        // Handle conv1d weight transpose
        let keys: Vec<String> = weights.keys().cloned().collect();
        for k in keys {
            if k.contains("conv1d.weight")
                && let Some(v) = weights.get(&k)
            {
                let shape = mlxcel_core::array_shape(v);
                if shape.len() >= 3 && shape[shape.len() - 1] != 1 {
                    // swap axes -1 and -2 (equivalent to moveaxis from -1 to -2)
                    let transposed = mlxcel_core::swap_axes(v, -1, -2);
                    weights.insert(k, transposed);
                }
            }
        }

        // Remove lm_head if tie_word_embeddings
        if config.tie_word_embeddings {
            weights.remove("lm_head.weight");
        }

        weights
    }

    pub fn from_weights(
        config: MambaConfig,
        mut weights: WeightMap,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let group_size = config.group_size();
        let bits = config.bits();

        // Build embeddings (auto-detect quantization)
        let embed_weight = weights
            .remove("backbone.embeddings.weight")
            .or_else(|| weights.remove("model.embed_tokens.weight"))
            .ok_or("Missing embedding weight")?;
        let embed_scales = weights
            .remove("backbone.embeddings.scales")
            .or_else(|| weights.remove("model.embed_tokens.scales"));
        let embed_biases = weights
            .remove("backbone.embeddings.biases")
            .or_else(|| weights.remove("model.embed_tokens.biases"));

        let embeddings = if let (Some(scales), Some(biases)) = (embed_scales, embed_biases) {
            UnifiedEmbedding::Quantized(mlxcel_core::layers::QuantizedEmbedding::new(
                embed_weight,
                scales,
                biases,
                group_size,
                bits,
            ))
        } else {
            UnifiedEmbedding::Regular(mlxcel_core::layers::Embedding::new(embed_weight))
        };

        // Build layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("backbone.layers.{}", i);

            // Conv1d weights
            let conv_weight = weights
                .remove(&format!("{}.mixer.conv1d.weight", prefix))
                .ok_or(format!("Missing conv1d weight for layer {}", i))?;
            let conv_bias = weights.remove(&format!("{}.mixer.conv1d.bias", prefix));

            // in_proj (quantized)
            let in_proj = UnifiedLinear::from_weights(
                &weights,
                &format!("{}.mixer.in_proj", prefix),
                group_size,
                bits,
            )?;

            // x_proj (quantized)
            let x_proj = UnifiedLinear::from_weights(
                &weights,
                &format!("{}.mixer.x_proj", prefix),
                group_size,
                bits,
            )?;

            // dt_proj (quantized)
            let dt_proj = UnifiedLinear::from_weights(
                &weights,
                &format!("{}.mixer.dt_proj", prefix),
                group_size,
                bits,
            )?;

            // out_proj (quantized)
            let out_proj = UnifiedLinear::from_weights(
                &weights,
                &format!("{}.mixer.out_proj", prefix),
                group_size,
                bits,
            )?;

            // SSM parameters
            let a_log = weights
                .remove(&format!("{}.mixer.A_log", prefix))
                .ok_or(format!("Missing A_log for layer {}", i))?;
            let d_param = weights
                .remove(&format!("{}.mixer.D", prefix))
                .ok_or(format!("Missing D for layer {}", i))?;

            let mixer = MambaBlock {
                hidden_size: config.hidden_size,
                intermediate_size: config.intermediate_size,
                state_size: config.state_size,
                conv_kernel_size: config.conv_kernel,
                time_step_rank: config.time_step_rank,
                use_bcdt_rms: config.use_bcdt_rms,
                mixer_rms_eps: config.mixer_rms_eps,
                conv_weight,
                conv_bias,
                in_proj,
                x_proj,
                dt_proj,
                out_proj,
                a_log,
                d_param,
            };

            // ResidualBlock norm
            let block_norm_weight = weights
                .remove(&format!("{}.norm.weight", prefix))
                .ok_or(format!("Missing block norm weight for layer {}", i))?;
            let block_norm = RMSNorm::new(block_norm_weight, config.rms_norm_eps);

            layers.push(ResidualBlock {
                mixer,
                norm: block_norm,
            });
        }

        // Final norm
        let norm_f_weight = weights
            .remove("backbone.norm_f.weight")
            .ok_or("Missing final norm weight")?;
        let norm_f = RMSNorm::new(norm_f_weight, config.rms_norm_eps);

        // LM head (if not tie_word_embeddings)
        let lm_head = if !config.tie_word_embeddings {
            Some(UnifiedLinear::from_weights(
                &weights, "lm_head", group_size, bits,
            )?)
        } else {
            None
        };

        // Create internal caches for LanguageModel trait compatibility
        let internal_caches: Vec<MambaCache> = (0..config.num_hidden_layers)
            .map(|_| MambaCache::new())
            .collect();

        Ok(Self {
            config,
            embeddings,
            layers,
            norm_f,
            lm_head,
            sequence_state: ModelOwnedSequenceState::new(internal_caches),
        })
    }
}

// LanguageModel trait implementation.
impl LanguageModel for MambaModel {
    fn forward(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Mamba v1 uses internal caching (MambaCache) instead of KV cache.
        // We use model-owned state to isolate scheduler sequence ids while
        // retaining a fallback slot for CLI / benchmark paths.
        self.sequence_state
            .with_sequence_state(None, |internal| self.forward_with_caches(input, internal))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        // Reset fallback internal caches
        self.sequence_state
            .replace_internal(MambaModel::make_caches(self));
        // Return dummy KV caches for trait compatibility
        (0..self.config.num_hidden_layers)
            .map(|_| KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn supports_padded_prefill(&self) -> bool {
        false // Padding tokens corrupt Mamba recurrent state
    }

    fn supports_batching(&self) -> bool {
        false // Mamba uses internal MambaCache state, not compatible with per-sequence KV isolation
    }

    fn prepare_sequence_state(&self, seq_id: mlxcel_core::cache::SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, MambaModel::make_caches(self));
    }

    fn release_sequence_state_by_id(&self, seq_id: mlxcel_core::cache::SequenceId) {
        self.sequence_state.release_sequence_state(seq_id);
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<mlxcel_core::cache::SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || MambaModel::make_caches(self),
            |internal| self.forward_with_caches(input_ids, internal),
        )
    }

    fn supports_snapshot_reuse(&self) -> bool {
        true
    }

    fn snapshot_sequence_state(
        &self,
        seq_id: mlxcel_core::cache::SequenceId,
        token_len: usize,
    ) -> Option<mlxcel_core::generate::ModelStateSnapshot> {
        self.sequence_state
            .with_sequence_state_ref(seq_id, |state| {
                let mut snapshot =
                    mlxcel_core::generate::ModelStateSnapshot::new("mamba", token_len);
                for (idx, cache) in state.iter().enumerate() {
                    cache.snapshot_into(&mut snapshot, &format!("layer{idx}"));
                }
                snapshot
            })
    }

    fn restore_sequence_state(
        &self,
        seq_id: mlxcel_core::cache::SequenceId,
        snapshot: &mlxcel_core::generate::ModelStateSnapshot,
    ) -> Result<(), String> {
        if snapshot.family() != "mamba" {
            return Err(format!(
                "cannot restore {} snapshot into Mamba",
                snapshot.family()
            ));
        }
        let mut state = MambaModel::make_caches(self);
        for (idx, cache) in state.iter_mut().enumerate() {
            cache.restore_from(snapshot, &format!("layer{idx}"));
        }
        self.sequence_state.replace_sequence_state(seq_id, state);
        Ok(())
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        parse_eos_token_ids(&self.config.eos_token_id, 2)
    }
}

#[cfg(test)]
#[path = "mamba_tests.rs"]
mod tests;
