//! Shared Inkling decoder-layer primitives.
//!
//! The text target and the native MTP drafter execute the same attention,
//! short-convolution, residual, and cache code. The feed-forward plane is
//! injected because target layers may be sparse while every MTP block is dense.

mod attention;
mod cache;
mod dense_mlp;

use crate::layers::RMSNorm;
use crate::weights::WeightMap;
use crate::{MlxArray, UniquePtr};

pub use attention::{InklingAttention, InklingShortConv, banded_additive_mask, log_scaling_tau};
pub use cache::InklingLayerCache;
pub use dense_mlp::InklingDenseMlp;

/// Runtime parameters shared by one target or MTP decoder layer.
#[derive(Debug, Clone, PartialEq)]
pub struct InklingLayerSpec {
    pub hidden_size: usize,
    pub rms_norm_eps: f32,
    pub is_sliding: bool,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub sliding_window_size: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub log_scaling_n_floor: Option<usize>,
    pub log_scaling_alpha: f32,
    pub sconv_kernel_size: usize,
    pub dense_intermediate_size: usize,
    pub quantization_group_size: i32,
    pub quantization_bits: i32,
}

impl InklingLayerSpec {
    pub fn attention_heads(&self) -> (usize, usize, usize) {
        (
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
    }

    pub fn relative_extent(&self) -> usize {
        if self.is_sliding {
            self.sliding_window_size
        } else {
            self.rel_extent
        }
    }
}

/// Feed-forward contract used by the shared decoder shell.
pub trait InklingFeedForward {
    fn forward(&self, input: &MlxArray) -> UniquePtr<MlxArray>;
}

/// One Inkling decoder layer, shared by the text target and native MTP head.
pub struct InklingDecoderLayer<M> {
    attention: InklingAttention,
    mlp: M,
    input_norm: RMSNorm,
    post_attention_norm: RMSNorm,
    attention_conv: InklingShortConv,
    mlp_conv: InklingShortConv,
}

impl<M: InklingFeedForward> InklingDecoderLayer<M> {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        spec: &InklingLayerSpec,
        mlp: M,
    ) -> Result<Self, String> {
        Ok(Self {
            attention: InklingAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                spec,
            )?,
            mlp,
            input_norm: RMSNorm::new(
                weight(weights, &format!("{prefix}.input_layernorm.weight"))?,
                spec.rms_norm_eps,
            ),
            post_attention_norm: RMSNorm::new(
                weight(
                    weights,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                spec.rms_norm_eps,
            ),
            attention_conv: InklingShortConv::from_weights(
                weights,
                &format!("{prefix}.attn_sconv.conv.weight"),
                spec.sconv_kernel_size,
            )?,
            mlp_conv: InklingShortConv::from_weights(
                weights,
                &format!("{prefix}.mlp_sconv.conv.weight"),
                spec.sconv_kernel_size,
            )?,
        })
    }

    pub fn forward(&self, input: &MlxArray, cache: &mut InklingLayerCache) -> UniquePtr<MlxArray> {
        let normed = self.input_norm.forward(input);
        let attended = self.attention.forward(&normed, cache);
        let hidden = self
            .attention_conv
            .forward(&attended, &mut cache.conv[2], Some(input));
        let normed = self.post_attention_norm.forward(&hidden);
        let projected = self.mlp.forward(&normed);
        self.mlp_conv
            .forward(&projected, &mut cache.conv[3], Some(&hidden))
    }
}

pub(crate) fn weight(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|value| crate::copy(value))
        .ok_or_else(|| format!("Weight not found: {name}"))
}
