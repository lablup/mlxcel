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

//! Per-token router selection and per-layer hidden states for the Bailing
//! MoE decoder, as Ming-Image's conditioner runs it.
//!
//! A port of `MingMoeBlock`, `MingDecoderLayer` and `MingMLLM` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ming_image/text_encoder.py>
//! (`#L23-L95`, mlx-vlm `73d366ec`). Ming's BailingMoeV2 backbone gives every
//! MoE layer a second router, `image_gate`, with the shape and routing rule of
//! the text `gate`. Both routers score every token; a boolean mask then picks,
//! per token, the image router's experts and weights for the image-patch
//! (learned query) tokens and the text router's for the rest. The routed
//! experts and the shared expert are the layer's own.
//!
//! Everything here is additive. [`BailingMoeSparseBlock::forward`],
//! [`DecoderLayer::forward`] and [`BailingMoeModel::forward`] are untouched,
//! so a plain Bailing MoE checkpoint decodes exactly as before; the image
//! routers live in [`ImageRouters`], next to the model rather than inside its
//! layers, and are only consulted by the entry points below.
//!
//! Used by: MingImage (`crate::imagegen::ming_image::text_encoder`).

use mlxcel_core::layers::KVCache;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{
    BailingMoeGate, BailingMoeModel, BailingMoeSparseBlock, DecoderLayer, FeedForward, ModelArgs,
    router_prefix_named,
};
use crate::models::gpt2::dim_eq;
use crate::models::switch_layers::moe_weighted_sum;

/// Module name of the second router under each MoE layer's `mlp`.
pub const IMAGE_ROUTER: &str = "image_gate";

/// The `image_gate` router of every MoE layer (`None` for the dense prefix
/// layers), indexed like [`BailingMoeModel::layers`].
pub struct ImageRouters {
    gates: Vec<Option<BailingMoeGate>>,
}

impl ImageRouters {
    /// Load `model.layers.{i}.mlp.image_gate` for every MoE layer of `args`,
    /// checking each router's shape the way `validate_weights` checks the
    /// text router (the matmul and the bias add would otherwise abort across
    /// the FFI at the first forward).
    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let mut gates = Vec::with_capacity(args.num_hidden_layers);
        for layer in 0..args.num_hidden_layers {
            if !args.is_moe_layer(layer) {
                gates.push(None);
                continue;
            }
            let mlp = format!("model.layers.{layer}.mlp");
            let prefix = router_prefix_named(weights, &mlp, IMAGE_ROUTER)?;
            super::validate_projection(
                weights,
                &prefix,
                args.num_experts(),
                args.hidden_size,
                args.group_size(),
                args.bits(),
            )?;
            if args.moe_router_enable_expert_bias
                && let Some(bias) = weights.get(&format!("{mlp}.{IMAGE_ROUTER}.expert_bias"))
            {
                let shape = mlxcel_core::array_shape(bias);
                if shape.len() != 1 || !dim_eq(shape[0], args.num_experts()) {
                    return Err(format!(
                        "unexpected {mlp}.{IMAGE_ROUTER}.expert_bias shape {shape:?}: expected [{}]",
                        args.num_experts()
                    ));
                }
            }
            gates.push(Some(BailingMoeGate::from_weights_named(
                weights,
                args,
                &mlp,
                IMAGE_ROUTER,
            )?));
        }
        Ok(Self { gates })
    }

    /// Wrap per-layer routers built elsewhere (tests).
    #[must_use]
    pub fn from_gates(gates: Vec<Option<BailingMoeGate>>) -> Self {
        Self { gates }
    }

    /// The image router of layer `layer`, `None` for a dense layer.
    #[must_use]
    pub fn get(&self, layer: usize) -> Option<&BailingMoeGate> {
        self.gates.get(layer).and_then(Option::as_ref)
    }

    /// Number of layers covered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.gates.len()
    }

    /// Whether no layer is covered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }
}

/// `x` (`[..., hidden]`) as `[tokens, hidden]`.
fn flatten_tokens(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let hidden = *shape.last().unwrap_or(&0);
    let tokens: i32 = shape[..shape.len().saturating_sub(1)].iter().product();
    mlxcel_core::reshape(x, &[tokens, hidden])
}

impl BailingMoeSparseBlock {
    /// Upstream `MingMoeBlock.__call__`: both routers score every token,
    /// `where(image_mask, image, text)` picks each token's expert indices and
    /// weights, and the routed experts and the shared expert run as in
    /// [`Self::forward`].
    ///
    /// `image_mask` is a boolean `[..]` with one entry per token of `x`
    /// (`[B, L]` for `x` of `[B, L, hidden]`). Always the gather path: the
    /// fused single-token kernel serves decode, which this never runs.
    pub fn forward_image_routed(
        &self,
        x: &MlxArray,
        image_gate: &BailingMoeGate,
        image_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let orig_shape = mlxcel_core::array_shape(x);
        let x_flat = flatten_tokens(x);
        let tokens = mlxcel_core::array_shape(&x_flat)[0];

        let (text_idx, text_weight) = self.gate.forward(&x_flat);
        let (image_idx, image_weight) = image_gate.forward(&x_flat);
        let select = mlxcel_core::reshape(image_mask, &[tokens, 1]);
        let indices = mlxcel_core::where_cond(&select, &image_idx, &text_idx);
        let scores = mlxcel_core::where_cond(&select, &image_weight, &text_weight);

        let expert_out = self.switch_mlp.forward(&x_flat, &indices);
        let routed = moe_weighted_sum(&expert_out, &scores, mlxcel_core::array_dtype(&x_flat));
        let routed = mlxcel_core::reshape(&routed, &orig_shape);
        match &self.shared_experts {
            Some(shared) => mlxcel_core::add(&routed, &shared.forward(x)),
            None => routed,
        }
    }
}

impl FeedForward {
    /// [`Self::forward`] with the image router of this layer: a dense layer
    /// (or a missing router) runs its plain path.
    pub fn forward_image_routed(
        &self,
        x: &MlxArray,
        image_gate: Option<&BailingMoeGate>,
        image_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        match (self, image_gate) {
            (Self::Sparse(block), Some(gate)) => block.forward_image_routed(x, gate, image_mask),
            _ => self.forward(x),
        }
    }
}

impl DecoderLayer {
    /// Upstream `MingDecoderLayer.__call__`: [`Self::forward`] with the
    /// per-token router choice in the feed-forward.
    pub fn forward_image_routed(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        image_gate: Option<&BailingMoeGate>,
        image_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.attention.forward(&normed, cache, mask);
        let h = mlxcel_core::add(x, &attn_out);
        let normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = self
            .mlp
            .forward_image_routed(&normed, image_gate, image_mask);
        mlxcel_core::add(&h, &mlp_out)
    }
}

impl BailingMoeModel {
    /// Upstream `MingMLLM.__call__`: run the decoder causally over
    /// `inputs_embeds` (`[B, L, hidden]`) and return `num_layers + 1` states,
    /// the input of every layer followed by the final normed output.
    ///
    /// With `routing = Some((routers, image_mask))` every MoE layer picks its
    /// router per token (see [`BailingMoeSparseBlock::forward_image_routed`]);
    /// with `None` the layers run their plain [`DecoderLayer::forward`], so
    /// the last state equals the normed input of [`Self::forward`]'s head.
    pub fn hidden_states_from_embeds(
        &self,
        inputs_embeds: &MlxArray,
        routing: Option<(&ImageRouters, &MlxArray)>,
    ) -> Result<Vec<UniquePtr<MlxArray>>, String> {
        if let Some((routers, _)) = routing
            && routers.len() != self.layers.len()
        {
            return Err(format!(
                "Bailing MoE image routers cover {} layers, the model has {}",
                routers.len(),
                self.layers.len()
            ));
        }
        let embeds_shape = mlxcel_core::array_shape(inputs_embeds);
        if embeds_shape.len() != 3 {
            return Err(format!(
                "Bailing MoE inputs_embeds must be [B, L, hidden], got {embeds_shape:?}"
            ));
        }
        if let Some((_, image_mask)) = routing {
            // `forward_image_routed` reshapes the mask to one entry per token;
            // a mismatch would throw across the FFI instead of erroring.
            let mask_shape = mlxcel_core::array_shape(image_mask);
            if mask_shape != embeds_shape[..2] {
                return Err(format!(
                    "Bailing MoE image mask {mask_shape:?} does not match the [B, L] = {:?} of inputs_embeds",
                    &embeds_shape[..2]
                ));
            }
            if mlxcel_core::array_dtype(image_mask) != mlxcel_core::dtype::BOOL {
                return Err("Bailing MoE image mask must be boolean".into());
            }
        }
        let mut states = Vec::with_capacity(self.layers.len() + 1);
        let mut h = mlxcel_core::copy(inputs_embeds);
        for (i, layer) in self.layers.iter().enumerate() {
            // A fresh cache per layer at offset 0: one causal prefill.
            let mut cache = KVCache::new();
            let next = match routing {
                Some((routers, image_mask)) => {
                    layer.forward_image_routed(&h, &mut cache, None, routers.get(i), image_mask)
                }
                None => layer.forward(&h, &mut cache, None),
            };
            states.push(h);
            h = next;
        }
        states.push(self.norm.forward(&h));
        Ok(states)
    }
}

#[cfg(test)]
#[path = "bailing_moe_image_router_tests.rs"]
mod tests;
