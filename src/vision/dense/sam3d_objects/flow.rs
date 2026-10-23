// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Structure/pose and sparse-latent flow networks.
use super::{
    bundle::array,
    config::Config,
    layers::{Array, Linear, MultiBlock, PreparedContext, TimestepEmbedder, layer_norm},
};
use anyhow::{Result, ensure};
use mlxcel_core::{self as mx, MlxArray, weights::WeightMap};
use std::collections::BTreeMap;
#[path = "flow_latent.rs"]
mod latent;
pub use latent::LatentFlow;
pub type State = BTreeMap<String, Array>;
pub type StructureCondition = Vec<BTreeMap<String, PreparedContext>>;
/// The source insertion order also specifies the upstream RNG draw order.
pub const LATENTS: [(&str, usize); 5] = [
    ("6drotation_normalized", 6),
    ("scale", 3),
    ("shape", 8),
    ("translation", 3),
    ("translation_scale", 1),
];
const POSE_NAMES: [&str; 4] = [
    "6drotation_normalized",
    "translation",
    "scale",
    "translation_scale",
];
struct Latent {
    input: Linear,
    output: Linear,
    position: Array,
}
impl Latent {
    fn from_weights(
        w: &WeightMap,
        p: &str,
        input_channels: usize,
        channels: usize,
        length: usize,
    ) -> Result<Self> {
        let input_dim = i32::try_from(input_channels)?;
        let hidden_dim = i32::try_from(channels)?;
        let length = i32::try_from(length)?;
        ensure!(
            input_dim > 0 && hidden_dim > 0 && length > 0,
            "invalid latent dimensions {p}"
        );
        let position = array(w, &format!("{p}.pos_emb"))?;
        ensure!(
            mx::array_shape(&position) == [length, hidden_dim],
            "invalid latent position {p}.pos_emb: expected [{length}, {hidden_dim}], got {:?}",
            mx::array_shape(&position)
        );
        let input = Linear::from_weights(w, &format!("{p}.input_layer"))?;
        let output = Linear::from_weights(w, &format!("{p}.out_layer"))?;
        ensure!(
            mx::array_shape(&input.weight) == [hidden_dim, input_dim]
                && mx::array_shape(&output.weight) == [input_dim, hidden_dim],
            "latent projection dimensions do not match configuration: {p}"
        );
        Ok(Self {
            input,
            output,
            position,
        })
    }
    fn encode(&self, x: &MlxArray) -> Array {
        let s = mx::array_shape(&self.position);
        mx::add(
            &self
                .input
                .forward(&mx::astype(x, mx::array_dtype(&self.input.weight))),
            &mx::reshape(&self.position, &[1, s[0], s[1]]),
        )
    }
    fn decode(&self, x: &MlxArray) -> Array {
        self.output.forward(&layer_norm(x, 1e-5))
    }
}
pub struct StructureFlow {
    time: TimestepEmbedder,
    delta: TimestepEmbedder,
    latents: BTreeMap<String, Latent>,
    blocks: Vec<MultiBlock>,
}
impl StructureFlow {
    pub fn from_weights(w: &WeightMap, p: &str, c: &Config) -> Result<Self> {
        let shape_tokens = c
            .latent_resolution
            .checked_pow(3)
            .ok_or_else(|| anyhow::anyhow!("latent token count overflow"))?;
        let latents = LATENTS
            .iter()
            .map(|(n, input_channels)| {
                Ok((
                    (*n).to_owned(),
                    Latent::from_weights(
                        w,
                        &format!("{p}.latent_mapping.{n}"),
                        *input_channels,
                        c.hidden_size,
                        if *n == "shape" { shape_tokens } else { 1 },
                    )?,
                ))
            })
            .collect::<Result<_>>()?;
        let blocks = (0..c.num_blocks)
            .map(|i| {
                MultiBlock::from_weights(w, &format!("{p}.blocks.{i}"), c.num_heads, c.hidden_size)
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            time: TimestepEmbedder::from_weights(w, &format!("{p}.t_embedder"))?,
            delta: TimestepEmbedder::from_weights(w, &format!("{p}.d_embedder"))?,
            latents,
            blocks,
        })
    }
    pub fn prepare_condition(&self, tokens: Option<&MlxArray>) -> Result<StructureCondition> {
        self.blocks.iter().map(|b| b.prepare(tokens)).collect()
    }
    fn encode(
        &self,
        latents: &State,
        time: &MlxArray,
        delta: Option<&MlxArray>,
    ) -> Result<(State, Array)> {
        ensure!(
            LATENTS.iter().all(|(n, _)| latents.contains_key(*n)),
            "missing structure latent"
        );
        let mut emb = self.time.forward(time);
        if let Some(delta) = delta {
            emb = mx::add(&emb, &self.delta.forward(delta));
        }
        let encoded: State = self
            .latents
            .iter()
            .map(|(n, m)| (n.clone(), m.encode(&latents[n])))
            .collect();
        let pose = mx::concatenate_many(
            &POSE_NAMES.iter().map(|n| &*encoded[*n]).collect::<Vec<_>>(),
            1,
        );
        Ok((
            BTreeMap::from([
                ("6drotation_normalized".into(), pose),
                ("shape".into(), mx::copy(&encoded["shape"])),
            ]),
            emb,
        ))
    }
    fn decode(
        &self,
        encoded: &State,
        emb: &MlxArray,
        condition: &StructureCondition,
    ) -> Result<State> {
        ensure!(
            condition.len() == self.blocks.len(),
            "structure context depth mismatch"
        );
        let mut h: State = encoded
            .iter()
            .map(|(n, v)| (n.clone(), mx::copy(v)))
            .collect();
        for (block, ctx) in self.blocks.iter().zip(condition) {
            h = block.forward(&h, emb, ctx)?;
        }
        let mut velocity = State::new();
        for (i, n) in POSE_NAMES.iter().enumerate() {
            velocity.insert(
                (*n).into(),
                self.latents[*n].decode(&mx::utils::slice_axis(
                    &h["6drotation_normalized"],
                    1,
                    i as i32,
                    i as i32 + 1,
                )),
            );
        }
        velocity.insert("shape".into(), self.latents["shape"].decode(&h["shape"]));
        Ok(velocity)
    }
    pub fn forward(
        &self,
        x: &State,
        time: &MlxArray,
        condition: &StructureCondition,
        delta: Option<&MlxArray>,
    ) -> Result<State> {
        let (encoded, emb) = self.encode(x, time, delta)?;
        self.decode(&encoded, &emb, condition)
    }
    pub fn guided(
        &self,
        x: &State,
        time: &MlxArray,
        condition: &StructureCondition,
        uncondition: &StructureCondition,
        delta: Option<&MlxArray>,
    ) -> Result<(State, State)> {
        let (encoded, emb) = self.encode(x, time, delta)?;
        Ok((
            self.decode(&encoded, &emb, condition)?,
            self.decode(&encoded, &emb, uncondition)?,
        ))
    }
}

#[cfg(test)]
#[path = "flow_tests.rs"]
mod tests;
