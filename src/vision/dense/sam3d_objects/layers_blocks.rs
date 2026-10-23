// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Time-modulated transformer blocks and asymmetric shape/pose attention.
use super::attention::component;
use super::{
    Array, Attention, FeedForward, HeadNorm, LayerNorm, Linear, PreparedContext, attend,
    gated_residual, layer_norm, modulate,
};
use anyhow::{Result, ensure};
use mlxcel_core::{self as mx, MlxArray, weights::WeightMap};
use std::collections::BTreeMap;

fn modulation(linear: &Linear, time: &MlxArray) -> Vec<Array> {
    let x = linear.forward(&mx::compiled_silu(time));
    let s = mx::array_shape(&x);
    let c = s[1] / 6;
    (0..6)
        .map(|i| mx::reshape(&mx::slice_last_dim(&x, i * c, (i + 1) * c), &[s[0], 1, c]))
        .collect()
}
fn cross(attn: &Attention, norm: &LayerNorm, x: &MlxArray, ctx: &PreparedContext) -> Result<Array> {
    Ok(match ctx {
        PreparedContext::Bias(b) => mx::add(x, b),
        _ => mx::add(x, &*attn.forward(&norm.forward(x), Some(ctx))?),
    })
}
pub struct ModulatedBlock {
    modulation: Linear,
    norm: LayerNorm,
    self_attn: Attention,
    pub cross_attn: Attention,
    mlp: FeedForward,
}
impl ModulatedBlock {
    pub fn from_weights(w: &WeightMap, p: &str, heads: usize) -> Result<Self> {
        Ok(Self {
            modulation: Linear::from_weights(w, &format!("{p}.adaLN_modulation.1"))?,
            norm: LayerNorm::from_weights(w, &format!("{p}.norm2"), 1e-6)?,
            self_attn: Attention::from_weights(w, &format!("{p}.self_attn"), heads, false, true)?,
            cross_attn: Attention::from_weights(w, &format!("{p}.cross_attn"), heads, true, false)?,
            mlp: FeedForward::from_weights(w, &format!("{p}.mlp"))?,
        })
    }
    pub fn forward(&self, x: &MlxArray, time: &MlxArray, ctx: &PreparedContext) -> Result<Array> {
        let m = modulation(&self.modulation, time);
        let h = self
            .self_attn
            .forward(&*modulate(&layer_norm(x, 1e-6), &m[1], &m[0])?, None)?;
        let x = gated_residual(x, &h, &m[2])?;
        let x = cross(&self.cross_attn, &self.norm, &x, ctx)?;
        let h = self
            .mlp
            .forward(&*modulate(&layer_norm(&x, 1e-6), &m[4], &m[3])?)?;
        gated_residual(&x, &h, &m[5])
    }
}
const NAMES: [&str; 2] = ["shape", "6drotation_normalized"];
struct MultiStream {
    qkv: Linear,
    out: Linear,
    qnorm: HeadNorm,
    knorm: HeadNorm,
    norm: LayerNorm,
    cross: Attention,
    mlp: FeedForward,
}
pub struct MultiBlock {
    modulation: Linear,
    streams: BTreeMap<String, MultiStream>,
    heads: i32,
}
impl MultiBlock {
    pub fn from_weights(w: &WeightMap, p: &str, heads: usize, channels: usize) -> Result<Self> {
        ensure!(
            heads > 0 && channels > 0 && channels.is_multiple_of(heads),
            "invalid multi-stream head dimensions"
        );
        let mut streams = BTreeMap::new();
        for n in NAMES {
            streams.insert(
                n.to_owned(),
                MultiStream {
                    qkv: Linear::from_weights(w, &format!("{p}.self_attn.to_qkv.{n}"))?,
                    out: Linear::from_weights(w, &format!("{p}.self_attn.to_out.{n}"))?,
                    qnorm: HeadNorm::from_weights(
                        w,
                        &format!("{p}.self_attn.q_rms_norm.{n}"),
                        heads,
                        channels / heads,
                    )?,
                    knorm: HeadNorm::from_weights(
                        w,
                        &format!("{p}.self_attn.k_rms_norm.{n}"),
                        heads,
                        channels / heads,
                    )?,
                    norm: LayerNorm::from_weights(w, &format!("{p}.norm2.{n}"), 1e-6)?,
                    cross: Attention::from_weights(
                        w,
                        &format!("{p}.cross_attn.{n}"),
                        heads,
                        true,
                        false,
                    )?,
                    mlp: FeedForward::from_weights(w, &format!("{p}.mlp.{n}"))?,
                },
            );
        }
        Ok(Self {
            modulation: Linear::from_weights(w, &format!("{p}.adaLN_modulation.1"))?,
            streams,
            heads: heads as i32,
        })
    }
    pub fn prepare(&self, tokens: Option<&MlxArray>) -> Result<BTreeMap<String, PreparedContext>> {
        self.streams
            .iter()
            .map(|(n, s)| Ok((n.clone(), s.cross.prepare(tokens)?)))
            .collect()
    }
    pub fn forward(
        &self,
        x: &BTreeMap<String, Array>,
        time: &MlxArray,
        ctx: &BTreeMap<String, PreparedContext>,
    ) -> Result<BTreeMap<String, Array>> {
        ensure!(
            NAMES
                .iter()
                .all(|n| x.contains_key(*n) && ctx.contains_key(*n)),
            "missing shape or pose stream"
        );
        let m = modulation(&self.modulation, time);
        let mut qkv = BTreeMap::new();
        for n in NAMES {
            let stream = &self.streams[n];
            let s = mx::array_shape(&x[n]);
            let h = modulate(&layer_norm(&x[n], 1e-6), &m[1], &m[0])?;
            let h = mx::reshape(&stream.qkv.forward(&h), &[s[0], s[1], 3, self.heads, -1]);
            qkv.insert(
                n,
                (
                    stream.qnorm.forward(&component(&h, 0))?,
                    stream.knorm.forward(&component(&h, 1))?,
                    component(&h, 2),
                ),
            );
        }
        let shape = &qkv["shape"];
        let pose = &qkv["6drotation_normalized"];
        let keys = mx::concatenate(&pose.1, &shape.1, 1);
        let vals = mx::concatenate(&pose.2, &shape.2, 1);
        let mut out = BTreeMap::new();
        for n in NAMES {
            let stream = &self.streams[n];
            let (q, k, v) = &qkv[n];
            let h = if n == "shape" {
                attend(q, k, v)
            } else {
                attend(q, &keys, &vals)
            };
            let h = gated_residual(&x[n], &stream.out.forward(&h), &m[2])?;
            let h = cross(&stream.cross, &stream.norm, &h, &ctx[n])?;
            let y = stream
                .mlp
                .forward(&*modulate(&layer_norm(&h, 1e-6), &m[4], &m[3])?)?;
            out.insert(n.to_owned(), gated_residual(&h, &y, &m[5])?);
        }
        Ok(out)
    }
}
