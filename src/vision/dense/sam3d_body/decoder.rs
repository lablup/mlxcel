// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Six-layer promptable decoder with explicit per-layer pose feedback.
use super::{
    config::Config,
    layers::{Array, Linear, Norm},
};
use anyhow::{Result, ensure};
use mlxcel_core::{self as mx, MlxArray, weights::WeightMap};
pub struct Ffn {
    first: Linear,
    last: Linear,
}
impl Ffn {
    pub fn from_weights(w: &WeightMap, p: &str) -> Result<Self> {
        Ok(Self {
            first: Linear::from_weights(w, &format!("{p}.layers.0.0"))?,
            last: Linear::from_weights(w, &format!("{p}.layers.1"))?,
        })
    }
    pub fn forward(&self, x: &MlxArray) -> Array {
        self.last.forward(&mx::maximum(
            &self.first.forward(x),
            &mx::full_f32(&[], 0.0, mx::dtype::FLOAT32),
        ))
    }
}
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    proj: Linear,
    heads: i32,
    dim: i32,
}
impl Attention {
    fn new(w: &WeightMap, p: &str, c: &Config) -> Result<Self> {
        Ok(Self {
            q: Linear::from_weights(w, &format!("{p}.q_proj"))?,
            k: Linear::from_weights(w, &format!("{p}.k_proj"))?,
            v: Linear::from_weights(w, &format!("{p}.v_proj"))?,
            proj: Linear::from_weights(w, &format!("{p}.proj"))?,
            heads: c.decoder_heads as i32,
            dim: c.decoder_head_dim as i32,
        })
    }
    fn forward(&self, q: &MlxArray, k: &MlxArray, v: &MlxArray) -> Array {
        let s = mx::array_shape(q);
        let project = |l: &Linear, x: &MlxArray| {
            mx::transpose_axes(
                &mx::reshape(&l.forward(x), &[s[0], -1, self.heads, self.dim]),
                &[0, 2, 1, 3],
            )
        };
        let (q, k, v) = (
            project(&self.q, q),
            project(&self.k, k),
            project(&self.v, v),
        );
        // SAFETY: no mask is requested; Q/K/V own valid same-width floating arrays.
        let y = unsafe {
            mx::fast_scaled_dot_product_attention(
                &q,
                &k,
                &v,
                (self.dim as f64).powf(-0.5) as f32,
                std::ptr::null(),
            )
        };
        self.proj.forward(&mx::reshape(
            &mx::transpose_axes(&y, &[0, 2, 1, 3]),
            &[s[0], s[1], self.heads * self.dim],
        ))
    }
}
pub struct Layer {
    pe1: Norm,
    pe2: Norm,
    ln1: Norm,
    ln21: Norm,
    ln22: Norm,
    ln3: Norm,
    self_attn: Attention,
    cross_attn: Attention,
    ffn: Ffn,
    first: bool,
}
impl Layer {
    fn new(w: &WeightMap, p: &str, c: &Config, first: bool) -> Result<Self> {
        let norm = |s: &str| Norm::from_weights(w, &format!("{p}.{s}"), 1e-6, true);
        Ok(Self {
            pe1: norm("ln_pe_1")?,
            pe2: norm("ln_pe_2")?,
            ln1: norm("ln1")?,
            ln21: norm("ln2_1")?,
            ln22: norm("ln2_2")?,
            ln3: norm("ln3")?,
            self_attn: Attention::new(w, &format!("{p}.self_attn"), c)?,
            cross_attn: Attention::new(w, &format!("{p}.cross_attn"), c)?,
            ffn: Ffn::from_weights(w, &format!("{p}.ffn"))?,
            first,
        })
    }
    pub fn forward(
        &self,
        x: &MlxArray,
        context: &MlxArray,
        pe: &MlxArray,
        context_pe: &MlxArray,
    ) -> Array {
        let pe = self.pe1.forward(pe);
        let context_pe = self.pe2.forward(context_pe);
        let norm = self.ln1.forward(x);
        let q = if self.first {
            mx::copy(&norm)
        } else {
            mx::add(&norm, &pe)
        };
        let x = mx::add(x, &self.self_attn.forward(&q, &q, &norm));
        let q = mx::add(&self.ln21.forward(&x), &pe);
        let value = self.ln22.forward(context);
        let key = mx::add(&value, &context_pe);
        let x = mx::add(&x, &self.cross_attn.forward(&q, &key, &value));
        mx::add(&x, &self.ffn.forward(&self.ln3.forward(&x)))
    }
}
pub struct Decoder {
    pub layers: Vec<Layer>,
    pub norm_final: Norm,
}
impl Decoder {
    pub fn from_weights(w: &WeightMap, c: &Config) -> Result<Self> {
        ensure!(
            c.decoder_depth == 6,
            "Body requires exactly six decoder layers"
        );
        Ok(Self {
            layers: (0..c.decoder_depth)
                .map(|i| Layer::new(w, &format!("decoder.layers.{i}"), c, i == 0))
                .collect::<Result<_>>()?,
            norm_final: Norm::from_weights(w, "decoder.norm_final", 1e-6, true)?,
        })
    }
}
