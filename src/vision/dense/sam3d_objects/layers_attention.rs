// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Dense and cross attention, including unconditional projection bias caching.
use super::{Array, HeadNorm, Linear};
use anyhow::{Result, ensure};
use mlxcel_core::{self as mx, MlxArray, weights::WeightMap};

pub enum PreparedContext {
    Bias(Array),
    KeyValue(Array, Array),
}
pub struct Attention {
    pub to_qkv: Option<Linear>,
    to_q: Option<Linear>,
    to_kv: Option<Linear>,
    pub to_out: Linear,
    q_norm: Option<HeadNorm>,
    k_norm: Option<HeadNorm>,
    heads: i32,
}
impl Attention {
    pub fn from_weights(
        w: &WeightMap,
        p: &str,
        heads: usize,
        cross: bool,
        qk_norm: bool,
    ) -> Result<Self> {
        ensure!(
            heads > 0 && i32::try_from(heads).is_ok(),
            "attention needs valid positive head count"
        );
        let to_out = Linear::from_weights(w, &format!("{p}.to_out"))?;
        ensure!(
            mx::array_shape(&to_out.weight)[1] % heads as i32 == 0,
            "attention width must divide head count"
        );
        Ok(Self {
            to_qkv: (!cross)
                .then(|| Linear::from_weights(w, &format!("{p}.to_qkv")))
                .transpose()?,
            to_q: cross
                .then(|| Linear::from_weights(w, &format!("{p}.to_q")))
                .transpose()?,
            to_kv: cross
                .then(|| Linear::from_weights(w, &format!("{p}.to_kv")))
                .transpose()?,
            q_norm: qk_norm
                .then(|| {
                    HeadNorm::from_weights(
                        w,
                        &format!("{p}.q_rms_norm"),
                        heads,
                        mx::array_shape(&to_out.weight)[1] as usize / heads,
                    )
                })
                .transpose()?,
            k_norm: qk_norm
                .then(|| {
                    HeadNorm::from_weights(
                        w,
                        &format!("{p}.k_rms_norm"),
                        heads,
                        mx::array_shape(&to_out.weight)[1] as usize / heads,
                    )
                })
                .transpose()?,
            to_out,
            heads: heads as i32,
        })
    }
    pub fn prepare(&self, context: Option<&MlxArray>) -> Result<PreparedContext> {
        let kv = self
            .to_kv
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("cannot prepare self attention"))?;
        let channels = mx::array_shape(&self.to_out.weight)[1];
        Ok(match context {
            None => {
                let value = match &kv.bias {
                    Some(b) => mx::slice_last_dim(b, channels, channels * 2),
                    None => mx::zeros(&[channels], mx::array_dtype(&kv.weight)),
                };
                PreparedContext::Bias(self.to_out.forward(&mx::reshape(&value, &[1, 1, channels])))
            }
            Some(ctx) => {
                let s = mx::array_shape(ctx);
                ensure!(s.len() == 3, "cross attention context must be BNC");
                let kv = mx::reshape(&kv.forward(ctx), &[s[0], s[1], 2, self.heads, -1]);
                PreparedContext::KeyValue(component(&kv, 0), component(&kv, 1))
            }
        })
    }
    pub fn forward(&self, x: &MlxArray, context: Option<&PreparedContext>) -> Result<Array> {
        let s = mx::array_shape(x);
        ensure!(s.len() == 3, "attention input must be BNC");
        if let Some(PreparedContext::Bias(bias)) = context {
            return Ok(mx::broadcast_to(bias, &s));
        }
        let (mut q, mut k, v) = match context {
            Some(PreparedContext::KeyValue(k, v)) => {
                let proj = self
                    .to_q
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing cross query projection"))?;
                (
                    mx::reshape(&proj.forward(x), &[s[0], s[1], self.heads, -1]),
                    mx::copy(k),
                    mx::copy(v),
                )
            }
            _ => {
                let proj = self
                    .to_qkv
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing self attention projection"))?;
                let qkv = mx::reshape(&proj.forward(x), &[s[0], s[1], 3, self.heads, -1]);
                (component(&qkv, 0), component(&qkv, 1), component(&qkv, 2))
            }
        };
        if let Some(norm) = &self.q_norm {
            q = norm.forward(&q)?;
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(&k)?;
        }
        Ok(self.to_out.forward(&attend(&q, &k, &v)))
    }
}
pub(super) fn component(x: &MlxArray, index: i32) -> Array {
    let s = mx::array_shape(x);
    mx::reshape(
        &mx::utils::slice_axis(x, 2, index, index + 1),
        &[s[0], s[1], s[3], s[4]],
    )
}
pub fn attend(q: &MlxArray, k: &MlxArray, v: &MlxArray) -> Array {
    let s = mx::array_shape(q);
    let (q, k, v) = (
        mx::transpose_axes(q, &[0, 2, 1, 3]),
        mx::transpose_axes(k, &[0, 2, 1, 3]),
        mx::transpose_axes(v, &[0, 2, 1, 3]),
    );
    // SAFETY: null mask requests unmasked attention; all arrays live through this call.
    let y = unsafe {
        mx::fast_scaled_dot_product_attention(
            &q,
            &k,
            &v,
            (s[3] as f32).powf(-0.5),
            std::ptr::null(),
        )
    };
    mx::reshape(&mx::transpose_axes(&y, &[0, 2, 1, 3]), &[s[0], s[1], -1])
}
