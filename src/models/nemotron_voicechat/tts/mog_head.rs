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

//! Mixture-of-Gaussians head that turns a backbone state into a continuous
//! RVQ latent sample.
//!
//! Ports `MoGHead.infer` and `_top_p_logits` from
//! `mlx_vlm/models/nemotron_voicechat/tts.py`. The component draw uses the
//! reference's own Gumbel construction, `-log(-log(u + 1e-8) + 1e-8)` over a
//! global-RNG `uniform` of the logits' shape, rather than
//! `mlx::random::gumbel`, so a run seeded with
//! [`mlxcel_core::random_seed`] consumes the global key sequence the same
//! way the Python reference does.

use mlxcel_core::dtype;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::MogConfig;
use super::norm_mlp::{MlpLayer, OffsetRmsNorm, scalar, weight, weight_with_shape};

/// Keep the smallest set of logits whose probability mass exceeds `top_p`
/// (ties at the boundary included); everything else becomes `-inf`.
///
/// Mirrors `_top_p_logits`: ascending `argsort` of the float32 softmax,
/// `cumsum > 1 - top_p` in sorted order, scattered back with
/// `put_along_axis`. `top_p >= 1` returns the logits unchanged.
pub fn top_p_logits(logits: &MlxArray, top_p: f64) -> Result<UniquePtr<MlxArray>, String> {
    if top_p >= 1.0 {
        return Ok(mlxcel_core::copy(logits));
    }
    if top_p <= 0.0 || top_p.is_nan() {
        return Err(format!("top_p must be in (0, 1], got {top_p}"));
    }
    let probs = mlxcel_core::softmax(&mlxcel_core::astype(logits, dtype::FLOAT32), -1);
    let indices = mlxcel_core::argsort(&probs, -1);
    let sorted = mlxcel_core::take_along_axis(&probs, &indices, -1);
    let cumulative = mlxcel_core::cumsum(&sorted, -1, false, true);
    let keep_sorted = mlxcel_core::greater(&cumulative, &scalar(1.0 - top_p, dtype::FLOAT32));
    let zeros = mlxcel_core::zeros(&mlxcel_core::array_shape(&keep_sorted), dtype::BOOL);
    let keep = mlxcel_core::put_along_axis(&zeros, &indices, &keep_sorted, -1);
    let neg_inf = scalar(f64::NEG_INFINITY, mlxcel_core::array_dtype(logits));
    Ok(mlxcel_core::where_cond(&keep, logits, &neg_inf))
}

/// `MoGHead` (key tree `mog_head`).
pub struct MogHead {
    layers: Vec<MlpLayer>,
    final_norm: OffsetRmsNorm,
    proj_logits: UnifiedLinear,
    /// `proj_mus.weight` viewed as `[num_predictions, low_rank, hidden]`.
    mus: UniquePtr<MlxArray>,
    proj_logs: UnifiedLinear,
    proj_else: UnifiedLinear,
    /// `[num_predictions, out_size, low_rank]`.
    low_mat: UniquePtr<MlxArray>,
    out_size: i32,
    min_log_std: f64,
}

impl MogHead {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden: usize,
        out_size: usize,
        cfg: &MogConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let layers = (0..cfg.num_layers)
            .map(|idx| {
                MlpLayer::from_weights(
                    weights,
                    &format!("{prefix}.mlp_stack.{idx}"),
                    hidden,
                    cfg.eps,
                    group_size,
                    bits,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let final_norm = OffsetRmsNorm::from_weights(
            weights,
            &format!("{prefix}.mlp_stack.{}", cfg.num_layers),
            hidden,
            cfg.eps,
        )?;
        let lin = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), group_size, bits)
        };
        // `proj_mus` is indexed as a table, so it has to be dense.
        let mus_key = format!("{prefix}.proj_mus");
        if weights.contains_key(&format!("{mus_key}.scales")) {
            return Err(format!(
                "{mus_key} must be dense (quantized MoG means are not supported)"
            ));
        }
        let (np, lr) = (cfg.num_predictions as i32, cfg.low_rank as i32);
        let mus = weight_with_shape(
            weights,
            &format!("{mus_key}.weight"),
            &[np * lr, hidden as i32],
        )?;
        let mus = mlxcel_core::reshape(&mus, &[np, lr, hidden as i32]);
        let low_mat = weight(weights, &format!("{prefix}.low_mat"))?;
        let lm_shape = mlxcel_core::array_shape(&low_mat);
        if lm_shape != [np, out_size as i32, lr] {
            return Err(format!(
                "{prefix}.low_mat: expected shape {:?}, got {lm_shape:?}",
                [np, out_size as i32, lr]
            ));
        }
        Ok(Self {
            layers,
            final_norm,
            proj_logits: lin("proj_logits")?,
            mus,
            proj_logs: lin("proj_logs")?,
            proj_else: lin("proj_else")?,
            low_mat,
            out_size: out_size as i32,
            min_log_std: cfg.min_log_std,
        })
    }

    /// `MoGHead.infer`: returns `(mu * exp(logs) + residual, logs)`.
    ///
    /// With `guidance_scale > 0` the batch is `[conditional; unconditional]`
    /// and the result has half the batch. Draws one global-RNG uniform tensor
    /// of the logits' shape.
    pub fn infer(
        &self,
        x: &MlxArray,
        guidance_scale: f64,
        top_p: f64,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
        let mut x = mlxcel_core::copy(x);
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        let mut x = self.final_norm.forward(&x);

        let shape = mlxcel_core::array_shape(&x);
        if guidance_scale > 0.0 {
            if shape[0] % 2 != 0 {
                return Err("classifier-free guidance requires an even batch".to_string());
            }
            let half = shape[0] / 2;
            let cond = mlxcel_core::slice(&x, &[0, 0, 0], &[half, shape[1], shape[2]]);
            let uncond = mlxcel_core::slice(&x, &[half, 0, 0], &[shape[0], shape[1], shape[2]]);
            let act = mlxcel_core::array_dtype(&x);
            let delta = mlxcel_core::subtract(&cond, &uncond);
            let delta = mlxcel_core::multiply(&scalar(guidance_scale, act), &delta);
            x = mlxcel_core::add(&cond, &delta);
        }
        let shape = mlxcel_core::array_shape(&x);
        let (b, t, h) = (shape[0], shape[1], shape[2]);

        let logits = top_p_logits(&self.proj_logits.forward(&x), top_p)?;
        let uniform = unsafe {
            // SAFETY: a null key selects MLX's global key sequence.
            mlxcel_core::random_uniform(
                0.0,
                1.0,
                &mlxcel_core::array_shape(&logits),
                dtype::FLOAT32,
                std::ptr::null(),
            )
        };
        let eps = scalar(1e-8, dtype::FLOAT32);
        let inner = mlxcel_core::negative(&mlxcel_core::log(&mlxcel_core::add(&uniform, &eps)));
        let gumbel = mlxcel_core::negative(&mlxcel_core::log(&mlxcel_core::add(&inner, &eps)));
        let log_probs = mlxcel_core::log(&mlxcel_core::softmax(
            &mlxcel_core::astype(&logits, dtype::FLOAT32),
            -1,
        ));
        let component = mlxcel_core::argmax(&mlxcel_core::add(&log_probs, &gumbel), -1, false);
        let flat_component =
            mlxcel_core::astype(&mlxcel_core::reshape(&component, &[-1]), dtype::INT32);

        let flat_x = mlxcel_core::reshape(&x, &[-1, h]);
        // The gathered tables stay as stored (bf16); cast the gathered slabs
        // to the activation dtype so `mu` stays f32 on CUDA builds, whose
        // promotion table resolves bf16 + f32 to bf16 (issue #2087). Exact,
        // and a no-op when the dtypes already match.
        let act = mlxcel_core::array_dtype(&flat_x);
        let mus = mlxcel_core::astype(&mlxcel_core::take(&self.mus, &flat_component, 0), act);
        let mu = mlxcel_core::matmul(&mus, &mlxcel_core::expand_dims(&flat_x, -1));
        let mu = mlxcel_core::squeeze_axis(&mu, -1);
        let low = mlxcel_core::astype(&mlxcel_core::take(&self.low_mat, &flat_component, 0), act);
        let mu = mlxcel_core::matmul(&low, &mlxcel_core::expand_dims(&mu, -1));
        let mu = mlxcel_core::reshape(&mlxcel_core::squeeze_axis(&mu, -1), &[b, t, self.out_size]);

        let residual = self.proj_else.forward(&x);
        let logs = self.proj_logs.forward(&x);
        let logs = mlxcel_core::maximum(
            &logs,
            &scalar(self.min_log_std, mlxcel_core::array_dtype(&logs)),
        );
        let sample = mlxcel_core::add(
            &mlxcel_core::multiply(&mu, &mlxcel_core::exp(&logs)),
            &residual,
        );
        Ok((sample, logs))
    }
}
