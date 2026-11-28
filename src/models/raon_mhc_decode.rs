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

//! Decode-step fast paths of a Raon_mHC site (issue #3458).
//!
//! A Raon-1 mini token crosses 50 sites. Profiling the 30 ms step on an M3
//! Ultra put about a third of it in one product per site, and a few percent
//! in the small ops around it:
//!
//! - **Selector product.** `flat @ selector^T` with one row and the
//!   transposed plane on the right ran MLX's transposed gemv kernel over a
//!   `[n*D, 17]` matrix, about 255 us a call against about 35 us for
//!   `selector @ flat^T` over the row-major plane
//!   (`RaonMhc::selector_product`). The two forms sum in different orders,
//!   so the one-row form is used wherever a row is formed alone (decode
//!   steps and the rows of a speculative verify block, which therefore stay
//!   bit-equal), and products of several rows keep the plain matmul.
//! - **Site graphs.** The route, aggregate and mix of a single-position step
//!   run as two compiled graphs ([`RaonMhc::enter`], [`RaonMhc::leave`])
//!   that replay the eager ops in the same order. On Metal they match the
//!   eager ops bit for bit; on CUDA the fused kernel may contract a multiply
//!   and an add into one FMA, so they can differ in the last bit. A
//!   speculative verify block therefore runs the same graphs one position at
//!   a time, which keeps each of its rows equal to the decode step it
//!   replaces on every backend (issue #3474).
//!
//! [`DecodeFusion`] (`MLXCEL_RAON_DECODE_FUSION`) turns either off.

use super::{RaonMhc, aggregate, mix_parts};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// What [`RaonMhc::enter`] hands to [`RaonMhc::leave`]: the coefficients of
/// the write-back, and the float32 streams when the eager path computed them.
pub struct MhcSite {
    /// `[B, L, n]`, already times `post_factor`.
    pub(super) h_post: UniquePtr<MlxArray>,
    /// `[B, L, n, n]`, doubly stochastic.
    pub(super) h_res: UniquePtr<MlxArray>,
    /// `None` on the compiled path, which casts inside its own graph.
    pub(super) streams_f32: Option<UniquePtr<MlxArray>>,
}

/// Float32 arrays the compiled site graphs take as inputs (issue #3458), so
/// every site shares one trace.
pub(super) struct CompiledConsts {
    /// `[2n + (n-1)^2]`: `alpha_pre` (n times), `alpha_post` (n times), `alpha_z`.
    alpha: UniquePtr<MlxArray>,
    /// `[2n + (n-1)^2]`: `b_pre`, `b_post`, `b_z`.
    bias: UniquePtr<MlxArray>,
    /// `[1]` each.
    eps: UniquePtr<MlxArray>,
    post_factor: UniquePtr<MlxArray>,
    gamma: UniquePtr<MlxArray>,
    /// `(1 - gamma) I`, `[n, n]`.
    identity: UniquePtr<MlxArray>,
}

impl CompiledConsts {
    /// `alphas` and `biases` in pre, post, z order; `(n, m)` the stream count
    /// and `(n-1)^2`; `eye` the float32 `[n, n]` identity.
    pub(super) fn new(
        alphas: [f32; 3],
        biases: [&MlxArray; 3],
        (n, m): (usize, usize),
        selector_eps: f32,
        post_factor: f32,
        gamma: f32,
        eye: &MlxArray,
    ) -> Self {
        let scalar = |v: f32| mlxcel_core::full_f32(&[1], v, dtype::FLOAT32);
        let alpha: Vec<f32> = std::iter::repeat_n(alphas[0], n)
            .chain(std::iter::repeat_n(alphas[1], n))
            .chain(std::iter::repeat_n(alphas[2], m))
            .collect();
        Self {
            alpha: mlxcel_core::from_slice_f32(&alpha, &[(2 * n + m) as i32]),
            bias: mlxcel_core::concatenate_many(&biases, 0),
            eps: scalar(selector_eps),
            post_factor: scalar(post_factor),
            gamma: scalar(gamma),
            identity: mlxcel_core::multiply_scalar(eye, 1.0 - gamma),
        }
    }
}

/// Which decode-step fast paths of the Raon_mHC sites run
/// (`MLXCEL_RAON_DECODE_FUSION`, issue #3458): `all` (the default), `off`
/// (the reference paths of the first Raon-1 mini release), or one of
/// `selector` / `mhc` alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeFusion {
    /// The row-major form of a one-row selector product.
    pub selector: bool,
    /// Compiled site graphs on single-position steps.
    pub mhc: bool,
}

impl DecodeFusion {
    pub const ALL: Self = Self {
        selector: true,
        mhc: true,
    };
    pub const OFF: Self = Self {
        selector: false,
        mhc: false,
    };

    /// The setting `value` names; unset or `all` keeps the default. A value
    /// naming an unknown fast path (or an empty list entry) also selects every
    /// fast path, with a warning naming the value.
    pub fn parse(value: Option<&str>) -> Self {
        let Some(value) = value.map(|v| v.trim().to_ascii_lowercase()) else {
            return Self::ALL;
        };
        if matches!(value.as_str(), "off" | "0" | "false" | "none") {
            return Self::OFF;
        }
        if value == "all" {
            return Self::ALL;
        }
        let mut chosen = Self::OFF;
        for name in value.split(',').map(str::trim) {
            match name {
                "selector" => chosen.selector = true,
                "mhc" => chosen.mhc = true,
                _ => {
                    tracing::warn!(
                        value = %value,
                        "unrecognized MLXCEL_RAON_DECODE_FUSION value; \
                         falling back to all fast paths (accepted: all, off, selector, mhc)"
                    );
                    return Self::ALL;
                }
            }
        }
        chosen
    }

    /// The process setting, read once.
    pub fn from_env() -> Self {
        static FUSION: std::sync::OnceLock<DecodeFusion> = std::sync::OnceLock::new();
        *FUSION
            .get_or_init(|| Self::parse(std::env::var("MLXCEL_RAON_DECODE_FUSION").ok().as_deref()))
    }
}

impl RaonMhc {
    /// Choose the form of a one-row selector product, whatever
    /// `MLXCEL_RAON_DECODE_FUSION` says: row-major (`true`) or the plain
    /// matmul over the transposed plane.
    #[cfg(test)]
    pub fn set_row_major_single(&mut self, on: bool) {
        self.row_major_single = on;
    }

    /// Enter the site: the branch input `sum_s h_pre[s] * streams[s]`
    /// (`[B, L, D]`, rounded once to the stream dtype) and the write-back
    /// coefficients for [`Self::leave`].
    ///
    /// With `compiled` set (issue #3458), a single-position step takes the
    /// compiled graph, and so does every position of a speculative verify
    /// block, one at a time, so each verify row equals its decode step even
    /// where the compiled graph and the eager ops differ in the last bit
    /// (CUDA, issue #3474). Every other shape (prefill) runs the eager ops.
    pub fn enter(&self, streams: &MlxArray, compiled: bool) -> (UniquePtr<MlxArray>, MhcSite) {
        let shape = mlxcel_core::array_shape(streams);
        if compiled && shape[1] == 1 {
            return self.enter_compiled(streams, &shape);
        }
        if compiled && is_verify_block(&shape) {
            let steps: Vec<_> = (0..shape[1])
                .map(|t| {
                    let step = position(streams, t);
                    self.enter_compiled(&step, &mlxcel_core::array_shape(&step))
                })
                .collect();
            let x = concat_positions(steps.iter().map(|(x, _)| &**x));
            let h_post = concat_positions(steps.iter().map(|(_, s)| &*s.h_post));
            let h_res = concat_positions(steps.iter().map(|(_, s)| &*s.h_res));
            let site = MhcSite {
                h_post,
                h_res,
                streams_f32: None,
            };
            return (x, site);
        }
        let dt = mlxcel_core::array_dtype(streams);
        let route = self.route(streams);
        let sf = mlxcel_core::astype(streams, dtype::FLOAT32);
        let x = aggregate(&sf, &route.h_pre, dt);
        let site = MhcSite {
            h_post: route.h_post,
            h_res: route.h_res,
            streams_f32: Some(sf),
        };
        (x, site)
    }

    fn enter_compiled(&self, streams: &MlxArray, shape: &[i32]) -> (UniquePtr<MlxArray>, MhcSite) {
        let flat = mlxcel_core::reshape(streams, &[shape[0], shape[1], shape[2] * shape[3]]);
        let proj = self.selector_product(&flat);
        let mut x = UniquePtr::null();
        let mut h_post = UniquePtr::null();
        let mut h_res = UniquePtr::null();
        let consts = &self.compiled;
        mlxcel_core::raon_mhc_route_aggregate(
            streams,
            &proj,
            &consts.alpha,
            &consts.bias,
            &consts.eps,
            &consts.post_factor,
            &self.basis,
            &self.basis_t,
            &self.chart0,
            &consts.identity,
            &consts.gamma,
            &mut x,
            &mut h_post,
            &mut h_res,
        );
        let site = MhcSite {
            h_post,
            h_res,
            streams_f32: None,
        };
        (x, site)
    }

    /// Leave the site: `h_res @ streams + h_post * branch` in float32,
    /// rounded once to the stream dtype. `streams` is what [`Self::enter`]
    /// read.
    pub fn leave(
        &self,
        streams: &MlxArray,
        branch: &MlxArray,
        site: &MhcSite,
    ) -> UniquePtr<MlxArray> {
        match &site.streams_f32 {
            Some(sf) => mix_parts(
                sf,
                branch,
                &site.h_post,
                &site.h_res,
                mlxcel_core::array_dtype(streams),
            ),
            None if mlxcel_core::array_shape(streams)[1] == 1 => {
                mlxcel_core::raon_mhc_mix(streams, branch, &site.h_post, &site.h_res)
            }
            // A verify block's compiled site: one decode-step graph per position.
            None => {
                let length = mlxcel_core::array_shape(streams)[1];
                let steps: Vec<_> = (0..length)
                    .map(|t| {
                        mlxcel_core::raon_mhc_mix(
                            &position(streams, t),
                            &position(branch, t),
                            &position(&site.h_post, t),
                            &position(&site.h_res, t),
                        )
                    })
                    .collect();
                concat_positions(steps.iter().map(|s| &**s))
            }
        }
    }
}

/// Whether `shape` (`[B, L, ..]`) is a single-sequence verify block the
/// compiled site splits per position.
fn is_verify_block(shape: &[i32]) -> bool {
    mlxcel_core::row_invariant::in_verify_block()
        && shape[0] == 1
        && (2..=mlxcel_core::row_invariant::MAX_ROWS).contains(&shape[1])
}

/// Position `t` of axis 1, keeping the axis.
fn position(x: &MlxArray, t: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::utils::slice_axis(x, 1, t, t + 1)
}

fn concat_positions<'a>(parts: impl Iterator<Item = &'a MlxArray>) -> UniquePtr<MlxArray> {
    let parts: Vec<&MlxArray> = parts.collect();
    mlxcel_core::concatenate_many(&parts, 1)
}
