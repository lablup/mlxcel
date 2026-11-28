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

//! Z1T building blocks: the fixed fan-in sparse linear, Dynamic Tanh, the
//! AFT-conv token mixer with its streaming state, the MLP and the block.
//!
//! Numerics follow `mlx_vlm/models/z1t/language.py`. The conv convention is
//! `out[t] = sum_j kernel[:, j] * ext[t + j]` over `ext = window ++ new`, so tap
//! `ksize - 1` is the current token and a fresh (all-zero) state is exactly the
//! flat forward's zero left padding.

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::Z1tConfig;

/// Cap on the elements of the transient `[B, chunk, out, k]` gather of one
/// sparse projection; longer inputs are gathered in sequence-axis chunks,
/// which is exact. Upstream `_SPARSE_CHUNK_ELEMS` caps `L * out` at 8M, which
/// ignores the batch and the fan-in, so a checkpoint with a large `k` would
/// allocate `8M * k` elements per projection. Measuring the whole transient
/// keeps the upstream chunk size for Z1T-0 (`B = 1`, `k = 4`) and bounds the
/// rest.
pub(super) const SPARSE_CHUNK_ELEMS: i64 = 32_000_000;

fn weight(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Z1T weight not found: {name}"))
}

fn expect_shape(name: &str, array: &MlxArray, want: &[i32]) -> Result<(), String> {
    let got = mlxcel_core::array_shape(array);
    if got != want {
        return Err(format!("Z1T {name} has shape {got:?}, expected {want:?}"));
    }
    Ok(())
}

/// Check the stored shape of a dense (`out`, `in`) projection before it loads.
///
/// Quantized checkpoints pack the input axis, so a layer that ships a
/// `.scales` tensor is skipped: only the unquantized shape is unambiguous.
pub(super) fn expect_dense_shape(
    weights: &WeightMap,
    prefix: &str,
    out_features: usize,
    in_features: usize,
) -> Result<(), String> {
    if weights.contains_key(&format!("{prefix}.scales")) {
        return Ok(());
    }
    match weights.get(&format!("{prefix}.weight")) {
        Some(w) => expect_shape(
            &format!("{prefix}.weight"),
            w,
            &[out_features as i32, in_features as i32],
        ),
        None => Ok(()),
    }
}

fn scalar_like(value: f32, like: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[], value, mlxcel_core::array_dtype(like))
}

/// Inclusive cumsum along the time axis of a `len`-token chunk. A one-token
/// chunk is its own cumsum, and skipping the scan there is load-bearing: on
/// Z1T-0 the strided scan over a length-1 axis cost 93 ms per layer per decode
/// step on an M3 Ultra, against under 1 ms for the whole mixer without it.
fn inclusive_cumsum_time(x: &MlxArray, len: i32) -> UniquePtr<MlxArray> {
    if len == 1 {
        mlxcel_core::copy(x)
    } else {
        mlxcel_core::cumsum(x, 1, false, true)
    }
}

/// `y = sum_k weight[o, k] * x[..., indices[o, k]] + bias[o]`, with a fixed,
/// checkpoint-provided fan-in table.
pub(super) struct SparseLinear {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
    indices: UniquePtr<MlxArray>,
    out_features: i32,
    fan_in: i32,
    pub(super) chunk_elems: i64,
}

impl SparseLinear {
    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        k: usize,
    ) -> Result<Self, String> {
        let shape = [out_features as i32, k as i32];
        let w = weight(weights, &format!("{prefix}.weight"))?;
        expect_shape(&format!("{prefix}.weight"), &w, &shape)?;
        let bias = weight(weights, &format!("{prefix}.bias"))?;
        expect_shape(&format!("{prefix}.bias"), &bias, &[out_features as i32])?;
        let mut indices = weight(weights, &format!("{prefix}.indices"))?;
        expect_shape(&format!("{prefix}.indices"), &indices, &shape)?;
        let index_dtype = mlxcel_core::array_dtype(&indices);
        if !matches!(
            index_dtype,
            dtype::INT32 | dtype::INT64 | dtype::UINT32 | dtype::INT16 | dtype::UINT16
        ) {
            return Err(format!(
                "Z1T {prefix}.indices must be an integer table, found dtype {index_dtype}"
            ));
        }
        if index_dtype != dtype::INT32 {
            indices = mlxcel_core::astype(&indices, dtype::INT32);
        }
        // The table comes from an untrusted checkpoint and feeds a GPU gather:
        // refuse any entry outside the input width at load time.
        let lo = mlxcel_core::item_i32(&mlxcel_core::min_all(&indices));
        let hi = mlxcel_core::item_i32(&mlxcel_core::max_all(&indices));
        if lo < 0 || hi as i64 >= in_features as i64 {
            return Err(format!(
                "Z1T {prefix}.indices spans [{lo}, {hi}], outside the input width {in_features}"
            ));
        }
        Ok(Self {
            weight: w,
            bias,
            indices,
            out_features: out_features as i32,
            fan_in: k as i32,
            chunk_elems: SPARSE_CHUNK_ELEMS,
        })
    }

    fn gather(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let axis = mlxcel_core::array_shape(x).len() as i32 - 1;
        let taken = mlxcel_core::take(x, &self.indices, axis);
        let weighted = mlxcel_core::multiply(&taken, &self.weight);
        let summed = mlxcel_core::sum_axis(&weighted, axis + 1, false);
        mlxcel_core::add(&summed, &self.bias)
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        // Elements of one sequence position's `[B, 1, out, k]` gather.
        let per_token = || {
            (shape[0] as i64)
                .saturating_mul(self.out_features as i64)
                .saturating_mul(self.fan_in as i64)
        };
        if shape.len() == 3 && (shape[1] as i64).saturating_mul(per_token()) > self.chunk_elems {
            let per_token = per_token();
            let len = shape[1];
            let step = (self.chunk_elems / per_token.max(1)).clamp(1, len as i64) as i32;
            let parts: Vec<UniquePtr<MlxArray>> = (0..len)
                .step_by(step as usize)
                .map(|start| {
                    let stop = (start + step).min(len);
                    let piece = mlxcel_core::slice(x, &[0, start, 0], &[shape[0], stop, shape[2]]);
                    self.gather(&piece)
                })
                .collect();
            let refs: Vec<&MlxArray> = parts.iter().map(|p| &**p).collect();
            return mlxcel_core::concatenate_many(&refs, 1);
        }
        self.gather(x)
    }
}

/// A Z1T internal projection: fixed-sparse when the fan-in is below the input
/// width, a dense `nn.Linear` otherwise (upstream `_make_linear`).
pub(super) enum Z1tLinear {
    Sparse(SparseLinear),
    Dense(UnifiedLinear),
}

impl Z1tLinear {
    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        fan_in: Option<usize>,
        config: &Z1tConfig,
    ) -> Result<Self, String> {
        let k = fan_in.map_or(in_features, |k| k.clamp(1, in_features));
        if k >= in_features {
            expect_dense_shape(weights, prefix, out_features, in_features)?;
            let linear =
                UnifiedLinear::from_weights(weights, prefix, config.group_size(), config.bits())?;
            return Ok(Self::Dense(linear));
        }
        SparseLinear::from_weights(weights, prefix, in_features, out_features, k).map(Self::Sparse)
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Sparse(linear) => linear.forward(x),
            Self::Dense(linear) => linear.forward(x),
        }
    }

    #[cfg(test)]
    pub(super) fn sparse_mut(&mut self) -> Option<&mut SparseLinear> {
        match self {
            Self::Sparse(linear) => Some(linear),
            Self::Dense(_) => None,
        }
    }
}

/// Dynamic Tanh: `weight * tanh(alpha * x) + bias`, `alpha` a `[1]` tensor.
pub(super) struct DyT {
    alpha: UniquePtr<MlxArray>,
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
}

impl DyT {
    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        dim: usize,
    ) -> Result<Self, String> {
        let alpha = weight(weights, &format!("{prefix}.alpha"))?;
        expect_shape(&format!("{prefix}.alpha"), &alpha, &[1])?;
        let w = weight(weights, &format!("{prefix}.weight"))?;
        expect_shape(&format!("{prefix}.weight"), &w, &[dim as i32])?;
        let bias = weight(weights, &format!("{prefix}.bias"))?;
        expect_shape(&format!("{prefix}.bias"), &bias, &[dim as i32])?;
        Ok(Self {
            alpha,
            weight: w,
            bias,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let squashed = mlxcel_core::tanh(&mlxcel_core::multiply(&self.alpha, x));
        mlxcel_core::add(&mlxcel_core::multiply(&self.weight, &squashed), &self.bias)
    }
}

/// Per-layer AFT-conv streaming state: running pool sums and the last
/// `ksize - 1` conv-window values. `None` everywhere means fresh, which equals
/// zeros, i.e. the flat forward's left padding.
#[derive(Default)]
pub struct Z1tLayerState {
    /// Tokens this sequence has consumed (drives the positional table).
    pub offset: usize,
    /// `[B, heads, head_dim]`.
    pub cum_ekv: Option<UniquePtr<MlxArray>>,
    /// `[B, heads]`.
    pub cum_ek: Option<UniquePtr<MlxArray>>,
    /// `[B, ksize - 1, heads, head_dim]`.
    pub win_ekv: Option<UniquePtr<MlxArray>>,
    /// `[B, ksize - 1, heads]`.
    pub win_ek: Option<UniquePtr<MlxArray>>,
}

impl Z1tLayerState {
    pub fn is_initialized(&self) -> bool {
        self.cum_ekv.is_some()
    }
}

/// AFT-conv token mixer: depthwise causal conv plus causal cumulative pool.
pub(super) struct AftConv {
    pub(super) qkv_proj: Z1tLinear,
    pub(super) out_proj: Z1tLinear,
    /// `exp(w_conv) - 1`, one `[1, 1, heads, 1]` tap per kernel column.
    taps_ekv: Vec<UniquePtr<MlxArray>>,
    /// The same taps shaped `[1, 1, heads]`.
    taps_ek: Vec<UniquePtr<MlxArray>>,
    dim: i32,
    heads: i32,
    ksize: i32,
    tanh_linear: bool,
}

impl AftConv {
    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Z1tConfig,
    ) -> Result<Self, String> {
        let dim = config.hidden_size;
        let heads = config.aft_heads;
        let ksize = config.aft_ksize;
        let fan = config.fan_in_attn();
        let qkv_proj = Z1tLinear::from_weights(
            weights,
            &format!("{prefix}.qkv_proj"),
            dim,
            2 * dim + heads,
            fan,
            config,
        )?;
        let out_proj = Z1tLinear::from_weights(
            weights,
            &format!("{prefix}.out_proj"),
            dim,
            dim,
            fan,
            config,
        )?;
        let w_conv = weight(weights, &format!("{prefix}.w_conv"))?;
        expect_shape(
            &format!("{prefix}.w_conv"),
            &w_conv,
            &[heads as i32, ksize as i32],
        )?;
        let kernel = mlxcel_core::subtract(&mlxcel_core::exp(&w_conv), &scalar_like(1.0, &w_conv));
        let (h, s) = (heads as i32, ksize as i32);
        let column = |j: i32| mlxcel_core::slice(&kernel, &[0, j], &[h, j + 1]);
        let taps_ekv = (0..s)
            .map(|j| mlxcel_core::reshape(&column(j), &[1, 1, h, 1]))
            .collect();
        let taps_ek = (0..s)
            .map(|j| mlxcel_core::reshape(&column(j), &[1, 1, h]))
            .collect();
        Ok(Self {
            qkv_proj,
            out_proj,
            taps_ekv,
            taps_ek,
            dim: dim as i32,
            heads: h,
            ksize: s,
            tanh_linear: config.tanh_linear,
        })
    }

    fn squash(&self, x: UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
        if self.tanh_linear {
            mlxcel_core::tanh(&x)
        } else {
            x
        }
    }

    /// Mix `x` (`[B, L, dim]`) and advance `state` by `L` tokens.
    pub(super) fn forward(&self, x: &MlxArray, state: &mut Z1tLayerState) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let (d, h, s) = (self.dim, self.heads, self.ksize);
        let hd = d / h;

        let qkv = self.squash(self.qkv_proj.forward(x));
        let q = mlxcel_core::slice(&qkv, &[0, 0, 0], &[b, l, d]);
        let v = mlxcel_core::slice(&qkv, &[0, 0, d], &[b, l, 2 * d]);
        let k = mlxcel_core::slice(&qkv, &[0, 0, 2 * d], &[b, l, 2 * d + h]);
        let qh = mlxcel_core::reshape(&q, &[b, l, h, hd]);
        let vh = mlxcel_core::reshape(&v, &[b, l, h, hd]);

        let ek = mlxcel_core::exp(&mlxcel_core::tanh(&k)); // [B, L, h]
        let ekv = mlxcel_core::multiply(&mlxcel_core::expand_dims(&ek, 3), &vh); // [B, L, h, hd]
        let act = mlxcel_core::array_dtype(&ekv);

        let cum_ekv = state
            .cum_ekv
            .take()
            .unwrap_or_else(|| mlxcel_core::zeros(&[b, h, hd], act));
        let cum_ek = state
            .cum_ek
            .take()
            .unwrap_or_else(|| mlxcel_core::zeros(&[b, h], act));

        // Depthwise causal conv over [window ++ new].
        let (ext_ekv, ext_ek) = if s > 1 {
            let win_ekv = state
                .win_ekv
                .take()
                .unwrap_or_else(|| mlxcel_core::zeros(&[b, s - 1, h, hd], act));
            let win_ek = state
                .win_ek
                .take()
                .unwrap_or_else(|| mlxcel_core::zeros(&[b, s - 1, h], act));
            (
                mlxcel_core::concatenate(&win_ekv, &ekv, 1),
                mlxcel_core::concatenate(&win_ek, &ek, 1),
            )
        } else {
            (mlxcel_core::copy(&ekv), mlxcel_core::copy(&ek))
        };
        let mut num: Option<UniquePtr<MlxArray>> = None;
        let mut den: Option<UniquePtr<MlxArray>> = None;
        for j in 0..s {
            let ekv_j = mlxcel_core::slice(&ext_ekv, &[0, j, 0, 0], &[b, j + l, h, hd]);
            let ek_j = mlxcel_core::slice(&ext_ek, &[0, j, 0], &[b, j + l, h]);
            let term_ekv = mlxcel_core::multiply(&self.taps_ekv[j as usize], &ekv_j);
            let term_ek = mlxcel_core::multiply(&self.taps_ek[j as usize], &ek_j);
            num = Some(match num {
                Some(acc) => mlxcel_core::add(&acc, &term_ekv),
                None => term_ekv,
            });
            den = Some(match den {
                Some(acc) => mlxcel_core::add(&acc, &term_ek),
                None => term_ek,
            });
        }

        // Global pool: prior running sum + inclusive cumsum within the chunk.
        let run_ekv = mlxcel_core::add(
            &mlxcel_core::expand_dims(&cum_ekv, 1),
            &inclusive_cumsum_time(&ekv, l),
        );
        let run_ek = mlxcel_core::add(
            &mlxcel_core::expand_dims(&cum_ek, 1),
            &inclusive_cumsum_time(&ek, l),
        );
        let num = mlxcel_core::add(&num.expect("ksize >= 1"), &run_ekv);
        let den = mlxcel_core::add(&den.expect("ksize >= 1"), &run_ek);
        let den = mlxcel_core::add(&mlxcel_core::expand_dims(&den, 3), &scalar_like(1e-6, &den));
        let ctx = mlxcel_core::divide(&num, &den);
        let y = mlxcel_core::reshape(
            &mlxcel_core::multiply(&mlxcel_core::tanh(&qh), &ctx),
            &[b, l, d],
        );
        let out = self.squash(self.out_proj.forward(&y));

        state.cum_ekv = Some(mlxcel_core::add(
            &cum_ekv,
            &mlxcel_core::sum_axis(&ekv, 1, false),
        ));
        state.cum_ek = Some(mlxcel_core::add(
            &cum_ek,
            &mlxcel_core::sum_axis(&ek, 1, false),
        ));
        // A lazy running sum holds this chunk's full-length `ekv` / `ek` until
        // the next step evaluates it; after a long prefill that pins every
        // layer's `[L, dim]` source. Schedule it so the source is released.
        for pooled in [&state.cum_ekv, &state.cum_ek].into_iter().flatten() {
            mlxcel_core::utils::schedule_prefill_state(pooled, l);
        }
        if s > 1 {
            state.win_ekv = Some(mlxcel_core::utils::materialized_state_tail(&ext_ekv, s - 1));
            state.win_ek = Some(mlxcel_core::utils::materialized_state_tail(&ext_ek, s - 1));
        }
        out
    }
}

/// Feed-forward: `proj2(act(proj1(x)))`, every projection tanh-wrapped when
/// `tanh_linear`, the activation `tanh` (`tanh_mlp`) or SiLU.
pub(super) struct Mlp {
    pub(super) proj1: Z1tLinear,
    pub(super) proj2: Z1tLinear,
    tanh_linear: bool,
    tanh_mlp: bool,
}

impl Mlp {
    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Z1tConfig,
    ) -> Result<Self, String> {
        let dim = config.hidden_size;
        let (hidden, fan) = (dim * 4, config.fan_in_mlp());
        let p1 = format!("{prefix}.proj1");
        let p2 = format!("{prefix}.proj2");
        Ok(Self {
            proj1: Z1tLinear::from_weights(weights, &p1, dim, hidden, fan, config)?,
            proj2: Z1tLinear::from_weights(weights, &p2, hidden, dim, fan, config)?,
            tanh_linear: config.tanh_linear,
            tanh_mlp: config.tanh_mlp,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut h = self.proj1.forward(x);
        if self.tanh_linear {
            h = mlxcel_core::tanh(&h);
        }
        h = if self.tanh_mlp {
            mlxcel_core::tanh(&h)
        } else {
            mlxcel_core::silu(&h)
        };
        let mut o = self.proj2.forward(&h);
        if self.tanh_linear {
            o = mlxcel_core::tanh(&o);
        }
        o
    }
}

/// `x += attn(norm1(x)); x += mlp(norm2(x))`.
pub(super) struct Block {
    norm1: DyT,
    pub(super) attn: AftConv,
    norm2: DyT,
    pub(super) mlp: Mlp,
}

impl Block {
    pub(super) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Z1tConfig,
    ) -> Result<Self, String> {
        let dim = config.hidden_size;
        Ok(Self {
            norm1: DyT::from_weights(weights, &format!("{prefix}.norm1"), dim)?,
            attn: AftConv::from_weights(weights, &format!("{prefix}.attn"), config)?,
            norm2: DyT::from_weights(weights, &format!("{prefix}.norm2"), dim)?,
            mlp: Mlp::from_weights(weights, &format!("{prefix}.mlp"), config)?,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray, state: &mut Z1tLayerState) -> UniquePtr<MlxArray> {
        let x = mlxcel_core::add(x, &self.attn.forward(&self.norm1.forward(x), state));
        mlxcel_core::add(&x, &self.mlp.forward(&self.norm2.forward(&x)))
    }
}
