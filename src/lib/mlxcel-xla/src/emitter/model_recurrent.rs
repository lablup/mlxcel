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

//! Recurrent (Mamba2) layers in the shared model graphs (epic #564, #3338).
//!
//! A child of `model`, so it reuses the private weight-argument and layer
//! plumbing. The state ABI is the one [`Config::state_tensors`] lists: every
//! state tensor stacks one row per recurrent layer (`[L_ssm, ...]`, per
//! sequence), so a graph slices the row of each layer before the layer and
//! writes the advanced row back after it.
//!
//! - Prefill takes the state tensors as inputs after `real_len` and returns them
//!   after the logits (and after the KV caches, which an attention-free model
//!   does not have).
//! - The single-sequence decode keeps each state tensor in a
//!   `util.global private mutable` named after it; `@state_import` seeds the
//!   globals from the prefill result, as `@kv_import` seeds the KV cache.
//! - The ragged (continuous-batching) decode takes one `[B, ...dims]` input per
//!   state tensor after the KV inputs and returns the advanced `[B, ...dims]`
//!   tensors after the KV results (issue #3340). Rows are independent: an
//!   inactive row is advanced like any other and the shim discards it.

use super::*;

/// The per-layer recurrent state a graph threads through its layer loop, plus
/// the prefill row count the prefill mixer masks with.
pub(super) struct RecurrentLayers {
    /// One entry per recurrent layer, indexed by [`Config::state_layer_index`].
    /// Prefill rows are per-sequence (`h [H, P, N]`, `conv [K-1, C]`); the
    /// single-sequence decode keeps a batch of one (`[1, H, P, N]`).
    layers: Vec<SsmLayerState>,
    /// `real_len` of the prefill; `None` in a decode graph.
    real_len: Option<Val>,
}

impl RecurrentLayers {
    /// No recurrent layer (every attention-only graph).
    pub(super) fn none() -> Self {
        Self {
            layers: Vec::new(),
            real_len: None,
        }
    }

    /// Split the stacked state tensors (`tensors`, in [`Config::state_tensors`]
    /// order) into per-layer states. `real_len` is `Some` for the prefill, whose
    /// mixer takes unbatched state; the decode keeps a batch axis of one.
    pub(super) fn split(
        b: &mut Builder,
        c: &Config,
        tensors: &[Val],
        real_len: Option<Val>,
    ) -> Self {
        let Some(ssm) = c.ssm.as_ref().filter(|_| !tensors.is_empty()) else {
            return Self::none();
        };
        let [h_all, conv_all] = tensors else {
            panic!("a Mamba2 config threads exactly two state tensors");
        };
        let (hh, p, n) = (ssm.n_heads, ssm.head_dim, ssm.state_size);
        let (k1, cd) = (ssm.conv_kernel - 1, ssm.conv_dim);
        let unbatched = real_len.is_some();
        let layers = (0..c.recurrent_layer_count())
            .map(|s| {
                let h = b.slice(h_all, &[(s, s + 1), (0, hh), (0, p), (0, n)]);
                let conv = b.slice(conv_all, &[(s, s + 1), (0, k1), (0, cd)]);
                if unbatched {
                    SsmLayerState {
                        h: b.reshape(&h, vec![hh, p, n]),
                        conv: b.reshape(&conv, vec![k1, cd]),
                    }
                } else {
                    SsmLayerState { h, conv }
                }
            })
            .collect();
        Self { layers, real_len }
    }

    /// Split per-slot state tensors (`[B, L_ssm, ...]`, the ragged decode
    /// inputs, in [`Config::state_tensors`] order) into per-layer `[B, ...]`
    /// states for the batched mixer step.
    pub(super) fn split_rows(b: &mut Builder, c: &Config, tensors: &[Val], bsz: usize) -> Self {
        let Some(ssm) = c.ssm.as_ref().filter(|_| !tensors.is_empty()) else {
            return Self::none();
        };
        let [h_all, conv_all] = tensors else {
            panic!("a Mamba2 config threads exactly two state tensors");
        };
        let (hh, p, n) = (ssm.n_heads, ssm.head_dim, ssm.state_size);
        let (k1, cd) = (ssm.conv_kernel - 1, ssm.conv_dim);
        let layers = (0..c.recurrent_layer_count())
            .map(|s| {
                let h = b.slice(h_all, &[(0, bsz), (s, s + 1), (0, hh), (0, p), (0, n)]);
                let conv = b.slice(conv_all, &[(0, bsz), (s, s + 1), (0, k1), (0, cd)]);
                SsmLayerState {
                    h: b.reshape(&h, vec![bsz, hh, p, n]),
                    conv: b.reshape(&conv, vec![bsz, k1, cd]),
                }
            })
            .collect();
        Self {
            layers,
            real_len: None,
        }
    }

    /// Write every advanced `[B, ...]` layer state back into its row of the
    /// per-slot tensors `base` (`[B, L_ssm, ...]`), in [`Config::state_tensors`]
    /// order.
    pub(super) fn join_rows(&self, b: &mut Builder, base: &[Val]) -> Vec<Val> {
        let [h_all, conv_all] = base else {
            return Vec::new();
        };
        let mut h_all = h_all.clone();
        let mut conv_all = conv_all.clone();
        let zero = b.const_i32(0);
        for (s, layer) in self.layers.iter().enumerate() {
            let row = b.const_i32(i32::try_from(s).expect("recurrent layer index fits i32"));
            let mut h_shape = vec![h_all.ty.shape[0], 1];
            h_shape.extend(&h_all.ty.shape[2..]);
            let mut conv_shape = vec![conv_all.ty.shape[0], 1];
            conv_shape.extend(&conv_all.ty.shape[2..]);
            let h = b.reshape(&layer.h, h_shape);
            let conv = b.reshape(&layer.conv, conv_shape);
            h_all = b.dynamic_update_slice(&h_all, &h, &[&zero, &row, &zero, &zero, &zero]);
            conv_all = b.dynamic_update_slice(&conv_all, &conv, &[&zero, &row, &zero, &zero]);
        }
        vec![h_all, conv_all]
    }

    /// Write every advanced layer state back into the stacked tensors `base`
    /// (the graph inputs or the loaded globals), in [`Config::state_tensors`]
    /// order.
    pub(super) fn join(&self, b: &mut Builder, base: &[Val]) -> Vec<Val> {
        let [h_all, conv_all] = base else {
            return Vec::new();
        };
        let mut h_all = h_all.clone();
        let mut conv_all = conv_all.clone();
        let zero = b.const_i32(0);
        for (s, layer) in self.layers.iter().enumerate() {
            let row = b.const_i32(i32::try_from(s).expect("recurrent layer index fits i32"));
            let mut h_shape = vec![1];
            h_shape.extend(&h_all.ty.shape[1..]);
            let mut conv_shape = vec![1];
            conv_shape.extend(&conv_all.ty.shape[1..]);
            let h = b.reshape(&layer.h, h_shape);
            let conv = b.reshape(&layer.conv, conv_shape);
            h_all = b.dynamic_update_slice(&h_all, &h, &[&row, &zero, &zero, &zero]);
            conv_all = b.dynamic_update_slice(&conv_all, &conv, &[&row, &zero, &zero]);
        }
        vec![h_all, conv_all]
    }
}

/// Take one Mamba2 layer's weights in the order `push_ssm_layer_specs`
/// (`weights.rs`) lists them: the block norm, in_proj (+ bias), conv1d (+ bias),
/// dt_bias, A_log, D, the gated-norm weight, out_proj (+ bias). Only the block
/// norm (`in_ln`) and the mixer are set on the returned [`LayerW`].
pub(super) fn take_ssm_layer_weights(
    b: &mut Builder,
    decls: &mut Vec<ArgDecl>,
    idx: &mut usize,
    c: &Config,
    li: usize,
) -> LayerW {
    let ssm = c
        .ssm
        .as_ref()
        .expect("a recurrent layer implies an SsmConfig");
    let h = c.hidden;
    let (di, cd, nh) = (ssm.d_inner(), ssm.conv_dim, ssm.n_heads);
    let p = |k: &str| format!("params['layers'][{li}]['{k}']");
    let in_ln = take_arg(decls, idx, Ty::f32(vec![h]), p("in_ln"));
    let in_proj = take_weight(b, decls, idx, c, ssm.in_proj_dim(), h, p("ssm.in_proj"));
    let in_proj_b = ssm.use_bias.then(|| {
        take_arg(
            decls,
            idx,
            Ty::f32(vec![ssm.in_proj_dim()]),
            p("ssm.in_proj_b"),
        )
    });
    let conv_w = take_arg(
        decls,
        idx,
        Ty::f32(vec![cd, 1, ssm.conv_kernel]),
        p("ssm.conv_w"),
    );
    let conv_b = ssm
        .use_conv_bias
        .then(|| take_arg(decls, idx, Ty::f32(vec![cd]), p("ssm.conv_b")));
    let dt_bias = take_arg(decls, idx, Ty::f32(vec![nh]), p("ssm.dt_bias"));
    let a_log = take_arg(decls, idx, Ty::f32(vec![nh]), p("ssm.a_log"));
    let d = take_arg(decls, idx, Ty::f32(vec![nh]), p("ssm.d"));
    let norm_w = take_arg(decls, idx, Ty::f32(vec![di]), p("ssm.norm_w"));
    let out_proj = take_weight(b, decls, idx, c, h, di, p("ssm.out_proj"));
    let out_proj_b = ssm
        .use_bias
        .then(|| take_arg(decls, idx, Ty::f32(vec![h]), p("ssm.out_proj_b")));
    LayerW {
        down: None,
        gate: None,
        in_ln: Some(in_ln),
        post_ln: None,
        up: None,
        wk: None,
        wo: None,
        wq: None,
        wv: None,
        mla: None,
        bk: None,
        bq: None,
        bv: None,
        q_norm: None,
        k_norm: None,
        pre_ff_ln: None,
        post_ff_ln: None,
        in_ln_bias: None,
        post_ln_bias: None,
        wo_bias: None,
        down_bias: None,
        gate_bias: None,
        up_bias: None,
        moe: None,
        phi4_lora: None,
        ssm: Some(Mamba2MixerW {
            in_proj,
            in_proj_b,
            conv_w,
            conv_b,
            dt_bias,
            a_log,
            d,
            norm_w,
            out_proj,
            out_proj_b,
        }),
    }
}

/// One pure Mamba2 block, `x + mixer(norm(x))`, at the rank of `layout`. The
/// layer's state row is read from and written back to `state`. A Mamba2 block
/// has no FFN (`Config::layer_has_ffn` is false), so nothing follows the mixer.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_mamba2_block(
    b: &mut Builder,
    c: &Config,
    k: &Consts,
    lw: &LayerW,
    li: usize,
    x: &Val,
    layout: &AttnLayout,
    state: &mut RecurrentLayers,
) -> Val {
    let ssm = c
        .ssm
        .as_ref()
        .expect("a recurrent layer implies an SsmConfig");
    let w = lw
        .ssm
        .as_ref()
        .expect("a recurrent layer takes Mamba2 mixer weights");
    let s = c
        .state_layer_index(li)
        .expect("a recurrent layer has a state row");
    debug_assert!(!c.layer_has_ffn(li), "a Mamba2 block has no FFN");
    let hn = arch_norm(b, c, k, layout, x, &lw.in_ln, None);
    let mixed = match layout {
        AttnLayout::Prefill { .. } => {
            let real_len = state
                .real_len
                .clone()
                .expect("the prefill threads real_len to its recurrent layers");
            let current = &state.layers[s];
            let (y, h, conv) =
                ssm::mamba2_mixer_prefill(b, ssm, w, &hn, &real_len, &current.h, &current.conv);
            state.layers[s] = SsmLayerState { h, conv };
            y
        }
        AttnLayout::Single { .. } => {
            let row = b.reshape(&hn, vec![1, c.hidden]);
            let (y, next) = ssm::mamba2_mixer_step(b, ssm, w, &row, &state.layers[s]);
            state.layers[s] = next;
            b.reshape(&y, vec![c.hidden])
        }
        AttnLayout::Ragged { .. } => {
            // `[B, hidden]` rows, one per slot, each advancing its own state row.
            let (y, next) = ssm::mamba2_mixer_step(b, ssm, w, &hn, &state.layers[s]);
            state.layers[s] = next;
            y
        }
    };
    b.add(x, &mixed)
}

/// Declare one prefill input per state tensor, at its per-sequence shape.
pub(super) fn take_state_args(decls: &mut Vec<ArgDecl>, idx: &mut usize, c: &Config) -> Vec<Val> {
    c.state_tensors()
        .into_iter()
        .map(|spec| take_arg(decls, idx, Ty::f32(spec.dims), spec.name.to_string()))
        .collect()
}

/// Declare one ragged-decode input per state tensor, at `[B, ...dims]`.
pub(super) fn take_batched_state_args(
    decls: &mut Vec<ArgDecl>,
    idx: &mut usize,
    c: &Config,
    bsz: usize,
) -> Vec<Val> {
    c.state_tensors()
        .into_iter()
        .map(|spec| {
            let mut dims = vec![bsz];
            dims.extend(spec.dims);
            take_arg(decls, idx, Ty::f32(dims), spec.name.to_string())
        })
        .collect()
}

/// Load each state tensor's decode global.
pub(super) fn load_state_globals(b: &mut Builder, c: &Config) -> Vec<Val> {
    c.state_tensors()
        .into_iter()
        .map(|spec| b.global_load(spec.name, Ty::f32(spec.dims)))
        .collect()
}

/// The decode module's state globals and its `@state_import` entry. Both are
/// empty for a config without recurrent state.
pub(super) fn decode_state_entries(c: &Config) -> (String, String) {
    let specs = c.state_tensors();
    if specs.is_empty() {
        return (String::new(), String::new());
    }
    let mut globals = String::new();
    let mut params = Vec::new();
    let mut stores = String::new();
    for (i, spec) in specs.iter().enumerate() {
        let ty = Ty::f32(spec.dims.clone()).render();
        globals.push_str(&format!(
            "  util.global private mutable @{} : {ty}\n",
            spec.name
        ));
        params.push(format!("%s{i}: {ty}"));
        stores.push_str(&format!(
            "    util.global.store %s{i}, @{} : {ty}\n",
            spec.name
        ));
    }
    let import = format!(
        "  func.func public @state_import({}) {{\n{stores}    return\n  }}\n",
        params.join(", ")
    );
    (globals, import)
}
