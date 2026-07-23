//! Gemma3n StableHLO prefill graphs.
//!
//! Both entries share the complete AltUp/LAUREL/shared-KV backbone. Token
//! prefill gathers ordinary and per-layer embeddings; multimodal prefill accepts
//! already post-scale merged embeddings plus dense projected PLE.

use super::builder::{Builder, Precision, Val, precision_from_env};
use super::gemma3n::{Gemma3nConfig, Gemma3nLayerType};
use super::gemma3n_emit_ops::{
    Constants, altup_correct, altup_predict, apply_sliding_window, attention, bf16_scalar,
    causal_mask, constants, geglu, linear_bf16, linear_seq_bf16, mean_planes_bf16, normalize_to,
    rms_last_bf16, round_bf16, sparse_gelu,
};
use super::gemma3n_qmv;
use super::gemma3n_schema::{Args, Decl, Input, Weights, build_schema};

pub(crate) fn emit_gemma3n_prefill(c: &Gemma3nConfig, sample: bool) -> String {
    emit(c, sample, false, precision_from_env())
}

pub(crate) fn emit_gemma3n_prefill_with(
    c: &Gemma3nConfig,
    sample: bool,
    precision: Precision,
) -> String {
    emit(c, sample, false, precision)
}

pub(crate) fn emit_gemma3n_prefill_with_qmv(
    c: &Gemma3nConfig,
    sample: bool,
    precision: Precision,
    native_qmv: bool,
) -> String {
    emit_inner(c, sample, false, precision, false, false, native_qmv)
}

pub(crate) fn emit_gemma3n_prefill_embeddings_ple(c: &Gemma3nConfig, sample: bool) -> String {
    emit(c, sample, true, precision_from_env())
}

pub(crate) fn emit_gemma3n_prefill_embeddings_ple_with(
    c: &Gemma3nConfig,
    sample: bool,
    precision: Precision,
) -> String {
    emit(c, sample, true, precision)
}

pub(crate) fn emit_gemma3n_prefill_embeddings_ple_with_qmv(
    c: &Gemma3nConfig,
    sample: bool,
    precision: Precision,
    native_qmv: bool,
) -> String {
    emit_inner(c, sample, true, precision, false, false, native_qmv)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gemma3nDiagnosticSegment {
    pub name: &'static str,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gemma3nDiagnosticLayout {
    pub segments: Vec<Gemma3nDiagnosticSegment>,
    pub total_len: usize,
}

impl Gemma3nDiagnosticLayout {
    pub fn validate(&self) -> Result<(), String> {
        let mut offset = 0usize;
        for segment in &self.segments {
            let len = segment
                .shape
                .iter()
                .try_fold(1usize, |product, dim| product.checked_mul(*dim))
                .ok_or_else(|| format!("diagnostic segment {} overflows", segment.name))?;
            if segment.offset != offset || segment.len != len {
                return Err(format!(
                    "diagnostic segment {} has offset/len {}/{}, expected {offset}/{len}",
                    segment.name, segment.offset, segment.len
                ));
            }
            offset = offset
                .checked_add(len)
                .ok_or_else(|| "diagnostic layout length overflows".to_string())?;
        }
        if offset != self.total_len {
            return Err(format!(
                "diagnostic layout total_len={} but segments end at {offset}",
                self.total_len
            ));
        }
        Ok(())
    }

    pub fn segment(&self, name: &str) -> Option<&Gemma3nDiagnosticSegment> {
        self.segments.iter().find(|segment| segment.name == name)
    }
}

pub fn gemma3n_diagnostic_layout(c: &Gemma3nConfig) -> Gemma3nDiagnosticLayout {
    let capacity = c.context_capacity;
    let shapes = [
        ("scaled_embeddings", vec![capacity, c.hidden]),
        (
            "projected_ple",
            vec![capacity, c.n_layers, c.hidden_per_layer_input],
        ),
        ("layer0_laurel", vec![capacity, c.hidden]),
        ("layer0_ple_injected", vec![capacity, c.hidden]),
        (
            "layer0_all_planes",
            vec![c.altup_num_inputs, capacity, c.hidden],
        ),
        ("layer0_active_plane", vec![capacity, c.hidden]),
        (
            "layer_mid_all_planes",
            vec![c.altup_num_inputs, capacity, c.hidden],
        ),
        ("layer_mid_active_plane", vec![capacity, c.hidden]),
        (
            "layer_last_all_planes",
            vec![c.altup_num_inputs, capacity, c.hidden],
        ),
        ("layer_last_active_plane", vec![capacity, c.hidden]),
        ("layer0_k", vec![capacity, c.n_kv, c.head_dim]),
        ("layer0_v", vec![capacity, c.n_kv, c.head_dim]),
        ("final_hidden", vec![capacity, c.hidden]),
        ("logits", vec![c.vocab]),
    ];
    let mut offset = 0usize;
    let segments = shapes
        .into_iter()
        .map(|(name, shape)| {
            let len = shape.iter().product();
            let segment = Gemma3nDiagnosticSegment {
                name,
                shape,
                offset,
                len,
            };
            offset += len;
            segment
        })
        .collect();
    Gemma3nDiagnosticLayout {
        segments,
        total_len: offset,
    }
}

fn gemma3n_all_layer_diagnostic_window(c: &Gemma3nConfig) -> (usize, usize) {
    const START_ENV: &str = "MLXCEL_XLA_GEMMA3N_TRACE_LAYER_START";
    let start = std::env::var(START_ENV)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .unwrap_or_else(|error| panic!("{START_ENV} must be an unsigned integer: {error}"))
        })
        .unwrap_or(0);
    assert!(
        start < c.n_layers,
        "{START_ENV}={start} must be less than the Gemma3n layer count {}",
        c.n_layers
    );
    (start, (c.n_layers - start).min(10))
}

pub fn gemma3n_all_layer_diagnostic_layout(c: &Gemma3nConfig) -> Gemma3nDiagnosticLayout {
    // Keep this targeted trace deliberately bounded. It exists to locate the
    // first cross-runtime divergence in one ten-layer window, not to expose
    // every capacity row or retain every logical layer in one IREE result.
    let (_, layer_count) = gemma3n_all_layer_diagnostic_window(c);
    let rows = c.context_capacity.min(3);
    let shapes = [
        (
            "all_layer_all_planes",
            vec![layer_count, c.altup_num_inputs, rows, c.hidden],
        ),
        ("all_layer_active_planes", vec![layer_count, rows, c.hidden]),
    ];
    let mut offset = 0usize;
    let segments = shapes
        .into_iter()
        .map(|(name, shape)| {
            let len = shape.iter().product();
            let segment = Gemma3nDiagnosticSegment {
                name,
                shape,
                offset,
                len,
            };
            offset += len;
            segment
        })
        .collect();
    Gemma3nDiagnosticLayout {
        segments,
        total_len: offset,
    }
}

#[cfg(feature = "diagnostics")]
pub(crate) fn emit_gemma3n_prefill_diagnostics_with(
    c: &Gemma3nConfig,
    precision: Precision,
) -> (String, Gemma3nDiagnosticLayout) {
    let layout = gemma3n_diagnostic_layout(c);
    layout
        .validate()
        .expect("validated Gemma3n diagnostic layout");
    (
        emit_inner(c, false, false, precision, true, false, false),
        layout,
    )
}

#[cfg(feature = "diagnostics")]
pub(crate) fn emit_gemma3n_prefill_diagnostics_with_qmv(
    c: &Gemma3nConfig,
    precision: Precision,
    native_qmv: bool,
) -> (String, Gemma3nDiagnosticLayout) {
    let layout = gemma3n_diagnostic_layout(c);
    layout
        .validate()
        .expect("validated Gemma3n diagnostic layout");
    (
        emit_inner(c, false, false, precision, true, false, native_qmv),
        layout,
    )
}

#[cfg(feature = "diagnostics")]
pub(crate) fn emit_gemma3n_all_layer_diagnostics_with_qmv(
    c: &Gemma3nConfig,
    precision: Precision,
    native_qmv: bool,
) -> (String, Gemma3nDiagnosticLayout) {
    let layout = gemma3n_all_layer_diagnostic_layout(c);
    layout
        .validate()
        .expect("validated Gemma3n all-layer diagnostic layout");
    (
        emit_inner(c, false, false, precision, false, true, native_qmv),
        layout,
    )
}

fn emit(c: &Gemma3nConfig, sample: bool, prepared: bool, precision: Precision) -> String {
    emit_inner(c, sample, prepared, precision, false, false, false)
}

fn emit_inner(
    c: &Gemma3nConfig,
    sample: bool,
    prepared: bool,
    precision: Precision,
    diagnostics: bool,
    all_layer_diagnostics: bool,
    native_qmv: bool,
) -> String {
    assert!(
        !(diagnostics && all_layer_diagnostics),
        "Gemma3n diagnostic modes are mutually exclusive"
    );
    let lp = c.context_capacity;
    // Gemma3n has an explicit BF16 activation stream with small F32 AltUp
    // coefficient islands. A global contraction precision would incorrectly
    // demote those coefficient matrices (and CUDA's generic default is F16), so
    // keep the builder F32 and place BF16 boundaries locally.
    let _ = precision;
    let mut b = Builder::new().with_gemma3n_qmv(native_qmv);
    let (decls, args) = build_schema(c, prepared, lp, false, false);
    let k = constants(&mut b, c);
    let (base, dense_ple) = input_head(&mut b, c, &k, &args, lp);
    let diagnostic_base = diagnostics.then(|| base.clone());
    let diagnostic_ple = diagnostics.then(|| dense_ple.clone());
    let mut diagnostic_layer0_laurel = None;
    let mut diagnostic_layer0_ple_injected = None;
    let mut diagnostic_layer0_planes = None;
    let mut diagnostic_layer_mid_planes = None;
    let mut diagnostic_layer_last_planes = None;
    let mut diagnostic_layer0_k = None;
    let mut diagnostic_layer0_v = None;
    let mut diagnostic_all_layer_planes = Vec::new();
    let target = magnitude(&mut b, &base, c, &k);
    let mut planes = vec![base.clone()];
    for projection in &args.weights.initial_projections {
        let projected = linear_seq_bf16(&mut b, &base, projection);
        planes.push(normalize_to(&mut b, &projected, &target, &k));
    }
    let full_mask = args
        .attention_bias
        .clone()
        .unwrap_or_else(|| causal_mask(&mut b, lp, None, &k));
    let sliding_mask = apply_sliding_window(&mut b, &full_mask, lp, c.sliding_window, &k);
    let cache_map = c
        .kv_cache_contract()
        .expect("validated Gemma3n KV map")
        .logical_to_physical;
    let cache_shape = vec![c.kv_cache_layers(), lp, c.n_kv, c.head_dim];
    let mut kcache = b.broadcast(&k.zero, &[], cache_shape.clone());
    let mut vcache = b.broadcast(&k.zero, &[], cache_shape.clone());
    for (layer, &cache_index) in cache_map.iter().enumerate() {
        let lw = &args.weights.layers[layer];
        let predicted = altup_predict(&mut b, &planes, lw, c, &k, lp);
        let active = &predicted[c.altup_active_idx];
        let normalized = rms_last_bf16(&mut b, active, Some(&lw.input_norm), &k.eps, &k.zero);
        let laurel = linear_seq_bf16(&mut b, &normalized, &lw.laurel_left);
        let laurel = linear_seq_bf16(&mut b, &laurel, &lw.laurel_right);
        let laurel = rms_last_bf16(&mut b, &laurel, Some(&lw.laurel_norm), &k.eps, &k.zero);
        let laurel = b.add(&normalized, &laurel);
        let laurel = round_bf16(&mut b, &laurel);
        let mask = match c.layer_types[layer] {
            Gemma3nLayerType::Full => &full_mask,
            Gemma3nLayerType::Sliding => &sliding_mask,
        };
        let attended = attention(
            &mut b,
            &normalized,
            &args.positions,
            &args.real_len,
            mask,
            lw,
            layer,
            cache_index,
            c,
            &k,
            &mut kcache,
            &mut vcache,
        );
        let attended = rms_last_bf16(&mut b, &attended, Some(&lw.post_attn_norm), &k.eps, &k.zero);
        let inv = b.broadcast(&k.inv_sqrt2, &[], vec![lp, c.hidden]);
        let residual = b.add(active, &attended);
        let residual = round_bf16(&mut b, &residual);
        let residual = b.add(&residual, &laurel);
        let residual = round_bf16(&mut b, &residual);
        let residual = b.multiply(&residual, &inv);
        let residual = round_bf16(&mut b, &residual);
        let ff_input = rms_last_bf16(&mut b, &residual, Some(&lw.pre_ff_norm), &k.eps, &k.zero);
        let gate = linear_seq_bf16(&mut b, &ff_input, &lw.gate);
        let up = linear_seq_bf16(&mut b, &ff_input, &lw.up);
        let mlp = if c.activation_sparsity[layer] > 0.0 {
            let activated_gate = sparse_gelu(&mut b, &gate, c.activation_sparsity[layer], &k);
            let product = b.multiply(&activated_gate, &up);
            round_bf16(&mut b, &product)
        } else {
            geglu(&mut b, &gate, &up)
        };
        let mlp = linear_seq_bf16(&mut b, &mlp, &lw.down);
        let mlp = rms_last_bf16(&mut b, &mlp, Some(&lw.post_ff_norm), &k.eps, &k.zero);
        let activated = b.add(&residual, &mlp);
        let activated = round_bf16(&mut b, &activated);
        planes = altup_correct(&mut b, &predicted, &activated, lw, c, &k, lp);
        let mut corrected_active = planes[c.altup_active_idx].clone();
        if c.altup_correct_scale {
            let scale = b.broadcast(&lw.correct_scale, &[1], vec![lp, c.hidden]);
            corrected_active = b.multiply(&corrected_active, &scale);
            corrected_active = round_bf16(&mut b, &corrected_active);
        }
        let ple = b.slice(
            &dense_ple,
            &[(0, lp), (layer, layer + 1), (0, c.hidden_per_layer_input)],
        );
        let ple = b.reshape(&ple, vec![lp, c.hidden_per_layer_input]);
        let ple_gate = linear_seq_bf16(&mut b, &corrected_active, &lw.ple_gate);
        let injected = geglu(&mut b, &ple_gate, &ple);
        let injected = linear_seq_bf16(&mut b, &injected, &lw.ple_projection);
        let injected = rms_last_bf16(&mut b, &injected, Some(&lw.ple_norm), &k.eps, &k.zero);
        for (plane, value) in planes.iter_mut().enumerate() {
            if plane != c.altup_active_idx {
                *value = b.add(value, &injected);
                *value = round_bf16(&mut b, value);
            }
        }
        if all_layer_diagnostics {
            let (trace_layer_start, trace_layer_count) = gemma3n_all_layer_diagnostic_window(c);
            if layer >= trace_layer_start && layer < trace_layer_start + trace_layer_count {
                diagnostic_all_layer_planes.push(planes.clone());
            }
        }
        if diagnostics {
            if layer == 0 {
                let layer0_cache = c
                    .kv_cache_contract()
                    .expect("validated Gemma3n KV map")
                    .logical_to_physical[0];
                assert_eq!(
                    cache_index, layer0_cache,
                    "layer0 diagnostic must capture its concrete physical cache"
                );
                diagnostic_layer0_laurel = Some(laurel.clone());
                diagnostic_layer0_ple_injected = Some(injected.clone());
                diagnostic_layer0_planes = Some(planes.clone());
                let keys = b.slice(
                    &kcache,
                    &[
                        (cache_index, cache_index + 1),
                        (0, lp),
                        (0, c.n_kv),
                        (0, c.head_dim),
                    ],
                );
                diagnostic_layer0_k = Some(b.reshape(&keys, vec![lp, c.n_kv, c.head_dim]));
                let values = b.slice(
                    &vcache,
                    &[
                        (cache_index, cache_index + 1),
                        (0, lp),
                        (0, c.n_kv),
                        (0, c.head_dim),
                    ],
                );
                diagnostic_layer0_v = Some(b.reshape(&values, vec![lp, c.n_kv, c.head_dim]));
            }
            if layer == c.n_layers / 2 {
                diagnostic_layer_mid_planes = Some(planes.clone());
            }
            if layer + 1 == c.n_layers {
                diagnostic_layer_last_planes = Some(planes.clone());
            }
        }
    }
    let target = magnitude(&mut b, &planes[c.altup_active_idx], c, &k);
    let mut collapsed_planes = vec![planes[c.altup_active_idx].clone()];
    for (index, projection) in args.weights.unembed_projections.iter().enumerate() {
        let projected = linear_seq_bf16(&mut b, &planes[index + 1], projection);
        let projected = normalize_to(&mut b, &projected, &target, &k);
        collapsed_planes.push(projected);
    }
    assert_eq!(collapsed_planes.len(), c.altup_num_inputs);
    let collapsed = mean_planes_bf16(&mut b, &collapsed_planes, &k.zero);
    let normalized = rms_last_bf16(
        &mut b,
        &collapsed,
        Some(&args.weights.final_norm),
        &k.eps,
        &k.zero,
    );
    let one = b.const_i32(1);
    let last_index = b.subtract(&args.real_len, &one);
    let zero = b.const_i32(0);
    let row = b.dynamic_slice(&normalized, &[&last_index, &zero], vec![1, c.hidden]);
    let row = b.reshape(&row, vec![c.hidden]);
    let logits = linear_bf16(&mut b, &row, &args.weights.embed);
    let logits = if let Some(cap) = c.final_logit_softcap {
        let cap = bf16_scalar(&mut b, cap);
        let cap_b = b.broadcast(&cap, &[], vec![c.vocab]);
        let logits = b.divide(&logits, &cap_b);
        let logits = round_bf16(&mut b, &logits);
        let logits = b.gemma3n_tanh(&logits);
        let logits = round_bf16(&mut b, &logits);
        let logits = b.multiply(&logits, &cap_b);
        round_bf16(&mut b, &logits)
    } else {
        logits
    };
    let (result, result_ty) = if all_layer_diagnostics {
        let (trace_layer_start, trace_layer_count) = gemma3n_all_layer_diagnostic_window(c);
        let trace_rows = c.context_capacity.min(3);
        assert_eq!(
            diagnostic_all_layer_planes.len(),
            trace_layer_count,
            "Gemma3n layer-window diagnostics must capture the bounded logical-layer window \
             starting at {trace_layer_start}"
        );
        let layout = gemma3n_all_layer_diagnostic_layout(c);
        let all_planes = flatten_all_layer_diagnostic_planes(
            &mut b,
            &diagnostic_all_layer_planes,
            trace_rows,
            c.hidden,
        );
        let active = b.slice(
            &diagnostic_all_layer_planes[0][c.altup_active_idx],
            &[(0, trace_rows), (0, c.hidden)],
        );
        let mut active_planes = b.reshape(&active, vec![trace_rows * c.hidden]);
        for layer_planes in &diagnostic_all_layer_planes[1..] {
            let active = b.slice(
                &layer_planes[c.altup_active_idx],
                &[(0, trace_rows), (0, c.hidden)],
            );
            let active = b.reshape(&active, vec![trace_rows * c.hidden]);
            active_planes = b.concatenate(&active_planes, &active, 0);
        }
        let all_planes = b.reshape(&all_planes, vec![layout.segments[0].len]);
        let active_planes = b.reshape(&active_planes, vec![layout.segments[1].len]);
        let flat = b.concatenate(&all_planes, &active_planes, 0);
        assert_eq!(flat.ty.shape, vec![layout.total_len]);
        (flat.name, flat.ty.render())
    } else if diagnostics {
        let layout = gemma3n_diagnostic_layout(c);
        let layer0_planes =
            diagnostic_layer0_planes.expect("Gemma3n diagnostics capture layer0 planes");
        let layer_mid_planes =
            diagnostic_layer_mid_planes.expect("Gemma3n diagnostics capture middle-layer planes");
        let layer_last_planes =
            diagnostic_layer_last_planes.expect("Gemma3n diagnostics capture last-layer planes");
        let layer0_active = layer0_planes[c.altup_active_idx].clone();
        let layer_mid_active = layer_mid_planes[c.altup_active_idx].clone();
        let layer_last_active = layer_last_planes[c.altup_active_idx].clone();
        let layer0_all = flatten_diagnostic_planes(&mut b, &layer0_planes, lp, c.hidden);
        let layer_mid_all = flatten_diagnostic_planes(&mut b, &layer_mid_planes, lp, c.hidden);
        let layer_last_all = flatten_diagnostic_planes(&mut b, &layer_last_planes, lp, c.hidden);
        let values = [
            diagnostic_base.expect("Gemma3n diagnostics capture input"),
            diagnostic_ple.expect("Gemma3n diagnostics capture PLE"),
            diagnostic_layer0_laurel.expect("Gemma3n diagnostics capture layer0 LAUREL"),
            diagnostic_layer0_ple_injected
                .expect("Gemma3n diagnostics capture layer0 PLE injection"),
            layer0_all,
            layer0_active,
            layer_mid_all,
            layer_mid_active,
            layer_last_all,
            layer_last_active,
            diagnostic_layer0_k.expect("Gemma3n diagnostics capture layer0 K"),
            diagnostic_layer0_v.expect("Gemma3n diagnostics capture layer0 V"),
            normalized.clone(),
            logits.clone(),
        ];
        let mut flat = b.reshape(&values[0], vec![layout.segments[0].len]);
        for (value, segment) in values.iter().skip(1).zip(layout.segments.iter().skip(1)) {
            let value = b.reshape(value, vec![segment.len]);
            flat = b.concatenate(&flat, &value, 0);
        }
        assert_eq!(flat.ty.shape, vec![layout.total_len]);
        (flat.name, flat.ty.render())
    } else if sample {
        let token = b.argmax(&logits);
        (token.name, token.ty.render())
    } else {
        (logits.name, logits.ty.render())
    };
    render_module(
        if prepared {
            "prefill_embeddings_ple"
        } else if diagnostics || all_layer_diagnostics {
            "prefill_diagnostics"
        } else {
            "prefill"
        },
        &decls,
        &b,
        &result,
        &result_ty,
        &kcache,
        &vcache,
    )
}

fn flatten_diagnostic_planes(b: &mut Builder, planes: &[Val], rows: usize, hidden: usize) -> Val {
    let mut flattened = b.reshape(&planes[0], vec![rows * hidden]);
    for plane in &planes[1..] {
        let plane = b.reshape(plane, vec![rows * hidden]);
        flattened = b.concatenate(&flattened, &plane, 0);
    }
    flattened
}

fn flatten_all_layer_diagnostic_planes(
    b: &mut Builder,
    layers: &[Vec<Val>],
    rows: usize,
    hidden: usize,
) -> Val {
    let mut flattened = flatten_diagnostic_plane_prefix(b, &layers[0], rows, hidden);
    for planes in &layers[1..] {
        let layer = flatten_diagnostic_plane_prefix(b, planes, rows, hidden);
        flattened = b.concatenate(&flattened, &layer, 0);
    }
    flattened
}

fn flatten_diagnostic_plane_prefix(
    b: &mut Builder,
    planes: &[Val],
    rows: usize,
    hidden: usize,
) -> Val {
    let first = b.slice(&planes[0], &[(0, rows), (0, hidden)]);
    let mut flattened = b.reshape(&first, vec![rows * hidden]);
    for plane in &planes[1..] {
        let plane = b.slice(plane, &[(0, rows), (0, hidden)]);
        let plane = b.reshape(&plane, vec![rows * hidden]);
        flattened = b.concatenate(&flattened, &plane, 0);
    }
    flattened
}

pub(super) fn input_head(
    b: &mut Builder,
    c: &Gemma3nConfig,
    k: &Constants,
    a: &Args,
    lp: usize,
) -> (Val, Val) {
    match &a.input {
        Input::Prepared {
            embeddings,
            dense_ple,
        } => (round_bf16(b, embeddings), round_bf16(b, dense_ple)),
        Input::Tokens(tokens) => token_input_head(b, c, k, &a.weights, tokens, lp),
    }
}

pub(super) fn token_input_head(
    b: &mut Builder,
    c: &Gemma3nConfig,
    k: &Constants,
    weights: &Weights,
    tokens: &Val,
    lp: usize,
) -> (Val, Val) {
    let ple_width = c.n_layers * c.hidden_per_layer_input;
    let tokens = if tokens.ty.shape.is_empty() {
        b.reshape(tokens, vec![1])
    } else {
        tokens.clone()
    };
    let indices = b.reshape(&tokens, vec![lp, 1]);
    let base = b.gather(&weights.embed, &indices);
    let scale = bf16_scalar(b, (c.hidden as f32).sqrt());
    let scale = b.broadcast(&scale, &[], vec![lp, c.hidden]);
    let base = b.multiply(&base, &scale);
    let base = round_bf16(b, &base);
    let limit = b.const_i32(c.per_layer_vocab as i32);
    let limit = b.broadcast(&limit, &[], vec![lp]);
    let zero_i = b.const_i32(0);
    let zero_ids = b.broadcast(&zero_i, &[], vec![lp]);
    let nonnegative = b.compare("GE", &tokens, &zero_ids, "SIGNED");
    let below = b.compare("LT", &tokens, &limit, "SIGNED");
    let valid = b.select(&nonnegative, &below, &nonnegative);
    let safe = b.select(&valid, &tokens, &zero_ids);
    let safe = b.reshape(&safe, vec![lp, 1]);
    let token_ple = b.gather(&weights.token_ple, &safe);
    let zeros = b.broadcast(&k.zero, &[], vec![lp, ple_width]);
    let valid = b.broadcast(&valid, &[0], vec![lp, ple_width]);
    let token_ple = b.select(&valid, &token_ple, &zeros);
    let token_scale = bf16_scalar(b, (c.hidden_per_layer_input as f32).sqrt());
    let token_scale = b.broadcast(&token_scale, &[], vec![lp, ple_width]);
    let token_ple = b.multiply(&token_ple, &token_scale);
    let token_ple = round_bf16(b, &token_ple);
    let projected = linear_seq_bf16(b, &base, &weights.ple_projection);
    let model_scale = bf16_scalar(b, (c.hidden as f32).sqrt().recip());
    let model_scale = b.broadcast(&model_scale, &[], vec![lp, ple_width]);
    let projected = b.multiply(&projected, &model_scale);
    let projected = round_bf16(b, &projected);
    let projected = b.reshape(&projected, vec![lp, c.n_layers, c.hidden_per_layer_input]);
    let projected = rms_last_bf16(
        b,
        &projected,
        Some(&weights.ple_projection_norm),
        &k.eps,
        &k.zero,
    );
    let token_ple = b.reshape(&token_ple, vec![lp, c.n_layers, c.hidden_per_layer_input]);
    let inv = b.broadcast(
        &k.inv_sqrt2,
        &[],
        vec![lp, c.n_layers, c.hidden_per_layer_input],
    );
    let combined = b.add(&projected, &token_ple);
    let combined = round_bf16(b, &combined);
    let combined = b.multiply(&combined, &inv);
    let combined = round_bf16(b, &combined);
    (base, combined)
}

fn magnitude(b: &mut Builder, value: &Val, c: &Gemma3nConfig, k: &Constants) -> Val {
    let sq = b.multiply(value, value);
    let sq = round_bf16(b, &sq);
    let sum = b.reduce_add(&sq, 1, &k.zero);
    let sum = round_bf16(b, &sum);
    let width = b.broadcast(&k.hidden, &[], vec![c.context_capacity]);
    let mean = b.divide(&sum, &width);
    let mean = round_bf16(b, &mean);
    let magnitude = b.sqrt(&mean);
    round_bf16(b, &magnitude)
}

fn render_module(
    name: &str,
    decls: &[Decl],
    b: &Builder,
    result: &str,
    result_ty: &str,
    kcache: &Val,
    vcache: &Val,
) -> String {
    let signature = decls
        .iter()
        .enumerate()
        .map(|(index, decl)| format!("%arg{index}: {} loc(\"{}\")", decl.ty.render(), decl.loc))
        .collect::<Vec<_>>()
        .join(", ");
    let cache_ty = kcache.ty.render();
    let target_alias = if b.gemma3n_qmv_enabled() {
        gemma3n_qmv::target_alias()
    } else {
        ""
    };
    let executable_source = if b.gemma3n_qmv_enabled() {
        gemma3n_qmv::executable_source()
    } else {
        String::new()
    };
    format!(
        "{target_alias}module @{name} {{\n{executable_source}  func.func public @main({signature}) -> ({result_ty}, \
         {cache_ty}, {cache_ty}) {{\n{body}    return {result}, {kc}, {vc} : \
         {result_ty}, {cache_ty}, {cache_ty}\n  }}\n}}\n",
        body = b.body(),
        kc = kcache.name,
        vc = vcache.name,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Gemma3nConfig {
        Gemma3nConfig::from_json_str(
            &serde_json::json!({
                "model_type": "gemma3n_text",
                "hidden_size": 8, "intermediate_size": [12, 12, 12, 12],
                "max_position_embeddings": 4096,
                "num_hidden_layers": 4, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 4, "rms_norm_eps": 1e-6,
                "vocab_size": 12, "vocab_size_per_layer_input": 10,
                "hidden_size_per_layer_input": 2,
                "layer_types": ["sliding_attention", "full_attention",
                                "sliding_attention", "full_attention"],
                "activation_sparsity_pattern": [0.5, 0.0, 0.0, 0.0],
                "sliding_window": 2, "rope_theta": 1000000.0,
                "rope_local_base_freq": 10000.0, "final_logit_softcapping": 30.0,
                "num_kv_shared_layers": 2, "altup_num_inputs": 2,
                "altup_active_idx": 0, "altup_coef_clip": 120.0,
                "altup_correct_scale": true, "laurel_rank": 2,
                "tie_word_embeddings": true
            })
            .to_string(),
        )
        .unwrap()
        .with_context_capacity(4)
        .unwrap()
    }

    #[test]
    fn emits_distinct_token_and_dense_ple_entries_with_physical_kv_count() {
        let c = tiny();
        let token = emit_gemma3n_prefill(&c, false);
        let prepared = emit_gemma3n_prefill_embeddings_ple(&c, false);
        assert!(token.contains("module @prefill"));
        assert!(!token.contains("loc(\"dense_ple\")"));
        assert!(prepared.contains("module @prefill_embeddings_ple"));
        assert!(prepared.contains("tensor<4x4x2xf32> loc(\"dense_ple\")"));
        assert!(prepared.contains("tensor<2x4x1x4xf32>"));
        assert!(prepared.contains("altup.prediction_coefs.weight"));
        assert!(prepared.contains("laurel.linear_left.weight"));
        assert!(!prepared.contains("layers.2.self_attn.k_proj.weight"));
        let decode = super::super::gemma3n_decode::emit_gemma3n_decode(&c, true);
        assert!(decode.contains("module @decode_step"));
        assert!(decode.contains("tensor<2x4x1x4xf32> loc(\"kcache\")"));
        assert!(!decode.contains("layers.2.self_attn.k_proj.weight"));
        let ragged = super::super::gemma3n_decode::emit_gemma3n_decode_ragged(&c, 4, false);
        assert!(ragged.contains("tensor<4x2x4x1x4xf32> loc(\"kcache\")"));
        assert!(ragged.contains("tensor<4x12xf32>"));
        assert!(!ragged.contains("layers.2.self_attn.k_proj.weight"));
        #[cfg(xla_iree_cuda)]
        {
            let mut native = c.clone();
            native.altup_num_inputs = 4;
            for (name, graph) in [
                (
                    "normal",
                    super::super::gemma3n_decode::emit_gemma3n_decode_with_qmv(
                        &native,
                        true,
                        Precision::F32,
                        true,
                    ),
                ),
                (
                    "ragged",
                    super::super::gemma3n_decode::emit_gemma3n_decode_ragged_with_qmv(
                        &native,
                        4,
                        false,
                        Precision::F32,
                        true,
                    ),
                ),
            ] {
                assert!(graph.contains("gemma3n-qmv:kernel-v8:abi-v6:min-sm80"));
                assert!(graph.contains("hal.executable.source private @custom_qmv"));
                for export in [
                    "gemma3n_qmv",
                    "gemma3n_tanh",
                    "gemma3n_altup_coeff",
                    "gemma3n_altup_predict",
                    "gemma3n_geglu_bf16",
                ] {
                    assert!(
                        graph.contains(&format!("flow.dispatch @custom_qmv::@{export}")),
                        "{name} decode is missing native {export} dispatch"
                    );
                }
                assert!(
                    !graph.contains("flow.dispatch @custom_qmv::@gemma3n_sdpa_vector"),
                    "{name} D=4 decode must retain materialized attention"
                );
            }

            let mut native_sdpa = native.clone();
            native_sdpa.hidden = 512;
            native_sdpa.n_q = 2;
            native_sdpa.n_kv = 1;
            native_sdpa.head_dim = 256;
            native_sdpa.sliding_window = native_sdpa.context_capacity;
            for (name, graph, expected_dispatches) in [
                (
                    "normal",
                    super::super::gemma3n_decode::emit_gemma3n_decode_with_qmv(
                        &native_sdpa,
                        true,
                        Precision::F32,
                        true,
                    ),
                    native_sdpa.n_layers,
                ),
                (
                    "ragged",
                    super::super::gemma3n_decode::emit_gemma3n_decode_ragged_with_qmv(
                        &native_sdpa,
                        4,
                        false,
                        Precision::F32,
                        true,
                    ),
                    native_sdpa.n_layers * 4,
                ),
            ] {
                assert_eq!(
                    graph
                        .matches("flow.dispatch @custom_qmv::@gemma3n_sdpa_vector")
                        .count(),
                    expected_dispatches,
                    "{name} safe D=256 decode must use native SDPA for every layer"
                );
                assert!(
                    !graph.contains("stablehlo.exponential"),
                    "{name} all-native decode must omit materialized softmax"
                );
            }

            native_sdpa.sliding_window = native_sdpa.context_capacity - 1;
            let mixed = super::super::gemma3n_decode::emit_gemma3n_decode_with_qmv(
                &native_sdpa,
                true,
                Precision::F32,
                true,
            );
            assert_eq!(
                mixed
                    .matches("flow.dispatch @custom_qmv::@gemma3n_sdpa_vector")
                    .count(),
                2,
                "only full-attention layers are native when the sliding window truncates"
            );
            assert!(
                mixed.contains("stablehlo.exponential"),
                "truncated sliding-window layers must retain materialized softmax"
            );
        }
    }

    #[test]
    fn gemma3n_precision_is_local_and_keeps_altup_coefficients_f32() {
        let c = tiny();
        let f32_graph = emit_gemma3n_prefill_with(&c, false, Precision::F32);
        assert_eq!(
            f32_graph,
            emit_gemma3n_prefill_with(&c, false, Precision::F16)
        );
        assert_eq!(
            f32_graph,
            emit_gemma3n_prefill_with(&c, false, Precision::Bf16)
        );
        assert!(f32_graph.contains("-> tensor<4x8xbf16>"));
        assert!(f32_graph.contains("(tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x4xf32>"));
        assert!(f32_graph.contains("altup.prediction_coefs.weight"));
        assert!(f32_graph.contains("altup.correction_coefs.weight"));
    }

    #[test]
    fn absent_softcap_skips_only_the_final_logit_tanh() {
        let with_softcap = emit_gemma3n_prefill_with(&tiny(), false, Precision::Bf16);
        let mut without = tiny();
        without.final_logit_softcap = None;
        let without_softcap = emit_gemma3n_prefill_with(&without, false, Precision::Bf16);
        assert_eq!(
            with_softcap.matches("stablehlo.tanh").count(),
            without_softcap.matches("stablehlo.tanh").count() + 1
        );
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn diagnostic_layout_is_flat_stable_and_absent_from_normal_prefill() {
        let c = tiny();
        let expected = [
            ("scaled_embeddings", vec![4, 8], 0, 32),
            ("projected_ple", vec![4, 4, 2], 32, 32),
            ("layer0_laurel", vec![4, 8], 64, 32),
            ("layer0_ple_injected", vec![4, 8], 96, 32),
            ("layer0_all_planes", vec![2, 4, 8], 128, 64),
            ("layer0_active_plane", vec![4, 8], 192, 32),
            ("layer_mid_all_planes", vec![2, 4, 8], 224, 64),
            ("layer_mid_active_plane", vec![4, 8], 288, 32),
            ("layer_last_all_planes", vec![2, 4, 8], 320, 64),
            ("layer_last_active_plane", vec![4, 8], 384, 32),
            ("layer0_k", vec![4, 1, 4], 416, 16),
            ("layer0_v", vec![4, 1, 4], 432, 16),
            ("final_hidden", vec![4, 8], 448, 32),
            ("logits", vec![12], 480, 12),
        ];
        let (diagnostic, layout) = emit_gemma3n_prefill_diagnostics_with(&c, Precision::Bf16);
        layout.validate().unwrap();
        assert_eq!(layout.total_len, 492);
        assert_eq!(layout.segments.len(), expected.len());
        for (segment, (name, shape, offset, len)) in layout.segments.iter().zip(expected) {
            assert_eq!(segment.name, name);
            assert_eq!(segment.shape, shape);
            assert_eq!(segment.offset, offset);
            assert_eq!(segment.len, len);
        }
        assert!(diagnostic.contains("module @prefill_diagnostics"));
        assert!(diagnostic.contains("tensor<492xf32>"));

        let normal = emit_gemma3n_prefill_with(&c, false, Precision::Bf16);
        assert!(normal.contains("module @prefill"));
        assert!(!normal.contains("@prefill_diagnostics"));
        assert!(!normal.contains("tensor<492xf32>"));
    }

    #[test]
    #[ignore = "requires the pinned IREE compiler; run explicitly for the production target"]
    fn iree_compiles_tiny_token_and_dense_ple_graphs() {
        let compiler = std::env::var_os("MLXCEL_XLA_IREE_COMPILE")
            .expect("set MLXCEL_XLA_IREE_COMPILE to the pinned iree-compile");
        let target =
            std::env::var("MLXCEL_XLA_IREE_TEST_TARGET").unwrap_or_else(|_| "local".to_string());
        let graphs = vec![
            ("token", emit_gemma3n_prefill(&tiny(), true)),
            (
                "embeddings-ple",
                emit_gemma3n_prefill_embeddings_ple(&tiny(), true),
            ),
            (
                "decode",
                super::super::gemma3n_decode::emit_gemma3n_decode(&tiny(), true),
            ),
            (
                "decode-ragged-b4",
                super::super::gemma3n_decode::emit_gemma3n_decode_ragged(&tiny(), 4, false),
            ),
        ];
        #[cfg(feature = "diagnostics")]
        let graphs = {
            let mut graphs = graphs;
            graphs.push((
                "diagnostics",
                emit_gemma3n_prefill_diagnostics_with(&tiny(), Precision::Bf16).0,
            ));
            graphs
        };
        for (tag, graph) in graphs {
            let stem = format!("mlxcel-gemma3n-{tag}-{}", std::process::id());
            let input = std::env::temp_dir().join(format!("{stem}.mlir"));
            let output = std::env::temp_dir().join(format!("{stem}.vmfb"));
            std::fs::write(&input, graph).unwrap();
            let mut command = std::process::Command::new(&compiler);
            command.arg("--iree-input-type=stablehlo");
            if target == "cuda" {
                command.arg("--iree-hal-target-device=cuda");
            } else {
                command
                    .arg("--iree-hal-target-device=local")
                    .arg("--iree-hal-local-target-device-backends=llvm-cpu");
            }
            let result = command.arg(&input).arg("-o").arg(&output).output().unwrap();
            let _ = std::fs::remove_file(&input);
            let _ = std::fs::remove_file(&output);
            assert!(
                result.status.success(),
                "{tag} failed IREE compile: {}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }
}
