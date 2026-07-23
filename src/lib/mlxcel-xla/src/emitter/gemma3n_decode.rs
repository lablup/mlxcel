//! Single-token and fixed-slot ragged Gemma3n decode StableHLO graphs.

use super::builder::{Builder, Precision, Ty, Val, precision_from_env};
use super::gemma3n::Gemma3nConfig;
use super::gemma3n_emit::token_input_head;
use super::gemma3n_emit_ops::{
    Constants, altup_correct, altup_predict, attention_decode, bf16_scalar, constants, geglu,
    linear_bf16, linear_seq_bf16, mean_planes_bf16, normalize_to, rms_last_bf16, round_bf16,
    sparse_gelu,
};
use super::gemma3n_qmv;
use super::gemma3n_schema::{Decl, Input, Weights, build_schema, take};

pub(crate) fn emit_gemma3n_decode(c: &Gemma3nConfig, sample: bool) -> String {
    emit_gemma3n_decode_with(c, sample, precision_from_env())
}

pub(crate) fn emit_gemma3n_decode_with(
    c: &Gemma3nConfig,
    sample: bool,
    precision: Precision,
) -> String {
    emit_gemma3n_decode_with_qmv(c, sample, precision, false)
}

pub(crate) fn emit_gemma3n_decode_with_qmv(
    c: &Gemma3nConfig,
    sample: bool,
    precision: Precision,
    native_qmv: bool,
) -> String {
    // Gemma3n's explicit BF16 boundaries coexist with F32 AltUp coefficient
    // islands, so generic whole-builder contraction demotion is incorrect.
    let _ = precision;
    let mut b = Builder::new().with_gemma3n_qmv(native_qmv);
    let (mut decls, args) = build_schema(c, false, 1, true, false);
    let token = match &args.input {
        Input::Tokens(token) => token.clone(),
        Input::Prepared { .. } => unreachable!("decode accepts token input"),
    };
    let mut index = decls.len();
    let cache_ty = scalar_cache_ty(c);
    let mut kcache = take(&mut decls, &mut index, cache_ty.clone(), "kcache");
    let mut vcache = take(&mut decls, &mut index, cache_ty.clone(), "vcache");
    let k = constants(&mut b, c);
    let logits = decode_row(
        &mut b,
        c,
        &k,
        &args.weights,
        &token,
        &args.positions,
        &mut kcache,
        &mut vcache,
    );
    let result = if sample { b.argmax(&logits) } else { logits };
    render_decode(&decls, &b, &result, &kcache, &vcache, &cache_ty)
}

/// Emit the fixed-slot ragged decode used by continuous batching.
///
/// Each row executes the same scalar Gemma3n body against its own rank-4
/// physical shared-KV slab. The graph is statically unrolled for `b_max`, then
/// returns `[B,V]` logits and rank-5 caches. This preserves the existing
/// device-side slot population path without turning logical shared layers into
/// duplicate cache storage.
pub(crate) fn emit_gemma3n_decode_ragged(c: &Gemma3nConfig, b_max: usize, sample: bool) -> String {
    emit_gemma3n_decode_ragged_with(c, b_max, sample, precision_from_env())
}

pub(crate) fn emit_gemma3n_decode_ragged_with(
    c: &Gemma3nConfig,
    b_max: usize,
    sample: bool,
    precision: Precision,
) -> String {
    emit_gemma3n_decode_ragged_with_qmv(c, b_max, sample, precision, false)
}

pub(crate) fn emit_gemma3n_decode_ragged_with_qmv(
    c: &Gemma3nConfig,
    b_max: usize,
    sample: bool,
    precision: Precision,
    native_qmv: bool,
) -> String {
    assert!(
        b_max > 0,
        "Gemma3n ragged decode requires at least one slot"
    );
    let _ = precision;
    let mut b = Builder::new().with_gemma3n_qmv(native_qmv);
    let (mut decls, args) = build_schema(c, false, b_max, false, true);
    let tokens = match &args.input {
        Input::Tokens(tokens) => tokens.clone(),
        Input::Prepared { .. } => unreachable!("decode accepts token input"),
    };
    let mut index = decls.len();
    let scalar_cache = scalar_cache_ty(c);
    let cache_ty = Ty::f32(vec![
        b_max,
        c.kv_cache_layers(),
        c.context_capacity,
        c.n_kv,
        c.head_dim,
    ]);
    let mut kcache = take(&mut decls, &mut index, cache_ty.clone(), "kcache");
    let mut vcache = take(&mut decls, &mut index, cache_ty.clone(), "vcache");
    let k = constants(&mut b, c);
    let mut rows = Vec::with_capacity(b_max);
    for row in 0..b_max {
        let token = b.slice(&tokens, &[(row, row + 1)]);
        let token = b.reshape(&token, Vec::new());
        let position = b.slice(&args.positions, &[(row, row + 1)]);
        let position = b.reshape(&position, Vec::new());
        let mut row_k = b.slice(
            &kcache,
            &[
                (row, row + 1),
                (0, c.kv_cache_layers()),
                (0, c.context_capacity),
                (0, c.n_kv),
                (0, c.head_dim),
            ],
        );
        row_k = b.reshape(&row_k, scalar_cache.shape.clone());
        let mut row_v = b.slice(
            &vcache,
            &[
                (row, row + 1),
                (0, c.kv_cache_layers()),
                (0, c.context_capacity),
                (0, c.n_kv),
                (0, c.head_dim),
            ],
        );
        row_v = b.reshape(&row_v, scalar_cache.shape.clone());
        let logits = decode_row(
            &mut b,
            c,
            &k,
            &args.weights,
            &token,
            &position,
            &mut row_k,
            &mut row_v,
        );
        rows.push(b.reshape(&logits, vec![1, c.vocab]));

        let zero = b.const_i32(0);
        let row_index = b.const_i32(row as i32);
        let row_k = b.reshape(
            &row_k,
            vec![
                1,
                c.kv_cache_layers(),
                c.context_capacity,
                c.n_kv,
                c.head_dim,
            ],
        );
        kcache = b.dynamic_update_slice(&kcache, &row_k, &[&row_index, &zero, &zero, &zero, &zero]);
        let row_v = b.reshape(
            &row_v,
            vec![
                1,
                c.kv_cache_layers(),
                c.context_capacity,
                c.n_kv,
                c.head_dim,
            ],
        );
        vcache = b.dynamic_update_slice(&vcache, &row_v, &[&row_index, &zero, &zero, &zero, &zero]);
    }
    let mut rows = rows.into_iter();
    let mut logits = rows.next().expect("b_max is nonzero");
    for row in rows {
        logits = b.concatenate(&logits, &row, 0);
    }
    let result = if sample {
        b.argmax_batched(&logits)
    } else {
        logits
    };
    render_decode(&decls, &b, &result, &kcache, &vcache, &cache_ty)
}

#[allow(clippy::too_many_arguments)]
fn decode_row(
    b: &mut Builder,
    c: &Gemma3nConfig,
    k: &Constants,
    weights: &Weights,
    token: &Val,
    position: &Val,
    kcache: &mut Val,
    vcache: &mut Val,
) -> Val {
    let (base, dense_ple) = token_input_head(b, c, k, weights, token, 1);
    let target = magnitude(b, &base, c.hidden, &k.zero, &k.one);
    let mut planes = vec![base.clone()];
    for projection in &weights.initial_projections {
        let projected = linear_seq_bf16(b, &base, projection);
        planes.push(normalize_to(b, &projected, &target, k));
    }
    let cache_map = c
        .kv_cache_contract()
        .expect("validated Gemma3n KV map")
        .logical_to_physical;
    for (layer, &cache_index) in cache_map.iter().enumerate() {
        let lw = &weights.layers[layer];
        let predicted = altup_predict(b, &planes, lw, c, k, 1);
        let active = &predicted[c.altup_active_idx];
        let normalized = rms_last_bf16(b, active, Some(&lw.input_norm), &k.eps, &k.zero);
        let laurel = linear_seq_bf16(b, &normalized, &lw.laurel_left);
        let laurel = linear_seq_bf16(b, &laurel, &lw.laurel_right);
        let laurel = rms_last_bf16(b, &laurel, Some(&lw.laurel_norm), &k.eps, &k.zero);
        let laurel = b.add(&normalized, &laurel);
        let laurel = round_bf16(b, &laurel);
        let attended = attention_decode(
            b,
            &normalized,
            position,
            lw,
            layer,
            cache_index,
            c,
            k,
            kcache,
            vcache,
        );
        let attended = rms_last_bf16(b, &attended, Some(&lw.post_attn_norm), &k.eps, &k.zero);
        let sum = b.add(active, &attended);
        let sum = round_bf16(b, &sum);
        let sum = b.add(&sum, &laurel);
        let sum = round_bf16(b, &sum);
        let inv = b.broadcast(&k.inv_sqrt2, &[], vec![1, c.hidden]);
        let residual = b.multiply(&sum, &inv);
        let residual = round_bf16(b, &residual);
        let ff_input = rms_last_bf16(b, &residual, Some(&lw.pre_ff_norm), &k.eps, &k.zero);
        let gate = linear_seq_bf16(b, &ff_input, &lw.gate);
        let up = linear_seq_bf16(b, &ff_input, &lw.up);
        let mlp = if c.activation_sparsity[layer] > 0.0 {
            let activated_gate = sparse_gelu(b, &gate, c.activation_sparsity[layer], k);
            let product = b.multiply(&activated_gate, &up);
            round_bf16(b, &product)
        } else {
            geglu(b, &gate, &up)
        };
        let mlp = linear_seq_bf16(b, &mlp, &lw.down);
        let mlp = rms_last_bf16(b, &mlp, Some(&lw.post_ff_norm), &k.eps, &k.zero);
        let activated = b.add(&residual, &mlp);
        let activated = round_bf16(b, &activated);
        planes = altup_correct(b, &predicted, &activated, lw, c, k, 1);
        let mut corrected_active = planes[c.altup_active_idx].clone();
        if c.altup_correct_scale {
            let scale = b.broadcast(&lw.correct_scale, &[1], vec![1, c.hidden]);
            corrected_active = b.multiply(&corrected_active, &scale);
            corrected_active = round_bf16(b, &corrected_active);
        }
        let ple = b.slice(
            &dense_ple,
            &[(0, 1), (layer, layer + 1), (0, c.hidden_per_layer_input)],
        );
        let ple = b.reshape(&ple, vec![1, c.hidden_per_layer_input]);
        let gate = linear_seq_bf16(b, &corrected_active, &lw.ple_gate);
        let injected = geglu(b, &gate, &ple);
        let injected = linear_seq_bf16(b, &injected, &lw.ple_projection);
        let injected = rms_last_bf16(b, &injected, Some(&lw.ple_norm), &k.eps, &k.zero);
        for (plane, value) in planes.iter_mut().enumerate() {
            if plane != c.altup_active_idx {
                *value = b.add(value, &injected);
                *value = round_bf16(b, value);
            }
        }
    }
    let target = magnitude(b, &planes[c.altup_active_idx], c.hidden, &k.zero, &k.one);
    let mut collapsed_planes = vec![planes[c.altup_active_idx].clone()];
    for (index, projection) in weights.unembed_projections.iter().enumerate() {
        let projected = linear_seq_bf16(b, &planes[index + 1], projection);
        let projected = normalize_to(b, &projected, &target, k);
        collapsed_planes.push(projected);
    }
    assert_eq!(collapsed_planes.len(), c.altup_num_inputs);
    let collapsed = mean_planes_bf16(b, &collapsed_planes, &k.zero);
    let normalized = rms_last_bf16(b, &collapsed, Some(&weights.final_norm), &k.eps, &k.zero);
    let normalized = b.reshape(&normalized, vec![c.hidden]);
    let logits = linear_bf16(b, &normalized, &weights.embed);
    if let Some(cap) = c.final_logit_softcap {
        let cap = bf16_scalar(b, cap);
        let cap = b.broadcast(&cap, &[], vec![c.vocab]);
        let logits = b.divide(&logits, &cap);
        let logits = round_bf16(b, &logits);
        let logits = b.gemma3n_tanh(&logits);
        let logits = round_bf16(b, &logits);
        let logits = b.multiply(&logits, &cap);
        round_bf16(b, &logits)
    } else {
        logits
    }
}

fn scalar_cache_ty(c: &Gemma3nConfig) -> Ty {
    Ty::f32(vec![
        c.kv_cache_layers(),
        c.context_capacity,
        c.n_kv,
        c.head_dim,
    ])
}

fn render_decode(
    decls: &[Decl],
    b: &Builder,
    result: &Val,
    kcache: &Val,
    vcache: &Val,
    cache_ty: &Ty,
) -> String {
    let signature = decls
        .iter()
        .enumerate()
        .map(|(i, d)| format!("%arg{i}: {} loc(\"{}\")", d.ty.render(), d.loc))
        .collect::<Vec<_>>()
        .join(", ");
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
        "{target_alias}module @decode_step {{\n{executable_source}  \
         func.func public @main({signature}) -> \
         ({result_ty}, {cache_ty}, {cache_ty}) {{\n{body}    return {result}, {kc}, {vc} : \
         {result_ty}, {cache_ty}, {cache_ty}\n  }}\n}}\n",
        result_ty = result.ty.render(),
        cache_ty = cache_ty.render(),
        body = b.body(),
        result = result.name,
        kc = kcache.name,
        vc = vcache.name,
    )
}

fn magnitude(b: &mut Builder, value: &Val, hidden: usize, zero: &Val, one: &Val) -> Val {
    let squared = b.multiply(value, value);
    let squared = round_bf16(b, &squared);
    let sum = b.reduce_add(&squared, 1, zero);
    let sum = round_bf16(b, &sum);
    let width = b.const_f32(hidden as f32);
    let width = b.broadcast(&width, &[], vec![1]);
    let mean = b.divide(&sum, &width);
    let mean = round_bf16(b, &mean);
    let magnitude = b.sqrt(&mean);
    let _ = one;
    round_bf16(b, &magnitude)
}
