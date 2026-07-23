//! StableHLO building blocks shared by Gemma3n token and embeddings+PLE prefill.

use super::builder::{Builder, Val};
use super::gemma3n::Gemma3nConfig;
use super::rope::{plain_inv_freq_with_base, rope_tables_from_inv};

pub(super) struct Constants {
    pub zero: Val,
    pub one: Val,
    pub eps: Val,
    pub neg_inf: Val,
    pub neg_big: Val,
    pub hidden: Val,
    pub inv_sqrt2: Val,
    pub cos_global: Val,
    pub sin_global: Val,
    pub cos_local: Val,
    pub sin_local: Val,
}

pub(super) struct LayerWeights {
    pub correct_scale: Val,
    pub correction: Val,
    pub router: Val,
    pub router_norm: Val,
    pub prediction: Val,
    pub laurel_left: Val,
    pub laurel_right: Val,
    pub laurel_norm: Val,
    pub input_norm: Val,
    pub post_attn_norm: Val,
    pub pre_ff_norm: Val,
    pub post_ff_norm: Val,
    pub wq: Val,
    pub wk: Option<Val>,
    pub wv: Option<Val>,
    pub wo: Val,
    pub q_norm: Val,
    pub k_norm: Option<Val>,
    pub gate: Val,
    pub up: Val,
    pub down: Val,
    pub ple_gate: Val,
    pub ple_projection: Val,
    pub ple_norm: Val,
}

pub(super) fn constants(b: &mut Builder, c: &Gemma3nConfig) -> Constants {
    let (cg, sg) = rope_tables_from_inv(
        &plain_inv_freq_with_base(c.head_dim, c.rope_theta),
        c.head_dim,
        c.context_capacity,
        false,
    );
    let (cl, sl) = rope_tables_from_inv(
        &plain_inv_freq_with_base(c.head_dim, c.rope_local_base),
        c.head_dim,
        c.context_capacity,
        false,
    );
    Constants {
        zero: b.const_f32(0.0),
        one: b.const_f32(1.0),
        eps: b.const_f32(c.eps),
        neg_inf: b.const_f32(f32::NEG_INFINITY),
        neg_big: b.const_f32(-1e30),
        hidden: b.const_f32(c.hidden as f32),
        inv_sqrt2: b.const_f32(std::f32::consts::FRAC_1_SQRT_2),
        cos_global: b.const_tensor_f32(&cg, vec![c.context_capacity, c.head_dim]),
        sin_global: b.const_tensor_f32(&sg, vec![c.context_capacity, c.head_dim]),
        cos_local: b.const_tensor_f32(&cl, vec![c.context_capacity, c.head_dim]),
        sin_local: b.const_tensor_f32(&sl, vec![c.context_capacity, c.head_dim]),
    }
}

pub(super) fn rms_last(
    b: &mut Builder,
    x: &Val,
    weight: Option<&Val>,
    eps: &Val,
    zero: &Val,
) -> Val {
    let axis = x.ty.shape.len() - 1;
    let width = x.ty.shape[axis];
    let reduced_shape: Vec<usize> =
        x.ty.shape
            .iter()
            .enumerate()
            .filter_map(|(i, d)| (i != axis).then_some(*d))
            .collect();
    let width_c = b.const_f32(width as f32);
    let width_b = b.broadcast(&width_c, &[], reduced_shape.clone());
    let sq = b.multiply(x, x);
    let sum = b.reduce_add(&sq, axis, zero);
    let mean = b.divide(&sum, &width_b);
    let eps_b = b.broadcast(eps, &[], reduced_shape);
    let mean = b.add(&mean, &eps_b);
    let inv = b.rsqrt(&mean);
    let keep: Vec<usize> = (0..axis).collect();
    let inv_b = b.broadcast(&inv, &keep, x.ty.shape.clone());
    let normalized = b.multiply(x, &inv_b);
    match weight {
        Some(weight) => {
            let wb = b.broadcast(weight, &[axis], x.ty.shape.clone());
            b.multiply(&normalized, &wb)
        }
        None => normalized,
    }
}

pub(super) fn normalize_to(b: &mut Builder, plane: &Val, target: &Val, k: &Constants) -> Val {
    let axis = plane.ty.shape.len() - 1;
    let width = plane.ty.shape[axis];
    let width_c = b.const_f32(width as f32);
    let reduced: Vec<usize> = plane.ty.shape[..axis].to_vec();
    let width_b = b.broadcast(&width_c, &[], reduced.clone());
    let sq = b.multiply(plane, plane);
    let sum = b.reduce_add(&sq, axis, &k.zero);
    let mean = b.divide(&sum, &width_b);
    let inverse = b.rsqrt(&mean);
    let one = b.broadcast(&k.one, &[], reduced.clone());
    let magnitude = b.divide(&one, &inverse);
    let eps_b = b.broadcast(&k.eps, &[], reduced.clone());
    let pred = b.compare("GT", &magnitude, &eps_b, "FLOAT");
    let safe = b.select(&pred, &magnitude, &eps_b);
    let scale = b.divide(target, &safe);
    let keep: Vec<usize> = (0..axis).collect();
    let scale_b = b.broadcast(&scale, &keep, plane.ty.shape.clone());
    b.multiply(plane, &scale_b)
}

pub(super) fn gelu(b: &mut Builder, x: &Val) -> Val {
    let shape = x.ty.shape.clone();
    let splat = |b: &mut Builder, value: f32, shape: &[usize]| {
        let scalar = b.const_f32(value);
        b.broadcast(&scalar, &[], shape.to_vec())
    };
    let root = splat(b, (2.0f32 / std::f32::consts::PI).sqrt(), &shape);
    let cubic = splat(b, 0.044_715, &shape);
    let half = splat(b, 0.5, &shape);
    let one = splat(b, 1.0, &shape);
    let x2 = b.multiply(x, x);
    let x3 = b.multiply(&x2, x);
    let cubic_x = b.multiply(&cubic, &x3);
    let inner = b.add(x, &cubic_x);
    let inner = b.multiply(&root, &inner);
    let tanh = b.tanh(&inner);
    let half_x = b.multiply(&half, x);
    let one_tanh = b.add(&one, &tanh);
    b.multiply(&half_x, &one_tanh)
}

pub(super) fn geglu(b: &mut Builder, gate: &Val, up: &Val) -> Val {
    let gate = gelu(b, gate);
    b.multiply(&gate, up)
}

pub(super) fn sparse_gelu(b: &mut Builder, x: &Val, sparsity: f32, k: &Constants) -> Val {
    let rows = x.ty.shape[0];
    let width = x.ty.shape[1];
    let width_scalar = b.const_f32(width as f32);
    let width_b = b.broadcast(&width_scalar, &[], vec![rows]);
    let sum = b.reduce_add(x, 1, &k.zero);
    let mean = b.divide(&sum, &width_b);
    let mean_b = b.broadcast(&mean, &[0], vec![rows, width]);
    let centered = b.subtract(x, &mean_b);
    let squared = b.multiply(&centered, &centered);
    let variance = b.reduce_add(&squared, 1, &k.zero);
    let variance = b.divide(&variance, &width_b);
    let inverse = b.rsqrt(&variance);
    let one = b.broadcast(&k.one, &[], vec![rows]);
    let stddev = b.divide(&one, &inverse);
    let multiplier = std::f32::consts::SQRT_2 * erfinv(2.0 * sparsity - 1.0);
    let multiplier = b.const_f32(multiplier);
    let multiplier = b.broadcast(&multiplier, &[], vec![rows]);
    let spread = b.multiply(&stddev, &multiplier);
    let cutoff = b.add(&mean, &spread);
    let cutoff = b.broadcast(&cutoff, &[0], vec![rows, width]);
    let shifted = b.subtract(x, &cutoff);
    let zeros = b.broadcast(&k.zero, &[], vec![rows, width]);
    let positive = b.compare("GT", &shifted, &zeros, "FLOAT");
    let shifted = b.select(&positive, &shifted, &zeros);
    let sqrt2 = b.const_f32(std::f32::consts::SQRT_2);
    let sqrt2 = b.broadcast(&sqrt2, &[], vec![rows, width]);
    let scaled = b.divide(&shifted, &sqrt2);
    let erf = b.erf(&scaled);
    let half = b.const_f32(0.5);
    let half = b.broadcast(&half, &[], vec![rows, width]);
    let one = b.broadcast(&k.one, &[], vec![rows, width]);
    let one_erf = b.add(&one, &erf);
    let scale = b.multiply(&half, &one_erf);
    b.multiply(&shifted, &scale)
}

fn erfinv(x: f32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    let a = 0.147;
    let ln = (1.0 - x * x).ln();
    let first = 2.0 / (std::f32::consts::PI * a) + ln / 2.0;
    let second = ln / a;
    x.signum() * ((first * first - second).sqrt() - first).sqrt()
}

pub(super) fn softmax(b: &mut Builder, scores: &Val, axis: usize, k: &Constants) -> Val {
    let shape = scores.ty.shape.clone();
    let keep: Vec<usize> = (0..shape.len()).filter(|&i| i != axis).collect();
    let max = b.reduce_max(scores, axis, &k.neg_inf);
    let max_b = b.broadcast(&max, &keep, shape.clone());
    let shifted = b.subtract(scores, &max_b);
    let exp = b.exponential(&shifted);
    let sum = b.reduce_add(&exp, axis, &k.zero);
    let sum_b = b.broadcast(&sum, &keep, shape);
    b.divide(&exp, &sum_b)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn rope(
    b: &mut Builder,
    x: &Val,
    positions: &Val,
    cosine: &Val,
    sine: &Val,
    rows: usize,
    heads: usize,
    dim: usize,
) -> Val {
    let indices = b.reshape(positions, vec![rows, 1]);
    let cos = b.gather(cosine, &indices);
    let sin = b.gather(sine, &indices);
    let cos = b.broadcast(&cos, &[0, 2], vec![rows, heads, dim]);
    let sin = b.broadcast(&sin, &[0, 2], vec![rows, heads, dim]);
    let half = dim / 2;
    let left = b.slice(x, &[(0, rows), (0, heads), (0, half)]);
    let right = b.slice(x, &[(0, rows), (0, heads), (half, dim)]);
    let right = b.negate(&right);
    let rotated = b.concatenate(&right, &left, 2);
    let cosine = b.multiply(x, &cos);
    let sine = b.multiply(&rotated, &sin);
    b.add(&cosine, &sine)
}

pub(super) fn causal_mask(
    b: &mut Builder,
    capacity: usize,
    window: Option<usize>,
    k: &Constants,
) -> Val {
    let row_iota = b.iota(capacity);
    let rows = b.broadcast(&row_iota, &[0], vec![capacity, capacity]);
    let col_iota = b.iota(capacity);
    let cols = b.broadcast(&col_iota, &[1], vec![capacity, capacity]);
    let causal = b.compare("LE", &cols, &rows, "SIGNED");
    let zeros = b.broadcast(&k.zero, &[], vec![capacity, capacity]);
    let masked = b.broadcast(&k.neg_big, &[], vec![capacity, capacity]);
    let base = b.select(&causal, &zeros, &masked);
    match window {
        None => base,
        Some(width) => {
            let age = b.subtract(&rows, &cols);
            let w = b.const_i32(width as i32);
            let wb = b.broadcast(&w, &[], vec![capacity, capacity]);
            let within = b.compare("LT", &age, &wb, "SIGNED");
            b.select(&within, &base, &masked)
        }
    }
}

pub(super) fn apply_sliding_window(
    b: &mut Builder,
    base: &Val,
    capacity: usize,
    width: usize,
    k: &Constants,
) -> Val {
    let row_iota = b.iota(capacity);
    let rows = b.broadcast(&row_iota, &[0], vec![capacity, capacity]);
    let col_iota = b.iota(capacity);
    let cols = b.broadcast(&col_iota, &[1], vec![capacity, capacity]);
    let age = b.subtract(&rows, &cols);
    let width = b.const_i32(width as i32);
    let width = b.broadcast(&width, &[], vec![capacity, capacity]);
    let within = b.compare("LT", &age, &width, "SIGNED");
    let masked = b.broadcast(&k.neg_big, &[], vec![capacity, capacity]);
    b.select(&within, base, &masked)
}

pub(super) fn altup_predict(
    b: &mut Builder,
    planes: &[Val],
    lw: &LayerWeights,
    c: &Gemma3nConfig,
    k: &Constants,
    rows: usize,
) -> Vec<Val> {
    let n = c.altup_num_inputs;
    let active = &planes[c.altup_active_idx];
    let routed = rms_last(b, active, Some(&lw.router_norm), &k.eps, &k.zero);
    let routed = b.linear_seq(&routed, &lw.router);
    let hidden_b = b.broadcast(&k.hidden, &[], vec![rows, n]);
    let modalities = b.divide(&routed, &hidden_b);
    let modalities = b.tanh(&modalities);
    let prediction = clipped(b, &lw.prediction, c.altup_coef_clip);
    let coefficients = b.linear_seq(&modalities, &prediction);
    let coefficients = b.reshape(&coefficients, vec![rows, n, n]);
    let coefficients = b.transpose(&coefficients, &[0, 2, 1]);
    let stacked = stack_planes(b, planes, rows, c.hidden);
    let by_feature = b.transpose(&stacked, &[1, 2, 0]);
    let delta = b.dot_general(
        &by_feature,
        &coefficients,
        &[0],
        &[0],
        &[2],
        &[1],
        vec![rows, c.hidden, n],
    );
    let delta = b.transpose(&delta, &[2, 0, 1]);
    let predicted = b.add(&stacked, &delta);
    split_planes(b, &predicted, n, rows, c.hidden)
}

pub(super) fn altup_correct(
    b: &mut Builder,
    predicted: &[Val],
    activated: &Val,
    lw: &LayerWeights,
    c: &Gemma3nConfig,
    k: &Constants,
    rows: usize,
) -> Vec<Val> {
    let n = c.altup_num_inputs;
    let active_prediction = &predicted[c.altup_active_idx];
    let routed = rms_last(b, activated, Some(&lw.router_norm), &k.eps, &k.zero);
    let routed = b.linear_seq(&routed, &lw.router);
    let hidden_b = b.broadcast(&k.hidden, &[], vec![rows, n]);
    let modalities = b.divide(&routed, &hidden_b);
    let modalities = b.tanh(&modalities);
    let correction = clipped(b, &lw.correction, c.altup_coef_clip);
    let coefficients = b.linear_seq(&modalities, &correction);
    let one_b = b.broadcast(&k.one, &[], vec![rows, n]);
    let coefficients = b.add(&coefficients, &one_b);
    let innovation = b.subtract(activated, active_prediction);
    let mut corrected = Vec::with_capacity(n);
    for (plane, predicted_plane) in predicted.iter().enumerate() {
        let coefficient = b.slice(&coefficients, &[(0, rows), (plane, plane + 1)]);
        let coefficient = b.broadcast(&coefficient, &[0, 1], vec![rows, c.hidden]);
        let delta = b.multiply(&innovation, &coefficient);
        corrected.push(b.add(predicted_plane, &delta));
    }
    corrected
}

fn clipped(b: &mut Builder, value: &Val, limit: f32) -> Val {
    let positive = b.const_f32(limit);
    let positive = b.broadcast(&positive, &[], value.ty.shape.clone());
    let negative = b.const_f32(-limit);
    let negative = b.broadcast(&negative, &[], value.ty.shape.clone());
    let above = b.compare("GT", value, &positive, "FLOAT");
    let value = b.select(&above, &positive, value);
    let below = b.compare("LT", &value, &negative, "FLOAT");
    b.select(&below, &negative, &value)
}

fn stack_planes(b: &mut Builder, planes: &[Val], rows: usize, hidden: usize) -> Val {
    let mut stacked = b.reshape(&planes[0], vec![1, rows, hidden]);
    for plane in &planes[1..] {
        let plane = b.reshape(plane, vec![1, rows, hidden]);
        stacked = b.concatenate(&stacked, &plane, 0);
    }
    stacked
}

fn split_planes(
    b: &mut Builder,
    stacked: &Val,
    count: usize,
    rows: usize,
    hidden: usize,
) -> Vec<Val> {
    (0..count)
        .map(|plane| {
            let value = b.slice(stacked, &[(plane, plane + 1), (0, rows), (0, hidden)]);
            b.reshape(&value, vec![rows, hidden])
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attention(
    b: &mut Builder,
    normalized: &Val,
    positions: &Val,
    mask: &Val,
    lw: &LayerWeights,
    layer: usize,
    cache_index: usize,
    c: &Gemma3nConfig,
    k: &Constants,
    kcache: &mut Val,
    vcache: &mut Val,
) -> Val {
    let rows = c.context_capacity;
    let d = c.head_dim;
    let group = c.n_q / c.n_kv;
    let q = b.linear_seq(normalized, &lw.wq);
    let q = b.reshape(&q, vec![rows, c.n_q, d]);
    let q = rms_last(b, &q, Some(&lw.q_norm), &k.eps, &k.zero);
    let (cos, sin) = match c.layer_types[layer] {
        super::gemma3n::Gemma3nLayerType::Full => (&k.cos_global, &k.sin_global),
        super::gemma3n::Gemma3nLayerType::Sliding => (&k.cos_local, &k.sin_local),
    };
    let q = rope(b, &q, positions, cos, sin, rows, c.n_q, d);
    let (keys, values) = match (&lw.wk, &lw.wv, &lw.k_norm) {
        (Some(wk), Some(wv), Some(k_norm)) => {
            let keys = b.linear_seq(normalized, wk);
            let keys = b.reshape(&keys, vec![rows, c.n_kv, d]);
            let keys = rms_last(b, &keys, Some(k_norm), &k.eps, &k.zero);
            let keys = rope(b, &keys, positions, cos, sin, rows, c.n_kv, d);
            let values = b.linear_seq(normalized, wv);
            let values = b.reshape(&values, vec![rows, c.n_kv, d]);
            let values = rms_last(b, &values, None, &k.eps, &k.zero);
            let ci = b.const_i32(cache_index as i32);
            let c0 = b.const_i32(0);
            let key_update = b.reshape(&keys, vec![1, rows, c.n_kv, d]);
            *kcache = b.dynamic_update_slice(kcache, &key_update, &[&ci, &c0, &c0, &c0]);
            let value_update = b.reshape(&values, vec![1, rows, c.n_kv, d]);
            *vcache = b.dynamic_update_slice(vcache, &value_update, &[&ci, &c0, &c0, &c0]);
            (keys, values)
        }
        (None, None, None) => {
            let keys = b.slice(
                kcache,
                &[
                    (cache_index, cache_index + 1),
                    (0, rows),
                    (0, c.n_kv),
                    (0, d),
                ],
            );
            let values = b.slice(
                vcache,
                &[
                    (cache_index, cache_index + 1),
                    (0, rows),
                    (0, c.n_kv),
                    (0, d),
                ],
            );
            (
                b.reshape(&keys, vec![rows, c.n_kv, d]),
                b.reshape(&values, vec![rows, c.n_kv, d]),
            )
        }
        _ => unreachable!("Gemma3n K/V projection and norm arguments are atomic"),
    };
    let q = b.reshape(&q, vec![rows, c.n_kv, group, d]);
    let scores = b.dot_general(
        &q,
        &keys,
        &[1],
        &[1],
        &[3],
        &[2],
        vec![c.n_kv, rows, group, rows],
    );
    // Gemma3n deliberately uses attention scale 1.0.
    let mask = b.broadcast(mask, &[1, 3], vec![c.n_kv, rows, group, rows]);
    let scores = b.add(&scores, &mask);
    let probabilities = softmax(b, &scores, 3, k);
    let context = b.dot_general(
        &probabilities,
        &values,
        &[0],
        &[1],
        &[3],
        &[0],
        vec![c.n_kv, rows, group, d],
    );
    let context = b.transpose(&context, &[1, 0, 2, 3]);
    let context = b.reshape(&context, vec![rows, c.n_q * d]);
    b.linear_seq(&context, &lw.wo)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attention_decode(
    b: &mut Builder,
    normalized: &Val,
    position: &Val,
    lw: &LayerWeights,
    layer: usize,
    cache_index: usize,
    c: &Gemma3nConfig,
    k: &Constants,
    kcache: &mut Val,
    vcache: &mut Val,
) -> Val {
    let d = c.head_dim;
    let group = c.n_q / c.n_kv;
    let position_row = b.reshape(position, vec![1]);
    let q = b.linear_seq(normalized, &lw.wq);
    let q = b.reshape(&q, vec![1, c.n_q, d]);
    let q = rms_last(b, &q, Some(&lw.q_norm), &k.eps, &k.zero);
    let (cos, sin, window) = match c.layer_types[layer] {
        super::gemma3n::Gemma3nLayerType::Full => (&k.cos_global, &k.sin_global, None),
        super::gemma3n::Gemma3nLayerType::Sliding => {
            (&k.cos_local, &k.sin_local, Some(c.sliding_window))
        }
    };
    let q = rope(b, &q, &position_row, cos, sin, 1, c.n_q, d);
    if let (Some(wk), Some(wv), Some(k_norm)) = (&lw.wk, &lw.wv, &lw.k_norm) {
        let keys = b.linear_seq(normalized, wk);
        let keys = b.reshape(&keys, vec![1, c.n_kv, d]);
        let keys = rms_last(b, &keys, Some(k_norm), &k.eps, &k.zero);
        let keys = rope(b, &keys, &position_row, cos, sin, 1, c.n_kv, d);
        let values = b.linear_seq(normalized, wv);
        let values = b.reshape(&values, vec![1, c.n_kv, d]);
        let values = rms_last(b, &values, None, &k.eps, &k.zero);
        let cache = b.const_i32(cache_index as i32);
        let zero = b.const_i32(0);
        let keys = b.reshape(&keys, vec![1, 1, c.n_kv, d]);
        *kcache = b.dynamic_update_slice(kcache, &keys, &[&cache, position, &zero, &zero]);
        let values = b.reshape(&values, vec![1, 1, c.n_kv, d]);
        *vcache = b.dynamic_update_slice(vcache, &values, &[&cache, position, &zero, &zero]);
    }
    let keys = b.slice(
        kcache,
        &[
            (cache_index, cache_index + 1),
            (0, c.context_capacity),
            (0, c.n_kv),
            (0, d),
        ],
    );
    let keys = b.reshape(&keys, vec![c.context_capacity, c.n_kv, d]);
    let values = b.slice(
        vcache,
        &[
            (cache_index, cache_index + 1),
            (0, c.context_capacity),
            (0, c.n_kv),
            (0, d),
        ],
    );
    let values = b.reshape(&values, vec![c.context_capacity, c.n_kv, d]);
    let q = b.reshape(&q, vec![1, c.n_kv, group, d]);
    let scores = b.dot_general(
        &q,
        &keys,
        &[1],
        &[1],
        &[3],
        &[2],
        vec![c.n_kv, 1, group, c.context_capacity],
    );
    let indices = b.iota(c.context_capacity);
    let position_b = b.broadcast(position, &[], vec![c.context_capacity]);
    let visible = b.compare("LE", &indices, &position_b, "SIGNED");
    let visible = if let Some(width) = window {
        let first = b.const_i32(width.saturating_sub(1) as i32);
        let first = b.subtract(position, &first);
        let first = b.broadcast(&first, &[], vec![c.context_capacity]);
        let local = b.compare("GE", &indices, &first, "SIGNED");
        b.select(&visible, &local, &visible)
    } else {
        visible
    };
    let zeros = b.broadcast(&k.zero, &[], vec![c.context_capacity]);
    let masked = b.broadcast(&k.neg_big, &[], vec![c.context_capacity]);
    let mask = b.select(&visible, &zeros, &masked);
    let mask = b.broadcast(&mask, &[3], vec![c.n_kv, 1, group, c.context_capacity]);
    let scores = b.add(&scores, &mask);
    let probabilities = softmax(b, &scores, 3, k);
    let context = b.dot_general(
        &probabilities,
        &values,
        &[0],
        &[1],
        &[3],
        &[0],
        vec![c.n_kv, 1, group, d],
    );
    let context = b.transpose(&context, &[1, 0, 2, 3]);
    let context = b.reshape(&context, vec![1, c.n_q * d]);
    b.linear_seq(&context, &lw.wo)
}
