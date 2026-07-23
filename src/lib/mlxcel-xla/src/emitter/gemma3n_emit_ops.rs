//! StableHLO building blocks shared by Gemma3n token and embeddings+PLE prefill.

use super::builder::{Builder, Val};
use super::gemma3n::Gemma3nConfig;
use super::rope::{plain_inv_freq_with_base, rope_tables_from_inv};
use crate::weights::round_bf16_f32;

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

/// Keep Gemma3n's native BF16 activation stream explicit while carrying values
/// through the public ABI as f32. This is intentionally local to Gemma3n:
/// dense-family precision policy must not change when reproducing MLX's
/// operation-by-operation BF16 rounding.
pub(super) fn round_bf16(b: &mut Builder, value: &Val) -> Val {
    let narrowed = b.convert(value, "bf16");
    b.convert(&narrowed, "f32")
}

/// Materialize a scalar that MLX constructs directly in a BF16 input's dtype.
///
/// Author the already-rounded host value as f32 instead of relying on a
/// constant f32→bf16→f32 convert pair: compiler constant folding may erase the
/// observable BF16 rounding even though the same pair remains correct for
/// dynamic values.
pub(super) fn bf16_scalar(b: &mut Builder, value: f32) -> Val {
    b.const_f32(round_bf16_f32(value))
}

/// Gemma3n quantized projections consume BF16 operands and return a BF16
/// activation. StableHLO's dot accumulates into f32; the explicit round pins the
/// observable MLX output dtype before the next operation.
pub(super) fn linear_seq_bf16(b: &mut Builder, input: &Val, weight: &Val) -> Val {
    if b.gemma3n_qmv_enabled() {
        return b.gemma3n_qmv(input, weight);
    }
    let input = b.convert(input, "bf16");
    let weight = b.convert(weight, "bf16");
    let output = b.linear_seq(&input, &weight);
    round_bf16(b, &output)
}

pub(super) fn linear_bf16(b: &mut Builder, input: &Val, weight: &Val) -> Val {
    if b.gemma3n_qmv_enabled() {
        return b.gemma3n_qmv(input, weight);
    }
    let input = b.convert(input, "bf16");
    let weight = b.convert(weight, "bf16");
    let output = b.linear(&input, &weight);
    round_bf16(b, &output)
}

/// Match MLX's `mean(stack(planes), axis=0)` schedule: reduce all BF16-carried
/// planes in f32, round the completed sum once, then multiply by a BF16
/// reciprocal and round the result. Rounding each pairwise add changes the
/// Gemma3n four-plane collapse.
pub(super) fn mean_planes_bf16(b: &mut Builder, planes: &[Val], zero: &Val) -> Val {
    assert!(!planes.is_empty(), "Gemma3n AltUp collapse needs planes");
    let plane_shape = planes[0].ty.shape.clone();
    let mut stacked_shape = vec![1];
    stacked_shape.extend_from_slice(&plane_shape);
    let mut stacked = b.reshape(&planes[0], stacked_shape.clone());
    for plane in &planes[1..] {
        assert_eq!(
            plane.ty.shape, plane_shape,
            "Gemma3n AltUp collapse plane shape mismatch"
        );
        let plane = b.reshape(plane, stacked_shape.clone());
        stacked = b.concatenate(&stacked, &plane, 0);
    }
    let sum = b.reduce_add(&stacked, 0, zero);
    let sum = round_bf16(b, &sum);
    let reciprocal = bf16_scalar(b, (planes.len() as f32).recip());
    let reciprocal = b.broadcast(&reciprocal, &[], plane_shape);
    let mean = b.multiply(&sum, &reciprocal);
    round_bf16(b, &mean)
}

pub(super) fn constants(b: &mut Builder, c: &Gemma3nConfig) -> Constants {
    // A shared-KV layer observes the mapped concrete cache after it has
    // advanced by the current prefill length, so its query positions span up
    // to `2 * context_capacity - 1`.
    let rope_capacity = c.context_capacity * 2;
    let (cg, sg) = rope_tables_from_inv(
        &plain_inv_freq_with_base(c.head_dim, c.rope_theta),
        c.head_dim,
        rope_capacity,
        false,
    );
    let (cl, sl) = rope_tables_from_inv(
        &plain_inv_freq_with_base(c.head_dim, c.rope_local_base),
        c.head_dim,
        rope_capacity,
        false,
    );
    let inv_sqrt2 = bf16_scalar(b, std::f32::consts::FRAC_1_SQRT_2);
    Constants {
        zero: b.const_f32(0.0),
        one: b.const_f32(1.0),
        eps: b.const_f32(c.eps),
        neg_inf: b.const_f32(f32::NEG_INFINITY),
        neg_big: b.const_f32(-1e30),
        hidden: b.const_f32(c.hidden as f32),
        inv_sqrt2,
        cos_global: b.const_tensor_f32(&cg, vec![rope_capacity, c.head_dim]),
        sin_global: b.const_tensor_f32(&sg, vec![rope_capacity, c.head_dim]),
        cos_local: b.const_tensor_f32(&cl, vec![rope_capacity, c.head_dim]),
        sin_local: b.const_tensor_f32(&sl, vec![rope_capacity, c.head_dim]),
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

pub(super) fn rms_last_bf16(
    b: &mut Builder,
    x: &Val,
    weight: Option<&Val>,
    eps: &Val,
    zero: &Val,
) -> Val {
    // MLX's CUDA RMSNorm kernel stores `x * inv_rms` through a `T` temporary
    // before applying the optional weight. For Gemma3n, `T` is BF16: preserve
    // that intermediate rounding boundary instead of fusing both multiplies in
    // f32 and rounding only the final result.
    let normalized = rms_last(b, x, None, eps, zero);
    let normalized = round_bf16(b, &normalized);
    match weight {
        Some(weight) => {
            let axis = x.ty.shape.len() - 1;
            let weight = b.broadcast(weight, &[axis], x.ty.shape.clone());
            let weighted = b.multiply(&normalized, &weight);
            round_bf16(b, &weighted)
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
    let sq = round_bf16(b, &sq);
    let sum = b.reduce_add(&sq, axis, &k.zero);
    let sum = round_bf16(b, &sum);
    let mean = b.divide(&sum, &width_b);
    let mean = round_bf16(b, &mean);
    let magnitude = b.sqrt(&mean);
    let magnitude = round_bf16(b, &magnitude);
    let eps_b = b.broadcast(&k.eps, &[], reduced.clone());
    let pred = b.compare("GT", &magnitude, &eps_b, "FLOAT");
    let safe = b.select(&pred, &magnitude, &eps_b);
    let safe = round_bf16(b, &safe);
    let scale = b.divide(target, &safe);
    let scale = round_bf16(b, &scale);
    let keep: Vec<usize> = (0..axis).collect();
    let scale_b = b.broadcast(&scale, &keep, plane.ty.shape.clone());
    let normalized = b.multiply(plane, &scale_b);
    round_bf16(b, &normalized)
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
    let tanh = b.gemma3n_tanh(&inner);
    let half_x = b.multiply(&half, x);
    let one_tanh = b.add(&one, &tanh);
    b.multiply(&half_x, &one_tanh)
}

pub(super) fn geglu(b: &mut Builder, gate: &Val, up: &Val) -> Val {
    if b.gemma3n_qmv_enabled() {
        return b.gemma3n_geglu_bf16(gate, up);
    }
    let gate = gelu(b, gate);
    let product = b.multiply(&gate, up);
    round_bf16(b, &product)
}

pub(super) struct SparseGeluStages {
    pub mean: Val,
    pub variance: Val,
    pub stddev: Val,
    pub cutoff: Val,
    pub shifted_raw: Val,
    pub shifted: Val,
    pub erf: Val,
    pub activated: Val,
}

/// Match the materialized BF16 carrier tape used by MLX's sparse GELU.
///
/// Reductions accumulate in f32, but every observable elementwise result is
/// stored through BF16 before it feeds the next operation.
pub(super) fn sparse_gelu_stages(
    b: &mut Builder,
    x: &Val,
    sparsity: f32,
    zero: &Val,
    one: &Val,
) -> SparseGeluStages {
    let rows = x.ty.shape[0];
    let width = x.ty.shape[1];
    let width_scalar = b.const_f32(width as f32);
    let width_b = b.broadcast(&width_scalar, &[], vec![rows]);
    let sum = b.reduce_add(x, 1, zero);
    let mean = b.divide(&sum, &width_b);
    let mean = round_bf16(b, &mean);
    let mean_b = b.broadcast(&mean, &[0], vec![rows, width]);
    let centered = b.subtract(x, &mean_b);
    let centered = round_bf16(b, &centered);
    let squared = b.multiply(&centered, &centered);
    let squared = round_bf16(b, &squared);
    let variance = b.reduce_add(&squared, 1, zero);
    let variance = b.divide(&variance, &width_b);
    let variance = round_bf16(b, &variance);
    let stddev = b.sqrt(&variance);
    let stddev = round_bf16(b, &stddev);
    let multiplier = std::f32::consts::SQRT_2 * erfinv(2.0 * sparsity - 1.0);
    let multiplier = b.const_f32(multiplier);
    let multiplier = b.broadcast(&multiplier, &[], vec![rows]);
    let spread = b.multiply(&stddev, &multiplier);
    let spread = round_bf16(b, &spread);
    let cutoff = b.add(&mean, &spread);
    let cutoff = round_bf16(b, &cutoff);
    let cutoff_b = b.broadcast(&cutoff, &[0], vec![rows, width]);
    let shifted_raw = b.subtract(x, &cutoff_b);
    let shifted_raw = round_bf16(b, &shifted_raw);
    let zeros = b.broadcast(zero, &[], vec![rows, width]);
    let positive = b.compare("GT", &shifted_raw, &zeros, "FLOAT");
    let shifted = b.select(&positive, &shifted_raw, &zeros);
    let shifted = round_bf16(b, &shifted);
    let sqrt2 = b.const_f32(std::f32::consts::SQRT_2);
    let sqrt2 = b.broadcast(&sqrt2, &[], vec![rows, width]);
    let scaled = b.divide(&shifted, &sqrt2);
    let scaled = round_bf16(b, &scaled);
    let erf = b.erf(&scaled);
    let erf = round_bf16(b, &erf);
    let half = b.const_f32(0.5);
    let half = b.broadcast(&half, &[], vec![rows, width]);
    let one = b.broadcast(one, &[], vec![rows, width]);
    let one_erf = b.add(&one, &erf);
    let one_erf = round_bf16(b, &one_erf);
    let scale = b.multiply(&half, &one_erf);
    let scale = round_bf16(b, &scale);
    let activated = b.multiply(&shifted, &scale);
    let activated = round_bf16(b, &activated);
    SparseGeluStages {
        mean,
        variance,
        stddev,
        cutoff,
        shifted_raw,
        shifted,
        erf,
        activated,
    }
}

pub(super) fn sparse_gelu(b: &mut Builder, x: &Val, sparsity: f32, k: &Constants) -> Val {
    sparse_gelu_stages(b, x, sparsity, &k.zero, &k.one).activated
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
    stable_softmax(b, scores, axis, &k.zero, &k.neg_inf)
}

fn stable_softmax(b: &mut Builder, scores: &Val, axis: usize, zero: &Val, neg_inf: &Val) -> Val {
    let shape = scores.ty.shape.clone();
    let keep: Vec<usize> = (0..shape.len()).filter(|&i| i != axis).collect();
    let max = b.reduce_max(scores, axis, neg_inf);
    let max_b = b.broadcast(&max, &keep, shape.clone());
    let shifted = b.subtract(scores, &max_b);
    let exp = b.exponential(&shifted);
    let sum = b.reduce_add(&exp, axis, zero);
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
    let rotated = b.add(&cosine, &sine);
    round_bf16(b, &rotated)
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
    let routed = rms_last_bf16(b, active, Some(&lw.router_norm), &k.eps, &k.zero);
    let hidden_b = b.broadcast(&k.hidden, &[], vec![rows, c.hidden]);
    let routed = b.divide(&routed, &hidden_b);
    let routed = round_bf16(b, &routed);
    let modalities = linear_seq_bf16(b, &routed, &lw.router);
    // MLX explicitly casts the quantized router output to f32 before tanh.
    let modalities = b.gemma3n_tanh(&modalities);
    let prediction = clipped(b, &lw.prediction, c.altup_coef_clip);
    let coefficients = b.gemma3n_altup_coeff(&modalities, &prediction);
    let coefficients = b.reshape(&coefficients, vec![rows, n, n]);
    let coefficients = b.transpose(&coefficients, &[0, 2, 1]);
    let stacked = stack_planes(b, planes, rows, c.hidden);
    let predicted = b.gemma3n_altup_predict(&stacked, &coefficients);
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
    let routed = rms_last_bf16(b, activated, Some(&lw.router_norm), &k.eps, &k.zero);
    let hidden_b = b.broadcast(&k.hidden, &[], vec![rows, c.hidden]);
    let routed = b.divide(&routed, &hidden_b);
    let routed = round_bf16(b, &routed);
    let modalities = linear_seq_bf16(b, &routed, &lw.router);
    // From here through coefficient projection, stay in the intentional f32
    // island. The correction/prediction matrices are stored as f32.
    let modalities = b.gemma3n_tanh(&modalities);
    let correction = clipped(b, &lw.correction, c.altup_coef_clip);
    let coefficients = b.gemma3n_altup_coeff(&modalities, &correction);
    let one_b = b.broadcast(&k.one, &[], vec![rows, n]);
    let coefficients = b.add(&coefficients, &one_b);
    let innovation = b.subtract(activated, active_prediction);
    let innovation = round_bf16(b, &innovation);
    let mut corrected = Vec::with_capacity(n);
    for (plane, predicted_plane) in predicted.iter().enumerate() {
        let coefficient = b.slice(&coefficients, &[(0, rows), (plane, plane + 1)]);
        let coefficient = b.broadcast(&coefficient, &[0, 1], vec![rows, c.hidden]);
        let (_, corrected_plane) =
            altup_correct_plane(b, predicted_plane, &innovation, &coefficient);
        corrected.push(corrected_plane);
    }
    corrected
}

pub(super) fn altup_correct_plane(
    b: &mut Builder,
    predicted: &Val,
    innovation: &Val,
    coefficient: &Val,
) -> (Val, Val) {
    let correction = b.multiply(innovation, coefficient);
    let correction = round_bf16(b, &correction);
    let corrected = b.add(predicted, &correction);
    let corrected = round_bf16(b, &corrected);
    (correction, corrected)
}

fn clipped(b: &mut Builder, value: &Val, limit: Option<f32>) -> Val {
    let Some(limit) = limit else {
        return value.clone();
    };
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

/// Match MLX CUDA's materialized explicit-mask attention carrier schedule.
///
/// Q/K/V arrive as logical BF16 values in f32 ABI tensors. The QK result,
/// additive mask, masked scores, and precise-softmax result are each stored as
/// BF16 before the next operation.
#[allow(clippy::too_many_arguments)]
fn materialized_attention_probabilities(
    b: &mut Builder,
    q: &Val,
    keys: &Val,
    mask: &Val,
    zero: &Val,
    neg_inf: &Val,
    rows: usize,
    kv_heads: usize,
    group: usize,
) -> Val {
    let scores = b.dot_general(
        q,
        keys,
        &[1],
        &[1],
        &[3],
        &[2],
        vec![kv_heads, rows, group, rows],
    );
    let scores = round_bf16(b, &scores);
    let mask = round_bf16(b, mask);
    let mask = b.broadcast(&mask, &[1, 3], vec![kv_heads, rows, group, rows]);
    let scores = b.add(&scores, &mask);
    let scores = round_bf16(b, &scores);
    let probabilities = stable_softmax(b, &scores, 3, zero, neg_inf);
    round_bf16(b, &probabilities)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attention(
    b: &mut Builder,
    normalized: &Val,
    positions: &Val,
    real_len: &Val,
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
    let q = linear_seq_bf16(b, normalized, &lw.wq);
    let q = b.reshape(&q, vec![rows, c.n_q, d]);
    let q = rms_last_bf16(b, &q, Some(&lw.q_norm), &k.eps, &k.zero);
    let (cos, sin) = match c.layer_types[layer] {
        super::gemma3n::Gemma3nLayerType::Full => (&k.cos_global, &k.sin_global),
        super::gemma3n::Gemma3nLayerType::Sliding => (&k.cos_local, &k.sin_local),
    };
    // The maintained MLX Gemma3n implementation reuses the concrete layer's
    // cache object for a shared-KV layer. That cache has already advanced by
    // the prefill length, so the shared layer's query RoPE starts at that
    // offset even though its K/V come from the earlier logical layer.
    let query_positions = if lw.wk.is_none() {
        let offset = b.broadcast(real_len, &[], vec![rows]);
        b.add(positions, &offset)
    } else {
        positions.clone()
    };
    let q = rope(b, &q, &query_positions, cos, sin, rows, c.n_q, d);
    let (keys, values) = match (&lw.wk, &lw.wv, &lw.k_norm) {
        (Some(wk), Some(wv), Some(k_norm)) => {
            let keys = linear_seq_bf16(b, normalized, wk);
            let keys = b.reshape(&keys, vec![rows, c.n_kv, d]);
            let keys = rms_last_bf16(b, &keys, Some(k_norm), &k.eps, &k.zero);
            let keys = rope(b, &keys, positions, cos, sin, rows, c.n_kv, d);
            let values = linear_seq_bf16(b, normalized, wv);
            let values = b.reshape(&values, vec![rows, c.n_kv, d]);
            let values = rms_last_bf16(b, &values, None, &k.eps, &k.zero);
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
    // Gemma3n deliberately uses attention scale 1.0.
    let probabilities = materialized_attention_probabilities(
        b, &q, &keys, mask, &k.zero, &k.neg_inf, rows, c.n_kv, group,
    );
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
    let context = round_bf16(b, &context);
    linear_seq_bf16(b, &context, &lw.wo)
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
    attention_decode_traced(
        b,
        normalized,
        position,
        lw,
        layer,
        cache_index,
        c,
        k,
        kcache,
        vcache,
    )
    .0
}

pub(super) struct DecodeAttentionTrace {
    pub q_projection: Val,
    pub q_norm: Val,
    pub q_rope: Val,
    pub k_projection: Option<Val>,
    pub k_norm: Option<Val>,
    pub k_rope: Option<Val>,
    pub v_projection: Option<Val>,
    pub v_norm: Option<Val>,
    pub keys: Val,
    pub values: Val,
    pub context: Val,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attention_decode_traced(
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
) -> (Val, DecodeAttentionTrace) {
    let d = c.head_dim;
    let group = c.n_q / c.n_kv;
    let position_row = b.reshape(position, vec![1]);
    let q_projection = linear_seq_bf16(b, normalized, &lw.wq);
    let q = b.reshape(&q_projection, vec![1, c.n_q, d]);
    let q_norm = rms_last_bf16(b, &q, Some(&lw.q_norm), &k.eps, &k.zero);
    let (cos, sin, window) = match c.layer_types[layer] {
        super::gemma3n::Gemma3nLayerType::Full => (&k.cos_global, &k.sin_global, None),
        super::gemma3n::Gemma3nLayerType::Sliding => {
            (&k.cos_local, &k.sin_local, Some(c.sliding_window))
        }
    };
    let query_position = if lw.wk.is_none() {
        let one = b.const_i32(1);
        b.add(position, &one)
    } else {
        position.clone()
    };
    let query_position = b.reshape(&query_position, vec![1]);
    let q_rope = rope(b, &q_norm, &query_position, cos, sin, 1, c.n_q, d);
    let projected_kv =
        if let (Some(wk), Some(wv), Some(k_norm_weight)) = (&lw.wk, &lw.wv, &lw.k_norm) {
            let k_projection = linear_seq_bf16(b, normalized, wk);
            let keys = b.reshape(&k_projection, vec![1, c.n_kv, d]);
            let k_norm = rms_last_bf16(b, &keys, Some(k_norm_weight), &k.eps, &k.zero);
            let k_rope = rope(b, &k_norm, &position_row, cos, sin, 1, c.n_kv, d);
            let v_projection = linear_seq_bf16(b, normalized, wv);
            let values = b.reshape(&v_projection, vec![1, c.n_kv, d]);
            let v_norm = rms_last_bf16(b, &values, None, &k.eps, &k.zero);
            let cache = b.const_i32(cache_index as i32);
            let zero = b.const_i32(0);
            let keys = b.reshape(&k_rope, vec![1, 1, c.n_kv, d]);
            *kcache = b.dynamic_update_slice(kcache, &keys, &[&cache, position, &zero, &zero]);
            let values = b.reshape(&v_norm, vec![1, 1, c.n_kv, d]);
            *vcache = b.dynamic_update_slice(vcache, &values, &[&cache, position, &zero, &zero]);
            Some((k_projection, k_norm, k_rope, v_projection, v_norm))
        } else {
            None
        };
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
    let context = if native_decode_sdpa_supported(b, c, window) {
        let q = b.reshape(&q_rope, vec![c.n_q, d]);
        let context = b.gemma3n_sdpa_vector(&q, &keys, &values, position, window, 1.0);
        b.reshape(&context, vec![1, c.n_q * d])
    } else {
        let q = b.reshape(&q_rope, vec![1, c.n_kv, group, d]);
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
        b.reshape(&context, vec![1, c.n_q * d])
    };
    let context = round_bf16(b, &context);
    let projected = linear_seq_bf16(b, &context, &lw.wo);
    let (k_projection, k_norm, k_rope, v_projection, v_norm) = projected_kv
        .map_or((None, None, None, None, None), |(kp, kn, kr, vp, vn)| {
            (Some(kp), Some(kn), Some(kr), Some(vp), Some(vn))
        });
    (
        projected,
        DecodeAttentionTrace {
            q_projection,
            q_norm,
            q_rope,
            k_projection,
            k_norm,
            k_rope,
            v_projection,
            v_norm,
            keys,
            values,
            context,
        },
    )
}

#[cfg(xla_iree_cuda)]
fn native_decode_sdpa_supported(b: &Builder, c: &Gemma3nConfig, window: Option<usize>) -> bool {
    b.gemma3n_qmv_enabled()
        && c.head_dim == 256
        && (1..=1024).contains(&c.context_capacity)
        && c.n_q > 0
        && c.n_kv > 0
        && c.n_q.is_multiple_of(c.n_kv)
        && window.is_none_or(|width| width >= c.context_capacity)
}

#[cfg(not(xla_iree_cuda))]
fn native_decode_sdpa_supported(_b: &Builder, _c: &Gemma3nConfig, _window: Option<usize>) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emitter::builder::Ty;

    #[test]
    fn bf16_scalar_is_host_rounded_without_a_convert_pair() {
        let value = (2048.0f32).sqrt();
        assert_eq!(round_bf16_f32(value), 45.25);

        let mut b = Builder::new();
        let scalar = bf16_scalar(&mut b, value);
        let mut expected = Builder::new();
        expected.const_f32(45.25);
        assert_eq!(scalar.ty.elt, "f32");
        assert_eq!(b.body(), expected.body());
        assert!(!b.body().contains("stablehlo.convert"));
    }

    #[test]
    fn bf16_round_is_explicit_in_stablehlo() {
        let mut b = Builder::new();
        let value = Builder::arg(0, Ty::f32(vec![2, 8]));
        let rounded = round_bf16(&mut b, &value);
        assert_eq!(rounded.ty.elt, "f32");
        assert_eq!(b.body().matches("stablehlo.convert").count(), 2);
        assert!(
            b.body()
                .contains("stablehlo.convert %arg0 : (tensor<2x8xf32>) -> tensor<2x8xbf16>")
        );
        assert!(b.body().contains("-> tensor<2x8xf32>"));
    }

    #[test]
    fn bf16_projection_has_f32_island_boundary() {
        let mut b = Builder::new();
        let input = Builder::arg(0, Ty::f32(vec![2, 8]));
        let weight = Builder::arg(1, Ty::f32(vec![4, 8]));
        let output = linear_seq_bf16(&mut b, &input, &weight);
        assert_eq!(output.ty.elt, "f32");
        assert_eq!(b.body().matches("stablehlo.convert").count(), 4);
        assert!(
            b.body()
                .contains("(tensor<2x8xbf16>, tensor<4x8xbf16>) -> tensor<2x4xf32>")
        );
        assert!(b.body().contains("-> tensor<2x4xbf16>"));
        assert!(b.body().contains("-> tensor<2x4xf32>"));
    }

    #[test]
    fn sparse_gelu_materializes_every_mlx_bf16_carrier() {
        let mut b = Builder::new();
        let input = Builder::arg(0, Ty::f32(vec![2, 8]));
        let zero = b.const_f32(0.0);
        let one = b.const_f32(1.0);
        let stages = sparse_gelu_stages(&mut b, &input, 0.5, &zero, &one);

        assert_eq!(stages.activated.ty.elt, "f32");
        assert_eq!(stages.activated.ty.shape, vec![2, 8]);
        let body = b.body();
        assert_eq!(
            body.matches("stablehlo.convert").count(),
            28,
            "14 dynamic MLX intermediates must each round through bf16"
        );
        assert_eq!(body.matches("stablehlo.sqrt").count(), 1);
        assert!(!body.contains("stablehlo.rsqrt"));
    }

    #[test]
    fn altup_correction_rounds_product_before_residual_add() {
        let mut b = Builder::new();
        let predicted = Builder::arg(0, Ty::f32(vec![2, 8]));
        let innovation = Builder::arg(1, Ty::f32(vec![2, 8]));
        let coefficient = Builder::arg(2, Ty::f32(vec![2, 8]));
        let (correction, corrected) =
            altup_correct_plane(&mut b, &predicted, &innovation, &coefficient);

        assert_eq!(correction.ty.elt, "f32");
        assert_eq!(correction.ty.shape, vec![2, 8]);
        assert_eq!(corrected.ty.elt, "f32");
        assert_eq!(corrected.ty.shape, vec![2, 8]);
        let body = b.body();
        assert_eq!(body.matches("stablehlo.multiply").count(), 1);
        assert_eq!(body.matches("stablehlo.add").count(), 1);
        assert_eq!(
            body.matches("stablehlo.convert").count(),
            4,
            "product and corrected value must independently round through bf16"
        );
        let multiply = body.find("stablehlo.multiply").unwrap();
        let first_round = body.find("stablehlo.convert").unwrap();
        let add = body.find("stablehlo.add").unwrap();
        assert!(multiply < first_round && first_round < add);
    }

    #[test]
    fn materialized_attention_pins_bf16_score_mask_and_probability_carriers() {
        let mut b = Builder::new();
        let q = Builder::arg(0, Ty::f32(vec![3, 1, 2, 4]));
        let keys = Builder::arg(1, Ty::f32(vec![3, 1, 4]));
        let mask = Builder::arg(2, Ty::f32(vec![3, 3]));
        let zero = b.const_f32(0.0);
        let neg_inf = b.const_f32(f32::NEG_INFINITY);
        let probabilities = materialized_attention_probabilities(
            &mut b, &q, &keys, &mask, &zero, &neg_inf, 3, 1, 2,
        );

        assert_eq!(probabilities.ty.elt, "f32");
        assert_eq!(probabilities.ty.shape, vec![1, 3, 2, 3]);
        let body = b.body();
        assert_eq!(body.matches("stablehlo.dot_general").count(), 1);
        assert_eq!(
            body.matches("stablehlo.convert").count(),
            8,
            "QK, mask, masked score, and softmax output each need bf16->f32 carriers"
        );
        let exponential = body.find("stablehlo.exponential").unwrap();
        assert_eq!(
            body[..exponential].matches("stablehlo.convert").count(),
            6,
            "score and mask carriers must be rounded before precise softmax"
        );
        assert_eq!(
            body[exponential..].matches("stablehlo.convert").count(),
            2,
            "precise softmax must store its result through BF16"
        );
    }

    #[test]
    fn bf16_rms_norm_rounds_before_and_after_weight() {
        let mut weighted = Builder::new();
        let x = Builder::arg(0, Ty::f32(vec![2, 8]));
        let weight = Builder::arg(1, Ty::f32(vec![8]));
        let eps = Builder::arg(2, Ty::scalar("f32"));
        let zero = Builder::arg(3, Ty::scalar("f32"));
        let result = rms_last_bf16(&mut weighted, &x, Some(&weight), &eps, &zero);
        assert_eq!(result.ty.elt, "f32");
        assert_eq!(weighted.body().matches("stablehlo.convert").count(), 4);
        assert_eq!(weighted.body().matches("stablehlo.multiply").count(), 3);

        let mut unweighted = Builder::new();
        let result = rms_last_bf16(&mut unweighted, &x, None, &eps, &zero);
        assert_eq!(result.ty.elt, "f32");
        assert_eq!(unweighted.body().matches("stablehlo.convert").count(), 2);
        assert_eq!(unweighted.body().matches("stablehlo.multiply").count(), 2);
    }

    #[test]
    fn absent_altup_clip_leaves_coefficients_unchanged() {
        let mut b = Builder::new();
        let value = Builder::arg(0, Ty::f32(vec![4, 4]));
        let unchanged = clipped(&mut b, &value, None);
        assert_eq!(unchanged.name, value.name);
        assert!(b.body().is_empty());

        let clipped = clipped(&mut b, &value, Some(0.0));
        assert_ne!(clipped.name, value.name);
        assert_eq!(b.body().matches("stablehlo.compare").count(), 2);
    }
}
