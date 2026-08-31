use crate::layers::{RMSNorm, UnifiedLinear};
use crate::weights::WeightMap;
use crate::{MlxArray, UniquePtr, dtype};

use super::{InklingLayerCache, InklingLayerSpec, weight};

/// Causal depth-wise convolution shared by Inkling target and MTP layers.
pub struct InklingShortConv {
    weight: UniquePtr<MlxArray>,
    decode_weight: UniquePtr<MlxArray>,
    kernel: i32,
}

impl InklingShortConv {
    pub fn from_weights(weights: &WeightMap, name: &str, kernel: usize) -> Result<Self, String> {
        let raw = weight(weights, name)?;
        let shape = crate::array_shape(&raw);
        if shape.len() != 3 || shape[1] != kernel as i32 || shape[2] != 1 {
            return Err(format!(
                "{name}: expected [channels, {kernel}, 1], got {shape:?}"
            ));
        }
        let weight = crate::astype(&raw, dtype::FLOAT32);
        let decode_weight = crate::transpose_axes(&weight, &[2, 1, 0]);
        let decode_weight = crate::contiguous(&decode_weight, false);
        crate::eval(&decode_weight);
        Ok(Self {
            weight,
            decode_weight,
            kernel: kernel as i32,
        })
    }

    pub fn forward(
        &self,
        input: &MlxArray,
        state: &mut Option<UniquePtr<MlxArray>>,
        residual: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let input_dtype = crate::array_dtype(input);
        let input_f32 = crate::astype(input, dtype::FLOAT32);
        let shape = crate::array_shape(&input_f32);
        let keep = self.kernel - 1;
        let padded = match state.as_ref().and_then(|value| value.as_ref()) {
            Some(previous) => crate::concatenate(previous, &input_f32, 1),
            None => {
                let zeros = crate::zeros(&[shape[0], keep, shape[2]], dtype::FLOAT32);
                crate::concatenate(&zeros, &input_f32, 1)
            }
        };
        let padded_len = crate::array_shape(&padded)[1];
        let tail = crate::utils::slice_axis(&padded, 1, padded_len - keep, padded_len);
        *state = Some(crate::contiguous(&tail, false));

        let convolved = if shape[1] == 1 {
            let window = crate::utils::slice_axis(&padded, 1, padded_len - self.kernel, padded_len);
            let multiplied = crate::multiply(&window, &self.decode_weight);
            crate::sum_axis(&multiplied, 1, true)
        } else {
            crate::conv1d(&padded, &self.weight, 1, 0, 1, shape[2])
        };
        let inner = crate::add(&convolved, &input_f32);
        let inner = crate::astype(&inner, input_dtype);
        match residual {
            Some(residual) => crate::add(residual, &inner),
            None => inner,
        }
    }
}

/// Inkling NoPE attention with learned banded relative logits.
pub struct InklingAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    r_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    k_sconv: InklingShortConv,
    v_sconv: InklingShortConv,
    rel_proj: UniquePtr<MlxArray>,
    is_sliding: bool,
    n_heads: i32,
    n_kv: i32,
    head_dim: i32,
    d_rel: i32,
    rel_extent: i32,
    window: i32,
    log_floor: Option<f32>,
    log_alpha: f32,
}

impl InklingAttention {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        spec: &InklingLayerSpec,
    ) -> Result<Self, String> {
        let (n_heads, n_kv, head_dim) = spec.attention_heads();
        let linear = |name: &str| {
            UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.{name}"),
                spec.quantization_group_size,
                spec.quantization_bits,
            )
        };
        let rel_extent = spec.relative_extent();
        let rel_proj = weight(weights, &format!("{prefix}.rel_proj"))?;
        let rel_shape = crate::array_shape(&rel_proj);
        if rel_shape != [spec.d_rel as i32, rel_extent as i32] {
            return Err(format!(
                "{prefix}.rel_proj: expected [{}, {}], got {rel_shape:?}",
                spec.d_rel, rel_extent
            ));
        }
        Ok(Self {
            q_proj: linear("q_proj")?,
            k_proj: linear("k_proj")?,
            v_proj: linear("v_proj")?,
            r_proj: linear("r_proj")?,
            o_proj: linear("o_proj")?,
            q_norm: RMSNorm::new(
                weight(weights, &format!("{prefix}.q_norm.weight"))?,
                spec.rms_norm_eps,
            ),
            k_norm: RMSNorm::new(
                weight(weights, &format!("{prefix}.k_norm.weight"))?,
                spec.rms_norm_eps,
            ),
            k_sconv: InklingShortConv::from_weights(
                weights,
                &format!("{prefix}.k_sconv.conv.weight"),
                spec.sconv_kernel_size,
            )?,
            v_sconv: InklingShortConv::from_weights(
                weights,
                &format!("{prefix}.v_sconv.conv.weight"),
                spec.sconv_kernel_size,
            )?,
            rel_proj,
            is_sliding: spec.is_sliding,
            n_heads: n_heads as i32,
            n_kv: n_kv as i32,
            head_dim: head_dim as i32,
            d_rel: spec.d_rel as i32,
            rel_extent: rel_extent as i32,
            window: spec.sliding_window_size as i32,
            log_floor: (!spec.is_sliding)
                .then_some(spec.log_scaling_n_floor)
                .flatten()
                .map(|value| value as f32),
            log_alpha: spec.log_scaling_alpha,
        })
    }

    pub fn forward(&self, input: &MlxArray, cache: &mut InklingLayerCache) -> UniquePtr<MlxArray> {
        let shape = crate::array_shape(input);
        let (batch, length) = (shape[0], shape[1]);
        let q = self.q_proj.forward(input);
        let k = self.k_proj.forward(input);
        let v = self.v_proj.forward(input);
        let r = self.r_proj.forward(input);
        let k = self.k_sconv.forward(&k, &mut cache.conv[0], None);
        let v = self.v_sconv.forward(&v, &mut cache.conv[1], None);

        let q = crate::reshape(&q, &[batch, length, self.n_heads, self.head_dim]);
        let k = crate::reshape(&k, &[batch, length, self.n_kv, self.head_dim]);
        let v = crate::reshape(&v, &[batch, length, self.n_kv, self.head_dim]);
        let r = crate::reshape(&r, &[batch, length, self.n_heads, self.d_rel]);
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);
        let mut q = crate::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = crate::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = crate::transpose_axes(&v, &[0, 2, 1, 3]);
        let (mut keys, mut values) = cache.kv.update_and_fetch(k, v);
        let before = crate::array_shape(&keys)[2];
        if self.is_sliding && before > length + self.window - 1 {
            let excess = before - (length + self.window - 1);
            cache.kv.trim_front(excess);
            keys = crate::utils::slice_axis(&keys, 2, excess, before);
            values = crate::utils::slice_axis(&values, 2, excess, before);
        }
        let source = crate::array_shape(&keys)[2];
        let offset = source - length;
        let mut mask = banded_additive_mask(
            &r,
            &self.rel_proj,
            offset,
            source,
            self.is_sliding.then_some(self.window),
            self.rel_extent,
        );
        if let Some(floor) = self.log_floor {
            let tau = log_scaling_tau(length, offset, floor, self.log_alpha);
            let tau4 = crate::reshape(&tau, &[1, 1, length, 1]);
            let tau4 = crate::astype(&tau4, crate::array_dtype(&q));
            q = crate::multiply(&q, &tau4);
            let threshold = crate::full_like(&mask, -1e29);
            let valid = crate::greater(&mask, &threshold);
            let scaled = crate::multiply(&mask, &tau4);
            mask = crate::where_cond(&valid, &scaled, &mask);
        }
        let mask = crate::astype(&mask, crate::array_dtype(&q));
        let mask_ptr = mask
            .as_ref()
            .map_or(std::ptr::null(), |value| value as *const MlxArray);
        // SAFETY: `mask_ptr` points to `mask`, which remains alive for the
        // duration of this call; q/k/v are valid MLX arrays owned above.
        let out = unsafe {
            crate::scaled_dot_product_attention(
                &q,
                &keys,
                &values,
                1.0 / self.head_dim as f32,
                mask_ptr,
            )
        };
        let out = crate::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = crate::reshape(&out, &[batch, length, self.n_heads * self.head_dim]);
        self.o_proj.forward(&out)
    }
}

pub fn banded_additive_mask(
    r: &MlxArray,
    projection: &MlxArray,
    offset: i32,
    source: i32,
    sliding_window: Option<i32>,
    rel_extent: i32,
) -> UniquePtr<MlxArray> {
    let shape = crate::array_shape(r);
    let (batch, length, heads) = (shape[0], shape[1], shape[2]);
    let relative = crate::matmul(r, projection);
    let relative = crate::transpose_axes(&relative, &[0, 2, 1, 3]);
    let queries = crate::arange_i32(offset, offset + length, 1);
    let queries = crate::reshape(&queries, &[length, 1]);
    let keys = crate::arange_i32(0, source, 1);
    let keys = crate::reshape(&keys, &[1, source]);
    let dist = crate::subtract(&queries, &keys);
    let dist = crate::reshape(&dist, &[1, 1, length, source]);
    let dist = crate::broadcast_to(&dist, &[batch, heads, length, source]);
    let low = crate::from_slice_i32(&[0], &[1]);
    let high = crate::from_slice_i32(&[rel_extent - 1], &[1]);
    let gather = crate::clip(&dist, &low, &high);
    let gathered = crate::take_along_axis(&relative, &gather, -1);
    let extent = crate::from_slice_i32(&[rel_extent], &[1]);
    let past_extent = crate::greater_equal(&dist, &extent);
    let zeros = crate::zeros_like(&gathered);
    let positional = crate::where_cond(&past_extent, &zeros, &gathered);
    let zero = crate::from_slice_i32(&[0], &[1]);
    let future = crate::less(&dist, &zero);
    let invalid = if let Some(window) = sliding_window {
        let window = crate::from_slice_i32(&[window], &[1]);
        crate::logical_or(&future, &crate::greater_equal(&dist, &window))
    } else {
        future
    };
    let blocked = crate::full_like(&positional, -1e30);
    crate::where_cond(&invalid, &blocked, &positional)
}

pub fn log_scaling_tau(length: i32, offset: i32, floor: f32, alpha: f32) -> UniquePtr<MlxArray> {
    let positions = crate::arange_f32((offset + 1) as f32, (offset + length + 1) as f32, 1.0);
    let ratio = crate::divide_scalar(&positions, floor);
    let ratio = crate::maximum(&ratio, &crate::ones_like(&ratio));
    let scaled = crate::multiply_scalar(&crate::log(&ratio), alpha);
    crate::add(&crate::ones_like(&scaled), &scaled)
}
