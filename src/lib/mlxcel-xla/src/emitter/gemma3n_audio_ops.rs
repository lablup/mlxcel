//! SSCP, feed-forward, and light-convolution audio graph operations.

use super::builder::{Builder, Val};
use super::gemma3n_audio_math::{
    clip, cumsum_time_f32, relu, rms_norm, round_bf16, sigmoid, silu, stride_time,
};
use super::gemma3n_audio_schema::EncoderArgs;
use crate::Gemma3nXlaAudioConfig;

pub(super) struct SscpOutput {
    pub conv0: Val,
    pub conv1: Val,
    pub hidden: Val,
    pub valid: Val,
}

pub(super) struct ConformerStages {
    pub feed_forward_start: Val,
    pub attention: Val,
    pub light_conv: Val,
    pub feed_forward_end: Val,
    pub final_norm: Val,
}

fn cumulative_group_norm(b: &mut Builder, x: &Val, weight: &Val, eps: f32) -> Val {
    let batch = x.ty.shape[0];
    let time = x.ty.shape[1];
    let frequency = x.ty.shape[2];
    let channels = x.ty.shape[3];
    let zero = b.const_f32(0.0);
    let sum_channels = b.reduce_add(x, 3, &zero);
    let sum_frequency = b.reduce_add(&sum_channels, 2, &zero);
    let sum_at_time = b.reshape(&sum_frequency, vec![batch, time, 1, 1]);
    let cumulative_sum = cumsum_time_f32(b, &sum_at_time);

    let index = b.iota(time);
    let index = b.convert(&index, "f32");
    let one = super::gemma3n_audio_math::scalar_like(b, 1.0, &index);
    let count = b.add(&index, &one);
    let width = super::gemma3n_audio_math::scalar_like(b, (frequency * channels) as f32, &count);
    let count = b.multiply(&count, &width);
    let count = b.reshape(&count, vec![1, time, 1, 1]);
    let count = b.broadcast(&count, &[0, 1, 2, 3], vec![batch, time, 1, 1]);
    let mean = b.divide(&cumulative_sum, &count);
    let mean = b.broadcast(&mean, &[0, 1, 2, 3], x.ty.shape.clone());
    let centered = b.subtract(x, &mean);

    let squared = b.multiply(&centered, &centered);
    let squared_channels = b.reduce_add(&squared, 3, &zero);
    let squared_frequency = b.reduce_add(&squared_channels, 2, &zero);
    let squared_at_time = b.reshape(&squared_frequency, vec![batch, time, 1, 1]);
    let cumulative_squared = cumsum_time_f32(b, &squared_at_time);
    let variance = b.divide(&cumulative_squared, &count);
    let epsilon = super::gemma3n_audio_math::scalar_like(b, eps, &variance);
    let variance = b.add(&variance, &epsilon);
    let inverse = b.rsqrt(&variance);
    let inverse = b.broadcast(&inverse, &[0, 1, 2, 3], x.ty.shape.clone());
    let normalized = b.multiply(&centered, &inverse);
    let weight = b.broadcast(weight, &[3], x.ty.shape.clone());
    let weighted = b.multiply(&normalized, &weight);
    round_bf16(b, &weighted)
}

#[allow(clippy::too_many_arguments)]
fn sscp_conv(
    b: &mut Builder,
    x: &Val,
    weight: &Val,
    norm: &Val,
    kernel: [usize; 2],
    stride: [usize; 2],
    groups: usize,
    eps: f32,
) -> Val {
    let zero = b.const_f32(0.0);
    let padded = b.pad(x, &zero, &[0, 0, 1, 0], &[0, kernel[0] - 1, 1, 0]);
    let weight = b.transpose(weight, &[1, 2, 3, 0]);
    // MLX stores both operands as BF16 but accumulates the convolution into an
    // F32 carrier before materializing the BF16 output tensor. The graph-wide
    // BF16 contraction mode otherwise gives StableHLO a BF16 result type,
    // allowing the local CPU backend to lose cancellation-sensitive lanes.
    let convolved =
        b.convolution_f32_accumulate(&padded, &weight, &stride, &[(0, 0), (0, 0)], groups);
    let convolved = round_bf16(b, &convolved);
    let normalized = cumulative_group_norm(b, &convolved, norm, eps);
    let activated = relu(b, &normalized);
    round_bf16(b, &activated)
}

fn subsample_convs(
    b: &mut Builder,
    args: &EncoderArgs,
    config: &Gemma3nXlaAudioConfig,
) -> (Val, Val) {
    let batch = args.mel.ty.shape[0];
    let time = args.mel.ty.shape[1];
    let frequency = args.mel.ty.shape[2];
    let x = b.reshape(&args.mel, vec![batch, time, frequency, 1]);
    // The maintained MLX path explicitly casts processor-produced F32 mel
    // features to `conv_0.weight.dtype` (BF16 in released checkpoints) before
    // the first SSCP convolution. Keep this boundary explicit instead of
    // relying on the graph-wide contraction precision: post-convolution
    // rounding cannot recover the products formed from BF16-rounded inputs.
    let x = round_bf16(b, &x);
    let root = "audio_tower.subsample_conv_projection";

    let conv0 = sscp_conv(
        b,
        &x,
        args.weight(&format!("{root}.conv_0.conv.weight")),
        args.weight(&format!("{root}.conv_0.norm.weight")),
        config.sscp_conv_kernel_size[0],
        config.sscp_conv_stride_size[0],
        1,
        config.sscp_conv_group_norm_eps,
    );
    let conv1 = sscp_conv(
        b,
        &conv0,
        args.weight(&format!("{root}.conv_1.conv.weight")),
        args.weight(&format!("{root}.conv_1.norm.weight")),
        config.sscp_conv_kernel_size[1],
        config.sscp_conv_stride_size[1],
        1,
        config.sscp_conv_group_norm_eps,
    );
    (conv0, conv1)
}

pub(super) fn subsample(
    b: &mut Builder,
    args: &EncoderArgs,
    config: &Gemma3nXlaAudioConfig,
) -> Val {
    let (_, conv1) = subsample_convs(b, args, config);
    let root = "audio_tower.subsample_conv_projection";
    let shape = &conv1.ty.shape;
    let flattened = b.reshape(&conv1, vec![shape[0], shape[1], shape[2] * shape[3]]);
    let projected = b.linear_last(
        &flattened,
        args.weight(&format!("{root}.input_proj_linear.weight")),
    );
    round_bf16(b, &projected)
}

pub(super) fn subsample_with_stages(
    b: &mut Builder,
    args: &EncoderArgs,
    config: &Gemma3nXlaAudioConfig,
) -> SscpOutput {
    let (conv0, conv1) = subsample_convs(b, args, config);
    let root = "audio_tower.subsample_conv_projection";
    let shape = &conv1.ty.shape;
    let flattened = b.reshape(&conv1, vec![shape[0], shape[1], shape[2] * shape[3]]);
    let projected = b.linear_last(
        &flattened,
        args.weight(&format!("{root}.input_proj_linear.weight")),
    );
    let hidden = round_bf16(b, &projected);
    let valid = stride_time(
        b,
        &args.valid_mask,
        config.time_stride_product().expect("validated stride"),
    );
    SscpOutput {
        conv0,
        conv1,
        hidden,
        valid,
    }
}

pub(super) fn feed_forward(
    b: &mut Builder,
    args: &EncoderArgs,
    config: &Gemma3nXlaAudioConfig,
    prefix: &str,
    input: &Val,
) -> Val {
    let clipped = clip(b, input, config.gradient_clipping);
    let normalized = rms_norm(
        b,
        &clipped,
        Some(args.weight(&format!("{prefix}.pre_layer_norm.weight"))),
        config.rms_norm_eps,
    );
    let normalized = round_bf16(b, &normalized);
    let first = b.linear_last(
        &normalized,
        args.weight(&format!("{prefix}.ffw_layer_1.weight")),
    );
    let first = round_bf16(b, &first);
    let first = silu(b, &first);
    let first = round_bf16(b, &first);
    let second = b.linear_last(&first, args.weight(&format!("{prefix}.ffw_layer_2.weight")));
    let second = round_bf16(b, &second);
    let second = clip(b, &second, config.gradient_clipping);
    let second = rms_norm(
        b,
        &second,
        Some(args.weight(&format!("{prefix}.post_layer_norm.weight"))),
        config.rms_norm_eps,
    );
    let second = round_bf16(b, &second);
    let residual_weight =
        super::gemma3n_audio_math::scalar_like(b, config.conf_residual_weight, &second);
    let update = b.multiply(&second, &residual_weight);
    let result = b.add(input, &update);
    round_bf16(b, &result)
}

pub(super) fn light_conv(
    b: &mut Builder,
    args: &EncoderArgs,
    config: &Gemma3nXlaAudioConfig,
    prefix: &str,
    input: &Val,
) -> Val {
    let hidden = config.hidden_size;
    let normalized = rms_norm(
        b,
        input,
        Some(args.weight(&format!("{prefix}.pre_layer_norm.weight"))),
        config.rms_norm_eps,
    );
    let normalized = round_bf16(b, &normalized);
    let projected = b.linear_last(
        &normalized,
        args.weight(&format!("{prefix}.linear_start.weight")),
    );
    let projected = round_bf16(b, &projected);
    let batch = projected.ty.shape[0];
    let time = projected.ty.shape[1];
    let left = b.slice(&projected, &[(0, batch), (0, time), (0, hidden)]);
    let right = b.slice(&projected, &[(0, batch), (0, time), (hidden, hidden * 2)]);
    let gate = sigmoid(b, &right);
    let gated = b.multiply(&left, &gate);
    let gated = round_bf16(b, &gated);
    let gated = b.reshape(&gated, vec![batch, time, 1, hidden]);
    let zero = b.const_f32(0.0);
    let padded = b.pad(
        &gated,
        &zero,
        &[0, config.conf_conv_kernel_size - 1, 0, 0],
        &[0, 0, 0, 0],
    );
    let weight = args.weight(&format!("{prefix}.depthwise_conv1d.weight"));
    let weight = b.transpose(weight, &[1, 2, 0]);
    let weight = b.reshape(&weight, vec![config.conf_conv_kernel_size, 1, 1, hidden]);
    let convolved = b.convolution(&padded, &weight, &[1, 1], &[(0, 0), (0, 0)], hidden);
    let convolved = round_bf16(b, &convolved);
    let convolved = b.reshape(&convolved, vec![batch, time, hidden]);
    let convolved = clip(b, &convolved, config.gradient_clipping);
    let convolved = rms_norm(
        b,
        &convolved,
        Some(args.weight(&format!("{prefix}.conv_norm.weight"))),
        config.rms_norm_eps,
    );
    let convolved = round_bf16(b, &convolved);
    let activated = silu(b, &convolved);
    let activated = round_bf16(b, &activated);
    let projected = b.linear_last(
        &activated,
        args.weight(&format!("{prefix}.linear_end.weight")),
    );
    let projected = round_bf16(b, &projected);
    let output = b.add(input, &projected);
    round_bf16(b, &output)
}
