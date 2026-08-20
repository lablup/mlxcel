//! Chunked local relative attention for Gemma3n audio.

use super::builder::{Builder, Val};
use super::gemma3n_audio_math::{clip, rms_norm, round_bf16, scalar_like, softmax_last, softplus};
use super::gemma3n_audio_schema::EncoderArgs;
use crate::Gemma3nXlaAudioConfig;

fn pad_time(b: &mut Builder, x: &Val, left: usize, right: usize, value: &Val) -> Val {
    let mut low = vec![0; x.ty.shape.len()];
    let mut high = vec![0; x.ty.shape.len()];
    low[1] = left;
    high[1] = right;
    b.pad(x, value, &low, &high)
}

fn query_blocks(b: &mut Builder, x: &Val, chunk: usize) -> Val {
    let batch = x.ty.shape[0];
    let time = x.ty.shape[1];
    let heads = x.ty.shape[2];
    let head_dim = x.ty.shape[3];
    let blocks = time.div_ceil(chunk);
    let zero = b.const_f32(0.0);
    let padded = pad_time(b, x, 0, blocks * chunk - time, &zero);
    b.reshape(&padded, vec![batch, blocks, chunk, heads, head_dim])
}

fn extract_context(b: &mut Builder, x: &Val, config: &Gemma3nXlaAudioConfig) -> Val {
    let time = x.ty.shape[1];
    let chunk = config.conf_attention_chunk_size;
    let left = config.conf_attention_context_left - 1;
    let right = config.conf_attention_context_right + chunk - 1;
    let context = config.context_size();
    let blocks = time.div_ceil(chunk);
    let padded = if x.ty.elt == "i1" {
        let zero = b.const_i1(false);
        pad_time(b, x, left, right, &zero)
    } else {
        let zero = b.const_f32(0.0);
        pad_time(b, x, left, right, &zero)
    };
    let mut time_first = vec![1, 0];
    time_first.extend(2..padded.ty.shape.len());
    let padded = b.transpose(&padded, &time_first);
    let trailing = padded.ty.shape[1..].iter().product::<usize>();
    let table = b.reshape(&padded, vec![padded.ty.shape[0], trailing]);
    let indices = (0..blocks)
        .flat_map(|block| {
            (0..context).map(move |offset| {
                i32::try_from(block * chunk + offset).expect("audio context index")
            })
        })
        .collect::<Vec<_>>();
    let indices = b.const_tensor_i32(&indices, vec![blocks * context]);
    let indices = b.reshape(&indices, vec![blocks * context, 1]);
    let gathered = b.gather(&table, &indices);
    let mut gathered_shape = vec![blocks, context];
    gathered_shape.extend_from_slice(&padded.ty.shape[1..]);
    let gathered = b.reshape(&gathered, gathered_shape);
    let mut batch_first = vec![2, 0, 1];
    batch_first.extend(3..gathered.ty.shape.len());
    b.transpose(&gathered, &batch_first)
}

fn causal_valid_mask(b: &mut Builder, config: &Gemma3nXlaAudioConfig) -> Val {
    let chunk = config.conf_attention_chunk_size;
    let context = config.context_size();
    let diagonal = config.conf_attention_context_left - 1 + config.conf_attention_context_right;
    let values = (0..chunk)
        .flat_map(|query| {
            (0..context).map(move |key| {
                if query <= key && key <= query + diagonal {
                    1.0
                } else {
                    0.0
                }
            })
        })
        .collect::<Vec<_>>();
    let values = b.const_tensor_f32(&values, vec![chunk, context]);
    let zero = scalar_like(b, 0.0, &values);
    b.compare("GT", &values, &zero, "FLOAT")
}

fn relative_signal(
    b: &mut Builder,
    args: &EncoderArgs,
    config: &Gemma3nXlaAudioConfig,
    prefix: &str,
) -> Val {
    let past = config.conf_attention_context_left - 1;
    let future = config.conf_attention_context_right;
    let positions = (0..=past + future)
        .rev()
        .map(|value| (value as i32 - future as i32) as f32)
        .collect::<Vec<_>>();
    let timescales = config.hidden_size / 2;
    let increment = 10_000.0f32.ln() / (timescales as f32 - 1.0).max(1.0);
    let inverse = (0..timescales)
        .map(|index| (-increment * index as f32).exp())
        .collect::<Vec<_>>();
    let span = positions.len();
    let positions = b.const_tensor_f32(&positions, vec![1, span, 1]);
    let inverse = b.const_tensor_f32(&inverse, vec![1, 1, timescales]);
    let positions = b.broadcast(&positions, &[0, 1, 2], vec![1, span, timescales]);
    let inverse = b.broadcast(&inverse, &[0, 1, 2], vec![1, span, timescales]);
    let scaled = b.multiply(&positions, &inverse);
    let sine = b.sine(&scaled);
    let cosine = b.cosine(&scaled);
    let signal = b.concatenate(&sine, &cosine, 2);
    let signal = b.linear_last(
        &signal,
        args.weight(&format!(
            "{prefix}.attn.relative_position_embedding.pos_proj.weight"
        )),
    );
    let signal = round_bf16(b, &signal);
    b.reshape(
        &signal,
        vec![span, config.conf_num_attention_heads, config.head_dim()],
    )
}

fn relative_logits(
    b: &mut Builder,
    queries: &Val,
    keys: &Val,
    args: &EncoderArgs,
    config: &Gemma3nXlaAudioConfig,
    prefix: &str,
) -> Val {
    let batch = queries.ty.shape[0];
    let blocks = queries.ty.shape[1];
    let chunk = queries.ty.shape[2];
    let heads = queries.ty.shape[3];
    let head_dim = queries.ty.shape[4];
    let context = keys.ty.shape[2];
    let span = config.conf_attention_context_left + config.conf_attention_context_right;
    let query = b.transpose(queries, &[0, 3, 1, 2, 4]);
    let key = b.transpose(keys, &[0, 3, 1, 4, 2]);
    let content = b.dot_general(
        &query,
        &key,
        &[0, 1, 2],
        &[0, 1, 2],
        &[4],
        &[3],
        vec![batch, heads, blocks, chunk, context],
    );
    let content = round_bf16(b, &content);

    let signal = relative_signal(b, args, config, prefix);
    let signal = b.transpose(&signal, &[1, 2, 0]);
    let signal = b.reshape(&signal, vec![1, heads, 1, head_dim, span]);
    let signal = b.broadcast(
        &signal,
        &[0, 1, 2, 3, 4],
        vec![batch, heads, blocks, head_dim, span],
    );
    let position = b.dot_general(
        &query,
        &signal,
        &[0, 1, 2],
        &[0, 1, 2],
        &[4],
        &[3],
        vec![batch, heads, blocks, chunk, span],
    );
    let position = round_bf16(b, &position);
    let zero = b.const_f32(0.0);
    let position = b.pad(
        &position,
        &zero,
        &[0, 0, 0, 0, 0],
        &[0, 0, 0, 0, context + 1 - span],
    );
    let position = b.reshape(&position, vec![batch, heads, blocks, chunk * (context + 1)]);
    let position = b.slice(
        &position,
        &[(0, batch), (0, heads), (0, blocks), (0, chunk * context)],
    );
    let position = b.reshape(&position, vec![batch, heads, blocks, chunk, context]);
    let logits = b.add(&content, &position);
    round_bf16(b, &logits)
}

pub(super) fn attention(
    b: &mut Builder,
    args: &EncoderArgs,
    config: &Gemma3nXlaAudioConfig,
    prefix: &str,
    input: &Val,
    valid: &Val,
) -> Val {
    let batch = input.ty.shape[0];
    let time = input.ty.shape[1];
    let heads = config.conf_num_attention_heads;
    let head_dim = config.head_dim();
    let clipped = clip(b, input, config.gradient_clipping);
    let normalized = rms_norm(
        b,
        &clipped,
        Some(args.weight(&format!("{prefix}.pre_attn_norm.weight"))),
        config.rms_norm_eps,
    );
    let normalized = round_bf16(b, &normalized);
    let project = |b: &mut Builder, name: &str| {
        let projected = b.linear_last(
            &normalized,
            args.weight(&format!("{prefix}.attn.{name}.weight")),
        );
        let projected = round_bf16(b, &projected);
        b.reshape(&projected, vec![batch, time, heads, head_dim])
    };
    let query = project(b, "q_proj");
    let key = project(b, "k_proj");
    let value = project(b, "v_proj");
    let scale = softplus(b, args.weight(&format!("{prefix}.attn.per_dim_scale")));
    let scale = round_bf16(b, &scale);
    let q_scale = (head_dim as f32).powf(-0.5) / 2.0f32.ln();
    let q_scale = scalar_like(b, q_scale, &scale);
    let scale = b.multiply(&scale, &q_scale);
    let scale = round_bf16(b, &scale);
    let scale = b.broadcast(&scale, &[3], query.ty.shape.clone());
    let query = b.multiply(&query, &scale);
    let query = round_bf16(b, &query);

    let query = query_blocks(b, &query, config.conf_attention_chunk_size);
    let keys = extract_context(b, &key, config);
    let values = extract_context(b, &value, config);
    let blocks = query.ty.shape[1];
    let logits = relative_logits(b, &query, &keys, args, config, prefix);
    let cap = scalar_like(b, config.conf_attention_logit_cap, &logits);
    let scaled = b.divide(&logits, &cap);
    let capped = b.tanh(&scaled);
    let capped = b.multiply(&capped, &cap);
    let capped = round_bf16(b, &capped);

    let context_valid = extract_context(b, valid, config);
    let context_valid = b.reshape(
        &context_valid,
        vec![batch, 1, blocks, 1, config.context_size()],
    );
    let context_valid = b.broadcast(
        &context_valid,
        &[0, 1, 2, 3, 4],
        vec![
            batch,
            heads,
            blocks,
            config.conf_attention_chunk_size,
            config.context_size(),
        ],
    );
    let causal = causal_valid_mask(b, config);
    let causal = b.reshape(
        &causal,
        vec![
            1,
            1,
            1,
            config.conf_attention_chunk_size,
            config.context_size(),
        ],
    );
    let causal = b.broadcast(&causal, &[0, 1, 2, 3, 4], capped.ty.shape.clone());
    let condition = b.logical_and(&context_valid, &causal);
    let minimum = scalar_like(b, -3.38e38, &capped);
    let masked = b.select(&condition, &capped, &minimum);
    let probabilities = softmax_last(b, &masked);
    let probabilities = round_bf16(b, &probabilities);
    let values = b.transpose(&values, &[0, 3, 1, 2, 4]);
    let context = b.dot_general(
        &probabilities,
        &values,
        &[0, 1, 2],
        &[0, 1, 2],
        &[4],
        &[3],
        vec![
            batch,
            heads,
            blocks,
            config.conf_attention_chunk_size,
            head_dim,
        ],
    );
    let context = round_bf16(b, &context);
    let context = b.transpose(&context, &[0, 2, 3, 1, 4]);
    let context = b.reshape(
        &context,
        vec![
            batch,
            blocks * config.conf_attention_chunk_size,
            config.hidden_size,
        ],
    );
    let context = if context.ty.shape[1] > time {
        b.slice(&context, &[(0, batch), (0, time), (0, config.hidden_size)])
    } else {
        context
    };
    let post = b.linear_last(&context, args.weight(&format!("{prefix}.post.weight")));
    let post = round_bf16(b, &post);
    let post = clip(b, &post, config.gradient_clipping);
    let post = rms_norm(
        b,
        &post,
        Some(args.weight(&format!("{prefix}.post_norm.weight"))),
        config.rms_norm_eps,
    );
    let post = round_bf16(b, &post);
    let output = b.add(input, &post);
    round_bf16(b, &output)
}
