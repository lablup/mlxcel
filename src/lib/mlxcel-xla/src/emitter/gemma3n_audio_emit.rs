//! Split Gemma3n `audio.encode` and `audio.merge_ple` StableHLO emitters.

use super::builder::{Builder, Precision, Val};
use super::gemma3n::Gemma3nConfig;
use super::gemma3n_audio_attention::attention;
use super::gemma3n_audio_math::{
    rms_norm, round_bf16, scalar_like, stride_time, zero_invalid, zeros_like,
};
use super::gemma3n_audio_ops::{feed_forward, light_conv, subsample_with_stages};
use super::gemma3n_audio_schema::{
    Decl, EncoderArgs, MergeArgs, build_encoder_schema, build_merge_schema,
};
use super::gemma3n_emit_ops::{bf16_scalar, dense_ple_input_head};
use crate::{GEMMA3N_AUDIO_FRAME_BUCKETS, GEMMA3N_AUDIO_SOFT_TOKENS, Gemma3nXlaAudioConfig};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gemma3nAudioDiagnosticStage {
    pub name: String,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gemma3nAudioDiagnosticLayout {
    pub stages: Vec<Gemma3nAudioDiagnosticStage>,
}

struct Trace {
    stages: Vec<(String, Val)>,
}

impl Trace {
    fn new() -> Self {
        Self { stages: Vec::new() }
    }

    fn push(&mut self, name: impl Into<String>, value: &Val) {
        self.stages.push((name.into(), value.clone()));
    }

    fn layout(&self) -> Gemma3nAudioDiagnosticLayout {
        Gemma3nAudioDiagnosticLayout {
            stages: self
                .stages
                .iter()
                .map(|(name, value)| Gemma3nAudioDiagnosticStage {
                    name: name.clone(),
                    shape: value.ty.shape.clone(),
                })
                .collect(),
        }
    }
}

struct EncodeOutputs {
    projected_audio: Val,
    hard_audio: Val,
    projected_lengths: Val,
    #[cfg(feature = "diagnostics")]
    diagnostics: Vec<(String, Val)>,
}

struct MergeOutputs {
    embeddings: Val,
    dense_ple: Val,
}

fn final_conformer_norm(
    b: &mut Builder,
    args: &EncoderArgs,
    audio: &Gemma3nXlaAudioConfig,
    prefix: &str,
    input: &Val,
) -> Val {
    let clipped = super::gemma3n_audio_math::clip(b, input, audio.gradient_clipping);
    let normalized = rms_norm(
        b,
        &clipped,
        Some(args.weight(&format!("{prefix}.norm.weight"))),
        audio.rms_norm_eps,
    );
    round_bf16(b, &normalized)
}

fn audio_encoder(
    b: &mut Builder,
    args: &EncoderArgs,
    audio: &Gemma3nXlaAudioConfig,
    trace: &mut Trace,
) -> (Val, Val, Val) {
    let subsampled = subsample_with_stages(b, args, audio);
    trace.push("sscp_conv_0_convolution", &subsampled.conv0_convolution);
    trace.push(
        "sscp_conv_0_norm_sum_at_time",
        &subsampled.conv0_norm_sum_at_time,
    );
    trace.push(
        "sscp_conv_0_norm_cumulative_sum",
        &subsampled.conv0_norm_cumulative_sum,
    );
    trace.push("sscp_conv_0_norm_mean", &subsampled.conv0_norm_mean);
    trace.push(
        "sscp_conv_0_norm_squared_at_time",
        &subsampled.conv0_norm_squared_at_time,
    );
    trace.push(
        "sscp_conv_0_norm_cumulative_squared",
        &subsampled.conv0_norm_cumulative_squared,
    );
    trace.push("sscp_conv_0_norm_variance", &subsampled.conv0_norm_variance);
    trace.push(
        "sscp_conv_0_norm_stabilized_variance",
        &subsampled.conv0_norm_stabilized_variance,
    );
    trace.push(
        "sscp_conv_0_norm_inverse_stddev",
        &subsampled.conv0_norm_inverse_stddev,
    );
    trace.push(
        "sscp_conv_0_norm_inverse_stddev_sqrt_reciprocal",
        &subsampled.conv0_norm_inverse_stddev_sqrt_reciprocal,
    );
    trace.push("sscp_conv_0_norm", &subsampled.conv0_norm);
    trace.push("sscp_conv_0", &subsampled.conv0);
    trace.push("sscp_conv_1_convolution", &subsampled.conv1_convolution);
    #[cfg(any(feature = "diagnostics", test))]
    trace.push(
        "sscp_conv_1_convolution_bf16_result",
        &subsampled.conv1_convolution_bf16_result,
    );
    trace.push("sscp_conv_1_norm", &subsampled.conv1_norm);
    trace.push("sscp_conv_1", &subsampled.conv1);
    trace.push("input_projection", &subsampled.hidden);
    let mut hidden = subsampled.hidden;
    let mut valid = subsampled.valid;
    for layer in 0..audio.conf_num_hidden_layers {
        let prefix = format!("audio_tower.conformer.{layer}");
        hidden = feed_forward(
            b,
            args,
            audio,
            &format!("{prefix}.ffw_layer_start"),
            &hidden,
        );
        trace.push(format!("conformer.{layer}.feed_forward_start"), &hidden);
        hidden = attention(
            b,
            args,
            audio,
            &format!("{prefix}.attention"),
            &hidden,
            &valid,
        );
        trace.push(format!("conformer.{layer}.attention"), &hidden);
        let masked = zero_invalid(b, &hidden, &valid);
        hidden = light_conv(b, args, audio, &format!("{prefix}.lconv1d"), &masked);
        trace.push(format!("conformer.{layer}.light_conv"), &hidden);
        hidden = feed_forward(b, args, audio, &format!("{prefix}.ffw_layer_end"), &hidden);
        trace.push(format!("conformer.{layer}.feed_forward_end"), &hidden);
        hidden = final_conformer_norm(b, args, audio, &prefix, &hidden);
        trace.push(format!("conformer.{layer}.final_norm"), &hidden);
    }
    if audio.conf_reduction_factor > 1 {
        hidden = stride_time(b, &hidden, audio.conf_reduction_factor);
        valid = stride_time(b, &valid, audio.conf_reduction_factor);
    }
    hidden = zero_invalid(b, &hidden, &valid);
    trace.push("encoded_reduced", &hidden);
    let valid_f32 = b.convert(&valid, "f32");
    let zero = b.const_f32(0.0);
    let lengths = b.reduce_add(&valid_f32, 1, &zero);
    let lengths = b.convert(&lengths, "i32");
    (hidden, valid, lengths)
}

fn project_audio(
    b: &mut Builder,
    args: &EncoderArgs,
    audio: &Gemma3nXlaAudioConfig,
    text: &Gemma3nConfig,
    encoded: &Val,
    valid: &Val,
    trace: &mut Trace,
) -> (Val, Val) {
    let soft = rms_norm(
        b,
        encoded,
        Some(args.weight("embed_audio.soft_embedding_norm.weight")),
        audio.rms_norm_eps,
    );
    let soft = round_bf16(b, &soft);
    trace.push("soft_norm", &soft);
    let soft = b.linear_last(
        &soft,
        args.weight("embed_audio.embedding_projection.weight"),
    );
    let soft = round_bf16(b, &soft);
    trace.push("soft_linear", &soft);
    let soft = rms_norm(b, &soft, None, audio.rms_norm_eps);
    let soft = round_bf16(b, &soft);
    trace.push("soft_post_norm", &soft);

    let hard_embedding = args.weight("embed_audio.embedding.weight");
    trace.push("hard_embedding", hard_embedding);
    let hard = rms_norm(
        b,
        hard_embedding,
        Some(args.weight("embed_audio.hard_embedding_norm.weight")),
        audio.rms_norm_eps,
    );
    let hard = round_bf16(b, &hard);
    trace.push("hard_norm", &hard);
    let hard = b.linear_last(
        &hard,
        args.weight("embed_audio.embedding_projection.weight"),
    );
    let hard = round_bf16(b, &hard);
    trace.push("hard_linear", &hard);
    let hard = rms_norm(b, &hard, None, audio.rms_norm_eps);
    let hard = round_bf16(b, &hard);
    trace.push("hard_post_norm", &hard);
    let padding = b.slice(
        &hard,
        &[(audio.vocab_size - 1, audio.vocab_size), (0, text.hidden)],
    );

    let batch = soft.ty.shape[0];
    let frames = soft.ty.shape[1];
    let valid = b.reshape(valid, vec![batch, frames, 1]);
    let valid = b.broadcast(&valid, &[0, 1, 2], soft.ty.shape.clone());
    let padding = b.reshape(&padding, vec![1, 1, text.hidden]);
    let padding = b.broadcast(&padding, &[0, 1, 2], soft.ty.shape.clone());
    let projected = b.select(&valid, &soft, &padding);
    let projected = if frames < GEMMA3N_AUDIO_SOFT_TOKENS {
        let padding = b.slice(
            &hard,
            &[(audio.vocab_size - 1, audio.vocab_size), (0, text.hidden)],
        );
        let padding = b.reshape(&padding, vec![1, 1, text.hidden]);
        let padding = b.broadcast(
            &padding,
            &[0, 1, 2],
            vec![batch, GEMMA3N_AUDIO_SOFT_TOKENS - frames, text.hidden],
        );
        b.concatenate(&projected, &padding, 1)
    } else {
        projected
    };
    (
        b.reshape(
            &projected,
            vec![batch * GEMMA3N_AUDIO_SOFT_TOKENS, text.hidden],
        ),
        hard,
    )
}

fn gather_rows(b: &mut Builder, table: &Val, indices: &Val) -> Val {
    let indices = b.reshape(indices, vec![indices.ty.shape[0], 1]);
    b.gather(table, &indices)
}

fn token_embeddings(
    b: &mut Builder,
    args: &MergeArgs,
    audio: &Gemma3nXlaAudioConfig,
    text: &Gemma3nConfig,
) -> Val {
    let capacity = text.context_capacity;
    let text_table = args.weight("model.language_model.embed_tokens.weight");
    let embedded = gather_rows(b, text_table, &args.tokens);
    let scale = scalar_like(b, (text.hidden as f32).sqrt(), &embedded);
    let embedded = b.multiply(&embedded, &scale);
    let mut merged = round_bf16(b, &embedded);

    let offset = b.const_i32(audio.vocab_offset);
    let offset = b.broadcast(&offset, &[], vec![capacity]);
    let hard_mask = b.compare("GE", &args.tokens, &offset, "SIGNED");
    let local = b.subtract(&args.tokens, &offset);
    let dummy = b.const_i32((audio.vocab_size - 1) as i32);
    let dummy = b.broadcast(&dummy, &[], vec![capacity]);
    let local = b.select(&hard_mask, &local, &dummy);
    let hard = gather_rows(b, &args.hard_audio, &local);
    let hard_mask = b.reshape(&hard_mask, vec![capacity, 1]);
    let hard_mask = b.broadcast(&hard_mask, &[0, 1], hard.ty.shape.clone());
    merged = b.select(&hard_mask, &hard, &merged);

    let zero = b.const_i32(0);
    let zero = b.broadcast(&zero, &[], vec![capacity]);
    let mapped = b.compare("GE", &args.audio_rows, &zero, "SIGNED");
    let audio_token = b.const_i32(audio.vocab_offset + 1);
    let audio_token = b.broadcast(&audio_token, &[], vec![capacity]);
    let token_is_audio = b.compare("EQ", &args.tokens, &audio_token, "SIGNED");
    let soft_mask = b.logical_and(&mapped, &token_is_audio);
    let rank = b.maximum(&args.audio_rows, &zero);
    let soft = gather_rows(b, &args.projected_audio, &rank);
    let soft_mask = b.reshape(&soft_mask, vec![capacity, 1]);
    let soft_mask = b.broadcast(&soft_mask, &[0, 1], soft.ty.shape.clone());
    b.select(&soft_mask, &soft, &merged)
}

fn dense_ple(b: &mut Builder, args: &MergeArgs, text: &Gemma3nConfig, embeddings: &Val) -> Val {
    let capacity = text.context_capacity;
    let zero = b.const_f32(0.0);
    let eps = b.const_f32(text.eps);
    let inv_sqrt2 = bf16_scalar(b, std::f32::consts::FRAC_1_SQRT_2);
    dense_ple_input_head(
        b,
        text,
        &args.tokens,
        embeddings,
        args.weight("model.language_model.embed_tokens_per_layer.weight"),
        args.weight("model.language_model.per_layer_model_projection.weight"),
        args.weight("model.language_model.per_layer_projection_norm.weight"),
        capacity,
        &eps,
        &zero,
        &inv_sqrt2,
    )
}

fn mask_output_rows(
    b: &mut Builder,
    tokens: &Val,
    real_len: &Val,
    embeddings: &Val,
    ple: &Val,
) -> (Val, Val) {
    let rows = tokens.ty.shape[0];
    let indices = b.iota(rows);
    let real_len = b.broadcast(real_len, &[], vec![rows]);
    let valid = b.compare("LT", &indices, &real_len, "SIGNED");
    let embeddings_valid = b.reshape(&valid, vec![rows, 1]);
    let embeddings_valid = b.broadcast(&embeddings_valid, &[0, 1], embeddings.ty.shape.clone());
    let embeddings_zero = zeros_like(b, embeddings);
    let embeddings = b.select(&embeddings_valid, embeddings, &embeddings_zero);
    let ple_valid = b.reshape(&valid, vec![rows, 1, 1]);
    let ple_valid = b.broadcast(&ple_valid, &[0, 1, 2], ple.ty.shape.clone());
    let ple_zero = zeros_like(b, ple);
    let ple = b.select(&ple_valid, ple, &ple_zero);
    (embeddings, ple)
}

fn build_encode(
    b: &mut Builder,
    args: &EncoderArgs,
    audio: &Gemma3nXlaAudioConfig,
    text: &Gemma3nConfig,
    trace: &mut Trace,
) -> EncodeOutputs {
    let (encoded, valid, projected_lengths) = audio_encoder(b, args, audio, trace);
    let (projected_audio, hard_audio) =
        project_audio(b, args, audio, text, &encoded, &valid, trace);
    trace.push("soft_projection", &projected_audio);
    trace.push("hard_projection", &hard_audio);
    #[cfg(feature = "diagnostics")]
    let diagnostics = [
        "sscp_conv_0_convolution",
        "sscp_conv_0_norm_sum_at_time",
        "sscp_conv_0_norm_cumulative_sum",
        "sscp_conv_0_norm_mean",
        "sscp_conv_0_norm_squared_at_time",
        "sscp_conv_0_norm_cumulative_squared",
        "sscp_conv_0_norm_variance",
        "sscp_conv_0_norm_stabilized_variance",
        "sscp_conv_0_norm_inverse_stddev",
        "sscp_conv_0_norm_inverse_stddev_sqrt_reciprocal",
        "sscp_conv_0_norm",
        "sscp_conv_0",
        "sscp_conv_1_convolution",
        "sscp_conv_1_convolution_bf16_result",
        "sscp_conv_1_norm",
        "sscp_conv_1",
        "input_projection",
        "conformer.0.feed_forward_start",
        "encoded_reduced",
        "soft_norm",
        "soft_linear",
        "soft_post_norm",
        "hard_embedding",
        "hard_norm",
        "hard_linear",
        "hard_post_norm",
    ]
    .into_iter()
    .map(|name| {
        trace
            .stages
            .iter()
            .find(|(stage, _)| stage == name)
            .cloned()
            .expect("selected Gemma3n audio diagnostic stage must exist")
    })
    .collect();
    EncodeOutputs {
        projected_audio,
        hard_audio,
        projected_lengths,
        #[cfg(feature = "diagnostics")]
        diagnostics,
    }
}

fn build_merge(
    b: &mut Builder,
    args: &MergeArgs,
    audio: &Gemma3nXlaAudioConfig,
    text: &Gemma3nConfig,
    trace: &mut Trace,
) -> MergeOutputs {
    let embeddings = token_embeddings(b, args, audio, text);
    trace.push("merged_embeddings", &embeddings);
    let ple = dense_ple(b, args, text, &embeddings);
    trace.push("dense_ple", &ple);
    let (embeddings, dense_ple) =
        mask_output_rows(b, &args.tokens, &args.real_len, &embeddings, &ple);
    MergeOutputs {
        embeddings,
        dense_ple,
    }
}

fn signature(decls: &[Decl]) -> String {
    decls
        .iter()
        .enumerate()
        .map(|(index, decl)| format!("%arg{index}: {} loc(\"{}\")", decl.ty.render(), decl.loc))
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_encode(decls: &[Decl], b: &Builder, outputs: &EncodeOutputs) -> String {
    let projected_ty = outputs.projected_audio.ty.render();
    let hard_ty = outputs.hard_audio.ty.render();
    let lengths_ty = outputs.projected_lengths.ty.render();
    let mut output_types = vec![projected_ty, hard_ty, lengths_ty];
    let mut output_values = vec![
        outputs.projected_audio.name.clone(),
        outputs.hard_audio.name.clone(),
        outputs.projected_lengths.name.clone(),
    ];
    #[cfg(feature = "diagnostics")]
    for (_, stage) in &outputs.diagnostics {
        output_types.push(stage.ty.render());
        output_values.push(stage.name.clone());
    }
    let output_types = output_types.join(", ");
    let output_values = output_values.join(", ");
    format!(
        "module @audio_encode {{\n  func.func public @main({signature}) -> ({output_types}) \
         {{\n{body}    return {output_values} : {output_types}\n  }}\n}}\n",
        signature = signature(decls),
        body = b.body(),
    )
}

fn render_merge(decls: &[Decl], b: &Builder, outputs: &MergeOutputs) -> String {
    let embeddings_ty = outputs.embeddings.ty.render();
    let ple_ty = outputs.dense_ple.ty.render();
    format!(
        "module @audio_merge_ple {{\n  func.func public @main({signature}) -> \
         ({embeddings_ty}, {ple_ty}) {{\n{body}    return {embeddings}, {ple} : \
         {embeddings_ty}, {ple_ty}\n  }}\n}}\n",
        signature = signature(decls),
        body = b.body(),
        embeddings = outputs.embeddings.name,
        ple = outputs.dense_ple.name,
    )
}

fn validate_artifact(
    text: &Gemma3nConfig,
    audio: &Gemma3nXlaAudioConfig,
    frame_bucket: usize,
    clips: usize,
) -> Result<(), String> {
    audio.artifact_identity(
        frame_bucket,
        clips,
        text.hidden,
        text.n_layers,
        text.hidden_per_layer_input,
    )?;
    if audio.projected_frames(frame_bucket)? > GEMMA3N_AUDIO_SOFT_TOKENS {
        return Err("Gemma3n audio bucket exceeds the fixed 188-token projection".into());
    }
    Ok(())
}

pub(crate) fn emit_gemma3n_audio_encode(
    text: &Gemma3nConfig,
    audio: &Gemma3nXlaAudioConfig,
    frame_bucket: usize,
    clips: usize,
    precision: Precision,
) -> Result<(String, Gemma3nAudioDiagnosticLayout), String> {
    validate_artifact(text, audio, frame_bucket, clips)?;
    let (decls, args) = build_encoder_schema(audio, text, frame_bucket, clips)?;
    let mut builder = Builder::new().with_precision(precision);
    let mut trace = Trace::new();
    let outputs = build_encode(&mut builder, &args, audio, text, &mut trace);
    Ok((render_encode(&decls, &builder, &outputs), trace.layout()))
}

pub(crate) fn emit_gemma3n_audio_merge_ple(
    text: &Gemma3nConfig,
    audio: &Gemma3nXlaAudioConfig,
    clips: usize,
    precision: Precision,
) -> Result<(String, Gemma3nAudioDiagnosticLayout), String> {
    // The merge artifact is context-bucketed and independent of mel length.
    // Validate against the smallest legal frame bucket so the shared
    // architecture identity is still checked without coupling this graph to a
    // particular encoder bucket.
    validate_artifact(text, audio, GEMMA3N_AUDIO_FRAME_BUCKETS[0], clips)?;
    let (decls, args) = build_merge_schema(audio, text, clips);
    let mut builder = Builder::new().with_precision(precision);
    let mut trace = Trace::new();
    let outputs = build_merge(&mut builder, &args, audio, text, &mut trace);
    Ok((render_merge(&decls, &builder, &outputs), trace.layout()))
}

#[cfg(test)]
#[path = "gemma3n_audio_emit_tests.rs"]
mod tests;
