//! Stable argument order for the split Gemma3n audio artifacts.

use std::collections::BTreeMap;

use super::builder::{Builder, Ty, Val};
use super::gemma3n::Gemma3nConfig;
#[cfg(test)]
use crate::GEMMA3N_AUDIO_CHECKPOINT_TENSOR_COUNT;
use crate::{GEMMA3N_AUDIO_SOFT_TOKENS, Gemma3nXlaAudioConfig, gemma3n_audio_checkpoint_specs};

pub(super) struct Decl {
    pub ty: Ty,
    pub loc: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WeightSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub projection: bool,
}

pub(super) struct EncoderArgs {
    weights: BTreeMap<String, Val>,
    pub mel: Val,
    pub valid_mask: Val,
}

impl EncoderArgs {
    pub fn weight(&self, name: &str) -> &Val {
        self.weights
            .get(name)
            .unwrap_or_else(|| panic!("Gemma3n audio encoder weight missing from schema: {name}"))
    }
}

pub(super) struct MergeArgs {
    weights: BTreeMap<String, Val>,
    pub projected_audio: Val,
    pub hard_audio: Val,
    pub tokens: Val,
    pub audio_rows: Val,
    pub real_len: Val,
}

impl MergeArgs {
    pub fn weight(&self, name: &str) -> &Val {
        self.weights
            .get(name)
            .unwrap_or_else(|| panic!("Gemma3n audio merge weight missing from schema: {name}"))
    }
}

fn logical_weight(name: impl Into<String>, shape: Vec<usize>) -> WeightSpec {
    let name = name.into();
    WeightSpec {
        projection: shape.len() == 2 && name.ends_with(".weight") && !name.contains("norm.weight"),
        name,
        shape,
    }
}

pub(crate) fn encoder_weight_specs(
    audio: &Gemma3nXlaAudioConfig,
    text: &Gemma3nConfig,
) -> Result<Vec<WeightSpec>, String> {
    let mut specs = gemma3n_audio_checkpoint_specs(audio, text.hidden)?
        .into_iter()
        .filter_map(|spec| {
            (!spec.name.starts_with("embed_audio.")).then(|| logical_weight(spec.name, spec.shape))
        })
        .collect::<Vec<_>>();
    specs.extend([
        logical_weight(
            "embed_audio.embedding.weight",
            vec![audio.vocab_size, audio.hidden_size],
        ),
        logical_weight(
            "embed_audio.embedding_projection.weight",
            vec![text.hidden, audio.hidden_size],
        ),
        logical_weight(
            "embed_audio.hard_embedding_norm.weight",
            vec![audio.hidden_size],
        ),
        logical_weight(
            "embed_audio.soft_embedding_norm.weight",
            vec![audio.hidden_size],
        ),
    ]);
    Ok(specs)
}

pub(crate) fn merge_weight_specs(text: &Gemma3nConfig) -> Vec<WeightSpec> {
    vec![
        logical_weight(
            "model.language_model.embed_tokens.weight",
            vec![text.vocab, text.hidden],
        ),
        logical_weight(
            "model.language_model.embed_tokens_per_layer.weight",
            vec![
                text.per_layer_vocab,
                text.n_layers * text.hidden_per_layer_input,
            ],
        ),
        logical_weight(
            "model.language_model.per_layer_model_projection.weight",
            vec![text.n_layers * text.hidden_per_layer_input, text.hidden],
        ),
        logical_weight(
            "model.language_model.per_layer_projection_norm.weight",
            vec![text.hidden_per_layer_input],
        ),
    ]
}

fn take(decls: &mut Vec<Decl>, index: &mut usize, ty: Ty, loc: impl Into<String>) -> Val {
    let value = Builder::arg(*index, ty.clone());
    decls.push(Decl {
        ty,
        loc: loc.into(),
    });
    *index += 1;
    value
}

pub(super) fn build_encoder_schema(
    audio: &Gemma3nXlaAudioConfig,
    text: &Gemma3nConfig,
    frame_bucket: usize,
    clips: usize,
) -> Result<(Vec<Decl>, EncoderArgs), String> {
    let specs = encoder_weight_specs(audio, text)?;
    let mut decls = Vec::with_capacity(specs.len() + 2);
    let mut index = 0;
    let mut weights = BTreeMap::new();
    for spec in specs {
        let value = take(
            &mut decls,
            &mut index,
            Ty::f32(spec.shape),
            spec.name.clone(),
        );
        weights.insert(spec.name, value);
    }
    let mel = take(
        &mut decls,
        &mut index,
        Ty::f32(vec![clips, frame_bucket, audio.input_feat_size]),
        "audio.mel",
    );
    let valid_mask = take(
        &mut decls,
        &mut index,
        Ty::new(vec![clips, frame_bucket], "i1"),
        "audio.valid_mask",
    );
    Ok((
        decls,
        EncoderArgs {
            weights,
            mel,
            valid_mask,
        },
    ))
}

pub(super) fn build_merge_schema(
    audio: &Gemma3nXlaAudioConfig,
    text: &Gemma3nConfig,
    clips: usize,
) -> (Vec<Decl>, MergeArgs) {
    let specs = merge_weight_specs(text);
    let mut decls = Vec::with_capacity(specs.len() + 5);
    let mut index = 0;
    let mut weights = BTreeMap::new();
    for spec in specs {
        let value = take(
            &mut decls,
            &mut index,
            Ty::f32(spec.shape),
            spec.name.clone(),
        );
        weights.insert(spec.name, value);
    }
    let projected_audio = take(
        &mut decls,
        &mut index,
        Ty::f32(vec![clips * GEMMA3N_AUDIO_SOFT_TOKENS, text.hidden]),
        "audio.projected",
    );
    let hard_audio = take(
        &mut decls,
        &mut index,
        Ty::f32(vec![audio.vocab_size, text.hidden]),
        "audio.hard_embeddings",
    );
    let tokens = take(
        &mut decls,
        &mut index,
        Ty::new(vec![text.context_capacity], "i32"),
        "tokens",
    );
    let audio_rows = take(
        &mut decls,
        &mut index,
        Ty::new(vec![text.context_capacity], "i32"),
        "audio.row_indices",
    );
    let real_len = take(&mut decls, &mut index, Ty::scalar("i32"), "real_len");
    (
        decls,
        MergeArgs {
            weights,
            projected_audio,
            hard_audio,
            tokens,
            audio_rows,
            real_len,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text() -> Gemma3nConfig {
        Gemma3nConfig::from_json_str(
            &serde_json::json!({
                "model_type": "gemma3n_text",
                "hidden_size": 8, "intermediate_size": [12, 12],
                "max_position_embeddings": 4096,
                "num_hidden_layers": 2, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 4, "rms_norm_eps": 1e-6,
                "vocab_size": 12, "vocab_size_per_layer_input": 10,
                "hidden_size_per_layer_input": 2,
                "layer_types": ["sliding_attention", "full_attention"],
                "activation_sparsity_pattern": [0.0, 0.0],
                "sliding_window": 2, "rope_theta": 1000000.0,
                "rope_local_base_freq": 10000.0, "final_logit_softcapping": 30.0,
                "num_kv_shared_layers": 0, "altup_num_inputs": 2,
                "altup_active_idx": 0, "altup_coef_clip": 120.0,
                "altup_correct_scale": true, "laurel_rank": 2
            })
            .to_string(),
        )
        .unwrap()
        .with_context_capacity(8)
        .unwrap()
    }

    #[test]
    fn split_schemas_partition_audio_and_language_weights() {
        let audio = Gemma3nXlaAudioConfig::default();
        let text = text();
        let encoder = encoder_weight_specs(&audio, &text).unwrap();
        let merge = merge_weight_specs(&text);
        assert_eq!(
            encoder.len() + merge.len(),
            GEMMA3N_AUDIO_CHECKPOINT_TENSOR_COUNT
        );
        assert!(
            encoder
                .iter()
                .all(|spec| !spec.name.starts_with("model.language_model."))
        );
        assert!(
            merge
                .iter()
                .all(|spec| spec.name.starts_with("model.language_model."))
        );
        assert!(
            encoder
                .iter()
                .any(|spec| spec.name == "embed_audio.embedding.weight"
                    && spec.shape == [128, 1_536]
                    && spec.projection)
        );
        assert_eq!(
            merge.last().unwrap().name,
            "model.language_model.per_layer_projection_norm.weight"
        );
    }
}
