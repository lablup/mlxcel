//! Gemma3n checkpoint tensors in a deterministic graph-argument order.

use super::gemma3n::Gemma3nConfig;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Gemma3nWeightSpec {
    Tensor(String),
    Projection(String),
}

impl Gemma3nWeightSpec {
    pub(crate) fn name(&self) -> &str {
        match self {
            Self::Tensor(name) | Self::Projection(name) => name,
        }
    }
}

fn tensor(out: &mut Vec<Gemma3nWeightSpec>, name: impl Into<String>) {
    out.push(Gemma3nWeightSpec::Tensor(name.into()));
}

fn projection(out: &mut Vec<Gemma3nWeightSpec>, name: impl Into<String>) {
    out.push(Gemma3nWeightSpec::Projection(name.into()));
}

/// Return the exact converted-MLX tensor names consumed by the Gemma3n graph.
///
/// Quantized checkpoints retain the same logical order.  The existing loader
/// resolves a `Projection` as either a dense tensor or the affine
/// `weight/scales/biases` triple; norms and AltUp coefficient matrices stay
/// ordinary tensors.
pub(crate) fn gemma3n_weight_specs(cfg: &Gemma3nConfig) -> Vec<Gemma3nWeightSpec> {
    // Hugging Face Gemma3n stores the text backbone below
    // `model.language_model`; unlike the sanitized MLX in-memory map, there is no
    // extra `.model` component in the safetensors files read by XLA.
    let root = "model.language_model";
    let mut out = Vec::new();
    projection(&mut out, format!("{root}.embed_tokens.weight"));
    projection(&mut out, format!("{root}.embed_tokens_per_layer.weight"));
    projection(
        &mut out,
        format!("{root}.per_layer_model_projection.weight"),
    );
    tensor(&mut out, format!("{root}.per_layer_projection_norm.weight"));
    tensor(&mut out, format!("{root}.norm.weight"));
    for plane in 0..cfg.altup_num_inputs - 1 {
        projection(&mut out, format!("{root}.altup_projections.{plane}.weight"));
    }
    for plane in 0..cfg.altup_num_inputs - 1 {
        projection(
            &mut out,
            format!("{root}.altup_unembed_projections.{plane}.weight"),
        );
    }
    for layer in 0..cfg.n_layers {
        let p = format!("{root}.layers.{layer}");
        tensor(&mut out, format!("{p}.altup.correct_output_scale"));
        tensor(&mut out, format!("{p}.altup.correction_coefs.weight"));
        projection(&mut out, format!("{p}.altup.modality_router.weight"));
        tensor(&mut out, format!("{p}.altup.router_norm.weight"));
        tensor(&mut out, format!("{p}.altup.prediction_coefs.weight"));
        projection(&mut out, format!("{p}.laurel.linear_left.weight"));
        projection(&mut out, format!("{p}.laurel.linear_right.weight"));
        tensor(&mut out, format!("{p}.laurel.post_laurel_norm.weight"));
        tensor(&mut out, format!("{p}.input_layernorm.weight"));
        tensor(&mut out, format!("{p}.post_attention_layernorm.weight"));
        tensor(&mut out, format!("{p}.pre_feedforward_layernorm.weight"));
        tensor(&mut out, format!("{p}.post_feedforward_layernorm.weight"));
        projection(&mut out, format!("{p}.self_attn.q_proj.weight"));
        if layer < cfg.kv_cache_layers() {
            projection(&mut out, format!("{p}.self_attn.k_proj.weight"));
            projection(&mut out, format!("{p}.self_attn.v_proj.weight"));
        }
        projection(&mut out, format!("{p}.self_attn.o_proj.weight"));
        tensor(&mut out, format!("{p}.self_attn.q_norm.weight"));
        if layer < cfg.kv_cache_layers() {
            tensor(&mut out, format!("{p}.self_attn.k_norm.weight"));
        }
        projection(&mut out, format!("{p}.mlp.gate_proj.weight"));
        projection(&mut out, format!("{p}.mlp.up_proj.weight"));
        projection(&mut out, format!("{p}.mlp.down_proj.weight"));
        projection(&mut out, format!("{p}.per_layer_input_gate.weight"));
        projection(&mut out, format!("{p}.per_layer_projection.weight"));
        tensor(&mut out, format!("{p}.post_per_layer_input_norm.weight"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Gemma3nConfig {
        Gemma3nConfig {
            context_capacity: 8,
            max_position_embeddings: 4096,
            hidden: 8,
            intermediate: vec![16; 4],
            n_layers: 4,
            n_q: 2,
            n_kv: 1,
            head_dim: 4,
            eps: 1e-6,
            vocab: 12,
            per_layer_vocab: 10,
            hidden_per_layer_input: 2,
            layer_types: vec![
                super::super::gemma3n::Gemma3nLayerType::Sliding,
                super::super::gemma3n::Gemma3nLayerType::Full,
                super::super::gemma3n::Gemma3nLayerType::Sliding,
                super::super::gemma3n::Gemma3nLayerType::Full,
            ],
            activation_sparsity: vec![0.5, 0.0, 0.0, 0.0],
            sliding_window: 4,
            rope_theta: 1e6,
            rope_local_base: 1e4,
            final_logit_softcap: Some(30.0),
            num_kv_shared_layers: 2,
            altup_num_inputs: 4,
            altup_active_idx: 0,
            altup_coef_clip: Some(120.0),
            altup_correct_scale: true,
            laurel_rank: 2,
            tie_word_embeddings: true,
            quantization: None,
        }
    }

    #[test]
    fn shared_layers_do_not_add_physical_kv_projection_arguments() {
        let specs = gemma3n_weight_specs(&cfg());
        assert!(
            specs
                .iter()
                .any(|s| s.name().ends_with("layers.1.self_attn.k_proj.weight"))
        );
        assert!(
            !specs
                .iter()
                .any(|s| s.name().ends_with("layers.2.self_attn.k_proj.weight"))
        );
        assert!(
            !specs
                .iter()
                .any(|s| s.name().ends_with("layers.3.self_attn.v_proj.weight"))
        );
        assert!(
            specs
                .iter()
                .any(|s| s.name().ends_with("layers.3.self_attn.q_proj.weight"))
        );
    }

    #[test]
    fn uses_real_hugging_face_gemma3n_tensor_paths() {
        let specs = gemma3n_weight_specs(&cfg());
        assert_eq!(specs[0].name(), "model.language_model.embed_tokens.weight");
        assert!(
            specs
                .iter()
                .any(|s| { s.name() == "model.language_model.layers.0.laurel.linear_left.weight" })
        );
        assert!(specs.iter().any(|s| {
            s.name() == "model.language_model.layers.0.altup.correction_coefs.weight"
        }));
    }

    #[test]
    #[ignore = "requires GEMMA3N_MODEL_DIR pointing at a real local checkpoint"]
    fn real_checkpoint_config_and_weight_index_cover_the_graph_schema() {
        let model_dir =
            std::path::PathBuf::from(std::env::var_os("GEMMA3N_MODEL_DIR").expect("model dir"));
        let cfg = Gemma3nConfig::from_json(&model_dir).unwrap();
        let index_path = model_dir.join("model.safetensors.index.json");
        let index: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
        let weights = index["weight_map"].as_object().expect("weight_map object");
        let missing: Vec<String> = gemma3n_weight_specs(&cfg)
            .into_iter()
            .map(|spec| spec.name().to_string())
            .filter(|name| !weights.contains_key(name))
            .collect();
        assert!(
            missing.is_empty(),
            "graph weights missing from index: {missing:?}"
        );
        assert_eq!(cfg.layer_to_cache().unwrap().len(), cfg.n_layers);
        assert_eq!(cfg.kv_cache_layers(), 20);
    }
}
