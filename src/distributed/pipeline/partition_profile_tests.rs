use super::*;
use serde_json::json;

#[test]
fn build_profile_for_llama_style_config() {
    let config = json!({
        "model_type": "llama",
        "num_hidden_layers": 4,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 2048,
        "tie_word_embeddings": false,
    });
    let profile = build_profile_from_json(&config, 4);
    assert_eq!(profile.num_layers, 4);
    let layer_bytes = profile.layer_bytes.unwrap();
    // Every layer has the same byte cost in dense llama.
    assert!(layer_bytes.iter().all(|&b| b == layer_bytes[0]));
    assert!(profile.embedding_param_bytes > 0);
    assert!(profile.adjacency.is_empty());
}

#[test]
fn build_profile_marks_moe_layers_heavier() {
    // Mixtral-style: every layer is MoE. Expect MoE layers to dwarf the
    // pure-dense footprint.
    let config = json!({
        "model_type": "mixtral",
        "num_hidden_layers": 4,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 2048,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
    });
    let moe_profile = build_profile_from_json(&config, 4);
    let moe_layer_bytes = moe_profile.layer_bytes.unwrap();

    let dense_config = json!({
        "model_type": "llama",
        "num_hidden_layers": 4,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 2048,
    });
    let dense_profile = build_profile_from_json(&dense_config, 4);
    let dense_layer_bytes = dense_profile.layer_bytes.unwrap();

    // 8 experts should produce a noticeably bigger layer.
    assert!(moe_layer_bytes[0] > dense_layer_bytes[0] * 4);
}

#[test]
fn build_profile_separates_dense_and_moe_in_deepseek() {
    // DeepSeek V3: first few layers are dense, the rest are MoE. Expect
    // the byte vector to show the boundary.
    let config = json!({
        "model_type": "deepseek_v3",
        "num_hidden_layers": 6,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "moe_intermediate_size": 512,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 2048,
        "first_k_dense_replace": 2,
        "moe_layer_freq": 1,
        "n_routed_experts": 8,
    });
    let profile = build_profile_from_json(&config, 6);
    // All `num_hidden_layers` entries are real decoder layers (the MTP
    // trailer, when present, lives at the out-of-range index
    // `num_hidden_layers` and is stripped by `sanitize_weights`), so the
    // profile carries all 6 (issue #525 round 3).
    assert_eq!(profile.num_layers, 6);
    let layers = profile.layer_bytes.unwrap();
    assert_eq!(layers.len(), 6);
    // First two layers are dense, third onward is MoE.
    let dense = layers[0];
    let moe = layers[2];
    assert_eq!(layers[1], dense);
    assert!(moe > dense);
}

#[test]
fn build_profile_emits_gemma4_kv_shared_adjacency() {
    let config = json!({
        "model_type": "gemma4",
        "num_hidden_layers": 6,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 2048,
        "num_kv_shared_layers": 2,
        "use_double_wide_mlp": true,
        "layer_types": [
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],
    });
    let profile = build_profile_from_json(&config, 6);
    assert!(!profile.adjacency.is_empty());
    // Shared consumers are layers 4, 5; sources are layers 2, 3 by type.
    let ranges: Vec<_> = profile.adjacency.iter().map(|g| g.layers.clone()).collect();
    // Consumer layer 4 (full_attention) pairs with last full_attention in
    // pre-shared region (layer 2).
    assert!(ranges.iter().any(|r| r.start == 2 && r.end == 5));
    // Consumer layer 5 (sliding_attention) pairs with layer 3.
    assert!(ranges.iter().any(|r| r.start == 3 && r.end == 6));
    // KV-shared consumers should also be heavier because of double-wide
    // MLP — assert the per-layer byte vector reflects that.
    let layers = profile.layer_bytes.unwrap();
    assert!(layers[4] > layers[0]);
    assert!(layers[5] > layers[1]);
}

#[test]
fn build_profile_handles_tied_word_embeddings() {
    let config_untied = json!({
        "model_type": "llama",
        "num_hidden_layers": 2,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 4096,
        "tie_word_embeddings": false,
    });
    let config_tied = json!({
        "model_type": "llama",
        "num_hidden_layers": 2,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 4096,
        "tie_word_embeddings": true,
    });
    let untied = build_profile_from_json(&config_untied, 2);
    let tied = build_profile_from_json(&config_tied, 2);
    assert_eq!(untied.embedding_param_bytes, tied.embedding_param_bytes);
    // Tied lm_head collapses to a tiny cost; untied matches the embedding.
    assert!(tied.lm_head_param_bytes < untied.lm_head_param_bytes);
}

#[test]
fn quantized_profile_reports_smaller_bytes_than_full_precision() {
    let q_config = json!({
        "model_type": "llama",
        "num_hidden_layers": 4,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 2048,
        "quantization": {"group_size": 64, "bits": 4},
    });
    let f_config = json!({
        "model_type": "llama",
        "num_hidden_layers": 4,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 2048,
    });
    let q = build_profile_from_json(&q_config, 4);
    let f = build_profile_from_json(&f_config, 4);
    let q_layers = q.layer_bytes.unwrap();
    let f_layers = f.layer_bytes.unwrap();
    assert!(
        q_layers[0] * 3 < f_layers[0],
        "quantised layer ({}) should be much smaller than full-precision ({})",
        q_layers[0],
        f_layers[0]
    );
}
